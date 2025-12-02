import os
import time
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Import model
from model import VapSwin2SR

#---------------------------------------------------------------------------------------------------
# Configuration Loader
#---------------------------------------------------------------------------------------------------
def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

#---------------------------------------------------------------------------------------------------
# Losses
#---------------------------------------------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Load VGG model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        feat_x = self.vgg(x)
        feat_y = self.vgg(y)
        return F.l1_loss(feat_x, feat_y)

#---------------------------------------------------------------------------------------------------
# Dataset Class
#---------------------------------------------------------------------------------------------------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=None, use_hflip=True, use_rot=True, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        # Augmentation
        if self.use_hflip and torch.rand(1) < 0.5:
            hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)
            lr_image = lr_image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.use_rot:
            k = torch.randint(0, 4, (1,)).item()
            hr_image = hr_image.rotate(90 * k)
            lr_image = lr_image.rotate(90 * k)

        if self.crop_size:
            lr_image, hr_image = self._random_crop(lr_image, hr_image, self.crop_size)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

    def _random_crop(self, lr_image, hr_image, crop_size):
        lr_width, lr_height = lr_image.size
        hr_width, hr_height = hr_image.size
        crop_width, crop_height = crop_size

        if lr_width < crop_width or lr_height < crop_height:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        lr_left = torch.randint(0, lr_width - crop_width + 1, (1,)).item()
        lr_top = torch.randint(0, lr_height - crop_height + 1, (1,)).item()

        scale_factor = hr_width // lr_width
        hr_left = lr_left * scale_factor
        hr_top = lr_top * scale_factor
        hr_crop_width = crop_width * scale_factor
        hr_crop_height = crop_height * scale_factor

        lr_image = lr_image.crop((lr_left, lr_top, lr_left + crop_width, lr_top + crop_height))
        hr_image = hr_image.crop((hr_left, hr_top, hr_left + hr_crop_width, hr_top + hr_crop_height))

        return lr_image, hr_image

#---------------------------------------------------------------------------------------------------
# Metrics & Utils
#---------------------------------------------------------------------------------------------------
def calculate_psnr(sr, hr, max_pixel_value=1.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def compute_psnr(output, target):
    output = output.clamp(0, 1)
    target = target.clamp(0, 1)
    return calculate_psnr(output, target)

def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR")
    plt.plot(epochs, val_psnrs, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plot.png") # Save instead of show
    print("Training plot saved as training_plot.png")

#---------------------------------------------------------------------------------------------------
# Training Engine
#---------------------------------------------------------------------------------------------------
def train():
    # Load config
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data Setup
    transform = transforms.ToTensor()
    train_dataset = SRDataset(cfg['paths']['train_lr'], cfg['paths']['train_hr'], transform=transform)
    test_dataset = SRDataset(cfg['paths']['test_lr'], cfg['paths']['test_hr'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model Setup
    print("Initializing Model...")
    model = VapSwin2SR(
        upscale=cfg['model']['upscale'],
        img_size=cfg['model']['img_size'],
        window_size=cfg['model']['window_size'],
        img_range=cfg['model']['img_range'],
        depths=cfg['model']['depths'],
        embed_dim=cfg['model']['embed_dim'],
        vab_d_atten=cfg['model']['vab_d_atten'],
        num_feat=cfg['model']['num_feat'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
        resi_connection=cfg['model']['resi_connection'],
        in_chans=cfg['model']['in_chans'],
        upsampler=cfg['model']['upsampler']
    ).to(device)

    # Optimization Setup
    mse_criterion = nn.MSELoss()
    perceptual_criterion = VGGPerceptualLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg['train']['learning_rate'], 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.15, patience=3, threshold=1e-6, threshold_mode='rel'
    )

    # Setup Logging
    log_path = cfg['train']['log_file']
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "TrainLoss", "TrainPSNR", "ValLoss", "ValPSNR"])

    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    total_avg_psnr = 0

    print("Starting Training...")
    epochs = cfg['train']['epochs']
    
    for epoch in range(cfg['train']['start_epoch'], epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        epoch_psnr = 0

        # Train Loop
        for img_lr, img_hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img_lr, img_hr = img_lr.to(device), img_hr.to(device)

            outputs = model(img_lr)
            mse_loss = mse_criterion(outputs, img_hr)
            perceptual_loss = perceptual_criterion(outputs, img_hr)
            loss = mse_loss + cfg['train']['lambda_perc'] * perceptual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = compute_psnr(outputs, img_hr)
            epoch_psnr += psnr
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_psnr = epoch_psnr / len(train_loader)
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)

        # Validation Loop
        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for img_lr, img_hr in val_loader:
                img_lr, img_hr = img_lr.to(device), img_hr.to(device)
                outputs = model(img_lr)
                
                mse_loss = mse_criterion(outputs, img_hr)
                perceptual_loss = perceptual_criterion(outputs, img_hr)
                loss = mse_loss + cfg['train']['lambda_perc'] * perceptual_loss
                
                psnr = compute_psnr(outputs, img_hr)
                val_psnr += psnr
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        total_avg_psnr += avg_val_psnr

        scheduler.step(val_loss)
        elapsed_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Time: {elapsed_time:.2f}s - "
              f"Train Loss: {avg_train_loss:.4f} - Train PSNR: {avg_train_psnr:.2f} dB - "
              f"Val Loss: {avg_val_loss:.4f} - Val PSNR: {avg_val_psnr:.2f} dB - "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save Checkpoint
        torch.save(model.state_dict(), cfg['train']['checkpoint_weight_name'])
        torch.save(optimizer.state_dict(), cfg['train']['checkpoint_optim_name'])
        
        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_train_psnr, avg_val_loss, avg_val_psnr])

    # End Training
    print(f'Average PSNR on validation: {total_avg_psnr/epochs}')
    torch.save(model.state_dict(), cfg['train']['final_model_name'])
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs)

if __name__ == '__main__':
    train()
