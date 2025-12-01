import os
import torch
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_func
import torchvision.utils as vutils
from model import VapSwin2SR
from train import SRDataset  # import the Dataset class from train.py

#---------------------------------------------------------------------------------------------------
# Utility Functions
#---------------------------------------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def tensor2img(tensor):
    """
    Mengonversi Tensor PyTorch (B, C, H, W) ke Numpy Image (H, W, C) dengan tipe uint8.
    Rentang input tensor dianggap 0-1.
    """
    img = tensor.squeeze(0).cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0)) # C,H,W -> H,W,C
    img = np.clip(img * 255.0, 0, 255) # Denormalize ke 0-255
    return img.astype(np.uint8)

def calculate_psnr(img1, img2):
    """
    Menghitung PSNR antara dua gambar numpy (uint8).
    """
    return cv2.PSNR(img1, img2)

def calculate_ssim(img1, img2):
    """
    Menghitung SSIM antara dua gambar numpy (uint8).
    Menggunakan channel_axis=2 karena format gambar adalah H, W, C.
    """
    return ssim_func(img1, img2, channel_axis=2, data_range=255)

#---------------------------------------------------------------------------------------------------
# Testing Engine
#---------------------------------------------------------------------------------------------------
def test():
    # 1. Load Config & Device
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on: {device}")

    # 2. Setup Results Folder (Opsional: untuk menyimpan hasil gambar)
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # 3. Load Model
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

    # 4. Load Weights
    # Prioritaskan model final, jika tidak ada cari checkpoint
    weight_path = cfg['train']['final_model_name']
    if not os.path.exists(weight_path):
        print(f"Final model {weight_path} not found, trying checkpoint...")
        weight_path = cfg['train']['checkpoint_weight_name']
    
    if os.path.exists(weight_path):
        print(f"Loading weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: No weight file found! Please train the model first.")
        return

    model.eval()

    # 5. Data Setup (Menggunakan Test/Validation Set)
    test_transform = torch.utils.data.DataLoader
    from torchvision import transforms
    transform = transforms.ToTensor()
    
    # Perhatikan: crop_size=None agar kita tes gambar full size
    test_dataset = SRDataset(
        lr_dir=cfg['paths']['test_lr'], 
        hr_dir=cfg['paths']['test_hr'], 
        crop_size=None, 
        use_hflip=False, 
        use_rot=False, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 6. Evaluation Loop
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    print("Starting Evaluation...")
    with torch.no_grad():
        for i, (img_lr, img_hr) in enumerate(tqdm(test_loader)):
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)

            # Inference
            output = model(img_lr)

            # Convert to Numpy (H, W, C) uint8 untuk perhitungan metrik standar
            # Kita ambil index 0 karena batch_size=1
            sr_img_np = tensor2img(output) 
            hr_img_np = tensor2img(img_hr)

            # Hitung Metrics
            psnr = calculate_psnr(sr_img_np, hr_img_np)
            ssim = calculate_ssim(sr_img_np, hr_img_np)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            # Save image result (opt)
            if i < 5:
                # Konversi BGR untuk OpenCV save
                save_img = cv2.cvtColor(sr_img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f"res_{i}_psnr{psnr:.2f}.png"), save_img)

    # 7. Print Summary
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print("\n" + "="*30)
    print("       TEST RESULTS       ")
    print("="*30)
    print(f"Total Images : {count}")
    print(f"Average PSNR : {avg_psnr:.4f} dB")
    print(f"Average SSIM : {avg_ssim:.4f}")
    print("="*30)
    print(f"Sample images saved in '{save_dir}/' folder.")

if __name__ == '__main__':
    test()
