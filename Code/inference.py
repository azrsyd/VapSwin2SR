import os
import torch
import yaml
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Import model dari file model.py
from model import VapSwin2SR

#---------------------------------------------------------------------------------------------------
# Utility Functions
#---------------------------------------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    """Memuat konfigurasi dari file YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(cfg, device):
    """Menginisialisasi dan memuat bobot ke model VapSwin2SR."""
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

    # Memuat Bobot (prioritas: final model, lalu checkpoint)
    weight_path = cfg['train']['final_model_name']
    if not os.path.exists(weight_path):
        print(f"Final model {weight_path} not found, trying checkpoint...")
        weight_path = cfg['train']['checkpoint_weight_name']
    
    if os.path.exists(weight_path):
        print(f"Loading weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Error: Tidak ada file bobot yang ditemukan di {weight_path}! Harap latih model terlebih dahulu.")

    model.eval()
    return model

#---------------------------------------------------------------------------------------------------
# Inference Engine
#---------------------------------------------------------------------------------------------------
def infer():
    # 1. Load Config & Device
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. Load Model
    try:
        model = load_model(cfg, device)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Get Input Image Path
    print("\n" + "="*50)
    image_path = input("Masukkan path gambar Low Resolution (LR) yang akan di-upscale: ")
    print("="*50)
    
    if not os.path.exists(image_path):
        print(f"Error: File tidak ditemukan di path: {image_path}")
        return

    # 4. Preprocessing Input Image (LR)
    try:
        lr_image_pil = Image.open(image_path).convert('RGB')
        
        # Simpan resolusi asli (W, H)
        W_orig, H_orig = lr_image_pil.size
        
        # Transformasi: PIL Image -> Tensor (tambah batch dimension [1, C, H, W])
        preprocess = transforms.ToTensor()
        lr_tensor = preprocess(lr_image_pil).unsqueeze(0).to(device) 

    except Exception as e:
        print(f"Error saat memuat atau memproses gambar: {e}")
        return

    # 5. Perform Inference
    print(f"Performing Super-Resolution (Scale: x{cfg['model']['upscale']})...")
    with torch.no_grad():
        # Pengukuran waktu inferensi
        start_time = time.time()
        sr_tensor = model(lr_tensor)
        end_time = time.time()
        inference_time = end_time - start_time
        
    print(f"Inference Time: {inference_time:.4f} seconds")

    # 6. Post-processing Output Image (SR)
    # Clamp to [0, 1] range, remove batch dimension, Tensor -> PIL Image
    sr_image_tensor = sr_tensor.squeeze(0).clamp(0, 1) 
    sr_image_pil = transforms.ToPILImage()(sr_image_tensor.cpu())
    
    # 7. Display Results
    W_sr, H_sr = sr_image_pil.size
    print(f"Original LR Size: {W_orig}x{H_orig}")
    print(f"Upscaled SR Size: {W_sr}x{H_sr}")

    # Upscale LR menggunakan Bicubic untuk membandingkan resolusi yang sama
    lr_bicubic = lr_image_pil.resize((W_sr, H_sr), Image.Resampling.BICUBIC)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(lr_bicubic)
    axes[0].set_title(f"Input (Bicubic upscaled) - {W_orig}x{H_orig}")
    axes[0].axis('off')

    axes[1].imshow(sr_image_pil)
    axes[1].set_title(f"Output (VapSwin2SR) - {W_sr}x{H_sr}")
    axes[1].axis('off')

    plt.suptitle(f"Visual Super-Resolution Comparison (Scale: x{cfg['model']['upscale']})", fontsize=16)
    plt.show()

if __name__ == '__main__':
    infer()
