import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO
from model import VapSwin2SR 

# --- Config ---
def load_config(config_path="demo_config.yml"):
    """Memuat konfigurasi dari file YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigurasi file tidak ditemukan: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# --- Utility ---
def list_images(folder):
    """Mendapatkan daftar file gambar dalam folder."""
    if not os.path.exists(folder):
        print(f"❌ Error: Folder gambar tidak ditemukan di path: {folder}")
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def load_image(path):
    """Memuat gambar menggunakan OpenCV."""
    return cv2.imread(path)

def show_image(img, title=""):
    """Menampilkan gambar menggunakan Matplotlib."""
    if img is None:
        print("Gambar tidak valid.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_objects(img, model):
    """Melakukan deteksi objek menggunakan model YOLO."""
    results = model(img)[0]
    return results.plot()

# --- Model Func ---

def load_sr_model(cfg, model_path, scale):
    """Memuat dan menginisialisasi model Super-Resolution menggunakan konfigurasi YAML."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr_cfg = cfg['sr_model_config']
    
    model = VapSwin2SR(
        upscale=scale,
        img_size=sr_cfg['img_size'],
        in_chans=sr_cfg['in_chans'],
        window_size=sr_cfg['window_size'],
        embed_dim=sr_cfg['embed_dim'],
        vab_d_atten=sr_cfg['vab_d_atten'],
        num_feat=sr_cfg['num_feat'],
        img_range=sr_cfg['img_range'],
        depths=sr_cfg['depths'],
        num_heads=sr_cfg['num_heads'],
        resi_connection=sr_cfg['resi_connection'],
        mlp_ratio=sr_cfg['mlp_ratio'],
        upsampler=sr_cfg['upsampler']
    ).to(device)
    
    # Memuat bobot
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def apply_super_resolution(img, model):
    """Menerapkan Super-Resolution ke gambar."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Konversi BGR OpenCV ke Tensor [1, C, H, W], normalisasi [0, 1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    # Inferensi
    with torch.no_grad():
        sr_tensor = model(img_tensor)
        
    # Konversi Tensor ke Numpy (BGR) dan denormalisasi [0, 255]
    sr_img = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)
    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
    return sr_img_bgr

# --- Interactive mode ---
def run_demo():
    print("✨ Demo Super-Resolution dan Deteksi Kendaraan ✨\n")
    
    try:
        # Memuat semua konfigurasi dari YAML
        cfg = load_config()
    except FileNotFoundError as e:
        print(e)
        return
        
    IMAGE_FOLDER = cfg['paths']['image_folder']
    SR_MODEL_PATHS = cfg['paths']['sr_model_weights']
    YOLO_MODEL_PATHS = cfg['paths']['yolo_model_weights']

    images = list_images(IMAGE_FOLDER)
    if not images:
        return

    print("Daftar gambar:")
    for idx, fname in enumerate(images):
        print(f"[{idx+1}]. {fname}")

    # --- 1. Pilih gambar ---
    try:
        selected_idx = int(input("\nPilih gambar (angka): ")) - 1
        if selected_idx < 0 or selected_idx >= len(images):
            print("❌ Pilihan tidak valid.")
            return
    except ValueError:
        print("❌ Masukan harus berupa angka.")
        return

    img_name = images[selected_idx]
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img_orig = load_image(img_path)
    
    if img_orig is None:
        print(f"❌ Gagal memuat gambar: {img_path}")
        return

    # --- 2. Tanya SR ---
    use_sr_input = input("Gunakan Super-Resolusi (SR) untuk Upscale? (y/n): ").strip().lower()
    use_sr = use_sr_input == 'y'
    
    scale = 1
    
    if use_sr:
        try:
            scale = int(input(f"Pilih skala perbesaran {list(SR_MODEL_PATHS.keys())}: "))
        except ValueError:
            print("❌ Masukan skala harus berupa angka. Menggunakan citra asli.")
            use_sr = False
        
        sr_model_path = SR_MODEL_PATHS.get(scale)
        if sr_model_path is None or not os.path.exists(sr_model_path):
            print(f"❌ Model SR untuk skala x{scale} tidak ditemukan. Menggunakan citra asli.")
            use_sr = False
        
        if use_sr:
            # --- Terapkan SR ---
            print("Memuat dan menerapkan model SR...")
            try:
                # Menggunakan fungsi load_sr_model yang sudah dimodifikasi
                sr_model = load_sr_model(cfg, sr_model_path, scale) 
                img_for_detection = apply_super_resolution(img_orig, sr_model)
                print(f"✅ Gambar berhasil di-upscale x{scale}. Resolusi baru: {img_for_detection.shape[1]}x{img_for_detection.shape[0]}")
            except Exception as e:
                print(f"❌ Error saat menjalankan SR: {e}. Menggunakan citra asli.")
                img_for_detection = img_orig
                use_sr = False
        else:
            img_for_detection = img_orig

    else:
        # Tidak menggunakan SR
        img_for_detection = img_orig

    # --- 3. Tentukan model YOLO dan deteksi ---
    scale_key = scale if use_sr and scale in YOLO_MODEL_PATHS else 'default'
    yolo_model_path = YOLO_MODEL_PATHS.get(scale_key)

    if not os.path.exists(yolo_model_path):
        print(f"\n❌ Error: Bobot model YOLO tidak ditemukan di path: {yolo_model_path}")
        return

    print(f"\nMemuat model deteksi YOLO ({os.path.basename(yolo_model_path)})...")
    yolo_model = YOLO(yolo_model_path)
    
    print("Mendeteksi kendaraan...")
    img_result = detect_objects(img_for_detection, yolo_model)

    # --- 4. Tampilkan Hasil ---
    method_title = f"VapSwin2SR x{scale}" if use_sr else "Citra Asli (No SR)"
    show_image(img_result, f"Hasil Deteksi Kendaraan pada Citra {method_title}")

if __name__ == '__main__':
    run_demo()
