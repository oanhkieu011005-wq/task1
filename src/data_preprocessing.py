# ============================================================
# data_preprocessing.py
# Mục đích: Định nghĩa transforms cho train / val / test.
#           Có hàm minh họa ảnh trước và sau preprocessing.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


# ----------------------------------------------------------
# 1. Hằng số chuẩn ImageNet (ResNet18 pretrained dùng các giá trị này)
# ----------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


# ----------------------------------------------------------
# 2. Transform cho tập TRAIN (có augmentation)
# ----------------------------------------------------------
train_transform = transforms.Compose([
    # Resize về 256 trước để có biên khi crop
    transforms.Resize(256),
    # Random crop ra đúng 224×224 — tăng tính đa dạng vị trí
    transforms.RandomCrop(IMAGE_SIZE),
    # Lật ngang ngẫu nhiên (50%) — X-quang đối xứng nên ok
    transforms.RandomHorizontalFlip(p=0.5),
    # Xoay nhẹ (±10°) — mô phỏng ảnh chụp hơi nghiêng
    transforms.RandomRotation(degrees=10),
    # Thay đổi độ sáng/tương phản nhẹ — mô phỏng máy X-quang khác nhau
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Chuyển sang Tensor (HWC → CHW, giá trị 0–1)
    transforms.ToTensor(),
    # Normalize theo chuẩn ImageNet
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ----------------------------------------------------------
# 3. Transform cho tập VAL và TEST (KHÔNG augment — đánh giá khách quan)
# ----------------------------------------------------------
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    # Center crop: lấy chính giữa — không random
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ----------------------------------------------------------
# 4. Hàm tiện ích: đảo Normalize để vẽ ảnh ra màn hình
# ----------------------------------------------------------
def denormalize(tensor):
    """
    Đảo ngược Normalize để có thể hiển thị ảnh.
    tensor: shape (C, H, W), giá trị đã normalize
    Trả về numpy array (H, W, C) giá trị 0–1
    """
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.permute(1, 2, 0).numpy()   # (C,H,W) → (H,W,C)
    img  = img * std + mean                   # đảo normalize
    img  = np.clip(img, 0, 1)
    return img


# ----------------------------------------------------------
# 5. Minh họa ảnh trước/sau preprocessing
# ----------------------------------------------------------
def visualize_preprocessing(image_paths: list, n: int = 4):
    """
    Vẽ n ảnh: cột trái = ảnh gốc, cột phải = sau train_transform.
    image_paths: list đường dẫn ảnh
    """
    n = min(n, len(image_paths))
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = [axes]

    fig.suptitle("Ảnh gốc  vs  Sau preprocessing (train_transform)", fontsize=13)

    for i, path in enumerate(image_paths[:n]):
        # --- ảnh gốc ---
        img_pil = Image.open(path).convert("RGB")
        axes[i][0].imshow(img_pil, cmap="gray" if img_pil.mode == "L" else None)
        axes[i][0].set_title(f"Gốc  {img_pil.size[0]}×{img_pil.size[1]}", fontsize=9)
        axes[i][0].axis("off")

        # --- sau transform ---
        tensor = train_transform(img_pil)
        img_np = denormalize(tensor)
        axes[i][1].imshow(img_np)
        axes[i][1].set_title(f"Sau xử lý  {IMAGE_SIZE}×{IMAGE_SIZE}", fontsize=9)
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("/kaggle/working/preprocessing_demo.png", dpi=120)
    plt.show()
    print("[visualize_preprocessing] Đã lưu → /kaggle/working/preprocessing_demo.png")


# Chạy trực tiếp để kiểm tra (cần truyền vào vài đường dẫn ảnh)
if __name__ == "__main__":
    print("train_transform:")
    print(train_transform)
    print("\nval_test_transform:")
    print(val_test_transform)
