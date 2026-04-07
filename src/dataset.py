# ============================================================
# dataset.py
# Mục đích: Định nghĩa ChestXrayDataset + chia train/val/test
#           + tạo DataLoader sẵn sàng cho Task 2.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

# Import từ 2 file đã tạo ở trên
from data_preprocessing import train_transform, val_test_transform


# ----------------------------------------------------------
# 1. Mapping nhãn chữ → số
# ----------------------------------------------------------
LABEL_MAP = {
    "NORMAL":    0,
    "PNEUMONIA": 1,
}


# ----------------------------------------------------------
# 2. Class Dataset
# ----------------------------------------------------------
class ChestXrayDataset(Dataset):
    """
    Dataset ảnh X-quang ngực cho bài toán phân loại nhị phân.

    Tham số:
        df        : DataFrame với cột 'image_path' và 'label' (chữ)
        transform : torchvision transform áp dụng lên ảnh
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row["image_path"]
        label = LABEL_MAP[row["label"]]   # NORMAL→0, PNEUMONIA→1

        # Mở ảnh và chuyển sang RGB (X-quang gốc thường là grayscale,
        # nhưng ResNet cần 3 channel)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------------------------
# 3. Chia train / val / test
# ----------------------------------------------------------
def split_dataset(df: pd.DataFrame,
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.15,
                  test_ratio:  float = 0.15,
                  random_state: int  = 42) -> tuple:
    """
    Chia DataFrame thành 3 tập với tỉ lệ mặc định 70/15/15.
    stratify=label đảm bảo mỗi tập có cùng tỉ lệ NORMAL/PNEUMONIA.

    Trả về: (df_train, df_val, df_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Tổng 3 tỉ lệ phải bằng 1.0"

    # Tách test trước
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=random_state,
    )

    # Tách val từ phần còn lại
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_adjusted,
        stratify=df_train_val["label"],
        random_state=random_state,
    )

    print("\n===== PHÂN CHIA DATASET =====")
    for name, subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = subset["label"].value_counts()
        print(f"  {name:5s}: {len(subset):5d} ảnh  |  "
              f"NORMAL={counts.get('NORMAL',0)}  PNEUMONIA={counts.get('PNEUMONIA',0)}")

    return df_train, df_val, df_test


# ----------------------------------------------------------
# 4. Tạo DataLoader
# ----------------------------------------------------------
def create_dataloaders(df_train, df_val, df_test,
                       batch_size=32, num_workers=2):
    
    # Tự động detect có GPU không
    use_pin_memory = torch.cuda.is_available()

    train_dataset = ChestXrayDataset(df_train, transform=train_transform)
    val_dataset   = ChestXrayDataset(df_val,   transform=val_test_transform)
    test_dataset  = ChestXrayDataset(df_test,  transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=use_pin_memory,   # ← thay True bằng này
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    print(f"\n===== DATALOADER =====")
    print(f"  batch_size  : {batch_size}")
    print(f"  train_loader: {len(train_loader)} batch ({len(train_dataset)} ảnh)")
    print(f"  val_loader  : {len(val_loader)} batch ({len(val_dataset)} ảnh)")
    print(f"  test_loader : {len(test_loader)} batch ({len(test_dataset)} ảnh)")

    return train_loader, val_loader, test_loader


# ----------------------------------------------------------
# 5. Minh họa một batch mẫu
# ----------------------------------------------------------
def visualize_batch(loader: DataLoader, n: int = 8):
    """Lấy 1 batch từ loader và hiển thị n ảnh đầu."""
    from data_preprocessing import denormalize

    images, labels = next(iter(loader))
    label_names = {v: k for k, v in LABEL_MAP.items()}

    n   = min(n, len(images))
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 5))
    axes = axes.flatten()

    for i in range(n):
        img_np = denormalize(images[i])
        axes[i].imshow(img_np)
        cls = label_names[labels[i].item()]
        color = "red" if cls == "PNEUMONIA" else "green"
        axes[i].set_title(cls, color=color, fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Mẫu ảnh từ train_loader (đã qua preprocessing)", fontsize=12)
    plt.tight_layout()
    plt.savefig("/kaggle/working/batch_sample.png", dpi=120)
    plt.show()
    print("[visualize_batch] Đã lưu → /kaggle/working/batch_sample.png")


# ----------------------------------------------------------
# 6. Hàm tổng hợp — gọi một lần từ notebook
# ----------------------------------------------------------
def build_dataloaders(df_clean: pd.DataFrame,
                      batch_size: int = 32) -> tuple:
    """
    Nhận df_clean (output của data_filtering.py),
    trả về (train_loader, val_loader, test_loader, df_train, df_val, df_test).
    """
    df_train, df_val, df_test = split_dataset(df_clean)
    train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test, batch_size=batch_size
    )
    return train_loader, val_loader, test_loader, df_train, df_val, df_test


# Chạy trực tiếp để test
if __name__ == "__main__":
    # Giả sử đã có file CSV sạch từ data_filtering.py
    df = pd.read_csv("/kaggle/working/filtered_dataset.csv")
    train_loader, val_loader, test_loader, *_ = build_dataloaders(df)
    visualize_batch(train_loader)
