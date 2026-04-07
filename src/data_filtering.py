# ============================================================
# data_filtering.py  (đã sửa lỗi)
# ============================================================


import os
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# 1. Đường dẫn
# ----------------------------------------------------------
CSV_PATH    = '/kaggle/input/datasets/organizations/nih-chest-xrays/data/Data_Entry_2017.csv'
IMAGES_ROOT = "/kaggle/input/datasets/organizations/nih-chest-xrays/data"


# ----------------------------------------------------------
# 2. Đọc CSV
# ----------------------------------------------------------
def load_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[load_csv] Tổng số dòng: {len(df)}")
    print(f"[load_csv] Các cột: {df.columns.tolist()}")
    print(df["Finding Labels"].value_counts().head(20))
    return df


# ----------------------------------------------------------
# 3. Lọc 2 class
# ----------------------------------------------------------
def filter_two_classes(df: pd.DataFrame) -> pd.DataFrame:
    mask_normal    = df["Finding Labels"] == "No Finding"
    mask_pneumonia = df["Finding Labels"] == "Pneumonia"
    df_filtered = df[mask_normal | mask_pneumonia].copy()
    df_filtered["label"] = df_filtered["Finding Labels"].map(
        {"No Finding": "NORMAL", "Pneumonia": "PNEUMONIA"}
    )
    df_filtered = df_filtered[["Image Index", "label"]].reset_index(drop=True)
    print(f"\n[filter_two_classes] Sau lọc: {len(df_filtered)} ảnh")
    print(df_filtered["label"].value_counts())
    return df_filtered


# ----------------------------------------------------------
# 4. [SỬA] Gắn đường dẫn ảnh — duyệt đệ quy toàn bộ IMAGES_ROOT
# ----------------------------------------------------------
def attach_image_paths(df: pd.DataFrame, images_root: str = IMAGES_ROOT) -> pd.DataFrame:
    """
    SỬA LỖI: Dùng os.walk() thay vì os.listdir() 1 cấp.
    os.walk() duyệt đệ quy tất cả subfolder nên tìm được ảnh
    dù cấu trúc là images_001/images/*.png hay bất kỳ cấp nào.
    Đồng thời in ra 5 path mẫu để dễ debug.
    """
    print(f"[attach_image_paths] Đang quét đệ quy: {images_root}")
    path_map = {}
    for root, dirs, files in os.walk(images_root):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                # Nếu trùng tên file, ưu tiên path tìm thấy đầu tiên
                if fname not in path_map:
                    path_map[fname] = os.path.join(root, fname)

    print(f"[attach_image_paths] Tổng file ảnh tìm thấy: {len(path_map)}")

    # Debug: in 5 path mẫu để kiểm tra cấu trúc thực tế
    sample_keys = list(path_map.keys())[:5]
    print("[attach_image_paths] Mẫu path tìm được:")
    for k in sample_keys:
        print(f"  {path_map[k]}")

    df = df.copy()
    df["image_path"] = df["Image Index"].map(path_map)

    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[attach_image_paths] Cảnh báo: {missing} ảnh không tìm thấy → bỏ qua")
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    print(f"[attach_image_paths] Còn lại {len(df)} ảnh có đường dẫn hợp lệ")
    return df


# ----------------------------------------------------------
# 5. [SỬA] Thống kê — guard khi DataFrame rỗng
# ----------------------------------------------------------
def print_statistics(df: pd.DataFrame):
    print("\n===== THỐNG KÊ DATASET =====")

    # SỬA LỖI: Kiểm tra rỗng trước khi vẽ
    if df.empty:
        print("[print_statistics] DataFrame rỗng, không có gì để thống kê.")
        print("  → Kiểm tra lại IMAGES_ROOT và cấu trúc thư mục ảnh.")
        return

    counts = df["label"].value_counts()
    print(counts.to_string())
    total = len(df)
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} ảnh ({cnt/total*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Phân phối class sau lọc")
    ax.set_xlabel("Class")
    ax.set_ylabel("Số ảnh")
    ax.set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 30, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("/kaggle/working/class_distribution.png", dpi=120)
    plt.show()
    print("[print_statistics] Đã lưu biểu đồ → /kaggle/working/class_distribution.png")


def save_filtered_csv(df: pd.DataFrame, out_path: str = "/kaggle/working/filtered_dataset.csv"):
    if df.empty:
        print("[save_filtered_csv] DataFrame rỗng, không lưu.")
        return
    df.to_csv(out_path, index=False)
    print(f"[save_filtered_csv] Đã lưu CSV sạch → {out_path}")


# ----------------------------------------------------------
# 6. Hàm tổng hợp
# ----------------------------------------------------------
def run_filtering() -> pd.DataFrame:
    df_raw      = load_csv()
    df_filtered = filter_two_classes(df_raw)
    df_filtered = attach_image_paths(df_filtered)
    print_statistics(df_filtered)
    save_filtered_csv(df_filtered)
    return df_filtered


if __name__ == "__main__":
    df_clean = run_filtering()
