import os
import random
import shutil


def split_to_save(src_dir, train_dir, val_dir, shuffle_ratio=0.8):
    files = [
        f for f in os.listdir(src_dir) if f.lower().endswith((".png", "jpg", "jpeg"))
    ]

    random.shuffle(files)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    split_index = int(len(files) * shuffle_ratio)

    for f in files[:split_index]:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(train_dir, f))

    for f in files[split_index:]:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(val_dir, f))

    print("Data split completed:")
    print(f"Training data: {len(files[:split_index])} images")
    print(f"Validation data: {len(files[split_index:])} images")


def main():
    split_to_save("Data/True", "Data/True/train", "Data/True/val")
    split_to_save("Data/False", "Data/False/train", "Data/False/val")


if __name__ == "__main__":
    random.seed(42)
    main()
