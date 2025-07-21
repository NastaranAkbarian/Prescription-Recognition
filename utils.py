import os
import pandas as pd
import cv2

def load_dataset(base_path, labels_filename="training_labels.csv", images_folder="training_words"):
    """
    بارگذاری دیتاست از مسیر مشخص شده
    - خواندن فایل لیبل‌ها (CSV یا Excel)
    - ایجاد مسیر کامل برای عکس‌ها
    """
    labels_path = os.path.join(base_path, labels_filename)
    images_path = os.path.join(base_path, images_folder)

    print("📄 در حال بارگذاری فایل لیبل‌ها:", labels_path)

    if labels_filename.endswith(".csv"):
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_excel(labels_path)

    df['image_path'] = df.iloc[:, 0].apply(lambda name: os.path.join(images_path, str(name)))
    df['label'] = df.iloc[:, 1]

    return df

def read_image(image_path):
    """
    خواندن تصویر از مسیر داده‌شده با استفاده از OpenCV
    """
    if not os.path.exists(image_path):
        print(f"❌ تصویر پیدا نشد: {image_path}")
        return None

    img = cv2.imread(image_path)
    return img
