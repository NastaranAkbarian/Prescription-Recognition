from utils import load_dataset, read_image


def main():
    base_path = r"C:\Users\ASUS\PycharmProjects\Prescription-Recognition\Doctor’s Handwritten Prescription BD dataset\training"

    train_df = load_dataset(base_path, labels_filename="training_labels.csv")

    print(f"\n🔢 تعداد نمونه‌ها: {len(train_df)}")
    print(train_df.head())

    for index, row in train_df.iterrows():
        img_path = row['image_path']
        label = row['label']

        print(f"\n📄 Label: {label}")
        print(f"🖼 Image Path: {img_path}")

        img = read_image(img_path)

        if img is None:
            print("⚠️ تصویر بارگذاری نشد.")
        else:
            print(f"✅ تصویر بارگذاری شد - ابعاد: {img.shape}")

        if index >= 4:
            break  # فقط 5 نمونه اول را نمایش بده


if __name__ == "__main__":
    main()
