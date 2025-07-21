import os
import csv
import pytesseract
from utils import preprocess_image, load_labels, clean_ocr_text, compare_with_labels


def batch_evaluate(
    image_folder="data/small_test", labels_file="labels.xlsx", output_file="results.csv"
):
    labels = load_labels(labels_file)
    results = []

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_folder, filename)
        image = preprocess_image(image_path)
        ocr_text = pytesseract.image_to_string(image, lang="fas+eng")
        cleaned_text = clean_ocr_text(ocr_text)
        is_match = compare_with_labels(cleaned_text, labels.get(filename, ""))

        print(f"Processed {filename}: Match={is_match}")

        results.append([filename, ocr_text.strip(), cleaned_text, is_match])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "OCR Output", "Cleaned Text", "Match"])
        writer.writerows(results)


if __name__ == "__main__":
    batch_evaluate()
