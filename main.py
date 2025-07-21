from utils import preprocess_image, load_labels, clean_ocr_text, compare_with_labels
import pytesseract
import os


def main():
    image_path = "data/small_test/rx_01.jpg"
    image = preprocess_image(image_path)

    labels = load_labels("labels.xlsx")

    ocr_output = pytesseract.image_to_string(image, lang="fas+eng")

    cleaned_text = clean_ocr_text(ocr_output)

    filename = os.path.basename(image_path)
    is_match = compare_with_labels(cleaned_text, labels.get(filename, ""))

    print(f"OCR Text: {ocr_output}")
    print(f"Match with label: {is_match}")

    import csv

    with open("results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "OCR Output", "Cleaned Text", "Match"])
        writer.writerow([filename, ocr_output.strip(), cleaned_text, is_match])


if __name__ == "__main__":
    main()
