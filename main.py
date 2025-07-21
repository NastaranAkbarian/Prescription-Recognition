from utils import preprocess_image, load_labels, clean_ocr_text, compare_with_labels

image = preprocess_image('data/small_test/rx_01.jpg')

labels = load_labels('labels.xlsx')


ocr_output = "استامینوفن 500mg"

cleaned_text = clean_ocr_text(ocr_output)

is_match = compare_with_labels(cleaned_text, labels['rx_01.jpg'])

print(f"Match with label: {is_match}")
