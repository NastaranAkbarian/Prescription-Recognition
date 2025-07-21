import cv2
import pandas as pd

def preprocess_image(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def load_labels(excel_path):

    df = pd.read_excel(excel_path)
    labels = dict(zip(df['image_name'], df['drugs']))
    return labels

def clean_ocr_text(text):

    return text.strip().lower().replace('\n', ' ')

def compare_with_labels(ocr_output, true_label):

    return true_label.lower() in ocr_output.lower()
