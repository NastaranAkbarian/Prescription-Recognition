Prescription-Recognition
Extract text from handwritten prescription images using Python and deep learning.

...................

 Project Phases

  Phase 1 – Model Comparison on Small Dataset
We test multiple OCR models (EasyOCR, PaddleOCR, TROCR ) on a small set of 5–10 prescription images to evaluate their effectiveness in recognizing handwritten text in Persian and English.

  Phase 2 – Full Dataset Evaluation
  we run them on the full dataset and evaluate the output against labeled ground truth.

  Phase 3 – Final Evaluation & Reporting
We calculate accuracy metrics and compare performance (precision, recall, match rate, edit distance) to decide which model works best.

## Fork improvements by @power0matin

- Added Tesseract OCR support for Persian and English (`pytesseract`)
- Added fuzzy text comparison with RapidFuzz (`partial_ratio` with threshold 85%)
- Batch evaluation script (`batch_evaluate.py`) to process multiple images and save results to `results.csv`
