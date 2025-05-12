# News Article Bias Detector

## Overview
This project uses machine learning techniques (such as Naïve Bayes, SVM, Random Forest, and BERT) to detect bias in news articles. The model is based on the BERT architecture, but you can easily replace it with your own trained model.

## Key Files
- **`app.py`**: The main application file that runs the bias detection.
- **`models/bert_model/`**: Directory containing the pre-trained BERT model and its configuration.
- **`models/bias_detection.py`**: The script used to detect bias in news articles using the trained model.

## Requirements
Before you begin, ensure you have the following installed:
- Python 3.x
- Required libraries (listed in `requirements.txt`)

Run the following command to install the dependencies:
```bash
pip install -r requirements.txt
