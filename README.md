# Author: Abhishek Paul
# Plaform: Upgrad
# University : Liverpool John Moores University, United Kingdom.
# Course: Masters in Airtificial Intelligenc and Machine Learning
# Student No: 1176066
# Dissertation Topic: Using Artificial Intelligence to Label Free-Text Operative and Ultrasound Reports for Grading Paediatric Appendicitis

# Paediatric Appendicitis AI System

An artificial intelligence system for automatically grading paediatric appendicitis severity from free-text operative and ultrasound reports.

## Project Overview

This research develops and evaluates an AI system to automate the labelling and grading of paediatric appendicitis from clinical narratives, comparing performance against manual data extraction and ChatGPT-4.

## Features

- **Data Preprocessing**: Automated cleaning and standardization of clinical text
- **Feature Engineering**: TF-IDF, clinical NER, and BERT embeddings
- **Multiple Models**: Traditional ML (Logistic Regression, Random Forest, SVM) and Deep Learning (LSTM, BERT)
- **Comprehensive Evaluation**: Comparison against manual extraction and ChatGPT-4
- **Reproducible Pipeline**: Modular, configurable, and well-documented codebase

## NOTE
- **The Labelled data would be present inside Test and Train csv files and would be presnent at the last column caled as label_encoded

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/paediatric_appendicitis_ai.git
cd paediatric_appendicitis_ai
```

2. Create and activate a conda environment:
```
conda create -n appendicitis-ai python=3.8
conda activate appendicitis-ai
```

3. Install dependencies:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4.Install scispaCy model:
```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

## Execution

1. Extract Data (optional):
```
python3 main.py --stage extract
```

2. Run Pipeline:
```
python3 main.py
```

## Results

```
Results can be found under \results folder

The test and Train Data would be present under data/processed directory.
```

## NOTE
- **The Labelled data would be present inside Test and Train csv files and would be presnent at the last column caled as label_encoded
