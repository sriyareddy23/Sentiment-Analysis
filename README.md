# SentimentPulse

## Project Overview

This project aims to perform sentiment analysis on tweets using various techniques, including Jaccard similarity, text preprocessing, and advanced models like spaCy and BERT. The goal is to extract meaningful insights from tweet data and classify them into sentiment categories such as positive, negative, and neutral.

## Features

- **Data Exploration:** Analyzed and visualized tweet data to understand the distribution of sentiments and word frequencies.
- **Text Preprocessing:** Cleaned and prepared text data by removing stopwords, special characters, and irrelevant information.
- **Jaccard Similarity:** Calculated Jaccard similarity scores between tweets and selected texts to measure overlap.
- **Word Cloud Visualization:** Generated visualizations to identify the most common words associated with different sentiments.
- **Sentiment Classification:** Implemented sentiment classification using spaCy and BERT models.

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `PIL`
- `wordcloud`
- `nltk`
- `spacy`
- `torch`
- `transformers`
- `tokenizers`
- `tqdm`

## Installation

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly pillow wordcloud nltk spacy torch transformers tokenizers tqdm
```

## Data

- **Training Data:** `train.csv` - Contains tweet text, selected text, and sentiment labels.
- **Test Data:** `test.csv` - Contains tweet text for sentiment prediction.
- **Submission File:** `sample_submission.csv` - Format for submission.

## Usage

1. **Data Loading and Preprocessing:**

   Load and preprocess the data by running the initial setup code provided in the script. This includes loading data, handling missing values, and cleaning the text.

2. **Exploratory Data Analysis (EDA):**

   Perform EDA to visualize word distributions and sentiment analysis. This includes generating plots and word clouds.

3. **Model Training:**

   - **spaCy Model:** Train a custom Named Entity Recognizer (NER) model using spaCy.
   - **BERT Model:** Train a sentiment classification model using BERT.

4. **Evaluation:**

   Evaluate the performance of the models and analyze the results. Generate visualizations to assess model effectiveness.

5. **Submission:**

   Prepare the final predictions in the required format and generate the submission file.

## Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

# Clean and preprocess data
# ... (cleaning code)

# Train spaCy model
# ... (spaCy training code)

# Train BERT model
# ... (BERT training code)
```

-> Extracted sentiment phrases and Trained the model on over **27,000** tweets and tested it on more than **3,000 tweets**, achieving around **94% accuracy**.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
