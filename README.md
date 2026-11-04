# Mobile Wallet Fraud Detection

## Overview

A Machine Learning project that detects fraudulent mobile wallet transactions using transaction data and classification algorithms.

The goal is to identify suspicious transactions accurately while minimizing false negatives and handling the highly imbalanced nature of financial fraud datasets.

## Dataset

This project uses the **PaySim Mobile Money Dataset**.

- Total transactions: 6,362,620
- Legitimate transactions: 6,354,407
- Fraudulent transactions: 8,213
- Target variable: `isFraud`

The dataset is larger than 500 MB and is not included in this repository because of GitHub's file size limitations.

**[Download Dataset](PASTE_YOUR_DATASET_LINK_HERE)**

After downloading, place `detection.csv` inside the `data/` directory.

## Technologies & Libraries

- Python
- Pandas & NumPy
- Scikit-learn
- Random Forest Classifier
- Joblib
- Streamlit

## Machine Learning Workflow

1. Data Loading and Exploration
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection and Engineering
5. Model Training
6. Model Evaluation
7. Model Deployment using Streamlit

### Selected Features

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`

The target variable is `isFraud`.

## Model Performance

The selected Random Forest model achieved the following validation results at a classification threshold of 0.5:

| Metric | Score |
|---|---:|
| Accuracy | 98.51% |
| Precision | 7.78% |
| Recall | 97.24% |
| F1-Score | 14.41% |
| Balanced Accuracy | 97.88% |
| Average Precision | 83.88% |

**Note:** Due to the significant class imbalance, recall, precision, F1-score, and Average Precision are important evaluation metrics alongside accuracy.

## Project Structure

```text
mobile_wallet_fraud_detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── README.md
│
├── models/
│   └── fraud_detection_pipeline.joblib
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/EngmoustafaAhmad/mobile_wallet_fraud_detection.git

cd mobile_wallet_fraud_detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the dataset using the link provided above and place it inside the `data/` directory.

### 5. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

## Model Deployment

The trained machine learning pipeline is saved using Joblib:

```text
models/fraud_detection_pipeline.joblib
```

The Streamlit application loads this pipeline to predict whether a transaction is legitimate or potentially fraudulent.

## Future Improvements

- Experiment with advanced classification algorithms.
- Improve fraud detection precision while maintaining high recall.
- Optimize classification thresholds.
- Implement real-time transaction monitoring.
- Enhance model explainability.

## Author

**Moustafa Ahmed**

Machine Learning & AI Developer

- GitHub: [EngmoustafaAhmad](https://github.com/EngmoustafaAhmad)
