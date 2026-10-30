# Loan Prediction Model

A Machine Learning project that explores loan applicant data, performs Exploratory Data Analysis (EDA), and builds classification models to predict loan approval status. The project also includes an interactive Streamlit web application for making predictions.

> **Important:** The `Loan_Status` target column in this project was synthetically generated for learning and experimentation. Predictions and evaluation metrics do not represent real-world banking performance.

## Project Overview

Loan approval decisions can depend on several factors, including applicant income, credit history, loan amount, education, employment status, and property area.

The goal of this project is to explore these factors, understand the dataset, build machine learning classification models, and create a simple web application for interacting with the selected model.

## Project Objectives

- Perform Exploratory Data Analysis (EDA) using Pandas, Matplotlib, and Seaborn.
- Understand numerical and categorical features.
- Identify missing values, outliers, and patterns in the data.
- Analyze relationships between applicant features.
- Preprocess data using imputation, encoding, and scaling.
- Train and compare multiple classification algorithms.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Save the selected trained model using Joblib.
- Build an interactive prediction application using Streamlit.

## Project Structure

```text
Loan_prediction_model/
│
├── data/
│   └── loan_prediction.csv
│
├── models/
│   └── loan_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   └── predict.py
│
├── app.py
├── README.md
└── requirements.txt
```

## Dataset Description

The dataset contains **367 records and 13 columns**, including the synthetically generated target column `Loan_Status`.

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan application identifier |
| `Gender` | Applicant's gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Applicant's education level |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Coapplicant's income |
| `LoanAmount` | Requested loan amount |
| `Loan_Amount_Term` | Loan repayment term in months |
| `Credit_History` | Credit history indicator |
| `Property_Area` | Urban, semiurban, or rural |
| `Loan_Status` | Synthetic target: `Y` or `N` |

### Target Variable

- `Y` — Loan approved, according to the synthetic label.
- `N` — Loan not approved, according to the synthetic label.

The target distribution is:

- `Y`: 283 records (77.1%)
- `N`: 84 records (22.9%)

These labels were generated for educational purposes and are not actual bank decisions.

## Exploratory Data Analysis

The EDA notebook covers the following analyses:

### 1. Univariate Analysis
- Income and loan amount distributions.
- Numerical feature boxplots.
- Applicant demographics and categorical feature counts.
- Target variable distribution.

### 2. Bivariate Analysis
- Applicant income versus loan amount.
- Education versus applicant income.
- Property area versus loan amount.
- Education and credit history against the target labels.

### 3. Multivariate Analysis
- Correlation heatmap for numerical features.
- Pairplots to examine relationships among numerical variables.
- Grouped comparisons across credit history, property area, education, and target labels.

### Key EDA Findings

1. Applicant income is right-skewed, with a small number of high-income applicants.
2. Coapplicant income includes many zero values.
3. Applicant income and loan amount have a moderate positive correlation of approximately **0.49**.
4. Several features contain missing values that require preprocessing.
5. Numerical boxplots show potential outliers that should be investigated before deciding whether to retain or remove them.
6. The dataset contains more male applicants and graduates than their respective comparison groups.
7. Credit history has a strong relationship with the synthetic target because of the target-generation rules.
8. The target classes are imbalanced, with approximately 77% `Y` and 23% `N`.

## Machine Learning Workflow

The project follows this workflow:

1. Load the dataset.
2. Separate features and target variable.
3. Remove the identifier column from model features.
4. Split the data into training and testing sets.
5. Handle missing numerical values using median imputation.
6. Handle missing categorical values using most-frequent imputation.
7. Encode categorical features using one-hot encoding.
8. Scale numerical features where appropriate.
9. Train and compare classification algorithms.
10. Evaluate model performance.
11. Save the selected model.
12. Use the model through a Streamlit web application.

### Algorithms Compared

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions |
| Precision | Measures how many predicted positive cases are actually positive |
| Recall | Measures how many actual positive cases are correctly identified |
| F1-score | Balances precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions by class |

The model comparison and evaluation results are available in `notebooks/02_modeling.ipynb`.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Preprocessing, classification, and evaluation
- **Joblib** — Model serialization
- **Jupyter Notebook** — EDA and experimentation
- **Streamlit** — Interactive web application

## Installation and Setup

### Prerequisites

Install Python and Git if they are not already installed.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Loan_prediction_model
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet populated, you can start with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit joblib
```

### 4. Prepare the dataset

Place the prepared CSV file at:

```text
data/loan_prediction.csv
```

The file must contain the `Loan_Status` column expected by the modeling notebook. In this project, that target is synthetic.

## Running the Project

### Run the EDA Notebook

Open VS Code, select the project's Python environment as the Jupyter kernel, and run:

```text
notebooks/01_EDA.ipynb
```

### Train and Save the Model

Open and run:

```text
notebooks/02_modeling.ipynb
```

The notebook trains and compares models, evaluates their results, and saves the selected model to:

```text
models/loan_model.pkl
```

Run the modeling notebook before launching the app if the saved model does not exist.

### Launch the Streamlit Application

From the project root directory, run:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. Enter the applicant details and submit the form to view the predicted class and, when supported by the selected model, class probabilities.

## Results and Limitations

The modeling notebook compares several classification algorithms using a consistent train-test split and evaluation metrics.

**Important limitations:**

- The target labels are synthetic, not actual loan approval outcomes.
- Because the synthetic labels were generated using rules related to applicant features, the model may simply learn those rules.
- The resulting scores should not be interpreted as evidence that the model can predict actual bank decisions.
- The current test-set comparison is suitable for an initial learning experiment; a stronger evaluation would use cross-validation for model selection and a separate holdout test set for final evaluation.

## Future Improvements

- Obtain a properly licensed dataset containing genuine loan outcomes.
- Use cross-validation and a separate holdout test set.
- Add hyperparameter tuning.
- Investigate class imbalance using appropriate evaluation methods.
- Add feature importance and model explainability.
- Improve input validation and error handling in the web application.
- Deploy the application to a suitable hosting platform.
- Add automated tests and a reproducible training pipeline.

##  Learning Outcomes

Through this project, I practiced:

- Data cleaning and exploratory data analysis.
- Identifying and handling missing values.
- Understanding distributions, outliers, and correlations.
- Building preprocessing pipelines with Scikit-learn.
- Training and comparing classification models.
- Evaluating model performance with multiple metrics.
- Saving and loading trained models.
- Integrating a machine learning model into a Streamlit application.

## Author

**Depesh Kumar**
*AI Engineer intern*
This project was developed as a hands-on learning exercise in Data Analysis and Machine Learning.

