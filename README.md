# Real-Time Card-Not-Present Fraud Detection for an E-Commerce Payment Gateway

## 📌 Project Overview

This project develops a machine learning system for detecting fraudulent **Card-Not-Present (CNP)** transactions in an e-commerce payment environment.

The project uses **Support Vector Machine (SVM)** models to distinguish between legitimate and fraudulent transactions. Since fraud transactions represent only a small portion of the dataset, the project focuses on metrics such as **Precision, Recall, F1-score, and PR-AUC** instead of relying only on accuracy.

The complete workflow includes:

* Exploratory Data Analysis (EDA)
* Class imbalance analysis
* Data preprocessing
* Feature scaling
* Categorical feature encoding
* Linear SVM baseline
* RBF and Polynomial kernel comparison
* Stratified 5-fold Cross-Validation
* Hyperparameter tuning using `GridSearchCV`
* Feature selection using `RFE`
* Confusion Matrix
* Precision-Recall Curve
* PR-AUC and ROC-AUC evaluation
* Training-time comparison

---

## 🎯 Objectives

The main objectives of this project are:

1. Detect fraudulent e-commerce payment transactions.
2. Analyze the imbalance between legitimate and fraudulent transactions.
3. Compare different SVM kernels.
4. Tune SVM hyperparameters such as `C` and `gamma`.
5. Reduce unnecessary features using Recursive Feature Elimination (RFE).
6. Evaluate the model using fraud-focused metrics.
7. Study the trade-off between false positives and false negatives.
8. Develop a model that can potentially be used as part of a real-time payment fraud detection pipeline.

---

## 🏦 Domain

**FinTech / E-Commerce Payments**

The system is designed around a Card-Not-Present transaction scenario where transactions are evaluated for potential fraud before or during payment processing.

---

## 📊 Dataset

The project uses:

`credit_card_fraud_dataset.csv`

Dataset characteristics:

* **Transactions:** 1,500
* **Columns:** 15
* **Target variable:** `Class`
* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction

The dataset is a **user-provided synthetic CNP transaction dataset** structurally comparable to the Kaggle Credit Card Fraud Detection dataset referenced in the assignment.

### Dataset Features

The notebook works with transaction-related information such as:

* Transaction amount
* Customer information
* Merchant category
* Card type
* Transaction timestamp
* Distance from previous transaction
* Ratio to median purchase price
* Chip usage
* PIN usage
* Online order information

Unique identifiers such as `TransactionID` and `CustomerID` are not used as predictive features.

---

## 🧠 Machine Learning Approach

### 1. Exploratory Data Analysis

The dataset is first analyzed to understand:

* Class distribution
* Fraud percentage
* Feature distributions
* Fraud vs. legitimate transaction behavior
* Correlation between numerical features
* Relationship between security-related features and fraud

The analysis shows that fraud is a minority class and that individual features do not provide a simple separation between fraudulent and legitimate transactions.

---

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Removing identifier fields that do not provide generalizable information
* Extracting transaction hour from timestamp data
* Separating numerical, binary, and categorical features
* Standardizing numerical features using `StandardScaler`
* One-hot encoding categorical features using `OneHotEncoder`

A `ColumnTransformer` and `Pipeline` are used to keep preprocessing integrated with the machine-learning model and reduce data leakage during cross-validation.

---

### 3. Baseline Linear SVM

A linear-kernel Support Vector Machine is first trained as a baseline.

The model uses:

```text
SVC(kernel="linear", class_weight="balanced")
```

`class_weight="balanced"` is used because fraudulent transactions are much less frequent than legitimate transactions.

---

### 4. Kernel Comparison

Three SVM kernels are compared:

* Linear
* RBF
* Polynomial

The models are evaluated using **stratified 5-fold cross-validation**.

The primary evaluation metrics are:

* Precision
* Recall
* F1-score

F1-score is particularly useful because the dataset contains a minority fraud class.

---

### 5. Hyperparameter Tuning

The RBF SVM is tuned using `GridSearchCV`.

The search evaluates different values of:

### `C`

Controls the trade-off between:

* A wider margin
* Classification errors

### `gamma`

Controls the influence of individual training samples and the complexity of the decision boundary.

The parameter grid used in the notebook is:

```python
{
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}
```

The best configuration is selected using **F1-score** with stratified cross-validation.

---

## 🔎 Feature Selection using RFE

The project uses **Recursive Feature Elimination (RFE)** to identify and remove less important features.

RFE is applied after preprocessing and one-hot encoding so that the model can evaluate the expanded feature space.

The project compares:

```text
Full Feature Set
        vs.
RFE-Reduced Feature Set
```

The comparison considers:

* Precision
* Recall
* F1-score
* Number of features
* Training time

This helps determine whether reducing the feature space can maintain fraud detection performance while reducing computational cost.

---

## 📈 Evaluation Metrics

Because fraud detection is a highly imbalanced classification problem, accuracy is not considered sufficient.

### Precision

Measures how many transactions predicted as fraud are actually fraudulent.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how much of the actual fraud is detected.

```text
Recall = TP / (TP + FN)
```

### F1-Score

Provides a balance between precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### PR-AUC

The Precision-Recall Area Under the Curve is used because it provides a more informative view of model performance on the minority fraud class.

### ROC-AUC

ROC-AUC is also calculated as an additional evaluation metric.

---

## 📊 Visualizations

The notebook generates several visualizations, including:

* Class distribution
* Feature distributions
* Fraud vs. legitimate feature comparisons
* Correlation matrix
* SVM kernel comparison
* Confusion matrix
* Precision-Recall curve

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Support Vector Machine (SVM)
* GridSearchCV
* Recursive Feature Elimination (RFE)
* Stratified K-Fold Cross-Validation

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Additional Library

* imbalanced-learn

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/fraud-detection-svm.git
```

Move into the project directory:

```bash
cd fraud-detection-svm
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

The project is implemented as a Jupyter Notebook.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
fraud_detection_svm.ipynb
```

Make sure the dataset is available in the notebook environment:

```text
credit_card_fraud_dataset.csv
```

If the dataset is not found, the notebook provides an upload option when running in Google Colab.

---

## 📁 Project Structure

```text
fraud-detection-svm/
│
├── fraud_detection_svm.ipynb
├── credit_card_fraud_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Class Imbalance Analysis
   ↓
Data Preprocessing
   ↓
Feature Scaling + Encoding
   ↓
Linear SVM Baseline
   ↓
Kernel Comparison
   ↓
RBF SVM
   ↓
GridSearchCV Hyperparameter Tuning
   ↓
Recursive Feature Elimination
   ↓
Model Evaluation
   ↓
Precision / Recall / F1 / PR-AUC
   ↓
Fraud Detection Model
```

---

## ⚠️ False Positives vs False Negatives

Fraud detection involves an important trade-off.

### False Positive

A legitimate transaction is incorrectly classified as fraud.

Possible consequences:

* Transaction rejection or additional verification
* Customer inconvenience
* Cart abandonment
* Additional customer-support requests

### False Negative

A fraudulent transaction is classified as legitimate.

Possible consequences:

* Financial loss
* Chargebacks
* Payment-network penalties
* Increased fraud exposure

Therefore, the model should be evaluated using fraud-specific metrics rather than accuracy alone.

---

## 🚀 Future Improvements

Possible future improvements include:

* Real-time API deployment using Flask or FastAPI
* Integration with a payment gateway simulator
* Real-time transaction scoring
* SMOTE or other imbalance-handling techniques
* Threshold optimization based on business costs
* Model monitoring and fraud-pattern drift detection
* Explainable AI using SHAP
* Docker containerization
* Cloud deployment
* Streaming transactions using Kafka
* Database integration for transaction history
* Automated model retraining

---

## 🔐 Disclaimer

This project is intended for educational and research purposes.

The dataset used in this project is a synthetic/user-provided dataset and should not be considered a production financial dataset.

The model should not be directly deployed for real financial transactions without additional validation, security testing, monitoring, regulatory compliance, and production-grade infrastructure.

---

## 👨‍💻 Author

**Nikhil Kumar**

B.Tech – Artificial Intelligence & Machine Learning

---

## ⭐ Project Highlights

* SVM-based fraud detection
* Linear vs RBF vs Polynomial kernel comparison
* Stratified 5-fold cross-validation
* Hyperparameter tuning with GridSearchCV
* RFE-based feature selection
* Imbalanced classification handling
* Precision, Recall and F1 evaluation
* Precision-Recall curve
* PR-AUC analysis
* Training-time comparison
* Production-oriented fraud detection discussion
