# 📊 Customer Churn Predictor

This project is a **Logistic Regression-based machine learning pipeline** designed to predict customer churn using the **Telco Customer Churn Dataset**. The project includes data cleaning, preprocessing, model training, hyperparameter tuning, evaluation, and model serialization for deployment.

---

## 📁 Dataset

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Columns**: Demographic, account, and service-related features
- **Target**: `Churn` (Yes/No)

---

## 📌 Features

- 🧹 Custom transformer to clean the `TotalCharges` column
- 🧼 Column-wise preprocessing using `ColumnTransformer`:
  - One-hot encoding for nominal categorical variables
  - Ordinal encoding for contract types
  - Standard scaling for numerical features
- 🔍 Hyperparameter tuning with `GridSearchCV`
- ✅ Final model is a `LogisticRegression` classifier with best-found parameters
- 🧪 Train/test evaluation with accuracy, classification report, and confusion matrix
- 💾 Final model is saved using `pickle`

---

## 🔧 Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/customer-churn-predictor.git
   cd customer-churn-predictor


2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 How It Works

### Step-by-step Overview:

1. **Data Loading**:

   * Reads the Telco dataset using `pandas`.

2. **Data Cleaning**:

   * Handles whitespace and non-numeric values in `TotalCharges`.

3. **Feature Engineering**:

   * Separates features into numerical, categorical, and ordinal columns.
   * Applies encoding and scaling using a `Pipeline`.

4. **Model Training**:

   * Splits the data (75% training, 25% testing).
   * Applies `GridSearchCV` to find the best `LogisticRegression` hyperparameters.

5. **Model Evaluation**:

   * Evaluates accuracy, precision, recall, and F1-score.
   * Generates a confusion matrix.

6. **Model Saving**:

   * Saves the final model pipeline (`cleaner + preprocessor + classifier`) as `model.pkl`.

---

## 📈 Model Performance

Example output:

```
Accuracy: 0.81
Precision, Recall, F1: See classification_report
Confusion Matrix: See output
```

## 🛠 Dependencies

* `pandas`
* `numpy`
* `scikit-learn`
* `seaborn`

Install via:

```bash
pip install -r requirements.txt
```

---

---

> You can deploy this app for free on [Customer Churn Predictor](https://co2-emission-predictor-dveuw46vonjd6z2x3rxfza.streamlit.app/)



## 📩 Contact

For questions or collaborations:

* GitHub: [GitNinja11](https://github.com/GitNinja)
* Email: [vaishnavinewalkar04l@gmail.com](vaishnavinewalkar04l@gmail.com)
