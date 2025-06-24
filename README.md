# 💡 Customer Churn Predictor
### ✅ Predicting Customer Retention Using Logistic Regression + Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) 
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange) 
![Streamlit](https://img.shields.io/badge/WebApp-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🎯 Project Objective

**Customer Churn** is one of the most crucial KPIs for subscription-based businesses.  
This project predicts if a customer is likely to churn based on their demographics, service usage, and payment behavior using a **Logistic Regression model**, and deploys the solution via an interactive **Streamlit** app.

---

## 🧠 Highlights

- 🔍 **Logistic Regression Model** with tuned hyperparameters (`GridSearchCV`)
- 🧼 End-to-End **ML Pipeline** (custom cleaning, encoding, scaling)
- 🌐 **Streamlit Web App** with real-time inputs
- 📈 **Live ROC Curve & Churn Probability**
- 📊 **Confusion Matrix & Classification Report**
- ⚡ Animated probability display + responsive UI
---

## 🔍 Dataset Overview

| Feature Category | Columns |
|------------------|---------|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling` |
| **Services**     | `InternetService`, `StreamingTV`, `OnlineSecurity`, etc. |
| **Target**       | `Churn` (Yes/No) |

📁 Source: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📊 Model Evaluation

| Metric               | Value   |
|----------------------|---------|
| Accuracy             | 80%     |
| ROC AUC              | 0.84    |
| Precision / Recall   | ✓ Balanced |
| Cross-validation     | 5-Fold Grid Search |

---

## 🛠️ Tech Stack

| Layer         | Tools & Libraries                        |
|---------------|-------------------------------------------|
| **Language**  | Python 3.10                               |
| **Modeling**  | Scikit-learn (Logistic Regression)        |
| **Preprocessing** | Pipeline, OneHotEncoder, OrdinalEncoder |
| **Frontend**  | Streamlit                                 |
| **Visualization** | Matplotlib, Seaborn                   |
| **Deployment**| Streamlit Cloud / Render (Optional)       |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch app
streamlit run app.py
````

---

## 📂 Project Structure

```
customer-churn-predictor/
│
├── app.py                # Streamlit frontend
├── utils.py              # Cleaning functions
├── X_test.pkl / y_test.pkl # Evaluation data
├── requirements.txt
├── README.md
└── Customer_Churn/
    └── data/
        └── Telco-Customer-Churn.csv
```

---

## ✅ Key Learnings

* Built a **clean ML pipeline** using `Pipeline` & `ColumnTransformer`
* Applied **logistic regression tuning** with `GridSearchCV`
* Created a **real-time dashboard** with **ROC analysis** & animation
* Learned to **deploy & present ML models** in a user-friendly interface

---
### 🌐 Live Demo

👉 Try out the deployed application:  
**[🧠 Customer Churn Predictor (Live App)](https://customer-churn-predictor-pxh6onc8wr2jhkyjtugtfp.streamlit.app/)**  
*Built with Streamlit and powered by a logistic regression model.*

---

## 📬 Connect With Me

* GitHub: [GitNinja11](https://github.com/GitNinja)
* Email: [vaishnavinewalkar04l@gmail.com](vaishnavinewalkar04l@gmail.com)

---


