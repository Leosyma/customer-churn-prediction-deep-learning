## 📉 Customer Churn Prediction

This project builds a **Neural Network model** to predict **customer churn** in a telecom company based on customer demographics, account information, and service usage.
The dataset used comes from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

### 🧩 Project Overview

The goal is to classify whether a customer will **churn** (leave the service) or **stay** based on multiple factors such as contract type, tenure, payment method, and service subscriptions.
A **feed-forward neural network** (built with TensorFlow/Keras) is trained on preprocessed data to learn the relationship between customer attributes and churn behavior.

### ⚙️ Workflow

1. **Exploratory Data Analysis:** handled missing values and converted `TotalCharges` to numeric.
2. **Data Cleaning & Encoding:** replaced categorical responses, created dummy variables with `pd.get_dummies`.
3. **Feature Scaling:** normalized numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) using `MinMaxScaler`.
4. **Model Building:** deep neural network with multiple hidden layers (`ReLU` activation) and sigmoid output for binary classification.
5. **Evaluation:** confusion matrix, accuracy score, and full classification report for precision, recall, and F1-score.

### 🧠 Technologies Used

* **Python:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Deep Learning:** TensorFlow / Keras
* **Visualization:** Matplotlib & Seaborn (for EDA and confusion matrix)

### 📈 Results

The neural network achieved strong classification performance, successfully distinguishing between churned and retained customers.
The model outputs performance metrics and a visual confusion matrix for interpretability.
