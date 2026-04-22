# 💳 Fraud Detection System

A Machine Learning-based web application that detects fraudulent financial transactions in real-time using classification models.

---

## 🚀 Project Overview

This project aims to identify fraudulent transactions before they are completed. It uses only **pre-transaction features** to simulate a real-world fraud detection system.

---

## ✨ Features

- 🔍 Real-time fraud prediction
- 📊 Multiple ML models comparison
- ⚖️ Imbalance handling using SMOTE
- 📈 Performance metrics (Precision, Recall, F1, ROC-AUC)
- 💰 Financial impact analysis
- 🌐 Interactive web app using Streamlit

---

## 📥 Input Features

The model uses only features available **before transaction completion**:

- `step` – Time step of transaction  
- `type` – Transaction type (PAYMENT, TRANSFER, etc.)  
- `amount` – Transaction amount  
- `nameOrig` – Sender ID  
- `oldbalanceOrg` – Sender's previous balance  
- `nameDest` – Receiver ID  

---

## 🧠 Model Training

### 📊 Dataset Used
- Financial transactions dataset (fraud detection dataset)

### 🏷️ Classes
- 0 → Non-Fraud  
- 1 → Fraud  

### 🤖 Models Used
- Logistic Regression  
- Random Forest  
- XGBoost  

### ⚙️ Techniques Applied
- Data Cleaning & Preprocessing  
- Feature Engineering  
- SMOTE (Handling class imbalance)  
- Hyperparameter Tuning  

---

## 📈 Model Performance

| Metric       | Score (Example) |
|-------------|---------------|
| Accuracy     | 99%           |
| Precision    | High          |
| Recall       | High          |
| ROC-AUC      | Excellent     |

---

## 💰 Financial Impact Analysis

- Reduced false negatives (important for fraud detection)
- Improved detection rate of fraudulent transactions
- Estimated savings from prevented fraud

---

## 📸 Screenshots

### 🔥 Fraud Detected from Input
![Fraud Detection](screenshots/fraud.png)

### 🌿 Normal Transaction
![Normal Detection](screenshots/normal.png)

### 🎥 Real-Time Detection (Streamlit App)
![Web App](screenshots/app.png)

---

## 🌐 Web App (Streamlit)

Run the app locally:

```bash
streamlit run app.py
