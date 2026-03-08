# 🛡 FraudShield — Insurance Fraud Detection System

A complete end-to-end machine learning project for detecting insurance fraud.

## 📁 Project Structure
```
project/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── insuranceFraud_Dataset.csv      # Dataset (place here)
├── models/                         # Saved ML models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   ├── model_results.pkl
│   └── best_model_name.pkl
├── notebooks/
│   └── Insurance_Fraud_Detection_Complete.ipynb
└── templates/
    └── index.html
```

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place dataset
Put `insuranceFraud_Dataset.csv` in the project root.

### Step 3: Run the notebook (optional — models are pre-trained)
Open `notebooks/Insurance_Fraud_Detection_Complete.ipynb` in Jupyter and run all cells.

### Step 4: Start the web app
```bash
python app.py
```

### Step 5: Open in browser
Visit: http://localhost:5000

## 📊 Model Performance
| Model               | Accuracy |
|---------------------|----------|
| Decision Tree ⭐    | 81.00%   |
| Random Forest       | 79.00%   |
| Logistic Regression | 79.00%   |
| KNN                 | 74.50%   |
| SVM                 | 74.50%   |
| Naive Bayes         | 70.50%   |

## 🔑 Key Features
- **4-page web app**: Home, Predict, Analysis, About
- **Real-time prediction** with confidence scores
- **6 ML models** trained and compared
- **Complete EDA** with visualizations
- **Pre-trained models** saved with joblib

## 🛠 Tech Stack
- Python, Flask, scikit-learn, pandas, numpy
- HTML5, CSS3, Vanilla JavaScript
- joblib for model serialization
