from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models and preprocessing artifacts
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
model_results = joblib.load(os.path.join(MODEL_DIR, 'model_results.pkl'))
best_model_name = joblib.load(os.path.join(MODEL_DIR, 'best_model_name.pkl'))

# Dropdown choices from original data
CHOICES = {
    'policy_state': ['OH', 'IN', 'IL'],
    'policy_csl': ['100/300', '250/500', '500/1000'],
    'insured_sex': ['MALE', 'FEMALE'],
    'insured_education_level': ['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College', 'JD'],
    'insured_occupation': ['craft-repair', 'machine-op-inspct', 'sales', 'armed-forces', 'tech-support',
                           'exec-managerial', 'prof-specialty', 'other-service', 'handlers-cleaners',
                           'transport-moving', 'adm-clerical', 'farming-fishing', 'protective-serv'],
    'insured_hobbies': ['sleeping', 'reading', 'board-games', 'bungie-jumping', 'base-jumping',
                        'polo', 'golf', 'camping', 'paintball', 'skydiving', 'movies', 'hiking',
                        'yachting', 'chess', 'basketball', 'cross-fit', 'exercise', 'kayaking'],
    'insured_relationship': ['husband', 'own-child', 'not-in-family', 'unmarried', 'wife', 'other-relative'],
    'incident_type': ['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car'],
    'collision_type': ['Side Collision', 'Rear Collision', 'Front Collision', 'UNKNOWN'],
    'incident_severity': ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'],
    'authorities_contacted': ['Police', 'Fire', 'Ambulance', 'Other', 'NONE'],
    'incident_state': ['SC', 'VA', 'NY', 'WV', 'NC', 'PA', 'OH'],
    'incident_city': ['Columbus', 'Riverwood', 'Springfield', 'Arlington', 'Hillsdale', 'Northbend', 'Northbrook'],
    'property_damage': ['YES', 'NO', 'UNKNOWN'],
    'police_report_available': ['YES', 'NO', 'UNKNOWN'],
    'auto_make': ['Saab', 'Mercedes', 'Dodge', 'Chevrolet', 'Accura', 'Ford', 'Jeep', 'BMW',
                  'Audi', 'Toyota', 'Honda', 'Nissan', 'Volkswagen', 'Suburu', 'Porsche'],
    'auto_model': ['92x', 'E400', 'RAM', 'Tahoe', 'RSX', 'Neon', 'Grand Cherokee',
                   'Civic', 'Camry', 'Accord', '3 Series', 'TL', 'A3', 'Malibu', 'Forrester']
}

@app.route('/')
def index():
    return render_template('index.html',
                           model_results=model_results,
                           best_model_name=best_model_name,
                           choices=CHOICES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Ensure all features present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Apply label encoding for known categorical cols
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0

        # Cast numerics
        for col in feature_names:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            except:
                input_df[col] = 0

        X = input_df[feature_names].values
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        return jsonify({
            'prediction': int(prediction),
            'label': 'FRAUD' if prediction == 1 else 'NOT FRAUD',
            'confidence': round(float(max(proba)) * 100, 2),
            'fraud_prob': round(float(proba[1]) * 100, 2),
            'not_fraud_prob': round(float(proba[0]) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-stats')
def model_stats():
    return jsonify({
        'results': {k: round(v*100, 2) for k, v in model_results.items()},
        'best': best_model_name,
        'best_accuracy': round(model_results[best_model_name]*100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
