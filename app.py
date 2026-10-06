import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Single HTML template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Classifier Portal</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: #f1f5f9;
            color: #334155;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: #ffffff;
            width: 100%;
            max-width: 520px;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        .header {
            margin-bottom: 24px;
            text-align: center;
        }

        .header h2 {
            font-size: 24px;
            color: #0f172a;
            font-weight: 700;
        }

        .header p {
            font-size: 14px;
            color: #64748b;
            margin-top: 4px;
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 6px;
        }

        input[type="number"], select {
            width: 100%;
            padding: 10px 14px;
            font-size: 15px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background-color: #f8fafc;
            color: #1e293b;
            outline: none;
            transition: all 0.2s ease;
        }

        input[type="number"]:focus, select:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
            background-color: #ffffff;
        }

        .form-row {
            display: flex;
            gap: 16px;
        }

        .form-row .form-group {
            flex: 1;
        }

        button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            background-color: #2563eb;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
            transition: all 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 6px 10px -1px rgba(37, 99, 235, 0.4);
            transform: translateY(-1px);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .result-yes {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #15803d;
        }

        .result-no {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #b91c1c;
        }

        .result-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            opacity: 0.8;
        }

        .result-value {
            font-size: 22px;
            font-weight: 700;
            margin-top: 4px;
        }

        .probability-container {
            margin-top: 16px;
            display: flex;
            justify-content: space-between;
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-size: 13px;
            color: #475569;
        }

        .error-message {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 12px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h2>KNN Classifier Portal</h2>
        <p>Predict user outcome based on parameters</p>
    </div>

    {% if error %}
    <div class="error-message">
        {{ error }}
    </div>
    {% endif %}

    <form action="/predict" method="POST">
        <div class="form-row">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" min="1" max="120" value="{{ form_data.age if form_data else 25 }}" required>
            </div>
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender">
                    <option value="Female" {% if form_data and form_data.gender == 'Female' %}selected{% endif %}>Female</option>
                    <option value="Male" {% if form_data and form_data.gender == 'Male' %}selected{% endif %}>Male</option>
                </select>
            </div>
        </div>

        <div class="form-row">
            <div class="form-group">
                <label for="review">Review Rating</label>
                <select id="review" name="review">
                    <option value="Poor" {% if form_data and form_data.review == 'Poor' %}selected{% endif %}>Poor</option>
                    <option value="Average" {% if form_data and form_data.review == 'Average' %}selected{% endif %}>Average</option>
                    <option value="Good" {% if form_data and form_data.review == 'Good' %}selected{% endif %}>Good</option>
                </select>
            </div>
            <div class="form-group">
                <label for="education">Education</label>
                <select id="education" name="education">
                    <option value="School" {% if form_data and form_data.education == 'School' %}selected{% endif %}>School</option>
                    <option value="UG" {% if form_data and form_data.education == 'UG' %}selected{% endif %}>UG</option>
                    <option value="PG" {% if form_data and form_data.education == 'PG' %}selected{% endif %}>PG</option>
                </select>
            </div>
        </div>

        <button type="submit">Predict Outcome</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box {% if is_yes %}result-yes{% else %}result-no{% endif %}">
        <div class="result-title">Prediction Decision</div>
        <div class="result-value">{% if is_yes %}✅ YES{% else %}❌ NO{% endif %}</div>
    </div>

    {% if proba %}
    <div class="probability-container">
        <span><strong>No Confidence:</strong> {{ "%.1f"|format(proba[0] * 100) }}%</span>
        <span><strong>Yes Confidence:</strong> {{ "%.1f"|format(proba[1] * 100) }}%</span>
    </div>
    {% endif %}
    {% endif %}
</div>

</body>
</html>
"""

def load_model():
    """Loads the pre-trained KNN model pickle file."""
    model_path = "knn.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error="Model file `knn.pkl` not found. Please ensure it exists in the root folder.",
            prediction=None
        )

    try:
        # Get raw data from input fields
        age = int(request.form.get("age", 25))
        gender = request.form.get("gender", "Female")
        review = request.form.get("review", "Poor")
        education = request.form.get("education", "School")

        # Categorical feature mappings
        gender_map = {"Female": 0, "Male": 1}
        review_map = {"Poor": 0, "Average": 1, "Good": 2}
        education_map = {"School": 0, "UG": 1, "PG": 2}

        # Vector assembly
        encoded_input = np.array([[
            age,
            gender_map.get(gender, 0),
            review_map.get(review, 0),
            education_map.get(education, 0)
        ]])

        # Model Inference
        raw_pred = model.predict(encoded_input)[0]
        
        # Safely convert prediction result to a string or Python scalar
        if hasattr(raw_pred, "item"):
            pred_val = raw_pred.item()
        else:
            pred_val = raw_pred

        proba = model.predict_proba(encoded_input)[0].tolist() if hasattr(model, "predict_proba") else None

        # Boolean evaluation check
        is_yes = str(pred_val).strip().lower() in ["yes", "1", "true"]

        form_data = {
            "age": age,
            "gender": gender,
            "review": review,
            "education": education
        }

        return render_template_string(
            HTML_TEMPLATE,
            prediction=pred_val,
            is_yes=is_yes,
            proba=proba,
            form_data=form_data
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            error=f"Prediction Error: {str(e)}",
            prediction=None
        )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)