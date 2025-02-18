# Filename: app.py
import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from torchvision import models, transforms

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Load the pre-trained ResNet model for classification
def get_classification_model(num_classes):
    model = models.resnet50(weights=None)  # Use None since we're loading custom weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Initialize model and load weights
num_classes = 5  # Number of classes: Mild, Moderate, Normal, Proliferative, Severe
model = get_classification_model(num_classes)
model.load_state_dict(torch.load("model/dr_stage_classification.pth", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformations for the prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same normalization as training
])

# Store previous results in a list (you can also use a database)
previous_results = []

# Recommendations based on the detected stage
def get_recommendations(stage):
    recommendations = {
        "Normal": {
            "therapy": "No specific therapy required.",
            "operation": "No surgical operation needed.",
            "suggestions": "Regular eye check-ups are recommended.",
            "health_tips": "Maintain a healthy diet, exercise regularly, and control blood sugar levels."
        },
        "Mild": {
            "therapy": "Early-stage therapy to control blood sugar levels.",
            "operation": "No surgical operation needed at this stage.",
            "suggestions": "Regular follow-up with an eye specialist.",
            "health_tips": "Avoid high blood sugar levels and monitor your eyes for any changes."
        },
        "Moderate": {
            "therapy": "Anti-VEGF injections or laser treatment might be recommended.",
            "operation": "Surgical operation is not usually needed unless complications arise.",
            "suggestions": "Consult an ophthalmologist for potential treatment options.",
            "health_tips": "Manage blood sugar and blood pressure to reduce the risk of progression."
        },
        "Severe": {
            "therapy": "Laser treatment or Anti-VEGF injections to slow the progression.",
            "operation": "Consider surgical options if vision is significantly affected.",
            "suggestions": "Intensive follow-up and more frequent eye examinations.",
            "health_tips": "Strict management of diabetes, blood pressure, and cholesterol."
        },
        "Proliferative": {
            "therapy": "Immediate treatment is required, often with Anti-VEGF injections or laser surgery.",
            "operation": "Vitrectomy surgery may be needed to remove blood or scar tissue from the eye.",
            "suggestions": "Seek medical intervention to prevent vision loss.",
            "health_tips": "Aggressively manage diabetes and other health conditions to prevent further complications."
        }
    }
    return recommendations.get(stage, {})

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", previous_results=previous_results)

@app.route("/predict", methods=["POST"])
def predict():
    # Check for patient name and uploaded file
    patient_name = request.form.get("patient_name", "")
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_image.jpg")
        file.save(file_path)

        # Open image and apply transformations
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Map prediction to class labels
        class_labels = ["Mild", "Moderate", "Normal", "Proliferative", "Severe"]
        detected_stage = class_labels[predicted.item()]
        result = f"Detected Stage: {detected_stage}"

        # Get recommendations for the detected stage
        recommendations = get_recommendations(detected_stage)

        # Separate recommendations for individual cards
        therapy = recommendations.get("therapy", "No therapy recommendation available.")
        operation = recommendations.get("operation", "No operation recommendation available.")
        suggestions = recommendations.get("suggestions", "No suggestions available.")
        health_tips = recommendations.get("health_tips", "No health tips available.")

        # Store the result along with the patient name and recommendations
        previous_results.append({"name": patient_name, "result": result, "therapy": therapy, "operation": operation, "suggestions": suggestions, "health_tips": health_tips})

        # Render the result on the index page along with the patient name and individual recommendations
        return render_template("index.html", result=result, patient_name=patient_name, previous_results=previous_results, therapy=therapy, operation=operation, suggestions=suggestions, health_tips=health_tips)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    # Clear all previous results
    previous_results.clear()
    return redirect(url_for("index"))

@app.route("/get_detection_results")
def get_detection_results():
    stages = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
    results = {stage: sum(1 for item in previous_results if item['result'].endswith(stage)) for stage in stages}
    chart_data = [{"stage": stage, "count": count} for stage, count in results.items()]
    return jsonify(chart_data)

if __name__ == "__main__":
    app.run(debug=True)