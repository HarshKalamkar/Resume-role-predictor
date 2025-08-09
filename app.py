from flask import Flask, render_template, request, url_for
import pickle
import fitz  # PyMuPDF
import re
import os

app = Flask(__name__)

# --- Model and Vectorizer Loading ---
# (Your model loading code remains the same)
with open("model/trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# (Your helper functions remain the same)
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text_from_pdf(pdf_stream):
    text = ""
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def predict_role_from_text(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned]).toarray()
    proba = model.predict_proba(vector)[0]
    idx = proba.argmax()
    role = model.classes_[idx]
    confidence = round(proba[idx] * 100, 2)
    return role, confidence

# --- Flask Routes ---
@app.route("/")
def home():
    """Renders the main upload page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the file upload and prediction."""
    if "resume" not in request.files:
        return "No file part in the request", 400
    
    file = request.files["resume"]
    
    if file.filename == "":
        return "No file selected for uploading", 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            pdf_stream = file.read()
            raw_text = extract_text_from_pdf(pdf_stream)
            role, confidence = predict_role_from_text(raw_text)
            return render_template("result.html", role=role, confidence=confidence)
        
        except Exception as e:
            return f"Error processing the PDF file: {e}", 500
    else:
        return "Invalid file type. Please upload a PDF.", 400

if __name__ == "__main__":
    app.run(debug=True)