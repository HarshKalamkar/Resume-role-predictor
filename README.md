# Resume Role Predictor

This is a web application built with Flask and scikit-learn that predicts the most suitable job role based on the content of an uploaded resume PDF.

## Features

-   Upload a resume in PDF format.
-   Uses Natural Language Processing (NLP) and a trained Machine Learning model to analyze the text.
-   Predicts the job role with a confidence score.
-   Clean, modern, and responsive user interface.

## Folder Structure

```
/
|-- app.py              # Main Flask application
|-- requirements.txt    # Python dependencies
|-- .gitignore          # Files to be ignored by Git
|-- /model/
|   |-- trained_model.pkl
|   `-- tfidf_vectorizer.pkl
|-- /static/
|   `-- style.css
`-- /templates/
    |-- index.html
    `-- result.html
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

With your virtual environment activated and dependencies installed, run the following command in your terminal:

```bash
flask run
```
Or
```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000`.

## Technologies Used
-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, PyMuPDF
-   **Frontend:** HTML, CSS