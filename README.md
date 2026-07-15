# Loan Approval Classifier

Files:
- [train.py](train.py): trains multiple classifiers and saves the best pipeline to `model.joblib` and metrics to `metrics.json`.
- [app.py](app.py): Streamlit app to input applicant data and predict loan approval.
- [requirements.txt](requirements.txt): Python dependencies.

Quick start:

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the models and save the best one:

```powershell
python train.py
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Then open the Streamlit URL typically at http://localhost:8501 to interact with the form and get predictions.
