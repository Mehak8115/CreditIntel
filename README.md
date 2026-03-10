<img width="1840" height="855" alt="image" src="https://github.com/user-attachments/assets/f177ef64-05d9-442b-8507-5ba0200aa128" />
# 🏦 CreditIntel

A machine learning-powered loan approval prediction system built with Streamlit and Random Forest Classifier. The system analyzes applicant financial data to predict loan approval probability with 97.07% accuracy.

## 📋 Overview

CreditIntel a Loan Approval Prediction System helps financial institutions evaluate loan applications efficiently by analyzing 11 key applicant features including income, employment status, credit history, loan amount, and asset values. The system provides instant predictions with detailed explanations and visualizations.

## ✨ Features

- **Real-time Predictions**: Instant loan approval/rejection predictions
- **High Accuracy**: 97.07% accuracy with Random Forest Classifier
- **Interactive Dashboard**: Beautiful Streamlit interface with multiple visualizations
- **Detailed Analysis**: 
  - Feature scorecard breakdown
  - Risk profile radar chart
  - Asset portfolio analysis
  - CIBIL score band positioning
  - Confusion matrix and model metrics
- **Comprehensive Scoring**: 100-point scoring system across 8 factors
- **Multiple Interfaces**: Two UI variants (app.py and loan_approval_app.py)

## 🎯 Model Performance

- **Accuracy**: 97.07%
- **Precision**: 96.23%
- **F1 Score**: 96.08%
- **ROC-AUC**: 96.84%

Test Set Confusion Matrix:
- True Positives: 523
- False Positives: 12
- False Negatives: 13
- True Negatives: 306

## 📊 Input Features (11)

1. **Dependents**: Number of financial dependents (0-10)
2. **Education**: Graduate / Not Graduate
3. **Self Employed**: Yes / No
4. **Annual Income**: Gross yearly income (₹)
5. **Loan Amount**: Requested loan amount (₹)
6. **Loan Term**: Repayment duration in years
7. **CIBIL Score**: Credit score (300-900)
8. **Residential Asset**: Residential property value (₹)
9. **Commercial Asset**: Commercial property value (₹)
10. **Luxury Asset**: Luxury goods/vehicles value (₹)
11. **Bank Asset**: Liquid bank balance/FDs (₹)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Run the Simple Interface (app.py):
```bash
streamlit run app.py
```

### Run the Advanced Interface (loan_approval_app.py):
```bash
streamlit run loan_approval_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 💡 Example

Here's a sample loan application scenario:

**Input:**
- Dependents: 2
- Education: Graduate
- Self Employed: No
- Annual Income: ₹600,000
- Loan Amount: ₹2,500,000
- Loan Term: 10 years
- CIBIL Score: 720
- Residential Asset: ₹3,000,000
- Commercial Asset: ₹1,000,000
- Luxury Asset: ₹500,000
- Bank Asset: ₹500,000

**Output:**
- **Decision**: ✅ APPROVED
- **Approval Probability**: 72.5%
- **RF Score**: 72.5/100

**Key Factors:**
- ✅ CIBIL score 720 is strong (≥700) - 21 points
- ✅ Annual income ₹600,000 meets requirements - 14 points
- ✅ Loan-to-income ratio 4.2x is healthy - 10 points
- ✅ Total asset coverage 2.0x covers loan well - 12 points

This applicant has a strong profile with good credit history, adequate income, and sufficient asset coverage, resulting in loan approval.

## 📈 Scoring System

The model uses a 100-point scoring system with the following weights:

- **CIBIL Score**: 28 points (28%)
- **Annual Income**: 18 points (18%)
- **Loan/Income Ratio**: 14 points (14%)
- **Asset Coverage**: 16 points (16%)
- **Loan Term**: 8 points (8%)
- **Dependents**: 8 points (8%)
- **Education**: 4 points (4%)
- **Employment Type**: 2 points (2%)

**Approval Threshold**: Score ≥ 55 points

## 🎨 UI Features

### Simple Interface (app.py)
- Clean, centered layout
- Gradient color scheme
- Metric cards with model performance
- Probability bar chart
- Input summary table

### Advanced Interface (loan_approval_app.py)
- Wide layout with sidebar
- Multiple interactive charts:
  - Gauge chart for approval probability
  - Feature score breakdown
  - Risk profile radar
  - Asset portfolio pie chart
  - CIBIL band positioning
  - Model performance metrics
  - Confusion matrix
- Detailed decision explanations
- Comprehensive model documentation

## 🤖 Model Details

The system uses a Random Forest Classifier with the following configuration:
- **Estimators**: 300 trees
- **Max Depth**: 12
- **Min Samples Split**: 4
- **Random State**: 42
- **Parallel Jobs**: -1 (uses all CPU cores)

The model is trained on synthetic data that mirrors real-world loan approval patterns. For production use, replace with your trained model by updating `rf_loan_model.pkl`.

## �️ Tech Stack

### Frontend
- **Streamlit** - Web application framework
- **HTML/CSS** - Custom styling and layouts
- **Plotly** - Interactive data visualizations

### Backend & ML
- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning library
- **Random Forest Classifier** - Prediction model
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Model Persistence
- **Pickle** - Model serialization

### Development Tools
- **Jupyter Notebook** - Model development and experimentation

## 📁 Project Structure

```
.
├── app.py                          # Simple Streamlit interface
├── loan_approval_app.py            # Advanced Streamlit interface
├── rf_loan_model.pkl               # Trained Random Forest model
├── Customer_Loan_Approval_System.ipynb  # Jupyter notebook
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── design.md                       # System design documentation
```

## 🔒 Security & Privacy

- All calculations are performed locally
- No data is stored or transmitted externally
- PII should be handled according to your organization's data protection policies

## 🛠️ Customization

### Update the Model
Replace the synthetic model with your trained model:
```python
# Train your model
clf = RandomForestClassifier(...)
clf.fit(X_train, y_train)

# Save it
import pickle
with open("rf_loan_model.pkl", "wb") as f:
    pickle.dump(clf, f)
```

### Adjust Scoring Weights
Modify the scoring logic in the `compute_rf_score()` function in `loan_approval_app.py`

### Customize UI Theme
Update the CSS in the `st.markdown()` sections at the top of each file


## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations using [Plotly](https://plotly.com/)

