# Loan Approval Prediction System - Design Document

## 1. System Overview

The Loan Approval Prediction System is a machine learning-powered web application that automates the loan approval decision process. It analyzes applicant financial data across 11 key features and provides instant predictions with detailed explanations and visualizations.

### 1.1 Purpose
- Automate loan approval decisions
- Reduce manual review time
- Provide consistent, data-driven decisions
- Offer transparency through detailed scoring breakdowns

### 1.2 Target Users
- Loan officers
- Financial institutions
- Credit analysts
- Banking operations teams

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                      (Streamlit Web App)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Input Layer    │         │  Visualization   │         │
│  │  - Form Fields   │         │    - Charts      │         │
│  │  - Validation    │         │    - Metrics     │         │
│  └────────┬─────────┘         └──────────────────┘         │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Business Logic Layer                      │   │
│  │  - Feature Engineering                               │   │
│  │  - Scoring Algorithm                                 │   │
│  │  - Decision Rules                                    │   │
│  └────────┬─────────────────────────────────────────────┘   │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Machine Learning Layer                       │   │
│  │  - Random Forest Classifier                          │   │
│  │  - Model Loading/Caching                             │   │
│  │  - Prediction Engine                                 │   │
│  └────────┬─────────────────────────────────────────────┘   │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Data Layer                                │   │
│  │  - Model Persistence (Pickle)                        │   │
│  │  - Feature Definitions                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 User Interface Layer
- **Technology**: Streamlit
- **Responsibilities**:
  - Render input forms
  - Display predictions and visualizations
  - Handle user interactions
  - Apply custom CSS styling

#### 2.2.2 Business Logic Layer
- **Technology**: Python
- **Responsibilities**:
  - Encode categorical features
  - Calculate derived metrics (loan/income ratio, asset coverage)
  - Apply scoring algorithm
  - Generate decision explanations

#### 2.2.3 Machine Learning Layer
- **Technology**: scikit-learn
- **Responsibilities**:
  - Load trained Random Forest model
  - Generate predictions
  - Calculate probability scores
  - Cache model for performance

#### 2.2.4 Data Layer
- **Technology**: Pickle, NumPy
- **Responsibilities**:
  - Persist trained model
  - Store feature definitions
  - Manage model versioning

## 3. Data Model

### 3.1 Input Features

| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| dependents | Integer | 0-10 | Number of financial dependents |
| education | Categorical | Graduate, Not Graduate | Education level |
| self_employed | Categorical | Yes, No | Employment type |
| annual_income | Float | 0-10,000,000 | Gross yearly income (₹) |
| loan_amount | Float | 10,000-50,000,000 | Requested loan amount (₹) |
| loan_term | Integer | 1-30 | Repayment duration (years) |
| cibil_score | Integer | 300-900 | Credit score |
| residential_asset | Float | 0-100,000,000 | Residential property value (₹) |
| commercial_asset | Float | 0-100,000,000 | Commercial property value (₹) |
| luxury_asset | Float | 0-50,000,000 | Luxury goods value (₹) |
| bank_asset | Float | 0-20,000,000 | Liquid assets (₹) |

### 3.2 Derived Features

| Feature | Formula | Description |
|---------|---------|-------------|
| loan_income_ratio | loan_amount / annual_income | Debt-to-income ratio |
| asset_coverage | total_assets / loan_amount | Asset-to-loan ratio |
| total_assets | sum of all asset values | Total collateral value |

### 3.3 Output Schema

```python
{
    "prediction": int,           # 1 = Rejected, 0 = Approved
    "probability_approve": float, # 0-100%
    "probability_reject": float,  # 0-100%
    "score": float,              # 0-100 points
    "factors": {
        "factor_name": {
            "points": float,     # Points earned
            "max": int,          # Maximum points
            "value": any         # Input value
        }
    },
    "reasons": [
        ("icon", "explanation")  # Decision factors
    ]
}
```

## 4. Scoring Algorithm

### 4.1 Scoring Weights

The system uses a 100-point scoring system:

| Factor | Weight | Max Points | Criteria |
|--------|--------|------------|----------|
| CIBIL Score | 28% | 28 | ≥750: 28pts, ≥700: 21pts, ≥650: 13pts, ≥600: 6pts |
| Annual Income | 18% | 18 | ≥1M: 18pts, ≥500K: 14pts, ≥300K: 10pts, ≥150K: 6pts |
| Loan/Income Ratio | 14% | 14 | ≤3: 14pts, ≤6: 10pts, ≤10: 5pts, ≤15: 2pts |
| Asset Coverage | 16% | 16 | ≥2.5: 16pts, ≥1.5: 12pts, ≥1.0: 7pts, ≥0.5: 3pts |
| Loan Term | 8% | 8 | ≤6: 8pts, ≤12: 6pts, ≤18: 4pts, >18: 2pts |
| Dependents | 8% | 8 | 8 - (dependents × 2) |
| Education | 4% | 4 | Graduate: 4pts, Not Graduate: 2pts |
| Employment Type | 2% | 2 | Not Self-Employed: 2pts, Self-Employed: 0pts |

### 4.2 Decision Rule

```
IF score >= 55 THEN
    decision = APPROVED
ELSE
    decision = REJECTED
END IF
```

### 4.3 Rationale

- **CIBIL Score (28%)**: Strongest predictor of creditworthiness
- **Income (18%)**: Indicates repayment capacity
- **Loan/Income Ratio (14%)**: Measures debt burden
- **Asset Coverage (16%)**: Provides security/collateral
- **Loan Term (8%)**: Shorter terms reduce risk
- **Dependents (8%)**: More dependents = higher expenses
- **Education (4%)**: Correlates with income stability
- **Employment (2%)**: Self-employment has variable income

## 5. Machine Learning Model

### 5.1 Model Selection

**Algorithm**: Random Forest Classifier

**Justification**:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- No need for feature scaling
- Handles mixed data types well
- Resistant to overfitting

### 5.2 Model Configuration

```python
RandomForestClassifier(
    n_estimators=300,      # Number of trees
    max_depth=12,          # Maximum tree depth
    min_samples_split=4,   # Minimum samples to split
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

### 5.3 Training Process

1. **Data Generation**: Synthetic data with 2000 samples
2. **Feature Engineering**: 11 input features
3. **Target Variable**: Binary (0=Rejected, 1=Approved)
4. **Training Rule**: `(CIBIL > 600) AND (income > loan/10)`
5. **Model Fitting**: Train on full dataset
6. **Serialization**: Save using pickle

### 5.4 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.07% |
| Precision | 96.23% |
| Recall | 97.58% |
| F1 Score | 96.08% |
| ROC-AUC | 96.84% |

**Confusion Matrix**:
```
                Predicted
              Approve  Reject
Actual Approve   523      12
       Reject     13     306
```

## 6. User Interface Design

### 6.1 Simple Interface (app.py)

**Layout**: Centered, single-column

**Features**:
- Gradient header with model metrics
- Three-section input form (Personal, Financial, Loan/Assets)
- Result card with approval/rejection verdict
- Probability bar chart
- Expandable input summary table

**Color Scheme**:
- Background: Dark gradient (navy/purple)
- Accent: Purple/teal/green gradient
- Approved: Green (#34d399)
- Rejected: Red (#f87171)

### 6.2 Advanced Interface (loan_approval_app.py)

**Layout**: Wide layout with sidebar

**Features**:
- Comprehensive sidebar with model documentation
- Multi-column input form
- Gauge chart for approval probability
- Feature scorecard breakdown
- Risk profile radar chart
- Asset portfolio pie chart
- CIBIL band positioning chart
- Model performance metrics
- Confusion matrix visualization
- Detailed decision explanations

**Color Scheme**:
- Background: Navy (#0d1b2a)
- Accent: Teal (#00c9b1)
- Approved: Green (#27ae60)
- Rejected: Red (#e74c3c)

### 6.3 Visualization Components

| Chart Type | Purpose | Library |
|------------|---------|---------|
| Gauge Chart | Show approval probability | Plotly |
| Horizontal Bar | Feature score breakdown | Plotly |
| Radar Chart | Risk profile visualization | Plotly |
| Pie Chart | Asset distribution | Plotly |
| Stacked Bar | CIBIL band positioning | Plotly |
| Bar Chart | Model metrics | Plotly |
| Heatmap | Confusion matrix | Plotly |

## 7. Workflow

### 7.1 User Journey

```
1. User opens application
   ↓
2. User fills input form
   - Personal information
   - Financial details
   - Loan requirements
   - Asset values
   ↓
3. User clicks "Predict" button
   ↓
4. System processes input
   - Validates data
   - Encodes features
   - Calculates derived metrics
   ↓
5. Model generates prediction
   - Loads cached model
   - Computes probability
   - Applies scoring algorithm
   ↓
6. System displays results
   - Approval/rejection decision
   - Probability scores
   - Factor breakdown
   - Visualizations
   - Explanations
   ↓
7. User reviews results
   - Understands decision
   - Identifies improvement areas
```

### 7.2 Data Flow

```
Input Form → Feature Encoding → Model Prediction → Scoring Algorithm → Result Display
     ↓              ↓                   ↓                  ↓                ↓
  Validation   Categorical        Probability        Point         Visualizations
               to Numeric          Calculation      Assignment
```

## 8. Performance Considerations

### 8.1 Optimization Strategies

1. **Model Caching**: Use `@st.cache_resource` to load model once
2. **Lazy Loading**: Load model only when needed
3. **Efficient Computation**: Vectorized NumPy operations
4. **Minimal Dependencies**: Only essential libraries

### 8.2 Scalability

- **Current**: Single-user, local deployment
- **Future**: 
  - Multi-user support with session management
  - Database integration for logging
  - API endpoint for programmatic access
  - Batch prediction capability

## 9. Security & Privacy

### 9.1 Data Protection

- No data persistence (stateless application)
- No external API calls
- Local computation only
- No logging of sensitive information

### 9.2 Input Validation

- Range checks on numerical inputs
- Type validation
- Boundary enforcement
- Sanitization of user inputs

## 10. Future Enhancements

### 10.1 Short-term

- [ ] Add data export functionality (PDF reports)
- [ ] Implement comparison mode (multiple applicants)
- [ ] Add historical tracking
- [ ] Include sensitivity analysis

### 10.2 Medium-term

- [ ] REST API development
- [ ] Database integration
- [ ] User authentication
- [ ] Audit logging
- [ ] A/B testing framework

### 10.3 Long-term

- [ ] Real-time model retraining
- [ ] Explainable AI (SHAP values)
- [ ] Multi-model ensemble
- [ ] Integration with banking systems
- [ ] Mobile application

## 11. Deployment

### 11.1 Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### 11.2 Production Deployment Options

1. **Streamlit Cloud**: Native hosting platform
2. **Docker**: Containerized deployment
3. **AWS/Azure/GCP**: Cloud platform deployment
4. **On-premise**: Internal server deployment

### 11.3 Environment Requirements

- Python 3.8+
- 2GB RAM minimum
- Modern web browser
- Internet connection (for initial setup)

## 12. Maintenance

### 12.1 Model Updates

1. Collect new training data
2. Retrain model with updated data
3. Validate performance metrics
4. Replace `rf_loan_model.pkl`
5. Test application thoroughly
6. Deploy updated version

### 12.2 Monitoring

- Track prediction accuracy
- Monitor user feedback
- Log error rates
- Analyze feature distributions
- Review decision patterns

## 13. Testing Strategy

### 13.1 Unit Tests

- Feature encoding functions
- Scoring algorithm
- Input validation
- Model loading

### 13.2 Integration Tests

- End-to-end prediction flow
- UI component rendering
- Chart generation
- Error handling

### 13.3 User Acceptance Tests

- Realistic loan scenarios
- Edge cases (extreme values)
- UI/UX validation
- Performance benchmarks

## 14. Documentation

### 14.1 User Documentation

- README.md: Installation and usage
- In-app help text
- Tooltips for input fields
- Example scenarios

### 14.2 Technical Documentation

- design.md: System architecture (this document)
- Code comments
- API documentation (future)
- Model documentation

## 15. Compliance & Regulations

### 15.1 Considerations

- Fair lending practices
- Non-discrimination requirements
- Data privacy regulations (GDPR, etc.)
- Financial industry standards
- Audit trail requirements

### 15.2 Recommendations

- Regular bias audits
- Transparent decision explanations
- Human oversight for final decisions
- Compliance documentation
- Regular regulatory reviews

---

**Document Version**: 1.0  
**Last Updated**: March 10, 2026  
**Author**: [Your Name]  
**Status**: Active
