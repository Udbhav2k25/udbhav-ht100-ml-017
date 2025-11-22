# Cardio Predictor — Heart Health Early Warning System

##  1. Problem Statement

Cardiovascular disease (CVD) remains one of the leading global causes of death.  
Early prediction enables timely lifestyle changes and clinical interventions.

The objective of this project is to build a **machine learning classifier** that predicts the probability of a person having **cardiovascular disease** using demographic, clinical, and behavioral parameters.

This tool can be used as a **decision-support system** for screening and risk assessment.

---

## 2. Dataset Overview

The project uses a **70,000-record cardiovascular dataset** containing **13 input features** and 1 binary target (`cardio`).

### **Features Used**

| Feature | Description |
|--------|-------------|
| `id` | Unique identifier |
| `age` | Age in days |
| `gender` | 1 = Female, 2 = Male |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic BP |
| `ap_lo` | Diastolic BP |
| `cholesterol` | 1 = Normal, 2 = Above normal, 3 = Well above normal |
| `gluc` | 1 = Normal, 2 = Above normal, 3 = Well above normal |
| `smoke` | 1 = Yes |
| `alco` | 1 = Yes |
| `active` | Physically active (1 = Yes) |
| **Target: `cardio`** | 1 = Has cardiovascular disease, 0 = No |

---

##  3. Methodology & ML Pipeline

### **Data Preprocessing**
- Removed unwanted columns such as `id` during modeling.
- Cleaned inconsistent entries (e.g., BP ranges).
- **Handled missing values** using `IterativeImputer`.
- **Normalized numerical features** using `StandardScaler`.

### **Modeling**
We trained 3 models:
- **Logistic Regression**
- **XGBoost Classifier**
- **Gradient Boosting Classifier**

Final prediction = **mean of all 3 model outputs** (simple ensemble voting).

### **Cross Validation**
- **10-Fold Cross Validation** to reduce overfitting.
- Stratified splits to preserve class imbalance.

---

## 4. Evaluation Metrics

Final ensemble performance:

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.8378** |
| **Recall** | **0.7195** |
| **F1-score** | **0.7424** |
| **Precision** | **0.7804** |

These metrics show a strong balance between **sensitivity (Recall)** and **precision**, making the model suitable for medical screening.

---

##  5. Installation & Usage

### **Install Dependencies**
```bash
pip install -r requirements.txt
