import pandas as pd
import numpy as np
import joblib

# =============================
# LOAD MODELS
# =============================
lgb_model = joblib.load("artifacts_s1_boosted/lgb_base.pkl")
xgb_model = joblib.load("artifacts_s1_boosted/xgb_base.pkl")
meta_model = joblib.load("artifacts_s1_boosted/meta_model.pkl")
feature_names = joblib.load("artifacts_s1_boosted/features.pkl")


# =============================
# PREDICTION FUNCTION
# =============================
def predict_single(input_data: dict):
    """
    input_data = {
        "age": 50,
        "gender": 1,
        "height": 165,
        "weight": 75,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    """
    
    df = pd.DataFrame([input_data])

    # ---------- CLEANING ----------
    df = df[(df["ap_hi"].between(80, 250)) & (df["ap_lo"].between(40, 150))]
    df = df[df["ap_hi"] > df["ap_lo"]]

    if df.empty:
        return {"error": "Invalid BP values."}

    # ---------- FEATURE ENGINEERING ----------
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["bmi_over"] = (df["bmi"] > 25).astype(int)

    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["pulse_cat"] = (df["pulse_pressure"] > 50).astype(int)

    df["bp_ratio"] = df["ap_hi"] / df["ap_lo"]
    df["bp_diff"] = df["ap_hi"] - df["ap_lo"]

    df["chol_high"] = (df["cholesterol"] > 1).astype(int)
    df["gluc_high"] = (df["gluc"] > 1).astype(int)

    df["chol_gluc_interact"] = df["cholesterol"] * df["gluc"]

    df["gender_female"] = (df["gender"] == 2).astype(int)
    df["age_squared"] = df["age"] ** 2

    # ---------- AGE QUANTILE RECONSTRUCTION ----------
    # Note: Using same binning as training but approximate (infer from training features)
    for col in feature_names:
        if col.startswith("agebin"):
            df[col] = 0

    # we approximate bins using domain knowledge
    age = df.loc[df.index[0], "age"]
    if age < 40:
        df["agebin_under40"] = 1
    elif age < 50:
        df["agebin_40_50"] = 1
    elif age < 60:
        df["agebin_50_60"] = 1
    else:
        df["agebin_60_plus"] = 1

    # ---------- MATCH TRAINING FEATURE ORDER ----------
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # ---------- BASE MODEL PREDICTIONS ----------
    lgb_pred = lgb_model.predict_proba(df)[:, 1]
    xgb_pred = xgb_model.predict_proba(df)[:, 1]

    # ---------- META STACKING ----------
    meta_input = np.vstack([lgb_pred, xgb_pred]).T  
    final_pred = meta_model.predict_proba(meta_input)[:, 1][0]

    # ---------- OUTPUT ----------
    return {
        "probability": float(final_pred),
        "prediction": int(final_pred >= 0.5)
    }


# =============================
# TEST EXAMPLE
# =============================
if __name__ == "__main__":
    sample = {
        "age": 52,
        "gender": 1,
        "height": 170,
        "weight": 81,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
import pandas as pd
import numpy as np
import joblib

# =============================
# LOAD MODELS
# =============================
lgb_model = joblib.load("artifacts_s1_boosted/lgb_base.pkl")
xgb_model = joblib.load("artifacts_s1_boosted/xgb_base.pkl")
meta_model = joblib.load("artifacts_s1_boosted/meta_model.pkl")
feature_names = joblib.load("artifacts_s1_boosted/features.pkl")


# =============================
# PREDICTION FUNCTION
# =============================
def predict_single(input_data: dict):
    """
    input_data = {
        "age": 50,
        "gender": 1,
        "height": 165,
        "weight": 75,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    """
    
    df = pd.DataFrame([input_data])

    # ---------- CLEANING ----------
    df = df[(df["ap_hi"].between(80, 250)) & (df["ap_lo"].between(40, 150))]
    df = df[df["ap_hi"] > df["ap_lo"]]

    if df.empty:
        return {"error": "Invalid BP values."}

    # ---------- FEATURE ENGINEERING ----------
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["bmi_over"] = (df["bmi"] > 25).astype(int)

    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["pulse_cat"] = (df["pulse_pressure"] > 50).astype(int)

    df["bp_ratio"] = df["ap_hi"] / df["ap_lo"]
    df["bp_diff"] = df["ap_hi"] - df["ap_lo"]

    df["chol_high"] = (df["cholesterol"] > 1).astype(int)
    df["gluc_high"] = (df["gluc"] > 1).astype(int)

    df["chol_gluc_interact"] = df["cholesterol"] * df["gluc"]

    df["gender_female"] = (df["gender"] == 2).astype(int)
    df["age_squared"] = df["age"] ** 2

    # ---------- AGE QUANTILE RECONSTRUCTION ----------
    # Note: Using same binning as training but approximate (infer from training features)
    for col in feature_names:
        if col.startswith("agebin"):
            df[col] = 0

    # we approximate bins using domain knowledge
    age = df.loc[df.index[0], "age"]
    if age < 40:
        df["agebin_under40"] = 1
    elif age < 50:
        df["agebin_40_50"] = 1
    elif age < 60:
        df["agebin_50_60"] = 1
    else:
        df["agebin_60_plus"] = 1

    # ---------- MATCH TRAINING FEATURE ORDER ----------
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # ---------- BASE MODEL PREDICTIONS ----------
    lgb_pred = lgb_model.predict_proba(df)[:, 1]
    xgb_pred = xgb_model.predict_proba(df)[:, 1]

    # ---------- META STACKING ----------
    meta_input = np.vstack([lgb_pred, xgb_pred]).T  
    final_pred = meta_model.predict_proba(meta_input)[:, 1][0]

    # ---------- OUTPUT ----------
    risk_percentage = round(final_pred * 100, 2)

    return {
        "heart_attack_risk_percent": risk_percentage,
        "prediction": int(final_pred >= 0.5)
    }



# =============================
# TEST EXAMPLE
# =============================
if __name__ == "__main__":
    sample = {
        "age": 60,
        "gender": 1,
        "height": 170,
        "weight": 90,
        "ap_hi": 140,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    print(predict_single(sample))

