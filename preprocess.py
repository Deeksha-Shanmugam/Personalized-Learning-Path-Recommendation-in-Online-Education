import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_merge(sample_size=None, replace=False):
    info = pd.read_csv("data/studentInfo.csv")
    assess = pd.read_csv("data/studentAssessment.csv")
    vle = pd.read_csv("data/studentVle.csv")

    # Aggregate assessment score per student
    assess_agg = assess.groupby("id_student")["score"].mean().reset_index().rename(columns={"score": "avg_score"})
    data = pd.merge(info, assess_agg, on="id_student", how="left")

    # Aggregate sum_click per student
    vle_agg = vle.groupby("id_student")["sum_click"].sum().reset_index().rename(columns={"sum_click": "total_clicks"})
    data = pd.merge(data, vle_agg, on="id_student", how="left")

    data["avg_score"] = data["avg_score"].fillna(0)
    data["total_clicks"] = data["total_clicks"].fillna(0)

    # Optional: Sample a subset of the data if sample_size is provided
    if sample_size:
        data = data.sample(n=sample_size, random_state=42, replace=True)

    # Encode categorical columns
    for col in ["gender", "age_band", "region", "highest_education", "disability", "final_result"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Select features and label
    features = ["gender", "age_band", "region", "highest_education", "disability", "studied_credits", "num_of_prev_attempts", "avg_score", "total_clicks"]
    label = "final_result"

    from sklearn.preprocessing import MinMaxScaler

    # After label encoding and selecting features
    scaler = MinMaxScaler()

    # Scale only numerical columns
    num_cols = ["studied_credits", "num_of_prev_attempts", "avg_score", "total_clicks"]
    data[num_cols] = scaler.fit_transform(data[num_cols])

    
    return data[["id_student", "code_module"] + features + [label]], features, label
