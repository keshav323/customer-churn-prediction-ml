import pandas as pd
import joblib

df = pd.read_csv("data/churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.fillna(0, inplace=True)

df = pd.get_dummies(df, drop_first=True)

feature_names = df.drop(
    "Churn_Yes",
    axis=1
).columns.tolist()

joblib.dump(
    feature_names,
    "models/features.pkl"
)

print("Features saved.")