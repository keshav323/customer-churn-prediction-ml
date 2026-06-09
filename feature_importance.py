import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.fillna(0, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)

model = joblib.load(
    "models/churn_model.pkl"
)

importance = model.feature_importances_

feature_imp = pd.Series(
    importance,
    index=X.columns
)

feature_imp.nlargest(10).plot(
    kind="barh"
)

plt.title(
    "Top 10 Important Features"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.show()