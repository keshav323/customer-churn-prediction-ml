import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/churn.csv")

# Churn count
df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")

plt.savefig(
    "churn_distribution.png"
)

plt.show()