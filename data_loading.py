import os
from sklearn.datasets import fetch_openml

os.makedirs("data", exist_ok=True)
bunch = fetch_openml(data_id=42178, as_frame=True)   # telco-customer-churn on OpenML
df = bunch.frame
df.to_csv("data/telco_clean.csv", index=False)

print(df.shape)
print(df.columns.tolist())
print(df["Churn"].value_counts())

import numpy as np, pandas as pd

df = pd.read_csv("data/telco_clean.csv")
rng = np.random.default_rng(42)
reasons = ["Price", "Competitor", "Service quality", "Moved"]

leaky = df.copy()
leaky["cancellation_reason"] = np.where(
    leaky["Churn"] == "Yes", rng.choice(reasons, len(leaky)), None

    
)
leaky.to_csv("data/telco_leaky.csv", index=False)
print(leaky.groupby("Churn")["cancellation_reason"].apply(lambda s: s.notna().mean()).to_dict())