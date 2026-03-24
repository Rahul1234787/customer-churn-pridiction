import pandas as pd
import pickle

df = pd.read_csv("data/churn.csv")

df = df[["tenure", "MonthlyCharges", "Churn"]]

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

from sklearn.model_selection import train_test_split

X = df[["tenure", "MonthlyCharges"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("NEW MODEL CREATED ✅")
