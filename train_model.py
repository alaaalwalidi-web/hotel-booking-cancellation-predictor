import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1) Load dataset
df = pd.read_csv("Hotel Reservations.csv")


# 2) Basic EDA
print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset shape:")
print(df.shape)

print("\nStatistical summary:")
print(df.describe())

print("\nDuplicated rows:")
print(df.duplicated().sum())

print("_" * 80)


# 3) Target distribution
print("\nBooking status counts:")
print(df["booking_status"].value_counts())

print("\nBooking status percentage:")
print(df["booking_status"].value_counts(normalize=True) * 100)

print("_" * 80)


# 4) Outliers check
print("\nOutlier-related columns summary:")
print(df[["lead_time", "avg_price_per_room", "no_of_children"]].describe())

print("_" * 80)


# 5) Outliers handling

# avg_price_per_room: remove bottom 1% and top 1%
q_low = df["avg_price_per_room"].quantile(0.01)
q_high = df["avg_price_per_room"].quantile(0.99)

df = df[
    (df["avg_price_per_room"] >= q_low) &
    (df["avg_price_per_room"] <= q_high)
]

# lead_time: remove top 1%
q_high = df["lead_time"].quantile(0.99)
df = df[df["lead_time"] <= q_high]

# no_of_children: remove unrealistic values
df = df[df["no_of_children"] <= 5]

print("\nShape after outlier handling:")
print(df.shape)

print("_" * 80)


# 6) Define X and y
y = df["booking_status"]

X = df.drop(columns=["booking_status", "Booking_ID"])


# 7) Encoding categorical variables
X = pd.get_dummies(X, drop_first=True, dtype="int8")


# 8) Train / validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9) Logistic Regression with scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

model1 = LogisticRegression(max_iter=2000)
model1.fit(X_train_scaled, y_train)

pred1 = model1.predict(X_valid_scaled)


print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_valid, pred1))
print(classification_report(y_valid, pred1))
print(confusion_matrix(y_valid, pred1))


# 10) Logistic Regression Cross Validation using Pipeline
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

cv_lr = cross_val_score(pipe_lr, X, y, cv=5, scoring="accuracy")

print("\nLogistic CV Accuracy:")
print(cv_lr.mean())


# 11) Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_valid)


print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_valid, pred_rf))
print(classification_report(y_valid, pred_rf))
print(confusion_matrix(y_valid, pred_rf))


# 12) Random Forest Cross Validation
cv_rf = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

print("\nRandom Forest CV Accuracy:")
print(cv_rf.mean())


# 13) Model comparison
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy_Test": [
        accuracy_score(y_valid, pred1),
        accuracy_score(y_valid, pred_rf)
    ],
    "Accuracy_CV": [
        cv_lr.mean(),
        cv_rf.mean()
    ]
})

print("\nModel Comparison:")
print(results)


# 14) Feature Importance
importance = pd.Series(rf.feature_importances_, index=X.columns)

importance.sort_values().tail(10).plot(kind="barh")
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# 15) Save final model and columns
best_model = rf

joblib.dump(best_model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel saved as model.pkl")
print("Columns saved as columns.pkl")