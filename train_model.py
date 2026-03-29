# ==========================================
# TRAIN BOTH MODELS + SAVE RESULTS
# ==========================================

import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df.sample(20000, random_state=42)

df["content"] = df["title"] + " " + df["text"]

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

df["content"] = df["content"].apply(clean)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["content"]).toarray()
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=100)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, y_pred_lr)
rf_acc = accuracy_score(y_test, y_pred_rf)

print("Logistic Regression:", lr_acc)
print("Random Forest:", rf_acc)

# Save everything
pickle.dump(lr, open("lr_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

pickle.dump((y_test, y_pred_lr, y_pred_rf), open("results.pkl", "wb"))

print("ALL MODELS SAVED")