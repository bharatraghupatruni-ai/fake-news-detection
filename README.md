# 📰 Fake News Detection System

## 📌 Overview
This project is a Machine Learning-based web application that classifies news articles as **Real or Fake** using Natural Language Processing (NLP).

The system compares:
- Logistic Regression (Traditional ML)
- Random Forest (Ensemble Learning)

It is deployed using Streamlit and provides:
✔ Real-time predictions  
✔ Confidence scores  
✔ Explanation (why fake/real)  
✔ Model comparison  
✔ Visualizations  

---

## 🚀 Live Features
- Text classification using TF-IDF
- Ensemble vs Traditional ML comparison
- Interactive Streamlit dashboard
- Confidence score visualization
- Reasoning behind predictions
- Accuracy comparison graph
- Confusion matrix visualization

---

## 📊 Dataset

### 🔗 Source (Kaggle)
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### 🔗 Direct CSV Links
- Fake News:  
https://raw.githubusercontent.com/rahulnyk/Detecting-Fake-News/master/data/Fake.csv

- Real News:  
https://raw.githubusercontent.com/rahulnyk/Detecting-Fake-News/master/data/True.csv

---

## 🧠 Models Used

| Model | Type | Description |
|------|------|------------|
| Logistic Regression | Traditional ML | Fast, baseline model |
| Random Forest | Ensemble Learning | Higher accuracy, robust |

---

## ⚙️ Technologies Used
- Python  
- scikit-learn  
- Streamlit  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

## ▶️ How to Run

### 1. Install Dependencies# 📰 Fake News Detection System

## 📌 Overview
This project is a Machine Learning-based web application that classifies news articles as **Real or Fake** using Natural Language Processing (NLP).

The system compares:
- Logistic Regression (Traditional ML)
- Random Forest (Ensemble Learning)

It is deployed using Streamlit and provides:
✔ Real-time predictions  
✔ Confidence scores  
✔ Explanation (why fake/real)  
✔ Model comparison  
✔ Visualizations  

---

## 🚀 Live Features
- Text classification using TF-IDF
- Ensemble vs Traditional ML comparison
- Interactive Streamlit dashboard
- Confidence score visualization
- Reasoning behind predictions
- Accuracy comparison graph
- Confusion matrix visualization

---

## 📊 Dataset

### 🔗 Source (Kaggle)
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### 🔗 Direct CSV Links
- Fake News:  
https://raw.githubusercontent.com/rahulnyk/Detecting-Fake-News/master/data/Fake.csv

- Real News:  
https://raw.githubusercontent.com/rahulnyk/Detecting-Fake-News/master/data/True.csv

---

## 🧠 Models Used

| Model | Type | Description |
|------|------|------------|
| Logistic Regression | Traditional ML | Fast, baseline model |
| Random Forest | Ensemble Learning | Higher accuracy, robust |

---

## ⚙️ Technologies Used
- Python  
- scikit-learn  
- Streamlit  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

## ▶️ How to Run

### 1. Install Dependencies
pip install streamlit pandas scikit-learn matplotlib seaborn
pip install streamlit pandas scikit-learn matplotlib seaborn


### 2. Train Model

python train_model.py


### 3. Run App

python -m streamlit run app.py

---

## 📈 Output
- Predicts **REAL or FAKE news**
- Displays **confidence score**
- Provides **reasoning**
- Shows **model comparison graphs**
- Displays **confusion matrix**

---

## 🔍 Example

**Input:**
Breaking shocking truth revealed government hiding secrets!!!

**Output:**
FAKE NEWS  
Confidence: 0.97  

---

## 📸 Screenshots

### 🏠 Dashboard
![Dashboard](screenshots/dashboard.png)

### 🔍 Prediction Output
![Prediction](screenshots/prediction.png)

### 📊 Model Comparison
![Comparison](screenshots/comparison.png)

### 📉 Confusion Matrix
![Confusion Matrix](screenshots/confusion.png)

---

## 🧠 Why Ensemble Learning?
Random Forest improves performance by:
- Combining multiple decision trees  
- Reducing overfitting  
- Capturing complex patterns  

---

## ⚠️ Limitations
- Does not verify factual correctness  
- Based on learned patterns  
- Short inputs may reduce accuracy  

---

## 📌 Conclusion
This project demonstrates that **ensemble learning outperforms traditional models** in fake news detection by improving accuracy and robustness.

