# 📊 Activity 2: Build a Custom Dataset and Train a Classifier

## 🕒 Time: ~2 hours

---

## 🎯 Objectives

- Create your own labeled dataset and save it as a `.csv` file
- Load and visualize your dataset using Python
- Train a classifier using `scikit-learn`
- Test predictions on custom data
- (Optional) Wrap it into a Django API

---

## 💻 Project Folder Structure

```
ml-custom-dataset/
├── my_dataset.csv                  <-- Your custom dataset (you'll create this)
├── train_model.py                  <-- Loads, visualizes, trains model
├── predict.py                      <-- Separate script to test predictions
├── requirements.txt                <-- Dependencies list
└── README.md                       
```

---

## 📦 Setup Instructions

### 1. Create Your Project Folder

```bash
mkdir ml-custom-dataset
cd ml-custom-dataset
```

---

### 2. Create Your Dataset (CSV)

Open Excel or Google Sheets. Make a dataset with at least 2 numeric input features and 1 label.

Example:

```
petal_length,petal_width,species
1.4,0.2,setosa
4.7,1.4,versicolor
5.5,2.1,virginica
...
```

- Save/export as `my_dataset.csv` in your project folder.
- Recommended size: 30–50 rows.

---

### 3. Create `requirements.txt`

```txt
pandas
matplotlib
seaborn
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

### 4. Create `train_model.py`

This script loads your CSV, visualizes the data, and trains a model.

```python
# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load dataset
df = pd.read_csv("my_dataset.csv")
print("First 5 rows:")
print(df.head())

# 2. Visualize (change x/y if you use different columns)
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title("Custom Dataset")
plt.show()

# 3. Prepare features and labels
X = df[["petal_length", "petal_width"]]
le = LabelEncoder()
y = le.fit_transform(df["species"])

# 4. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 5. Save model and encoder for later use
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved as model.pkl")
```

---

### 5. Create `predict.py`

This file loads the trained model and predicts new data.

```python
# predict.py

import joblib

# Load saved model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Test input (edit this!)
sample = [[5.1, 1.8]]  # petal_length, petal_width

# Predict
pred = model.predict(sample)
label = le.inverse_transform(pred)

print("Prediction:", label[0])
```

---

## 🔥 (Optional) API Extension

Already familiar with Django from Activity 1?  
Recreate a `/predict/` endpoint using your trained `model.pkl`.

---

## 📄 Final Report Format

### ✅ Submit: GitHub Repo + Screenshot Folder or README

**Repo name:** `ml-custom-dataset`

**Expected Files:**

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `my_dataset.csv`         | Your custom dataset                  |
| `train_model.py`         | Trains and saves model               |
| `predict.py`             | Loads and predicts new input         |
| `README.md`              | Final writeup and summary            |
| `report/` folder         | Screenshots (optional)               |

---

### 📷 Screenshot Requirements

| Screenshot Topic              | Description                                 |
|-------------------------------|---------------------------------------------|
| 1. Raw dataset                | CSV file shown in Excel or Sheets           |
| 2. pandas preview             | `print(df.head())` from `train_model.py`    |
| 3. Visualization              | Scatterplot or seaborn output               |
| 4. Training output            | CLI print confirming model was trained      |
| 5–7. Sample predictions       | Console printouts from `predict.py`         |
| 8–10. (Optional) Postman/API  | If API was added                           |

---

### 📝 `README.md` Should Include:

- ✅ Brief explanation of your dataset (what & why)
- ✅ Features and label used
- ✅ Classifier used (e.g., RandomForest)
- ✅ At least 2 sample predictions
- ✅ How to run your code

---

### 🧠 Reflection Questions (Add to README)

- Why did you choose this dataset?
- What do you think affects prediction accuracy?
- How could you improve this in the future?

---

## 💡 Bonus Ideas (Optional for +5 pts)

- Add a third feature (3D scatter plot!)
- Export predictions to a new CSV
- Compare accuracy between different classifiers (e.g., KNN vs RF)

---

## ✅ Grading Guide

| Criteria                             | Points |
|--------------------------------------|--------|
| Dataset Created and Loaded Correctly | 20     |
| Visualization with Plot              | 20     |
| Model Training                       | 20     |
| Sample Predictions                   | 20     |
| Organized Repo + Report              | 20     |
| **TOTAL**                            | **100**|

---

## 🎉 Congratulations!

You've now created your own dataset, trained a model, made predictions, and optionally exposed it as an API — just like real-world data science!
