# 📊 Activity 2: Create, Visualize, and Model Your Own Dataset

## 🕒 Time: 2 hours

---

## 🎯 Objectives

By the end of this activity, you should be able to:

- Generate synthetic datasets using Scikit-Learn
- Visualize datasets with Matplotlib or Seaborn
- Train a basic classifier (e.g., RandomForest or KNN)
- Test predictions from user input or hardcoded samples
- (Optional) Wrap it in a Django API endpoint
- Prepare and submit a clear final report

---

## 📦 Requirements

- Python 3.10+
- `scikit-learn`, `matplotlib`, `seaborn`, `pandas`
- CLI or Jupyter Notebook or Django (your choice)
- (Optional) Postman if API is implemented

---

## 🧪 What You'll Build

A mini machine learning project that:

1. Creates your **own synthetic dataset**
2. **Visualizes** the dataset in 2D
3. Trains a **simple model**
4. Predicts class/label of **new data**
5. (Optional) Exposes your model as a **Django API**

---

## 🛠️ Instructions

### 1. Setup Your Environment

Create a folder or repo called `ml-data-lab`.

Install dependencies:

```bash
pip install scikit-learn matplotlib seaborn pandas
```

---

### 2. Generate Your Dataset

Choose one:

```python
from sklearn.datasets import make_classification, make_blobs

# Example: 3 classes, 2 features
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)
```

OR

```python
X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)
```

---

### 3. Visualize the Dataset

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
plt.title("Custom Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
```

---

### 4. Train a Classifier

Choose one: RandomForestClassifier, KNeighborsClassifier, or DecisionTreeClassifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
```

---

### 5. Make Predictions

Ask for user input or test with hardcoded samples:

```python
sample = [[5.5, 2.1]]
prediction = model.predict(sample)
print("Predicted class:", prediction)
```

---

### 6. (Optional) API Extension

If you have extra time, wrap your model in a Django API like last time.  
Use the same structure from your previous activity but replace the iris model with your own.

---

## 📄 Final Report Format (To Be Submitted)

### ✅ Submit as: **GitHub Repository + Screenshot Report**

- **GitHub Repo Name:** `ml-data-lab`
- **Files Expected:**
  - `generate_data.py` or `notebook.ipynb`
  - `model.py`
  - `predict.py` or `api/` folder (if using Django)

---

### 📷 Screenshot Requirements

Place these screenshots in a folder called `report/` or inside your README.md:

| Screenshot Topic              | Description                                      |
|-------------------------------|--------------------------------------------------|
| 1. Dataset Visualization      | Scatter plot showing your generated dataset     |
| 2. Model Training Output      | Any CLI output showing model was trained        |
| 3–6. Predictions              | At least 3–5 different input/output samples      |
| 7–10. (Optional) Postman      | Screenshots of API working with your dataset     |

---

### 📝 README.md Must Contain

- Brief description of your dataset (what kind of distribution)
- What classifier you chose and why
- Sample prediction example (input + output)
- How to run your code (instructions or command)

---

## 💡 Grading Guide

| Criteria                       | Points |
|--------------------------------|--------|
| Dataset Generated & Visualized | 20     |
| Model Trained Successfully     | 20     |
| Predictions Made & Explained   | 20     |
| Code is Organized & Readable   | 20     |
| Report + Repo Submitted        | 20     |
| **Total**                      | **100**|

---

## 🧠 Bonus Questions (Optional for +5 points)

Answer in your README:

- What would happen if you added more noise or classes?
- Would your model still work well with more overlap?
- What real-world data might resemble this?

---

## 🚀 Now Go Build Your Own ML Lab!
