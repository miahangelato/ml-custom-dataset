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

## 🧪 What You'll Do

You’ll create your own dataset in Excel or Google Sheets, convert it to CSV, and use it to train a machine learning model. You’ll then test predictions and (optionally) turn it into an API.

---

## 💡 Dataset Guidelines

Create a small dataset (~30–50 rows) with at least **2 input features (X)** and **1 target label (Y)**.

Here are examples to inspire you:

| Example Dataset           | X Features (inputs)                              | Y Label                 |
|---------------------------|--------------------------------------------------|-------------------------|
| Flowers                   | petal length, petal width                        | species (setosa, etc.)  |
| Fruits                    | weight, sugar %, color intensity                 | type (apple, banana...) |
| Vehicles                  | engine size, fuel type, tire size                | category (SUV, sedan)   |
| Students                  | study hours, sleep hours                         | pass/fail               |
| Drinks                    | calories, caffeine %, sugar level                | drink type              |

---

## 🛠️ Instructions

### 1. Create Your CSV Dataset

- Use Excel or Google Sheets
- Make at least **2 columns for input features**, and 1 for the **target label**
- Example:

```csv
petal_length,petal_width,species
1.4,0.2,setosa
4.7,1.4,versicolor
5.5,2.1,virginica
```

- Save/export it as: `my_dataset.csv`

---

### 2. Load and Visualize Your Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("my_dataset.csv")
print(df.head())

# Visualize the inputs colored by label
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title("My Custom Dataset")
plt.show()
```

---

### 3. Encode Labels & Train a Model

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

X = df[["petal_length", "petal_width"]]
y = LabelEncoder().fit_transform(df["species"])

model = RandomForestClassifier()
model.fit(X, y)
```

---

### 4. Make Predictions

```python
sample = [[5.1, 1.8]]
predicted = model.predict(sample)
print("Predicted label:", predicted)
```

---

### 5. (Optional) Wrap in a Django API

If you finish early, re-use your Django setup from last week and:

- Load your `my_dataset.csv`
- Train the model on startup
- Expose a POST endpoint like `/api/predict/` that takes input fields from your dataset

---

## 📄 Final Report Format

### ✅ Submit: GitHub Repo + Screenshot Folder or README

**Repo name:** `ml-custom-dataset`

**Expected files:**
- `my_dataset.csv`
- `train_model.py` or `notebook.ipynb`
- `README.md`
- `report/` folder (or screenshots inside README)

---

### 📷 Screenshots to Include

| Screenshot Topic              | Description                                |
|-------------------------------|--------------------------------------------|
| 1. CSV file opened            | Show your raw dataset in Excel/Sheets      |
| 2. pandas preview             | `print(df.head())`                         |
| 3. Dataset visualization      | Scatterplot or seaborn output              |
| 4. Model training output      | Console or cell showing training completed |
| 5–7. Sample predictions       | Different inputs + printed predictions     |
| 8–10. (Optional) Postman API  | If API was created                         |

---

### 📝 README.md Must Contain

- A short description of your dataset
- Features used as input, and target label
- Classifier used (e.g., RandomForest)
- 2–3 sample prediction results
- Instructions to run your code

---

## 🧠 Reflection Questions (Add to README)

- Why did you choose this dataset?
- Was it easy or hard to separate the classes visually?
- What would improve your model?

---

## 🔥 Bonus Challenge (+5 points)

- Add more than 2 features and visualize with a pairplot or 3D scatter
- Use a different model like `KNeighborsClassifier` or `LogisticRegression`
- Export your predictions to a new CSV file

---

## 🧠 Key Takeaway

Creating and cleaning your own dataset is where real machine learning begins. It teaches data design, visualization, and basic model thinking — from scratch.
