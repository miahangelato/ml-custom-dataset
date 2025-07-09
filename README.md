## 📝 Manese    Sanchez     To

---
# 📊 Activity 3: Build a Custom Dataset and Expose a Machine Learning API

---
## 🕒 Time: ~2 hours

---

## 🎯 Objectives

- Create your own labeled dataset and save it as a `.csv` file
- Load and visualize your dataset using Python
- Train a classifier using `scikit-learn`
- Save the model and label encoder to `.pkl` files
- Create a Django REST API that accepts input and returns predictions

---

## 💻 Project Folder Structure

```
ml-custom-dataset/
├── my_dataset.csv                  <-- Your custom dataset
├── train_model.py                  <-- Trains model, saves .pkl files
├── predict.py                      <-- (Optional) CLI tester for predictions
├── requirements.txt
├── README.md
├── report/                         <-- (Optional) Screenshots folder
└── ml_api_project/                 <-- Your Django project root
    ├── db.sqlite3
    ├── manage.py
    ├── ml_api_project/
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── ml_api/
        ├── __init__.py
        ├── admin.py
        ├── apps.py
        ├── label_encoder.pkl
        ├── model.pkl
        ├── migrations/
        │   └── __init__.py
        ├── models.py
        ├── tests.py
        ├── urls.py
        └── views.py

```

---

## 🛠️ Part 1: Dataset and Model

### Our Dataset in Excel or Sheets

```
hours_sleep,screen_time,caffeine_mg,alertness_level
8,2,0,high
6,5,100,medium
5,6,150,low
7,3,50,medium
4,7,200,low
9,1,0,high
6,4,120,medium
3,8,250,low
7,2,30,high
5,6,180,low
6,3,60,medium
8,2,20,high
4,7,220,low
7,4,90,medium
9,1,0,high

```

Saved as `my_dataset.csv`.

---

### Created `train_model.py`

```python
# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("my_dataset.csv")

sns.scatterplot(data=df, x="hours_sleep",y="screen_time",hue="alertness_level")
plt.title("Custom Dataset")
plt.show()

X = df[["hours_sleep", "screen_time","caffeine_mg" ]]
le = LabelEncoder()
y = le.fit_transform(df["alertness_level"])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")
```

---

### Created `predict.py`

```python
# predict.py

import joblib

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

sample = [[7, 3, 50]]  # update with 3 features matching your dataset
prediction = model.predict(sample)
print("Prediction:", le.inverse_transform(prediction)[0])

```

---

## 🌐 Part 2: Django API Setup

---

### Created Django Project and App

```bash
django-admin startproject ml_api_project .
py manage.py startapp ml_api
```

---

### Updated `ml_api_project/settings.py`

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
    'ml_api',
]
```

---

### Created `ml_api/urls.py`

```python
from django.urls import path
from .views import PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]
```

---

### 4. Updated `ml_api_project/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ml_api.urls')),
]
```

---

### Created `ml_api/views.py`

```python
# ml_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class PredictView(APIView):
    def post(self, request):
        try:
            hours_sleep = float(request.data.get("hours_sleep"))
            screen_time = float(request.data.get("screen_time"))
            caffeine_mg = float(request.data.get("caffeine_mg"))

            prediction = model.predict([[hours_sleep, screen_time, caffeine_mg]])
            label = label_encoder.inverse_transform(prediction)[0]

            return Response({"prediction": label})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


---

### 6. Run Server

```bash
py manage.py runserver
```

Test via Postman:
- POST to `http://localhost:8000/api/predict/`
- Body (JSON):

```json
{
  "hours_sleep": 7,
  "screen_time": 3,
  "caffeine_mg": 50
}
```

---

## ✅ Grading Guide

| Criteria                             | Points |
|--------------------------------------|--------|
| Dataset Created and Loaded Correctly | 20     |
| Visualization with Plot              | 20     |
| Model Training                       | 20     |
| API Working with Prediction Output   | 20     |
| Organized Repo + Report              | 20     |
| **TOTAL**                            | **100**|

---
