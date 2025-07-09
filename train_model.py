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