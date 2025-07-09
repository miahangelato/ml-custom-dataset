import joblib

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

sample = [[7, 3, 50]]  # update with 3 features matching your dataset
prediction = model.predict(sample)
print("Prediction:", le.inverse_transform(prediction)[0])
