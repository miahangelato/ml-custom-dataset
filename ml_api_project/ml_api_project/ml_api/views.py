from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
from django.conf import settings

# Construct paths to your model and label encoder files inside the ml_api folder
model_path = os.path.join(settings.BASE_DIR, "ml_api", "model.pkl")
encoder_path = os.path.join(settings.BASE_DIR, "ml_api", "label_encoder.pkl")

# Load the model and label encoder once
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

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
