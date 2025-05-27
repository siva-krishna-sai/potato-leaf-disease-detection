from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (mobile access)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
MODEL = tf.keras.models.load_model("../training/models/EfficientNetB0.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

REMEDIES = {
    "Early Blight": """- Remove and destroy infected leaves.
- Apply fungicides like chlorothalonil or mancozeb.
- Practice crop rotation and avoid overhead watering.""",

    "Late Blight": """- Remove infected plants immediately.
- Use fungicides containing copper or metalaxyl.
- Ensure proper drainage and avoid water accumulation.""",

    "Healthy": "No disease detected. Keep monitoring and ensure proper crop care."
}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    remedy = REMEDIES.get(predicted_class, "No remedy available.")

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'remedy': remedy
    }

if __name__ == "__main__":
    # Host is set to 0.0.0.0 for network access
    uvicorn.run(app, host='0.0.0.0', port=8000)
