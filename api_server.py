from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn
from utils import model_train as train
import argparse

app = FastAPI()

# Load model on startup
parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path",
    nargs="?",
    default="model.mdl",
    help="Path to save model. Default = 'model.mdl'",
)
args = parser.parse_args()

try:
    model = train.SimpleNN.load_model(args.model_path)
    model.eval()
except:
    print("Model not found")
    exit()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    output = model.predict(image)
    return {"Prediction": model.classes[output]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
