from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os
import uvicorn
from utils import model_train as train
import argparse
import base64
import torch
from utils import dataloader

data = dataloader.Dataset("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)  # Disable gradient computation
except:
    print("Model not found")
    exit()

data.load_kdtree("data.db")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    output = model.predict(image)
    return {"Prediction": model.classes[output]}


@app.post("/recommend")
async def recommend_image(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Get similar images using dataset methods
    similar_indices = data.get_similar_images_indices(image, model, k=4)
    similar_images = data.indices_to_images(similar_indices)

    # Convert tensor images to bytes
    image_64 = []
    for tensor_img in similar_images:
        with torch.no_grad():  # Disable gradient tracking
            tensor_new = tensor_img.mul(255).clamp(0, 255).byte()
            img_np = tensor_new.cpu().numpy()
            img_pil = Image.fromarray(img_np)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="jpeg", quality=85, optimize=True)
            image_64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return {"Recommendations": image_64}


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0" if os.getenv("DOCKER") else "127.0.0.1")
    uvicorn.run(app, host=HOST, port=PORT)
