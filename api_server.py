from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn
from utils import model_train as train
import argparse
import base64

from utils import dataloader

data = dataloader.Dataset("data")
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
        tensor_new = tensor_img.mul(255).clamp(0, 255).byte()
        # Convert tensor to numpy array and transpose to correct format
        img_np = tensor_new.numpy()
        # Convert to PIL Image

        img_pil = Image.fromarray(img_np)
        # Convert to bytes
        buffer = io.BytesIO()
        img_pil.save(buffer, format="jpeg")
        image_64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return {"Recommendations": image_64}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
