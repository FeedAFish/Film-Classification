import gradio as gr
import requests
from io import BytesIO
from PIL import Image
import base64
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_via_api(image):
    if image is None:
        return ["No image uploaded", None, None, None, None]
    # Convert Gradio image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="jpeg")
    img_byte_arr = img_byte_arr.getvalue()

    # Create file-like object for API request
    files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
    response = requests.post(f"{API_URL}/predict", files=files)
    result = response.json()
    class_result = result["Prediction"]
    # Get recommendation
    response = requests.post("{API_URL}/recommend", files=files)
    recommended_64 = response.json()["Recommendations"]
    recommended_images = []
    for img_64 in recommended_64:
        img_data = base64.b64decode(img_64)
        img_buffer = BytesIO(img_data)
        img = Image.open(img_buffer)
        recommended_images.append(img)
    return ["Prediction: " + class_result, *recommended_images]


with gr.Blocks() as iface:
    gr.Markdown("# Image Analysis and Recommendation System")
    gr.Markdown("Upload an image to get predictions and similar images")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")

        with gr.Column():
            prediction_text = gr.Text(label="Prediction")

            with gr.Row():
                similar_image1 = gr.Image(label="Recommended 1")
                similar_image2 = gr.Image(label="Recommended 2")
                similar_image3 = gr.Image(label="Recommended 3")
                similar_image4 = gr.Image(label="Recommended 4")

    input_image.change(
        fn=predict_via_api,
        inputs=input_image,
        outputs=[
            prediction_text,
            similar_image1,
            similar_image2,
            similar_image3,
            similar_image4,
        ],
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()
