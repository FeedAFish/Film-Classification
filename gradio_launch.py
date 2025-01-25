import gradio as gr
import requests
from io import BytesIO


def predict_via_api(image):
    # Convert Gradio image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="jpeg")
    img_byte_arr = img_byte_arr.getvalue()

    # Create file-like object for API request
    files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    result = response.json()
    return f"Model Output: {result['Prediction']}"


# Create the Gradio interface
iface = gr.Interface(
    fn=predict_via_api,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="PyTorch Model Prediction API",
    description="Upload an image to get predictions via API.",
    examples=[["data/animation/158.jpg"]],
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
