import gradio as gr
import requests
from io import BytesIO
from PIL import Image
import base64
import os
from typing import Optional, List, Tuple

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def process_image(image: Image) -> bytes:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="jpeg")
    return img_byte_arr.getvalue()

def decode_base64_images(encoded_images: List[str]) -> List[Image.Image]:
    images = []
    for img_64 in encoded_images:
        img_data = base64.b64decode(img_64)
        img_buffer = BytesIO(img_data)
        img = Image.open(img_buffer)
        images.append(img)
    return images

def predict_via_api(image: Optional[Image.Image]) -> Tuple[str, Image.Image, Image.Image, Image.Image, Image.Image]:
    if image is None:
        return ("No image uploaded", None, None, None, None)

    # Convert and prepare image
    img_bytes = process_image(image)
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    try:
        # Get prediction
        pred_response = requests.post(f"{API_URL}/predict", files=files)
        pred_response.raise_for_status()
        prediction = pred_response.json()["Prediction"]

        # Get recommendations
        rec_response = requests.post(f"{API_URL}/recommend", files=files)
        rec_response.raise_for_status()
        recommended_64 = rec_response.json()["Recommendations"]
        recommended_images = decode_base64_images(recommended_64)

        return (f"Prediction: {prediction}", *recommended_images)

    except requests.exceptions.RequestException as e:
        return (f"Error: {str(e)}", None, None, None, None)


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="Film Classification System") as interface:
        # Header
        gr.Markdown("🎬 Film Classification System")
        gr.Markdown("Upload a movie poster or scene to get predictions and similar images")

        # Main layout
        with gr.Row():
            input_image = gr.Image(
                label="Input Image",
                type="pil",
            )

            with gr.Column():
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                prediction_text = gr.Text(
                    label="Prediction",
                    interactive=False
                )

                gr.Markdown("### Similar Images")
                with gr.Row():
                    similar_images = [
                        gr.Image(label=f"Recommended {i+1}", 
                                interactive=False,
                                show_label=True)
                        for i in range(4)
                    ]

        # Connect components
        analyze_btn.click(
            fn=predict_via_api,
            inputs=input_image,
            outputs=[prediction_text, *similar_images],
        )

    return interface

# Launch the app
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        show_api=False,
        server_port=7860
    )
