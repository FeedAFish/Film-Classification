import gradio as gr
import argparse
from utils import model_train as train

parser = argparse.ArgumentParser(
    prog="Image Classification",
    description="Gradio Interface for PyTorch Model Prediction",
    epilog="---",
)
parser.add_argument(
    "model_path",
    nargs="?",
    default="model.mdl",
    help="Path to save model. Default = 'model.mdl'",
)
args = parser.parse_args()

model = train.SimpleNN.load_model(args.model_path)
model.eval()  # Set the model to evaluation mode


# Define the prediction function
def predict_pil_image(image):
    output2 = model.predict(image)
    # Process the output (adjust depending on your model's task)
    return f"Model Output: {model.classes[output2]}"


# Create the Gradio interface
iface = gr.Interface(
    fn=predict_pil_image,  # Prediction function
    inputs=gr.Image(type="pil"),  # Input accepts PIL images
    outputs=gr.Textbox(label="Prediction"),  # Display the result as text
    title="PyTorch Model Prediction",
    description="Upload an image to see predictions made by the PyTorch model.",
    examples=[["data/animation/158.jpg"]],  # Replace with paths to example images
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
