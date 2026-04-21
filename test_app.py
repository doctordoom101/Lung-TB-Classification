import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='./model/mobilenetv2/mobilenetv2_tuberculosis_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define image dimensions (must match training dimensions)
IMG_HEIGHT = 224
IMG_WIDTH = 224

def predict_image(image_input):
    # Preprocess the image
    img = Image.fromarray(image_input) # Gradio passes image as numpy array
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype(np.float32)

    # Convert to RGB if it's grayscale (TFLite model expects 3 channels)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4: # RGBA to RGB
        img_array = img_array[:, :, :3]

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    tuberculosis_prob = output_data[0][0] # Assuming single output for binary classification

    # Calculate Normal probability
    normal_prob = 1 - tuberculosis_prob

    # Format output
    result = {
        "Normal": normal_prob,
        "Tuberculosis": tuberculosis_prob
    }

    return result

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray Image"),
    outputs=gr.Label(num_top_classes=2),
    title="Tuberculosis Detection from Chest X-ray",
    description="Upload a chest X-ray image to get a prediction for Normal or Tuberculosis."
)

# Launch the interface
iface.launch(debug=True)