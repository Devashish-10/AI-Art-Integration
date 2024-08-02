from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib as mpl

app = Flask(__name__)

# Setting default plot parameters
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# Define paths for style prediction and style transform models
style_predict_model_path = "static/style_predict_model.tflite"
style_transform_model_path = "static/style_transform_model.tflite"

# Function to load an image from a file and add a batch dimension
def load_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess an image by resizing and central cropping it
def preprocess_image(image, target_dim):
    shape = np.array(image.shape[1:-1], dtype=np.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = (shape * scale).astype(int)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image

# Load TFLite models
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

style_predict_interpreter = load_model(style_predict_model_path)
style_transform_interpreter = load_model(style_transform_model_path)

def predict_style(image):
    input_details = style_predict_interpreter.get_input_details()
    output_details = style_predict_interpreter.get_output_details()
    image = preprocess_image(image, 256)  # Assuming the model expects 256x256
    style_predict_interpreter.set_tensor(input_details[0]['index'], image)
    style_predict_interpreter.invoke()
    style_bottleneck = style_predict_interpreter.get_tensor(output_details[0]['index'])
    return style_bottleneck

def transform_style(style_bottleneck, image):
    input_details = style_transform_interpreter.get_input_details()
    output_details = style_transform_interpreter.get_output_details()
    image = preprocess_image(image, 384)  # Assuming the model expects 384x384
    style_transform_interpreter.set_tensor(input_details[0]['index'], image)
    style_transform_interpreter.set_tensor(input_details[1]['index'], style_bottleneck)
    style_transform_interpreter.invoke()
    stylized_image = style_transform_interpreter.get_tensor(output_details[0]['index'])
    return stylized_image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content_image_file = request.files['content_image']
        style_image_file = request.files['style_image']
        
        if content_image_file and style_image_file:
            content_image = load_image(content_image_file.read())
            style_image = load_image(style_image_file.read())

            style_bottleneck = predict_style(style_image)
            stylized_image = transform_style(style_bottleneck, content_image)

            # Convert the result to an image for display
            stylized_image = np.squeeze(stylized_image, axis=0)  # Remove batch dimension
            stylized_image = (stylized_image * 255.0).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(stylized_image).save(buffer, format="JPEG")
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return render_template("result.html", image_data=image_data)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
