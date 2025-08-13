import base64
import io
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, Response

# --- Flask App Configuration ---
app = Flask(__name__, static_folder=".")
model = None

# --- Model Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['COPY / INVALID ❌', 'GENUINE ✅']

def load_model():
    """Load the trained Keras model."""
    global model
    try:
        # Construct the path to the model relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "results", "true_qr_classifier.keras")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found at: {model_path}")
            return

        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Loaded Keras model from {model_path}")
    except Exception as e:
        print(f"[ERROR] Could not load Keras model: {e}")

@app.before_request
def log_request_info():
    """Log incoming request paths."""
    if request.path != '/favicon.ico':
        print(f"[REQUEST] Path: {request.path}, Method: {request.method}")

@app.route("/")
def index():
    """Serve the main index.html page."""
    return send_from_directory(".", "index.html")

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    return Response(status=204)

@app.route("/verify", methods=['POST'])
def verify_image():
    """Receive an image frame, run inference, and return the result."""
    if model is None:
        return jsonify({"result": "Error: Model not loaded.", "detail": {}}), 500

    try:
        # --- 1. Decode Image ---
        data = request.get_json()
        image_data = data['image']
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image_np = np.frombuffer(binary_data, dtype=np.uint8)
        
        # Decode image in color for the model
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"result": "Error: Image decoding failed.", "detail": {}}), 400

        # --- 2. Preprocess Image ---
        # Convert from BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model's expected input size
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to array and create a batch
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        # --- 3. Run Inference ---
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        # --- 4. Format Response ---
        result_text = CLASS_NAMES[1] if score > 0.9 else CLASS_NAMES[0]
        
        detail = {
            "score": score,
            "prediction": result_text,
            "is_genuine": bool(score > 0.9)
        }
        
        return jsonify({"result": result_text, "detail": detail})

    except Exception as e:
        print(f"[ERROR] An exception occurred in /verify: {e}")
        return jsonify({"result": "Server Error", "detail": str(e)}), 500

@app.route("/<path:path>")
def serve_file(path):
    """Serve other static files like CSS and JS."""
    return send_from_directory(".", path)

if __name__ == "__main__":
    load_model()
    # The 'debug=True' parameter reloads the server on code changes,
    # but it can cause issues with loading the model twice.
    # For production or stable testing, it's better to run without debug mode
    # or use app.run(host="0.0.0.0", port=8000, debug=False)
    app.run(host="0.0.0.0", port=8000, debug=True)