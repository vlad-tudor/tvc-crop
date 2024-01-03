# app.py

from flask import Flask, request, send_file, render_template, jsonify
import base64
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename
import os
from image_utils.image_process import load_all_models, process_single_image
import traceback


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load all models once when the Flask app starts
load_all_models()

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/process-image', methods=['POST'])
def process_image_api():
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    tolerance = request.form.get('tolerance', default=10, type=int)

    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Process the image using the pre-loaded models
        try:
            cropped_path, basnet_path, deeplab_path = process_single_image(filepath, tolerance)
            # Read and send the images in the response
            with open(cropped_path, 'rb') as cropped, open(basnet_path, 'rb') as basnet, open(deeplab_path, 'rb') as deeplab:
                response = {
                    'croppedImage': base64.b64encode(cropped.read()).decode('utf-8'),
                    'basnetOutput': base64.b64encode(basnet.read()).decode('utf-8'),
                    'deeplabOutput': base64.b64encode(deeplab.read()).decode('utf-8')
                }

            # Delete the images
    
            os.remove(cropped_path)
            os.remove(basnet_path)
            os.remove(deeplab_path)
            os.remove(filepath)

            return jsonify(response)

        except Exception as e:
            traceback.print_exc()  # Print the stack trace
            return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)