# app.py

from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import os
from image_process import load_all_models, process_single_image

app = Flask(__name__)

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
            cropped_image = process_single_image(filepath, tolerance)
        except Exception as e:
            return str(e), 500

        output_path = os.path.join('results', filename)
        cropped_image.save(output_path)

        return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)