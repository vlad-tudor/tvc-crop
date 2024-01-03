# TVC-CROP

A Flask API to crop the subject of a photo. Uses DeepLabV3 and BasNet.

### Instructions

1. `git clone` **the repository**

2. `python -m venv venv` **to create a virtual environment**

3. `source venv/bin/activate` On Unix or MacOS (`\venv\Scripts\activate` on Windows)

4. `pip install -r requirements.txt` **to install required packages**

5. `python load_models.py` **downloads all models**

6. `python app.py` **starts the server**

<br>

### TODOs

- More expansive/dynamic model loading/usage
- Expand API params to include target resolutions
- Return bounding boxes only instead of parsed images
- Image background generation
