from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
model = load_model('crime.h5')

# Define the list of crime categories
crime_categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents',  'Shooting' ,'Shoplifting','Robbery', 'Stealing', 'Vandalism']

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'  # Add this line for serving static files

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    # If the user does not select a file, submit an empty part without filename
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # If the file is allowed and properly uploaded
    if file and allowed_file(file.filename):
        # Make prediction
        prediction = make_prediction(file)

        return render_template('result.html', filename=file.filename, prediction=prediction)

    else:
        return render_template('index.html', error='File type not allowed')

def make_prediction(file):
    try:
        # Load image using io.BytesIO
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        predicted_class = crime_categories[np.argmax(pred)]
        return predicted_class
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error making prediction"

if __name__ == '__main__':
    app.run(debug=True)
