from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

#app = Flask(__name__)
app = Flask(__name__, static_folder='static')
app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)

class_names = ['organic', 'recyclable']
#class_names = [ 'recyclable','organic']

# Load the mbv2
model = load_model('mobilenet_model')

# Create the Folder for the 'uploads'
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def sharpen_and_adjust_brightness(img_array):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img_array, -1, kernel)
    brightness_factor = np.random.uniform(0.8, 1.2)
    adjusted = cv2.convertScaleAbs(sharpened, alpha=1, beta=brightness_factor * 25)
    return adjusted

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = sharpen_and_adjust_brightness(img_array.astype('uint8'))
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/',methods=['POST', 'GET'])
def home():
    return render_template('home.html')

@app.route('/workflow',methods=['POST', 'GET'])
def workflow():
    return render_template('workflow.html')

@app.route('/display',methods=['POST', 'GET'])
def display():
    return render_template('display.html')

# For the file type checking 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path, target_size=(160, 160))
        prediction = model.predict(processed_image)
        
        prediction_probabilities = tf.nn.softmax(prediction).numpy()
        
        class_idx = np.argmax(prediction_probabilities)
        class_name = class_names[class_idx]
        confidence = np.max(prediction_probabilities)
        #os.remove(file_path)

        image_url = os.path.join('/uploads', filename) 

        return render_template('prediction.html', class_name=class_name, confidence=float(confidence), image_url=image_url)
    else:
        return jsonify(error="Incorrect File format "), 400
    


if __name__ == '__main__':
    app.run(debug=False)