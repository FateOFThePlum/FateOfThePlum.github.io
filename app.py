from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import threading
import detect
detect.Debug_mode = False
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the image data from the request
    data = request.get_json()

    # Perform any processing with the image data here
    image_data = data.get('image')

    # Convert base64 image data to a NumPy array
    img_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display the image in an OpenCV window
    cv2.imshow('Captured Image', img)
    detect.app_port(img)
    print('Object Width From Library', detect.measurement.width)
    cv2.destroyAllWindows()

    # Return a response, you can modify this based on your requirements
    return jsonify({'message': 'Image received successfully'})

if __name__ == '__main__':
    app.run(debug=True)


