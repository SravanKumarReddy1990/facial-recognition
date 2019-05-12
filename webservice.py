# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import math
from sklearn import neighbors
import os
import json
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
             return detect_faces_in_image(file,"knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream,train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    s1=[]
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

	
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_encodings(image)
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
            knn_clf.fit(X, y)
            # Save the trained KNN classifier
            if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f) 

            # Load the uploaded image file
            #img = face_recognition.load_image_file(file_stream)
            # Get face encodings for any faces in the uploaded image
            #unknown_face_encodings = face_recognition.face_encodings(img)
            #if knn_clf is None:
            #    with open(model_save_path, 'rb') as f:
            #       knn_clf = pickle.load(f)

            # Load image file and find face locations
            X_img = face_recognition.load_image_file(file_stream)
            X_face_locations = face_recognition.face_locations(X_img)

            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []

            # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            pridictions= [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

            #match_results = face_recognition.compare_faces([face_bounding_boxes], unknown_face_encodings[0])
            #print(match_results[0])
            #if (match_results[0]==True).all():
                #is_obama = True
                # Return the result as json
                result = {
                    #"face_found_in_image": face_found,
                    "is_picture_of": "- Found {} at ({}, {})".format(name, left, top)
                }
                #users = json.dumps(result)
                s1.append(result)
	
    return jsonify(s1)

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
