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
from flask import Flask, jsonify, request, redirect, render_template, url_for
import math
from sklearn import neighbors
import os
import io
import json
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


#app = Flask(__name__)
app = Flask(__name__, template_folder='templates',static_url_path = "", static_folder = 'knn_examples/train/sravan')
PEOPLE_FOLDER = os.path.join('knn_examples', 'train','sravan')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

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
             #open("knn_examples/test/sample.jpg", 'wb')
             with open("knn_examples/test/sample.jpg", 'wb') as f:
                 image = Image.open(io.BytesIO(file.read()))
                 image.save(f)

             print("Training KNN classifier...")
             classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
             print("Training complete!")
             s1=[]
             # STEP 2: Using the trained classifier, make predictions for unknown images
             for image_file in os.listdir("knn_examples/test"):
                 full_file_path = os.path.join("knn_examples/test", image_file)

                 print("Looking for faces in {}".format(image_file))

                 # Find all people in the image using a trained classifier model
                 # Note: You can pass in either a classifier file name or a classifier model instance
                 predictions = predict(full_file_path, model_path="trained_knn_model.clf")

                 # Print results on the console
                 
                 for name, (top, right, bottom, left) in predictions:
                     print("- Found {} at ({}, {})".format(name, left, top))
                     result = {
                       "face_found_in_image": name,
                       "is_picture_of": left,
                       "top":top,
                       "right":right,
                       "bottom":bottom
                     }
                     #users = json.dumps(result)
                     s1.append(result)
             for item in s1: 
                 full_filename=item["face_found_in_image"]
                 return render_template("index.html", user_image = full_filename.split("/")[-1])
             #return detect_faces_in_image(file,"knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)

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

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(img_path)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
  
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def detect_faces_in_image(file_stream,train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    s1=[]
    
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

	
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_encodings(image)

            # Load the uploaded image file
            img = face_recognition.load_image_file(file_stream)
            # Get face encodings for any faces in the uploaded image
            unknown_face_encodings = face_recognition.face_encodings(img)

            face_found = False
            is_obama = False

            #if len(unknown_face_encodings) > 0:
            face_found = True
            # See if the first face in the uploaded image matches the known face of Obama
            match_results = face_recognition.compare_faces([face_bounding_boxes], unknown_face_encodings[0])
            #print(match_results[0])
            if (match_results[0]==True).all():
                is_obama = True
                # Return the result as json
                result = {
                    "face_found_in_image": face_found,
                    "is_picture_of": img_path
                }
                #users = json.dumps(result)
                s1.append(result)
	
    return jsonify(s1)

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
