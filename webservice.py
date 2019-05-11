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
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
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
            print("Training KNN classifier...")
            classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf",n_neighbors=2)
            print("Training complete!")

            # STEP 2: Using the trained classifier, make predictions for unknown images
            #for image_file in os.listdir("knn_examples/test"):
            	#full_file_path = os.path.join("knn_examples/test", image_file)

            print("Looking for faces in {}".format(file.filename))
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            dir_path = os.path.dirname(os.path.realpath(__file__))
            string_in_string = "{}/trained_knn_model.clf".format(dir_path)
            predictions = predict(file, model_path=string_in_string)

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        # show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)

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


def detect_faces_in_image(file_stream):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    known_face_encoding = [-0.09634063, 0.12095481, -0.00436332, -0.07643753, 0.0080383,
                           0.01902981, -0.07184699, -0.09383309, 0.18518871, -0.09588896,
                           0.23951106, 0.0986533, -0.22114635, -0.1363683, 0.04405268,
                           0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                           0.03416885, -0.00267565, 0.09203379, 0.04713435, -0.12731361,
                           -0.35371891, -0.0503444, -0.17841317, -0.00310897, -0.09844551,
                           -0.06910533, -0.00503746, -0.18466514, -0.09851682, 0.02903969,
                           -0.02174894, 0.02261871, 0.0032102, 0.20312519, 0.02999607,
                           -0.11646006, 0.09432904, 0.02774341, 0.22102901, 0.26725179,
                           0.06896867, -0.00490024, -0.09441824, 0.11115381, -0.22592428,
                           0.06230862, 0.16559327, 0.06232892, 0.03458837, 0.09459756,
                           -0.18777156, 0.00654241, 0.08582542, -0.13578284, 0.0150229,
                           0.00670836, -0.08195844, -0.04346499, 0.03347827, 0.20310158,
                           0.09987706, -0.12370517, -0.06683611, 0.12704916, -0.02160804,
                           0.00984683, 0.00766284, -0.18980607, -0.19641446, -0.22800779,
                           0.09010898, 0.39178532, 0.18818057, -0.20875394, 0.03097027,
                           -0.21300618, 0.02532415, 0.07938635, 0.01000703, -0.07719778,
                           -0.12651891, -0.04318593, 0.06219772, 0.09163868, 0.05039065,
                           -0.04922386, 0.21839413, -0.02394437, 0.06173781, 0.0292527,
                           0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486,
                           0.01428208, -0.03637431, 0.03971229, 0.13983178, -0.23006812,
                           0.04999552, 0.0108454, -0.03970895, 0.02501768, 0.08157793,
                           -0.03224047, -0.04502571, 0.0556995, -0.24374914, 0.25514284,
                           0.24795187, 0.04060191, 0.17597422, 0.07966681, 0.01920104,
                           -0.01194376, -0.02300822, -0.17204897, -0.0596558, 0.05307484,
                           0.07417042, 0.07126575, 0.00209804]

    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    face_found = False
    is_obama = False

    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
        if match_results[0]:
            is_obama = True

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "is_picture_of_obama": is_obama
    }
    return jsonify(result)



def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
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
                y.append(class_dir)

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
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    #if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #    raise Exception("Invalid image path: {}".format(X_img_path))

    #if knn_clf is None and model_path is None:
    #    raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

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
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations,are_matches)]

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
