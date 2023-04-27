from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import cv2

app = Flask(__name__)

#CORS for Flask server, Please change the location of your frontend in line 15
CORS(app, resources={r"/*": {"origins": "http://localhost:3001"}})

hiraganaModel1 = load_model("assets/Models/hiraganamodel1.h5")
hiraganaModel2 = load_model("assets/Models/hiraganamodel2.h5")
hiraganaModel3 = load_model("assets/Models/hiraganamodel3.h5")
hiraganaModel4 = load_model("assets/Models/hiraganamodel4.h5")

Hiragana1=["あ", "い", "し", "た"]
Hiragana2=['え', 'お', 'そ', 'み']
Hiragana3=['う', 'く', 'す', 'ん']
Hiragana4=['か', 'き', 'ち', 'に']

katakanaModel1 = load_model("assets/Models/katakanamodel1.h5")
katakanaModel2 = load_model("assets/Models/katakanamodel2.h5")
katakanaModel3 = load_model("assets/Models/katakanamodel3.h5")
katakanaModel4 = load_model("assets/Models/katakanamodel4.h5")

Katakana1 = ["イ", "ス", "ソ", "ミ"]
Katakana2 = ["ウ", "ク", "ニ", "ン"]
Katakana3 = ["エ", "キ", "シ", "タ"]
Katakana4 = ["ア", "オ", "カ", "チ"]

@app.route('/')
def base():
    return "Hello world, this is flask"

# hiragana prediction
@app.route('/writing/verify/hiragana', methods=['POST'])
def hiraganaVerify():
    dataURL = request.json['dataURL']# retrieve the image data from the request
    ans = request.json['answer']
    # decoding  the image
    _, encoded = dataURL.split(',', 1)
    image_data = base64.b64decode(encoded)
    
    # load the image data into a numpy array
    img_np = np.array(Image.open(BytesIO(image_data)))

    # convert the numpy array to a BGR OpenCV image
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # perform image processing on the OpenCV image
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # Crop the image to focus on the character
    crop = gray[y:y+h, x:x+w]
    # Resize the cropped image to 72x72 pixels
    resized = cv2.resize(crop, (72, 72))
    # Convert the resized image to a 3-channel image
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # Normalize the pixel values to be between 0 and 1
    resized = resized / 255.0
    # Wrap the resulting image in a numpy array of shape (1, 72, 72, 3)
    image = np.expand_dims(resized, axis=0)
    # Save the preprocessed image to a file
    cv2.imwrite('assets/images/preprocessed_image.png', image[0] * 255)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test = test_datagen.flow_from_directory(
    'assets',
    target_size=(72, 72),
    shuffle= False)
  
    # preddicting models
    result = []
    for x  in range(0,4):
        model = hiraganaModel1
        label = []
        if(x == 0):
            model = hiraganaModel1
            label = Hiragana1
        if(x == 1):
            model = hiraganaModel2
            label = Hiragana2
        if(x == 2):
            model = hiraganaModel3
            label = Hiragana3
        if(x == 3):
            model = hiraganaModel4
            label = Hiragana4
        
        prediction = model.predict(test)
        print(prediction)
        index = np.argmax(prediction)
        character = label[index]
        predict_result = {character: prediction[0][index]}
        result.append(predict_result)

    predicted = ""
    isCorrect = False
    # Check if the character is a key in any of the dictionaries
    if any(ans in d for d in result):
      isCorrect = True
      predicted = ans
    # Find the dictionary with the highest value
    else:
      max_dict = max(result, key=lambda x: list(x.values())[0])
      predicted = list(max_dict.keys())[0]
      isCorrect = False

    response_data={"isCorrect": isCorrect, "output": predicted}
    return jsonify(response_data)

# hiragana prediction
@app.route('/writing/verify/katakana', methods=['POST'])
def katakanaVerify():
    dataURL = request.json['dataURL']# retrieve the image data from the request
    ans = request.json['answer']
    # decoding  the image
    _, encoded = dataURL.split(',', 1)
    image_data = base64.b64decode(encoded)
    
    # load the image data into a numpy array
    img_np = np.array(Image.open(BytesIO(image_data)))

    # convert the numpy array to a BGR OpenCV image
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # perform image processing on the OpenCV image
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # Crop the image to focus on the character
    crop = gray[y:y+h, x:x+w]
    # Resize the cropped image to 72x72 pixels
    resized = cv2.resize(crop, (72, 72))
    # Convert the resized image to a 3-channel image
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # Normalize the pixel values to be between 0 and 1
    resized = resized / 255.0
    # Wrap the resulting image in a numpy array of shape (1, 72, 72, 3)
    image = np.expand_dims(resized, axis=0)
    # Save the preprocessed image to a file
    cv2.imwrite('assets/images/preprocessed_image.png', image[0] * 255)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test = test_datagen.flow_from_directory(
    'assets',
    target_size=(72, 72),
    shuffle= False)
  
    # preddicting models
    result = []
    for x  in range(0,4):
        model = katakanaModel1
        label = []
        if(x == 0):
            model = katakanaModel1
            label = Katakana1
        if(x == 1):
            model = katakanaModel2
            label = Katakana2
        if(x == 2):
            model = katakanaModel3
            label = Katakana3
        if(x == 3):
            model = katakanaModel4
            label = Katakana4
        
        prediction = model.predict(test)
        print(prediction)
        index = np.argmax(prediction)
        character = label[index]
        predict_result = {character: prediction[0][index]}
        result.append(predict_result)

    predicted = ""
    isCorrect = False
    # Check if the character is a key in any of the dictionaries
    if any(ans in d for d in result):
      isCorrect = True
      predicted = ans
    # Find the dictionary with the highest value
    else:
      max_dict = max(result, key=lambda x: list(x.values())[0])
      predicted = list(max_dict.keys())[0]
      isCorrect = False

    response_data={"isCorrect": isCorrect, "output": predicted}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
