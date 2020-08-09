
import os
import numpy as np
import werkzeug
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
from wtforms import (Form, TextField, validators, SubmitField, DecimalField, IntegerField)
from PIL import Image as pilImage

#create app
app = Flask(__name__,static_url_path='', static_folder='static/')
api = Api(app)

resultMap = {
    '0':'T-shirt/top',
    '1':'Trouser',
    '2':'Pullover',
    '3':'Dress',
    '4':'Coat',
    '5':'Sandal',
    '6':'Shirt',
    '7':'Sneaker',
    '8':'Bag',
    '9':'Ankle boot'
}


#load train model
def load_keras_model():
    """Load in the pre-trained model"""
    global trainModel
    global graph
    
    trainModel = load_model('./model/fashion_item_model.h5')
    graph = tf.compat.v1.get_default_graph()

    

#service class
class AiService():
    def __init__(self):
        pass

    def classify(self, imgFile):
        filename = imgFile.filename
        resizeFilename = 'resize_'+filename

        # print(f' \n\n ')

        print(f'<Save, convert and resize Image>:')

        #save and read file
        folderPath = './static/tmp/'
        imgFile.save(folderPath + filename)

        #convert image to single color layer
        img = pilImage.open(folderPath + filename).convert('L')
        print(f'Before: type: {type(img)}, im.size: {img.size}, im.mode: {img.mode}')

        #resize
        resizeImg = img.resize((28, 28))
        print(f'After:  type: {type(resizeImg)}, im.size: {resizeImg.size}, im.mode: {resizeImg.mode}')

        resizeImg.save('./static/tmp/'+resizeFilename)
        print('----------------------\n')

        #format input
        print(f'<Format input - Normalize + reshape>:')
        # imgArr = np.asarray(resizeImg / 255, dtype=float)
        
        imgArr = np.array(resizeImg)
        print(f'Before Normalize - Array: shape: {imgArr.shape} \n {imgArr[0]}')

        normImgArr = np.asarray(imgArr/255,dtype=float)
        print(f'After Normalize - Array: shape: {normImgArr.shape} \n {normImgArr[0]}')
        print('----------------------')

        print(f'Before reshape - Array: shape: {normImgArr.shape} ')
        normImgArr = normImgArr.reshape(1,28,28,1)
        print(f'After reshape - Array: shape: {normImgArr.shape} ')
        print('----------------------\n')

        #invoke model to predict
        print(f'<Invoke model>:')
        rawResult = trainModel.predict(normImgArr)
        result = str(np.argmax(rawResult))
        print(f'RAW Predict result: {result}')
        
        os.remove(folderPath + filename)
        os.remove(folderPath + resizeFilename)
        
        return result

#route class
class AiRoute(Resource):
    def __init__(self):
        self.aiService = AiService()

    def post(self):
        parser = reqparse.RequestParser()

        #add input para
        parser.add_argument('test')
        parser.add_argument('image',type=werkzeug.datastructures.FileStorage, location='files')

        #predict
        args = parser.parse_args()
        imgFile = args['image']
        predictResult = self.aiService.classify(imgFile)
        predictResultStr = resultMap[predictResult]

        #grab result from file name
        actualResult = str(imgFile.filename)
        actualResult = actualResult[0:1]
        actualResultStr = resultMap[actualResult]

        finalResult = 'INCORRECT!!!!! Sorry!'
        if(actualResult == predictResult):
            finalResult = 'Correct! Cheers~'
        result = {
            'Result:': finalResult,
            'Prediction':predictResultStr, 
            'Actual Result':actualResultStr
        }

        
        return result, 201

#routing list
api.add_resource(AiRoute, '/api/ai')

#constructor
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_keras_model()
    # Run app
    app.run(host="0.0.0.0", port=50000)