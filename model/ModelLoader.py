import tensorflow as tf
from numpy import load
from sklearn.svm import SVC

# from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, Normalizer

# service class


class ModelLoader:
    def __init__(self):
        print(
            "* \n Loading Keras model...\n please wait until server has fully started "
        )
        self.loadFaceNetModel()
        print(
            "* \n Model loaded. Start processing\n "
        )

    # load train model
    def loadFaceNetModel(self):
        # create model for embedding
        self.faceNetModel = load_model(
            "./model/facenet_keras_hiroki_taniai.h5")

    # fit face net model with trained data and labels(int)

    def fitFaceNetModelFromFile(self):
        # load trained data
        self.trainedData = load("./model/trained/face_embeddings.npz")
        embedFaceList = self.trainedData["arr_0"]
        nameList = self.trainedData["arr_1"]

        self.fitFaceNetModel(embedFaceList, nameList)

    def fitFaceNetModel(self, embedFaceList, nameList):
        # convert label from string to integer for later retrieval
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(nameList)
        nameIntList = self.labelEncoder.transform(nameList)

        # fit model with embedded face and name in integer
        self.faceNetModel = SVC(kernel="linear", probability=True)
        self.faceNetModel.fit(embedFaceList, nameIntList)
