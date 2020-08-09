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

        self.trainedData = load("./model/trained/face_embeddings.npz")
        self.loadLabelsEncoder()
        self.loadFaceNetModel()

        print(
            "* \n Model loaded. Start processing\n "
        )

    # load train model
    def loadFaceNetModel(self):
        #create model for embedding
        self.faceNetModel = load_model("./model/facenet_keras_hiroki_taniai.h5")


    def loadLabelsEncoder(self):
        nameList = self.trainedData["arr_1"]

        # convert label from string to integer
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(nameList)

    # fit face net model with trained data and labels(int)
    def fitFaceNetModel(self):
        # load trained data
        embedFaceList = self.trainedData["arr_0"]
        nameList = self.trainedData["arr_1"]

        # normalize embed face and name for model fitting
        norm = Normalizer(norm='l2') #L2 = least squares
        embedFaceList = norm.transform(embedFaceList)

        nameListInt = self.labelEncoder.transform(nameList)
        self.faceNetModel = SVC(kernel="linear", probability=True)
        self.faceNetModel.fit(embedFaceList, nameListInt)
