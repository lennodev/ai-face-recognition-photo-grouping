import os
import shutil
import numpy as np
import tensorflow as tf
import sys
from PIL import Image as pilImage

from model.ModelLoader import ModelLoader
from service.FaceExtractService import FaceExtractService
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score


class TrainService:
    def __init__(self, modelLoader, faceExtractService):
        self.modelLoader = modelLoader
        self.faceExtractService = faceExtractService
        self.outputDataPath = './model/trained/'

    def run(self, path):
        print(f"<Read folder images>: {path}")

        shutil.rmtree(self.outputDataPath, ignore_errors=True)

        # read faces and names mappings
        fullPath = path + "/train"
        faceList, nameList = self.faceExtractService.getTrainEmbedFaceNameList(
            fullPath)

        # noramlize embed face
        print(
            f'Before normalization shape:{np.asarray(faceList).shape}, value[0]: {faceList[0][:5]}')
        self.normalizer = Normalizer(norm='l2')  # L2 = least squares
        faceList = self.normalizer.transform(faceList)
        print(
            f'After normalization shape:{np.asarray(faceList).shape}, value[0]: {faceList[0][:5]}')

        # save as ouptut model
        os.makedirs(self.outputDataPath, exist_ok=True)
        np.savez_compressed(self.outputDataPath + "face_embeddings.npz",
                            faceList, nameList)

        self.verify(path, faceList, nameList)

        print("----------End------------\n")

    def verify(self, path, tFaceList, tNameList):
        print("----------Start Verification----------")

        # convert person name as integer
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(tNameList)
        tNameIntList = self.labelEncoder.transform(tNameList)

        # prepare validation dataset
        fullPath = path + "/val"
        faceList, nameList = self.faceExtractService.getTrainEmbedFaceNameList(
            fullPath)

        # normalize face, map name to int
        faceList = self.normalizer.transform(faceList)
        nameIntList = self.labelEncoder.transform(nameList)

        # fit face and name to model for prediction
        self.modelLoader.fitFaceNetModel(tFaceList, tNameIntList)

        # prediction
        tPredictResult = self.modelLoader.faceNetModel.predict(tFaceList)
        vPredictResult = self.modelLoader.faceNetModel.predict(faceList)

        # score
        tScore = accuracy_score(tNameIntList, tPredictResult)
        vScore = accuracy_score(nameIntList, vPredictResult)

        # summarize
        print("----------Verification Result----------")
        print('Accuracy: Train=%.3f, Validation=%.3f' %
              (tScore*100, vScore*100))

        self.randomCheck(faceList, nameIntList, 5)

    def randomCheck(self, faceList, nameIntList, noOfSample):
        # random a sample for display
        # #### test model on a random example from the test dataset

        print(f"No of faces: {faceList.shape[0]}")
        for i in range(0, noOfSample):
            randomIdx = np.random.randint(faceList.shape[0])
            # randomIdx = 31
            randomFaceEmb = faceList[randomIdx]
            randomFaceNameInt = nameIntList[randomIdx]
            randomFaceNameStr = self.labelEncoder.inverse_transform(
                [randomFaceNameInt])

            # # prediction for the face
            currEmbFace = np.expand_dims(randomFaceEmb, axis=0)
            predictClass = self.modelLoader.faceNetModel.predict(currEmbFace)
            predictProb = self.modelLoader.faceNetModel.predict_proba(
                currEmbFace)

            # prepare result for display
            predictClassInt = predictClass[0]
            # get probability of predict item
            predictClassProb = predictProb[0, predictClassInt] * 100
            predictClassName = self.modelLoader.labelEncoder.inverse_transform(
                predictClass)

            print("Random Picture for verification:")
            # guess result
            if(randomFaceNameInt == predictClassInt):
                print('Bingo!')
            else:
                print('WRONG!!!!!')

            print(
                f'randomIdx: {randomIdx}, TargetFaceNameStr: {randomFaceNameStr}, TargetFaceNameInt: {randomFaceNameInt}')

            print(
                f'predictClass: {predictClassInt}, predictProb: {predictProb}, predictClassName: {predictClassName}, predictClassProb:{predictClassProb}')
            print('--------------------------------')
