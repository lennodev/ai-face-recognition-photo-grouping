import os
import shutil
import numpy as np
import tensorflow as tf
import sys
from PIL import Image as pilImage
import math

from model.ModelLoader import ModelLoader
from service.FaceExtractService import FaceExtractService
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score


class TrainService:
    def __init__(self, modelLoader, faceExtractService, logger, configLoader):
        self.configLoader = configLoader
        self.modelLoader = modelLoader
        self.logger = logger
        self.faceExtractService = faceExtractService

        self.userTrainPath = self.configLoader.getConfig(
            "path", "userTrainPath")
        self.userValPath = self.configLoader.getConfig("path", "userValPath")
        self.valThreshold = 0.3  # 30% image use as validation set
        self.imageExt = (".png", ".jpeg", ".jpg", ".JPG", ".gif")

    def run(self):
        self.logger.logInfo(
            f"Start training now. Please wait.....")
        embedPath = self.configLoader.getConfig("path", "embeddingPath")
        embedName = self.configLoader.getConfig("path", "embeddingConfigName")

        self.logger.logDebug(f"<Read folder images>: {self.userTrainPath}")
        shutil.rmtree(embedPath, ignore_errors=True)

        self.splitTrainTestData()

        # read faces and names mappings
        fullPath = self.userTrainPath
        faceList, nameList = self.faceExtractService.getTrainEmbedFaceNameList(
            fullPath)

        # noramlize embed face
        self.logger.logDebug(
            f'Before normalization shape:{np.asarray(faceList).shape}, value[0]: {faceList[0][:5]}')
        self.normalizer = Normalizer(norm='l2')  # L2 = least squares
        faceList = self.normalizer.transform(faceList)
        self.logger.logDebug(
            f'After normalization shape:{np.asarray(faceList).shape}, value[0]: {faceList[0][:5]}')

        # save as ouptut model
        os.makedirs(embedPath, exist_ok=True)
        np.savez_compressed(embedPath + "/" + embedName, faceList, nameList)

        self.verify(faceList, nameList)

        self.logger.logDebug("----------End------------\n")
        self.logger.logInfo(
            f"Training completed. You can group your photo now")

    def verify(self, tFaceList, tNameList):
        self.logger.logDebug("----------Start Verification----------")

        # convert person name as integer
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(tNameList)
        tNameIntList = self.labelEncoder.transform(tNameList)

        # prepare validation dataset
        fullPath = self.userValPath
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
        self.logger.logDebug("----------Verification Result----------")
        self.logger.logDebug('Accuracy: Train=%.3f, Validation=%.3f' %
                             (tScore*100, vScore*100))

        self.randomCheck(faceList, nameIntList, 5)

    def randomCheck(self, faceList, nameIntList, noOfSample):
        # random a sample for display
        # #### test model on a random example from the test dataset

        self.logger.logDebug(f"No of faces: {faceList.shape[0]}")
        for i in range(0, noOfSample):
            randomIdx = np.random.randint(faceList.shape[0])
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

            self.logger.logDebug("Random Picture for verification:")
            # guess result
            if(randomFaceNameInt == predictClassInt):
                self.logger.logDebug('Bingo!')
            else:
                self.logger.logDebug('WRONG!!!!!')

            self.logger.logDebug(
                f'randomIdx: {randomIdx}, TargetFaceNameStr: {randomFaceNameStr}, TargetFaceNameInt: {randomFaceNameInt}')

            self.logger.logDebug(
                f'predictClass: {predictClassInt}, predictProb: {predictProb}, predictClassName: {predictClassName}, predictClassProb:{predictClassProb}')
            self.logger.logDebug('--------------------------------')

    def splitTrainTestData(self):
        # check train folder

        trainPath = self.userTrainPath
        if(os.path.exists(trainPath)):
            # check any user folder exist
            if(len(os.listdir(trainPath)) > 0):
                for folderName in os.listdir(trainPath):
                    fullPath = trainPath + "/" + folderName
                    isCopyRequired = False

                    # found user
                    if os.path.isdir(fullPath):
                        # check user exist in validation folder
                        valPath = self.userValPath+"/"+folderName
                        if(os.path.exists(valPath)):
                            # check if any image exist
                            if(len(os.listdir(valPath)) == 0):
                                isCopyRequired = True
                        else:
                            # create validation folder
                            os.makedirs(valPath, exist_ok=True)
                            isCopyRequired = True

                        # copy image from train to val
                        if(isCopyRequired):
                            imgCnt = len(os.listdir(fullPath))
                            # calculate no. of image copy to validation
                            valCnt = math.ceil(imgCnt * self.valThreshold)
                            self.logger.logDebug(
                                f"valCnt: {valCnt}, imgCnt:{imgCnt}, self.valThreshold:{self.valThreshold}")
                            copyCnt = 0
                            for fileName in os.listdir(fullPath):
                                if copyCnt >= valCnt:
                                    break

                                if fileName.endswith(self.imageExt):
                                    filePath = fullPath + "/" + fileName
                                    valFilePath = valPath + "/" + fileName
                                    shutil.move(filePath, valFilePath)
                                    copyCnt = copyCnt + 1

            else:
                self.logger.logWarning(
                    f"No user found! Please register user first")
        else:
            self.logger.logWarning(
                "No user found! Please register a user first")
