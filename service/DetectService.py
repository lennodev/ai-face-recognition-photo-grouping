import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image as pilImage

from model.ModelLoader import ModelLoader
from service.FaceExtractService import FaceExtractService
from sklearn.preprocessing import LabelEncoder, Normalizer


class DetectService:
    def __init__(self, modelLoader, faceExtractService, logger, configLoader):
        self.configLoader = configLoader
        self.modelLoader = modelLoader
        self.logger = logger
        self.faceExtractService = faceExtractService
        self.predictThreshold = 80
        self.outputPath = self.configLoader.getConfig("path", "outputPath")
        self.normalizer = Normalizer(norm='l2')  # L2 = least squares

    def run(self, path):
        self.logger.logInfo(
            f"Start grouping process now. Please wait.....")
        self.logger.logDebug(f"<Read folder images>: {path}")

        # clean up output folder
        shutil.rmtree(self.outputPath, ignore_errors=True)

        # read faces and names mappings
        fileNameFaceMap = self.faceExtractService.getEmbedFileNameFaceMap(path)
        self.logger.logDebug(
            f">loaded fileNameFaceMap: {len(fileNameFaceMap)}")

        # prepare model for prediction
        self.modelLoader.fitFaceNetModelFromFile()

        # predict face
        # self.logger.logDebug(f">Check is match with file name")
        # self.guessIsMatchWithFileName(fileNameFaceMap)

        self.logger.logDebug(f">Guess who is in the picture")
        self.guessWhoInFile(path, fileNameFaceMap)

        self.logger.logDebug("----------End------------\n")

        self.logger.logInfo(
            f"Photo grouping completed. View photo in /output folder")

    def guessIsMatchWithFileName(self, fileNameFaceMap):
        labelEncoder = self.modelLoader.labelEncoder
        correctCnt = 0
        invalidCnt = 0
        unknownCnt = 0

        for key in fileNameFaceMap:
            fileName = key
            embedFaceList = fileNameFaceMap[key]

            # each face in file
            for embedFace in embedFaceList:
                # prediction for the face
                currEmbFace = np.expand_dims(embedFace, axis=0)
                predictClassArr = self.modelLoader.faceNetModel.predict(
                    currEmbFace)
                predictClassProbArr = self.modelLoader.faceNetModel.predict_proba(
                    currEmbFace
                )

                # prepare result for display
                predictClassInt = predictClassArr[0]
                predictClassProb = (
                    predictClassProbArr[0, predictClassInt] * 100
                )  # get probability of predict item
                predictClassName = labelEncoder.inverse_transform(
                    predictClassArr)

                if(predictClassProb < self.predictThreshold):
                    personName = "Accuracy < " + \
                        str(self.predictThreshold) + "%, classify as Unknown"
                else:
                    personName = predictClassName[0]

                if(fileName.find(personName) >= 0):
                    resultStr = "Correct!"
                    correctCnt = correctCnt+1
                else:
                    if(personName.find(", classify as Unknown") >= 0):
                        resultStr = "Unkonwn!!!!!"
                        unknownCnt = unknownCnt+1
                    else:
                        resultStr = "WRONG!!!!!!!!"
                        invalidCnt = invalidCnt+1

                self.logger.logDebug(
                    f"Result: {resultStr}, Actual Person: {fileName}, Predict Person: {personName}, Proba: {str('{0:.2f}'.format(predictClassProb)) }%")
                self.logger.logDebug("predictClass: %s" % predictClassInt)
                self.logger.logDebug("predictProb: %s" % predictClassProbArr)
                self.logger.logDebug(
                    "predictProb[0, predictClassIdx]: %s" % predictClassProb)
                self.logger.logDebug("predictClassName: %s" % predictClassName)

            self.logger.logDebug(
                f"correctCnt: {correctCnt}, invalidCnt: {invalidCnt}, unknownCnt:{unknownCnt}")
            self.logger.logDebug("----------------------\n")

    def guessWhoInFile(self, path, fileNameFaceMap):
        # loop each file
        for key in fileNameFaceMap:
            fileName = key
            self.logger.logDebug(
                f"------------------- Processing file {fileName} -------------------")

            embedFaceList = fileNameFaceMap[key]
            self.logger.logDebug(
                f'embedFaceList shape:{np.asarray(embedFaceList).shape}')

            # normalize list
            embedFaceList = self.normalizer.transform(embedFaceList)

            # each face in file
            folderList = list()

            for embedFace in embedFaceList:
                # prediction for the face
                currFace = np.expand_dims(embedFace, axis=0)
                predictClassArr = self.modelLoader.faceNetModel.predict(
                    currFace)
                predictClassProbArr = self.modelLoader.faceNetModel.predict_proba(
                    currFace)
                self.logger.logDebug(
                    f"Direct result: predictClassArr: {predictClassArr}, predictClassProbArr: {predictClassProbArr}")

                # prepare result for display
                predictClassInt = predictClassArr[0]
                predictClassProb = (
                    predictClassProbArr[0, predictClassInt]*100
                )  # get probability of predict item
                predictClassName = self.modelLoader.labelEncoder.inverse_transform(
                    predictClassArr)

                self.logger.logDebug(
                    f"predictClassInt: {predictClassInt}, predictClassName: {predictClassName}, predictClassProb: {predictClassProb}")

                if(predictClassProb < self.predictThreshold):
                    personName = "classify as Unknown - " + \
                        str(predictClassProbArr) + ", " + predictClassName[0]

                    targetFolder = "Unknown"
                else:
                    personName = predictClassName[0]
                    targetFolder = personName

                self.logger.logDebug(
                    f"Guess: {personName} @ {str('{0:.2f}'.format(predictClassProb))}")

                # prepare for copy to target folder
                if(not targetFolder in folderList):
                    folderList.append(targetFolder)

            # copy image to output folder
            for folder in folderList:
                srcPath = path+"/"+fileName
                targetPath = self.outputPath+"/"+folder
                os.makedirs(targetPath, exist_ok=True)
                shutil.copy(srcPath, targetPath)

            self.logger.logDebug(
                f"File {fileName} copied to {len(folderList)} folder: {' | '.join(map(str, folderList))}\n")
