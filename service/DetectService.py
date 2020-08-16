import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image as pilImage

from model.ModelLoader import ModelLoader
from service.FaceExtractService import FaceExtractService
from sklearn.preprocessing import LabelEncoder, Normalizer


class DetectService:
    def __init__(self, modelLoader, faceExtractService):
        self.modelLoader = modelLoader
        self.faceExtractService = faceExtractService
        self.predictThreshold = 80
        self.outputPath = "./output"
        self.normalizer = Normalizer(norm='l2')  # L2 = least squares

    def run(self, path):
        print(f"<Read folder images>: {path}")

        # clean up output folder
        shutil.rmtree(self.outputPath, ignore_errors=True)

        # read faces and names mappings
        fileNameFaceMap = self.faceExtractService.getEmbedFileNameFaceMap(path)
        # print(f">loaded fileNameFaceMap: {len(fileNameFaceMap)}")

        # prepare model for prediction
        self.modelLoader.fitFaceNetModelFromFile()

        # predict face
        # print(f">Check is match with file name")
        # self.guessIsMatchWithFileName(fileNameFaceMap)

        # print(f">Guess who is in the picture")
        self.guessWhoInFile(path, fileNameFaceMap)

        print("----------End------------\n")

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

                print(
                    f"Result: {resultStr}, Actual Person: {fileName}, Predict Person: {personName}, Proba: {str('{0:.2f}'.format(predictClassProb)) }%")
                # print("predictClass: %s" % predictClassInt)
                # print("predictProb: %s" % predictClassProbArr)
                # print("predictProb[0, predictClassIdx]: %s" % predictClassProb)
                # print("predictClassName: %s" % predictClassName)

            print(
                f"correctCnt: {correctCnt}, invalidCnt: {invalidCnt}, unknownCnt:{unknownCnt}")
            print("----------------------\n")

    def guessWhoInFile(self, path, fileNameFaceMap):
        # loop each file
        for key in fileNameFaceMap:
            fileName = key
            print(
                f"------------------- Processing file {fileName} -------------------")

            embedFaceList = fileNameFaceMap[key]
            print(f'embedFaceList shape:{np.asarray(embedFaceList).shape}')

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
                # print(
                #     f"Direct result: predictClassArr: {predictClassArr}, predictClassProbArr: {predictClassProbArr}")

                # prepare result for display
                predictClassInt = predictClassArr[0]
                predictClassProb = (
                    predictClassProbArr[0, predictClassInt]*100
                )  # get probability of predict item
                predictClassName = self.modelLoader.labelEncoder.inverse_transform(
                    predictClassArr)

                # print(
                #     f"predictClassInt: {predictClassInt}, predictClassName: {predictClassName}, predictClassProb: {predictClassProb}")

                if(predictClassProb < self.predictThreshold):
                    personName = "classify as Unknown - " + \
                        str(predictClassProbArr) + ", " + predictClassName[0]

                    targetFolder = "Unknown"
                else:
                    personName = predictClassName[0]
                    targetFolder = personName

                print(
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

            print(
                f"\nFile {fileName} copied to {len(folderList)} folder: {' | '.join(map(str, folderList))}\n")
