import os
import numpy as np
# import tensorflow as tf
from PIL import Image as pilImage

from util.Logger import Logger
from model.ModelLoader import ModelLoader
from sklearn.preprocessing import LabelEncoder

from mtcnn.mtcnn import MTCNN

# service class


class FaceExtractService:
    def __init__(self, modelLoader, logger):
        self.logger = logger
        self.modelLoader = modelLoader
        self.imageExt = (".png", ".jpeg", ".jpg", ".JPG", ".gif")
        self.mtcnnDetector = MTCNN()

    # for training image service
    def getTrainEmbedFaceNameList(self, path):
        faceList = list()
        nameList = list()

        for folderName in os.listdir(path):
            fullPath = path + "/" + folderName + "/"

            if os.path.isdir(fullPath):
                self.logger.logDebug(f'Scanning Folder: {fullPath}')
                tmpMap = self.getEmbedFileNameFaceMap(
                    fullPath)  # fileName and face as map

                # convert map(fileName, face) to 2 separate lists
                for key in tmpMap:
                    for embedFace in tmpMap[key]:
                        faceList.append(embedFace)
                        nameList.append(folderName)

                self.logger.logDebug(
                    f"Person: ({folderName}), No of image loaded: {len(tmpMap.keys())}, No of faces extracted: {len(faceList)}")
            else:
                self.logger.logDebug(f'Skip - Not a folder: {fullPath}')
                continue

        self.logger.logDebug(f"Total faces loaded: {len(faceList)} ")

        return faceList, nameList

    # for detect image service
    def getEmbedFileNameFaceMap(self, path):
        counter = 0
        fileNameFaceMap = {}

        for fileName in os.listdir(path):
            if not fileName.endswith(self.imageExt):
                continue
            else:
                filePath = path + "/" + fileName
                self.logger.logDebug(f'Scanning File: {filePath}')

                counter += 1

                # extract faces
                faceList = self.extractFaceList(filePath)

                # skip image with no face detected
                if(len(faceList) == 0):
                    self.logger.logWarning(
                        f'********* WARNING ******* File skipped: {filePath}, Reason: No face found')
                    continue

                # convert faces to embedding faces
                embedFaceList = self.getFaceEmbeddingList(faceList)

                # use file name and faces as map to return
                fileNameFaceMap[fileName] = embedFaceList
                self.logger.logDebug(
                    f'******* fileNameFaceMap:{len(fileNameFaceMap)}, embedFaceList: {len(embedFaceList)}')

        return fileNameFaceMap

    def extractFaceList(self, filePath):
        face_size = (160, 160)
        faceList = list()

        img = pilImage.open(filePath).convert("RGB")
        imgArr = np.array(img)

        # use MTCNN to detect all faces
        faceArr = self.mtcnnDetector.detect_faces(imgArr)

        idx = 0
        for face in faceArr:
            x1, y1, width, height = face["box"]
            confidence = face["confidence"]
            keypoints = face["keypoints"]

            # prevent negative return x, y value
            x1, y1 = abs(x1), abs(y1)

            # get location of x2, y2 for face extraction
            x2, y2 = x1 + width, y1 + height

            # extract the face from image array
            targetFace = imgArr[y1:y2, x1:x2]

            # resize detected face, convert as array for output
            faceImg = pilImage.fromarray(targetFace)
            faceImg = faceImg.resize(face_size)

            # add to list for later return
            faceList.append(np.array(faceImg))

            # show extracted face
            # plot_face(faceImg)

            idx += 1
            self.logger.logDebug(f'face {idx}')
            self.logger.logDebug(
                f'x1 {x1}, y1 {y1}, width {width}, height {height}')
            self.logger.logDebug(f'confidence {confidence}')
            self.logger.logDebug(f'keypoints {keypoints}')

        self.logger.logDebug(
            f'Extract face: {filePath}, No. of face:{len(faceList)} , data shape:{np.asarray(faceList).shape}')

        return faceList

    # get the face embedding for one face
    def getFaceEmbeddingList(self, faceList):
        self.logger.logDebug(
            f'Before embed LIST shape:{np.asarray(faceList).shape}')

        tmpFaceList = list()
        for faceArr in faceList:
            self.logger.logDebug(
                f'Before embed shape:{np.asarray(faceArr).shape}')
            # convert face array to float for standardize
            faceArr = faceArr.astype("float32")

            # standardize value
            mean, std = faceArr.mean(), faceArr.std()
            faceArr = (faceArr - mean) / std

            # add 1 layer from 3-dimension array to 4-dimension array
            reshapeFaceArr = np.expand_dims(faceArr, axis=0)
            self.logger.logDebug(
                f'After standardize shape:{np.asarray(reshapeFaceArr).shape}')

            # Get embedding result of each face, become 1, 128 shape
            embedFace = self.modelLoader.faceNetModel.predict(
                reshapeFaceArr)

            # store shape 128 only, ignore 1 (orginal shape: 1, 128)
            tmpFaceList.append(embedFace[0])

        self.logger.logDebug(
            f'Embeding face: {len(tmpFaceList)} ,shape:{np.asarray(tmpFaceList).shape}\n')
        return tmpFaceList
