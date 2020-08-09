import os
import numpy as np
import tensorflow as tf
from PIL import Image as pilImage

from model.ModelLoader import ModelLoader
from sklearn.preprocessing import LabelEncoder

from mtcnn.mtcnn import MTCNN

# service class
class AiService:
    def __init__(self, modelLoader):
        self.modelLoader = modelLoader
        self.imageExt = (".png", ".jpeg", ".jpg",".JPG", ".gif")
        self.mtcnnDetector = MTCNN()

    def run(self, path):
        print(f"<Read folder images>:")

        #read faces and names mappings
        embedFaceList, labelList, fileNameFaceMap = self.readFacesNamesFromFolder(path)
        print(f">loaded faceList: {len(embedFaceList)}, labelList: {len(labelList)}, fileNameFaceMap: {len(fileNameFaceMap)}")

        # prepare model for prediction
        self.modelLoader.fitFaceNetModel()

        # predict face
        # print(f">Check is match with file name")
        # self.guessIsMatchWithFileName(embedFaceList, labelList)


        # print(f">Guess who is in the picture")
        self.guessWhoInFile(fileNameFaceMap)

        print("----------End------------\n")

    def readFacesNamesFromFolder(self, path):
        embedFaceList = list()
        labelList = list()
        counter = 0
        fileNameFaceMap = {}

        for item in os.listdir(path):
            if item.endswith(self.imageExt):
                counter += 1

                # get face
                filePath = path + "/" + item
                faceList = self.extractFaceList(filePath)
                # print(f">{filePath} loaded face load: {len(faceList)}")

                tmpFaceList = list()
                for faceArr in faceList:
                    tmpFace = self.getFaceEmbedding(faceArr)
                    tmpFaceList.append(tmpFace)
                    
                embedFaceList.extend(tmpFaceList)
                labelList.append(item)

                fileNameFaceMap[item] = tmpFaceList
                # print(f">item: {item}, tmpFaceList: {tmpFaceList[0][:2]}, fileNameFaceMap[item]:{fileNameFaceMap[item][0][:2]} \n")
                # print(f">counter: {counter}, tmpFaceList: {len(tmpFaceList)}, embedFaceList: {len(embedFaceList)}, labelList: {len(labelList)}")
        
        return embedFaceList, labelList, fileNameFaceMap

    def extractFaceList(self, filePath):
        face_size = (160, 160)

        img = pilImage.open(filePath).convert("RGB")
        imgArr = np.array(img)
        faceArr = self.mtcnnDetector.detect_faces(imgArr)
        # initialize two dimension arrays
        # faceArray = [[0 for x in range(1)] for y in range(len(faceList))]

        faceList = list()
        #     print(f'{filePath} No. of faces detected: {len(faceArr)}')

        # show image with rectangle
        #     plot_image(filePath, faceArr)

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
        #         print(f'face {idx}')
        #         print(f'x1 {x1}, y1 {y1}, width {width}, height {height}')
        #         print(f'confidence {confidence}')
        #         print(f'keypoints {keypoints}')

        #     print('>loaded %d faces for image: %s' % (len(faceList), filePath))
        #     print(f'[{filePath}] No. of faceList extracted: {len(faceList)}')

        # print(f"Face shape:{np.asarray(faceList).shape}, value[0]: {faceList[0]}")
        return faceList

    # get the face embedding for one face
    def getFaceEmbedding(self, faceArr):
        # print(f'Before standardize shape:{np.asarray(faceArr).shape}, value[0]: {faceArr[0][:2]}')

        # convert face array to float for standardize
        faceArr = faceArr.astype("float32")

        # standardize value
        mean, std = faceArr.mean(), faceArr.std()
        faceArr = (faceArr - mean) / std

        # convert face frmo one-dimension array to two-dimension array
        reshapeFaceArr = np.expand_dims(faceArr, axis=0)
        # print(f'After standardize shape:{np.asarray(reshapeFaceArr).shape}, value[0]: {reshapeFaceArr[0][0][:2]}')

        # Get embedding result of each face
        embedFace = self.modelLoader.faceNetModel.predict(reshapeFaceArr)
        # print(f'Embedding face shape:{np.asarray(embedFace).shape}, value[0]: {embedFace[0][:2]}')
        # print(f'--------------------------')
        return embedFace[0]

    def guessIsMatchWithFileName(self, embedFaceList, labelList):
        labelEncoder = self.modelLoader.labelEncoder
        for idx in range(0, len(embedFaceList)):
            embedFace = embedFaceList[idx]
            # prediction for the face
            currEmbFace = np.expand_dims(embedFace, axis=0)
            predictClassArr = self.modelLoader.faceNetModel.predict(currEmbFace)
            predictClassProbArr = self.modelLoader.faceNetModel.predict_proba(
                currEmbFace
            )

            # prepare result for display
            predictClassInt = predictClassArr[0]
            predictClassProb = (
                predictClassProbArr[0, predictClassInt] * 100
            )  # get probability of predict item
            predictClassName = labelEncoder.inverse_transform(predictClassArr)

            if(labelList[idx].find(predictClassName[0]) >= 0):
                resultStr = "Correct!"
            else:
                resultStr = "WRONG!!!!!!!!"

            print(f"Result: {resultStr}")
            print(f"Actual Person: {labelList[idx]}, Predict Person: {predictClassName[0]}, Proba: {predictClassProb}%")
            # print("predictClass: %s" % predictClassInt)
            # print("predictProb: %s" % predictClassProbArr)
            # print("predictProb[0, predictClassIdx]: %s" % predictClassProb)
            # print("predictClassName: %s" % predictClassName)
            print("----------------------\n")

            idx += 1

    def guessWhoInFile(self, fileNameFaceMap):
        labelEncoder = self.modelLoader.labelEncoder

        #loop each file
        for key in fileNameFaceMap:
            fileName = key
            embedFaceList = fileNameFaceMap[key]

            #each face in file
            personList = list()
            for embedFace in embedFaceList:
                # prediction for the face
                currFace = np.expand_dims(embedFace, axis=0)
                predictClassArr = self.modelLoader.faceNetModel.predict(currFace)
                predictClassProbArr = self.modelLoader.faceNetModel.predict_proba(currFace)

                # prepare result for display
                predictClassInt = predictClassArr[0]
                predictClassProb = (
                    predictClassProbArr[0, predictClassInt] * 100
                )  # get probability of predict item
                predictClassName = labelEncoder.inverse_transform(predictClassArr)
                personList.append(predictClassName[0]+ " - "+ str(predictClassProb) + "%")
                
            print(f"File: {fileName}, \nPredict Person: {(', '.join(map(str, personList)))}\n")
        