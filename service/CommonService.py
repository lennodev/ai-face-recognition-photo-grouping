import os
import shutil
import sys

from util.Logger import Logger


class CommonService:
    def __init__(self, logger, configLoader):
        self.configLoader = configLoader
        self.logger = logger

    def register(self, name):
        name = name.capitalize()
        self.logger.logDebug(f"<Register User>: {name}")

        self.userTrainPath = self.configLoader.getConfig(
            'path', 'userTrainPath')
        fullpath = "./" + self.userTrainPath+"/"+name

        result = self.createFolder(fullpath)

        if(result == True):
            self.logger.logInfo(
                f"User register successfully!")
        else:
            self.logger.logInfo(
                f"User already registered!")

        self.logger.logInfo(
            f"Copy {name}'s photos to {os.path.abspath(fullpath)}")

        self.logger.logDebug("----------End------------\n")

    def createFolder(self, fullpath):
        isFolderExist = False

        # detect if user folder exist
        if(os.path.exists(fullpath)):
            # if any file exist in folder under its name, program end
            if(len(os.listdir(fullpath)) < 0):
                return False
            else:
                isFolderExist = True

        # create folder for user if folder not exist
        if(not isFolderExist):
            os.makedirs(fullpath, exist_ok=True)

        return True
