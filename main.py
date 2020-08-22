from loader import Service, Model
import sys

import logging
import os


def run(args):
    mode = str(args[1])

    commonService = Service.commonService()

    if(mode == "Register"):
        if(len(args) != 3):
            commonService.logger.logWarning(
                f"Invalid option. Correct: Register <Username>")
            sys.exit()

        commonService.register(str(args[2]))
    elif(mode == "Train"):
        if(len(args) != 2):
            commonService.logger.logWarning(f"Invalid option. Correct: Train")
            sys.exit()

        trainService = Service.trainService()
        trainService.run()
    elif(mode == "Group"):
        if(len(args) != 3):
            commonService.logger.logWarning(
                f"Invalid option. Correct: Group <Source_Folder_Path>")
            sys.exit()

        detectService = Service.detectService()
        detectService.run(str(args[2]))
    else:
        commonService.logger.logWarning(
            f"Invalid Mode: {mode}. Please select mode: [Register|Train|Group]")


if __name__ == "__main__":
    # disable tensorflow warning for MTCNN model
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    run(sys.argv)
