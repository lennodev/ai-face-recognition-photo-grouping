from loader import Service, Model
import sys

import logging
import os

# DI load service
# aiService = Service.aiService()
trainService = Service.trainService()
detectService = Service.detectService()


def run(mode, args):
    print(f"Received mode: {mode}")
    if(mode == "Register"):
        # commonService.register(args)
    if(mode == "Train"):
        dsPath = "train_dataset"
        trainService.run(dsPath)
    elif(mode == "Group"):
        dsPath = "detect_dataset"
        args = dsPath #TODO: to be removed, for testing
        detectService.run(args)
    else:
        print("Fail to run, mode incorrect!")

    # aiService.run(path)


if __name__ == "__main__":
    # disable tensorflow warning for MTCNN model
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    run(str(sys.argv[1]),str(sys.argv[2]))
