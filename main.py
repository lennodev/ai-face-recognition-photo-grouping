from loader import Service, Model
import sys

import logging
import os

# DI load service
# aiService = Service.aiService()
trainService = Service.trainService()
detectService = Service.detectService()


def run(mode):
    print(f"Received mode: {mode}")
    if(mode == "Train"):
        dsPath = "train_dataset"
        trainService.run(dsPath)
    elif(mode == "Detect"):
        dsPath = "detect_dataset"
        detectService.run(dsPath)
    else:
        print("Fail to run, mode incorrect!")

    # aiService.run(path)


if __name__ == "__main__":
    # disable tensorflow warning for MTCNN model
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    run(str(sys.argv[1]))
