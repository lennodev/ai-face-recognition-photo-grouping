from service.AiService import AiService
from loader import Service, Model
import sys

import logging
import os

# DI load service
aiService = Service.aiService()


def run(path):
    print(f"Received path: {path}")
    aiService.run(path)


if __name__ == "__main__":
    #disable tensorflow warning for MTCNN model
    logging.disable(logging.WARNING) 
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # run(str(sys.argv[1]))
    run("/Users/niueva/dev/python/ai-face-recognition-photo-grouping/real_dataset")
