from dependency_injector import providers, containers
from service.AiService import AiService
from service.TrainService import TrainService
from service.DetectService import DetectService
from service.FaceExtractService import FaceExtractService
from util.Logger import Logger
from model.ModelLoader import ModelLoader


class Model(containers.DeclarativeContainer):
    modelLoader = providers.Singleton(ModelLoader)


class Common(containers.DeclarativeContainer):
    logger = providers.Singleton(Logger)

    feService = providers.Singleton(
        FaceExtractService, modelLoader=Model.modelLoader, logger=logger)


class Service(containers.DeclarativeContainer):

    aiService = providers.Singleton(AiService, modelLoader=Model.modelLoader)

    trainService = providers.Singleton(
        TrainService, modelLoader=Model.modelLoader, faceExtractService=Common.feService)

    detectService = providers.Singleton(
        DetectService, modelLoader=Model.modelLoader, faceExtractService=Common.feService)
