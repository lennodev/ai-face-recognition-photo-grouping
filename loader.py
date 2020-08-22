from dependency_injector import providers, containers
from service.TrainService import TrainService
from service.DetectService import DetectService
from service.FaceExtractService import FaceExtractService
from service.CommonService import CommonService
from util.Logger import Logger
from util.ConfigLoader import ConfigLoader
from model.ModelLoader import ModelLoader


class Common(containers.DeclarativeContainer):
    configLoader = providers.Singleton(ConfigLoader)

    logger = providers.Singleton(Logger, configLoader=configLoader)


class Model(containers.DeclarativeContainer):
    modelLoader = providers.Singleton(
        ModelLoader, configLoader=Common.configLoader, logger=Common.logger)


class Service(containers.DeclarativeContainer):

    feService = providers.Singleton(
        FaceExtractService, modelLoader=Model.modelLoader, logger=Common.logger)

    trainService = providers.Singleton(
        TrainService, modelLoader=Model.modelLoader, faceExtractService=feService, logger=Common.logger, configLoader=Common.configLoader)

    detectService = providers.Singleton(
        DetectService, modelLoader=Model.modelLoader, faceExtractService=feService, logger=Common.logger, configLoader=Common.configLoader)

    commonService = providers.Singleton(
        CommonService, logger=Common.logger, configLoader=Common.configLoader)
