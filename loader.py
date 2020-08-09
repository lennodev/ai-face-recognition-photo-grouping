from dependency_injector import providers, containers
from service.AiService import AiService
from model.ModelLoader import ModelLoader


class Model(containers.DeclarativeContainer):
    modelLoader = providers.Singleton(ModelLoader)


class Service(containers.DeclarativeContainer):
    aiService = providers.Singleton(AiService, modelLoader=Model.modelLoader)

