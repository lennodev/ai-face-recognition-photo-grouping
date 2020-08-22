import configparser


class ConfigLoader:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read('config.ini')

    def getConfig(self, session, attribute):
        return self.config.get(session, attribute)
