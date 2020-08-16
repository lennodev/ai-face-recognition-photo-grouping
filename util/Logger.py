import configparser


class Logger:
    def __init__(self):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read('config.ini')
        self.logLevel = config.get('logging', 'log_level')

    def logDebug(self,  message):
        if(self.logLevel == "DEBUG"):
            print(message)

    def logInfo(self,  message):
        if(self.logLevel == "INFO" or self.logLevel == "DEBUG"):
            print(message)

    def logWarning(self,  message):
        if(self.logLevel == "WARNING" or self.logLevel == "INFO" or self.logLevel == "DEBUG"):
            print(message)
