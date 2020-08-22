import configparser


class Logger:
    def __init__(self, configLoader):
        self.logLevel = configLoader.getConfig('logging', 'log_level')

    def logDebug(self,  message):
        if(self.logLevel == "DEBUG"):
            prefixStr = "> Debug: "
            print(prefixStr + message)

    def logInfo(self,  message):
        if(self.logLevel == "INFO" or self.logLevel == "DEBUG"):
            prefixStr = ">>> Info: "
            print(prefixStr + message)

    def logWarning(self,  message):
        if(self.logLevel == "WARNING" or self.logLevel == "INFO" or self.logLevel == "DEBUG"):
            prefixStr = "*** Warning: "
            print(prefixStr + message)
