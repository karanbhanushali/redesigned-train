import tensorflow as tf
from src.controllers.KrusaderController import *



def main():
    print(tf.__version__)
    print(tf.config.list_physical_devices("GPU"))
    initializeApplication()
    controller = KrusaderController()
    controller.start()
    print()

def initializeApplication():
    configuration = ConfigManager()     
    if not os.path.exists(configuration.appConfig.logFolder):
        create_folders(configuration.appConfig.logFolder)    
    if not os.path.exists(configuration.openSmileConfig.LLDOutputFolder):
        create_folders(configuration.openSmileConfig.LLDOutputFolder)
    if not os.path.exists(configuration.openSmileConfig.FunctionalsOuputFolder):
        create_folders(configuration.openSmileConfig.FunctionalsOuputFolder)
    if not os.path.exists(configuration.librosaConfig.outputFolder):
        create_folders(configuration.librosaConfig.outputFolder)
    if not os.path.exists(configuration.wav2vecConfig.outputFolder):
        create_folders(configuration.wav2vecConfig.outputFolder)
    

def create_folders(path):
    os.makedirs(path, exist_ok=True)


    
if __name__ == '__main__':
    main()


    # decision tree classification
    # don't overparameterise;
    # learns the padding, the leangth of the word

    # split for regions in the crossvalidation manner
    # spilt outputs? 0-4 and 5-9
    # maybe augmentation on validation data
    # recording some data
    # more precise error analysis
    # more metrics than accuracy
