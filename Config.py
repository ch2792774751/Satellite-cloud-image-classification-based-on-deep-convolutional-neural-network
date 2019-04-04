class Config(object):
    NAME = "Resnet"
    INPUT_SHAPE = [28,28,4]
    BATCH_SIZE = 128
    RESNET = [3,4,6,3]
    CLASSES = 4
    EPOCH = 100
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

if __name__=='__main__':
    config=Config()
    print(config.NAME)


