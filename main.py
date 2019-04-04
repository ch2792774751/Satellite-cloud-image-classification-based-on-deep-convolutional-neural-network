import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils  
from keras.utils.vis_utils import plot_model  
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import keras
import pickle
import time
from collections import Counter
from keras.callbacks import History
from Config import Config
from resnet import *
from lenet import *
from utils.py import *
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

fr1 = open('./train_test.pickle','rb')   
trainData, trainLabels,testData, testLabels =  pickle.load(fr1)
testLabels = testLabels.reshape([-1,])
trainLabels = trainLabels.reshape([-1,])

print("数据集信息如下:")
print("trainData.shape    训练集的维度..................", trainData.shape)
print("trainLabels.shape  训练集标签的维度.............", trainLabels.shape)
print("testData.shape     测试集的维度..................", testData.shape)
print("testLabels.shape   测试集标签的维度.............", testLabels.shape)
print('训练集个数',trainData.shape[0])
print('测试集个数',testData.shape[0])
trainLabels = np_utils.to_categorical(trainLabels, Config.CLASSES)  
testLabels = np_utils.to_categorical(testLabels, Config.CLASSES)

resnet = resnet_model(Config.CLASSES,Config.INPUT_SHAPE)
sgd = SGD(lr=Config.LEARNING_RATE, decay=config.WEIGHT_DECAY, momentum=config.LEARNING_MOMENTUM, nesterov=True)
#resnet.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
resnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  
resnet.summary()

time_start = time.time()
print("开始训练resnet神经网络模型")
history = History()
history=resnet.fit(trainData,trainLabels,batch_size=config.BATCH_SIZE,epochs=config.EPOCH,verbose=1,shuffle=True,validation_data=(testData,testLabels),callbacks=[lr_reducer, early_stopper])
resnet.save('./resnet_epochs.h5')
scores = resnet.evaluate(testData,testLabels,verbose=0)
print("scores = ",scores)
time_end = time.time()
print("resnet神经网络模型训练结束")
print('resnet神经网络模型训练的时间是:',time_end - time_start,'s')
plot_model(resnet,to_file="./model/resnet50.png",show_shapes=True)
save_data(file_dir='./result/' ,filename='val_loss',data=history.history["val_loss"])
save_data(file_dir='./result/' ,filename='train_loss',data=history.history["loss"])
plot_fig(history.history["loss"], history.history["val_loss"],'train_loss', 'val_loss')
# fig, ax1 = plt.subplots(1,1)
# ax1.plot(history.history["val_loss"])
# ax1.plot(history.history["loss"])
# plt.show()
