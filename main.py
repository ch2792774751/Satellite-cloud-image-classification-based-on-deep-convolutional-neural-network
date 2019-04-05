import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras.optimizers import SGD,Adam
from keras.utils import np_utils  
from keras.utils.vis_utils import plot_model  
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import keras
import pickle
import time
from collections import Counter
from keras.callbacks import History
from sklearn.preprocessing import MinMaxScaler
from BP import *
from Config import *
from resnet import *
from lenet import *
from utils import *

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.1e-6)
early_stopper = EarlyStopping(min_delta=0.01, patience=100)

gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

fr1 = open('./train2.pickle','rb')   
trainData, trainLabels,testData, testLabels =  pickle.load(fr1)
#trainData, testData = trainData.astype(np.float16)/255.0, testData.astype(np.float16)/255.0
trainData = trainData[:,:,:,0:4] 
testData = testData[:,:,:,0:4] 

#trainData = trainData.reshape(trainData.shape[0],trainData.shape[1]*trainData.shape[2])
#testData = testData.reshape(testData.shape[0],testData.shape[1]*testData.shape[2])

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

#BP
#model = BP(Config.CLASSES,Config.BP_INPUT_SHAPE)
#model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy']) 

time_start = time.time()
print("开始训练神经网络模型")


#resnet
#model = resnet(Config.CLASSES,Config.INPUT_SHAPE)
#sgd = SGD(lr=Config.LEARNING_RATE, decay=Config.WEIGHT_DECAY, momentum=Config.LEARNING_MOMENTUM, nesterov=True)
#model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  
#model.summary()

#lenet
model = Lenet(Config.CLASSES,Config.INPUT_SHAPE_4)
sgd = SGD(lr=Config.LEARNING_RATE, decay=Config.WEIGHT_DECAY, momentum=Config.LEARNING_MOMENTUM, nesterov=True)
#model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  
model.summary()

history = History()
history=model.fit(trainData,trainLabels,batch_size=Config.BATCH_SIZE,epochs=Config.EPOCH,verbose=1,shuffle=True,validation_data=(testData,testLabels),callbacks=[lr_reducer, early_stopper])
#model.save('./resnet_1_100epochs.h5')
model.save("./lenet_4_50.h5")


scores = model.evaluate(testData,testLabels,verbose=0)
print("scores = ",scores)
time_end = time.time()
print("resnet神经网络模型训练结束")
print('resnet神经网络模型训练的时间是:',time_end - time_start,'s')
#plot_model(resnet,to_file="./model/resnet50.png",show_shapes=True)
save_data(file_dir='./result/' ,filename='val_loss',data=history.history["val_loss"])
save_data(file_dir='./result/' ,filename='train_loss',data=history.history["loss"])
save_data(file_dir='./result/' ,filename='val_acc',data=history.history["val_acc"])
save_data(file_dir='./result/' ,filename='train_acc',data=history.history["acc"])

plot_fig(history.history["loss"], history.history["val_loss"],'train_loss', 'val_loss','epoch','loss')
plot_fig(history.history["acc"], history.history["val_acc"],'train_acc', 'val_acc','epoch','acc')
# fig, ax1 = plt.subplots(1,1)
# ax1.plot(history.history["val_loss"])
# ax1.plot(history.history["loss"])
# plt.show()
