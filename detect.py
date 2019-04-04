import numpy as np
import cv2
import pickle
import time
from keras.models import load_model

model = load_model('./resnet.h5')#/home/songwenzhu/caohui/lenet__1_100_epochs.h5
with open('./test.pickle','rb') as f:
        test = pickle.load(f)
test = test[:,:,:,0:4]
test = np.array(test)
print(test.shape)

prediction = model.predict(test)
print("预测图片的维度是:",prediction.shape)

prediction = np.argmax(prediction,axis=1)
prediction = prediction.reshape([w,h])#w,h 长　宽
print("预测图片的维度是:",prediction.shape)
pre_image = np.zeros([w,h,3])

pred_time_start = time.time()
for i in range(pred_image.shape[0]):
        for j in range(pred_image.shape[1]):
                if pred_image[i][j] == 0:
                        img[i][j][0] = 255
                        img[i][j][1] = 0
                        img[i][j][2] = 0
                elif pred_image[i][j] == 1:
                        img[i][j][0] = 0
                        img[i][j][1] = 0
                        img[i][j][2] = 255
                elif pred_image[i][j] == 2:
                        img[i][j][0] = 255
                        img[i][j][1] = 255
                        img[i][j][2] = 255
                else:
                        img[i][j][0] = 0
                        img[i][j][1] = 0
                        img[i][j][2] = 0
cv2.imwrite('./*.png',img)
pred_time_end = time.time()
print("预测时间是:",pred_time_end - pred_time_start,'s')
print('预测结束') 
