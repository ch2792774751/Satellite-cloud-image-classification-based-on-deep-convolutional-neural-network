from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.utils as keras_utils
#https://www.jianshu.com/p/acf4c3586031

#残差块
def conv_block(input_tensor, filters):
    filter1, filter2, filter3 = filters
    #1x1的卷积
    x = KL.Conv2D(filter1,(1,1),strides=1)(input_tensor)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    #3x3的卷积
    x = KL.Conv2D(filter2,(3,3),strides=1,padding='same')(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    #1x1的卷积
    x = KL.Conv2D(filter3,(1,1),strides=1)(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)

    shoutcut = KL.Conv2D(filter3,(1,1),strides=1)(input_tensor)
    x = KL.add([x,shoutcut])
    x = KL.Activation('relu')(x)
    return x

#残差块
def identity_block(input_tensor, filters):
    filter1, filter2, filter3 = filters
    #1x1的卷积
    x = KL.Conv2D(filter1,(1,1),strides=1)(input_tensor)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    #3x3的卷积
    x = KL.Conv2D(filter2,(3,3),strides=1,padding='same')(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    #1x1的卷积
    x = KL.Conv2D(filter3,(1,1),strides=1)(x)
    x = KL.BatchNormalization(axis=-1)(x)

    x = KL.add([x,input_tensor])
    x = KL.Activation('relu')(x)
    return x

#构造深度残差卷积神经网络
def resnet(out_class, input_shape):
    #图像输入28x28x4
    inputs = KL.Input(shape=input_shape)  
    #常规卷积层
    x = KL.Conv2D(64, (7, 7), strides=2, padding='same')(inputs) 
    x = KL.BatchNormalization(axis=-1)(x) 
    x = KL.Activation('relu')(x) 
    x = KL.MaxPooling2D(pool_size=(3,3),strides=2)(x) 
    #第一个残差块 3
    x = conv_block(x, [64, 64, 256]) 
    x = identity_block(x, [64, 64, 256]) 
    x = identity_block(x, [64, 64, 256]) 
    #第二个残差块 4  
    x = conv_block(x, [128,128,512]) 
    x = identity_block(x, [128,128,512]) 
    x = identity_block(x, [128,128,512]) 
    x = identity_block(x, [128, 128, 512])  
    #第三个残差块 6
    x = conv_block(x, [256,256,1024])  
    x = identity_block(x, [256, 256, 1024]) 
    x = identity_block(x, [256, 256, 1024])  
    x = identity_block(x, [256, 256, 1024])  
    x = identity_block(x, [256, 256, 1024])  
    x = identity_block(x, [256, 256, 1024]) 
    '''
    #第四个残差块 3 
    x = conv_block(x, [512,512,2048])  
    x = identity_block(x, [512, 512, 2048])  
    x = identity_block(x, [512, 512, 2048])
    '''
    #全局平均池化
    x = KL.AveragePooling2D()(x) 
    x = KL.Flatten()(x)
    x = KL.Dense(out_class)(x)  
    out = KL.Activation('softmax')(x)
    model = KM.Model(inputs=inputs, outputs=out,name = "Resnet50")
    return model

if __name__ == "__main__":
    #环境卫星图像数据集分为4类
    classes = 4
    resnet = resnet_model(classes,(28,28,4))
    resnet.summary()
