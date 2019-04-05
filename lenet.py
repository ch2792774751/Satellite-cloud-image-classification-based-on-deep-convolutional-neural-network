from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.utils as keras_utils


def Lenet(out_class, input_shape):
    inputs = KL.Input(shape = input_shape)
    x = KL.Conv2D(filters= 10, kernel_size= (2,2), strides= (1, 1), padding= 'same',activation= 'relu',use_bias=True)(inputs)
    x = KL.MaxPooling2D(pool_size= (2, 2), strides= (2, 2), padding= 'same')(x)
    x = KL.Conv2D(filters= 20, kernel_size= (2,2), strides= (1, 1), padding= 'same', activation= 'relu', use_bias=True)(x)
    x = KL.MaxPooling2D(pool_size= (2, 2), strides= (2, 2), padding= 'valid')(x)
    x = KL.Conv2D(filters= 40, kernel_size= (2,2), strides= (1, 1), padding= 'same', activation= 'relu', use_bias=True)(x)
    x = KL.Flatten()(x)
    x = KL.Dense(10, activation= keras.activations.relu, use_bias=True)(x)
    out = KL.Dense(out_class, activation= 'softmax', use_bias=True)(x)
    model = KM.Model(inputs= inputs, outputs = out)
    return model

if __name__ == "__main__":
    model = Lenet(4,(28,28,4))
    model.summary()






