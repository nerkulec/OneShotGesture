import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Subtract, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
from random import random
import datetime
from dataset import load_data

# input images will be 256x256x3
# input_shape = (256, 256, 3)
input_shape = (28, 28, 1)
encoding_size = 64 # CHANGE TO 64

# encoder components
# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')
pool_1 = MaxPooling2D((2,2), name='pool_1')
drop_1 = Dropout(0.7, name='dropout_1')
batn_1 = BatchNormalization(name='batch_norm_1')

# size is now 128x128x64

# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2') # CHANGE TO 128
drop_2 = Dropout(0.7, name='dropout_2')
pool_2 = MaxPooling2D((2,2), name='pool_2')
batn_2 = BatchNormalization(name='batch_norm_2')

# size is now 64x64x128

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_3 = Conv2D(32, (1,1), activation='relu', padding='same', name='conv_3')
drop_3 = Dropout(0.7, name='dropout_3')
batn_3 = BatchNormalization(name='batch_norm_3')
conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')
pool_3 = MaxPooling2D((2,2), name='pool_3')
drop_4 = Dropout(0.7, name='dropout_4')
batn_4 = BatchNormalization(name='batch_norm_4')

# size is now 32x32x256

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_5 = Conv2D(128, (1,1), activation='relu', padding='same', name='conv_5')
drop_5 = Dropout(0.7, name='dropout_5')
batn_5 = BatchNormalization(name='batch_norm_5')
conv_6 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_6')
pool_4 = MaxPooling2D((2,2), name='pool_4')
drop_6 = Dropout(0.7, name='dropout_6')
batn_6 = BatchNormalization(name='batch_norm_6')

# size is now 16x16x64

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM

conv_7 = Conv2D(8, (1,1), activation='relu', padding='same', name='conv_7')
drop_7 = Dropout(0.7, name='dropout_7')
batn_7 = BatchNormalization(name='batch_norm_7')
conv_8 = Conv2D(16, (3,3), activation='relu', padding='same', name='conv_8')
pool_5 = MaxPooling2D((2,2), name='pool_5')
drop_8 = Dropout(0.5, name='dropout_8')
batn_8 = BatchNormalization(name='batch_norm_8')

# size is now 8x8x16

# FLAT -> DENSE -> DROPOUT -> BATCH NORM -> DENSE -> DROPOUT -> BATCH NORM -> DENSE

flat_1 = Flatten(name='flatten_1')
dens_1 = Dense(64, activation='relu', name='dense_1') # CHANGE TO 64
drop_8 = Dropout(0.3, name='dropout_8')
batn_9 = BatchNormalization(name='batch_norm_9')
dens_2 = Dense(encoding_size, activation='relu', name='dense_2')

def encoder_forward(X):
    X = conv_1(X)
    X = pool_1(X)
    X = drop_1(X)
    X = batn_1(X)

    # X = conv_2(X)
    # X = pool_2(X)
    # X = drop_2(X)
    # X = batn_2(X)

    # X = conv_3(X)
    # X = drop_3(X)
    # X = batn_3(X)
    # X = conv_4(X)
    # X = pool_3(X)
    # X = drop_4(X)
    # X = batn_4(X)

    # X = conv_5(X)
    # X = drop_5(X)
    # X = batn_5(X)
    # X = conv_6(X)
    # X = pool_4(X)
    # X = drop_6(X)
    # X = batn_6(X)

    X = conv_7(X)
    X = drop_7(X)
    X = batn_7(X)
    X = conv_8(X)
    X = pool_5(X)
    X = drop_8(X)
    X = batn_8(X)

    X = flat_1(X)
    # X = dens_1(X)
    # X = drop_8(X)
    # X = batn_9(X)
    X = dens_2(X)
    return X

def encoder_model():
    X_input = Input(input_shape, name='encoder_input')
    X_encoded = encoder_forward(X_input)
    model = Model(inputs=X_input, outputs=X_encoded, name='encoder')
    return model

dens_3 = Dense(16, activation='relu', name='dense_3')
drop_9 = Dropout(0.4, name='dropout_9')
batn_10 = BatchNormalization(name='batch_norm_10')
dens_4 = Dense(1, activation='sigmoid', name='dense_4')

def comparator_forward(X_1, X_2, encoded=False):
    if encoded:
        X_1_encoding = X_1
        X_2_encoding = X_2
    else:
        X_1_encoding = encoder_forward(X_1)
        X_2_encoding = encoder_forward(X_2)
    X = Lambda(abs)(Subtract(name='difference')([X_2_encoding, X_1_encoding]))
    # X = dens_3(X)
    # X = drop_9(X)
    # X = batn_10(X)
    X = dens_4(X)
    return X

def comparator_model(encoded=False):
    if encoded:
        X_1 = Input(encoding_size, name='encoded_comparator_input_1')
        X_2 = Input(encoding_size, name='encoded_comparator_input_2')
    else:
        X_1 = Input(input_shape, name='comparator_input_1')
        X_2 = Input(input_shape, name='comparator_input_2')
    output = comparator_forward(X_1, X_2, encoded)
    model = Model(inputs=[X_1, X_2], outputs=output, name=('encoding ' if encoded else '')+'comparator')
    return model

if __name__ == '__main__':

    encoder = encoder_model()
    comparator = comparator_model(encoded=False)
    encoding_comparator = comparator_model(encoded=True)
        
    encoder.summary()
    # comparator.summary()
    # encoding_comparator.summary()
    
    comparator.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    (x_train, y_train), (x_test, y_test) = load_data()

    comparator.fit(x=x_train, y = y_train,
          epochs=20,
          batch_size=64,
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

    comparator.save('./saves/comparator.h5')

    
