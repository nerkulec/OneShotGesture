import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Model

# input images will be 256x256

# encoder components
# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')
pool_1 = MaxPooling2D((2,2), name='pool_1')
drop_1 = Dropout(0.7, name='dropout_1')
batn_1 = BatchNormalization(name='bach_norm_1')

# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2')
drop_2 = Dropout(0.7, name='dropout_2')
pool_2 = MaxPooling2D((2,2), name='pool_2')
batn_2 = BatchNormalization(name='bach_norm_2')

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_3 = Conv2D(64, (1,1), activation='relu', padding='same', name='conv_3')
drop_3 = Dropout(0.7, name='dropout_3')
batn_3 = BatchNormalization(name='bach_norm_3')
conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')
pool_3 = MaxPooling2D((2,2), name='pool_3')
drop_4 = Dropout(0.7, name='dropout_4')
batn_4 = BatchNormalization(name='bach_norm_4')

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_5 = Conv2D(64, (1,1), activation='relu', padding='same', name='conv_5')
drop_5 = Dropout(0.7, name='dropout_5')
batn_5 = BatchNormalization(name='bach_norm_5')
conv_6 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_6')
pool_4 = MaxPooling2D((2,2), name='pool_4')
drop_6 = Dropout(0.7, name='dropout_6')
batn_6 = BatchNormalization(name='bach_norm_6')

# FLAT -> DENSE -> DROPOUT -> BATCH NORM -> DENSE -> DROPOUT -> BATCH NORM -> DENSE
flat_1 = Flatten(name='flatten_1')
dens_1 = Dense(256, activation='relu')
drop_7 = Dropout(0.7, name='dropout_7')
batn_7 = BatchNormalization(name='bach_norm_7')
dens_2 = Dense(128, activation='relu')
drop_8 = Dropout(0.7, name='dropout_8')
batn_8 = BatchNormalization(name='bach_norm_8')
dens_3 = Dense(64,  activation='relu')

def encoder(X):
    X = conv_1(X)
    X = pool_1(X)
    X = drop_1(X)
    X = batn_1(X)

