import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# input images will be 256x256x3
input_shape = (256, 256, 3)
encoding_size = 64

# encoder components
# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')
pool_1 = MaxPooling2D((2,2), name='pool_1')
drop_1 = Dropout(0.7, name='dropout_1')
batn_1 = BatchNormalization(name='bach_norm_1')

# size is now 128x128x64

# 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2')
drop_2 = Dropout(0.7, name='dropout_2')
pool_2 = MaxPooling2D((2,2), name='pool_2')
batn_2 = BatchNormalization(name='bach_norm_2')

# size is now 64x64x128

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_3 = Conv2D(64, (1,1), activation='relu', padding='same', name='conv_3')
drop_3 = Dropout(0.7, name='dropout_3')
batn_3 = BatchNormalization(name='bach_norm_3')
conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')
pool_3 = MaxPooling2D((2,2), name='pool_3')
drop_4 = Dropout(0.7, name='dropout_4')
batn_4 = BatchNormalization(name='bach_norm_4')

# size is now 32x32x256

# 1x1 CONV -> DROPOUT -> BATCH NORM -> 3x3 CONV -> POOL -> DROPOUT -> BATCH NORM
conv_5 = Conv2D(128, (1,1), activation='relu', padding='same', name='conv_5')
drop_5 = Dropout(0.7, name='dropout_5')
batn_5 = BatchNormalization(name='bach_norm_5')
conv_6 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_6')
pool_4 = MaxPooling2D((2,2), name='pool_4')
drop_6 = Dropout(0.7, name='dropout_6')
batn_6 = BatchNormalization(name='bach_norm_6')

# size is now 16x16x64

# 1x1 CONV -> FLAT -> DENSE -> DROPOUT -> BATCH NORM -> DENSE -> DROPOUT -> BATCH NORM -> DENSE
conv_7 = Conv2D(16, (1,1), activation='relu', padding='same', name='conv_7')
flat_1 = Flatten(name='flatten_1')
dens_1 = Dense(128, activation='relu', name='dense_1')
drop_7 = Dropout(0.7, name='dropout_7')
batn_7 = BatchNormalization(name='bach_norm_7')
dens_2 = Dense(encoding_size, activation='relu', name='dense_2')

def encoder_forward(X):
    X = conv_1(X)
    X = pool_1(X)
    X = drop_1(X)
    X = batn_1(X)

    X = conv_2(X)
    X = pool_2(X)
    X = drop_2(X)
    X = batn_2(X)

    X = conv_3(X)
    X = drop_3(X)
    X = batn_3(X)
    X = conv_4(X)
    X = pool_3(X)
    X = drop_4(X)
    X = batn_4(X)

    X = conv_5(X)
    X = drop_5(X)
    X = batn_5(X)
    X = conv_6(X)
    X = pool_4(X)
    X = drop_6(X)
    X = batn_6(X)

    X = conv_7(X)
    X = flat_1(X)
    X = dens_1(X)
    X = drop_7(X)
    X = batn_7(X)
    X = dens_2(X)
    return X

def encoder_model():
    X_input = Input(input_shape, name='encoder_input')
    X_encoded = encoder_forward(X_input)
    model = Model(inputs=X_input, outputs=X_encoded, name='encoder')
    return model

dens_3 = Dense(16, activation='relu', name='dense_3')
drop_8 = Dropout(0.4, name='dropout_8')
batn_8 = BatchNormalization(name='bach_norm_8')
dens_4 = Dense(1, activation='sigmoid', name='dense_4')

def comparator_forward(X_1, X_2, encoded=False):
    if encoded:
        X_1_encoding = X_1
        X_2_encoding = X_2
    else:
        X_1_encoding = encoder_forward(X_1)
        X_2_encoding = encoder_forward(X_2)
    X = tf.abs(tf.subtract(X_2_encoding, X_1_encoding, name='difference'), name='abs')
    X = dens_3(X)
    X = drop_8(X)
    X = batn_8(X)
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

encoder = encoder_model()
comparator = comparator_model(encoded=False)
encoding_comparator = comparator_model(encoded=True)
    
encoder.summary()
comparator.summary()
encoding_comparator.summary()
