from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import *
import tensorflow as tf

def proposed(input_size=(256, 256, 3),lr=1e-4):
    inputs_us = Input(input_size)
    inputs_dpl = Input(input_size)

    us_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_us)
    us_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_conv1)
    us_conv1 = BatchNormalization()(us_conv1)
    dpl_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_dpl)
    dpl_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_conv1)
    dpl_conv1 = BatchNormalization()(dpl_conv1)
    [us_cross1,dpl_cross1] = cross([us_conv1,dpl_conv1])
    us_conv1 =  us_cross1
    dpl_conv1 = dpl_cross1
    conv1 = concatenate([us_conv1,dpl_conv1])
    us_pool1 = MaxPooling2D(pool_size=(2, 2))(us_conv1)
    dpl_pool1 = MaxPooling2D(pool_size=(2, 2))(dpl_conv1)

    us_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_pool1)
    us_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_conv2)
    us_conv2 = BatchNormalization()(us_conv2)
    dpl_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_pool1)
    dpl_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_conv2)
    dpl_conv2 = BatchNormalization()(dpl_conv2)
    us_cross2,dpl_cross2 =cross([us_conv2, dpl_conv2])
    us_conv2 = us_cross2
    dpl_conv2 = dpl_cross2
    conv2 = concatenate([us_conv2, dpl_conv2])
    us_pool2 = MaxPooling2D(pool_size=(2, 2))(us_conv2)
    dpl_pool2 = MaxPooling2D(pool_size=(2, 2))(dpl_conv2)

    us_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_pool2)
    us_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_conv3)
    us_conv3 = BatchNormalization()(us_conv3)
    dpl_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_pool2)
    dpl_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_conv3)
    dpl_conv3 = BatchNormalization()(dpl_conv3)
    us_cross3, dpl_cross3 = cross([us_conv3, dpl_conv3])
    us_conv3 = us_cross3
    dpl_conv3 = dpl_cross3
    conv3 = concatenate([us_conv3, dpl_conv3])
    us_pool3 = MaxPooling2D(pool_size=(2, 2))(us_conv3)
    dpl_pool3 = MaxPooling2D(pool_size=(2, 2))(dpl_conv3)

    us_conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_pool3)
    us_conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_conv4)
    us_conv4 = BatchNormalization()(us_conv4)
    dpl_conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_pool3)
    dpl_conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_conv4)
    dpl_conv4 = BatchNormalization()(dpl_conv4)
    us_cross4, dpl_cross4 = cross([us_conv4, dpl_conv4])
    us_conv4 = us_cross4
    dpl_conv4 = dpl_cross4
    conv4 = concatenate([us_conv4, dpl_conv4])
    us_pool4 = MaxPooling2D(pool_size=(2, 2))(us_conv4)
    dpl_pool4 = MaxPooling2D(pool_size=(2, 2))(dpl_conv4)

    us_conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_pool4)
    us_conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(us_conv5)
    us_conv5 = BatchNormalization()(us_conv5)
    dpl_conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_pool4)
    dpl_conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dpl_conv5)
    dpl_conv5 = BatchNormalization()(dpl_conv5)
    us_cross5, dpl_cross5 = cross([us_conv5, dpl_conv5])
    us_conv5 = us_cross5
    dpl_conv5 =dpl_cross5

    conv5 = concatenate([us_conv5, dpl_conv5])

    out = up_block(conv1,conv2,conv3,conv4,conv5)
    model = Model(input=[inputs_us,inputs_dpl], output=[out])
    # from parallel_model import ParallelModel
    # model = ParallelModel(model, 2)
    model.compile(optimizer=Adam(lr=lr), loss=[dice_coef_loss],metrics=['accuracy'])

    return model


def channel_attention_module(inputs, reduction_ratio, reuse=None, scope='channel_attention'):
    A,B = inputs
    input_channel = A.get_shape().as_list()[-1]
    num_squeeze = input_channel //reduction_ratio
    #MODALITY Alignment?
    dense1 = Dense(num_squeeze)
    dense2 = Dense(input_channel)

    avg_pool_A = GlobalAveragePooling2D()(A)
    avg_pool_A = dense1(avg_pool_A)
    avg_pool_A = dense2(avg_pool_A)
    max_pool_A = GlobalMaxPooling2D()(A)
    max_pool_A = dense1(max_pool_A)
    max_pool_A = dense2(max_pool_A)
    scale_A = add([avg_pool_A,max_pool_A])
    scale_A = Activation('sigmoid')(scale_A)
    channel_attention_A = Multiply()([scale_A ,A])


    num_squeeze = input_channel // reduction_ratio
    avg_pool_B = GlobalAveragePooling2D()(B)
    avg_pool_B = dense1(avg_pool_B)
    avg_pool_B = dense2(avg_pool_B)
    max_pool_B = GlobalMaxPooling2D()(B)
    max_pool_B = dense1(max_pool_B)
    max_pool_B = dense2(max_pool_B)
    scale_B = add([avg_pool_B, max_pool_B])
    scale_B = Activation('sigmoid')(scale_B)
    channel_attention_B = Multiply()([scale_B, B])

    # print('channel_attention',channel_attention.shape)
    return channel_attention_A,channel_attention_B

def _reduce_mean(x):
    return tf.reduce_mean(x,axis=3,keep_dims=True)
def spatial_attention_module(inputs, kernel_size=7):
    A,B = inputs
    conv = Conv2D(1, kernel_size, padding='same')

    avg_pool_A = Lambda(_reduce_mean)(A)
    assert avg_pool_A.get_shape()[-1] == 1
    max_pool_A = Lambda(_reduce_mean)(A)
    assert max_pool_A.get_shape()[-1] == 1
    concat_A = concatenate([avg_pool_A, max_pool_A], axis=3)
    assert concat_A.get_shape()[-1] == 2
    concat_A = conv(concat_A)
    scale_A =Activation('sigmoid')(concat_A)
    spatial_attention_A =  Multiply()([scale_A ,A])

    avg_pool_B = Lambda(_reduce_mean)(B)
    assert avg_pool_B.get_shape()[-1] == 1
    max_pool_B = Lambda(_reduce_mean)(B)
    assert max_pool_B.get_shape()[-1] == 1
    concat_B = concatenate([avg_pool_B, max_pool_B], axis=3)
    assert concat_B.get_shape()[-1] == 2
    concat_B = conv(concat_B)
    scale_B =Activation('sigmoid')(concat_B)
    spatial_attention_B =  Multiply()([scale_B ,B])



    return spatial_attention_A,spatial_attention_B

def cbam_block_channel_first(inputs, reduction_ratio=16):

    channel_attention_A,channel_attention_B = channel_attention_module(inputs, reduction_ratio)
    spatial_attention_A ,spatial_attention_B = spatial_attention_module([channel_attention_A,channel_attention_B ], kernel_size=7)

    return spatial_attention_A ,spatial_attention_B


def cross(inputs):
    assert isinstance(inputs,list)
    A,B = cbam_block_channel_first(inputs)
    # A = cbam_block_channel_first(A)
    return A,B
