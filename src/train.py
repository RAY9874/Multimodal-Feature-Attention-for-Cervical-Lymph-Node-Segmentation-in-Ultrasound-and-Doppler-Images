import os
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.optimizers import SGD
# from data_generator import DataGenerator

from model import proposed,proposed_dual_stream,base_line_unet_us,base_line_unet_dpl,base_line_unet_us_dpl,coarse_to_fine_FCN_A,coarse_to_fine_FCN_B,base_line_unet_us_dpl_small
import time
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
#设置gpu动态使用，不会沾满全部内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

train_list ='./data_list/us_dpl_ori_data/train.txt'
val_list ='./data_list/us_dpl_ori_data/val.txt'
test_list = './data_list/us_dpl_ori_data/test.txt'

from data_generator import DataGenerator
train_gen =DataGenerator(train_list)
val_gen =DataGenerator(val_list)

experimenet_name ='baseline_us_dpl_small'
now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_checkpoint = ModelCheckpoint(experimenet_name+".h5",
                                   monitor='val_acc',verbose=0,
                                   save_best_only=True)
tb =TensorBoard(log_dir='./logs/'+experimenet_name+'/'+now_time, histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=2,mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
            patience=20, min_lr=0)
print('trianing')
model = base_line_unet_us_dpl_small((192,192,3),lr=1e-4)
model.fit_generator(train_gen.next_batch(mode='train'),
                    steps_per_epoch=100,
                    epochs=120,
                    validation_data = val_gen.next_batch(mode='val'),
                    validation_steps = 30,
                    verbose=1,
                    callbacks=[tb,model_checkpoint,reduce_lr])

###
### 对比实验
###
#
# from coars_to_fine_data_generator import *
#
# # stage 1
# train_gen =DataGenerator(train_list,FCNmode="A")
# val_gen =DataGenerator(val_list,FCNmode="A")
#
# experimenet_name ='coarse_to_fine_model_stageA'
# now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
# model_checkpoint = ModelCheckpoint(experimenet_name+".h5",
#                                    monitor='val_acc',verbose=0,
#                                    save_best_only=True)
# tb =TensorBoard(log_dir='./logs/'+experimenet_name+'/'+now_time, histogram_freq=0, write_graph=True, write_images=True)
# early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=2,mode='max')
#
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
#             patience=20, min_lr=0)
#
# model = coarse_to_fine_FCN_A((192,192,3),lr=1e-4,num_class=2)
# model.fit_generator(train_gen.next_batch(mode='train'),
#                     steps_per_epoch=100,
#                     epochs=120,
#                     validation_data = val_gen.next_batch(mode='val'),
#                     validation_steps = 30,
#                     verbose=1,
#                     callbacks=[tb,model_checkpoint,reduce_lr])
#
# #stage 2
#
# train_gen =DataGenerator(train_list,FCNmode="B")
# val_gen =DataGenerator(val_list,FCNmode="B")
#
# experimenet_name ='coarse_to_fine_model_stageB'
# now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
# model_checkpoint = ModelCheckpoint(experimenet_name+".h5",
#                                    monitor='val_acc',verbose=0,
#                                    save_best_only=True)
# tb =TensorBoard(log_dir='./logs/'+experimenet_name+'/'+now_time, histogram_freq=0, write_graph=True, write_images=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
#             patience=20, min_lr=0)
#
# model = coarse_to_fine_FCN_B((192,192,3),lr=1e-4,num_class=4)
# model.fit_generator(train_gen.next_batch(mode='train'),
#                     steps_per_epoch=100,
#                     epochs=120,
#                     validation_data = val_gen.next_batch(mode='val'),
#                     validation_steps = 30,
#                     verbose=1,
#                     callbacks=[tb,model_checkpoint,reduce_lr])