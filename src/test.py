from utils import *
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
# matplotlib.use('Agg')
from data_generator import DataGenerator
# from coars_to_fine_data_generator import *
from model import proposed,proposed_dual_stream,base_line_unet_us,base_line_unet_us_dpl,base_line_unet_dpl,coarse_to_fine_FCN_A,coarse_to_fine_FCN_B,base_line_unet_us_dpl_small
import keras.backend as K
import math

train_list ='./data_list/us_dpl_ori_data/train.txt'
val_list ='./data_list/us_dpl_ori_data/val.txt'
test_list = './data_list/us_dpl_ori_data/test.txt'

experimenet_name ='baseline_us_dpl_small'

model = base_line_unet_us_dpl_small((192,192,3),lr=1e-5)
model.load_weights(experimenet_name+".h5")

visualize=False

out_dir = './test/'+experimenet_name
if not os.path.exists(out_dir):
        os.mkdir(out_dir)
testgen =DataGenerator(test_list)

aps =[]
count =0
next_batch = testgen.next_batch(mode='test')
dices =[]
vessel_dices=[]
muscle_dices=[]
lymph_dices =[]
while True:
        try:
                print('predicting : ',count+1)
                img_path,inputs,outputs= next_batch.__next__()
                img_us,img_dpl= inputs

                gt_mask=outputs

                # print(img_us.shape,gt_mask.shape)
                # print('img shape and gt_mask shape',img.shape, gt_mask.shape)
                pre_mask= model.predict(inputs)
                #去掉batch维度
                img_us=img_us[0, :, :, :]
                img_dpl =img_dpl[0,:,:,:]
                gt_mask=gt_mask[0, :, :, :]
                pre_mask = pre_mask[0, :, :, :]

                masked_image = img_us.astype(np.float64).copy()

                # 使用crf后处理
                # pre_mask = dense_crf(masked_image, pre_mask)

                # 取softmax最大
                pre_mask = pre_mask.argmax(axis=-1)[..., None] == np.arange(pre_mask.shape[-1])
                pre_mask = pre_mask.astype(pre_mask.dtype)
                # pre_mask = dense_crf(img.astype(np.uint32), pre_mask.astype(np.bool))
                #只计算淋巴准确率，第0层
                dice = dice_coef(gt_mask[:,:,:3],pre_mask[:,:,:3])
                dices.append(dice)
                if np.max(gt_mask[:,:,0])==0 and np.max(pre_mask[:,:,0])==0:
                        lymph_dice=1
                else:
                        lymph_dice=dice_coef(gt_mask[:,:,0],pre_mask[:,:,0])
                lymph_dices.append(lymph_dice)
                if np.max(gt_mask[:, :, 2]) == 0 and np.max(pre_mask[:, :, 2]) == 0:
                        vessel_dice=1
                else :
                        vessel_dice = dice_coef(gt_mask[:, :, 2], pre_mask[:, :, 2])
                vessel_dices.append(vessel_dice)
                if np.max(gt_mask[:, :, 1]) == 0 and np.max(pre_mask[:, :, 1]) == 0:
                        muscle_dice=1
                else:
                        muscle_dice = dice_coef(gt_mask[:, :, 1], pre_mask[:, :, 1])
                muscle_dices.append(muscle_dice)

                if visualize:
                        # visual
                        pred_img = visual_mask(masked_image,pre_mask)
                        gt_img = visual_mask(img_us.astype(np.float64),gt_mask)

                        fig = plt.figure(figsize=(24, 8))
                        fig.tight_layout()

                        plt.suptitle('image_path :' + img_path +
                                  '   dice :' + str(dice)
                                  , fontsize=25)
                        plt.axis('off')

                        ax1 = fig.add_subplot(141)
                        ax1.set_title("US image")
                        ax1.imshow(img_us.astype(np.float32))
                        ax1.axis('off')

                        ax2 = fig.add_subplot(142)
                        ax2.set_title("DPL image")
                        ax2.imshow(img_dpl.astype(np.float32))
                        ax2.axis('off')

                        ax3 = fig.add_subplot(143)
                        ax3.set_title("Predict"+str(dice))
                        ax3.imshow(pred_img.astype(np.float32))
                        ax3.axis('off')

                        ax4 = fig.add_subplot(144)
                        ax4.set_title("GroundTruth")
                        ax4.imshow(gt_img.astype(np.float32))
                        ax4.axis('off')

                        # plt.margins(0, 0)
                        plt.savefig(os.path.join(out_dir,str(dice)[:6]+str(count+1)+'.jpg'),bbox_inches = 'tight',dpi=300)
                        # plt.savefig('xx.svg',format='svg')
                        plt.close()
                count+=1
        except:
                print('no imgs')
                break

print(sorted(dices))
print('mean dice : %.4f'%(np.mean(dices)))
print('lymph dice:%.4f' %np.mean(lymph_dices))
print('vessel dice:%.4f'% np.mean(vessel_dices))
print('muscle dice:%.4f'% np.mean(muscle_dices))
# 可视化dice值
# plt.plot(sorted(dices))
# plt.savefig('dices.jpg')
# print('mean dice :',np.mean(dices))

