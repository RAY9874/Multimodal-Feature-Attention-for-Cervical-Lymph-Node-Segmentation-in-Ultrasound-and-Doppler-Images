import skimage
from keras.utils.np_utils import *
from distutils.version import LooseVersion
import skimage.transform as trans
import cv2
import os
from random import randint
import imgaug as ia
import random
import imgaug.augmenters as iaa
import pandas as pd

class DataGenerator(object):
    def __init__(self,data_list):
        #先把config 写在这吧
        self.img_max_dim = 192
        self.img_min_dim = 100
        self.batch_size = 12
        self.data_list = data_list
        self.normalize_flag = True
        self.augment_flag = False

    def next_batch(self,mode):
        assert mode in ['train','val','test']
        with open(self.data_list, "r") as fin:
            lines = list(fin.readlines())
        i=0
        b=0
        classes = to_categorical([0,1,2],3)
        while(True):
            try:
                if mode == 'train':
                    i = (i+1)%len(lines)#顺序提取
                    # i = randint(0,len(lines)-1)#随机提取
                elif mode == 'val':
                    i = (i+1)%len(lines)#顺序提取
                    self.augment_flag = False
                elif mode == 'test' :
                    self.augment_flag = False
                    self.batch_size = 1
                    i = i+1

                ar = lines[i]
                a = ar.strip().split("\t")
                name=a[0]
                us_image_path = a[2]
                mask_path = a[3]
                dpl_image_path = a[4]

                if not os.path.exists(us_image_path):
                    # print(us_image_path)
                    continue

                us_image = cv2.imread(us_image_path)
                dpl_image = cv2.imread(dpl_image_path)
                # dpl_image = self.cvt_lab(dpl_image)
                mask =np.load(mask_path)

                if self.augment_flag:
                    us_image,dpl_image = self.random_aug(us_image,dpl_image)
                #归一化
                if self.normalize_flag:
                    us_image = self.normalize(us_image)
                    dpl_image = self.normalize(dpl_image)

                #resize
                us_image,_,scale,_,_ = self.resize_image(img=us_image,min_dim=self.img_min_dim,max_dim=self.img_max_dim)
                dpl_image, _, _, _, _ = self.resize_image(img=dpl_image, min_dim=self.img_min_dim,max_dim=self.img_max_dim)
                mask, _, _, _, _ = self.resize_image(img=mask, min_dim=self.img_min_dim,max_dim=self.img_max_dim)
                #将补零的黑边也标为背景
                mask[:, :, 3] = ~np.any(mask[:, :, :3], axis=2)
                dpl_mask = mask[:,:,2]
                dpl_mask = np.expand_dims(dpl_mask,axis=-1)
                if b == 0:
                    batch_us_images=np.zeros((self.batch_size,) + us_image.shape, dtype=np.float32)
                    batch_dpl_images = np.zeros((self.batch_size,) + dpl_image.shape, dtype=np.float32)
                    batch_masks = np.zeros((self.batch_size,) + mask.shape, dtype=np.uint8)
                    batch_dpl_masks = np.zeros((self.batch_size,) + dpl_mask.shape, dtype=np.uint8)

                # us_image = cv2.cvtColor(us_image, cv2.COLOR_RGB2GRAY)
                batch_us_images[b,:,:,:] = us_image.astype(np.float32)
                batch_dpl_images[b, :, :, :] = dpl_image.astype(np.float32)
                batch_masks[b, :] = mask.astype(np.int8)
                batch_dpl_masks[b,:] = dpl_mask.astype(np.int8)
                b += 1

                # keras generator中yield input, target的target是无法获取的
                # 参考github issues:https://github.com/keras-team/keras/issues/11812
                # 所以为了取到target,我们必须须把target也当作inputs的一部分传进来即
                if b >= self.batch_size:
                    input=[batch_us_images, batch_dpl_images]
                    output = batch_masks
                    if mode == 'test':
                        #测试集还要返回图像路径，便于验证模型
                        yield name,input,output
                        b = 0
                    else:
                        yield input, output
                        b = 0
            except(GeneratorExit):
                print(GeneratorExit)
                raise

    def resize_image(self,img, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        image = img
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None
        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = self.resize(image, (round(h * scale), round(w * scale)),
                           preserve_range=True)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0.)

            window = (top_pad, left_pad, h + top_pad, w + left_pad)

        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop

    def resize(self,image, output_shape, order=1, mode='constant', cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        """A wrapper for Scikit-Image resize().

        Scikit-Image generates warnings on every call to resize() if it doesn't
        receive the right parameters. The right parameters depend on the version
        of skimage. This solves the problem by using different parameters per
        version. And it provides a central place to control resizing defaults.
        """
        if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
            # New in 0.14: anti_aliasing. Default it to False for backward
            # compatibility with skimage 0.13.
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma)
        else:
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range)

    def cvt_lab(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, g, b = cv2.split(img)
        l = np.zeros(l.shape, dtype=np.uint8)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        l = np.expand_dims(l, axis=2)
        lab = np.concatenate([l, g, b], axis=2)
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return rgb

    def normalize(self,image):
        image = image.astype(np.float64)
        mean = np.mean(image)
        if np.max(image)!= 0 :
            image = image - np.min(image)
            image = image / np.max(image)
        return image
    def random_aug(self,us_img,dpl_img):
        seq1 = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        ])
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        num1 = random.uniform(-0.1, 0.1)
        num1 = round(num1, 2)
        num2 = random.uniform(-0.1, 0.1)
        num2 = round(num2, 2)
        num3 = random.uniform(-0.1, 0.1)
        num3 = round(num3, 2)
        num4 = random.uniform(-0.1, 0.1)
        num4 = round(num4, 2)
        seq2 = iaa.Sequential([
                iaa.CropAndPad(percent=(num1, num2, num3, num4),
                    pad_cval=0)])
        seq3 = iaa.Sequential([
            # iaa.GaussianBlur(sigma=(0, 1.0)),  # blur images with a sigma of 0 to 3.0
            iaa.AllChannelsHistogramEqualization()
        ])

        seq4 = iaa.Sequential([
            iaa.Noop()
            # iaa.AllChannelsHistogramEqualization()
        ])

        seq5 = iaa.Sequential([iaa.Affine(
            rotate=(-45, 45), # rotate by -45 to +45 degrees
        )])

        seq6 = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)])

        seq_list = [seq1,seq4,seq5]
        seqs = random.sample(seq_list,1)
        for seq in seqs:
            us_img = seq.augment_image(us_img)
            dpl_img = seq.augment_image(dpl_img)
        return us_img,dpl_img

    #####
    ## 新版超声报告所用多输出
    #####

    def get_changjing(self, name, df, max_changjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['长径（px）']]
        changjing = temp_df.values[0]
        changjing = changjing / max_changjing
        return changjing

    def get_duanjing(self, name, df, max_duanjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['短径（px）']]
        duanjing = temp_df.values[0]
        duanjing = duanjing / max_duanjing
        return duanjing

    def get_pizhihoudu(self, name, df, max_pizhihoudu):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['皮质厚度']]
        pizhihoudu = temp_df.values[0]
        pizhihoudu = pizhihoudu / max_pizhihoudu
        return pizhihoudu

    def get_linbamenhengjing(self, name, df, max_linbamenhengjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['淋巴门横径']]
        linbamenhengjing = temp_df.values[0]
        linbamenhengjing = linbamenhengjing / max_linbamenhengjing
        return linbamenhengjing

    def get_jiejieleixing_huisheng(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['结节类型（高回声/低回声/等回声）_低回声',
                           '结节类型（高回声/低回声/等回声）_等回声',
                           '结节类型（高回声/低回声/等回声）_高回声', ]]
        iejieleixing_huisheng = temp_df.values[0]
        return iejieleixing_huisheng

    def get_jiejieleixing_fa(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['结节类型（多发/单发）_单发',
                           '结节类型（多发/单发）_多发', ]]
        jiejieleixing_fa = temp_df.values[0]
        return jiejieleixing_fa

    def get_bianjie(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['边界（清/不）_不清',
                           '边界（清/不）_清']]
        bianjie = temp_df.values[0]
        return bianjie

    def get_xingtai(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['形态（规/不）_不规则',
                           '形态（规/不）_规则']]
        xingtai = temp_df.values[0]
        return xingtai

    def get_zonghengbi(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['纵横比＞2（有：>2/无：<2）(=2改为<2， 因：均为恶性)_<2',
                           '纵横比＞2（有：>2/无：<2）(=2改为<2， 因：均为恶性)_>2', ]]
        zonghengbi = temp_df.values[0]
        return zonghengbi

    def get_pizhileixing(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['皮质类型（偏心型，不偏心型、无淋巴门）_不偏心',
                           '皮质类型（偏心型，不偏心型、无淋巴门）_偏心',
                           '皮质类型（偏心型，不偏心型、无淋巴门）_无淋巴门', ]]
        pizhileixing = temp_df.values[0]
        return pizhileixing

    def get_linbamenjiegou(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['淋巴门结构（偏心；有）_无淋巴门',
                           '淋巴门结构（偏心；有）_有', ]]
        linbamenjiegou = temp_df.values[0]
        return linbamenjiegou

    def get_gaihua(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['钙化（点状；粗大；无）_无钙化',
                           '钙化（点状；粗大；无）_点状钙化',
                           '钙化（点状；粗大；无）_粗大钙化']]
        gaihua = temp_df.values[0]
        return gaihua

    def get_nangxingqu(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['囊性区（有；无）_无',
                           '囊性区（有；无）_有']]
        nangxingqu = temp_df.values[0]
        return nangxingqu

    def get_xueliu(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[
            ['血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_I',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_II',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_III',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_IV',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_V']]
        xueliu = temp_df.values[0]
        return xueliu
        ######
    ## 旧版超声报告所用多输出
    ######
    # def get_size(self,name, df,scale):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['长径（px）','短径（px）']]
    #     sizes = temp_df.values[0]*scale
    #     return np.array(sorted(sizes,reverse=True))
    #
    # def get_jiejieleixing(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['结节类型_一低回声肿块', '结节类型_不均质回声结节',
    #                        '结节类型_低回声结节', '结节类型_多发低回声结节', '结节类型_多发淋巴结',
    #                        '结节类型_异常淋巴结', '结节类型_肿大淋巴结']]
    #     jiejieleixing = temp_df.values[0]
    #     return jiejieleixing
    #
    # def get_bianjie(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['边界_不清', '边界_尚清', '边界_欠清', '边界_清楚']]
    #     bianjie = temp_df.values[0]
    #     return bianjie
    #
    # def get_xingtai(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['形态_不规则', '形态_欠规则', '形态_结构未见异常', '形态_结构消失', '形态_规则']]
    #     xingtai = temp_df.values[0]
    #     return xingtai
    #
    # def get_pisuizhifenjie(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['皮髓质分界_不清', '皮髓质分界_欠清', '皮髓质分界_皮质增厚']]
    #     pisuizhifenjie = temp_df.values[0]
    #     return pisuizhifenjie
    #
    # def get_linbamenjiegou(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['淋巴门结构_不清', '淋巴门结构_偏心', '淋巴门结构_可见', '淋巴门结构_周边可见',
    #                        '淋巴门结构_显示不清', '淋巴门结构_显示欠清', '淋巴门结构_未见', '淋巴门结构_欠清',
    #                        '淋巴门结构_消失', '淋巴门结构_结构尚清', '淋巴门结构_部分可见', '淋巴门结构_部分未见',
    #                        '淋巴门结构_部分结节内可见', '淋巴门结构_隐约可见']]
    #     linbamenjiegou = temp_df.values[0]
    #     return linbamenjiegou
    #
    # def get_cdfi(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['CDFI示其内血流信号_分隔及周边可见', 'CDFI示其内血流信号_可见',
    #                        'CDFI示其内血流信号_可见不规则', 'CDFI示其内血流信号_可见丰富',
    #                        'CDFI示其内血流信号_可见少许', 'CDFI示其内血流信号_可见点状',
    #                        'CDFI示其内血流信号_可见稍丰富', 'CDFI示其内血流信号_可见较丰富',
    #                        'CDFI示其内血流信号_周边可见', 'CDFI示其内血流信号_周边可见少量',
    #                        'CDFI示其内血流信号_未见', 'CDFI示其内血流信号_未见异常', 'CDFI示其内血流信号_见点状',
    #                        'CDFI示其内血流信号_部分结节内可见']]
    #     cdfi = temp_df.values[0]
    #     return cdfi
    #
    # def get_beizhu(self,name, df):
    #     temp_df = df[df['编号'] == name]
    #     temp_df = temp_df[['备注_内可见多发斑状强回声', '备注_内可见多发点状强回声', '备注_内可见强回声钙化',
    #                        '备注_内可见斑状强回声', '备注_内可见点条样强回声', '备注_内可见点状强回声',
    #                        '备注_内可见透声区', '备注_内回声不均匀', '备注_内回声不均匀，可见多发点状强回声',
    #                        '备注_内回声欠均匀', '备注_内见透声区', '备注_内部回声不均匀，可见多发斑点状强回声',
    #                        '备注_内部回声不均，可见密集点状强回声', '备注_呈偏心性', '备注_皮质增厚',
    #                        '备注_纵横比异常，皮髓质结构显示不清']]
    #     beizhu = temp_df.values[0]
    #     return beizhu
