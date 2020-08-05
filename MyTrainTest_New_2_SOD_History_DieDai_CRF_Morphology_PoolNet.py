import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from SODData import SODData
import torch.optim as optim
from skimage import morphology
from torchvision import models
import multiprocessing as multi_p
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from skimage.io import imread, imsave
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock as ResBlock
from pydensecrf.utils import unary_from_softmax, unary_from_labels
import torchvision.utils as vutils
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import utils, functional as F


#######################################################################################################################
# 0 CRF
class CRFTool(object):

    @staticmethod
    def _get_k1_k2(img, black_th=0.25, white_th=0.75, ratio_th=16):
        black = np.count_nonzero(img < black_th) + 1
        white = np.count_nonzero(img > white_th) + 1
        ratio = black / white
        if ratio > 1:  # 物体小
            ratio = int(ratio)
            ratio = ratio_th if ratio > ratio_th else ratio
            k1 = (ratio_th + ratio) // 2  # 多膨胀
            k2 = (ratio_th - ratio) // 2  # 少腐蚀
        else:  # 物体大
            ratio = int(1 / ratio)
            ratio = ratio_th if ratio > ratio_th else ratio
            k1 = (ratio_th - ratio) // 2  # 少膨胀
            k2 = (ratio_th + ratio) // 2  # 多腐蚀
            pass
        return k1, k2

    @classmethod
    def get_uncertain_area(cls, annotation, black_th=0.3, white_th=0.5, ratio_th=16):
        annotation = np.copy(annotation)
        k1, k2 = cls._get_k1_k2(annotation, black_th=black_th, white_th=white_th, ratio_th=ratio_th)
        result1 = morphology.closing(annotation, morphology.disk(3))  # 闭运算：消除噪声
        result2 = morphology.dilation(result1, morphology.disk(k1))  # 膨胀：增加不确定边界
        result3 = morphology.erosion(result1, morphology.disk(k2))  # 腐蚀：增加不确定边界

        other = (black_th + white_th) / 2
        result_annotation = np.zeros_like(annotation) + other
        result_annotation[result2 < black_th] = 0  # black
        result_annotation[result3 > white_th] = 1  # white
        change = result_annotation == other
        return result_annotation, change

    @staticmethod
    def crf(image, annotation, t=5, n_label=2, a=0.1, b=0.9):  # [3, w, h], [1, w, h]
        image = np.ascontiguousarray(image)
        annotation = np.concatenate([annotation, 1 - annotation], axis=0)
        h, w = image.shape[:2]

        d = dcrf.DenseCRF2D(w, h, 2)
        unary = unary_from_softmax(annotation)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(image), compat=10)
        q = d.inference(t)

        result = np.array(q).reshape((2, h, w))
        return result[0]

    @staticmethod
    def crf_label(image, annotation, t=5, n_label=2, a=0.3, b=0.5):
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        annotation = np.squeeze(np.array(annotation))

        a, b = (a * 255, b * 255) if np.max(annotation) > 1 else (a, b)
        label_extend = np.zeros_like(annotation, dtype=np.int)
        label_extend[annotation >= b] = 2
        label_extend[annotation <= a] = 1
        _, label = np.unique(label_extend, return_inverse=True)

        d = dcrf.DenseCRF2D(w, h, n_label)
        u = unary_from_labels(label, n_label, gt_prob=0.7, zero_unsure=True)
        u = np.ascontiguousarray(u)
        d.setUnaryEnergy(u)
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=np.copy(image), compat=10)
        q = d.inference(t)
        map_result = np.argmax(q, axis=0)
        result = map_result.reshape((h, w))
        return result

    pass


#######################################################################################################################
# 1 Data


class FixedResized(object):

    def __init__(self, img_w=300, img_h=300):
        self.img_w, self.img_h = img_w, img_h
        pass

    def __call__(self, img, label, image_crf=None, param=None):
        img = img.resize((self.img_w, self.img_h))
        label = label.resize((self.img_w, self.img_h))
        if image_crf is not None:
            image_crf = image_crf.resize((self.img_w, self.img_h))
        return img, label, image_crf, param

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, image_crf=None, param=None):
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
            if image_crf is not None:
                image_crf = transforms.functional.hflip(image_crf)
            if param is not None:
                param.append(1)
            pass
        else:
            if param is not None:
                param.append(0)
        return img, label, image_crf, param

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, label, image_crf=None, param=None):
        img = super().__call__(img)
        label = super().__call__(label)
        if image_crf is not None:
            image_crf = super().__call__(image_crf)
        return img, label, image_crf, param

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, label, image_crf=None, param=None):
        img = super().__call__(img)
        return img, label, image_crf, param

    pass


class Compose(transforms.Compose):
    def __call__(self, img, label, image_crf=None, param=None):
        for t in self.transforms:
            img, label, image_crf, param = t(img, label, image_crf, param)
        return img, label, image_crf, param

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list, cam_lbl_name_list, his_train_lbl_name_list,
                 his_save_lbl_name_list, size_train=224, label_a=0.2, label_b=0.5, has_crf=False):
        self.img_name_list = img_name_list
        self.tra_lab_name_list = lab_name_list
        self.cam_lbl_name_list = cam_lbl_name_list
        self.his_train_lbl_name_list = his_train_lbl_name_list
        self.his_save_lbl_name_list = his_save_lbl_name_list
        self.transform = Compose([FixedResized(size_train, size_train), RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.label_a = label_a
        self.label_b = label_b
        self.has_crf = has_crf

        self.lbl_name_list_for_train = None
        self.lbl_name_list_for_save = self.his_save_lbl_name_list

        self.image_size_list = [Image.open(image_name).size for image_name in self.img_name_list]
        Tools.print("DatasetUSOD: size_train={}".format(size_train))
        pass

    def set_label(self, is_supervised, cam_for_train=True):
        Tools.print("DatasetUSOD change label: is_supervised={} cam_for_train={}".format(is_supervised, cam_for_train))
        if is_supervised:
            self.lbl_name_list_for_train = self.tra_lab_name_list
        else:
            self.lbl_name_list_for_train = self.cam_lbl_name_list if cam_for_train else self.his_train_lbl_name_list
            pass
        pass

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        assert self.lbl_name_list_for_train is not None

        image = Image.open(self.img_name_list[idx]).convert("RGB")
        if os.path.exists(self.lbl_name_list_for_train[idx]):
            label = Image.open(self.lbl_name_list_for_train[idx]).convert("L")
        else:
            label = Image.fromarray(np.zeros_like(np.asarray(image), dtype=np.uint8)).convert("L")
            pass

        image, label, image_for_crf, param = self.transform(image, label, image, [])

        # 重要参数1: 处理 label
        label_final = torch.zeros_like(label)
        label_final[label >= self.label_b] = 1
        label_final[label <= self.label_a] = 0
        label_final[(self.label_a < label) & (label < self.label_b)] = 255

        return image, label_final, image_for_crf, idx, param

    def save_history(self, history, idx):
        h_path = self.lbl_name_list_for_save[idx]
        h_path = Tools.new_dir(h_path)

        history = np.asarray(np.squeeze(history) * 255, dtype=np.uint8)
        im = Image.fromarray(history).resize(self.image_size_list[idx])
        im.save(h_path)
        pass

    def _crf_one_pool(self, pool_id, epoch, img_name_list, his_save_lbl_name_list, his_train_lbl_name_list):
        for i, (img_name, save_lbl_name, train_lbl_name) in enumerate(zip(
                img_name_list, his_save_lbl_name_list, his_train_lbl_name_list)):
            try:
                img = np.asarray(Image.open(img_name).convert("RGB"))  # 图像
                ann = np.asarray(Image.open(save_lbl_name).convert("L")) / 255  # 训练的输出
                if self.has_crf:
                    # 0.0001
                    # 2_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
                    # 2020-07-31 01:21:26 Test 29 avg mae=0.10352051467412994 score=0.6654844658526717
                    # 2020-07-31 01:24:02 Train 29 avg mae=0.07711194122040814 score=0.8706896792050232
                    # ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    # ann = (0.75 * ann + 0.25 * ann_label)

                    # 不确定区域; 对不确定区域进行CRF; 修改不确定区域
                    # 1_Morphology_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
                    # 2020-08-01 13:09:27 Test 27 avg mae=0.11535412584032331 score=0.648697955898359
                    # 2020-08-01 13:11:38 Train 27 avg mae=0.07713918824764815 score=0.8691519831500819
                    # if epoch <= 10:
                    #     ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    #     ann = (0.75 * ann + 0.25 * ann_label)
                    # else:
                    #     ann, change = CRFTool.get_uncertain_area(ann, black_th=self.label_a,
                    #                                              white_th=self.label_b, ratio_th=16)
                    #     ann2 = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    #     ann[change] = ann2[change]
                    #     pass

                    # 2_Morphology_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
                    # 2020-08-01 22:47:48 Test 25 avg mae=0.12813315100613096 score=0.6673295398451972
                    # 2020-08-01 22:49:49 Train 25 avg mae=0.08134926897896962 score=0.872017064360255
                    # if epoch <= 10:
                    #     ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    #     ann = (0.75 * ann + 0.25 * ann_label)
                    # else:
                    #     ann, change = CRFTool.get_uncertain_area(ann, black_th=self.label_a,
                    #                                              white_th=self.label_b, ratio_th=10)
                    #     ann2 = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    #     ann[change] = ann2[change]

                    # 3_Morphology_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
                    # 2020-08-02 01:38:45 Test 15 avg mae=0.1323865410827455 score=0.6618741786757457
                    # 2020-08-02 01:40:51 Train 15 avg mae=0.08220166247338057 score=0.8720659252577703
                    # if epoch <= 2:
                    #     ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    #     ann = (0.75 * ann + 0.25 * ann_label)
                    # else:
                    #     ann, change = CRFTool.get_uncertain_area(ann, black_th=self.label_a,
                    #                                              white_th=self.label_b, ratio_th=10)
                    #     ann2 = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    #     ann[change] = ann2[change]

                    # 1_PoolNet_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
                    ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    ann = (0.75 * ann + 0.25 * ann_label)
                    pass

                imsave(Tools.new_dir(train_lbl_name), np.asarray(ann * 255, dtype=np.uint8), check_contrast=False)
            except Exception:
                Tools.print("{} {} {} {}".format(pool_id, epoch, img_name, save_lbl_name))
            pass
        pass

    def crf_dir(self, epoch=0):
        Tools.print("DatasetUSOD crf_dir form {}".format(self.his_save_lbl_name_list[0]))
        Tools.print("DatasetUSOD crf_dir   to {}".format(self.his_train_lbl_name_list[0]))

        pool_num = multi_p.cpu_count()
        pool = multi_p.Pool(processes=pool_num)
        one_num = len(self.img_name_list) // pool_num + 1
        for i in range(pool_num):
            img_name_list = self.img_name_list[one_num*i: one_num*(i+1)]
            his_save_lbl_name_list = self.his_save_lbl_name_list[one_num*i: one_num*(i+1)]
            his_train_lbl_name_list = self.his_train_lbl_name_list[one_num*i: one_num*(i+1)]
            pool.apply_async(self._crf_one_pool, args=(i, epoch, img_name_list,
                                                       his_save_lbl_name_list, his_train_lbl_name_list))
            pass
        pool.close()
        pool.join()

        Tools.print("DatasetUSOD crf_dir OVER")
        pass

    pass


class DatasetEvalUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list, save_lbl_name_list, size_test=256):
        self.image_name_list = img_name_list
        self.label_name_list = lab_name_list
        self.transform = Compose([FixedResized(size_test, size_test), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.save_lbl_name_list = save_lbl_name_list
        self.image_size_list = [Image.open(image_name).size for image_name in self.image_name_list]
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        label = Image.open(self.label_name_list[idx]).convert("L")
        image, label, image_for_crf, _ = self.transform(image, label, image, None)
        return image, label, image_for_crf, idx

    def save_history(self, history, idx):
        h_path = self.save_lbl_name_list[idx]
        h_path = Tools.new_dir(h_path)

        history = np.asarray(np.squeeze(history) * 255, dtype=np.uint8)
        im = Image.fromarray(history).resize(self.image_size_list[idx])
        im.save(h_path)
        pass

    @staticmethod
    def eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def eval_pr(y_pred, y, th_num):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    pass


#######################################################################################################################
# 2 Model


class ConvBNReLU(nn.Module):

    def __init__(self, cin, cout, stride=1, ks=3, has_relu=True, has_bn=True, bias=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, is_not_last, has_bn=True):
        super(DeepPoolLayer, self).__init__()
        self.is_not_last = is_not_last

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv1 = ConvBNReLU(k, k, ks=3, stride=1, has_relu=False, has_bn=has_bn, bias=False)
        self.conv2 = ConvBNReLU(k, k, ks=3, stride=1, has_relu=False, has_bn=has_bn, bias=False)
        self.conv3 = ConvBNReLU(k, k, ks=3, stride=1, has_relu=False, has_bn=has_bn, bias=False)

        self.conv_sum = ConvBNReLU(k, k_out, ks=3, stride=1, has_relu=False, has_bn=has_bn, bias=False)
        if self.is_not_last:
            self.conv_sum_c = ConvBNReLU(k_out, k_out, ks=3, stride=1, has_relu=False, has_bn=has_bn, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, x, x2=None):
        x_size = x.size()

        y1 = self.conv1(self.pool2(x))
        y2 = self.conv2(self.pool4(x))
        y3 = self.conv3(self.pool8(x))
        res = torch.add(x, F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True))
        res = self.relu(res)
        res = self.conv_sum(res)

        if self.is_not_last:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)
            res = self.conv_sum_c(torch.add(res, x2))
        return res

    pass


class BASNet(nn.Module):

    def __init__(self, has_bn=True):
        super(BASNet, self).__init__()

        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBNReLU(3, 64, has_relu=True)  # 64 * 224 * 224
        self.encoder1 = resnet.layer1  # 64 * 224 * 224
        self.encoder2 = resnet.layer2  # 128 * 112 * 112
        self.encoder3 = resnet.layer3  # 256 * 56 * 56
        self.encoder4 = resnet.layer4  # 512 * 28 * 28

        # -------------Decoder-------------
        # DEEP POOL
        self.deep_pool3 = DeepPoolLayer(512, 256, True, has_bn=has_bn)
        self.deep_pool2 = DeepPoolLayer(256, 128, True, has_bn=has_bn)
        self.deep_pool1 = DeepPoolLayer(128, 128, False, has_bn=has_bn)
        # ScoreLayer
        self.score = nn.Conv2d(128, 1, 1, 1)
        pass

    def forward(self, x):
        # -------------Encoder-------------
        e0 = self.encoder0(x)  # 64 * 224 * 224
        e1 = self.encoder1(e0)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28

        # -------------Decoder-------------
        # DEEP POOL
        merge = self.deep_pool3(e4, e3)  # A + F
        merge = self.deep_pool2(merge, e2)  # A + F
        merge = self.deep_pool1(merge)  # A

        # ScoreLayer
        out = self.score(merge)
        out_sigmoid = torch.sigmoid(out)  # 1 * 112 * 112  # 小输出
        out_up = self._up_to_target(out, x)  # 1 * 224 * 224
        out_up_sigmoid = torch.sigmoid(out_up)  # 1 * 224 * 224  # 大输出
        return_result = {"out": out, "out_sigmoid": out_sigmoid,
                         "out_up": out_up, "out_up_sigmoid": out_up_sigmoid}
        return return_result

    @staticmethod
    def _up_to_target(source, target):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode='bilinear', align_corners=True)
            pass
        return source

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, batch_size=8, size_train=224, size_test=256, label_a=0.2, label_b=0.5, has_crf=True,
                 tra_img_name_list=None, tra_lbl_name_list=None, tra_data_name_list=None, learning_rate=None,
                 cam_label_dir="../BASNetTemp/cam/CAM_123_224_256", cam_label_name='cam_up_norm_C123',
                 his_label_dir="../BASNetTemp/his/CAM_123_224_256", model_dir="./saved_models/model"):
        self.batch_size = batch_size
        self.size_train = size_train
        self.size_test = size_test

        self.label_a = label_a
        self.label_b = label_b
        self.has_crf = has_crf

        # Dataset
        self.model_dir = model_dir
        self.img_name_list = tra_img_name_list
        self.lbl_name_list = tra_lbl_name_list
        self.dataset_name_list = tra_data_name_list
        (self.cam_lbl_name_list, self.his_train_lbl_name_list,
         self.his_save_lbl_name_list) = self.get_tra_img_label_name(self.img_name_list, self.dataset_name_list,
                                                                    cam_label_dir, cam_label_name, his_label_dir)
        self.dataset_sod = DatasetUSOD(
            img_name_list=self.img_name_list, lab_name_list=self.lbl_name_list,
            cam_lbl_name_list=self.cam_lbl_name_list, his_train_lbl_name_list=self.his_train_lbl_name_list,
            his_save_lbl_name_list=self.his_save_lbl_name_list, size_train=self.size_train,
            label_a=self.label_a, label_b=self.label_b, has_crf=self.has_crf)
        self.data_loader_sod = DataLoader(self.dataset_sod, self.batch_size, shuffle=True, num_workers=8)
        self.data_batch_num = len(self.data_loader_sod)

        # Model
        self.net = BASNet()
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        # Loss and optimizer
        self.bce_loss = nn.BCELoss().cuda()
        self.learning_rate = [[0, 0.0001], [20, 0.00001]] if learning_rate is None else learning_rate
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate[0][1])
        pass

    def _adjust_learning_rate(self, epoch):
        for param_group in self.optimizer.param_groups:
            for lr in self.learning_rate:
                if epoch == lr[0]:
                    learning_rate = lr[1]
                    param_group['lr'] = learning_rate
            pass
        pass

    def load_model(self, model_file_name):
        checkpoint = torch.load(model_file_name)
        # checkpoint = {key: checkpoint[key] for key in checkpoint.keys() if "_c1." not in key}
        self.net.load_state_dict(checkpoint, strict=False)
        Tools.print("restore from {}".format(model_file_name))
        pass

    @staticmethod
    def get_tra_img_label_name( img_name_list, dataset_name_list, cam_label_dir, cam_label_name, his_label_dir):
        cam_lbl_name_list = [os.path.join(
            cam_label_dir, '{}_{}_{}.bmp'.format(dataset_name, os.path.splitext(os.path.basename(
                img_path))[0], cam_label_name)) for img_path, dataset_name in zip(img_name_list, dataset_name_list)]

        his_train_lbl_name_list = [os.path.join(
            his_label_dir, "train", '{}_{}_{}.bmp'.format(dataset_name, os.path.splitext(os.path.basename(
                img_path))[0], cam_label_name)) for img_path, dataset_name in zip(img_name_list, dataset_name_list)]
        his_save_lbl_name_list = [os.path.join(
            his_label_dir, "save", '{}_{}_{}.bmp'.format(dataset_name, os.path.splitext(os.path.basename(
                img_path))[0], cam_label_name)) for img_path, dataset_name in zip(img_name_list, dataset_name_list)]

        Tools.print("train images: {}".format(len(img_name_list)))
        return cam_lbl_name_list, his_train_lbl_name_list, his_save_lbl_name_list

    def all_loss_fusion(self, sod_output, sod_label, ignore_label=255.0):
        positions = sod_label.view(-1, 1) != ignore_label
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        return loss_bce

    def save_histories(self, histories, indexes):
        for history, index in zip(histories, indexes):
            self.dataset_sod.save_history(idx=int(index), history=np.asarray(history.squeeze()))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=2,
              is_supervised=False, has_history=False, history_epoch_start=10, history_epoch_freq=10):
        all_loss = 0
        for epoch in range(start_epoch, epoch_num+1):
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 0 准备
            if epoch == 0:  # 0
                self.dataset_sod.set_label(is_supervised=is_supervised, cam_for_train=True)
            elif epoch == history_epoch_start:  # 5
                self.dataset_sod.set_label(is_supervised=is_supervised, cam_for_train=False)

            if epoch >= history_epoch_start and (epoch - history_epoch_start) % history_epoch_freq == 0:  # 5, 10, 15
                self.dataset_sod.crf_dir(epoch)
            ###########################################################################

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            for i, (inputs, targets, image_for_crf, indexes, params) in tqdm(enumerate(self.data_loader_sod),
                                                                             total=self.data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()
                targets = targets.type(torch.FloatTensor).cuda()

                self.optimizer.zero_grad()
                return_m = self.net(inputs)
                sod_output = return_m["out_up_sigmoid"]

                loss = self.all_loss_fusion(sod_output, targets)
                loss.backward()
                self.optimizer.step()
                all_loss += loss.item()

                ##############################################
                if has_history:
                    histories = np.asarray(targets.detach().cpu())
                    sod_output = np.asarray(sod_output.detach().cpu())

                    # 处理翻转
                    history_list, sod_output_list = [], []
                    for index, (history, sod_one) in enumerate(zip(histories, sod_output)):
                        if params[0][index] == 1:
                            history = np.expand_dims(np.fliplr(history[0]), 0)
                            sod_one = np.expand_dims(np.fliplr(sod_one[0]), 0)
                        history_list.append(history)
                        sod_output_list.append(sod_one)
                        pass
                    histories, sod_output = np.asarray(history_list), np.asarray(sod_output_list)

                    # 正式开始
                    self.save_histories(indexes=indexes, histories=sod_output)
                    pass
                ##############################################
                pass
            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f}".format(epoch, epoch_num, all_loss/self.data_batch_num))
            ###########################################################################

            ###########################################################################
            # 2 保存模型
            if (epoch + 1) % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "{}_train_{:.3f}.pth".format(epoch, all_loss/self.data_batch_num)))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()

                ###########################################################################
                # 3 评估模型
                self.eval(self.net, epoch=epoch, is_test=True, batch_size=self.batch_size, size_test=self.size_test)
                self.eval(self.net, epoch=epoch, is_test=False, batch_size=self.batch_size, size_test=self.size_test)
                ###########################################################################
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            # self.eval(self.net, epoch=epoch, is_test=True, batch_size=self.batch_size, size_test=self.size_test)
            ###########################################################################
            pass

        # Final Save
        save_file_name = Tools.new_dir(os.path.join(
            self.model_dir, "{}_train_{:.3f}.pth".format(epoch_num, all_loss/self.data_batch_num)))
        torch.save(self.net.state_dict(), save_file_name)

        Tools.print()
        Tools.print("Save Model to {}".format(save_file_name))
        Tools.print()
        pass

    @staticmethod
    def eval(net, epoch=0, is_test=True, size_test=256, batch_size=16, th_num=25, beta_2=0.3, save_path=None):
        which = "TE" if is_test else "TR"
        data_dir = '/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-{}'.format(which)
        image_dir, label_dir = 'DUTS-{}-Image'.format(which), 'DUTS-{}-Mask'.format(which)

        # 数据
        img_name_list = glob.glob(os.path.join(data_dir, image_dir, '*.jpg'))
        lbl_name_list = [os.path.join(data_dir, label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list]
        save_lbl_name_list = [os.path.join(save_path, '{}.bmp'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list] if save_path else None
        dataset_eval_sod = DatasetEvalUSOD(img_name_list=img_name_list, save_lbl_name_list=save_lbl_name_list,
                                           lab_name_list=lbl_name_list, size_test=size_test)
        data_loader_eval_sod = DataLoader(dataset_eval_sod, batch_size, shuffle=False, num_workers=24)

        # 执行
        avg_mae = 0.0
        avg_prec = np.zeros(shape=(th_num,)) + 1e-6
        avg_recall = np.zeros(shape=(th_num,)) + 1e-6
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels, _, indexes) in tqdm(enumerate(data_loader_eval_sod),
                                                        total=len(data_loader_eval_sod)):
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs

                now_label = labels.squeeze().data.numpy()
                return_m = net(inputs)

                now_pred = return_m["out_up_sigmoid"].squeeze().cpu().data.numpy()

                if save_path:
                    for history, index in zip(now_pred, indexes):
                        dataset_eval_sod.save_history(idx=int(index), history=np.asarray(history.squeeze()))
                        pass
                    pass

                mae = dataset_eval_sod.eval_mae(now_pred, now_label)
                prec, recall = dataset_eval_sod.eval_pr(now_pred, now_label, th_num)

                avg_mae += mae
                avg_prec += prec
                avg_recall += recall
                pass
            pass

        # 结果
        avg_mae = avg_mae / len(data_loader_eval_sod)
        avg_prec = avg_prec / len(data_loader_eval_sod)
        avg_recall = avg_recall / len(data_loader_eval_sod)
        score = (1 + beta_2) * avg_prec * avg_recall / (beta_2 * avg_prec + avg_recall)
        score[score != score] = 0
        Tools.print("{} {} avg mae={} score={}".format("Test" if is_test else "Train", epoch, avg_mae, score.max()))
        pass

    pass


#######################################################################################################################
# 4 Main


"""
2020-07-13 00:11:42  Test  64 avg mae=0.06687459443943410 score=0.8030195696294923
2020-07-13 09:57:32 Train 190 avg mae=0.02006155672962919 score=0.9667652002840796
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    _size_train, _size_test = 224, 256
    _batch_size = 16 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    _is_all_data = False
    _is_supervised = False
    _has_history = True

    ####################################################################################################
    _history_epoch_start, _history_epoch_freq, _save_epoch_freq = 2, 1, 1
    _label_a, _label_b, _has_crf = 0.3, 0.5, True

    _learning_rate = [[0, 0.0001], [20, 0.0001]]
    _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse"
    _cam_label_name = 'cam_up_norm_C23_crf'
    ####################################################################################################

    _name_model = "1_PoolNet_Train_{}_{}_{}{}{}{}_DieDai{}_{}_{}".format(
        os.path.basename(_cam_label_dir), _size_train, _size_test, "_{}".format(_cam_label_name),
        "_Supervised" if _is_supervised else "", "_History" if _has_history else "",
        "_CRF" if _has_crf else "",  "{}_{}".format(_label_a, _label_b),
        "{}{}{}".format(_history_epoch_start, _history_epoch_freq, _save_epoch_freq))
    _his_label_dir = "../BASNetTemp/his2/{}".format(_name_model)

    Tools.print()
    Tools.print(_name_model)
    Tools.print(_cam_label_name)
    Tools.print(_cam_label_dir)
    Tools.print(_his_label_dir)
    Tools.print()

    sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
    all_image, all_mask, all_dataset_name = sod_data.get_all_train_and_mask() if _is_all_data else sod_data.duts_tr()

    bas_runner = BASRunner(batch_size=_batch_size, size_train=_size_train, size_test=_size_test,
                           cam_label_dir=_cam_label_dir, cam_label_name=_cam_label_name, his_label_dir=_his_label_dir,
                           label_a=_label_a, label_b=_label_b, has_crf=_has_crf,
                           tra_img_name_list=all_image, tra_lbl_name_list=all_mask,
                           tra_data_name_list=all_dataset_name, learning_rate=_learning_rate,
                           model_dir="../BASNetTemp/saved_models/{}".format(_name_model))

    # bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256/930_train_1.172.pth")
    # bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256_DTrue/1000_train_1.072.pth")
    bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256_DFalse/1000_train_1.154.pth")
    bas_runner.train(epoch_num=30, start_epoch=0, history_epoch_start=_history_epoch_start,
                     history_epoch_freq=_history_epoch_freq, is_supervised=_is_supervised, has_history=_has_history)

    # _model_name = "CAM_123_SOD_224_256_cam_up_norm_C123_crf_Filter_History_DieDai_CRF"
    # bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/{}/30_train_0.011.pth".format(_model_name))
    # bas_runner.eval(bas_runner.net, epoch=0, is_test=True, batch_size=_batch_size, size_test=_size_test,
    #                 save_path="/media/ubuntu/4T/ALISURE/USOD/BASNetTemp/his/{}/test".format(_model_name))
    pass
