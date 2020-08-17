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
import torch.nn.functional as F
import multiprocessing as multi_p
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from skimage.io import imread, imsave
from torchvision.models import resnet
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tran_fn
from torchvision.models._utils import IntermediateLayerGetter
from pydensecrf.utils import unary_from_softmax, unary_from_labels


#######################################################################################################################
# 0 CRF
class CRFTool(object):

    @staticmethod
    def get_ratio(img, ratio_th=0.5):
        black = np.count_nonzero(img < ratio_th) + 1
        white = np.count_nonzero(img > ratio_th) + 1
        ratio = white / (white + black)
        return ratio

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

    @classmethod
    def crf_torch(cls, img, annotation, t=5):
        img_data = np.asarray(img, dtype=np.uint8)
        annotation_data = np.asarray(annotation)
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            result_one = cls.crf(img_data_one, annotation_data_one, t=t)
            # result_one = cls.crf_label(img_data_one, annotation_data_one, t=t)
            result.append(np.expand_dims(result_one, axis=0))
            pass
        return torch.tensor(np.asarray(result))

    pass


#######################################################################################################################
# 1 Data


class Eval(object):

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
            prec[i], recall[i] = tp / (y_temp.sum() + 1), tp / (y.sum() + 1)
            pass
        return prec, recall

    pass


class FixedResized(object):

    def __init__(self, img_w=300, img_h=300):
        self.img_w, self.img_h = img_w, img_h
        pass

    def __call__(self, img, label=None, image_crf=None, param=None):
        img = img.resize((self.img_w, self.img_h))
        if label is not None:
            label = label.resize((self.img_w, self.img_h))
        if image_crf is not None:
            image_crf = image_crf.resize((self.img_w, self.img_h))
        return img, label, image_crf, param

    pass


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __call__(self, img, label=None, image_crf=None, param=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = tran_fn.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if label is not None:
            label = tran_fn.resized_crop(label, i, j, h, w, self.size, self.interpolation)
        if image_crf is not None:
            image_crf = tran_fn.resized_crop(image_crf, i, j, h, w, self.size, self.interpolation)
        if param is not None:
            param["crop"] = [i, j, h, w]
        return img, label, image_crf, param

    pass


class ColorJitter(transforms.ColorJitter):

    def __call__(self, img, label=None, image_crf=None, param=None):
        img = super().__call__(img)
        return img, label, image_crf, param

    pass


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, label=None, image_crf=None, param=None):
        img = super().__call__(img)
        return img, label, image_crf, param

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, label=None, image_crf=None, param=None):
        if random.random() < self.p:
            img = tran_fn.hflip(img)
            if label is not None:
                label = tran_fn.hflip(label)
            if image_crf is not None:
                image_crf = tran_fn.hflip(image_crf)
            if param is not None:
                param["flip"] = 1
            pass
        else:
            if param is not None:
                param["flip"] = 0
        return img, label, image_crf, param

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, label=None, image_crf=None, param=None):
        img = super().__call__(img)
        if label is not None:
            label = super().__call__(label)
        if image_crf is not None:
            image_crf = super().__call__(image_crf)
        return img, label, image_crf, param

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, label=None, image_crf=None, param=None):
        img = super().__call__(img)
        return img, label, image_crf, param

    pass


class Compose(transforms.Compose):
    def __call__(self, img, label=None, image_crf=None, param=None):
        for t in self.transforms:
            img, label, image_crf, param = t(img, label, image_crf, param)
        return img, label, image_crf, param

    pass


# 训练聚类
class DatasetCAM(Dataset):

    def __init__(self, img_name_list, size_train=224):
        self.image_name_list = img_name_list

        self.transform = Compose([RandomResizedCrop(size=size_train, scale=(0.3, 1.)),
                                  ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2),
                                  RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # self.transform = Compose([FixedResized(size_train, size_train),
        #                           ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2),
        #                           RandomHorizontalFlip(), ToTensor(),
        #                           Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image, _, _, param = self.transform(image, label=None, image_crf=image, param=None)
        return image, idx

    pass


# 保存CAM结果
class DatasetCAMVIS(Dataset):

    def __init__(self, img_name_list, cam_name_list=None, size_vis=256, multi_num=16):
        self.image_name_list = img_name_list
        self.cam_name_list = cam_name_list
        self.multi_num = multi_num

        self.transform = Compose([FixedResized(size_vis, size_vis),
                                  ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2),
                                  ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_vis_simple = Compose([FixedResized(size_vis, size_vis), ToTensor(),
                                             Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.image_size_list = [Image.open(image_name).size for image_name in self.image_name_list]
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        image_simple, _, image_for_crf, _ = self.transform_vis_simple(image, label=None, image_crf=image, param=None)
        image_list = [image_simple]
        for i in range(self.multi_num - 1):
            image_now, _, _, _ = self.transform(image, label=None, image_crf=image, param=None)
            image_list.append(image_now)
            pass

        return image_list, image_for_crf, idx

    def save_cam(self, idx, cam, name=None):
        if self.cam_name_list is not None:
            h_path = self.cam_name_list[idx]
            if name is not None:
                h_path = "{}_{}{}".format(os.path.splitext(h_path)[0], name, os.path.splitext(h_path)[1])
            h_path = Tools.new_dir(h_path)

            cam = np.transpose(cam, (1, 2, 0)) if cam.shape[0] == 1 or cam.shape[0] == 3 else cam
            im = Image.fromarray(np.asarray(cam * 255, dtype=np.uint8)).resize(self.image_size_list[idx])
            im.save(h_path)
        pass

    @staticmethod
    def collate_fn(samples):
        if len(samples) == 1:
            image_list, image_for_crf, idx = samples[0]
            image = torch.cat([torch.unsqueeze(image, dim=0) for image in image_list], dim=0)
        else:
            image = torch.cat([torch.unsqueeze(sample[0][0], 0) for sample in samples], dim=0)
            image_for_crf = [sample[1] for sample in samples]
            idx = [sample[2] for sample in samples]
            pass
        return image, image_for_crf, idx

    pass


# 评估保存在图片中的结果
class DatasetEval(object):

    def __init__(self, lab_name_list, lab2_name_list, size_eval=None, th_num=25):
        self.label_name_list = lab_name_list
        self.label2_name_list = lab2_name_list
        self.size_eval = size_eval
        self.th_num = th_num
        pass

    def __len__(self):
        return len(self.label_name_list)

    def __getitem__(self, idx):
        label = Image.open(self.label_name_list[idx]).convert("L")
        label2 = Image.open(self.label2_name_list[idx]).convert("L")

        if self.size_eval is not None:
            label = label.resize((self.size_eval, self.size_eval))
            label2 = label2.resize((self.size_eval, self.size_eval))
            pass
        else:
            if label.size[0] != label2.size[0] or label.size[1] != label2.size[1]:
                label2 = label2.resize(label.size)
                pass
            pass

        label = np.asarray(label) / 255
        label2 = np.asarray(label2) / 255

        mae = Eval.eval_mae(label, label2)
        prec, recall = Eval.eval_pr(label, label2, th_num=self.th_num)
        return mae, prec, recall

    pass


# 训练USOD
class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list, cam_name_list, his_lbl_name_list,
                 his_save_lbl_name_list, size_train=224, label_a=0.2, label_b=0.5, has_crf=False):
        self.img_name_list = img_name_list
        self.lab_name_list = lab_name_list
        self.cam_name_list = cam_name_list
        self.his_lbl_name_list = his_lbl_name_list
        self.his_save_lbl_name_list = his_save_lbl_name_list

        self.label_a = label_a
        self.label_b = label_b
        self.has_crf = has_crf

        self.lbl_name_list_for_train = None
        self.lbl_name_list_for_save = self.his_save_lbl_name_list

        self.transform = Compose([FixedResized(size_train, size_train),
                                  RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.image_size_list = [Image.open(image_name).size for image_name in self.img_name_list]
        Tools.print("DatasetUSOD: size_train={}".format(size_train))
        pass

    def set_label(self, is_supervised, cam_for_train=True):
        Tools.print("DatasetUSOD change label: is_supervised={} cam_for_train={}".format(is_supervised, cam_for_train))
        if is_supervised:
            self.lbl_name_list_for_train = self.lab_name_list
        else:
            self.lbl_name_list_for_train = self.cam_name_list if cam_for_train else self.his_lbl_name_list
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

        image, label, image_for_crf, param = self.transform(image, label, image, {})

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

    def _crf_one_pool(self, pool_id, epoch, img_name_list, his_save_lbl_name_list, his_lbl_name_list):
        for i, (img_name, save_lbl_name, train_lbl_name) in enumerate(zip(
                img_name_list, his_save_lbl_name_list, his_lbl_name_list)):
            try:
                img = np.asarray(Image.open(img_name).convert("RGB"))  # 图像
                ann = np.asarray(Image.open(save_lbl_name).convert("L")) / 255  # 训练的输出
                if self.has_crf:
                    # 1
                    # E2E_R50_52_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_29.pth
                    # 2020-08-10 15:12:58 Test 29 avg mae=0.09098734770705745 score=0.7143106332672082
                    # 2020-08-10 15:16:23 Test 29 avg mae=0.06376292646269907 score=0.9013778045254351

                    # E2E_R50_53_FLoss_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_23.pth
                    # F loss
                    # 2020-08-10 23:38:15 Test 23 avg mae=0.09834574701584828 score=0.6917040112039919
                    # 2020-08-10 23:41:55 Test 23 avg mae=0.08204551649590333 score=0.8938418660886055

                    # E2E_R50_54_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_41.pth
                    # lr decay
                    # 2020-08-11 09:17:23 Test 41 avg mae=0.0886142789890432 score=0.7235586773565861
                    # 2020-08-11 09:20:53 Test 41 avg mae=0.058555378223007375 score=0.9041142600123854
                    ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    ann = (0.75 * ann + 0.25 * ann_label)

                    # 2
                    # ann_label = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    # ann = (0.75 * ann + 0.25 * ann_label)

                    # 3
                    # if epoch <= 2:
                    #     ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    #     ann = (0.75 * ann + 0.25 * ann_label)
                    # else:
                    #     ann, change = CRFTool.get_uncertain_area(ann, black_th=self.label_a,
                    #                                              white_th=self.label_b, ratio_th=16)
                    #     ann2 = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    #     ann[change] = ann2[change]
                    #     pass

                    # 4
                    # E2E_R50_55_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_26.pth
                    # 2020-08-12 04:50:16 Test 26 avg mae=0.0934651192987249 score=0.7288557277777711
                    # 2020-08-12 04:54:05 Test 26 avg mae=0.0623184949752282 score=0.900374185683203
                    # if epoch <= 20:
                    #     ann_label = CRFTool.crf(img, np.expand_dims(ann, axis=0))
                    #     ann = (0.75 * ann + 0.25 * ann_label)
                    # else:
                    #     ratio_th = int(10 - (epoch - 20) / 3)
                    #     ratio_th = ratio_th if ratio_th >= 1 else 1
                    #
                    #     ann, change = CRFTool.get_uncertain_area(ann, black_th=self.label_a,
                    #                                              white_th=self.label_b, ratio_th=ratio_th)
                    #     ann2 = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
                    #     ann[change] = ann2[change]
                    #     pass

                    pass

                imsave(Tools.new_dir(train_lbl_name), np.asarray(ann * 255, dtype=np.uint8), check_contrast=False)
            except Exception:
                Tools.print("{} {} {} {}".format(pool_id, epoch, img_name, save_lbl_name))
            pass
        pass

    def crf_dir(self, epoch=0):
        Tools.print("DatasetUSOD crf_dir form {}".format(self.his_save_lbl_name_list[0]))
        Tools.print("DatasetUSOD crf_dir   to {}".format(self.his_lbl_name_list[0]))

        pool_num = multi_p.cpu_count()
        pool = multi_p.Pool(processes=pool_num)
        one_num = len(self.img_name_list) // pool_num + 1
        for i in range(pool_num):
            img_name_list = self.img_name_list[one_num*i: one_num*(i+1)]
            his_save_lbl_name_list = self.his_save_lbl_name_list[one_num*i: one_num*(i+1)]
            his_lbl_name_list = self.his_lbl_name_list[one_num*i: one_num*(i+1)]
            pool.apply_async(self._crf_one_pool, args=(i, epoch, img_name_list,
                                                       his_save_lbl_name_list, his_lbl_name_list))
            pass
        pool.close()
        pool.join()

        Tools.print("DatasetUSOD crf_dir OVER")
        pass

    pass


# 测试USOD
class DatasetUSODTest(Dataset):

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

    pass


#######################################################################################################################
# 2 Model


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, has_relu=True, has_bn=True):
        super(ConvBlock, self).__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
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


class MICNormalize(nn.Module):

    def __init__(self, power=2):
        super(MICNormalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    pass


class MICProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
        pass

    def init(self):
        class_per_name = self.n_sample // self.out_dim
        self.class_num += class_per_name
        for i in range(self.out_dim):
            self.classes[i * class_per_name: (i + 1) * class_per_name] = i
            pass
        np.random.shuffle(self.classes)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        top_k = out.data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        return torch.tensor(self.classes[indexes.cpu().numpy()]).long()

    pass


class BASNet(nn.Module):

    def __init__(self, clustering_num=None, is_supervised_pre_train=False,
                 is_unsupervised_pre_train=True, unsupervised_pre_train_path="./pre_model/MoCov2.pth"):
        super(BASNet, self).__init__()
        self.clustering_num = 256 if clustering_num is None else clustering_num

        # -------------Encoder--------------
        backbone = resnet.__dict__["resnet50"](pretrained=is_supervised_pre_train,
                                               replace_stride_with_dilation=[False, False, False])
        if is_unsupervised_pre_train:
            backbone = self.load_unsupervised_pre_train(backbone, unsupervised_pre_train_path)
        return_layers = {'relu': 'e0', 'layer1': 'e1', 'layer2': 'e2', 'layer3': 'e3', 'layer4': 'e4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # -------------Convert-------------
        self.convert_5 = ConvBlock(2048, 512)
        self.convert_4 = ConvBlock(1024, 512)
        self.convert_3 = ConvBlock(512, 256)
        self.convert_2 = ConvBlock(256, 256)
        self.convert_1 = ConvBlock(64, 128)

        # -------------MIC-------------
        convert_dim = 2048
        self.mic_c1 = ConvBlock(convert_dim, convert_dim, has_relu=True)  # 28 32 40
        self.mic_c2 = ConvBlock(convert_dim, convert_dim, has_relu=True)
        self.mic_l1 = nn.Linear(convert_dim, self.clustering_num)
        self.mic_l2norm = MICNormalize(2)

        # -------------Decoder-------------
        self.decoder_1_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_c = ConvBlock(512, 512, has_relu=True)  # 40

        self.decoder_2_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_c = ConvBlock(512, 256, has_relu=True)  # 40

        self.decoder_3_b1 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_b2 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_c = ConvBlock(256, 256, has_relu=True)  # 80

        self.decoder_4_b1 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_b2 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_c = ConvBlock(256, 128, has_relu=True)  # 160

        self.decoder_5_b1 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_b2 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_out = nn.Conv2d(128, 1, 3, padding=1, bias=False)  # 160
        pass

    def forward(self, x, has_mic=False, has_cam=False, has_sod=False):
        result = {}

        # -------------Encoder-------------
        feature = self.backbone(x)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)

        # -------------MIC-------------
        if has_mic:
            e4 = feature["e4"]  # (2048, 10)

            mic_feature = self.mic_c2(self.mic_c1(e4))  # (512, 10)
            mic_1x1 = F.adaptive_avg_pool2d(mic_feature, output_size=(1, 1)).view((mic_feature.size()[0], -1))
            smc_logits = self.mic_l1(mic_1x1)
            smc_l2norm = self.mic_l2norm(smc_logits)

            return_mic = {"smc_logits": smc_logits, "smc_l2norm": smc_l2norm}
            result["mic"] = return_mic

            if has_cam:
                cam = self.cluster_activation_map(smc_logits, mic_feature, self.mic_l1.weight)  # 簇激活图
                cam = self._feature_norm(cam)
                cam = self._up_to_target(cam, x)

                result["cam"] = {"cam": cam}
                pass

            pass

        # -------------Decoder-------------
        if has_sod:
            # -------------Convert-------------
            e0 = self.convert_1(feature["e0"])  # 128
            e1 = self.convert_2(feature["e1"])  # 256
            e2 = self.convert_3(feature["e2"])  # 256
            e3 = self.convert_4(feature["e3"])  # 512
            e4 = self.convert_5(feature["e4"])  # 512

            # ---------------SOD---------------
            d1 = self.decoder_1_b2(self.decoder_1_b1(e4))  # 512 * 40 * 40
            d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 40 * 40

            d2 = self.decoder_2_b2(self.decoder_2_b1(d1_d2))  # 512 * 21 * 21
            d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 512 * 40 * 40

            d3 = self.decoder_3_b2(self.decoder_3_b1(d2_d3))  # 256 * 40 * 40
            d3_d4 = self._up_to_target(self.decoder_3_c(d3), e1) + e1  # 256 * 80 * 80

            d4 = self.decoder_4_b2(self.decoder_4_b1(d3_d4))  # 256 * 80 * 80
            d4_d5 = self._up_to_target(self.decoder_4_c(d4), e0) + e0  # 128 * 160 * 160

            d5 = self.decoder_5_b2(self.decoder_5_b1(d4_d5))  # 128 * 160 * 160
            d5_out = self.decoder_5_out(d5)  # 1 * 160 * 160
            d5_out_sigmoid = torch.sigmoid(d5_out)  # 1 * 160 * 160  # 小输出
            d5_out_up = self._up_to_target(d5_out, x)  # 1 * 320 * 320
            d5_out_up_sigmoid = torch.sigmoid(d5_out_up)  # 1 * 320 * 320  # 大输出

            return_result = {"out": d5_out, "out_sigmoid": d5_out_sigmoid,
                             "out_up": d5_out_up, "out_up_sigmoid": d5_out_up_sigmoid}
            result["sod"] = return_result
            pass

        return result

    @staticmethod
    def cluster_activation_map(smc_logits, mic_feature, weight_softmax):
        bz, nc, h, w = mic_feature.shape

        cam_list = []
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        for i in range(bz):
            cam_weight = weight_softmax[top_k_index[i][0]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(mic_feature[i])
            cam = torch.sum(torch.mul(cam_weight, mic_feature[i]), dim=0, keepdim=True)
            cam_list.append(torch.unsqueeze(cam, 0))
            pass
        return torch.cat(cam_list)

    @staticmethod
    def _up_to_target(source, target):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode='bilinear', align_corners=False)
            pass
        return source

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def load_unsupervised_pre_train(model, pre_train_path, change_key="module.encoder."):
        pre_train = torch.load(pre_train_path)["model"]
        checkpoint = {key.replace(change_key, ""): pre_train[key] for key in pre_train.keys() if change_key in key}
        result = model.load_state_dict(checkpoint, strict=False)
        if len(result.unexpected_keys) == 0:
            Tools.print("Success Load Unsupervised pre train from {}".format(pre_train_path))
        else:
            Tools.print("Error Load Unsupervised pre train from {}".format(pre_train_path))
            pass
        return model

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, model_dir, clustering_num, clustering_ratio,
                 img_name_list, lbl_name_list, cam_label_dir, his_label_dir, cam_label_name,
                 mic_size_train, data_name_list, has_f_loss=False, has_crf=False, mic_batch_size=8, sod_batch_size=8,
                 size_cam=256, size_sod_train=320, size_sod_test=320, multi_num=5, label_a=0.3, label_b=0.5):
        self.model_dir = model_dir

        self.mic_train_prepare = self._mic_train_prepare(
            clustering_num=clustering_num, clustering_ratio=clustering_ratio,
            img_name_list=img_name_list, size_mic=mic_size_train, batch_size=mic_batch_size)
        self.mic_vis_prepare = self._mic_vis_prepare(
            img_name_list=img_name_list, data_name_list=data_name_list, cam_label_dir=cam_label_dir,
            size_cam=size_cam, multi_num=multi_num, batch_size=mic_batch_size)
        self.sod_train_prepare = self._sod_train_prepare(
            batch_size=sod_batch_size, size_sod_train=size_sod_train, size_sod_test=size_sod_test,
            img_name_list=img_name_list, lbl_name_list=lbl_name_list, data_name_list=data_name_list,
            cam_label_dir=cam_label_dir, cam_label_name=cam_label_name, has_f_loss=has_f_loss,
            his_label_dir=his_label_dir, label_a=label_a, label_b=label_b, has_crf=has_crf)

        # Model
        self.net = BASNet(clustering_num)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        # Loss
        self.bce_loss = nn.BCELoss().cuda()
        self.mic_loss = nn.CrossEntropyLoss().cuda()
        pass

    @staticmethod
    def _mic_train_prepare(clustering_num, clustering_ratio, img_name_list, size_mic=224, batch_size=8):
        result = {}

        dataset_cam = DatasetCAM(img_name_list=img_name_list, size_train=size_mic)
        data_loader_cam = DataLoader(dataset_cam, batch_size, shuffle=True, num_workers=16)
        data_num, data_batch_num = len(dataset_cam), len(data_loader_cam)

        result["dataset"] = dataset_cam
        result["data_loader"] = data_loader_cam
        result["data_num"] = data_num
        result["data_batch_num"] = data_batch_num

        # MIC
        produce_class11 = MICProduceClass(data_num, out_dim=clustering_num, ratio=clustering_ratio)
        produce_class12 = MICProduceClass(data_num, out_dim=clustering_num, ratio=clustering_ratio)
        produce_class11.init()
        produce_class12.init()
        result["produce_class11"] = produce_class11
        result["produce_class12"] = produce_class12

        return result

    @staticmethod
    def _mic_vis_prepare(img_name_list, data_name_list, cam_label_dir, size_cam=256, multi_num=5, batch_size=4):
        result = {}

        if multi_num > 1:
            batch_size = 1

        cam_label_dir = Tools.new_dir(cam_label_dir)
        tra_cam_name_list = [os.path.join(cam_label_dir, '{}_{}.bmp'.format(dataset_name, os.path.splitext(
            os.path.basename(img_path))[0])) for img_path, dataset_name in zip(img_name_list, data_name_list)]
        dataset_cam_vis = DatasetCAMVIS(img_name_list=img_name_list, multi_num=multi_num,
                                        cam_name_list=tra_cam_name_list, size_vis=size_cam)
        data_loader_cam_vis = DataLoader(dataset_cam_vis, batch_size, shuffle=False,
                                         num_workers=4, collate_fn=dataset_cam_vis.collate_fn)
        data_num, data_batch_num = len(dataset_cam_vis), len(data_loader_cam_vis)

        result["dataset"] = dataset_cam_vis
        result["data_loader"] = data_loader_cam_vis
        result["data_num"] = data_num
        result["data_batch_num"] = data_batch_num
        result["multi_num"] = multi_num
        return result

    def _sod_train_prepare(self, batch_size=8, size_sod_train=224, size_sod_test=256, has_f_loss=False,
                           img_name_list=None, lbl_name_list=None, data_name_list=None,
                           cam_label_dir="../BASNetTemp/cam/CAM_123_224_256", cam_label_name='cam_c12',
                           his_label_dir="../BASNetTemp/his/CAM_123_224_256",
                           label_a=0.3, label_b=0.5, has_crf=False):
        result = {}

        cam_lbl_name_list, his_train_lbl_name_list, his_save_lbl_name_list = self.get_sod_img_label_name(
            img_name_list, data_name_list, cam_label_dir, cam_label_name, his_label_dir)
        result["cam_lbl_name_list"] = cam_lbl_name_list
        result["his_train_lbl_name_list"] = his_train_lbl_name_list
        result["his_save_lbl_name_list"] = his_save_lbl_name_list
        result["size_sod_test"] = size_sod_test
        result["img_name_list"] = img_name_list
        result["lbl_name_list"] = lbl_name_list

        dataset_sod = DatasetUSOD(img_name_list=img_name_list, lab_name_list=lbl_name_list,
                                  cam_name_list=cam_lbl_name_list, his_lbl_name_list=his_train_lbl_name_list,
                                  his_save_lbl_name_list=his_save_lbl_name_list, size_train=size_sod_train,
                                  label_a=label_a, label_b=label_b, has_crf=has_crf)
        data_loader_sod = DataLoader(dataset_sod, batch_size, shuffle=True, num_workers=8)
        data_batch_num = len(data_loader_sod)
        result["dataset"] = dataset_sod
        result["data_loader"] = data_loader_sod
        result["data_batch_num"] = data_batch_num
        result["has_f_loss"] = has_f_loss
        return result

    # 训练MIC
    def train_mic(self, epoch_num=100, start_epoch=0, save_epoch_freq=5, lr=0.00001, model_file_name=None):
        produce_class11 = self.mic_train_prepare["produce_class11"]
        produce_class12 = self.mic_train_prepare["produce_class12"]
        data_loader = self.mic_train_prepare["data_loader"]
        data_batch_num = self.mic_train_prepare["data_batch_num"]

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name))
            self.load_model(model_file_name)
            pass

        if start_epoch >= 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            produce_class11.reset()
            with torch.no_grad():
                for _idx, (inputs, indexes) in tqdm(enumerate(data_loader), total=data_batch_num):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()
                    result = self.net(inputs, has_mic=True, has_cam=False, has_sod=False)
                    produce_class11.cal_label(result["mic"]["smc_logits"], indexes)
                    pass
                pass
            classes = produce_class12.classes
            produce_class12.classes = produce_class11.classes
            produce_class11.classes = classes
            Tools.print("Update: [{}] 1-{}/{}".format(start_epoch, produce_class11.count, produce_class11.count_2))
            pass

        optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)

        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()

            produce_class11.reset()
            for i, (inputs, indexes) in tqdm(enumerate(data_loader), total=data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                optimizer.zero_grad()

                result = self.net(inputs, has_mic=True, has_cam=False, has_sod=False)

                ######################################################################################################
                # MIC
                produce_class11.cal_label(result["mic"]["smc_logits"], indexes)
                mic_target = result["mic"]["smc_logits"]
                mic_labels = produce_class12.get_label(indexes).cuda()

                loss = self.mic_loss_fn(mic_target, mic_labels)
                ######################################################################################################

                loss.backward()
                optimizer.step()

                all_loss += loss.item()
                pass

            Tools.print("[E:{:3d}/{:3d}] mic loss:{:.3f}".format(epoch, epoch_num, all_loss/data_batch_num))

            classes = produce_class12.classes
            produce_class12.classes = produce_class11.classes
            produce_class11.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, produce_class11.count, produce_class11.count_2))

            ###########################################################################
            # 2 保存模型
            if epoch % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "mic_{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()
                pass
            ###########################################################################

            pass

        # Final Save
        save_file_name = Tools.new_dir(os.path.join(
            self.model_dir, "mic_final_{}.pth".format(epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)

        Tools.print()
        Tools.print("Save Model to {}".format(save_file_name))
        Tools.print()
        pass

    # 获得MIC的结果CAM
    def vis_cam(self, model_file_name=None):
        dataset = self.mic_vis_prepare["dataset"]
        data_loader = self.mic_vis_prepare["data_loader"]
        data_batch_num = self.mic_vis_prepare["data_batch_num"]
        multi_num = self.mic_vis_prepare["multi_num"]

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name))
            self.load_model(model_file_name)
            pass

        self.net.eval()
        with torch.no_grad():
            for _idx, (inputs, image_for_crf, index) in tqdm(enumerate(data_loader), total=data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()

                result = self.net(inputs, has_mic=True, has_cam=True, has_sod=False)
                result = result["cam"]
                for key in result.keys():
                    result[key] = result[key].detach().cpu()
                    pass

                if multi_num > 1:
                    dataset.save_cam(idx=int(index), name="image", cam=np.asarray(image_for_crf))
                    for key in result.keys():
                        value_mean = torch.mean(result[key], dim=0)
                        dataset.save_cam(idx=int(index), name=key, cam=np.asarray(value_mean.squeeze()))

                        value_crf = CRFTool.crf_torch(torch.unsqueeze(image_for_crf, dim=0) * 255,
                                                      torch.unsqueeze(value_mean, dim=0), t=5)
                        dataset.save_cam(idx=int(index), name="{}_crf".format(key), cam=np.asarray(value_crf.squeeze()))
                        pass
                else:
                    # 不加速
                    for _i in range(inputs.shape[0]):
                        self._save_cam(dataset, index, result, image_for_crf, _i)
                        pass

                    # 加速
                    # pool_num = multi_p.cpu_count()
                    # pool_num = pool_num if pool_num < inputs.shape[0] else inputs.shape[0]
                    # pool = multi_p.Pool(processes=pool_num)
                    # for _i in range(inputs.shape[0]):
                    #     pool.apply_async(self._save_cam, args=(dataset, index, result, image_for_crf, _i))
                    #     pass
                    # pool.close()
                    # pool.join()
                    pass

                pass
            pass

        pass

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr, e=0.9, start_epoch=30):
        if epoch > start_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * np.power(e, epoch - start_epoch)
                pass
        pass

    # 训练SOD
    def train_sod(self, epoch_num=50, start_epoch=0, model_file_name=None, save_epoch_freq=1, is_supervised=False,
                  has_history=True, history_epoch_start=2, history_epoch_freq=1, lr=0.0001):
        dataset = self.sod_train_prepare["dataset"]
        data_loader = self.sod_train_prepare["data_loader"]
        data_batch_num = self.sod_train_prepare["data_batch_num"]
        has_f_loss = self.sod_train_prepare["has_f_loss"]
        size_sod_test = self.sod_train_prepare["size_sod_test"]
        img_name_list = self.sod_train_prepare["img_name_list"]
        lbl_name_list = self.sod_train_prepare["lbl_name_list"]

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name))
            self.load_model(model_file_name)
            pass

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(start_epoch, epoch_num+1):
            Tools.print()
            self._adjust_learning_rate(optimizer, epoch, lr, start_epoch=30)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 0 准备
            if epoch == 0:  # 0
                dataset.set_label(is_supervised=is_supervised, cam_for_train=True)
            elif epoch == history_epoch_start:  # 5
                dataset.set_label(is_supervised=is_supervised, cam_for_train=False)

            if not is_supervised and epoch >= history_epoch_start and \
                    (epoch - history_epoch_start) % history_epoch_freq == 0:  # 5, 10, 15
                dataset.crf_dir(epoch)
            ###########################################################################

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            for i, (inputs, targets, image_for_crf, indexes, params) in tqdm(enumerate(data_loader),
                                                                             total=data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()
                targets = targets.type(torch.FloatTensor).cuda()

                optimizer.zero_grad()
                result = self.net(inputs, has_mic=False, has_cam=False, has_sod=True)
                sod_output = result["sod"]["out_up_sigmoid"]

                loss = self.sod_loss_fn(sod_output, targets, has_f_loss)
                loss.backward()
                optimizer.step()
                all_loss += loss.item()

                ##############################################
                if has_history:
                    histories = np.asarray(targets.detach().cpu())
                    sod_output = np.asarray(sod_output.detach().cpu())

                    # 处理翻转
                    history_list, sod_output_list = [], []
                    for index, (history, sod_one) in enumerate(zip(histories, sod_output)):
                        if params["flip"][index] == 1:
                            history = np.expand_dims(np.fliplr(history[0]), 0)
                            sod_one = np.expand_dims(np.fliplr(sod_one[0]), 0)
                        history_list.append(history)
                        sod_output_list.append(sod_one)
                        pass
                    histories, sod_output = np.asarray(history_list), np.asarray(sod_output_list)

                    # 正式开始
                    for history, index in zip(sod_output, indexes):
                        dataset.save_history(idx=int(index), history=np.asarray(history.squeeze()))
                    pass
                ##############################################
                pass
            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f}".format(epoch, epoch_num, all_loss/data_batch_num))
            ###########################################################################

            ###########################################################################
            # 2 保存模型
            if (epoch + 1) % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "sod_{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()

                ###########################################################################
                # 3 评估模型
                self.eval_duts(self.net, epoch=epoch, is_test=True,
                               batch_size=data_loader.batch_size, size_test=size_sod_test)
                self.eval_by_image_label(self.net, img_name_list, lbl_name_list, epoch=epoch,
                                         size_test=size_sod_test, batch_size=data_loader.batch_size)
                # self.eval_duts(self.net, epoch=epoch, is_test=True,
                #                batch_size=data_loader.batch_size, size_test=size_sod_test)
                ###########################################################################
                pass
            ###########################################################################
            pass

        # Final Save
        save_file_name = Tools.new_dir(os.path.join(
            self.model_dir, "sod_final_{}.pth".format(epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)

        Tools.print()
        Tools.print("Save Model to {}".format(save_file_name))
        Tools.print()
        pass

    @classmethod
    def eval_duts(cls, net, epoch=0, is_test=True, size_test=256, batch_size=16, th_num=25, beta_2=0.3, save_path=None):
        which = "TE" if is_test else "TR"
        data_dir = '/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-{}'.format(which)
        image_dir, label_dir = 'DUTS-{}-Image'.format(which), 'DUTS-{}-Mask'.format(which)

        # 数据
        img_name_list = glob.glob(os.path.join(data_dir, image_dir, '*.jpg'))
        lbl_name_list = [os.path.join(data_dir, label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list]
        Tools.print("Test images: {}".format(len(lbl_name_list)))

        # EVAL
        cls.eval_by_image_label(net, img_name_list, lbl_name_list, epoch=epoch, size_test=size_test,
                                batch_size=batch_size, th_num=th_num, beta_2=beta_2, save_path=save_path)
        pass

    # 预测评估
    @staticmethod
    def eval_by_image_label(net, img_name_list, lbl_name_list, epoch=0,
                            size_test=256, batch_size=16, th_num=25, beta_2=0.3, save_path=None):
        # 数据
        save_lbl_name_list = [os.path.join(save_path, '{}.bmp'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list] if save_path else None
        dataset_eval_sod = DatasetUSODTest(img_name_list=img_name_list, save_lbl_name_list=save_lbl_name_list,
                                           lab_name_list=lbl_name_list, size_test=size_test)
        data_loader_eval_sod = DataLoader(dataset_eval_sod, batch_size, shuffle=False, num_workers=4)

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
                result = net(inputs, has_mic=False, has_cam=False, has_sod=True)
                now_pred = result["sod"]["out_up_sigmoid"].squeeze(dim=1).cpu().data.numpy()

                if save_path:
                    for history, index in zip(now_pred, indexes):
                        dataset_eval_sod.save_history(idx=int(index), history=np.asarray(history.squeeze()))
                        pass
                    pass

                mae = Eval.eval_mae(now_pred, now_label)
                prec, recall = Eval.eval_pr(now_pred, now_label, th_num)

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
        Tools.print("Test {} avg mae={} score={}".format(epoch, avg_mae, score.max()))
        pass

    # 直接评估图片形式的结果
    @staticmethod
    def eval_by_label_predict(img_name_list, lbl_name_list, data_name_list, vis_label_dir,
                              vis_label_name, size_eval=None, th_num=25, beta_2=0.3, has_data_name=True):
        if has_data_name:
            lbl2_name_list = [os.path.join(vis_label_dir, '{}_{}_{}.bmp'.format(
                data_name, os.path.splitext(os.path.basename(img_path))[0], vis_label_name)
                                           ) for img_path, data_name in zip(img_name_list, data_name_list)]
        else:
            lbl2_name_list = [os.path.join(vis_label_dir, '{}_{}.bmp'.format(
                 os.path.splitext(os.path.basename(img_path))[0], vis_label_name)
                                           ) for img_path, data_name in zip(img_name_list, data_name_list)]
            pass

        eval_cam = DatasetEval(lbl_name_list, lbl2_name_list, size_eval=size_eval, th_num=th_num)

        avg_mae = 0.0
        avg_prec = np.zeros(shape=(th_num,)) + 1e-6
        avg_recall = np.zeros(shape=(th_num,)) + 1e-6
        for i, (mae, prec, recall) in tqdm(enumerate(eval_cam), total=len(eval_cam)):
            avg_mae += mae
            avg_prec += prec
            avg_recall += recall
            pass

        avg_mae = avg_mae / len(eval_cam)
        avg_prec = avg_prec / len(eval_cam)
        avg_recall = avg_recall / len(eval_cam)
        score = (1 + beta_2) * avg_prec * avg_recall / (beta_2 * avg_prec + avg_recall)
        score[score != score] = 0
        Tools.print("Train avg mae={} score={}".format(avg_mae, score.max()))
        pass

    def load_model(self, model_file_name):
        checkpoint = torch.load(model_file_name)
        self.net.load_state_dict(checkpoint, strict=False)
        Tools.print("restore from {}".format(model_file_name))
        pass

    @staticmethod
    def get_sod_img_label_name(img_name_list, dataset_name_list, cam_label_dir, cam_label_name, his_label_dir):
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

    def mic_loss_fn(self, mic_out, mic_labels):
        loss_mic = self.mic_loss(mic_out, mic_labels)
        return loss_mic

    def sod_loss_fn(self, sod_output, sod_label, has_f_loss, ignore_label=255.0):
        positions = sod_label.view(-1, 1) != ignore_label
        loss = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        if has_f_loss:
            loss += 0.1 * self._f_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        return loss

    @staticmethod
    def _f_loss(sod_output, sod_label):
        tp = torch.sum(sod_output * sod_label)
        fp = torch.sum(sod_output * (1 - sod_label))
        tn = torch.sum((1 - sod_output) * sod_label)
        precision = tp / (tp + fp)
        recall = tp / (tp + tn)
        loss = 1 - (1 + 0.3) * (precision * recall) / (0.3 * precision + recall)
        return loss

    # 多线程加速
    @staticmethod
    def _save_cam(_dataset, _index, _result, _image_for_crf, __i):
        _dataset.save_cam(idx=int(_index[__i]), name="image", cam=np.asarray(_image_for_crf[__i]))
        for _key in _result.keys():
            _value = _result[_key][__i]
            _dataset.save_cam(idx=int(_index[__i]), name=_key, cam=np.asarray(_value.squeeze()))

            _value_crf = CRFTool.crf_torch(torch.unsqueeze(_image_for_crf[__i], dim=0) * 255,
                                           torch.unsqueeze(_value, dim=0), t=5)
            _dataset.save_cam(idx=int(_index[__i]), name="{}_crf".format(_key),
                              cam=np.asarray(_value_crf.squeeze()))
            pass
        pass

    pass


#######################################################################################################################
# 4 Main

"""
../BASNetTemp_E2E/saved_models/E2E_R50_1_CAM_12_224_256_A1_256_240_240_cam_c12_crf_H_CRF_0.3_0.5_411/sod_45.pth
2020-08-07 23:28:50 Test 45 avg mae=0.09397493337496653 score=0.7018533038193493
2020-08-07 23:32:54 Test 45 avg mae=0.06710982034617866 score=0.8916157764119035

../BASNetTemp_E2E/saved_models/E2E_R50_1_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_411/sod_47.pth
2020-08-08 04:02:57 Test 47 avg mae=0.09273115525062159 score=0.7118480586974127
2020-08-08 04:08:39 Test 47 avg mae=0.06463708462913266 score=0.8945038943200226

../BASNetTemp_E2E/saved_models/E2E_R50_2_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_211/sod_18.pth
2020-08-08 15:54:07 Test 18 avg mae=0.10198798415007865 score=0.701357815108232
2020-08-08 15:57:37 Test 18 avg mae=0.07011022425510667 score=0.8901714355930765
"""


"""
/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /media/ubuntu/4T/ALISURE/USOD/BASNet/MyTrainTest_New_3_E2E.py
2020-08-10 16:47:04 train images: 10553
2020-08-10 16:47:05 DatasetUSOD: size_train=320
2020-08-10 16:47:06 Success Load Unsupervised pre train from ./pre_model/MoCov2.pth
2020-08-10 16:47:22 Load model form ../BASNetTemp_E2E/saved_models/E2E_R50_52_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_29.pth
2020-08-10 16:47:23 restore from ../BASNetTemp_E2E/saved_models/E2E_R50_52_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_29.pth
2020-08-10 16:47:23 Begin eval CSSD 320
2020-08-10 16:47:36 Test 0 avg mae=0.06641534959467557 score=0.8964471958190837
2020-08-10 16:47:36 Begin eval ECSSD 320
2020-08-10 16:48:04 Test 0 avg mae=0.09437845854295625 score=0.8516763608617447
2020-08-10 16:48:04 Begin eval ASD 320
2020-08-10 16:48:32 Test 0 avg mae=0.03103637614006561 score=0.9324128527747216
2020-08-10 16:48:32 Begin eval MSRA10K 320
2020-08-10 16:53:06 Test 0 avg mae=0.05082927868366242 score=0.9075040025787761
2020-08-10 16:53:06 Begin eval MSRA-B 320
2020-08-10 16:55:22 Test 0 avg mae=0.05216425020521441 score=0.8920480088838162
2020-08-10 16:55:22 Begin eval SED2 320
2020-08-10 16:55:27 Test 0 avg mae=0.10561412679297584 score=0.8252037479800354
2020-08-10 16:55:27 Begin eval DUT-OMRON 320
2020-08-10 16:57:51 Test 0 avg mae=0.0764440885303836 score=0.7763795778521467
2020-08-10 16:57:51 Begin eval HKU-IS 320
2020-08-10 16:59:54 Test 0 avg mae=0.07611641488808522 score=0.8347665904223385
2020-08-10 16:59:54 Begin eval SOD 320
2020-08-10 17:00:05 Test 0 avg mae=0.18468468949982994 score=0.7361671032860957
2020-08-10 17:00:05 Begin eval THUR15000-Butterfly 320
2020-08-10 17:03:01 Test 0 avg mae=0.08388529848307372 score=0.7388173530696376
2020-08-10 17:03:01 Begin eval PASCAL1500 320
2020-08-10 17:03:44 Test 0 avg mae=0.10157587759672328 score=0.7772667603178893
2020-08-10 17:03:44 Begin eval PASCAL-S 320
2020-08-10 17:04:10 Test 0 avg mae=0.13853593846714055 score=0.7533637254895215
2020-08-10 17:04:10 Begin eval Judd 320
2020-08-10 17:04:41 Test 0 avg mae=0.13300316257957825 score=0.5950713043199602
2020-08-10 17:04:41 Begin eval DUTS-TE 320
2020-08-10 17:06:59 Test 0 avg mae=0.09104125406949004 score=0.7052402966977283
2020-08-10 17:06:59 Begin eval DUTS-TR 320
2020-08-10 17:11:37 Test 0 avg mae=0.06376214164422091 score=0.8964237915831634
2020-08-10 17:11:37 Begin eval CUB-200-2011 320
2020-08-10 17:16:54 Test 0 avg mae=0.04975149012960524 score=0.8326531540389689

Process finished with exit code 0

"""


"""
/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /media/ubuntu/4T/ALISURE/USOD/BASNet/MyTrainTest_New_3_E2E.py
2020-08-11 11:16:23 train images: 10553
2020-08-11 11:16:24 DatasetUSOD: size_train=320
2020-08-11 11:16:25 Success Load Unsupervised pre train from ./pre_model/MoCov2.pth
2020-08-11 11:17:03 Load model form ../BASNetTemp_E2E/saved_models/E2E_R50_54_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_41.pth
2020-08-11 11:17:03 restore from ../BASNetTemp_E2E/saved_models/E2E_R50_54_CAM_12_224_256_A1_256_320_320_cam_c12_crf_H_CRF_0.3_0.5_111/sod_41.pth
2020-08-11 11:17:03 Begin eval CSSD 320
2020-08-11 11:18:09 Test 0 avg mae=0.06123341863545088 score=0.8972040504684442
2020-08-11 11:18:09 Begin eval ECSSD 320
2020-08-11 11:18:36 Test 0 avg mae=0.08576142804194538 score=0.8571564643298396
2020-08-11 11:18:36 Begin eval ASD 320
2020-08-11 11:19:03 Test 0 avg mae=0.03082757583627152 score=0.9269228123321578
2020-08-11 11:19:04 Begin eval MSRA10K 320
2020-08-11 11:23:47 Test 0 avg mae=0.0467939906835556 score=0.9099989345016014
2020-08-11 11:23:47 Begin eval MSRA-B 320
2020-08-11 11:25:59 Test 0 avg mae=0.048737608300992094 score=0.8953633125581038
2020-08-11 11:25:59 Begin eval SED2 320
2020-08-11 11:26:11 Test 0 avg mae=0.08935124001332692 score=0.8409925161521241
2020-08-11 11:26:11 Begin eval DUT-OMRON 320
2020-08-11 11:28:37 Test 0 avg mae=0.07653751499351148 score=0.7753613363713566
2020-08-11 11:28:37 Begin eval HKU-IS 320
2020-08-11 11:30:36 Test 0 avg mae=0.06953853451171153 score=0.8415979395262111
2020-08-11 11:30:36 Begin eval SOD 320
2020-08-11 11:30:54 Test 0 avg mae=0.176586352288723 score=0.7390200650053829
2020-08-11 11:30:54 Begin eval THUR15000-Butterfly 320
2020-08-11 11:33:47 Test 0 avg mae=0.0827751415853317 score=0.7461239742752215
2020-08-11 11:33:47 Begin eval PASCAL1500 320
2020-08-11 11:34:30 Test 0 avg mae=0.0925332700556263 score=0.7899130985053977
2020-08-11 11:34:30 Begin eval PASCAL-S 320
2020-08-11 11:35:02 Test 0 avg mae=0.12966213292545742 score=0.7609180275204515
2020-08-11 11:35:02 Begin eval Judd 320
2020-08-11 11:35:32 Test 0 avg mae=0.13407025564658015 score=0.592518990353439
2020-08-11 11:35:32 Begin eval DUTS-TE 320
2020-08-11 11:37:47 Test 0 avg mae=0.08867833954964284 score=0.7127342376070377
2020-08-11 11:37:48 Begin eval DUTS-TR 320
2020-08-11 11:42:24 Test 0 avg mae=0.05855378706262193 score=0.8993162879195974
2020-08-11 11:42:24 Begin eval CUB-200-2011 320
2020-08-11 11:47:39 Test 0 avg mae=0.048635533903415844 score=0.8365654758500304

Process finished with exit code 0
"""


"""
2020-08-15 22:50:57 Train avg mae=0.14293060756922538 score=0.7612911469517494
E2E_R50_10_CAM_224_256_A1_256_320_320_cam_crf_H_CRF_0.3_0.5_111/sod_16.pth
2020-08-16 02:30:59 Test 16 avg mae=0.08962148620436589 score=0.7685933875567427
2020-08-16 02:34:18 Test 16 avg mae=0.054536611563093786 score=0.9062014117524748


/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /media/ubuntu/4T/ALISURE/USOD/BASNet/MyTrainTest_New_3_E2E3.py
2020-08-16 09:43:44 train images: 10553
2020-08-16 09:43:45 DatasetUSOD: size_train=320
2020-08-16 09:43:45 Success Load Unsupervised pre train from ./pre_model/MoCov2.pth
../BASNetTemp_E2E3/DUTS-TR/saved_models/E2E_R50_10_CAM_224_256_A1_256_320_320_cam_crf_H_CRF_0.3_0.5_111/sod_16.pth
2020-08-16 09:44:39 Begin eval CSSD 320
100%|██████████| 29/29 [00:27<00:00,  1.04it/s]
2020-08-16 09:45:07 Test 0 avg mae=0.057852370600248205 score=0.9031653954407721
2020-08-16 09:45:07 Begin eval ECSSD 320
100%|██████████| 143/143 [00:33<00:00,  4.31it/s]
2020-08-16 09:45:41 Test 0 avg mae=0.06928932563225915 score=0.8828413401172505
2020-08-16 09:45:41 Begin eval ASD 320
100%|██████████| 143/143 [00:22<00:00,  6.31it/s]
2020-08-16 09:46:05 Test 0 avg mae=0.039987711007934766 score=0.92431186335668
2020-08-16 09:46:05 Begin eval MSRA10K 320
100%|██████████| 1429/1429 [03:45<00:00,  6.33it/s]
2020-08-16 09:49:51 Test 0 avg mae=0.04690992082107455 score=0.9169410689455916
2020-08-16 09:49:52 Begin eval MSRA-B 320
100%|██████████| 715/715 [02:02<00:00,  5.84it/s]
2020-08-16 09:52:00 Test 0 avg mae=0.05055346848065411 score=0.9007063795704169
2020-08-16 09:52:00 Begin eval SED2 320
93%|█████████▎| 14/15 [00:02<00:00,  6.42it/s]
2020-08-16 09:52:03 Test 0 avg mae=0.09253517364462217 score=0.8254137736923006
100%|██████████| 15/15 [00:02<00:00,  5.71it/s]
2020-08-16 09:52:03 Begin eval DUT-OMRON 320
100%|██████████| 738/738 [01:55<00:00,  6.39it/s]
2020-08-16 09:54:01 Test 0 avg mae=0.09051706168931836 score=0.7869403829277833
2020-08-16 09:54:01 Begin eval HKU-IS 320
100%|██████████| 636/636 [01:40<00:00,  6.35it/s]
2020-08-16 09:55:43 Test 0 avg mae=0.06378672802059343 score=0.8653938818330627
2020-08-16 09:55:43 Begin eval SOD 320
100%|██████████| 43/43 [00:07<00:00,  6.02it/s]
2020-08-16 09:55:51 Test 0 avg mae=0.1471217947817126 score=0.7834924971540327
2020-08-16 09:55:51 Begin eval THUR15000-Butterfly 320
100%|██████████| 891/891 [02:31<00:00,  5.88it/s]
2020-08-16 09:58:30 Test 0 avg mae=0.10153399431904492 score=0.7420539700359444
2020-08-16 09:58:30 Begin eval PASCAL1500 320
100%|██████████| 215/215 [00:35<00:00,  6.14it/s]
2020-08-16 09:59:07 Test 0 avg mae=0.09048344187958296 score=0.8079889713614006
2020-08-16 09:59:07 Begin eval PASCAL-S 320
100%|██████████| 122/122 [00:19<00:00,  6.15it/s]
2020-08-16 09:59:28 Test 0 avg mae=0.11797443198681366 score=0.794574397921672
2020-08-16 09:59:28 Begin eval Judd 320
100%|██████████| 129/129 [00:26<00:00,  3.13it/s]
2020-08-16 09:59:55 Test 0 avg mae=0.16025334377159445 score=0.6015836843980746
100%|██████████| 129/129 [00:26<00:00,  4.91it/s]
2020-08-16 09:59:55 Begin eval DUTS-TE 320
100%|██████████| 717/717 [01:55<00:00,  6.21it/s]
2020-08-16 10:02:19 Test 0 avg mae=0.08966732796182858 score=0.7645120979410843
2020-08-16 10:02:19 Begin eval DUTS-TR 320
100%|██████████| 1508/1508 [03:56<00:00,  6.38it/s]
2020-08-16 10:06:17 Test 0 avg mae=0.05453140121842648 score=0.9061582202405974
2020-08-16 10:06:17 Begin eval CUB-200-2011 320
100%|██████████| 1684/1684 [04:31<00:00,  6.19it/s]
2020-08-16 10:11:32 Test 0 avg mae=0.053960402718481924 score=0.8424766254331478

Process finished with exit code 0

"""


"""
2020-08-16 15:03:04 Test 31 avg mae=0.048352293784005775 score=0.8657299014654163
2020-08-16 15:06:30 Test 31 avg mae=0.011440804680260961 score=0.9845271974918457
"""


def train(mic_batch_size, sod_batch_size):
    # 数据
    sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
    img_name_list, lbl_name_list, data_name_list = sod_data.duts_tr()
    # img_name_list, lbl_name_list, data_name_list = sod_data.msra10k()

    save_root_dir = "../BASNetTemp_E2E3/{}4".format(data_name_list[0])

    # 流程控制
    is_train = True
    has_train_mic = True
    has_save_cam = True
    has_train_sod = True
    train_sod_is_supervised = False
    train_sod_has_history = True
    mic_epoch_num = 300
    sod_epoch_num = 50

    # 参数
    # mic_size_train, size_cam, size_sod_train, size_sod_test = 224, 256, 240, 256
    mic_size_train, size_cam, size_sod_train, size_sod_test = 224, 256, 320, 320
    multi_num, label_a, label_b, has_crf, has_f_loss = 1, 0.3, 0.5, True, False

    history_epoch_start, history_epoch_freq, save_sod_epoch_freq, save_mic_epoch_freq = 1, 1, 1, 10
    cam_label_dir = "{}/cam/CAM_{}_{}_A{}".format(save_root_dir, mic_size_train, size_cam, multi_num)
    cam_label_name = 'cam_crf'

    name_model = "E2E_R50_10{}_{}_{}{}{}{}{}_{}_{}".format(
        "_FLoss" if has_f_loss else "", os.path.basename(cam_label_dir),
        "{}_{}_{}".format(size_cam, size_sod_train, size_sod_test), "_{}".format(cam_label_name),
        "_S" if train_sod_is_supervised else "", "_H" if train_sod_has_history else "",
        "_CRF" if has_crf else "",  "{}_{}".format(label_a, label_b),
        "{}{}{}".format(history_epoch_start, history_epoch_freq, save_sod_epoch_freq))

    his_label_dir = "{}/his/{}".format(save_root_dir, name_model)
    model_dir = "{}/saved_models/{}".format(save_root_dir, name_model)

    bas_runner = BASRunner(
        model_dir=model_dir, clustering_num=512, clustering_ratio=2, img_name_list=img_name_list,
        lbl_name_list=lbl_name_list, cam_label_dir=cam_label_dir, his_label_dir=his_label_dir,
        cam_label_name=cam_label_name, data_name_list=data_name_list, has_f_loss=has_f_loss, has_crf=has_crf,
        mic_batch_size=mic_batch_size, sod_batch_size=sod_batch_size, mic_size_train=mic_size_train, size_cam=size_cam,
        size_sod_train=size_sod_train, size_sod_test=size_sod_test,
        multi_num=multi_num, label_a=label_a, label_b=label_b)

    if is_train:
        # 训练MIC
        if has_train_mic:
            model_file_name = None
            # model_file_name = "../BASNetTemp_E2E/saved_models/R50_CAM_12_224_256_DFalse/mic_final_200.pth"
            bas_runner.train_mic(epoch_num=mic_epoch_num, start_epoch=0, model_file_name=model_file_name,
                                 save_epoch_freq=save_mic_epoch_freq, lr=0.0001)
            pass

        # 保存CAM
        if has_save_cam:
            model_file_name = None
            # model_file_name = "../BASNetTemp_E2E/saved_models/R50_CAM_12_224_256_DFalse/mic_final_200.pth"
            bas_runner.vis_cam(model_file_name=model_file_name)

            # 计算指标
            bas_runner.eval_by_label_predict(img_name_list, lbl_name_list, data_name_list, cam_label_dir,
                                             cam_label_name, size_eval=None, th_num=25, beta_2=0.3, has_data_name=True)
            pass

        # 训练SOD
        if has_train_sod:
            model_file_name = None
            # model_file_name = "../BASNetTemp_E2E3/DUTS-TR/saved_models/E2E_R50_10_CAM_224_256_A1_256_320_320_cam_crf_H_CRF_0.3_0.5_111/mic_final_300.pth"
            bas_runner.train_sod(epoch_num=sod_epoch_num, start_epoch=0, save_epoch_freq=save_sod_epoch_freq,
                                 is_supervised=train_sod_is_supervised, has_history=True,
                                 lr=0.0001, model_file_name=model_file_name,
                                 history_epoch_start=history_epoch_start, history_epoch_freq=history_epoch_freq)
            pass
    else:
        save_dir = "../BASNetTemp_E2E3/DUTS-TR"
        model_name = "E2E_R50_10_CAM_224_256_A1_256_320_320_cam_crf_H_CRF_0.3_0.5_111"
        model_file_name = "{}/saved_models/{}/sod_16.pth".format(save_dir, model_name)
        Tools.print("Load model form {}".format(model_file_name))
        bas_runner.load_model(model_file_name)

        sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
        for data_set in [sod_data.cssd, sod_data.ecssd, sod_data.msra_1000_asd, sod_data.msra10k,
                         sod_data.msra_b, sod_data.sed2, sod_data.dut_dmron_5166, sod_data.hku_is,
                         sod_data.sod, sod_data.thur15000, sod_data.pascal1500, sod_data.pascal_s,
                         sod_data.judd, sod_data.duts_te, sod_data.duts_tr, sod_data.cub_200_2011]:
            img_name_list, lbl_name_list, dataset_name_list = data_set()
            Tools.print("Begin eval {} {}".format(dataset_name_list[0], size_sod_test))
            bas_runner.eval_by_image_label(bas_runner.net, img_name_list, lbl_name_list,
                                           epoch=0, batch_size=sod_batch_size, size_test=size_sod_test,
                                           save_path="{}/eval/{}/{}/{}/test".format(
                                               save_dir, size_sod_test, model_name, dataset_name_list[0]))
            pass
        pass

    pass


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    _mic_batch_size = 64 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    _sod_batch_size = 7 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    train(_mic_batch_size, _sod_batch_size)
    pass
