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


#######################################################################################################################
# 0 CRF
class CRFTool(object):

    @staticmethod
    def crf(img, annotation, t=5, normalization=dcrf.NORMALIZE_SYMMETRIC):  # [3, w, h], [1, w, h]
        img = np.ascontiguousarray(img)
        annotation = np.concatenate([annotation, 1 - annotation], axis=0)

        h, w = img.shape[:2]

        d = dcrf.DenseCRF2D(w, h, 2)
        unary = unary_from_softmax(annotation)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)
        # DIAG_KERNEL     CONST_KERNEL FULL_KERNEL
        # NORMALIZE_BEFORE NORMALIZE_SYMMETRIC     NO_NORMALIZATION  NORMALIZE_AFTER
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=normalization)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10,
                               kernel=dcrf.DIAG_KERNEL, normalization=normalization)
        q = d.inference(t)

        result = np.array(q).reshape((2, h, w))
        return result[0]

    @staticmethod
    def crf_label(image, annotation, t=5, n_label=2, a=0.1, b=0.9):
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
    def crf_list(cls, img, annotation, t=5, normalization=dcrf.NORMALIZE_SYMMETRIC):
        img_data = np.asarray(img, dtype=np.uint8)
        annotation_data = np.asarray(annotation)
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            result_one = cls.crf(img_data_one, annotation_data_one, t=t, normalization=normalization)
            result.append(np.expand_dims(result_one, axis=0))
            pass
        return np.asarray(result)

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
                img = np.asarray(Image.open(img_name).convert("RGB"))
                ann = np.asarray(Image.open(save_lbl_name).convert("L"))
                ann = ann / 255

                if self.has_crf:
                    ann = CRFTool.crf_label(img, np.expand_dims(ann, axis=0), a=self.label_a, b=self.label_b)
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


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, has_relu=True):
        super(ConvBlock, self).__init__()
        self.has_relu = has_relu

        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    pass


class BASNet(nn.Module):

    def __init__(self):
        super(BASNet, self).__init__()

        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBlock(3, 64, has_relu=True)  # 64 * 224 * 224
        self.encoder1 = resnet.layer1  # 64 * 224 * 224
        self.encoder2 = resnet.layer2  # 128 * 112 * 112
        self.encoder3 = resnet.layer3  # 256 * 56 * 56
        self.encoder4 = resnet.layer4  # 512 * 28 * 28

        # -------------Decoder-------------
        self.decoder_1_b1 = ResBlock(512, 512)  # 28
        self.decoder_1_b2 = ResBlock(512, 512)  # 28
        self.decoder_1_b3 = ResBlock(512, 512)  # 28
        self.decoder_1_c = ConvBlock(512, 256, has_relu=True)  # 28

        self.decoder_2_b1 = ResBlock(256, 256)  # 56
        self.decoder_2_b2 = ResBlock(256, 256)  # 56
        self.decoder_2_b3 = ResBlock(256, 256)  # 56
        self.decoder_2_c = ConvBlock(256, 128, has_relu=True)  # 56

        self.decoder_3_b1 = ResBlock(128, 128)  # 112
        self.decoder_3_b2 = ResBlock(128, 128)  # 112
        self.decoder_3_b3 = ResBlock(128, 128)  # 112
        self.decoder_3_out = nn.Conv2d(128, 1, 3, padding=1, bias=False)  # 112
        pass

    def forward(self, x):
        # -------------Encoder-------------
        e0 = self.encoder0(x)  # 64 * 224 * 224
        e1 = self.encoder1(e0)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28

        # -------------Decoder-------------
        d1 = self.decoder_1_b3(self.decoder_1_b2(self.decoder_1_b1(e4)))  # 512 * 28 * 28
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 56 * 56

        d2 = self.decoder_2_b3(self.decoder_2_b2(self.decoder_2_b1(d1_d2)))  # 256 * 56 * 56
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 128 * 112 * 112

        d3 = self.decoder_3_b3(self.decoder_3_b2(self.decoder_3_b1(d2_d3)))  # 128 * 112 * 112
        d3_out = self.decoder_3_out(d3)  # 1 * 112 * 112
        d3_out_sigmoid = torch.sigmoid(d3_out)  # 1 * 112 * 112  # 小输出
        d3_out_up = self._up_to_target(d3_out, x)  # 1 * 224 * 224
        d3_out_up_sigmoid = torch.sigmoid(d3_out_up)  # 1 * 224 * 224  # 大输出

        return_result = {"out": d3_out, "out_sigmoid": d3_out_sigmoid,
                         "out_up": d3_out_up, "out_up_sigmoid": d3_out_up_sigmoid}
        return return_result

    @staticmethod
    def _up_to_target(source, target):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode='bilinear', align_corners=False)
            pass
        return source

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, batch_size=8, size_train=224, size_test=256, label_a=0.2, label_b=0.5, has_crf=True,
                 tra_img_name_list=None, tra_lbl_name_list=None, tra_data_name_list=None,
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
        self.learning_rate = [[0, 0.001], [70, 0.0001], [90, 0.00001]]
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
                self.dataset_sod.crf_dir()
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
    def eval(net, epoch=0, is_test=True, size_test=256, batch_size=16, th_num=100, beta_2=0.3, save_path=None):
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
2020-07-13 00:08:59 [E: 64/200] loss:0.026
2020-07-13 00:11:42  Test  64 avg mae=0.06687459443943410 score=0.8030195696294923
2020-07-13 09:57:32 Train 190 avg mae=0.02006155672962919 score=0.9667652002840796

2020-07-13 14:27:31 [E: 37/ 50] loss:0.057
2020-07-13 14:30:10  Test  37 avg mae=0.07661225112855055 score=0.7964876461003649
2020-07-13 15:46:29 Train  50 avg mae=0.03753526809089112 score=0.949905201516531


CAM_123_224_256_AVG_1 CAM_123_SOD_224_256_cam_up_norm_C123_crf
2020-07-13 21:51:39  Train 0  avg mae=0.21805560385639017 score=0.7703388524630066
2020-07-13 21:55:06   Test 0  avg mae=0.2145372756822094 score=0.4716003589535502
2020-07-13 21:44:26 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/0_train_0.414.pth
2020-07-14 05:13:23 Train 50  avg mae=0.1699602273941943 score=0.7199582214745165
2020-07-14 05:16:22  Test 50  avg mae=0.19143414872277315 score=0.4486911659550564
2020-07-14 05:16:22 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/50_train_0.121.pth

CAM_123_224_256_AVG_9 CAM_123_SOD_224_256_cam_up_norm_C123_crf
2020-07-13 21:51:51 Train   0 avg mae=0.2213046217506582 score=0.760475277575875
2020-07-13 21:55:19  Test   0 avg mae=0.24761777138634092 score=0.4727486014204659
2020-07-13 21:44:33 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/0_train_0.404.pth
2020-07-14 05:14:13 Train  50 avg mae=0.16833973647744366 score=0.7238657640463328
2020-07-14 05:16:58  Test  50 avg mae=0.18463462503377798 score=0.4538614644286766
2020-07-14 05:16:58 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/50_train_0.120.pth

CAM_123_224_256_AVG_30 CAM_123_SOD_224_256_cam_up_norm_C123_crf
2020-07-13 21:51:51 Train   0 avg mae=0.22676707558108097 score=0.7757895484733058
2020-07-13 21:55:18  Test   0 avg mae=0.22966912004408563 score=0.4839709072537858
2020-07-13 21:44:34 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/0_train_0.406.pth
2020-07-14 05:05:13 Train  50 avg mae=0.16828741925683888 score=0.7230961001281037
2020-07-14 05:08:14  Test  50 avg mae=0.18572145795366565 score=0.45241677565391286
2020-07-14 05:08:14 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf/50_train_0.120.pth


2020-07-14 12:01:09 Train  10 avg mae=0.19015622608589403 score=0.7464089826350994
2020-07-14 12:04:01  Test  10 avg mae=0.22109915031369326 score=0.4474207165369616
2020-07-14 11:55:04 Save Model to ../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf_Filter/10_train_0.262.pth

"""


"""
../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf_Filter_History_DieDai_CRF/30_train_0.011.pth
2020-07-17 15:24:13  Test 29 avg mae=0.13881954726330034 score=0.5359092284712121
2020-07-17 15:30:10 Train 29 avg mae=0.12736118257497298 score=0.7961752969320430
2020-07-17 23:05:02  Test  0 avg mae=0.14013933748761310 score=0.5356773349322078


../BASNetTemp/saved_models/CAM_123_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_CRF/30_train_0.010.pth
2020-07-17 15:24:29  Test 29 avg mae=0.14622405152411977 score=0.5431987285811558
2020-07-17 15:30:53 Train 29 avg mae=0.11897108441952503 score=0.8048967167590105
2020-07-17 23:00:01  Test  0 avg mae=0.14102674718899064 score=0.5564712463838543


../BASNetTemp/saved_models/CAM_123_SOD_320_320_cam_up_norm_C123_crf_History_DieDai_CRF/30_train_0.009.pth
2020-07-17 20:42:04  Test 29 avg mae=0.12673813816468427 score=0.5533320124713951
2020-07-17 20:51:51 Train 29 avg mae=0.14355711485400344 score=0.7856480570591794
2020-07-17 22:46:44  Test  0 avg mae=0.12388358673282490 score=0.5612494814195114

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_CRF_Label/14_train_0.102.pth
2020-07-18 21:57:49 Test 14 avg mae=0.14567803747597194 score=0.5523819198974637
2020-07-18 22:03:20 Train 14 avg mae=0.10139584287323734 score=0.8428110060199896

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_CRF_0.2_0.5/8_train_0.143.pth
2020-07-19 11:31:31 Test 8 avg mae=0.15584810364431298 score=0.5628301869443335
2020-07-19 11:39:51 Train 8 avg mae=0.11538822498087856 score=0.833913384298952

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_0.2_0.5/20_train_0.026.pth
2020-07-19 13:20:54 Test 20 avg mae=0.24891340896167163 score=0.5774716674568292
2020-07-19 13:27:42 Train 20 avg mae=0.1492173440119421 score=0.805897788240359

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_0.3_0.5/8_train_0.053.pth
2020-07-19 17:24:25 Test 8 avg mae=0.19789807427327785 score=0.5502766274410585
2020-07-19 17:31:40 Train 8 avg mae=0.12541010602462022 score=0.8155412308300685

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_CRF_0.3_0.5/14_train_0.102.pth
2020-07-19 18:32:07 Test 14 avg mae=0.13176580212636893 score=0.571132824037257
2020-07-19 18:40:08 Train 14 avg mae=0.10157277859662744 score=0.8461293087280946

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_0.2_0.5_211/8_train_0.030.pth
2020-07-19 18:39:35 Test 8 avg mae=0.27348523190453966 score=0.5844968546566901
2020-07-19 18:47:37 Train 8 avg mae=0.16903361074457116 score=0.7711723267682993

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_CRF_0.2_0.5_211/2_train_0.205.pth
2020-07-19 16:54:58 Test 2 avg mae=0.17555865894354045 score=0.5616564696108529
2020-07-19 17:01:05 Train 2 avg mae=0.13811385584148494 score=0.8328454841678359

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_224_256_cam_up_norm_C123_crf_History_DieDai_0.4_0.6_211/11_train_0.032.pth
2020-07-19 21:42:47 Test 11 avg mae=0.14603591932601748 score=0.5654083052136333
2020-07-19 21:50:41 Train 11 avg mae=0.11518737672489475 score=0.8126329613212944

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_320_320_cam_up_norm_C123_crf_History_DieDai_0.4_0.6_211/7_train_0.047.pth
2020-07-19 22:12:16 Test 7 avg mae=0.12730444034644564 score=0.5778294759011047
2020-07-19 22:25:03 Train 7 avg mae=0.12324442881526369 score=0.8176507201039612

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_320_320_cam_up_norm_C123_crf_History_DieDai_0.3_0.5_4 1 1/5_train_0.068.pth
2020-07-20 01:40:57 Test 5 avg mae=0.14707826134885194 score=0.6079573040650241
2020-07-20 01:51:13 Train 5 avg mae=0.11038006722475543 score=0.8283362765800202

../BASNetTemp/saved_models/CAM_123_AVG_1_SOD_320_320_cam_up_norm_C123_crf_History_DieDai_CRF_0.3_0.5_4 1 1/7_train_0.143.pth
2020-07-20 02:31:56 Test 7 avg mae=0.12767476480525392 score=0.6121793060648243
2020-07-20 02:41:45 Train 7 avg mae=0.1078902934931896 score=0.8558218656264965

"""


"""
1. 迭代优化: 训练多代后, 使用模型的输出作为标签
2. 端到端训练时更新SOD
+ 3. 挑选训练样本!!!  
4. 在ImageNet上训练!!!

无CRF偏向于多,但是边界不准确
有CRF偏向于少,但是边界准确
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    _size_train, _size_test = 224, 256
    _batch_size = 12 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # _size_train, _size_test = 320, 320
    # _batch_size = 8 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    _is_all_data = False
    _is_supervised = False
    _has_history = True
    _history_epoch_start, _history_epoch_freq, _save_epoch_freq = 4, 1, 1

    ####################################################################################################
    # _label_a, _label_b, _has_crf = 0.2, 0.5, False  # OK
    # _label_a, _label_b, _has_crf = 0.2, 0.5, True  # OK
    # _label_a, _label_b, _has_crf = 0.3, 0.5, False  # OK
    # _label_a, _label_b, _has_crf = 0.3, 0.5, True  # OK OK large 0.612, 0.856
    # _label_a, _label_b, _has_crf = 0.4, 0.6, False  # OK
    # _label_a, _label_b, _has_crf = 0.4, 0.6, True

    # _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_1"
    # _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_9"
    # _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_30"
    # _cam_label_name = 'cam_up_norm_C123_crf'
    ####################################################################################################

    ####################################################################################################
    _label_a, _label_b, _has_crf = 0.3, 0.5, True
    _cam_label_dir = "CAM_123_224_256_A5_SFalse_DFalse"
    _cam_label_name = 'cam_up_norm_C23_crf'
    ####################################################################################################

    _name_model = "{}_{}_{}{}{}{}_DieDai{}_{}_{}".format(
        _cam_label_dir, _size_train, _size_test, "_{}".format(_cam_label_name),
        "_Supervised" if _is_supervised else "", "_History" if _has_history else "",
        "_CRF" if _has_crf else "",  "{}_{}".format(_label_a, _label_b),
        "{}{}{}".format(_history_epoch_start, _history_epoch_freq, _save_epoch_freq))
    _his_label_dir = "../BASNetTemp/his/{}".format(_name_model)

    Tools.print()
    Tools.print(_name_model)
    Tools.print(_cam_label_name)
    Tools.print(_cam_label_dir)
    Tools.print(_his_label_dir)
    Tools.print()

    sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
    all_image, all_mask, all_dataset_name = sod_data.get_all_train_and_mask() if _is_all_data else sod_data.duts()

    bas_runner = BASRunner(batch_size=_batch_size, size_train=_size_train, size_test=_size_test,
                           cam_label_dir=_cam_label_dir, cam_label_name=_cam_label_name, his_label_dir=_his_label_dir,
                           label_a=_label_a, label_b=_label_b, has_crf=_has_crf,
                           tra_img_name_list=all_image, tra_lbl_name_list=all_mask,
                           tra_data_name_list=all_dataset_name,
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
