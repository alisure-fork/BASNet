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

    def __init__(self, img_name_list, lab_name_list, size_train=224):
        self.img_name_list = img_name_list
        self.lab_name_list = lab_name_list
        self.transform = Compose([FixedResized(size_train, size_train), RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.image_size_list = [Image.open(image_name).size for image_name in self.img_name_list]
        Tools.print("DatasetUSOD: size_train={}".format(size_train))
        pass

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_name_list[idx]).convert("RGB")
        label = Image.open(self.lab_name_list[idx]).convert("L")
        image, label, image_for_crf, param = self.transform(image, label, image, [])
        return image, label, image_for_crf, idx, param
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

    def __init__(self, batch_size=8, size_train=224, size_test=256, is_f_loss=False, tra_img_name_list=None,
                 tra_lbl_name_list=None, learning_rate=None, model_dir="./saved_models/model"):
        self.batch_size = batch_size
        self.size_train = size_train
        self.size_test = size_test
        self.is_f_loss = is_f_loss

        # Dataset
        self.model_dir = model_dir
        self.img_name_list = tra_img_name_list
        self.lbl_name_list = tra_lbl_name_list
        self.dataset_sod = DatasetUSOD(
            img_name_list=self.img_name_list, lab_name_list=self.lbl_name_list, size_train=self.size_train)
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

    def all_loss_fusion(self, sod_output, sod_label, ignore_label=255.0):
        positions = sod_label.view(-1, 1) != ignore_label
        if self.is_f_loss:
            loss_bce = self.f_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        else:
            loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        return loss_bce

    @staticmethod
    def f_loss(sod_output, sod_label):
        tp = torch.sum(sod_output * sod_label)
        fp = torch.sum(sod_output * (1 - sod_label))
        tn = torch.sum((1 - sod_output) * sod_label)
        precision = tp / (tp + fp)
        recall = tp / (tp + tn)
        loss = 1 - (1 + 0.3) * (precision * recall) / (0.3 * precision + recall)
        return loss

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=2):
        all_loss = 0
        for epoch in range(start_epoch, epoch_num+1):
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

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
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            self.eval(self.net, epoch=epoch, is_test=True, batch_size=self.batch_size, size_test=self.size_test)
            self.eval(self.net, epoch=epoch, is_test=False, batch_size=self.batch_size, size_test=self.size_test)
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
3_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211
../BASNetTemp/saved_models/1_NewTrain_train_224_256__Supervised/29_train_0.090.pth 
2020-08-03 20:53:42 Test 29 avg mae=0.12678796203830575 score=0.6612264767707188
2020-08-03 20:55:53 Train 29 avg mae=0.09095999124375256 score=0.8653318802792803 
"""


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    _size_train, _size_test = 224, 256
    _batch_size = 16 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    _is_supervised = True
    _is_f_loss = False

    ####################################################################################################
    _learning_rate = [[0, 0.000001], [40, 0.0000001]]
    _label_name = "4_Morphology_Train_CAM_123_224_256_A5_SFalse_DFalse_224_256_cam_up_norm_C23_crf_History_DieDai_CRF_0.3_0.5_211"
    _label_dir = "../BASNetTemp/his2/{}/train".format(_label_name)
    ####################################################################################################
    _name_model = "2_NewTrain_{}_{}_{}_{}".format(
        os.path.basename(_label_dir), _size_train, _size_test, "Supervised" if _is_supervised else "")

    Tools.print()
    Tools.print(_name_model)
    Tools.print()

    sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
    all_image, all_mask, all_dataset_name = sod_data.duts_tr()
    _label_name = "cam_up_norm_C23_crf"
    all_label = [os.path.join(_label_dir, "{}_{}_{}.bmp".format(
        name, os.path.splitext(os.path.basename(image))[0],
        _label_name)) for image, name in zip(all_image, all_dataset_name)]
    for label in all_label:
        assert os.path.exists(label)

    bas_runner = BASRunner(batch_size=_batch_size, size_train=_size_train, size_test=_size_test, is_f_loss=_is_f_loss,
                           tra_img_name_list=all_image, tra_lbl_name_list=all_label, learning_rate=_learning_rate,
                           model_dir="../BASNetTemp/saved_models/{}".format(_name_model))
    bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256_DFalse/1000_train_1.154.pth")
    bas_runner.train(epoch_num=50, start_epoch=0)

    pass
