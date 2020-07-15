import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from pydensecrf.utils import unary_from_softmax
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock as ResBlock


#######################################################################################################################
# 0 CRF
class CRFTool(object):

    @staticmethod
    def crf(img, annotation, t=5):  # [3, w, h], [1, w, h]
        img = np.ascontiguousarray(img)
        annotation = np.concatenate([annotation, 1 - annotation], axis=0)

        h, w = img.shape[:2]

        d = dcrf.DenseCRF2D(w, h, 2)
        unary = unary_from_softmax(annotation)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)
        # DIAG_KERNEL     CONST_KERNEL FULL_KERNEL
        # NORMALIZE_BEFORE NORMALIZE_SYMMETRIC     NO_NORMALIZATION  NORMALIZE_AFTER
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10,
                               kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        q = d.inference(t)

        result = np.array(q).reshape((2, h, w))
        return result[0]

    @classmethod
    def crf_list(cls, img, annotation, t=5):
        img_data = np.asarray(img, dtype=np.uint8)
        annotation_data = np.asarray(annotation)
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            result_one = cls.crf(img_data_one, annotation_data_one, t=t)
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

    def __init__(self, img_name_list, lab_name_list,
                 cam_lbl_name_list, his_lbl_name_list, size_train=224, is_filter=False):
        self.is_filter = is_filter
        self.img_name_list = img_name_list
        self.tra_lab_name_list = lab_name_list
        self.cam_lbl_name_list = cam_lbl_name_list
        self.his_lbl_name_list = his_lbl_name_list
        self.transform = Compose([FixedResized(size_train, size_train), RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.lab_name_list = None

        self.image_size_list = [Image.open(image_name).size for image_name in self.img_name_list]
        Tools.print("DatasetUSOD: size_train={} is_filter={}".format(size_train, is_filter))
        pass

    def set_label(self, is_supervised, has_history):
        self.lab_name_list = self.tra_lab_name_list if is_supervised else self.cam_lbl_name_list
        self.lab_name_list = self.his_lbl_name_list if has_history else self.lab_name_list
        Tools.print("DatasetUSOD change label: is_supervised={} has_history={}".format(is_supervised, has_history))
        pass

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        assert self.lab_name_list is not None

        param = []
        image = Image.open(self.img_name_list[idx]).convert("RGB")
        if os.path.exists(self.lab_name_list[idx]):
            label = Image.open(self.lab_name_list[idx]).convert("L")
        else:
            label = Image.fromarray(np.zeros_like(np.asarray(image), dtype=np.uint8)).convert("L")
        image, label, image_for_crf, param = self.transform(image, label, image, param)

        if self.is_filter:
            num = np.sum(np.asarray(label))
            num_all = label.shape[1] * label.shape[2]
            ratio = num / num_all
            if ratio < 0.01 or ratio > 0.9:
                Tools.print("{} {:.4f} {}".format(idx, ratio, self.lab_name_list[idx]))
                image, label, image_for_crf, idx, param = self.__getitem__(np.random.randint(0, self.__len__()))
            pass

        return image, label, image_for_crf, idx, param

    def save_history(self, history, idx):
        h_path = self.his_lbl_name_list[idx]
        h_path = Tools.new_dir(h_path)

        history = np.asarray(np.squeeze(history) * 255, dtype=np.uint8)
        im = Image.fromarray(history).resize(self.image_size_list[idx])
        im.save(h_path)
        pass

    pass


class DatasetEvalUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list, size_test=256):
        self.image_name_list = img_name_list
        self.label_name_list = lab_name_list
        self.transform = Compose([FixedResized(size_test, size_test), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        label = Image.open(self.label_name_list[idx]).convert("L")
        image, label, image_for_crf, _ = self.transform(image, label, image, None)
        return image, label, image_for_crf

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

    def __init__(self, batch_size=8, size_train=224, size_test=256, is_filter=False,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image', tra_label_dir="DUTS-TR-Mask",
                 cam_label_dir="../BASNetTemp/cam/CAM_123_224_256", cam_label_name='cam_up_norm_C123',
                 his_label_dir="../BASNetTemp/his/CAM_123_224_256", model_dir="./saved_models/model"):
        self.batch_size = batch_size
        self.size_train = size_train
        self.size_test = size_test

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.img_name_list, self.lbl_name_list, self.cam_lbl_name_list, self.his_lbl_name_list = \
            self.get_tra_img_label_name(tra_image_dir, tra_label_dir, cam_label_dir, cam_label_name, his_label_dir)
        self.dataset_sod = DatasetUSOD(img_name_list=self.img_name_list, lab_name_list=self.lbl_name_list,
                                       cam_lbl_name_list=self.cam_lbl_name_list,
                                       his_lbl_name_list=self.his_lbl_name_list,
                                       is_filter=is_filter, size_train=self.size_train)
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

    def get_tra_img_label_name(self, tra_image_dir, tra_label_dir, cam_label_dir, cam_label_name, his_label_dir):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, tra_image_dir, '*.jpg'))

        tra_lbl_name_list = [os.path.join(self.data_dir, tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]

        cam_lbl_name_list = [os.path.join(cam_label_dir, '{}_{}.bmp'.format(os.path.splitext(
            os.path.basename(img_path))[0], cam_label_name)) for img_path in tra_img_name_list]

        his_lbl_name_list = [os.path.join(his_label_dir, '{}_{}.bmp'.format(os.path.splitext(
            os.path.basename(img_path))[0], cam_label_name)) for img_path in tra_img_name_list]

        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list, cam_lbl_name_list, his_lbl_name_list

    def all_loss_fusion(self, sod_output, sod_label, ignore_label=255.0):
        positions = sod_label.view(-1, 1) != ignore_label
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        return loss_bce

    def save_histories(self, histories, indexes):
        for history, index in zip(histories, indexes):
            self.dataset_sod.save_history(idx=int(index), history=np.asarray(history.squeeze()))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, which_history=1,
              is_supervised=False, has_history=False, history_start_epoch=1):
        self.dataset_sod.set_label(is_supervised=is_supervised, has_history=False)

        all_loss = 0
        for epoch in range(start_epoch, epoch_num+1):
            ###########################################################################
            # 0 准备
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            if epoch == start_epoch + history_start_epoch:
                self.dataset_sod.set_label(is_supervised=is_supervised, has_history=has_history)
                pass
            ###########################################################################

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            Tools.print()
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
                    if epoch == start_epoch + history_start_epoch - 1:
                        histories = histories
                    elif epoch >= start_epoch + history_start_epoch:
                        if which_history == 0:  # 1: 55
                            sod_crf = CRFTool.crf_list(image_for_crf * 255, sod_output, t=5)
                            histories = 0.5 * histories + 0.5 * sod_crf
                        elif which_history == 1:  # 2: 91
                            sod_crf = CRFTool.crf_list(image_for_crf * 255, sod_output, t=5)
                            histories = 0.9 * histories + 0.1 * sod_crf
                        elif which_history == 2:  # 3: 55
                            histories = 0.5 * histories + 0.5 * sod_output
                            histories = CRFTool.crf_list(image_for_crf * 255, histories, t=5)
                        elif which_history == 3:  # 4: 91
                            histories = 0.9 * histories + 0.1 * sod_output
                            histories = CRFTool.crf_list(image_for_crf * 255, histories, t=5)
                        else:
                            raise Exception("{}...................".format(which_history))
                    else:
                        histories = sod_output
                        pass
                    self.save_histories(indexes=indexes, histories=histories)
                    pass
                ##############################################
                pass
            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f}".format(epoch, epoch_num, all_loss/self.data_batch_num))
            ###########################################################################

            ###########################################################################
            # 2 保存模型
            if epoch % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "{}_train_{:.3f}.pth".format(epoch, all_loss/self.data_batch_num)))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()

                ###########################################################################
                # 3 评估模型
                # self.eval(self.net, epoch=epoch, is_test=True, batch_size=self.batch_size, size_test=self.size_test)
                self.eval(self.net, epoch=epoch, is_test=False, batch_size=self.batch_size, size_test=self.size_test)
                ###########################################################################
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            self.eval(self.net, epoch=epoch, is_test=True, batch_size=self.batch_size, size_test=self.size_test)
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
    def eval(net, epoch=0, is_test=True, size_test=256, batch_size=16, th_num=100, beta_2=0.3):
        which = "TE" if is_test else "TR"
        data_dir = '/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-{}'.format(which)
        image_dir, label_dir = 'DUTS-{}-Image'.format(which), 'DUTS-{}-Mask'.format(which)

        # 数据
        img_name_list = glob.glob(os.path.join(data_dir, image_dir, '*.jpg'))
        lbl_name_list = [os.path.join(data_dir, label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list]
        dataset_eval_sod = DatasetEvalUSOD(img_name_list=img_name_list,
                                           lab_name_list=lbl_name_list, size_test=size_test)
        data_loader_eval_sod = DataLoader(dataset_eval_sod, batch_size, shuffle=False, num_workers=24)

        # 执行
        avg_mae = 0.0
        avg_prec = np.zeros(shape=(th_num,)) + 1e-6
        avg_recall = np.zeros(shape=(th_num,)) + 1e-6
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels, _) in tqdm(enumerate(data_loader_eval_sod), total=len(data_loader_eval_sod)):
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs

                now_label = labels.squeeze().data.numpy()
                return_m = net(inputs)

                now_pred = return_m["out_up_sigmoid"].squeeze().cpu().data.numpy()

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
1. 迭代优化: 训练多代后, 使用模型的输出作为标签
2. 端到端训练时更新SOD
+ 3. 挑选训练样本!!!  
4. 在ImageNet上训练!!!
"""


if __name__ == '__main__':
    _which_history = 1

    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(_which_history)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    _batch_size = 16 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    _size_train = 224
    _size_test = 256

    _is_supervised = False
    _has_history = True
    _is_filter = False
    _history_start_epoch = 11

    _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_1"
    # _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_9"
    # _cam_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_30"
    _cam_label_name = 'cam_up_norm_C123_crf'

    Tools.print()
    _name_model = "CAM_123_SOD_{}_{}{}{}{}{}_{}".format(
        _size_train, _size_test, "_{}".format(_cam_label_name), "_Filter" if _is_filter else "",
        "_Supervised" if _is_supervised else "", "_History" if _has_history else "", _which_history)
    _his_label_dir = "../BASNetTemp/his/{}".format(_name_model)
    Tools.print(_name_model)
    Tools.print(_cam_label_name)
    Tools.print(_cam_label_dir)
    Tools.print(_his_label_dir)
    Tools.print()

    bas_runner = BASRunner(batch_size=_batch_size, size_train=_size_train, size_test=_size_test,
                           data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR", is_filter=_is_filter,
                           cam_label_dir=_cam_label_dir, cam_label_name=_cam_label_name, his_label_dir=_his_label_dir,
                           model_dir="../BASNetTemp/saved_models/{}".format(_name_model))
    bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256/930_train_1.172.pth")
    bas_runner.train(epoch_num=100, start_epoch=0, which_history=_which_history,
                     history_start_epoch=_history_start_epoch,
                     is_supervised=_is_supervised, has_history=_has_history)
    pass
