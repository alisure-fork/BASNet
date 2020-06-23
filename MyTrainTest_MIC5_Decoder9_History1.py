import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from skimage import io
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock as ResBlock


#######################################################################################################################
# 1 Data


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img, parm, his=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        parm.extend([i, j, h, w])
        if his is not None:
            his = transforms.functional.resized_crop(his, i, j, h, w, self.size, self.interpolation)
        return img, parm, his

    pass


class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, parm, his=None):
        img = super().__call__(img)
        return img, parm, his

    pass


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, parm, his=None):
        img = super().__call__(img)
        return img, parm, his

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, parm, his=None):
        if random.random() < self.p:
            parm.append(1)
            img = transforms.functional.hflip(img)
            if his is not None:
                his = transforms.functional.hflip(his)
        else:
            parm.append(0)
        return img, parm, his

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, parm, his=None):
        img = super().__call__(img)
        if his is not None:
            his = super().__call__(his)
        return img, parm, his

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, parm, his=None):
        img = super().__call__(img)
        return img, parm, his

    pass


class Compose(transforms.Compose):
    def __call__(self, img, parm, his=None):
        for t in self.transforms:
            img, parm, his = t(img, parm, his)
        return img, parm, his

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, his_name_list=None, is_train=True):
        self.image_name_list = img_name_list
        self.history_name_list = his_name_list
        self.has_history = self.history_name_list is not None

        self.is_train = is_train
        self.transform_train = Compose([RandomResizedCrop(size=224, scale=(0.3, 1.)),
                                        ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2),
                                        RandomHorizontalFlip(), ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_test = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        history = None
        if self.has_history:
            h_path = self.history_name_list[idx]
            history = Image.open(h_path).convert("L")if os.path.exists(h_path) else Image.new("L", size=image.size)

        param = [image.size[0], image.size[1]]
        image, param, history = self.transform_train(
            image, param, history) if self.is_train else self.transform_test(image, param, history)

        return image, history if self.has_history else image, np.asarray(param), idx

    def save_history(self, idx, his, param):
        """ his: [0, 1] """
        if self.history_name_list is not None:
            h_path = Tools.new_dir(self.history_name_list[idx])
            history = Image.open(h_path).convert("L")if os.path.exists(
                h_path) else Image.new("L", size=[param[0], param[1]])

            im = Image.fromarray(np.asarray(his * 255, dtype=np.uint8)).resize((param[5], param[4]))
            im = im.transpose(Image.FLIP_LEFT_RIGHT) if param[6] else im
            history.paste(im, (param[3], param[2]))
            history.save(h_path)
        pass

    pass


class DatasetEvalUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list):
        self.image_name_list = np.asarray(img_name_list)
        self.label_name_list = np.asarray(lab_name_list)

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image = self.transform_test(image)

        label_shape = [image.shape[0], image.shape[1], 1]
        if 0 == len(self.label_name_list):
            label = np.zeros(label_shape)
        else:
            label = io.imread(self.label_name_list[idx])
            if 3 == len(label.shape):
                label = label[:, :, 0]
                pass
            label = label[:, :, np.newaxis]
            pass

        return image, label

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

    def __init__(self, n_channels, clustering_num_list):
        super(BASNet, self).__init__()
        self.clustering_num_list = clustering_num_list

        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBlock(n_channels, 64, has_relu=True)  # 64 * 224 * 224
        self.encoder1 = resnet.layer1  # 64 * 224 * 224
        self.encoder2 = resnet.layer2  # 128 * 112 * 112
        self.encoder3 = resnet.layer3  # 256 * 56 * 56
        self.encoder4 = resnet.layer4  # 512 * 28 * 28

        # -------------MIC-------------
        self.mic_l2norm = MICNormalize(2)

        # MIC 1
        self.mic_1_b1 = ResBlock(512, 512)  # 28
        self.mic_1_b2 = ResBlock(512, 512)
        self.mic_1_b3 = ResBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)

        # MIC 2
        self.mic_3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_2_b1 = ResBlock(512, 512)  # 14
        self.mic_2_b2 = ResBlock(512, 512)
        self.mic_2_b3 = ResBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)

        # MIC 3
        self.mic_3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_3_b1 = ResBlock(512, 512)  # 7
        self.mic_3_b2 = ResBlock(512, 512)
        self.mic_3_b3 = ResBlock(512, 512)
        self.mic_3_c1 = ConvBlock(512, self.clustering_num_list[2], has_relu=True)

        # -------------Decoder-------------
        # Decoder 1
        self.decoder_1_b = ResBlock(512, 512)  # 28
        self.decoder_1_c = ConvBlock(512, 256, has_relu=True)  # 28

        # Decoder 2
        self.decoder_2_b = ResBlock(256, 256)  # 56
        self.decoder_2_c = ConvBlock(256, 128, has_relu=True)  # 56

        # Decoder 3
        self.decoder_3_b = ResBlock(128, 128)  # 112
        self.decoder_3_out = nn.Conv2d(128, 1, 3, padding=1, bias=False)  # 112
        pass

    def forward(self, x):
        x_for_up = x

        # -------------Encoder-------------
        e0 = self.encoder0(x)  # 64 * 224 * 224
        e1 = self.encoder1(e0)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28

        # -------------MIC-------------
        # 1
        mic_1_feature = self.mic_1_b3(self.mic_1_b2(self.mic_1_b1(e4)))  # 512 * 28 * 28
        mic_1 = self.mic_1_c1(mic_1_feature)  # 128 * 28 * 28
        smc_logits_1, smc_l2norm_1 = self.salient_map_clustering(mic_1)
        return_m1 = {"smc_logits": smc_logits_1, "smc_l2norm": smc_l2norm_1}

        # 2
        mic_2_feature = self.mic_2_b3(self.mic_2_b2(self.mic_2_b1(mic_1_feature)))  # 512 * 14 * 14
        mic_2 = self.mic_2_c1(mic_2_feature)  # 256 * 14 * 14
        smc_logits_2, smc_l2norm_2 = self.salient_map_clustering(mic_2)
        return_m2 = {"smc_logits": smc_logits_2, "smc_l2norm": smc_l2norm_2}

        # 3
        mic_3_feature = self.mic_3_b3(self.mic_3_b2(self.mic_3_b1(mic_2_feature)))  # 512 * 7 * 7
        mic_3 = self.mic_3_c1(mic_3_feature)  # 512 * 7 * 7
        smc_logits_3, smc_l2norm_3 = self.salient_map_clustering(mic_3)
        return_m3 = {"smc_logits": smc_logits_3, "smc_l2norm": smc_l2norm_3}

        # -------------Label-------------
        cam_1 = self.cluster_activation_map(smc_logits_1, mic_1)  # 簇激活图：Cluster Activation Map
        cam_1_up = self._up_to_target(cam_1, x_for_up)
        cam_1_up_norm = self._feature_norm(cam_1_up)

        cam_2 = self.cluster_activation_map(smc_logits_2, mic_2)  # 簇激活图：Cluster Activation Map
        cam_2_up = self._up_to_target(cam_2, cam_1_up)
        cam_2_up_norm = self._feature_norm(cam_2_up)

        cam_3 = self.cluster_activation_map(smc_logits_3, mic_3)  # 簇激活图：Cluster Activation Map
        cam_3_up = self._up_to_target(cam_3, cam_1_up)
        cam_3_up_norm = self._feature_norm(cam_3_up)

        # cam_up_norm = (cam_1_up_norm + cam_2_up_norm + cam_3_up_norm) / 3
        # cam_up_norm = (cam_1_up_norm + cam_2_up_norm) / 2
        cam_up_norm = cam_1_up_norm

        # label = self.salient_map_divide(cam_up_norm, obj_th=0.8, bg_th=0.2, more_obj=False)  # 显著图划分
        label = cam_up_norm

        return_l = {"label": label, "cam_up_norm": cam_up_norm, "cam_1_up_norm": cam_1_up_norm,
                    "cam_2_up_norm": cam_2_up_norm, "cam_3_up_norm": cam_3_up_norm}

        # -------------Decoder-------------
        # decoder
        d1 = self.decoder_1_b(e4)  # 512 * 28 * 28
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 56 * 56

        d2 = self.decoder_2_b(d1_d2)  # 256 * 56 * 56
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 128 * 112 * 112

        # d3
        d3 = self.decoder_3_b(d2_d3)  # 128 * 112 * 112
        d3_out = self.decoder_3_out(d3)  # 1 * 112 * 112
        d3_out_sigmoid = torch.sigmoid(d3_out)  # 1 * 112 * 112  # 小输出
        d3_out_up = self._up_to_target(d3_out, x_for_up)  # 1 * 224 * 224
        d3_out_up_sigmoid = torch.sigmoid(d3_out_up)  # 1 * 224 * 224  # 大输出
        return_d3 = {"out": d3_out, "out_sigmoid": d3_out_sigmoid,
                     "out_up": d3_out_up, "out_up_sigmoid": d3_out_up_sigmoid}

        output = {"output": d3_out_up_sigmoid}

        return_m = {"m1": return_m1, "m2": return_m2, "m3": return_m3}
        return_l = {"label": return_l, "output": output}
        return_d = {"d3": return_d3}
        return return_m, return_l, return_d

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, output_size=(1, 1)).view((mic.size()[0], -1))
        smc_l2norm = self.mic_l2norm(smc_logits)
        return smc_logits, smc_l2norm

    @staticmethod
    def cluster_activation_map(smc_logits, mic_feature):
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        cam = torch.cat([mic_feature[i:i+1, top_k_index[i], :, :] for i in range(mic_feature.size()[0])])
        return cam

    def salient_map_divide(self, cam_up_sigmoid, obj_th=0.7, bg_th=0.2, more_obj=False):
        cam_up_sigmoid = self._feature_norm(cam_up_sigmoid)

        input_size = tuple(cam_up_sigmoid.size())
        label = torch.zeros(input_size).fill_(255).cuda()

        # bg
        label[cam_up_sigmoid < bg_th] = 0.0

        # obj
        if more_obj:
            for i in range(input_size[0]):
                mask_pos_i = cam_up_sigmoid[i] > obj_th
                if torch.sum(mask_pos_i) < input_size[-1] * input_size[-2] / 22:
                    mask_pos_i = cam_up_sigmoid[i] > (obj_th * 0.9)
                    pass
                label[i][mask_pos_i] = 1.0
                pass
            pass
        else:
            label[cam_up_sigmoid > obj_th] = 1.0
            pass

        return label

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

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, batch_size_train=8, clustering_num_1=128, clustering_num_2=256, clustering_num_3=512,
                 clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2, only_mic=False, has_history=False,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic_only",
                 history_dir="./history/my_train_mic5_large_history1"):
        self.batch_size_train = batch_size_train
        self.only_mic = only_mic

        # History
        self.history_dir = Tools.new_dir(history_dir)

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list, self.tra_his_name_list = self.get_tra_img_label_name()

        self.has_history = has_history
        self.tra_his_name_list = self.tra_his_name_list if self.has_history else None
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list,
                                        his_name_list=self.tra_his_name_list, is_train=True)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=32)

        # Model
        self.net = BASNet(3, clustering_num_list=[clustering_num_1, clustering_num_2, clustering_num_3])

        ###########################################################################
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
            cudnn.benchmark = True
        ###########################################################################

        # MIC
        self.produce_class11 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class21 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class31 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_3, ratio=clustering_ratio_3)
        self.produce_class12 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class22 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class32 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_3, ratio=clustering_ratio_3)
        self.produce_class11.init()
        self.produce_class21.init()
        self.produce_class31.init()
        self.produce_class12.init()
        self.produce_class22.init()
        self.produce_class32.init()

        # Loss and optimizer
        self.bce_loss = nn.BCELoss().cuda()
        self.mic_loss = nn.CrossEntropyLoss().cuda()
        self.learning_rate = [[0, 0.001], [300, 0.0001], [400, 0.00001]]
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate[0][1], betas=(0.9, 0.999), weight_decay=1e-4)
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

    def get_tra_img_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, self.tra_image_dir, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_dir, self.tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        tra_his_name_list = [os.path.join(self.history_dir, '{}.bmp'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        Tools.print("train history: {}".format(len(tra_his_name_list)))
        return tra_img_name_list, tra_lbl_name_list, tra_his_name_list

    @staticmethod
    def sigmoid(x, a=10):
        return 1 / (1 + np.exp(-(x - a)))
        pass

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_3_out, mic_labels_1, mic_labels_2, mic_labels_3,
                        sod_output, sod_label, only_mic=False):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)
        loss_mic_3 = self.mic_loss(mic_3_out, mic_labels_3)

        positions = sod_label.view(-1, 1) < 255.0
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])

        loss_all = loss_mic_1 + loss_mic_2 + loss_mic_3
        if not only_mic:
            loss_all = loss_all + 10 * loss_bce
            pass
        return loss_all, [loss_mic_1, loss_mic_2, loss_mic_3], loss_bce

    def save_history_info(self, histories, params, indexes):
        for history, param, index in zip(histories, params, indexes):
            self.dataset_usod.save_history(idx=int(index),
                                           his=np.asarray(history.squeeze().detach().cpu()), param=np.asarray(param))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50, eval_epoch_freq=10):

        if start_epoch > 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            with torch.no_grad():
                for _idx, (inputs, histories, params, indexes) in tqdm(enumerate(self.dataloader_usod),
                                                                       total=len(self.dataloader_usod)):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    return_m, return_l, return_d = self.net(inputs)

                    self.produce_class11.cal_label(return_m["m1"]["smc_l2norm"], indexes)
                    self.produce_class21.cal_label(return_m["m2"]["smc_l2norm"], indexes)
                    self.produce_class31.cal_label(return_m["m3"]["smc_l2norm"], indexes)
                    pass
                pass
            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            classes = self.produce_class32.classes
            self.produce_class32.classes = self.produce_class31.classes
            self.produce_class31.classes = classes
            Tools.print("Update: [{}] 1-{}/{}".format(start_epoch, self.produce_class11.count,
                                                      self.produce_class11.count_2))
            Tools.print("Update: [{}] 2-{}/{}".format(start_epoch, self.produce_class21.count,
                                                      self.produce_class21.count_2))
            Tools.print("Update: [{}] 3-{}/{}".format(start_epoch, self.produce_class31.count,
                                                      self.produce_class31.count_2))
            pass

        all_loss = 0
        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic_1, all_loss_mic_2, all_loss_mic_3, all_loss_sod = 0.0, 0.0, 0.0, 0.0, 0.0
            Tools.print()
            self.net.train()

            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()

            for i, (inputs, histories, params, indexes) in tqdm(enumerate(self.dataloader_usod),
                                                                total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                histories = histories.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                self.optimizer.zero_grad()

                return_m, return_l, _ = self.net(inputs)

                ######################################################################################################
                # MIC
                self.produce_class11.cal_label(return_m["m1"]["smc_logits"], indexes)
                self.produce_class21.cal_label(return_m["m2"]["smc_logits"], indexes)
                self.produce_class31.cal_label(return_m["m3"]["smc_logits"], indexes)
                mic_target_1 = return_m["m1"]["smc_logits"]
                mic_target_2 = return_m["m2"]["smc_logits"]
                mic_target_3 = return_m["m3"]["smc_logits"]
                mic_labels_1 = self.produce_class12.get_label(indexes).cuda()
                mic_labels_2 = self.produce_class22.get_label(indexes).cuda()
                mic_labels_3 = self.produce_class32.get_label(indexes).cuda()
                ######################################################################################################

                ######################################################################################################
                # SOD
                histories = histories  # Annotation
                sod_label = return_l["label"]["label"].detach()  # CAM
                sod_output = return_l["output"]["output"]  # Predict

                if self.has_history:
                    # NO History
                    # histories = (histories + sod_output) / 2

                    # 历史信息 = 历史信息 + CAM + SOD
                    sod_label = self.sigmoid(sod_label * 20, a=12)
                    sod_label = sod_label if histories.max() == 0 else (histories * 0.5 + sod_label * 0.5)
                    # histories = sod_label * 2 / 3 + sod_output * 1 / 3
                    histories = sod_label

                    self.save_history_info(histories=histories, params=params, indexes=indexes)
                    pass
                ######################################################################################################

                loss, loss_mic, loss_sod = self.all_loss_fusion(
                    mic_target_1, mic_target_2, mic_target_3, mic_labels_1, mic_labels_2, mic_labels_3,
                    sod_output, sod_label, only_mic=self.only_mic)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic_1 += loss_mic[0].item()
                all_loss_mic_2 += loss_mic[1].item()
                all_loss_mic_3 += loss_mic[2].item()
                all_loss_sod += loss_sod.item()
                if i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d}, b:{:4d}/{:4d}] l:{:.2f}/{:.2f} "
                                "mic1:{:.2f}/{:.2f} mic2:{:.2f}/{:.2f} mic3:{:.2f}/{:.2f} "
                                "sod:{:.2f}/{:.2f}".format(epoch, epoch_num, i, len(self.dataloader_usod),
                                                           all_loss/(i+1), loss.item(), all_loss_mic_1/(i+1),
                                                           loss_mic[0].item(), all_loss_mic_2/(i+1),
                                                           loss_mic[1].item(), all_loss_mic_3/(i+1),
                                                           loss_mic[2].item(), all_loss_sod/(i+1), loss_sod.item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic1:{:.3f} mic2:{:.3f} mic3:{:.3f} sod:{:.3f}".format(
                epoch, epoch_num, all_loss / (len(self.dataloader_usod) + 1),
                all_loss_mic_1 / (len(self.dataloader_usod) + 1),
                all_loss_mic_2 / (len(self.dataloader_usod) + 1),
                all_loss_mic_3 / (len(self.dataloader_usod) + 1),
                all_loss_sod / (len(self.dataloader_usod) + 1)))

            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            classes = self.produce_class32.classes
            self.produce_class32.classes = self.produce_class31.classes
            self.produce_class31.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, self.produce_class11.count, self.produce_class11.count_2))
            Tools.print("Train: [{}] 2-{}/{}".format(epoch, self.produce_class21.count, self.produce_class21.count_2))
            Tools.print("Train: [{}] 3-{}/{}".format(epoch, self.produce_class31.count, self.produce_class31.count_2))

            ###########################################################################
            # 2 测试模型
            # if epoch % eval_epoch_freq == 0:
            #     Tools.print()
            #     self.eval(self.net, epoch=epoch, is_test=False)
            #     Tools.print()
            #     self.eval(self.net, epoch=epoch, is_test=True)
            #     pass
            ###########################################################################

            ###########################################################################
            # 3 保存模型
            if epoch % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "{}_train_{:.3f}.pth".format(epoch, all_loss / len(self.dataloader_usod))))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()
                pass
            ###########################################################################

            pass

        # Final Save
        save_file_name = Tools.new_dir(os.path.join(
            self.model_dir, "{}_train_{:.3f}.pth".format(epoch_num, all_loss / len(self.dataloader_usod))))
        torch.save(self.net.state_dict(), save_file_name)

        Tools.print()
        Tools.print("Save Model to {}".format(save_file_name))
        Tools.print()
        pass

    def eval(self, net, epoch=0, is_test=True, print_ite_num=100, th_num=100, beta_2=0.3):
        which = "TE" if is_test else "TR"
        data_dir = "{}/DUTS-{}".format(os.path.split(self.data_dir)[0], which)
        image_dir, label_dir = 'DUTS-{}-Image'.format(which), 'DUTS-{}-Mask'.format(which)

        # 数据
        img_name_list = glob.glob(os.path.join(data_dir, image_dir, '*.jpg'))
        lbl_name_list = [os.path.join(data_dir, label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list]
        dataset_eval_usod = DatasetEvalUSOD(img_name_list=img_name_list, lab_name_list=lbl_name_list)
        dataloader_eval_usod = DataLoader(dataset_eval_usod, 1, shuffle=False, num_workers=8)

        # 执行
        d_m = [["output", "output_1"], ["output", "output_2"],
               ["d1", "out_up_sigmoid"], ["d2", "out_up_sigmoid"], ["d3", "out_up_sigmoid"]]
        avg_mae = [0.0] * len(d_m)
        avg_prec = np.zeros(shape=(len(d_m), th_num)) + 1e-6
        avg_recall = np.zeros(shape=(len(d_m), th_num)) + 1e-6
        net.eval()
        for i, (inputs, labels) in enumerate(dataloader_eval_usod):
            inputs = inputs.type(torch.FloatTensor).cuda()
            now_label = labels.squeeze().data.numpy() / 255

            return_m, return_l, return_d = net(inputs)

            mae_list = [0.0] * len(d_m)
            for index, key in enumerate(d_m):
                return_which = return_l if index < 2 else return_d
                d_out = return_which[key[0]][key[1]].squeeze().cpu().data.numpy()
                now_pred = np.asarray(Image.fromarray(d_out * 255).resize(
                    (now_label.shape[1], now_label.shape[0]), resample=Image.BILINEAR)) / 255

                mae = dataset_eval_usod.eval_mae(now_pred, now_label)
                prec, recall = dataset_eval_usod.eval_pr(now_pred, now_label, th_num)

                mae_list[index] = mae
                avg_mae[index] += mae
                avg_prec[index] += prec
                avg_recall[index] += recall
                pass

            if i % print_ite_num == 0:
                now = ""
                for index, key in enumerate(d_m):
                    now += "{}:{:.2f}/{:.2f} ".format(key[0], avg_mae[index] / (i + 1), mae_list[index])
                    pass
                Tools.print("{} [E:{:4d}, b:{:4d}/{:4d}] {}".format("Test" if is_test else "Train",
                                                                    epoch, i, len(dataloader_eval_usod), now))
                pass

            pass

        # 结果
        score = np.zeros(shape=(len(d_m), th_num))
        for index, key in enumerate(d_m):
            avg_mae[index] = avg_mae[index] / len(dataloader_eval_usod)
            avg_prec[index] = avg_prec[index] / len(dataloader_eval_usod)
            avg_recall[index] = avg_recall[index] / len(dataloader_eval_usod)
            score[index] = (1+beta_2)*avg_prec[index]*avg_recall[index]/(beta_2*avg_prec[index]+avg_recall[index])
            Tools.print("{} {} {} {} {}".format("Test" if is_test else "Train", epoch,
                                                key[0], avg_mae[index], score[index].max()))
            pass

        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    bas_runner = BASRunner(batch_size_train=16 * 5, data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           clustering_num_1=128 * 4, clustering_num_2=128 * 4, clustering_num_3=128 * 4,
                           history_dir="../BASNetTemp/history/my_train_mic5_large_history9_1", has_history=True,
                           model_dir="../BASNetTemp/saved_models/my_train_mic5_large_history9_1")
    bas_runner.load_model('../BASNetTemp/saved_models/my_train_mic5_large/500_train_0.880.pth')
    bas_runner.train(epoch_num=500, start_epoch=1)
    pass
