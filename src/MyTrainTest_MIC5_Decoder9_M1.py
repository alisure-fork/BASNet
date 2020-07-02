import os
import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import io
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader, Dataset


#######################################################################################################################
# 1 Data

class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, is_train=True):
        self.image_name_list = img_name_list

        self.is_train = is_train
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.3, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image = self.transform_train(image) if self.is_train else self.transform_test(image)
        return image, idx

    pass


class DatasetEvalUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list, small_data=1):
        self.small_data = small_data
        self.image_name_list = np.asarray(img_name_list)
        self.label_name_list = np.asarray(lab_name_list)
        if self.small_data > 1:
            index = np.arange(len(self.image_name_list))
            np.random.shuffle(index)
            index = index[0: int(len(self.image_name_list) / small_data)]
            self.image_name_list = self.image_name_list[index]
            self.label_name_list = self.label_name_list[index]
            pass

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


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        pass

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    pass


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

    def __init__(self, n_channels, clustering_num_list=None):
        super(BASNet, self).__init__()
        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBlock(n_channels, 64, has_relu=True)  # 64 * 224 * 224
        self.encoder1 = resnet.layer1  # 64 * 224 * 224
        self.encoder2 = resnet.layer2  # 128 * 112 * 112
        self.encoder3 = resnet.layer3  # 256 * 56 * 56
        self.encoder4 = resnet.layer4  # 512 * 28 * 28
        self.encoder5_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.encoder5_b1 = ResBlock(512, 512)
        self.encoder5_b2 = ResBlock(512, 512)
        self.encoder5_b3 = ResBlock(512, 512)  # 512 * 14 * 14

        # -------------MIC-------------
        self.clustering_num_list = list([128, 256, 512]) if clustering_num_list is None else clustering_num_list

        # MIC 1
        self.mic_1_b1 = ResBlock(256, 256)  # 56
        self.mic_1_c1 = ConvBlock(256, self.clustering_num_list[0], has_relu=True)
        self.mic_1_l2norm = MICNormalize(2)

        # MIC 2
        self.mic_2_b1 = ResBlock(512, 512)  # 28
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)
        self.mic_2_l2norm = MICNormalize(2)

        # MIC 3
        self.mic_3_b1 = ResBlock(512, 512)  # 14
        self.mic_3_c1 = ConvBlock(512, self.clustering_num_list[2], has_relu=True)
        self.mic_3_l2norm = MICNormalize(2)

        # Decoder 0
        self.decoder_0_b = ResBlock(512, 512)  # 14
        self.decoder_0_out = nn.Conv2d(512, 1, 3, padding=1)
        self.decoder_0_c = ConvBlock(512, 512, has_relu=True)  # 28

        # Decoder 1
        self.decoder_1_b = ResBlock(512, 512)  # 28
        self.decoder_1_out = nn.Conv2d(512, 1, 3, padding=1)
        self.decoder_1_c = ConvBlock(512, 256, has_relu=True)  # 56

        # Decoder 2
        self.decoder_2_b = ResBlock(256, 256)  # 56
        self.decoder_2_out = nn.Conv2d(256, 1, 3, padding=1)
        self.decoder_2_c = ConvBlock(256, 128, has_relu=True)  # 112

        # Decoder 3
        self.decoder_3_b = ResBlock(128, 128)  # 112
        self.decoder_3_out = nn.Conv2d(128, 1, 3, padding=1)
        pass

    def forward(self, x):
        x_for_up = x

        # -------------Encoder-------------
        e0 = self.encoder0(x)  # 64 * 224 * 224
        e1 = self.encoder1(e0)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28
        e5 = self.encoder5_b3(self.encoder5_b2(self.encoder5_b1(self.encoder5_pool(e4))))  # 512 * 14 * 14

        # -------------MIC-------------
        # 1
        mic_f_1 = self.mic_1_b1(e3)  # 256 * 56 * 56
        mic_1 = self.mic_1_c1(mic_f_1)  # 128 * 56 * 56
        smc_logits_1, smc_l2norm_1, smc_sigmoid_1 = self.salient_map_clustering(mic_1)
        cam_1 = self.cluster_activation_map(smc_logits_1, mic_1)  # 簇激活图：Cluster Activation Map
        return_m1 = {"smc_logits": smc_logits_1, "smc_l2norm": smc_l2norm_1, "smc_sigmoid": smc_sigmoid_1, "cam": cam_1}

        # 2
        mic_f_2 = self.mic_2_b1(e4)  # 512 * 28 * 28
        mic_2 = self.mic_2_c1(mic_f_2)  # 256 * 28 * 28
        smc_logits_2, smc_l2norm_2, smc_sigmoid_2 = self.salient_map_clustering(mic_2)
        cam_2 = self.cluster_activation_map(smc_logits_2, mic_2)  # 簇激活图：Cluster Activation Map
        return_m2 = {"smc_logits": smc_logits_2, "smc_l2norm": smc_l2norm_2, "smc_sigmoid": smc_sigmoid_2, "cam": cam_2}

        # 3
        mic_f_3 = self.mic_3_b1(e5)  # 512 * 14 * 14
        mic_3 = self.mic_3_c1(mic_f_3)  # 512 * 14 * 14
        smc_logits_3, smc_l2norm_3, smc_sigmoid_3 = self.salient_map_clustering(mic_3)
        cam_3 = self.cluster_activation_map(smc_logits_3, mic_3)  # 簇激活图：Cluster Activation Map
        return_m3 = {"smc_logits": smc_logits_3, "smc_l2norm": smc_l2norm_3, "smc_sigmoid": smc_sigmoid_3, "cam": cam_3}

        # -------------Label-------------
        cam_norm_1_up = self._up_to_target(cam_1, x_for_up)
        cam_norm_2_up = self._up_to_target(cam_2, cam_norm_1_up)
        cam_norm_3_up = self._up_to_target(cam_3, cam_norm_1_up)
        cam_norm_up = (cam_norm_1_up + cam_norm_2_up + cam_norm_3_up) / 2
        # label = self.salient_map_divide(cam_norm_up, obj_th=0.8, bg_th=0.2)
        # label = self.salient_map_divide(cam_norm_up, obj_th=0.7, bg_th=0.3)
        label = self.salient_map_divide(cam_norm_up, obj_th=0.6, bg_th=0.4)
        # label = self.salient_map_divide(cam_norm_up, obj_th=0.5, bg_th=0.5)
        return_l = {"label": label,  "cam_norm_up": cam_norm_up, "cam_norm_1_up": cam_norm_1_up,
                    "cam_norm_2_up": cam_norm_2_up, "cam_norm_3_up": cam_norm_3_up}

        # -------------Decoder-------------
        # d0
        d0 = self.decoder_0_b(e5)  # 512 * 14 * 14
        d0_d1 = self._up_to_target(self.decoder_0_c(d0), e4) + e4  # 512 * 28 * 28
        d0_out = self.decoder_0_out(d0)  # 1 * 14 * 14
        d0_out_sigmoid = torch.sigmoid(d0_out)  # 1 * 14 * 14  # 小输出
        d0_out_up = self._up_to_target(d0_out, x_for_up)  # 1 * 224 * 224
        d0_out_up_sigmoid = torch.sigmoid(d0_out_up)  # 1 * 224 * 224  # 大输出
        return_d0 = {"out": d0_out,  "out_sigmoid": d0_out_sigmoid,
                     "out_up": d0_out_up, "out_up_sigmoid": d0_out_up_sigmoid}

        # d1
        d1 = self.decoder_1_b(d0_d1)  # 512 * 28 * 28
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 256 * 56 * 56
        d1_out = self.decoder_1_out(d1)  # 1 * 28 * 28
        d1_out_sigmoid = torch.sigmoid(d1_out)  # 1 * 28 * 28  # 小输出
        d1_out_up = self._up_to_target(d1_out, x_for_up)  # 1 * 224 * 224
        d1_out_up_sigmoid = torch.sigmoid(d1_out_up)  # 1 * 224 * 224  # 大输出
        return_d1 = {"out": d1_out,  "out_sigmoid": d1_out_sigmoid,
                     "out_up": d1_out_up, "out_up_sigmoid": d1_out_up_sigmoid}

        # d2
        d2 = self.decoder_2_b(d1_d2)  # 256 * 56 * 56
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 128 * 112 * 112
        d2_out = self.decoder_2_out(d2)  # 1 * 56 * 56
        d2_out_sigmoid = torch.sigmoid(d2_out)  # 1 * 56 * 56  # 小输出
        d2_out_up = self._up_to_target(d2_out, x_for_up)  # 1 * 224 * 224
        d2_out_up_sigmoid = torch.sigmoid(d2_out_up)  # 1 * 224 * 224  # 大输出
        return_d2 = {"out": d2_out,  "out_sigmoid": d2_out_sigmoid,
                     "out_up": d2_out_up, "out_up_sigmoid": d2_out_up_sigmoid}

        # d3
        d3 = self.decoder_3_b(d2_d3)  # 128 * 112 * 112
        d3_out = self.decoder_3_out(d3)  # 1 * 112 * 112
        d3_out_sigmoid = torch.sigmoid(d3_out)  # 1 * 112 * 112  # 小输出
        d3_out_up = self._up_to_target(d3_out, x_for_up)  # 1 * 224 * 224
        d3_out_up_sigmoid = torch.sigmoid(d3_out_up)  # 1 * 224 * 224  # 大输出
        return_d3 = {"out": d3_out, "out_sigmoid": d3_out_sigmoid,
                     "out_up": d3_out_up, "out_up_sigmoid": d3_out_up_sigmoid}

        return_m = {"m1": return_m1, "m2": return_m2, "m3": return_m3}
        return_d = {"label": return_l, "d0": return_d0, "d1": return_d1, "d2": return_d2, "d3": return_d3}
        return return_m, return_d

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, 1).view((mic.size()[0], -1))  # 512
        smc_l2norm = self.mic_1_l2norm(smc_logits)
        smc_sigmoid = torch.sigmoid(smc_logits)
        return smc_logits, smc_l2norm, smc_sigmoid

    def cluster_activation_map(self, smc_logits, mic_feature):
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        cam = torch.cat([mic_feature[i:i+1, top_k_index[i], :, :] for i in range(mic_feature.size()[0])])
        cam_norm = self._feature_norm(cam)
        return cam_norm

    def salient_map_divide(self, cam_norm_up, obj_th=0.7, bg_th=0.3):
        cam_norm_up = self._feature_norm(cam_norm_up)

        label = torch.zeros(tuple(cam_norm_up.size())).fill_(255)
        label = label.cuda() if torch.cuda.is_available() else label
        label[cam_norm_up < bg_th] = 0.0
        label[cam_norm_up > obj_th] = 1.0
        return label

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def _up_to_target(source, target):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode='bilinear')
            pass
        return source

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, epoch_num=1000, batch_size_train=8, only_mic=False,
                 clustering_num_1=128, clustering_num_2=256, clustering_num_3=512,
                 clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic_only"):
        self.epoch_num = epoch_num
        self.batch_size_train = batch_size_train
        self.only_mic = only_mic

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list = self.get_tra_img_label_name()
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list, is_train=True)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=8)

        # Model
        self.net = BASNet(3, clustering_num_list=[clustering_num_1, clustering_num_2, clustering_num_3])
        self.net = self.net.cuda() if torch.cuda.is_available() else self.net

        # MIC
        self.produce_class_1 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class_2 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class_3 = MICProduceClass(n_sample=len(self.dataset_usod),
                                               out_dim=clustering_num_3, ratio=clustering_ratio_3)

        # Loss and Optim
        self.bce_loss = nn.BCELoss()
        self.mic_loss = nn.CrossEntropyLoss()
        self.bce_loss = self.bce_loss.cuda() if torch.cuda.is_available() else self.bce_loss
        self.mic_loss = self.mic_loss.cuda() if torch.cuda.is_available() else self.mic_loss

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        pass

    def load_model(self, model_file_name):
        self.net.load_state_dict(torch.load(model_file_name), strict=False)
        Tools.print("restore from {}".format(model_file_name))
        pass

    def get_tra_img_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, self.tra_image_dir, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_dir, self.tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_3_out, mic_labels_1, mic_labels_2, mic_labels_3,
                        sod_0_sigmoid, sod_1_sigmoid, sod_2_sigmoid, sod_3_sigmoid, sod_label, only_mic=False):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)
        loss_mic_3 = self.mic_loss(mic_3_out, mic_labels_3)

        positions = sod_label.view(-1, 1) < 255.0
        loss_bce_0 = self.bce_loss(sod_0_sigmoid.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        loss_bce_1 = self.bce_loss(sod_1_sigmoid.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        loss_bce_2 = self.bce_loss(sod_2_sigmoid.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])
        loss_bce_3 = self.bce_loss(sod_3_sigmoid.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])

        loss_all = loss_mic_1 + loss_mic_2 + loss_mic_3
        if not only_mic:
            loss_all = loss_all + 10 * (loss_bce_0 + loss_bce_1 + loss_bce_2 + loss_bce_3)
            pass
        return loss_all, [loss_mic_1, loss_mic_2, loss_mic_3], [loss_bce_0, loss_bce_1, loss_bce_2, loss_bce_3]

    def train(self, save_epoch_freq=5, eval_epoch_freq=5, print_ite_num=100, update_epoch_freq=1):

        for epoch in range(0, self.epoch_num):

            ###########################################################################
            # 0 更新标签
            if epoch % update_epoch_freq == 0:
                Tools.print()
                Tools.print("Update label {} .......".format(epoch))
                self.net.eval()

                self.produce_class_1.reset()
                self.produce_class_2.reset()
                self.produce_class_3.reset()
                for batch_idx, (inputs, indexes) in enumerate(self.dataloader_usod):
                    inputs = inputs.type(torch.FloatTensor)
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    indexes = indexes.cuda() if torch.cuda.is_available() else indexes

                    return_m, return_d = self.net(inputs)

                    self.produce_class_1.cal_label(return_m["m1"]["smc_l2norm"], indexes)
                    self.produce_class_2.cal_label(return_m["m2"]["smc_l2norm"], indexes)
                    self.produce_class_3.cal_label(return_m["m3"]["smc_l2norm"], indexes)
                    pass

                Tools.print("Epoch: [{}] {}/{} {}/{} {}/{}".format(
                    epoch, self.produce_class_1.count, self.produce_class_1.count_2, self.produce_class_2.count,
                    self.produce_class_2.count_2, self.produce_class_3.count, self.produce_class_3.count_2))
                pass

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            all_loss_mic_1, all_loss_mic_2, all_loss_mic_3 = 0.0, 0.0, 0.0
            all_loss_sod_0, all_loss_sod_1, all_loss_sod_2, all_loss_sod_3 = 0.0, 0.0, 0.0, 0.0
            Tools.print()
            self.net.train()
            for i, (inputs, indexes) in enumerate(self.dataloader_usod):
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                indexes = indexes.cuda() if torch.cuda.is_available() else indexes
                self.optimizer.zero_grad()

                return_m, return_d = self.net(inputs)

                mic_labels_1 = self.produce_class_1.get_label(indexes)
                mic_labels_2 = self.produce_class_2.get_label(indexes)
                mic_labels_3 = self.produce_class_3.get_label(indexes)

                mic_labels_1 = mic_labels_1.cuda() if torch.cuda.is_available() else mic_labels_1
                mic_labels_2 = mic_labels_2.cuda() if torch.cuda.is_available() else mic_labels_2
                mic_labels_3 = mic_labels_3.cuda() if torch.cuda.is_available() else mic_labels_3

                sod_label = return_d["label"]["label"]
                mic_target_1 = return_m["m1"]["smc_logits"]
                mic_target_2 = return_m["m2"]["smc_logits"]
                mic_target_3 = return_m["m3"]["smc_logits"]
                sod_target_0 = return_d["d0"]["out_up_sigmoid"]
                sod_target_1 = return_d["d1"]["out_up_sigmoid"]
                sod_target_2 = return_d["d2"]["out_up_sigmoid"]
                sod_target_3 = return_d["d3"]["out_up_sigmoid"]

                loss, loss_mic, loss_sod = self.all_loss_fusion(
                    mic_target_1, mic_target_2, mic_target_3, mic_labels_1, mic_labels_2, mic_labels_3,
                    sod_target_0, sod_target_1, sod_target_2, sod_target_3, sod_label, only_mic=self.only_mic)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic_1 += loss_mic[0].item()
                all_loss_mic_2 += loss_mic[1].item()
                all_loss_mic_3 += loss_mic[2].item()
                all_loss_sod_0 += loss_sod[0].item()
                all_loss_sod_1 += loss_sod[1].item()
                all_loss_sod_2 += loss_sod[2].item()
                all_loss_sod_3 += loss_sod[3].item()
                if i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d}, b:{:4d}/{:4d}] l:{:.2f}/{:.2f} "
                                "mic1:{:.2f}/{:.2f} mic2:{:.2f}/{:.2f} mic3:{:.2f}/{:.2f} "
                                "sod0:{:.2f}/{:.2f} sod1:{:.2f}/{:.2f} sod2:{:.2f}/{:.2f} sod3:{:.2f}/{:.2f}".format(
                        epoch, self.epoch_num, i, len(self.dataloader_usod), all_loss/(i+1), loss.item(),
                        all_loss_mic_1/(i+1), loss_mic[0].item(),
                        all_loss_mic_2/(i+1), loss_mic[1].item(),
                        all_loss_mic_3/(i+1), loss_mic[2].item(),
                        all_loss_sod_0/(i+1), loss_sod[0].item(),
                        all_loss_sod_1/(i+1), loss_sod[1].item(),
                        all_loss_sod_2/(i+1), loss_sod[2].item(),
                        all_loss_sod_3/(i+1), loss_sod[3].item()))
                    pass

                pass

            ###########################################################################
            # 2 测试模型
            if epoch % eval_epoch_freq == 0:
                Tools.print()
                self.eval(self.net, epoch=epoch, is_test=False, small_data=20)
                Tools.print()
                self.eval(self.net, epoch=epoch, is_test=True, small_data=10)
                pass

            ###########################################################################
            # 3 保存模型
            if epoch % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "{}_train_{:.3f}.pth".format(epoch, all_loss / len(self.dataloader_usod))))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                pass

            pass

        pass

    @staticmethod
    def eval(net, epoch=0, is_test=True, small_data=5, print_ite_num=100, th_num=100, beta_2=0.3):
        which = "TE" if is_test else "TR"
        data_dir = '/mnt/4T/Data/SOD/DUTS/DUTS-{}'.format(which)
        image_dir, label_dir = 'DUTS-{}-Image'.format(which), 'DUTS-{}-Mask'.format(which)

        # 数据
        img_name_list = glob.glob(os.path.join(data_dir, image_dir, '*.jpg'))
        lbl_name_list = [os.path.join(data_dir, label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in img_name_list]
        dataset_eval_usod = DatasetEvalUSOD(img_name_list=img_name_list,
                                            lab_name_list=lbl_name_list, small_data=small_data)
        dataloader_eval_usod = DataLoader(dataset_eval_usod, 1, shuffle=False, num_workers=8)

        # 执行
        d_m = ["d0", "d1", "d2", "d3"]
        avg_mae = [0.0] * len(d_m)
        avg_prec = np.zeros(shape=(len(d_m), th_num)) + 1e-6
        avg_recall = np.zeros(shape=(len(d_m), th_num)) + 1e-6
        net.eval()
        for i, (inputs, labels) in enumerate(dataloader_eval_usod):
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs

            now_label = labels.squeeze().data.numpy() / 255
            return_m, return_d = net(inputs)

            mae_list = [0.0] * len(d_m)
            for index, key in enumerate(d_m):
                d_out = return_d[key]["out_up_sigmoid"].squeeze().cpu().data.numpy()
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
                    now += "{}:{:.2f}/{:.2f} ".format(key, avg_mae[index] / (i + 1), mae_list[index])
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
                                                key, avg_mae[index], score[index].max()))
            pass

        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    bas_runner = BASRunner(batch_size_train=8, only_mic=False,
                           model_dir="./saved_models/mtt_mic5_decoder9_m1_label64_mic_no_only")
    # bas_runner.load_model('./saved_models/my_train5_diff_aug_mask/125_train_6.569.pth')
    bas_runner.train()
    pass
