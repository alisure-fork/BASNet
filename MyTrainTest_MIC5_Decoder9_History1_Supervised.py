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
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from multiprocessing.pool import Pool
from pydensecrf.utils import unary_from_softmax
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock as ResBlock


#######################################################################################################################
# 1 Data


class RandomResized(object):

    def __init__(self, img_w=300, img_h=300):
        self.img_w, self.img_h = img_w, img_h
        pass

    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        img = img.resize((self.img_w, self.img_h))
        if label is not None:
            label = label.resize((self.img_w, self.img_h))
        if his is not None:
            his = his.resize((self.img_w, self.img_h))
        if image_crf is not None:
            image_crf = image_crf.resize((self.img_w, self.img_h))
        return img, parm, label, his, image_crf

    pass


class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, label, his, image_crf

    pass


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, label, his, image_crf

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        if random.random() < self.p:
            parm.append(1)
            img = transforms.functional.hflip(img)
            if label is not None:
                label = transforms.functional.hflip(label)
            if his is not None:
                his = transforms.functional.hflip(his)
            if image_crf is not None:
                image_crf = transforms.functional.hflip(image_crf)
        else:
            parm.append(0)
        return img, parm, label, his, image_crf

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        img = super().__call__(img)
        if label is not None:
            label = super().__call__(label)
        if his is not None:
            his = super().__call__(his)
        if image_crf is not None:
            image_crf = super().__call__(image_crf)
        return img, parm, label, his, image_crf

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, label, his, image_crf

    pass


class Compose(transforms.Compose):
    def __call__(self, img, parm, label=None, his=None, image_crf=None):
        for t in self.transforms:
            img, parm, label, his, image_crf = t(img, parm, label, his, image_crf)
        return img, parm, label, his, image_crf

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, lab_name_list=None, his_name_list=None, is_train=True):
        self.image_name_list = img_name_list
        self.label_name_list = lab_name_list
        self.has_label = self.label_name_list is not None
        self.history_name_list = his_name_list
        self.has_history = self.history_name_list is not None

        self.is_train = is_train
        self.transform_train = Compose([RandomResized(img_w=256, img_h=256), ColorJitter(0.4, 0.4, 0.4, 0.4),
                                        RandomGrayscale(p=0.2), RandomHorizontalFlip(), ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_test = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        label = None
        if self.has_label:
            l_path = self.label_name_list[idx]
            label = Image.open(l_path).convert("L")if os.path.exists(l_path) else Image.new("L", size=image.size)
            pass

        history = None
        if self.has_history:
            h_path = self.history_name_list[idx]
            history = Image.open(h_path).convert("L")if os.path.exists(h_path) else Image.new("L", size=image.size)
            pass

        param = [image.size[0], image.size[1]]
        image, param, label, history, image_for_crf = self.transform_train(image, param, label, history, image) \
            if self.is_train else self.transform_test(image, param, label, history, image)

        return image, label, history if self.has_history else image, image_for_crf, np.asarray(param), idx

    def save_history(self, idx, his, param, name=None):
        """ his: [0, 1] """
        if self.history_name_list is not None:
            h_path = self.history_name_list[idx]
            if name is not None:
                h_path = "{}_{}.{}".format(os.path.splitext(h_path)[0], name, os.path.splitext(h_path)[1])
            h_path = Tools.new_dir(h_path)

            im = Image.fromarray(np.asarray(his * 255, dtype=np.uint8)).resize((param[0], param[1]))
            im = im.transpose(Image.FLIP_LEFT_RIGHT) if param[2] == 1 else im
            im.save(h_path)
        pass

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
        self.mic_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # MIC 1
        self.mic_1_b1 = ResBlock(512, 512)  # 28
        self.mic_1_b2 = ResBlock(512, 512)
        self.mic_1_b3 = ResBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)

        # MIC 2
        self.mic_2_b1 = ResBlock(512, 512)  # 14
        self.mic_2_b2 = ResBlock(512, 512)
        self.mic_2_b3 = ResBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)

        # MIC 3
        self.mic_3_b1 = ResBlock(512, 512)  # 7
        self.mic_3_b2 = ResBlock(512, 512)
        self.mic_3_b3 = ResBlock(512, 512)
        self.mic_3_c1 = ConvBlock(512, self.clustering_num_list[2], has_relu=True)

        # -------------Decoder-------------
        # Decoder 1
        self.decoder_1_b1 = ResBlock(512, 512)  # 28
        self.decoder_1_b2 = ResBlock(512, 512)  # 28
        self.decoder_1_b3 = ResBlock(512, 512)  # 28
        self.decoder_1_c = ConvBlock(512, 256, has_relu=True)  # 28

        # Decoder 2
        self.decoder_2_b1 = ResBlock(256, 256)  # 56
        self.decoder_2_b2 = ResBlock(256, 256)  # 56
        self.decoder_2_b3 = ResBlock(256, 256)  # 56
        self.decoder_2_c = ConvBlock(256, 128, has_relu=True)  # 56

        # Decoder 3
        self.decoder_3_b1 = ResBlock(128, 128)  # 112
        self.decoder_3_b2 = ResBlock(128, 128)  # 112
        self.decoder_3_b3 = ResBlock(128, 128)  # 112
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
        mic_2_feature = self.mic_pool(self.mic_2_b3(self.mic_2_b2(self.mic_2_b1(mic_1_feature))))  # 512 * 14 * 14
        mic_2 = self.mic_2_c1(mic_2_feature)  # 256 * 14 * 14
        smc_logits_2, smc_l2norm_2 = self.salient_map_clustering(mic_2)
        return_m2 = {"smc_logits": smc_logits_2, "smc_l2norm": smc_l2norm_2}

        # 3
        mic_3_feature = self.mic_pool(self.mic_3_b3(self.mic_3_b2(self.mic_3_b1(mic_2_feature))))  # 512 * 7 * 7
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

        cam_up_norm = (cam_1_up_norm + cam_2_up_norm + cam_3_up_norm) / 3
        # cam_up_norm = (cam_2_up_norm + cam_3_up_norm) / 2
        # cam_up_norm = cam_3_up_norm

        # label = self.salient_map_divide(cam_up_norm, obj_th=0.8, bg_th=0.2, more_obj=False)  # 显著图划分
        label = cam_up_norm

        return_l = {"label": label, "cam_up_norm": cam_up_norm, "cam_1_up_norm": cam_1_up_norm,
                    "cam_2_up_norm": cam_2_up_norm, "cam_3_up_norm": cam_3_up_norm}

        # -------------Decoder-------------
        # decoder
        d1 = self.decoder_1_b3(self.decoder_1_b2(self.decoder_1_b1(e4)))  # 512 * 28 * 28
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 56 * 56

        d2 = self.decoder_2_b3(self.decoder_2_b2(self.decoder_2_b1(d1_d2)))  # 256 * 56 * 56
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 128 * 112 * 112

        # d3
        d3 = self.decoder_3_b3(self.decoder_3_b2(self.decoder_3_b1(d2_d3)))  # 128 * 112 * 112
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
                 clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2,
                 only_mic=False, only_decoder=False, only_label=False, has_history=False,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic_only",
                 history_dir="./history/my_train_mic5_large_history1"):
        self.batch_size_train = batch_size_train
        self.only_mic = only_mic
        self.only_decoder = only_decoder
        self.only_label = only_label
        self.has_history = has_history

        # History
        self.history_dir = Tools.new_dir(history_dir)

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, self.tra_lbl_name_list, self.tra_his_name_list = self.get_tra_img_label_name()

        self.tra_his_name_list = self.tra_his_name_list if self.has_history else None
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list, lab_name_list=self.tra_lbl_name_list,
                                        his_name_list=self.tra_his_name_list, is_train=True)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=8)

        # Model
        self.net = BASNet(3, clustering_num_list=[clustering_num_1, clustering_num_2, clustering_num_3])

        ###########################################################################
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
            cudnn.benchmark = True
        ###########################################################################
        if self.only_decoder:
            for name, param in self.net.named_parameters():
                if "decoder" not in name:
                    param.requires_grad = False
                pass
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
        self.learning_rate = [[0, 0.001], [100, 0.0001], [150, 0.00001]]
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate[0][1], betas=(0.9, 0.999), weight_decay=0)
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
        return 1 / (1 + torch.exp(-(x - a)))
        pass

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_3_out, mic_labels_1, mic_labels_2, mic_labels_3,
                        sod_output, sod_label, sod_w=2, ignore_label=255.0):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)
        loss_mic_3 = self.mic_loss(mic_3_out, mic_labels_3)
        loss_mic = loss_mic_1 + loss_mic_2 + loss_mic_3

        positions = sod_label.view(-1, 1) != ignore_label
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])

        if self.only_mic:
            loss_all = loss_mic
        elif self.only_label or self.only_decoder:
            loss_all = loss_bce
        else:
            loss_all = loss_mic + sod_w * loss_bce
            pass

        return loss_all, [loss_mic_1, loss_mic_2, loss_mic_3], loss_bce

    def save_history_info(self, histories, params, indexes, name=None):
        for history, param, index in zip(histories, params, indexes):
            self.dataset_usod.save_history(idx=int(index), name=name, param=np.asarray(param),
                                           his=np.asarray(history.squeeze().detach().cpu()))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50, sod_w=2):

        if start_epoch > 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            with torch.no_grad():
                for _idx, (inputs, targets, histories, image_for_crf, params, indexes) in tqdm(
                        enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
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

            for i, (inputs, targets, histories, image_for_crf, params, indexes) in tqdm(
                    enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                targets = targets.type(torch.FloatTensor).cuda()
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
                # 历史信息 = 历史信息 + CAM + SOD
                histories = histories
                sod_label_ori = return_l["label"]["label"].detach()  # CAM
                sod_output = return_l["output"]["output"]  # Predict
                sod_label = targets

                if self.has_history:
                    self.save_history_info(histories=histories, params=params, indexes=indexes)
                    self.save_history_info(sod_label_ori, params=params, indexes=indexes, name="label_before_crf")
                    self.save_history_info(sod_output, params=params, indexes=indexes, name="output")
                    self.save_history_info(targets, params=params, indexes=indexes, name="label")
                    pass
                ######################################################################################################

                loss, loss_mic, loss_sod = self.all_loss_fusion(
                    mic_target_1, mic_target_2, mic_target_3, mic_labels_1, mic_labels_2, mic_labels_3,
                    sod_output, sod_label, sod_w=sod_w)
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
            # 2 保存模型
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

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    _sod_w = 2
    _name = "my_train_mic5_large_history1_Supervised"
    Tools.print(_name)

    bas_runner = BASRunner(batch_size_train=12,
                           only_mic=False, only_decoder=False, only_label=True, has_history=True,
                           clustering_num_1=128 * 4, clustering_num_2=128 * 4, clustering_num_3=128 * 4,
                           data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           history_dir="../BASNetTemp/history/{}".format(_name),
                           model_dir="../BASNetTemp/saved_models/{}".format(_name))
    # bas_runner.load_model('../BASNetTemp/saved_models/my_train_mic5_large/500_train_0.880.pth')
    bas_runner.load_model("../BASNetTemp/saved_models/my_train_mic5_large_history1_123/460_train_9.385.pth")
    bas_runner.train(epoch_num=200, start_epoch=1, sod_w=_sod_w)
    pass
