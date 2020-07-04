import os
import math
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from pydensecrf.utils import unary_from_softmax
from torch.utils.data import DataLoader, Dataset


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
    def crf_torch(cls, img, annotation, t=5):
        img_data = np.asarray(img.detach().cpu() * 255, dtype=np.uint8)
        annotation_data = np.asarray(annotation.detach().cpu())
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            result_one = cls.crf(img_data_one, annotation_data_one, t=t)
            result.append(np.expand_dims(result_one, axis=0))
            pass
        return torch.tensor(np.asarray(result)).cuda()

    pass


#######################################################################################################################
# 1 Data

class RandomResizedCrop(transforms.RandomResizedCrop):

    def __call__(self, img, parm, his=None, image_crf=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        # parm.extend([i, j, h, w])
        if his is not None:
            his = transforms.functional.resized_crop(his, i, j, h, w, self.size, self.interpolation)
        if image_crf is not None:
            image_crf = transforms.functional.resized_crop(image_crf, i, j, h, w, self.size, self.interpolation)
        return img, parm, his, image_crf

    pass


class RandomResized(object):

    def __init__(self, img_w=300, img_h=300):
        self.img_w, self.img_h = img_w, img_h
        pass

    def __call__(self, img, parm, his=None, image_crf=None):
        img = img.resize((self.img_w, self.img_h))
        if his is not None:
            his = his.resize((self.img_w, self.img_h))
        if image_crf is not None:
            image_crf = image_crf.resize((self.img_w, self.img_h))
        return img, parm, his, image_crf

    pass


class ColorJitter(transforms.ColorJitter):

    def __call__(self, img, parm, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, his, image_crf

    pass


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, parm, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, his, image_crf

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, parm, his=None, image_crf=None):
        if random.random() < self.p:
            parm.append(1)
            img = transforms.functional.hflip(img)
            if his is not None:
                his = transforms.functional.hflip(his)
            if image_crf is not None:
                image_crf = transforms.functional.hflip(image_crf)
        else:
            parm.append(0)
        return img, parm, his, image_crf

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, parm, his=None, image_crf=None):
        img = super().__call__(img)
        if his is not None:
            his = super().__call__(his)
        if image_crf is not None:
            image_crf = super().__call__(image_crf)
        return img, parm, his, image_crf

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, parm, his=None, image_crf=None):
        img = super().__call__(img)
        return img, parm, his, image_crf

    pass


class Compose(transforms.Compose):
    def __call__(self, img, parm, his=None, image_crf=None):
        for t in self.transforms:
            img, parm, his, image_crf = t(img, parm, his, image_crf)
        return img, parm, his, image_crf

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, his_name_list=None, is_train=True, only_mic=False, size=256):
        self.image_name_list = img_name_list
        self.history_name_list = his_name_list
        self.has_history = self.history_name_list is not None

        self.is_train = is_train

        transform_train = Compose([RandomResizedCrop(size=size, scale=(0.3, 1.)), ColorJitter(0.4, 0.4, 0.4, 0.4),
                                   RandomGrayscale(p=0.2), RandomHorizontalFlip(),
                                   ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform_train_sod = Compose([RandomResized(img_w=size, img_h=size), RandomHorizontalFlip(),
                                       ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.transform_train = transform_train if only_mic else transform_train_sod
        self.transform_test = Compose([RandomResized(img_w=size, img_h=size),
                                       ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # self.image_list = self._pre_load_data()
        self.image_list = None
        pass

    def __len__(self):
        return len(self.image_name_list)

    def _pre_load_data(self):
        Tools.print("load data")
        image_list = []
        for image_name in self.image_name_list:
            image = Image.open(image_name).convert("RGB")
            image_list.append(image)
            pass
        Tools.print("load data end")
        return image_list

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB") \
            if self.image_list is None else self.image_list[idx]

        history = None
        if self.has_history:
            h_path = self.history_name_list[idx]
            history = Image.open(h_path).convert("L")if os.path.exists(h_path) else Image.new("L", size=image.size)
            pass

        param = [image.size[0], image.size[1]]
        image, param, history, image_for_crf = self.transform_train(
            image, param, history, image) if self.is_train else self.transform_test(image, param, history, image)

        return image, history if self.has_history else image, image_for_crf, np.asarray(param), idx

    def save_history(self, idx, his, param, name=None):
        """ his: [0, 1] """
        if self.history_name_list is not None:
            h_path = self.history_name_list[idx]
            if name is not None:
                h_path = "{}_{}.{}".format(os.path.splitext(h_path)[0], name, os.path.splitext(h_path)[1])
            h_path = Tools.new_dir(h_path)

            his = np.transpose(his, (1, 2, 0)) if his.shape[0] == 1 or his.shape[0] == 3 else his
            im = Image.fromarray(np.asarray(his * 255, dtype=np.uint8)).resize((param[0], param[1]))
            im = im.transpose(Image.FLIP_LEFT_RIGHT) if param[2] == 1 else im
            im.save(h_path)
        pass

    pass


#######################################################################################################################
# 2 Model


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, ks=3, padding=1, has_relu=True, has_bn=True, has_bias=True):
        super(ConvBlock, self).__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=padding, bias=has_bias)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

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

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

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


class VGG16BN(nn.Module):

    def __init__(self):
        super(VGG16BN, self).__init__()
        #  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.layer1 = nn.Sequential(*[ConvBlock(3, 64, 1, has_relu=True),
                                      ConvBlock(64, 64, 1, has_relu=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)])  # 112
        self.layer2 = nn.Sequential(*[ConvBlock(64, 128, 1, has_relu=True),
                                      ConvBlock(128, 128, 1, has_relu=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)])  # 56
        self.layer3 = nn.Sequential(*[ConvBlock(128, 256, 1, has_relu=True),
                                      ConvBlock(256, 256, 1, has_relu=True),
                                      ConvBlock(256, 256, 1, has_relu=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)])  # 28
        self.layer4 = nn.Sequential(*[ConvBlock(256, 512, 1, has_relu=True),
                                      ConvBlock(512, 512, 1, has_relu=True),
                                      ConvBlock(512, 512, 1, has_relu=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)])  # 14
        self.layer5 = nn.Sequential(*[ConvBlock(512, 512, 1, has_relu=True),
                                      ConvBlock(512, 512, 1, has_relu=True),
                                      ConvBlock(512, 512, 1, has_relu=True)])  # 14
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

    pass


class BASNet(nn.Module):

    def __init__(self, clustering_num, has_attention=True, sigmoid_attention=True, softmax_attention=False):
        super(BASNet, self).__init__()
        self.clustering_num = clustering_num
        self.has_attention = has_attention
        self.sigmoid_attention = sigmoid_attention
        self.softmax_attention = softmax_attention

        # -------------Encoder--------------
        self.vgg16 = VGG16BN()

        # -------------MIC-------------
        self.mic_l2norm = MICNormalize(2)
        self.mic_max_pool = nn.MaxPool2d(2, 2)
        self.mic_b1 = ConvBlock(512, 512, 1, has_relu=True)  # 10 8 9
        self.mic_b2 = ConvBlock(512, 512, 1, has_relu=True)
        self.mic_c1 = ConvBlock(512, self.clustering_num, has_relu=False)

        self.mic_for_sigmoid_attention = ConvBlock(self.clustering_num, 1, 1, ks=1, padding=0, has_relu=False)
        self.mic_for_softmax_attention = ConvBlock(self.clustering_num, 1, 1, ks=1, padding=0, has_relu=False)

        # -------------Decoder-------------
        # self.decoder_0_b1 = ResBlock(512, 512)  # 10
        # self.decoder_0_b2 = ResBlock(512, 512)  # 10
        # self.decoder_0_c = ConvBlock(512, 512, has_relu=True)  # 10

        # self.decoder_1_b1 = ResBlock(512, 512)  # 20
        # self.decoder_1_b2 = ResBlock(512, 512)  # 20
        # self.decoder_1_c = ConvBlock(512, 256, has_relu=True)  # 20

        # self.decoder_2_b1 = ResBlock(256, 256)  # 40
        # self.decoder_2_b2 = ResBlock(256, 256)  # 40
        # self.decoder_2_c = ConvBlock(256, 128, has_relu=True)  # 40

        # self.decoder_3_b1 = ResBlock(128, 128)  # 80
        # self.decoder_3_b2 = ResBlock(128, 128)  # 80
        # self.decoder_3_out = nn.Conv2d(128, 1, 3, padding=1, bias=False)  # 80
        pass

    def forward(self, x):
        return_result = {}

        # -------------Encoder-------------
        e1 = self.vgg16.layer1(x)  # 64 * 160 * 160
        e2 = self.vgg16.layer2(e1)  # 128 * 80 * 80
        e3 = self.vgg16.layer3(e2)  # 256 * 40 * 40
        e4 = self.vgg16.layer4(e3)  # 512 * 20 * 20
        e5 = self.vgg16.layer5(e4)  # 512 * 20 * 20

        # -------------MIC-------------
        e5_pool = self.mic_max_pool(e5)
        mic_feature = self.mic_b2(self.mic_b1(e5_pool))  # 512 * 10 * 10

        mic = self.mic_c1(mic_feature)  # 128 * 10 * 10
        return_result = self.attention_mic(mic=mic, return_result=return_result)

        # -------------Label-------------
        # cam = self.cluster_activation_map(smc_logits, mic)  # 簇激活图：Cluster Activation Map
        # cam_norm = self._feature_norm(cam)
        # cam_up_norm = self._up_to_target(cam_norm, x)
        # return_result["cam_norm"] = cam_norm
        # return_result["cam_up_norm"] = cam_up_norm

        # -------------Decoder-------------
        # d0 = self.decoder_0_b2(self.decoder_0_b1(mic_feature))  # 512 * 10 * 10
        # d0_d1 = self._up_to_target(self.decoder_0_c(d0), e4) + e4  # 512 * 20 * 20
        # d1 = self.decoder_1_b2(self.decoder_1_b1(d0_d1))  # 512 * 20 * 20
        # d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 40 * 40
        # d2 = self.decoder_2_b2(self.decoder_2_b1(d1_d2))  # 256 * 40 * 40
        # d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 128 * 80 * 80
        # d3 = self.decoder_3_b2(self.decoder_3_b1(d2_d3))  # 128 * 80 * 80

        # d3_out = self.decoder_3_out(d3)  # 1 * 80 * 80
        # d3_out_sigmoid = torch.sigmoid(d3_out)  # 1 * 80 * 80  # 小输出
        # d3_out_up = self._up_to_target(d3_out, x)  # 1 * 320 * 320
        # d3_out_up_sigmoid = torch.sigmoid(d3_out_up)  # 1 * 320 * 320  # 大输出

        # return_result["output"] = d3_out_sigmoid
        # return_result["output_up"] = d3_out_up_sigmoid

        # --------------Result-------------
        return return_result

    def attention_mic(self, mic, return_result={}):
        view_shape = (mic.size()[0], -1)

        mic_attention = mic
        if self.has_attention:
            att = self.mic_for_sigmoid_attention(mic_attention)  # 1 * 10 * 10  attention
            return_result["attention_before_sigmoid"] = att
            if self.sigmoid_attention:  # sigmoid attention
                att = torch.sigmoid(att)  # 1 * 10 * 10
                return_result["attention_after_sigmoid"] = att
                pass

            # Attention 1
            mic_attention = mic * att

            if self.softmax_attention:  # softmax attention
                att = self.mic_for_softmax_attention(mic_attention)
                return_result["attention_before_softmax"] = att
                att = torch.softmax(att.view(view_shape), dim=-1).view_as(att)
                return_result["attention_after_softmax"] = self._feature_norm(att)

                # Attention 2
                mic_attention = mic_attention * att

                smc_logits = mic_attention.sum(3).sum(2).view(view_shape)  # sum global
            else:
                smc_logits = F.adaptive_avg_pool2d(mic_attention, output_size=(1, 1)).view(view_shape)  # avg global
                pass
            pass
        else:
            smc_logits = F.adaptive_avg_pool2d(mic_attention, output_size=(1, 1)).view(view_shape)
            pass

        return_result["smc_logits"] = smc_logits
        return_result["smc_l2norm"] = self.mic_l2norm(smc_logits)
        return return_result

    @staticmethod
    def cluster_activation_map(smc_logits, mic_feature):
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        cam = torch.cat([mic_feature[i:i+1, top_k_index[i], :, :] for i in range(mic_feature.size()[0])])
        return cam

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

    def __init__(self, batch_size_train=8, clustering_num=128, clustering_ratio=2, has_crf=False, size=256,
                 only_mic=False, only_decoder=False, has_history=False, learning_rate=None,
                 sigmoid_attention=True, softmax_attention=False, has_attention=True,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic_only",
                 history_dir="./history/my_train_mic5_large_history1", num_workers=32):
        self.batch_size_train = batch_size_train
        self.only_mic = only_mic
        self.only_decoder = only_decoder
        self.has_crf = has_crf

        # History
        self.has_history = has_history
        self.history_dir = Tools.new_dir(history_dir) if self.has_history else history_dir

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list, self.tra_his_name_list = self.get_tra_img_label_name()
        self.tra_his_name_list = self.tra_his_name_list if self.has_history else None
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list, only_mic=self.only_mic,
                                        his_name_list=self.tra_his_name_list, is_train=True, size=size)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train,
                                          shuffle=True, num_workers=num_workers)

        # Model
        self.net = BASNet(clustering_num=clustering_num, has_attention=has_attention,
                          sigmoid_attention=sigmoid_attention, softmax_attention=softmax_attention)

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
        self.produce_class1 = MICProduceClass(n_sample=len(self.dataset_usod),
                                              out_dim=clustering_num, ratio=clustering_ratio)
        self.produce_class2 = MICProduceClass(n_sample=len(self.dataset_usod),
                                              out_dim=clustering_num, ratio=clustering_ratio)
        self.produce_class1.init()
        self.produce_class2.init()

        # Loss and optimizer
        self.bce_loss = nn.BCELoss().cuda()
        self.mic_loss = nn.CrossEntropyLoss().cuda()
        self.learning_rate = [[0, 0.001], [100, 0.0001], [150, 0.00001]] if learning_rate is None else learning_rate
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

    def all_loss_fusion(self, mic_out, mic_labels, sod_output, sod_label, only_mic=False, sod_w=2, ignore_label=255.0):
        loss_mic = self.mic_loss(mic_out, mic_labels)
        loss_mic = loss_mic

        positions = sod_label.view(-1, 1) != ignore_label
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])

        loss_all = loss_mic + sod_w * loss_bce
        if only_mic:
            loss_all = loss_mic
            pass
        if self.only_decoder:
            loss_all = loss_bce
            pass

        return loss_all, loss_mic, loss_bce

    def all_loss_fusion_mic(self, mic_out, mic_labels):
        loss_mic = self.mic_loss(mic_out, mic_labels)
        return loss_mic, loss_mic, loss_mic

    def save_history_info(self, histories, indexes, params, name=None):
        for history, param, index in zip(histories, params, indexes):
            self.dataset_usod.save_history(idx=int(index), name=name, param=param,
                                           his=np.asarray(history.squeeze().detach().cpu()))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50, t=5, sod_w=2):

        if start_epoch >= 0:
            self.net.eval()
            Tools.print()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class1.reset()
            with torch.no_grad():
                for _idx, (inputs, histories, image_for_crf, params, indexes) in tqdm(
                        enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    return_result = self.net(inputs)

                    self.produce_class1.cal_label(return_result["smc_l2norm"], indexes)
                    pass
                pass
            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Update: [{}] 1-{}/{}".format(start_epoch, self.produce_class1.count,
                                                      self.produce_class1.count_2))
            pass

        all_loss = 0
        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic, all_loss_sod = 0.0, 0.0, 0.0
            self.net.train()

            self.produce_class1.reset()
            for i, (inputs, histories, image_for_crf, params, indexes) in tqdm(
                    enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                histories = histories.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                self.optimizer.zero_grad()

                return_result = self.net(inputs)

                ######################################################################################################
                # MIC
                self.produce_class1.cal_label(return_result["smc_logits"], indexes)
                mic_target = return_result["smc_logits"]
                mic_labels = self.produce_class2.get_label(indexes).cuda()

                loss, loss_mic, loss_sod = self.all_loss_fusion_mic(mic_target, mic_labels)
                ######################################################################################################

                ######################################################################################################
                # # 历史信息 = 历史信息 + CAM + SOD
                # histories = histories
                # sod_label_ori = return_result["cam_up_norm"].detach()  # CAM
                # sod_output = return_result["output_up"]  # Predict

                # ignore_label = 255.0
                # sod_label = sod_label_ori
                # if self.has_history:
                #     self.save_history_info(sod_output, params=params, indexes=indexes, name="output")
                #     self.save_history_info(sod_label_ori, params=params, indexes=indexes, name="label_before_crf")
                #
                #     # 1
                #     # sod_label = sod_label_crf
                #
                #     if self.has_crf:
                #         sod_label_crf = CRFTool.crf_torch(image_for_crf, sod_label_ori, t=t)
                #
                #         # 2
                #         ignore_label = 0.5
                #         sod_label = torch.ones_like(sod_label_ori) * ignore_label
                #         sod_label[(sod_label_ori > 0.5) & (sod_label_crf > 0.5)] = 1.0
                #         sod_label[(sod_label_ori < 0.3) & (sod_label_crf < 0.5)] = 0.0
                #
                #         self.save_history_info(sod_label_crf, params=params, indexes=indexes, name="label_after_crf")
                #         pass
                #
                #     self.save_history_info(histories=histories, params=params, indexes=indexes)
                #     self.save_history_info(sod_label, params=params, indexes=indexes, name="label")
                #     pass
                ######################################################################################################

                # loss, loss_mic, loss_sod = self.all_loss_fusion(
                #     mic_target, mic_labels, sod_output, sod_label,
                #     only_mic=self.only_mic, sod_w=sod_w, ignore_label=ignore_label)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic += loss_mic.item()
                all_loss_sod += loss_sod.item()
                if print_ite_num != 0 and i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d}, b:{:4d}/{:4d}] l:{:.2f}/{:.2f} mic:{:.2f}/{:.2f} sod:{:.2f}/{:.2f}"
                                "".format(epoch, epoch_num, i, len(self.dataloader_usod),
                                          all_loss/(i+1), loss.item(), all_loss_mic/(i+1),
                                          loss_mic.item(), all_loss_sod/(i+1), loss_sod.item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic:{:.3f} sod:{:.3f}".format(
                epoch, epoch_num, all_loss / (len(self.dataloader_usod) + 1),
                all_loss_mic / (len(self.dataloader_usod) + 1), all_loss_sod / (len(self.dataloader_usod) + 1)))

            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, self.produce_class1.count, self.produce_class1.count_2))

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

    def vis(self):
        self.net.eval()
        with torch.no_grad():
            for _idx, (inputs, histories, image_for_crf, params, indexes) in tqdm(
                    enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()

                self.save_history_info(image_for_crf, indexes=indexes, params=params, name="image")

                return_result = self.net(inputs)
                for key in return_result.keys():
                    if "attention_after" in key:
                        value = return_result[key].detach()
                        self.save_history_info(value, indexes=indexes, params=params, name=key)
                        # value_crf = CRFTool.crf_torch(image_for_crf, value, t=5)
                        # self.save_history_info(value_crf, indexes=indexes, params=params, name="{}_crf".format(key))
                        pass
                    pass

                pass
            pass
        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    """
    2020-07-03 02:09:55 Epoch:499, lr=0.00010
    2020-07-03 02:11:00 [E:499/500] loss:1.958 mic:1.958 sod:1.958
    2020-07-03 02:11:00 Train: [499] 1-6250/754
    2020-07-03 02:11:00 Save Model to ../BASNetTemp/saved_models/History1_CRF_FULL_1MIC_VGG_256/500_train_1.976.pth
    
    2020-07-03 02:10:31 Epoch:499, lr=0.00010
    2020-07-03 02:11:33 [E:499/500] loss:1.964 mic:1.964 sod:1.964
    2020-07-03 02:11:33 Train: [499] 1-6290/882
    Save Model to ../BASNetTemp/saved_models/History1_CRF_FULL_1MIC_VGG_Attention_256_sigmoidTrue/500_train_1.982.pth
    
    
    MIC1_VGG_256_AttentionTrue_sigmoidTrue_softmaxTrue
    2020-07-03 23:29:26 Epoch:499, lr=0.00010
    2020-07-03 23:31:10 [E:499/500] loss:2.272 mic:2.272 sod:2.272
    2020-07-03 23:31:10 Train: [499] 1-7057/1038
    Save Model to ../BASNetTemp/saved_models/MIC1_VGG_256_AttentionTrue_sigmoidTrue_softmaxTrue/500_train_2.279.pth
    
    MIC1_VGG_256_AttentionTrue_sigmoidTrue_softmaxFalse
    2020-07-03 23:49:27 Epoch:499, lr=0.00010
    2020-07-03 23:51:06 [E:499/500] loss:2.316 mic:2.316 sod:2.316
    2020-07-03 23:51:06 Train: [499] 1-7151/1038
    Save Model to ../BASNetTemp/saved_models/MIC1_VGG_256_AttentionTrue_sigmoidTrue_softmaxFalse/500_train_2.323.pth

    MIC1_VGG_256_AttentionTrue_sigmoidFalse_softmaxFalse
    2020-07-03 23:57:44 Epoch:499, lr=0.00010
    2020-07-03 23:59:24 [E:499/500] loss:2.220 mic:2.220 sod:2.220
    2020-07-03 23:59:24 Train: [499] 1-6929/916
    Save Model to ../BASNetTemp/saved_models/MIC1_VGG_256_AttentionTrue_sigmoidFalse_softmaxFalse/500_train_2.227.pth

    MIC1_VGG_256_AttentionTrue_sigmoidFalse_softmaxTrue
    2020-07-03 23:58:06 Epoch:499, lr=0.00010
    2020-07-03 23:59:43 [E:499/500] loss:2.222 mic:2.222 sod:2.222
    2020-07-03 23:59:43 Train: [499] 1-6911/887
    Save Model to ../BASNetTemp/saved_models/MIC1_VGG_256_AttentionTrue_sigmoidFalse_softmaxTrue/500_train_2.229.pth
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # _lr = [[0, 0.01], [300, 0.001], [400, 0.0001]]
    _lr = [[0, 0.0001], [10, 0.001], [20, 0.01], [200, 0.001], [400, 0.0001]]
    _epoch_num = 500
    _num_workers = 32
    _t = 5
    _sod_w = 2
    _size = 256
    _has_history = True
    _has_attention = True
    _sigmoid_attention = True
    _softmax_attention = True
    _name = "MIC1_VGG_{}_Attention{}_sigmoid{}_softmax{}".format(
        _size, _has_attention, _sigmoid_attention, _softmax_attention)
    Tools.print(_name)

    bas_runner = BASRunner(batch_size_train=16 * 2, clustering_num=128, clustering_ratio=2,
                           learning_rate=_lr, num_workers=_num_workers, size=_size,
                           sigmoid_attention=_sigmoid_attention,
                           softmax_attention=_softmax_attention,
                           has_attention=_has_attention,
                           has_history=_has_history, only_decoder=False, only_mic=True, has_crf=False,
                           data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           history_dir="../BASNetTemp/history/{}".format(_name),
                           model_dir="../BASNetTemp/saved_models/{}".format(_name))
    bas_runner.load_model(
        '../BASNetTemp/saved_models/MIC1_VGG_256_AttentionTrue_sigmoidTrue_softmaxTrue/500_train_2.279.pth')
    # bas_runner.train(epoch_num=_epoch_num, start_epoch=0, t=_t, sod_w=_sod_w, print_ite_num=0)
    bas_runner.vis()
    pass
