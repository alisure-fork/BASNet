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
import torch.nn.functional as F
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
    def crf_torch(cls, img, annotation, t=5):
        img_data = np.asarray(img, dtype=np.uint8)
        annotation_data = np.asarray(annotation)
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            result_one = cls.crf(img_data_one, annotation_data_one, t=t)
            result.append(np.expand_dims(result_one, axis=0))
            pass
        return torch.tensor(np.asarray(result))

    pass


#######################################################################################################################
# 1 Data


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __call__(self, img, image_crf=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if image_crf is not None:
            image_crf = transforms.functional.resized_crop(image_crf, i, j, h, w, self.size, self.interpolation)
        return img, image_crf

    pass


class FixedResized(object):

    def __init__(self, img_w=300, img_h=300):
        self.img_w, self.img_h = img_w, img_h
        pass

    def __call__(self, img, image_crf=None):
        img = img.resize((self.img_w, self.img_h))
        if image_crf is not None:
            image_crf = image_crf.resize((self.img_w, self.img_h))
        return img, image_crf

    pass


class ColorJitter(transforms.ColorJitter):

    def __call__(self, img, image_crf=None):
        img = super().__call__(img)
        return img, image_crf

    pass


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, image_crf=None):
        img = super().__call__(img)
        return img, image_crf

    pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, image_crf=None):
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            if image_crf is not None:
                image_crf = transforms.functional.hflip(image_crf)
                pass
            pass
        return img, image_crf

    pass


class ToTensor(transforms.ToTensor):
    def __call__(self, img, image_crf=None):
        img = super().__call__(img)
        if image_crf is not None:
            image_crf = super().__call__(image_crf)
        return img, image_crf

    pass


class Normalize(transforms.Normalize):
    def __call__(self, img, image_crf=None):
        img = super().__call__(img)
        return img, image_crf

    pass


class Compose(transforms.Compose):
    def __call__(self, img, image_crf=None):
        for t in self.transforms:
            img, image_crf = t(img, image_crf)
        return img, image_crf

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, size_train=224):
        self.image_name_list = img_name_list

        self.transform = Compose([RandomResizedCrop(size=size_train, scale=(0.3, 1.)),
                                  ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2),
                                  RandomHorizontalFlip(), ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image, image_for_crf = self.transform(image, image)
        return image, image_for_crf, idx

    pass


class DatasetUSODVIS(Dataset):

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
        im = Image.open(self.image_name_list[idx]).convert("RGB")

        image_simple, image_for_crf = self.transform_vis_simple(im, im)
        image_list = [image_simple]
        for i in range(self.multi_num - 1):
            image, _ = self.transform(im, im)
            image_list.append(image)
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
        image_list, image_for_crf, idx = samples[0]
        image = torch.cat([torch.unsqueeze(image, dim=0) for image in image_list], dim=0)
        return image, image_for_crf, idx

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

    def __init__(self, n_channels, is_weight_sum=False, clustering_num_list=None):
        super(BASNet, self).__init__()
        self.is_weight_sum = is_weight_sum

        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBlock(n_channels, 64, has_relu=True)  # 224 256 320
        self.encoder1 = resnet.layer1  # 224 256 320
        self.encoder2 = resnet.layer2  # 112 128 160
        self.encoder3 = resnet.layer3  # 56 64 80
        self.encoder4 = resnet.layer4  # 28 32 40

        # -------------MIC-------------
        self.clustering_num_list = list([128, 256, 512]) if clustering_num_list is None else clustering_num_list

        self.mic_l2norm = MICNormalize(2)
        self.mic_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # MIC 1
        self.mic_1_b1 = ResBlock(512, 512)  # 28 32 40
        self.mic_1_b2 = ResBlock(512, 512)
        self.mic_1_b3 = ResBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)

        # MIC 2
        self.mic_2_b1 = ResBlock(512, 512)  # 14 16 20
        self.mic_2_b2 = ResBlock(512, 512)
        self.mic_2_b3 = ResBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)

        # MIC 3
        self.mic_3_b1 = ResBlock(512, 512)  # 7 8 10
        self.mic_3_b2 = ResBlock(512, 512)
        self.mic_3_b3 = ResBlock(512, 512)
        self.mic_3_c1 = ConvBlock(512, self.clustering_num_list[2], has_relu=True)
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
        mic_2_feature = self.mic_2_b3(self.mic_2_b2(self.mic_2_b1(self.mic_pool(mic_1_feature))))  # 512 * 14 * 14
        mic_2 = self.mic_2_c1(mic_2_feature)  # 256 * 14 * 14
        smc_logits_2, smc_l2norm_2 = self.salient_map_clustering(mic_2)
        return_m2 = {"smc_logits": smc_logits_2, "smc_l2norm": smc_l2norm_2}

        # 3
        mic_3_feature = self.mic_3_b3(self.mic_3_b2(self.mic_3_b1(self.mic_pool(mic_2_feature))))  # 512 * 7 * 7
        mic_3 = self.mic_3_c1(mic_3_feature)  # 512 * 7 * 7
        smc_logits_3, smc_l2norm_3 = self.salient_map_clustering(mic_3)
        return_m3 = {"smc_logits": smc_logits_3, "smc_l2norm": smc_l2norm_3}

        return_m = {"m1": return_m1, "m2": return_m2, "m3": return_m3}

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

        cam_up_norm_123 = (cam_1_up_norm + cam_2_up_norm + cam_3_up_norm) / 3
        cam_up_norm_12 = (cam_1_up_norm + cam_2_up_norm) / 2
        cam_up_norm_13 = (cam_1_up_norm + cam_3_up_norm) / 2
        cam_up_norm_23 = (cam_2_up_norm + cam_3_up_norm) / 2

        return_l = {"cam_up_norm_C123": cam_up_norm_123, "cam_up_norm_C12": cam_up_norm_12,
                    "cam_up_norm_C13": cam_up_norm_13, "cam_up_norm_C23": cam_up_norm_23,
                    "cam_1_up_norm": cam_1_up_norm, "cam_2_up_norm": cam_2_up_norm, "cam_3_up_norm": cam_3_up_norm}

        return return_m, return_l

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, output_size=(1, 1)).view((mic.size()[0], -1))
        smc_l2norm = self.mic_l2norm(smc_logits)
        return smc_logits, smc_l2norm

    def cluster_activation_map(self, smc_logits, mic_feature):
        if self.is_weight_sum:
            smc_soft_max = torch.softmax(smc_logits, dim=-1)
            weight = torch.unsqueeze(torch.unsqueeze(smc_soft_max, dim=-1),
                                     dim=-1).repeat(1, 1, mic_feature.shape[-2], mic_feature.shape[-1])
            cam = torch.sum(weight * mic_feature, dim=1, keepdim=True)
        else:
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

    def __init__(self, batch_size=8, multi_num=16, size_train=224, size_vis=256, is_train=True,
                 clustering_num_1=128, clustering_num_2=256, clustering_num_3=512, is_weight_sum=False,
                 clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/cam", cam_dir="./cam/cam"):
        self.batch_size = batch_size
        self.is_train = is_train
        self.cam_dir = Tools.new_dir(cam_dir) if not self.is_train else None

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list, self.tra_cam_name_list = self.get_tra_img_label_name()
        self.tra_cam_name_list = None if self.is_train else self.tra_cam_name_list

        if self.is_train:
            self.dataset_sod = DatasetUSOD(img_name_list=self.tra_img_name_list, size_train=size_train)
            self.data_loader_sod = DataLoader(self.dataset_sod, self.batch_size, shuffle=True, num_workers=32)
        else:
            self.dataset_sod = DatasetUSODVIS(img_name_list=self.tra_img_name_list, multi_num=multi_num,
                                              cam_name_list=self.tra_cam_name_list, size_vis=size_vis)
            self.data_loader_sod = DataLoader(self.dataset_sod, 1, shuffle=False, num_workers=1,
                                              collate_fn=self.dataset_sod.collate_fn)
            pass
        self.data_num = len(self.dataset_sod)
        self.data_batch_num = len(self.data_loader_sod)

        # Model
        self.net = BASNet(3, is_weight_sum=is_weight_sum,
                          clustering_num_list=[clustering_num_1, clustering_num_2, clustering_num_3])
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
            cudnn.benchmark = True
            pass

        # MIC
        self.produce_class11 = MICProduceClass(self.data_num, out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class21 = MICProduceClass(self.data_num, out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class31 = MICProduceClass(self.data_num, out_dim=clustering_num_3, ratio=clustering_ratio_3)
        self.produce_class12 = MICProduceClass(self.data_num, out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class22 = MICProduceClass(self.data_num, out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class32 = MICProduceClass(self.data_num, out_dim=clustering_num_3, ratio=clustering_ratio_3)
        self.produce_class11.init()
        self.produce_class21.init()
        self.produce_class31.init()
        self.produce_class12.init()
        self.produce_class22.init()
        self.produce_class32.init()

        # Loss and optimizer
        self.bce_loss = nn.BCELoss().cuda()
        self.mic_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
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
        tra_cam_name_list = [os.path.join(self.cam_dir, '{}.bmp'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        Tools.print("train cams: {}".format(len(tra_cam_name_list)))
        return tra_img_name_list, tra_lbl_name_list, tra_cam_name_list

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_3_out, mic_labels_1, mic_labels_2, mic_labels_3):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)
        loss_mic_3 = self.mic_loss(mic_3_out, mic_labels_3)

        loss_all = loss_mic_1 + loss_mic_2 + loss_mic_3
        return loss_all, [loss_mic_1, loss_mic_2, loss_mic_3]

    def save_cam_info(self, cams, indexes, name=None):
        for cam, index in zip(cams, indexes):
            self.dataset_sod.save_cam(idx=int(index), cam=np.asarray(cam.squeeze().detach().cpu()), name=name)
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50):
        if not self.is_train:
            raise Exception("......................")

        if start_epoch >= 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            with torch.no_grad():
                for _idx, (inputs, _, indexes) in tqdm(enumerate(self.data_loader_sod), total=self.data_batch_num):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    return_m, _ = self.net(inputs)

                    self.produce_class11.cal_label(return_m["m1"]["smc_logits"], indexes)
                    self.produce_class21.cal_label(return_m["m2"]["smc_logits"], indexes)
                    self.produce_class31.cal_label(return_m["m3"]["smc_logits"], indexes)
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
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic_1, all_loss_mic_2, all_loss_mic_3, all_loss_sod = 0.0, 0.0, 0.0, 0.0, 0.0
            self.net.train()

            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()

            for i, (inputs, _, indexes) in tqdm(enumerate(self.data_loader_sod), total=self.data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                self.optimizer.zero_grad()

                return_m, _ = self.net(inputs)

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

                loss, loss_mic = self.all_loss_fusion(mic_target_1, mic_target_2, mic_target_3,
                                                      mic_labels_1, mic_labels_2, mic_labels_3)
                ######################################################################################################

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic_1 += loss_mic[0].item()
                all_loss_mic_2 += loss_mic[1].item()
                all_loss_mic_3 += loss_mic[2].item()
                if i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d},b:{:4d}/{:4d}] l:{:.2f}/{:.2f} mic:{:.2f}/{:.2f}-{:.2f}/{:.2f}-"
                                "{:.2f}/{:.2f}".format(epoch, epoch_num, i, self.data_batch_num, all_loss / (i + 1),
                                                       loss.item(), all_loss_mic_1 / (i + 1), loss_mic[0].item(),
                                                       all_loss_mic_2 / (i + 1), loss_mic[1].item(),
                                                       all_loss_mic_3 / (i + 1), loss_mic[2].item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic1:{:.3f} mic2:{:.3f} mic3:{:.3f} sod:{:.3f}".format(
                epoch, epoch_num, all_loss / self.data_batch_num,
                all_loss_mic_1 / self.data_batch_num, all_loss_mic_2 / self.data_batch_num,
                all_loss_mic_3 / self.data_batch_num, all_loss_sod / self.data_batch_num))

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
                    self.model_dir, "{}_train_{:.3f}.pth".format(epoch, all_loss / self.data_batch_num)))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()
                pass
            ###########################################################################

            pass

        # Final Save
        save_file_name = Tools.new_dir(os.path.join(
            self.model_dir, "{}_train_{:.3f}.pth".format(epoch_num, all_loss / self.data_batch_num)))
        torch.save(self.net.state_dict(), save_file_name)

        Tools.print()
        Tools.print("Save Model to {}".format(save_file_name))
        Tools.print()
        pass

    def vis(self):
        if self.is_train:
            raise Exception("....................................")

        self.net.eval()
        with torch.no_grad():
            for _idx, (inputs, image_for_crf, index) in tqdm(enumerate(self.data_loader_sod),
                                                             total=self.data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()

                _, return_l = self.net(inputs)

                self.dataset_sod.save_cam(idx=int(index), name="image", cam=np.asarray(image_for_crf))
                for key in return_l.keys():
                    value = return_l[key].detach().cpu()
                    value_mean = torch.mean(value, dim=0)

                    self.dataset_sod.save_cam(idx=int(index), name=key, cam=np.asarray(value_mean.squeeze()))

                    value_crf = CRFTool.crf_torch(torch.unsqueeze(image_for_crf, dim=0) * 255,
                                                  torch.unsqueeze(value_mean, dim=0), t=5)

                    self.dataset_sod.save_cam(idx=int(index), name="{}_crf".format(key),
                                              cam=np.asarray(value_crf.squeeze()))
                    pass

                pass
            pass
        pass

    pass


#######################################################################################################################
# 4 Main

"""
2020-07-11 10:16:24 [E:930/1000] loss:1.161 mic1:0.497 mic2:0.364 mic3:0.300 sod:0.000
2020-07-11 10:16:24 Train: [930] 1-1231/521
2020-07-11 10:16:24 Train: [930] 2-1077/464
2020-07-11 10:16:24 Train: [930] 3-973/419
2020-07-11 10:16:25 Save Model to ../BASNetTemp/saved_models/CAM_123_224/930_train_1.172.pth
"""

"""
1.多个增强融合
2.多个模型融合
3.输出正则归一化
4.判断那些样本参加训练
5.如何进行端到端训练
6.Weight Sum
"""

if __name__ == '__main__':
    _is_train = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" if _is_train else "0"
    _batch_size = 16 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    _is_weight_sum = True
    _size_train = 224
    _size_vis = 256
    _multi_num = 10
    _name_model = "CAM_123_{}_{}".format(_size_train, _size_vis)
    _name_cam = "CAM_123_{}_{}_AVG_{}{}".format(_size_train, _size_vis, _multi_num, "_S" if _is_weight_sum else "")

    bas_runner = BASRunner(batch_size=_batch_size, multi_num=_multi_num, is_weight_sum=_is_weight_sum,
                           size_train=_size_train, size_vis=_size_vis, is_train=_is_train,
                           clustering_num_1=128 * 4, clustering_num_2=128 * 4, clustering_num_3=128 * 4,
                           clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2,
                           data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           cam_dir="../BASNetTemp/cam/{}".format(_name_cam),
                           model_dir="../BASNetTemp/saved_models/{}".format(_name_model))

    if _is_train:  # 训练
        bas_runner.train(epoch_num=1000, start_epoch=0)
    else:  # 得到响应图
        bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256/930_train_1.172.pth")
        bas_runner.vis()
        pass

    pass
