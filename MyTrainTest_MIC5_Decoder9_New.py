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
from multiprocessing.pool import Pool
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

    def __call__(self, img, image_crf=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if image_crf is not None:
            image_crf = transforms.functional.resized_crop(image_crf, i, j, h, w, self.size, self.interpolation)
        return img, image_crf

    pass


class RandomResized(object):

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

    def __init__(self, img_name_list, cam_name_list=None, size=224):
        self.image_name_list = img_name_list
        self.cam_name_list = cam_name_list

        self.transform_train = Compose([RandomResizedCrop(size=size, scale=(0.3, 1.)),
                                        ColorJitter(0.4, 0.4, 0.4, 0.4), RandomGrayscale(p=0.2), RandomHorizontalFlip(),
                                        ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_test = Compose([RandomResized(size, size),
                                       ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = self.set_transform()

        self.image_size_list = [Image.open(image_name).size for image_name in self.image_name_list]
        pass

    def set_transform(self, is_train=True):
        self.transform = self.transform_train if is_train else self.transform_test
        return self.transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image, image_for_crf = self.transform(image, image)
        return image, image_for_crf, idx

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

        return_m = {"m1": return_m1, "m2": return_m2, "m3": return_m3}
        return return_m, return_l

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, output_size=(1, 1)).view((mic.size()[0], -1))
        smc_l2norm = self.mic_l2norm(smc_logits)
        return smc_logits, smc_l2norm

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

    def __init__(self, batch_size_train=8, clustering_num_1=128, clustering_num_2=256, clustering_num_3=512,
                 size=224, clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2, save_cam=False,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/cam", cam_dir="./cam/cam"):
        self.batch_size_train = batch_size_train

        self.save_cam = save_cam
        self.cam_dir = Tools.new_dir(cam_dir)

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list, self.tra_cam_name_list = self.get_tra_img_label_name()

        self.tra_cam_name_list = self.tra_cam_name_list if self.save_cam else None
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list,
                                        cam_name_list=self.tra_cam_name_list, size=size)
        self.dataset_usod.set_transform(is_train=True)
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
            self.dataset_usod.save_cam(idx=int(index), cam=np.asarray(cam.squeeze().detach().cpu()), name=name)
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50):

        if start_epoch >= 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            with torch.no_grad():
                for _idx, (inputs, _, indexes) in tqdm(enumerate(self.dataloader_usod),
                                                       total=len(self.dataloader_usod)):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    return_m, _ = self.net(inputs)

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
            self.net.train()

            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()

            for i, (inputs, _, indexes) in tqdm(enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
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
                    Tools.print("[E:{:4d}/{:4d}, b:{:4d}/{:4d}] l:{:.2f}/{:.2f} mic1:{:.2f}/{:.2f} mic2:{:.2f}/{:.2f} "
                                "mic3:{:.2f}/{:.2f}".format(epoch, epoch_num, i, len(self.dataloader_usod),
                                                            all_loss/(i+1), loss.item(),
                                                            all_loss_mic_1/(i+1), loss_mic[0].item(),
                                                            all_loss_mic_2/(i+1), loss_mic[1].item(),
                                                            all_loss_mic_3/(i+1), loss_mic[2].item()))
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

    def vis(self):
        self.net.eval()
        with torch.no_grad():
            self.dataset_usod.set_transform(is_train=False)
            self.dataloader_usod.dataset.set_transform(is_train=False)
            for _idx, (inputs, histories, image_for_crf, params, indexes) in tqdm(
                    enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()

                self.save_cam_info(image_for_crf, indexes=indexes, name="image")

                return_m, return_l = self.net(inputs)
                for key in return_l.keys():
                    value = return_l[key].detach()
                    self.save_cam_info(value, indexes=indexes, name=key)
                    value_crf = CRFTool.crf_torch(image_for_crf, value, t=5)
                    self.save_cam_info(value_crf, indexes=indexes, name="{}_crf".format(key))
                    pass

                pass
            pass
        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    _size = 256
    _name = "CAM_123_{}".format(_size)
    bas_runner = BASRunner(batch_size_train=16 * 2, data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           clustering_num_1=128 * 4, clustering_num_2=128 * 4, clustering_num_3=128 * 4,
                           size=_size, cam_dir="../BASNetTemp/cam/{}".format(_name), save_cam=False,
                           model_dir="../BASNetTemp/saved_models/{}".format(_name))
    bas_runner.train(epoch_num=500, start_epoch=0)
    pass
