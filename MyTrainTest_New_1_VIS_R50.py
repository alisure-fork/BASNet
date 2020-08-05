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
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from pydensecrf.utils import unary_from_softmax, unary_from_labels


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

    @staticmethod
    def crf_label(image, annotation, t=5, n_label=2):
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        annotation = np.squeeze(np.array(annotation))

        a, b = (0.8, 0.1)
        if np.max(annotation) > 1:
            a, b = a * 255, b * 255
            pass
        label_extend = np.zeros_like(annotation)
        label_extend[annotation > a] = 2
        label_extend[annotation < b] = 1

        _, label = np.unique(label_extend, return_inverse=True)

        d = dcrf.DenseCRF2D(w, h, n_label)
        u = unary_from_labels(label, n_label, gt_prob=0.7, zero_unsure=True)
        u = np.ascontiguousarray(u)
        d.setUnaryEnergy(u)
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image, compat=10)
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


class EvalCAM(object):

    def __init__(self, lab_name_list, lab2_name_list, size_eval=None, th_num=100):
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

        label = np.asarray(label) / 255
        label2 = np.asarray(label2) / 255

        mae = self.eval_mae(label, label2)
        prec, recall = self.eval_pr(label, label2, th_num=self.th_num)
        return mae, prec, recall

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

    def __init__(self, is_train=False, clustering_num_list=None, is_supervised_pre_train=False,
                 is_unsupervised_pre_train=True, unsupervised_pre_train_path="./pre_model/MoCov2.pth"):
        super(BASNet, self).__init__()
        self.is_train = is_train

        # -------------Encoder--------------
        backbone = resnet.__dict__["resnet50"](pretrained=is_supervised_pre_train,
                                               replace_stride_with_dilation=[False, True, True])
        if is_unsupervised_pre_train:
            backbone = self.load_unsupervised_pre_train(backbone, unsupervised_pre_train_path)
        return_layers = {'relu': 'e0', 'layer1': 'e1', 'layer2': 'e2', 'layer3': 'e3', 'layer4': 'e4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.change_channel = ConvBlock(2048, 512)

        # -------------MIC-------------
        self.clustering_num_list = list([256, 512]) if clustering_num_list is None else clustering_num_list

        self.mic_l2norm = MICNormalize(2)
        self.mic_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # MIC 1
        self.mic_1_b1 = resnet.BasicBlock(512, 512)  # 28 32 40
        self.mic_1_b2 = resnet.BasicBlock(512, 512)
        self.mic_1_b3 = resnet.BasicBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)

        # MIC 2
        self.mic_2_b1 = resnet.BasicBlock(512, 512)  # 14 16 20
        self.mic_2_b2 = resnet.BasicBlock(512, 512)
        self.mic_2_b3 = resnet.BasicBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)
        pass

    def forward(self, x):
        x_for_up = x

        # -------------Encoder-------------
        feature = self.backbone(x)  # (64, 160), (256, 80), (512, 40), (1024, 40), (2048, 40)
        e4 = self.change_channel(feature["e4"])  # (512, 40)
        # -------------MIC-------------
        # 1
        mic_1_feature = self.mic_1_b3(self.mic_1_b2(self.mic_1_b1(e4)))  # (512, 40)
        mic_1 = self.mic_1_c1(mic_1_feature)  # (256, 40)
        smc_logits_1, smc_l2norm_1 = self.salient_map_clustering(mic_1)
        return_m1 = {"smc_logits": smc_logits_1, "smc_l2norm": smc_l2norm_1}

        # 2
        mic_2_feature = self.mic_2_b3(self.mic_2_b2(self.mic_2_b1(self.mic_pool(mic_1_feature))))  # (512, 20)
        mic_2 = self.mic_2_c1(mic_2_feature)  # (512, 20)
        smc_logits_2, smc_l2norm_2 = self.salient_map_clustering(mic_2)
        return_m2 = {"smc_logits": smc_logits_2, "smc_l2norm": smc_l2norm_2}

        return_m = {"m1": return_m1, "m2": return_m2}

        # -------------Label-------------
        return_l = None
        if not self.is_train:
            cam_1 = self.cluster_activation_map(smc_logits_1, mic_1)  # 簇激活图：Cluster Activation Map
            cam_1_up = self._up_to_target(cam_1, x_for_up)
            cam_1_up_norm = self._feature_norm(cam_1_up)

            cam_2 = self.cluster_activation_map(smc_logits_2, mic_2)  # 簇激活图：Cluster Activation Map
            cam_2_up = self._up_to_target(cam_2, cam_1_up)
            cam_2_up_norm = self._feature_norm(cam_2_up)

            cam_up_norm_12 = (cam_1_up_norm + cam_2_up_norm) / 2

            return_l = {"cam_up_norm_C1": cam_1_up_norm, "cam_up_norm_C2": cam_2_up_norm,
                        "cam_up_norm_C12": cam_up_norm_12}
            pass

        return return_m, return_l

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, output_size=(1, 1)).view((mic.size()[0], -1))
        smc_l2norm = self.mic_l2norm(smc_logits)
        return smc_logits, smc_l2norm

    @staticmethod
    def cluster_activation_map(smc_logits, mic_feature):
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        cam = torch.cat([mic_feature[i:i + 1, top_k_index[i], :, :] for i in range(mic_feature.size()[0])])
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

    @staticmethod
    def load_unsupervised_pre_train(model, pre_train_path, change_key="module.encoder."):
        pre_train = torch.load(pre_train_path)["model"]
        checkpoint = {key.replace(change_key, ""): pre_train[key] for key in pre_train.keys() if change_key in key}
        model.load_state_dict(checkpoint, strict=False)
        return model

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, batch_size=8, multi_num=16, size_train=224, size_vis=256, is_train=True,
                 clustering_num_1=256, clustering_num_2=512, clustering_ratio_1=1, clustering_ratio_2=2,
                 tra_img_name_list=None, tra_lbl_name_list=None, tra_data_name_list=None,
                 model_dir="./saved_models/cam", cam_dir="./cam/cam"):
        self.batch_size = batch_size
        self.is_train = is_train
        self.cam_dir = Tools.new_dir(cam_dir) if not self.is_train else None

        # Dataset
        self.model_dir = model_dir
        self.tra_img_name_list = tra_img_name_list
        self.tra_lbl_name_list = tra_lbl_name_list
        self.tra_data_name_list = tra_data_name_list
        self.tra_cam_name_list = self.get_tra_cam_name() if not self.is_train else None

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
        self.net = BASNet(is_train=self.is_train, clustering_num_list=[clustering_num_1, clustering_num_2])
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
            cudnn.benchmark = True
            pass

        # MIC
        self.produce_class11 = MICProduceClass(self.data_num, out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class21 = MICProduceClass(self.data_num, out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class12 = MICProduceClass(self.data_num, out_dim=clustering_num_1, ratio=clustering_ratio_1)
        self.produce_class22 = MICProduceClass(self.data_num, out_dim=clustering_num_2, ratio=clustering_ratio_2)
        self.produce_class11.init()
        self.produce_class21.init()
        self.produce_class12.init()
        self.produce_class22.init()

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

    def get_tra_cam_name(self):
        tra_cam_name_list = [os.path.join(self.cam_dir, '{}_{}.bmp'.format(dataset_name, os.path.splitext(
            os.path.basename(img_path))[0])) for img_path, dataset_name in zip(
            self.tra_img_name_list, self.tra_data_name_list)]
        Tools.print("train images: {}".format(len(self.tra_img_name_list)))
        Tools.print("train labels: {}".format(len(self.tra_lbl_name_list)))
        Tools.print("train cams: {}".format(len(tra_cam_name_list)))
        return tra_cam_name_list

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_labels_1, mic_labels_2):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)

        loss_all = loss_mic_1 + loss_mic_2
        return loss_all, [loss_mic_1, loss_mic_2]

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
            with torch.no_grad():
                for _idx, (inputs, _, indexes) in tqdm(enumerate(self.data_loader_sod), total=self.data_batch_num):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    return_m, _ = self.net(inputs)

                    self.produce_class11.cal_label(return_m["m1"]["smc_logits"], indexes)
                    self.produce_class21.cal_label(return_m["m2"]["smc_logits"], indexes)
                    pass
                pass
            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            Tools.print("Update: [{}] 1-{}/{}".format(start_epoch, self.produce_class11.count,
                                                      self.produce_class11.count_2))
            Tools.print("Update: [{}] 2-{}/{}".format(start_epoch, self.produce_class21.count,
                                                      self.produce_class21.count_2))
            pass

        all_loss = 0
        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic_1, all_loss_mic_2 = 0.0, 0.0, 0.0
            self.net.train()

            self.produce_class11.reset()
            self.produce_class21.reset()

            for i, (inputs, _, indexes) in tqdm(enumerate(self.data_loader_sod), total=self.data_batch_num):
                inputs = inputs.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                self.optimizer.zero_grad()

                return_m, _ = self.net(inputs)

                ######################################################################################################
                # MIC
                self.produce_class11.cal_label(return_m["m1"]["smc_logits"], indexes)
                self.produce_class21.cal_label(return_m["m2"]["smc_logits"], indexes)
                mic_target_1 = return_m["m1"]["smc_logits"]
                mic_target_2 = return_m["m2"]["smc_logits"]
                mic_labels_1 = self.produce_class12.get_label(indexes).cuda()
                mic_labels_2 = self.produce_class22.get_label(indexes).cuda()

                loss, loss_mic = self.all_loss_fusion(mic_target_1, mic_target_2, mic_labels_1, mic_labels_2)
                ######################################################################################################

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic_1 += loss_mic[0].item()
                all_loss_mic_2 += loss_mic[1].item()
                if i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d},b:{:4d}/{:4d}] l:{:.2f}/{:.2f} mic:{:.2f}/{:.2f}-{:.2f}/{:.2f}".format(
                        epoch, epoch_num, i, self.data_batch_num, all_loss / (i + 1),
                        loss.item(), all_loss_mic_1 / (i + 1), loss_mic[0].item(),
                        all_loss_mic_2 / (i + 1), loss_mic[1].item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic1:{:.3f} mic2:{:.3f}".format(
                epoch, epoch_num, all_loss / self.data_batch_num,
                all_loss_mic_1 / self.data_batch_num, all_loss_mic_2 / self.data_batch_num))

            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, self.produce_class11.count, self.produce_class11.count_2))
            Tools.print("Train: [{}] 2-{}/{}".format(epoch, self.produce_class21.count, self.produce_class21.count_2))

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

    @staticmethod
    def eval_vis(img_name_list, lbl_name_list, data_name_list,
                 vis_label_dir, vis_label_name, size_eval=None, th_num=25, beta_2=0.3, has_data_name=True):
        if has_data_name:
            lbl2_name_list = [os.path.join(vis_label_dir, '{}_{}_{}.bmp'.format(
                data_name, os.path.splitext(os.path.basename(img_path))[0], vis_label_name)
                                           ) for img_path, data_name in zip(img_name_list, data_name_list)]
        else:
            lbl2_name_list = [os.path.join(vis_label_dir, '{}_{}.bmp'.format(
                 os.path.splitext(os.path.basename(img_path))[0], vis_label_name)
                                           ) for img_path, data_name in zip(img_name_list, data_name_list)]
            pass

        eval_cam = EvalCAM(lbl_name_list, lbl2_name_list, size_eval=size_eval, th_num=th_num)

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

    pass


#######################################################################################################################
# 4 Main


"""
2020-07-11 10:16:24 [E:930/1000] loss:1.161 mic1:0.497 mic2:0.364 mic3:0.300 sod:0.000
2020-07-11 10:16:24 Train: [930] 1-1231/521
2020-07-11 10:16:24 Train: [930] 2-1077/464
2020-07-11 10:16:24 Train: [930] 3-973/419
2020-07-11 10:16:25 Save Model to ../BASNetTemp/saved_models/CAM_123_224/930_train_1.172.pth

2020-07-14 13:34:05 ../BASNetTemp/cam/CAM_123_224_256_AVG_1 cam_up_norm_C123_crf
2020-07-14 13:40:06 Train avg mae=0.18007975575910706 score=0.6450108439345925
2020-07-14 13:41:38 ../BASNetTemp/cam/CAM_123_224_256_AVG_9 cam_up_norm_C123_crf
2020-07-14 13:47:55 Train avg mae=0.17801150496007748 score=0.6452717806037506
2020-07-14 13:41:42 ../BASNetTemp/cam/CAM_123_224_256_AVG_30 cam_up_norm_C123_crf
2020-07-14 13:47:43 Train avg mae=0.17759568890576982 score=0.6455211460931307

2020-07-26 08:17:53 [E:999/1000] loss:1.072 mic1:0.444 mic2:0.332 mic3:0.296 sod:0.000
2020-07-26 08:17:53 Train: [999] 1-7135/2175
2020-07-26 08:17:53 Train: [999] 2-6100/2297
2020-07-26 08:17:53 Train: [999] 3-5929/2481
2020-07-26 08:17:54 Save Model to ../BASNetTemp/saved_models/CAM_123_224_256_DTrue/1000_train_1.072.pth
2020-07-27 00:08:22 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DTrue cam_up_norm_C23_crf
2020-07-27 00:11:17 Train avg mae=0.15425343497672997 score=0.29836294297909255

2020-07-28 18:17:51 [E:999/1000] loss:1.154 mic1:0.474 mic2:0.368 mic3:0.312 sod:0.000
2020-07-28 18:17:51 Train: [999] 1-1750/681
2020-07-28 18:17:51 Train: [999] 2-1534/646
2020-07-28 18:17:51 Train: [999] 3-1420/657
2020-07-28 18:17:51 Save Model to ../BASNetTemp/saved_models/CAM_123_224_256_DFalse/1000_train_1.154.pth


2020-07-28 20:30:46 ../BASNetTemp/cam/CAM_123_224_256_AVG_1 cam_up_norm_C12_crf
2020-07-28 20:41:49 Train avg mae=0.18613760967550433 score=0.5685004443218294
2020-07-28 20:30:13 ../BASNetTemp/cam/CAM_123_224_256_AVG_1 cam_up_norm_C23_crf
2020-07-28 20:39:44 Train avg mae=0.20597653803163998 score=0.686193559271971
2020-07-28 20:30:31 ../BASNetTemp/cam/CAM_123_224_256_AVG_1 cam_up_norm_C123_crf
2020-07-28 20:37:40 Train avg mae=0.18007975575910684 score=0.6447458188033726

2020-07-28 20:31:04 ../BASNetTemp/cam/CAM_123_224_256_AVG_9 cam_up_norm_C12_crf
2020-07-28 20:41:19 Train avg mae=0.18508573617764634 score=0.5669514546957682
2020-07-28 20:31:09 ../BASNetTemp/cam/CAM_123_224_256_AVG_9 cam_up_norm_C23_crf
2020-07-28 20:41:10 Train avg mae=0.2027546845221534 score=0.6884822742758568
2020-07-28 20:31:00 ../BASNetTemp/cam/CAM_123_224_256_AVG_9 cam_up_norm_C123_crf
2020-07-28 20:39:38 Train avg mae=0.17801150496007723 score=0.644987676666338

2020-07-28 20:24:01 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C12_crf
2020-07-28 20:26:08 Train avg mae=0.18763395561814353 score=0.493304236445042
2020-07-28 20:21:56 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C23_crf
2020-07-28 20:28:40 Train avg mae=0.18351527000009912 score=0.622681934726851
2020-07-28 20:23:35 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C123_crf
2020-07-28 20:26:06 Train avg mae=0.17520064073057032 score=0.5644434479389823

2020-07-28 20:24:22 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C12_crf
2020-07-28 20:25:22 Train avg mae=0.15861528231772198 score=0.3611402350558168
2020-07-28 20:24:30 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C23_crf
2020-07-28 20:25:39 Train avg mae=0.19076016670506263 score=0.4730370864863008
2020-07-28 20:24:27 ../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse cam_up_norm_C123_crf
2020-07-28 20:25:31 Train avg mae=0.16173122906402748 score=0.41750517170681284
"""


"""
1.多个增强融合
2.多个模型融合
3.输出正则归一化
4.判断那些样本参加训练
5.如何进行端到端训练
6.Weight Sum, 修改成全连接!!!!
"""


if __name__ == '__main__':
    _is_all_data = False
    _is_train = True
    _is_eval = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" if _is_train else "0"
    _batch_size = 4 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    _size_train = 224
    _size_vis = 256
    _multi_num = 5
    _name_model = "R50_CAM_123_{}_{}_D{}".format(_size_train, _size_vis, _is_all_data)
    _name_cam = "R50_CAM_123_{}_{}_A{}_D{}".format(_size_train, _size_vis, _multi_num, _is_all_data)

    sod_data = SODData(data_root_path="/media/ubuntu/4T/ALISURE/Data/SOD")
    all_image, all_mask, all_dataset_name = sod_data.get_all_train_and_mask() if _is_all_data else sod_data.duts_tr()

    if _is_eval:
        # _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_1"
        # _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_9"
        _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_AVG_30"
        _tra_label_name = 'cam_up_norm_C12_crf'

        # _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_A1_SFalse_DTrue"
        # _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DTrue"
        # _tra_label_name = 'cam_up_norm_C23_crf'

        # _tra_label_dir = "../BASNetTemp/cam/CAM_123_224_256_A5_SFalse_DFalse"
        # _tra_label_name = 'cam_up_norm_C23_crf'

        Tools.print("{} {}".format(_tra_label_dir, _tra_label_name))
        _image_list, _mask_list, _dataset_list = sod_data.duts_tr()
        # _image_list, _mask_list, _dataset_list = sod_data.duts_te()
        _has_data_name = False
        BASRunner.eval_vis(_image_list, _mask_list, _dataset_list,
                           _tra_label_dir, _tra_label_name, size_eval=_size_vis, has_data_name=_has_data_name)
    else:
        bas_runner = BASRunner(batch_size=_batch_size, multi_num=_multi_num,
                               size_train=_size_train, size_vis=_size_vis, is_train=_is_train,
                               clustering_num_1=128 * 4, clustering_num_2=128 * 4,
                               clustering_ratio_1=1, clustering_ratio_2=2,
                               tra_img_name_list=all_image, tra_lbl_name_list=all_mask,
                               tra_data_name_list=all_dataset_name, cam_dir="../BASNetTemp/cam/{}".format(_name_cam),
                               model_dir="../BASNetTemp/saved_models/{}".format(_name_model))
        if _is_train:  # 训练
            bas_runner.train(epoch_num=500, start_epoch=0)
        else:  # 得到响应图
            # bas_runner.load_model(model_file_name="../BASNetTemp/saved_models/CAM_123_224_256/930_train_1.172.pth")
            # bas_runner.load_model(
            #     model_file_name="../BASNetTemp/saved_models/CAM_123_224_256_DTrue/1000_train_1.072.pth")
            bas_runner.load_model(
                model_file_name="../BASNetTemp/saved_models/CAM_123_224_256_DFalse/1000_train_1.154.pth")
            bas_runner.vis()
            pass
        pass

    pass
