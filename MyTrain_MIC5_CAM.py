import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from alisuretool.Tools import Tools
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
        assert img.shape[0] == annotation.shape[0]
        img = torch.nn.functional.interpolate(img, size=(annotation.shape[2], annotation.shape[3]))

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

    def __init__(self, img_name_list, his_name_list=None, size=224):
        self.image_name_list = img_name_list
        self.history_name_list = his_name_list
        self.has_history = self.history_name_list is not None
        self.transform_test = Compose([FixedResized(size, size), ToTensor(),
                                       Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.image_size_list = [Image.open(image_name).size for image_name in self.image_name_list]
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image, image_for_crf = self.transform_test(image, image)
        return image, image_for_crf, idx

    def save_history(self, idx, his, name=None):
        if self.history_name_list is not None:
            h_path = self.history_name_list[idx]
            if name is not None:
                h_path = "{}_{}{}".format(os.path.splitext(h_path)[0], name, os.path.splitext(h_path)[1])
            h_path = Tools.new_dir(h_path)

            his = np.transpose(his, (1, 2, 0)) if his.shape[0] == 1 or his.shape[0] == 3 else his
            im = Image.fromarray(np.asarray(his * 255, dtype=np.uint8)).resize(self.image_size_list[idx])
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

    def __init__(self, n_channels, clustering_num_list=None):
        super(BASNet, self).__init__()
        resnet = models.resnet18(pretrained=False)

        # -------------Encoder--------------
        self.encoder0 = ConvBlock(n_channels, 64, has_relu=True)  # 224 256 320
        self.encoder1 = resnet.layer1  # 224 256 320
        self.encoder2 = resnet.layer2  # 112 128 160
        self.encoder3 = resnet.layer3  # 56 64 80
        self.encoder4 = resnet.layer4  # 28 32 40

        # -------------MIC-------------
        self.clustering_num_list = list([128, 256, 512]) if clustering_num_list is None else clustering_num_list

        # MIC 1
        self.mic_1_b1 = ResBlock(512, 512)  # 28 32 40
        self.mic_1_b2 = ResBlock(512, 512)
        self.mic_1_b3 = ResBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)
        self.mic_1_l2norm = MICNormalize(2)

        # MIC 2
        self.mic_2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_2_b1 = ResBlock(512, 512)  # 14 16 20
        self.mic_2_b2 = ResBlock(512, 512)
        self.mic_2_b3 = ResBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)
        self.mic_2_l2norm = MICNormalize(2)

        # MIC 3
        self.mic_3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_3_b1 = ResBlock(512, 512)  # 7 8 10
        self.mic_3_b2 = ResBlock(512, 512)
        self.mic_3_b3 = ResBlock(512, 512)
        self.mic_3_c1 = ConvBlock(512, self.clustering_num_list[2], has_relu=True)
        self.mic_3_l2norm = MICNormalize(2)
        pass

    def forward(self, x):
        # -------------Encoder-------------
        e0 = self.encoder0(x)  # 64 * 224 * 224
        e1 = self.encoder1(e0)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28

        # -------------Decoder-------------
        # 1
        mic_f_1 = self.mic_1_b1(e4)
        mic_f_1 = self.mic_1_b2(mic_f_1)
        mic_f_1 = self.mic_1_b3(mic_f_1)
        mic_1 = self.mic_1_c1(mic_f_1)  # 512 * 28 * 28
        smc_logits_1, smc_l2norm_1 = self.salient_map_clustering(mic_1)

        # 2
        mic_f_2 = self.mic_2_pool(mic_f_1)  # 512 * 14 * 14
        mic_f_2 = self.mic_2_b1(mic_f_2)
        mic_f_2 = self.mic_2_b2(mic_f_2)
        mic_f_2 = self.mic_2_b3(mic_f_2)
        mic_2 = self.mic_2_c1(mic_f_2)  # 512 * 14 * 14
        smc_logits_2, smc_l2norm_2 = self.salient_map_clustering(mic_2)

        # 3
        mic_f_3 = self.mic_3_pool(mic_f_2)  # 512 * 7 * 7
        mic_f_3 = self.mic_3_b1(mic_f_3)
        mic_f_3 = self.mic_3_b2(mic_f_3)
        mic_f_3 = self.mic_3_b3(mic_f_3)
        mic_3 = self.mic_3_c1(mic_f_3)  # 512 * 7 * 7
        smc_logits_3, smc_l2norm_3 = self.salient_map_clustering(mic_3)

        return_mic = {
            "smc_logits_1": smc_logits_1, "smc_l2norm_1": smc_l2norm_1,
            "smc_logits_2": smc_logits_2, "smc_l2norm_2": smc_l2norm_2,
            "smc_logits_3": smc_logits_3, "smc_l2norm_3": smc_l2norm_3,
        }

        # -------------Label-------------
        cam_1 = self.cluster_activation_map(smc_logits_1, mic_1)  # 簇激活图：Cluster Activation Map
        cam_1_norm = self._feature_norm(cam_1)
        cam_1_up = self._up_to_target(cam_1, x)
        cam_1_up_norm = self._feature_norm(cam_1_up)

        cam_2 = self.cluster_activation_map(smc_logits_2, mic_2)  # 簇激活图：Cluster Activation Map
        cam_2_norm = self._feature_norm(cam_2)
        cam_2_up = self._up_to_target(cam_2, cam_1_up)
        cam_2_up_norm = self._feature_norm(cam_2_up)

        cam_3 = self.cluster_activation_map(smc_logits_3, mic_3)  # 簇激活图：Cluster Activation Map
        cam_3_norm = self._feature_norm(cam_3)
        cam_3_up = self._up_to_target(cam_3, cam_1_up)
        cam_3_up_norm = self._feature_norm(cam_3_up)

        cam_up_norm = (cam_1_up_norm + cam_2_up_norm + cam_3_up_norm) / 3

        # return_cam = {"cam_1_up_norm": cam_1_up_norm, "cam_1_norm": cam_1_norm,
        #               "cam_2_up_norm": cam_2_up_norm, "cam_2_norm": cam_2_norm,
        #               "cam_3_up_norm": cam_3_up_norm, "cam_3_norm": cam_3_norm}

        return_cam = {"cam_up_norm": cam_up_norm}

        return return_mic, return_cam

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, 1).view((mic.size()[0], -1))  # 512
        smc_l2norm = self.mic_1_l2norm(smc_logits)
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

    def __init__(self, batch_size_train=8, clustering_num_list=[128, 256, 512], size=224,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', history_dir="./history/my_train_mic5_large_history1"):
        self.batch_size_train = batch_size_train

        # History
        self.history_dir = Tools.new_dir(history_dir)

        # Dataset
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list, self.tra_his_name_list = self.get_tra_img_label_name()
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list,
                                        his_name_list=self.tra_his_name_list, size=size)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=False, num_workers=8)

        # Model
        self.net = BASNet(3, clustering_num_list=clustering_num_list).cuda()
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

    def vis(self):
        self.net.eval()
        with torch.no_grad():
            for _idx, (inputs, image_for_crf, indexes) in tqdm(enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()

                self.save_history_info(image_for_crf, indexes=indexes, name="image")

                return_mic, return_cam = self.net(inputs)
                for key in return_cam.keys():
                    value = return_cam[key].detach()
                    self.save_history_info(value, indexes=indexes, name=key)
                    value_crf = CRFTool.crf_torch(image_for_crf, value, t=5)
                    self.save_history_info(value_crf, indexes=indexes, name="{}_crf".format(key))
                    pass

                pass
            pass
        pass

    def save_history_info(self, histories, indexes, name=None):
        for history, index in zip(histories, indexes):
            self.dataset_usod.save_history(idx=int(index), name=name,
                                           his=np.asarray(history.squeeze().detach().cpu()))
        pass
    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    _size = 224
    _name = "MyTrain_MIC5_{}".format(_size)
    bas_runner = BASRunner(batch_size_train=1, data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           clustering_num_list=[128 * 4, 128 * 4, 128 * 4], size=_size,
                           history_dir="../BASNetTemp/Label/{}".format(_name))
    bas_runner.load_model('../BASNetTemp/saved_models/my_train_mic5_large/500_train_0.880.pth')
    bas_runner.vis()
    pass
