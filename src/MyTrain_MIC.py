import os
import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from skimage import io, transform
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader, Dataset


#######################################################################################################################
# 1 Data


class RescaleT(object):

    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), 0, mode='constant', preserve_range=True)
        return {'image': img, 'label': lbl}

    pass


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        return {'image': image, 'label': label}

    pass


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        label = label if np.max(label) < 1e-6 else (label / np.max(label))

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            pass

        tmpLbl[:, :, 0] = label[:, :, 0]
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),  'label': torch.from_numpy(tmpLbl)}

    pass


class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, lbl_name_list=None, transform=None):
        # self.image_name_list = img_name_list[:20]
        # self.label_name_list = lbl_name_list[:20]
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        label_3 = io.imread(self.label_name_list[idx]) if self.label_name_list else np.zeros(image.shape)

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3
            pass

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]
            pass

        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']
            pass

        return image, label, idx

    pass


#######################################################################################################################
# 2 Model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class BASNet(nn.Module):

    def __init__(self, n_channels, pretrained=True):
        super(BASNet, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)

        # -------------Encoder--------------
        self.encoder0_conv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.encoder0_bn = nn.BatchNorm2d(64)
        self.encoder0_relu = nn.ReLU(inplace=True)
        self.encoder1 = resnet.layer1  # 224
        self.encoder2 = resnet.layer2  # 112
        self.encoder3 = resnet.layer3  # 56
        self.encoder4 = resnet.layer4  # 28

        # -------------Bridge--------------
        self.bridge_conv_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 28
        self.bridge_bn_1 = nn.BatchNorm2d(512)
        self.bridge_relu_1 = nn.ReLU(inplace=True)
        self.bridge_conv_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bridge_bn_2 = nn.BatchNorm2d(512)
        self.bridge_relu_2 = nn.ReLU(inplace=True)
        self.bridge_conv_3 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bridge_bn_3 = nn.BatchNorm2d(512)
        self.bridge_relu_3 = nn.ReLU(inplace=True)

        # -------------Decoder-------------
        # MIC 1
        self.mic_l2norm = MICNormalize(2)
        self.out_conv_bridge = nn.Conv2d(512, 1, 3, padding=1)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        pass

    def forward(self, x):
        ex = self.encoder0_relu(self.encoder0_bn(self.encoder0_conv(x)))  # 64 * 224 * 224

        # -------------Encoder-------------
        e1 = self.encoder1(ex)  # 64 * 224 * 224
        e2 = self.encoder2(e1)  # 128 * 112 * 112
        e3 = self.encoder3(e2)  # 256 * 56 * 56
        e4 = self.encoder4(e3)  # 512 * 28 * 28

        # -------------Bridge-------------
        bridge_1 = self.bridge_relu_1(self.bridge_bn_1(self.bridge_conv_1(e4)))
        bridge_2 = self.bridge_relu_2(self.bridge_bn_2(self.bridge_conv_2(bridge_1)))
        bridge_3 = self.bridge_relu_3(self.bridge_bn_3(self.bridge_conv_3(bridge_2)))  # 512 * 28 * 28

        # -------------Decoder-------------
        ob = self.out_conv_bridge(bridge_3)  # 1 * 28 * 28
        ob_up = self.upscore4(ob)  # 1 * 224 * 224
        so = torch.sigmoid(ob)  # 1 * 28 * 28  # 小输出
        so_up = torch.sigmoid(ob_up)  # 1 * 224 * 224  # 大输出

        # More
        # 显著图聚类：Salient Map Clustering
        smc_logits, smc_l2norm, smc_sigmoid = self.salient_map_clustering(bridge_3, so)
        cam = self.cluster_activation_map(smc_logits, bridge_3)  # 簇激活图：Cluster Activation Map
        sme = self.salient_map_divide(cam)  # 显著图划分：Salient Map Divide

        return so, so_up, cam, sme, smc_logits, smc_l2norm, bridge_3

    def salient_map_clustering(self, feature_for_smc, mask_b):
        # m1
        # smc = feature_for_smc * mask_b  # 512 * 28 * 28
        smc = feature_for_smc  # 512 * 28 * 28

        # m2
        gaussian_mask = self._mask_gaussian([smc.size()[2], smc.size()[3]], sigma=smc.size()[2] * smc.size()[3] // 2)
        smc_gaussian = smc * torch.tensor(gaussian_mask).cuda()
        # smc_gaussian = smc

        smc_logits = F.adaptive_avg_pool2d(smc_gaussian, 1).view((smc_gaussian.size()[0], -1))  # 512

        smc_l2norm = self.mic_l2norm(smc_logits)
        smc_sigmoid = torch.sigmoid(smc_logits)
        return smc_logits, smc_l2norm, smc_sigmoid

    @staticmethod
    def cluster_activation_map(smc_logits, feature_for_cam, k=5):
        # top_k_value, top_k_index = torch.topk(smc_logits, k, 1)
        # cam = torch.cat([feature_for_cam[i:i+1, top_k_index[i], :, :].mean(1, keepdim=True)
        #                  for i in range(feature_for_cam.size()[0])])
        top_k_value, top_k_index = torch.topk(smc_logits, 1, 1)
        cam = torch.cat([feature_for_cam[i:i+1, top_k_index[i], :, :] for i in range(feature_for_cam.size()[0])])
        return cam

    def salient_map_divide(self, cam, obj_th=0.7, bg_th=0.1):
        for_cam_norm = self._feature_norm(cam)  # 1 * 28 * 28

        mask = torch.zeros(tuple(for_cam_norm.size())).fill_(255)
        mask = mask.cuda() if torch.cuda.is_available() else mask
        mask[for_cam_norm > obj_th] = 1.0
        mask[for_cam_norm < bg_th] = 0.0

        return mask

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def _mask_gaussian(image_size, where=None, sigma=20):

        x = np.arange(0, image_size[1], 1, float)
        y = np.arange(0, image_size[0], 1, float)
        y = y[:, np.newaxis]

        if where:
            x0, y0 = where[1], where[0]
        else:
            x0, y0 = image_size[1] // 2, image_size[0] // 2
            pass

        # 生成高斯掩码
        mask = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma).astype(np.float32)
        return mask

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


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, epoch_num=100000, batch_size_train=8, clustering_out_dim=512,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic_only"):
        self.epoch_num = epoch_num
        self.batch_size_train = batch_size_train

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, self.tra_lbl_name_list = self.get_tra_img_label_name()
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list, lbl_name_list=self.tra_lbl_name_list,
                                        transform=transforms.Compose([RescaleT(256), RandomCrop(224), ToTensor()]))
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=1)

        # Model
        self.net = BASNet(3, pretrained=True)
        self.net = self.net.cuda() if torch.cuda.is_available() else self.net

        # MIC
        self.produce_class = MICProduceClass(n_sample=len(self.dataset_usod), out_dim=clustering_out_dim, ratio=3)

        # Loss and Optim
        self.bce_loss = nn.BCELoss()
        self.mic_loss = nn.CrossEntropyLoss()
        self.bce_loss = self.bce_loss.cuda() if torch.cuda.is_available() else self.bce_loss
        self.mic_loss = self.mic_loss.cuda() if torch.cuda.is_available() else self.mic_loss

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        pass

    def load_model(self, model_file_name):
        self.net.load_state_dict(torch.load(model_file_name))
        Tools.print("restore from {}".format(model_file_name))
        pass

    def get_tra_img_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, self.tra_image_dir, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_dir, self.tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list

    def all_loss_fusion(self, bce_out, bce_label, mic_out, mic_label):
        positions = bce_label.view(-1, 1) < 255.0
        loss_bce = self.bce_loss(bce_out.view(-1, 1)[positions], bce_label.view(-1, 1)[positions])

        loss_mic = self.mic_loss(mic_out, mic_label)

        # loss_all = loss_bce + loss_mic
        loss_all = loss_mic

        return loss_all, loss_bce, loss_mic

    def train(self, save_epoch_freq=5, print_ite_num=100, update_epoch_freq=1):

        for epoch in range(0, self.epoch_num):

            ###########################################################################
            # 0 更新标签
            if epoch % update_epoch_freq == 0:
                Tools.print()
                Tools.print("Update label {} .......".format(epoch))
                self.net.eval()

                self.produce_class.reset()
                for batch_idx, (inputs, labels, indexes) in enumerate(self.dataloader_usod):
                    inputs = inputs.type(torch.FloatTensor)
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    indexes = indexes.cuda() if torch.cuda.is_available() else indexes

                    so_out, so_up_out, cam_out, sme_out, smc_logits_out, smc_l2norm_out, bridge_out = self.net(inputs)
                    self.produce_class.cal_label(smc_l2norm_out, indexes)
                    pass

                Tools.print("Epoch: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
                Tools.print()
                pass

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_bce, all_loss_mic = 0.0, 0.0, 0.0
            self.net.train()
            for i, (inputs, labels, indexes) in enumerate(self.dataloader_usod):
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                indexes = indexes.cuda() if torch.cuda.is_available() else indexes
                self.optimizer.zero_grad()

                so_out, so_up_out, cam_out, sme_out, smc_logits_out, smc_l2norm_out, bridge_out = self.net(inputs)
                mic_labels = self.produce_class.get_label(indexes)
                mic_labels = mic_labels.cuda() if torch.cuda.is_available() else mic_labels

                # Tools.print("{} {} {}".format(i, smc_logits_out.size(), mic_labels.size()))
                loss, loss_bce, loss_mic = self.all_loss_fusion(so_out, sme_out, smc_logits_out, mic_labels)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_bce += loss_bce.item()
                all_loss_mic += loss_mic.item()
                if i % print_ite_num == 0:
                    Tools.print("[Epoch:{:5d}/{:5d}, batch:{:5d}/{:5d}]  avg loss:{:.3f} loss:{:.3f} "
                                "avg bce:{:.3f} bce:{:.3f} avg mic:{:.3f} mic:{:.3f}".format(
                        epoch + 1, self.epoch_num, i, len(self.dataloader_usod),  all_loss/(i+1), loss.item(),
                        all_loss_bce/(i+1), loss_bce.item(), all_loss_mic/(i+1), loss_mic.item()))
                    pass

                pass

            ###########################################################################
            # 2 保存模型
            if epoch % save_epoch_freq == 0:
                save_file_name = Tools.new_dir(os.path.join(
                    self.model_dir, "usod_{}_train_{:.3f}.pth".format(epoch, all_loss / len(self.dataloader_usod))))
                torch.save(self.net.state_dict(), save_file_name)

                Tools.print()
                Tools.print("Save Model to {}".format(save_file_name))
                Tools.print()
                pass

            pass

        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # bas_runner = BASRunner(batch_size_train=2, data_dir='D:\\data\\SOD\\DUTS\\DUTS-TR')
    bas_runner = BASRunner(batch_size_train=12, model_dir="./saved_models/my_train_mic_sigmoid_mask")
    # bas_runner.load_model('./saved_models/my_train_mic_1/usod_5_train_4.661.pth')
    bas_runner.train()
    pass
