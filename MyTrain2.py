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
from torch.autograd import Variable
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


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        label_3 = np.zeros(image.shape) if 0 == len(self.label_name_list) else io.imread(self.label_name_list[idx])

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

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
        bridge = self.bridge_relu_3(self.bridge_bn_3(self.bridge_conv_3(
            self.bridge_relu_2(self.bridge_bn_2(self.bridge_conv_2(
                self.bridge_relu_1(self.bridge_bn_1(self.bridge_conv_1(e4)))))))))  # 512 * 28 * 28

        # -------------Decoder-------------
        ob = self.out_conv_bridge(bridge)  # 1 * 28 * 28
        ob_up = self.upscore4(ob)  # 1 * 224 * 224
        so = F.sigmoid(ob)  # 1 * 224 * 224  # 可考虑使用norm
        so_up = F.sigmoid(ob_up)  # 1 * 224 * 224  # 可考虑使用norm

        # More
        smc = self.smc(bridge, so)
        cam = self.cam(smc, bridge)
        sme = self.sme(cam)

        ###########################
        # x_data = np.transpose(x.data.numpy()[0], axes=[1, 2, 0])
        # Image.fromarray(np.asarray(((x_data-np.min(x_data)) / np.max(x_data - np.min(x_data))) * 255, dtype=np.uint8)).show()
        # Image.fromarray(np.asarray(so_up.data.numpy()[0][0] * 255, dtype=np.uint8)).show()
        # x_data = np.sum(np.transpose(bridge.data.numpy()[0], axes=[1, 2, 0])[:,:,top_k_index[0]], axis=2)
        # Image.fromarray(np.asarray(((x_data-np.min(x_data)) / np.max(x_data - np.min(x_data))) * 255, dtype=np.uint8)).show()
        ###########################

        return so_up, sme

    @staticmethod
    def smc(feature_for_smc, mask_b):
        smc = feature_for_smc * mask_b  # 512 * 28 * 28
        feature_smc = F.adaptive_avg_pool2d(smc, 1).view((smc.size()[0], -1))  # 512
        return feature_smc

    @staticmethod
    def cam(feature_for_clustering, feature_for_cam, k=5):
        top_k_value, top_k_index = torch.topk(feature_for_clustering, k, 1)
        cam = torch.cat([feature_for_cam[i:i+1, top_k_index[i], :, :].mean(
            1, keepdim=True) for i in range(feature_for_cam.size()[0])])
        return cam

    def sme(self, cam):
        for_cam_norm = self.feature_norm(cam)  # 1 * 28 * 28
        for_cam_norm_up = self.upscore4(for_cam_norm)  # 1 * 224 * 224
        for_cam_norm_up[for_cam_norm_up > 0.6] = 2
        for_cam_norm_up[for_cam_norm_up < 0.05] = 1
        for_cam_norm_up[for_cam_norm_up < 1] = 0
        return for_cam_norm_up

    @staticmethod
    def feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, epoch_num=100000, batch_size_train=8, batch_size_val=1,
                 data_dir='D:\\data\\SOD\\DUTS\\DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir=".\\saved_models\\basnet_bce_simple"):
        self.epoch_num = epoch_num
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        # Dataset
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.model_dir = model_dir
        self.tra_img_name_list, self.tra_lbl_name_list = self.get_tra_img_label_name()
        self.salobj_dataset = SalObjDataset(
            img_name_list=self.tra_img_name_list, lbl_name_list=self.tra_lbl_name_list,
            transform=transforms.Compose([RescaleT(256), RandomCrop(224), ToTensor()]))
        self.salobj_dataloader = DataLoader(self.salobj_dataset, self.batch_size_train, shuffle=True, num_workers=1)

        # Model
        self.net = BASNet(3, pretrained=True)
        if torch.cuda.is_available():
            self.net.cuda()
            pass

        # Loss and Optim
        self.bce_loss = nn.BCELoss(size_average=True)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        pass

    def get_tra_img_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, self.tra_image_dir, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_dir, self.tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list

    def all_bce_loss_fusion(self, so_up, labels_v):
        loss_so_up = self.bce_loss(so_up, labels_v)
        return loss_so_up

    def load_model(self, model_file_name):
        self.net.load_state_dict(torch.load(model_file_name))
        Tools.print("restore from {}".format(model_file_name))
        pass

    def train(self, save_ite_num=300):
        ite_num = 0
        ite_num4val = 0
        running_loss = 0.0
        self.net.train()

        for epoch in range(0, self.epoch_num):
            for i, data in enumerate(self.salobj_dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                self.optimizer.zero_grad()
                so_up = self.net(inputs)
                loss = self.all_bce_loss_fusion(so_up, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                Tools.print("[Epoch:{:5d}/{:5d},batch:{:5d}/{:5d},ite:{}] avg loss:{:.3f} loss:{:.3f}".format(
                    epoch + 1, self.epoch_num, (i + 1) * self.batch_size_train, len(self.tra_img_name_list),
                    ite_num, running_loss / ite_num4val, loss.item()))

                if ite_num % save_ite_num == 0:
                    save_file_name = Tools.new_dir(os.path.join(
                        self.model_dir, "basnet_{}_train_{:.3f}.pth".format(ite_num, running_loss / ite_num4val)))
                    torch.save(self.net.state_dict(), save_file_name)

                    running_loss = 0.0
                    ite_num4val = 0
                    Tools.print()
                    Tools.print("Save Model to {}".format(save_file_name))
                    Tools.print()
                    pass
                pass
            pass

        pass

    pass


#######################################################################################################################
# 4 Main


if __name__ == '__main__':
    bas_runner = BASRunner(batch_size_train=2)
    bas_runner.load_model('./saved_models/basnet_bce_simple/basnet_2100_train_0.310.pth')
    bas_runner.train()
    pass
