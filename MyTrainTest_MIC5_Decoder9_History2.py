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

    def __init__(self, n_channels, clustering_num):
        super(BASNet, self).__init__()
        self.clustering_num = clustering_num

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
        self.mic_b1 = ResBlock(512, 512)  # 28
        self.mic_b2 = ResBlock(512, 512)
        self.mic_b3 = ResBlock(512, 512)
        self.mic_c1 = ConvBlock(512, self.clustering_num, has_relu=True)

        # -------------Decoder-------------
        # Decoder 1
        self.decoder_b1 = ResBlock(512, 512)  # 28
        self.decoder_b2 = ResBlock(512, 512)  # 28
        self.decoder_b3 = ResBlock(512, 512)  # 28
        self.decoder_out = nn.Conv2d(512, 1, 3, padding=1, bias=False)  # 28
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
        mic_feature = self.mic_b3(self.mic_b2(self.mic_b1(e4)))  # 512 * 28 * 28
        mic = self.mic_c1(mic_feature)  # 128 * 28 * 28
        smc_logits, smc_l2norm = self.salient_map_clustering(mic)
        result = {"smc_logits": smc_logits, "smc_l2norm": smc_l2norm}

        # -------------Label-------------
        cam = self.cluster_activation_map(smc_logits, mic)  # 簇激活图：Cluster Activation Map
        cam_up = self._up_to_target(cam, x_for_up)
        cam_up_norm = self._feature_norm(cam_up)
        result["cam_up_norm"] = cam_up_norm

        # label = self.salient_map_divide(cam_up_norm, obj_th=0.8, bg_th=0.2, more_obj=False)  # 显著图划分
        label = cam_up_norm
        result["label"] = label

        # -------------Decoder-------------
        # decoder
        d = self.decoder_b3(self.decoder_b2(self.decoder_b1(e4)))  # 512 * 28 * 28
        d_out = self.decoder_out(d)  # 1 * 28 * 28
        d_out_sigmoid = torch.sigmoid(d_out)  # 1 * 28 * 28 # 小输出
        d_out_up = self._up_to_target(d_out, x_for_up)  # 1 * 224 * 224
        d_out_up_sigmoid = torch.sigmoid(d_out_up)  # 1 * 224 * 224  # 大输出
        result["d_out"] = d_out
        result["d_out_sigmoid"] = d_out_sigmoid
        result["d_out_up"] = d_out_up
        result["d_out_up_sigmoid"] = d_out_up_sigmoid

        return result

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

    def __init__(self, batch_size_train=8, clustering_num=128, clustering_ratio=1, only_mic=False, has_history=False,
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
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=16)

        # Model
        self.net = BASNet(3, clustering_num=clustering_num)

        ###########################################################################
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
            cudnn.benchmark = True
        ###########################################################################

        # MIC
        self.produce_class1 = MICProduceClass(
            n_sample=len(self.dataset_usod), out_dim=clustering_num, ratio=clustering_ratio)
        self.produce_class2 = MICProduceClass(
            n_sample=len(self.dataset_usod), out_dim=clustering_num, ratio=clustering_ratio)
        self.produce_class1.init()
        self.produce_class2.init()

        # Loss and optimizer
        self.bce_loss = nn.BCELoss().cuda()
        self.mic_loss = nn.CrossEntropyLoss().cuda()
        self.learning_rate = [[0, 0.001], [300, 0.0001], [400, 0.00001]]
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

    def all_loss_fusion(self, mic_out, mic_label, sod_output, sod_label, only_mic=False):
        loss_mic = self.mic_loss(mic_out, mic_label)

        positions = sod_label.view(-1, 1) < 255.0
        loss_bce = self.bce_loss(sod_output.view(-1, 1)[positions], sod_label.view(-1, 1)[positions])

        loss_all = loss_mic
        if not only_mic:
            loss_all = loss_all + loss_bce
            pass
        return loss_all, loss_mic, loss_bce

    def save_history_info(self, histories, params, indexes):
        for history, param, index in zip(histories, params, indexes):
            self.dataset_usod.save_history(idx=int(index),
                                           his=np.asarray(history.squeeze().detach().cpu()), param=np.asarray(param))
        pass

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=50, eval_epoch_freq=10):

        if start_epoch > 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class1.reset()
            with torch.no_grad():
                for _idx, (inputs, histories, params, indexes) in tqdm(
                        enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                    inputs = inputs.type(torch.FloatTensor).cuda()
                    indexes = indexes.cuda()

                    result = self.net(inputs)

                    self.produce_class1.cal_label(result["smc_l2norm"], indexes)
                    pass
                pass
            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Update: [{}] {}/{}".format(
                start_epoch, self.produce_class1.count, self.produce_class1.count_2))
            pass

        all_loss = 0
        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic, all_loss_sod = 0.0, 0.0, 0.0
            Tools.print()
            self.net.train()

            self.produce_class1.reset()

            for i, (inputs, histories, params, indexes) in tqdm(
                    enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                histories = histories.type(torch.FloatTensor).cuda()
                indexes = indexes.cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)

                ######################################################################################################
                # MIC
                self.produce_class1.cal_label(result["smc_logits"], indexes)
                mic_target = result["smc_logits"]
                mic_label = self.produce_class2.get_label(indexes).cuda()
                ######################################################################################################

                ######################################################################################################
                # SOD
                histories = histories  # Annotation
                sod_label = result["label"].detach()  # CAM
                sod_output = result["d_out_up_sigmoid"]  # Predict

                if self.has_history:
                    # NO History
                    # histories = (histories + sod_output) / 2

                    # 历史信息 = 历史信息 + CAM + SOD
                    # sod_label = self.sigmoid(sod_label * 20, a=12)
                    # sod_label = sod_label if histories.max() == 0 else (histories * 0.5 + sod_label * 0.5)
                    sod_label = sod_label if histories.max() == 0 else (histories * 0.8 + sod_label * 0.2)
                    # histories = sod_label * 2 / 3 + sod_output * 1 / 3
                    histories = sod_label

                    self.save_history_info(histories=histories, params=params, indexes=indexes)
                    pass
                ######################################################################################################

                loss, loss_mic, loss_sod = self.all_loss_fusion(
                    mic_target, mic_label, sod_output, sod_label, only_mic=self.only_mic)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic += loss_mic.item()
                all_loss_sod += loss_sod.item()
                if i % print_ite_num == 0:
                    Tools.print("[E:{:4d}/{:4d}, b:{:4d}/{:4d}] l:{:.2f}/{:.2f} "
                                "mic:{:.2f}/{:.2f} sod:{:.2f}/{:.2f}".format(
                        epoch, epoch_num, i, len(self.dataloader_usod),  all_loss/(i+1), loss.item(),
                        all_loss_mic/(i+1), loss_mic.item(), all_loss_sod/(i+1), loss_sod.item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic:{:.3f} sod:{:.3f}".format(
                epoch, epoch_num, all_loss / (len(self.dataloader_usod) + 1),
                all_loss_mic / (len(self.dataloader_usod) + 1), all_loss_sod / (len(self.dataloader_usod) + 1)))

            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class1.count, self.produce_class1.count_2))

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    bas_runner = BASRunner(batch_size_train=16 * 3, clustering_num=128 * 4, has_history=True,
                           data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           history_dir="../BASNetTemp/history/my_train_mic5_large_history2_one",
                           model_dir="../BASNetTemp/saved_models/my_train_mic5_large_history2_one")
    bas_runner.load_model('../BASNetTemp/saved_models/my_train_mic5_large_history2_one/60_train_5.365.pth')
    bas_runner.train(epoch_num=500, start_epoch=1)
    pass
