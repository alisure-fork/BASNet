import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
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

class DatasetUSOD(Dataset):

    def __init__(self, img_name_list, is_train=True):
        # self.image_name_list = img_name_list[:20]
        # self.label_name_list = lbl_name_list[:20]
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
        self.encoder0 = ConvBlock(n_channels, 64, has_relu=True)
        self.encoder1 = resnet.layer1  # 224
        self.encoder2 = resnet.layer2  # 112
        self.encoder3 = resnet.layer3  # 56
        self.encoder4 = resnet.layer4  # 28

        # -------------MIC-------------
        self.clustering_num_list = list([128, 256, 512]) if clustering_num_list is None else clustering_num_list

        # MIC 1
        self.mic_1_b1 = ResBlock(512, 512)  # 28
        self.mic_1_b2 = ResBlock(512, 512)
        self.mic_1_b3 = ResBlock(512, 512)
        self.mic_1_c1 = ConvBlock(512, self.clustering_num_list[0], has_relu=True)
        self.mic_1_l2norm = MICNormalize(2)

        # MIC 2
        self.mic_2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_2_b1 = ResBlock(512, 512)  # 14
        self.mic_2_b2 = ResBlock(512, 512)
        self.mic_2_b3 = ResBlock(512, 512)
        self.mic_2_c1 = ConvBlock(512, self.clustering_num_list[1], has_relu=True)
        self.mic_2_l2norm = MICNormalize(2)

        # MIC 3
        self.mic_3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.mic_3_b1 = ResBlock(512, 512)  # 7
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
        smc_logits_1, smc_l2norm_1, smc_sigmoid_1 = self.salient_map_clustering(mic_1)

        return_1 = {
            "mic_f": mic_f_1,
            "mic": mic_1,
            "smc_logits": smc_logits_1,
            "smc_l2norm": smc_l2norm_1,
            "smc_sigmoid": smc_sigmoid_1,
        }

        # 2
        mic_f_2 = self.mic_2_pool(mic_f_1)  # 512 * 14 * 14
        mic_f_2 = self.mic_2_b1(mic_f_2)
        mic_f_2 = self.mic_2_b2(mic_f_2)
        mic_f_2 = self.mic_2_b3(mic_f_2)
        mic_2 = self.mic_2_c1(mic_f_2)  # 512 * 14 * 14
        smc_logits_2, smc_l2norm_2, smc_sigmoid_2 = self.salient_map_clustering(mic_2)

        return_2 = {
            "mic_f": mic_f_2,
            "mic": mic_2,
            "smc_logits": smc_logits_2,
            "smc_l2norm": smc_l2norm_2,
            "smc_sigmoid": smc_sigmoid_2,
        }

        # 3
        mic_f_3 = self.mic_3_pool(mic_f_2)  # 512 * 7 * 7
        mic_f_3 = self.mic_3_b1(mic_f_3)
        mic_f_3 = self.mic_3_b2(mic_f_3)
        mic_f_3 = self.mic_3_b3(mic_f_3)
        mic_3 = self.mic_3_c1(mic_f_3)  # 512 * 7 * 7
        smc_logits_3, smc_l2norm_3, smc_sigmoid_3 = self.salient_map_clustering(mic_3)

        return_3 = {
            "mic_f": mic_f_3,
            "mic": mic_3,
            "smc_logits": smc_logits_3,
            "smc_l2norm": smc_l2norm_3,
            "smc_sigmoid": smc_sigmoid_3,
        }

        return return_1, return_2, return_3

    def salient_map_clustering(self, mic):
        smc_logits = F.adaptive_avg_pool2d(mic, 1).view((mic.size()[0], -1))  # 512

        smc_l2norm = self.mic_1_l2norm(smc_logits)
        smc_sigmoid = torch.sigmoid(smc_logits)
        return smc_logits, smc_l2norm, smc_sigmoid

    pass


#######################################################################################################################
# 3 Runner


class BASRunner(object):

    def __init__(self, batch_size_train=8, clustering_num_1=128, clustering_num_2=256, clustering_num_3=512,
                 clustering_ratio_1=1, clustering_ratio_2=1.5, clustering_ratio_3=2,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/my_train_mic5"):
        self.batch_size_train = batch_size_train

        # Dataset
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.tra_img_name_list, tra_lbl_name_list = self.get_tra_img_label_name()
        self.dataset_usod = DatasetUSOD(img_name_list=self.tra_img_name_list, is_train=True)
        self.dataloader_usod = DataLoader(self.dataset_usod, self.batch_size_train, shuffle=True, num_workers=8)

        # Model
        self.net = BASNet(3, clustering_num_list=[clustering_num_1,
                                                  clustering_num_2, clustering_num_3])

        ###########################################################################
        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
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

        # Loss and Optim
        self.bce_loss = nn.BCELoss()
        self.mic_loss = nn.CrossEntropyLoss()
        self.bce_loss = self.bce_loss.cuda() if torch.cuda.is_available() else self.bce_loss
        self.mic_loss = self.mic_loss.cuda() if torch.cuda.is_available() else self.mic_loss

        self.learning_rate = [[0, 0.01], [400, 0.001]]
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate[0][1],
                                    betas=(0.9, 0.999), weight_decay=0)
        pass

    def _adjust_learning_rate(self, epoch):
        learning_rate = self.learning_rate[0][1]
        for param_group in self.optimizer.param_groups:
            for lr in self.learning_rate:
                if epoch == lr[0]:
                    learning_rate = lr[1]
                    param_group['lr'] = learning_rate
            pass
        return learning_rate

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
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list

    def all_loss_fusion(self, mic_1_out, mic_2_out, mic_3_out, mic_labels_1, mic_labels_2, mic_labels_3):
        loss_mic_1 = self.mic_loss(mic_1_out, mic_labels_1)
        loss_mic_2 = self.mic_loss(mic_2_out, mic_labels_2)
        loss_mic_3 = self.mic_loss(mic_3_out, mic_labels_3)

        loss_all = loss_mic_1 + loss_mic_2 + loss_mic_3
        return loss_all, loss_mic_1, loss_mic_2, loss_mic_3

    def train(self, epoch_num=200, start_epoch=0, save_epoch_freq=10, print_ite_num=100):

        if start_epoch > 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            with torch.no_grad():
                for _idx, (inputs, indexes) in tqdm(enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                    inputs = inputs.type(torch.FloatTensor)
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    indexes = indexes.cuda() if torch.cuda.is_available() else indexes

                    return_1, return_2, return_3 = self.net(inputs)
                    self.produce_class11.cal_label(return_1["smc_logits"], indexes)
                    self.produce_class21.cal_label(return_2["smc_logits"], indexes)
                    self.produce_class31.cal_label(return_3["smc_logits"], indexes)
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
            Tools.print("Train: [{}] 1-{}/{}".format(start_epoch, self.produce_class11.count,
                                                     self.produce_class11.count_2))
            Tools.print("Train: [{}] 2-{}/{}".format(start_epoch, self.produce_class21.count,
                                                     self.produce_class21.count_2))
            Tools.print("Train: [{}] 3-{}/{}".format(start_epoch, self.produce_class31.count,
                                                     self.produce_class31.count_2))
            pass

        all_loss = 0
        for epoch in range(start_epoch, epoch_num):
            Tools.print()
            lr = self._adjust_learning_rate(epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f} lr={:.5f}'.format(epoch, lr, self.optimizer.param_groups[0]['lr']))

            ###########################################################################
            # 1 训练模型
            all_loss, all_loss_mic_1, all_loss_mic_2, all_loss_mic_3 = 0.0, 0.0, 0.0, 0.0
            self.net.train()

            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()

            for i, (inputs, indexes) in tqdm(enumerate(self.dataloader_usod), total=len(self.dataloader_usod)):
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                indexes = indexes.cuda() if torch.cuda.is_available() else indexes
                self.optimizer.zero_grad()

                return_1, return_2, return_3 = self.net(inputs)

                self.produce_class11.cal_label(return_1["smc_logits"], indexes)
                self.produce_class21.cal_label(return_2["smc_logits"], indexes)
                self.produce_class31.cal_label(return_3["smc_logits"], indexes)

                mic_labels_1 = self.produce_class12.get_label(indexes)
                mic_labels_1 = mic_labels_1.cuda() if torch.cuda.is_available() else mic_labels_1
                mic_labels_2 = self.produce_class22.get_label(indexes)
                mic_labels_2 = mic_labels_2.cuda() if torch.cuda.is_available() else mic_labels_2
                mic_labels_3 = self.produce_class32.get_label(indexes)
                mic_labels_3 = mic_labels_3.cuda() if torch.cuda.is_available() else mic_labels_3

                loss, loss_mic_1, loss_mic_2, loss_mic_3 = self.all_loss_fusion(
                    return_1["smc_logits"], return_2["smc_logits"], return_3["smc_logits"],
                    mic_labels_1, mic_labels_2, mic_labels_3)
                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                all_loss_mic_1 += loss_mic_1.item()
                all_loss_mic_2 += loss_mic_2.item()
                all_loss_mic_3 += loss_mic_3.item()
                if i % print_ite_num == 0:
                    Tools.print(
                        "[E:{:3d}/{:3d}, b:{:3d}/{:3d}] loss:{:.3f} loss:{:.3f} "
                        "mic 1:{:.3f}/{:.3f} mic2:{:.3f}/{:.3f} mic3:{:.3f}/{:.3f}".format(
                            epoch, epoch_num, i, len(self.dataloader_usod),
                            all_loss/(i+1), loss.item(), all_loss_mic_1/(i+1), loss_mic_1.item(),
                            all_loss_mic_2/(i+1), loss_mic_2.item(), all_loss_mic_3/(i+1), loss_mic_3.item()))
                    pass

                pass

            Tools.print("[E:{:3d}/{:3d}] loss:{:.3f} mic 1:{:.3f} mic2:{:.3f} mic3:{:.3f}".format(
                epoch, epoch_num, all_loss / (len(self.dataloader_usod) + 1),
                all_loss_mic_1 / (len(self.dataloader_usod) + 1),
                all_loss_mic_2 / (len(self.dataloader_usod) + 1),
                all_loss_mic_3 / (len(self.dataloader_usod) + 1)))

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

    """
    2020-06-15 06:11:23 [E:299/300] loss:3.849 mic 1:1.476 mic2:1.213 mic3:1.159
    """

    bas_runner = BASRunner(batch_size_train=16 * 4, data_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR",
                           clustering_num_1=128 * 4, clustering_num_2=128 * 4, clustering_num_3=128 * 4,
                           model_dir="../BASNetTemp/saved_models/my_train_mic5_large_demo")
    bas_runner.load_model('../BASNetTemp/saved_models/my_train_mic5_large/500_train_0.880.pth')
    bas_runner.train(epoch_num=500, start_epoch=0)
    pass
