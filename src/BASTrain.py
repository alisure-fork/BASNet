import os
import glob
import torch
import torch.nn as nn
from src.BASSsim import SSIM
from src.BASNet import BASNet
import torch.optim as optim
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.BASData import RescaleT, RandomCrop, ToTensor, SalObjDataset


class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
        pass

    def forward(self, pred, target):
        return self._iou(pred, target)

    @staticmethod
    def _iou(pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            IoU = IoU + (1 - IoU1)
            pass
        return IoU / b

    pass


class BASRunner(object):

    def __init__(self, epoch_num=100000, batch_size_train=4,
                 data_dir='/mnt/4T/Data/SOD/DUTS/DUTS-TR', tra_image_dir='DUTS-TR-Image',
                 tra_label_dir='DUTS-TR-Mask', model_dir="./saved_models/basnet_bsi1"):
        self.epoch_num = epoch_num
        self.batch_size_train = batch_size_train

        # Dataset
        self.data_dir = data_dir
        self.tra_image_dir = tra_image_dir
        self.tra_label_dir = tra_label_dir
        self.model_dir = model_dir
        self.tra_img_name_list, self.tra_lbl_name_list = self.get_tra_img_label_name()
        self.salobj_dataset = SalObjDataset(
            img_name_list=self.tra_img_name_list, lbl_name_list=self.tra_lbl_name_list,
            transform=transforms.Compose([RescaleT(256), RandomCrop(224), ToTensor()]))
        self.salobj_dataloader = DataLoader(self.salobj_dataset, self.batch_size_train, shuffle=True, num_workers=4)

        # Model
        self.net = BASNet(3, pretrained=False)
        if torch.cuda.is_available():
            self.net.cuda()
            pass

        # Loss
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = IOU()

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        pass

    def load_model(self, model_dir):
        self.net.load_state_dict(torch.load(model_dir))
        Tools.print("restore from {}".format(model_dir))
        pass

    def get_tra_img_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_dir, self.tra_image_dir, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_dir, self.tra_label_dir, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        Tools.print("train images: {}".format(len(tra_img_name_list)))
        Tools.print("train labels: {}".format(len(tra_lbl_name_list)))
        return tra_img_name_list, tra_lbl_name_list

    def bce_ssim_loss(self, pred, target):
        bce_out = self.bce_loss(pred, target)
        ssim_out = 1 - self.ssim_loss(pred, target)
        iou_out = self.iou_loss(pred, target)
        loss = bce_out + ssim_out + iou_out
        return loss

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, d7, labels_v, is_print=False):
        loss0 = self.bce_ssim_loss(d0, labels_v)
        loss1 = self.bce_ssim_loss(d1, labels_v)
        loss2 = self.bce_ssim_loss(d2, labels_v)
        loss3 = self.bce_ssim_loss(d3, labels_v)
        loss4 = self.bce_ssim_loss(d4, labels_v)
        loss5 = self.bce_ssim_loss(d5, labels_v)
        loss6 = self.bce_ssim_loss(d6, labels_v)
        loss7 = self.bce_ssim_loss(d7, labels_v)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7  # + 5.0*lossa
        if is_print:
            Tools.print("L0:{:.3f}, L1:{:.3f}, L2:{:.3f}, L3:{:.3f}, L4:{:.3f}, L5:{:.3f}, L6:{:.3f}".format(
                loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
            pass
        return loss0, loss

    def train(self, save_ite_num=5000, print_ite_num=100):
        ite_num = 0
        ite_num4val = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        self.net.train()

        for epoch in range(0, self.epoch_num):
            for i, data in enumerate(self.salobj_dataloader):
                is_print = ite_num % print_ite_num == 0
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # y zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6, d7 = self.net(inputs)
                loss0, loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels, is_print)
                loss.backward()
                self.optimizer.step()

                # # print statistics
                running_loss += loss.item()
                running_tar_loss += loss0.item()

                if is_print:
                    Tools.print("[Epoch:{:5d}/{:5d},batch:{:5d}/{:5d},ite:{}] train loss: {:.3f}, tar: {:.3f}".format(
                        epoch + 1, self.epoch_num, (i + 1) * self.batch_size_train, len(self.tra_img_name_list),
                        ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                    pass

                if ite_num % save_ite_num == 0:
                    save_file_name = Tools.new_dir(os.path.join(
                        self.model_dir, "basnet_bsi_itr_{}_train_{:.3f}_tar_{:.3f}.pth".format(
                            ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)))
                    torch.save(self.net.state_dict(), save_file_name)

                    running_loss = 0.0
                    running_tar_loss = 0.0
                    ite_num4val = 0
                    Tools.print()
                    Tools.print("Save Model to {}".format(save_file_name))
                    Tools.print()
                    pass

                pass
            pass

        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    bas_runner = BASRunner()
    bas_runner.load_model('./saved_models/basnet_bsi2/basnet_bsi_itr_50000_train_3.627_tar_0.350.pth')
    bas_runner.train()
    pass
