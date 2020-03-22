import os
import glob
import torch
from PIL import Image
from skimage import io
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from src.MyTrain_MIC5 import BASNet, DatasetUSOD


def one_5():

    # --------- 1. get path ---------
    # model_dir = './saved_models/my_train5_diff_aug_mask/125_train_6.569.pth'
    # prediction_dir = Tools.new_dir('./test_data/my_train5_diff_aug_mask_125_image')
    model_dir = './saved_models/my_train5_diff_aug_nomask/125_train_6.044.pth'
    prediction_dir = Tools.new_dir('./test_data/my_train5_diff_aug_nomask_125_image')

    # --------- 2. data loader ---------
    image_dir = '/mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/'
    img_name_list = glob.glob(image_dir + '*.jpg')
    test_dataset = DatasetUSOD(img_name_list=img_name_list, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # --------- 3. model define ---------
    Tools.print("...load BASNet...")
    net = BASNet(3, clustering_num_list=[128, 256, 512], pretrained=False)
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load(model_dir))

    # --------- 4. inference for each image ---------
    net.eval()
    for i_test, (inputs_test, _) in enumerate(test_dataloader):
        Tools.print("inference: {} {}".format(i_test, img_name_list[i_test]))
        inputs_test = inputs_test.type(torch.FloatTensor).cuda()

        return_1, return_2, return_3 = net(inputs_test)
        return_which = return_1

        top_k_value, top_k_index = torch.topk(return_which["smc_logits"], 5, 1)
        smc_result = top_k_index.cpu().detach().numpy()[0]

        img_name = img_name_list[i_test]
        result_path = os.path.join(prediction_dir, str(smc_result[0]))
        result_path = Tools.new_dir(result_path)

        # 1
        result_name = os.path.join(result_path, os.path.split(img_name)[1])
        im_data = io.imread(img_name)
        io.imsave(result_name, im_data)

        # 2
        for i, smc in enumerate(smc_result):
            d = return_which["mic"][:, smc, :, :]
            pred = (d - torch.min(d)) / (torch.max(d) - torch.min(d))
            predict_np = pred.squeeze().cpu().data.numpy()
            im = Image.fromarray(predict_np * 255).convert('RGB')
            imo = im.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
            imo.save(os.path.join(result_path, '{}_{}_{}.png'.format(
                os.path.splitext(os.path.basename(img_name))[0], i, smc)))
            pass

        pass

    pass


def one_1_3():
    # --------- 1. get path ---------
    # model_dir = './saved_models/my_train5_diff_aug_mask/125_train_6.569.pth'
    # prediction_dir = Tools.new_dir('./test_data/my_train5_diff_aug_mask_125_image2')
    model_dir = './saved_models/my_train5_diff_aug_nomask/125_train_6.044.pth'
    prediction_dir = Tools.new_dir('./test_data/my_train5_diff_aug_nomask_125_image2')

    # --------- 2. data loader ---------
    image_dir = '/mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/'
    img_name_list = glob.glob(image_dir + '*.jpg')
    test_dataset = DatasetUSOD(img_name_list=img_name_list, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # --------- 3. model define ---------
    Tools.print("...load BASNet...")
    net = BASNet(3, clustering_num_list=[128, 256, 512], pretrained=False)
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load(model_dir))

    # --------- 4. inference for each image ---------
    net.eval()
    for i_test, (inputs_test, _) in enumerate(test_dataloader):
        Tools.print("inference: {} {}".format(i_test, img_name_list[i_test]))
        inputs_test = inputs_test.type(torch.FloatTensor).cuda()

        return_1, return_2, return_3 = net(inputs_test)

        top_k_value, top_k_index = torch.topk(return_1["smc_logits"], 1, 1)
        smc_result = top_k_index.cpu().detach().numpy()[0][0]

        img_name = img_name_list[i_test]
        result_path = os.path.join(prediction_dir, str(smc_result))
        result_path = Tools.new_dir(result_path)

        # 1
        result_name = os.path.join(result_path, os.path.split(img_name)[1])
        im_data = io.imread(img_name)
        io.imsave(result_name, im_data)

        for i, return_which in enumerate([return_1, return_2, return_3]):
            top_k_value, top_k_index = torch.topk(return_which["smc_logits"], 1, 1)
            smc_result = top_k_index.cpu().detach().numpy()[0][0]
            d = return_which["mic"][:, smc_result, :, :]
            pred = (d - torch.min(d)) / (torch.max(d) - torch.min(d))
            predict_np = pred.squeeze().cpu().data.numpy()
            im = Image.fromarray(predict_np * 255).convert('RGB')
            imo = im.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
            imo.save(os.path.join(result_path, '{}_{}_{}.png'.format(
                    os.path.splitext(os.path.basename(img_name))[0], i, smc_result)))
            pass
        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    one_5()
    pass
