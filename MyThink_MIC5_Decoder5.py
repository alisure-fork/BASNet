import os
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from MyTrain_MIC5_Decoder5 import BASNet, DatasetUSOD


def one_decoder():
    # --------- 1. get path ---------
    has_mask = True
    model_dir = './saved_models/my_train_mic5_decoder5_aug_mask_1_small/125_train_1.699.pth'
    prediction_dir = Tools.new_dir('./test_data/my_train_mic5_decoder5_aug_mask_1_small_125_image_decoder')

    # --------- 2. data loader ---------
    image_dir = '/mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/'
    img_name_list = glob.glob(image_dir + '*.jpg')
    test_dataset = DatasetUSOD(img_name_list=img_name_list, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # --------- 3. model define ---------
    Tools.print("...load BASNet...")
    net = BASNet(3, clustering_num_list=[128, 256, 512], pretrained=False, has_mask=has_mask)
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load(model_dir), strict=False)

    # --------- 4. inference for each image ---------
    net.eval()
    for i_test, (inputs_test, _) in enumerate(test_dataloader):
        Tools.print("inference: {} {}".format(i_test, img_name_list[i_test]))
        inputs_test = inputs_test.type(torch.FloatTensor).cuda()

        return_1, return_2, return_3, return_d = net(inputs_test)

        top_k_value, top_k_index = torch.topk(return_1["smc_logits"], 1, 1)
        smc_result = top_k_index.cpu().detach().numpy()[0][0]

        img_name = img_name_list[i_test]
        result_path = os.path.join(prediction_dir, str(smc_result))
        result_path = Tools.new_dir(result_path)

        # 1
        result_name = os.path.join(result_path, os.path.split(img_name)[1])
        im_data = io.imread(img_name)
        io.imsave(result_name, im_data)

        # 2
        cam1 = return_d["cam_norm_1_up"].squeeze().cpu().data.numpy()
        cam2 = return_d["cam_norm_2_up"].squeeze().cpu().data.numpy()
        cam3 = return_d["cam_norm_3_up"].squeeze().cpu().data.numpy()

        im1 = Image.fromarray(cam1 * 255).convert('RGB')
        im2 = Image.fromarray(cam2 * 255).convert('RGB')
        im3 = Image.fromarray(cam3 * 255).convert('RGB')

        imo1 = im1.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
        imo2 = im2.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
        imo3 = im3.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)

        imo1.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], 1, smc_result)))
        imo2.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], 2, smc_result)))
        imo3.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], 3, smc_result)))

        # 3
        camf = return_d["cam_norm_up"].squeeze().cpu().data.numpy()
        imf = Image.fromarray(camf * 255).convert('RGB')
        imof = imf.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
        imof.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], "f", smc_result)))

        # 4
        label = return_d["label"].squeeze().cpu().data.numpy()
        im_label = Image.fromarray((np.asarray(label, dtype=np.uint8) + 1) * 127).convert('RGB')
        imo_label = im_label.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
        imo_label.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], "l", smc_result)))

        # 5
        d1_out_up_sigmoid = return_d["d1_out_up_sigmoid"].squeeze().cpu().data.numpy()
        im_d1_out_up_sigmoid = Image.fromarray(d1_out_up_sigmoid * 255).convert('RGB')
        imo_d1_out_up_sigmoid = im_d1_out_up_sigmoid.resize((im_data.shape[1], im_data.shape[0]),
                                                            resample=Image.BILINEAR)
        imo_d1_out_up_sigmoid.save(os.path.join(result_path, '{}_{}_{}.png'.format(
            os.path.splitext(os.path.basename(img_name))[0], "d", smc_result)))
        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    one_decoder()
    pass

