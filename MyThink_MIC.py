import os
import glob
import torch
from PIL import Image
from skimage import io
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from MyTrain_MIC import BASNet, RescaleT, ToTensor, DatasetUSOD


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # --------- 1. get image path and name ---------
    # model_dir = './saved_models/my_train_mic_only/usod_80_train_1.992.pth'
    # model_dir = './saved_models/my_train_mic_only_nomask_mask/usod_60_train_1.749.pth'
    model_dir = './saved_models/my_train_mic_only_norelu_mask/usod_55_train_2.062.pth'
    # model_dir = './saved_models/my_train_mic_only_norelu/usod_75_train_2.160.pth'
    prediction_dir = Tools.new_dir('./test_data/my_train_mic_only_norelu_mask_55_image_relu')
    image_dir = '/mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/'

    # --------- 2. data loader ---------
    img_name_list = glob.glob(image_dir + '*.jpg')
    test_dataset = DatasetUSOD(img_name_list=img_name_list, lbl_name_list=None,
                               transform=transforms.Compose([RescaleT(256), ToTensor()]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    Tools.print("...load BASNet...")
    net = BASNet(3, pretrained=False)
    if torch.cuda.is_available():
        net.cuda()

    net.load_state_dict(torch.load(model_dir))

    # --------- 4. inference for each image ---------
    net.eval()
    for i_test, (inputs_test, _, _) in enumerate(test_dataloader):
        Tools.print("inference: {} {}".format(i_test, img_name_list[i_test]))
        inputs_test = inputs_test.type(torch.FloatTensor).cuda()

        so_out, so_up_out, cam_out, sme_out, smc_logits_out, smc_l2norm_out, bridge_out = net(inputs_test)
        top_k_value, top_k_index = torch.topk(smc_logits_out, 5, 1)
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
            d = bridge_out[:, smc, :, :]
            pred = (d - torch.min(d)) / (torch.max(d) - torch.min(d))
            predict_np = pred.squeeze().cpu().data.numpy()
            im = Image.fromarray(predict_np * 255).convert('RGB')
            imo = im.resize((im_data.shape[1], im_data.shape[0]), resample=Image.BILINEAR)
            imo.save(os.path.join(result_path, '{}_{}_{}.png'.format(
                os.path.splitext(os.path.basename(img_name))[0], i, smc)))
            pass

        pass

    pass
