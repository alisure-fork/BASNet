import os
import glob
import torch
from PIL import Image
from skimage import io
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from MyTrain_MIC import BASNet, RescaleT, ToTensor, DatasetUSOD


def norm_PRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_name, predict, d_dir):
    predict_np = predict.squeeze().cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    imo.save(os.path.join(d_dir, '{}.png'.format(os.path.splitext(os.path.basename(image_name))[0])))
    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # --------- 1. get image path and name ---------
    model_dir = './saved_models/my_train_mic_1/usod_45_train_2.651.pth'
    image_dir = './test_data/test_images/'
    prediction_dir = Tools.new_dir('./test_data/my_train_mic_1_45')
    img_name_list = glob.glob(image_dir + '*.jpg')

    # --------- 2. data loader ---------
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
        Tools.print("inference: {}".format(img_name_list[i_test]))
        inputs_test = inputs_test.type(torch.FloatTensor).cuda()

        so_up_out, sme_out, smc_logits_out, smc_l2norm_out = net(inputs_test)

        # normalization and save
        pred = norm_PRED(so_up_out[:, 0, :, :])
        save_output(img_name_list[i_test], pred, prediction_dir)
        pass
