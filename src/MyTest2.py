import os
import glob
import torch
from PIL import Image
from skimage import io
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.MyTrain2 import BASNet, RescaleT, ToTensor, SalObjDataset


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

    # --------- 1. get image path and name ---------
    model_dir = './saved_models/my_train_2/basnet_50000_train_0.033.pth'
    image_dir = './test_data/test_images/'
    prediction_dir = Tools.new_dir('./test_data/test_results_my_train_2_50000')
    img_name_list = glob.glob(image_dir + '*.jpg')

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensor()]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    Tools.print("...load BASNet...")
    net = BASNet(3, pretrained=False)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()

    # --------- 4. inference for each image ---------
    net.eval()
    for i_test, data_test in enumerate(test_salobj_dataloader):
        Tools.print("inferencing: {}".format(img_name_list[i_test]))

        inputs_test = data_test['image'].type(torch.FloatTensor)
        inputs_test = Variable(inputs_test.cuda()) if torch.cuda.is_available() else Variable(inputs_test)

        do, _ = net(inputs_test)

        # normalization and save
        pred = norm_PRED(do[:, 0, :, :])
        save_output(img_name_list[i_test], pred, prediction_dir)
        pass
