import os
import glob
import numpy as np
from alisuretool.Tools import Tools
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_softmax


def crf_inference(img, probs, t=10, label_num=2):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, label_num)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = d.inference(t)

    return np.array(q).reshape((label_num, h, w))


def demo():
    image_list = ["ILSVRC2012_test_00000340", "ILSVRC2012_test_00000363",
                  "ILSVRC2012_test_00000450", "ILSVRC2012_test_00000678", "ILSVRC2012_test_00000692"]
    for image_name in image_list:
        image = imread("./data/{}.jpg".format(image_name))
        annotated_image = imread("./data/{}.bmp".format(image_name))

        annotated_data = np.expand_dims(annotated_image / 255, axis=0)
        annotated_data = np.concatenate([annotated_data, 1 - annotated_data], axis=0)
        output = crf_inference(image, annotated_data)

        output1 = np.asarray(output[0] * 255, dtype=np.uint8)
        output2 = np.asarray(output[1] * 255, dtype=np.uint8)

        imsave("./result/{}_1.bmp".format(image_name), output1)
        imsave("./result/{}_2.bmp".format(image_name), output2)
        pass
    pass


def crf_inference2(img, annotation, normalization=dcrf.NORMALIZE_SYMMETRIC):
    annotation = np.expand_dims(annotation, axis=0)
    annotation = np.concatenate([annotation, 1 - annotation], axis=0)

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(annotation)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    # DIAG_KERNEL           CONST_KERNEL     FULL_KERNEL
    # NORMALIZE_SYMMETRIC   NORMALIZE_BEFORE NO_NORMALIZATION  NORMALIZE_AFTER
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=normalization)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10,
                           kernel=dcrf.DIAG_KERNEL, normalization=normalization)
    q = d.inference(10)

    result = np.array(q).reshape((2, h, w))
    return result[0]


def demo_2():
    image_list = ["ILSVRC2012_test_00000340", "ILSVRC2012_test_00000363",
                  "ILSVRC2012_test_00000450", "ILSVRC2012_test_00000678", "ILSVRC2012_test_00000692"]
    for image_name in image_list:
        image = imread("./data/{}.jpg".format(image_name))
        annotated_image = imread("./data/{}.bmp".format(image_name))
        annotated_image = annotated_image / 255

        output = crf_inference2(image, annotated_image)

        output = np.asarray(output * 255, dtype=np.uint8)
        imsave("./result/{}_1.bmp".format(image_name), output, check_contrast=False)
        pass
    pass


def demo_3(image_dir, anno_dir, anno_name, result_dir):
    Tools.new_dir(result_dir)

    img_name_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    img_name_list = sorted(img_name_list)
    lbl_name_list = [os.path.join(anno_dir, '{}_{}.bmp'.format(os.path.splitext(
        os.path.basename(img_path))[0], anno_name)) for img_path in img_name_list]
    for i, (img_name, lbl_name) in enumerate(zip(img_name_list, lbl_name_list)):
        if i < 100:
            name = os.path.splitext(os.path.basename(img_name))[0]
            Tools.print("{}/{}, {}".format(i, len(img_name_list), name))

            image = imread(img_name)
            imsave("{}/{}.jpg".format(result_dir, name), image, check_contrast=False)

            annotated_image = imread(lbl_name)
            annotated_image = annotated_image / 255
            imsave("{}/{}_1.bmp".format(result_dir, name),
                   np.asarray(annotated_image * 255, dtype=np.uint8), check_contrast=False)

            output = crf_inference2(image, annotated_image, normalization=dcrf.NO_NORMALIZATION)
            imsave("{}/{}_2.bmp".format(result_dir, name),
                   np.asarray(output * 255, dtype=np.uint8), check_contrast=False)

            annotated_image[output > 0.5] = 1.0
            imsave("{}/{}_3.bmp".format(result_dir, name),
                   np.asarray(annotated_image * 255, dtype=np.uint8), check_contrast=False)

            output = crf_inference2(image, annotated_image, normalization=dcrf.NORMALIZE_SYMMETRIC)
            annotated_image[output > 0.5] = 1.0
            imsave("{}/{}_4.bmp".format(result_dir, name),
                   np.asarray(annotated_image * 255, dtype=np.uint8), check_contrast=False)
        pass
    pass


if __name__ == '__main__':
    # his_dir = "CAM_123_SOD_224_256_cam_up_norm_C123_crf_History"
    his_dir = "CAM_123_SOD_224_256_cam_up_norm_C123_crf_Filter_History_DieDai3"
    demo_3(image_dir="/media/ubuntu/4T/ALISURE/Data/DUTS/DUTS-TR/DUTS-TR-Image",
           anno_dir="/media/ubuntu/4T/ALISURE/USOD/BASNetTemp/his/{}/1".format(his_dir),
           anno_name="cam_up_norm_C123_crf",
           result_dir="/media/ubuntu/4T/ALISURE/USOD/BASNetTemp/his/{}/1_crf".format(his_dir))
    pass
