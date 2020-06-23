import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def crf_inference(img, probs, t=10, label_num=2):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, label_num)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((label_num, h, w))


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
