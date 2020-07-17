import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels


def crf(original_image, annotated_label):
    colors, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = len(set(labels.flat))

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
    u = unary_from_labels(labels, n_labels, gt_prob=0.8, zero_unsure=False)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = d.inference(5)
    result = np.argmax(q, axis=0)
    result = result.reshape((original_image.shape[0], original_image.shape[1]))
    return result


# 1
image = imread("./data/ILSVRC2012_test_00000340.jpg")
annotated_label = imread("./data/ILSVRC2012_test_00000340.bmp")
Image.fromarray(np.asarray(annotated_label, dtype=np.uint8)).show()

annotated_label = annotated_label / 255
annotated_label[annotated_label > 0.6] = 1
annotated_label[annotated_label < 0.4] = 0
Image.fromarray(np.asarray(annotated_label * 127, dtype=np.uint8)).show()

output1 = crf(image, annotated_label)
Image.fromarray(np.asarray(output1 * 127, dtype=np.uint8)).show()
