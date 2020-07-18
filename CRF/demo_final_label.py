import sys
import cv2
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from alisuretool.Tools import Tools
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def crf_softmax(image, annotation, t=10, n_label=2):
    image = np.array(image)
    annotation = np.array(annotation)
    annotation_shape = annotation.shape

    if np.max(annotation) > 1:
        annotation = annotation / 255
    annotation = np.expand_dims(annotation, axis=0)
    annotation = np.concatenate([annotation, 1 - annotation], axis=0)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_label)
    unary = unary_from_softmax(annotation)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(image), compat=10)
    q = d.inference(t)
    map_result = np.argmin(q, axis=0)
    result = map_result.reshape(annotation_shape)
    return result


def crf_label(image, annotation, t=10, n_label=2):
    image = np.array(image)
    annotation = np.array(annotation)

    a, b = (0.6, 0.2)
    if np.max(annotation) > 1:
        a, b =a * 255, b * 255
        pass
    label_extend = np.zeros_like(annotation)
    label_extend[annotation > a] = 2
    label_extend[annotation < b] = 1

    _, label = np.unique(label_extend, return_inverse=True)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_label)
    u = unary_from_labels(label, n_label, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image,
                           compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = d.inference(t)
    map_result = np.argmax(q, axis=0)
    result = map_result.reshape(annotation.shape)
    return label_extend, result


image_list = ["ILSVRC2012_test_00000340", "ILSVRC2012_test_00000363",
              "ILSVRC2012_test_00000450", "ILSVRC2012_test_00000678", "ILSVRC2012_test_00000692"]
for name in image_list:
    image_path = "./data/{}.jpg".format(name)
    annotation_path = "./data/{}.bmp".format(name)
    edge_path = "./data/{}_edge.bmp".format(name)

    im = Image.open(image_path).convert("RGB")
    annotation = Image.open(annotation_path).convert("L")
    edge = Image.open(edge_path).convert("L")

    im.save(Tools.new_dir("./result2/{}_0.jpg".format(name)))
    annotation.save(Tools.new_dir("./result2/{}_1.bmp".format(name)))
    edge.save(Tools.new_dir("./result2/{}_2.bmp".format(name)))

    result_softmax = crf_softmax(np.array(im), np.array(annotation))
    Image.fromarray(np.asarray(result_softmax * 255, dtype=np.uint8)).save("./result2/{}_3.bmp".format(name))

    label1, result_label1 = crf_label(np.array(im), np.array(annotation))
    Image.fromarray(np.asarray(result_label1 * 255, dtype=np.uint8)).save("./result2/{}_4.bmp".format(name))
    Image.fromarray(np.asarray(label1 * 127, dtype=np.uint8)).save("./result2/{}_5.bmp".format(name))

    result_softmax2 = crf_softmax(np.array(im), np.array(edge))
    Image.fromarray(np.asarray(result_softmax2 * 255, dtype=np.uint8)).save("./result2/{}_6.bmp".format(name))

    # fusion = np.array(annotation, dtype=np.int32) + np.array(edge, dtype=np.int32) + result_label1 * 255
    fusion = np.array(annotation, dtype=np.int32) + result_label1 * 255
    fusion[fusion>255] = 255
    fusion = np.asarray(fusion, dtype=np.uint8)
    Image.fromarray(np.asarray(fusion, dtype=np.uint8)).save("./result2/{}_7.bmp".format(name))
    result_softmax3 = crf_softmax(np.array(im), fusion)
    Image.fromarray(np.asarray(result_softmax3 * 255, dtype=np.uint8)).save("./result2/{}_8.bmp".format(name))

    label2, result_label2 = crf_label(np.array(im), np.array(fusion))
    Image.fromarray(np.asarray(result_label2 * 255, dtype=np.uint8)).save("./result2/{}_9.bmp".format(name))
    Image.fromarray(np.asarray(label2 * 127, dtype=np.uint8)).save("./result2/{}_91.bmp".format(name))
    pass
