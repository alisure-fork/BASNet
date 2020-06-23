import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def crf(original_image, annotated_image, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    if len(annotated_image.shape) < 3:
        annotated_image = gray2rgb(annotated_image)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:, :, 0] + (annotated_image[:, :, 1] << 8) + (annotated_image[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))

    print("No of labels in the Image are {}".format(n_labels))

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        pass

    # Run Inference for 5 steps
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    return MAP.reshape(original_image.shape)


# 1
image = imread("orginal_1.png")
annotated_image1 = imread("annotation1_fcn16.png")
annotated_image2 = imread("annotation1_fcn8.png")
plt.imshow(image)
plt.imshow(annotated_image1)
plt.imshow(annotated_image2)

output1 = crf(image,annotated_image1,"crf1_fcn16.png")
output2 = crf(image,annotated_image2,"crf1_fcn8.png")
output2 = rgb2gray(output2)
output1 = rgb2gray(output1)

plt.subplot(1,2,1)
plt.imshow(output1)
plt.subplot(1,2,2)
plt.imshow(output2)


# 2
image = imread("orginal3.png")
annotated_image1 = imread("annotation3_fcn16.png")
annotated_image2 = imread("annotation3_fcn8.png")
plt.imshow(image)
plt.imshow(annotated_image1)
plt.imshow(annotated_image2)

output1 = crf(image,annotated_image1,"crf1_fcn16.png")
output2 = crf(image,annotated_image2,"crf1_fcn8.png")

output2 = rgb2gray(output2)
output1 = rgb2gray(output1)
plt.subplot(1,2,1)
plt.imshow(output1)
plt.subplot(1,2,2)
plt.imshow(output2)


# 3
image = imread("orginal4.png")
annotated_image1 = imread("annotation4_fcn16.png")
annotated_image2 = imread("annotation4_fcn8.png")

plt.imshow(image)
plt.imshow(annotated_image1)
plt.imshow(annotated_image2)

output1 = crf(image, annotated_image1, "crf1_fcn16.png")
output2 = crf(image, annotated_image2, "crf1_fcn8.png")
output2 = rgb2gray(output2)
output1 = rgb2gray(output1)
plt.subplot(1,2,1)
plt.imshow(output1)
plt.subplot(1,2,2)
plt.imshow(output2)
