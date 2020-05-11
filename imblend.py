import cv2 as cv
import numpy as np
import scipy as sp
import scipy.signal
import scipy.ndimage
import warnings


# Global filter
guassianFilter = np.array([[0.0025, 0.0125, 0.02  , 0.0125, 0.0025],
                            [0.0125, 0.0625, 0.1   , 0.0625, 0.0125],
                            [0.02  , 0.1   , 0.16  , 0.1   , 0.02  ],
                            [0.0125, 0.0625, 0.1   , 0.0625, 0.0125],
                            [0.0025, 0.0125, 0.02  , 0.0125, 0.0025]])


def convolveAndDownsample(img):
    """
    Used to build a Guassian pyramid. Convolves image with guassianFilter
    and downsamples by 2
    """
    # Select every other pixel from G
    G = sp.signal.convolve2d(img, guassianFilter, 'same')
    return G[::2, ::2]


def buildGuassianPyramid(img, size):
    """
    Construct size # of blurred and downsampled images of img
    """
    G = img
    pyramid = [G]
    for i in range(size):
        G = convolveAndDownsample(G)
        pyramid.append(G)

    return pyramid


def buildLaplacianPyramid(guassianPyramid):
    """
    Construct edge detecting Laplacian pyramid using a guassian Pyramid
    L_i = G_i - (convolve(K, G_i))
    """
    pyramid = []
    for i in range(len(guassianPyramid)-1):
        Gi = upsample(guassianPyramid[i+1])
        G = guassianPyramid[i]
        r, c = G.shape[:2]
        L = G - Gi[:r, :c]
        pyramid.append(L)

    pyramid.append(guassianPyramid[-1])
    return pyramid


def upsample(img):
    """
    Upsamples an image by a factor of 2
    """

    filtered = sp.signal.convolve2d(img, guassianFilter, 'same')
    i, j = img.shape
    upsampled = np.zeros((i*2, j*2))
    for r in range(i):
        upsampled[2 * r, ::2] = img[r, ::]
    for c in range(j):
        upsampled[::2, 2 * c] = img[::, c]

    # Need to raise values of upsampled image by 4 (1px in original -> 4px in upsampled)
    return 4 * sp.signal.convolve2d(upsampled, guassianFilter, 'same')



def reconstructImageFromPyramid(pyramid):
    """
    Sum over the images in the pyramid, upsampling at each level.
    Starts from the bottom. Upsamples and adds to the second last, and repeats
    """
    for i in range(len(pyramid)-1, 0, -1):
        r, c = pyramid[i - 1].shape[:2]
        pyramid[i - 1] += upsample(pyramid[i])[:r, :c]

    return pyramid[0]


def imBlend(src, mask, target, size):
    """
    @param src: greyscale source image
    @param mask: binary mask
    @param target: greyscale target image
    @return: grayscale blended image
    """

    # Build guassian pyramids for all inputs
    guass_src = buildGuassianPyramid(src, size)
    guass_tar = buildGuassianPyramid(target, size)
    guass_mask = buildGuassianPyramid(mask, size)

    # Build laplacian pyramid for target and source
    lapl_src = buildLaplacianPyramid(guass_src)
    lapl_tar = buildLaplacianPyramid(guass_tar)

    blended_pyramid = []
    for i in range(size):
        m, s, t = guass_mask[i], lapl_src[i], lapl_tar[i]
        img_blend = np.zeros(m.shape)
        img_blend = m*s + (1.-m)*t
        blended_pyramid.append(img_blend)

    return np.clip(reconstructImageFromPyramid(blended_pyramid), 0, 1)

def dummyPlace(src, mask, target):
    binMask = np.where(mask > 0.5, 1, 0)
    return np.where(binMask, src, target)
