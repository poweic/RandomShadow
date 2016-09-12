#!/usr/bin/python
import cv2
import PIL.Image
import numpy as np
from scipy import ndimage
from numpy.random import rand
from scipy.ndimage.filters import gaussian_filter

def main():

    scene = np.array(PIL.Image.open("img_raw00181.jpg"), dtype=np.float32)

    tree = extract_tree_silhouette('aachen_000001_000019_leftImg8bit.png', 'aachen_000001_000019_gtFine_labelIds.png');

    for i in range(100):
        shadow = gen_shadow(tree, scene.shape[:2])
        result = (scene * shadow[..., None]).astype(np.uint8)
        PIL.Image.fromarray(result).save("results/a{:04d}.jpg".format(i))
    
    pass

# treeId=21 is defined in Cityscapes dataset
def extract_tree_silhouette(raw_path, label_path, treeId=21, nStd=0.2):
    scene = np.array(PIL.Image.open(raw_path))
    label = np.array(PIL.Image.open(label_path))

    mask = (label == 21)
    nz_0 = np.sum(mask, axis=0).nonzero()[0]
    nz_1 = np.sum(mask, axis=1).nonzero()[0]
    x_min, x_max, y_min, y_max = nz_0[0], nz_0[-1], nz_1[0], nz_1[-1]
    print x_min, x_max, y_min, y_max

    scene = scene[y_min:y_max, x_min:x_max, :]
    mask = mask[y_min:y_max, x_min:x_max]
    PIL.Image.fromarray(mask.astype(np.uint8) * 255).save("labeled.png")

    # Calculate RGB statistics
    rgbs = scene[mask, :]
    mean = np.mean(rgbs, axis=0)
    std = np.std(rgbs, axis=0)

    nStd = nStd + rand() * 0.3

    for i in [0, 1, 2]:
        mask &= abs(scene[..., i] - mean[i]) < nStd * std[i]

    return mask

def rotateAndScale(img, deg):
    (oldY, oldX) = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=deg, scale=1) #rotate about center of image.

    # include this if you want to prevent corners being cut off
    r = np.deg2rad(deg)
    newX,newY = (abs(np.sin(r)*oldY) + abs(np.cos(r)*oldX),abs(np.sin(r)*oldX) + abs(np.cos(r)*oldY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg

def gen_shadow(tree, bg_size, min_illumination=0.6, min_dist=3, max_dist=15):

    H, W = bg_size
    
    # Random rotate/direction
    deg_range = 30
    direction = np.sign(rand() - 0.5)
    deg = (90 + (rand() - 0.5) * deg_range) + direction
    rotated_tree = rotateAndScale(tree.astype(np.uint8) * 255, deg)
    # PIL.Image.fromarray(rotated_tree).save("rotated_tree.png")

    # Random crop
    h, w = rotated_tree.shape
    crop_w = int(w * np.clip(rand(), 0.2, 0.8))
    crop_h = int(h * np.clip(rand(), 0.2, 0.8))

    x0 = int(rand() * (h - crop_h - 1) * 0.99)
    y0 = int(rand() * (w - crop_w - 1) * 0.99)

    cropped = rotated_tree[x0:x0+crop_h, y0:y0+crop_w]

    nz_0 = np.sum(cropped, axis=0).nonzero()[0]
    nz_1 = np.sum(cropped, axis=1).nonzero()[0]
    # print "cropped.shape = {}, nz_0 = {}, nz_1 = {}".format(cropped.shape, nz_0, nz_1)
    if len(nz_0) != 0:
        cropped = cropped[:, nz_0[0]:nz_0[-1]]
    if len(nz_1) != 0:
        cropped = cropped[nz_1[0]:nz_1[-1], :]

    pad_width = int(min(cropped.shape) * 0.15)
    cropped = np.pad(cropped, pad_width=pad_width, mode='constant')

    # Random light source (intensity / distance)
    illumination = rand() * (1 - min_illumination) + min_illumination
    distance = rand() * (max_dist - min_dist) + min_dist

    resized = cv2.resize(cropped, (W, int(H * 0.6)))
    blurred = gaussian_filter(resized, sigma=distance).astype(np.float32) / 255.

    shadow = np.ones((H, W), dtype=np.float32)
    shadow[H - blurred.shape[0]:, :] = 1.0 - blurred * illumination

    return shadow

main()
