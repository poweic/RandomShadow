#!/usr/bin/python
import os 
import cv2
import PIL.Image
from path import Path
import numpy as np
from numpy.random import rand
from scipy.ndimage.filters import gaussian_filter

class RandomShadowGenerator():
    def __init__(self):
        N_TREES = 46
        DIR = os.path.dirname(os.path.realpath(__file__)) + "/../samples/"

        self.trees = []
        for i in range(N_TREES):
            tree_fn = "{}/{:02d}.png".format(DIR, i)
            self.trees.append(np.array(PIL.Image.open(tree_fn)))

    def __call__(self, size):
        idx = np.random.randint(len(self.trees))
        tree = self.extract_tree_silhouette(self.trees[idx])
        shadow = self.gen_shadow(tree, size)
        return shadow[..., None]

    def extract_tree_silhouette(self, tree, nStd=0.2):
        mask = (tree[:, :, 0] == 255) & (tree[:, :, 1] == 255) & (tree[:, :, 2] == 255)
        mask = ~mask

        nz_0 = np.sum(mask, axis=0).nonzero()[0]
        nz_1 = np.sum(mask, axis=1).nonzero()[0]
        x_min, x_max, y_min, y_max = nz_0[0], nz_0[-1], nz_1[0], nz_1[-1]

        tree = tree[y_min:y_max, x_min:x_max, :]
        mask = mask[y_min:y_max, x_min:x_max]

        # Calculate RGB statistics
        rgbs = tree[mask, :]
        mean = np.mean(rgbs, axis=0)
        std = np.std(rgbs, axis=0)

        nStd = nStd + rand() * 0.8

        for i in [0, 1, 2]:
            mask &= abs(tree[..., i] - mean[i]) < nStd * std[i]

        return mask

    def gen_shadow(self, tree, bg_size, min_illumination=0.6, min_dist=3, max_dist=8):

        H, W = bg_size
        
        # Random rotate/direction
        deg_range = 30
        direction = np.sign(rand() - 0.5)
        deg = (90 + (rand() - 0.5) * deg_range) + direction
        rotated_tree = self.rotateAndScale(tree.astype(np.uint8) * 255, deg)

        # Random crop
        h, w = rotated_tree.shape
        crop_w = int(w * np.clip(rand(), 0.2, 0.8))
        crop_h = int(h * np.clip(rand(), 0.35, 0.65))

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

        # Resize shadow to be a little bit wider in width, shorter in height
        h_ratio = 0.6 + 0.1 * rand()
        w_ratio = 1.2 + 0.3 * rand()

        shadow = np.ones((H, W), dtype=np.float32)
        try:
            resized = cv2.resize(cropped, (int(W * w_ratio), int(H * h_ratio)))

            w_offset = int((resized.shape[1] - W) / 2)
            resized = resized[:, w_offset:w_offset + W]
            blurred = gaussian_filter(resized, sigma=distance).astype(np.float32) / 255.

            shadow[H - blurred.shape[0]:, :] = 1.0 - blurred * illumination
        except:
            pass

        return shadow

    def rotateAndScale(self, img, deg):
        (oldY, oldX) = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
        M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=deg, scale=1) #rotate about center of image.

        # include this if you want to prevent corners being cut off
        r = np.deg2rad(deg)
        newX,newY = (abs(np.sin(r)*oldY) + abs(np.cos(r)*oldX),abs(np.sin(r)*oldX) + abs(np.cos(r)*oldY))

        M[0,2] += (newX - oldX) / 2
        M[1,2] += (newY - oldY) / 2

        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
        return rotatedImg
