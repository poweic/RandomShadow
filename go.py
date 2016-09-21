#!/usr/bin/python
import PIL.Image
import numpy as np
from random_shadow import RandomShadowGenerator

def main():

    scene = np.array(PIL.Image.open("img_raw00181.jpg"), dtype=np.float32)

    RSG = RandomShadowGenerator()

    for i in range(100):
        shadow = RSG(scene.shape[:2])
        result = (scene * shadow).astype(np.uint8)
        PIL.Image.fromarray(result).save("results-2/a{:04d}.jpg".format(i))

main()
