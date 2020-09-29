import numpy as np 
import os
from PIL import Image, ImageOps
from random import random

image_dir = '../lfw-deepfunneled'

def display_image(image, save_dir = None):
    if image.shape[0] == 1:
        image = image[0]
    image = Image.fromarray(np.clip(image, 0, 255).astype('uint8'))
    image.show()
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        count = 0
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                count += 1
        image.save(os.path.join(root, str(count) + '.png'))

def unpreprocess_image(img):
    img = (img + 1.0) * 127.5
    img = img.astype('uint8')
    return img

def load_image(filename, size = (250,250)):
    img = Image.open(filename).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    # if random() > 0.5:
    #     img = ImageOps.mirror(img)
    img = np.array(img).astype('float32')
    img = (img / 127.5) - 1.0
    return img

if __name__ == '__main__':
    pass