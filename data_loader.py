import numpy as np
import tensorflow as tf 
import os
import image_utils
from random import sample
from training_params import batch_size
import training_params


count_by_folders = {}
image_paths = {}
for root, dir, files in os.walk(image_utils.image_dir):
    count = 0
    for file in files:
        if file.split('.')[-1] == 'jpg':
            count += 1
            if root in image_paths:
                    image_paths[root].append(file)
            else:
                    image_paths[root] = [file]
            # display_image(unpreprocess_image(load_image(os.path.join(root, file))))
    count_by_folders[root] = count

singleton_folders = []
multiple_folders = []
for folder in count_by_folders:
        if count_by_folders[folder] == 1:
                singleton_folders.append(folder)
        elif count_by_folders[folder] > 1:
                multiple_folders.append(folder)
        else:
                print("EMPTY FOLDER: ", folder)


def get_batch():
        number_of_pairs = ((batch_size + 1) // 2) // 2
        singles = sample(singleton_folders, k=batch_size - number_of_pairs * 2)
        multiples = sample(multiple_folders, k=number_of_pairs)
        batch = []
        for m in multiples:
                ms = sample(image_paths[m], k=2)
                # print(ms)
                for filename in ms:
                        batch.append(image_utils.load_image(os.path.join(m, filename), training_params.shape))
        for s in singles:
                batch.append(image_utils.load_image(os.path.join(s, image_paths[s][0]), training_params.shape))
        return np.array(batch)


if __name__ == '__main__':
        batch = get_batch()
        print(batch.shape)
        image_utils.display_image(image_utils.unpreprocess_image(batch[0]))
        image_utils.display_image(image_utils.unpreprocess_image(batch[1]))
