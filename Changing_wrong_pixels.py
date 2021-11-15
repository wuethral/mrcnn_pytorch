import numpy as np
import os
from PIL import Image
import shutil


def cleaning_row(array, i, pixel_value, width):

    new_row_image_matrix = [0] * width
    for i in range(width):
        if array[i] != 0:
            new_row_image_matrix[i] = pixel_value

    return new_row_image_matrix


def clean_this_shit(image_name,path):

    if image_name.startswith('zz_test_spire1'):
        image_to_polish = Image.open(path)
        height = image_to_polish.height
        width = image_to_polish.width
        mask_matrix = np.zeros((height, width))
        array = np.array(image_to_polish)
        #print(array)
        for i in range(height):
            new_row = cleaning_row(array[i], i, 254, width)
            mask_matrix[i, :] = new_row

        matrix_to_array = np.squeeze(np.asarray(mask_matrix))
        matrix_to_array = np.reshape(matrix_to_array, (height, width)).astype(np.uint8)
        repaired_mask = Image.fromarray(matrix_to_array)
        os.remove(path)
        repaired_mask.save(path)


for image_name in os.listdir('DataSet/Masks_all'):
    path = 'DataSet/Masks_all/' + image_name
    mask = Image.open(path)
    nr_of_unique_pixels = len(np.unique(mask))
    if nr_of_unique_pixels != 2:
        clean_this_shit(image_name, path)