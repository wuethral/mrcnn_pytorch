import numpy as np
import os
from PIL import Image

for image_name in os.listdir('PennFudanPed/PedMasks'):
    path = 'PennFudanPed/PedMasks/' + image_name
    mask = Image.open(path)
    print(image_name)
    print(np.unique(mask))