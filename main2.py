from PIL import Image
import os
import numpy as np
''' 
image = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
width, height = image.size
count = 0
for i in range(width):
    for j in range(height):
        print(image.getpixel((i,j)))
    
'''

idx = 9
imgs = list(sorted(os.listdir(os.path.join("PennFudanPed/PNGImages"))))
masks = list(sorted(os.listdir(os.path.join("PennFudanPed/PedMasks"))))


img_path = os.path.join("PennFudanPed/PNGImages", imgs[idx])
mask_path = os.path.join("PennFudanPed/PedMasks",masks[idx])
img = Image.open(img_path).convert("RGB")
# print(img.height, img.width)
# note that we haven't converted the mask to RGB,
# because each color corresponds to a different instance
# with 0 being background
mask = Image.open(mask_path)
# convert the PIL Image into a numpy array
mask = np.array(mask)
# instances are encoded as different colors
obj_ids = np.unique(mask)
# first id is the background, so remove it
#print(mask)
obj_ids = obj_ids[1:]
# split the color-encoded mask into a set
# of binary masks
masks = mask == obj_ids[:, None, None]
print(masks)
# get bounding box coordinates for each mask
num_objs = len(obj_ids)