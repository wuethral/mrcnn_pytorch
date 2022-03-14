import torch
import os
import cv2


#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#model.eval()

#model = model.cuda()
''' 
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',A
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''

COCO_INSTANCE_CATEGORY_NAMES = ['0', 'hand', 'screw', 'screwdriver', 'spire2', 'angle', 'spire1', 'pliers']
# I will link the notebook in the description
# You can copy the class names from the description
# or the notebook
len(COCO_INSTANCE_CATEGORY_NAMES) # 91 classes including background

from PIL import Image
from torchvision import transforms as T
import numpy as np
import requests
from io import BytesIO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# the io and requests libraries are just for loading images from URLS


def get_prediction(img_path, threshold=0.5, url=False):
  if url: # We have to request the image
    response = requests.get(img_path)
    img = Image.open(BytesIO(response.content))
  else:
    img = Image.open(img_path) # This is for local images
  transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
  img = transform(img)
  img = img.cuda() # Only if GPU, otherwise comment this line

  model = torch.load('trained_models/trained_model_0')
  #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  model.eval()
  model = model.cuda()
  img = img.cuda()
  pred = model([img])

  # Now we need to extract the bounding boxes and masks
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  print('predscores:', pred_score)
  print(pred[0]['labels'])

  prediction_threshold = 0.9
  pred_t = [pred_score.index(x) for x in pred_score if x > prediction_threshold][-1]
  masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]

  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  masks = masks[:pred_t+1]
  #if only one pred, otherwise take above


  pred_boxes = pred_boxes[:pred_t+1]
  #if only one pred, otherwise take above
  #pred_boxes = pred_boxes[0]
  pred_class = pred_class[:pred_t+1]
  #if only one pred, otherwise take above
  #pred_class = pred_class[0]
  return masks, pred_boxes, pred_class, pred_score


from urllib.request import urlopen
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
  resp = urlopen(url) # We want to convert URL to cv2 image here, so we can draw the mask and bounding boxes
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, readFlag)
  return image

import random

def random_color_masks(image):
  # I will copy a list of colors here
  colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image==1], g[image==1], b[image==1] = (0,0,255)  #colors[random.randrange(0, 10)]
  colored_mask = np.stack([r,g,b], axis=2)
  return colored_mask

def instance_segmentation(img_path, threshold=0.5, rect_th=3,
                          text_size=1, text_th=3, url=False):
  masks, boxes, pred_cls, pred_score = get_prediction(img_path, threshold=threshold, url=url)
  if url:
    img = url_to_image(img_path) # If we have a url image
  else: # Local image
    img = cv2.imread(img_path)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR
  for i in range(len(masks)):
    rgb_mask = random_color_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #print(img[4][4])
    boxes01 = int(boxes[i][0][0])
    boxes02 = int(boxes[i][0][1])
    boxes1 = (boxes01, boxes02)
    boxes11 = int(boxes[i][1][0])
    boxes12 = int(boxes[i][1][1])
    boxes2 = (boxes11, boxes12)
    boxes3 = (boxes11, boxes12-60)
    cv2.rectangle(img, boxes1, boxes2, color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img, pred_cls[i], boxes3, cv2.FONT_HERSHEY_SIMPLEX, text_size*2, (0, 255, 0), thickness=int(text_th*1.5))
    cv2.putText(img, str(pred_score[i]), boxes2, cv2.FONT_HERSHEY_SIMPLEX, text_size*2, (0, 255, 0), thickness=int(text_th*1.5))
  return img, pred_cls, masks[i]

# I will link the following image in the description
# We are going to try the function out, first we will download an image
#!wget https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg -O car.jpg
img_nr = 1
''' 
for i in range(334):
  path_in = 'angles/image_' + str(img_nr) + '.png'
  img, pred_classes, masks = instance_segmentation(path_in, rect_th=5, text_th=4)
  path_out = 'angle_masks/angle_mask_' + str(img_nr) + '.png'

  cv2.imwrite(path_out, img)
  img_nr += 1
'''
#image_list = os.listdir('D:/Operation_images/video_2(green_screen)')
image_list = os.listdir('D:/Trained_models_new_augmentation/Model_6/Validation_R_but_green_bg_from_NR/DataSet/images')
''' 
for i in range(1, len(image_list), 1):

  if i % 10 == 0:
    #image_path = 'D:/Operation_images/video_2(green_screen)/image_' + str(i) + '.png'
    image_path = 'D:/Trained_models_new_augmentation/Model_6/Testing_images/image_' + str(i) + '.png'
    img, pred_classes, masks = instance_segmentation(image_path, rect_th=5, text_th=4)
    prediction_mask_path = 'D:Trained_models_new_augmentation/Model_6/predicted_masks_testing_images/mask_' + str(i) + '.png'
    cv2.imwrite(prediction_mask_path, img)
'''
for image_name in image_list:
  image_path = 'D:/Trained_models_new_augmentation/Model_6/Validation_R_but_green_bg_from_NR/DataSet/images/' + image_name
  img, pred_classes, masks = instance_segmentation(image_path, rect_th=5, text_th=4)
  prediction_mask_path = 'D:/Trained_models_new_augmentation/Model_6/Validation_R_but_green_bg_from_NR/visual_evaluation/' + image_name
  cv2.imwrite(prediction_mask_path, img)