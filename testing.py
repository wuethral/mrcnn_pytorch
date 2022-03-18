import torch
import os
import cv2
from PIL import Image
from torchvision import transforms as T
import numpy as np


def get_prediction(class_names, prediction_threshold, path_to_model, img_path):
  '''Method to get mask, bounding boxes, prediction classes and prediction scores'''

  img = Image.open(img_path) # This is for local images
  transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
  img = transform(img)
  img = img.cuda() # Only if GPU, otherwise comment this line

  model = torch.load(path_to_model)
  model.eval()
  model = model.cuda()
  img = img.cuda()
  pred = model([img])

  # Now we need to extract the bounding boxes and masks
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  print('predscores:', pred_score)
  print(pred[0]['labels'])

  pred_t = [pred_score.index(x) for x in pred_score if x > prediction_threshold][-1]
  masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

  pred_class = [class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]

  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  masks = masks[:pred_t+1]

  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]

  return masks, pred_boxes, pred_class, pred_score


def random_color_masks(image, mask_color):
  '''Method to change the color of the mask'''

  # Accessing the r, g and b values of the mask
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  # Changing the mask's r,g and b values
  r[image==1], g[image==1], b[image==1] = mask_color
  # Stacking the rgb values
  colored_mask = np.stack([r,g,b], axis=2)

  return colored_mask


def instance_segmentation(class_names, prediction_threshold, path_to_model, img_path, mask_color, rect_th,
                          text_size, text_th):
  '''Method to draw masks, bounding boxes and text on image'''

  masks, boxes, pred_cls, pred_score = get_prediction(class_names, prediction_threshold, path_to_model,img_path)
  img = cv2.imread(img_path)

  for i in range(len(masks)):
    rgb_mask = random_color_masks(masks[i], mask_color)
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
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

  return img, rgb_mask


if __name__ == '__main__':

  # Path to the images for the visual prediction of the masks
  image_list = os.listdir('for_visual_evaluation/images')

  # Looping through all images
  for image_name in image_list:
    # Names of the classes. The position in the list represents the label. So '0' is label 0 (background),
    # 'hand' is label 1, and so on...
    class_names = ['0', 'hand', 'screw', 'screwdriver', 'spire2', 'angle', 'spire1', 'pliers']
    # Threshold of prediction certainty, determining if mask is created or not. If prediction certainty is larger than
    # threshold, masks are created.
    prediction_threshold = 0.9
    # Path to trained model
    path_to_model = 'trained_models/trainedmodel_9.pt'
    # Path to image
    image_path = 'for_visual_evaluation/images/' + image_name
    # Color of mask
    mask_color = (0,0,255)
    # Thickness of bounding box, text size and text thickness
    rect_th = 3
    text_size = 1
    text_th = 3
    # Getting the predicted images, classes and masks
    img, rgb_mask = instance_segmentation(class_names, prediction_threshold, path_to_model, image_path, mask_color,
                                          rect_th, text_size, text_th)
    # Save the predicted mask on top of image
    prediction_mask_path = 'for_visual_evaluation/predicted_masks/image_masks/' + image_name
    cv2.imwrite(prediction_mask_path, img)
    # Save the predicted mask (without image)
    prediction_mask_path_2 = 'for_visual_evaluation/predicted_masks/masks/' + image_name
    cv2.imwrite(prediction_mask_path_2, rgb_mask)