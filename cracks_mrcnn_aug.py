import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


from os import listdir
from os.path import isfile, join

import os

import random

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

directories = ["Adachi","Chiba","Ichihara","Muroran","Nagakute","Numazu","Sumida"]

basic_dir = "../../road_damage_dataset/"

annotation_files = [] #path of all the annotation files

for d in directories:
    annotation_directory = basic_dir+ d + "/Annotations/" 
    onlyfiles = [f for f in listdir(annotation_directory) if isfile(join(annotation_directory, f))]
    onlyfiles = [annotation_directory+s for s in onlyfiles]
    annotation_files.append(onlyfiles)


train_annotation_files = []
valid_annotation_files = []
for i in range(len(annotation_files)):
    l = annotation_files[i]
    num  = len(l)
    train = int(0.9 * num)
    valid = num - train
    tr_files = random.sample(l, train)
    val_files = list(set(l) - set(tr_files))
    train_annotation_files.append(tr_files)
    valid_annotation_files.append(val_files)
    
train_annotation_files = [item for sublist in train_annotation_files for item in sublist]
valid_annotation_files = [item for sublist in valid_annotation_files for item in sublist]
train_img_files = [re.sub(".xml",".jpg",s) for s in train_annotation_files]
valid_img_files = [re.sub(".xml",".jpg",s) for s in valid_annotation_files]

train_img_files = [re.sub("Annotations","JPEGImages",s) for s in train_img_files]
valid_img_files = [re.sub("Annotations","JPEGImages",s) for s in valid_img_files]

for n in range(len(train_annotation_files)):
    tex = train_annotation_files[n]
    tex = re.sub(".xml",".xml.npy",tex)
    tex = re.sub("road_damage_dataset/","road_damage_dataset/masked_dict/",tex)
    train_annotation_files[n] = tex
    
for n in range(len(valid_annotation_files)):
    tex = valid_annotation_files[n]
    tex = re.sub(".xml",".xml.npy",tex)
    tex = re.sub("road_damage_dataset/","road_damage_dataset/masked_dict/",tex)
    valid_annotation_files[n] = tex
    
dir_to_masks = "../../road_damage_dataset/masked_dict/"

for t in train_img_files:
    if "train_Numazu_01264" in t:
        ind = train_img_files.index(t)
        train_img_files = np.delete(np.array(train_img_files),ind)
    else:
        pass

train_img_files =  list(train_img_files)

for t in train_annotation_files:
    if "train_Numazu_01264" in t:
        ind = train_annotation_files.index(t)
        train_annotation_files = np.delete(np.array(train_annotation_files),ind)
    else:
        pass

train_annotation_files = list(train_annotation_files)





class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cracks"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # background + 9 types of cracks

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32,64, 128, 256, 512)  # anchor side in pixels
    
    MEAN_PIXEL = np.array([109.45249984771704, 117.70290089791496, 116.77272288808695])
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    USE_MINI_MASK = True
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1850

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 182
    RPN_ANCHOR_RATIOS = [.5, 1, 2]

config = ShapesConfig()

class cracksDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_crack(self,subset):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        self.add_class("cracks", 1, "D00")
        self.add_class("cracks", 2, "D01")
        self.add_class("cracks", 3, "D10")
        self.add_class("cracks", 4, "D11")
        self.add_class("cracks", 5, "D20")
        self.add_class("cracks", 6, "D40")
        self.add_class("cracks", 7, "D43")
        self.add_class("cracks", 8, "D44")
        self.add_class("cracks", 9, "D30")
        
        assert subset in ["train", "val"]
        if subset == "val":
            image_ids = valid_img_files
        elif subset == "train":
            image_ids = train_img_files
        height, width = (512,512)
        
        for image_id in image_ids:
            temp_id = re.sub(".jpg",".xml",image_id.split("road_damage_dataset/")[1])
            temp_id = re.sub("JPEGImages","Annotations",temp_id)
            self.add_image(
                "cracks",
                image_id=temp_id,
                path=image_id)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cracks":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)            

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        path = info["id"]
        real_path = dir_to_masks+path+".npy"
        mask_orig = np.load(real_path).item()
        class_ids = mask_orig[path][1]
        for cid in range(len(class_ids)):
            cat = class_ids[cid]
            if cat == "D00":
                class_ids[cid] = 1
            elif cat == "D01":
                class_ids[cid] = 2
            elif cat == "D10":
                class_ids[cid] = 3
            elif cat == "D11":
                class_ids[cid] = 4
            elif cat == "D20":
                class_ids[cid] = 5
            elif cat == "D40":
                class_ids[cid] = 6
            elif cat == "D43":
                class_ids[cid] = 7
            elif cat == "D44":
                class_ids[cid] = 8
            elif cat == "D30":
                class_ids[cid] = 9
                
                
        class_ids = np.array(class_ids)                
        mask = np.array(mask_orig[path][0])

#         if (len(class_ids) ==1):
#             mask = m[:,:,0]
#             mask = np.reshape(mask,(600,600,1))
#             mask = np.array(mask)
#         else:
#             mask = np.concatenate((np.reshape(m[0],(600,600,1)), np.reshape(m[1],(600,600,1))),axis =-1)
#             for i in range(2,len(class_ids)):
#                 mask = np.concatenate((mask,np.reshape(m[i],(600,600,1))),axis = -1)
            
#             mask = np.array(mask)
        
        return mask.astype(np.bool), class_ids.astype(np.int32)

    
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
                          
init_with = "last"  # imagenet, coco, or last

#if init_with == "imagenet":
#    model.load_weights(model.get_imagenet_weights(), by_name=True)
#elif init_with == "coco":
#    # Load weights trained on MS COCO, but skip layers that
#    # are different due to the different number of classes
#    # See README for instructions to download the COCO weights
#    model.load_weights(COCO_MODEL_PATH, by_name=True,
#                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
#                                "mrcnn_bbox", "mrcnn_mask"])
#elif init_with == "last":
#    # Load the last model you trained and continue training
#    model.load_weights(model.find_last()[1], by_name=True)
    
#model.load_weights("C:/Users/Janpreet/data_science_experiments/Mask_RCNN/logs/cracks20181104T1120/mask_rcnn_cracks_0006.h5", by_name=True)

# Training dataset
dataset_train = cracksDataset()

dataset_train.load_crack(subset="train")

dataset_train.prepare()

# Training dataset
dataset_val = cracksDataset()

dataset_val.load_crack(subset="val")

dataset_val.prepare()

import warnings
warnings.filterwarnings("ignore")

import imgaug

augmentation = imgaug.augmenters.Fliplr(.5)

#model.train(dataset_train, dataset_val, 
#            learning_rate=config.LEARNING_RATE, 
#           epochs=2, 
#          layers='heads',augmentation=augmentation)


#model.train(dataset_train, dataset_val, 
#            learning_rate=0.0001, 
#            epochs=2, 
#            layers='all',augmentation=augmentation)            

            
model.train(dataset_train, dataset_val, 
            learning_rate=0.001, 
            epochs=3, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_3.h5")

model.train(dataset_train, dataset_val, 
            learning_rate=0.0001, 
            epochs=6, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_6.h5")

model.train(dataset_train, dataset_val, 
            learning_rate=0.00001, 
            epochs=10, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_10.h5")

model.train(dataset_train, dataset_val, 
            learning_rate=0.000001, 
            epochs=15, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_15.h5")