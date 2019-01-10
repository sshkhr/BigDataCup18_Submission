import pandas as pd
from tqdm import tqdm
import skimage
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

import os
import re
import random
import glob

directories = ["Adachi","Chiba","Ichihara","Muroran","Nagakute","Numazu","Sumida"]

basic_dir = "../../road_damage_dataset/"

from os import listdir
from os.path import isfile, join
test_files = []
for d in directories:
    mypath = basic_dir + d + "/JPEGImages/"
    onlyfiles = glob.glob(mypath+"*.jpg") 
    for f in onlyfiles:
        if "test" in f:
            test_files.append(f)

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
    RPN_ANCHOR_SCALES = (32,64, 128, 256)  # anchor side in pixels
    
    MEAN_PIXEL = np.array([109.45249984771704, 117.70290089791496, 116.77272288808695])
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    USE_MINI_MASK = True
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1627

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
        return mask.astype(np.bool), class_ids.astype(np.int32)


class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = "mask_rcnn_shapes_aug_epoch_6.h5"
model.load_weights(model_path, by_name=True)  ## set model path

df = pd.DataFrame(columns=["ImageId","PredictionString"])

for im in tqdm(test_files):
    im_id = re.sub("\\\\","",im.split("JPEGImages")[1])  ## change this according to ubuntu
    im_id = re.sub(".jpg","",im_id)
    img = skimage.io.imread(im)
    m = model.detect([img])
    final_list = []
    class_ids= m[0]["class_ids"]
    bboxes = m[0]["rois"]
    for i in range(len(class_ids)):
        final_list.append(class_ids[i])
        bbox = bboxes[i]
        bbox = [bbox[1],bbox[0],bbox[3],bbox[2]]
        for j in range(4):
            final_list.append(int(((bbox[j])/512)*600))
    pred = " ".join(str(e) for e in final_list)        
    temp_df = pd.DataFrame([im_id],columns=["ImageId"])
    temp_df["PredictionString"] = pred
    df = pd.concat([df,temp_df])
    df.index = range(df.shape[0])
    
df['ImageId'] = df['ImageId'].str.replace(r"/", '')
df['ImageId'] = df['ImageId'].astype(str)+'.jpg'

df.sort_values(['ImageId'], inplace=True)
df.reset_index(drop=True, inplace=True)


df.to_csv("submission_epoch_6_with_aug_half_lr_anneal.csv",index = False, header = False)