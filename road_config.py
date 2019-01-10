import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import config

class RoadConfig(Config):
    """
    Mask RCNN configuration for IEEE BigData 2018 Road Damage Challenge
    """

    # Give the configuration a recognizable name
    NAME = "road_damage_detection"

    # Random crop larger images
    CROP = True
    CROP_SHAPE = np.array([256, 256, 3])

    # Whether to use image augmentation in training mode
    AUGMENT = True

    # Whether to use image scaling and rotations in training mode
    SCALE = True

    # Optimizer, default is 'SGD'
    OPTIMIZER = 'ADAM'

    # Train on 1 GPU and 2 images per GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # background + 

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Backbone encoder architecture
    BACKBONE = 'resnet101'

    # Using smaller anchors because nuclei are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [.5, 1, 2]

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320  #

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 2048

    # Number of ROIs per image to feed to classifier/mask heads
    TRAIN_ROIS_PER_IMAGE = 512

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.7
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 20

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 20

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.75

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3  # 0.3

    # Threshold number for mask binarization, only used in inference mode
    DETECTION_MASK_THRESHOLD = 0.35


# Root directory of the project
ROOT_DIR = '../data/'

# Directory to save logs and trained model
MODEL_DIR = '../data/logs'


def load_img(fname, color='RGB'):

    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if color == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)

    return img  