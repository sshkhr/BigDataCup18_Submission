import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils


class RoadDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_images(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("cracks", 1, "D00")
        self.add_class("cracks", 2, "D01")
        self.add_class("cracks", 3, "D10")
        self.add_class("cracks", 4, "D11")
        self.add_class("cracks", 5, "D20")
        self.add_class("cracks", 6, "D40")
        self.add_class("cracks", 7, "D43")
        self.add_class("cracks", 8, "D44")
        self.add_class("cracks", 9, "D30")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


"""
Training part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.
author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""

from config import *
import h5py


class KaggleDataset(utils.Dataset):
    """wrapper for loading bowl datasets
    """

    def load_shapes(self, id_list, train_path):
        """initialize the class with dataset info.
        """
        # Add classes
        self.add_class('images', 1, "nucleus")
        self.train_path = train_path

        # Add images
        for i, id_ in enumerate(id_list):
            self.add_image('images', image_id=i, path=None,
                           img_name=id_)

    def load_image(self, image_id, color):
        """Load image from directory
        """

        info = self.image_info[image_id]
        path = self.train_path + info['img_name'] + \
            '/images/' + info['img_name'] + '.png'

        img = load_img(path, color=color)

        return img

    def image_reference(self, image_id):
        """Return the images data of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'images':
            return info['images']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for images of the given image ID.
        """

        info = self.image_info[image_id]

        path = self.train_path + info['img_name'] + \
            '/masks/' + info['img_name'] + '.h5'
        if os.path.exists(path):
            # For faster data loading run augment_preprocess.py file first
            # That should save masks in a single h5 file
            with h5py.File(path, "r") as hf:
                mask = hf["arr"][()]
        else:
            path = self.train_path + info['img_name']
            mask = []
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                if 'png' in mask_file:
                    mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                    mask_ = np.where(mask_ > 128, 1, 0)
                    # Fill holes in the mask
                    mask_ = binary_fill_holes(mask_).astype(np.int32)
                    # Add mask only if its area is larger than one pixel
                    if np.sum(mask_) >= 1:
                        mask.append(np.squeeze(mask_))

            mask = np.stack(mask, axis=-1)
            mask = mask.astype(np.uint8)

        # Class ids: all ones since all are foreground objects
        class_ids = np.ones(mask.shape[2])

        return mask.astype(np.uint8), class_ids.astype(np.int8)


def train_validation_split(train_path, seed=10, test_size=0.1):

    """
    Split the dataset into train and validation sets.
    External data and mosaics are directly appended to training set.
    """
    from sklearn.model_selection import train_test_split

    image_ids = list(
        filter(lambda x: ('mosaic' not in x) and ('TCGA' not in x), os.listdir(train_path)))
    mosaic_ids = list(filter(lambda x: 'mosaic' in x, os.listdir(train_path)))
    external_ids = list(filter(lambda x: 'TCGA' in x, os.listdir(train_path)))

    # Load and preprocess the dataset with train image modalities
    df = pd.read_csv('../data/classes.csv')
    df['labels'] = df['foreground'].astype(str) + df['background']
    df['filename'] = df['filename'].apply(lambda x: x[:-4])
    df = df.set_index('filename')
    df = df.loc[image_ids]

    # Split training set based on provided image modalities
    # This ensures that model validates on all image modalities.
    train_list, val_list = train_test_split(df.index, test_size=test_size,
                                            random_state=seed, stratify=df['labels'])

    # Add external data and mos ids to training list
    train_list = list(train_list) + mosaic_ids + external_ids
    val_list = list(val_list)

    return train_list, val_list