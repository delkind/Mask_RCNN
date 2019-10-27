import os
import sys
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from mrcnn import utils
from mrcnn.config import Config
# Training dataset
from mrcnn import utils

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

TEST_IMAGES = [3, 9, 21, 27, 28, 29, 30]
IMAGES = list(set(range(1, 56)) - set(TEST_IMAGES))


class HippocampusDataset(utils.Dataset):
    def __init__(self, dataset_dir, validation_split):
        super().__init__()
        self._images = dict()
        self._images['val'] = np.random.choice(IMAGES, len(IMAGES) * validation_split // 100).tolist()
        self._images['train'] = list(set(IMAGES) - set(self._images['val']))
        self._images['test'] = TEST_IMAGES
        self._dataset_dir = dataset_dir

    def load_hippocampus(self, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("hippocampus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]

        image_ids = self._images[subset]

        # Add images
        for image_id in image_ids:
            self.add_image(
                "hippocampus",
                image_id=image_id,
                path=os.path.join(self._dataset_dir, "images/{:02d}_img.png".format(image_id)))

    # def load_image(self, image_id):
    #     """Load the specified image and return a [H,W,3] Numpy array.
    #     """
    #     # Load image
    #     image = skimage.io.imread(self.image_info[image_id]['path'])
    #     image = image.reshape((*(image.shape), 1))
    #     return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_path = os.path.join(self._dataset_dir, "masks/{:02d}_mask.png".format(info['id']))

        # Read mask files from .png image
        mask = []
        m = skimage.io.imread(mask_path)
        for i in range(1, 256):
            instance = m == i
            if np.any(instance):
                mask.append(instance)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hippocampus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Configurations
############################################################

class HippocampusConfig(Config):
    def __init__(self, batch_size, iterations_per_epoch, validation_split, backbone, momentum):
        self.IMAGES_PER_GPU = batch_size
        self.STEPS_PER_EPOCH = iterations_per_epoch * (len(IMAGES) * (100 - validation_split) // 100) // self.IMAGES_PER_GPU
        self.VALIDATION_STEPS = max(1, len(IMAGES) * validation_split // 100 // self.IMAGES_PER_GPU)
        self.LEARNING_MOMENTUM = momentum
        # Backbone network architecture
        # Supported values are: resnet50, resnet101
        self.BACKBONE = backbone

        super().__init__()

    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "hippocampus"

    # Adjust depending on your GPU memory

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Input image resizing
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([110.07])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


def train(batch_size, iterations_per_epoch, validation_split, backbone, momentum, default_logs_dir, weights,
          optimizer, layers, learning_rate, epochs, reduce_lr_on_plateau, reduce_lr_factor, reduce_lr_tolerance,
          tensorboard_update_freq, monitor):
    config = HippocampusConfig(batch_size, iterations_per_epoch, validation_split, backbone, momentum)

    dataset = HippocampusDataset('images/hippocampus', validation_split)
    dataset.load_hippocampus('train')
    dataset.prepare()

    dataset_val = HippocampusDataset('images/hippocampus', validation_split)
    dataset.load_hippocampus('val')
    dataset.prepare()

    import mrcnn.model as modellib

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=default_logs_dir)

    # Select weights file to load
    if weights == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    if optimizer == 'Adabound':
        from keras_adabound import AdaBound

        optimizer = AdaBound(lr=1e-3, final_lr=0.1)

    print("Train network layers: " + layers)
    model.train(dataset, dataset_val,
                learning_rate=learning_rate,
                epochs=epochs,
                augmentation=augmentation,
                layers=layers,
                optimizer=optimizer,
                reduce_lr_on_plateau=reduce_lr_on_plateau,
                reduce_lr_factor=reduce_lr_factor,
                reduce_lr_tolerance=reduce_lr_tolerance,
                tensorboard_update_freq=tensorboard_update_freq,
                monitor=monitor
                )


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for cells counting and segmentation')
    parser.add_argument('--batch_size', default=3, type=int, action='store', help='Some help')
    parser.add_argument('--iterations_per_epoch', default=100, type=int, action='store', help='Some help')
    parser.add_argument('--validation_split', default=10, type = int, action='store', help='Some help')
    parser.add_argument('--backbone', default='resnet50', action='store', help='Some help')
    parser.add_argument('--momentum', default=0, type=int, action='store', help='Some help')
    parser.add_argument('--default_logs_dir', required=True, action='store', help='Some help')
    parser.add_argument('--weights', default='imagenet', action='store', help='Some help')
    parser.add_argument('--optimizer', default='SGD', action='store', help='Some help')
    parser.add_argument('--layers', default='all', action='store', help='Some help')
    parser.add_argument('--learning_rate', default=1e-3, type=float, action='store', help='Some help')
    parser.add_argument('--epochs', default=100, type=int, action='store', help='Some help')
    parser.add_argument('--reduce_lr_on_plateau', action='store_true', help='Some help')
    parser.add_argument('--reduce_lr_factor', default=0.1, type=float, action='store', help='Some help')
    parser.add_argument('--reduce_lr_tolerance', default=1, type=int, action='store', help='Some help')
    parser.add_argument('--tensorboard_update_freq', default=30, type=int, action='store', help='Some help')
    parser.add_argument('--monitor', default='loss', action='store', help='Some help')
    args = parser.parse_args()
    train(args.batch_size, args.iterations_per_epoch, args.validation_split, args.backbone, args.momentum,
          args.default_logs_dir, args.weights, args.optimizer, args.layers, args.learning_rate, args.epochs,
          args.reduce_lr_on_plateau, args.reduce_lr_factor, args.reduce_lr_tolerance, args.tensorboard_update_freq,
          args.monitor)