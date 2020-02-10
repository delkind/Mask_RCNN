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
from samples.hippocampus.dataset import HippocampusDataset
from samples.hippocampus.predict_full import create_crops_coords_list

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

TEST_IMAGES = [3, 9, 21, 27, 28, 29, 30]
IMAGES = list(set(range(1, 56)) - set(TEST_IMAGES))


############################################################
#  Configurations
############################################################

class NewHippocampusConfig(Config):
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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    VALIDATION_STEPS = 1

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_SCALE = 0.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

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
    MEAN_PIXEL = np.array([110.07, 110.07, 110.07])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
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


def train(image_dir, project, crop_size, batch_size, iterations_per_epoch, validation_split, backbone, momentum, default_logs_dir, weights,
          optimizer, layers, learning_rate, epochs, reduce_lr_on_plateau, reduce_lr_factor, reduce_lr_tolerance,
          tensorboard_update_freq, monitor):
    config = NewHippocampusConfig(batch_size, iterations_per_epoch, validation_split, backbone, momentum)

    dataset = HippocampusDataset(image_dir, project, validation_split, crop_size)

    train_subset = dataset.get_subset('train')
    val_subset = dataset.get_subset('val')

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
    model.train(train_subset, val_subset,
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

    parser.add_argument('--image_dir', action='store', required=True)
    parser.add_argument('--project', action='store', required=True)
    parser.add_argument('--crop_size', action='store', default=320, type=int)
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
    print(vars(args))

    train(**vars(args))
