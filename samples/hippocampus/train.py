import os

import numpy as np

import samples.hippocampus.hippocampus as hippocampus
import mrcnn.model as modellib
import mrcnn.utils as utils

ROOT_DIR = '../../'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = hippocampus.HippocampusConfig()
config.display()

# Training dataset
from mrcnn import visualize, utils

dataset_train = hippocampus.HippocampusDataset('../../images/hippocampus')
dataset_train.load_hippocampus('train')
dataset_train.prepare()

# Validation dataset
dataset_val = hippocampus.HippocampusDataset('../../images/hippocampus')
dataset_val.load_hippocampus('val')
dataset_val.prepare()

#%%

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

#%% md

## Create Model

#%%

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

