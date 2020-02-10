import json
import os

import cv2
import numpy as np

import skimage.io

from mrcnn import utils, visualize
from samples.hippocampus.predict_full import create_crops_coords_list


def extract_components(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image, 1, 100, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = np.zeros((*image.shape, 3), dtype=np.uint8)

    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts).reshape((-1, 1, 2)).astype(int)
        cv2.polylines(labels, [verts], True, (0, 255, 0))
        cv2.fillPoly(labels, [verts], (0, 255, 0))

    ret, labels = cv2.connectedComponents(cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY))
    if ret:
        return labels


class HippocampusDataset:
    def __init__(self, image_dir, json_path, validation_split=10, crop_size=320, border_size=0):
        super().__init__()
        with open(json_path, "rt") as in_file:
            self.project = json.load(in_file)
        self.border_size = border_size
        self.crop_size = crop_size
        valid_images = [k for (k, v) in self.project['_via_img_metadata'].items() if v['regions']]
        self.images = [os.path.join(image_dir, self.project['_via_img_metadata'][x]['filename']) for x in valid_images]
        self.masks = [self.project['_via_img_metadata'][x]['regions'] for x in valid_images]
        ids = range(len(self.images))
        self._subsets = dict()
        self._subsets['val'] = np.random.choice(ids, len(ids) * validation_split // 100, replace=False).tolist()
        ids = list(set(ids) - set(self._subsets['val']))
        self._subsets['test'] = np.random.choice(ids, len(ids) * validation_split // 100, replace=False).tolist()
        ids = list(set(ids) - set(self._subsets['test']))
        self._subsets['train'] = ids
        self.subsets = {n: HippocampusSubset(self, n) for n in ['train', 'val', 'test']}

    def get_subset(self, subset):
        return self.subsets[subset]

    def get_image(self, img_index):
        if isinstance(self.images[img_index], str):
            image = skimage.io.imread(self.images[img_index])
            if image.shape[0] != image.shape[1]:
                image = image[:, :image.shape[0], ...]
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = self.images[img_index]

        return image

    def get_mask(self, img_index):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        if isinstance(self.masks[img_index], list):
            mask = []
            shape = self.get_image(img_index).shape
            for region in self.masks[img_index]:
                layer = np.zeros((shape[0], shape[1]), dtype=np.int8)
                polygon = np.array([[[x, y] for (x, y) in (zip(region['shape_attributes']['all_points_x'],
                                                               region['shape_attributes']['all_points_y']))]])
                cv2.fillPoly(layer, polygon, 255)
                mask += [layer != 0]
            if mask:
                mask = np.stack(mask, axis=-1)
            else:
                mask = np.array(mask)
            self.masks[img_index] = mask
        else:
            mask = self.masks[img_index]
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


class HippocampusSubset(utils.Dataset):
    def __init__(self, dataset, subset):
        super().__init__()
        self.subset = subset
        self.dataset = dataset
        self.num_classes = 2

        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("hippocampus", 0, "background")
        self.add_class("hippocampus", 1, "nucleus")
        self.source_class_ids = {'background': 0, 'hippocampus': 1}

        # Add images
        for image_id, _ in enumerate(self.image_ids):
            self.add_image(
                "hippocampus",
                image_id=image_id,
                path=None)

    def load_image(self, image_id):
        return self.dataset.get_image(self.dataset._subsets[self.subset][image_id])

    def load_mask(self, image_id):
        return self.dataset.get_mask(self.dataset._subsets[self.subset][image_id])

    @property
    def image_ids(self):
        return list(range(len(self.dataset._subsets[self.subset])))

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hippocampus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


if __name__ == '__main__':
    ds = HippocampusDataset('/Users/david/study/openu/thesis/hipposeg/combined/images',
                            '/Users/david/study/openu/thesis/hipposeg/combined/images/David2-Proc1.json')
    train = ds.get_subset('train')

    image_ids = train.image_ids
    for image_id in image_ids:
        image = train.load_image(image_id)
        mask, class_ids = train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, ['', 'cell'], limit=1)
