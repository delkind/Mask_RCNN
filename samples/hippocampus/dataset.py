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
    @staticmethod
    def create(image_dir, mask_dir, validation_split=10, crop_size=320, border_size=0):
        files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        images = [os.path.join(image_dir, f) for f in files]
        masks = [os.path.join(mask_dir, f) for f in files]
        return HippocampusDataset(images, masks, validation_split, crop_size, border_size)

    def __init__(self, images, masks, validation_split=10, crop_size=320, border_size=0):
        super().__init__()
        self.border_size = border_size
        self.crop_size = crop_size
        self.masks = masks
        self.images = images
        self.crops = []

        for num, image in enumerate(self.images):
            c = create_crops_coords_list(self.crop_size, self.border_size, skimage.io.imread(image))
            self.crops += list(zip([num] * len(c), c))

        ids = list(range(len(self.crops)))
        self._images = dict()
        self._images['val'] = np.random.choice(ids, len(self.crops) * validation_split // 100, replace=False).tolist()
        ids = list(set(ids) - set(self._images['val']))
        self._images['test'] = np.random.choice(ids, len(self.crops) * validation_split // 100, replace=False).tolist()
        ids = list(set(ids) - set(self._images['test']))
        self._images['train'] = ids
        pass

    def get_subset(self, subset):
        return HippocampusSubset(self, subset)

    def get_crop(self, image_id):
        img_index, (y, x) = self.crops[image_id]
        self.images[img_index] = self.get_image(img_index)

        return self.images[img_index][y:y + self.crop_size, x:x + self.crop_size, ...]

    def get_image(self, img_index):
        if isinstance(self.images[img_index], str):
            image = skimage.io.imread(self.images[img_index])
            if len(image.shape) == 2:
                image = image.reshape((*image.shape, 1))
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = self.images[img_index]

        return image

    def get_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        img_index, (y, x) = self.crops[image_id]
        if isinstance(self.masks[img_index], str):
            if os.path.isfile(self.masks[img_index]):
                image = skimage.io.imread(self.masks[img_index])
                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                self.masks[img_index] = image
            else:
                self.masks[img_index] = np.zeros_like(self.get_image(img_index))

        raw_mask = self.masks[img_index][y:y + self.crop_size, x:x + self.crop_size, ...].copy()
        m = extract_components(raw_mask)
        mask = []

        for i in range(m.max()):
            instance = m == (i + 1)
            if np.any(instance):
                mask.append(instance)

        if not mask:
            mask += [m != 0]

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


class HippocampusSubset(utils.Dataset):
    def __init__(self, dataset, subset):
        super().__init__()
        self.subset = subset
        self.dataset = dataset
        self.num_classes = 2

        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("hippocampus", 1, "nucleus")
        self.source_class_ids = {'hippocampus':  1}

        # Add images
        for image_id, _ in enumerate(self.image_ids):
            self.add_image(
                "hippocampus",
                image_id=image_id,
                path=None)

    def load_image(self, image_id):
        return self.dataset.get_crop(self.dataset._images[self.subset][image_id])

    def load_mask(self, image_id):
        return self.dataset.get_mask(self.dataset._images[self.subset][image_id])

    @property
    def image_ids(self):
        return list(range(len(self.dataset._images[self.subset])))

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hippocampus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


if __name__ == '__main__':
    ds = HippocampusDataset.create('images/hippocampus/crops', 'images/hippocampus/labeled_crops')
    train = ds.get_subset('train')

    image_ids = train.image_ids
    for image_id in image_ids:
        image = train.load_image(image_id)
        mask, class_ids = train.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, ['', 'cell'], limit=1)
