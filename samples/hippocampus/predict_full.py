import argparse
import itertools

import cv2
import numpy as np


def split_image(image_path, crop_size, border_size):
    image = cv2.imread(image_path)

    vert = list(range(0, image.shape[0], crop_size - 2 * border_size))
    horiz = list(range(0, image.shape[1], crop_size - 2 * border_size))
    vert = list(filter(lambda v: v + crop_size <= image.shape[0], vert)) + [image.shape[0] - crop_size]
    horiz = list(filter(lambda v: v + crop_size <= image.shape[1], horiz)) + [image.shape[1] - crop_size]

    crop_coords = list(itertools.product(vert, horiz))
    crops = [image[i:i + crop_size, j:j + crop_size, ...] for (i, j) in crop_coords]
    return list(zip(crops, crop_coords)), image


def filter_rois(rois, border_size, crop_size):
    valid = list(map(lambda r: not (at_border((r[0], r[2]), border_size, crop_size) or at_border((r[1], r[3]),
                                                                                                 border_size,
                                                                                                 crop_size)), rois))
    # return np.nonzero(valid)[0]
    return range(len(rois))


def at_border(r, border_size, crop_size):
    return (r[0] < border_size and r[1] < border_size) or (
            r[0] > crop_size - border_size and r[1] > crop_size - border_size)


def predict_full_image(weights, image_path, crop_size, border_size, output_image_path, bounding_boxes,
                       backbone='resnet50'):
    import matplotlib.pyplot as plt
    crops, image = split_image(image_path, crop_size, border_size)
    model = create_model(backbone, weights)

    for num, crop in enumerate(crops):
        print("Processing crop {} out of {}...".format(num + 1, len(crops)))
        result = model.detect([crop[0]], verbose=0)[0]
        mask_image(crop[0], result['rois'], result['masks'], result['class_ids'],
                   {1: 'cell'}, show_bbox=bounding_boxes)

    cv2.imwrite(output_image_path, image)


def adjust_results(border_size, coords, crop_size, image, result):
    filtered_result = filter_result(result, border_size, crop_size)
    filtered_result['rois'] = np.array([a + [coords[0], coords[1], coords[0], coords[1]]
                                        for a in filtered_result['rois']])
    filtered_result['masks'] = apply_masks(coords, filtered_result['masks'])
    return filtered_result


def apply_masks(coords, masks):
    result = []
    for i in range(masks.shape[-1]):
        mask = masks[..., i]
        result += [(mask, coords)]
    return result


def filter_result(result, border_size, crop_size):
    to_retain = filter_rois(result['rois'], border_size, crop_size)
    return {
        'rois': result['rois'][to_retain, ...],
        'masks': result['masks'][:, :, to_retain],
        'scores': result['scores'][to_retain, ...],
        'class_ids': result['class_ids'][to_retain, ...]
    }


def create_model(backbone, weights):
    from samples.hippocampus.train import HippocampusConfig
    from mrcnn import model as modellib

    class HippocampusInferenceConfig(HippocampusConfig):
        def __init__(self, batch_size):
            super().__init__(batch_size, 0, 0, backbone, 0)

        # Don't resize imager for inferencing
        IMAGE_RESIZE_MODE = "pad64"
        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7

    inference_config = HippocampusInferenceConfig(1)
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir='.')
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = weights
    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def mask_image(image, boxes, masks, class_ids, class_names,
               scores=None, title="",
               figsize=(32, 32), ax=None,
               show_bbox=True,
               colors=None, captions=None):
    from skimage.measure import find_contours

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if colors is None:
        colors = [(0, 255, 0)] * N
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        # Mask
        mask = masks[..., i]

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask

        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = (np.fliplr(verts) - 1).reshape(-1, 1, 2).astype(int)
            cv2.polylines(image, [verts], True, color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for cells counting and segmentation - predictions')
    parser.add_argument('--weights', required=True, action='store', help='Some help')
    parser.add_argument('--full_image', required=True, action='store', help='Some help')
    parser.add_argument('--output_image', required=True, action='store', help='Some help')
    parser.add_argument('--crop_size', default=320, type=int, action='store', help='Some help')
    parser.add_argument('--border_size', default=20, type=int, action='store', help='Some help')
    parser.add_argument('--bounding_boxes', action='store_true', help='Some help')
    args = parser.parse_args()

    predict_full_image(args.weights, args.full_image, args.crop_size, args.border_size, args.output_image,
                       args.bounding_boxes)
