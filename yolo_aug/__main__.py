import glob
import os

import configargparse as argparse
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

from . import yolo_to_xyxy, xyxy_to_yolo


def default_filter():
    return iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-3, 3),
            shear=(-8, 8)
        )
    ], random_order=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yolo_dir", type=str, help="YOLO directory")
    parser.add_argument("--format", type=str, default='jpg', help="Image format (default jpg)")
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    args = parser.parse_args()

    root = args.yolo_dir
    if not os.path.isdir(root):
        raise SystemError(f'invalid yolo directory {root}')

    seq_det = default_filter()

    for txt_path in glob.glob(os.path.join(root, "*.txt")):
        chip_name = os.path.splitext(txt_path)[0]
        img_path = f'{chip_name}.{args.format}'
        if os.path.isfile(img_path):
            im = np.array(Image.open(img_path))
            imsz = im.shape

            iaboxes = []
            with open(txt_path, 'r') as boxes:
                for ln in boxes.readlines():
                    splits = ln.split(' ')
                    yolo_coords = map(lambda v: float(v), splits[1:])
                    b = yolo_to_xyxy(imsz, yolo_coords)
                    iaboxes.append(ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=splits[0]))

            bbs = ia.BoundingBoxesOnImage(iaboxes, shape=imsz)

            img_aug = seq_det.augment_images([im])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

            if args.debug:
                img_aug = bbs.draw_on_image(img_aug, thickness=1, color=[255, 0, 0])
                img_aug = bbs_aug.draw_on_image(img_aug, thickness=3, color=[0, 255, 0])

            aug_chip = f'{chip_name}a'
            with open(f'{aug_chip}.{args.format}', 'wb') as img_out, open(f'{aug_chip}.txt', 'w') as txt_out:
                outfmt = args.format
                if outfmt == 'jpg':
                    outfmt = 'jpeg'

                Image.fromarray(img_aug).save(img_out, format=outfmt)

                for i in range(len(bbs_aug.bounding_boxes)):
                    bb = bbs_aug.bounding_boxes[i]

                    if bb and not bb.area > 0:
                        x, y, w, h = xyxy_to_yolo(imsz, bb)
                        txt_out.write(f'{bb.label} {x} {y} {w} {h}\n')
        else:
            # todo;; log warning
            continue