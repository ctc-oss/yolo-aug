import glob
import os

import logging
import configargparse as argparse
import imgaug as ia
import numpy as np
from PIL import Image

from . import yolo_to_xyxy, xyxy_to_yolo, load_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yolo_dir", type=str, help="YOLO directory")
    parser.add_argument("--format", type=str, default='jpg', help="Image format (default jpg)")
    parser.add_argument("--pipeline", type=str, help="Name of pipeline function to run")
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    parser.add_argument("--chip_dir", type=str, default='chipped', help='Debug mode')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("yolo")

    yolo_dir = args.yolo_dir
    if not os.path.isdir(yolo_dir):
        raise SystemError(f'invalid yolo directory {yolo_dir}')

    training_list_txt = os.path.join(yolo_dir, 'training_list.txt')
    if not os.path.exists(training_list_txt):
        raise SystemError(f'invalid yolo training list {training_list_txt}')

    chip_dir = os.path.join(yolo_dir, args.chip_dir)
    if not os.path.isdir(chip_dir):
        raise SystemError(f'invalid yolo chip directory {chip_dir}')

    aug_pipeline = load_pipeline(args.pipeline)()
    aug_chips = []

    for txt_path in glob.glob(os.path.join(chip_dir, "*.txt")):
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
            img_aug, bbs_aug = aug_pipeline(image=im, bounding_boxes=bbs)

            if args.debug:
                img_aug = bbs.draw_on_image(img_aug, size=1, color=[128, 0, 0], alpha=.5)
                img_aug = bbs_aug.draw_on_image(img_aug, size=2, color=[0, 255, 0])

            chip = f'{chip_name}a'
            img_chip = f'{chip}.{args.format}'
            txt_chip = f'{chip}.txt'
            with open(img_chip, 'wb') as img_out, \
                    open(txt_chip, 'w') as txt_out, \
                    open(training_list_txt, 'a') as training_list:

                outfmt = args.format
                if outfmt == 'jpg':
                    outfmt = 'jpeg'

                Image.fromarray(img_aug).save(img_out, format=outfmt)
                training_list.write(f'{img_chip}\n')

                for i in range(len(bbs_aug.bounding_boxes)):
                    bb = bbs_aug.bounding_boxes[i]

                    if bb and bb.area > 0:
                        x, y, w, h = xyxy_to_yolo(imsz, bb)
                        txt_out.write(f'{bb.label} {x} {y} {w} {h}\n')
        else:
            # todo;; log warning
            continue

    logger.info(f'file://{yolo_dir}')
