"""Writes the train/val/test patches of the PV ground truth for one or all of its five splits.

Input: the ground-truth release (masks/, splits.json) and the RGB patches rebuilt with its get_rgb.py.
Output: <out>/fold<k>/{train,val,test}/{images_512,masks_512}/<id>_<m>.png, as read by geoseg.datasets.pv_dataset.
As in the published experiments, training patches are added flipped horizontally (m = 1) and vertically (m = 2).

python tools/pv_patch_split.py --gt ../PV_ground_truth --images ../PV_ground_truth/images --out data/pv
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

PATCH = 512


def load(path):
    a = np.array(Image.open(path))
    if a.shape[0] != PATCH:  # a whole 1024 x 1024 tile (get_rgb.py --full): keep the centre
        o = (a.shape[0] - PATCH) // 2
        a = a[o:o + PATCH, o:o + PATCH]
    return a


def write_fold(gt, images, fold, out):
    for part in ('train', 'val', 'test'):
        img_dir = os.path.join(out, 'fold{}'.format(fold), part, 'images_512')
        mask_dir = os.path.join(out, 'fold{}'.format(fold), part, 'masks_512')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for tile_id in fold_ids(gt, fold)[part]:
            img = load(os.path.join(images, tile_id + '.png'))[..., :3]
            mask = load(os.path.join(gt, 'masks', tile_id + '.png'))
            versions = [(img, mask)]
            if part == 'train':
                versions += [(img[:, ::-1], mask[:, ::-1]), (img[::-1], mask[::-1])]
            for m, (i, k) in enumerate(versions):
                name = '{}_{}.png'.format(tile_id, m)
                Image.fromarray(np.ascontiguousarray(i)).save(os.path.join(img_dir, name))
                Image.fromarray(np.ascontiguousarray(k)).save(os.path.join(mask_dir, name))
        print('fold {} {}: {} tiles'.format(fold, part, len(fold_ids(gt, fold)[part])))


def fold_ids(gt, fold):
    return json.load(open(os.path.join(gt, 'splits.json')))['folds'][fold]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt', required=True, help='ground-truth folder with masks/ and splits.json')
    parser.add_argument('--images', required=True, help='RGB patches <id>.png from get_rgb.py')
    parser.add_argument('--out', default='data/pv')
    parser.add_argument('--fold', type=int, choices=range(5), help='only this split (default: all five)')
    args = parser.parse_args()
    for fold in ([args.fold] if args.fold is not None else range(5)):
        write_fold(args.gt, args.images, fold, args.out)


if __name__ == '__main__':
    main()
