"""Example: PV module detection on the 8 test tiles of split 0 of the PV ground truth.

1. Rebuild the RGB patches of these tiles (the imagery is not redistributed), e.g. with a Google Maps Platform key:
     python <ground_truth>/get_rgb.py google --key YOUR_KEY --fold 0 --part test --out example/images
2. Run a model trained on split 0 and compare with the ground truth:
     python example/run_example.py --gt <ground_truth> --images example/images -w <models>/ftunetformer_fom/fold0.safetensors

Writes example/output/: an error map per tile (white = correct PV, magenta = false PV, cyan = missed PV),
overview.png with all tiles, and results.txt with F1 and IoU per tile and over all tiles.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geoseg.pv import load_model, predict_proba  # noqa: E402

COLOURS = {'tp': (255, 255, 255), 'fp': (230, 0, 200), 'fn': (0, 200, 230), 'tn': (40, 40, 40)}


def scores(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt', required=True, help='ground-truth folder (masks/, splits.json)')
    p.add_argument('--images', default=os.path.join(HERE, 'images'), help='RGB patches <id>.png from get_rgb.py')
    p.add_argument('-w', '--weights', nargs='+', required=True, help='.safetensors trained on split 0; several are averaged')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--out', default=os.path.join(HERE, 'output'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ids = json.load(open(os.path.join(args.gt, 'splits.json')))['folds'][args.fold]['test']
    models = [load_model(w, args.device) for w in args.weights]
    os.makedirs(args.out, exist_ok=True)
    total = np.zeros(3)
    lines, maps = [], []
    for tile_id in ids:
        img = np.array(Image.open(os.path.join(args.images, tile_id + '.png')).convert('RGB'))
        gt = np.array(Image.open(os.path.join(args.gt, 'masks', tile_id + '.png'))) > 0
        pred = predict_proba(models, img, device=args.device) >= 0.5
        counts = np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()])
        total += counts
        f1, iou = scores(*counts)
        lines.append('{:24s} F1 {:.4f}  IoU {:.4f}'.format(tile_id, f1, iou))
        err = np.zeros(gt.shape + (3,), np.uint8)
        for key, sel in (('tn', ~pred & ~gt), ('tp', pred & gt), ('fp', pred & ~gt), ('fn', ~pred & gt)):
            err[sel] = COLOURS[key]
        Image.fromarray(err).save(os.path.join(args.out, tile_id + '_errors.png'))
        maps.append((tile_id, f1, err))
        print(lines[-1], flush=True)
    f1, iou = scores(*total)
    lines.append('{:24s} F1 {:.4f}  IoU {:.4f}'.format('all {} tiles'.format(len(ids)), f1, iou))
    print(lines[-1])
    with open(os.path.join(args.out, 'results.txt'), 'w') as f:
        f.write('weights: {}\n'.format(', '.join(os.path.basename(os.path.dirname(w)) + '/' + os.path.basename(w) for w in args.weights)))
        f.write('\n'.join(lines) + '\n')

    # overview: 4 x 2 error maps at half resolution
    cols, cell = 4, 256
    rows = (len(maps) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 18)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (tile_id, f1, err) in enumerate(maps):
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        sheet.paste(Image.fromarray(err).resize((cell, cell), Image.NEAREST), (x, y + 18))
        draw.text((x + 4, y + 3), '{}  F1 {:.3f}'.format(tile_id, f1), fill=(0, 0, 0))
    sheet.save(os.path.join(args.out, 'overview.png'))


if __name__ == '__main__':
    main()
