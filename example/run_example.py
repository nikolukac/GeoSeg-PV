"""Example: PV module detection on the 8 test tiles of split 0 of the PV ground truth.

1. Rebuild the RGB patches of these tiles (the imagery is not redistributed), e.g. with a Google Maps Platform key:
     python <ground_truth>/get_rgb.py google --key YOUR_KEY --fold 0 --part test --out example/images
2. Run a model trained on split 0 and compare with the ground truth:
     python example/run_example.py --gt <ground_truth> --images example/images -w <models>/ftunetformer_fom/fold0.safetensors

Writes example/output/: an error map per tile (white = correct PV, magenta = false PV, cyan = missed PV),
overview.png with all tiles, and results.txt with F1 and IoU per tile and over all tiles.
With --overlay also overlay.png: the satellite tiles with the detected modules overlaid (orange).
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
OVERLAY_COLOUR = (255, 140, 0)
IMAGERY_CREDIT = 'Imagery (c) 2023 Google, Airbus, CNES/Airbus, Maxar Technologies'


def scores(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)


def overlay(img, pred):
    """The RGB image with the detected modules tinted and outlined in OVERLAY_COLOUR."""
    out = img.astype(np.float32)
    colour = np.array(OVERLAY_COLOUR, np.float32)
    out[pred] = 0.5 * out[pred] + 0.5 * colour
    p = np.pad(pred, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    out[pred & ~inner] = colour
    return out.astype(np.uint8)


def contact_sheet(cells, cols=4, cell=256, credit=None):
    """Tiles (label, RGB array) in a grid at cell x cell pixels, with an optional credit line."""
    rows = (len(cells) + cols - 1) // cols
    foot = 18 if credit else 0
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 18) + foot), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (label, rgb) in enumerate(cells):
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        resample = Image.NEAREST if credit is None else Image.LANCZOS
        sheet.paste(Image.fromarray(rgb).resize((cell, cell), resample), (x, y + 18))
        draw.text((x + 4, y + 3), label, fill=(0, 0, 0))
    if credit:
        draw.text((4, rows * (cell + 18) + 3), credit, fill=(90, 90, 90))
    return sheet


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt', required=True, help='ground-truth folder (masks/, splits.json)')
    p.add_argument('--images', default=os.path.join(HERE, 'images'), help='RGB patches <id>.png from get_rgb.py')
    p.add_argument('-w', '--weights', nargs='+', required=True, help='.safetensors trained on split 0; several are averaged')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--out', default=os.path.join(HERE, 'output'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--overlay', action='store_true', help='also write overlay.png: the imagery with the detected modules')
    args = p.parse_args()

    ids = json.load(open(os.path.join(args.gt, 'splits.json')))['folds'][args.fold]['test']
    models = [load_model(w, args.device) for w in args.weights]
    os.makedirs(args.out, exist_ok=True)
    total = np.zeros(3)
    lines, maps, overlays = [], [], []
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
        if args.overlay:
            overlays.append(('{}  F1 {:.3f}'.format(tile_id, f1), overlay(img, pred)))
        print(lines[-1], flush=True)
    f1, iou = scores(*total)
    lines.append('{:24s} F1 {:.4f}  IoU {:.4f}'.format('all {} tiles'.format(len(ids)), f1, iou))
    print(lines[-1])
    with open(os.path.join(args.out, 'results.txt'), 'w') as f:
        f.write('weights: {}\n'.format(', '.join(os.path.basename(os.path.dirname(w)) + '/' + os.path.basename(w) for w in args.weights)))
        f.write('\n'.join(lines) + '\n')

    # overview: 4 x 2 error maps at half resolution
    contact_sheet([('{}  F1 {:.3f}'.format(t, f), e) for t, f, e in maps]).save(os.path.join(args.out, 'overview.png'))
    # overlay: the same tiles with the imagery and the detected modules (only this sheet holds imagery)
    if args.overlay:
        contact_sheet(overlays, credit=IMAGERY_CREDIT).save(os.path.join(args.out, 'overlay.png'))


if __name__ == '__main__':
    main()
