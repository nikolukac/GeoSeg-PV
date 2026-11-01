"""Detects PV modules in RGB images of any size (GeoTIFF, PNG or JPEG) with released weights.

python pv_inference.py -i image.tif -w ftunetformer_fom/fold0.safetensors -o output/

The models were trained on Google satellite imagery with ~0.1 m ground sampling distance (GSD). For a
georeferenced input with a different GSD, --gsd 0.1 resamples it first (the result is written on the input grid).
Outputs per input: <name>_pv.tif or .png (PV = 255) and, with --prob, <name>_prob.tif or .png (0-255).
"""
import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

from geoseg.pv import load_model, predict_proba


def read_image(path):
    """uint8 RGB array and rasterio profile (None for plain images)."""
    if path.lower().endswith(('.tif', '.tiff')):
        import rasterio
        with rasterio.open(path) as r:
            a = r.read([1, 2, 3])
            profile = r.profile
        if a.dtype != np.uint8:
            raise ValueError('{}: expected 8-bit RGB, got {}'.format(path, a.dtype))
        return np.transpose(a, (1, 2, 0)), profile
    return np.array(Image.open(path).convert('RGB')), None


def ground_gsd(profile):
    """Ground sampling distance in metres of a projected GeoTIFF (Web Mercator corrected by cos(latitude))."""
    import rasterio
    from rasterio.warp import transform
    t, crs = profile['transform'], profile['crs']
    if crs is None or not crs.is_projected:
        raise ValueError('--gsd needs a projected CRS in metres')
    gsd = (abs(t.a) + abs(t.e)) / 2
    if crs.to_epsg() == 3857:
        _, lat = transform(crs, 'EPSG:4326', [t.c + t.a * profile['width'] / 2], [t.f + t.e * profile['height'] / 2])
        gsd *= math.cos(math.radians(lat[0]))
    return gsd


def write(path, array, profile):
    if profile is None:
        Image.fromarray(array).save(path + '.png')
        return
    import rasterio
    p = dict(profile, count=1, dtype='uint8', compress='deflate', nodata=None)
    p.pop('photometric', None)
    with rasterio.open(path + '.tif', 'w', **p) as dst:
        dst.write(array, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', nargs='+', required=True)
    parser.add_argument('-w', '--weights', nargs='+', required=True, help='.safetensors; several are averaged (ensemble)')
    parser.add_argument('-o', '--output', default='output')
    parser.add_argument('--gsd', type=float, help='resample georeferenced inputs to this GSD in metres first (e.g. 0.1)')
    parser.add_argument('--threshold', type=float, default=0.5, help='PV probability threshold')
    parser.add_argument('--overlap', type=int, default=128, help='overlap of the 512 x 512 patches in pixels')
    parser.add_argument('--prob', action='store_true', help='also write the PV probability')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    models = [load_model(w, args.device) for w in args.weights]
    for path in args.input:
        image, profile = read_image(path)
        scale = 1.0
        if args.gsd:
            if profile is None:
                raise SystemExit('--gsd needs a georeferenced GeoTIFF')
            scale = ground_gsd(profile) / args.gsd
        work = image
        if abs(scale - 1) > 0.01:
            h, w = image.shape[:2]
            work = np.array(Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.BICUBIC))
        prob = predict_proba(models, work, overlap=args.overlap, device=args.device)
        if work is not image:
            prob = np.array(Image.fromarray(prob).resize((image.shape[1], image.shape[0]), Image.BILINEAR))
        name = os.path.join(args.output, os.path.splitext(os.path.basename(path))[0])
        write(name + '_pv', np.where(prob >= args.threshold, 255, 0).astype(np.uint8), profile)
        if args.prob:
            write(name + '_prob', np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8), profile)
        print('{}: {:.2%} PV'.format(path, (prob >= args.threshold).mean()))


if __name__ == '__main__':
    main()
