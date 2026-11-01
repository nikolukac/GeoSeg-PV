"""Evaluates PV module detection on a test split (F1 and IoU of the PV class).

Trained checkpoint of a config:     python pv_test.py -c config/pv/ftunetformer_fom.py
Released weights (one or more):     python pv_test.py -w ftunetformer_fom/fold0.safetensors --data data/pv/fold0/test
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoseg.datasets.pv_dataset import CLASSES, PVDataset, val_aug
from geoseg.pv import PV_CLASS, load_model
from tools.metric import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="config whose trained checkpoint (weights_path/test_weights_name.ckpt) is tested")
    arg("-w", "--weights", nargs='+', help="released .safetensors weights; several are averaged (ensemble)")
    arg("--data", help="split folder with images_512/ and masks_512/ (with -w; default: the config's test_dataset)")
    arg("-o", "--output_path", type=Path, help="where to write the predicted masks (PV = 255)")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--per-image", action='store_true', help="also print F1 and IoU of every image")
    arg("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # mean of the softmax outputs, returned as log-probabilities so that softmax gives them back
        return torch.log(sum(torch.softmax(m(x), dim=1) for m in self.models) / len(self.models))


def main():
    args = get_args()
    if bool(args.config_path) == bool(args.weights):
        raise SystemExit('give either -c or -w')
    if args.weights:
        model = Ensemble([load_model(w) for w in args.weights])
        dataset = PVDataset(data_root=args.data, mode='test', transform=val_aug) if args.data else None
    else:
        from train_supervision import Supervision_Train, py2cfg
        config = py2cfg(args.config_path)
        model = Supervision_Train.load_from_checkpoint(
            os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config, map_location='cpu')
        dataset = PVDataset(data_root=args.data, mode='test', transform=val_aug) if args.data else config.test_dataset
    if dataset is None:
        raise SystemExit('-w needs --data')
    model.to(args.device).eval()

    if args.tta == "lr":
        model = tta.SegmentationTTAWrapper(model, tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()]))
    elif args.tta == "d4":
        model = tta.SegmentationTTAWrapper(model, tta.Compose([
            tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)]))
    if args.output_path:
        args.output_path.mkdir(exist_ok=True, parents=True)

    evaluator = Evaluator(num_class=len(CLASSES))
    single = Evaluator(num_class=len(CLASSES))
    loader = DataLoader(dataset, batch_size=2, num_workers=2, drop_last=False)
    with torch.no_grad():
        for batch in tqdm(loader):
            predictions = nn.Softmax(dim=1)(model(batch['img'].to(args.device))).argmax(dim=1).cpu().numpy()
            masks_true = batch['gt_semantic_seg'].numpy()
            for img_id, pred, gt in zip(batch['img_id'], predictions, masks_true):
                evaluator.add_batch(pre_image=pred, gt_image=gt)
                if args.per_image:
                    single.reset()
                    single.add_batch(pre_image=pred, gt_image=gt)
                    print('{}: F1 {:.4f}, IoU {:.4f}'.format(img_id, single.F1()[PV_CLASS], single.Intersection_over_Union()[PV_CLASS]))
                if args.output_path:
                    cv2.imwrite(str(args.output_path / (img_id + '.png')), np.where(pred == PV_CLASS, 255, 0).astype(np.uint8))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    for class_name, class_iou, class_f1 in zip(CLASSES, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), evaluator.OA()))


if __name__ == "__main__":
    main()
