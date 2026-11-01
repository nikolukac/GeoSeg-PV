import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image
import random
from .transform import *

# Binary segmentation of PV modules. Label 0 is PV, label 1 is background, so that
# metrics computed over all classes except the last one are the PV metrics.
CLASSES = ('PV', 'Background')
PALETTE = [[255, 255, 255], [0, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[1.0], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75,
                                    ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def mask_to_label(mask):
    """Ground-truth mask (any non-zero value is PV) to labels (0 = PV, 1 = background)."""
    return (np.array(mask) == 0).astype(np.uint8)


class PVDataset(Dataset):
    """512 x 512 patches in <data_root>/<img_dir> and masks with the same file names in <data_root>/<mask_dir>,
    as written by tools/pv_patch_split.py."""

    def __init__(self, data_root='data/pv/test', mode='val', img_dir='images_512', mask_dir='masks_512',
                 img_suffix='.png', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = sorted(osp.splitext(id)[0] for id in mask_filename_list)
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.fromarray(mask_to_label(Image.open(mask_name).convert('L')))
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        imgs, masks = zip(*(self.load_img_and_mask(i) for i in indexes))
        imgs = [np.array(i) for i in imgs]
        masks = [np.array(m) for m in masks]

        h, w = self.img_size
        offset_x = random.randint(w // 4, w - w // 4)
        offset_y = random.randint(h // 4, h - h // 4)
        crop_sizes = [(offset_x, offset_y), (w - offset_x, offset_y), (offset_x, h - offset_y), (w - offset_x, h - offset_y)]
        crops = [albu.RandomCrop(width=cw, height=ch)(image=i.copy(), mask=m.copy())
                 for (cw, ch), i, m in zip(crop_sizes, imgs, masks)]

        top = np.concatenate((crops[0]['image'], crops[1]['image']), axis=1)
        bottom = np.concatenate((crops[2]['image'], crops[3]['image']), axis=1)
        img = np.ascontiguousarray(np.concatenate((top, bottom), axis=0))
        top_mask = np.concatenate((crops[0]['mask'], crops[1]['mask']), axis=1)
        bottom_mask = np.concatenate((crops[2]['mask'], crops[3]['mask']), axis=1)
        mask = np.ascontiguousarray(np.concatenate((top_mask, bottom_mask), axis=0))
        return Image.fromarray(img), Image.fromarray(mask)
