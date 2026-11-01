"""PV module detection: the 10 networks of the PV benchmark, loading of released weights and tiled prediction."""
import json
import numpy as np
import torch
from torch import nn

from geoseg.models.A2FPN import A2FPN
from geoseg.models.BANet import BANet
from geoseg.models.DCSwin import dcswin_small
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.MANet import MANet
from geoseg.models.UNetFormer import UNetFormer

NUM_CLASSES = 2  # 0 = PV, 1 = background
PV_CLASS = 0
PATCH_SIZE = 512
# albumentations.Normalize() defaults, as used in training
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ARCHS = {
    'unetformer_r18': lambda pretrained, fom: UNetFormer(num_classes=NUM_CLASSES, backbone_name='resnet18', pretrained=pretrained, fom=fom),
    'unetformer_effb5': lambda pretrained, fom: UNetFormer(num_classes=NUM_CLASSES, backbone_name='efficientnet_b5', pretrained=pretrained, fom=fom),
    'unetformer_seresnext26d': lambda pretrained, fom: UNetFormer(num_classes=NUM_CLASSES, backbone_name='seresnext26d_32x4d', pretrained=pretrained, fom=fom),
    'ftunetformer': lambda pretrained, fom: ft_unetformer(pretrained=pretrained, num_classes=NUM_CLASSES, decoder_channels=256, fom=fom),
    'dcswin': lambda pretrained, fom: dcswin_small(pretrained=pretrained, num_classes=NUM_CLASSES, fom=fom),
    'banet': lambda pretrained, fom: BANet(num_classes=NUM_CLASSES, weight_path='pretrain_weights/rest_lite.pth' if pretrained else None, fom=fom),
    'a2fpn': lambda pretrained, fom: A2FPN(class_num=NUM_CLASSES, pretrained=pretrained, fom=fom),
    'manet_r18': lambda pretrained, fom: MANet(num_classes=NUM_CLASSES, backbone_name='resnet18', pretrained=pretrained, fom=fom),
    'manet_effb5': lambda pretrained, fom: MANet(num_classes=NUM_CLASSES, backbone_name='efficientnet_b5', pretrained=pretrained, fom=fom),
    'manet_seresnext26d': lambda pretrained, fom: MANet(num_classes=NUM_CLASSES, backbone_name='seresnext26d_32x4d', pretrained=pretrained, fom=fom),
}


def build_model(arch, fom=False, pretrained=False):
    """Network of the benchmark. pretrained=True loads ImageNet backbone weights (needed for training only)."""
    if arch not in ARCHS:
        raise ValueError('unknown arch {}, one of {}'.format(arch, ', '.join(ARCHS)))
    return ARCHS[arch](pretrained, fom)


def unique_state_dict(model):
    """state_dict without aliases: A2FPN reaches the same ResNet modules through two attribute paths."""
    seen, state = set(), {}
    for k, v in model.state_dict().items():
        if v.data_ptr() not in seen:
            seen.add(v.data_ptr())
            state[k] = v.contiguous()
    return state


def save_weights(model, path, metadata):
    from safetensors.torch import save_file
    save_file(unique_state_dict(model), path, metadata=metadata)


def load_model(weights_path, device='cpu'):
    """Network with released weights (.safetensors with 'arch' and 'fom' in its metadata), in eval mode."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(weights_path, framework='pt') as f:
        meta = f.metadata()
    model = build_model(meta['arch'], fom=json.loads(meta['fom']), pretrained=False)
    missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
    aliases = set(model.state_dict()) - set(unique_state_dict(model))
    if unexpected or not set(missing) <= aliases:
        raise RuntimeError('weights do not match {} (fom={}): missing {}, unexpected {}'.format(
            meta['arch'], meta['fom'], sorted(set(missing) - aliases), unexpected))
    return model.to(device).eval()


def normalize(image):
    """uint8 RGB (H, W, 3) to a normalised float tensor (3, H, W)."""
    x = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def _window(size):
    # smooth weights for blending overlapping patches, highest in the patch centre
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), 1e-3)


@torch.no_grad()
def predict_proba(models, image, overlap=128, batch_size=4, device='cpu'):
    """PV probability (H, W) of an uint8 RGB image (H, W, 3) of any size.

    The image is processed in 512 x 512 patches with the given overlap; the softmax outputs of
    overlapping patches (and of all models, if several are given) are averaged.
    """
    if isinstance(models, nn.Module):
        models = [models]
    h, w = image.shape[:2]
    p = PATCH_SIZE
    stride = p - overlap
    ph = max(p, int(np.ceil((h - p) / stride)) * stride + p)
    pw = max(p, int(np.ceil((w - p) / stride)) * stride + p)
    padded = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode='reflect' if min(h, w) > 1 else 'edge')
    x = normalize(padded)
    prob = np.zeros((ph, pw), np.float32)
    weight = np.zeros((ph, pw), np.float32)
    win = _window(p)
    origins = [(y, x0) for y in range(0, ph - p + 1, stride) for x0 in range(0, pw - p + 1, stride)]
    for i in range(0, len(origins), batch_size):
        batch = origins[i:i + batch_size]
        inp = torch.stack([x[:, y:y + p, x0:x0 + p] for y, x0 in batch]).to(device)
        out = sum(torch.softmax(m(inp), dim=1)[:, PV_CLASS] for m in models) / len(models)
        out = out.cpu().numpy()
        for (y, x0), o in zip(batch, out):
            prob[y:y + p, x0:x0 + p] += o * win
            weight[y:y + p, x0:x0 + p] += win
    return (prob / weight)[:h, :w]
