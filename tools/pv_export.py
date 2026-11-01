"""Exports the network weights of a training checkpoint (.ckpt) to .safetensors, for geoseg.pv.load_model.

python -m tools.pv_export model_weights/pv/fold0/ftunetformer_fom/ftunetformer_fom.ckpt ftunetformer_fom_fold0.safetensors --arch ftunetformer --fom
"""
import argparse
import json
import torch

from geoseg.pv import ARCHS, build_model, save_weights


def export(ckpt_path, out_path, arch, fom, metadata=None):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
    model = build_model(arch, fom=fom)
    model.load_state_dict(state, strict=True)  # fails if arch/fom do not match the weights
    meta = {'arch': arch, 'fom': json.dumps(fom), 'epoch': str(ckpt.get('epoch', ''))}
    meta.update({k: str(v) for k, v in (metadata or {}).items()})
    save_weights(model, out_path, meta)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('out')
    parser.add_argument('--arch', required=True, choices=sorted(ARCHS))
    parser.add_argument('--fom', action='store_true', help='the network has the Fine Optimization Module')
    args = parser.parse_args()
    export(args.ckpt, args.out, args.arch, args.fom)


if __name__ == '__main__':
    main()
