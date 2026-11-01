import os
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.pv_dataset import *
from geoseg.models.MANet import MANet
from tools.utils import Lookahead
from tools.utils import process_model_params

# data split: data/pv/fold<PV_FOLD>/{train,val,test}, written by tools/pv_patch_split.py
fold = int(os.environ.get('PV_FOLD', 0))
data_root = 'data/pv/fold{}'.format(fold)

# training hparam
max_epoch = 200
early_stopping_patience = 20
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.00025
backbone_lr = 1e-4
backbone_weight_decay = 0.00025
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "manet_seresnext26d"
weights_path = "model_weights/pv/fold{}/{}".format(fold, weights_name)
test_weights_name = "manet_seresnext26d"
log_name = 'pv/fold{}/{}'.format(fold, weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = MANet(num_classes=num_classes, backbone_name='seresnext26d_32x4d', fom=False)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

# define the dataloader

train_dataset = PVDataset(data_root=os.path.join(data_root, 'train'), mode='train', transform=train_aug)
val_dataset = PVDataset(data_root=os.path.join(data_root, 'val'), transform=val_aug)
test_dataset = PVDataset(data_root=os.path.join(data_root, 'test'), mode='test', transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-7)
