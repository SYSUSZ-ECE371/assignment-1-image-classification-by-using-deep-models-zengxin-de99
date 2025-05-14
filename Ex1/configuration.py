_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_140e.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
    init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5),
)

# dataset settings
data_preprocessor = dict(
    num_classes=5,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_prefix='data/imagenet/flower_dataset/train',
        ann_file='flower_dataset/train.txt',
        classes='data/imagenet/flower_dataset/classes.txt',
    )
)
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_prefix='data/imagenet/flower_dataset/val',
        ann_file='flower_dataset/val.txt',
        classes='data/imagenet/flower_dataset/classes.txt',
    )
)
val_evaluator = dict(type='Accuracy', topk=1)

# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=20)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
)