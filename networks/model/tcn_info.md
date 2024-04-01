## 结构
https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/recognizers/base.py

"""Base class for recognizers.

    Args:
        backbone (Union[ConfigDict, dict]): Backbone modules to
            extract feature.
        cls_head (Union[ConfigDict, dict], optional): Classification head to
            process feature. Defaults to None.
        neck (Union[ConfigDict, dict], optional): Neck for feature fusion.
            Defaults to None.
        train_cfg (Union[ConfigDict, dict], optional): Config for training.
            Defaults to None.
        test_cfg (Union[ConfigDict, dict], optional): Config for testing.
            Defaults to None.
        data_preprocessor (Union[ConfigDict, dict], optional): The pre-process
           config of :class:`ActionDataPreprocessor`.  it usually includes,
            ``mean``, ``std`` and ``format_shape``. Defaults to None.
"""

## TSN
https://github.com/open-mmlab/mmaction2/blob/main/configs/_base_/models/tsn_r50.py
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)


## tsn_head
https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/heads/tsn_head.py


## data_preprocessor
https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/data_preprocessors/data_preprocessor.py

