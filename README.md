## Mask RCNN 性能优化 Log

## Baseline

```bash
python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml -o LearningRate.base_lr=0.0001 log_iter=1 use_gpu=True save_dir=./test_tipc/output/mask_rcnn_r50_1x_coco/benchmark_train/norm_train_gpus_0_autocast_fp32 epoch=3 eval=True TrainReader.batch_size=2 filename=mask_rcnn_r50_1x_coco TrainReader.shuffle=False --enable_ce=True
```

## AMP

```bash
python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml -o LearningRate.base_lr=0.0001 log_iter=1 use_gpu=True save_dir=./test_tipc/output/mask_rcnn_r50_1x_coco/benchmark_train/norm_train_gpus_0_autocast_fp32 epoch=3 amp=True eval=True TrainReader.batch_size=2 filename=mask_rcnn_r50_1x_coco TrainReader.shuffle=False --enable_ce=True
```

## AMP+NHWC

首先需要对于修改配置文件

```bash
FLAGS_cudnn_batchnorm_spatial_persistent=1 python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml -o LearningRate.base_lr=0.0001 log_iter=1 use_gpu=True save_dir=./test_tipc/output/mask_rcnn_r50_1x_coco/benchmark_train/norm_train_gpus_0_autocast_fp32 epoch=3 amp=True eval=True TrainReader.batch_size=2 filename=mask_rcnn_r50_1x_coco TrainReader.shuffle=False --enable_ce=True
```

