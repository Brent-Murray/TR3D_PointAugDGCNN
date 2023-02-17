# TR3D_PointAugDGCNN
PointAugment and DGCNN Classification Benchmark for TR3D

Models
----
| Model | Description | Reference |
| ----- | ----------- | --------- |
| PointAugment | Adaptation of PointAugment model for tree species point cloud augmentations | [(Li et al., 2020)](https://arxiv.org/abs/2002.10876) |
| DGCNN | Adaptation of Dynamic Graph Covolutional Neural Network (DGCNN) model for tree species classification on point clouds | [(Wang et al., 2019)](https://arxiv.org/abs/1801.07829) |

Contents
----
| Folder | File | Description |
| ------ | ---- | ----------- |
| |[main.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/main.py) | Main script to run the model |
| augment | [augmentor.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/augment/augmentor.py) | The augmentor (generator) model |
| common | [loss_utils.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/common/loss_utils.py) | The loss fucntions for the adapted models |
| models | [dgcnn.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/tree/main/models) | Pytorch Implementation of DGCNN |
| utils | [augmentatoin.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/augmentation.py) | A script that performs the manual augmentations on point clouds |
| utils | [resample_point_clouds.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/resample_point_clouds.py) | A script that performs resampling of point clouds (current methods are fps and cluster fps) |
| utils | [send_telegram.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/send_telegram.py) | Functions that send telegram messages + photos |
| utils | [tools.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/tools.py) | A script of useful functions |
| utils | [train.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/train.py) | A script that defines the training/validation/testing process |
