# TR3D_PointAugDGCNN
PointAugment and DGCNN tree species classification for [3DForEcoTech Tr3D Species Benchmark](https://github.com/stefp/Tr3D_species). Model achieved the highest F1 score of 0.767 during the training/validation phase.

### Confusion Matrix of Best Performance
![Confusion Matrix](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/output/confusion_matrix.png)

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
| root | [species_classes.csv](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/species_classes.csv) | csv of species and associated class number |
| root | [main.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/main.py) | Main script to run the model |
| augment | [augmentor.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/augment/augmentor.py) | The augmentor (generator) model |
| checkpoints/dgcnn_pointaugment_4096 | [f1.png](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/f1.png) | Image of the validation and training F1 scores |
| checkpoints/dgcnn_pointaugment_4096 | [loss_f1.csv](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/loss_f1.csv) | csv of the augmentor losses, classifier losses, and F1 scores |
| checkpoints/dgcnn_pointaugment_4096 | [run.log](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/run.log) | Run log of printed outputs |
| checkpoints/dgcnn_pointaugment_4096/models | [best_model.t7](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/models/best_model.t7) | Pytorch model weights of the best run |
| checkpoints/dgcnn_pointaugment_4096/output | [confusion_matrix.png](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/output/confusion_matrix.png) | Image of confusion matrix of best model |
| checkpoints/dgcnn_pointaugment_4096/output | [output.csv](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/checkpoints/dgcnn_pointaugment_4096/output/output.csv) | csv of true and predicted classes |
| common | [loss_utils.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/common/loss_utils.py) | The loss fucntions for the adapted models |
| models | [dgcnn.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/tree/main/models) | Pytorch Implementation of DGCNN |
| utils | [augmentation.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/augmentation.py) | A script that performs the manual augmentations on point clouds |
| utils | [resample_point_clouds.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/resample_point_clouds.py) | A script that performs resampling of point clouds (current methods are fps and cluster fps) |
| utils | [send_telegram.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/send_telegram.py) | Functions that send telegram messages + photos |
| utils | [tools.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/tools.py) | A script of useful functions |
| utils | [train.py](https://github.com/Brent-Murray/TR3D_PointAugDGCNN/blob/main/utils/train.py) | A script that defines the training/validation/testing process |
