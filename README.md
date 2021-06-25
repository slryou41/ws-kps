# Weakly Supervised Keypoint Discovery
PyTorch implementation of Weakly Supervised Keypoint Discovery, Serim Ryou and Pietro Perona

1. Download the datasets
2. Set the dataloader path
3. Train the model with the script:

```python
python train.py [path_to_dataset] --checkpoint [path_to_checkpoint] --gpu [gpu_id] --lr 0.001 --batch-size [batch_size] --nkpts [number_of_keypoints] --nclass [number_of_class_category] --dataset [dataset_name]
```