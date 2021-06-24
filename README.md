# Weakly Supervised Keypoint Discovery
PyTorch implementation of Weakly Supervised Keypoint Discovery, Serim Ryou and Pietro Perona

1. Download the datasets
2. Set the dataloader path
3. Train the model with the script:
``
python train.py [data_path] --checkpoint [path_to_checkpoint] --gpu [gpu_id] --lr 0.001 --batch-size [batch_size] --nkpts [number of keypoints] --nclass [number of class category] 
``