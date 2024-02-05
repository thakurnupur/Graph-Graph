# Graph(Graph): A Nested Graph-Based Framework for Early Accident Anticipation (WACV 2024)
Official PyTorch implementation of WACV 2024 paper - [Graph(Graph): A Nested Graph-Based Framework for Early Accident Anticipation](https://openaccess.thecvf.com/content/WACV2024/papers/Thakur_GraphGraph_A_Nested_Graph-Based_Framework_for_Early_Accident_Anticipation_WACV_2024_paper.pdf)

## Setup  
- Python 3.9
- CUDA - 11.8

Create a conda environment and install all the dependencies using the following commands: 
```python
pip install -r requirements.txt
```

## Dataset 
Download the data from [link](link) and place it in `data` folder. There are 3 folders for each dataset: 
- `obj_feat`: The object data for both datasets is downloaded from [1].
- `i3d_feat`: We extracted I3D features for all the frames using the code and pretrained model available at [2].
- `frames_stat`: This contains the resolution for every frame of a video.  

## Training
To train use the following commands: 
DAD dataset- 
```python
python train_dad.py --test_only 0  
```

CCD dataset- 
```python
python train_ccd.py --test_only 0 
```

The models will be saved in the `model_checkpoints/'dataset-name'` folder. 

## Evaluation 
Download our trained models for both the datasets from [here](here). Place them in `model_checkpoints` folder. 

Use the following command for evaluation: 

DAD dataset- 
```python
python train_dad.py --test_only 1 --checkpoint_model "model_checkpoints/dad_model.pth" 
```

CCD dataset- 
```python
python train_ccd.py --test_only 1 --checkpoint_model "model_checkpoints/ccd_model.pth"
```

## References
1. [https://github.com/Cogito2012/UString](https://github.com/Cogito2012/UString)
2. [https://github.com/piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
3. [https://github.com/eriklindernoren/Action-Recognition](https://github.com/eriklindernoren/Action-Recognition)

## Citation
If you find this code and our work helpful, please cite our paper:
```
@inproceedings{thakur2024graph,
  title={Graph (Graph): A Nested Graph-Based Framework for Early Accident Anticipation},
  author={Thakur, Nupur and Gouripeddi, PrasanthSai and Li, Baoxin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7533--7541},
  year={2024}
}
```
In case of any questions, feel free to reach out at [nsthaku1@asu.edu](nsthaku1@asu.edu) or open issues on the repo.
