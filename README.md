<div align="center">

<h1>[ICCV2023] 🧊 FrozenRecon: Pose-free 3D Scene Reconstruction with Frozen Depth Models </h1>

[Guangkai Xu](https://github.com/guangkaixu/)<sup>1*</sup>, &nbsp; 
[Wei Yin](https://yvanyin.net/)<sup>2*</sup>, &nbsp; 
[Hao Chen](https://stan-haochen.github.io/)<sup>3</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>3,4</sup>, &nbsp;
[Kai Cheng](https://cklibra.github.io/)<sup>1</sup>, &nbsp;
[Feng Zhao](https://scholar.google.co.uk/citations?user=r6CvuOUAAAAJ&hl=en/)<sup>1</sup>

<sup>1</sup>University of Science and Technology of China &nbsp;&nbsp; 
<sup>2</sup>DJI Technology &nbsp;&nbsp; 
<sup>3</sup>Zhejiang University &nbsp;&nbsp; 
<sup>4</sup>Ant Group

### [Project Page](https://aim-uofa.github.io/FrozenRecon/) | [arXiv](https://arxiv.org/abs/2308.05733) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_FrozenRecon_Pose-free_3D_Scene_Reconstruction_with_Frozen_Depth_Models_ICCV_2023_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Xu_FrozenRecon_Pose-free_3D_ICCV_2023_supplemental.pdf)

#### ⏳ Reconstruct your pose-free video with 🧊 FrozenRecon in around a quarter of an hour! ⌛
</div>
<div align="center">
<img width="800" alt="image" src="figs/frozenrecon-demo.png">
</div>
We propose a novel test-time optimization approach that can transfer the robustness of affine-invariant depth models such as LeReS to challenging diverse scenes while ensuring inter-frame consistency, with only dozens of parameters to optimize per video frame. Specifically, our approach involves freezing the pre-trained affine-invariant depth model's depth predictions, rectifying them by optimizing the unknown scale-shift values with a geometric consistency alignment module, and employing the resulting scale-consistent depth maps to robustly obtain camera poses and camera intrinsic simultaneously. Dense scene reconstruction demo is shown as below.


## Prerequisite

### Pre-trained Checkpoints
In this project, we use LeReS to predict affine-invariant depth maps. Please download the pre-trained [checkpoint](https://pan.baidu.com/s/1o2oVMiLRu770Fdpa65Pdbw?pwd=g3yi) of [LeReS](https://github.com/aim-uofa/AdelaiDepth/tree/main), and place it in `FrozenRecon/LeReS/res101.pth`. If optimize outdoor scenes, the [checkpoint](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw) of [Segformer](https://github.com/NVlabs/SegFormer) should also be downloaded and placed in `FrozenRecon/SegFormer/segformer.b3.512x512.ade.160k.pth`

### Demo Data
We provide one demo data for each scene, and another in-the-wild video captured from iPhone14 Pro without any lidar sensor information. Download from [BaiduNetDisk](https://pan.baidu.com/s/1bSH8G0-ZQ8LpytZdZ6zIQw?pwd=e6hc), and place it in `FrozenRecon/demo_data`.

### Installation
```bash
git clone --recursive https://github.com/aim-uofa/FrozenRecon.git
cd FrozenRecon
conda create -y -n frozenrecon python=3.8
conda activate frozenrecon
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html # pytorch 1.7.1 for SegFormer
pip install -r requirements.txt

# (Optional) For outdoor scenes, we recommand to mask the sky regions and cars (potential dynamic objects)
pip install timm==0.3.2
pip install --upgrade mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch171/index.html
# pip install "mmsegmentation==0.11.0"
pip install ipython attr 
git clone https://github.com/NVlabs/SegFormer.git
cd SegFormer && pip install -e . & cd ..
# After installing SegFormer, please downlaod segformer.b3.512x512.ade.160k.pth checkpoint following https://github.com/NVlabs/SegFormer, and place it in SegFormer/

# (Optional) Install lietorch. It can make optimization faster.
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch && python setup.py install & cd ..
```

## Optimization

### 1. In-the-wild Video Input
```bash
# Take demo data as an example
python src/optimize.py --video_path demo_data/IMG_8765.MOV

# # For self-captured videos
# python src/optimize.py --video_path PATH_TO_VIDEO --scene_name SCENE_NAME
```

### 2. In-the-wild Extracted Images Input
```bash
python src/optimize.py --img_root PATH_TO_IMG_FOLDER --scene_name SCENE_NAME
```

### 3. Datasets (Optional with GT Priors)
```bash
# Export ground-truth data root here.
export GT_ROOT='./demo_data' # PATH_TO_GT_DATA_ROOT, you can download demo_data following "Data" subsection.

# FrozenRecon with datasets, take NYUDepthVideo classroom_0004 as example.
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --scene_name classroom_0004 

# FrozenRecon with GT priors.
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --save_suffix gt_intrinsic --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_pose_flag --save_suffix gt_pose --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_depth_flag --save_suffix gt_depth --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_pose_flag --save_suffix gt_intrinsic_gt_pose --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_pose_flag --gt_depth_flag --save_suffix gt_depth_gt_pose --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_depth_flag --save_suffix gt_intrinsic_gt_depth --scene_name classroom_0004
python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_pose_flag --gt_depth_flag --save_suffix gt_intrinsic_gt_depth_gt_pose --scene_name classroom_0004
```

### 4. Outdoor Scenes
```bash
export GT_ROOT='./demo_data' # PATH_TO_GT_DATA_ROOT, you can download demo_data following "Data" subsection.
# We suggest to use GT intrinsic for stable optimization.
python src/optimize.py --dataset_name NYUDepthVideo --scene_name SCENE_NAME --gt_root $GT_ROOT --gt_intrinsic_flag --scene_name 2011_09_26_drive_0001_sync --outdoor_scenes 
```


## 🎫 License

For non-commercial use, this code is released under the [LICENSE](LICENSE).
For commercial use, please contact Chunhua Shen.

## 🖊️ Citation

If you find this project useful in your research, please consider cite:


```BibTeX
@inproceedings{xu2023frozenrecon,
  title={FrozenRecon: Pose-free 3D Scene Reconstruction with Frozen Depth Models},
  author={Xu, Guangkai and Yin, Wei and Chen, Hao and Shen, Chunhua and Cheng, Kai and Zhao, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9310--9320},
  year={2023}
}
```
