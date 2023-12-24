conda activate frozenrecon

# # change GT_ROOT if you would like to use gt priors
# export GT_ROOT='PATH_TO_GT_DATA_ROOT'

# # FrozenRecon with datasets
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --scene_name classroom_0004 

# # FrozenRecon with partial gt labels
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --save_suffix gt_intrinsic --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_pose_flag --save_suffix gt_pose --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_depth_flag --save_suffix gt_depth --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_pose_flag --save_suffix gt_intrinsic_gt_pose --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_pose_flag --gt_depth_flag --save_suffix gt_depth_gt_pose --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_depth_flag --save_suffix gt_intrinsic_gt_depth --scene_name classroom_0004
# python src/optimize.py --dataset_name NYUDepthVideo --gt_root $GT_ROOT --gt_intrinsic_flag --gt_pose_flag --gt_depth_flag --save_suffix gt_intrinsic_gt_depth_gt_pose --scene_name classroom_0004

# # FrozenRecon with in-the-wild video input
# python src/optimize.py --video_path PATH_TO_VIDEO --scene_name SCENE_NAME

# FrozenRecon with in-the-wild images input
python src/optimize.py --img_root PATH_TO_IMG_FOLDER --scene_name SCENE_NAME

# # Outdoor Scenes
# python src/optimize.py --dataset_name NYUDepthVideo --scene_name SCENE_NAME --gt_root $GT_ROOT --gt_intrinsic_flag --scene_name "2011_09_26_drive_0001_sync" --outdoor_scenes 