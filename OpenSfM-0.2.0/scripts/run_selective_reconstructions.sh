declare -a datasets=("TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_kidnap" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz")

for i in "${datasets[@]}"
do
        c=$(echo "/home/ubuntu/Results/non-bruteforce-matching/ours/"$i);
	echo $c
	bash ./scripts/clean_reconstruction.sh $c
	bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1
done

