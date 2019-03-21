# declare -a datasets=("TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant" "TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_kidnap" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2" "TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static" "TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz")
# declare -a datasets=("ETH3D/exhibition_hall" "ETH3D/boulders" "ETH3D/lecture_room" "UIUCTag/ece_floor2_hall/" "UIUCTag/ece_floor3_loop_ccw/" "UIUCTag/ece_floor3_loop_cw/")
# declare -a datasets=("ETH3D/boulders" "ETH3D/lecture_room" "UIUCTag/ece_floor2_hall/")


# declare -a datasets=("ETH3D/boulders" "ETH3D/courtyard" "ETH3D/lecture_room" "ETH3D/exhibition_hall" "TanksAndTemples/Meetingroom")
# declare -a datasets=("TanksAndTemples-1/Truck" "UIUCTag-1/ece_floor3_loop_ccw" "UIUCTag-1/ece_floor3_loop_cw" "UIUCTag-1/ece_floor5_wall")
# for i in "${datasets[@]}"
# do
#     c=$(echo "/hdd/Research/psfm-iccv/data/variance-exps/"$i);
#     echo $c
#     bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1
# done


# for c in /hdd/Research/psfm-iccv/data/variance-exps/*/*/;
# do
#   bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1;
# done

# for c in /hdd/Research/psfm-iccv/data/variance-exps/ETH3D-*/boulders/;
# do
#   bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1;
# done

# for c in /hdd/Research/psfm-iccv/data/variance-exps/UIUCTag-*/ece_floor3_loop_c*/;
# do
#   bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1;
# done

# declare -a datasets=("ETH3D/boulders" "UIUCTag/ece_floor5_wall")
# declare -a datasets=("ETH3D/boulders")
# declare -a datasets=("UIUCTag/ece_floor5_wall")
# declare -a datasets=("ETH3D/exhibition_hall")
# declare -a datasets=("ETH3D/boulders" "ETH3D/exhibition_hall")

# declare -a datasets=("ETH3D/boulders" "ETH3D/courtyard" "ETH3D/lecture_room" "ETH3D/exhibition_hall" "UIUCTag/ece_floor5_wall" "UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw" "TanksAndTemples/Meetingroom")
declare -a datasets=("ETH3D/lecture_room" "ETH3D/exhibition_hall" "UIUCTag/ece_floor5_wall" "UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw" "TanksAndTemples/Meetingroom")
# declare -a datasets=("UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw")
# declare -a datasets=("ETH3D/exhibition_hall" "ETH3D/courtyard" "UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw")

for i in "${datasets[@]}"; do
  for j in `seq 1 3`; do
    dset=(${i//// });
    root=${dset[0]};
    sequence=${dset[1]};
    c=$(echo "/hdd/Research/psfm-iccv/data/variance-exps/$root-$j/$sequence/");
    # echo $c
    bash ./scripts/run_reconstructions.sh $c > $c/output.log 2>&1;
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction_gt.json
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction-imc-False-wr-False-colmapr-False-gm-False-wfm-False-imt-False-spp-False.json
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction-imc-False-wr-False-colmapr-True-gm-False-wfm-False-imt-False-spp-False.json
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction-imc-False-wr-False-colmapr-True-gm-True-wfm-False-imt-False-spp-False.json
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction-imc-False-wr-False-colmapr-True-gm-True-wfm-False-imt-False-spp-True.json
    # open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/$root-$j/$sequence/reconstruction-imc-False-wr-False-colmapr-True-gm-False-wfm-False-imt-False-spp-True.json
  done;
done