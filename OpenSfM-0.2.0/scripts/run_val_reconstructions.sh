declare -a yan_datasets=("Yan/books" "Yan/cereal" "Yan/cup" "Yan/desk" "Yan/oats" "Yan/street")
declare -a eth3d_datasets=("ETH3D/boulders" "ETH3D/courtyard" "ETH3D/exhibition_hall" "ETH3D/lecture_room" "ETH3D/living_room" "ETH3D/pipes" "ETH3D/terrains")
declare -a uiuctag_datasets=("UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw" "UIUCTag/ece_floor5" "UIUCTag/ece_floor5_wall" "UIUCTag/yeh_night_all" "UIUCTag/yeh_night_backward" "UIUCTag/yeh_night_forward")
declare -a duplicate_structures_datasets=("DuplicateStructures/alexander_nevsky_cathedral" "DuplicateStructures/arc_de_triomphe" "DuplicateStructures/brandenburg_gate" "DuplicateStructures/cereal" 
  "DuplicateStructures/church_on_spilled_blood" "DuplicateStructures/indoor" "DuplicateStructures/radcliffe_camera" "DuplicateStructures/street")
declare -a tanksandtemples_datasets=("TanksAndTemples/Ballroom" "TanksAndTemples/Barn" "TanksAndTemples/Meetingroom" "TanksAndTemples/Museum")

datasets=( "${yan_datasets[@]}" "${eth3d_datasets[@]}" "${uiuctag_datasets[@]}" "${duplicate_structures_datasets[@]}" "${tanksandtemples_datasets[@]}" )

for i in "${datasets[@]}"; do
  for j in `seq 0 0`; do
    dset=(${i//// });
    root=${dset[0]};
    sequence=${dset[1]};
    c=$(echo "/home/ubuntu/Results/completed-classifier-datasets-bruteforce/$root/$sequence/");
    # c=$(echo "/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/$root/$sequence/");
    # echo $c
    bash ./scripts/run_reconstructions.sh $c $j > $c/output-$j.log 2>&1;
  done;
done