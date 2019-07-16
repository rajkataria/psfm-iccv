declare -a datasets=("ETH3D/boulders" "ETH3D/courtyard" "ETH3D/exhibition_hall" "UIUCTag/ece_floor5_wall" "UIUCTag/ece_floor3_loop_ccw" "TanksAndTemples/Meetingroom")

for i in "${datasets[@]}"; do
  for j in `seq 0 2`; do
    dset=(${i//// });
    root=${dset[0]};
    sequence=${dset[1]};
    c=$(echo "/home/ubuntu/Results/completed-classifier-datasets-bruteforce/$root/$sequence/");

    bash ./scripts/run_reconstructions.sh $c $j > $c/output-$j.log 2>&1;
  done;
done