for i in ./*/; 
do
	python2.7 associate.py $i/rgb.txt $i/groundtruth.txt > $i/rgb-gt.txt;
done
