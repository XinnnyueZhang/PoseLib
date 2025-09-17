for sceneID in {0,1,3}; do
    python fisheye_experiments/runhloc_remove_covisibilty.py --nSample 50 --sceneID $sceneID --num_remove_covisibilty 10
done

for sceneID in {2,4,5,6,7,8,9}; do
    python fisheye_experiments/runhloc_remove_covisibilty.py --nSample 20 --sceneID $sceneID --num_remove_covisibilty 10
done