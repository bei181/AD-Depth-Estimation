# python run.py --Final \
# --data_dir /home/tata/Project/dataset/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data \
# --output_dir ./output \
# --depthNet 0

# python run.py --R0 \
# --data_dir /home/tata/Project/dataset/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data \
# --output_dir ./output/output_midas_R0 \
# --depthNet 0

# python run.py --R20 \
# --data_dir /home/tata/Project/dataset/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data \
# --output_dir ./output/output_midas_R20 \
# --depthNet 0

python run.py --colorize_results --Final \
--data_dir /home/tata/Project/dataset/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data \
--output_dir ./output/output_midas_color \
--depthNet 0