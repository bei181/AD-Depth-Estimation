prediction_dir="/home/tata/Project/struct2depth/output/kitti_data/2011_09_26/2011_09_26_drive_0019_sync/image_02/data"
model_ckpt="/home/tata/Project/models/struct2depth/kitti/model-199160"
handle_motion="false"
size_constraint_weight="0" # This must be zero when not handling motion.

# If running on KITTI, set as follows:
data_dir="/home/tata/Project/dataset/kitti_data_processed"
triplet_list_file="$data_dir/kitti_triplet_list_online.txt"
triplet_list_file_remains=None # "$data_dir/test_files_eigen_triplets_remains.txt"
ft_name="kitti"

# # If running on Cityscapes, set as follows:
# data_dir="CITYSCAPES_SEQ2_LR_TEST/" # Set for Cityscapes
# triplet_list_file="/CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets.txt"
# triplet_list_file_remains="CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets_remains.txt"
# ft_name="cityscapes"

python optimize.py \
  --logtostderr \
  --output_dir $prediction_dir \
  --data_dir $data_dir \
  --triplet_list_file $triplet_list_file \
  --triplet_list_file_remains $triplet_list_file_remains \
  --ft_name $ft_name \
  --model_ckpt $model_ckpt \
  --file_extension png \
  --handle_motion $handle_motion \
  --size_constraint_weight $size_constraint_weight