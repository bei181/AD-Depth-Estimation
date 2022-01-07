output_dir="/home/tata/Project/struct2depth/output"
model_checkpoint="/home/tata/Project/models/struct2depth/kitti/model-199160"

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion true \
    --input_list_file /home/tata/Project/dataset/kitti_test.txt \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint