DEPTH_LAYERS = 50#resnet50
POSE_LAYERS = 18#resnet18
FRAME_IDS = [0, -1, 1] #0 refers to current frame, -1 and 1 refer to temperally adjacent frames, 's' refers to stereo adjacent frame.
IMGS_PER_GPU = 2 #the number of images fed to each GPU
HEIGHT = 544 #input image height
WIDTH = 960 #input image width

data = dict(
    name = 'mydata',#dataset name
    split = 'mydata',#training split name
    height = HEIGHT,
    width = WIDTH,
    frame_ids = FRAME_IDS,
    in_path = './datasets/my_dataset/image_undistort', # path to raw data
    gt_depth_path = '', # path to gt data
    png = False,  # image format
    stereo_scale = True if 's' in FRAME_IDS else False,
)

model = dict(
    name = 'mono_fm',# select a model by name
    depth_num_layers = DEPTH_LAYERS,
    pose_num_layers = POSE_LAYERS,
    frame_ids = FRAME_IDS,
    imgs_per_gpu = IMGS_PER_GPU,
    height = HEIGHT,
    width = WIDTH,
    scales = [0, 1, 2, 3],# output different scales of depth maps
    min_depth = 0.1, # minimum of predicted depth value
    max_depth = 100.0, # maximum of predicted depth value
    depth_pretrained_path = './weights/resnet{}.pth'.format(DEPTH_LAYERS),# pretrained weights for resnet
    pose_pretrained_path =  './weights/resnet{}.pth'.format(POSE_LAYERS),# pretrained weights for resnet
    extractor_pretrained_path = './log/my_autoencoder/latest.pth', # pretrained weights for autoencoder
    automask = False if 's' in FRAME_IDS else True,
    disp_norm = False if 's' in FRAME_IDS else True,
    perception_weight = 1e-3,
    smoothness_weight = 1e-3,
)

# resume_from = '/node01_data5/monodepth2-test/model/ms/ms.pth'#directly start training from provide weights
resume_from = None
finetune = './weights/fm_depth.pth'
total_epochs = 40
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = 4
validate = False

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20,30],
    gamma=0.5,
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]