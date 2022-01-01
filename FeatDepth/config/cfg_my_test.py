split = 'exp'
dataset = 'folder'
height = 544
width = 960
disparity_smoothness = 1e-3
scales = [0, 1, 2, 3, 4]
min_depth = 0.1
max_depth = 100.0
frame_ids = [0, -1, 1]
learning_rate = 1e-4
depth_num_layers = 50
pose_num_layers = 18
total_epochs = 45
device_ids = range(8)

depth_pretrained_path = './weights/resnet{}.pth'.format(depth_num_layers)
pose_pretrained_path =  './weights/resnet{}.pth'.format(pose_num_layers)

in_path = './datasets/my_dataset/image_undistort'
gt_depth_path = ''
checkpoint_path = '/node01_data5/monodepth2-test/model/refine/smallfigure.pth'
imgs_per_gpu = 2
workers_per_gpu = 4

model = dict(
    name = 'mono_fm',# select a model by name
    depth_num_layers = depth_num_layers,
    pose_num_layers = pose_num_layers,
    frame_ids = frame_ids,
    imgs_per_gpu = imgs_per_gpu,
    height = height,
    width = width,
    scales = [0, 1, 2, 3],# output different scales of depth maps
    min_depth = 0.1, # minimum of predicted depth value
    max_depth = 100.0, # maximum of predicted depth value
    depth_pretrained_path = './weights/resnet{}.pth'.format(depth_num_layers),# pretrained weights for resnet
    pose_pretrained_path =  './weights/resnet{}.pth'.format(pose_num_layers),# pretrained weights for resnet
    extractor_pretrained_path = './weights/autoencoder.pth', # pretrained weights for autoencoder
    automask = False if 's' in frame_ids else True,
    disp_norm = False if 's' in frame_ids else True,
    perception_weight = 1e-3,
    smoothness_weight = 1e-3,
)

validate = False

png = False
scale_invariant = False
plane_fitting = False
finetune = False
perception = False
focus_loss = False

scale_invariant_weight = 0.01
plane_fitting_weight = 0.0001
perceptional_weight = 0.001

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[15,25,35],
    gamma=0.5,
    )

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook'),])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]