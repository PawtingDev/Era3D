# python test_mvdiffusion_unclip.py --config configs/test_unclip-512-6view.yaml \
#     pretrained_model_name_or_path='pengHTYX/MacLab-Era3D-512-6view' \
#     validation_dataset.crop_size=512 \
#     validation_dataset.root_dir=avatar \
#     seed=42 \
#     save_dir='mv_res'  \
#     save_mode='rgb'


python test_mvdiffusion_unclip.py --config configs/test_unclip-512-6view.yaml \
    pretrained_model_name_or_path='pengHTYX/wonderhuman' \
    validation_dataset.crop_size=512 \
    validation_dataset.root_dir=avatar \
    seed=42 \
    save_dir='mv_res_'  \
    save_mode='rgb'