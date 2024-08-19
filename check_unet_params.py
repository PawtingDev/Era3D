from diffusers import DDPMScheduler, AutoencoderKL, EMAModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

# base model
pretrained_model_name_or_path = 'pengHTYX/MacLab-Era3D-512-6view'
# fine-tuning model
# pretrained_unet_path = 'outputs/wonder3D-joint-0327/unet-15000'
# pretrained_unet_path = 'outputs/wonder3D-joint/unet-150'
# pretrained_unet_path = 'outputs/wonder3D-joint/checkpoint'
pretrained_unet_path = 'pengHTYX/MacLab-Era3D-512-6view'

trainable_modules = ('regressor')
revision = 'main'
unet_from_pretrained_kwargs = {
    "unclip": True,
    "sdxl": False,
    "num_views": 6,
    "sample_size": 64,
    "zero_init_conv_in": False,  # modify
    "regress_elevation": True,
    "regress_focal_length": True,
    "camera_embedding_type": 'e_de_da_sincos',
    "projection_camera_embeddings_input_dim": 4,  # 2 for elevation and 6 for focal_length
    "zero_init_camera_projection": False,
    "num_regress_blocks": 3,
    "cd_attention_last": False,
    "cd_attention_mid": False,
    "multiview_attention": True,
    "sparse_mv_attention": True,
    "selfattn_block": 'self_rowwise',
    "mvcd_attention": True,
    "use_dino": False,
}

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path,
                                                              subfolder="image_encoder", revision=revision)
feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor",
                                                       revision=revision)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
print("load pre-trained unet from ", pretrained_unet_path)
unet = UNetMV2DConditionModel.from_pretrained(pretrained_unet_path, subfolder="unet", revision=revision,
                                              **unet_from_pretrained_kwargs)

ema_unet = EMAModel(unet.parameters(), model_cls=UNetMV2DConditionModel, model_config=unet.config)

for name, module in unet.named_modules():
    # print(name, module)
    if name.endswith(trainable_modules):
    #     print(name)
    #
        for params in module.parameters():
            print("trainable params: ", params, params.shape)
            exit()
    #         # params.requires_grad = True
# for name, param in unet.named_parameters():
#     print(name, param.shape, param.dtype)
