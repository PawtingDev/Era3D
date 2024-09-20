# accelerate launch --config_file node_config/1gpu.yaml train_wonderhuman.py \
#                   --config configs/train-512-6view.yaml

accelerate launch --config_file node_config/8gpu.yaml train_wonderhuman.py \
                  --config configs/train-512-6view.yaml