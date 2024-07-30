python inference.py \
--device 'cuda:0' \
--dataset_dir /path/to/dataset \
--model_dir /pretrained_weights/Cityscapes/lamda3072/ckpt.cityscapes.pth.tar \
--output_dir /path/to/output \
--data_name 'cityscapes' \
--choose_entropy_estimation \
--network 'Main'