python inference.py \
--device 'cuda:0' \
--dataset_dir /path/to/dataset \
--model_dir /pretrained_weights/Instereo2K/lamda2048/ckpt.instereo2k.pth.tar \
--output_dir /path/to/output \
--data_name 'instereo2k' \
--choose_entropy_estimation \
--network 'Main'