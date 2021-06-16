gpu=$1
temp=$2
qus=$3

save_dir=data/retriever_result_${temp}
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=${gpu} python train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 12 \
	--do_lower_case \
	--train_file "data/retriever/nq-train.json" \
	--dev_file "data/retriever/nq-dev.json" \
	--output_dir ${save_dir} \
	--learning_rate 2e-05 \
	--num_train_epochs 15 \
	--dev_batch_size 12 \
	--val_av_rank_start_epoch 10 --number_of_qus ${qus} --temperature ${temp}

