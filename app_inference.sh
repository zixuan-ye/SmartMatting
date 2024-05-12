CUDA_VISIBLE_DEVICES=0
python inference_app.py \
    --config-dir './configs/smartmatting.py' \
    --checkpoint-dir './ckpt/SMat.pth' \
    --device 'cuda'