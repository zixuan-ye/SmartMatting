bench='aim'
ckpt='ckpt/SMat.pth'
device='cuda:1'
config='configs/smartmatting.py'

python inference_dataset.py --benchmark $bench --prompt box --output outputs/$bench-box --checkpoint $ckpt --device $device --config $config
python inference_dataset.py --benchmark $bench --prompt none --output outputs/$bench-none --checkpoint $ckpt --device $device --config $config
# python inference_dataset.py --benchmark $bench --prompt scribble --output outputs/$bench-scribble --checkpoint $ckpt --device $device --config $config
# python inference_dataset.py --benchmark $bench --prompt point --output outputs/$bench-point --checkpoint $ckpt --device $device --config $config