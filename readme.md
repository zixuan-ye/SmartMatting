
## ___***[CVPR2024] Unifying Automatic and Interactive Matting with Pretrained ViTs***___

## 🔆 Introduction




## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n SMat python=3.8
conda activate SMat
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
```


## 💫 Inference 
### Local Gradio demo

1. Download the pretrained models  and put them in the './ckpt' dir.
2. Input the following commands in terminal.
```bash
  sh app_inference.sh
```


---
## 😉 Citation
```
@inproceedings{ye2024unifying,
      title={Unifying Automatic and Interactive Matting with Pretrained ViTs}, 
      author={Ye, Zixuan and Liu, Wenze and Guo, He and Liang, Yujia and Hong, Chaoyi and Lu, Hao and Cao, Zhiguo},
      booktitle={Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```


## 🤗 Acknowledgements
Our codebase builds on [ViTMatte](https://github.com/hustvl/ViTMatte). 
Thanks the authors for sharing their awesome codebases! 


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****