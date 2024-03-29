<h2 align="center">Location-aware Single Image Reflection Removal</h2>

![Examples](./doc_gif/demo.gif)

<p align='center'>
The shown images are provided by the datasets from <a href="https://github.com/JHL-HUST/IBCLN">IBCLN</a>, <a href="https://github.com/Vandermode/ERRNet">ERRNet</a>, <a href="https://sir2data.github.io/">SIR<sup>2</sup></a> and the Internet images.
</p>
<p align='center'>
The code and pretrained model for our paper: <b>Location-aware Single Image Reflection Removal</b> [<a href="https://arxiv.org/pdf/2012.07131.pdf">Arxiv Preprint</a>]
</p>

---

## Prerequisites
Our code has been tested under the following platform and environment:
- Ubuntu. CPU or NVIDIA GPU + CUDA, CuDNN
- Python 3.7.3, Pytorch 1.2.0
- Requirements: numpy, tqdm, Pillow, dominate, scikit-image

## Setup
- Clone or Download this repo
- ```$ cd Location-aware-SIRR```
- ```$ mkdir model```
- Download the pretrained model [here](https://drive.google.com/file/d/1TjH5YUBC-cDt09tDXhE5GbeO5FuO7FVZ/view)
- Move the downloaded model(```model.pth```) to ```./model``` folder

## Usage
- The example test images are provided in ```./test_images/blend``` folder
- If you have ground truth blackground images, put them into ```./test_images/transmission``` folder ( Note that the same pair of images need to be named the same ).
- Run ```python3 inference.py```
- The inference results are in the ```./results``` folder

   
## Citation
If you find our work helpful to your research, please cite our paper.
```bibtex
@article{dong2020location,
  author = {Zheng Dong and Ke Xu and Yin Yang and Hujun Bao and Weiwei Xu and Rynson W.H. Lau},
  title = {Location-aware Single Image Reflection Removal},
  journal={ArXiv},
  volume={abs/2012.07131},
  year = {2020}
}
```

