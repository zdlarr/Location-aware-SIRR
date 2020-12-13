<h2 align="center">Location-aware Single Image Reflection Removal</h2>

<div align='center'>
<img src='doc_gif/gif1.gif' height="110px"/><img src='doc_gif/gif2.gif' height="110px"/><img src='doc_gif/gif3.gif' height="110px"/><img src='doc_gif/gif4.gif' height="110px"/><img src='doc_gif/gif5.gif' height="110px"/>
</div>

<p align='center'>
The shown images are provided by <a href="https://github.com/JHL-HUST/IBCLN">dataset from IBCLN</a>, <a href="https://sir2data.github.io/">SIR<sup>2</sup> benchmark dataset</a>, <a href="https://github.com/Vandermode/ERRNet">dataset from ERRNet</a> and the Internet images.
</p>
<p align='center'>
Code and pretrained model for our paper <b>Location-aware Single Image Reflection Removal</b><!--   (<a href="1">Arxiv Preprint</a>) -->
</p>

---

## Prerequisites
Our code has been tested under the following platform and environment:
- Ubuntu 18.04, CPU or NVIDIA RTX 2080Ti GPU  + CUDA 10.0 CuDNN
- Python 3.7.5, Pytorch 1.2.0
- Requirements: numpy, tqdm, Pillow，dominate, scikit-image

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
If you find our work helpful to your research, please consider to cite our paper.
```bibtex
```


