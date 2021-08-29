# EDNet
This is the code for:
 - [Pixel-level pavement crack segmentation with encoder-decoder network](https://www.sciencedirect.com/science/article/abs/pii/S0263224121008538)
# Installation
This code is tested on:

Ubuntu 20.04

CUDA 11.0

Anaconda

To install:
```Shell
git clone https://github.com/Youzhi1234/EDNet.git
cd EDNet
conda env create -f environment.yml
```
# Training
This code only provide a few images to test. You can replace it with your own datasets. 

Open jupyter notebook in brower:
```Shell
conda activate EDNet
conda jupyter notebook
```
First, use '1_decoder.ipynb' to train the decoder network.

Then, use '2_encoder.ipynb' to train the encoder network.

Finally, use '3_EDNet.ipynb' to assemble EDNet, and test it.
# Citation
We will appreciate your citation
```
@article{tang2021pixel,
  title={Pixel-level Pavement Crack Segmentation with Encoder-decoder Network},
  author={Tang, Youzhi and Zhang, Allen A and Luo, Lei and Wang, Guolong and Yang, Enhui},
  journal={Measurement},
  pages={109914},
  year={2021},
  publisher={Elsevier}
}
```
