# EDNet
This is the code for:
 - [Pixel-level pavement crack segmentation with encoder-decoder network](https://www.sciencedirect.com/science/article/abs/pii/S0263224121008538)
# Installation
Ubuntu 20.04

CUDA 11.0

Anaconda
```Shell
git clone https://github.com/Youzhi1234/EDNet.git
cd EDNet
conda env create -f environment.yml
conda acitivate EDNet
```
# Training
This code only provide a few images to test. You can replace it with your own datasets. 

```Shell
cd EDNet
conda jupyter notebook
```
First, use '1_decoder.ipynb' to train the decoder network.

Then, use '2_encoder.ipynb' to train the encoder network.

Finally, use '3_EDNet.ipynb' to assemble EDNet.
