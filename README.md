# u-net-pytorch
U Net in PyTorch for Image Segmentation


# Setup

Get started by downloading the dataset from kaggle into the data folder, 
such that the data folder contains the folders [train](data/train) & [train_masks](data/train_masks) 

https://www.kaggle.com/c/carvana-image-masking-challenge

Then install the python dependencies with

```shell
pip install -r requirements.txt
```

Split the labeled data into train and validation, 
by running the script [prepare_train_validation_data.py](src/data/prepare_train_validation_data.py)
(You can adjust how many unique cars are moved with the parameter n)