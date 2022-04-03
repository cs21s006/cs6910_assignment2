# CS6910 Assignment 2 Part A

[Link to Weights & Biases Report](https://wandb.ai/cs21s006_cs21s043/Assigment2_P1_Q2_/reports/CS6910-Assignment-2-Report-from-Scratch--VmlldzoxNzYzMjI1)

## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies.

```
pip install requirements.txt
python train.py
```

The number of filters, size of filters and activation function in each layer can be changed and  the number of neurons in the dense layer can be changed by passing command line arguments to the training script

```
python train.py --num_filters 32 --filter_size 256 --activation relu --neurons_dense 128
``` 

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-p`, `--project_name` | neural-networks-fashion-mnist | Project name used to track experiments in Weights & Biases dashboard |
| `-e`, `--epochs` | 30 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-o`, `--optimizer` | adam | choices:  ["sgd", "rmsprop", "adam", "nadam"] | 
| `-ft`, `--full_training` | False | choices:  [True, False] | 
| `-da`, `--data_aug` | False | choices:  [True, False] | 
| `-lr`, `--learning_rate` | 0.0003 | Learning rate used to optimize model parameters | 
| `-nf`, `--num_filters` | 16 | Number of filters used in convolutional layers. | 
| `-fs`, `--filter_size` | 3 | Size of filters in convolutional layers. |
| `-a`, `--activation` | relu | choices:  ["sigmoid", "tanh", "relu"] |
| `-sz`, `--neurons_dense` | 32 | NUmber of hidden neurons in dense layer. |

## Examples, Usage and More

### Defining a Model

```python
from models import Model, load_optimizer, load_img_shape

model = Model(in_channels=3, num_filters=32, filter_size=3,
              activation=nn.ReLU(), neurons_dense=128,
              load_img_shape("small_cnn"))

# Forward pass
y_pred = model.forward(X)  # X is i+ nput
```


## Quick Links

* [Question 1](<Question 1.ipynb>)
* [Question 2](<Question 2.ipynb>) 
* [Question 4](<Question 4.ipynb>)
* [Question 5](<Question 5.ipynb>)

## Team 
* [CS21S043](https://github.com/jainsaurabh426)
* [CS21S006](https://github.com/cs21s006)