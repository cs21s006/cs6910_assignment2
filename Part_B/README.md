# CS6910 Assignment 2 Part B

[Link to Weights & Biases Report](https://wandb.ai/cs21s006_cs21s043/Assigment2_P1_Q2_/reports/CS6910-Assignment-2-Report-from-Scratch--VmlldzoxNzYzMjI1#part-b-:-fine-tuning-a-pre-trained-model)

## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies.

```
pip install requirements.txt
python train.py
```

The model, number of layers to freeze, pretrained can be changed by passing command line arguments to the training script

```
python train.py --model xception --freeze_k 70 --pretrained True
``` 

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-p`, `--project_name` | neural-networks-fashion-mnist | Project name used to track experiments in Weights & Biases dashboard |
| `-m`, `--model` | resnet50 | ["resnet50", "inceptionv3", "inceptionresnetv2", "xception"] |
| `-e`, `--epochs` | 30 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-o`, `--optimizer` | adam | choices:  ["sgd", "rmsprop", "adam", "nadam"] | 
| `-da`, `--data_aug` | False | choices:  [True, False] | 
| `-pt`, `--pretrained` | False | choices:  [True, False] | 
| `-lr`, `--learning_rate` | 0.0003 | Learning rate used to optimize model parameters | 
| `-fk`, `--freeze_k` | 45 | Number of layers to freeze in the network |


## Examples, Usage and More

### Defining a Model

```python
from model import load_img_shape, load_model, load_optimizer

model = load_model(model="inceptionv3", pretrained=True, freeze_k=47).to(device)

# Forward pass
y_pred = model.forward(X)  # X is input
```


## Quick Links

* [Question 3](notebooks/)

## Team 
* [CS21S043](https://github.com/jainsaurabh426)
* [CS21S006](https://github.com/cs21s006)