import torch
import torchvision
torch.manual_seed(7)
from torch import nn
from . import inceptionresnetv2, xception


def load_img_shape(model_name):
  if model_name in ['inceptionv3', 'inceptionresnetv2', 'xception']:
    return (1, 3, 299, 299)
  else:
    return (1, 3, 224, 224)


def freeze_layers(model, freeze_k):
	if freeze_k == -1:  # freeze all layers
		for param in model.parameters():
			param.requires_grad = False
	else:
		k = 0
		for param in model.parameters():
			k += 1
			param.requires_grad = False
			if k > freeze_k:
				return


def load_model(model_name, pretrained, freeze_k=-1):
	if model_name == "resnet50":
		model = torchvision.models.resnet50(pretrained=pretrained, progress=True)
		freeze_layers(model, freeze_k)
		model.fc= nn.Linear(2048, 10, bias=True)
		return model

	if model_name == "inceptionv3":
		model = torchvision.models.inception_v3(pretrained=pretrained, progress=True)
		freeze_layers(model, freeze_k)
		model.AuxLogits.fc = nn.Linear(768, 10,bias=True)
		model.fc = nn.Linear(2048, 10, bias=True)
		return model

	if model_name == "densenet121":
		model = torchvision.models.densenet121(pretrained=pretrained, progress=True)
		freeze_layers(model, freeze_k)
		model.classifier=nn.Linear(1024,10, bias=True)
		return model

	if model_name == "inceptionresnetv2":
		pretrain = 'imagenet' if pretrained else None
		model = inceptionresnetv2(num_classes=1000, pretrained=pretrain)
		freeze_layers(model, freeze_k)
		model.last_linear = nn.Linear(1536, 10)
		return model
	
	if model_name == "xception":
		pretrain = 'imagenet' if pretrained else None
		model = xception(num_classes=1000, pretrained=pretrain)
		freeze_layers(model, freeze_k)
		model.last_linear = nn.Linear(2048, 10)
		return model

