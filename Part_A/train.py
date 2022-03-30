import wandb
import argparse
from tqdm import tqdm 

import torch
from torch import  nn
import torch.nn.functional as F

from dataset import create_data
from models import Model, load_optimizer, load_img_shape
from utils import check_accuracy

torch.manual_seed(7)


def train(args):
    '''Trains a small CNN based on command-line arguments passed.'''
    train_data_path = 'nature_12K/inaturalist_12K/train/'
    test_data_path = 'nature_12K/inaturalist_12K/val/'
    
    run_name = 'model-'+args.model+'-epochs-'+str(args.epochs)+'-b_s-'+str(args.batch_size)+'-opt-'+str(args.optimizer) \
                      +'-lr-'+ str(args.learning_rate)+ '-da-'+str(args.data_aug)+'-full_training-'+str(args.full_training)
    print(run_name)

    #function to find compatible input image shape based on model
    image_shape= load_img_shape(args.model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Small CNN Model creation
    activation = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[args.activation]
    model = Model(3, args.num_filters, args.filter_size, activation, args.neurons_dense, image_shape)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = load_optimizer(model.parameters(), args.optimizer , args.learning_rate)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        train_loader, valid_loader = create_data("train",train_data_path,args.data_aug,image_shape[2:], args.batch_size)
      
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            if(args.model == "inceptionv3"):
              scores= scores[1] #getting final output of inceptionv3 model
              scores=F.softmax(scores,dim=1)
            else:
              scores=F.softmax(scores,dim=1)

            #computing loss  
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent step
            optimizer.step()
            break

        train_acc = check_accuracy(device, train_loader, model, args.model)
        val_acc = check_accuracy(device, valid_loader, model, args.model)
        metrics = {'accuracy': train_acc, 'val_accuracy': val_acc}
        # wandb.log(metrics)
        print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', type=str,
                        default='cs6910_assignment2_cnn',
                        help='Project name used to setup wandb.ai dashboard.')
    parser.add_argument('-m', '--model', type=str, default='small_cnn')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train model.')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size to be used to train and evaluate model')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=["adam","rmsprop","nadam","sgd"])
    parser.add_argument('-da', '--data_aug', type=bool, default=False)
    parser.add_argument('-ft', '--full_training', type=bool, default=False)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-nf', '--num_filters', type=int, default=16,
                        help='Number of convolutional filters')
    parser.add_argument('-fs', '--filter_size', type=int, default=3,
                        help='Size of convolutional filters')                        
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=["relu", "tanh", "sigmoid"],
                        help='Activation function of layer')
    parser.add_argument('-sz', '--neurons_dense', type=int, default=32,
                        help='Number of hidden neurons in a layer')
    args = parser.parse_args()
    train(args)