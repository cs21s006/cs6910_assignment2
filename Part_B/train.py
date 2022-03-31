import wandb
import argparse
import torch
from tqdm import tqdm 
torch.manual_seed(7)
from torch import  nn
import torch.nn.functional as F

from model import load_img_shape, load_model, load_optimizer
from data import iNaturalist, create_data
from utils import evaluate

def train(args):

    train_data_path = '../Part_A/nature_12K/inaturalist_12K/train/'
    test_data_path = '../Part_A/nature_12K/inaturalist_12K/val/'
    
    run_name = 'model-'+str(args.model)+'-epochs-'+str(args.epochs)+'-b_s-'+str(args.batch_size)+'-opt-'+str(args.optimizer) \
                      +'-lr-'+ str(args.learning_rate)+ '-da-'+str(args.data_aug)+'-pretrained-'+str(args.pretrained)
    print(run_name)

    #function to find compatible input image shape based on model
    image_shape= load_img_shape(args.model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #CNN Model creation
    model = load_model(args.model, args.pretrained, args.freeze_k).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = load_optimizer(model.parameters(), args.optimizer , args.learning_rate)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        train_loader, valid_loader = create_data("train",
              train_data_path, args.data_aug, image_shape[2:], args.batch_size)
        test_loader =  create_data("test",
              test_data_path, False, image_shape[2:], args.batch_size)
      
        train_correct, train_loss = .0, .0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            outputs = model(data)
            if(args.model == "inceptionv3"):
              outputs, _ = outputs #getting final output of inceptionv3 model
              outputs = F.softmax(outputs, dim=1)
            else:
              outputs = F.softmax(outputs, dim=1)

            #computing accuracy
            _, predictions = outputs.max(1)
            train_correct += (predictions == targets).sum()

            #computing loss  
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent step
            optimizer.step()

        train_loss /= len(train_loader)

        train_acc = round((train_correct / len(train_loader.dataset)).item() * 100, 4)
        val_acc, val_loss = evaluate(device, valid_loader, model, args.model)
        test_acc, test_loss = evaluate(device, test_loader, model, args.model)
        # wandb.log(
        #   {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        # )
        print('\nEpoch ', epoch, 'train_acc', train_acc, 'val_acc', val_acc, 'test_acc', test_acc, 'train_loss', train_loss, 'val_loss', val_loss, 'test_loss', test_loss) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', type=str,
                        default='cs6910_assignment2_cnn',
                        help='Project name used to setup wandb.ai dashboard.')
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        choices=['resnet50', 'inceptionv3', 'inceptionresnetv2', 'xception'])
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train model.')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size to be used to train and evaluate model')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=["adam","rmsprop","nadam","sgd"])
    parser.add_argument('-da', '--data_aug', type=bool, default=False)
    parser.add_argument('-pt', '--pretrained', type=bool, default=True,
                        help='Set to True to load pretrained model weights else False.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-fk', '--freeze_k', type=int, default=45,
                        help='Number of layers to freeze in the network.')                        
    args = parser.parse_args()
    train(args)