import torch
torch.manual_seed(7)
import torch.nn.functional as F

# Function to Check accuracy to see how good our model
def check_accuracy(device,loader, model,model_name):
    correct_samples = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            if(model_name == "inceptionv3"):
              scores= scores[1] #getting final output of inceptionv3 model
              scores=F.softmax(scores,dim=1)
            else:
              scores=F.softmax(scores,dim=1)

            _, predictions = scores.max(1)
            correct_samples += int(sum(predictions == y))
            total_samples += predictions.size(0)
           
    acc= round((correct_samples / total_samples) * 100, 4)
    return acc      