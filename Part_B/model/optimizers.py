from torch import optim

def load_optimizer(model_param, opt_name, learning_rate):
  if opt_name == "adam":
    optimizer = optim.Adam(model_param, lr=learning_rate)
    return optimizer
  elif opt_name == "rmsprop":
    optimizer = optim.RMSprop(model_param, lr=learning_rate)
    return optimizer
  elif opt_name == "nadam":
    optimizer = optim.NAdam(model_param, lr=learning_rate)
    return optimizer
  elif opt_name == "sgd":
    optimizer = optim.SGD(model_param, lr=learning_rate)
    return optimizer