import torch
import tqdm
import torch.nn as nn

from dataset import get_data_loader

def get_optimizer(net, lr,wd,momentum):
  #optimizer = torch.optim.SGD(net.parameters() ,lr=lr, weight_decay=wd,momentum=momentum)
  optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
  return optimizer 

def get_loss_function():
  loss_function = nn.CrossEntropyLoss()
  return loss_function

def train(net, data_loader, optimizer, loss_function, device):
  
  samples=0
  cumulative_loss=0
  cumulative_accuracy=0
  net.train()

  # try AMP
  scaler = torch.cuda.amp.GradScaler()


  with tqdm.tqdm(total=len(data_loader)) as pbar:

    for batch_idx, (inputs, targets) in enumerate(data_loader):

      
      # Load data into GPU
      inputs, targets = inputs.type(dtype=torch.float16).to(device), targets.type(dtype=torch.LongTensor).to(device)
      
      optimizer.zero_grad(set_to_none=True) # reset the optimizer

      with torch.cuda.amp.autocast():
        outputs = net(inputs) # Forward pass
        loss = loss_function(outputs, targets) # Apply the loss
        
      scaler.scale(loss).backward()
      #loss.backward() # Backward pass
      scaler.step(optimizer)
      #optimizer.step()  # update parameters
      scaler.update()

      samples += inputs.shape[0]
      cumulative_loss += loss.item()
      _, predicted = outputs.max(1)
      cumulative_accuracy += predicted.eq(targets).sum().item()
      pbar.set_postfix_str("training with Current loss: {:.4f}, Accuracy: {:.4f}, at iteration: {:.1f}".format(cumulative_loss/ samples, cumulative_accuracy / samples*100, float(batch_idx)))
      pbar.update()
  return cumulative_loss/samples, cumulative_accuracy/samples*100

  # we define a test function
def test(net, data_loader, loss_function, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.
  net.eval() # Strictly needed if network contains layers which have different behaviours between train and test
  with tqdm.tqdm(total=len(data_loader)) as pbar:  
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs, targets = inputs.type(dtype=torch.float16).to(device), targets.type(dtype=torch.LongTensor).to(device)
        
        with torch.cuda.amp.autocast():
          outputs = net(inputs)
          loss = loss_function(outputs, targets)

        # Forward pass
        #outputs = net(inputs)
        _, predicted = outputs.max(1)
        #loss = loss_function(outputs,targets) 
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        cumulative_accuracy += predicted.eq(targets).sum().item()
        pbar.set_postfix_str("validation with Current loss: {:.4f}, Accuracy: {:.4f}, at iteration: {:.1f}".format(cumulative_loss/ samples, cumulative_accuracy / samples*100, float(batch_idx)))
        pbar.update()
  return cumulative_loss/samples, cumulative_accuracy/samples*100



def trainer(
    # lets define the basic hyperparameters
    train_loader,
    val_loader,
    test_loader,
    learning_rate=0.01,
    weight_decay=0.000001,
    momentum=0.9,
    epochs=2,
    model=None,
    device="cuda:0"):
  
  

  # defining the optimizer
  optimizer = get_optimizer(model, learning_rate, weight_decay, momentum)
  # defining the loss function
  loss_function = get_loss_function()
  # finaly training the model 


  # In order to save the accuracy and loss we use a list to save them in each epoch 
  val_loss_list = []
  val_accuracy_list = []
  train_loss_list = []
  train_accuracy_list = []

  for e in range(epochs):
    print('training epoch number {:.2f} of total epochs of {:.2f}'.format(e,epochs))
    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_function,device)
    val_loss, val_accuracy = test(model, val_loader, loss_function,device)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)


    print('Epoch: {:d}'.format(e+1))
    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
    train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
    val_accuracy))
  print('-----------------------------------------------------')
  print('After training:')
  train_loss, train_accuracy = test(model, train_loader, loss_function,device)
  val_loss, val_accuracy = test(model, val_loader, loss_function,device)
  test_loss, test_accuracy = test(model, test_loader, loss_function,device)
  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
  train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
  val_accuracy))
  print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')
  return val_loss_list, val_accuracy_list, train_loss_list, train_accuracy_list
