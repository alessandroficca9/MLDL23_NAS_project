
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import pyvww
from torchvision import transforms as T

def get_data_loader(root_data, path_annotations_train,path_annotations_val,batch_size, test_batch_size=256):

  # prepare data trasformations
  # Check resize try other values
  # Data augumentation? 
  transform = T.Compose([
      T.Resize((227,227)),
      T.ToTensor()
  ])

  
  full_training_data = pyvww.pytorch.VisualWakeWordsClassification(
    root=root_data,
    transform=transform,
    annFile=path_annotations_train
  )

  test_data = pyvww.pytorch.VisualWakeWordsClassification(
    root=root_data,
    transform=transform,
    annFile=path_annotations_val
  )

  # Indices for the split dataset: Split full_train_dataset into train and val dataset
  num_samples = len(full_training_data)
  training_samples = int(num_samples*0.8+1)
  validation_samples = num_samples - training_samples 

  training_data, validation_data = torch.utils.data.random_split(
      full_training_data,
      [training_samples, validation_samples]
  ) 

  # Initialize dataloader
  train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False)

  return train_loader, val_loader, test_loader
