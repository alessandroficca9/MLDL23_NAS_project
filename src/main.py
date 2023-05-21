from argparse import ArgumentParser
from models.MobileNetV2 import MobileNetV2
from train import trainer
import torch
from dataset import get_data_loader 

def main():

    #parser = ArgumentParser()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_data = "COCOdataset/all2017"
    path_annotations_train = "visualwakewords/annotations/instances_train.json"
    path_annotations_val = "visualwakewords/annotations/instances_val.json"

    train_dataloader, val_dataloader, test_dataloader = get_data_loader(root_data,
                                                                        path_annotations_train,
                                                                        path_annotations_val,
                                                                        128)
    
    model = MobileNetV2(width_mult=0.35).to(device)

    MV2_val_loss, MV2_val_accuracy, MV2_train_loss, MV2_train_accuracy = trainer(train_dataloader,
                                                                                 val_dataloader,
                                                                                 test_dataloader,
                                                                                 learning_rate=0.01,
                                                                                 weight_decay=0.000001,
                                                                                 momentum=0.9,
                                                                                 epochs=2,
                                                                                 model= model,
                                                                                 device=device )

    return 






if __name__ == '__main__':
    main()