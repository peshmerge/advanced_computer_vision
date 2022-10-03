
# from torchvideotransforms import video_transforms, volume_transforms
from src.dataset import Ego4D
from src.model import cnnLSTM
from torch import optim
from torch import nn
from src.experimental_setup import train
from src.utils import checkCuda
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

torch.cuda.empty_cache() 

if __name__ =="__main__":

#     video_transform_list = [video_transforms.Resize((224,224)),
#                         volume_transforms.ClipToTensor()]
#     transforms = video_transforms.Compose(video_transform_list) 
    
    RESIZE_HEIGHT_WIDTH= (224,224)

    transform = transforms.Compose([
        transforms.Resize(RESIZE_HEIGHT_WIDTH),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        
    ])  
    
#     print (transform)
# Changed to official pytorch library

    
    dataset = Ego4D(transform=transform,df="training_annot.csv",playOnIndexing=False,seq_length=60,filter_dataframe=False,clipPath="v1/clips/")

    # dataset[84]
    
    train_loader = DataLoader(dataset,batch_size=1,shuffle=True)

    validation_dataset = Ego4D(transform=transform,df="testing_annot.csv",playOnIndexing=False,seq_length=60, filter_dataframe=False,clipPath="v1/clips/")  

    validation_loader = DataLoader(validation_dataset,shuffle=False)

    device = checkCuda()

    # #Model hyperparameters

    learning_rate = 0.00001


    model = cnnLSTM(input_size = 4096, n_hidden = 30,
                    n_layers = 1, no_classes=len(dataset.classes),bidirectional=True).to(device)
#     optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    model_settings = {"model":model,
                      "optimizer":optimizer,
                      "loss":loss,
                       "device":device,
                       "epoch":50}

#     # print(dataset[30][1])



#     # print (dataset[10][0].size())
    train(train_loader,model_settings,validation_loader)