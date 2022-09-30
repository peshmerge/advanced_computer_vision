"""
Model implementation


"""





from unicodedata import bidirectional
from torch import nn
from torchvision import models
import os
import torch

# MODEL_CACHE = None

# def extractFeatures(method="vgg16"):
#     global MODEL_CACHE
#     method = method.lower()
#     if method=="vgg16":
#         if not MODEL_CACHE:
#             #Create model if not cached, and store it as global variable so it won't be re-run everytime we create an object
#             model = models.vgg16(pretrained=True)
#             for i in model.parameters():
#                 i.requires_grad = False

#             model.classifier = model.classifier[:2]
#             # model = model.features
#             # model = model.features
#             print (f"[+]{os.path.basename(__file__)} - cached model")
#             MODEL_CACHE = model
#             return MODEL_CACHE
#         else:
#             return MODEL_CACHE
     

"""
The below class is for VGG16 model
"""
class FeatureExtraction(nn.Module):
    def __init__(self):

        super(FeatureExtraction,self).__init__()
        print ("[+]Initializing feature extraction model")
        self.feature_model = models.resnet50(pretrained=True)
        for i in self.feature_model.parameters():
            i.requres_grad=False
        # self.feature_model.classifier = self.feature_model.classifier[:2] vgg16
        self.feature_model = torch.nn.Sequential(*(list(self.feature_model.children())[:-1]))
        print("[+]Loaded feature extraction model")
    
    def forward(self,x):
        with torch.no_grad():
            batch_size,seq_length,c,h,w = x.size()
            x = x.view(batch_size*seq_length,c,h,w)
            x = self.feature_model(x)
            return x.view(batch_size,seq_length,-1)


class cnnLSTM(nn.Module):
    def __init__(self,input_size,n_hidden,n_layers,no_classes,bidirectional=False):
        super(cnnLSTM,self).__init__()


        self.fe = FeatureExtraction()
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.no_classes = no_classes
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.n_hidden,
                           num_layers=self.n_layers,batch_first=True,bidirectional=bidirectional)
        self.bidirectional=bidirectional
        self.linear = nn.Linear(self.n_hidden if not bidirectional else 2*self.n_hidden,self.no_classes)
        self.softmax = nn.Softmax(dim=-1)
        


    def forward(self, x):


        c_out = self.fe(x)
        r_out, (h_n, h_c) = self.rnn(c_out)  

        out = self.linear(r_out)
        out  = self.softmax(out[:,-1])
        # print (out.size())
        #Getting the last timestep
        # return out[:,-1,:]
        return out




# class ConvLSTM(nn.Module):
#     def __init__(self,input_dim):
#         #pass
#         self.input_dim = input_dim
#         self.conv = nn.Conv2D()

#     def forward(self,x):
#         return None
#         #pass




# class convLSTM:
#     pass

# model = extractFeatures()
# print (model)
# lstm_1 = cnnLSTM(input_size=4096,n_hidden = 2,
#                 n_layers = 2,no_classes=4,model=model)


# a = torch.randn((6,40,3,200,200))
# # # m

# fe = FeatureExtraction()


# print(fe(a).size())
# # lstm_1(a)

# extractFeatures(1)