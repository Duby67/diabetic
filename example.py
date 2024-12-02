import pandas as pd 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
import copy 
import os 
import torch
from PIL import Image 
from torch.utils.data import Dataset 
import torchvision
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import torch.nn as nn 
from torchvision import utils 
from torchvision.datasets import ImageFolder
from torchsummary import summary
import torch.nn.functional as F
from sklearn.metrics import classification_report
import itertools 
from tqdm.notebook import trange, tqdm 
from torch import optim
import warnings
from torchvision.io import read_image
warnings.filterwarnings('ignore')
from torchvision import transforms as T

class CNN_Retino(nn.Module):
    
    def __init__(self, params):
        
        super(CNN_Retino, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # CNN Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

def findConv2dOutShape(hin,win,conv,pool=2):
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


def get_transform():
    transforms = []
    transforms.append(T.Resize((255,255)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

dir_path = os.getcwd()
test_path = os.getcwd() + "/testing"
batch_size = 64
model = torch.load(dir_path + "/Retino_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 
ImagesNames = []
CvImages = []
eval_transform = get_transform()
model.eval()
for root, dirs, imgs in os.walk(test_path):
        for img in imgs:
            ImagesNames.append(img)
            img_path = os.path.join(root, img)
            image = Image.open(img_path)
            CVimage= cv.imread(img_path)
            with torch.no_grad():
                x = eval_transform(image)
                x.to(device)
                predictions = model(x)
                probabilities = torch.softmax(predictions, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                for predicted_class in predicted_classes:
                    if predicted_class.item() == 0:
                        label = "Diabetic retinopathy"
                    elif predicted_class.item() == 1:
                        label = "No diabetic retinopathy"
                    CVimage = cv.resize(CVimage,(400,400))
                    cv.putText(CVimage, label,  (20,20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    CvImages.append(CVimage)
i = 0
fl = True
while True:
    if fl:
        cv.destroyAllWindows()
        cv.imshow(str(ImagesNames[i]) + " Position " + str(i), CvImages[i])
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()
        break
    if key == 81:
        if i > 0:
            i = i - 1
            fl = True
        else:
            fl = False
    elif key == 83:
        if i < len(CvImages):
            i = i + 1
            fl = True
        else:
            fl = False
    else:
        fl = False