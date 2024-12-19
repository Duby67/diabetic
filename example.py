from cv2 import destroyAllWindows as destroyWindows
from cv2 import imshow as show
from cv2 import waitKey as wait
from cv2 import imread as read
from cv2 import putText as put
from cv2 import resize as resize
from cv2 import FONT_HERSHEY_SIMPLEX as font
from numpy import floor as floor
from os.path import join as join
from os import walk as walk
from os import getcwd as getcwd
from PIL.Image import open as openImage
from torch import argmax as argmax
from torch import softmax as softmax
from torch import no_grad as nograd
from torch import device as device
from torch import load as load
from torch.cuda import is_available as is_available
import torch.nn.modules.module as module
import torch.nn.modules.conv as conv2d
import torch.nn.modules.linear as linear
import torch.nn.functional as F
from torchvision import transforms as T

class CNN_Retino(module.Module):
    
    def __init__(self, params):
        
        super(CNN_Retino, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # CNN Layers
        self.conv1 = conv2d.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = conv2d.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = conv2d.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = conv2d.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = linear.Linear(self.num_flatten, num_fc1)
        self.fc2 = linear.Linear(num_fc1, num_classes)

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

    hout = floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout = floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout), int(wout)

def get_transform():
    transforms = []
    transforms.append(T.Resize((255,255)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

dir_path = getcwd()
test_path = getcwd() + "/testing"
model = load(dir_path + "/Retino_model.pt")
Mydevice = device("cuda" if is_available() else "cpu")
model = model.to(Mydevice)
ImagesNames = []
CvImages = []
eval_transform = get_transform()
'''
for root, dirs, imgs in walk(test_path):
        for img in imgs:
            ImagesNames.append(img)
            img_path = join(root, img)
            image = openImage(img_path)
            CVimage= read(img_path)
            with nograd():
                x = eval_transform(image)
                x.to(Mydevice)
                predictions = model(x)
                probabilities = softmax(predictions, dim=1)
                predicted_classes = argmax(probabilities, dim=1)
                for predicted_class in predicted_classes:
                    if predicted_class.item() == 0:
                        label = "Diabetic retinopathy"
                    elif predicted_class.item() == 1:
                        label = "No diabetic retinopathy"
                    CVimage = resize(CVimage,(400,400))
                    put(CVimage, label,  (20,20), font, 1, (0,255,0), 2)
                    CvImages.append(CVimage)
       
i = 0
fl = True
while True:
    if fl:
        destroyWindows()
        show(str(ImagesNames[i]) + " Position " + str(i), CvImages[i])
    key = wait(0)
    if key == 27:
        destroyWindows()
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
'''   