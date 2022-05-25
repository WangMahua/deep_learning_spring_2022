##### DLP_LAB4-1_310605001_王盈驊

# LAB4_2
###### tags: `deep learning`
# Introduction (20%)
Resnet 全名為 Risdual Neural Network。直覺上來說，較深的網路理論上可以提取更多特徵點而導致效果越好，然而在Resnet出現之前，層數過高的神經網路因為會因為出現梯度消失的問題，所以往往表現反而呈現飽和甚至低於層數較低的網路，這主要是源於太深的神經網路會出現梯度消失或者爆炸問題。
因此Resnet的作者提出了一個新的想法，就是新增一條捷徑去複製前一層的輸出(如下圖所示)，當前一層做back propagation時，如果參數逼近於0，依然存在前一刻的輸出使梯度不會消失，而後以此為基底，此種防止梯度消失的做法也普遍被用在各種神經網路，使後續神經網路層數得以更加深。

![](https://i.imgur.com/l2YU9jq.png)

在這個lab中我們要分別用Resnet18與Resnet18訓練視網膜的照片去判別糖尿病導致視網膜病變之程度。除了要記錄最高的test accuracy 也要畫出 confusion matrix。

![](https://i.imgur.com/I4WFues.png)



# Experiment setups (30%)
## The details of your model (ResNet)
由於pytorch本身就有Resnet18 與 Restnet50的模型，我們只需要去修改最後一層的output feature為5即可

``` python =
#!/usr/bin/env python3
import torch.nn as nn
from torchvision import transforms,models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet18(nn.Module):
    def __init__(self, num_class, feature_extract, use_pretrained):
        super(ResNet18,self).__init__()
        self.model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model, feature_extract)
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)

    def forward(self,x):
        out=self.model(x)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_class, feature_extract, use_pretrained):
        super(ResNet50,self).__init__()
        self.model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model, feature_extract)
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
    def forward(self,x):
        out=self.model(x)
        return out
``` 
## The details of your Dataloader
1. 透過getitem()獲得指定index中的相片資料
2. 透過tran()正規化資料 


``` python =
import pandas as pd
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.trans = self.__trans(mode)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        rgb_image = Image.open(os.path.join(self.root, self.img_name[index] + ".jpeg")).convert('RGB')
        img = self.trans(rgb_image)
        label = self.label[index]

        return img, label

    def __trans(self, mode):
        
        if mode == "train":
            transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        else:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform
``` 
## Describing your evaluation through the confusion matrix
- confusion matrix : 用來判斷學習的好壞，兩個座標軸分別為實際上的分類別與預測出的類別，如果對角線機率越高則表示估測越為準確，同時也可以從圖中看出資料的大概分佈，亦可拿來判斷訓練集的好壞。
``` python =
def plot_confusion_matrix(self, num_class):
    matrix = np.zeros((self.num_class,self.num_class))
    self.model.eval()
    for i, (data, label) in enumerate(self.testloader):
        data, label = Variable(data),Variable(label)
        data, label = data.cuda(), label.cuda()

        with torch.no_grad():
            prediction = self.model(data)
            pred = prediction.data.max(1, keepdim=True)[1]

            ground_truth = pred.cpu().numpy().flatten()
            actual = label.cpu().numpy().astype('int')

            for j in range(len(ground_truth)):
                matrix[actual[j]][ground_truth[j]] += 1

    for i in range(self.num_class):
        matrix[i,:] /=  sum(matrix[i,:])

    plt.figure(1)
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Normalized confusion matrix")
    plt.colorbar()

    thresh = np.max(matrix) / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(self.num_class)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(self.picture_path, self.file_name + '_confusion_matrix.png'))
```



# Experimental results (30%)
## A. The highest testing accuracy
#### RseNet18
- learning rate : 0.001
- batch size : 16
- epochs : 10

pretrain | yes| no| 
---|-----|-------|
accuracy  |81.99% | 73.50% | 

#### RseNet50
- learning rate : 0.001
- batch size : 8
- epochs : 5

pretrain | yes| no| 
---|-----|-------|
accuracy  |81.96% | 73.32% | 

### Screenshot
![](https://i.imgur.com/9ORNCDf.png)

### learning rate
![](https://i.imgur.com/eBkQ2qr.png)
learning rate | 0.0015| 0.0018| 
---|-----|-------|
accuracy  |81.99% | 81.99% | 

改變learning rate後發現兩者準確度都不低，但是0.0018平均起來表現較原參數好。

## B. Comparison figures
### confusion matrix 
##### Resnet18(without pretrain)
![](https://i.imgur.com/LGCBvdp.png)
##### Resnet18(with pretrain)
![](https://i.imgur.com/r7EsZHO.png)

##### Resnet50(without pretrain)
![](https://i.imgur.com/7Ifbn50.png)

##### Resnet50(with pretrain)
![](https://i.imgur.com/TXMdRGI.png)

### Plotting the comparison figures(RseNet18/50, with/without pretraining)
##### Resnet18
![](https://i.imgur.com/Z4MKumk.png)

##### Resnet50
![](https://i.imgur.com/dwiFpHI.png)


# Discussion (20%)
- 從實驗結果可以明顯看出只要是有pretrain過得基本上都優於沒經過pretrain的model，可能原因有：
	- 可以從confusion matrix看出再沒有pretrain的情況下，model會把所有的test data都判定為class 0 ，而且準確率也高達73％左右，說明train data與test data其實是嚴重不平衡，導致訓練結果無法發揮
	- 官方給予的pretrain model讓我們得以用已訓練過得Resnet來訓練本次訓練集，省去提取基本特徵點的時間，也可以更加提昇訓練效果
