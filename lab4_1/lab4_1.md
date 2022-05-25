##### DLP_LAB4-1_310605001_王盈驊

# LAB4_1
###### tags: `deep learning`
In this lab, we need to implement simple EEG classification models which are EEGNet and DeepConvNet with BCI competition dataset, and also use different kinds of activation function including ReLU, Leaky ReLU, ELU. Try to adjust model hyper-parameters to get the highest accuracy of two architectures with three kinds of activation functions, and plot each epoch accuracy during training and testing.

![](https://i.imgur.com/a5YrLUa.png)
Overall visualization of the EEGNet architecture. Lines denote the convolutional kernel connectivity between inputs and outputs (called feature maps) . The network starts with a temporal convolution (second column) to learn frequency filters, then uses a depthwise convolution (middle column), connected to each feature map individually, to learn frequency-specific spatial filters. The separable convolution (fourth column) is a combination of a depthwise convolution, which learns a temporal summary for each feature map individually, followed by a pointwise convolution, which learns how to optimally mix the feature maps together.

![](https://i.imgur.com/xQVdU2t.png)
The dataset has two channel, each of them has 750 data point and be classified as left and right hand.

# Introduction
## A : The detail of your model

### EEGNet
![](https://i.imgur.com/W0nU7os.png)
``` python =
#!/usr/bin/env python3
import torch.nn as nn

class EEG(nn.Module):
	def __init__(self,act_func='ELU'):
		super(EEG, self).__init__()
		if act_func == 'ELU': self.act_func = nn.ReLU()
		if act_func == 'LeakyReLU': self.act_func = nn.LeakyReLU()
		if act_func == 'ReLU': self.act_func = nn.ReLU()
		self.pipe0 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51), stride=(1,1),padding=(0,25), bias=False),
			nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
		)
		self.pipe1 = nn.Sequential(
			nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
			nn.Dropout(p=0.25),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
		)
		self.pipe2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
			nn.Dropout(p=0.25),
			nn.Flatten(),
			nn.Linear(in_features=736,out_features=2,bias=True)
		)
	def forward(self,x):
		x = self.pipe0(x)
		x = self.act_func(x)
		x = self.pipe1(x)
		x = self.act_func(x)
		x = self.pipe2(x)
		return x
```

### DeepConvNet
![](https://i.imgur.com/DKTFmdS.png)
``` python =
#!/usr/bin/env python3
import torch.nn as nn

class DeepConvNet(nn.Module):
	def __init__(self,act_func='ELU'):
		super(DeepConvNet, self).__init__()
		if act_func == 'ELU': self.act_func = nn.ELU()
		if act_func == 'LeakyReLU': self.act_func = nn.LeakyReLU()
		if act_func == 'ReLU': self.act_func = nn.ReLU()
		C, T, N = 2, 750, 2
		self.pipe0 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5), stride = (1,2)),
			nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(C,1)),
			nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
		)
		self.pipe1 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5), stride = (1,2)),
			nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
		)
		self.pipe2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5)),
			nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
		)
		self.pipe3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5)),
			nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
		)
		self.pipe4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Flatten(),
			nn.Linear(in_features=1600,out_features=2)
		)

	def forward(self,x):
		x = self.pipe0(x)
		x = self.act_func(x)
		x = self.pipe1(x)
		x = self.act_func(x)
		x = self.pipe2(x)
		x = self.act_func(x)
		x = self.pipe3(x)
		x = self.act_func(x)
		x = self.pipe4(x)
		return x

```

## B : Explain the active function(ReLU, Leaky ReLU, ELU)
The activation function can improve model non-linear property, increase model generalization, and get more high accuracy.

### ReLU

[<img src="https://i.imgur.com/Q0iZha9.png" width="500"/>](https://i.imgur.com/Q0iZha9.png)
![](https://i.imgur.com/OA3gJDN.png)

#### Advantage:
- Efficient computation: Only comparison, addition and multiplication.
- The calculation is faster than tanh and Sigmoid, because Relu's mathematical operations are simpler.
- Avoid and correct the problem of vanishing gradients.
#### Disadvantage:
- Non-differentiable at zero; however, it is differentiable anywhere else, and the value of the derivative at zero can be arbitrarily chosen to be 0 or 1.
- Not zero-centered 
- Unbounded
- Dying ReLU problem: ReLU (Rectified Linear Unit) neurons can sometimes be pushed into states in which they become inactive for essentially all inputs. In this state, no gradients flow backward through the neuron, and so the neuron becomes stuck in a perpetually inactive state and "dies". This is a form of the vanishing gradient problem. In some cases, large numbers of neurons in a network can become stuck in dead states, effectively decreasing the model capacity. This problem typically arises when the learning rate is set too high. It may be mitigated by using leaky ReLUs instead, which assign a small positive slope for $x < 0$; however, the performance is reduced.

### Leaky ReLU
[<img src="https://i.imgur.com/Ys9W8Rr.png" width="500"/>](https://i.imgur.com/Ys9W8Rr.png)
![](https://i.imgur.com/OAOg313.png)


#### Advantage:
- It has all of the advantage of ReLU and solve the Dying ReLU problem, which assign a small positive slope for $x < 0$.
#### Disadvantage:
- It will not have poor performance than ReLU since it assign a small positive slope

### ELU
[<img src="https://i.imgur.com/Gf5ywAO.png" width="500"/>](https://i.imgur.com/Gf5ywAO.png)
![](https://i.imgur.com/4tch0SC.png)


#### Advantage:
- It can solve the Dying ReLU problem, which assign a small positive slope for $x < 0$.
#### Disadvantage:
- Higher computation.


# Experiment results
## A : The highest testing accuracy

#### EEG
- learning rate : 0.001
- batch size : 128
- epochs : 600

Activate function | ReLU| ELU| LeakyReLU
---|-----|-------|-----|
accuracy  | 82.59% | 82.13% | 85.93% |

#### DeepConv
- learning rate : 0.001
- batch size : 128
- epochs : 600

Activate function | ReLU| ELU| LeakyReLU
---|-----|-------|-----|
accuracy  |79.35% | 78.15% | 80.00% |


###  ScreenShot with two models
[<img src="https://i.imgur.com/L2MoQzn.png" width="500"/>](https://i.imgur.com/L2MoQzn.png)


### 2. Annything you want to show
**:scroll:Accuracy of Different learning rate**
- model : EEGnet
- epochs : 600
- batch size : 256
- activate function : Leaky ReLU 
- optimizer : Adam


laerning rate | 0.0002 |0.0005|0.0007|0.0008|0.001
---|-----|-----|-------|---|-------|
accuracy  | 83.98% | 85.00% |83.80% | 85.09%|85.37% |

**:scroll:Accuracy of Different optimizer**
- model : EEGnet
- epochs : 600
- batch size : 128
- activate function : Leaky ReLU   
- learning rate : 0.001

laerning rate | SGD |Adagrad | Adam | RMSProp
---|-----|-----|-------|---|
accuracy  | 82.69% | 79.72% |85.93% | 85.28%|

> Adam has the better performance in EEG classification

## B : Comparison figures
### EEGNet


#### Accuracy
[<img src="https://i.imgur.com/HzsY57e.png" width="500"/>](https://i.imgur.com/HzsY57e.png)

#### Loss
[<img src="https://i.imgur.com/o7eh8n5.png" width="500"/>](https://i.imgur.com/o7eh8n5.png)

### DeepConvNet
#### Accuracy
[<img src="https://i.imgur.com/xtFcJxG.png" width="500"/>](https://i.imgur.com/xtFcJxG.png)

#### Loss

[<img src="https://i.imgur.com/PkxYCGk.png" width="500"/>](https://i.imgur.com/PkxYCGk.png)


# Discussion
1. The accuracy of DeepConvNet is lower than EEGNet. And the learning rate of DeepConvNet is also lower than EEGNet. We can find that in the same condition EEGNet is not easily driven by noise and has good performance.

[<img src="https://i.imgur.com/HzsY57e.png" width="300"/>](https://i.imgur.com/HzsY57e.png) [<img src="https://i.imgur.com/xtFcJxG.png" width="300"/>](https://i.imgur.com/xtFcJxG.png)

2. Since the features are not hand-designed by human engineers, understanding the meaning of those features poses a significant challenge in producing interpretable model. In each of these result, EEGNet is more capable of extracting interpretable features that generally corresponded to known neurophysiological phenomena than DeepConvNet.
3. If we enlarge the batchsize, we need to enlarge the epochs to get the better porformance.
4. If the batchsize is small (this picture show the case of 16), the learning rate will be not that smooth.
[<img src="https://i.imgur.com/3N04RQ7.png" width="500"/>](https://i.imgur.com/3N04RQ7.png)




<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
