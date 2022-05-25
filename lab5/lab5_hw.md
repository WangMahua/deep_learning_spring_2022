##### DLP_LAB5_310605001_王盈驊

# LAB5
###### tags: `deep learning`

## Introduction 

在本次的實驗中，我們要透過CVAE (Conditional Variational AutoEncoder) 去預測影片動作。 CVAE 是多層神經網絡的一種非監督式學習算法，它可以幫助資料分類、視覺化、儲存。其架構中可細分為 Encoder（編碼器）和 Decoder（解碼器）兩部分，我們會先將真實照片透過encoder進行編碼，並在編碼過程增加了一些限制，迫使生成的向量遵從高斯分佈，同時也符合condition；然後在將這些被編碼過的數據丟到一個decoder中解碼，並產生出樣本，反覆訓練模型讓decoder可以產生出和真實情況一樣接近的照片


![](https://i.imgur.com/u5HMYHm.png)

## Derivation of CVAE

假設平常訓練模型輸入為X，以及條件c，輸出一個類別Y，則我們會建立一個模型$f(X;\theta)$，使得輸入X輸出Y的機率最大，即是$p(Y|X)$最大。而若想使用生成模型，則必須轉為貝氏定理。而p(z)的z機率能由我們決定，而較難解決的則是$p(X|z)$，若隱藏變量是一個連續機率分布的空間，公式如下圖，分母意思是一個連續機率為隱藏變量z輸出X的機率乘上z的機率，而這在數學上要求導是有難度的。

$$
p(z \mid X,c ; \theta)=\frac{p(X \mid  c ; \theta) p(z)}{\int_{} p(X \mid z, c ; \theta) p(z) d z}
$$
所以我們將其改寫

$$
p\left(X,\left.Z\right|{c} ; \theta\right)=p\left(\left.X\right|{c} ; \theta\right) p(Z \mid X, c ; \theta)
$$

同時對兩邊取log，並整理為下式

$\log [p(X, Z \mid c ; \theta)]=\log[ p\left(\left.X\right|{c} ; \theta\right)]+\log [p(Z \mid X, c ; \theta)]$

$\log[ p\left(\left.X\right|{c} ; \theta\right)] = \log [p(X, Z \mid c ; \theta)]-\log [p(Z \mid X, c ; \theta)]$

將等號兩邊同乘 q(Z)並對 Z 做積分

$\begin{gathered}
\int q(Z) \log p(X \mid c ; \theta) d Z=\int q(Z) \log p(X, Z \mid c ; \theta) d Z-\int q(Z) \log p(Z \mid X, c ; \theta) d Z \\
=\int q(Z) \log p(X, Z \mid c ; \theta) d Z-\int q(Z) \log q(z Z) d Z \\
+\int q(Z) \log q(Z) d Z-\int q(Z) \log p(Z \mid X, c ; \theta) d Z \\
\because \quad q(Z) \log q(Z) d Z-\int q(Z) \log p(Z \mid X, c ; \theta) d Z=K L(q(Z)|| p(Z \mid X, c ; \theta)) \\
\therefore \log p(X \mid c ; \theta)=\mathcal{L}(X, q, \theta)+K L(q(Z) \| p(Z \mid X, c ; \theta))
\end{gathered}$

若要最大化 $\log p(X \mid c ; \theta)$，則我們需要最大化$\mathcal{L}(X, q, \theta)$


$\begin{aligned}
\mathcal{L}(X, q, \theta) &=\int q(Z) \log p\left(X,\left.Z\right|_{c} ; \theta\right) d Z   
-\int q(Z) \log q(Z) d Z
\end{aligned}$

其中 

$\int q(Z) \log p\left(X,\left.Z\right|_{c} ; \theta\right) d Z=E_{Z \sim q(Z)}[\log p(X, Z \mid c ; \theta)]$ 

$\int q(Z) \log q(Z) d Z=E_{Z \sim q(Z)}[\log q(Z)]$

令 $q(Z)=q\left(Z \mid X, c ; \theta^{\prime}\right)$

$\mathcal{L}(X, q, \theta)=E_{Z \sim q\left(Z \mid X, c ; \theta^{\prime}\right)}[\log p(X, Z \mid c ; \theta)]-E_{Z \sim q\left(Z \mid X, c ; \theta^{\prime}\right)}\left[\log q\left(Z \mid X, c ; \theta^{\prime}\right)\right]$

$p(X, Z \mid c ; \theta)=p(X \mid Z, c ; \theta) p(Z \mid c)$

$\mathcal{L}(X, q, \theta)=E_{Z \sim q\left(z \mid X, c ; \theta^{\prime}\right)}[p(X \mid Z, c ; \theta)]+E_{Z \sim q\left(z \mid X, c ; \theta^{\prime}\right)}\left[\log p(Z \mid c)-\log q\left(Z \mid X, c ; \theta^{\prime}\right)\right]$

$\because E_{Z \sim q(Z \mid X, c ; \theta \prime)}\left[\log p(Z \mid c)-\log q\left(Z \mid X, c ; \theta^{\prime}\right)\right]=-K L\left(q\left(Z \mid X, c ; \theta^{\prime}\right)|| p(Z \mid c)\right)$

$\therefore \quad \mathcal{L}(X, q, \theta)=E_{Z \sim q(Z \mid X, c ; \theta \prime)}[p(X \mid Z, c ; \theta)]-K L\left(q\left(Z \mid X, c ; \theta^{\prime}\right)|| p(Z \mid c)\right)$


## Implementation details

### Describe how you implement your model
- encoder

  此部分已被預先完成，主要是基於原torch的model，並重新自訂義層數
```python=
class vgg_encoder(nn.Module):
    def __init__(self, dim):
        super(vgg_encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(3, 64),
                vgg_layer(64, 64),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]
```
- decoder
```python=
class vgg_decoder(nn.Module):
    def __init__(self, dim):
        super(vgg_decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128),
                vgg_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64),
                nn.ConvTranspose2d(64, 3, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16 
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16 
        up3 = self.up(d3) # 8 -> 32 
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output
```
- reparameterization
```python=
def reparameterize(self, mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = Variable(logvar.data.new(logvar.size()).normal_())
    return eps.mul(logvar).add_(mu)
```
- train

在進入迴圈之前會先判斷是否使用teacher force(第12行)

將所有資料丟入encoder(第14行)

若進到prediction階段，則開始將資料丟入encoder中(第24~29行)

計算posterior，用以生成下一階段的predictor的依據(第33行)

丟入decoder，產生新數據，與真實資料作比對計算mse(第40~42行)

```python=
def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False

    h_seq = [ modules['encoder'](x[:,i]) for i in range(args.n_past + args.n_future)] # x : [10,12,3,64,64] h_seq : [12,10,128]
    for i in range(1, args.n_past + args.n_future):
        h_target = h_seq[i][0]

        if args.last_frame_skip or i < args.n_past:	
            h = h_seq[i-1][0] 
            skip = h_seq[i-1][1]
        else:
            h = h_seq[i-1][0]

        if i > 1:
            previous_img = x_pred
            pr_latent = modules['encoder'](previous_img)
            h_no_teacher = pr_latent[0]
        else:
            h_no_teacher = h    

        c = cond[:, i, :].float()

        z_t, mu, logvar = modules['posterior'](h_target)

        if use_teacher_forcing:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, c], 1))
        else:
            h_pred = modules['frame_predictor'](torch.cat([h_no_teacher, z_t, c], 1))
            
        x_pred = modules['decoder']([h_pred, skip])
        mse += mse_criterion(x_pred, x[:,i])
        kld += kl_criterion(mu, logvar,args)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), \
        mse.detach().cpu().numpy() / (args.n_past + args.n_future), \
        kld.detach().cpu().numpy() / (args.n_future + args.n_past)

```
- predict 

  基本上和train架構略同，但必須注意拿來解碼的x_in從第二個time step後須拿上一個time step pred 與decode出來的結果
```python=
def pred(x, cond, modules, args, device):
    x = x.float()
    # print(x.shape)

    gen_seq = []
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()


    h_seq = [ modules['encoder'](x[:,i]) for i in range(args.n_past + args.n_future)] # x : [10,12,3,64,64] h_seq : [12,10,128]
    x_in = x[:, 0]
    for i in range(1, args.n_eval):
        c = cond[:, i, :].float()
        h = modules['encoder'](x_in)

        if args.last_frame_skip or i < args.n_past:   
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        if i < args.n_past:
            h_target = modules['encoder'](x[:,i])[0].detach()
            _, z_t, _ = modules['posterior'](h_target)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_() 
        
        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t, c], 1))
            x_in = x[:, i]
        else:
            h = modules['frame_predictor'](torch.cat([h, z_t, c], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in.data.cpu().numpy())
  
    gen_seq = torch.tensor(np.array(gen_seq))
    #print('gen_seq', gen_seq.dtype, gen_seq.shape)
    gen_seq = gen_seq.permute(1,0,2,3,4)

    return gen_seq
```

- dataloader

    在拿取資料時先透過set_seed，隨機設定亂數種子，以改變每次選取的相片集，分別透過get_seq獲得來當訓練集的資料、get_csv獲得訓練集的action還有position
```python = 
import torch
import os
import numpy as np
import csv
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root_dir = args.data_root 
        if mode == 'train':
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = False
        elif mode == 'train':
            self.data_dir = '%s/test' % self.root_dir
            self.ordered = True 
        else :
            self.data_dir = '%s/validate' % self.root_dir
            self.ordered = True 
        
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = args.n_eval
        # self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0
        self.d_now = None
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        self.d_now = d
            
    def __len__(self):
        return len(self.dirs)
        
    def get_seq(self):

        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (self.d_now, i)
            im = Image.open(fname)  # read an PIL image
            img = np.array(im).reshape(1, 64, 64, 3)/255.
            image_seq.append(img)

        image_seq = np.concatenate(image_seq, axis=0)
        image_seq = torch.from_numpy(image_seq)
        image_seq = image_seq.permute(0,3,1,2)
        return image_seq 
    
    def get_csv(self):
        action = None 
        position = None 
        data = None

        with open('%s/actions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        action = np.array(data)

        with open('%s/endeffector_positions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        position = np.array(data)

        c = np.concatenate((action,position),axis=1)
        c = c[:self.seq_len]
        c = c.astype(np.float)

        return c
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq  = self.get_seq()
        cond =  self.get_csv()
        return seq,cond

```

### Describe the teacher forcing
- main idea
在訓練的時候，選擇是否要使用ground truth 來做訓練而不是使用自己predict的資料。
- benefits
使用teacher forcing 的好處是可以加速收斂過程
- drawbacks
過分仰賴teacher forcing可能會導致Overcorrect

## Results and discussion

### Show your results of video prediction

#### (a) Make videos or gif images for test result
left : ground truth / right : prediction

![](https://i.imgur.com/STQQxEM.gif)

#### (b) Output the prediction at each time step
上方為griund truth 下方為prediction

![](https://i.imgur.com/I999hjI.png)![](https://i.imgur.com/m0tWUUm.png)![](https://i.imgur.com/f21DWeg.png)![](https://i.imgur.com/56RWOKM.png)![](https://i.imgur.com/UYUVrAI.png)![](https://i.imgur.com/uDrsSfh.png)![](https://i.imgur.com/29uoFh9.png)![](https://i.imgur.com/OYwmZno.png)![](https://i.imgur.com/JpmQPvq.png)![](https://i.imgur.com/JZuW0t6.png)

![](https://i.imgur.com/hoAbuLx.png)![](https://i.imgur.com/nrVbUAC.png)![](https://i.imgur.com/7dpoTf7.png)![](https://i.imgur.com/KWCHg7v.png)![](https://i.imgur.com/tDYASUo.png)![](https://i.imgur.com/ZgLh9VG.png)![](https://i.imgur.com/OAsGW8V.png)![](https://i.imgur.com/1fkMfBN.png)![](https://i.imgur.com/tSO2A1P.png)![](https://i.imgur.com/utMI2uU.png)

### Plot the KL loss and PSNR curves during training
#### (1) fp

1. learning rate : 0.001
2. batch size : 16
3. niter : 280
4. epoch size : 300
5. kl_anneal_cyclical : true
6. kl_anneal_cycle : 3
7. tfr : 1.0
8. tfr_start_decay_epoch : 50
9. tfr_decay_step : 0.005
10. tfr_lower_bound :0.0

![](https://i.imgur.com/1qKwbdm.png)

1. learning rate : 0.001
2. batch size : 16
3. niter : 280
4. epoch size : 300
5. kl_anneal_cyclical : false
6. kl_anneal_cycle : 3
7. tfr : 1.0
8. tfr_start_decay_epoch : 50
9. tfr_decay_step : 0.005
10. tfr_lower_bound :0.0

![](https://i.imgur.com/4PPZOof.png)

(2)lp
1. learning rate : 0.001
2. batch size : 16
3. niter : 280
4. epoch size : 300
5. kl_anneal_cyclical : true
6. kl_anneal_cycle : 3
7. tfr : 1.0
8. tfr_start_decay_epoch : 50
9. tfr_decay_step : 0.005
10. tfr_lower_bound :0.0

![](https://i.imgur.com/tFBlRBZ.png)


1. learning rate : 0.001
2. batch size : 16
3. niter : 280
4. epoch size : 300
5. kl_anneal_cyclical : false
6. kl_anneal_cycle : 3
7. tfr : 1.0
8. tfr_start_decay_epoch : 50
9. tfr_decay_step : 0.005
10. tfr_lower_bound :0.0

![](https://i.imgur.com/QFgqpvC.png)


### Discuss the results according to your setting of teacher forcing ratio, KL weight, and learning rate. Note that this part mainly focuses on your discussion, if you simply just paste your results, you will get a low score

- teacher force 

teacher force  | yes | no| 
----|-----|-------|
highest average psnr |26.95 | 19.97 | 

加入teacher force 會對模型psnr提升效果相當顯著，因為完全沒有ground truth一起訓練的話，純粹只靠predictor，在訓練初期常常做出錯誤的判斷，連帶影響下一個time step預測效果變差，會讓模型學習效果沒那麼好

- KL weight

KL weight  | cyclical | monotonic | 
----|-----|-------|
highest average psnr |28.17 | 27.95 | 

cyclical在表現上會比monotonic稍微高一點，在cyclical 模式中，每一次cycle重新開始之前，KLD都會發生顯著且劇烈的變化、psnr也會先下降之後快速回升；在monotonic 模式中psnr則是呈現穩定提升，而個人認為調整讓KL weight一直反覆循環可以讓model強健性要變得稍高一點。

- learning rate 

learning rate  | 0.001 | 0.002 | 
----|-----|-------|
highest average psnr |19.07 |20.23 | 

在本次作業中learning rate調整從我的結果看來差不多，未感受到特別顯著的差異。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>



