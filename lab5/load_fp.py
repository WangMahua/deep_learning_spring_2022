import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import glob
from PIL import Image

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, eval_seq, plot_rec,plot_pred,pred,finn_eval_seq,mse_metric

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0010-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000-niter=300-epoch_size=300-cyclical', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=50, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.005, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--c_dim', type=int, default=7, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true') 

    args = parser.parse_args()
    return args

mse_criterion = nn.MSELoss()

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

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.total_epochs = args.niter
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.beta_grad = (self.kl_anneal_cycle*2)/self.total_epochs
        self.mode = 'cyclical' if args.kl_anneal_cyclical else 'monotonic'
        self.beta = 0.0
        self.count = 0

    def update(self):
        self.count += 1
        if self.mode == 'cyclical' :
            self.beta += self.beta_grad
            self.beta = 1.0 if self.beta > 1.0 else self.beta 
            if self.count % int(self.total_epochs/self.kl_anneal_cycle)==0 : 
                self.beta = 0.0 
        else :
            self.beta += self.beta_grad
            self.beta = 1.0 if self.beta > 1.0 else self.beta 
    
    def get_beta(self):
        #self.update()
        return self.beta

def plot_result(KLD, MSE, LOSS, PSNR, BETA, TFR, epoch, args):

    with open('./plot_record.txt', 'w') as result:
                    result.write('kld: {}\n'.format(KLD))
                    result.write('\nmse: {}\n'.format(MSE))
                    result.write('\nloss: {}\n'.format(LOSS))
                    result.write('\npsnr: {}\n'.format(PSNR))
                    result.write('\nbeta: {}\n'.format(BETA))
                    result.write('\ntfr: {}\n'.format(TFR))

    fig = plt.figure()
    ratio = plt.subplot()
    value = ratio.twinx()

    l1, = ratio.plot(BETA, color='red', linestyle='dashed')

    l2, = ratio.plot(TFR, color='orange', linestyle='dashed')

    l3, = ratio.plot(KLD, color='blue')

    l4, = ratio.plot(MSE, color='green')

    l5, = ratio.plot(LOSS, color='cyan')

    x_sparse = np.linspace(0, epoch, np.size(PSNR))
    l6 = value.scatter(x_sparse, PSNR, color='yellow')
    ratio.set_ylim([0.0, 1.05])
    ratio.set_xlabel('Iterations')
    ratio.set_ylabel("ratio/weight")
    value.set_ylabel('Loss')
    plt.title("Training loss / ratio curve")
    plt.legend([l1, l2, l3, l4, l5, l6], ["kl_beta", "tfr", "KLD", "mse", "loss", "PSNR"])
    os.makedirs('%s/plot/' % args.log_dir, exist_ok=True)
    plt.savefig('./{b}/plot/plot_{a}.png'.format(a = epoch,b=args.log_dir))

def make_gif(test,pred,best_psnr_num,args):
    test_seq_ = np.array(test.cpu().numpy())
    pred_seq_ = np.array(pred)
    best_psnr_num = 1
    os.makedirs('%s/plot/' % args.log_dir, exist_ok=True)
    for i in range(args.n_past+args.n_future):
        if i <args.n_past:
            tt = (np.transpose(test_seq_[best_psnr_num,i,:,:], (1, 2, 0)) + 1) / 2.0 * 255.0
            data = Image.fromarray(np.uint8(tt))
            data.save('{log_dir}/plot/{seq}_gt.png'.format(log_dir=args.log_dir,seq = i))
            data.save('{log_dir}/plot/{seq}_pred.png'.format(log_dir=args.log_dir,seq = i))
        else :
            tt = (np.transpose(test_seq_[best_psnr_num,i,:,:], (1, 2, 0)) + 1) / 2.0 * 255.0
            data = Image.fromarray(np.uint8(tt))
            pp = (np.transpose(pred_seq_[best_psnr_num,i-args.n_past,:,:], (1, 2, 0)) + 1) / 2.0 * 255.0
            data2 = Image.fromarray(np.uint8(pp))
            data.save('{log_dir}/plot/{seq}_gt.png'.format(log_dir=args.log_dir,seq = i))
            data2.save('{log_dir}/plot/{seq}_pred.png'.format(log_dir=args.log_dir,seq = i))    


def main():
    KLD_plot, MSE_plot, LOSS_plot, PSNR_plot, BETA_plot, TFR_plot =[],[],[],[],[],[]
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f-niter=%d-epoch_size=%d'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta,args.niter,args.epoch_size)
        text = "-cyclical" if args.kl_anneal_cyclical else "-monotonic"
        name = name + text
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.c_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    test_data = bair_robot_pushing_dataset(args, 'test')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------
    psnr_list = []
    seq, cond =None,None
    epoch_num = 0
    max_psnr = 0
    best_test_seq = []
    best_pred_seq = []
    best_psnr_num = 0

    for _ in range(len(test_data) // args.batch_size):
        
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)
        
        test_seq, test_cond = test_seq.type(torch.FloatTensor),test_cond.type(torch.FloatTensor)
        test_seq, test_cond = test_seq.to(device),test_cond.to(device)

        pred_seq = pred(test_seq, test_cond, modules, args, device)
        # plot 


        _, _, psnr = finn_eval_seq(test_seq[:,args.n_past:], pred_seq[:])

        ave_psnr = np.mean(np.concatenate(psnr))

        with open('./{}/test_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('test trial : %d === test psnr = %.5f \n'%(epoch_num,ave_psnr)))
        epoch_num+=1
        if ave_psnr > max_psnr :
            best_test_seq = test_seq
            best_pred_seq = pred_seq
            


    make_gif(best_test_seq,best_pred_seq,best_psnr_num,args)

                
if __name__ == '__main__':
    main()
        
