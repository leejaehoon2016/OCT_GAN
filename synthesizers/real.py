import warnings, random
warnings.filterwarnings(action='ignore')
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2" 

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, TripletMarginLoss
from torch.nn import functional as F
from torchdiffeq._impl.adjoint import odeint_adjoint
from torchdiffeq import odeint 
from synthesizers.base import BaseSynthesizer
# from synthesizers.utils import GeneralTransformer
from synthesizers.utils import BGMTransformer
from evaluate import compute_scores
from data import load_dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
writer = SummaryWriter()

import pandas as pd
import time
time = time.strftime("%b%d_%H-%M-%S", time.localtime(time.time()))
import copy
from benchmark import benchmark

short = ["adult", "alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
test_set = ['grid', 'alarm', 'adult']
long = ["covtype", "intrusion",  "mnist28"]

DEFAULT_DATASETS = ["adult", "alarm", "asia", "census", "child", "covtype", "credit", "grid", "gridr", "insurance",
                    "intrusion", "mnist12", "mnist28", "news", "ring"]

# HyperParameter
lambda_grad = 10
stability_regularizer_factor = 1e-5
rtol=1e-3
atol=1e-3

save_loc = '/home/bigdyl/jindunh/CTGAN_local/result'

parser = argparse.ArgumentParser("ctgan with odes")
parser.add_argument('--seed', type=int, default=777)
parser.add_argument('--D_lr', type=float, default=2e-4)
parser.add_argument('--G_lr', type=float, default=2e-4)
parser.add_argument('--batch', type=int, default=500)
parser.add_argument('--num_split', type=int, default=3)
parser.add_argument('--data', type=str)

args = parser.parse_args()

print("experiment C")
print("settings;")
print("seed: ", args.seed)
print("D_lr: ", args.D_lr)
print("G_lr: ", args.G_lr)
print("batch: ", args.batch)
print("num_split: ", args.num_split)
print("start time: ", time)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class ODEFuncG(nn.Module):

    def __init__(self, first_layer_dim):
        super(ODEFuncG, self).__init__()

        self.dim = first_layer_dim

        self.layer_start = PixelNorm()
        seq = [ nn.Linear(first_layer_dim + 1, first_layer_dim + 1),
                nn.LeakyReLU(0.2) ]
        seq *= 7
        seq.append(nn.Linear(first_layer_dim + 1, first_layer_dim)) 
        self.layer_t = Sequential(*seq)

        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):

        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class ODEFuncD(nn.Module):

    def __init__(self, first_layer_dim):
        super(ODEFuncD, self).__init__()
        self.layer_start = nn.Sequential(nn.BatchNorm1d(first_layer_dim),
                                    nn.ReLU())

        self.layer_t = nn.Sequential(nn.Linear(first_layer_dim + 1, first_layer_dim * 2),
                                     nn.BatchNorm1d(first_layer_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(first_layer_dim * 2, first_layer_dim * 1),
                                     nn.BatchNorm1d(first_layer_dim * 1),
                                     nn.ReLU())
        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class ODEBlockG(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlockG, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=rtol, atol=atol) 

        return out[1] 

def ODETime(num_split):
        return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device=device) for
                i in range(1, num_split)]

class ODEBlockD(nn.Module):
    def __init__(self, odefunc, num_split):
        super(ODEBlockD, self).__init__()
        self.odefunc = odefunc
        self.num_split = num_split

    def forward(self, x):

        initial_value = x[0]
        integration_time = torch.cat(x[1], dim = 0).to(device)
        zero = torch.tensor([0.], requires_grad=False).to(device)
        one = torch.tensor([1.], requires_grad=False).to(device)

        all_time = torch.cat( [zero, integration_time, one],dim=0).to(device)
        self.total_integration_time1 = [all_time[i:i+2] for i in range(self.num_split)]

        out = [[1, initial_value]]
        for i in range(len(self.total_integration_time1)):
            self.integration_time = self.total_integration_time1[i].type_as(initial_value)
            out_ode = odeint(self.odefunc, out[i][1], self.integration_time, rtol=rtol, atol=atol)
            out.append(out_ode)
        return [i[1] for i in out]

class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        self.ode = ODEBlockG(ODEFuncG(dim))

        seq = []
        for item in list(gen_dims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.ode(input)
        data = self.seq(data)
        return data

class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, num_split, pack=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.num_split = num_split

        seq = []
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.ode = ODEBlockD(ODEFuncD(dim), self.num_split)

        self.traj_dim = dim * (self.num_split + 1)
        self.last1 = nn.Linear(self.traj_dim, self.traj_dim*2)
        self.last3 = nn.Linear(self.traj_dim*2, self.traj_dim)
        self.last4 = nn.Linear(self.traj_dim, 1)

    def forward(self, x):
        value = x[0]
        time = x[1]
        out = self.seq(value.view(-1, self.packdim))
        out1_time = [out, time]
        out = self.ode(out1_time)
        out = torch.cat(out, dim = 1)

        out = F.leaky_relu(self.last1(out))
        out = F.leaky_relu(self.last3(out))
        out = self.last4(out)
        return out


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


class Cond(object):
    def __init__(self, data, output_info):

        self.model = []

        st = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                max_interval = max(max_interval, ed - st)
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        st = 0
        self.p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                st += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    st += item[0]
                    skip = False
                    continue
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
            else:
                assert 0
        self.interval = np.asarray(self.interval)

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    skip = False
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            skip = True

        elif item[1] == 'softmax':
            if skip:
                skip = False
                st += item[0]
                continue

            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none'
            )
            loss.append(tmp)
            st = ed
            st_c = ed_c

        else:
            assert 0
    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)

        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def calc_gradient_penalty(netD, real_data, fake_data, t_pairs, device='cpu', pac=1, lambda_=10):
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)

    disc_interpolates = netD([interpolates, t_pairs])

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_grad
    return gradient_penalty


class CTGANSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self,
                 embedding_dim=128,
                 gen_dim=(256, 256),
                 dis_dim=(256, 256),
                 l2scale=1e-6,
                 batch_size=args.batch,
                 epochs=300):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   
    def model_load(self, generator_location, dataset_name):
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        self.train = train_data.copy()
        self.test = test_data
        self.meta = meta_data
        self.transformer = BGMTransformer()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim)
        #self.generator = nn.DataParallel(self.generator).to(self.device)

        self.generator.load_state_dict(torch.load(generator_location, map_location = device))

    def model_test(self, times, dataset_name):
        train, test, meta, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)
        lst = []
        for i in range(times):
            
            syn_data = self.sample(train.shape[0])
            tmp = compute_scores(train, test, syn_data, meta)
            tmp["iters"] = i
            lst.append(tmp)
        return pd.concat(lst,axis=0)


    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.train = train_data.copy()
        self.test = test_data
        self.meta = meta_data
        # self.transformer = GeneralTransformer()
        self.transformer = BGMTransformer()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim)

        self.generator = nn.DataParallel(self.generator).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim, args.num_split)

        discriminator = nn.DataParallel(discriminator).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=args.G_lr, betas=(0.5, 0.9), weight_decay=self.l2scale)
        
        optimizerD = optim.Adam(
            discriminator.parameters(), lr=args.D_lr, betas=(0.5, 0.9))

        all_time = ODETime(args.num_split)
        optimizerT = optim.Adam(
            all_time, lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size

        iter = 0
        scores = pd.DataFrame()

        best_score1 = -100
        best_score2 = -100

        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):

                iter += 1
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                ######## update discriminator #########
                y_fake = discriminator([fake_cat,all_time])
                y_real = discriminator([real_cat,all_time])

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, all_time, self.device)                
                
                loss_d = loss_d + pen 
                optimizerD.zero_grad()
                optimizerT.zero_grad()

                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerT.step()

                # clipping time points t.
                with torch.no_grad():
                    for j in range(len(all_time)):
                        if j == 0:
                            start = 0 + 0.00001
                        else:
                            start = all_time[j - 1].item() + 0.00001

                        if j == len(all_time) - 1:
                            end = 1 - 0.00001
                        else:
                            end = all_time[j + 1].item() - 0.00001
                        all_time[j] = all_time[j].clamp_(min=start, max=end)

                ######### update generator ##########
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                if c1 is not None:
                    y_fake = discriminator([torch.cat([fakeact, c1], dim=1), all_time])
                else:
                    y_fake = discriminator([fakeact,all_time])

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
                
                ######### plotting #########
                writer.add_scalar('losses/G_loss', loss_g, iter)
                writer.add_scalar('losses/D_loss', loss_d, iter)

            if True:
            # if i >= 150 and i % 2 == 0:
                # compute scores every epochs
                syn_data = self.sample(train_data.shape[0])
                score = compute_scores(self.train, self.test, syn_data, self.meta) 
                self.generator.train()
                    
                if len(score) == 4: # real-classification: it considers only one score; F1
                    s = score.loc[0].index.to_list()
                    for j in range(len(score)):
                        for k in range(1, 7):
                            writer.add_scalar(score['name'][j]+'/'+s[k], score.iloc[j, k], i)
                            writer.add_scalar('average/'+s[k], score.iloc[:, k].mean(), i)
                    
                    if args.data == "census" or args.data == "credit":
                        avg_score = score.iloc[[0, 1, 3], 2].mean()
                        print("epoch: {:03d} | {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(i, score.iloc[0, 2], score.iloc[1, 2],score.iloc[3, 2],avg_score))
                    
                    else: # covtype and intrusion
                        avg_score = score.iloc[[0, 3], 2].mean()
                        print("epoch: {:03d} | {:.5f}, {:.5f}, {:.5f}".format(i, score.iloc[0, 2], score.iloc[3, 2],avg_score))

                    if avg_score > best_score1:
                        best_score1 = avg_score
                        torch.save(self.generator.state_dict(),save_loc+"/G_{}_best_model_{}.pth".format(dataset_name, time))
                        torch.save(discriminator.state_dict(),save_loc+"/D_{}_best_model_{}.pth".format(dataset_name, time))

                else: # real-regression(r2), gaussian, bayesian: L_syn, L_test
                    s = score.loc[0].index.to_list()
                    for j in range(len(score)):
                        for k in range(1, 5):
                            writer.add_scalar(score['name'][j]+'/'+s[k], score.iloc[j, k] , i)
                            writer.add_scalar('average/'+s[k], score.iloc[:, k].mean(), i)

                    if args.data == "news":
                        avg_score = score.iloc[:, 1].mean()
                        print("epoch: {:03d} | {:.5f}".format(i, avg_score))

                    else:
                        L_syn = score.iloc[:, 1].mean()
                        L_test = score.iloc[:, 2].mean()
                        print("epoch: {:03d} | {:.5f}, {:.5f}".format(i, L_syn, L_test))
                        avg_score = L_test

                    if avg_score > best_score1:
                        best_score1 = avg_score
                        torch.save(self.generator.state_dict(),save_loc+"/G_{}_best_model_{}.pth".format(dataset_name, time))
                        torch.save(discriminator.state_dict(), save_loc+"/D_{}_best_model_{}.pth".format(dataset_name, time))

            times = {'t'+str(t+1):time.item() for t, time in enumerate(all_time)}
            writer.add_scalars('time_points', times, iter)

    def sample(self, n):
        self.generator.eval()
        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)
        # return self.transformer.inverse_transform(data)

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])

# 테스트 시 이하 주석 처리.
for kind in [args.data]:
    print(kind)
    a,b = benchmark(CTGANSynthesizer, datasets=[kind], iterations=1, add_leaderboard=False)
    a.to_csv(save_loc + "/{}_final1_{}.csv".format(kind, time))
    b.to_csv(save_loc + "/{}_final2_{}.csv".format(kind, time))
