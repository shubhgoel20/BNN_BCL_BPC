import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from .utils import BayesianSGD
from .common import *
import random


class Appr(object):

    def __init__(self,model,args,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=1000):
        print("UCB New2")
        self.model=model
        self.device = args.device
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.init_lr=args.lr
        self.sbatch=args.sbatch
        self.nepochs=args.nepochs

        self.arch=args.arch
        self.samples=args.samples
        self.lambda_=1.

        self.output=args.output
        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.shared_model_cache = shared_model_task_cache

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

        self.prev_task_data = self.shared_model_cache["prev_task_data"]
        self.task_freq = self.shared_model_cache["task_frquencies"]
        self.replay_buffer_perc = args.rbuff_size

    def get_prob_from_freq(self):
        # probs = {key:1/value for key,value in self.task_freq.items()}
        # sum_ = sum(probs.values())
        # probs = {key:value/sum_ for key,value in probs.items()}
        n = len(self.task_freq.keys())
        probs = {key:1/n for key in self.task_freq.keys()}
        return probs

    def sample_prev_task(self, probs):
        keys = list(probs.keys())
        values = list(probs.values())
        return random.choices(keys, weights=values, k=1)[0]

    def get_buffer(self, prev_task, buffer_size):
        prev_xtrain = self.prev_task_data[prev_task]['xtrain']
        prev_ytrain = self.prev_task_data[prev_task]['ytrain']
        n_ = prev_xtrain.shape[0]
        if(n_ <= buffer_size): return prev_xtrain, prev_ytrain
        indices = np.random.choice(n_,buffer_size, replace=False)
        prev_xtrain = prev_xtrain[indices]
        prev_ytrain = prev_ytrain[indices]        
        return prev_xtrain, prev_ytrain

    def train(self,task_num,xtrain,ytrain,xvalid,yvalid):

        # Update the next learning rate for each parameter based on their uncertainty
        update_last_task(task_num)
        params_dict = self.get_model_params()
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss=np.inf

        # best_model=copy.deepcopy(self.model)
        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience

        # Loop epochs
        try:
            for e in range(self.nepochs):
                if(len(self.prev_task_data.keys()) > 0):
                    probs = self.get_prob_from_freq()
                    sampled_task = self.sample_prev_task(probs)
                    self.task_freq[sampled_task]+=1
                    prev_xtrain, prev_ytrain = self.get_buffer(sampled_task, int(self.replay_buffer_perc*xtrain.shape[0]))
                    self.sampled_task = sampled_task
                    # xtrain = torch.cat((xtrain,prev_xtrain), dim=0)
                    # ytrain = torch.cat((ytrain,prev_ytrain), dim=0)
                else:
                    prev_xtrain, prev_ytrain = None, None
                clock0=time.time()
                self.train_epoch(task_num, xtrain, ytrain, prev_xtrain, prev_ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(task_num,xtrain,ytrain)
                clock2=time.time()

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),
                    train_loss,100*train_acc),end='')
                valid_loss,valid_acc=self.eval(task_num,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.shared_model_cache["models"][task_num] = copy.deepcopy(self.model.state_dict())
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience

                        params_dict = self.update_lr(task_num, lr=lr)
                        self.optimizer=BayesianSGD(params=params_dict)

                print()
        except KeyboardInterrupt:
            print()
        self.task_freq[task_num] = int(1/self.replay_buffer_perc)

        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)



    def get_model_params(self):
        params_dict = []
        params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
        return params_dict

    def update_lr(self,task_num, lr=None):
        params_dict = []
        if task_num==0:
            params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
        else:
            for name in self.modules_names_without_cls:
                n = name.split('.')
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                else:
                    print(name)
                params_dict.append({'params': m.weight_rho, 'lr': lr})
                params_dict.append({'params': m.bias_rho, 'lr': lr})

        return params_dict


    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)
        if not with_classifier:
            shared_model_task_cache["modules_names_without_cls"] = modules_names
        return modules_names

    def logs(self,task_num):
        lp_, lvp = 0.0, 0.0
        for name in self.modules_names_without_cls:
            n = name.split('.')
            if len(n) == 1:
                m = self.model._modules[n[0]]
            elif len(n) == 3:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
            
            lp_ += m.log_prior
            lvp += m.log_variational_posterior

        lp__, last_model_available = get_log_posterior_from_last_task(self.model)
        lp = lp__ if last_model_available else lp_
        lp += self.model.classifier[task_num].log_prior
        lvp += self.model.classifier[task_num].log_variational_posterior

        return lp, lvp


    def train_epoch(self, task_num, x, y, prevx = None, prevy = None):
        self.train_epoch_(task_num, x, y)
        if prevx is not None and prevy is not None:
            self.train_epoch_(self.sampled_task, prevx, prevy)

    def train_epoch_(self,task_num, x, y):

        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(self.device)

        num_batches = len(x)//self.sbatch
        # Loop batches
        for i in range(0,len(r),self.sbatch):

            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            # Forward
            loss=self.elbo_loss(images,targets,task_num,num_batches,sample=True).to(self.device)

            # Backward
            self.model.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.model.cuda()

            # Update parameters
            self.optimizer.step()
        return


    def eval(self,task_num,x,y,debug=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            num_batches = len(x)//self.sbatch
            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                outputs=self.model(images,sample=False)
                output=outputs[task_num]
                loss = self.elbo_loss(images, targets, task_num, num_batches,sample=False,debug=debug)

                _,pred=output.max(1, keepdim=True)

                total_loss += loss.detach()*len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += len(b)           

        return total_loss/total_num, total_acc/total_num

    def elbo_loss(self, input, target, task_num, num_batches, sample, debug=False):
        if sample:
            lps, lvps, predictions = [], [], []
            for i in range(self.samples):
                predictions.append(self.model(input, sample=sample)[task_num])
                lp, lv = self.logs(task_num)
                lps.append(lp)
                lvps.append(lv)

            # hack
            w1 = 1.e-3
            w2 = 1.e-3
            w3 = 5.e-2

            outputs = torch.stack(predictions,dim=0).to(self.device)
            log_var = w1*torch.as_tensor(lvps, device=self.device).mean()
            log_p = w2*torch.as_tensor(lps, device=self.device).mean()
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return (log_var - log_p)/num_batches + nll

        else:
            predictions = []
            for i in range(self.samples):
                pred = self.model(input,sample=False)[task_num]
                predictions.append(pred)
            w3 = 5.e-6

            outputs = torch.stack(predictions,dim=0).to(self.device)
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return nll

    def save_model(self,task_num):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_num)))
