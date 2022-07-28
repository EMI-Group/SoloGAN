from __future__ import print_function
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid

from models.model_cls import Discriminator, Generator, Style, Content, weights_init


from util.loss import GANLoss, VGGLoss
from util.util import tensor2im
import numpy as np
import torch.nn.functional as F
import random
import time


class SoloGAN():
    def name(self):
        return 'SoloGAN'

    def initialize(self, opt):
        cudnn.benchmark = True
        self.opt = opt
        self.build_models()

    def print_networks(self, net, name):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        
    def build_models(self):
                            ######### generator #########
        self.G = Generator(ngf=self.opt.ngf, ns=self.opt.c_num, nd=self.opt.d_num)
        self.print_networks(self.G, 'Generator')
        
                            ######### encoder #########
        self.S = Style(output_nc=self.opt.c_num, nef=self.opt.nef, nd=self.opt.d_num, adding_CIN=self.opt.Add_CIN_SE)
        self.print_networks(self.S, 'Style Encoder')
        
        self.C = Content(dim=self.opt.ngf, nd=self.opt.d_num)
        self.print_networks(self.C, 'Content Encoder')
        
        if self.opt.isTrain:
                            ######## discriminators #########
            self.Ds = Discriminator(ndf=self.opt.ndf, block_num=self.opt.dis_nums, nd=self.opt.d_num)    # 4, multi-scale

            # init_weights
            if self.opt.continue_train:
                self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.S.load_state_dict(torch.load('{}/S_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.C.load_state_dict(torch.load('{}/C_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.Ds.load_state_dict(torch.load('{}/D_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            else:
                self.G.apply(weights_init(self.opt.init_type))
                self.S.apply(weights_init(self.opt.init_type))
                self.C.apply(weights_init(self.opt.init_type))
                self.Ds.apply(weights_init(self.opt.init_type))
                    
            # use GPU
            self.G.cuda()
            self.S.cuda()
            self.C.cuda()
            self.Ds.cuda()

            # set criterion
            self.criterionGAN = GANLoss(mse_loss=True)
            if self.opt.lambda_vgg > 0:
                self.vggloss = VGGLoss(device=self.opt.gpu)

            # define optimizers
            self.G_opt = self.define_optimizer(self.G, self.opt.G_lr)
            self.S_opt = self.define_optimizer(self.S, self.opt.G_lr)
            self.C_opt = self.define_optimizer(self.C, self.opt.G_lr)
            self.Ds_opt = self.define_optimizer(self.Ds, self.opt.D_lr)
        else:
            self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            self.G.cuda()
            self.G.eval()
            if self.C is not None:
                self.S.load_state_dict(torch.load('{}/S_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.S.cuda()
                self.S.eval()
                self.C.load_state_dict(torch.load('{}/C_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.C.cuda()
                self.C.eval()

    def sample_latent_code(self, size):
        c = torch.cuda.FloatTensor(size).normal_()
        return Variable(c)

    def get_domain_code(self, domainLable):     # change the labels into one-hot vector
        domainCode = torch.zeros([len(domainLable), self.opt.d_num])
        for index in range(len(domainLable)):  # using 1 instead
            domainCode[index, domainLable[index]] = 1
        return Variable(domainCode).cuda()

    def define_optimizer(self, Net, lr):
        return optim.Adam(Net.parameters(), lr=lr, betas=(0.5, 0.999))

    def prepare_image(self, data):
        img, sourceD, name = data
        return Variable(torch.cat(img, 0)).cuda(), torch.cat(sourceD, 0), name

    def update_D(self, D, D_opt, real, fake):
        D.zero_grad()
        pred_fake = D(fake.detach())
        pred_real = D(real)
        errD = self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_real, True)
        errD.backward()
        D_opt.step()
        return errD

    def calculate_G(self, D, fake):
        pred_fake = D(fake)
        errG = self.criterionGAN(pred_fake, True)
        return errG

    def classification_loss(self, logit, target):
        cls_err = 0
        for i, cls in enumerate(logit):
            if cls is not None:
                cls_err += F.cross_entropy(cls.squeeze(), target.view(target.size(0)))
                # print("    ### Using Classifier ###    ")
        return cls_err

    def update_model(self, data):
        self.real, sourceD, _ = self.prepare_image(data)      # prepare data

        # get the targetD by random selection
        index = []
        for i_ in range(self.opt.d_num):
            a = random.randint(0, self.opt.d_num-1)
            while i_ == a:
                a = random.randint(0, self.opt.d_num-1)
            index.append(a)
        targetD = sourceD[index]    # target domain label

        sourceDV = self.get_domain_code(sourceD)  # one-hot vector of source label
        targetDV = self.get_domain_code(targetD)  # one-hot vector of target label
        sourceD = Variable(torch.LongTensor(sourceD)).cuda()
        targetD = Variable(torch.LongTensor(targetD)).cuda()

        content = self.C(self.real)
        style = self.S(self.real, sourceDV)
        s_rand = self.sample_latent_code(style.size())      # ramdom selected style codes
        source = torch.cat([sourceDV, style], 1)    # style is concated with one-hot vector
        target = torch.cat([targetDV, s_rand], 1)   # style is concated with one-hot vector

        self.fake = self.G(content, target)
        content_rec = self.C(self.fake)

                        ### update D ###
        self.errDs = 0
        self.Ds.zero_grad()
        dis_real, cls_real = self.Ds(self.real, sourceD.cuda())
        dis_fake, _ = self.Ds(self.fake.detach(), targetD.cuda())
        self.errDs = self.criterionGAN(dis_fake, False) + self.criterionGAN(dis_real, True)

        self.errDs += self.classification_loss(cls_real, sourceD.cuda())
        errDs = self.errDs
        errDs.backward(retain_graph=True)
        self.Ds_opt.step()
        self.Ds.zero_grad()

                        ### update G ###
        errG_total = 0
        self.G.zero_grad()
        self.C.zero_grad()
        self.S.zero_grad()

        # adversarial loss
        dis_fake, cls_fake = self.Ds(self.fake, targetD.cuda())
        errG = self.criterionGAN(dis_fake, True)
        errG += self.classification_loss(cls_fake, targetD.cuda())
        errG_total += errG
        
        # image reconstruction,
        if self.opt.lambda_rec > 0:
            self.rec = self.G(content, source)      # rec = real
            self.errRec = torch.mean(torch.abs(self.rec - self.real)) * self.opt.lambda_rec
            errG_total += self.errRec
        else:
            self.rec = None
            print("    ###   Training Without self reconstruction  ###   ")
            
        # Latent reconstruction
        if self.opt.lambda_c > 0:
            self.errRec_c = torch.mean(torch.abs(content_rec - content)) * self.opt.lambda_c
            errG_total += self.errRec_c
            style_rec = self.S(self.fake, targetDV)
            self.errRec_s = torch.mean(torch.abs(style_rec - s_rand)) * self.opt.lambda_c
            errG_total += self.errRec_s
        else:
            print("    ###   Training Without latent reconstruction  ###   ")
        
        # cycle reconstruction
        if self.opt.lambda_cyc > 0:
            self.cyc = self.G(content_rec, source)
            self.errCyc = torch.mean(torch.abs(self.cyc - self.real)) * self.opt.lambda_cyc
            errG_total += self.errCyc
        else:
            self.cyc = None
            print("    ###   Training Without cycle reconstruction  ###   ")

        if self.opt.lambda_vgg > 0:
            self.perceptual_loss = self.vggloss(self.real, self.fake)
            errG_total += self.perceptual_loss * self.opt.lambda_vgg
        else:
            print("    ###   Training Without Perceptual loss  ###   ")

        errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.S_opt.step()
        self.C_opt.step()
        self.G.zero_grad()
        self.S.zero_grad()
        self.C.zero_grad()

    def get_current_visuals(self):
        real = make_grid(self.real.data, nrow=self.real.size(0), padding=0)
        fake = make_grid(self.fake.data, nrow=self.real.size(0), padding=0)
        
        if self.opt.lambda_rec == 0:
            self.rec = self.real
        rec = make_grid(self.rec.data, nrow=self.real.size(0), padding=0)
        
        if self.opt.lambda_cyc == 0:
            self.cyc = self.real
        cyc = make_grid(self.cyc.data, nrow=self.real.size(0), padding=0)
        
        img = [real, rec, fake, cyc]
        name = 'src, rec, fake, cyc'
        img = torch.cat(img, 1)
        return OrderedDict([(name, tensor2im(img))])

    def translation(self, data, domain_names=None):
        input, sourceD, img_names = self.prepare_image(data)
        sub_names = []
        if self.opt.name == 'Office31':
            for i in range(len(img_names)):
                img_name = img_names[i]
                list = img_name[0].split('/')
                print(list)
                sub_name = '{}_{}'.format(list[-2], list[-1].split('.jpg')[0])
                print(sub_name)
                sub_names.append(sub_name)
            
        sourceDC = self.get_domain_code(sourceD)
        print(sourceD)

        images, names = [], []
        for i in range(self.opt.d_num):
            images.append([])
            names.append([])
            
        content = self.C(input)
        
        # input
        for i in range(max(sourceD) + 1):
            images[i].append(tensor2im(input[i].data))
            if self.opt.name == 'Office31':
                names[i].append('{}'.format(sub_names[i]))
            else:
                names[i].append('D_{}'.format(i))

        # get the targetD by select given style ramdonly
        if self.opt.d_num == 2:
            indexs = [[1, 0]]
        elif self.opt.d_num == 3:
            indexs = [[1, 2, 0], [2, 0, 1]]
        elif self.opt.d_num == 4:  # self.opt.d_num = 4
            indexs = [[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        else:  # 5
            indexs = [[1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]]

        for targetD in indexs:
            targetDv = self.get_domain_code(targetD)
        # random sample, multi-modal
            for i in range(self.opt.n_samples):
                s_rand = self.sample_latent_code(torch.Size([input.size(0), self.opt.c_num]))
                target = torch.cat([targetDv, s_rand], 1)   # torch.LongTensor(
                output = self.G(content, Variable(target).cuda())
                for j in range(output.size(0)):
                    images[sourceD[j]].append(tensor2im(output[j].data))
                    if domain_names == None:
                        if self.opt.name == 'Office31': 
                            names[sourceD[j]].append('{}_{}_{}2{}'.format(sub_names[j], i, sourceD[j], targetD[j]))
                        else:
                            names[sourceD[j]].append('{}_{}2{}'.format(i, sourceD[j], targetD[j])) 
                    else:
                        if self.opt.name == 'Office31': 
                            names[sourceD[j]].append('{}_{}_{}2{}'.format(sub_names[j], i, domain_names[sourceD[j]], domain_names[targetD[j]]))
                        else:
                            names[sourceD[j]].append('{}_{}2{}'.format(i, domain_names[sourceD[j]], domain_names[targetD[j]]))

        return images, names

    def translation_time(self, data, domain_names=None):
        input, sourceD, _ = self.prepare_image(data)
        targetDC = self.get_domain_code(sourceD)

        images, names = [], []
        for i in range(self.opt.d_num):
            images.append([])
            names.append([])
            
        for i in range(max(sourceD) + 1):
            images[i].append(tensor2im(input[i].data))
            names[i].append('D_{}'.format(i))
                
        img = input[0].unsqueeze(0)
        
        
        c_rand = self.sample_latent_code(torch.Size([2, self.opt.c_num]))
        targetC = torch.cat([targetDC, c_rand], 1)[0]
        
        start = time.time()
        content = self.C(img)
        for i in range(100):
            output = self.G(content, targetC)
        end = time.time()
        print("    ### Generating 100 images use %.3f    ###"%(end - start))

        return images, names
        
    def update_lr(self, D_lr, G_lr):
        for param_group in self.G_opt.param_groups:
            param_group['lr'] = G_lr
        for param_group in self.S_opt.param_groups:
            param_group['lr'] = G_lr
        for param_group in self.C_opt.param_groups:
            param_group['lr'] = G_lr
            
        for param_group in self.Ds_opt.param_groups:
            param_group['lr'] = D_lr

    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.S.state_dict(), '{}/S_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.C.state_dict(), '{}/C_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.Ds.state_dict(), '{}/D_{}.pth'.format(self.opt.model_dir, name))
