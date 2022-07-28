import argparse
import os
from util import util
import pickle
import datetime
import dateutil.tz


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    #  male, smiling, eyeglasses, bangs, beard   
     
        # summer2winter, cat2dog, apple2orange, edges2shoes_handbags, day2night, horse2zebra_AWA
        
        # vkitti, cat_dog_tiger, hairs, collie_shepherd_fox
        
        # seasons, photo2arts, leopard_lion_tiger_bobcat
        
        # unimodal translation: Cityscapes_256, Cityscapes_512, maps, facades, blurred2sharp, face_detection, text_deblurred
        
        # age regression: ages1, ages2, selfie2anime, portrait, landscape2photo

        self.parser.add_argument('--dataroot', default='/raid/huangsh/datasets', help='path to images (should have subfolders trainA, trainB, testA, testB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--name', type=str, default='landscape2photo', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dir_name', type=str, default='SNEG_cls_100')  # latent, cyc, SoloGAN_uni, _128, _200, SoloGAN_no_latent
        self.parser.add_argument('--gpu', type=int, default=10, help='gpu id')

        self.parser.add_argument('--dis_nums', type=int, default=6, help='then crop to this size')  # 286, 572
        
        self.parser.add_argument('--format', type=str, default='jpg')
        self.parser.add_argument('--d_num', type=int, default=2, help='# of domain number')
        self.parser.add_argument('--c_num', type=int, default=8, help='#of latent dimension')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/raid/huangsh/SoloGAN_improved/checkpoints', help='models are saved here')    # /raid/emilab/SoloGAN_improved_V2/checkpoints

        self.parser.add_argument('--crop_size', type=int, default=256, help='scale images to this size')  # 128, 256, 512
        self.parser.add_argument('--img_size', type=int, default=286, help='then crop to this size')  # 158, 286, 572
        
        self.parser.add_argument('--Add_CIN_CE', type=bool, default=False, help='Whether adding CIN into Content_Encoder')      # True, False
        self.parser.add_argument('--Add_CIN_SE', type=bool, default=True, help='Whether adding CIN into Style_Encoder')

        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')    # 64, 48
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--e_blocks', type=int, default=6, help='# of encoder blocks')
        self.parser.add_argument('--up_type', type=str, default='Trp', help='upsample type, Ner: nearest upsample with convolution, Trp: transposed convolution')
        self.parser.add_argument('--up_paired', action='store_true', help='if specified, use unpaired datasets')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# sthreads for loading data')
        
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8000,  help='visdom display port')
        self.parser.add_argument('--c_gan_mode', type=str, default='lsgan', help='use dcgan or lsgan')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # lambda parameters
        self.opt.lambda_vgg = 10.0  # 0
        self.opt.lambda_cyc = 10.0  # 0
        self.opt.lambda_rec = 10.0  # 0
        self.opt.lambda_c = 1.0   # 0
            
        # epochs
        self.opt.niter = 50
        self.opt.niter_decay = 50
        
        if self.opt.name == 'ages1':
            self.opt.niter = 50
            self.opt.niter_decay = 50
            self.opt.d_num = 5
            self.opt.crop_size = 128
            self.opt.img_size = 140
            
        if self.opt.name == 'ages2':
            self.opt.niter = 40
            self.opt.niter_decay = 40
            self.opt.d_num = 4
            self.opt.crop_size = 128
            self.opt.img_size = 140
            
        self.opt.isTrain = self.isTrain  # train or test
        if not self.opt.isTrain:
            if self.opt.crop_size == 128:
                self.opt.img_size = 128
            
            if self.opt.crop_size == 256:
                self.opt.img_size = 256
                
            if self.opt.crop_size == 512:
                self.opt.img_size = 512   
                        
        if self.opt.crop_size == 512:
            self.opt.dis_nums = 7
            
        if self.opt.crop_size == 128:
            self.opt.dis_nums = 5

        if self.opt.name == 'hairs' or self.opt.name =='edges2shoes_handbags':
            self.opt.niter = 30
            self.opt.niter_decay = 30
        
        if self.opt.name == 'vkitti' or self.opt.name == 'seasons':
            self.opt.niter = 60
            self.opt.niter_decay = 60
            
        if self.opt.name == 'hairs' or self.opt.name == 'vkitti' or self.opt.name == 'cat_dog_tiger':
            self.opt.d_num = 3
        
        if self.opt.name == 'photo2arts' or self.opt.name == 'seasons' or self.opt.name == 'leopard_lion_tiger_bobcat':
            self.opt.d_num = 4

        if self.opt.name == 'hairs' or self.opt.name == 'male' or self.opt.name == 'eyeglasses' or self.opt.name == 'bangs':
            self.opt.crop_size = 178
            self.opt.img_size = 256
            
        self.opt.D_lr = 0.0002
        self.opt.G_lr = 0.0002

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        self.opt.expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.dir_name)
        self.opt.model_dir = os.path.join(self.opt.expr_dir, 'model')
        
        if self.isTrain:
            util.mkdirs(self.opt.model_dir)
            pkl_file = os.path.join(self.opt.expr_dir, 'opt.pkl')
            pickle.dump(self.opt, open(pkl_file, 'wb'))

            # save to the disk
            file_name = os.path.join(self.opt.expr_dir, 'opt_train.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        else:
            if self.opt.results_dir == '':
                self.opt.results_dir = os.path.join(self.opt.expr_dir,  '{}_{}_results'.format(self.opt.name, self.opt.dir_name))
                      
            results_dir = self.opt.results_dir
            util.mkdirs(results_dir)
        return self.opt
