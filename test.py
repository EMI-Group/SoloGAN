import os
from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from util.visualizer import save_images
from itertools import islice

from models.solver import SoloGAN
from util import html, util
import torch

opt = TestOptions().parse()
opt.n_samples = 5
opt.how_many = 100  # 250
opt.isTrain = False
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)

dataset_name = opt.name
opt.dataroot = '{}/{}'.format(opt.dataroot, dataset_name)

opt.no_flip = True
opt.batchSize = 1

data_loader = CreateDataLoader(opt)

model = SoloGAN()
model.initialize(opt)

img_dir = '{}/images'.format(opt.results_dir)
if os.path.isdir(img_dir):
    for file in os.listdir(img_dir):
        os.remove("{}/{}".format(img_dir, file))

web_dir = os.path.join(opt.results_dir)

webpage = html.HTML(web_dir, 'task {}'.format(opt.name))

if opt.name == 'cat_dog_tiger':
    domain_names = ['cat', 'dog', 'tiger']

elif opt.name == 'portrait':
    domain_names = ['portrait', 'face']

elif opt.name == 'leopard_lion_tiger_bobcat':
    domain_names = ['leopard', 'lion', 'tiger', 'bobcat']

elif opt.name == 'seasons':
    domain_names = ['autum', 'spring', 'summer', 'winter']

elif opt.name == 'hairs':
    domain_names = ['black', 'blond', 'brown']

elif opt.name == 'Segmentations_256':
    domain_names = ['cityscapes', 'gta5', 'bdd']

elif opt.name == 'photo2arts':
    domain_names = ['photo', 'Vangh', 'Monet', 'Cezzan']

elif opt.name == 'Office31':
    domain_names = ['amazon', 'dslr', 'webcam']

elif opt.name == 'vkitti':
    domain_names = ['frog', 'sunny', 'rain']

elif opt.name == 'facades':
    domain_names = ['label', 'facades']

elif opt.name == 'eyeglasses' or opt.name == 'bangs' or opt.name == 'beard' or opt.name == 'smiling' or opt.name == 'male':
    domain_names = ['yes', 'no']

elif opt.name == 'Cityscapes_256' or opt.name == 'maps':
    domain_names = ['photo', 'seg']

elif opt.name == 'face_detection':
    domain_names = ['photo', 'det']

elif opt.name == 'text_deblurred':
    domain_names = ['blurred', 'sharp']

elif opt.name == 'ages1':
    domain_names = ['0_3', '12_17', '30_40', '56_65', '81_116']

elif opt.name == 'ages2':
    domain_names = ['4_11', '18_29', '41_55', '66_80']

else:
    domain_names = opt.name.split('2')


def test():
    for i, data in enumerate(islice(data_loader, opt.how_many)):
        print('process input image %3.3d/%3.3d' % (i, opt.how_many))
        with torch.no_grad():
            all_images, all_names = model.translation(data, domain_names)
            #            all_images, all_names = model.translation_time(data, domain_names)
            img_path = 'image%3.3i' % i

            # all_images, all_names = model.trans_single(data, domain_names)
            # img_path = None
        save_images(webpage, all_images, all_names, img_path, title='image%3.3i' % i,
                    width=opt.img_size)  # opt.img_size
    webpage.save()


if __name__ == '__main__':
    test()
