import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from pdb import set_trace as st
import math


# save image to the disk
def save_images(webpage, images_list, names_list, image_path, title=None, width=256):
    image_dir = webpage.get_image_dir()
    # name = os.path.splitext(short_path)[0]
    if image_path == None:
        name = None
        title = ntpath.basename(title)
    else:
        short_path = ntpath.basename(image_path)
        name = short_path
        
    if not title:
        title = name
        
    webpage.add_header(title)
    ims = []
    txts = []
    links = []
    
    for names, images in zip(names_list, images_list):
        for label, image_numpy in zip(names, images):
            if name == None:
                image_name = '%s.jpg' % (label)
            else:
                image_name = '%s_%s.jpg' % (name, label)
                
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.log_path = os.path.join(opt.expr_dir, 'train_log.txt')

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.expr_dir,'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, ncols=2, save_result=False, image_format='jpg'):
        if self.display_id > 0:  # show images in the browser
            title = self.name
            nrows = int(math.ceil(len(visuals.items()) / float(ncols)))
            images = []
            idx = 0
            for label, image_numpy in visuals.items():
                title += " | " if idx % nrows == 0 else ", "
                title += label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
            if len(visuals.items()) % ncols != 0:
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                images.append(white_image)
            self.vis.images(images, nrow=nrows, win=self.display_id + 1,
                            opts=dict(title=title))

        if self.use_html and save_result:  # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.%s' % (epoch, label, image_format))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.%s' % (n, label, image_format)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += ', %s: %.3f' % (k, v)

        print(message)
        # write losses to text file as well
        with open(self.log_path, "a") as log_file:
            log_file.write(message)

    # save image to the disk
    def save_images_old(self, webpage, visuals, image_path, short=False):
        image_dir = webpage.get_image_dir()
        if short:
            short_path = ntpath.basename(image_path)
            name = os.path.splitext(short_path)[0]
        else:
            name = image_path

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
