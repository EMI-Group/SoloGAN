import time
from options.train_options import TrainOptions
from data.dataloader import CreateDataLoader
from util.visualizer import Visualizer
from models.solver import SoloGAN

opt = TrainOptions().parse()
dataset_name = opt.name
opt.dataroot = '{}/{}'.format(opt.dataroot, dataset_name)

data_loader = CreateDataLoader(opt)
dataset_size = len(data_loader) * opt.batchSize
visualizer = Visualizer(opt)

model = SoloGAN()
model.initialize(opt)


def train():
    total_steps = 0
    D_lr = opt.D_lr
    G_lr = opt.G_lr
    total_epoch = opt.niter + opt.niter_decay + 1
    for epoch in range(1, total_epoch):
        epoch_start_time = time.time()
        save_result = True
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.update_model(data)
            t = (time.time() - iter_start_time)
            print('epoch: {}/{}, iters={}: time={}'.format(epoch, total_epoch, i, t))

            if save_result or total_steps % opt.display_freq == 0:
                save_result = save_result or total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, ncols=1, save_result=save_result)
                save_result = False

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        if epoch > opt.niter:
            D_lr -= opt.D_lr / opt.niter_decay
            G_lr -= opt.G_lr / opt.niter_decay
            model.update_lr(D_lr, G_lr)
    model.save('latest')
    model.save(epoch)


if __name__ == '__main__':
    train()