import numpy as np
import tqdm

import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image

__all__ = ['AnnealRunner']

def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score, device_ids=[1])

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score, device_ids=[1])

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 5

        imgs = []
        if self.config.data.dataset == 'MNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)