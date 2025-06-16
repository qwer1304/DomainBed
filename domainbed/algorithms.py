# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
            LARS,  SupConLossLambda
    )
import os
import logging

ALGORITHMS = [
    'ERM',
    'ERMPlusPlus',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'EQRM',
    'RDM',
    'ADRMX',
    'URM',
    'GLSD_SSD',
    'GLSD_FSD',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERMPlusPlus(Algorithm,ErmPlusPlusMovingAvg):
    """
    Empirical Risk Minimization with improvements (ERM++)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self,input_shape, num_classes, num_domains,hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        if self.hparams["lars"]:
            self.optimizer = LARS(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        linear_parameters = []
        for n, p in self.network[1].named_parameters():
            linear_parameters.append(p)

        if self.hparams["lars"]:
            self.linear_optimizer = LARS(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 1)
        ErmPlusPlusMovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):

        if self.global_iter > self.hparams["linear_steps"]:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer



        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)

        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        self.update_sma()
        if not self.hparams["freeze_bn"]:
            self.network_sma.train()
            self.network_sma(all_x)

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)

    def set_lr(self, eval_loaders_iid=None, schedule=None,device=None):
        with torch.no_grad():
             if self.global_iter > self.hparams["linear_steps"]:
                 if schedule is None:
                     self.network_sma.eval()
                     val_losses = []
                     for loader in eval_loaders_iid:
                         loss = 0.0
                         for x, y in loader:
                             x = x.to(device)
                             y = y.to(device)
                             loss += F.cross_entropy(self.network_sma(x),y)
                         val_losses.append(loss / len(loader ))
                     val_loss = torch.mean(torch.stack(val_losses))
                     self.scheduler.step(val_loss)
                     self.lr_schedule.append(self.scheduler._last_lr)
                     if len(self.lr_schedule) > 1:
                         if self.lr_schedule[-1] !=  self.lr_schedule[-2]:
                            self.lr_schedule_changes += 1
                     if self.lr_schedule_changes == 3:
                         self.lr_schedule[-1] = [0.0]
                     return self.lr_schedule
                 else:
                     self.optimizer.param_groups[0]['lr'] = (torch.Tensor(schedule[0]).requires_grad_(False))[0]
                     schedule = schedule[1:]
             return schedule

class URM(ERM):
    """
    Implementation of Uniform Risk Minimization, as seen in Uniformly Distributed Feature Representations for
    Fair and Robust Learning. TMLR 2024 (https://openreview.net/forum?id=PgLbS5yp8n)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)

        # setup discriminator model for URM adversarial training
        self._setup_adversarial_net()

        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _modify_generator_output(self):
        print('--> Modifying encoder output:', self.hparams['urm_generator_output'])
        
        from domainbed.lib import wide_resnet
        assert type(self.featurizer) in [networks.MLP, networks.MNIST_CNN, wide_resnet.Wide_ResNet, networks.ResNet]

        if self.hparams['urm_generator_output'] == 'tanh':
            self.featurizer.activation = nn.Tanh()

        elif self.hparams['urm_generator_output'] == 'sigmoid':
            self.featurizer.activation = nn.Sigmoid()
        
        elif self.hparams['urm_generator_output'] == 'identity':
            self.featurizer.activation = nn.Identity()

        elif self.hparams['urm_generator_output'] == 'relu':
            self.featurizer.activation = nn.ReLU()

        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

    def _setup_adversarial_net(self):
        print('--> Initializing discriminator <--')        
        self.discriminator = self._init_discriminator()
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss(reduction="mean") # apply on logit

        # featurizer optimized by self.optimizer only
        if self.hparams["urm_discriminator_optimizer"] == 'sgd':
            self.discriminator_opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'], momentum=0.9)
        elif self.hparams["urm_discriminator_optimizer"] == 'adam':
            self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'])
        else:
            raise Exception('%s unimplemented' % self.hparams["urm_discriminator_optimizer"])

        self._modify_generator_output()
        self.sigmoid = nn.Sigmoid() # to compute discriminator acc.
            
    def _init_discriminator(self):
        """
        3 hidden layer MLP
        """
        model = nn.Sequential()
        model.add_module("dense1", nn.Linear(self.featurizer.n_outputs, 100))
        model.add_module("act1", nn.LeakyReLU())

        for _ in range(self.hparams['urm_discriminator_hidden_layers']):            
            model.add_module("dense%d" % (2+_), nn.Linear(100, 100))
            model.add_module("act2%d" % (2+_), nn.LeakyReLU())

        model.add_module("output", nn.Linear(100, 1)) 
        return model

    def _generate_noise(self, feats):
        """
        If U is a random variable uniformly distributed on [0, 1), then (b-a)*U + a is uniformly distributed on [a, b).
        """
        if self.hparams['urm_generator_output'] == 'tanh':
            a,b = -1,1
        elif self.hparams['urm_generator_output'] == 'relu':
            a,b = 0,1
        elif self.hparams['urm_generator_output'] == 'sigmoid':
            a,b = 0,1
        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

        uniform_noise = torch.rand(feats.size(), dtype=feats.dtype, layout=feats.layout, device=feats.device) # U~[0,1]
        n = ((b-a) * uniform_noise) + a # n ~ [a,b)
        return n

    def _generate_soft_labels(self, size, device, a ,b):
        # returns size random numbers in [a,b]
         uniform_noise = torch.rand(size, device=device) # U~[0,1]
         return ((b-a) * uniform_noise) + a

    def get_accuracy(self, y_true, y_prob):
        # y_prob is binary probability
        assert y_true.ndim == 1 and y_true.size() == y_prob.size()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def return_feats(self, x):
        return self.featurizer(x)

    def _update_discriminator(self, x, y, feats):
        # feats = self.return_feats(x)
        feats = feats.detach() # don't backbrop through encoder in this step
        noise = self._generate_noise(feats)
        
        noise_logits = self.discriminator(noise) # (N,1)
        feats_logits = self.discriminator(feats) # (N,1)

        # hard targets
        hard_true_y = torch.tensor([1] * noise.shape[0], device=noise.device, dtype=noise.dtype) # [1,1...1] noise is true
        hard_fake_y = torch.tensor([0] * feats.shape[0], device=feats.device, dtype=feats.dtype) # [0,0...0] feats are fake (generated)

        if self.hparams['urm_discriminator_label_smoothing']:
            # label smoothing in discriminator
            soft_true_y = self._generate_soft_labels(noise.shape[0], noise.device, 1-self.hparams['urm_discriminator_label_smoothing'], 1.0) # random labels in range
            soft_fake_y = self._generate_soft_labels(feats.shape[0], feats.device, 0, 0+self.hparams['urm_discriminator_label_smoothing']) # random labels in range
            true_y = soft_true_y
            fake_y = soft_fake_y
        else:
            true_y = hard_true_y
            fake_y = hard_fake_y

        noise_loss = self.discriminator_loss(noise_logits.squeeze(1), true_y) # pass logits to BCEWithLogitsLoss
        feats_loss = self.discriminator_loss(feats_logits.squeeze(1), fake_y) # pass logits to BCEWithLogitsLoss

        d_loss = 1*noise_loss + self.hparams['urm_adv_lambda']*feats_loss

        # update discriminator
        self.discriminator_opt.zero_grad()
        d_loss.backward()
        self.discriminator_opt.step()

    def _compute_loss(self, x, y):
        feats = self.return_feats(x)
        ce_loss = self.loss(self.classifier(feats), y).mean()

        # train generator/encoder to make discriminator classify feats as noise (label 1)
        true_y = torch.tensor(feats.shape[0]*[1], device=feats.device, dtype=feats.dtype)
        g_logits = self.discriminator(feats)
        g_loss = self.discriminator_loss(g_logits.squeeze(1), true_y) # apply BCEWithLogitsLoss to discriminator's logit output
        loss = ce_loss + self.hparams['urm_adv_lambda']*g_loss

        return loss, feats

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
            
        loss, feats = self._compute_loss(all_x, all_y)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self._update_discriminator(all_x, all_y, feats)
    
        return {'loss': loss.item()}

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}

class RDM(ERM):
    """RDM - Domain Generalization via Risk Distribution Matching (https://arxiv.org/abs/2310.18598) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RDM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy
    
    @staticmethod
    def _moment_penalty(p_mean, q_mean, p_var, q_var):
        return (p_mean - q_mean) ** 2 + (p_var - q_var) ** 2
    
    @staticmethod
    def _kl_penalty(p_mean, q_mean, p_var, q_var):
        return 0.5 * torch.log(q_var/p_var)+ ((p_var)+(p_mean-q_mean)**2)/(2*q_var) - 0.5
    
    def _js_penalty(self, p_mean, q_mean, p_var, q_var):
        m_mean = (p_mean + q_mean) / 2
        m_var = (p_var + q_var) / 4
        
        return self._kl_penalty(p_mean, m_mean, p_var, m_var) + self._kl_penalty(q_mean, m_mean, q_var, m_var)
    
    def update(self, minibatches, unlabeled=None, held_out_minibatches=None):
        matching_penalty_weight = (self.hparams['rdm_lambda'] if self.update_count
                          >= self.hparams['rdm_penalty_anneal_iters'] else
                          0.)

        variance_penalty_weight = (self.hparams['variance_weight'] if self.update_count
                          >= self.hparams['rdm_penalty_anneal_iters'] else
                          0.)

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        losses = torch.zeros(len(minibatches)).cuda()
        all_logits_idx = 0
        all_confs_envs = None

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)
            
            nll = F.cross_entropy(logits, y, reduction = "none").unsqueeze(0)
        
            if all_confs_envs is None:
                all_confs_envs = nll
            else:
                all_confs_envs = torch.cat([all_confs_envs, nll], dim = 0)
                
        erm_loss = losses.mean()
        
        ## squeeze the risks
        all_confs_envs = torch.squeeze(all_confs_envs)
        
        ## find the worst domain
        worst_env_idx = torch.argmax(torch.clone(losses))
        all_confs_worst_env = all_confs_envs[worst_env_idx]

        ## flatten the risk
        all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)
    
        matching_penalty = self.mmd(all_confs_worst_env_flat.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1)) 
        
        ## variance penalty
        variance_penalty = torch.var(all_confs_all_envs_flat)
        variance_penalty += torch.var(all_confs_worst_env_flat)
        
        total_loss = erm_loss + matching_penalty_weight * matching_penalty + variance_penalty_weight * variance_penalty
            
        if self.update_count == self.hparams['rdm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["rdm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.update_count += 1

        return {'update_count': self.update_count.item(), 'total_loss': total_loss.item(), 'erm_loss': erm_loss.item(), 'matching_penalty': matching_penalty.item(), 'variance_penalty': variance_penalty.item(), 'rdm_lambda' : self.hparams['rdm_lambda']}

class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in â€œSupervised Contrastive Learningâ€œ.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64))
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def risk(self, x, y):
        return F.cross_entropy(self.network(x), y).reshape(1)

    def update(self, minibatches, unlabeled=None):
        env_risks = torch.cat([self.risk(x, y) for x, y in minibatches])

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams['eqrm_burnin_iters']:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'loss': loss.item()}


class ADRMX(Algorithm):
    '''ADRMX: Additive Disentanglement of Domain Features with Remix Loss from (https://arxiv.org/abs/2308.06624)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ADRMX, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mix_num = 1
        self.scl_int = SupConLossLambda(lamda=0.5)
        self.scl_final = SupConLossLambda(lamda=0.5)

        self.featurizer_label = networks.Featurizer(input_shape, self.hparams)
        self.featurizer_domain = networks.Featurizer(input_shape, self.hparams)

        self.discriminator = networks.MLP(self.featurizer_domain.n_outputs,
            num_domains, self.hparams)

        self.classifier_label_1 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_label_2 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_domain = networks.Classifier(
            self.featurizer_domain.n_outputs,
            num_domains,
            is_nonlinear=True)


        self.network = nn.Sequential(self.featurizer_label, self.classifier_label_1)

        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))

        self.opt = torch.optim.Adam(
            (list(self.featurizer_label.parameters()) +
             list(self.featurizer_domain.parameters()) +
             list(self.classifier_label_1.parameters()) +
                list(self.classifier_label_2.parameters()) +
                list(self.classifier_domain.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))
                                                    
    def update(self, minibatches, unlabeled=None):

        self.update_count += 1
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        feat_label = self.featurizer_label(all_x)
        feat_domain = self.featurizer_domain(all_x)
        feat_combined = feat_label - feat_domain

        # get domain labels
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=all_x.device)
            for i, (x, _) in enumerate(minibatches)
        ])
        # predict domain feats from disentangled features
        disc_out = self.discriminator(feat_combined) 
        disc_loss = F.cross_entropy(disc_out, disc_labels) # discriminative loss for final labels (ascend/descend)

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        # alternating losses
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            # in discriminator turn
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'loss_disc': disc_loss.item()}
        else:
            # in generator turn

            # calculate CE from x_domain
            domain_preds = self.classifier_domain(feat_domain)
            classifier_loss_domain = F.cross_entropy(domain_preds, disc_labels) # domain clf loss
            classifier_remixed_loss = 0

            # calculate CE and contrastive loss from x_label
            int_preds = self.classifier_label_1(feat_label)
            classifier_loss_int = F.cross_entropy(int_preds, all_y) # intermediate CE Loss
            cnt_loss_int = self.scl_int(feat_label, all_y, disc_labels)

            # calculate CE and contrastive loss from x_dinv
            final_preds = self.classifier_label_2(feat_combined)
            classifier_loss_final = F.cross_entropy(final_preds, all_y) # final CE Loss
            cnt_loss_final = self.scl_final(feat_combined, all_y, disc_labels)

            # remix strategy
            for i in range(self.num_classes):
                indices = torch.where(all_y == i)[0]
                for _ in range(self.mix_num):
                    # get two instances from same class with different domains
                    perm = torch.randperm(indices.numel())
                    if len(perm) < 2:
                        continue
                    idx1, idx2 = perm[:2]
                    # remix
                    remixed_feat = feat_combined[idx1] + feat_domain[idx2]
                    # make prediction
                    pred = self.classifier_label_1(remixed_feat.view(1,-1))
                    # accumulate the loss
                    classifier_remixed_loss += F.cross_entropy(pred.view(1, -1), all_y[idx1].view(-1))
            # normalize
            classifier_remixed_loss /= (self.num_classes * self.mix_num)

            # generator loss negates the discrimination loss (negative update)
            gen_loss = (classifier_loss_int +
                        classifier_loss_final +
                        self.hparams["dclf_lambda"] * classifier_loss_domain +
                        self.hparams["rmxd_lambda"] * classifier_remixed_loss +
                        self.hparams['cnt_lambda'] * (cnt_loss_int + cnt_loss_final) + 
                        (self.hparams['disc_lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.opt.zero_grad()
            gen_loss.backward()
            self.opt.step()

            return {'loss_total': gen_loss.item(), 
                'loss_cnt_int': cnt_loss_int.item(),
                'loss_cnt_final': cnt_loss_final.item(),
                'loss_clf_int': classifier_loss_int.item(), 
                'loss_clf_fin': classifier_loss_final.item(), 
                'loss_dmn': classifier_loss_domain.item(), 
                'loss_disc': disc_loss.item(),
                'loss_remixed': classifier_remixed_loss.item(),
                }
    
    def predict(self, x):
        return self.network(x)

class GradNormLossBalancer:
    def __init__(self, model, initial_weights, alpha=1.2, device='cpu', smoothing=False, tau=None):
        """
        Args:
            model (nn.Module): The model (e.g., ResNet18).
            initial_weights (dict): Initial task weights, e.g., {'fsd': 1.0, 'ssd': 1.0, 'nll': 1.0}
            alpha (float): Moving average smoothing factor for task loss rates.
        """
        self.model = model
        self.task_weights = {
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, requires_grad=True, device=device,))
            for k, v in initial_weights.items()
        }
        self.task_names = list(initial_weights.keys())
        self.alpha = alpha
        self.initial_losses = {}
        self.running_loss_rates = {k: 1.0 for k in self.task_names}  # Initialized to 1.0
        self.smoothing = smoothing
        self.device = device
        if tau is None:
            tau = [1.0 for k in self.task_names]
        else:
            mtau = sum([v for v in tau.values()]) / len(self.task_names)
            tau = [v / mtau for v in tau.values()] 
        self.tau = torch.tensor(tau, device=device, requires_grad=False)

    def reset_weights(self, new_initial_weights):
        for k, new_val in new_initial_weights.items():
            if k not in self.task_weights:
                raise ValueError(f"Task '{k}' not found in existing task_weights.")
            with torch.no_grad():
                self.task_weights[k].copy_(torch.tensor(new_val, device=self.device))

        # Optionally reset other internal state
        self.initial_losses = {}
        self.running_loss_rates = {k: 1.0 for k in self.task_names}

    def parameters(self):
        # So you can pass these to the optimizer
        return list(self.task_weights.values())

    def _get_shared_params(self):
        # Use only the final fully connected layer of ResNet18
        return list(self.model.classifier.parameters())

    def compute_weights_and_loss(self, losses_dict):
        """
        Args:
            losses_dict (dict): Mapping from task name to loss tensor.
        
        Returns:
            dict: weight for each loss.
            torch.Tensor: gradnorm_loss
        """
        task_losses = [losses_dict[k] for k in self.task_names]
        shared_params = self._get_shared_params()

        # Step 1: Store initial losses if not done
        for i, name in enumerate(self.task_names):
            if name not in self.initial_losses:
                self.initial_losses[name] = task_losses[i].detach()

        # Step 2: Compute gradient norms of each task loss
        grads = []
        for loss in task_losses:
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            grad_norm = 0.0
            for p in shared_params:
                if p.grad is not None:
                    grad_norm += (p.grad.norm(2)) ** 2
            grads.append(grad_norm.sqrt())

        grads = torch.stack(grads)
        weights = torch.stack([self.task_weights[k] for k in self.task_names])
        weighted_grads = grads * weights
        avg_grad = weighted_grads.mean()

        # Step 3: Compute inverse training rates
        loss_ratios = torch.stack([losses_dict[k] / self.initial_losses[k] for k in self.task_names])

        normalized_ratios = loss_ratios / loss_ratios.mean().detach()
        loss_rates = normalized_ratios / self.tau
        
        if not self.smoothing:        
            loss_rates = loss_rates ** self.alpha
            smoothed_rates = loss_rates
        else:
            # Step 4: Update running rates (smoothing)
            for i, k in enumerate(self.task_names):
                self.running_loss_rates[k] = (
                    self.alpha * self.running_loss_rates[k] + (1 - self.alpha) * loss_rates[i].item()
                )
            smoothed_rates = torch.tensor([self.running_loss_rates[k] for k in self.task_names], device=grads.device)

        # Step 5: GradNorm loss
        gradnorm_loss = (weighted_grads - avg_grad * smoothed_rates).abs().sum()
        #gradnorm_loss = ((weighted_grads - avg_grad * smoothed_rates) ** 2).sum()

        # Step 6: Normalize task weights
        
        raw_weights = F.softplus(torch.stack([v for v in self.task_weights.values()]))
        weights_sum = raw_weights.sum()
        normed_weights = len(self.task_names) * raw_weights / weights_sum
        normalized_weights = {k: normed_weights[i].detach().to(self.device) \
                for i, k in enumerate(self.task_names)
        }
        #print(raw_weights, normalized_weights)
        return normalized_weights, gradnorm_loss, grads

    def state_dict(self):
        return {
            "task_weights": {k: v.detach() for k, v in self.task_weights.items()},
            "initial_losses": {k: v for k, v in self.initial_losses.items()},
            "running_loss_rates": self.running_loss_rates,
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict["task_weights"].items():
            self.task_weights[k] = torch.nn.Parameter(v.clone().requires_grad_())
        self.initial_losses = {
            k: v.clone() for k, v in state_dict["initial_losses"].items()
        }
        self.running_loss_rates = state_dict["running_loss_rates"]

class LossBalancer:
    def __init__(self, losses, alpha=0.99):
        """
        Args:
            losses (list of losses): Names of the losses to track w/ inital values or None
            alpha (float): Smoothing factor for exponential moving average
        """
        self.alpha = alpha
        self.running_avgs = {name: val for name, val in losses}

    def update(self, loss_dict):
        """
        Args:
            loss_dict (dict of str -> torch.Tensor): Dictionary mapping loss names to loss values.
        
        Returns:
            dict of str -> torch.Tensor: Normalized losses.
        """
        normalized = {}
        for name, loss in loss_dict.items():
            val = abs(loss.detach().item())
            if self.running_avgs[name] is None:
                self.running_avgs[name] = val
            else:
                self.running_avgs[name] = self.alpha * self.running_avgs[name] + (1 - self.alpha) * val

            normalized[name] = loss / (self.running_avgs[name] + 1e-8)

        return normalized

    def weighted_sum(self, norm_losses, weights):
        """
        Compute a weighted sum of normalized losses.

        Args:
            norm_losses (dict of str -> torch.Tensor): Normalized losses.
            weights (dict of str -> float): Weights for each loss.

        Returns:
            torch.Tensor: Weighted sum of normalized losses.
        """
        total = 0.0
        for name, weight in weights.items():
            total += weight * norm_losses[name]
        return total

class BatchedCircularBuffer:
    def __init__(self, capacity, shape, dtype=torch.float32, device='cpu'):
        self.capacity = capacity
        self.shape = shape
        self.device = device
        self.buffer = torch.empty((capacity, *shape), dtype=dtype, device=device)
        self.index = 0
        self.size = 0

    def append(self, batch):
        batch_size = batch.size(0)
        if batch_size > self.capacity:
            batch = batch[-self.capacity:]  # keep only last capacity elements
            batch_size = self.capacity

        end_index = self.index + batch_size
        if end_index <= self.capacity:
            self.buffer[self.index:end_index] = batch
        else:
            first_part = self.capacity - self.index
            self.buffer[self.index:] = batch[:first_part]
            self.buffer[:end_index % self.capacity] = batch[first_part:]

        self.index = (self.index + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def get(self, count=None):
        if self.size == 0:
            return torch.empty((0, *self.shape), dtype=self.buffer.dtype, device=self.device)

        count = self.size if count is None else min(count, self.size)
        start = (self.index - self.size + self.capacity) % self.capacity

        if start + count <= self.capacity:
            return self.buffer[start:start + count]
        else:
            first = self.capacity - start
            return torch.cat([self.buffer[start:], self.buffer[:count - first]], dim=0)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Buffer is empty; cannot sample.")

        count = min(batch_size, self.size)
        indices = torch.randint(0, self.size, (count,), device=self.device)
        full = self.get()  # guaranteed to be contiguous
        return full[indices]

    def __len__(self):
        return self.size

class DictCircularBuffer:
    def __init__(self, capacity, spec: dict, device='cpu'):
        """
        Args:
            capacity: number of entries to hold
            spec: dict mapping keys to (shape, dtype) tuples
        """
        self.capacity = capacity
        self.device = device
        self.buffers = {
            key: torch.empty((capacity, *shape), dtype=dtype, device=device)
            for key, (shape, dtype) in spec.items()
        }
        self.index = 0
        self.size = 0

    def append(self, batch: dict):
        batch_size = next(iter(batch.values())).shape[0]
        if batch_size > self.capacity:
            batch = {k: v[-self.capacity:] for k, v in batch.items()}
            batch_size = self.capacity

        end = self.index + batch_size
        for key, buf in self.buffers.items():
            data = batch[key]
            if end <= self.capacity:
                buf[self.index:end] = data
            else:
                first = self.capacity - self.index
                buf[self.index:] = data[:first]
                buf[:end % self.capacity] = data[first:]

        self.index = (self.index + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        count = min(batch_size, self.size)
        indices = torch.randint(0, self.size, (count,), device=self.device)
        return {k: self.get_all()[k][indices] for k in self.buffers}

    def get_all(self):
        result = {}
        start = (self.index - self.size + self.capacity) % self.capacity
        for key, buf in self.buffers.items():
            if start + self.size <= self.capacity:
                result[key] = buf[start:start + self.size]
            else:
                first = self.capacity - start
                result[key] = torch.cat([buf[start:], buf[:self.size - first]], dim=0)
        return result

    def __len__(self):
        return self.size
        
class CombinedOptimizer:
    def __init__(
        self,
        model_params,
        gradnorm_params,
        base_optimizer_cls=torch.optim.Adam,
        *args,
        model_args=None,
        gradnorm_args=None,
        model_kwargs=None,
        gradnorm_kwargs=None,
        **kwargs
    ):
        """
        Args:
            model_params: Parameters of the model (e.g., model.parameters()).
            gradnorm_params: Parameters of GradNorm (e.g., gradnorm_weights.values()).
            base_optimizer_cls: torch optimizer class (default: Adam).
            *args: Positional args shared by both optimizers (e.g., lr).
            model_args: Positional args specifically for the model optimizer.
            gradnorm_args: Positional args specifically for the GradNorm optimizer.
            model_kwargs: Keyword args specifically for model optimizer.
            gradnorm_kwargs: Keyword args specifically for GradNorm optimizer.
            **kwargs: Shared keyword args (if model_kwargs/gradnorm_kwargs not provided).
        """
        model_args = model_args or args
        gradnorm_args = gradnorm_args or args

        model_kwargs = model_kwargs or kwargs
        gradnorm_kwargs = gradnorm_kwargs or kwargs

        self.model_optimizer = base_optimizer_cls(model_params, *model_args, **model_kwargs)
        self.gradnorm_optimizer = base_optimizer_cls(gradnorm_params, *gradnorm_args, **gradnorm_kwargs)

    def backward(self, **kwargs):
        pass

    def zero_grad(self):
        self.model_optimizer.zero_grad()
        self.gradnorm_optimizer.zero_grad()

    def step(self):
        self.model_optimizer.step()
        self.gradnorm_optimizer.step()

    def state_dict(self):
        return {
            'model_optimizer': self.model_optimizer.state_dict(),
            'gradnorm_optimizer': self.gradnorm_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.model_optimizer.load_state_dict(state_dict['model_optimizer'])
        self.gradnorm_optimizer.load_state_dict(state_dict['gradnorm_optimizer'])
        
    def move_to_device(self, device):
        def move_this_optimizer_to_device(optimizer, device):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
       
        move_this_optimizer_to_device(self.model_optimizer, device)
        move_this_optimizer_to_device(self.gradnorm_optimizer, device)

class GLSD(ERM):
    """GLSD algorithm """

    def init_optimizer(self):
        if self.hparams["glsd_optimizer"] == "sgd":
            base_optimizer_cls=torch.optim.SGD
            extra_pars = {"momentun": 0.9}
        elif self.hparams["glsd_optimizer"] == "adam":
            base_optimizer_cls=torch.optim.Adam
            extra_pars = {}
        elif self.hparams["glsd_optimizer"] == "adamw":
            base_optimizer_cls=torch.optim.AdamW
            extra_pars = {}
            
        model_kwargs = {"lr": 1.0*self.hparams["lr"], "weight_decay": self.hparams['weight_decay'], **extra_pars}        
        gradnorm_kwargs = {"lr": 10.0*self.hparams["lr"], "weight_decay": self.hparams['weight_decay'], **extra_pars}        
        
        return CombinedOptimizer(self.network.parameters(), self.gradnorm_balancer.parameters(), 
                base_optimizer_cls=base_optimizer_cls, model_kwargs=model_kwargs, gradnorm_kwargs=gradnorm_kwargs)
    

    def __init__(self, SSD, input_shape, num_classes, num_domains, hparams):
        super(GLSD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        
        self.SSD = SSD

        self.hparams = hparams
        capacity = 5*num_domains*hparams['batch_size']
        shape_eta = ()
        shape_envs = ()
        """
        rb = BatchedCircularBuffer(capacity, shape, device=device)
        """
        spec = {"sorted_eta": (shape_eta, torch.float32), "envs": (shape_envs, torch.int)}
            
        rb = DictCircularBuffer(capacity, spec, device=device)
        self.buffer = rb
        self.register_buffer('update_count', torch.tensor([0]))
        pi_init = torch.rand(2,num_domains)
        pi_init /= pi_init.sum(1, keepdim=True)
        self.register_buffer('pi', pi_init)
        pi_init = torch.rand(2,num_domains)
        pi_init /= pi_init.sum(1, keepdim=True)
        self.register_buffer('pi_prev', pi_init)
        self.register_buffer('margin', torch.tensor([0.2]))
        initial_weights = {"cls": 1.0, "penalty": 1.0, }
        losses_to_balance = [("cls",None), ("penalty",None)]
        tau = None
        #tau = {"nll": hparams["glsd_nll_lambda"], "penalty": 1.0, } # smaller tau = faster learning
        
        self.loss_balancer = LossBalancer(losses_to_balance, alpha=hparams["glsd_lossbalancer_alpha"])
        self.gradnorm_balancer = GradNormLossBalancer(self, initial_weights=initial_weights, 
                alpha=hparams["glsd_gradnorm_alpha"], device=device, smoothing=hparams["glsd_gradnorm_smoothing"], 
                tau=tau)

        self.optimizer = self.init_optimizer()
       
        self.glsd_after_load_state_count = 0

        """
        def GLSD_load_state_post_hook(module, incompatible_keys):
            module.glsd_after_load_state_count = module.hparams["glsd_after_load_state_count"]

        self.register_load_state_dict_post_hook(GLSD_load_state_post_hook)
        """

    """
    module = self
    register_load_state_dict_pre_hook(hook): Register a pre-hook to be run before module's load_state_dict() is called.
        hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None 
    register_load_state_dict_post_hook(hook): Register a post-hook to be run after module's load_state_dict() is called.
        hook(module, incompatible_keys) -> None
        The given incompatible_keys can be modified inplace if needed.
    register_state_dict_post_hook(hook): Register a post-hook for the state_dict() method.
        hook(module, state_dict, prefix, local_metadata) -> None
        The registered hooks can modify the state_dict inplace.
    register_state_dict_pre_hook(hook): Register a pre-hook for the state_dict() method.
        hook(module, prefix, keep_vars) -> None
    """

    def get_extra_state(self):
        # Return any extra state to include in the module's state_dict.
        # Dumps the replay buffer and returns the state_dict to add to module's state_dict
        # This function is called when building the module's state_dict().
        # state_dict(): Return a dictionary containing references to the whole state of the module
        return {"buffer": self.buffer,
                "loss_balancer": {"alpha": self.loss_balancer.alpha,
                                  "running_avgs": self.loss_balancer.running_avgs, 
                }, 
                "gradnorm_balancer": self.gradnorm_balancer.state_dict(),
       }

    def set_extra_state(self, state):
        # This function is called from load_state_dict()
        # load_state_dict(state_dict): Copy parameters and buffers from state_dict into this module and its descendants.
        self.buffer = state["buffer"]
        self.loss_balancer.alpha = state["loss_balancer"]["alpha"]
        self.loss_balancer.running_avgs = state["loss_balancer"]["running_avgs"]
        self.gradnorm_balancer.load_state_dict(state["gradnorm_balancer"])
    
    def update(self, minibatches, unlabeled=None):
    
        def calculate_Fks(x, lambdas=None, tau=1e-2):
            """
            Calculate F1, F2, and etas for x.
                x: (n,samples) of n distributions, which we want to maximize. (n,b)
                lambdas: (n,weights) or None of per example weights. None means 1/b (n,b)
                tau: temperature parameter for smoothness
            Returns:
                Per environment F1 (n,nb)
                Per environment F2 (n,nb)
                tuple (sorted_eta, envs, lambdas) of eta for which Fk were computed, mapping to which env it came from and its weight 
            """    
            device = x.device
            n,b = x.size()

            x_flat = x.view(n, b)                                   # (n, b)
            sorted_x_flat, sorted_x_idx = torch.sort(x_flat, dim=1) # (n, b)
            # eta: shape (n*b,) = sorted losses (t_k)
            eta = x.reshape(-1)                                     # (nb,)
            sorted_eta, sorted_idx = torch.sort(eta)                # (nb,)
            sorted_eta_all = sorted_eta.unsqueeze(0).unsqueeze(2)   # (1, nb, 1)
            sorted_x_i = sorted_x_flat.unsqueeze(1)                 # (n, 1, b)
            envs = (torch.ones_like(x) * (torch.arange(0,n,device=device).unsqueeze(1))).reshape(-1) # (nb,)
            envs = envs[sorted_idx] # tells which environment each eta came from

            # Compute sigmoid((t_k - x_ij)/tau) for all i, k, j
            # For each domain, for each eta for each example in batch give 1 if x<eta and 0 otherwise
            sigmoid_matrix = torch.sigmoid((sorted_eta_all - sorted_x_i) / tau) # (n, nb, b)

            # Sum over b (within env), average to get soft-CDF
            if lambdas is None:
                F1_soft = sigmoid_matrix.sum(dim=2) / b  # shape (n, nb)
                lambdas_sorted = torch.ones_like(x) / b
            else:
                lambdas_sorted = torch.gather(lambdas, dim=1, index=sorted_x_idx)
                #             (n,nb,b)                (n,1,b)
                F1_soft = (sigmoid_matrix * (lambdas_sorted.unsqueeze(1))).sum(dim=2)  # shape (n, nb)
            
            lambdas_sorted = lambdas_sorted.reshape(-1) # (nb,)
            lambdas_sorted_all = lambdas_sorted[sorted_idx]
                
            F1 = F1_soft

            # Calculate eta_i - eta_{i-1}
            h = sorted_eta - torch.roll(sorted_eta,1) #torch.roll is circular shift rigt one place
            F2_incre = h*torch.roll(F1,1,dims=1) # (eta_i - eta_{i-1})*F_1(X; eta_{i-1}); roll shifts right by one place to align F_1 and h
            F2_incre = F2_incre.clone() # n x n*b
            F2_incre[:,0] = 0 # zero out the 1st index in-place
            # Calculate F2 for all etas
            F2 = torch.cumsum(F2_incre,1)

            return (sorted_eta, envs, lambdas_sorted_all), F1, F2

        def extreme_affine_combination(x, dominating, order, tau=1.0):
            """
            First-order dominating cdf.
                x: (n,samples) of n distributions, which we want to maximize.
                dominating: True/False
                order: Fk, k=1/2
            Returns:
                pi: a probability vector (n,) which gives for each domain i 
                    the probability of being assigned the positive lambda of an affine combination
                    (lambdas = (pi * lambda_pos + (1 - pi) * lambda_min)
                sorted_eta: eta for which Fk were computed 
                env: env each eta came from
            """    
            
            n,b = x.size()
            (sorted_eta, envs, _), F1, F2 = calculate_Fks(x)
            if order == 1:
                Fk = F1
            else: 
                Fk = F2 
            diffs = Fk.unsqueeze(1) - Fk.unsqueeze(0) # shape: [n, n, b]
            if dominating:
                clamp = {"max": 0}
            else:
                clamp = {"min": 0}
            
            # i is dominating j if T[i,k] <= T[j,k] for all k and there's some m s.t. T[i,m] < T[j,m]
            # i is dominated by j if T[i,k] >= T[j,k] for all k and there's some m s.t. T[i,m] > T[j,m]

            # Find i such that T[i] <= T[j] for all j != i and some T[i,k] < T[j,k]            
            diffs = torch.clamp(diffs, **clamp) # leave only etas where i is dominating/dominated
            scores = torch.sum(diffs, (1,2)) # sum all scores over other environments
            # Softmax over dominating scores to get positive weights sum to 1
            pi = torch.softmax(scores / tau, dim=0)  # tau = temperature > 0                     
            return pi, sorted_eta, envs

        def fill_list(x):
            # Identify nonzero elements
            nonzero_mask = x != 0
            # Create cumulative sum of nonzero indicators
            group_ids = torch.cumsum(nonzero_mask.to(torch.int), dim=0) - 1

            # Identify which elements should be replaced (only after the first nonzero)
            valid_mask = group_ids >= 0
            # Store the nonzero values
            nonzero_values = x[nonzero_mask]

            # Prepare output by copying input
            filled = x.clone()
            # Replace only valid (non-leading) zeros
            filled[valid_mask] = nonzero_values[group_ids[valid_mask]]
            return filled

        def interp1_linear_torch(x, xp, fp):
            """
            Differentiable 1D linear interpolation in PyTorch (like MATLAB interp1, linear case).

            Args:
                x (torch.Tensor): target x values
                xp (torch.Tensor): known x values (1D, must be sorted ascending)
                fp (torch.Tensor): known y values (1D, same length as xp)

            Returns:
                torch.Tensor: interpolated y values at x
            """
            # Find indices for the left point
            idx = torch.searchsorted(xp, x, right=False).clamp(1, len(xp) - 1)
            idx0 = idx - 1
            idx1 = idx

            x0 = xp[idx0]
            x1 = xp[idx1]
            y0 = fp[idx0]
            y1 = fp[idx1]

            # Linear interpolation formula
            slope = (y1 - y0) / (x1 - x0)
            y = y0 + slope * (x - x0)

            return y

        
        def calc_F1_loss(sorted_eta, seta_x, F1x, F1y, rel_tau=0.3, get_utility=False, beta=20, margin=0.01):
            eps = torch.finfo(F1x.dtype).eps
            eta_values = sorted_eta + eps # [2b,]
            eta_values = eta_values.detach() # From Shicong's implementation: here eta are treated as constants not as coming from model

            delta = F1x - F1y
            tau = (torch.max(delta) - torch.min(delta))*rel_tau
            mu = torch.exp((delta - torch.max(delta))/tau)
            mu = mu/(torch.sum(mu)+eps)
            mu = mu.detach()

            eta = eta_values.unsqueeze(1) # [2b,1]

            # Previous code (Dai 2023) suggests relu
            if get_utility:
                #                         (2b,1)      (1,2b)
                ux = torch.sum(F.softplus(eta     - (seta_x.unsqueeze(0) - margin), beta=beta)*(mu.unsqueeze(1)), dim=0)
                #ux = torch.sum(F.relu(eta - (seta_x.unsqueeze(0)))*(mu.unsqueeze(1)), dim=0)
                return ux
            else:
                #                          (2b,1)     (1,2b)
                ex = torch.mean(F.softplus(eta    - seta_x.unsqueeze(0) - margin, beta=beta), dim=1)
                #ex = torch.mean(F.relu(eta - seta_x.unsqueeze(0)), dim=1)
                #loss =(ex*mu).clamp(min=0).sum()
                loss =(ex*mu).sum()
                return loss
       
        def xsd_1st_cdf(seta_x, seta_y, lambda_x, lambda_y, rel_tau=0.3, get_utility=False):
            """First-order stochastic dominance loss.

            Args:
                seta_ (n,) : sorted eta (utilities) (correspondig to Fk) from two distributions, which we want to maximize.
                      x - is the new samples, y - is the reference
                      x - X_{\theta_{t,\bar{t}}, y - X_{\theta_t}
                lambda_ (n,) : per-example weight
                rel_tau: Softmax temperature control
                get_utility: Return array u(x) instead of a scalar loss
                Shicong commented that when working with sampling dependent on theta (Algorithm 3),
                you can pass get_utility=True to get ux values and plug it into the REINFORCE algorithm in place of cumulative rewards.

            Returns:
                Loss value to minimize, or the utility function u(x)
                combined sorted_eta, sorted_lambda
            """    
            
            nX, nY = len(seta_x), len(seta_y)
            xy = torch.vstack((seta_x,seta_y)) # assumes both are same length
            lambda_xy = torch.vstack((lambda_x,lambda_y)) # assumes both are same length
            (sorted_eta, envs, sorted_lambda), F1, _ = calculate_Fks(xy, lambda_xy) # (2b,), (2,2b), (2,2b)
            
            F1x = F1[0].squeeze() # (2b,)
            F1y = F1[1].squeeze() # (2b,)
            
            ret_val = calc_F1_loss(sorted_eta, seta_x, F1x, F1y, rel_tau=rel_tau, get_utility=get_utility)
            return ret_val, sorted_eta, envs
            
        def xsd_2nd_cdf(seta_x, seta_y, lambda_x, lambda_y, rel_tau=0.3, get_utility=False, margin=0.0, get_F1=False):
            """Second-order stochastic dominance loss. Implements algorithm 2

            Args:
                sorted_ (n,) : sorted eta (correspondig to Fk) from two distributions, which we want to maximize.
                      x - is the new samples, y - is the reference
                      x - X_{\theta_{t,\bar{t}}, y - X_{\theta_t}
                lambda_ (n,) : per-example weight
                rel_tau: Softmax temperature control
                get_utility: Return array u(x) instead of a scalar loss
                Shicong commented that when working with sampling dependent on theta (Algorithm 3),
                you can pass get_utility=True to get ux values and plug it into the REINFORCE algorithm in place of cumulative rewards.

            Returns:
                Loss value to minimize, or the utility function u(x)
                sorted_eta
                envs
            """    
            
            device = seta_x.device
            nX, nY = len(seta_x), len(seta_y)
            xy = torch.vstack((seta_x,seta_y)) # assumes both are same length
            lambda_xy = torch.vstack((lambda_x,lambda_y)) # assumes both are same length
            (sorted_eta, sorted_is_y, sorted_lambda), F1, F2 = calculate_Fks(xy, lambda_xy) # (2b,), (2,nb), (2,nb)
            
            F1x = F1[0].squeeze() # (2b,)
            F1y = F1[1].squeeze() # (2b,)
            F2x = F2[0].squeeze() # (2b,)
            F2y = F2[1].squeeze() # (2b,)
                      
            """
            assuming that x_0 < x_1 < ... < x_n is sorted, for any x_i the corresponding F2x entry 
            before the gradient correction line is computed as F_2(x_i) = \frac{1}{n} \sum_{j=0}^{i-1} (x_i - x_j). 
            (Practically, the division by 1/n is omitted.)
            After this line we cancel out the gradient of x_i, and essentially convert it to 
            F_2(stop_grad(x_i)) = \frac{1}{n} \sum_{j=0}^{i-1} (stop_grad(x_i) - x_j). 
            The reason is that x_i is to be picked up by an argmax/softmax operator mu, 
            and per Danskin's theorem it should be detached from gradient computation.
            softmax is a smooth approximator of argmax
            """
            # correct gradient computation
            F2x = F2x + (1-sorted_is_y)*F1x*(sorted_eta.detach()-sorted_eta)
            F2y = F2y.detach()

            eps = torch.finfo(x.dtype).eps

            delta = F2x - F2y
            # mu = argmax(delta)
            tau = (torch.max(delta) - torch.min(delta))*rel_tau
            mu = torch.exp((delta - torch.max(delta))/tau)
            mu = mu/(torch.sum(mu)+eps)
            mu = mu.detach()

            """
            Calculates the utility in lines 6,7 or returns sum(F2x-F2y)*mu
            which applies when sampling of x_i is independent of theta.
            Note that eventhough we search for the worst input distribution dependent on theta,
            we do not sample from the input distribution - only the weights are adjusted, so
            this trick still applies.
            """
            if get_utility:
                u1 = torch.cumsum(mu[::-1])[::-1]
                u2_incre = (torch.roll(sorted_eta,-1) - sorted_eta)*torch.roll(u1,-1)
                u2_incre[0] = 0
                u2 = torch.cumsum(u2_incre[::-1])[::-1]
                ux = u2[torch.argsort(idx_sort)[:nX]]
                ret_val_F2 = ux
            else:
                # Create a loss function (of theta) in such a way that it can be differentiated to obtain the gradients
                # w.r.t. theta to improve theta. This is done by using Dankin's theorem.
                # loss = delta*mu, i.e. delta[argmax(delta)]
                #loss = (delta*mu + margin).clamp(min=0).sum()
                loss = (delta*mu + margin).sum()
                ret_val_F2 = loss  
            
            if get_F1:
                ret_val_F1 = calc_F1_loss(sorted_eta, seta_x, F1x, F1y, rel_tau=rel_tau, get_utility=get_utility)
            else:
                ret_val_F1 = torch.zeros_like(ret_val_F2)
            
            return ret_val_F2, ret_val_F1, sorted_eta, sorted_is_y

        def generate_samples_from_affine_hull(K, n, lambda_min, device):
            """Generates samples from semi-bounded affine hull
            Args:
                K: Number of sets to generate
                n: Size of each sample (number of domains)
                lambda_min: minimal lambda, can (and normally will) be negative
                device: device to put the result on
            Returns:
                A tensor (n,K) of affine coefficients
            """   
            if K > 0:
                lambda_max = 1 - (n - 1) * lambda_min
                Lambdas = torch.rand(n-1,K,device=device)
                Lambdas = lambda_min + (lambda_max - lambda_min)*Lambdas # move to [a,b]
                last_row = 1 - Lambdas.sum(dim=0)  # shape: (K,)
                Lambdas = torch.cat([Lambdas, last_row.unsqueeze(0)], dim=0)  # shape: (n, K)
                # Lambdas: shape (n, K)
                perms = torch.stack([torch.randperm(n) for _ in range(K)], dim=1).to(device)  # shape (n, K)
                # Use gather to permute each column independently
                Lambdas = torch.gather(Lambdas, 0, perms)
            else:
                Lambdas = torch.empty(n,0,device=device,dtype=torch.float)
            return Lambdas

        def get_total_grad_norm(model):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                return 0.0
            return torch.norm(torch.stack([g.norm() for g in grads])).item()
            
        def make_extreme_lambda(self, pi, worst, lambda_min):
            """
            Args:
                pi: vector (n,) of probabilities to assign domain i lambda_pos
                worst: 0/1 index of pi
                lambda_min: minimal lambda of the affine combination
            Returns:
                lambda: (n,) affine combination
            """
            n = len(pi)
            update_worst_env_every_steps = self.hparams['glsd_update_worst_env_every_steps']
            ministep = self.update_count.item() % update_worst_env_every_steps
            if ministep == 0:
                self.pi_prev[worst] = self.pi[worst]
                self.pi[worst] = pi                  

            alpha_max = update_worst_env_every_steps / self.hparams['glsd_lambda_alpha_div']
            alpha = min(ministep/alpha_max,1)
            pi = alpha*self.pi[worst] + (1-alpha)*self.pi_prev[worst]

            pi = pi.detach() # (n,)

            lambda_pos = 1 - (n - 1) * lambda_min
            # (n,)          (n,)
            lambda_val = pi * lambda_pos + (1 - pi) * lambda_min

            return lambda_val

        def E(x, weights, keepdim=False):
            """ Calculates expectation of sample with signed distribution
                x: samples, (n,b)
                weights: probabilities, (n,b)
            """
            e = (x * weights).sum(1, keepdim=keepdim)
            return e

        def u(x, weights, utype, **kwargs):
            if utype==-1:
                return x
            if utype==0:
                return -((x - E(x,weights,keepdim=True)).square())
            elif utype==1:
                return -(((x - E(x,weights,keepdim=True)).square() + 1e-6).sqrt())
            elif utype==2:
                return torch.log1p(torch.exp(x - kwargs["tau"])) 
            elif utype==3:
                return torch.sigmoid(x - kwargs["tau"])
            elif utype==4:
                return 1 - torch.exp(-x*kwargs["tau"])
            elif utype==5:
                return (x - E(x,weights,keepdim=True)) / (torch.std(x, keepdim=True) + 1e-8)
            elif utype==6:
                mu_i = x.mean(1,keepdim=True)
                sigma_i2 = (x - mu_i).square().mean(1,keepdim=True)
                mu = E(x,weights,keepdim=True)
                sigma_2 = ((sigma_i2 + (mu_i - mu).square()) * weights).sum(1,keepdim=True) # (n,1)
                return sigma_2
            else:
                assert False, f"Unknown u() type {utype}"
    
        def imagine_domains(K, n, lambda_min, device, include_base_domains=False):
            lambdas = generate_samples_from_affine_hull(K, n, lambda_min, device=device) # (n,K-1)

            if include_base_domains:
                perm = torch.randperm(n, device=device)
                one_hot = torch.zeros(n, n, dtype=torch.float32, device=device)
                one_hot[torch.arange(n, device=device), perm] = 1.0
                one_hot.requires_grad_(False) # (n,n)
                # (n,K')             (n,K)    (n,n), K' = K + n
                lambdas = torch.cat([lambdas, one_hot],dim=1)
            return lambdas
                       

        def prepare_lambdas(self, losses, lambda_min, device, dominating=False, dominated=False):
            n = losses.size()[0]
            assert (self.hparams["glsd_K"] > 0 or dominating or dominated), "No lambdas requested!"
            with torch.no_grad():
                # (n,K'-1)
                lambdas = imagine_domains(self.hparams["glsd_K"]-1, n, lambda_min, device, include_base_domains=self.hparams["glsd_dominate_all_domains"])
                if dominated:
                    pi_worst, _, _ = extreme_affine_combination(losses, dominating=False, order=int(self.SSD)+1) # sorted_eta depend on network (nb,)
                    # (n,)          (n,)
                    lambda_worst =  make_extreme_lambda(self, pi_worst, worst=0, lambda_min=lambda_min).unsqueeze(1).to(device)
                    # (n,K')          (n,K'-1)           (n,1)   
                    lambdas = torch.cat([lambdas, lambda_worst], dim=1)
                if dominating:
                    pi_best, _, _  = extreme_affine_combination(losses, dominating=True,  order=int(self.SSD)+1) # sorted_eta depend on network (nb,)
                    # (n,)          (n,)
                    lambda_best  =  make_extreme_lambda(self, pi_best,  worst=1, lambda_min=lambda_min).unsqueeze(1).to(device)
                    lambdas = torch.cat([lambdas, lambda_best], dim=1)
            
            lambdas = lambdas.detach()
            return lambdas

        def soft_upper_clamp(nll, threshold, sharpness=10.0):  # slope ~1
            s = torch.sigmoid(-sharpness * (nll - threshold))
            return s * nll + (1 - s) * threshold
            
        # What are minibatches? Looks like they're minibatch per environment
        penalty_weight = 1.0
        nll = 0.
        n = len(minibatches)
        lambda_min = -self.hparams['glsd_affine_hull_gamma'] / np.sqrt(n)

        # Step 1: Get per-domain losses (nlls)
        all_x = torch.cat([x for x, _ in minibatches])
        all_logits = self.network(all_x) # all_logits depend on network
        all_logits_idx = 0
        device = self.device

        # calculate per environment nll from logits calculated for inputs from all environments
        losses = []
        for x, y in minibatches:
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y, reduction='none') # nll depends on network
            losses.append(nll) # losses depend on network
        losses = torch.stack(losses) # env x b, Concatenates a sequence of tensors along a new dimension.
       
        nll = soft_upper_clamp(losses, self.hparams["glsd_nll_threshold_sample"])
        nll = nll.sum(1).mean().unsqueeze(0) # sum over batch, mean over envs

        u_kwargs = self.hparams["glsd_u_kwargs"]
        if self.hparams["glsd_classifier_loss"] == "glsd":
            """
            In classification we want min_theta max_lambda E[loss].
            For utility-view with u=-loss this gives: 
                min_theta max_lambda E[-u] = min_theta max_lambda -E[u] = min_theta min_lambda E[u] 
            This means we're looking for a dominated environment (one with smallest u)
            """
            lambdas = prepare_lambdas(self, -losses, lambda_min, device, dominating=True, dominated=False)
            K = lambdas.size()[1]
            
            (sorted_eta, envs, _), _, _ = calculate_Fks(-losses)

            if len(self.buffer) == 0:
                data = {"sorted_eta": sorted_eta.detach().to(device), "envs": envs} # assume we're no backproping the error to previous rounds           
                self.buffer.append(data)

            ref = self.buffer.sample(len(sorted_eta))
            envs_ref = ref["envs"]

            if True:
                final_margin = 0
                initial_margin = self.margin
                total_steps = self.hparams["n_steps"]
                margin = initial_margin + (final_margin - initial_margin) * min((self.update_count / total_steps), 1.0)
                self.margin = margin
            else:
                margin = 0.0

            loss_ssd = torch.tensor([0.0],device=device,requires_grad=True,dtype=torch.float)
            loss_fsd = torch.tensor([0.0],device=device,requires_grad=True,dtype=torch.float)
            for i in range(K):
                lambda_i = lambdas[:,i].squeeze() # (n,)
                lambda_ii = torch.tensor([lambda_i[int(e.item())] for e in envs], device=device) / sorted_eta.size()[0] # (nb,)
                lambda_ref = torch.tensor([lambda_i[int(e.item())] for e in envs_ref], device=device) / sorted_eta.size()[0] # (nb,)
                if self.SSD:
                    l_ssd, _, _, _ = xsd_2nd_cdf(sorted_eta, ref["sorted_eta"], \
                                   lambda_ii, lambda_ref, margin=margin, get_F1=False)
                    l_fsd = torch.zeros_like(loss_ssd)
                else:
                    l_fsd, _, _ = xsd_1st_cdf(sorted_eta, ref["sorted_eta"], \
                                   lambda_ii, lambda_ref)
                    l_ssd = torch.zeros_like(loss_fsd)
                loss_ssd = loss_ssd + l_ssd
                loss_fsd = loss_fsd + l_fsd
            loss_ssd = loss_ssd / K
            loss_fsd = loss_fsd / K

            data = {"sorted_eta": sorted_eta.detach(), "envs": envs.detach()} # assume we're no backproping the error to previous rounds
            self.buffer.append(data)

            loss_signs = {"cls": 1.0, }
            loss_names = ["cls"]
            if self.SSD:
                losses_dict = {"cls": loss_ssd, }
            else:
                losses_dict = {"cls": loss_fsd, }
      
        elif self.hparams["glsd_classifier_loss"] == "nll": 
                print(losses.size(),losses)
                losses_dict = {"cls": losses, }
                loss_signs = {"cls": 1.0, }
                loss_names = ["cls"]
                
        else:
            assert False, f'Unknown classifier loss {self.hparams["glsd_classifier_loss"]}'
            
        if self.hparams["glsd_regularizer"] == "imagined_domains":  
            # Here the domains are non-weighted yet
            b = losses.size()[1]
            lambda_ii = torch.ones_like(losses) / b # (n,b)
            losses = u(-losses, lambda_ii, self.hparams["glsd_utype"], **u_kwargs)
            lambdas = prepare_lambdas(self, losses, lambda_min, device, dominating=False, dominated=False)

        elif self.hparams["glsd_regularizer"] == "imagined_domains&bestworst" or self.hparams["glsd_regularizer"] == "VREx":  
            """
            Use Fk of the different domains as regularizer:
            1. Calculate Fk for all domains (including the imagined ones) for all etas and configurations.
            2. Take the squared difference between all domain pairs.
            3. Sum over all pairs, all etas.
            4. Normalize by the number of configurations (K).
            5. Use this as a penalty (i.e., we want all Fks to be the same for all etas)
            """
            # Here the domains are non-weighted yet
            b = losses.size()[1]
            lambda_ii = torch.ones_like(losses) / b # (n,b)
            losses = u(-losses, lambda_ii, self.hparams["glsd_utype"], **u_kwargs)
            lambdas = prepare_lambdas(self, losses, lambda_min, device, dominating=True, dominated=True)
                        
        elif self.hparams["glsd_regularizer"] == "bestworst": 
            """
            Use Fk of the best and worst affine combinations as regularizer:
            1a. Calculate best and worst affine combinations
            1b. Calculate Fk for these combinations.
            2. Take the squared difference between them for each eta.
            3. Sum over both, for  all etas.
            4. Use this as a penalty (i.e., we want all Fks to be the same for all etas)
            """
            # Here the domains are non-weighted yet
            b = losses.size()[1]
            lambda_ii = torch.ones_like(losses) / b # (n,b), requires_grad=False, device=losses.device()
            losses = u(-losses, lambda_ii, self.hparams["glsd_utype"], **u_kwargs)
            lambdas = prepare_lambdas(self, losses, lambda_min, device, dominating=True, dominated=True)

        elif self.hparams["glsd_regularizer"] == "nll" or \
             self.hparams["glsd_regularizer"] == "-nll":
                losses_dict = dict(**losses_dict, penalty=losses)
                reg_sign = 1.0 if self.hparams["glsd_regularizer"]=="nll" else -1.0
                loss_signs = dict(**loss_signs, penalty=reg_sign)
                penalty_names = ["penalty"]
        else:
            assert False, f'Unknown regulaizer {self.hparams["glsd_regularizer"]}'
            
        if self.hparams["glsd_regularizer"] == "imagined_domains" or \
            self.hparams["glsd_regularizer"] == "bestworst" or \
            self.hparams["glsd_regularizer"] == "imagined_domains&bestworst": 
            
            K = lambdas.size()[1] # update number of lambdas
            b = losses.size()[1]
            loss_ssd_list = []
            loss_fsd_list = []
            for i in range(K):
                lambda_i = lambdas[:,i].squeeze() # (n,)
                # Need lambdas: (n,weights)
                # Here the domains are non-weighted yet
                (sorted_eta, envs, _), _, _ = calculate_Fks(losses) # (nb,)
                # Create true affine combination
                lambda_ii = torch.tensor([lambda_i[int(e.item())] for e in envs], device=device) / sorted_eta.size()[0] # (nb,)
                _, l_fsd, l_ssd = calculate_Fks(sorted_eta.unsqueeze(0), lambda_ii.unsqueeze(0)) # (1, nb)
                l_fsd = l_fsd.squeeze() # (nb,)
                l_ssd = l_ssd.squeeze() # (nb,)
                               
                loss_ssd_list.append(l_ssd)
                loss_fsd_list.append(l_fsd)
            
            # Stack along new dim = -1 (nb, K)
            loss_ssd = torch.stack(loss_ssd_list, dim=-1)
            loss_fsd = torch.stack(loss_fsd_list, dim=-1)
            if self.SSD:
                Fk = loss_ssd # (nb,K)
            else:
                Fk = loss_fsd # (nb,K)       
            
            diffs = Fk.unsqueeze(2) - Fk.unsqueeze(1) # shape: [nb, K, K]
            penalty = diffs.abs()
            #penalty = F.softplus(diffs) + F.softplus(-diffs)
            nnz_penalty = (penalty > 0).sum().detach()
            nnz_penalty = nnz_penalty if nnz_penalty > 0 else 1
            penalty = penalty.sum() / nnz_penalty
            
            # Sign for each task
            loss_signs = dict(**loss_signs, penalty=1.0)
            losses_dict = dict(**losses_dict, penalty=penalty.squeeze())
            penalty_names = ["penalty"]
        
        elif self.hparams["glsd_regularizer"] == "VREx":
            
            K = lambdas.size()[1] # update number of lambdas
            b = losses.size()[1]
            loss_means_list = []
            for i in range(K):
                lambda_i = lambdas[:,i].squeeze() # (n,)
                # Need lambdas: (n,weights)
                # Here the domains are non-weighted yet
                (sorted_eta, envs, _), _, _ = calculate_Fks(losses) # (nb,)
                # Create true affine combination
                lambda_ii = torch.tensor([lambda_i[int(e.item())] for e in envs], device=device) / sorted_eta.size()[0] # (nb,)
                (sorted_eta, _, lambdas_sorted_all), _, _ = calculate_Fks(sorted_eta.unsqueeze(0), lambda_ii.unsqueeze(0)) # (1, nb)
                l_mean = E(sorted_eta.unsqueeze(1), lambdas_sorted_all.unsqueeze(1)) # ()
                               
                loss_means_list.append(l_mean)
            
            # Stack along new dim = -1 (K,)
            loss_mean = torch.stack(loss_means_list, dim=-1)
            
            penalty = loss_mean.var()

            # Sign for each task
            loss_signs = dict(**loss_signs, penalty=1.0)
            losses_dict = dict(**losses_dict, penalty=penalty.squeeze())
            penalty_names = ["penalty"]

        """ --------------------------------------------------------------
        Determine weights, run optimizer
           Inputs:  losses_dict
                    loss_signs
                    loss_names
                    penalty_names
        """
        def penalty_weight(t, 
                    penalty_min=self.hparams['glsd_penalty_lambda_min'], 
                    penalty_max=self.hparams['glsd_penalty_lambda_max'], 
                    tau=self.hparams['glsd_penalty_tau'], 
                    penalty_power=self.hparams['glsd_penalty_power']):
            s_power = np.power(t.cpu().item()/tau,penalty_power)
            s_exp = np.exp(-s_power)
            s_1m = 1 - s_exp
            p = np.maximum(s_1m*penalty_max, 0)
            p = penalty_min + p
            #print(t.item(), s_power, s_exp, s_1m, p)
            return p

        if self.update_count > self.hparams["glsd_lossbalancer_warmup"]:
            losses_dict = self.loss_balancer.update(losses_dict)
            pweight = penalty_weight(self.update_count - self.hparams["glsd_lossbalancer_warmup"])
        elif self.update_count == self.hparams["glsd_lossbalancer_warmup"]:
            self.optimizer = self.init_optimizer()
            losses_dict = self.loss_balancer.update(losses_dict)
            pweight = penalty_weight(self.update_count - self.hparams["glsd_lossbalancer_warmup"])
        else:
            pweight = self.hparams['glsd_penalty_lambda_min']

        loss_weights = {k: torch.tensor([1.0], device=device) for k in loss_names}
        penalty_weights = {k: torch.tensor([pweight], device=device) for k in penalty_names}
        loss_weights = dict(**loss_weights, **penalty_weights)
        loss_gradnorm = torch.tensor([0], device=device)
        grads = torch.zeros(len(loss_weights), device=device)

        if self.hparams["glsd_gradnorm_warmup"] is not None and self.update_count >= self.hparams["glsd_gradnorm_warmup"]:
            if self.update_count == self.hparams["glsd_gradnorm_warmup"]:
                new_initial_weights = {k: v.item() for k,v in loss_weights.items() }
                self.gradnorm_balancer.reset_weights(new_initial_weights)

            loss_weights, loss_gradnorm, grads = self.gradnorm_balancer.compute_weights_and_loss(losses_dict)  

        # Combine weights
        signed_weighted_losses = {
            name: loss_signs[name] * loss_weights[name] * losses_dict[name] for name in loss_weights
        }
        # Final total loss
        sloss = sum(signed_weighted_losses.values())
        loss = sloss + self.hparams["glsd_gradnorm_lambda"] * loss_gradnorm

        # Do the real backward pass on the total loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if False and (self.update_count % 100 == 0):
            print(self.update_count.item(), ":", get_total_grad_norm(self.network), get_total_grad_norm(self.gradnorm_balancer), 
                loss_gradnorm.item(), nll.item(), penalty.item(), loss.item(), grads.tolist())

        self.optimizer.step()

        self.update_count += 1

        scalar_losses = {k: v.item() for k, v in losses_dict.items()}
        scalar_loss_weights = {'w_'+k: v.item() for k, v in loss_weights.items()}
        return {'loss': loss.item(), **scalar_losses, 'loss_gradnorm': loss_gradnorm.item(), **scalar_loss_weights, }      

class GLSD_SSD(GLSD):
    """GLSD_SSD algorithm """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GLSD_SSD, self).__init__(True, input_shape, num_classes, num_domains,
                                  hparams)
class GLSD_FSD(GLSD):
    """GLSD_FSD algorithm """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GLSD_FSD, self).__init__(False, input_shape, num_classes, num_domains,
                                  hparams)


