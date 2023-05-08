"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""

import torch
import torch.nn as nn
import torch.jit

from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
import matplotlib.pyplot as plt


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class DiffusionCoTTA(TTAMethod):
    """CoTTA
    """
    def __init__(self, model, optimizer, steps, episodic, window_length, dataset_name, mt_alpha=0.999, rst_m=0.01, ap=0.9, n_augmentations=32):
        super().__init__(model.cuda(), optimizer, steps, episodic, window_length)

        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.n_augmentations = n_augmentations
        self.counter = 0

        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = self.copy_model(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.softmax_entropy = softmax_entropy_cifar if "cifar" in dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(dataset_name)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        # import ipdb
        # ipdb.set_trace()
        imgs_test = x[0]
        imgs_c = imgs_test[:,0]
        imgs_t = imgs_test[:,1]
        f, ax = plt.subplots(1,2)
        ax[0].imshow(imgs_c[0].permute(1,2,0).cpu().numpy())
        ax[1].imshow(imgs_t[0].permute(1,2,0).cpu().numpy())
        plt.savefig("diff_cotta_testing1.png")
        plt.close()
        # imgs_t = 0.5*(imgs_t + imgs_c)
        anchor_prob_c = torch.nn.functional.softmax(self.model_anchor(imgs_c), dim=1).max(1)[0]
        anchor_prob_t = torch.nn.functional.softmax(self.model_anchor(imgs_t), dim=1).max(1)[0]
        anchor_prob = 0.5*(anchor_prob_c + anchor_prob_t)
        if anchor_prob_c.mean(0) > anchor_prob_t.mean(0):
            imgs_t = 0.5*imgs_t + 0.5*imgs_c
            # imgs_t = imgs_c 
        outputs_c = self.model(imgs_c)
        outputs_t = self.model(imgs_t)
        

        # outputs_mask = torch.nn.functional.softmax(outputs_c, dim = 1).max(1)[0] > torch.nn.functional.softmax(outputs_t, dim = 1).max(1)[0]
        # outputs_mask = outputs_mask.float().unsqueeze(1)
        # outputs = (outputs_mask)*outputs_c + (1-outputs_mask)*(outputs_t + outputs_c)
        # outputs = 0.5*(outputs_c + outputs_t)

        # Create the prediction of the anchor (source) model and the teacher model
        standard_ema_c = self.model_ema(imgs_c).detach()
        standard_ema_t = self.model_ema(imgs_t).detach()
        # mask = anchor_prob_c > anchor_prob_t
        # mask = mask.float().unsqueeze(1)
        # standard_ema_mask = torch.nn.functional.softmax(standard_ema_c, dim = 1).max(1)[0] > torch.nn.functional.softmax(standard_ema_t, dim = 1).max(1)[0]
        # standard_ema_mask = standard_ema_mask.float().unsqueeze(1)
        # standard_ema = standard_ema_mask*standard_ema_c + (1-standard_ema_mask)*(standard_ema_t + standard_ema_c)
        # standard_ema = 0.5*(standard_ema_c + standard_ema_t)
        # anchor_mask = anchor_prob_c > anchor_prob_t

        # Augmentation-averaged Prediction
        outputs_emas = []
        outputs_cs_ = []
        outputs_ts_ = [] 
        if anchor_prob_c.mean(0) < self.ap:
            self.counter += 1
            for i in range(self.n_augmentations):
                outputs_c_ = self.model_ema(self.transform(imgs_c)).detach()
                outputs_t_ = self.model_ema(self.transform(imgs_t)).detach()
                # outputs_ = 0.5*(outputs_c_ + outputs_t_)
                # outputs_mask = torch.nn.functional.softmax(outputs_c_, dim = 1).max(1)[0] > torch.nn.functional.softmax(outputs_t_, dim = 1).max(1)[0]
                # outputs_mask_ = outputs_mask.float().unsqueeze(1)
                # outputs_ = outputs_mask_*outputs_c_ + (1-outputs_mask_)*(outputs_t_ + outputs_c_)
                # outputs_emas.append(outputs_)
                outputs_cs_.append(outputs_c_)
                outputs_ts_.append(outputs_t_)

            # Threshold choice discussed in supplementary
            outputs_c_ = torch.stack(outputs_cs_).mean(0)
            outputs_t_ = torch.stack(outputs_ts_).mean(0)
        else:
            outputs_c_ = standard_ema_c
            outputs_t_ = standard_ema_t
        mask = torch.nn.functional.softmax(outputs_c_, dim = 1).max(1)[0] > torch.nn.functional.softmax(outputs_t_, dim = 1).max(1)[0]
        # mask_c = torch.nn.functional.softmax(outputs_c_, dim = 1).max(1)[0] > self.ap
        # mask_t = torch.nn.functional.softmax(outputs_t_, dim = 1).max(1)[0] > self.ap
        mask = mask.float().unsqueeze(1)
        # mask_c = mask_c.float().unsqueeze(1)
        # mask_t = mask_t.float().unsqueeze(1)
        # outputs_ema = mask*mask_c*outputs_c_ + (1-mask)*mask_t*(outputs_t_) + (1-mask)*(1-mask_t)*(outputs_t_ + outputs_c_) + mask*(1-mask_c)*(outputs_t_ + outputs_c_) 
        # outputs = mask*mask_c*outputs_c + (1-mask)*mask_t*(outputs_t) + (1-mask)*(1-mask_t)*(outputs_t + outputs_c) + mask*(1-mask_c)*(outputs_t + outputs_c)
        # outputs_ema = mask*outputs_c_ + (1-mask)*0.5*(outputs_t_ + outputs_c_)
        # outputs = mask*outputs_c + (1-mask)*0.5*(outputs_t + outputs_c)
        outputs_ema = mask*outputs_c_ + (1-mask)*(outputs_t_)
        outputs = mask*outputs_c + (1-mask)*0.5*(outputs_t)
        
        # print(outputs.shape)
        # print(outputs_ema.shape)
        # Student update
        loss = (self.softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        if self.rst > 0.:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model_ema(imgs_test)

    @staticmethod
    def configure_model(model):
        """Configure model."""
        # model.train()
        model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
        return model


@torch.jit.script
def softmax_entropy_cifar(x, x_ema):# -> torch.Tensor: 
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema):# -> torch.Tensor:       
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1) 
