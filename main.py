import json
import math
import pdb
from decimal import Decimal

import cv2
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import data
import model
import utility
from model.edsr import PAMS_EDSR
from model.edsr_org import EDSR
# from model.rdn import PAMS_RDN
# from model.rdn_org import RDN

from option import args
from utils import common as util
from utils.common import AverageMeter

import wandb
from datetime import datetime as dt

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')

class Trainer():
    def __init__(self, args, loader, t_model, s_model, ckp):
        self.args = args
        self.scale = args.scale

        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.t_model = t_model
        self.s_model = s_model
        arch_param = [v for k, v in self.s_model.named_parameters() if 'alpha' not in k]
        alpha_param = [v for k, v in self.s_model.named_parameters() if 'alpha' in k]

        params = [{'params': arch_param}, {'params': alpha_param, 'lr': 1e-2}]

        self.optimizer = torch.optim.Adam(params, lr=args.lr, betas = args.betas, eps=args.epsilon)
        self.sheduler = StepLR(self.optimizer, step_size=int(args.decay), gamma=args.gamma)
        self.writer_train = SummaryWriter(ckp.dir + '/run/train')
        
        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.s_model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.sheduler.load_state_dict(ckpt['scheduler'])

        self.losses = AverageMeter()
        self.att_losses = AverageMeter()
        self.nor_losses = AverageMeter()

    def train(self):
        self.sheduler.step(self.epoch)
        self.epoch = self.epoch + 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.writer_train.add_scalar(f'lr', lr, self.epoch)
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch, Decimal(lr))
        )

        self.t_model.eval()
        self.s_model.train()
        
        self.s_model.apply(lambda m: setattr(m, 'epoch', self.epoch))
        
        num_iterations = len(self.loader_train)
        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            num_iters = num_iterations * (self.epoch-1) + batch

            lr, hr = self.prepare(lr, hr)
            data_size = lr.size(0) 
            
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            if hasattr(self.t_model, 'set_scale'):
                self.t_model.set_scale(idx_scale)
            if hasattr(self.s_model, 'set_scale'):
                self.s_model.set_scale(idx_scale)

            with torch.no_grad():
                if args.model.lower() == 'carn':
                    t_sr, t_res, _ = self.t_model(lr/255., self.scale[0])
                    t_sr *= 255.
                else:
                    t_sr, t_res = self.t_model(lr)
            if args.model.lower() == 'carn':
                s_sr, s_res, _ = self.s_model(lr/255., self.scale[0])
                s_sr *= 255.
            else:
                s_sr, s_res = self.s_model(lr)

            nor_loss = args.w_l1 * F.l1_loss(s_sr, hr)
            att_loss = args.w_at * util.at_loss(s_res, t_res)

            loss = nor_loss  + att_loss

            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            self.losses.update(loss.item(),data_size)
            display_loss = f'Loss: {self.losses.avg: .3f}'

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    display_loss,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        
            for name, value in self.s_model.named_parameters():
                if 'alpha' in name:
                    if value.grad is not None:
                        self.writer_train.add_scalar(f'{name}_grad', value.grad.cpu().data.numpy(), num_iters)
                        self.writer_train.add_scalar(f'{name}_data', value.cpu().data.numpy(), num_iters)


    def test(self, is_teacher=False):
        torch.set_grad_enabled(False)
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        if is_teacher:
            model = self.t_model
        else:
            model = self.s_model
        model.eval()
        timer_test = utility.timer()
        
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                i = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    i += 1
                    lr, hr = self.prepare(lr, hr)

                    if args.model.lower() == 'carn':
                        sr, s_res, _ = model(lr/255., scale)
                        sr *= 255. 
                    else:
                        sr, s_res = model(lr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    cur_psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    cur_ssim = utility.calc_ssim(sr, hr, scale, benchmark=d.dataset.benchmark)
                    self.ckp.ssim_log[-1, idx_data, idx_scale] += cur_ssim
                    self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        save_name = f'{args.k_bits}bit_{filename[0]}'
                        self.ckp.save_results(d, save_name, save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}] PSNR: {:.3f}; SSIM: {:.3f}; (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.ssim_log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                self.writer_train.add_scalar(f'psnr', self.ckp.log[-1, idx_data, idx_scale], self.epoch)

        if self.args.save_results:
            self.ckp.end_background()
            
        if not self.args.test_only:
            is_best = (best[1][0, 0] + 1 == epoch)

            state = {
            'epoch': epoch,
            'state_dict': self.s_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.sheduler.state_dict()
        }
            util.save_checkpoint(state, is_best, checkpoint =self.ckp.dir + '/model')
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.cuda()

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

def main():
    if checkpoint.ok:
        loader = data.Data(args)
        if args.model.lower() == 'edsr':
            t_model = EDSR(args, is_teacher=True).to(device)
            # t_model = torch.nn.DataParallel(t_model)

            s_model = PAMS_EDSR(args, bias=True).to(device)
            
        # elif args.model.lower() == 'rdn':
        #     t_model = RDN(args, is_teacher=True).to(device)
        #     s_model = PAMS_RDN(args).to(device).to(device)
        else:
            raise ValueError('not expected model = {}'.format(args.model))

        if args.pre_train is not None:
            t_checkpoint = torch.load(args.pre_train) 
            t_checkpoint = t_checkpoint['state_dict'] if 'state_dict' in t_checkpoint else t_checkpoint
            t_model.load_state_dict(t_checkpoint)

            # quantized model load pre-train weighs
            s_model_dict = s_model.state_dict()
            pre_trained_dict = {}
            for k, v in t_checkpoint.items():
                if args.model.lower() == 'edsr':
                    if k in s_model_dict:
                        pre_trained_dict[k] = v
                    elif k.replace('.body.2', '.body.3') in s_model_dict:
                        pre_trained_dict[k.replace('.body.2', '.body.3')] = v
                    else:
                        print(k)
                elif args.model.lower() == 'carn':
                    if k in s_model_dict:
                        pre_trained_dict[k] = v
                    elif k.replace('.body.0', '.body.1') in s_model_dict:
                        pre_trained_dict[k.replace('.body.0', '.body.1')] = v
                    elif k.replace('.body.2', '.body.4') in s_model_dict:
                        pre_trained_dict[k.replace('.body.2', '.body.4')] = v
                    else:
                        print(k)

                else:
                    if k in s_model_dict:
                        pre_trained_dict[k] = v
                    else:
                        print(k)
            # check all pre-train parameter are loaded
            for k in pre_trained_dict:
                if args.model.lower() == 'edsr':
                    if k not in s_model_dict and k.replace('.body.2', '.body.3') not in s_model_dict:
                        print(k)
                else:
                    if k not in s_model_dict:
                        print(k)

            assert len(pre_trained_dict) == len(t_model.state_dict())
            print(len(pre_trained_dict), len(s_model_dict))
            s_model_dict.update(pre_trained_dict)
            s_model.load_state_dict(s_model_dict)
        
        if args.test_only:
            if args.refine is None:
                ckpt = torch.load(f'{args.save}/model/model_best.pth.tar')
                refine_path = f'{args.save}/model/model_best.pth.tar'
            else:
                ckpt = torch.load(f'{args.refine}')
                refine_path = args.refine

            s_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            s_model.load_state_dict(s_checkpoint)
            print(f"Load model from {refine_path}")

        t = Trainer(args, loader, t_model, s_model, checkpoint)

        # #Wandb
        # if not args.test_only:
        #     PROJECT = "test_gpu"
        #     run_id = f"{dt.now()}"
        #     name = f"{PROJECT}_{args.save}_{run_id}"
        #     wandb.init(project=PROJECT, name=name)
        
        print(f'{args.save} start!')
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()
        print(f'{args.save} done!')

        torch.save(s_model.state_dict(), "./s_model.pt")


if __name__ == '__main__':
    main()
