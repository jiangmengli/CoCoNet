from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse

from torchvision import transforms
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter, set_requires_grad

from models.alexnet import AlexNet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageFolderInstance
from gswnn import GSW_NN

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""
torch.autograd.set_detect_anomaly(True)

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # set hyper-params
    parser.add_argument('--alpha', type=float, default=1e-8)
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1e-4)

    # gsw
    parser.add_argument('--gswnn_type', default='max_gsw', type=str, help='gswnn_type', choices=['gsw', 'max_gsw'])
    parser.add_argument('--gswnn_depth', default=3, type=int, help='gswnn_depth')
    parser.add_argument('--gsw_nofprojections', default='num_classes', type=str,
                        help='gsw_nofprojections, could be:'
                             '1. \'num_classes\' (str), means using \'num_classes\' as the projections number'
                             '2. \'(numbers)\' (str) such as \'100\', means using this number as projections number')
    parser.add_argument('--gsw_lr', default=1e-5, type=float, help='gsw_lr')
    parser.add_argument('--gsw_wd', default=0.0005, type=float, help='gsw_wd')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet100', 'imagenet'])
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder + args.dataset + '/', 'train')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = AlexNet(args.feat_dim)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l_ori = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ori_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_l_comp = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab_comp = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_comp_ori = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab_l = criterion_ab_l.cuda()
        criterion_l_ori = criterion_l_ori.cuda()
        criterion_ori_ab = criterion_l_ori.cuda()
        criterion_comp_ori = criterion_l_ori.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab_l, criterion_l_ori, criterion_ori_ab, criterion_l_comp, criterion_ab_comp, criterion_comp_ori


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_ab_l, criterion_l_ori, criterion_ori_ab, criterion_l_comp, criterion_ab_comp, criterion_comp_ori, gsw, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()

        # ===================forward=====================
        feat_l, feat_ab, feat_ori, feat_comp = model(inputs)
        out_l_ori, out_ab_l, out_ori_ab, out_ab_comp, out_l_comp, out_comp_ori = contrast(feat_l, feat_ab, feat_ori, feat_comp, index)

        l_ori_loss = criterion_l_ori(out_l_ori)
        ab_l_loss = criterion_ab_l(out_ab_l)
        ori_ab_loss = criterion_ori_ab(out_ori_ab)
        l_comp_loss = criterion_l_comp(out_l_comp)
        ab_comp_loss = criterion_ab_comp(out_ab_comp)
        comp_ori_loss = criterion_comp_ori(out_comp_ori)

        loss = opt.beta * (l_ori_loss + ab_l_loss + ori_ab_loss) + opt.alpha * (l_comp_loss + ab_comp_loss + comp_ori_loss)

        # Calc gsw
        set_requires_grad(model, requires_grad=False)
        if opt.gswnn_type == 'max_gsw':
            loss += opt.gamma * gsw.max_gsw(feat_l, feat_ab, feat_ori)
        else:
            loss += opt.gamma * gsw.gsw(feat_l, feat_ab, feat_ori)
        set_requires_grad(model, requires_grad=True)

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return True


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab_l, criterion_l_ori, criterion_ori_ab, criterion_l_comp, criterion_ab_comp, criterion_comp_ori = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set gsw
    if torch.cuda.is_available() == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.gsw_nofprojections == 'num_classes':
        gsw_nofprojections = int(args.num_classes)
    else:
        gsw_nofprojections = int(args.gsw_nofprojections)
    gsw = GSW_NN(device, args.gsw_lr, din=128, num_filters=args.batch_size, nofprojections=gsw_nofprojections,
                 model_depth=args.gswnn_depth, train_wd=args.gsw_wd)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        _ = train(epoch, train_loader, model, contrast, criterion_ab_l, criterion_l_ori, criterion_ori_ab, criterion_l_comp, criterion_ab_comp, criterion_comp_ori, gsw, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
