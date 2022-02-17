from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import argparse
import datetime
import socket
from torch.utils.data import distributed

from torchvision import transforms, datasets
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import AlexNet
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNet


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model_name', type=str, default='cmvc_wd', help='model name')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--model_path', type=str, default='./model_path/cmvc_wd/ckpt_epoch_240.pth', help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['stl-10', 'tiny-imagenet','cifar-10','cifar-100'])

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # path definition
    parser.add_argument('--data_folder', type=str, default='../../datasets/', help='path to data')
    parser.add_argument('--save_path', type=str, default='./save_path/', help='path to save linear classifier')
    parser.add_argument('--tb_path', type=str, default='./path_to_tensorboard/', help='path to tensorboard')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--mark', default=None, type=str, help='mark')

    parser.add_argument('--data_augmentation_en', default='no', type=str)

    parser.add_argument('--start_time', default=datetime.datetime.fromtimestamp(time.time()), type=str)

    opt_test = parser.parse_args()

    if (opt_test.data_folder is None) or (opt_test.save_path is None) or (opt_test.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path | tb_path')

    if opt_test.dataset == 'imagenet':
        if 'alexnet' not in opt_test.model:
            opt_test.crop_low = 0.08

    iterations = opt_test.lr_decay_epochs.split(',')
    opt_test.lr_decay_epochs = list([])
    for it in iterations:
        opt_test.lr_decay_epochs.append(int(it))

    opt_test.model_name = 'calibrated_{}_{}_bsz_{}_lr_{}_decay_{}'.format(opt_test.model_name, opt_test.model_path.split('/')[-2],
                                                                     opt_test.batch_size, opt_test.learning_rate, opt_test.weight_decay)

    opt_test.model_name = '{}_view_{}'.format(opt_test.model_name, opt_test.view)

    opt_test.tb_folder = os.path.join(opt_test.tb_path, opt_test.model_name + '_layer{}'.format(opt_test.layer))
    if not os.path.isdir(opt_test.tb_folder):
        os.makedirs(opt_test.tb_folder)

    opt_test.save_folder = os.path.join(opt_test.save_path, opt_test.model_name)
    if not os.path.isdir(opt_test.save_folder):
        os.makedirs(opt_test.save_folder)

    if opt_test.dataset == 'imagenet100':
        opt_test.n_label = 100
    if opt_test.dataset == 'imagenet':
        opt_test.n_label = 1000
    if opt_test.dataset == 'stl-10':
        opt_test.n_label = 10
    if opt_test.dataset == 'tiny-imagenet':
        opt_test.n_label = 200
    if opt_test.dataset == 'cifar-10':
        opt_test.n_label = 10
    if opt_test.dataset == 'cifar-100':
        opt_test.n_label = 100
    return opt_test


def get_train_val_loader(args):
    train_folder = os.path.join(args.data_folder, 'train')
    val_folder = os.path.join(args.data_folder, 'val')

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
    if args.data_augmentation_en == 'yes':
        train_dataset = datasets.ImageFolder(
            train_folder,
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.ImageFolder(
            val_folder,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif args.data_augmentation_en == 'no':
        train_dataset = datasets.ImageFolder(
            train_folder,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.ImageFolder(
            val_folder,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                color_transfer,
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise NotImplementedError('data_augmentation_en not supported: {}'.format(args.data_augmentation_en))
    print('number of train: {}'.format(len(train_dataset)))
    print('number of val: {}'.format(len(val_dataset)))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


def set_model(args):
    if args.model.startswith('alexnet'):
        model = AlexNet()
        classifier = LinearClassifierAlexNet(layer=args.layer, n_label=args.n_label, pool_type='max')
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    model.eval()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model, classifier, criterion


def set_optimizer(args, classifier):
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
    """
    one epoch training
    """
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        with torch.no_grad():
            feat_l, feat_ab, feat_ori = model(input, opt.layer)
            feat = torch.cat((feat_l.detach(), feat_ab.detach(), feat_ori.detach()), dim=1)

        output = classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 10))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat_l, feat_ab, feat_ori= model(input, opt.layer)
            feat = torch.cat((feat_l.detach(), feat_ab.detach(), feat_ori.detach()), dim=1)
            output = classifier(feat)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def main():
    global best_acc1
    best_acc1 = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("==========the args:=========\n", args)
    print("============================\n\n")

    print("==========the start time:=========\n", args.start_time)
    print("============================\n\n")

    # set the data loader
    train_loader, val_loader, train_sampler = get_train_val_loader(args)

    # set the model
    model, classifier, criterion = set_model(args)

    # set optimizer
    optimizer = set_optimizer(args, classifier)

    cudnn.benchmark = True

    # optionally resume linear classifier
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume)) 
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda()
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
        train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'best_acc1': train_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

        # tensorboard logger
        pass

    for i in ["first", "second", "third"]:
        print("==> testing ", i, "...")
        _ = validate(val_loader, model, classifier, criterion, args)


if __name__ == '__main__':
    best_acc1 = 0
    main()
