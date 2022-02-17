import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
from torch.utils.data import DataLoader,Dataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,accuracy
)
####dataloader
from src.multicropdataset import MultiCropDataset
from src.multimodality import MultiModalityDataset
from src.multiview import MultiviewDataset
from src.testloader import ClassifyDataset

###backbone



import src.resnet50 as resnet_models
import src.con5 as con5_models
import src.fc as fc_models

logger = getLogger()
test_logger=getLogger()


parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--use_pil_blur", type=bool_flag, default=True,
                    help="""use PIL library to perform blur instead of opencv""")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--improve_numerical_stability", default=False, type=bool_flag,
                    help="improves numerical stability in Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=50 , type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=15,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=0,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=1, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="fc", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default="./cifar-fc-3-30-new",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

################ 
parser.add_argument("--test_dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")
parser.add_argument("--test_epochs", default=150, type=int,
                    help="number of total epochs to run")
parser.add_argument("--classify_lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--classify_gamma", type=float, default=0.1, help="decay factor")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])

def main():
    
    data_path='/data/multiview-cifar10-set'
    global args
    global best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    test_logger,test_training_stats = initialize_exp(
        args, "epoch", "loss"
    )
   
    # train_dataset=MultiModalityDataset(data_path,'/data/cifar10-multi.txt')
    train_dataset=MultiviewDataset(data_path,'/data/cifar10-multi.txt')    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    test_train_dataset = ClassifyDataset(os.path.join('/data/cifar10-imageformat', "train"))
    test_train_loader=DataLoader(test_train_dataset,batch_size=args.batch_size,shuffle=True)
    
    test_val_dataset = ClassifyDataset(os.path.join('/data/cifar10-imageformat', "test"))
    test_val_loader=DataLoader(test_val_dataset,batch_size=args.batch_size,shuffle=True)
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )


    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    
    model = fc_models.__dict__['fc'](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
  
    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
        # print(queue)
        training_stats.update(scores)


        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)
        
        
        if  epoch >100 and epoch % 25==0:
            linear_classifier = RegLog(10, args.arch, args.global_pooling, args.use_bn)
            linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)
            linear_classifier = linear_classifier.cuda()
            linear_classifier = nn.parallel.DistributedDataParallel(
                        linear_classifier,
                        device_ids=[args.gpu_to_work_on],
                        find_unused_parameters=True,
            )
            model.eval()

            # set optimizer
            optimizer = torch.optim.SGD(
                linear_classifier.parameters(),
                lr=args.classify_lr,
                nesterov=args.nesterov,
                momentum=0.9,
                weight_decay=args.wd,
            )

            # set scheduler
            if args.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, args.decay_epochs, gamma=args.classify_gamma
                )
            elif args.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, args.epochs, eta_min=args.final_lr
                )

            # Optionally resume from a checkpoint
            to_restore = {"epoch": 0, "best_acc": 0.}
            start_epoch = to_restore["epoch"]
            best_acc = to_restore["best_acc"]
            
            
            
            for epoch in range(0, args.epochs):

                # train the network for one epoch
                test_logger.info("============ Starting epoch %i ... ============" % epoch)

                scores = test_train(model, linear_classifier, optimizer, test_train_loader, epoch)
                scores_val = test_validate_network(test_val_loader, model, linear_classifier)
                print(scores,scores_val)
                test_training_stats.update(scores + scores_val)

                scheduler.step()

                # save checkpoint
                if args.rank == 0:
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.test_dump_path, "checkpoint.pth.tar"))
            test_logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                        "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    softmax = nn.Softmax(dim=1).cuda()
    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)
        d_loss=0

        # update learning rate
       
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        _, embedding, output = model(inputs)
        embedding = embedding.detach()
        # print(embedding)


        # print(output)
        ########embeddingtorch.Size([256, 128])

        #####output#####torch.Size([256, 3000])
        
        bs = inputs[0].size(0)
        
        h_s=embedding[bs*0:bs*(1)]
        h_t=embedding[bs*1:bs*(2)] 

        loss = 0
      
        
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]        
                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                   
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                
                
                # get assignments
                q = out / args.epsilon
                if args.improve_numerical_stability:
                    M = torch.max(q)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q -= M
                q = torch.exp(q).t()  #转置
                q = distributed_sinkhorn(q, args.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction      
            subloss = 0
            
            # [A, B, a1, b1, a2, b2, a3, b3]
            view1=np.array([0, 2, 4, 6])
            view2=np.array([1, 3, 5, 7])
            if crop_id == 0:

                for v in view2:
                    p = softmax(output[bs * v: bs * (v + 1)] / args.temperature)
                
                    subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            else:
                for v in view1:
                    p = softmax(output[bs * v: bs * (v + 1)] / args.temperature)
                    subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            
           
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)
        
        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            elif arch == "fc":
                s = 128
            elif arch == "conv5":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
       
        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def test_train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()
    # print(model)
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        # move to gpu
        # inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
      
        # forward
        with torch.no_grad(): 
    
            if not isinstance(inp, list):
                inp = [inp]


            idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([x.shape[-1] for x in inp]),
            return_counts=True,
            )[1], 0)

         ####backbone
         
            output = model(torch.cat(inp).cuda(non_blocking=True))[0]
                
        
        output = reglog(output)
        target=torch.cat([target for i in range(2)],dim=0)
        
        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            test_logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def test_validate_network(val_loader, model, linear_classifier):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu  
            
            target = target.cuda(non_blocking=True)
            if not isinstance(inp, list):
                inp = [inp]
           
            idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([x.shape[-1] for x in inp]),
            return_counts=True,
            )[1], 0)

            output = model(torch.cat(inp).cuda(non_blocking=True))[0]
            # print(output)
            output = linear_classifier(output)
            target=torch.cat([target for i in range(2)],dim=0)

            
            
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    if args.rank == 0:
        test_logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))

    return losses.avg, top1.avg.item(), top5.avg.item()

if __name__ == "__main__":
    main()
