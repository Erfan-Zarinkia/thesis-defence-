from ast import arg
from pyexpat import model
from xmlrpc.client import Boolean, boolean
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import tqdm
from torch.backends import cudnn
import pickle
import argparse
from defenses import *
from utils import *

import os.path
from os import path

from loss.spc import SupervisedContrastiveLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform


cudnn.benchmark = True




############################## DATASET AND TRAINING CODE ##################################

def train(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, print_every=100, kmeans = 1):
    """
    Trains the provided model
    
    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    """
    
    for epoch in range(num_epochs):
        with torch.no_grad():
            loss, acc = evaluate(model, test_loader)
        print('Epoch: {}, Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        for i, (bx, by) in enumerate(loader):
            bx = bx.cuda()
            by = by.cuda()

            # forward pass
            logits = model(bx)


            #####
            predicted_new = (by/kmeans).type('torch.IntTensor').cuda()
            
            #####
            loss = loss_fn(logits, predicted_new.type('torch.cuda.LongTensor'))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print(i, loss.item())

    with torch.no_grad():
        loss, acc = evaluate(model, test_loader)
    print('Final:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))
    model.eval()

def train_contrastive(model, loader, optimizer, scheduler,loss_fn, num_epochs=50):

    best_loss = float("inf")

    model.train()

    for epoch in range(num_epochs):

        train_loss = 0
        
        for i, (bx, by) in enumerate(loader):

            bx = torch.cat(bx)
            by = by.repeat(2)
        
            bx = bx.cuda()
            by = by.cuda()

            # forward pass
            logits = model.module.forward_constrative(bx)
            loss = loss_fn(logits, by)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        loss_average = train_loss/len(loader)
        print('Epoch: {}, Test Loss: {:.7f}'.format(epoch, loss_average))


        #if epoch % 10 == 0:
        #if (train_loss / (i + 1)) < best_loss:
    print("Saving..")
    state = {
        "net": model.state_dict(),
        "epoch": epoch,
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, "./checkpoint/ckpt_contrastive.pth")


    print('Finish')



################################################################

def cross_entropy_loss(logits, gt_target):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


def main(args):


    if (args.training_mode == "cross-entropy"):

        train_data, _ = load_data(args.dataset, train=True)
        test_data, num_classes = load_data(args.dataset, train=False)

        def misinformation_loss(logits, gt_target):
            """
            :param logits: tensor of shape (N, C); the predicted logits
            :param gt_target: long tensor of shape (N,); the gt class labels
            :returns: cross entropy loss
            """
            smax = torch.softmax(logits, dim=1)
            loss = -1 * (((1 - smax) * torch.nn.functional.one_hot(gt_target, num_classes=num_classes)).sum(1) + 1e-12).log().mean(0)
            return loss
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=False, pin_memory=True)

        if args.misinformation == '1':
            loss = misinformation_loss
        else:
            loss = cross_entropy_loss

        ################# TRAINING ####################
        print('\nTraining model on: {}'.format(args.dataset))

        
        teacher = load_model(args.dataset, num_classes, arch=args.arch)
        num_epochs = args.epochs
        lr = 0.01 if args.dataset == 'cub200' else 0.1
        optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

        train(teacher, train_loader, test_loader, optimizer, scheduler, loss, num_epochs=min(args.epochs ,args.early_stop_epochs))
    
    else:


        train_data, _ = load_data(args.dataset, train=True,training_mode =args.training_mode , custom = args.custom , data_path = args.data_path)
        test_data, num_classes = load_data(args.dataset, train=False)



        # def misinformation_loss(logits, gt_target):
        #     """
        #     :param logits: tensor of shape (N, C); the predicted logits
        #     :param gt_target: long tensor of shape (N,); the gt class labels
        #     :returns: cross entropy loss
        #     """
        #     smax = torch.softmax(logits, dim=1)
        #     loss = -1 * (((1 - smax) * torch.nn.functional.one_hot(gt_target, num_classes=num_classes)).sum(1) + 1e-12).log().mean(0)
        #     return loss
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=False, pin_memory=True)

        # if args.misinformation == '1':
        #     loss = misinformation_loss
        # else:
        #     loss = cross_entropy_loss

        ################# TRAINING ####################

        print('\nTraining model on: {}'.format(args.dataset))

        number_class = max(num_classes,args.number_class)

        teacher = load_model(args.dataset, number_class , training_mode = args.training_mode , arch=args.arch)

        num_epochs = args.epochs_contrastive
        lr = args.lr_contrastive /10 if args.dataset == 'cub200' else args.lr_contrastive
        optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))



        criterion = SupervisedContrastiveLoss(temperature=args.temperature)
        criterion.cuda()
        train_contrastive(teacher, train_loader, optimizer, scheduler,criterion, num_epochs)
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
        teacher.load_state_dict(checkpoint["net"])

        train_data, _ = load_data(args.dataset, train=True, custom = args.custom, data_path = args.data_path)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=True, pin_memory=True)
        num_epochs = args.epochs
        lr = 0.01 if args.dataset == 'cub200' else 0.1
        teacher.module.freeze_projection()
        optimizer = optim.SGD(
            teacher.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))
        criterion = nn.CrossEntropyLoss()
        criterion.cuda()

        #args.best_acc = 0.0
        train(teacher, train_loader, test_loader, optimizer, scheduler, criterion, num_epochs = num_epochs, kmeans=args.kmeans)


    print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
    torch.save(teacher.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--dataset', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--data_path', type=str, default='/kaggle/working/train_vic' ,help='data for training model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--misinformation', type=str,
        help='if "1", train a network with the misinformation loss for the Adaptive Misinformation method', default='0')
    
    parser.add_argument(
        "--training_mode",
        default="cross-entropy",
        choices=["contrastive", "cross-entropy"],
        help="Type of training use either a two steps contrastive then cross-entropy or \
                         just cross-entropy",
    )
    parser.add_argument('--number_class', type=int, default=10)
    parser.add_argument('--kmeans', type=int, default=1)
    parser.add_argument('--custom', type=bool, default=False)
    parser.add_argument("--lr_contrastive", default=0.2, type=float)
    parser.add_argument("--temperature", default=0.01, type=float, help="Constant for loss no thorough ")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_contrastive', type=int, default=50)
    parser.add_argument('--early_stop_epochs', type=int, default=50)

    parser.add_argument('--arch', type=str, default='wrn')

    args = parser.parse_args()
    print(args)

    main(args)
