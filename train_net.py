import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from dataset import *
# from base_model import *
from utils import *
from models import *


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    training_set, validation_set = return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4,
    }
    training_loader = data.DataLoader(training_set, **params)

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    model = GCNnet(cfg)
    # Load backbone
    model.loadmodel(cfg.stage1_model_path)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)

    model.train()
    model.apply(set_bn_eval)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)

    if cfg.test_before_train:
        test_info = test(validation_loader, model, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result = {'epoch': 0, 'activities_acc': 0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # One epoch of forward and backward
        train_info = train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        test_info = test(validation_loader, model, device, epoch, cfg)
        show_epoch_info('Test', cfg.log_path, test_info)

        if test_info['activities_acc'] > best_result['activities_acc']:
            best_result = test_info
        print_log(cfg.log_path,
                  'Best group activity accuracy: %.2f%% at epoch #%d.' % (
                  best_result['activities_acc'], best_result['epoch']))


def train(data_loader, model, device, optimizer, epoch, cfg):
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]

        batch_size = batch_data[0].shape[0]

        activities_in = torch.squeeze(batch_data[2])

        # forward
        activities_scores = model((batch_data[0], batch_data[1], batch_data[3]))

        # Predict activities
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]

        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss = activities_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return train_info


def test(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()

    epoch_timer = Timer()
    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_data_test = [b.to(device=device) for b in batch_data_test]
            batch_size = batch_data_test[0].shape[0]
            activities_in_test = torch.squeeze(batch_data_test[2])

            # forward
            activities_scores = model((batch_data_test[0], batch_data_test[1], batch_data_test[3]))

            # Predict activities
            activities_loss_test = F.cross_entropy(activities_scores, activities_in_test)
            activities_labels_test = torch.argmax(activities_scores, dim=1)

            activities_correct_test = torch.sum(torch.eq(activities_labels_test.int(), activities_in_test.int()).float())

            # Get accuracy
            activities_accuracy = activities_correct_test.item() / activities_scores.shape[0]

            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss_test
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return test_info


if __name__ == '__main__':
    cfg = Config('volleyball')

    cfg.device_list = "0"
    cfg.training_stage = 2
    cfg.stage1_model_path = './result/STAGE1_MODEL.pth'  # PATH OF THE BASE MODEL
    cfg.train_backbone = False

    cfg.batch_size = 16  # 32
    cfg.test_batch_size = 16
    cfg.num_frames = 6
    cfg.train_learning_rate = 1e-4
    cfg.lr_plan = {41: 5e-5, 81: 1e-5, 121: 5e-6}
    cfg.max_epoch = 100
    cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    cfg.exp_note = 'Volleyball_stage2'
    train_net(cfg)
