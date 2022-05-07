from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
from arcface import getAugmentation, ShopeeEncoderBackBone
from config import CFG
from dataset import ShopeeDataset
from shopee_scheduler import ShopeeScheduler
import os


def training_one_epoch(epoch_num, model, dataloader, optimizer, scheduler, device, loss_criteria):
    avgloss = 0.0
    model.train()
    tq = tqdm(enumerate(dataloader), total=len(dataloader))
    y_true = []
    y_pred = []
    for idx, data in tq:
        batch_size = data[0].shape[0]
        images = data[0]
        targets = data[1]

        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        output, loss = model(images, targets)

        loss.backward()

        optimizer.step()

        predicted_label = torch.argmax(output, 1)
        y_true.extend(targets.detach().cpu().numpy())
        y_pred.extend(predicted_label.detach().cpu().numpy())
        avgloss += loss.item()

        tq.set_postfix({'loss': '%.6f' % float(avgloss / (idx + 1)), 'LR': optimizer.param_groups[0]['lr']})

    scheduler.step()
    f1_score_metric = f1_score(y_true, y_pred, average='micro')
    tq.set_postfix({'Training f1 score': '%.6f' % float(f1_score_metric)})
    return avgloss / len(dataloader), f1_score_metric


def validation_one_epoch(model, dataloader, epoch, device, loss_criteria):
    avgloss = 0.0
    model.eval()
    tq = tqdm(enumerate(dataloader), desc="Training Epoch { }" + str(epoch + 1))

    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, data in tq:
            batch_size = data[0].shape[0]
            images = data[0]
            targets = data[1]

            images = images.to(device)
            targets = targets.to(device)
            output, loss = model(images, targets)
            predicted_label = torch.argmax(output, 1)
            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(predicted_label.detach().cpu().numpy())

            avgloss += loss.item()

            tq.set_postfix({'validation loss': '%.6f' % float(avgloss / (idx + 1))})
    f1_score_metric = f1_score(y_true, y_pred, average='micro')
    tq.set_postfix({'validation f1 score': '%.6f' % float(f1_score_metric)})
    return avgloss / len(dataloader), f1_score_metric


import numpy as np


def get_class_weights(data):
    """
    Function get Class Weight, These weights will be useful for handing class imbalance issue
    Args:
        data : dataframe
    Returns:
        for each data point return class weight

    """
    # Word dictionary keys will be label and value will be frequency of label in dataset
    weight_dict = Counter(data['label_group'])
    # for each data point get label count data
    class_sample_count = np.array([weight_dict[row[4]] for row in data.values])
    # each data point weight will be inverse of frequency
    weight = 1. / class_sample_count
    weight = torch.from_numpy(weight)
    return weight


def run_training(model=None, history=None):
    data = pd.read_csv(CFG.DATA_DIR)
    labelencoder = LabelEncoder()
    data['label_group_original'] = data['label_group']
    data['label_group'] = labelencoder.fit_transform(data['label_group'])

    # create training_data and validation data initially not using k fold
    train_data = data[data['fold'] != 0]
    validation_data = data[data['fold'] == 0]

    train_aug = getAugmentation(CFG.img_size, isTraining=True)
    validation_aug = getAugmentation(CFG.img_size, isTraining=False)
    # create custom train and validation dataset

    trainset = ShopeeDataset(train_data, CFG.TRAIN_DIR, isTraining=True, transform=train_aug)
    validset = ShopeeDataset(validation_data, CFG.TRAIN_DIR, isTraining=False, transform=validation_aug)

    # get weights for  classes
    samples_weight = get_class_weights(train_data)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, num_samples=len(samples_weight))

    # create custom training and validation data loader num_workers=CFG.num_workers,
    train_dataloader = DataLoader(trainset, batch_size=CFG.batch_size,
                                  drop_last=True, pin_memory=True, sampler=sampler)

    validation_dataloader = DataLoader(validset, batch_size=CFG.batch_size,
                                       drop_last=True, pin_memory=True)

    # define loss function
    loss_criteria = nn.CrossEntropyLoss()
    loss_criteria.to(CFG.device)
    # define model

    if not model:
        model = ShopeeEncoderBackBone()
        model.to(CFG.device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.scheduler_params['lr_start'])

    # learning rate scheudler
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=1, eta_min=1e-6, last_epoch=-1)
    scheduler = ShopeeScheduler(optimizer, **CFG.scheduler_params)
    if not history:
        history = {'train_loss': [], 'validation_loss': [], 'train_f1_score': [], 'scheduler': [], 'valid_f1_score': []}
    for epoch in range(CFG.epochs):
        # get current epoch training loss
        avg_train_loss, avg_f1_score = training_one_epoch(epoch_num=epoch,
                                                          model=model,
                                                          dataloader=train_dataloader,
                                                          optimizer=optimizer,
                                                          scheduler=scheduler,
                                                          device=CFG.device,
                                                          loss_criteria=loss_criteria)
        print("Epoch : {} avg f1 {}".format(epoch + 1, avg_f1_score))
        # get current epoch validation loss
        avg_validation_loss, avg_valid_f1_score = validation_one_epoch(model=model,
                                                                       dataloader=validation_dataloader,
                                                                       epoch=epoch,
                                                                       device=CFG.device,
                                                                       loss_criteria=loss_criteria)

        print("Epoch : {} avg Validation f1 {}".format(epoch + 1, avg_f1_score))

        history['train_loss'].append(avg_train_loss)
        history['validation_loss'].append(avg_validation_loss)
        history['train_f1_score'].append(avg_f1_score)
        history['valid_f1_score'].append(avg_valid_f1_score)
        history['scheduler'].append(scheduler.state_dict())
        # save model
        torch.save(model.state_dict(), CFG.MODEL_PATH + "Train_F1_score_" + str(avg_f1_score) + "valid_f1_score" + str(
            avg_valid_f1_score) + "_" + "Epoch_" + str(epoch) + "_lr_start_" + str(
            CFG.scheduler_params['lr_start']) + "_lr_max_" + str(
            CFG.scheduler_params['lr_max']) + '_softmax_512x512_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            CFG.MODEL_PATH + "F1_score_" + str(avg_f1_score) + "valid_f1_score" + str(
                avg_valid_f1_score) + "_" + "Epoch_" + str(epoch) + "_lr_start_" + str(
                CFG.scheduler_params['lr_start']) + "_lr_max_" + str(
                CFG.scheduler_params['lr_max']) + '_softmax_512x512_{}.pt'.format(CFG.model_name)
        )

    return model, history


if __name__ == '__main__':
    # Training Mode on
    CFG.isTraining=True
    model, history = run_training()
    epoch_list = [i + 1 for i in range(CFG.epochs)]
    history_fm = pd.DataFrame.from_dict(history)
    history_fm.to_csv(os.path.join(CFG.OUT_DIR, 'model_history.csv'))
