# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the model."""


#!/usr/bin/python

import os
import re
import argparse
import datasets
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data
import torchvision
from torchvision.utils import save_image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
import gimli
import gimli_v2
import gimli_with_attention
from pickle import dump
from utils import SaveBestModel, save_model, save_plots

writer = SummaryWriter()

def load_dataset(dir_path): 
    print('\nInitializing dataset!')
    print('\n==> Train data...')
    trainset = datasets.CSSDataset(
        path=dir_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    )
    print('==> Test data...\n')
    testset = datasets.CSSDataset(
        path=dir_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    )
    print('\nDataset initialized!')
    print('\ttrainset size:', len(trainset))
    print('\ttestset size:', len(testset))
    
    return trainset, testset

def train(data, model, lang_model, optimizer, epoch, args, loss_dict):
    model.train()
    

    avg_loss = 0.0
    n_batches = 0
    epoch_loss = []
    progress_bar = tqdm(data)
    criterion = nn.MSELoss(reduction='mean')
    for batch_idx, sample_batched in enumerate(progress_bar):
        
        assert type(sample_batched) is list
        src_img = np.stack([d['source_img_data'] for d in sample_batched])
        src_img = torch.from_numpy(src_img).float()
        target_img = np.stack([d['target_img_data'] for d in sample_batched])
        target_img = torch.from_numpy(target_img).float()
        if torch.cuda.device_count() > 0 and args.cuda:
            src_img = torch.autograd.Variable(src_img).cuda()
            target_img = torch.autograd.Variable(target_img).cuda()
        else:
            src_img = torch.autograd.Variable(src_img).cpu()
            target_img = torch.autograd.Variable(target_img).cpu()
        action_list = [str(d['mod']['str']) for d in sample_batched]
        action_embeddings = lang_model.encode(action_list)
        embedding_tensor = torch.from_numpy(action_embeddings)
       
        
        # Forward pass
        model.zero_grad()
        output, _, _ ,_ = model(src_img, embedding_tensor)
        
        loss = criterion(output, target_img)
        # Store loss for tracking
        epoch_loss.append(loss.item())
        
        # Higher weights on pixels that change more
        # Weight decay depending on number of epochs
        weight_value = 0       
        if epoch < 1000:
            weight_value = 1000-epoch
        elif epoch >= 1000:
            weight_value = 1
            
        weight = ((src_img - target_img)**2)*weight_value
        
        if torch.cuda.device_count() > 0 and args.cuda:
            weight = weight.type(torch.FloatTensor).cuda()
        else:
            weight = weight.type(torch.FloatTensor).cpu()
            
        loss = loss*weight
        loss = loss.mean()
        
        # Bacward pass
        loss.backward()
        
        # Store loss for tracking
        epoch_loss.append(loss.item())
        
            
        # Gradient Clipping
        if args.clip_norm:
            clip_grad_norm_(model.parameters(), args.clip_norm)
            
        optimizer.step()
        
        # Show progress
        progress_bar.set_postfix(dict(loss=loss.item()))
        avg_loss += loss.item()
        n_batches += 1
        
        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('\nTrain Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0
    
        
    if epoch % 50 == 0:
            tensor = torch.cat((src_img.cpu().data, target_img.cpu().data, output.cpu().data), 0)
            save_image(tensor, './dc_img/modelG_{}_{}.png'.format(args.model, epoch), nrow=args.batch_size, normalize=True, range=(-1, 1))
            with open('actions_{}_{}.pkl'.format(args.model, epoch), 'wb') as file:
                dump(action_list, file)
    
    loss_value = sum(epoch_loss)/len(epoch_loss)
    writer.add_scalar('Loss/train', loss_value, epoch)
    loss_dict[epoch] = loss_value
    
    return criterion
    
    
def validate(data, model, lang_model, epoch, args, loss_dict):
    model.eval()
    

    avg_loss = 0.0
    n_batches = 0
    epoch_loss = []
    progress_bar = tqdm(data)
    criterion = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(progress_bar):
            
            assert type(sample_batched) is list
            src_img = np.stack([d['source_img_data'] for d in sample_batched])
            src_img = torch.from_numpy(src_img).float()
            target_img = np.stack([d['target_img_data'] for d in sample_batched])
            target_img = torch.from_numpy(target_img).float()
            if torch.cuda.device_count() > 0 and args.cuda:
                src_img = torch.autograd.Variable(src_img).cuda()
                target_img = torch.autograd.Variable(target_img).cuda()
            else:
                src_img = torch.autograd.Variable(src_img).cpu()
                target_img = torch.autograd.Variable(target_img).cpu()
            action_list = [str(d['mod']['str']) for d in sample_batched]
            action_embeddings = lang_model.encode(action_list)
            embedding_tensor = torch.from_numpy(action_embeddings)
        
            
            # Forward pass
            output, _, _ ,_ = model(src_img, embedding_tensor)
            
            loss = criterion(output, target_img)
            # Store loss for tracking
            epoch_loss.append(loss.item())
            
            # Higher weights on pixels that change more
            # Weight decay depending on number of epochs
            # weight_value = 0
            # tot_epochs = args.from_epoch + args.epochs
            # if 0 <= (args.from_epoch + epoch) < (tot_epochs / 4):
            #     weight_value = 999
            # elif (tot_epochs / 4) <= (args.from_epoch + epoch) < (tot_epochs / 2):
            #     weight_value = 99
            # elif (tot_epochs / 2) <= (args.from_epoch + epoch) < (3 * tot_epochs / 4):
            #     weight_value = 9
            # elif (args.from_epoch + epoch) > (3 * tot_epochs / 4):
            #     weight_value = 0
            # weight = ((src_img - target_img)**2)*weight_value + 1
            # if torch.cuda.device_count() > 0 and args.cuda:
            #     weight = weight.type(torch.FloatTensor).cuda()
            # else:
            #     weight = weight.type(torch.FloatTensor).cpu()
            # loss = loss*weight
            # loss = loss.mean()
            

            # Show progress
            progress_bar.set_postfix(dict(loss=loss.item()))
            avg_loss += loss.item()
            n_batches += 1
            
            if batch_idx % args.log_interval == 0:
                avg_loss /= n_batches
                processed = batch_idx * args.batch_size
                n_samples = len(data) * args.batch_size
                progress = float(processed) / n_samples
                print('\nVal Epoch: {} [{}/{} ({:.0%})] Val loss: {}'.format(
                    epoch, processed, n_samples, progress, avg_loss))
                avg_loss = 0.0
                n_batches = 0
    
        loss_value = sum(epoch_loss)/len(epoch_loss)
        writer.add_scalar('Loss/val', loss_value, epoch)
        loss_dict[epoch] = loss_value
        
        return loss_value
    
    
def test(data, model, lang_model, epoch, args, loss_dict):
    model.eval()

    if args.loss == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    avg_loss = 0.0
    n_batches = 0
    epoch_loss = []
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        
        assert type(sample_batched) is list
        src_img = np.stack([d['source_img_data'] for d in sample_batched])
        src_img = torch.from_numpy(src_img).float()
        target_img = np.stack([d['target_img_data'] for d in sample_batched])
        target_img = torch.from_numpy(target_img).float()
        if torch.cuda.device_count() > 0 and args.cuda:
            src_img = torch.autograd.Variable(src_img).cuda()
            target_img = torch.autograd.Variable(target_img).cuda()
        else:
            src_img = torch.autograd.Variable(src_img).cpu()
            target_img = torch.autograd.Variable(target_img).cpu()
        action_list = [str(d['mod']['str']) for d in sample_batched]
        action_embeddings = lang_model.encode(action_list)
        embedding_tensor = torch.from_numpy(action_embeddings)
        
        output, _, _, _ = model(src_img, embedding_tensor)
        
        if args.loss == 'loglikelihood':
            loss = F.nll_loss(output, target_img)
        else:
            loss = criterion(output, target_img)

        
        progress_bar.set_postfix(dict(loss=loss.item()))
        avg_loss += loss.item()
        epoch_loss.append(loss.item())
        n_batches += 1
        
        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('\nTest Epoch: {} [{}/{} ({:.0%})] Test loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            
        
    avg_loss /= len(data)
    writer.add_scalar('Loss/test', avg_loss, epoch)
    # Store epoch avg loss
    loss_value = sum(epoch_loss)/len(epoch_loss)
    loss_dict[epoch] = loss_value


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Define parameters
    batch_shuffle = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    trainset, testset = load_dataset(args.data_dir)
    train_test = torch.utils.data.ConcatDataset([trainset, testset])
    train_size = int(0.8 * len(train_test))
    test_val_size = int(0.1 * len(train_test)) + 1
    print('Size of')
    print('\tconcatenated dataset: ', len(train_test))
    print('\ttrain set: ', train_size)
    print('\ttest set: ', test_val_size)
    print('\tval set: ', test_val_size)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(train_test, [train_size, test_val_size, test_val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   drop_last=True,
                                                   collate_fn=lambda i: i)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   drop_last=True,
                                                   collate_fn=lambda i: i)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   drop_last=True,
                                                   collate_fn=lambda i: i)
    model_name = ''
    start_epoch = 1
    if args.resume:
        filename = args.resume
        if os.path.isfile(filename):
            print('\t==> loading checkpoint {}...'.format(filename))
            if args.cuda:
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location=torch.device('cpu'))
                
            model = checkpoint
            print('\t==> loaded checkpoint {}\n'.format(filename))
        
        start_epoch = int(re.match(r'.*epoch_(\d+).pth', args.resume).groups()[0]) + 1
        print('\nFinished loading checkpoints. Starting from epoch {}\n\n!'.format(start_epoch))
    else:
        if 'original' in args.model.lower():
            model = gimli.generator()
            model_name = 'gimli-og'
        elif 'v2' in args.model.lower():
            model = gimli_v2.generator()
            model_name = 'gimli-1k'
        elif 'attention' in args.model.lower():
            model = gimli_with_attention.generator()
            model_name = 'gimli-attn'
    
    # print(model)
    # language_model = SentenceTransformer('all-MiniLM-L6-v2')
    language_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    if torch.cuda.device_count() > 0 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()
    
    if args.cuda:
        model.cuda()
    
    if not args.resume:
        model.apply(gimli_v2.weights_init)
    
    progress_bar = trange(start_epoch, args.epochs + 1)
    t_progress_bar = trange(start_epoch, args.epochs + 1)
    v_progress_bar = trange(start_epoch, args.epochs + 1)
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma)
    scheduler.last_epoch = start_epoch
    train_loss = {}
    val_loss = {}
    test_loss = {}
    start_time = time()
    save_best = SaveBestModel()
    print('\nTraining ({} epochs) is starting...'.format(args.epochs))
    for epoch in progress_bar:
        
        print('\n\tCurrent learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        # TRAIN
        progress_bar.set_description('TRAIN')
        criterion = train(train_dataloader, model, language_model, optimizer, epoch, args, train_loss)

        # VALIDATE
        v_progress_bar.set_description('VAL')
        validation = validate(val_dataloader, model, language_model, epoch, args, val_loss)

        # Save the best model
        save_best(validation, epoch, model, optimizer, criterion, model_name)
        
        # TEST
        t_progress_bar.set_description('TEST')
        test(test_dataloader, model, language_model, epoch, args, test_loss)


        if epoch % 50 == 0:
            torch.save(model.state_dict(), './{}_epoch_{}.pth'.format(model_name, epoch))
    
    total_time = time() - start_time
    print('\nTotal time taken:\n\t{}\n'.format(total_time))
    writer.add_scalar('Total time', total_time)
    writer.flush()
    # Store training loss
    with open('./{}_train_loss_{}_epochs.pkl'.format(model_name, args.epochs), 'wb') as file:
        dump(train_loss, file)
    # Store validation loss
    with open('./{}_val_loss_{}_epochs.pkl'.format(model_name, args.epochs), 'wb') as file:
        dump(val_loss, file)
    # Store testing loss
    with open('./{}_test_loss_{}_epochs.pkl'.format(model_name, args.epochs), 'wb') as file:
        dump(test_loss, file)
    # writer.close()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='number of workers in data loader (default: 0)')
    # .add_argument('--test-batch-size', type=int, default=640,
    #                     help='input batch size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR',
                        help='learning rate (default: 0.000005)')
    parser.add_argument('--clip-norm', type=int, default=50,
                        help='max norm for gradients; set to 0 to disable gradient clipping (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    parser.add_argument('--data-dir', type=str, default='../',
                        help='base directory of CSS3D dataset containing .npy file and /images/ directory')
    parser.add_argument('--model', type=str, default='original-fp',
                        help='which model is used to train the network')
    # parser.add_argument('--no-invert-questions', action='store_true', default=False,
    #                     help='invert the question word indexes for LSTM processing')
    # parser.add_argument('--test', action='store_true', default=False,
    #                     help='perform only a single test. To use with --resume')
    # parser.add_argument('--conv-transfer-learn', type=str,
    #                 help='use convolutional layer from another training')
    # parser.add_argument('--lr-max', type=float, default=0.0005,
    #                 help='max learning rate')
    parser.add_argument('--lr-gamma', type=float, default=2, 
                        help='increasing rate for the learning rate. 1 to keep LR constant.')
    parser.add_argument('--lr-step', type=int, default=20,
                        help='number of epochs before lr update')
    parser.add_argument('--loss', type=str, default='mse',
                        help='use log-likelihood loss function (default: mean squared error)')
    parser.add_argument('--from-epoch', type=int, default=0,
                        help='when using a pre-trained model')
    # parser.add_argument('--bs-max', type=int, default=-1,
    #                     help='max batch-size')
    # parser.add_argument('--bs-gamma', type=float, default=1, 
    #                     help='increasing rate for the batch size. 1 to keep batch-size constant.')
    # parser.add_argument('--bs-step', type=int, default=20, 
    #                     help='number of epochs before batch-size update')
    # parser.add_argument('--dropout', type=float, default=-1,
    #                     help='dropout rate. -1 to use value from configuration')
    # parser.add_argument('--config', type=str, default='config.json',
    #                     help='configuration file for hyperparameters loading')
    # parser.add_argument('--question-injection', type=int, default=-1, 
    #                     help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    args = parser.parse_args()
    # args.invert_questions = not args.no_invert_questions
    main(args)
