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

import argparse
import datasets
import numpy as np
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
import gimli_v2

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

def train(data, model, lang_model, optimizer, epoch, args):
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
        
        # Higher weights on pixels that change more
        # Weight decay depending on number of epochs
        weight_value = 0
        tot_epochs = args.from_epoch + args.epochs
        if 0 <= (args.from_epoch + epoch) < (tot_epochs / 4):
            weight_value = 999
        elif (tot_epochs / 4) <= (args.from_epoch + epoch) < (tot_epochs / 2):
            weight_value = 99
        elif (tot_epochs / 2) <= (args.from_epoch + epoch) < (3 * tot_epochs / 4):
            weight_value = 9
        elif (args.from_epoch + epoch) > (3 * tot_epochs / 4):
            weight_value = 0
        weight = ((src_img - target_img)**2)*weight_value + 1
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
    
        
    if epoch % 10 == 0:
            tensor = torch.cat((src_img.cpu().data, target_img.cpu().data, output.cpu().data), 0)
            save_image(tensor, './dc_img/modelG_{}.png'.format(epoch), nrow=args.batch_size, normalize=True, range=(-1, 1))
    
    loss_value = sum(epoch_loss)/len(epoch_loss)
    writer.add_scalar('Loss/train', loss_value, epoch)
    
    
def test(data, model, lang_model, epoch, args):
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




def main(args):
    # Define parameters
    workers = 0
    batch_shuffle = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    trainset, testset = load_dataset(args.data_dir)
    train_loader = trainset.get_loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = testset.get_loader(batch_size=args.batch_size, shuffle=batch_shuffle, num_workers=args.num_workers)

    model = gimli_v2.generator()
    print(model)
    # language_model = SentenceTransformer('all-MiniLM-L6-v2')
    language_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    if torch.cuda.device_count() > 0 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()
    
    if args.cuda:
        model.cuda()
    
    model.apply(gimli_v2.weights_init)
    
    start_epoch = 1
    progress_bar = trange(start_epoch, args.epochs + 1)
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma)
    scheduler.last_epoch = start_epoch
    print('\nTraining ({} epochs) is starting...'.format(args.epochs))
    for epoch in progress_bar:
        
        print('\n\tCurrent learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        # TRAIN
        progress_bar.set_description('TRAIN')
        train(train_loader, model, language_model, optimizer, epoch, args)

    
        # TEST
        progress_bar.set_description('TEST')
        test(test_loader, model, language_model, epoch, args)


        if epoch % 10 == 0:
            torch.save(model, './modelG_epoch_{}.pth'.format(epoch))
                    
        
    writer.flush()
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
    # parser.add_argument('--resume', type=str,
    #                     help='resume from model stored')
    parser.add_argument('--data-dir', type=str, default='../',
                        help='base directory of CSS3D dataset containing .npy file and /images/ directory')
    # parser.add_argument('--model', type=str, default='original-fp',
    #                     help='which model is used to train the network')
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
