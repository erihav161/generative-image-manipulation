import os
import argparse
import re
import datasets
import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision.utils import save_image
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
import gimli
import gimli_v2
import gimli_with_attention
from pickle import dump

writer = SummaryWriter()


def load_dataset(dir_path): # use load_dataset(opt) if multiple datasets and models are available
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


def train(data, modelG, modelD, lang_model, optimizerG, optimizerD, epoch, args, g_loss_dict, d_loss_dict):
    modelG.train()
    modelD.train()
    
    criterion = nn.BCEWithLogitsLoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss(reduction='none') # for weighted loss
    avg_Dloss = 0.0
    avg_Gloss = 0.0
    epoch_dloss = []
    epoch_gloss = []
    n_batches = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        
        assert type(sample_batched) is list
        src_img = np.stack([d['source_img_data'] for d in sample_batched])
        src_img = torch.from_numpy(src_img).float()
        target_img = np.stack([d['target_img_data'] for d in sample_batched])
        target_img = torch.from_numpy(target_img).float()
        if args.cuda:
            src_img = torch.autograd.Variable(src_img).cuda()
            target_img = torch.autograd.Variable(target_img).cuda()
        else:
            src_img = torch.autograd.Variable(src_img).cpu()
            target_img = torch.autograd.Variable(target_img).cpu()
        # wrong_img = np.stack([d['wrong_img_data'] for d in sample_batched])
        # wrong_img = torch.from_numpy(wrong_img).float()
        # wrong_img = torch.autograd.Variable(wrong_img).cuda()
        action_list = [str(d['mod']['str']) for d in sample_batched]
        action_embeddings = lang_model.encode(action_list)
        embedding_tensor = torch.from_numpy(action_embeddings)
        
        real_labels = torch.ones(target_img.size(0))
        fake_labels = torch.zeros(target_img.size(0))
        # ======== One sided label smoothing ==========
        # Helps preventing the discriminator from overpowering the
        # generator adding penalty when the discriminator is too confident
        # =============================================
        smoothed_real_labels = torch.FloatTensor(real_labels.numpy() - 0.1)
        if args.cuda:
            real_labels = Variable(real_labels).cuda()
            smoothed_real_labels = Variable(smoothed_real_labels).cuda()
            fake_labels = Variable(fake_labels).cuda()
        else:
            real_labels = Variable(real_labels).cpu()
            smoothed_real_labels = Variable(smoothed_real_labels).cpu()
            fake_labels = Variable(fake_labels).cpu()
        
        
        # Train the discriminator
        modelD.zero_grad()
        fake_images, phi, phi_im, phi_s = modelG(src_img, embedding_tensor)
        outputs, _ = modelD(target_img, phi.detach(), phi_im.detach(), phi_s.detach())
        outputs = torch.squeeze(outputs, dim=(1,2,3))
        real_loss = criterion(outputs, smoothed_real_labels)
        outputs, _ = modelD(fake_images, phi.detach(), phi_im.detach(), phi_s.detach())
        outputs = torch.squeeze(outputs, dim=(1,2,3))
        fake_loss = criterion(outputs, fake_labels)
        # outputs, _ = modelD(wrong_img, phi.detach(), phi_im.detach(), phi_s.detach())
        # wrong_loss = criterion(outputs, fake_labels)
        d_loss = real_loss + fake_loss # + wrong_loss
        d_loss.backward()
        optimizerD.step()
        
        
        # Train the generator
        modelG.zero_grad()
        fake_images, phi, phi_im, phi_s = modelG(src_img, embedding_tensor)
        outputs, activation_fake = modelD(fake_images, phi.detach(), phi_im.detach(), phi_s.detach())
        _, activation_real = modelD(target_img, phi.detach(), phi_im.detach(), phi_s.detach())
        activation_fake = torch.mean(activation_fake, 0)
        activation_real = torch.mean(activation_real, 0)
        
        # ======= Generator Loss function============
        # This is a customized loss function, the first term is the regular cross entropy loss
        # The second term is feature matching loss, this measure the distance between the real and generated
        # images statistics by comparing intermediate layers activations
        # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
        # because it links the embedding feature vector directly to certain pixel values.
        # ===========================================
        outputs = torch.squeeze(outputs, dim=(1,2,3))
        g_fake_loss = criterion(outputs, real_labels)
        g_l2loss = 10 * l2_loss(activation_fake, activation_real.detach())
        g_l1loss = 10 * l1_loss(fake_images, target_img.detach())
        g_MSEloss = MSE_loss(src_img, target_img.detach())
        g_MSEloss = g_MSEloss*100
        g_MSEloss = g_MSEloss.mean()
        g_loss = g_fake_loss + g_l2loss + g_l1loss + g_MSEloss
        g_loss.backward()
        optimizerG.step()
        
        # Show progress
        progress_bar.set_postfix(dict(loss=d_loss.item()))
        avg_Dloss += d_loss.item()
        avg_Gloss += g_loss.item()
        epoch_dloss.append(d_loss.item())
        epoch_gloss.append(g_loss.item())
        n_batches += 1
        
        if batch_idx % args.log_interval == 0:
            avg_Dloss /= n_batches
            avg_Gloss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('\nTrain Epoch: {} [{}/{} ({:.0%})] Train loss: D{} G{}'.format(epoch, processed, n_samples, progress, avg_Dloss, avg_Gloss))
            writer.add_scalar('Loss/Train: Discriminator', avg_Dloss, epoch)
            writer.add_scalar('Loss/Train: Generator', avg_Gloss, epoch)
            avg_Gloss = 0.0
            avg_Dloss = 0.0
            n_batches = 0
    
    if epoch % 50 == 0:
        tensor = torch.cat((src_img.cpu().data, target_img.cpu().data, fake_images.cpu().data), 0)
        save_image(tensor, 'DCGAN_{}.png'.format(epoch), nrow=args.batch_size, normalize=True, range=(-1, 1))
        with open('actions_{}_{}_gan.pkl'.format(args.model, epoch), 'wb') as file:
            dump(action_list, file)
    # Store loss values
    g_loss_dict[epoch] = sum(epoch_gloss) / len(epoch_gloss)
    d_loss_dict[epoch] = sum(epoch_dloss) / len(epoch_dloss)
    
    
def validate(data, modelG, modelD, lang_model, epoch, args, g_loss_dict, d_loss_dict):
    modelG.eval()
    modelD.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss(reduction='none') # for weighted loss
    avg_Dloss = 0.0
    avg_Gloss = 0.0
    epoch_dloss = []
    epoch_gloss = []
    n_batches = 0
    progress_bar = tqdm(data)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(progress_bar):
            assert type(sample_batched) is list
            src_img = np.stack([d['source_img_data'] for d in sample_batched])
            src_img = torch.from_numpy(src_img).float()
            target_img = np.stack([d['target_img_data'] for d in sample_batched])
            target_img = torch.from_numpy(target_img).float()
            if args.cuda:
                src_img = torch.autograd.Variable(src_img).cuda()
                target_img = torch.autograd.Variable(target_img).cuda()
            else:
                src_img = torch.autograd.Variable(src_img).cpu()
                target_img = torch.autograd.Variable(target_img).cpu()
            # wrong_img = np.stack([d['wrong_img_data'] for d in sample_batched])
            # wrong_img = torch.from_numpy(wrong_img).float()
            # wrong_img = torch.autograd.Variable(wrong_img).cuda()
            action_list = [str(d['mod']['str']) for d in sample_batched]
            action_embeddings = lang_model.encode(action_list)
            embedding_tensor = torch.from_numpy(action_embeddings)
            
            real_labels = torch.ones(target_img.size(0))
            fake_labels = torch.zeros(target_img.size(0))
            # ======== One sided label smoothing ==========
            # Helps preventing the discriminator from overpowering the
            # generator adding penalty when the discriminator is too confident
            # =============================================
            smoothed_real_labels = torch.FloatTensor(real_labels.numpy() - 0.1)
            if args.cuda:
                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()
            else:
                real_labels = Variable(real_labels).cpu()
                smoothed_real_labels = Variable(smoothed_real_labels).cpu()
                fake_labels = Variable(fake_labels).cpu()
            
            
            # Validate the discriminator
            fake_images, phi, phi_im, phi_s = modelG(src_img, embedding_tensor)
            outputs, _ = modelD(target_img, phi.detach(), phi_im.detach(), phi_s.detach())
            outputs = torch.squeeze(outputs, dim=(1,2,3))
            real_loss = criterion(outputs, smoothed_real_labels)
            outputs, _ = modelD(fake_images, phi.detach(), phi_im.detach(), phi_s.detach())
            outputs = torch.squeeze(outputs, dim=(1,2,3))
            fake_loss = criterion(outputs, fake_labels)
            d_loss = real_loss + fake_loss # + wrong_loss
            
            
            # Validate the generator
            fake_images, phi, phi_im, phi_s = modelG(src_img, embedding_tensor)
            outputs, activation_fake = modelD(fake_images, phi.detach(), phi_im.detach(), phi_s.detach())
            _, activation_real = modelD(target_img, phi.detach(), phi_im.detach(), phi_s.detach())
            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)
            
            # ======= Generator Loss function============
            # This is a customized loss function, the first term is the regular cross entropy loss
            # The second term is feature matching loss, this measure the distance between the real and generated
            # images statistics by comparing intermediate layers activations
            # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
            # because it links the embedding feature vector directly to certain pixel values.
            # ===========================================
            outputs = torch.squeeze(outputs, dim=(1,2,3))
            g_fake_loss = criterion(outputs, real_labels)
            g_l2loss = 10 * l2_loss(activation_fake, activation_real.detach())
            g_l1loss = 10 * l1_loss(fake_images, target_img.detach())
            g_MSEloss = MSE_loss(src_img, target_img.detach())
            weight = ((src_img - target_img)**2)*1000
            if args.cuda:
                weight = weight.type(torch.FloatTensor).cuda()
            else:
                weight = weight.type(torch.FloatTensor).cpu()
            g_MSEloss = g_MSEloss*100
            g_MSEloss = g_MSEloss.mean()
            g_loss = g_fake_loss + g_l2loss + g_l1loss + g_MSEloss

            
            # Show progress
            progress_bar.set_postfix(dict(loss=d_loss.item()))
            avg_Dloss += d_loss.item()
            avg_Gloss += g_loss.item()
            epoch_dloss.append(d_loss.item())
            epoch_gloss.append(g_loss.item())
            n_batches += 1
            
            if batch_idx % args.log_interval == 0:
                avg_Dloss /= n_batches
                avg_Gloss /= n_batches
                processed = batch_idx * args.batch_size
                n_samples = len(data) * args.batch_size
                progress = float(processed) / n_samples
                print('\nValidation Epoch: {} [{}/{} ({:.0%})] Validation loss: D{} G{}'.format(epoch, processed, n_samples, progress, avg_Dloss, avg_Gloss))
                writer.add_scalar('Loss/Val: Discriminator', avg_Dloss, epoch)
                writer.add_scalar('Loss/Val: Generator', avg_Gloss, epoch)
                avg_Gloss = 0.0
                avg_Dloss = 0.0
                n_batches = 0
    
       # Store loss values
        g_loss_dict[epoch] = sum(epoch_gloss) / len(epoch_gloss)
        d_loss_dict[epoch] = sum(epoch_dloss) / len(epoch_dloss)
    
    
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
    # Supress warning tokenizer warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Define parameters
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
    

    # language_model = SentenceTransformer('all-MiniLM-L6-v2')
    language_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    start_epoch = 1
    if args.resume:
        filenames = args.resume.split()
        for filename in filenames:
            if os.path.isfile(filename):
                print('\t==> loading checkpoint {}...'.format(filename))
                if args.cuda:
                    checkpoint = torch.load(filename)
                else:
                    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
                    
                
                # # removes 'module' from dict entries, pytorch bug
                # if torch.cuda.device_count() == 1 and any(k.startswith('module.') for k in checkpoint.keys()):
                #     checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
                # if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
                #     checkpoint = {'module.'+k: v for k,v in checkpoint.items()}
                
                # load correct model in right place
                if 'modelG' in filename:
                    # modelG.load_state_dict(checkpoint.state_dict())
                    modelG = checkpoint
                elif 'modelD' in filename:
                    # modelD.load_state_dict(checkpoint)
                    modelD = checkpoint
                else:
                    print('Wrong model files')
                    return
                
                print('\t==> loaded checkpoint {}\n'.format(filename))
        
        start_epoch = int(re.match(r'.*epoch_(\d+).pth', args.resume).groups()[0]) + 1
        print('\nFinished loading checkpoints. Starting from epoch {}\n\n!'.format(start_epoch))
    else:
        if 'original' in args.model.lower():
            modelG = gimli.generator()
            modelD = gimli.discriminator()
        elif 'v2' in args.model.lower():
            modelG = gimli_v2.generator()
            modelD = gimli_v2.discriminator()
        elif 'attention' in args.model.lower():
            modelG = gimli_with_attention.generator()
            modelD = gimli_with_attention.discriminator()
        # print(modelG)
        # print(modelD)
    
    if torch.cuda.device_count() > 0 and args.cuda:
        modelG = torch.nn.DataParallel(modelG)
        modelD = torch.nn.DataParallel(modelD)
        modelG.module.cuda()
        modelD.module.cuda()
    
    if args.cuda:
        modelG.cuda()
        modelD.cuda()
    
    if not args.resume:
        modelG.apply(gimli_v2.weights_init)
        modelD.apply(gimli_v2.weights_init)
                
    progress_bar = trange(start_epoch, args.epochs + 1)
    t_progress_bar = trange(start_epoch, args.epochs + 1)
    v_progress_bar = trange(start_epoch, args.epochs + 1)
    
    optimizerG = torch.optim.Adamax(modelG.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizerD = torch.optim.Adamax(modelD.parameters(), lr=args.lr, weight_decay=1e-5)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step, gamma=args.lr_gamma)
    schedulerG.last_epoch = start_epoch
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, args.lr_step, gamma=args.lr_gamma)
    schedulerD.last_epoch = start_epoch
    g_loss_dict = {}
    d_loss_dict = {}
    val_g_loss_dict = {}
    val_d_loss_dict = {}
    test_loss_dict = {}
    start_time = time()
    print('\nTraining ({} epochs) is starting...'.format(args.epochs))
    for epoch in progress_bar:
        
        print('\n\tCurrent learning rate: {}'.format(optimizerD.param_groups[0]['lr']))

        # TRAIN
        progress_bar.set_description('TRAIN')
        train(train_dataloader, modelG, modelD, language_model, optimizerG, optimizerD, epoch, args, g_loss_dict, d_loss_dict)

        # VALIDATE
        v_progress_bar.set_description('VAL')
        validate(val_dataloader, modelG, modelD, language_model, epoch, args, val_g_loss_dict, val_d_loss_dict)
    
        # TEST
        t_progress_bar.set_description('TEST')
        test(test_dataloader, modelG, language_model, epoch, args, test_loss_dict)


        if epoch % 10 == 0:
            torch.save(modelG, './modelG_epoch_{}.pth'.format(epoch))
            torch.save(modelD, './modelD_epoch_{}.pth'.format(epoch))
                    
    total_time = time() - start_time
    print('\nTotal time elapsed:\n\t{}\n'.format(total_time))
    with open('./Generator_train_loss_{}_epochs.pkl'.format(args.epochs), 'wb') as file:
        dump(g_loss_dict, file)
    with open('./Discriminator_train_loss_{}_epochs.pkl'.format(args.epochs), 'wb') as file:
        dump(d_loss_dict, file)
    with open('./Generator_val_loss_{}_epochs.pkl'.format(args.epochs), 'wb') as file:
        dump(val_g_loss_dict, file)
    with open('./Discriminator_val_loss_{}_epochs.pkl'.format(args.epochs), 'wb') as file:
        dump(val_d_loss_dict, file)
    with open('./GAN_test_loss_{}_epochs.pkl'.format(args.epochs), 'wb') as file:
        dump(test_loss_dict, file)
    writer.flush()
    writer.close()

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
                        help='resume from model stored. modelG first, separate with blank space')
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
