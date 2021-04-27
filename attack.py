import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

import copy
import random
import numpy as np
import argparse
import pickle
from train import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # numpy
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)  # cpu
torch.cuda.manual_seed(RANDOM_SEED)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

cls_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def getAttackDataset(args):
    if args.correct1k:
        attack_dataset = correct1kDataset('./correct_1k.pkl')
        sample_idx = np.arange(0, 1000, 1)
    else: 
        test_y_pred, test_y_truth, test_acc = test(args)
        sample_idx = []
        for i in range(len(test_y_pred)):
            if test_y_pred[i] == test_y_truth[i]:
                sample_idx.append(i)
        test_dataset = datasets.FashionMNIST(root='./FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
        attack_dataset = Data.Subset(test_dataset, sample_idx[:1000])
    return attack_dataset, sample_idx[:1000]


def whiteBoxAttack(args, attack_dataset, sample_idx, save_attack_fig):
    attack_loader = Data.DataLoader(attack_dataset, batch_size=1, shuffle=False)

    if args.backbone == '3_conv':
        model = cnn_model(input_dim=1, output_dim=10).to(device)
        model_name = '3_conv_avgpool'
    elif args.backbone == 'resnet':
        model = ResNet18_Mnist(in_channel=1, num_classes=10).to(device)
        model_name = 'ResNet18'
    else:
        raise ValueError("parameter 'backbone' can only be strings '3_conv' or 'resnet'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CE_loss = nn.CrossEntropyLoss()

    if args.train_with_attacked_data:
        checkpoint = torch.load('./ckpt/best_augmented_'+model_name+'_model.ckpt', map_location=device)
    else:
        checkpoint = torch.load('./ckpt/best_'+model_name+'_model.ckpt', map_location=device)
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param in model.parameters():
            param.requires_grad = False

    success_idx = []
    attacked_dataset = []

    for idx, (attack_x, attack_y) in enumerate(attack_loader):
        original_x = Variable(attack_x, requires_grad=True).to(device)
        attack_x = original_x
        cur_y = attack_y.squeeze().cpu().detach().numpy()
        #print("Sample %d, Original class %d, Attack to predict class %d" % (idx, cur_y, (cur_y+1)%10))
        attack_y = torch.tensor((attack_y + 1) % 10).to(device)
        attack_noise = Variable(torch.zeros((1, 1 ,28, 28), dtype=torch.float), requires_grad=True).to(device)
        model.eval()

        for epoch in range(args.attack_epochs):
            attack_x = original_x + attack_noise
            y_pred = model(attack_x).to(device)
            loss = CE_loss(y_pred, attack_y)
            noise_grad = torch.autograd.grad(loss, attack_x)[0]
            pred_prob = y_pred[0, attack_y.squeeze().cpu().numpy()]
            #print(loss)
            if np.argmax(y_pred.cpu().detach().numpy()) != ((cur_y + 1) % 10):
                adj_noise = 0.2 * noise_grad / torch.max(torch.abs(noise_grad))
                adj_noise = torch.clamp(adj_noise, min=-0.015, max=0.015)
                attack_noise -= adj_noise
                # attack_noise = torch.clamp(attack_noise, min=-0.05, max=0.05)
                # delta_x = attack_x - original_x
                # delta_x = torch.clamp(delta_x, min=-0.05, max=0.05)
                # attack_x = original_x + delta_x
            else:
                success_idx.append(sample_idx[idx])
                print("Attack Success on Epoch %d! Sample %d, Original class %d, Attack to predict class %d" % (epoch, sample_idx[idx], cur_y, (cur_y+1)%10))
                if not save_attack_fig:
                    plt.figure(figsize=(24, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(original_x.squeeze().cpu().detach().numpy())
                    plt.title('Original: class %s' % cls_name[cur_y])

                    plt.subplot(1, 3, 2)
                    plt.imshow(attack_x.squeeze().cpu().detach().numpy())
                    plt.title('Attacked: class %s prob: %.4f' % (cls_name[(cur_y+1)%10], pred_prob))

                    plt.subplot(1, 3, 3)
                    plt.imshow(attack_noise.squeeze().cpu().detach().numpy())
                    plt.title('Noise Added')
                    if args.train_with_attacked_data:
                        plt.savefig('./whitebox_re-attack_res/'+str(sample_idx[idx])+'_'+str(cur_y)+'_'+str((cur_y+1)%10)+'.jpg')
                    else:
                        plt.savefig('./whitebox_attack_res/'+str(sample_idx[idx])+'_'+str(cur_y)+'_'+str((cur_y+1)%10)+'.jpg')
                else:
                    attacked_dataset.append((attack_x.cpu().detach(), cur_y))
                break

        if sample_idx[idx] not in success_idx:
            print('Fail to Attack! Sample %d, Original class %d, Attack to predict class %d' % (sample_idx[idx], cur_y, (cur_y+1)%10))
    
    f = open('./whiteboxAttackedData.pkl', 'wb')
    pickle.dump(attack_dataset, f)
    f.close()

    return success_idx, len(success_idx)/len(sample_idx)


def blackBoxAttack(args, attack_dataset, sample_idx):
    attack_loader = Data.DataLoader(attack_dataset, batch_size=1, shuffle=False)

    if args.backbone == '3_conv':
        model = cnn_model(input_dim=1, output_dim=10).to(device)
        model_name = '3_conv_avgpool'
    elif args.backbone == 'resnet':
        model = ResNet18_Mnist(in_channel=1, num_classes=10).to(device)
        model_name = 'ResNet18'
    else:
        raise ValueError("parameter 'backbone' can only be strings '3_conv' or 'resnet'.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CE_loss = nn.CrossEntropyLoss()

    if args.train_with_attacked_data:
        checkpoint = torch.load('./ckpt/best_augmented_'+model_name+'_model.ckpt', map_location=device)
    else:
        checkpoint = torch.load('./ckpt/best_'+model_name+'_model.ckpt', map_location=device)
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param in model.parameters():
            param.requires_grad = False

    success_idx = []

    for idx, (attack_x, attack_y) in enumerate(attack_loader):
        original_x = Variable(attack_x, requires_grad=False).to(device)
        attack_x = original_x
        cur_y = attack_y.squeeze().cpu().detach().numpy()
        #print("Sample %d, Original class %d, Attack to predict class %d" % (idx, cur_y, (cur_y+1)%10))
        attack_y = torch.tensor((attack_y + 1) % 10).to(device)
        attack_noise = Variable(torch.zeros((1, 1 ,28, 28), dtype=torch.float), requires_grad=True).to(device)
        model.eval()

        for epoch in range(args.attack_epochs):
            noise_alpha = []
            noise_xs = 0.05 * torch.randn((10, 28, 28)).to(device)
            for i in range(10):
                noise_x = noise_xs[i]
                new_attack_x = attack_x + noise_x.reshape(1,1,28,28)
                y_pred = model(new_attack_x).to(device)
                pred_prob = y_pred[0, attack_y.squeeze().cpu().numpy()]
                noise_alpha.append(pred_prob.squeeze().cpu().detach().numpy())

            # noise_alpha /= sum(noise_alpha)
            # for i in range(10):
            #     attack_noise = noise_alpha[i] * noise_xs[i]
            noise_idx = np.argmax(np.array(noise_alpha))
            attack_x = attack_x + noise_xs[noise_idx]
            y_pred = model(attack_x).to(device)
            loss = CE_loss(y_pred, attack_y)
            #print(noise_grad, loss)
            pred_prob = y_pred[0, attack_y.squeeze().cpu().numpy()]
            #print(pred_prob)
            if np.argmax(y_pred.cpu().detach().numpy()) == ((cur_y + 1) % 10):
                success_idx.append(sample_idx[idx])
                print("Attack Success on Epoch %d! Sample %d, Original class %d, Attack to predict class %d" % (epoch, sample_idx[idx], cur_y, (cur_y+1)%10))
                plt.figure(figsize=(24, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(original_x.squeeze().cpu().detach().numpy())
                plt.title('Original: class %s' % cls_name[cur_y])

                plt.subplot(1, 3, 2)
                plt.imshow(attack_x.squeeze().cpu().detach().numpy())
                plt.title('Attacked: class %s prob: %.4f' % (cls_name[(cur_y+1)%10], pred_prob))

                plt.subplot(1, 3, 3)
                plt.imshow((attack_x-original_x).squeeze().cpu().detach().numpy())
                plt.title('Noise Added')
                plt.show()
                if args.correct1k:
                    fn_str = '_correct1k'
                else:
                    fn_str = ''
                if args.train_with_attacked_data:
                    plt.savefig('./blackbox'+fn_str+'_re-attack_res/'+str(sample_idx[idx])+'_'+str(cur_y)+'_'+str((cur_y+1)%10)+'.jpg')
                else:
                    plt.savefig('./blackbox'+fn_str+'_attack_res/'+str(sample_idx[idx])+'_'+str(cur_y)+'_'+str((cur_y+1)%10)+'.jpg')
                break         

            else:
                attack_noise = 0.9*attack_noise + noise_xs[noise_idx]
                attack_noise = torch.clamp(attack_noise, min=-0.1, max=0.1)
                attack_x = original_x + attack_noise

        if idx not in success_idx:
            print('Fail to Attack! Sample %d, Original class %d, Attack to predict class %d' % (sample_idx[idx], cur_y, (cur_y+1)%10))
    
    return success_idx, len(success_idx)/len(sample_idx)


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()

    attack_dataset, sample_idx = getAttackDataset(args)
    if args.mode == 'whitebox':
        success_idx, suc_rate = whiteBoxAttack(args, attack_dataset, sample_idx, save_attack_fig=False)
    elif args.mode == 'blackbox':
        success_idx, suc_rate = blackBoxAttack(args, attack_dataset, sample_idx)
    else:
        raise ValueError("parameter 'mode' can only be strings 'whitebox' or 'blackbox'.")

    print("Total success rate: %.4f" % suc_rate)

        
