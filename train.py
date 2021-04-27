import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import copy
import random
import numpy as np
import argparse
import pickle
from convModel import *
from ResNet18 import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # numpy
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)  # cpu
torch.cuda.manual_seed(RANDOM_SEED)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--mode", default="blackbox", type=str)
    parser.add_argument("--backbone", default='resnet', type=str)
    parser.add_argument("--attack_epochs", default=200, type=int)
    parser.add_argument("--train_with_attacked_data", action='store_true', default=False)
    parser.add_argument("--correct1k", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    return parser


class whiteboxAttackedDataset(Data.Dataset):
    def __init__(self, pkl_file):
        self.file_list = pickle.load(open(pkl_file, 'rb'))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        return self.file_list[index]


class correct1kDataset(Data.Dataset):
    def __init__(self, pkl_file):
        self.file_list = pickle.load(open(pkl_file, 'rb'))
        self.x_list = self.file_list[0]
        self.y_list = self.file_list[1]

    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, index):
        cur_x = torch.tensor((self.x_list[index] / 255.0), dtype=torch.float).to(device)
        cur_y = torch.tensor(self.y_list[index]).to(device)
        cur_y = torch.argmax(cur_y)
        return cur_x, cur_y


def train(args):
    full_dataset = datasets.FashionMNIST(root='./FashionMNIST/', train=True, download=True, transform=transforms.ToTensor())
    train_size = 50000
    dev_size = 10000
    train_dataset, dev_dataset = Data.random_split(full_dataset, [train_size, dev_size])
    if args.train_with_attacked_data:
        attacked_dataset = whiteboxAttackedDataset('./whiteboxAttackedData.pkl')
        train_dataset = Data.ConcatDataset([train_dataset, attacked_dataset])
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False)

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

    epoch_loss = []
    epoch_dev_loss = []
    best_dev_acc = 0
    for epoch in range(args.epochs):
        batch_loss = []
        dev_loss = []
        total_y_pred = []
        total_y_truth = []
        model.train()
        print('--------Training--------')
        for step, (train_x, train_y) in enumerate(train_loader):
            ## Training
            optimizer.zero_grad()
            train_x = torch.tensor(train_x).to(device)
            train_y = torch.tensor(train_y).to(device)
            total_y_truth.extend(train_y.cpu().detach().numpy().tolist())

            output = model(train_x).to(device)
            loss = CE_loss(output, train_y)
            _, pred_idx = torch.max(output, dim=1)
            total_y_pred.extend(pred_idx.cpu().detach().numpy().tolist())

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            batch_loss.append(loss.cpu().detach().numpy())
            if step % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f' % (epoch, step, loss.cpu().detach().numpy()))
            
        epoch_loss.append(np.mean(np.array(batch_loss)))
        total_y_pred = np.array(total_y_pred)
        total_y_truth = np.array(total_y_truth)
        # print(total_y_pred, total_y_truth)
        train_acc = sum([total_y_pred[i] == total_y_truth[i] for i in range(len(total_y_truth))]) / len(total_y_truth)
        print('Epoch %d Loss: %.4f, Acc: %.4f' % (epoch, epoch_loss[-1], train_acc))

        print('--------Validating--------')
        with torch.no_grad():
            model.eval()
            dev_y_pred = []
            dev_y_truth = []
            for step, (dev_x, dev_y) in enumerate(dev_loader):
                dev_x = torch.tensor(dev_x).to(device)
                dev_y = torch.tensor(dev_y).to(device)
                dev_y_truth.extend(dev_y.cpu().detach().numpy().tolist())

                output = model(dev_x).to(device)
                _, pred_idx = torch.max(output, dim=1)
                dev_y_pred.extend(pred_idx.cpu().detach().numpy().tolist())
                loss = CE_loss(output, dev_y)
                dev_loss.append(loss.cpu().detach().numpy())
                if step % 10 == 0:
                    print('Epoch %d Batch %d: Dev Loss = %.4f' % (epoch, step, np.mean(np.array(dev_loss))))
        
        epoch_dev_loss.append(np.mean(np.array(dev_loss)))
        dev_y_pred = np.array(dev_y_pred)
        dev_y_truth = np.array(dev_y_truth)
        # print(dev_y_pred, dev_y_truth)
        dev_acc = sum([dev_y_pred[i] == dev_y_truth[i] for i in range(len(dev_y_truth))]) / len(dev_y_truth)
        
        print('Epoch %d Dev Loss: %.4f, Dev Acc: %.4f' % (epoch, epoch_dev_loss[-1], dev_acc))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            if args.train_with_attacked_data:
                torch.save(state, './ckpt/best_augmented_'+model_name+'_model.ckpt')
            else:
                torch.save(state, './ckpt/best_'+model_name+'_model.ckpt')
            print('------------ Save best model - Acc: %.4f ------------'%dev_acc)  

        model.train()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(epoch_dev_loss)), epoch_dev_loss, c='red', label='dev')
    ax1.plot(range(len(epoch_loss)), epoch_loss, c='blue', label='train')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    plt.legend(loc='best')
    if args.train_with_attacked_data:
        plt.savefig('./augmented_'+model_name+'_loss.png')
    else:
        plt.savefig('./'+model_name+'loss.png')
    plt.show()


def test(args):
    if args.correct1k:
        test_dataset = correct1kDataset('./correct_1k.pkl')
    else:
        test_dataset = datasets.FashionMNIST(root='./FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

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

    print('--------Testing--------')
    test_loss = []
    with torch.no_grad():
        model.eval()
        test_y_pred = []
        test_y_truth = []
        for step, (test_x, test_y) in enumerate(test_loader):
            test_x = torch.tensor(test_x).to(device)
            test_y = torch.tensor(test_y).to(device)
            test_y_truth.extend(test_y.cpu().detach().numpy().tolist())

            output = model(test_x).to(device)
            _, pred_idx = torch.max(output, dim=1)
            test_y_pred.extend(pred_idx.cpu().detach().numpy().tolist())
            loss = CE_loss(output, test_y)
            test_loss.append(loss.cpu().detach().numpy())
            if step % 10 == 0:
                print('Batch %d: Test Loss = %.4f' % (step, np.mean(np.array(test_loss))))
    
    mean_test_loss = np.mean(np.array(test_loss))
    test_y_pred = np.array(test_y_pred)
    test_y_truth = np.array(test_y_truth)
    #print(test_y_pred, test_y_truth)
    test_acc = sum([test_y_pred[i] == test_y_truth[i] for i in range(len(test_y_truth))]) / len(test_y_truth)
    print("Mean Test Loss: %.4f Acc: %.4f" % (mean_test_loss, test_acc))
    return test_y_pred, test_y_truth, test_acc


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    if not args.test:
        train(args)
    test(args)

