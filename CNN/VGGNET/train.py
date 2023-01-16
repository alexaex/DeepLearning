import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import time
import torchinfo
import argparse
def evaluate_accuracy(net, val_iter, device=None):
    if not device:
        device = next(iter(net.parameters())).device
    net.eval()
    meteric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            pred = torch.argmax(pred, dim=1)
            meteric[0] += (pred == y).sum().item()
            meteric[1] += y.numel()
    return meteric[0] / meteric[1]


def train_loop(net, loss_fn, num_epochs, learning_rate, train_iter, val_iter, device):
    # transfer model to device
    net.to(device)
    device_name = ''
    # time cost
    timer = 0.0
    # print which device does the model run on
    if device == 'cpu':
        from win32com.client import GetObject
        root_winmgmts = GetObject("winmgmts:root\cimv2")
        cpus = root_winmgmts.ExecQuery("Select * from Win32_Processor")
        device_name = cpus[0].Name
        print('Running on:', cpus[0].Name)
    else:
        device_name = torch.cuda.get_device_name(device)
        print('Running on:', torch.cuda.get_device_name(device))

    # weights initialize
    def __init__weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    net.apply(__init__weight)

    # define optimizer

    # optimizer = optim.SGD(params=net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
    # define containers
    epochs = [k + 1 for k in range(num_epochs)]
    train_loss = []
    train_acc = []
    val_acc = []


    # start training

    for i in range(num_epochs):
        # define container to record the loss and accuracy rate on training set
        Metric = [0.0, 0.0, 0.0]
        # set train model
        net.train()
        for batch, (X, y) in enumerate(train_iter):
            start = time.time()
            X, y = X.to(device), y.to(device)

            pred = net(X)

            # calculate loss
            loss = loss_fn(pred, y).to(device)

            # grad set zero
            optimizer.zero_grad()

            # calculate grad and update parameters
            loss.backward()
            optimizer.step()
            end = time.time()

            timer += (end - start)

            with torch.no_grad():
                Metric[0] += loss.item() * X.shape[0]
                Metric[1] += (torch.argmax(pred, dim=1) == y).sum().item()
                Metric[2] += X.shape[0]

        test_acc = evaluate_accuracy(net, val_iter)
        train_loss.append(Metric[0] / Metric[2])
        train_acc.append(Metric[1] / Metric[2])
        val_acc.append(test_acc)

        print('=============================================================================')
        print(f'=============================epochs:{i + 1}========================================')
        print(
            f'train loss:{Metric[0] / Metric[2]:.3f},train acc:{(Metric[1] / Metric[2]) * 100:.2f}%, val_acc:{test_acc * 100:.2f}%')

    # print runtime
    print('time costs:', timer, 'running on:' + device_name)
    # Training Visualize
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 6), dpi=100)
    axs[0].plot(epochs, train_loss, color='royalblue')
    axs[0].set_title('Training loss')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('Training loss')
    axs[0].set_xticks([i for i in epochs])

    axs[1].plot(epochs, train_acc, color='orange')
    axs[1].set_title('Training accuracy rate')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('Training accuracy rate')
    axs[1].set_xticks([i for i in epochs])
    axs[1].set_yticks([i / 100 for i in range(0, 101, 10)])

    axs[2].plot(epochs, val_acc, color='red')
    axs[2].set_title('Test accuracy rate')
    axs[2].set_xlabel('epochs')
    axs[2].set_ylabel('Testing accuracy rate')
    axs[2].set_xticks([i for i in epochs])
    axs[2].set_yticks([i / 100 for i in range(0, 101, 10)])

    plt.savefig('Train.png')
    plt.show()

