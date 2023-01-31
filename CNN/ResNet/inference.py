import torch
import matplotlib.pyplot as plt
import torchvision
import random


def predict(dataset, net, labels):
    if type(labels) != list:
        print("Invalid label.")
        pass
    to_img = torchvision.transforms.ToPILImage()
    net.eval()
    net.to('cpu')
    imgs_index = []
    imgs = []
    pred_labels = []
    accuracy = 0.0

    for i in range(20):
        imgs_index.append(random.randint(0, 2000))
    for item in imgs_index:
        pred_labels.append(torch.argmax(net(dataset[item][0].reshape(1, 1, 96, 96)), dim=1).item())
        imgs.append((to_img(dataset[item][0]), dataset[item][1]))
    fig, axs = plt.subplots(2, 10, figsize=(19.5, 5), dpi=100)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i][j].imshow(imgs[i * 10 + j][0])
            if pred_labels[i * 10 + j] == imgs[i * 10 + j][1]:
                axs[i][j].set_title(labels[pred_labels[i * 10 + j]])
                accuracy += 1
            else:
                axs[i][j].set_title(labels[pred_labels[i * 10 + j]], color='orange')
    print(f'accuracy rate:{accuracy / 20 * 100}%')
