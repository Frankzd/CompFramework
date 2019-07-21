import torch
import torchvision
import torchvision.transforms as transforms
def get_dataset(batch_size=128,n_worker=8,data_root='../../data'):
    cifar_tran_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    cifar_tran_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    print('=> Preparing data..')
    transform_train = transforms.Compose(cifar_tran_train)
    transform_test = transforms.Compose(cifar_tran_test)
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, pin_memory=True, sampler=None)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker, pin_memory=True)
    n_class = 10

    return train_loader, val_loader