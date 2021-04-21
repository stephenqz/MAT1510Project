def get_dataset_properties(dataset):
    if dataset == 'MNIST':
        return 28, 10, 1, (60000, 10000)
    elif dataset == 'FashionMNIST':
        return 28, 10, 1, (60000, 10000)
    elif dataset == 'CIFAR10':
        return 32, 10, 3, (50000, 10000)
    elif dataset == 'CIFAR100':
        return 32, 100, 3, (50000, 10000)
    elif dataset == 'STL10':
        return 96, 10, 3, (5000, 8000)
    elif dataset == 'SVHN':
        return 32, 10, 3, (73257, 26032)
    else:
        raise Exception('Dataset not found!')
        
        