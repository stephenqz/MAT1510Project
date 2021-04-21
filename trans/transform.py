__metaclass__ = type

from torchvision import transforms

class TransformConstructor:
    def construct(self, phase, dataset_name, obj):
        targetClass = getattr(self, dataset_name)
        instance = targetClass(obj)
        
        if phase == 'train':
            return instance.train_transform()
        elif (phase == 'test') or (phase == 'analysis'):
            return instance.test_transform()
        else:
            raise Exception('Wrong phase!')
    
    class Dataset():
        def __init__(self, obj):
            self.obj = obj
            self.pad = int((self.obj.padded_im_size - self.obj.im_size)/2)
            self.mean = ()
            self.std = ()
            for _ in range(self.obj.input_ch):
                self.mean += (0.5,)
                self.std  += (0.5,)
                
        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
        
        def test_transform(self):
            return self.train_transform()
            
    
    class MNIST(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.MNIST, self).__init__(obj)
            self.mean = (0.1307,)
            self.std = (0.3081,)
    
    class FashionMNIST(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.FashionMNIST, self).__init__(obj)
            self.mean = (0.2860,)
            self.std = (0.3530,)

    class CIFAR10(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.CIFAR10, self).__init__(obj)
            self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            self.std = [x / 255 for x in [63.0, 62.1, 66.7]]
            
        def train_transform(self):

            return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
        
        def test_transform(self):
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

    class CIFAR100(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.CIFAR100, self).__init__(obj)
            self.mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            self.std = [x / 255 for x in [68.2, 65.4, 70.4]]

        def train_transform(self):

            return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
        
        def test_transform(self):
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
    
    class STL10(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.STL10, self).__init__(obj)
            self.mean = (0.4467, 0.4398, 0.4066)
            self.std = (0.2603, 0.2566, 0.2713)
            
        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
        
        def test_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
    
    class SVHN(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.SVHN, self).__init__(obj)
            self.mean = (0.4377, 0.4438, 0.4728)
            self.std = (0.1980, 0.2010, 0.1970)
            
        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
        
        def test_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
    
    