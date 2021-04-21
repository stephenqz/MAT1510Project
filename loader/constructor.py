from torchvision import datasets
from torch.utils.data import DataLoader
from trans.transform import TransformConstructor
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler


def get_loader(self, phase, drop_last):
    # take care of fashionmnist/mnist ambiguity
    if self.dataset not in self.dataset_path:
        self.dataset_path = self.dataset_path+'/'+self.dataset
        
    func = getattr(datasets, self.dataset)
    
    constructor = TransformConstructor()
    transform = constructor.construct(phase, self.dataset, self)
    
    try:
        dataset = func(root=self.dataset_path,
                       train=(phase=='train'),
                       transform=transform,
                       download=True)
    except:
        pass
    
    try:
        dataset = func(root=self.dataset_path,
                       train=(phase=='train'),
                       transform=transform,
                       download=True,
                       classes=self.num_classes)
    except:
        pass
    
    try:
        dataset = func(root=self.dataset_path,
                       split=phase,
                       transform=transform,
                       download=True)
    except:
        pass
    
    batch_size = getattr(self, phase + '_batch_size')
        
    batch_samp = BatchSampler(RandomSampler(dataset),
                              batch_size,
                              drop_last)

    return DataLoader(dataset=dataset,
                      num_workers=self.threads,
                      pin_memory=True,
                      sampler=None,
                      shuffle=None,
                      drop_last=None,
                      batch_sampler=batch_samp)

