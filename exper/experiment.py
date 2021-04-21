import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
from os import makedirs
from net.network import Network
from utils.dump import DumpJSON
from utils import check_point as checkp
from loader.constructor import get_loader
from utils.lr_scheduler import MultiStepLR
from utils.accuracy import compute_accuracy
from utils.average_meter import AverageMeter
from exper.trackNC import trackNC

class Experiment:
    def __init__(self, opts):
        for key, value in opts.items():
            setattr(self, key, value)
    
        try:
            makedirs(self.training_results_path)
        except:
            pass
        
        # datasets and loaders
        self.train_loader = get_loader(self, 'train', drop_last=True)
        self.test_loader = get_loader(self, 'test', drop_last=False)

        # model
        self.model = Network().construct(self.net, self)

        # move model to device
        self.model.to(self.device)
        print("================")
        print(self.device)
        print("================")

        #layers to avoid pruning
        self.pruning_layers_to_ignore = []

        #Register hooks
        # Last Layer
        self.model.lastLayer.register_forward_hook(forwardHookLL)

        # loss
        func = getattr(nn, self.crit)
        self.criterion = func()

        self.totalEpochsRun = 1

    def resetOptimizer(self):
        func = getattr(optim, self.optim)
        self.optimizer = func(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.lr,
                              **self.optim_kwargs)

        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=self.lr_milestones,
                                        gamma=self.gamma)

    def run(self):
        # seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        results_path = self.training_results_path + '/' + 'results'
        results = DumpJSON(read_path=(results_path+'.json'),
                               write_path=(results_path+'.json'))

        self.resetOptimizer()
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # train
            results = self.run_epoch("train",
                                     self.totalEpochsRun,
                                     self.train_loader,
                                     results)

            # Save the model at these checkpoints
            if self.totalEpochsRun in self.save_checkpoints:
                checkp.save_check_point(self, self.totalEpochsRun)


            results = self.run_test_epoch(epoch,
                                          self.test_loader,
                                          results)

            results.save()
            results.to_csv()
            self.totalEpochsRun += 1

        results = self.run_test_epoch(epoch,
                                      self.test_loader,
                                      results)

        results.save()
        results.to_csv()

    def run_test_epoch(self,
                  epoch,
                  loader,
                  results):

        #Disable autograd when testing the model
        with torch.no_grad():
            # Rescale the weights
            if self.prune_technique == "ClassDropout":
                probabilityForAveraging.probabilities = []
                for layer_idx in range(len(self.model.get_imp_layers())):
                    probabilityForAveraging.probabilities.append(1-self.probability_list[layer_idx])
                averagingHandles = []
                for layer_idx, layer in enumerate(self.model.get_imp_layers()):
                    averagingHandles.append(
                        layer.register_forward_hook(createDropoutHooks(probabilityForAveraging.probabilities, layer_idx)))
            results = self.run_epoch("test", epoch, loader, results)
            # Delete the hooks
            if self.prune_technique == "ClassDropout":
                for hook in averagingHandles:
                    hook.remove()
            return results
            
            
    def run_epoch(self,
                  phase,
                  epoch,
                  loader,
                  results,
                  ):
        
        # average meters
        meters = {}
        for name in self.stats:
            meters[name] = AverageMeter()

        # switch phase
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise Exception('Phase must be train, test or analysis!')
        
        for iter, batch in enumerate(loader, 1):
            
            # input and target
            input   = batch[0]
            target  = batch[1]

            if not isinstance(target,torch.LongTensor):
                target = target.view(input.shape[0],-1).type(torch.LongTensor)
            
            input = input.to(self.device)
            target = target.to(self.device)

            # Register dropout hooks and generate class specific dropout
            if phase == "train":
                if self.prune_technique == "ClassDropout":
                    dropoutMasks.masks = []
                    for layer_idx in range(len(self.model.get_imp_layers())):
                        # Get layers output dimension
                        layerMasks = torch.bernoulli((1-self.probability_list[layer_idx]) * torch.ones(self.num_classes, self.model.getShapeOfImp()[layer_idx]))
                        # One hot encoding buffer that you create out of the loop and just keep reusing
                        onehot = torch.zeros(self.train_batch_size, self.num_classes).to(self.device)
                        unsqueezeTarget = torch.unsqueeze(target, 1).to(self.device)
                        onehot.scatter_(1, unsqueezeTarget, 1)
                        layerMasks = layerMasks.to(self.device)
                        dropoutMasks.masks.append(onehot@layerMasks)

                    dropoutHandles = []
                    for layer_idx, layer in enumerate(self.model.get_imp_layers()):
                        dropoutHandles.append(
                            layer.register_forward_hook(createDropoutHooks(dropoutMasks.masks, layer_idx)))

            # run model on input and compare estimated result to target
            est = self.model(input)
            loss = self.criterion(est, target)

            # compute gradient and do optimizer step
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            #Remove dropout hooks
            if phase == "train":
                if self.prune_technique == "ClassDropout":
                    for hook in dropoutHandles:
                        hook.remove()

            # record statistics
            for name in self.stats:
                if name == 'top1':
                    val = compute_accuracy(est, target.data, topk=(1,))[0][0].item()
                elif name == 'top5':
                    val = compute_accuracy(est, target.data, topk=(5,))[0][0].item()
                elif name == 'loss':
                    val = loss.item()
                
                meters[name].update(val, input.data.shape[0])

            # print statistics
            if phase == "train":
                output = '{}\t'                                                 \
                         'Network: {}\t'                                        \
                         'Dataset: {}\t'                                        \
                         'Prune Technique: {}\t'                               \
                         'Epoch: [{}/{}][{}/{}]\t'                              \
                         .format(phase.capitalize(),
                                 self.net,
                                 self.dataset,
                                 self.prune_technique,
                                 self.totalEpochsRun,
                                 self.epochs,
                                 iter,
                                 len(loader))

            elif phase == "test":
                output = '{}\t' \
                         'Network: {}\t' \
                         'Dataset: {}\t' \
                         'Prune Technique: {}\t' \
                         'Iter/Batch: [{}/{}]\t' \
                    .format(phase.capitalize(),
                            self.net,
                            self.dataset,
                            self.prune_technique,
                            iter,
                            len(loader))

            for name, meter in meters.items():
                output = output + '{}: {meter.val:.8f} ({meter.avg:.8f})\t' \
                    .format(name, meter=meter)

            print(output)
            sys.stdout.flush()

        if results is not None:
            if phase == "train" and self.totalEpochsRun in self.nc_list:
                results = trackNC(self, results, self.totalEpochsRun, self.device, loader, featuresLastLayer)

            stats = {'phase': phase,
                     'epoch': self.totalEpochsRun,
                     }
            for name, meter in meters.items():
                stats['avg_' + name] = meter.avg

            results.append(dict(self.__getstate__(), **stats))

            #Record L^P norms
            with torch.no_grad():
                if phase == "train":
                    for p in self.p_list:
                        runningSumOfNorms = 0
                        for name, module in self.model.named_modules():
                            if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear):
                                weightCopy = module.weight.data.clone().detach().cpu()
                                raisedWeights = torch.pow(torch.abs(weightCopy), p)
                                currNorm = torch.sum(raisedWeights).item()
                                runningSumOfNorms += currNorm
                                if name.startswith("model."):
                                    name = name[len("model."):]
                                lp_stats = {'phase': 'train',
                                            'epoch': self.totalEpochsRun,
                                            'Layer': name,
                                            'p': p,
                                            'pNormValue': currNorm,
                                            'pNormRatio' : currNorm/torch.numel(weightCopy)
                                }
                                results.append(dict(self.__getstate__(), **lp_stats))

                        lp_stats = {'phase': 'train',
                                    'epoch': self.totalEpochsRun,
                                    'Layer': 'Entire Model',
                                    'p': p,
                                    'pNormValue': runningSumOfNorms,
                                    'pNormRatio': runningSumOfNorms/self.model.totalNumberOfParams
                                    }
                        results.append(dict(self.__getstate__(), **lp_stats))

        return results
    
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = [
                      'train_transform',
                      'test_transform',
                      'loader_type',
                      'pytorch_dataset',
                      'dataset_path',
                      'im_size',
                      'padded_im_size',
                      'num_classes',
                      'input_ch',
                      'lr_milestones',
                      'threads',
                      'epc_seed',
                      'crit',
                      'stats',
                      'train_dump_file',
                      'train_loader',
                      'test_loader',
                      'model',
                      'init_model',
                      'criterion',
                      'optimizer',
                      'lr_scheduler',
                      'device',
                      'iters',
                      'seed',
                      'test_batch_size',
                      'gamma',
                      'forwardHandles',
                      'backwardHandles',
                      'paramsToKeep',
                      'totalEpochs',
                      'totalEpochsRun',
                      'epochs',
                      'optim_kwargs',
                      'optim',
                      'droprate',
                      'memory_efficient',
                      'resnet_type',
                      'pretrained',
                      'prune_strategy',
                      'pruning_layers_to_ignore',
                      'save_checkpoints',
                      'nc_list',
                      'checkSparsityMilestones',
                      'colIndices',
                      'decompCheckP',
                      'timesApplied',
                      'pruningSpecific',
                      'p_list',
                      'histo_list',
                      'hardRankRestriction',
                      'startIdx',
                      'desiredAmount',
                      'backwardsPruneThresh',
                      'backwardsCounter',
                      'hoyerHyper',
                      'percent',
                      'milestoneForHeat',
                      'listOfPermutations',
                      'whenBlock',
                      'probability_list'
                      ]
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state

# Neural Collapse Hooks

class featuresLastLayer:
    pass

def forwardHookLL(self, input, output):
    featuresLastLayer.value = input[0].clone()

# Dropout Forward Hooks
class dropoutMasks:
    pass

def createDropoutHooks(dropoutMaskContainer, layer_idx):
    def forwardHook(self, input, output):
        #Vectorize
        threeDSwitch = False
        batchSize = output.shape[0]
        channels = output.shape[1]
        if len(list(output.shape)) > 2:
            height = output.shape[2]
            width = output.shape[3]
            threeDSwitch = True
        # Apply mask
        output = output.view(batchSize, -1)
        output = dropoutMaskContainer[layer_idx] * output
        # Unvectorize
        if threeDSwitch:
            output = output.view(batchSize, channels, height, width)
        else:
            output = output.view(batchSize, channels)
        return output
    return forwardHook

#Dropout averaging hooks
class probabilityForAveraging:
    pass

def createAveragingHooks(probabilityContainer, layer_idx):
    def forwardHook(self, input, output):
        output = probabilityContainer[layer_idx] * output
        return output
    return forwardHook