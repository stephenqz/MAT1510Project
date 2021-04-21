import torch

from os import makedirs
from IPython import embed
from os.path import exists

def get_check_point_path(obj, epoch, idx=None, digits=None, prefix='', milestones=False):
    training_results_path = getattr(obj, prefix+'training_results_path')
    return training_results_path + '/' \
         + get_check_point_name(obj, epoch, prefix, idx, digits, save_milestones=milestones) + '.pth'

def save_check_point(obj, epoch, idx=None, milestones=False):
    if not exists(obj.training_results_path):
        makedirs(obj.training_results_path)
    
    path = get_check_point_path(obj, epoch, idx, milestones=milestones)
    
    if isinstance(obj.model, list):
        torch.save(obj.model[idx].state_dict(), path)
    else:
        torch.save(obj.model.state_dict(), path)

    # Store previously saved model name
    obj.prev_saved_model = path
    
    print('Checkpoint saved to {}'.format(path))

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def get_check_point_name(obj, epoch, prefix='', idx=None, digits=None, save_milestones=False):
    if hasattr(obj, prefix+'train_dataset'):
        train_dataset = getattr(obj, prefix+'train_dataset')
    else:
        train_dataset = obj.dataset
    
    net = getattr(obj, prefix+'net')
    if isinstance(net, list):
        net = net[idx]
    
    name = 'dataset=' + train_dataset + '-net=' + net
    
    if hasattr(obj, prefix+'lr'):
        lr = getattr(obj, prefix+'lr')
        try:
            # lr is a list
            if digits is None:
                name = name + '-lr=' + str(lr).replace(".", "p")
            else:
                # truncate digits and add wildcare
                tmp = [str(truncate(x, digits)).replace(".", "p").replace("e","*e")+'*' for x in lr]
                name = name + '-lr=[' + tmp[0]
                for x in tmp[1:]:
                    name = name + ',' + x
                name = name + ']'
        except:
            # lr is a float
            if digits is None:
                name = name + '-lr=' + str(lr).replace(".", "p")
            else:
                # truncate digits and add wildcare
                name = name + '-lr=' + str(truncate(lr, digits)).replace(".", "p").replace("e","*e")+'*'

    if hasattr(obj, prefix+'prune_technique'):
        prune_technique = getattr(obj, prefix+'prune_technique')
        name = name + '-prune_technique=' + prune_technique

    if hasattr(obj, prefix+'normalizationStrat'):
        normalizationStrat = getattr(obj, prefix+'normalizationStrat')
        name = name + '-normalizationStrat=' + normalizationStrat

    if hasattr(obj, prefix +'desiredAmount'):
        desiredAmount = getattr(obj, prefix +  'desiredAmount')
        name = name + '-sparsity=' + str(desiredAmount).replace(".", "p")
            
    if hasattr(obj, prefix+'examples_per_class'):
        examples_per_class = getattr(obj, prefix+'examples_per_class')
        if not isinstance(examples_per_class, int):
            examples_per_class = None
        name = name + '-examples_per_class=' + str(examples_per_class)
    
    if hasattr(obj, prefix+'num_classes'):
        num_classes = getattr(obj, prefix+'num_classes')
        name = name + '-num_classes=' + str(num_classes)
    
    if hasattr(obj, prefix+'examples_per_class'):
        if isinstance(examples_per_class, int):
            if hasattr(obj,'epc_seed'):
                epc_seed = getattr(obj, prefix+'epc_seed')
                name = name + '-epc_seed=' + str(epc_seed)
    
    if hasattr(obj, prefix+'bootstrap_seed'):
        bootstrap_seed = getattr(obj, prefix+'bootstrap_seed')
        name = name + '-bootstrap_seed=' + str(bootstrap_seed)
        
    if hasattr(obj, prefix+'train_seed'):
        train_seed = getattr(obj, prefix+'train_seed')
        name = name + '-train_seed=' + str(train_seed)
        
    if hasattr(obj, prefix+'num_layers'):
        num_layers = getattr(obj, prefix+'num_layers')
        name = name + '-num_layers=' + str(num_layers)

    if hasattr(obj, prefix+'net_width'): # absolute width
        net_width = getattr(obj, prefix+'net_width')
        name = name + '-net_width=' + str(net_width)
        
    if hasattr(obj, prefix+'forward_class'):
        if obj.forward_class:
            forward_class = getattr(obj, prefix+'forward_class')
            name = name + '-forward_class=' + forward_class

    if hasattr(obj, prefix+'width') and obj.width: # multiplicative factor for width
        width = getattr(obj, prefix+'width')
        name = name + '-width=' + str(width)

    if save_milestones:
        milestones = getattr(obj, prefix + 'milestones')
        name = name + '-milestones=' + str(milestones)

    if hasattr(obj, 'save_weight_decay') and obj.save_weight_decay:
        if hasattr(obj,'optim_kwargs'):
            weight_decay = obj.optim_kwargs[prefix + 'weight_decay']
        elif hasattr(obj,'weight_decay'):
            weight_decay = getattr(obj, prefix+'weight_decay')
        else:
            raise Exception('Can not find weight decay parameter')

        name = name + '-weight_decay=' + str(weight_decay)

    if epoch is not None:
        name = name + '-epoch=' + str(epoch)
    
    return name

def get_optim_name(obj, epoch, idx=None, prefix=''):
    net = getattr(obj, prefix+'net')
    if isinstance(net, list):
        net = net[idx]
        
    return obj.training_results_path    \
           + '/optim'                   \
           + '-net='+net                \
           + '-epoch='+str(epoch)       \
           + '.pth'

