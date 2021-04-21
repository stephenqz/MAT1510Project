import abc
import torch
import torch.nn as nn

class DropoutNet(abc.ABC, torch.nn.Module):
    """The base class used by all models in this codebase."""
    @property
    def prunable_layer_names(self):
        """A list of the names of Tensors of this model that are valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        """

        return [name + '.weight' for name, module in self.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]

    @property
    def linearWeightsNames(self):
        """A list of the names of Tensors of this model that are valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        """

        return [name + '.weight' for name, module in self.named_modules() if
                isinstance(module, torch.nn.modules.linear.Linear)]

    @property
    def totalNumberOfParams(self):
        with torch.no_grad():
            numOfParams = 0
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear):
                    numOfParams+= torch.numel(module.weight)
            return numOfParams

    @property
    def totalNumberOfLinearParams(self):
        with torch.no_grad():
            numOfParams = 0
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.modules.linear.Linear):
                    numOfParams += torch.numel(module.weight)
            return numOfParams

    @property
    def underlyingStateDict(self):
        return self.state_dict()

    @property
    def lastLayer(self):
        with torch.no_grad():
            return self.classifier

    @property
    def precedingLayers(self):
        with torch.no_grad():
            otherLayers =[]
            for param in self.parameters():
                if param is not self.lastLayer.weight:
                    otherLayers += param
            return otherLayers

    def computeSingularValues(self):
        """
        Computes the singular values of the weight matrix W and returns a dictionary
        """
        eigenvalues = torch.svd(self.lastLayer.weight, compute_uv=False)[1].clone().detach().cpu()
        return eigenvalues