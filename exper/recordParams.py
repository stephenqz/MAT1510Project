import torch

def generateSubsets(self):
    paramsToKeep = {}
    with torch.no_grad():
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weightDimensions = list(module.weight.data.size())
                rowIndices = torch.unsqueeze(torch.randint(0, weightDimensions[0] - 1, (100,)), 1)
                colInices = torch.unsqueeze(torch.randint(0, weightDimensions[1] - 1, (100,)), 1)
                weightSub = torch.cat((rowIndices, colInices), -1)
                paramsToKeep[name + '.weight'] = weightSub

            elif isinstance(module, torch.nn.Conv2d):
                weightDimensions = list(module.weight.data.size())
                rowIndices = torch.unsqueeze(torch.randint(0, weightDimensions[0], (100,)), 1)
                colInices = torch.unsqueeze(torch.randint(0, weightDimensions[1], (100,)), 1)
                kernSizeOne = torch.unsqueeze(torch.randint(0, weightDimensions[2], (100,)), 1)
                kernSizeTwo = torch.unsqueeze(torch.randint(0, weightDimensions[3], (100,)), 1)
                weightSub = torch.cat((rowIndices, colInices, kernSizeOne, kernSizeTwo), -1)
                paramsToKeep[name + '.weight'] = weightSub

        return paramsToKeep


def recordParams(self, phase, epoch, prune_iter, iter, totalIters, param_results):
    if phase == "train":
        with torch.no_grad():
            for name in self.model.prunable_layer_names:
                weightsToRecord = self.paramsToKeep[name].cpu().numpy()
                weights = self.model.underlyingStateDict[name].clone().cpu().detach().numpy()
                # Record the weight values
                for i, row in enumerate(weightsToRecord):
                    if len(row) == 4:
                        stats = {'phase': phase,
                                 'dataset': self.dataset,
                                 'epoch': epoch,
                                 'iter': iter,
                                 'prune_iter': prune_iter,
                                 'iters': totalIters,
                                 'layer_name': name,
                                 'param_type': "weight",
                                 'param_name': str(list(row)),
                                 'param_val': str(weights[row[0]][row[1]][row[2]][row[3]])}
                        param_results.append(dict(self.__getstate__(), **stats))
                    elif len(row) == 2:
                        for i, row in enumerate(weightsToRecord):
                            stats = {'phase': phase,
                                     'dataset': self.dataset,
                                     'epoch': epoch,
                                     'iter': iter,
                                     'prune_iter': prune_iter,
                                     'iters': totalIters,
                                     'layer_name': name,
                                     'param_type': "weight",
                                     'param_name': str(list(row)),
                                     'param_val': str(weights[row[0]][row[1]])}
                            param_results.append(dict(self.__getstate__(), **stats))
                    else:
                        raise TypeError(
                            "There is an error when recording weights!"
                        )

                param_results.save()
                param_results.to_csv()