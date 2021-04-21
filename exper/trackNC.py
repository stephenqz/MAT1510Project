import torch
from scipy.sparse.linalg import svds
import numpy as np

def trackNC(self, results, epoch, device, loader, features):

    C = self.num_classes
    model = self.model
    model.eval()
    classifier = self.model.lastLayer

    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    net_correct   = 0
    NCC_match_net = 0

    for computation in ['Mean','Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):
            data, target = data.to(device), target.to(device)

            output = model(data)
            h = features.value.data.view(data.shape[0],-1) # B CHW

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov
                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    if torch.numel(output[idxs,:]) != 0:
                        net_pred = torch.argmax(output[idxs,:], dim=1)
                        net_correct += sum(net_pred==target[idxs]).item()

                        # 2) agreement between prediction and nearest class center
                        NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                                  for i in range(h_c.shape[0])])
                        NCC_pred = torch.argmin(NCC_scores, dim=1)
                        NCC_match_net += sum(NCC_pred==net_pred).item()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
        elif computation == 'Cov':
            Sw /= sum(N)

    trainAccuracy = net_correct/sum(N)
    nccAccuracy = 1-NCC_match_net/sum(N)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, sigval, _ = svds(M.cpu().numpy(), k=C - 1)
    eigval = sigval ** 2 / C
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    wcv = np.trace(Sw @ inv_Sb)

    # avg norm
    W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    MCov = (torch.std(M_norms)/torch.mean(M_norms)).item()
    WCov = (torch.std(W_norms)/torch.mean(W_norms)).item()

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    selfDuality = (torch.linalg.norm((normalized_W - normalized_M)**2)).item()

    #Equiangularity
    WEquiang = coherence(W.T/W_norms, self.num_classes, self.device)
    MEquiang = coherence(M_/M_norms, self.num_classes, self.device)

    if results is not None:
        stats = {'phase': "train",
                 'epoch': epoch,
                 'trainAccuracy': trainAccuracy,
                 'nccAcc': nccAccuracy,
                 'NCCategory': 'weights',
                 'Self-Duality': selfDuality,
                 'Equinorm': WCov,
                 'wcv': float(wcv),
                 'Max Equiangularity': WEquiang
                 }

        results.append(dict(self.__getstate__(), **stats))

        stats = {'phase': "train",
                 'epoch': epoch,
                 'trainAccuracy': trainAccuracy,
                 'nccAcc': nccAccuracy,
                 'NCCategory': 'features',
                 'Equinorm': MCov,
                 'Max Equiangularity': MEquiang
                 }

        results.append(dict(self.__getstate__(), **stats))

    return results


def coherence(V, C, device):
    G = V.T @ V
    G += torch.ones((C, C), device=device) / (C - 1)
    G -= torch.diag(torch.diag(G))
    return torch.norm(G, 1).item() / (C * (C - 1))