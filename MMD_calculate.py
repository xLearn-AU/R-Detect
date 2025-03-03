import numpy as np
import torch


def L2_distance_get(x, y, value_factor=1):
    """compute the paired distance between x and y."""
    x_norm = ((x / value_factor) ** 2).sum(1).view(-1, 1)
    y_norm = ((y / value_factor) ** 2).sum(1).view(1, -1)
    Pdist = (
        x_norm
        + y_norm
        - 2.0 * torch.mm(x / value_factor, torch.transpose(y / value_factor, 0, 1))
    )
    Pdist[Pdist < 0] = 0
    return Pdist


def guassian_kernel(source, target, kernel_mul, kernel_num=10, fix_sigma=None):

    n_samples = int(source.size()[0]) + int(target.size()[0])
    L2_distance = L2_distance_get(source, target)

    # Calculate the bandwidth of each core in a multi-core
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        # bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
        assert not torch.isinf(bandwidth).any()
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    scale_factor = 0
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # bandwidth_list = [torch.clamp(bandwidth * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]

    # The formula of Gaussian kernel, exp( |x y|/bandwith)
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]

    return sum(kernel_val) / kernel_num  # Merge multiple cores together


def mmd(source, target, kernel_mul=2.0, kernel_num=10, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K ss matrix, source< >source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K st matrix, source< >target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K ts matrix,target< >source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K tt matrix,target< >target

    loss = (XX + XY).sum() + (YX + YY).sum()
    # loss = XY.sum()
    return loss
