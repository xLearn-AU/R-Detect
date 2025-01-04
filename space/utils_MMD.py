import torch


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist < 0] = 0
    return Pdist


def MMD_batch2(
    Fea,
    len_s,
    Fea_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_var_computed=True,
    use_1sample_U=True,
    coeff_xy=2,
):
    X = Fea[0:len_s, :]
    Y = Fea[len_s:, :]
    if is_smooth:
        X_org = Fea_org[0:len_s, :]
        Y_org = Fea_org[len_s:, :]
    L = 1  # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device)
    # Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y).transpose(0, 1)
    Kx = torch.exp(-Dxx / sigma0)
    Ky = torch.exp(-Dyy / sigma0)
    Kxy = torch.exp(-Dxy / sigma0)

    nx = Kx.shape[0]

    is_unbiased = False
    xx = torch.div((torch.sum(Kx)), (nx * nx))
    yy = Ky.reshape(-1)
    xy = torch.div(torch.sum(Kxy, dim=1), (nx))

    mmd2 = xx - 2 * xy + yy
    return mmd2
