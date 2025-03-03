import torch


def flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0=0.1, epsilon=1e-08):
    """Flexible kernel calculation as in MMDu."""
    Dxy = Pdist2(X, Y)
    Dxy_org = Pdist2(X_org, Y_org)
    L = 1
    Kxy = (1 - epsilon) * torch.exp(
        -((Dxy / sigma0) ** L) - Dxy_org / sigma
    ) + epsilon * torch.exp(-Dxy_org / sigma)
    return Kxy


def MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon=1e-08):
    """Compute the variance of the difference statistic MMDXY - MMDXZ."""
    """Referenced from: https://github.com/eugenium/MMD/blob/master/mmd.py"""
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    # Remove diagonal elements
    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    u_yy = torch.sum(Kyynd) * (1.0 / (n * (n - 1)))
    u_zz = torch.sum(Kzznd) * (1.0 / (r * (r - 1)))
    u_xy = torch.sum(Kxy) / (m * n)
    u_xz = torch.sum(Kxz) / (m * r)

    t1 = (1.0 / n**3) * torch.sum(Kyynd.T @ Kyynd) - u_yy**2
    t2 = (1.0 / (n**2 * m)) * torch.sum(Kxy.T @ Kxy) - u_xy**2
    t3 = (1.0 / (n * m**2)) * torch.sum(Kxy @ Kxy.T) - u_xy**2
    t4 = (1.0 / r**3) * torch.sum(Kzznd.T @ Kzznd) - u_zz**2
    t5 = (1.0 / (r * m**2)) * torch.sum(Kxz @ Kxz.T) - u_xz**2
    t6 = (1.0 / (r**2 * m)) * torch.sum(Kxz.T @ Kxz) - u_xz**2
    t7 = (1.0 / (n**2 * m)) * torch.sum(Kyynd @ Kxy.T) - u_yy * u_xy
    t8 = (1.0 / (n * m * r)) * torch.sum(Kxy.T @ Kxz) - u_xz * u_xy
    t9 = (1.0 / (r**2 * m)) * torch.sum(Kzznd @ Kxz.T) - u_zz * u_xz

    if type(epsilon) == torch.Tensor:
        epsilon_tensor = epsilon.clone().detach()
    else:
        epsilon_tensor = torch.tensor(epsilon, device=Kyy.device)
    zeta1 = torch.max(t1 + t2 + t3 + t4 + t5 + t6 - 2 * (t7 + t8 + t9), epsilon_tensor)
    zeta2 = torch.max(
        (1 / m / (m - 1)) * torch.sum((Kyynd - Kzznd - Kxy.T - Kxy + Kxz + Kxz.T) ** 2)
        - (u_yy - 2 * u_xy - (u_zz - 2 * u_xz)) ** 2,
        epsilon_tensor,
    )

    data = {
        "t1": t1.item(),
        "t2": t2.item(),
        "t3": t3.item(),
        "t4": t4.item(),
        "t5": t5.item(),
        "t6": t6.item(),
        "t7": t7.item(),
        "t8": t8.item(),
        "t9": t9.item(),
        "zeta1": zeta1.item(),
        "zeta2": zeta2.item(),
    }

    Var = (4 * (m - 2) / (m * (m - 1))) * zeta1
    Var_z2 = Var + (2.0 / (m * (m - 1))) * zeta2

    return Var, Var_z2, data


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


# MMD for three samples
def MMD_3_Sample_Test(
    ref_fea,
    fea_y,
    fea_z,
    ref_fea_org,
    fea_y_org,
    fea_z_org,
    sigma,
    sigma0,
    epsilon,
    alpha,
):
    """Run three-sample test (TST) using deep kernel kernel."""
    X = ref_fea.clone().detach()
    Y = fea_y.clone().detach()
    Z = fea_z.clone().detach()
    X_org = ref_fea_org.clone().detach()
    Y_org = fea_y_org.clone().detach()
    Z_org = fea_z_org.clone().detach()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon)

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    Diff_Var, _, _ = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon)

    u_yy = torch.sum(Kyynd) / (Y.shape[0] * (Y.shape[0] - 1))
    u_zz = torch.sum(Kzznd) / (Z.shape[0] * (Z.shape[0] - 1))
    u_xy = torch.sum(Kxy) / (X.shape[0] * Y.shape[0])
    u_xz = torch.sum(Kxz) / (X.shape[0] * Z.shape[0])

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)
    if Diff_Var.item() <= 0:
        Diff_Var = torch.max(epsilon, torch.tensor(1e-08))
    p_value = torch.distributions.Normal(0, 1).cdf(-t / torch.sqrt((Diff_Var)))
    t = t / torch.sqrt(Diff_Var)

    if p_value > alpha:
        h = 0
    else:
        h = 1

    return h, p_value.item(), t.item()
