import os
import numpy as np
import torch
import scipy as sp

# import torchvision # use it for torch.utils.data
# import freqopttest.data as data
# import freqopttest.tst as tst
# import scipy.stats as stats
# import pdb

is_cuda = True


# class ModelLatentF(torch.nn.Module):
# 	"""define deep networks."""
# 	def __init__(self, x_in, H, x_out):
# 		"""Init latent features."""
# 		super(ModelLatentF, self).__init__()
# 		self.restored = False

# 		self.latent = torch.nn.Sequential(
# 			torch.nn.Linear(x_in, H, bias=True),
# 			torch.nn.Softplus(),
# 			torch.nn.Linear(H, H, bias=True),
# 			torch.nn.Softplus(),
# 			torch.nn.Linear(H, H, bias=True),
# 			torch.nn.Softplus(),
# 			torch.nn.Linear(H, x_out, bias=True),
# 		)
# 	def forward(self, input):
# 		"""Forward the LeNet."""
# 		fealant = self.latent(input)
# 		return fealant


def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x


def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x


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


def h1_mean_var_gram(
    Kx,
    Ky,
    Kxy,
    is_var_computed,
    use_1sample_U=True,
    is_unbiased=True,
    coeff_xy=2,
    is_yy_zero=False,
    is_xx_zero=False,
):
    """compute value of MMD and std of MMD using kernel matrix."""
    if not is_yy_zero:
        coeff_yy = 1
    else:
        coeff_yy = 0
    if not is_xx_zero:
        coeff_xx = 1
    else:
        coeff_xx = 0
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]

    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div(
                (torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1))
            )
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx * coeff_xx + Ky * coeff_yy - (Kxy + Kxy.transpose(0, 1)) * coeff_xy / 2
    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4 * (V1 - V2**2)
    if varEst == 0.0:
        print("error_var!!" + str(V1))
    return mmd2, varEst, Kxyxy

    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""


def MMDu(
    Fea,
    len_s,
    Fea_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_smooth=True,
    is_var_computed=True,
    use_1sample_U=True,
    is_unbiased=True,
    coeff_xy=2,
    is_yy_zero=False,
    is_xx_zero=False,
):
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :]  # fetch the original sample 1
    Y_org = Fea_org[len_s:, :]  # fetch the original sample 2
    L = 1  # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1 - epsilon) * torch.exp(
            -((Dxx / sigma0) ** L) - Dxx_org / sigma
        ) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1 - epsilon) * torch.exp(
            -((Dyy / sigma0) ** L) - Dyy_org / sigma
        ) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1 - epsilon) * torch.exp(
            -((Dxy / sigma0) ** L) - Dxy_org / sigma
        ) + epsilon * torch.exp(-Dxy_org / sigma)
    # Kx = (1-epsilon) * (-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * (-Dxx_org / sigma)
    # Ky = (1-epsilon) * (-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * (-Dyy_org / sigma)
    # Kxy = (1-epsilon) * (-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * (-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(
        Kx,
        Ky,
        Kxy,
        is_var_computed,
        use_1sample_U,
        is_unbiased,
        coeff_xy,
        is_yy_zero,
        is_xx_zero,
    )


def MMDu_L2(Fea, len_s):
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    return torch.mean(Dxx), torch.mean(Dyy), torch.mean(Dxy)


def MMD_batch(
    Fea,
    len_s,
    Fea_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_smooth=True,
    is_var_computed=True,
    use_1sample_U=True,
):
    X = Fea_org[0:len_s, :]
    Y = Fea_org[len_s:, :]
    X_fea = Fea[0:len_s, :]
    Y_fea = Fea[len_s:, :]
    dis_vector = torch.zeros((Y.shape[0]), device=Fea.device)
    for i in range(Y.shape[0]):
        dis_vector[i] = MMDu(
            torch.cat((X_fea, Y_fea[[i]]), dim=0),
            len_s,
            torch.cat((X, Y[[i]]), dim=0).view(len_s + 1, -1),
            sigma,
            sigma0,
            epsilon,
            is_unbiased=False,
        )[0]
    return dis_vector


def MMD_batch2(
    Fea,
    len_s,
    Fea_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_smooth=True,
    is_var_computed=True,
    use_1sample_U=True,
    coeff_xy=2,
):
    X = Fea[0:len_s, :]  # 400,300
    Y = Fea[len_s:, :]  # 2000,300
    if is_smooth:
        X_org = Fea_org[0:len_s, :]  # 400,76800
        Y_org = Fea_org[len_s:, :]  # 2000,76800
    L = 1  # generalized Gaussian (if L>1)

    nx = X.shape[0]  # 400
    ny = Y.shape[0]  # 2000
    Dxx = Pdist2(X, X)  # 400,400
    Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device)  # 2000,1  0
    # Dyy = Pdist2(Y, Y) # 2000, 1  0
    Dxy = Pdist2(X, Y).transpose(0, 1)  # 400，2000 => 2000, 400
    if is_smooth:
        Dxx_org = Pdist2(X_org, X_org)
        Dyy_org = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device)  # 2000,1  0
        # Dyy_org = Pdist2(Y_org, Y_org) # 1，1  0
        Dxy_org = Pdist2(X_org, Y_org).transpose(0, 1)  # 400, 2000 => 2000, 400
    # K_Ix = torch.eye(nx).cuda() # Create the identity matrix
    # K_Iy = torch.eye(ny).cuda()

    if is_smooth:
        Kx = (1 - epsilon) * torch.exp(
            -((Dxx / sigma0) ** L) - Dxx_org / sigma
        ) + epsilon * torch.exp(
            -Dxx_org / sigma
        )  # 400,400
        Ky = (1 - epsilon) * torch.exp(
            -((Dyy / sigma0) ** L) - Dyy_org / sigma
        ) + epsilon * torch.exp(
            -Dyy_org / sigma
        )  # 2000,1 value 1
        Kxy = (1 - epsilon) * torch.exp(
            -((Dxy / sigma0) ** L) - Dxy_org / sigma
        ) + epsilon * torch.exp(
            -Dxy_org / sigma
        )  # 400,2000
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    nx = Kx.shape[0]  # 400
    # ny = 1 # 2000
    is_unbiased = False
    if 1:
        xx = torch.div((torch.sum(Kx)), (nx * nx))  # 400,400 => 1
        yy = Ky.reshape(-1)  # 2000
        # yy = torch.div((torch.sum(Ky)), (ny * ny)) # 2000,1 => 2000 1  [1]
        # one-sample U-statistic.

        xy = torch.div(torch.sum(Kxy, dim=1), (nx))  # 2000

        mmd2 = xx - 2 * xy + yy
    return mmd2


# def MMDu_linear_kernel(Fea, len_s, is_var_computed=True, use_1sample_U=True):
# 	"""compute value of (deep) lineaer-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
# 	try:
# 		X = Fea[0:len_s, :]
# 		Y = Fea[len_s:, :]
# 	except:
# 		X = Fea[0:len_s].unsqueeze(1)
# 		Y = Fea[len_s:].unsqueeze(1)

# 	Kx = X.mm(X.transpose(0,1))
# 	Ky = Y.mm(Y.transpose(0,1))
# 	Kxy = X.mm(Y.transpose(0,1))

# 	return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

# def C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,N_epoch,batch_size,device,dtype):
# 	"""Train a deep network for C2STs."""
# 	N = S.shape[0]
# 	if is_cuda:
# 		model_C2ST = ModelLatentF(x_in, H, x_out).cuda()
# 	else:
# 		model_C2ST = ModelLatentF(x_in, H, x_out)
# 	w_C2ST = torch.randn([x_out, 2]).to(device, dtype)
# 	b_C2ST = torch.randn([1, 2]).to(device, dtype)
# 	w_C2ST.requires_grad = True
# 	b_C2ST.requires_grad = True
# 	optimizer_C2ST = torch.optim.Adam(list(model_C2ST.parameters()) + [w_C2ST] + [b_C2ST], lr=learning_rate_C2ST)
# 	criterion = torch.nn.CrossEntropyLoss()
# 	f = torch.nn.Softmax()
# 	ind = np.random.choice(N, N, replace=False)
# 	tr_ind = ind[:np.int(np.ceil(N * 1))]
# 	te_ind = tr_ind
# 	dataset = torch.utils.data.TensorDataset(S[tr_ind, :], y[tr_ind])
# 	dataloader_C2ST = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 	len_dataloader = len(dataloader_C2ST)
# 	for epoch in range(N_epoch):
# 		data_iter = iter(dataloader_C2ST)
# 		tt = 0
# 		while tt < len_dataloader:
# 			# training model using source data
# 			data_source = data_iter.next()
# 			S_b, y_b = data_source
# 			output_b = model_C2ST(S_b).mm(w_C2ST) + b_C2ST
# 			loss_C2ST = criterion(output_b, y_b)
# 			optimizer_C2ST.zero_grad()
# 			loss_C2ST.backward(retain_graph=True)
# 			# Update sigma0 using gradient descent
# 			optimizer_C2ST.step()
# 			tt = tt + 1
# 		if epoch % 100 == 0:
# 			print(criterion(model_C2ST(S).mm(w_C2ST) + b_C2ST, y).item())

# 	output = f(model_C2ST(S[te_ind, :]).mm(w_C2ST) + b_C2ST)
# 	pred = output.max(1, keepdim=True)[1]
# 	STAT_C2ST = abs(pred[:N1].type(torch.FloatTensor).mean() - pred[N1:].type(torch.FloatTensor).mean())
# 	return pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST

# def gauss_kernel(X, test_locs, X_org, test_locs_org, sigma, sigma0, epsilon):
# 	"""compute a deep kernel matrix between a set of samples between test locations."""
# 	DXT = Pdist2(X, test_locs)
# 	DXT_org = Pdist2(X_org, test_locs_org)
# 	# Kx = torch.exp(-(DXT / sigma0))
# 	Kx = (1 - epsilon) * torch.exp(-(DXT / sigma0) - DXT_org / sigma) + epsilon * torch.exp(-DXT_org / sigma)
# 	return Kx

# def compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon):
# 	"""compute a deep kernel based ME statistic."""
# 	# if gwidth is None or gwidth <= 0:
# 	#     raise ValueError('require gaussian_width > 0. Was %s.' % (str(gwidth)))
# 	reg = 0#10**(-8)
# 	n = X.shape[0]
# 	J = T.shape[0]
# 	g = gauss_kernel(X, T, X_org, T_org, sigma, sigma0, epsilon)
# 	h = gauss_kernel(Y, T, Y_org, T_org, sigma, sigma0, epsilon)
# 	Z = g - h
# 	W = Z.mean(0)
# 	Sig = ((Z - W).transpose(1, 0)).mm((Z - W))
# 	if is_cuda:
# 		IJ = torch.eye(J).cuda()
# 	else:
# 		IJ = torch.eye(J)
# 	s = n*W.unsqueeze(0).mm(torch.solve(W.unsqueeze(1),Sig + reg*IJ)[0])
# 	return s

# def mmd2_permutations(K, n_X, permutations=200):
# 	"""
# 		Fast implementation of permutations using kernel matrix.
# 	"""
# 	K = torch.as_tensor(K)
# 	n = K.shape[0]
# 	assert K.shape[0] == K.shape[1]
# 	n_Y = n_X
# 	assert n == n_X + n_Y
# 	w_X = 1
# 	w_Y = -1
# 	ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
# 	ws[-1, :n_X] = w_X
# 	for i in range(permutations):
# 		ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
# 	biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
# 	if True:  # u-stat estimator
# 		# need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
# 		# first two are just trace, but last is harder:
# 		is_X = ws > 0
# 		X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
# 		Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
# 		del is_X, ws
# 		cross_terms = K.take(Y_inds * n + X_inds).sum(1)
# 		del X_inds, Y_inds
# 		ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
# 	est = ests[-1]
# 	rest = ests[:-1]
# 	p_val = (rest > est).float().mean()
# 	return est.item(), p_val.item(), rest

# def TST_MMD_adaptive_bandwidth(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
# 	"""run two-sample test (TST) using ordinary Gaussian kernel."""
# 	mmd_vector = np.zeros(N_per)
# 	TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, is_smooth=False)
# 	mmd_value = get_item(TEMP[0],is_cuda)
# 	Kxyxy = TEMP[2]
# 	count = 0
# 	nxy = Fea.shape[0]
# 	nx = N1
# 	for r in range(N_per):
# 		# print r
# 		ind = np.random.choice(nxy, nxy, replace=False)
# 		# divide into new X, Y
# 		indx = ind[:nx]
# 		# print(indx)
# 		indy = ind[nx:]
# 		Kx = Kxyxy[np.ix_(indx, indx)]
# 		# print(Kx)
# 		Ky = Kxyxy[np.ix_(indy, indy)]
# 		Kxy = Kxyxy[np.ix_(indx, indy)]
# 		TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
# 		mmd_vector[r] = TEMP[0]
# 		if mmd_vector[r] > mmd_value:
# 			count = count + 1
# 		if count > np.ceil(N_per * alpha):
# 			h = 0
# 			threshold = "NaN"
# 			break
# 		else:
# 			h = 1
# 	if h == 1:
# 		S_mmd_vector = np.sort(mmd_vector)
# 		#        print(np.int(np.ceil(N_per*alpha)))
# 		threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	return h, threshold, mmd_value.item()


def TST_MMD_u_old(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth, is_var_computed=False)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()


def TST_MMD_u(
    Fea,
    N_per,
    N1,
    Fea_org,
    sigma,
    sigma0,
    epsilon,
    alpha,
    is_smooth=True,
    is_yy_zero=False,
):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(
        Fea,
        N1,
        Fea_org,
        sigma,
        sigma0,
        epsilon,
        is_smooth,
        is_var_computed=False,
        is_yy_zero=is_yy_zero,
    )
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=N_per)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h, p_val, mmd_value_nn


def mmd2_permutations(K, n_X, permutations=500):
    """
    Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1

    # Initialize ws matrix
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X

    # Generate random permutations
    perm_indices = torch.rand((permutations, n), device=K.device).argsort(dim=1)
    perm_indices = perm_indices[:, :n_X]

    for i in range(permutations):
        ws[i, perm_indices[i]] = w_X

    # Compute biased estimates
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)

    if True:  # u-stat estimator
        # Calculate cross terms
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)

        del is_X, ws

        # Use advanced indexing to get cross terms efficiently
        cross_terms = K[Y_inds.unsqueeze(2), X_inds.unsqueeze(1)].sum(dim=(1, 2))

        del X_inds, Y_inds

        # Compute unbiased estimates
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))

    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()

    return est.item(), p_val.item(), rest


def TST_C2ST_L(pred_C2ST, N_per, N1, alpha):
    """Run C2ST-L."""
    # np.random.seed(seed=1102)
    # torch.manual_seed(1102)
    # torch.cuda.manual_seed(1102)
    N = pred_C2ST.shape[0]
    # f = torch.nn.Softmax()
    # output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    # pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(
        pred_C2ST[:N1, 1].type(torch.FloatTensor).mean()
        - pred_C2ST[N1:, 1].type(torch.FloatTensor).mean()
    )
    STAT_vector = np.zeros(N_per)

    # ind = np.random.choice(N, (N_per, N), replace=False) # error and use the following three lines

    # ind = np.random.rand(N_per, N)
    # ind_argsort = np.argsort(ind, axis=1)
    # ind = np.take_along_axis(ind_argsort, ind_argsort, axis=1)
    # ind = torch.tensor(ind)
    # ind_X = ind[:, :N1]
    # ind_Y = ind[:, N1:]
    # STAT_vector = np.abs(pred_C2ST[:,1][ind_X].type(torch.FloatTensor).mean(1) - pred_C2ST[:,1][ind_Y].type(torch.FloatTensor).mean(1))

    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(
            pred_C2ST[ind_X, 1].type(torch.FloatTensor).mean()
            - pred_C2ST[ind_Y, 1].type(torch.FloatTensor).mean()
        )

    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT


def TST_C2ST_S(pred_C2ST, N_per, N1, alpha):
    """Run C2ST-S."""
    # np.random.seed(seed=1102)
    # torch.manual_seed(1102)
    # torch.cuda.manual_seed(1102)
    N = pred_C2ST.shape[0]
    pred_C2ST[:, [0, 1]] = pred_C2ST[:, [1, 0]]
    # f = torch.nn.Softmax()
    # output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    pred_C2ST = pred_C2ST.max(1, keepdim=True)[1]
    STAT = abs(
        pred_C2ST[:N1].type(torch.FloatTensor).mean()
        - pred_C2ST[N1:].type(torch.FloatTensor).mean()
    )
    STAT_vector = np.zeros(N_per)

    # ind = np.random.choice(N, (N_per, N), replace=False) # error and use the following three lines

    # ind = np.random.rand(N_per, N)
    # ind_argsort = np.argsort(ind, axis=1)
    # ind = np.take_along_axis(ind_argsort, ind_argsort, axis=1)
    # ind = torch.tensor(ind)

    # ind_X = ind[:, :N1]
    # ind_Y = ind[:, N1:]
    # STAT_vector = np.abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean(1) - pred_C2ST[ind_Y].type(torch.FloatTensor).mean(1))

    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(
            pred_C2ST[ind_X].type(torch.FloatTensor).mean()
            - pred_C2ST[ind_Y].type(torch.FloatTensor).mean()
        )

    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT


# def TST_MMD_u_linear_kernel(Fea, N_per, N1, alpha,  device, dtype):
# 	"""run two-sample test (TST) using (deep) lineaer kernel kernel."""
# 	mmd_vector = np.zeros(N_per)
# 	TEMP = MMDu_linear_kernel(Fea, N1)
# 	mmd_value = get_item(TEMP[0], is_cuda)
# 	Kxyxy = TEMP[2]
# 	count = 0
# 	nxy = Fea.shape[0]
# 	nx = N1

# 	for r in range(N_per):
# 		# print r
# 		ind = np.random.choice(nxy, nxy, replace=False)
# 		# divide into new X, Y
# 		indx = ind[:nx]
# 		# print(indx)
# 		indy = ind[nx:]
# 		Kx = Kxyxy[np.ix_(indx, indx)]
# 		# print(Kx)
# 		Ky = Kxyxy[np.ix_(indy, indy)]
# 		Kxy = Kxyxy[np.ix_(indx, indy)]

# 		TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
# 		mmd_vector[r] = TEMP[0]
# 		if mmd_vector[r] > mmd_value:
# 			count = count + 1
# 		if count > np.ceil(N_per * alpha):
# 			h = 0
# 			threshold = "NaN"
# 			break
# 		else:
# 			h = 1
# 	if h == 1:
# 		S_mmd_vector = np.sort(mmd_vector)
# 		#        print(np.int(np.ceil(N_per*alpha)))
# 		threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	return h, threshold, mmd_value.item()

# def TST_C2ST(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
# 	"""run C2ST-S on non-image datasets."""
# 	np.random.seed(seed=1102)
# 	torch.manual_seed(1102)
# 	torch.cuda.manual_seed(1102)
# 	N = S.shape[0]
# 	f = torch.nn.Softmax()
# 	output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
# 	pred_C2ST = output.max(1, keepdim=True)[1]
# 	STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
# 	STAT_vector = np.zeros(N_per)
# 	for r in range(N_per):
# 		ind = np.random.choice(N, N, replace=False)
# 		# divide into new X, Y
# 		ind_X = ind[:N1]
# 		ind_Y = ind[N1:]
# 		# print(indx)
# 		STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
# 	S_vector = np.sort(STAT_vector)
# 	threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
# 	h = 0
# 	if STAT.item() > threshold:
# 		h = 1
# 	# if STAT.item() < threshold_lower:
# 	#     h = 1
# 	return h, threshold, STAT

# def TST_LCE(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST, device,dtype):
# 	"""run C2ST-L on non-image datasets."""
# 	np.random.seed(seed=1102)
# 	torch.manual_seed(1102)
# 	torch.cuda.manual_seed(1102)
# 	N = S.shape[0]
# 	f = torch.nn.Softmax()
# 	output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
# 	# pred_C2ST = output.max(1, keepdim=True)[1]
# 	STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
# 	STAT_vector = np.zeros(N_per)
# 	for r in range(N_per):
# 		ind = np.random.choice(N, N, replace=False)
# 		# divide into new X, Y
# 		ind_X = ind[:N1]
# 		ind_Y = ind[N1:]
# 		# print(indx)
# 		STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
# 	S_vector = np.sort(STAT_vector)
# 	threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
# 	h = 0
# 	if STAT.item() > threshold:
# 		h = 1
# 	return h, threshold, STAT

# def TST_ME(Fea, N1, alpha, is_train, test_locs, gwidth, J = 1, seed = 15):
# 	"""run ME test."""
# 	Fea = get_item(Fea,is_cuda)
# 	tst_data = data.TSTData(Fea[0:N1,:], Fea[N1:,:])
# 	h = 0
# 	if is_train:
# 		op = {
# 			'n_test_locs': J,  # number of test locations to optimize
# 			'max_iter': 300,  # maximum number of gradient ascent iterations
# 			'locs_step_size': 1.0,  # step size for the test locations (features)
# 			'gwidth_step_size': 0.1,  # step size for the Gaussian width
# 			'tol_fun': 1e-4,  # stop if the objective does not increase more than this.
# 			'seed': seed + 5,  # random seed
# 		}
# 		test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tst_data, alpha, **op)
# 		return test_locs, gwidth
# 	else:
# 		met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
# 		test_result = met_opt.perform_test(tst_data)
# 		if test_result['h0_rejected']:
# 			h = 1
# 		return h

# def TST_SCF(Fea, N1, alpha, is_train, test_freqs, gwidth, J = 1, seed = 15):
# 	"""run SCF test."""
# 	Fea = get_item(Fea,is_cuda)
# 	tst_data = data.TSTData(Fea[0:N1,:], Fea[N1:,:])
# 	h = 0
# 	if is_train:
# 		op = {'n_test_freqs': J, 'seed': seed, 'max_iter': 300,
# 			  'batch_proportion': 1.0, 'freqs_step_size': 0.1,
# 			  'gwidth_step_size': 0.01, 'tol_fun': 1e-4}
# 		test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tst_data, alpha, **op)
# 		return test_freqs, gwidth
# 	else:
# 		scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha=alpha)
# 		test_result = scf_opt.perform_test(tst_data)
# 		if test_result['h0_rejected']:
# 			h = 1
# 		return h

# def TST_C2ST_D(S,N1,N_per,alpha,discriminator,device,dtype):
# 	"""run C2ST-S on MNIST and CIFAR datasets."""
# 	np.random.seed(seed=1102)
# 	torch.manual_seed(1102)
# 	torch.cuda.manual_seed(1102)
# 	N = S.shape[0]
# 	f = torch.nn.Softmax()
# 	output = discriminator(S)
# 	pred_C2ST = output.max(1, keepdim=True)[1]
# 	STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
# 	STAT_vector = np.zeros(N_per)
# 	for r in range(N_per):
# 		ind = np.random.choice(N, N, replace=False)
# 		# divide into new X, Y
# 		ind_X = ind[:N1]
# 		ind_Y = ind[N1:]
# 		STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
# 	S_vector = np.sort(STAT_vector)
# 	threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
# 	h = 0
# 	if STAT.item() > threshold:
# 		h = 1
# 	return h, threshold, STAT

# def TST_LCE_D(S,N1,N_per,alpha,discriminator,device,dtype):
# 	"""run C2ST-L on MNIST and CIFAR datasets."""
# 	np.random.seed(seed=1102)
# 	torch.manual_seed(1102)
# 	torch.cuda.manual_seed(1102)
# 	N = S.shape[0]
# 	f = torch.nn.Softmax()
# 	output = discriminator(S)
# 	STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
# 	STAT_vector = np.zeros(N_per)
# 	for r in range(N_per):
# 		ind = np.random.choice(N, N, replace=False)
# 		# divide into new X, Y
# 		ind_X = ind[:N1]
# 		ind_Y = ind[N1:]
# 		# print(indx)
# 		STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
# 	S_vector = np.sort(STAT_vector)
# 	threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	h = 0
# 	if STAT.item() > threshold:
# 		h = 1
# 	return h, threshold, STAT

# def TST_ME_DK(X, Y, T, X_org, Y_org, T_org, alpha, sigma, sigma0, epsilon, flag_debug = False):
# 	"""run deep-kernel ME test (using chi^2 to confirm the threshold) on CIFAR datasets (this code does not work)."""
# 	J = T.shape[0]
# 	s = compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon)
# 	pvalue = stats.chi2.sf(s.item(), J)
# 	if pvalue<alpha:
# 		h = 1
# 	else:
# 		h = 0
# 	if flag_debug:
# 		pdb.set_trace()
# 	return h, pvalue, s

# def TST_ME_DK_per(X, Y, T, X_org, Y_org, T_org, alpha, sigma, sigma0, epsilon):
# 	"""run deep-kernel ME test (using permutations to confirm the threshold) on CIFAR datasets."""
# 	N_per = 100
# 	J = T.shape[0]
# 	s = compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon)
# 	Fea = torch.cat([X.cpu(), Y.cpu()], 0).cuda()
# 	Fea_org = torch.cat([X_org.cpu(), Y_org.cpu()], 0).cuda()
# 	N1 = X.shape[0]
# 	N = Fea.shape[0]
# 	STAT_vector = np.zeros(N_per)
# 	for r in range(N_per):
# 		ind = np.random.choice(N, N, replace=False)
# 		# divide into new X, Y
# 		ind_X = ind[:N1]
# 		ind_Y = ind[N1:]
# 		# print(indx)
# 		STAT_vector[r] = compute_ME_stat(Fea[ind_X,:], Fea[ind_Y,:], T, Fea_org[ind_X,:], Fea_org[ind_Y,:], T_org, sigma, sigma0, epsilon)
# 	S_vector = np.sort(STAT_vector)
# 	threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
# 	h = 0
# 	if s.item() > threshold:
# 		h = 1
# 	return h, threshold,


def flexible_kernel(
    X, Y, X_org, Y_org, sigma, sigma0=0.1, epsilon=1e-08, is_smooth=True
):
    """Flexible kernel calculation as in MMDu."""
    Dxy = Pdist2(X, Y)
    Dxy_org = Pdist2(X_org, Y_org)
    L = 1
    if is_smooth:
        Kxy = (1 - epsilon) * torch.exp(
            -((Dxy / sigma0) ** L) - Dxy_org / sigma
        ) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kxy = torch.exp(-Dxy / sigma0)
    return Kxy


def MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon=1e-08):
    """Compute the variance of the difference statistic MMDXY - MMDXZ."""
    """Referenced from: https://github.com/eugenium/MMD/blob/master/mmd.py"""
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    # Ensure the matrices are on the GPU
    Kyy = Kyy.cuda()
    Kzz = Kzz.cuda()
    Kxy = Kxy.cuda()
    Kxz = Kxz.cuda()

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


def MMD_Diff_Var_Baseline(Kyy, Kzz, Kxy, Kxz, block_size=1024):
    """
    Compute the variance of the difference statistic MMDXY-MMDXZ
    See http://arxiv.org/pdf/1511.04581.pdf Appendix for derivations
    """
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    def remove_diag_inplace(K, block_size):
        for i in range(0, K.shape[0], block_size):
            end_i = min(i + block_size, K.shape[0])
            K[i:end_i, i:end_i] -= torch.diag(torch.diag(K[i:end_i, i:end_i]))

    # Remove diagonal elements in-place to avoid creating a large copy of the matrix
    remove_diag_inplace(Kyy, block_size)
    remove_diag_inplace(Kzz, block_size)

    u_yy = torch.sum(Kyy) * (1.0 / (n * (n - 1)))
    u_zz = torch.sum(Kzz) * (1.0 / (r * (r - 1)))
    u_xy = torch.sum(Kxy) / (m * n)
    u_xz = torch.sum(Kxz) / (m * r)

    # Helper function to compute sum of dot products in blocks
    def block_sum_dot_product(A, B):
        total_sum = 0
        for i in range(0, A.shape[0], block_size):
            end_i = min(i + block_size, A.shape[0])
            for j in range(0, B.shape[1], block_size):
                end_j = min(j + block_size, B.shape[1])
                total_sum += torch.sum(torch.mm(A[i:end_i], B[:, j:end_j]))
        return total_sum

    # compute zeta1
    t1 = (1.0 / n**3) * block_sum_dot_product(Kyy.T, Kyy) - u_yy**2
    t2 = (1.0 / (n**2 * m)) * block_sum_dot_product(Kxy.T, Kxy) - u_xy**2
    t3 = (1.0 / (n * m**2)) * block_sum_dot_product(Kxy, Kxy.T) - u_xy**2
    t4 = (1.0 / r**3) * block_sum_dot_product(Kzz.T, Kzz) - u_zz**2
    t5 = (1.0 / (r * m**2)) * block_sum_dot_product(Kxz, Kxz.T) - u_xz**2
    t6 = (1.0 / (r**2 * m)) * block_sum_dot_product(Kxz.T, Kxz) - u_xz**2
    t7 = (1.0 / (n**2 * m)) * block_sum_dot_product(Kyy, Kxy.T) - u_yy * u_xy
    t8 = (1.0 / (n * m * r)) * block_sum_dot_product(Kxy.T, Kxz) - u_xz * u_xy
    t9 = (1.0 / (r**2 * m)) * block_sum_dot_product(Kzz, Kxz.T) - u_zz * u_xz

    zeta1 = t1 + t2 + t3 + t4 + t5 + t6 - 2.0 * (t7 + t8 + t9)

    zeta2 = (1 / m / (m - 1)) * torch.sum(
        (Kyy - Kzz - Kxy.T - Kxy + Kxz + Kxz.T) ** 2
    ) - (u_yy - 2.0 * u_xy - (u_zz - 2.0 * u_xz)) ** 2

    data = dict(
        {
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
    )

    Var = (4.0 * (m - 2) / (m * (m - 1))) * zeta1
    Var_z2 = Var + (2.0 / (m * (m - 1))) * zeta2

    return Var, Var_z2, data


def TST_MMD_u_3S(
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
    is_smooth=True,
):
    """Run three-sample test (TST) using deep kernel kernel."""
    X = ref_fea.clone().detach().cuda()
    Y = fea_y.clone().detach().cuda()
    Z = fea_z.clone().detach().cuda()
    X_org = ref_fea_org.clone().detach().cuda()
    Y_org = fea_y_org.clone().detach().cuda()
    Z_org = fea_z_org.clone().detach().cuda()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon, is_smooth)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon, is_smooth)

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


def TST_MMD_u_3S_AUROC(
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
    is_smooth=True,
):
    """Run three-sample test (TST) using deep kernel kernel."""
    X = ref_fea.clone().detach().cuda()
    Y = fea_y.clone().detach().cuda()
    Z = fea_z.clone().detach().cuda()
    X_org = ref_fea_org.clone().detach().cuda()
    Y_org = fea_y_org.clone().detach().cuda()
    Z_org = fea_z_org.clone().detach().cuda()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon, is_smooth)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon, is_smooth)

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    u_yy = torch.sum(Kyynd) / (Y.shape[0] * (Y.shape[0] - 1))
    u_zz = torch.sum(Kzznd) / (Z.shape[0] * (Z.shape[0] - 1))
    u_xy = torch.sum(Kxy) / (X.shape[0] * Y.shape[0])
    u_xz = torch.sum(Kxz) / (X.shape[0] * Z.shape[0])

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)
    return t.item()


def TST_MMD_u_3S_Permutation(
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
    num_permutations=1000,
    is_smooth=True,
    t_stat_division=False,
):
    """Run three-sample test (TST) using permutation test."""
    X = ref_fea.clone().detach().cuda()
    Y = fea_y.clone().detach().cuda()
    Z = fea_z.clone().detach().cuda()
    X_org = ref_fea_org.clone().detach().cuda()
    Y_org = fea_y_org.clone().detach().cuda()
    Z_org = fea_z_org.clone().detach().cuda()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon, is_smooth)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon, is_smooth)

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    if t_stat_division:
        Diff_Var, _, _ = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon)

    u_yy = torch.sum(Kyynd) / (Y.shape[0] * (Y.shape[0] - 1))
    u_zz = torch.sum(Kzznd) / (Z.shape[0] * (Z.shape[0] - 1))
    u_xy = torch.sum(Kxy) / (X.shape[0] * Y.shape[0])
    u_xz = torch.sum(Kxz) / (X.shape[0] * Z.shape[0])

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)
    if t_stat_division:
        if Diff_Var.item() <= 0:
            Diff_Var = torch.max(epsilon, torch.tensor(1e-08))
        t_original = t / torch.sqrt(Diff_Var)
    else:
        t_original = t

    perm_t_values = []

    for _ in range(num_permutations):
        combined = torch.cat((X, Y), dim=0)
        perm = torch.randperm(combined.size(0))
        X_perm = combined[perm[: X.size(0)]]
        Y_perm = combined[perm[X.size(0) :]]

        Kyy_perm = flexible_kernel(Y_perm, Y_perm, sigma, sigma0, epsilon, is_smooth)
        Kxy_perm = flexible_kernel(X_perm, Y_perm, sigma, sigma0, epsilon, is_smooth)

        Kyynd_perm = Kyy_perm - torch.diag(torch.diag(Kyy_perm))

        u_yy_perm = torch.sum(Kyynd_perm) / (Y_perm.shape[0] * (Y_perm.shape[0] - 1))
        u_xy_perm = torch.sum(Kxy_perm) / (X_perm.shape[0] * Y_perm.shape[0])

        t_perm = u_yy_perm - 2 * u_xy_perm - (u_zz - 2 * u_xz)
        if t_stat_division:
            t_perm = t_perm / torch.sqrt(Diff_Var)
        perm_t_values.append(t_perm)

    perm_t_values = torch.tensor(perm_t_values).to("cuda")
    p_value = (torch.sum((t_original.to("cuda") < perm_t_values).int()).item() + 1) / (
        num_permutations + 1
    )

    if p_value < alpha:
        h = 1
    else:
        h = 0

    return h, p_value, t_original.item()


def TST_MMD_u_3S_Permutation_Kernel(
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
    num_permutations=1000,
    is_smooth=True,
    t_stat_division=False,
):
    """Run three-sample test (TST) using permutation test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = ref_fea.clone().detach().cuda()
    Y = fea_y.clone().detach().cuda()
    Z = fea_z.clone().detach().cuda()
    X_org = ref_fea_org.clone().detach().cuda()
    Y_org = fea_y_org.clone().detach().cuda()
    Z_org = fea_z_org.clone().detach().cuda()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon, is_smooth)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon, is_smooth)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon, is_smooth)

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    if t_stat_division:
        Diff_Var, _, _ = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon)

    u_yy = torch.sum(Kyynd) / (Y.shape[0] * (Y.shape[0] - 1))
    u_zz = torch.sum(Kzznd) / (Z.shape[0] * (Z.shape[0] - 1))
    u_xy = torch.sum(Kxy) / (X.shape[0] * Y.shape[0])
    u_xz = torch.sum(Kxz) / (X.shape[0] * Z.shape[0])

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)
    if t_stat_division:
        if Diff_Var.item() <= 0:
            Diff_Var = torch.max(epsilon, torch.tensor(1e-08).to(device))
        t_original = t / torch.sqrt(Diff_Var)
    else:
        t_original = t

    # Combine all features
    combined = torch.cat((X, Y, Z), dim=0)
    n = combined.size(0)
    n_X = X.size(0)
    n_Y = Y.size(0)
    n_Z = Z.size(0)

    # Compute the combined kernel matrix
    K_combined = flexible_kernel(combined, combined, sigma, sigma0, epsilon, is_smooth)

    # Create weight matrices for permutations
    ws = torch.full(
        (num_permutations + 1, n), -1, dtype=K_combined.dtype, device=device
    )
    ws[:, :n_X] = 1
    for i in range(num_permutations):
        ws[i, torch.randperm(n)[:n_X]] = 1

    # Compute biased estimates for all permutations
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K_combined, ws)
    if t_stat_division:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(num_permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(num_permutations + 1, n_Y)
        cross_terms = K_combined.take(Y_inds * n + X_inds).sum(1)
        ests = (biased_ests - K_combined.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    else:
        ests = biased_ests

    est = ests[-1]
    rest = ests[:-1]
    p_value = (rest > est).float().mean().item()

    if p_value < alpha:
        h = 1
    else:
        h = 0

    return h, p_value, t_original.item()


def TST_MMD_u_3S_Baseline(
    X,
    Y,
    Z,
    sigma=-1,
    SelectSigma=2,
    alpha=0.05,
    return_sigma=False,
    sigma_path=None,
    no_median_heuristic=False,
):
    """Run three-sample test (TST) using basic Gaussian kernel."""
    if not no_median_heuristic and sigma < 0:
        # Similar heuristics
        if SelectSigma > 1:
            ## Try to load sigma when it's not training mode (return_sigma=False)
            if (
                sigma_path is not None
                and os.path.exists(sigma_path)
                and not return_sigma
            ):
                sigma = np.load(sigma_path)
                sigma = torch.tensor(sigma, device=X.device)
            else:
                siz = min(1000, X.shape[0])
                sigma1 = kernelwidthPair(X[:siz], Y[:siz])
                sigma2 = kernelwidthPair(X[:siz], Z[:siz])
                sigma = (sigma1 + sigma2) / 2.0
        ## Problematic therefore not adapted
        # else:
        #     siz = min(1000, X.shape[0] * 3)
        #     Zem = torch.cat((X[:siz // 3], Y[:siz // 3], Z[:siz // 3]), dim=0)
        #     sigma = kernelwidth(Zem)

        Kyy = grbf(Y, Y, sigma)
        Kzz = grbf(Z, Z, sigma)
        Kxy = grbf(X, Y, sigma)
        Kxz = grbf(X, Z, sigma)
    else:
        Kyy = simple_grbf(Y, Y, sigma)
        Kzz = simple_grbf(Z, Z, sigma)
        Kxy = simple_grbf(X, Y, sigma)
        Kxz = simple_grbf(X, Z, sigma)

    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    u_yy = torch.sum(Kyynd) * (1.0 / (n * (n - 1)))
    u_zz = torch.sum(Kzznd) * (1.0 / (r * (r - 1)))
    u_xy = torch.sum(Kxy) / (m * n)
    u_xz = torch.sum(Kxz) / (m * r)

    # Compute the test statistic
    t = u_yy - 2.0 * u_xy - (u_zz - 2.0 * u_xz)
    Diff_Var, Diff_Var_z2, data = MMD_Diff_Var_Baseline(Kyy, Kzz, Kxy, Kxz)
    if Diff_Var.item() <= 0:
        Diff_Var = torch.tensor(1e-08)

    p_value = torch.distributions.Normal(0, 1).cdf(-t / torch.sqrt(Diff_Var))

    t = t / torch.sqrt(Diff_Var)

    if p_value > alpha:
        h = 0
    else:
        h = 1

    if return_sigma:
        return h, p_value.item(), t.item(), sigma
    return h, p_value.item(), t.item()


def grbf(x1, x2, sigma, block_size=1024):
    """Calculates the Gaussian radial base function kernel with memory optimization."""
    if x1.ndim == 3:
        x1 = x1.reshape(-1, x1.shape[-1])
    if x2.ndim == 3:
        x2 = x2.reshape(-1, x2.shape[-1])

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    # Ensure the data is on the GPU
    x1 = x1.to("cuda")
    x2 = x2.to("cuda")

    # Initialize the kernel matrix
    K = torch.zeros((n, m), device="cuda")

    # Compute the kernel matrix in blocks
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, m, block_size):
            end_j = min(j + block_size, m)
            block = torch.cdist(x1[i:end_i], x2[j:end_j]) ** 2
            K[i:end_i, j:end_j] = torch.exp(-block / (2 * sigma**2))

    return K


def simple_grbf(x1, x2, sigma):
    """Simplified Gaussian radial basis function kernel."""
    # Reshape if necessary
    if x1.ndim == 3:
        x1 = x1.reshape(-1, x1.shape[-1])
    if x2.ndim == 3:
        x2 = x2.reshape(-1, x2.shape[-1])

    # Compute the squared Euclidean distance
    dist = torch.cdist(x1, x2) ** 2

    # Compute the Gaussian kernel
    K = torch.exp(-dist / (2 * sigma**2))

    return K


def kernelwidthPair(x1, x2, block_size=1024, chunk_size=1024 * 1024):
    """Implementation of the median heuristic with memory optimization on GPU."""
    if x1.ndim == 3:
        x1 = x1.reshape(-1, x1.shape[-1])
    if x2.ndim == 3:
        x2 = x2.reshape(-1, x2.shape[-1])

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    # Ensure the data is on the GPU
    x1 = x1.to("cuda")
    x2 = x2.to("cuda")

    # Initialize distance matrix in blocks
    h = torch.zeros((n, m), device="cuda")

    # Compute pairwise distances in blocks to save memory
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, m, block_size):
            end_j = min(j + block_size, m)
            block = torch.sum((x1[i:end_i, None, :] - x2[None, j:end_j, :]) ** 2, dim=2)
            h[i:end_i, j:end_j] = block

    # Flatten the distance matrix
    h_flat = h.flatten()

    # Split h_flat into chunks and filter non-zero distances
    num_chunks = (h_flat.numel() + chunk_size - 1) // chunk_size
    non_zero_distances = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, h_flat.numel())
        chunk = h_flat[start_idx:end_idx]
        non_zero_chunk = chunk[chunk > 0]
        non_zero_distances.append(non_zero_chunk)

    # Concatenate all non-zero distances
    non_zero_distances = torch.cat(non_zero_distances)

    # Compute the median of non-zero distances
    mdist = torch.median(non_zero_distances)

    sigma = torch.sqrt(mdist / 2.0)
    if sigma == 0:
        sigma = 1.0

    return sigma.item()
