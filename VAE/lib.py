import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def gaussian_nice_loglkhd(h,device):
    return - 0.5*torch.sum(torch.pow(h,2),dim=1) - h.size(1)*0.5*torch.log(torch.tensor(2*np.pi).to(device))


def log_reconstruction_loss(data, output, sigma, mask, device):
    """Expectation term in the VAE loss function (see above).
    :param data: dataset
    :param output: output of the decoder
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param mask: mask to apply to data and reconstructions
    :returns: (torch tensor)
    """
    recon = output * mask
    test = data.to(device) * mask
    test_minus_mu = test - recon
    log_p_x_given_z = - (test_minus_mu * test_minus_mu / (2 * sigma * sigma) ).sum(1)
    return -1 * log_p_x_given_z


def average_VAE_loss(mu_code1, log_sigma_code1, data, output, sigma, mask, device):
    """Compute average VAE loss over a dataset.
    :param mu_code1: latent mean embeddings
    :param log_sigma_code1: latent covariance matrix diagonal embeddings
    :param data: images
    :param output: output of net on the dataset
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param mask: mask to apply to data and reconstructions
    :param device: device to compute on
    :returns: (torch tensor) average VAE loss over images
    """
    # negative KL term in loss function (see explanation above)
    negative_KL = (torch.ones_like(mu_code1) + 2 * log_sigma_code1 - mu_code1 * mu_code1 - torch.exp(
      2 * log_sigma_code1)).sum(1) / 2

    # reconstruction loss term
    #print(data.shape, output.shape, mask.shape)
    log_reconst_loss = log_reconstruction_loss(data, output, sigma, mask, device)
    # optimize average loss value over data
    nkl= negative_KL.mean()
    lgr = log_reconst_loss.mean()
    average_loss = lgr - nkl
    #print('lgr, nkl:', lgr, nkl)

    return average_loss, lgr


def find_code(data, net, shift_transform, mu_code1, log_sigma_code1, shift_codes, cls_label, mask, sigma, device, iternum=50, lr=0.01, shifts=False):
    """Optimize the latent space embedding for data of a given class.  The
    parameters of the model are fixed and we run gradient descent through the
    embeddings to optimize the VAE loss.
    :param data: training data, all of the same class
    :param net: model to use, with fixed parameters
    :param shift_transform: shift transformer to apply to images after decoding
    :param mu_code1: latent space embedding of the mean
    :param log_sigma_code1: latent space log square root of the diagonal of the
        covariance matrix
    :param shift_codes: parameters that shift images after they have been decoded
    :param cls_label: integer - class label of the training data being supplied
    :param mask: mask to apply to the reconstructions when computing loss
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param device: device to execute on
    :param iternum: number of epochs used in optimization
    :param lr: learning rate to use for optimizing mu and sigma values
    :param shifts: whether to do shifting of images after decoding and
        backpropogate through shift codes
    :returns: optimized mu and sigma embeddings
    """

    # require gradients so we can backpropogate through these
    mu_code1.requires_grad_(True)
    log_sigma_code1.requires_grad_(True)

    # optimizers
    optimizer_mu = torch.optim.Adam(
        [mu_code1],
        lr=lr
    )
    optimizer_log_sigma = torch.optim.Adam(
        [log_sigma_code1],
        lr=lr
    )

    if shifts:
        # learning the shifts of images
        shift_codes.requires_grad_(True)
        optimizer_shifts = torch.optim.Adam(
            [shift_codes],
            lr=lr
        )

    for j in range(iternum):
        # zero out gradients
        optimizer_mu.zero_grad()
        optimizer_log_sigma.zero_grad()

        if shifts:
            optimizer_shifts.zero_grad()

        # sample from latent space
        eps = torch.randn_like(mu_code1)
        z = mu_code1 + torch.exp(log_sigma_code1) * eps
        # decode using pretrained decoder - possibly doing shifting
        output, _ = net(z, cls_label)

        # TODO this was rendered obsolete
        # if shifts:
        #     # apply shift to decoded images and reflatten
        #     output = output.view(-1, 50, 50, 3)
        #     output = output.permute(0, 3, 1, 2)
        #     output = shift_transform(output, shift_codes)
        #     # invert the reshaping
        #     output = output.permute(0, 2, 3, 1)
        #     output = torch.flatten(output, start_dim=1)

        # compute loss - shift only effects reconstruction, is not regularized
        # like the other latent codes
        average_loss, mse = average_VAE_loss(
            mu_code1,
            log_sigma_code1,
            data,
            output,
            sigma,
            mask,
            device
        )
        average_loss.backward(retain_graph=True)
        optimizer_mu.step()
        optimizer_log_sigma.step()

        if shifts:
            optimizer_shifts.step()

    return mu_code1, log_sigma_code1, shift_codes


def rgb_to_hsv(input, dv):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0], 1).to(dv)
    # if False: #'xla' not in device.type:
    #     h.to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]

    for i, id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (input[logi, id[0]] - input[logi, id[1]]) / df[logi] + s), 360)

    s = torch.zeros(input.shape[0], 1).to(dv)  #
    # if False: #'xla' not in device.type:
    #     s.to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100

    v = mx.reshape(input.shape[0], 1) * 100

    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)

    output = output.reshape(sh).transpose(1, 3)
    return output


def hsv_to_rgb(input, dv):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    hh = input[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None];
    v = input[:, 2][:, None]
    s = input[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));

    output = torch.zeros_like(input).to(dv)  # .to(device)
    # if False: #'xla' not in device.type:
    #     output.to(device)
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)

    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output


def deform_data(x_in, perturb, trans, s_factor, h_factor, embedd, dv):

    if perturb == 0:
        return x_in
    # t1=time.time()
    h = x_in.shape[2]
    w = x_in.shape[3]
    nn = x_in.shape[0]
    v = ((torch.rand(nn, 6) - .5) * perturb).to(dv)
    # v=(torch.rand(nn, 6) * perturb)+perturb/4.
    # vs=2*(torch.rand(nn,6)>.5)-1
    # v=v*vs
    rr = torch.zeros(nn, 6).to(dv)
    if not embedd:
        ii = torch.randperm(nn).to(dv)
        u = torch.zeros(nn, 6).to(dv)
        u[ii[0:nn // 2]] = v[ii[0:nn // 2]]
    else:
        u = v
    # Ammplify the shift part of the
    u[:, [2, 5]] *= 2.

    rr[:, [0, 4]] = 1
    if trans == 'shift':
        u[:, [0, 1, 3, 4]] = 0
        u[:, [2, 5]] = torch.tensor([perturb, 0])
    elif trans == 'scale':
        u[:, [1, 3]] = 0
    elif 'rotate' in trans:
        u[:, [0, 1, 3, 4]] *= 1.5
        ang = u[:, 0]
        v = torch.zeros(nn, 6)
        v[:, 0] = torch.cos(ang)
        v[:, 1] = -torch.sin(ang)
        v[:, 4] = torch.cos(ang)
        v[:, 3] = torch.sin(ang)
        s = torch.ones(nn)
        if 'scale' in trans:
            s = torch.exp(u[:, 1])
        u[:, [0, 1, 3, 4]] = v[:, [0, 1, 3, 4]] * s.reshape(-1, 1).expand(nn, 4)
        rr[:, [0, 4]] = 0
    theta = (u + rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [nn, 1, h, w], align_corners=True)
    x_out = F.grid_sample(x_in, grid, padding_mode='border', align_corners=True)

    if x_in.shape[1] == 3 and s_factor > 0:
        v = torch.rand(nn, 2).to(dv)
        vv = torch.pow(2, (v[:, 0] * s_factor - s_factor / 2)).reshape(nn, 1, 1)
        uu = ((v[:, 1] - .5) * h_factor).reshape(nn, 1, 1)
        x_out_hsv = rgb_to_hsv(x_out, dv)
        x_out_hsv[:, 1, :, :] = torch.clamp(x_out_hsv[:, 1, :, :] * vv, 0., 1.)
        x_out_hsv[:, 0, :, :] = torch.remainder(x_out_hsv[:, 0, :, :] + uu, 1.)
        x_out = hsv_to_rgb(x_out_hsv, dv)
    if trans != 'shift':
        ii = torch.where(torch.bernoulli(torch.ones(nn) * .5) == 1)
        for i in ii:
            x_out[i] = x_out[i].flip(3)

    # print('Def time',time.time()-t1)
    return x_out


def pane_plot(images, mask=None, pane_label="images", filename=None):
    """Plot a set of images."""
    fig, axs = plt.subplots(8, 10, figsize=(12, 10))

    n = 0
    for label in range(10):
      for j in range(8):
        fig.axes[j+8*label].get_xaxis().set_visible(False)
        fig.axes[j+8*label].get_yaxis().set_visible(False)
        # plot reconstuction
        reconstruction_to_plot = images[n]

        if mask is not None:
            reconstruction_to_plot = reconstruction_to_plot * mask

        axs[j, label].imshow(reconstruction_to_plot.reshape(50,50,3))
        #axs[j, label].set_title(str(label))
        n += 1

    print(pane_label)
    #fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)
    fig.show()


def random_batch_shift(x_in, lower_bound, upper_bound, dv):
    """Applies a random shift to a batch of images, where the shift
    is different for each image in the batch (unlike torchvision.RandomAffine).
    """
    n_batch = x_in.shape[0]
    # affine matrices
    u = torch.zeros(n_batch, 6).to(dv)
    # random shifts
    u[:, [2, 5]] = torch.FloatTensor(n_batch, 2).uniform_(
        lower_bound, upper_bound
    ).to(dv)

    # identity
    rr = torch.zeros(n_batch, 6).to(dv)
    rr[:, [0, 4]] = 1
    # height and width
    h = x_in.shape[2]
    w = x_in.shape[3]

    # apply the shift
    theta = (u + rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [n_batch, 1, h, w], align_corners=True)
    x_out = F.grid_sample(x_in, grid, padding_mode='zeros', align_corners=True)

    return x_out
