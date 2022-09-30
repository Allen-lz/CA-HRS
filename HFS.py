from torch.nn import functional as F
import torch

def hfs(x, n, s, use_square_kernel=True):
    """
    :param x: input_tensor
    :param n: maximum number of kernel
    :param s: the scale of the largest kernel is 1 / s of the size of the input_tensor
    :param type: square kernel or proportional kernel
    :return: x * mask
    """
    b, c, h, w = x.size()
    # k_size = (1, 1)
    # -------------------------------------------------------------------------
    avg = x.mean(1)
    thr = avg.view(b, -1).sum(-1) / (h * w)
    mask = (avg > thr.unsqueeze(-1).unsqueeze(-1)).float()
    # -------------------------------------------------------------------------
    xs = list()
    if use_square_kernel:
        k_stride = max(1, min(h, w) // n - 1)
        k_size = min(h, w) // s
        for i in range(n):
            if k_size <= 1:
                break
            xs.append(F.avg_pool2d(x, kernel_size=k_size, stride=2))
            k_size = k_size - k_stride
    else:
        k_stride = (max(1, h // n - 1), max(1, w // n))

        k_size = [h // s, w // s]
        for i in range(n):
            if k_size[0] <= 1 or k_size[1] <= 1:
                break
            # print(k_size)
            xs.append(F.avg_pool2d(x, kernel_size=tuple(k_size), stride=2))
            k_size = [k_size[0] - k_stride[0], k_size[1] - k_stride[1]]

    for i, xi in enumerate(xs):
        avgi = xi.mean(1)
        _, hi, wi = avgi.size()
        thri = avgi.view(b, -1).sum(-1) / (hi * wi)
        maski = (avg > thri.unsqueeze(-1).unsqueeze(-1)).float()
        xs[i] = maski

    for maski in xs:
        mask += maski
    mask = mask + 1
    mask = torch.log10(mask) + 1
    # mask = F.sigmoid(mask)
    # mask = F.normalize(mask.view(b, -1), p=1, dim=1)
    # mask = mask.view(b, h, w)
    mask = mask.unsqueeze(1)
    return mask