import numpy as np
import torch as tr
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Sequence
import torch

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * tr.log(input_ + epsilon)
    entropy = tr.sum(entropy, dim=1)
    return entropy

class CELabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)

        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class CELabelSmooth_raw(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth_raw, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = tr.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=-1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = tr.log_softmax(inputs, dim=1)
        labels = tr.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -tr.mean(loss)
        elif self.reduction == 'sum':
            outputs = -tr.sum(loss)
        else:
            outputs = -loss

        return outputs


class ConsistencyLoss(nn.Module):
    """
    Label consistency loss.
    """

    def __init__(self, num_select=2):
        super(ConsistencyLoss, self).__init__()
        self.num_select = num_select

    def forward(self, prob):
        dl = 0.
        count = 0
        for i in range(prob.shape[1] - 1):
            for j in range(i + 1, prob.shape[1]):
                dl += self.jensen_shanon(prob[:, i, :], prob[:, j, :], dim=1)
                count += 1
        return dl / count

    @staticmethod
    def jensen_shanon(pred1, pred2, dim):
        """
        Jensen-Shannon Divergence.
        """
        m = (tr.softmax(pred1, dim=dim) + tr.softmax(pred2, dim=dim)) / 2
        pred1 = F.log_softmax(pred1, dim=dim)
        pred2 = F.log_softmax(pred2, dim=dim)
        return (F.kl_div(pred1, m.detach(), reduction='batchmean') + F.kl_div(pred2, m.detach(),
                                                                              reduction='batchmean')) / 2

class source_inconsistency_loss(nn.Module):
    # source models inconsistency loss.
    def __init__(self, th_max=0.1):
        super(source_inconsistency_loss, self).__init__()
        self.th_max = th_max
    def forward(self, prob):  # [4, 8, 4]
        si_std = tr.std(prob, dim=1).mean(dim=1).mean(dim=0)
        return si_std

class InformationMaximizationLoss(nn.Module):
    """
    Information maximization loss.
    """

    def __init__(self):
        super(InformationMaximizationLoss, self).__init__()

    def forward(self, pred_prob, epsilon):
        softmax_out = nn.Softmax(dim=1)(pred_prob)
        ins_entropy_loss = tr.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        class_entropy_loss = tr.sum(-msoftmax * tr.log(msoftmax + epsilon))
        im_loss = ins_entropy_loss - class_entropy_loss

        return im_loss

# =============================================================DAN Function=============================================
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tr.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        features = tr.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss

def _update_index_matrix(batch_size: int, index_matrix: Optional[tr.Tensor] = None,
                         linear: Optional[bool] = True) -> tr.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = tr.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = tr.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: tr.Tensor) -> tr.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * tr.mean(l2_distance_square.detach())

        return tr.exp(-l2_distance_square / (2 * self.sigma_square))


# =============================================================CDANE Function===========================================
def CDANE(input_list, ad_net, random_layer, entropy=None, coeff=None, args=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is not None: 
        op_out = tr.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))

    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    '''
        domain label
    '''
    dc_target = tr.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if args.data_env != 'local':
        dc_target = dc_target.cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + tr.exp(-entropy)
        source_mask = tr.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = tr.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / tr.sum(source_weight).detach().item() + \
                 target_weight / tr.sum(target_weight).detach().item()
        return tr.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / tr.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, use_cuda=True):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        if use_cuda:
            self.random_matrix = [tr.randn(input_dim_list[i], output_dim).cuda() for i in range(self.input_num)]
        else:
            self.random_matrix = [tr.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [tr.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = tr.mul(return_tensor, single)
        return return_tensor
    '''
    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
    '''


# =============================================================MCC Function=============================================
class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self, t):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self, output: tr.Tensor) -> tr.Tensor:
        # output:  torch.Size([32, 2])
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)    
        entropy_weight = Entropy(softmax_out).detach()   
        entropy_weight = 1 + tr.exp(-entropy_weight)    
        entropy_weight = (n_sample * entropy_weight / tr.sum(entropy_weight)).unsqueeze(dim=1)   
        class_confusion_matrix = tr.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out) 
        class_confusion_matrix = class_confusion_matrix / tr.sum(class_confusion_matrix, dim=1)   
        mcc_loss = (tr.sum(class_confusion_matrix) - tr.trace(class_confusion_matrix)) / n_class   
        return mcc_loss


# =============================================================DSAN Function============================================
def lmmd(source, target, s_label, t_label, class_num, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = cal_weight(s_label, t_label, class_num=class_num)
    weight_ss = tr.from_numpy(weight_ss).cuda()
    weight_tt = tr.from_numpy(weight_tt).cuda()
    weight_st = tr.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = tr.Tensor([0]).cuda()
    if tr.sum(tr.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += tr.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    loss = loss / batch_size  # calculate the mean value or not
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = tr.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tr.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tr.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def convert_to_onehot(sca_label, class_num=2):

    return np.eye(class_num)[sca_label]


def cal_weight(s_label, t_label, class_num=None):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    t_vec_label = t_label.cpu().data.numpy()

    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    set_s = set(s_sca_label)
    set_t = set(t_sca_label)
    count = 0
    for i in range(class_num):
        if i in set_s and i in set_t:
            s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
            t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
            ss = np.dot(s_tvec, s_tvec.T)
            weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
            tt = np.dot(t_tvec, t_tvec.T)
            weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
            st = np.dot(s_tvec, t_tvec.T)
            weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
            count += 1

    length = count  # len( set_s ) * len( set_t )
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# =============================================================JAN Function=============================================
class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_
    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as
    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\
    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None
    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`
    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True, thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = tr.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = tr.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss
