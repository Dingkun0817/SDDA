import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from tl.models.EEGNet import EEGNet_feature, EEGNet
from tl.models.FC import FC, FC_xy, FC_diff



def backbone_net(args, return_type='y'):
    netF = EEGNet_feature(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=4,
                        D=2,
                        F2=8,
                        dropoutRate=0.25,
                        norm_rate=0.5,
                        return_type=return_type)

    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    elif return_type == '001xy':
        print("Hi! it's 001xy!")
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


# dynamic change the weight of the domain-discriminator
def calc_coeff(iter_num, alpha=10.0, max_iter=10000.0):
    return float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)





class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(tr.tensor(1.) * init_weights)

    def forward(self, x):
        x = self.w * tr.ones((x.shape[0]), 1).cuda()
        x = tr.sigmoid(x)
        return x


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1
