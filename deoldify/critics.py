from fastai.basic_train import Learner
from fastai.core import *
from fastai.layers import NormType, conv_layer
from fastai.torch_core import *
from fastai.vision import *
from fastai.vision.data import ImageDataBunch
from fastai.vision.gan import AdaptiveLoss, accuracy_thresh_expand
"Critic to train a `GAN`."
'''
定义一个生成对抗网络的判别器，使用一系列卷积层和丢弃层构建一个深度神经网络，用于判别输入图像是否经过正确的着色处理
'''

'''
定义一个字典变量，其中包含了卷积层的相关参数，
其中 leaky表示使用LeakyReLU 作为激活函数，其中的斜率为 0.2
norm_type=NormType.Spectral 表示使用谱归一化（spectral normalization）进行权重归一化
'''
_conv_args = dict(leaky=0.2, norm_type=NormType.Spectral)


# 创建一个卷积层

def _conv(ni: int, nf: int, ks: int = 3, stride: int = 1, **kwargs):
    return conv_layer(ni, nf, ks=ks, stride=stride, **_conv_args, **kwargs)

'''
GAN判别器生成函数
n_channels：图像通道参数 3个通道（彩色）
nf：卷积核数
n_blocks：卷积块数
'''

def custom_gan_critic(
    n_channels: int = 3, nf: int = 256, n_blocks: int = 3, p: int = 0.15
):
    layers = [_conv(n_channels, nf, ks=4, stride=2), nn.Dropout2d(p / 2)]
    for i in range(n_blocks):
        layers += [
            _conv(nf, nf, ks=3, stride=1),
            nn.Dropout2d(p),
            _conv(nf, nf * 2, ks=4, stride=2, self_attention=(i == 0)),
        ]
        nf *= 2
    layers += [
        _conv(nf, nf, ks=3, stride=1),
        _conv(nf, 1, ks=4, bias=False, padding=0, use_activ=False),
        Flatten(),
    ]
    return nn.Sequential(*layers)

'''
    函数 colorize_crit_learner 利用 Learner 类创建了一个用于训练鉴别器模型的学习器对象，
它接受数据集、鉴别器模型、损失函数、评价指标等参数，用于完成模型的训练和评估。
'''
def colorize_crit_learner(
    data: ImageDataBunch,
    loss_critic=AdaptiveLoss(nn.BCEWithLogitsLoss()),
    nf: int = 256,
) -> Learner:
    return Learner(
        data,
        custom_gan_critic(nf=nf),
        metrics=accuracy_thresh_expand,
        loss_func=loss_critic,
        wd=1e-3,
    )
