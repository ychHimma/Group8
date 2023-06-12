from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.layers import NormType
from fastai.torch_core import SplitFuncOrIdxList, apply_init, to_device
from fastai.vision import *
from fastai.vision.learner import cnn_config, create_body
from torch import nn
from .unet import DynamicUnetWide, DynamicUnetDeep
from .dataset import *


'''
这个函数用于加载预训练的图像生成模型并返回一个Learner对象，用于在推理（inference）阶段生成图像。
具体来说，它使用了一个预定义的数据集，将其传递给一个生成器模型，该模型基于ResNet架构，并使用L1损失函数进行训练，最终加载预训练权重。
该函数的参数包括预训练模型的权重名称、存储权重的文件夹路径、用于增加特征通道的因子和模型架构。
'''
def gen_inference_wide(
    root_folder: Path, weights_name: str, nf_factor: int = 2, arch=models.resnet101) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_wide(
        data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn

'''
这个函数用于创建一个基于UNET模型的Learner对象，用于对图像进行超分辨率处理。
它接受一个ImageDataBunch对象作为数据输入，一个损失函数(gen_loss)，一个指定模型架构的参数(arch)，以及一个表示网络通道数扩大倍数的参数(nf_factor)。
在函数内部，它使用unet_learner_wide函数创建一个Learner对象，并设置了一些参数，
例如权重衰减(wd)、模糊(blur)、归一化类型(norm_type)、自注意力机制(self_attention)、输出范围(y_range)和通道数扩大倍数(nf_factor)等。
最后，该函数返回Learner对象供进一步使用。
------------------------其中unet_learner_wide函数构建了一个基于传递参数的Unet模型------------------------
'''
def gen_learner_wide(
    data: ImageDataBunch, gen_loss, arch=models.resnet101, nf_factor: int = 2
) -> Learner:
    return unet_learner_wide(
        data,
        arch=arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


'''
函数首先使用 arch 构建一个模型体系结构，并在 body 中保存其主体部分。然后使用 DynamicUnetWide 类构建一个宽的 Unet 模型，该模型包括指定的 body、输出通道数（由数据集中的类数决定）、指定的参数等。
接着将这个模型放到 GPU 上运行，并使用 Learner 类将其封装，最后返回这个封装好的 Learner 对象。
'''
def unet_learner_wide(
    data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    split_on: Optional[SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: int = 1,
    **kwargs: Any
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(
        DynamicUnetWide(
            body,
            n_classes=data.c,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


# ---------------------------------------------------------------------- 这个分割线表示不同模块
'''
这个函数用于生成一个预先训练好的图像生成模型，用于进行图像生成。
它采用了一个深层的 ResNet 架构，并加载了预训练的权重，将模型设置为评估模式，最后返回该模型的 Learner 对象以供后续使用。
'''
def gen_inference_deep(
    root_folder: Path, weights_name: str, arch=models.resnet34, nf_factor: float = 1.5) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_deep(
        data=data, gen_loss=F.l1_loss, arch=arch, nf_factor=nf_factor
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn

'''
定义了一个用于训练图像生成模型的函数 gen_learner_deep，它使用深度卷积神经网络来实现图像生成，
data 是一个 ImageDataBunch 对象，它封装了训练、验证和测试数据集的路径、大小、变换等信息。
gen_loss 是一个用于计算图像生成模型损失的函数，它通常包括对抗损失、内容损失等，用于指导模型生成更逼真的图像。
arch 是一个预定义的卷积神经网络模型，这里使用的是 ResNet34。
nf_factor 是一个用于计算卷积神经网络中特征通道数的因子，它可以控制模型的大小和复杂度。
------------------其中，unet_learner_deep函数的作用是根据传入的参数，创建 U-Net 模型并返回相应的 Learner 对象的函数-----------
'''
def gen_learner_deep(
    data: ImageDataBunch, gen_loss, arch=models.resnet34, nf_factor: float = 1.5
) -> Learner:
    return unet_learner_deep(
        data,
        arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )

'''
定义了一个创建 U-Net 模型并返回相应的 Learner 对象的函数 unet_learner_deep

'''
# The code below is meant to be merged into fastaiv1 ideally
def unet_learner_deep(
    data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    split_on: Optional[SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: float = 1.5,
    **kwargs: Any
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(
        DynamicUnetDeep(
            body,
            n_classes=data.c,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


# -----------------------------
