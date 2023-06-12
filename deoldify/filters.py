from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from abc import ABC, abstractmethod
from fastai.core import *
from fastai.vision import *
from fastai.vision.image import *
from fastai.vision.data import *
from fastai import *
import cv2
from PIL import Image as PilImage
from deoldify import device as device_settings
import logging


'''
orig_image表示原始图像，filtered_image表示过滤后的图像，render_factor表示渲染因子。这个抽象方法返回一个PilImage对象，即处理后的图像。
'''
class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass

'''
这也是一个间接的抽象类，在这个类中实现对图像大小的变化，将输入图像转换成模型能识别的图像，然后经过模型着色后，将输出的矩阵转换成输入图像大小返回
'''
class BaseFilter(IFilter):
    '''
    在__init__方法中，该类初始化了一个learn参数（即学习器）和一个stats参数（即用于数据标准化的元组），并使用normalize_funcs方法对它们进行了标准化。
    '''
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        
        if not device_settings.is_gpu():
            self.learn.model = self.learn.model.cpu()
        
        self.device = next(self.learn.model.parameters()).device
        self.norm, self.denorm = normalize_funcs(*stats)

    '''
    在_transform方法中，该类定义了一个简单的变换，用于对图像进行转换。这个方法默认是返回原始图像，子类可以覆盖这个方法来实现其他的转换。
    '''
    def _transform(self, image: PilImage) -> PilImage:
        return image

    '''
    在_scale_to_square方法中，该类定义了一个方法，用于将图像缩放到一个指定的大小，并保持其宽高比例。
    '''
    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)
    '''
    在_get_model_ready_image方法中，该类定义了一个方法，
    用于将图像转换为模型所需的格式，即先将其缩放到一个指定的大小，然后使用_transform方法对其进行转换。
    '''
    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)  # 缩放图像
        result = self._transform(result)
        return result

    '''
    在_model_process方法中，该类定义了一个方法，用于对模型处理图像。
    它将图像转换为模型所需的格式，然后使用learn.pred_batch方法进行预测。最后，将预测结果转换回图像格式，并将其返回。
    '''
    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x = x.to(self.device)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        
        try:
            result = self.learn.pred_batch(
                ds_type=DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True
            )
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            logging.warn('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image
            
        out = result[0]
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PilImage.fromarray(out)

    # 在_unsquare方法中，该类定义了一个方法，用于将处理后的图像还原回原始图像的大小和比例。
    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image

'''
这个就是着色类了，调用基类一些函数，然后这一块有一个亮点就是，它利用人类眼球对亮度不完美更加敏感的事实，将原始图像与生成图像的亮度融合
'''
class ColorizerFilter(BaseFilter):
    # 在__init__()方法中，它调用了BaseFilter类的构造函数，并为渲染大小设置了一个基础大小。
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16

    # 在filter()方法中，它调用了_model_process()方法，使用训练好的深度学习模型对输入图像进行着色处理，然后根据需要进行后处理，以得到最终的着色图像。
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    # _transform()方法将图像转换为灰度图像，然后将其转换为RGB图像。
    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    '''
    这利用了人眼对色度不完美比亮度不完美更不敏感的事实。
    这意味着我们可以在模型中节省很多内存和处理能力，但最终得到一个很好的高分辨率结果。这主要是用于推理过程。
    '''
    # _post_process()方法使用OpenCV库将颜色转换回RGB空间，并将其与原始图像的亮度值进行融合，以获得更好的结果。
    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        # do a black and white transform first to get better luminance values
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
        final = PilImage.fromarray(final)
        return final

'''
这部分定义了一个MasterFilter类，它继承了BaseFilter类，并且包含了一个IFilter类型的列表filters和一个render_factor参数。
MasterFilter类实现了filter()方法，
接收原始图像orig_image、处理后的图像filtered_image、可选的渲染因子render_factor和一个可选的后处理标志post_process，
并依次调用filters列表中的每个过滤器对图像进行处理，最后返回处理后的图像。
这个类可以方便地将多个过滤器组合起来，形成一个更复杂的处理流程。
'''
class MasterFilter(BaseFilter):
    def __init__(self, filters: List[IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image
