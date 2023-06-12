from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats

'''
get_colorize_data: 该函数用于创建用于颜色还原任务的 ImageDataBunch 对象，
该对象包含从低分辨率图像到高分辨率图像的图像对，用于训练颜色还原模型。
函数的输入包括：sz 表示图像的大小，bs 表示批量大小，crappy_path 表示低分辨率图像的文件夹路径，
            good_path 表示高分辨率图像的文件夹路径，random_seed 表示随机种子，keep_pct 表示用于训练的图像比例，
            num_workers 表示用于读取图像的工作线程数量，stats 表示用于归一化的图像统计信息，xtra_tfms 表示额外的图像变换。
该函数首先使用 ImageImageList.from_folder 函数创建一个 ImageImageList 对象，该对象包含低分辨率图像，
然后使用 use_partial_data 函数从中抽样一部分图像用于训练，再使用 split_by_rand_pct 函数将剩余的图像分为训练集和验证集。
接着使用 label_from_func 函数将每个低分辨率图像与其对应的高分辨率图像进行配对，使用 transform 函数对图像进行增强，
使用 databunch 函数创建 ImageDataBunch 对象，最后使用 normalize 函数对图像进行归一化，并将 c 属性设置为 3。
'''
def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:
    
    src = (
        ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: good_path / x.relative_to(crappy_path))
        .transform(
            get_transforms(
                max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
            ),
            size=sz,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )

    data.c = 3
    return data

'''
get_dummy_databunch: 该函数用于创建一个虚拟的 ImageDataBunch 对象，用于测试模型。
该函数创建一个大小为 1x1 的低分辨率图像和高分辨率图像，并使用 get_colorize_data 函数创建 ImageDataBunch 对象。
'''
def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )
