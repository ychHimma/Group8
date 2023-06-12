#### **本模型借鉴的模型仓库地址**：https://github.com/jantic/DeOldify

#### 一切版权归原作者所有！！！





##### 一、引言

​	近年来，深度学习和人工智能技术的发展极大地改变了许多领域的方式，尤其是图像处理领域。人们通过图像处理技术来修复或改进旧照片或图像，使其能够更好地适应现代视觉的需求。开发一种自动化且高效的黑白旧图片上色技术对于实现图像修复和改进的自动化具有非常重要的意义。

​	随着生成对抗网络（GAN）的出现，基于GAN的图像生成和修复技术已经被广泛应用于图像处理领域。然而，GAN的训练和调参需要大量的计算资源和时间，而且往往存在模型崩溃、模式崩塌等问题。针对这些问题，最近出现了一种新型的GAN训练模型NoGAN，它提供了GAN训练的好处，同时花最少的时间进行直接GAN训练，并实现高质量的图像上色效果。在这个领域的最新研究表明， 一种新型的GAN训练，NoGAN提供了GAN训练的好处，同时花最少的时间进行直接GAN训练。且大部分的训练时间是用更直接、快速和可靠的传统方法（如逐层预训练、自编码器和分类器等）分别对生成器和判别器进行预训练。那些更 "传统 "的方法通常可以得到我们所需要的大部分结果，而GAN可以用来缩小现实性方面的差距。

​	最近有一些研究人员使用 NoGAN 进行黑白图像的自动上色，该技术已经在多个领域得到应用，例如老照片修复、电影和视频游戏制作等。这些应用的成功说明了 NoGAN 在实际应用中的潜力。

​	需要注意的是，虽然 NoGAN 降低了 GAN 模型的资源和时间成本，但仍需要足够的训练数据集和一些调整超参数的技巧，以达到最佳效果。此外，NoGAN 对于复杂的图像仍存在限制，这需要在实际应用中进行考虑。

​	基于NoGAN网络的黑白旧图片上色技术，不仅可以用于个人照片的修复和改进，还可以应用于文化遗产和历史文化保护等领域，实现对历史文化遗产的修复和数字化展示。因此，本项目选择基于NoGAN网络的黑白旧图片上色技术为研究对象，旨在探索一种自动化且高效的黑白旧图片上色方法，并将其应用于历史文化遗产的数字化展示和保护。




##### 二、国内外研究现状

目前国内在图像上色方面的研究已经非常活跃，下面是几种上色方法和具体例子：

1. 基于深度学习的图像上色方法：采用深度学习方法可以学习到图像颜色空间的映射关系，能够还原出自然真实细腻的色彩。如《基于神经网络的彩色图像自动上色方法》（柳占超、刘韶辉等，2018）项目，训练神经网络将灰度图像映射到彩色图像，可还原出鲜艳自然的真实色彩。

2. 基于光学物理特性的图像上色方法：利用传统的图像上色算法往往会造成色彩失真和不自然的效果，而基于光学物理特性的算法能够更好地还原出真实的色彩。如《高动态范围图像无损颜色处理方法研究》（吴飚、陈艳波等，2019）项目，通过研究光学物理特性建立了高动态范围(HDR)图像的颜色转换模型，能够将HDR图像自然地转换到低动态范围（LDR）色彩空间。

3. 基于人工智能的图像上色应用：人工智能技术的发展促进了图像上色应用的发展。如（瑞幸咖啡的人工智能涂鸦花式上色）项目，用户可将涂鸦上传到瑞幸咖啡的App中，App通过人工智能技术将涂鸦自动上色，增加了趣味性和互动性。

此外目前国外存在许多研究，其中一些代表性的研究如下：

1. GAN based colorization：利用生成对抗网络（GAN）进行图像上色。通过训练一个生成器网络和一个判别器网络，可以生成具有高质量和真实性的彩色图像。

2. Semi-automatic colorization：这种方法结合了计算机算法和人类智力，通过在图像中选择一些具有代表性的颜色点，然后使用算法自动填充剩余的部分进行上色。

3. Deep learning based colorization：这种方法利用深度学习算法，如卷积神经网络（CNN）和循环神经网络（RNN），从原始灰度图像中预测图像的颜色。

4. Interactive colorization：这种方法让用户在一个交互式界面上与图像进行互动，在用户指定一个区域后，系统会自动填充对应的颜色。

5. Example-based colorization：这种方法利用已有的彩色图像作为参考，然后通过算法推测其他图像的颜色。这种方法可以使得上色过程更具参考性和准确性。

总之，在图像上色的研究领域，国内外都已经有了一些非常成熟的技术和算法，并且经过不断地改进和创新，上色效果已经达到了非常高的水平。




##### 三、模型和算法

**3.1** **总体模型描述**

​	本模型虽然取名NoGAN网络，但整体模型也是基于GAN网络整体框架基础之上进行搭建。NoGAN的意思是Not only GAN，也就本模型不仅采用了GAN网络的框架进行搭建，还在原始GAN网络上进行了优化。如对生成器的优化，对损失函数的优化等。

​	对于生成器的优化部分不仅仅体现在构建生成器模型时加入self-attention和光谱归一化模块，更体现在对模型训练上。传统的GAN网络训练是对生成器和判别器的同时训练，这样需要模型花费大量的时间对生成器参数进行优化。并且由于训练深度的不断加深，往往会导致得不到满意的效果。而NoGAN网络在训练整体GAN网络框架之前会先对生成器进行预训练，使生成器具备较好的为黑白图片上色的能力。然后再对整体GAN网络框架进行训练，这样便能在较短时间内训练出损失较小的上色模型。

​	对于损失函数优化部分，NoGAN网络的生成器损失函数不仅是由生成器和判别器的损失函数进行构建，NoGAN网络还增加了基于VGG16的感知损失模块，该模块虽然仅仅是让生成器产生偏差以复制输入的图像，但该模块却能在很大程度上提高生成器的梯度下降速率。

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

NoGAN网络的构成

**3.2** **生成器模块**

​	在生成器中，该模型采用的是基于**U-net网络**之上，并结合了**ResNet101网络**的模型。将学习更强的ResNet作为U-net做下采样的BackNone，增强了特征的表达能力。另外，该模型在框架基础之上，还加入了self-attention和光谱归一化模块，使得模型对图像输入的理解能力更强。其中U-net网络与ResNet101网络结构如下图：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

​																											（U-Net网络和ResNet101网络）

总体上的实现模型类似于如下图的D-LinkNet网络：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)

​																													（D-LinkNet网络）

**3.3** **判别器模块**

​	在判别器模块，该模型采用了一个卷积神经网络（CNN）模型，它由多个卷积层，自注意力层，丢弃层和平坦化层组成。卷积层用于提取图像的特征，自注意力层用于增强特征的全局依赖性，丢弃层用于防止过拟合，平坦化层用于将多维张量转换为一维向量。

​	本模型中判别器的CNN结构相对简单，采用了一系列的卷积层、Dropout层和Flatten层组成的序列，在设计上做出了一些简化和权衡，以便更好地适应GAN中的训练过程和图像上色的质量评估。其中的CNN网络结构信息如下表:

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

**3.4** **损失函数模块**

​	本模型的损失函数模块主要由两部分组成，一部分是基于VGG16的基本感知损失函数，另一部分是由判别器与生成器损失一同构建的损失函数。

其中基于VGG16的感知损失函数，这是为了让生成器产生偏差以复制输入的图像，其结构如下图：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

​																															(VGG16损失函数图)

​	基于判别器与生成器的损失构建的损失函数，再训练整体GAN网络时，对生成器的优化起着至关重要的作业。它能帮助生成器在短时间内获得不错的更加准确的生成效果，其定义如下图：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)




##### 四、实验结果分析

​	本次实验模型训练环境是通过在网络租借 GPU 服务器完成基本的模型训练，并通过对COCO-Stuff原训练数据集进行去颜色处理，使之成为本项目的训练数据集。保存具体模型参数后，在本地运行。

经过项目模型部署与多轮次的训练，模型可以生成还原度比较高的彩色图片。并可以实现对简短视频的上色。具体效果如下图所示：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)

上色实例图一

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg)

上色实例图二

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image018.jpg)

上色实例图三

​	用人工评判的方法可简单判断，以上图片的上色结果均较为还原，上色后图片的色彩、细节、逼真度、一致性等方面都符合预期。

另外，在评估模型训练效果的过程中，据不完全统计，大概比重为20%的图片，上色效果与原图略有差别，具体图片样例如下图所示：

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image020.jpg)

上色实例图四

![img](file:///C:/Users/YCH/AppData/Local/Temp/msohtmlclip1/01/clip_image022.png)

上色实例图五

​	由图四、图五可看出，上色后图片部分颜色与原图存在一定误差。这也是我们训练过程中最难以突破的瓶颈。对本次实验结果作进一步分析，可发现存在一些问题：

（1）模型在一些常见场景（如自然场景和肖像）中表现不稳定。

（2）部分图片上色结果比较暗淡。

针对以上问题，本小组也设想了一些改进方向:

（1）增加训练数据集图像种类，以使模型适应更多的需求和应用环境。

（2）要对模型进行更多的测试和评估，在不同场景中收集更多的数据和反馈，以提高模型在各种场景中的适应性和稳定性。

（3）训练过程根据结果自适应调整渲染因子。

综上所述，从实验结果上分析，模型训练成果能够达到本项目的目标，即通过本实验得到的模型，能够基本完成黑白旧图片的上色，实现图像修复和改进的自动化。




##### 五、结论

​	NoGAN网络是一个针对黑白图片和视频进行着色的自动化模型，能够将过去的老照片和电影还原为逼真、自然的彩色图像。为了更好实现该模型对黑白图片和视频着色的功能，该模型对上一代版本进行了不少的修改，其中就包括对训练网络的迭代升级和判别器的训练。

​	该项目抛弃了以往的GAN神经网络，选用了NoGAN神经网络进行模型训练。NoGAN网络提供了GAN网络训练的好处，同时花费最少的时间进行了直接的GAN训练，将大部分的时间花在了用更直接、快速和可靠的传统方法分别对生成器和判别器进行预训练上。同时，我们可以在先前的GAN训练之后，在生成的图像上对判别器进行重复的预训练，然后以相同的方式重复GAN训练本身。

​	虽然该模型的功能如此强大，但是它依旧有不少问题，比如图像质量不稳定、渲染分辨率未经优化、上色结果暗淡等，这些仍需我们去解决。

而从应用的角度看，该模型在艺术文化领域拥有着相当宽广的应用前景，可以给老旧电影、图片带来新的活力，为许多传世经典着上彩色的外衣，现实意义显著。

总的来说，该模型为图片上色提供了一种相当不错的解决方案。不仅在技术上进行了钻研革新，而且在现实中也增大了我们还原经典的可能性，是一个具有非凡意义的模型。我们非常期待看到该模型在实践中的卓越成果。




##### 六、参考文献

[1] Zhang H, Goodfellow I, Metaxas D, et al. Self-attention generative adversarial networks[C]//International conference on machine learning. PMLR, 2019: 7354-7363. 

[2] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial networks[J]. Communications of the ACM, 2020, 63(11): 139-144.

[3] Creswell A, White T, Dumoulin V, et al. Generative adversarial networks: An overview[J]. IEEE signal processing magazine, 2018, 35(1): 53-65.

[4] Metz L, Poole B, Pfau D, et al. Unrolled generative adversarial networks[J]. arXiv preprint arXiv:1611.02163, 2016.

[5] Mao X, Li Q, Xie H, et al. Least squares generative adversarial networks[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2794-2802.

[6] Krizhevsky, I. Sutskever and G. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks” (2012)

[7] K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition” (2016)

[8] Y. LeCun, Y. Bengio and G. Hinton, "Deep learning" (2015)

[9] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville and Y. Bengio, "Generative Adversarial Nets" (2014)

[10] M. Arjovsky, S. Chintala and L. Bottou, "Wasserstein Generative Adversarial Networks" (2017)

[11] T. Karras, S. Laine and T. Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks" (2018)

[12] E. Gavves, W. Liu, C. G. M. Snoek, A. W. M. Smeulders, “Local Alignments for Fine-Grained Categorization” (2014)

[13] R. Zhang, P. Isola, A. A. Efros, “Colorful Image Colorization” (2016)

[14] J. Johnson, A. Alahi, L. Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and Super-Resolution” (2016)