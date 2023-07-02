# SJWL_FINAL

## 问题一
### 研究目标
1.实现一个自监督学习算法并进行自监督预训练，利用自选的数据集和ResNet-18网络架构。
2.在CIFAR-10或CIFAR-100数据集\cite{krizhevsky2009learning}上应用Linear Classification Protocol，使用预训练的特征进行线性分类器的训练和评估。
3.比较监督学习算法和自监督学习算法在CIFAR-10或CIFAR-100图像分类任务上的性能表现，包括准确率、损失值等指标。
4.分析和讨论监督学习和自监督学习的优势、局限性，并探讨实验结果对于进一步改进自监督学习算法和应用于实际场景的启示。
### 实验设置
 我们使用4张Tesla V100S-PCIE-32GB进行并行运算。训练集和测试集的划分遵循CIFAR-100数据集的原始划分，即50000张图片作为训练集，10000张图片作为测试集。batch size为128，学习率设置为0.1，使用交叉熵损失函数作为训练的损失函数，并且使用AdamW作为优化器，一共进行了50个epoch的实验。

代码中使用了ResNet-18作为模型，并通过自监督学习算法进行预训练，然后在CIFAR-10数据集上进行训练和测试。

自监督学习算法在这里是通过CPC（Contrastive Predictive Coding）实现的。CPC算法的核心思想是通过预测图像中不同部分之间的关系来学习有用的表示。在代码中，网络被训练用于预测图像的上下文部分和预测部分之间的关系。这样，网络可以学习到图像的潜在语义信息，从而提取更丰富和有用的特征表示。

在自监督学习算法中，通过CPC进行预训练后，可以使用Linear Classification Protocol来评估学习到的特征表示的质量。Linear Classification Protocol是一种常用的评估方法，用于衡量学习到的特征在下游任务上的可用性。

### 具体步骤
模型训练：使用CPC算法对模型进行训练，通过预测图像的上下文部分和预测部分之间的关系来学习特征表示。这一步骤旨在学习到具有潜在语义信息的特征表示。

特征提取：使用预训练的模型对训练集中的图像进行特征提取。将图像输入到预训练的模型中，获取模型最后一层隐藏层的特征表示。

线性分类器训练：使用提取的特征表示作为输入，训练一个线性分类器（例如，使用支持向量机或逻辑回归）。这个线性分类器被训练用于将图像的特征表示映射到图像的标签类别。

评估性能：使用测试集评估线性分类器在图像分类任务上的性能。计算分类准确度作为评估指标，衡量预训练模型学习到的特征表示在图像分类任务上的可用性。

通过执行这个线性分类器训练和评估的步骤，可以衡量预训练模型在特征表示学习上的性能。这个步骤也可以用于比较不同自监督学习算法的性能，或者用于选择最佳的预训练模型来进行下游任务的训练。

# 问题二
## 问题描述
设计与第二次作业1模型 “相同参数量” 的Transformer网络模型，进行CIFAR-100的训练，并与期中作业1的模型结果进行比较。

## Transformer-22M模型介绍
我们设计的模型是Transformer-22M，它具有以下结构：

1. Patch Embedding：首先，我们使用一个全连接层将输入的图像（假设为32x32的RGB图像）展平为一个向量，然后将其映射到一个512维的特征空间。这种操作可以看作是一种Patch Embedding，用于将输入的像素值转化为具有更丰富信息的特征表示。

2. Transformer Encoder：接下来，我们使用了一个Transformer Encoder进行特征的进一步提取。这个Encoder由10层Transformer Encoder Layer组成，每层都是一个标准的Transformer Encoder Layer，包含一个自注意力机制（Self-Attention Mechanism）和一个前馈神经网络（Feed Forward Neural Network）。在这里，我们设置了8个注意力头，并将前馈神经网络的隐藏层维度设为1024。

3. Flattening and Permuting：为了适配Transformer Encoder的输入要求，我们对Patch Embedding的输出进行了一系列维度变换。首先，将其reshape为(batch\_size, 8, 8, channels)的形状，然后再进行维度调换，使其变为(batch\_size, channels, 8, 8)。最后，我们再次将其展平为(batch\_size, channels * 8 * 8)，以满足Transformer Encoder的输入要求。

4. Classification Head：最后，我们在Transformer Encoder的输出上接了一个全连接层，用于进行分类任务。全连接层的输出节点数为num\_classes=100，对应分类任务的类别数。

## 实验设置
我们使用4张Tesla V100S-PCIE-32GB进行并行运算。训练集和测试集的划分遵循CIFAR-100数据集的原始划分，即50000张图片作为训练集，10000张图片作为测试集。batch size为128，学习率设置为0.1，使用交叉熵损失函数作为训练的损失函数，并且使用AdamW作为优化器，一共进行了50个epoch的实验。我们使用Transformer-22M模型，同时对数据采用三种方式进行增强，分别为MIXUP、CUTOUT和CUTMIX方法。

# 问题三
## 问题描述
使用具有泛化能力的 NeRF 模型，自己构建物体数据集（如手机拍摄），对物体进行三维重建。
## NeRF模型介绍
NeRF(Neural Radiance Field)模型是发表在ECCV2020的NeRF: Representing Scenes as neural Radiance Fields for View Synthesis\cite{mildenhall2021nerf}提出的模型，它仅用二维的图片作为监督，输入同一场景不同角度的二维图片和相机位姿，即可表示复杂的三维场景。
## 具有泛化能力的 NeRF 模型
不同于其他模型注重神经网络对数据的泛化能力，NeRF 使用的是 MLP的拟合能力来表示场景，Overfit 到一个场景上对场景进行表征，所以提出具有泛化能力的NeRF模型具有重要的意义。一些著名的具有泛化能力的NeRF模型有：PixelNeRF、IBRNet、MVSNeRF。
        
# 参考文献
[1] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu,
and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 14124–14133, 2021.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning
for image recognition. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 770–778, 2016.

[3] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from
tiny images. 2009.

[4] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi
Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for
view synthesis. Communications of the ACM, 65(1):99–106, 2021.

[5] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou,
Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser.
Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 4690–4699, 2021.

[6] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural
radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4578–4587, 2021
