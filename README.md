# 今日论文推荐 - 2025-12-18

共 43 篇论文

---

## 1. In Pursuit of Pixel Supervision for Visual Pre-training

**论文链接:** [http://arxiv.org/abs/2512.15715v1](http://arxiv.org/abs/2512.15715v1)

**作者:** Lihe Yang, Shang-Wen Li, Yang Li, Xinjie Lei, Dong Wang, Abdelrahman Mohamed, Hengshuang Zhao, Hu Xu

**发布时间:** 2025-12-17

**备注:** Project page: https://github.com/facebookresearch/pixio

### GPT解析

### 总结

本文提出了一种名为'Pixio'的增强型掩码自编码器，证明了基于自编码器的自监督学习方法在今天仍然具有竞争力，能够为下游任务生成强大表示。

### 背景

像素是我们感知世界视觉信息的基本来源，包含从低级属性到高级概念的所有层次信息。自编码器是一种从像素或其他原始输入学习表示的经典且长期存在的范式。

### 目的

展示基于自编码器的自监督学习方法在今天仍然具有竞争力，能够为下游任务生成强大表示，同时保持简单、稳定和高效。

### 方法

作者提出了名为'Pixio'的模型，这是一种增强型掩码自编码器，具有更具挑战性的预训练任务和更强大的架构。模型使用自策略策略在20亿网络爬取的图像上进行训练，最小化人工策展。

### 主要发现

Pixio在广泛的下游任务中表现具有竞争力，包括单目深度估计、前馈3D重建、语义分割和机器人学习，性能优于或匹敌类似规模训练的DINOv3。

### 结论

像素空间的自监督学习可以作为潜在空间方法的一种有前景的替代和补充。

### 翻译

在最基本的层面上，像素是我们通过其感知世界的视觉信息的来源。像素包含所有层次的信息，从低级属性到高级概念。自编码器代表了一种从像素或其他原始输入学习表示的经典且长期存在的范式。在这项工作中，我们证明了基于自编码器的自监督学习在今天仍然具有竞争力，可以为下游任务生成强大的表示，同时保持简单、稳定和高效。我们的模型命名为'Pixio'，是一种增强型掩码自编码器，具有更具挑战性的预训练任务和更强大的架构。该模型使用自策展策略在20亿网络爬取的图像上进行训练，最小化人工策展。Pixio在广泛的下游任务中表现具有竞争力，包括单目深度估计、前馈3D重建、语义分割和机器人学习，性能优于或匹敌类似规模训练的DINOv3。我们的结果表明，像素空间的自监督学习可以作为一种有前景的替代和补充，补充潜在空间方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何改进基于像素级监督的自监督视觉预训练方法，使其能够更好地捕捉从低级属性到高级概念的全层次视觉信息。这个问题很重要，因为像素是视觉信息的根本来源，而现有的自监督学习方法要么依赖过多人类先验（如对比学习），要么在小规模数据上训练，限制了模型的泛化能力和性能上限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先回顾了视觉表示学习的演变，从基于人工标注的监督学习，到基于网络图像-文本对的方法，再到自监督学习。他们注意到MAE（掩码自编码器）是一个经典但可能被低估的方法，在大数据、大模型场景下设计不够优化。作者借鉴了MAE的基本框架，但从算法和数据两方面进行了改进：算法上增加了预训练难度（更大掩码块）和模型能力（更深解码器、更多类别令牌）；数据上构建了大规模多样化数据集并采用软自策略筛选。这些改进基于对MAE局限性的深入分析和实验验证。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是像素作为视觉信息的根本来源，包含了从低级到高级的全层次信息，通过掩码和像素重建任务，模型可以学习压缩和重新组织这些视觉知识。整体实现流程包括：1) 基于MAE进行三个关键算法改进（更深解码器、更大掩码块、更多类别令牌）；2) 收集20亿网络爬取图像并采用软自策略进行数据筛选；3) 在大规模数据集上使用高掩码比进行预训练；4) 在多种下游任务（深度估计、3D重建、语义分割、机器人学习）上评估模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 算法改进：更深解码器解决编码器需牺牲容量处理低级细节的问题，更大掩码块提供更丰富上下文并减少真实值泄露，更多类别令牌捕获多样化全局属性；2) 数据策略：使用20亿网络爬取图像并采用软自策略筛选，减少人工偏见；3) 性能表现：在多种下游任务上达到或超越DINOv3等先进模型。相比之前工作，Pixio基于像素级监督而非潜在空间目标，避免了过多人类先验；相比原始MAE，在算法和数据上进行了全面改进；相比需要大量人工标注的方法，采用最小人工干预的数据策略。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过改进掩码自编码器的架构设计并采用大规模多样化数据训练，证明了基于像素级监督的自学习方法可以成为潜在空间方法的强大替代方案，在多种视觉任务上达到或超越最先进性能。'}


### 论文摘要

At the most basic level, pixels are the source of the visual information through which we perceive the world. Pixels contain information at all levels, ranging from low-level attributes to high-level concepts. Autoencoders represent a classical and long-standing paradigm for learning representations from pixels or other raw inputs. In this work, we demonstrate that autoencoder-based self-supervised learning remains competitive today and can produce strong representations for downstream tasks, while remaining simple, stable, and efficient. Our model, codenamed "Pixio", is an enhanced masked autoencoder (MAE) with more challenging pre-training tasks and more capable architectures. The model is trained on 2B web-crawled images with a self-curation strategy with minimal human curation. Pixio performs competitively across a wide range of downstream tasks in the wild, including monocular depth estimation (e.g., Depth Anything), feed-forward 3D reconstruction (i.e., MapAnything), semantic segmentation, and robot learning, outperforming or matching DINOv3 trained at similar scales. Our results suggest that pixel-space self-supervised learning can serve as a promising alternative and a complement to latent-space approaches.

---

## 2. Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2512.15508v1](http://arxiv.org/abs/2512.15508v1)

**作者:** Arthur Moreau, Richard Shaw, Michal Nazarczuk, Jisu Shin, Thomas Tanay, Zhensong Zhang, Songcen Xu, Eduardo Pérez-Pellitero

**发布时间:** 2025-12-17

### GPT解析

### 总结

这项研究提出了一种新的前馈架构，通过亚像素级别的3D高斯原语检测，用自适应的'非网格'分布替代了像素网格，解决了传统前馈3D高斯飞溅模型中像素对齐原语放置次优的问题。该模型能够实时生成照片级真实感的场景，在无需相机姿态标签的情况下实现了前馈模型的最先进新视角合成性能。

### 背景

现有的前馈3D高斯飞溅(3DGS)模型可以实现实时场景生成，但受到次优像素对齐原语放置的限制，这种放置依赖于密集的刚性网格，从而限制了质量和效率。

### 目的

开发一种新的前馈架构，能够以亚像素级别检测3D高斯原语，用自适应的'非网格'分布替代传统的像素网格，从而提高场景生成的质量和效率，减少伪影，并减少所需原语的数量。

### 方法

研究引入了一种受关键点检测启发的新前馈架构，使用多分辨率解码器学习在图像块上分布原语。该模块通过自监督学习与3D重建骨干网络进行端到端训练。

### 主要发现

新模型在使用更少原语的情况下优于竞争对手，证明了更准确和高效的原语分配能够捕捉精细细节并减少伪影。此外，通过学习渲染3D高斯，3D重建骨干网络能够改进相机姿态估计。

### 结论

研究表明，通过用自适应的'非网格'分布替代传统的像素网格，可以显著提高前馈3D高斯飞溅模型的性能，为无标签训练基础模型提供了可能性。

### 翻译

前馈3D高斯飞溅(3DGS)模型能够实现实时场景生成，但受到次优像素对齐原语放置的限制，这种放置依赖于密集的刚性网格，从而限制了质量和效率。我们引入了一种新的前馈架构，能够在亚像素级别检测3D高斯原语，用自适应的'非网格'分布替代像素网格。受关键点检测启发，我们的多分辨率解码器学习在图像块上分布原语。该模块通过自监督学习与3D重建骨干网络进行端到端训练。我们得到的无需姿态模型能够在几秒钟内生成照片级真实感的场景，实现了前馈模型在新型视角合成方面的最先进性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决前馈式3D高斯泼溅(3DGS)模型中基元放置次优的问题。现有方法使用密集、刚性的网格(如像素对齐或体素对齐)来放置3D高斯基元，这限制了模型的质量和效率。这个问题很重要，因为3D高斯泼溅能实现照片级实时渲染，但前馈模型需要大量基元(通常每个像素一个)，只能处理低分辨率图像，且渲染质量有限。改进基元分配可以减少计算资源需求，提高渲染质量，扩展应用场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者质疑规则网格是否是最佳基元分配方式，观察到优化技术能自适应分布基元但前馈模型缺乏此能力。他们提出在3个级别控制基元分配：子像素级检测基元位置、多密度解码器分配更多基元到详细区域、学习置信值聚合基元。方法借鉴了关键点检测技术(如SuperPoint)使用热图提取连续坐标；借鉴APT工作使用熵度量块内容密度；采用标准U-Net架构；并在VGGT模型基础上进行微调。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'Off-The-Grid'高斯，不受限于像素网格，可在子像素级精确放置基元，并自适应分配基元数量。整体流程：1)使用VGGT骨干网络预测深度和相机参数；2)3D高斯解码器通过U-Net提取特征，用热图确定基元位置，根据图像块熵分配不同密度基元；3)将2D点转换为3D高斯并预测参数；4)多视图聚合时用置信值修剪冗余基元；5)通过光度损失、几何一致性损失等自监督训练，无需3D标注。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)子像素级基元检测而非像素对齐；2)自适应密度机制根据内容分配基元；3)置信值学习聚合多视图；4)自监督训练无需3D标注；5)改进相机姿态估计。相比之前工作：不同于像素对齐方法(如PixelSplat)为每个像素分配基元，我们使用更少基元实现更好质量；不同于体素对齐方法(如AnySplat)使用规则3D网格，我们避免了网格可见性问题；实现了更好的几何和渲染质量，特别是在放大时；训练方式上通过渲染反馈同时改进基元检测和3D重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文提出'Off-The-Grid'高斯基元检测方法，通过自适应分配和子像素级定位，显著提高了前馈式3D高斯泼溅模型的效率和渲染质量，实现了无需相机姿态标注的高质量3D场景重建。"}


### 论文摘要

Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid and limits both quality and efficiency. We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, "Off The Grid" distribution. Inspired by keypoint detection, our multi-resolution decoder learns to distribute primitives across image patches. This module is trained end-to-end with a 3D reconstruction backbone using self-supervised learning. Our resulting pose-free model generates photorealistic scenes in seconds, achieving state-of-the-art novel view synthesis for feed-forward models. It outperforms competitors while using far fewer primitives, demonstrating a more accurate and efficient allocation that captures fine details and reduces artifacts. Moreover, we observe that by learning to render 3D Gaussians, our 3D reconstruction backbone improves camera pose estimation, suggesting opportunities to train these foundational models without labels.

---

## 3. SMART: Semantic Matching Contrastive Learning for Partially View-Aligned Clustering

**论文链接:** [http://arxiv.org/abs/2512.15396v1](http://arxiv.org/abs/2512.15396v1)

**作者:** Liang Peng, Yixuan Ye, Cheng Liu, Hangjun Che, Fei Wang, Zhiwen Yu, Si Wu, Hau-San Wong

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了一种名为SMART（语义匹配对比学习模型）的方法，用于解决部分视图对齐聚类问题，通过减轻跨视图分布偏移来充分利用对齐和非对齐数据中的语义关系。

### 背景

多视图聚类可通过利用多视图数据的互补信息提高学习性能，但在现实场景中收集严格对齐的视图具有挑战性，因此需要从对齐和非对齐数据中学习。部分视图对齐聚类（PVC）旨在学习未对齐视图样本间的对应关系，以利用跨视图的一致性和互补性。

### 目的

解决现有PVC方法未能利用非对齐数据捕获同一聚类样本间共享语义的问题，以及多视图数据异质性导致的表示分布偏移问题，这些问题阻碍了建立有意义的跨视图特征对应关系。

### 方法

提出SMART（语义匹配对比学习模型），通过减轻跨视图分布偏移的影响，促进语义匹配对比学习，从而充分利用对齐和非对齐数据中的语义关系。

### 主要发现

在八个基准数据集上的实验表明，该方法在PVC问题上始终优于现有方法。

### 结论

SMART模型通过有效处理跨视图分布偏移问题，能够更好地利用对齐和非对齐数据中的语义关系，在部分视图对齐聚类任务上取得更好的性能。

### 翻译

多视图聚类已被经验证明可以通过利用多个数据视图之间的内在互补信息来提高学习性能。然而，在现实场景中，收集严格对齐的视图具有挑战性，并且从对齐和非对齐数据中学习成为一种更实用的解决方案。部分视图对齐聚类（PVC）旨在学习未对齐视图样本之间的对应关系，以更好地利用跨视图的潜在一致性和互补性，包括对齐和非对齐数据。然而，大多数现有的PVC方法未能利用非对齐数据来捕获同一聚类样本之间的共享语义。此外，多视图数据的内在异质性会导致表示中的分布偏移，从而在建立跨视图潜在特征之间的有意义对应关系时产生不准确，进而损害学习效果。为了解决这些挑战，我们为PVC提出了一种语义匹配对比学习模型（SMART）。我们方法的主要思想是减轻跨视图分布偏移的影响，从而促进语义匹配对比学习，充分利用对齐和非对齐数据中的语义关系。在八个基准数据集上进行的大量实验表明，我们的方法在PVC问题上始终优于现有方法。


### 论文摘要

Multi-view clustering has been empirically shown to improve learning performance by leveraging the inherent complementary information across multiple views of data. However, in real-world scenarios, collecting strictly aligned views is challenging, and learning from both aligned and unaligned data becomes a more practical solution. Partially View-aligned Clustering aims to learn correspondences between misaligned view samples to better exploit the potential consistency and complementarity across views, including both aligned and unaligned data. However, most existing PVC methods fail to leverage unaligned data to capture the shared semantics among samples from the same cluster. Moreover, the inherent heterogeneity of multi-view data induces distributional shifts in representations, leading to inaccuracies in establishing meaningful correspondences between cross-view latent features and, consequently, impairing learning effectiveness. To address these challenges, we propose a Semantic MAtching contRasTive learning model (SMART) for PVC. The main idea of our approach is to alleviate the influence of cross-view distributional shifts, thereby facilitating semantic matching contrastive learning to fully exploit semantic relationships in both aligned and unaligned data. Extensive experiments on eight benchmark datasets demonstrate that our method consistently outperforms existing approaches on the PVC problem.

---

## 4. Automated Motion Artifact Check for MRI (AutoMAC-MRI): An Interpretable Framework for Motion Artifact Detection and Severity Assessment

**论文链接:** [http://arxiv.org/abs/2512.15315v1](http://arxiv.org/abs/2512.15315v1)

**作者:** Antony Jerald, Dattesh Shanbhag, Sudhanya Chatterjee

**发布时间:** 2025-12-17

### GPT解析

### 总结

本研究提出了AutoMAC-MRI框架，通过监督对比学习和亲和评分方法，实现了对MRI图像中运动伪影的可解释性分级评估，有助于提高MRI质量控制效率并减少不必要的重扫描。

### 背景

运动伪影会降低MRI图像质量并增加患者重检率。现有的自动化质量评估方法大多仅限于二元决策，提供有限的可解释性。

### 目的

开发一个可解释的框架，用于评估不同MRI对比度和方向上的运动伪影等级，使等级分配透明且可解释。

### 方法

使用监督对比学习学习运动严重性的判别性表示，在特征空间中计算特定等级的亲和分数来量化图像与每个运动等级的接近程度。在5000多张专家标注的脑部MRI切片上进行了评估，这些切片涵盖了多种对比度和视图。

### 主要发现

亲和分数与专家标签的评估显示良好的一致性，支持将亲和分数用作运动严重性的可解释度量。

### 结论

通过将准确的等级检测与每个等级的亲和评分相结合，AutoMAC-MRI能够实现MRI质量控制，有潜力减少不必要的重扫描并提高工作效率。

### 翻译

运动伪影会降低MRI图像质量并增加患者重检率。现有的自动化质量评估方法大多仅限于二元决策，提供有限的可解释性。我们引入了AutoMAC-MRI，这是一个可解释的框架，用于评估不同MRI对比度和方向上的运动伪影等级。该方法使用监督对比学习来学习运动严重性的判别性表示。在这个特征空间中，我们计算特定等级的亲和分数，量化图像与每个运动等级的接近程度，从而使等级分配透明且可解释。我们在5000多张专家标注的脑部MRI切片上评估了AutoMAC-MRI，这些切片涵盖了多种对比度和视图。将亲和分数与专家标签进行对比的实验显示，这些分数与专家判断高度一致，支持将其用作运动严重性的可解释度量。通过将准确的等级检测与每个等级的亲和评分相结合，AutoMAC-MRI能够实现MRI质量控制，有潜力减少不必要的重扫描并提高工作效率。


### 论文摘要

Motion artifacts degrade MRI image quality and increase patient recalls. Existing automated quality assessment methods are largely limited to binary decisions and provide little interpretability. We introduce AutoMAC-MRI, an explainable framework for grading motion artifacts across heterogeneous MR contrasts and orientations. The approach uses supervised contrastive learning to learn a discriminative representation of motion severity. Within this feature space, we compute grade-specific affinity scores that quantify an image's proximity to each motion grade, thereby making grade assignments transparent and interpretable. We evaluate AutoMAC-MRI on more than 5000 expert-annotated brain MRI slices spanning multiple contrasts and views. Experiments assessing affinity scores against expert labels show that the scores align well with expert judgment, supporting their use as an interpretable measure of motion severity. By coupling accurate grade detection with per-grade affinity scoring, AutoMAC-MRI enables inline MRI quality control, with the potential to reduce unnecessary rescans and improve workflow efficiency.

---

## 5. Dual-coding contrastive learning based on ConvNeXt and ViT models for morphological classification of galaxies in COSMOS-Web

**论文链接:** [http://arxiv.org/abs/2512.15129v1](http://arxiv.org/abs/2512.15129v1)

**作者:** Shiwei Zhu, Guanwen Fang, Chichun Zhou, Jie Song, Zesen Lin, Yao Dai, Xu Kong

**发布时间:** 2025-12-17

### GPT解析

### 总结

这项研究提出了一种改进的无监督机器学习方法，通过对比学习升级了USmorph框架，提高了星系形态分类的效率和准确性。

### 背景

作者之前提出了一个名为USmorph的机器学习框架，用于高效分类星系形态。在这项研究中，他们提出了一种自监督方法，称为对比学习，以升级USmorph框架中的无监督机器学习部分。

### 目的

提高特征提取步骤的效率。

### 方法

使用卷积自编码器对星系图像去噪并增强旋转不变性；采用基于ConvNeXt和ViT的预训练双编码器卷积神经网络编码图像数据并应用对比学习降低特征维度；使用基于Bagging的聚类模型对星系进行分组；将模型应用于COSMOS-Web场中红移范围在0.5至6.0的星系光学图像。

### 主要发现

改进的无监督机器学习方法成功分类了73%的星系；使用GoogleNet算法对剩余27%的星系进行形态分类；分类结果与星系演化具有良好一致性。

### 结论

由于其更高的效率，更新的算法非常适合应用于未来的中国空间望远镜任务。

### 翻译

在我们之前的工作中，我们提出了一个名为USmorph的机器学习框架，用于高效分类星系形态。在本研究中，我们提出了一种称为对比学习的自监督方法，以升级USmorph框架中的无监督机器学习部分，旨在提高此步骤中特征提取的效率。升级的无监督机器学习方法主要包括以下三个方面。（1）我们采用卷积自编码器对星系图像去噪，并使用自适应极坐标变换增强模型的旋转不变性。（2）使用基于ConvNeXt和ViT的预训练双编码器卷积神经网络对图像数据进行编码，然后应用对比学习降低特征的维度。（3）我们采用基于Bagging的聚类模型将具有相似特征的星系聚类到不同的组中。通过仔细划分红移区间，我们将此模型应用于COSMOS-Web场中红移范围在0.5 < z < 6.0的星系的光学图像。与之前的算法相比，改进的无监督机器学习方法成功分类了73%的星系。使用GoogleNet算法，我们对剩余27%的星系进行形态分类。为了验证我们更新算法的可靠性，我们将分类结果与其他星系形态参数进行了比较，发现与星系演化具有良好的一致性。得益于其更高的效率，这种更新的算法非常适合应用于未来的中国空间望远镜任务。


### 论文摘要

In our previous works, we proposed a machine learning framework named \texttt{USmorph} for efficiently classifying galaxy morphology. In this study, we propose a self-supervised method called contrastive learning to upgrade the unsupervised machine learning (UML) part of the \texttt{USmorph} framework, aiming to improve the efficiency of feature extraction in this step. The upgraded UML method primarily consists of the following three aspects. (1) We employ a Convolutional Autoencoder to denoise galaxy images and the Adaptive Polar Coordinate Transformation to enhance the model's rotational invariance. (2) A pre-trained dual-encoder convolutional neural network based on ConvNeXt and ViT is used to encode the image data, while contrastive learning is then applied to reduce the dimension of the features. (3) We adopt a Bagging-based clustering model to cluster galaxies with similar features into distinct groups. By carefully dividing the redshift bins, we apply this model to the rest-frame optical images of galaxies in the COSMOS-Web field within the redshift range of $0.5 < z < 6.0$. Compared to the previous algorithm, the improved UML method successfully classifies 73\% galaxies. Using the GoogleNet algorithm, we classify the morphology of the remaining 27\% galaxies. To validate the reliability of our updated algorithm, we compared our classification results with other galaxy morphological parameters and found a good consistency with galaxy evolution. Benefiting from its higher efficiency, this updated algorithm is well-suited for application in future China Space Station Telescope missions.

---

## 6. Trustworthy Neighborhoods Mining: Homophily-Aware Neutral Contrastive Learning for Graph Clustering

**论文链接:** [http://arxiv.org/abs/2512.15027v1](http://arxiv.org/abs/2512.15027v1)

**作者:** Liang Peng, Yixuan Ye, Cheng Liu, Hangjun Che, Man-Fai Leung, Si Wu, Hau-San Wong

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了一种名为NeuCGC的新型邻域中性对比图聚类方法，通过引入中性对和两个关键组件，解决了传统对比学习在低同质性图中面临的邻域信息不可靠问题，有效处理了不同同质性水平的图聚类任务。

### 背景

邻居对比学习被引入到聚类中以有效利用邻域信息，但这些方法依赖于同质性假设（连接节点共享相似类别标签），未能考虑现实世界中图的不同同质性水平，导致在低同质性图中应用对比学习时可能产生难以区分的节点表示。

### 目的

开发一种能够处理不同同质性水平图的图聚类方法，克服传统对比学习在同质性假设上的局限性，识别具有不同同质性水平图中的可信邻域。

### 方法

提出NeuCGC方法，扩展传统对比学习，引入中性对（视为加权正对而非严格正或负对），这些中性对根据图的同质性水平动态调整；包含两个关键组件：(1)自适应对比邻域分布对齐，根据给定属性图的同质性水平调整；(2)对比邻域节点特征一致性学习机制，利用高置信度图的可信邻域信息学习鲁棒节点表示。

### 主要发现

实验结果表明，NeuCGC方法在图聚类任务中表现出有效性和鲁棒性，优于其他最先进的图聚类方法，能够有效减轻不同同质性水平的负面影响并充分利用可信邻域信息。

### 结论

NeuCGC通过引入中性对和自适应机制，能够更灵活、鲁棒地处理不同同质性水平的图聚类问题，有效解决了传统对比学习在同质性假设上的局限性。

### 翻译

最近，基于邻居的对比学习已被引入到聚类中，以有效利用邻域信息。然而，这些方法依赖于同质性假设-即连接的节点共享相似的类别标签，因此在特征空间中应该接近-这未能考虑现实世界中图的不同同质性水平。因此，将对比学习应用于低同质性图可能导致节点表示难以区分，因为邻域信息不可靠，使得在具有不同同质性水平的图中识别可信邻域在图聚类中具有挑战性。为了解决这个问题，我们引入了一种新颖的邻域中性对比图聚类方法NeuCGC，它通过引入中性对-被视为加权正对而非严格正或负对，扩展了传统对比学习。这些中性对根据图的同质性水平动态调整，使学习过程更加灵活和鲁棒。利用对比学习中的中性对，我们的方法包含两个关键组件：(1)自适应对比邻域分布对齐，根据给定属性图的同质性水平进行调整，确保邻域分布的有效对齐；(2)对比邻域节点特征一致性学习机制，利用来自高置信度图的可信邻域信息学习鲁棒节点表示，减轻不同同质性水平的负面影响，并有效利用高度可信的邻域信息。实验结果证明了我们方法的有效性和鲁棒性，优于其他最先进的图聚类方法。我们的代码可在https://github.com/THPengL/NeuCGC获取。


### 论文摘要

Recently, neighbor-based contrastive learning has been introduced to effectively exploit neighborhood information for clustering. However, these methods rely on the homophily assumption-that connected nodes share similar class labels and should therefore be close in feature space-which fails to account for the varying homophily levels in real-world graphs. As a result, applying contrastive learning to low-homophily graphs may lead to indistinguishable node representations due to unreliable neighborhood information, making it challenging to identify trustworthy neighborhoods with varying homophily levels in graph clustering. To tackle this, we introduce a novel neighborhood Neutral Contrastive Graph Clustering method, NeuCGC, that extends traditional contrastive learning by incorporating neutral pairs-node pairs treated as weighted positive pairs, rather than strictly positive or negative. These neutral pairs are dynamically adjusted based on the graph's homophily level, enabling a more flexible and robust learning process. Leveraging neutral pairs in contrastive learning, our method incorporates two key components: (1) an adaptive contrastive neighborhood distribution alignment that adjusts based on the homophily level of the given attribute graph, ensuring effective alignment of neighborhood distributions, and (2) a contrastive neighborhood node feature consistency learning mechanism that leverages reliable neighborhood information from high-confidence graphs to learn robust node representations, mitigating the adverse effects of varying homophily levels and effectively exploiting highly trustworthy neighborhood information. Experimental results demonstrate the effectiveness and robustness of our approach, outperforming other state-of-the-art graph clustering methods. Our code is available at https://github.com/THPengL/NeuCGC.

---

## 7. Preserving Marker Specificity with Lightweight Channel-Independent Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.15410v1](http://arxiv.org/abs/2512.15410v1)

**作者:** Simon Gutwein, Arthur Longuefosse, Jun Seita, Sabine Taschner-Mandl, Roxane Licandro

**发布时间:** 2025-12-17

**备注:** 16 pages, 9 figures, MIDL 2026 conference

### GPT解析

### 总结

本研究探讨了多重组织成像中自监督表示学习的最佳架构方法，发现轻量级通道独立架构比深度早期融合模型表现更好。

### 背景

多重组织成像技术能够测量每个细胞的数十种蛋白质标记，但大多数深度学习模型仍采用早期通道融合，假设标记间存在共享结构。

### 目的

研究保持标记独立性结合浅层架构是否比增加模型规模更适合多重数据中的自监督表示学习。

### 方法

使用包含145,000个细胞和49个标记的霍奇金淋巴瘤CODEX数据集，比较标准早期融合CNN与通道分离架构，包括标记感知基线和新型浅层通道独立模型(CIM-S，5.5K参数)，采用对比预训练和线性评估方法。

### 主要发现

早期融合模型保留标记特定信息能力有限，尤其在稀有细胞区分方面表现不佳；通道独立架构特别是CIM-S尽管尺寸紧凑但实现了更强表示能力；这些发现在多个自监督框架中保持一致，在不同增强设置下稳定，并且在49个和18个标记设置中均可重复。

### 结论

轻量级、通道独立的架构可以匹配或超越深度早期融合CNN和基础模型，用于多重表示学习。

### 翻译

多重组织成像测量每个细胞的数十种蛋白质标记，但大多数深度学习模型仍应用早期通道融合，假设标记间存在共享结构。我们研究保持标记独立性，结合故意浅层架构，是否比增加模型规模更适合多重数据中的自监督表示学习。使用包含145,000个细胞和49个标记的霍奇金淋巴瘤CODEX数据集，我们比较标准的早期融合CNN与通道分离架构，包括标记感知基线和我们的新型浅层通道独立模型(CIM-S，含5.5K参数)。经过对比预训练和线性评估后，早期融合模型显示保留标记特定信息的能力有限，特别是在稀有细胞区分方面表现不佳。通道独立架构，特别是CIM-S，尽管尺寸紧凑，实现了显著更强的表示能力。这些发现在多个自监督框架中保持一致，在不同增强设置下稳定，并且在49个标记和减少到18个标记的设置中均可重复。这些结果表明，轻量级、通道独立的架构可以匹配或超越深度早期融合CNN和基础模型，用于多重表示学习。代码可在https://github.com/SimonBon/CIM-S获取。


### 论文摘要

Multiplexed tissue imaging measures dozens of protein markers per cell, yet most deep learning models still apply early channel fusion, assuming shared structure across markers. We investigate whether preserving marker independence, combined with deliberately shallow architectures, provides a more suitable inductive bias for self-supervised representation learning in multiplex data than increasing model scale. Using a Hodgkin lymphoma CODEX dataset with 145,000 cells and 49 markers, we compare standard early-fusion CNNs with channel-separated architectures, including a marker-aware baseline and our novel shallow Channel-Independent Model (CIM-S) with 5.5K parameters. After contrastive pretraining and linear evaluation, early-fusion models show limited ability to retain marker-specific information and struggle particularly with rare-cell discrimination. Channel-independent architectures, and CIM-S in particular, achieve substantially stronger representations despite their compact size. These findings are consistent across multiple self-supervised frameworks, remain stable across augmentation settings, and are reproducible across both the 49-marker and reduced 18-marker settings. These results show that lightweight, channel-independent architectures can match or surpass deep early-fusion CNNs and foundation models for multiplex representation learning. Code is available at https://github.com/SimonBon/CIM-S.

---

## 8. Topological Metric for Unsupervised Embedding Quality Evaluation

**论文链接:** [http://arxiv.org/abs/2512.15285v1](http://arxiv.org/abs/2512.15285v1)

**作者:** Aleksei Shestov, Anton Klenitskiy, Daria Denisova, Amurkhan Dzagkoev, Daniil Petrovich, Andrey Savchenko, Maksim Makarenko

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了Persistence，一种基于持久同调的无监督评估指标，用于量化嵌入空间的几何结构和拓扑丰富性，无需标签数据即可评估嵌入质量。

### 背景

现代表示学习越来越多地依赖在大规模无标签数据上训练的无监督和自监督方法，这些方法虽在跨任务和跨领域实现了令人印象深刻的泛化能力，但在无标签情况下评估嵌入质量仍是开放挑战。

### 目的

提出一种名为Persistence的拓扑感知指标，基于持久同调理论，以完全无监督的方式量化嵌入空间的几何结构和拓扑丰富性。

### 方法

Persistence是一种基于持久同调的拓扑感知指标，能够捕捉嵌入空间的全局和多尺度组织结构，不假设线性可分性或依赖协方差结构。

### 主要发现

在多个不同领域的实验结果表明，Persistence与下游性能始终保持顶级相关性，优于现有的无监督指标，并能够实现可靠模型和超参数选择。

### 结论

Persistence提供了一种有效的无监督嵌入质量评估方法，不依赖于标签数据，能够捕捉嵌入空间的全局拓扑结构，在模型选择和超参数优化方面具有实用价值。

### 翻译

现代表示学习越来越依赖于在大规模无标签数据上训练的无监督和自监督方法。虽然这些方法在跨任务和跨领域方面实现了令人印象深刻的泛化能力，但在没有标签的情况下评估嵌入质量仍然是一个开放的挑战。在这项工作中，我们提出了Persistence，一种基于持久同调的拓扑感知指标，以完全无监督的方式量化嵌入空间的几何结构和拓扑丰富性。与假设线性可分性或依赖协方差结构的指标不同，Persistence能够捕捉全局和多尺度的组织结构。在多个不同领域的实验结果表明，Persistence与下游性能始终保持顶级相关性，优于现有的无监督指标，并能够实现可靠模型和超参数选择。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何在没有标签的情况下评估无监督学习嵌入质量的问题。随着现代表征学习越来越依赖无监督和自监督方法，评估这些方法生成的嵌入质量变得至关重要，因为可靠的标签无关评估方法对于可扩展和自主的学习系统（如推荐系统和信息检索）不可或缺，而现有评估方法存在局限性，无法全面捕捉嵌入空间的几何结构复杂性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到嵌入必须保留数据流形的几何结构才能在未见任务上表现良好，而现有评估方法依赖简化假设（如线性可分性或协方差结构）。作者借鉴了拓扑数据分析中的持久同调工具，这是一种已被成功应用于其他领域但在嵌入评估中较少探索的技术。通过构建Vietoris-Rips复形并计算拓扑特征寿命的总和，作者设计了Persistence指标来量化嵌入空间的几何丰富性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是嵌入质量可以通过其拓扑结构评估，高质量的嵌入应保留原始数据的拓扑特性。实现流程包括：1)构建嵌入点云的Vietoris-Rips复形；2)追踪当连接半径增加时拓扑特征的出现和消失；3)记录拓扑特征的出生和死亡值；4)计算持久条形图总结跨尺度拓扑转换；5)计算总持久性作为拓扑复杂性的度量，反映嵌入保留的内在维度和判别结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出基于持久同调的Persistence指标；构建Vietoris-Rips复形分析嵌入空间拓扑结构；计算H0和H1同调群的持久图；提供无需标签的几何敏感度量。相比之前工作，Persistence不依赖线性可分性假设或协方差结构；能捕捉全局和多尺度结构；是模型无关的；评估更全面的几何结构，包括环、空腔等复杂拓扑特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于持久同调的拓扑感知指标Persistence，能够在无监督条件下准确评估嵌入质量，通过量化嵌入空间的几何结构丰富性，为模型选择和超参数优化提供了可靠指导，在多个领域的实验中显著优于现有无监督评估方法。'}


### 论文摘要

Modern representation learning increasingly relies on unsupervised and self-supervised methods trained on large-scale unlabeled data. While these approaches achieve impressive generalization across tasks and domains, evaluating embedding quality without labels remains an open challenge. In this work, we propose Persistence, a topology-aware metric based on persistent homology that quantifies the geometric structure and topological richness of embedding spaces in a fully unsupervised manner. Unlike metrics that assume linear separability or rely on covariance structure, Persistence captures global and multi-scale organization. Empirical results across diverse domains show that Persistence consistently achieves top-tier correlations with downstream performance, outperforming existing unsupervised metrics and enabling reliable model and hyperparameter selection.

---

## 9. On the Use of Self-Supervised Representation Learning for Speaker Diarization and Separation

**论文链接:** [http://arxiv.org/abs/2512.15224v1](http://arxiv.org/abs/2512.15224v1)

**作者:** Séverin Baroudi, Hervé Bredin, Joseph Razik, Ricard Marxer

**发布时间:** 2025-12-17

**备注:** accepted at ASRU25

### GPT解析

### 总结

本文研究了近期自监督语音模型在说话人分割和语音分离这两个与说话人身份相关任务上的质量表现，指出了当前文献中的评估不足问题。

### 背景

过去几年，wav2vec2.0和WavLM等自监督语音模型已被证明能显著提高许多下游语音任务的性能，特别是在低资源环境中。

### 目的

研究近期自监督语音表示在说话人分割和语音分离这两个与说话人身份相关任务上的质量。

### 方法

对近期自监督语音表示在说话人分割和语音分离任务上的质量进行评估。

### 主要发现

当前文献中存在评估不足的问题，这源于现有基准测试的限制，特别是评估数据集的多样性不足以及与分割和分离相关的下游系统种类有限。

### 结论

需要更全面和多样化的评估来充分了解自监督语音模型在说话人相关任务上的性能。

### 翻译

近年来，wav2vec2.0和WavLM等自监督语音模型已被证明能显著提高许多下游语音任务的性能，特别是在低资源环境中。尽管如此，在说话人分割和语音分离等任务上的评估仍然有限。本文研究了近期自监督语音表示在这两个与说话人身份相关任务上的质量，指出了当前文献中的差距，这些差距源于现有基准测试的限制，特别是评估数据集的多样性不足以及与分割和分离相关的下游系统种类有限。


### 论文摘要

Self-supervised speech models such as wav2vec2.0 and WavLM have been shown to significantly improve the performance of many downstream speech tasks, especially in low-resource settings, over the past few years. Despite this, evaluations on tasks such as Speaker Diarization and Speech Separation remain limited. This paper investigates the quality of recent self-supervised speech representations on these two speaker identity-related tasks, highlighting gaps in the current literature that stem from limitations in the existing benchmarks, particularly the lack of diversity in evaluation datasets and variety in downstream systems associated to both diarization and separation.

---

## 10. Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption

**论文链接:** [http://arxiv.org/abs/2512.15112v1](http://arxiv.org/abs/2512.15112v1)

**作者:** Sunwoo Kim, Soo Yong Lee, Kyungho Kim, Hyunjin Hwang, Jaemin Yoo, Kijung Shin

**发布时间:** 2025-12-17

**备注:** Published in AAAI 2026

### GPT解析

### 总结

本文提出了一种名为FUEL的无监督节点表示学习方法，它能够自适应地学习图卷积使用的适当程度，通过增强嵌入空间中的类内相似性和类间可分性来提升性能。FUEL利用节点特征识别节点簇，并将这些簇视为类的代理。在15种基线方法和14个基准数据集上的广泛实验表明，FUEL在不同同质性的图上都能实现最先进的性能。

### 背景

无监督节点表示学习旨在不依赖节点标签的情况下获取有意义的节点嵌入。通常使用图卷积来聚合邻居节点的信息，从而编码节点特征和图拓扑结构。然而，过度依赖图卷积可能是次优的，特别是在非同质图中，因为它可能对特征或拓扑属性不同的节点产生过于相似的嵌入。

### 目的

解决无监督场景下调整图卷积使用程度的研究不足问题，提出一种自适应学习图卷积使用适当程度的方法，以增强嵌入空间中的类内相似性和类间可分性。

### 方法

提出FUEL方法，它自适应地学习图卷积使用的适当程度，利用节点特征识别节点簇，并将这些簇视为类的代理，从而增强嵌入空间中的类内相似性和类间可分性。

### 主要发现

通过15种基线方法和14个基准数据集的广泛实验，证明了FUEL在下游任务中的有效性，在不同同质性的图上实现了最先进的性能。

### 结论

FUEL成功解决了无监督场景下调整图卷积使用程度的问题，通过自适应学习图卷积使用的适当程度，显著提升了节点表示学习的性能。

### 翻译

无监督节点表示学习旨在不依赖节点标签的情况下获取有意义的节点嵌入。为实现这一目标，通常采用图卷积（聚合来自邻居节点的信息）来编码节点特征和图拓扑结构。然而，过度依赖图卷积可能是次优的，特别是在非同质图中，因为它可能对特征或拓扑属性不同的节点产生过于相似的嵌入。因此，调整图卷积使用程度的问题已在监督学习环境中得到积极探索，而在无监督场景中，此类方法仍研究不足。为解决这一问题，我们提出了FUEL，它通过旨在增强嵌入空间中的类内相似性和类间可分性，自适应地学习图卷积使用的适当程度。由于类别未知，FUEL利用节点特征识别节点簇，并将这些簇视为类别的代理。通过使用15种基线方法和14个基准数据集的广泛实验，我们证明了FUEL在下游任务中的有效性，在不同同质性水平的图上实现了最先进的性能。


### 论文摘要

Unsupervised node representation learning aims to obtain meaningful node embeddings without relying on node labels. To achieve this, graph convolution, which aggregates information from neighboring nodes, is commonly employed to encode node features and graph topology. However, excessive reliance on graph convolution can be suboptimal-especially in non-homophilic graphs-since it may yield unduly similar embeddings for nodes that differ in their features or topological properties. As a result, adjusting the degree of graph convolution usage has been actively explored in supervised learning settings, whereas such approaches remain underexplored in unsupervised scenarios. To tackle this, we propose FUEL, which adaptively learns the adequate degree of graph convolution usage by aiming to enhance intra-class similarity and inter-class separability in the embedding space. Since classes are unknown, FUEL leverages node features to identify node clusters and treats these clusters as proxies for classes. Through extensive experiments using 15 baseline methods and 14 benchmark datasets, we demonstrate the effectiveness of FUEL in downstream tasks, achieving state-of-the-art performance across graphs with diverse levels of homophily.

---

## 11. Magnification-Aware Distillation (MAD): A Self-Supervised Framework for Unified Representation Learning in Gigapixel Whole-Slide Images

**论文链接:** [http://arxiv.org/abs/2512.14796v1](http://arxiv.org/abs/2512.14796v1)

**作者:** Mahmut S. Gokmen, Mitchell A. Klusty, Peter T. Nelson, Allison M. Neltner, Sen-Ching Samson Cheung, Thomas M. Pearce, David A Gutman, Brittany N. Dugger, Devavrat S. Bisht, Margaret E. Flanagan, V. K. Cody Bumgardner

**发布时间:** 2025-12-16

**备注:** 10 pages, 4 figures, 5 tables, submitted to AMIA 2026 Informatics Summit

### GPT解析

### 总结

本研究提出了一种名为放大感知蒸馏(MAD)的自监督策略，解决了全切片图像(WSIs)中不同放大级别被视为独立视图的问题，使模型能够学习到分辨率不变的表示。

### 背景

全切片图像包含分布在多个放大级别上的组织信息，但大多数自监督方法将这些尺度视为独立视图，这种分离阻止了模型学习在分辨率变化时保持稳定的表示，而这对于神经病理学工作流程至关重要。

### 目的

开发一种能够连接低放大倍数上下文与高放大倍数细节的自监督方法，使模型学习粗略组织结构与精细细胞模式之间的关系，实现分辨率不变的表示学习。

### 方法

引入放大感知蒸馏(MAD)策略，基于跨尺度对应关系完全无注释地训练基础模型MAD-NP，将低放大倍数上下文与空间对齐的高放大倍数细节联系起来。

### 主要发现

仅在10倍嵌入上训练的线性分类器应用于未见过的40倍切片时保持96.7%的性能，证明了强大的分辨率不变表示学习能力；分割输出在不同放大倍数下保持一致，保留解剖边界并最小化噪声。

### 结论

研究结果突显了使用统一嵌入空间进行可扩展、放大倍数鲁棒的全切片图像分析的可行性。

### 翻译

全切片图像包含分布在多个放大级别上的组织信息，然而大多数自监督方法将这些尺度视为独立视图。这种分离阻止了模型学习在分辨率变化时保持稳定的表示，这是实际神经病理学工作流程的关键要求。本研究引入了放大感知蒸馏(MAD)，一种自监督策略，将低放大倍数上下文与空间对齐的高放大倍数细节联系起来，使模型能够学习粗略组织结构与精细细胞模式之间的关系。由此产生的基础模型MAD-NP完全通过这种跨尺度对应关系进行训练，无需注释。仅在10倍嵌入上训练的线性分类器应用于未见过的40倍切片时保持96.7%的性能，证明了强大的分辨率不变表示学习能力。分割输出在不同放大倍数下保持一致，保留解剖边界并最小化噪声。这些结果突显了使用统一嵌入空间进行可扩展、放大倍数鲁棒的全切片图像分析的可行性。


### 论文摘要

Whole-slide images (WSIs) contain tissue information distributed across multiple magnification levels, yet most self-supervised methods treat these scales as independent views. This separation prevents models from learning representations that remain stable when resolution changes, a key requirement for practical neuropathology workflows. This study introduces Magnification-Aware Distillation (MAD), a self-supervised strategy that links low-magnification context with spatially aligned high-magnification detail, enabling the model to learn how coarse tissue structure relates to fine cellular patterns. The resulting foundation model, MAD-NP, is trained entirely through this cross-scale correspondence without annotations. A linear classifier trained only on 10x embeddings maintains 96.7% of its performance when applied to unseen 40x tiles, demonstrating strong resolution-invariant representation learning. Segmentation outputs remain consistent across magnifications, preserving anatomical boundaries and minimizing noise. These results highlight the feasibility of scalable, magnification-robust WSI analysis using a unified embedding space

---

## 12. SkyCap: Bitemporal VHR Optical-SAR Quartets for Amplitude Change Detection and Foundation-Model Evaluation

**论文链接:** [http://arxiv.org/abs/2512.14755v1](http://arxiv.org/abs/2512.14755v1)

**作者:** Paul Weinmann, Ferdinand Schenck, Martin Šiklar

**发布时间:** 2025-12-15

**备注:** 8 pages, 0 figures. Accepted at Advances in Representation Learning for Earth Observation (REO) at EurIPS 2025

### GPT解析

### 总结

本研究介绍了SkyCap数据集，一个用于线性基础设施监测的双时相高分辨率光学-SAR数据集，并评估了基础模型在SAR振幅变化检测任务上的性能。

### 背景

线性基础设施监测需要可靠的高分辨率数据和规律的采集节奏。光学超高分辨率图像易于解释和标注，但云层会破坏采集节奏。合成孔径雷达(SAR)可以在任何天气条件下获取数据，但难以标注。

### 目的

构建一个双时相VHR光学-SAR数据集；通过光学到SAR的标签转移获得SAR振幅变化检测标签，无需SAR专家标注；评估基础模型在SAR ACD任务上的性能。

### 方法

通过档案匹配和共配准SkySat(光学)和Capella Space(SAR)场景构建SkyCap数据集；使用光学到SAR的标签转移技术获取SAR ACD标签；在SAR数据上继续预训练SARATR-X模型；在SkyCap数据集上评估经过预训练的SAR特定基础模型和光学基础模型，使用不同的预处理选择。

### 主要发现

MTP(ViT-B+RVSA)，一个光学基础模型，使用dB+Z-score预处理获得最佳结果(F1_c = 45.06)，优于直接在Capella数据上进一步预训练的SAR特定基础模型；模型对与预训练统计对齐的预处理非常敏感；光学模型在光学变化检测上的排名不会一对一地转移到SAR ACD。

### 结论

据作者所知，这是第一个在VHR SAR ACD上评估基础模型的研究，表明光学基础模型在某些条件下可能优于专门针对SAR训练的模型。

### 翻译

线性基础设施监测的变化检测需要可靠的高分辨率数据和规律的采集节奏。光学超高分辨率(VHR)图像易于解释和标注，但云层会破坏这种采集节奏。合成孔径雷达(SAR)能够在任何天气条件下获取数据，但难以标注。我们介绍了SkyCap，一个通过档案匹配和共配准(光学)SkySat和Capella Space(SAR)场景构建的双时相VHR光学-SAR数据集。我们利用光学到SAR的标签转移来获取SAR振幅变化检测(ACD)标签，而无需SAR专家标注。我们在SAR数据上对SARATR-X进行了持续预训练，并在SkyCap数据集上，针对不同的预处理选择，将由此产生的SAR特定基础模型(FMs)与SARATR-X以及光学FMs进行了基准测试。在评估的模型中，MTP(ViT-B+RVSA)，一个光学FM，使用dB+Z-score预处理获得了最佳结果(F1_c = 45.06)，优于直接在Capella数据上进一步预训练的SAR特定FMs。我们观察到模型对与预训练统计对齐的预处理非常敏感，并且光学模型在光学变化检测上的排名不会一对一地转移到SAR ACD。据我们所知，这是第一个在VHR SAR ACD上评估基础模型的研究。


### 论文摘要

Change detection for linear infrastructure monitoring requires reliable high-resolution data and regular acquisition cadence. Optical very-high-resolution (VHR) imagery is interpretable and straightforward to label, but clouds break this cadence. Synthetic Aperture Radar (SAR) enables all-weather acquisitions, yet is difficult to annotate. We introduce SkyCap, a bitemporal VHR optical-SAR dataset constructed by archive matching and co-registration of (optical) SkySat and Capella Space (SAR) scenes. We utilize optical-to-SAR label transfer to obtain SAR amplitude change detection (ACD) labels without requiring SAR-expert annotations. We perform continued pretraining of SARATR-X on our SAR data and benchmark the resulting SAR-specific foundation models (FMs) together with SARATR-X against optical FMs on SkyCap under different preprocessing choices. Among evaluated models, MTP(ViT-B+RVSA), an optical FM, with dB+Z-score preprocessing attains the best result (F1$_c$ = 45.06), outperforming SAR-specific FMs further pretrained directly on Capella data. We observe strong sensitivity to preprocessing alignment with pretraining statistics, and the ranking of optical models on optical change detection does not transfer one-to-one to SAR ACD. To our knowledge, this is the first evaluation of foundation models on VHR SAR ACD.

---

## 13. Spatia: Video Generation with Updatable Spatial Memory

**论文链接:** [http://arxiv.org/abs/2512.15716v1](http://arxiv.org/abs/2512.15716v1)

**作者:** Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, Yan Lu

**发布时间:** 2025-12-17

**备注:** Project page: https://zhaojingjing713.github.io/Spatia/

### GPT解析

### 总结

Spatia是一种空间记忆感知的视频生成框架，通过保存3D场景点云作为持久空间记忆，并使用视觉SLAM更新它，解决了现有模型在长期空间和时间一致性方面的问题。

### 背景

现有视频生成模型难以维持长期空间和时间一致性，因为视频信号具有密集、高维的特性。

### 目的

克服现有视频生成模型在维持长期空间和时间一致性方面的局限性。

### 方法

提出Spatia框架，明确保存3D场景点云作为持久空间记忆，基于此空间记忆迭代生成视频片段，并通过视觉SLAM持续更新它。采用动态-静态解耦设计增强空间一致性。

### 主要发现

动态-静态解耦设计可以增强整个生成过程中的空间一致性，同时保留了模型产生逼真动态实体的能力。

### 结论

Spatia实现了显式相机控制和3D感知交互编辑等应用，为可扩展、内存驱动的视频生成提供了几何基础框架。

### 翻译

现有的视频生成模型由于视频信号的密集、高维特性，难以维持长期的空间和时间一致性。为了克服这一局限，我们提出了Spatia，一种空间记忆感知的视频生成框架，明确地将3D场景点云保存为持久空间记忆。Spatia基于此空间记忆迭代生成视频片段，并通过视觉SLAM持续更新它。这种动态-静态解耦设计增强了整个生成过程中的空间一致性，同时保留了模型产生逼真动态实体的能力。此外，Spatia实现了显式相机控制和3D感知交互编辑等应用，为可扩展、内存驱动的视频生成提供了几何基础框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有视频生成模型难以维持长期空间和时间一致性的问题。视频信号是密集且高维的，导致模型难以编码长期历史信息，例如5秒视频就包含约36,000个时空token，而相同数量的token在语言模型中可表示约27,000个单词。这个问题在现实中很重要，因为许多应用如世界模型、AI游戏生成和具身AI需要长时间的视频生成，要求时间一致性和持久记忆，而现有模型难以满足这些需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有视频生成模型的局限性，借鉴了语言模型中的记忆机制，但考虑到视频与文本的本质差异，设计了专门针对视频的显式记忆机制。他们借鉴了视觉SLAM技术用于3D场景重建和更新，参考了3D高斯溅射的渲染过程用于相机控制，并采用了多模态条件生成框架。通过将静态场景与动态实体分离，使用3D场景点云作为空间记忆，并迭代更新，解决了长期一致性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是维护一个3D场景点云作为显式的空间记忆，在视频生成过程中基于这个记忆进行条件生成，并通过视觉SLAM不断更新它。这种方法实现了动态-静态分离，将静态场景作为持久记忆，同时生成与场景交互的动态实体。训练阶段包括数据预处理、视图特定场景点云估计、参考帧检索和多模态条件生成。推理阶段从初始图像估计初始点云，用户指定文本和相机轨迹，渲染场景投影视频，检索参考帧，生成新视频片段，然后更新空间记忆，迭代执行以生成长视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 动态-静态分离，将静态场景作为空间记忆同时生成动态实体，而之前方法通常只能处理静态场景；2) 空间一致性生成，可从不同视角生成保持空间结构一致的视频；3) 显式相机控制，通过3D感知方式直接应用相机路径到点云并渲染，而非间接编码相机轨迹；4) 3D感知交互式编辑，用户可直接编辑点云并反映在生成视频中。相比之前工作，Spatia首次实现了显式空间记忆机制，同时支持动态内容生成、多视角一致性、精确相机控制和交互式编辑。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Spatia通过维护和迭代更新3D场景点云作为显式空间记忆，实现了具有长期空间和时间一致性的高质量视频生成，同时支持动态内容生成、多视角一致性、精确相机控制和3D感知交互式编辑。'}


### 论文摘要

Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory. Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM. This dynamic-static disentanglement design enhances spatial consistency throughout the generation process while preserving the model's ability to produce realistic dynamic entities. Furthermore, Spatia enables applications such as explicit camera control and 3D-aware interactive editing, providing a geometrically grounded framework for scalable, memory-driven video generation.

---

## 14. OMCL: Open-vocabulary Monte Carlo Localization

**论文链接:** [http://arxiv.org/abs/2512.15557v1](http://arxiv.org/abs/2512.15557v1)

**作者:** Evgenii Kruzhkov, Raphael Memmesheimer, Sven Behnke

**发布时间:** 2025-12-17

**备注:** Accepted to IEEE RA-L

### GPT解析

### 总结

本文提出了一种基于视觉-语言特征的蒙特卡洛定位方法，实现机器人测量与环境地图特征的稳健关联，支持通过自然语言描述进行全局定位初始化，并在室内外场景中展示了良好的泛化能力。

### 背景

稳健的机器人定位是导航规划的重要前提。当环境地图由不同传感器创建时，机器人测量必须与地图特征进行稳健关联。

### 目的

扩展蒙特卡洛定位方法，利用视觉-语言特征实现机器人观测与3D地图之间的稳健关联，并支持通过自然语言描述进行全局定位初始化。

### 方法

使用视觉-语言特征扩展蒙特卡洛定位，这些开放词汇特征能根据相机姿态和由RGB-D图像或对齐点云创建的3D地图，稳健计算视觉观测的可能性；抽象视觉-语言特征可关联不同模态的观测和地图元素；全局定位可通过位置周围物体的自然语言描述初始化。

### 主要发现

该方法在Matterport3D和Replica数据集的室内场景以及SemanticKITTI户外场景中进行了评估，展示了良好的泛化能力。

### 结论

基于视觉-语言特征的蒙特卡洛定位方法能够有效处理多传感器创建的环境地图，实现稳健的机器人定位，并通过自然语言描述支持全局定位初始化。

### 翻译

稳健的机器人定位是导航规划的重要前提。如果环境地图是由不同传感器创建的，机器人测量必须与地图特征进行稳健关联。在这项工作中，我们使用视觉-语言特征扩展了蒙特卡洛定位。这些开放词汇特征能够根据相机姿态和由姿态RGB-D图像或对齐点云创建的3D地图，稳健地计算视觉观测的可能性。抽象视觉-语言特征能够关联来自不同模态的观测和地图元素。全局定位可以通过位置周围存在的物体的自然语言描述来初始化。我们使用Matterport3D和Replica评估了室内场景的方法，并在SemanticKITTI上展示了户外场景的泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人在复杂环境中的鲁棒定位问题，特别是在使用不同传感器创建的地图中如何将机器人观测与地图特征进行稳健关联。这个问题在现实中非常重要，因为准确的定位是机器人导航的前提，而现实环境中的地图通常由多种传感器（如RGB-D相机、激光雷达等）创建，不同传感器数据存在模态差异，如何有效关联这些数据是机器人自主导航的关键挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统定位方法在处理多模态传感器数据时的局限性，然后考虑如何利用语义信息提高定位鲁棒性。他们借鉴了蒙特卡洛定位(MCL)的基本框架，但扩展它以使用开放词汇的视觉-语言特征。作者采用了对比语言-图像预训练模型(如CLIP)来提取抽象特征，这些特征能够关联不同模态的观测和地图元素。他们还设计了两种映射方法处理不同传感器数据，并引入了基于文本提示的初始化方法。作者确实借鉴了现有工作，包括MCL框架、视觉-语言模型、语义分割技术和八叉树地图表示方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用开放词汇的视觉-语言特征来表示环境和观测，将蒙特卡洛定位与这些特征相结合，通过计算观测特征与地图特征之间的相似度来确定机器人位置。整体流程分为三个阶段：1)映射阶段：创建八叉树语言地图存储视觉-语言特征，支持从RGB-D图像或点云构建；2)定位阶段：使用蒙特卡洛定位，从RGB图像提取特征，通过光线追踪获取地图对应位置特征，计算相似度赋予权重，采用分层光线采样提高效率；3)初始化阶段：使用文本提示描述可能初始位置，在匹配位置附近初始化粒子。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)语言基础的定位：将位姿估计建立在语言特征基础上，使用开放词汇提示加速全局定位；2)跨模态传感器使用：构建统一的稀疏语言地图，支持不同传感器用于映射和定位；3)泛化能力：兼容独立构建的点云，可在室内外环境有效工作。相比之前的工作，不同之处在于：不依赖特定几何特征而使用语义一致性；使用开放词汇特征而非预定义类别；专注于定位任务而非场景理解；使用八叉树语言地图高效存储特征；支持从不同传感器输入构建统一地图。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OMCL通过结合蒙特卡洛定位与开放词汇的视觉-语言特征，实现了跨模态传感器下的鲁棒机器人定位，并支持自然语言描述的初始化，显著提升了定位的准确性和灵活性。'}


### 论文摘要

Robust robot localization is an important prerequisite for navigation planning. If the environment map was created from different sensors, robot measurements must be robustly associated with map features. In this work, we extend Monte Carlo Localization using vision-language features. These open-vocabulary features enable to robustly compute the likelihood of visual observations, given a camera pose and a 3D map created from posed RGB-D images or aligned point clouds. The abstract vision-language features enable to associate observations and map elements from different modalities. Global localization can be initialized by natural language descriptions of the objects present in the vicinity of locations. We evaluate our approach using Matterport3D and Replica for indoor scenes and demonstrate generalization on SemanticKITTI for outdoor scenes.

---

## 15. ISS Policy : Scalable Diffusion Policy with Implicit Scene Supervision

**论文链接:** [http://arxiv.org/abs/2512.15020v1](http://arxiv.org/abs/2512.15020v1)

**作者:** Wenlong Xia, Jinhao Zhang, Ce Zhang, Yaojia Wang, Youmin Gong, Jie Mei

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了一种名为隐式场景监督（ISS）策略的3D视觉运动DiT扩散模型，通过点云观测预测连续动作序列，解决了基于视觉的模仿学习训练效率低和泛化能力差的问题。

### 背景

基于视觉的模仿学习已实现令人印象深刻的机器人操作技能，但依赖物体外观而忽略底层3D场景结构，导致训练效率低下和泛化能力差。

### 目的

解决现有视觉模仿学习方法效率低和泛化能力差的问题，提高机器人操作的性能和鲁棒性。

### 方法

扩展DiT模型，添加新颖的隐式场景监督模块，鼓励模型产生与场景几何演化一致的输出，从点云观测中预测连续动作序列。

### 主要发现

ISS策略在单臂操作任务（MetaWorld）和灵巧手操作（Adroit）上达到最先进性能；在真实世界实验中表现出强大泛化能力和鲁棒性；方法能随数据和参数增加有效扩展。

### 结论

隐式场景监督策略显著提高了机器人操作的性能和泛化能力，代码和视频将公开发布。

### 翻译

基于视觉的模仿学习已实现了令人印象深刻的机器人操作技能，但其对物体外观的依赖而忽略底层3D场景结构导致训练效率低下和泛化能力差。为解决这些挑战，我们引入隐式场景监督（ISS）策略，一种基于3D视觉运动DiT的扩散策略，从点云观测中预测连续动作序列。我们通过扩展DiT并添加新颖的隐式场景监督模块来鼓励模型产生与场景几何演化一致的输出，从而提高策略的性能和鲁棒性。值得注意的是，ISS策略在单臂操作任务（MetaWorld）和灵巧手操作（Adroit）上都达到了最先进的性能。在真实世界实验中，它还表现出强大的泛化能力和鲁棒性。额外的消融研究表明，我们的方法能够随着数据和参数的增加而有效扩展。代码和视频将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉引导机器人学习方法中过度依赖物体外观而忽略底层3D场景结构的问题，这导致训练效率低下和泛化能力差。这个问题很重要，因为机器人需要在非结构化环境中执行复杂任务，而精确物体状态数据稀缺且获取成本高；2D图像方法缺乏深度信息造成空间歧义；现有3D方法计算密集且学习效率低下，限制了机器人操作在实际应用中的性能和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：2D方法缺乏深度信息，3D方法计算密集且学习效率低。基于这些分析，他们设计了一个基于DiT的3D视觉运动策略，使用点云作为输入提供丰富几何信息。他们借鉴了扩散模型在机器人控制中的应用（如Diffusion Policy和DP3），采用DiT架构提高可扩展性，并创新性地引入了隐式场景监督模块，灵感来自如果模型输出正确动作应能准确预测未来场景的观点。整个设计旨在平衡几何理解与计算效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过隐式场景监督模块预测未来点云特征，强制执行长期几何一致性，使策略能隐式建模场景动态，同时利用点云提供的丰富几何信息提高操作性能。整体流程包括：1)将单视图深度转换为稀疏点云并编码为上下文表示；2)使用基于DiT的编码器-解码器架构处理带噪声的动作序列；3)通过隐式场景监督模块预测K步未来点云；4)结合去噪损失和隐式场景监督损失进行训练；5)推理时仅使用DiT策略，不增加计算开销。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入隐式场景监督模块，强制模型输出与场景几何演化一致；2)使用稀疏点云而非密集体素网格，平衡几何信息与计算效率；3)采用DiT架构提高可扩展性和计算效率；4)设计计划采样策略稳定训练。相比2D方法，本文提供更丰富的几何上下文，更好处理遮挡和精细接触任务；相比现有3D方法，本文引入额外几何监督信号，提高学习效率，具有更好可扩展性和稳定性，推理速度更快（比DP3快约3倍）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了ISS Policy，一种通过隐式场景监督有效利用点云几何信息的3D视觉运动策略，实现了高训练效率、强可扩展性和卓越泛化能力，在模拟和真实世界机器人操作任务中达到最先进性能。'}


### 论文摘要

Vision-based imitation learning has enabled impressive robotic manipulation skills, but its reliance on object appearance while ignoring the underlying 3D scene structure leads to low training efficiency and poor generalization. To address these challenges, we introduce \emph{Implicit Scene Supervision (ISS) Policy}, a 3D visuomotor DiT-based diffusion policy that predicts sequences of continuous actions from point cloud observations. We extend DiT with a novel implicit scene supervision module that encourages the model to produce outputs consistent with the scene's geometric evolution, thereby improving the performance and robustness of the policy. Notably, ISS Policy achieves state-of-the-art performance on both single-arm manipulation tasks (MetaWorld) and dexterous hand manipulation (Adroit). In real-world experiments, it also demonstrates strong generalization and robustness. Additional ablation studies show that our method scales effectively with both data and parameters. Code and videos will be released.

---

## 16. M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.12378v2](http://arxiv.org/abs/2512.12378v2)

**作者:** Junqiao Fan, Yunjiao Zhou, Yizhuo Yang, Xinyuan Cui, Jiarui Zhang, Lihua Xie, Jianfei Yang, Chris Xiaoxuan Lu, Fangqiang Ding

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文介绍了M4Human，一个大规模多模态基准数据集，用于人类网格重建研究。该数据集包含661K帧的高分辨率毫米波雷达、RGB和深度数据，是目前最大规模的同类数据集，比之前最大的数据集大9倍。数据集包括20个受试者和50种不同动作的高质量动作捕捉标注。

### 背景

现有的大规模HMR数据集主要依赖视线RGB输入，但视觉感知受到遮挡、光照变化和隐私问题的限制。虽然毫米波雷达可解决这些问题，但当前雷达数据集存在稀疏骨架标签、规模有限和动作简单等限制。

### 目的

为了推进HMR研究社区，引入M4Human这一大规模多模态基准数据集，克服现有数据集的局限性，为雷达基础人体建模研究提供更丰富的资源。

### 方法

创建M4Human数据集，包含原始雷达张量(RT)和处理后的雷达点云(RPC)，支持不同级别的射频信号粒度研究；提供高质量动作捕捉标注，包括3D网格和全局轨迹；涵盖20个受试者和50种多样动作；建立RT和RPC模态基准，以及与RGB-D模态的多模态融合基准。

### 主要发现

广泛实验结果突显了M4Human对雷达基础人体建模的重要性，同时揭示了在快速、不受约束运动下仍然存在的挑战。

### 结论

M4Human数据集和代码将在论文发表后发布，这将促进HMR研究社区的发展，特别是在使用毫米波雷达进行隐私保护室内人体感知方面。

### 翻译

人类网格重建(HMR)为身体与环境之间的直接互动提供了洞察，使各种沉浸式应用成为可能。虽然现有的大规模HMR数据集严重依赖视线RGB输入，但基于视觉的感知受到遮挡、光照变化和隐私问题的限制。为了克服这些限制，最近的研究探索了射频毫米波雷达用于隐私保护的室内人体感知。然而，当前的雷达数据集受到稀疏骨架标签、规模有限和简单原地动作的限制。为了推进HMR研究社区，我们引入了M4Human，这是目前最大规模(661K帧)(比之前最大的数据集大9倍)的多模态基准，具有高分辨率的毫米波雷达、RGB和深度数据。M4Human提供原始雷达张量(RT)和处理后的雷达点云(RPC)，以支持不同级别的射频信号粒度的研究。M4Human包含具有3D网格和全局轨迹的高质量动作捕捉(MoCap)标注，涵盖20个受试者和50种不同动作，包括原地动作、原地坐姿和自由空间运动或康复动作。我们在RT和RPC模态以及与RGB-D模态的多模态融合上建立了基准。广泛的结果突显了M4Human对于雷达基础人体建模的重要性，同时揭示了在快速、不受约束运动下仍然存在的挑战。该数据集和代码将在论文发表后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有人类网格重建(HMR)数据集过度依赖RGB视觉输入的问题，以及毫米波雷达数据集规模小、动作简单、标签质量有限的问题。这个问题很重要，因为视觉传感在隐私敏感场景(如老人和儿童护理)中存在隐私问题，且容易受光照变化和遮挡影响，而毫米波雷达能提供隐私保护、光照不变性和抗遮挡能力，对VR/AR、虚拟试衣、康复训练等多种应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有视觉传感和毫米波雷达在HMR中的局限性，设计了一个整合高分辨率毫米波雷达、RGB-D相机和Vicon动作捕捉系统的多模态平台。他们借鉴了现有动作捕捉技术进行高质量标注，参考了雷达数据处理方法(包括RT和RPC)，并采用了SMPL-X模型表示人体。同时，他们参考了现有评估指标，但创新性地结合了多种模态和高质量标注，构建了一个前所未有的大规模数据集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个大规模、高质量、多模态的毫米波雷达数据集，支持高保真度的RF-based HMR。实现流程包括：1)使用高分辨率雷达、RGB-D相机和Vicon系统采集多模态数据；2)通过PnP问题和雷达可见目标进行精确的空间校准；3)使用Vicon系统进行标记采集和人工清理；4)用SOMA神经网络重建SMPL-X风格人体网格；5)组织包含20个受试者和50类动作的多样化数据集；6)建立单模态和多模态融合的基准测试。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)构建了目前最大的毫米波雷达HMR数据集(661K帧，是之前最大数据集的9倍)；2)提供四种同步模态(RGB、深度、RT和RPC)；3)使用基于标记的Vicon系统提供高质量3D网格标注；4)包含50类多样化动作，突破简单原地动作限制；5)同时提供原始雷达张量(RT)和雷达点云(RPC)；6)提出首个直接从RT进行HMR的RT-Mesh方法。相比之前工作，M4Human在规模、动作复杂性、标注质量和数据模态上都有显著突破。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'M4Human是一个大规模、高质量、多模态的毫米波雷达数据集，通过提供原始雷达张量和高质量3D网格标注，突破了现有数据集在规模、动作复杂性和标注质量上的局限，为毫米波雷达在高保真度人类网格重建中的应用奠定了基础。'}


### 论文摘要

Human mesh reconstruction (HMR) provides direct insights into body-environment interaction, which enables various immersive applications. While existing large-scale HMR datasets rely heavily on line-of-sight RGB input, vision-based sensing is limited by occlusion, lighting variation, and privacy concerns. To overcome these limitations, recent efforts have explored radio-frequency (RF) mmWave radar for privacy-preserving indoor human sensing. However, current radar datasets are constrained by sparse skeleton labels, limited scale, and simple in-place actions. To advance the HMR research community, we introduce M4Human, the current largest-scale (661K-frame) ($9\times$ prior largest) multimodal benchmark, featuring high-resolution mmWave radar, RGB, and depth data. M4Human provides both raw radar tensors (RT) and processed radar point clouds (RPC) to enable research across different levels of RF signal granularity. M4Human includes high-quality motion capture (MoCap) annotations with 3D meshes and global trajectories, and spans 20 subjects and 50 diverse actions, including in-place, sit-in-place, and free-space sports or rehabilitation movements. We establish benchmarks on both RT and RPC modalities, as well as multimodal fusion with RGB-D modalities. Extensive results highlight the significance of M4Human for radar-based human modeling while revealing persistent challenges under fast, unconstrained motion. The dataset and code will be released after the paper publication.

---

## 17. Graph Contextual Reinforcement Learning for Efficient Directed Controller Synthesis

**论文链接:** [http://arxiv.org/abs/2512.15295v1](http://arxiv.org/abs/2512.15295v1)

**作者:** Toshihide Ubukata, Enhong Mu, Takuto Yamauchi, Mingyue Zhang, Jialong Li, Kenji Tei

**发布时间:** 2025-12-17

### GPT解析

### 总结

控制器合成是一种形式化方法，用于自动生成满足特定性质的标记转换系统控制器。本文提出GCRL方法，通过集成图神经网络增强基于强化学习的控制器合成方法，提高了学习效率和泛化能力。

### 背景

控制器合成的效率严重依赖于探索策略，现有方法通常依赖固定规则或仅考虑有限当前特征的强化学习策略，限制了性能提升。

### 目的

解决现有控制器合成方法中探索策略的局限性，开发一种能捕捉更广泛上下文信息的方法，提高合成效率和泛化能力。

### 方法

提出GCRL方法，将标记转换系统(LTS)的探索历史编码为图结构，利用图神经网络(GNN)捕捉非基于当前的环境上下文，增强强化学习方法的性能。

### 主要发现

在五个基准域的对比实验中，GCRL在四个域中表现出比最先进方法更优的学习效率和泛化能力，但在一个具有高度对称性和严格局部交互的特殊域中表现不佳。

### 结论

GCRL通过集成图神经网络和考虑探索历史，显著提高了控制器合成的效率和泛化能力，但在处理具有高度对称性和严格局部交互的系统时仍有改进空间。

### 翻译

控制器合成是一种形式化方法，用于自动生成满足特定性质的标记转换系统控制器。然而，合成过程的效率严重依赖于探索策略。这些策略通常依赖固定规则或通过强化学习学习的策略，仅考虑有限的当前特征集。为解决这一限制，本文引入GCRL，一种通过集成图神经网络增强基于RL的方法。GCRL将LTS探索历史编码为图结构，使其能够捕捉更广泛的非基于当前的环境上下文。在与最先进方法的对比实验中，GCRL在五个基准域中的四个表现出更优的学习效率和泛化能力，除一个具有高度对称性和严格局部交互的特殊域外。


### 论文摘要

Controller synthesis is a formal method approach for automatically generating Labeled Transition System (LTS) controllers that satisfy specified properties. The efficiency of the synthesis process, however, is critically dependent on exploration policies. These policies often rely on fixed rules or strategies learned through reinforcement learning (RL) that consider only a limited set of current features. To address this limitation, this paper introduces GCRL, an approach that enhances RL-based methods by integrating Graph Neural Networks (GNNs). GCRL encodes the history of LTS exploration into a graph structure, allowing it to capture a broader, non-current-based context. In a comparative experiment against state-of-the-art methods, GCRL exhibited superior learning efficiency and generalization across four out of five benchmark domains, except one particular domain characterized by high symmetry and strictly local interactions.

---

## 18. Accelerating High-Throughput Catalyst Screening by Direct Generation of Equilibrium Adsorption Structures

**论文链接:** [http://arxiv.org/abs/2512.15228v1](http://arxiv.org/abs/2512.15228v1)

**作者:** Songze Huo, Xiao-Ming Cao

**发布时间:** 2025-12-17

### GPT解析

### 总结

DBCata是一种深度生成模型，通过结合周期性布朗桥框架和等变图神经网络，能够在不明确需要能量或力信息的情况下，准确预测吸附结构和能量，比现有方法更准确，可用于加速催化剂筛选。

### 背景

目前广泛使用的机器学习原子间势(MLIP)的训练数据主要来自近平衡结构，分布有限，导致吸附结构和吸附能量预测不可靠。

### 目的

开发一个能够可靠预测吸附结构和能量的方法，用于大规模催化剂筛选。

### 方法

提出了DBCata，这是一个深度生成模型，结合了周期性布朗桥框架和等变图神经网络，建立了非弛豫结构和DFT弛豫结构之间的低维过渡流形，不需要明确的能量或力信息。

### 主要发现

DBCata在Catalysis-Hub数据集上达到0.035 Å的原子间距离平均绝对误差(DMAE)，比当前最先进的机器学习势模型好近三倍；通过结合化学启发式和自监督异常检测方法，在94%的情况下，相应的DFT精度可提高0.1 eV以内。

### 结论

DBCata的卓越性能促进了高效合金催化剂在氧还原反应中的加速高通量计算筛选，突显了DBCata作为催化剂设计和优化强大工具的潜力。

### 翻译

吸附能量作为大规模催化剂筛选的关键描述符，然而，广泛使用的机器学习原子间势(MLIP)的训练数据分布有限，主要来自近平衡结构，导致吸附结构和吸附能量预测不可靠。在此背景下，我们提出了DBCata，一个深度生成模型，它结合了周期性布朗桥框架和等变图神经网络，建立了非弛豫结构和DFT弛豫结构之间的低维过渡流形，不需要明确的能量或力信息。训练后，DBCata能有效生成高保真吸附几何结构，在Catalysis-Hub数据集上达到0.035 Å的原子间距离平均绝对误差(DMAE)，比当前最先进的机器学习势模型好近三倍。此外，通过结合化学启发式和自监督异常检测方法识别和优化异常预测，在94%的情况下，相应的DFT精度可提高0.1 eV以内。我们证明了DBCata的卓越性能促进了高效合金催化剂在氧还原反应中的加速高通量计算筛选，突显了DBCata作为催化剂设计和优化强大工具的潜力。


### 论文摘要

The adsorption energy serves as a crucial descriptor for the large-scale screening of catalysts. Nevertheless, the limited distribution of training data for the extensively utilised machine learning interatomic potential (MLIP), predominantly sourced from near-equilibrium structures, results in unreliable adsorption structures and consequent adsorption energy predictions. In this context, we present DBCata, a deep generative model that integrates a periodic Brownian-bridge framework with an equivariant graph neural network to establish a low-dimensional transition manifold between unrelaxed and DFT-relaxed structures, without requiring explicit energy or force information. Upon training, DBCata effectively generates high-fidelity adsorption geometries, achieving an interatomic distance mean absolute error (DMAE) of 0.035 \textÅ on the Catalysis-Hub dataset, which is nearly three times superior to that of the current state-of-the-art machine learning potential models. Moreover, the corresponding DFT accuracy can be improved within 0.1 eV in 94\% of instances by identifying and refining anomalous predictions through a hybrid chemical-heuristic and self-supervised outlier detection approach. We demonstrate that the remarkable performance of DBCata facilitates accelerated high-throughput computational screening for efficient alloy catalysts in the oxygen reduction reaction, highlighting the potential of DBCata as a powerful tool for catalyst design and optimisation.

---

## 19. RELIC-GNN: Efficient State Registers Identification with Graph Neural Network for Reverse Engineering

**论文链接:** [http://arxiv.org/abs/2512.15037v1](http://arxiv.org/abs/2512.15037v1)

**作者:** Weitao Pan, Meng Dong, Zhiliang Qiu, Jianlei Yang, Zhixiong Di, Yiming Gao

**发布时间:** 2025-12-17

### GPT解析

### 总结

RELIC-GNN是一种基于图神经网络的状态寄存器识别方法，用于门级网表逆向工程，能有效分离控制信号和数据信号。

### 背景

门级网表逆向工程对于硬件木马检测和设计盗版应对至关重要，现有方法通过拓扑比较识别状态寄存器分离控制信号和数据信号。

### 目的

提出一种高效的状态寄存器识别方法，解决现有方法在大规模网表中效率低下的问题。

### 方法

RELIC-GNN将寄存器的路径结构建模为图，在训练过程中考虑节点属性和图结构来生成相应表示，训练后的GNN模型可高效识别寄存器类型。

### 主要发现

RELIC-GNN在不同设计上平均达到100%的召回率、30.49%的精确度和88.37%的准确率，比之前的方法有显著改进。

### 结论

RELIC-GNN是一种高效的状态寄存器识别方法，能够有效解决大规模网表中的门级网表逆向工程问题。

### 翻译

门级网表的逆向工程对于硬件木马检测和设计盗版应对至关重要。门级逆向工程的主要任务是从网表中分离控制信号和数据信号，这主要通过识别具有拓扑比较的状态寄存器来实现。然而，这些方法对于大规模网表变得效率低下。在这项工作中，我们提出了RELIC-GNN，一种基于图神经网络的状态寄存器识别方法，以解决这些问题。RELIC-GNN将寄存器的路径结构建模为图，并在训练过程中通过考虑节点属性和图结构来生成相应的表示。训练后的GNN模型可以非常高效地找到寄存器类型。实验结果表明，RELIC-GNN在不同设计上平均可以达到100%的召回率、30.49%的精确度和88.37%的准确率，比之前的方法获得了显著的改进。


### 论文摘要

Reverse engineering of gate-level netlist is critical for Hardware Trojans detection and Design Piracy counteracting. The primary task of gate-level reverse engineering is to separate the control and data signals from the netlist, which is mainly realized by identifying state registers with topological comparison.However, these methods become inefficient for large scale netlist. In this work, we propose RELIC-GNN, a graph neural network based state registers identification method, to address these issues. RELIC-GNN models the path structure of register as a graph and generates corresponding representation by considering node attributes and graph structure during training. The trained GNN model could be adopted to find the registers type very efficiently. Experimental results show that RELIC-GNN could achieve 100% in recall, 30.49% in precision and 88.37% in accuracy on average across different designs, which obtains significant improvements than previous approaches.

---

## 20. ATLAS: Adaptive Topology-based Learning at Scale for Homophilic and Heterophilic Graphs

**论文链接:** [http://arxiv.org/abs/2512.14908v1](http://arxiv.org/abs/2512.14908v1)

**作者:** Turja Kundu, Sanjukta Bhowmick

**发布时间:** 2025-12-16

**备注:** Preprint

### GPT解析

### 总结

ATLAS是一种新颖的图学习算法，解决了图神经网络在异质图准确率下降和大规模图可扩展性方面的挑战，通过提取多级社区拓扑信息并应用多层感知器，实现了与基线方法相当的准确率。

### 背景

图神经网络在处理异质图时准确率会下降，且迭代特征聚合限制了其在大规模图上的可扩展性。

### 目的

解决GNNs在异质图上的准确率下降问题，提高GNNs在大规模图上的可扩展性。

### 方法

ATLAS通过提取多级细化的图社区拓扑信息，将社区分配连接到特征向量，并对得到的表示应用多层感知器(MLPs)，提供节点及其邻域的拓扑上下文而不需要聚合。

### 主要发现

ATLAS在广泛图集上实现了与基线方法相当的准确率；对于具有负结构偏置的异质图，比GCN提高高达20个百分点；对于同质图，比MLP提高11个百分点；多分辨率社区特征系统性地调节同质和异质设置中的性能。

### 结论

ATLAS为可解释的图学习提供了一条原则性的路径，通过多分辨率社区特征系统性地调节性能。

### 翻译

我们提出了ATLAS（针对同质图和异质图的基于自适应拓扑的大规模学习），这是一种新颖的图学习算法，解决了图神经网络(GNNs)中的两个重要挑战。首先，当图是异质图时，GNNs的准确率会下降。其次，迭代特征聚合限制了GNNs在大规模图上的可扩展性。我们通过提取多级细化的图社区拓扑信息，将社区分配连接到特征向量，并对得到的表示应用多层感知器(MLPs)来解决这些挑战。这提供了关于节点及其邻域的拓扑上下文，而不需要调用聚合。由于MLPs通常比GNNs更具可扩展性，我们的方法适用于大规模图而无需采样。在广泛的图集中，ATLAS实现了与基线方法相当的准确率，对于具有负结构偏置的异质图比GCN提高高达20个百分点，对于同质图比MLP提高11个百分点。此外，我们展示了多分辨率社区特征如何系统性地调节同质和异质设置中的性能，为可解释的图学习开辟了一条原则性的路径。


### 论文摘要

We present ATLAS (Adaptive Topology-based Learning at Scale for Homophilic and Heterophilic Graphs), a novel graph learning algorithm that addresses two important challenges in graph neural networks (GNNs). First, the accuracy of GNNs degrades when the graph is heterophilic. Second, iterative feature aggregation limits the scalability of GNNs to large graphs. We address these challenges by extracting topological information about graph communities at multiple levels of refinement, concatenating community assignments to the feature vector, and applying multilayer perceptrons (MLPs) to the resulting representation. This provides topological context about nodes and their neighborhoods without invoking aggregation. Because MLPs are typically more scalable than GNNs, our approach applies to large graphs without the need for sampling. Across a wide set of graphs, ATLAS achieves comparable accuracy to baseline methods, with gains as high as 20 percentage points over GCN for heterophilic graphs with negative structural bias and 11 percentage points over MLP for homophilic graphs. Furthermore, we show how multi-resolution community features systematically modulate performance in both homophilic and heterophilic settings, opening a principled path toward explainable graph learning.

---

## 21. A Roadmap for Applying Graph Neural Networks to Numerical Data: Insights from Cementitious Materials

**论文链接:** [http://arxiv.org/abs/2512.14855v1](http://arxiv.org/abs/2512.14855v1)

**作者:** Mahmuda Sharmin, Taihao Han, Jie Huang, Narayanan Neithalath, Gaurav Sant, Aditya Kumar

**发布时间:** 2025-12-16

### GPT解析

### 总结

本研究探索了图神经网络(GNN)在混凝土材料设计中的应用，通过将表格数据转换为图形表示，实现了与传统机器学习方法相当的性能，为水泥基材料的多模态和物理信息AI模型奠定了基础。

### 背景

机器学习在混凝土研究中应用日益增多，但面临可用数据库规模和多样性有限的挑战。传统ML框架通常局限于单一数据模态，而多模态数据库（结合数值和图形数据）是解决这一问题的有前景方案。

### 目的

开发一种能够利用图形结构数据的神经网络方法，建立将表格数据转换为图形表示的清晰且可复现的路径，为从传统ML向先进AI架构过渡提供基础路线图。

### 方法

采用图神经网络(GNN)架构，利用k近邻(k-NN)方法将表格数据转换为图形表示，系统优化模型超参数和特征选择以提高预测性能，并将物理定律直接嵌入GNN架构中。

### 主要发现

GNN能够捕获不规则或依赖于拓扑的连接关系，不仅适用于图形数据，还能从数值数据集中提取相关性。GNN性能与基准随机森林相当，后者已被证明对水泥基材料能产生可靠预测，且GNN提供了可解释的物理信息预测能力。

### 结论

本研究是首批实施GNN设计混凝土的研究之一，所提出的框架为未来的多模态和物理信息GNN模型奠定了坚实基础，这些模型能够捕获复杂的材料行为，并加速水泥基材料的设计和优化。

### 翻译

机器学习(ML)越来越多地应用于混凝土研究以优化性能和混合设计。然而，将ML应用于水泥基材料的主要挑战是可用数据库的规模和多样性有限。多模态数据库（结合数值和图形数据）是一个有前景的解决方案。传统ML框架在水泥研究中通常局限于单一数据模态。图神经网络(GNN)是新一代神经网络架构，能够从图形结构数据中学习，通过不规则或依赖于拓扑的连接关系捕获关系，而不仅仅是固定的空间坐标。虽然GNN专为图形数据设计，但可以调整以从数值数据集中提取相关性，并将物理定律直接嵌入其架构中，实现可解释的物理信息预测。本研究是首批实施GNN设计混凝土的研究之一，重点是建立使用k近邻(k-NN)方法将表格数据转换为图形表示的清晰且可复现的路径。系统优化模型超参数和特征选择以提高预测性能。GNN的性能与基准随机森林相当，后者已被许多研究证明对水泥基材料能产生可靠预测。总体而言，本研究为从传统ML向先进AI架构过渡提供了基础路线图。所提出的框架为未来的多模态和物理信息GNN模型奠定了坚实基础，这些模型能够捕获复杂的材料行为，并加速水泥基材料的设计和优化。


### 论文摘要

Machine learning (ML) has been increasingly applied in concrete research to optimize performance and mixture design. However, one major challenge in applying ML to cementitious materials is the limited size and diversity of available databases. A promising solution is the development of multi-modal databases that integrate both numerical and graphical data. Conventional ML frameworks in cement research are typically restricted to a single data modality. Graph neural network (GNN) represents a new generation of neural architectures capable of learning from data structured as graphs, capturing relationships through irregular or topology-dependent connections rather than fixed spatial coordinates. While GNN is inherently designed for graphical data, they can be adapted to extract correlations from numerical datasets and potentially embed physical laws directly into their architecture, enabling explainable and physics-informed predictions. This work is among the first few studies to implement GNNs to design concrete, with a particular emphasis on establishing a clear and reproducible pathway for converting tabular data into graph representations using the k-nearest neighbor (K-NN) approach. Model hyperparameters and feature selection are systematically optimized to enhance prediction performance. The GNN shows performance comparable to the benchmark random forest, which has been demonstrated by many studies to yield reliable predictions for cementitious materials. Overall, this study provides a foundational roadmap for transitioning from traditional ML to advanced AI architectures. The proposed framework establishes a strong foundation for future multi-modal and physics-informed GNN models capable of capturing complex material behaviors and accelerating the design and optimization of cementitious materials.

---

## 22. Dual-Axis RCCL: Representation-Complete Convergent Learning for Organic Chemical Space

**论文链接:** [http://arxiv.org/abs/2512.14418v2](http://arxiv.org/abs/2512.14418v2)

**作者:** Dejun Hu, Zhiming Li, Jia-Rui Shen, Jia-Ning Tu, Zi-Hao Ye, Junliang Zhang

**发布时间:** 2025-12-16

**备注:** 33 pages, 10 figures

### GPT解析

### 总结

该研究提出了一种双轴表示-完全收敛学习（RCCL）策略，通过结合图卷积网络和无桥图编码，实现了对广阔化学空间的近乎完全覆盖，并展示了分子表示、结构完整性和模型泛化之间的定量联系。

### 背景

机器学习正在深刻改变分子和材料建模，但化学空间极其庞大（10^30-10^60），模型能否在这个空间中实现收敛学习仍然是一个开放的科学问题。

### 目的

解决机器学习在广阔化学空间中能否实现收敛学习的科学问题，并建立分子表示与模型泛化之间的定量联系。

### 方法

引入RCCL策略，使用结合图卷积网络（GCN）编码的局部价环境和无桥图（NBG）编码的环/笼拓扑结构的分子表示方法；开发FD25数据集，覆盖13,302个局部价单位和165,726个环/笼拓扑结构，实现有机分子（H/C/N/O/F元素）的近乎完全组合覆盖。

### 主要发现

图神经网络在FD25上训练表现出表示完全收敛学习和强大的分布外泛化能力；在外部基准测试中，整体预测误差约为1.0 kcal/mol MAE；建立了分子表示、结构完整性和模型泛化之间的定量联系。

### 结论

该研究为可解释、可转移和数据高效的分子智能提供了基础。

### 翻译

机器学习正在深刻改变分子和材料建模；然而，鉴于化学空间的巨大规模（10^30-10^60），模型能否在这个空间中实现收敛学习仍然是一个开放的科学问题。我们引入了一种双轴表示-完全收敛学习（RCCL）策略，通过结合基于现代价键理论的图卷积网络（GCN）对局部价环境的编码，以及无桥图（NBG）对环/笼拓扑结构的编码，提供了一种化学空间覆盖的定量度量。该框架形式化了表示完整性，为构建支持大模型收敛学习的数据集奠定了原则性基础。在该RCCL框架指导下，我们开发了FD25数据集，系统覆盖了13,302个局部价单位和165,726个环/笼拓扑结构，实现了含H/C/N/O/F元素的有机分子的近乎完全组合覆盖。在FD25上训练的图神经网络表现出表示完全收敛学习和强大的分布外泛化能力，在外部基准测试中整体预测误差约为1.0 kcal/mol MAE。我们的结果建立了分子表示、结构完整性和模型泛化之间的定量联系，为可解释、可转移和数据高效的分子智能奠定了基础。


### 论文摘要

Machine learning is profoundly reshaping molecular and materials modeling; however, given the vast scale of chemical space (10^30-10^60), it remains an open scientific question whether models can achieve convergent learning across this space. We introduce a Dual-Axis Representation-Complete Convergent Learning (RCCL) strategy, enabled by a molecular representation that integrates graph convolutional network (GCN) encoding of local valence environments, grounded in modern valence bond theory, together with no-bridge graph (NBG) encoding of ring/cage topologies, providing a quantitative measure of chemical-space coverage. This framework formalizes representation completeness, establishing a principled basis for constructing datasets that support convergent learning for large models. Guided by this RCCL framework, we develop the FD25 dataset, systematically covering 13,302 local valence units and 165,726 ring/cage topologies, achieving near-complete combinatorial coverage of organic molecules with H/C/N/O/F elements. Graph neural networks trained on FD25 exhibit representation-complete convergent learning and strong out-of-distribution generalization, with an overall prediction error of approximately 1.0 kcal/mol MAE across external benchmarks. Our results establish a quantitative link between molecular representation, structural completeness, and model generalization, providing a foundation for interpretable, transferable, and data-efficient molecular intelligence.

---

## 23. EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.15160v1](http://arxiv.org/abs/2512.15160v1)

**作者:** Jiaxu Wan, Xu Wang, Mengwei Xie, Hang Zhang, Mu Xu, Yang Han, Hong Zhang, Ding Yuan, Yifan Yang

**发布时间:** 2025-12-17

**备注:** 13 pages, 7 figures, 6 tables

### GPT解析

### 总结

EagleVision是一种双阶段空间智能框架，通过宏观感知和微观验证解决空间思维链中的关键挑战，在开源视觉语言模型中实现了最先进的性能。

### 背景

现有的空间智能方法通常将3D线索附加到2D推理流程中，或者将多模态大语言模型与黑盒重建模块耦合，导致空间一致性弱，视点多样性有限，证据链无法追溯到支持视图。'图像思维'框架虽然展示了逐步多模态推理的能力，但未解决空间思维链中的三个关键挑战。

### 目的

解决空间思维链中的三个关键挑战：在严格的token预算下建立全局空间感知；明确将3D假设与视频帧关联以进行验证；为强化学习设计空间基础奖励。

### 方法

提出EagleVision双阶段框架：1) 宏观感知阶段：使用语义-视角融合确定性点过程在固定token预算下从长视频中选择几何和语义感知关键帧；2) 微观验证阶段：将空间思维链形式化为BEV基础姿态查询，代理迭代预测姿态，检索最近实际帧，并通过空间基础奖励进行强化学习训练。

### 主要发现

在VSI-Bench基准测试上，EagleVision在开源视觉语言模型中实现了最先进的性能，展示了强大且可泛化的空间理解能力。

### 结论

EagleVision通过双阶段框架有效解决了空间思维链中的关键挑战，在空间智能任务上表现出色，具有强大的泛化能力。

### 翻译

最近的空间智能方法通常将3D线索附加到2D推理流程中，或将多模态大语言模型与黑盒重建模块耦合，导致空间一致性弱，视点多样性有限，且证据链无法追溯到支持视图。'图像思维'框架（如ChatGPT-o3和DeepEyes）表明，通过交错假设形成和主动获取视觉证据，可以逐步出现多模态推理，但它们没有解决空间思维链中的三个关键挑战：在严格的token预算下建立全局空间感知，明确将3D假设与视频帧关联以进行验证，以及为强化学习设计空间基础奖励。为解决这些问题，我们提出了EagleVision，一个通过宏观感知和微观验证进行渐进式空间认知的双阶段框架。在宏观感知阶段，EagleVision使用语义-视角融合确定性点过程在固定token预算下从长视频中选择紧凑的几何和语义感知关键帧集合。在微观验证阶段，我们将空间思维链形式化为BEV基础姿态查询：代理在BEV平面上迭代预测姿态，检索最近的实际帧，并通过纯粹的空间基础奖励强化学习进行训练，该奖励评分预测姿态与观察视图之间的一致性。在VSI-Bench上，EagleVision在开源视觉语言模型中实现了最先进的性能，展示了强大且可泛化的空间理解能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决空间智能中的三个关键问题：空间一致性弱、视角多样性有限以及证据链无法追溯到支持视图。这些问题在现实中很重要，因为空间智能对于具身感知和多模态理解至关重要，而现有方法未能充分利用空间作为主动推理工作区，难以在有限资源下构建全局空间理解，也缺乏有效的机制将抽象的3D假设与具体视觉证据关联起来。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析现有空间智能方法的局限性，识别出空间链式思维的三大挑战：全局空间感知构建、3D假设与视频帧的关联、空间奖励设计。他们借鉴了'thinking with images'的方法论，如ChatGPT-o3和DeepEyes，并使用了SLAM系统(Vipe)、FG-CLIP视觉语言模型、确定性点过程(DPP)和强化学习(GRPO)等技术。在此基础上，他们创新性地设计了双阶段框架：宏观感知使用SPF-DPP选择关键帧，微观验证实现BEV基础的链式思维。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将空间推理分解为宏观感知和微观验证两个阶段，使用BEV表示作为共同坐标系，并通过主动查询机制获取额外视觉证据形成可验证证据链。宏观感知阶段使用SLAM估计相机姿态和深度，投影到BEV平面，结合语义相关性通过SPF-DPP选择关键帧；微观验证阶段维护推理状态，通过BEV定位工具主动查询特定视角的帧，添加到证据集合中，使用强化学习训练查询策略。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：双阶段框架设计、SPF-DPP算法、BEV基础的空间链式思维形式化、空间奖励机制、纯强化学习训练方法。相比之前工作，它不仅解决了空间一致性和视角多样性问题，还建立了可追溯的证据链，明确将3D假设与视觉证据关联，并在严格token预算下实现了高效的空间推理，无需依赖黑盒3D重建模块。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'EagleVision通过双阶段框架结合基于BEV的链式思维，实现了高效的空间智能推理，在VSI-Bench上取得了开源视觉语言模型的最先进性能。'}


### 论文摘要

Recent spatial intelligence approaches typically attach 3D cues to 2D reasoning pipelines or couple MLLMs with black-box reconstruction modules, leading to weak spatial consistency, limited viewpoint diversity, and evidence chains that cannot be traced back to supporting views. Frameworks for "thinking with images" (e.g., ChatGPT-o3 and DeepEyes) show that stepwise multimodal reasoning can emerge by interleaving hypothesis formation with active acquisition of visual evidence, but they do not address three key challenges in spatial Chain-of-Thought (CoT): building global space perception under strict token budgets, explicitly associating 3D hypotheses with video frames for verification, and designing spatially grounded rewards for reinforcement learning. To address these issues, we present EagleVision, a dual-stage framework for progressive spatial cognition through macro perception and micro verification. In the macro perception stage, EagleVision employs a semantics-perspective-fusion determinantal point process (SPF-DPP) to select a compact set of geometry- and semantics-aware keyframes from long videos under a fixed token budget. In the micro verification stage, we formalize spatial CoT as BEV-grounded pose querying: the agent iteratively predicts poses on a BEV plane, retrieves the nearest real frames, and is trained purely by reinforcement learning with a spatial grounding reward that scores the consistency between predicted poses and observed views. On VSI-Bench, EagleVision achieves state-of-the-art performance among open-source vision-language models, demonstrating strong and generalizable spatial understanding.

---

## 24. HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles

**论文链接:** [http://arxiv.org/abs/2512.15047v1](http://arxiv.org/abs/2512.15047v1)

**作者:** Yunheng Wang, Yixiao Feng, Yuetong Fang, Shuning Zhang, Tan Jing, Jian Li, Xiangrui Jiang, Renjing Xu

**发布时间:** 2025-12-17

### GPT解析

### 总结

论文介绍HERO框架，用于构建层次化可遍历3D场景图，解决了现有方法在处理可操作障碍物方面的局限性，提高了导航效率和可达性。

### 背景

3D场景图是对物理世界的强大表示，能明确建模实体间复杂的空间、语义和功能关系。具身导航利用3DSGs实现长时程推理和规划，但先前工作依赖静态世界假设，仅基于静态空间布局定义可遍历空间，将可交互障碍物视为不可遍历，限制了在真实世界场景中的有效性。

### 目的

解决现有3D场景图导航方法在处理可操作障碍物方面的局限性，提高导航系统的效率、可达性和扩展性。

### 方法

提出HERO框架构建层次化可遍历3D场景图，将可操作障碍物建模为路径，捕捉它们的物理交互性、功能语义和场景的层次关系来重新定义可遍历性。

### 主要发现

与基线相比，HERO在部分阻塞环境中将路径长度减少了35.1%，在完全阻塞环境中将成功率提高了79.4%，显示出更高的效率和可达性。

### 结论

HERO框架通过更好地建模可操作障碍物，显著提高了3D场景图导航系统的性能，特别是在充满障碍物的环境中。

### 翻译

3D场景图是对物理世界的强大表示，其特点是能够明确建模实体之间复杂的空间、语义和功能关系，提供了基础理解，使智能体能够与其环境进行智能交互并执行多样化行为。具身导航作为此类能力的关键组成部分，利用3DSGs的紧凑和表达性，在复杂大规模环境中实现长时程推理和规划。然而，先前的工作依赖于静态世界假设，仅基于静态空间布局定义可遍历空间，从而将可交互障碍物视为不可遍历。这一基本限制严重削弱了它们在真实世界场景中的有效性，导致有限的可达性、低效率和较差的扩展性。为解决这些问题，我们提出了HERO，一个用于构建层次化可遍历3D场景图的新框架，它通过将可操作障碍物建模为路径，捕捉它们的物理交互性、功能语义和场景的层次关系来重新定义可遍历性。结果显示，与基线相比，HERO在部分阻塞环境中将路径长度减少了35.1%，在完全阻塞环境中将成功率提高了79.4%，显示出更高的效率和可达性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决现有3D场景图在导航任务中的局限性，即它们基于'静态世界假设'，将可交互的障碍物(如门、窗帘、可移动家具)视为不可通行区域。这个问题在现实中很重要，因为传统方法会导致机器人导航能力受限，表现为：有限的可达性(某些物理可达区域因障碍物被认为不可达)、低效率(过于保守的避障规划导致不必要的绕路)和差的可扩展性(无法完成需要与物体交互的复杂任务)。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类导航行为获得灵感：人类不会将所有阻挡物视为绝对障碍，而是评估物体属性和潜在可用性。作者借鉴了现有工作：基于H-3DSGs的层次结构(如HOV-SG)、Voronoi导航图、语义分割技术(SAM)、可视性净化策略和拓扑聚类策略。创新点在于将可移动障碍物整合到导航图中，并基于效率驱动的角度重新定义可通行性，而不仅仅是基于物理属性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是构建'可通行的分层3D场景图'，它不仅捕捉场景结构和语义，还明确建模物体的交互性和可移动性，将可移动障碍物视为潜在通路而非绝对障碍。实现流程分三阶段：1)粗粒度场景图构建(几何分解为楼层-房间结构，应用可见性净化策略)；2)细粒度场景图构建(拓扑聚类策略聚合物体节点，恢复语义信息)；3)可通行拓扑图构建(可通行性更新策略，识别可移动物体并整合到导航图中)。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)可通行的分层3D场景图表示，明确建模物体交互属性；2)可见性净化策略解决跨房间语义污染；3)拓扑聚类策略利用3D拓扑完整性聚合物体；4)可通行性更新策略基于效率评估确定物体可移动性。相比之前工作：传统方法将障碍物视为静态不可通行，而HERO将其视为潜在通路；传统方法产生保守的避障路径，而HERO重新定义可通行区域提高效率；传统场景图缺乏交互性建模，而HERO的表示支持与物体交互的导航决策。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HERO通过构建可通行的分层3D场景图，将可移动障碍物整合为潜在通路而非绝对障碍，显著提高了机器人在复杂环境中的导航可达性和效率。'}


### 论文摘要

3D Scene Graphs (3DSGs) constitute a powerful representation of the physical world, distinguished by their abilities to explicitly model the complex spatial, semantic, and functional relationships between entities, rendering a foundational understanding that enables agents to interact intelligently with their environment and execute versatile behaviors. Embodied navigation, as a crucial component of such capabilities, leverages the compact and expressive nature of 3DSGs to enable long-horizon reasoning and planning in complex, large-scale environments. However, prior works rely on a static-world assumption, defining traversable space solely based on static spatial layouts and thereby treating interactable obstacles as non-traversable. This fundamental limitation severely undermines their effectiveness in real-world scenarios, leading to limited reachability, low efficiency, and inferior extensibility. To address these issues, we propose HERO, a novel framework for constructing Hierarchical Traversable 3DSGs, that redefines traversability by modeling operable obstacles as pathways, capturing their physical interactivity, functional semantics, and the scene's relational hierarchy. The results show that, relative to its baseline, HERO reduces PL by 35.1% in partially obstructed environments and increases SR by 79.4% in fully obstructed ones, demonstrating substantially higher efficiency and reachability.

---

## 25. Beyond Accuracy: A Geometric Stability Analysis of Large Language Models in Chess Evaluation

**论文链接:** [http://arxiv.org/abs/2512.15033v1](http://arxiv.org/abs/2512.15033v1)

**作者:** Xidan Song, Weiqi Wang, Ruifeng Cao, Qingya Hu

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出几何稳定性框架评估大型语言模型在复杂推理领域的真实理解能力，发现高准确率模型在几何变换下表现极差，表明其可能依赖模式匹配而非抽象空间逻辑。

### 背景

在复杂推理领域评估大型语言模型通常依赖于与真实标准的一致性表现。在国际象棋领域，这表现为与Stockfish等强大引擎的准确率基准测试。然而，高标量准确率并不一定意味着稳健的概念理解。

### 目的

解决标准准确率指标无法区分真实几何推理和规范棋盘状态的表面记忆这一问题，提出一种新的评估方法来更全面地评估模型的推理能力。

### 方法

提出几何稳定性框架，一种新颖的评估方法，严格测试模型在不变变换下的一致性，包括棋盘旋转、镜像对称、颜色反转和格式转换。使用约3000个位置的数据集对六种最先进的LLM（包括GPT-5.1、Claude Sonnet 4.5和Kimi K2 Turbo）进行比较分析。

### 主要发现

发现了显著的准确率-稳定性悖论。GPT-5.1等模型在标准位置上接近最佳准确率，但在几何扰动下表现出灾难性退化，特别是在旋转任务中错误率激增600%以上。相反，Claude Sonnet 4.5和Kimi K2 Turbo表现出优越的双重稳健性，在所有变换轴上都保持高度一致性。此外，Gemini 2.5 Flash在拒绝非法状态方面表现最佳（96.0%）。

### 结论

几何稳定性为AI评估提供了正交且必要的指标，为分离大规模模型的推理能力、数据污染和过拟合提供了必要的代理。

### 翻译

在复杂推理领域评估大型语言模型通常依赖于与真实标准的一致性表现。在国际象棋领域，这一标准表现为与Stockfish等强大引擎的准确率基准测试。然而，高标量准确率并不一定意味着稳健的概念理解。本文认为标准准确率指标无法区分真实的几何推理和规范棋盘状态的表面记忆。为解决这一差距，我们提出了几何稳定性框架，一种新颖的评估方法，严格测试模型在不变变换下的一致性，包括棋盘旋转、镜像对称、颜色反转和格式转换。我们将此框架应用于六种最先进的大型语言模型的比较分析，包括GPT-5.1、Claude Sonnet 4.5和Kimi K2 Turbo，使用了约3000个位置的数据集。我们的结果揭示了显著的准确率-稳定性悖论。虽然GPT-5.1等模型在标准位置上实现了接近最佳的准确率，但在几何扰动下表现出灾难性退化，特别是在旋转任务中错误率激增600%以上。这种差异表明了对模式匹配而非抽象空间逻辑的依赖。相反，Claude Sonnet 4.5和Kimi K2 Turbo表现出优越的双重稳健性，在所有变换轴上都保持高度一致性。此外，我们分析了有用性和安全性之间的权衡，确定Gemini 2.5 Flash在拒绝非法状态方面处于领先地位（96.0%）。我们得出结论，几何稳定性为AI评估提供了正交且必要的指标，为分离大规模模型的推理能力、数据污染和过拟合提供了必要的代理。


### 论文摘要

The evaluation of Large Language Models (LLMs) in complex reasoning domains typically relies on performance alignment with ground-truth oracles. In the domain of chess, this standard manifests as accuracy benchmarks against strong engines like Stockfish. However, high scalar accuracy does not necessarily imply robust conceptual understanding. This paper argues that standard accuracy metrics fail to distinguish between genuine geometric reasoning and the superficial memorization of canonical board states. To address this gap, we propose a Geometric Stability Framework, a novel evaluation methodology that rigorously tests model consistency under invariant transformations-including board rotation, mirror symmetry, color inversion, and format conversion. We applied this framework to a comparative analysis of six state-of-the-art LLMs including GPT-5.1, Claude Sonnet 4.5, and Kimi K2 Turbo, utilizing a dataset of approximately 3,000 positions. Our results reveal a significant Accuracy-Stability Paradox. While models such as GPT-5.1 achieve near-optimal accuracy on standard positions, they exhibit catastrophic degradation under geometric perturbation, specifically in rotation tasks where error rates surge by over 600%. This disparity suggests a reliance on pattern matching over abstract spatial logic. Conversely, Claude Sonnet 4.5 and Kimi K2 Turbo demonstrate superior dual robustness, maintaining high consistency across all transformation axes. Furthermore, we analyze the trade-off between helpfulness and safety, identifying Gemini 2.5 Flash as the leader in illegal state rejection (96.0%). We conclude that geometric stability provides an orthogonal and essential metric for AI evaluation, offering a necessary proxy for disentangling reasoning capabilities from data contamination and overfitting in large-scale models.

---

## 26. An updated efficient galaxy morphology classification model based on ConvNeXt encoding with UMAP dimensionality reduction

**论文链接:** [http://arxiv.org/abs/2512.15137v1](http://arxiv.org/abs/2512.15137v1)

**作者:** Guanwen Fang, Shiwei Zhu, Jun Xu, Shiying Lu, Chichun Zhou, Yao Dai, Zesen Lin, Xu Kong

**发布时间:** 2025-12-17

### GPT解析

### 总结

研究团队改进了原有的USmorph分类框架，通过结合预训练的ConvNeXt卷积神经网络和UMAP非线性流形学习，提高了星系形态分类的效率，使其适用于大规模巡天项目。

### 背景

星系形态分类在天文学中具有重要意义，特别是在大规模巡天时代，需要高效自动化的分类方法。

### 目的

开发增强的无监督机器学习模块，提高星系形态分类效率，以适应中国空间站望远镜(CSST)等大型巡天项目。

### 方法

使用预训练的ConvNeXt CNN进行分层特征提取（迁移学习），结合UMAP进行拓扑感知的降维；将算法识别的20个聚类合并为5个物理形态类型；应用于红移0.2<z<1.2的99,806个COSMOS星系的I波段图像。

### 主要发现

聚类数量从50优化为20，实现显著计算节省；约51%的星系被成功分类；对大质量星系的形态参数测试表明分类结果与星系演化理论高度一致。

### 结论

改进的算法显著提高了星系形态分类效率，适合大规模巡天项目如中国空间站望远镜(CSST)。

### 翻译

我们展示了在之前的USmorph分类框架内增强的无监督机器学习模块，包含两个组成部分：通过使用迁移学习的预训练ConvNeXt卷积神经网络进行分层特征提取，以及使用均匀流形近似和投影进行非线性流形学习实现拓扑感知的降维。我们将升级后的UML应用于红移0.2<z<1.2且I波段星等小于25的99,806个COSMOS星系的I波段图像。预定义的聚类数量优化为20（从原始框架的50减少），实现了显著的计算节省。20个算法识别的聚类被合并为五个物理形态类型。约51%的星系被成功分类。我们的分类结果与星系演化理论高度一致。这种改进的算法显著提高了星系形态分类的效率，适合中国空间站望远镜等计划进行的大规模巡天项目。


### 论文摘要

We present an enhanced unsupervised machine learning (UML) module within our previous \texttt{USmorph} classification framework featuring two components: (1) hierarchical feature extraction via a pre-trained ConvNeXt convolutional neural network (CNN) with transfer learning, and (2) nonlinear manifold learning using Uniform Manifold Approximation and Projection (UMAP) for topology-aware dimensionality reduction. This dual-stage design enables efficient knowledge transfer from large-scale visual datasets while preserving morphological pattern geometry through UMAP's neighborhood preservation. We apply the upgraded UML on I-band images of 99,806 COSMOS galaxies at redshift $0.2<z<1.2$ (to ensure rest-frame optical morphology) with $I_{\mathrm{mag}}<25$. The predefined cluster number is optimized to 20 (reduced from 50 in the original framework), achieving significant computational savings. The 20 algorithmically identified clusters are merged into five physical morphology types. About 51\% of galaxies (50,056) were successfully classified. To assess classification effectiveness, we tested morphological parameters for massive galaxies with $M_{*}>10^{9}~M_{\odot}$. Our classification results align well with galaxy evolution theory. This improved algorithm significantly enhances galaxy morphology classification efficiency, making it suitable for large-scale sky surveys such as those planned with the China Space Station Telescope (CSST).

---

## 27. See It Before You Grab It: Deep Learning-based Action Anticipation in Basketball

**论文链接:** [http://arxiv.org/abs/2512.15386v1](http://arxiv.org/abs/2512.15386v1)

**作者:** Arnau Barrera Roy, Albert Clapés Sintes

**发布时间:** 2025-12-17

### GPT解析

### 总结

这篇论文引入了篮球广播视频中的动作预测任务，专注于预测投篮后哪个队伍将获得球权。研究团队构建了一个包含10万个篮球视频片段、超过300小时视频和2000多个手动标注的篮板事件的新数据集，并使用最先进的动作预测方法提供了全面的基线结果。

### 背景

计算机视觉和视频理解已经通过从广播录像中实现大规模、自动化的比赛动态分析，彻底改变了体育分析领域。尽管在球员和球跟踪、姿态估计、动作定位和自动犯规识别方面取得了显著进展，但在体育视频中预测动作发生之前的研究相对较少。

### 目的

引入篮球广播视频中的动作预测任务，专注于预测投篮后哪个队伍将获得球权，并建立这一任务的基准数据集，探索篮板分类和篮板检测等互补任务。

### 方法

构建了一个包含10万个篮球视频片段、超过300小时视频和2000多个手动标注的篮板事件的新数据集，并使用最先进的动作预测方法提供了全面的基线结果，代表了深度学习技术首次应用于篮球篮板预测。

### 主要发现

实验结果突显了预测篮板的可行性和内在挑战，为动态多代理体育场景的预测建模提供了有价值的见解。通过预测篮板发生前的队伍球权，使实时自动广播和赛后分析工具支持决策的应用成为可能。

### 结论

这项工作通过预测篮板发生前的队伍球权，为实时自动广播和赛后分析工具支持决策的应用奠定了基础，为动态多代理体育场景的预测建模提供了有价值的见解。

### 翻译

计算机视觉和视频理解已经通过从广播录像中实现大规模、自动化的比赛动态分析，彻底改变了体育分析领域。尽管在球员和球跟踪、姿态估计、动作定位和自动犯规识别方面取得了显著进展，但在体育视频中预测动作发生之前的研究相对较少。这项工作引入了篮球广播视频中的动作预测任务，专注于预测投篮后哪个队伍将获得球权。为了建立这一任务的基准，展示了一个新的自建数据集，包含10万个篮球视频片段、超过300小时视频和2000多个手动标注的篮板事件。研究团队使用最先进的动作预测方法提供了全面的基线结果，代表了深度学习技术首次应用于篮球篮板预测。此外，还探索了两个互补任务：篮板分类和篮板检测，展示该数据集支持广泛的篮球视频理解应用，目前没有类似的数据集存在。实验结果突显了预测篮板的可行性和内在挑战，为动态多代理体育场景的预测建模提供了有价值的见解。通过预测篮板发生前的队伍球权，这项工作使实时自动广播和赛后分析工具支持决策的应用成为可能。


### 论文摘要

Computer vision and video understanding have transformed sports analytics by enabling large-scale, automated analysis of game dynamics from broadcast footage. Despite significant advances in player and ball tracking, pose estimation, action localization, and automatic foul recognition, anticipating actions before they occur in sports videos has received comparatively little attention. This work introduces the task of action anticipation in basketball broadcast videos, focusing on predicting which team will gain possession of the ball following a shot attempt. To benchmark this task, a new self-curated dataset comprising 100,000 basketball video clips, over 300 hours of footage, and more than 2,000 manually annotated rebound events is presented. Comprehensive baseline results are reported using state-of-the-art action anticipation methods, representing the first application of deep learning techniques to basketball rebound prediction. Additionally, two complementary tasks, rebound classification and rebound spotting, are explored, demonstrating that this dataset supports a wide range of video understanding applications in basketball, for which no comparable datasets currently exist. Experimental results highlight both the feasibility and inherent challenges of anticipating rebounds, providing valuable insights into predictive modeling for dynamic multi-agent sports scenarios. By forecasting team possession before rebounds occur, this work enables applications in real-time automated broadcasting and post-game analysis tools to support decision-making.

---

## 28. Explainable Action Form Assessment by Exploiting Multimodal Chain-of-Thoughts Reasoning

**论文链接:** [http://arxiv.org/abs/2512.15153v1](http://arxiv.org/abs/2512.15153v1)

**作者:** Mengshi Qi, Yeteng Wu, Xianlin Zhang, Huadong Ma

**发布时间:** 2025-12-17

### GPT解析

### 总结

本研究提出了一种新的人体动作形式评估（AFA）任务和相关数据集CoT-AFA，以及一个名为可解释性健身评估器的框架，用于评估人类动作的标准化程度并提供解释和解决方案。

### 背景

评估人类行为标准化并提供反馈在现实场景中重要但具挑战性，当前视频理解方法主要关注行为是什么和在哪里，无法满足需求；现有数据集缺乏表示行为标准化程度的标签，行为质量评估数据集缺乏可解释性和详细反馈。

### 目的

定义新的AFA任务，创建包含健身和武术视频的多样化数据集CoT-AFA，并提出可解释性健身评估器框架，用于判断动作、解释原因并提供解决方案。

### 方法

创建包含多级注释的CoT-AFA数据集，采用Chain-of-Thought解释范式提供完整推理过程；提出可解释性健身评估器框架，该框架采用两个并行处理流和动态门控机制融合视觉和语义信息。

### 主要发现

实验结果表明，该方法在解释生成（CIDEr提高+16.0%）、动作分类（准确率提高+2.7%）和质量评估（准确率提高+2.1%）方面取得了改进，展示了CoT-AFA对未来研究的巨大潜力。

### 结论

CoT-AFA数据集和可解释性健身评估器框架为动作标准化评估提供了有效解决方案，数据集和源代码已公开可用。

### 翻译

评估人类行为是否标准并提供合理反馈以改进行为标准化在现实场景中非常重要但具有挑战性。然而，当前视频理解方法主要关注行为是什么和在哪里，无法满足需求。同时，大多数现有数据集缺乏表示行为标准化程度的标签，而行为质量评估数据集缺乏可解释性和详细反馈。因此，我们定义了一个新的人体动作形式评估（AFA）任务，并引入了一个新的多样化数据集CoT-AFA，其中包含大量健身和武术视频，具有多级注释用于全面视频分析。我们通过新颖的思维链解释范式丰富了CoT-AFA数据集。与提供孤立反馈不同，我们的解释提供了完整的推理过程——从识别动作步骤到分析结果并提出具体解决方案。此外，我们提出了一个名为可解释性健身评估器的框架，不仅可以判断动作，还可以解释原因并提供解决方案。该框架采用两个并行处理流和动态门控机制来融合视觉和语义信息，从而提高其分析能力。实验结果表明，我们的方法在解释生成（例如，CIDEr提高+16.0%）、动作分类（准确率提高+2.7%）和质量评估（准确率提高+2.1%）方面取得了改进，揭示了CoT-AFA对未来研究的巨大潜力。我们的数据集和源代码可在https://github.com/MICLAB-BUPT/EFA获取。


### 论文摘要

Evaluating whether human action is standard or not and providing reasonable feedback to improve action standardization is very crucial but challenging in real-world scenarios. However, current video understanding methods are mainly concerned with what and where the action is, which is unable to meet the requirements. Meanwhile, most of the existing datasets lack the labels indicating the degree of action standardization, and the action quality assessment datasets lack explainability and detailed feedback. Therefore, we define a new Human Action Form Assessment (AFA) task, and introduce a new diverse dataset CoT-AFA, which contains a large scale of fitness and martial arts videos with multi-level annotations for comprehensive video analysis. We enrich the CoT-AFA dataset with a novel Chain-of-Thought explanation paradigm. Instead of offering isolated feedback, our explanations provide a complete reasoning process--from identifying an action step to analyzing its outcome and proposing a concrete solution. Furthermore, we propose a framework named Explainable Fitness Assessor, which can not only judge an action but also explain why and provide a solution. This framework employs two parallel processing streams and a dynamic gating mechanism to fuse visual and semantic information, thereby boosting its analytical capabilities. The experimental results demonstrate that our method has achieved improvements in explanation generation (e.g., +16.0% in CIDEr), action classification (+2.7% in accuracy) and quality assessment (+2.1% in accuracy), revealing great potential of CoT-AFA for future studies. Our dataset and source code is available at https://github.com/MICLAB-BUPT/EFA.

---

## 29. FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation

**论文链接:** [http://arxiv.org/abs/2512.15116v1](http://arxiv.org/abs/2512.15116v1)

**作者:** Runze Li, Hanchen Wang, Wenjie Zhang, Binghao Li, Yu Zhang, Xuemin Lin, Ying Zhang

**发布时间:** 2025-12-17

**备注:** This work has been submitted to the IEEE for possible publication. 15 pages, 8 figures

### GPT解析

### 总结

本文提出了一种名为FADTI的基于扩散的框架，用于多变量时间序列插值，通过傅里叶偏置投影模块注入频率感知特征调制，结合自注意力和门控卷积进行时间建模，在多个基准测试中表现优异。

### 背景

多变量时间序列插值在医疗保健、交通预测和生物建模等领域至关重要，传感器故障和不规则采样导致数据普遍存在缺失值。

### 目的

解决现有Transformer和扩散模型在处理结构化缺失模式和分布变化时泛化能力不足的问题，提高高缺失率情况下的插值性能。

### 方法

提出FADTI框架，通过可学习的傅里叶偏置投影（FBP）模块注入频率感知特征调制，结合自注意力和门控卷积进行时间建模，FBP支持多种频谱基以编码平稳和非平稳模式。

### 主要发现

在多个基准测试（包括新引入的生物时间序列数据集）上，FADTI始终优于最先进的方法，特别是在高缺失率情况下表现优异。

### 结论

通过将频域归纳偏置注入生成插值过程，FADTI有效提升了多变量时间序列插值的性能和鲁棒性。

### 翻译

多变量时间序列插值在医疗保健、交通预测和生物建模等应用中是基础性的，其中传感器故障和不规则采样导致普遍的缺失值。然而，现有的基于Transformer和扩散的模型缺乏明确的归纳偏置和频率感知能力，限制了它们在结构化缺失模式和分布变化情况下的泛化能力。我们提出了FADTI，一个基于扩散的框架，它通过可学习的傅里叶偏置投影（FBP）模块注入频率感知的特征调制，并通过自注意力和门控卷积与时间建模相结合。FBP支持多种频谱基，能够自适应编码平稳和非平稳模式。这种设计将频域归纳偏置注入到生成插值过程中。在多个基准测试（包括新引入的生物时间序列数据集）上的实验表明，FADTI始终优于最先进的方法，特别是在高缺失率情况下。代码可在https://anonymous.4open.science/r/TimeSeriesImputation-52BF获取。


### 论文摘要

Multivariate time series imputation is fundamental in applications such as healthcare, traffic forecasting, and biological modeling, where sensor failures and irregular sampling lead to pervasive missing values. However, existing Transformer- and diffusion-based models lack explicit inductive biases and frequency awareness, limiting their generalization under structured missing patterns and distribution shifts. We propose FADTI, a diffusion-based framework that injects frequency-informed feature modulation via a learnable Fourier Bias Projection (FBP) module and combines it with temporal modeling through self-attention and gated convolution. FBP supports multiple spectral bases, enabling adaptive encoding of both stationary and non-stationary patterns. This design injects frequency-domain inductive bias into the generative imputation process. Experiments on multiple benchmarks, including a newly introduced biological time series dataset, show that FADTI consistently outperforms state-of-the-art methods, particularly under high missing rates. Code is available at https://anonymous.4open.science/r/TimeSeriesImputation-52BF

---

## 30. LADY: Linear Attention for Autonomous Driving Efficiency without Transformers

**论文链接:** [http://arxiv.org/abs/2512.15038v1](http://arxiv.org/abs/2512.15038v1)

**作者:** Jihao Huang, Xi Xia, Zhiyuan Li, Tianle Liu, Jingke Wang, Junbo Chen, Tengju Ye

**发布时间:** 2025-12-17

**备注:** Under review

### GPT解析

### 总结

本文提出了LADY，首个基于完全线性注意力的生成模型，用于端到端自动驾驶，解决了Transformer架构的二次方注意力成本问题，实现了高效的跨模态和跨时序交互。

### 背景

端到端范式在自动驾驶中表现出巨大潜力，但现有方法大多基于Transformer架构，其二次方注意力成本限制了在资源受限边缘平台上的应用，特别是对长时空序列建模能力的影响。

### 目的

开发一种能够高效融合长期时序上下文并支持跨模态交互的自动驾驶模型，解决Transformer架构的计算效率问题。

### 方法

提出LADY模型，采用完全线性注意力机制，实现恒定计算和内存成本的长期时序上下文融合，并引入轻量级线性交叉注意力机制促进跨模态信息交换。

### 主要发现

在NAVSIM和Bench2Drive基准测试中，LADY实现了最先进的性能，具有恒定的时间和内存复杂度，提供改进的规划性能和显著降低的计算成本，已在边缘设备上成功部署验证。

### 结论

LADY通过线性注意力机制有效解决了Transformer的计算效率瓶颈，实现了自动驾驶所需的跨模态和跨时序交互，在资源受限的边缘平台上表现出色，具有实际应用价值。

### 翻译

端到端范式已展现出自动驾驶的巨大潜力。此外，大多数现有方法都基于Transformer架构构建。然而，Transformer会产生二次方注意力成本，限制了其在资源受限的边缘平台上对长时空序列建模的能力。由于自动驾驶本质上需要高效的时序建模，这一挑战严重限制了它们的部署和实时性能。最近，线性注意力机制因其优越的时空复杂度而受到越来越多的关注。然而，现有的线性注意力架构仅限于自注意力，缺乏对跨模态和跨时序交互的支持，而这两种交互对自动驾驶都至关重要。在这项工作中，我们提出了LADY，这是首个基于完全线性注意力的生成模型，用于端到端自动驾驶。LADY能够在推理时融合长期时序上下文，具有恒定的计算和内存成本，无论相机和LiDAR特征的历史长度如何。此外，我们引入了一种轻量级线性交叉注意力机制，实现了有效的跨模态信息交换。在NAVSIM和Bench2Drive基准测试上的实验表明，LADY实现了最先进的性能，具有恒定的时间和内存复杂度，提供了改进的规划性能和显著降低的计算成本。此外，该模型已在边缘设备上部署和验证，展示了其在资源受限场景中的实用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中基于Transformer的端到端模型计算效率低下的问题，特别是Transformer的二次方注意力复杂度限制了其在资源受限的边缘平台上的实时性能。这个问题在现实中非常重要，因为自动驾驶系统需要实时处理多帧传感器数据并做出安全决策，而现有方法难以在保持高性能的同时满足实时性和边缘设备部署的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有自动驾驶方法的局限性，特别是Transformer的计算瓶颈。他们借鉴了语言模型中的线性注意力机制(如RWKV-7)，将其应用于自动驾驶领域。同时参考了扩散模型(DiffusionDrive)用于生成多模态轨迹，以及iPad的评分机制来选择最佳轨迹。作者设计了一个完全基于线性注意力的架构，包括创新的线性交叉注意力(LICA)机制，以实现高效的多模态信息交换和长时序上下文融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是用线性注意力替代Transformer的二次方注意力机制，实现计算复杂度的线性化，同时保持模型性能。整体流程包括：1)多帧摄像头和LiDAR特征通过RWKV-7进行融合；2)使用线性交叉注意力(LICA)进行跨模态信息交换；3)通过扩散模型生成多模态轨迹；4)使用评分机制选择最佳轨迹。训练时多帧特征并行处理，推理时当前帧特征与时间隐藏状态顺序融合，支持长时序上下文。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个完全线性注意力端到端自动驾驶模型；2)轻量级线性交叉注意力(LICA)机制；3)支持长时序上下文融合且计算开销恒定；4)多模态轨迹生成与评分。相比之前工作，LADY完全避免了Transformer的二次方复杂度，实现了真正的线性计算复杂度；能够融合无限长度的历史信息而不增加计算开销；支持跨模态交互的线性注意力机制；在保持或提高性能的同时显著降低了计算和内存需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LADY首次将完全线性注意力机制引入自动驾驶领域，通过创新的线性交叉注意力和长时序上下文融合方法，在保持高性能的同时显著降低了计算复杂度，实现了资源受限边缘平台上的高效端到端自动驾驶。'}


### 论文摘要

End-to-end paradigms have demonstrated great potential for autonomous driving. Additionally, most existing methods are built upon Transformer architectures. However, transformers incur a quadratic attention cost, limiting their ability to model long spatial and temporal sequences-particularly on resource-constrained edge platforms. As autonomous driving inherently demands efficient temporal modeling, this challenge severely limits their deployment and real-time performance. Recently, linear attention mechanisms have gained increasing attention due to their superior spatiotemporal complexity. However, existing linear attention architectures are limited to self-attention, lacking support for cross-modal and cross-temporal interactions-both crucial for autonomous driving. In this work, we propose LADY, the first fully linear attention-based generative model for end-to-end autonomous driving. LADY enables fusion of long-range temporal context at inference with constant computational and memory costs, regardless of the history length of camera and LiDAR features. Additionally, we introduce a lightweight linear cross-attention mechanism that enables effective cross-modal information exchange. Experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that LADY achieves state-of-the-art performance with constant-time and memory complexity, offering improved planning performance and significantly reduced computational cost. Additionally, the model has been deployed and validated on edge devices, demonstrating its practicality in resource-limited scenarios.

---

## 31. HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering

**论文链接:** [http://arxiv.org/abs/2512.14870v1](http://arxiv.org/abs/2512.14870v1)

**作者:** Dan Ben-Ami, Gabriele Serussi, Kobi Cohen, Chaim Baskin

**发布时间:** 2025-12-16

### GPT解析

### 总结

论文介绍了HERBench，一个专门用于评估视频大语言模型跨时间整合多个视觉证据能力的新型视频问答基准测试。该基准包含26,000个五选一多项选择题，组织成12个组合任务，评估身份绑定、跨实体关系、时间顺序等能力。研究发现当前最先进的Video-LLMs在HERBench上表现不佳，准确率仅31-42%，主要存在检索缺陷和融合缺陷两个瓶颈。

### 背景

视频大语言模型(Video-LLMs)正在快速发展，但当前的视频问答(VideoQA)基准测试通常允许问题从单一显著线索中得到回答，无法充分测试需要聚合多个时间分离的视觉证据的推理能力。

### 目的

开发一个专门的视频问答基准测试(HERBench)，用于评估模型跨时间整合多个证据的能力，确保问题必须聚合至少三个不重叠的证据线索，仅依靠语言先验或单一快照无法回答。

### 方法

创建HERBench基准，包含26,000个五选一多项选择题，组织成12个组合任务；引入'最小必需帧集'(MRFS)概念来量化证据需求；评估13个最先进的Video-LLMs在HERBench上的表现；分析失败原因，分为检索缺陷和融合缺陷。

### 主要发现

HERBench对证据的要求显著高于先前数据集(平均MRFS为5.5，而先前数据集为2.6-4.2)；13个最先进的Video-LLMs在HERBench上的准确率仅为31-42%，仅略高于20%的随机猜测基线；模型失败主要归因于检索缺陷和融合缺陷两个关键瓶颈。

### 结论

通过使跨时间证据成为不可避免且可量化的内容，HERBench为推进稳健的组合视频理解建立了原则性目标。

### 翻译

视频大语言模型(Video-LLMs)正在迅速发展，但当前的视频问答(VideoQA)基准测试通常允许问题从单一显著线索中得到回答，这无法充分测试需要聚合多个时间分离的视觉证据的推理能力。我们提出了HERBench，这是一个专门构建的视频问答基准，用于评估跨时间多证据整合。每个问题需要聚合至少三个来自不同视频段的不重叠证据线索，因此语言先验或单一快照都不足以回答。HERBench包含26,000个五选一多项选择题，组织成12个组合任务，探索身份绑定、跨实体关系、时间顺序、共现验证和计数能力。为了使证据需求可量化，我们引入了'最小必需帧集'(MRFS)，即模型必须融合的最小帧数才能正确回答，并表明HERBench的要求显著高于先前数据集(平均MRFS为5.5，而先前为2.6-4.2)。在HERBench上评估13个最先进的Video-LLMs显示出普遍失败：准确率仅为31-42%，仅略高于20%的随机猜测基线。我们将这种失败分解为两个关键瓶颈：(1)检索缺陷，帧选择器忽略关键证据；(2)融合缺陷，即使提供了所有必要证据，模型也无法整合信息。通过使跨时间证据成为不可避免且可量化的内容，HERBench为推进稳健的组合视频理解建立了原则性目标。


### 论文摘要

Video Large Language Models (Video-LLMs) are rapidly improving, yet current Video Question Answering (VideoQA) benchmarks often allow questions to be answered from a single salient cue, under-testing reasoning that must aggregate multiple, temporally separated visual evidence. We present HERBench, a VideoQA benchmark purpose-built to assess multi-evidence integration across time. Each question requires aggregating at least three non-overlapping evidential cues across distinct video segments, so neither language priors nor a single snapshot can suffice. HERBench comprises 26K five-way multiple-choice questions organized into twelve compositional tasks that probe identity binding, cross-entity relations, temporal ordering, co-occurrence verification, and counting. To make evidential demand measurable, we introduce the Minimum Required Frame-Set (MRFS), the smallest number of frames a model must fuse to answer correctly, and show that HERBench imposes substantially higher demand than prior datasets (mean MRFS 5.5 vs. 2.6-4.2). Evaluating 13 state-of-the-art Video-LLMs on HERBench reveals pervasive failures: accuracies of 31-42% are only slightly above the 20% random-guess baseline. We disentangle this failure into two critical bottlenecks: (1) a retrieval deficit, where frame selectors overlook key evidence, and (2) a fusion deficit, where models fail to integrate information even when all necessary evidence is provided. By making cross-time evidence both unavoidable and quantifiable, HERBench establishes a principled target for advancing robust, compositional video understanding.

---

## 32. IMKD: Intensity-Aware Multi-Level Knowledge Distillation for Camera-Radar Fusion

**论文链接:** [http://arxiv.org/abs/2512.15581v1](http://arxiv.org/abs/2512.15581v1)

**作者:** Shashank Mishra, Karan Patil, Didier Stricker, Jason Rambach

**发布时间:** 2025-12-17

**备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026. 22 pages, 8 figures. Includes supplementary material

### GPT解析

### 总结

本文提出了一种名为IMKD的新型雷达-相机融合框架，基于多级知识蒸馏实现在不使用LiDAR情况下的高性能3D目标检测，该方法保留各传感器固有特性并增强互补优势。

### 背景

现有知识蒸馏方法通常将特定模态特征直接传输到各传感器，这会扭曲传感器独特特性并降低各自优势，而高性能雷达-相机3D目标检测可通过知识蒸馏实现且推理时无需LiDAR。

### 目的

解决现有方法中特征传输导致传感器特性扭曲的问题，提出IMKD框架以保留各传感器固有特性同时增强互补优势。

### 方法

IMKD采用三阶段强度感知蒸馏策略：(1)LiDAR到雷达的强度感知特征蒸馏，用细粒度结构线索增强雷达表示；(2)LiDAR到融合的强度引导特征蒸馏，选择性地突出几何和深度信息，促进模态互补性；(3)相机-雷达强度引导融合机制，促进有效特征对齐和校准。

### 主要发现

在nuScenes基准上，IMKD达到67.0%的NDS和61.0%的mAP，性能优于所有先前基于蒸馏的雷达-相机融合方法。

### 结论

IMKD框架成功解决了现有知识蒸馏方法的问题，通过多级知识蒸馏保留传感器特性并增强互补优势，实现了高性能3D目标检测。

### 翻译

高性能的雷达-相机3D目标检测可以通过利用知识蒸馏在推理时不使用LiDAR来实现。然而，现有的蒸馏方法通常将特定模态的特征直接传输到每个传感器，这可能会扭曲它们的独特特性并降低它们各自的优势。为了解决这个问题，我们引入了IMKD，这是一种基于多级知识蒸馏的雷达-相机融合框架，在保留每个传感器固有特性的同时增强它们的互补优势。IMKD应用三阶段、强度感知的蒸馏策略来丰富整个架构中的融合表示：(1)LiDAR到雷达的强度感知特征蒸馏，用细粒度结构线索增强雷达表示；(2)LiDAR到融合的强度引导特征蒸馏，在融合级别选择性地突出有用的几何和深度信息，促进模态间的互补性而非强制对齐；(3)相机-雷达强度引导融合机制，促进有效的特征对齐和校准。在nuScenes基准上的大量实验表明，IMKD达到了67.0%的NDS和61.0%的mAP，优于所有先前基于蒸馏的雷达-相机融合方法。我们的代码和模型可在https://github.com/dfki-av/IMKD/获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有知识蒸馏方法在雷达-摄像头融合3D目标检测中的局限性：这些方法通常直接将模态特定特征传输到每个传感器，扭曲了它们的独特特性并降低了各自优势。这个问题在现实中很重要，因为它关系到如何构建低成本但高效的自动驾驶感知系统——LiDAR虽然精确但昂贵且范围有限，而摄像头和雷达的组合可以提供更经济的替代方案，但需要有效融合两种传感器的优势。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法独立将知识蒸馏到每个模态的局限性，忽略了传感器独特特性。他们设计了一个三阶段强度感知蒸馏策略，包括LiDAR到雷达的特征蒸馏、LiDAR到融合特征的蒸馏以及摄像头-雷达融合机制。作者借鉴了LabelDistill的标签引导知识蒸馏、BEVDepth的深度估计和BEVFormer的时序融合等现有工作，但创新性地引入了强度感知机制，利用LiDAR强度作为可靠性先验，突出几何一致区域，同时保留雷达的鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用强度感知的多级知识蒸馏增强摄像头-雷达融合，保留每个传感器的内在特性同时放大互补优势，使用LiDAR强度作为可靠性先验指导知识传递。整体流程包括：1)摄像头特征提取与视图转换；2)雷达特征提取与网格化；3)强度感知特征融合；4)自适应强度引导雷达特征增强；5)LiDAR引导特征增强；6)基于标签的知识蒸馏；7)多损失联合优化。训练时使用LiDAR和标签作为特权信息，推理时仅使用摄像头和雷达，确保高效部署。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)强度感知的多级知识蒸馏框架；2)三阶段强度感知蒸馏策略；3)在联合融合特征空间而非单个模态上执行知识蒸馏；4)强度感知的雷达-摄像头融合模块；5)引入结构化监督减少对LiDAR的依赖。相比之前工作，IMKD避免了直接强制雷达或摄像头模仿LiDAR表示，保留了传感器独特特性；引入了强度感知机制和自适应权重指导监督，而非使用均匀或二元掩码；在融合级别而非模态级别执行蒸馏，更好地利用了跨模态交互信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IMKD通过引入强度感知的多级知识蒸馏框架，有效解决了雷达-摄像头融合3D目标检测中传感器特性被扭曲的问题，在保持推理效率的同时显著提升了检测性能，为低成本自动驾驶感知系统提供了新思路。'}


### 论文摘要

High-performance Radar-Camera 3D object detection can be achieved by leveraging knowledge distillation without using LiDAR at inference time. However, existing distillation methods typically transfer modality-specific features directly to each sensor, which can distort their unique characteristics and degrade their individual strengths. To address this, we introduce IMKD, a radar-camera fusion framework based on multi-level knowledge distillation that preserves each sensor's intrinsic characteristics while amplifying their complementary strengths. IMKD applies a three-stage, intensity-aware distillation strategy to enrich the fused representation across the architecture: (1) LiDAR-to-Radar intensity-aware feature distillation to enhance radar representations with fine-grained structural cues, (2) LiDAR-to-Fused feature intensity-guided distillation to selectively highlight useful geometry and depth information at the fusion level, fostering complementarity between the modalities rather than forcing them to align, and (3) Camera-Radar intensity-guided fusion mechanism that facilitates effective feature alignment and calibration. Extensive experiments on the nuScenes benchmark show that IMKD reaches 67.0% NDS and 61.0% mAP, outperforming all prior distillation-based radar-camera fusion methods. Our code and models are available at https://github.com/dfki-av/IMKD/.

---

## 33. DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding

**论文链接:** [http://arxiv.org/abs/2512.15000v1](http://arxiv.org/abs/2512.15000v1)

**作者:** Ruiyi Zhang, Peijia Qin, Qi Cao, Pengtao Xie

**发布时间:** 2025-12-17

### GPT解析

### 总结

DreamPRM-Code是一种专注于编程的进程奖励模型(PRM)，通过函数链提示策略和元学习校正机制解决了传统PRM在编程应用中的局限性，在LiveCodeBench上取得了80.9%的pass@1率，超越了OpenAI o4-mini。

### 背景

进程奖励模型(PRM)已成为通过测试时扩展提高大型语言模型的重要工具，但在编程领域的应用效果有限，主要原因是代码中缺乏有意义的步骤分解以及蒙特卡洛生成的部分标签存在噪声。

### 目的

提出一个专注于编程的PRM(DreamPRM-Code)，以解决传统PRM在编程应用中的局限性，提高其在代码生成任务中的效果。

### 方法

DreamPRM-Code将函数视为推理步骤，采用函数链提示策略(Chain-of-Function prompting)来诱导模块化代码生成；同时引入基于元学习的校正机制，利用干净的最终解决方案单元测试标签，并通过双层优化来改进中间标签。

### 主要发现

在测试时扩展应用中，DreamPRM-Code在LiveCodeBench上取得了最先进的性能，pass@1率达到80.9%，超过了OpenAI o4-mini。

### 结论

DreamPRM-Code通过创新的方法解决了PRM在编程领域应用中的关键问题，实现了显著的性能提升，为编程任务中的进程奖励模型应用提供了新思路。

### 翻译

进程奖励模型(PRM)已成为通过测试时扩展提高大型语言模型的重要工具，但在编程领域的应用效果有限，原因是代码中缺乏有意义的步骤分解以及蒙特卡洛生成的部分标签存在噪声。我们提出了DreamPRM-Code，这是一种专注于编程的PRM，它使用函数链提示策略将函数视为推理步骤，以诱导模块化代码生成，使PRM的训练和应用类似于数学推理任务。为解决标签噪声问题，DreamPRM-Code引入了一种基于元学习的校正机制，利用干净的最终解决方案单元测试标签，并进行双层优化来改进中间标签。在测试时扩展应用中，DreamPRM-Code在LiveCodeBench上取得了80.9%的pass@1率的最先进性能，超越了OpenAI o4-mini。


### 论文摘要

Process Reward Models (PRMs) have become essential for improving Large Language Models (LLMs) via test-time scaling, yet their effectiveness in coding remains limited due to the lack of meaningful step decompositions in code and the noise of Monte-Carlo-generated partial labels. We propose DreamPRM-Code, a coding-focused PRM that treats functions as reasoning steps using a Chain-of-Function prompting strategy to induce modular code generation, enabling PRM training and application analogous to mathematical reasoning tasks. To address label noise, DreamPRM-Code introduces a meta-learning-based correction mechanism that leverages clean final-solution unit-test labels and performs bi-level optimization to refine intermediate labels. Applying on test-time scaling, DreamPRM-Code achieved state-of-the-art performance on LiveCodeBench with 80.9 pass@1 rate, surpassing OpenAI o4-mini.

---

## 34. EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery

**论文链接:** [http://arxiv.org/abs/2512.13857v2](http://arxiv.org/abs/2512.13857v2)

**作者:** Kamer Ali Yuksel

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文介绍了EvoLattice框架，用于程序和多智能体系统的演化，通过在有向无环图中维护多个候选方案，解决了传统方法中丢弃有用变体和破坏性编辑的问题。

### 背景

大型语言模型越来越多地用于程序和多智能体系统的演化，但大多数现有方法基于覆盖式突变，一次只维护一个候选方案，导致有用变体被丢弃、破坏性编辑和脆弱的搜索空间。

### 目的

引入EvoLattice框架，提供一种能够在单一结构中维护整个候选种群的方法，同时保证结构正确性和提供更密集的反馈信号。

### 方法

EvoLattice在一个有向无环图中表示整个候选程序或智能体行为的种群，每个节点存储多个持久性替代方案，每条有效路径定义一个不同的可执行候选，产生大的组合搜索空间而不重复结构。

### 主要发现

EvoLattice实现了细粒度的替代级别评估，统计数据显示局部设计选择如何影响全局性能，为LLM引导的演化提供密集、数据驱动的反馈，同时通过自修复机制保证结构正确性，并能自然扩展到智能体演化。

### 结论

在程序合成方面，EvoLattice比之前的LLM引导方法产生更稳定的演化、更大的表达能力和更强的改进轨迹，其动力学类似于质量多样性优化，是从内部多替代表示中隐式出现的。

### 翻译

大型语言模型(LLMs)越来越多地用于程序和多智能体系统的演化，但大多数现有方法依赖于基于覆盖的突变，一次只维护一个候选方案。这类方法丢弃有用的变体，容易受到破坏性编辑的影响，并探索一个容易发生结构性失败的脆弱搜索空间。我们引入了EvoLattice，一个框架，它在一个有向无环图中表示整个候选程序或智能体行为的种群。每个节点存储多个持久性替代方案，通过图中的每条有效路径定义一个不同的可执行候选，产生大的组合搜索空间而不重复结构。EvoLattice通过对每条路径中出现的每个替代方案进行评分，实现细粒度的替代级别评估，产生统计数据显示局部设计选择如何影响全局性能。这些统计为LLM引导的突变、重组和修剪提供了密集、数据驱动的反馈信号，同时保留成功的组件。结构正确性通过确定性自修复机制保证，该机制独立于LLM强制执行无环性和依赖一致性。EvoLattice通过将替代方案解释为提示片段或子智能体行为，自然地扩展到智能体演化。在程序合成(代理和优化器元学习)方面，EvoLattice比之前的LLM引导方法产生更稳定的演化、更大的表达能力和更强的改进轨迹。由此产生的动力学类似于质量多样性优化，是从EvoLattice的内部多替代表示中隐式出现的，而不是来自显式的外部档案。


### 论文摘要

Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.

---

## 35. Multi-View Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.15708v1](http://arxiv.org/abs/2512.15708v1)

**作者:** Leo Segre, Or Hirschorn, Shai Avidan

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了一种将基础模型转换为多视图基础模型的方法，通过添加3D感知注意力层增强Transformer模型，实现多视图图像中对应点特征的一致性匹配。

### 背景

基础模型在计算机视觉应用中至关重要，它们输入单张RGB图像并输出有用的深度特征表示。然而，当处理同一3D场景的多个视图时，这些模型独立处理每张图像，无法保证同一3D点特征的一致性。

### 目的

提出一种方法将基础模型转换为多视图基础模型，输入一组图像，为每张图像输出特征图，使对应点的特征尽可能保持一致。

### 方法

通过增强基于Transformer的基础模型（如DINO、SAM、CLIP）添加中间3D感知注意力层，帮助匹配不同视图的特征。这种方法绕过了构建一致的3D特征模型的必要性，允许在图像空间直接操作。

### 主要发现

定量实验表明，与当前基础模型相比，该方法显著改善了特征匹配性能，在表面法线估计和多视图分割等任务中表现优异。

### 结论

所提出的多视图基础模型能够有效处理多视图图像中的特征一致性问题，为计算机视觉应用提供了更强大的工具。

### 翻译

基础模型是各种计算机视觉应用中的重要工具。它们输入单张RGB图像并输出适用于各种应用的深度特征表示。然而，当我们有同一3D场景的多个视图时，它们独立处理每张图像，并不总是为同一3D点生成一致的特征。我们提出了一种将基础模型转换为多视图基础模型的方法。这种模型输入一组图像，为每张图像输出特征图，使得对应点的特征尽可能一致。这种方法绕过了构建一致的3D特征模型的必要性，允许在图像空间直接操作。具体来说，我们展示了如何通过添加中间3D感知注意力层来增强基于Transformer的基础模型（即DINO、SAM、CLIP），帮助匹配不同视图的特征。作为主要示例，我们展示了表面法线估计和多视图分割任务。定量实验表明，与当前基础模型相比，我们的方法显著改善了特征匹配。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决2D基础模型（如DINO、SAM、CLIP）在处理同一3D场景的多个视图时无法为相同3D点产生一致特征的问题。这个问题很重要，因为许多3D视觉任务（如3D重建、增强现实、多视图几何）需要一致的特征表示，而现有方法要么需要昂贵的场景级优化，要么缺乏全局3D一致性，限制了基础模型在3D应用中的效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到2D基础模型缺乏3D意识，并认识到将2D扩展到3D的挑战。他们借鉴了'lifting'原则（从2D扩展到3D）、FiT3D工作（但发现其缺乏多视图一致性）、Kim等人的时间注意力层工作（扩展到3D几何）以及LoRA微调技术。作者设计了一种在预训练模型中插入多视图适配器的方案，通过参数高效训练保留2D模型的语义能力，同时添加3D意识。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是为预训练2D模型添加多视图适配器，使其能够跨不同视图进行特征匹配，同时明确融入相机姿态信息确保几何一致性。整体流程：1)在Transformer骨干网络每个块间插入空间适配器；2)用Plücker嵌入表示相机姿态构建射线图；3)将姿态编码与特征连接；4)使用几何感知密集损失训练；5)添加正则化防止特征偏离原始空间；6)推理时输入多视图图像输出一致特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新：1)提出可扩展多视图适配框架，实现单次前向传播产生跨视图几何一致特征；2)设计相机条件的多视图适配器和高效训练策略；3)纯推理时操作，避免昂贵的3D特征表示和场景级优化。相比之前工作：与需要场景级优化的方法相比计算效率更高；与FiT3D相比使用多视图创建统一3D表示；与Lift3D相比不需要渲染新视图；适用于多种基础模型；在保持语义一致性的同时显著提高几何一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种通过多视图适配器将2D基础模型转换为具有3D意识的多视图一致模型的方法，在保持原始模型语义能力的同时显著提高了跨视图的几何一致性，且无需场景级优化。'}


### 论文摘要

Foundation models are vital tools in various Computer Vision applications. They take as input a single RGB image and output a deep feature representation that is useful for various applications. However, in case we have multiple views of the same 3D scene, they operate on each image independently and do not always produce consistent features for the same 3D point. We propose a way to convert a Foundation Model into a Multi-View Foundation Model. Such a model takes as input a set of images and outputs a feature map for each image such that the features of corresponding points are as consistent as possible. This approach bypasses the need to build a consistent 3D model of the features and allows direct manipulation in the image space. Specifically, we show how to augment Transformers-based foundation models (i.e., DINO, SAM, CLIP) with intermediate 3D-aware attention layers that help match features across different views. As leading examples, we show surface normal estimation and multi-view segmentation tasks. Quantitative experiments show that our method improves feature matching considerably compared to current foundation models.

---

## 36. MoonSeg3R: Monocular Online Zero-Shot Segment Anything in 3D with Reconstructive Foundation Priors

**论文链接:** [http://arxiv.org/abs/2512.15577v1](http://arxiv.org/abs/2512.15577v1)

**作者:** Zhipeng Du, Duolikun Danier, Jan Eric Lenssen, Hakan Bilen

**发布时间:** 2025-12-17

### GPT解析

### 总结

这项研究提出了MoonSeg3R方法，实现在线零样本单目3D实例分割，利用CUT3R模型和三个关键组件，达到与基于RGB-D系统相媲美的性能。

### 背景

现有在线零样本单目3D实例分割方法表现不佳，因为它们依赖于已配准的RGB-D序列，而实际应用中往往只有单目RGB输入。

### 目的

克服现有方法的局限性，开发一种能够仅从单目RGB流中进行在线3D实例分割的方法。

### 方法

利用CUT3R重构基础模型提供几何先验，并提出MoonSeg3R框架，包含三个关键组件：(1)带有空间-语义蒸馏的自监督查询细化模块；(2)3D查询索引内存；(3)状态分布令牌作为掩码身份描述符。

### 主要发现

在ScanNet200和SceneNN数据集上，MoonSeg3R是首个实现在线单目3D分割的方法，性能与最先进的基于RGB-D系统相当。

### 结论

MoonSeg3R成功解决了在线零样本单目3D实例分割的挑战，证明了仅使用单目RGB输入也能达到高性能3D分割。

### 翻译

在本文中，我们专注于在线零样本单目3D实例分割，这是一个新颖实用的设置，因为现有方法依赖于已配准的RGB-D序列而无法在此设置下表现。为了克服这一限制，我们利用了CUT3R，一种最近的重构基础模型，从单个RGB流中提供可靠的几何先验。我们提出了MoonSeg3R，它引入了三个关键组件：(1)带有空间-语义蒸馏的自监督查询细化模块，将2D视觉基础模型的分割掩码转换为判别性3D查询；(2)3D查询索引内存，通过检索上下文查询提供时间一致性；(3)来自CUT3R的状态分布令牌，作为掩码身份描述符，增强跨帧融合。在ScanNet200和SceneNN上的实验表明，MoonSeg3R是首个实现在线单目3D分割的方法，并实现了与最先进的基于RGB-D系统相竞争的性能。代码和模型将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目在线零样本3D实例分割问题，即仅从单目RGB图像流（无需深度信息）实时进行3D场景重建和实例分割。这个问题在现实中非常重要，因为许多应用场景（如机器人导航、自动驾驶）中深度传感器可能不可用或不实用，现有方法依赖于RGB-D输入或需要3D真实值监督，限制了在没有专用深度传感器的平台上的部署。在复杂环境中，能够仅从单目摄像头进行实时3D感知对具身智能和自主操作至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的设计思路是结合两种基础模型的优势：利用重建基础模型（RFM，特别是CUT3R）提供可靠的几何先验，以及利用视觉基础模型（VFM，如CropFormer）提供强大的2D掩码先验。作者首先识别现有方法依赖RGB-D输入或需要3D真实值监督的局限性，发现CUT3R等RFM可从单目RGB流进行实时3D重建，但优化目标是重建而非分割，缺乏语义意识、几何预测有噪声且记忆难以解释。因此，作者设计了将RFM几何先验与VFM语义能力相结合的方法。该方法借鉴了CUT3R的重建能力、CropFormer的分割能力，以及查询机制和记忆机制设计了查询索引记忆（QIM）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将2D视觉基础模型生成的分割掩码提升到3D，通过自监督查询精炼和跨帧关联实现一致的3D分割，同时利用重建基础模型的几何先验替代显式深度监督。整体流程包括：1)输入处理：RFM预测几何信息，VFM生成分割掩码；2)3D原型表示：将2D掩码提升为3D原型查询；3)查询精炼：通过交叉注意力精炼查询并融入上下文信息；4)空间-语义蒸馏：自监督训练查询参数；5)查询索引记忆：维护全局查询库和索引图实现跨帧关联；6)在线掩码融合：帧内合并和跨帧匹配实现最终分割结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个单目在线零样本3D分割框架，直接从单目RGB流进行在线3D实例分割；2)自监督查询精炼和蒸馏策略，强制实例级判别性和几何一致性；3)3D查询索引记忆(QIM)，通过空间键和上下文查询检索实现跨帧关联；4)基于注意力的在线掩码融合策略，利用状态分布token增强跨帧融合。相比之前工作，本文方法仅需单目RGB输入（无需深度信息），完全无监督（无需3D几何或掩码监督），实现实时在线处理，且在ScanNet200和SceneNN上实现了与最先进RGB-D系统相当的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MoonSeg3R首次实现了仅从单目RGB流进行在线零样本3D实例分割，通过结合重建基础模型的几何先验和视觉基础模型的语义能力，无需深度或掩码监督即可实现实时3D场景重建和分割。'}


### 论文摘要

In this paper, we focus on online zero-shot monocular 3D instance segmentation, a novel practical setting where existing approaches fail to perform because they rely on posed RGB-D sequences. To overcome this limitation, we leverage CUT3R, a recent Reconstructive Foundation Model (RFM), to provide reliable geometric priors from a single RGB stream. We propose MoonSeg3R, which introduces three key components: (1) a self-supervised query refinement module with spatial-semantic distillation that transforms segmentation masks from 2D visual foundation models (VFMs) into discriminative 3D queries; (2) a 3D query index memory that provides temporal consistency by retrieving contextual queries; and (3) a state-distribution token from CUT3R that acts as a mask identity descriptor to strengthen cross-frame fusion. Experiments on ScanNet200 and SceneNN show that MoonSeg3R is the first method to enable online monocular 3D segmentation and achieves performance competitive with state-of-the-art RGB-D-based systems. Code and models will be released.

---

## 37. On the Effectiveness of Textual Prompting with Lightweight Fine-Tuning for SAM3 Remote Sensing Segmentation

**论文链接:** [http://arxiv.org/abs/2512.15564v1](http://arxiv.org/abs/2512.15564v1)

**作者:** Roni Blushtein-Livnon, Osher Rafaeli, David Ioffe, Amir Boger, Karen Sandberg Esquenazi, Tal Svoray

**发布时间:** 2025-12-17

### GPT解析

### 总结

该研究评估了SAM3框架在遥感图像分割中的应用，比较了不同提示策略和监督规模下的性能表现。

### 背景

遥感图像分割受限于标注数据的有限性，以及高空图像与用于训练基础模型的自然图像之间的差距，这促使在有限监督下进行有效适应。

### 目的

评估SAM3框架用于遥感图像分割的效果，比较文本、几何和混合提示策略在不同监督规模下的性能，包括零样本推理和轻量级微调。

### 方法

使用SAM3概念驱动框架，该框架可以从文本提示生成掩码而无需任务特定修改。研究在四种目标类型上评估了不同提示策略，并在不同规模的监督下进行测试。

### 主要发现

结合语义和几何提示在所有目标和指标上表现最佳；纯文本提示表现最差，特别是对不规则形状目标；文本提示配合轻量微调对几何规则和视觉显著目标提供了实用的性能-努力权衡；随着监督规模增加，性能提升但收益递减；精确度和IoU之间持续存在差距，表明欠分割和边界不准确仍是主要问题。

### 结论

SAM3框架在有限监督下对遥感图像分割有效，适度的几何标注工作量足以实现有效适应，但欠分割和边界不准确仍需进一步解决。

### 翻译

遥感图像分割受限于标注数据的有限性以及高空图像与用于训练基础模型的自然图像之间的差距。这促使在有限监督下进行有效适应。SAM3概念驱动框架可以从文本提示生成掩码而无需任务特定修改，这可能实现这种适应。我们在四种目标类型上评估了SAM3用于遥感图像的效果，比较了文本、几何和混合提示策略，在逐渐增加监督的轻量级微调规模下进行测试，同时包括零样本推理。结果表明，结合语义和几何提示在所有目标和指标上产生最高性能。纯文本提示表现最差，对于不规则形状的目标存在明显的分数差距，这反映了SAM3文本表示与其高空外观之间的语义对齐有限。然而，对于几何规则和视觉显著的目标，文本提示配合轻量微调提供了实用的性能-努力权衡。在所有目标中，性能在零样本推理和微调之间有所提升，随后随着监督规模增加而收益递减。也就是说，适度的几何标注工作量足以实现有效适应。精确度和IoU之间持续存在的差距进一步表明，欠分割和边界不准确仍然是遥感任务中的常见错误模式，特别是对于不规则和不常见目标。


### 论文摘要

Remote sensing (RS) image segmentation is constrained by the limited availability of annotated data and a gap between overhead imagery and natural images used to train foundational models. This motivates effective adaptation under limited supervision. SAM3 concept-driven framework generates masks from textual prompts without requiring task-specific modifications, which may enable this adaptation. We evaluate SAM3 for RS imagery across four target types, comparing textual, geometric, and hybrid prompting strategies, under lightweight fine-tuning scales with increasing supervision, alongside zero-shot inference. Results show that combining semantic and geometric cues yields the highest performance across targets and metrics. Text-only prompting exhibits the lowest performance, with marked score gaps for irregularly shaped targets, reflecting limited semantic alignment between SAM3 textual representations and their overhead appearances. Nevertheless, textual prompting with light fine-tuning offers a practical performance-effort trade-off for geometrically regular and visually salient targets. Across targets, performance improves between zero-shot inference and fine-tuning, followed by diminishing returns as the supervision scale increases. Namely, a modest geometric annotation effort is sufficient for effective adaptation. A persistent gap between Precision and IoU further indicates that under-segmentation and boundary inaccuracies remain prevalent error patterns in RS tasks, particularly for irregular and less prevalent targets.

---

## 38. Reducing Pilots in Channel Estimation With Predictive Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.15562v1](http://arxiv.org/abs/2512.15562v1)

**作者:** Xingyu Zhou, Le Liang, Hao Ye, Jing Zhang, Chao-Kai Wen, Shi Jin

**发布时间:** 2025-12-17

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本文提出了一种基于预测基础模型的信道估计框架，用于实现准确、低开销和可泛化的信道状态信息(CSI)获取。该框架结合了跨领域训练的基础模型和基于视觉Transformer的导频处理网络，通过高效融合机制实现可靠CSI重建。

### 背景

精确的信道状态信息(CSI)获取对现代无线系统至关重要，但在大天线阵列、严格的导频开销限制和多样化的部署环境下变得日益困难。现有人工智能解决方案通常缺乏鲁棒性，无法在不同场景中有效泛化。

### 目的

解决现有AI解决方案缺乏鲁棒性和跨场景泛化能力的问题，实现准确、低开销且可泛化的CSI获取方法。

### 方法

提出预测基础模型框架，使用在跨领域大规模CSI数据上训练的基础模型提取通用信道表示并提供预测先验；设计基于视觉Transformer架构的导频处理网络捕获空间、时间和频率相关性；实现高效融合机制整合预测先验与实时测量。

### 主要发现

在不同配置下的广泛评估表明，所提出的估计器在准确性、鲁棒性和泛化能力方面显著优于传统和数据驱动的基线方法。

### 结论

预测基础模型框架能够实现准确、低开销和可泛化的CSI获取，即使在稀疏或有噪声条件下也能可靠重建CSI。

### 翻译

精确的信道状态信息(CSI)获取对现代无线系统至关重要，但在大天线阵列、严格的导频开销限制和多样化的部署环境下，这变得越来越困难。现有人工智能解决方案通常缺乏鲁棒性，无法在不同场景中泛化。为解决这一局限，本文引入了一种基于预测基础模型的信道估计框架，可实现准确、低开销和可泛化的CSI获取。所提出的框架使用在跨领域大规模CSI数据上训练的预测基础模型来提取通用信道表示，并提供具有强跨场景可转移性的预测先验。进一步设计了基于视觉Transformer架构的导频处理网络，从导频观测中捕获空间、时间和频率相关性。高效的融合机制将预测先验与实时测量相结合，即使在稀疏或有噪声条件下也能实现可靠的CSI重建。在多种配置下的广泛评估表明，所提出的估计器在准确性、鲁棒性和泛化能力方面显著优于传统和数据驱动的基线方法。


### 论文摘要

Accurate channel state information (CSI) acquisition is essential for modern wireless systems, which becomes increasingly difficult under large antenna arrays, strict pilot overhead constraints, and diverse deployment environments. Existing artificial intelligence-based solutions often lack robustness and fail to generalize across scenarios. To address this limitation, this paper introduces a predictive-foundation-model-based channel estimation framework that enables accurate, low-overhead, and generalizable CSI acquisition. The proposed framework employs a predictive foundation model trained on large-scale cross-domain CSI data to extract universal channel representations and provide predictive priors with strong cross-scenario transferability. A pilot processing network based on a vision transformer architecture is further designed to capture spatial, temporal, and frequency correlations from pilot observations. An efficient fusion mechanism integrates predictive priors with real-time measurements, enabling reliable CSI reconstruction even under sparse or noisy conditions. Extensive evaluations across diverse configurations demonstrate that the proposed estimator significantly outperforms both classical and data-driven baselines in accuracy, robustness, and generalization capability.

---

## 39. Photorealistic Phantom Roads in Real Scenes: Disentangling 3D Hallucinations from Physical Geometry

**论文链接:** [http://arxiv.org/abs/2512.15423v1](http://arxiv.org/abs/2512.15423v1)

**作者:** Hoang Nguyen, Xiaohao Xu, Xiaonan Huang

**发布时间:** 2025-12-17

### GPT解析

### 总结

本研究针对单目深度基础模型在几何平面但感知模糊输入中产生幻觉性3D结构的现象（称为'3D Mirage'）提出了首个端到端框架，包括基准测试、评估方法和改进策略。

### 背景

单目深度基础模型通过学习大规模语义先验知识实现了显著的泛化能力，但这导致了一个关键弱点：它们会在几何平面但感知模糊的输入中产生幻觉性的3D结构。

### 目的

探测、量化和控制这种未量化的安全风险（3D Mirage现象），并提供诊断和减轻这一现象的基本工具。

### 方法

1. 探测：提出3D-Mirage基准测试，包含现实世界幻觉案例（如街头艺术）及其精确平面区域标注；2. 量化：基于拉普拉斯评估框架，引入偏差综合分数(DCS)和混淆综合分数(CCS)；3. 控制：提出Grounded Self-Distillation策略，在幻觉区域强制平面性同时保留背景知识。

### 主要发现

单目深度基础模型存在'3D Mirage'现象，即在几何平面但感知模糊的输入中会产生幻觉性的3D结构。

### 结论

研究提供了诊断和减轻3D Mirage现象的工具，呼吁深度估计评估从像素级准确性转向结构和上下文鲁棒性，并将公开代码和基准测试促进该研究方向。

### 翻译

单目深度基础模型通过学习大规模语义先验知识实现了显著的泛化能力，但这造成了一个关键弱点：它们会从几何平面但感知模糊的输入中产生幻觉性的3D结构。我们将这种失败称为'3D Mirage'。本文介绍了第一个端到端框架，用于探测、量化和控制这一未量化的安全风险。为了探测，我们提出了3D-Mirage，这是第一个具有精确平面区域标注和上下文限制裁剪的现实世界幻觉基准测试（例如街头艺术）。为了量化，我们提出了一个基于拉普拉斯的评估框架，包含两个指标：用于衡量虚假非平面性的偏差综合分数(DCS)和用于衡量上下文不稳定性的混淆综合分数(CCS)。为了控制这种失败，我们引入了Grounded Self-Distillation，这是一种参数高效策略，可以在幻觉ROI上强制执行平面性，同时使用冻结的教师模型保留背景知识，从而避免灾难性遗忘。我们的工作提供了诊断和减轻这一现象的基本工具，呼吁MDE评估从像素级准确性向结构和上下文鲁棒性的必要转变。我们的代码和基准测试将公开可用，以促进这一令人兴奋的研究方向。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决单目深度估计模型中的'3D Mirage'问题，即模型会将几何上平坦但感知上模糊的输入（如3D街头艺术）产生幻觉式的3D结构。这个问题在现实中非常重要，特别是在自动驾驶等安全关键应用中，当视野受限或部分遮挡时，模型可能会产生不存在的3D障碍物，导致严重的感知错误和安全隐患。此外，现有的评估指标（如MAE、RMSE）无法检测这种结构失败，使得这一漏洞在模型部署前难以被发现。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别并定义了'3D Mirage'这一失败模式，发现现有SOTA模型在感知模糊的2D模式和受限视野场景中普遍失败。他们意识到标准评估指标无法检测这种结构失败，因此提出需要系统性地探测、量化和控制3D幻觉。在方法设计上，作者借鉴了参数高效微调（PEFT）方法如低秩自适应（LoRA），并创新性地应用了自蒸馏概念。他们还参考了现有深度估计模型架构（如Depth-Anything V2）和数据集（如Penn-Fudan和CamVid）进行正则化，以防止过度平坦化和边缘漂移。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过参数高效的微调方法，在保持模型原有知识的同时，专门针对幻觉区域进行校正。具体实现流程包括：1) 创建3D-Mirage基准数据集，包含真实世界3D错觉图像和平面ROI标注；2) 在预训练模型编码器中注入LoRA适配器；3) 设计三流训练过程（教师流、学生完整图像流、学生裁剪流）；4) 使用复合损失函数（LHKR损失强制幻觉区域平面性，LNKP损失保持背景区域原始知识）；5) 仅优化LoRA参数，训练1个周期以避免灾难性遗忘。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 首次系统性识别并分析'3D Mirage'现象；2) 创建首个专注于真实世界光学错觉场景的基准数据集；3) 提出基于拉普拉斯的评估框架和DCS/CCS指标；4) 提出Grounded Self-Distillation参数高效缓解策略；5) 设计复合损失函数同时处理幻觉校正和知识保存。相比之前工作，本文不关注传统像素级精度，而是关注结构和上下文鲁棒性；方法参数高效，避免灾难性遗忘；专门针对幻觉区域进行手术式校正，而非全局调整；上下文受限裁剪专门针对视野受限场景进行设计。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文首次系统性地识别、量化和缓解了单目深度估计模型中的'3D Mirage'幻觉现象，通过创新的基准数据集、评估指标和参数高效的微调方法，显著提高了模型在安全关键应用中的可靠性和鲁棒性。"}


### 论文摘要

Monocular depth foundation models achieve remarkable generalization by learning large-scale semantic priors, but this creates a critical vulnerability: they hallucinate illusory 3D structures from geometrically planar but perceptually ambiguous inputs. We term this failure the 3D Mirage. This paper introduces the first end-to-end framework to probe, quantify, and tame this unquantified safety risk. To probe, we present 3D-Mirage, the first benchmark of real-world illusions (e.g., street art) with precise planar-region annotations and context-restricted crops. To quantify, we propose a Laplacian-based evaluation framework with two metrics: the Deviation Composite Score (DCS) for spurious non-planarity and the Confusion Composite Score (CCS) for contextual instability. To tame this failure, we introduce Grounded Self-Distillation, a parameter-efficient strategy that surgically enforces planarity on illusion ROIs while using a frozen teacher to preserve background knowledge, thus avoiding catastrophic forgetting. Our work provides the essential tools to diagnose and mitigate this phenomenon, urging a necessary shift in MDE evaluation from pixel-wise accuracy to structural and contextual robustness. Our code and benchmark will be publicly available to foster this exciting research direction.

---

## 40. Leveraging Foundational Models and Simple Fusion for Multi-modal Physiological Signal Analysis

**论文链接:** [http://arxiv.org/abs/2512.15250v1](http://arxiv.org/abs/2512.15250v1)

**作者:** Youssef Ghallab, Omar Iraqy, Mohamed Kandil, Mohamed Ashraf, Saadeldine Eletter, Morougue Ghazal, Ayman Khalafallah, Nagwa El-Makky

**发布时间:** 2025-12-17

**备注:** Published at NeurIPS 2025 Workshop on Foundation Models for the Brain and Body

### GPT解析

### 总结

该研究提出了一种方法来解决多模态生理信号（ECG和EEG）集成的挑战，特别是在标记数据有限的情况下。通过调整CBraMod编码器并引入双掩码策略，研究者成功捕获了导联内和导联间的依赖关系，并通过简单的嵌入连接融合不同模态的表示，实现了有效的下游学习。

### 背景

生理信号如心电图(ECG)和脑电图(EEG)能提供关于人类健康和认知的互补见解，但由于多模态标记数据有限和模态特定差异，多模态集成具有挑战性。

### 目的

克服多模态生理信号集成的挑战，特别是在标记数据有限的情况下实现有效的下游学习。

### 方法

为大规模自监督ECG预训练调整CBraMod编码器，引入双掩码策略捕获导联内和导联间的依赖关系；利用预训练的CBraMod编码器处理EEG，预训练对称的ECG编码器；通过简单的嵌入连接融合这些表示，允许分类头学习跨模态交互。

### 主要发现

在情感识别任务上，该方法取得了接近最先进的性能，证明精心设计的生理编码器即使使用简单的融合也能显著提高下游性能。

### 结论

基础模型方法在利用生理信号的特性方面具有潜力，能够为医疗保健和情感计算提供可扩展、标记效率高且通用的解决方案。

### 翻译

生理信号如心电图(ECG)和脑电图(EEG)能提供关于人类健康和认知的互补见解，但由于多模态标记数据有限和模态特定差异，多模态集成具有挑战性。在这项工作中，我们为大规模自监督ECG预训练调整了CBraMod编码器，引入了双掩码策略来捕获导联内和导联间的依赖关系。为了克服上述挑战，我们利用预训练的CBraMod编码器处理EEG，并预训练了一个对称的ECG编码器，为每个模态配备了丰富的基础表示。然后通过简单的嵌入连接融合这些表示，允许分类头学习跨模态交互，从而在有限的多模态监督下实现有效的下游学习。在情感识别任务上的评估表明，我们的方法取得了接近最先进的性能，证明精心设计的生理编码器即使使用简单的融合也能显著提高下游性能。这些结果突显了基础模型方法在利用生理信号整体特性方面的潜力，为医疗保健和情感计算提供了可扩展、标记效率高且通用的解决方案。


### 论文摘要

Physiological signals such as electrocardiograms (ECG) and electroencephalograms (EEG) provide complementary insights into human health and cognition, yet multi-modal integration is challenging due to limited multi-modal labeled data, and modality-specific differences . In this work, we adapt the CBraMod encoder for large-scale self-supervised ECG pretraining, introducing a dual-masking strategy to capture intra- and inter-lead dependencies. To overcome the above challenges, we utilize a pre-trained CBraMod encoder for EEG and pre-train a symmetric ECG encoder, equipping each modality with a rich foundational representation. These representations are then fused via simple embedding concatenation, allowing the classification head to learn cross-modal interactions, together enabling effective downstream learning despite limited multi-modal supervision. Evaluated on emotion recognition, our approach achieves near state-of-the-art performance, demonstrating that carefully designed physiological encoders, even with straightforward fusion, substantially improve downstream performance. These results highlight the potential of foundation-model approaches to harness the holistic nature of physiological signals, enabling scalable, label-efficient, and generalizable solutions for healthcare and affective computing.

---

## 41. Efficient Nudged Elastic Band Method using Neural Network Bayesian Algorithm Execution

**论文链接:** [http://arxiv.org/abs/2512.14993v1](http://arxiv.org/abs/2512.14993v1)

**作者:** Pranav Kakhandiki, Sathya Chitturi, Daniel Ratner, Sean Gasiorowski

**发布时间:** 2025-12-17

**备注:** 21 pages, 12 figures

### GPT解析

### 总结

NN-BAX是一种新框架，通过联合学习能量景观和最小能量路径，显著减少了发现亚稳态间最小能量路径所需的计算量，在保持高精度的同时将计算时间从数周缩短至数小时或数天。

### 背景

发现亚稳态之间的最小能量路径对于催化剂和生物分子设计等科学任务至关重要，但标准的NEB算法需要数百到数万次计算密集型模拟，使得复杂系统应用成本过高。

### 目的

开发一种能够减少计算需求的方法，用于高效发现亚稳态之间的最小能量路径。

### 方法

引入神经网络贝叶斯算法执行(NN-BAX)，这是一种联合学习能量景观和最小能量路径的框架。NN-BAX通过主动选择样本针对性地改进最小能量路径，逐步微调基础模型。

### 主要发现

在Lennard-Jones和嵌入原子方法系统上的测试表明，该方法在能量和力评估方面实现了1-2个数量级的减少，同时最小能量路径的精度损失可忽略不计，并且可扩展到100维以上的系统。

### 结论

这项工作为消除科学相关系统中最小能量路径发现的计算障碍提供了有希望的步骤，表明可能以最小的精度损失将数周长的计算缩短到几小时或几天。

### 翻译

发现亚稳态之间的最小能量路径对于包括催化剂和生物分子设计在内的科学任务至关重要。然而，标准的弹力带算法需要数百到数万次计算密集型模拟，这使得在复杂系统中的应用成本过高。我们引入了神经网络贝叶斯算法执行(NN-BAX)，这是一种联合学习能量景观和最小能量路径的框架。NN-BAX通过主动选择样本针对性地改进最小能量路径，逐步微调基础模型。在Lennard-Jones和嵌入原子方法系统上的测试表明，我们的方法在能量和力评估方面实现了1-2个数量级的减少，同时最小能量路径的精度损失可忽略不计，并展示了扩展到100维以上系统的能力。因此，这项工作是消除科学相关系统中最小能量路径发现的计算障碍的有希望的一步，表明可能以最小的精度损失将数周长的计算缩短到几小时或几天。


### 论文摘要

The discovery of a minimum energy pathway (MEP) between metastable states is crucial for scientific tasks including catalyst and biomolecular design. However, the standard nudged elastic band (NEB) algorithm requires hundreds to tens of thousands of compute-intensive simulations, making applications to complex systems prohibitively expensive. We introduce Neural Network Bayesian Algorithm Execution (NN-BAX), a framework that jointly learns the energy landscape and the MEP. NN-BAX sequentially fine-tunes a foundation model by actively selecting samples targeted at improving the MEP. Tested on Lennard-Jones and Embedded Atom Method systems, our approach achieves a one to two order of magnitude reduction in energy and force evaluations with negligible loss in MEP accuracy and demonstrates scalability to >100-dimensional systems. This work is therefore a promising step towards removing the computational barrier for MEP discovery in scientifically relevant systems, suggesting that weeks-long calculations may be achieved in hours or days with minimal loss in accuracy.

---

## 42. PANDA-PLUS-Bench: A Clinical Benchmark for Evaluating Robustness of AI Foundation Models in Prostate Cancer Diagnosis

**论文链接:** [http://arxiv.org/abs/2512.14922v1](http://arxiv.org/abs/2512.14922v1)

**作者:** Joshua L. Ebbert, Dennis Della Corte

**发布时间:** 2025-12-16

**备注:** 21 pages, 5 figures, 6 Tables

### GPT解析

### 总结

本研究引入了PANDA-PLUS-Bench基准数据集，用于评估人工智能基础模型在前列腺癌格里森分级中的稳健性，揭示了模型在区分生物信号与标本特异性伪影方面的差异。

### 背景

人工智能基础模型越来越多地用于前列腺癌格里森分级，其中GP3/GP4的区别直接影响治疗决策。然而，这些模型可能通过学习标本特异性伪影而非可推广的生物特征来实现高验证准确率，限制了其在真实世界临床中的应用价值。

### 目的

创建一个专门的数据集来量化人工智能基础模型在前列腺癌格里森分级中可能出现的失败模式，即学习标本特异性伪影而非生物特征。

### 方法

开发了PANDA-PLUS-Bench基准数据集，包含来自九名患者的九个全载玻图像，具有不同的格里森模式，并在八种增强条件下提取了512x512和224x224像素分辨率的非重叠组织块。使用此基准评估了七个基础模型将生物信号与载玻片水平混淆因素分离的能力。

### 主要发现

不同模型之间的稳健性存在显著差异；专门在前列腺组织上训练的HistoEncoder表现最佳，具有最高的跨载玻片准确率和最强的载玻片水平编码；所有模型都表现出可测量的载玻片内与跨载玻片准确率差距，幅度从19.9个百分点到26.9个百分点不等。

### 结论

PANDA-PLUS-Bench为评估基础模型在前列腺癌格里森分级中的稳健性提供了专门资源，解决了该领域评估中的一个关键空白，并提供了开源工具供研究人员进一步评估模型。

### 翻译

人工智能基础模型越来越多地被用于前列腺癌格里森分级，其中GP3/GP4的区别直接影响治疗决策。然而，这些模型可能通过学习标本特异性伪影而非可推广的生物特征来实现高验证准确率，限制了其在真实世界临床中的应用价值。我们引入了PANDA-PLUS-Bench，这是一个从专家注释的前列腺活检中精心策划的基准数据集，专门用于量化这种失败模式。该基准包含来自九名独特患者的九个精心挑选的全载玻图像，包含不同的格里森模式，并在八种增强条件下提取了512x512和224x224像素分辨率的非重叠组织块。使用此基准，我们评估了七个基础模型将生物信号与载玻片水平混淆因素分离的能力。我们的结果显示不同模型之间的稳健性存在显著差异：Virchow2在大规模模型中具有最低的载玻片水平编码，但表现出第二低的跨载玻片准确率。专门在前列腺组织上训练的HistoEncoder显示出最高的跨载玻片准确率和最强的载玻片水平编码，表明组织特异性训练可能增强生物特征捕获和载玻片特异性特征。所有模型都表现出可测量的载玻片内与跨载玻片准确率差距，尽管幅度从19.9个百分点到26.9个百分点不等。我们提供了一个开源的Google Colab笔记本，使研究人员能够使用标准化指标评估额外的基础模型与我们的基准。PANDA-PLUS-Bench通过提供专门为格里森分级这一临床重要背景下的稳健性评估而设计的资源，解决了基础模型评估中的一个关键空白。


### 论文摘要

Artificial intelligence foundation models are increasingly deployed for prostate cancer Gleason grading, where GP3/GP4 distinction directly impacts treatment decisions. However, these models may achieve high validation accuracy by learning specimen-specific artifacts rather than generalizable biological features, limiting real-world clinical utility. We introduce PANDA-PLUS-Bench, a curated benchmark dataset derived from expert-annotated prostate biopsies designed specifically to quantify this failure mode. The benchmark comprises nine carefully selected whole slide images from nine unique patients containing diverse Gleason patterns, with non-overlapping tissue patches extracted at both 512x512 and 224x224 pixel resolutions across eight augmentation conditions. Using this benchmark, we evaluate seven foundation models on their ability to separate biological signal from slide-level confounders. Our results reveal substantial variation in robustness across models: Virchow2 achieved the lowest slide-level encoding among large-scale models (81.0%) yet exhibited the second-lowest cross-slide accuracy (47.2%). HistoEncoder, trained specifically on prostate tissue, demonstrated the highest cross-slide accuracy (59.7%) and the strongest slide-level encoding (90.3%), suggesting tissue-specific training may enhance both biological feature capture and slide-specific signatures. All models exhibited measurable within-slide vs. cross-slide accuracy gaps, though the magnitude varied from 19.9 percentage points to 26.9 percentage points. We provide an open-source Google Colab notebook enabling researchers to evaluate additional foundation models against our benchmark using standardized metrics. PANDA-PLUS-Bench addresses a critical gap in foundation model evaluation by providing a purpose-built resource for robustness assessment in the clinically important context of Gleason grading.

---

## 43. MMGR: Multi-Modal Generative Reasoning

**论文链接:** [http://arxiv.org/abs/2512.14691v2](http://arxiv.org/abs/2512.14691v2)

**作者:** Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Wen Xiao, Jiuxiang Gu, Nanyun Peng, Junjie Hu

**发布时间:** 2025-12-16

**备注:** work in progress

### GPT解析

### 总结

本文提出了MMGR（多模态生成推理评估和基准）框架，用于评估视频和图像生成模型在物理、逻辑、空间和时间推理方面的能力，揭示了当前模型在抽象推理和长期空间规划方面的显著局限性。

### 背景

视频基础模型能够生成视觉上逼真且时间上一致的内容，但它们作为世界模拟器的可靠性取决于是否能捕捉物理、逻辑和空间约束。现有评估指标如FVD过于关注感知质量，忽略了推理失败问题。

### 目的

开发全面的评估框架，衡量生成模型在物理、逻辑、3D空间、2D空间和时间推理上的表现，揭示当前模型局限性并指导未来研究方向。

### 方法

提出基于五种推理能力的MMGR评估框架，在抽象推理、具身导航和物理常识三个领域进行评估，应用细粒度指标要求视频和图像生成整体正确，并对领先的视频和图像模型进行基准测试。

### 主要发现

模型在物理常识任务上表现中等，但在抽象推理上表现较差（ARC-AGI准确率低于10%），在具身环境中的长期空间规划方面存在困难。当前模型存在过度依赖感知数据、全局状态一致性弱，以及奖励视觉 plausible 性而非因果正确性的局限性。

### 结论

MMGR提供了统一的诊断基准，朝向推理感知的生成世界模型发展，有助于改进现有模型并指导未来研究方向。

### 翻译

视频基础模型生成视觉上逼真且时间上一致的内容，但它们作为世界模拟器的可靠性取决于是否能捕捉物理、逻辑和空间约束。现有指标如Frechet Video Distance强调感知质量而忽略了推理失败，包括因果性、物理和全局一致性的违反。我们引入了MMGR（多模态生成推理评估和基准），这是一个基于五种推理能力的原则性评估框架：物理、逻辑、3D空间、2D空间和时间。MMGR评估了三个领域的生成推理：抽象推理、具身导航和物理常识。MMGR应用了细粒度指标，要求视频和图像生成在整体上正确。我们对领先的视频模型和图像模型进行了基准测试，揭示了不同领域之间存在明显的性能差距。模型在物理常识任务上表现中等，但在抽象推理上表现较差，并且在具身环境中的长期空间规划方面存在困难。我们的分析揭示了当前模型的关键局限性，包括过度依赖感知数据、全局状态一致性弱，以及奖励视觉 plausible 性而非因果正确性的目标。MMGR提供了一个统一的诊断基准和朝向推理感知的生成世界模型的发展路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的问题是现有视频和图像生成模型缺乏对现实世界的基本推理能力，包括物理、逻辑、空间和时间约束的理解。这个问题很重要，因为现有评估指标主要关注视觉质量而非推理能力，导致模型可能生成违反物理规律或逻辑一致性的内容（如台球穿过彼此），这些内容虽然视觉吸引人但不符合现实世界规律。提高生成模型的推理能力对电影制作、科学可视化和机器人等领域至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到现有评估指标的局限性，基于人类认知理论提出了五种核心推理能力框架。他们设计了三个互补评估领域：抽象推理、具身导航和物理常识。借鉴了人类认知理论中的'核心知识'概念、ARC-AGI评估任务、VideoPhy本体以及VLM评估方法。他们结合了现有工作并进行了创新，构建了一个系统评估生成模型推理能力的全新框架。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于五种推理能力（物理、逻辑、3D空间、2D空间、时间推理）构建评估框架，使用细粒度指标要求整体正确性而非部分成功。整体流程包括：1) 创建包含1,853个样本的评估数据集；2) 对每个提示生成5个样本；3) 使用VLM自动评估和人工评估验证；4) 分析模型在不同推理能力上的表现。三个评估领域分别是抽象推理（迷宫、数独等）、具身导航（3D导航等）和物理常识（体育活动等）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 首个系统评估生成模型推理能力的框架；2) 五种核心推理能力框架；3) 三个互补评估领域；4) 细粒度整体正确性评估指标；5) 多模态评估视频和图像模型；6) 人机结合评估。相比之前工作，MMGR从'理解'转向'生成'评估，超越感知保真度关注逻辑一致性，提供全面评估而非特定失败模式，并使用更严格的整体正确性指标。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MMGR通过系统评估生成模型在物理、逻辑、空间和时间推理方面的能力，揭示了当前模型在抽象推理和长期空间规划上的严重不足，为开发真正具有世界模拟能力的生成模型提供了明确的评估框架和改进方向。'}


### 论文摘要

Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models.

---

