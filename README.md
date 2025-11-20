# 今日论文推荐 - 2025-11-20

共 161 篇论文

---

## 1. Text2Loc++: Generalizing 3D Point Cloud Localization from Natural Language

**论文链接:** [http://arxiv.org/abs/2511.15308v1](http://arxiv.org/abs/2511.15308v1)

**作者:** Yan Xia, Letian Shi, Yilin Di, Joao F. Henriques, Daniel Cremers

**发布时间:** 2025-11-19

**备注:** This paper builds upon and extends our earlier conference paper Text2Loc presented at CVPR 2024

### GPT解析

### 总结

Text2Loc++是一种新型神经网络，用于通过自然语言描述定位3D点云子图，在粗到精定位流程中实现语言和点云间的有效跨模态对齐。

### 背景

研究使用复杂多样的自然语言描述来定位3D点云子图的挑战，需要处理多样化的城市环境和语言表达。

### 目的

开发一个能够处理复杂语言描述并准确定位3D点云子图的系统，提高跨模态对齐的准确性和鲁棒性。

### 方法

Text2Loc++采用粗到精定位流程：全局位置识别阶段结合预训练语言模型、层次Transformer与最大池化(HTM)和基于注意力的点云编码器；引入掩码实例训练(MIT)过滤非对齐对象；采用模态感知层次对比学习(MHCL)增强嵌入空间；精确定位阶段使用原型地图克隆(PMC)和级联交叉注意力Transformer(CCAT)的轻量级框架。

### 主要发现

在KITTI360Pose数据集上，Text2Loc++比现有方法性能提高15%；在新数据集上展现出强大的泛化能力，能有效处理复杂的语言表达和多样化的城市环境。

### 结论

Text2Loc++通过创新的跨模态对齐技术和层次化方法，显著提高了基于自然语言的3D点云子图定位性能，具有广泛的应用前景。

### 翻译

我们解决了使用复杂多样的自然语言描述来定位3D点云子图的问题，并提出了Text2Loc++，一种为粗到精定位流程中语言和点云间有效跨模态对齐而设计的新型神经网络。为支持基准测试，我们引入了一个新的城市规模数据集，包含来自多样化城市场景的彩色和非彩色点云，并将位置描述组织为三个语言复杂度级别。在全局位置识别阶段，Text2Loc++结合了预训练语言模型和层次Transformer与最大池化(HTM)用于句子级语义，并采用基于注意力的点云编码器进行空间理解。我们进一步提出了掩码实例训练(MIT)来过滤未对齐对象并提高多模态鲁棒性。为增强嵌入空间，我们引入了模态感知层次对比学习(MHCL)，包括跨模态、子图、文本和实例级损失。在精确定位阶段，我们完全移除了显式文本-实例匹配，并设计了一个基于原型地图克隆(PMC)和级联交叉注意力Transformer(CCAT)的轻量级但强大的框架。在KITTI360Pose数据集上的大量实验表明，Text2Loc++比现有方法性能提高高达15%。此外，在新数据集上评估时，所提出的模型表现出强大的泛化能力，能有效处理复杂的语言表达和多样化的城市环境。代码和数据集将公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决使用自然语言描述来定位3D点云地图中的位置问题。这个问题非常重要，因为它解决了'最后一公里问题'，在自动驾驶、商品配送和车辆接送等实际应用中至关重要。GPS信号在高层建筑密集或植被茂密的环境中往往会减弱或失效，而人类通常依赖口头指引来找到正确目的地，特别是在不熟悉或被遮挡的位置。这种能力使自主智能体能够与人类有效协作进行路径规划。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3D点云定位面临的四个主要挑战：自然语言的模糊性、语言和3D数据结构不同、文本描述和子地图不能准确对齐、文本风格和点云分布在不同城市/国家间差异大。然后创建了一个新的城市规模文本-点云定位数据集，包含来自不同城市和国家的点云。方法设计上，作者使用预训练语言模型T5和分层Transformer处理文本，基于注意力的点云编码器处理空间信息，引入掩码实例训练过滤非对齐对象，提出模态感知分层对比学习增强嵌入空间，并在精确定位阶段移除显式的文本-实例匹配。作者借鉴了CLIP的对比学习方法、Text2Pose和Text2Loc等先前的工作、PointNet++作为点云编码器、Transformer架构处理文本，以及LoRA方法微调T5编码器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用自然语言与3D点云之间的跨模态对齐实现定位，采用粗到精的两阶段定位策略，通过掩码实例训练和模态感知分层对比学习处理多对多匹配的模糊性，并在精确定位阶段完全移除文本-实例匹配。整体流程分为两个阶段：全局位置识别阶段，文本分支使用T5和分层Transformer处理文本，点云分支使用PointNet++提取特征，结合掩码实例训练和模态感知分层对比学习；精确定位阶段，使用原型地图克隆生成多样化子地图变体，通过级联交叉注意力Transformer融合文本和子云特征，最后用轻量级多层感知器回归目标位置。训练策略包括文本蒸馏处理复杂文本和LoRA微调T5编码器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)构建了包含不同城市点云和三个层次语言复杂度的新数据集；2)设计分层Transformer架构处理文本层次结构；3)提出掩码实例训练过滤非相关对象；4)引入模态感知分层对比学习增强嵌入空间；5)完全移除精确定位阶段的文本-实例匹配。相比之前工作，Text2Loc++能处理复杂多样的自然语言描述，在不同城市和国家间表现出强大的泛化能力，通过无匹配设计简化模型并提高效率，引入文本蒸馏处理复杂文本，结合多种对比学习损失使模型学习更有意义的表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Text2Loc++通过创新的跨模态对齐技术和无匹配精确定位框架，实现了从复杂自然语言描述到多样化城市环境3D点云的高效定位，显著提升了定位准确率和跨城市泛化能力。'}


### 论文摘要

We tackle the problem of localizing 3D point cloud submaps using complex and diverse natural language descriptions, and present Text2Loc++, a novel neural network designed for effective cross-modal alignment between language and point clouds in a coarse-to-fine localization pipeline. To support benchmarking, we introduce a new city-scale dataset covering both color and non-color point clouds from diverse urban scenes, and organize location descriptions into three levels of linguistic complexity. In the global place recognition stage, Text2Loc++ combines a pretrained language model with a Hierarchical Transformer with Max pooling (HTM) for sentence-level semantics, and employs an attention-based point cloud encoder for spatial understanding. We further propose Masked Instance Training (MIT) to filter out non-aligned objects and improve multimodal robustness. To enhance the embedding space, we introduce Modality-aware Hierarchical Contrastive Learning (MHCL), incorporating cross-modal, submap-, text-, and instance-level losses. In the fine localization stage, we completely remove explicit text-instance matching and design a lightweight yet powerful framework based on Prototype-based Map Cloning (PMC) and a Cascaded Cross-Attention Transformer (CCAT). Extensive experiments on the KITTI360Pose dataset show that Text2Loc++ outperforms existing methods by up to 15%. In addition, the proposed model exhibits robust generalization when evaluated on the new dataset, effectively handling complex linguistic expressions and a wide variety of urban environments. The code and dataset will be made publicly available.

---

## 2. Wave-Former: Through-Occlusion 3D Reconstruction via Wireless Shape Completion

**论文链接:** [http://arxiv.org/abs/2511.14152v2](http://arxiv.org/abs/2511.14152v2)

**作者:** Laura Dodds, Maisy Lam, Waleed Akbar, Yibo Cheng, Fadel Adib

**发布时间:** 2025-11-18

### GPT解析

### 总结

Wave-Former是一种新方法，能够高精度重建完全遮挡的日常物品的三维形状，利用毫米波信号穿透遮挡物并从隐藏物体反射。

### 背景

过去的毫米波重建方法存在覆盖范围有限和高噪声的问题，限制了3D形状重建的准确性。

### 目的

开发一种能够高精度重建完全遮挡物品3D形状的方法，以扩展在机器人、增强现实和物流等领域的应用。

### 方法

Wave-Former采用一个三阶段流程：提出候选几何表面、使用专门为毫米波信号设计的基于transformer的形状补全模型，以及执行熵引导的表面选择。该方法使用完全合成的点云进行训练。

### 主要发现

Wave-Former在直接比较中将召回率从54%提高到72%，同时保持85%的高精度，并且能够很好地泛化到真实世界数据。

### 结论

Wave-Former通过利用毫米波信号的物理特性和创新的三阶段流程，实现了对完全遮挡物品的高精度3D重建，为多个领域开辟了新的应用可能性。

### 翻译

我们提出了Wave-Former，一种新颖的方法，能够高精度地重建完全遮挡的、多样化的日常物品的三维形状。这种能力可以开启横跨机器人、增强现实和物流等领域的全新应用。我们的方法利用毫米波无线信号，这些信号可以穿透常见的遮挡物并从隐藏的物体上反射。与过去的毫米波重建方法相比（这些方法存在覆盖范围有限和高噪声的问题），Wave-Former引入了一种物理感知的形状补全模型，能够推断完整的3D几何形状。Wave-Former设计的核心是一个新颖的三阶段流程，该流程通过结合毫米波信号的物理特性，将原始无线信号与基于视觉的形状补全的最新进展联系起来。该流程提出候选几何表面，采用专门为毫米波信号设计的基于transformer的形状补全模型，并最终执行熵引导的表面选择。这使得Wave-Former能够完全使用合成的点云进行训练，同时展示出对真实数据的出色泛化能力。与最先进的基线进行直接比较时，Wave-Former将召回率从54%提高到72%，同时保持85%的高精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决如何重建完全被遮挡物体的完整3D形状的问题。这个问题在现实中非常重要，因为光学传感器（如相机和激光雷达）在物体被完全遮挡时无法工作，而现有毫米波重建方法存在覆盖范围有限、噪声高的问题。成功解决这个问题可以应用于机器人抓取、增强现实、物流扫描等多个领域，使系统能够'看穿'障碍物识别物体。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到毫米波信号能够穿透常见遮挡物并从隐藏物体反射的特性，但现有毫米波重建方法只能捕捉物体表面的一小部分。他们尝试将基于视觉的形状完成模型应用于毫米波数据，但效果不佳。因此，作者设计了一种物理感知的训练框架，将毫米波的物理特性（如镜面反射、低信噪比）直接嵌入到学习过程中。他们借鉴了基于transformer的形状完成模型（如PoinTr）和毫米波成像技术（如Backprojection和mmNorm），但进行了专门改进以适应毫米波的物理特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个物理感知的训练框架，将毫米波信号的物理特性直接嵌入到学习过程中。这包括：1)引入镜面反射感知的归纳偏差，模拟毫米波信号的稀疏镜面反射；2)考虑反射依赖的可见性模式，使模型能预测可能无法观察到的区域；3)调整损失函数以联合优化嘈杂输入和完成缺失表面。整体实现流程分为物理感知训练管道和真实世界推理过程：训练阶段使用合成数据学习物理特性；推理阶段包括：将原始毫米波信号转换为候选部分表面，应用物理感知的形状完成模型，最后通过熵引导的表面选择输出最终3D重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)物理感知的毫米波形状完成框架，首次实现对多样化完全遮挡物体的完整3D重建；2)最先进的性能，在真实世界数据集上将召回率从54%提高到72%，同时保持85%的精度；3)消除了对真实世界毫米波训练数据的需求，使用合成数据即可实现良好泛化。相比之前工作，不同之处在于：过去方法只能重建面向雷达的表面，而Wave-Former能重建完整3D形状；过去方法受限于有限对象类别，而Wave-Former可处理多样化物体；Wave-Former专门考虑了毫米波的物理特性，而现有方法未充分利用这些特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Wave-Former提出了一种物理感知的毫米波形状完成框架，首次实现了对多样化完全遮挡物体的高精度完整3D重建，为机器人、增强现实和物流等领域开辟了新的应用可能性。'}


### 论文摘要

We present Wave-Former, a novel method capable of high-accuracy 3D shape reconstruction for completely occluded, diverse, everyday objects. This capability can open new applications spanning robotics, augmented reality, and logistics. Our approach leverages millimeter-wave (mmWave) wireless signals, which can penetrate common occlusions and reflect off hidden objects. In contrast to past mmWave reconstruction methods, which suffer from limited coverage and high noise, Wave-Former introduces a physics-aware shape completion model capable of inferring full 3D geometry. At the heart of Wave-Former's design is a novel three-stage pipeline which bridges raw wireless signals with recent advancements in vision-based shape completion by incorporating physical properties of mmWave signals. The pipeline proposes candidate geometric surfaces, employs a transformer-based shape completion model designed specifically for mmWave signals, and finally performs entropy-guided surface selection. This enables Wave-Former to be trained using entirely synthetic point-clouds, while demonstrating impressive generalization to real-world data. In head-to-head comparisons with state-of-the-art baselines, Wave-Former raises recall from 54% to 72% while maintaining a high precision of 85%.

---

## 3. Learning from Imperfect Labels: A Physics-Aware Neural Operator with Application to DAS Data Denoising

**论文链接:** [http://arxiv.org/abs/2511.15638v1](http://arxiv.org/abs/2511.15638v1)

**作者:** Yang Cui, Denis Anikiev, Umair Bin Waheed, Yangkang Chen

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了一种物理感知UFNO(PAUFNO)框架，通过物理感知损失函数和改进的U-Net增强傅里叶神经算子，有效解决了监督深度学习在有缺陷标签上训练时性能下降的问题，并在分布式声学传感数据去噪任务中表现出色。

### 背景

监督深度学习方法通常需要大型数据集和高质量标签才能实现可靠的预测。然而，当在有缺陷的标签上训练时，它们的性能往往会下降。

### 目的

应对标签不完美导致的深度学习方法性能下降的挑战，开发一种能够从有缺陷标签中有效学习的框架。

### 方法

1. 提出一种物理感知损失函数作为惩罚项，减轻训练过程中的标签缺陷；2. 引入改进的U-Net增强傅里叶神经算子(UFNO)，实现高保真特征表示；3. 结合上述两个组件，开发物理感知UFNO框架；4. 应用框架于犹他FORGE站分布式声学传感数据的去噪；5. 采用基于修补的数据增强策略，包括提升步骤、空间域卷积操作、谱卷积和投影层。

### 主要发现

大量的数值实验表明，所提出的框架实现了卓越的去噪性能，有效地增强了DAS记录，并高精度地恢复了隐藏信号。

### 结论

物理感知UFNO框架能够有效地从有缺陷的标签中学习，在DAS数据去噪任务中表现出色。

### 翻译

监督深度学习方法通常需要大型数据集和高质量标签才能实现可靠的预测。然而，当在有缺陷的标签上训练时，它们的性能往往会下降。为了应对这一挑战，我们提出了一种物理感知损失函数，作为惩罚项，以减轻训练过程中的标签缺陷。此外，我们引入了一种改进的U-Net增强傅里叶神经算子(UFNO)，该算子能够在函数空间中利用算子学习的优势，实现高保真特征表示。通过结合这两个组件，我们开发了一种物理感知UFNO(PAUFNO)框架，能够有效地从有缺陷的标签中学习。为了评估所提出的框架，我们将其应用于犹他FORGE站分布式声学传感(DAS)数据的去噪。标签数据是使用集成滤波方法生成的，但在近井筒通道中仍包含残余耦合噪声。去噪工作流程包括基于修补的数据增强策略，包括提升步骤、空间域卷积操作、谱卷积以及将数据恢复到所需形状的投影层。大量的数值实验表明，所提出的框架实现了卓越的去噪性能，有效地增强了DAS记录，并高精度地恢复了隐藏信号。


### 论文摘要

Supervised deep learning methods typically require large datasets and high-quality labels to achieve reliable predictions. However, their performance often degrades when trained on imperfect labels. To address this challenge, we propose a physics-aware loss function that serves as a penalty term to mitigate label imperfections during training. In addition, we introduce a modified U-Net-Enhanced Fourier Neural Operator (UFNO) that achieves high-fidelity feature representation while leveraging the advantages of operator learning in function space. By combining these two components, we develop a physics-aware UFNO (PAUFNO) framework that effectively learns from imperfect labels. To evaluate the proposed framework, we apply it to the denoising of distributed acoustic sensing (DAS) data from the Utah FORGE site. The label data were generated using an integrated filtering-based method, but still contain residual coupling noise in the near-wellbore channels. The denoising workflow incorporates a patching-based data augmentation strategy, including an uplifting step, spatial-domain convolutional operations, spectral convolution, and a projection layer to restore data to the desired shape. Extensive numerical experiments demonstrate that the proposed framework achieves superior denoising performance, effectively enhancing DAS records and recovering hidden signals with high accuracy.

---

## 4. 论文ID: 2511.15633v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15633v1.json'

---

## 5. CODE: A global approach to ODE dynamics learning

**论文链接:** [http://arxiv.org/abs/2511.15619v1](http://arxiv.org/abs/2511.15619v1)

**作者:** Nils Wildt, Daniel M. Tartakovsky, Sergey Oladyshkin, Wolfgang Nowak

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了ChaosODE（CODE）方法，一种多项式混沌ODE展开技术，用于从稀疏观测数据中学习物理系统的动力学行为。CODE使用任意多项式混沌展开（aPCE）表示ODE的右侧，实现动力学的全局正交多项式表示。实验表明，CODE在新的初始条件下表现出卓越的外推能力，优于神经网络和核逼近器方法。

### 背景

常微分方程（ODEs）是描述物理系统观测动力学的传统方法。科学家通常假设动力学行为，提出数学模型，并将其预测与数据进行比较。现代计算和算法进步使得可以直接从观测中纯粹数据驱动地学习控制动力学。然而，高时间分辨率的密集测量既繁琐又昂贵，因此通常只有稀疏采样数据可用。

### 目的

开发一种能够从稀疏观测数据中学习动力学系统的方法，特别是在数据稀缺和测量噪声存在的情况下，保持良好的外推能力。

### 方法

引入ChaosODE（CODE），这是一种多项式混沌ODE展开方法，使用任意多项式混沌展开（aPCE）来表示ODE的右侧，实现动力学的全局正交多项式表示。在Lotka-Volterra系统上进行实验，评估不同噪声水平、初始条件和远期预测（包括未见过的初始条件）下的性能。

### 主要发现

CODE即使在新的初始条件下评估时也表现出卓越的外推能力，相比使用神经网络（NeuralODE）或核逼近器（KernelODE）作为右侧表示器的方法具有优势。观察到NeuralODE和KernelODE的高灵活性在数据稀缺和测量噪声情况下会降低外推能力。

### 结论

CODE是一种从稀疏数据学习动力学的有效方法，具有良好的外推能力。研究还提供了动力学学习问题的稳健优化实用指南，并在配套代码中进行了说明。

### 翻译

常微分方程（ODEs）是描述物理系统观测动力学的传统方式。科学家通常假设动力学行为，提出数学模型，并将其预测与数据进行比较。然而，现代计算和算法进步现在使得可以直接从观测中纯粹数据驱动地学习控制动力学。在数据驱动设置中，人们学习ODE的右侧（RHS）。通常假设密集测量，但高时间分辨率通常既繁琐又昂贵。因此，通常只有稀疏采样数据。在这项工作中，我们引入了ChaosODE（CODE），这是一种多项式混沌ODE展开，其中我们使用任意多项式混沌展开（aPCE）来表示ODE的右侧，从而实现动力学的全局正交多项式表示。我们在Lotka-Volterra系统上的几个实验中评估了CODE的性能，包括不同的噪声水平、初始条件和远期预测，甚至包括以前未见过的初始条件。即使在新的初始条件下评估时，CODE也表现出卓越的外推能力，相比使用神经网络（NeuralODE）或核逼近器（KernelODE）作为右侧表示器的方法具有优势。我们观察到，在数据稀缺和测量噪声情况下，NeuralODE和KernelODE的高灵活性会降低外推能力。最后，我们提供了动力学学习问题稳健优化的实用指南，并在配套代码中进行了说明。


### 论文摘要

Ordinary differential equations (ODEs) are a conventional way to describe the observed dynamics of physical systems. Scientists typically hypothesize about dynamical behavior, propose a mathematical model, and compare its predictions to data. However, modern computing and algorithmic advances now enable purely data-driven learning of governing dynamics directly from observations. In data-driven settings, one learns the ODE's right-hand side (RHS). Dense measurements are often assumed, yet high temporal resolution is typically both cumbersome and expensive. Consequently, one usually has only sparsely sampled data. In this work we introduce ChaosODE (CODE), a Polynomial Chaos ODE Expansion in which we use an arbitrary Polynomial Chaos Expansion (aPCE) for the ODE's right-hand side, resulting in a global orthonormal polynomial representation of dynamics. We evaluate the performance of CODE in several experiments on the Lotka-Volterra system, across varying noise levels, initial conditions, and predictions far into the future, even on previously unseen initial conditions. CODE exhibits remarkable extrapolation capabilities even when evaluated under novel initial conditions and shows advantages compared to well-examined methods using neural networks (NeuralODE) or kernel approximators (KernelODE) as the RHS representer. We observe that the high flexibility of NeuralODE and KernelODE degrades extrapolation capabilities under scarce data and measurement noise. Finally, we provide practical guidelines for robust optimization of dynamics-learning problems and illustrate them in the accompanying code.

---

## 6. SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2511.15605v1](http://arxiv.org/abs/2511.15605v1)

**作者:** Senyu Fei, Siyin Wang, Li Ji, Ao Li, Shiduo Zhang, Liming Liu, Jinlong Hou, Jingjing Gong, Xianzhong Zhao, Xipeng Qiu

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了一种名为自参考策略优化(SRPO)的新型VLA-RL框架，通过利用模型自身成功轨迹作为自我参考，解决了传统VLA模型对专家演示的依赖问题，并在LIBERO基准测试上实现了显著性能提升。

### 背景

视觉-语言-行动(VLA)模型在机器人操作中表现出色，但严重依赖专家演示，导致演示偏差和性能限制。强化学习(RL)作为训练后策略面临奖励稀疏性问题，现有方法依赖二元成功指标，浪费了失败轨迹中的有价值信息，导致训练效率低。

### 目的

开发一种无需外部演示或手动奖励工程的VLA-RL框架，通过利用模型自身成功轨迹来评估进展，提高训练效率和性能。

### 方法

提出自参考策略优化(SRPO)框架，利用模型在当前训练批次中生成的成功轨迹作为自我参考，为失败尝试分配进展奖励。核心创新是使用潜在世界表示来稳健衡量行为进展，利用世界模型潜在空间的压缩、可转移编码来捕获跨环境的进展模式，实现准确的轨迹比较。

### 主要发现

在LIBERO基准测试上，SRPO从48.9%成功的监督基线开始，仅用200个RL步骤就达到了99.2%的最先进成功率，相对提高了103%，无需任何额外监督。在LIBERO-Plus基准测试上，SRPO表现出显著的鲁棒性，性能提高了167%。

### 结论

SRPO通过利用模型自身成功轨迹作为自我参考，有效解决了传统VLA模型对专家演示的依赖问题，显著提高了训练效率和性能，展现出强大的泛化能力和鲁棒性。

### 翻译

视觉-语言-行动(VLA)模型在机器人操作中表现出色，但其严重依赖专家演示的限制导致了演示偏差并限制了性能。强化学习(RL)是克服这些限制的重要训练后策略，然而当前VLA-RL方法，包括基于群体的优化方法，都受到严重奖励稀疏性的困扰。依赖二元成功指标浪费了失败轨迹中的有价值信息，导致训练效率低下。为解决这一问题，我们提出了自参考策略优化(SRPO)，一种新型的VLA-RL框架。SRPO通过利用当前训练批次中生成的模型自身成功轨迹作为自我参考，消除了对外部演示或手动奖励工程的依赖。这使得我们能够为失败的尝试分配进展奖励。核心创新是使用潜在世界表示来稳健地衡量行为进展。我们不依赖原始像素或需要领域特定的微调，而是利用来自世界模型潜在空间的压缩、可转移编码。这些表示自然地捕获跨环境的进展模式，能够进行准确、通用的轨迹比较。在LIBERO基准测试上的实证评估证明了SRPO的效率和有效性。从48.9%成功的监督基线开始，SRPO仅用200个RL步骤就达到了99.2%的最新的成功率，代表了103%的相对提升，无需任何额外监督。此外，SRPO在LIBERO-Plus基准测试上表现出显著的鲁棒性，实现了167%的性能提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决Vision-Language-Action (VLA)模型在机器人操作中面临的奖励稀疏问题。现有方法如GRPO仅依赖稀疏的二元成功/失败奖励，无法有效利用失败轨迹中的有价值信息，导致训练效率低下。这个问题在现实中限制了机器人学习效率，增加了部署成本；在研究中则阻碍了VLA模型性能的提升和泛化能力的增强，因为大量有价值的失败尝试信息被浪费。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：GRPO仅依靠结果奖励而丢弃失败轨迹信息，而手工过程奖励模型需要昂贵的外部演示和任务特定工程。基于此，作者提出自参考学习范式，将问题从'如何获得专家标签'转变为'如何从自身成功中提取进度奖励'。核心创新在于利用世界模型的潜在表示来测量行为进展，而非原始像素。方法借鉴了GRPO的群体优化框架，但改进了奖励机制；参考了过程监督思想，但避免了对外部演示的依赖；采用了轨迹级奖励而非细粒度奖励塑形，以避免收敛到次优解。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用模型自身在当前训练批次中生成的成功轨迹作为自参考，为失败的尝试分配进度奖励，从而充分利用失败轨迹中的信息。整体实现流程包括：1) 收集成功和失败轨迹；2) 使用预训练的世界模型提取轨迹的潜在表示；3) 计算失败轨迹与成功轨迹聚类中心的距离作为行为相似性度量；4) 根据距离计算进度奖励（成功轨迹奖励为1.0，失败轨迹奖励与距离成反比）；5) 使用这些奖励进行优势估计和策略优化，在KL正则化下提升策略性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 自参考学习范式，消除对外部演示的依赖；2) 基于潜在世界表示的进度奖励方法，克服传统像素级世界模型的泛化限制；3) 轨迹级奖励设计，避免收敛到次优解；4) 高效利用失败轨迹信息。相比之前工作的不同：与GRPO相比，SRPO能从失败轨迹中提取学习信号而非丢弃；与手工过程奖励模型相比，SRPO无需专家演示和任务特定工程；与传统像素级世界模型相比，SRPO具有更好的泛化能力且无需领域特定微调；与基于ImageBind等方法相比，SRPO提供更单调、更符合物理规律的奖励信号。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SRPO通过自参考学习和潜在世界表示，提出了一种无需外部演示的高效VLA强化学习框架，显著提升了训练效率和任务性能，同时增强了模型的泛化能力。'}


### 论文摘要

Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.

---

## 7. US-X Complete: A Multi-Modal Approach to Anatomical 3D Shape Recovery

**论文链接:** [http://arxiv.org/abs/2511.15600v1](http://arxiv.org/abs/2511.15600v1)

**作者:** Miruna-Alexandra Gafencu, Yordanka Velikova, Nassir Navab, Mohammad Farid Azampour

**发布时间:** 2025-11-19

**DOI:** 10.1007/978-3-032-06774-6_17

**备注:** Accepted at the Workshop on Shape in Medical Imaging at MICCAI 2025

### GPT解析

### 总结

本文提出了一种新型的多模态深度学习方法，通过结合单张X射线图像的信息来补充3D超声成像中被遮挡的脊柱解剖结构，特别是解决超声成像中椎体可视化的局限性问题。

### 背景

超声成像是一种无辐射、成本效益高的实时成像技术，可显示脊柱标志、脊柱旁软组织和神经血管结构，在脊柱手术中有重要价值。然而，由于骨骼引起的声影效应，超声在显示完整椎体解剖结构方面存在固有局限性。

### 目的

开发一种方法，通过利用单张X射线图像中的互补信息，来完成3D超声中被遮挡的解剖结构，特别是解决超声成像中椎体可视化的局限性。

### 方法

作者提出了一种多模态深度学习方法，通过生成成对的训练数据来支持模型训练：(1) 模拟X射线扫描的2D侧位椎体视图，和(2) 模拟超声脊柱成像中有限可见性和遮挡的3D部分椎体表示。该方法整合了两种成像模态的形态学信息。

### 主要发现

与当前最先进的3D超声椎体完成技术相比，该方法在椎体重建方面显示出显著改进。通过初步的模型研究，实现了更准确、完整的腰椎脊柱体积可视化，可以叠加在超声扫描上，而无需与计算机断层扫描等术前成像模态进行配准。

### 结论

整合单张X射线投影可以减轻超声的主要局限性，同时保留其作为主要成像模态的优势。该方法为未来临床转化提供了基础。

### 翻译

超声提供了一种无辐射、成本效益高的解决方案，可实时显示脊柱标志、脊柱旁软组织和神经血管结构，使其在脊柱手术中的术中引导具有重要价值。然而，由于骨骼引起的声影效应，超声在显示完整椎体解剖结构方面存在固有局限性。在这项工作中，我们提出了一种新型的多模态深度学习方法，通过利用单张X射线图像中的互补信息来完成3D超声中被遮挡的解剖结构。为了支持训练，我们生成了成对的训练数据，包括：(1) 模拟X射线扫描的2D侧位椎体视图，和(2) 模拟超声脊柱成像中有限可见性和遮挡的3D部分椎体表示。我们的方法整合了两种成像模态的形态学信息，并在椎体重建方面显示出比当前最先进的3D超声椎体完成技术显著改进。我们进行了模型研究作为未来临床转化的初步步骤，实现了更准确、完整的腰椎脊柱体积可视化叠加在超声扫描上，而无需与计算机断层扫描等术前成像模态进行配准。这表明，整合单张X射线投影可以减轻超声的主要局限性，同时保留其作为主要成像模态的优势。代码和数据可在https://github.com/miruna20/US-X-Complete获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是超声成像在脊柱手术中的局限性问题，特别是由于骨骼声影效应导致椎体等结构无法完整可视化的难题。这个问题在现实中非常重要，因为超声成像具有无辐射、实时、成本低的优点，对脊柱手术中的术中引导极为有价值，但无法完整显示脊柱解剖结构限制了其独立应用。解决这一问题能提高手术精度，减少并发症和手术创伤，同时避免依赖有辐射的CT扫描。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：基于CT的配准方法技术挑战大且依赖术前CT，而纯超声重建方法继承了超声的基本局限性。因此，作者提出结合超声和X光的互补优势——超声提供局部实时信息，X光提供全局几何尺度。作者借鉴了点云形状完成工作（如PCN、PoinTr）和多模态形状完成方法（如ViPC、CSDN），但专门针对脊柱成像特点设计了新框架，通过将两种模态注册到共享3D表示空间，实现统一的多模态部分观测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合超声和X光两种成像模式的互补优势，通过多模态深度学习方法完成被遮挡的3D解剖结构重建。整体流程包括：1)数据生成阶段，从CT模拟超声和X光数据并构建联合3D点云；2)两阶段完成流程，粗略阶段使用早期融合模块整合全局特征生成椎体模板，精细阶段使用晚期融合策略结合局部几何细节进行精细重建；3)评估阶段使用多种指标评估重建准确性，特别关注椎体和椎弓的完成质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个针对脊柱成像的多模态形状恢复框架；2)融合两种模态解剖特征的联合3D点云表示空间；3)两阶段完成架构结合粗略模板生成和精细细节重建；4)新型早期融合和晚期融合模块。相比之前工作，不同之处在于：解决了超声声影效应限制，提供更准确的椎体重建；避免了基于CT配准方法的复杂流程和辐射暴露；专门针对医学成像特点设计，处理了医学图像特有的结构化遮挡问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合超声和X光的多模态深度学习方法，有效解决了脊柱超声成像中的声影效应问题，实现了更准确、完整的3D脊柱解剖结构重建，无需依赖术前CT扫描或复杂配准流程。'}


### 论文摘要

Ultrasound offers a radiation-free, cost-effective solution for real-time visualization of spinal landmarks, paraspinal soft tissues and neurovascular structures, making it valuable for intraoperative guidance during spinal procedures. However, ultrasound suffers from inherent limitations in visualizing complete vertebral anatomy, in particular vertebral bodies, due to acoustic shadowing effects caused by bone. In this work, we present a novel multi-modal deep learning method for completing occluded anatomical structures in 3D ultrasound by leveraging complementary information from a single X-ray image. To enable training, we generate paired training data consisting of: (1) 2D lateral vertebral views that simulate X-ray scans, and (2) 3D partial vertebrae representations that mimic the limited visibility and occlusions encountered during ultrasound spine imaging. Our method integrates morphological information from both imaging modalities and demonstrates significant improvements in vertebral reconstruction (p < 0.001) compared to state of art in 3D ultrasound vertebral completion. We perform phantom studies as an initial step to future clinical translation, and achieve a more accurate, complete volumetric lumbar spine visualization overlayed on the ultrasound scan without the need for registration with preoperative modalities such as computed tomography. This demonstrates that integrating a single X-ray projection mitigates ultrasound's key limitation while preserving its strengths as the primary imaging modality. Code and data can be found at https://github.com/miruna20/US-X-Complete

---

## 8. A Hybrid CNN-ViT-GNN Framework with GAN-Based Augmentation for Intelligent Weed Detection in Precision Agriculture

**论文链接:** [http://arxiv.org/abs/2511.15535v1](http://arxiv.org/abs/2511.15535v1)

**作者:** Pandiyaraju V, Abishek Karthik, Sreya Mynampati, Poovarasan L, D. Saraswathi

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究提出了一种混合深度学习框架用于杂草检测，结合了CNN、ViT和GNN，并使用GAN进行数据增强和自监督对比预训练，在多个基准数据集上达到了99.33%的高准确率，可实现实时边缘设备部署，促进可持续精准农业。

### 背景

杂草检测是精准农业的关键环节，准确的物种识别使农民能够选择性应用除草剂，符合可持续农业作物管理理念。

### 目的

开发一种混合深度学习框架，用于在各种田间条件下实现鲁棒的杂草检测。

### 方法

结合卷积神经网络(CNNs)、视觉变换器(ViTs)和图神经网络(GNNs)构建模型；采用基于生成对抗网络(GAN)的数据增强方法平衡类别分布；应用自监督对比预训练方法从有限标注数据中学习更多特征。

### 主要发现

在多个基准数据集上实现了99.33%的准确率、精确率、召回率和F1分数，表明模型具有优异的性能。

### 结论

所提出的模型架构能够捕获局部、全局和关系特征表示，具有高可解释性和适应性；可实时部署到边缘设备进行自动化杂草检测，减少对除草剂的过度依赖，提供可扩展的可持续精准农业解决方案。

### 翻译

杂草检测是精准农业的重要组成部分，因为准确的物种识别使农民能够选择性地应用除草剂，并符合可持续农业作物管理。本文提出了一种用于杂草检测的混合深度学习框架，利用卷积神经网络(CNNs)、视觉变换器(ViTs)和图神经网络(GNNs)来构建对多种田间条件的鲁棒性。采用基于生成对抗网络(GAN)的数据增强方法来平衡类别分布并提高模型的泛化能力。此外，自监督对比预训练方法有助于从有限的标注数据中学习更多特征。实验结果在多个基准数据集上取得了99.33%的准确率、精确率、召回率和F1分数的优异结果。所提出的模型架构能够捕获局部、全局和关系特征表示，并提供高可解释性和适应性。实际上，该框架可以实时、高效地部署到边缘设备进行自动化杂草检测，减少对除草剂的过度依赖，并提供可扩展的可持续精准农业选择。


### 论文摘要

The task of weed detection is an essential element of precision agriculture since accurate species identification allows a farmer to selectively apply herbicides and fits into sustainable agriculture crop management. This paper proposes a hybrid deep learning framework recipe for weed detection that utilizes Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Graph Neural Networks (GNNs) to build robustness to multiple field conditions. A Generative Adversarial Network (GAN)-based augmentation method was imposed to balance class distributions and better generalize the model. Further, a self-supervised contrastive pre-training method helps to learn more features from limited annotated data. Experimental results yield superior results with 99.33% accuracy, precision, recall, and F1-score on multi-benchmark datasets. The proposed model architecture enables local, global, and relational feature representations and offers high interpretability and adaptability. Practically, the framework allows real-time, efficient deployment to edge devices for automated weed detecting, reducing over-reliance on herbicides and providing scalable, sustainable precision-farming options.

---

## 9. Learning to Expand Images for Efficient Visual Autoregressive Modeling

**论文链接:** [http://arxiv.org/abs/2511.15499v1](http://arxiv.org/abs/2511.15499v1)

**作者:** Ruiqing Yang, Kaixin Zhang, Zheng Zhang, Shan You, Tao Huang

**发布时间:** 2025-11-19

**备注:** 16 pages, 18 figures, includes appendix with additional visualizations, submitted as arXiv preprint

### GPT解析

### 总结

本文提出了一种名为扩展自回归表示(EAR)的新型视觉生成范式，通过模拟人类视觉系统的中心向外感知模式，解决了现有自回归视觉生成模型的效率问题。

### 背景

自回归模型最近在视觉生成领域显示出巨大潜力，通过利用类似于语言建模的离散令牌序列。然而，现有方法往往效率低下，原因可能是逐令牌解码或多尺度表示的复杂性。

### 目的

开发一种更高效的视觉生成方法，通过模拟人类视觉感知模式，提高生成质量和计算效率。

### 方法

引入EAR生成范式，以螺旋顺序从中心展开图像令牌并逐渐向外扩展，保持空间连续性并实现高效并行解码。同时提出长度自适应解码策略，动态调整每步预测的令牌数量。

### 主要发现

在ImageNet上的实验表明，EAR在单尺度自回归模型中实现了保真度和效率之间的最先进权衡，为可扩展和认知对齐的自回归图像生成设定了新方向。

### 结论

这种生物启发式设计不仅降低了计算成本，还通过将生成顺序与感知相关性对齐提高了生成质量，为自回归图像生成提供了一种新方向。

### 翻译

自回归模型最近通过利用类似于语言建模的离散令牌序列在视觉生成领域显示出巨大潜力。然而，现有方法往往效率低下，要么是由于逐令牌解码，要么是由于多尺度表示的复杂性。在这项工作中，我们引入了扩展自回归表示(EAR)，一种新型生成范式，它模拟人类视觉系统的中心向外感知模式。EAR以螺旋顺序从中心展开图像令牌并逐渐向外扩展，保持空间连续性并实现高效并行解码。为进一步提高灵活性和速度，我们提出了一种长度自适应解码策略，动态调整每步预测的令牌数量。这种生物启发式设计不仅降低了计算成本，还通过将生成顺序与感知相关性对齐提高了生成质量。在ImageNet上的广泛实验表明，EAR在单尺度自回归模型中实现了保真度和效率之间的最先进权衡，为可扩展和认知对齐的自回归图像生成设定了新方向。


### 论文摘要

Autoregressive models have recently shown great promise in visual generation by leveraging discrete token sequences akin to language modeling. However, existing approaches often suffer from inefficiency, either due to token-by-token decoding or the complexity of multi-scale representations. In this work, we introduce Expanding Autoregressive Representation (EAR), a novel generation paradigm that emulates the human visual system's center-outward perception pattern. EAR unfolds image tokens in a spiral order from the center and progressively expands outward, preserving spatial continuity and enabling efficient parallel decoding. To further enhance flexibility and speed, we propose a length-adaptive decoding strategy that dynamically adjusts the number of tokens predicted at each step. This biologically inspired design not only reduces computational cost but also improves generation quality by aligning the generation order with perceptual relevance. Extensive experiments on ImageNet demonstrate that EAR achieves state-of-the-art trade-offs between fidelity and efficiency on single-scale autoregressive models, setting a new direction for scalable and cognitively aligned autoregressive image generation.

---

## 10. NTK-Guided Implicit Neural Teaching

**论文链接:** [http://arxiv.org/abs/2511.15487v1](http://arxiv.org/abs/2511.15487v1)

**作者:** Chen Zhang, Wei Zuo, Bingyang Cheng, Yikun Wang, Wei-Bin Kou, Yik Chung WU, Ngai Wong

**发布时间:** 2025-11-19

**备注:** Preprint

### GPT解析

### 总结

本文提出了一种名为NTK引导的隐式神经教学(NINT)的方法，用于加速隐式神经表示(INRs)的训练过程。该方法通过神经切线核(NTK)动态选择能最大化全局功能更新的坐标，显著减少了近一半的训练时间同时保持或提高了表示质量。

### 背景

隐式神经表示(INRs)通过多层感知器(MLPs)参数化连续信号，能够对图像、音频和3D重建等任务进行紧凑、分辨率无关的建模。然而，拟合高分辨率信号需要优化数百万个坐标，导致计算成本过高。

### 目的

开发一种方法来加速隐式神经表示的训练过程，解决高分辨率信号拟合中的计算成本问题。

### 方法

提出NTK引导的隐式神经教学(NINT)方法，通过动态选择能最大化全局功能更新的坐标来加速训练。该方法利用神经切线核(NTK)对样本进行评分，基于NTK增强的损失梯度的范数，同时考虑拟合误差和异构杠杆效应(自影响和跨坐标耦合)。

### 主要发现

NINT相比现有方法能实现更快的收敛。通过广泛实验证明，NINT能显著减少近一半的训练时间，同时保持或提高了表示质量，在最近的基于采样的策略中建立了最先进的加速效果。

### 结论

NINT是一种有效的加速INR训练的方法，在保持表示质量的同时大幅减少了训练时间，为高分辨率信号的隐式神经表示提供了更高效的解决方案。

### 翻译

隐式神经表示(INRs)通过多层感知器(MLPs)参数化连续信号，能够对图像、音频和3D重建等任务进行紧凑、分辨率无关的建模。然而，拟合高分辨率信号需要优化数百万个坐标，导致计算成本过高。为此，我们提出了NTK引导的隐式神经教学(NINT)，通过动态选择能最大化全局功能更新的坐标来加速训练。利用神经切线核(NTK)，NINT通过NTK增强的损失梯度范数对样本进行评分，同时捕捉拟合误差和异构杠杆效应(自影响和跨坐标耦合)。这种双重考虑使得NINT相比现有方法能实现更快的收敛。通过大量实验，我们证明了NINT能显著减少近一半的训练时间同时保持或提高表示质量，在最近的基于采样的策略中建立了最先进的加速效果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决隐式神经表示（INRs）训练过程中的计算效率问题。INRs通过多层感知机对连续信号进行建模，能够实现图像、音频和3D重建等任务，但拟合高分辨率信号需要优化数百万个坐标，导致计算成本极高。这个问题在现实中非常重要，因为随着高分辨率信号在计算机视觉、图形学和音频处理等领域的广泛应用，提高INR的训练效率可以加速研究进展并使实际应用更加可行，特别是在资源受限的环境中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有INR加速方法的局限性：分区方法增加架构复杂性，混合显式-隐式方法增加内存消耗，元学习方法需要大量同质数据集进行预训练，而基于采样的方法虽轻量但依赖静态启发式，忽略了训练动态。作者通过神经切线核（NTK）分析INR训练动态，发现了异构自杠杆和功能耦合的重要性。作者借鉴了现有的隐式神经表示和基于采样的加速技术，特别是非参数教学思想，将INR加速重新表述为战略选择信息量最大坐标的问题，但通过引入NTK捕捉训练动态，改进了这些方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是NINT通过利用神经切线核（NTK）动态选择最大化全局功能更新的坐标来加速INR训练。它同时考虑两个因素：(1)损失相对于网络输出的梯度，识别高拟合差异区域；(2)每个坐标通过NTK诱导的对参数更新的影响，量化点在训练过程中驱动全局模型变化的强度。通过优先选择结合高拟合误差和强动态影响的示例，确保每个训练批次最大限度地加速全局收敛。实现流程包括：初始化MLP参数；在每次迭代中进行前向传播、计算损失梯度、计算NTK行、基于NTK增强的梯度范数选择样本坐标、更新参数；重复直到收敛或达到预定迭代次数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有：(1)NTK中心的INR动态分析，揭示了仅基于误差的采样由于忽略自杠杆和跨坐标耦合而存在的缺陷；(2)NINT采样策略，一种有原则的即插即用方法，通过最大化NTK增强的梯度范数选择示例；(3)最先进的加速效果，实验表明NINT将训练时间减少近一半同时保持或提高质量。与之前工作的不同在于：NINT不需要增加架构复杂性或内存消耗，不需要大量预训练数据，且明确考虑了NTK中编码的异构自杠杆和功能耦合，而不仅仅是静态的输出误差测量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NINT通过利用神经切线核动态选择最有影响力的训练坐标，显著提高了隐式神经表示的训练效率，在保持或提高表示质量的同时将训练时间减少近一半。'}


### 论文摘要

Implicit Neural Representations (INRs) parameterize continuous signals via multilayer perceptrons (MLPs), enabling compact, resolution-independent modeling for tasks like image, audio, and 3D reconstruction. However, fitting high-resolution signals demands optimizing over millions of coordinates, incurring prohibitive computational costs. To address it, we propose NTK-Guided Implicit Neural Teaching (NINT), which accelerates training by dynamically selecting coordinates that maximize global functional updates. Leveraging the Neural Tangent Kernel (NTK), NINT scores examples by the norm of their NTK-augmented loss gradients, capturing both fitting errors and heterogeneous leverage (self-influence and cross-coordinate coupling). This dual consideration enables faster convergence compared to existing methods. Through extensive experiments, we demonstrate that NINT significantly reduces training time by nearly half while maintaining or improving representation quality, establishing state-of-the-art acceleration among recent sampling-based strategies.

---

## 11. RS-CA-HSICT: A Residual and Spatial Channel Augmented CNN Transformer Framework for Monkeypox Detection

**论文链接:** [http://arxiv.org/abs/2511.15476v1](http://arxiv.org/abs/2511.15476v1)

**作者:** Rashid Iqbal, Saddam Hussain Khan

**发布时间:** 2025-11-19

**备注:** 33 Pages, 12 Figure, 4 Tables

### GPT解析

### 总结

本文提出了一种名为RS-CA-HSICT的混合深度学习方法，结合CNN和Transformer的优势，用于猴痘检测。该方法通过HSICT块、残差CNN模块、空间CNN块和通道增强组成，能够增强特征空间的多样性、病变细节信息和长程依赖关系。

### 背景

猴痘(Mpox)是一种需要准确检测的疾病。现有的CNN和ViTs方法在猴痘检测中可能存在局限性，需要更有效的检测方法。

### 目的

开发一种结合CNN和Transformer优势的混合深度学习方法，提高猴痘检测的准确性和效率。

### 方法

提出RS-CA-HSICT框架，包含HSICT块、残差CNN模块、空间CNN块和通道增强。HSICT模块整合stem CNN的抽象表示和自定义ICT块，用于多注意力头和结构化CNN层。通过逆残差学习增强梯度消失问题，阶段性分辨率减少确保尺度不变性。通道融合和注意力块精炼通道，空间注意力机制细化像素选择。

### 主要发现

在Kaggle基准和多样化猴痘数据集上，所提方法达到98.30%的分类准确率和98.13%的F1分数，性能优于现有的CNN和ViTs方法。

### 结论

RS-CA-HSICT框架通过结合CNN和Transformer的优势，有效提高了猴痘检测的准确性，能够捕获全局和局部结构线索、细微纹理和对比度变化。

### 翻译

本文提出了一种混合深度学习方法，即基于残差和空间学习的通道增强集成CNN-Transformer架构(RS-CA-HSICT)，该方法利用CNN和Transformer的优势，用于增强猴痘检测。所提出的RS-CA-HSICT框架由HSICT块、残差CNN模块、空间CNN块和CA组成，增强了多样化的特征空间、详细的病变信息和长程依赖关系。新的HSICT模块首先整合了stem CNN的抽象表示和自定义ICT块，用于高效的多头注意力和具有同质(H)和结构(S)操作的结构化CNN层。自定义ICT块学习全局上下文交互和局部纹理提取。此外，H和S层通过减少噪声和建模复杂形态变化来学习空间同质性和精细结构细节。此外，逆残差学习增强了梯度消失，阶段性分辨率减少确保了尺度不变性。此外，RS-CA-HSICT框架通过基于迁移学习的残差和空间CNN映射增强学习的HSICT通道，用于增强多尺度特征空间捕获全局和局部结构线索、细微纹理和对比度变化。这些通道在增强前通过通道融合和注意力块进行精炼，该块保留判别性通道同时抑制冗余通道，从而实现高效计算。最后，空间注意力机制细化像素选择，以检测猴痘中的细微模式和类内对比度变化。在Kaggle基准和多样化猴痘数据集上的实验结果报告分类准确率高达98.30%，F1得分为98.13%，优于现有的CNN和ViTs。


### 论文摘要

This work proposes a hybrid deep learning approach, namely Residual and Spatial Learning based Channel Augmented Integrated CNN-Transformer architecture, that leverages the strengths of CNN and Transformer towards enhanced MPox detection. The proposed RS-CA-HSICT framework is composed of an HSICT block, a residual CNN module, a spatial CNN block, and a CA, which enhances the diverse feature space, detailed lesion information, and long-range dependencies. The new HSICT module first integrates an abstract representation of the stem CNN and customized ICT blocks for efficient multihead attention and structured CNN layers with homogeneous (H) and structural (S) operations. The customized ICT blocks learn global contextual interactions and local texture extraction. Additionally, H and S layers learn spatial homogeneity and fine structural details by reducing noise and modeling complex morphological variations. Moreover, inverse residual learning enhances vanishing gradient, and stage-wise resolution reduction ensures scale invariance. Furthermore, the RS-CA-HSICT framework augments the learned HSICT channels with the TL-driven Residual and Spatial CNN maps for enhanced multiscale feature space capturing global and localized structural cues, subtle texture, and contrast variations. These channels, preceding augmentation, are refined through the Channel-Fusion-and-Attention block, which preserves discriminative channels while suppressing redundant ones, thereby enabling efficient computation. Finally, the spatial attention mechanism refines pixel selection to detect subtle patterns and intra-class contrast variations in Mpox. Experimental results on both the Kaggle benchmark and a diverse MPox dataset reported classification accuracy as high as 98.30% and an F1-score of 98.13%, which outperforms the existing CNNs and ViTs.

---

## 12. LCS: A Learnlet-Based Sparse Framework for Blind Source Separation

**论文链接:** [http://arxiv.org/abs/2511.15475v1](http://arxiv.org/abs/2511.15475v1)

**作者:** V. Bonjean, A. Gkogkou, J. L. Starck, P. Tsakalides

**发布时间:** 2025-11-19

**备注:** 11 pages, 8 figures

### GPT解析

### 总结

本文介绍了一种名为Learnlet组件分离器(LCS)的新型盲源分离框架，结合了经典稀疏技术与现代深度学习，用于从多频率观测中提取天体物理信号。

### 背景

盲源分离(BSS)在现代天体物理学中起着关键作用，能够从多频率观测中提取有科学意义的信号。传统BSS方法（如基于固定小波字典的方法）在组件分离时强制稀疏性，但在处理真实天体物理信号的固有复杂性时表现不足。

### 目的

开发一种新的BSS框架，结合经典稀疏技术与现代深度学习，解决传统方法在处理复杂天体物理信号时的局限性。

### 方法

LCS利用Learnlet变换（一种结构化卷积神经网络，设计为学习的小波类多尺度表示），将学习稀疏表示集成到迭代源分离过程中，实现多通道观测的有效分解。这种混合设计保留了小波的可解释性和稀疏性，同时获得了学习模型的适应性和表现力。

### 主要发现

在合成和真实数据集上评估显示，LCS与最先进的方法相比表现出优越的分离性能，在玩具模型示例中平均增益约5分贝。

### 结论

结合信号处理先验与深度学习的混合方法，有潜力解决下一代宇宙学实验的挑战。

### 翻译

盲源分离(BSS)通过从多频率观测中提取具有科学意义的信号，在现代天体物理学中发挥着关键作用。传统的BSS方法，如那些依赖固定小波字典的方法，在组件分离时强制稀疏性，但在面对真实天体物理信号的固有复杂性时可能表现不足。在这项工作中，我们引入了Learnlet组件分离器(LCS)，一种新型BSS框架，它连接了经典稀疏技术与现代深度学习。LCS利用Learnlet变换：一种设计为作为学习的小波类多尺度表示的结构化卷积神经网络。这种混合设计保留了小波的可解释性和稀疏性，同时获得了学习模型的适应性和表现力。LCS算法将这种学习稀疏表示集成到迭代源分离过程中，实现了多通道观测的有效分解。虽然在概念上受稀疏BSS方法启发，但LCS引入了学习表示层，显著偏离了经典固定基假设。我们在合成和真实数据集上评估了LCS，与最先进的方法相比表现出优越的分离性能（在玩具模型示例中平均增益约5分贝）。我们的结果突显了结合信号处理先验与深度学习的混合方法在解决下一代宇宙学实验挑战方面的潜力。


### 论文摘要

Blind source separation (BSS) plays a pivotal role in modern astrophysics by enabling the extraction of scientifically meaningful signals from multi-frequency observations. Traditional BSS methods, such as those relying on fixed wavelet dictionaries, enforce sparsity during component separation, but may fall short when faced with the inherent complexity of real astrophysical signals. In this work, we introduce the Learnlet Component Separator (LCS), a novel BSS framework that bridges classical sparsity-based techniques with modern deep learning. LCS utilizes the Learnlet transform: a structured convolutional neural network designed to serve as a learned, wavelet-like multiscale representation. This hybrid design preserves the interpretability and sparsity, promoting properties of wavelets while gaining the adaptability and expressiveness of learned models. The LCS algorithm integrates this learned sparse representation into an iterative source separation process, enabling effective decomposition of multi-channel observations. While conceptually inspired by sparse BSS methods, LCS introduces a learned representation layer that significantly departs from classical fixed-basis assumptions. We evaluate LCS on both synthetic and real datasets, demonstrating superior separation performance compared to state-of-the-art methods (average gain of about 5 dB on toy model examples). Our results highlight the potential of hybrid approaches that combine signal processing priors with deep learning to address the challenges of next-generation cosmological experiments.

---

## 13. SIGMMA: Hierarchical Graph-Based Multi-Scale Multi-modal Contrastive Alignment of Histopathology Image and Spatial Transcriptome

**论文链接:** [http://arxiv.org/abs/2511.15464v1](http://arxiv.org/abs/2511.15464v1)

**作者:** Dabin Jeong, Amirhossein Vahidi, Ciro Ramírez-Suástegui, Marie Moullet, Kevin Ly, Mohammad Vali Sanian, Sebastian Birk, Yinshui Chang, Adam Boxall, Daniyal Jafree, Lloyd Steele, Vijaya Baskar MS, Muzlifah Haniffa, Mohammad Lotfollahi

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种名为Sigmma的多模态对比对齐框架，用于学习组织学图像与空间转录组特征在多个尺度上的分层表示，有效捕捉了细胞间的相互作用，并在基因表达预测和跨模态检索任务中取得显著性能提升。

### 背景

现有计算病理学方法通常在单一尺度上将H&E图像块与其对应的ST特征进行对齐，忽略了细粒度的细胞结构及其空间组织。

### 目的

开发一个能够学习H&E图像和空间转录组特征在多个尺度上分层表示的框架，捕捉从细粒度到粗粒度的细胞-细胞相互作用。

### 方法

提出Sigmma框架，引入多尺度对比对齐确保不同尺度表示在模态间保持一致性；通过将细胞相互作用表示为图并整合子图内和子图间关系来捕捉组织微环境中的细胞相互作用。

### 主要发现

Sigmma学习到的表示能更好地捕捉跨模态对应关系，在基因表达预测任务中平均提高9.78%，在跨模态检索任务中平均提高26.93%；在下游分析中能学习到有意义的多组织组织结构。

### 结论

Sigmma框架通过多尺度对比对齐和细胞相互作用建模，有效解决了现有方法在单一尺度上对齐的问题，显著提升了跨模态任务的性能。

### 翻译

最近的计算病理学进展利用视觉-语言模型来学习苏木精-伊红(HE)图像与空间转录组(ST)特征的联合表示。然而，现有方法通常在单一尺度上将HE图像块与其对应的ST特征进行对齐，忽略了细粒度的细胞结构及其空间组织。为此，我们提出了Sigmma，一个多模态对比对齐框架，用于学习HE图像和空间转录组特征在多个尺度上的分层表示。Sigmma引入了多尺度对比对齐，确保在不同尺度上学到的表示在模态之间保持一致性。此外，通过将细胞相互作用表示为图并整合子图内和子图间关系，我们的方法有效地捕捉了组织微环境中从细粒度到粗粒度的细胞-细胞相互作用。我们证明，Sigmma学习到的表示能更好地捕捉跨模态对应关系，在基因表达预测任务中平均提高9.78%，在跨模态检索任务中平均提高26.93%。我们进一步证明，在下游分析中，它学习到了有意义的多组织组织结构。


### 论文摘要

Recent advances in computational pathology have leveraged vision-language models to learn joint representations of Hematoxylin and Eosin (HE) images with spatial transcriptomic (ST) profiles. However, existing approaches typically align HE tiles with their corresponding ST profiles at a single scale, overlooking fine-grained cellular structures and their spatial organization. To address this, we propose Sigmma, a multi-modal contrastive alignment framework for learning hierarchical representations of HE images and spatial transcriptome profiles across multiple scales. Sigmma introduces multi-scale contrastive alignment, ensuring that representations learned at different scales remain coherent across modalities. Furthermore, by representing cell interactions as a graph and integrating inter- and intra-subgraph relationships, our approach effectively captures cell-cell interactions, ranging from fine to coarse, within the tissue microenvironment. We demonstrate that Sigmm learns representations that better capture cross-modal correspondences, leading to an improvement of avg. 9.78\% in the gene-expression prediction task and avg. 26.93\% in the cross-modal retrieval task across datasets. We further show that it learns meaningful multi-tissue organization in downstream analyses.

---

## 14. Representation Space Constrained Learning with Modality Decoupling for Multimodal Object Detection

**论文链接:** [http://arxiv.org/abs/2511.15433v1](http://arxiv.org/abs/2511.15433v1)

**作者:** YiKang Shao, Tao Shi

**发布时间:** 2025-11-19

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本文针对多模态目标检测中的融合退化问题进行了系统理论研究，提出了一种表示空间约束学习与模态解耦方法，有效缓解了融合退化并取得了最先进的性能。

### 背景

多模态目标检测因其增强的鲁棒性而在学术界和工业界受到广泛关注。然而，尽管许多研究专注于改进模态融合策略，但大多数忽略了融合退化问题，且没有对其根本原因提供理论分析。

### 目的

填补这一空白，对多模态检测中的融合退化进行系统的理论研究，并确定其根本原因，然后提出有效解决方案。

### 方法

提出了一种表示空间约束学习与模态解耦方法，包含两个模块：RSC模块用于放大被抑制的梯度，MD模块用于消除模态间耦合干扰以及模态不平衡。

### 主要发现

确定了两个关键的优化缺陷：(1)在多模态架构下，单模态分支骨干网络的梯度受到严重抑制，导致单模态分支优化不足；(2)模态质量的差异导致较弱的模态经历更强的梯度抑制，进而导致模态学习不平衡。

### 结论

所提出的方法有效缓解了融合退化，并在FLIR、LLVIP、M3FD和MFAD数据集上的多个基准测试中取得了最先进的性能。代码和训练过程将在https://github.com/yikangshao/RSC-MD上发布。

### 翻译

多模态目标检测因其增强的鲁棒性而在学术界和工业界引起了广泛关注。尽管许多研究专注于改进模态融合策略，但大多数忽略了融合退化问题，且没有对其根本原因提供理论分析。为填补这一空白，本文对多模态检测中的融合退化进行了系统的理论研究，并确定了两个关键的优化缺陷：(1)在多模态架构下，单模态分支骨干网络的梯度受到严重抑制，导致单模态分支优化不足；(2)模态质量的差异导致较弱的模态经历更强的梯度抑制，进而导致模态学习不平衡。为解决这些问题，本文提出了一种表示空间约束学习与模态解耦方法，包含两个模块。RSC模块和MD模块分别被设计为放大被抑制的梯度和消除模态间耦合干扰以及模态不平衡，从而实现对每个模态特定骨干网络的全面优化。在FLIR、LLVIP、M3FD和MFAD数据集上的大量实验表明，所提出的方法有效缓解了融合退化，并在多个基准测试中取得了最先进的性能。代码和训练过程将在https://github.com/yikangshao/RSC-MD上发布。


### 论文摘要

Multimodal object detection has attracted significant attention in both academia and industry for its enhanced robustness. Although numerous studies have focused on improving modality fusion strategies, most neglect fusion degradation, and none provide a theoretical analysis of its underlying causes. To fill this gap, this paper presents a systematic theoretical investigation of fusion degradation in multimodal detection and identifies two key optimization deficiencies: (1) the gradients of unimodal branch backbones are severely suppressed under multimodal architectures, resulting in under-optimization of the unimodal branches; (2) disparities in modality quality cause weaker modalities to experience stronger gradient suppression, which in turn results in imbalanced modality learning. To address these issues, this paper proposes a Representation Space Constrained Learning with Modality Decoupling (RSC-MD) method, which consists of two modules. The RSC module and the MD module are designed to respectively amplify the suppressed gradients and eliminate inter-modality coupling interference as well as modality imbalance, thereby enabling the comprehensive optimization of each modality-specific backbone. Extensive experiments conducted on the FLIR, LLVIP, M3FD, and MFAD datasets demonstrate that the proposed method effectively alleviates fusion degradation and achieves state-of-the-art performance across multiple benchmarks. The code and training procedures will be released at https://github.com/yikangshao/RSC-MD.

---

## 15. Towards Understanding Layer Contributions in Tabular In-Context Learning Models

**论文链接:** [http://arxiv.org/abs/2511.15432v1](http://arxiv.org/abs/2511.15432v1)

**作者:** Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger

**发布时间:** 2025-11-19

**备注:** Accepted at the EurIPS 2025 Workshop on AI for Tabular Data

### GPT解析

### 总结

该研究探讨了表格上下文学习模型各层对预测的贡献，发现了模型中的冗余层，并提出了模型压缩和可解释性改进的机会。

### 背景

表格上下文学习模型与大型语言模型在架构上具有相似性，但关于单个层如何贡献于表格预测的研究还很少。

### 目的

研究表格ICL模型中潜在空间如何随层变化，识别可能的冗余层，并与LLMs中的动态进行比较。

### 方法

通过'层作为画家'的视角分析TabPFN和TabICL模型，研究各层的潜在空间表征。

### 主要发现

只有部分层共享共同的表征语言，表明模型中存在结构冗余。

### 结论

表格ICL模型中的结构冗余为模型压缩和改进可解释性提供了机会。

### 翻译

尽管表格上下文学习模型与大型语言模型在架构上具有相似性，但关于单个层如何贡献于表格预测的信息还很少。在本文中，我们研究了表格ICL模型中潜在空间如何随层变化，识别了可能的冗余层，并将这些动态与在LLMs中观察到的动态进行了比较。我们通过'层作为画家'的视角分析了TabPFN和TabICL，发现只有部分层共享共同的表征语言，这表明存在结构冗余，并为模型压缩和改进可解释性提供了机会。


### 论文摘要

Despite the architectural similarities between tabular in-context learning (ICL) models and large language models (LLMs), little is known about how individual layers contribute to tabular prediction. In this paper, we investigate how the latent spaces evolve across layers in tabular ICL models, identify potential redundant layers, and compare these dynamics with those observed in LLMs. We analyze TabPFN and TabICL through the "layers as painters" perspective, finding that only subsets of layers share a common representational language, suggesting structural redundancy and offering opportunities for model compression and improved interpretability.

---

## 16. One algebra for all : Geometric Algebra methods for neurosymbolic XR scene authoring, animation and neural rendering

**论文链接:** [http://arxiv.org/abs/2511.15398v1](http://arxiv.org/abs/2511.15398v1)

**作者:** Manos Kamarianakis, Antonis Protopsaltis, George Papagiannakis

**发布时间:** 2025-11-19

**备注:** 10 pages, 9 Figures

### GPT解析

### 总结

这篇论文探讨了几何代数(GA)在计算机图形学(CG)和扩展现实(XR)领域的变革性作用，特别是在角色动画、渲染、绑定、神经渲染和生成式AI驱动的场景编辑方面。研究表明，GA通过统一的代数表达式封装几何形式和变换，能够显著提升传统CG算法的性能和精度。

### 背景

传统的CG算法在处理旋转、平移和均匀缩放等操作时，通常使用矩阵、四元数和向量等表示形式，但这些方法在精度和性能方面存在局限性。在对象渲染、绑定模型动画、软体变形和XR模拟等操作中，这些传统表示形式无法保持关键几何属性在多步变换中的完整性。

### 目的

探讨几何代数如何提高角色绑定动画的保真度，增强软体模拟效果，简化实时渲染流程，并优化神经和生成式AI场景编辑，特别是在XR环境中实现更优的视觉结果和计算效率。

### 方法

采用几何代数作为数学框架，将其应用于计算机图形学和扩展现实的多个领域。通过将传统的矩阵、四元数和向量表示替换为GA的统一表达式，展示GA如何保持几何属性在多步变换中的完整性，并作为神经符号XR场景创作的统一数学基础。

### 主要发现

几何代数能够显著提升CG和XR中的关键算法性能；通过统一代数表达式封装几何形式和变换，保持关键几何属性；为神经符号XR场景创作提供统一的数学基础；改善绑定角色动画的保真度；增强软体模拟效果；简化实时渲染流程；优化神经和生成式AI场景编辑。

### 结论

几何代数为计算机图形学和扩展现实中的多种过程提供了一致且高效的框架，特别是在XR环境中能够产生卓越的视觉效果和计算效率。通过统一处理几何变换和表示，解决了传统方法在精度和性能方面的局限性，为CG和XR领域的发展提供了新的数学基础。

### 翻译

这篇立场论文深入探讨了几何代数(GA)在推动计算机图形学(CG)和扩展现实(XR)特定领域发展中的变革性作用，特别是在角色动画、渲染、绑定、神经渲染以及生成式AI驱动的场景编辑方面。常见的CG算法在对象渲染、绑定模型动画、软体变形和XR模拟等操作中需要处理旋转、平移和均匀缩放等变换。传统的表示形式，如矩阵、四元数和向量，常常在精度和性能方面引入限制。GA使用的最新突破表明，它可以通过将几何形式和变换封装为统一的代数表达式来显著提升这些过程，这些表达式在多步变换中保持关键的几何属性。此外，我们探讨了GA如何作为神经符号XR场景创作的统一数学基础，弥合学习到的神经表示和明确的几何推理之间的差距。本文概述了基于GA的方法如何提高绑定角色动画的保真度，增强软体模拟效果，简化实时渲染流程，并优化神经和生成式AI场景编辑。GA为这些过程提供了一致且高效的框架，在XR环境中产生卓越的视觉效果和计算效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决计算机图形学领域中数学表示的碎片化问题。传统上，图形学使用向量、矩阵和四元数等不同工具处理位置、变换和旋转，导致性能开销、数值不稳定性以及概念不连贯。这个问题在现实中非常重要，因为随着XR和实时渲染应用普及，对性能和精度要求越来越高；生成式AI与图形学结合需要统一框架；协作XR应用需要高效数据传输；医疗模拟等高保真应用需要精确几何操作。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别传统图形学中数学表示的碎片化问题，然后研究几何代数作为统一框架的潜力，将其应用到多个图形学领域包括几何处理、角色动画、网络化XR环境和神经图形。作者确实借鉴了现有工作，包括Clifford Algebra和Geometric Algebra的基础理论，以及计算机图形学中现有的动画、渲染和模拟技术，并在前人研究基础上进行了扩展和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是几何代数提供统一数学框架，其中几何实体和变换都可表示为多向量，几何关系在代数操作中得到保持。实现流程包括：1)使用3D欧几里得几何代数和共形几何代数表示几何实体；2)将旋转、平移和缩放表示为特殊多向量并通过'三明治积'应用变换；3)在网格操作、角色动画、网络传输和神经渲染中应用GA；4)开发GA-Unity包集成到Unity引擎并提供与传统表示的转换接口。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提供统一数学框架替代碎片化工具；2)开发基于GA的实时网格变形、切割和撕裂算法；3)使用GA实现改进的角色动画蒙皮和插值；4)优化网络化XR减少带宽需求；5)结合GA与AI开发自然语言指令处理和神经渲染系统。相比之前工作，GA方法具有更好的统一性、效率、精度和功能，支持传统方法难以实现的动态拓扑变化等高级操作，并通过GA-Unity实现了实际应用集成。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文展示了几何代数如何作为统一数学框架革新整个计算机图形学流程，从场景创作到动画制作再到神经渲染，提高了效率、精度和功能，并为XR和AI驱动的图形应用开辟了新可能性。'}


### 论文摘要

This position paper delves into the transformative role of Geometric Algebra (GA) in advancing specific areas of Computer Graphics (CG) and Extended Reality (XR), particularly in character animation, rendering, rigging, neural rendering, and generative AI-driven scene editing. Common CG algorithms require handling rotations, translations, and dilations (uniform scalings) in operations such as object rendering, rigged model animation, soft-body deformation, and XR simulations. Traditional representation forms - such as matrices, quaternions, and vectors - often introduce limitations in precision and performance. Recent breakthroughs in the use of GA suggest it can significantly enhance these processes by encapsulating geometric forms and transformations into uniform algebraic expressions, which maintain critical geometric properties throughout multi-step transformations. Furthermore, we explore how GA can serve as a unifying mathematical substrate for neurosymbolic XR scene authoring, bridging learned neural representations and explicit geometric reasoning. This paper outlines how GA-based approaches can improve the fidelity of rigged character animations, enhance soft-body simulations, streamline real-time rendering, and optimize neural and generative AI scene editing. GA offers a coherent and efficient framework for these processes, resulting in superior visual outcomes and computational efficiency, particularly in XR environments.

---

## 17. ShelfOcc: Native 3D Supervision beyond LiDAR for Vision-Based Occupancy Estimation

**论文链接:** [http://arxiv.org/abs/2511.15396v1](http://arxiv.org/abs/2511.15396v1)

**作者:** Simon Boeder, Fabian Gigengack, Simon Roesler, Holger Caesar, Benjamin Risse

**发布时间:** 2025-11-19

### GPT解析

### 总结

ShelfOcc是一种不依赖LiDAR的纯视觉方法，通过从视频中生成度量一致的语义体素标签，将监督引入原生3D空间，克服了现有2D投影或渲染监督的几何不一致性和深度渗出问题。

### 背景

当前自监督和弱监督占用估计主要依赖2D投影或渲染监督，存在几何不一致性和严重深度渗出问题；现有基于视觉的3D几何基础模型在动态驾驶场景中存在稀疏、噪声和不一致的几何问题。

### 目的

引入一种不依赖LiDAR的纯视觉方法，实现真正的3D监督，无需额外传感器或手动3D标注。

### 方法

ShelfOcc通过专用框架跨帧一致地过滤和累积静态几何，处理动态内容并将语义信息传播到稳定的体素表示中，实现数据中心监督，可用于任何SOTA占用模型架构。

### 主要发现

高质量监督对鲁棒占用学习至关重要，是架构创新的重要补充；在Occ3D-nuScenes基准测试上，ShelfOcc相对提升高达34%，显著优于所有之前的弱监督/自监督方法。

### 结论

ShelfOcc为无LiDAR的3D场景理解建立了新的数据驱动方向。

### 翻译

最近在自监督和弱监督占用估计方面的进展主要依赖于2D投影或基于渲染的监督，这些方法存在几何不一致性和严重的深度渗出问题。因此，我们引入了ShelfOcc，这是一种不依赖LiDAR的纯视觉方法，克服了这些局限性。ShelfOcc通过从视频中生成度量一致的语义体素标签，将监督引入原生3D空间，实现了真正的3D监督，无需任何额外传感器或手动3D标注。虽然最近的基于视觉的3D几何基础模型提供了有前途的先验知识来源，但由于几何稀疏、噪声和不一致，它们不能直接用作预测，特别是在动态驾驶场景中。我们的方法引入了一个专用框架，通过跨帧一致地过滤和累积静态几何来缓解这些问题，处理动态内容并将语义信息传播到稳定的体素表示中。这种在弱监督/自监督占用估计中的数据中心监督转变，使得可以使用任何最先进的占用模型架构，而不依赖LiDAR数据。我们认为这种高质量监督对于鲁棒的占用学习至关重要，并构成了架构创新的重要补充途径。在Occ3D-nuScenes基准测试上，ShelfOcc显著优于所有之前的弱监督/自监督方法（相对提升高达34%），为无LiDAR的3D场景理解建立了新的数据驱动方向。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于视觉的3D占用率估计中依赖2D投影或渲染监督导致的几何不一致性和深度渗出问题。这个问题在现实中非常重要，因为准确的3D占用率估计对自动驾驶的安全性和可靠性至关重要，而目前方法严重依赖昂贵的LiDAR传感器获取3D标注，限制了其在实际应用中的扩展性和可访问性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有2D渲染监督方法的局限性，认识到直接应用3D几何基础模型到动态场景的挑战。他们设计了一个专门的框架，通过过滤和跨帧累积静态几何，处理动态内容并将语义信息传播到稳定的体素表示中。该方法借鉴了MapAnything作为3D几何基础模型和GroundedSAM进行2D语义分割，但创新性地结合它们并设计了处理动态场景的专门流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将监督引入原生3D空间，通过从视频生成度量一致的语义体素标签，实现真正的3D监督，无需额外传感器或手动标注。整体流程包括：1)使用GroundedSAM生成2D语义分割掩码；2)使用MapAnything估计3D几何并分离静态/动态场景；3)累积静态点云并应用置信度过滤；4)每帧重新引入动态对象；5)将点云转换为3D体素网格并生成可见性掩码；6)使用生成的伪标签训练占用率网络。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)监督策略的范式转变，实现原生3D架监督；2)仅基于视觉的3D伪标签生成管道；3)数据中心性能提升。相比之前的工作，ShelfOcc直接在3D体素空间训练而非依赖2D渲染，专门处理动态场景挑战，提供即插即用的解决方案，无需LiDAR或复杂渲染机制，显著提升了性能（相对提升高达34%）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ShelfOcc提出了一种仅基于视觉的原生3D监督框架，通过结合几何和语义基础模型生成高质量的3D伪标签，显著提升了占用率估计的性能，无需LiDAR或手动3D标注。'}


### 论文摘要

Recent progress in self- and weakly supervised occupancy estimation has largely relied on 2D projection or rendering-based supervision, which suffers from geometric inconsistencies and severe depth bleeding. We thus introduce ShelfOcc, a vision-only method that overcomes these limitations without relying on LiDAR. ShelfOcc brings supervision into native 3D space by generating metrically consistent semantic voxel labels from video, enabling true 3D supervision without any additional sensors or manual 3D annotations. While recent vision-based 3D geometry foundation models provide a promising source of prior knowledge, they do not work out of the box as a prediction due to sparse or noisy and inconsistent geometry, especially in dynamic driving scenes. Our method introduces a dedicated framework that mitigates these issues by filtering and accumulating static geometry consistently across frames, handling dynamic content and propagating semantic information into a stable voxel representation. This data-centric shift in supervision for weakly/shelf-supervised occupancy estimation allows the use of essentially any SOTA occupancy model architecture without relying on LiDAR data. We argue that such high-quality supervision is essential for robust occupancy learning and constitutes an important complementary avenue to architectural innovation. On the Occ3D-nuScenes benchmark, ShelfOcc substantially outperforms all previous weakly/shelf-supervised methods (up to a 34% relative improvement), establishing a new data-driven direction for LiDAR-free 3D scene understanding.

---

## 18. EVA-Net: Interpretable Brain Age Prediction via Continuous Aging Prototypes from EEG

**论文链接:** [http://arxiv.org/abs/2511.15393v1](http://arxiv.org/abs/2511.15393v1)

**作者:** Kunyu Zhang, Mingxuan Wang, Xiangjie Shi, Haoxing Xu, Chao Zhang

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出EVA-Net框架，将脑龄重新定义为可解释的异常检测问题，使用稀疏注意力Transformer和变分信息瓶颈处理不完美的脑电图数据，实现脑健康评估和疾病检测。

### 背景

脑龄是评估脑健康的关键指标，脑电图(EEG)是实用工具。现有模型难以处理不完美的医疗数据，如从弱监督的健康队列中学习'正常'基线。标准模型通常是黑盒，缺乏可解释结构。

### 目的

开发一种能够处理不完美医疗数据并提供可解释结果的脑龄评估框架，用于异常检测和疾病识别。

### 方法

EVA-Net使用稀疏注意力Transformer建模长EEG序列，采用变分信息瓶颈处理噪声和变异性，将表示与连续原型网络对齐以学习正常健康衰老流形。在1297名健康受试者上训练。

### 主要发现

EVA-Net在健康受试者上达到最先进精度。在27名MCI和AD患者验证中，病理组显示显著更高的脑龄差距和新型原型对齐误差，证实偏离健康流形。

### 结论

EVA-Net为使用不完美医疗数据的医疗智能提供了可解释框架，能有效识别脑健康异常。

### 翻译

脑龄是脑健康的关键指标。虽然脑电图(EEG)是这项任务的实用工具，但现有模型难以处理不完美的医疗数据这一常见挑战，例如从弱监督的健康队列中学习'正常'基线。这是识别疾病的关键异常检测任务，但标准模型通常是黑盒，缺乏可解释的结构。我们提出了EVA-Net，一种将脑龄重新定义为可解释异常检测问题的新型框架。EVA-Net使用高效的稀疏注意力Transformer来建模长EEG序列。为了处理不完美数据中的噪声和变异性，它采用变分信息瓶颈学习鲁棒、压缩的表示。为了可解释性，该表示与连续原型网络对齐，明确学习正常健康衰老流形。在1297名健康受试者上训练的EVA-Net实现了最先进精度。我们在27名MCI和AD患者的未见队列上验证了其异常检测能力。该病理组显示出显著更高的脑龄差距和一种新型原型对齐误差，证实了他们偏离健康流形的情况。EVA-Net为使用不完美医疗数据的医疗智能提供了可解释的框架。


### 论文摘要

The brain age is a key indicator of brain health. While electroencephalography (EEG) is a practical tool for this task, existing models struggle with the common challenge of imperfect medical data, such as learning a ``normal'' baseline from weakly supervised, healthy-only cohorts. This is a critical anomaly detection task for identifying disease, but standard models are often black boxes lacking an interpretable structure. We propose EVA-Net, a novel framework that recasts brain age as an interpretable anomaly detection problem. EVA-Net uses an efficient, sparsified-attention Transformer to model long EEG sequences. To handle noise and variability in imperfect data, it employs a Variational Information Bottleneck to learn a robust, compressed representation. For interpretability, this representation is aligned to a continuous prototype network that explicitly learns the normative healthy aging manifold. Trained on 1297 healthy subjects, EVA-Net achieves state-of-the-art accuracy. We validated its anomaly detection capabilities on an unseen cohort of 27 MCI and AD patients. This pathological group showed significantly higher brain-age gaps and a novel Prototype Alignment Error, confirming their deviation from the healthy manifold. EVA-Net provides an interpretable framework for healthcare intelligence using imperfect medical data.

---

## 19. Terra Nova: A Comprehensive Challenge Environment for Intelligent Agents

**论文链接:** [http://arxiv.org/abs/2511.15378v1](http://arxiv.org/abs/2511.15378v1)

**作者:** Trevor McInroe

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文介绍了Terra Nova，一个受文明V启发的新综合挑战环境(CCE)，用于强化学习研究。CCE是一个单一环境，其中同时存在多个典型的RL挑战，要求智能体具备集成、长远的跨变量理解能力。

### 背景

现有的多任务基准测试主要关注智能体能否编目和切换不相关的策略，而非测试在多种交互挑战中进行深度推理的能力。

### 目的

创建一个能同时呈现多种RL挑战的综合环境，以评估智能体在复杂、交互式场景中的深度推理和长期规划能力。

### 方法

设计并实现了一个名为Terra Nova的单一综合挑战环境，该环境基于文明V游戏，同时包含部分可观察性、信用分配、表示学习、巨大动作空间等多个典型RL挑战。

### 主要发现

掌握Terra Nova环境需要智能体具备集成、长远的跨变量理解能力，这区别于简单地聚合不相关任务的多任务学习环境。

### 结论

Terra Nova作为一个综合挑战环境，能够更有效地测试智能体在多种交互挑战中的深度推理能力，为强化学习研究提供了新的评估标准。

### 翻译

我们介绍了Terra Nova，一个受文明V启发的新综合挑战环境(CCE)，用于强化学习研究。CCE是一个单一环境，其中同时出现多个典型的RL挑战（如部分可观察性、信用分配、表示学习、巨大的动作空间等）。掌握因此需要跨多个交互变量进行集成、长远的理解。我们强调，这一定义不包括那些只是在不相关、并行流中聚合不相关任务的挑战（例如，同时学习玩所有Atari游戏）。这些聚合的多任务基准测试主要评估智能体能否编目和切换不相关的策略，而不是测试智能体在许多交互挑战中进行深度推理的能力。


### 论文摘要

We introduce Terra Nova, a new comprehensive challenge environment (CCE) for reinforcement learning (RL) research inspired by Civilization V. A CCE is a single environment in which multiple canonical RL challenges (e.g., partial observability, credit assignment, representation learning, enormous action spaces, etc.) arise simultaneously. Mastery therefore demands integrated, long-horizon understanding across many interacting variables. We emphasize that this definition excludes challenges that only aggregate unrelated tasks in independent, parallel streams (e.g., learning to play all Atari games at once). These aggregated multitask benchmarks primarily asses whether an agent can catalog and switch among unrelated policies rather than test an agent's ability to perform deep reasoning across many interacting challenges.

---

## 20. Fidelity-Preserving Quantum Encoding for Quantum Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.15363v1](http://arxiv.org/abs/2511.15363v1)

**作者:** Yuhu Lu, Jinjing Shi

**发布时间:** 2025-11-19

**备注:** Under review

### GPT解析

### 总结

本文提出了一种保真度保留量子编码(FPQE)框架，能够高效地将经典视觉数据编码到量子态中，同时保留空间和语义信息，在复杂数据集上表现出色。

### 背景

将经典视觉数据高效编码到量子态对于实现实用量子神经网络至关重要，但现有编码方案在将高维图像适应到NISQ设备的有限量子位时，往往会丢弃重要的空间和语义信息。

### 目的

开发一种能够进行近乎无损数据压缩和量子编码的框架，解决现有方法在信息保留方面的局限性。

### 方法

FPQE框架使用卷积编码器-解码器学习紧凑的多通道表示，这些表示能够以高保真度重建原始数据，然后通过振幅编码将这些表示映射到量子态。

### 主要发现

FPQE在简单数据集(如MNIST)上表现与传统方法相当，在更复杂的数据集上则明显优于传统方法，在Cifar-10上比基于PCA和剪枝的编码方法准确率提高了高达10.2%，且性能增益随数据复杂度增加而增长。

### 结论

通过在经典到量子转换过程中保持保真度，FPQE为高质量的量子表示学习建立了可扩展且硬件高效的基础。

### 翻译

将经典视觉数据高效编码到量子态对于实现实用量子神经网络(QNNs)至关重要。然而，现有的编码方案在将高维图像适应到噪声中等规模量子(NISQ)设备的有限量子位时，往往会丢弃空间和语义信息。我们提出了一种保真度保留量子编码(FPQE)框架，能够实现近乎无损的数据压缩和量子编码。FPQE采用卷积编码器-解码器学习紧凑的多通道表示，这些表示能够以高保真度重建原始数据，然后通过振幅编码将这些表示映射到量子态。实验结果表明，FPQE在MNIST等简单数据集上表现与传统方法相当，而在更复杂的数据集上则实现了明显改进，在Cifar-10上比基于PCA和剪枝的编码方法准确率提高了高达10.2%。性能增益随数据复杂度增加而增长，展示了FPQE跨不同视觉域保留高级结构信息的能力。通过在经典到量子转换过程中保持保真度，FPQE为高质量的量子表示学习建立了可扩展且硬件高效的基础。


### 论文摘要

Efficiently encoding classical visual data into quantum states is essential for realizing practical quantum neural networks (QNNs). However, existing encoding schemes often discard spatial and semantic information when adapting high-dimensional images to the limited qubits of Noisy Intermediate-Scale Quantum (NISQ) devices. We propose a Fidelity-Preserving Quantum Encoding (FPQE) framework that performs near lossless data compression and quantum encoding. FPQE employs a convolutional encoder-decoder to learn compact multi-channel representations capable of reconstructing the original data with high fidelity, which are then mapped into quantum states through amplitude encoding. Experimental results show that FPQE performs comparably to conventional methods on simple datasets such as MNIST, while achieving clear improvements on more complex ones, outperforming PCA and pruning based encodings by up to 10.2\% accuracy on Cifar-10. The performance gain grows with data complexity, demonstrating FPQE's ability to preserve high-level structural information across diverse visual domains. By maintaining fidelity during classical to quantum transformation, FPQE establishes a scalable and hardware efficient foundation for high-quality quantum representation learning.

---

## 21. From Machine Learning Documentation to Requirements: Bridging Processes with Requirements Languages

**论文链接:** [http://arxiv.org/abs/2511.15340v1](http://arxiv.org/abs/2511.15340v1)

**作者:** Yi Peng, Hans-Martin Heyn, Jennifer Horkoff

**发布时间:** 2025-11-19

**备注:** To be published in proceedings of the 26th International Conference on Product-Focused Software Process Improvement (PROFES 2025). All raw and processed data are available in online repository, see https://doi.org/10.6084/m9.figshare.28564058.v1

### GPT解析

### 总结

本研究探讨了如何从机器学习文档中提取需求工程相关信息，并将其转化为结构化需求，以解决ML系统开发中集成和验证ML组件的挑战。

### 背景

在机器学习系统的软件工程过程中，集成和验证ML组件是一个主要挑战。传统需求工程过程在规范ML组件需求（包括模型和数据）方面面临新的障碍。

### 目的

研究ML文档中需求工程相关信息的内容和性质，并评估如何有效地将ML特定知识转化为结构化需求。

### 方法

首先研究20个公开可用的ModelCards和DataSheets中需求工程相关信息的数量和性质；然后评估三种成熟的需求工程表示法（EARS、Rupp的模板和Volere）如何将这些知识结构化为需求。

### 主要发现

ModelCards和DataSheets包含大量潜在的需求工程相关信息，并且有将ML特定知识转化为结构化需求的途径。

### 结论

可以将ML文档纳入ML系统的软件工程过程中，将ML特定知识转化为结构化需求。

### 翻译

在机器学习系统的软件工程过程中，集成和验证ML组件是一个主要挑战。一个前提是规范ML组件需求，包括模型和数据，这是传统需求工程过程面临的新障碍。在此背景下，ML文档（如ModelCards和DataSheets）是一个未被充分探索的需求工程相关信息来源。然而，不确定可以从这些文档中提取多少需求工程相关信息。本研究首先研究了20个公开可用的ModelCards和DataSheets中需求工程相关信息的数量和性质。我们表明这些文档包含大量潜在的需求工程相关信息。接下来，我们评估了三种成熟的需求工程表示法（EARS、Rupp的模板和Volere）如何有效地将这些知识结构化为需求。我们的结果表明，有将ML特定知识转化为结构化需求的途径，可以在ML系统的软件工程过程中纳入ML文档。


### 论文摘要

In software engineering processes for machine learning (ML)-enabled systems, integrating and verifying ML components is a major challenge. A prerequisite is the specification of ML component requirements, including models and data, an area where traditional requirements engineering (RE) processes face new obstacles. An underexplored source of RE-relevant information in this context is ML documentation such as ModelCards and DataSheets. However, it is uncertain to what extent RE-relevant information can be extracted from these documents. This study first investigates the amount and nature of RE-relevant information in 20 publicly available ModelCards and DataSheets. We show that these documents contain a significant amount of potentially RE-relevant information. Next, we evaluate how effectively three established RE representations (EARS, Rupp's template, and Volere) can structure this knowledge into requirements. Our results demonstrate that there is a pathway to transform ML-specific knowledge into structured requirements, incorporating ML documentation in software engineering processes for ML systems.

---

## 22. On the Internal Semantics of Time-Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.15324v1](http://arxiv.org/abs/2511.15324v1)

**作者:** Atharva Pandey, Abhilash Neog, Gautam Jajoo

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究对时间序列基础模型(TSFMs)中的概念可解释性进行了系统性调查，探索了不同层如何编码概念、概念参数的线性可恢复性、表示随模型深度的演变以及概念组合的处理方式。

### 背景

时间序列基础模型(TSFMs)最近出现作为学习不同时间域的通用范式，尽管这些模型在经验上取得了成功，但它们表示基本时间序列概念的内部机制仍不清楚。

### 目的

系统研究TSFM中的概念可解释性，检查：(i)哪些层编码哪些概念，(ii)概念参数是否可以线性恢复，(iii)表示如何随模型深度在概念解缠和抽象方面演变，(iv)模型如何处理概念组合。

### 方法

使用逐层分析、线性可恢复性测试和表示相似性度量来系统探索这些问题，提供TSFM语义的结构化描述。

### 主要发现

早期层主要捕获局部、时域模式（如AR(1)、水平偏移、趋势），更深层编码分散和变化时间信号，而谱和变形因子仍然是最难线性恢复的；在组合设置中，探测性能下降，揭示了概念之间的干扰。

### 结论

尽管原子概念可以可靠地定位，但组合仍然是一个挑战，突显了当前TSFM表示相互作用的时间现象能力的关键局限性。

### 翻译

时间序列基础模型(TSFMs)最近出现作为学习不同时间域的通用范式。然而，尽管这些模型在经验上取得了成功，但它们表示基本时间序列概念的内部机制仍不清楚。在这项工作中，我们对TSFM中的概念可解释性进行了系统性研究。具体来说，我们检查：(i)哪些层编码哪些概念，(ii)概念参数是否可以线性恢复，(iii)表示如何随模型深度在概念解缠和抽象方面演变，以及(iv)模型如何处理概念组合。我们使用逐层分析、线性可恢复性测试和表示相似性度量系统探索了这些问题，提供了TSFM语义的结构化描述。由此获得的见解表明，早期层主要捕获局部、时域模式（如AR(1)、水平偏移、趋势），而更深层编码分散和变化时间信号，其中谱和变形因子仍然是最难线性恢复的。然而，在组合设置中，探测性能下降，揭示了概念之间的干扰。这突显出，虽然原子概念可以可靠地定位，但组合仍然是一个挑战，强调了当前TSFM表示相互作用的时间现象能力的关键局限性。


### 论文摘要

Time-series Foundation Models (TSFMs) have recently emerged as a universal paradigm for learning across diverse temporal domains. However, despite their empirical success, the internal mechanisms by which these models represent fundamental time-series concepts remain poorly understood. In this work, we undertake a systematic investigation of concept interpretability in TSFMs. Specifically, we examine: (i) which layers encode which concepts, (ii) whether concept parameters are linearly recoverable, (iii) how representations evolve in terms of concept disentanglement and abstraction across model depth, and (iv) how models process compositions of concepts. We systematically probe these questions using layer-wise analyses, linear recoverability tests, and representation similarity measures, providing a structured account of TSFM semantics. The resulting insights show that early layers mainly capture local, time-domain patterns (e.g., AR(1), level shifts, trends), while deeper layers encode dispersion and change-time signals, with spectral and warping factors remaining the hardest to recover linearly. In compositional settings, however, probe performance degrades, revealing interference between concepts. This highlights that while atomic concepts are reliably localized, composition remains a challenge, underscoring a key limitation in current TSFMs' ability to represent interacting temporal phenomena.

---

## 23. A Multimodal Transformer Approach for UAV Detection and Aerial Object Recognition Using Radar, Audio, and Video Data

**论文链接:** [http://arxiv.org/abs/2511.15312v1](http://arxiv.org/abs/2511.15312v1)

**作者:** Mauro Larrat, Claudomiro Sales

**发布时间:** 2025-11-19

**备注:** 23 pages, 7 figures

### GPT解析

### 总结

本研究设计并评估了一种新颖的多模态Transformer模型，整合雷达、视觉、红外和音频数据流，用于无人机检测和空中物体识别，实现了高精度和高效率。

### 背景

无人机检测和空中物体识别对现代监控和安全至关重要，需要克服单模态方法的局限性。

### 目的

设计并严格评估一种新颖的多模态Transformer模型，整合雷达、视觉带视频(RGB)、红外(IR)视频和音频等多种数据流。

### 方法

该架构有效地融合了每种模态的不同特征，利用Transformer的自注意力机制学习全面的、互补的和高判别性的表示用于分类。

### 主要发现

模型在独立测试集上表现出色，达到宏平均指标：0.9812准确率、0.9873召回率、0.9787精确率、0.9826 F1分数和0.9954特异性。在区分无人机与其他空中物体方面表现出特别高的精确率和召回率。计算分析证实了其效率，为1.09 GFLOPs，1.22百万参数，推理速度为41.11 FPS。

### 结论

该研究在空中物体分类方面取得了重大进展，通过Transformer架构验证了多模态数据融合的有效性，实现了最先进的性能，为复杂空域中的无人机检测和监控提供了高度准确和有弹性的解决方案。

### 翻译

无人机检测和空中物体识别对现代监控和安全至关重要，促使需要克服单模态方法局限性的鲁棒系统。本研究通过设计并严格评估一种新颖的多模态Transformer模型来解决这些挑战，该模型整合了多种数据流：雷达、视觉带视频(RGB)、红外(IR)视频和音频。该架构有效地融合了每种模态的不同特征，利用Transformer的自注意力机制学习全面的、互补的和高判别性的表示用于分类。模型在独立测试集上表现出色，达到宏平均指标：0.9812准确率、0.9873召回率、0.9787精确率、0.9826 F1分数和0.9954特异性。值得注意的是，它在区分无人机与其他空中物体方面表现出特别高的精确率和召回率。此外，计算分析证实了其效率，为1.09 GFLOPs，1.22百万参数，推理速度为41.11 FPS，突显了其适用于实时应用。本研究在空中物体分类方面取得了重大进展，通过Transformer架构验证了多模态数据融合的有效性，实现了最先进的性能，从而为复杂空域中的无人机检测和监控提供了高度准确和有弹性的解决方案。


### 论文摘要

Unmanned aerial vehicle (UAV) detection and aerial object recognition are critical for modern surveillance and security, prompting a need for robust systems that overcome limitations of single-modality approaches. This research addresses these challenges by designing and rigorously evaluating a novel multimodal Transformer model that integrates diverse data streams: radar, visual band video (RGB), infrared (IR) video, and audio. The architecture effectively fuses distinct features from each modality, leveraging the Transformer's self-attention mechanisms to learn comprehensive, complementary, and highly discriminative representations for classification. The model demonstrated exceptional performance on an independent test set, achieving macro-averaged metrics of 0.9812 accuracy, 0.9873 recall, 0.9787 precision, 0.9826 F1-score, and 0.9954 specificity. Notably, it exhibited particularly high precision and recall in distinguishing drones from other aerial objects. Furthermore, computational analysis confirmed its efficiency, with 1.09 GFLOPs, 1.22 million parameters, and an inference speed of 41.11 FPS, highlighting its suitability for real-time applications. This study presents a significant advancement in aerial object classification, validating the efficacy of multimodal data fusion via a Transformer architecture for achieving state-of-the-art performance, thereby offering a highly accurate and resilient solution for UAV detection and monitoring in complex airspace.

---

## 24. GRPO-RM: Fine-Tuning Representation Models via GRPO-Driven Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.15256v1](http://arxiv.org/abs/2511.15256v1)

**作者:** Yanchen Xu, Ziheng Jiao, Hongyuan Zhang, Xuelong Li

**发布时间:** 2025-11-19

### GPT解析

### 总结

这篇论文提出了GRPO-RM方法，将GRPO（一种用于微调大型语言模型的强化学习方法）推广到表示学习模型中。通过建立预定义输出集替代LLM中的令牌序列采样，并设计专门的奖励函数，该方法在各种真实世界数据集上得到了验证。

### 背景

GRPO是一种用于微调大型语言模型的强化学习方法，已在DeepSeek-R1等实际应用中证明其有效性。这引发了一个问题：GRPO是否可以推广到表示学习模型中。

### 目的

研究GRPO类似策略在表示模型后训练中的性能，并提出GRPO-RM方法，将GRPO原理应用于表示学习模型。

### 方法

提出GRPO-RM方法，建立预定义输出集来功能性地替代LLM中的令牌序列采样，从而生成输出组，这对于GRPO的概率驱动优化至关重要。此外，还设计了一个专门的奖励函数以适应表示模型的特性。

### 主要发现

通过在各种真实世界数据集上进行大量实验，验证了所提出方法的有效性。

### 结论

GRPO可以成功推广到表示学习模型中，GRPO-RM方法在表示模型的后训练中表现良好。

### 翻译

群相对策略优化（GRPO）是一种用于微调大型语言模型的强化学习方法，已在DeepSeek-R1等实际应用中证明其有效性。这引发了一个问题：GRPO是否可以推广到表示学习模型中。在本文中，我们提出了面向表示模型的群相对策略优化（GRPO-RM），并研究了GRPO类似策略在表示模型后训练中的性能。具体而言，我们的方法建立了预定义输出集来功能性地替代LLM中的令牌序列采样，从而生成输出组，这对于GRPO的概率驱动优化至关重要。此外，还设计了一个专门的奖励函数以适应表示模型的特性。在各种真实世界数据集上进行了大量实验，以验证我们提出方法的有效性。


### 论文摘要

The Group Relative Policy Optimization (GRPO), a reinforcement learning method used to fine-tune large language models (LLMs), has proved its effectiveness in practical applications such as DeepSeek-R1. It raises a question whether GRPO can be generalized to representation learning models. In this paper, we propose Group Relative Policy Optimization for Representation Model (GRPO-RM), and investigate the performance of GRPO-like policy in post-training representation models. Specifically, our method establishes a predefined output set to functionally replace token sequence sampling in LLMs, thereby generating an output group, which is essential for the probability-driven optimization of GRPO. In addition, a specialized reward function is designed to accommodate the properties of representation models. Extensive experiments are conducted on various real-world datasets to validate the effectiveness of our proposed method.

---

## 25. PLATONT: Learning a Platonic Representation for Unified Network Tomography

**论文链接:** [http://arxiv.org/abs/2511.15251v1](http://arxiv.org/abs/2511.15251v1)

**作者:** Chengze Du, Heng Xu, Zhiwei Yu, Bo Liu, Jialong Li

**发布时间:** 2025-11-19

### GPT解析

### 总结

论文提出了PLATONT统一框架，通过多模态对齐和对比学习学习共享的潜在网络状态，提高跨任务泛化能力，实验显示该方法在链路估计、拓扑推断和流量预测方面优于现有方法。

### 背景

网络诊断旨在从外部观测中推断隐藏的网络状态，如链路性能、流量负载和拓扑结构。现有方法通常单独解决这些问题，并依赖于有限的特定任务信号，限制了泛化能力和可解释性。

### 目的

开发一个统一框架，将不同的网络指标（如延迟、丢失、带宽）建模为共享潜在网络状态的投影，提高跨任务泛化能力和准确性。

### 方法

提出PLATONT框架，遵循柏拉图表示假设，通过多模态对齐和对比学习来学习潜在网络状态，在共享的潜在空间内训练多个诊断任务，构建紧凑和结构化的表示。

### 主要发现

在合成和真实数据集上的实验表明，PLATONT在链路估计、拓扑推断和流量预测方面始终优于现有方法，实现了更高的准确性和在不同网络条件下的更强鲁棒性。

### 结论

PLATONT作为一个统一的框架，能够有效整合多种网络诊断任务，通过共享潜在网络状态的学习提高了泛化能力和准确性，为网络诊断提供了更强大、更通用的解决方案。

### 翻译

网络诊断旨在从外部观测中推断隐藏的网络状态，如链路性能、流量负载和拓扑结构。大多数现有方法单独解决这些问题，并依赖于有限的特定任务信号，这限制了泛化能力和可解释性。我们提出了PLATONT，一个统一框架，将不同的网络指标（如延迟、丢失、带宽）建模为共享潜在网络状态的投影。在柏拉图表示假设的指导下，PLATONT通过多模态对齐和对比学习来学习这种潜在状态。通过在共享的潜在空间内训练多个诊断任务，它构建了紧凑且结构化的表示，提高了跨任务泛化能力。在合成和真实数据集上的实验表明，PLATONT在链路估计、拓扑推断和流量预测方面始终优于现有方法，实现了更高的准确性和在不同网络条件下的更强鲁棒性。


### 论文摘要

Network tomography aims to infer hidden network states, such as link performance, traffic load, and topology, from external observations. Most existing methods solve these problems separately and depend on limited task-specific signals, which limits generalization and interpretability. We present PLATONT, a unified framework that models different network indicators (e.g., delay, loss, bandwidth) as projections of a shared latent network state. Guided by the Platonic Representation Hypothesis, PLATONT learns this latent state through multimodal alignment and contrastive learning. By training multiple tomography tasks within a shared latent space, it builds compact and structured representations that improve cross-task generalization. Experiments on synthetic and real-world datasets show that PLATONT consistently outperforms existing methods in link estimation, topology inference, and traffic prediction, achieving higher accuracy and stronger robustness under varying network conditions.

---

## 26. Towards Unbiased Cross-Modal Representation Learning for Food Image-to-Recipe Retrieval

**论文链接:** [http://arxiv.org/abs/2511.15201v1](http://arxiv.org/abs/2511.15201v1)

**作者:** Qing Wang, Chong-Wah Ngo, Ee-Peng Lim

**发布时间:** 2025-11-19

### GPT解析

### 总结

该论文研究了食谱和食物图像在跨模态检索中的表示学习挑战，提出使用因果理论建模并解决现有方法中的偏见问题，显著提高了检索性能。

### 背景

食谱与其烹饪成品之间存在因果关系，但现有方法将食谱视为描述菜肴外观的文本来源，会产生偏见并误导相似性判断。食物图像无法完全捕捉食谱中的所有细节，因为烹饪过程、菜肴呈现和图像捕捉条件等因素影响。当前表示学习倾向于捕捉主要视觉-文本对齐，而忽略了决定检索相关性的细微变化。

### 目的

建模跨模态表示学习中的偏见，提高食谱到图像的检索性能，解决现有方法中的相似性判断偏差问题。

### 方法

使用因果理论建模跨模态表示学习中的偏见，将配料视为混杂因素来源，通过后门调整减轻偏见。通过因果干预重新制定传统检索模型，添加额外项去除相似性判断中的潜在偏见。提出一个即插即用的多标签配料分类器神经模块用于消除偏见。

### 主要发现

在Recipe1M数据集上，通过理论引导的公式，证明了检索性能在1K、10K和50K的不同测试数据规模下都达到了MedR=1。该数据集上报告了新的最先进搜索性能。

### 结论

使用因果理论建模跨模态表示学习中的偏见可以有效提高检索性能。提出的即插即用神经模块能够有效消除偏见，显著提升检索效果。

### 翻译

本文解决了跨模态检索问题中学习食谱和食物图像表示的挑战。由于食谱与其烹饪成品之间的关系是因果关系，将食谱视为描述菜肴外观的文本来源进行表示学习（如现有方法所做的那样）会产生偏见，误导图像和食谱相似性判断。具体而言，由于烹饪过程、菜肴呈现和图像捕捉条件等因素，食物图像可能无法同等捕捉食谱中的每个细节。当前的表示学习倾向于捕捉主要的视觉-文本对齐，而忽略了决定检索相关性的细微变化。在本文中，我们使用因果理论对跨模态表示学习中的这种偏见进行建模。该问题的因果观点表明配料是混杂因素来源之一，简单的后门调整可以减轻这种偏见。通过因果干预，我们重新制定了传统的食物到食谱检索模型，添加了一个额外项以去除相似性判断中的潜在偏见。基于这种理论引导的公式，我们在Recipe1M数据集上经验性地证明了检索的oracle性能在1K、10K甚至50K的不同测试数据规模下都达到了MedR=1。我们还提出了一个即插即用的神经模块，本质上是一个用于消除偏见的多标签配料分类器。在Recipe1M数据集上报告了新的最先进搜索性能。


### 论文摘要

This paper addresses the challenges of learning representations for recipes and food images in the cross-modal retrieval problem. As the relationship between a recipe and its cooked dish is cause-and-effect, treating a recipe as a text source describing the visual appearance of a dish for learning representation, as the existing approaches, will create bias misleading image-and-recipe similarity judgment. Specifically, a food image may not equally capture every detail in a recipe, due to factors such as the cooking process, dish presentation, and image-capturing conditions. The current representation learning tends to capture dominant visual-text alignment while overlooking subtle variations that determine retrieval relevance. In this paper, we model such bias in cross-modal representation learning using causal theory. The causal view of this problem suggests ingredients as one of the confounder sources and a simple backdoor adjustment can alleviate the bias. By causal intervention, we reformulate the conventional model for food-to-recipe retrieval with an additional term to remove the potential bias in similarity judgment. Based on this theory-informed formulation, we empirically prove the oracle performance of retrieval on the Recipe1M dataset to be MedR=1 across the testing data sizes of 1K, 10K, and even 50K. We also propose a plug-and-play neural module, which is essentially a multi-label ingredient classifier for debiasing. New state-of-the-art search performances are reported on the Recipe1M dataset.

---

## 27. Multimodal Wireless Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.15162v1](http://arxiv.org/abs/2511.15162v1)

**作者:** Ahmed Aboulfotouh, Hatem Abou-Zeid

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了第一个多模态无线基础模型，能够同时处理原始IQ流和类图像无线模态（如频谱图和CSI），并在多种任务上表现良好，为AI原生6G和感知、通信与定位联合愿景提供了具体步骤。

### 背景

当前的无线基础模型只处理一种模态，但根据任务和操作条件，最有信息量的模态会变化，没有单一模态适合所有任务，这限制了无线基础模型的应用范围。

### 目的

设计能够接受多种模态的无线基础模型，以支持更广泛和多样化的任务和场景。

### 方法

提出并构建了第一个多模态无线基础模型，能够处理原始IQ流和类图像无线模态。引入了多模态设置下的掩码无线建模，这是一种自监督目标和预训练方法，可以从IQ流和类图像无线模态中学习联合表示。

### 主要发现

该多模态无线基础模型与单模态无线基础模型具有竞争力，并且在某些情况下超越了它们的性能。研究结果表明开发支持不同模态下多样化无线任务的多模态无线基础模型具有巨大潜力。

### 结论

多模态无线基础模型为AI原生6G以及感知、通信和定位的联合愿景提供了具体进展。

### 翻译

无线基础模型最近展示了有前景的能力，能够联合执行多种无线功能并有效适应新环境。然而，虽然当前的无线基础模型只处理一种模态，但根据任务和操作条件，最有信息量的模态会变化，没有单一模态适合所有任务。因此，无线基础模型应该被设计为接受多种模态，以实现更广泛和多样化的任务和场景。在这项工作中，我们提出并构建了第一个多模态无线基础模型，能够处理原始IQ流和类图像无线模态（如频谱图和CSI），并在两种模态上执行多种任务。我们引入了多模态设置下的掩码无线建模，这是一种自监督目标和预训练方法，可以从IQ流和类图像无线模态中学习联合表示。我们在两种模态家族的五个任务上评估了该模型：基于图像的（人类活动感知、射频信号分类、5G NR定位）和基于IQ的（射频设备指纹识别、干扰检测/分类）。多模态无线基础模型与单模态无线基础模型具有竞争力，并且在几种情况下超越了它们的性能。我们的研究结果表明开发支持不同模态下多样化无线任务的多模态无线基础模型具有巨大潜力。我们相信这为AI原生6G以及感知、通信和定位的联合愿景提供了具体步骤。


### 论文摘要

Wireless foundation models (WFMs) have recently demonstrated promising capabilities, jointly performing multiple wireless functions and adapting effectively to new environments. However, while current WFMs process only one modality, depending on the task and operating conditions, the most informative modality changes and no single modality is best for all tasks. WFMs should therefore be designed to accept multiple modalities to enable a broader and more diverse range of tasks and scenarios. In this work, we propose and build the first multimodal wireless foundation model capable of processing both raw IQ streams and image-like wireless modalities (e.g., spectrograms and CSI) and performing multiple tasks across both. We introduce masked wireless modeling for the multimodal setting, a self-supervised objective and pretraining recipe that learns a joint representation from IQ streams and image-like wireless modalities. We evaluate the model on five tasks across both modality families: image-based (human activity sensing, RF signal classification, 5G NR positioning) and IQ-based (RF device fingerprinting, interference detection/classification). The multimodal WFM is competitive with single-modality WFMs, and in several cases surpasses their performance. Our results demonstrates the strong potential of developing multimodal WFMs that support diverse wireless tasks across different modalities. We believe this provides a concrete step toward both AI-native 6G and the vision of joint sensing, communication, and localization.

---

## 28. Generating Natural-Language Surgical Feedback: From Structured Representation to Domain-Grounded Evaluation

**论文链接:** [http://arxiv.org/abs/2511.15159v1](http://arxiv.org/abs/2511.15159v1)

**作者:** Firdavs Nasriddinov, Rafal Kocielnik, Anima Anandkumar, Andrew J. Hung

**发布时间:** 2025-11-19

**备注:** Accepted as proceedings paper for ML4H 2025

### GPT解析

### 总结

该研究提出了一种结构感知的管道，通过学习手术动作本体论和使用IAT三元组表征来指导反馈生成，显著提高了自动生成的手术培训反馈的质量和临床相关性。

### 背景

高质量的手术中反馈对于提高学员表现和长期技能获取至关重要。自动化自然的、类似培训师风格的反馈可以提供及时、可访问和一致的指导，但需要理解临床相关表征的模型。

### 目的

开发一个能够理解临床相关表征的模型，以自动化生成类似培训师的反馈；创建一个结构感知的管道，从真实的培训师对学员的转录文本中学习手术动作本体论，并使用它来条件化反馈生成。

### 方法

从真实的反馈文本中挖掘仪器-动作-目标(IAT)三元组，并将表面形式聚类到规范化类别中；微调一个视频到IAT的模型，利用手术程序和任务上下文以及细粒度的仪器时间运动；使用IAT三元组表征来指导GPT-4o生成临床基础的、类似培训师的反馈。

### 主要发现

在视频到IAT识别任务中，上下文注入和时间跟踪带来了一致的AUC增益；在反馈文本生成任务中，仅使用视频的GPT-4o得分为2.17，而IAT条件化达到2.44（提升12.4%），合格生成比例从21%增加到42%；传统文本相似度指标也得到改善：词错误率降低15-31%，ROUGE增加9-64%。

### 结论

在明确的IAT结构基础上进行生成可以提高保真度，并产生临床可验证的理由，支持在手术培训中的可审计使用。

### 翻译

高质量的手术培训师术中反馈对于提高学员表现和长期技能获取至关重要。自动化自然的、培训师风格的反馈承诺提供及时、可访问和一致的规模化指导，但需要理解临床相关表征的模型。我们提出了一种结构感知的管道，从真实的培训师对学员的转录文本（33台手术）中学习手术动作本体论，并使用它来条件化反馈生成。我们的贡献包括：（1）从真实的反馈文本中挖掘仪器-动作-目标（IAT）三元组，并将表面形式聚类到规范化类别中；（2）微调一个视频到IAT模型，利用手术程序和任务上下文以及细粒度的仪器时间运动；（3）展示如何有效地使用IAT三元组表征来指导GPT-4o生成临床基础的、培训师风格的反馈。我们表明，在任务1：视频到IAT识别中，我们的上下文注入和时间跟踪带来了一致的AUC增益（仪器：0.67到0.74；动作：0.60到0.63；组织：0.74到0.79）。对于任务2：反馈文本生成（在1-5保真度评分标准上评定，其中1=相反/不安全，3=可接受，5=与人类培训师完美匹配），仅从视频生成的GPT-4o得分为2.17，而IAT条件化达到2.44（+12.4%），将得分≥3的可接受生成比例从21%增加到42%。传统的文本相似度指标也得到改善：词错误率降低15-31%，ROUGE（短语/子字符串重叠）增加9-64%。在明确的IAT结构基础上进行生成可以提高保真度，并产生临床可验证的理由，支持在手术培训中的可审计使用。


### 论文摘要

High-quality intraoperative feedback from a surgical trainer is pivotal for improving trainee performance and long-term skill acquisition. Automating natural, trainer-style feedback promises timely, accessible, and consistent guidance at scale but requires models that understand clinically relevant representations. We present a structure-aware pipeline that learns a surgical action ontology from real trainer-to-trainee transcripts (33 surgeries) and uses it to condition feedback generation. We contribute by (1) mining Instrument-Action-Target (IAT) triplets from real-world feedback text and clustering surface forms into normalized categories, (2) fine-tuning a video-to-IAT model that leverages the surgical procedure and task contexts as well as fine-grained temporal instrument motion, and (3) demonstrating how to effectively use IAT triplet representations to guide GPT-4o in generating clinically grounded, trainer-style feedback. We show that, on Task 1: Video-to-IAT recognition, our context injection and temporal tracking deliver consistent AUC gains (Instrument: 0.67 to 0.74; Action: 0.60 to 0.63; Tissue: 0.74 to 0.79). For Task 2: feedback text generation (rated on a 1-5 fidelity rubric where 1 = opposite/unsafe, 3 = admissible, and 5 = perfect match to a human trainer), GPT-4o from video alone scores 2.17, while IAT conditioning reaches 2.44 (+12.4%), doubling the share of admissible generations with score >= 3 from 21% to 42%. Traditional text-similarity metrics also improve: word error rate decreases by 15-31% and ROUGE (phrase/substring overlap) increases by 9-64%. Grounding generation in explicit IAT structure improves fidelity and yields clinician-verifiable rationales, supporting auditable use in surgical training.

---

## 29. DCL-SE: Dynamic Curriculum Learning for Spatiotemporal Encoding of Brain Imaging

**论文链接:** [http://arxiv.org/abs/2511.15151v1](http://arxiv.org/abs/2511.15151v1)

**作者:** Meihua Zhou, Xinyu Tong, Jiarui Zhao, Min Cheng, Li Yang, Lei Tian, Nan Wan

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了一种名为DCL-SE的高维神经影像分析框架，通过数据驱动的时空编码和动态课程学习策略，有效解决了临床诊断中时空保真度和模型适应性受限的问题。

### 背景

高维神经影像分析用于临床诊断常面临时空保真度妥协和大规模通用模型适应性有限的问题。

### 目的

开发一种能够克服上述挑战的端到端框架，提高神经影像分析的准确性和实用性。

### 方法

基于数据驱动的时空编码(DaSE)，利用近似秩池化(ARP)将三维脑体积数据编码为二维动态表示，并通过动态组机制(DGM)指导的动态课程学习策略逐步训练解码器，从全局到精细层面改进特征提取。

### 主要发现

在六个公开数据集(包括阿尔茨海默病和脑肿瘤分类、脑动脉分割和脑龄预测)上评估，DCL-SE在准确性、鲁棒性和可解释性方面均优于现有方法。

### 结论

在大规模预训练网络时代，紧凑、特定任务架构对于高维神经影像分析至关重要。

### 翻译

高维神经影像分析用于临床诊断通常受到时空保真度妥协和大规模通用模型有限适应性的约束。为应对这些挑战，我们引入了用于时空编码的动态课程学习(DCL-SE)，这是一个以数据驱动的时空编码(DaSE)为中心的端到端框架。我们利用近似秩池化(ARP)将三维体积脑数据高效编码为信息丰富的二维动态表示，然后采用由动态组机制(DGM)指导的动态课程学习策略，逐步训练解码器，从全局解剖结构到精细病理细节改进特征提取。在六个公开可用数据集(包括阿尔茨海默病和脑肿瘤分类、脑动脉分割和脑龄预测)上的评估表明，DCL-SE在准确性、鲁棒性和可解释性方面始终优于现有方法。这些发现强调了在大规模预训练网络时代，紧凑、特定任务架构的关键重要性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高维神经影像分析在临床诊断中的时空保真度不足和大规模通用模型适应性有限的问题。这个问题很重要，因为神经系统疾病如阿尔茨海默病和脑肿瘤是临床医学中的重大挑战，早期准确诊断能显著影响患者预后，而现代神经影像技术虽能提供详细脑部结构可视化，但将其转化为可靠诊断决策仍面临困难，特别是在处理3D到2D转换时容易丢失关键空间信息。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法的局限性：2D方法缺乏全局解剖上下文，3D模型计算需求高，混合方法易产生插值错误，传统课程学习依赖静态难度级别，而大规模模型在临床应用中受限于数据稀缺和隐私问题。作者借鉴了课程学习的基本思想、秩池化的有序敏感聚合技术以及分组卷积的参数高效特性，但创新性地将它们结合，设计了数据驱动的时空编码框架和动态组机制，以模仿临床专家的分层推理过程，从全局结构逐步关注到病理细节。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过数据驱动的时空编码将3D脑影像高效转换为信息丰富的2D表示，然后采用动态课程学习策略，模仿临床诊断的渐进推理过程。整体流程分为两个主要阶段：首先使用近似秩池化技术将有序3D脑MRI切片编码为紧凑的二维动态表示，保留空间结构和解剖进展信息；然后通过动态课程学习策略，由动态组机制指导，逐步从全局解剖结构到细病理特征进行特征提取和解码，最终进行任务特定预测（如疾病分类、组织分割或脑龄预测）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) DCL-SE轻量级自适应框架，能根据复杂性和临床显著性动态优先特征提取；2) 数据驱动的时空编码方法，结合ARP编码和动态课程解码；3) 动态课程学习策略，根据特征复杂性自动调整学习进度；4) 动态组机制，在每个课程阶段自适应重新校准特征重要性。相比之前工作，不同之处在于：传统方法难以平衡计算效率和空间保真度，而DCL-SE通过DaSE框架既保留3D空间信息又利用2D处理高效性；传统课程学习使用静态难度级别，而DCL采用数据驱动的动态调整；大规模预训练模型在临床应用中受限，而DCL-SE是专门为临床神经影像设计的轻量级解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DCL-SE通过数据驱动的时空编码和动态课程学习策略，高效地将3D脑影像转换为2D表示同时保留关键时空信息，在多个神经影像任务上实现了优于现有方法的性能，为临床神经影像分析提供了轻量级、高效且可解释的解决方案。'}


### 论文摘要

High-dimensional neuroimaging analyses for clinical diagnosis are often constrained by compromises in spatiotemporal fidelity and by the limited adaptability of large-scale, general-purpose models. To address these challenges, we introduce Dynamic Curriculum Learning for Spatiotemporal Encoding (DCL-SE), an end-to-end framework centered on data-driven spatiotemporal encoding (DaSE). We leverage Approximate Rank Pooling (ARP) to efficiently encode three-dimensional volumetric brain data into information-rich, two-dimensional dynamic representations, and then employ a dynamic curriculum learning strategy, guided by a Dynamic Group Mechanism (DGM), to progressively train the decoder, refining feature extraction from global anatomical structures to fine pathological details. Evaluated across six publicly available datasets, including Alzheimer's disease and brain tumor classification, cerebral artery segmentation, and brain age prediction, DCL-SE consistently outperforms existing methods in accuracy, robustness, and interpretability. These findings underscore the critical importance of compact, task-specific architectures in the era of large-scale pretrained networks.

---

## 30. Cross-Modal Consistency-Guided Active Learning for Affective BCI Systems

**论文链接:** [http://arxiv.org/abs/2511.15138v1](http://arxiv.org/abs/2511.15138v1)

**作者:** Hyo-Jeong Jang, Hye-Bin Shin, Kang Yin

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种不确定性感知的主动学习框架，通过结合模型不确定性和跨模态一致性来增强对标签噪声的鲁棒性，解决了基于脑电图(EEG)的情感识别中高质量标签稀缺的问题。

### 背景

深度学习模型在大量高质量标签条件下表现最佳，但在基于EEG的情感识别中，这种条件很少能达到。EEG信号容易被伪影和个体差异污染，而情感标签通常来自主观且不一致的报告，这使得稳健的情感解码特别困难。

### 目的

开发一种能够增强对标签噪声鲁棒性的方法，通过联合利用模型不确定性和跨模态一致性来提高EEG情感识别的准确性。

### 方法

提出不确定性感知的主动学习框架，不仅依赖基于EEG的不确定性估计，还评估跨模态对齐以确定不确定性来源。通过表示对齐模块将EEG和面部特征嵌入共享潜在空间，强制模态间语义一致性，并将残差差异视为噪声引起的不一致性，在主动学习过程中选择性查询这些样本获取反馈。

### 主要发现

在ASCERTAIN数据集上的实验验证了所提出方法的效率和鲁棒性，证明了其作为数据高效且噪声容忍的EEG情感解码方法的潜力。

### 结论

该反馈驱动的过程能够引导网络朝向可靠、信息丰富的样本，减少噪声标签的影响，为脑机接口系统中的EEG情感解码提供了有效解决方案。

### 翻译

深度学习模型在丰富、高质量标签条件下表现最佳，然而在基于脑电图(EEG)的情感识别中，这种条件很少能够实现。脑电图信号容易被伪影和个体差异污染，而情感标签通常来自主观且不一致的报告，这使得稳健的情感解码尤其困难。我们提出了一种不确定性感知的主动学习框架，通过联合利用模型不确定性和跨模态一致性来增强对标签噪声的鲁棒性。该方法不仅依赖基于EEG的不确定性估计，还评估跨模态对齐以确定不确定性是源于认知模糊还是传感器噪声。表示对齐模块将EEG和面部特征嵌入到共享的潜在空间中，强制模态之间的语义一致性。残差差异被视为噪声引起的不一致性，这些样本在主动学习过程中被选择性查询以获取专家反馈。这种反馈驱动的过程引导网络朝向可靠、信息丰富的样本，并减少噪声标签的影响。在ASCERTAIN数据集上的实验检验了我们方法的效率和鲁棒性，突显了其作为脑机接口系统中基于EEG情感解码的数据高效且噪声容忍方法的潜力。


### 论文摘要

Deep learning models perform best with abundant, high-quality labels, yet such conditions are rarely achievable in EEG-based emotion recognition. Electroencephalogram (EEG) signals are easily corrupted by artifacts and individual variability, while emotional labels often stem from subjective and inconsistent reports-making robust affective decoding particularly difficult. We propose an uncertainty-aware active learning framework that enhances robustness to label noise by jointly leveraging model uncertainty and cross-modal consistency. Instead of relying solely on EEG-based uncertainty estimates, the method evaluates cross-modal alignment to determine whether uncertainty originates from cognitive ambiguity or sensor noise. A representation alignment module embeds EEG and face features into a shared latent space, enforcing semantic coherence between modalities. Residual discrepancies are treated as noise-induced inconsistencies, and these samples are selectively queried for oracle feedback during active learning. This feedback-driven process guides the network toward reliable, informative samples and reduces the impact of noisy labels. Experiments on the ASCERTAIN dataset examine the efficiency and robustness of ours, highlighting its potential as a data-efficient and noise-tolerant approach for EEG-based affective decoding in brain-computer interface systems.

---

## 31. Multi-Aspect Cross-modal Quantization for Generative Recommendation

**论文链接:** [http://arxiv.org/abs/2511.15122v1](http://arxiv.org/abs/2511.15122v1)

**作者:** Fuwei Zhang, Xiaoyu Liu, Dongbo Xi, Jishen Yin, Huan Chen, Peng Yan, Fuzhen Zhuang, Zhao Zhang

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

本文提出了一种名为MACRec的多方面跨模态量化生成式推荐方法，通过引入多模态信息并从不同角度整合到语义ID学习和生成模型训练中，解决了生成式推荐系统在语义ID构建和模型训练方面的挑战。

### 背景

生成式推荐(GR)已成为推荐系统的新范式，它通过量化表示离散化项目特征，将用户历史交互建模为离散标记序列，并基于这些序列使用下一标记预测方法预测下一个项目。

### 目的

解决当前生成式推荐方法在利用多模态信息和捕捉不同模态间深度复杂交互方面的局限性，从而学习高质量语义ID并有效训练GR模型。

### 方法

1)在ID学习过程中引入跨模态量化，通过多模态信息的互补集成降低冲突率，提高码本可用性；2)集成多方面跨模态对齐，包括隐式和显式对齐，以增强GR模型的生成能力。

### 主要发现

通过在三个著名推荐数据集上的广泛实验，验证了所提出的MACRec方法的有效性。

### 结论

MACRec方法通过多方面跨模量化和对齐技术，有效解决了生成式推荐系统在语义ID构建和模型训练方面的挑战，提高了推荐系统的性能。

### 翻译

生成式推荐(GR)已成为推荐系统的一种新范式。这种方法依赖于量化表示来离散化项目特征，将用户的历史交互建模为离散标记的序列。基于这些标记序列，GR通过使用下一标记预测方法来预测下一个项目。GR的挑战在于构建高质量、层次组织、最小冲突且有利于有效生成模型训练的语义标识符(IDs)。然而，当前方法在利用多模态信息和捕捉不同模态间深度复杂交互方面的能力有限，而这对于学习高质量语义ID和有效训练GR模型都是必不可少的。为了解决这一问题，我们提出了多方面跨模态量化生成式推荐(MACRec)，该方法从不同方面引入多模态信息，并将其融入到语义ID学习和生成模型训练中。具体而言，我们首先在ID学习过程中引入跨模态量化，通过多模态信息的互补集成有效降低冲突率，从而提高码本可用性。此外，为了进一步增强我们GR模型的生成能力，我们集成了多方面跨模态对齐，包括隐式和显式对齐。最后，我们在三个著名的推荐数据集上进行了广泛的实验，证明了我们提出方法的有效性。


### 论文摘要

Generative Recommendation (GR) has emerged as a new paradigm in recommender systems. This approach relies on quantized representations to discretize item features, modeling users' historical interactions as sequences of discrete tokens. Based on these tokenized sequences, GR predicts the next item by employing next-token prediction methods. The challenges of GR lie in constructing high-quality semantic identifiers (IDs) that are hierarchically organized, minimally conflicting, and conducive to effective generative model training. However, current approaches remain limited in their ability to harness multimodal information and to capture the deep and intricate interactions among diverse modalities, both of which are essential for learning high-quality semantic IDs and for effectively training GR models. To address this, we propose Multi-Aspect Cross-modal quantization for generative Recommendation (MACRec), which introduces multimodal information and incorporates it into both semantic ID learning and generative model training from different aspects. Specifically, we first introduce cross-modal quantization during the ID learning process, which effectively reduces conflict rates and thus improves codebook usability through the complementary integration of multimodal information. In addition, to further enhance the generative ability of our GR model, we incorporate multi-aspect cross-modal alignments, including the implicit and explicit alignments. Finally, we conduct extensive experiments on three well-known recommendation datasets to demonstrate the effectiveness of our proposed method.

---

## 32. Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit

**论文链接:** [http://arxiv.org/abs/2511.15120v1](http://arxiv.org/abs/2511.15120v1)

**作者:** Bohan Zhang, Zihao Wang, Hengyu Fu, Jason D. Lee

**发布时间:** 2025-11-19

**备注:** 86 pages, 2 figures. The order of the first two authors was determined by a coin flip

### GPT解析

### 总结

该研究探讨了深度学习中神经网络如何高效学习高维特征的问题，通过分析高斯多索引模型的梯度下降学习过程，证明了标准两层神经网络在适当条件下能够以最优的样本和时间复杂度学习目标函数。

### 背景

深度学习中一个核心问题是理解神经网络如何高效学习高维特征，表征学习是研究这一问题的关键设置。

### 目的

探索高斯多索引模型的梯度下降学习过程，研究标准两层神经网络在表征学习中的性能和效率。

### 方法

研究通过逐层梯度下降训练的标准两层神经网络，分析其在非退化假设下的学习过程和性能表现。

### 主要发现

在链接函数的非退化假设下，网络可以使用接近最优的测试错误率学习目标；样本和时间复杂度均达到信息理论限制；梯度下降学习的第一阶段，内部权重执行幂迭代过程，隐式模拟隐藏子空间的谱启动；第一层需要训练超过一定步数才能实现最优结果。

### 结论

该工作展示了神经网络在样本和时间效率方面有效学习层次函数的能力，证明了深度学习模型在表征学习中的高效性。

### 翻译

在深度学习中，一个核心问题是理解神经网络如何高效学习高维特征。为此，我们探索了具有隐藏子空间U的高斯多索引模型f(x)=g(Ux)的梯度下降学习，这是研究表征学习的标准设置。我们证明，在链接函数的通用非退化假设下，通过逐层梯度下降训练的标准两层神经网络可以使用接近最优的测试错误率学习目标，仅需O(d)样本和O(d^2)时间。样本和时间复杂度均与信息理论限制一致，因此是最优的。在梯度下降学习的第一阶段，证明过程显示内部权重可以执行幂迭代过程。这一过程隐式模拟了整个隐藏子空间的谱启动，并最终消除有限样本噪声，恢复该子空间。令人惊讶的是，只有当第一层训练超过O(1)步时，才能实现最优结果。这项工作展示了神经网络在样本和时间效率方面有效学习层次函数的能力。


### 论文摘要

In deep learning, a central issue is to understand how neural networks efficiently learn high-dimensional features. To this end, we explore the gradient descent learning of a general Gaussian Multi-index model $f(\boldsymbol{x})=g(\boldsymbol{U}\boldsymbol{x})$ with hidden subspace $\boldsymbol{U}\in \mathbb{R}^{r\times d}$, which is the canonical setup to study representation learning. We prove that under generic non-degenerate assumptions on the link function, a standard two-layer neural network trained via layer-wise gradient descent can agnostically learn the target with $o_d(1)$ test error using $\widetilde{\mathcal{O}}(d)$ samples and $\widetilde{\mathcal{O}}(d^2)$ time. The sample and time complexity both align with the information-theoretic limit up to leading order and are therefore optimal. During the first stage of gradient descent learning, the proof proceeds via showing that the inner weights can perform a power-iteration process. This process implicitly mimics a spectral start for the whole span of the hidden subspace and eventually eliminates finite-sample noise and recovers this span. It surprisingly indicates that optimal results can only be achieved if the first layer is trained for more than $\mathcal{O}(1)$ steps. This work demonstrates the ability of neural networks to effectively learn hierarchical functions with respect to both sample and time efficiency.

---

## 33. Graph Query Networks for Object Detection with Automotive Radar

**论文链接:** [http://arxiv.org/abs/2511.15271v1](http://arxiv.org/abs/2511.15271v1)

**作者:** Loveneet Saini, Hasan Tercan, Tobias Meisen

**发布时间:** 2025-11-19

**备注:** Accepted in WACV 2026 Main Conference

### GPT解析

### 总结

本研究提出了一种基于图查询网络(GQN)的新型目标检测框架，用于解决3D雷达在汽车360度感知中的挑战。GQN通过将雷达感知的物体建模为图结构，并采用动态图查询机制和两个专门设计的模块，显著提高了目标检测性能，同时降低了计算开销。

### 背景

3D雷达在汽车360度感知中至关重要，但雷达的长波长导致稀疏和不规则的反射数据，这给传统的基于网格和序列的卷积和Transformer检测器带来了挑战。

### 目的

开发一种能够有效处理雷达稀疏数据的新型目标检测框架，提高检测精度并降低计算复杂度。

### 方法

提出图查询网络(GQN)，一种基于注意力的框架，将雷达感知的物体建模为图。GQN采用图查询概念动态关注鸟瞰图空间，构建特定于物体的图，并通过两个新颖模块处理：EdgeFocus用于关系推理，DeepContext Pooling用于上下文聚合。

### 主要发现

在NuScenes数据集上，GQN将相对mAP提高了高达53%，比之前最强的雷达方法提高了8.2%，同时将图构建的峰值开销减少了80%，且计算成本适中。

### 结论

图查询网络(GQN)为3D雷达目标检测提供了一种有效解决方案，通过图结构建模和专门设计的模块，显著提高了检测性能并优化了计算效率。

### 翻译

使用3D雷达进行目标检测对汽车360度感知至关重要，但雷达的长波长会产生稀疏和不规则的反射，这对传统的基于网格和序列的卷积和Transformer检测器提出了挑战。本文引入了图查询网络(GQN)，一种基于注意力的框架，将雷达感知的物体建模为图，以提取个性化的关系和上下文特征。GQN采用新颖的图查询概念动态关注鸟瞰图(BEV)空间，构建特定于物体的图，并通过两个新颖模块处理：EdgeFocus用于关系推理，DeepContext Pooling用于上下文聚合。在NuScenes数据集上，GQN将相对mAP提高了高达53%，包括比之前最强的雷达方法提高8.2%，同时将图构建的峰值开销减少了80%，且计算成本适中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D雷达目标检测中的挑战：由于雷达波长较长，产生的反射稀疏且不规则，使得传统基于网格和序列的检测方法难以有效处理。这个问题在现实中非常重要，因为3D雷达是360度汽车感知系统的关键组成部分，相比激光雷达具有低成本、全天候工作的优势，准确可靠的目标检测对自动驾驶安全至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了雷达数据的特点（稀疏、不规则、非确定性），指出传统CNN和Transformer方法的局限性在于强制使用规则结构，可能扭曲雷达数据的关键信息。他们设计了基于图的表示方法，保留数据的不规则性。作者借鉴了注意力机制、图神经网络和transformer中的查询机制，但针对雷达数据的特殊性进行了创新，引入了图查询概念和两个关键模块（EdgeFocus和DeepContext Pooling）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将雷达检测转化为图查询问题，通过动态构建和处理对象特定的图来提取特征。整体流程包括：1)图查询初始化：使用注意力机制从BEV空间选择节点；2)图查询更新：通过EdgeFocus模块处理节点间关系；3)图上下文建模：使用DeepContext Pooling模块实现查询间信息共享；4)多集图推理：使用不同采样比例的查询组捕获多层次特征；5)统一推理架构：整合时间、空间和图特征进行最终检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)图查询机制，动态构建对象特定图；2)EdgeFocus模块，提取细粒度关系特征；3)DeepContext Pooling模块，建模对象间交互；4)多集图推理设计，捕获不同稀疏度级别的结构；5)统一推理架构，整合多种特征提取方式。相比之前工作，GQN更高效（减少80%图构建开销），更灵活（可作为即插即用模块），能同时提取关系和上下文特征，在NuScenes数据集上实现了高达53%的相对mAP提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了图查询网络(GQN)，一种基于注意力机制的框架，通过动态构建和处理对象特定的图结构，有效解决了雷达数据稀疏性和不规则性带来的目标检测挑战，显著提升了雷达目标检测的性能和效率。'}


### 论文摘要

Object detection with 3D radar is essential for 360-degree automotive perception, but radar's long wavelengths produce sparse and irregular reflections that challenge traditional grid and sequence-based convolutional and transformer detectors. This paper introduces Graph Query Networks (GQN), an attention-based framework that models objects sensed by radar as graphs, to extract individualized relational and contextual features. GQN employs a novel concept of graph queries to dynamically attend over the bird's-eye view (BEV) space, constructing object-specific graphs processed by two novel modules: EdgeFocus for relational reasoning and DeepContext Pooling for contextual aggregation. On the NuScenes dataset, GQN improves relative mAP by up to +53%, including a +8.2% gain over the strongest prior radar method, while reducing peak graph construction overhead by 80% with moderate FLOPs cost.

---

## 34. SceneEdited: A City-Scale Benchmark for 3D HD Map Updating via Image-Guided Change Detection

**论文链接:** [http://arxiv.org/abs/2511.15153v1](http://arxiv.org/abs/2511.15153v1)

**作者:** Chun-Jung Lin, Tat-Jun Chin, Sourav Garg, Feras Dayoub

**发布时间:** 2025-11-19

**备注:** accepted by WACV 2026

### GPT解析

### 总结

本文介绍了SceneEdited数据集，这是首个专为高清地图维护研究设计的城市规模数据集，通过3D点云更新解决变化检测与3D地图更新之间的差距。

### 背景

高清地图对城市规划、基础设施监测和自动驾驶至关重要，但随环境变化很快过时。现有变化检测技术虽进步显著，但在检测变化与更新3D地图间存在差距，尤其基于2D图像的方法有限。

### 目的

解决变化检测与3D地图更新之间的技术差距，创建支持高清地图维护研究的数据集和工具包。

### 方法

构建SceneEdited数据集，包含800+最新场景(73公里驾驶路程，约3平方公里城市区域)，创建23,000+合成对象变化(手动和自动)，模拟真实城市修改。提供RGB图像、LiDAR扫描和变化掩码，以及基线方法和综合工具包。

### 主要发现

SceneEdited数据集填补了变化检测与3D地图更新间的技术空白，提供了训练、评估和未来扩展的基础设施。

### 结论

数据集和工具包已在GitHub公开，建立了3D地图更新研究的标准化基准，支持可扩展性、可追踪性和可移植性。

### 翻译

准确、最新式的高清地图对城市规划、基础设施监测和自动驾驶至关重要。然而，随着环境的变化，这些地图很快就会过时，需要强大的方法不仅能够检测变化，还能将它们整合到更新的三维表示中。尽管变化检测技术取得了显著进展，但在检测变化和实际更新三维地图之间仍然存在明显的差距，特别是在依赖基于二维图像的变化检测时。为了解决这一差距，我们引入了SceneEdited，这是第一个明确设计用于支持高清地图维护研究的三维点云更新的城市规模数据集。SceneEdited包含800多个最新场景，覆盖73公里的驾驶路程和约3平方公里的城市区域，在2000多个过时版本中创建了超过23,000个手动和自动合成的对象变化，模拟了缺失的路边基础设施、建筑物、天桥和电线杆等真实的城市修改。每个场景都包含校准的RGB图像、LiDAR扫描和详细的变化掩码，用于训练和评估。我们还提供了使用基础图像结构从运动流水线的基线方法来更新过时场景，以及一个支持可扩展性、可追踪性和可移植性的综合工具包，用于未来的数据集扩展和过时对象注释的统一。数据集和工具包可在https://github.com/ChadLin9596/ScenePoint-ETK公开获取，为三维地图更新研究建立了标准化基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何准确且高效地更新高清3D地图以适应不断变化的城市环境。这个问题非常重要，因为准确、最新的高清地图对自动驾驶、城市规划和基础设施监控至关重要；过时的地图会导致系统性能下降，带来安全风险；而目前依赖专业昂贵的测绘系统和耗时的手动工作，使得频繁重建不切实际。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有变化检测技术主要停留在定位变化阶段，而没有解决如何将变化整合到3D地图中；他们注意到点云地图层更新的支持非常有限。设计上借鉴了现有的高清地图构建技术、点云表示方法和变化检测技术，特别是使用了Argoverse数据集作为基础，并参考了图像到3D预测方法进行评估。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个城市规模基准数据集，通过图像引导的变化检测来更新3D点云地图，并提供包含最新和过时场景的数据集及变化掩码。整体流程包括：1)使用Argoverse数据准备静态点云；2)通过自动和手动编辑创建过时场景；3)生成精确的变化掩码；4)评估点添加和点删除方法；5)开发支持高效存储和扩展的工具包。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个城市规模的3D点云地图更新数据集；受控的场景变化合成；自动场景编辑工具包；标准化基准任务和指标。相比之前工作，SceneEdited专注于地图更新而非仅变化检测；同时提供2D图像、3D点云和变化掩码；覆盖更大城市区域(73公里轨迹，3平方公里)；包含更多样化的变化类型(23000+对象变化)；并提供支持扩展和统一的工具包。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SceneEdited引入了首个专门用于通过图像引导变化检测来更新城市规模3D高清地图的基准数据集和工具包，弥合了现有变化检测技术与实际地图更新应用之间的差距。'}


### 论文摘要

Accurate, up-to-date High-Definition (HD) maps are critical for urban planning, infrastructure monitoring, and autonomous navigation. However, these maps quickly become outdated as environments evolve, creating a need for robust methods that not only detect changes but also incorporate them into updated 3D representations. While change detection techniques have advanced significantly, there remains a clear gap between detecting changes and actually updating 3D maps, particularly when relying on 2D image-based change detection. To address this gap, we introduce SceneEdited, the first city-scale dataset explicitly designed to support research on HD map maintenance through 3D point cloud updating. SceneEdited contains over 800 up-to-date scenes covering 73 km of driving and approximate 3 $\text{km}^2$ of urban area, with more than 23,000 synthesized object changes created both manually and automatically across 2000+ out-of-date versions, simulating realistic urban modifications such as missing roadside infrastructure, buildings, overpasses, and utility poles. Each scene includes calibrated RGB images, LiDAR scans, and detailed change masks for training and evaluation. We also provide baseline methods using a foundational image-based structure-from-motion pipeline for updating outdated scenes, as well as a comprehensive toolkit supporting scalability, trackability, and portability for future dataset expansion and unification of out-of-date object annotations. Both the dataset and the toolkit are publicly available at https://github.com/ChadLin9596/ScenePoint-ETK, establising a standardized benchmark for 3D map updating research.

---

## 35. PAVE: An End-to-End Dataset for Production Autonomous Vehicle Evaluation

**论文链接:** [http://arxiv.org/abs/2511.14185v2](http://arxiv.org/abs/2511.14185v2)

**作者:** Xiangyu Li, Chen Wang, Yumao Liu, Dengbo He, Jiahao Zhang, Ke Ma

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了首个完全由自动驾驶模式收集的端到端基准数据集，用于评估黑盒控制下自动驾驶车辆的真实行为安全性。数据集包含超过100小时的自然驾驶数据，分割为32,727个关键帧，每个关键帧包含多视角图像、高精度定位数据、车辆轨迹和详细标注。数据集具有丰富的场景级属性，并持续每周扩展超过10小时新数据。所提出的端到端运动规划模型在自动驾驶帧上实现了1.4米的平均位移误差。

### 背景

现有的自动驾驶数据集（如KITTI、nuScenes和Waymo感知数据集）是通过人类驾驶模式或未识别驾驶模式收集的，只能作为自动驾驶车辆感知和预测的早期训练。

### 目的

评估黑盒控制下自动驾驶车辆的真实行为安全性，并创建首个完全由自动驾驶模式在真实世界收集的端到端基准数据集。

### 方法

收集市场上多个量产自动驾驶车型超过100小时的自然驾驶数据；将原始数据分割为32,727个关键帧，每个关键帧包含四个同步摄像头图像和高精度GNSS/IMU数据；提供过去6秒和未来5秒的20Hz车辆轨迹；提供周围车辆、行人、交通灯和交通标志的详细2D标注；使用端到端运动规划模型评估AVs的安全性；数据集每周新增超过10小时数据持续扩展。

### 主要发现

成功创建了首个完全由自动驾驶模式收集的端到端基准数据集；数据集包含丰富的场景级属性和精确的标注信息；所提出的端到端运动规划模型在自动驾驶帧上实现了1.4米的平均位移误差。

### 结论

PAVE数据集为自动驾驶车辆驾驶行为分析和安全评估研究提供了可持续的基础，并已公开可用。

### 翻译

大多数现有的自动驾驶数据集（如KITTI、nuScenes和Waymo感知数据集）通过人类驾驶模式或未识别驾驶模式收集，只能作为自动驾驶车辆感知和预测的早期训练。为了评估黑盒控制下自动驾驶车辆的真实行为安全性，我们提出了首个完全由自动驾驶模式在真实世界收集的端到端基准数据集。该数据集包含市场上多个量产自动驾驶车型超过100小时的自然驾驶数据。我们将原始数据分割为32,727个关键帧，每个关键帧由四个同步摄像头图像和高精度GNSS/IMU数据组成（0.8厘米定位精度）。对于每个关键帧，提供过去6秒和未来5秒的20Hz车辆轨迹，以及周围车辆、行人、交通灯和交通标志的详细2D标注。这些关键帧具有丰富的场景级属性，包括驾驶员意图、区域类型（涵盖高速公路、城市道路和住宅区）、光照（白天、夜间或黄昏）、天气（晴朗或雨天）、路面（铺砌或未铺砌）、交通和弱势道路使用者（VRU）密度、交通灯和交通标志（警告、禁止和指示）。为了评估AVs的安全性，我们采用端到端运动规划模型，在自动驾驶帧上的平均位移误差（ADE）为1.4米。该数据集每周新增超过10小时数据，持续扩展，从而为自动驾驶车辆驾驶行为分析和安全评估研究提供了可持续的基础。PAVE数据集可在https://hkustgz-my.sharepoint.com/:f:/g/personal/kema_hkust-gz_edu_cn/IgDXyoHKfdGnSZ3JbbidjduMAXxs-Z3NXzm005A_Ix9tr0Q?e=9HReCu公开获取。


### 论文摘要

Most existing autonomous-driving datasets (e.g., KITTI, nuScenes, and the Waymo Perception Dataset), collected by human-driving mode or unidentified driving mode, can only serve as early training for the perception and prediction of autonomous vehicles (AVs). To evaluate the real behavioral safety of AVs controlled in the black box, we present the first end-to-end benchmark dataset collected entirely by autonomous-driving mode in the real world. This dataset contains over 100 hours of naturalistic data from multiple production autonomous-driving vehicle models in the market. We segment the original data into 32,727 key frames, each consisting of four synchronized camera images and high-precision GNSS/IMU data (0.8 cm localization accuracy). For each key frame, 20 Hz vehicle trajectories spanning the past 6 s and future 5 s are provided, along with detailed 2D annotations of surrounding vehicles, pedestrians, traffic lights, and traffic signs. These key frames have rich scenario-level attributes, including driver intent, area type (covering highways, urban roads, and residential areas), lighting (day, night, or dusk), weather (clear or rain), road surface (paved or unpaved), traffic and vulnerable road users (VRU) density, traffic lights, and traffic signs (warning, prohibition, and indication). To evaluate the safety of AVs, we employ an end-to-end motion planning model that predicts vehicle trajectories with an Average Displacement Error (ADE) of 1.4 m on autonomous-driving frames. The dataset continues to expand by over 10 hours of new data weekly, thereby providing a sustainable foundation for research on AV driving behavior analysis and safety evaluation. The PAVE dataset is publicly available at https://hkustgz-my.sharepoint.com/:f:/g/personal/kema_hkust-gz_edu_cn/IgDXyoHKfdGnSZ3JbbidjduMAXxs-Z3NXzm005A_Ix9tr0Q?e=9HReCu.

---

## 36. Real-time Point Cloud Data Transmission via L4S for 5G-Edge-Assisted Robotics

**论文链接:** [http://arxiv.org/abs/2511.15677v1](http://arxiv.org/abs/2511.15677v1)

**作者:** Gerasimos Damigos, Achilleas Santi Seisa, Nikolaos Stathoulopoulos, Sara Sandberg, George Nikolakopoulos

**发布时间:** 2025-11-11

**备注:** IFAC Submission

### GPT解析

### 总结

本文提出了一种新型实时激光雷达数据传输框架，利用速率自适应技术和点云编码方法实现低延迟、低损耗的数据流传输。

### 背景

需要实时数据传输的机器人应用通常需要通过互联网进行卸载处理，这需要高效的传输方法处理高比特率的3D激光雷达数据。

### 目的

开发一种低延迟、低损耗、可扩展吞吐量的激光雷达数据传输系统，能够动态压缩高比特率3D激光雷达数据，同时保持最小端到端延迟，并限制编码误差以满足机器人应用的准确性要求。

### 方法

扩展了L4S支持的SCReAM v2传输框架，集成了Draco几何压缩算法，根据感知的信道容量和网络负载动态压缩3D激光雷达数据，并在公共5G网络上进行了多公里城市环境下的真实实验。

### 主要发现

通过真实世界实验验证了方法的有效性，在公共5G网络的多公里城市环境中保持了低延迟和低损耗要求，通过实时卸载和评估3D SLAM算法验证了框架在实际用例中的性能。

### 结论

该框架能够有效支持机器人应用中的实时激光雷达数据传输，满足低延迟、低损耗要求，同时保持数据准确性。

### 翻译

这篇文章提出了一种用于实时激光雷达数据传输的新框架，该框架利用速率自适应技术和点云编码方法来确保低延迟、低损耗的数据流传输。所提出的框架不仅限于需要通过互联网进行卸载处理以实现实时数据传输的机器人应用。具体而言，扩展了支持低延迟、低损耗、可扩展吞吐量的L4S SCReAM v2传输框架，集成了Draco几何压缩算法，能够根据感知的信道容量和网络负载动态压缩高比特率的3D激光雷达数据。该低延迟3D激光雷达流系统设计用于在保持最小端到端延迟的同时，限制编码误差以满足机器人应用的准确性要求。我们通过在公共5G网络上穿越多公里城市环境进行的真实世界实验，证明了所提出方法的有效性。在保持低延迟和低损耗要求的同时，通过实时卸载和评估3D SLAM算法，验证了该框架在实际用例中的性能。


### 论文摘要

This article presents a novel framework for real-time Light Detection and Ranging (LiDAR) data transmission that leverages rate-adaptive technologies and point cloud encoding methods to ensure low-latency, and low-loss data streaming. The proposed framework is intended for, but not limited to, robotic applications that require real-time data transmission over the internet for offloaded processing. Specifically, the Low Latency, Low Loss, Scalable Throughput L4S-enabled SCReAM v2 transmission framework is extended to incorporate the Draco geometry compression algorithm, enabling dynamic compression of high-bitrate 3D LiDAR data according to the sensed channel capacity and network load. The low-latency 3D LiDAR streaming system is designed to maintain minimal end-to-end delay while constraining encoding errors to meet the accuracy requirements of robotic applications. We demonstrate the effectiveness of the proposed method through real-world experiments conducted over a public 5G network across multi-kilometer urban environments. The low-latency and low-loss requirements are preserved, while real-time offloading and evaluation of 3D SLAM algorithms are used to validate the framework's performance in practical use cases.

---

## 37. IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics

**论文链接:** [http://arxiv.org/abs/2511.15004v1](http://arxiv.org/abs/2511.15004v1)

**作者:** Halil S. Kelebek, Linnea M. Wolniewicz, Michael D. Vergalla, Simone Mestici, Giacomo Acciarini, Bala Poduval, Olga Verkhoglyadova, Madhulika Guhathakurta, Thomas E. Berger, Frank Soboczenski, Atılım Güneş Baydin

**发布时间:** 2025-11-19

**备注:** 11 pages, 7 figures, 3 tables. Accepted as a poster presentation at the Machine Learning for the Physical Sciences Workshop at NeurIPS 2025

### GPT解析

### 总结

本文介绍了IonCast，一套专门用于电离层变化预测和建模的深度学习模型，利用时空学习预测全球总电子含量，整合多种物理驱动因素和观测数据集。

### 背景

电离层是近地空间的关键组成部分，影响全球导航卫星系统精度、高频通信和航空运营，因此电离层变化的准确预测和建模变得越来越重要。

### 目的

为了解决电离层预测和建模的挑战，作者提出了IonCast，这是一套专门针对电离层动力学的深度学习模型。

### 方法

IonCast利用时空学习预测全球总电子含量，整合多种物理驱动因素和观测数据集，采用可扩展的基于图的时空学习方法统一异构数据。

### 主要发现

在保留的风暴时间和平静条件下的验证表明，IonCast比持续性预测具有更好的技能，展示了机器学习对增强电离层变化物理理解的潜力。

### 结论

通过基于图的时空学习统一异构数据，IonCast展示了机器学习如何增强对电离层变化的物理理解，并推进空间天气的运营弹性。

### 翻译

电离层是近地空间的关键组成部分，它影响着全球导航卫星系统的准确性、高频通信和航空运营。由于这些原因，电离层变化的准确预测和建模变得越来越重要。为了解决这一差距，我们提出了IonCast，这是一套深度学习模型，其中包括一个专为电离层动力学设计的、受GraphCast启发的模型。IonCast利用时空学习来预测全球总电子含量(TEC)，整合了多种物理驱动因素和观测数据集。在保留的风暴时间和平静条件下的验证显示了相比持续性预测的改进技能。通过使用可扩展的基于图的时空学习来统一异构数据，IonCast展示了机器学习如何增强对电离层变化的物理理解，并推进空间天气的运营弹性。


### 论文摘要

The ionosphere is a critical component of near-Earth space, shaping GNSS accuracy, high-frequency communications, and aviation operations. For these reasons, accurate forecasting and modeling of ionospheric variability has become increasingly relevant. To address this gap, we present IonCast, a suite of deep learning models that include a GraphCast-inspired model tailored for ionospheric dynamics. IonCast leverages spatiotemporal learning to forecast global Total Electron Content (TEC), integrating diverse physical drivers and observational datasets. Validating on held-out storm-time and quiet conditions highlights improved skill compared to persistence. By unifying heterogeneous data with scalable graph-based spatiotemporal learning, IonCast demonstrates how machine learning can augment physical understanding of ionospheric variability and advance operational space weather resilience.

---

## 38. Multi-Stage Residual-Aware Unsupervised Deep Learning Framework for Consistent Ultrasound Strain Elastography

**论文链接:** [http://arxiv.org/abs/2511.15640v1](http://arxiv.org/abs/2511.15640v1)

**作者:** Shourov Joarder, Tushar Talukder Showrav, Md. Kamrul Hasan

**发布时间:** 2025-11-19

**备注:** 13 pages, 9 figures

### GPT解析

### 总结

本文提出了一种名为MUSSE-Net的新型深度学习框架，用于解决超声应变弹性成像在临床应用中的局限性，实现了稳健和一致的应变估计。

### 背景

超声应变弹性成像是一种评估组织机械特性的强大无创成像技术，具有高临床价值，但其应用受组织去相关噪声、真实标签稀缺以及不同变形条件下应变估计不一致等因素限制。

### 目的

开发一种能够克服现有超声应变弹性成像技术局限性，实现稳健和一致应变估计的创新方法。

### 方法

提出MUSSE-Net框架，核心是USSE-Net多流编码器-解码器架构，并行处理变形前后射频序列估计位移场和轴向应变。架构包含基于上下文感知互补特征融合的编码器、三交叉注意力瓶颈和交叉注意力融合的顺序解码器，采用定制一致性损失确保时间一致性和应变稳定性，并通过二次残差细化提高精度和抑制噪声。

### 主要发现

在模拟数据上，MUSSE-Net实现了最先进性能，目标信噪比24.54，背景信噪比132.76，对比度噪声比59.81，弹性信噪比9.73。在BUET数据集上，该方法产生了增强病变与背景对比度的应变图，显著抑制噪声，产生临床可解释的应变模式。

### 结论

MUSSE-Net通过创新的深度学习架构解决了超声应变弹性成像在临床应用中的关键挑战，提高了应变估计的准确性和一致性，为临床诊断提供了更可靠的工具。

### 翻译

超声应变弹性成像(USE)是一种评估组织机械特性的强大无创成像技术，在各种临床应用中具有重要诊断价值。然而，其临床应用受到组织去相关噪声、真实标签稀缺以及在不同变形条件下应变估计不一致等因素的限制。克服这些障碍，我们提出了MUSSE-Net，一个残差感知、多阶段无监督顺序深度学习框架，专为稳健和一致的应变估计而设计。其核心是我们提出的USSE-Net，一个端到端多流编码器-解码器架构，并行处理变形前后的射频序列以估计位移场和轴向应变。新颖的架构包含基于上下文感知互补特征融合(CACFF)的编码器、具有三交叉注意力(TCA)瓶颈的交叉注意力融合(CAF)顺序解码器。为确保不同变形水平下的时间一致性和应变稳定性，该架构利用了定制的一致性损失。最后，通过MUSSE-Net框架，二次残差细化阶段进一步提高了精度并抑制了噪声。在模拟、体内以及来自孟加拉国工程技术大学(BUET)医疗中心的私人临床数据集上的广泛验证表明，MUSSE-Net优于现有的无监督方法。在模拟数据上，MUSSE-Net实现了最先进的性能，目标信噪比为24.54，背景信噪比为132.76，对比度噪声比为59.81，弹性信噪比为9.73。特别是在BUET数据集上，MUSSE-Net产生了具有增强病变与背景对比度的应变图，并显著抑制了噪声，产生了临床可解释的应变模式。


### 论文摘要

Ultrasound Strain Elastography (USE) is a powerful non-invasive imaging technique for assessing tissue mechanical properties, offering crucial diagnostic value across diverse clinical applications. However, its clinical application remains limited by tissue decorrelation noise, scarcity of ground truth, and inconsistent strain estimation under different deformation conditions. Overcoming these barriers, we propose MUSSE-Net, a residual-aware, multi-stage unsupervised sequential deep learning framework designed for robust and consistent strain estimation. At its backbone lies our proposed USSE-Net, an end-to-end multi-stream encoder-decoder architecture that parallelly processes pre- and post-deformation RF sequences to estimate displacement fields and axial strains. The novel architecture incorporates Context-Aware Complementary Feature Fusion (CACFF)-based encoder with Tri-Cross Attention (TCA) bottleneck with a Cross-Attentive Fusion (CAF)-based sequential decoder. To ensure temporal coherence and strain stability across varying deformation levels, this architecture leverages a tailored consistency loss. Finally, with the MUSSE-Net framework, a secondary residual refinement stage further enhances accuracy and suppresses noise. Extensive validation on simulation, in vivo, and private clinical datasets from Bangladesh University of Engineering and Technology (BUET) medical center, demonstrates MUSSE-Net's outperformed existing unsupervised approaches. On MUSSE-Net achieves state-of-the-art performance with a target SNR of 24.54, background SNR of 132.76, CNR of 59.81, and elastographic SNR of 9.73 on simulation data. In particular, on the BUET dataset, MUSSE-Net produces strain maps with enhanced lesion-to-background contrast and significant noise suppression yielding clinically interpretable strain patterns.

---

## 39. Multimodal Optical Imaging Platform for Quantitative Burn Assessment

**论文链接:** [http://arxiv.org/abs/2511.15509v1](http://arxiv.org/abs/2511.15509v1)

**作者:** Nathaniel Hanson, Mateusz Wolak, Jonathan Richardson, Patrick Walker, David M. Burmeister, Chakameh Jafari

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种多模态光学成像框架，用于定量评估烧伤严重程度，解决了在受伤初期准确评估烧伤深度这一临床挑战。

### 背景

烧伤严重程度评估在受伤初期仍面临重大挑战，缺乏客观方法检测深层组织损伤。在战场和大规模伤亡情况下，快速可靠评估烧伤深度对分类和手术决策至关重要。

### 目的

开发一种紧凑、小型化、低功耗的可部署设备，用于定量烧伤评估，建立多模态光学成像框架。

### 方法

系统整合宽带高光谱成像（VSWIR，400-2100纳米）和激光散斑对比成像，联合评估生化成分和微血管灌注；利用短波红外波长开发与水、脂质和胶原蛋白吸收特征相关的新型深层组织参数；实现并验证无监督学习方法进行光谱特征提取、波段降选择和聚类。

### 主要发现

新型参数提高了烧伤组织区分度和烧伤严重程度分类能力；无监督学习方法与组织学结果验证一致。

### 结论

为在恶劣环境中进行早期定量烧伤评估的坚固数据驱动设备奠定了基础。

### 翻译

准确的烧伤严重程度评估在受伤初期仍然是一个重大临床挑战，因为缺乏检测深层组织损伤的客观方法。这一限制在战场和大规模伤亡情况下尤为关键，因为快速可靠地评估烧伤深度对分类和手术决策至关重要。我们提出了一种多模态光学成像框架，为紧凑、小型化、低功耗（low-SWaP）的可部署设备用于定量烧伤评估奠定了基础。该系统整合了宽带高光谱成像（VSWIR，400-2100纳米）和激光散斑对比成像，联合评估生化成分和微血管灌注。利用短波红外（SWIR，>1000纳米）波长，我们开发并验证了与水、脂质和胶原蛋白吸收特征相关的新型深层组织参数，这些参数提高了烧伤组织区分度和烧伤严重程度分类能力。我们实现并验证了无监督学习方法用于光谱特征提取、波段降选择和聚类，与组织学结果一致，为在恶劣环境中进行早期定量烧伤评估的坚固数据驱动设备奠定了基础。


### 论文摘要

Accurate assessment of burn severity at injury onset remains a major clinical challenge due to the lack of objective methods for detecting subsurface tissue damage. This limitation is critical in battlefield and mass-casualty settings, where rapid and reliable evaluation of burn depth is essential for triage and surgical decision-making. We present a multimodal optical imaging framework that establishes the foundation for a compact, low-size, weight, and power (low-SWaP) field-deployable device for quantitative burn assessment. The system integrates broadband hyperspectral imaging (VSWIR, 400 -- 2100 nm) with laser speckle contrast imaging to jointly evaluate biochemical composition and microvascular perfusion. Using short-wave infrared (SWIR, >1000 nm) wavelengths, we developed and validated novel deep-tissue parameters linked to water, lipid, and collagen absorption features that enhance burn-tissue separability and burn severity classification. We implemented and validated unsupervised learning methods for spectral feature extraction, band down-selection, and clustering against histology, establishing a foundation for a rugged, data-driven device for early quantitative burn evaluation in austere environments.

---

## 40. D4C: Data-free Quantization for Contrastive Language-Image Pre-training Models

**论文链接:** [http://arxiv.org/abs/2511.15411v1](http://arxiv.org/abs/2511.15411v1)

**作者:** Wenlun Zhang, Yunshan Zhong, Zihao Ding, Xinyu Li, Kentaro Yoshioka

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究提出了D4C框架，首个专门针对视觉-语言模型CLIP的无数据量化方法，通过三个关键组件解决了现有DFQ技术在CLIP上应用时的性能退化问题。

### 背景

Data-Free Quantization (DFQ)是一种无需访问真实数据的模型压缩方法，在隐私敏感场景中具有吸引力，但在视觉-语言模型如CLIP上的应用研究不足。

### 目的

开发专门针对CLIP的DFQ框架，解决直接应用现有DFQ技术导致的性能退化问题。

### 方法

提出D4C框架，包含三个关键组件：(1)提示引导的语义注入，使用文本提示使生成图像与真实世界语义对齐；(2)结构对比生成，利用前景-背景对比合成重现自然图像结构；(3)感知增强的增强，应用受控扰动提高样本多样性和鲁棒性。

### 主要发现

D4C能够合成语义信息丰富且结构多样的图像，有效弥合了DFQ在CLIP上的性能差距，在各种位宽和模型上显示出显著的性能改进。

### 结论

通过广泛实验验证了D4C的有效性，例如在W4A8设置下，使用CLIP ResNet-50和ViT-B/32模型，在零样本分类任务中CIFAR-10、CIFAR-100和ImageNet-1K上的Top-1准确率分别提高了12.4%/18.9%、6.8%/19.7%和1.4%/5.7%。

### 翻译

无数据量化(DFQ)提供了一种无需访问真实数据的模型压缩实用解决方案，在隐私敏感场景中特别有吸引力。虽然DFQ在单模态模型上显示出前景，但其扩展到视觉-语言模型(如对比语言-图像预训练(CLIP)模型)的研究仍然不足。在本工作中，我们揭示直接将现有DFQ技术应用于CLIP会导致性能显著下降，这是由于两个关键限制：合成样本的语义内容不足和图像内多样性低。为解决这些挑战，我们提出了D4C，首个专门针对CLIP的DFQ框架。D4C通过三个关键组件合成语义丰富且结构多样的伪图像：(1)提示引导的语义注入使用文本提示将生成的图像与真实世界语义对齐；(2)结构对比生成利用前景-背景对比合成重现自然图像的组成结构；(3)感知增强的增强应用受控扰动提高样本多样性和鲁棒性。这些组件共同使D4C能够合成语义信息丰富且结构多样的图像，有效弥合了DFQ在CLIP上的性能差距。大量实验验证了D4C的有效性，显示出在各种位宽和模型上的显著性能改进。例如，在W4A8设置下使用CLIP ResNet-50和ViT-B/32模型，在零样本分类中，CIFAR-10上的Top-1准确率分别提高了12.4%和18.9%，CIFAR-100上提高了6.8%和19.7%，ImageNet-1K上提高了1.4%和5.7%。


### 论文摘要

Data-Free Quantization (DFQ) offers a practical solution for model compression without requiring access to real data, making it particularly attractive in privacy-sensitive scenarios. While DFQ has shown promise for unimodal models, its extension to Vision-Language Models such as Contrastive Language-Image Pre-training (CLIP) models remains underexplored. In this work, we reveal that directly applying existing DFQ techniques to CLIP results in substantial performance degradation due to two key limitations: insufficient semantic content and low intra-image diversity in synthesized samples. To tackle these challenges, we propose D4C, the first DFQ framework tailored for CLIP. D4C synthesizes semantically rich and structurally diverse pseudo images through three key components: (1) Prompt-Guided Semantic Injection aligns generated images with real-world semantics using text prompts; (2) Structural Contrastive Generation reproduces compositional structures of natural images by leveraging foreground-background contrastive synthesis; and (3) Perturbation-Aware Enhancement applies controlled perturbations to improve sample diversity and robustness. These components jointly empower D4C to synthesize images that are both semantically informative and structurally diverse, effectively bridging the performance gap of DFQ on CLIP. Extensive experiments validate the effectiveness of D4C, showing significant performance improvements on various bit-widths and models. For example, under the W4A8 setting with CLIP ResNet-50 and ViT-B/32, D4C achieves Top-1 accuracy improvement of 12.4% and 18.9% on CIFAR-10, 6.8% and 19.7% on CIFAR-100, and 1.4% and 5.7% on ImageNet-1K in zero-shot classification, respectively.

---

## 41. Learning Depth from Past Selves: Self-Evolution Contrast for Robust Depth Estimation

**论文链接:** [http://arxiv.org/abs/2511.15167v1](http://arxiv.org/abs/2511.15167v1)

**作者:** Jing Cao, Kui Jiang, Shenyi Li, Xiaocheng Feng, Yong Huang

**发布时间:** 2025-11-19

### GPT解析

### 总结

论文提出了SEC-Depth自进化对比学习框架，用于解决自监督深度估计在恶劣天气条件下性能下降的问题。

### 背景

自监督深度估计在自动驾驶和机器人领域受到广泛关注，但现有方法在雨、雾等恶劣天气条件下性能显著下降，能见度降低严重影响深度预测。

### 目的

提出一种新的自进化对比学习框架SEC-Depth，用于自监督鲁棒深度估计任务，以解决恶劣天气条件下性能下降的问题。

### 方法

利用训练过程中生成的中间参数构建时间演化的延迟模型，设计自进化对比方案减轻困难条件下的性能损失；设计延迟模型动态更新策略捕获训练阶段优化状态；引入自进化对比损失(SECL)将历史延迟模型输出视为负样本，自适应调整学习目标并隐式感知天气退化严重程度。

### 主要发现

SEC-Depth方法可以无缝集成到各种基线模型中，在零样本评估中显著提高了鲁棒性。

### 结论

SEC-Depth框架能有效解决恶劣天气条件下自监督深度估计性能下降的问题，减少人工干预需求。

### 翻译

自监督深度估计在自动驾驶和机器人领域已获得显著关注。然而，现有方法在雨、雾等恶劣天气条件下表现出严重的性能下降，在这些情况下能见度降低严重影响了深度预测。为解决这一问题，我们提出了一个名为SEC-Depth的自进化对比学习框架，用于自监督鲁棒深度估计任务。我们的方法利用训练过程中生成的中间参数构建时间演化的延迟模型。利用这些模型，我们设计了一个自进化对比方案来减轻困难条件下的性能损失。具体而言，我们首先为深度估计任务设计了延迟模型的动态更新策略，以捕获训练阶段的优化状态。为有效利用延迟模型，我们引入了自进化对比损失(SECL)，将历史延迟模型的输出视为负样本。这种机制自适应地调整学习目标，同时隐式感知天气退化严重程度，减少了对人工干预的需求。实验表明，我们的方法可以无缝集成到各种基线模型中，并在零样本评估中显著增强了鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自监督深度估计方法在恶劣天气条件下（如雨、雾、雪）性能显著下降的问题。这个问题在现实中非常重要，因为深度估计是自动驾驶和机器人等应用的核心技术，而这些系统需要在各种天气条件下可靠工作。恶劣天气会降低能见度，破坏传统方法依赖的光度一致性假设，导致深度预测不准确，限制了深度估计技术的实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法在恶劣天气条件下失效的原因，然后受图像恢复方法的启发，发现模型在不同训练阶段的中间状态（延迟模型）包含有价值的信息。他们利用这些历史模型生成负样本，并通过对比学习增强模型鲁棒性。方法借鉴了对比学习的基本思想，但创新性地使用模型自身的历史参数而非外部数据构建负样本，避免了现有知识蒸馏方法的局限性，也解决了传统对比学习方法可能导致的解决方案崩溃问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用模型训练过程中的历史参数（延迟模型）作为负样本，通过对比当前模型与历史模型的输出来增强对恶劣天气的鲁棒性。整体流程包括：1)初始化并维护一个历史模型队列；2)在清晰图像上进行基础自监督训练；3)定期注入恶劣天气样本并计算对比损失；4)将深度图离散化为概率分布以处理局部不一致；5)使用自我演化对比损失比较锚点、正例和负例样本；6)根据总损失更新模型参数并定期更新历史模型队列。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)自我演化对比框架，首次利用模型历史参数构建负样本；2)动态延迟模型更新策略，确保负样本队列的时效性；3)基于区间的深度分布约束，将连续深度值离散化为概率分布；4)自我演化对比损失，动态调整学习目标。相比之前工作，SEC-Depth无需预设数据集或复杂课程学习策略，不依赖外部教师模型，避免了传统对比学习方法可能导致的解决方案崩溃问题，且是一个即插即用的框架，可无缝集成到各种基线模型中。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SEC-Depth通过利用模型训练过程中的历史参数构建自我演化对比学习框架，显著提高了自监督深度估计在恶劣天气条件下的鲁棒性，且无需预设数据集或修改模型架构。'}


### 论文摘要

Self-supervised depth estimation has gained significant attention in autonomous driving and robotics. However, existing methods exhibit substantial performance degradation under adverse weather conditions such as rain and fog, where reduced visibility critically impairs depth prediction. To address this issue, we propose a novel self-evolution contrastive learning framework called SEC-Depth for self-supervised robust depth estimation tasks. Our approach leverages intermediate parameters generated during training to construct temporally evolving latency models. Using these, we design a self-evolution contrastive scheme to mitigate performance loss under challenging conditions. Concretely, we first design a dynamic update strategy of latency models for the depth estimation task to capture optimization states across training stages. To effectively leverage latency models, we introduce a self-evolution contrastive Loss (SECL) that treats outputs from historical latency models as negative samples. This mechanism adaptively adjusts learning objectives while implicitly sensing weather degradation severity, reducing the needs for manual intervention. Experiments show that our method integrates seamlessly into diverse baseline models and significantly enhances robustness in zero-shot evaluations.

---

## 42. Deep Learning Assisted Prediction of Electrochemical Lithiation State in Spinel Lithium Titanium Oxide Thin Films

**论文链接:** [http://arxiv.org/abs/2511.15109v1](http://arxiv.org/abs/2511.15109v1)

**作者:** Devin Chugh, Bhagath Sreenarayanan, Steven Suwito, Ganesh Raghavendran, Bing Joe Hwang, Ying Shirley Meng, Weinien Su

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究提出了一种基于机器学习和深度学习的框架，用于预测尖晶石Li4Ti5O12薄膜的电化学锂化状态和相关电导率，使用拉曼光谱数据作为输入，并比较了传统机器学习模型与卷积神经网络的性能。

### 背景

机器学习和深度学习框架在预测材料特性方面发展迅速并引起广泛关注。拉曼光谱因其快速、非破坏性和高分辨率的特点，适合用于监测材料中的动态电化学变化。

### 目的

利用机器学习和深度学习框架，基于拉曼光谱数据预测尖晶石Li4Ti5O12薄膜的电化学锂化状态和相关电导率。

### 方法

收集并预处理了3,272个代表0%到100%锂化状态的拉曼光谱数据集，使用宇宙射线去除、平滑、基线校正、归一化和数据增强等技术进行预处理；评估了支持向量机、线性判别分析、随机森林等传统机器学习模型以及卷积神经网络的性能。

### 主要发现

传统机器学习模型达到了中等至高准确率，但在泛化能力和噪声敏感性方面存在不足；CNN模型表现出卓越性能，准确率超过99.5%，能有效捕获非线性光谱特征，并对实验变化具有韧性；该管道能同时实现锂化状态分类和电导率估计。

### 结论

该研究提供了一种可扩展的方法用于实时电池材料表征，并有望扩展到其他光谱数据集应用中。

### 翻译

机器学习（ML）和深度学习（DL）框架已经迅速发展，并在预测材料特性方面引起了相当大的兴趣。在本工作中，我们利用ML-DL框架，使用拉曼光谱数据来预测尖晶石Li4Ti5O12（LTO）薄膜的电化学锂化状态及相关电导率。拉曼光谱凭借其快速、非破坏性和高分辨率的能力，被用来监测LTO薄膜中的动态电化学变化。我们收集并预处理了代表0%到100%锂化状态的3,272个拉曼光谱数据集，使用了包括宇宙射线去除、平滑、基线校正、归一化和数据增强在内的先进技术。我们评估了支持向量机（SVM）、线性判别分析（LDA）和随机森林（RF）等传统机器学习模型，以及卷积神经网络（CNN）。虽然传统模型达到了中等至高准确率，但它们在泛化能力和噪声敏感性方面存在问题。相比之下，CNN表现出卓越的性能，准确率超过99.5%，并对未见样本做出稳健预测。CNN模型有效捕获了非线性光谱特征，并显示出对实验变化的韧性。该管道不仅能实现准确的锂化状态分类，还有助于电导率估计，为实时电池材料表征提供了一种可扩展的方法，并可能扩展到其他光谱数据集。


### 论文摘要

Machine Learning (ML) and Deep Learning (DL) based framework have evolved rapidly and generated considerable interests for predicting the properties of materials. In this work, we utilize ML-DL framework to predict the electrochemical lithiation state and associated electrical conductivity of spinel Li4Ti5O12 (LTO) thin films using Raman spectroscopy data. Raman spectroscopy, with its rapid, non-destructive, and high-resolution capabilities, is leveraged to monitor dynamic electrochemical changes in LTO films. A comprehensive dataset of 3,272 Raman spectra, representing lithiation states from 0% to 100%, was collected and preprocessed using advanced techniques including cosmic ray removal, smoothing, baseline correction, normalization, and data augmentation. Classical machine learning models such as Support Vector Machine (SVM), Linear Discriminant Analysis (LDA), and Random Forest (RF) were evaluated alongside a Convolutional Neural Network (CNN). While traditional models achieved moderate to high accuracy, they struggled with generalization and noise sensitivity. In contrast, the CNN demonstrated superior performance, achieving over 99.5% accuracy and robust predictions on unseen samples. The CNN model effectively captured non-linear spectral features and showed resilience to experimental variability. This pipeline not only enables accurate lithiation state classification but also facilitates conductivity estimation, offering a scalable approach for real-time battery material characterization and potential extension to other spectroscopic datasets.

---

## 43. The Sequential Nature of Science: Quantifying Learning from a Sequence of Studies

**论文链接:** [http://arxiv.org/abs/2511.14996v1](http://arxiv.org/abs/2511.14996v1)

**作者:** Jonas M. Mikhaeil, Donald P. Green, David Blei

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种称为顺序元分析研究轨迹(SMART)的新方法，用于量化每个研究在进入文献时的影响力，与传统元分析相比能更好地捕捉新研究如何影响先前信念和增加集体不确定性。

### 背景

科学进步本质上是顺序性的，集体知识随着新研究进入文献而不断更新。

### 目的

开发一种能够量化每个研究在进入文献时影响力的方法，捕捉新研究对先前信念的质疑作用。

### 方法

提出顺序元分析研究轨迹(SMART)方法，与传统元分析对比，重新分析心理学和劳动经济学的两个元分析数据集，其中一个使用单一方法学，另一个包含在重要方法创新前后进行的研究。

### 主要发现

新研究可能对先前工作提出方法学批判并提出更优方法；即使小型研究在回顾性元分析中影响不大，但在出现时可能具有重要影响力；方法创新的重要性可能被传统元分析所忽视。

### 结论

顺序学习的形式化突显了方法创新的重要性，SMART方法能够更好地捕捉科学知识随时间的演变过程。

### 翻译

科学进步本质上是顺序性的：随着新研究进入文献，集体知识得到更新。我们提出了顺序元分析研究轨迹(SMART)，它量化了每个研究在进入文献时的影响力。与传统元分析相比，我们的方法可以捕捉新研究如何对先前持有的信念提出质疑，增加集体不确定性。例如，新研究可能对先前的工作提出方法学批判并提出更优方法。即使是小型研究，在回顾性元分析中可能不会产生实质影响，但在出现时可能是有影响力的。为了将SMART与传统元分析进行对比，我们重新分析了心理学和劳动经济学的两个元分析数据集。一个数据集使用单一方法学组装研究；另一个数据集包含在重要方法创新之前或之后的研究。我们对顺序学习的形式化突显了方法创新的重要性，而这种创新可能会被传统元分析所忽视。


### 论文摘要

Scientific progress is inherently sequential: collective knowledge is updated as new studies enter the literature. We propose the sequential meta-analysis research trace (SMART), which quantifies the influence of each study at the time it enters the literature. In contrast to classical meta-analysis, our method can capture how new studies may cast doubt on previously held beliefs, increasing collective uncertainty. For example, a new study may present a methodological critique of prior work and propose a superior method. Even small studies, which may not materially affect a retrospective meta-analysis, can be influential at the time they appeared. To contrast SMART with classical meta-analysis, we re-analyze two meta-analysis datasets, from psychology and labor economics. One assembles studies using a single methodology; the other contains studies that predate or follow an important methodological innovation. Our formalization of sequential learning highlights the importance of methodological innovation that might otherwise be overlooked by classical meta-analysis.

---

## 44. Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation

**论文链接:** [http://arxiv.org/abs/2511.14993v1](http://arxiv.org/abs/2511.14993v1)

**作者:** Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Viacheslav Vasilev, Alexey Letunovskiy, Maria Kovaleva, Nikolai Vaulin, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Nikita Kiselev, Alexander Varlamov, Dmitrii Mikhailov, Vladimir Polovnikov, Andrey Shutkin, Ilya Vasiliev, Julia Agafonova, Anastasiia Kargapoltseva, Anna Dmitrienko, Anastasia Maltseva, Anna Averchenkova, Olga Kim, Tatiana Nikulina, Denis Dimitrov

**发布时间:** 2025-11-19

**备注:** Website: https://kandinskylab.ai/

### GPT解析

### 总结

本报告介绍了Kandinsky 5.0，一个用于高分辨率图像和10秒视频合成的前沿基础模型家族，包含三种核心模型系列，并详细描述了其数据策划流程、优化技术和应用前景。

### 背景

生成式AI模型在图像和视频合成领域不断发展，需要更高质量、更高效的模型来满足多样化的应用需求。

### 目的

介绍Kandinsky 5.0模型家族，展示其架构、训练方法和性能优势，并推动高质量生成模型在研究社区的发展与可及性。

### 方法

采用多阶段训练流程，包括数据收集、处理、过滤和聚类，结合大量预训练以及质量增强技术如自监督微调(SFT)和基于强化学习(RL)的后训练，同时进行架构、训练和推理优化。

### 主要发现

Kandinsky 5.0包含三种模型系列：Image Lite(60亿参数图像生成模型)、Video Lite(20亿参数快速轻量视频模型)和Video Pro(190亿参数高质量视频模型)，实现了高生成速度和跨任务的最先进性能。

### 结论

作为大规模公开可用的生成框架，Kandinsky 5.0可适应各种生成应用，其开源代码和训练检查点的发布将促进高质量生成模型的发展。

### 翻译

本报告介绍了Kandinsky 5.0，一个用于高分辨率图像和10秒视频合成的前沿基础模型家族。该框架包含三种核心模型系列：Kandinsky 5.0 Image Lite - 60亿参数的图像生成模型系列；Kandinsky 5.0 Video Lite - 快速轻量的20亿参数文本到视频和图像到视频模型；以及Kandinsky 5.0 Video Pro - 190亿参数模型，实现卓越的视频生成质量。我们提供了数据策划生命周期的全面回顾，包括收集、处理、过滤和聚类，用于多阶段训练流程，该流程涉及大量预训练并采用质量增强技术，如自监督微调(SFT)和基于强化学习(RL)的后训练。我们还展示了新颖的架构、训练和推理优化，使Kandinsky 5.0能够实现高生成速度和跨各种任务的最先进性能，如人类评估所示。作为大规模、公开可用的生成框架，Kandinsky 5.0利用其预训练和后续阶段的全部潜力，可适应各种生成应用。我们希望本报告以及开源代码和训练检查点的发布将显著推进高质量生成模型的发展和可及性，为研究社区提供支持。


### 论文摘要

This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community.

---

## 45. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness

**论文链接:** [http://arxiv.org/abs/2511.14989v1](http://arxiv.org/abs/2511.14989v1)

**作者:** Saeefa Rubaiyet Nowmi, Jesus Lopez, Md Mahmudul Alam Imon, Shahrooz Pouryouse, Mohammad Saidur Rahman

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文对量子机器学习中的对抗鲁棒性进行了系统化研究，评估了三种威胁模型下的攻击效果，并探讨了不同编码方案在噪声环境下的表现。

### 背景

量子机器学习将量子计算原理整合到学习算法中，提供了更好的表示能力和计算效率，但其安全性和鲁棒性在对抗条件下尚未得到充分探索。

### 目的

对量子机器学习中的对抗鲁棒性进行系统化整理，整合概念组织和三种威胁模型(黑盒、灰盒和白盒)的实证评估。

### 方法

在三种威胁模型中实现代表性攻击(黑盒的标签翻转、灰盒的QUID编码器级数据投毒、白盒的FGSM和PGD)，使用在MNIST和AZ-Class数据集上训练的量子神经网络，测试了多个电路深度(2、5、10和50层)和两种编码方案(角度和幅度)。

### 主要发现

幅度编码在深度无噪声电路中准确率最高(MNIST上93%，AZ-Class上67%)，但在对抗性扰动下急剧下降至5%以下；角度编码表示能力较低但更稳定；QUID攻击成功率较高但受量子噪声影响；噪声可作为NISQ系统的自然防御机制。

### 结论

研究结果指导了安全弹性QML架构的开发，强调了设计在真实噪声下保持可靠性的威胁感知模型的重要性。

### 翻译

量子机器学习将量子计算原理整合到学习算法中，提供了更好的表示能力和计算效率。然而，QML系统的安全性和鲁棒性尚未得到充分探索，特别是在对抗条件下。本文对QML中的对抗鲁棒性进行了系统化整理，整合了概念组织和三种威胁模型(黑盒、灰盒和白盒)的实证评估。我们在每个类别中实现了代表性攻击，包括黑盒的标签翻转、灰盒的QUID编码器级数据投毒、白盒的FGSM和PGD，使用在两个不同领域数据集上训练的量子神经网络：计算机视觉的MNIST和Android恶意软件的AZ-Class，测试了多个电路深度和两种编码方案。我们的评估显示，幅度编码在深度、无噪声电路中提供了最高的干净准确率，但在对抗性扰动下急剧下降；相比之下，角度编码虽然表示能力较低，但在浅层、有噪声的环境中保持更稳定。此外，QUID攻击获得了更高的攻击成功率，但量子噪声会削弱其影响。这表明噪声可以在NISQ系统中充当自然防御机制。总体而言，我们的研究结果指导了安全、弹性QML架构的开发，强调了设计在NISQ设置中真实噪声下保持可靠性的威胁感知模型的重要性。


### 论文摘要

Quantum Machine Learning (QML) integrates quantum computational principles into learning algorithms, offering improved representational capacity and computational efficiency. Nevertheless, the security and robustness of QML systems remain underexplored, especially under adversarial conditions. In this paper, we present a systematization of adversarial robustness in QML, integrating conceptual organization with empirical evaluation across three threat models-black-box, gray-box, and white-box. We implement representative attacks in each category, including label-flipping for black-box, QUID encoder-level data poisoning for gray-box, and FGSM and PGD for white-box, using Quantum Neural Networks (QNNs) trained on two datasets from distinct domains: MNIST from computer vision and AZ-Class from Android malware, across multiple circuit depths (2, 5, 10, and 50 layers) and two encoding schemes (angle and amplitude). Our evaluation shows that amplitude encoding yields the highest clean accuracy (93% on MNIST and 67% on AZ-Class) in deep, noiseless circuits; however, it degrades sharply under adversarial perturbations and depolarization noise (p=0.01), dropping accuracy below 5%. In contrast, angle encoding, while offering lower representational capacity, remains more stable in shallow, noisy regimes, revealing a trade-off between capacity and robustness. Moreover, the QUID attack attains higher attack success rates, though quantum noise channels disrupt the Hilbert-space correlations it exploits, weakening its impact in image domains. This suggests that noise can act as a natural defense mechanism in Noisy Intermediate-Scale Quantum (NISQ) systems. Overall, our findings guide the development of secure and resilient QML architectures for practical deployment. These insights underscore the importance of designing threat-aware models that remain reliable under real-world noise in NISQ settings.

---

## 46. LFreeDA: Label-Free Drift Adaptation for Windows Malware Detection

**论文链接:** [http://arxiv.org/abs/2511.14963v1](http://arxiv.org/abs/2511.14963v1)

**作者:** Adrian Shuai Li, Elisa Bertino

**发布时间:** 2025-11-18

### GPT解析

### 总结

LFreeDA是一种端到端框架，通过结合图像和CFG表示的优势，实现了无需标记的恶意软件检测器适应，显著提高了性能，接近完全监督方法的水平。

### 背景

基于机器学习的恶意软件检测器会随着时间推移而性能下降，原因是概念漂移引入了新的且不断演变的恶意软件家族。重新训练受到手动标记或沙箱分析成本和时间的限制。现有方法通过漂移检测和选择性标记来缓解这个问题，但完全无标记的适应仍然 largely未被探索。

### 目的

提出一个无需手动标记或漂移检测的框架，使恶意软件分类器能够适应漂移。

### 方法

LFreeDA框架首先在恶意软件图像上执行无监督域适应，联合训练标记和未标记样本以推断伪标签并修剪噪声标签。然后在CFG表示上使用标记和选择的伪标记数据来适应分类器，利用图像的可扩展性进行伪标记，利用CFG的更丰富语义进行最终适应。

### 主要发现

在真实世界的MB-24+数据集上评估显示，LFreeDA将准确率提高了高达12.6%，F1提高了11.1%，相比无适应基线。在准确率和F1上分别仅比完全监督上限低4%和3.4%。与为300个目标样本提供真实标签的最先进方法性能相当。在两个受控漂移基准上的额外结果进一步确认LFreeDA能够在无需人工标记的情况下维持恶意软件检测性能。

### 结论

LFreeDA是一种有效的端到端框架，能够适应恶意软件分类器的漂移，而无需手动标记或漂移检测。

### 翻译

基于机器学习(ML)的恶意软件检测器会随着时间推移而性能下降，因为概念漂移引入了训练过程中未见的新兴且不断演变的恶意软件家族。重新训练受到手动标记或沙箱分析成本和时间的限制。现有方法通过漂移检测和选择性标记来缓解这一问题，但完全无标记的适应仍然 largely未被探索。最近的自我训练方法使用先前训练的模型为未标记数据生成伪标签，然后在这些标签上训练新模型。未标记数据仅用于推理，不参与训练早期模型。我们认为，当适当地纳入训练时，这些未标记样本仍然携带有价值的信息。本文介绍了LFreeDA，一个端到端框架，无需手动标记或漂移检测即可使恶意软件分类器适应漂移。LFreeDA首先在恶意软件图像上执行无监督域适应，联合训练标记和未标记样本以推断伪标签并修剪噪声标签。然后使用标记和选择的伪标记数据在CFG表示上适应分类器，利用图像的可扩展性进行伪标记，利用CFG的更丰富语义进行最终适应。在真实世界的MB-24+数据集上的评估显示，LFreeDA将准确率提高了高达12.6%，F1提高了11.1%，相比无适应基线，在准确率和F1上分别仅比完全监督上限低4%和3.4%。它还与为300个目标样本提供真实标签的最先进方法性能相当。在两个受控漂移基准上的额外结果进一步确认，LFreeDA能够在无需人工标记的情况下维持恶意软件检测性能，随着恶意软件的演变。


### 论文摘要

Machine learning (ML)-based malware detectors degrade over time as concept drift introduces new and evolving families unseen during training. Retraining is limited by the cost and time of manual labeling or sandbox analysis. Existing approaches mitigate this via drift detection and selective labeling, but fully label-free adaptation remains largely unexplored. Recent self-training methods use a previously trained model to generate pseudo-labels for unlabeled data and then train a new model on these labels. The unlabeled data are used only for inference and do not participate in training the earlier model. We argue that these unlabeled samples still carry valuable information that can be leveraged when incorporated appropriately into training. This paper introduces LFreeDA, an end-to-end framework that adapts malware classifiers to drift without manual labeling or drift detection. LFreeDA first performs unsupervised domain adaptation on malware images, jointly training on labeled and unlabeled samples to infer pseudo-labels and prune noisy ones. It then adapts a classifier on CFG representations using the labeled and selected pseudo-labeled data, leveraging the scalability of images for pseudo-labeling and the richer semantics of CFGs for final adaptation. Evaluations on the real-world MB-24+ dataset show that LFreeDA improves accuracy by up to 12.6% and F1 by 11.1% over no-adaptation lower bounds, and is only 4% and 3.4% below fully supervised upper bounds in accuracy and F1, respectively. It also matches the performance of state-of-the-art methods provided with ground truth labels for 300 target samples. Additional results on two controlled-drift benchmarks further confirm that LFreeDA maintains malware detection performance as malware evolves without human labeling.

---

## 47. Structured Contrastive Learning for Interpretable Latent Representations

**论文链接:** [http://arxiv.org/abs/2511.14920v1](http://arxiv.org/abs/2511.14920v1)

**作者:** Zhengyang Shen, Hua Tu, Mayue Shi

**发布时间:** 2025-11-18

**备注:** Comments: 10 pages, 6 figures. Applications to medical signal retrieval and activity recognition. Correspondence: m.shi16@imperial.ac.uk

### GPT解析

### 总结

本研究提出了一种结构化对比学习(SCL)框架，解决了神经网络对语义无关转换的脆弱性问题，通过将潜在空间分为不变特征、变体特征和自由特征三类，实现了同时提高鲁棒性和可解释性的目标。

### 背景

神经网络对语义上无关的转换表现出严重的脆弱性，例如75毫秒的心电图相位偏移会使潜在余弦相似度从1.0下降到0.2，传感器旋转会使使用惯性测量单元的活动识别性能崩溃。

### 目的

解决神经网络在'自由放任'表示学习下，潜在空间无约束演化导致的脆弱性问题，实现同时提高模型鲁棒性和可解释性的目标。

### 方法

提出结构化对比学习(SCL)框架，将潜在空间表示分为三类：不变特征(在给定转换下保持一致)、变体特征(通过新颖机制主动区分转换)和自由特征(保持任务灵活性)。该方法无需架构修改，可无缝集成到现有训练管道中。

### 主要发现

ECG相位不变性实验显示相似度从0.25提高到0.91；WISDM活动识别实现86.65%的准确率和95.38%的旋转一致性，性能一致优于传统数据增强方法。

### 结论

这项工作代表了从反应性数据增强到主动结构学习的范式转变，使神经网络中潜在表示具有可解释性，同时提高了对变换的鲁棒性。

### 翻译

神经网络对语义上无关的转换表现出严重的脆弱性。仅75毫秒的心电图(ECG)相位偏移就会使潜在余弦相似度从1.0降至0.2，而传感器旋转会使使用惯性测量单元(IMU)的活动识别性能崩溃。我们确定根本原因是'自由放任'的表示学习，其中潜在空间在满足任务性能的条件下无约束地演化。我们提出了结构化对比学习(SCL)，这是一个将潜在空间表示分为三个语义组的框架：不变特征在给定转换下保持一致(如相位偏移或旋转)，变体特征通过新颖的变体机制主动区分转换，以及自由特征保持任务灵活性。这创造了可控的推拉动态，使不同的潜在维度服务于不同的、可解释的目的。变体机制通过鼓励变体特征在正样本对内区分来增强对比学习，实现同时的鲁棒性和可解释性。我们的方法无需架构修改，可无缝集成到现有训练管道中。ECG相位不变性和IMU旋转鲁棒性的实验展示了卓越性能：在相位偏移下，ECG相似度从0.25提高到0.91，而WISDM活动识别实现了86.65%的准确率和95.38%的旋转一致性，性能持续优于传统数据增强。这项工作代表了从反应性数据增强到主动结构学习的范式转变，使神经网络中的潜在表示具有可解释性。


### 论文摘要

Neural networks exhibit severe brittleness to semantically irrelevant transformations. A mere 75ms electrocardiogram (ECG) phase shift degrades latent cosine similarity from 1.0 to 0.2, while sensor rotations collapse activity recognition performance with inertial measurement units (IMUs). We identify the root cause as "laissez-faire" representation learning, where latent spaces evolve unconstrained provided task performance is satisfied. We propose Structured Contrastive Learning (SCL), a framework that partitions latent space representations into three semantic groups: invariant features that remain consistent under given transformations (e.g., phase shifts or rotations), variant features that actively differentiate transformations via a novel variant mechanism, and free features that preserve task flexibility. This creates controllable push-pull dynamics where different latent dimensions serve distinct, interpretable purposes. The variant mechanism enhances contrastive learning by encouraging variant features to differentiate within positive pairs, enabling simultaneous robustness and interpretability. Our approach requires no architectural modifications and integrates seamlessly into existing training pipelines. Experiments on ECG phase invariance and IMU rotation robustness demonstrate superior performance: ECG similarity improves from 0.25 to 0.91 under phase shifts, while WISDM activity recognition achieves 86.65% accuracy with 95.38% rotation consistency, consistently outperforming traditional data augmentation. This work represents a paradigm shift from reactive data augmentation to proactive structural learning, enabling interpretable latent representations in neural networks.

---

## 48. X-WIN: Building Chest Radiograph World Model via Predictive Sensing

**论文链接:** [http://arxiv.org/abs/2511.14918v1](http://arxiv.org/abs/2511.14918v1)

**作者:** Zefan Yang, Ge Wang, James Hendler, Mannudeep K. Kalra, Pingkun Yan

**发布时间:** 2025-11-18

### GPT解析

### 总结

该论文提出了X-WIN模型，一种新型胸部X光片世界模型，通过从胸部CT中提取体积知识来克服传统X光片的局限性，实现对三维解剖结构的理解和预测。

### 背景

胸部X光片(CXR)是疾病诊断的重要医学成像技术，但作为二维投影图像，受结构重叠限制，无法捕获三维解剖结构，这给表示学习和疾病诊断带来挑战。

### 目的

解决传统胸部X光片无法捕获三维解剖结构的问题，通过开发一个能够理解和预测三维空间中各种变换下的CXR图像的世界模型。

### 方法

提出X-WIN模型，通过学习预测潜在空间中的二维投影，从胸部CT中提取体积知识；引入基于相似性的对比对齐损失来捕获来自同一体积的投影间的相关信息；通过掩码图像建模将真实CXR纳入训练；采用域分类器确保真实和模拟CXR表示的统计相似性。

### 主要发现

X-WIN在使用线性探测和少样本微调的各种下游任务中优于现有基础模型；X-WIN能够渲染二维投影以重建三维CT体积。

### 结论

X-WIN模型成功地从胸部CT中提取了三维知识，能够预测和生成高质量的胸部X光片，并在多种下游任务中表现出色。

### 翻译

胸部X光摄影(CXR)是疾病诊断必不可少的医学成像技术。然而，作为二维投影图像，CXR受结构重叠的限制，因此无法捕获三维解剖结构。这一限制使得表示学习和疾病诊断变得困难。为了解决这一挑战，我们提出了一种新颖的CXR世界模型，名为X-WIN，它通过学习预测潜在空间中的二维投影，从胸部计算机断层扫描(CT)中提取体积知识。核心思想是，一个内部化三维解剖结构知识的世界模型可以预测三维空间中各种变换下的CXR。在投影预测过程中，我们引入了一种基于相似性的对比对齐损失，它利用相互相似性来捕获来自同一体积的投影之间的丰富相关信息。为了提高模型适应性，我们通过掩码图像建模将真实的CXR纳入训练，并采用域分类器来确保真实和模拟CXR的表示在统计上相似。全面的实验表明，在使用线性探测和少样本微调的各种下游任务中，X-WIN优于现有的基础模型。X-WIN还展示了渲染二维投影以重建三维CT体积的能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决胸部X射线摄影(CXR)作为2D投影图像无法捕捉3D解剖结构的问题，这限制了表示学习和疾病诊断的准确性。这个问题很重要，因为CXR是最广泛使用的胸部疾病检测成像技术，具有成本低、辐射剂量小和可及性高的优点，但相比CT缺乏3D空间信息；而CT虽然能提供详细3D结构，但成本高、辐射大且可及性有限，特别是在欠发达地区。因此，开发一种能从CT中学习3D知识并应用于CXR的方法，可以在保持CXR优势的同时提高诊断准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到放射科医生能通过正面和侧面CXR在认知上重建胸部3D模型，这启发了他们构建能捕获3D解剖结构知识的世界模型。他们借鉴了世界建模概念，将世界内部理解嵌入代理模型；参考了医学领域的CheXWorld世界模型，但认识到其仅限于2D的局限性；还借鉴了知识蒸馏思想，将CT中的知识'蒸馏'到CXR模型中。具体设计上，作者引入了三种损失函数：基于亲和力的对比对齐损失捕捉投影间的相关信息，掩码图像建模损失增强局部特征学习能力，结构保持域适应损失弥合真实和模拟数据域的差距。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过在潜在空间中学习预测胸部CT的2D投影，从CT中提取体积知识，使模型能够理解X射线图像在3D空间变换下的变化，从而内部化3D解剖结构知识。整体实现流程包括：1)架构设计，包含编码器、指数移动平均编码器和基于动作的预测器；2)动作设计，将动作定义为辐射源的旋转以获取不同角度的投影；3)训练过程，使用三种损失函数进行训练；4)应用，将训练好的模型用于下游CXR解释任务。模型接收常规X射线投影作为输入，通过预测新投影视图来学习3D结构知识。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将3D空间知识整合到CXR世界模型中，填补了现有模型与放射科医生能力间的差距；2)引入基于亲和力的对比对齐目标，增强判别特征编码并利用投影间的丰富对应关系；3)展示有效的3D重建能力，证明模型对3D结构的稳健理解。相比之前的工作，X-WIN突破了2D限制，从CT中提取体积知识而非仅依赖2D图像；采用对比损失而非疾病标签进行预训练；实现了模型对3D解剖结构的内部化理解，而不仅是2D特征学习。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'X-WIN通过从3D CT中提取体积知识并应用于2D CXR，首次实现了在胸部X射线放射照片世界中模型内部化3D解剖结构的能力，显著提升了疾病诊断的性能。'}


### 论文摘要

Chest X-ray radiography (CXR) is an essential medical imaging technique for disease diagnosis. However, as 2D projectional images, CXRs are limited by structural superposition and hence fail to capture 3D anatomies. This limitation makes representation learning and disease diagnosis challenging. To address this challenge, we propose a novel CXR world model named X-WIN, which distills volumetric knowledge from chest computed tomography (CT) by learning to predict its 2D projections in latent space. The core idea is that a world model with internalized knowledge of 3D anatomical structure can predict CXRs under various transformations in 3D space. During projection prediction, we introduce an affinity-guided contrastive alignment loss that leverages mutual similarities to capture rich, correlated information across projections from the same volume. To improve model adaptability, we incorporate real CXRs into training through masked image modeling and employ a domain classifier to encourage statistically similar representations for real and simulated CXRs. Comprehensive experiments show that X-WIN outperforms existing foundation models on diverse downstream tasks using linear probing and few-shot fine-tuning. X-WIN also demonstrates the ability to render 2D projections for reconstructing a 3D CT volume.

---

## 49. Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization

**论文链接:** [http://arxiv.org/abs/2511.14846v1](http://arxiv.org/abs/2511.14846v1)

**作者:** Yifeng Ding, Hung Le, Songyang Han, Kangrui Ruan, Zhenghui Jin, Varun Kumar, Zijian Wang, Anoop Deoras

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究提出了GTPO算法，通过三个创新点解决了现有强化学习方法在多轮工具集成推理任务中训练停滞的问题，并在推理基准测试上表现优异。

### 背景

训练大型语言模型进行多轮工具集成推理（模型迭代推理、生成代码并通过执行验证）对现有强化学习方法仍然具有挑战性。当前RL方法如GRPO依赖于粗糙的轨迹级奖励，为复杂多轮交互提供不足的学习信号，导致训练停滞。

### 目的

开发一种新型强化学习算法，专门用于训练大型语言模型完成多轮工具集成推理任务，解决现有方法中奖励信号不足的问题。

### 方法

提出了GTPO（Group Turn Policy Optimization）算法，包含三个关键创新：1）轮次级奖励分配，为单个轮次提供细粒度反馈；2）基于回报的优势估计，使用归一化折扣回报计算优势；3）自监督奖励塑造，利用生成代码中的自监督信号来丰富稀疏的基于二元结果的奖励。

### 主要发现

全面评估表明，GTPO在多个不同的推理基准测试上平均比GRPO高出3.0%，证明了其在推进现实世界复杂数学推理方面的有效性。

### 结论

GTPO算法通过提供更细粒度的奖励信号和更有效的学习机制，显著提高了大型语言模型在多轮工具集成推理任务上的性能，为复杂推理任务提供了新的解决方案。

### 翻译

训练大型语言模型进行多轮工具集成推理——模型迭代推理、生成代码并通过执行验证——对现有的强化学习方法来说仍然具有挑战性。当前RL方法，以组相对策略优化为例，存在粗糙的轨迹级奖励问题，为复杂的多轮交互提供了不足的学习信号，导致训练停滞。为解决这个问题，我们提出了组轮次策略优化，这是一种专门为在多轮TIR任务上训练LLMs而设计的新型RL算法。GTPO引入了三个关键创新：（1）轮次级奖励分配，为单个轮次提供细粒度反馈；（2）基于回报的优势估计，其中归一化折扣回报被计算为优势；（3）自监督奖励塑造，利用生成代码中的自监督信号来丰富稀疏的基于二元结果的奖励。我们的全面评估表明，GTPO在多个不同的推理基准测试上平均比GRPO高出3.0%，证明了其在推进现实世界复杂数学推理方面的有效性。


### 论文摘要

Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world.

---

## 50. Jasper-Token-Compression-600M Technical Report

**论文链接:** [http://arxiv.org/abs/2511.14405v2](http://arxiv.org/abs/2511.14405v2)

**作者:** Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu

**发布时间:** 2025-11-18

**备注:** 10 pages, 1 figure

### GPT解析

### 总结

这篇技术报告介绍了开源的Jasper-Token-Compression-600M模型的训练方法和评估结果，该模型于2025年11月发布。该模型基于之前英语Stella和Jasper模型的蒸馏方法，成功扩展到英汉双语领域，并通过对比学习进一步提高了模型性能。

### 背景

该研究基于之前英语Stella和Jasper模型的蒸馏方法，将其扩展到双语领域。

### 目的

开发一个高效的双语模型，通过结合知识蒸馏和标记压缩技术，在保持高质量的同时提高推理效率。

### 方法

引入基于一维卷积的标记压缩模块，在训练过程中动态调整压缩率，结合知识蒸馏和标记压缩技术。

### 主要发现

通过结合知识蒸馏和标记压缩技术，显著提高了嵌入质量和推理效率。模型比传统的0.6B模型效率更高，同时性能可比8B模型。

### 结论

成功开发了一个高效的双语模型，在保持高性能的同时提高了推理效率。

### 翻译

这篇技术报告介绍了开源的Jasper-Token-Compression-600M模型的训练方法和评估结果，该模型于2025年11月发布。基于之前英语Stella和Jasper模型的蒸馏方法，我们成功将这种方法扩展到英汉双语领域，并通过融入对比学习进一步提高了模型性能。我们模型的一个关键创新是引入了基于一维卷积的标记压缩模块。我们在训练过程中动态调整压缩率，使模型能够学习更鲁棒和高效的压缩文本表示。通过结合知识蒸馏和标记压缩技术，我们在嵌入质量和推理效率方面都取得了显著改进。我们的模型比传统的0.6B模型效率更高，同时实现了与8B模型相当的性能。有关模型发布的更多信息，请访问：https://huggingface.co/infgrad/Jasper-Token-Compression-600M。


### 论文摘要

This technical report presents the training methodology and evaluation results of the open-source Jasper-Token-Compression-600M model, released in November 2025. Building on previous distillation-based recipes from the English Stella and Jasper models, we successfully extend this approach to a bilingual (English and Chinese) domain, further enhancing model performance through the incorporation of contrastive learning. A key innovation of our model is the introduction of a one-dimensional convolution-based token compression module. We dynamically adjust the compression rate during training, enabling the model to learn more robust and efficient compressed text representations. By combining knowledge distillation with token compression techniques, we achieve significant improvements in both embedding quality and inference efficiency. Our model performs with higher efficiency than a traditional 0.6B model while achieving performance comparable to that of an 8B model. For more information on the model release, visit: https://huggingface.co/infgrad/Jasper-Token-Compression-600M.

---

## 51. In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data

**论文链接:** [http://arxiv.org/abs/2511.15704v1](http://arxiv.org/abs/2511.15704v1)

**作者:** Xiongyi Cai, Ri-Zhao Qiu, Geng Chen, Lai Wei, Isabella Liu, Tianshu Huang, Xuxin Cheng, Xiaolong Wang

**发布时间:** 2025-11-19

**备注:** Project webpage: https://xiongyicai.github.io/In-N-On/

### GPT解析

### 总结

该论文提出了一种可扩展的收集和使用独角视角视频数据的方案，通过分类人类数据并创建大型数据集，训练了一个能够遵循语言指令、具有少样本学习能力和提高鲁棒性的Human0模型。

### 背景

独角视角视频是学习和操作策略的有价值且可扩展的数据源。然而，由于显著的数据异质性，大多数现有方法利用人类数据进行简单预训练，这并未完全释放其潜力。

### 目的

提供一种可扩展的收集和使用独角视角数据的方案，通过将人类数据分为自然场景和任务导向两类，并系统分析如何使用这些数据。

### 方法

创建了一个数据集PHSD，包含超过1000小时的多样化自然场景独角视角数据和超过20小时直接对齐目标操作任务的任务导向数据；学习了一个大型独角视角语言条件流匹配策略Human0；使用领域适应技术最小化人类和类人机器人之间的差距。

### 主要发现

从扩展人类数据中获得了几个新特性：仅使用人类数据就能遵循语言指令；具有少样本学习能力；使用任务导向数据提高了鲁棒性。

### 结论

通过分类人类数据并系统分析如何使用这些数据，可以解锁独角视角数据的全部潜力，Human0模型展示了从扩展人类数据中获得的新特性。

### 翻译

独角视角视频是学习和操作策略的有价值且可扩展的数据源。然而，由于显著的数据异质性，大多数现有方法利用人类数据进行简单预训练，这并未完全释放其潜力。本文首先通过将人类数据分为自然场景和任务导向两类，并提供如何使用这些数据的系统分析，提供了收集和使用独角视角数据的可扩展方案。我们首先整理了一个数据集PHSD，包含超过1000小时的多样化自然场景独角视角数据和超过20小时直接对齐目标操作任务的任务导向数据。这使得能够学习大型独角视角语言条件流匹配策略Human0。通过领域适应技术，Human0最小化了人类和类人机器人之间的差距。经验上，我们展示了从扩展人类数据获得的几个新特性，包括仅使用人类数据就能遵循语言指令、少样本学习以及使用任务导向数据提高鲁棒性。


### 论文摘要

Egocentric videos are a valuable and scalable data source to learn manipulation policies. However, due to significant data heterogeneity, most existing approaches utilize human data for simple pre-training, which does not unlock its full potential. This paper first provides a scalable recipe for collecting and using egocentric data by categorizing human data into two categories: in-the-wild and on-task alongside with systematic analysis on how to use the data. We first curate a dataset, PHSD, which contains over 1,000 hours of diverse in-the-wild egocentric data and over 20 hours of on-task data directly aligned to the target manipulation tasks. This enables learning a large egocentric language-conditioned flow matching policy, Human0. With domain adaptation techniques, Human0 minimizes the gap between humans and humanoids. Empirically, we show Human0 achieves several novel properties from scaling human data, including language following of instructions from only human data, few-shot learning, and improved robustness using on-task data. Project website: https://xiongyicai.github.io/In-N-On/

---

## 52. From Qubits to Couplings: A Hybrid Quantum Machine Learning Framework for LHC Physics

**论文链接:** [http://arxiv.org/abs/2511.15672v1](http://arxiv.org/abs/2511.15672v1)

**作者:** Marwan Ait Haddou, Mohamed Belfkir, Salah Eddine El Harrauss

**发布时间:** 2025-11-19

**备注:** 30 pages, 10 figures

### GPT解析

### 总结

本文提出了一种新的混合量子机器学习框架，用于提高双希格斯玻色子搜索的灵敏度，在大型强子对撞机13.6 TeV能量下对HH→b̄bγγ末态进行研究。

### 背景

在高能物理实验中，寻找双希格斯玻色子信号是研究希格斯玻色子性质和电弱对称性破缺机制的重要途径，但需要高灵敏度的分析方法。

### 目的

提高双希格斯玻色子搜索的灵敏度，特别是在HH→b̄bγγ末态下的检测能力。

### 方法

提出一种混合量子机器学习框架，结合参数化量子电路与经典神经网络元模型，使事件级特征能够嵌入量子特征空间，同时保持经典学习的优化稳定性。

### 主要发现

混合模型比先进的XGBoost模型和纯量子实现模型性能提高一倍；在10%和50%的背景归一化不确定性下，实现了非共振双希格斯玻色子产生截面的95%置信水平上限分别为1.9倍和2.1倍标准模型截面；对希格斯玻色子自耦合和四重矢量玻色子-希格斯耦合的约束预期也有所改善。

### 结论

混合量子机器学习方法能够有效提高高能物理实验中稀有粒子过程的检测灵敏度，为未来粒子物理研究提供了新的分析工具。

### 翻译

在本文中，我们提出了一种新的混合量子机器学习框架，以提高在√s=13.6 TeV下双希格斯玻色子搜索在HH→b̄bγγ末态的灵敏度。所提出的模型将参数化量子电路与经典神经网络元模型相结合，使事件级特征能够嵌入量子特征空间，同时保持经典学习的优化稳定性。混合模型的性能比先进的XGBoost模型和纯量子实现提高了一倍，在背景归一化不确定性分别为10%和50%的情况下，实现了非共振双希格斯玻色子产生截面的95%置信水平上限为1.9×σ_SM和2.1×σ_SM。此外，与经典和纯量子模型相比，对希格斯玻色子自耦合κ_λ和四重矢量玻色子-希格斯耦合κ_2V的预期约束也有所改善。


### 论文摘要

In this paper, we propose a new Hybrid Quantum Machine Learning (HyQML) framework to improve the sensitivity of double Higgs boson searches in the $HH \to b\bar{b}γγ$ final state at $\sqrt{s}$ = 13.6 TeV. The proposed model combines parameterized quantum circuits with a classical neural network meta-model, enabling event-level features to be embedded in a quantum feature space while maintaining the optimization stability of classical learning. The hybrid model outperforms both a state-of-the-art XGBoost model and a purely quantum implementation by a factor of two, achieving an expected 95% CL upper limit on the non-resonant double Higgs boson production cross-section of $1.9\timesσ_{\text{SM}}$ and $2.1\timesσ_{\text{SM}}$ under background normalization uncertainties of 10% and 50%, respectively. In addition, expected constraints on the Higgs boson self-coupling $κ_λ$ and quartic vector-boson-Higgs coupling $κ_{2V}$ are found to be improved compared to the classical and purely quantum models.

---

## 53. Meta-Black-Box Optimization with Bi-Space Landscape Analysis and Dual-Control Mechanism for SAEA

**论文链接:** [http://arxiv.org/abs/2511.15551v1](http://arxiv.org/abs/2511.15551v1)

**作者:** Yukun Du, Haiyue Yu, Xiaotong Xie, Yan Zheng, Lixin Zhan, Yudong Du, Chongshuang Hu, Boxuan Wang, Jiang Jiang

**发布时间:** 2025-11-19

### GPT解析

### 总结

论文提出了一种名为DB-SAEA的新型元黑盒优化框架，通过双重控制和双空间探索性景观分析技术，显著提升了代理辅助进化算法在多目标优化中的性能和灵活性。

### 背景

代理辅助进化算法(SAEAs)被广泛用于昂贵的黑盒优化问题，但它们在搜索过程中依赖刚性、手动设计的组件(如填充标准和进化策略)，限制了跨任务的灵活性。

### 目的

解决传统SAEAs的局限性，提出一种能够实现双重控制并适应多目标问题的元黑盒优化框架。

### 方法

DB-SAEA学习元策略共同调节候选生成和填充标准选择；采用基于注意力的双空间ELA模块从真实和代理评估空间捕获优化状态；集成TabPFN作为代理模型；通过强化学习进行训练，利用并行采样和集中训练提高效率和可迁移性。

### 主要发现

DB-SAEA在多样化基准测试中优于最先进基线方法，并在高维度未见任务上展现出强大的零样本迁移能力。

### 结论

该工作首次实现了对SAEAs的双重控制MetaBBO框架，并引入了能捕获代理模型信息的双空间ELA技术。

### 翻译

代理辅助进化算法(SAEEAs)被广泛用于昂贵的黑盒优化。然而，它们在搜索过程中依赖刚性、手动设计的组件，如填充标准和进化策略，限制了跨任务的灵活性。为解决这些限制，我们提出双控制双空间代理辅助进化算法(DB-SAEA)，这是一种针对多目标问题定制的元黑盒优化(MetaBBO)框架。DB-SAEA学习一个元策略，共同调节候选生成和填充标准选择，实现双重控制。DB-SAEA中的双空间探索性景观分析(ELA)模块采用基于注意力的架构，从真实和代理评估空间中捕获优化状态，同时确保跨问题维度、种群大小和目标的可扩展性。此外，我们集成TabPFN作为代理模型，进行准确高效的预测和不确定性估计。该框架通过强化学习进行训练，利用并行采样和集中训练提高效率和跨任务可迁移性。实验结果表明，DB-SAEA不仅在多样化基准测试中优于最先进的基线方法，而且在更高维度的未见任务上表现出强大的零样本迁移能力。这项工作引入了第一个具有SAEAs双重控制的MetaBBO框架和能够捕获代理模型信息的双空间ELA。


### 论文摘要

Surrogate-Assisted Evolutionary Algorithms (SAEAs) are widely used for expensive Black-Box Optimization. However, their reliance on rigid, manually designed components such as infill criteria and evolutionary strategies during the search process limits their flexibility across tasks. To address these limitations, we propose Dual-Control Bi-Space Surrogate-Assisted Evolutionary Algorithm (DB-SAEA), a Meta-Black-Box Optimization (MetaBBO) framework tailored for multi-objective problems. DB-SAEA learns a meta-policy that jointly regulates candidate generation and infill criterion selection, enabling dual control. The bi-space Exploratory Landscape Analysis (ELA) module in DB-SAEA adopts an attention-based architecture to capture optimization states from both true and surrogate evaluation spaces, while ensuring scalability across problem dimensions, population sizes, and objectives. Additionally, we integrate TabPFN as the surrogate model for accurate and efficient prediction with uncertainty estimation. The framework is trained via reinforcement learning, leveraging parallel sampling and centralized training to enhance efficiency and transferability across tasks. Experimental results demonstrate that DB-SAEA not only outperforms state-of-the-art baselines across diverse benchmarks, but also exhibits strong zero-shot transfer to unseen tasks with higher-dimensional settings. This work introduces the first MetaBBO framework with dual-level control over SAEAs and a bi-space ELA that captures surrogate model information.

---

## 54. Convergence and Sketching-Based Efficient Computation of Neural Tangent Kernel Weights in Physics-Based Loss

**论文链接:** [http://arxiv.org/abs/2511.15530v1](http://arxiv.org/abs/2511.15530v1)

**作者:** Max Hirsch, Federico Pichi

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文研究了多目标优化中基于神经切线核(NTK)的自适应权重选择方法，解决了收敛性和计算效率问题

### 背景

在多目标优化中，多个损失项通过加权组合成单一目标，物理信息神经网络(PINNs)中常使用基于NTK的自适应权重来改善网络泛化性能，但收敛性不明确且计算负担大

### 目的

证明增强自适应NTK权重的梯度下降算法的收敛性，并提高NTK计算的效率

### 方法

通过理论分析证明梯度下降在适当条件下收敛；开发受预测器-校正器方法和矩阵素描启发的随机算法，产生NTK的无偏估计

### 主要发现

在适当条件下，增强自适应NTK权重的梯度下降算法是收敛的；提出的随机算法能有效估计NTK且具有任意小的离散化误差

### 结论

数值实验验证了理论发现和随机算法的有效性，代码已公开在GitHub平台

### 翻译

在多目标优化中，多个损失项被加权并相加形成单一目标。这些权重根据某些元目标被选择，以适当地平衡竞争性损失。例如，在物理信息神经网络(PINNs)中，这些权重通常自适应选择以提高网络的泛化误差。自适应权重的流行选择基于PINN的神经切线核(NTK)，它描述了训练期间网络在预测器空间中的演化。这种自适应权重算法的收敛性事先并不明确。此外，这些基于NTK的权重在训练过程中会频繁更新，进一步增加了学习过程的计算负担。在本文中，我们证明在适当条件下，增强自适应NTK权重的梯度下降在适当意义上是收敛的。然后，我们通过开发一种受预测器-校正器方法和矩阵素描启发的随机算法来解决计算效率问题，该算法产生NTK的无偏估计，具有任意小的离散化误差。最后，我们提供数值实验来支持我们的理论发现，并展示我们的随机算法的有效性。代码可用性：https://github.com/maxhirsch/Efficient-NTK


### 论文摘要

In multi-objective optimization, multiple loss terms are weighted and added together to form a single objective. These weights are chosen to properly balance the competing losses according to some meta-goal. For example, in physics-informed neural networks (PINNs), these weights are often adaptively chosen to improve the network's generalization error. A popular choice of adaptive weights is based on the neural tangent kernel (NTK) of the PINN, which describes the evolution of the network in predictor space during training. The convergence of such an adaptive weighting algorithm is not clear a priori. Moreover, these NTK-based weights would be updated frequently during training, further increasing the computational burden of the learning process. In this paper, we prove that under appropriate conditions, gradient descent enhanced with adaptive NTK-based weights is convergent in a suitable sense. We then address the problem of computational efficiency by developing a randomized algorithm inspired by a predictor-corrector approach and matrix sketching, which produces unbiased estimates of the NTK up to an arbitrarily small discretization error. Finally, we provide numerical experiments to support our theoretical findings and to show the efficacy of our randomized algorithm. Code Availability: https://github.com/maxhirsch/Efficient-NTK

---

## 55. Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining

**论文链接:** [http://arxiv.org/abs/2511.15456v1](http://arxiv.org/abs/2511.15456v1)

**作者:** Qian'ang Mao, Yuxuan Zhang, Jiaman Chen, Wenjun Zhou, Jiaqi Yan

**发布时间:** 2025-11-19

**备注:** Written in 2025 Q1

### GPT解析

### 总结

本文提出了交易意图挖掘(TIM)框架，用于解决去中心化金融(DeFi)中用户意图理解的问题。该框架结合了基于扎根理论的DeFi意图分类法和多智能体大语言模型系统，通过元级规划器协调领域专家分解任务，问题解决者处理多模态数据，认知评估器确保结果可靠性。

### 背景

随着DeFi的发展，理解用户交易意图至关重要但具有挑战性，原因包括复杂的智能合约交互、多方面的链上/链下因素和不透明的十六进制日志记录，现有方法缺乏深入的语义洞察。

### 目的

提出交易意图挖掘(TIM)框架，以解决DeFi用户意图推断的挑战，提供更可靠的用户动机理解。

### 方法

TIM框架利用基于扎根理论的DeFi意图分类法和多智能体大语言模型(LLM)系统。元级规划器动态协调领域专家，将多视角特定意图分析分解为可解决的子任务。问题解决者使用多模态链上/链下数据处理这些任务。认知评估器减轻LLM幻觉并确保结果可验证性。

### 主要发现

实验表明TIM显著优于机器学习模型、单一LLM和单一智能体基线。研究还分析了意图推断中的核心挑战。

### 结论

这项工作有助于更可靠地理解DeFi中的用户动机，为复杂的区块链活动提供上下文感知的解释。

### 翻译

随着去中心化金融(DeFi)的发展，理解DeFi交易背后的用户意图至关重要却充满挑战，这源于复杂的智能合约交互、多方面的链上/链下因素以及不透明的十六进制日志记录。现有方法缺乏深入的语义洞察。为此，我们提出了交易意图挖掘(TIM)框架。TIM利用基于扎根理论的DeFi意图分类法和多智能体大语言模型(LLM)系统来稳健地推断用户意图。元级规划器动态协调领域专家，将多视角特定意图分析分解为可解决的子任务。问题解决者使用多模态链上/链下数据处理这些任务，而认知评估器则减轻LLM幻觉并确保可验证性。实验表明，TIM显著优于机器学习模型、单一LLM和单一智能体基线。我们还分析了意图推断中的核心挑战。这项工作有助于更可靠地理解DeFi中的用户动机，为复杂的区块链活动提供上下文感知的解释。


### 论文摘要

As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity.

---

## 56. MAPROC at AHaSIS Shared Task: Few-Shot and Sentence Transformer for Sentiment Analysis of Arabic Hotel Reviews

**论文链接:** [http://arxiv.org/abs/2511.15291v1](http://arxiv.org/abs/2511.15291v1)

**作者:** Randa Zarnoufi

**发布时间:** 2025-11-19

### GPT解析

### 总结

这项研究针对阿拉伯方言的情感分析，特别是在酒店评论领域，采用SetFit框架进行少样本学习，在26个参赛者中排名第12位，F1得分为73%。

### 背景

阿拉伯方言情感分析面临重大挑战，主要由于语言多样性和标注数据稀缺。这项研究专注于酒店领域的阿拉伯方言情感分析。

### 目的

分类摩洛哥和沙特阿拉伯方言撰写的酒店评论的情感倾向（积极、消极或中性）。

### 方法

采用SetFit（Sentence Transformer Fine-tuning）框架，这是一种数据高效的少样本学习技术。

### 主要发现

在官方评估集上，系统获得了73%的F1分数，在26名参与者中排名第12位。

### 结论

这项工作强调了少样本学习在解决酒店评论等专门领域内处理细微阿拉伯方言文本的数据稀缺问题方面的潜力。

### 翻译

阿拉伯方言的情感分析由于语言多样性和标注数据的稀缺而面临重大挑战。本文描述了我们对AHaSIS共享任务的方法，该任务专注于酒店领域阿拉伯方言的情感分析。数据集包含用摩洛哥和沙特方言撰写的酒店评论，目标是评论者的情感分类为积极、消极或中性。我们采用了SetFit（Sentence Transformer Fine-tuning）框架，这是一种数据高效的少样本学习技术。在官方评估集上，我们的系统获得了73%的F1分数，在26名参与者中排名第12位。这项工作强调了少样本学习在解决酒店评论等专门领域内处理细微阿拉伯方言文本的数据稀缺问题方面的潜力。


### 论文摘要

Sentiment analysis of Arabic dialects presents significant challenges due to linguistic diversity and the scarcity of annotated data. This paper describes our approach to the AHaSIS shared task, which focuses on sentiment analysis on Arabic dialects in the hospitality domain. The dataset comprises hotel reviews written in Moroccan and Saudi dialects, and the objective is to classify the reviewers sentiment as positive, negative, or neutral. We employed the SetFit (Sentence Transformer Fine-tuning) framework, a data-efficient few-shot learning technique. On the official evaluation set, our system achieved an F1 of 73%, ranking 12th among 26 participants. This work highlights the potential of few-shot learning to address data scarcity in processing nuanced dialectal Arabic text within specialized domains like hotel reviews.

---

## 57. Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution

**论文链接:** [http://arxiv.org/abs/2511.15262v1](http://arxiv.org/abs/2511.15262v1)

**作者:** Tomas Espana, Yadh Hafsi, Fabrizio Lillo, Edoardo Vittori

**发布时间:** 2025-11-19

### GPT解析

### 总结

研究使用强化学习执行元订单的最优策略，采用无模型、数据驱动框架，通过队列反应模型生成限价订单簿模拟，训练双深度Q网络代理，结果表明该策略能适应市场条件并优于传统方法。

### 背景

传统方法采用参数化方法进行价格动态和影响建模，而本研究采用无模型、数据驱动框架来处理元订单执行问题。

### 目的

在较长时间内执行增量大型订单，同时最小化实现 shortfall 和市场影响。

### 方法

采用无模型、数据驱动框架，使用队列反应模型生成限价订单簿模拟，训练双深度Q网络代理，状态空间包括时间、库存、价格和深度变量，并与既定基准比较性能。

### 主要发现

代理学习到的策略既是战略性的又是战术性的，能够有效适应订单簿条件，在多种训练配置中优于标准方法。

### 结论

无模型强化学习可以为最优执行问题提供适应性强且稳健的解决方案。

### 翻译

我们研究了强化学习在元订单最优执行中的应用，目标是在较长时间内执行增量大型订单，同时最小化实现 shortfall 和市场影响。与传统参数化方法对价格动态和影响建模不同，我们采用无模型、数据驱动框架。由于策略优化需要历史数据无法提供的反事实反馈，我们采用队列反应模型生成真实且易于处理的限价订单簿模拟，包含瞬时价格影响以及非线性动态订单流响应。方法上，我们在包含时间、库存、价格和深度变量的状态空间上训练双深度Q网络代理，并对照既定基准评估其性能。数值模拟结果表明，代理学习到的策略既是战略性的又是战术性的，能有效适应订单簿条件，并在多种训练配置中优于标准方法。这些发现提供了强有力的证据，表明无模型强化学习可以为最优执行问题提供适应性强且稳健的解决方案。


### 论文摘要

We investigate the use of Reinforcement Learning for the optimal execution of meta-orders, where the objective is to execute incrementally large orders while minimizing implementation shortfall and market impact over an extended period of time. Departing from traditional parametric approaches to price dynamics and impact modeling, we adopt a model-free, data-driven framework. Since policy optimization requires counterfactual feedback that historical data cannot provide, we employ the Queue-Reactive Model to generate realistic and tractable limit order book simulations that encompass transient price impact, and nonlinear and dynamic order flow responses. Methodologically, we train a Double Deep Q-Network agent on a state space comprising time, inventory, price, and depth variables, and evaluate its performance against established benchmarks. Numerical simulation results show that the agent learns a policy that is both strategic and tactical, adapting effectively to order book conditions and outperforming standard approaches across multiple training configurations. These findings provide strong evidence that model-free Reinforcement Learning can yield adaptive and robust solutions to the optimal execution problem.

---

## 58. Learning Where, What and How to Transfer: A Multi-Role Reinforcement Learning Approach for Evolutionary Multitasking

**论文链接:** [http://arxiv.org/abs/2511.15199v1](http://arxiv.org/abs/2511.15199v1)

**作者:** Jiajun Zhan, Zeyuan Ma, Yue-Jiao Gong, Kay Chen Tan

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种通过强化学习设计系统化且可泛化的知识转移策略，用于进化多任务算法，解决了确定转移任务、转移知识和转移机制三个主要挑战。

### 背景

进化多任务算法通常需要专门设计的知识转移机制，以确保多任务优化中的收敛性和最优性，而现有方法往往需要针对特定任务进行定制化设计。

### 目的

探索通过强化学习设计一个系统化且可泛化的知识转移策略，以解决EMT算法中知识转移的三个关键问题：转移的任务、转移的知识和转移的机制。

### 方法

构建了一个多角色强化学习系统，包含三组专门代理：任务路由代理(基于注意力的相似性识别模块确定源-目标转移对)、知识控制代理(确定转移的精英解比例)和策略适应代理(通过控制超参数调节转移强度)。通过在增强的多任务问题分布上对所有网络模块进行端到端预训练，获得可泛化的元策略。

### 主要发现

全面的验证实验表明，与代表性基线相比，该方法达到了最先进的性能。深入分析不仅揭示了所提方法的原理，还提供了对系统所学内容的深入解释。

### 结论

通过强化学习设计的知识转移策略能够有效解决EMT算法中的知识转移问题，该方法具有系统性和可泛化性，不需要针对特定任务进行定制设计。

### 翻译

进化多任务算法通常需要专门设计的知识转移机制，以确保多任务优化中的收敛性和最优性。在本文中，我们探索通过强化学习设计一个系统化且可泛化的知识转移策略。我们首先确定了三个主要挑战：确定转移的任务(哪里)、转移的知识(什么)以及转移的机制(如何)。为应对这些挑战，我们构建了一个多角色强化学习系统，其中三组策略网络作为专门代理：任务路由代理整合了基于注意力的相似性识别模块，通过注意力分数确定源-目标转移对；知识控制代理确定转移的精英解比例；一组策略适应代理通过动态控制基础EMT框架中的超参数来控制转移强度。通过在增强的多任务问题分布上对所有网络模块进行端到端预训练，获得了可泛化的元策略。全面的验证实验表明，与代表性基线相比，我们的方法达到了最先进的性能。进一步的深入分析不仅揭示了所提方法的原理，还提供了对系统所学内容的深入解释。


### 论文摘要

Evolutionary multitasking (EMT) algorithms typically require tailored designs for knowledge transfer, in order to assure convergence and optimality in multitask optimization. In this paper, we explore designing a systematic and generalizable knowledge transfer policy through Reinforcement Learning. We first identify three major challenges: determining the task to transfer (where), the knowledge to be transferred (what) and the mechanism for the transfer (how). To address these challenges, we formulate a multi-role RL system where three (groups of) policy networks act as specialized agents: a task routing agent incorporates an attention-based similarity recognition module to determine source-target transfer pairs via attention scores; a knowledge control agent determines the proportion of elite solutions to transfer; and a group of strategy adaptation agents control transfer strength by dynamically controlling hyper-parameters in the underlying EMT framework. Through pre-training all network modules end-to-end over an augmented multitask problem distribution, a generalizable meta-policy is obtained. Comprehensive validation experiments show state-of-the-art performance of our method against representative baselines. Further in-depth analysis not only reveals the rationale behind our proposal but also provide insightful interpretations on what the system have learned.

---

## 59. FaultDiffusion: Few-Shot Fault Time Series Generation with Diffusion Model

**论文链接:** [http://arxiv.org/abs/2511.15174v1](http://arxiv.org/abs/2511.15174v1)

**作者:** Yi Xu, Zhigang Chen, Rui Wang, Yangfan Li, Fengxiao Tang, Ming Zhao, Jiaqi Liu

**发布时间:** 2025-11-19

**备注:** 4 figures, 5 tables ,8 pages

### GPT解析

### 总结

本文提出了一种基于扩散模型的新型少样本故障时间序列生成框架，通过正负差异适配器和多样性损失解决了工业设备监控中故障数据稀缺导致的生成样本真实性和多样性不足的问题

### 背景

在工业设备监控中，故障诊断对确保系统可靠性和实现预测性维护至关重要，但故障数据稀缺阻碍了数据驱动方法的发展

### 目的

提出一种基于扩散模型的新型少样本故障时间序列生成框架，以解决现有模型在少样本场景下难以捕捉故障分布的问题

### 方法

采用正负差异适配器利用预训练的正常数据分布建模正常和故障领域差异；引入多样性损失防止模式崩溃，通过样本间差异正则化鼓励生成多样化故障样本

### 主要发现

实验结果表明，该模型在真实性和多样性方面明显优于传统方法，在关键基准测试上达到了最先进的性能

### 结论

该研究提出的框架有效解决了工业设备监控中少样本故障时间序列生成的挑战

### 翻译

在工业设备监控中，故障诊断对于确保系统可靠性和实现预测性维护至关重要。然而，由于故障事件的稀有性和数据标注的高成本，故障数据稀缺，这显著阻碍了数据驱动方法的发展。现有的针对丰富正常数据优化的时间序列生成模型，在少样本场景下难以捕捉故障分布，由于故障领域差距大和类内变异性高，产生的样本缺乏真实性和多样性。为解决这一问题，我们提出了一种基于扩散模型的新型少样本故障时间序列生成框架。我们的方法采用正负差异适配器，利用预训练的正常数据分布来建模正常和故障领域之间的差异，以实现准确的故障合成。此外，引入了多样性损失来防止模式崩溃，通过样本间差异正则化鼓励生成多样化的故障样本。实验结果表明，我们的模型在真实性和多样性方面显著优于传统方法，在关键基准测试上取得了最先进的性能。


### 论文摘要

In industrial equipment monitoring, fault diagnosis is critical for ensuring system reliability and enabling predictive maintenance. However, the scarcity of fault data, due to the rarity of fault events and the high cost of data annotation, significantly hinders data-driven approaches. Existing time-series generation models, optimized for abundant normal data, struggle to capture fault distributions in few-shot scenarios, producing samples that lack authenticity and diversity due to the large domain gap and high intra-class variability of faults. To address this, we propose a novel few-shot fault time-series generation framework based on diffusion models. Our approach employs a positive-negative difference adapter, leveraging pre-trained normal data distributions to model the discrepancies between normal and fault domains for accurate fault synthesis. Additionally, a diversity loss is introduced to prevent mode collapse, encouraging the generation of diverse fault samples through inter-sample difference regularization. Experimental results demonstrate that our model significantly outperforms traditional methods in authenticity and diversity, achieving state-of-the-art performance on key benchmarks.

---

## 60. WiCo-PG: Wireless Channel Foundation Model for Pathloss Map Generation via Synesthesia of Machines

**论文链接:** [http://arxiv.org/abs/2511.15030v1](http://arxiv.org/abs/2511.15030v1)

**作者:** Mingran Sun, Lu Bai, Ziwei Huang, Xuesong Cai, Xiang Cheng, Jianjun Wu

**发布时间:** 2025-11-19

### GPT解析

### 总结

论文提出了一种基于机器通感(SoM)的无线信道基础模型WiCo-PG，用于6G无人机到地面(U2G)场景的路径损耗图生成。该模型通过构建多模态数据集预训练，利用双向量量化生成对抗网络和Transformer以及频率引导的共享路由专家混合架构实现跨模态路径损耗图生成，实验表明其性能优于现有方案。

### 背景

针对第六代(6G)无人机到地面(U2G)通信场景，需要高效准确的路径损耗图生成方法来优化无线网络规划和部署。

### 目的

开发一种无线信道基础模型WiCo-PG，通过机器通感(SoM)实现路径损耗图的高精度生成，特别是在6G U2G场景中。

### 方法

构建包含多种U2G场景、不同飞行高度和多种频段的多模态感知-通信数据集；提出基于双向量量化生成对抗网络(VQGANs)和Transformer的新型网络架构；设计频率引导的共享路由专家混合(S-R MoE)架构；利用RGB图像实现跨模态路径损耗图生成。

### 主要发现

WiCo-PG通过预训练实现了归一化均方误差(NMSE)为0.012的路径损耗图生成精度；比基于大语言模型的LLM4PG方案和传统深度学习方案高出6.98 dB；在少样本泛化场景中，仅使用2.7%的样本即可比LLM4PG至少提高1.37 dB的性能。

### 结论

WiCo-PG模型通过创新的架构设计和预训练策略，在6G U2G场景的路径损耗图生成任务中表现出色，具有良好的泛化能力和精度优势。

### 翻译

首次开发了基于机器通感(SoM)的无线信道基础模型WiCo-PG用于路径损耗图生成。考虑到第六代(6G)无人机到地面(U2G)场景，构建了一个新的多模态感知-通信数据集用于WiCo-PG预训练，包括多种U2G场景、不同的飞行高度和多样的频段。基于构建的数据集，所提出的WiCo-PG能够利用来自不同场景和飞行高度的RGB图像实现跨模态路径损耗图生成。在WiCo-PG中，提出了一个基于双向量量化生成对抗网络(VQGANs)和Transformer的新型网络架构，用于跨模态路径损耗图生成。此外，设计了一种新颖的频率引导的共享路由专家混合(S-R MoE)架构用于跨模态路径损耗图生成。仿真结果表明，所提出的WiCo-PG通过预训练实现了改进的路径损耗图生成精度，归一化均方误差(NMSE)为0.012，优于基于大语言模型(LLM)的方案即LLM4PG和传统基于深度学习的方案超过6.98 dB。所提出的WiCo-PG增强的泛化能力可以在少样本泛化中使用2.7%的样本进一步比LLM4PG至少提高1.37 dB。


### 论文摘要

A wireless channel foundation model for pathloss map generation (WiCo-PG) via Synesthesia of Machines (SoM) is developed for the first time. Considering sixth-generation (6G) uncrewed aerial vehicle (UAV)-to-ground (U2G) scenarios, a new multi-modal sensing-communication dataset is constructed for WiCo-PG pre-training, including multiple U2G scenarios, diverse flight altitudes, and diverse frequency bands. Based on the constructed dataset, the proposed WiCo-PG enables cross-modal pathloss map generation by leveraging RGB images from different scenarios and flight altitudes. In WiCo-PG, a novel network architecture designed for cross-modal pathloss map generation based on dual vector quantized generative adversarial networks (VQGANs) and Transformer is proposed. Furthermore, a novel frequency-guided shared-routed mixture of experts (S-R MoE) architecture is designed for cross-modal pathloss map generation. Simulation results demonstrate that the proposed WiCo-PG achieves improved pathloss map generation accuracy through pre-training with a normalized mean squared error (NMSE) of 0.012, outperforming the large language model (LLM)-based scheme, i.e., LLM4PG, and the conventional deep learning-based scheme by more than 6.98 dB. The enhanced generality of the proposed WiCo-PG can further outperform the LLM4PG by at least 1.37 dB using 2.7% samples in few-shot generalization.

---

## 61. Reflexive Evidence-Based Multimodal Learning for Clean Energy Transitions: Causal Insights on Cooking Fuel Access, Urbanization, and Carbon Emissions

**论文链接:** [http://arxiv.org/abs/2511.15342v1](http://arxiv.org/abs/2511.15342v1)

**作者:** Shan Shan

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究介绍了ClimateAgents，一个基于AI的框架，结合大型语言模型和领域专业代理，用于支持假设生成和情景探索，以实现可持续发展目标7（经济实惠和清洁能源）。

### 背景

实现可持续发展目标7需要技术创新和对影响能源获取和碳排放的社会经济因素的更深入理解。目前存在关键问题，如如何量化这些因素对能源系统的影响、建模跨域相互作用，以及在能源转型背景下捕捉反馈动态。

### 目的

解决上述知识空白，引入ClimateAgents框架来支持假设生成和情景探索，并以数据驱动的方式识别碳排放的关键决定因素。

### 方法

开发了ClimateAgents框架，使用来自265个经济体、国家和地区的20年社会经济和排放数据，以及世界银行数据库的98个指标，应用基于机器学习的因果推理方法，以证据驱动的方式识别碳排放的关键决定因素。

### 主要发现

分析确定了三个主要驱动因素：农村地区清洁烹饪燃料的获取、城市地区清洁烹饪燃料的获取，以及城市地区人口比例。这些发现强调了清洁烹饪技术和城市化模式在塑造排放结果方面的关键作用。

### 结论

ClimateAgents提供了一个模块化和反思性学习系统，支持生成可信和可行的政策见解。通过整合异构数据模式，该框架有助于适应性政策制定基础设施，能够应对复杂的社会技术挑战。该方法旨在支持从孤立建模转向为动态、上下文感知的气候行动设计的反思性、模块化系统。

### 翻译

实现可持续发展目标7（经济实惠和清洁能源）不仅需要技术创新，还需要更深入地理解影响能源获取和碳排放的社会经济因素。虽然这些因素正受到关注，但关键问题仍然存在，特别是关于如何量化它们对能源系统的影响、建模它们的跨域相互作用，以及在能源转型更广泛的背景下捕捉反馈动态。为解决这些空白，本研究引入了ClimateAgents，一个基于AI的框架，结合大型语言模型和领域专业代理，以支持假设生成和情景探索。利用来自265个经济体、国家和地区的20年社会经济和排放数据，以及从世界银行数据库提取的98个指标，该框架应用基于机器学习的因果推理方法，以证据驱动、数据驱动的方式识别碳排放的关键决定因素。分析强调了三个主要驱动因素：农村地区清洁烹饪燃料的获取、城市地区清洁烹饪燃料的获取，以及城市地区人口比例。这些发现强调了清洁烹饪技术和城市化模式在塑造排放结果方面的关键作用。响应基于证据的AI政策日益增长的呼声，ClimateAgents提供了一个模块化和反思性学习系统，支持生成可信和可行的政策见解。通过整合异构数据模式，包括结构化指标、政策文档和语义推理，该框架有助于适应性政策制定基础设施，能够应对复杂的社会技术挑战。这种方法旨在支持从孤立建模转向为动态、上下文感知的气候行动设计的反思性、模块化系统。


### 论文摘要

Achieving Sustainable Development Goal 7 (Affordable and Clean Energy) requires not only technological innovation but also a deeper understanding of the socioeconomic factors influencing energy access and carbon emissions. While these factors are gaining attention, critical questions remain, particularly regarding how to quantify their impacts on energy systems, model their cross-domain interactions, and capture feedback dynamics in the broader context of energy transitions. To address these gaps, this study introduces ClimateAgents, an AI-based framework that combines large language models with domain-specialized agents to support hypothesis generation and scenario exploration. Leveraging 20 years of socioeconomic and emissions data from 265 economies, countries and regions, and 98 indicators drawn from the World Bank database, the framework applies a machine learning based causal inference approach to identify key determinants of carbon emissions in an evidence-based, data driven manner. The analysis highlights three primary drivers: access to clean cooking fuels in rural areas, access to clean cooking fuels in urban areas, and the percentage of population living in urban areas. These findings underscore the critical role of clean cooking technologies and urbanization patterns in shaping emission outcomes. In line with growing calls for evidence-based AI policy, ClimateAgents offers a modular and reflexive learning system that supports the generation of credible and actionable insights for policy. By integrating heterogeneous data modalities, including structured indicators, policy documents, and semantic reasoning, the framework contributes to adaptive policymaking infrastructures that can evolve with complex socio-technical challenges. This approach aims to support a shift from siloed modeling to reflexive, modular systems designed for dynamic, context-aware climate action.

---

## 62. Joint Semantic-Channel Coding and Modulation for Token Communications

**论文链接:** [http://arxiv.org/abs/2511.15699v1](http://arxiv.org/abs/2511.15699v1)

**作者:** Jingkai Ying, Zhijin Qin, Yulong Feng, Liejun Wang, Xiaoming Tao

**发布时间:** 2025-11-19

**备注:** 14 pages, 14 figures, 2 tables

### GPT解析

### 总结

本文研究了一种基于Transformer架构的token通信方法，特别是在点云数据处理中的应用，提出了一种联合语义-信道和调制(JSCCM)方案，用于高效可靠地传输tokens。

### 背景

Transformer架构近年来在各种任务和模态中表现出色，token是这些模型中统一的输入和输出表示，已成为基本信息单元。点云是一种流行的三维格式，与图像或视频相比具有更复杂的空间结构。

### 目的

研究如何高效可靠地传输tokens，特别是在点云数据处理场景中。

### 方法

使用集合抽象方法获取点tokens，提出JSCCM方案，该方案包括两个并行的基于Point Transformer的编码器和一个结合Gumel-softmax和软量化方法的差分调制器，同时开发了速率分配器和信道适配器，根据语义信息和信道条件自适应生成高质量的调制tokens。

### 主要发现

所提出的方法在重建性能上比联合语义信道编码和传统分离编码高出超过1dB，在调制符号压缩比上提高了6倍以上。

### 结论

JSCCM方案在点云token通信中表现出色，能够高效可靠地传输tokens，同时保持高质量的重建效果和高压缩比。

### 翻译

近年来，Transformer架构在广泛的任务和模态中取得了卓越的性能。Token是基于Transformer模型中统一的输入和输出表示，已成为基本信息单元。在这项工作中，我们考虑token通信问题，研究如何高效可靠地传输tokens。点云是一种流行的三维格式，与图像或视频相比具有更复杂的空间结构，我们选择其作为信息源。我们使用集合抽象方法获取点tokens。随后，为了获得更具信息性和传输友好的基于token的表示，我们为token编码器提出了一种联合语义-信道和调制(JSCCM)方案，将点token映射到标准数字星座点(调制token)。具体而言，JSCCM包括两个并行的基于Point Transformer的编码器和一个结合了Gumel-softmax和软量化方法的差分调制器。此外，开发了速率分配器和信道适配器，根据语义信息和信道条件促进高质量调制token的自适应生成。大量模拟实验表明，所提出的方法优于联合语义信道编码和传统分离编码，在重建性能上实现了超过1dB的提升，在调制符号压缩比上提高了6倍以上。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决token通信问题，即如何高效可靠地传输tokens（作为Transformer模型的基本信息单元）。论文选择点云作为信息源，因为点云是重要的三维数据表示方式，在沉浸式媒体、自动驾驶和机器人等领域有广泛应用，但其传输会导致数据量显著增加，对现有通信系统构成重大挑战。研究这个问题的重要性在于：1)点云数据在现实应用中日益重要但带宽需求巨大；2)当前通信系统难以高效传输点云数据；3)Transformer架构已成为主流模型，其token的高效传输对建立'集成AI与通信'系统至关重要；4)语义通信可显著提高通信效率和鲁棒性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的设计思路是：1)认识到Transformer架构的token作为统一表示形式的重要性；2)点云相比图像/视频有更复杂空间结构，是研究token通信的良好选择；3)需要解决语义通信缺乏统一表示和跨模态架构差距的问题。设计方法包括：使用Set Abstraction获取点tokens；提出联合语义-信道编码和调制(JSCCM)方案；采用两个并行的Point Transformer编码器和差分调制器；开发速率分配器和信道适配器。借鉴的工作包括：PointNet++的Set Abstraction方法；Point Transformer架构；Gumbel-Softmax和软量化方法；现有语义通信系统设计如SEPT方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1)联合语义-信道编码与调制，将语义编码与信道编码结合并直接调制；2)并行编码结构使用两个Point Transformer提取不同层次特征；3)通过速率分配器和信道适配器实现自适应传输；4)结合Gumbel-Softmax和软量化实现可微分调制。整体流程：发送端-点Tokenizer将点云转换为tokens→JSCCM编码器处理tokens→速率分配器和信道适配器(可选)调整→功率归一化→无线传输；接收端-信号处理→解调→Token解码器处理→点De-tokenizer重建点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个端到端点云token通信系统；2)JSCCM方案使用并行Point Transformer和差分调制器；3)速率分配器基于语义动态调整传输符号数；4)信道适配器使模型适应不同信道环境。相比之前工作的不同：1)与传统分离式编码不同，本文联合优化信源和信道编码；2)与现有语义通信方法不同，本文将输出映射到有限信道符号集，实现与数字通信系统兼容；3)与现有token通信工作不同，本文专门应用于点云传输并设计专门组件；4)调制方法结合Gumbel-Softmax和软量化优点，优于单独使用任一方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于联合语义-信道编码和调制的新型点云token通信方法，通过自适应速率分配和信道适配，实现了在复杂信道条件下高效可靠的三维点云数据传输，显著提升了点云重建质量和压缩效率。'}


### 论文摘要

In recent years, the Transformer architecture has achieved outstanding performance across a wide range of tasks and modalities. Token is the unified input and output representation in Transformer-based models, which has become a fundamental information unit. In this work, we consider the problem of token communication, studying how to transmit tokens efficiently and reliably. Point cloud, a prevailing three-dimensional format which exhibits a more complex spatial structure compared to image or video, is chosen to be the information source. We utilize the set abstraction method to obtain point tokens. Subsequently, to get a more informative and transmission-friendly representation based on tokens, we propose a joint semantic-channel and modulation (JSCCM) scheme for the token encoder, mapping point tokens to standard digital constellation points (modulated tokens). Specifically, the JSCCM consists of two parallel Point Transformer-based encoders and a differential modulator which combines the Gumel-softmax and soft quantization methods. Besides, the rate allocator and channel adapter are developed, facilitating adaptive generation of high-quality modulated tokens conditioned on both semantic information and channel conditions. Extensive simulations demonstrate that the proposed method outperforms both joint semantic-channel coding and traditional separate coding, achieving over 1dB gain in reconstruction and more than 6x compression ratio in modulated symbols.

---

## 63. CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

**论文链接:** [http://arxiv.org/abs/2511.15580v1](http://arxiv.org/abs/2511.15580v1)

**作者:** Sifan Zhou, Yichao Cao, Jiahao Nie, Yuqian Fu, Ziyu Zhao, Xiaobo Lu, Shuo Wang

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

CompTrack是一种新颖的端到端框架，通过消除点云中的空间冗余和信息冗余，解决了3D单目标跟踪中的双重冗余挑战，实现了高性能和高效率的平衡。

### 背景

3D单目标跟踪在激光雷达点云中是计算机视觉和自动驾驶的关键任务。尽管取得了显著进展，但点云的固有稀疏性带来了双重冗余挑战：一是来自背景噪声的大量空间冗余损害了准确性，二是前景中的信息冗余阻碍了效率。

### 目的

系统性地消除点云中的两种冗余问题，提高3D单目标跟踪的准确性和效率。

### 方法

提出CompTrack框架，包含两个主要模块：(1)空间前景预测器(SFP)模块，基于信息熵过滤不相关背景噪声；(2)信息瓶颈引导的动态令牌压缩(IB-DTC)模块，利用低秩近似理论和在线SVD分析将冗余前景自适应压缩为紧凑且信息丰富的代理令牌集合。

### 主要发现

在KITTI、nuScenes和Waymo数据集上的大量实验表明，CompTrack实现了最先进的跟踪性能，同时在单块RTX 3090 GPU上能够以实时90 FPS的速度运行。

### 结论

CompTrack通过有效消除点云中的双重冗余，显著提高了3D单目标跟踪的准确性和效率，为自动驾驶等应用提供了有效的解决方案。

### 翻译

激光雷达点云中的3D单目标跟踪是计算机视觉和自动驾驶中的关键任务。尽管取得了巨大成功，但点云的固有稀疏性带来了双重冗余挑战，限制了现有跟踪器：(1)来自背景噪声的大量空间冗余损害了准确性，(2)前景中的信息冗余阻碍了效率。为解决这些问题，我们提出了CompTrack，一种新颖的端到端框架，系统性地消除点云中的两种冗余。首先，Comp集成了一个空间前景预测器(SFP)模块，基于信息熵过滤掉不相关的背景噪声，解决空间冗余问题。随后，其核心是一个信息瓶颈引导的动态令牌压缩(IB-DTC)模块，消除前景中的信息冗余。该模块基于低秩近似理论，利用在线SVD分析将冗余前景自适应压缩为紧凑且信息丰富的代理令牌集合。在KITTI、nuScenes和Waymo数据集上的大量实验表明，CompTrack实现了最先进的跟踪性能，同时具有更高的效率，在单块RTX 3090 GPU上以实时90 FPS的速度运行。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云数据中的双重冗余问题：1) 空间冗余 - 大量不相关的背景噪声和空点淹没了实际目标特征；2) 信息冗余 - 前景点中并非所有点都具有同等信息价值，导致大量低信息量、高度相关的点主导表示。这个问题在自动驾驶和机器人领域至关重要，因为3D单目标跟踪是这些应用的核心任务，而双重冗余限制了跟踪器的效率和准确性，难以满足实时处理需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从信息论角度分析点云数据的双重冗余问题，识别出现有方法只处理空间冗余而忽视信息冗余的局限性。他们设计了一个端到端框架CompTrack，包含空间前景预测器(SFP)过滤背景噪声，以及信息瓶颈引导的动态令牌压缩(IB-DTC)模块消除信息冗余。方法借鉴了PillarHist处理点云、CenterPoint的中心引导训练策略、Transformer的注意力机制以及Hui等人的损失函数设计，但创新性地结合了信息瓶颈理论和低秩近似来同时解决两种冗余问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是同时消除点云数据中的空间冗余和信息冗余，利用信息瓶颈理论和低秩近似将冗余数据压缩为紧凑且信息丰富的表示。整体流程：1) 将点云转换为鸟瞰图(BEV)特征；2) 空间前景预测器(SFP)过滤背景噪声；3) 信息瓶颈引导的动态令牌压缩(IB-DTC)模块通过在线SVD分析和可学习查询机制压缩前景；4) 使用交叉注意力生成压缩后的代理令牌；5) 预测目标位置和方向。整个系统通过组合损失函数进行端到端训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 首次系统性地处理点云跟踪中的双重冗余问题；2) 空间前景预测器(SFP)基于信息熵分析过滤背景；3) 信息瓶颈引导的动态令牌压缩(IB-DTC)结合SVD分析和可学习查询；4) 实现90 FPS的高效跟踪。相比之前工作，不同之处在于：之前方法主要关注空间冗余而忽视信息冗余；CompTrack同时处理两种冗余；引入信息瓶颈理论和低秩近似作为理论基础；在保持精度的同时显著提高了处理速度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CompTrack通过同时消除点云数据中的空间冗余和信息冗余，实现了高效且准确的3D单目标跟踪，在保持领先性能的同时将处理速度提升至90 FPS。'}


### 论文摘要

3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU.

---

## 64. Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.15311v1](http://arxiv.org/abs/2511.15311v1)

**作者:** Mehran Tamjidi, Hamidreza Dastmalchi, Mohammadreza Alimoradijazi, Ali Cheraghian, Aijun An, Morteza Saberi

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI 2026. 7 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为Uni-Adapter的无需训练的在线测试时适应策略，用于解决3D视觉语言基础模型在噪声、不完整或分布不同的实际场景中表现不佳的问题。

### 背景

3D视觉语言基础模型在开放世界点云处理任务中展现出强大的泛化和零样本识别能力，但在实际应用场景中，当数据存在噪声、不完整或与训练数据分布不同时，这些模型的表现往往不尽如人意。

### 目的

开发一种基于动态原型学习的训练免费在线测试时适应策略，以提高3D视觉语言基础模型在异构数据分布下的适应性和性能。

### 方法

Uni-Adapter通过定义3D缓存存储类特定的聚类中心作为动态原型，这些原型被持续更新以捕获异构数据分布中的类内变异性；同时使用基于图的标签平滑模块捕获原型间的相似性，确保相似原型间的标签一致性；最后通过熵加权聚合统一原始模型和改进缓存的预测结果。

### 主要发现

无需重新训练，Uni-Adapter能有效缓解分布偏移问题，在多个3D基准测试上实现了最先进的性能，相比原始3D视觉语言基础模型，ModelNet-40C提高了10.55%，ScanObjectNN-C提高了8.26%，ShapeNet-C提高了4.49%。

### 结论

Uni-Adapter是一种有效的3D视觉语言基础模型测试时适应方法，能够在不重新训练的情况下显著提高模型在实际应用场景中的性能表现。

### 翻译

3D视觉语言基础模型在开放世界点云处理任务中表现出强大的泛化和零样本识别能力。然而，在实际场景中，当数据存在噪声、不完整或与训练数据分布不同时，这些模型的表现往往不佳。为此，我们提出了Uni-Adapter，这是一种基于动态原型学习的、无需训练的在线测试时适应策略，用于3D视觉语言基础模型。我们定义了一个3D缓存来存储类特定的聚类中心作为原型，这些原型被持续更新以捕获异构数据分布中的类内变异性。这些动态原型通过相似度评分作为基于缓存的计算的锚点。同时，一个基于图的标签平滑模块捕获原型间的相似性，以强制相似原型之间的标签一致性。最后，我们使用熵加权聚合统一原始3D视觉语言基础模型和改进的3D缓存的预测结果，以实现可靠的适应。无需重新训练，Uni-Adapter有效缓解了分布偏移问题，在不同3D基准测试上针对不同3D视觉语言基础模型实现了最先进的性能，相比源3D视觉语言基础模型，ModelNet-40C提高了10.55%，ScanObjectNN-C提高了8.26%，ShapeNet-C提高了4.49%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D视觉语言基础模型在实际场景中性能下降的问题，当3D点云数据存在噪声、不完整或与训练数据分布不同时，这些模型的性能会显著降低。这个问题在现实中非常重要，因为实际应用中的3D数据常常受到传感器噪声、遮挡和分辨率低等因素的影响，而现有模型难以适应这些变化；同时，传统的领域适应方法通常需要重新训练模型或获取目标域的标注数据，这在实际应用中往往不可行。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D视觉语言模型在分布变化下的局限性，指出测试时适应是一种有前景的解决方案。他们发现现有的缓存策略主要基于高置信度样本，但3D数据中同一类别往往存在多种结构模式，高置信度样本无法覆盖所有变化，导致决策边界不准确。因此，作者设计了一种基于聚类的缓存策略，借鉴了视觉语言模型的跨模态表示学习、测试时适应的缓存框架、图正则化技术和在线聚类算法，通过动态维护多个聚类中心作为原型来捕获数据分布的多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过基于聚类的动态原型学习增强3D视觉语言模型在测试时的适应能力，而不需要重新训练。具体包括：1)为每个类别维护多个聚类中心作为原型，捕获3D数据的多样性；2)构建原型间的相似性图，通过图正则化减少噪声伪标签影响；3)根据预测不确定性动态融合原始模型和缓存模型的预测。整体流程分为四个步骤：在线原型模块(动态更新聚类中心)、原型重新分配模块(通过图正则化优化标签)、缓存日志计算(计算输入与原型的相似度)、基于熵的融合(根据不确定性加权预测结果)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于聚类的缓存策略，为每个类别维护多个聚类中心；2)图正则化的标签平滑，减少噪声伪标签影响；3)熵加权预测融合，根据不确定性动态调整权重；4)完全免训练的测试时适应，适合实时应用。相比之前的工作，Uni-Adapter不同于仅缓存高置信度样本的方法(如Point-Cache)，能更好地捕获3D数据的多样性；不同于需要梯度计算的TTA方法(如TPT)，计算效率更高；专门针对3D点云数据特点进行了优化，而非简单套用2D视觉语言模型的适应方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Uni-Adapter提出了一种基于动态聚类的免训练测试时适应框架，通过维护多样化的3D原型和图正则化标签平滑，显著提升了视觉语言模型在噪声和不完整3D点云数据上的鲁棒性和泛化能力，无需重新训练即可适应新的数据分布。'}


### 论文摘要

3D Vision-Language Foundation Models (VLFMs) have shown strong generalization and zero-shot recognition capabilities in open-world point cloud processing tasks. However, these models often underperform in practical scenarios where data are noisy, incomplete, or drawn from a different distribution than the training data. To address this, we propose Uni-Adapter, a novel training-free online test-time adaptation (TTA) strategy for 3D VLFMs based on dynamic prototype learning. We define a 3D cache to store class-specific cluster centers as prototypes, which are continuously updated to capture intra-class variability in heterogeneous data distributions. These dynamic prototypes serve as anchors for cache-based logit computation via similarity scoring. Simultaneously, a graph-based label smoothing module captures inter-prototype similarities to enforce label consistency among similar prototypes. Finally, we unify predictions from the original 3D VLFM and the refined 3D cache using entropy-weighted aggregation for reliable adaptation. Without retraining, Uni-Adapter effectively mitigates distribution shifts, achieving state-of-the-art performance on diverse 3D benchmarks over different 3D VLFMs, improving ModelNet-40C by 10.55%, ScanObjectNN-C by 8.26%, and ShapeNet-C by 4.49% over the source 3D VLFMs.

---

## 65. MambaTrack3D: A State Space Model Framework for LiDAR-Based Object Tracking under High Temporal Variation

**论文链接:** [http://arxiv.org/abs/2511.15077v1](http://arxiv.org/abs/2511.15077v1)

**作者:** Shengjing Tian, Yinan Han, Xiantong Zhao, Xuehu Liu, Qi Lang

**发布时间:** 2025-11-19

**备注:** This work has been submitted to a journal for possible publication

### GPT解析

### 总结

MambaTrack3D是一种基于状态空间模型Mamba的新型高时变(HTV)导向跟踪框架，用于解决LiDAR点云中的3D单目标跟踪问题，在保持高精度的同时实现了近线性计算复杂度。

### 背景

具有高时变(HTV)特性的动态户外环境对LiDAR点云中的3D单目标跟踪提出了重大挑战。现有的基于内存的跟踪器通常面临二次计算复杂度、时间冗余和几何先验利用不足等问题。

### 目的

解决现有内存跟踪器在HTV环境中的计算复杂度高、时间冗余和几何先验利用不足的问题，提出一种高效且准确的3D单目标跟踪方法。

### 方法

提出了MambaTrack3D，包含两个主要模块：1) 基于Mamba的帧间传播(MIP)模块：用高效的帧间传播替代传统的单帧特征提取，实现近线性复杂度并显式建模历史帧间的空间关系；2) 分组特征增强模块(GFEM)：在通道级别分离前景和背景语义，减少内存库中的时间冗余。

### 主要发现

在KITTI-HTV和nuScenes-HTV基准测试中，MambaTrack3D持续优于HTV导向和常规场景跟踪器，相比HVTrack在中度时间间隔下实现高达6.5%的成功率和9.5%的精度提升。在标准KITTI数据集上，MambaTrack3D与最先进的常规场景跟踪器保持高度竞争力，证明了其强大的泛化能力。

### 结论

MambaTrack3D实现了卓越的精度-效率权衡，在专门的HTV和传统跟踪场景中都能提供强大的性能。

### 翻译

具有高时变(HTV)特性的动态户外环境对LiDAR点云中的3D单目标跟踪提出了重大挑战。现有的基于内存的跟踪器通常面临二次计算复杂度、时间冗余和几何先验利用不足等问题。为了解决这些问题，我们提出了MambaTrack3D，一种基于状态空间模型Mamba的新型HTV导向跟踪框架。具体而言，我们设计了一个基于Mamba的帧间传播(MIP)模块，用高效的帧间传播替代传统的单帧特征提取，实现近线性复杂度的同时显式建模历史帧间的空间关系。此外，引入了分组特征增强模块(GFEM)，在通道级别分离前景和背景语义，从而减少内存库中的时间冗余。在KITTI-HTV和nuScenes-HTV基准上的大量实验表明，MambaTrack3D持续优于HTV导向和常规场景跟踪器，相比HVTrack在中度时间间隔下实现高达6.5%的成功率和9.5%的精度提升。在标准KITTI数据集上，MambaTrack3D与最先进的常规场景跟踪器保持高度竞争力，证实了其强大的泛化能力。总体而言，MambaTrack3D实现了卓越的精度-效率权衡，在专门的HTV和传统跟踪场景中都能提供强大的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于激光雷达(LiDAR)点云的3D单目标跟踪在高时变(HTV)场景下面临的挑战。现有方法存在计算效率低(二次计算复杂度)、时间冗余和几何先验利用不足的问题。这个问题在现实中非常重要，因为HTV跟踪可以显著降低计算需求，对无人机等边缘设备的部署至关重要，在自动驾驶、机器人视觉和增强现实等领域不可或缺，同时现有方法在规范场景中表现良好但在高时变场景下性能显著下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有基于内存的跟踪器(如MBPTrack)使用Transformer架构导致的二次计算复杂度问题，以及它们在处理时间冗余和利用历史帧几何先验方面的不足。然后借鉴了状态空间模型Mamba的近线性复杂度特点，设计了基于Mamba的帧间传播(MIP)模块替代传统单帧特征提取，并提出了分组特征增强模块(GFEM)在通道级别分离前景和背景语义以减轻时间冗余。该方法借鉴了Siamese范式的特征融合思想、内存库跟踪方法利用历史上下文的思路、Mamba模型在序列建模方面的优势，以及点云处理技术用于特征提取。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用状态空间模型Mamba的近线性计算复杂度提高跟踪效率，通过帧间特征传播替代传统单帧特征提取显式建模历史帧空间关系，并使用分组特征增强分离前景和背景语义减少时间冗余。整体流程包括：1)输入当前帧点云和历史帧信息；2)MIP模块进行特征采样、历史帧连接、掩码整合、局部邻域构建、特征传播和双向状态空间模型处理；3)GFEM模块将特征分为前景和背景两组并执行交叉注意力；4)将处理后的特征输入定位网络输出跟踪结果；5)更新内存库信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Mamba-based Inter-frame Propagation (MIP)模块替代传统单帧特征提取，实现近线性计算复杂度；2)Grouped Feature Enhancement Module (GFEM)在通道级别分离前景和背景语义减轻时间冗余；3)高效的时空关系建模利用动态关键点选择和最优传输匹配。相比之前工作，该方法具有更高的计算效率(在8192个输入点下仍保持100 FPS)，利用多帧特征传播而非仅从最新帧传递线索，通过GFEM有效减轻时间冗余，显式利用历史帧几何先验信息，不仅在HTV场景下表现优异，在常规场景下也保持竞争力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MambaTrack3D通过基于状态空间模型的帧间特征传播和分组特征增强，实现了高计算效率和高精度的激光雷达点云目标跟踪，特别是在高时变场景下显著优于现有方法，同时保持了在常规场景下的竞争力。'}


### 论文摘要

Dynamic outdoor environments with high temporal variation (HTV) pose significant challenges for 3D single object tracking in LiDAR point clouds. Existing memory-based trackers often suffer from quadratic computational complexity, temporal redundancy, and insufficient exploitation of geometric priors. To address these issues, we propose MambaTrack3D, a novel HTV-oriented tracking framework built upon the state space model Mamba. Specifically, we design a Mamba-based Inter-frame Propagation (MIP) module that replaces conventional single-frame feature extraction with efficient inter-frame propagation, achieving near-linear complexity while explicitly modeling spatial relations across historical frames. Furthermore, a Grouped Feature Enhancement Module (GFEM) is introduced to separate foreground and background semantics at the channel level, thereby mitigating temporal redundancy in the memory bank. Extensive experiments on KITTI-HTV and nuScenes-HTV benchmarks demonstrate that MambaTrack3D consistently outperforms both HTV-oriented and normal-scenario trackers, achieving improvements of up to 6.5 success and 9.5 precision over HVTrack under moderate temporal gaps. On the standard KITTI dataset, MambaTrack3D remains highly competitive with state-of-the-art normal-scenario trackers, confirming its strong generalization ability. Overall, MambaTrack3D achieves a superior accuracy-efficiency trade-off, delivering robust performance across both specialized HTV and conventional tracking scenarios.

---

## 66. CPSL: Representing Volumetric Video via Content-Promoted Scene Layers

**论文链接:** [http://arxiv.org/abs/2511.14927v1](http://arxiv.org/abs/2511.14927v1)

**作者:** Kaiyuan Hu, Yili Jin, Junhua Liu, Xize Duan, Hong Kang, Xue Liu

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了Content-Promoted Scene Layers (CPSL)，一种紧凑的2.5D视频表示方法，能够在不显著增加成本的情况下，为传统2D内容提供体积视频的感知优势。

### 背景

体积视频通过支持自由视角探索和真实运动视差，提供沉浸式和交互式视觉体验。然而，现有的体积视频表示方法从显式点云到隐式神经场，在捕获、计算和渲染方面成本高昂，限制了其在按需视频中的可扩展性和实时通信中的可行性。

### 目的

开发一种能够降低体积视频成本的方法，为传统2D内容带来体积视频的感知优势，解决现有体积视频表示方法的高成本问题。

### 方法

提出Content-Promoted Scene Layers (CPSL)，一种紧凑的2.5D视频表示方法。该方法通过逐帧深度和内容显著性指导，将每帧分解为具有几何一致性的图层，配备软alpha带和边缘深度缓存保持遮挡顺序和边界连续性。通过深度加权的变形和从前到后的alpha合成实现视差校正的新颖视图合成，绕过昂贵的3D重建。使用运动引导的传播和每层编码保持帧间一致性，支持标准视频编解码器的实时播放。

### 主要发现

在多个基准测试中，与基于图层和神经场的方法相比，CPSL实现了更好的感知质量和边界保真度，同时将存储和渲染成本降低了数倍。

### 结论

该方法为从2D视频到可扩展的2.5D沉浸式媒体提供了一条实用路径。

### 翻译

体积视频通过支持自由视角探索和真实运动视差，能够提供沉浸式和交互式的视觉体验。然而，从显式点云到隐式神经场的现有体积视频表示方法，在捕获、计算和渲染方面仍然成本高昂，这限制了它们在按需视频中的可扩展性，并降低了它们在实时通信中的可行性。为了弥合这一差距，我们提出了内容促进的场景图层(Content-Promoted Scene Layers, CPSL)，这是一种紧凑的2.5D视频表示方法，将体积视频的感知优势带入传统2D内容。在逐帧深度和内容显著性的指导下，CPSL将每帧分解为一小组具有几何一致性的图层，这些图层配备软alpha带和边缘深度缓存，共同保持遮挡顺序和边界连续性。这些轻量级的2D可编码资产通过深度加权的变形和从前到后的alpha合成，实现视差校正的新颖视图合成，绕过了昂贵的3D重建。在时间上，CPSL使用运动引导的传播和每层编码保持帧间一致性，支持使用标准视频编解码器的实时播放。在多个基准测试中，与基于图层和神经场的方法相比，CPSL实现了更好的感知质量和边界保真度，同时将存储和渲染成本降低了数倍。我们的方法为从2D视频到可扩展的2.5D沉浸式媒体提供了一条实用路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决体积视频的高效表示和渲染问题。现有的体积视频表示方法（从显式点云到隐式神经场）在捕获、计算和渲染方面成本高昂，限制了它们在按需视频和实时通信中的可扩展性。这个问题很重要，因为体积视频能提供沉浸式交互体验，在远程呈现、交互娱乐和医疗等领域有广泛应用，但现有方法无法在保持高质量的同时实现实时通信和低带宽需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有体积视频表示方法的局限性，然后设定了开发一种紧凑2.5D视频表示的目标。他们选择了分层表示作为基础路线，但认识到传统方法在大视角偏移下保真度下降的问题。CPSL结合了深度图像渲染(DIBR)的几何扭曲、分层表示(MPI/LDI)的效率优势，以及神经场方法的高保真度，同时避免了它们的缺点。作者还借鉴了语义分割和显著性检测技术，实现了内容感知的分区策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'CPSL的核心思想是将单目视频分解为一小组深度排序的、内容对齐的RGBA层，这些层保留了视差、遮挡和边界连续性，同时保持轻量级和可流式传输。整体流程包括：1)层集生成：通过深度-语义融合、实例提升和背景合并创建内容感知的层；2)动态像素带(DPS)：在层边界处合成中间像素，解决重投影时的裂纹问题；3)视图自适应重投影和合成：根据观察者视点扭曲和合成层；4)时间组织：使用GOP结构和运动矢量传播保持时间连贯性；5)编码和播放：将层作为独立RGBA流编码，并在播放时进行实时合成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)内容引导的场景层(CPSL)：结合深度和语义信息进行内容感知的分区；2)动态像素带(DPS)：解决分层渲染在大视角变化下的裂纹问题；3)可扩展到全场景体积视频。相比之前工作，CPSL避免了显式方法(如点云)的高存储需求和隐式方法(如NeRF)的高计算成本，同时解决了传统分层表示(如MPI)在大视角偏移下的保真度下降问题。实验显示CPSL比点云流式传输减少7倍带宽，同时保持更高的感知质量和边界保真度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CPSL提出了一种内容引导的场景层表示方法，通过将单目视频分解为少量深度排序的RGBA层并配合动态像素带技术，实现了高质量体积视频的紧凑表示与高效渲染，为从2D视频到可扩展的2.5D沉浸式媒体提供了实用路径。'}


### 论文摘要

Volumetric video enables immersive and interactive visual experiences by supporting free viewpoint exploration and realistic motion parallax. However, existing volumetric representations from explicit point clouds to implicit neural fields, remain costly in capture, computation, and rendering, which limits their scalability for on-demand video and reduces their feasibility for real-time communication.   To bridge this gap, we propose Content-Promoted Scene Layers (CPSL), a compact 2.5D video representation that brings the perceptual benefits of volumetric video to conventional 2D content. Guided by per-frame depth and content saliency, CPSL decomposes each frame into a small set of geometry-consistent layers equipped with soft alpha bands and an edge-depth cache that jointly preserve occlusion ordering and boundary continuity. These lightweight, 2D-encodable assets enable parallax-corrected novel-view synthesis via depth-weighted warping and front-to-back alpha compositing, bypassing expensive 3D reconstruction. Temporally, CPSL maintains inter-frame coherence using motion-guided propagation and per-layer encoding, supporting real-time playback with standard video codecs. Across multiple benchmarks, CPSL achieves superior perceptual quality and boundary fidelity compared with layer-based and neural-field baselines while reducing storage and rendering cost by several folds. Our approach offer a practical path from 2D video to scalable 2.5D immersive media.

---

## 67. Point Cloud Quantization through Multimodal Prompting for 3D Understanding

**论文链接:** [http://arxiv.org/abs/2511.12079v2](http://arxiv.org/abs/2511.12079v2)

**作者:** Hongxuan Li, Wencheng Zhu, Huiying Xu, Xinzhong Zhu, Pengfei Zhu

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026. 11 pages, 7 figures

### GPT解析

### 总结

本研究提出了一种基于多模态提示驱动的点云分析量化框架，利用预训练模型的文本嵌入作为稳健的原型先验，通过双约束量化空间融合视觉和原型特征，形成同时编码几何和语义信息的混合表示。实验证明该方法在ModelNet40和ScanObjectNN数据集上具有优越效果。

### 背景

向量量化已成为大规模多模态模型中的强大工具，通过离散标记编码统一异构表示。然而，其有效性依赖于稳健的码本设计。当前基于原型的方法依赖于可训练向量或聚类质心，在代表性和可解释性方面存在不足，尽管多模态对齐在视觉语言模型中显示出其潜力。

### 目的

解决现有基于原型的向量量化方法在代表性和可解释性方面的局限性，提高多模态模型中向量量化的效果，特别是在点云分析任务中。

### 方法

提出一个简单的多模态提示驱动量化框架，基于两个核心洞见：1) 预训练模型的文本嵌入通过多对一对比对齐自然编码视觉语义，可作为稳健的原型先验；2) 多模态提示能够自适应地优化这些原型，有效缓解视觉语言语义差距。框架引入了由紧凑性和分离正则化强制执行的双约束量化空间，无缝集成视觉和原型特征，形成同时编码几何和语义信息的混合表示。采用Gumbel-Softmax松弛实现可微离散化，同时保持量化稀疏性。

### 主要发现

在ModelNet40和ScanObjectNN数据集上的大量实验明确展示了所提出方法的优越有效性。

### 结论

通过结合文本嵌入作为原型先验和多模态提示的优化能力，所提出的框架成功解决了现有向量量化方法的局限性，实现了更高效的多模态表示学习，特别是在点云分析任务中表现出色。

### 翻译

向量量化已成为大规模多模态模型中的一个强大工具，通过离散标记编码统一异构表示。然而，其有效性依赖于稳健的码本设计。当前依赖于可训练向量或聚类质心的基于原型的方法在代表性和可解释性方面存在不足，即使多模态对齐在视觉语言模型中显示出其前景。为解决这些局限性，我们提出了一种用于点云分析的简单多模态提示驱动量化框架。我们的方法基于两个核心洞见：1) 预训练模型的文本嵌入通过多对一对比对齐自然编码视觉语义，自然地作为稳健的原型先验；2) 多模态提示能够自适应地优化这些原型，有效缓解视觉语言语义差距。该框架引入了由紧凑性和分离正则化强制执行的双约束量化空间，无缝集成视觉和原型特征，形成同时编码几何和语义信息的混合表示。此外，我们采用Gumbel-Softmax松弛实现可微离散化，同时保持量化稀疏性。在ModelNet40和ScanObjectNN数据集上的大量实验明确展示了所提出方法的优越有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云分析中的向量量化问题，特别是如何设计更强大的代码书来提高表示能力和可解释性。这个问题很重要，因为点云是3D视觉的重要表示形式，广泛应用于自动驾驶、机器人等领域。当前方法无法有效捕捉几何和语义信息，限制了模型在复杂场景下的表现，提高点云理解对于实现更高级的3D场景理解至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于两个核心洞察设计方法：1)预训练模型中的文本嵌入通过多对一对比对齐自然编码了视觉语义，可作为强大原型先验；2)多模态提示可自适应改进这些原型，缓解视觉-语言语义差距。作者借鉴了CLIP、ULIP等大型多模态模型的对比学习思想，以及VQ-Wave2Vec等工作的Gumbel-Softmax松弛技术，创新性地将文本嵌入重新定义为可训练的视觉原型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将预训练视觉-语言模型中的文本嵌入作为语义原型，通过多模态提示改进这些原型，使用双约束量化空间整合视觉和原型特征，并采用Gumbel-Softmax松弛实现可微分离散化。整体流程包括：1)使用ULIP-2提取文本和点云特征；2)通过可学习提示和双约束损失优化文本原型；3)将点云特征量化为文本原型空间；4)融合原始点云特征与量化特征；5)使用对比、紧凑性和分离损失训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)文本驱动的3D量化框架统一视觉-语言对齐；2)将文本特征作为可训练原型并使用Gumbel分布实现端到端训练；3)双约束量化空间整合几何和语义信息；4)参数高效的微调策略。相比之前工作，该方法利用预训练模型的文本嵌入作为语义而非传统聚类质心，同时结合几何细节和高层语义，且采用参数高效的微调而非完全微调，显著减少了计算资源需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过多模态提示进行点云量化的新方法，利用文本嵌入作为语义原型，有效结合了几何细节和高层语义信息，显著提升了3D点云理解任务的性能，特别是在少样本学习和跨数据集泛化方面表现优异。'}


### 论文摘要

Vector quantization has emerged as a powerful tool in large-scale multimodal models, unifying heterogeneous representations through discrete token encoding. However, its effectiveness hinges on robust codebook design. Current prototype-based approaches relying on trainable vectors or clustered centroids fall short in representativeness and interpretability, even as multimodal alignment demonstrates its promise in vision-language models. To address these limitations, we propose a simple multimodal prompting-driven quantization framework for point cloud analysis. Our methodology is built upon two core insights: 1) Text embeddings from pre-trained models inherently encode visual semantics through many-to-one contrastive alignment, naturally serving as robust prototype priors; and 2) Multimodal prompts enable adaptive refinement of these prototypes, effectively mitigating vision-language semantic gaps. The framework introduces a dual-constrained quantization space, enforced by compactness and separation regularization, which seamlessly integrates visual and prototype features, resulting in hybrid representations that jointly encode geometric and semantic information. Furthermore, we employ Gumbel-Softmax relaxation to achieve differentiable discretization while maintaining quantization sparsity. Extensive experiments on the ModelNet40 and ScanObjectNN datasets clearly demonstrate the superior effectiveness of the proposed method.

---

## 68. Walrus: A Cross-Domain Foundation Model for Continuum Dynamics

**论文链接:** [http://arxiv.org/abs/2511.15684v1](http://arxiv.org/abs/2511.15684v1)

**作者:** Michael McCabe, Payel Mukhopadhyay, Tanya Marwah, Bruno Regaldo-Saint Blancard, Francois Rozet, Cristiana Diaconu, Lucas Meyer, Kaze W. K. Wong, Hadi Sotoudeh, Alberto Bietti, Irina Espejo, Rio Fear, Siavash Golkar, Tom Hehir, Keiya Hirashima, Geraud Krawezik, Francois Lanusse, Rudy Morel, Ruben Ohana, Liam Parker, Mariel Pettee, Jeff Shen, Kyunghyun Cho, Miles Cranmer, Shirley Ho

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了Walrus，一个基于transformer的基础模型，用于物理模拟中的类流体连续介质动力学。通过解决数据异构性、不稳定长时动态以及不同分辨率和维度带来的挑战，Walrus在19个多样化场景上进行了预训练，并在下游任务上表现出色，优于之前的基础模型。

### 背景

Foundation models已经彻底改变了语言和视觉领域的机器学习，但在物理模拟方面仍面临挑战。

### 目的

开发一个基础模型，能够有效处理物理模拟中的数据异构性、不稳定长时动态以及不同分辨率和维度的问题。

### 方法

采用基于谐波分析的稳定方法、负载平衡的分布式2D和3D训练策略，以及计算自适应的tokenization技术来克服物理模拟中的挑战。

### 主要发现

Walrus在19个多样化场景上进行了预训练，涵盖天体物理学、地球科学、流变学、等离子体物理学、声学和经典流体。实验表明，在下游任务上，无论是短期还是长期预测范围，Walrus都优于之前的基础模型。

### 结论

通过新的稳定方法、分布式训练策略和自适应tokenization技术，成功开发了Walrus基础模型，有效解决了物理模拟中的关键挑战，并在多种物理场景中展现出优越的性能。

### 翻译

基础模型已经改变了语言和视觉的机器学习，但在物理模拟中取得同等影响仍然是一个挑战。数据异构性和不稳定的长时动态阻碍了从足够多样化的动态中学习，而不同的分辨率和维度对在现代硬件上高效训练提出了挑战。通过经验分析和理论分析，我们纳入了新方法来减轻这些障碍，包括基于谐波分析的稳定方法、负载平衡的分布式2D和3D训练策略，以及计算自适应的tokenization。利用这些工具，我们开发了Walrus，一个基于transformer的基础模型，主要用于类流体连续介质动力学。Walrus在19个多样化场景上进行了预训练，涵盖天体物理学、地球科学、流变学、等离子体物理学、声学和经典流体。实验表明，在下游任务上，无论是短期还是长期预测范围，Walrus都优于之前的基础模型，而消融研究证实了我们的贡献对预测稳定性、训练吞吐量和传输性能的价值。代码和权重已发布供社区使用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决物理模拟领域的基础模型开发问题，具体包括数据异质性导致的多样化动态学习困难，以及不稳定长期动态和变化分辨率对高效训练的挑战。这个问题重要是因为数值模拟是现代工程和科学工作的基石，但计算成本高昂，传统模拟需要严格定义偏微分方程，这在多物理场景中不可行，而基础模型在物理模拟领域的应用还面临诸多障碍。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析物理模拟领域的多尺度和系统异质性挑战，认识到需要开发能处理不同分辨率和维度的通用模型。他们借鉴了语言和视觉领域的基础模型范式，将其扩展到物理模拟任务，同时采用transformer架构、数据增强和分布式训练等技术，并引入了自适应计算标记化等较新方法。设计过程结合了实证和理论分析，确定了需要克服的具体障碍并提出了针对性解决方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是开发一个能处理多种物理场景的跨域基础模型，通过新方法克服现有障碍并强调数据多样性。整体流程包括：1) 使用19个多样化场景数据集预训练；2) 采用空间-时间分解的transformer架构；3) 应用patch jittering提高稳定性；4) 通过2D到3D数据增强联合处理不同维度数据；5) 使用自适应计算标记化动态分配资源；6) 采用拓扑感知采样提高训练效率；7) 在多种下游任务评估短期和长期预测性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Patch jittering稳定方法，减少89%预训练场景的长期误差；2) 2D到3D数据增强技术；3) 自适应计算标记化；4) 拓扑感知采样提高262%训练吞吐量；5) 跨域基础模型覆盖19个多样化场景。相比之前工作，Walrus处理真正的2D和3D数据而非仅限2D，解决了长期预测不稳定问题，实现了更高效的训练，并在多个科学领域而非单一类型物理系统上表现优异。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Walrus是一个创新的跨域物理模拟基础模型，通过稳定方法、高效训练策略和自适应计算技术，在多种物理场景的短期和长期预测任务上实现了最先进性能。'}


### 论文摘要

Foundation models have transformed machine learning for language and vision, but achieving comparable impact in physical simulation remains a challenge. Data heterogeneity and unstable long-term dynamics inhibit learning from sufficiently diverse dynamics, while varying resolutions and dimensionalities challenge efficient training on modern hardware. Through empirical and theoretical analysis, we incorporate new approaches to mitigate these obstacles, including a harmonic-analysis-based stabilization method, load-balanced distributed 2D and 3D training strategies, and compute-adaptive tokenization. Using these tools, we develop Walrus, a transformer-based foundation model developed primarily for fluid-like continuum dynamics. Walrus is pretrained on nineteen diverse scenarios spanning astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids. Experiments show that Walrus outperforms prior foundation models on both short and long term prediction horizons on downstream tasks and across the breadth of pretraining data, while ablation studies confirm the value of our contributions to forecast stability, training throughput, and transfer performance over conventional approaches. Code and weights are released for community use.

---

## 69. CODE-II: A large-scale dataset for artificial intelligence in ECG analysis

**论文链接:** [http://arxiv.org/abs/2511.15632v1](http://arxiv.org/abs/2511.15632v1)

**作者:** Petrus E. O. G. B. Abreu, Gabriela M. M. Paixão, Jiawei Li, Paulo R. Gomes, Peter W. Macfarlane, Ana C. S. Oliveira, Vinicius T. Carvalho, Thomas B. Schön, Antonio Luiz P. Ribeiro, Antônio H. Ribeiro

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了CODE-II，一个大规模真实世界心电图数据集，包含270多万个12导联心电图，来自209万多名成年患者。该数据集具有标准化诊断注释和66个临床诊断类别，并提供了开放子集。使用CODE-II预训练的神经网络在外部基准测试上表现优越。

### 背景

基于人工智能的心电图分析方法快速发展，但数据集在注释质量、大小和范围方面的限制仍是主要挑战。大型数据集虽推动了AI心电图分析进展，但现有数据集存在局限性。

### 目的

创建一个大规模、高质量的心电图数据集，解决当前数据集在注释质量、大小和范围方面的局限性，促进心电图分析的人工智能研究。

### 方法

收集巴西米纳斯吉拉斯州远程医疗网络的270多万个12导联心电图，使用标准化诊断标准进行注释并由心脏病专家审查。开发66个临床诊断类别，提供CODE-II-open（15,000名患者子集）和CODE-II-test（8,475次检查子集）。

### 主要发现

在CODE-II上预训练的神经网络在外部基准测试(PTB-XL和CPSC 2018)上实现优越迁移性能，优于在更大数据集上训练的替代模型。

### 结论

CODE-II数据集为心电图分析的人工智能研究提供了宝贵资源，其规模、质量和多样性使基于深度学习的心电图分析方法能取得更好性能。

### 翻译

心电图解释的数据驱动方法正在迅速发展。大型数据集推动了基于人工智能的心电图分析进展，但注释质量、大小和范围的限制仍是主要挑战。在此，我们提出了CODE-II，这是一个大规模的真实世界数据集，包含来自巴西米纳斯吉拉斯州远程医疗网络的270多万个12导联心电图，来自209万多名成年患者。每次检查都使用标准化的诊断标准进行注释，并由心脏病专家审查。CODE-II的一个显著特征是一套66个临床上有意义的诊断类别，这些类别是在心脏病专家的参与下开发的，并在远程医疗实践中常规使用。我们还提供了一个开放获取的子集：CODE-II-open，这是一个包含15,000名患者的公共子集，以及CODE-II-test，这是一个包含8,475次检查的非重叠集合，由多位心脏病专家审查用于盲法评估。在CODE-II上预训练的神经网络在外部基准测试上实现了优越的迁移性能，并优于在更大数据集上训练的替代模型。


### 论文摘要

Data-driven methods for electrocardiogram (ECG) interpretation are rapidly progressing. Large datasets have enabled advances in artificial intelligence (AI) based ECG analysis, yet limitations in annotation quality, size, and scope remain major challenges. Here we present CODE-II, a large-scale real-world dataset of 2,735,269 12-lead ECGs from 2,093,807 adult patients collected by the Telehealth Network of Minas Gerais (TNMG), Brazil. Each exam was annotated using standardized diagnostic criteria and reviewed by cardiologists. A defining feature of CODE-II is a set of 66 clinically meaningful diagnostic classes, developed with cardiologist input and routinely used in telehealth practice. We additionally provide an open available subset: CODE-II-open, a public subset of 15,000 patients, and the CODE-II-test, a non-overlapping set of 8,475 exams reviewed by multiple cardiologists for blinded evaluation. A neural network pre-trained on CODE-II achieved superior transfer performance on external benchmarks (PTB-XL and CPSC 2018) and outperformed alternatives trained on larger datasets.

---

## 70. Variance-reduced extreme value index estimators using control variates in a semi-supervised setting

**论文链接:** [http://arxiv.org/abs/2511.15561v1](http://arxiv.org/abs/2511.15561v1)

**作者:** Louison Bocquet-Nouaille, Jérôme Morio, Benjamin Bobbia

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种基于控制变量的迁移学习方法，用于减少极端价值指数估计中的高方差问题，即使在目标分布和源分布尾部轻重程度不相似的情况下也能实现显著的方差减少。

### 背景

极端价值分析中的极端价值指数(EVI)估计存在高方差问题，因为它只依赖于少数极端观测值。

### 目的

开发一种在半监督框架下的迁移学习方法，结合少量成对的目标和源观测值与大量未配对的源数据，以减少EVI估计的方差而不引入偏差。

### 方法

将目标EVI的Hill估计器表示为均值的比率，对分子和分母应用近似控制变量，使用联合优化的系数来保证方差减少而不引入偏差。

### 主要发现

迁移的Hill估计器的渐近相对方差减少与目标变量和源变量之间的尾部依赖性成正比，与它们的EVI值无关，即使目标分布和源分布的尾部轻重程度不相似，也能实现显著的方差减少。

### 结论

该方法可以扩展到其他表示为均值比率的EVI估计器，如矩估计器，并在多保真度的水涌和冰积聚数据集上证明了其实用价值。

### 翻译

极端价值指数(EVI)的估计是极端价值分析的基础，但由于仅依赖于少数极端观测值而存在高方差问题。我们提出了一种基于控制变量的迁移学习方法，在半监督框架下，将少量成对的目标和源观测值与大量未配对的源数据相结合。通过将目标EVI的Hill估计器表示为均值的比率，我们对分子和分母都应用近似控制变量，使用联合优化的系数来保证方差减少而不引入偏差。我们从理论和模拟上证明，迁移的Hill估计器的渐近相对方差减少与目标变量和源变量之间的尾部依赖性成正比，且与它们的EVI值无关。因此，即使目标分布和源分布的尾部轻重程度不相似，也能实现显著的方差减少。该方法可以扩展到其他表示为均值比率的EVI估计器，如在矩估计器上所展示的那样。所提出方法的实用价值在多保真度的水涌和冰积聚数据集上得到了说明。


### 论文摘要

The estimation of the Extreme Value Index (EVI) is fundamental in extreme value analysis but suffers from high variance due to reliance on only a few extreme observations. We propose a control variates based transfer learning approach in a semi-supervised framework, where a small set of coupled target and source observations is combined with abundant unpaired source data. By expressing the Hill estimator of the target EVI as a ratio of means, we apply approximate control variates to both numerator and denominator, with jointly optimized coefficients that guarantee variance reduction without introducing bias. We show theoretically and through simulations that the asymptotic relative variance reduction of the transferred Hill estimator is proportional to the tail dependence between the target and source variables and independent of their EVI values. Thus, substantial variance reduction can be achieved even without similarity in tail heaviness of the target and source distributions. The proposed approach can be extended to other EVI estimators expressed with ratio of means, as demonstrated on the moment estimator. The practical value of the proposed method is illustrated on multi-fidelity water surge and ice accretion datasets.

---

## 71. A Review of Machine Learning for Cavitation Intensity Recognition in Complex Industrial Systems

**论文链接:** [http://arxiv.org/abs/2511.15497v1](http://arxiv.org/abs/2511.15497v1)

**作者:** Yu Sha, Ningtao Liu, Haofeng Liu, Junqi Tao, Zhenxing Niu, Guojun Huang, Yao Yao, Jiaqi Liang, Moxian Qian, Horst Stoecker, Domagoj Vnucec, Andreas Widl, Kai Zhou

**发布时间:** 2025-11-19

**备注:** 43 pages

### GPT解析

### 总结

这是一篇关于空化强度识别技术的综述论文，系统梳理了2002-2025年间CIR技术的发展历程，并提出了未来研究方向。

### 背景

空化强度识别是检测和评估水力机械中空化现象的关键技术，对复杂工业系统的运行安全、性能优化和维护成本降低具有重要意义。尽管已有大量研究，但仍缺乏系统性的综述来追踪发展轨迹并为未来研究提供明确指导。

### 目的

填补系统性综述的空白，通过全面回顾和分析2002-2025年间关于智能CIR的数百篇出版物，总结技术演变并为未来发展提供见解。

### 方法

对2002-2025年间各类机械设备上的智能CIR技术进行综述和分析，总结技术演变。

### 主要发现

早期阶段以传统机器学习方法为主，依赖领域专家知识指导的手工设计特征；深度学习的出现推动了端到端模型的发展，能够从多源信号中自动提取特征，显著提高了识别性能和鲁棒性；最近，物理信息诊断模型被提出，将领域知识嵌入深度学习模型中，提高了可解释性和跨条件泛化能力。

### 结论

未来，迁移学习、多模态融合、轻量级网络架构和工业代理的部署有望推动CIR技术进入新阶段，解决多源数据获取、标准化评估和工业实施方面的挑战。论文旨在系统概述CIR技术的演变，并强调深度学习与物理知识融合的新趋势，为复杂工业系统智能空化诊断领域的研究人员和从业者提供重要参考。

### 翻译

空化强度识别是检测和评估水力机械中空化现象的关键技术，对复杂工业系统的运行安全、性能优化和维护成本降低具有重要意义。尽管研究取得实质性进展，但仍缺乏系统性综述来系统追踪发展轨迹并为未来研究提供明确指导。为填补这一空白，本文对2002-2025年间各类机械设备上的智能CIR技术进行了全面回顾和分析，总结了技术演变并为未来发展提供见解。早期阶段以传统机器学习方法为主，依赖领域专家知识指导的手工设计特征。深度学习的出现推动了端到端模型的发展，能够从多源信号中自动提取特征，显著提高了识别性能和鲁棒性。最近，物理信息诊断模型被提出，将领域知识嵌入深度学习模型中，提高了可解释性和跨条件泛化能力。未来，迁移学习、多模态融合、轻量级网络架构和工业代理的部署有望推动CIR技术进入新阶段，解决多源数据获取、标准化评估和工业实施方面的挑战。本文旨在系统概述CIR技术的演变，并强调深度学习与物理知识融合的新趋势，为复杂工业系统智能空化诊断领域的研究人员和从业者提供重要参考。


### 论文摘要

Cavitation intensity recognition (CIR) is a critical technology for detecting and evaluating cavitation phenomena in hydraulic machinery, with significant implications for operational safety, performance optimization, and maintenance cost reduction in complex industrial systems. Despite substantial research progress, a comprehensive review that systematically traces the development trajectory and provides explicit guidance for future research is still lacking. To bridge this gap, this paper presents a thorough review and analysis of hundreds of publications on intelligent CIR across various types of mechanical equipment from 2002 to 2025, summarizing its technological evolution and offering insights for future development. The early stages are dominated by traditional machine learning approaches that relied on manually engineered features under the guidance of domain expert knowledge. The advent of deep learning has driven the development of end-to-end models capable of automatically extracting features from multi-source signals, thereby significantly improving recognition performance and robustness. Recently, physical informed diagnostic models have been proposed to embed domain knowledge into deep learning models, which can enhance interpretability and cross-condition generalization. In the future, transfer learning, multi-modal fusion, lightweight network architectures, and the deployment of industrial agents are expected to propel CIR technology into a new stage, addressing challenges in multi-source data acquisition, standardized evaluation, and industrial implementation. The paper aims to systematically outline the evolution of CIR technology and highlight the emerging trend of integrating deep learning with physical knowledge. This provides a significant reference for researchers and practitioners in the field of intelligent cavitation diagnosis in complex industrial systems.

---

## 72. LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering

**论文链接:** [http://arxiv.org/abs/2511.15424v1](http://arxiv.org/abs/2511.15424v1)

**作者:** Yuanjie Zhu, Liangwei Yang, Ke Xu, Weizhi Zhang, Zihe Song, Jindong Wang, Philip S. Yu

**发布时间:** 2025-11-19

### GPT解析

### 总结

大型语言模型(LLMs)通过其深度的语义理解能力改变无监督学习，特别是在文本聚类方面。然而直接应用受到缺乏状态记忆和难以管理聚类粒度的限制。作者提出LLM-MemCluster框架，将聚类重新概念化为完全基于LLM的任务，利用动态内存和双重提示策略解决上述问题，并在基准数据集上显著优于现有方法。

### 背景

大型语言模型(LLMs)正在通过其深度的语义理解能力改变无监督学习，特别是在文本聚类方面展现出前所未有的能力。

### 目的

解决直接应用LLMs进行文本聚类时的两个主要限制：缺乏状态记忆进行迭代精化，以及难以管理聚类粒度问题，提供一个真正的端到端解决方案。

### 方法

作者提出了LLM-MemCluster框架，这是一个新颖的框架，将聚类重新概念化为完全基于LLM的任务。该框架利用动态内存来引入状态感知，并采用双重提示策略使模型能够推理并确定聚类数量。

### 主要发现

在几个基准数据集上的评估表明，LLM-MemCluster这种无需调整的框架显著且一致地优于强基线方法，为基于LLM的文本聚类提供了一种有效、可解释且真正的端到端范式。

### 结论

LLM-MemCluster为基于LLM的文本聚类提供了一种有效、可解释且真正的端到端范式。

### 翻译

大型语言模型(LLMs)正在通过其深度的语义理解能力重塑无监督学习，提供了前所未有的文本聚类能力。然而，直接应用受到缺乏状态记忆进行迭代精化和难以管理聚类粒度的根本性限制。因此，现有方法通常依赖带有外部模块的复杂管道，牺牲了真正的端到端方法。我们介绍了LLM-MemCluster，一个新颖的框架，将聚类重新概念化为完全基于LLM的任务。它利用动态内存来引入状态感知，并采用双重提示策略使模型能够推理并确定聚类数量。在几个基准数据集上的评估表明，我们这种无需调整的框架显著且一致地优于强基线方法。LLM-MemCluster为基于LLM的文本聚类提供了一种有效、可解释且真正的端到端范式。


### 论文摘要

Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.

---

## 73. IPR-1: Interactive Physical Reasoner

**论文链接:** [http://arxiv.org/abs/2511.15407v1](http://arxiv.org/abs/2511.15407v1)

**作者:** Mingyu Zhang, Lifeng Zhuo, Tianxi Tan, Guocan Xie, Xian Nie, Yan Li, Renjie Zhao, Zizhu He, Ziyu Wang, Jiting Cai, Yong-Lu Li

**发布时间:** 2025-11-19

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

本研究提出了一种名为IPR（Interactive Physical Reasoner）的智能体，它能够通过与环境的交互获取类似人类的物理推理能力，并在更多经验中持续改进。研究在Game-to-Unseen（G2U）场景下进行，涉及1000多种具有不同物理和因果机制的多样化游戏。

### 背景

人类通过观察、与环境的互动以及内化物理和因果关系来学习。论文探讨了智能体是否能够通过类似的交互方式获取类似人类的推理能力，并随着经验积累而不断改进。

### 目的

研究智能体是否能够通过交互获得类似人类的推理能力，并随着经验积累而持续改进。

### 方法

在Game-to-Unseen（G2U）场景下进行研究，整理了1000多种具有不同物理和因果机制的多样化游戏；在三个类别人类水平上评估：生存、好奇心、效用；提出IPR（Interactive Physical Reasoner），使用世界模型滚动评分和强化VLM的策略；引入PhysCode，一种以物理为中心的行动代码，将语义意图与动力学对齐，为预测和推理提供共享的行动空间；在1000多种游戏上进行预训练。

### 主要发现

VLM/VLA智能体能够推理但在交互环境中缺乏前瞻性；世界模型能够想象但模仿视觉模式而非分析物理和因果关系；IPR在三个水平上表现稳健，总体上与GPT-5相当，在好奇心方面超越GPT-5；随着更多训练游戏和交互步骤的增加，性能有所提高；模型能够零样本迁移到未见过的游戏。

### 结论

以物理为中心的交互是持续改进物理推理的一条有效途径。

### 翻译

人类通过观察、与环境互动以及内化物理和因果关系来学习。在这里，我们旨在探讨一个智能体是否能够通过交互获得类似人类的推理能力，并随着更多经验而不断改进。我们在Game-to-Unseen（G2U）场景下研究这个问题，整理了1000多种具有不同物理和因果机制的多样化游戏，并在三个类别人类水平上评估：从原始直觉到目标驱动推理的生存、好奇心和效用。我们的分析揭示了互补的失败：VLM/VLA智能体能够推理但在交互环境中缺乏前瞻性，而世界模型能够想象但模仿视觉模式而非分析物理和因果关系。因此，我们提出了IPR（Interactive Physical Reasoner），使用世界模型滚动评分和强化VLM的策略，并引入PhysCode，一种以物理为中心的行动代码，将语义意图与动力学对齐，为预测和推理提供共享的行动空间。在1000多种游戏上预训练后，我们的IPR在三个水平上表现稳健，总体上与GPT-5相当，并在好奇心方面超越它。我们发现性能随着更多训练游戏和交互步骤的增加而提高，而且该模型还能零样本迁移到未见过的游戏。这些结果支持以物理为中心的交互作为持续改进物理推理的一条途径。


### 论文摘要

Humans learn by observing, interacting with environments, and internalizing physics and causality. Here, we aim to ask whether an agent can similarly acquire human-like reasoning from interaction and keep improving with more experience. We study this in a Game-to-Unseen (G2U) setting, curating 1,000+ heterogeneous games with diverse physical and causal mechanisms, and evaluate at three human-like levels: Survival, Curiosity, Utility, from primitive intuition to goal-driven reasoning. Our analysis reveals complementary failures: VLM/VLA agents reason but lack look-ahead in interactive settings, while world models imagine but imitate visual patterns rather than analyze physics and causality. We therefore propose IPR (Interactive Physical Reasoner), using world-model rollouts to score and reinforce a VLM's policy, and introduce PhysCode, a physics-centric action code aligning semantic intent with dynamics to provide a shared action space for prediction and reasoning. Pretrained on 1,000+ games, our IPR performs robustly on three levels, matches GPT-5 overall, and surpasses it on Curiosity. We find that performance improves with more training games and interaction steps, and that the model also zero-shot transfers to unseen games. These results support physics-centric interaction as a path to steadily improving physical reasoning.

---

## 74. Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception

**论文链接:** [http://arxiv.org/abs/2511.15279v1](http://arxiv.org/abs/2511.15279v1)

**作者:** Jiashu Yang, Yifan Han, Yucheng Xie, Ning Guo, Wenzhao Lian

**发布时间:** 2025-11-19

### GPT解析

### 总结

EyeVLA是一种用于主动视觉感知的机器人眼球系统，能够根据指令主动采取行动，在广域覆盖与细粒度细节获取之间取得平衡，实现强大的环境感知能力。

### 背景

在具身AI感知系统中，视觉感知应该是主动的而非被动处理静态图像。现有视觉模型和固定的RGB-D相机系统无法在广域覆盖与细粒度细节获取之间取得平衡，限制了它们在开放世界机器人应用中的效能。

### 目的

解决现有系统在广域覆盖与细粒度细节获取之间的矛盾，提出一种能够根据指令主动采取行动的机器人眼球系统，实现对细粒度目标对象和广泛空间范围内详细信息的清晰观测。

### 方法

提出EyeVLA系统，将行动行为离散化为行动令牌并与视觉-语言模型集成，在单个自回归序列中实现视觉、语言和行动的联合建模；使用2D边界框坐标引导推理链，应用强化学习优化视点选择策略，仅使用少量真实世界数据将VLM的开放场景理解能力转移到视觉语言行动策略中。

### 主要发现

实验表明，该系统能够在真实环境中高效执行指令场景，通过旋转和缩放的指令驱动行动主动获取更准确的视觉信息，实现了强大的环境感知能力。

### 结论

EyeVLA引入了一种新颖的机器人视觉系统，利用详细且空间丰富的大规模具身数据，主动获取对下游具身任务高度有价值的视觉观测。

### 翻译

在具身AI感知系统中，视觉感知应该是主动的：目标不是被动处理静态图像，而是在像素和空间预算约束下主动获取信息量更大的数据。现有的视觉模型和固定的RGB-D相机系统无法从根本上协调广域覆盖与细粒度细节获取，严重限制了它们在开放世界机器人应用中的效能。为解决这个问题，我们提出了EyeVLA，这是一种用于主动视觉感知的机器人眼球，可以根据指令采取主动行动，实现对细粒度目标对象和广泛空间范围内详细信息的清晰观测。EyeVLA将行动行为离散化为行动令牌，并将其与具有强大开放世界理解能力的视觉-语言模型集成，在单个自回归序列中实现视觉、语言和行动的联合建模。通过使用2D边界框坐标引导推理链，并应用强化学习优化视点选择策略，我们仅使用少量真实世界数据，将VLM的开放场景理解能力转移到视觉语言行动策略中。实验表明，我们的系统能够在真实环境中高效执行指令场景，并通过旋转和缩放的指令驱动行动主动获取更准确的视觉信息，从而实现强大的环境感知能力。EyeVLA引入了一种新颖的机器人视觉系统，它利用详细且空间丰富的大规模具身数据，主动获取对下游具身任务高度有价值的视觉观测。


### 论文摘要

In embodied AI perception systems, visual perception should be active: the goal is not to passively process static images, but to actively acquire more informative data within pixel and spatial budget constraints. Existing vision models and fixed RGB-D camera systems fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. To address this issue, we propose EyeVLA, a robotic eyeball for active visual perception that can take proactive actions based on instructions, enabling clear observation of fine-grained target objects and detailed information across a wide spatial extent. EyeVLA discretizes action behaviors into action tokens and integrates them with vision-language models (VLMs) that possess strong open-world understanding capabilities, enabling joint modeling of vision, language, and actions within a single autoregressive sequence. By using the 2D bounding box coordinates to guide the reasoning chain and applying reinforcement learning to refine the viewpoint selection policy, we transfer the open-world scene understanding capability of the VLM to a vision language action (VLA) policy using only minimal real-world data. Experiments show that our system efficiently performs instructed scenes in real-world environments and actively acquires more accurate visual information through instruction-driven actions of rotation and zoom, thereby achieving strong environmental perception capabilities. EyeVLA introduces a novel robotic vision system that leverages detailed and spatially rich, large-scale embodied data, and actively acquires highly informative visual observations for downstream embodied tasks.

---

## 75. The Walls Have Ears: Unveiling Cross-Chain Sandwich Attacks in DeFi

**论文链接:** [http://arxiv.org/abs/2511.15245v1](http://arxiv.org/abs/2511.15245v1)

**作者:** Chuanlei Li, Zhicheng Sun, Jing Xin Yuu, Xuechao Wang

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究揭示了跨链桥协议中的关键安全漏洞，即跨链三明治攻击，攻击者可以利用源链事件获取目标链交易信息，从而获利超过527万美元。

### 背景

跨链互操作性是现代区块链基础设施的核心组成部分，但跨链消息的透明性可能意外暴露敏感交易信息，为攻击者创造利用价值的机会。

### 目的

研究跨链三明治攻击针对流动性池-based跨链桥协议的机制，并量化其威胁。

### 方法

进行实证研究，使用两个月（2025年8月10日至10月10日）的Symbiosis协议跨链交易数据，以及定制的启发式检测模型进行分析。

### 主要发现

发现攻击者可以利用源链发出的事件了解目标链交易详情，利用信息优势放置前置和后置交易，且当前的三明治攻击防御措施对这种跨链变种无效；分析发现攻击者集体获利超过527万美元，相当于总桥接量的1.28%。

### 结论

跨链桥协议存在严重的安全漏洞，需要开发新的防御措施来应对这种跨链三明治攻击。

### 翻译

跨链互操作性是现代区块链基础设施的核心组成部分，使资产能够在多个区块链生态系统之间无缝转移和组合应用。然而，跨链消息的透明性可能会意外暴露敏感交易信息，为攻击者通过操纵或前置策略利用价值创造机会。在这项工作中，我们研究针对基于流动性池的跨链桥协议的跨链三明治攻击。我们发现了一个关键漏洞，攻击者可以利用源链发出的事件来了解目标链上的交易详情，这些详情在目标链内存池中出现之前就已存在。这种信息优势使攻击者能够战略性地放置前置和后置交易，确保他们的前置交易总是优先于监控目标链内存池的现有MEV机器人。此外，当前的三明治攻击防御措施对这种新的跨链变种无效。为了量化这一威胁，我们使用Symbiosis协议两个月的跨链交易数据（2025年8月10日至10月10日）和定制的启发式检测模型进行了实证研究。我们的分析发现，攻击者集体获利超过527万美元，相当于总桥接量的1.28%。


### 论文摘要

Cross-chain interoperability is a core component of modern blockchain infrastructure, enabling seamless asset transfers and composable applications across multiple blockchain ecosystems. However, the transparency of cross-chain messages can inadvertently expose sensitive transaction information, creating opportunities for adversaries to exploit value through manipulation or front-running strategies.   In this work, we investigate cross-chain sandwich attacks targeting liquidity pool-based cross-chain bridge protocols. We uncover a critical vulnerability where attackers can exploit events emitted on the source chain to learn transaction details on the destination chain before they appear in the destination chain mempool. This information advantage allows attackers to strategically place front-running and back-running transactions, ensuring that their front-running transactions always precede those of existing MEV bots monitoring the mempool of the destination chain. Moreover, current sandwich-attack defenses are ineffective against this new cross-chain variant. To quantify this threat, we conduct an empirical study using two months (August 10 to October 10, 2025) of cross-chain transaction data from the Symbiosis protocol and a tailored heuristic detection model. Our analysis identifies attacks that collectively garnered over \(5.27\) million USD in profit, equivalent to 1.28\% of the total bridged volume.

---

## 76. Testing relevant difference in high-dimensional linear regression with applications to detect transferability

**论文链接:** [http://arxiv.org/abs/2511.15236v1](http://arxiv.org/abs/2511.15236v1)

**作者:** Xu Liu

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了一种在高维线性回归模型中检验系数β的新方法，不同于传统的显著性检验，而是关注β与0之间是否有实质性差异，以解决迁移学习中源数据可转移性问题。

### 背景

大多数研究关注高维线性回归模型中系数β的显著性检验，采用传统的假设检验问题形式H₀ᶜ: β=0 vs H₁ᶜ: β≠0。

### 目的

研究零假设为β和0之间没有实质性差异的检验问题，即H₀: ||β||≤δ₀ vs H₁: ||β||>δ₀，其中δ₀是预先指定的小常数。这种检验问题源于在迁移学习框架中检测源数据可转移性的迫切需求。

### 方法

提出了一种新的检验程序，结合了高维协方差矩阵最大特征值的估计，并借助随机矩阵理论。在高维 nuisance 参数存在的情况下，建立了所提出检验统计量的渐近正态性。

### 主要发现

通过应用所提出的检验方法检测源数据的可转移性，统一的迁移学习模型相比现有方法同时实现了更低的估计和预测误差。通过模拟研究验证了新检验的有限样本特性。

### 结论

所提出的新检验方法在高维线性回归模型中有效，能够检测β与0之间的实质性差异，并在迁移学习应用中表现出优越的性能。

### 翻译

大多数关于高维线性回归模型中系数β显著性检验的研究人员考虑的是经典的假设检验问题H₀ᶜ: β=0 vs H₁ᶜ: β≠0。我们采取了不同的视角，研究了零假设为β和0之间没有实质性差异的检验问题，即H₀: ||β||≤δ₀ vs H₁: ||β||>δ₀，其中δ₀是预先指定的小常数。这种检验问题源于在迁移学习框架中检测源数据可转移性的迫切需求。我们提出了一种新的检验程序，结合了高维协方差矩阵最大特征值的估计，并借助随机矩阵理论。在高维 nuisance 参数存在的情况下，我们在零假设和备择假设下建立了所提出检验统计量的渐近正态性。通过应用所提出的检验方法检测源数据的可转移性，统一的迁移学习模型相比现有方法同时实现了更低的估计和预测误差。我们通过模拟研究研究了新检验的有限样本特性，并通过分析GTEx数据说明了其性能。


### 论文摘要

Most of researchers on testing a significance of coefficient $\ubeta$ in high-dimensional linear regression models consider the classical hypothesis testing problem $H_0^{c}: \ubeta=\uzero \mbox{ versus } H_1^{c}: \ubeta \neq \uzero$. We take a different perspective and study the testing problem with the null hypothesis of no relevant difference between $\ubeta$ and $\uzero$, that is, $H_0: \|\ubeta\|\leq δ_0 \mbox{ versus } H_1: \|\ubeta\|> δ_0$, where $δ_0$ is a prespecified small constant. This testing problem is motivated by the urgent requirement to detect the transferability of source data in the transfer learning framework. We propose a novel test procedure incorporating the estimation of the largest eigenvalue of a high-dimensional covariance matrix with the assistance of the random matrix theory. In the more challenging setting in the presence of high-dimensional nuisance parameters, we establish the asymptotic normality for the proposed test statistics under both the null and alternative hypotheses. By applying the proposed test approaches to detect the transferability of source data, the unified transfer learning models simultaneously achieve lower estimation and prediction errors with comparison to existing methods. We study the finite-sample properties of the new test by means of simulation studies and illustrate its performance by analyzing the GTEx data.

---

## 77. Effective Code Membership Inference for Code Completion Models via Adversarial Prompts

**论文链接:** [http://arxiv.org/abs/2511.15107v1](http://arxiv.org/abs/2511.15107v1)

**作者:** Yuan Jiang, Zehao Li, Shan Huang, Christoph Treude, Xiaohong Su, Tiantian Wang

**发布时间:** 2025-11-19

### GPT解析

### 总结

论文提出了AdvPrompt-MIA方法，这是一种针对代码补全模型的成员推理攻击方法，结合了代码特定的对抗性扰动和深度学习技术，能够更准确地推断训练数据集中是否包含给定的代码片段。

### 背景

现有的黑盒和灰盒成员推理攻击依赖于昂贵的代理模型或手动设计的启发式规则，这些方法难以捕捉过参数化代码语言模型所展示的细微记忆模式。

### 目的

提出一种专门针对代码补全模型的方法，能够更准确地推断训练数据集中是否包含给定的代码片段，以评估隐私风险。

### 方法

AdvPrompt-MIA设计了一系列对抗性提示，诱导目标代码模型的输出变化，通过将这些输出与真实完成进行比较，构建特征向量来训练分类器，从而自动区分成员和非成员样本。

### 主要发现

在Code Llama 7B模型上对APPS和HumanEval基准进行的综合评估表明，该方法一致优于最先进的基线，AUC增益最高达102%。此外，该方法在不同模型和数据集上表现出强大的可转移性。

### 结论

AdvPrompt-MIA能够捕获更丰富的记忆模式，准确推断训练集成员身份，具有实用性和通用性。

### 翻译

对代码补全模型的成员推理攻击提供了一种有效的方法，通过推断给定的代码片段是否是训练数据的一部分来评估隐私风险。现有的黑盒和灰盒成员推理攻击依赖于昂贵的代理模型或手动设计的启发式规则，这限制了它们捕捉过参数化代码语言模型所展示的细微记忆模式的能力。为了解决这些挑战，我们提出了AdvPrompt-MIA，这是一种专门为代码补全模型设计的方法，结合了代码特定的对抗性扰动和深度学习。我们方法的核心创新在于设计一系列对抗性提示，这些提示会诱导目标代码模型的输出发生变化。通过将这些输出与真实完成进行比较，我们构建特征向量来训练分类器，自动区分成员和非成员样本。这种设计使我们的方法能够捕获更丰富的记忆模式，并准确推断训练集成员身份。我们在广泛采用的模型（如Code Llama 7B）上对APPS和HumanEval基准进行了全面评估。结果表明，我们的方法一致优于最先进的基线，AUC增益最高达102%。此外，我们的方法在不同模型和数据集上表现出强大的可转移性，强调了其实用性和通用性。


### 论文摘要

Membership inference attacks (MIAs) on code completion models offer an effective way to assess privacy risks by inferring whether a given code snippet was part of the training data. Existing black- and gray-box MIAs rely on expensive surrogate models or manually crafted heuristic rules, which limit their ability to capture the nuanced memorization patterns exhibited by over-parameterized code language models. To address these challenges, we propose AdvPrompt-MIA, a method specifically designed for code completion models, combining code-specific adversarial perturbations with deep learning. The core novelty of our method lies in designing a series of adversarial prompts that induce variations in the victim code model's output. By comparing these outputs with the ground-truth completion, we construct feature vectors to train a classifier that automatically distinguishes member from non-member samples. This design allows our method to capture richer memorization patterns and accurately infer training set membership. We conduct comprehensive evaluations on widely adopted models, such as Code Llama 7B, over the APPS and HumanEval benchmarks. The results show that our approach consistently outperforms state-of-the-art baselines, with AUC gains of up to 102%. In addition, our method exhibits strong transferability across different models and datasets, underscoring its practical utility and generalizability.

---

## 78. Transferable potential for molecular dynamics simulations of borosilicate glasses and structural comparison of machine learning optimized parameters

**论文链接:** [http://arxiv.org/abs/2511.14982v1](http://arxiv.org/abs/2511.14982v1)

**作者:** Kai Yang, Ruoxia Chen, Anders K. R. Christensen, Mathieu Bauchy, N. M. Anoop Krishnan, Morten M. Smedskjaer, Fabian Rosner

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究人员开发了一种机器学习优化的经典势函数，用于模拟不同成分的硼硅酸盐玻璃，实现了跨成分的可转移性，并准确预测了玻璃的结构特性。

### 背景

硼硅酸盐玻璃的模拟具有挑战性，因为硼原子的配位状态依赖于玻璃成分和温度。

### 目的

开发一种新的机器学习优化的经典势函数，用于分子动力学模拟，实现跨不同硼硅酸盐玻璃成分的可转移性。

### 方法

开发了一种机器学习优化的经典势函数，专注于密度和四配位硼分数的优化，并通过分子动力学模拟验证了实验X射线结构因子数据。

### 主要发现

该势函数能准确预测不同玻璃成分中短程和中程有序性的玻璃结构变化；研究了力场公式中经验参数对微观键长、键角和宏观密度的影响；提供了原子间势函数与块状玻璃行为之间关系的新见解。

### 结论

成功开发了具有跨成分可转移性的机器学习优化经典势函数，能够准确预测硼硅酸盐玻璃的结构特性。

### 翻译

硼硅酸盐玻璃的模拟具有挑战性，因为硼原子的配位状态依赖于成分和温度。在此，我们提出了一种新开发的机器学习优化经典势函数，用于分子动力学模拟，实现了跨不同硼硅酸盐玻璃成分的可转移性。我们的势函数准确预测了不同玻璃成分中短程和中程有序性的玻璃结构变化，包括将我们的势函数与实验X射线结构因子数据进行验证。值得注意的是，这些数据未包含在优化框架中，该框架仅专注于密度和四配位硼分数。我们进一步研究了力场公式中经验参数对微观键长、键角和宏观密度的影响，为原子间势函数与块状玻璃行为之间的关系提供了新见解。


### 论文摘要

The simulation of borosilicate glasses is challenging due to the composition and temperature dependent coordination state of boron atoms. Here, we present a newly developed machine learning optimized classical potential for molecular dynamics simulations that achieves transferability across diverse borosilicate glass compositions. Our potential accurately predicts the glass structural variations in short- and medium-range order in different glass compositions, including validating our potential against experimental X-ray structure factor data. Notably, these data are not included in the optimization framework, which focuses exclusively on density and four-fold coordinated boron fraction. We further investigate the impact of empirical parameters in the force field formulation on the microscopic bond lengths, bond angles and the macroscopic densities, providing new insights into the relationship between interatomic potentials and bulk glass behaviors.

---

## 79. Logit-Based Losses Limit the Effectiveness of Feature Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2511.14981v1](http://arxiv.org/abs/2511.14981v1)

**作者:** Nicholas Cooper, Lijun Chen, Sailesh Dwivedy, Danna Gurari

**发布时间:** 2025-11-18

**备注:** NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps), December 2025

### GPT解析

### 总结

本文提出了一种仅使用基于特征的损失函数（不使用基于logits的损失函数如交叉熵）的知识蒸馏框架，通过引入知识质量指标识别最有效的教师层，在多个数据集上实现了高达15%的top-1准确率提升。

### 背景

知识蒸馏方法用于将参数量大的教师模型知识转移到轻量级学生模型，当前特征KD方法主要基于logits和中间层特征的损失函数。

### 目的

开发一种仅使用基于特征的损失函数来训练学生骨干网络的KD框架，无需依赖基于logits的损失函数。

### 方法

利用潜在表示几何形状的最新研究成果，引入知识质量指标识别提供最有效知识的教师层，构建纯特征KD框架。

### 主要发现

在三个图像分类数据集上使用四种不同学生-教师对（包括CNN和Vision Transformer）的实验表明，该方法达到最先进性能，top-1准确率提升最高达15%。

### 结论

纯特征KD框架优于传统方法，无需交叉熵等基于logits的损失函数即可实现显著性能提升，作者已公开代码供未来研究使用。

### 翻译

知识蒸馏(KD)方法可以将参数量大的教师模型的知识转移到轻量级的学生模型上。目前特征KD方法的标准是利用基于logits（即预softmax类别分数）和中间层特征（即潜在表示）的损失函数。与以往方法不同，我们提出了一种仅使用基于特征的损失函数（即不包括交叉熵等基于logits的损失函数）来训练学生骨干网络的KD框架。利用关于潜在表示几何形状的最新研究成果，我们引入了一个知识质量指标，用于识别哪些教师层提供最有效的知识蒸馏。在三个图像分类数据集上使用四种不同的学生-教师对（包括卷积神经网络和视觉变换器）进行的实验表明，我们的KD方法达到了最先进的性能，与标准方法相比，top-1准确度提高了高达15%。我们公开分享了代码，以促进未来的研究工作，网址为https://github.com/Thegolfingocto/KD_wo_CE。


### 论文摘要

Knowledge distillation (KD) methods can transfer knowledge of a parameter-heavy teacher model to a light-weight student model. The status quo for feature KD methods is to utilize loss functions based on logits (i.e., pre-softmax class scores) and intermediate layer features (i.e., latent representations). Unlike previous approaches, we propose a feature KD framework for training the student's backbone using feature-based losses exclusively (i.e., without logit-based losses such as cross entropy). Leveraging recent discoveries about the geometry of latent representations, we introduce a knowledge quality metric for identifying which teacher layers provide the most effective knowledge for distillation. Experiments on three image classification datasets with four diverse student-teacher pairs, spanning convolutional neural networks and vision transformers, demonstrate our KD method achieves state-of-the-art performance, delivering top-1 accuracy boosts of up to 15% over standard approaches. We publically share our code to facilitate future work at https://github.com/Thegolfingocto/KD_wo_CE.

---

## 80. Quality-Controlled Multimodal Emotion Recognition in Conversations with Identity-Based Transfer Learning and MAMBA Fusion

**论文链接:** [http://arxiv.org/abs/2511.14969v1](http://arxiv.org/abs/2511.14969v1)

**作者:** Zanxu Wang, Homayoon Beigi

**发布时间:** 2025-11-18

**DOI:** 10.13140/RG.2.2.33632.55045

**备注:** 8 pages, 14 images, 3 tables, Recognition Technologies, Inc. Technical Report RTI-20251118-01

### GPT解析

### 总结

论文通过系统质量控制和多阶段迁移学习解决了对话多模态情感识别中的数据质量问题，在MELD和IEMOCAP数据集上实现了高准确率。

### 背景

对话多模态情感识别领域存在数据质量问题，需要通过质量控制方法解决。

### 目的

解决对话多模态情感识别中的数据质量问题，提高情感识别的准确率。

### 方法

实现质量控制流程验证说话人身份、音频文本对齐和人脸检测；利用迁移学习提取说话人和面部嵌入；微调MPNet-v2获取情感感知文本表示；通过情感特定MLP适应特征；使用MAMBA进行三模态融合。

### 主要发现

基于MAMBA的三模态融合在MELD上达到64.8%的准确率，在IEMOCAP上达到74.3%的准确率；结合基于身份的音频和视觉嵌入与情感调整的文本表示能够提供有竞争力的性能。

### 结论

在质量受控的数据子集上结合不同模态的特征能够有效提升对话多模态情感识别性能，并为低频情感类别的进一步改进提供基础。

### 翻译

这篇论文通过系统质量控制和多阶段迁移学习解决对话多模态情感识别中的数据质量问题。我们为MELD和IEMOCAP数据集实现了质量控制流程，验证说话人身份、音频文本对齐和人脸检测。我们利用说话人和人脸识别的迁移学习，假设身份判别性嵌入不仅捕获稳定的声学和面部特征，还捕获情感表达的特定个人模式。我们使用RecoMadeEasy(R)引擎提取512维说话人和面部嵌入，微调MPNet-v2用于情感感知文本表示，并通过在单模态数据集上训练的情感特定MLP来适应这些特征。基于MAMBA的三模态融合在MELD上达到64.8%的准确率，在IEMOCAP上达到74.3%的准确率。这些结果表明，在质量受控的数据子集上结合基于身份的音频和视觉嵌入与情感调整的文本表示，能够为对话多模态情感识别提供一致的有竞争力的性能，并为在具有挑战性的低频情感类别上进一步改进提供基础。


### 论文摘要

This paper addresses data quality issues in multimodal emotion recognition in conversation (MERC) through systematic quality control and multi-stage transfer learning. We implement a quality control pipeline for MELD and IEMOCAP datasets that validates speaker identity, audio-text alignment, and face detection. We leverage transfer learning from speaker and face recognition, assuming that identity-discriminative embeddings capture not only stable acoustic and Facial traits but also person-specific patterns of emotional expression. We employ RecoMadeEasy(R) engines for extracting 512-dimensional speaker and face embeddings, fine-tune MPNet-v2 for emotion-aware text representations, and adapt these features through emotion-specific MLPs trained on unimodal datasets. MAMBA-based trimodal fusion achieves 64.8% accuracy on MELD and 74.3% on IEMOCAP. These results show that combining identity-based audio and visual embeddings with emotion-tuned text representations on a quality-controlled subset of data yields consistent competitive performance for multimodal emotion recognition in conversation and provides a basis for further improvement on challenging, low-frequency emotion classes.

---

## 81. Artificial intelligence approaches for energy-efficient laser cutting machines

**论文链接:** [http://arxiv.org/abs/2511.14952v1](http://arxiv.org/abs/2511.14952v1)

**作者:** Mohamed Abdallah Salem, Hamdy Ahmed Ashour, Ahmed Elshenawy

**发布时间:** 2025-11-18

**DOI:** 10.1109/IMSA58542.2023.10217625

### GPT解析

### 总结

本研究提出了一种创新的深度学习方法，通过闭环配置动态调整CO2激光切割中抽吸泵的功率，根据被切割材料和烟雾水平进行自适应控制，实现了显著的能源节约。

### 背景

激光切割过程中存在能源消耗大和环境影响严重的问题，目前CO2激光抽吸泵缺乏自适应控制，采用开环控制方式。

### 目的

开发新的深度学习方法，实现激光切割过程中的能源消耗减少，提高可持续性。

### 方法

采用闭环配置，根据被切割材料和烟雾水平动态调整泵功率；引入材料分类方法，包括无透镜散斑传感技术和USB摄像头结合VGG16迁移学习；使用单独的深度学习模型进行烟雾水平检测；整合系统使抽吸泵在非活动时间自动停止，操作期间动态调整功率。

### 主要发现

实验证明烟雾抽吸泵的能源消耗减少了20%至50%，显著提高了能源利用效率。

### 结论

该研究通过创新的深度学习方法实现了激光切割过程中的能源节约，对制造业的可持续发展做出了实质性贡献。

### 翻译

本研究通过提出创新的深度学习方法，解决激光切割过程中的能源消耗和环境影响重大挑战。认识到当前CO2激光抽吸泵缺乏自适应控制和开环特性，本研究利用闭环配置，根据被切割材料和产生的烟雾水平动态调整泵功率。为实现这一自适应系统，引入了多种材料分类方法，包括利用无透镜散斑传感技术和定制卷积神经网络(CNN)的方法，以及使用USB摄像头并通过预训练的VGG16 CNN模型进行迁移学习的方法。此外，还采用单独的深度学习模型进行烟雾水平检测，同时优化泵的功率输出。这种整合使排气抽吸泵在非活动时间自动停止，并在操作期间动态调整功率，实验证明并实现了显著的能源节约，结果显示烟雾抽吸泵的能源消耗减少了20%至50%，从而为制造业的可持续发展做出了重大贡献。


### 论文摘要

This research addresses the significant challenges of energy consumption and environmental impact in laser cutting by proposing novel deep learning (DL) methodologies to achieve energy reduction. Recognizing the current lack of adaptive control and the open-loop nature of CO2 laser suction pumps, this study utilizes closed-loop configurations that dynamically adjust pump power based on both the material being cut and the smoke level generated. To implement this adaptive system, diverse material classification methods are introduced, including techniques leveraging lens-less speckle sensing with a customized Convolutional Neural Network (CNN) and an approach using a USB camera with transfer learning via the pre-trained VGG16 CNN model. Furthermore, a separate DL model for smoke level detection is employed to simultaneously refine the pump's power output. This integration prompts the exhaust suction pump to automatically halt during inactive times and dynamically adjust power during operation, leading to experimentally proven and remarkable energy savings, with results showing a 20% to 50% reduction in the smoke suction pump's energy consumption, thereby contributing substantially to sustainable development in the manufacturing sector.

---

## 82. Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis

**论文链接:** [http://arxiv.org/abs/2511.14900v1](http://arxiv.org/abs/2511.14900v1)

**作者:** Zehao Liu, Wejieying Ren, Jipeng Zhang, Tianxiang Zhao, Jingxi Zhu, Xiaoting Li, Vasant G. Honavar

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了一种名为SkinR1的新型皮肤病视觉语言模型(VLM)，通过结合基于教科书的深度推理和强化学习的广泛泛化能力，解决了当前VLMs在皮肤病诊断中的三个主要限制：数据异质性、缺乏基础诊断理由以及有限的扩展性和泛化能力。

### 背景

视觉语言模型(VLMs)的出现为临床推理开辟了新可能性，并在皮肤病学诊断中展现出有前景的性能。然而，其可靠性和临床效用常受三个因素限制：数据异质性、缺乏基础诊断理由以及有限的扩展性和泛化能力。

### 目的

开发一种新型皮肤病VLM模型SkinR1，解决当前模型在皮肤病诊断中的局限性，提高诊断准确性和可靠性。

### 方法

提出SkinR1模型，采用统一端到端框架：(1)设计基于教科书的推理生成器，合成高保真、层次感知和鉴别诊断信息轨迹；(2)利用构建轨迹进行监督微调，赋予模型基础推理能力；(3)开发融入疾病层次结构的新型强化学习范式，将基础推理模式转移到大规模稀疏数据。

### 主要发现

在多个皮肤病数据集上的广泛实验表明，SkinR1实现了卓越的诊断准确性。消融研究证明了监督微调灌输的推理基础的重要性。

### 结论

SkinR1有效解决了皮肤病VLM中的关键挑战，通过结合基于教科书的推理和强化学习，显著提高了皮肤病诊断的准确性和可靠性。

### 翻译

视觉语言模型(VLMs)的出现为临床推理开辟了新的可能性，并在皮肤病学诊断中显示出有前景的性能。然而，它们的可靠性和临床效用通常受到三个主要因素的限制：(1)数据异质性，即多样化数据集缺乏一致的诊断标签和临床概念注释；(2)缺乏基础诊断理由，导致可靠的推理监督不足；(3)有限的扩展性和泛化能力，因为在小型密集注释数据集上训练的模型难以将细微推理转移到大型稀疏注释数据集。为解决这些局限性，我们提出了SkinR1，一种结合深度基于教科书的推理和强化学习(RL)广泛泛化能力的新型皮肤病VLM。SkinR1通过统一的端到端框架系统性地解决了关键挑战。首先，我们设计了一个基于教科书的推理生成器，合成高保真、层次感知和鉴别诊断(DDx)信息轨迹，提供可靠的专家级监督。其次，我们利用构建的轨迹进行监督微调(SFT)，赋予模型基础推理能力。第三，我们开发了一种新的RL范式，通过融入疾病的层次结构，有效将这些基础推理模式转移到大规模稀疏数据。在多个皮肤病数据集上的广泛实验表明，SkinR1实现了卓越的诊断准确性。消融研究证明了SFT灌输的推理基础的重要性。


### 论文摘要

The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones.   To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT.

---

## 83. 论文ID: 2511.14897v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.14897v1.json'

---

## 84. Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections

**论文链接:** [http://arxiv.org/abs/2511.15412v1](http://arxiv.org/abs/2511.15412v1)

**作者:** Evgenii Vinogradov, Aymen Fakhreddine, Abdul Saboor, Sergi Abadal, Sofie Pollin

**发布时间:** 2025-11-19

**备注:** International Conference on Computing, Networking and Communications (ICNC 2026)

### GPT解析

### 总结

本文提出了一种用于无人机辅助网络中空间一致的半确定性空对地信道建模方法，通过高效3D建筑阴影投影确定LOS区域，结合LOS感知的确定性路径损耗与随机阴影衰落，生成适合环境感知和移动感知信道评估的A2G无线电地图。

### 背景

无人机辅助网络中的空对地信道建模面临挑战，需要一种既能反映真实传播特性又具有计算效率的方法。

### 目的

开发一种空间一致的半确定性A2G信道建模方法，用于环境感知和移动感知的信道评估和性能预测。

### 方法

使用高效的3D建筑阴影投影确定LOS区域，快速生成LOS地图，结合LOS感知的确定性路径损耗与随机阴影衰落，产生空间一致的A2G无线电地图。

### 主要发现

在符合ITU标准的曼哈顿网格环境中的仿真结果表明，该模型能够反映关键的都市传播特性，如LOS阻塞模式和中断行为。

### 结论

提出的方法是对射线追踪或完全随机模型的有效替代方案，特别适用于6G非陆地网络中的用户移动性、链路规划和无线电图生成。

### 翻译

我们提出了一种用于无人机辅助网络中空间一致的半确定性空对地信道建模方法。我们使用高效的3D建筑阴影投影来确定视距区域，从而能够快速生成LOS地图。通过将LOS感知的确定性路径损耗与随机阴影衰落相结合，该方法产生了适合环境感知和移动感知的信道评估及性能预测的空间一致A2G无线电地图。在符合ITU标准的曼哈顿网格环境中的仿真结果表明，该模型能够反映关键的都市传播特性，如LOS阻塞模式和中断行为。所提出的方法为射线追踪或完全随机模型提供了高效的替代方案，特别与6G非陆地网络中的用户移动性、链路规划和无线电图生成相关。


### 论文摘要

We present an approach for spatially-consistent semi-deterministic Air-to-Ground (A2G) channel modeling in Unmanned Aerial Vehicle-assisted networks. We use efficient 3D building shadow projections to determine Line-of-Sight (LOS) regions, enabling fast generation of LOS maps. By integrating LOS-aware deterministic path loss with stochastic shadow fading, the approach produces spatially consistent A2G radio maps suitable for environment- and mobility-aware channel evaluation and performance prediction. Simulation results in ITU-compliant Manhattan grid environments demonstrate the model's ability to reflect key urban propagation characteristics, such as LOS blockage patterns and outage behavior. The proposed approach provides an efficient alternative to ray tracing or fully stochastic models, with particular relevance for user mobility, link planning, and radio map generation in 6G non-terrestrial networks.

---

## 85. Transformer-Guided Deep Reinforcement Learning for Optimal Takeoff Trajectory Design of an eVTOL Drone

**论文链接:** [http://arxiv.org/abs/2511.14887v1](http://arxiv.org/abs/2511.14887v1)

**作者:** Nathan M. Roberts, Xiaosong Du

**发布时间:** 2025-11-18

**备注:** Conference version with 12 pages and 2 figures

### GPT解析

### 总结

本研究提出了一种transformer引导的深度强化学习方法，用于优化电动垂直起降(eVTOL)飞机的起飞轨迹，以最小化能耗。该方法显著提高了训练效率，并在能耗优化方面优于传统深度强化学习方法。

### 背景

电动垂直起降(eVTOL)飞机的快速发展为缓解城市交通拥堵提供了机会，但需要开发最优起飞轨迹以最小化能耗。传统最优控制方法受限于问题的维度和复杂性，而深度强化学习虽然能处理复杂非线性系统，但训练困难是主要瓶颈。

### 目的

开发一种transformer引导的深度强化学习方法，以解决eVTOL飞机起飞轨迹优化中的训练难题，实现最小能耗起飞。

### 方法

提出transformer引导的DRL方法，通过在每个时间步使用transformer探索真实状态空间来减轻训练难度。该方法应用于eVTOL无人机最优起飞轨迹设计，通过改变控制变量（功率和翼角）实现最小能耗起飞，同时满足起飞条件。

### 主要发现

transformer引导的DRL代理在4.57×10^6个时间步内学会起飞，而普通DRL需要19.79×10^6个时间步（节省了75%的训练时间）；transformer引导的DRL在最优能耗方面达到97.2%的准确率，而普通DRL为96.3%。

### 结论

transformer引导的DRL在训练效率和最优设计验证方面都优于普通DRL，为eVTOL飞机的起飞轨迹优化提供了更有效的解决方案。

### 翻译

电动垂直起降(eVTOL)飞机的快速发展为缓解城市交通拥堵提供了有希望的机会。因此，开发最小能耗的最优起飞轨迹对于eVTOL飞机的更广泛应用变得至关重要。传统的最优控制方法（如动态规划和线性二次调节器）提供了高效且成熟的解决方案，但受限于问题的维度和复杂性。深度强化学习(DRL)作为一种特殊的人工智能，能够处理复杂的非线性系统；然而，训练难度是限制DRL应用的关键瓶颈。为了解决这些挑战，我们提出了transformer引导的DRL方法，通过在每个时间步使用transformer探索真实状态空间来减轻训练难度。所提出的transformer引导的DRL在eVTOL无人机最优起飞轨迹设计上得到了验证，通过改变控制变量（即功率和垂直翼角）实现最小能耗起飞，同时满足起飞条件（即最小垂直位移和最小水平速度）。结果表明，transformer引导的DRL代理在4.57×10^6个时间步内学会了起飞，这仅是普通DRL代理所需的19.79×10^6个时间步的25%。此外，与基于仿真的最优参考相比，transformer引导的DRL在最优能耗方面实现了97.2%的准确率，而普通DRL实现了96.3%的准确率。因此，在训练效率和最优设计验证方面，所提出的transformer引导的DRL都优于普通DRL。


### 论文摘要

The rapid advancement of electric vertical take-off and landing (eVTOL) aircraft offers a promising opportunity to alleviate urban traffic congestion. Thus, developing optimal takeoff trajectories for minimum energy consumption becomes essential for broader eVTOL aircraft applications. Conventional optimal control methods (such as dynamic programming and linear quadratic regulator) provide highly efficient and well-established solutions but are limited by problem dimensionality and complexity. Deep reinforcement learning (DRL) emerges as a special type of artificial intelligence tackling complex, nonlinear systems; however, the training difficulty is a key bottleneck that limits DRL applications. To address these challenges, we propose the transformer-guided DRL to alleviate the training difficulty by exploring a realistic state space at each time step using a transformer. The proposed transformer-guided DRL was demonstrated on an optimal takeoff trajectory design of an eVTOL drone for minimal energy consumption while meeting takeoff conditions (i.e., minimum vertical displacement and minimum horizontal velocity) by varying control variables (i.e., power and wing angle to the vertical). Results presented that the transformer-guided DRL agent learned to take off with $4.57\times10^6$ time steps, representing 25% of the $19.79\times10^6$ time steps needed by a vanilla DRL agent. In addition, the transformer-guided DRL achieved 97.2% accuracy on the optimal energy consumption compared against the simulation-based optimal reference while the vanilla DRL achieved 96.3% accuracy. Therefore, the proposed transformer-guided DRL outperformed vanilla DRL in terms of both training efficiency as well as optimal design verification.

---

## 86. Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard

**论文链接:** [http://arxiv.org/abs/2511.14876v1](http://arxiv.org/abs/2511.14876v1)

**作者:** Henry Wong, Clement Fung, Weiran Lin, Karen Li, Stanley Chen, Lujo Bauer

**发布时间:** 2025-11-18

**备注:** 12 pages

### GPT解析

### 总结

这篇论文评估了对抗性例子对自动驾驶系统的实际影响，发现虽然某些攻击可以误导机器学习模型，但驾驶代理中的其他模块（如PID控制或基于GPS的规则）可能会覆盖这些被操纵的预测。

### 背景

自动驾驶系统使用机器学习模型、控制器逻辑和自定义模块的组合输出。虽然先前的研究表明对抗性例子可以误导自动驾驶环境中的机器学习模型，但这些攻击是否对各种代理、环境和场景产生有害的驾驶行动仍不清楚。

### 目的

评估对抗性例子对自动驾驶的风险，通过针对各种驾驶代理进行攻击评估，而不是孤立地针对机器学习模型。

### 方法

使用CARLA城市驾驶模拟器创建和评估对抗性例子，设计对抗性补丁来停止或引导驾驶代理，在运行时将它们流式传输到CARLA模拟器中，并针对来自CARLA排行榜的代理进行评估。评估攻击时不创建或修改任何驾驶代理代码，而是针对包含机器学习模型的所有代理部分。

### 主要发现

尽管某些攻击可以成功地误导机器学习模型，预测错误的停止或转向命令，但一些驾驶代理使用的模块（如PID控制或基于GPS的规则）可以覆盖攻击者操纵的机器学习模型预测。

### 结论

对抗性攻击对自动驾驶系统的实际影响取决于整个代理架构，而不仅仅是机器学习模型。某些安全机制（如PID控制或基于GPS的规则）可以减轻对抗性攻击的影响。

### 翻译

为了自主控制车辆，驾驶代理使用机器学习模型、控制器逻辑和自定义模块的组合输出。尽管许多先前的工作已经表明对抗性例子可以误导自动驾驶环境中使用的机器学习模型，但这些攻击是否对各种代理、环境和场景产生有害的驾驶行动仍然不清楚。为了评估对抗性例子对自动驾驶的风险，我们评估了针对各种驾驶代理的攻击，而不是孤立地针对机器学习模型。为了支持这一评估，我们利用CARLA（一个城市驾驶模拟器）来创建和评估对抗性例子。我们创建了旨在停止或引导驾驶代理的对抗性补丁，在运行时将它们流式传输到CARLA模拟器中，并针对来自CARLA排行榜的代理进行评估，该排行榜是年度研究竞赛中表现最佳的开源自动驾驶代理的公共存储库。与之前的工作不同，我们评估了对自动驾驶系统的攻击，而没有创建或修改任何驾驶代理代码，并且针对包含机器学习模型的所有代理部分。我们对两种攻击策略进行了案例研究调查，针对来自CARLA排行榜的三个开源驾驶代理，在多种驾驶场景、光照条件和地点下进行。有趣的是，我们表明，虽然某些攻击可以成功地误导机器学习模型，预测错误的停止或转向命令，但一些驾驶代理使用的模块，如PID控制或基于GPS的规则，可以覆盖攻击者操纵的机器学习模型预测。


### 论文摘要

To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios.   To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model.   We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.

---

## 87. Stability bounds for the generalized Kadanoff-Baym ansatz in the Holstein dimer

**论文链接:** [http://arxiv.org/abs/2511.15582v1](http://arxiv.org/abs/2511.15582v1)

**作者:** O. Moreno Segura, Y. Pavlyukh, R. Tuovinen

**发布时间:** 2025-11-19

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

本研究探讨了广义卡达诺夫-拜姆近似(GKBA)在预测相关系统实时动力学时的失败原因和条件，为可靠的GKBA模拟提供了实用指导。

### 背景

预测相关系统中的实时动力学具有挑战性：精确的双时格林函数方法准确但计算成本过高，而GKBA提供时间线性传播却存在不可控行为的风险。

### 目的

研究GKBA在何时以及为何失败，特别是在描述电子-声子耦合的Holstein二聚体这一最小但信息丰富的模型中。

### 方法

使用守恒的、完全自洽的电子-声子自能，映射出GKBA动力学稳定和不稳定的参数区域，并将失败的起源追溯到模型基态解的定性变化。

### 主要发现

GKBA的失败源于模型基态解的定性变化，这为GKBA时间演化提供了实际稳定性界限；将二聚体与电子引线耦合可以减弱并部分治愈这些不稳定性。

### 结论

研究结果为可靠的GKBA模拟电子-声子动力学提供了简单的诊断和指导方针。

### 翻译

预测相关系统中的实时动力学具有挑战性：精确的双时格林函数方法准确但通常成本过高，而广义卡达诺夫-拜姆近似(GKBA)提供时间线性传播却存在不可控行为的风险。我们研究了GKBA在描述电子-声子耦合的Holstein二聚体这一最小但信息丰富的模型中何时以及为何失败。使用守恒的、完全自洽的电子-声子自能，我们绘制出GKBA动力学稳定和不稳定的参数区域。我们将这些失败的起始追溯到从完整非平衡格林函数理论获得的模型基态解的定性变化，从而为GKBA时间演化提供了实际的稳定性界限。我们进一步表明，将二聚体与电子引线耦合可以减弱并部分治愈这些不稳定性。这些结果为可靠的GKBA模拟电子-声子动力学提供了简单的诊断和指导方针。


### 论文摘要

Predicting real-time dynamics in correlated systems is demanding: exact two-time Green's function methods are accurate but often too costly, while the Generalized Kadanoff-Baym Ansatz (GKBA) offers time-linear propagation at the risk of uncontrolled behavior. We examine when and why GKBA fails in a minimal yet informative setting, the Holstein dimer that describes electron-phonon coupling. Using a conserving, fully self-consistent electron-phonon self-energy, we map out parameter regions where GKBA dynamics is stable and where it becomes unstable. We trace the onset of these failures to qualitative changes in the model's ground-state solutions obtained from the full nonequilibrium Green's function theory, thereby providing practical stability bounds for GKBA time evolution. We further show that coupling the dimer to electronic leads can damp and, in part, cure these instabilities. The results supply simple diagnostics and guidelines for reliable GKBA simulations of electron-phonon dynamics.

---

## 88. NMPC-based Motion Planning with Adaptive Weighting for Dynamic Object Interception

**论文链接:** [http://arxiv.org/abs/2511.15532v1](http://arxiv.org/abs/2511.15532v1)

**作者:** Chen Cai, Saksham Kohli, Steven Liu

**发布时间:** 2025-11-19

**备注:** This work has been submitted to the IFAC World Congress for possible publication. Under review

### GPT解析

### 总结

本文提出了一种基于非线性模型预测控制(MPC)的运动规划器，用于解决合作机械臂系统捕捉快速移动物体时的协调挑战，通过自适应终端(AT)MPC方法显著提高了运动质量和鲁棒性。

### 背景

捕捉快速移动的物体是机器人敏捷性的基准，对于合作机械臂系统来说，由于固有的闭环链约束，协调控制具有重大挑战。

### 目的

开发一个连接高层拦截规划和实时关节空间控制的运动规划器，使由两个合作手臂组成的系统能够实现动态物体拦截。

### 方法

引入一种具有成本塑造的自适应终端(AT)MPC公式，与依赖终端惩罚实现快速收敛的简单原始终端(PT)方法形成对比，有效解决执行器功率限制违反问题。

### 主要发现

实验结果显示AT方案在双臂机器人平台上平均规划周期计算时间为19毫秒(不足系统采样时间40毫秒的一半)，以最小计算开销实现了显著改进的运动质量和鲁棒性。

### 结论

自适应终端MPC公式非常适合动态的合作拦截任务，相比PT基线方法具有明显优势。

### 翻译

捕捉快速移动的物体作为机器人敏捷性的基准，为持有捕捉器的合作机械臂系统带来了显著的协调挑战，特别是由于固有的闭环链约束。本文提出了一种基于非线性模型预测控制(MPC)的运动规划器，它连接了高层拦截规划和实时关节空间控制，使由两个合作手臂组成的系统能够实现动态物体拦截。我们引入了一种具有成本塑造的自适应终端(AT)MPC公式，这与依赖终端惩罚实现快速收敛的简单原始终端(PT)方法形成对比。所提出的AT公式被证明可以有效解决与PT策略相关的执行器功率限制违反问题，产生显著减少控制 effort的轨迹。在具有两个合作手臂的机器人平台上的实验结果展示了出色的实时性能，平均规划周期计算时间约为19毫秒，不到40毫秒系统采样时间的一半。这些结果表明，与PT基线相比，AT公式以最小的计算开销实现了显著改进的运动质量和鲁棒性，非常适合动态、合作拦截任务。


### 论文摘要

Catching fast-moving objects serves as a benchmark for robotic agility, posing significant coordination challenges for cooperative manipulator systems holding a catcher, particularly due to inherent closed-chain constraints. This paper presents a nonlinear model predictive control (MPC)-based motion planner that bridges high-level interception planning with real-time joint space control, enabling dynamic object interception for systems comprising two cooperating arms. We introduce an Adaptive- Terminal (AT) MPC formulation featuring cost shaping, which contrasts with a simpler Primitive-Terminal (PT) approach relying heavily on terminal penalties for rapid convergence. The proposed AT formulation is shown to effectively mitigate issues related to actuator power limit violations frequently encountered with the PT strategy, yielding trajectories and significantly reduced control effort. Experimental results on a robotic platform with two cooperative arms, demonstrating excellent real time performance, with an average planner cycle computation time of approximately 19 ms-less than half the 40 ms system sampling time. These results indicate that the AT formulation achieves significantly improved motion quality and robustness with minimal computational overhead compared to the PT baseline, making it well-suited for dynamic, cooperative interception tasks.

---

## 89. Quantum field theory approach to neutrino oscillations in dark matter and its implication in the Juno experiment

**论文链接:** [http://arxiv.org/abs/2511.15494v1](http://arxiv.org/abs/2511.15494v1)

**作者:** Wei Chao

**发布时间:** 2025-11-19

**备注:** 12 pages, 1 figure

### GPT解析

### 总结

本研究探讨了标量型超轻暗物质中质量中微子的物质效应，使用量子场论方法计算中微子振荡概率，发现其与量子力学方法的结果相比没有额外的时间依赖性。

### 背景

中微子振荡是一个值得深入探索的重要物理过程。

### 目的

研究标量型超轻暗物质中质量中微子的物质效应，并使用量子场论方法计算中微子振荡概率。

### 方法

使用量子场论方法计算中微子振荡概率。

### 主要发现

量子场论方法推导出的中微子振荡概率没有额外的时间依赖性，这是与量子力学方法获得的振荡结果最显著的区别。此外，还讨论了朱诺实验关于标量型超轻暗物质中中微子振荡行为的预测。

### 结论

这项研究扩展了对中微子与暗物质相互作用的理解，值得进一步探索。

### 翻译

中微子振荡是一个值得深入探索的重要物理过程。在本文中，我们研究了标量型超轻暗物质中质量中微子的物质效应，并使用量子场论方法计算中微子振荡概率。结果表明，量子场论方法推导出的中微子振荡概率没有额外的时间依赖性，这与量子力学方法获得的振荡结果形成了最显著的区别。此外，我们还讨论了朱诺实验关于标量型超轻暗物质中中微子振荡行为的预测。这项研究扩展了对中微子与暗物质相互作用的理解，值得进一步探索。


### 论文摘要

Neutrino oscillation is a significant physical process worthy of in-depth exploration. In this paper, we investigate the matter effect of massive neutrinos in a scalar-type ultra-light dark matter and calculate the neutrino oscillation probability using the quantum field theory method. The result reveals that the neutrino oscillation probability derived from the quantum field theory approach exhibits no additional time dependence, which marks the most significant distinction from the oscillation result obtained through the quantum mechanics method. Furthermore, we discuss predictions of the Juno experiment regarding neutrino oscillation behavior in scalar-type ultra-light dark matter. This study extends the understanding of the interaction between neutrinos and dark matter, which warrants further exploration.

---

## 90. Behavior Trees vs Executable Ontologies: a Comparative Analysis of Robot Control Paradigms

**论文链接:** [http://arxiv.org/abs/2511.15274v1](http://arxiv.org/abs/2511.15274v1)

**作者:** Alexander Boldachev

**发布时间:** 2025-11-19

**备注:** 22 pages, 8 figures

### GPT解析

### 总结

这篇论文比较了行为树(BTs)和可执行本体(EO)两种机器人行为建模方法，展示了EO通过事件驱动架构实现与BTs相当的特性，同时提供了BTs难以实现的功能。

### 背景

传统机器人控制存在语义-过程差距，需要更有效的行为建模方法。

### 目的

比较BTs和EO两种机器人行为建模方法，评估EO作为替代框架的潜力。

### 方法

通过boldsea框架实现EO，与BTs在控制流和数据流驱动方面进行对比，并在实际移动操作任务中进行测试。

### 主要发现

EO通过事件驱动的状态传播实现了与BTs相当的反应性和模块性，同时支持运行时模型修改、时间可追溯性和统一表示。

### 结论

EO提供了一个从过程编程转向语义领域建模的替代框架，特别适合动态、演进的机器人系统，而BTs在既定、可预测场景中仍有优势。

### 翻译

这篇论文比较了两种不同的机器人行为建模方法：命令式的行为树(BTs)和声明式的可执行本体(EO)，后者通过boldsea框架实现。BTs使用控制流来分层结构化行为，而EO则将领域表示为基于时间的、事件驱动的语义图，由数据流规则驱动。研究表明，EO通过根本不同的架构实现了与BTs相当的反应性和模块化：用基于事件的状态传播替代了基于轮询的tick执行。作者提出EO提供了一个替代框架，从过程编程转向语义领域建模，以解决传统机器人控制中的语义-过程差距。EO支持运行时模型修改、完整的时间可追溯性以及数据、逻辑和界面的统一表示 - 这些功能用BTs很难或有时无法实现，尽管BTs在既定、可预测的场景中表现出色。比较基于一个实际的移动操作任务。这种比较突出了每种方法在动态、演进的机器人系统中的各自操作优势。


### 论文摘要

This paper compares two distinct approaches to modeling robotic behavior: imperative Behavior Trees (BTs) and declarative Executable Ontologies (EO), implemented through the boldsea framework. BTs structure behavior hierarchically using control-flow, whereas EO represents the domain as a temporal, event-based semantic graph driven by dataflow rules. We demonstrate that EO achieves comparable reactivity and modularity to BTs through a fundamentally different architecture: replacing polling-based tick execution with event-driven state propagation. We propose that EO offers an alternative framework, moving from procedural programming to semantic domain modeling, to address the semantic-process gap in traditional robotic control. EO supports runtime model modification, full temporal traceability, and a unified representation of data, logic, and interface - features that are difficult or sometimes impossible to achieve with BTs, although BTs excel in established, predictable scenarios. The comparison is grounded in a practical mobile manipulation task. This comparison highlights the respective operational strengths of each approach in dynamic, evolving robotic systems.

---

## 91. RLS Framework with Segmentation of the Forgetting Profile and Low Rank Updates

**论文链接:** [http://arxiv.org/abs/2511.15273v1](http://arxiv.org/abs/2511.15273v1)

**作者:** Alexander Stotsky

**发布时间:** 2025-11-19

**DOI:** 10.5120/ijca2025925940

### GPT解析

### 总结

该研究提出了一种基于遗忘曲线分割的新正则化方法，用于滑动窗口最小二乘估计，通过三个不同特性的段来提高估计器的性能。

### 背景

滑动窗口最小二乘估计在信号处理和预测中广泛应用，但存在估计速度、信息矩阵条件数、准确性和数值稳定性等方面的挑战。

### 目的

开发一种新的正则化方法，通过分割遗忘曲线来优化估计器的性能，包括提高估计速度、减少信息矩阵条件数、增强准确性和数值稳定性，并能够整合信号特性的先验信息。

### 方法

将遗忘曲线分为三个段：第一段采用快速指数遗忘近期数据确保估计速度；第二段作为过渡；第三段使用慢速指数遗忘远期数据以减少信息矩阵条件数。开发了基于新矩阵求逆引理的递归和计算高效算法，具有低秩更新特性。

### 主要发现

条件数减少可以减轻误差传播，从而提高估计的准确性和稳定性。新算法显著提高了低分辨率日温度测量的近似精度。

### 结论

这种基于遗忘曲线分割的正则化方法有效改善了滑动窗口最小二乘估计的性能，特别是在温度预测应用中提高了预测的可靠性。

### 翻译

该报告描述了一种基于滑动窗口最小二乘估计中遗忘曲线分割的新正则化方法。每个段被设计用来强制执行估计器的特定期望属性，如快速性、期望的信息矩阵条件数、准确性、数值稳定性等。遗忘曲线被分为三个部分，其中第一部分通过采用近期数据的快速指数遗忘确保估计速度。第二部分特征是曲线下降，标记向第三部分的过渡，第三部分以慢速指数遗忘为特征，使用更远的数据减少信息矩阵的条件数。条件数减少减轻了误差传播，从而提高了准确性和稳定性。这种方法有助于将关于信号特性的先验信息纳入估计器。开发了基于新矩阵求逆引理的递归和计算高效算法，用于与这种正则化方法相关联的移动窗口，具有低秩更新。新算法显著提高了在斯德哥尔摩老天文台获得的低分辨率日温度测量的近似精度，从而提高了温度预测的可靠性。


### 论文摘要

This report describes a new regularization approach based on segmentation of the forgetting profile in sliding window least squares estimation. Each segment is designed to enforce specific desirable properties of the estimator such as rapidity, desired condition number of the information matrix, accuracy, numerical stability, etc. The forgetting profile is divided in three segments, where the speed of estimation is ensured by the first segment, which employs rapid exponential forgetting of recent data.The second segment features a decline in the profile and marks the transition to the third segment, characterized by slow exponential forgetting to reduce the condition number of the information matrix using more distant data. Condition number reduction mitigates error propagation, thereby enhancing accuracy and stability. This approach facilitates the incorporation of a priori information regarding signal characteristics (i.e., the expected behavior of the signal) into the estimator. Recursive and computationally efficient algorithm with low rank updates based on new matrix inversion lemma for moving window associated with this regularization approach is developed. New algorithms significantly improve the approximation accuracy of low resolution daily temperature measurements obtained at the Stockholm Old Astronomical Observatory, thereby enhancing the reliability of temperature predictions.

---

## 92. Corporate Earnings Calls and Analyst Beliefs

**论文链接:** [http://arxiv.org/abs/2511.15214v1](http://arxiv.org/abs/2511.15214v1)

**作者:** Giuseppe Matera

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究探讨了叙事如何影响经济行为和分析师预期预测，展示了从企业沟通中提取的叙事对预测的重要价值。

### 背景

经济行为不仅受定量信息影响，也受信息传达和解释的叙事影响(Shiller, 2017)。

### 目的

展示从盈利电话会议中提取的叙事能显著改善对已实现盈利和分析师预期的预测。

### 方法

引入一种新颖的文本变形方法，大型语言模型生成反事实文本，系统性地改变主题重点(主导叙事)同时保持定量内容不变，以精确测量分析师对不同叙事维度的反应。

### 主要发现

分析师对情绪(乐观)反应过度，而对风险和不确定性的叙事反应不足，表明存在系统性认知偏差。

### 结论

分析通过企业沟通中嵌入的竞争性叙事，提供了关于预期形成机制的细致视角。

### 翻译

经济行为不仅受定量信息影响，也受信息传达和解释的叙事影响(Shiller, 2017)。研究表明，从盈利电话会议中提取的叙事能显著改善对已实现盈利和分析师预期的预测。为了揭示潜在机制，研究者引入了一种新颖的文本变形方法，其中大型语言模型生成反事实文本，系统性地改变主题重点(主导叙事)同时保持定量内容不变。这一框架使研究者能够精确测量分析师对不同叙事维度的反应不足或过度反应。结果显示存在系统性偏差：分析师对情绪(乐观)反应过度，而对风险和不确定性的叙事反应不足。总体而言，该分析通过企业沟通中嵌入的竞争性叙事，提供了关于预期形成机制的细致视角。


### 论文摘要

Economic behavior is shaped not only by quantitative information but also by the narratives through which such information is communicated and in- terpreted (Shiller, 2017). I show that narratives extracted from earnings calls significantly improve the prediction of both realized earnings and analyst ex- pectations. To uncover the underlying mechanisms, I introduce a novel text- morphing methodology in which large language models generate counterfac- tual transcripts that systematically vary topical emphasis (the prevailing narra- tive) while holding quantitative content fixed. This framework allows me to precisely measure how analysts under- and over-react to specific narrative di- mensions. The results reveal systematic biases: analysts over-react to sentiment (optimism) and under-react to narratives of risk and uncertainty. Overall, the analysis offers a granular perspective on the mechanisms of expectation forma- tion through the competing narratives embedded in corporate communication.

---

## 93. Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story

**论文链接:** [http://arxiv.org/abs/2511.15210v1](http://arxiv.org/abs/2511.15210v1)

**作者:** Vladislav Pedashenko, Laida Kushnareva, Yana Khassan Nibal, Eduard Tulchinskii, Kristian Kuznetsov, Vladislav Zharchinskii, Yury Maximov, Irina Piontkovskaya

**发布时间:** 2025-11-19

### GPT解析

### 总结

这篇论文研究了大型语言模型中的内在维度(ID)与可解释文本属性之间的关系，通过三种分析方法发现ID与熵指标互补，不同文本类型有不同ID值，以及特定文本特征对ID的因果影响。

### 背景

内在维度(ID)是现代LLM分析的重要工具，用于研究训练动态、扩展行为和数据集结构，但其文本决定因素尚未得到充分探索。

### 目的

通过跨编码器分析、语言特征和稀疏自编码器(SAEs)等方法，将ID与可解释的文本属性联系起来，为ID的正确使用和基于ID的结果的合理解释提供指导。

### 方法

采用跨编码器分析、语言特征提取和稀疏自编码器(SAEs)技术，通过控制变量实验和转向实验来研究ID与文本属性的关系。

### 主要发现

ID与基于熵的指标互补，在控制长度后两者不相关，ID捕捉与预测质量正交的几何复杂性；不同文本类型有不同ID值：科学散文(~8)、百科内容(~9)、创意/观点写作(~10.5)；科学信号(正式语调、报告模板、统计)降低ID；人性化信号(个性化、情感、叙事)增加ID，这些效应是因果的。

### 结论

当代大型语言模型认为科学写作相对'简单'，而小说、观点和情感则增加了表示自由度，这一发现有助于理解和应用ID分析。

### 翻译

内在维度(ID)是现代LLM分析的重要工具，为研究训练动态、扩展行为和数据集结构提供信息，但其文本决定因素仍未得到充分探索。我们提供了首个综合研究，通过跨编码器分析、语言特征和稀疏自编码器(SAEs)，将ID与可解释的文本属性联系起来。在本工作中，我们建立了三个关键发现。首先，ID与基于熵的指标互补：在控制长度后，两者不相关，ID捕捉了与预测质量正交的几何复杂性。其次，ID表现出稳健的体裁分层：科学散文显示低ID(~8)，百科内容中等ID(~9)，创意/观点写作高ID(~10.5)，在所有测试的模型中均如此。这表明当代LLM认为科学文本'表示简单'，而小说则需要额外的自由度。第三，使用SAEs，我们确定了因果特征：科学信号(正式语调、报告模板、统计)降低ID；人性化信号(个性化、情感、叙事)增加ID。转向实验确认了这些效应是因果的。因此，对于当代模型，科学写作相对'简单'，而小说、观点和情感则增加了表示自由度。我们的多方面分析为ID的正确使用和基于ID的结果的合理解释提供了实用指导。


### 论文摘要

Intrinsic dimension (ID) is an important tool in modern LLM analysis, informing studies of training dynamics, scaling behavior, and dataset structure, yet its textual determinants remain underexplored. We provide the first comprehensive study grounding ID in interpretable text properties through cross-encoder analysis, linguistic features, and sparse autoencoders (SAEs). In this work, we establish three key findings. First, ID is complementary to entropy-based metrics: after controlling for length, the two are uncorrelated, with ID capturing geometric complexity orthogonal to prediction quality. Second, ID exhibits robust genre stratification: scientific prose shows low ID (~8), encyclopedic content medium ID (~9), and creative/opinion writing high ID (~10.5) across all models tested. This reveals that contemporary LLMs find scientific text "representationally simple" while fiction requires additional degrees of freedom. Third, using SAEs, we identify causal features: scientific signals (formal tone, report templates, statistics) reduce ID; humanized signals (personalization, emotion, narrative) increase it. Steering experiments confirm these effects are causal. Thus, for contemporary models, scientific writing appears comparatively "easy", whereas fiction, opinion, and affect add representational degrees of freedom. Our multi-faceted analysis provides practical guidance for the proper use of ID and the sound interpretation of ID-based results.

---

## 94. MMCM: Multimodality-aware Metric using Clustering-based Modes for Probabilistic Human Motion Prediction

**论文链接:** [http://arxiv.org/abs/2511.15179v1](http://arxiv.org/abs/2511.15179v1)

**作者:** Kyotaro Tokoro, Hiromu Taketsugu, Norimichi Ukita

**发布时间:** 2025-11-19

**备注:** Accepted to WACV2026

### GPT解析

### 总结

本文提出了一种用于人类运动预测(HMP)的新型评估指标MMCM，解决了现有指标在评估多模态预测时的局限性，通过聚类方法明确评估预测运动的覆盖率和有效性。

### 背景

单一过去序列可能导致多种可能的未来运动，概率性HMP方法预测多种可能的未来运动。现有指标即使预测的运动只分布在单一模式中且运动学无效，也简单赞赏广泛分布的运动。

### 目的

开发一种能够准确评估多模态人类运动预测的指标，同时考虑预测运动的覆盖率和有效性两个关键准则。

### 方法

提出了一种基于聚类的多模态感知指标(MMCM)。对于覆盖率，MMCM将运动空间划分为多个聚类作为模式，评估预测运动是否分布在多个模式中；对于有效性，MMCM通过从运动数据集中收集可能的未来运动来识别有效模式。

### 主要发现

实验验证了MMCM的聚类方法能够产生合理的模式定义，并且该指标能够准确评分多模态预测结果。

### 结论

MMCM指标解决了现有评估方法的不足，能够更准确地评估多模态人类运动预测的质量，考虑了预测运动的分布多样性和运动学有效性。

### 翻译

本文提出了一种用于人类运动预测(HMP)的新型指标。由于单一过去序列可能导致多种可能的未来，概率性HMP方法预测多种可能的运动。而确定性方法预测的单一运动只需与真实运动比较差异，多种预测的运动还应基于其分布进行评估。为此，本文关注以下两个标准。(a)覆盖率：运动应分布在多个运动模式中以覆盖各种可能性。(b)有效性：运动应是从给定过去运动可观察到的未来运动，在运动学上是有效的。然而，现有指标即使运动只分布在单一模式中且运动学无效，也简单赞赏广泛分布的运动。为解决这些缺点，本文提出了一种基于聚类的多模态感知指标(MMCM)。对于(a)覆盖率，MMCM将运动空间划分为几个聚类，每个被视为一个模式。这些模式用于明确评估预测的运动是否分布在多个模式中。对于(b)有效性，MMCM通过从运动数据集中收集可能的未来运动来识别有效模式。我们的实验验证了聚类产生合理的模式定义，且MMCM能准确评分多模态预测。代码：https://github.com/placerkyo/MMCM


### 论文摘要

This paper proposes a novel metric for Human Motion Prediction (HMP). Since a single past sequence can lead to multiple possible futures, a probabilistic HMP method predicts such multiple motions. While a single motion predicted by a deterministic method is evaluated only with the difference from its ground truth motion, multiple predicted motions should also be evaluated based on their distribution. For this evaluation, this paper focuses on the following two criteria. \textbf{(a) Coverage}: motions should be distributed among multiple motion modes to cover diverse possibilities. \textbf{(b) Validity}: motions should be kinematically valid as future motions observable from a given past motion. However, existing metrics simply appreciate widely distributed motions even if these motions are observed in a single mode and kinematically invalid. To resolve these disadvantages, this paper proposes a Multimodality-aware Metric using Clustering-based Modes (MMCM). For (a) coverage, MMCM divides a motion space into several clusters, each of which is regarded as a mode. These modes are used to explicitly evaluate whether predicted motions are distributed among multiple modes. For (b) validity, MMCM identifies valid modes by collecting possible future motions from a motion dataset. Our experiments validate that our clustering yields sensible mode definitions and that MMCM accurately scores multimodal predictions. Code: https://github.com/placerkyo/MMCM

---

## 95. Effects of Interactions and Defect Motion on Ramp Reversal Memory in Locally Phase Separated Materials

**论文链接:** [http://arxiv.org/abs/2511.15147v1](http://arxiv.org/abs/2511.15147v1)

**作者:** Y. Sun, M. Alzate Banguero, P. Salev, Ivan K. Schuller, L. Aigouy, A. Zimmers, E. W. Carlson

**发布时间:** 2025-11-19

**备注:** 13 + 11 pages, 10 + 7 figures, first submitted version before peer-review, accepted by Advanced Electronic Materials

### GPT解析

### 总结

本研究扩展了金属绝缘体过渡金属氧化物中的斜坡反转记忆效应模型，通过结合随机场伊辛模型与缺陷扩散-偏析，实现了更准确的滞后建模，并揭示了域相互作用与RRM效应的关系。

### 背景

金属绝缘体过渡金属氧化物中的斜坡反转记忆效应是由重复温度循环引起的非易失性电阻变化，在神经形态计算和非易失性存储器件中具有重要应用价值。先前提出的缺陷运动模型虽能解释VO2中的RRM，但未考虑金属和绝缘域之间的相互作用。

### 目的

扩展先前的缺陷运动模型，使其能够包含金属和绝缘域之间的相互作用，建立更准确的滞后模型，并预测RRM与域相互作用之间的关系。

### 方法

将随机场伊辛模型与缺陷扩散-偏析相结合，通过模拟实验研究RRM效应与域相互作用的关系。

### 主要发现

1) 当转折温度接近变暖分支拐点时，RRM达到最大值，与VO2实验观察一致；2) 增加最近邻相互作用可增强最大记忆效应，为优化RRM性能提供机制；3) RRM可能是表现出电子域图案化相共存的材料中的普遍现象。

### 结论

这项工作不仅推进了对TMOs中存储行为的基本理解，还为优化器件应用建立了急需的理论框架。

### 翻译

斜坡反转记忆效应(RRM)是金属绝缘体过渡金属氧化物(TMOs)中的一种非易失性电阻变化，由重复温度循环引起，在神经形态计算和非易失性存储器件中引起了相当大的兴趣。我们先前引入的缺陷运动模型成功地解释了二氧化钒(VO2)中的RRM，捕捉到了整个样品中观察到的临界温度偏移和记忆累积。然而，这种方法缺乏金属和绝缘域之间的相互作用，而RRM仅在TMOs进入金属绝缘体共存状态时才会出现。在这里，我们通过结合随机场伊辛模型与缺陷扩散-偏析来扩展我们的模型，从而能够准确模拟滞后同时预测RRM与域相互作用之间的关系。我们的模拟表明，当转折温度接近变暖分支拐点时，RRM达到最大值，这与VO2上的实验观察一致。最重要的是，我们发现增加最近邻相互作用增强了最大记忆效应，从而为优化RRM性能提供了明确的机制。由于我们的模型采用最少的假设，我们预测RRM应该是表现出电子域图案化相共存的材料中的普遍现象。这项工作不仅推进了对TMOs中存储行为的基本理解，还为优化器件应用建立了一个急需的理论框架。


### 论文摘要

The ramp-reversal memory (RRM) effect in metal-insulator transition metal oxides (TMOs), a non-volatile resistance change induced by repeated temperature cycling, has attracted considerable interest in neuromorphic computing and non-volatile memory devices. Our previously introduced defect motion model successfully explained RRM in vanadium dioxide (VO$_2$), capturing observed critical temperature shifts and memory accumulation throughout the sample. However, this approach lacked interactions between metallic and insulating domains, whereas the RRM only appears when TMOs are brought into the metal-insulator coexistence regime. Here, we extend our model by combining the Random Field Ising Model with defect diffusion-segregation, thereby enabling accurate hysteresis modeling while predicting the relationship between RRM and domain interactions. Our simulations demonstrate that maximum RRM occurs when the turnaround temperature approaches the warming branch inflection point, consistent with experimental observations on VO$_2$. Most significantly, we find that increasing nearest-neighbor interactions enhances the maximum memory effect, thus providing a clear mechanism for optimizing RRM performance. Since our model employs minimal assumptions, we predict that RRM should be a widespread phenomenon in materials exhibiting patterned phase coexistence of electronic domains. This work not only advances fundamental understanding of memory behavior in TMOs but also establishes a much-needed theoretical framework for optimizing device applications.

---

## 96. SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification

**论文链接:** [http://arxiv.org/abs/2511.14977v1](http://arxiv.org/abs/2511.14977v1)

**作者:** Xiangyu Li, Zhaomiao Guo

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为SVBRD-LLM的框架，能够从真实交通视频中自动发现、验证和应用可解释的自动驾驶车辆行为规则，为交通安全分析、政策制定和公众接受度提供支持。

### 背景

随着更多自动驾驶车辆在公共道路上运行，理解自动驾驶车辆的真实世界行为对于分析交通安全、制定政策和公众接受度至关重要。

### 目的

提出一个框架，通过零样本提示工程从真实交通视频中自动发现、验证和应用可解释的行为规则，以分析自动驾驶车辆的行为特征。

### 方法

使用YOLOv8和ByteTrack提取车辆轨迹，计算运动学特征，采用GPT-5零样本提示比较自动驾驶和人类驾驶车辆生成35个行为规则假设，在验证集上测试并迭代改进规则，过滤虚假相关性，最终编译成高置信度规则库，并在独立测试集上评估框架性能。

### 主要发现

在超过1500小时的真实交通视频实验中，该框架在自动驾驶车辆识别任务上实现了90.0%的准确率和93.3%的F1分数；发现的规则揭示了自动驾驶车辆在速度控制平滑性、车道变化保守性和加速度稳定性方面的独特特征，每条规则都配有语义描述、适用情境和验证置信度。

### 结论

SVBRD-LLM框架能够有效地从真实交通数据中发现和验证自动驾驶车辆的行为规则，为自动驾驶技术的安全评估和政策制定提供了新的方法。

### 翻译

随着更多自动驾驶车辆在公共道路上运行，理解自动驾驶车辆的真实世界行为对于分析交通安全、制定政策和公众接受度至关重要。本文提出了SVBRD-LLM框架，通过零样本提示工程从真实交通视频中自动发现、验证和应用可解释的行为规则。该框架使用YOLOv8和ByteTrack提取车辆轨迹，计算运动学特征，并采用GPT-5零样本提示来比较自动驾驶和人类驾驶车辆，生成35个结构化的行为规则假设。这些规则在验证集上进行测试，基于失败案例迭代改进，过滤虚假相关性，并编译成高置信度规则库。该框架在独立测试集上进行了速度变化预测、车道变化预测和自动驾驶车辆识别任务的评估。在超过1500小时真实交通视频的实验中，该框架在自动驾驶车辆识别任务上实现了90.0%的准确率和93.3%的F1分数。发现的规则清楚地揭示了自动驾驶车辆在速度控制平滑性、车道变化保守性和加速度稳定性方面的独特特征，每条规则都配有语义描述、适用情境和验证置信度。


### 论文摘要

As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.

---

## 97. Teaching signal synchronization in deep neural networks with prospective neurons

**论文链接:** [http://arxiv.org/abs/2511.14917v1](http://arxiv.org/abs/2511.14917v1)

**作者:** Nicoas Zucchet, Qianqian Feng, Axel Laborieux, Friedemann Zenke, Walter Senn, João Sacramento

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究展示了具有自适应电流的前瞻性神经元如何补偿层级组织中缓慢整合神经元带来的累积延迟，通过预测未来输入与它们同步，解决教学信号与神经活动不同步的问题，使动态环境中的高效学习成为可能。

### 背景

工作记忆需要大脑维持最近过去的信息以指导持续行为。神经元可以通过缓慢整合其输入来贡献这种能力，创造出持续活动，但这种层级组织会引入累积延迟，导致表示行为正确性的教学信号与它们旨在指导的神经活动到达时间不同步。

### 目的

解决层级组织中缓慢整合神经元带来的累积延迟问题，实现教学信号与神经活动的同步，从而支持高效学习。

### 方法

展示具有自适应电流的神经元如何前瞻性地响应外部刺激来补偿延迟；通过数学分析前瞻编码机制；在运动控制任务上进行学习实验。

### 主要发现

前瞻性神经元能够在各种传播误差信号通过层级网络的学习算法中实现教学信号同步；这成功地指导了缓慢整合神经元的学习，使它们能够在扩展的时间尺度上形成和检索记忆。

### 结论

神经适应可以解决关键的时序问题，使动态环境中的高效学习成为可能。

### 翻译

工作记忆需要大脑维持来自最近过去的信息以指导持续行为。神经元可以通过缓慢整合它们的输入来贡献这种能力，创造出持续活动，这种活动会持续超过原始刺激的存在时间。然而，当这些缓慢整合的神经元以层级方式组织时，它们引入累积延迟，这给学习带来了根本性挑战：表示行为是否正确的教学信号与它们旨在指导的神经活动到达时间不同步。在这里，我们展示具有自适应电流的神经元可以通过前瞻性地响应外部刺激来补偿这些延迟——有效地预测未来输入以与它们同步。首先，我们展示这种前瞻性神经元能够在各种传播误差信号通过层级网络的学习算法中实现教学信号同步。其次，我们证明这成功地指导了缓慢整合神经元的学习，使它们能够在扩展的时间尺度上形成和检索记忆。我们通过前瞻编码机制的数学分析和运动控制任务上的学习实验来支持我们的发现。总之，我们的结果揭示了神经适应如何解决关键的时序问题，并使动态环境中的高效学习成为可能。


### 论文摘要

Working memory requires the brain to maintain information from the recent past to guide ongoing behavior. Neurons can contribute to this capacity by slowly integrating their inputs over time, creating persistent activity that outlasts the original stimulus. However, when these slowly integrating neurons are organized hierarchically, they introduce cumulative delays that create a fundamental challenge for learning: teaching signals that indicate whether behavior was correct or incorrect arrive out-of-sync with the neural activity they are meant to instruct. Here, we demonstrate that neurons enhanced with an adaptive current can compensate for these delays by responding to external stimuli prospectively -- effectively predicting future inputs to synchronize with them. First, we show that such prospective neurons enable teaching signal synchronization across a range of learning algorithms that propagate error signals through hierarchical networks. Second, we demonstrate that this successfully guides learning in slowly integrating neurons, enabling the formation and retrieval of memories over extended timescales. We support our findings with a mathematical analysis of the prospective coding mechanism and learning experiments on motor control tasks. Together, our results reveal how neural adaptation could solve a critical timing problem and enable efficient learning in dynamic environments.

---

## 98. Breakdown of Quantum Chaos in the Staggered-Field XXZ Chain: Confinement and Meson Formation

**论文链接:** [http://arxiv.org/abs/2511.14847v1](http://arxiv.org/abs/2511.14847v1)

**作者:** Julia Wildeboer, Marton Lajer, Robert M. Konik

**发布时间:** 2025-11-18

**备注:** 20 pages, 11 figures

### GPT解析

### 总结

研究分数量子化禁闭现象对多体能谱的影响，特别是在具有交错场的自旋1/2 XXZ链中，发现自旋子结合成域壁'介子'的现象。

### 背景

分数量子化禁闭现象可以重构多体能谱，特别是在具有交错场的自旋1/2 XXZ链中。

### 目的

研究禁闭诱导的非遍历性和量子疤痕形成，以及介子谱学在量子自旋链中的行为。

### 方法

通过精确对角化分析对称性分辨的能级，研究相邻能级比，分析关联和纠缠测量，并进行介子谱学分析，与理论预测比较。

### 主要发现

从弱各向异性到强反铁磁区域，系统从混沌能级统计转变为非遍历行为；在域壁数中观察到特征性能带结构；Page-like纠缠圆顶被抑制的能带分辨纠缠取代；低能谱与理论预测定量一致。

### 结论

建立了禁闭诱导非遍历性的统一解释，为量子自旋链中的定量介子谱学提供了模板。

### 翻译

分数量子化激发的禁闭可以强烈重构多体能谱。我们在具有交错场的能隙自旋-1/2 XXZ链中研究了这一现象，在反铁磁相深处，自旋子结合成域壁'介子'。我们提出证据表明，这个非可积模型在由各向异性参数Δ控制的情况下，既表现出希尔伯特空间分数量化，又表现出量子疤痕形成。通过对称性分辨部分的精确对角化，揭示了从弱各向异性Δ~1的高斯正交(混沌)能级统计到反铁磁区域Δ>>1深处非遍历行为的交叉，通过仔细检查相邻能级比，同时在关联和纠缠测量的域壁数中观察到显著的能带结构。混沌能谱特征性的Page-like纠缠圆顶让位于与域壁准守恒一致的、被抑制的能带分辨纠缠。为了进一步研究介子疤痕态的形成机制，我们在双自旋子阈值附近进行介子谱学研究，并与Rutkevich预测的解析梯子进行比较。我们通过连续相对结合、偏移移除的艾里标度坍缩和确定稳定介子能级数量的明确双介子阈值来验证该理论。低能谱显示出密切的定量一致性，而高能偏差与有限尺寸和次级修正一致。这些结果建立了禁闭诱导非遍历性的统一解释，并为量子自旋链中的定量介子谱学提供了模板。


### 论文摘要

Confinement of fractionalized excitations can strongly restructure many-body spectra. We investigate this phenomenon in the gapped spin-$\frac{1}{2}$ XXZ chain subject to a staggered field, where spinons bind into domain-wall ``mesons'' deep in the antiferromagnetic phase. We present evidence that this non-integrable model exhibits both Hilbert space fractionalization and quantum scar formation as controlled by the anisotropy parameter $Δ$. Exact diagonalization across symmetry-resolved sectors reveals a crossover from Gaussian-orthogonal (chaotic) level statistics at weak anisotropy $Δ\sim 1$ to non-ergodic behavior deep in the antiferromagnetic regime $Δ\gg 1$ through scrutinizing the adjacent gap ratios, accompanied by a striking banding of eigenstates by domain-wall number in correlation and entanglement measures. The Page-like entanglement dome characteristic of chaotic spectra gives way to suppressed, band-resolved entanglement consistent with emergent quasi-conservation of domain walls. To investigate further the formation mechanism of mesonic scar states, we carry out meson spectroscopy near the two-spinon threshold and compare with the analytic ladder predicted by Rutkevich [Phys. Rev. B 106, 134405 (2022)]. We test the theory through continuum-relative bindings, an offset-removed Airy scaling collapse, and explicit two-meson thresholds that determine the number of stable meson levels. The low-lying spectrum shows close quantitative agreement, while deviations at higher energies are consistent with finite-size and subleading corrections. These results establish a unified account of confinement-induced nonergodicity and provide a template for quantitative meson spectroscopy in quantum spin chains.

---

## 99. Magnetic Fields in the Shapley Supercluster Core with POSSUM: Challenging Model Predictions

**论文链接:** [http://arxiv.org/abs/2511.14377v2](http://arxiv.org/abs/2511.14377v2)

**作者:** D. Alonso-López, S. P. O'Sullivan, A. Bonafede, L. M. Böss, C. Stuardi, E. Osinga, C. S. Anderson, C. L. Van Eck, E. Carretti, J. L. West, T. Akahori, K. Dolag, S. Giacintucci, A. Khadir, Y. K. Ma, S. Malik, N. McClure-Griffiths, L. Rudnick, B. A. Seidel, S. Tiwari, T. Venturi

**发布时间:** 2025-11-18

**备注:** 23 pages, 15 figures. Accepted for publication in A&A

### GPT解析

### 总结

该研究研究了Shapley超星系团核心区域的法拉第旋转量信号，以约束气体磁场特性，结合RM网格数据与热SZ效应数据，分析RM散射行为并与模型比较。

### 背景

法拉第旋转量网格是追踪宇宙环境中磁化等离子体的敏感手段。Shapley超星系团核心区域(z≈0.048)包含两个星系团A3558和A3562，以及它们之间的两个星系群。

### 目的

约束星系团和星系群中气体的磁场特性，确定气体密度、磁场特性及其相关性，研究RM散射行为及其与距离的关系。

### 方法

结合RM网格数据与POSSUM试点调查和Planck获得的热SZ效应数据；分析RM散射及其与到最近星系团/星系群距离的关系；将观测结果与半解析高斯随机场模型和宇宙磁流体动力学模拟进行比较。

### 主要发现

1. 以36 RMs/deg²的天空密度，检测到30.5±4.6 rad/m²的过量RM散射；2. 星系团和星系群中的平均磁场强度为1-3 μG；3. RM散射剖面比模型预期的更平坦，η<0.5更受支持；4. 磁场在相互作用星系团郊区被湍流速度放大，尺度约小于0.8 r₅₀₀。

### 结论

密集的RM网格和POSSUM提供的精度使得能够探测SSC星系团和星系群内及超出其r₅₀₀尺度的磁化气体。比预期更平坦的RM散射剖面揭示了一个重大挑战：即使是最现实的宇宙MHD模拟预测，也难以协调观测结果与相互作用星系团郊区的数据。

### 翻译

法拉第旋转量(RM)网格提供了一种在广泛的宇宙环境中追踪磁化等离子体的敏感手段。我们研究了来自Shapley超星系团核心(SSC)的RM信号，以约束气体的磁场特性。SSC区域包含两个星系团A3558和A3562，以及它们之间的两个星系群，位于z≈0.048。我们将RM网格数据与POSSUM试点调查和Planck分别获得的热苏尼亚耶夫-泽尔多维奇效应数据相结合。为了稳健地确定气体密度、其磁场特性及其相关性，我们研究了SSC区域的RM散射及其作为到最近星系团/星系群距离函数的行为。我们将观测结果与半解析高斯随机场模型和更现实的宇宙磁流体动力学模拟进行了比较。以36 RMs/deg²的天空密度，我们在SSC区域检测到30.5±4.6 rad/m²的过量RM散射。与模型比较，我们发现星系团和星系群中的平均磁场强度为1-3 μG。从所有对象0.3-1.8 r₅₀₀范围的数据得出的RM散射剖面比模型预期的系统性地更平坦，η<0.5更受支持。尽管存在这种差异，我们发现与SSC结构最匹配的宇宙MHD模拟最符合磁场在相互作用星系团郊区尺度≲0.8 r₅₀₀上的湍流速度放大的情景。POSSUM提供的密集RM网格和精度使得我们能够探测SSC星系团和星系群内及超出其r₅₀₀尺度的磁化气体。比预期更平坦的RM散射剖面揭示了一个重大挑战：即使是最现实的宇宙MHD模拟预测，也难以协调观测结果与相互作用星系团郊区的数据。


### 论文摘要

Faraday Rotation Measure (RM) Grids provide a sensitive means to trace magnetized plasma across a wide range of cosmic environments. We study the RM signal from the Shapley Supercluster Core (SSC), in order to constrain the magnetic field properties of the gas. The SSC region consists of two galaxy clusters A3558 and A3562, and two galaxy groups between them, at $z\simeq 0.048$. We combine RM Grid data with thermal Sunyaev-Zeldovich effect data, obtained from the POSSUM pilot survey, and Planck, respectively. To robustly determine the gas density, its magnetic field properties, and their correlation, we study the RM scatter in the SSC region and its behavior as a function of distance to the nearest cluster/group. We compare observational results with semi-analytic Gaussian random field models and more realistic cosmological MHD simulations. With a sky-density of 36 RMs/deg$^{2}$, we detect an excess RM scatter of $30.5\pm 4.6 \, \mathrm{rad/m^2}$ in the SSC region. Comparing with models, we find an average magnetic field strength of 1-3 $μ$G (in the groups and clusters). The RM scatter profile, derived from data ranging from 0.3-1.8 $r_{500}$ for all objects, is systematically flatter than expected compared to models, with $η<0.5$ being favored. Despite this discrepancy, we find that cosmological MHD simulations matched to the SSC structure most closely align with scenarios where the magnetic field is amplified by the turbulent velocity in the intercluster regions on scales $\lesssim 0.8\,r_{500}$. The dense RM grid and precision provided by POSSUM allows us to probe magnetized gas in the SSC clusters and groups on scales within and beyond their $r_{500}$. Flatter-than-expected RM scatter profiles reveal a significant challenge in reconciling observations with even the most realistic predictions from cosmological MHD simulations in the outskirts of interacting clusters.

---

## 100. Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels

**论文链接:** [http://arxiv.org/abs/2511.15496v1](http://arxiv.org/abs/2511.15496v1)

**作者:** Maria Pilligua, David Serrano-Lozano, Pai Peng, Ramon Baldrich, Michael S. Brown, Javier Vazquez-Corral

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究引入了多光照低光(MILL)数据集，解决了低光增强领域中缺乏辐射度多样性的问题，并通过提出的改进方法显著提高了算法性能。

### 背景

低光环境下的成像具有挑战性，因为场景辐射度降低导致传感器噪声增加和色彩饱和度降低。大多数基于学习的低光增强方法依赖于在单一低光条件和良好照明参考下捕获的成对训练数据，缺乏辐射度多样性限制了我们对增强技术在不同光照强度下表现的理解。

### 目的

引入多光照低光(MILL)数据集，包含在受控条件下以不同光照强度捕获的图像，具有固定的相机设置和精确的照度测量，使增强算法在各种光照条件下的全面评估成为可能。

### 方法

构建了MILL数据集，包含多种光照条件下的图像；对比了多种最先进的方法，揭示了不同强度水平下的显著性能差异；利用数据集独特的多光照结构，提出了改进方法，增强了在不同光照场景下的鲁棒性。

### 主要发现

增强算法在不同光照强度下表现出显著的性能变化；提出的改进方法在高清图像上实现了高达10分贝的PSNR提升(DSLR)和2分贝的提升(智能手机)。

### 结论

MILL数据集能够全面评估增强算法在不同光照条件下的性能；提出的改进方法显著提高了算法在多种光照条件下的鲁棒性。

### 翻译

在低光环境中成像具有挑战性，因为场景辐射度降低，导致传感器噪声增加和色彩饱和度降低。大多数基于学习的低光增强方法依赖于在单一低光条件和良好照明参考下捕获的成对训练数据。辐射度多样性的缺乏限制了我们理解增强技术在不同光照强度下的表现。我们引入了多光照低光(MILL)数据集，包含在受控条件下以不同光照强度捕获的图像，具有固定的相机设置和精确的照度测量。MILL使增强算法在各种光照条件下的全面评估成为可能。我们对几种最先进的方法进行了基准测试，并揭示了不同强度水平下的显著性能差异。利用我们数据集独特的多光照结构，我们提出了改进，增强了在不同光照场景下的鲁棒性。我们的修改在高清图像上实现了高达10分贝的PSNR提升(DSLR)和2分贝的提升(智能手机)。


### 论文摘要

Imaging in low-light environments is challenging due to reduced scene radiance, which leads to elevated sensor noise and reduced color saturation. Most learning-based low-light enhancement methods rely on paired training data captured under a single low-light condition and a well-lit reference. The lack of radiance diversity limits our understanding of how enhancement techniques perform across varying illumination intensities. We introduce the Multi-Illumination Low-Light (MILL) dataset, containing images captured at diverse light intensities under controlled conditions with fixed camera settings and precise illuminance measurements. MILL enables comprehensive evaluation of enhancement algorithms across variable lighting conditions. We benchmark several state-of-the-art methods and reveal significant performance variations across intensity levels. Leveraging the unique multi-illumination structure of our dataset, we propose improvements that enhance robustness across diverse illumination scenarios. Our modifications achieve up to 10 dB PSNR improvement for DSLR and 2 dB for the smartphone on Full HD images.

---

## 101. Insights on Gas Distribution and Dynamics in Massive Proto-cluster G358.46$-$0.39: Possible Multiplicity in G358.46$-$0.39 MM1a

**论文链接:** [http://arxiv.org/abs/2511.15400v1](http://arxiv.org/abs/2511.15400v1)

**作者:** Chukwuebuka J. Ugwua, James O. Chibuezea, Willice Obonyoa, Mavis Seidu

**发布时间:** 2025-11-19

**备注:** 14 pages, 8 figures, accepted for publication in New Astronomy

### GPT解析

### 总结

本研究利用ALMA第7波段存档数据，探索了G358.46-0.39原星团中C17O、SiO、HC3N和SO2分子的空间分布以及外流的能量学，旨在增进对该原恒星特性、气体运动学和动力学的理解。

### 背景

G358.46-0.39原星团先前已知由4个尘埃连续谱核心组成，分别为MM1a、MM1b、MM1c和MM2。

### 目的

提供对原恒星特性、气体运动学和动力学的更好理解，探索分子的空间分布，研究外流的能量学。

### 方法

使用ALMA第7波段的存档数据进行分析，并估计了质量、动量和能量外流率以及其他外流参数。

### 主要发现

1) C17O积分强度图显示丝状和哑铃状结构，可能是HII区域MM2膨胀压缩的气体；2) SiO发射显示空间重叠的蓝色和红色外流瓣，可能由MM1a中未解析的年轻天体驱动；3) HC3N和SO2分子在MM1a中呈紧凑形态，在其他核心中未检测到；4) SO2发射在MM1a中显示清晰速度梯度和大速度弥散，与旋转结构一致；5) SiO外流与先前观测的12CO外流形态不同；6) SiO和12CO外流可能与不同朝向的盘相关，表明MM1a中存在多重年轻天体。

### 结论

MM1a的性质表明它是一个正在积极吸积和经历恒星形成的巨大原恒星。

### 翻译

本研究利用ALMA第7波段存档数据，探索了G358.46-0.39原星团中C17O、SiO、HC3N和SO2分子的空间分布以及外流的能量学，旨在增进对该原恒星特性、气体运动学和动力学的理解。G358.46-0.39先前已知由4个尘埃连续谱核心组成。C17O的积分强度图揭示丝状和哑铃状结构，可能是HII区域MM2膨胀压缩的气体。SiO发射显示空间重叠的蓝色和红色外流瓣，可能由MM1a中未解析的年轻天体驱动。HC3N和SO2分子在MM1a中的空间分布显示紧凑形态，在其他核心中未检测到。SO2发射在MM1a中显示出清晰的速度梯度以及核心内部的大速度弥散，与旋转结构一致。我们估计了质量、动量和能量外流率以及其他外流参数。SiO外流与先前在MM1a中观测到的12CO外流形态不同。SiO和12CO外流可能与分别有一个正面朝向和一个边缘朝向的盘相关，表明MM1a中存在多重年轻天体。MM1a的性质表明它是一个正在积极吸积和经历恒星形成的巨大原恒星。


### 论文摘要

This work explored the spatial distribution of C$^{17}$O, SiO, HC$_{3}$N and SO$_{2}$ molecules, as well as the energetics of outflows in G358.46$-$0.39 proto-cluster using ALMA band 7 archival data, with the aim of providing an improved understanding of its protostellar nature, gas kinematics and dynamics. G358.46$-$0.39 is previously known to consist of 4 dust continuum cores (MM1a, MM1b, MM1c and MM2). The integrated intensity map of C$^{17}$O reveals filamentary and dumbbell-shaped structures that are probably compressed gases from the expansion of the HII region MM2. The SiO emission reveals spatially overlapped blue and red outflow lobes, likely driven by an unresolved young stellar object (YSO) in MM1a. The spatial distribution of HC$_{3}$N and SO$_{2}$ molecules in MM1a shows a compact morphology, with no detectable HC$_{3}$N and SO$_{2}$ emissions in the other cores. The SO$_{2}$ emission reveals a clear velocity gradient in MM1a, as well as large velocity dispersion ($\sim$ 3\,\kms) within the inner core of MM1a, which are consistent with rotating structures. We estimated the mass, momentum and energy outflow rate, as well as other outflow parameters. The SiO outflow exhibits a different morphology compared to the $^{12}$CO outflow morphology previously observed in MM1a. The SiO and $^{12}$CO outflows are probably associated with disks of separate cores with one face-on and the other edge-on, pointing to multiplicity of YSOs in MM1a. The properties of MM1a indicate that it is a massive protostar that is actively accreting and undergoing star formation.

---

## 102. C2F-Space: Coarse-to-Fine Space Grounding for Spatial Instructions using Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.15333v1](http://arxiv.org/abs/2511.15333v1)

**作者:** Nayoung Oh, Dohyun Kim, Junhyeong Bang, Rohan Paul, Daehyung Park

**发布时间:** 2025-11-19

**备注:** 16 pages, 12 figures

### GPT解析

### 总结

本文提出了一种名为C2F-Space的新型从粗到细的空间定位框架，有效解决了传统方法和视觉-语言模型在空间定位中的局限性，在基准测试中显著优于现有方法，并在机器人抓取任务中展示了实用性。

### 背景

空间定位是指在自然语言指令中定位一组空间参考点。传统方法难以处理距离、几何关系和对象间关系等复杂推理，而视觉-语言模型虽有强大推理能力但无法产生细粒度输出区域。

### 目的

克服传统方法和视觉-语言模型的局限性，提出一种能够精确处理空间定位问题的新型框架。

### 方法

提出C2F-Space框架，包含两个步骤：首先使用视觉-语言模型估算近似但空间一致的区域；然后通过超像素化技术调整区域以适应当地环境。粗略估计采用基于网格的视觉定位提示和提出-验证策略，精细调整则使区域局部适应周围环境而不过度扩展。

### 主要发现

构建了新的空间定位基准，C2F-Space在成功率和交并比指标上显著优于五种最先进基线方法。消融研究证实了框架中各模块的有效性及整体框架的协同效应。

### 结论

C2F-Space在模拟机器人抓取和放置任务中展示了良好的适用性，为空间定位问题提供了有效解决方案。

### 翻译

空间定位是指在自然语言指令中定位一组空间参考点。传统方法通常无法考虑复杂的推理，如距离、几何和对象间关系，而视觉-语言模型尽管有强大的推理能力，却难以产生细粒度的输出区域。为了克服这些限制，我们提出了C2F-Space，一种新颖的从粗到细的空间定位框架，它(i)使用视觉-语言模型估算近似但空间一致的区域，然后(ii)通过超像素化调整区域以适应当地环境。对于粗略估计，我们设计了一种基于网格的视觉定位提示，采用提出-验证策略，最大化视觉-语言模型的空间理解能力，产生物理和语义上有效的规范区域（即椭圆）。对于精细调整，我们将区域局部适应到周围环境，而不过度扩展到自由空间。我们构建了一个新的空间定位基准，并使用成功率和交并比将C2F-Space与五种最先进的基线方法进行比较。我们的C2F-Space显著优于所有基线方法。我们的消融研究证实了两个步骤中每个模块的有效性以及组合框架的协同效应。最后，我们展示了C2F-Space在模拟机器人抓取和放置任务中的适用性。


### 论文摘要

Space grounding refers to localizing a set of spatial references described in natural language instructions. Traditional methods often fail to account for complex reasoning -- such as distance, geometry, and inter-object relationships -- while vision-language models (VLMs), despite strong reasoning abilities, struggle to produce a fine-grained region of outputs. To overcome these limitations, we propose C2F-Space, a novel coarse-to-fine space-grounding framework that (i) estimates an approximated yet spatially consistent region using a VLM, then (ii) refines the region to align with the local environment through superpixelization. For the coarse estimation, we design a grid-based visual-grounding prompt with a propose-validate strategy, maximizing VLM's spatial understanding and yielding physically and semantically valid canonical region (i.e., ellipses). For the refinement, we locally adapt the region to surrounding environment without over-relaxed to free space. We construct a new space-grounding benchmark and compare C2F-Space with five state-of-the-art baselines using success rate and intersection-over-union. Our C2F-Space significantly outperforms all baselines. Our ablation study confirms the effectiveness of each module in the two-step process and their synergistic effect of the combined framework. We finally demonstrate the applicability of C2F-Space to simulated robotic pick-and-place tasks.

---

## 103. Edge-Centric Relational Reasoning for 3D Scene Graph Prediction

**论文链接:** [http://arxiv.org/abs/2511.15288v1](http://arxiv.org/abs/2511.15288v1)

**作者:** Yanni Ma, Hao Liu, Yulan Guo, Theo Gevers, Martin R. Oswald

**发布时间:** 2025-11-19

### GPT解析

### 总结

LEO框架通过引入线图神经网络和边到物体的推理范式，有效解决了现有方法在捕捉高阶关系依赖方面的局限性，提高了3D场景图预测的准确性。

### 背景

现有3D场景图预测方法采用以物体为中心的图神经网络，将关系表示限制在成对物体上下文中，难以捕捉对准确关系预测至关重要的更高阶关系依赖。

### 目的

开发一个能够从关系级上下文逐步推理到物体级理解的框架，以捕捉高阶关系依赖并提高关系预测准确性。

### 方法

LEO框架首先预测物体对之间的潜在链接抑制无关边，将原始场景图转换为线图（每个关系视为节点），应用线图神经网络进行以边为中心的关系推理，然后将丰富的关系特征整合回原始物体中心图增强物体级推理。

### 主要发现

在3DSSG数据集上使用两个有竞争力的基线进行的实验显示了一致的改进，验证了边到物体推理范式的有效性。

### 结论

LEO框架是模型无关的，可与任何现有以物体为中心的方法集成，有效解决了高阶关系依赖捕捉的挑战，提升了3D场景图预测性能。

### 翻译

3D场景图预测旨在将复杂的3D环境抽象为由物体及其成对关系组成的结构化图。现有方法通常采用以物体为中心的图神经网络，其中关系边特征通过聚合连接物体节点的消息进行迭代更新。然而，这种设计将关系表示限制在成对物体上下文中，难以捕捉对准确关系预测至关重要的更高阶关系依赖。为了解决这一限制，我们提出了一个名为LEO的引导边为中心的关系推理框架，该框架能够从关系级上下文逐步推理到物体级理解。具体来说，LEO首先预测物体对之间的潜在链接以抑制无关边，然后将原始场景图转换为线图，其中每个关系被视为一个节点。应用线图神经网络进行以边为中心的关系推理，以捕捉关系间上下文。丰富的关系特征随后被整合到原始以物体为中心的图中，以增强物体级推理并改进关系预测。我们的框架与模型无关，可以与任何现有的以物体为中心的方法集成。在3DSSG数据集上使用两个有竞争力的基线进行的实验显示了一致的改进，突显了边到物体推理范式的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景图预测中现有方法难以捕获高阶关系依赖的问题。现有以对象为中心的图神经网络将关系表示限制在成对对象上下文中，无法有效捕捉关系间的相互依赖。这个问题很重要，因为准确的3D场景图对视觉问答、机器人导航等下游任务至关重要，而关系间的依赖性（如语义模糊的空间关系）是准确理解场景的关键挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有以对象为中心方法的局限性，意识到它们难以处理关系间的相互依赖。他们借鉴了线图神经网络的概念，将场景图重构为线图使关系成为节点，从而能够建模关系间而非仅对象间的依赖。同时，他们还借鉴了链接预测的思想来过滤不相关的边。设计上采用三阶段流程：链接预测、线图转换与边中心推理、最后整合回对象中心推理，形成渐进式推理范式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将场景图转换为线图，使关系成为图中的节点，从而显式建模关系间的高阶依赖，超越传统仅关注对象对关系的局限。整体流程包括：1)初始化3D场景图，提取对象和边特征；2)通过链接预测为对象对分配软权重；3)将原始图转换为线图并应用LineGNN进行边中心推理；4)将增强的关系特征整合回原始图进行对象中心推理；5)最终预测对象和关系类别。这种方法实现了从边到对象的渐进式推理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出LEO框架，实现边到对象的渐进式推理；2)首次将3D场景图重构为线图进行关系中心推理；3)设计链接预测模块调制关系强度；4)提出模型无关的框架可集成到现有方法中。相比之前工作，不同之处在于：从对象中心转向边中心推理范式，能够捕获高阶关系依赖；通过线图变换显式建模关系间交互；在3DSSG数据集上显著提升了关系预测准确性，特别是在处理语义模糊和空间关系相互依赖问题上。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LEO通过引入边中心的关系推理范式和线图变换，有效捕获了3D场景中的高阶关系依赖，显著提升了场景图预测的准确性和鲁棒性。'}


### 论文摘要

3D scene graph prediction aims to abstract complex 3D environments into structured graphs consisting of objects and their pairwise relationships. Existing approaches typically adopt object-centric graph neural networks, where relation edge features are iteratively updated by aggregating messages from connected object nodes. However, this design inherently restricts relation representations to pairwise object context, making it difficult to capture high-order relational dependencies that are essential for accurate relation prediction. To address this limitation, we propose a Link-guided Edge-centric relational reasoning framework with Object-aware fusion, namely LEO, which enables progressive reasoning from relation-level context to object-level understanding. Specifically, LEO first predicts potential links between object pairs to suppress irrelevant edges, and then transforms the original scene graph into a line graph where each relation is treated as a node. A line graph neural network is applied to perform edge-centric relational reasoning to capture inter-relation context. The enriched relation features are subsequently integrated into the original object-centric graph to enhance object-level reasoning and improve relation prediction. Our framework is model-agnostic and can be integrated with any existing object-centric method. Experiments on the 3DSSG dataset with two competitive baselines show consistent improvements, highlighting the effectiveness of our edge-to-object reasoning paradigm.

---

## 104. Photoluminescence Mapping of Mobile and Fixed Defects in Halide Perovskite Films

**论文链接:** [http://arxiv.org/abs/2511.15281v1](http://arxiv.org/abs/2511.15281v1)

**作者:** Sarah C. Gillespie, Jérome Gautier, Linde M. van de Ven, Agustin O. Alvarez, Bruno Ehrler, L. J. Geerligs, Veronique S. Gevaerts, Gianluca Coletti, Erik C. Garnett

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究使用局部强度调制光致发光光谱学(IMPLS)技术，提供了一种无接触的方法来提取钙钛矿材料中的横向离子扩散系数，同时能够区分不同类型的缺陷。

### 背景

金属卤化物钙钛矿具有耦合的电子和离子特性，这些特性决定了其光伏性能和运行稳定性。理解和量化离子传输对于推进钙钛矿光电器件至关重要。传统的电学方法如阻抗谱需要完全集成的器件，且其解释常常受到界面和接触效应的复杂影响，限制了分离固有离子行为的能力。

### 目的

开发一种光学方法来探测钙钛矿薄膜中的横向离子传输，克服传统电学方法的局限性，实现对离子行为的直接测量和缺陷类型的空间区分。

### 方法

采用局部强度调制光致发光光谱学(IMPLS)技术，在受控载流子注入水平下测量频率依赖的光致发光响应，并将其与光致发光量子产率(PLQY)相关联。通过缺陷对比系数(DCC)分离可移动和不可移动缺陷的贡献。

### 主要发现

提出的扩散模型表明，移动的离子缺陷从高光强度区域横向迁移，产生特征性的光致发光调制。从IMPLS中提取的离子扩散系数与文献中通过电学测量获得的值吻合良好。IMPLS映射通过缺陷对比系数(DCC)分离了可移动和不可移动缺陷的贡献，该系数量化了平均光致发光强度和相位数据之间的归一化差异。

### 结论

局部IMPLS提供了一种无接触的方法来提取横向离子扩散系数，同时能够在整个样品空间上区分不同类型的缺陷，为理解和优化钙钛矿材料的离子传输行为提供了新的工具。

### 翻译

金属卤化物钙钛矿表现出耦合的电子和离子特性，这些特性决定了它们的光伏性能和运行稳定性。因此，理解和量化离子传输对于推进钙钛矿光电器件至关重要。传统的电学方法如阻抗谱需要完全集成的器件，并且它们的解释常常受到界面和接触效应的复杂影响，限制了分离固有离子行为的能力。在这里，利用强度调制光致发光光谱学(IMPLS)的局部适应来光学探测钙钛矿薄膜中的横向离子传输。在受控载流子注入水平下测量频率依赖的光致发光响应，并将其与光致发光量子产率(PLQY)相关联。提出的扩散模型表明，移动的离子缺陷从高光强度区域横向迁移，产生特征性的光致发光调制。从IMPLS中提取的离子扩散系数与文献中通过电学测量获得的值吻合良好。重要的是，IMPLS映射通过缺陷对比系数(DCC)分离了可移动和不可移动缺陷的贡献，该系数量化了平均光致发光强度和相位数据之间的归一化差异。这项工作最终证明了局部IMPLS提供了一种无接触的方法来提取横向离子扩散系数，同时能够在整个样品上空间区分缺陷类型。


### 论文摘要

Metal halide perovskites exhibit coupled electronic and ionic properties that determine their photovoltaic performance and operational stability. Understanding and quantifying ionic transport are therefore essential for advancing perovskite optoelectronics. Conventional electrical methods such as impedance spectroscopy require fully integrated devices, and their interpretation is often complicated by interfacial and contact effects, limiting the ability to isolate intrinsic ionic behavior. Here, a localized adaptation of intensity-modulated photoluminescence spectroscopy (IMPLS) is utilized to optically probe lateral ionic transport in perovskite films. The frequency-dependent photoluminescence response is measured under controlled carrier injection levels and correlated with the photoluminescence quantum yield (PLQY). The proposed diffusion model indicates that mobile ionic defects laterally migrate from high light intensity regions, giving rise to characteristic photoluminescence modulations. Ionic diffusion coefficients extracted from IMPLS agree well with literature values obtained from electrical measurements. Importantly, IMPLS mapping separates mobile and immobile defect contributions through a defect contrast coefficient (DCC), which quantifies the normalized difference between the area-averaged photoluminescence intensity and phase data. This work ultimately demonstrates that localized IMPLS provides a contact-free means to extract lateral ion diffusion coefficients while spatially distinguishing defect types across the sample.

---

## 105. Eq.Bot: Enhance Robotic Manipulation Learning via Group Equivariant Canonicalization

**论文链接:** [http://arxiv.org/abs/2511.15194v1](http://arxiv.org/abs/2511.15194v1)

**作者:** Jian Deng, Yuandong Wang, Yangfu Zhu, Tao Feng, Tianyu Wo, Zhenzhou Shao

**发布时间:** 2025-11-19

**备注:** 12 pages, 4 figures and 3 tables

### GPT解析

### 总结

本研究提出了Eq.Bot，一种基于SE(2)群等变性理论的通用规范化框架，用于机器人操作学习，能够处理空间变换问题并提升性能。

### 背景

机器人操作系统在多个领域被广泛应用，但现有多模态学习框架缺乏几何一致性保证，难以处理旋转和平移等空间变换。现有方法虽有尝试通过特定架构修改引入等变性，但存在实现复杂、计算成本高和可移植性差的问题。

### 目的

开发一种通用框架，使模型具有空间等变性，而无需对现有架构进行修改。

### 方法

提出Eq.Bot框架，该框架将观测值转换为规范空间，应用现有策略，然后将得到的动作映射回原始空间。作为一种模型无关的解决方案，它基于SE(2)群等变性理论，旨在赋予模型空间等变性而不需要架构修改。

### 主要发现

Eq.Bot在各种基于CNN（如CLIPort）和基于Transformer（如OpenVLA-OFT）架构的机器人操作任务上均优于现有方法，最大性能改进可达50.0%。

### 结论

Eq.Bot是一种有效的通用解决方案，能够解决机器人操作学习中的空间变换问题，显著提升性能，且不依赖特定架构。

### 翻译

机器人操作系统正被部署到各个不同领域。然而，现有的多模态学习框架缺乏内在的几何一致性保证，难以处理旋转和平移等空间变换。虽然最近的工作试图通过特定的架构修改来引入等变性，但这些方法存在实现复杂度高、计算成本高和可移植性差的问题。受人类空间推理认知过程的启发，我们提出了Eq.Bot，一个基于SE(2)群等变性理论的通用规范化框架，用于机器人操作学习。我们的框架将观测值转换为规范空间，应用现有策略，然后将得到的动作映射回原始空间。作为一种模型无关的解决方案，Eq.Bot旨在在不进行架构修改的情况下赋予模型空间等变性。大量实验证明，在各种机器人操作任务上，Eq.Bot在基于CNN（如CLIPort）和基于Transformer（如OpenVLA-OFT）的架构上都优于现有方法，其中最大改进可达50.0%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作学习框架中缺乏几何一致性的问题，即现有系统难以处理物体旋转、平移等空间变换。这个问题在现实中非常重要，因为机器人需要在各种环境中操作物体，而这些物体的空间位置和方向经常变化。缺乏对空间变换的稳健性会导致机器人在面对不同视角或物体方向时表现不佳，限制了机器人在复杂环境中的适应性和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受到人类认知过程中空间推理的启发，人类会将物体映射到心理的'标准空间'，在这个标准化表示中执行动作，然后将结果转换回物理空间。作者从群等变理论的角度对现有多模态框架进行了系统的理论分析，发现现有方法如GEM和EquAct虽然引入了等变性，但需要对架构进行大量重新设计，增加了实现复杂性并降低了可移植性。作者借鉴了现有的等变学习方法，如群等变卷积神经网络(G-CNNs)，但设计了一个通用的标准化框架，可以与现有策略无缝集成，无需修改基础架构。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用SE(2)群等变理论构建一个通用的标准化框架，将观察结果转换为标准空间，应用现有策略，然后将动作映射回原始空间。这种方法不需要对现有架构进行修改，而是作为一个即插即用的组件。整体实现流程分为三个阶段：1) 输入观察通过群等变网络进行标准化，得到标准化观察和估计的变换参数；2) 标准化后的观察通过基础操作策略生成在标准化坐标系中的动作预测；3) 最后，标准化动作通过逆变换恢复到原始坐标对齐，用于执行。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出了系统的理论分析，分析了当前多模态机器人学习框架中的等变性不足，并严格证明了机器人操作的群理论基础；2) 提出了通用的、与模型无关的标准化框架，可以在不进行架构修改的情况下提供空间泛化增强；3) 在不同骨干架构的各种机器人操作任务上进行了全面的实验验证。相比之前的工作，Eq.Bot作为一个即插即用的组件，可以无缝集成到现有的多模态架构中，而现有方法如GEM和EquAct需要对架构进行大量重新设计。实验表明，Eq.Bot在大多数评估任务上表现优异，例如将CLIPort的成功率从62.4%提高到93.6%，提升了约50.0%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Eq.Bot提出了一种基于群等变理论的通用标准化框架，通过将观察结果转换为标准空间、应用现有策略并将动作映射回原始空间，显著提高了机器人操作学习对空间变换的鲁棒性，无需对现有架构进行修改。'}


### 论文摘要

Robotic manipulation systems are increasingly deployed across diverse domains. Yet existing multi-modal learning frameworks lack inherent guarantees of geometric consistency, struggling to handle spatial transformations such as rotations and translations. While recent works attempt to introduce equivariance through bespoke architectural modifications, these methods suffer from high implementation complexity, computational cost, and poor portability. Inspired by human cognitive processes in spatial reasoning, we propose Eq.Bot, a universal canonicalization framework grounded in SE(2) group equivariant theory for robotic manipulation learning. Our framework transforms observations into a canonical space, applies an existing policy, and maps the resulting actions back to the original space. As a model-agnostic solution, Eq.Bot aims to endow models with spatial equivariance without requiring architectural modifications. Extensive experiments demonstrate the superiority of Eq.Bot under both CNN-based (e.g., CLIPort) and Transformer-based (e.g., OpenVLA-OFT) architectures over existing methods on various robotic manipulation tasks, where the most significant improvement can reach 50.0%.

---

## 106. BBox DocVQA: A Large Scale Bounding Box Grounded Dataset for Enhancing Reasoning in Document Visual Question Answer

**论文链接:** [http://arxiv.org/abs/2511.15090v1](http://arxiv.org/abs/2511.15090v1)

**作者:** Wenhan Yu, Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Lei Sha, Deguo Xia, Jizhou Huang

**发布时间:** 2025-11-19

**备注:** 22 pages, 4 figures

### GPT解析

### 总结

本文提出了BBox DocVQA数据集，一个大规模的、基于边界框的文档视觉问答数据集，旨在增强视觉语言模型的空间推理和证据定位能力。通过自动构建流程和人工验证，创建了包含3.6K文档和32K QA对的数据集。评估显示现有VLM在空间定位方面存在挑战，但微调后性能显著提升。

### 背景

现有的DocVQA数据集局限于页面级别，缺乏细粒度的空间定位能力，限制了视觉语言模型的可解释性和推理能力。

### 目的

解决现有DocVQA数据集的空间定位局限，增强视觉语言模型的空间推理和证据定位能力。

### 方法

提出'Segment Judge and Generate'自动构建流程，整合区域分割模型、语义判断VLM和问答生成VLM，通过人工验证确保质量，创建基于边界框的QA数据集。

### 主要发现

1) 现有先进VLM在BBox DocVQA上的测试显示空间定位和推理准确性存在持续挑战；2) 在BBox DocVQA上进行微调显著提高了边界框定位和答案生成能力。

### 结论

BBox DocVQA数据集能有效增强视觉语言模型的空间推理能力，为可解释和空间定位的视觉语言推理研究提供了新资源。

### 翻译

文档视觉问答（DocVQA）是多模态文档理解的基本任务，也是视觉语言推理的关键测试平台。然而，大多数现有的DocVQA数据集仅限于页面级别，缺乏细粒度的空间定位，限制了视觉语言模型的可解释性和推理能力。为解决这一差距，我们引入了BBox DocVQA，这是一个大规模的、基于边界框的数据集，旨在增强视觉文档中的空间推理和证据定位能力。我们进一步提出了一个自动构建流程'Segment Judge and Generate'，它整合了用于区域分割的分割模型、用于语义判断的VLM以及用于问答生成的另一个先进VLM，随后进行人工验证以确保质量。生成的数据集包含3.6K个多样化文档和32K个QA对，涵盖了单区域/多区域以及单页面/多页面的场景。每个QA实例都基于明确的边界框进行定位，能够对空间语义对齐进行细粒度评估。在BBox DocVQA上对多个最先进的VLM（如GPT-4、Qwen2.5 VL和InternVL）进行基准测试，揭示了空间定位和推理准确性方面的持续挑战。此外，在BBox DocVQA上进行微调显著提高了边界框定位和答案生成能力，验证了其增强VLM推理能力的有效性。我们的数据集和代码将公开发布，以促进可解释和空间定位的视觉语言推理研究。


### 论文摘要

Document Visual Question Answering (DocVQA) is a fundamental task for multimodal document understanding and a key testbed for vision language reasoning. However, most existing DocVQA datasets are limited to the page level and lack fine grained spatial grounding, constraining the interpretability and reasoning capability of Vision Language Models (VLMs). To address this gap, we introduce BBox DocVQA a large scale, bounding box grounded dataset designed to enhance spatial reasoning and evidence localization in visual documents. We further present an automated construction pipeline, Segment Judge and Generate, which integrates a segment model for region segmentation, a VLM for semantic judgment, and another advanced VLM for question answer generation, followed by human verification for quality assurance. The resulting dataset contains 3.6 K diverse documents and 32 K QA pairs, encompassing single and multi region as well as single and multi page scenarios. Each QA instance is grounded on explicit bounding boxes, enabling fine grained evaluation of spatial semantic alignment. Benchmarking multiple state of the art VLMs (e.g., GPT 5, Qwen2.5 VL, and InternVL) on BBox DocVQA reveals persistent challenges in spatial grounding and reasoning accuracy. Furthermore, fine tuning on BBox DocVQA substantially improves both bounding box localization and answer generation, validating its effectiveness for enhancing the reasoning ability of VLMs. Our dataset and code will be publicly released to advance research on interpretable and spatially grounded vision language reasoning.

---

## 107. Knowledge Graphs as Structured Memory for Embedding Spaces: From Training Clusters to Explainable Inference

**论文链接:** [http://arxiv.org/abs/2511.14961v1](http://arxiv.org/abs/2511.14961v1)

**作者:** Artur A. Oliveira, Mateus Espadoto, Roberto M. Cesar, Roberto Hirata

**发布时间:** 2025-11-18

**备注:** Submitted to GRIVAPP 2026 (21st International Conference on Computer Graphics, Interaction, Visualization Theory and Applications), Marbella, Spain, March 9-11 2026

### GPT解析

### 总结

本文提出了一种名为Graph Memory (GM)的结构化非参数框架，通过在区域级原型上添加关系记忆来增强基于嵌入的推理，实现了实例检索、原型推理和标签传播的统一，在保持竞争力的同时提高了校准效果并减少了样本需求。

### 背景

传统的嵌入推理方法通常将每个训练实例视为独立的，缺乏对嵌入空间中关系结构的显式建模，导致难以平衡局部证据和全局一致性。

### 目的

开发一种能够统一实例检索、原型推理和标签传播的框架，提高模型的校准效果和决策边界平滑度，同时减少对样本数量的依赖。

### 方法

Graph Memory框架将嵌入空间总结为带有可靠性指示符的原型节点，并通过编码几何和上下文关系的边连接这些节点，形成一个结构化的记忆系统。

### 主要发现

在包括乳腺组织病理学在内的多个数据集上，GM的准确性与kNN和标签传播相当，但提供了更好的校准和更平滑的决策边界，且使用的样本数量减少了一个数量级。

### 结论

通过明确建模可靠性和关系结构，GM为非参数学习中局部证据和全局一致性之间提供了原则性的桥梁，是一种高效且可解释的推理框架。

### 翻译

我们引入了图记忆(GM)，这是一种结构化的非参数框架，通过在区域级原型上添加紧凑的关系记忆来增强基于嵌入的推理。GM不是将每个训练实例视为孤立的，而是将嵌入空间总结为带有可靠性指示符的原型节点，并通过编码几何和上下文关系的边连接这些节点。这种设计将实例检索、基于原型的推理和基于图的标签传播统一在一个单一的归纳模型中，支持高效推理和忠实解释。在包括乳腺组织病理学(IDC)在内的合成和真实数据集上的实验表明，GM的准确性与kNN和标签传播具有竞争力，同时提供了更好的校准和更平滑的决策边界，且使用的样本数量减少了一个数量级。通过明确建模可靠性和关系结构，GM为非参数学习中局部证据和全局一致性之间提供了原则性的桥梁。


### 论文摘要

We introduce Graph Memory (GM), a structured non-parametric framework that augments embedding-based inference with a compact, relational memory over region-level prototypes. Rather than treating each training instance in isolation, GM summarizes the embedding space into prototype nodes annotated with reliability indicators and connected by edges that encode geometric and contextual relations. This design unifies instance retrieval, prototype-based reasoning, and graph-based label propagation within a single inductive model that supports both efficient inference and faithful explanation. Experiments on synthetic and real datasets including breast histopathology (IDC) show that GM achieves accuracy competitive with $k$NN and Label Spreading while offering substantially better calibration and smoother decision boundaries, all with an order of magnitude fewer samples. By explicitly modeling reliability and relational structure, GM provides a principled bridge between local evidence and global consistency in non-parametric learning.

---

## 108. RocSync: Millisecond-Accurate Temporal Synchronization for Heterogeneous Camera Systems

**论文链接:** [http://arxiv.org/abs/2511.14948v1](http://arxiv.org/abs/2511.14948v1)

**作者:** Jaro Meyer, Frédéric Giraud, Joschua Wüthrich, Marc Pollefeys, Philipp Fürnstahl, Lilian Calvet

**发布时间:** 2025-11-18

**备注:** 16 pages, 6 figures

### GPT解析

### 总结

本研究提出了一种低成本、通用目的的多视角视频流同步方法，能够在异构摄像机系统间实现毫秒级时间对准，支持可见光和红外模式，并显著改善了下游计算机视觉任务性能。

### 背景

多视角视频流的精确时空对准对于多视角3D重建、姿态估计和场景理解等多种动态场景应用至关重要。然而，同步多个摄像机，特别是在异构设置中（专业和消费级设备结合、可见光和红外传感器结合、有音频和无音频系统结合），仍然是一个重大挑战。在真实世界环境中，受控捕获条件不可行时，这一限制尤为明显。

### 目的

开发一种低成本、通用目的的同步方法，能够在不同摄像机系统之间实现毫秒级时间对准，同时支持可见光(RGB)和红外(IR)模式，解决异构摄像机系统的同步问题。

### 方法

提出使用定制的'LED时钟'解决方案，通过红色和红外LED编码时间，允许从记录的帧中视觉解码曝光窗口（开始和结束时间）以实现毫秒级同步。

### 主要发现

与硬件同步相比，该方法在多次记录中实现了1.34毫秒的均方根误差；在进一步的实验中，该方法优于基于光、音频和时间码的同步方法；该方法直接改进了下游计算机视觉任务，包括多视角姿态估计和3D重建；在涉及超过25个跨越IR和RGB模式的异构摄像机的大规模手术记录中验证了系统有效性。

### 结论

该解决方案简化并优化了同步流程，扩展了在不受约束环境中（包括工业和临床应用）获取高级基于视觉的感知能力，为多视角视频分析提供了实用的同步工具。

### 翻译

多视角视频流的精确时空对准对于多种动态场景应用（如多视角3D重建、姿态估计和场景理解）至关重要。然而，同步多个摄像机仍然是一个重大挑战，特别是在异构设置中（专业和消费级设备结合、可见光和红外传感器结合、有音频和无音频系统结合），其中常见的硬件同步功能通常不可用。这一限制在真实世界环境中尤为明显，因为受控捕获条件不可行。在这项工作中，我们提出了一种低成本、通用目的的同步方法，能够在不同摄像机系统之间实现毫秒级时间对准，同时支持可见光(RGB)和红外(IR)模式。所提出的解决方案使用定制的LED时钟，通过红色和红外LED编码时间，允许从记录的帧中视觉解码曝光窗口（开始和结束时间）以实现毫秒级同步。我们将我们的方法与硬件同步进行了基准测试，在多次记录中实现了1.34毫秒的均方根误差。在进一步的实验中，我们的方法优于基于光、音频和时间码的同步方法，并直接改进了下游计算机视觉任务，包括多视角姿态估计和3D重建。最后，我们在涉及超过25个跨越IR和RGB模式的异构摄像机的大规模手术记录中验证了该系统。该解决方案简化并优化了同步流程，扩展了在不受约束环境中（包括工业和临床应用）获取高级基于视觉的感知能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决异构相机系统（专业级与消费级设备、可见光与红外传感器等组合）中的高精度时间同步问题。这个问题在现实中非常重要，因为在3D重建、姿态估计和场景理解等应用中，即使只有几毫秒的时间偏移也会导致显著的重建错误、模糊伪影或不准确的运动轨迹。在医疗、工业等实际场景中，精确同步不同类型的相机数据对分析和决策至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者系统分析了现有同步方法（硬件同步、时间码同步、内容同步）的局限性，包括精度不足、兼容性差、依赖特定场景等问题。在此基础上，作者设计了一种创新的基于LED时钟的解决方案，借鉴了视觉时间戳编码的思想，但通过创新的硬件设计解决了现有方法的局限性。作者还利用了计算机视觉中常用的ArUco标记检测和单应性校正技术，以及线性最小二乘法和鲁棒回归来估计时间偏移和漂移。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个特殊的LED时钟设备，通过红光和红外LED编码时间信息，相机捕获这些图像后解码曝光窗口的开始和结束时间，从而计算每个相机的时间偏移和漂移，实现高精度同步。整体流程包括：1)硬件部分：25×25厘米的PCB板上有100个环形LED和16个二进制计数器LED，中央有ArUco标记，四角有定位LED；2)软件处理：检测ArUco标记→粗略重投影→定位角点LED→精确重投影→解码LED→时间戳拟合；3)使用：录制开始和结束时将LED时钟依次放在每个相机前，处理视频提取时间戳并计算同步参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)支持异构相机系统（RGB和红外）的通用解决方案；2)通过LED环形设计和二进制计数器实现毫秒级精度的全局时间戳编码；3)创新的硬件设计结合红光和红外LED；4)鲁棒的软件处理流程；5)在大规模实际应用中验证。相比之前工作，不同之处在于：无需专用硬件和复杂布线；不受时钟漂移影响；不依赖场景内容或重叠视野；支持红外相机；提供子帧精度和插值能力；成本更低且更易于部署。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于定制LED时钟的创新方法，实现了异构相机系统的毫秒级精确时间同步，为多视图3D重建和姿态估计等计算机视觉任务提供了高精度的时空对齐解决方案，并在大规模实际应用中得到了验证。'}


### 论文摘要

Accurate spatiotemporal alignment of multi-view video streams is essential for a wide range of dynamic-scene applications such as multi-view 3D reconstruction, pose estimation, and scene understanding. However, synchronizing multiple cameras remains a significant challenge, especially in heterogeneous setups combining professional and consumer-grade devices, visible and infrared sensors, or systems with and without audio, where common hardware synchronization capabilities are often unavailable. This limitation is particularly evident in real-world environments, where controlled capture conditions are not feasible. In this work, we present a low-cost, general-purpose synchronization method that achieves millisecond-level temporal alignment across diverse camera systems while supporting both visible (RGB) and infrared (IR) modalities. The proposed solution employs a custom-built \textit{LED Clock} that encodes time through red and infrared LEDs, allowing visual decoding of the exposure window (start and end times) from recorded frames for millisecond-level synchronization. We benchmark our method against hardware synchronization and achieve a residual error of 1.34~ms RMSE across multiple recordings. In further experiments, our method outperforms light-, audio-, and timecode-based synchronization approaches and directly improves downstream computer vision tasks, including multi-view pose estimation and 3D reconstruction. Finally, we validate the system in large-scale surgical recordings involving over 25 heterogeneous cameras spanning both IR and RGB modalities. This solution simplifies and streamlines the synchronization pipeline and expands access to advanced vision-based sensing in unconstrained environments, including industrial and clinical applications.

---

## 109. FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding

**论文链接:** [http://arxiv.org/abs/2511.14901v1](http://arxiv.org/abs/2511.14901v1)

**作者:** Zhenshi Li, Weikang Yu, Dilxat Muhtar, Xueliang Zhang, Pengfeng Xiao, Pedram Ghamisi, Xiao Xiang Zhu

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为FarSLIP的细粒度对齐遥感语言-图像预训练框架，解决了现有CLIP模型在遥感领域细粒度细节捕获能力有限的问题，通过构建多粒度数据集和创新的patch-to-patch蒸馏方法，显著提升了模型在遥感任务上的性能。

### 背景

CLIP的全局对齐限制了其捕获细粒度细节的能力，当前遥感特定的CLIP变体仍继承这种有限的空间感知能力。作者识别出两个关键限制：(1)当前RS图像-文本数据集从对象级标签生成全局标题，导致原始对象级监督未被充分利用；(2)通用领域的区域-文本对齐方法直接应用于RS数据通常会导致性能下降。

### 目的

解决当前RS CLIP变体的空间感知能力有限问题，提高RS领域的细粒度视觉-语言对齐能力，提升模型在遥感任务上的表现。

### 方法

构建了首个多粒度RS图像-文本数据集MGRS-200k，包含丰富的对象级文本监督；提出FarSLIP框架，采用patch-to-patch蒸馏而非常见的patch-to-CLS自蒸馏来对齐局部和全局视觉线索；使用基于CLS令牌的区域-类别对齐而非显式的patch级对齐，以有效利用区域-文本监督。

### 主要发现

当前显式区域-文本对齐方法由于CLIP语义连贯性的严重下降而表现不佳；FarSLIP通过patch-to-patch蒸馏和基于CLS令牌的对齐方法，在保持语义连贯性的同时提高了特征判别力，增强了空间感知能力。

### 结论

FarSLIP在RS开放词汇语义分割、零样本分类和图像-文本检索等任务上达到了新的最先进水平，证明了其在遥感领域的有效性和优越性。

### 翻译

由于CLIP的全局对齐限制了其捕获细粒度细节的能力，最近的工作集中在增强其区域-文本对齐上。然而，当前遥感(RS)特定的CLIP变体仍然继承这种有限的空间感知能力。我们确定了这背后的两个关键限制：(1)当前RS图像-文本数据集从对象级标签生成全局标题，导致原始对象级监督未被充分利用；(2)尽管区域-文本对齐方法在通用领域取得成功，但直接应用于RS数据通常会导致性能下降。为解决这些问题，我们构建了首个多粒度RS图像-文本数据集MGRS-200k，为RS区域-类别对齐提供了丰富的对象级文本监督。我们进一步研究了现有的细粒度CLIP调优策略，发现当前显式区域-文本对齐方法，无论直接还是间接，都由于CLIP语义连贯性的严重下降而表现不佳。基于此，我们提出FarSLIP，一种细粒度对齐的RS语言-图像预训练框架。FarSLIP采用patch-to-patch蒸馏而非常用的patch-to-CLS自蒸馏，对齐局部和全局视觉线索，在保持语义连贯性的同时提高了特征判别力。此外，为有效利用区域-文本监督，它采用简单的基于CLS令牌的区域-类别对齐而非显式的patch级对齐，进一步增强了空间感知能力。FarSLIP改进了RS领域的细粒度视觉-语言对齐能力，不仅在RS开放词汇语义分割上，而且在零样本分类和图像-文本检索等图像级任务上设置了新的最先进水平。我们的数据集、代码和模型可在https://github.com/NJU-LHRS/FarSLIP获取。


### 论文摘要

As CLIP's global alignment limits its ability to capture fine-grained details, recent efforts have focused on enhancing its region-text alignment. However, current remote sensing (RS)-specific CLIP variants still inherit this limited spatial awareness. We identify two key limitations behind this: (1) current RS image-text datasets generate global captions from object-level labels, leaving the original object-level supervision underutilized; (2) despite the success of region-text alignment methods in general domain, their direct application to RS data often leads to performance degradation. To address these, we construct the first multi-granularity RS image-text dataset, MGRS-200k, featuring rich object-level textual supervision for RS region-category alignment. We further investigate existing fine-grained CLIP tuning strategies and find that current explicit region-text alignment methods, whether in a direct or indirect way, underperform due to severe degradation of CLIP's semantic coherence. Building on these, we propose FarSLIP, a Fine-grained Aligned RS Language-Image Pretraining framework. Rather than the commonly used patch-to-CLS self-distillation, FarSLIP employs patch-to-patch distillation to align local and global visual cues, which improves feature discriminability while preserving semantic coherence. Additionally, to effectively utilize region-text supervision, it employs simple CLS token-based region-category alignment rather than explicit patch-level alignment, further enhancing spatial awareness. FarSLIP features improved fine-grained vision-language alignment in RS domain and sets a new state of the art not only on RS open-vocabulary semantic segmentation, but also on image-level tasks such as zero-shot classification and image-text retrieval. Our dataset, code, and models are available at https://github.com/NJU-LHRS/FarSLIP.

---

## 110. GeoSceneGraph: Geometric Scene Graph Diffusion Model for Text-guided 3D Indoor Scene Synthesis

**论文链接:** [http://arxiv.org/abs/2511.14884v1](http://arxiv.org/abs/2511.14884v1)

**作者:** Antonio Ruiz, Tao Wu, Andrew Melnik, Qing Cheng, Xuqin Wang, Lu Liu, Yongliang Wang, Yanfeng Zhang, Helge Ritter

**发布时间:** 2025-11-18

### GPT解析

### 总结

GeoSceneGraph是一种利用3D场景的图结构和几何对称性从文本提示合成3D场景的方法，无需依赖预定义关系类，性能可与使用真实关系的方法相媲美。

### 背景

从文本提示合成室内3D场景的方法在电影制作、室内设计、视频游戏、虚拟现实和智能体训练数据生成等领域有广泛应用。现有方法要么从头训练生成模型，要么利用视觉语言模型(VLMs)。VLMs在处理复杂提示时表现良好，但在资源受限设备上部署需要更小模型。许多生成方法忽略了室内场景的固有图结构，而结合场景图的方法要么需要用户提供语义图，要么依赖真实关系标注，限制了多样化对象交互的捕捉。

### 目的

解决现有室内3D场景合成方法的局限性，提出一种不依赖预定义关系类且能捕捉场景图结构的方法。

### 方法

GeoSceneGraph方法利用3D场景的图结构和几何对称性合成3D场景，模型基于等变图神经网络(EGNNs)构建，并提出了使EGNNs能够基于文本特征进行条件化的简单有效策略。

### 主要发现

尽管不使用真实关系标注，GeoSceneGraph的性能与使用真实关系的方法相当，通过消融研究验证了设计的有效性。

### 结论

GeoSceneGraph提供了一种有效的室内3D场景合成方法，解决了现有方法在场景连贯性、真实性和对象交互多样性方面的局限性，同时不依赖用户提供的语义图或真实关系标注。

### 翻译

从文本提示合成室内3D场景的方法在电影制作、室内设计、视频游戏、虚拟现实和用于训练智能体的合成数据生成等领域有广泛的应用。现有的方法通常要么从头开始训练生成模型，要么利用视觉语言模型(VLMs)。虽然VLMs能够实现强大的性能，特别是在处理复杂或开放式提示时，但在扩展现实(XR)眼镜或手机等资源受限设备上部署仍需要更小的专用模型。然而，许多从头训练的生成方法忽略了室内场景的固有图结构，这可能限制场景的连贯性和真实性。相反，结合场景图的方法要么需要用户提供语义图，这通常不方便且有限制，要么依赖于真实的关系标注，限制了它们捕捉更多样化对象交互的能力。为了解决这些挑战，我们引入了GeoSceneGraph，一种利用3D场景的图结构和几何对称性从文本提示合成3D场景的方法，无需依赖预定义的关系类别。尽管不使用真实关系，GeoSceneGraph的性能可与使用真实关系的方法相媲美。我们的模型基于等变图神经网络(EGNNs)构建，但现有的EGNN方法通常仅限于低维条件化，并且不是为处理文本等复杂模态而设计的。我们提出了一种简单有效的策略，使EGNNs能够基于文本特征进行条件化，并通过消融研究验证了我们的设计。


### 论文摘要

Methods that synthesize indoor 3D scenes from text prompts have wide-ranging applications in film production, interior design, video games, virtual reality, and synthetic data generation for training embodied agents. Existing approaches typically either train generative models from scratch or leverage vision-language models (VLMs). While VLMs achieve strong performance, particularly for complex or open-ended prompts, smaller task-specific models remain necessary for deployment on resource-constrained devices such as extended reality (XR) glasses or mobile phones. However, many generative approaches that train from scratch overlook the inherent graph structure of indoor scenes, which can limit scene coherence and realism. Conversely, methods that incorporate scene graphs either demand a user-provided semantic graph, which is generally inconvenient and restrictive, or rely on ground-truth relationship annotations, limiting their capacity to capture more varied object interactions. To address these challenges, we introduce GeoSceneGraph, a method that synthesizes 3D scenes from text prompts by leveraging the graph structure and geometric symmetries of 3D scenes, without relying on predefined relationship classes. Despite not using ground-truth relationships, GeoSceneGraph achieves performance comparable to methods that do. Our model is built on equivariant graph neural networks (EGNNs), but existing EGNN approaches are typically limited to low-dimensional conditioning and are not designed to handle complex modalities such as text. We propose a simple and effective strategy for conditioning EGNNs on text features, and we validate our design through ablation studies.

---

## 111. First Frame Is the Place to Go for Video Content Customization

**论文链接:** [http://arxiv.org/abs/2511.15700v1](http://arxiv.org/abs/2511.15700v1)

**作者:** Jingxi Chen, Zongxia Li, Zhichao Liu, Guangyao Shi, Xiyang Wu, Fuxiao Liu, Cornelia Fermuller, Brandon Y. Feng, Yiannis Aloimonos

**发布时间:** 2025-11-19

**备注:** Project Website: https://firstframego.github.io/

### GPT解析

### 总结

作者揭示了视频生成模型中第一帧的新角色，并展示了这一见解如何用于实现高效的视频内容定制。

### 背景

传统上，视频生成模型中的第一帧被视为视频的时空起点，仅作为后续动画的种子。

### 目的

挑战传统观点，探索第一帧在视频生成模型中的真正作用。

### 方法

通过实验验证第一帧作为概念记忆缓冲区的观点，展示仅使用20-50个训练示例即可实现视频内容定制，无需架构更改或大规模微调。

### 主要发现

视频模型隐式地将第一帧视为概念记忆缓冲区，用于在生成过程中存储视觉实体以便后续重用。

### 结论

这揭示了视频生成模型基于参考的视频定制的一个强大但被忽视的能力。

### 翻译

第一帧在视频生成模型中扮演什么角色？传统上，它被视为视频的时空起点，仅作为后续动画的种子。在这项工作中，我们揭示了一个根本不同的视角：视频模型隐式地将第一帧视为概念记忆缓冲区，用于在生成过程中存储视觉实体以便后续重用。利用这一见解，我们证明可以在各种场景中实现强大且通用的视频内容定制，仅使用20-50个训练示例，无需架构更改或大规模微调。这揭示了视频生成模型基于参考的视频定制的一个强大但被忽视的能力。


### 论文摘要

What role does the first frame play in video generation models? Traditionally, it's viewed as the spatial-temporal starting point of a video, merely a seed for subsequent animation. In this work, we reveal a fundamentally different perspective: video models implicitly treat the first frame as a conceptual memory buffer that stores visual entities for later reuse during generation. Leveraging this insight, we show that it's possible to achieve robust and generalized video content customization in diverse scenarios, using only 20-50 training examples without architectural changes or large-scale finetuning. This unveils a powerful, overlooked capability of video generation models for reference-based video customization.

---

## 112. GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI

**论文链接:** [http://arxiv.org/abs/2511.15658v1](http://arxiv.org/abs/2511.15658v1)

**作者:** Naomi Simumba, Nils Lehmann, Paolo Fraccaro, Hamed Alemohammad, Geeth De Mel, Salman Khan, Manil Maskey, Nicolas Longepe, Xiao Xiang Zhu, Hannah Kerner, Juan Bernabe-Moreno, Alexander Lacoste

**发布时间:** 2025-11-19

### GPT解析

### 总结

GEO-Bench-2是一个综合性的评估框架，用于地理空间基础模型(GeoFMs)，涵盖多种任务和数据集，通过'能力'组对模型进行排名，支持公平比较和方法创新。

### 背景

地理空间基础模型(GeoFMs)正在改变地球观测(Earth Observation, EO)领域，但缺乏标准化的评估协议。

### 目的

解决GeoFMs评估缺乏标准化协议的问题，提供一种描述性且灵活的评估协议，支持公平比较和方法创新，并确定未来工作需要改进的领域。

### 方法

创建一个综合评估框架，涵盖分类、分割、回归、目标检测和实例分割等任务，跨越19个许可宽松的数据集；引入'能力'组来对具有共同特征的数据集上的模型进行排名；定义一种描述性且灵活的评估协议。

### 主要发现

没有单一模型在所有任务上都占主导地位；在自然图像上预训练的模型在高分辨率任务上表现优异；特定于地球观测的模型在多光谱应用上表现更好；最佳模型选择取决于任务需求、数据模态和约束条件。

### 结论

开发一个在所有任务上都表现良好的单一GeoFM模型仍然是未来研究的目标；GEO-Bench-2能够实现针对特定用例的、可复现的GeoFM评估。

### 翻译

地理空间基础模型(GeoFMs)正在改变地球观测(Earth Observation)领域，但评估缺乏标准化协议。GEO-Bench-2通过一个综合框架解决了这一问题，该框架涵盖分类、分割、回归、目标检测和实例分割，跨越19个许可宽松的数据集。我们引入'能力'组，以便在具有共同特征的数据集上对模型进行排名(如分辨率、波段、时间性)。这使用户能够识别每个能力中表现优异的模型，并确定未来工作需要改进的领域。为了支持公平比较和方法创新，我们定义了一种描述性且灵活的评估协议。这不仅确保了基准测试的一致性，还促进了模型适应策略的研究，这是推进GeoFMs用于下游任务的关键且开放的挑战。我们的实验表明，没有单一模型在所有任务上都占主导地位，这证实了架构设计和预训练过程中选择的具体性。虽然在自然图像上预训练的模型(ConvNext ImageNet, DINO V3)在高分辨率任务上表现优异，但特定于地球观测的模型(TerraMind, Prithvi, 和 Clay)在农业和灾害响应等多光谱应用上表现更好。这些发现表明，最佳模型选择取决于任务需求、数据模态和约束条件。这表明，开发一个在所有任务上都表现良好的单一GeoFM模型仍然是未来研究的目标。GEO-Bench-2能够实现针对特定用例的、可复现的GeoFM评估。GEO-Bench-2的代码、数据和排行榜已在宽松许可下公开发布。


### 论文摘要

Geospatial Foundation Models (GeoFMs) are transforming Earth Observation (EO), but evaluation lacks standardized protocols. GEO-Bench-2 addresses this with a comprehensive framework spanning classification, segmentation, regression, object detection, and instance segmentation across 19 permissively-licensed datasets. We introduce ''capability'' groups to rank models on datasets that share common characteristics (e.g., resolution, bands, temporality). This enables users to identify which models excel in each capability and determine which areas need improvement in future work. To support both fair comparison and methodological innovation, we define a prescriptive yet flexible evaluation protocol. This not only ensures consistency in benchmarking but also facilitates research into model adaptation strategies, a key and open challenge in advancing GeoFMs for downstream tasks.   Our experiments show that no single model dominates across all tasks, confirming the specificity of the choices made during architecture design and pretraining. While models pretrained on natural images (ConvNext ImageNet, DINO V3) excel on high-resolution tasks, EO-specific models (TerraMind, Prithvi, and Clay) outperform them on multispectral applications such as agriculture and disaster response. These findings demonstrate that optimal model choice depends on task requirements, data modalities, and constraints. This shows that the goal of a single GeoFM model that performs well across all tasks remains open for future research. GEO-Bench-2 enables informed, reproducible GeoFM evaluation tailored to specific use cases. Code, data, and leaderboard for GEO-Bench-2 are publicly released under a permissive license.

---

## 113. The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification

**论文链接:** [http://arxiv.org/abs/2511.15622v1](http://arxiv.org/abs/2511.15622v1)

**作者:** Dante Francisco Wasmuht, Otto Brookes, Maximillian Schall, Pablo Palencia, Chris Beirne, Tilo Burghardt, Majid Mirmehdi, Hjalmar Kühl, Mimi Arandjelovic, Sam Pottie, Peter Bermant, Brandon Asheim, Yi Jin Toh, Adam Elzinga, Jason Holmberg, Andrew Whitworth, Eleanor Flatt, Laura Gustafson, Chaitanya Ryali, Yuan-Ting Hu, Baishan Guo, Andrew Westbury, Kate Saenko, Didac Suris

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文介绍了SA-FARI，目前最大的野生动物开源多动物跟踪数据集，包含来自四大洲741个地点的11,609个相机陷阱视频，跨越约10年(2014-2024)时间，涵盖99个物种类别。数据集包含约46小时的密集标注素材，有16,224个masklet身份和942,702个个体边界框、分割掩码和物种标签。

### 背景

自动视频分析对野生动物保护至关重要，多动物跟踪(MAT)是该领域的基础任务，支撑个体重新识别和行为识别等应用。然而，现有数据集规模有限，仅限于少数物种，或缺乏足够的时空多样性，没有适合训练通用MAT模型的基准。

### 目的

为了解决现有数据集的局限性，引入SA-FARI数据集，提供具有高物种多样性、多区域覆盖和高质量时空注释的大型数据集，为推进野外可推广的多动物跟踪奠定新基础。

### 方法

收集并标注11,609个相机陷阱视频，来自741个地点跨越四大洲，时间跨度约10年。每个视频详尽标注，产生约46小时的密集标注素材。使用最先进的视觉语言模型(包括SAM 3)进行基准测试，使用物种特定和通用动物提示评估，并与专门为野生动物分析开发的纯视觉方法比较。

### 主要发现

SA-FARI是首个结合高物种多样性、多区域覆盖和高质量时空注释的大型数据集。通过使用最先进的视觉语言模型和专门为野生动物分析开发的纯视觉方法进行基准测试，展示了该数据集的有效性和实用性。

### 结论

SA-FARI数据集填补了野生动物MAT领域大型、多样化数据集的空白，为训练和应用通用MAT模型提供了宝贵资源，将促进野生动物保护领域的研究和技术发展。

### 翻译

自动视频分析对野生动物保护至关重要。该领域的基础任务是多动物跟踪(MAT)，它支撑着个体重新识别和行为识别等应用。然而，现有的数据集在规模上有限，仅限于少数物种，或缺乏足够的时空多样性，没有适合训练适用于野生动物种群通用MAT模型的基准。为解决这一问题，我们引入了SA-FARI，这是目前最大的野生动物开源MAT数据集。它包含来自四大洲741个地点约10年(2014-2024)期间收集的11,609个相机陷阱视频，涵盖99个物种类别。每个视频都经过了详尽标注，最终产生约46小时的密集标注素材，包含16,224个masklet身份和942,702个个体边界框、分割掩码和物种标签。除了任务特定的标注外，我们还发布了每个视频的匿名化相机陷阱位置信息。最后，我们使用最先进的视觉语言模型(包括SAM 3)对SA-FARI进行了全面的基准测试，使用物种特定和通用动物提示进行评估。我们还与专门为野生动物分析开发的纯视觉方法进行了比较。SA-FARI是首个结合高物种多样性、多区域覆盖和高质量时空注释的大型数据集，为推进野外可推广的多动物跟踪提供了新基础。该数据集可在conservationxlabs.com/SA-FARI获取。


### 论文摘要

Automated video analysis is critical for wildlife conservation. A foundational task in this domain is multi-animal tracking (MAT), which underpins applications such as individual re-identification and behavior recognition. However, existing datasets are limited in scale, constrained to a few species, or lack sufficient temporal and geographical diversity - leaving no suitable benchmark for training general-purpose MAT models applicable across wild animal populations. To address this, we introduce SA-FARI, the largest open-source MAT dataset for wild animals. It comprises 11,609 camera trap videos collected over approximately 10 years (2014-2024) from 741 locations across 4 continents, spanning 99 species categories. Each video is exhaustively annotated culminating in ~46 hours of densely annotated footage containing 16,224 masklet identities and 942,702 individual bounding boxes, segmentation masks, and species labels. Alongside the task-specific annotations, we publish anonymized camera trap locations for each video. Finally, we present comprehensive benchmarks on SA-FARI using state-of-the-art vision-language models for detection and tracking, including SAM 3, evaluated with both species-specific and generic animal prompts. We also compare against vision-only methods developed specifically for wildlife analysis. SA-FARI is the first large-scale dataset to combine high species diversity, multi-region coverage, and high-quality spatio-temporal annotations, offering a new foundation for advancing generalizable multianimal tracking in the wild. The dataset is available at $\href{https://www.conservationxlabs.com/sa-fari}{\text{conservationxlabs.com/SA-FARI}}$.

---

## 114. QTIS: A QAOA-Based Quantum Time Interval Scheduler

**论文链接:** [http://arxiv.org/abs/2511.15590v1](http://arxiv.org/abs/2511.15590v1)

**作者:** José A. Tirado-Domínguez, Eladio Gutiérrez, Oscar Plata

**发布时间:** 2025-11-19

**备注:** 19 pages, 13 figures

### GPT解析

### 总结

本研究提出了一种新的量子近似优化算法(QAOA)变体，称为量子时间区间调度器(QTIS)，用于解决时间区间受限和资源有限的任务调度问题。QTIS通过辅助量子电路动态检测和惩罚重叠任务，增强调度约束执行。研究探索了两种互补的重叠检测实现方法，并将问题哈密顿量分解为两个分量。实验表明，使用单独参数处理不同分量可提高解决方案质量。QTIS在复杂调度环境中具有推进混合量子-经典优化的潜力。

### 背景

任务调度在时间区间受限和资源有限的情况下仍然是一个基本挑战，存在于制造、物流、云计算和医疗保健等多个领域。

### 目的

提出一种新的量子近似优化算法(QAOA)变体，用于解决表述为二次无约束二元优化(QUBO)模型的任务调度问题。

### 方法

提出量子时间区间调度器(QTIS)方法，集成辅助量子电路动态检测和惩罚重叠任务；探索两种互补的重叠检测实现：基于RY旋转和CCNOT门的量子方法，以及依赖预处理区间比较的经典替代方案；将问题哈密顿量分解为目标函数分量和惩罚项分量；评估三种最小化策略：标准QAOA、T-QAOA和HT-QAOA。

### 主要发现

使用单独的参数来处理问题哈密顿量的不同分量会导致更低的能量值和改进的解决方案质量。

### 结论

QTIS在调度具有固定时间窗口的任务同时最小化冲突方面是高效的，展示了其在复杂调度环境中推进混合量子-经典优化的潜力。

### 翻译

任务调度在受约束的时间区间和有限资源的情况下仍然是制造、物流、云计算和医疗保健等领域的基本挑战。本研究提出了一种量子近似优化算法(QAOA)的新变体，旨在解决表述为二次无约束二元优化(QUBO)模型的任务调度问题。所提出的方法称为量子时间区间调度器(QTIS)，它集成了一个辅助辅助量子电路，用于动态检测和惩罚重叠任务，从而增强调度约束的执行。探索了两种互补的重叠检测实现：一种基于RY旋转和CCNOT门的量子方法，以及一种依赖预处理区间比较的经典替代方案。QTIS将问题哈密顿量(Hp)分解为两个分量，每个分量由不同的角度参数化。第一个分量编码目标函数，而第二个分量捕获与重叠区间相关的惩罚项，这些惩罚项由辅助电路控制。随后，评估了三种最小化策略：标准QAOA、T-QAOA和HT-QAOA，表明使用单独的参数来处理问题哈密顿量的不同分量会导致更低的能量值和改进的解决方案质量。结果证实了QTIS在调度具有固定时间窗口的任务同时最小化冲突方面的效率，展示了其在复杂调度环境中推进混合量子-经典优化的潜力。


### 论文摘要

Task scheduling with constrained time intervals and limited resources remains a fundamental challenge across domains such as manufacturing, logistics, cloud computing, and healthcare. This study presents a novel variant of the Quantum Approximate Optimization Algorithm (QAOA) designed to address the task scheduling problem formulated as a Quadratic Unconstrained Binary Optimization (QUBO) model. The proposed method, referred to as Quantum Time Interval Scheduler (QTIS), integrates an ancilla-assisted quantum circuit to dynamically detect and penalize overlapping tasks, enhancing the enforcement of scheduling constraints. Two complementary implementations are explored for overlap detection: a quantum approach based on RY rotations and CCNOT gates, and a classical alternative relying on preprocessed interval comparisons. QTIS decomposes the problem Hamiltonian, Hp, into two components, each parameterized by a distinct angle. The first component encodes the objective function, while the second captures penalty terms associated with overlapping intervals, which are controlled by the auxiliary circuit. Subsequently, three minimization strategies are evaluated: standard QAOA, T-QAOA, and HT-QAOA, showing that employing separate parameters for the different components of the problem Hamiltonian leads to lower energy values and improved solution quality. Results confirm the efficiency of QTIS in scheduling tasks with fixed temporal windows while minimizing conflicts, demonstrating its potential to advance hybrid quantum-classical optimization in complex scheduling environments.

---

## 115. AVATAAR: Agentic Video Answering via Temporal Adaptive Alignment and Reasoning

**论文链接:** [http://arxiv.org/abs/2511.15578v1](http://arxiv.org/abs/2511.15578v1)

**作者:** Urjitkumar Patel, Fang-Chun Yeh, Chinmay Gondhalekar

**发布时间:** 2025-11-19

**备注:** Accepted in the 5th IEEE Big Data Workshop on Multimodal AI (MMAI 2025), Dec 8-11, Macau, China, 2025 (Preprint Copy)

### GPT解析

### 总结

AVATAAR是一个创新的模块化框架，通过结合全局和局部视频上下文以及特殊的思考模块，显著提高了长视频问答的性能，特别是在时间推理、技术查询、主题问题和叙事理解方面。

### 背景

随着视频内容的日益普及，有效理解和回答关于长视频的问题对许多应用至关重要。虽然大型视觉语言模型提高了性能，但它们在处理需要全面理解和详细分析的微妙查询时常常面临挑战。

### 目的

为了克服这些障碍，作者引入了AVATAAR框架，旨在提高长视频问答的能力，解决LVLMs在处理复杂视频查询时的局限性。

### 方法

AVATAAR是一个模块化且可解释的框架，结合了全局和局部视频上下文，以及一个预检索思考代理和一个重新思考模块。该框架创建了一个持久的全局摘要，并在重新思考模块和预检索思考代理之间建立了反馈循环，使系统能够根据部分答案改进检索策略，并模拟类似人类的迭代推理。

### 主要发现

在CinePile基准测试中，AVATAAR相比基线模型取得了显著改进，在时间推理方面相对提升+5.6%，技术查询方面+5%，基于主题的问题方面+8%，叙事理解方面+8.2%。实验确认每个模块都对整体性能有积极贡献，反馈循环对系统的适应性至关重要。

### 结论

AVATAAR在增强视频理解能力方面非常有效，最终为长视频问答提供了一种可扩展的解决方案，结合了准确性、可解释性和可扩展性。

### 翻译

随着视频内容的日益普及，有效理解和回答关于长视频的问题对众多应用已变得至关重要。尽管大型视觉语言模型提升了性能，但它们在处理需要全面理解和详细分析的微妙查询时常常面临挑战。为了克服这些障碍，我们引入了AVATAAR，这是一个模块化且可解释的框架，结合了全局和局部视频上下文，以及一个预检索思考代理和一个重新思考模块。AVATAAR创建了一个持久的全局摘要，并在重新思考模块和预检索思考代理之间建立了反馈循环，使系统能够根据部分答案改进其检索策略，并模拟类似人类的迭代推理。在CinePile基准测试中，AVATAAR相比基线模型取得了显著改进，在时间推理方面相对提升+5.6%，技术查询方面+5%，基于主题的问题方面+8%，叙事理解方面+8.2%。我们的实验确认每个模块都对整体性能有积极贡献，反馈循环对适应性至关重要。这些发现突显了AVATAAR在增强视频理解能力方面的有效性。最终，AVATAAR为长视频问答提供了一种可扩展的解决方案，融合了准确性、可解释性和可扩展性。


### 论文摘要

With the increasing prevalence of video content, effectively understanding and answering questions about long form videos has become essential for numerous applications. Although large vision language models (LVLMs) have enhanced performance, they often face challenges with nuanced queries that demand both a comprehensive understanding and detailed analysis. To overcome these obstacles, we introduce AVATAAR, a modular and interpretable framework that combines global and local video context, along with a Pre Retrieval Thinking Agent and a Rethink Module. AVATAAR creates a persistent global summary and establishes a feedback loop between the Rethink Module and the Pre Retrieval Thinking Agent, allowing the system to refine its retrieval strategies based on partial answers and replicate human-like iterative reasoning. On the CinePile benchmark, AVATAAR demonstrates significant improvements over a baseline, achieving relative gains of +5.6% in temporal reasoning, +5% in technical queries, +8% in theme-based questions, and +8.2% in narrative comprehension. Our experiments confirm that each module contributes positively to the overall performance, with the feedback loop being crucial for adaptability. These findings highlight AVATAAR's effectiveness in enhancing video understanding capabilities. Ultimately, AVATAAR presents a scalable solution for long-form Video Question Answering (QA), merging accuracy, interpretability, and extensibility.

---

## 116. Spatiotemporal Activity-Driven Networks

**论文链接:** [http://arxiv.org/abs/2511.15533v1](http://arxiv.org/abs/2511.15533v1)

**作者:** Zsófia Simon, Jari Saramäki

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究引入了一个空间活动驱动模型，该模型考虑了空间约束，能够捕捉时空网络的联合效应。该模型具有解析可处理性，并能够重现社交和接触网络的几个特征特性，如强连接和弱连接、聚类以及权重高于中位数的三角形。该框架适用于模拟社交距离等干预措施，并能够系统地探索时空网络上的动态过程。

### 背景

时变网络模型为理解时间变化连接如何塑造传播等动态过程提供了关键见解。活动驱动模型是广泛使用的、解析可处理的基准模型。然而，许多时空网络（如物理接近网络）也嵌入在空间中，空间约束已知会强烈影响网络上展开的动态。尽管如此，目前缺乏类似的简单可解的时空接触结构模型。

### 目的

引入一个空间活动驱动模型，其中短距离接触更频繁，该模型是解析可处理的，并能捕捉时空的联合效应。通过该模型研究空间对网络动态的影响，特别是对传播动态的影响，并探索社交距离等干预措施的效果。

### 方法

提出并分析了一个空间活动驱动模型，其中短距离接触更频繁。通过解析和数值方法验证该模型，并在模型网络上模拟传播动态。研究社交距离作为干预措施的效果，特别是空间定向减少接触总数的影响。

### 主要发现

1. 该模型重现了社交和接触网络的几个特征特性，包括强连接和弱连接、聚类以及权重高于中位数的三角形。2. 这些特性可以归因于空间作为一种记忆形式的作用。3. 模型网络上的传播动态模拟进一步说明了空间的作用，突出了局部化如何减慢传播。4. 与非空间网络不同，即使是空间定向的接触总数的小幅减少也可以非常有效。

### 结论

通过提供一个可处理的框架，该模型能够系统地探索时空网络上的动态过程。该模型特别适用于模拟社交距离等干预措施，并表明空间定向的干预措施可能比非定向措施更有效。

### 翻译

时变网络模型为理解时间变化连接如何塑造传播等动态过程提供了关键见解。其中，活动驱动模型是一种广泛使用的、解析可处理的基准模型。然而，许多时空网络，如物理接近网络，也嵌入在空间中，空间约束已知会强烈影响网络上展开的动态。尽管如此，目前缺乏类似的简单可解的时空接触结构模型。在此，我们引入了一个空间活动驱动模型，其中短距离接触更频繁。该模型是解析可处理的，并能捕捉时空的联合效应。我们通过解析和数值方法证明，该模型重现了社交和接触网络的几个特征特性，包括强连接和弱连接、聚类以及权重高于中位数的三角形。这些特性可以归因于空间作为一种记忆形式的作用。模型网络上的传播动态模拟进一步说明了空间的作用，突出了局部化如何减慢传播。此外，该框架适用于有原则地模拟社交距离作为旨在减少长距离连接的干预措施。我们发现，与非空间网络不同，即使是空间定向的接触总数的小幅减少也可以非常有效。更广泛地说，通过提供一个可处理的框架，该模型能够系统地探索时空网络上的动态过程。


### 论文摘要

Temporal-network models have provided key insights into how time-varying connectivity shapes dynamical processes such as spreading. Among them, the activity-driven model is a widely used, analytically tractable benchmark. Yet many temporal networks, such as those of physical proximity, are also embedded in space, and spatial constraints are known to affect dynamics unfolding on the networks strongly. Despite this, there is a lack of similar simple and solvable models for spatiotemporal contact structures. Here, we introduce a spatial activity-driven model in which short-range contacts are more frequent. This model is analytically tractable and captures the joint effects of space and time. We show analytically and numerically that the model reproduces several characteristic features of social and contact networks, including strong and weak ties, clustering, and triangles having weights above the median. These traits can be attributed to space acting as a form of memory. Simulations of spreading dynamics on top of the model networks further illustrate the role of space, highlighting how localisation slows down spreading. Furthermore, the framework is well-suited for modelling social distancing in a principled way as an intervention measure aimed at reducing long-range links. We find that, unlike for non-spatial networks, even a small spatially targeted reduction in the total number of contacts can be very effective. More broadly, by offering a tractable framework, the model enables systematic exploration of dynamical processes on spatiotemporal networks.

---

## 117. Advancing Identification method of Gamma-Ray Bursts with Data and Feature Enhancement

**论文链接:** [http://arxiv.org/abs/2511.15470v1](http://arxiv.org/abs/2511.15470v1)

**作者:** Peng Zhang, Bing Li, Ren-Zhou Gui, Shao-Lin Xiong, Yu Wang, Shi-Jie Zheng, Guang-Cheng Xiao, Xiao-Bo Li, Yue Huang, Chen-Wei Wang, Jia-Cong Liu, Yan-Qiu Zhang, Wang-Chen Xue, Chao Zheng, Yue Wang

**发布时间:** 2025-11-19

**备注:** Under review. Dataset and model related discussions are welcome!

### GPT解析

### 总结

研究提出了一种结合自适应频率特征增强模块和物理信息增强数据的一维卷积神经网络，用于伽马射线暴的识别，通过生成10万个合成样本提高训练数据多样性，实现97.46%的高分类准确率。

### 背景

伽马射线暴因其瞬变特性、复杂的时间轮廓和有限的观测数据而难以识别。

### 目的

开发一种有效的伽马射线暴识别方法，解决传统方法面临的挑战。

### 方法

使用一维卷积神经网络结合自适应频率特征增强模块和物理信息增强数据，生成10万个合成伽马射线暴样本以增加训练数据的多样性和数量，同时保持物理保真度。

### 主要发现

模型达到97.46%的分类准确率，优于传统方法；特征可视化显示模型专注于深层次的形态特征；降维和聚类显示具有相似形态或起源的伽马射线暴在特征空间中聚集，将学习特征与物理性质联系起来。

### 结论

该框架为识别千新星和超新星相关的伽马射线暴候选者提供了新诊断工具，有助于建立多信使早期预警系统；同时可推广到其他稀有瞬变现象，促进大量观测数据中的自动检测。

### 翻译

伽马射线暴因其瞬变特性、复杂的时间轮廓和有限的观测数据而难以识别。我们通过结合自适应频率特征增强模块和物理信息增强数据的一维卷积神经网络来解决这一问题。我们的框架生成10万个合成伽马射线暴样本，增加了训练数据的多样性和数量，同时保持物理保真度，特别是对低显著性事件。该模型达到97.46%的分类准确率，优于所有测试过的传统增强模块变体，突显了增强的领域特定特征捕获能力。特征可视化显示模型专注于深层次的形态特征，并确认了提取物理上有意义的爆发特征的能力。降维和聚类显示具有相似形态或起源的伽马射线暴在特征空间中聚集，将学习到的特征与物理性质联系起来。这可能为识别千新星和超新星相关的伽马射线暴候选者提供了一种新的诊断工具，建立了增强多信使早期预警系统的标准。该框架有助于当前时域调查，可推广到其他稀有瞬变现象，并促进大量观测数据中的自动检测。


### 论文摘要

Gamma-ray bursts (GRBs) are challenging to identify due to their transient nature, complex temporal profiles, and limited observational datasets. We address this with a one-dimensional convolutional neural network integrated with an Adaptive Frequency Feature Enhancement module and physics-informed data augmentation. Our framework generates 100,000 synthetic GRB samples, expanding training data diversity and volume while preserving physical fidelity-especially for low-significance events. The model achieves 97.46% classification accuracy, outperforming all tested variants with conventional enhancement modules, highlighting enhanced domain-specific feature capture. Feature visualization shows model focuses on deep-seated morphological features and confirms the capability of extracting physically meaningful burst characteristics. Dimensionality reduction and clustering reveal GRBs with similar morphologies or progenitor origins cluster in the feature space, linking learned features to physical properties. This perhaps offers a novel diagnostic tool for identifying kilonova- and supernova-associated GRB candidates, establishing criteria to enhance multi-messenger early-warning systems. The framework aids current time-domain surveys, generalizes to other rare transients, and advances automated detection in large-volume observational data.

---

## 118. Computation for Epidemic Prediction with Graph Neural Network by Model Combination

**论文链接:** [http://arxiv.org/abs/2511.15469v1](http://arxiv.org/abs/2511.15469v1)

**作者:** Xiangxin Kong, Hang Wang, Yutong Li, Yanghao Chen, Zudi Lu

**发布时间:** 2025-11-19

**备注:** 37pages, 24 figures

### GPT解析

### 总结

论文提出了一种名为EpiHybridGNN的新型混合图神经网络模型，该模型整合了EpiGNN和ColaGNN两种流行疫情预测模型的优势，在时空疫情传播预测方面表现出色。

### 背景

对COVID-19等疫情事件在时间和空间维度进行建模是一项重要但具有挑战性的任务。EpiGNN和ColaGNN是两种流行的基于图神经网络(GNN)的区域疫情预测模型，各自具有优势但也存在不足。

### 目的

设计并实现一种新的混合图神经网络模型EpiHybridGNN，整合EpiGNN和ColaGNN的优势，形成更全面和鲁棒的时空疫情传播预测方法。

### 方法

EpiHybridGNN结合了EpiGNN的传播风险编码模块、区域感知图学习器(RAGL)以及多尺度卷积和图卷积网络的优势，同时融入了ColaGNN的跨位置注意力机制、多尺度扩张卷积和图消息传递能力，以实现更准确的长期预测。

### 主要发现

通过多个真实数据实验验证，EpiHybridGNN在疫情预测方面显著优于EpiGNN和ColaGNN两种模型，能够提供更全面和鲁棒的时空疫情传播预测。

### 结论

EpiHybridGNN通过整合现有模型的优势，形成了一个更全面、更强大的疫情预测框架，为时空疫情传播建模提供了新思路。

### 翻译

对COVID-19等疫情事件在时间和空间维度进行建模是一项重要但具有挑战性的任务。在对两种流行的基于图神经网络(GNN)的区域疫情预测模型EpiGNN和ColaGNN进行深入评估的基础上，我们提出了一种新型混合图神经网络模型EpiHybridGNN，它整合了EpiGNN和ColaGNN的优势。在EpiGNN中，通过其传播风险编码模块和区域感知图学习器(RAGL)，结合了多尺度卷积和图卷积网络(GCN)，旨在有效捕获区域间的时空传播动力学，并支持外部资源整合以提高预测性能。而在ColaGNN中，利用跨位置注意力机制、多尺度扩张卷积和图消息传递，通过动态图结构和时空特征融合来解决长期预测挑战。两者各有优势但也存在共同缺点。因此，我们的EpiHybridGNN设计结合了EpiGNN的风险编码和RAGL优势，以及ColaGNN的长期预测能力和动态注意力机制。这有助于形成更全面和鲁棒的时空疫情传播预测。我们提供了所提出的EpiHybridGNN的计算架构、核心公式及其解释。多个真实数据实验验证了我们的EpiHybridGNN在疫情预测方面显著优于EpiGNN和ColaGNN，同时提供了全面的见解和参考。


### 论文摘要

Modelling epidemic events such as COVID-19 cases in both time and space dimensions is an important but challenging task. Building on in-depth review and assessment of two popular graph neural network (GNN)-based regional epidemic forecasting models of \textbf{EpiGNN} and \textbf{ColaGNN}, we propose a novel hybrid graph neural network model, \textbf{EpiHybridGNN}, which integrates the strengths of both EpiGNN and \textbf{ColaGNN}. In the EpiGNN, through its transmission risk encoding module and Region-Aware Graph Learner (RAGL), both multi-scale convolutions and Graph Convolutional Networks (GCNs) are combined, aiming to effectively capture spatio-temporal propagation dynamics between regions and support the integration of external resources to enhance forecasting performance. While, in the ColaGNN, a cross-location attention mechanism, multi-scale dilated convolutions, and graph message passing are utilized to address the challenges of long-term forecasting through dynamic graph structures and spatio-temporal feature fusion. Both enjoy respective advantages but also share mutual shortcomings. Our EpiHybridGNN is therefore designed to combine the advantages of both EpiGNN, in its risk encoding and RAGL, and ColaGNN, in its long-term forecasting capabilities and dynamic attention mechanisms. This helps to form a more comprehensive and robust prediction of spatio-temporal epidemic propagation. The computational architecture, core formulas and their interpretations of our proposed EpiHybridGNN are provided. Multiple numerical real data experiments validate that our EpiHybridGNN significantly outperforms both EpiGNN and ColaGNN in epidemic forecasting with comprehensive insights and references offered.

---

## 119. 3D printed waveguides for optogenetics applications: design optimization and optical characterization

**论文链接:** [http://arxiv.org/abs/2511.15420v1](http://arxiv.org/abs/2511.15420v1)

**作者:** Giorgio Scordo, Kostas Kanellopulos, Surangrat Thongkorn, Samuel Tavares da Silva Maraschin, Kambiz Ghaseminasab, Evgeniy Shkondin, Deepshika Arasu, Stephan Sylvest Keller, Arto Rainer Heiskanen, Marta Perez Pereira, Jenny Emnéus

**发布时间:** 2025-11-19

**备注:** 18 pages, 7 figures

### GPT解析

### 总结

本研究介绍了一种基于3D打印的光传递系统，用于脑类器官的光遗传学刺激，通过投影微立体光刻技术制造，并验证了其功能性和应用潜力。

### 背景

光遗传学已成为疾病建模的强大工具，通过光刺激精确控制细胞活动，为疾病机制和治疗方法提供见解。微LED、光纤和微/纳米探针等创新材料技术实现了对目标细胞光传递的精确时空控制，而3D打印技术进步进一步增强了光遗传学应用，可制造可植入、可定制、小型化的高空间分辨率光刺激系统。

### 目的

开发一种基于3D打印的光传递系统用于脑类器官的光遗传学刺激，探索投影微立体光刻技术的能力，评估3D打印树脂的光学特性对光传输效率的影响，优化设计并进行功能验证。

### 方法

研究使用投影微立体光刻技术制造3D打印光传递系统，表征基于丙烯酸的高分辨率3D打印树脂的光学特性（折射率和消光系数），使用有限元方法模拟优化设计，开发光遗传学设置，并对光遗传学修饰的细胞进行初步测试，评估光诱导多巴胺释放的效率。

### 主要发现

3D打印波导成功实现了光传递功能，初步测试显示光诱导的多巴胺释放，刺激效率为2.8%，证实了该系统的可行性。

### 结论

这种光刺激工具在推进可定制的光遗传学应用方面具有强大潜力，研究结果为未来优化提供了指导方向。

### 翻译

光遗传学已成为疾病建模的强大工具，能够通过光刺激精确控制细胞活动，并为疾病机制和治疗方法提供宝贵见解。微LED、光纤和微纳米探针等创新材料和技术已被开发，以实现对目标细胞光传递的精确时空控制。3D打印技术的最新进展进一步增强了光遗传学应用，使得能够制造可植入、可定制和微型化的高空间分辨率光刺激系统。在本研究中，我们引入了一种用于脑类器官刺激的3D打印光传递系统的新概念，探索了投影微立体光刻的能力。我们表征了高分辨率丙烯酸基3D打印树脂的光学特性，即折射率和消光系数，以评估光传输效率是否会限制光遗传学刺激系统的性能。采用有限元方法模拟优化了3D打印设计。开发了一种用于最佳光传递的光遗传学设置，对光遗传学修饰的细胞进行的初步测试显示光诱导多巴胺释放，刺激效率为2.8%，确认了3D打印波导功能并指导未来优化。我们的结果表明，这种光刺激工具在推进可定制的光遗传学应用方面具有巨大潜力。


### 论文摘要

Optogenetics has emerged as a powerful tool for disease modeling, enabling precise control of cellular activities through light stimulation and providing a valuable insights into disease mechanisms and therapeutic possibilities. Innovative materials and technologies such as micro-LEDs, optical fibers and micro/nano probes have been developed to allow precise spatial and temporal control of light delivery to target cells. Recent advances in 3D printing have further enhanced optogenetic applications by enabling the fabrication of implantable, customizable, and miniaturized light stimulation systems with high spatial resolution. In this study, we introduce a novel concept of a 3D printed light delivery system for brain organoid stimulation exploring the capabilities of projection microstereolithography (P$μ$SL). We characterized the optical properties of the high-resolution acrylate-based 3D print resin, i.e., refractive index and extinction coefficient, to evaluate if the light transmission efficiency might limit the performance of the optogenetic stimulation systems. Finite element method simulations were employed to optimize the 3D printed design. An optogenetic setup was developed for optimal light delivery, and initial tests with optogenetically modified cells showed light-induced dopamine release with a stimulation efficiency of 2.8\%, confirming the 3D printed waveguide functionality and guiding future optimization. Our results demonstrate that this light stimulation tool offers strong potential for advancing customizable optogenetic applications.

---

## 120. Zero-Shot Open-Vocabulary Human Motion Grounding with Test-Time Training

**论文链接:** [http://arxiv.org/abs/2511.15379v1](http://arxiv.org/abs/2511.15379v1)

**作者:** Yunjiao Zhou, Xinyan Chen, Junlang Qian, Lihua Xie, Jianfei Yang

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了ZOMG，一个零样本、开放词汇的框架，能够将运动序列分割为语义上有意义的子动作，无需任何注释或微调。

### 背景

理解复杂人类活动需要将运动分解为细粒度、语义对齐的子动作，这对行为分析、具身AI和虚拟现实至关重要。然而，现有方法大多依赖密集监督和预定义动作类别，这在开放词汇、真实世界场景中不可行。

### 目的

开发一个零样本、开放词汇的框架，能够将运动序列分割为语义上有意义的子动作，且无需任何注释或微调。

### 方法

ZOMG框架包含两个技术组件：(1)语言语义分割，利用大型语言模型将指令分解为有序的子动作单元；(2)软掩码优化，学习特定实例的时间掩码，专注于对子动作至关重要的帧，同时保持段内连续性和强制段间分离，且不改变预训练编码器。

### 主要发现

在三个运动语言数据集上的实验证明了ZOMG在运动接地性能上的最先进有效性和效率，在HumanML3D基准测试上比先前方法提高了8.7% mAP，同时在下游检索任务中也存在显著改进。

### 结论

ZOMG建立了一种无需注释的运动理解新范式，为开放词汇场景下的运动分析提供了有效解决方案。

### 翻译

理解复杂的人类活动需要能够将运动分解为细粒度、语义对齐的子动作。这种运动接地过程对行为分析、具身AI和虚拟现实至关重要。然而，大多数现有方法依赖于密集监督和预定义的动作类别，这在开放词汇、真实世界场景中不可行。在本文中，我们提出了ZOMG，一个零样本、开放词汇的框架，能够将运动序列分割为语义上有意义的子动作，无需任何注释或微调。技术上，ZOMG集成了(1)语言语义分割，利用大型语言模型将指令分解为有序的子动作单元，以及(2)软掩码优化，学习特定实例的时间掩码，专注于对子动作至关重要的帧，同时保持段内连续性和强制段间分离，且不改变预训练编码器。在三个运动语言数据集上的实验证明了ZOMG在运动接地性能上的最先进有效性和效率，在HumanML3D基准测试上比先前方法提高了8.7% mAP。同时，在下游检索中也存在显著改进，为无需注释的运动理解建立了新范式。


### 论文摘要

Understanding complex human activities demands the ability to decompose motion into fine-grained, semantic-aligned sub-actions. This motion grounding process is crucial for behavior analysis, embodied AI and virtual reality. Yet, most existing methods rely on dense supervision with predefined action classes, which are infeasible in open-vocabulary, real-world settings. In this paper, we propose ZOMG, a zero-shot, open-vocabulary framework that segments motion sequences into semantically meaningful sub-actions without requiring any annotations or fine-tuning. Technically, ZOMG integrates (1) language semantic partition, which leverages large language models to decompose instructions into ordered sub-action units, and (2) soft masking optimization, which learns instance-specific temporal masks to focus on frames critical to sub-actions, while maintaining intra-segment continuity and enforcing inter-segment separation, all without altering the pretrained encoder. Experiments on three motion-language datasets demonstrate state-of-the-art effectiveness and efficiency of motion grounding performance, outperforming prior methods by +8.7\% mAP on HumanML3D benchmark. Meanwhile, significant improvements also exist in downstream retrieval, establishing a new paradigm for annotation-free motion understanding.

---

## 121. Detection of spiking motifs of arbitrary length in neural activity using bounded synaptic delays

**论文链接:** [http://arxiv.org/abs/2511.15296v1](http://arxiv.org/abs/2511.15296v1)

**作者:** Thomas Kronland-Martinet, Stéphane Viollet, Laurent U Perrinet

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种在脉冲神经网络中处理有界突触延迟下识别任意长度脉冲基序的方法，通过使用一系列输出神经元检测子基序来实现基序识别，并在音频数据和随机基序上验证了其有效性。

### 背景

在脉冲神经网络中，时间编码比速率编码更具优势，但突触延迟有限且可能短于基序持续时间，限制了传统基于异构延迟的基序识别方法的应用。

### 目的

开发一种在有界突触延迟条件下能够识别任意长度脉冲基序的方法，解决传统方法因延迟限制而无法应用的问题。

### 方法

使用一系列通过有界突触延迟连接到输入神经元的输出神经元，每个输出神经元检测一个有界持续时间的子基序，当所有子基序被顺序检测到时即认为基序被识别。使用漏积分放电神经元模拟网络，并在SHD数据库和随机同时基序上进行测试。

### 主要发现

网络可有效识别从SHD数据库提取的任意长度基序，十个同时基序的正确检测率约60%，五个基序时可达80%，显示了对噪声的鲁棒性；在大量输入神经元和稀疏基序条件下，单个基序与其他基序重叠的识别效果最佳。

### 结论

该方法为存储和检索任意时间长度的神经信息提供了更通用模型的基础，解决了突触延迟限制下的基序识别问题。

### 翻译

在脉冲神经网络背景下，由于处理速度和能量效率的优势，时间编码信号正逐渐优于速率编码假设。时间编码中，突触延迟对于处理具有精确脉冲时间信号的脉冲基序至关重要。然而，大脑中的突触延迟是有限的，可能短于基序的持续时间。这阻碍了使用由设置异构延迟以在作为符合检测器的单个输出神经元上同步输入脉冲的基序识别方法。为解决这一问题，我们开发了一种使用通过有界突触延迟连接到输入神经元的一系列输出神经元来检测任意长度基序的方法。每个输出神经元与一个有界持续时间的子基序相关联。如果所有子基序被输出神经元顺序检测到，则基序被识别。我们使用漏积分放电神经元模拟了该网络，并在通过耳蜗模型将音频数据转换为脉冲的海德堡脉冲数字(SHD)数据库以及随机同时基序上对其进行了测试。结果表明，网络可以有效识别从SHD数据库中提取的任意长度基序。在存在来自SHD数据集的十个同时基序的情况下，我们的方法具有约60%的正确检测率，五个基序时可达80%，显示了网络对噪声的鲁棒性。对随机重叠模式的结果表明，对于大量输入神经元和稀疏基序，识别与其他基序重叠的单个基序最为有效。我们的方法为存储和检索任意时间长度的神经信息提供了更通用模型的基础。


### 论文摘要

In the context of spiking neural networks, temporal coding of signals is increasingly preferred over the rate coding hypothesis due to its advantages in processing speed and energy efficiency. In temporal coding, synaptic delays are crucial for processing signals with precise spike timings, known as spiking motifs. Synaptic delays are however bounded in the brain and can thus be shorter than the duration of a motif. This prevents the use of motif recognition methods that consist of setting heterogeneous delays to synchronize the input spikes on a single output neuron acting as a coincidence detector. To address this issue, we developed a method to detect motifs of arbitrary length using a sequence of output neurons connected to input neurons by bounded synaptic delays. Each output neuron is associated with a sub-motif of bounded duration. A motif is recognized if all sub-motifs are sequentially detected by the output neurons. We simulated this network using leaky integrate-and-fire neurons and tested it on the Spiking Heidelberg Digits (SHD) database, that is, on audio data converted to spikes via a cochlear model, as well as on random simultaneous motifs. The results demonstrate that the network can effectively recognize motifs of arbitrary length extracted from the SHD database. Our method features a correct detection rate of about 60% in presence of ten simultaneous motifs from the SHD dataset and up to 80% for five motifs, showing the robustness of the network to noise. Results on random overlapping patterns show that the recognition of a single motif overlapping with other motifs is most effective for a large number of input neurons and sparser motifs. Our method provides a foundation for more general models for the storage and retrieval of neural information of arbitrary temporal lengths.

---

## 122. 论文ID: 2511.15188v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15188v1.json'

---

## 123. 论文ID: 2511.15173v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15173v1.json'

---

## 124. Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs

**论文链接:** [http://arxiv.org/abs/2511.15163v1](http://arxiv.org/abs/2511.15163v1)

**作者:** Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, Xiaozhong Liu

**发布时间:** 2025-11-19

**备注:** AAAI 2026 Workshop

### GPT解析

### 总结

研究提出了TASA框架，整合学生个性、记忆和遗忘动态，用于个性化数学学习，通过大型语言模型提供适应性教学，取得优于基线的学习成果。

### 背景

大型语言模型越来越多地被整合到智能辅导系统中，但现有方法未能捕捉学生知识在熟练度、概念差距和遗忘模式方面的动态演变，这在数学辅导中尤为突出。

### 目的

解决现有智能辅导系统无法捕捉学生知识动态演变的问题，开发一种能根据学生能力水平提供精确调整指导的框架。

### 方法

提出TASA框架，维护结构化学生画像捕捉能力档案，记录学习交互事件记忆，结合连续遗忘曲线和知识追踪动态更新学生掌握状态，生成上下文合适、难度调整的问题和解释。

### 主要发现

实证结果表明，TASA与代表性基线相比实现了更好的学习成果和更自适应的辅导行为，证明了建模时间遗忘和学习者画像的重要性。

### 结论

在基于大型语言模型的辅导系统中，建模时间遗忘和学习者画像对于提供有效的个性化教学至关重要，TASA框架成功解决了现有方法的局限性。

### 翻译

大型语言模型（LLMs）越来越多地被整合到智能辅导系统中，以提供类人化和自适应的指导。然而，大多数现有方法未能捕捉学生知识在其熟练度、概念差距和遗忘模式方面的动态演变。这一挑战在数学辅导中尤为突出，因为有效的指导需要根据每个学生的掌握水平和认知记忆进行精细调整的脚手架支持。为了解决这个问题，我们提出了TASA（Teaching According to Students' Aptitude），一个学生感知的辅导框架，整合了个性、记忆和遗忘动态，用于个性化数学学习。具体来说，TASA维护一个捕捉能力档案的结构化学生画像，并记录先前学习交互的事件记忆。通过结合连续遗忘曲线和知识追踪，TASA动态更新每个学生的掌握状态，并生成上下文合适、难度调整的问题和解释。实证结果表明，与代表性基线相比，TASA实现了更好的学习成果和更自适应的辅导行为，强调了在基于大型语言模型的辅导系统中建模时间遗忘和学习者画像的重要性。


### 论文摘要

Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems.

---

## 125. Fourier-KAN-Mamba: A Novel State-Space Equation Approach for Time-Series Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.15083v1](http://arxiv.org/abs/2511.15083v1)

**作者:** Xiancheng Wang, Lin Wang, Rui Wang, Zhibo Zhang, Minghang Zhao

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种名为Fourier-KAN-Mamba的新型混合架构，用于时间序列异常检测，整合了Fourier层、Kolmogorov-Arnold Networks和Mamba选择性状态空间模型，在多个数据集上显著优于现有方法。

### 背景

时间序列异常检测在工业监测和故障诊断等众多实际应用中扮演着关键角色。Mamba基于的状态空间模型在长序列建模方面表现出色，但直接应用于异常检测任务时存在挑战。

### 目的

开发一种能够有效捕捉复杂时间模式和非线性动力学的时间序列异常检测方法，以解决直接应用Mamba模型时的局限性。

### 方法

提出Fourier-KAN-Mamba混合架构，包含：1) Fourier层提取多尺度频率特征；2) KAN增强非线性表示能力；3) Mamba选择性状态空间模型；4) 时间门控控制机制提高区分正常和异常模式的能力。

### 主要发现

在MSL、SMAP和SWaT数据集上的广泛实验表明，所提出的方法显著优于现有的最先进方法。

### 结论

Fourier-KAN-Mamba架构通过整合多种技术有效解决了时间序列异常检测中的挑战，特别是在捕捉复杂时间模式和非线性动力学方面表现优异。

### 翻译

时间序列异常检测在众多实际应用中扮演关键角色，包括工业监测和故障诊断。最近，基于Mamba的状态空间模型在长序列建模中表现出显著效率。然而，将Mamba直接应用于异常检测任务在捕捉复杂时间模式和非线性动力学方面仍面临挑战。本文提出了Fourier-KAN-Mamba，一种新颖的混合架构，整合了Fourier层、Kolmogorov-Arnold网络和Mamba选择性状态空间模型。Fourier层提取多尺度频率特征，KAN增强非线性表示能力，时间门控控制机制进一步提高了模型区分正常和异常模式的能力。在MSL、SMAP和SWaT数据集上的大量实验证明，我们的方法显著优于现有的最先进方法。


### 论文摘要

Time-series anomaly detection plays a critical role in numerous real-world applications, including industrial monitoring and fault diagnosis. Recently, Mamba-based state-space models have shown remarkable efficiency in long-sequence modeling. However, directly applying Mamba to anomaly detection tasks still faces challenges in capturing complex temporal patterns and nonlinear dynamics. In this paper, we propose Fourier-KAN-Mamba, a novel hybrid architecture that integrates Fourier layer, Kolmogorov-Arnold Networks (KAN), and Mamba selective state-space model. The Fourier layer extracts multi-scale frequency features, KAN enhances nonlinear representation capability, and a temporal gating control mechanism further improves the model's ability to distinguish normal and anomalous patterns. Extensive experiments on MSL, SMAP, and SWaT datasets demonstrate that our method significantly outperforms existing state-of-the-art approaches.   Keywords: time-series anomaly detection, state-space model, Mamba, Fourier transform, Kolmogorov-Arnold Network

---

## 126. Cement2: Temporal Hardware Transactions for High-Level and Efficient FPGA Programming

**论文链接:** [http://arxiv.org/abs/2511.15073v1](http://arxiv.org/abs/2511.15073v1)

**作者:** Youwei Xiao, Zizhang Luo, Weijie Peng, Yuyang Zou, Yun Liang

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究提出了一种名为'时序硬件事务'的新抽象方法，通过在事务性语言级别引入周期级时序意识，解决了硬件设计中提高抽象层次与保持底层细节控制之间的矛盾。该方法在Cement2事务性HDL中实现，验证了其在FPGA设计中的高效性和适用性。

### 背景

硬件设计面临一个基本挑战：如何在提高抽象层次以提升生产力的同时，保持对底层细节（如周期精度）的控制。传统RTL设计通过接线式连接组合模块，但对行为正确性保证较弱。虽然HLS和新兴抽象试图解决此问题，但要么引入不可预测开销，要么限制设计通用性。事务性HDL虽提供了有前景的基础，但仅建模周期内行为，不反映本机时序设计特性，阻碍了其在FPGA编程场景中的适用性。

### 目的

提出一种新的抽象方法，将周期级时序意识引入到事务性语言级别，使设计者能够描述跨越多个时钟周期的规则动作，为多周期架构行为提供直观抽象，从而提高硬件设计的生产力和适用性。

### 方法

提出'时序硬件事务'这一新抽象方法，建模规则之间的时序关系，并支持描述跨越多个时钟周期的规则动作。在Cement2（一种嵌入在Rust中的事务性HDL）中实现此方法，使硬件构造函数能够构建周期内和时序事务。Cement2的综合框架通过多个分析和优化阶段降低描述抽象，生成高效硬件。

### 主要发现

使用Cement2的抽象，成功编程了RISC-V软核处理器、自定义CPU指令、线性代数内核和脉动阵列加速器，利用高级抽象提高了生产力。评估表明，与手工编写的RTL设计相比，Cement2不会牺牲性能和资源。

### 结论

'时序硬件事务'抽象方法在保持硬件性能的同时提高了设计抽象层次，证明了其在通用FPGA设计任务中的高适用性，解决了硬件设计中提高抽象与保持控制之间的矛盾。

### 翻译

硬件设计面临一个根本性挑战：提高抽象层次以改善生产力，同时保持对周期精度等底层细节的控制。像SystemVerilog这样的传统RTL设计通过接线式连接组合模块，对行为正确性提供弱保证。虽然高层次综合（HLS）和新兴的抽象试图解决这个问题，但它们要么引入不可预测的开销，要么限制设计的通用性。虽然事务性HDL通过将设计抽象提升到原子和可组合的规则提供了有前景的基础，但它们仅建模周期内行为，不反映本机时序设计特性，阻碍了其在FPGA编程场景中的适用性和生产力。我们提出时序硬件事务，一种新的抽象，将周期级时序意识引入到事务性语言级别。我们的方法建模规则之间的时序关系，并支持描述跨越多个时钟周期的规则动作，为描述多周期架构行为提供直观的抽象。我们在Cement2中实现了这一点，Cement2是一种嵌入在Rust中的事务性HDL，使硬件构造函数能够构建周期内和时序事务。Cement2的综合框架通过多个分析和优化阶段降低描述抽象，生成高效的硬件。使用Cement2的抽象，我们编程了一个RISC-V软核处理器、自定义CPU指令、线性代数内核和脉动阵列加速器，利用高级抽象提高了生产力。评估显示，与手工编写的RTL设计相比，Cement2不会牺牲性能和资源，证明了其在通用FPGA设计任务中的高适用性。


### 论文摘要

Hardware design faces a fundamental challenge: raising abstraction to improve productivity while maintaining control over low-level details like cycle accuracy. Traditional RTL design in languages like SystemVerilog composes modules through wiring-style connections that provide weak guarantees for behavioral correctness. While high-level synthesis (HLS) and emerging abstractions attempt to address this, they either introduce unpredictable overhead or restrict design generality. Although transactional HDLs provide a promising foundation by lifting design abstraction to atomic and composable rules, they solely model intra-cycle behavior and do not reflect the native temporal design characteristics, hindering applicability and productivity for FPGA programming scenarios.   We propose temporal hardware transactions, a new abstraction that brings cycle-level timing awareness to designers at the transactional language level. Our approach models temporal relationships between rules and supports the description of rules whose actions span multiple clock cycles, providing intuitive abstraction to describe multi-cycle architectural behavior. We implement this in Cement2, a transactional HDL embedded in Rust, enabling programming hardware constructors to build both intra-cycle and temporal transactions. Cement2's synthesis framework lowers description abstraction through multiple analysis and optimization phases, generating efficient hardware. With Cement2's abstraction, we program a RISC-V soft-core processor, custom CPU instructions, linear algebra kernels, and systolic array accelerators, leveraging the high-level abstraction for boosted productivity. Evaluation shows that Cement2 does not sacrifice performance and resources compared to hand-coded RTL designs, demonstrating the high applicability for general FPGA design tasks.

---

## 127. Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks

**论文链接:** [http://arxiv.org/abs/2511.15065v1](http://arxiv.org/abs/2511.15065v1)

**作者:** Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, Chenglin Wu

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究探索了视频模型通过视频生成进行推理的能力，并提出了VR-Bench基准来评估视频模型的推理能力。研究发现视频模型在空间推理任务中表现出色，优于视觉语言模型，且测试时多样化采样可提高推理可靠性。

### 背景

视频模型在高保真视频生成方面已取得显著成功，类似于语言模型从文本生成发展到基于文本的推理，视频模型也可能发展出推理能力。视频将推理基于明确的空间布局和时间连续性，使其成为空间推理的理想基础。

### 目的

探索视频模型通过视频生成进行推理的范式，并开发一个全面评估视频模型推理能力的基准。

### 方法

引入VR-Bench基准，这是一个基于迷宫解决任务的评估框架，包含五种迷宫类型和多样化视觉风格的7,920个程序生成的视频。使用SFT（监督微调）来激发视频模型的推理能力，并在推理过程中采用多样化采样策略。

### 主要发现

1) SFT可以有效激发视频模型的推理能力；2) 视频模型在推理过程中表现出更强的空间感知能力，优于领先的视觉语言模型；3) 视频模型能很好地在不同场景、任务和复杂度级别上泛化；4) 推理过程中的多样化采样可以将推理可靠性提高10-20%；5) 通过视频进行推理在空间推理任务中具有独特潜力和可扩展性。

### 结论

通过视频进行推理是一种有前景的范式，特别适用于空间推理任务。视频模型不仅能够生成高质量的视频，还能通过视频生成进行有效的推理，这一发现为视频模型的应用开辟了新的方向。

### 翻译

视频模型在高保真视频生成方面取得了显著成功，具有连贯的运动动态。类似于语言模型从文本发展到文本推理的历程，视频模型的发展促使我们思考：视频模型能否通过视频生成进行推理？与离散文本语料库相比，视频将推理基于明确的空间布局和时间连续性，这使其成为空间推理的理想基础。在本工作中，我们探索了通过视频进行推理的范式，并引入了VR-Bench——一个全面评估视频模型推理能力的基准。基于 inherently 需要空间规划和多步推理的迷宫解决任务，VR-Bench包含五种迷宫类型和多样化视觉风格的7,920个程序生成的视频。我们的经验分析表明，SFT可以有效激发视频模型的推理能力。视频模型在推理过程中表现出更强的空间感知能力，优于领先的视觉语言模型，并能很好地在不同场景、任务和复杂度级别上泛化。我们还发现了一种测试时扩展效应，即在推理过程中进行多样化采样可以将推理可靠性提高10-20%。这些发现突显了通过视频进行推理在空间推理任务中的独特潜力和可扩展性。


### 论文摘要

Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks.

---

## 128. Aligning Generative Music AI with Human Preferences: Methods and Challenges

**论文链接:** [http://arxiv.org/abs/2511.15038v1](http://arxiv.org/abs/2511.15038v1)

**作者:** Dorien Herremans, Abhinaba Roy

**发布时间:** 2025-11-19

**备注:** Accepted at the AAAI-2026 Senior Member Track

### GPT解析

### 总结

生成式AI在音乐领域虽取得显著进展，但因使用的损失函数无法与人类偏好对齐，导致生成的音乐无法满足人类微妙需求。本文倡导应用偏好对齐技术，弥合计算优化与人类音乐欣赏间的差距。

### 背景

生成式AI在音乐生成方面已达到令人印象保真的程度和风格多样性，但这些系统由于使用的特定损失函数，往往无法与人类微妙的偏好保持一致。

### 目的

倡导系统性地应用偏好对齐技术到音乐生成中，解决计算优化与人类音乐欣赏之间的根本差距。

### 方法

借鉴MusicRL的大规模偏好学习、DiffRhythm+等多偏好对齐框架，以及Text2midi-InferAlign等推理时优化技术。

### 主要发现

这些技术可以解决音乐领域的独特挑战，包括时间连贯性、和谐一致性和主观质量评估。

### 结论

偏好对齐的音乐生成将推动交互式作曲工具和个性化音乐服务等变革性应用，需要机器学习和音乐理论的持续跨学科研究，创造真正满足人类创造和体验需求的音乐AI系统。

### 翻译

生成式AI在音乐领域的最新进展已经取得了令人印象保真的程度和风格多样性，然而这些系统由于使用的特定损失函数，往往无法与人类微妙的偏好保持一致。本文倡导系统性地应用偏好对齐技术到音乐生成中，解决计算优化与人类音乐欣赏之间的根本差距。借鉴MusicRL的大规模偏好学习、DiffRhythm+等多偏好对齐框架，以及Text2midi-InferAlign等推理时优化技术，我们讨论了这些技术如何解决音乐领域的独特挑战：时间连贯性、和谐一致性和主观质量评估。我们确定了关键研究挑战，包括扩展到长篇作品的规模、偏好建模的可靠性等。展望未来，我们期望偏好对齐的音乐生成能够推动交互式作曲工具和个性化音乐服务等变革性应用。这项工作呼吁持续进行跨学科研究，结合机器学习和音乐理论的进步，创造真正满足人类创造和体验需求的音乐AI系统。


### 论文摘要

Recent advances in generative AI for music have achieved remarkable fidelity and stylistic diversity, yet these systems often fail to align with nuanced human preferences due to the specific loss functions they use. This paper advocates for the systematic application of preference alignment techniques to music generation, addressing the fundamental gap between computational optimization and human musical appreciation. Drawing on recent breakthroughs including MusicRL's large-scale preference learning, multi-preference alignment frameworks like diffusion-based preference optimization in DiffRhythm+, and inference-time optimization techniques like Text2midi-InferAlign, we discuss how these techniques can address music's unique challenges: temporal coherence, harmonic consistency, and subjective quality assessment. We identify key research challenges including scalability to long-form compositions, reliability amongst others in preference modelling. Looking forward, we envision preference-aligned music generation enabling transformative applications in interactive composition tools and personalized music services. This work calls for sustained interdisciplinary research combining advances in machine learning, music-theory to create music AI systems that truly serve human creative and experiential needs.

---

## 129. 论文ID: 2511.15003v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15003v1.json'

---

## 130. Task Specific Sharpness Aware O-RAN Resource Management using Multi Agent Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.15002v1](http://arxiv.org/abs/2511.15002v1)

**作者:** Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah

**发布时间:** 2025-11-19

**备注:** Accepted to be published in IEEE Transaction on Machine Learning in Communication and Networking (TMLCN)

### GPT解析

### 总结

本文提出了一种结合锐度感知最小化(SAM)和软演员评论家(SAC)算法的资源管理方法，在分布式多智能体强化学习框架中提高了下一代网络的资源分配效率和QoS满足度。

### 背景

下一代网络采用开放无线接入网络(O-RAN)架构通过无线接入网络智能控制器(RIC)实现动态资源管理，但深度强化学习(DRL)模型在动态环境中往往面临鲁棒性和泛化能力不足的问题。

### 目的

引入一种新的资源管理方法，在分布式多智能体强化学习(MARL)框架中通过将锐度感知最小化(SAM)算法增强软演员评论家(SAC)算法，提高资源管理效果。

### 方法

提出自适应和选择性的SAM机制，由时序差分(TD)误差方差驱动正则化，确保只有面临高环境复杂性的智能体被正则化；引入动态ρ调度方案优化智能体间的探索-利用权衡。

### 主要发现

实验结果表明，该方法显著优于传统DRL方法，在资源分配效率方面提高了高达22%，并确保了跨不同O-RAN切片的优越服务质量(QoS)满足度。

### 结论

所提出的方法通过结合SAM和SAC算法在分布式MARL框架中实现了更高效、更稳定的资源管理，显著提高了资源分配效率和QoS满足度。

### 翻译

下一代网络利用开放无线接入网络(O-RAN)架构通过无线接入网络智能控制器(RIC)实现动态资源管理。虽然深度强化学习(DRL)模型在优化网络资源方面显示出潜力，但它们在动态环境中往往面临鲁棒性和泛化能力不足的问题。本文介绍了一种新颖的资源管理方法，在分布式多智能体强化学习(MARL)框架中通过锐度感知最小化(SAM)算法增强了软演员评论家(SAC)算法。我们的方法引入了一种自适应和选择性的SAM机制，其中正则化明确由时序差分(TD)误差方差驱动，确保只有面临高环境复杂性的智能体被正则化。这种有针对性的策略减少了不必要的开销，提高了训练稳定性，增强了泛化能力，同时不牺牲学习效率。我们进一步融入了动态ρ调度方案，以优化智能体之间的探索-利用权衡。实验结果表明，我们的方法显著优于传统DRL方法，在资源分配效率方面提高了高达22%，并确保了跨不同O-RAN切片的优越服务质量(QoS)满足度。


### 论文摘要

Next-generation networks utilize the Open Radio Access Network (O-RAN) architecture to enable dynamic resource management, facilitated by the RAN Intelligent Controller (RIC). While deep reinforcement learning (DRL) models show promise in optimizing network resources, they often struggle with robustness and generalizability in dynamic environments. This paper introduces a novel resource management approach that enhances the Soft Actor Critic (SAC) algorithm with Sharpness-Aware Minimization (SAM) in a distributed Multi-Agent RL (MARL) framework. Our method introduces an adaptive and selective SAM mechanism, where regularization is explicitly driven by temporal-difference (TD)-error variance, ensuring that only agents facing high environmental complexity are regularized. This targeted strategy reduces unnecessary overhead, improves training stability, and enhances generalization without sacrificing learning efficiency. We further incorporate a dynamic $ρ$ scheduling scheme to refine the exploration-exploitation trade-off across agents. Experimental results show our method significantly outperforms conventional DRL approaches, yielding up to a $22\%$ improvement in resource allocation efficiency and ensuring superior QoS satisfaction across diverse O-RAN slices.

---

## 131. FinCriticalED: A Visual Benchmark for Financial Fact-Level OCR Evaluation

**论文链接:** [http://arxiv.org/abs/2511.14998v1](http://arxiv.org/abs/2511.14998v1)

**作者:** Yueru He, Xueqing Peng, Yupeng Cao, Yan Wang, Lingfei Qian, Haohang Li, Yi Han, Ruoyu Xiang, Mingquan Lin, Prayag Tiwari, Jimin Huang, Guojun Xiong, Sophia Ananiadou

**发布时间:** 2025-11-19

**备注:** Yueru He, Xueqing Peng: These two authors contributed equally to this work

### GPT解析

### 总结

这篇论文介绍了FinCriticalED，一个用于评估OCR和视觉语言模型在金融文档上事实级表现的视觉基准测试。该基准包含500张图像和HTML对，由金融专家标注了七百多个数字和时间事实。研究建立了第一个金融文档理解的事实级评估基准，引入了严格的质量控制和LLM作为评估管道，并在多种模型上进行了基准测试，发现即使是最好的模型在复杂视觉上下文中仍存在实质性错误。

### 背景

金融文档包含视觉密集和表格密集的布局，数字和时间信息与结构紧密耦合。在高风险环境中，小的OCR错误（如符号反转或日期偏移）可能导致完全不同的解释，而传统OCR指标仅捕获表面文本相似性。

### 目的

创建一个事实级评估基准，将评估从词汇重叠转移到领域关键的事实正确性，提高金融文档理解的准确性。

### 方法

创建500张图像和HTML对的基准测试集；建立金融文档理解的事实级评估基准；使用金融专家创建和验证所有注释，进行严格质量控制；开发LLM作为评估管道，执行结构化事实提取和上下文验证；在OCR系统、开源视觉语言模型和专有模型上进行基准测试。

### 主要发现

最强的专有模型实现了最高的事实准确性，但在视觉复杂的数字和时间上下文中仍存在实质性错误。

### 结论

FinCriticalED通过定量评估和专家案例研究，为金融和其他精度关键领域推进视觉事实精度提供了严格的基础。

### 翻译

我们介绍了FinCriticalED（金融关键错误检测），这是一个用于评估OCR和视觉语言模型在金融文档上事实级表现的视觉基准。金融文档包含视觉密集和表格密集的布局，其中数字和时间信息与结构紧密耦合。在高风险环境中，小的OCR错误（如符号反转或日期偏移）可能导致完全不同的解释，而传统的OCR指标如ROUGE和编辑距离仅捕获表面文本相似性。FinCriticalED提供了500张图像和HTML对，包含金融专家标注的超过七百个数字和时间事实。它引入了三个关键贡献。首先，它建立了金融文档理解的事实级评估基准的第一个实例，将评估从词汇重叠转移到领域关键的事实正确性。其次，所有注释都由金融专家创建和验证，对符号、大小和时间表达式进行严格的质量控制。第三，我们开发了LLM作为评估管道，执行视觉复杂金融文档的结构化事实提取和上下文验证。我们在FinCriticalED上对OCR系统、开源视觉语言模型和专有模型进行了基准测试。结果表明，尽管最强的专有模型实现了最高的事实准确性，但在视觉复杂的数字和时间上下文中仍存在实质性错误。通过定量评估和专家案例研究，FinCriticalED为金融和其他精度关键领域推进视觉事实精度提供了严格的基础。


### 论文摘要

We introduce FinCriticalED (Financial Critical Error Detection), a visual benchmark for evaluating OCR and vision language models on financial documents at the fact level. Financial documents contain visually dense and table heavy layouts where numerical and temporal information is tightly coupled with structure. In high stakes settings, small OCR mistakes such as sign inversion or shifted dates can lead to materially different interpretations, while traditional OCR metrics like ROUGE and edit distance capture only surface level text similarity. \ficriticaled provides 500 image-HTML pairs with expert annotated financial facts covering over seven hundred numerical and temporal facts. It introduces three key contributions. First, it establishes the first fact level evaluation benchmark for financial document understanding, shifting evaluation from lexical overlap to domain critical factual correctness. Second, all annotations are created and verified by financial experts with strict quality control over signs, magnitudes, and temporal expressions. Third, we develop an LLM-as-Judge evaluation pipeline that performs structured fact extraction and contextual verification for visually complex financial documents. We benchmark OCR systems, open source vision language models, and proprietary models on FinCriticalED. Results show that although the strongest proprietary models achieve the highest factual accuracy, substantial errors remain in visually intricate numerical and temporal contexts. Through quantitative evaluation and expert case studies, FinCriticalED provides a rigorous foundation for advancing visual factual precision in financial and other precision critical domains.

---

## 132. Reconstruction of three-dimensional shapes of normal and disease-related erythrocytes from partial observations using multi-fidelity neural networks

**论文链接:** [http://arxiv.org/abs/2511.14962v1](http://arxiv.org/abs/2511.14962v1)

**作者:** Haizhou Wen, He Li, Zhen Li

**发布时间:** 2025-11-18

**备注:** 29 pages, 10 figures, 3 appendices

### GPT解析

### 总结

本研究提出了一种多保真度神经网络（MFNN）方法，用于从部分横截面观察重建3D红细胞形态，能够准确重建复杂红细胞形态，达到95%以上的坐标准确率，为红细胞形态的定量分析提供了新工具。

### 背景

从部分观察（如显微镜图像）重建3D红细胞形态对于理解红细胞衰老生理学和各种红细胞疾病的病理学至关重要。传统方法难以准确重建复杂红细胞形态。

### 目的

开发一种能够从部分横截面观察准确重建3D红细胞形态的方法，特别是针对正常和衰老人群中的各种红细胞形态。

### 方法

研究提出了一种多保真度神经网络（MFNN）方法，结合了卷积神经网络（处理低保真度参考红细胞数据）和前馈神经网络（捕获非线性形态相关性），并通过表面积和体积约束进行正则化。该方法基于球体和3D红细胞表面之间的拓扑同胚理论，并使用由耗散粒子动力学模拟生成的红细胞形态（口盘形-盘形-棘形转变）进行训练。

### 主要发现

MFNN预测器可以在提供至少两个正交横截面时，以超过95%的坐标准确率重建复杂的红细胞形态；包含棘形红细胞棘尖信息的斜横截面可以改善局部和全局特征重建；研究还评估了采样策略、形状不相似性和噪声的影响，显示在物理约束训练下具有更强的鲁棒性。

### 结论

MFNN能够从常规显微镜图像中观察到的部分横截面重建正常和衰老红细胞的3D形状，有助于对正常和疾病相关红细胞样本进行形态参数的定量分析。

### 翻译

从部分观察（如显微镜图像）重建3D红细胞或红血细胞（RBC）形态对于理解红细胞衰老的生理学和各种红细胞疾病的病理学至关重要。在本研究中，我们提出了一种多保真度神经网络（MFNN）方法，将高保真度的RBC横截面与形态相似的低保真度参考3D RBC形状融合，以恢复其完整的3D表面。MFNN预测器结合了在低保真度参考RBC数据上训练的卷积神经网络和捕获非线性形态相关性的前馈神经网络，并通过表面积和体积约束对低保真度分支进行正则化增强训练。该方法基于球体和3D RBC表面之间的拓扑同胚理论，并由口盘形-盘形-棘形转变的耗散粒子动力学模拟生成训练数据。在跨越正常和衰老人群观察的各种红细胞形态的基准测试中，我们的结果表明，当提供至少两个正交横截面时，MFNN预测器可以重建复杂的红细胞形态，超过95%的坐标准确度。观察到，包含棘形红细胞棘尖信息的斜横截面可以改善局部和全局特征重建，突出了特征感知采样的价值。我们的研究进一步评估了采样策略、形状不相似性和噪声的影响，显示在物理约束训练下具有更强的鲁棒性。总之，这些结果表明MFNN能够从常规显微镜图像中观察到的部分横截面重建正常和衰老红细胞的3D形状，这可能促进对正常和疾病相关红细胞样本中红细胞形态参数的定量分析。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从部分观测（如显微镜图像）重建红血细胞完整三维形状的问题。这个问题很重要，因为红血细胞形态是评估其功能和疾病状态的关键指标，准确的三维形态重建对于理解红血细胞老化生理学、各种红细胞疾病病理学以及揭示形态-力学特性相关性等生物物理机制至关重要，同时也能促进对正常和疾病相关红血细胞样本的形态参数定量分析。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统3D成像技术的局限性，然后借鉴了机器学习在细胞生物物理学中的应用经验，特别是多保真度神经网络框架。作者从拓扑学同胚理论出发，论证了球面和RBC表面之间存在连续可逆映射，为多保真度建模提供了理论基础。在训练过程中引入表面积和体积等物理约束，并使用耗散粒子动力学模拟生成训练数据。作者借鉴了现有机器学习方法、RBC形态分类研究、DPD模拟技术和传统成像技术，但将其创新性地应用于红血细胞三维形态重建这一特定问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用多保真度神经网络框架，结合高保真度的目标RBC部分观测数据和低保真度的参考RBC完整三维形状数据，重建目标RBC的完整三维表面。整体流程包括：1)使用DPD模拟生成SDE形态变换的训练数据；2)构建包含低保真CNN和高保真FNN的模型架构；3)低保真CNN学习球面到基础RBC形状的映射，高保真FNN学习基础RBC与目标RBC间的形态相关性；4)引入物理约束并优化旋转矩阵；5)输入目标RBC部分观测数据，预测完整三维形态；6)使用坐标精度和相对误差评估预测性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将多保真度神经网络应用于红血细胞三维形态重建；2)基于拓扑学同胚理论提供严格数学基础；3)引入物理约束增强模型鲁棒性；4)发现特征感知采样可改善重建质量；5)使用DPD模拟生成多样化训练数据。相比之前的工作，该方法大大减少了训练数据需求，利用参考形状先验知识增强预测，通过物理约束提高抗噪能力，强调特征感知采样重要性，具有更好通用性和计算效率，更适合实际应用场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文开发了一种基于多保真度神经网络的创新方法，能够从部分观测数据中高精度重建正常和疾病相关红血细胞的三维形态，为血液疾病研究和诊断提供了高效工具。'}


### 论文摘要

Reconstruction of 3D erythrocyte or red blood cell (RBC) morphology from partial observations, such as microscope images, is essential for understanding the physiology of RBC aging and the pathology of various RBC disorders. In this study, we propose a multi-fidelity neural network (MFNN) approach to fuse high-fidelity cross-sections of an RBC, with a morphologically similar low-fidelity reference 3D RBC shape to recover its full 3D surface. The MFNN predictor combines a convolutional neural network trained on low-fidelity reference RBC data with a feedforward neural network that captures nonlinear morphological correlations, and augments training with surface area and volume constraints for regularization in the low-fidelity branch. This approach is theoretically grounded by a topological homeomorphism between a sphere and 3D RBC surfaces, with training data generated by dissipative particle dynamics simulations of stomatocyte-discocyte-echinocyte transformation. Benchmarking across diverse RBC shapes observed in normal and aged populations, our results show that the MFNN predictor can reconstruct complex RBC morphologies with over 95% coordinate accuracy when provided with at least two orthogonal cross-sections. It is observed that informative oblique cross-sections intersecting spicule tips of echinocytes improve both local and global feature reconstruction, highlighting the value of feature-aware sampling. Our study further evaluates the influence of sampling strategies, shape dissimilarity, and noise, showing enhanced robustness under physically constrained training. Altogether, these results demonstrate the capability of MFNN to reconstruct the 3D shape of normal and aged RBCs from partial cross-sections as observed in conventional microscope images, which could facilitate the quantitative analysis of RBC morphological parameters in normal and disease-related RBC samples.

---

## 133. RoMa v2: Harder Better Faster Denser Feature Matching

**论文链接:** [http://arxiv.org/abs/2511.15706v1](http://arxiv.org/abs/2511.15706v1)

**作者:** Johan Edstedt, David Nordström, Yushan Zhang, Georg Bökman, Jonathan Astermark, Viktor Larsson, Anders Heyden, Fredrik Kahl, Mårten Wadenbäck, Michael Felsberg

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种新的密集特征匹配方法，通过一系列系统改进解决了现有方法的局限性，实现了更高的准确性和效率。

### 背景

密集特征匹配已成为估计3D场景两张图像之间所有对应关系的黄金标准，因为它具有高准确性和鲁棒性。然而，现有方法在困难的现实场景中表现不佳，且高精度模型通常速度慢，限制了应用范围。

### 目的

解决现有密集匹配器的弱点，创建一个性能显著提升的模型，使其能够处理复杂匹配任务，同时提高训练速度并减少内存使用。

### 方法

构建新颖的匹配架构和损失函数，结合多样化训练分布；采用解耦的两阶段匹配-then-细化管道加速训练；使用自定义CUDA内核减少内存消耗；利用DINOv3基础模型增强鲁棒性和无偏性。

### 主要发现

提出的新型匹配器建立了新的最先进水平，比现有方法显著更准确。

### 结论

通过系统性改进，成功创建了性能显著提升的密集特征匹配模型，在准确性和效率方面均超越了现有方法。

### 翻译

密集特征匹配旨在估计3D场景两张图像之间的所有对应关系，并因其高准确性和鲁棒性最近被确立为黄金标准。然而，现有的密集匹配器在许多困难的现实场景中仍然失败或表现不佳，且高精度模型通常速度慢，限制了它们的适用性。在本文中，我们通过一系列系统改进从多个角度攻击这些弱点，共同产生了一个显著更好的模型。特别是，我们构建了一种新颖的匹配架构和损失函数，结合精心挑选的多样化训练分布，使我们的模型能够解决许多复杂的匹配任务。我们通过解耦的两阶段匹配-then-细化管道进一步加速训练，同时通过自定义CUDA内核显著减少细化阶段的内存使用。最后，我们利用最近的DINOv3基础模型以及其他多种见解使模型更加鲁棒和无偏。在我们的大量实验中，结果表明，提出的新型匹配器建立了新的最先进水平，比其前身显著更准确。代码可在 https://github.com/Parskatt/romav2 获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决密集特征匹配中的两个核心挑战：现有方法在许多具有挑战性的现实场景中表现不佳，以及高精度模型通常运行速度慢限制了实际应用。这个问题很重要，因为密集特征匹配是计算机视觉中许多下游任务（如视觉定位和3D重建）的基础，精确可靠的对应关系对这些任务的稳健运行至关重要，而现有方法在极端光照变化、视点变化等场景下表现不佳且效率低下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析RoMa和UFM两种方法的优缺点进行设计：RoMa在极端外观变化下鲁棒但速度慢，UFM速度快但需要微调且在极端外观变化下表现差。作者结合两者的优点，采用冻结的DINOv3基础模型提高鲁棒性，使用两阶段匹配-精炼流水线提高效率，并构建新的匹配架构和损失函数。该方法借鉴了RoMa的两阶段架构和UFM的解耦训练范式，同时升级了特征提取器为DINOv3。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合鲁棒性和效率：使用冻结基础模型提高对极端变化的鲁棒性，采用两阶段架构提高效率，预测误差协方差提升下游性能，使用多样化训练数据平衡不同场景表现。整体流程分为两阶段：1)粗匹配器使用DINOv3提取特征，通过多视图Transformer预测粗略变换和置信度；2)精炼器使用三个不同步长的CNN模型进行亚像素精炼，并预测误差协方差。训练采用先匹配后精炼的两阶段方式，使用不同学习率和批量大小。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的匹配架构结合变换和相关损失，使用注意力机制替代高斯过程；2)更高效的精炼器通过自定义CUDA内核减少内存使用；3)多样化的训练数据混合平衡宽基线和窄基线场景；4)像素级误差协方差预测提升下游任务性能；5)使用EMA解决亚像素偏差。相比RoMa，它升级了特征提取器，改进了架构，添加了误差预测，提高了效率；相比UFM，它保持冻结编码器提高鲁棒性，添加辅助目标，在保持精度的同时内存占用更小。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RoMa v2通过结合新型匹配架构、高效精炼策略、多样化训练数据和误差协方差预测，实现了在保持高精度的同时显著提高速度和降低内存占用的密集特征匹配，在多个基准测试上设立了新的最先进水平。'}


### 论文摘要

Dense feature matching aims to estimate all correspondences between two images of a 3D scene and has recently been established as the gold-standard due to its high accuracy and robustness. However, existing dense matchers still fail or perform poorly for many hard real-world scenarios, and high-precision models are often slow, limiting their applicability. In this paper, we attack these weaknesses on a wide front through a series of systematic improvements that together yield a significantly better model. In particular, we construct a novel matching architecture and loss, which, combined with a curated diverse training distribution, enables our model to solve many complex matching tasks. We further make training faster through a decoupled two-stage matching-then-refinement pipeline, and at the same time, significantly reduce refinement memory usage through a custom CUDA kernel. Finally, we leverage the recent DINOv3 foundation model along with multiple other insights to make the model more robust and unbiased. In our extensive set of experiments we show that the resulting novel matcher sets a new state-of-the-art, being significantly more accurate than its predecessors. Code is available at https://github.com/Parskatt/romav2

---

## 134. Think Visually, Reason Textually: Vision-Language Synergy in ARC

**论文链接:** [http://arxiv.org/abs/2511.15703v1](http://arxiv.org/abs/2511.15703v1)

**作者:** Beichen Zhang, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang

**发布时间:** 2025-11-19

### GPT解析

### 总结

研究提出了一种结合视觉和语言的协同推理方法，解决了前沿基础模型在从最小示例中进行抽象推理方面的局限性，在ARC-AGI任务上实现了最高4.33%的性能提升。

### 背景

从最小示例中进行抽象推理仍是GPT-5和Grok 4等前沿基础模型未解决的核心问题，这些模型无法从少量示例中推断结构化转换规则，而ARC-AGI为这一能力提供了严格的测试平台。现有方法主要将ARC-AGI视为纯文本推理任务，忽视了人类解决此类谜题时对视觉抽象的依赖。

### 目的

探索如何结合视觉和语言的优势，提高基础模型在ARC-AGI任务上的抽象推理能力，实现更接近人类的智能表现。

### 方法

提出两种协同策略：(1)视觉-语言协同推理(VLSR)，将ARC-AGI分解为模态对齐的子任务；(2)模态切换自我纠正(MSSC)，利用视觉验证基于文本的推理实现内在错误纠正。

### 主要发现

简单将ARC-AGI网格渲染为图像会因规则执行不精确导致性能下降；视觉和语言在不同推理阶段具有互补优势，视觉支持全局模式抽象和验证，语言专门从事符号规则制定和精确执行；结合视觉和语言的协同方法能显著提升性能。

### 结论

将视觉抽象与语言推理相结合是未来基础模型实现可推广、类人智能的关键一步，研究源代码即将发布。

### 翻译

从最小示例中进行抽象推理仍然是GPT-5和Grok 4等前沿基础模型未解决的核心问题。这些模型仍无法从少量示例中推断出结构化转换规则，这是人类智能的关键特征。用于人工通用智能的抽象与推理语料库(ARC-AGI)为这一能力提供了严格的测试平台，要求概念规则归纳和向新任务的迁移。大多数现有方法将ARC-AGI视为纯文本推理任务，忽视了人类在解决此类谜题时严重依赖视觉抽象的事实。然而，我们的初步实验揭示了一个悖论：简单地将ARC-AGI网格渲染为图像会因规则执行不精确而导致性能下降。这引出了我们的核心假设：视觉和语言在不同的推理阶段具有互补优势：视觉支持全局模式抽象和验证，而语言专门从事符号规则制定和精确执行。基于这一见解，我们引入了两种协同策略：(1)视觉-语言协同推理(VLSR)，将ARC-AGI分解为模态对齐的子任务；(2)模态切换自我纠正(MSSC)，利用视觉来验证基于文本的推理，实现内在错误纠正。大量实验表明，该方法在多种旗舰模型和多个ARC-AGI任务上比仅文本的基线提高了高达4.33%的性能。我们的研究结果表明，将视觉抽象与语言推理相结合是未来基础模型实现可推广、类人智能的关键一步。源代码即将发布。


### 论文摘要

Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code will be released soon.

---

## 135. MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

**论文链接:** [http://arxiv.org/abs/2511.15690v1](http://arxiv.org/abs/2511.15690v1)

**作者:** Yushi Huang, Zining Wang, Zhihang Yuan, Yifu Ding, Ruihao Gong, Jinyang Guo, Xianglong Liu, Jun Zhang

**发布时间:** 2025-11-19

**备注:** Code will be released upon acceptance

### GPT解析

### 总结

本文提出MoDES框架，一种训练-free方法，通过全局调节的局部门控机制和双模态阈值化方法实现MoE多模态大语言模型的高效准确推理。

### 背景

MoE多模态大语言模型在视觉-语言任务中表现出色但计算效率低下。现有专家跳过方法专为单模态设计，直接应用于多模态会导致显著性能下降。

### 目的

开发训练-free框架，实现专家自适应跳过，提高MoE多模态大语言模型推理效率和准确性。

### 方法

1) 提出MoDES框架；2) 实现全局调节的局部门控(GMLG)机制整合全局层重要性；3) 应用双模态阈值化(DMT)方法分别处理各模态令牌；4) 引入前沿搜索算法设置最优阈值，缩短收敛时间。

### 主要发现

1) 单模态专家跳过方法直接用于多模态会导致性能下降；2) 原因是未考虑专家异构贡献和令牌模态特定行为；3) MoDES在3个模型系列13个基准测试中表现优异；4) 跳过88%专家时性能提升达10.67%；5) 推理速度显著提高，prefilling时间提升2.16倍，decoding时间提升1.26倍。

### 结论

MoDES是首个解决MoE多模态大语言模型推理效率问题的训练-free框架，通过自适应专家跳实现了高效准确推理，同时显著提高推理速度而不牺牲模型性能。

### 翻译

混合专家(MoE)多模态大语言模型(MLLMs)在视觉-语言任务中表现出色，但存在高计算效率低下问题。专家跳过方法被提出用于减少推理开销，但这些专为单模态设计的方法应用于多模态时会导致显著性能下降，主要因为未考虑MoE层中专家的异构贡献和令牌的模态特定行为。受此启发，我们提出MoDES，首个训练-free框架，通过全局调节的局部门控机制和双模态阈值化方法实现专家自适应跳过，并引入前沿搜索算法优化阈值设置。实验证明MoDES在保持准确性的同时显著提高推理速度。


### 论文摘要

Mixture-of-Experts (MoE) Multimodal large language models (MLLMs) excel at vision-language tasks, but they suffer from high computational inefficiency. To reduce inference overhead, expert skipping methods have been proposed to deactivate redundant experts based on the current input tokens. However, we find that applying these methods-originally designed for unimodal large language models (LLMs)-to MLLMs results in considerable performance degradation. This is primarily because such methods fail to account for the heterogeneous contributions of experts across MoE layers and modality-specific behaviors of tokens within these layers. Motivated by these findings, we propose MoDES, the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. It incorporates a globally-modulated local gating (GMLG) mechanism that integrates global layer-wise importance into local routing probabilities to accurately estimate per-token expert importance. A dual-modality thresholding (DMT) method is then applied, which processes tokens from each modality separately, to derive the skipping schedule. To set the optimal thresholds, we introduce a frontier search algorithm that exploits monotonicity properties, cutting convergence time from several days to a few hours. Extensive experiments for 3 model series across 13 benchmarks demonstrate that MoDES far outperforms previous approaches. For instance, when skipping 88% experts for Qwen3-VL-MoE-30B-A3B-Instruct, the performance boost is up to 10.67% (97.33% vs. 86.66%). Furthermore, MoDES significantly enhances inference speed, improving the prefilling time by 2.16$\times$ and the decoding time by 1.26$\times$.

---

## 136. Generalized Borel Sets

**论文链接:** [http://arxiv.org/abs/2511.15663v1](http://arxiv.org/abs/2511.15663v1)

**作者:** Claudio Agostini, Nick Chapman, Luca Motto Ros, Beatrice Pitton

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文推广了经典描述集合论中的Borel层次结构，在满足特定条件的不可数基数κ的Polish-like空间框架中研究相关问题。

### 背景

经典描述集合论中的Borel层次结构推广引发了一些基础性问题，需要在新框架下进行探索。

### 目的

系统研究在一般框架下的Borel层次结构问题，探索其性质、不坍塌条件以及不同基数情况下的独特现象。

### 方法

在满足2的κ次方小于等于κ的不可数基数κ（可能是奇异的）的Polish-like空间框架中研究，使用集合论和拓扑学方法，并通过强制构造各种模型来验证理论。

### 主要发现

1) 任何权重不超过κ的正则Hausdorff空间的κ⁺-Borel层次结构的基本性质及其不坍塌的充分条件；2) 在奇异基数情况下存在第二个、不同的κ-Borel层次结构，它比κ⁺-Borel层次结构更精细；3) 对于正则基数，解决了广义Baire空间子空间上κ⁺-Borel层次结构行为的三个问题，实现了该层次结构长度的几种非平凡构型。

### 结论

通过研究，揭示了在一般框架下Borel层次结构的性质，特别是在奇异基数情况下的独特现象，以及正则基数情况下κ⁺-Borel层次结构的行为特征。

### 翻译

推广经典描述集合论引发关于Borel层次结构的基础性问题。本文我们在满足2的κ次方小于等于κ的不可数基数κ（可能是奇异的）的Polish-like空间的一般框架中系统研究这些问题。我们提供了任何权重不超过κ的正则Hausdorff空间的κ⁺-Borel层次结构的基本性质，并建立了其不坍塌的充分条件。我们突显了在奇异基数情况下出现的独特现象，即存在第二个、不同的Borel层次结构，即κ-Borel层次结构：我们证明它比κ⁺-Borel层次结构更精细，然后精确描述两者之间的关系。最后，对于正则基数，我们通过强制构造各种模型，解决了关于广义Baire空间子空间上κ⁺-Borel层次结构行为的三个问题，实现了该层次结构长度的几种非平凡构型。


### 论文摘要

Generalizing classical descriptive set theory opens foundational questions about the Borel hierarchy. In this paper we systematically study those questions, working in the general framework of Polish-like spaces relative to an uncountable cardinal $κ$, possibly singular, satisfying $2^{<κ}=κ$. We provide fundamental properties of the $κ^+$-Borel hierarchy of any regular Hausdorff space of weight at most $κ$, and establish sufficient conditions for its non-collapse. We highlight a unique phenomenon that arises in the case of singular cardinals, namely, the existence of a second, distinct Borel hierarchy, the $κ$-Borel hierarchy: we prove that it is strictly finer than the $κ^+$-Borel hierarchy, and then characterize the precise relationship between the two. Finally, for regular cardinals, we resolve three questions about the behavior of the $κ^+$-Borel hierarchy on subspaces of the generalized Baire space ${}^κκ$, constructing various models via forcing where several nontrivial constellations for the length of the $κ^+$-Borel hierarchy on the space are realized.

---

## 137. VisPlay: Self-Evolving Vision-Language Models from Images

**论文链接:** [http://arxiv.org/abs/2511.15661v1](http://arxiv.org/abs/2511.15661v1)

**作者:** Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, Yonghui Yang

**发布时间:** 2025-11-19

### GPT解析

### 总结

VisPlay是一种自我进化的强化学习框架，使视觉语言模型能够利用大量未标记图像数据自主提高推理能力，无需人工标注标签或特定任务启发式方法。

### 背景

强化学习为改进视觉语言模型在复杂推理任务上提供了原则性框架，但现有RL方法通常依赖人工标注标签或任务特定启发式方法来定义可验证奖励，这些方法成本高昂且难以扩展。

### 目的

开发一种自我进化的RL框架，使VLMs能够利用大量未标记图像数据自主提高推理能力。

### 方法

VisPlay将模型分为两个交互角色：基于图像条件的提问者(生成具有挑战性但可回答的视觉问题)和多模态推理者(生成银响应)。这两个角色通过组相对策略优化(GRPO)进行联合训练，融入多样性和难度奖励以平衡问题复杂性与答案质量。

### 主要发现

VisPlay在Qwen2.5-VL和MiMo-VL两个模型系列上高效扩展，在八个基准测试(包括MM-Vet和MMMU)上，在视觉推理、组合泛化和幻觉减少方面取得持续改进。

### 结论

VisPlay展示了一条通向自我进化多模态智能的可扩展路径。

### 翻译

强化学习(RL)为改进视觉语言模型(VLMs)在复杂推理任务上提供了原则性框架。然而，现有的RL方法通常依赖人工标注的标签或任务特定的启发式方法来定义可验证的奖励，这些方法成本高昂且难以扩展。我们介绍了VisPlay，一个自我进化的RL框架，使VLMs能够利用大量未标记的图像数据自主提高其推理能力。从基础VLM开始，VisPlay将模型分配为两个交互角色：基于图像条件的提问者(生成具有挑战性但可回答的视觉问题)和多模态推理者(生成银响应)。这两个角色通过组相对策略优化(GRPO)进行联合训练，该方法融入了多样性和难度奖励，以平衡生成问题的复杂性与银答案的质量。VisPlay能够在两个模型系列(Qwen2.5-VL和MiMo-VL)上高效扩展。在八个基准测试(包括MM-Vet和MMMU)上，VisPlay在视觉推理、组合泛化和幻觉减少方面取得了持续改进，展示了通向自我进化多模态智能的可扩展路径。项目页面可在https://bruno686.github.io/VisPlay/获取。


### 论文摘要

Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/

---

## 138. Spatial scale separation and emergent patterns in coupled diffusive-nondiffusive systems

**论文链接:** [http://arxiv.org/abs/2511.15648v1](http://arxiv.org/abs/2511.15648v1)

**作者:** Théo André, Szymon Cygan, Anna Marciniak-Czochra, Finn Münnich

**发布时间:** 2025-11-19

### GPT解析

### 总结

本研究探讨了具有扩散和非扩散组分的反应扩散系统中的模式形成，建立了远离平衡态模式的存在性，并为扩散驱动不稳定性提供了必要和充分条件。研究发现了新型模式形成机制，扩展了经典反应扩散框架的理论基础。

### 背景

经典反应扩散方程在模式形成研究中已有广泛应用，但之前的研究主要关注纯扩散系统或仅将DDI与纯非扩散子系统的不稳定性相联系，限制了模式形成的理论范围。

### 目的

研究旨在探索具有扩散和非扩散组分的反应扩散系统中的模式形成机制，建立DDI的必要和充分条件，并揭示超出经典反应扩散框架的新模式形成可能性。

### 方法

通过理论分析和证明，结合数值分岔分析和计算机模拟，研究扩散和非扩散组分之间的相互作用如何影响模式形成。特别关注了具有两个扩散和一个非扩散组分的系统案例。

### 主要发现

1) 证明了存在远离平衡态的模式，在非扩散组分中表现出分支切换和不连续性；2) DDI可以来自非扩散和慢扩散组分的子系统，而不仅限于纯非扩散子系统；3) 为具有任意数量组分的系统提供了DDI的简单充分条件；4) 完全分类了两个扩散和一个非扩散组分系统中所有可能的DDI来源。

### 结论

扩散和非扩散动力学之间的耦合可以产生超出经典反应扩散框架范围的模式，这些发现扩展了模式形成的理论基础，为理解和控制复杂系统中的自组织行为提供了新的视角。

### 翻译

本研究探讨了具有扩散和非扩散组分的反应扩散系统中的模式形成，建立了远离平衡态模式的存在性，并为扩散驱动不稳定性(DDI)提供了必要和充分条件。特别是，我们证明了存在远离平衡态的模式，这些模式在非扩散组分中表现出分支切换和不连续性，这在经典反应扩散方程中是不可能发生的。虽然之前的工作将DDI与纯非扩散子系统的不稳定性联系起来，从而破坏了所有规则的Turing模式，但我们表明DDI也可以来自涉及非扩散和慢扩散组分的子系统。这为具有任意数量组分的系统中的DDI提供了简单的充分条件。此外，在两个扩散和一个非扩散组分的情况下，我们完全分类了所有可能的DDI来源，并通过数值分岔分析和模拟支持的基于受体的模型来说明我们的结果。这些发现扩展了模式形成的理论基础，展示了扩散和非扩散动力学之间的耦合如何产生超出经典反应扩散框架范围的模式。


### 论文摘要

This paper investigates pattern formation in reaction-diffusion systems with both diffusive and nondiffusive components, establishing the existence of far-from-equilibrium patterns and providing necessary and sufficient conditions for diffusion-driven instability (DDI). In particular, we prove the existence of far-from-equilibrium patterns exhibiting branch-switching and discontinuities in the nondiffusive components, which cannot occur in classical reaction-diffusion equations. While previous work has linked DDI to instability in the purely nondiffusive subsystem -- thereby destabilizing all regular Turing patterns -- we show that DDI can also arise from subsystems involving nondiffusive and slow-diffusive components. This leads to simple sufficient conditions for DDI in systems with arbitrary numbers of components. Further, we fully classify all possible sources of DDI in the case of two diffusive and one nondiffusive component, illustrating our results with a receptor-based model supported by numerical bifurcation analysis and simulations. These findings extend the theoretical foundations of pattern formation, demonstrating how coupling between diffusive and nondiffusive dynamics can generate patterns beyond the reach of the classical reaction-diffusion framework.

---

## 139. Qualitative and quantitative hard-tissue MRI with portable Halbach scanners

**论文链接:** [http://arxiv.org/abs/2511.15617v1](http://arxiv.org/abs/2511.15617v1)

**作者:** Jose Borreguero, Luiz G. C. Santos, Lorena Vega Cid, Elisa Castañón, Marina Fernández-García, Pablo Benlloch, Rubén Bosch, Jesús Conejero, Pablo García-Cristóbal, Alba González-Cebrián, Teresa Guallart-Naval, Eduardo Pallás, Laia Porcar, Lucas Swistunow, Jose Miguel Algarín, Fernando Galve, Joseba Alonso

**发布时间:** 2025-11-19

**备注:** 15 pages, 12 figures

### GPT解析

### 总结

本研究开发了一种低成本便携式MRI扫描仪，用于对软组织和硬组织进行体内成像和定量弛豫映射，建立了在强场不均匀性系统中的零回波时间(ZTE)成像方法学基础。通过优化PETRA序列，可在15分钟内获得膝盖和踝关节的3D图像，揭示传统自旋回波序列不可见的硬组织结构。扩展的SPDS方法实现了准确的场图绘制，VFA方法提供了低磁场条件下硬组织的首次体内T1测量。

### 背景

便携式低场MRI系统在临床应用中面临强场不均匀性的挑战，限制了成像质量，特别是对硬组织的成像能力有限。

### 目的

展示使用低成本、便携式MRI扫描仪对软组织和硬组织进行体内成像和定量弛豫映射的可行性，并建立零回波时间(ZTE)成像在强场不均匀性系统中的方法学基础。

### 方法

开发完整的无伪影低场ZTE成像框架，包括RF脉冲预/反强调校准、扩展的单点双 shot (SPDS)协议用于同时B0和B1映射、以及基于模型的重建方法。在phantom和人体膝盖、踝关节上进行ZTE成像和VFA T1映射，并与标准RARE和STIR采集进行对比。

### 主要发现

优化的PETRA序列可在临床兼容时间内(<15分钟)产生膝盖和踝关节的3D图像，揭示韧带、肌腱、软骨和骨骼等硬组织；扩展的SPDS方法实现了准确的场图绘制；VFA方法提供了在B0 < 0.1 T条件下硬组织的首次体内T1测量。

### 结论

所提出的框架扩展了便携式低场MRI中可行的脉冲序列范围，证明了ZTE在基于Halbach的实惠系统中对肌肉骨骼组织进行定量和结构成像的潜力。

### 翻译

Purpose: To demonstrate the feasibility of performing in vivo imaging and quantitative relaxation mapping of soft and hard tissues using a low-cost, portable MRI scanner, and to establish the methodological foundations for zero echo time (ZTE) imaging in systems affected by strong field inhomogeneities. Methods: A complete framework for artifact-free ZTE imaging at low field was developed, including: (i) RF pulse pre/counteremphasis calibration to minimize ring-down and electronics switching time; (ii) an extension of a recent single-point double-shot (SPDS) protocol for simultaneous B0 and B1 mapping; and (iii) a model-based reconstruction incorporating these field maps into the encoding matrix. ZTE imaging and variable flip angle (VFA) T1 mapping were performed on phantoms and in vivo human knees and ankles, and benchmarked against standard RARE and STIR acquisitions. Results: The optimized PETRA sequence produced 3D images of knees and ankles within clinically compatible times (< 15 min), revealing hard tissues such as ligaments, tendons, cartilage, and bone that are invisible in spin-echo sequences. The extended SPDS method enabled accurate field mapping, while the VFA approach provided the first in vivo T1 measurements of hard tissues at B0 < 0.1 T. Conclusions: The proposed framework broadens the range of pulse sequences feasible in portable low-field MRI and demonstrates the potential of ZTE for quantitative and structural imaging of musculoskeletal tissues in affordable Halbach-based systems.


### 论文摘要

Purpose: To demonstrate the feasibility of performing in vivo imaging and quantitative relaxation mapping of soft and hard tissues using a low-cost, portable MRI scanner, and to establish the methodological foundations for zero echo time (ZTE) imaging in systems affected by strong field inhomogeneities. Methods: A complete framework for artifact-free ZTE imaging at low field was developed, including: (i) RF pulse pre/counteremphasis calibration to minimize ring-down and electronics switching time; (ii) an extension of a recent single-point double-shot (SPDS) protocol for simultaneous B0 and B1 mapping; and (iii) a model-based reconstruction incorporating these field maps into the encoding matrix. ZTE imaging and variable flip angle (VFA) T1 mapping were performed on phantoms and in vivo human knees and ankles, and benchmarked against standard RARE and STIR acquisitions. Results: The optimized PETRA sequence produced 3D images of knees and ankles within clinically compatible times (< 15 min), revealing hard tissues such as ligaments, tendons, cartilage, and bone that are invisible in spin-echo sequences. The extended SPDS method enabled accurate field mapping, while the VFA approach provided the first in vivo T1 measurements of hard tissues at B0 < 0.1 T. Conclusions: The proposed framework broadens the range of pulse sequences feasible in portable low-field MRI and demonstrates the potential of ZTE for quantitative and structural imaging of musculoskeletal tissues in affordable Halbach-based systems.

---

## 140. When to Think and When to Look: Uncertainty-Guided Lookback

**论文链接:** [http://arxiv.org/abs/2511.15613v1](http://arxiv.org/abs/2511.15613v1)

**作者:** Jing Bi, Filippos Bellos, Junjia Guo, Yayuan Li, Chao Huang, Yunlong, Tang, Luchuan Song, Susan Liang, Zhongfei, Zhang, Jason J. Corso, Chenliang Xu

**发布时间:** 2025-11-19

### GPT解析

### 总结

研究测试时思维对大型视觉语言模型性能的影响，发现更多思维并不总是更好，提出不确定性引导回看策略提升模型表现。

### 背景

测试时思维（生成明确的中间推理链）已知能提高大型语言模型性能，并最近在大型视觉语言模型中显示出显著提升，但缺乏对思维如何影响视觉推理的系统性分析。

### 目的

提供首个关于思维如何影响视觉推理的大规模、系统性分析，评估LVLMs在视觉推理任务上的表现并改进解码策略。

### 方法

在慷慨的token预算和多通道解码条件下，评估InternVL3.5和Qwen3-VL家族的十种变体；提出不确定性引导回看策略，结合不确定性信号与自适应回看提示及广度搜索。

### 主要发现

更多思维并不总是更好，长链推理常产生忽略图像的错误轨迹；成功的推理轨迹中明确引用图像的短回看短语明显丰富，与更好的视觉基础相关；所提方法在标准思维较弱的类别中提升最大。

### 结论

不确定性引导回看策略提高了整体MMMU性能，超过多个解码基线，在固定模型家族和token预算下达到新水平，且在五个额外基准测试中具有泛化性。

### 翻译

测试时思维（即生成明确的中间推理链）已知能提高大型语言模型的性能，并最近在大型视觉语言模型中显示出显著提升。然而，尽管这些结果很有希望，仍然缺乏对思维如何实际影响视觉推理的系统性分析。我们提供了首个此类分析，通过大规模、受控的比较评估了InternVL3.5和Qwen3-VL家族中的十种变体在MMMU-val上的表现，使用了慷慨的token预算和多通道解码。我们表明，更多的思维并不总是更好；长链推理常常产生忽略图像的错误轨迹，表现不如标准指令模式运行的相同模型。更深入的分析显示，成功的推理轨迹中明确引用图像的短回看短语明显丰富，与更好的视觉基础相关。基于这一见解，我们提出不确定性引导回看，一种无需训练的解码策略，结合不确定性信号与自适应回看提示及广度搜索。我们的方法提高了整体MMMU性能，在标准思维较弱的类别中带来最大提升，并超过多个强大的解码基线，在固定模型家族和token预算下达到新水平。我们进一步表明，该解码策略具有泛化性，在五个额外的基准测试中取得一致改进，包括两个广泛的多模态套件和专注于数学的视觉推理数据集。


### 论文摘要

Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.

---

## 141. Cartan meets Cramér-Rao

**论文链接:** [http://arxiv.org/abs/2511.15612v1](http://arxiv.org/abs/2511.15612v1)

**作者:** Sunder Ram Krishnan

**发布时间:** 2025-11-19

**备注:** 8 pages, version 1

### GPT解析

### 总结

本文为作者先前工作中引入的Cramér-Rao界的曲率感知改进建立了射丛和嘉当几何基础，展示了方差界的外部修正可以在嘉当延拓框架内以内在形式表达。

### 背景

之前的工作中引入了对Cramér-Rao界的曲率感知改进，这些改进基于模型密度f(·;θ)的平方根嵌入s_θ=√f(·;θ)∈L^2(μ)的第二基本形式推导出的外部修正。

### 目的

为方差界的外部修正提供内在的几何表述，建立方差界、曲率和估计量效率之间的几何联系，为高阶信息不等式提供新的微分方程和联络论解释。

### 方法

从有限射丛J^m(R×R)的规范接触形式和全导数出发，构建嘉当分布和相关Ehresmann联络；在统计射丛E=R×L^2(μ)中，分析估计量误差与平方根映射满足微分方程之间的关系；研究由微分方程定义的J^m(E)的子流形及其与嘉当向量场的关系。

### 主要发现

估计量误差位于s_θ直到m阶的导数张成的空间中的条件等价于平方根映射满足一个m阶线性微分方程；由该方程定义的子流形表示m阶有效模型的轨迹；CRB和Bhattacharyya型界背后的代数投影条件与射丛层次中统计截面的几何可积条件之间存在一一对应关系。

### 结论

所得框架通过嘉当分布的几何将方差界、曲率和估计量效率联系起来，为高阶信息不等式提供了新的微分方程和联络论解释，深化了对统计估计中曲率修正的几何理解。

### 翻译

本文为我们在先前工作中引入的Cramér-Rao界的曲率感知改进建立了射丛和嘉当几何基础。我们表明，先前从模型密度f(·;θ)的平方根嵌入s_θ=√f(·;θ)∈L^2(μ)的第二基本形式推导出的方差界外部修正，可以在嘉当延拓框架内以内在形式表达。从有限射丛J^m(R×R)的规范接触形式和全导数开始，我们构建了嘉当分布和相关Ehresmann联络，其不可积性和挠率编码了统计估计中曲率修正的几何来源。在统计射丛E=R×L^2(μ)中，我们指出估计量误差位于s_θ直到m阶的导数张成的空间中的条件等价于平方根映射满足一个m阶线性微分方程。由该方程定义的J^m(E)的子流形表示m阶有效模型的轨迹，且延拓截面必须形成受限嘉当向量场的积分曲线。这建立了CRB和Bhattacharyya型界背后的代数投影条件与射丛层次中统计截面的几何可积条件之间的一一对应关系。所得框架通过嘉当分布的几何将方差界、曲率和估计量效率联系起来，为高阶信息不等式提供了新的微分方程和联络论解释。


### 论文摘要

This paper develops a jet bundle and Cartan geometric foundation for the curvature-aware refinements of the Cramér-Rao bound (CRB) introduced in our earlier work. We show that the extrinsic corrections to variance bounds, previously derived from the second fundamental form of the square root embedding $s_θ=\sqrt{f(\cdot;θ)}\in L^2(μ)$ for model density $f(\cdot;θ)$ with scalar parameter $θ$, admit an intrinsic formulation within the Cartan prolongation framework. Starting from the canonical contact forms and total derivative on the finite jet bundle $J^m(\mathbb{R}\times \mathbb{R})$, we construct the Cartan distribution and the associated Ehresmann connection, whose non-integrability and torsion encode the geometric source of curvature corrections in statistical estimation. In the statistical jet bundle $E=\mathbb{R}\times L^2(μ)$, we point out that the condition for an estimator error to lie in the span of derivatives of $s_θ$ up to order $m$ is equivalent to the square root map satisfying a linear differential equation of order~$m$. The corresponding submanifold of $J^m(E)$ defined by this equation represents the locus of $m$-th order efficient models, and the prolonged section must form an integral curve of the restricted Cartan vector field. This establishes a one-to-one correspondence between algebraic projection conditions underlying CRB and Bhattacharyya-type bounds and geometric integrability conditions for the statistical section in the jet bundle hierarchy. The resulting framework links variance bounds, curvature, and estimator efficiency through the geometry of Cartan distributions, offering a new differential equation and connection-theoretic interpretation of higher-order information inequalities.

---

## 142. HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning

**论文链接:** [http://arxiv.org/abs/2511.15574v1](http://arxiv.org/abs/2511.15574v1)

**作者:** Qihao Yang, Xuelin Wang, Jiale Chen, Xuelian Dong, Yuxin Hao, Tianyong Hao

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI-2026

### GPT解析

### 总结

本文提出了HSKBenchmark，首个针对中文第二语言习得(SLA)的大型语言模型(LLMs)分阶段建模和写作评估基准，通过课程调整框架模拟人类学习轨迹，微调后的模型表现与高级人类学习者相当。

### 背景

语言获取对揭示人类语言智能本质很重要，且能提高LLMs的可解释性。控制人类学习者语言输入在伦理和实践上不可行，给语言获取建模带来挑战，特别是在中文SLA领域。虽然LLMs提供了可控替代方案，但缺乏系统基准支持分阶段建模和评估。

### 目的

创建HSKBenchmark，覆盖HSK 3-6级，包含真实教材(6.76百万token)、16K合成指令样本、30个测试主题和基于语言学的评估系统，支持中文SLA的分阶段建模和评估。

### 方法

引入课程调整框架训练模型从初学者到高级水平；创建评估系统检查级别语法覆盖、写作错误、词汇和句法复杂性及整体评分；构建HSKAgent并在10K学习者作文上微调。

### 主要发现

HSKBenchmark有效建模中文SLA并作为LLMs动态写作评估的可靠基准；微调后的LLMs写作表现与高级人类学习者相当，展示类似人类的获取特征。

### 结论

HSKBenchmark、HSKAgent和检查点作为基础工具和资源，为语言获取建模和LLMs可解释性研究提供基础，代码和数据已在GitHub公开。

### 翻译

语言获取对于揭示人类语言智能的本质至关重要，最近已成为提高大型语言模型可解释性的一个有前景的视角。然而，进行需要控制人类学习者语言输入的实验在伦理和实践上都是不可行的。这给语言获取建模的可验证性和可扩展性带来了挑战，特别是在中文第二语言习得方面。虽然大型语言模型提供了可控且可重复的替代方案，但仍然缺乏支持分阶段建模和评估的系统基准。在本文中，我们提出了HSKBenchmark，这是首个针对中文第二语言习得中大型语言模型分阶段建模和写作评估的基准。它涵盖HSK 3至6级，包含6.76百万token的真实教材、16K合成指令样本、30个测试主题以及一个基于语言学的评估系统。为了模拟人类学习轨迹，我们引入了一个课程调整框架，将模型从初学者水平训练到高级水平。创建了一个评估系统来检查基于级别的语法覆盖、写作错误、词汇和句法复杂性以及整体评分。我们还构建了HSKAgent，在10K学习者作文上进行了微调。大量的实验结果表明，HSKBenchmark不仅有效地建模了中文第二语言习得，还作为大型语言模型动态写作评估的可靠基准。我们微调后的大型语言模型在写作方面表现与高级人类学习者相当，并表现出类似人类的获取特征。HSKBenchmark、HSKAgent和检查点作为基础工具和资源，有潜力为未来的语言获取建模和大型语言模型可解释性研究铺平道路。代码和数据可在https://github.com/CharlesYang030/HSKB获取。


### 论文摘要

Language acquisition is vital to revealing the nature of human language intelligence and has recently emerged as a promising perspective for improving the interpretability of large language models (LLMs). However, it is ethically and practically infeasible to conduct experiments that require controlling human learners' language inputs. This poses challenges for the verifiability and scalability of language acquisition modeling, particularly in Chinese second language acquisition (SLA). While LLMs provide a controllable and reproducible alternative, a systematic benchmark to support phase-wise modeling and assessment is still lacking. In this paper, we present HSKBenchmark, the first benchmark for staged modeling and writing assessment of LLMs in Chinese SLA. It covers HSK levels 3 to 6 and includes authentic textbooks with 6.76 million tokens, 16K synthetic instruction samples, 30 test topics, and a linguistically grounded evaluation system. To simulate human learning trajectories, we introduce a curriculum-tuning framework that trains models from beginner to advanced levels. An evaluation system is created to examine level-based grammar coverage, writing errors, lexical and syntactic complexity, and holistic scoring. We also build HSKAgent, fine-tuned on 10K learner compositions. Extensive experimental results demonstrate that HSKBenchmark not only models Chinese SLA effectively, but also serves as a reliable benchmark for dynamic writing assessment in LLMs. Our fine-tuned LLMs have writing performance on par with advanced human learners and exhibit human-like acquisition characteristics. The HSKBenchmark, HSKAgent, and checkpoints serve as foundational tools and resources, with the potential to pave the way for future research on language acquisition modeling and LLMs interpretability. Code and data are publicly available at: https://github.com/CharlesYang030/HSKB.

---

## 143. Multimodal Evaluation of Russian-language Architectures

**论文链接:** [http://arxiv.org/abs/2511.15552v1](http://arxiv.org/abs/2511.15552v1)

**作者:** Artem Chervyakov, Ulyana Isaeva, Anton Emelyanov, Artem Safin, Maria Tikhonova, Alexander Kharitonov, Yulia Lyakh, Petr Surovtsev, Denis Shevelev Vildan Saburov, Vasily Konovalov, Elisei Rykov, Ivan Sviridov, Amina Miftakhova, Ilseyar Alimova, Alexander Panchenko, Alexander Kapitanov, Alena Fenogenova

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文介绍了Mera Multi，一个面向俄语架构的开源多模态评估框架，包含18个针对文本、图像、音频和视频模态的新建评估任务，旨在解决多模态大语言模型智能、局限性和风险理解不足的问题。

### 背景

多模态大语言模型已成为研究焦点，规模和能力迅速发展，但其智能、局限性和风险尚未被充分理解。特别是在俄语语境下，目前还不存在多模态基准测试。

### 目的

为了解决多模态大语言模型的理解不足问题，特别是针对俄语语境，作者引入了Mera Multi多模态评估框架。

### 方法

创建基于指令的评估框架，包含文本、图像、音频和视频模态，构建18个评估任务，适用于通用模型和特定模态架构。提供防止基准泄漏的方法论，包括水印和许可。

### 主要发现

贡献包括：多模态能力的通用分类法；18个关注俄语文化和语言特性的数据集；闭源和开源模型的基线结果；防止基准泄漏的方法论。

### 结论

虽然目前关注的是俄语，但提出的基准为构建类型多样的语言（特别是斯拉夫语系）中的多模态基准提供了可复制的方法论。

### 翻译

多模态大语言模型目前处于研究关注中心，在规模和能力上显示出快速进展，但它们的智能、局限性和风险仍未被充分理解。为解决这些问题，特别是在俄语语境下（目前尚无多模态基准），我们引入了Mera Multi，一个面向俄语架构的开源多模态评估框架。该基准基于指令，包含默认的文本、图像、音频和视频模态，共包含18个新构建的评估任务，适用于通用模型和特定模态架构（图像到文本、视频到文本和音频到文本）。我们的贡献包括：(i) 多模态能力的通用分类法；(ii) 18个完全从头创建的数据集，关注俄语文化和语言特异性、统一提示和指标；(iii) 闭源和开源模型的基线结果；(iv) 防止基准泄漏的方法论，包括私有集的水印和许可。虽然我们目前关注的是俄语，但提出的基准为构建类型多样的语言（特别是斯拉夫语系）中的多模态基准提供了可复制的方法论。


### 论文摘要

Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.

---

## 144. A Full-Induction Magnetohydrodynamics Solver for Liquid Metal Fusion Blankets in Vertex-CFD

**论文链接:** [http://arxiv.org/abs/2511.15549v1](http://arxiv.org/abs/2511.15549v1)

**作者:** Eirik Endeve, Doug Stefanski, Marc-Olivier G. Delchini, Stuart Slattery, Cory D. Hauck, Bruno Turcksin, Sergey Smolentsev

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种集成在开源Vertex-CFD框架中的完整感应磁流体动力学(MHD)求解器，用于模拟液态金属聚变包层在瞬态条件下的多物理场行为，并通过基准问题验证了其准确性和鲁棒性。

### 背景

液态金属聚变包层在聚变反应中产生氚并将中子能量转化为热能，其多物理场建模对于预测性能、确保结构完整性和优化能量生产至关重要。

### 目的

提出、实现并初步验证一个完整感应MHD求解器，集成到开源的Vertex-CFD框架中，实现紧密的多物理场耦合，灵活的软件设计以及跨计算平台的性能可移植性。

### 方法

使用有限元空间离散化、隐式Runge-Kutta时间积分和不精确Newton方法求解离散非线性系统，利用Trilinos包进行高效计算。

### 主要发现

通过选定基准问题的验证表明求解器的准确性和鲁棒性。当应用于理想化的2.5维和全三维包层模型时，Vertex-CFD的结果与最近发表的准二维模拟结果吻合良好。

### 结论

这些发现为使用Vertex-CFD模拟液态金属包层中的瞬态MHD现象建立了计算基础，并为未来的扩展和性能优化开辟了途径。

### 翻译

液态金属聚变包层产生氚并将聚变反应产生的中子能量转化为热能，其多物理场建模对于预测性能、确保结构完整性和优化能量生产至关重要。虽然传统的包层建模在正常稳态条件下通常使用磁流体动力学(MHD)方程的无感应近似，但对于等离子体约束磁场在毫秒时间尺度变化的瞬态场景，需要使用完整感应MHD方法，通过时间相关的感应方程动态演化磁场。本文提出了一个完整感应MHD求解器的公式化、实现和初步验证，该求解器集成在开源的Vertex-CFD框架中，旨在实现紧密的多物理场耦合，灵活的软件设计便于物理模型的扩展和添加，以及跨计算平台的性能可移植性。该求解器使用有限元空间离散化、隐式Runge-Kutta时间积分和不精确Newton方法求解离散非线性系统，利用Trilinos包进行高效计算。针对选定基准问题的验证证明了求解器的准确性和鲁棒性。此外，当求解器应用于理想化的2.5维和全三维包层模型时，使用Vertex-CFD获得的结果与最近发表的准二维模拟结果吻合良好。这些发现为使用Vertex-CFD模拟液态金属包层中的瞬态MHD现象建立了计算基础，并为未来的扩展和性能优化开辟了途径。


### 论文摘要

Multiphysics modeling of liquid metal fusion blankets, which produce tritium and convert energy of neutrons created via fusion reactions into heat, is crucial for predicting performance, ensuring structural integrity, and optimizing energy production. While traditional blanket modeling of liquid metal flows during normal steady operating conditions commonly employs the inductionless approximation of the magnetohydrodynamics (MHD) equations, transient scenarios, when the plasma-confining magnetic field varies on millisecond time scales, require a full-induction MHD approach that dynamically evolves the magnetic field via the time-dependent induction equation. This paper presents the formulation, implementation, and initial verification of a full-induction MHD solver integrated within the open-source Vertex-CFD framework, which aims to achieve tight multiphysics coupling, a flexible software design enabling easy extension and addition of physics models, and performance portability across computing platforms. The solver utilizes finite element spatial discretization, implicit Runge--Kutta time integration, and an inexact Newton method to solve the resulting discrete nonlinear system, leveraging Trilinos packages for efficient computation. Verification against selected benchmark problems demonstrates accuracy and robustness of the solver. Furthermore, when the solver is applied to an idealized blanket model in 2.5D and full 3D, results obtained with Vertex-CFD are in good agreement with recently published quasi-2D simulations. These findings establish a computational foundation for future simulations of transient MHD phenomena in liquid metal blankets with Vertex-CFD, and open avenues for future extensions and performance optimizations.

---

## 145. FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI

**论文链接:** [http://arxiv.org/abs/2511.15481v1](http://arxiv.org/abs/2511.15481v1)

**作者:** Luisa Gallée, Yiheng Xiong, Meinrad Beer, Michael Götz

**发布时间:** 2025-11-19

### GPT解析

### 总结

FunnyNodules是一个全参数化的合成数据集，用于系统性分析医学AI模型中基于属性的推理，解决了医学图像数据集中诊断推理信息稀缺的问题。

### 背景

医学图像数据集稀缺，特别是那些不仅包含诊断标签还包含诊断背后推理过程的数据集。这类推理相关的注释对于开发和评估可解释AI(xAI)模型至关重要，这些模型需要像放射科医生一样基于正确理由做出正确预测。

### 目的

解决医学图像数据集中诊断推理信息稀缺的问题，引入FunnyNodules数据集，用于系统性分析医学AI模型中基于属性的推理。

### 方法

FunnyNodules是一个全参数化的合成数据集，生成具有可控视觉属性的抽象肺结节状形状，如圆形度、边缘锐度和毛刺。目标类别源自预定义的属性组合，允许完全控制将属性链接到诊断类别的决策规则。

### 主要发现

FunnyNodules可用于模型无关的评估，以评估模型是否学习正确的属性-目标关系，解释属性预测中的过度或不足表现，以及分析注意力与属性特定感兴趣区域的对齐情况。

### 结论

FunnyNodules框架完全可定制，支持数据集复杂性、目标定义、类别平衡等方面的变化。具有完整的真实信息，为医学图像分析中的可解释AI方法开发、基准测试和深入分析提供了多功能基础。

### 翻译

密集标注的医学图像数据集不仅捕获诊断标签，还捕获这些诊断背后的基本推理，此类数据集很少见。这类推理相关的注释对于开发和评估可解释AI(xAI)模型至关重要，这些模型的推理方式类似于放射科医生：基于正确理由做出正确预测。为解决这一差距，我们引入了FunnyNodules，这是一个全参数化的合成数据集，专为医学AI模型中基于属性的推理的系统分析而设计。该数据集生成具有可控视觉属性的抽象肺结节状形状，如圆形度、边缘锐度和毛刺。目标类别源自预定义的属性组合，允许完全控制将属性链接到诊断类别的决策规则。我们展示了如何使用FunnyNodules进行与模型无关的评估，以评估模型是否学习正确的属性-目标关系，解释属性预测中的过度或不足表现，以及分析注意力与属性特定感兴趣区域的对齐情况。该框架完全可定制，支持数据集复杂性、目标定义、类别平衡等方面的变化。凭借完整的真实信息，FunnyNodules为医学图像分析中的可解释AI方法的开发、基准测试和深入分析提供了多功能基础。


### 论文摘要

Densely annotated medical image datasets that capture not only diagnostic labels but also the underlying reasoning behind these diagnoses are scarce. Such reasoning-related annotations are essential for developing and evaluating explainable AI (xAI) models that reason similarly to radiologists: making correct predictions for the right reasons. To address this gap, we introduce FunnyNodules, a fully parameterized synthetic dataset designed for systematic analysis of attribute-based reasoning in medical AI models. The dataset generates abstract, lung nodule-like shapes with controllable visual attributes such as roundness, margin sharpness, and spiculation. Target class is derived from a predefined attribute combination, allowing full control over the decision rule that links attributes to the diagnostic class. We demonstrate how FunnyNodules can be used in model-agnostic evaluations to assess whether models learn correct attribute-target relations, to interpret over- or underperformance in attribute prediction, and to analyze attention alignment with attribute-specific regions of interest. The framework is fully customizable, supporting variations in dataset complexity, target definitions, class balance, and beyond. With complete ground truth information, FunnyNodules provides a versatile foundation for developing, benchmarking, and conducting in-depth analyses of explainable AI methods in medical image analysis.

---

## 146. TSFM in-context learning for time-series classification of bearing-health status

**论文链接:** [http://arxiv.org/abs/2511.15447v1](http://arxiv.org/abs/2511.15447v1)

**作者:** Michel Tokic, Slobodan Djukanović, Anja von Beuningen, Cheng Feng

**发布时间:** 2025-11-19

**备注:** Preprint submitted to ESANN 2026

### GPT解析

### 总结

本文提出了一种基于时间序列基础模型(TSFM)的上下文学习方法，用于对未见过的数据进行分类，无需微调模型。该方法通过将示例表示为目标和协变量形式，应用于振动数据以评估伺服电机轴承的健康状态。

### 背景

研究使用时间序列基础模型(TSFM)，并应用于振动数据以评估伺服电机轴承的健康状态，旨在从特定AI解决方案向更广泛的AI驱动维护系统发展。

### 目的

提出一种无需微调模型即可对TSFM训练数据集以外的数据进行分类的方法，并扩展到更广泛的AI驱动维护系统应用。

### 方法

使用上下文学习进行分类，在模型提示中将示例表示为目标(类别ID)和协变量(数据矩阵)的形式；将频域参考信号转换为伪时间序列模式，生成对齐的协变量和目标信号，利用TSFM预测分类数据对应预定义标签的概率。

### 主要发现

该方法能够在不微调模型的情况下对未见过的数据进行分类，利用预训练模型的扩展性在不同操作条件下都表现出有效性。

### 结论

这标志着从定制化狭义AI解决方案向更广泛的AI驱动维护系统的重要进展。

### 翻译

本文介绍了一种在时间序列基础模型(TSFM)中使用上下文学习的分类方法。我们展示了如何对不属于TSFM训练数据集的数据进行分类，而无需微调模型。示例在模型提示中表示为目标(类别ID)和协变量(数据矩阵)的形式，这使通过上下文学习能够沿预测轴对未知协变量数据模式进行分类。我们将此方法应用于振动数据，以评估伺服电机内轴承的健康状态。该方法将频域参考信号转换为伪时间序列模式，生成对齐的协变量和目标信号，并使用TSFM预测分类数据对应预定义标签的概率。利用预训练模型的扩展性，该方法在不同操作条件下都表现出有效性。这标志着从定制化狭义AI解决方案向更广泛的AI驱动维护系统迈出了重要进展。


### 论文摘要

This paper introduces a classification method using in-context learning in time-series foundation models (TSFM). We show how data, which was not part of the TSFM training data corpus, can be classified without the need of finetuning the model. Examples are represented in the form of targets (class id) and covariates (data matrix) within the prompt of the model, which enables to classify an unknown covariate data pattern alongside the forecast axis through in-context learning. We apply this method to vibration data for assessing the health state of a bearing within a servo-press motor. The method transforms frequency domain reference signals into pseudo time-series patterns, generates aligned covariate and target signals, and uses the TSFM to predict probabilities how classified data corresponds to predefined labels. Leveraging the scalability of pre-trained models this method demonstrates efficacy across varied operational conditions. This marks significant progress beyond custom narrow AI solutions towards broader, AI-driven maintenance systems.

---

## 147. CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search

**论文链接:** [http://arxiv.org/abs/2511.15443v1](http://arxiv.org/abs/2511.15443v1)

**作者:** Ao Xie, Jiahui Chen, Quanzhi Zhu, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li

**发布时间:** 2025-11-19

**备注:** AAAI-2026, Oral

### GPT解析

### 总结

本文提出了CroPS(Cross-Perspective Positive Samples)，一种新型检索数据引擎，旨在解决密集检索系统中的过滤气泡效应问题，通过引入多角度多样化正面样本来提升检索性能。

### 背景

密集检索已成为现代搜索系统的基础范式，特别是在短视频平台上。然而，工业系统通常采用自我强化的训练流程，依赖历史用户交互数据进行监督，导致过滤气泡效应。

### 目的

解决过滤气泡效应问题，通过引入多样化且语义上有意义的正面样本来缓解模型偏向狭窄和保守检索的问题。

### 方法

CroPS通过三个层面增强训练：用户查询重构行为(查询层面)、推荐流中的参与数据(系统层面)以及大语言模型合成的世界知识(知识层面)。同时引入分层标签分配(HLA)策略和H-InfoNCE损失函数来实现细粒度、相关性感知的优化。

### 主要发现

在快手搜索平台上的实验表明，CroPS在离线和实时A/B测试中都显著优于强基线，实现了更好的检索性能并减少了查询重构率。

### 结论

CroPS已在快手搜索全面部署，每日服务数亿用户，证明了其在实际工业环境中的有效性和实用性。

### 翻译

密集检索已成为现代搜索系统的基础范式，特别是在短视频平台上。然而，大多数工业系统采用一种自我强化的训练流程，依赖于历史上暴露的用户交互进行监督。这种范式不可避免地导致过滤气泡效应，即潜在相关但以前未见过的内容被排除在训练信号之外，使模型偏向于狭窄和保守的检索。在本文中，我们提出了CroPS(跨视角正面样本)，这是一种新型检索数据引擎，旨在通过从多个角度引入多样化且语义上有意义的正面样本来缓解此问题。CroPS通过用户查询重构行为(查询层面)、推荐流中的参与数据(系统层面)以及大型语言模型合成的世界知识(知识层面)来增强训练。为了有效利用这些异构信号，我们引入了分层标签分配(HLA)策略和相应的H-InfoNCE损失，它们共同实现了细粒度、相关性感知的优化。在快手搜索这一大规模商业短视频搜索平台上进行的广泛实验表明，CroPS在离线和实时A/B测试中都显著优于强基线，实现了卓越的检索性能并减少了查询重构率。CroPS现已全面部署在快手搜索中，每日服务数亿用户。


### 论文摘要

Dense retrieval has become a foundational paradigm in modern search systems, especially on short-video platforms. However, most industrial systems adopt a self-reinforcing training pipeline that relies on historically exposed user interactions for supervision. This paradigm inevitably leads to a filter bubble effect, where potentially relevant but previously unseen content is excluded from the training signal, biasing the model toward narrow and conservative retrieval. In this paper, we present CroPS (Cross-Perspective Positive Samples), a novel retrieval data engine designed to alleviate this problem by introducing diverse and semantically meaningful positive examples from multiple perspectives. CroPS enhances training with positive signals derived from user query reformulation behavior (query-level), engagement data in recommendation streams (system-level), and world knowledge synthesized by large language models (knowledge-level). To effectively utilize these heterogeneous signals, we introduce a Hierarchical Label Assignment (HLA) strategy and a corresponding H-InfoNCE loss that together enable fine-grained, relevance-aware optimization. Extensive experiments conducted on Kuaishou Search, a large-scale commercial short-video search platform, demonstrate that CroPS significantly outperforms strong baselines both offline and in live A/B tests, achieving superior retrieval performance and reducing query reformulation rates. CroPS is now fully deployed in Kuaishou Search, serving hundreds of millions of users daily.

---

## 148. HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation

**论文链接:** [http://arxiv.org/abs/2511.15435v1](http://arxiv.org/abs/2511.15435v1)

**作者:** Linyin Luo, Yujuan Ding, Yunshan Ma, Wenqi Fan, Hanjiang Lai

**发布时间:** 2025-11-19

### GPT解析

### 总结

这篇论文研究了一种针对多模态检索增强生成(MRAG)系统的视觉攻击方法，通过在用户输入图像中添加不可察觉的扰动来干扰系统，而不需要操作其他组件。

### 背景

多模态检索增强生成(MRAG)技术已被广泛应用于增强大型多模态模型(LMMs)的能力，但同时也带来了新的安全问题。现有的对抗性研究揭示了MRAG系统容易受到知识投毒攻击的脆弱性，这些攻击会欺骗检索器，使其召回被注入的毒害内容。

### 目的

研究一种新的攻击方式：仅通过在用户输入的图像中添加不可察觉的扰动来对MRAG系统进行视觉攻击，而不需要操作其他组件。

### 方法

提出了一种分层视觉攻击方法，通过使MRAG生成器的两个输入（多模态查询和增强知识）失准和混乱来干扰生成。设计了一个分层两阶段策略来获取失准的增强知识，通过优化扰动首先破坏跨模态对齐，然后破坏多模态语义对齐，来干扰检索器的图像输入，使其从原始数据库中检索无关知识。

### 主要发现

在OK-VQA和InfoSeek两个常用的MRAG数据集上进行了大量实验，使用基于CLIP的检索器和两个LMMs（BLIP-2和LLaVA）作为生成器。结果表明，这种视觉攻击对MRAG系统有效，显著降低了检索和生成性能。

### 结论

通过分层视觉攻击方法，可以有效地干扰MRAG系统，仅通过图像输入的不可察觉扰动就能显著降低系统性能。

### 翻译

先进的多模态检索增强生成(MRAG)技术已被广泛应用于增强大型多模态模型(LMMs)的能力，但同时也带来了新的安全问题。现有的对抗性研究揭示了MRAG系统容易受到知识投毒攻击的脆弱性，这些攻击会欺骗检索器，使其召回被注入的毒害内容。然而，我们的工作考虑了不同的场景：仅通过在用户输入的图像中添加不可察觉的扰动来对MRAG进行视觉攻击，而不需要操作任何其他组件。由于微调检索器和大规模生成器的鲁棒性，以及视觉扰动通过RAG链传播可能被进一步削弱，这具有挑战性。我们提出了一种新颖的分层视觉攻击方法，通过使MRAG生成器的两个输入（多模态查询和增强知识）失准和混乱来干扰其生成。我们进一步设计了一个分层两阶段策略来获取失准的增强知识。我们通过优化扰动来破坏检索器的图像输入，使其从原始数据库中检索无关知识，这些扰动首先破坏跨模态对齐，然后破坏多模态语义对齐。我们在两个广泛使用的MRAG数据集上进行了大量实验：OK-VQA和InfoSeek。我们使用基于CLIP的检索器和两个LMMs（BLIP-2和LLaVA）作为生成器。结果表明，我们的视觉攻击显著降低了检索和生成性能，证明了其对MRAG的有效性。


### 论文摘要

Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance.

---

## 149. Small Language Models for Phishing Website Detection: Cost, Performance, and Privacy Trade-Offs

**论文链接:** [http://arxiv.org/abs/2511.15434v1](http://arxiv.org/abs/2511.15434v1)

**作者:** Georg Goldenits, Philip Koenig, Sebastian Raubitzek, Andreas Ekelhart

**发布时间:** 2025-11-19

### GPT解析

### 总结

该论文研究了小型语言模型(SLMs)用于检测钓鱼网站的可行性，评估了不同规模SLMs在分类准确性、计算需求和成本效益方面的表现。

### 背景

钓鱼网站是主要网络安全威胁，传统机器学习方法需要大量特征工程、持续重新训练和昂贵基础设施维护。大型语言模型虽表现优异，但运营成本和对外部提供商的依赖限制了其在商业环境中的应用。

### 目的

研究仅使用原始HTML代码的小型语言模型检测钓鱼网站的可行性，评估其在本地基础设施部署的优势。

### 方法

系统评估了15种常用小型语言模型(参数范围从10亿到700亿)，基准测试了它们的分类准确性、计算需求和成本效益。

### 主要发现

SLMs在检测性能上不如最先进的专有大型语言模型，但仍是外部LLM服务的可行且可扩展的替代方案。研究强调了检测性能与资源消耗之间的权衡关系。

### 结论

通过成本和效益比较分析，该研究为未来在钓鱼检测系统中调整、微调和部署SLMs奠定了基础，旨在平衡安全有效性和经济实用性。

### 翻译

钓鱼网站构成主要网络安全威胁，利用不知情的用户并造成重大的财务和组织伤害。用于钓鱼检测的传统机器学习方法通常需要大量特征工程、持续重新训练和昂贵的基础设施维护。同时，专有大型语言模型在钓鱼相关分类任务中表现出色，但它们的运营成本和对外部提供商的依赖限制了它们在许多商业环境中的实际采用。本文研究了仅使用原始HTML代码的小型语言模型检测钓鱼网站的可行性。这些模型的一个关键优势是它们可以在本地基础设施上部署，为组织提供对数据和操作的更大控制。我们系统评估了15种常用小型语言模型(参数从10亿到700亿)，基准测试了它们的分类准确性、计算需求和成本效益。我们的结果强调了检测性能与资源消耗之间的权衡，表明虽然小型语言模型 compared to 最先进的专有大型语言模型表现不佳，但它们仍然可以成为外部大型语言模型服务的可行且可扩展的替代方案。通过呈现成本和效益的比较分析，这项工作为未来在钓鱼检测系统中调整、微调和部署小型语言模型的研究奠定了基础，旨在平衡安全有效性和经济实用性。


### 论文摘要

Phishing websites pose a major cybersecurity threat, exploiting unsuspecting users and causing significant financial and organisational harm. Traditional machine learning approaches for phishing detection often require extensive feature engineering, continuous retraining, and costly infrastructure maintenance. At the same time, proprietary large language models (LLMs) have demonstrated strong performance in phishing-related classification tasks, but their operational costs and reliance on external providers limit their practical adoption in many business environments. This paper investigates the feasibility of small language models (SLMs) for detecting phishing websites using only their raw HTML code. A key advantage of these models is that they can be deployed on local infrastructure, providing organisations with greater control over data and operations. We systematically evaluate 15 commonly used Small Language Models (SLMs), ranging from 1 billion to 70 billion parameters, benchmarking their classification accuracy, computational requirements, and cost-efficiency. Our results highlight the trade-offs between detection performance and resource consumption, demonstrating that while SLMs underperform compared to state-of-the-art proprietary LLMs, they can still provide a viable and scalable alternative to external LLM services. By presenting a comparative analysis of costs and benefits, this work lays the foundation for future research on the adaptation, fine-tuning, and deployment of SLMs in phishing detection systems, aiming to balance security effectiveness and economic practicality.

---

## 150. 论文ID: 2511.15404v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15404v1.json'

---

## 151. Breaking Expert Knowledge Limits: Self-Pruning for Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.15390v1](http://arxiv.org/abs/2511.15390v1)

**作者:** Haidong Kang, Lihong Lin, Enneng Yang, Hongning Dai, Hao Wang

**发布时间:** 2025-11-19

### GPT解析

### 总结

这篇论文介绍了AutoPrune，一种新型的大语言模型剪枝方法，利用LLMs自身设计最优剪枝算法，解决了现有方法依赖专家知识和在高剪枝率下性能下降的问题。

### 背景

大型语言模型在多种任务上取得了显著性能，但由于其巨大尺寸阻碍了实际部署。现有的针对LLMs的剪枝方法(如Wanda)严重依赖人工设计的剪枝算法，导致巨大的劳动成本和需要专家知识。此外，在高剪枝率下，由于均匀稀疏性导致的严重异常值问题会造成性能急剧下降。

### 目的

提出一种方法，使LLMs能够自行设计最优剪枝算法，无需专家知识，并解决在高剪枝率下的异常值问题，从而实现高效且性能保持的模型剪枝。

### 方法

提出了AutoPrune方法，主要包括：利用LLMs自身自动设计最优剪枝算法，无需专家知识；提出图驱动的思维链(GCoT)来优化提示，增强学习剪枝算法的推理过程；引入基于偏斜的动态稀疏分配(SDSA)来解决异常值问题，减轻在高剪枝率下的性能下降。

### 主要发现

首次识别了在高剪枝率下由均匀稀疏性引起的严重异常值问题；LLMs可以通过适当提示自行设计高性能剪枝算法；GCoT可以显著提升LLMs学习剪枝算法的推理过程；SDSA可以有效解决异常值问题，保持高剪枝率下的性能。

### 结论

AutoPrune在主流LLMs基准测试中表现出色，始终优于最先进的竞争对手，为LLMs的高效剪枝提供了新思路。

### 翻译

大型语言模型在广泛任务上取得了显著性能，但由于其巨大尺寸阻碍了实际部署。现有的针对LLMs的剪枝方法严重依赖人工设计的剪枝算法，导致巨大的劳动成本和需要专家知识。此外，我们首次识别了在高剪枝率下由均匀稀疏性引起的严重异常值问题，这引发了如何为LLMs设计自适应剪枝稀疏性的额外关注。LLMs能否自行剪枝？在这项工作中，我们通过提出一种名为AutoPrune的新型剪枝方法给出了肯定答案，该方法首先通过利用LLMs自动设计最优剪枝算法克服了专家知识的限制，无需任何专家知识。具体而言，为了缓解LLMs的黑盒特性，我们提出了图驱动的思维链来优化提示，显著增强了学习剪枝算法的推理过程，使我们能够生成具有卓越性能和可解释性的下一代剪枝算法。最后，基于对异常值问题的见解，我们引入了基于偏斜的动态稀疏分配来克服异常值问题，减轻在高剪枝率下的性能下降。我们在主流LLMs基准上进行了大量实验，证明了AutoPrune的优越性，它始终优于最先进的竞争对手。


### 论文摘要

Large language models (LLMs) have achieved remarkable performance on a wide range of tasks, hindering real-world deployment due to their massive size. Existing pruning methods (e.g., Wanda) tailored for LLMs rely heavily on manual design pruning algorithms, thereby leading to \textit{huge labor costs} and \textit{requires expert knowledge}. Furthermore, we are the first to identify the serious \textit{outlier value issue} behind dramatic performance degradation under high pruning ratios that are caused by uniform sparsity, raising an additional concern about how to design adaptive pruning sparsity ideal for LLMs. Can LLMs prune by themselves? In this work, we introduce an affirmative answer by proposing a novel pruning method called \textbf{AutoPrune}, which first overcomes expert knowledge limits by leveraging LLMs to design optimal pruning algorithms for themselves automatically without any expert knowledge. Specifically, to mitigate the black-box nature of LLMs, we propose a Graph-driven Chain-of-Thought (GCoT) to optimize prompts, significantly enhancing the reasoning process in learning the pruning algorithm and enabling us to generate pruning algorithms with superior performance and interpretability in the next generation. Finally, grounded in insights of outlier value issue, we introduce Skew-aware Dynamic Sparsity Allocation (SDSA) to overcome the outlier value issue, mitigating performance degradation under high pruning ratios. We conduct extensive experiments on mainstream LLMs benchmarks, demonstrating the superiority of AutoPrune, which consistently excels state-of-the-art competitors. The code is available at: https://anonymous.4open.science/r/AutoPrune.

---

## 152. 论文ID: 2511.15375v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.15375v1.json'

---

## 153. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs

**论文链接:** [http://arxiv.org/abs/2511.15316v1](http://arxiv.org/abs/2511.15316v1)

**作者:** Zhihan Ren, Lijun He, Jiaxi Liang, Xinzhu Fu, Haixia Bi, Fan Li

**发布时间:** 2025-11-19

### GPT解析

### 总结

FIA-Flow是一种黑盒特征反转攻击框架，可以从分割深度神经网络的中间特征实现高保真图像重建，揭示了比之前认识到的更严重的隐私威胁。

### 背景

Split DNNs允许边缘设备将密集计算卸载到云服务器，但这种范式存在隐私漏洞，因为中间特征可能被利用通过特征反转攻击(FIA)来重建私有输入。

### 目的

为了揭示泄露特征的隐私风险，提出一种能够实现高保真图像重建的黑盒FIA框架。

### 方法

设计了潜在特征空间对齐模块(LFSAM)来弥合中间特征空间和潜在空间之间的语义差距；开发了确定性反转流匹配(DIFM)来修正分布不匹配；基于大型视觉语言模型提出了两个指标来量化隐私泄露。

### 主要发现

FIA-Flow在各种模型和层上实现了更忠实和语义对齐的特征反转，揭示了Split DNNs中比之前认识到的更严重的隐私威胁。

### 结论

FIA-Flow通过解耦设计简化了学习过程，并允许使用少量图像-特征对进行有效训练，证明了Split DNNs存在严重的隐私风险。

### 翻译

分割深度神经网络通过将密集计算卸载到云服务器使边缘设备成为可能，但这种范式暴露了隐私漏洞，因为中间特征可以被利用通过特征反转攻击(FIA)来重建私有输入。现有的FIA方法通常产生有限的重建质量，难以评估隐私泄露的真实程度。为了揭示泄露特征的隐私风险，我们引入了FIA-Flow，一个黑盒FIA框架，可以从中间特征实现高保真图像重建。为了利用中间特征中的语义信息，我们设计了潜在特征空间对齐模块(LFSAM)来弥合中间特征空间和潜在空间之间的语义差距。此外，为了修正分布不匹配，我们开发了确定性反转流匹配(DIFM)，它将离流形特征投影到目标流形上，实现单步推理。这种解耦设计简化了学习过程，并允许使用少量图像-特征对进行有效训练。为了从人类角度量化隐私泄露，我们还基于大型视觉语言模型提出了两个指标。实验表明，FIA-Flow在各种模型(AlexNet, ResNet, Swin Transformer, DINO, 和 YOLO11)和层上实现了更忠实和语义对齐的特征反转，揭示了比之前认识到的更严重的Split DNNs隐私威胁。


### 论文摘要

Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.

---

## 154. Fast and Certified Bounding of Security-Constrained DCOPF via Interval Bound Propagation

**论文链接:** [http://arxiv.org/abs/2511.15624v1](http://arxiv.org/abs/2511.15624v1)

**作者:** Eren Tekeler, Xiangru Zhong, Huan Zhang, Samuel Chevalier

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究提出了一种基于GPU加速的神经网络验证工具区间边界传播(IBP)来解决大规模安全约束直流最优潮流(SC-DCOPF)问题，能够在极短时间内提供高质量的解边界，帮助系统运营商快速决策。

### 背景

安全约束直流最优潮流(SC-DCOPF)是输电系统运营商的重要工具，能够实现经济高效且物理安全的调度决策。然而，基于CPU的商业求解器在系统规模和事故数量增长到数千时，性能迅速下降，给寻求及时决策的系统运营商带来了计算瓶颈。

### 目的

设计一种计算图表示方法，利用GPU加速的神经网络验证工具快速解决大规模SC-DCOPF问题，为系统运营商提供高质量的解边界，帮助他们在广泛潜在威胁范围内及时做出决策。

### 方法

受第三届ARPA-E电网优化竞赛(GO3)启发，研究设计了基于SC-DCOPF的市场清算问题的计算图表示，并使用GPU加速的神经网络验证工具区间边界传播(IBP)来快速界定大规模SC-DCOPF问题的最优解。

### 主要发现

使用IBP计算认证边界，对多达617个母线实例的最大间隙为6.53%，同时在多达8,316个母线的挑战性系统上展示了可扩展性，运行时间约为0.07秒。

### 结论

IBP能够以极快速度提供高质量的解边界，并有助于识别具有挑战性的SC-DCOPF实例中的不可行性驱动因素，为系统运营商提供及时决策支持。

### 翻译

安全约束直流最优潮流(SC-DCOPF)是输电系统运营商的重要工具，能够实现经济高效且物理安全的调度决策。尽管基于CPU的商业求解器(如Gurobi)可以有效解决具有合理数量安全约束的SC-DCOPF问题，但随着系统规模和事故数量增长到数千，它们的性能迅速下降，导致显著的计算负担。这为寻求在广泛潜在威胁范围内及时决策的系统运营商带来了瓶颈。在本文中，我们受第三届ARPA-E电网优化竞赛(GO3)启发，设计了基于SC-DCOPF的市场清算问题的计算图表示。我们能够使用称为区间边界传播(IBP)的GPU加速神经网络验证工具，快速大规模SC-DCOPF问题的最优解边界。使用IBP，我们计算了认证边界，对多达617个母线实例的最大间隙为6.53%，同时在多达8,316个母线的挑战性系统上展示了可扩展性，运行时间约为0.07秒。这些结果表明，IBP能够以极快速度提供高质量的解边界，并可以帮助识别具有挑战性的SC-DCOPF实例中的不可行性驱动因素。


### 论文摘要

Security-Constrained DC Optimal Power Flow (SC DCOPF) is an important tool for transmission system operators, enabling economically efficient and physically secure dispatch decisions. Although CPU-based commercial solvers (e.g., Gurobi) can efficiently solve SC-DCOPF problems with a reasonable number of security constraints, their performance degrades rapidly as both system size and the number of contingencies grow into thousands, leading to a significant computational burden. This introduces a bottleneck for system operators who seek timely decision-making across a wide range of potential threats. In this paper, we design a computational graph representation of the SC-DCOPF-based market-clearing problem, inspired by the third ARPA-E Grid Optimization Competition (GO3). We are able to quickly bound the optimal solution of large-scale SC-DCOPF problems using a GPU-accelerated Neural Network verification tool called Interval Bound Propagation (IBP). Using IBP, we compute certified bounds with a maximum gap of 6.53% for instances up to 617 buses, while demonstrating scalability on challenging systems up to 8,316 buses with a runtime of approximately 0.07 seconds. These results demonstrate that IBP can provide high-quality solution bounds at very fast speeds, and it can help identify infeasibility drivers in challenging SC-DCOPF instances.

---

## 155. Platform-Agnostic Reinforcement Learning Framework for Safe Exploration of Cluttered Environments with Graph Attention

**论文链接:** [http://arxiv.org/abs/2511.15358v1](http://arxiv.org/abs/2511.15358v1)

**作者:** Gabriele Calzolari, Vidya Sumathy, Christoforos Kanellakis, George Nikolakopoulos

**发布时间:** 2025-11-19

**备注:** 8 pages, 6 figures, submitted to the 2026 IEEE International Conference on Robotics & Automation

### GPT解析

### 总结

该研究提出了一种新的平台无关的强化学习框架，用于在障碍丰富的空间中进行自主探索，结合图神经网络政策和安全过滤器，确保高效且安全的探索。

### 背景

自主探索障碍丰富空间需要确保效率和安全的策略，以避免与障碍物碰撞。

### 目的

研究一种新的平台无关的强化学习框架，结合基于图神经网络的政策进行下一个航点选择，并确保安全移动。

### 方法

使用近端策略优化(PPO)算法通过强化训练神经网络，最大化探索效率同时最小化安全过滤器的干预；当政策提出不可行的行动时，安全过滤器会覆盖为最接近的可行替代方案；引入由势场塑造的奖励函数，考虑代理接近未探索区域和到达这些区域时的预期信息增益；结合基于强化学习的探索政策的适应性和显式安全机制提供的可靠性。

### 主要发现

在模拟和实验室环境中的大量评估表明，该方法能够在杂乱空间中实现高效且安全的探索。

### 结论

该框架使基于学习的政策能够在真实世界环境中运行的机器人平台上部署。

### 翻译

障碍丰富空间的自主探索需要确保效率同时保证避免与障碍物碰撞的安全策略。本文研究了一种新颖的平台无关强化学习框架，整合了基于图神经网络的下一个航点选择策略，以及确保安全移动的安全过滤器。具体而言，神经网络通过近端策略优化(PPO)算法使用强化学习进行训练，以最大化探索效率同时最小化安全过滤器的干预。因此，当策略提出不可行的行动时，安全过滤器会将其覆盖为最接近的可行替代方案，确保系统行为的一致性。此外，本文引入了一种由势场塑造的奖励函数，该函数考虑了代理接近未探索区域以及到达这些区域时的预期信息增益。所提出的框架结合了基于强化学习的探索政策的适应性和显式安全机制提供的可靠性。这一特性在使基于学习的政策能够在真实世界环境中运行的机器人平台上部署方面起着关键作用。在模拟和实验室环境中进行的广泛评估表明，该方法在杂乱空间中实现了高效且安全的探索。


### 论文摘要

Autonomous exploration of obstacle-rich spaces requires strategies that ensure efficiency while guaranteeing safety against collisions with obstacles. This paper investigates a novel platform-agnostic reinforcement learning framework that integrates a graph neural network-based policy for next-waypoint selection, with a safety filter ensuring safe mobility. Specifically, the neural network is trained using reinforcement learning through the Proximal Policy Optimization (PPO) algorithm to maximize exploration efficiency while minimizing safety filter interventions. Henceforth, when the policy proposes an infeasible action, the safety filter overrides it with the closest feasible alternative, ensuring consistent system behavior. In addition, this paper introduces a reward function shaped by a potential field that accounts for both the agent's proximity to unexplored regions and the expected information gain from reaching them. The proposed framework combines the adaptability of reinforcement learning-based exploration policies with the reliability provided by explicit safety mechanisms. This feature plays a key role in enabling the deployment of learning-based policies on robotic platforms operating in real-world environments. Extensive evaluations in both simulations and experiments performed in a lab environment demonstrate that the approach achieves efficient and safe exploration in cluttered spaces.

---

## 156. LaguerreNet: Advancing a Unified Solution for Heterophily and Over-smoothing with Adaptive Continuous Polynomials

**论文链接:** [http://arxiv.org/abs/2511.15328v1](http://arxiv.org/abs/2511.15328v1)

**作者:** Huseyin Goksu

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种基于连续拉盖尔多项式的新型GNN滤波器LaguerreNet，解决了谱图神经网络在异构图上的表现不佳和过平滑问题。

### 背景

谱图神经网络(GNNs)存在两个关键限制：在'异构性'图上表现不佳，以及在较高多项式阶数(K)时出现性能崩溃(过平滑)。这些问题源于标准滤波器(如ChebyNet)的静态、低通特性。

### 目的

提出一种新的GNN滤波器，解决异构图上的表现问题和过平滑问题。

### 方法

提出LaguerreNet，一种基于连续拉盖尔多项式的新型GNN滤波器。通过使其核心alpha参数可训练，学习滤波器的频谱形状，并使用基于LayerNorm的稳定化技术解决了这些无界多项式的严重数值不稳定问题。

### 主要发现

1) LaguerreNet在具有挑战性的异构性基准测试上取得了最先进的结果；2) 它对过平滑具有极强的鲁棒性，性能在K=10时达到峰值，比ChebyNet崩溃时的K值高一个数量级。

### 结论

LaguerreNet通过基于连续拉盖尔多项式的自适应多项式滤波器，成功解决了GNN在异构图上的表现问题和过平滑问题。

### 翻译

谱图神经网络(GNNs)遭受两个关键限制：在'异构性'图上表现不佳以及在较高多项式阶数(K)时出现性能崩溃，即过平滑。这两个问题都源于标准滤波器(如ChebyNet)的静态、低通特性。虽然自适应多项式滤波器(如离散MeixnerNet)已成为潜在的统一解决方案，但它们向连续域的扩展以及具有无界系数时的稳定性仍然是开放性问题。在这项工作中，我们提出了'LaguerreNet'，一种基于连续拉盖尔多项式的新型GNN滤波器。LaguerreNet通过使其核心alpha参数可训练来学习滤波器的频谱形状，从而推进了自适应多项式方法。我们使用基于LayerNorm的稳定化技术解决了这些无界多项式的严重数值不稳定问题。我们通过实验证明这种方法非常有效：1) LaguerreNet在具有挑战性的异构性基准测试上取得了最先进的结果；2) 它对过平滑具有极强的鲁棒性，性能在K=10时达到峰值，比ChebyNet崩溃时的K值高一个数量级。


### 论文摘要

Spectral Graph Neural Networks (GNNs) suffer from two critical limitations: poor performance on "heterophilic" graphs and performance collapse at high polynomial degrees (K), known as over-smoothing. Both issues stem from the static, low-pass nature of standard filters (e.g., ChebyNet). While adaptive polynomial filters, such as the discrete MeixnerNet, have emerged as a potential unified solution, their extension to the continuous domain and stability with unbounded coefficients remain open questions. In this work, we propose `LaguerreNet`, a novel GNN filter based on continuous Laguerre polynomials. `LaguerreNet` learns the filter's spectral shape by making its core alpha parameter trainable, thereby advancing the adaptive polynomial approach. We solve the severe O(k^2) numerical instability of these unbounded polynomials using a `LayerNorm`-based stabilization technique. We demonstrate experimentally that this approach is highly effective: 1) `LaguerreNet` achieves state-of-the-art results on challenging heterophilic benchmarks. 2) It is exceptionally robust to over-smoothing, with performance peaking at K=10, an order of magnitude beyond where ChebyNet collapses.

---

## 157. KrawtchoukNet: A Unified GNN Solution for Heterophily and Over-smoothing with Adaptive Bounded Polynomials

**论文链接:** [http://arxiv.org/abs/2511.15327v1](http://arxiv.org/abs/2511.15327v1)

**作者:** Huseyin Goksu

**发布时间:** 2025-11-19

### GPT解析

### 总结

KrawtchoukNet是一种基于离散Krawtchouk多项式的图神经网络滤波器，解决了传统谱GNN在异质性图上性能崩溃和高多项式阶数时过平滑的问题。

### 背景

基于多项式滤波器的谱图神经网络（如ChebyNet）存在两个关键限制：在'异质性'图上性能崩溃，以及在高多项式阶数时出现性能崩溃（过平滑问题），这两个问题都源于标准滤波器的静态、低通特性。

### 目的

提出KrawtchoukNet，一种基于离散Krawtchouk多项式的GNN滤波器，通过两个关键设计选择为这两个问题提供统一解决方案。

### 方法

通过将多项式的域N固定为小常数（如N=20），创建第一个递归系数固有有界的GNN滤波器；使滤波器的形状参数p可学习，使滤波器适应图数据的频谱响应。

### 主要发现

KrawtchoukNet对过平滑问题具有极强的鲁棒性（在K=10时达到SOTA结果）；其自适应性质使其在具有挑战性的异质性基准（Texas、Cornell）上达到SOTA性能；明显优于标准的GNN，如GAT和APPNP。

### 结论

KrawtchoukNet解决了现有谱GNN的两个关键问题，通过创新的设计选择提供了统一的解决方案。

### 翻译

基于多项式滤波器的谱图神经网络（如ChebyNet）存在两个关键限制：1）在'异质性'图上性能崩溃；2）在高多项式阶数(K)时出现性能崩溃，称为过平滑。这两个问题都源于标准滤波器的静态、低通特性。在这项工作中，我们提出了KrawtchoukNet，一种基于离散Krawtchouk多项式的GNN滤波器。我们通过两个关键设计选择证明KrawtchoukNet为这两个问题提供了统一解决方案。首先，通过将多项式的域N固定为小常数（如N=20），我们创建了第一个递归系数固有有界的GNN滤波器，使其对过平滑具有极强的鲁棒性（在K=10时达到SOTA结果）。其次，通过使滤波器的形状参数p可学习，滤波器适应其频谱响应到图数据。我们表明这种自适应性质使KrawtchoukNet在具有挑战性的异质性基准（Texas、Cornell）上达到SOTA性能，明显优于标准的GNN如GAT和APPNP。


### 论文摘要

Spectral Graph Neural Networks (GNNs) based on polynomial filters, such as ChebyNet, suffer from two critical limitations: 1) performance collapse on "heterophilic" graphs and 2) performance collapse at high polynomial degrees (K), known as over-smoothing. Both issues stem from the static, low-pass nature of standard filters. In this work, we propose `KrawtchoukNet`, a GNN filter based on the discrete Krawtchouk polynomials. We demonstrate that `KrawtchoukNet` provides a unified solution to both problems through two key design choices. First, by fixing the polynomial's domain N to a small constant (e.g., N=20), we create the first GNN filter whose recurrence coefficients are \textit{inherently bounded}, making it exceptionally robust to over-smoothing (achieving SOTA results at K=10). Second, by making the filter's shape parameter p learnable, the filter adapts its spectral response to the graph data. We show this adaptive nature allows `KrawtchoukNet` to achieve SOTA performance on challenging heterophilic benchmarks (Texas, Cornell), decisively outperforming standard GNNs like GAT and APPNP.

---

## 158. D2D Power Allocation via Quantum Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2511.15246v1](http://arxiv.org/abs/2511.15246v1)

**作者:** Tung Giang Le, Xuan Tung Nguyen, Won-Joo Hwang

**发布时间:** 2025-11-19

**DOI:** 10.23919/ICMU65253.2025.11219153

### GPT解析

### 总结

本文提出了一种全量子的图神经网络(QGNN)，通过参数化量子电路实现消息传递，在无线网络资源管理领域实现了与经典方法相当的性能但参数更少且具有固有并行性。

### 背景

无线网络日益复杂，需要可扩展的资源管理方法。传统的图神经网络在图学习方面表现出色，但在大规模设置中计算成本很高。

### 目的

开发一种全量子的图神经网络，用于解决大规模无线网络中的资源管理问题。

### 方法

通过参数化量子电路(PQCs)实现消息传递。量子图卷积层(QGCLs)将特征编码为量子状态，用NISQ兼容的幺正运算处理图，并通过测量检索嵌入。

### 主要发现

该QGNN在D2D功率控制和SINR最大化问题上，使用更少的参数且具有固有并行性，能够达到与经典方法相当的性能。

### 结论

这种基于端到端PQC的GNN标志着量子加速无线优化的进步。

### 翻译

日益增长的无线网络复杂性需要可扩展的资源管理。经典图神经网络在图学习方面表现出色，但在大规模设置中会带来高昂的计算成本。我们提出了一种全量子图神经网络(QGNN)，通过参数化量子电路(PQCs)实现消息传递。我们的量子图卷积层(QGCLs)将特征编码为量子状态，用NISQ兼容的幺正运算处理图，并通过测量检索嵌入。应用于D2D功率控制和SINR最大化，我们的QGNN使用更少的参数和固有的并行性，达到了与经典方法相当的性能。这种基于端到端PQC的GNN标志着量子加速无线优化的进步。


### 论文摘要

Increasing wireless network complexity demands scalable resource management. Classical GNNs excel at graph learning but incur high computational costs in large-scale settings. We present a fully quantum Graph Neural Network (QGNN) that implements message passing via Parameterized Quantum Circuits (PQCs). Our Quantum Graph Convolutional Layers (QGCLs) encode features into quantum states, process graphs with NISQ-compatible unitaries, and retrieve embeddings through measurement. Applied to D2D power control for SINR maximization, our QGNN matches classical performance with fewer parameters and inherent parallelism. This end-to-end PQC-based GNN marks a step toward quantum-accelerated wireless optimization.

---

## 159. Why Physics Still Matters: Improving Machine Learning Prediction of Material Properties with Phonon-Informed Datasets

**论文链接:** [http://arxiv.org/abs/2511.15222v1](http://arxiv.org/abs/2511.15222v1)

**作者:** Pol Benítez, Cibrán López, Edgardo Saucedo, Teruyasu Mizoguchi, Claudio Cazorla

**发布时间:** 2025-11-19

**备注:** 12 pages; 5 figures

### GPT解析

### 总结

这篇论文研究了机器学习在材料科学中的应用，特别关注了训练数据集的构建方式对模型性能的影响。研究比较了基于随机生成原子构型和基于物理信息采样(晶格振动)构建的数据集上训练的图神经网络模型，发现在预测材料电子和机械特性时，物理信息指导的数据生成方法比随机方法更有效，即使使用更少的数据点也能获得更好的性能。

### 背景

机器学习方法已成为预测材料特性的强大工具，具有接近第一性原理的精度和大幅降低的计算成本。然而，ML模型的性能高度依赖于训练数据集的质量、大小和多样性。在材料科学中，低对称性原子构型能捕捉热激发、结构缺陷和化学无序等现实材料中普遍存在的特征，但这些特征在大多数数据集中代表性不足。缺乏生成代表性训练数据的系统性策略可能限制ML模型在能源转换和光子学等技术关键领域的预测能力。

### 目的

评估两种不同类型数据集上训练的图神经网络模型的有效性：一种由随机生成的原子构型组成，另一种基于晶格振动使用物理信息采样构建。研究旨在确定哪种数据生成方法能更有效地训练模型来预测材料在有限温度条件下的电子和机械特性。

### 方法

研究使用图神经网络模型作为案例研究，针对典型光电材料族在现实有限温度条件下预测电子和机械特性的任务。比较了基于随机生成原子构型的数据集和基于晶格振动物理信息采样构建的数据集上训练的模型性能。还进行了可解释性分析，以了解模型如何分配权重给不同的化学键。

### 主要发现

尽管使用的数据点更少，但基于声学信息(物理信息采样)的模型始终优于随机训练的对应模型。可解释性分析显示，高性能模型对控制特性变化的化学上有意义的键赋予更大的权重，这强调了物理指导数据生成的重要性。

### 结论

更大的数据集不一定能产生更好的GNN预测模型。研究引入了一种简单通用的策略，用于在材料信息学中高效构建高质量训练数据，这种方法基于物理信息采样而非随机生成。

### 翻译

机器学习(ML)方法已成为预测材料特性的强大工具，具有接近第一性原理的精度和大幅降低的计算成本。然而，ML模型的性能严重依赖于训练数据集的质量、大小和多样性。在材料科学中，这种依赖性对于从低对称性原子构型中学习尤为重要，这些构型能捕捉热激发、结构缺陷和化学无序等特征，这些特征在现实材料中普遍存在但在大多数数据集中代表性不足。因此，缺乏生成代表性训练数据的系统性策略可能限制ML模型在能源转换和光子学等技术关键领域的预测能力。在本工作中，我们评估了在两种 fundamentally different 类型的数据集上训练的图神经网络(GNN)模型的有效性：一种由随机生成的原子构型组成，另一种基于晶格振动使用物理信息采样构建。作为案例研究，我们解决了在现实有限温度条件下预测典型光电材料族电子和机械特性的具有挑战性的任务。我们发现，尽管使用的数据点更少，但基于声学信息的模型始终优于随机训练的对应模型。可解释性分析进一步揭示，高性能模型对控制特性变化的化学上有意义的键赋予更大的权重，强调了物理指导数据生成的重要性。总体而言，这项工作表明更大的数据集不一定能产生更好的GNN预测模型，并引入了一种简单通用的策略，用于在材料信息学中高效构建高质量训练数据。


### 论文摘要

Machine learning (ML) methods have become powerful tools for predicting material properties with near first-principles accuracy and vastly reduced computational cost. However, the performance of ML models critically depends on the quality, size, and diversity of the training dataset. In materials science, this dependence is particularly important for learning from low-symmetry atomistic configurations that capture thermal excitations, structural defects, and chemical disorder, features that are ubiquitous in real materials but underrepresented in most datasets. The absence of systematic strategies for generating representative training data may therefore limit the predictive power of ML models in technologically critical fields such as energy conversion and photonics. In this work, we assess the effectiveness of graph neural network (GNN) models trained on two fundamentally different types of datasets: one composed of randomly generated atomic configurations and another constructed using physically informed sampling based on lattice vibrations. As a case study, we address the challenging task of predicting electronic and mechanical properties of a prototypical family of optoelectronic materials under realistic finite-temperature conditions. We find that the phonons-informed model consistently outperforms the randomly trained counterpart, despite relying on fewer data points. Explainability analyses further reveal that high-performing models assign greater weight to chemically meaningful bonds that control property variations, underscoring the importance of physically guided data generation. Overall, this work demonstrates that larger datasets do not necessarily yield better GNN predictive models and introduces a simple and general strategy for efficiently constructing high-quality training data in materials informatics.

---

## 160. Vehicle Routing Problems via Quantum Graph Attention Network Deep Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.15175v1](http://arxiv.org/abs/2511.15175v1)

**作者:** Le Tung Giang, Vu Hoang Viet, Nguyen Xuan Tung, Trinh Van Chien, Won-Joo Hwang

**发布时间:** 2025-11-19

**备注:** 11 pages, 3 figures, 2 tables. Accepted by SOICT 2025

### GPT解析

### 总结

本文提出了一种量子图注意力网络(Q-GAT)用于解决车辆路径问题(VRP)，通过参数化量子电路替代传统多层感知器，减少了超过50%的可训练参数，同时提高了路由效率并降低了约5%的成本。

### 背景

车辆路径问题(VRP)是智能交通系统中的基础NP难问题，在物流和配送领域有广泛应用。深度强化学习(DRL)与图神经网络(GNN)结合显示出了潜力，但经典模型依赖大型多层感知器，存在参数量大和内存受限的问题。

### 目的

开发一种更高效的求解方法，减少参数量同时提高VRP的求解性能。

### 方法

提出量子图注意力网络(Q-GAT)，将其集成在深度强化学习框架中，使用参数化量子电路(PQCs)替代传统多层感知器，采用近端策略优化(PPO)算法，结合贪心解码和随机解码进行训练和推理。

### 主要发现

混合模型保持了图注意力编码器的表达能力，同时减少可训练参数超过50%；在VRP基准测试中，Q-GAT比经典GAT基线实现更快的收敛速度，路由成本降低约5%。

### 结论

PQC增强的图神经网络可作为紧凑有效的求解器，在大规模路由和物流优化方面具有应用潜力。

### 翻译

车辆路径问题(VRP)是智能交通系统中的一个基础NP难任务，在物流和配送领域有广泛应用。深度强化学习(DRL)结合图神经网络(GNN)已显示出潜力，但经典模型依赖大型多层感知器(MLPs)，这些模型参数量大且内存受限。我们在DRL框架内提出了量子图注意力网络(Q-GAT)，其中参数化量子电路(PQCs)在关键读取阶段替代了传统的MLPs。混合模型保持了图注意力编码器的表达能力，同时将可训练参数减少了50%以上。使用近端策略优化(PPO)结合贪心解码和随机解码，在VRP基准实验中，Q-GAT比经典GAT基线实现更快的收敛速度，并将路由成本降低了约5%。这些结果表明，PQC增强的GNNs作为紧凑有效的求解器，在大规模路由和物流优化方面具有潜力。


### 论文摘要

The vehicle routing problem (VRP) is a fundamental NP-hard task in intelligent transportation systems with broad applications in logistics and distribution. Deep reinforcement learning (DRL) with Graph Neural Networks (GNNs) has shown promise, yet classical models rely on large multi-layer perceptrons (MLPs) that are parameter-heavy and memory-bound. We propose a Quantum Graph Attention Network (Q-GAT) within a DRL framework, where parameterized quantum circuits (PQCs) replace conventional MLPs at critical readout stages. The hybrid model maintains the expressive capacity of graph attention encoders while reducing trainable parameters by more than 50%. Using proximal policy optimization (PPO) with greedy and stochastic decoding, experiments on VRP benchmarks show that Q-GAT achieves faster convergence and reduces routing cost by about 5% compared with classical GAT baselines. These results demonstrate the potential of PQC-enhanced GNNs as compact and effective solvers for large-scale routing and logistics optimization.

---

## 161. Integrating Causal Inference with Graph Neural Networks for Alzheimer's Disease Analysis

**论文链接:** [http://arxiv.org/abs/2511.14922v1](http://arxiv.org/abs/2511.14922v1)

**作者:** Pranay Kumar Peddi, Dhrubajyoti Ghosh

**发布时间:** 2025-11-18

### GPT解析

### 总结

Causal-GCN是一种创新的干预性图卷积框架，能够识别对阿尔茨海默病进展产生稳定因果影响的脑区，同时保持与现有模型相当的分类性能。

### 背景

深度图学习在阿尔茨海默病(AD)的MRI图像分类方面取得了进展，但大多数模型仅能发现相关性，无法区分人口统计和遗传因素与疾病特定特征之间的因果关系。

### 目的

开发一个能够识别对AD进展有因果影响的脑区的框架，而不仅仅是发现相关性，从而提高AD诊断的可解释性。

### 方法

将每个受试者的MRI表示为结构连接组，节点代表脑区，边表示解剖连接；通过主成分分析处理混杂因素（年龄、性别、APOE4基因型）；训练后模拟对单个区域的干预，通过改变传入边和节点特征来估计因果效应。

### 主要发现

在ADNI队列的484名受试者上测试，Causal-GCN实现了与基线GNN相当的性能；提供了可解释的因果效应排名，突出了后部、扣带回和脑岛等枢纽区域，这与已知的AD神经病理学一致。

### 结论

Causal-GCN不仅能够有效分类AD，还能提供对疾病机制的因果见解，有助于理解AD的病理发展过程。

### 翻译

深度图学习已推进了阿尔茨海默病(AD)从MRI图像的分类，但大多数模型仍然保持相关性，将人口统计和遗传因素与疾病特定特征混淆。我们提出了Causal-GCN，一种基于do-calculus的反向调整的干预性图卷积框架，用于识别对AD进展产生稳定因果影响的脑区。每个受试者的MRI被表示为结构连接组，其中节点代表皮层和皮层下区域，边编码解剖连接性。混杂因素如年龄、性别和APOE4基因型通过主成分分析总结并包含在因果调整集合中。训练后，通过对单个区域进行干预（通过改变其传入边和节点特征）来模拟，以估计疾病概率的平均因果效应。应用于ADNI队列的484名受试者，Causal-GCN实现了与基线GNN相当的性能，同时提供可解释的因果效应排名，突出了与已建立的AD神经病理学一致的后部、扣带回和脑岛枢纽。


### 论文摘要

Deep graph learning has advanced Alzheimer's (AD) disease classification from MRI, but most models remain correlational, confounding demographic and genetic factors with disease specific features. We present Causal-GCN, an interventional graph convolutional framework that integrates do-calculus-based back-door adjustment to identify brain regions exerting stable causal influence on AD progression. Each subject's MRI is represented as a structural connectome where nodes denote cortical and subcortical regions and edges encode anatomical connectivity. Confounders such as age, sec, and APOE4 genotype are summarized via principal components and included in the causal adjustment set. After training, interventions on individual regions are simulated by serving their incoming edges and altering node features to estimate average causal effects on disease probability. Applied to 484 subjects from the ADNI cohort, Causal-GCN achieves performance comparable to baseline GNNs while providing interpretable causal effect rankings that highlight posterior, cingulate, and insular hubs consistent with established AD neuropathology.

---

