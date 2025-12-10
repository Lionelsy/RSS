# 今日论文推荐 - 2025-12-10

共 58 篇论文

---

## 1. PAGE-4D: Disentangled Pose and Geometry Estimation for VGGT-4D Perception

**论文链接:** [http://arxiv.org/abs/2510.17568v3](http://arxiv.org/abs/2510.17568v3)

**作者:** Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, Mengyu Wang

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究提出了PAGE-4D模型，一种扩展VGGT以处理动态场景的前馈模型，实现了相机姿态估计、深度预测和点云重建功能，无需后处理。

### 背景

现有的3D前馈模型（如VGGT）在静态场景中表现出色，但在涉及移动人类或可变形物体等复杂动态元素的真实世界场景中表现不佳。

### 目的

开发一种能够处理动态场景的前馈模型，实现相机姿态估计、深度预测和点云重建，无需后处理。

### 方法

提出了PAGE-4D模型，并引入了一个动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息，为姿态估计抑制运动线索，同时为几何重建放大这些线索。

### 主要发现

PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点图重建方面取得了更好的结果。

### 结论

PAGE-4D成功解决了多任务4D重建中的内在冲突，能够在动态场景中实现准确的相机姿态估计和几何重建。

### 翻译

最近的3D前馈模型，如视觉几何基础Transformer（VGGT），在推断静态场景的3D属性方面表现出强大的能力。然而，由于它们通常在静态数据集上训练，这些模型在涉及复杂动态元素（如移动的人或像伞这样的可变形物体）的真实世界场景中往往表现不佳。为了解决这一局限性，我们引入了PAGE-4D，一种将VGGT扩展到动态场景的前馈模型，能够实现相机姿态估计、深度预测和点云重建——所有这些都无需后处理。多任务4D重建中的一个核心挑战是任务之间的固有冲突：准确的相机姿态估计需要抑制动态区域，而几何重建需要建模这些区域。为了解决这种张力，我们提出了一个动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息——为姿态估计抑制运动线索，同时为几何重建放大这些线索。大量实验表明，PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点云图重建方面取得了优越的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何将现有的静态场景3D重建模型扩展到动态场景中的问题。在现实世界中，大多数场景都包含移动的人和物体，静态模型无法处理这些动态内容，而现有处理动态场景的方法要么计算效率低，要么需要大量标注数据，难以实际应用。这个问题对于自动驾驶、机器人导航、增强现实等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到VGGT在静态场景中表现良好但在动态场景中性能下降，通过可视化分析发现VGGT倾向于忽略动态内容。他们识别出动态场景中的核心矛盾：运动信息对几何估计有价值，但同时会破坏相机姿态估计中的静态极线约束。基于这一洞察，他们设计了解耦动态信息在不同任务中作用的方案，借鉴了VGGT的基础架构、DINO风格编码器、Transformer架构和多任务损失函数，但扩展了聚合器部分并引入了动态感知机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是解耦动态信息在不同任务中的作用：对相机姿态估计任务抑制动态区域影响，对几何重建任务则利用动态区域的运动信息。整体流程包括：1)输入RGB帧序列；2)使用DINO风格编码器提取图像特征；3)三阶段动态感知聚合(含动态掩码预测)；4)任务特定应用动态掩码(姿态估计抑制动态，几何重建保留动态)；5)轻量级解码器输出深度和点云，更大解码器输出相机姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)PAGE-4D动态感知框架，扩展VGGT到动态场景；2)动态感知聚合器，结合掩码预测和选择性注意力机制；3)针对性微调策略，只调整对动态敏感的层。相比之前工作，PAGE-4D无需将问题分解为多个子模块，避免了计算成本增加和误差累积；不需要大规模标注数据；保持前馈方法通用性；只需微调部分参数而非整个网络，提高了效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAGE-4D通过解耦动态信息在不同3D重建任务中的作用，实现了高效的前馈式动态场景理解，在保持计算效率的同时显著提升了相机姿态估计和几何重建的准确性。'}


### 论文摘要

Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes. However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas. To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, and point cloud reconstruction -- all without post-processing. A central challenge in multi-task 4D reconstruction is the inherent conflict between tasks: accurate camera pose estimation requires suppressing dynamic regions, while geometry reconstruction requires modeling them. To resolve this tension, we propose a dynamics-aware aggregator that disentangles static and dynamic information by predicting a dynamics-aware mask -- suppressing motion cues for pose estimation while amplifying them for geometry reconstruction. Extensive experiments show that PAGE-4D consistently outperforms the original VGGT in dynamic scenarios, achieving superior results in camera pose estimation, monocular and video depth estimation, and dense point map reconstruction.

---

## 2. InfiniteVL: Synergizing Linear and Sparse Attention for Highly-Efficient, Unlimited-Input Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.08829v1](http://arxiv.org/abs/2512.08829v1)

**作者:** Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, Xinggang Wang

**发布时间:** 2025-12-09

**备注:** 16 pages, 8 figures, conference or other essential info

### GPT解析

### 总结

InfiniteVL是一种结合滑动窗口注意力和Gated DeltaNet的线性复杂度视觉语言模型架构，通过三阶段训练策略，使用极少训练数据实现了高性能与高效率的平衡。

### 背景

Window attention和linear attention是减轻视觉语言模型中二次复杂度和KV缓存问题的主要策略，但window-based VLMs在序列长度超过窗口大小时性能下降，linear attention在OCR和文档理解等信息密集型任务上表现不佳。

### 目的

克服现有方法的局限性，提出一种既能保持长期记忆又能高效处理的线性复杂度VLM架构。

### 方法

提出InfiniteVL架构，结合滑动窗口注意力(SWA)与Gated DeltaNet；设计三阶段训练策略：蒸馏预训练、指令调优和长序列SFT；使用不到领先VLMs所需训练数据的2%。

### 主要发现

InfiniteVL显著优于之前的线性复杂度VLMs，性能与基于Transformer的领先VLMs相匹配；实现了超过3.6倍的推理加速；在流视频理解场景中保持24 FPS的实时预填充速度，同时保留长期记忆缓存。

### 结论

InfiniteVL在保持与领先模型相当性能的同时，大幅提升了推理速度和内存效率，是一种高效实用的视觉语言模型架构。

### 翻译

窗口注意力和线性注意力代表了减轻视觉语言模型中二次复杂度和不断增长的KV缓存的两种主要策略。然而，我们观察到当序列长度超过窗口大小时，基于窗口的VLMs性能下降，而线性注意力在OCR和文档理解等信息密集型任务上表现不佳。为了克服这些局限性，我们提出了InfiniteVL，这是一种结合滑动窗口注意力(SWA)和Gated DeltaNet的线性复杂度VLM架构。为了在有限资源下实现具有竞争力的多模态性能，我们设计了一个包含蒸馏预训练、指令调优和长序列SFT的三阶段训练策略。值得注意的是，InfiniteVL使用的训练数据不到领先VLMs所需的2%，不仅显著优于之前的线性复杂度VLMs，而且性能与基于Transformer的领先VLMs相匹配，同时展示了有效的长期记忆保留能力。与使用FlashAttention-2加速的类似大小基于Transformer的VLMs相比，InfiniteVL实现了超过3.6倍的推理加速，同时保持恒定的延迟和内存占用。在流视频理解场景中，它保持稳定的24 FPS实时预填充速度，同时保留长期内存缓存。代码和模型可在https://github.com/hustvl/InfiniteVL获取。


### 论文摘要

Window attention and linear attention represent two principal strategies for mitigating the quadratic complexity and ever-growing KV cache in Vision-Language Models (VLMs). However, we observe that window-based VLMs suffer performance degradation when sequence length exceeds the window size, while linear attention underperforms on information-intensive tasks such as OCR and document understanding. To overcome these limitations, we propose InfiniteVL, a linear-complexity VLM architecture that synergizes sliding window attention (SWA) with Gated DeltaNet. For achieving competitive multimodal performance under constrained resources, we design a three-stage training strategy comprising distillation pretraining, instruction tuning, and long-sequence SFT. Remarkably, using less than 2\% of the training data required by leading VLMs, InfiniteVL not only substantially outperforms previous linear-complexity VLMs but also matches the performance of leading Transformer-based VLMs, while demonstrating effective long-term memory retention. Compared to similar-sized Transformer-based VLMs accelerated by FlashAttention-2, InfiniteVL achieves over 3.6\times inference speedup while maintaining constant latency and memory footprint. In streaming video understanding scenarios, it sustains a stable 24 FPS real-time prefill speed while preserving long-term memory cache. Code and models are available at https://github.com/hustvl/InfiniteVL.

---

## 3. Long-Sequence LSTM Modeling for NBA Game Outcome Prediction Using a Novel Multi-Season Dataset

**论文链接:** [http://arxiv.org/abs/2512.08591v1](http://arxiv.org/abs/2512.08591v1)

**作者:** Charles Rios, Longzhen Han, Almas Baimagambetov, Nikolaos Polatidis

**发布时间:** 2025-12-09

### GPT解析

### 总结

该研究构建了一个新的纵向NBA数据集和深度学习框架，通过长短期记忆(LSTM)架构有效解决了现有预测模型中的概念漂移、时间上下文有限和跨季节不稳定问题，显著提高了比赛结果预测的准确性。

### 背景

预测职业篮球比赛结果，特别是NBA比赛，对教练策略、球迷参与和体育博彩变得越来越重要。然而，许多现有预测模型面临着概念漂移、有限的时间上下文和跨季节不稳定性的挑战。

### 目的

推进篮球比赛预测领域的发展，通过引入新的纵向NBA数据集和深度学习框架，以模拟长期表现趋势并提高预测准确性。

### 方法

构建了一个覆盖2004-05至2024-25赛季的纵向NBA数据集，提出了一个基于长短期记忆(LSTM)的深度学习框架，利用相当于八个完整NBA赛季的9,840场比赛的扩展序列长度来捕捉团队动态和跨季节依赖关系。该模型与逻辑回归、随机森林、多层感知器和卷积神经网络等基线模型进行了比较。

### 主要发现

LSTM模型在所有指标上都取得了最佳性能，准确率达到72.35%，精确率为73.15%，AUC-ROC为76.13。这些结果表明长序列时间建模对篮球结果预测至关重要。

### 结论

长序列时间建模对篮球比赛结果预测至关重要，新构建的多季节数据集对于开发稳健、可推广的NBA预测系统具有重要价值。

### 翻译

预测职业篮球比赛的结果，特别是在国家篮球协会(NBA)中，对教练策略、球迷参与和体育博彩变得越来越重要。然而，许多现有的预测模型面临着概念漂移、有限的时间上下文和跨季节不稳定性的挑战。为了推进这一领域的预测研究，我们介绍了一个新构建的纵向NBA数据集，涵盖了2004-05至2024-25赛季，并提出了一个深度学习框架，旨在模拟长期表现趋势。我们的主要贡献是一个长短期记忆(LSTM)架构，它利用相当于八个完整NBA赛季的9,840场比赛的扩展序列长度来捕捉不断发展的团队动态和跨季节依赖关系。我们将该模型与几种传统的机器学习(ML)和深度学习(DL)基线进行了比较，包括逻辑回归、随机森林、多层感知器(MLP)和卷积神经网络(CNN)。LSTM在所有指标上都取得了最佳性能，准确率为72.35%，精确率为73.15%，AUC-ROC为76.13。这些结果表明长序列时间建模在篮球结果预测中的重要性，并突显了我们新的多季节数据集在开发稳健、可推广的NBA预测系统中的价值。


### 论文摘要

Predicting the outcomes of professional basketball games, particularly in the National Basketball Association (NBA), has become increasingly important for coaching strategy, fan engagement, and sports betting. However, many existing prediction models struggle with concept drift, limited temporal context, and instability across seasons. To advance forecasting in this domain, we introduce a newly constructed longitudinal NBA dataset covering the 2004-05 to 2024-25 seasons and present a deep learning framework designed to model long-term performance trends. Our primary contribution is a Long Short-Term Memory (LSTM) architecture that leverages an extended sequence length of 9,840 games equivalent to eight full NBA seasons to capture evolving team dynamics and season-over-season dependencies. We compare this model against several traditional Machine Learning (ML) and Deep Learning (DL) baselines, including Logistic Regression, Random Forest, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN). The LSTM achieves the best performance across all metrics, with 72.35 accuracy, 73.15 precision and 76.13 AUC-ROC. These results demonstrate the importance of long-sequence temporal modeling in basketball outcome prediction and highlight the value of our new multi-season dataset for developing robust, generalizable NBA forecasting systems.

---

## 4. Towards Effective and Efficient Long Video Understanding of Multimodal Large Language Models via One-shot Clip Retrieval

**论文链接:** [http://arxiv.org/abs/2512.08410v1](http://arxiv.org/abs/2512.08410v1)

**作者:** Tao Chen, Shaobo Ju, Qiong Wu, Chenxin Fang, Kun Zhang, Jun Peng, Hui Li, Yiyi Zhou, Rongrong Ji

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种名为OneClip-RAG的有效高效方法，解决了多模态大语言模型处理长视频时的内存限制问题，通过视频片段检索增强实现了在知识完整性和语义连贯性方面的优势。

### 背景

由于过高的内存开销，大多数多模态大语言模型只能处理有限帧数的视频，限制了它们在长视频理解方面的能力。

### 目的

开发一种有效且高效的范式来弥补MLLMs在处理长视频方面的不足，提高其对长视频的理解能力。

### 方法

提出OneClip-RAG范式，充分利用视频片段的优势增强视频理解；配备查询引导视频分块算法，统一片段分块和跨模态检索；提出SynLongVideo数据集和渐进式训练方案；将OneClip-RAG集成到五个最近的MLLM中进行验证。

### 主要发现

实验结果表明OneClip-RAG显著提升了MLLMs的性能，如将InternLV2 8B和Qwen2-VL 7B在MLVU上提升到GPT-4o水平；同时展示了卓越的效率，如在单块4090 GPU上使LLaVA-Video能在2.2分钟内理解长达一小时的视频。

### 结论

OneClip-RAG成功解决了MLLMs在处理长视频时的内存限制问题，同时保持了高效的性能表现，为视频理解领域提供了新的解决方案。

### 翻译

由于过高的内存开销，大多数多模态大语言模型只能处理有限帧数的视频。在本文中，我们提出了一种有效且高效的范式来弥补这一不足，称为单次视频片段检索增强(OneClip-RAG)。与现有的视频RAG方法相比，OneClip-RAG在知识完整性和语义连贯性方面充分利用了视频片段的优势来增强视频理解。此外，它还配备了一种新颖的查询引导视频分块算法，可以在一个处理步骤中统一片段分块和跨模态检索，避免冗余计算。为了提高指令遵循能力，我们进一步提出了一个名为SynLongVideo的新数据集，并为OneClip-RAG设计了渐进式训练方案。OneClip-RAG被集成到五个最近的MLLM中，并在一系列长视频基准上进行了验证。实验结果不仅显示OneClip-RAG在MLLMs上带来了明显的性能提升，例如在MLVU上将InternLV2 8B和Qwen2-VL 7B提升到GPT-4o的水平，还展示了其在处理长视频方面的卓越效率，例如在单个4090 GPU上使LLaVA-Video能够在2.2分钟内理解长达一小时的视频。


### 论文摘要

Due to excessive memory overhead, most Multimodal Large Language Models (MLLMs) can only process videos of limited frames. In this paper, we propose an effective and efficient paradigm to remedy this shortcoming, termed One-shot video-Clip based Retrieval AuGmentation (OneClip-RAG). Compared with existing video RAG methods, OneClip-RAG makes full use of the merits of video clips for augmented video understanding in terms of both knowledge integrity and semantic coherence. Besides, it is also equipped with a novel query-guided video chunking algorithm that can unify clip chunking and cross-modal retrieval in one processing step, avoiding redundant computations. To improve instruction following, we further propose a new dataset called SynLongVideo and design a progressive training regime for OneClip-RAG. OneClip-RAG is plugged into five recent MLLMs and validated on a set of long-video benchmarks. Experimental results not only show the obvious performance gains by OneClip-RAG over MLLMs, e.g., boosting InternLV2 8B and Qwen2-VL 7B to the level of GPT-4o on MLVU, but also show its superior efficiency in handling long videos. e.g., enabling LLaVA-Video understand up to an hour of videos in less than 2.2 minutes on a single 4090 GPU.

---

## 5. deepspat: An R package for modeling nonstationary spatial and spatio-temporal Gaussian and extremes data through deep deformations

**论文链接:** [http://arxiv.org/abs/2512.08137v1](http://arxiv.org/abs/2512.08137v1)

**作者:** Quan Vu, Xuanjie Shao, Raphaël Huser, Andrew Zammit-Mangion

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文介绍了R软件包deepspat，用于处理非平稳空间和时空数据的建模、拟合和预测

### 背景

空间和时空过程中的非平稳性在环境数据集中普遍存在，但由于实现非平稳模型的统计软件包稀缺，实践中很少得到解决

### 目的

开发一个R软件包，允许对高斯和极值数据应用非平稳空间和时空模型进行建模、拟合和预测

### 方法

使用原始空间或时空域的深度多层变形构建非平稳模型，并通过tensorflow进行基于梯度的优化和自动微分来估计模型参数

### 主要发现

deepspat软件包提供了一种简单易用的方法来处理非平稳空间和时空数据，填补了非平稳模型在环境数据分析中应用的空白

### 结论

deepspat软件包通过提供易于实现的工具，促进了非平稳模型在实际应用中的使用

### 翻译

空间和时空过程中的非平稳性在环境数据集中普遍存在，但由于实现非平稳模型的统计软件包稀缺，实践中很少得到解决。在本文中，我们介绍了R软件包deepspat，它允许对高斯和极值数据应用非平稳空间和时空模型进行建模、拟合和预测。我们软件包中的非平稳模型是使用原始空间或时空域的深度多层变形构建的，并且易于实现。模型参数是使用tensorflow对自定义损失函数进行基于梯度的优化来估计的，tensorflow实现了自动微分。软件包的功能通过模拟研究和对尼泊尔温度数据的应用来说明。


### 论文摘要

Nonstationarity in spatial and spatio-temporal processes is ubiquitous in environmental datasets, but is not often addressed in practice, due to a scarcity of statistical software packages that implement nonstationary models. In this article, we introduce the R software package deepspat, which allows for modeling, fitting and prediction with nonstationary spatial and spatio-temporal models applied to Gaussian and extremes data. The nonstationary models in our package are constructed using a deep multi-layered deformation of the original spatial or spatio-temporal domain, and are straightforward to implement. Model parameters are estimated using gradient-based optimization of customized loss functions with tensorflow, which implements automatic differentiation. The functionalities of the package are illustrated through simulation studies and an application to Nepal temperature data.

---

## 6. Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment

**论文链接:** [http://arxiv.org/abs/2512.08930v1](http://arxiv.org/abs/2512.08930v1)

**作者:** Youming Deng, Songyou Peng, Junyi Zhang, Kathryn Heal, Tiancheng Sun, John Flynn, Steve Marschner, Lucy Chai

**发布时间:** 2025-12-09

**备注:** Project Page: https://denghilbert.github.io/selfi/

### GPT解析

### 总结

本文提出了一种名为Selfi的自改进3D重建管道，通过特征对齐技术解决了视觉基础模型VGGT在多视图几何一致性方面的问题，显著提升了新颖视图合成和相机姿态估计的性能。

### 背景

传统的Novel View Synthesis方法依赖于具有显式3D归纳偏好的模型和来自Structure-from-Motion的已知相机参数。而最近的视觉基础模型如VGGT通过隐式学习3D知识，能够直接从未校准图像中预测相机参数和3D表示。

### 目的

提高VGGT特征的几何一致性，以改善新颖视图合成和相机姿态估计任务的性能。

### 方法

提出Selfi，一个通过特征对齐进行自改进的3D重建管道。该方法利用VGGT自身的输出作为伪真实值，训练一个轻量级特征适配器，使用基于重投影的一致性损失将VGGT输出蒸馏到几何对齐的特征空间，捕获3D中的空间邻近性。

### 主要发现

改进3D特征一致性有利于NVS和姿态估计任务；特征对齐是下游3D推理的一个非常有价值的步骤。

### 结论

Selfi通过特征对齐技术将VGGT转变为高保真3D重建引擎，在NVS和相机姿态估计方面实现了最先进的性能。

### 翻译

新颖视图合成传统上依赖于具有显式3D归纳偏好的模型，并结合来自运动结构法的已知相机参数。最近的视觉基础模型如VGGT采用了一种正交方法——通过训练数据和损失目标隐式获取3D知识，能够直接从一组未校准图像中前馈预测相机参数和3D表示。虽然灵活，但VGGT特征缺乏显式的多视图几何一致性，我们发现提高这种3D特征一致性有利于NVS和姿态估计任务。我们提出了Selfi，一种通过特征对齐进行自改进的3D重建管道，利用VGGT自身的输出作为伪真实值将其骨干转变为高保真3D重建引擎。具体来说，我们使用基于重投影的一致性损失训练了一个轻量级特征适配器，将VGGT输出蒸馏到一个新的几何对齐的特征空间，该空间捕获3D中的空间邻近性。这使我们在NVS和相机姿态估计方面实现了最先进的性能，表明特征对齐是下游3D推理的一个非常有价值的步骤。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决新型视图合成(NVS)方法在处理未校准图像时质量下降的问题。传统方法依赖精确相机参数，而现有视觉基础模型(VFM)如VGGT虽然能绕过传统SfM步骤，但其特征缺乏跨视图几何一致性，导致NVS质量不高。这个问题在AR/VR、数字孪生、虚拟旅游等领域至关重要，因为这些应用需要从有限图像生成高质量新视角，而精确相机参数往往难以获取。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到VFM特征缺乏几何一致性是NVS质量低的主要原因，提出利用VFM自身输出作为自监督信号来学习几何对齐特征。他们冻结VGGT主干，添加轻量级特征适配器，使用基于重投影的一致性损失进行训练。方法借鉴了VGGT作为基础模型、3D高斯溅射作为3D表示、DPT作为特征适配器、传统束调整改进姿态估计，以及球面谐波建模视图相关效果等现有工作，但创新性地组合这些技术解决几何一致性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自监督特征对齐将预训练VFM转换为高质量3D重建引擎，利用模型自身预测作为伪真实值学习几何一致特征。流程分三步：1)几何特征对齐 - 使用VGGT预测的深度和相机参数作为伪真实值，训练特征适配器使对应3D邻近位置的2D特征相似；2)前馈高斯预测 - 使用对齐特征预测3D高斯参数，包括位置、旋转、缩放、颜色和视图相关的不透明度；3)密集束调整与深度偏移 - 用对齐特征改进相机姿态，并通过深度偏移变换保持高斯与改进后几何的一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)自监督几何特征学习，无需3D标注；2)基于重投影的特征对齐策略，使特征相似性同时捕获语义和3D空间关系；3)视图相关密度建模，使用球面谐波调整不透明度作为置信度指标；4)深度偏移变换解决姿态改进后的几何不一致。相比传统方法，它不依赖精确相机参数；相比VFM方法，显著提高了NVS质量；相比其他前馈高斯方法，实现了更好的几何一致性和处理遮挡能力；相比自监督对应学习，更专注于3D几何一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Selfi通过自监督几何特征对齐，将预训练视觉基础模型转换为高质量3D重建引擎，实现了从未校准图像生成新视图的最先进性能，并提供了改进相机姿态估计的能力。'}


### 论文摘要

Novel View Synthesis (NVS) has traditionally relied on models with explicit 3D inductive biases combined with known camera parameters from Structure-from-Motion (SfM) beforehand. Recent vision foundation models like VGGT take an orthogonal approach -- 3D knowledge is gained implicitly through training data and loss objectives, enabling feed-forward prediction of both camera parameters and 3D representations directly from a set of uncalibrated images. While flexible, VGGT features lack explicit multi-view geometric consistency, and we find that improving such 3D feature consistency benefits both NVS and pose estimation tasks. We introduce Selfi, a self-improving 3D reconstruction pipeline via feature alignment, transforming a VGGT backbone into a high-fidelity 3D reconstruction engine by leveraging its own outputs as pseudo-ground-truth. Specifically, we train a lightweight feature adapter using a reprojection-based consistency loss, which distills VGGT outputs into a new geometrically-aligned feature space that captures spatial proximity in 3D. This enables state-of-the-art performance in both NVS and camera pose estimation, demonstrating that feature alignment is a highly beneficial step for downstream 3D reasoning.

---

## 7. Can TabPFN Compete with GNNs for Node Classification via Graph Tabularization?

**论文链接:** [http://arxiv.org/abs/2512.08798v1](http://arxiv.org/abs/2512.08798v1)

**作者:** Jeongwhan Choi, Woosung Kang, Minseo Kim, Jongwoo Kim, Noseong Park

**发布时间:** 2025-12-09

**备注:** Rejected from LoG 2025 (submitted August 2025)

### GPT解析

### 总结

论文提出了TabPFN-GN方法，将图节点分类重新表述为表格学习问题，通过提取节点属性、结构特性、位置编码和邻域特征，使TabPFN能够直接进行节点分类。实验表明该方法在同质图上与GNNs性能相当，在异质图上则优于GNNs。

### 背景

基于大型数据预训练的基础模型展示了跨领域的零样本泛化能力。TabPFN在表格数据和时间序列上的成功应用为本研究提供了基础。

### 目的

研究是否可以将图节点分类有效地重新表述为表格学习问题，探索图数据与表格数据之间的桥梁。

### 方法

提出TabPFN-GN方法，通过提取节点属性、结构特性、位置编码和可选的平滑邻域特征，将图数据转换为表格特征，使TabPFN能够直接进行节点分类，无需图特定训练或语言模型依赖。

### 主要发现

在12个基准数据集上的实验表明，TabPFN-GN在同质图上与GNNs具有竞争性的性能，在异质图上则持续优于GNNs。

### 结论

有原则的特征工程可以弥合表格和图领域之间的差距，为特定任务的GNN训练和依赖LLM的图基础模型提供了实用的替代方案。

### 翻译

在大型数据上预训练的基础模型展示了跨领域的卓越零样本泛化能力。基于TabPFN在表格数据上的成功及其最近对时间序列的扩展，我们研究是否可以将图节点分类有效地重新表述为表格学习问题。我们引入了TabPFN-GN，它通过提取节点属性、结构特性、位置编码和可选的平滑邻域特征，将图数据转换为表格特征。这使得TabPFN能够直接进行节点分类，无需任何图特定的训练或语言模型依赖。我们在12个基准数据集上的实验显示，TabPFN-GN在同质图上与GNNs具有竞争性的性能，在异质图上则持续优于它们。这些结果表明，有原则的特征工程可以弥合表格和图领域之间的差距，为特定任务的GNN训练和依赖LLM的图基础模型提供了实用的替代方案。


### 论文摘要

Foundation models pretrained on large data have demonstrated remarkable zero-shot generalization capabilities across domains. Building on the success of TabPFN for tabular data and its recent extension to time series, we investigate whether graph node classification can be effectively reformulated as a tabular learning problem. We introduce TabPFN-GN, which transforms graph data into tabular features by extracting node attributes, structural properties, positional encodings, and optionally smoothed neighborhood features. This enables TabPFN to perform direct node classification without any graph-specific training or language model dependencies. Our experiments on 12 benchmark datasets reveal that TabPFN-GN achieves competitive performance with GNNs on homophilous graphs and consistently outperforms them on heterophilous graphs. These results demonstrate that principled feature engineering can bridge the gap between tabular and graph domains, providing a practical alternative to task-specific GNN training and LLM-dependent graph foundation models.

---

## 8. A Multi-Robot Platform for Robotic Triage Combining Onboard Sensing and Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.08754v1](http://arxiv.org/abs/2512.08754v1)

**作者:** Jason Hughes, Marcel Hussing, Edward Zhang, Shenbagaraj Kannapiran, Joshua Caswell, Kenneth Chaney, Ruichen Deng, Michaela Feehery, Agelos Kratimenos, Yi Fan Li, Britny Major, Ethan Sanchez, Sumukh Shrote, Youkang Wang, Jeremy Wang, Daudi Zein, Luying Zhang, Ruijun Zhang, Alex Zhou, Tenzi Zhouga, Jeremy Cannon, Zaffir Qasim, Jay Yelon, Fernando Cladera, Kostas Daniilidis, Camillo J. Taylor, Eric Eaton

**发布时间:** 2025-12-09

**备注:** Technical Report for the DARPA Triage Challenge PRONTO team

### GPT解析

### 总结

该研究提出了一种异构机器人系统，用于大规模伤亡事件的远程初步分类，通过无人机和无人地面车辆的协作团队来定位受害者、评估伤势并优先安排医疗援助。

### 背景

大规模伤亡事件(MCIs)中，第一响应者的生命安全面临风险，需要远程解决方案进行初步分类。

### 目的

开发一种完整的分类系统，包括受害者定位、生命体征测量、伤害严重程度分类、精神状态评估和数据整合，而不危及第一响应者的生命。

### 方法

使用协调的空中-地面团队，无人机负责识别并提供伤亡的俯瞰视图，配备专用传感器的无人地面车辆测量生命体征并检测和定位身体损伤。

### 主要发现

该系统能够完成完整的分类过程，不同于之前专注于探索或有限医疗评估的研究，解决了从受害者定位到数据整合的全流程需求。

### 结论

多机器人系统可以在灾难响应场景中增强人类能力，最大限度地挽救生命。

### 翻译

本报告提出了一种用于大规模伤亡事件(MCIs)远程初步分类的异构机器人系统。该系统采用协调的空中-地面团队，包括无人机(UAV)和无人地面车辆(UGV)，用于定位受害者、评估伤势和优先安排医疗援助，而不危及第一响应者的生命。无人机识别并提供伤亡的俯瞰视图，而配备专用传感器的无人地面车辆测量生命体征并检测和定位身体损伤。与专注于探索或有限医疗评估的先前研究不同，该系统解决了完整的分类过程：受害者定位、生命体征测量、伤害严重程度分类、精神状态评估以及为第一响应者整合数据。作为DARPA分类挑战的一部分开发，该方法展示了多机器人系统如何在灾难响应场景中增强人类能力，以最大限度地挽救生命。


### 论文摘要

This report presents a heterogeneous robotic system designed for remote primary triage in mass-casualty incidents (MCIs). The system employs a coordinated air-ground team of unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) to locate victims, assess their injuries, and prioritize medical assistance without risking the lives of first responders. The UAV identify and provide overhead views of casualties, while UGVs equipped with specialized sensors measure vital signs and detect and localize physical injuries. Unlike previous work that focused on exploration or limited medical evaluation, this system addresses the complete triage process: victim localization, vital sign measurement, injury severity classification, mental status assessment, and data consolidation for first responders. Developed as part of the DARPA Triage Challenge, this approach demonstrates how multi-robot systems can augment human capabilities in disaster response scenarios to maximize lives saved.

---

## 9. Towards Foundation Models with Native Multi-Agent Intelligence

**论文链接:** [http://arxiv.org/abs/2512.08743v1](http://arxiv.org/abs/2512.08743v1)

**作者:** Shuyue Hu, Haoyang Yan, Yiqun Zhang, Yang Chen, Dongzhan Zhou, Lei Bai

**发布时间:** 2025-12-09

### GPT解析

### 总结

这篇论文讨论了基础模型(FMs)作为AI智能体'大脑'的角色，以及如何赋予它们原生多智能体智能。作者识别了多智能体环境中的四个核心能力：理解、规划、高效通信和适应。

### 背景

基础模型(FMs)正越来越多地扮演AI智能体的'大脑'角色。最近的努力已经开始赋予基础模型原生单智能体能力，如GUI交互或集成工具使用。

### 目的

作者认为下一个前沿是赋予基础模型原生多智能体智能，并为此确定了四个核心能力：理解、规划、高效通信和适应。同时，旨在解决单智能体表现与多智能体智能之间的差距。

### 方法

作者通过41个大语言模型的广泛实证研究，检验了单智能体表现与多智能体智能之间的关系。

### 主要发现

研究表明，强大的单智能体表现并不自动产生稳健的多智能体智能，这与关于这些能力自发出现的假设相矛盾。

### 结论

作者提出了构建具有原生多智能体智能的基础模型的关键研究方向，包括数据集构建、评估、训练范式和安全考虑。

### 翻译

基础模型(FMs)正越来越多地扮演AI智能体的'大脑'角色。虽然最近的努力已经开始赋予基础模型原生单智能体能力--如GUI交互或集成工具使用--但我们认为下一个前沿是赋予基础模型原生多智能体智能。我们确定了基础模型在多智能体环境中的四个核心能力：理解、规划、高效通信和适应。与关于这些能力自发出现的假设相反，我们在41个大语言模型上提供了广泛的实证证据，表明强大的单智能体表现并不自动产生稳健的多智能体智能。为了解决这一差距，我们概述了构建具有原生多智能体智能的基础模型的关键研究方向--涵盖数据集构建、评估、训练范式和安全考虑。


### 论文摘要

Foundation models (FMs) are increasingly assuming the role of the "brain" of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence across 41 large language models showing that strong single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence.

---

## 10. Scale-invariant and View-relational Representation Learning for Full Surround Monocular Depth

**论文链接:** [http://arxiv.org/abs/2512.08700v1](http://arxiv.org/abs/2512.08700v1)

**作者:** Kyumin Hwang, Wonhyeok Choi, Kiljoon Han, Wonjoon Choi, Minwoo Choi, Yongcheon Na, Minwoo Park, Sunghoon Im

**发布时间:** 2025-12-09

**备注:** Accepted at IEEE Robotics and Automation Letters (RA-L) 2026

### GPT解析

### 总结

该研究提出了一种新颖的知识蒸馏策略，将foundation model的深度知识转移到轻量级全环绕单目深度估计网络，解决了高计算成本和难以估计度量尺度深度的问题。

### 背景

最近的foundation models在单目深度估计中表现出强大的泛化能力，但直接应用于全环绕单目深度估计(FSMDE)面临两个主要挑战：高计算成本限制了实时性能，以及难以估计度量尺度深度，因为这些模型通常只训练来预测相对深度。

### 目的

提出一种新颖的知识蒸馏策略，将foundation model的鲁棒深度知识转移到轻量级的FSMDE网络中，解决计算成本高和深度尺度估计困难的问题。

### 方法

利用混合回归框架结合知识蒸馏方案和深度分箱模块提高尺度一致性；引入交叉交互知识蒸馏方案，将foundation模型的尺度不变深度分箱概率蒸馏到学生网络，同时指导它推断度量尺度深度分箱中心；提出视图关系知识蒸馏，编码相邻摄像头视图间的结构关系并转移它们，以增强跨视图深度一致性。

### 主要发现

在DDAD和nuScenes数据集上的实验表明，该方法相比传统监督方法和现有知识蒸馏方法更有效，且在性能和效率之间取得了良好的平衡。

### 结论

该方法能够满足实时要求，为全环绕单目深度估计提供了一种有效的解决方案。

### 翻译

最近的foundation models在单目深度估计中表现出强大的泛化能力。然而，直接将这些模型应用于全环绕单目深度估计(FSMDE)存在两个主要挑战：(1)高计算成本，限制了实时性能；(2)难以估计度量尺度深度，因为这些模型通常只训练来预测相对深度。为解决这些限制，我们提出了一种新颖的知识蒸馏策略，将foundation model的鲁棒深度知识转移到轻量级FSMDE网络中。我们的方法利用混合回归框架，结合传统用于分类的知识蒸馏方案和深度分箱模块来提高尺度一致性。具体来说，我们引入了交叉交互知识蒸馏方案，将foundation模型的尺度不变深度分箱概率蒸馏到学生网络，同时指导它从真实深度推断度量尺度深度分箱中心。此外，我们提出了视图关系知识蒸馏，编码相邻摄像头视图之间的结构关系并转移它们，以增强跨视图深度一致性。在DDAD和nuScenes上的实验证明了我们的方法相比传统监督方法和现有知识蒸馏方法的有效性。此外，我们的方法在性能和效率之间取得了良好的平衡，满足实时要求。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决将基础模型应用于全环形单目深度估计(FSMDE)时面临的两个挑战：高计算成本和难以估计度量尺度深度。这个问题在自动驾驶领域非常重要，因为全环相机系统是LiDAR的经济替代方案，实时深度估计对于可靠决策至关重要，而轻量级模型对于车辆部署必不可少，同时跨视图一致深度对准确场景理解至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分解问题认识到直接应用基础模型的两个主要挑战，然后借鉴知识蒸馏概念将大型教师模型知识转移到小型学生模型。他们采用监督深度估计中的混合回归范式结合深度分箱技术，设计了交叉交互知识蒸馏(CKD)和视图关系知识蒸馏(VRKD)两种创新方法。该方法借鉴了多种现有工作，包括知识蒸馏技术(KD, FitNets等)、深度分箱方法(AdaBins, LocalBins等)、全环单目深度估计方法(FSM, SurroundDepth等)和基础模型(DepthAnything)。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将大型基础模型的鲁棒深度知识转移到轻量级FSMDE学生网络中，通过深度分箱模块处理尺度不变性和度量尺度深度估计的矛盾，结合概率级别知识蒸馏和视图关系蒸馏提高跨视图一致性。实现流程包括：1)构建教师(基础模型)和学生(轻量级网络)模型架构并共享深度分箱模块；2)通过CKD将教师模型的深度分箱概率转移到学生网络；3)通过VRKD编码相邻摄像头视图间的结构关系并转移到学生网络；4)结合监督损失、CKD损失和VRKD损失进行训练；5)训练后的学生模型可实时预测全环场景深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)交叉交互知识蒸馏(CKD)在概率级别进行知识蒸馏，解决基础模型输出尺度模糊问题；2)视图关系知识蒸馏(VRKD)编码相邻摄像头视图间的结构关系，提高多视图深度一致性；3)将分类任务中的知识蒸馏与深度分箱模块结合提高尺度一致性。相比之前工作，该方法专门针对FSMDE场景，在概率级别而非输出级别进行蒸馏，引入视图关系建模，实现了更好的效率-性能权衡(5.13-11.14倍速度提升)，并在两个标准数据集上进行了更全面的评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种新颖的知识蒸馏策略，通过交叉交互和视图关系知识蒸馏，将大型基础模型的鲁棒深度知识高效转移到轻量级全环形单目深度估计网络中，在保持实时性能的同时显著提高了深度估计的准确性和跨视图一致性。'}


### 论文摘要

Recent foundation models demonstrate strong generalization capabilities in monocular depth estimation. However, directly applying these models to Full Surround Monocular Depth Estimation (FSMDE) presents two major challenges: (1) high computational cost, which limits real-time performance, and (2) difficulty in estimating metric-scale depth, as these models are typically trained to predict only relative depth. To address these limitations, we propose a novel knowledge distillation strategy that transfers robust depth knowledge from a foundation model to a lightweight FSMDE network. Our approach leverages a hybrid regression framework combining the knowledge distillation scheme--traditionally used in classification--with a depth binning module to enhance scale consistency. Specifically, we introduce a cross-interaction knowledge distillation scheme that distills the scale-invariant depth bin probabilities of a foundation model into the student network while guiding it to infer metric-scale depth bin centers from ground-truth depth. Furthermore, we propose view-relational knowledge distillation, which encodes structural relationships among adjacent camera views and transfers them to enhance cross-view depth consistency. Experiments on DDAD and nuScenes demonstrate the effectiveness of our method compared to conventional supervised methods and existing knowledge distillation approaches. Moreover, our method achieves a favorable trade-off between performance and efficiency, meeting real-time requirements.

---

## 11. 论文ID: 2512.08648v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.08648v1.json'

---

## 12. OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics

**论文链接:** [http://arxiv.org/abs/2512.08625v1](http://arxiv.org/abs/2512.08625v1)

**作者:** Jisang Yoo, Gyeongjin Kang, Hyun-kyu Ko, Hyeonwoo Yu, Eunbyung Park

**发布时间:** 2025-12-09

**备注:** 8 pages, 4 figures

### GPT解析

### 总结

本文提出了OpenMonoGS-SLAM，这是首个将3D高斯飞溅与开放式语义理解相结合的单目SLAM框架，无需深度输入或3D语义真实值，仅依靠自监督学习目标。

### 背景

SLAM是机器人、AR/VR和自主系统的基础组件。随着空间AI的兴起，将SLAM与语义理解相结合对实现智能感知与交互变得越来越重要。

### 目的

开发一种不依赖深度传感器或封闭式语义模型，且能在开放世界环境中具有良好可扩展性和适应性的SLAM框架。

### 方法

利用视觉基础模型(VFMs)的最新进展，包括MASt3R用于视觉几何，SAM和CLIP用于开放式词汇语义，并提出了专门管理高维语义特征的内存机制。

### 主要发现

该方法在封闭集和开放集分割任务中实现了与现有基线相当或更好的性能，且不依赖深度图或语义注释等辅助传感器。

### 结论

OpenMonoGS-SLAM成功实现了单目SLAM框架，通过结合3D高斯飞溅与开放式语义理解，在开放世界环境中表现良好。

### 翻译

同步定位与地图构建(SLAM)是机器人、AR/VR和自主系统的基础组件。随着近年来空间AI的兴起，将SLAM与语义理解相结合已变得越来越重要，以实现智能感知与交互。最近的研究探索了这种集成，但它们通常依赖深度传感器或封闭式语义模型，限制了它们在开放世界环境中的可扩展性和适应性。在这项工作中，我们提出了OpenMonoGS-SLAM，这是第一个将3D高斯飞溅(3DGS)与开放式语义理解相结合的单目SLAM框架。为实现我们的目标，我们利用了视觉基础模型(VFMs)的最新进展，包括MASt3R用于视觉几何，以及SAM和CLIP用于开放式词汇语义。这些模型提供了跨不同任务的强大泛化能力，实现了精确的单目相机跟踪和映射，以及对开放世界环境中语义的丰富理解。我们的方法无需任何深度输入或3D语义真实值，仅依靠自监督学习目标。此外，我们提出了专门设计用于管理高维语义特征的内存机制，有效构建了高斯语义特征图，实现了强大的整体性能。实验结果表明，我们的方法在封闭集和开放集分割任务中实现了与现有基线相当或更好的性能，且不依赖深度图或语义注释等辅助传感器。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有单目SLAM系统在语义理解方面的局限性，特别是大多数方法依赖深度传感器或封闭集语义模型，限制了它们在开放世界环境中的应用。这个问题很重要，因为SLAM是机器人、AR/VR和自主系统的基础组件，而随着空间AI的兴起，结合SLAM与语义理解对于实现智能感知和交互变得至关重要。现有方法无法在只有RGB相机的低成本平台上工作，也无法处理开放环境中未知类别的物体，限制了系统的实际应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有3D高斯飞溅SLAM方法大多依赖深度传感器，限制了在只有RGB相机的场景中的应用；同时发现语义SLAM大多局限于封闭集识别，无法处理开放集环境。作者借鉴了MASt3R-SLAM作为骨干模型进行相机跟踪，利用视觉基础模型(VFMs)包括MASt3R、SAM和CLIP的优势，参考了SAGA进行多视图语义一致性处理，并受M3方法的启发实现了可扩展的记忆库但将其适应到在线SLAM场景。整体设计思路是整合这些现有技术，解决单目和开放集语义的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D高斯飞溅(3DGS)与开放集语义理解相结合，利用视觉基础模型提供强大泛化能力，通过记忆机制管理高维语义特征，并使用自监督学习进行优化。整体流程包括：1)使用MASt3R进行单目相机跟踪和3D点重建；2)将3D点初始化为具有颜色属性和可学习语义特征的3D高斯；3)使用SAM生成2D对象掩码并提升到3D空间；4)利用CLIP特征为分割区域生成嵌入，实现语言先验注入；5)结合光度监督、多视图语义一致性和语言引导语义对齐等自监督学习目标进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次统一3D高斯飞溅和开放集语义理解的单目SLAM框架；2)设计高效紧凑的语义融合策略，通过注意力机制在CLIP特征记忆库上构建低维高斯语义图；3)利用多尺度方式表达语义特征，实现尺度感知和开放集语义映射。相比之前工作，OpenMonoGS-SLAM不需要深度传感器输入，仅使用单目RGB；支持开放集语义理解而非局限于预定义类别；不需要3D语义真实标签，完全依赖自监督学习；结合多种视觉基础模型优势；使用记忆机制高效管理高维语义特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OpenMonoGS-SLAM首次将3D高斯飞溅与开放集语义理解相结合，仅使用单目RGB输入和自监督学习，实现了在开放世界环境中高质量实时SLAM系统。'}


### 论文摘要

Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems. With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction. Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments. In this work, we present OpenMonoGS-SLAM, the first monocular SLAM framework that unifies 3D Gaussian Splatting (3DGS) with open-set semantic understanding. To achieve our goal, we leverage recent advances in Visual Foundation Models (VFMs), including MASt3R for visual geometry and SAM and CLIP for open-vocabulary semantics. These models provide robust generalization across diverse tasks, enabling accurate monocular camera tracking and mapping, as well as a rich understanding of semantics in open-world environments. Our method operates without any depth input or 3D semantic ground truth, relying solely on self-supervised learning objectives. Furthermore, we propose a memory mechanism specifically designed to manage high-dimensional semantic features, which effectively constructs Gaussian semantic feature maps, leading to strong overall performance. Experimental results demonstrate that our approach achieves performance comparable to or surpassing existing baselines in both closed-set and open-set segmentation tasks, all without relying on supplementary sensors such as depth maps or semantic annotations.

---

## 13. LapFM: A Laparoscopic Segmentation Foundation Model via Hierarchical Concept Evolving Pre-training

**论文链接:** [http://arxiv.org/abs/2512.08439v1](http://arxiv.org/abs/2512.08439v1)

**作者:** Qing Xu, Kun Yuan, Yuxiang Luo, Yuhao Zhai, Wenting Duan, Nassir Navab, Zhen Chen

**发布时间:** 2025-12-09

### GPT解析

### 总结

LapFM是一种创新的腹腔镜分割基础模型，通过分层概念演化预训练范式解决手术分割中的标注稀缺和语义不一致问题。该方法建立了腹腔镜概念层次结构，并提出基于置信度的演化标注方法，实验证明LapFM在腹腔镜分割任务上显著优于现有方法。

### 背景

手术分割对于场景理解至关重要，但面临标注稀缺和不同手术间语义不一致的挑战。现有方法通常只使用有限监督微调自然基础模型（如SAM），仅作为领域适配器而非手术基础模型，难以泛化到各种手术目标的巨大变异性。

### 目的

提出一个名为LapFM的基础模型，旨在从未标记的海量手术图像中发展出强大的分割能力，解决现有方法在手术分割中的泛化问题。

### 方法

提出分层概念演化预训练范式，建立腹腔镜概念层次结构（LCH）通过具有父子查询嵌入的分层掩码解码器，将不同实体统一为可扩展的知识结构；提出基于置信度的演化标注方法，迭代生成和过滤伪标签并逐步纳入可靠样本；创建LapBench-114K基准数据集。

### 主要发现

LapFM显著优于最先进的方法，为腹腔镜分割中的粒度自适应泛化建立了新标准。

### 结论

LapFM有效解决了手术分割中的泛化问题，通过分层概念层次结构和基于置信度的演化标注方法，从未标记的海量手术图像中学习出强大的分割能力。

### 翻译

手术分割对于场景理解至关重要，但仍受限于不同程序中的标注稀缺和语义不一致性。现有方法通常使用有限监督微调自然基础模型（如SAM），仅作为领域适配器而非手术基础模型。因此，它们难以泛化到手术目标的巨大变异性。为了弥补这一差距，我们提出了LapFM，这是一个旨在从未标记的海量手术图像中发展强大分割能力的基础模型。与依赖低效自监督代理任务的医学基础模型不同，LapFM利用分层概念演化预训练范式。首先，我们通过具有父子查询嵌入的分层掩码解码器建立了腹腔镜概念层次结构，将不同实体（即解剖结构、组织和器械）统一为具有跨粒度语义一致性的可扩展知识结构。其次，我们提出了基于置信度的演化标注方法，基于层次一致性迭代生成和过滤伪标签，逐步将未标记图像中的可靠样本纳入训练。这一过程产生了LapBench-114K，这是一个包含114K图像-掩码对的大规模基准。大量实验表明，LapFM显著优于最先进的方法，为通用腹腔镜分割中的粒度自适应泛化树立了新标准。源代码可在https://github.com/xq141839/LapFM获取。


### 论文摘要

Surgical segmentation is pivotal for scene understanding yet remains hindered by annotation scarcity and semantic inconsistency across diverse procedures. Existing approaches typically fine-tune natural foundation models (e.g., SAM) with limited supervision, functioning merely as domain adapters rather than surgical foundation models. Consequently, they struggle to generalize across the vast variability of surgical targets. To bridge this gap, we present LapFM, a foundation model designed to evolve robust segmentation capabilities from massive unlabeled surgical images. Distinct from medical foundation models relying on inefficient self-supervised proxy tasks, LapFM leverages a Hierarchical Concept Evolving Pre-training paradigm. First, we establish a Laparoscopic Concept Hierarchy (LCH) via a hierarchical mask decoder with parent-child query embeddings, unifying diverse entities (i.e., Anatomy, Tissue, and Instrument) into a scalable knowledge structure with cross-granularity semantic consistency. Second, we propose a Confidence-driven Evolving Labeling that iteratively generates and filters pseudo-labels based on hierarchical consistency, progressively incorporating reliable samples from unlabeled images into training. This process yields LapBench-114K, a large-scale benchmark comprising 114K image-mask pairs. Extensive experiments demonstrate that LapFM significantly outperforms state-of-the-art methods, establishing new standards for granularity-adaptive generalization in universal laparoscopic segmentation. The source code is available at https://github.com/xq141839/LapFM.

---

## 14. Prismatic World Model: Learning Compositional Dynamics for Planning in Hybrid Systems

**论文链接:** [http://arxiv.org/abs/2512.08411v1](http://arxiv.org/abs/2512.08411v1)

**作者:** Mingwei Li, Xiaoyuan Zhang, Chengwei Yang, Zilong Zheng, Yaodong Yang

**发布时间:** 2025-12-09

### GPT解析

### 总结

这篇论文介绍了PRISM-WM（Prismatic World Model），一种用于处理机器人领域混合动态的新型结构化架构，能够准确建模系统动力学中的尖锐模式转换，为轨迹优化算法提供高保真基础。

### 背景

在机器人领域，基于模型的规划面临物理动力学的混合性质挑战，其中连续运动被离散事件（如接触和冲击）打断。传统的潜在世界模型通常使用整体神经网络强制全局连续性，不可避免地对不同的动态模式（如粘附与滑动，飞行与站立）进行过度平滑处理，导致在长时程规划中产生不可靠的预测。

### 目的

解决传统潜在世界模型在处理混合动力学时的局限性，特别是它们过度平滑不同动态模式的问题，导致在长时程规划中产生不可靠的预测。

### 方法

作者提出了PRISM-WM，一种结构化架构，旨在将复杂的混合动力学分解为可组合的基元。PRISM-WM利用了上下文感知的专家混合框架，其中门控机制隐式识别当前物理模式，专门的专家预测相关的转换动力学。此外，他们引入了潜在正交化目标来确保专家多样性，有效防止模式崩溃。

### 主要发现

通过准确建模系统动力学中的尖锐模式转换，PRISM-WM显著减少了轨迹漂移。在具有挑战性的连续控制基准测试上的广泛实验，包括高维度人形机器人和多样化的多任务设置，证明PRISM-WM为轨迹优化算法提供了优越的高保真基础。

### 结论

PRISM-WM证明其有潜力作为下一代基于模型的智能体的强大基础模型。

### 翻译

机器人领域中的基于模型的规划从根本上受到物理动力学混合性质的挑战，其中连续运动被离散事件（如接触和冲击）打断。传统的潜在世界模型通常采用整体神经网络强制全局连续性，不可避免地对不同的动态模式进行过度平滑。为解决这一问题，我们引入了PRISM-WM，一种旨在将复杂混合动力学分解为可组合基元的设计结构化架构。PRISM-WM利用上下文感知的专家混合框架，其中门控机制隐式识别当前物理模式，专门的专家预测相关的转换动力学。我们进一步引入了潜在正交化目标以确保专家多样性，有效防止模式崩溃。通过准确建模系统动力学中的尖锐模式转换，PRISM-WM显著减少了轨迹漂移。在具有挑战性的连续控制基准测试上的广泛实验证明，PRISM-WM为轨迹优化算法提供了优越的高保真基础，证明其有潜力作为下一代基于模型的智能体的强大基础模型。


### 论文摘要

Model-based planning in robotic domains is fundamentally challenged by the hybrid nature of physical dynamics, where continuous motion is punctuated by discrete events such as contacts and impacts. Conventional latent world models typically employ monolithic neural networks that enforce global continuity, inevitably over-smoothing the distinct dynamic modes (e.g., sticking vs. sliding, flight vs. stance). For a planner, this smoothing results in catastrophic compounding errors during long-horizon lookaheads, rendering the search process unreliable at physical boundaries. To address this, we introduce the Prismatic World Model (PRISM-WM), a structured architecture designed to decompose complex hybrid dynamics into composable primitives. PRISM-WM leverages a context-aware Mixture-of-Experts (MoE) framework where a gating mechanism implicitly identifies the current physical mode, and specialized experts predict the associated transition dynamics. We further introduce a latent orthogonalization objective to ensure expert diversity, effectively preventing mode collapse. By accurately modeling the sharp mode transitions in system dynamics, PRISM-WM significantly reduces rollout drift. Extensive experiments on challenging continuous control benchmarks, including high-dimensional humanoids and diverse multi-task settings, demonstrate that PRISM-WM provides a superior high-fidelity substrate for trajectory optimization algorithms (e.g., TD-MPC), proving its potential as a powerful foundational model for next-generation model-based agents.

---

## 15. SOP^2: Transfer Learning with Scene-Oriented Prompt Pool on 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.08223v1](http://arxiv.org/abs/2512.08223v1)

**作者:** Ching-Hung Cheng, Hsiu-Fu Wu, Bing-Chen Wu, Khanh-Phong Bui, Van-Tin Luu, Ching-Chun Huang

**发布时间:** 2025-12-09

**DOI:** 10.1109/AVSS65446.2025.11149933

### GPT解析

### 总结

这篇论文研究了提示调优方法在3D目标检测中的有效性，探讨了基于大规模Waymo数据集训练的模型能否作为基础模型并适应3D目标检测领域的其他场景，并提出了场景导向的提示池方法。

### 背景

大型语言模型（如GPT-3）展现出强大的泛化能力，通过迁移学习技术（如微调和提示调优）可以适应各种下游任务且只需最小化的参数调整，这种方法在自然语言处理领域尤为常见。

### 目的

探索常见的提示调优方法在3D目标检测中的有效性，研究基于Waymo数据集训练的模型能否作为3D目标检测领域的基础模型并适应其他场景。

### 方法

依次检查提示令牌和提示生成器的影响，并提出了场景导向的提示池方法。

### 主要发现

证明了提示池在3D目标检测中的有效性。

### 结论

提示池方法在3D目标检测中有效，旨在激励未来研究人员更深入地探索提示在3D领域的潜力。

### 翻译

随着GPT-3等大型语言模型的兴起，这些模型展现出强大的泛化能力。通过微调和提示调优等迁移学习技术，它们可以适应各种下游任务，只需进行最小化的参数调整。这种方法在自然语言处理领域尤为常见。本文旨在探索常见提示调优方法在3D目标检测中的有效性。我们研究了在大型Waymo数据集上训练的模型能否作为基础模型，并适应3D目标检测领域内的其他场景。本文依次检查了提示令牌和提示生成器的影响，并进一步提出了场景导向的提示池。我们证明了提示池在3D目标检测中的有效性，旨在激励未来研究人员更深入地探索提示在3D领域的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D目标检测中的领域适应问题，即如何在一个大规模数据集（如Waymo）上训练的模型能够适应到其他不同的3D目标检测场景中。这个问题在现实中非常重要，因为3D目标检测对自动驾驶、机器人、增强现实和城市规划等应用至关重要，而不同场景间的传感器配置、环境条件和数据收集方法差异会导致模型性能下降，解决这一挑战对开发稳健且适应性强的3D目标检测系统至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到自然语言处理领域的大型语言模型通过迁移学习技术表现出强大的泛化能力，然后探索这些方法在3D目标检测中的适用性。作者系统地研究了提示令牌和提示生成器的影响，并最终设计了场景导向的提示池。该方法借鉴了多个现有工作：提示调优（在模型输入中引入任务特定提示）、低秩适应（减少可训练参数）、提示池（存储并动态选择提示）、视觉提示调优（将提示调优扩展到视觉领域）以及动态视觉提示调优（根据输入数据动态生成提示）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是为3D场景中的每个分区维护一个提示池，使模型能够根据当前场景动态选择最适合的提示令牌，专注于学习场景特定信息而非任务特定信息。实现流程包括：1)使用DSVT框架处理点云数据并分区；2)为每个分区分配一个包含键值对的提示池；3)使用查询函数将输入投影到键的维度；4)通过选择函数计算相似度并选择top-K最相似键；5)收集对应值构建集合特定提示；6)将提示与场景信息集成；7)在提示后的集合上应用多头自注意力进行信息交换。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)场景导向的提示方法，专注于学习领域特定信息；2)分区特定的提示池，为每个分区提供独立提示池；3)动态提示选择，根据当前场景选择最合适提示；4)高效的参数利用，仅用0.82M参数实现高性能。相比前人工作，SOP2使用提示池而非单一提示，允许更灵活的选择；与提示生成器不同，它从预定义池中选择而非生成提示；与非监督领域适应相比，它不需要复杂的训练策略；与低秩适应相比，它专注于提示而非权重矩阵近似，在性能上显示出优势。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了场景导向的提示池（SOP2）方法，通过为3D场景中的每个分区维护可选择的提示池，实现了在保持参数效率的同时显著提升3D目标检测的跨领域适应性能。'}


### 论文摘要

With the rise of Large Language Models (LLMs) such as GPT-3, these models exhibit strong generalization capabilities. Through transfer learning techniques such as fine-tuning and prompt tuning, they can be adapted to various downstream tasks with minimal parameter adjustments. This approach is particularly common in the field of Natural Language Processing (NLP). This paper aims to explore the effectiveness of common prompt tuning methods in 3D object detection. We investigate whether a model trained on the large-scale Waymo dataset can serve as a foundation model and adapt to other scenarios within the 3D object detection field. This paper sequentially examines the impact of prompt tokens and prompt generators, and further proposes a Scene-Oriented Prompt Pool (\textbf{SOP$^2$}). We demonstrate the effectiveness of prompt pools in 3D object detection, with the goal of inspiring future researchers to delve deeper into the potential of prompts in the 3D field.

---

## 16. Tumor-anchored deep feature random forests for out-of-distribution detection in lung cancer segmentation

**论文链接:** [http://arxiv.org/abs/2512.08216v1](http://arxiv.org/abs/2512.08216v1)

**作者:** Aneesh Rangnekar, Harini Veeraraghavan

**发布时间:** 2025-12-09

### GPT解析

### 总结

RF-Deep是一种轻量级的后处理OOD检测框架，能有效检测分布外输入，提高肿瘤分割的可靠性，在不同深度和预训练策略的网络中保持一致性能。

### 背景

从3D CT扫描中准确分割癌变病变对自动治疗计划和反应评估至关重要。然而，即使是结合自监督学习预训练转换器和卷积解码器的先进模型，也容易受到分布外(OOD)输入影响，产生自信但不正确的肿瘤分割，对安全临床部署构成风险。

### 目的

开发一种轻量级的即插即用后处理OOD检测框架，增强肿瘤分割的可靠性，无需增加模型参数和计算成本。

### 方法

提出名为RF-Deep的随机森林基础OOD检测框架，利用具有有限异常值暴露的深度特征。通过重用预训练后微调的主干编码器层次特征，增强对成像变化的泛化能力，并从锚定到预测肿瘤分割的多个感兴趣区域提取特征，提供任务相关的OOD检测。

### 主要发现

使用1,916个CT扫描进行比较，RF-Deep在近OOD数据集上实现AUROC > 93.50，在远OOD数据集上实现近乎完美检测(AUROC > 99.00)，显著优于基于logit和放射组学的方法，且在不同网络架构中保持性能一致性。

### 结论

RF-Deep是一种有效、轻量级且与架构无关的方法，能显著提高从CT体积中肿瘤分割的可靠性，适合临床安全部署。

### 翻译

从3D计算机断层扫描(CT)中准确分割癌变病变对于自动治疗计划和反应评估至关重要。然而，即使是结合了自监督学习(SSL)预训练转换器和卷积解码器的最先进模型，也容易受到分布外(OOD)输入的影响，产生自信但不正确的肿瘤分割，这对安全临床部署构成风险。现有的基于logit的方法存在任务特定模型偏差，而显式检测OOD的架构增强会增加参数和计算成本。因此，我们引入了一种即插即用和轻量级的后处理随机森林基础OOD检测框架RF-Deep，它利用具有有限异常值暴露的深度特征。RF-Deep通过重用预训练后微调的主干编码器的层次特征，增强了对成像变化的泛化能力，并通过从锚定到预测肿瘤分割的多个感兴趣区域提取特征，提供任务相关的OOD检测。因此，它可扩展到不同视野的图像。我们使用1,916个CT扫描，在近OOD(肺栓塞、阴性COVID-19)和远OOD(肾癌、健康胰腺)数据集上，将RF-Deep与现有的OOD检测方法进行了比较。RF-Deep在具有挑战性的近OOD数据集上实现了AUROC > 93.50，在远OOD数据集上实现了近乎完美的检测(AUROC > 99.00)，显著优于基于logit和放射组学的方法。RF-Deep在不同深度和预训练策略的网络中保持了一致的性能，证明其作为轻量级、与架构无关的方法的有效性，可提高从CT体积中肿瘤分割的可靠性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决肺癌分割任务中准确检测分布外(OOD)输入数据的问题。这个问题重要是因为即使最先进的模型也容易对OOD数据产生自信但不正确的分割结果，这对临床安全部署构成风险；深度学习模型通常在狭窄数据集上开发，难以推广到受不同成像采集、概念漂移影响的临床病例；模型会产生自信但不准确的预测，阻碍临床医生利用不确定性进行决策；过度依赖AI存在临床专家技能下降风险，增加患者伤害可能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者结合了学习型方法和过滤型方法的优点：借鉴了自监督学习(SSL)预训练技术来增强模型泛化能力；利用了特征提取和随机森林分类的现有方法；针对现有基于logit方法的模型偏差和架构增强的高计算成本问题，设计了即插即用的轻量级框架。创新性地采用肿瘤锚定方法，将检测集中在肿瘤周围区域而非整个图像，通过多尺度特征提取和异常暴露训练来区分同一解剖部位的不同病理变化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用深度学习模型的分层特征，通过随机森林分类器检测OOD数据，特别关注肿瘤周围区域而非整个图像，从而区分同一解剖部位的不同病理变化。实现流程分为四步：1)ROI提取：使用分割模型生成肿瘤分割，提取多个包含肿瘤的3D区域；2)特征提取：从编码器五个阶段提取多尺度特征，通过全局平均池聚合成扫描级特征向量；3)检测器训练：用ID和OOD特征训练随机森林，学习区分决策边界；4)在线推理：对新扫描提取ROI和特征，由随机森林生成扫描级OOD分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)肿瘤锚定方法：将检测集中在肿瘤周围区域，能区分同一解剖部位的不同病理；2)多尺度特征提取：利用分层特征捕获不同抽象级别信息；3)异常暴露训练：在训练时引入已知异常样本；4)轻量级即插即用设计：无需修改现有分割模型；5)通用性：适用于不同预训练策略和架构。相比之前工作，不同之处在于：不依赖特定任务模型偏差(优于基于logit方法)；使用深度特征而非手工特征(优于放射组学方法)；轻量级不增加参数(优于架构增强方法)；学习复杂非线性决策边界(优于基于距离方法)；专注过滤不可靠预测而非增强一致性(优于学习型方法)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RF-Deep通过结合深度特征和随机森林分类器，为肺癌分割提供了一个轻量级、即插即用的分布外检测框架，显著提高了对近OOD和远OOD病例的检测准确性，增强了临床应用的安全性。'}


### 论文摘要

Accurate segmentation of cancerous lesions from 3D computed tomography (CT) scans is essential for automated treatment planning and response assessment. However, even state-of-the-art models combining self-supervised learning (SSL) pretrained transformers with convolutional decoders are susceptible to out-of-distribution (OOD) inputs, generating confidently incorrect tumor segmentations, posing risks for safe clinical deployment. Existing logit-based methods suffer from task-specific model biases, while architectural enhancements to explicitly detect OOD increase parameters and computational costs. Hence, we introduce a plug-and-play and lightweight post-hoc random forests-based OOD detection framework called RF-Deep that leverages deep features with limited outlier exposure. RF-Deep enhances generalization to imaging variations by repurposing the hierarchical features from the pretrained-then-finetuned backbone encoder, providing task-relevant OOD detection by extracting the features from multiple regions of interest anchored to the predicted tumor segmentations. Hence, it scales to images of varying fields-of-view. We compared RF-Deep against existing OOD detection methods using 1,916 CT scans across near-OOD (pulmonary embolism, negative COVID-19) and far-OOD (kidney cancer, healthy pancreas) datasets. RF-Deep achieved AUROC > 93.50 for the challenging near-OOD datasets and near-perfect detection (AUROC > 99.00) for the far-OOD datasets, substantially outperforming logit-based and radiomics approaches. RF-Deep maintained similar performance consistency across networks of different depths and pretraining strategies, demonstrating its effectiveness as a lightweight, architecture-agnostic approach to enhance the reliability of tumor segmentation from CT volumes.

---

## 17. Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation

**论文链接:** [http://arxiv.org/abs/2512.08186v1](http://arxiv.org/abs/2512.08186v1)

**作者:** Meng Wei, Chenyang Wan, Jiaqi Peng, Xiqian Yu, Yuqiang Yang, Delin Feng, Wenzhe Cai, Chenming Zhu, Tai Wang, Jiangmiao Pang, Xihui Liu

**发布时间:** 2025-12-09

### GPT解析

### 总结

DualVLN是一种双系统视觉语言导航基础模型，通过整合高级推理与低级动作执行，解决了现有方法中的碎片化运动、高延迟和动态障碍物规避等问题。

### 背景

最近的大型视觉语言模型在视觉语言导航方面提高了泛化能力，但现有方法通常依赖于端到端管道，直接映射视觉语言输入到短时程离散动作，导致碎片化运动、高延迟和在动态环境中表现不佳。

### 目的

提出DualVLN，第一个双系统VLN基础模型，通过协同整合高级推理与低级动作执行来改进视觉语言导航性能。

### 方法

DualVLN包含两个系统：System 2是基于VLM的全局规划器，通过图像推理预测中期航点目标；System 1是轻量级多模态条件Diffusion Transformer策略，利用System 2的像素目标和特征生成轨迹。双系统设计通过解耦训练使VLM保持泛化能力，同时实现可解释有效的本地导航。

### 主要发现

DualVLN在所有VLN基准测试中均优于先前方法，真实世界实验证明其在动态环境中具有强大的长时程规划和实时适应性。

### 结论

双系统设计使复杂动态环境中的实时控制和自适应本地决策成为可能，DualVLN代表了视觉语言导航领域的重大进步。

### 翻译

尽管最近的大型视觉语言模型在视觉语言导航方面提高了泛化能力，但现有方法通常依赖于端到端管道，直接将视觉语言输入映射到短时程离散动作。此类设计通常产生碎片化的运动，导致高延迟，并且在动态障碍物规避等现实世界挑战中表现不佳。我们提出DualVLN，这是第一个双系统VLN基础模型，通过协同整合高级推理与低级动作执行。System 2是基于VLM的全局规划器，通过基于图像的推理预测中期航点目标，'缓慢思考'。System 1是轻量级、多模态条件Diffusion Transformer策略，利用System 2的显式像素目标和潜在特征生成平滑准确的轨迹，'快速行动'。双系统设计使复杂动态环境中的实时控制和自适应本地决策成为可能。通过解耦训练，VLM保持其泛化能力，同时System 1实现可解释且有效的本地导航。DualVLN在所有VLN基准测试中均优于先前方法，真实世界实验证明了在动态环境中具有强大的长时程规划和实时适应性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言导航(VLN)中现有方法依赖端到端管道直接将视觉语言输入映射到短时域离散动作的问题，这导致产生碎片化运动、高延迟，难以处理动态障碍物避让等现实挑战。这个问题很重要，因为机器人需要在复杂动态环境中高效安全地移动，而现有方法无法满足实时控制需求，缺乏层次化决策的明确协调，难以适应真实世界的动态变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了人类导航行为的启发——人类会先进行高层次推理确定方向('ground slow')，然后快速执行具体动作('move fast')。基于此，他们将VLN管道解耦为两个互补系统：System 2作为基于VLM的慢速全局规划器，System 1作为轻量级扩散变换器策略模型。作者还参考了现有的双系统架构思想，但将其扩展到支持长距离指令跟随和跨建筑导航的场景，并引入了显式像素目标和隐式潜在目标的概念增强系统间信息流动。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是解耦高层次的语义推理与低层次的动作执行，通过'慢思考、快行动'的双系统设计实现更鲁棒、高效的导航。整体流程：System 2(2Hz)接收RGB图像和指令，通过自导视图调整后预测像素目标；生成包含语言、视觉和目标信息的特征序列，附加可学习潜在查询形成潜在目标；System 1(30Hz)接收潜在目标和当前RGB输入，通过扩散变换器生成32个密集路径点；两个系统异步运行，System 2提供长期规划，System 1实现实时控制，共同产生平滑连续的导航轨迹。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双系统架构，首次明确解耦高层次推理与低层次控制；2)异步设计，System 2慢速推理(2Hz)，System 1快速控制(30Hz)；3)结合显式像素目标和隐式潜在目标，既利用VLM空间定位能力又提供自适应指导；4)分阶段训练保留VLM泛化能力；5)提出Social-VLN新基准评估动态环境中的社交意识。相比之前工作，不同在于现有方法多采用端到端映射短时域动作，其他双系统架构主要关注桌面任务，而DualVLN首次支持长距离指令跟随和跨建筑导航的异步双系统架构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出DualVLN双系统视觉语言导航模型，通过解耦高层次推理与低层次控制、结合显式与隐式目标，实现了在复杂动态环境中的鲁棒高效导航，并在多个基准测试和真实世界实验中取得了最先进的结果。'}


### 论文摘要

While recent large vision-language models (VLMs) have improved generalization in vision-language navigation (VLN), existing methods typically rely on end-to-end pipelines that map vision-language inputs directly to short-horizon discrete actions. Such designs often produce fragmented motions, incur high latency, and struggle with real-world challenges like dynamic obstacle avoidance. We propose DualVLN, the first dual-system VLN foundation model that synergistically integrates high-level reasoning with low-level action execution. System 2, a VLM-based global planner, "grounds slowly" by predicting mid-term waypoint goals via image-grounded reasoning. System 1, a lightweight, multi-modal conditioning Diffusion Transformer policy, "moves fast" by leveraging both explicit pixel goals and latent features from System 2 to generate smooth and accurate trajectories. The dual-system design enables robust real-time control and adaptive local decision-making in complex, dynamic environments. By decoupling training, the VLM retains its generalization, while System 1 achieves interpretable and effective local navigation. DualVLN outperforms prior methods across all VLN benchmarks and real-world experiments demonstrate robust long-horizon planning and real-time adaptability in dynamic environments.

---

## 18. Unveiling Latent Knowledge in Chemistry Language Models through Sparse Autoencoders

**论文链接:** [http://arxiv.org/abs/2512.08077v1](http://arxiv.org/abs/2512.08077v1)

**作者:** Jaron Cohen, Alexander G. Hasson, Sara Tanovic

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究扩展稀疏自编码器技术，揭示和检查化学语言模型(CLMs)中的可解释特征，应用于Foundation Models for Materials (FM4M) SMI-TED化学基础模型，提取语义上有意义的潜在特征并分析其激活模式。

### 背景

自机器学习出现以来，可解释性一直是一个持续的挑战，随着生成模型在药物和材料发现等高风险应用中的使用，这一挑战变得更加紧迫。大语言模型架构的最新进展产生了具有分子属性预测和分子生成能力的化学语言模型，但这些模型如何在内部表示化学知识仍然知之甚少。

### 目的

扩展稀疏自编码器技术以揭示和检查化学语言模型中的可解释特征，分析这些模型如何编码化学知识。

### 方法

将稀疏自编码器技术应用于化学语言模型，特别是Foundation Models for Materials (FM4M) SMI-TED化学基础模型，提取语义上有意义的潜在特征，并分析不同分子数据集上的激活模式。

### 主要发现

这些模型编码了丰富的化学概念景观，特定的潜在特征与不同的化学知识领域之间存在相关性，包括结构基序、物理化学性质和药物类别。

### 结论

该方法为揭示专注于化学的AI系统中的潜在知识提供了一个可推广的框架，这项工作对基础理解和实际部署都有影响，有潜力加速计算化学研究。

### 翻译

自从机器学习出现以来，可解释性一直是一个持续的挑战，随着生成模型在药物和材料发现等高风险应用中的支持，这一挑战变得越来越紧迫。大语言模型架构的最新进展产生了在分子属性预测和分子生成方面具有令人印象深刻能力的化学语言模型。然而，这些模型如何在内部表示化学知识仍然知之甚少。在这项工作中，我们将稀疏自编码器技术扩展到揭示和检查化学语言模型中的可解释特征。将我们的方法应用于Foundation Models for Materials (FM4M) SMI-TED化学基础模型，我们提取语义上有意义的潜在特征，并分析它们在不同分子数据集上的激活模式。我们的研究结果表明，这些模型编码了丰富的化学概念景观。我们确定了特定的潜在特征与不同的化学知识领域之间的相关性，包括结构基序、物理化学性质和药物类别。我们的方法为揭示专注于化学的AI系统中的潜在知识提供了一个可推广的框架。这项工作对基础理解和实际部署都有影响；有可能加速计算化学研究。


### 论文摘要

Since the advent of machine learning, interpretability has remained a persistent challenge, becoming increasingly urgent as generative models support high-stakes applications in drug and material discovery. Recent advances in large language model (LLM) architectures have yielded chemistry language models (CLMs) with impressive capabilities in molecular property prediction and molecular generation. However, how these models internally represent chemical knowledge remains poorly understood. In this work, we extend sparse autoencoder techniques to uncover and examine interpretable features within CLMs. Applying our methodology to the Foundation Models for Materials (FM4M) SMI-TED chemistry foundation model, we extract semantically meaningful latent features and analyse their activation patterns across diverse molecular datasets. Our findings reveal that these models encode a rich landscape of chemical concepts. We identify correlations between specific latent features and distinct domains of chemical knowledge, including structural motifs, physicochemical properties, and pharmacological drug classes. Our approach provides a generalisable framework for uncovering latent knowledge in chemistry-focused AI systems. This work has implications for both foundational understanding and practical deployment; with the potential to accelerate computational chemistry research.

---

## 19. VFM-VLM: Vision Foundation Model and Vision Language Model based Visual Comparison for 3D Pose Estimation

**论文链接:** [http://arxiv.org/abs/2512.07215v2](http://arxiv.org/abs/2512.07215v2)

**作者:** Md Selim Sarowar, Sungho Kim

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文对基于CLIP和基于DINOv2的方法在3D手部抓取物体姿态估计方面进行了全面的视觉比较，评估了它们在6D物体姿态估计任务上的表现，展示了各自的互补优势。

### 背景

视觉基础模型(VFMs)和视觉语言模型(VLMs)通过提供丰富的语义和几何表示彻底改变了计算机视觉领域。

### 目的

对基于CLIP和基于DINOv2的方法在3D手部抓取物体姿态估计方面进行全面的视觉比较。

### 方法

评估这两种模型在6D物体姿态估计任务上的表现，并通过大量实验展示它们的互补优势。

### 主要发现

CLIP在通过语言锚定的语义理解方面表现出色，而DINOv2提供了更优的密集几何特征；基于CLIP的方法实现了更好的语义一致性，而基于DINOv2的方法展示了具有增强几何精度的竞争性性能。

### 结论

该分析为选择适合机器人操作和抓取、拾取应用的视觉模型提供了见解。

### 翻译

视觉基础模型(VFMs)和视觉语言模型(VLMs)通过提供丰富的语义和几何表示彻底改变了计算机视觉。本文对基于CLIP和基于DINOv2的方法在3D手部抓取物体姿态估计场景中进行了全面的视觉比较。我们在6D物体姿态估计任务上评估了这两种模型，并展示了它们的互补优势：CLIP通过语言锚定在语义理解方面表现出色，而DINOv2提供了更优的密集几何特征。通过对基准数据集的大量实验，我们表明基于CLIP的方法实现了更好的语义一致性，而基于DINOv2的方法展示了具有增强几何精度的竞争性性能。我们的分析为选择适合机器人操作和抓取、拾取应用的视觉模型提供了见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要想解决的问题是评估和比较两种视觉基础模型（CLIP和DINOv2）在3D姿态估计任务上的性能表现，特别是在手部物体抓取场景中的6D物体姿态估计。这个问题在现实中很重要，因为3D姿态估计是机器人操作和抓取的关键技术，对于实现机器人与物体的交互至关重要。在研究中也很重要，因为现有的深度学习方法虽然数学上正确，但缺乏实际应用所需的上下文知识，而不同基础模型各有优势，了解它们的优缺点有助于选择适合特定应用的视觉模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先定义了问题：给定一个包含感兴趣物体的RGB图像，估计该物体的6D姿态。然后选择了两种不同的视觉基础模型：CLIP（基于对比学习的双编码器架构）和DINOv2（基于自监督学习的自蒸馏框架）。作者为每种模型设计了专门的架构：CLIP-Based架构利用语义理解进行跨模态融合，而DINOv2-Based架构强调密集几何特征。作者借鉴了现有工作，包括使用CLIP和DINOv2这两种已有的视觉基础模型，应用了PnP-RANSAC等传统计算机视觉技术进行几何推理，以及采用ICP等几何精炼方法和标准的6D姿态评估指标。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是比较两种不同类型的视觉基础模型（CLIP和DINOv2）在3D姿态估计任务上的表现，探索语义理解与几何精度之间的权衡。整体实现流程包括：1) 数据准备，使用多个数据集的组合；2) CLIP-Based方法流程，包括特征提取、跨模态融合和姿态回归；3) DINOv2-Based方法流程，包括密集特征提取、关键点检测、几何推理和精炼；4) 评估，结合定量评估（ADD、ADD-S等指标）和定性的视觉分析。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次对基于CLIP和DINOv2的架构在6D物体姿态估计任务进行系统比较；2) 揭示了两种模型的互补优势：CLIP在语义理解方面表现出色，DINOv2提供优越的几何精度；3) 结合定量评估和视觉分析，提供多角度评估；4) 为机器人操作应用提供了选择合适视觉模型的见解；5) 提出了结合两种模型优势的混合架构的潜在方向。相比之前的工作，这篇论文不仅比较了不同模型，还深入分析了它们在不同场景下的表现和适用性，并提供了实际应用指导。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统比较CLIP和DINOv2两种视觉基础模型在3D姿态估计任务上的表现，揭示了它们各自的语义理解和几何精度优势，为机器人操作应用提供了选择合适视觉模型的指导。'}


### 论文摘要

Vision Foundation Models (VFMs) and Vision Language Models (VLMs) have revolutionized computer vision by providing rich semantic and geometric representations. This paper presents a comprehensive visual comparison between CLIP based and DINOv2 based approaches for 3D pose estimation in hand object grasping scenarios. We evaluate both models on the task of 6D object pose estimation and demonstrate their complementary strengths: CLIP excels in semantic understanding through language grounding, while DINOv2 provides superior dense geometric features. Through extensive experiments on benchmark datasets, we show that CLIP based methods achieve better semantic consistency, while DINOv2 based approaches demonstrate competitive performance with enhanced geometric precision. Our analysis provides insights for selecting appropriate vision models for robotic manipulation and grasping, picking applications.

---

## 20. Evaluating and Preserving High-level Fidelity in Super-Resolution

**论文链接:** [http://arxiv.org/abs/2512.07037v2](http://arxiv.org/abs/2512.07037v2)

**作者:** Josep M. Rocafort, Shaolin Su, Alexandra Gomez-Villa, Javier Vazquez-Corral

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文研究了图像超分辨率(SR)模型的高级别保真度测量问题，提出了一个新的评估标准，构建了首个带有保真度分数的标注数据集，并通过实验验证了该标准在模型评估和优化中的价值。

### 背景

现代图像超分辨率模型在重建细节和提供视觉上令人满意的输出方面取得了显著效果，但其过强的生成能力有时会产生'幻觉'，改变图像内容，这种现象容易被人类识别但在现有低级别图像质量指标中研究不足。

### 目的

建立测量SR模型高级别保真度的重要性作为补充标准，构建首个带保真度分数的标注数据集，评估SOTA SR模型在保持高级别保真度方面的表现，分析现有指标与保真度的相关性，展示基础模型的优势，并通过保真度反馈微调改进模型性能。

### 方法

构建首个带有不同SR模型保真度分数的标注数据集，评估SOTA SR模型在保持高级别保真度方面的表现，分析现有图像质量指标与保真度测量的相关性，使用基础模型处理高级别任务，通过基于保真度反馈微调SR模型来改进性能。

### 主要发现

高级别保真度是评估SR模型可靠性的重要补充标准；现有SOTA SR模型在保持高级别保真度方面存在不足；现有图像质量指标与保真度测量相关性有限；基础模型能更好地处理高级别任务；通过保真度反馈微调可以同时提高语义保真度和感知质量。

### 结论

提出的高级别保真度标准在模型评估和优化方面具有潜在价值，作者将在论文接受后发布数据集、代码和模型。

### 翻译

最近的图像超分辨率(SR)模型在重建细节和提供视觉上令人满意的输出方面取得了显著效果。然而，过强的生成能力有时会产生'幻觉'，从而改变图像内容，尽管获得了高视觉质量。这种高级别的改变容易被人类识别，但在现有的低级别图像质量指标中研究不足。在本文中，我们确立了测量SR模型高级别保真度的重要性，作为揭示生成式SR模型可靠性的补充标准。我们构建了第一个带有不同SR模型保真度分数的标注数据集，并评估了最先进的(SOTA)SR模型在保持高级别保真度方面的实际表现。基于该数据集，我们分析了现有图像质量指标与保真度测量的相关性，并进一步展示了这种高级别任务可以通过基础模型更好地解决。最后，通过基于我们的保真度反馈微调SR模型，我们展示了语义保真度和感知质量都可以得到改善，证明了我们提出的标准在模型评估和优化中的潜在价值。我们将在论文接受后发布数据集、代码和模型。


### 论文摘要

Recent image Super-Resolution (SR) models are achieving impressive effects in reconstructing details and delivering visually pleasant outputs. However, the overpowering generative ability can sometimes hallucinate and thus change the image content despite gaining high visual quality. This type of high-level change can be easily identified by humans yet not well-studied in existing low-level image quality metrics. In this paper, we establish the importance of measuring high-level fidelity for SR models as a complementary criterion to reveal the reliability of generative SR models. We construct the first annotated dataset with fidelity scores from different SR models, and evaluate how state-of-the-art (SOTA) SR models actually perform in preserving high-level fidelity. Based on the dataset, we then analyze how existing image quality metrics correlate with fidelity measurement, and further show that this high-level task can be better addressed by foundation models. Finally, by fine-tuning SR models based on our fidelity feedback, we show that both semantic fidelity and perceptual quality can be improved, demonstrating the potential value of our proposed criteria, both in model evaluation and optimization. We will release the dataset, code, and models upon acceptance.

---

## 21. Delay-Oriented Distributed Scheduling with TransGNN

**论文链接:** [http://arxiv.org/abs/2512.08799v1](http://arxiv.org/abs/2512.08799v1)

**作者:** Boxuan Wen, Junyu Luo

**发布时间:** 2025-12-09

**备注:** 10 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种基于Transformer GNN的延迟导向分布式调度框架，用于解决无线多跳网络中的传输延迟问题。模型使用基于注意力的图编码器生成自适应的每链路效用分数，并通过局部贪婪求解器构建可行的独立链路集，确保分布式和无冲突的调度。

### 背景

无线多跳网络中的传输延迟最小化是一个基本但具有挑战性的任务，因为干扰、队列动态和分布式控制之间复杂的耦合关系。传统的调度算法主要优化吞吐量，但通常存在高延迟问题，尤其是在异构或动态变化的拓扑结构中。

### 目的

解决传统调度算法在高延迟方面的问题，以及常规图卷积网络(GCN)在建模冲突图中长程依赖性方面的局限性。

### 方法

提出了一种基于Transformer GNN的延迟导向分布式调度框架。该模型采用基于注意力的图编码器生成反映队列积压和干扰强度的自适应每链路效用分数，并通过局部贪婪求解器(LGS)构建可行的独立链路集。

### 主要发现

传统的基于最大权重或队列长度的调度算法虽然优化了吞吐量，但在异构或动态拓扑中存在高延迟问题。常规图卷积网络(GCN)由于其局部聚合机制，无法建模冲突图中的长程依赖性。

### 结论

基于Transformer GNN的延迟导向分布式调度框架能够有效解决无线多跳网络中的传输延迟问题，通过基于注意力的图编码器和局部贪婪求解器实现分布式和无冲突的调度。

### 翻译

在无线多跳网络中最小化传输延迟是一个基本但具有挑战性的任务，因为干扰、队列动态和分布式控制之间复杂的耦合关系。传统的调度算法，如最大权重或基于队列长度的策略，主要旨在优化吞吐量，但通常遭受高延迟，特别是在异构或动态变化的拓扑中。最近基于学习的方法，特别是那些采用图神经网络(GNN)的方法，在捕获空间干扰结构方面显示出潜力。然而，传统的图卷积网络(GCN)仍然受到其局部聚合机制的限制，以及它们在冲突图中建模长程依赖性的能力不足。为了应对这些挑战，本文提出了一种基于Transformer GNN的延迟导向分布式调度框架。所提出的模型采用基于注意力的图编码器生成反映队列积压和干扰强度的自适应每链路效用分数。然后，局部贪婪求解器(LGS)利用这些效用分数构建可行的独立链路集用于传输，确保分布式和无冲突的调度。


### 论文摘要

Minimizing transmission delay in wireless multi-hop networks is a fundamental yet challenging task due to the complex coupling among interference, queue dynamics, and distributed control. Traditional scheduling algorithms, such as max-weight or queue-length-based policies, primarily aim to optimize throughput but often suffer from high latency, especially in heterogeneous or dynamically changing topologies. Recent learning-based approaches, particularly those employing Graph Neural Networks (GNNs), have shown promise in capturing spatial interference structures. However, conventional Graph Convolutional Networks (GCNs) remain limited by their local aggregation mechanism and their inability to model long-range dependencies within the conflict graph. To address these challenges, this paper proposes a delay-oriented distributed scheduling framework based on Transformer GNN. The proposed model employs an attention-based graph encoder to generate adaptive per-link utility scores that reflect both queue backlog and interference intensity. A Local Greedy Solver (LGS) then utilizes these utilities to construct a feasible independent set of links for transmission, ensuring distributed and conflict-free scheduling.

---

## 22. Learning and Editing Universal Graph Prompt Tuning via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.08763v1](http://arxiv.org/abs/2512.08763v1)

**作者:** Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Hewei Wang, Yijie Li, Edith C. H. Ngai

**发布时间:** 2025-12-09

**备注:** Accepted by KDD 2026

### GPT解析

### 总结

本文提出了一种名为LEAP的新型图提示调整方法，通过向所有节点添加提示并使用强化学习来选择和编辑提示，既保留了通用图提示调整的理论基础，又实现了更理想的提示效果。

### 背景

早期图提示调整方法依赖于GNN的任务特定设计，限制了跨预训练策略的适应性。通用图提示调整在输入图特征空间中操作，理论上可实现任何提示函数的等效效果。最近提出的基于选择性节点的图提示调整可能损害了通用图提示调整的理论基础。

### 目的

加强通用图提示调整的理论基础，证明向所有节点添加提示是实现图提示普遍性的必要条件，同时追求更理想的提示效果。

### 方法

提出LEAP(学习和编辑通用图提示调整)模型，首先构建基本通用图提示以保留理论基础，然后使用参与者-批评者强化学习来选择节点和编辑提示。

### 主要发现

在各种预训练策略下的全样本和少样本场景中，针对图级和节点级任务的实验表明，LEAP持续优于微调和其他基于提示的方法。

### 结论

向所有节点添加提示是实现图提示普遍性的必要条件，LEAP方法在保留通用图提示调整理论基础的同时，实现了更理想的性能表现。

### 翻译

早期的图提示调整方法依赖于图神经网络(GNNs)的任务特定设计，限制了它们在不同预训练策略中的适应性。相比之下，另一个有前途的研究方向是通用图提示调整，它在输入图的特征空间中直接操作，建立了理论基础，证明通用图提示调整理论上可以实现任何提示函数的等效效果，消除了对特定预训练策略的依赖。最近的工作提出了基于选择性节点的图提示调整，以追求更理想的提示。然而，我们认为基于选择性节点的图提示调整不可避免地损害了通用图提示调整的理论基础。在本文中，我们通过引入更严格的约束来加强通用图提示调整的理论基础，证明向所有节点添加提示是实现图提示普遍性的必要条件。为此，我们提出了一种新颖的模型和范式——学习和编辑通用图提示调整(LEAP)，它在保留通用图提示调整理论基础的同时追求更理想的提示。具体而言，我们首先构建基本的通用图提示以保留理论基础，然后采用参与者-批评者强化学习来选择节点和编辑提示。在各种预训练策略下，针对图级和节点级任务的广泛实验(包括全样本和少样本场景)表明，LEAP持续优于微调和其他基于提示的方法。


### 论文摘要

Early graph prompt tuning approaches relied on task-specific designs for Graph Neural Networks (GNNs), limiting their adaptability across diverse pre-training strategies. In contrast, another promising line of research has investigated universal graph prompt tuning, which operates directly in the input graph's feature space and builds a theoretical foundation that universal graph prompt tuning can theoretically achieve an equivalent effect of any prompting function, eliminating dependence on specific pre-training strategies. Recent works propose selective node-based graph prompt tuning to pursue more ideal prompts. However, we argue that selective node-based graph prompt tuning inevitably compromises the theoretical foundation of universal graph prompt tuning. In this paper, we strengthen the theoretical foundation of universal graph prompt tuning by introducing stricter constraints, demonstrating that adding prompts to all nodes is a necessary condition for achieving the universality of graph prompts. To this end, we propose a novel model and paradigm, Learning and Editing Universal GrAph Prompt Tuning (LEAP), which preserves the theoretical foundation of universal graph prompt tuning while pursuing more ideal prompts. Specifically, we first build the basic universal graph prompts to preserve the theoretical foundation and then employ actor-critic reinforcement learning to select nodes and edit prompts. Extensive experiments on graph- and node-level tasks across various pre-training strategies in both full-shot and few-shot scenarios show that LEAP consistently outperforms fine-tuning and other prompt-based approaches.

---

## 23. RF sensing with dense IoT network graphs: An EM-informed analysis

**论文链接:** [http://arxiv.org/abs/2512.08746v1](http://arxiv.org/abs/2512.08746v1)

**作者:** Federica Fieramosca, Vittorio Rampa, Michele D'Amico, Stefano Savazzi

**发布时间:** 2025-12-09

**备注:** accepted to IEEE Internet of Things Journal

### GPT解析

### 总结

RF传感利用无线网络中电磁波的特性来捕捉环境信息，如人和物体的存在和移动，实现被动定位和视觉应用。本文研究了密集网络中RF传感系统的理论界限，并提出了一种基于深度图神经网络的检测方法。

### 背景

射频传感在研究、标准化和行业中日益受到关注，特别是在物联网应用方面具有潜力。通过利用无线网络中电磁波的特性，RF传感能够捕捉环境和物体的信息。

### 目的

研究密集网络中RF传感系统的准确度和分辨率的理论界限，分析同时存在的人数的可区分性限制，并探索这些限制如何取决于各种因素。

### 方法

采用电磁模型预测各种场景中的身体阻挡效应；提出一种基于接收信号强度样本的深度图神经网络，这些样本被结构化为密集图，节点代表天线，边代表无线电链路。

### 主要发现

分析了同时对存在人数的理论限制，这些限制取决于无线电链路数量、监控区域大小和受试者身体尺寸等因素。这些界限能够在网络预部署阶段预测系统性能。

### 结论

室内案例研究的结果表明该方法的有效性，并确认了模型在网络设计阶段的预测潜力。

### 翻译

射频传感在研究、标准化和行业中日益受到关注，特别是在物联网应用方面具有潜力。通过利用无线网络中使用的电磁波特性，RF传感捕捉环境和物体的存在和移动信息，实现被动定位和视觉应用。本文研究了密集网络中RF传感系统的准确度和分辨率的理论界限。它采用电磁模型预测各种场景中的身体阻挡效应。为了检测人体运动，本文提出了一种深度图神经网络，该网络基于从电磁模型生成的接收信号强度样本进行训练。这些样本被结构化为密集图，节点代表天线，边代表无线电链路。本文聚焦于识别监控区域内同时存在的人数问题，分析了可区分人数的理论极限，探索了这些极限如何取决于无线电链路数量、监控区域大小和受试者身体尺寸等因素。这些界限能够在网络预部署阶段预测系统性能。本文还展示了一个室内案例研究的结果，证明了该方法的有效性，并确认了模型在网络设计阶段的预测潜力。


### 论文摘要

Radio Frequency (RF) sensing is attracting interest in research, standardization, and industry, especially for its potential in Internet of Things (IoT) applications. By leveraging the properties of the ElectroMagnetic (EM) waves used in wireless networks, RF sensing captures environmental information such as the presence and movement of people and objects, enabling passive localization and vision applications. This paper investigates the theoretical bounds on accuracy and resolution for RF sensing systems within dense networks. It employs an EM model to predict the effects of body blockage in various scenarios. To detect human movements, the paper proposes a deep graph neural network, trained on Received Signal Strength (RSS) samples generated from the EM model. These samples are structured as dense graphs, with nodes representing antennas and edges as radio links. Focusing on the problem of identifying the number of human subjects co-present in a monitored area over time, the paper analyzes the theoretical limits on the number of distinguishable subjects, exploring how these limits depend on factors such as the number of radio links, the size of the monitored area and the subjects physical dimensions. These bounds enable the prediction of the system performance during network pre-deployment stages. The paper also presents the results of an indoor case study, which demonstrate the effectiveness of the approach and confirm the model's predictive potential in the network design stages.

---

## 24. A Hybrid Model for Stock Market Forecasting: Integrating News Sentiment and Time Series Data with Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.08567v1](http://arxiv.org/abs/2512.08567v1)

**作者:** Nader Sadek, Mirette Moawad, Christina Naguib, Mariam Elzahaby

**发布时间:** 2025-12-09

**DOI:** 10.34190/icair.5.1.4294

**备注:** 11 pages, 6 figures. Published in the Proceedings of the 5th International Conference on Artificial Intelligence Research (ICAIR 2025). Published version available at: https://papers.academic-conferences.org/index.php/icair/article/view/4294

### GPT解析

### 总结

该研究提出了一种结合公司新闻文章和历史股票数据的多模态方法，使用图神经网络进行股票市场预测，相比传统LSTM模型取得了更好的预测效果。

### 背景

股票市场预测是金融领域的长期挑战，准确的预测支持明智的投资决策。传统模型主要依赖历史价格，但研究表明金融新闻可以提供有用的外部信号。

### 目的

研究一种多模态方法，整合公司的新闻文章和历史股票数据，以提高股票市场预测性能。

### 方法

比较图神经网络模型与基准LSTM模型。使用LSTM编码历史数据，语言模型嵌入新闻标题，形成异构图节点，通过GraphSAGE捕获文章、公司和行业间的交互。评估二元变化方向和基于显著性的两个预测目标。

### 主要发现

在美国股票和彭博数据集上，GNN模型优于LSTM基线，在第一个目标上达到53%的准确率，在第二个目标上提高4%的精确度。与更多新闻相关的公司预测准确率更高，且新闻标题比完整文章包含更强的预测信号。

### 结论

整合新闻数据与历史股票数据的多模态方法可以提高股票市场预测准确性，特别是使用图神经网络模型时，简洁的新闻摘要在短期市场反应中起重要作用。

### 翻译

股票市场预测是金融领域中一个长期的挑战，因为准确的预测支持明智的投资决策。传统模型主要依赖历史价格，但最近的研究表明金融新闻可以提供有用的外部信号。本文研究了一种多模态方法，整合公司的新闻文章与其历史股票数据以提高预测性能。我们将图神经网络模型与基准LSTM模型进行比较。每个公司的历史数据使用LSTM编码，而新闻标题通过语言模型嵌入。这些嵌入形成异构图中的节点，使用GraphSAGE捕获文章、公司和行业之间的交互。我们评估两个目标：二元变化方向标签和基于显著性的标签。在美国股票和彭博数据集上的实验表明，GNN优于LSTM基线，在第一个目标上达到53%的准确率，在第二个目标上提高4%的精确度。结果还表明，与更多新闻相关的公司产生更高的预测准确率。此外，标题比完整文章包含更强的预测信号，表明简洁的新闻摘要在短期市场反应中起重要作用。


### 论文摘要

Stock market prediction is a long-standing challenge in finance, as accurate forecasts support informed investment decisions. Traditional models rely mainly on historical prices, but recent work shows that financial news can provide useful external signals. This paper investigates a multimodal approach that integrates companies' news articles with their historical stock data to improve prediction performance. We compare a Graph Neural Network (GNN) model with a baseline LSTM model. Historical data for each company is encoded using an LSTM, while news titles are embedded with a language model. These embeddings form nodes in a heterogeneous graph, and GraphSAGE is used to capture interactions between articles, companies, and industries. We evaluate two targets: a binary direction-of-change label and a significance-based label. Experiments on the US equities and Bloomberg datasets show that the GNN outperforms the LSTM baseline, achieving 53% accuracy on the first target and a 4% precision gain on the second. Results also indicate that companies with more associated news yield higher prediction accuracy. Moreover, headlines contain stronger predictive signals than full articles, suggesting that concise news summaries play an important role in short-term market reactions.

---

## 25. Solving Over-Smoothing in GNNs via Nonlocal Message Passing: Algebraic Smoothing and Depth Scalability

**论文链接:** [http://arxiv.org/abs/2512.08475v1](http://arxiv.org/abs/2512.08475v1)

**作者:** Weiqi Guan, Junlin He

**发布时间:** 2025-12-09

**备注:** 18 pages, 4 figures

### GPT解析

### 总结

本研究探讨了Layer Normalization (LN)放置与过平滑现象之间的关系，提出了一种解决Pre-LN和Post-LN架构各自缺点的新方法。

### 背景

Layer Normalization (LN)放置与过平滑现象之间的关系尚未得到充分探索。Pre-LN架构避免了过平滑但受到深度诅咒的影响，而Post-LN架构绕过了深度诅咒但经历了过平滑。

### 目的

解决Pre-LN和Post-LN架构各自的缺点，开发一种既能避免过平滑又能绕过深度诅咒的方法。

### 方法

提出一种基于Post-LN的新方法，通过诱导代数平滑来防止过平滑，同时不受深度诅咒的影响。

### 主要发现

该方法在五个基准测试上表现出色，支持多达256层的更深层网络，提高了性能，且不需要额外参数。

### 结论

通过理论分析、原则性解决方案和实证验证，本研究解决了LN放置与过平滑现象之间的关系问题，为构建更深、更有效的GNN提供了新思路。

### 翻译

Layer Normalization (LN)放置与过平滑现象之间的关系仍未得到充分探索。我们确定了一个关键困境：Pre-LN架构避免了过平滑但受到深度诅咒的影响，而Post-LN架构绕过了深度诅咒但经历了过平滑。为了解决这个问题，我们提出了一种基于Post-LN的新方法，通过诱导代数平滑来防止过平滑，同时不受深度诅咒的影响。在五个基准测试上的实证结果表明，我们的方法支持更深的网络（多达256层）并提高了性能，不需要额外参数。主要贡献包括：LN动力学及其对过平滑和深度诅咒影响的理论分析；一种诱导代数平滑并避免过平滑和深度诅咒的参数高效方法；广泛的经验验证，证明该方法在更深GNNs中的有效性。


### 论文摘要

The relationship between Layer Normalization (LN) placement and the over-smoothing phenomenon remains underexplored. We identify a critical dilemma: Pre-LN architectures avoid over-smoothing but suffer from the curse of depth, while Post-LN architectures bypass the curse of depth but experience over-smoothing.   To resolve this, we propose a new method based on Post-LN that induces algebraic smoothing, preventing over-smoothing without the curse of depth. Empirical results across five benchmarks demonstrate that our approach supports deeper networks (up to 256 layers) and improves performance, requiring no additional parameters.   Key contributions:   Theoretical Characterization: Analysis of LN dynamics and their impact on over-smoothing and the curse of depth.   A Principled Solution: A parameter-efficient method that induces algebraic smoothing and avoids over-smoothing and the curse of depth.   Empirical Validation: Extensive experiments showing the effectiveness of the method in deeper GNNs.

---

## 26. Enhancing Explainability of Graph Neural Networks Through Conceptual and Structural Analyses and Their Extensions

**论文链接:** [http://arxiv.org/abs/2512.08344v1](http://arxiv.org/abs/2512.08344v1)

**作者:** Tien Cuong Bui

**发布时间:** 2025-12-09

**备注:** 157 pages, Doctoral dissertation at Seoul National University (submitted in 2024.08 to SNU library, slightly updated in 2025.11 for open digital version)

### GPT解析

### 总结

图神经网络已成为处理图结构数据的有力工具，但其复杂性阻碍了决策过程的解释。当前可解释AI方法难以理清图中的复杂关系和交互，现有的事后解释方法需要额外计算资源且可靠性较低，而可解释模型的泛化能力存在问题。

### 背景

图神经网络在众多应用中被广泛采用，证明了其价值。然而，这些方法的复杂性往往阻碍了对决策过程的理解。当前可解释人工智能方法难以理清图中的复杂关系和交互。

### 目的

开发一种专门针对基于图的机器学习的新型XAI框架，提供适应性强的、计算效率高的GNN解释。

### 方法

提出一种超越单个特征分析的框架，捕捉图结构如何影响预测，旨在解决事后解释方法计算资源需求高和可解释模型泛化能力差的问题。

### 主要发现

摘要中未明确提及具体研究发现，主要介绍了研究动机和目标。

### 结论

需要开发新型XAI框架来解决现有方法的局限性，提供更高效、更可靠的图神经网络解释方法。

### 翻译

图神经网络已成为建模和分析图结构数据的强大工具。在众多应用中的广泛采用证明了这些模型的价值。然而，这些方法的复杂性往往阻碍了对决策过程的理解。当前的可解释人工智能方法难以理清图中的复杂关系和交互。几种方法试图通过事后方法或自解释设计来弥合这一差距。它们大多侧重于图结构分析，以确定与预测结果相关的基本模式。虽然事后解释方法具有适应性，但它们需要额外的计算资源，并且由于对模型内部工作原理的访问有限，可能可靠性较低。相反，可解释模型可以提供即时解释，但它们在不同场景中的泛化能力仍然是一个主要问题。为解决这些不足，本论文寻求开发一种专门针对基于图的机器学习的新型XAI框架。所提出的框架旨在为GNN提供适应性强、计算效率高的解释，超越单个特征分析，捕捉图结构如何影响预测。


### 论文摘要

Graph Neural Networks (GNNs) have become a powerful tool for modeling and analyzing data with graph structures. The wide adoption in numerous applications underscores the value of these models. However, the complexity of these methods often impedes understanding their decision-making processes. Current Explainable AI (XAI) methods struggle to untangle the intricate relationships and interactions within graphs. Several methods have tried to bridge this gap via a post-hoc approach or self-interpretable design. Most of them focus on graph structure analysis to determine essential patterns that correlate with prediction outcomes. While post-hoc explanation methods are adaptable, they require extra computational resources and may be less reliable due to limited access to the model's internal workings. Conversely, Interpretable models can provide immediate explanations, but their generalizability to different scenarios remains a major concern. To address these shortcomings, this thesis seeks to develop a novel XAI framework tailored for graph-based machine learning. The proposed framework aims to offer adaptable, computationally efficient explanations for GNNs, moving beyond individual feature analysis to capture how graph structure influences predictions.

---

## 27. gHAWK: Local and Global Structure Encoding for Scalable Training of Graph Neural Networks on Knowledge Graphs

**论文链接:** [http://arxiv.org/abs/2512.08274v1](http://arxiv.org/abs/2512.08274v1)

**作者:** Humera Sabir, Fatima Farooq, Ashraf Aboulnaga

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了gHAWK，一种新型的可扩展图神经网络训练框架，用于处理大型知识图谱。通过预计算节点的结构特征，解决了传统消息传递GNN在处理大型知识图谱时的可扩展性问题，提高了训练效率并提升了模型准确性。

### 背景

知识图谱是结构化、异构数据的丰富来源，支持广泛的应用。利用这些数据的一种常见方法是训练图神经网络在知识图谱上。然而，现有的消息传递GNN难以扩展到大型知识图谱，因为它们依赖于迭代消息传递过程来学习图结构，这种方法效率低下，特别是在小批量训练中，节点只能看到其邻域的部分视图。

### 目的

解决现有消息传递GNN在大型知识图谱上难以扩展的问题，提出一种新的可扩展GNN训练框架，以提高训练效率、减少内存使用并改善模型准确性。

### 方法

gHAWK引入了一个预处理步骤，计算：(a)Bloom过滤器以紧凑地编码局部邻域结构；(b)TransE嵌入以表示节点在图中的全局位置。这些特征随后与任何领域特定特征(如文本嵌入)融合，生成节点特征向量，可整合到任何GNN技术中。通过在消息传递训练中增强结构先验，gHAWK显著减少了内存使用，加速了收敛，并提高了模型准确性。

### 主要发现

在Open Graph Benchmark的大型数据集上进行的广泛实验表明，gHAWK在节点属性预测和链接预测任务上实现了最先进的准确性和更低的训练时间，在三个图的OGB排行榜上名列前茅。

### 结论

gHAWK通过预计算结构特征并融合领域特定特征，有效地解决了传统GNN在大型知识图谱上的可扩展性问题，显著提高了训练效率和模型性能，在各种任务上取得了最先进的结果。

### 翻译

知识图谱是结构化、异构数据的丰富来源，支持广泛的应用。利用这些数据的一种常见方法是训练图神经网络在知识图谱上。然而，现有的消息传递GNN难以扩展到大型知识图谱，因为它们依赖于迭代消息传递过程来学习图结构，这种方法效率低下，特别是在小批量训练中，节点只能看到其邻域的部分视图。在本文中，我们解决了这个问题并提出了gHAWK，一种新型的可扩展GNN训练框架，用于大型知识图谱。关键思想是在GNN训练开始之前为每个节点预计算捕获其局部和全局结构特征。具体来说，gHAWK引入了一个预处理步骤，计算：(a)Bloom过滤器以紧凑地编码局部邻域结构，(b)TransE嵌入以表示节点在图中的全局位置。然后这些特征与任何领域特定特征(如文本嵌入)融合，产生节点特征向量，可以整合到任何GNN技术中。通过在消息传递训练中增强结构先验，gHAWK显著减少了内存使用，加速了收敛，并提高了模型准确性。在Open Graph Benchmark的大型数据集上的广泛实验表明，gHAWK在节点属性预测和链接预测任务上实现了最先进的准确性和更低的训练时间，在三个图的OGB排行榜上名列前茅。


### 论文摘要

Knowledge Graphs (KGs) are a rich source of structured, heterogeneous data, powering a wide range of applications. A common approach to leverage this data is to train a graph neural network (GNN) on the KG. However, existing message-passing GNNs struggle to scale to large KGs because they rely on the iterative message passing process to learn the graph structure, which is inefficient, especially under mini-batch training, where a node sees only a partial view of its neighborhood. In this paper, we address this problem and present gHAWK, a novel and scalable GNN training framework for large KGs. The key idea is to precompute structural features for each node that capture its local and global structure before GNN training even begins. Specifically, gHAWK introduces a preprocessing step that computes: (a)~Bloom filters to compactly encode local neighborhood structure, and (b)~TransE embeddings to represent each node's global position in the graph. These features are then fused with any domain-specific features (e.g., text embeddings), producing a node feature vector that can be incorporated into any GNN technique. By augmenting message-passing training with structural priors, gHAWK significantly reduces memory usage, accelerates convergence, and improves model accuracy. Extensive experiments on large datasets from the Open Graph Benchmark (OGB) demonstrate that gHAWK achieves state-of-the-art accuracy and lower training time on both node property prediction and link prediction tasks, topping the OGB leaderboard for three graphs.

---

## 28. Restoring Network Evolution from Static Structure

**论文链接:** [http://arxiv.org/abs/2512.08209v1](http://arxiv.org/abs/2512.08209v1)

**作者:** Jiu Zhang, Zhanwei Du, Hongwei Hu, Ke Wu, Tongchao Li, Chuan Shi, Xiaohui Huang, Yamir Moreno, Yanqing Hu

**发布时间:** 2025-12-09

### GPT解析

### 总结

研究者开发了一个可迁移的机器学习框架，能够仅从网络当前拓扑结构推断其演化轨迹，实现了高达95.3%的准确率，应用于果蝇大脑连接组研究揭示了神经连接形成时间与功能的关系。

### 背景

复杂网络的动态演化是自然和人工系统中结构与功能关系的基础，但仅从单个静态快照恢复网络的形成仍然具有挑战性。

### 目的

提出一个可迁移的机器学习框架，仅从当前拓扑推断网络演化轨迹。

### 方法

集成图神经网络与transformer，从静态拓扑中直接解锁潜在的时间维度。

### 主要发现

框架在不同领域评估中实现高达95.3%的迁移准确率；应用于果蝇大脑连接组恢复了超过260万个神经连接的形成时间；早期形成的连接支持基本行为如交配和觅食，后期形成的连接支持复杂的感觉和社会功能。

### 结论

大部分演化信息编码在静态网络架构中，为阐明复杂系统隐藏的时间动态提供了强大通用工具。

### 翻译

复杂网络的动态演化是自然和人工系统中结构与功能关系的基础。然而，仅从单个静态快照恢复网络的形成仍然具有挑战性。在此，我们提出一个可迁移的机器学习框架，仅从当前拓扑推断网络演化轨迹。通过将图神经网络与transformer相结合，我们的方法直接从静态拓扑中解锁了潜在的时间维度。在多个领域的评估中，该框架实现了高达95.3%的高迁移准确率，展示了其鲁棒性和可迁移性。应用于果蝇大脑连接组，它恢复了超过260万个神经连接的形成时间，揭示早期形成的连接支持交配和觅食等基本行为，而后期形成的连接则支持复杂的感觉和社会功能。这些结果表明，大部分演化信息编码在静态网络架构中，为阐明复杂系统隐藏的时间动态提供了强大通用工具。


### 论文摘要

The dynamical evolution of complex networks underpins the structure-function relationships in natural and artificial systems. Yet, restoring a network's formation from a single static snapshot remains challenging. Here, we present a transferable machine learning framework that infers network evolutionary trajectories solely from present topology. By integrating graph neural networks with transformers, our approach unlocks a latent temporal dimension directly from the static topology. Evaluated across diverse domains, the framework achieves high transfer accuracy of up to 95.3%, demonstrating its robustness and transferability. Applied to the Drosophila brain connectome, it restores the formation times of over 2.6 million neural connections, revealing that early-forming links support essential behaviors such as mating and foraging, whereas later-forming connections underpin complex sensory and social functions. These results demonstrate that a substantial fraction of evolutionary information is encoded within static network architecture, offering a powerful, general tool for elucidating the hidden temporal dynamics of complex systems.

---

## 29. ExPUFFIN: Thermodynamic Consistent Viscosity Prediction in an Extended Path-Unifying Feed-Forward Interfaced Network

**论文链接:** [http://arxiv.org/abs/2512.06927v2](http://arxiv.org/abs/2512.06927v2)

**作者:** Carine Menezes Rebello, Ulderico Di Caprio, Jenny Steen-Hansen, Bruno Rodrigues, Erbet Almeida Costa, Anderson Rapello dos Santos, Flora Esposito, Mumin Enis Leblebici, Idelfonso B. R. Nogueira

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文介绍了一种名为ExPUFFIN的混合图神经网络框架，用于预测纯烃的温度相关粘度，通过在输出层强制执行机制归纳偏差确保热力学一致性，相比纯数据驱动模型提高了准确性、稳健性和可转移性。

### 背景

液体粘度的准确预测对工艺设计和模拟至关重要，但对新型分子具有挑战性。传统基团贡献模型难以区分异构体、处理大分子且参数可用性有限，而纯数据驱动的图神经网络需要大量数据集且可解释性有限，同时缺乏热力学一致性。

### 目的

开发一种能够准确预测纯烃温度相关粘度的方法，同时确保热力学一致性和可靠性。

### 方法

ExPUFFIN是一个混合GNN框架，将分子信息作为图结构输入，通过图卷积网络编码，并基于两个热物理相关（三参数Andrade型方程和四参数经验粘度-温度关系）映射到归纳偏差神经元，在输出层强制执行机制归纳偏差以确保热力学一致性。

### 主要发现

基于Andrade的ExPUFFIN变体与纯数据驱动基线相比RMSE降低了37%，提供了平滑、物理一致的粘度-温度曲线的内插和外推；经验ExPUFFIN模型提供了相当的准确性同时保持了稳健趋势；在GNN输出中嵌入基于物理的结构提高了准确性、稳健性和可转移性。

### 结论

ExPUFFIN能够对复杂烃分子进行可靠的粘度预测，该方法易于扩展到其他特性和更广泛的化学领域。

### 翻译

准确预测液体粘度对于工艺设计和模拟至关重要，但对新型分子仍然具有挑战性。传统的基团贡献模型难以区分异构体、处理大分子，且参数可用性有限，而纯数据驱动的图神经网络需要大量数据集且可解释性有限。即使可行，纯数据驱动的模型在预测中缺乏热力学一致性，不是可靠的解决方案。本文介绍了ExPUFFIN，这是Path-unifying Feed-Forward Interfaced Network的扩展版本，是一个混合GNN框架，可以直接从分子图预测纯烃的温度相关粘度，同时在输出层强制执行机制归纳偏差以确保热力学一致性。分子信息以图结构给出，编码为图卷积网络，并基于两个热物理相关映射到归纳偏差神经元：三参数Andrade型方程和四参数经验粘度-温度关系。将这些模型的准确性与纯数据驱动的预测进行了比较。基于Andrade的ExPUFFIN变体与纯数据驱动基线相比RMSE降低了37%，并提供了平滑、物理一致的粘度-温度曲线的内插和外推，这些性质在纯数据驱动模型中未观察到。经验ExPUFFIN模型提供了相当的准确性同时保持了稳健趋势。总体而言，在GNN输出中嵌入基于物理的结构提高了准确性、稳健性和可转移性，能够对复杂烃分子进行可靠的粘度预测。该方法易于扩展到其他特性和更广泛的化学领域。


### 论文摘要

Accurate prediction of liquid viscosity is essential for process design and simulation, yet remains challenging for novel molecules. Conventional group-contribution models struggle with isomer discrimination, large molecules, and parameter availability, while purely data-driven graph neural networks (GNNs) demand large datasets and offer limited interpretability. Even when feasible to be applied, purely data-driven models lack thermodynamic consistency in their predictions and are not a reliable solution. This work introduces ExPUFFIN, an extended version of the Path-unifying Feed-Forward Interfaced Network, consisting of a hybrid GNN-based framework that directly predicts temperature-dependent viscosities of pure hydrocarbons from molecular graphs, while enforcing mechanistic inductive biases in the output layer to ensure thermodynamic consistency. Molecular information is given as graph structures, encoded as a graph convolutional network, and mapped to an inductive bias neuron based on two thermophysical correlations: a three-parameter Andrade-type equation and a four-parameter empirical viscosity-temperature relation. The accuracy of these models is compared with a solely data-driven prediction. The Andrade-based ExPUFFIN variant reduces RMSE compared to the purely data-driven baseline of 37 percent and yields smooth, physically consistent interpolation and extrapolation of viscosity-temperature curves, properties that are not observed in purely data-driven models. The empirical ExPUFFIN model provides comparable accuracy while retaining robust trends. Overall, embedding physics-based structure in GNN outputs improves accuracy, robustness, and transferability, enabling reliable viscosity predictions for complex hydrocarbon molecules. The approach is readily extendable to other properties and significantly broader chemical domains.

---

## 30. Open Polymer Challenge: Post-Competition Report

**论文链接:** [http://arxiv.org/abs/2512.08896v1](http://arxiv.org/abs/2512.08896v1)

**作者:** Gang Liu, Sobin Alosious, Subhamoy Mahajan, Eric Inae, Yihan Zhu, Yuhan Liu, Renzheng Zhang, Jiaxin Xu, Addison Howard, Ying Li, Tengfei Luo, Meng Jiang

**发布时间:** 2025-12-09

**备注:** The report for the competition: "NeurIPS - Open Polymer Prediction 2025". Kaggle Page: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025. Website: https://open-polymer-challenge.github.io

### GPT解析

### 总结

这篇论文介绍了开放聚合物挑战(OPC)，这是一个为聚合物信息学开发的社区基准测试，包含10K聚合物和5种属性数据集。该挑战专注于多任务聚合物属性预测，参与者使用多种机器学习技术开发模型，研究结果为聚合物科学中的分子AI奠定了新基础，有望加速可持续和节能材料的发展。

### 背景

机器学习为发现可持续聚合物材料提供了强大途径，但进展受到缺乏大规模、高质量、公开可用的聚合物数据集的限制。

### 目的

通过发布首个社区开发的聚合物信息学基准测试，解决高质量聚合物数据集缺乏的问题，促进多任务聚合物属性预测模型的发展，加速可持续和节能材料的开发。

### 方法

发布包含10K聚合物和5种属性(热导率、回转半径、密度、自由体积分数和玻璃化转变温度)的数据集；组织多任务聚合物属性预测挑战；参与者在小数据、标签不平衡和异构模拟源等现实约束下开发模型；使用基于特征的增强、迁移学习、自监督预训练和有针对性的集成策略等技术。

### 主要发现

竞赛揭示了数据准备、分布转换和跨组模拟一致性的重要经验；开发的模型、分析和发布的数据为聚合物科学中的分子AI创造了新基础；释放了测试数据集(https://www.kaggle.com/datasets/alexliu99/neurips-open-polymer-prediction-2025-test-data)；发布了数据生成管道(https://github.com/sobinalosious/ADEPT)，可模拟25种以上的属性。

### 结论

开放聚合物挑战及其相关资源为聚合物信息学提供了重要基础设施，预计将加速可持续和节能材料的发展，并为未来大规模聚合物数据集的最佳实践提供指导。

### 翻译

机器学习(ML)为发现可持续聚合物材料提供了一条强大的途径，但由于缺乏大型、高质量且公开可用的聚合物数据集，进展受到限制。开放聚合物挑战(OPC)通过发布首个社区开发的聚合物信息学基准测试解决了这一差距，该数据集包含10K聚合物和5种属性：热导率、回转半径、密度、自由体积分数和玻璃化转变温度。该挑战专注于多任务聚合物属性预测，这是材料发现虚拟筛选流程中的核心步骤。参与者在小数据、标签不平衡和异构模拟源等现实约束下开发了模型，使用了基于特征的增强、迁移学习、自监督预训练和有针对性的集成策略等技术。比赛还揭示了关于数据准备、分布转换和跨组模拟一致性的重要经验，为未来大规模聚合物数据集的最佳实践提供了指导。由此产生的模型、分析和发布的数据为聚合物科学中的分子AI创造了新基础，预计将加速可持续和节能材料的发展。除了比赛，我们在https://www.kaggle.com/datasets/alexliu99/neurips-open-polymer-prediction-2025-test-data上发布了测试数据集。我们还发布了数据生成管道https://github.com/sobinalosious/ADEPT，它可以模拟25种以上的属性，包括热导率、回转半径和密度。


### 论文摘要

Machine learning (ML) offers a powerful path toward discovering sustainable polymer materials, but progress has been limited by the lack of large, high-quality, and openly accessible polymer datasets. The Open Polymer Challenge (OPC) addresses this gap by releasing the first community-developed benchmark for polymer informatics, featuring a dataset with 10K polymers and 5 properties: thermal conductivity, radius of gyration, density, fractional free volume, and glass transition temperature. The challenge centers on multi-task polymer property prediction, a core step in virtual screening pipelines for materials discovery. Participants developed models under realistic constraints that include small data, label imbalance, and heterogeneous simulation sources, using techniques such as feature-based augmentation, transfer learning, self-supervised pretraining, and targeted ensemble strategies. The competition also revealed important lessons about data preparation, distribution shifts, and cross-group simulation consistency, informing best practices for future large-scale polymer datasets. The resulting models, analysis, and released data create a new foundation for molecular AI in polymer science and are expected to accelerate the development of sustainable and energy-efficient materials. Along with the competition, we release the test dataset at https://www.kaggle.com/datasets/alexliu99/neurips-open-polymer-prediction-2025-test-data. We also release the data generation pipeline at https://github.com/sobinalosious/ADEPT, which simulates more than 25 properties, including thermal conductivity, radius of gyration, and density.

---

## 31. Unsupervised Learning of Density Estimates with Topological Optimization

**论文链接:** [http://arxiv.org/abs/2512.08895v1](http://arxiv.org/abs/2512.08895v1)

**作者:** Suina Tanweer, Firas A. Khasawneh

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种基于拓扑的无监督学习方法，用于自动选择核密度估计中的最优带宽参数。

### 背景

核密度估计是机器学习、贝叶斯推断、随机动态和信号处理中各种算法的关键组成部分，但需要调整关键超参数核带宽，该带宽通过控制偏差-方差权衡来影响密度估计的拓扑特征。

### 目的

开发一种使用基于拓扑的损失函数的无监督学习方法，实现带宽参数的自动和无监督选择。

### 方法

采用拓扑数据分析方法量化拓扑特征，设计基于拓扑的损失函数，并与经典技术进行基准测试。

### 主要发现

所提出的方法在不同维度中表现出良好的性能和潜力。

### 结论

基于拓扑的无监督带宽选择方法是一种有效的技术，可应用于不同维度的密度估计问题。

### 翻译

核密度估计是机器学习、贝叶斯推断、随机动态和信号处理中各种算法的关键组成部分。然而，这种无监督密度估计技术需要调整一个关键的超参数：核带宽。带宽的选择至关重要，因为它通过过度或不足平滑拓扑特征来控制偏差-方差权衡。拓扑数据分析提供了数学量化拓扑特征的方法，如连通分量、环、空隙等，即使在无法可视化密度估计的高维空间中也是如此。在本文中，我们提出了一种使用基于拓扑的损失函数的无监督学习方法，用于自动和无监督地选择最优带宽，并与经典技术进行基准测试，展示了其在不同维度中的潜力。


### 论文摘要

Kernel density estimation is a key component of a wide variety of algorithms in machine learning, Bayesian inference, stochastic dynamics and signal processing. However, the unsupervised density estimation technique requires tuning a crucial hyperparameter: the kernel bandwidth. The choice of bandwidth is critical as it controls the bias-variance trade-off by over- or under-smoothing the topological features. Topological data analysis provides methods to mathematically quantify topological characteristics, such as connected components, loops, voids et cetera, even in high dimensions where visualization of density estimates is impossible. In this paper, we propose an unsupervised learning approach using a topology-based loss function for the automated and unsupervised selection of the optimal bandwidth and benchmark it against classical techniques -- demonstrating its potential across different dimensions.

---

## 32. An Additive Manufacturing Part Qualification Framework: Transferring Knowledge of Stress-strain Behaviors from Additively Manufactured Polymers to Metals

**论文链接:** [http://arxiv.org/abs/2512.08699v1](http://arxiv.org/abs/2512.08699v1)

**作者:** Chenglong Duan, Dazhong Wu

**发布时间:** 2025-12-09

### GPT解析

### 总结

该研究开发了一种结合动态时间规整和迁移学习的新框架，用于增材制造零件认证。该方法能够通过将聚合物的应力-应变行为知识迁移到金属上，有效预测金属的应力-应变行为。实验证明该方法具有高准确性和优越性。

### 背景

增材制造中零件认证非常重要，它确保增材制造的零件能够被一致地生产并在关键应用中可靠使用。零件认证旨在验证增材制造的零件是否满足性能要求，因此预测增材制造零件复杂的应力-应变行为至关重要。

### 目的

开发一种动态时间规整（DTW）-迁移学习（TL）框架用于增材制造零件认证，通过将低成本聚合物的应力-应变行为知识迁移到金属上来实现这一目标。

### 方法

使用DTW选择与目标金属数据集最相关的聚合物数据集作为源域，使用长短期记忆（LSTM）模型进行研究。实验使用了四种源聚合物（尼龙、PLA、CF-ABS和树脂）和三种目标金属（AlSi10Mg、Ti6Al4V和碳钢），这些材料由不同的增材制造技术制造。

### 主要发现

DTW-TL框架能够识别聚合物和金属之间的最接近匹配，选择单个聚合物数据集作为源域。当使用三种金属作为目标域时，DTW-TL模型实现了最低的平均绝对百分比误差12.41%和最高的决定系数0.96。该框架优于没有迁移学习的普通LSTM模型，也优于在四个聚合物数据集上预训练的迁移学习模型。

### 结论

DTW-TL框架是有效的增材制造零件认证方法。通过迁移学习，可以成功地将聚合物的应力-应变行为知识应用到金属上。

### 翻译

零件认证在增材制造（AM）中至关重要，因为它确保增材制造的零件能够被一致地生产并在关键应用中可靠使用。零件认证旨在验证增材制造的零件是否满足性能要求；因此，预测增材制造零件复杂的应力-应变行为至关重要。我们通过将增材制造的低成本聚合物的应力-应变行为知识迁移到金属上，开发了一种用于增材制造零件认证的动态时间规整（DTW）-迁移学习（TL）框架。具体而言，该框架采用DTW选择与目标金属数据集最相关的聚合物数据集作为源域。使用长短期记忆（LSTM）模型，利用四种源聚合物（即尼龙、PLA、CF-ABS和树脂）和三种目标金属（即AlSi10Mg、Ti6Al4V和碳钢）通过不同的增材制造技术制造来证明DTW-TL框架的有效性。实验结果表明，DTW-TL框架能够识别聚合物和金属之间的最接近匹配，选择单个聚合物数据集作为源域。当使用三种金属作为目标域时，DTW-TL模型分别实现了最低的平均绝对百分比误差12.41%和最高的决定系数0.96，优于没有迁移学习的普通LSTM模型，也优于在四个聚合物数据集上预训练的迁移学习模型。


### 论文摘要

Part qualification is crucial in additive manufacturing (AM) because it ensures that additively manufactured parts can be consistently produced and reliably used in critical applications. Part qualification aims at verifying that an additively manufactured part meets performance requirements; therefore, predicting the complex stress-strain behaviors of additively manufactured parts is critical. We develop a dynamic time warping (DTW)-transfer learning (TL) framework for additive manufacturing part qualification by transferring knowledge of the stress-strain behaviors of additively manufactured low-cost polymers to metals. Specifically, the framework employs DTW to select a polymer dataset as the source domain that is the most relevant to the target metal dataset. Using a long short-term memory (LSTM) model, four source polymers (i.e., Nylon, PLA, CF-ABS, and Resin) and three target metals (i.e., AlSi10Mg, Ti6Al4V, and carbon steel) that are fabricated by different AM techniques are utilized to demonstrate the effectiveness of the DTW-TL framework. Experimental results show that the DTW-TL framework identifies the closest match between polymers and metals to select one single polymer dataset as the source domain. The DTW-TL model achieves the lowest mean absolute percentage error of 12.41% and highest coefficient of determination of 0.96 when three metals are used as the target domain, respectively, outperforming the vanilla LSTM model without TL as well as the TL model pre-trained on four polymer datasets as the source domain.

---

## 33. A Lightweight Transfer Learning-Based State-of-Health Monitoring with Application to Lithium-ion Batteries in Unmanned Air Vehicles

**论文链接:** [http://arxiv.org/abs/2512.08512v1](http://arxiv.org/abs/2512.08512v1)

**作者:** Jiang Liu, Yan Qin, Wei Dai, Chau Yuen

**发布时间:** 2025-12-09

**DOI:** 10.1109/TII.2025.3631012

**备注:** Accepted in IEEE Transactions on Industrial Informatics

### GPT解析

### 总结

本研究提出了一种基于轻量级迁移学习的锂离子电池健康状态监测方法(CITL)，解决了传统迁移学习方法在便携式移动设备中计算资源消耗大、工作续航时间短的问题。

### 背景

准确的锂离子电池健康状态监测对便携式移动设备的能量信息指示至关重要。迁移学习技术可利用源工作条件知识减少目标工作条件下SOH监测所需的训练数据，但传统方法在移动设备上会消耗大量计算资源，降低工作续航时间。

### 目的

开发一种轻量级的基于迁移学习的SOH监测方法，使其适用于便携式移动设备，同时保持高精度和低计算资源消耗。

### 方法

提出建设性增量迁移学习(CITL)方法：1)利用目标域未标记数据，通过迭代添加网络节点实现半监督迁移学习机制，最小化监测残差；2)通过结构风险最小化、迁移失配最小化和流形一致性最大化保证跨域能力；3)提供收敛性分析，从理论上保证性能和网络紧凑性。

### 主要发现

通过无人机电池数据集验证，CITL在SOH估计方面分别优于SS-TCA、MMD-LSTM-DA、DDAN、BO-CNN-TL和AS$^3$LSTM，性能提升83.73%、61.15%、28.24%、87.70%和57.34%。

### 结论

CITL方法成功解决了传统迁移学习方法在便携式移动设备中的应用限制，实现了高效且精确的锂离子电池健康状态监测，显著提高了SOH估计的准确性。

### 翻译

准确且快速的锂离子电池健康状态监测对于指示便携式移动设备的能量信息具有重要作用。为了应对其多变的工作条件，迁移学习(TL)成为一种有前景的技术，可以利用数据丰富的源工作条件中的知识，显著减少目标工作条件下SOH监测所需的训练数据。然而，当应用于便携式移动设备时，传统的基于TL的SOH监测是不可行的，因为在TL阶段会消耗大量计算资源，并意外地降低工作续航时间。为了解决这些挑战，本文提出了一种基于轻量级TL的SOH监测方法，采用建设性增量迁移学习(CITL)。


### 论文摘要

Accurate and rapid state-of-health (SOH) monitoring plays an important role in indicating energy information for lithium-ion battery-powered portable mobile devices. To confront their variable working conditions, transfer learning (TL) emerges as a promising technique for leveraging knowledge from data-rich source working conditions, significantly reducing the training data required for SOH monitoring from target working conditions. However, traditional TL-based SOH monitoring is infeasible when applied in portable mobile devices since substantial computational resources are consumed during the TL stage and unexpectedly reduce the working endurance. To address these challenges, this paper proposes a lightweight TL-based SOH monitoring approach with constructive incremental transfer learning (CITL). First, taking advantage of the unlabeled data in the target domain, a semi-supervised TL mechanism is proposed to minimize the monitoring residual in a constructive way, through iteratively adding network nodes in the CITL. Second, the cross-domain learning ability of node parameters for CITL is comprehensively guaranteed through structural risk minimization, transfer mismatching minimization, and manifold consistency maximization. Moreover, the convergence analysis of the CITL is given, theoretically guaranteeing the efficacy of TL performance and network compactness. Finally, the proposed approach is verified through extensive experiments with a realistic unmanned air vehicles (UAV) battery dataset collected from dozens of flight missions. Specifically, the CITL outperforms SS-TCA, MMD-LSTM-DA, DDAN, BO-CNN-TL, and AS$^3$LSTM, in SOH estimation by 83.73%, 61.15%, 28.24%, 87.70%, and 57.34%, respectively, as evaluated using the index root mean square error.

---

## 34. High-Throughput Unsupervised Profiling of the Morphology of 316L Powder Particles for Use in Additive Manufacturing

**论文链接:** [http://arxiv.org/abs/2512.06012v2](http://arxiv.org/abs/2512.06012v2)

**作者:** Emmanuel Akeweje, Conall Kirk, Chi-Wai Chan, Denis Dowling, Mimi Zhang

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种基于机器学习的自动化框架，用于高通量表征金属粉末形貌，解决了传统粉末表征方法产量低、无法捕捉工业批次不均匀性的问题。

### 背景

选择性激光熔化是一种粉末床增材制造技术，其零件质量高度依赖于原料粉末的形貌特征。然而，传统的粉末表征方法产量低且定性，无法有效捕捉工业规模批次粉末的不均匀性。

### 目的

开发一种自动化的机器学习框架，将高通量成像与形状提取和聚类相结合，以大规模表征金属粉末形貌，并建立粉末形貌与最终零件质量之间的联系。

### 方法

研究开发并评估了三种聚类流程：自编码器流程、形状描述符流程和函数数据流程。通过内部有效性指标评估这些流程的性能，包括戴维斯-鲍尔丁指数和卡林斯基-哈拉巴兹分数。

### 主要发现

在包含约126,000张粉末图像的数据集中，傅里叶描述符+k均值流程被识别为最有效的聚类方法，在标准台式工作站上实现了每粒子亚毫秒级的运行时间，同时获得了最佳的聚类效果指标。

### 结论

这种无监督学习框架能够快速、自动地评估粉末形貌，并支持在重用周期中跟踪形状演变，为选择性激光熔化工作流程中的原料实时监测提供了可行途径。

### 翻译

选择性激光熔化是一种粉末床增材制造技术，其零件质量高度依赖于原料粉末的形貌特征。然而，传统的粉末表征方法产量低且定性，无法捕捉工业规模批次的不均匀性。我们提出了一种自动化的机器学习框架，将高通量成像与形状提取和聚类相结合，以大规模表征金属粉末形貌。我们开发并评估了三种聚类流程：自编码器流程、形状描述符流程和函数数据流程。在包含约126,000张粉末图像（直径0.5-102微米）的数据集中，内部有效性指标识别出傅里叶描述符+k均值流程是最有效的，在标准台式工作站上实现了每粒子亚毫秒级的运行时间，同时获得了最低的戴维斯-鲍尔丁指数和最高的卡林斯基-哈拉巴兹分数。虽然这项工作重点是建立形貌聚类框架，但由此产生的形状组为未来研究它们与流动性、堆积密度和选择性激光熔化零件质量的关系奠定了基础。总的来说，这种无监督学习框架能够快速、自动地评估粉末形貌，并支持在重用周期中跟踪形状演变，为选择性激光熔化工作流程中的原料实时监测提供了可行途径。


### 论文摘要

Selective Laser Melting (SLM) is a powder-bed additive manufacturing technique whose part quality depends critically on feedstock morphology. However, conventional powder characterization methods are low-throughput and qualitative, failing to capture the heterogeneity of industrial-scale batches. We present an automated, machine learning framework that couples high-throughput imaging with shape extraction and clustering to profile metallic powder morphology at scale. We develop and evaluate three clustering pipelines: an autoencoder pipeline, a shape-descriptor pipeline, and a functional-data pipeline. Across a dataset of approximately 126,000 powder images (0.5-102 micrometer diameter), internal validity metrics identify the Fourier-descriptor + k-means pipeline as the most effective, achieving the lowest Davies-Bouldin index and highest Calinski-Harabasz score while maintaining sub-millisecond runtime per particle on a standard desktop workstation. Although the present work focuses on establishing the morphological-clustering framework, the resulting shape groups form a basis for future studies examining their relationship to flowability, packing density, and SLM part quality. Overall, this unsupervised learning framework enables rapid, automated assessment of powder morphology and supports tracking of shape evolution across reuse cycles, offering a path toward real-time feedstock monitoring in SLM workflows.

---

## 35. Decoupling Template Bias in CLIP: Harnessing Empty Prompts for Enhanced Few-Shot Learning

**论文链接:** [http://arxiv.org/abs/2512.08606v1](http://arxiv.org/abs/2512.08606v1)

**作者:** Zhenyu Zhang, Guangyao Chen, Yixiong Zou, Zhimeng Huang, Yuhua Li

**发布时间:** 2025-12-09

**备注:** 14 pages, 8 figures, Association for the Advancement of Artificial Intelligence (AAAI2026, poster)

### GPT解析

### 总结

本研究针对CLIP模型中的模板-样本相似性偏差问题，提出使用空提示框架来减少偏差，提高分类准确性和鲁棒性。

### 背景

对比语言-图像预训练(CLIP)模型在少样本学习中表现出色，但模板-样本相似性(TSS)会引入偏差，导致模型依赖模板接近度而非真实的样本到类别对齐。

### 目的

开发一种框架来减少CLIP模型中的模板偏差，提高分类准确性和鲁棒性。

### 方法

提出使用空提示(empty prompts)的框架，这些提示传达'空'的概念而不包含类别信息。框架包含两个阶段：预训练阶段使用空提示揭示并减少CLIP编码器中的模板诱导偏差；少样本微调阶段使用偏差校准损失强制图像与其类别之间的正确对齐。

### 主要发现

模板-样本相似性(TSS)引入了偏差，导致模型依赖模板接近度而非真实的样本到类别对齐，降低了分类的准确性和鲁棒性。使用空提示可以捕获无偏的模板特征并抵消TSS偏差。

### 结论

通过在多个基准测试中的实验证明，模板校正方法显著减少了由TSS引起的性能波动，产生了更高的分类准确性和更强的鲁棒性。

### 翻译

对比语言-图像预训练(CLIP)模型通过将视觉和文本表示进行对齐，在少样本学习中表现出色。我们的研究表明，模板-样本相似性(TSS)（定义为文本模板和图像样本之间的相似性）引入了偏差。这种偏差导致模型依赖模板接近度而非真实的样本到类别对齐，降低了分类的准确性和鲁棒性。我们提出了一种使用空提示（传达'空'的概念但不包含类别信息的文本输入）的框架。这些提示捕获无偏的模板特征并抵消TSS偏差。该框架采用两个阶段。在预训练期间，空提示揭示并减少CLIP编码器内的模板诱导偏差。在少样本微调期间，偏差校准损失强制图像与其类别之间的正确对齐，确保模型专注于相关的视觉线索。在多个基准测试中的实验表明，我们的模板校正方法显著减少了由TSS引起的性能波动，产生了更高的分类准确性和更强的鲁棒性。本项目的代码库可在https://github.com/zhenyuZ-HUST/Decoupling-Template-Bias-in-CLIP获取。


### 论文摘要

The Contrastive Language-Image Pre-Training (CLIP) model excels in few-shot learning by aligning visual and textual representations. Our study shows that template-sample similarity (TSS), defined as the resemblance between a text template and an image sample, introduces bias. This bias leads the model to rely on template proximity rather than true sample-to-category alignment, reducing both accuracy and robustness in classification. We present a framework that uses empty prompts, textual inputs that convey the idea of "emptiness" without category information. These prompts capture unbiased template features and offset TSS bias. The framework employs two stages. During pre-training, empty prompts reveal and reduce template-induced bias within the CLIP encoder. During few-shot fine-tuning, a bias calibration loss enforces correct alignment between images and their categories, ensuring the model focuses on relevant visual cues. Experiments across multiple benchmarks demonstrate that our template correction method significantly reduces performance fluctuations caused by TSS, yielding higher classification accuracy and stronger robustness. The repository of this project is available at https://github.com/zhenyuZ-HUST/Decoupling-Template-Bias-in-CLIP.

---

## 36. Self-Evolving 3D Scene Generation from a Single Image

**论文链接:** [http://arxiv.org/abs/2512.08905v1](http://arxiv.org/abs/2512.08905v1)

**作者:** Kaizhi Zheng, Yue Fan, Jing Gu, Zishuo Xu, Xuehai He, Xin Eric Wang

**发布时间:** 2025-12-09

### GPT解析

### 总结

EvoScene是一个自演化、无需训练的框架，可以从单张图像逐步重建完整的3D场景，解决了现有方法在处理复杂、大规模场景时的局限性。

### 背景

从单张图像生成高质量的纹理3D场景在视觉和图形学领域仍然是一个基本挑战。最近的图像到3D生成器能够从单视图恢复合理的几何结构，但它们以物体为中心的训练方式限制了泛化能力。

### 目的

提出EvoScene框架，能够从单张图像逐步重建完整的3D场景，特别是处理复杂、大规模场景，保持几何稳定性和纹理一致性。

### 方法

结合3D生成模型的几何推理能力和视频生成模型的视觉知识，通过三个迭代阶段实现：空间先验初始化、视觉引导的3D场景网格生成和空间引导的新视角生成，在2D和3D域之间交替进行。

### 主要发现

实验表明，与强大的基线方法相比，EvoScene实现了更优的几何稳定性、视角一致的纹理，以及对未见区域的补全能力。

### 结论

EvoScene成功地解决了从单张图像生成高质量3D场景的挑战，特别是在处理复杂、大规模场景方面表现出色，生成的3D模型具有几何稳定性和纹理一致性，适用于实际应用。

### 翻译

从单张图像生成高质量的纹理3D场景在视觉和图形学领域仍然是一个基本挑战。最近的图像到3D生成器能够从单视图恢复合理的几何结构，但它们以物体为中心的训练方式限制了它们对复杂、大规模场景的泛化能力，这些场景需要保持真实的结构和纹理。我们提出了EvoScene，这是一个自演化、无需训练的框架，可以从单张图像逐步重建完整的3D场景。核心思想是结合现有模型的互补优势：来自3D生成模型的几何推理能力和来自视频生成模型的视觉知识。通过三个迭代阶段——空间先验初始化、视觉引导的3D场景网格生成和空间引导的新视角生成——EvoScene在2D和3D域之间交替进行，逐步改进场景的结构和外观。在不同场景上的实验表明，与强大的基线方法相比，EvoScene实现了更优的几何稳定性、视角一致的纹理，以及对未见区域的补全能力，能够为实际应用生成可直接使用的3D网格。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从单张图像生成高质量、完整的3D场景的问题。这个问题在现实中非常重要，因为游戏、电影、动画、模拟等数字应用严重依赖高质量的3D视觉效果，而目前手动制作3D资产需要大量人工建模和纹理制作，难以规模化。在研究中，这是一个基本挑战，因为现有方法在处理复杂场景时表现不佳，产生粗糙的几何形状和错位或模糊的纹理，且受限于物体级数据，难以扩展到大规模场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：单图像3D场景生成面临有限观察覆盖范围和难以确保高分辨率、全局一致纹理的挑战。基于此，作者提出'自我进化'的迭代框架，结合现有模型的互补优势：3D生成模型的几何推理能力和视频生成模型的视觉知识。作者借鉴了深度估计、点云融合、3D扩散模型和视频生成等现有技术，但创新性地设计了三阶段迭代流程（空间先验初始化、视觉引导的3D场景网格生成、空间引导的新视角生成），通过循环交替处理2D和3D领域，使几何和外观相互改进，逐步完善3D场景。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'自我进化'的迭代框架，结合3D生成模型的几何推理能力和视频生成模型的视觉知识，逐步完善从单张图像生成的3D场景。整体流程包括三个循环阶段：1)空间先验初始化：从2D图像估计深度并构建点云作为几何约束；2)视觉引导的3D场景网格生成：使用3D扩散模型将点云提升为完整网格；3)空间引导的新视角生成：利用网格渲染深度图引导视频生成模型合成新视角图像。这些新视角被反馈回下一迭代，形成几何和外观相互改进的循环，逐步扩大视角覆盖范围，最终生成完整、高质量的3D场景。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)自我进化的迭代框架，通过循环逐步完善3D场景；2)三阶段模块化设计，有效整合3D模型的几何知识和视频模型的视觉先验；3)几何-外观共同进化机制，使两者相互改进。相比之前工作，不同之处在于：不局限于单次生成或物体级数据，能够处理复杂场景；不依赖预定义相机轨迹，实现更全面场景覆盖；明确整合3D和视频生成模型的互补优势；使用深度条件视频生成确保多视图一致性；无需大量手动标注或特定领域数据集。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'EvoScene通过一个自我进化的三阶段迭代框架，成功结合了3D生成模型的几何推理能力和视频生成模型的视觉知识，实现了从单张图像生成高质量、完整且照片级真实的3D场景，显著优于现有的图像到3D生成方法。'}


### 论文摘要

Generating high-quality, textured 3D scenes from a single image remains a fundamental challenge in vision and graphics. Recent image-to-3D generators recover reasonable geometry from single views, but their object-centric training limits generalization to complex, large-scale scenes with faithful structure and texture. We present EvoScene, a self-evolving, training-free framework that progressively reconstructs complete 3D scenes from single images. The key idea is combining the complementary strengths of existing models: geometric reasoning from 3D generation models and visual knowledge from video generation models. Through three iterative stages--Spatial Prior Initialization, Visual-guided 3D Scene Mesh Generation, and Spatial-guided Novel View Generation--EvoScene alternates between 2D and 3D domains, gradually improving both structure and appearance. Experiments on diverse scenes demonstrate that EvoScene achieves superior geometric stability, view-consistent textures, and unseen-region completion compared to strong baselines, producing ready-to-use 3D meshes for practical applications.

---

## 37. Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference

**论文链接:** [http://arxiv.org/abs/2512.08860v1](http://arxiv.org/abs/2512.08860v1)

**作者:** Amit Bendkhale

**发布时间:** 2025-12-09

**备注:** 6 pages, 3 figures. Code and data: https://github.com/Amiton7/Tri-Bench. Accepted to the AAAI 2026 Workshop on Trust and Control in Agentic AI (TrustAgent)

### GPT解析

### 总结

研究提出了Tri-Bench基准测试，评估视觉语言模型(VLMs)在几何推理方面的能力，发现VLMs在真实场景变化下表现不佳，特别是在处理3D空间和识别特定三角形类型时。

### 背景

可验证的几何推理是可信且可控的智能AI的关键组成部分。尽管视觉语言模型(VLMs)具有令人印象深刻的能力，但在真实场景变化下常常失败。

### 目的

提出Tri-Bench基准测试，隔离相对几何推理因素，同时评估相机姿态和场景上下文对VLMs几何推理能力的影响。

### 方法

创建包含平面三角形问题的紧凑基准测试，测试相机姿态(平面vs倾斜)和物体干扰(10个日常物体)两个因素。使用固定提示评估四个VLMs，提示中包含描述周围正方形边界的防护栏。在二值和连续目标上评估六个简单任务。

### 主要发现

VLMs关于3D真实情况的平均准确率约为69%(最好75%，最差64%)；与图像平面中的2D投影对齐度更高，平均准确率约72%；在识别等边、等腰和直角三角形时准确率降至约0%；相机倾斜导致准确率下降约4.1%；模型未能有效利用提示中的参考框架；物体干扰对准确率无显著影响。

### 结论

VLMs在几何推理方面存在局限性，特别是在处理3D空间和识别特定三角形类型时，且未能有效利用提供的参考框架信息。

### 翻译

可验证的几何推理是可信且可控的智能AI的关键组成部分。尽管视觉语言模型(VLMs)具有令人印象深刻的能力，但在真实场景变化下常常失败。我们提出了Tri-Bench，一个平面三角形问题的紧凑基准测试，它隔离了相对几何推理，同时强调了两个部署关键因素：相机姿态(平面vs倾斜)和通过物体干扰(10个日常物体)的场景上下文。为了测试可验证性和控制性，我们使用单一固定提示评估了四个最近的VLMs，该提示的防护栏明确描述了一个周围的正方形边界，通过单应性可实现正确答案。我们在二值和连续目标上评估了六个简单任务，观察到关于3D真实情况的整体准确率适中，平均约69%(最好约75%，最差约64%)。相同的响应与图像平面中的2D投影更为一致，平均准确率约72%。所有四个VLMs在识别少数形状类别(等边、等腰、直角三角形)时 consistently fail，准确率降至约0%。此外，相机倾斜下VLMs整体准确率下降约4.1%。这表明模型未能正确利用提示中提供的明确参考框架提示，默认使用2D图像平面提示。最后，我们发现物体干扰对VLMs准确率没有显著影响。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉语言模型（VLMs）在空间推理任务中的可靠性问题，特别是在相机倾斜和物体干扰条件下的表现不足。这个问题非常重要，因为可信和可控的代理AI（如机器人导航、AR/VR测量工具、3D重建等）严重依赖可验证的空间推理能力。如果VLMs在基本几何推理上不可靠，就无法安全部署在关键应用中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者发现现有空间推理基准测试要么关注绝对距离和角度估计，要么在抽象场景中测试，缺乏对实际部署中关键因素（如相机姿态和物体干扰）的压力测试。因此，作者设计了一个专门的基准测试Tri-Bench，借鉴了现有工作但填补了这一空白。作者选择了三角形作为基本几何结构，因为它们能很好地评估相对空间推理能力，并设计了四种拍摄条件来测试相机倾斜和物体干扰的影响。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过测试VLMs在不同条件下的三角形推理能力，揭示其在真实场景中的空间推理局限性。整体流程包括：1) 构建包含100个不同三角形的400张图像数据集，每个三角形有四种拍摄条件（平面无物体、平面有物体、倾斜无物体、倾斜有物体）；2) 定义六个几何推理任务，涉及三角形形状分类和角度距离比较；3) 使用统一的提示和评估方法测试四个最新VLMs；4) 分析结果，比较3D真实世界和2D图像平面上的准确性，以及不同条件下的性能差异。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出Tri-Bench基准测试，专门测试相机倾斜和物体干扰对VLMs空间推理的影响；2) 发现VLMs即使有明确的3D参考框架提示，仍默认使用2D图像平面推理；3) 识别出精度任务中的多数类偏差，在少数形状类别上准确率几乎为零；4) 发现相机倾斜显著降低性能，而物体干扰影响很小。相比之前工作，Tri-Bench更专注于隔离特定因素对相对几何推理的影响，而非测试广泛认知技能或抽象几何问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Tri-Bench基准测试揭示了视觉语言模型在空间推理中的关键失败模式，特别是在相机倾斜条件下表现不佳且无法有效利用3D参考框架提示，为构建真正可靠、可控和可信的代理AI提供了重要诊断工具。'}


### 论文摘要

Verifiable geometric reasoning is a critical component for trustworthy and controllable agentic AI. Despite impressive capabilities, Vision-Language Models (VLMs) often fail under realistic scene changes. We present Tri-Bench, a compact benchmark of planar triangle problems that isolates relative geometric reasoning while stressing two deployment-critical factors: camera pose (planar vs. tilted) and scene context via object interference (10 everyday objects). To test verifiability and control, we evaluate four recent VLMs using a single, fixed prompt whose guardrail explicitly describes a surrounding square border, enabling correct answers via homography. We evaluate six simple tasks over binary and continuous targets, and observe that the overall accuracy with respect to 3D ground truth is modest, ~69% on average (best ~75%, worst ~64%). The same responses align even more closely with 2D projections in the image plane, where mean accuracy is ~72%. All four VLMs consistently fail, with accuracy falling to ~0%, on recognizing minority shape classes (equilateral, isosceles, right-angled triangles). Additionally, overall VLM accuracy degrades by ~4.1% under camera tilt. This demonstrates that models fail to correctly utilize the explicit frame-of-reference hint provided in the prompt and default to 2D image plane cues. Finally, we find that object interference has no significant effect on VLM accuracy.

---

## 38. Mind to Hand: Purposeful Robotic Control via Embodied Reasoning

**论文链接:** [http://arxiv.org/abs/2512.08580v1](http://arxiv.org/abs/2512.08580v1)

**作者:** Peijun Tang, Shangjin Xie, Binyan Sun, Baifu Huang, Kuncheng Luo, Haotian Yang, Weiqi Jin, Jianan Wang

**发布时间:** 2025-12-09

**备注:** 49 pages, 25 figures

### GPT解析

### 总结

Lumo-1是一个通用的视觉-语言-行动模型，成功将机器人推理与行动统一起来，通过三阶段预训练管道和强化学习优化，在复杂机器人任务中表现出色并具有强泛化能力。

### 背景

人类行为有上下文和意图，推理起核心作用。虽然互联网规模数据使AI系统具备广泛推理能力，但这些能力在物理行动中的基础仍是一大挑战。

### 目的

介绍Lumo-1模型，将机器人推理（'思维'）与机器人行动（'手'）统一起来，实现具身推理和行动预测。

### 方法

基于预训练视觉-语言模型的多模态推理能力，逐步扩展到具身推理和行动预测，最终实现结构化推理和推理-行动对齐。采用三阶段预训练管道：1)精选视觉-语言数据继续VLM预训练；2)跨具身机器人数据与视觉-语言数据联合训练；3)在Astribot S1上收集的轨迹上进行带推理过程的行动训练。最后集成强化学习改进推理-行动一致性。

### 主要发现

Lumo-1在具身视觉-语言推理方面取得显著性能提升，是通用机器人控制的关键组成部分。在真实世界评估中，Lumo-1在广泛挑战性机器人任务中超越强基线模型，对新颖物体和环境有强泛化能力，特别擅长长时程任务和响应需要推理策略、概念和空间的人类自然指令。

### 结论

Lumo-1成功统一了机器人推理与行动，通过三阶段预训练和强化学习优化，实现了在复杂机器人任务中的优异表现和强泛化能力。

### 翻译

人类行为具有上下文和意图，推理起着核心作用。虽然互联网规模的数据使AI系统具备了广泛的推理能力，但这些能力在物理行动中的基础仍然是一个重大挑战。我们介绍了Lumo-1，一个通用的视觉-语言-行动（VLA）模型，将机器人推理（'思维'）与机器人行动（'手'）统一起来。我们的方法基于预训练的视觉-语言模型（VLMs）的多模态推理能力，逐步扩展到具身推理和行动预测，最终实现结构化推理和推理-行动对齐。这形成了一个三阶段预训练管道：(1)在精选的视觉-语言数据上继续VLM预训练，以增强规划、空间理解和轨迹预测等具身推理技能；(2)在跨具身机器人数据与视觉-语言数据上进行联合训练；(3)在Astribot S1（一个具有类人灵活性和敏捷性的双臂移动操作器）上收集的轨迹上进行带有推理过程的行动训练。最后，我们集成强化学习以进一步改进推理-行动一致性，并在语义推理和运动控制之间形成闭环。大量实验证明，Lumo-1在具身视觉-语言推理方面取得了显著的性能提升，这是通用机器人控制的关键组成部分。真实世界的评估进一步表明，Lumo-1在广泛的挑战性机器人任务中超越了强大的基线模型，对新颖物体和环境有很强的泛化能力，特别是在长时程任务中表现出色，并且能够响应需要推理策略、概念和空间的人类自然指令。


### 论文摘要

Humans act with context and intention, with reasoning playing a central role. While internet-scale data has enabled broad reasoning capabilities in AI systems, grounding these abilities in physical action remains a major challenge. We introduce Lumo-1, a generalist vision-language-action (VLA) model that unifies robot reasoning ("mind") with robot action ("hand"). Our approach builds upon the general multi-modal reasoning capabilities of pre-trained vision-language models (VLMs), progressively extending them to embodied reasoning and action prediction, and ultimately towards structured reasoning and reasoning-action alignment. This results in a three-stage pre-training pipeline: (1) Continued VLM pre-training on curated vision-language data to enhance embodied reasoning skills such as planning, spatial understanding, and trajectory prediction; (2) Co-training on cross-embodiment robot data alongside vision-language data; and (3) Action training with reasoning process on trajectories collected on Astribot S1, a bimanual mobile manipulator with human-like dexterity and agility. Finally, we integrate reinforcement learning to further refine reasoning-action consistency and close the loop between semantic inference and motor control. Extensive experiments demonstrate that Lumo-1 achieves significant performance improvements in embodied vision-language reasoning, a critical component for generalist robotic control. Real-world evaluations further show that Lumo-1 surpasses strong baselines across a wide range of challenging robotic tasks, with strong generalization to novel objects and environments, excelling particularly in long-horizon tasks and responding to human-natural instructions that require reasoning over strategy, concepts and space.

---

## 39. CVP: Central-Peripheral Vision-Inspired Multimodal Model for Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2512.08135v1](http://arxiv.org/abs/2512.08135v1)

**作者:** Zeyuan Chen, Xiang Zhang, Haiyang Xu, Jianwen Xie, Zhuowen Tu

**发布时间:** 2025-12-09

**备注:** Accepted to WACV 2026

### GPT解析

### 总结

研究提出了一种受中央-周边视觉启发的CVP框架，这是一种简单有效的多模态模型，用于空间推理。该模型通过目标亲和性标记和以自我为中心的网格两个互补组件，实现了对3D环境的结构化、上下文感知理解，并在多个基准测试中取得了最先进的性能。

### 背景

现有空间推理方法主要依赖非结构化表示（如点云、体素或补丁特征），并通过坐标嵌入隐式注入场景上下文，这往往导致空间推理能力有限，因为缺乏明确的高层次结构理解。

### 目的

为了解决现有方法在空间推理方面的局限性，研究旨在开发一种能够提供明确、高层次结构理解的多模态模型，从而增强对复杂3D环境的理解能力。

### 方法

研究提出了CVP框架，在大型多模态模型架构中引入了两个互补组件：1）目标亲和性标记，类似于中央视觉，引导模型关注与查询相关的对象；2）以自我为中心的网格，类似于周边视觉，捕捉全局场景上下文和空间排列。这两个组件协同工作，实现对复杂3D环境的结构化、上下文感知理解。

### 主要发现

实验表明，CVP框架在一系列3D场景理解基准测试中取得了最先进的性能，证明了其在空间推理任务上的有效性。

### 结论

CVP框架通过模拟人类视觉系统的中央和周边视觉特性，成功解决了现有方法在空间推理方面的局限性，为3D场景理解提供了一种新的有效方法。

### 翻译

我们提出了一种受中央-周边视觉启发的框架（CVP），这是一种简单而有效的多模态模型，用于空间推理，其灵感来自人类视觉的两种类型——中央视觉和周边视觉。现有方法主要依赖非结构化表示，如点云、体素或补丁特征，并通过坐标嵌入隐式注入场景上下文。然而，由于缺乏明确的高层次结构理解，这往往导致空间推理能力有限。为解决这一限制，我们在基于大型多模态模型的架构中引入了两个互补组件：类似于中央视觉的目标亲和性标记，引导模型关注与查询相关的对象；类似于周边视觉的以自我为中心的网格，捕捉全局场景上下文和空间排列。这些组件协同工作，实现对复杂3D环境的结构化、上下文感知理解。实验表明，CVP在一系列3D场景理解基准测试中取得了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景理解中的空间推理能力不足问题。现有方法依赖非结构化表示（如点云、体素）并通过坐标嵌入隐式注入场景上下文，导致缺乏明确的高层次结构理解。这个问题很重要，因为3D场景理解是机器人、自主导航和具身AI等领域的基础能力，需要模型能够联合推理几何、语义和空间关系，实现复杂的3D问答、物体定位和场景描述等任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类视觉系统的双重特性获得灵感：人类视觉由高清晰度的中心区域（中央凹）和低清晰度的周边区域组成，允许人们在保持广泛环境感知的同时清晰聚焦感兴趣区域。作者借鉴了计算机视觉中类似人类视觉机制的工作（如场景解析、导航和显著性检测），以及大型多模态模型在图像和视频理解方面的进展。此外，还参考了3D场景理解领域的点云处理、物体中心表示和多视图渲染等方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是模拟人类视觉的双重机制：中心视觉和周边视觉。具体实现两个互补组件：1）目标亲和性token，引导模型关注与查询相关的物体；2）自我中心网格，从世界中心视角捕获全局场景上下文和空间关系。整体流程：输入多视图图像和用户问题→提取视觉特征并反向投影到3D空间→构建自我中心网格提供全局上下文→添加目标亲和性token引导关注→使用语言模型生成回答→通过对比损失优化目标亲和性token。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）受人类视觉系统启发的双重视觉机制模拟；2）目标亲和性token，明确教导模型关注与查询相关的区域；3）自我中心网格，将物体位置离散化为鸟瞰图网格并转换为文本描述。相比之前工作，CVP不再依赖非结构化3D表示和隐式坐标嵌入，同时关注细粒度物体细节和全局空间上下文，通过文本形式的自我中心网格避免模态不对齐问题，并使用显式对比监督增强对目标相关物体的注意力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CVP通过模拟人类视觉的中心-周边双重机制，引入目标亲和性token和自我中心网格两个创新组件，显著提升了大型多模态模型在3D场景理解任务中的空间推理能力，实现了在多个基准测试上的最先进性能。'}


### 论文摘要

We present a central-peripheral vision-inspired framework (CVP), a simple yet effective multimodal model for spatial reasoning that draws inspiration from the two types of human visual fields -- central vision and peripheral vision. Existing approaches primarily rely on unstructured representations, such as point clouds, voxels, or patch features, and inject scene context implicitly via coordinate embeddings. However, this often results in limited spatial reasoning capabilities due to the lack of explicit, high-level structural understanding. To address this limitation, we introduce two complementary components into a Large Multimodal Model-based architecture: target-affinity token, analogous to central vision, that guides the model's attention toward query-relevant objects; and allocentric grid, akin to peripheral vision, that captures global scene context and spatial arrangements. These components work in tandem to enable structured, context-aware understanding of complex 3D environments. Experiments show that CVP achieves state-of-the-art performance across a range of 3D scene understanding benchmarks.

---

## 40. SSCATeR: Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling for Real-Time 3D Object Detection in LiDAR Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.08557v1](http://arxiv.org/abs/2512.08557v1)

**作者:** Alexander Dow, Manduhu Manduhu, Matheus Santos, Ben Bartlett, Gerard Dooly, James Riordan

**发布时间:** 2025-12-09

**备注:** 22 Pages, 26 Figures, This work has been submitted to the IEEE Sensors Journal for possible publication

### GPT解析

### 总结

本文提出了一种基于激光雷达扫描的物体检测方法，通过聚焦点云变化区域，利用时间数据回收技术显著提高了处理效率。

### 背景

传统激光雷达物体检测方法需要对整个点云进行处理，计算效率较低。

### 目的

减少激光雷达物体检测中的计算量，提高处理效率而不牺牲检测准确性。

### 方法

使用短步长滑动时间窗口，存储扫描间的卷积结果；提出稀疏散射卷积算法(SSCATeR)，将LiDAR数据视为连续流，只处理点云变化部分。

### 主要发现

通过聚焦变化区域，实现了6.61倍的处理时间减少，同时保持了与传统方法相同的特征图质量。

### 结论

SSCATeR算法能够显著提高激光雷达物体检测的计算效率，同时保持检测准确性。

### 翻译

本研究利用激光雷达扫描的连续运动特性，将物体检测工作集中在从一帧到另一帧点数据发生变化的特定区域。我们通过使用短步长的滑动时间窗口并存储扫描间的卷积结果来实现这一点。这使我们能够忽略未变化的区域，显著减少每次前向传播的卷积操作数量而不牺牲准确性。这种数据重用方案为检测数据引入了极高的稀疏性。为了利用这种稀疏性，我们扩展了之前关于基于散射的卷积工作，提出了带有时间数据回收的稀疏散射卷积算法(SSCATeR)。该算法将传入的LiDAR数据视为连续流，只作用于点云的变化部分。通过这样做，我们实现了最高6.61倍的处理时间减少。测试结果表明，我们方法输出的特征图与传统稀疏卷积技术产生的特征图相同，同时大大提高了网络的计算效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云数据处理速度慢的问题，特别是在实时3D目标检测场景中。这个问题在现实中非常重要，因为自主无人机和自动驾驶车辆需要快速感知和检测环境以避免碰撞，尤其在无人机群执行任务时，处理延迟可能导致严重的安全问题。现有方法要么牺牲准确性来提高速度，要么提高准确性但牺牲速度，难以满足实时性要求，特别是在资源受限的嵌入式系统上。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到LiDAR扫描具有连续性特点，且连续帧之间大部分点云保持不变，但传统方法会重复计算未变化区域的特征。他们首先意识到即使使用稀疏卷积处理空间稀疏性，也忽视了LiDAR扫描的时序方面。作者借鉴了PointPillars架构进行修改，利用了稀疏卷积技术处理点云稀疏性，参考了StrObe和PolarStream等流式处理方法，但专注于单传感器处理，并扩展了之前的scatter-based卷积操作以支持数据重用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用LiDAR扫描的连续性，只处理变化区域，重用未变化区域的计算结果。具体实现流程包括：1)收集无人机搭载LiDAR的数据；2)将点云组织成pillars；3)创建change maps跟踪哪些pillars在10ms内有变化；4)对变化的pillars进行1D卷积生成特征；5)将特征散射回特征网格创建伪图像；6)使用SSCATeR卷积处理变化区域，重用未变化区域结果；7)使用修改后的SSD检测头生成3D边界框。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)时间数据重用方案，通过滑动时间窗口和短步长重用未变化区域结果；2)SSCATeR算法，扩展scatter-based卷积支持数据重用；3)变化图跟踪需要重新计算的区域；4)实现实时处理(小于10ms)。相比之前工作的不同：比PointPillars减少59.85%处理时间同时保持相同准确性；比稀疏卷积平均减少72.8%活动站点处理；避免Transformer方法的计算复杂性；不使用可能丢失安全关键目标的激进采样；不需要360°LiDAR扫描，适用于非重复扫描模式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SSCATeR算法通过利用LiDAR扫描的时序连续性和数据重用机制，实现了实时3D目标检测，在保持准确性的同时显著提高了处理速度，特别适用于资源受限的嵌入式系统。'}


### 论文摘要

This work leverages the continuous sweeping motion of LiDAR scanning to concentrate object detection efforts on specific regions that receive a change in point data from one frame to another. We achieve this by using a sliding time window with short strides and consider the temporal dimension by storing convolution results between passes. This allows us to ignore unchanged regions, significantly reducing the number of convolution operations per forward pass without sacrificing accuracy. This data reuse scheme introduces extreme sparsity to detection data. To exploit this sparsity, we extend our previous work on scatter-based convolutions to allow for data reuse, and as such propose Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling (SSCATeR). This operation treats incoming LiDAR data as a continuous stream and acts only on the changing parts of the point cloud. By doing so, we achieve the same results with as much as a 6.61-fold reduction in processing time. Our test results show that the feature maps output by our method are identical to those produced by traditional sparse convolution techniques, whilst greatly increasing the computational efficiency of the network.

---

## 41. Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.08247v1](http://arxiv.org/abs/2512.08247v1)

**作者:** Haowen Zheng, Hu Zhu, Lu Deng, Weihao Gu, Yang Yang, Yanyan Liang

**发布时间:** 2025-12-09

**备注:** AAAI-26

### GPT解析

### 总结

提出了一种名为未来时序知识蒸馏(FTKD)的稀疏查询方法，有效将离线教师模型的未来帧知识转移到在线学生模型，在3D目标检测任务中显著提升了性能。

### 背景

基于相机的时序3D目标检测在自动驾驶领域表现出色，离线模型通过使用未来帧提高了准确性。然而，现有的知识蒸馏方法忽略了未来帧，主要关注严格帧对齐下的空间特征蒸馏或时间关系蒸馏，使得在线模型难以有效学习未来知识。

### 目的

设计一种方法，能够将离线教师模型中的未来帧知识有效转移到在线学生模型，克服现有知识蒸馏方法的局限性。

### 方法

提出未来时序知识蒸馏(FTKD)方法，包括：1) 未来感知特征重建策略，鼓励学生模型在不严格帧对齐的情况下捕获未来特征；2) 未来引导的logit蒸馏，利用教师的稳定前景和背景上下文。将该方法应用于两个高性能的3D目标检测基线。

### 主要发现

在nuScenes数据集上实现了高达1.3 mAP和1.3 NDS的性能提升，同时实现了最准确的速度估计，且没有增加推理成本。

### 结论

FTKD方法成功解决了现有知识蒸馏方法忽略未来帧的问题，通过未来感知特征重建和未来引导的logit蒸馏，有效提升了在线模型的3D目标检测性能，为自动驾驶领域提供了新的解决方案。

### 翻译

基于相机的时序3D目标检测在自动驾驶领域已展现出令人印象深刻的结果，离线模型通过使用未来帧提高了准确性。知识蒸馏可以是一种将离线模型中的丰富信息转移到在线模型的有吸引力的框架。然而，现有的知识蒸馏方法忽略了未来帧，因为它们主要关注严格帧对齐下的空间特征蒸馏或时间关系蒸馏，这使得在线模型难以有效学习未来知识。为此，我们提出了一种基于稀疏查询的方法，即未来时序知识蒸馏(FTKD)，有效将离线教师模型的未来帧知识转移到在线学生模型。具体来说，我们提出了一种未来感知特征重建策略，鼓励学生模型在不严格帧对齐的情况下捕获未来特征。此外，我们进一步引入了未来引导的logit蒸馏，利用教师的稳定前景和背景上下文。FTKD应用于两个高性能的3D目标检测基线，在nuScenes数据集上实现了高达1.3 mAP和1.3 NDS的提升，以及最准确的速度估计，同时没有增加推理成本。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在线3D物体检测模型无法利用未来帧信息的问题。在自动驾驶中，离线模型可以通过使用未来帧信息提高检测精度，特别是对小物体和被遮挡物体的检测，但在线模型无法访问这些未来帧。这个问题很重要，因为它限制了在线检测模型的性能，而在线模型是实际应用中必须使用的，因为它需要实时处理数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有知识蒸馏方法的局限性，包括空间特征蒸馏需要严格帧对齐和时间关系蒸馏忽视未来帧信息。他们借鉴了知识蒸馏的基本框架，但针对3D物体检测的特殊需求进行了改进。具体借鉴了掩码特征重建的思想(MGD方法)、稀疏查询表示(如SparseBEV)、匈牙利算法进行查询匹配(DETR方法)以及时间自适应混合机制(AdaMixer)。作者设计了未来感知特征重建和未来引导的logit蒸馏两个核心组件，使在线模型能够有效学习未来知识而不受限于严格的帧对齐。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一种未来时间知识蒸馏(FTKD)方法，将离线教师模型中的未来帧知识有效传递给在线学生模型。整体流程包括：1)未来感知特征重建：对视角特征应用时间自注意力从未来帧提取语义信息，对稀疏查询特征使用时间自适应混合融合历史和未来特征，生成随机掩码后用生成器恢复被掩码的学生特征，并在教师时间聚合特征的监督下进行重建；2)未来引导的logit蒸馏：使用匈牙利算法在教师和学生预测间建立匹配，同时考虑前景和背景查询；3)结合特征重建损失和logit蒸馏损失作为整体蒸馏损失，训练学生模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)未来时间知识蒸馏框架，有效传递未来帧知识；2)未来感知特征重建策略，克服了严格帧对齐的限制；3)未来引导的logit蒸馏，利用教师模型的稳定前景和背景上下文。相比之前的工作，FTKD不要求严格的帧对齐，能够利用未来帧信息；不仅关注帧间关系，还特别关注未来帧；同时考虑前景和背景信息，而不仅仅是前景物体；在保持高效推理速度的同时显著提高了检测性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种未来时间知识蒸馏方法，通过未来感知的特征重建和未来引导的logit蒸馏，有效将离线教师模型中的未来帧知识传递给在线学生模型，显著提高了3D物体检测性能，同时保持了高效的推理速度。'}


### 论文摘要

Camera-based temporal 3D object detection has shown impressive results in autonomous driving, with offline models improving accuracy by using future frames. Knowledge distillation (KD) can be an appealing framework for transferring rich information from offline models to online models. However, existing KD methods overlook future frames, as they mainly focus on spatial feature distillation under strict frame alignment or on temporal relational distillation, thereby making it challenging for online models to effectively learn future knowledge. To this end, we propose a sparse query-based approach, Future Temporal Knowledge Distillation (FTKD), which effectively transfers future frame knowledge from an offline teacher model to an online student model. Specifically, we present a future-aware feature reconstruction strategy to encourage the student model to capture future features without strict frame alignment. In addition, we further introduce future-guided logit distillation to leverage the teacher's stable foreground and background context. FTKD is applied to two high-performing 3D object detection baselines, achieving up to 1.3 mAP and 1.3 NDS gains on the nuScenes dataset, as well as the most accurate velocity estimation, without increasing inference cost.

---

## 42. Dual-Branch Center-Surrounding Contrast: Rethinking Contrastive Learning for 3D Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.08673v1](http://arxiv.org/abs/2512.08673v1)

**作者:** Shaofeng Zhang, Xuanqi Chen, Xiangdong Zhang, Sitong Wu, Junchi Yan

**发布时间:** 2025-12-09

**DOI:** 10.13140/RG.2.2.16132.18563

**备注:** 16 pages, 6 figures

### GPT解析

### 总结

本文提出了一种名为CSCon的新型双分支中心-周围对比框架，用于3D点云自监督学习，结合了生成式和对比式方法的优点，能够有效捕获高层次判别特征和局部几何细节。

### 背景

现有3D点云自监督学习方法大多基于掩码自编码器的生成式方法，但难以有效捕获高层次判别特征；对比式方法在图像数据中表现优异，但在3D数据中应用有限，且简单将2D对比学习方法应用于3D无法有效学习局部细节。

### 目的

解决3D点云自监督学习中生成式方法难以捕获高层次判别特征，以及对比式方法无法有效学习3D局部细节的挑战。

### 方法

提出CSCon双分支框架，分别对中心和周围部分应用掩码，构建具有中心偏向和周围偏向表示的双分支输入；同时引入补丁级别对比损失，增强高层次信息和局部敏感性。

### 主要发现

在FULL和ALL协议下，CSCon性能与生成式方法相当；在MLP-LINEAR、MLP-3和ONLY-NEW协议下达到最先进水平，甚至超越跨模态方法；在MLP-LINEAR协议下，在ScanObjectNN三个变体上分别比基线高出7.9%、6.7%和10.3%。

### 结论

CSCon框架成功解决了3D点云自监督学习中生成式与对比式方法的局限性，通过双分支设计和补丁级别对比损失，有效结合了高层次特征和局部细节学习。

### 翻译

大多数现有的3D点云自监督学习(SSL)方法都是由基于掩码自编码器(MAE)的生成式方法主导。然而，这些生成式方法已被证明难以有效捕获高层次的判别特征，导致在线性探测和其他下游任务中表现不佳。相比之下，对比式方法在图像数据中擅长判别特征表示和泛化能力。尽管如此，3D数据中的对比学习(CL)仍然很少。此外，简单地将为2D数据设计的CL方法应用于3D数据无法有效学习3D局部细节。为了解决这些挑战，我们提出了一种新颖的双分支中心-周围对比(CSCon)框架。具体来说，我们分别对中心和周围部分应用掩码，构建具有中心偏向和周围偏向表示的双分支输入，以更好地捕获丰富的几何信息。同时，我们引入了补丁级别的对比损失，进一步增强高层次信息和局部敏感性。在FULL和ALL协议下，CSCon实现了与生成式方法相当的性能；在MLP-LINEAR、MLP-3和ONLY-NEW协议下，我们的方法达到了最先进的成果，甚至超越了跨模态方法。特别是在MLP-LINEAR协议下，我们的方法在ScanObjectNN的三个变体上分别比基线(Point-MAE)高出7.9%、6.7%和10.3%。代码将公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云自监督学习中生成方法难以有效捕获高级判别性特征的问题，以及3D点云领域对比学习研究不足的问题。这个问题很重要，因为3D点云在自动驾驶、机器人、AR/VR等现实应用中至关重要，而自监督学习可以减少对大量标注数据的依赖，同时高级判别性特征和局部细节对于下游任务(如分类、分割)的表现至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到生成方法在高级特征提取上的局限性，认识到对比学习在图像领域的优势，发现简单将2D对比学习应用于3D的不足。他们借鉴了图像对比学习的基本原理、点云处理中的patch-based范式(如Point-MAE)、使用FPS和KNN算法生成patch、PointNet编码局部patch以及Transformer作为骨干网络等现有工作，但针对3D点云的特性进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云分割为中心和周围两部分，分别进行掩码处理，构建中心偏向和周围偏向的双分支输入，并设计patch级别的对比损失，增强高级信息和局部敏感性。整体流程包括：1)使用FPS选择中心点，KNN生成patch并进行坐标归一化；2)使用位置投影器和PointNet分别编码中心点和周围点；3)构建双分支输入序列，分别掩码中心部分和周围部分；4)通过共享参数的编码器处理，并使用patch级别对比损失优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双分支中心-周围对比范式，分别掩码处理中心和周围部分；2)实例内patch级别对比损失，专注于patch级别的对比而非全局；3)单视角范式，不需要多视角预训练和解码器，减少计算开销。相比之前工作，CSCon使用对比学习而非重建任务，专注于patch级别而非全局级别，不需要复杂增强策略生成不同视图，且使用更少参数实现了更好性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CSCon通过双分支中心-周围对比学习框架，有效解决了3D点云自监督学习中高级特征提取和局部细节捕获的挑战，实现了在多个下游任务上的最先进性能。'}


### 论文摘要

Most existing self-supervised learning (SSL) approaches for 3D point clouds are dominated by generative methods based on Masked Autoencoders (MAE). However, these generative methods have been proven to struggle to capture high-level discriminative features effectively, leading to poor performance on linear probing and other downstream tasks. In contrast, contrastive methods excel in discriminative feature representation and generalization ability on image data. Despite this, contrastive learning (CL) in 3D data remains scarce. Besides, simply applying CL methods designed for 2D data to 3D fails to effectively learn 3D local details. To address these challenges, we propose a novel Dual-Branch \textbf{C}enter-\textbf{S}urrounding \textbf{Con}trast (CSCon) framework. Specifically, we apply masking to the center and surrounding parts separately, constructing dual-branch inputs with center-biased and surrounding-biased representations to better capture rich geometric information. Meanwhile, we introduce a patch-level contrastive loss to further enhance both high-level information and local sensitivity. Under the FULL and ALL protocols, CSCon achieves performance comparable to generative methods; under the MLP-LINEAR, MLP-3, and ONLY-NEW protocols, our method attains state-of-the-art results, even surpassing cross-modal approaches. In particular, under the MLP-LINEAR protocol, our method outperforms the baseline (Point-MAE) by \textbf{7.9\%}, \textbf{6.7\%}, and \textbf{10.3\%} on the three variants of ScanObjectNN, respectively. The code will be made publicly available.

---

## 43. A Sensor-Aware Phenomenological Framework for Lidar Degradation Simulation and SLAM Robustness Evaluation

**论文链接:** [http://arxiv.org/abs/2512.08653v1](http://arxiv.org/abs/2512.08653v1)

**作者:** Doumegna Mawuto Koudjo Felix, Xianjia Yu, Zhuo Zou, Tomi Westerlund

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种传感器感知的、现象学框架，用于直接在真实点云上模拟可解释的激光雷达退化，实现可控和可重复的SLAM压力测试。该框架保留每点的几何形状、强度和时间结构，同时应用多种退化技术，具有自主主题和传感器检测功能，模块化配置包含四个严重性级别，实时性能良好。实验验证揭示了不同激光雷达架构和SLAM系统的独特鲁棒性模式。

### 背景

基于激光雷达的SLAM系统对不良条件（如遮挡、噪声和视场退化）高度敏感，而现有的鲁棒性评估方法要么缺乏物理基础，要么无法捕捉传感器特定行为。

### 目的

开发一个传感器感知的、现象学框架，用于直接在真实点云上模拟可解释的激光雷达退化，实现可控和可重复的SLAM压力测试。

### 方法

与图像派生的损坏基准或仅模拟方法不同，该框架保留每点的几何形状、强度和时间结构，同时应用结构化丢弃、视场减少、高斯噪声、遮挡掩码、稀疏化和运动失真。框架具有自主主题和传感器检测功能，模块化配置包含四个严重性级别（轻度-极端），实时性能（每帧小于20毫秒），兼容ROS工作流。

### 主要发现

在三种激光雷达架构和五个最先进的SLAM系统上的实验验证揭示了由传感器设计和环境背景形成的独特鲁棒性模式。

### 结论

开源实现为在具有物理意义的退化场景下基准测试基于激光雷达的SLAM提供了实用基础。

### 翻译

基于激光雷达的SLAM系统对不良条件（如遮挡、噪声和视场(FoV)退化）高度敏感，然而现有的鲁棒性评估方法要么缺乏物理基础，要么无法捕捉传感器特定行为。本文提出了一种传感器感知的、现象学框架，用于直接在真实点云上模拟可解释的激光雷达退化，实现可控和可重复的SLAM压力测试。与图像派生的损坏基准（如SemanticKITTI-C）或仅模拟方法（如lidarsim）不同，所提出的系统在应用结构化丢弃、视场减少、高斯噪声、遮挡掩码、稀疏化和运动失真的同时，保留了每点的几何形状、强度和时间结构。该框架具有自主主题和传感器检测功能，模块化配置包含四个严重性级别（轻度-极端），实时性能（每帧小于20毫秒），兼容ROS工作流。在三种激光雷达架构和五个最先进的SLAM系统上的实验验证揭示了由传感器设计和环境背景形成的独特鲁棒性模式。开源实现为在具有物理意义的退化场景下基准测试基于激光雷达的SLAM提供了实用基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决激光雷达SLAM系统在不利条件下（如遮挡、噪声和视场退化）的鲁棒性评估方法不足的问题。现有方法要么缺乏物理基础，要么无法捕捉传感器特定行为。这个问题很重要，因为随着自动驾驶和机器人技术的发展，SLAM系统需要在各种实际环境中可靠工作，准确评估系统在恶劣条件下的鲁棒性对于确保安全性和可靠性至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了激光雷达在不利条件下的性能下降问题及现有评估方法的局限性，包括缺乏物理基础和传感器特定行为。他们借鉴了先前研究对激光雷达退化的理解（如雾引起的衰减、噪声建模等），参考了现有基准使用离散强度级别的设计，并扩展了作者之前在正常条件下表征不同激光雷达的数据集。设计时遵循物理相关性和可比较性原则，采用现象学建模策略，保留了几何一致性、时间顺序和传感器特征扫描模式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个传感器感知的现象学框架，直接在真实点云上模拟可解释的激光雷达退化，同时保留每个点的几何、强度和时间结构。整体流程包括：1)处理多传感器激光雷达流并进行点云解析与对齐；2)通过自主配置机制自动识别不同传感器；3)根据配置文件中的严重性级别应用五种退化模块（点dropout、视场减少、噪声、遮挡、运动失真）；4)保持时间一致性并实时发布增强后的点云；5)提供可视化界面和性能监控。整个系统以混合C++/Python实现，处理延迟小于20毫秒。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)传感器感知的退化建模，针对不同激光雷达架构定制；2)模块化配置系统，具有四个严重性级别；3)实时ROS实现，支持自主传感器检测和可视化；4)系统化的SLAM鲁棒性评估框架。相比之前工作，不同于基于图像的损坏基准（直接在原始激光雷达测量上应用退化）、纯仿真方法（不需要完整场景几何）和数据增强方法（针对SLAM评估而非神经网络训练），此方法保留了真实传感器特性、噪声特性和时间一致性，同时支持实时处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种传感器感知的现象学框架，通过直接在原始激光雷达点云上应用物理可解释的退化，实现了对SLAM系统在恶劣条件下鲁棒性的可控、可复制的评估，并揭示了不同传感器架构和环境背景下的独特鲁棒性模式。'}


### 论文摘要

Lidar-based SLAM systems are highly sensitive to adverse conditions such as occlusion, noise, and field-of-view (FoV) degradation, yet existing robustness evaluation methods either lack physical grounding or do not capture sensor-specific behavior. This paper presents a sensor-aware, phenomenological framework for simulating interpretable lidar degradations directly on real point clouds, enabling controlled and reproducible SLAM stress testing. Unlike image-derived corruption benchmarks (e.g., SemanticKITTI-C) or simulation-only approaches (e.g., lidarsim), the proposed system preserves per-point geometry, intensity, and temporal structure while applying structured dropout, FoV reduction, Gaussian noise, occlusion masking, sparsification, and motion distortion. The framework features autonomous topic and sensor detection, modular configuration with four severity tiers (light--extreme), and real-time performance (less than 20 ms per frame) compatible with ROS workflows. Experimental validation across three lidar architectures and five state-of-the-art SLAM systems reveals distinct robustness patterns shaped by sensor design and environmental context. The open-source implementation provides a practical foundation for benchmarking lidar-based SLAM under physically meaningful degradation scenarios.

---

## 44. OCCDiff: Occupancy Diffusion Model for High-Fidelity 3D Building Reconstruction from Noisy Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.08506v1](http://arxiv.org/abs/2512.08506v1)

**作者:** Jialu Sui, Rui Liu, Hongsheng Zhang

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种名为OCCDiff的新方法，通过在占用函数空间中应用潜在扩散技术，解决了从LiDAR点云重建建筑物时面临的不同点密度和噪声干扰下的表面捕捉问题。该方法结合了潜在扩散过程和函数自动编码器架构，能够生成高质量的3D建筑物轮廓，并对噪声数据表现出鲁棒性。

### 背景

从LiDAR点云重建建筑物面临的主要挑战在于准确捕捉不同点密度和噪声干扰下的建筑物表面。传统方法难以灵活适应各种条件下的点云数据质量。

### 目的

开发一种能够灵活收集不同分辨率下建筑物高质量3D轮廓的方法，解决LiDAR点云重建中的表面捕捉难题。

### 方法

提出OCCDiff方法，在占用函数空间中应用潜在扩散技术。该方法结合潜在扩散过程和函数自动编码器架构生成连续占用函数，并设计了点编码器为扩散学习提供条件特征。同时采用多任务训练策略增强模型性能。

### 主要发现

OCCDiff能够生成与目标分布高度一致的物理一致性样本，且对噪声数据表现出显著的鲁棒性，有效解决了不同点密度条件下的建筑物表面重建问题。

### 结论

OCCDiff通过在占用函数空间中应用潜在扩散技术，结合点编码器和多任务训练策略，有效解决了从LiDAR点云重建建筑物时面临的关键挑战，特别是在处理不同点密度和噪声干扰方面表现出色。

### 翻译

从LiDAR点云重建建筑物的一个主要挑战在于准确捕捉不同点密度和噪声干扰下的建筑物表面。为了灵活收集不同分辨率下建筑物的高质量3D轮廓，我们提出了在占用函数空间中应用潜在扩散的OCCDiff方法。我们的OCCDiff结合了潜在扩散过程和函数自动编码器架构，生成可在任意位置评估的连续占用函数。此外，我们提出了点编码器，为扩散学习提供条件特征，约束最终占用预测给占用解码器，并将多模态特征插入到潜在编码器中以进行潜在生成。为进一步增强模型性能，我们采用了多任务训练策略，确保点编码器学习多样且鲁棒的特征表示。实验结果表明，我们的方法生成与目标分布高度一致的物理一致性样本，并对噪声数据表现出鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从嘈杂的点云数据中进行高保真3D建筑重建的问题。这个问题在现实中非常重要，因为3D建筑重建是城市分析、模拟和数字孪生开发等应用的基础，而现有的点云数据往往存在稀疏、密度不均、遮挡和噪声等问题，导致重建困难。同时，建筑重建需要在不同场景下平衡精度与效率，这对城市规划、基础设施管理和智慧城市建设等实际应用具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：端到端训练管道限制了模型泛化能力，而扩散模型在3D形状生成中虽有潜力但分辨率受限于输入点云密度。作者借鉴了扩散模型在3D生成中的应用，结合occupancy networks的隐式表示方法，设计了在函数空间中的潜在扩散模型。同时采用了Transformer架构和DGCNN作为特征提取器，并引入Flow Matching技术学习数据生成过程。这种方法融合了多种现有技术的优势，同时针对建筑重建的特殊需求进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在函数空间表示中应用潜在扩散模型，结合函数自编码器和点编码器，实现从部分点云到完整3D建筑的高保真重建。整体流程分为两个阶段：首先训练函数自编码器和点编码器，使用占用损失和CD损失进行多任务学习；然后训练扩散模型，学习完整建筑形状的特征分布。推理时，从噪声开始通过扩散模型生成潜在特征，结合点编码器的条件特征，通过占用解码器重建完整3D建筑，最后转换为网格表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 函数空间上的潜在扩散框架，生成连续函数表示；2) 具有全局注意力的点编码器，提供几何先验；3) 多任务学习策略保留特征几何保真度。相比之前的工作，OCCDiff不依赖端到端训练，避免了边界模糊问题；在函数空间而非点云或体素上操作，提供更高分辨率；结合扩散模型与自编码器，增强了对噪声数据的鲁棒性；相比SDF方法计算成本更低，任务更简单。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OCCDiff通过在函数空间中应用潜在扩散模型，结合多任务学习和点编码器条件特征，实现了从嘈杂点云数据中高保真、鲁棒的3D建筑重建，能够灵活生成不同分辨率的精细建筑形状。'}


### 论文摘要

A major challenge in reconstructing buildings from LiDAR point clouds lies in accurately capturing building surfaces under varying point densities and noise interference. To flexibly gather high-quality 3D profiles of the building in diverse resolution, we propose OCCDiff applying latent diffusion in the occupancy function space. Our OCCDiff combines a latent diffusion process with a function autoencoder architecture to generate continuous occupancy functions evaluable at arbitrary locations. Moreover, a point encoder is proposed to provide condition features to diffusion learning, constraint the final occupancy prediction for occupancy decoder, and insert multi-modal features for latent generation to latent encoder. To further enhance the model performance, a multi-task training strategy is employed, ensuring that the point encoder learns diverse and robust feature representations. Empirical results show that our method generates physically consistent samples with high fidelity to the target distribution and exhibits robustness to noisy data.

---

## 45. SDT-6D: Fully Sparse Depth-Transformer for Staged End-to-End 6D Pose Estimation in Industrial Multi-View Bin Picking

**论文链接:** [http://arxiv.org/abs/2512.08430v1](http://arxiv.org/abs/2512.08430v1)

**作者:** Nico Leuze, Maximilian Hoh, Samed Doğan, Nicolas R. -Peña, Alfred Schoettl

**发布时间:** 2025-12-09

**备注:** Accepted to WACV 2026. Preprint version

### GPT解析

### 总结

本文提出了一种在密集堆积工业取件环境中准确恢复物体6D姿态的整体深度估计方法，通过多视图深度图融合和稀疏处理技术解决遮挡、反射和无纹理部件带来的挑战。

### 背景

在密集堆积的工业取件环境中准确恢复6D姿态面临严重挑战，主要由于遮挡、反射和无纹理部件的存在，现有方法难以有效处理。

### 目的

开发一种能够处理密集堆积环境中物体6D姿态估计的整体深度估计方法，克服现有技术的局限性。

### 方法

提出一种仅使用深度信息的6D姿态估计方法，融合多视图深度图为细粒度3D点云或稀疏TSDF；采用分阶段热图机制产生场景自适应注意力先验；引入密度感知稀疏变换器块动态处理遮挡和非均匀数据分布；使用完全稀疏框架实现高分辨率体积表示；通过体素级投票策略预测多个物体的6D姿态。

### 主要发现

稀疏3D方法在近距离机器人应用中具有未被充分探索的潜力；完全稀疏框架能够捕获对准确姿态估计至关重要的精细几何细节；所提方法在高度杂乱的工业和家庭取件场景中展现出具有竞争力的性能。

### 结论

所提出的整体深度估计方法有效地解决了密集堆积工业取件环境中的6D姿态估计挑战，通过多视图深度图融合、稀疏处理技术和自适应注意力机制实现了准确高效的姿态预测。

### 翻译

在密集堆积的工业取件环境中准确恢复6D姿态仍然是一个严峻挑战，这是由于遮挡、反射和无纹理部件的存在。我们引入了一种整体的仅使用深度信息的6D姿态估计方法，将多视图深度图融合为细粒度3D点云或稀疏的截断符号距离场。我们框架的核心是一种分阶段热图机制，能够在不同分辨率上产生场景自适应的注意力先验，引导计算朝向前景区域，从而在高分辨率下保持内存需求的可行性。同时，我们提出了一种密度感知的稀疏变换器块，能够动态关注（自）遮挡和3D数据的非均匀分布。虽然稀疏3D方法在长距离感知方面已被证明有效，但在近距离机器人应用中的潜力尚未得到充分探索。我们的框架完全稀疏运行，能够实现高分辨率体积表示，以捕获对杂乱环境中准确姿态估计至关重要的精细几何细节。我们的方法整体处理整个场景，通过一种新颖的体素级投票策略预测6D姿态，可以同时预测任意数量的目标物体的姿态。我们在最近发布的IPD和MV-YCB多视图数据集上验证了我们的方法，在高度杂乱的工业和家庭取件场景中展示了具有竞争力的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决工业多视角抓取场景中6D姿态估计的挑战，特别是处理密集堆积环境中的遮挡、反射表面和无纹理物体等问题。这个问题在现实中非常重要，因为准确的6D姿态估计是机器人抓取的基础，而工业环境中的这些挑战使得现有方法难以可靠地工作，限制了机器人在复杂工业场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了6D姿态估计在工业场景中的核心挑战是分辨率与内存可行性之间的矛盾。他们观察到抓取场景在精细分辨率下表现出极端的空间稀疏性（仅3%的体素被占用），因此提出了基于稀疏3D编码的方法。作者借鉴了现有稀疏3D检测器在长距离感知中的成功经验，如子流形稀疏卷积和VoxelNext等全稀疏管道的设计理念。同时，他们扩展了高斯平滑热图目标策略，并针对近距离机器人应用的特殊需求进行了创新设计，如分阶段热图机制和密度感知稀疏变换器块。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个全稀疏的6D姿态估计框架，通过分阶段热图机制逐步聚焦于前景区域，同时利用多视角深度信息融合和体素级投票策略实现整体场景处理。整体流程包括：1)多视角深度图融合成稀疏3D体素网格；2)RoI热图阶段粗略定位感兴趣区域并丢弃背景；3)物体性热图阶段精细识别物体部分；4)稀疏变换器块处理非均匀分布和遮挡问题；5)6D姿态回归阶段预测平移偏移和旋转，通过聚类和ICP精炼获得最终姿态估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分阶段热图机制，自适应关注场景重要区域；2)全稀疏架构，只在有数据的体素上计算，实现高分辨率表示；3)密度感知稀疏变换器块，使用双分支设计同时捕获几何细节和邻域上下文；4)整体场景处理和体素级投票策略，支持同时预测任意数量物体的姿态。相比之前的工作，SDT-6D不依赖于上游检测模块，避免了错误传播；不使用随机下采样，而是自适应关注重要区域；不仅用于背景抑制，还保留高分辨率几何细节；针对近距离机器人应用优化，而非长距离感知。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SDT-6D通过全稀疏架构和分阶段热图机制，解决了工业抓取场景中6D姿态估计面临的遮挡和无纹理物体等挑战，实现了高效且准确的多物体姿态预测。'}


### 论文摘要

Accurately recovering 6D poses in densely packed industrial bin-picking environments remain a serious challenge, owing to occlusions, reflections, and textureless parts. We introduce a holistic depth-only 6D pose estimation approach that fuses multi-view depth maps into either a fine-grained 3D point cloud in its vanilla version, or a sparse Truncated Signed Distance Field (TSDF). At the core of our framework lies a staged heatmap mechanism that yields scene-adaptive attention priors across different resolutions, steering computation toward foreground regions, thus keeping memory requirements at high resolutions feasible. Along, we propose a density-aware sparse transformer block that dynamically attends to (self-) occlusions and the non-uniform distribution of 3D data. While sparse 3D approaches has proven effective for long-range perception, its potential in close-range robotic applications remains underexplored. Our framework operates fully sparse, enabling high-resolution volumetric representations to capture fine geometric details crucial for accurate pose estimation in clutter. Our method processes the entire scene integrally, predicting the 6D pose via a novel per-voxel voting strategy, allowing simultaneous pose predictions for an arbitrary number of target objects. We validate our method on the recently published IPD and MV-YCB multi-view datasets, demonstrating competitive performance in heavily cluttered industrial and household bin picking scenarios.

---

## 46. PointDico: Contrastive 3D Representation Learning Guided by Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.08330v1](http://arxiv.org/abs/2512.08330v1)

**作者:** Pengbo Li, Yiding Sun, Haozhe Cheng

**发布时间:** 2025-12-09

**备注:** Accepted by IJCNN 2025

### GPT解析

### 总结

PointDico是一种新型3D表征学习模型，通过整合扩散模型和对比模型的优势，解决了3D数据表示中的挑战，在多个基准测试中取得了最先进性能。

### 背景

自监督表征学习在自然语言处理和2D计算机视觉中已取得显著进步，但在处理3D数据时面临困难，因为3D数据具有无序性和不均匀密度的特点。

### 目的

开发一种能够有效学习3D数据表征的方法，结合对比模型和扩散模型的优势，解决现有方法中的过拟合和点云处理问题。

### 方法

提出PointDico模型，通过知识蒸馏同时从去噪生成建模和跨模态对比学习中学习，其中扩散模型指导对比模型。引入分层金字塔条件生成器进行多尺度几何特征提取，采用双通道设计融合局部和全局上下文信息。

### 主要发现

对比模型在3D数据表示中容易过拟合，3D掩码自编码器难以处理无序点云；通过整合扩散模型和对比模型的优势可以有效解决这些问题。

### 结论

PointDico成功实现了扩散模型和对比模型的有机结合，在3D表征学习任务中达到了新的最先进水平，如ScanObjectNN上94.32%的准确率和ShapeNetPart上86.5%的实例mIoU。

### 翻译

自监督表征学习在自然语言处理和2D计算机视觉中显示出显著改进。然而，由于3D数据的无序和不均匀密度，现有方法在表示3D数据时面临困难。通过对主流对比生成方法的深入分析，我们发现对比模型容易过拟合，而3D掩码自编码器难以处理无序点云。这促使我们通过共享扩散模型和对比模型的优势来学习3D表征，但由于两种范式之间存在模式差异，这并非易事。在本文中，我们提出了PointDico，一种无缝整合这些方法的新颖模型。PointDico通过知识蒸馏从去噪生成建模和跨模态对比学习中学习，其中扩散模型作为对比模型的指导。我们引入了分层金字塔条件生成器用于多尺度几何特征提取，并采用双通道设计有效融合局部和全局上下文信息。PointDico在3D表征学习中取得了新的最先进水平，例如在ScanObjectNN上达到94.32%的准确率，在ShapeNetPart上达到86.5%的实例mIoU。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云数据表示学习的挑战，特别是处理点云的无序性和密度不均匀性问题。这个问题在现实中非常重要，因为3D数据在自动驾驶、机器人、医疗成像等领域有广泛应用，而3D数据的标注和收集比2D图像和文本更加耗时昂贵，导致数据稀缺。现有方法如对比学习容易过拟合，而3D掩码自编码器难以处理无序点云，限制了3D表示学习的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的优缺点进行思考：对比学习容易过拟合，3D MAE难以处理无序点云。受扩散模型去噪过程启发，作者认为去噪自编码器比MAE更适合点云重建，因为通过添加随机噪声可以学习点云的全局结构和分布特征。作者借鉴了扩散模型、对比学习、知识蒸馏和多模态学习等现有工作，创新性地将这些方法结合，设计了PointDico模型，通过知识蒸馏将扩散模型作为教师指导对比学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将扩散模型和对比学习相结合，通过知识蒸馏将扩散模型作为教师模型，指导对比模型学习更好的3D表示，同时利用多模态数据增强训练多样性。整体流程包括：1)条件点扩散：前向过程添加噪声，反向过程在条件指导下恢复点云；2)分层金字塔条件生成器：H2 Net捕获多尺度特征，DIP Net处理高低频特征；3)跨模态对比学习：结合点云、图像和文本特征计算对比损失；4)整体训练：总损失为扩散损失和对比损失之和，通过端到端方式训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)第一个扩散引导的跨模态对比学习框架；2)分层金字塔条件生成器(H2 Net和DIP Net)；3)多模态数据融合策略。相比之前工作，PointDico不同于纯对比学习(避免过拟合)、纯生成方法(更好处理无序点云)、其他扩散模型(专注于表示学习而非生成)和其他跨模态方法(结合文本模态和知识蒸馏)，通过结合扩散模型的全局结构学习能力和对比学习的特征表示能力，实现了更有效的3D表示学习。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointDico创新性地将扩散模型与对比学习相结合，通过知识蒸馏和多模态数据融合，实现了3D点云表示学习的显著提升，为处理无序且密度不均匀的3D数据提供了新思路。'}


### 论文摘要

Self-supervised representation learning has shown significant improvement in Natural Language Processing and 2D Computer Vision. However, existing methods face difficulties in representing 3D data because of its unordered and uneven density. Through an in-depth analysis of mainstream contrastive and generative approaches, we find that contrastive models tend to suffer from overfitting, while 3D Mask Autoencoders struggle to handle unordered point clouds. This motivates us to learn 3D representations by sharing the merits of diffusion and contrast models, which is non-trivial due to the pattern difference between the two paradigms. In this paper, we propose \textit{PointDico}, a novel model that seamlessly integrates these methods. \textit{PointDico} learns from both denoising generative modeling and cross-modal contrastive learning through knowledge distillation, where the diffusion model serves as a guide for the contrastive model. We introduce a hierarchical pyramid conditional generator for multi-scale geometric feature extraction and employ a dual-channel design to effectively integrate local and global contextual information. \textit{PointDico} achieves a new state-of-the-art in 3D representation learning, \textit{e.g.}, \textbf{94.32\%} accuracy on ScanObjectNN, \textbf{86.5\%} Inst. mIoU on ShapeNetPart.

---

## 47. Query-aware Hub Prototype Learning for Few-Shot 3D Point Cloud Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.08253v1](http://arxiv.org/abs/2512.08253v1)

**作者:** YiLin Zhou, Lili Wei, Zheming Xu, Ziyi Chen, Congyan Lang

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种新颖的查询感知中心原型(QHP)学习方法，用于解决Few-shot 3D点云语义分割中的原型偏差问题，通过中心原型生成和原型分布优化两个模块，有效缩小了原型与查询集之间的语义差距。

### 背景

Few-shot 3D点云语义分割(FS-3DSeg)旨在仅用少量标记样本分割新类别。现有的基于度量的原型学习方法仅从支持集生成原型，而不考虑它们与查询数据的相关性。

### 目的

解决原型偏差问题，即原型过度适应支持集特定特征而无法推广到查询分布的问题，特别是在存在分布偏移的情况下，以提高分割性能。

### 方法

提出查询感知中心原型(QHP)学习方法，包含两个模块：1)中心原型生成(HPG)模块，构建连接查询点和支持点的二分图，识别频繁链接的支持中心，生成查询相关原型；2)原型分布优化(PDO)模块，采用纯度加权的对比损失，通过优化原型表示来减少不良中心和模糊原型的影响。

### 主要发现

在S3DIS和ScanNet上的大量实验表明，QHP比最先进方法实现了显著的性能提升，有效缩小了FS-3DSeg中原型与查询集之间的语义差距。

### 结论

QHP方法通过明确建模支持集和查询集之间的语义相关性，能够有效解决Few-shot 3D点云语义分割中的原型偏差问题，提高分割性能。

### 翻译

少样本3D点云语义分割(FS-3DSeg)旨在仅用少量标记样本分割新类别。然而，现有的基于度量的原型学习方法仅从支持集生成原型，而不考虑它们与查询数据的相关性。这常常导致原型偏差，即原型过度适应支持集特定特征而无法推广到查询分布，特别是在存在分布偏移的情况下，导致分割性能下降。为解决这一问题，我们提出了一种新颖的查询感知中心原型(QHP)学习方法，明确建模支持集和查询集之间的语义相关性。具体而言，我们提出了一个中心原型生成(HPG)模块，构建连接查询点和支持点的二分图，识别频繁连接的支持中心，并生成与查询相关的原型，更好地捕捉跨集语义。为进一步减少不良中心和类边界附近模糊原型的影响，我们引入了一个原型分布优化(PDO)模块，该模块采用纯度加权的对比损失，通过将不良中心和离群原型拉近到相应的类中心来优化原型表示。在S3DIS和ScanNet上的大量实验表明，QHP比最先进方法实现了显著的性能提升，有效缩小了FS-3DSeg中原型与查询集之间的语义差距。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决小样本3D点云语义分割中的原型偏差问题。现有方法仅从支持集生成原型，不考虑与查询数据的相关性，导致原型过度拟合支持集特征，无法泛化到查询分布。这个问题很重要，因为3D点云分割对自动驾驶和机器人至关重要，而完全监督方法依赖大量标注数据，难以泛化到新类别。解决原型偏差可以显著提升模型在新类别上的分割性能，减少对大量标注数据的依赖。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别现有方法的局限性：仅从支持集生成原型而忽略与查询数据的语义相关性。他们注意到'hubness'现象（某些点经常出现在其他点的最近邻列表中），认为这些hub点能自然反映支持-查询语义相关性。与现有工作不同，作者不将hub视为有害并避免，而是认为同类内的hub（好hub）可准确捕获支持-查询关系。方法借鉴了原型学习框架和对比学习，设计了两个关键模块：Hub Prototype Generation (HPG)和Prototype Distribution Optimization (PDO)。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用hub点作为原型，生成与查询相关的hub原型，并优化坏hub和离群原型的分布。整体流程：1)使用共享骨干网络提取支持集和查询集特征；2)HPG模块通过Hub Point Mining识别高频hub点，再通过Hub Prototype Clustering生成查询相关原型；3)通过相似度测量将查询点与hub原型匹配生成分割结果；4)PDO模块在训练中识别坏hub，使用纯度重新加权对比损失将坏hub和离群原型拉向类中心；5)结合交叉熵损失和PC损失进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)Query-aware Hub Prototype学习框架，首次显式建模支持-查询语义相关性；2)Hub Prototype Generation模块，利用hub点生成查询相关原型；3)Prototype Distribution Optimization模块，优化坏hub和离群原型的分布。不同之处：之前方法仅从支持集生成原型，而QHP考虑查询数据；之前工作视hub为有害并避免，QHP利用hub作为原型并专门处理坏hub；QHP关注原型与查询集的语义相关性，而非仅关注内部表示能力或多样性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种查询感知的hub原型学习方法，通过显式建模支持集和查询集之间的语义相关性，生成与查询相关的hub原型并优化其分布，有效解决了小样本3D点云语义分割中的原型偏差问题，显著提升了分割性能。'}


### 论文摘要

Few-shot 3D point cloud semantic segmentation (FS-3DSeg) aims to segment novel classes with only a few labeled samples. However, existing metric-based prototype learning methods generate prototypes solely from the support set, without considering their relevance to query data. This often results in prototype bias, where prototypes overfit support-specific characteristics and fail to generalize to the query distribution, especially in the presence of distribution shifts, which leads to degraded segmentation performance. To address this issue, we propose a novel Query-aware Hub Prototype (QHP) learning method that explicitly models semantic correlations between support and query sets. Specifically, we propose a Hub Prototype Generation (HPG) module that constructs a bipartite graph connecting query and support points, identifies frequently linked support hubs, and generates query-relevant prototypes that better capture cross-set semantics. To further mitigate the influence of bad hubs and ambiguous prototypes near class boundaries, we introduce a Prototype Distribution Optimization (PDO) module, which employs a purity-reweighted contrastive loss to refine prototype representations by pulling bad hubs and outlier prototypes closer to their corresponding class centers. Extensive experiments on S3DIS and ScanNet demonstrate that QHP achieves substantial performance gains over state-of-the-art methods, effectively narrowing the semantic gap between prototypes and query sets in FS-3DSeg.

---

## 48. Geometry-Aware Sparse Depth Sampling for High-Fidelity RGB-D Depth Completion in Robotic Systems

**论文链接:** [http://arxiv.org/abs/2512.08229v1](http://arxiv.org/abs/2512.08229v1)

**作者:** Tony Salloom, Dandi Zhou, Xinhai Sun

**发布时间:** 2025-12-09

### GPT解析

### 总结

本研究提出了一种基于法线引导的稀疏深度采样策略，用于改进深度补全方法的性能。通过利用PCA-based表面法线估计计算每像素深度可靠性度量，并根据可靠性分布抽取稀疏深度样本，该方法能够生成更真实的稀疏深度数据。与基于扩散的深度补全模型Marigold-DC集成后，在NYU Depth v2数据集上的实验表明，该方法提高了深度补全的准确性，减少了边缘和不连续处的伪影，并产生了更真实的训练条件。

### 背景

准确的三维感知对执行操作、检查和导航任务的现代工业机器人系统至关重要。RGB-D和立体视觉传感器常用于此目的，但由于传感器限制和环境条件，它们产生的深度图通常存在噪声、不完整或有偏差的问题。深度补全方法旨在从RGB图像和稀疏深度输入生成密集、可靠的深度图。

### 目的

解决当前深度补全流程中生成不真实稀疏深度的问题，即稀疏像素通常从密集的真实深度中均匀随机选择，忽略了真实传感器表现出的与几何相关和空间非均匀的可靠性特性。

### 方法

提出了一种基于法线引导的稀疏深度采样策略，利用基于PCA的表面法线估计在RGB-D点云上计算每像素深度可靠性度量，然后根据此可靠性分布抽取稀疏深度样本。将此采样方法与基于扩散的深度补全模型Marigold-DC集成，并在NYU Depth v2数据集上使用标准指标进行评估。

### 主要发现

实验表明，几何感知的稀疏深度提高了深度补全的准确性，减少了边缘和不连续处的伪影，并产生了更真实的训练条件，这些条件更好地反映了真实传感器的行为。

### 结论

通过引入几何感知的稀疏深度采样策略，可以显著改善深度补全的性能，使生成的深度图更加可靠和真实，更好地满足工业机器人系统对三维感知的需求。

### 翻译

准确的三维感知对于执行操作、检查和导航任务的现代工业机器人系统至关重要。RGB-D和立体视觉传感器广泛用于此目的，但由于传感器限制和环境条件，它们产生的深度图通常存在噪声、不完整或有偏差。深度补全方法旨在从RGB图像和稀疏深度输入生成密集、可靠的深度图。然而，当前深度补全流程的一个关键局限是不真实地生成稀疏深度：稀疏像素通常从密集的真实深度中均匀随机选择，忽略了真实传感器表现出与几何相关和空间非均匀的可靠性这一事实。在本工作中，我们提出了一种基于法线引导的稀疏深度采样策略，利用基于PCA的表面法线估计在RGB-D点云上计算每像素深度可靠性度量。然后根据此可靠性分布抽取稀疏深度样本。我们将此采样方法与基于扩散的深度补全模型Marigold-DC集成，并在NYU Depth v2上使用标准指标进行评估。实验表明，我们的几何感知稀疏深度提高了准确性，减少了边缘和不连续处的伪影，并产生了更真实的训练条件，这些条件更好地反映了真实传感器的行为。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决当前深度补全方法中稀疏深度采样的非现实性问题。现有的方法通常从密集真实深度图中均匀随机选择稀疏像素，忽略了真实传感器表现出的几何依赖性和空间非均匀可靠性。这个问题在工业机器人领域尤为重要，因为准确的三维感知对操作、检查和导航任务至关重要，而传感器噪声和不完整深度会显著影响下游任务的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析RGB-D和立体相机深度图中误差的来源，认识到误差统计强烈依赖于局部几何形状、遮挡和纹理，且可靠性在空间上分布不均。他们借鉴了基于PCA的表面法线估计技术，以及深度补全领域的研究，特别是Marigold-DC扩散模型。作者设计了一种基于法线的稀疏深度采样策略，利用局部表面法线和曲率来计算深度可靠性分数，使采样更符合真实传感器的行为特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一种几何感知的稀疏深度采样策略，不再均匀随机选择稀疏深度像素，而是根据局部表面几何特征计算每像素的深度可靠性分数，并据此进行采样。整体流程包括：1) 点云提取：从RGB图像和深度图生成3D点云；2) 表面法线估计：使用PCA分析局部邻域计算表面法线和曲率；3) 深度可靠性度量：基于法线与视线角度计算可靠性分数；4) 按可靠性概率采样稀疏深度点；5) 与Marigold-DC模型集成，用于深度补全。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出基于PCA表面法线估计的几何感知采样策略；2) 根据可靠性分布而非均匀随机选择稀疏深度点；3) 与扩散模型Marigold-DC集成评估。相比之前工作，不同之处在于：不再假设所有像素同等可靠，而是利用几何线索模拟真实传感器的异构可靠性模式；提供了更符合实际传感器行为的训练信号；考虑了真实传感器产生的非均匀和场景相关的采样模式，而非固定、随机的稀疏性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于几何感知的稀疏深度采样方法，通过PCA估计表面法线和深度可靠性，改善了深度补全模型在机器人系统中的性能，使生成的深度图更接近真实传感器的行为特征。'}


### 论文摘要

Accurate three-dimensional perception is essential for modern industrial robotic systems that perform manipulation, inspection, and navigation tasks. RGB-D and stereo vision sensors are widely used for this purpose, but the depth maps they produce are often noisy, incomplete, or biased due to sensor limitations and environmental conditions. Depth completion methods aim to generate dense, reliable depth maps from RGB images and sparse depth input. However, a key limitation in current depth completion pipelines is the unrealistic generation of sparse depth: sparse pixels are typically selected uniformly at random from dense ground-truth depth, ignoring the fact that real sensors exhibit geometry-dependent and spatially nonuniform reliability. In this work, we propose a normal-guided sparse depth sampling strategy that leverages PCA-based surface normal estimation on the RGB-D point cloud to compute a per-pixel depth reliability measure. The sparse depth samples are then drawn according to this reliability distribution. We integrate this sampling method with the Marigold-DC diffusion-based depth completion model and evaluate it on NYU Depth v2 using the standard metrics. Experiments show that our geometry-aware sparse depth improves accuracy, reduces artifacts near edges and discontinuities, and produces more realistic training conditions that better reflect real sensor behavior.

---

## 49. Fused Gromov-Wasserstein Contrastive Learning for Effective Enzyme-Reaction Screening

**论文链接:** [http://arxiv.org/abs/2512.08508v1](http://arxiv.org/abs/2512.08508v1)

**作者:** Gengmo Zhou, Feng Yu, Wenda Wang, Zhifeng Gao, Guolin Ke, Zhewei Wei, Zhen Wang

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种名为FGW-CLIP的新型对比学习框架，基于优化融合Gromov-Wasserstein距离，用于高效酶识别和筛选，在多个基准测试中取得了最先进性能。

### 背景

酶是关键的生物催化剂，从大量蛋白质库中高效识别特定酶对推动生物催化至关重要。传统计算方法耗时且资源密集，而现有深度学习方法只关注酶与反应间的相互作用，忽略了各自领域内的层次关系。

### 目的

解决传统酶筛选方法的局限性，同时克服现有深度学习方法忽略酶和反应各自领域内固有层次关系的问题。

### 方法

引入FGW-CLIP框架，采用基于优化融合Gromov-Wasserstein距离的对比学习方法，纳入跨域对齐（反应与酶之间）和域内对齐（酶和反应内部）。通过定制正则化项最小化酶和反应空间间的距离，增强域间信息整合。

### 主要发现

FGW-CLIP在具有挑战性的酶-反应任务中表现优越。在EnzymeMap基准测试中，通过BEDROC和EF指标衡量的酶虚拟筛选性能达到最先进水平。在ReactZyme（最大酶-反应基准）的所有三个分割中均表现优异，显示出对新酶和新反应的强大泛化能力。

### 结论

FGW-CLIP是复杂生化环境中酶发现的有前景框架，在各类筛选场景中具有强大适应性。

### 翻译

酶是促进广泛生化反应的关键催化剂。从大量蛋白质库中高效识别特定酶对于推动生物催化至关重要。传统的酶筛选和检索计算方法耗时且资源密集。最近，深度学习方法显示出潜力。然而，这些方法仅关注酶与反应之间的相互作用，忽略了各自领域内的固有层次关系。为解决这些局限性，我们引入了FGW-CLIP，一种基于优化融合Gromov-Wasserstein距离的新型对比学习框架。FGW-CLIP纳入了多种对齐方式，包括反应与酶之间的跨域对齐，以及酶和反应内部的域内对齐。通过引入定制的正则化项，我们的方法最小化了酶和反应空间之间的Gromov-Wasserstein距离，从而增强了这些域间的信息整合。广泛的评估证明了FGW-CLIP在具有挑战性的酶-反应任务中的优越性。在广泛使用的EnzymeMap基准测试中，FGW-CLIP在酶虚拟筛选方面取得了最先进的性能，通过BEDROC和EF指标衡量。此外，FGW-CLIP在最大的酶-反应基准测试ReactZyme的所有三个分割中均表现优于其他方法，显示出对新酶和新反应的强大泛化能力。这些结果将FGW-CLIP定位为复杂生化环境中酶发现的有前景框架，在各类筛选场景中具有强大的适应性。


### 论文摘要

Enzymes are crucial catalysts that enable a wide range of biochemical reactions. Efficiently identifying specific enzymes from vast protein libraries is essential for advancing biocatalysis. Traditional computational methods for enzyme screening and retrieval are time-consuming and resource-intensive. Recently, deep learning approaches have shown promise. However, these methods focus solely on the interaction between enzymes and reactions, overlooking the inherent hierarchical relationships within each domain. To address these limitations, we introduce FGW-CLIP, a novel contrastive learning framework based on optimizing the fused Gromov-Wasserstein distance. FGW-CLIP incorporates multiple alignments, including inter-domain alignment between reactions and enzymes and intra-domain alignment within enzymes and reactions. By introducing a tailored regularization term, our method minimizes the Gromov-Wasserstein distance between enzyme and reaction spaces, which enhances information integration across these domains. Extensive evaluations demonstrate the superiority of FGW-CLIP in challenging enzyme-reaction tasks. On the widely-used EnzymeMap benchmark, FGW-CLIP achieves state-of-the-art performance in enzyme virtual screening, as measured by BEDROC and EF metrics. Moreover, FGW-CLIP consistently outperforms across all three splits of ReactZyme, the largest enzyme-reaction benchmark, demonstrating robust generalization to novel enzymes and reactions. These results position FGW-CLIP as a promising framework for enzyme discovery in complex biochemical settings, with strong adaptability across diverse screening scenarios.

---

## 50. BUT Systems for Environmental Sound Deepfake Detection in the ESDD 2026 Challenge

**论文链接:** [http://arxiv.org/abs/2512.08319v1](http://arxiv.org/abs/2512.08319v1)

**作者:** Junyi Peng, Lin Zhang, Jin Li, Oldrich Plchot, Jan Cernocky

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文介绍了BUT参加ESDD 2026挑战赛的提交作品，专注于赛道1的环境声音深度伪造检测，提出了一种强大的集成框架来处理未见过的生成器算法。

### 背景

这是BUT团队参加ESDD 2026挑战赛的作品，特别针对赛道1：使用未见过的生成器的环境声音深度伪造检测任务。

### 目的

解决模型对未见过的合成算法生成的音频进行泛化的关键挑战。

### 方法

提出了一种基于多样化自监督学习模型的集成框架，分析了通用音频SSL模型（BEATs、EAT、Dasheng）和语音特定SSL模型，结合轻量级多头因子化注意力后端捕获判别性表示，并引入基于分布不确定性建模的特征域增强策略增强鲁棒性，所有模型仅使用官方EnvSDD数据训练。

### 主要发现

最佳单一系统在开发集、进度集和最终评估集上分别实现了0.00%、4.60%和4.80%的等错误率；融合系统进一步提高了泛化能力，在同一数据分区上实现了0.00%、3.52%和4.38%的等错误率。

### 结论

所提出的方法在环境声音深度伪造检测任务中表现出色，特别是在处理未见过的生成器方面具有显著优势。

### 翻译

本文描述了BUT参加ESDD 2026挑战赛的提交作品，特别专注于赛道1：使用未见过的生成器的环境声音深度伪造检测。为解决对未见过的合成算法生成的音频进行泛化的关键挑战，我们提出了一个强大的集成框架，利用多样化的自监督学习模型。我们对通用音频SSL模型（包括BEATs、EAT和Dasheng）和语音特定的SSL模型进行了全面分析。这些前端与轻量级的多头因子化注意力后端相结合，以捕获判别性表示。此外，我们引入了一种基于分布不确定性建模的特征域增强策略，增强模型对未见过的频谱失真的鲁棒性。所有模型仅在官方的EnvSDD数据上进行训练，不使用任何外部资源。实验结果证明了我们方法的有效性：我们的最佳单一系统在开发集、进度集（赛道1）和最终评估集上分别实现了0.00%、4.60%和4.80%的等错误率。融合系统进一步提高了泛化能力，在同一数据分区上实现了0.00%、3.52%和4.38%的等错误率。


### 论文摘要

This paper describes the BUT submission to the ESDD 2026 Challenge, specifically focusing on Track 1: Environmental Sound Deepfake Detection with Unseen Generators. To address the critical challenge of generalizing to audio generated by unseen synthesis algorithms, we propose a robust ensemble framework leveraging diverse Self-Supervised Learning (SSL) models. We conduct a comprehensive analysis of general audio SSL models (including BEATs, EAT, and Dasheng) and speech-specific SSLs. These front-ends are coupled with a lightweight Multi-Head Factorized Attention (MHFA) back-end to capture discriminative representations. Furthermore, we introduce a feature domain augmentation strategy based on distribution uncertainty modeling to enhance model robustness against unseen spectral distortions. All models are trained exclusively on the official EnvSDD data, without using any external resources. Experimental results demonstrate the effectiveness of our approach: our best single system achieved Equal Error Rates (EER) of 0.00\%, 4.60\%, and 4.80\% on the Development, Progress (Track 1), and Final Evaluation sets, respectively. The fusion system further improved generalization, yielding EERs of 0.00\%, 3.52\%, and 4.38\% across the same partitions.

---

## 51. Beyond Traditional Diagnostics: Transforming Patient-Side Information into Predictive Insights with Knowledge Graphs and Prototypes

**论文链接:** [http://arxiv.org/abs/2512.08261v1](http://arxiv.org/abs/2512.08261v1)

**作者:** Yibowen Zhao, Yinan Zhang, Zhixiang Su, Lizhen Cui, Chunyan Miao

**发布时间:** 2025-12-09

**备注:** This work has been accepted by ICDE 2026 and is available on arXiv for early access

### GPT解析

### 总结

本文提出了KPI框架，通过整合医学知识图谱、构建疾病原型和利用对比学习提高疾病预测准确性，特别是对长尾疾病，同时使用大型语言模型提供患者特定的医学解释，增强预测的可解释性和可靠性。

### 背景

仅从患者侧信息（如人口统计和自我报告症状）预测疾病已成为研究热点，可增强患者意识、促进早期医疗参与并提高医疗系统效率。然而，现有方法面临疾病分布不平衡和缺乏可解释性的挑战，导致预测存在偏差或不可靠。

### 目的

解决现有方法面临的疾病分布不平衡和缺乏可解释性问题，提高预测准确性（特别是对长尾疾病），并提供患者特定的医学相关解释，增强可解释性和可靠性。

### 方法

提出知识图谱增强、原型感知和可解释性（KPI）框架，整合结构化和可信的医疗知识到统一疾病知识图谱，构建临床意义的疾病原型，采用对比学习提高预测准确性，并利用大型语言模型生成患者特定的医学解释。

### 主要发现

在真实世界数据集上的实验表明，KPI在预测准确性上优于最先进方法，且提供的临床解释与患者叙述高度一致。

### 结论

KPI框架在以患者为中心的医疗保健交付方面具有实用价值，能有效提高疾病预测的准确性和可解释性。

### 翻译

仅从患者侧信息（如人口统计和自我报告的症状）预测疾病已引起显著研究关注，因为它有可能增强患者意识，促进早期医疗参与，并提高医疗系统效率。然而，现有方法面临关键挑战，包括疾病分布不平衡和缺乏可解释性，导致预测存在偏差或不可靠。为了解决这些问题，我们提出了知识图谱增强、原型感知和可解释性（KPI）框架。KPI系统地将结构化和可信的医疗知识整合到统一的疾病知识图谱中，构建具有临床意义的疾病原型，并采用对比学习来提高预测准确性，这对于长尾疾病尤为重要。此外，KPI利用大型语言模型（LLMs）生成患者特定的、医学相关的解释，从而提高可解释性和可靠性。在真实世界数据集上的广泛实验表明，KPI在预测准确性上优于最先进的方法，并提供与患者叙述密切一致的临床上有效的解释，突显了其在以患者为中心的医疗保健交付方面的实用价值。


### 论文摘要

Predicting diseases solely from patient-side information, such as demographics and self-reported symptoms, has attracted significant research attention due to its potential to enhance patient awareness, facilitate early healthcare engagement, and improve healthcare system efficiency. However, existing approaches encounter critical challenges, including imbalanced disease distributions and a lack of interpretability, resulting in biased or unreliable predictions. To address these issues, we propose the Knowledge graph-enhanced, Prototype-aware, and Interpretable (KPI) framework. KPI systematically integrates structured and trusted medical knowledge into a unified disease knowledge graph, constructs clinically meaningful disease prototypes, and employs contrastive learning to enhance predictive accuracy, which is particularly important for long-tailed diseases. Additionally, KPI utilizes large language models (LLMs) to generate patient-specific, medically relevant explanations, thereby improving interpretability and reliability. Extensive experiments on real-world datasets demonstrate that KPI outperforms state-of-the-art methods in predictive accuracy and provides clinically valid explanations that closely align with patient narratives, highlighting its practical value for patient-centered healthcare delivery.

---

## 52. PolyLingua: Margin-based Inter-class Transformer for Robust Cross-domain Language Detection

**论文链接:** [http://arxiv.org/abs/2512.08143v1](http://arxiv.org/abs/2512.08143v1)

**作者:** Ali Lotfi Rezaabad, Bikram Khanal, Shashwat Chaurasia, Lu Zeng, Dezhi Hong, Hossein Beshashati, Thomas Butler, Megan Ganji

**发布时间:** 2025-12-09

### GPT解析

### 总结

PolyLingua是一个轻量级的基于Transformer的语言检测模型，用于领域内语言检测和细粒度语言分类，在两个具有挑战性的数据集上取得了高准确率，同时参数量大大减少。

### 背景

语言识别在多语言系统中是关键的第一步，如聊天机器人和虚拟助手。错误可能导致下游失败，因此对准确性要求很高。现有的语言识别工具在处理某些关键情况（如音乐请求中歌曲标题和用户语言不同的情况）时存在困难。开源工具如LangDetect和FastText速度快但准确率较低，而大型语言模型虽然有效但在低延迟或低资源环境中往往成本过高。

### 目的

开发一个轻量级的语言检测模型，能够在保持高准确率的同时适用于计算和延迟受限的环境。

### 方法

PolyLingua采用基于Transformer的轻量级模型，使用两级对比学习框架，结合实例级分离和类级对齐以及自适应边界，即使对于密切相关的语言也能产生紧凑且良好分离的嵌入。

### 主要发现

在Amazon Massive（多语言数字助手语音）和Song数据集（音乐请求，经常有代码切换）两个具有挑战性的数据集上，PolyLingua分别实现了99.25%和98.15%的F1分数，超越了Sonnet 3.5，同时使用的参数量减少了10倍。

### 结论

PolyLingua是一个高效的语言检测模型，特别适合计算和延迟受限的环境，能够提供高准确率的语言识别。

### 翻译

语言识别是多语言系统（如聊天机器人和虚拟助手）中的关键第一步，能够实现语言和文化上准确的用户体验。此阶段的错误可能会级联到下游失败，因此对准确性要求很高。然而，现有的语言识别工具在处理关键情况时存在困难，例如在音乐请求中歌曲标题和用户语言不同的情况。像LangDetect、FastText这样的开源工具速度快但准确率较低，而大型语言模型虽然有效但在低延迟或低资源环境中往往成本过高。我们引入了PolyLingua，一个用于领域内语言检测和细粒度语言分类的轻量级基于Transformer的模型。它采用两级对比学习框架，结合实例级分离和类级对齐以及自适应边界，即使对于密切相关的语言也能产生紧凑且良好分离的嵌入。在两个具有挑战性的数据集上进行了评估——Amazon Massive（多语言数字助手语音）和Song数据集（音乐请求，经常有代码切换）——PolyLingua分别实现了99.25%和98.15%的F1分数，超越了Sonnet 3.5，同时使用的参数量减少了10倍，使其成为计算和延迟受限环境的理想选择。


### 论文摘要

Language identification is a crucial first step in multilingual systems such as chatbots and virtual assistants, enabling linguistically and culturally accurate user experiences. Errors at this stage can cascade into downstream failures, setting a high bar for accuracy. Yet, existing language identification tools struggle with key cases -- such as music requests where the song title and user language differ. Open-source tools like LangDetect, FastText are fast but less accurate, while large language models, though effective, are often too costly for low-latency or low-resource settings. We introduce PolyLingua, a lightweight Transformer-based model for in-domain language detection and fine-grained language classification. It employs a two-level contrastive learning framework combining instance-level separation and class-level alignment with adaptive margins, yielding compact and well-separated embeddings even for closely related languages. Evaluated on two challenging datasets -- Amazon Massive (multilingual digital assistant utterances) and a Song dataset (music requests with frequent code-switching) -- PolyLingua achieves 99.25% F1 and 98.15% F1, respectively, surpassing Sonnet 3.5 while using 10x fewer parameters, making it ideal for compute- and latency-constrained environments.

---

## 53. Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts

**论文链接:** [http://arxiv.org/abs/2512.08814v1](http://arxiv.org/abs/2512.08814v1)

**作者:** Yifan Lyu, Liang Zhang

**发布时间:** 2025-12-09

### GPT解析

### 总结

ROME是一种新型人格检测框架，通过将心理学知识明确注入到检测过程中，利用大型语言模型模拟用户对心理测量问卷的响应，生成可解释的、基于问卷的证据，并将问答作为人格检测的辅助任务，显著提高了检测性能。

### 背景

理解人类个性对网络应用如个性化推荐和心理健康评估至关重要。现有研究主要采用'帖子->用户向量->标签'的建模范式，但这种方法受限于标签稀缺导致的监督信号有限，以及用户语言与抽象心理结构之间语义映射不够明确的问题。

### 目的

提出ROME框架，通过明确注入心理学知识来解决人格检测中的标签稀缺和语义映射不明确问题。

### 方法

受标准化自我评估测试启发，ROME利用大型语言模型的角色扮演能力模拟用户对心理测量问卷的响应，将自由形式的用户帖子转化为可解释的、基于问卷的证据。采用基于问题的专家混合模块联合处理帖子和问题表示，在多任务学习框架中将问答作为人格检测的辅助任务。

### 主要发现

在两个真实世界数据集上的实验表明，ROME始终优于最先进的基线方法，在Kaggle数据集上实现了15.41%的性能改进。

### 结论

ROME框架通过注入心理学知识和利用大型语言模型的角色扮演能力，有效解决了人格检测中的标签稀缺和语义映射问题，多任务学习框架中问答作为辅助任务的方法显著提高了人格检测的性能。

### 翻译

理解人类个性对网络应用如个性化推荐和心理健康评估至关重要。现有的人格检测研究主要采用'帖子->用户向量->标签'的建模范式，将社交媒体帖子编码为用户表示用于预测人格标签（如MBTI标签）。尽管大型语言模型的最新进展提高了文本编码能力，但这些方法仍受限于标签稀缺导致的监督信号有限，以及用户语言与抽象心理结构之间语义映射不够明确的问题。我们通过提出ROME这一新型框架来解决这些挑战，该框架明确将心理学知识注入到人格检测中。受标准化自我评估测试启发，ROME利用大型语言模型的角色扮演能力来模拟用户对验证过的心理测量问卷的响应。这些生成的问答级别回答将自由形式的用户帖子转化为可解释的、基于问卷的证据，将语言线索与人格标签联系起来，从而提供丰富的中间监督以减轻标签稀缺问题，同时提供语义推理链来指导文本到人格映射的学习。随后，基于问题的专家混合模块联合处理帖子和问题表示，在明确监督下学习回答问卷项目。预测的答案被汇总到一个可解释的答案向量中，并在多任务学习框架内与用户表示融合用于最终预测，其中问答作为人格检测的强大辅助任务。在两个真实世界数据集上的大量实验表明，ROME始终优于最先进的基线方法，在Kaggle数据集上实现了15.41%的改进。


### 论文摘要

Understanding human personality is crucial for web applications such as personalized recommendation and mental health assessment. Existing studies on personality detection predominantly adopt a "posts -> user vector -> labels" modeling paradigm, which encodes social media posts into user representations for predicting personality labels (e.g., MBTI labels). While recent advances in large language models (LLMs) have improved text encoding capacities, these approaches remain constrained by limited supervision signals due to label scarcity, and under-specified semantic mappings between user language and abstract psychological constructs. We address these challenges by proposing ROME, a novel framework that explicitly injects psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME leverages LLMs' role-play capability to simulate user responses to validated psychometric questionnaires. These generated question-level answers transform free-form user posts into interpretable, questionnaire-grounded evidence linking linguistic cues to personality labels, thereby providing rich intermediate supervision to mitigate label scarcity while offering a semantic reasoning chain that guides and simplifies the text-to-personality mapping learning. A question-conditioned Mixture-of-Experts module then jointly routes over post and question representations, learning to answer questionnaire items under explicit supervision. The predicted answers are summarized into an interpretable answer vector and fused with the user representation for final prediction within a multi-task learning framework, where question answering serves as a powerful auxiliary task for personality detection. Extensive experiments on two real-world datasets demonstrate that ROME consistently outperforms state-of-the-art baselines, achieving improvements (15.41% on Kaggle dataset).

---

## 54. GeoDiffMM: Geometry-Guided Conditional Diffusion for Motion Magnification

**论文链接:** [http://arxiv.org/abs/2512.08325v1](http://arxiv.org/abs/2512.08325v1)

**作者:** Xuedeng Liu, Jiabao Guo, Zheng Zhang, Fei Wang, Zhi Liu, Dan Guo

**发布时间:** 2025-12-09

### GPT解析

### 总结

GeoDiffMM是一种基于扩散的拉格朗日视频运动放大框架，利用光流作为几何线索，实现结构一致的运动放大。该框架包含无噪声光流增强策略、扩散运动放大器和基于流的视频合成三个主要部分，在真实和合成数据集上优于现有方法，显著提高了运动放大效果。

### 背景

视频运动放大技术能够将细微的宏观运动放大到可感知的水平。现有的主流欧拉方法通过解耦表示学习(如纹理、形状和频率方案)来解决放大引起的噪声问题，但当运动位移非常小时，仍然难以将光子噪声与真实微观运动分离。

### 目的

开发一种能够有效分离光子噪声与真实微观运动，实现结构一致运动放大的视频运动放大框架。

### 方法

GeoDiffMM框架包含三个主要部分：1)无噪声光流增强策略：合成多样化的无光子噪声的非刚性运动场作为监督，帮助模型学习更准确的几何感知光流并提高泛化能力；2)扩散运动放大器：将去噪过程条件化于光流作为几何先验和可学习的放大因子控制幅度，选择性放大与场景语义和结构一致的运动分量；3)基于流的视频合成：将放大的运动映射回图像域，实现高保真度。

### 主要发现

在真实和合成数据集上的大量实验表明，GeoDiffMM优于最先进的方法，并显著提高了运动放大效果。

### 结论

GeoDiffMM通过结合扩散模型和光流作为几何线索，有效解决了微小运动位移下光子噪声与真实微观运动分离的难题，实现了结构一致的高质量运动放大。

### 翻译

视频运动放大将细微的宏观运动放大到可感知的水平。最近，现有的主流欧拉方法通过解耦表示学习(如纹理、形状和频率方案)来解决放大引起的噪声问题，但当运动位移非常小时，它们仍然难以将光子噪声与真实微观运动分离。我们提出了GeoDiffMM，一种基于扩散的拉格朗日视频运动放大框架，以光流作为几何线索，实现结构一致的运动放大。具体来说，我们设计了一种无噪声光流增强策略，合成多样化的无光子噪声的非刚性运动场作为监督，帮助模型学习更准确的几何感知光流并提高泛化能力。接下来，我们开发了一种扩散运动放大器，将去噪过程条件化于光流作为几何先验和可学习的放大因子控制幅度，从而选择性放大与场景语义和结构一致的运动分量，同时抑制内容无关的扰动。最后，我们进行基于流的视频合成，将放大的运动映射回图像域，实现高保真度。在真实和合成数据集上的大量实验表明，GeoDiffMM优于最先进的方法，并显著提高了运动放大效果。


### 论文摘要

Video Motion Magnification (VMM) amplifies subtle macroscopic motions to a perceptible level. Recently, existing mainstream Eulerian approaches address amplification-induced noise via decoupling representation learning such as texture, shape and frequancey schemes, but they still struggle to separate photon noise from true micro-motion when motion displacements are very small. We propose GeoDiffMM, a novel diffusion-based Lagrangian VMM framework conditioned on optical flow as a geometric cue, enabling structurally consistent motion magnification. Specifically, we design a Noise-free Optical Flow Augmentation strategy that synthesizes diverse nonrigid motion fields without photon noise as supervision, helping the model learn more accurate geometry-aware optial flow and generalize better. Next, we develop a Diffusion Motion Magnifier that conditions the denoising process on (i) optical flow as a geometry prior and (ii) a learnable magnification factor controlling magnitude, thereby selectively amplifying motion components consistent with scene semantics and structure while suppressing content-irrelevant perturbations. Finally, we perform Flow-based Video Synthesis to map the amplified motion back to the image domain with high fidelity. Extensive experiments on real and synthetic datasets show that GeoDiffMM outperforms state-of-the-art methods and significantly improves motion magnification.

---

## 55. Persistent Topological Structures and Cohomological Flows as a Mathematical Framework for Brain-Inspired Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.08241v1](http://arxiv.org/abs/2512.08241v1)

**作者:** Preksha Girish, Rachana Mysore, Mahanthesha U, Shrey Kumar, Shipra Prashant

**发布时间:** 2025-12-09

**备注:** 6 pages, 2 figures

### GPT解析

### 总结

本文提出了一种基于大脑启发的表示学习数学框架，该框架建立在持久拓扑结构和上同调流的相互作用基础上。

### 背景

神经计算需要新的数学基础来捕捉大脑状态中跨时间、空间和功能的不变量表示。

### 目的

开发一个数学上严谨的大脑启发表示学习框架，能够捕捉跨不同大脑状态的不变量。

### 方法

将神经计算重新表述为动态单纯复形上链映射的演化，整合代数拓扑和微分几何构建上同调算子，在同调景观中推广基于梯度的学习，并使用持久同调、层上同调和谱拉普拉斯ians分析数据。

### 主要发现

该模型在保持流形一致性和抗噪性方面优于图神经网络和基于流形的深度架构。

### 结论

建立了拓扑驱动表示学习的连贯数学基础，为大脑启发的表示学习提供了新的理论框架。

### 翻译

本文提出了一种基于大脑启发的表示学习的数学严谨框架，该框架建立在持久拓扑结构和上同调流的相互作用基础上。神经计算被重新表述为动态单纯复形上链映射的演化，使得能够捕捉跨时间、空间和功能大脑状态的不变量表示。所提出的架构将代数拓扑与微分几何相结合，构建在同调景观中推广基于梯度学习的上同调算子。使用持久同调、层上同调和谱拉普拉斯ians对具有受控拓扑特征的合成数据和真实神经数据进行联合分析，以量化稳定性、连续性和结构保持。实证结果表明，该模型在保持流形一致性和抗噪性方面优于图神经网络和基于流形的深度架构，为拓扑驱动的表示学习建立了连贯的数学基础。


### 论文摘要

This paper presents a mathematically rigorous framework for brain-inspired representation learning founded on the interplay between persistent topological structures and cohomological flows. Neural computation is reformulated as the evolution of cochain maps over dynamic simplicial complexes, enabling representations that capture invariants across temporal, spatial, and functional brain states. The proposed architecture integrates algebraic topology with differential geometry to construct cohomological operators that generalize gradient-based learning within a homological landscape. Synthetic data with controlled topological signatures and real neural datasets are jointly analyzed using persistent homology, sheaf cohomology, and spectral Laplacians to quantify stability, continuity, and structural preservation. Empirical results demonstrate that the model achieves superior manifold consistency and noise resilience compared to graph neural and manifold-based deep architectures, establishing a coherent mathematical foundation for topology-driven representation learning.

---

## 56. PR-CapsNet: Pseudo-Riemannian Capsule Network with Adaptive Curvature Routing for Graph Learning

**论文链接:** [http://arxiv.org/abs/2512.08218v1](http://arxiv.org/abs/2512.08218v1)

**作者:** Ye Qin, Jingchao Wang, Yang Shi, Haiying Huang, Junxu Li, Weijian Liu, Tinghui Chen, Jinghui Qin

**发布时间:** 2025-12-09

**备注:** To appear in WSDM 2026 (ACM International Conference on Web Search and Data Mining)

### GPT解析

### 总结

本文提出了一种伪黎曼胶囊网络(PR-CapsNet)，通过将欧几里得胶囊路由扩展到测地线不连通的伪黎曼流形，解决了传统胶囊网络对现实世界图复杂几何建模不足的问题，显著提升了图表示学习性能。

### 背景

胶囊网络(CapsNets)虽通过动态路由和向量化分层表示展现出色图表示能力，但因固定曲率空间固有的测地线不连通问题，对现实世界图的复杂几何建模效果不佳，导致次优性能。虽然非欧几里得伪黎曼流形为图数据嵌入提供了特定归纳偏置，但如何利用它们改进CapsNets仍探索不足。

### 目的

将欧几里得胶囊路由扩展到测地线不连通的伪黎曼流形，推导出自适应曲率的伪黎曼胶囊网络(PR-CapsNet)，用于图表示学习。

### 方法

PR-CapsNet利用伪黎曼几何增强CapsNet：1)部署伪黎曼切线空间路由，通过微分同胚变换将胶囊状态分解为球面-时间和欧几里得-空间子空间；2)开发自适应曲率路由，通过具有局部流形几何特性的几何注意力的可学习曲率张量，自适应融合不同曲率空间特征；3)开发保持几何特性的伪黎曼胶囊分类器，将胶囊嵌入投影到切线空间，使用曲率加权的softmax进行分类。

### 主要发现

在节点和图分类基准上的大量实验表明，PR-CapsNet优于最先进模型，验证了其对复杂图结构的强大表示能力。PR-CapsNet能同时通过其多功能的伪黎曼度量建模层次结构和集群或循环图结构，优于单曲率或子空间划分方法。

### 结论

PR-CapsNet通过将胶囊网络与伪黎曼几何相结合，有效解决了传统方法对复杂图结构建模不足的问题，为图表示学习提供了新的有效途径。

### 翻译

胶囊网络(CapsNets)通过动态路由和向量化分层表示展现出色的图表示能力，但由于固定曲率空间固有的测地线不连通问题，它们对现实世界图的复杂几何建模效果不佳，导致次优性能。最近研究发现非欧几里得伪黎曼流形为嵌入图数据提供了特定的归纳偏置，但如何利用它们改进CapsNets仍探索不足。本文将欧几里得胶囊路由扩展到测地线不连通的伪黎曼流形，推导出自适应曲率的伪黎曼胶囊网络(PR-CapsNet)，用于图表示学习。具体而言，PR-CapsNet利用伪黎曼几何通过自适应伪黎曼切线空间路由增强CapsNet。与单曲率或子空间划分方法不同，PR-CapsNet通过其多功能的伪黎曼度量同时建模层次结构和集群或循环图结构。它首先部署伪黎曼切线空间路由，通过微分同胚变换将胶囊状态分解为球面-时间和欧几里得-空间子空间。然后，开发出自适应曲率路由，通过具有局部流形几何特性的几何注意力的可学习曲率张量，自适应融合不同曲率空间的特征以处理复杂图。最后，开发出保持几何特性的伪黎曼胶囊分类器，将胶囊嵌入投影到切线空间，并使用曲率加权的softmax进行分类。在节点和图分类基准上的大量实验表明，PR-CapsNet优于最先进模型，验证了PR-CapsNet对复杂图结构的强大表示能力。


### 论文摘要

Capsule Networks (CapsNets) show exceptional graph representation capacity via dynamic routing and vectorized hierarchical representations, but they model the complex geometries of real\-world graphs poorly by fixed\-curvature space due to the inherent geodesical disconnectedness issues, leading to suboptimal performance. Recent works find that non\-Euclidean pseudo\-Riemannian manifolds provide specific inductive biases for embedding graph data, but how to leverage them to improve CapsNets is still underexplored. Here, we extend the Euclidean capsule routing into geodesically disconnected pseudo\-Riemannian manifolds and derive a Pseudo\-Riemannian Capsule Network (PR\-CapsNet), which models data in pseudo\-Riemannian manifolds of adaptive curvature, for graph representation learning. Specifically, PR\-CapsNet enhances the CapsNet with Adaptive Pseudo\-Riemannian Tangent Space Routing by utilizing pseudo\-Riemannian geometry. Unlike single\-curvature or subspace\-partitioning methods, PR\-CapsNet concurrently models hierarchical and cluster or cyclic graph structures via its versatile pseudo\-Riemannian metric. It first deploys Pseudo\-Riemannian Tangent Space Routing to decompose capsule states into spherical\-temporal and Euclidean\-spatial subspaces with diffeomorphic transformations. Then, an Adaptive Curvature Routing is developed to adaptively fuse features from different curvature spaces for complex graphs via a learnable curvature tensor with geometric attention from local manifold properties. Finally, a geometric properties\-preserved Pseudo\-Riemannian Capsule Classifier is developed to project capsule embeddings to tangent spaces and use curvature\-weighted softmax for classification. Extensive experiments on node and graph classification benchmarks show PR\-CapsNet outperforms SOTA models, validating PR\-CapsNet's strong representation power for complex graph structures.

---

## 57. CAMO: Causality-Guided Adversarial Multimodal Domain Generalization for Crisis Classification

**论文链接:** [http://arxiv.org/abs/2512.08071v1](http://arxiv.org/abs/2512.08071v1)

**作者:** Pingchuan Ma, Chengshuai Zhao, Bohan Jiang, Saketh Vishnubhatla, Ujun Jeong, Alimohammad Beigi, Adrienne Raglin, Huan Liu

**发布时间:** 2025-12-08

### GPT解析

### 总结

该研究提出了一种因果引导的多模态域泛化(MMDG)框架，用于社交媒体中的危机分类任务，解决了现有方法在未见过的危机类型上泛化能力差的问题。

### 背景

社交媒体中的危机分类旨在从多模态帖子中提取可操作的灾难相关信息，这对提高态势感知能力和促进及时应急响应至关重要。现有方法主要利用深度学习融合文本和视觉线索，但在域内设置外表现不佳，特别是在未见过的危机类型上。

### 目的

解决现有方法在跨域危机分类中表现不佳的问题，提高模型在未见过的灾难场景中的泛化能力。

### 方法

提出因果引导的多模态域泛化(MMDG)框架，结合对抗性解耦和统一表示学习。对抗性目标鼓励模型分离并专注于域不变的因果特征，而统一表示则在不同模态特征间建立对齐，使单模态域泛化策略能够扩展到多模态学习。

### 主要发现

在不同数据集上的实验表明，该方法在未见过的灾难场景中取得了最佳性能，证明了其有效性和优越性。

### 结论

通过分离因果特征和对齐多模态表示，所提出的MMDG框架能够有效解决多模态危机分类中的域泛化挑战，提高模型在未见过的危机类型上的泛化能力。

### 翻译

社交媒体中的危机分类旨在从多模态帖子中提取可操作的灾难相关信息，这是提高态势感知能力和促进及时应急响应的关键任务。然而，危机类型的广泛变化使得在未见过的灾难上实现可泛化的性能成为一个持续的挑战。现有方法主要利用深度学习融合文本和视觉线索进行危机分类，在域内设置下取得了数值上合理的结果。然而，它们在未见过的危机类型上表现不佳，因为：1)它们没有分离虚假特征和因果特征，导致在域偏移下性能下降；2)它们无法在共享空间中对齐异构模态表示，这阻碍了已建立的单模态域泛化技术在多模态环境中的直接应用。为解决这些问题，我们引入了一个因果引导的多模态域泛化(MMDG)框架，结合对抗性解耦和统一表示学习进行危机分类。对抗性目标鼓励模型分离并专注于域不变的因果特征，从而实现基于稳定因果机制的更可泛化的分类。统一表示在不同模态的特征之间对齐，使单模态域泛化策略能够无缝扩展到多模态学习。在不同数据集上的实验表明，我们的方法在未见过的灾难场景中取得了最佳性能。


### 论文摘要

Crisis classification in social media aims to extract actionable disaster-related information from multimodal posts, which is a crucial task for enhancing situational awareness and facilitating timely emergency responses. However, the wide variation in crisis types makes achieving generalizable performance across unseen disasters a persistent challenge. Existing approaches primarily leverage deep learning to fuse textual and visual cues for crisis classification, achieving numerically plausible results under in-domain settings. However, they exhibit poor generalization across unseen crisis types because they 1. do not disentangle spurious and causal features, resulting in performance degradation under domain shift, and 2. fail to align heterogeneous modality representations within a shared space, which hinders the direct adaptation of established single-modality domain generalization (DG) techniques to the multimodal setting. To address these issues, we introduce a causality-guided multimodal domain generalization (MMDG) framework that combines adversarial disentanglement with unified representation learning for crisis classification. The adversarial objective encourages the model to disentangle and focus on domain-invariant causal features, leading to more generalizable classifications grounded in stable causal mechanisms. The unified representation aligns features from different modalities within a shared latent space, enabling single-modality DG strategies to be seamlessly extended to multimodal learning. Experiments on the different datasets demonstrate that our approach achieves the best performance in unseen disaster scenarios.

---

## 58. Mapping Still Matters: Coarse-Graining with Machine Learning Potentials

**论文链接:** [http://arxiv.org/abs/2512.07692v2](http://arxiv.org/abs/2512.07692v2)

**作者:** Franz Görlich, Julija Zavadlav

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究探讨了粗粒化建模中映射选择对等变机器学习势能(MLPs)学习表示的影响，发现键合与非键相互作用长度尺度重叠会导致非物理键排列，同时正确编码物种和保持立体化学对避免非物理对称性至关重要。

### 背景

粗粒化建模使分子模拟能够达到全原子方法无法企及的时间和长度尺度。对于经典CG模型，映射选择(原子如何分组为CG位点)是决定准确性和可转移性的关键因素。机器学习势能(MLPs)的出现为构建能够学习任何映射真实平均力势的CG模型提供了新机会。

### 目的

系统研究映射选择如何影响等变MLPs学习的表示，通过研究液态己烷、氨基酸和多聚丙氨酸系统进行验证。

### 方法

使用等变机器学习势能(MLPs)研究不同映射对模型表示的影响，分析液态己烷、氨基酸和多聚丙氨酸系统。

### 主要发现

当键合和非键相互作用的长度尺度重叠时，可能会出现非物理的键排列；正确编码物种和保持立体化学至关重要，因为忽略任何一方都会引入非物理对称性。

### 结论

研究结果为选择与现代架构兼容的CG映射提供了实用指导，并为开发可转移的CG模型提供指导。

### 翻译

粗粒化(CG)建模使分子模拟能够达到全原子方法无法达到的时间和长度尺度。对于经典CG模型，映射的选择，即原子如何分组为CG位点，是准确性和可转移性的主要决定因素。同时，机器学习势能(MLPs)的出现为构建CG模型提供了新机会，这些模型原则上可以学习任何映射的真实平均力势。在本工作中，我们通过研究液态己烷、氨基酸和多聚丙氨酸，系统研究了映射选择如何影响等变MLPs学习的表示。我们发现，当键合和非键相互作用的长度尺度重叠时，可能会出现非物理的键排列。我们还证明了正确编码物种和保持立体化学至关重要，因为忽略任何一方都会引入非物理对称性。我们的发现为选择与现代架构兼容的CG映射提供了实用指导，并指导了可转移CG模型的发展。


### 论文摘要

Coarse-grained (CG) modeling enables molecular simulations to reach time and length scales inaccessible to fully atomistic methods. For classical CG models, the choice of mapping, that is, how atoms are grouped into CG sites, is a major determinant of accuracy and transferability. At the same time, the emergence of machine learning potentials (MLPs) offers new opportunities to build CG models that can in principle learn the true potential of the mean force for any mapping. In this work, we systematically investigate how the choice of mapping influences the representations learned by equivariant MLPs by studying liquid hexane, amino acids, and polyalanine. We find that when the length scales of bonded and nonbonded interactions overlap, unphysical bond permutations can occur. We also demonstrate that correctly encoding species and maintaining stereochemistry are crucial, as neglecting either introduces unphysical symmetries. Our findings provide practical guidance for selecting CG mappings compatible with modern architectures and guide the development of transferable CG models.

---

