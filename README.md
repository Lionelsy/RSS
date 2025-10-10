# 今日论文推荐 - 2025-10-10

共 52 篇论文

---

## 1. Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.08442v1](http://arxiv.org/abs/2510.08442v1)

**作者:** Andrew Lee, Ian Chuang, Dechen Gao, Kai Fukazawa, Iman Soltani

**发布时间:** 2025-10-09

**备注:** Project page: https://andrewcwlee.github.io/gaze-on-the-prize

### GPT解析

### 总结

该论文提出了一种新的视觉强化学习方法，通过引入受人类视觉注视启发的可学习注意力机制，解决了高维图像数据中样本效率低下的问题。该方法利用回报引导的对比学习，使代理能够关注与任务相关的特征，从而提高学习效率和稳定性。

### 背景

视觉强化学习代理必须基于高维图像数据学习行动，其中只有一小部分像素与任务相关。这迫使代理在无关特征上浪费探索和计算资源，导致样本效率低下且学习不稳定。

### 目的

解决视觉强化学习中样本效率低下和不稳定学习的问题，引入一个受人类视觉注视启发的框架，使代理能够专注于任务相关特征。

### 方法

提出了'Gaze on the Prize'框架，通过可学习的中心注意力机制增强视觉RL，使用源自代理追求更高回报经验的自监督信号引导。通过回报差异揭示任务相关特征，实现回报引导的对比学习，训练注意力区分与成功和失败相关的特征，利用相似视觉表示的分组构建对比三元组提供训练信号。

### 主要发现

方法实现了高达2.4倍的样本效率提升，可以解决基线方法无法学习的任务，在ManiSkill3基准测试的一系列操作任务上得到验证。

### 结论

该框架无需修改底层算法或超参数即可显著提高视觉强化学习的效率和稳定性，使代理能够更有效地学习从高维图像数据中提取任务相关信息。

### 翻译

视觉强化学习代理必须基于高维图像数据学习行动，其中只有一小部分像素与任务相关。这迫使代理在无关特征上浪费探索和计算资源，导致样本效率低下且学习不稳定。为此，受人类视觉注视的启发，我们引入了'Gaze on the Prize'。该框架通过可学习的中心注意力机制增强了视觉RL，并由源自代理追求更高回报经验的自监督信号引导。我们的关键见解是回报差异揭示了什么最重要：如果两个相似表示产生不同结果，它们的区分特征很可能与任务相关，注视应相应地关注它们。这是通过回报引导的对比学习实现的，该训练注意力区分与成功和失败相关的特征。我们根据回报差异将相似的视觉表示分组为正样本和负样本，并使用生成的标签构建对比三元组。这些三元组提供训练信号，教导注意力机制为与不同结果相关的状态生成可区分的表示。我们的方法在样本效率上实现了高达2.4倍的提升，并且可以解决基线方法无法学习的任务，这已在ManiSkill3基准测试的一系列操作任务上得到证明，且无需修改底层算法或超参数。


### 论文摘要

Visual Reinforcement Learning (RL) agents must learn to act based on high-dimensional image data where only a small fraction of the pixels is task-relevant. This forces agents to waste exploration and computational resources on irrelevant features, leading to sample-inefficient and unstable learning. To address this, inspired by human visual foveation, we introduce Gaze on the Prize. This framework augments visual RL with a learnable foveal attention mechanism (Gaze), guided by a self-supervised signal derived from the agent's experience pursuing higher returns (the Prize). Our key insight is that return differences reveal what matters most: If two similar representations produce different outcomes, their distinguishing features are likely task-relevant, and the gaze should focus on them accordingly. This is realized through return-guided contrastive learning that trains the attention to distinguish between the features relevant to success and failure. We group similar visual representations into positives and negatives based on their return differences and use the resulting labels to construct contrastive triplets. These triplets provide the training signal that teaches the attention mechanism to produce distinguishable representations for states associated with different outcomes. Our method achieves up to 2.4x improvement in sample efficiency and can solve tasks that the baseline fails to learn, demonstrated across a suite of manipulation tasks from the ManiSkill3 benchmark, all without modifying the underlying algorithm or hyperparameters.

---

## 2. Contrastive Self-Supervised Learning at the Edge: An Energy Perspective

**论文链接:** [http://arxiv.org/abs/2510.08374v1](http://arxiv.org/abs/2510.08374v1)

**作者:** Fernanda Famá, Roberto Pereira, Charalampos Kalalas, Paolo Dini, Lorena Qendro, Fahim Kawsar, Mohammad Malekzadeh

**发布时间:** 2025-10-09

### GPT解析

### 总结

本研究评估了四种对比学习框架在资源受限设备上的部署可行性，重点关注能源消耗和训练数据减少条件下的表现，并发现SimCLR实际上具有最低的能源消耗。

### 背景

对比学习在自监督表征学习中显示出巨大潜力，但在资源受限设备上的部署研究仍然不足。传统CL框架的训练需要大量计算资源，对能源消耗、数据可用性和内存使用构成挑战。

### 目的

评估四种广泛使用的CL框架（SimCLR、MoCo、SimSiam和Barlow Twins）在边缘和雾计算环境中的实际可行性，并提供关于在处理能力有限的边缘/雾环境中部署CL的资源影响见解。

### 方法

对四种CL框架进行系统评估，引入包括能耗分析和减少训练数据条件在内的基准测试策略，并评估轻量级神经网络架构与CL框架配对时的表现。

### 主要发现

SimCLR尽管被认为计算成本高，但在各种数据条件下显示出最低的能源消耗，这与其普遍认知相反。

### 结论

研究为在处理能力有限的边缘/雾环境中部署对比学习提供了资源影响见解，并为未来优化开辟了几个研究方向。

### 翻译

尽管对比学习(CL)在自监督表征学习中显示出巨大潜力，但其资源受限设备上的部署在很大程度上仍未被探索。传统CL框架训练所需的巨大计算需求带来了一系列挑战，特别是在能源消耗、数据可用性和内存使用方面。我们对四种广泛使用的CL框架进行了评估：SimCLR、MoCo、SimSiam和Barlow Twins。我们关注这些CL框架在边缘和雾计算部署中的实际可行性，并引入了一种包括能耗分析和减少训练数据条件的系统基准测试策略。我们的研究结果显示，SimCLR与其感知的计算成本相反，在各种数据条件下表现出最低的能源消耗。最后，我们还通过评估轻量级神经网络架构与CL框架配对时的表现扩展了我们的分析。我们的研究旨在提供关于在处理能力有限的边缘/雾环境中部署CL的资源影响见解，并为其未来优化开辟了几个研究方向。


### 论文摘要

While contrastive learning (CL) shows considerable promise in self-supervised representation learning, its deployment on resource-constrained devices remains largely underexplored. The substantial computational demands required for training conventional CL frameworks pose a set of challenges, particularly in terms of energy consumption, data availability, and memory usage. We conduct an evaluation of four widely used CL frameworks: SimCLR, MoCo, SimSiam, and Barlow Twins. We focus on the practical feasibility of these CL frameworks for edge and fog deployment, and introduce a systematic benchmarking strategy that includes energy profiling and reduced training data conditions. Our findings reveal that SimCLR, contrary to its perceived computational cost, demonstrates the lowest energy consumption across various data regimes. Finally, we also extend our analysis by evaluating lightweight neural architectures when paired with CL frameworks. Our study aims to provide insights into the resource implications of deploying CL in edge/fog environments with limited processing capabilities and opens several research directions for its future optimization.

---

## 3. Channel Charting based Fast Beam Tracking Design and Implementation

**论文链接:** [http://arxiv.org/abs/2510.08144v1](http://arxiv.org/abs/2510.08144v1)

**作者:** Jiawei Zhang, Shihan Wang, Jienan Chen, Fan Wu, Jiyun Tao, Zheqi Gu

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文提出了一种基于信道映射的低开销波束跟踪算法，用于毫米波通信系统，显著减少了波束扫描时间同时保持高准确度。

### 背景

在第五代移动通信技术（B5G）和即将到来的第六代移动通信技术（6G）无线通信系统中，毫米波技术是提供额外带宽资源和缓解频谱拥堵的很有前景的解决方案。

### 目的

引入一种基于信道映射的低开销波束跟踪算法，显著减少跟踪过程中的波束扫描时间。

### 方法

通过将波束信息投影到信道映射中，将波束跟踪问题转换为在信道映射中获取波束簇；利用对比学习，将高维信道状态信息投影到保持空间邻近性的低维特征空间；使用动态候选波束获取策略，显著降低波束跟踪算法的复杂度。

### 主要发现

提出的算法在模拟环境中实现了98.27%的准确度；与现有方法相比，可以减少高达55.9%的波束扫描时间；现场测试展示了移动过程中的优秀通信质量。

### 结论

基于信道映射的低开销波束跟踪算法能够有效提高毫米波通信系统的波束跟踪性能，减少扫描时间同时保持高准确度。

### 翻译

在第五代移动通信技术（B5G）和即将到来的第六代移动通信技术（6G）无线通信系统中，毫米波技术是提供额外带宽资源和缓解频谱拥堵的很有前景的解决方案。波束跟踪是毫米波通信系统中提供可靠通信服务的关键程序，面临的挑战是提供持续且准确的跟踪性能。在本研究中，我们引入了一种基于信道映射的低开销波束跟踪算法，显著减少了跟踪过程中的波束扫描时间。通过将波束信息投影到信道映射中，波束跟踪问题转换为在信道映射中获取波束簇。利用对比学习，所提出的信道映射将高维信道状态信息投影到保持空间邻近性的低维特征空间。使用动态候选波束获取策略，我们显著降低了波束跟踪算法的复杂度。所提出的算法在保持高预测准确度的同时显著降低了扫描复杂度，在模拟环境中实现了98.27%的准确度。与现有方法相比，所提出的方法可以减少高达55.9%的波束扫描时间。此外，我们还进行了现场测试，测量结果展示了移动过程中的优秀通信质量。


### 论文摘要

In the beyond fifth-generation (B5G) and upcoming sixth-generation (6G) wireless communication systems, millimeter (mmWave) wave technology is a promising solution for offering additional bandwidth resources and mitigating spectrum congestion. Beam tracking is an essential procedure for providing reliable communication services in the mmWave communication system, with the challenge of providing consistent and accurate tracking performance. In this study, we introduce a low-overhead beam tracking algorithm based on channel charting, which significantly reduces beam scanning times during the tracking process. By projecting the beam information to the channel chart, the beam tracking problem is transformed into the acquisition of the beam cluster in the channel chart. Leveraging contrastive learning, the proposed channel chart projects high-dimensional channel state information into a low-dimensional feature space that preserves spatial proximities. Using a dynamic candidate beam acquisition strategy, the complexity of our beam tracking algorithm is significantly reduced. The proposed algorithm significantly reduces scanning complexity while maintaining high prediction accuracy, achieving an accuracy of 98.27\% in simulation environments. Compared to existing methods, the proposed method can reduce beam scanning times by up to 55.9\%. In addition, we also performed field tests, and the measured results demonstrated excellent communication quality during mobility.

---

## 4. SALAD-VAE: Semantic Audio Compression with Language-Audio Distillation

**论文链接:** [http://arxiv.org/abs/2510.07592v1](http://arxiv.org/abs/2510.07592v1)

**作者:** Sebastian Braun, Hannes Gamper, Dimitra Emmanouilidou

**发布时间:** 2025-10-08

**备注:** submitted to ICASSP 2026

### GPT解析

### 总结

SALAD-VAE是一种连续且高度紧凑的语义音频变分自编码器，在频域中运行，实现了低潜在帧率下的最先进压缩，同时揭示语义结构并产生高质量音频。

### 背景

现代生成和多模态模型越来越依赖于紧凑的潜在表示，这些表示在语义丰富度和高保真度重建之间进行权衡和平衡。

### 目的

引入SALAD-VAE，实现音频领域的高效压缩和高质量重建，同时保持语义结构。

### 方法

在频域中操作，使用7.8Hz的极低潜在帧率，增强标准VAE的语义损失和数据增强，采用对比学习和基于CLAP的嵌入蒸馏，构建计算复杂度显著低于同类模型的架构。

### 主要发现

SALAD-VAE能够在多样化的音频领域泛化，与最先进的VAE重建质量相匹配，同时在各种分类基准测试中持续优于它们，并提供训练好的CLAP投影层用于零样本任务。

### 结论

SALAD-VAE实现了高效音频表示，在低潜在帧率下保持高质量重建，具有多功能性，可用于零样本音频标题和分类任务。

### 翻译

现代生成和多模态模型越来越依赖于紧凑的潜在表示，这些表示在语义丰富度与高保真度重建之间进行权衡和平衡。我们引入了SALAD-VAE，一种连续且高度紧凑的语义音频变分自编码器，它在频域中运行，以极低的潜在帧率(7.8Hz)实现最先进的压缩，同时揭示语义结构并产生高质量的音频。我们增强了标准VAE的语义损失和数据增强，特别是对比学习和基于CLAP的嵌入蒸馏，使其能够泛化到多样化的音频领域。与可比的最先进VAE相比，SALAD-VAE的计算复杂度显著降低，同时匹配它们的重建质量，并在各种分类基准测试中持续优于它们。此外，提出的额外损失函数提供了训练好的CLAP投影层，可用于零样本音频标题和分类，与预训练的CLAP音频文本嵌入相匹配。


### 论文摘要

Modern generative and multimodal models increasingly rely on compact latent representations that trade and balance semantic richness with high-fidelity reconstruction. We introduce SALAD-VAE, a continuous and highly compact semantic Audio Variational Autoencoder, which operates in the frequency domain and achieves state-of-the-art compression with very low latent frame rate (7.8 Hz) while surfacing semantic structure and producing high audio quality. We enhance the standard VAE semantic losses and augmentation, specifically contrastive learning and CLAP-based embedding distillation, enabling it to generalize across diverse audio domains. With a significantly less computational complex architecture than comparable state-of-the-art VAEs, SALAD-VAE matches their reconstruction quality while it consistently outperforms them on a wide range of classification benchmarks. Furthermore, the proposed additional loss function provides a trained CLAP projection layer, which can be used zero-shot audio captioning and classification matching pretrained CLAP audio-text embeddings.

---

## 5. From Moments to Models: Graphon Mixture-Aware Mixup and Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.03690v2](http://arxiv.org/abs/2510.03690v2)

**作者:** Ali Azizpour, Reza Ramezanpour, Ashutosh Sabharwal, Santiago Segarra

**发布时间:** 2025-10-04

### GPT解析

### 总结

该研究提出了一种统一框架，用于处理现实世界图数据集中的混合群体问题，通过图矩聚类和解混合成分，改进了图对比学习和数据增强技术，在无监督和监督学习中都取得了最先进的结果。

### 背景

现实世界图数据集通常由多个不同的底层分布生成，形成混合群体结构。然而，现代图表示学习方法，如图对比学习和Mixup等增强方法，通常忽略了这种混合结构。

### 目的

提出一种能够明确建模图数据作为底层概率图生成模型混合的框架，并通过模型感知的分区改进图学习任务，包括数据增强和对比学习。

### 方法

利用图矩（motif密度）来聚类来自同一模型的图，解混合成分并识别不同的生成机制。提出图混合感知的Mixup(GMAM)作为数据增强技术，以及模型自适应的对比学习框架MGCL，改进负采样策略。

### 主要发现

建立了图切割距离与图矩密度之间的理论保证，表明具有小切割距离的图具有相似motif密度。实验显示，MGCL在无监督学习中八个数据集上获得平均排名第一，GMAM在监督学习中七个数据集中的六个上达到最先进准确率。

### 结论

所提出的模型感知框架有效解决了图数据中的混合群体问题，通过理论保证和实验验证了其在图学习任务中的优越性，为图表示学习提供了新思路。

### 翻译

现实世界图数据集通常由混合群体组成，其中图是从多个不同的底层分布生成的。然而，现代表示学习方法，如图对比学习和Mixup等增强方法，通常忽略了这种混合结构。在这项工作中，我们提出了一个统一框架，明确地将数据建模为由图表示的底层概率图生成模型的混合。为了表征这些图，我们利用图矩（motif密度）来聚类来自同一模型的图。这使我们能够解混合成分并识别其不同的生成机制。这种模型感知的分区有利于两个关键的图学习任务：1) 它使图混合感知的Mixup(GMAM)成为可能，这是一种数据增强技术，在估计的图引导下在语义有效的空间中进行插值，而不是假设每类只有一个图。2) 对于GCL，它使模型自适应和有原则的增强成为可能。此外，通过引入新的模型感知目标，我们提出的方法（称为MGCL）通过将负样本限制为来自其他模型的图来改进负采样。我们建立了一个关键的理论保证：一个新的、更紧的界限表明，从具有小切割距离的图采样的图将具有相似的motif密度，具有高概率。在基准数据集上的广泛实验展示了强大的实证性能。在无监督学习中，MGCL取得了最先进的结果，在八个数据集上获得了平均排名第一。在监督学习中，GMAM始终优于现有策略，在7个数据集中的6个上取得了新的最先进准确率。


### 论文摘要

Real-world graph datasets often consist of mixtures of populations, where graphs are generated from multiple distinct underlying distributions. However, modern representation learning approaches, such as graph contrastive learning (GCL) and augmentation methods like Mixup, typically overlook this mixture structure. In this work, we propose a unified framework that explicitly models data as a mixture of underlying probabilistic graph generative models represented by graphons. To characterize these graphons, we leverage graph moments (motif densities) to cluster graphs arising from the same model. This enables us to disentangle the mixture components and identify their distinct generative mechanisms. This model-aware partitioning benefits two key graph learning tasks: 1) It enables a graphon-mixture-aware mixup (GMAM), a data augmentation technique that interpolates in a semantically valid space guided by the estimated graphons, instead of assuming a single graphon per class. 2) For GCL, it enables model-adaptive and principled augmentations. Additionally, by introducing a new model-aware objective, our proposed approach (termed MGCL) improves negative sampling by restricting negatives to graphs from other models. We establish a key theoretical guarantee: a novel, tighter bound showing that graphs sampled from graphons with small cut distance will have similar motif densities with high probability. Extensive experiments on benchmark datasets demonstrate strong empirical performance. In unsupervised learning, MGCL achieves state-of-the-art results, obtaining the top average rank across eight datasets. In supervised learning, GMAM consistently outperforms existing strategies, achieving new state-of-the-art accuracy in 6 out of 7 datasets.

---

## 6. Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models

**论文链接:** [http://arxiv.org/abs/2510.08492v1](http://arxiv.org/abs/2510.08492v1)

**作者:** Sharut Gupta, Shobhita Sundaram, Chenyu Wang, Stefanie Jegelka, Phillip Isola

**发布时间:** 2025-10-09

**备注:** 63 pages, 29 tables, and 47 figures

### GPT解析

### 总结

论文提出了一种名为UML（非配对多模态学习器）的新训练范式，利用辅助的非配对多模态数据增强目标模态的表示学习，无需显式配对数据。

### 背景

传统多模态学习器为视觉问答等任务寻找统一表示，但严重依赖于配对数据集。

### 目的

探索能否利用辅助的非配对多模态数据来直接增强目标模态中的表示学习。

### 方法

提出UML（Unpaired Multimodal Learner），一种与模态无关的训练范式，其中单个模型交替处理来自不同模态的输入，同时在它们之间共享参数。这种设计利用了不同模态是共享底层现实投影的假设。

### 主要发现

在线性数据生成假设下，证明非配对的辅助数据可以产生比单模态训练更严格地关于数据生成过程的信息表示；使用来自辅助模态的非配对数据可以持续改善各种单模态目标的下游性能。

### 结论

UML能够有效利用非配对多模态数据增强表示学习，无需显式配对，为多模态学习提供了新思路。

### 翻译

传统多模态学习器为视觉问答等任务寻找统一表示，但严重依赖于配对数据集。然而，一个被忽视但可能很强大的问题是：能否利用辅助的非配对多模态数据来直接增强目标模态中的表示学习？我们提出了UML：非配对多模态学习器，一种与模态无关的训练范式，其中单个模型交替处理来自不同模态的输入，同时在它们之间共享参数。这种设计利用了不同模态是共享底层现实投影的假设，使模型能够从跨模态结构中受益，而不需要显式的配对。理论上，在线性数据生成假设下，我们证明非配对的辅助数据可以产生比单模态训练更严格地关于数据生成过程的信息表示。实验上，我们表明，使用来自辅助模态的非配对数据（如文本、音频或图像）可以持续改善各种单模态目标（如图像和音频）的下游性能。我们的项目页面：https://unpaired-multimodal.github.io/


### 论文摘要

Traditional multimodal learners find unified representations for tasks like visual question answering, but rely heavily on paired datasets. However, an overlooked yet potentially powerful question is: can one leverage auxiliary unpaired multimodal data to directly enhance representation learning in a target modality? We introduce UML: Unpaired Multimodal Learner, a modality-agnostic training paradigm in which a single model alternately processes inputs from different modalities while sharing parameters across them. This design exploits the assumption that different modalities are projections of a shared underlying reality, allowing the model to benefit from cross-modal structure without requiring explicit pairs. Theoretically, under linear data-generating assumptions, we show that unpaired auxiliary data can yield representations strictly more informative about the data-generating process than unimodal training. Empirically, we show that using unpaired data from auxiliary modalities -- such as text, audio, or images -- consistently improves downstream performance across diverse unimodal targets such as image and audio. Our project page: https://unpaired-multimodal.github.io/

---

## 7. Selection, Reflection and Self-Refinement: Revisit Reasoning Tasks via a Causal Lens

**论文链接:** [http://arxiv.org/abs/2510.08222v1](http://arxiv.org/abs/2510.08222v1)

**作者:** Yunlong Deng, Boyang Sun, Yan Li, Lingjing Kong, Zeyu Tang, Kun Zhang, Guangyi Chen

**发布时间:** 2025-10-09

### GPT解析

### 总结

论文从因果角度重新审视推理任务，提出SR²框架，通过将估计的潜在变量作为反馈纳入选择机制，显著提高了模型在推理任务上的性能。

### 背景

推理任务因其内在复杂性，长期以来被视为评估机器学习模型（特别是大型语言模型）能力的严格基准。虽然人类可以轻松解决这些任务，但现有模型即使在大量预训练和后训练后，仍然无法可靠地执行推理。

### 目的

从因果角度重新审视推理任务，试图理解它们在潜在空间中的行为，并为解决这些挑战提供见解。

### 方法

将推理任务描述为选择机制，其中高级逻辑概念作为对给定观察的选择算子；提出SR²框架，包含三个关键模块：反思性表征学习、依赖自我细化和周期性中间对齐，将估计的潜在变量作为反馈纳入选择机制。

### 主要发现

实验表明，该方法在推理准确性方面取得了显著提升，在数独和迷宫任务上，与最新进展相比，参数量减少8倍的情况下，性能提高了10%以上。

### 结论

通过从因果角度重新审视推理任务，并引入SR²框架，能够有效提高模型在推理任务上的性能。

### 翻译

由于其内在复杂性，推理任务长期以来一直被视为评估机器学习模型（特别是大型语言模型）能力的严格基准。虽然人类可以轻松解决这些任务，但现有模型即使在大量预训练和后训练后，仍然无法可靠地执行推理。在本文中，我们从因果角度重新审视推理任务，试图理解它们在潜在空间中的行为，并为解决这些挑战提供见解。具体来说，我们将推理任务描述为一种选择机制，其中高级逻辑概念作为对给定观察的选择算子，例如，在数学问题中识别正确答案或在数独中填写适当条目。我们强调了这个表述的两个关键特性，这些特性揭示了推理任务的难度。首先，即使正确答案完全由观察到的输入决定，潜在空间的复杂性也超过了观察空间。其次，与逻辑思维对应的潜在变量是密集结构的，并表现出强相关性。基于这一表述，我们引入了一个名为SR²的框架，该框架将估计的潜在变量作为反馈纳入选择机制，从而促进潜在表示之间密集依赖关系的学习。该框架包含三个关键模块：反思性表征学习、依赖自我细化和周期性中间对齐。实验表明，我们的方法在推理准确性方面取得了显著提升，例如，在数独和迷宫任务上，与最新进展相比，参数量减少8倍的情况下，性能提高了10%以上。


### 论文摘要

Due to their inherent complexity, reasoning tasks have long been regarded as rigorous benchmarks for assessing the capabilities of machine learning models, especially large language models (LLMs). Although humans can solve these tasks with ease, existing models, even after extensive pre-training and post-training at scale, still fail to perform reasoning reliably. In this paper, we revisit reasoning tasks from a causal perspective, seeking to understand their behavior in latent space and to offer insights for addressing their challenges. Specifically, we cast reasoning tasks as a selection mechanism, in which high-level logical concepts function as selection operators on the given observations, such as, identifying the correct answer in a math problem or filling the appropriate entry in Sudoku. We emphasize two key properties of this formulation that shed light on the difficulty of reasoning tasks. First, the latent space exceeds the observation space in complexity, even when the correct answer is fully determined by the observed input. Second, the latent variables, corresponding to logical thought, are densely structured and exhibit strong dependencies. Building on this formulation, we introduce a framework, called SR$^2$, that incorporates the estimated latent variables as feedback into the selection mechanism, thereby facilitating the learning of dense dependencies among latent representations. The framework consists of three key modules: reflective representation learning, dependency self-refinement, and periodic intermediate alignment. Experimentally, we show that our approach yields significant gains in reasoning accuracy, for example, attaining over 10$\%$ improvement in performance with 8$\times$ fewer parameters on the Sudoku and Maze tasks over the recent advances.

---

## 8. MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation

**论文链接:** [http://arxiv.org/abs/2510.07910v1](http://arxiv.org/abs/2510.07910v1)

**作者:** Chongmyung Kwon, Yujin Kim, Seoeun Park, Yunji Lee, Charmgil Hong

**发布时间:** 2025-10-09

**备注:** Medical Image Computing and Computer-Assisted Intervention (MICCAI)  Predictive Intelligence in Medicine Workshop (MICCAI PRIME) 2025; 13 pages

### GPT解析

### 总结

本研究提出了一种名为MMM的新型框架，通过整合三维量子化学信息到药物表示学习中，结合电子局域化函数(ELF)生成的三维电子密度图和二部图编码器，以捕捉药物分子的全局电子特性和局部子结构相互作用。实验结果表明，该方法在药物-药物相互作用预测上比基线模型有统计学上的显著改进，有潜力提高临床药物推荐的安全性。

### 背景

药物推荐是机器学习支持的临床决策支持系统中的关键任务，但联合用药之间的药物-药物相互作用(DDI)风险仍然是一个重大挑战。之前的研究使用图神经网络(GNN)来表示药物结构，但其简化的离散形式无法完全捕捉分子结合亲和力和反应性。

### 目的

提出一个名为MMM的新型框架，将三维量子化学信息整合到药物表示学习中，以提高药物-药物相互作用的预测准确性。

### 方法

使用电子局域化函数(ELF)生成三维电子密度图，结合ELF衍生的特征(编码全局电子特性)和二部图编码器(建模局部子结构相互作用)，以学习药物分子的互补特性。在MIMIC-III数据集(250种药物，442个子结构)上评估该方法。

### 主要发现

与基于GNN的SafeDrug模型相比，MMM在F1分数、Jaccard指数和DDI率方面显示出统计学上的显著改进。

### 结论

ELF基础的3D表示具有提高预测准确率的潜力，可以支持临床实践中更安全的组合药物处方。

### 翻译

药物推荐是基于机器学习的临床决策支持系统中的关键任务。然而，联合处方药物之间的药物-药物相互作用(DDI)风险仍然是一个重大挑战。先前的研究使用图神经网络(GNN)来表示药物结构。尽管如此，它们简化的离散形式无法完全捕捉分子结合亲和力和反应性。因此，我们提出了多模态DDI预测结合分子电子局域化函数(ELF)图(MMM)，这是一个新型框架，将三维量子化学信息整合到药物表示学习中。它使用ELF生成三维电子密度图。为了捕捉治疗相关性和相互作用风险，MMM结合了ELF衍生的特征，这些特征编码全局电子特性，以及一个建模局部子结构相互作用的二部图编码器。这种设计使学习药物分子的互补特性成为可能。我们在MIMIC-III数据集上评估MMM，并将其与几个基线模型进行比较。特别是，与基于GNN的SafeDrug模型的比较表明，在多个指标方面有统计学上的显著改进。这些结果表明，基于ELF的三维表示有可能提高预测准确性，并支持临床实践中更安全的组合药物处方。


### 论文摘要

Drug recommendation is an essential task in machine learning-based clinical decision support systems. However, the risk of drug-drug interactions (DDI) between co-prescribed medications remains a significant challenge. Previous studies have used graph neural networks (GNNs) to represent drug structures. Regardless, their simplified discrete forms cannot fully capture the molecular binding affinity and reactivity. Therefore, we propose Multimodal DDI Prediction with Molecular Electron Localization Function (ELF) Maps (MMM), a novel framework that integrates three-dimensional (3D) quantum-chemical information into drug representation learning. It generates 3D electron density maps using the ELF. To capture both therapeutic relevance and interaction risks, MMM combines ELF-derived features that encode global electronic properties with a bipartite graph encoder that models local substructure interactions. This design enables learning complementary characteristics of drug molecules. We evaluate MMM in the MIMIC-III dataset (250 drugs, 442 substructures), comparing it with several baseline models. In particular, a comparison with the GNN-based SafeDrug model demonstrates statistically significant improvements in the F1-score (p = 0.0387), Jaccard (p = 0.0112), and the DDI rate (p = 0.0386). These results demonstrate the potential of ELF-based 3D representations to enhance prediction accuracy and support safer combinatorial drug prescribing in clinical practice.

---

## 9. Self-Supervised Learning Strategies for a Platform to Test the Toxicity of New Chemicals and Materials

**论文链接:** [http://arxiv.org/abs/2510.07853v1](http://arxiv.org/abs/2510.07853v1)

**作者:** Thomas Lautenschlager, Nils Friederich, Angelo Jovin Yamachui Sitcheu, Katja Nau, Gaëlle Hayot, Thomas Dickmeis, Ralf Mikut

**发布时间:** 2025-10-09

### GPT解析

### 总结

本研究展示了如何利用自监督学习表征来有效识别毒物诱导的变化，并通过EmbryoNet数据集验证了该方法能够区分不同化合物的作用机制。

### 背景

高通量毒性测试提供了一种快速且经济高效的方法来测试大量化合物，而自动化评估通过机器学习模型实现是这类系统的关键组成部分。

### 目的

解决高通量毒性测试领域的关键挑战，并展示如何通过自监督学习学习的表征来有效识别毒物诱导的变化。

### 方法

利用公开的EmbryoNet数据集进行概念验证，该数据集包含斑马鱼胚胎的十种表型，这些表型由针对早期胚胎发育中不同过程的各种化学化合物引起，并采用自监督学习方法学习表征。

### 主要发现

使用自监督学习学习的表征能够有效区分不同化合物的作用机制。

### 结论

讨论了在TOXBOX项目中机器学习模型与物理毒性测试设备的集成前景。

### 翻译

高通量毒性测试提供了一种快速且经济高效的方法来测试大量化合物。这类系统的关键组成部分是通过机器学习模型进行自动化评估。在本文中，我们解决了该领域的关键挑战，并展示了如何通过自监督学习学习的表征来有效识别毒物诱导的变化。我们提供了一个概念验证，利用公开的EmbryoNet数据集，该数据集包含由针对早期胚胎发育中不同过程的各种化学化合物引起的十种斑马鱼胚胎表型。我们的分析显示，使用自监督学习学习的表征适合有效区分不同化合物的作用机制。最后，我们在TOXBOX项目的背景下讨论了机器学习模型与物理毒性测试设备的集成。


### 论文摘要

High-throughput toxicity testing offers a fast and cost-effective way to test large amounts of compounds. A key component for such systems is the automated evaluation via machine learning models. In this paper, we address critical challenges in this domain and demonstrate how representations learned via self-supervised learning can effectively identify toxicant-induced changes. We provide a proof-of-concept that utilizes the publicly available EmbryoNet dataset, which contains ten zebrafish embryo phenotypes elicited by various chemical compounds targeting different processes in early embryonic development. Our analysis shows that the learned representations using self-supervised learning are suitable for effectively distinguishing between the modes-of-action of different compounds. Finally, we discuss the integration of machine learning models in a physical toxicity testing device in the context of the TOXBOX project.

---

## 10. Causality Guided Representation Learning for Cross-Style Hate Speech Detection

**论文链接:** [http://arxiv.org/abs/2510.07707v1](http://arxiv.org/abs/2510.07707v1)

**作者:** Chengshuai Zhao, Shu Wan, Paras Sheth, Karan Patwa, K. Selçuk Candan, Huan Liu

**发布时间:** 2025-10-09

### GPT解析

### 总结

CADET是一种因果表示学习框架，通过解构仇恨言论为可解释的潜在因素并控制混杂因素，有效分离真实仇恨意图与表面语言线索，实现仇恨言论的稳健检测。

### 背景

网络仇恨言论泛滥威胁网络和谐；显性仇恨易识别，但隐性仇恨（通过讽刺、反讽等表达）更难检测；现有模型依赖表面语言线索，难以泛化；不同平台仇恨言论针对不同群体且风格独特，导致虚假相关性。

### 目的

建立仇恨言论生成的因果图模型，包含上下文环境、创作者动机、目标和风格；提出CADET框架以提升仇恨言论检测的泛化能力。

### 方法

CADET框架将仇恨言论解构为可解释的潜在因素；控制混杂因素分离真实仇恨意图；通过在潜在空间中对风格进行干预实现反事实推理。

### 主要发现

CADET在综合实验中表现出优越性能；因果先验在推进可泛化的仇恨言论检测方面具有潜力。

### 结论

因果表示学习方法可以有效解决仇恨言论检测中的泛化问题，使模型能够稳健识别各种形式的仇恨言论。

### 翻译

网络仇恨言论的泛滥对网络和谐构成严重威胁。虽然显性仇恨言论通过明显的侮辱性语言容易被识别，但隐性仇恨言论通常通过讽刺、反讽、刻板印象或编码语言表达——这使得它更难被检测。现有的仇恨言论检测模型主要依赖表面语言线索，无法在多样化的风格变化中有效泛化。此外，不同平台上传播的仇恨言论往往针对不同的群体并采用独特的风格，这可能导致它们与标签之间产生虚假相关性，进一步挑战了当前的检测方法。受这些观察的启发，我们假设仇恨言论的生成可以建模为一个包含关键因素的因果图：上下文环境、创作者动机、目标和风格。受此图指导，我们提出了CADET，一个因果表示学习框架，该框架将仇恨言论解构为可解释的潜在因素，然后控制混杂因素，从而将真实的仇恨意图与表面语言线索分离。此外，CADET允许通过在潜在空间中对风格进行干预来实现反事实推理，自然地引导模型以稳健地识别各种形式的仇恨言论。CADET在综合实验中表现出优越性能，突显了因果先验在推进可泛化的仇恨言论检测方面的潜力。


### 论文摘要

The proliferation of online hate speech poses a significant threat to the harmony of the web. While explicit hate is easily recognized through overt slurs, implicit hate speech is often conveyed through sarcasm, irony, stereotypes, or coded language -- making it harder to detect. Existing hate speech detection models, which predominantly rely on surface-level linguistic cues, fail to generalize effectively across diverse stylistic variations. Moreover, hate speech spread on different platforms often targets distinct groups and adopts unique styles, potentially inducing spurious correlations between them and labels, further challenging current detection approaches. Motivated by these observations, we hypothesize that the generation of hate speech can be modeled as a causal graph involving key factors: contextual environment, creator motivation, target, and style. Guided by this graph, we propose CADET, a causal representation learning framework that disentangles hate speech into interpretable latent factors and then controls confounders, thereby isolating genuine hate intent from superficial linguistic cues. Furthermore, CADET allows counterfactual reasoning by intervening on style within the latent space, naturally guiding the model to robustly identify hate speech in varying forms. CADET demonstrates superior performance in comprehensive experiments, highlighting the potential of causal priors in advancing generalizable hate speech detection.

---

## 11. Bridged Clustering for Representation Learning: Semi-Supervised Sparse Bridging

**论文链接:** [http://arxiv.org/abs/2510.07182v2](http://arxiv.org/abs/2510.07182v2)

**作者:** Patrick Peixuan Ye, Chen Shani, Ellen Vitercik

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种名为Bridged Clustering的半监督学习框架，用于从无配对的输入X和输出Y数据集中学习预测器。

### 背景

传统半监督学习和密集传输方法在处理无配对数据时存在局限性，前者没有充分利用仅输出数据，后者缺乏稀疏性和可解释性。

### 目的

开发一种能够利用无配对数据、保持稀疏性和可解释性的半监督学习方法。

### 方法

首先独立地对输入X和输出Y进行聚类，然后使用少量配对样本学习聚类之间的稀疏桥接。在推理时，将新输入分配到最近的输入聚类，然后返回链接的输出聚类的质心作为预测。

### 主要发现

理论分析表明，在有界聚类错误和桥接错误率的情况下，该算法成为有效且高效的预测器。实验证明，该方法与最先进的方法具有竞争力。

### 结论

Bridged Clustering是一种简单、模型无关且在低监督设置中具有高标签效率的方法，能够有效处理无配对数据学习问题。

### 翻译

我们引入了Bridged Clustering，一种半监督框架，用于从任意无配对的输入X和输出Y数据集中学习预测器。我们的方法首先独立地对X和Y进行聚类，然后仅使用少量配对样本学习聚类之间的稀疏、可解释的桥接。在推理时，新的输入x被分配到其最近的输入聚类，链接的输出聚类的质心被返回作为预测ŷ。与传统SSL不同，Bridged Clustering明确利用仅输出数据，与密集传输方法不同，它保持稀疏和可解释的对齐。通过理论分析，我们表明在有界聚类错误和桥接错误率的情况下，我们的算法成为有效且高效的预测器。经验上，我们的方法与最先进的方法具有竞争力，同时保持简单、模型无关，并且在低监督设置中具有高标签效率。


### 论文摘要

We introduce Bridged Clustering, a semi-supervised framework to learn predictors from any unpaired input $X$ and output $Y$ dataset. Our method first clusters $X$ and $Y$ independently, then learns a sparse, interpretable bridge between clusters using only a few paired examples. At inference, a new input $x$ is assigned to its nearest input cluster, and the centroid of the linked output cluster is returned as the prediction $\hat{y}$. Unlike traditional SSL, Bridged Clustering explicitly leverages output-only data, and unlike dense transport-based methods, it maintains a sparse and interpretable alignment. Through theoretical analysis, we show that with bounded mis-clustering and mis-bridging rates, our algorithm becomes an effective and efficient predictor. Empirically, our method is competitive with SOTA methods while remaining simple, model-agnostic, and highly label-efficient in low-supervision settings.

---

## 12. Learning from Limited Multi-Phase CT: Dual-Branch Prototype-Guided Framework for Early Recurrence Prediction in HCC

**论文链接:** [http://arxiv.org/abs/2510.07347v1](http://arxiv.org/abs/2510.07347v1)

**作者:** Hsin-Pei Yu, Si-Qin Lyu, Yi-Hsien Hsieh, Weichung Wang, Tung-Hung Su, Jia-Horng Kao, Che Lin

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究提出了一种名为DuoProto的双分支原型引导框架，用于从单相CT图像中增强肝细胞癌早期复发预测，通过在训练过程中利用有限的多相数据。

### 背景

在肝细胞癌的根治性切除术后，早期复发预测仍是临床管理的重大挑战。临床指南推荐全多期对比增强CT，但并非所有机构都能提供完整期相覆盖。实践中，单相门静脉扫描常被单独使用，特别是在成像资源有限、采集协议不同或患者存在对比剂不耐受或运动伪影的情况下。这种差异导致理想化模型假设与实际部署约束不匹配，需要能有效利用有限多相数据的方法。

### 目的

开发一种能够从单相CT图像中有效预测肝细胞癌早期复发的方法，同时通过训练过程中利用多相数据提高预测准确性，解决实际临床环境中数据不完整的问题。

### 方法

提出DuoProto框架，采用双分支架构：主分支处理单相图像，辅助分支利用可用多期扫描通过跨域原型对齐引导表示学习。结构化原型表示作为类别锚点提高特征辨别能力，结合基于排序的监督机制纳入临床相关的复发风险因素。

### 主要发现

大量实验证明，DuoProto优于现有方法，特别是在类别不平衡和缺失期相条件下。消融研究进一步验证了双分支、原型引导设计的有效性。

### 结论

该框架符合当前临床应用需求，为肝细胞癌复发风险预测提供了通用解决方案，支持更明智的决策制定。

### 翻译

在肝细胞癌的根治性切除术后，早期复发预测仍然是临床管理中的一个关键挑战。尽管临床指南推荐使用全多期采集的对比增强CT，并且在许多三级中心常规执行，但完整的期相覆盖并非在所有机构中都能一致可用。在实践中，单相门静脉扫描常常单独使用，特别是在成像资源有限、采集协议不同或患者相关因素(如对比剂不耐受或运动伪影)的情况下。这种差异性导致理想化模型假设与实际部署约束之间的不匹配，突显了需要能够有效利用有限多相数据的方法。为了应对这一挑战，我们提出了一个双分支原型引导框架，通过在训练过程中利用有限的多相数据，从单相CT中增强早期复发预测。该框架采用双分支架构：主分支处理单相图像，而辅助分支利用可用的多期扫描通过跨域原型对齐来引导表示学习。结构化的原型表示作为类别锚点来提高特征辨别能力，而基于排序的监督机制结合了临床相关的复发风险因素。大量实验证明，该框架优于现有方法，特别是在类别不平衡和缺失期相条件下。消融研究进一步验证了双分支、原型引导设计的有效性。我们的框架符合当前临床应用需求，并为肝细胞癌复发风险预测提供了通用解决方案，支持更明智的决策制定。


### 论文摘要

Early recurrence (ER) prediction after curative-intent resection remains a critical challenge in the clinical management of hepatocellular carcinoma (HCC). Although contrast-enhanced computed tomography (CT) with full multi-phase acquisition is recommended in clinical guidelines and routinely performed in many tertiary centers, complete phase coverage is not consistently available across all institutions. In practice, single-phase portal venous (PV) scans are often used alone, particularly in settings with limited imaging resources, variations in acquisition protocols, or patient-related factors such as contrast intolerance or motion artifacts. This variability results in a mismatch between idealized model assumptions and the practical constraints of real-world deployment, underscoring the need for methods that can effectively leverage limited multi-phase data. To address this challenge, we propose a Dual-Branch Prototype-guided (DuoProto) framework that enhances ER prediction from single-phase CT by leveraging limited multi-phase data during training. DuoProto employs a dual-branch architecture: the main branch processes single-phase images, while the auxiliary branch utilizes available multi-phase scans to guide representation learning via cross-domain prototype alignment. Structured prototype representations serve as class anchors to improve feature discrimination, and a ranking-based supervision mechanism incorporates clinically relevant recurrence risk factors. Extensive experiments demonstrate that DuoProto outperforms existing methods, particularly under class imbalance and missing-phase conditions. Ablation studies further validate the effectiveness of the dual-branch, prototype-guided design. Our framework aligns with current clinical application needs and provides a general solution for recurrence risk prediction in HCC, supporting more informed decision-making.

---

## 13. New Machine Learning Approaches for Intrusion Detection in ADS-B

**论文链接:** [http://arxiv.org/abs/2510.08333v1](http://arxiv.org/abs/2510.08333v1)

**作者:** Mikaëla Ngamboé, Jean-Simon Marrocco, Jean-Yves Ouattara, José M. Fernandez, Gabriela Nicolescu

**发布时间:** 2025-10-09

**备注:** This is the author's version of the work accepted for publication  Digital Avionics Systems Conference (DASC) 2025. The final version will be  available via IEEE Xplore

### GPT解析

### 总结

该研究探讨了利用机器学习模型提升ADS-B协议入侵检测系统性能的方法，比较了transformer编码器和xLSTM网络两种实现，结果表明xLSTM模型在检测性能上表现更优。

### 背景

自动相关监视-广播(ADS-B)协议在空中交通管理(ATM)中的使用日益增加，但其存在安全漏洞，确保其安全性至关重要。

### 目的

研究新兴的机器学习模型和训练策略，提高基于AI的入侵检测系统(IDS)对ADS-B的保护能力，专注于地面ATM系统。

### 方法

评估两种深度学习IDS实现：transformer编码器和xLSTM网络(首个基于xLSTM的ADS-B IDS)；采用迁移学习策略，在良性ADS-B消息上预训练，然后使用包含篡改消息的标记数据进行微调。

### 主要发现

该方法在识别微妙攻击方面优于现有方法；xLSTM-based IDS达到98.9%的F1分数，超过transformer-based模型的94.3%；对未见攻击的测试验证了xLSTM模型的泛化能力；xLSTM引入7.26秒延迟，在SSR刷新间隔内，但对时间关键操作有限制；transformer实现2.1秒延迟但检测性能较低。

### 结论

基于xLSTM的IDS在检测性能方面表现优异，延迟时间在可接受范围内，但对于某些时间关键场景可能需要进一步优化。

### 翻译

随着空中交通管理(ATM)对易受攻击的自动相关监视-广播(ADS-B)协议的依赖日益增长，确保安全至关重要。本研究调查了新兴的机器学习模型和训练策略，以改进用于ADS-B的基于AI的入侵检测系统(IDS)。专注于地面ATM系统，我们评估了两种深度学习IDS实现：一种使用transformer编码器，另一种使用扩展长短期记忆(xLSTM)网络，这是首个基于xLSTM的ADS-B IDS。采用迁移学习策略，包括在良性ADS-B消息上进行预训练，并使用包含篡改消息实例的标记数据进行微调。结果表明，这种方法优于现有方法，特别是在识别逐渐破坏态势感知的微妙攻击方面。基于xLSTM的IDS达到98.9%的F1分数，超过了基于transformer的模型的94.3%。对未见攻击的测试验证了xLSTM模型的泛化能力。推理延迟分析显示，xLSTM-based IDS引入的7.26秒延迟在辅助监视雷达(SSR)刷新间隔(5-12秒)内，但对于时间关键操作可能有限制。虽然基于transformer的IDS实现2.1秒延迟，但这是以较低的检测性能为代价的。


### 论文摘要

With the growing reliance on the vulnerable Automatic Dependent Surveillance-Broadcast (ADS-B) protocol in air traffic management (ATM), ensuring security is critical. This study investigates emerging machine learning models and training strategies to improve AI-based intrusion detection systems (IDS) for ADS-B. Focusing on ground-based ATM systems, we evaluate two deep learning IDS implementations: one using a transformer encoder and the other an extended Long Short-Term Memory (xLSTM) network, marking the first xLSTM-based IDS for ADS-B. A transfer learning strategy was employed, involving pre-training on benign ADS-B messages and fine-tuning with labeled data containing instances of tampered messages. Results show this approach outperforms existing methods, particularly in identifying subtle attacks that progressively undermine situational awareness. The xLSTM-based IDS achieves an F1-score of 98.9%, surpassing the transformer-based model at 94.3%. Tests on unseen attacks validated the generalization ability of the xLSTM model. Inference latency analysis shows that the 7.26-second delay introduced by the xLSTM-based IDS fits within the Secondary Surveillance Radar (SSR) refresh interval (5-12 s), although it may be restrictive for time-critical operations. While the transformer-based IDS achieves a 2.1-second latency, it does so at the cost of lower detection performance.

---

## 14. Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices

**论文链接:** [http://arxiv.org/abs/2510.07545v1](http://arxiv.org/abs/2510.07545v1)

**作者:** Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Ridwan Mahbub, Mizanur Rahman, Amran Bhuiyan, Israt Jahan, Mir Tafseer Nayeem, Shafiq Joty, Enamul Hoque, Jimmy Huang

**发布时间:** 2025-10-08

**备注:** Accepted to the EMNLP 2025 Industry Track

### GPT解析

### 总结

研究提出两种方法提高小型视觉-语言模型作为图表理解任务自动评估工具的性能，实现成本效益评估。

### 背景

70亿参数的大型视觉-语言模型在图表理解任务中作为自动评估工具有良好表现，但小型模型(≤20亿参数)表现不佳，限制了在资源受限环境中的应用。

### 目的

提出两种方法确保成本效益评估，解决小型模型表现不佳的问题。

### 方法

1) 多标准提示：将单独的评估标准组合到单个查询中；2) 领域自适应迁移学习：在图表数据集上的合成判断上微调一个20亿参数的LVLM，创建ChartJudge模型。

### 主要发现

多标准提示暴露了鲁棒性差距，导致70亿模型性能大幅下降，包括专业评估工具如LLaVA-Critic；小型LVLM (ChartJudge)能有效将知识从一个数据集转移到另一个数据集，使其更专业；对图表类型和查询复杂性的细粒度分析提供了模型大小、提示设计和可移植性之间权衡的可操作见解。

### 结论

该工作实现了图表推理任务的可扩展、低成本评估。

### 翻译

仅具有70亿参数的大型视觉-语言模型在图表理解任务中作为自动评估工具已显示出良好前景。然而，小型模型(≤20亿参数)作为评估工具时仍表现不佳，限制了它们在资源受限环境中的实际应用。为解决这一问题，我们提出了两种方法确保成本效益评估：(i)多标准提示，将单独的评估标准组合到单个查询中；(ii)领域自适应迁移学习，我们在图表数据集上的合成判断上微调一个20亿参数的LVLM，创建了ChartJudge。实验表明，多标准提示暴露了鲁棒性差距，导致70亿模型性能大幅下降，包括专业的LVLM评估工具如LLaVA-Critic。此外，我们发现我们的小型LVLM (ChartJudge)可以有效地将知识从一个数据集转移到另一个数据集，使其成为更专业的模型。我们对图表类型和查询复杂性的细粒度分析提供了模型大小、提示设计和可移植性之间权衡的可操作见解，实现了图表推理任务的可扩展、低成本评估。我们的代码和数据将公开提供。


### 论文摘要

Large Vision-Language Models (LVLMs) with only 7B parameters have shown promise as automated judges in chart comprehension tasks. However, tiny models (<=2B parameters) still perform poorly as judges, limiting their real-world use in resource-constrained settings. To address this, we propose two approaches to ensure cost-efficient evaluation: (i) multi-criteria prompting, which combines separate evaluation criteria into a single query, and (ii) domain-adaptive transfer learning, in which we fine-tune a 2B-parameter LVLM on synthetic judgments in a chart dataset to create the ChartJudge. Experiments show that multi-criteria prompting exposes robustness gaps, which led to a huge drop in performance for 7B models, including specialized LVLM judges like LLaVA-Critic. In addition, we find that our tiny LVLM (ChartJudge) can effectively transfer knowledge from one dataset to another to make it a more specialized model. Our fine-grained analysis across chart types and query complexities offers actionable insights into trade-offs between model size, prompt design, and transferability, enabling scalable, low-cost evaluation for chart reasoning tasks. Our code and the data will be made publicly available.

---

## 15. RayFusion: Ray Fusion Enhanced Collaborative Visual Perception

**论文链接:** [http://arxiv.org/abs/2510.08017v1](http://arxiv.org/abs/2510.08017v1)

**作者:** Shaohong Wang, Bin Lu, Xinyu Xiao, Hanzhi Zhong, Bowen Pang, Tong Wang, Zhiyu Xiang, Hangguan Shan, Eryun Liu

**发布时间:** 2025-10-09

**备注:** Accepted by NeurIPS2025

### GPT解析

### 总结

提出了一种名为RayFusion的基于射线的融合方法，用于解决协作视觉感知中深度信息缺失的问题，提高自动驾驶系统中基于摄像头的感知性能。

### 背景

协作视觉感知方法在自动驾驶领域受到广泛关注，能够解决传感器限制问题。然而，基于摄像头的感知系统（如3D物体检测）因缺乏明确的深度信息，难以生成准确预测。

### 目的

减轻深度估计中的模糊性，提高纯摄像头协作感知系统的检测性能。

### 方法

提出RayFusion，一种基于射线的融合方法，利用协作者的射线占用信息，减少沿相机射线的冗余和假阳性预测。

### 主要发现

通过全面实验证明，该方法持续优于现有的最先进模型，显著提升了协作视觉感知的性能。

### 结论

RayFusion有效解决了协作视觉感知中深度信息缺失的问题，为自动驾驶领域的感知系统提供了更准确的解决方案。

### 翻译

近年来，由于能够解决传感器限制问题，协作视觉感知方法在自动驾驶社区中获得了广泛关注。然而，缺乏明确的深度信息常常使得基于摄像头的感知系统（例如3D物体检测）难以生成准确的预测。为了减轻深度估计中的模糊性，我们提出了RayFusion，一种用于协作视觉感知的基于射线的融合方法。利用来自协作者的射线占用信息，RayFusion减少了沿相机射线的冗余和假阳性预测，增强了纯摄像头协作感知系统的检测性能。全面的实验表明，我们的方法持续优于现有的最先进模型，显著推进了协作视觉感知的性能。代码可在https://github.com/wangsh0111/RayFusion获取。


### 论文摘要

Collaborative visual perception methods have gained widespread attention in the autonomous driving community in recent years due to their ability to address sensor limitation problems. However, the absence of explicit depth information often makes it difficult for camera-based perception systems, e.g., 3D object detection, to generate accurate predictions. To alleviate the ambiguity in depth estimation, we propose RayFusion, a ray-based fusion method for collaborative visual perception. Using ray occupancy information from collaborators, RayFusion reduces redundancy and false positive predictions along camera rays, enhancing the detection performance of purely camera-based collaborative perception systems. Comprehensive experiments show that our method consistently outperforms existing state-of-the-art models, substantially advancing the performance of collaborative visual perception. The code is available at https://github.com/wangsh0111/RayFusion.

---

## 16. SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.08531v1](http://arxiv.org/abs/2510.08531v1)

**作者:** Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang

**发布时间:** 2025-10-09

**备注:** Project Page: https://zju-real.github.io/SpatialLadder/ Code:  https://github.com/ZJU-REAL/SpatialLadder

### GPT解析

### 总结

本文提出了一种名为SpatialLadder的新方法，通过分阶段训练和新的数据集构建视觉语言模型的空间推理能力，取得了最先进的性能。

### 背景

空间推理对视觉语言模型(VLMs)来说仍然是一个基本挑战，尽管最近有所进展，但现有方法难以实现稳健的性能。现有方法的局限性在于它们试图直接学习空间推理，而没有建立感知和理解的层次基础。

### 目的

解决VLMs在空间推理方面的挑战，通过建立空间智能的渐进式方法论，提升模型在空间推理任务上的性能。

### 方法

1. 引入了SpatialLadder-26k多模态数据集，包含26,610个样本，涵盖目标定位、单图像、多视图和视频空间推理任务；2. 设计了一个三阶段渐进训练框架：(1)通过目标定位建立空间感知，(2)通过多维空间任务发展空间理解，(3)通过具有可验证奖励的强化学习加强复杂推理；3. 基于此方法开发了SpatialLadder模型，一个拥有30亿参数的模型。

### 主要发现

SpatialLadder模型在空间推理基准测试上取得了最先进的性能，比基础模型平均提高了23.4%，比GPT-4o高出20.8%，比Gemini-2.0-Flash高出10.1%。在领域外基准测试上保持了强大的泛化能力，提高了7.2%。

### 结论

从感知到推理的渐进式训练对于构建稳健的空间智能至关重要。

### 翻译

空间推理对视觉语言模型(VLMs)来说仍然是一个基本挑战，尽管最近有所进展，但现有方法难以实现稳健的性能。我们确定这一局限性源于一个关键差距：现有方法试图直接学习空间推理，而没有建立感知和理解的层次基础。为了应对这一挑战，我们提出了一个构建空间智能的全面渐进式方法。我们引入了SpatialLadder-26k，这是一个包含26,610个样本的多模态数据集，涵盖目标定位、单图像、多视图和视频空间推理任务，通过标准化的管道构建，确保跨模态的系统覆盖。基于此数据集，我们设计了一个三阶段渐进训练框架，(1)通过目标定位建立空间感知，(2)通过多维空间任务发展空间理解，(3)通过具有可验证奖励的强化学习加强复杂推理。这种方法产生了SpatialLadder，一个拥有30亿参数的模型，在空间推理基准测试上取得了最先进的性能，比基础模型平均提高23.4%，超越GPT-4o 20.8%，超越Gemini-2.0-Flash 10.1%。值得注意的是，SpatialLadder在领域外基准测试上保持了强大的泛化能力，提高了7.2%，这表明从感知到推理的渐进式训练对于稳健的空间智能是必不可少的。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型(VLMs)中的空间推理能力不足问题。当前VLMs在基础视觉任务上表现良好，但在空间推理方面存在显著瓶颈，这限制了它们在机器人导航、自动驾驶和虚拟现实等需要空间智能的应用中的部署。空间推理是人类理解视觉场景的基本能力，但对VLMs来说仍然是一个重大挑战，这个问题在现实世界中非常重要，因为它直接影响AI系统对物理环境的理解和交互能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过控制实验验证了假设：空间推理失败的原因是感知基础不足，而非推理能力本身。他们发现当提供渐进的感知提示(位置提示和方向提示)时，模型性能显著提升。基于这一发现，作者设计了层次化空间智能构建方法，借鉴了现有工作中的3D场景重建技术(ScanNet)、问题模板设计(VSI-Bench)和强化学习方法(GRPO)，但创新性地将这些技术整合到一个三阶段渐进式训练框架中，从感知基础开始，逐步发展到复杂推理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是空间智能需要通过渐进式训练系统构建，从基础感知能力开始，逐步发展到复杂推理能力，而非一次性学习完整的空间推理能力。整体流程包括：1)构建SpatialLadder-26k多模态数据集，包含26,610个样本覆盖物体定位、单图像、多视图和视频空间推理任务；2)设计三阶段训练框架：阶段1通过物体定位建立空间感知，阶段2通过多维任务发展空间理解，阶段3通过强化学习强化复杂推理；3)使用可验证奖励函数和思维链技术确保推理质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)SpatialLadder-26k数据集，首个系统覆盖从基础感知到复杂推理的全谱空间推理多模态数据集；2)三阶段渐进式训练框架，首次提出系统构建空间推理能力的层次化方法；3)显著提升的性能，在多个基准测试上超越现有方法。相比之前工作，不同之处在于：现有数据集碎片化且范围狭窄，而SpatialLadder-26k系统性地覆盖多种模态；现有方法直接优化推理输出，而SpatialLadder建立从感知到推理的层次结构；现有方法将空间推理视为单一能力，而SpatialLadder认识到需要渐进式构建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出一个基于渐进式训练的三阶段框架和构建SpatialLadder-26k多模态数据集，显著提升了视觉语言模型的空间推理能力，实现了从感知到理解再到推理的系统化空间智能构建。'}


### 论文摘要

Spatial reasoning remains a fundamental challenge for Vision-Language Models (VLMs), with current approaches struggling to achieve robust performance despite recent advances. We identify that this limitation stems from a critical gap: existing methods attempt to learn spatial reasoning directly without establishing the hierarchical foundations of perception and understanding. To address this challenge, we present a comprehensive methodology for building spatial intelligence progressively. We introduce SpatialLadder-26k, a multimodal dataset containing 26,610 samples spanning object localization, single image, multi-view, and video spatial reasoning tasks, constructed through a standardized pipeline that ensures systematic coverage across modalities. Building on this dataset, we design a three-stage progressive training framework that (1) establishes spatial perception through object localization, (2) develops spatial understanding through multi-dimensional spatial tasks, and (3) strengthens complex reasoning via reinforcement learning with verifiable rewards. This approach yields SpatialLadder, a 3B-parameter model that achieves state-of-the-art performance on spatial reasoning benchmarks, with 23.4% average improvement over the base model, surpassing GPT-4o by 20.8% and Gemini-2.0-Flash by 10.1%. Notably, SpatialLadder maintains strong generalization with 7.2% improvement on out-of-domain benchmarks, demonstrating that progressive training from perception to reasoning is essential for robust spatial intelligence.

---

## 17. The impact of abstract and object tags on image privacy classification

**论文链接:** [http://arxiv.org/abs/2510.07976v1](http://arxiv.org/abs/2510.07976v1)

**作者:** Darya Baranouskaya, Andrea Cavallaro

**发布时间:** 2025-10-09

**备注:** This work has been submitted to the ICASSP 2026

### GPT解析

### 总结

本研究探讨了对象标签和抽象标签在图像隐私分类任务中的适用性，发现当标签预算有限时抽象标签更有效，而在标签数量充足时对象标签同样有用。

### 背景

对象标签表示具体实体，是许多计算机视觉任务的核心；抽象标签捕获更高级别的信息，对需要上下文理解的任务相关；两者都有助于图像解释。

### 目的

探索哪种类型的标签更适合图像隐私这一上下文相关且本质上主观的任务。

### 方法

比较对象标签和抽象标签在图像隐私分类中的效果，考虑不同标签预算情况。

### 主要发现

对象标签通常用于隐私分类；当标签预算有限时，抽象标签更有效；当每个图像有更多标签可用时，对象信息同样有用。

### 结论

这些发现将指导未来研究开发更准确的图像隐私分类器，考虑标签类型和数量的作用。

### 翻译

对象标签表示具体实体，是许多计算机视觉任务的核心，而抽象标签捕获更高级别的信息，与需要上下文理解的任务相关。从图像中提取的对象和抽象标签也有助于可解释性。在本文中，我们探索哪种类型的标签更适合图像隐私这一上下文相关且本质上主观的任务。虽然对象标签通常用于隐私分类，但我们表明当标签预算有限时，抽象标签更有效。相反，当每个图像有更多标签可用时，对象信息同样有用。我们相信这些发现将指导未来研究开发更准确的图像隐私分类器，考虑标签类型和数量的作用。


### 论文摘要

Object tags denote concrete entities and are central to many computer vision tasks, whereas abstract tags capture higher-level information, which is relevant for tasks that require a contextual, potentially subjective scene understanding. Object and abstract tags extracted from images also facilitate interpretability. In this paper, we explore which type of tags is more suitable for the context-dependent and inherently subjective task of image privacy. While object tags are generally used for privacy classification, we show that abstract tags are more effective when the tag budget is limited. Conversely, when a larger number of tags per image is available, object-related information is as useful. We believe that these findings will guide future research in developing more accurate image privacy classifiers, informed by the role of tag types and quantity.

---

## 18. CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.07944v1](http://arxiv.org/abs/2510.07944v1)

**作者:** Tianrui Zhang, Yichen Liu, Zilin Guo, Yuxin Guo, Jingcheng Ni, Chenjing Ding, Dan Xu, Lewei Lu, Zehuan Wu

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文提出了CVD-STORM，一种利用空间-时间重建变分自编码器的跨视角视频扩散模型，能够在各种控制输入下生成具有4D重建能力的长期多视角视频，通过微调VAE和集成到视频扩散过程，显著提高了生成质量。

### 背景

生成模型已被广泛应用于环境模拟和未来状态预测的世界建模。随着自动驾驶的发展，对高保真视频生成以及深度估计等多样化有意义信息的需求不断增长。

### 目的

提出一个名为CVD-STORM的跨视角视频扩散模型，生成具有4D重建能力的长期多视角视频，并接受各种控制输入，以满足自动驾驶领域对高质量视频和多样化信息的需求。

### 方法

使用空间-时间重建变分自编码器，首先通过辅助4D重建任务对VAE进行微调，增强其编码3D结构和时间动态的能力，然后将此VAE集成到视频扩散过程中，并采用联合训练的高斯溅射解码器来重建动态场景。

### 主要发现

实验结果表明，该模型在FID和FVD指标上均取得了显著改进，联合训练的高斯溅射解码器能有效重建动态场景，为全面场景理解提供有价值的几何信息。

### 结论

CVD-STORM模型能够满足自动驾驶领域对高保真视频生成和多样化信息的需求，通过4D重建能力和高质量视频生成，为场景理解和预测提供了有效工具。

### 翻译

生成模型已被广泛应用于环境模拟和未来状态预测的世界建模。随着自动驾驶的发展，不仅对高保真视频生成有需求，而且对深度估计等多样化有意义信息的需求也在增长。为此，我们提出了CVD-STORM，这是一种利用空间-时间重建变分自编码器的跨视角视频扩散模型，能够在各种控制输入下生成具有4D重建能力的长期多视角视频。我们的方法首先通过辅助4D重建任务对VAE进行微调，增强其编码3D结构和时间动态的能力。随后将此VAE集成到视频扩散过程中，显著提高生成质量。实验结果表明，我们的模型在FID和FVD指标上均取得了显著改进。此外，联合训练的高斯溅射解码器能有效重建动态场景，为全面场景理解提供有价值的几何信息。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶领域中如何同时生成高质量长序列多视角视频并实现动态4D场景重建的问题。这个问题很重要，因为自动驾驶系统需要可靠的环境模拟和未来状态预测，而现有方法往往缺乏明确的3D信息，无法准确表示真实世界的几何结构，限制了它们作为世界模型的适用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视频生成方法在自动驾驶领域的局限性，然后提出将生成模型与重建任务相结合的思路。他们借鉴了STORM的4D重建方法，将其整合到VAE框架中创建了STORM-VAE；基于UniMLVG的多视角视频生成架构进行改进；使用Stable Diffusion 3.5作为基础模型，并应用DiT架构。整体采用两阶段训练策略：先学习场景重建，再训练条件世界模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合视频生成和4D场景重建任务，创建统一框架，使模型同时学习生成高质量视频和理解3D场景结构。整体流程分为三部分：1) STORM-VAE训练阶段：使用预训练VAE并添加高斯散射解码器，通过多视角图像和LiDAR数据训练；2) CVD-STORM训练阶段：使用STORM-VAE作为潜在编码器，构建基于DiT的视频扩散模型，采用三种transformer块处理不同维度数据；3) 推理阶段：生成六视角视频并重建动态3D场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) STORM-VAE：扩展传统VAE架构，集成高斯散射解码器，能编码空间时间信息；2) CVD-STORM框架：统一多视角视频生成和4D场景重建，采用简化训练策略；3) 表示学习增强：通过辅助重建任务提升生成质量。相比之前工作，它实现了端到端联合训练，提供绝对深度估计而非相对深度，同时优化了生成质量和重建精度，并支持更丰富的条件输入。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CVD-STORM通过结合跨视角视频生成和空间-时间重建模型，实现了自动驾驶场景中高质量多视角视频生成与精确4D场景重建的统一框架，显著提升了世界模型的表示能力和几何理解。'}


### 论文摘要

Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding.

---

## 19. USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots

**论文链接:** [http://arxiv.org/abs/2510.07869v1](http://arxiv.org/abs/2510.07869v1)

**作者:** Junwen Gu, Zhiheng wu, Pengxuan Si, Shuang Qiu, Yukai Feng, Luoyang Sun, Laien Luo, Lianyi Yu, Jian Wang, Zhengxing Wu

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文介绍了USIM（基于模拟的多任务视觉-语言-行动数据集）和U0（适用于水下机器人的VLA模型），解决了水下机器人操作面临的挑战，在多种任务中表现出色，成功率达到80%，在移动操作任务中将目标距离减少了21.2%

### 背景

水下环境为机器人操作带来独特挑战，包括复杂水动力学、有限能见度和受限通信。尽管数据驱动方法已推动陆地机器人具身智能发展，并实现特定任务水下机器人自主操作，但开发能自主执行多任务的水下智能仍面临挑战，因大规模高质量水下数据集稀缺

### 目的

解决水下机器人面临的挑战，开发能自主执行多任务的水下智能，构建大规模高质量水下数据集，并基于此数据集开发适用于水下机器人的VLA模型

### 方法

引入USIM数据集，包含来自1,852个轨迹的561K帧，总计15.6小时BlueROV2交互数据，涵盖9种场景下的20个任务；提出U0模型，通过多模态融合整合双目视觉和其他传感器模态，采用基于卷积-注意力的感知增强模块(CAP)提高空间理解和移动操作能力

### 主要发现

在检查、避障、扫描和动态跟踪等任务中，框架实现80%成功率；在具有挑战性的移动操作任务中，与基线方法相比，目标距离减少21.2%

### 结论

USIM和U0表明VLA模型可有效应用于水下机器人应用，为可扩展数据集构建、改进任务自主性和通用水下机器人的实际实现奠定基础

### 翻译

水下环境为机器人操作带来独特挑战，包括复杂水动力学、有限能见度和受限通信。尽管数据驱动方法已推动陆地机器人具身智能发展，并实现特定任务水下机器人自主操作，但开发能自主执行多任务的水下智能仍面临挑战，因大规模高质量水下数据集稀缺。为解决这些限制，我们引入USIM，一个基于模拟的多任务视觉-语言-行动数据集，专为水下机器人设计。USIM包含来自1,852个轨迹的561K帧，总计15.6小时BlueROV2交互数据，涵盖9种不同场景下的20个任务，从视觉导航到移动操作。基于此数据集，我们提出U0，一个适用于通用水下机器人的VLA模型，通过多模态融合整合双目视觉和其他传感器模态，并采用基于卷积-注意力的感知增强模块(CAP)提高空间理解和移动操作能力。在检查、避障、扫描和动态跟踪等任务中，框架实现80%成功率；在具有挑战性的移动操作任务中，与基线方法相比，目标距离减少21.2%，证明其有效性。USIM和U0表明VLA模型可有效应用于水下机器人应用，为可扩展数据集构建、改进任务自主性和通用水下机器人的实际实现奠定基础

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决水下机器人缺乏大规模高质量数据集和通用智能模型的问题。这个问题很重要，因为海洋覆盖地球71%的面积，水下环境对人类操作极具挑战性，而现有水下机器人大多只能执行特定任务且依赖人工远程操作，缺乏自主执行多种任务的能力。这限制了人类对海洋的探索和开发，也使水下任务（如海洋生态调查、资源开发、管道检查等）效率低下且成本高昂。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到数据驱动方法在陆地机器人领域的成功，注意到水下机器人面临的独特挑战（如流体动力学、能见度限制、通信受限）。他们选择在仿真环境中构建多样化的水下场景来高效安全地收集大规模数据，借鉴了Stonefish仿真器和室内机器人VLA数据集的构建方法，但针对水下环境进行了调整。模型设计上基于Isaac-GR00T N1.5架构，但增加了针对水下环境的多模态传感器融合和卷积-注意力感知焦点增强模块(CAP)，以提高水下目标检测和定位能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建大规模多样化的水下仿真数据集，开发针对水下环境的视觉-语言-动作(VLA)模型，整合多模态感知和语言理解能力，并通过卷积-注意力机制增强模型在水下环境中的感知能力。整体流程包括：1)使用Stonefish仿真器构建9种不同水下场景和BlueROV2机器人模型；2)在仿真环境中收集20个任务的机器人-环境交互数据，形成USIM数据集；3)基于Isaac-GR00T N1.5构建U0模型，整合多模态传感器数据和CAP模块；4)使用USIM数据集对模型进行微调；5)通过开环离线评估和闭环在线测试验证模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个大规模水下多任务VLA数据集USIM，包含561K帧和20个任务；2)首个针对水下机器人的通用VLA模型U0，集成多模态融合和CAP模块；3)以机器人为中心的坐标表示方法，增强动态场景理解；4)可扩展的数据到任务框架，实现80%的任务成功率和21.2%的移动操作性能提升。相比之前工作，USIM提供了统一的跨任务数据框架，而不仅是特定任务数据；U0是首个结合语言理解、视觉感知和动作执行的通用水下模型，专门针对水下环境特点进行了优化，而非直接应用陆地机器人模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过构建首个大规模水下多任务视觉-语言-动作数据集USIM和开发首个通用水下机器人VLA模型U0，为水下机器人的自主能力提升和智能化发展提供了新的基础和方法。'}


### 论文摘要

Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots.

---

## 20. TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics

**论文链接:** [http://arxiv.org/abs/2510.07181v2](http://arxiv.org/abs/2510.07181v2)

**作者:** Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang

**发布时间:** 2025-10-08

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

TIGeR框架将视觉语言模型从感知估计器转变为几何计算机，通过外部工具实现厘米级精度的几何计算，在机器人任务中表现优异。

### 背景

视觉语言模型在空间推理方面表现出色，但仅限于定性精度，缺乏机器人技术所需的计算精度。当前方法未能利用深度传感器和相机校准的度量线索，无法提供机器人操作所需的厘米级精度。

### 目的

提出TIGeR框架，使视觉语言模型能够通过外部工具生成和执行精确的几何计算，满足机器人应用的精度需求。

### 方法

TIGeR使模型识别几何推理需求，合成计算代码，并调用专门库进行精确计算。引入TIGeR-300K数据集，涵盖点变换、姿态估计和空间兼容性验证。通过监督微调和具有分层奖励设计的强化微调两阶段训练管道进行训练。

### 主要发现

TIGeR在几何推理基准测试上实现了最先进性能，并在真实世界的机器人操作任务中展示了厘米级精度。

### 结论

TIGeR成功地将视觉语言模型转变为能够执行精确几何计算的模型，为机器人应用提供了必要的精度。

### 翻译

视觉语言模型在空间推理方面表现出色，但它们本质上仅限于定性精度，缺乏机器人技术所需的计算精度。当前方法未能利用深度传感器和相机校准的度量线索，而是将几何问题简化为无法提供机器人操作所需的厘米级精度的模式识别任务。我们提出TIGeR（工具集成几何推理）框架，将VLMs从感知估计器转变为几何计算机，使它们能够通过外部工具生成和执行精确的几何计算。TIGeR不是尝试在神经网络内部化复杂的几何操作，而是使模型能够识别几何推理需求，合成适当的计算代码，并调用专门库进行精确计算。为此，我们引入了TIGeR-300K数据集，涵盖点变换、姿态估计和空间兼容性验证，包含工具调用序列和中间计算。通过结合监督微调（SFT）和具有分层奖励设计的强化微调（RFT）的两阶段训练管道，TIGeR在几何推理基准测试上实现了最先进性能，并在真实世界的机器人操作任务中展示了厘米级精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文解决视觉语言模型(VLMs)在几何推理上的局限性，即它们只能提供定性的空间关系评估(如'在左边')，而无法进行精确的定量计算。这个问题很重要，因为机器人在物理世界中操作需要厘米级精度的几何推理能力，如计算3D姿态、旋转矩阵和无碰撞轨迹，缺乏这种能力会阻碍VLMs在现实世界机器人应用中的实用性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有VLMs在几何推理上的感知和输出局限性，然后提出让模型识别几何推理需求、生成计算代码并调用外部工具执行精确计算，而非将几何运算内化到神经网络中。该方法借鉴了工具集成推理(TIR)的三个类别(基于提示、SFT和RL的方法)以及空间理解和推理的数据驱动与工具方法，但创新性地提出了两阶段SFT-RFT训练流程和新的过程奖励函数，并整合了现有的视觉基础模型作为工具。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将VLMs从感知估计器转变为几何计算机，通过生成和执行精确的几何计算实现厘米级精度。整体流程包括：1)工具分类(视觉感知工具和几何计算工具)；2)数据生成(TIGeR-300K数据集，结合模板合成和大模型重写)；3)两阶段训练(SFT初始化工具使用能力，RFT使用GRPO算法增强能力)；4)层次化奖励设计(格式、工具调用、参数内容、代码生成和答案奖励)；5)推理过程(逐步推理、选择上下文、调用工具、生成代码、推导结果)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)概念与方法创新，强调几何推理的核心作用并引入工具集成框架；2)数据集创新，构建TIGeR-300K数据集提供完整工具调用序列；3)训练方法创新，开发两阶段SFT-RFT管道和层次化奖励设计；4)技术实现创新，分类工具并引入代码生成子程序。相比之前工作，TIGeR不将几何计算内化到神经网络中，而是利用外部工具实现精确计算；结合了结果导向和过程导向的奖励函数，提供细粒度监督；超越了纯数据驱动方法的定性限制和现有工具集成方法的准确性不足。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TIGeR通过工具集成几何推理框架，使视觉语言模型能够生成和执行精确代码，实现了厘米级精度的机器人几何推理，超越了传统方法的定性空间理解限制。'}


### 论文摘要

Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.

---

## 21. Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Model

**论文链接:** [http://arxiv.org/abs/2510.07345v1](http://arxiv.org/abs/2510.07345v1)

**作者:** Danush Kumar Venkatesh, Adam Schmidt, Muhammad Abdullah Jamal, Omid Mohareri

**发布时间:** 2025-10-07

**备注:** 29 pages, 16 figures

### GPT解析

### 总结

本文提出了一种名为SurgiFlowVid的稀疏且可控的视频扩散框架，用于生成代表性不足类别的手术视频，以解决手术视频数据集不平衡的问题。

### 背景

手术视频数据集对于场景理解、程序建模和术中支持至关重要，但这些数据集通常严重不平衡，稀有动作和工具代表性不足，限制了下游模型的鲁棒性。

### 目的

解决手术视频数据集中的数据不平衡问题，提高模型对稀有类别和工具的识别能力。

### 方法

SurgiFlowVid框架包含双预测扩散模块，联合去噪RGB帧和光流，提供时间归纳偏置；以及稀疏视觉编码器，使用轻量级信号调节生成过程，实现可控性而不需要密集标注。

### 主要发现

在三个手术数据集上的实验表明，合成数据比竞争基线方法提高10-20%，验证了该方法的有效性。

### 结论

SurgiFlowVid是一种有前景的策略，可以缓解数据不平衡问题并推进手术视频理解方法的发展。

### 翻译

手术视频数据集对于场景理解、程序建模和术中支持至关重要。然而，这些数据集通常严重不平衡，稀有动作和工具代表性不足，这限制了下游模型的鲁棒性。我们通过SurgiFlowVid解决了这一挑战，这是一种用于生成代表性不足类别手术视频的稀疏且可控的视频扩散框架。我们的方法引入了双预测扩散模块，联合去噪RGB帧和光流，提供时间归纳偏置，从有限样本改进运动建模。此外，稀疏视觉编码器使用轻量级信号（如稀疏分割掩码或RGB帧）调节生成过程，实现可控性而不需要密集标注。我们在三个手术数据集上验证了我们的方法，任务包括动作识别、工具存在检测和腹腔镜运动预测。我们方法生成的合成数据比竞争基线方法提高10-20%，确立了SurgiFlowVid作为一种有前景的策略，可以缓解数据不平衡并推进手术视频理解方法。


### 论文摘要

Surgical video datasets are essential for scene understanding, enabling procedural modeling and intra-operative support. However, these datasets are often heavily imbalanced, with rare actions and tools under-represented, which limits the robustness of downstream models. We address this challenge with $SurgiFlowVid$, a sparse and controllable video diffusion framework for generating surgical videos of under-represented classes. Our approach introduces a dual-prediction diffusion module that jointly denoises RGB frames and optical flow, providing temporal inductive biases to improve motion modeling from limited samples. In addition, a sparse visual encoder conditions the generation process on lightweight signals (e.g., sparse segmentation masks or RGB frames), enabling controllability without dense annotations. We validate our approach on three surgical datasets across tasks including action recognition, tool presence detection, and laparoscope motion prediction. Synthetic data generated by our method yields consistent gains of 10-20% over competitive baselines, establishing $SurgiFlowVid$ as a promising strategy to mitigate data imbalance and advance surgical video understanding methods.

---

## 22. Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games

**论文链接:** [http://arxiv.org/abs/2510.07813v1](http://arxiv.org/abs/2510.07813v1)

**作者:** Valerio La Gatta, Dolev Mutzari, Sarit Kraus, VS Subrahmanian

**发布时间:** 2025-10-09

**备注:** 15 pages, 13 figures

### GPT解析

### 总结

本研究提出了一种在对抗环境中智能体如何平衡信息获取与风险暴露的决策框架

### 背景

对抗环境中的智能体需要在获取信息以增强态势感知和暴露于威胁之间做出战略权衡

### 目的

研究这种信息获取与风险暴露之间的紧张关系，并开发一个有效的决策框架

### 方法

提出PEEC（追逃-暴露-隐藏博弈）框架和SHADOW（战争部分观察下的战略通信混合动作决策）多头序贯强化学习框架，整合连续导航控制、离散通信动作和对手建模

### 主要发现

SHADOW追击者比六个竞争基线方法实现更高成功率；时间序列建模和对手建模对有效决策至关重要；学习到的策略在不同通信风险和智能体物理不对称性上具有良好的泛化能力

### 结论

SHADOW框架能有效平衡对抗环境中的观察性和风险

### 翻译

对抗环境需要智能体应对一个关键战略权衡：获取信息可以增强态势感知，但同时也可能使它们暴露于威胁之中。为了研究这种紧张关系，我们提出了一个追逃-暴露-隐藏博弈（PEEC），其中追击智能体必须决定何时通信以获取逃逸者的位置。每次通信都会暴露追击者的位置，增加被瞄准的风险。两个智能体都通过强化学习学习其移动策略，而追击者 additionally 学习一种通信策略以平衡观察性和风险。我们提出了SHADOW（战争部分观察下的战略通信混合动作决策），这是一个多头序贯强化学习框架，集成了连续导航控制、离散通信动作和对手建模以进行行为预测。经验评估显示，SHADOW追击者比六个竞争基线方法实现了更高的成功率。我们的消融研究证实时间序列建模和对手建模对有效决策至关重要。最后，我们的敏感性分析揭示学习到的策略在不同通信风险和智能体之间的物理不对称性上具有良好的泛化能力。


### 论文摘要

Adversarial environments require agents to navigate a key strategic trade-off: acquiring information enhances situational awareness, but may simultaneously expose them to threats. To investigate this tension, we formulate a PursuitEvasion-Exposure-Concealment Game (PEEC) in which a pursuer agent must decide when to communicate in order to obtain the evader's position. Each communication reveals the pursuer's location, increasing the risk of being targeted. Both agents learn their movement policies via reinforcement learning, while the pursuer additionally learns a communication policy that balances observability and risk. We propose SHADOW (Strategic-communication Hybrid Action Decision-making under partial Observation for Warfare), a multi-headed sequential reinforcement learning framework that integrates continuous navigation control, discrete communication actions, and opponent modeling for behavior prediction. Empirical evaluations show that SHADOW pursuers achieve higher success rates than six competitive baselines. Our ablation study confirms that temporal sequence modeling and opponent modeling are critical for effective decision-making. Finally, our sensitivity analysis reveals that the learned policies generalize well across varying communication risks and physical asymmetries between agents.

---

## 23. Reconstructing the local density field with combined convolutional and point cloud architecture

**论文链接:** [http://arxiv.org/abs/2510.08573v1](http://arxiv.org/abs/2510.08573v1)

**作者:** Baptiste Barthe-Gold, Nhat-Minh Nguyen, Leander Thiele

**发布时间:** 2025-10-09

**备注:** 6 pages, 4 figures, 1 table. Accepted at the NeurIPS 2025 Workshop:  ML4PS. Comments welcome!

### GPT解析

### 总结

该研究构建了一种结合卷积U-Net和点云DeepSets的混合神经网络，用于根据暗物质晕的特殊速度重建局部暗物质密度场。这种混合方法比单独使用U-Net能更好地利用小尺度信息，提高了重建质量，特别是在小尺度上能更准确地恢复聚类幅度和相位。

### 背景

暗物质密度场的重建是宇宙学研究中的重要问题。暗物质晕的特殊速度作为暗物质场的有偏差示踪物，提供了关于局部暗物质密度分布的信息。

### 目的

开发一种神经网络方法，能够根据暗物质晕的特殊速度有效重建局部暗物质密度场，特别是在小尺度上提高重建精度。

### 方法

构建了一种混合神经网络架构，结合了卷积U-Net和点云DeepSets。U-Net擅长处理网格化数据，而DeepSets则能有效处理点云数据，这种组合能够充分利用小尺度信息。

### 主要发现

混合神经网络比单独使用U-Net的方法能更好地重建暗物质密度场，特别是在小尺度上能更准确地恢复聚类幅度和相位。

### 结论

结合卷积U-Net和点云DeepSets的混合神经网络架构是重建局部暗物质密度场的有效方法，特别是在处理小尺度信息方面具有优势。

### 翻译

我们构建了一个神经网络，用于根据暗物质晕的视线特殊速度（暗物质场的有偏差示踪物）对局部暗物质密度场进行回归分析。我们的架构结合了卷积U-Net和点云DeepSets。这种组合能够有效利用小尺度信息，并相对于仅使用U-Net的方法提高了重建质量。具体来说，我们的混合网络在小尺度上比U-Net更好地恢复了聚类幅度和相位。


### 论文摘要

We construct a neural network to perform regression on the local dark-matter density field given line-of-sight peculiar velocities of dark-matter halos, biased tracers of the dark matter field. Our architecture combines a convolutional U-Net with a point-cloud DeepSets. This combination enables efficient use of small-scale information and improves reconstruction quality relative to a U-Net-only approach. Specifically, our hybrid network recovers both clustering amplitudes and phases better than the U-Net on small scales.

---

## 24. Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression

**论文链接:** [http://arxiv.org/abs/2510.08512v1](http://arxiv.org/abs/2510.08512v1)

**作者:** Nikolaos Stathoulopoulos, Christoforos Kanellakis, George Nikolakopoulos

**发布时间:** 2025-10-09

**备注:** Accepted for publication in IEEE Robotics and Automation Letters  (RA-L). 8 pages, 6 figures

### GPT解析

### 总结

提出了一种基于语义场景图的深度压缩框架，用于高效传输3D点云数据，在保持结构和语义保真度的同时，实现了高达98%的数据压缩率。

### 背景

3D点云数据传输对于集中式和分散式多机器人系统的高级感知至关重要，特别是在如今越来越依赖边缘和云处理的背景下。然而，点云数据量大且复杂，在带宽受限和间歇性连接条件下造成挑战，常常降低系统性能。

### 目的

开发一种高效压缩3D点云数据的方法，解决在带宽受限和间歇性连接条件下的传输挑战，同时保持数据的质量和可用性。

### 方法

将点云分解为语义连贯的块，使用基于特征线性调制(FiLM)的语义感知编码器将它们编码为紧凑的潜在表示，并通过由潜在特征和图节点属性引导的基于折叠的解码器实现结构准确的重建。

### 主要发现

在SemanticKITTI和nuScenes数据集上的实验表明，该框架实现了最先进的压缩率，可将数据大小减少高达98%，同时保持结构和语义保真度；该框架还支持多机器人位姿图优化和地图合并等下游应用，实现的轨迹精度和地图对齐可与原始LiDAR扫描获得的结果相媲美。

### 结论

基于语义场景图的深度压缩框架有效解决了3D点云数据在带宽受限环境下的传输问题，同时保持了数据的质量和可用性，为多机器人系统的高级感知提供了重要支持。

### 翻译

3D点云数据的高效传输对于集中式和分散式多机器人系统中的高级感知至关重要，特别是在如今越来越依赖边缘和云处理的背景下。然而，点云数据量大且复杂的特性在带宽受限和间歇性连接条件下带来了挑战，常常降低系统性能。我们提出了一种基于语义场景图的深度压缩框架。该方法将点云分解为语义连贯的块，并使用由特征线性调制(FiLM)调节的语义感知编码器将它们编码为紧凑的潜在表示。基于折叠的解码器，由潜在特征和图节点属性引导，实现了结构准确的重建。在SemanticKITTI和nuScenes数据集上的实验表明，该框架实现了最先进的压缩率，在保持结构和语义保真度的同时，将数据大小减少了高达98%。此外，它支持多机器人位姿图优化和地图合并等下游应用，实现了与原始LiDAR扫描获得的轨迹精度和地图对齐相媲美的效果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云数据的高效传输问题。在多机器人系统中，点云数据对高级感知至关重要，但其庞大和复杂的特性在带宽受限和间歇性连接条件下会造成挑战，导致系统性能下降。随着机器人系统对边缘和云处理的依赖增长，解决这一问题对于实现高效的多机器人协作和实时感知具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点云压缩方法的局限性，注意到语义场景图已被证明是导航和规划的有效表示，但现有压缩方法很少利用其结构信息来指导原始几何的有损编码。作者借鉴了FiLM（特征线性调制）技术进行语义条件化，以及基于折叠的解码器结构，同时结合了场景图表示方法和点云压缩技术，创造了一个新的框架，将语义场景图作为压缩的核心组件。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用语义场景图指导点云压缩，将点云分解为语义一致的块，并用语义感知编码器将这些块编码为紧凑的潜在表示，最后通过基于折叠的解码器在潜在特征和图节点属性指导下进行结构准确的重建。整体流程包括：1) 将原始点云转换为语义场景图；2) 将场景划分为特定层次的语义块；3) 使用基于transformer的自动编码器将每个块编码为紧凑的潜在向量；4) 解码这些潜在向量重建完整点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出基于图的点云深度压缩自动编码器，将点云分解为语义一致块并通过FiLM条件化的transformer编码；2) 将关系结构整合到压缩中，利用语义场景图指导整个压缩过程；3) 实现高达98%的数据减少，同时保持几何和语义保真度。相比之前工作，本文方法同时考虑了几何和语义保真度，将场景图作为压缩核心组件，并支持下游机器人任务如多机器人位姿图优化和地图合并。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于语义场景图的深度点云压缩方法，通过将点云分解为语义一致的块并使用语义感知编码器进行压缩，实现了高达98%的数据减少，同时保持了几何和语义保真度，并有效支持下游机器人任务。'}


### 论文摘要

Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans.

---

## 25. Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge

**论文链接:** [http://arxiv.org/abs/2510.08316v1](http://arxiv.org/abs/2510.08316v1)

**作者:** Yu Huang, Zelin Peng, Changsong Wen, Xiaokang Yang, Wei Shen

**发布时间:** 2025-10-09

**备注:** Work in process

### GPT解析

### 总结

该论文提出了一种语义基础学习范式，通过跨模态亲和力传输(CMAT)预训练策略和跨模态功能分割Transformer(CAST)，解决了3D功能分割中的语义边界不清晰问题，并在标准基准测试上取得了最先进的结果。

### 背景

功能分割旨在将3D对象解析为功能不同的部分，用于机器人操作、具身AI和AR应用。现有方法通常利用视觉或文本提示来指导这一过程，但它们往往将点云编码器作为通用特征提取器，忽视了3D数据固有的挑战，如稀疏性、噪声和几何模糊性。

### 目的

解决3D功能分割中语义边界不清晰的问题，通过将大规模2D视觉基础模型(VFMs)的丰富语义知识转移到3D领域，提高3D功能分割的准确性和语义一致性。

### 方法

提出了语义基础学习范式，具体包括：引入跨模态亲和力传输(CMAT)预训练策略，将3D编码器与提升的2D语义对齐，并联合优化重建、亲和力和多样性以产生语义组织的表示；设计跨模态功能分割Transformer(CAST)，将多模态提示与CMAT预训练特征集成，生成精确的、提示感知的分割图。

### 主要发现

在标准基准测试上进行的广泛实验表明，该框架为3D功能分割建立了新的最先进结果。

### 结论

通过将2D视觉基础模型的语义知识转移到3D领域，并设计专门的预训练和分割模型，可以有效解决3D功能分割中的语义边界不清晰问题，提高分割的准确性和语义一致性。

### 翻译

功能分割旨在将3D对象解析为功能不同的部分，为机器人操作、具身AI和AR应用中的识别与交互搭建桥梁。尽管最近的研究利用视觉或文本提示来指导这一过程，但它们通常依赖点云编码器作为通用特征提取器，忽视了3D数据固有的挑战，如稀疏性、噪声和几何模糊性。因此，孤立学习的3D特征通常缺乏清晰且语义一致的功能边界。为了解决这一瓶颈，我们提出了一种语义基础学习范式，将大规模2D视觉基础模型(VFMs)的丰富语义知识转移到3D领域。具体来说，我们引入了跨模态亲和力传输(CMAT)预训练策略，将3D编码器与提升的2D语义对齐，并联合优化重建、亲和力和多样性，以产生语义组织的表示。基于此主干，我们进一步设计了跨模态功能分割Transformer(CAST)，它将多模态提示与CMAT预训练特征集成，生成精确的、提示感知的分割图。在标准基准测试上的广泛实验表明，我们的框架为3D功能分割建立了新的最先进结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决3D功能分割中的语义模糊性问题，即如何将3D物体解析为功能不同的部分（如椅子的座位和腿）。这个问题在现实中至关重要，因为它使机器能够从被动感知转向主动交互，为机器人操作、具身AI和AR应用提供基础，使智能系统能够理解物体的'如何使用'而不仅仅是'是什么'。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D数据的内在挑战（稀疏性、噪声、几何模糊性），并发现现有方法将3D点云编码器作为通用特征提取器未能解决这些问题。他们借鉴了从2D视觉基础模型向3D领域转移语义知识的有前景范式，利用多视图特征提升技术。基于此，他们设计了三阶段框架：首先从2D模型提取语义指导，然后通过CMAT预训练3D编码器学习结构化特征，最后使用CAST架构融合多模态提示进行任务适应。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将2D视觉模型的丰富语义知识转移到3D领域，解决3D数据的语义模糊性。整体流程分为三阶段：1)基础语义基础：从预训练2D模型提取多视图特征并投影回3D点云；2)结构化表示学习：使用CMAT预训练策略，通过几何重建、语义对齐和特征多样性三个目标优化3D编码器；3)提示驱动任务适应：使用CAST架构融合预训练3D特征和多模态提示，生成功能分割图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一学习范式整合跨模态知识转移；2)CMAT预训练策略将2D语义知识蒸馏到3D编码器；3)CAST架构有效融合预训练特征和多模态提示。相比之前工作，本文直接解决3D数据的内在挑战，更明确地建模部分间关系而非仅特征对齐，实现了更精细的部分级区分，并支持更灵活的多模态提示融合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过将2D视觉模型的语义知识转移到3D领域，提出了一种新的语义基础学习范式，显著提升了3D功能分割的性能，使机器能够更准确地理解和分割物体的功能部分。'}


### 论文摘要

Affordance segmentation aims to parse 3D objects into functionally distinct parts, bridging recognition and interaction for applications in robotic manipulation, embodied AI, and AR. While recent studies leverage visual or textual prompts to guide this process, they often rely on point cloud encoders as generic feature extractors, overlooking the intrinsic challenges of 3D data such as sparsity, noise, and geometric ambiguity. As a result, 3D features learned in isolation frequently lack clear and semantically consistent functional boundaries. To address this bottleneck, we propose a semantic-grounded learning paradigm that transfers rich semantic knowledge from large-scale 2D Vision Foundation Models (VFMs) into the 3D domain. Specifically, We introduce Cross-Modal Affinity Transfer (CMAT), a pre-training strategy that aligns a 3D encoder with lifted 2D semantics and jointly optimizes reconstruction, affinity, and diversity to yield semantically organized representations. Building on this backbone, we further design the Cross-modal Affordance Segmentation Transformer (CAST), which integrates multi-modal prompts with CMAT-pretrained features to generate precise, prompt-aware segmentation maps. Extensive experiments on standard benchmarks demonstrate that our framework establishes new state-of-the-art results for 3D affordance segmentation.

---

## 26. Towards Precise Channel Knowledge Map: Exploiting Environmental Information from 2D Visuals to 3D Point Clouds

**论文链接:** [http://arxiv.org/abs/2510.08140v1](http://arxiv.org/abs/2510.08140v1)

**作者:** Yancheng Wang, Chuan Huang, Songyang Zhang, Guanying Chen, Wei Guo, Shenglun Lan, Lexi Xu, Xinzhou Cheng, Xiongyan Tang, Shuguang Cui

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文提出了一种基于三维环境信息的信道知识图(CKM)构建方法，解决传统导频信道探测资源消耗过大的问题，为未来6G网络提供可扩展的信道感知解决方案。

### 背景

传统基于导频的信道探测消耗大量通信资源，给具有大规模信道维度、超宽带宽和密集用户部署的未来6G网络带来了严重的可扩展性挑战。

### 目的

利用三维环境信息构建高精度信道知识图(CKM)，解决传统方法资源消耗过大的问题，为未来6G网络提供可扩展的信道感知解决方案。

### 方法

提出一种新框架，通过混合模型和数据驱动方法将三维点云整合到CKM构建中，利用三维环境信息而非传统的二维视觉表示来构建高精度CKM。

### 主要发现

基于具有语义理解的三维环境可以构建精确的CKM，这些CKM在下一代无线通信应用中展现出巨大潜力。

### 结论

三维环境信息对于构建高精度信道知识图至关重要，所提出的框架为未来6G网络的信道感知提供了可扩展解决方案。

### 翻译

传统导频信道探测所消耗的大量通信资源带来了不可持续的开销，为具有大规模信道维度、超宽带宽和密集用户部署的未来6G网络带来了严重的可扩展性挑战。作为无线电地图的泛化，信道知识图(CKM)提供了一种范式转变，使无需全面测量即可访问位置标记的信道信息。为了充分利用CKM的潜力，本文强调了利用三维环境信息的必要性，超越传统的二维视觉表示，以构建高精度CKM。具体而言，我们提出了一种新框架，通过混合模型和数据驱动方法将三维点云整合到CKM构建中，并在真实场景案例研究中进行了广泛研究。实验结果表明，基于具有语义理解的三维环境构建精确CKM的潜力，以及它们在下一代无线通信中的应用。我们还发布了一个与高分辨率三维环境数据配对的实际测量信道数据集，以支持未来研究和验证。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何构建高精度的信道知识图(CKM)问题。传统方法依赖2D视觉信息，无法充分捕捉无线传播环境的复杂特性。这个问题在未来6G网络中至关重要，因为6G具有大规模信道维度、超宽带宽和密集用户部署的特点，传统基于导频的信道探测会消耗大量资源，带来不可持续的开销。CKM作为无线电图的泛化，可以在不进行大量测量的情况下提供位置标记的信道信息，对网络优化、波束成形等应用具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有CKM构建方法的局限性：传统插值方法需要大量样本且难以捕捉信号突变；射线追踪计算开销大；基于AI的方法大多操作在2D环境。作者认识到无线信道本质上由周围3D环境决定，因此提出利用3D点云信息。他们设计了一个混合模型和数据驱动的框架，借鉴了3D重建技术(如SfM、MVS)、语义理解技术(如开放集语义分割)、点云处理方法(如PointNet++)和无线信道建模原理，但将这些技术专门针对CKM构建问题进行了定制和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用高分辨率3D点云环境和语义理解构建更精确的CKM，采用混合模型和数据驱动方法。整体流程包括：1)3D环境重建：结合无人机图像和LiDAR扫描数据，使用SfM和MVS技术重建点云；2)环境语义理解：使用计算机视觉模型为3D点分配语义标签；3)CKM构建：包含点选择器(利用椭球几何原理过滤无关点)和神经信道增益估计器(学习环境特征到信道参数的映射)。这种方法结合了物理模型的解释能力和数据驱动方法的泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)利用3D点云而非2D视觉信息，保留完整几何结构；2)混合模型和数据驱动的框架，结合物理原理和机器学习；3)创新的点选择器设计，基于共焦椭球壳原理过滤点云；4)环境语义理解，考虑材料特性对信号传播的影响。相比之前工作，不同之处在于：传统插值方法不需要大量样本但无法捕捉突变；射线追踪精度高但计算复杂；基于AI的方法使用3D信息而非2D投影；本文方法结合了物理模型和数据驱动优势，能更好处理复杂3D环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的混合框架，通过利用高分辨率3D点云环境和语义理解，显著提高了信道知识图的构建精度和鲁棒性，为未来6G网络中的环境感知通信提供了新解决方案。'}


### 论文摘要

The substantial communication resources consumed by conventional pilot-based channel sounding impose an unsustainable overhead, presenting a critical scalability challenge for the future 6G networks characterized by massive channel dimensions, ultra-wide bandwidth, and dense user deployments. As a generalization of radio map, channel knowledge map (CKM) offers a paradigm shift, enabling access to location-tagged channel information without exhaustive measurements. To fully utilize the power of CKM, this work highlights the necessity of leveraging three-dimensional (3D) environmental information, beyond conventional two-dimensional (2D) visual representations, to construct high-precision CKMs. Specifically, we present a novel framework that integrates 3D point clouds into CKM construction through a hybrid model- and data-driven approach, with extensive case studies in real-world scenarios. The experimental results demonstrate the potential for constructing precise CKMs based on 3D environments enhanced with semantic understanding, together with their applications in the next-generation wireless communications. We also release a real-world dataset of measured channel paired with high-resolution 3D environmental data to support future research and validation.

---

## 27. PIT-QMM: A Large Multimodal Model For No-Reference Point Cloud Quality Assessment

**论文链接:** [http://arxiv.org/abs/2510.07636v1](http://arxiv.org/abs/2510.07636v1)

**作者:** Shashank Gupta, Gregoire Phillips, Alan C. Bovik

**发布时间:** 2025-10-09

**备注:** Oral presentation at ICIP 2025

### GPT解析

### 总结

该研究提出了一种名为PIT-QMM的新型大型多模态模型，用于无参考点云质量评估，能够端到端处理文本、图像和点云数据，显著优于现有方法，并支持失真定位功能。

### 背景

大型多模态模型在图像和视频质量评估领域已取得显著进展，但这些进步尚未在3D资产领域得到充分探索。

### 目的

利用大型多模态模型进行无参考点云质量评估，自动评估点云的感知质量而无需参考点云。

### 方法

构建PIT-QMM模型，该模型能够处理文本描述、2D投影和3D点云视图这三种互补信息模态，端到端预测点云质量分数。

### 主要发现

在流行基准测试上，所提出的方法以显著优势优于最先进方法，且训练迭代次数更少；该框架还能实现失真定位和识别，为模型可解释性和交互性开辟新途径。

### 结论

PIT-QMM模型在点云质量评估方面表现出色，不仅提高了评估准确性，还提供了失真定位功能，代码和数据集已公开可用。

### 翻译

大型多模态模型最近在图像和视频质量评估领域取得了显著进展，但这一进步尚未在3D资产领域得到充分探索。我们有兴趣使用这些模型进行无参考点云质量评估，其目标是在没有参考点云的情况下自动评估点云的感知质量。我们从观察开始，不同模态的数据——文本描述、2D投影和3D点云视图——提供了关于点云质量的互补信息。然后我们构建了PIT-QMM，这是一种用于无参考点云质量评估的新型大型多模态模型，能够端到端地消费文本、图像和点云来预测质量分数。大量实验表明，在流行的基准测试上，我们提出的方法以显著优势优于最先进方法，且训练迭代次数更少。我们还展示了我们的框架能够实现失真定位和识别，这为模型可解释性和交互性开辟了新的前进道路。代码和数据集可在https://www.github.com/shngt/pit-qmm获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决无参考点云质量评估问题，即自动评估点云感知质量而不需要参考原始高质量点云。这个问题在现实中很重要，因为点云是自动驾驶、沉浸式游戏和数字孪生等应用的基础，容易受传感器不准确、压缩和传输错误影响，而传统图像质量评估指标无法捕捉3D数据的复杂性，现有学习方法效果也不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到不同模态数据（文本、2D投影和3D点云）提供关于点云质量的互补信息，而现有模型要么擅长2D质量评估，要么擅长3D理解，但没有一个完全捕捉PCQA所需的两方面。他们借鉴了预训练基础模型（如Point-BERT、ViT-L/14）、Q-Align的离散化策略、两阶段训练策略和LoRA技术，但针对点云质量评估的特殊需求进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个端到端的多模态模型PIT-QMM，同时处理文本、图像和点云来预测质量分数，利用不同模态的互补优势：点云补丁捕获局部变化，图像投影提供全局视角，文本输入增加心理测量上下文。流程包括：1)数据准备（点云采样、图像投影、文本提示）；2)模型架构（图像编码器、点云编码器、点云投影仪和LLM主干）；3)两阶段训练（特征对齐和指令微调）；4)推理（离散化训练，加权平均映射到连续分数）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个端到端的点-图像-文本多模态模型用于PCQA；任务感知提示设计；高效的编码器感知点云采样策略；两阶段训练策略；失真识别和定位能力。相比之前工作，PIT-QMM在更少训练迭代中取得更好性能；保留局部变化信息；无需昂贵的预处理；是首个能进行失真定位的点云质量评估模型，增强了可解释性和交互性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PIT-QMM首次实现了端到端的点-图像-文本多模态模型，通过融合不同模态的互补信息，显著提升了无参考点云质量评估的性能，并开创了点云失真定位的新方向。'}


### 论文摘要

Large Multimodal Models (LMMs) have recently enabled considerable advances in the realm of image and video quality assessment, but this progress has yet to be fully explored in the domain of 3D assets. We are interested in using these models to conduct No-Reference Point Cloud Quality Assessment (NR-PCQA), where the aim is to automatically evaluate the perceptual quality of a point cloud in absence of a reference. We begin with the observation that different modalities of data - text descriptions, 2D projections, and 3D point cloud views - provide complementary information about point cloud quality. We then construct PIT-QMM, a novel LMM for NR-PCQA that is capable of consuming text, images and point clouds end-to-end to predict quality scores. Extensive experimentation shows that our proposed method outperforms the state-of-the-art by significant margins on popular benchmarks with fewer training iterations. We also demonstrate that our framework enables distortion localization and identification, which paves a new way forward for model explainability and interactivity. Code and datasets are available at https://www.github.com/shngt/pit-qmm.

---

## 28. Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.07594v1](http://arxiv.org/abs/2510.07594v1)

**作者:** Shitij Govil, Jack P. Rodgers, Yuan-Tang Chou, Siqi Miao, Amit Saha, Advaith Anand, Kilian Lieret, Gage DeZoort, Mia Liu, Javier Duarte, Pan Li, Shih-Chieh Hsu

**发布时间:** 2025-10-08

**备注:** Accepted to NeurIPS 2025 Machine Learning and the Physical Sciences  Workshop

### GPT解析

### 总结

本文提出了一种改进的 HEPTv2 方法，通过引入轻量级解码器消除了聚类阶段，直接预测轨迹分配，显著提高了带电粒子轨迹重建的速度。

### 背景

带电粒子轨迹重建是 collider 实验的基础任务和主要计算瓶颈。图神经网络(GNNs)虽然表现良好，但存在计算成本高、计算不规则和随机内存访问模式等问题限制了其吞吐量。

### 目的

提供对 HEPT 和基于 GNN 的管道在相同数据集和指标下的物理跟踪性能的统一、公平评估，并开发一种更高效的轨迹重建方法。

### 方法

提出 HEPTv2，通过扩展 HEPT 添加轻量级解码器，消除聚类阶段，直接预测轨迹分配，保留了 HEPT 的规则、硬件友好的计算特性。

### 主要发现

在 TrackML 数据集上，优化后的 HEPTv2 在 A100 上每事件处理时间约为 28 毫秒，同时保持了具有竞争力的跟踪效率。

### 结论

HEPTv2 是一种实用、可扩展的替代方案，可以替代基于 GNN 的管道进行快速跟踪，在保持竞争性跟踪效率的同时显著提高了处理速度。

### 翻译

带电粒子轨迹重建是 collider 实验的基础任务，也是粒子重建中的主要计算瓶颈。图神经网络(GNNs)在这个问题上表现出色，但昂贵的图构建、不规则的计算和随机内存访问模式大大限制了它们的吞吐量。最近提出的基于散列的高效点变换器(HEPT)通过在注意力计算中使用局部敏感散列(LSH)为大型点云处理提供了理论上保证的近线性复杂度；然而，它的评估主要集中在嵌入质量上，并且 HEPT 所依赖的对象凝聚管道需要一个后处理聚类步骤（如 DBScan），这可能会占用运行时间。在这项工作中，我们做出了两个贡献。首先，我们在相同的数据集和指标下，对 HEPT 和一个代表性的基于 GNN 的管道的物理跟踪性能进行了统一、公平的评估。其次，我们通过添加一个轻量级解码器扩展了 HEPT，该解码器消除了聚类阶段，直接预测轨迹分配。这种修改保留了 HEPT 的规则、硬件友好的计算特性，同时实现了超快的端到端推理。在 TrackML 数据集上，优化后的 HEPTv2 在 A100 上每事件处理时间约为 28 毫秒，同时保持了竞争性的跟踪效率。这些结果使 HEPTv2 成为基于 GNN 的管道用于快速跟踪的实用、可扩展的替代方案。


### 论文摘要

Charged particle track reconstruction is a foundational task in collider experiments and the main computational bottleneck in particle reconstruction. Graph neural networks (GNNs) have shown strong performance for this problem, but costly graph construction, irregular computations, and random memory access patterns substantially limit their throughput. The recently proposed Hashing-based Efficient Point Transformer (HEPT) offers a theoretically guaranteed near-linear complexity for large point cloud processing via locality-sensitive hashing (LSH) in attention computations; however, its evaluations have largely focused on embedding quality, and the object condensation pipeline on which HEPT relies requires a post-hoc clustering step (e.g., DBScan) that can dominate runtime. In this work, we make two contributions. First, we present a unified, fair evaluation of physics tracking performance for HEPT and a representative GNN-based pipeline under the same dataset and metrics. Second, we introduce HEPTv2 by extending HEPT with a lightweight decoder that eliminates the clustering stage and directly predicts track assignments. This modification preserves HEPT's regular, hardware-friendly computations while enabling ultra-fast end-to-end inference. On the TrackML dataset, optimized HEPTv2 achieves approximately 28 ms per event on an A100 while maintaining competitive tracking efficiency. These results position HEPTv2 as a practical, scalable alternative to GNN-based pipelines for fast tracking.

---

## 29. Human Action Recognition from Point Clouds over Time

**论文链接:** [http://arxiv.org/abs/2510.05506v3](http://arxiv.org/abs/2510.05506v3)

**作者:** James Dickens

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种基于3D视频的新人类动作识别方法，利用点云数据作为第三种识别途径，结合基于点和稀疏卷积的技术，在NTU RGB-D 120数据集上达到了89.3%的准确率，超过了之前的点云动作识别方法。

### 背景

最近的人类动作识别研究主要集中在骨骼动作识别和基于视频的方法。随着消费级深度传感器和激光雷达设备的普及，利用密集3D数据进行动作识别的机会正在增加，为动作识别提供了第三种可能性。

### 目的

开发一种不同于骨骼动作识别和基于视频方法的第三种动作识别途径，通过3D视频识别动作，并提高识别准确率。

### 方法

提出了一种新方法，包括一个流程，用于分割场景中人体点云与背景，跟踪个体随时间变化，并进行身体部位分割。该方法支持来自深度传感器和单目深度估计的点云。提出的HAR框架核心是一种新的3D动作识别骨干网络，结合了基于点技术和稀疏卷积网络。实验包括表面法线、颜色、红外强度和身体部位解析标签等辅助点特征。

### 主要发现

在NTU RGB-D 120数据集上的评估表明，该方法与现有的骨骼动作识别算法具有竞争力。在集成设置中结合基于传感器和估计的深度输入，当考虑不同受试者进行训练和测试时，该方法达到89.3%的准确率，超过了之前的点云动作识别方法。

### 结论

该方法利用3D点云数据提供了一种有效的人类动作识别方法，结合了基于点和稀疏卷积的技术，并能够处理来自不同来源的深度数据。

### 翻译

最近的人类动作识别研究主要集中在骨骼动作识别和基于视频的方法。随着消费级深度传感器和激光雷达设备的日益普及，利用密集3D数据进行动作识别的机会正在增加，这为动作识别提供了第三种方法。本文通过引入一种流程，提出了一种从3D视频识别动作的新方法，该流程分割场景中人体点云与背景，跟踪个体随时间变化，并进行身体部位分割。该方法支持来自深度传感器和单目深度估计的点云。所提出的HAR框架核心是一种用于3D动作识别的新骨干网络，它将基于点技术与应用于体素映射点云序列的稀疏卷积网络相结合。实验包括辅助点特征，如表面法线、颜色、红外强度和身体部位解析标签，以提高识别准确性。在NTU RGB-D 120数据集上的评估表明，该方法与现有的骨骼动作识别算法具有竞争力。此外，在集成设置中结合基于传感器和估计的深度输入，当考虑不同受试者进行训练和测试时，该方法达到89.3%的准确率，超过了之前的点云动作识别方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于点云序列的人体动作识别问题。目前人体动作识别主要依赖骨骼数据和视频数据，而随着深度传感器普及，利用3D点云数据是新兴方向。这个问题在现实中非常重要，可用于监控中的异常检测和跌倒识别、自动视频标注（如体育分析）、以及自动驾驶中确保行人安全等场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：视频方法需降低帧大小影响精度，骨骼方法易受遮挡和噪声影响，点云方法缺乏人体分割和背景去除。作者借鉴了点云深度学习中的PointNet系列和稀疏卷积网络（如Minkowski Networks），以及动作识别领域的3DV、PSTNet等方法。作者设计了一个新流程，支持深度传感器和RGB视频两种输入方式，通过人体分割、跟踪和身体部位分割去除背景干扰，并开发了结合点基技术和稀疏卷积的混合架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用点云序列而非传统视频或骨骼数据进行人体动作识别，结合点基技术和稀疏卷积网络优势，通过人体分割去除背景干扰，并利用辅助特征提高精度。整体流程包括：1)人体点云获取（深度传感器或RGB+单目深度估计）；2)预处理（实例掩码去噪、3D点云去噪、人体跟踪、点采样、表面法线计算）；3)动作识别模型（T-Net嵌入、体素映射、稀疏CNN主干、全局稀疏最大池化、全连接分类）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新型人体点云获取流程，支持两种输入方式并进行完整人体分割；2)混合架构主干网络，结合点基技术和稀疏卷积；3)系统利用表面法线、红外强度/颜色、身体部位标签等辅助特征；4)高效的多人物处理方法。相比之前工作，本文方法提供了图像和3D空间中的人体分割，支持单目深度估计，采用混合架构而非单一技术路线，并系统探索了辅助特征的影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种新颖的基于点云序列的人体动作识别方法，通过结合人体分割、稀疏卷积网络和辅助特征，在NTU RGB-D 120数据集上实现了与现有骨骼动作识别算法相竞争的性能，并在跨主题设置下达到了新的最先进水平。'}


### 论文摘要

Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods. With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way. This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation. The method supports point clouds from both depth sensors and monocular depth estimation. At the core of the proposed HAR framework is a novel backbone for 3D action recognition, which combines point-based techniques with sparse convolutional networks applied to voxel-mapped point cloud sequences. Experiments incorporate auxiliary point features including surface normals, color, infrared intensity, and body part parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D 120 dataset demonstrates that the method is competitive with existing skeletal action recognition algorithms. Moreover, combining both sensor-based and estimated depth inputs in an ensemble setup, this approach achieves 89.3% accuracy when different human subjects are considered for training and testing, outperforming previous point cloud action recognition methods.

---

## 30. gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity

**论文链接:** [http://arxiv.org/abs/2510.08450v1](http://arxiv.org/abs/2510.08450v1)

**作者:** Hugh Blayney, Álvaro Arroyo, Xiaowen Dong, Michael M. Bronstein

**发布时间:** 2025-10-09

**备注:** 22 pages, 22 figures, 7 tables

### GPT解析

### 总结

本文研究了图神经网络中的过压缩问题，从模型存储和检索容量角度重新审视此现象，引入新合成任务证明信息瓶颈会饱和容量，并借鉴序列建模思想开发新型GNN架构提升容量。

### 背景

图神经网络利用图结构通过消息传递机制在节点间传输信息，但存在过压缩问题，即大量感受野的信息被压缩到固定大小的向量中，导致信息瓶颈。

### 目的

重新审视过压缩现象，研究现有测量过压缩任务的局限性，引入新合成任务证明信息瓶颈的影响，并开发具有改进容量的新型GNN架构。

### 方法

借鉴序列建模文献中的关联记忆、快速权重编程和xLSTM模型思想，开发新型GNN架构；在容量合成任务和多种真实世界图基准测试上评估性能。

### 主要发现

信息瓶颈会饱和模型的存储和检索容量；现有测量过压缩的任务存在局限性；新型GNN架构在容量任务和真实图基准上表现出强大性能。

### 结论

通过从存储和检索容量角度研究过压缩问题，并借鉴序列建模思想，成功开发了具有改进容量的新型GNN架构，解决了过压缩导致的信息瓶颈问题。

### 翻译

图神经网络(GNNs)利用图结构在节点间传输信息，通常通过消息传递机制。虽然这些模型已找到广泛应用，但它们容易受到过压缩的影响，即大量感受野的节点表示信息被压缩到单个固定大小的向量中，导致信息瓶颈。本文我们从模型存储和检索容量的角度重新审视过压缩现象，将其定义为节点表示中可存储供后续使用的信息量。我们研究了现有用于测量过压缩的任务的一些局限性，并引入了一个新的合成任务来证明信息瓶颈会饱和此容量。此外，我们借鉴序列建模文献中关于关联记忆、快速权重编程和xLSTM模型的思想，开发了一种具有改进容量的新型GNN架构。我们在容量合成任务以及多种真实世界图基准测试上证明了该架构的强大性能。


### 论文摘要

Graph Neural Networks (GNNs) leverage the graph structure to transmit information between nodes, typically through the message-passing mechanism. While these models have found a wide variety of applications, they are known to suffer from over-squashing, where information from a large receptive field of node representations is collapsed into a single fixed sized vector, resulting in an information bottleneck. In this paper, we re-examine the over-squashing phenomenon through the lens of model storage and retrieval capacity, which we define as the amount of information that can be stored in a node's representation for later use. We study some of the limitations of existing tasks used to measure over-squashing and introduce a new synthetic task to demonstrate that an information bottleneck can saturate this capacity. Furthermore, we adapt ideas from the sequence modeling literature on associative memories, fast weight programmers, and the xLSTM model to develop a novel GNN architecture with improved capacity. We demonstrate strong performance of this architecture both on our capacity synthetic task, as well as a range of real-world graph benchmarks.

---

## 31. Verifying Graph Neural Networks with Readout is Intractable

**论文链接:** [http://arxiv.org/abs/2510.08045v1](http://arxiv.org/abs/2510.08045v1)

**作者:** Artem Chernobrovkin, Marco Sälzer, François Schwarzentruber, Nicolas Troquard

**发布时间:** 2025-10-09

### GPT解析

### 总结

本研究介绍了一种用于量化聚合组合图神经网络(ACR-GNNs)的逻辑语言，证明了量化GNN验证任务的计算复杂性，并通过实验验证了量化ACR-GNN模型的效率和性能。

### 背景

图神经网络(GNNs)在许多领域有广泛应用，但量化GNN的验证问题在计算上是复杂的，需要研究如何确保基于GNN的系统的安全性。

### 目的

开发一种逻辑语言来推理量化聚合组合图神经网络，并研究其验证任务的计算复杂性，同时探索量化模型的效率和性能。

### 方法

提出了一种逻辑语言来表征量化ACR-GNNs，使用该逻辑语言证明了量化GNN验证任务的计算复杂性，并通过实验评估了量化模型的性能。

### 主要发现

提供了量化ACR-GNNs的逻辑表征；证明了具有读取功能的量化GNN验证任务是(co)NEXPTIME完全的；量化ACR-GNN模型是轻量级的，同时保持良好的准确性和泛化能力。

### 结论

量化GNN的验证在计算上是不可处理的，这促使了确保基于GNN的系统安全性的大量研究努力。量化ACR-GNN模型在保持性能的同时具有计算效率优势。

### 翻译

我们介绍了一种用于推理具有全局读取功能的量化聚合组合图神经网络(ACR-GNNs)的逻辑语言。我们提供了逻辑表征并使用它证明了具有读取功能的量化GNN验证任务是(co)NEXPTIME完全的。这一结果表明量化GNN的验证在计算上是不可处理的，促使了大量研究努力以确保基于GNN的系统的安全性。我们还通过实验证明，量化ACR-GNN模型是轻量级的，同时与非量化模型相比保持良好的准确性和泛化能力。


### 论文摘要

We introduce a logical language for reasoning about quantized aggregate-combine graph neural networks with global readout (ACR-GNNs). We provide a logical characterization and use it to prove that verification tasks for quantized GNNs with readout are (co)NEXPTIME-complete. This result implies that the verification of quantized GNNs is computationally intractable, prompting substantial research efforts toward ensuring the safety of GNN-based systems. We also experimentally demonstrate that quantized ACR-GNN models are lightweight while maintaining good accuracy and generalization capabilities with respect to non-quantized models.

---

## 32. GraphEnet: Event-driven Human Pose Estimation with a Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2510.07990v1](http://arxiv.org/abs/2510.07990v1)

**作者:** Gaurvi Goyal, Pham Cong Thuong, Arren Glover, Masayoshi Mizuno, Chiara Bartolozzi

**发布时间:** 2025-10-09

### GPT解析

### 总结

该研究提出了一种名为GraphEnet的图神经网络，用于基于事件相机的2D人体姿态估计，利用事件数据的稀疏特性和基于线条的中间表示实现高频姿态估计。

### 背景

人体姿态估计是人机交互应用的关键模块，深度学习技术使RGB相机和商业GPU的鲁棒方法普及；基于事件的相机因其低延迟和低能耗优势在视觉研究社区受到关注，特别适合便携式电子设备和移动机器人等资源受限场景。

### 目的

开发一种能够利用事件相机稀疏输出特性的方法，通过基于线条的中间事件表示，实现单人2D人体姿态的高频率估计。

### 方法

提出名为GraphEnet的图神经网络架构，采用新颖的偏移向量学习范式和基于置信度的池化技术进行人体姿态估计，这是首次将图神经网络应用于事件数据的人体姿态估计工作。

### 主要发现

GraphEnet能够有效利用事件相机的稀疏特性，实现高频2D人体姿态估计，为资源受限场景提供了高效解决方案。

### 结论

基于事件相机和图神经网络的人体姿态估计方法为低延迟、低能耗应用提供了新途径，相关代码已开源。

### 翻译

人体姿态估算是人机交互应用中的关键模块，特别是自深度学习技术兴起以来，使用RGB相机和商业GPU的鲁棒方法已可供消费者使用。另一方面，基于事件的相机在视觉研究社区中因其低延迟和低能耗优势而越来越受欢迎，这些优势使它们成为资源受限应用（如便携式电子设备和移动机器人）的理想选择。在这项工作中，我们提出了一种图神经网络GraphEnet，它利用事件相机输出的稀疏特性，通过基于线条的中间事件表示，以高频率估计单人的2D人体姿态。该架构结合了一种新颖的偏移向量学习范式和基于置信度的池化来估计人体姿态。这是首次将图神经网络应用于事件数据进行人体姿态估计的工作。代码已在https://github.com/event-driven-robotics/GraphEnet-NeVi-ICCV2025开源。


### 论文摘要

Human Pose Estimation is a crucial module in human-machine interaction applications and, especially since the rise in deep learning technology, robust methods are available to consumers using RGB cameras and commercial GPUs. On the other hand, event-based cameras have gained popularity in the vision research community for their low latency and low energy advantages that make them ideal for applications where those resources are constrained like portable electronics and mobile robots. In this work we propose a Graph Neural Network, GraphEnet, that leverages the sparse nature of event camera output, with an intermediate line based event representation, to estimate 2D Human Pose of a single person at a high frequency. The architecture incorporates a novel offset vector learning paradigm with confidence based pooling to estimate the human pose. This is the first work that applies Graph Neural Networks to event data for Human Pose Estimation. The code is open-source at https://github.com/event-driven-robotics/GraphEnet-NeVi-ICCV2025.

---

## 33. Meta-Learning Based Few-Shot Graph-Level Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.07847v1](http://arxiv.org/abs/2510.07847v1)

**作者:** Liting Li, Yumeng Wang, Yueheng Sun

**发布时间:** 2025-10-09

**备注:** Accepted by ARRML2025

### GPT解析

### 总结

本文提出了一种名为MA-GAD的元学习框架，用于解决图级别异常检测在少样本条件下的挑战，通过结合图压缩模块和元学习方法提高异常检测性能和鲁棒性。

### 背景

图级别异常检测在欺诈检测、评论分类和生物化学等领域具有重要作用。虽然图神经网络(GNNs)在该领域取得了进展，但现有方法严重依赖大量标记数据，而现实场景中这类数据往往不可用。此外，基于GNN的少样本异常检测方法容易受到噪声干扰，导致嵌入质量差和模型鲁棒性降低。

### 目的

解决现有图级别异常检测方法对大量标记数据的依赖问题，以及少样本条件下噪声干扰导致的性能下降问题，提高模型在有限样本情况下的异常检测性能和鲁棒性。

### 方法

提出MA-GAD框架，包含三个关键组件：1)图压缩模块减少图的大小，减轻噪声干扰同时保留关键节点信息；2)元学习方法从相似网络中提取元异常信息，学习能快速适应新任务的初始化模型；3)偏置网络增强异常节点与正常节点之间的区分度。

### 主要发现

基于四个真实生物化学数据集的实验结果表明，在少样本条件下，MA-GAD在图级别异常检测任务上优于现有最先进的方法。在图异常和子图异常检测任务上的实验验证了该框架在真实数据集上的有效性。

### 结论

MA-GAD框架通过结合图压缩和元学习方法，有效解决了少样本条件下图级别异常检测面临的挑战，提高了异常检测的性能和鲁棒性，为现实场景中的异常检测问题提供了有效解决方案。

### 翻译

图级别异常检测旨在识别图数据集中的异常图或子图，在欺诈检测、评论分类和生物化学等多个领域发挥着重要作用。虽然图神经网络(GNNs)在该领域取得了显著进展，但现有方法严重依赖大量标记数据，而现实场景中这类数据往往不可用。此外，基于GNN的少样本异常检测方法容易受到噪声干扰，导致嵌入质量差和模型鲁棒性降低。为解决这些挑战，我们提出了一种新颖的基于元学习的图级别异常检测框架(MA-GAD)，包含一个图压缩模块，该模块减少图的大小，减轻噪声干扰同时保留基本节点信息。我们还利用元学习从相似网络中提取元异常信息，使模型能够学习一种初始化模型，该模型可以快速适应有限样本的新任务。这提高了目标图上的异常检测性能，并且使用偏置网络来增强异常节点与正常节点之间的区分度。基于四个真实生物化学数据集的实验结果表明，在少样本条件下，MA-GAD在图级别异常检测任务上优于现有最先进的方法。在图异常和子图异常检测任务上的实验验证了该框架在真实数据集上的有效性。


### 论文摘要

Graph-level anomaly detection aims to identify anomalous graphs or subgraphs within graph datasets, playing a vital role in various fields such as fraud detection, review classification, and biochemistry. While Graph Neural Networks (GNNs) have made significant progress in this domain, existing methods rely heavily on large amounts of labeled data, which is often unavailable in real-world scenarios. Additionally, few-shot anomaly detection methods based on GNNs are prone to noise interference, resulting in poor embedding quality and reduced model robustness. To address these challenges, we propose a novel meta-learning-based graph-level anomaly detection framework (MA-GAD), incorporating a graph compression module that reduces the graph size, mitigating noise interference while retaining essential node information. We also leverage meta-learning to extract meta-anomaly information from similar networks, enabling the learning of an initialization model that can rapidly adapt to new tasks with limited samples. This improves the anomaly detection performance on target graphs, and a bias network is used to enhance the distinction between anomalous and normal nodes. Our experimental results, based on four real-world biochemical datasets, demonstrate that MA-GAD outperforms existing state-of-the-art methods in graph-level anomaly detection under few-shot conditions. Experiments on both graph anomaly and subgraph anomaly detection tasks validate the framework's effectiveness on real-world datasets.

---

## 34. DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support

**论文链接:** [http://arxiv.org/abs/2510.07620v1](http://arxiv.org/abs/2510.07620v1)

**作者:** Muhammad Usman, Yugyung Lee

**发布时间:** 2025-10-08

**备注:** 18 pages, 9 figures, 5 tables

### GPT解析

### 总结

DGTEN是一种基于深度高斯的信任评估网络，通过统一图框架实现动态信任评估，能够捕捉变化关系、表达校准置信度并抵抗对手操纵。

### 背景

在大型、快速演化的图中进行动态信任评估需要能够捕捉变化关系、表达校准置信度并抵抗对手操纵的模型。

### 目的

开发一个统一的图框架，实现不确定性感知的消息传递、表达性时间建模和对信任定向攻击的内置防御。

### 方法

DGTEN将节点和边表示为高斯分布，使用混合绝对-高沙漏位置编码和基于Kolmogorov-Arnold网络的无偏多头注意力机制建模信任演化，采用基于常微分方程的残差学习模块捕捉突变和平滑趋势，并通过鲁棒的自适应集成系数分析降低可疑交互的权重。

### 主要发现

在Bitcoin-Alpha上的单时隙预测中，MCC比最佳动态基线提高10.77%；在冷启动场景中，实现16.41%的MCC增益；在对抗性开关攻击下，MCC比基线高出高达11.63%。

### 结论

DGTEN统一框架的有效性在两个签名比特币信任网络上得到了验证。

### 翻译

在大型、快速演化的图中进行动态信任评估需要能够捕捉变化关系、表达校准置信度并抵抗对手操纵的模型。DGTEN（基于深度高斯的信任评估网络）引入了一个统一的图框架，通过结合不确定性感知的消息传递、表达性时间建模和对信任定向攻击的内置防御，实现了这三个目标。它将节点和边表示为高斯分布，使语义信号和认知不确定性通过图神经网络传播，从而实现风险感知的信任决策，而非过度自信的猜测。为了建模信任的演化，它采用混合绝对-高沙漏（HAGH）位置编码和基于Kolmogorov-Arnold网络的无偏多头注意力， followed by an ordinary differential equation (ODE)-based residual learning module to jointly capture abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity measures, mitigating reputation laundering, sabotage, and on/off attacks. On two signed Bitcoin trust networks, DGTEN delivers significant improvements: in single-timeslot prediction on Bitcoin-Alpha, it improves MCC by 10.77% over the best dynamic baseline; in the cold-start scenario, it achieves a 16.41% MCC gain - the largest across all tasks and datasets. Under adversarial on/off attacks, it surpasses the baseline by up to 11.63% MCC. These results validate the effectiveness of the unified DGTEN framework.


### 论文摘要

Dynamic trust evaluation in large, rapidly evolving graphs requires models that can capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-based Trust Evaluation Network) introduces a unified graph framework that achieves all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To model how trust evolves, it employs hybrid Absolute-Gaussian-Hourglass (HAGH) positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, followed by an ordinary differential equation (ODE)-based residual learning module to jointly capture abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity measures, mitigating reputation laundering, sabotage, and on/off attacks. On two signed Bitcoin trust networks, DGTEN delivers significant improvements: in single-timeslot prediction on Bitcoin-Alpha, it improves MCC by 10.77% over the best dynamic baseline; in the cold-start scenario, it achieves a 16.41% MCC gain - the largest across all tasks and datasets. Under adversarial on/off attacks, it surpasses the baseline by up to 11.63% MCC. These results validate the effectiveness of the unified DGTEN framework.

---

## 35. Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting

**论文链接:** [http://arxiv.org/abs/2510.07426v1](http://arxiv.org/abs/2510.07426v1)

**作者:** Walid Guettala, Yufan Zhao, László Gulyás

**发布时间:** 2025-10-08

**备注:** Accepted to IEEE ICTAI 2025. Version 0.9. 10 pages, 5 figures.  Preprint differs from the published version in formatting and minor wording

### GPT解析

### 总结

本研究提出了TESTAM+框架，通过引入空间语义专家整合物理道路拓扑和数据驱动特征相似性，显著提升了交通预测性能，同时减少了计算复杂度。

### 背景

交通预测对智能交通系统至关重要，可缓解拥堵并减少排放。然而，现有的混合专家框架如TESTAM缺乏对物理道路网络拓扑的明确整合，限制了其空间能力。

### 目的

开发一个增强的空间时间预测框架TESTAM+，通过结合物理道路拓扑和数据驱动特征相似性来提高交通预测的准确性和效率。

### 方法

引入了新的空间语义专家(SpatioSemantic Expert)，通过混合图构建整合物理道路拓扑和数据驱动特征相似性，改进了原有的TESTAM框架。

### 主要发现

TESTAM+相比TESTAM有显著改进：在METR LA上MAE减少1.3%，在PEMS BAY上改进4.1%；战略专家选择优于简单集成聚合；单个专家表现出色，自适应专家和空间语义专家都达到1.63 MAE；最佳配置相比MegaCRN在METR LA上实现11.5%的MAE减少；推理延迟减少53.1%。

### 结论

更少但战略设计的专家优于复杂的多专家集成，以优越的计算效率建立了新的最先进性能，适合实时部署。

### 翻译

交通预测是智能交通系统的基础，能够在日益复杂的城市环境中实现拥堵缓解和排放减少。虽然最近的图神经网络方法已推进了空间时间建模，但现有的混合专家框架如时间增强空间时间注意力模型(TESTAM)缺乏对物理道路网络拓扑的明确整合，限制了其空间能力。我们提出了TESTAM+，一个增强的空间时间预测框架，引入了新的空间语义专家，通过混合图构建将物理道路拓扑与数据驱动特征相似性相结合。TESTAM+相比TESTAM实现了显著改进：在METR LA上MAE减少1.3%（3.10 vs 3.14），在PEMS BAY上改进4.1%（1.65 vs 1.72）。通过全面的消融研究，我们发现战略专家选择从根本上优于简单的集成聚合。单个专家表现出 remarkable 的有效性：自适应专家在PEMS BAY上达到1.63 MAE，优于原始三个专家的TESTAM（1.72 MAE），而空间语义专家以相同的1.63 MAE匹配这一性能。最佳的Identity + Adaptive配置相比最先进的MegaCRN在METR LA上实现了11.5%的MAE减少（2.99 vs 3.38），同时相比完整的四个专家TESTAM+减少了53.1%的推理延迟。我们的发现揭示出，更少但战略设计的专家优于复杂的多专家集成，以优越的计算效率建立了新的最先进性能，适合实时部署。


### 论文摘要

Traffic forecasting is fundamental to intelligent transportation systems, enabling congestion mitigation and emission reduction in increasingly complex urban environments. While recent graph neural network approaches have advanced spatial temporal modeling, existing mixture of experts frameworks like Time Enhanced Spatio Temporal Attention Model (TESTAM) lack explicit incorporation of physical road network topology, limiting their spatial capabilities. We present TESTAM+, an enhanced spatio temporal forecasting framework that introduces a novel SpatioSemantic Expert integrating physical road topology with data driven feature similarity through hybrid graph construction. TESTAM+ achieves significant improvements over TESTAM: 1.3% MAE reduction on METR LA (3.10 vs. 3.14) and 4.1% improvement on PEMS BAY (1.65 vs. 1.72). Through comprehensive ablation studies, we discover that strategic expert selection fundamentally outperforms naive ensemble aggregation. Individual experts demonstrate remarkable effectiveness: the Adaptive Expert achieves 1.63 MAE on PEMS BAY, outperforming the original three expert TESTAM (1.72 MAE), while the SpatioSemantic Expert matches this performance with identical 1.63 MAE. The optimal Identity + Adaptive configuration achieves an 11.5% MAE reduction compared to state of the art MegaCRN on METR LA (2.99 vs. 3.38), while reducing inference latency by 53.1% compared to the full four expert TESTAM+. Our findings reveal that fewer, strategically designed experts outperform complex multi expert ensembles, establishing new state of the art performance with superior computational efficiency for real time deployment.

---

## 36. Out-of-Distribution Generalization in Climate-Aware Yield Prediction with Earth Observation Data

**论文链接:** [http://arxiv.org/abs/2510.07350v1](http://arxiv.org/abs/2510.07350v1)

**作者:** Aditya Chakravarty

**发布时间:** 2025-10-08

### GPT解析

### 总结

该研究评估了两种深度学习模型(GNN-RNN和MMST-ViT)在气候变化背景下农作物产量预测中的跨区域泛化能力，发现模型性能在不同地理区域间存在显著差异，GNN-RNN表现出更好的泛化能力和计算效率。

### 背景

气候变化正在扰乱农业系统，准确的农作物产量预测对粮食安全至关重要。深度学习模型在使用卫星和天气数据进行产量预测方面显示出潜力，但它们跨地理区域和年份泛化的能力尚未得到充分检验。

### 目的

在现实的分布外(OOD)条件下对两种最先进的模型GNN-RNN和MMST-ViT进行基准测试，评估其跨区域和跨年份的泛化能力。

### 方法

使用大规模CropNet数据集（涵盖2017-2022年美国1200多个县），在七个美国农业部农场资源区域进行留一集群交叉验证和提前一年预测场景，比较两种模型的性能。

### 主要发现

GNN-RNN表现出更好的泛化能力和计算效率（训练速度比MMST-ViT快135倍）；MMST-ViT在域内表现良好但在OOD条件下性能急剧下降；心脏地带和大平原北部地区转移稳定（RMSE<10蒲式耳/英亩），而草原通道持续表现不佳（RMSE>20蒲式耳/英亩），反映了气候和灌溉差异的影响。

### 结论

空间-时间对齐是稳健泛化的关键因素，需要透明的OOD评估协议以确保公平可靠的气候感知农业预测系统。

### 翻译

气候变化日益扰乱农业系统，使准确的农作物产量预测对粮食安全至关重要。虽然深度学习模型在使用卫星和天气数据进行产量预测方面显示出潜力，但它们跨地理区域和年份泛化的能力（对实际部署至关重要）尚未得到充分检验。我们使用大规模CropNet数据集（涵盖2017-2022年美国1200多个县）在现实的分布外(OOD)条件下对两种最先进的模型GNN-RNN和MMST-ViT进行基准测试。通过在七个美国农业部农场资源区域进行留一集群交叉验证和提前一年预测场景，我们确定了跨区域转移能力的显著变异性。GNN-RNN表现出更好的泛化能力，在地理变化下呈现正相关，而MMST-ViT在域内表现良好但在OOD条件下性能急剧下降。心脏地带和大平原北部地区显示出稳定的转移动态（大豆RMSE小于10蒲式耳/英亩），而草原通道在两种模型和作物中持续表现不佳（RMSE大于20蒲式耳/英亩），揭示了由半干旱气候、灌溉模式和光谱覆盖不完整导致的结构差异。除了准确性差异外，GNN-RNN的训练速度比MMST-ViT快135倍（14分钟对比31.5小时），使其更可持续部署。我们的研究结果强调，空间-时间对齐（不仅仅是模型复杂性或数据规模）是稳健泛化的关键，并需要透明的OOD评估协议以确保公平可靠的气候感知农业预测。


### 论文摘要

Climate change is increasingly disrupting agricultural systems, making accurate crop yield forecasting essential for food security. While deep learning models have shown promise in yield prediction using satellite and weather data, their ability to generalize across geographic regions and years - critical for real-world deployment - remains largely untested. We benchmark two state-of-the-art models, GNN-RNN and MMST-ViT, under realistic out-of-distribution (OOD) conditions using the large-scale CropNet dataset spanning 1,200+ U.S. counties from 2017-2022. Through leave-one-cluster-out cross-validation across seven USDA Farm Resource Regions and year-ahead prediction scenarios, we identify substantial variability in cross-region transferability. GNN-RNN demonstrates superior generalization with positive correlations under geographic shifts, while MMST-ViT performs well in-domain but degrades sharply under OOD conditions. Regions like Heartland and Northern Great Plains show stable transfer dynamics (RMSE less than 10 bu/acre for soybean), whereas Prairie Gateway exhibits persistent underperformance (RMSE greater than 20 bu/acre) across both models and crops, revealing structural dissimilarities likely driven by semi-arid climate, irrigation patterns, and incomplete spectral coverage. Beyond accuracy differences, GNN-RNN achieves 135x faster training than MMST-ViT (14 minutes vs. 31.5 hours), making it more viable for sustainable deployment. Our findings underscore that spatial-temporal alignment - not merely model complexity or data scale - is key to robust generalization, and highlight the need for transparent OOD evaluation protocols to ensure equitable and reliable climate-aware agricultural forecasting.

---

## 37. FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption

**论文链接:** [http://arxiv.org/abs/2510.08217v1](http://arxiv.org/abs/2510.08217v1)

**作者:** Justus Viga, Penelope Mueck, Alexander Löser, Torben Weis

**发布时间:** 2025-10-09

**备注:** This preprint has not undergone peer review or any post-submission  improvements or corrections. The Version of Record of this contribution will  be published in "ECML PKDD Workshop 2025 - Advanced Analytics and Learning on  Temporal Data"

### GPT解析

### 总结

本研究提出了一种新的船舶燃料消耗预测方法，通过引入新数据集、标准化基准和基础模型应用，实现了准确预测。

### 背景

在航运业中，燃料消耗和排放是关键因素，对经济效率和环境影响重大。准确预测船舶燃料消耗对优化海运运营至关重要。

### 目的

解决船舶燃料消耗预测中存在的异构方法和有限高质量数据集问题，提高预测准确性。

### 方法

引入并发布包含三艘船运营和环境数据的新数据集；定义标准化基准覆盖表格回归和时间序列回归；研究使用TabPFN基础模型进行船舶消耗建模的上下文学习应用。

### 主要发现

所有评估模型表现良好，支持船上数据驱动燃料预测的可行性；包含环境条件的模型优于仅依赖速度的基线；TabPFN略优于其他技术；包含时间背景可提高准确性。

### 结论

具有上下文学习能力的基础模型在船舶燃料消耗预测中具有良好性能，为优化海运运营提供了有效工具。

### 翻译

在航运业中，燃料消耗和排放是关键因素，因为它们对经济效率和环境影响重大。准确预测船舶燃料消耗对于进一步优化海运运营至关重要。然而，异构方法和有限的高质量数据集阻碍了建模方法的直接比较。本文做出三项关键贡献：(1)我们引入并发布了一个新数据集，包含三艘船的运营和环境数据；(2)我们定义了一个标准化基准，涵盖表格回归和时间序列回归；(3)我们研究了使用TabPFN基础模型进行船舶消耗建模的上下文学习应用-据我们所知，这是该领域的首次尝试。我们的结果表明所有评估的模型都表现出强大的性能，支持了船上、数据驱动的燃料预测的可行性。包含环境条件的模型始终优于仅依赖船舶速度的多项式基线。TabPFN略优于其他技术，突显了具有上下文学习能力的基础模型在表格预测中的潜力。此外，包含时间背景可以提高准确性。


### 论文摘要

In the shipping industry, fuel consumption and emissions are critical factors due to their significant impact on economic efficiency and environmental sustainability. Accurate prediction of ship fuel consumption is essential for further optimization of maritime operations. However, heterogeneous methodologies and limited high-quality datasets hinder direct comparison of modeling approaches. This paper makes three key contributions: (1) we introduce and release a new dataset (https://huggingface.co/datasets/krohnedigital/FuelCast) comprising operational and environmental data from three ships; (2) we define a standardized benchmark covering tabular regression and time-series regression (3) we investigate the application of in-context learning for ship consumption modeling using the TabPFN foundation model - a first in this domain to our knowledge. Our results demonstrate strong performance across all evaluated models, supporting the feasibility of onboard, data-driven fuel prediction. Models incorporating environmental conditions consistently outperform simple polynomial baselines relying solely on vessel speed. TabPFN slightly outperforms other techniques, highlighting the potential of foundation models with in-context learning capabilities for tabular prediction. Furthermore, including temporal context improves accuracy.

---

## 38. Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection

**论文链接:** [http://arxiv.org/abs/2510.08073v1](http://arxiv.org/abs/2510.08073v1)

**作者:** Shuhai Zhang, ZiHao Lian, Jiahao Yang, Daiyuan Li, Guoxuan Pang, Feng Liu, Bo Han, Shutao Li, Mingkui Tan

**发布时间:** 2025-10-09

**备注:** Accepted at NeurIPS 2025 spotlight

### GPT解析

### 总结

该研究提出了一种基于物理原理的AI生成视频检测方法，通过引入归一化时空梯度(Normalized Spatiotemporal Gradient, NSG)统计量，开发了一种高效的AI生成视频检测框架NSG-VD，实验证明该方法在召回率和F1分数上显著优于现有基线方法。

### 背景

AI生成的视频已达到近乎完美的视觉真实感（如Sora模型），迫切需要可靠的检测机制。然而，这类视频的检测面临两大挑战：一是建模高维时空动力学，二是识别违反物理规律的细微异常。

### 目的

开发一种基于物理原理的AI生成视频检测方法，有效解决现有方法在检测高维时空动态和物理异常方面的局限性。

### 方法

提出基于概率流守恒原理的物理驱动检测范式；定义NSG统计量量化空间概率梯度与时间密度变化的比率；利用预训练扩散模型通过空间梯度近似和运动感知时间建模开发NSG估计器；提出NSG-VD方法计算测试视频与真实视频NSG特征间的最大均值差异作为检测指标；推导真实与生成视频NSG特征距离上界。

### 主要发现

NSG-VD在召回率上比最先进基线方法高16.00%，在F1分数上高10.75%；生成视频因分布偏移表现出比真实视频更大的NSG特征差异；所提方法无需复杂运动分解同时保持物理约束。

### 结论

基于物理原理的NSG-VD方法在AI生成视频检测任务中表现出色，显著优于现有方法，为应对AI生成内容带来的真实性挑战提供了有效解决方案。

### 翻译

AI生成的视频已经达到了近乎完美的视觉真实感（例如Sora模型），迫切需要可靠的检测机制。然而，这类视频的检测在建模高维时空动力学和识别违反物理规律的细微异常方面面临重大挑战。在本文中，我们提出了一种基于概率流守恒原理的物理驱动AI生成视频检测范式。具体而言，我们提出了一种称为归一化时空梯度(Normalized Spatiotemporal Gradient, NSG)的统计量，它量化了空间概率梯度与时间密度变化的比率，明确捕捉了自然视频动态的偏差。利用预训练的扩散模型，我们通过空间梯度近似和运动感知的时间建模开发了NSG估计器，无需复杂的运动分解同时保持物理约束。基于此，我们提出了一种基于NSG的视频检测方法(NSG-VD)，将测试视频和真实视频的NSG特征之间的最大均值差异(MMD)作为检测指标。最后，我们推导了真实视频和生成视频NSG特征距离的上界，证明生成视频由于分布偏移表现出放大的差异。大量实验证实，NSG-VD在召回率上比最先进的基线方法高出16.00%，在F1分数上高出10.75%，验证了NSG-VD的卓越性能。源代码可在https://github.com/ZSHsh98/NSG-VD获取。


### 论文摘要

AI-generated videos have achieved near-perfect visual realism (e.g., Sora), urgently necessitating reliable detection mechanisms. However, detecting such videos faces significant challenges in modeling high-dimensional spatiotemporal dynamics and identifying subtle anomalies that violate physical laws. In this paper, we propose a physics-driven AI-generated video detection paradigm based on probability flow conservation principles. Specifically, we propose a statistic called Normalized Spatiotemporal Gradient (NSG), which quantifies the ratio of spatial probability gradients to temporal density changes, explicitly capturing deviations from natural video dynamics. Leveraging pre-trained diffusion models, we develop an NSG estimator through spatial gradients approximation and motion-aware temporal modeling without complex motion decomposition while preserving physical constraints. Building on this, we propose an NSG-based video detection method (NSG-VD) that computes the Maximum Mean Discrepancy (MMD) between NSG features of the test and real videos as a detection metric. Last, we derive an upper bound of NSG feature distances between real and generated videos, proving that generated videos exhibit amplified discrepancies due to distributional shifts. Extensive experiments confirm that NSG-VD outperforms state-of-the-art baselines by 16.00% in Recall and 10.75% in F1-Score, validating the superior performance of NSG-VD. The source code is available at https://github.com/ZSHsh98/NSG-VD.

---

## 39. MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.07915v1](http://arxiv.org/abs/2510.07915v1)

**作者:** Peiran Wu, Zhuorui Yu, Yunze Liu, Chi-Hao Wu, Enmin Zhou, Junxiao Shen

**发布时间:** 2025-10-09

### GPT解析

### 总结

MARC是一种创新的令牌压缩方法，通过结合结构化检索和强化学习蒸馏技术，显著减少了视频处理中的计算负担，同时保持了较高的准确性，适用于资源受限环境下的实时视频理解应用。

### 背景

大语言模型(LLMs)的快速发展为多模态模型奠定了基础。然而，视觉语言模型(VLMs)在从图像扩展到视频时仍面临高昂的计算成本，这主要是由于高帧率和长持续时间导致的。

### 目的

解决视觉语言模型在视频处理中的高计算成本问题，通过令牌压缩技术实现高效的视频理解，同时保持较高的准确性。

### 方法

提出了MARC(基于记忆增强强化学习的令牌压缩)方法，结合了结构化检索和基于强化学习的蒸馏技术。采用'先检索后压缩'策略，使用视觉记忆检索器(VMR)选择关键片段，并使用压缩组相对策略优化(C-GRPO)框架将推理能力从教师模型蒸馏到学生模型。

### 主要发现

在六个视频基准测试中，MARC仅使用一帧的令牌就能接近基线准确性；视觉令牌减少了95%，GPU内存减少了72%，延迟减少了23.9%。

### 结论

MARC在资源受限的环境(如视频问答、监控和自动驾驶)中具有高效实时视频理解的潜力。

### 翻译

大语言模型(LLMs)的快速发展为多模态模型奠定了基础。然而，视觉语言模型(VLMs)在从图像扩展到视频时仍面临高昂的计算成本，这主要是由于高帧率和长持续时间导致的。令牌压缩是一个有前景的解决方案，但大多数现有的无需训练方法会导致信息丢失和性能下降。为了克服这一问题，我们提出了MARC(基于记忆增强强化学习的令牌压缩)，该方法结合了结构化检索和基于强化学习的蒸馏技术。MARC采用'先检索后压缩'策略，使用视觉记忆检索器(VMR)选择关键片段，并使用压缩组相对策略优化(C-GRPO)框架将推理能力从教师模型蒸馏到学生模型。在六个视频基准测试上的实验表明，MARC仅使用一帧的令牌就能接近基线准确性——视觉令牌减少了95%，GPU内存减少了72%，延迟减少了23.9%。这证明了它在资源受限环境(如视频问答、监控和自动驾驶)中进行高效实时视频理解的潜力。


### 论文摘要

The rapid progress of large language models (LLMs) has laid the foundation for multimodal models. However, visual language models (VLMs) still face heavy computational costs when extended from images to videos due to high frame rates and long durations. Token compression is a promising solution, yet most existing training-free methods cause information loss and performance degradation. To overcome this, we propose \textbf{Memory-Augmented Reinforcement Learning-based Token Compression (MARC)}, which integrates structured retrieval and RL-based distillation. MARC adopts a \textit{retrieve-then-compress} strategy using a \textbf{Visual Memory Retriever (VMR)} to select key clips and a \textbf{Compression Group Relative Policy Optimization (C-GRPO)} framework to distil reasoning ability from a teacher to a student model. Experiments on six video benchmarks show that MARC achieves near-baseline accuracy using only one frame's tokens -- reducing visual tokens by \textbf{95\%}, GPU memory by \textbf{72\%}, and latency by \textbf{23.9\%}. This demonstrates its potential for efficient, real-time video understanding in resource-constrained settings such as video QA, surveillance, and autonomous driving.

---

## 40. GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.07791v1](http://arxiv.org/abs/2510.07791v1)

**作者:** Qinghongbing Xie, Zhaoyuan Xia, Feng Zhu, Lijun Gong, Ziyue Li, Rui Zhao, Long Zeng

**发布时间:** 2025-10-09

**备注:** 20 pages, 13 figures

### GPT解析

### 总结

该论文提出了Geo-Temporal Reasoning benchmark (GTR-Bench)，一个用于评估视觉语言模型(VLMs)地理时空推理能力的新基准测试。研究评估了10多个流行VLMs的表现，发现当前模型存在三个主要缺陷，为时空智能研究提供了新方向。

### 背景

现有时空基准测试主要关注两种推理：基于图像/视频的第一人称视角推理和基于地图的地理视角推理。这些基准测试无法评估VLMs在同时使用图像/视频和图形上下文时的地理时空智能，而这种能力对交通管理和应急响应等领域非常重要。

### 目的

解决现有基准测试的局限性，提出一个能够全面评估VLMs地理时空智能的新基准测试，特别关注大规模摄像头网络中移动目标的地理时间推理。

### 方法

作者设计了Geo-Temporal Reasoning benchmark (GTR-Bench)，这是一个针对大规模摄像头网络中移动目标的地理时间推理的挑战。该基准测试要求模型在地图和视频之间进行多视角切换、跨越多个非重叠视野的视频进行联合推理，以及对未观察到的时空区域进行推理。

### 主要发现

1. 即使是表现最好的专有模型Gemini-2.5-Pro(34.9%)也远低于人类表现(78.61%)
2. 当前VLMs在地理时间推理方面存在三个主要缺陷：
   - 时空上下文利用不平衡
   - 时间预测能力较弱
   - 缺乏对齐地图数据与多视图视频输入的能力

### 结论

GTR-Bench为时空智能的研究和应用提供了有价值的见解和新的机会，有助于推动VLMs在地理时空推理方面的发展。基准测试和代码将公开发布。

### 翻译

最近，视觉语言模型(VLMs)的时空智能因其在自动驾驶、具身人工智能和通用人工智能领域的重要性而受到广泛关注。现有的时空基准测试主要关注基于图像/视频上下文的第一人称视角推理，或基于图形上下文(如地图)的地理视角推理，因此无法评估VLMs在同时使用图像/视频和图形上下文时的地理时空智能，这对交通管理和应急响应等领域非常重要。为解决这些差距，我们引入了Geo-Temporal Reasoning基准测试(GTR-Bench)，这是一个针对大规模摄像头网络中移动目标的地理时间推理的新挑战。GTR-Bench更具挑战性，因为它需要在地图和视频之间进行多视角切换、跨越多个非重叠视野的视频进行联合推理，以及对任何视频上下文都未观察到的时空区域进行推理。对10多个流行VLMs在GTR-Bench上的评估表明，即使是表现最好的专有模型Gemini-2.5-Pro(34.9%)也远低于人类表现(78.61%)。此外，我们在GTR-Bench上的综合分析揭示了当前模型在地理时间推理方面的三个主要缺陷：(1)VLMs的推理受到时空上下文不平衡利用的损害。(2)VLMs在时间预测方面能力较弱，导致在时间强调任务上的表现比在空间强调任务上更差。(3)VLMs缺乏理解或对齐地图数据与多视图视频输入的能力。我们相信GTR-Bench为时空智能的研究和应用提供了有价值的见解和开辟了新的机会。基准测试和代码将在https://github.com/X-Luffy/GTR-Bench发布。


### 论文摘要

Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for Autonomous Driving, Embodied AI and General Artificial Intelligence. Existing spatial-temporal benchmarks mainly focus on egocentric perspective reasoning with images/video context, or geographic perspective reasoning with graphics context (eg. a map), thus fail to assess VLMs' geographic spatial-temporal intelligence with both images/video and graphics context, which is important for areas like traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench demonstrate that even the best proprietary model, Gemini-2.5-Pro (34.9%), significantly lags behind human performance (78.61%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three primary deficiencies of current models for geo-temporal reasoning. (1) VLMs' reasoning is impaired by an imbalanced utilization of spatial-temporal context. (2) VLMs are weak in temporal forecasting, which leads to worse performance on temporal-emphasized tasks than on spatial-emphasized tasks. (3) VLMs lack the proficiency to comprehend or align the map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.

---

## 41. Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models

**论文链接:** [http://arxiv.org/abs/2510.05034v3](http://arxiv.org/abs/2510.05034v3)

**作者:** Yolo Yunlong Tang, Jing Bi, Pinxin Liu, Zhenyu Pan, Zhangyun Tan, Qianxiang Shen, Jiani Liu, Hang Hua, Junjia Guo, Yunzhong Xiao, Chao Huang, Zhiyuan Wang, Susan Liang, Xinyi Liu, Yizhi Song, Yuhe Nie, Jia-Xing Zhong, Bozheng Li, Daiqing Qi, Ziyun Zeng, Ali Vosoughi, Luchuan Song, Zeliang Zhang, Daiki Shimada, Han Liu, Jiebo Luo, Chenliang Xu

**发布时间:** 2025-10-06

**备注:** The 1st version

### GPT解析

### 总结

这篇论文是对Video-LMMs（视频大型多模态模型）后训练方法的首次全面综述，涵盖了监督微调、强化学习和测试时扩展三个基本支柱，提供了结构化的分类法和关键设计原则，旨在为研究人员和实践者提供推进Video-LMMs能力的统一框架。

### 背景

视频理解是计算机视觉领域最具挑战性的前沿，需要模型能够处理复杂的时空关系、长期依赖和多模态证据。Video-LMMs的出现展示了在视频理解任务中的显著能力，但将这些模型从基础感知系统转变为复杂推理引擎的后训练阶段在文献中仍然分散。

### 目的

提供对Video-LMMs后训练方法的首次全面检查，建立统一框架，帮助研究人员和实践者推进Video-LMMs的能力，同时确定关键挑战并提供必要的评估工具。

### 方法

研究通过三个基本支柱分析后训练方法：1）监督微调（SFT）与思维链；2）可验证目标的强化学习（RL）；3）通过增强推理计算的测试时扩展（TTS）。研究还提出了结构化的分类法，明确这些技术在视频理解中的角色和相互关系。

### 主要发现

1) Video-LMMs在视频理解任务中展示了显著能力；2) 后训练方法对将模型转变为复杂推理引擎至关重要；3) 三个基本支柱各有独特作用和挑战；4) 视频特定挑战包括时间定位、时空定位、长视频效率和多模态证据集成；5) 关键设计原则包括奖励设计、可扩展性和成本性能优化。

### 结论

这篇综述为研究人员和实践者提供了推进Video-LMMs能力的统一框架，确定了关键挑战和必要的评估工具，有助于促进该领域的进一步发展。

### 翻译

视频理解代表了计算机视觉中最具挑战性的前沿，需要模型能够推理复杂的时空关系、长期依赖和多模态证据。最近出现的Video-LMMs将视觉编码器与强大的基于解码器的语言模型相结合，在视频理解任务中展示了显著的能力。然而，将这些模型从基础感知系统转变为复杂推理引擎的关键阶段——后训练，在文献中仍然分散。这篇综述首次对Video-LMMs的后训练方法进行了全面检查，涵盖了三个基本支柱：思维链监督微调、可验证目标的强化学习和通过增强推理计算的测试时扩展。我们提出了一个结构化的分类法，阐明了这些技术的作用、相互关系和视频特定适应，解决了时间定位、时空定位、长视频效率和多模态证据集成等独特挑战。通过对代表性方法的系统分析，我们综合了关键设计原则、见解和评估协议，同时确定了奖励设计、可扩展性和成本性能优化方面的关键开放挑战。


### 论文摘要

Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training

---

## 42. ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation

**论文链接:** [http://arxiv.org/abs/2510.08569v1](http://arxiv.org/abs/2510.08569v1)

**作者:** Qin Liu, Jacob Dineen, Yuxi Huang, Sheng Zhang, Hoifung Poon, Ben Zhou, Muhao Chen

**发布时间:** 2025-10-09

**备注:** Preprint

### GPT解析

### 总结

这篇论文介绍了ArenaBencher，一个自动基准测试演进框架，旨在解决大语言模型基准测试中的数据泄露问题。该框架能够在保持可比性的同时更新测试用例，通过推断测试用例核心能力、生成候选问答对、验证正确性和意图，以及从多个模型聚合反馈来选择暴露共同弱点的候选。该框架应用于多个领域，能够产生经过验证的、多样化的和公平的更新，揭示新的失败模式，增加难度并提高模型可分离性。

### 背景

基准测试对于衡量大语言模型的能力和指导模型开发至关重要。然而，从预训练语料库中的广泛数据泄露损害了基准测试的有效性。模型可以匹配记忆内容而非展示真正的泛化能力，这 inflated 了分数，扭曲了跨模型比较，并错误地代表了进展。

### 目的

作者提出ArenaBencher框架的目的是创建一个与模型无关的自动基准测试演进方法，能够在保持可比性的同时更新测试用例，以解决数据泄露问题，确保基准测试能够真正衡量模型的能力而非记忆内容。

### 方法

ArenaBencher框架的工作流程包括：给定现有基准测试和一组要评估的多样化模型，推断每个测试用例的核心能力；生成保留原始目标的候选问答对；使用LLM作为法官验证正确性和意图；从多个模型聚合反馈以选择暴露共同弱点的候选。该过程使用上下文演示迭代运行，引导生成更具挑战性和诊断性的案例。

### 主要发现

作者将ArenaBencher应用于数学问题解决、常识推理和安全领域，发现它能够产生经过验证的、多样化的和公平的更新，这些更新能够揭示新的失败模式，增加测试难度同时保持测试目标对齐，并提高模型可分离性。

### 结论

ArenaBencher框架为基础模型的快速进展提供了一条可扩展的持续发展基准测试的路径，有助于确保基准测试的有效性和准确性，从而更准确地评估和比较大语言模型的能力。

### 翻译

基准测试对于衡量大语言模型的能力和指导模型开发至关重要，然而从预训练语料库中的广泛数据泄露损害了其有效性。模型可以匹配记忆内容而非展示真正的泛化能力，这 inflated 了分数，扭曲了跨模型比较，并错误地代表了进展。我们引入ArenaBencher，一个与模型无关的自动基准测试演进框架，它在保持可比性的同时更新测试用例。给定一个现有基准测试和一组要评估的多样化模型，ArenaBencher推断每个测试用例的核心能力，生成保留原始目标的候选问答对，使用LLM作为法官验证正确性和意图，并从多个模型聚合反馈以选择暴露共同弱点的候选。该过程使用上下文演示迭代运行，这些演示引导生成更具挑战性和诊断性的案例。我们将ArenaBencher应用于数学问题解决、常识推理和安全领域，并表明它产生了经过验证的、多样化的和公平的更新，这些更新揭示了新的失败模式，增加了难度同时保持测试目标对齐，并提高了模型可分离性。该框架为基础模型的快速进展提供了一条可扩展的持续发展基准测试的路径。


### 论文摘要

Benchmarks are central to measuring the capabilities of large language models and guiding model development, yet widespread data leakage from pretraining corpora undermines their validity. Models can match memorized content rather than demonstrate true generalization, which inflates scores, distorts cross-model comparisons, and misrepresents progress. We introduce ArenaBencher, a model-agnostic framework for automatic benchmark evolution that updates test cases while preserving comparability. Given an existing benchmark and a diverse pool of models to be evaluated, ArenaBencher infers the core ability of each test case, generates candidate question-answer pairs that preserve the original objective, verifies correctness and intent with an LLM as a judge, and aggregates feedback from multiple models to select candidates that expose shared weaknesses. The process runs iteratively with in-context demonstrations that steer generation toward more challenging and diagnostic cases. We apply ArenaBencher to math problem solving, commonsense reasoning, and safety domains and show that it produces verified, diverse, and fair updates that uncover new failure modes, increase difficulty while preserving test objective alignment, and improve model separability. The framework provides a scalable path to continuously evolve benchmarks in step with the rapid progress of foundation models.

---

## 43. ARTDECO: Towards Efficient and High-Fidelity On-the-Fly 3D Reconstruction with Structured Scene Representation

**论文链接:** [http://arxiv.org/abs/2510.08551v1](http://arxiv.org/abs/2510.08551v1)

**作者:** Guanghao Li, Kerui Ren, Linning Xu, Zhewen Zheng, Changjian Jiang, Xin Gao, Bo Dai, Jian Pu, Mulin Yu, Jiangmiao Pang

**发布时间:** 2025-10-09

### GPT解析

### 总结

ARTDECO是一种统一框架，结合了前馈模型的效率和基于SLAM管道的可靠性，用于从单目图像序列进行实时3D重建。

### 背景

从单目图像序列进行实时3D重建是计算机视觉中的一个长期挑战，对real-to-sim、AR/VR和机器人等应用至关重要。现有方法面临权衡：针对每个场景的优化计算成本高，而前馈基础模型实时性好但准确性和鲁棒性不足。

### 目的

开发一个统一框架，结合前馈模型的效率和基于SLAM管道的可靠性，实现高质量、高效率的实时3D重建。

### 方法

提出ARTDECO框架，使用3D基础模型进行姿态估计和点预测，结合高斯解码器将多尺度特征转换为结构化3D高斯，并设计层次化高斯表示和基于细节层次的渲染策略，以提高渲染保真度同时减少冗余。

### 主要发现

在八个多样化的室内和室外基准测试中，ARTDECO提供了与SLAM相当的交互性能、与前馈系统相似的鲁棒性，以及接近每个场景优化质量的重建质量。

### 结论

ARTDECO为实时数字化真实世界环境提供了实用途径，既能实现精确几何又能保持高视觉保真度。

### 翻译

从单目图像序列进行实时3D重建是计算机视觉中的一个长期挑战，对real-to-sim、AR/VR和机器人等应用至关重要。现有方法面临一个重大权衡：针对每个场景的优化能产生高保真度，但计算成本高，而前馈基础模型可以实现实时推理，但在准确性和鲁棒性方面存在问题。在这项工作中，我们提出了ARTDECO，这是一个统一框架，结合了前馈模型的效率和基于SLAM管道的可靠性。ARTDECO使用3D基础模型进行姿态估计和点预测，结合高斯解码器将多尺度特征转换为结构化3D高斯。为了在大规模下保持保真度和效率，我们设计了一种层次化高斯表示和基于细节层次的渲染策略，这提高了渲染保真度同时减少了冗余。在八个多样化的室内和室外基准测试中，ARTDECO提供了与SLAM相当的交互性能、与前馈系统相似的鲁棒性，以及接近每个场景优化质量的重建质量，为实时数字化具有精确几何和高视觉保真度的真实世界环境提供了实用途径。在我们的项目页面上探索更多演示：https://city-super.github.io/artdeco/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单目图像序列进行即时3D重建的挑战。这个问题在现实中非常重要，因为它关系到AR/VR应用、机器人导航、实时模拟和数字孪生等技术领域，这些领域都需要高效且高质量的3D场景重建能力。现有方法要么计算成本高但质量好，要么速度快但质量差，难以兼顾效率和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的优缺点：基于场景的优化方法质量高但计算昂贵，前馈模型速度快但准确性和鲁棒性不足。他们借鉴了SLAM管道的可靠性和前馈模型的效率，结合两者优势。具体来说，他们使用了3D基础模型进行姿态估计和点预测，借鉴了高斯溅射技术，并设计了分层高斯表示和基于细节级别的渲染策略，这些都是在现有工作基础上进行的创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'ARTDECO的核心思想是将前馈模型的效率与基于SLAM的管道的可靠性相结合，实现高效且高保真的即时3D重建。整体流程分为三个模块：1）前端模块：估计相对姿态并将帧分类为普通帧、映射帧或关键帧；2）后端模块：通过回环检测和全局束调整优化关键帧姿态；3）映射模块：从帧初始化3D高斯，并增量优化它们。特别地，系统使用分层半隐式高斯结构和基于细节级别的densification策略，平衡了重建质量和渲染效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': 'ARTDECO的关键创新点包括：1）统一框架：将定位、重建和渲染整合到一个管道中，能在各种环境中稳健运行；2）模块化组件：将前馈基础模型用于姿态估计、回环检测和密集点预测；3）分层半隐式高斯表示：具有基于细节级别的densification策略，实现保真度和效率的平衡；4）实验验证：在多样化的室内和室外基准测试中，实现了SLAM级别的效率、前馈鲁棒性和接近场景优化质量的结果。相比之前的工作，ARTDECO不需要在每场景优化和前馈模型之间做权衡，而是结合了两者的优点。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ARTDECO通过结合前馈模型的效率和基于SLAM的管道的可靠性，实现了从单目图像序列进行高效、高保真即时3D重建的创新框架。'}


### 论文摘要

On-the-fly 3D reconstruction from monocular image sequences is a long-standing challenge in computer vision, critical for applications such as real-to-sim, AR/VR, and robotics. Existing methods face a major tradeoff: per-scene optimization yields high fidelity but is computationally expensive, whereas feed-forward foundation models enable real-time inference but struggle with accuracy and robustness. In this work, we propose ARTDECO, a unified framework that combines the efficiency of feed-forward models with the reliability of SLAM-based pipelines. ARTDECO uses 3D foundation models for pose estimation and point prediction, coupled with a Gaussian decoder that transforms multi-scale features into structured 3D Gaussians. To sustain both fidelity and efficiency at scale, we design a hierarchical Gaussian representation with a LoD-aware rendering strategy, which improves rendering fidelity while reducing redundancy. Experiments on eight diverse indoor and outdoor benchmarks show that ARTDECO delivers interactive performance comparable to SLAM, robustness similar to feed-forward systems, and reconstruction quality close to per-scene optimization, providing a practical path toward on-the-fly digitization of real-world environments with both accurate geometry and high visual fidelity. Explore more demos on our project page: https://city-super.github.io/artdeco/.

---

## 44. Synthetic Series-Symbol Data Generation for Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.08445v1](http://arxiv.org/abs/2510.08445v1)

**作者:** Wenxuan Wang, Kai Wu, Yujian Betterest Li, Dan Wang, Xiaoyu Zhang

**发布时间:** 2025-10-09

**备注:** 63 pages, NeurIPS 2025 accepted

### GPT解析

### 总结

本研究提出了一种基于复杂动态系统理论的时间序列分析基础模型SymTime，通过系列-符号数据生成机制克服数据稀缺和不平衡问题，在五个主要TSA任务上展现出与真实数据预训练模型相媲美的性能。

### 背景

时间序列分析基础模型已引起广泛关注，但训练数据稀缺和不平衡问题仍然阻碍其发展。

### 目的

设计一种能够无限制创建高质量时间序列数据及其对应符号表达式的数据生成机制，并开发利用这些数据对增强时间序列表示的基础模型。

### 方法

受复杂动态系统理论启发，设计了系列-符号数据生成机制，并开发了名为SymTime的预训练基础模型，利用具有强相关性的系列-符号数据对增强时间序列表示能力。

### 主要发现

SymTime在五个主要时间序列分析任务上展现出有竞争力的性能，当使用下游任务微调时，与在真实数据集上预训练的基础模型相媲美。

### 结论

系列-符号数据生成和预训练机制在克服数据稀缺性和提高任务性能方面具有巨大潜力。

### 翻译

时间序列分析的基础模型已引起广泛关注。然而，训练数据稀缺和不平衡等挑战继续阻碍其发展。受复杂动态系统理论的启发，我们设计了一系列-符号数据生成机制，能够无限制地创建高质量的时间序列数据及其对应的符号表达式。为了利用具有强相关性的系列-符号数据对，我们开发了SymTime，这是一个使用符号信息增强时间序列表示的预训练基础模型。当使用下游任务微调时，SymTime在五个主要的TSA任务上展现出有竞争力的性能，与在真实数据集上预训练的基础模型相媲美。这种方法强调了系列-符号数据生成和预训练机制在克服数据稀缺性和提高任务性能方面的潜力。代码可在https://github.com/wwhenxuan/SymTime获取。


### 论文摘要

Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as training data scarcity and imbalance continue to hinder their development. Inspired by complex dynamic system theories, we design a series-symbol data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic expressions. To leverage series-symbol data pairs with strong correlations, we develop \texttt{SymTime}, a pre-trained foundation model for enhancing time series representation using symbolic information. \texttt{SymTime} demonstrates competitive performance across five major TSA tasks when fine-tunes with downstream tasks, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of series-symbol data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. The code is available at https://github.com/wwhenxuan/SymTime.

---

## 45. FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

**论文链接:** [http://arxiv.org/abs/2510.08396v1](http://arxiv.org/abs/2510.08396v1)

**作者:** Heming Zou, Yunliang Zang, Wutong Xu, Yao Zhu, Xiangyang Ji

**发布时间:** 2025-10-09

**备注:** NeurIPS 2025 accepted paper

### GPT解析

### 总结

FlyLoRA是一种基于果蝇嗅觉电路启发的隐式MoE-based LoRA变体，通过秩级专家激活和隐式路由器设计解决了LoRA的参数干扰问题，并在多任务场景下有效缓解了任务间干扰。

### 背景

Low-Rank Adaptation (LoRA)是一种广泛使用的参数高效微调方法，但存在参数干扰问题导致性能不理想。虽然Mixture-of-Experts (MoE)基础的LoRA变体在缓解单任务指令调优中的任务内相关性方面有前景，但引入了额外的路由器参数，并在多任务模型合并中仍然无效。

### 目的

解决LoRA的参数干扰问题，缓解任务内相关性和任务间干扰，同时提高计算效率。

### 方法

提出FlyLoRA，包含两个关键组件：(1)在上投影矩阵中引入秩级专家激活；(2)设计隐式路由器统一专家路由和下投影，用冻结的稀疏随机投影矩阵替代传统的密集可训练版本。这一设计消除了对显式路由器的需求，并利用随机矩阵的正交特性内在缓解任务间干扰。

### 主要发现

FlyLoRA解决了任务内解相关性和计算效率之间的权衡问题。在四个领域(通用知识理解、科学问答、数学推理和代码生成)的广泛实验中，相对于现有方法实现了持续的性能改进。

### 结论

FlyLoRA不仅带来了经验上的改进，还展示了生物结构如何能够启发AI技术的创新。

### 翻译

低秩适应(LoRA)是基础模型的广泛使用的参数高效微调方法，但它受到参数干扰的影响，导致次优性能。尽管基于专家混合(MoE)的LoRA变体在缓解单任务指令调优中的任务内相关性方面显示出前景，但它们引入了额外的路由器参数，并且在出现任务间干扰的多任务模型合并中仍然无效。受果蝇嗅觉电路的启发，我们提出了FlyLoRA，这是一种隐式MoE-based LoRA变体，它引入：(1)在上投影矩阵中进行秩级专家激活，以及(2)一个隐式路由器，统一专家路由和下投影，其中冻结的稀疏随机投影矩阵替代了传统的密集可训练版本。这种设计通过消除对显式路由器的需求，解决了任务内解相关性和计算效率之间的权衡，同时由于随机矩阵的正交特性，内在地缓解了任务间干扰。在四个领域(通用知识理解、科学问答、数学推理和代码生成)的广泛实验中，展示了相对于现有方法的持续性能改进。除了经验上的收益，FlyLoRA还展示了生物结构如何能够启发AI技术的创新。代码可在https://github.com/gfyddha/FlyLoRA获取。


### 论文摘要

Low-Rank Adaptation (LoRA) is a widely used parameter-efficient fine-tuning method for foundation models, but it suffers from parameter interference, resulting in suboptimal performance. Although Mixture-of-Experts (MoE)-based LoRA variants show promise in mitigating intra-task correlations in single-task instruction tuning, they introduce additional router parameters and remain ineffective in multi-task model merging where inter-task interference arises. Inspired by the fly olfactory circuit, we propose FlyLoRA, an implicit MoE-based LoRA variant that introduces: (1) rank-wise expert activation in the up-projection matrix, and (2) an implicit router that unifies expert routing and down-projection, where a frozen sparse random projection matrix replaces the traditional dense trainable version. This design resolves the trade-off between intra-task decorrelation and computational efficiency by eliminating the need for an explicit router, while inherently mitigating inter-task interference due to the orthogonality property of random matrices. Extensive experiments across four domains -- general knowledge understanding, scientific question answering, mathematical reasoning, and code generation -- demonstrate consistent performance improvements over existing methods. Beyond empirical gains, FlyLoRA highlights how biological structures can inspire innovations in AI technologies. Code is available at https://github.com/gfyddha/FlyLoRA.

---

## 46. 论文ID: 2510.08177v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.08177v1.json'

---

## 47. UniMMVSR: A Unified Multi-Modal Framework for Cascaded Video Super-Resolution

**论文链接:** [http://arxiv.org/abs/2510.08143v1](http://arxiv.org/abs/2510.08143v1)

**作者:** Shian Du, Menghan Xia, Chang Liu, Quande Liu, Xintao Wang, Pengfei Wan, Xiangyang Ji

**发布时间:** 2025-10-09

### GPT解析

### 总结

UniMMVSR是一种创新的视频超分辨率框架，能够整合文本、图像和视频等多种模态条件，显著提升了生成视频的质量和与多模态条件的符合度，并实现了4K视频的多模态引导生成。

### 背景

级联视频超分辨率技术是一种有前途的技术，可以解耦使用大型基础模型生成高分辨率视频的计算负担。但现有研究主要局限于文本到视频任务，无法利用文本以外的生成条件，这对于确保多模态视频生成的保真度至关重要。

### 目的

提出UniMMVSR，第一个统一生成视频超分辨率框架，可以整合混合模态条件，包括文本、图像和视频。

### 方法

在潜在视频扩散模型中，对条件注入策略、训练方案和数据混合技术进行了全面探索。设计不同的数据构建和条件利用方法，使模型能够精确利用所有条件类型，考虑到它们与目标视频的不同相关性。

### 主要发现

实验表明，UniMMVSR显著优于现有方法，生成的视频具有更优质的细节和更高的多模态条件符合度。验证了将UniMMVSR与基础模型结合以实现4K视频的多模态引导生成的可行性。

### 结论

UniMMVSR代表了视频超分辨率技术的重要进步，通过整合多模态条件，实现了更高质量的视频生成，并能够生成4K视频。

### 翻译

级联视频超分辨率已成为一种有前途的技术，用于解耦使用大型基础模型生成高分辨率视频的计算负担。然而，现有研究大多局限于文本到视频任务，无法利用文本以外的生成条件，而这些条件对于确保多模态视频生成的保真度至关重要。我们通过提出UniMMVSR来解决这个问题，这是第一个统一生成视频超分辨率框架，可以整合混合模态条件，包括文本、图像和视频。我们在潜在视频扩散模型中对条件注入策略、训练方案和数据混合技术进行了全面探索。一个关键挑战是设计不同的数据构建和条件利用方法，使模型能够精确利用所有条件类型，考虑到它们与目标视频的不同相关性。我们的实验证明，UniMMVSR显著优于现有方法，生成的视频具有更优质的细节和更高的多模态条件符合度。我们还验证了将UniMMVSR与基础模型结合以实现多模态引导的4K视频生成的可行性，这是现有技术无法实现的。


### 论文摘要

Cascaded video super-resolution has emerged as a promising technique for decoupling the computational burden associated with generating high-resolution videos using large foundation models. Existing studies, however, are largely confined to text-to-video tasks and fail to leverage additional generative conditions beyond text, which are crucial for ensuring fidelity in multi-modal video generation. We address this limitation by presenting UniMMVSR, the first unified generative video super-resolution framework to incorporate hybrid-modal conditions, including text, images, and videos. We conduct a comprehensive exploration of condition injection strategies, training schemes, and data mixture techniques within a latent video diffusion model. A key challenge was designing distinct data construction and condition utilization methods to enable the model to precisely utilize all condition types, given their varied correlations with the target video. Our experiments demonstrate that UniMMVSR significantly outperforms existing methods, producing videos with superior detail and a higher degree of conformity to multi-modal conditions. We also validate the feasibility of combining UniMMVSR with a base model to achieve multi-modal guided generation of 4K video, a feat previously unattainable with existing techniques.

---

## 48. Mitigating Subject Dependency in EEG Decoding with Subject-Specific Low-Rank Adapters

**论文链接:** [http://arxiv.org/abs/2510.08059v1](http://arxiv.org/abs/2510.08059v1)

**作者:** Timon Klein, Piotr Minakowski, Sebastian Sager

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文提出了一种名为Subject-Conditioned Layer的自适应层，用于解决脑电图解码中个体间分布差异对基础模型开发的阻碍问题。该层通过将权重分解为共享通用组件和受试者特定修正，使模型能够同时保持通用性和个性化适应性，实验证明其性能优于传统方法。

### 背景

个体间分布差异是开发脑电图解码基础模型的重要障碍，不同受试者的脑电信号存在显著差异，这限制了模型的泛化能力。

### 目的

开发一种能够处理个体间分布差异的自适应层，使基础模型能够同时具备通用知识和个性化适应能力，提高跨受试者脑电解码的性能。

### 方法

提出Subject-Conditioned Layer，一种可替代标准线性或卷积层的自适应层。该层通过将权重分解为共享的、不受试者影响的通用组件和轻量级的、低秩的、针对每个受试者的独特修正，实现通用知识与个性化适应的明确分离。

### 主要发现

采用Subject-Conditioned Layer的模型在性能上超过了仅使用共享权重的模型(不受试者影响的模型)和单独训练的受试者特定模型的平均值，证明了该方法的有效性。

### 结论

Subject-Conditioned Layer为构建有效的跨受试者脑电图基础模型提供了一种实用且可扩展的解决方案，能够使模型在保持通用性的同时具备个性化适应能力。

### 翻译

受试者特定的分布差异是开发脑电图解码基础模型的重要障碍。为解决这一问题，我们提出了受试者条件化层，这是一种自适应层，可作为任何神经网络架构中标准线性或卷积层的替代品。我们的层通过将权重分解为共享的、不受试者影响的组件和轻量级的、低秩的、每个受试者独特的修正，来捕获受试者特定的变异性。这种将通用知识与个性化适应明确分离的方式，使现有模型能够对受试者差异具有鲁棒性。经验上，配备我们层的模型在性能上优于仅使用共享权重的模型(不受试者影响的模型)和单独训练的受试者特定模型的平均值。因此，受试者条件化层为构建有效的跨受试者脑电图基础模型提供了一种实用且可扩展的途径。


### 论文摘要

Subject-specific distribution shifts represent an important obstacle to the development of foundation models for EEG decoding. To address this, we propose Subject-Conditioned Layer,, an adaptive layer designed as a drop-in replacement for standard linear or convolutional layers in any neural network architecture. Our layer captures subject-specific variability by decomposing its weights into a shared, subject-invariant component and a lightweight, low-rank correction unique to each subject. This explicit separation of general knowledge from personalized adaptation allows existing models to become robust to subject shifts. Empirically, models equipped with our layer outperform both a shared-weight-only model (subject-agnostic model) and the average of individually trained subject-specific models. Consequently, the Subject-Conditioned Layer, offers a practical and scalable path towards building effective cross-subject foundation models for EEG.

---

## 49. TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

**论文链接:** [http://arxiv.org/abs/2510.07940v1](http://arxiv.org/abs/2510.07940v1)

**作者:** Leigang Qu, Ziyang Wang, Na Zheng, Wenjie Wang, Liqiang Nie, Tat-Seng Chua

**发布时间:** 2025-10-09

**备注:** Project page: https://ttom-t2v.github.io/

### GPT解析

### 总结

本文提出了Test-Time Optimization and Memorization (TTOM)框架，解决了Video Foundation Models (VFMs)在组合场景中的局限性，通过在推理过程中优化参数实现更好的文本-图像对齐。

### 背景

Video Foundation Models (VFMs)在视觉生成方面表现出色，但在处理组合场景（如运动、计数和空间关系）时存在困难。

### 目的

开发一个无需训练的框架，在推理过程中将VFM输出与时空布局对齐，提高文本-图像对齐效果，并支持组合视频生成。

### 方法

提出Test-Time Optimization and Memorization (TTOM)框架，集成并优化由布局-注意力目标引导的新参数；在流式设置下制定视频生成；使用参数化内存机制维护历史优化上下文，支持插入、读取、更新和删除等操作。

### 主要发现

TTOM能够解耦组合世界知识，显示出强大的可转移性和泛化能力。

### 结论

TTOM被证明是一个有效、实用、可扩展且高效的框架，可以实现跨模态对齐和即时的组合视频生成。

### 翻译

视频基础模型(VFMs)展现出卓越的视觉生成性能，但在组合场景（如运动、计数和空间关系）中表现不佳。在本工作中，我们提出了测试时优化和记忆(TTOM)，这是一个无需训练的框架，在推理过程中将VFM输出与时空布局对齐，以实现更好的文本-图像对齐。与现有工作中直接干预潜在表示或每个样本的注意力机制不同，我们集成并优化由通用布局-注意力目标引导的新参数。此外，我们在流式设置下制定视频生成，并使用参数化内存机制维护历史优化上下文，支持插入、读取、更新和删除等灵活操作。值得注意的是，我们发现TTOM能够解耦组合世界知识，显示出强大的可转移性和泛化能力。在T2V-CompBench和Vbench基准测试上的实验结果确立了TTOM作为一个有效、实用、可扩展且高效的框架，用于实现组合视频生成的跨模态对齐。


### 论文摘要

Video Foundation Models (VFMs) exhibit remarkable visual generation performance, but struggle in compositional scenarios (e.g., motion, numeracy, and spatial relation). In this work, we introduce Test-Time Optimization and Memorization (TTOM), a training-free framework that aligns VFM outputs with spatiotemporal layouts during inference for better text-image alignment. Rather than direct intervention to latents or attention per-sample in existing work, we integrate and optimize new parameters guided by a general layout-attention objective. Furthermore, we formulate video generation within a streaming setting, and maintain historical optimization contexts with a parametric memory mechanism that supports flexible operations, such as insert, read, update, and delete. Notably, we found that TTOM disentangles compositional world knowledge, showing powerful transferability and generalization. Experimental results on the T2V-CompBench and Vbench benchmarks establish TTOM as an effective, practical, scalable, and efficient framework to achieve cross-modal alignment for compositional video generation on the fly.

---

## 50. AlignGS: Aligning Geometry and Semantics for Robust Indoor Reconstruction from Sparse Views

**论文链接:** [http://arxiv.org/abs/2510.07839v1](http://arxiv.org/abs/2510.07839v1)

**作者:** Yijie Gao, Houqiang Zhong, Tianchi Zhu, Zhengxue Cheng, Qiang Hu, Li Song

**发布时间:** 2025-10-09

### GPT解析

### 总结

AlignGS是一种新颖的框架，通过协同优化几何和语义，从稀疏视图中生成更连贯、更完整的3D模型。

### 背景

室内场景语义丰富的3D模型需求快速增长，由增强现实、虚拟现实和机器人应用驱动，但从稀疏视图创建这些模型具有几何歧义性。

### 目的

将语义理解作为主动引导力量，用于鲁棒的稀疏视图3D重建。

### 方法

AlignGS框架从2D基础模型中提取丰富先验，通过深度一致性和多方面法线正则化等语义到几何的引导机制直接正则化3D表示。

### 主要发现

在新视图合成方面取得最先进结果，产生具有更高几何精度的重建，证明利用语义先验作为几何正则化器可从有限输入视图产生更连贯、更完整的3D模型。

### 结论

语义理解应作为主动引导力量，而非被动特征，用于3D重建过程。

### 翻译

对室内场景语义丰富的3D模型需求正快速增长，这由增强现实、虚拟现实和机器人应用所驱动。然而，从稀疏视图创建这些模型仍然是一个挑战，因为存在几何歧义性。现有方法通常将语义作为被动特征，应用于已经形成且可能有缺陷的几何体上。我们认为，对于鲁棒的稀疏视图重建，语义理解应该成为主动的引导力量。本文介绍了AlignGS，这是一个新颖的框架，通过几何和语义的协同端到端优化实现了这一愿景。我们的方法从2D基础模型中提取丰富的先验知识，并通过一套新颖的语义到几何的引导机制（包括深度一致性和多方面法线正则化）直接正则化3D表示。在标准基准上的广泛评估表明，我们的方法在新视图合成方面取得了最先进的结果，并产生了具有更高几何精度的重建结果。这些结果验证了，利用语义先验作为几何正则化器可以从有限的输入视图中产生更连贯、更完整的3D模型。我们的代码可在https://github.com/MediaX-SJTU/AlignGS获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从稀疏视图进行鲁棒的室内场景3D重建问题。这个问题很重要，因为室内场景通常因复杂布局和频繁遮挡导致可用视角稀疏，而现有的3D重建方法在这种情况下难以产生准确完整的几何结构。同时，对语义丰富的3D室内模型的需求正在快速增长，由增强现实、虚拟现实和机器人等应用驱动，从稀疏视图创建这些模型对现实应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认为传统方法将语义作为被动特征而非主动引导几何优化的力量，因此提出语义理解应该成为主动引导力量。作者借鉴了现有的2D基础模型(如DINOv2和Mask2Former)提取语义先验，同时采用3D高斯泼溅作为基础表示方法。方法设计包括使用VGGT进行无SfM初始化，为每个高斯添加语义向量，设计语义到几何的引导机制，以及采用双重监督策略转移2D语义知识到3D表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将语义理解作为主动引导几何优化的力量，通过2D先验知识直接约束3D表示，解决稀疏视图下的几何歧义。整体流程分为四步：1)初始化阶段：使用VGGT生成初始点云和相机姿态，初始化3D高斯模型；2)语义知识转移：通过双重监督策略将2D语义知识转移到3D表示；3)几何优化：应用深度一致性和多方面法线一致性约束；4)联合优化：同时优化所有几何和语义属性，协同改进场景的几何结构和语义理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)端到端稀疏视图室内重建框架，利用2D先验直接正则化3D几何；2)语义引导的几何约束机制，包括深度一致性和边界感知法线一致性；3)鲁棒的无SfM初始化方法。相比之前工作，AlignGS将语义从被动特征转变为主动引导力量，采用端到端联合优化而非两阶段方法，使用无SfM初始化提高稀疏视图下的鲁棒性，并引入基于语义的高级先验作为几何正则化器。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AlignGS通过将语义理解作为主动引导几何优化的力量，实现了从稀疏视图进行鲁棒室内重建，显著提高了3D重建的几何准确性和语义一致性。'}


### 论文摘要

The demand for semantically rich 3D models of indoor scenes is rapidly growing, driven by applications in augmented reality, virtual reality, and robotics. However, creating them from sparse views remains a challenge due to geometric ambiguity. Existing methods often treat semantics as a passive feature painted on an already-formed, and potentially flawed, geometry. We posit that for robust sparse-view reconstruction, semantic understanding instead be an active, guiding force. This paper introduces AlignGS, a novel framework that actualizes this vision by pioneering a synergistic, end-to-end optimization of geometry and semantics. Our method distills rich priors from 2D foundation models and uses them to directly regularize the 3D representation through a set of novel semantic-to-geometry guidance mechanisms, including depth consistency and multi-faceted normal regularization. Extensive evaluations on standard benchmarks demonstrate that our approach achieves state-of-the-art results in novel view synthesis and produces reconstructions with superior geometric accuracy. The results validate that leveraging semantic priors as a geometric regularizer leads to more coherent and complete 3D models from limited input views. Our code is avaliable at https://github.com/MediaX-SJTU/AlignGS .

---

## 51. FedBook: A Unified Federated Graph Foundation Codebook with Intra-domain and Inter-domain Knowledge Modeling

**论文链接:** [http://arxiv.org/abs/2510.07755v1](http://arxiv.org/abs/2510.07755v1)

**作者:** Zhengyu Wu, Yinlin Zhu, Xunkai Li, Ziang Qiu, Rong-Hua Li, Guoren Wang, Chenghu Zhou

**发布时间:** 2025-10-09

**备注:** Under Review

### GPT解析

### 总结

本文提出了FedBook，一种统一的联邦图基础码本，用于在联邦学习环境下构建强大的图基础模型。该方法通过两阶段过程（领域内协作和跨领域集成）系统性地聚合客户端的局部码本，既增强了领域内一致性，又保留了跨领域多样性。

### 背景

Foundation models在语言和视觉领域表现出卓越的跨领域泛化能力，启发了图基础模型(GFMs)的发展。然而，现有GFMs通常假设可集中访问多领域图，这在实际应用中因隐私和机构限制而不可行。联邦图基础模型(FedGFMs)虽解决了这一问题，但有效构建全局码本仍面临挑战。

### 目的

开发一种方法，能够在联邦学习环境下构建强大的全局图基础码本，实现领域内一致性同时保持跨领域多样性，从而突破隐私和机构限制下的多领域图学习挑战。

### 方法

提出FedBook，一种统一的联邦图基础码本，在服务器端联邦预训练期间系统性地聚合客户端局部码本。采用两阶段过程：(1)领域内协作：通过参考客户端间语义可靠的高频令牌改进低频令牌，增强领域特定一致性；(2)跨领域集成：根据码本语义独特性对客户端贡献加权，保留跨领域多样性。

### 主要发现

在8个跨多个领域和任务的基准测试上，FedBook持续优于21个基线方法，包括孤立监督学习、FL/FGL、集中式GFMs的联邦适配以及FedGFM技术，证明了其在联邦图学习中的优越性能。

### 结论

FedBook有效地解决了联邦环境下构建强大图基础模型的挑战，通过系统性地聚合客户端码本，在保持跨领域多样性的同时增强了领域内一致性，为隐私保护下的多领域图学习提供了新思路。

### 翻译

基础模型在语言和视觉领域展示了卓越的跨领域泛化能力，启发了图基础模型(GFMs)的发展。然而，现有GFMs通常假设可以集中访问多领域图，这往往因隐私和机构限制而不可行。联邦图基础模型(FedGFMs)解决了这一限制，但其有效性根本上取决于构建一个强大的全局码本，该码本通过整合每个领域内相互强化的语义实现领域内一致性，同时通过保留跨领域的异构知识维持领域间多样性。为此，我们提出FedBook，一个统一的联邦图基础码本，在服务器端联邦预训练期间系统性地聚合客户端的局部码本。FedBook遵循两阶段过程：(1)领域内协作，通过参考客户端间语义上更可靠的高频令牌来改进低频令牌，增强领域特定一致性；(2)跨领域集成，在全局GFM聚合期间，根据码本的语义独特性对客户端贡献进行加权，从而保留跨领域多样性。在8个跨多个领域和任务的基准测试上的广泛实验表明，FedBook持续优于21个基线，包括孤立的监督学习、FL/FGL、集中式GFMs的联邦适配和FedGFM技术。


### 论文摘要

Foundation models have shown remarkable cross-domain generalization in language and vision, inspiring the development of graph foundation models (GFMs). However, existing GFMs typically assume centralized access to multi-domain graphs, which is often infeasible due to privacy and institutional constraints. Federated Graph Foundation Models (FedGFMs) address this limitation, but their effectiveness fundamentally hinges on constructing a robust global codebook that achieves intra-domain coherence by consolidating mutually reinforcing semantics within each domain, while also maintaining inter-domain diversity by retaining heterogeneous knowledge across domains. To this end, we propose FedBook, a unified federated graph foundation codebook that systematically aggregates clients' local codebooks during server-side federated pre-training. FedBook follows a two-phase process: (1) Intra-domain Collaboration, where low-frequency tokens are refined by referencing more semantically reliable high-frequency tokens across clients to enhance domain-specific coherence; and (2) Inter-domain Integration, where client contributions are weighted by the semantic distinctiveness of their codebooks during the aggregation of the global GFM, thereby preserving cross-domain diversity. Extensive experiments on 8 benchmarks across multiple domains and tasks demonstrate that FedBook consistently outperforms 21 baselines, including isolated supervised learning, FL/FGL, federated adaptations of centralized GFMs, and FedGFM techniques.

---

## 52. Multi-modal Foundation Model for Cosmological Simulation Data

**论文链接:** [http://arxiv.org/abs/2510.07684v1](http://arxiv.org/abs/2510.07684v1)

**作者:** Bin Xia, Nesar Ramachandra, Azton I. Wells, Salman Habib, John Wise

**发布时间:** 2025-10-09

### GPT解析

### 总结

这篇论文介绍了一个用于天体物理学星系数据的多模态基础模型，能够在模拟和观测数据之间进行映射。模型使用仅编码器transformer架构，支持从部分输入查询星系属性，并在红移估计和恒星质量推断方面表现出显著改进。

### 背景

天体物理学研究中需要有效处理来自模拟和观测的星系数据，并建立两者之间的联系。现有方法可能难以有效处理多模态数据并支持多种任务。

### 目的

开发一个能够处理天体物理学星系数据的多模态基础模型，实现模拟和观测星系特征之间的映射，支持多任务训练，并能够从部分输入中查询任意星系属性。

### 方法

使用仅编码器transformer架构，能够处理标量量（如红移、星系质量）和矢量（如恒星形成历史、光谱）数据。采用动态掩码策略，从部分输入中查询星系属性。使用来自千兆秒差距级宇宙模拟的185,000个模拟星系进行训练。

### 主要发现

当结合LSST和SPHEREx测光数据时，红移估计比仅使用LSST测光数据提高了50%；当结合晚期恒星形成历史与LSST测光数据时，恒星质量推断比结合早期恒星形成历史与LSST测光数据提高了63%。模型在多模态任务上表现出强大的泛化能力。

### 结论

该方法为连接模拟和观测提供了统一框架，为未来集成更高维度和结构化数据（如图像、合并树和3D场）奠定了基础，推动了可推广天体物理学基础模型的发展。

### 翻译

我们提出了一种用于天体物理学星系数据的多模态基础模型，旨在映射基于模拟和观测的星系特征。我们的仅编码器transformer能够灵活地吸收标量量（如红移、星系质量）和矢量（如恒星形成历史、光谱），支持包括模态内重建和跨模态预测的多任务训练。通过动态掩码策略，模型可以从部分输入中查询任意星系属性——包括根据红移和质量预测光谱，或根据宽带星等估计测光红移——同时恢复模态内的缺失部分。模型在来自千兆秒差距级宇宙模拟的185,000个模拟星系上进行了训练，当结合LSST和SPHEREx测光数据时，红移估计比仅使用LSST测光数据提高了50%，当结合晚期恒星形成历史与LSST测光数据时，恒星质量推断比结合早期恒星形成历史与LSST测光数据提高了63%。该模型在多模态任务上表现出强大的泛化能力，为未来集成更高维度和结构化数据（如图像、合并树和3D场）奠定了基础。这种方法为连接模拟和观测提供了统一框架，推动了可推广天体物理学基础模型的发展。


### 论文摘要

We present a multi-modal foundation model for astrophysical galaxy data, designed to map between simulation- and observation-based galactic features. Our encoder-only transformer flexibly ingests scalar quantities (e.g., redshifts, galaxy masses) and vectors (e.g., star formation histories, spectra), supporting multi-task training that includes within-modality reconstruction and cross-modality prediction. With a dynamic masking strategy, the model can query arbitrary galaxy properties from partial inputs -- including predicting spectra from redshift and mass, or estimating photometric redshifts from broadband magnitudes -- while also recovering missing segments within a modality. Trained on 185,000 simulated galaxies from a gigaparsec-scale Cosmology simulation, the model yields a 50% improvement in redshift estimation when combining LSST and SPHEREx photometry over LSST photometry alone, and a 63% improvement in stellar mass inference when combining late-time SFH with LSST photometry over early-time SFH with LSST photometry. The model demonstrates strong generalization across multi-modal tasks and lays the groundwork for future integration of higher-dimensional and structured data such as images, merger trees, and 3D fields. This approach provides a unified framework for connecting simulations and observations, advancing the development of generalizable astrophysical foundation models.

---

