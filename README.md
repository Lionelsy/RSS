# 今日论文推荐 - 2025-11-19

共 169 篇论文

---

## 1. Wave-Former: Through-Occlusion 3D Reconstruction via Wireless Shape Completion

**论文链接:** [http://arxiv.org/abs/2511.14152v1](http://arxiv.org/abs/2511.14152v1)

**作者:** Laura Dodds, Maisy Lam, Waleed Akbar, Yibo Cheng, Fadel Adib

**发布时间:** 2025-11-18

### GPT解析

### 总结

Wave-Former是一种新颖的方法，能够对完全遮挡的、多样化的日常物体进行高精度的三维形状重建。这种方法利用毫米波无线信号，可以穿透常见遮挡物并从隐藏物体上反射回来。

### 背景

现有的毫米波重建方法存在覆盖范围有限和高噪声的问题，难以准确重建被完全遮挡物体的三维形状。

### 目的

开发一种能够高精度重建完全遮挡物体三维形状的方法，以拓展在机器人技术、增强现实和物流等领域的应用。

### 方法

Wave-Former采用一个三阶段管道，将原始无线信号与基于视觉的形状完成技术相结合，并考虑毫米波信号的物理特性。该方法提出候选几何表面，使用专为毫米波信号设计的基于Transformer的形状完成模型，最后执行熵引导的表面选择。这种方法完全使用合成点云进行训练，但能够很好地推广到真实世界数据。

### 主要发现

与最先进的基线方法相比，Wave-Former将召回率从54%提高到72%，同时保持了85%的高精度。

### 结论

Wave-Former通过引入物理感知的形状完成模型，克服了现有毫米波重建方法的局限性，实现了对完全遮挡物体的高精度三维重建，为机器人技术、增强现实和物流等领域开辟了新的应用可能性。

### 翻译

我们提出Wave-Former，一种新颖的方法，能够对完全遮挡的、多样化的日常物体进行高精度的三维形状重建。这种能力可以开辟跨越机器人技术、增强现实和物流等领域的全新应用。我们的方法利用毫米波无线信号，这些信号可以穿透常见遮挡物并从隐藏物体上反射回来。与过去的毫米波重建方法（存在覆盖范围有限和高噪声问题）相比，Wave-Former引入了一种物理感知的形状完成模型，能够推断完整的三维几何结构。Wave-Former设计的核心是一个新颖的三阶段管道，该管道通过结合毫米波信号的物理特性，将原始无线信号与最近基于视觉的形状完成进展联系起来。该管道提出候选几何表面，采用专为毫米波信号设计的基于Transformer的形状完成模型，最后执行熵引导的表面选择。这使得Wave-Former能够完全使用合成点云进行训练，同时展现出对真实世界数据的出色泛化能力。与最先进的基线方法直接比较，Wave-Former将召回率从54%提高到72%，同时保持85%的高精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何使用毫米波信号完全重建被遮挡物体的3D形状问题。这个问题很重要，因为传统光学传感器（如相机和LiDAR）在物体被完全遮挡时无法工作，而虽然毫米波可以穿透遮挡，但现有方法只能重建物体朝向雷达的部分表面，无法获取完整形状。解决这个问题可以扩展到机器人、增强现实和物流等多个应用领域，使系统能够感知封闭盒子或杂物下的物体。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到毫米波信号与可见光的不同特性：毫米波主要发生镜面反射而非漫反射，噪声更大，分辨率更低。这导致现有基于视觉的形状补全模型无法直接应用于毫米波数据。作者借鉴了传统毫米波成像方法（如Backprojection和mmNorm）和基于视觉的形状补全模型（如PoinTr），但设计了物理感知的训练框架，将毫米波的物理特性直接嵌入学习过程。核心思路是使用合成数据训练，同时引入镜面感知的归纳偏差和反射依赖的可见性模式，使模型能够预测可能无法观测到的区域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过物理感知的训练框架将毫米波信号的物理特性直接嵌入学习过程，实现完全被遮挡物体的3D形状重建。整体实现采用三阶段流程：1)毫米波表面提议：将原始毫米波测量值转换为候选部分表面；2)物理感知的形状补全：将训练好的模型应用于每个候选表面，生成完整重建；3)熵引导的表面选择：通过量化局部熵选择最优重建结果。这种方法可以在完全合成的数据上训练，同时实现对真实世界数据的出色泛化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)物理感知的毫米波形状补全框架，首次实现对完全被遮挡物体的3D重建；2)最先进的性能，在真实数据集上将召回率从54%提高到72%，同时保持85%的高精度；3)与普通视觉模型的对比研究，证明其优越性。相比之前的工作，Wave-Former的不同之处在于：嵌入毫米波物理特性到训练过程；使用三阶段推理流程；完全在合成数据上训练而无需真实毫米波数据；能够处理高噪声和低覆盖率的毫米波测量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Wave-Former通过物理感知的训练框架和三阶段推理流程，首次实现了使用毫米波信号对完全被遮挡的多样化日常物体进行高精度完整3D重建，突破了传统毫米波成像只能重建物体表面的限制。'}


### 论文摘要

We present Wave-Former, a novel method capable of high-accuracy 3D shape reconstruction for completely occluded, diverse, everyday objects. This capability can open new applications spanning robotics, augmented reality, and logistics. Our approach leverages millimeter-wave (mmWave) wireless signals, which can penetrate common occlusions and reflect off hidden objects. In contrast to past mmWave reconstruction methods, which suffer from limited coverage and high noise, Wave-Former introduces a physics-aware shape completion model capable of inferring full 3D geometry. At the heart of Wave-Former's design is a novel three-stage pipeline which bridges raw wireless signals with recent advancements in vision-based shape completion by incorporating physical properties of mmWave signals. The pipeline proposes candidate geometric surfaces, employs a transformer-based shape completion model designed specifically for mmWave signals, and finally performs entropy-guided surface selection. This enables Wave-Former to be trained using entirely synthetic point-clouds, while demonstrating impressive generalization to real-world data.In head-to-head comparisons with state-of-the-art baselines, Wave-Former raises recall from 54% to 72% while maintaining a high precision of 85%.

---

## 2. Diffusion As Self-Distillation: End-to-End Latent Diffusion In One Model

**论文链接:** [http://arxiv.org/abs/2511.14716v1](http://arxiv.org/abs/2511.14716v1)

**作者:** Xiyuan Wang, Muhan Zhang

**发布时间:** 2025-11-18

**备注:** Tech Report. 10 pages

### GPT解析

### 总结

该研究提出了一种名为扩散作为自蒸馏的新框架，成功将标准潜在扩散模型的三部分架构统一为单个端到端可训练的网络，在ImageNet图像生成任务上取得了优异性能。

### 背景

标准潜在扩散模型采用复杂的三部分架构（编码器、解码器和扩散网络），需要多阶段训练，计算效率低下，性能次优，且无法与视觉基础模型中常见的单网络架构统一。

### 目的

将潜在扩散模型的编码器、解码器和扩散网络三个组件统一为单个端到端可训练的网络，提高计算效率和性能。

### 方法

提出扩散作为自蒸馏框架，通过修改训练目标来稳定潜在空间，解决简单联合训练方法导致的'潜在崩溃'问题。

### 主要发现

简单的联合训练方法会因'潜在崩溃'而灾难性失败，扩散训练目标会干扰网络学习良好潜在表示的能力；通过将扩散与基于自蒸馏的无监督学习方法进行类比，确定了这种不稳定性的根本原因。

### 结论

扩散作为自蒸馏框架首次实现了单网络稳定端到端训练，该网络同时学习编码、解码和执行扩散，在ImageNet 256×256条件生成任务上仅用42M/118M/205M参数和50个训练周期就取得了FID=13.44/6.38/4.25的优异性能，无需使用分类器引导。

### 翻译

标准潜在扩散模型依赖于复杂的三部分架构，包括单独的编码器、解码器和扩散网络，这些组件需要多阶段训练。这种模块化设计计算效率低下，导致性能次优，并阻碍了扩散与视觉基础模型中常见的单网络架构的统一。我们的目标是将这三个组件统一为单个端到端可训练的网络。我们首先证明，由于'潜在崩溃'，简单的联合训练方法会灾难性地失败，其中扩散训练目标干扰了网络学习良好潜在表示的能力。我们通过将扩散与基于自蒸馏的无监督学习方法进行新颖类比，确定了这种不稳定性的根本原因。基于这一见解，我们提出了扩散作为自蒸馏框架，对训练目标进行了关键修改以稳定潜在空间。这种方法首次实现了单网络的稳定端到端训练，该网络同时学习编码、解码和执行扩散。DSD在ImageNet 256×256条件生成任务上取得了卓越性能：仅使用42M/118M/205M参数和50个训练周期，FID得分为13.44/6.38/4.25，且未使用无分类器引导。


### 论文摘要

Standard Latent Diffusion Models rely on a complex, three-part architecture consisting of a separate encoder, decoder, and diffusion network, which are trained in multiple stages. This modular design is computationally inefficient, leads to suboptimal performance, and prevents the unification of diffusion with the single-network architectures common in vision foundation models. Our goal is to unify these three components into a single, end-to-end trainable network. We first demonstrate that a naive joint training approach fails catastrophically due to ``latent collapse'', where the diffusion training objective interferes with the network's ability to learn a good latent representation. We identify the root causes of this instability by drawing a novel analogy between diffusion and self-distillation based unsupervised learning method. Based on this insight, we propose Diffusion as Self-Distillation (DSD), a new framework with key modifications to the training objective that stabilize the latent space. This approach enables, for the first time, the stable end-to-end training of a single network that simultaneously learns to encode, decode, and perform diffusion. DSD achieves outstanding performance on the ImageNet $256\times 256$ conditional generation task: FID=13.44/6.38/4.25 with only 42M/118M/205M parameters and 50 training epochs on ImageNet, without using classifier-free-guidance.

---

## 3. Near-Lossless Model Compression Enables Longer Context Inference in DNA Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.14694v1](http://arxiv.org/abs/2511.14694v1)

**作者:** Rui Zhu, Xiaopu Zhou, Haixu Tang, Stephen W. Scherer, Lucila Ohno-Machado

**发布时间:** 2025-11-18

### GPT解析

### 总结

这篇论文介绍了FOCUS(面向特征的超长自注意力压缩)，一种创新的渐进式上下文压缩模块，可解决DNA大语言模型在处理超长序列时面临的计算和内存约束问题。

### 背景

DNA大语言模型在跨物种DNA语料库上训练，学习基因组序列的基本'语法'和进化模式，成为DNA序列建模的强大先验知识，特别是在长距离范围内。

### 目的

解决DNA大语言模型在实际应用中的两个主要约束：自注意力的二次计算复杂度和自回归解码过程中键值缓存所需的不断增长的内存需求。

### 方法

FOCUS结合了基因组学中的k-mer表示和可学习的分层压缩：在k-mer粒度插入摘要令牌，在多个Transformer层中渐进式压缩注意力键和值激活，只保留跨窗口的摘要KV状态，并采用共享边界窗口方案实现静态跨窗口接口。

### 主要发现

FOCUS在保留的人类染色体上实现了接近无损的保真度，将1kb上下文压缩为仅10个摘要令牌(约100倍)，每个核苷酸概率的平均偏移仅为约0.0004；同时将有效推理扩展从O(N^2)转换为近线性O(N)，在消费级GPU上实现了约100倍更长的推理窗口。

### 结论

FOCUS成功解决了DNA大语言模型在处理超长序列时的计算和内存约束问题，通过创新的压缩方法实现了接近无损的保真度，同时大幅提高了处理效率。

### 翻译

Trained on massive cross-species DNA corpora, DNA large language models (LLMs) learn the fundamental 'grammar' and evolutionary patterns of genomic sequences. This makes them powerful priors for DNA sequence modeling, particularly over long ranges. However, two major constraints hinder their use in practice: the quadratic computational cost of self-attention and the growing memory required for key-value (KV) caches during autoregressive decoding. These constraints force the use of heuristics such as fixed-window truncation or sliding windows, which compromise fidelity on ultra-long sequences by discarding distant information. We introduce FOCUS (Feature-Oriented Compression for Ultra-long Self-attention), a progressive context-compression module that can be plugged into pretrained DNA LLMs. FOCUS combines the established k-mer representation in genomics with learnable hierarchical compression: it inserts summary tokens at k-mer granularity and progressively compresses attention key and value activations across multiple Transformer layers, retaining only the summary KV states across windows while discarding ordinary-token KV. A shared-boundary windowing scheme yields a stationary cross-window interface that propagates long-range information with minimal loss. We validate FOCUS on an Evo-2-based DNA LLM fine-tuned on GRCh38 chromosome 1 with self-supervised training and randomized compression schedules to promote robustness across compression ratios. On held-out human chromosomes, FOCUS achieves near-lossless fidelity: compressing a 1 kb context into only 10 summary tokens (about 100x) shifts the average per-nucleotide probability by only about 0.0004. Compared to a baseline without compression, FOCUS reduces KV-cache memory and converts effective inference scaling from O(N^2) to near-linear O(N), enabling about 100x longer inference windows on commodity GPUs with near-lossless fidelity.


### 论文摘要

Trained on massive cross-species DNA corpora, DNA large language models (LLMs) learn the fundamental "grammar" and evolutionary patterns of genomic sequences. This makes them powerful priors for DNA sequence modeling, particularly over long ranges. However, two major constraints hinder their use in practice: the quadratic computational cost of self-attention and the growing memory required for key-value (KV) caches during autoregressive decoding. These constraints force the use of heuristics such as fixed-window truncation or sliding windows, which compromise fidelity on ultra-long sequences by discarding distant information. We introduce FOCUS (Feature-Oriented Compression for Ultra-long Self-attention), a progressive context-compression module that can be plugged into pretrained DNA LLMs. FOCUS combines the established k-mer representation in genomics with learnable hierarchical compression: it inserts summary tokens at k-mer granularity and progressively compresses attention key and value activations across multiple Transformer layers, retaining only the summary KV states across windows while discarding ordinary-token KV. A shared-boundary windowing scheme yields a stationary cross-window interface that propagates long-range information with minimal loss. We validate FOCUS on an Evo-2-based DNA LLM fine-tuned on GRCh38 chromosome 1 with self-supervised training and randomized compression schedules to promote robustness across compression ratios. On held-out human chromosomes, FOCUS achieves near-lossless fidelity: compressing a 1 kb context into only 10 summary tokens (about 100x) shifts the average per-nucleotide probability by only about 0.0004. Compared to a baseline without compression, FOCUS reduces KV-cache memory and converts effective inference scaling from O(N^2) to near-linear O(N), enabling about 100x longer inference windows on commodity GPUs with near-lossless fidelity.

---

## 4. MRI Embeddings Complement Clinical Predictors for Cognitive Decline Modeling in Alzheimer's Disease Cohorts

**论文链接:** [http://arxiv.org/abs/2511.14601v1](http://arxiv.org/abs/2511.14601v1)

**作者:** Nathaniel Putera, Daniel Vilet Rodríguez, Noah Videcrantz, Julia Machnio, Mostafa Mehdipour Ghazi

**发布时间:** 2025-11-18

**备注:** Accepted at SPIE - Medical Imaging Conference 2026

### GPT解析

### 总结

本研究评估了表格和基于transformer的MRI表示在阿尔茨海默病认知衰退预测中的贡献，发现不同模态具有互补优势，临床特征擅长识别高风险极端，而transformer-based MRI嵌入对稳定性标志物更敏感。

### 背景

准确建模阿尔茨海默病的认知衰退对早期分层和个性化管理至关重要，但表格预测因子在捕捉细微脑变化方面存在局限性。

### 目的

评估表格和基于图像的表示在预测认知衰退方面的贡献，重点关注基于transformer的磁共振成像(MRI)嵌入。

### 方法

引入基于动态时间规整聚类的轨迹感知标记策略；通过无监督重建训练3D视觉Transformer以获取MRI嵌入；使用传统和深度学习方法评估预训练编码器；与表格表示和卷积网络基线比较。

### 主要发现

临床和体积特征在预测轻度至重度进展方面AUC最高(约0.70)；ViT模型的MRI嵌入在区分认知稳定个体方面最有效(AUC=0.71)；所有方法在异质性中度组中表现不佳。

### 结论

临床特征擅长识别高风险极端，transformer-based MRI嵌入对稳定性标志物更敏感，建议采用多模态融合策略进行AD进展建模。

### 翻译

阿尔茨海默病认知衰退的准确建模对于早期分层和个性化管理至关重要。虽然表格预测因子提供了稳健的全局风险标志物，但它们捕捉细微脑变化的能力仍然有限。在本研究中，我们评估了表格和基于图像的表示的预测贡献，重点关注基于transformer的磁共振成像(MRI)嵌入。我们引入了基于动态时间规整聚类的轨迹感知标记策略，以捕捉认知变化的异质性模式，并通过在对齐和增强的MRI数据上进行无监督重建来训练3D视觉Transformer(ViT)，以获得不依赖进展标签的保留解剖结构的嵌入。随后使用传统机器学习分类器和深度学习头评估预训练的编码器嵌入，并与表格表示和卷积网络基线进行比较。结果突显了不同模态的互补优势。临床和体积特征在预测轻度至重度进展方面获得了最高的AUC值(约0.70)，强调了它们在捕捉全局衰退轨迹方面的效用。相比之下，ViT模型的MRI嵌入在区分认知稳定个体方面最为有效，AUC为0.71。然而，所有方法在异质性中度组中表现不佳。这些发现表明，临床特征在识别高风险极端方面表现出色，而基于transformer的MRI嵌入对稳定性的微妙标志物更为敏感，这推动了AD进展建模的多模态融合策略。


### 论文摘要

Accurate modeling of cognitive decline in Alzheimer's disease is essential for early stratification and personalized management. While tabular predictors provide robust markers of global risk, their ability to capture subtle brain changes remains limited. In this study, we evaluate the predictive contributions of tabular and imaging-based representations, with a focus on transformer-derived Magnetic Resonance Imaging (MRI) embeddings. We introduce a trajectory-aware labeling strategy based on Dynamic Time Warping clustering to capture heterogeneous patterns of cognitive change, and train a 3D Vision Transformer (ViT) via unsupervised reconstruction on harmonized and augmented MRI data to obtain anatomy-preserving embeddings without progression labels. The pretrained encoder embeddings are subsequently assessed using both traditional machine learning classifiers and deep learning heads, and compared against tabular representations and convolutional network baselines. Results highlight complementary strengths across modalities. Clinical and volumetric features achieved the highest AUCs of around 0.70 for predicting mild and severe progression, underscoring their utility in capturing global decline trajectories. In contrast, MRI embeddings from the ViT model were most effective in distinguishing cognitively stable individuals with an AUC of 0.71. However, all approaches struggled in the heterogeneous moderate group. These findings indicate that clinical features excel in identifying high-risk extremes, whereas transformer-based MRI embeddings are more sensitive to subtle markers of stability, motivating multimodal fusion strategies for AD progression modeling.

---

## 5. Mind the Gaps: Measuring Visual Artifacts in Dimensionality Reduction

**论文链接:** [http://arxiv.org/abs/2511.14544v1](http://arxiv.org/abs/2511.14544v1)

**作者:** Jaume Ros, Alessio Arleo, Fernando Paulovich

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究引入了一种名为Warping Index(WI)的新指标，用于评估降维投影的质量，特别关注点之间空区域的正确保留，以提供更忠实的数据视觉表示。

### 背景

降维技术常用于高维数据的视觉探索和分析，能够将高维数据集投影到2D平面上。然而，低维投影通常存在不易识别的失真，可能导致误导性结论。

### 目的

开发一种新的投影质量度量方法，不仅关注数据全局或局部结构的保留，还考虑可视化图的视觉失真，避免异常值或伪影误导视觉分析。

### 方法

提出Warping Index(WI)作为衡量降维投影到2D平面质量的新指标，基于点之间空区域正确保留对数据忠实视觉表示至关重要的假设。

### 主要发现

现有投影质量度量工具大多只关注数据结构的保留，而忽视了视觉失真和异常值的影响；点之间空区域的正确保留对忠实的数据可视化至关重要。

### 结论

Warping Index(WI)为评估降维投影质量提供了新视角，特别关注空区域的保留，有助于减少视觉分析中的误导性结论。

### 翻译

降维(DR)技术因其能够将高维数据集投影到2D平面而常用于高维数据的视觉探索和分析。然而，在较低维度上投影数据集通常涉及一些失真，这些失真不一定容易被识别，但可能导致用户得出误导性结论。已开发了几种投影质量度量(PQMs)作为量化DR投影拟合优度的工具；但它们主要侧重于衡量投影在多大程度上捕捉了数据的全局或局部结构，而没有考虑所得可视化图的视觉失真，因此常常忽略可能误导投影视觉分析的异常值或伪影。在这项工作中，我们引入了Warping Index(WI)，这是一种用于衡量DR投影到2D平面质量的新指标，基于点之间的空区域正确保留对于数据的忠实视觉表示至关重要的假设。


### 论文摘要

Dimensionality Reduction (DR) techniques are commonly used for the visual exploration and analysis of high-dimensional data due to their ability to project datasets of high-dimensional points onto the 2D plane. However, projecting datasets in lower dimensions often entails some distortion, which is not necessarily easy to recognize but can lead users to misleading conclusions. Several Projection Quality Metrics (PQMs) have been developed as tools to quantify the goodness-of-fit of a DR projection; however, they mostly focus on measuring how well the projection captures the global or local structure of the data, without taking into account the visual distortion of the resulting plots, thus often ignoring the presence of outliers or artifacts that can mislead a visual analysis of the projection. In this work, we introduce the Warping Index (WI), a new metric for measuring the quality of DR projections onto the 2D plane, based on the assumption that the correct preservation of empty regions between points is of crucial importance towards a faithful visual representation of the data.

---

## 6. Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details

**论文链接:** [http://arxiv.org/abs/2511.14539v1](http://arxiv.org/abs/2511.14539v1)

**作者:** Qiang Bai, Bojian Wu, Xi Yang, Zhizhong Han

**发布时间:** 2025-11-18

**备注:** Accepted as an Poster paper at the AAAI Conference on Artificial Intelligence (AAAI-26)

### GPT解析

### 总结

该研究提出了一种在共同空间中表示多个神经符号距离函数（SDFs）的方法，以解决分析具有高保真几何细节的多个SDF时遇到的潜在空间信息有限和几何细节丢失的问题。

### 背景

神经符号距离函数（Neural SDFs）是使用神经网络表示3D形状或场景的重要方法，SDF是一种隐式函数，可以在特定坐标查询符号距离来恢复3D表面。然而，隐式函数在单个形状或场景上表现良好，但在分析多个具有高保真几何细节的SDF时存在障碍。

### 目的

在共同空间中表示多个SDF，用更紧凑的潜在表示恢复更多高保真几何细节。

### 方法

充分利用基于泛化和过拟合的学习策略来保留高保真几何细节，并引入一种新的采样策略来采样训练查询，以提高训练效率并消除由其他SDF影响引起的伪影。

### 主要发现

在广泛使用的基准上进行的数值和视觉评估表明，该方法在表示能力和紧凑性方面优于最新方法。

### 结论

该方法能够有效地表示多个SDF，并在保持高保真几何细节的同时实现更紧凑的潜在表示。

### 翻译

神经符号距离函数（SDFs）一直是使用神经网络表示3D形状或场景的重要表示方法。SDF是一种隐式函数，可以在特定坐标查询符号距离以恢复3D表面。虽然隐式函数在单个形状或场景上效果良好，但由于SDF潜在空间中编码的信息有限以及几何细节的丢失，在分析具有高保真几何细节的多个SDF时存在障碍。为了克服这些障碍，我们引入了一种在共同空间中表示多个SDF的方法，旨在用更紧凑的潜在表示恢复更多高保真几何细节。我们的关键思想是充分利用基于泛化和过拟合的学习策略的好处，这些策略能够用紧凑的潜在代码保留高保真几何细节。基于此框架，我们还引入了一种新的采样策略来采样训练查询。这种采样可以提高训练效率，并消除由其他SDF影响引起的伪影。我们在广泛使用的基准上报告了数值和视觉评估，以验证我们的设计，并展示了在表示能力和紧凑性方面优于最新方法的优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决神经符号距离函数（Neural SDFs）在表示多个具有高保真几何细节的3D形状时面临的挑战。具体来说，现有方法在潜在空间中编码的信息有限，且容易丢失几何细节。这个问题在计算机视觉和机器人领域非常重要，因为精确的3D表示对增强现实、虚拟现实、自动驾驶等应用至关重要，能够紧凑地表示多个高细节3D模型对3D内容创作和生成模型的发展有重要推动作用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：基于泛化的神经网络方法（如DeepSDF）难以恢复高频几何细节，而过拟合的体积网格方法（如Instant-NGP）难以在多个形状间共享表示。作者认识到这两种方法的互补性，设计了双分支架构：一个泛化分支用于远离表面的区域，一个过拟合分支用于表面附近。该方法借鉴了神经SDF的隐式表示思想、体积网格的局部细节处理能力、位置编码技术和marching cubes重建算法，但创新性地将它们结合并提出了新的采样策略来解决多形状训练中的不平衡问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合泛化-based和过拟合-based两种策略的优势：使用神经网络建模远离表面的符号距离场，利用体积网格建模表面附近的符号距离场，两者共享一个紧凑的潜在代码表示不同形状。整体流程包括：1) 数据预处理；2) 初始化双分支网络和共享体积网格；3) 训练过程，包括平衡约束采样、双分支前向传播、损失计算和参数更新；4) 推理过程，使用marching cubes重建表面并根据查询点位置选择使用哪个分支的预测；5) 评估重建质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 双分支混合架构，结合神经网络和体积网格的优势；2) 紧凑潜在空间表示，支持多形状共享；3) 平衡约束采样策略，避免形状间相互干扰；4) 选择性距离场融合，根据查询位置选择最佳预测。相比之前工作，不同于纯神经网络方法（能恢复更高细节）、纯体积网格方法（支持多形状共享）、其他混合方法（有针对性地在不同区域使用不同方法）和其他多形状表示方法（保持高保真细节且表示更紧凑）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种双分支混合架构，结合神经网络和体积网格的优势，在紧凑的潜在空间中实现了具有高保真几何细节的多神经符号距离函数的高效表示与重建。'}


### 论文摘要

Neural signed distance functions (SDFs) have been a vital representation to represent 3D shapes or scenes with neural networks. An SDF is an implicit function that can query signed distances at specific coordinates for recovering a 3D surface. Although implicit functions work well on a single shape or scene, they pose obstacles when analyzing multiple SDFs with high-fidelity geometry details, due to the limited information encoded in the latent space for SDFs and the loss of geometry details. To overcome these obstacles, we introduce a method to represent multiple SDFs in a common space, aiming to recover more high-fidelity geometry details with more compact latent representations. Our key idea is to take full advantage of the benefits of generalization-based and overfitting-based learning strategies, which manage to preserve high-fidelity geometry details with compact latent codes. Based on this framework, we also introduce a novel sampling strategy to sample training queries. The sampling can improve the training efficiency and eliminate artifacts caused by the influence of other SDFs. We report numerical and visual evaluations on widely used benchmarks to validate our designs and show advantages over the latest methods in terms of the representative ability and compactness.

---

## 7. DeCo-VAE: Learning Compact Latents for Video Reconstruction via Decoupled Representation

**论文链接:** [http://arxiv.org/abs/2511.14530v1](http://arxiv.org/abs/2511.14530v1)

**作者:** Xiangchen Yin, Jiahui Yuan, Zhangchi Hu, Wenzhang Sun, Jie Chen, Xiaozhen Qiao, Hao Li, Xiaoyan Sun

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了解耦VAE(DeCo-VAE)模型，通过将视频内容分解为关键帧、运动和残差三个独立组件，并为每个组件学习专门的潜在表示，实现了紧凑的视频潜在表示和优越的重建性能。

### 背景

现有的视频变分自编码器(VAEs)通常忽略帧内容之间的相似性，导致冗余的潜在建模。

### 目的

实现紧凑的潜在表示，提高视频重建性能。

### 方法

提出解耦VAE(DeCo-VAE)，通过显式解耦将视频内容分解为关键帧、运动和残差三个组件，为每个组件设计专用编码器并采用共享的3D解码器，同时采用解耦适应策略在训练过程中冻结部分编码器以确保稳定训练和准确学习静态和动态特征。

### 主要发现

通过大量定量和定性实验证明，DeCo-VAE实现了优越的视频重建性能。

### 结论

DeCo-VAE通过解耦视频内容并分别建模不同组件，实现了更紧凑的潜在表示和更好的视频重建性能。

### 翻译

现有的视频变分自编码器(VAEs)通常忽略了帧内容之间的相似性，导致冗余的潜在建模。在本文中，我们提出了解耦VAE(DeCo-VAE)来实现紧凑的潜在表示。我们不是直接编码RGB像素，而是通过显式解耦将视频内容分解为不同的组件：关键帧、运动和残差，并为每个组件学习专门的潜在表示。为了避免组件间的干扰，我们为每个解耦组件设计了专门的编码器，并采用共享的3D解码器在重建过程中保持时空一致性。我们进一步利用了解耦适应策略，在训练其他编码器时冻结部分编码器，确保稳定训练并准确学习静态和动态特征。大量的定量和定性实验证明，DeCo-VAE实现了优越的视频重建性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有视频变分自编码器忽略帧间相似性导致的冗余潜在建模问题。这个问题很重要，因为视频数据本质上高度冗余应该更容易被压缩，而视频重建的质量和效率直接影响下游生成任务（如视频扩散模型）的性能。现有方法在效率与质量间难以平衡：要么使用密集3D网络提高质量但增加计算复杂度，要么使用轻量级架构降低成本但难以建模复杂动态。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受视频编解码器启发，将视频分解为关键帧、运动和残差三个组件来去除冗余。他们设计专用编码器处理每个组件，使用共享3D解码器保持时空一致性，并采用分阶段训练策略冻结部分编码器。这种方法借鉴了视频编解码器的分解思想、传统VAE架构以及现有视频扩散模型的需求，但创新性地将这些思想整合到视频VAE框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过显式解耦视频内容（关键帧、运动和残差）来学习紧凑的潜在表示，避免原始像素编码中的特征纠缠。实现流程包括：1)视频解耦-选择首帧为关键帧，提取运动信息，计算残差；2)潜在表示学习-三个专用编码器分别处理各组件，采样潜在向量；3)视频重建-共享3D解码器生成重建组件，通过重耦合操作组合成最终视频；4)训练策略-分阶段冻结部分编码器，使用多种损失函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)显式视频内容解耦为关键帧、运动和残差三个组件；2)专用编码器和共享解码器架构避免组件间干扰；3)解耦适应策略确保稳定训练。相比之前工作，DeCo-VAE不是直接编码原始像素，而是对解耦组件编码，更好地利用帧间冗余；比其他解耦方法（如VidTwin）更彻底地分离运动和静态特征；比轻量级架构更好地建模复杂动态；比密集3D网络更高效地保持高质量重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DeCo-VAE通过将视频解耦为关键帧、运动和残差三个组件，并采用专用编码器和共享解码器架构，实现了高效且高质量的视频重建，为下游视频生成任务提供了紧凑且可解释的潜在表示。'}


### 论文摘要

Existing video Variational Autoencoders (VAEs) generally overlook the similarity between frame contents, leading to redundant latent modeling. In this paper, we propose decoupled VAE (DeCo-VAE) to achieve compact latent representation. Instead of encoding RGB pixels directly, we decompose video content into distinct components via explicit decoupling: keyframe, motion and residual, and learn dedicated latent representation for each. To avoid cross-component interference, we design dedicated encoders for each decoupled component and adopt a shared 3D decoder to maintain spatiotemporal consistency during reconstruction. We further utilize a decoupled adaptation strategy that freezes partial encoders while training the others sequentially, ensuring stable training and accurate learning of both static and dynamic features. Extensive quantitative and qualitative experiments demonstrate that DeCo-VAE achieves superior video reconstruction performance.

---

## 8. Enhancing End-to-End Autonomous Driving with Risk Semantic Distillaion from VLM

**论文链接:** [http://arxiv.org/abs/2511.14499v1](http://arxiv.org/abs/2511.14499v1)

**作者:** Jack Qin, Zhitao Wang, Yinan Zheng, Keyu Chen, Yang Zhou, Yuanxin Zhong, Siyuan Cheng

**发布时间:** 2025-11-18

### GPT解析

### 总结

自动驾驶系统在复杂场景中表现良好但泛化能力有限，作者提出风险语义蒸馏(RSD)框架，利用视觉语言模型增强端到端AD主干网络训练，通过RiskHead模块将VLM中的风险估计蒸馏到BEV特征中，生成可解释的风险注意力图，提高模型处理复杂驾驶场景的能力。

### 背景

当前自动驾驶系统在复杂驾驶场景中表现出色，但泛化能力仍是一个关键限制。相关研究探索了使用视觉语言模型(VLMs)解决少样本或零样本任务，但引入了混合AD系统可能导致不一致性的问题。另一种视觉语言动作(VLA)框架直接从VLM生成控制动作，但计算需求过高。

### 目的

解决自动驾驶系统的泛化能力限制，处理未见过的场景或不熟悉的传感器配置，同时避免混合系统的不一致性和端到端解决方案的高计算需求。

### 方法

提出风险语义蒸馏(RSD)框架，利用视觉语言模型增强端到端AD主干网络的训练。引入RiskHead模块，将VLM中的因果风险估计蒸馏到鸟瞰图(BEV)特征中，生成可解释的风险注意力图，使BEV特征能够学习更丰富、更细致的风险注意力表示。

### 主要发现

RSD方法使BEV特征能够学习更丰富、更细致的风险注意力表示，增强模型处理空间边界和危险对象的能力。通过关注风险注意力，RSD更符合人类驾驶行为，有利于在复杂动态环境中导航。实验证明RSD在处理复杂不可预测驾驶条件方面有效，显著提高了感知和规划能力。

### 结论

风险语义蒸馏(RSD)框架有效解决了自动驾驶系统的泛化能力限制，通过利用视觉语言模型增强端到端AD主干网络训练，显著提高了模型在复杂和动态环境中的感知和规划能力。

### 翻译

自动驾驶(AD)系统在复杂驾驶场景中表现出色。然而，泛化能力仍然是当前系统的一个关键限制，指的是处理未见过的场景或不熟悉的传感器配置的能力。相关研究探索了使用视觉语言模型(VLMs)来解决少样本或零样本任务。虽然这些方法很有前景，但它们引入了一个新的挑战：混合AD系统的出现，其中两个不同的系统用于规划轨迹，可能导致潜在的不一致性。替代研究方向探索了视觉语言动作(VLA)框架，直接从VLM生成控制动作。然而，这些端到端解决方案表现出过高的计算需求。为了克服这些挑战，我们引入了风险语义蒸馏(RSD)，这是一个新颖的框架，利用VLMs增强端到端(2E)AD主干网络的训练。通过为关键对象提供风险注意力，RSD解决了泛化问题。具体来说，我们引入了RiskHead，这是一个插件模块，将视觉语言模型中的因果风险估计蒸馏到鸟瞰图(BEV)特征中，产生可解释的风险-注意力图。这种方法使BEV特征能够学习更丰富、更细致的风险注意力表示，直接增强模型处理空间边界和危险对象的能力。通过关注风险注意力，RSD更好地符合人类驾驶行为，这对于在复杂和动态环境中导航至关重要。我们在Bench2Drive基准测试上的实验证明了RSD在处理复杂和不可预测的驾驶条件方面的有效性。由于RSD实现的增强BEV表示，我们观察到感知和规划能力都有显著提高。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决自动驾驶系统在处理罕见但关键的'长尾场景'时的泛化能力不足问题。这一问题在现实中非常重要，因为自动驾驶系统必须能够应对各种复杂、不可预测的驾驶情况，包括罕见但高风险的场景，如突然切入的车辆、施工区域或事故现场。当前系统在处理这些场景时的局限性是确保自动驾驶安全性和可靠性的主要障碍，也是阻碍这项技术广泛应用的关键因素。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的优缺点：双系统架构虽然平衡了计算效率和VLM的泛化能力，但会导致'分裂脑'问题；而直接将VLM微调为视觉语言动作(VLA)模型则计算需求过高，不适合实时部署。基于此，他们采用知识蒸馏思想，设计了风险语义蒸馏(RSD)框架。他们借鉴了VLM的环境理解能力、BEVFormer的BEV表示方法、变形注意力机制以及知识蒸馏技术，将这些技术与自动驾驶系统结合，创造了一个即插即用的解决方案，既利用了VLM的推理能力，又保持了实时性和效率。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过风险语义蒸馏，将VLM对交通参与者风险的认知能力转移到端到端自动驾驶系统中，使系统能够像人类一样关注高风险对象。整体流程分为三部分：1) VLM增强的风险语义标注：使用OV-DINO获取对象标签和边界框，通过Qwen模型进行风险推理；2) 风险语义蒸馏：设计RiskHead模块，通过BEV重新批处理、最近邻匹配和变形注意力机制提取风险语义；3) 端到端模型集成：在VAD模型基础上加入风险语义损失函数，训练模型关注高风险对象，最终增强BEV表示，提高感知和规划能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 专门针对自动驾驶场景的VLM风险检测流程，实现零样本能力；2) 专门的蒸馏架构，有效将VLM知识转移到紧凑模型中；3) 即插即用的RiskHead模块；4) 跨视图特征对齐机制。相比前人工作，不同于双系统架构的'分裂脑'问题，RSD保持系统一致性；不同于直接微调VLM的高计算需求，RSD推理速度快；不同于传统端到端方法，RSD显著提高了长尾场景处理能力；不同于通用知识蒸馏，RSD专注于风险语义，使模型更符合人类驾驶行为。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了风险语义蒸馏框架，通过将视觉语言模型的风险感知能力高效地转移到端到端自动驾驶系统中，在不增加计算负担的情况下显著提高了系统在复杂和长尾场景中的泛化能力和安全性。'}


### 论文摘要

The autonomous driving (AD) system has exhibited remarkable performance in complex driving scenarios. However, generalization is still a key limitation for the current system, which refers to the ability to handle unseen scenarios or unfamiliar sensor configurations.Related works have explored the use of Vision-Language Models (VLMs) to address few-shot or zero-shot tasks. While promising, these methods introduce a new challenge: the emergence of a hybrid AD system, where two distinct systems are used to plan a trajectory, leading to potential inconsistencies. Alternative research directions have explored Vision-Language-Action (VLA) frameworks that generate control actions from VLM directly. However, these end-to-end solutions demonstrate prohibitive computational demands. To overcome these challenges, we introduce Risk Semantic Distillation (RSD), a novel framework that leverages VLMs to enhance the training of End-to-End (E2E) AD backbones. By providing risk attention for key objects, RSD addresses the issue of generalization. Specifically, we introduce RiskHead, a plug-in module that distills causal risk estimates from Vision-Language Models into Bird's-Eye-View (BEV) features, yielding interpretable risk-attention maps.This approach allows BEV features to learn richer and more nuanced risk attention representations, which directly enhance the model's ability to handle spatial boundaries and risky objects.By focusing on risk attention, RSD aligns better with human-like driving behavior, which is essential to navigate in complex and dynamic environments. Our experiments on the Bench2Drive benchmark demonstrate the effectiveness of RSD in managing complex and unpredictable driving conditions. Due to the enhanced BEV representations enabled by RSD, we observed a significant improvement in both perception and planning capabilities.

---

## 9. Towards Stable and Structured Time Series Generation with Perturbation-Aware Flow Matching

**论文链接:** [http://arxiv.org/abs/2511.14488v1](http://arxiv.org/abs/2511.14488v1)

**作者:** Jintao Zhang, Mingyue Cheng, Zirui Liu, Xianquan Wang, Yitong Zhou, Qi Liu

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为PAFM（Perturbation-Aware Flow Matching）的时间序列生成框架，能够有效处理由局部扰动引起的结构一致性问题，通过扰动引导训练和双路径速度场增强模型对扰动轨迹的敏感性，在生成稳定且结构一致的时间序列方面表现出色。

### 背景

时间序列生成对广泛应用至关重要，支持下游分析和决策任务。然而，由局部扰动引起的内在时间异质性对生成结构一致的时间序列提出了重大挑战。

### 目的

解决现有时间序列生成方法在处理由局部扰动引起的结构一致性问题上的局限性，提出一种能够生成稳定且结构一致的时间序列的框架。

### 方法

提出了名为PAFM（Perturbation-Aware Flow Matching）的框架，通过建模扰动轨迹确保稳定和结构一致的时间序列生成；引入扰动引导训练模拟局部扰动；利用双路径速度场捕获扰动下的轨迹偏差；使用具有流路由的专家混合解码器动态分配建模能力以响应不同的轨迹动态。

### 主要发现

流匹配通过轨迹级监督建模时间动力学提供了有前景的范式，但全局共享参数限制了速度场的统一表示，无法充分捕捉扰动时间序列中的突然转变；PAFM在无条件和条件生成任务上都优于强大的基线方法。

### 结论

PAFM框架能有效处理由局部扰动引起的挑战，生成稳定且结构一致的时间序列；通过扰动引导训练和双路径速度场，增强了模型对扰动轨迹的敏感性。

### 翻译

时间序列生成对广泛的应用至关重要，它极大地支持下游分析和决策任务。然而，由局部扰动引起的内在时间异质性给生成结构一致的时间序列带来了重大挑战。虽然流匹配通过轨迹级监督建模时间动力学提供了一种有前景的范式，但它无法充分捕捉扰动时间序列中的突然转变，因为使用全局共享参数将速度场限制为统一表示。为解决这些局限性，我们引入了PAFM，一个扰动感知流匹配框架，通过建模扰动轨迹来确保稳定和结构一致的时间序列生成。该框架集成了扰动引导训练来模拟局部扰动，并利用双路径速度场来捕获扰动下的轨迹偏差，从而增强了对扰动行为的精细建模以提高结构一致性。为进一步提高对轨迹扰动的敏感性同时增强表达能力，采用具有流路由的专家混合解码器能够根据不同的轨迹动态响应地分配建模能力。在无条件和条件生成任务上的大量实验表明，PAFM始终优于强大的基线方法。代码可在https://anonymous.4open.science/r/PAFM-03B2获取。


### 论文摘要

Time series generation is critical for a wide range of applications, which greatly supports downstream analytical and decision-making tasks. However, the inherent temporal heterogeneous induced by localized perturbations present significant challenges for generating structurally consistent time series. While flow matching provides a promising paradigm by modeling temporal dynamics through trajectory-level supervision, it fails to adequately capture abrupt transitions in perturbed time series, as the use of globally shared parameters constrains the velocity field to a unified representation. To address these limitations, we introduce \textbf{PAFM}, a \textbf{P}erturbation-\textbf{A}ware \textbf{F}low \textbf{M}atching framework that models perturbed trajectories to ensure stable and structurally consistent time series generation. The framework incorporates perturbation-guided training to simulate localized disturbances and leverages a dual-path velocity field to capture trajectory deviations under perturbation, enabling refined modeling of perturbed behavior to enhance the structural coherence. In order to further improve sensitivity to trajectory perturbations while enhancing expressiveness, a mixture-of-experts decoder with flow routing dynamically allocates modeling capacity in response to different trajectory dynamics. Extensive experiments on both unconditional and conditional generation tasks demonstrate that PAFM consistently outperforms strong baselines. Code is available at https://anonymous.4open.science/r/PAFM-03B2.

---

## 10. Notes on Kernel Methods in Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.14485v1](http://arxiv.org/abs/2511.14485v1)

**作者:** Diego Armando Pérez-Rosero, Danna Valentina Salazar-Dubois, Juan Camilo Lugo-Rojas, Andrés Marino Álvarez-Meza, Germán Castellanos-Dominguez

**发布时间:** 2025-11-18

### GPT解析

### 总结

这篇笔记提供了关于核方法及其在机器学习中几何基础的自包含介绍，从希尔伯特空间构建开始，发展了正定核理论、再生核希尔伯特空间和希尔伯特-施密特算子理论，强调了它们在统计估计和概率测度表示中的作用。

### 背景

核方法在机器学习中具有广泛应用，其理论基础涉及希尔伯特空间几何和概率论，为理解经典统计概念提供了新视角。

### 目的

提供核方法及其几何基础的自包含介绍，为更高级的机器学习主题奠定理论基础，包括高斯过程、核贝叶斯推断和现代机器学习的泛函分析方法。

### 方法

从希尔伯特空间的构建开始，发展正定核理论、再生核希尔伯特空间和希尔伯特-施密特算子理论；通过希尔伯特空间几何视角重新审视协方差、回归和信息量等经典概念；介绍核密度估计、分布的核嵌入和最大均值差异等技术。

### 主要发现

核方法为统计估计和概率测度表示提供了强有力的工具；希尔伯特空间几何为理解经典统计概念提供了新视角；核密度估计、分布的核嵌入和最大均值差异等方法在机器学习中具有重要作用。

### 结论

核方法及其几何基础是理解现代机器学习的重要工具，这些内容为学习更高级主题如高斯过程、核贝叶斯推断和泛函分析方法奠定了基础。

### 翻译

这些笔记提供了核方法及其在机器学习中的几何基础的自包含介绍。从希尔伯特空间的构建开始，我们发展了正定核理论、再生核希尔伯特空间(RKHS)和希尔伯特-施密特算子理论，强调了它们在统计估计和概率测度表示中的作用。通过希尔伯特空间几何的视角，我们重新审视了协方差、回归和信息量等经典概念。我们还介绍了核密度估计、分布的核嵌入和最大均值差异(MMD)。这些阐述旨在作为更高级主题的基础，包括高斯过程、核贝叶斯推断和现代机器学习的泛函分析方法。


### 论文摘要

These notes provide a self-contained introduction to kernel methods and their geometric foundations in machine learning. Starting from the construction of Hilbert spaces, we develop the theory of positive definite kernels, reproducing kernel Hilbert spaces (RKHS), and Hilbert-Schmidt operators, emphasizing their role in statistical estimation and representation of probability measures. Classical concepts such as covariance, regression, and information measures are revisited through the lens of Hilbert space geometry. We also introduce kernel density estimation, kernel embeddings of distributions, and the Maximum Mean Discrepancy (MMD). The exposition is designed to serve as a foundation for more advanced topics, including Gaussian processes, kernel Bayesian inference, and functional analytic approaches to modern machine learning.

---

## 11. CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring

**论文链接:** [http://arxiv.org/abs/2511.14469v1](http://arxiv.org/abs/2511.14469v1)

**作者:** Mingchen Zhong, Xin Lu, Dong Li, Senyan Xu, Ruixuan Jiang, Xueyang Fu, Baocai Yin

**发布时间:** 2025-11-18

### GPT解析

### 总结

CompEvent是一种创新的低光照视频去模糊方法，通过复值神经网络实现事件数据和RGB帧的整体融合，在处理低光照和运动模糊复合退化方面表现出色，优于现有方法。

### 背景

低光照视频去模糊在夜间监控和自动驾驶等应用中面临重大挑战，这些挑战由昏暗光线和长曝光引起。虽然事件相机具有优越的低光照敏感性和高时间分辨率，但现有的融合方法通常采用分阶段策略，限制了它们对复合低光照和运动模糊退化的有效性。

### 目的

克服现有方法的局限性，提出CompEvent，一个复杂的神经网络框架，实现事件数据和RGB帧的整体全过程融合，以增强联合恢复能力。

### 方法

CompEvent具有两个核心组件：1) 复杂时间对齐GRU，利用复值卷积并通过GRU迭代处理视频和事件流以实现时间对齐和连续融合；2) 复杂空间-频率学习模块，在空间域和频域执行统一的复值信号处理，通过空间结构和系统级特性促进深度融合。该方法利用复值神经网络的全程表示能力，实现全过程时空融合，最大化模态之间的互补学习。

### 主要发现

通过大量实验证明，CompEvent在解决这一具有挑战性的任务中优于最先进的方法，代码可在https://github.com/YuXie1/CompEvent获取。

### 结论

CompEvent通过整体全过程融合事件数据和RGB帧，有效解决了低光照视频去模糊问题，其两个核心组件共同实现了高效的去模糊能力。

### 翻译

低光照视频去模糊在夜间监控和自动驾驶等应用中因光线昏暗和长曝光而面临重大挑战。虽然事件相机凭借其优越的低光照敏感性和高时间分辨率提供了潜在解决方案，但现有的融合方法通常采用分阶段策略，限制了它们应对复合低光照和运动模糊退化的有效性。为此，我们提出了CompEvent，一个复杂的神经网络框架，能够实现事件数据和RGB帧的整体全过程融合，以增强联合恢复能力。CompEvent具有两个核心组件：1) 复杂时间对齐GRU，利用复值卷积并通过GRU迭代处理视频和事件流以实现时间对齐和连续融合；2) 复杂空间-频率学习模块，在空间域和频域执行统一的复值信号处理，通过空间结构和系统级特性促进深度融合。通过利用复值神经网络的全程表示能力，CompEvent实现了全过程时空融合，最大化了模态之间的互补学习，并显著增强了低光照视频去模糊能力。大量实验证明，CompEvent在解决这一具有挑战性的任务中优于最先进的方法。代码可在https://github.com/YuXie1/CompEvent获取。


### 论文摘要

Low-light video deblurring poses significant challenges in applications like nighttime surveillance and autonomous driving due to dim lighting and long exposures. While event cameras offer potential solutions with superior low-light sensitivity and high temporal resolution, existing fusion methods typically employ staged strategies, limiting their effectiveness against combined low-light and motion blur degradations. To overcome this, we propose CompEvent, a complex neural network framework enabling holistic full-process fusion of event data and RGB frames for enhanced joint restoration. CompEvent features two core components: 1) Complex Temporal Alignment GRU, which utilizes complex-valued convolutions and processes video and event streams iteratively via GRU to achieve temporal alignment and continuous fusion; and 2) Complex Space-Frequency Learning module, which performs unified complex-valued signal processing in both spatial and frequency domains, facilitating deep fusion through spatial structures and system-level characteristics. By leveraging the holistic representation capability of complex-valued neural networks, CompEvent achieves full-process spatiotemporal fusion, maximizes complementary learning between modalities, and significantly strengthens low-light video deblurring capability. Extensive experiments demonstrate that CompEvent outperforms SOTA methods in addressing this challenging task. The code is available at https://github.com/YuXie1/CompEvent.

---

## 12. From Topology to Behavioral Semantics: Enhancing BGP Security by Understanding BGP's Language with LLMs

**论文链接:** [http://arxiv.org/abs/2511.14467v1](http://arxiv.org/abs/2511.14467v1)

**作者:** Heng Zhao, Ruoyu Wang, Tianhang Zheng, Qi Li, Bo Lv, Yuyi Wang, Wenliang Du

**发布时间:** 2025-11-18

**备注:** 18 pages, 10 figures

### GPT解析

### 总结

BGPShield是一种基于LLM嵌入的BGP异常检测框架，通过捕获AS的行为画像和路由策略原理而非仅依赖拓扑结构，实现了高精度和高效能的异常检测。

### 背景

边界网关协议(BGP)基于信任的性质使其容易受到前缀劫持和错误配置等干扰，威胁路由稳定性。传统检测方法依赖人工检查可扩展性有限，而机器/深度学习方法存在精度不足、泛化能力有限和重训练成本高的问题。

### 目的

解决现有方法仅关注自治系统(AS)的拓扑结构而忽略全面语义特征的问题，开发能够捕获AS行为画像和路由策略原理的异常检测框架。

### 方法

提出BGPShield框架，使用LLM嵌入捕获AS的行为画像和路由策略原理；采用分段聚合方案将AS描述转换为LLM表示；设计轻量级对比缩减网络压缩表示；开发AR-DTW算法对齐和累积语义距离以揭示行为不一致性。

### 主要发现

在16个真实数据集上评估，BGPShield能检测出100%的已验证异常，误报率低于5%；使用的LLMs在评估前已发布，验证了泛化能力；BGPShield能在1秒内为未见过的AS构建表示，显著优于需要65小时重训练的BEAM方法。

### 结论

BGPShield通过利用LLM嵌入捕获AS的语义特征，有效解决了BGP异常检测中的精度和泛化问题，无需重训练即可快速处理新AS，具有实用优势。

### 翻译

边界网关协议(BGP)的基于信任的性质使其容易受到前缀劫持和错误配置等干扰，威胁路由稳定性。传统检测依赖人工检查，可扩展性有限。机器/深度学习方法可以自动化检测，但存在精度不足、泛化能力有限和重训练成本高的问题。这是因为现有方法关注拓扑结构而非自治系统(AS)的全面语义特征，常常误判功能相似但拓扑距离较远的AS。为此，我们提出BGPShield，一种基于LLM嵌入的异常检测框架，捕获每个AS的行为画像和路由策略原理，超越拓扑层面，如运营规模和全球角色。我们提出分段聚合方案将AS描述转换为LLM表示而不损失信息，以及轻量级对比缩减网络将它们压缩为语义一致版本。使用这些表示，我们的AR-DTW算法对齐并累积语义距离以揭示行为不一致性。在16个真实数据集上评估，BGPShield检测出100%的已验证异常，误报率低于5%。值得注意的是，所使用的LLMs在评估事件发布前已发布，验证了泛化能力。此外，BGPShield在1秒内为未见过的AS构建表示，显著优于需要昂贵重训练(平均65小时)的BEAM。


### 论文摘要

The trust-based nature of Border Gateway Protocol (BGP) makes it vulnerable to disruptions like prefix hijacking and misconfigurations, threatening routing stability. Traditional detection relies on manual inspection with limited scalability. Machine/Deep Learning (M/DL) approaches automate detection but suffer from suboptimal precision, limited generalizability, and high retraining costs. This is because existing methods focus on topological structures rather than comprehensive semantic characteristics of Autonomous Systems (ASes), often misinterpreting functionally similar but topologically distant ASes.   To address this, we propose BGPShield, an anomaly detection framework built on LLM embeddings that captures the Behavior Portrait and Routing Policy Rationale of each AS beyond topology, such as operational scale and global role. We propose a segment-wise aggregation scheme to transform AS descriptions into LLM representations without information loss, and a lightweight contrastive reduction network to compress them into a semantic-consistent version. Using these representations, our AR-DTW algorithm aligns and accumulates semantic distances to reveal behavioral inconsistencies. Evaluated on 16 real-world datasets, BGPShield detects 100% of verified anomalies with a false discovery rate below 5%. Notably, the employed LLMs were released prior to evaluation events, verifying generalizability. Furthermore, BGPShield constructs representations for unseen ASes within one second, significantly outperforming BEAM which demands costly retraining (averaging 65 hours).

---

## 13. Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.14427v1](http://arxiv.org/abs/2511.14427v1)

**作者:** Rickmer Krohn, Vignesh Prasad, Gabriele Tiboni, Georgia Chalvatzaki

**发布时间:** 2025-11-18

**备注:** 9 pages, 10 figures, preprint

### GPT解析

### 总结

提出了一种名为多感官动态预训练(MSDP)的新框架，用于学习适合任务导向策略学习的表达性多感官表征，解决了强化学习在多感官环境中学习困难的问题。

### 背景

有效接触密集型操作需要机器人协同利用视觉、力和本体感觉，但强化学习代理在多感官环境中学习困难，特别是在面对感官噪声和动态变化时。

### 目的

开发一种能够学习表达性多感官表征的框架，特别针对任务导向的策略学习，提高机器人在复杂环境中的操作能力。

### 方法

基于掩码自动编码的MSP框架，通过transformer编码器从部分传感器嵌入重建多感官观察，实现跨模态预测和传感器融合；采用非对称架构，通过交叉注意力机制让批评者提取动态特征，行动者接收稳定池化表示指导动作。

### 主要发现

该方法在各种干扰下表现出加速学习和稳健性能，包括传感器噪声和物体动力学变化；在多个接触密集型机器人操作任务中展示了有效性；仅需6,000次在线交互即可在真实机器人上实现高成功率。

### 结论

MSDP提供了一种简单而强大的复杂多感官机器人控制解决方案，对干扰表现出强大鲁棒性。

### 翻译

有效的接触密集型操作需要机器人协同利用视觉、力和本体感觉。然而，强化学习代理在这样的多感官环境中学习困难，特别是在感官噪声和动态变化的情况下。我们提出了多感官动态预训练(MSDP)，这是一种新颖的框架，用于学习适合任务导向策略学习的表达性多感官表征。MSDP基于掩码自动编码，并通过仅从一部分传感器嵌入重建多感官观察来训练基于transformer的编码器，实现跨模态预测和传感器融合。对于下游策略学习，我们引入了一种新的非对称架构，其中交叉注意力机制允许批评者从冻结的嵌入中提取动态的、任务特定的特征，而行动者接收稳定的池化表示来指导其动作。我们的方法在各种干扰下展示了加速学习和稳健性能，包括传感器噪声和物体动力学变化。在模拟和现实世界中多个具有挑战性的接触密集型机器人操作任务中的评估证明了MSDP的有效性。我们的方法对干扰表现出强大的鲁棒性，并在真实机器人上仅用6,000次在线交互就实现了高成功率，为复杂多感官机器人控制提供了一种简单而强大的解决方案。


### 论文摘要

Effective contact-rich manipulation requires robots to synergistically leverage vision, force, and proprioception. However, Reinforcement Learning agents struggle to learn in such multisensory settings, especially amidst sensory noise and dynamic changes. We propose MultiSensory Dynamic Pretraining (MSDP), a novel framework for learning expressive multisensory representations tailored for task-oriented policy learning. MSDP is based on masked autoencoding and trains a transformer-based encoder by reconstructing multisensory observations from only a subset of sensor embeddings, leading to cross-modal prediction and sensor fusion. For downstream policy learning, we introduce a novel asymmetric architecture, where a cross-attention mechanism allows the critic to extract dynamic, task-specific features from the frozen embeddings, while the actor receives a stable pooled representation to guide its actions. Our method demonstrates accelerated learning and robust performance under diverse perturbations, including sensor noise, and changes in object dynamics. Evaluations in multiple challenging, contact-rich robot manipulation tasks in simulation and the real world showcase the effectiveness of MSDP. Our approach exhibits strong robustness to perturbations and achieves high success rates on the real robot with as few as 6,000 online interactions, offering a simple yet powerful solution for complex multisensory robotic control.

---

## 14. Cranio-ID: Graph-Based Craniofacial Identification via Automatic Landmark Annotation in 2D Multi-View X-rays

**论文链接:** [http://arxiv.org/abs/2511.14411v1](http://arxiv.org/abs/2511.14411v1)

**作者:** Ravi Shankar Prasad, Nandani Sharma, Dinesh Singh

**发布时间:** 2025-11-18

**备注:** 11 pages, 6 figures

### GPT解析

### 总结

论文提出了一个名为Cranio-ID的新框架，用于法医颅面识别中的颅骨标志点自动定位和跨模态匹配。

### 背景

在法医颅面识别和许多生物医学应用中，颅骨标志点很重要。传统定位标志点的方法耗时且需要专业知识。当前使用的方法包括叠加和基于深度学习的自动标注方法，但这些方法因缺乏大规模验证研究而不够可靠。

### 目的

开发一个可靠的框架，用于自动标注2D颅骨上的标志点（面部X光扫描）及其相应的光学图像，并实现跨模态匹配。

### 方法

首先使用训练好的YOLO-pose模型自动标注2D颅骨上的标志点及其相应的光学图像；然后将这些标志点转化为图表示；最后使用跨注意力和最优传输框架在两种模态的图之间找到语义对应关系。

### 主要发现

在S2F和CUHK数据集上进行的广泛实验表明，所提出的框架在可靠性和准确性方面都有显著改进，并且在法医科学中的跨域颅骨到面部和素描到面部匹配中有效。

### 结论

Cranio-ID框架为法医颅面识别和其他生物医学应用提供了一个更可靠、更准确的解决方案，特别是在跨模态匹配方面。

### 翻译

在法医颅面识别和许多生物医学应用中，颅骨测量标志点很重要。定位标志点的传统方法耗时且需要专业知识和技能。当前使用的方法包括叠加和基于深度学习的自动标注方法，但这些方法因缺乏大规模验证研究而不够可靠。在本文中，我们提出了一个名为Cranio-ID的新框架：首先，使用我们训练的YOLO-pose模型对2D颅骨（面部X光扫描）及其相应的光学图像上的标志点进行自动标注。其次，通过将这些标志点转化为图表示，然后使用跨注意力和最优传输框架在这两种模态的图之间找到语义对应关系来实现跨模态匹配。我们在S2F和CUHK数据集上验证了所提出的框架（CUHK数据集类似于S2F数据集）。进行了大量实验来评估所提出框架的性能，这证明了它在可靠性和准确性方面的显著改进，以及在法医科学中跨域颅骨到面部和素描到面部匹配的有效性。


### 论文摘要

In forensic craniofacial identification and in many biomedical applications, craniometric landmarks are important. Traditional methods for locating landmarks are time-consuming and require specialized knowledge and expertise. Current methods utilize superimposition and deep learning-based methods that employ automatic annotation of landmarks. However, these methods are not reliable due to insufficient large-scale validation studies. In this paper, we proposed a novel framework Cranio-ID: First, an automatic annotation of landmarks on 2D skulls (which are X-ray scans of faces) with their respective optical images using our trained YOLO-pose models. Second, cross-modal matching by formulating these landmarks into graph representations and then finding semantic correspondence between graphs of these two modalities using cross-attention and optimal transport framework. Our proposed framework is validated on the S2F and CUHK datasets (CUHK dataset resembles with S2F dataset). Extensive experiments have been conducted to evaluate the performance of our proposed framework, which demonstrates significant improvements in both reliability and accuracy, as well as its effectiveness in cross-domain skull-to-face and sketch-to-face matching in forensic science.

---

## 15. Jasper-Token-Compression-600M Technical Report

**论文链接:** [http://arxiv.org/abs/2511.14405v1](http://arxiv.org/abs/2511.14405v1)

**作者:** Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu

**发布时间:** 2025-11-18

**备注:** 10 pages, 1 figure

### GPT解析

### 总结

这篇技术报告介绍了开源的Jasper-Token-Compression-600M模型的训练方法和评估结果。该模型基于之前的蒸馏方法扩展到双语领域，引入了一维卷积的token压缩模块，通过动态调整压缩率提高了模型性能，实现了比传统0.6B模型更高的效率，同时达到与8B模型相当的性能。

### 背景

研究基于之前基于蒸馏的英语Stella和Jasper模型的配方，旨在将这种方法扩展到双语领域。

### 目的

成功将基于蒸馏的模型配方扩展到双语（英语和中文）领域，并增强模型性能，提高推理效率。

### 方法

基于之前的蒸馏方法进行扩展，引入一维卷积的token压缩模块，在训练过程中动态调整压缩率，结合知识蒸馏和token压缩技术，整合对比学习增强模型性能。

### 主要发现

成功将模型扩展到双语领域，通过对比学习增强了模型性能，token压缩模块使模型能够学习更稳健和高效的压缩文本表示，模型效率高于传统的0.6B模型，模型性能可与8B模型相媲美。

### 结论

通过结合知识蒸馏和token压缩技术，模型在保持高性能的同时显著提高了推理效率，证明了这种方法的可行性和有效性。

### 翻译

这份技术报告介绍了开源的Jasper-Token-Compression-600M模型的训练方法和评估结果，该模型于2025年11月发布。基于之前基于蒸馏的英语Stella和Jasper模型的配方，我们成功将这种方法扩展到双语（英语和中文）领域，并通过整合对比学习进一步增强了模型性能。我们模型的一个关键创新是引入了一维卷积的token压缩模块。我们在训练过程中动态调整压缩率，使模型能够学习更稳健和高效的压缩文本表示。通过结合知识蒸馏和token压缩技术，我们在嵌入质量和推理效率方面都取得了显著改进。我们的模型效率高于传统的0.6B模型，同时性能可与8B模型相媲美。有关模型发布的信息，请访问：https://huggingface.co/infgrad/Jasper-Token-Compression-600M。


### 论文摘要

This technical report presents the training methodology and evaluation results of the open-source Jasper-Token-Compression-600M model, released in November 2025. Building on previous distillation-based recipes from the English Stella and Jasper models, we successfully extend this approach to a bilingual (English and Chinese) domain, further enhancing model performance through the incorporation of contrastive learning. A key innovation of our model is the introduction of a one-dimensional convolution-based token compression module. We dynamically adjust the compression rate during training, enabling the model to learn more robust and efficient compressed text representations. By combining knowledge distillation with token compression techniques, we achieve significant improvements in both embedding quality and inference efficiency. Our model performs with higher efficiency than a traditional 0.6B model while achieving performance comparable to that of an 8B model. For more information on the model release, visit: https://huggingface.co/infgrad/Jasper-Token-Compression-600M.

---

## 16. Infer As You Train: A Symmetric Paradigm of Masked Generative for Click-Through Rate Prediction

**论文链接:** [http://arxiv.org/abs/2511.14403v1](http://arxiv.org/abs/2511.14403v1)

**作者:** Moyu Zhang, Yujun Jin, Yun Chen, Jinxin Hu, Yu Zhang, Xiaoyi Zeng

**发布时间:** 2025-11-18

**备注:** 4 pages, 4 tables, 1 figure

### GPT解析

### 总结

本文提出了一种名为SGCTR的对称掩码生成范式，解决了CTR预测中生成模型在训练和推理阶段的不对称问题，通过在推理阶段也应用生成能力来迭代重新定义输入样本特征，从而减轻噪声特征影响并提高预测准确性。

### 背景

生成模型越来越多地被用于CTR预测领域以克服传统判别范式的局限性，但现有生成模型通常只在训练阶段使用生成范式，主要用于表示学习。

### 目的

解决现有生成模型在训练和推理阶段之间的不对称性问题，使生成范式能够在两个阶段都发挥作用，从而释放其全部潜力。

### 方法

提出SGCTR框架，在训练阶段通过学习特征依赖关系获取生成能力，在在线推理阶段应用这些生成能力迭代地重新定义输入样本的特征，减轻噪声特征影响。

### 主要发现

大量实验验证了SGCTR的优越性，证明在训练和推理中对称地应用生成范式显著释放了CTR预测中生成范式的力量。

### 结论

通过建立训练和推理阶段之间的对称性，SGCTR框架能够充分发挥生成模型在CTR预测中的潜力，提高预测准确性。

### 翻译

生成模型越来越多地被探索用于点击率(CTR)预测领域，以克服传统判别范式的局限性，后者依赖于简单的二元分类目标。然而，现有生成模型通常将生成范式限制在训练阶段，主要用于表示学习。在在线推理期间，它们回归到标准判别范式，无法利用其强大的生成能力进一步提高预测准确性。这种训练和推理阶段之间的基本不对称性阻碍了生成范式实现其全部潜力。为解决这一局限性，我们提出了CTR预测的对称掩码生成范式(SGCTR)，这是一个新框架，在训练和推理阶段之间建立对称性。具体而言，在训练期间通过学习特征依赖关系获取生成能力后，SGCTR在在线推理期间应用这些生成能力迭代地重新定义输入样本的特征，从而减轻噪声特征的影响并提高预测准确性。大量实验验证了SGCTR的优越性，证明在训练和推理中对称地应用生成范式显著释放了CTR预测中生成范式的力量。


### 论文摘要

Generative models are increasingly being explored in click-through rate (CTR) prediction field to overcome the limitations of the conventional discriminative paradigm, which rely on a simple binary classification objective. However, existing generative models typically confine the generative paradigm to the training phase, primarily for representation learning. During online inference, they revert to a standard discriminative paradigm, failing to leverage their powerful generative capabilities to further improve prediction accuracy. This fundamental asymmetry between the training and inference phases prevents the generative paradigm from realizing its full potential. To address this limitation, we propose the Symmetric Masked Generative Paradigm for CTR prediction (SGCTR), a novel framework that establishes symmetry between the training and inference phases. Specifically, after acquiring generative capabilities by learning feature dependencies during training, SGCTR applies the generative capabilities during online inference to iteratively redefine the features of input samples, which mitigates the impact of noisy features and enhances prediction accuracy. Extensive experiments validate the superiority of SGCTR, demonstrating that applying the generative paradigm symmetrically across both training and inference significantly unlocks its power in CTR prediction.

---

## 17. Language as an Anchor: Preserving Relative Visual Geometry for Domain Incremental Learning

**论文链接:** [http://arxiv.org/abs/2511.14401v1](http://arxiv.org/abs/2511.14401v1)

**作者:** Shuyi Geng, Tao Zhou, Yi Zhou

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出LAVA（Language-Anchored Visual Alignment）框架，通过基于文本的参考锚点驱动的相对对齐替代直接特征对齐，解决领域增量学习中的知识保留问题，使各领域视觉表示保持一致的相对几何结构，实现跨领域的类感知先验知识检索和鲁棒特征聚合。

### 背景

领域增量学习(DIL)面临的关键挑战是在不断变化的分布下持续学习，同时保留之前领域的知识。现有方法存在一个基本困境：一方面，将所有领域投影到单一统一的视觉空间会导致域间干扰和语义扭曲；另一方面，隔离特定于领域的参数会导致知识碎片化，形成'知识孤岛'，阻碍知识重用并加剧遗忘。

### 目的

解决领域增量学习中的知识保留问题，设计一种新框架，能够在不断学习新领域的同时有效保留之前学到的知识，避免域间干扰和知识碎片化。

### 方法

提出LAVA（Language-Anchored Visual Alignment）框架，用基于文本的参考锚点驱动的相对对齐替代直接特征对齐。LAVA引导每个新领域的视觉表示保持一致的相对几何结构，该结构通过镜像类别名称之间的成对语义相似性来定义。这种锚定的几何结构作为跨领域的桥梁，实现类感知先验知识的检索和鲁棒特征聚合。

### 主要发现

在标准DIL基准上的大量实验表明，LAVA相较于现有最先进方法取得了显著的性能提升。

### 结论

LAVA框架通过引入基于文本的参考锚点和相对几何对齐策略，有效解决了领域增量学习中的知识保留问题，避免了现有方法的困境，显著提升了性能。

### 翻译

领域增量学习(DIL)中的一个关键挑战是在不断变化的分布下持续学习，同时保留来自先前领域的知识。现有方法面临一个基本困境。一方面，将所有领域投影到单一统一的视觉空间会导致域间干扰和语义扭曲，因为大的变化可能不仅涉及视觉外观，还涉及底层语义。另一方面，隔离特定于领域的参数会导致知识碎片化，形成阻碍知识重用并加剧遗忘的'知识孤岛'。为解决这一问题，我们提出了LAVA（Language-Anchored Visual Alignment），一种新的DIL框架，它用基于文本的参考锚点驱动的相对对齐替代直接特征对齐。LAVA引导每个新领域的视觉表示保持一致的相对几何结构，该结构通过镜像类别名称之间的成对语义相似性来定义。这种锚定的几何结构作为跨领域的桥梁，使类感知先验知识的检索和鲁棒特征聚合成为可能。在标准DIL基准上的大量实验表明，LAVA相较于现有最先进方法取得了显著的性能提升。代码可在https://github.com/ShuyiGeng/LAVA获取。


### 论文摘要

A key challenge in Domain Incremental Learning (DIL) is to continually learn under shifting distributions while preserving knowledge from previous domains. Existing methods face a fundamental dilemma. On one hand, projecting all domains into a single unified visual space leads to inter-domain interference and semantic distortion, as large shifts may vary with not only visual appearance but also underlying semantics. On the other hand, isolating domain-specific parameters causes knowledge fragmentation, creating "knowledge islands" that hamper knowledge reuse and exacerbate forgetting. To address this issue, we propose LAVA (Language-Anchored Visual Alignment), a novel DIL framework that replaces direct feature alignment with relative alignment driven by a text-based reference anchor. LAVA guides the visual representations of each incoming domain to preserve a consistent relative geometry, which is defined by mirroring the pairwise semantic similarities between the class names. This anchored geometric structure acts as a bridge across domains, enabling the retrieval of class-aware prior knowledge and facilitating robust feature aggregation. Extensive experiments on standard DIL benchmarks demonstrate that LAVA achieves significant performance improvements over state-of-the-arts. Code is available at https://github.com/ShuyiGeng/LAVA.

---

## 18. Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning

**论文链接:** [http://arxiv.org/abs/2511.14396v1](http://arxiv.org/abs/2511.14396v1)

**作者:** Xiuxiu Qi, Yu Yang, Jiannong Cao, Luyao Bai, Chongshan Fan, Chengtai Cao, Hongpeng Wang

**发布时间:** 2025-11-18

**备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/

### GPT解析

### 总结

本研究提出了一种名为CCoL的新型BC框架，通过连续视觉-语言-动作协同学习与语义-物理对齐，解决了行为克隆中的累积误差问题，提高了人机交互的性能。

### 背景

语言条件操作通过行为克隆促进人机交互，BC从人类演示中学习控制策略，是具身人工智能的基石。然而，在顺序动作决策中克服累积误差是提高BC性能的主要挑战。

### 目的

开发一种新型BC框架，确保时间一致的执行和细粒度的语义基础，克服现有方法中的物理不连续性和语义-物理不匹配问题。

### 方法

提出CCoL框架，通过跨视觉、语言和本体感受输入的连续协同学习生成稳健平滑的动作执行轨迹，并利用双向交叉注意力将语言语义锚定在视觉运动表示上，学习动作生成的上下文信息。

### 主要发现

CCoL在三个模拟套件上平均实现了8.0%的相对改进，在人类演示的双臂插入任务中实现了高达19.2%的相对增益。在7自由度机器人上的真实世界测试证实了CCoL在未见过的嘈杂物体状态下的泛化能力。

### 结论

CCoL框架成功克服了行为克隆中的累积误差问题，提高了动作执行的准确性和一致性，为具身人工智能和人机交互领域提供了新的解决方案。

### 翻译

语言条件操作通过行为克隆促进人机交互，BC从人类演示中学习控制策略，是具身人工智能的基石。克服顺序动作决策中的累积误差是提高BC性能的主要挑战。现有方法通过数据增强、表达性表示或时间抽象来减轻累积误差，但它们存在物理不连续性和语义-物理不匹配问题，导致动作克隆不准确和执行间歇性。本文提出了一种名为连续视觉-语言-动作协同学习与语义-物理对齐(CCoL)的新型BC框架，确保时间一致的执行和细粒度的语义基础。它通过跨视觉、语言和本体感受输入的连续协同学习，生成稳健平滑的动作执行轨迹。同时，通过双向交叉注意力将语言语义锚定在视觉运动表示上，学习动作生成的上下文信息，成功克服了语义-物理不匹配问题。大量实验表明，CCoL在三个模拟套件上平均实现了8.0%的相对改进，在人类演示的双臂插入任务中实现了高达19.2%的相对增益。在7自由度机器人上的真实世界测试进一步证实了CCoL在未见过的嘈杂物体状态下的泛化能力。


### 论文摘要

Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.

---

## 19. The Tokenization Bottleneck: How Vocabulary Extension Improves Chemistry Representation Learning in Pretrained Language Models

**论文链接:** [http://arxiv.org/abs/2511.14365v1](http://arxiv.org/abs/2511.14365v1)

**作者:** Prathamesh Kalamkar, Ned Letcher, Meissane Chami, Sahger Lad, Shayan Mohanty, Prasanna Pendse

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种解决大型语言模型在化学应用中分词瓶颈问题的方法，通过扩展词汇表和领域持续预训练，实现了自然语言和分子结构的统一表示，并在多种化学任务中取得了优越性能。

### 背景

大型语言模型在化学领域的应用常受到'分词瓶颈'的阻碍，通用文本的分词器会将化学表示（如SMILES）分割成语义信息不足的子标记，限制了模型性能。

### 目的

解决大型语言模型在化学应用中的分词瓶颈问题，通过统一自然语言和分子结构的表示方式来提升模型性能。

### 方法

提出了一种有原则的方法论，包括有针对性的词汇扩展（在预训练的LLM词汇表中添加化学相关标记），然后在化学领域文本上进行持续预训练以整合新知识。

### 主要发现

通过实证证明了该策略的有效性，显示该方法在多种下游化学任务上表现更优。

### 结论

通过词汇扩展和领域持续预训練的方法可以有效解决大型语言模型在化学应用中的分词瓶颈问题，提升模型在化学任务中的性能。

### 翻译

将大型语言模型应用于化学领域常受到'分词瓶颈'的阻碍，其中针对通用文本调整的分词器倾向于将化学表示（如SMILES）分割成语义信息不足的子标记。本文通过在单一模型中统一自然语言和分子结构的表示，提出了一种有原则的方法论来解决这一瓶颈。我们的方法包括有针对性的词汇扩展——用化学上显著的标记增强预训练LLM的词汇表，然后在化学领域文本上进行持续预训练以整合这些新知识。我们通过实证证明了这一策略的有效性，显示我们的方法在各种下游化学任务上取得了优越性能。


### 论文摘要

The application of large language models (LLMs) to chemistry is frequently hampered by a "tokenization bottleneck", where tokenizers tuned on general-domain text tend to fragment chemical representations such as SMILES into semantically uninformative sub-tokens. This paper introduces a principled methodology to resolve this bottleneck by unifying the representation of natural language and molecular structures within a single model. Our approach involves targeted vocabulary extension-augmenting a pretrained LLM's vocabulary with chemically salient tokens, followed by continued pretraining on chemistry-domain text to integrate this new knowledge. We provide an empirical demonstration of the effectiveness of this strategy, showing that our methodology leads to superior performance on a range of downstream chemical tasks.

---

## 20. MA-SLAM: Active SLAM in Large-Scale Unknown Environment using Map Aware Deep Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.14330v1](http://arxiv.org/abs/2511.14330v1)

**作者:** Yizhen Yin, Yuhua Qi, Dapeng Feng, Hongbo Chen, Hongjun Ma, Jin Wu, Yi Jiang

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种基于深度强化学习的地图感知主动SLAM系统(MA-SLAM)，旨在解决大规模环境中的高效探索挑战。

### 背景

Active SLAM涉及机器人系统运动的战略规划和精确控制，以构建周围环境的高精度表示。当前方法在小规模受控环境中有效，但在大规模多样化环境中面临挑战，包括长时间的探索和次优的发现路径。

### 目的

设计MA-SLAM系统，解决大规模环境中高效探索的挑战，减少探索时间和距离。

### 方法

提出了一种新的结构化地图表示方法，通过离散化空间数据并集成边界点和历史轨迹，简洁有效地封装已访问区域；实现了一个先进的全局规划器，利用远距离目标点优化探索路径；基于深度强化学习构建决策模块。

### 主要发现

在三个模拟环境和真实无人地面车辆(UGV)上的实验结果表明，与最先进的方法相比，MA-SLAM显著减少了探索的时间和距离。

### 结论

MA-SLAM系统能够有效应对大规模环境中的探索挑战，提供更高效的路径规划，提高SLAM系统的性能。

### 翻译

主动同步定位与映射(Active SLAM)涉及机器人系统运动的战略规划和精确控制，以构建周围环境的高精度、全面表示，这一领域在研究界受到广泛关注。当前方法在小规模受控环境中表现出色，但在大规模多样化环境中面临挑战，表现为长时间的探索和次优的发现路径。本文提出了MA-SLAM，一种基于深度强化学习的地图感知主动SLAM系统，旨在解决大规模环境中高效探索的挑战。为实现这一目标，我们提出了一种新的结构化地图表示方法。通过离散化空间数据并集成边界点和历史轨迹，结构化地图简洁有效地封装了已访问区域，从而作为基于深度强化学习的决策模块的输入。在决策模块中，我们没有顺序预测下一个动作步骤，而是实现了一个先进的全局规划器，利用远距离目标点优化探索路径。我们在三个模拟环境中进行了实验，并在真实无人地面车辆(UGV)上部署，结果表明，与最先进的方法相比，我们的方法显著减少了探索的时间和距离。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大规模未知环境中机器人主动SLAM（Active SLAM）的效率问题。当前方法在小规模环境中表现良好，但在大规模复杂环境中面临探索时间长和路径次优的挑战。这个问题在现实应用中非常重要，因为搜索救援、探索任务等场景需要机器人高效地进入未知环境并创建地图，提高探索效率可以减少任务时间、节省能源，并降低机器人在危险环境中的风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有Active SLAM方法的局限性：基于前沿的方法计算资源消耗大，采样方法性能不稳定，基于信息的方法计算需求高，而DRL方法在大规模环境中效率低下。基于这些分析，作者借鉴了深度强化学习（特别是PPO框架）、SLAM技术（如Gmapping算法）和路径规划技术（如A*算法），结合分层探索框架的思想，设计出了MA-SLAM系统。该方法将复杂地图信息压缩为结构化表示，并使用全局规划器优化探索路径，而不是简单地预测下一个动作步骤。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结构化地图表示和深度强化学习相结合，提高机器人在大规模未知环境中的探索效率。整体流程包括：1) SLAM模块收集传感器数据并构建环境地图；2) 结构化地图表示将地图划分为感兴趣区域，进行语义分类，并处理边界点；3) 决策模块使用深度强化学习网络处理结构化地图，输出目标点；4) 动作优化单元将可能位于未知区域的目标点投影到最近的边界点上；5) 路径规划和运动控制模块生成无碰撞轨迹并控制机器人移动；6) 机器人移动后更新地图，重复上述过程直到完成探索。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 分层和轻量级探索框架，将高层决策与低级运动控制解耦；2) 新颖的结构化地图表示，结合离散化地图分区、姿态编码和动态边界点处理；3) 带动作优化的地图感知深度强化学习，输出中间路径点而非原始动作；4) 全局规划器的集成，利用远距离目标点优化探索路径。相比之前的工作，MA-SLAM特别针对大规模复杂环境进行了优化，显著提高了探索效率，采用分层决策机制结合深度强化学习和传统路径规划，并通过动作优化单元提高了决策的鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于地图感知深度强化学习的MA-SLAM系统，通过结构化地图表示和分层决策框架，显著提高了机器人在大规模未知环境中的探索效率，减少了探索时间和路径长度。'}


### 论文摘要

Active Simultaneous Localization and Mapping (Active SLAM) involves the strategic planning and precise control of a robotic system's movement in order to construct a highly accurate and comprehensive representation of its surrounding environment, which has garnered significant attention within the research community. While the current methods demonstrate efficacy in small and controlled settings, they face challenges when applied to large-scale and diverse environments, marked by extended periods of exploration and suboptimal paths of discovery. In this paper, we propose MA-SLAM, a Map-Aware Active SLAM system based on Deep Reinforcement Learning (DRL), designed to address the challenge of efficient exploration in large-scale environments. In pursuit of this objective, we put forward a novel structured map representation. By discretizing the spatial data and integrating the boundary points and the historical trajectory, the structured map succinctly and effectively encapsulates the visited regions, thereby serving as input for the deep reinforcement learning based decision module. Instead of sequentially predicting the next action step within the decision module, we have implemented an advanced global planner to optimize the exploration path by leveraging long-range target points. We conducted experiments in three simulation environments and deployed in a real unmanned ground vehicle (UGV), the results demonstrate that our approach significantly reduces both the duration and distance of exploration compared with state-of-the-art methods.

---

## 21. GEN3D: Generating Domain-Free 3D Scenes from a Single Image

**论文链接:** [http://arxiv.org/abs/2511.14291v1](http://arxiv.org/abs/2511.14291v1)

**作者:** Yuxin Zhang, Ziyu Lu, Hongbo Duan, Keyu Fan, Pengting Luo, Peiyu Zhuang, Mengyu Yang, Houde Liu

**发布时间:** 2025-11-18

**备注:** 5 pages , 2 figures

### GPT解析

### 总结

本文提出了Gen3d方法，可以从单张图像生成高质量、广泛范围和通用的3D场景，解决了神经3D重建依赖密集多视角捕获的限制问题

### 背景

神经3D重建的最新进展仍依赖于密集的多视角捕获，限制了其广泛应用；3D场景生成对推进具身AI和世界模型至关重要，这些模型依赖于多样化、高质量的场景进行学习和评估

### 目的

提出一种新颖的方法，可以从单张图像生成高质量、广泛范围和通用的3D场景

### 方法

通过提升RGBD图像创建初始点云，Gen3d维护并扩展其世界模型，最后通过优化高斯飞溅表示来完成3D场景的生成

### 主要发现

在多样化数据集上的大量实验证明了该方法具有强大的泛化能力，在生成世界模型和合成高保真度且一致的全新视图方面表现出优越性能

### 结论

Gen3d方法能够有效解决神经3D重建对密集多视角捕获的依赖问题，为具身AI和世界模型的训练和评估提供高质量的3D场景

### 翻译

尽管神经3D重建最近取得了进展，但它们对密集多视角捕获的依赖限制了其更广泛的应用。此外，3D场景生成对于推进具身AI和世界模型至关重要，这些模型依赖于多样化、高质量的场景进行学习和评估。在这项工作中，我们提出了Gen3d，一种从单张图像生成高质量、广泛范围和通用3D场景的新方法。在通过提升RGBD图像创建初始点云后，Gen3d维护并扩展其世界模型。3D场景通过优化高斯飞溅表示来完成。在多样化数据集上的大量实验证明了我们的方法在生成世界模型和合成高保真度且一致的全新视图方面具有强大的泛化能力和优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单张图像生成高质量3D场景的问题。这个问题很重要，因为传统3D重建方法需要密集的多视图图像，限制了应用范围；同时，3D场景生成对具身AI和世界模型的发展至关重要，这些模型需要多样化高质量场景进行学习和评估。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到几何重建方法在未见视角会产生伪影，而视频生成方法缺乏几何一致性。现有方法要么局限于室内场景，要么专注于室外场景，缺乏通用性。作者借鉴了Stable Diffusion进行图像合成，3D高斯泼溅作为3D表示方法，深度感知的SAM模型进行图像分割，以及Lama-ControlNet进行图像修复等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是采用分层策略，将输入图像分解为前景和背景层，通过移动相机并利用图像修复技术逐步扩展点云，最终使用3D高斯泼溅优化表示。流程包括：1)使用深度感知SAM分割前景和背景；2)对背景进行文本条件修复；3)从初始RGBD图像生成点云；4)沿相机轨迹移动，在每个位置投影点云、修复缺失区域、将新图像提升回3D空间并合并点云；5)使用3D高斯泼溅优化最终场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出领域无关的高质量3D场景生成方法GEN3D；2)采用分层策略构建点云作为几何先验；3)支持多种输入类型；4)利用Stable Diffusion和深度估计实现更好泛化。相比之前工作，GEN3D减少了未见视角的伪影，提供了更好的几何一致性，在室内外场景中都有更好表现，在WorldScore基准测试中取得了最高分。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GEN3D通过结合Stable Diffusion、3D高斯泼溅和分层点云生成策略，实现了从单张图像或文本提示生成高质量、领域无关且具有几何一致性的3D场景，解决了传统方法对多视图输入的依赖和场景生成泛化能力有限的问题。'}


### 论文摘要

Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability. Additionally, 3D scene generation is vital for advancing embodied AI and world models, which depend on diverse, high-quality scenes for learning and evaluation. In this work, we propose Gen3d, a novel method for generation of high-quality, wide-scope, and generic 3D scenes from a single image. After the initial point cloud is created by lifting the RGBD image, Gen3d maintains and expands its world model. The 3D scene is finalized through optimizing a Gaussian splatting representation. Extensive experiments on diverse datasets demonstrate the strong generalization capability and superior performance of our method in generating a world model and Synthesizing high-fidelity and consistent novel views.

---

## 22. NeuralBoneReg: A Novel Self-Supervised Method for Robust and Accurate Multi-Modal Bone Surface Registration

**论文链接:** [http://arxiv.org/abs/2511.14286v1](http://arxiv.org/abs/2511.14286v1)

**作者:** Luohong Wu, Matthias Seibold, Nicola A. Cavalcanti, Yunke Ao, Roman Flepp, Aidana Massalimova, Lilian Calvet, Philipp Fürnstahl

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种名为NeuralBoneReg的自监督、基于表面的骨表面配准框架，用于解决计算机和机器人辅助骨科手术中的跨模态配准挑战。

### 背景

在计算机和机器人辅助骨科手术中，患者特定的手术计划需要从术前成像转移到术中，但不同成像模态之间的异质性使得这种配准具有挑战性且容易出错。

### 目的

开发一种稳健、自动且模态无关的骨表面配准方法，以实现术前和术中数据之间的精确交叉配准。

### 方法

NeuralBoneReg是一个自监督、基于表面的框架，使用3D点云作为模态无关表示。它包含两个模块：一个隐式神经无符号距离场(UDF)用于学习术前骨模型，以及一个基于MLP的配准模块，通过生成变换假设执行全局初始化和局部细化。

### 主要发现

NeuralBoneReg在多个公开数据集上匹配或超过了现有方法，在UltraBones100k上平均RRE/RTE为1.68度/1.86毫米，在UltraBones-Hip上为1.88度/1.89毫米，在SpineDepth上为3.79度/2.45毫米。

### 结论

NeuralBoneReg展示了跨解剖结构和模态的强泛化能力，为CAOS提供了稳健且准确的跨模态对齐。

### 翻译

在计算机和机器人辅助骨科手术中，从术前成像推导的患者特定手术计划定义了目标位置和植入轨迹。在手术过程中，这些计划必须被准确转移，依赖于术前和术中数据之间的精确交叉配准。然而，不同成像模态之间的实质性异质性使得这种配准具有挑战性且容易出错。因此，稳健、自动且模态无关的骨表面配准在临床上非常重要。我们提出了NeuralBoneReg，一种自监督的基于表面的框架，它使用3D点云作为模态无关表示来配准骨表面。NeuralBoneReg包含两个模块：一个隐式神经无符号距离场(UDF)，用于学习术前骨模型；以及一个基于MLP的配准模块，通过生成变换假设来执行全局初始化和局部细化，以将术中点云与神经UDF对齐。与最先进的监督方法不同，NeuralBoneReg以自监督方式运行，不需要跨主题训练数据。我们在两个公开可用的多模态数据集上评估了NeuralBoneReg：一个包含腓骨和胫骨的CT-超声数据集(UltraBones100k)和一个包含脊柱椎骨的CT-RGB-D数据集(SpineDepth)。评估还包括一个新引入的包含股骨和骨盆的尸体受试者CT-超声数据集(UltraBones-Hip)，该数据集将公开提供。NeuralBoneReg在所有数据集上都匹配或超过了现有方法，在UltraBones100k上平均RRE/RTE为1.68度/1.86毫米，在UltraBones-Hip上为1.88度/1.89毫米，在SpineDepth上为3.79度/2.45毫米。这些结果展示了跨解剖结构和模态的强泛化能力，为CAOS提供了稳健且准确的跨模态对齐。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决计算机和机器人辅助骨科手术（CAOS）中，术前和术中数据的精确配准问题。具体来说，就是如何实现不同成像模态（如CT与超声、RGB-D等）之间的骨表面自动配准。这个问题非常重要，因为手术计划的精确执行依赖于术前计划到术中环境的准确转移，而不同成像设备间的差异导致配准过程具有挑战性且容易出错，直接影响手术导航精度、器械引导和手术结果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有配准方法的局限性，包括基于图像的方法在鲁棒性和泛化性上的不足，以及基于表面的方法面临的点云噪声、分辨率差异和表面不完整等挑战。作者借鉴了隐式神经表示（INR）来建模骨几何形状，参考了UltraBoneUDF方法学习无符号距离场（UDF），并受到Bishop混合密度网络的启发设计了并行假设生成机制。整体设计思路是创建一个两阶段框架：术前学习神经UDF表示，术中使用并行假设探索机制来避免局部最优解，同时采用自监督学习减少对大量标注数据的依赖。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用隐式神经表示学习骨表面的连续距离场，并通过并行假设生成机制探索变换空间来解决骨解剖的几何对称性问题。整体流程分为两阶段：1）术前阶段：从CT/MRI等术前数据提取骨表面点云，训练NeuralUDF模块学习神经无符号距离场（UDF），该网络能预测任意点到骨表面的距离；2）术中阶段：从超声/RGB-D等术中数据提取部分骨表面点云，使用NeuralReg模块生成多个变换假设，每个假设由共享主干和多个独立头组成，评估各假设下变换后点云在预训练UDF中的平均距离，选择最优解作为最终配准结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）首个基于隐式神经表示的自监督多模态骨表面配准方法，无需配对训练数据；2）NeuralUDF模块实现连续可微的距离计算，将计算负载转移到术前；3）并行假设生成机制通过共享主干探索SE(3)变换空间，解决几何模糊性问题；4）跨三个不同数据集的综合评估和消融研究；5）发布新的UltraBones-Hip数据集。相比传统方法，它不依赖手工特征且对噪声更鲁棒；相比监督深度学习，它无需大量标注数据且泛化性更好；相比其他自监督方法，它无需大型预训练数据集且结合了隐式表示与并行探索的优势。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NeuralBoneReg通过结合隐式神经表示和并行假设生成，提出了一种自监督框架，实现了跨不同成像模态和骨解剖结构的鲁棒、准确骨表面配准，解决了计算机辅助骨科手术中术前-术中数据配准的关键挑战。'}


### 论文摘要

In computer- and robot-assisted orthopedic surgery (CAOS), patient-specific surgical plans derived from preoperative imaging define target locations and implant trajectories. During surgery, these plans must be accurately transferred, relying on precise cross-registration between preoperative and intraoperative data. However, substantial modality heterogeneity across imaging modalities makes this registration challenging and error-prone. Robust, automatic, and modality-agnostic bone surface registration is therefore clinically important. We propose NeuralBoneReg, a self-supervised, surface-based framework that registers bone surfaces using 3D point clouds as a modality-agnostic representation. NeuralBoneReg includes two modules: an implicit neural unsigned distance field (UDF) that learns the preoperative bone model, and an MLP-based registration module that performs global initialization and local refinement by generating transformation hypotheses to align the intraoperative point cloud with the neural UDF. Unlike SOTA supervised methods, NeuralBoneReg operates in a self-supervised manner, without requiring inter-subject training data. We evaluated NeuralBoneReg against baseline methods on two publicly available multi-modal datasets: a CT-ultrasound dataset of the fibula and tibia (UltraBones100k) and a CT-RGB-D dataset of spinal vertebrae (SpineDepth). The evaluation also includes a newly introduced CT--ultrasound dataset of cadaveric subjects containing femur and pelvis (UltraBones-Hip), which will be made publicly available. NeuralBoneReg matches or surpasses existing methods across all datasets, achieving mean RRE/RTE of 1.68°/1.86 mm on UltraBones100k, 1.88°/1.89 mm on UltraBones-Hip, and 3.79°/2.45 mm on SpineDepth. These results demonstrate strong generalizability across anatomies and modalities, providing robust and accurate cross-modal alignment for CAOS.

---

## 23. Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning

**论文链接:** [http://arxiv.org/abs/2511.14279v1](http://arxiv.org/abs/2511.14279v1)

**作者:** Tong Zhang, Yifan Zhao, Liangyu Wang, Jia Li

**发布时间:** 2025-11-18

**备注:** Accepted to IJCV 2025

### GPT解析

### 总结

该研究提出了一种跨域少样本学习方法，通过构建中间域代理(IDP)来解决语义不连续、域差异大和数据稀缺三大挑战，并利用中间域的视觉风格和语义内容属性进行快速域对齐，实现了在8个基准测试上超越最先进模型的性能。

### 背景

跨域少样本学习(CDFSL)旨在仅使用少量训练数据将源域的泛化知识迁移到目标域，但面临语义不连续、域差异大和数据稀缺三重挑战。

### 目的

解决跨域少样本学习中的三大挑战，提高模型在目标域上的性能。

### 方法

构建中间域代理(IDP)，使用源特征嵌入作为码本重建目标域特征；研究中间域代理的视觉风格和语义内容属性；开发基于中间域属性的快速域对齐方法；实现中间域重建与目标特征变换的协同学习。

### 主要发现

中间域代理具有可利用的视觉风格和语义内容属性，这些属性可以作为学习指导用于目标域特征变换。

### 结论

通过中间域重建和目标特征变换的协同学习，所提出的模型在8个跨域少样本学习基准测试上显著超越了最先进模型。

### 翻译

跨域少样本学习(CDFSL)致力于仅使用少量训练数据将源域的泛化知识迁移到目标域，同时面临三重学习挑战，即语义不连续、大的域差异和数据稀缺。与专注于泛化表示的主流CDFSL工作不同，我们尝试构建中间域代理(IDP)，使用源特征嵌入作为码本，并用学习到的码本重建目标域特征。然后，我们从视觉风格和语义内容的角度对中间域代理的内在属性进行了实证研究。从这些中间域属性中获益，我们开发了一种快速域对齐方法，使用这些代理作为目标域特征变换的学习指导。通过中间域重建和目标特征变换的协同学习，我们提出的模型在8个跨域少样本学习基准测试上以显著优势超越了最先进的模型。


### 论文摘要

Cross-Domain Few-Shot Learning (CDFSL) endeavors to transfer generalized knowledge from the source domain to target domains using only a minimal amount of training data, which faces a triplet of learning challenges in the meantime, i.e., semantic disjoint, large domain discrepancy, and data scarcity. Different from predominant CDFSL works focused on generalized representations, we make novel attempts to construct Intermediate Domain Proxies (IDP) with source feature embeddings as the codebook and reconstruct the target domain feature with this learned codebook. We then conduct an empirical study to explore the intrinsic attributes from perspectives of visual styles and semantic contents in intermediate domain proxies. Reaping benefits from these attributes of intermediate domains, we develop a fast domain alignment method to use these proxies as learning guidance for target domain feature transformation. With the collaborative learning of intermediate domain reconstruction and target feature transformation, our proposed model is able to surpass the state-of-the-art models by a margin on 8 cross-domain few-shot learning benchmarks.

---

## 24. Comparing Task-Agnostic Embedding Models for Tabular Data

**论文链接:** [http://arxiv.org/abs/2511.14276v1](http://arxiv.org/abs/2511.14276v1)

**作者:** Frederik Hoppe, Lars Kleinemeier, Astrid Franz, Udo Göbel

**发布时间:** 2025-11-18

**备注:** Accepted at AI for Tabular Data (EurIPS 2025 Workshop)

### GPT解析

### 总结

该研究评估了表格基础模型的任务无关表示与经典特征工程在不同应用任务中的性能表现。

### 背景

最近的表格数据基础模型通过上下文学习实现了强大的任务特定性能，但它们将表示学习和任务特定推理封装在一个资源密集型的单一网络中。

### 目的

专门关注表示学习即可转移的、任务无关的嵌入，而非直接预测。

### 方法

系统地评估了表格基础模型（TabPFN和TabICL）的任务无关表示以及经典特征工程（TableVectorizer）在异常检测（ADBench）和监督学习（TabArena Lite）等任务中的表现。

### 主要发现

简单的TableVectorizer特征实现了相当或更好的性能，同时比表格基础模型快三个数量级。

### 结论

传统特征工程方法在性能和效率上可能优于复杂的表格基础模型。

### 翻译

最近的表格数据基础模型通过上下文学习在特定任务上实现了强大的性能。然而，它们专注于直接预测，将表示学习和任务特定推理封装在一个资源密集型的单一网络中。这项工作专门关注表示学习即可转移的、任务无关的嵌入。我们系统地评估了来自表格基础模型（TabPFN和TabICL）的任务无关表示以及经典特征工程（TableVectorizer）在各种应用任务中的表现，如异常检测（ADBench）和监督学习（TabArena Lite）。我们发现简单的TableVectorizer特征实现了相当或更好的性能，同时比表格基础模型快三个数量级。代码可在https://github.com/ContactSoftwareAI/TabEmbedBench获取。


### 论文摘要

Recent foundation models for tabular data achieve strong task-specific performance via in-context learning. Nevertheless, they focus on direct prediction by encapsulating both representation learning and task-specific inference inside a single, resource-intensive network. This work specifically focuses on representation learning, i.e., on transferable, task-agnostic embeddings. We systematically evaluate task-agnostic representations from tabular foundation models (TabPFN and TabICL) alongside with classical feature engineering (TableVectorizer) across a variety of application tasks as outlier detection (ADBench) and supervised learning (TabArena Lite). We find that simple TableVectorizer features achieve comparable or superior performance while being up to three orders of magnitude faster than tabular foundation models. The code is available at https://github.com/ContactSoftwareAI/TabEmbedBench.

---

## 25. Algebraformer: A Neural Approach to Linear Systems

**论文链接:** [http://arxiv.org/abs/2511.14263v1](http://arxiv.org/abs/2511.14263v1)

**作者:** Pietro Sittoni, Francesco Tudisco

**发布时间:** 2025-11-18

### GPT解析

### 总结

这是一项关于使用深度学习解决线性系统，特别是病态系统的研究。作者提出了Algebraformer模型，一种基于Transformer的架构，能够端到端地学习解决线性系统，并在应用驱动的线性问题上展示了其有效性。

### 背景

深度学习在解决经典算法任务方面开辟了新可能性。现有的针对病态线性系统的数值方法通常需要仔细的参数调整、预处理或领域专业知识来确保准确性和稳定性。

### 目的

研究解决线性系统的基本任务，特别是那些病态系统。

### 方法

提出Algebraformer，一种基于Transformer的架构，能够端到端地学习解决线性系统，即使在严重病态的情况下也能工作。该模型利用了一种新的编码方案，能够高效表示矩阵和向量输入，内存复杂度随输入规模平方增长，支持可扩展推理。

### 主要发现

在应用驱动的线性问题上展示了其有效性，包括谱方法边界值问题的插值任务和牛顿方法的加速。Algebraformer实现了具有显著更低计算开销的竞争性准确性。

### 结论

表明通用神经架构可以有效地减少传统科学计算流程中的复杂性。

### 翻译

最近的深度学习工作为使用端到端学习的模型解决经典算法任务开辟了新的可能性。在这项工作中，我们研究解决线性系统的基本任务，特别是那些病态系统。现有的针对病态系统的数值方法通常需要仔细的参数调整、预处理或领域专业知识来确保准确性和稳定性。在这项工作中，我们提出了Algebraformer，一种基于Transformer的架构，能够学习端到端地解决线性系统，即使在严重病态的情况下也是如此。我们的模型利用了一种新的编码方案，能够高效表示矩阵和向量输入，内存复杂度随输入规模平方增长，支持可扩展推理。我们在应用驱动的线性问题上展示了其有效性，包括谱方法边界值问题的插值任务和牛顿方法的加速。Algebraformer实现了具有显著更低计算开销的竞争性准确性，表明通用神经架构可以有效地减少传统科学计算流程中的复杂性。


### 论文摘要

Recent work in deep learning has opened new possibilities for solving classical algorithmic tasks using end-to-end learned models. In this work, we investigate the fundamental task of solving linear systems, particularly those that are ill-conditioned. Existing numerical methods for ill-conditioned systems often require careful parameter tuning, preconditioning, or domain-specific expertise to ensure accuracy and stability. In this work, we propose Algebraformer, a Transformer-based architecture that learns to solve linear systems end-to-end, even in the presence of severe ill-conditioning. Our model leverages a novel encoding scheme that enables efficient representation of matrix and vector inputs, with a memory complexity of $O(n^2)$, supporting scalable inference. We demonstrate its effectiveness on application-driven linear problems, including interpolation tasks from spectral methods for boundary value problems and acceleration of the Newton method. Algebraformer achieves competitive accuracy with significantly lower computational overhead at test time, demonstrating that general-purpose neural architectures can effectively reduce complexity in traditional scientific computing pipelines.

---

## 26. Object-Centric World Models for Causality-Aware Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.14262v1](http://arxiv.org/abs/2511.14262v1)

**作者:** Yosuke Nishimoto, Takashi Matsubara

**发布时间:** 2025-11-18

**备注:** Accepted by AAAI-26

### GPT解析

### 总结

本研究提出了STICA框架，该框架使用面向对象的Transformers作为世界模型和因果感知策略与价值网络，在对象丰富的环境中表现出优越性能。

### 背景

世界模型已被开发用于支持样本高效的深度强化学习智能体，但大多数世界模型学习所有环境组件的整体表示，难以准确复制高维度、非静态且由多个具有丰富交互的对象组成的环境。

### 目的

受人类将环境分解为离散物体以便于高效决策的启发，作者提出STICA框架，旨在解决现有世界模型在处理复杂环境时的局限性。

### 方法

STICA框架将每个观察表示为一组面向对象的标记，以及智能体行动和相应结果的标记，使世界模型能够预测标记级别的动态和交互。策略和价值网络估计标记级别的因果关系，并在注意力层中使用它们，实现因果引导的决策制定。

### 主要发现

在对象丰富的基准测试中，STICA在样本效率和最终性能方面都一致地超越了最先进的智能体。

### 结论

STICA框架通过采用面向对象的表示和因果感知的决策机制，有效解决了复杂环境中的世界建模挑战，表现出优越的性能。

### 翻译

世界模型已被开发用于支持样本高效的深度强化学习智能体。然而，对于高维度、非静态且由多个具有丰富交互的对象组成的环境，世界模型仍然难以准确复制，因为大多数世界模型学习所有环境组件的整体表示。相比之下，人类通过将环境分解为离散物体来感知环境，促进高效决策。受此启发，我们提出了STICA（Slot Transformer Imagination with CAusality-aware reinforcement learning）框架，在该框架中，面向对象的Transformers作为世界模型和因果感知策略与价值网络。STICA将每个观察表示为一组面向对象的标记，以及智能体行动和相应结果的标记，使世界模型能够预测标记级别的动态和交互。然后，策略和价值网络估计标记级别的因果关系并在注意力层中使用它们，产生因果引导的决策。在对象丰富的基准测试中，STICA在样本效率和最终性能方面都一致地优于最先进的智能体。


### 论文摘要

World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dynamics and interactions. The policy and value networks then estimate token-level cause--effect relations and use them in the attention layers, yielding causality-guided decision-making. Experiments on object-rich benchmarks demonstrate that STICA consistently outperforms state-of-the-art agents in both sample efficiency and final performance.

---

## 27. Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning

**论文链接:** [http://arxiv.org/abs/2511.14249v1](http://arxiv.org/abs/2511.14249v1)

**作者:** Rui Liu, Yuan Zhao, Zhenqi Jia

**发布时间:** 2025-11-18

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了Authentic-Dubber，一种基于检索增强型导演-演员互动学习的电影配音模型，通过模拟真实电影配音工作流程中的导演-演员互动关系，提高配音的情感表现力。

### 背景

现有的自动电影配音模型模拟了简化的工作流程，让演员直接配音而不做准备，忽略了导演与演员之间的关键互动环节。真实的工作流程中，导演会引导演员理解并内化情感等上下文线索，然后再进行表演。

### 目的

解决现有配音模型中缺失的导演-演员互动问题，提出一种能够模拟真实电影配音工作流程的系统，提高配音的情感表现力。

### 方法

Authentic-Dubber包含三个创新机制：(1)构建多模态参考素材库，整合大型语言模型(LLMs)实现多模态信号中情感表示的深度理解；(2)提出基于情感相似性的检索增强策略，模拟演员内化导演提供素材的过程；(3)开发渐进式图结构语音生成方法，逐步整合检索到的多模态情感知识。

### 主要发现

Authentic-Dubber能够忠实地复制真实的配音工作流程，在情感表达方面取得了全面的改进。在V2C Animation基准数据集上的主观和客观评估验证了该方法的有效性。

### 结论

通过模拟真实的导演-演员互动过程，Authentic-Dubber显著提升了自动配音的情感表现力，代码和演示已公开可供使用。

### 翻译

自动电影配音模型能够根据给定脚本生成生动的语音，从简短的音色提示中复制说话人的音色，同时确保与无声视频的口型同步。现有方法模拟了一个简化的工作流程，即演员直接配音而不做准备，忽略了导演-演员之间的关键互动。相比之下，真实的工作流程涉及动态协作：导演积极参与演员表演，引导他们在表演前内化上下文线索，特别是情感。为解决这一问题，我们提出了一个新的检索增强型导演-演员互动学习方案来实现真实的电影配音，称为Authentic-Dubber，它包含三个新机制：(1)我们构建了一个多模态参考素材库来模拟导演提供的学习素材，我们整合大型语言模型(LLMs)来实现对多模态信号中情感表示的深度理解。(2)为了模拟演员在配音过程中如何高效全面地内化导演提供的素材，我们提出了基于情感相似性的检索增强策略。该策略检索与目标无声视频最相关的多模态信息。(3)我们开发了一种渐进式图结构语音生成方法，逐步整合检索到的多模态情感知识，从而模拟演员的最终配音过程。上述机制使Authentic-Dubber能够忠实地复制真实的配音工作流程，在情感表达方面取得了全面的改进。在V2C Animation基准数据集上的主观和客观评估验证了其有效性。代码和演示可在https://github.com/AI-S2-Lab/Authentic-Dubber获取。


### 论文摘要

The automatic movie dubbing model generates vivid speech from given scripts, replicating a speaker's timbre from a brief timbre prompt while ensuring lip-sync with the silent video. Existing approaches simulate a simplified workflow where actors dub directly without preparation, overlooking the critical director-actor interaction. In contrast, authentic workflows involve a dynamic collaboration: directors actively engage with actors, guiding them to internalize the context cues, specifically emotion, before performance. To address this issue, we propose a new Retrieve-Augmented Director-Actor Interaction Learning scheme to achieve authentic movie dubbing, termed Authentic-Dubber, which contains three novel mechanisms: (1) We construct a multimodal Reference Footage library to simulate the learning footage provided by directors. Note that we integrate Large Language Models (LLMs) to achieve deep comprehension of emotional representations across multimodal signals. (2) To emulate how actors efficiently and comprehensively internalize director-provided footage during dubbing, we propose an Emotion-Similarity-based Retrieval-Augmentation strategy. This strategy retrieves the most relevant multimodal information that aligns with the target silent video. (3) We develop a Progressive Graph-based speech generation approach that incrementally incorporates the retrieved multimodal emotional knowledge, thereby simulating the actor's final dubbing process. The above mechanisms enable the Authentic-Dubber to faithfully replicate the authentic dubbing workflow, achieving comprehensive improvements in emotional expressiveness. Both subjective and objective evaluations on the V2C Animation benchmark dataset validate the effectiveness. The code and demos are available at https://github.com/AI-S2-Lab/Authentic-Dubber.

---

## 28. Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility

**论文链接:** [http://arxiv.org/abs/2511.14248v1](http://arxiv.org/abs/2511.14248v1)

**作者:** Hongju Lee, Youngjun Park, Jisun An, Dongman Lee

**发布时间:** 2025-11-18

**备注:** Accepted at ASONAM 2025

### GPT解析

### 总结

该研究提出了一种新型时间序列预测框架，用于预测区域级别的Airbnb市场指标（收入、预订天数和预订数量）。通过整合列表特征与外部上下文因素，将结构化数据转换为大型语言模型输入，生成区域嵌入并输入到先进时间序列模型中，提高了预测准确性，为城市政策制定提供了实用见解。

### 背景

短期租赁平台（如Airbnb）的扩张显著扰乱了当地住房市场，通常导致租金价格上涨和住房可负担性问题。准确预测区域Airbnb市场趋势可以为政策制定者和城市规划者提供关键见解，以减轻这些影响。

### 目的

开发一种新型时间序列预测框架，用于预测区域级别的三个关键Airbnb指标（收入、预订天数和预订数量），为政策制定者和城市规划者提供决策支持。

### 方法

采用滑动窗口方法，预测未来1至3个月的趋势。通过整合列表特征与外部上下文因素（如城市可达性和人流）构建区域表示。将结构化表格数据转换为基于提示的大型语言模型输入，生成区域嵌入，然后输入到先进的时间序列模型（RNN、LSTM、Transformer）中，以捕捉复杂的时空动态。

### 主要发现

在首尔Airbnb数据集上的实验表明，与传统基线（包括传统统计和机器学习模型）相比，该方法将平均RMSE和MAE降低了约48%。

### 结论

该框架不仅提高了预测准确性，还提供了检测供应过剩区域和支持数据驱动城市政策决策的实际见解。

### 翻译

短期租赁平台（如Airbnb）的扩张显著扰乱了当地住房市场，通常导致租金价格上涨和住房可负担性问题。因此，准确预测区域Airbnb市场趋势可以为旨在减轻这些影响的政策制定者和城市规划者提供关键见解。本研究提出了一种新型时间序列预测框架，用于预测区域级别的三个关键Airbnb指标——收入、预订天数和预订数量。采用滑动窗口方法，模型可预测未来1至3个月的趋势。与专注于固定时间点单个列表的先前研究不同，我们的方法通过整合列表特征与外部上下文因素（如城市可达性和人流）来构建区域表示。我们将结构化表格数据转换为基于提示的大型语言模型输入，生成全面的区域嵌入。然后将这些嵌入输入到先进的时间序列模型（RNN、LSTM、Transformer）中，以更好地捕捉复杂的时空动态。在首尔Airbnb数据集上的实验表明，与传统基线（包括传统统计和机器学习模型）相比，我们的方法将平均RMSE和MAE降低了约48%。我们的框架不仅提高了预测准确性，还为检测供应过剩区域和支持数据驱动的城市政策决策提供了实用见解。


### 论文摘要

The expansion of short-term rental platforms, such as Airbnb, has significantly disrupted local housing markets, often leading to increased rental prices and housing affordability issues. Accurately forecasting regional Airbnb market trends can thus offer critical insights for policymakers and urban planners aiming to mitigate these impacts. This study proposes a novel time-series forecasting framework to predict three key Airbnb indicators -- Revenue, Reservation Days, and Number of Reservations -- at the regional level. Using a sliding-window approach, the model forecasts trends 1 to 3 months ahead. Unlike prior studies that focus on individual listings at fixed time points, our approach constructs regional representations by integrating listing features with external contextual factors such as urban accessibility and human mobility. We convert structured tabular data into prompt-based inputs for a Large Language Model (LLM), producing comprehensive regional embeddings. These embeddings are then fed into advanced time-series models (RNN, LSTM, Transformer) to better capture complex spatio-temporal dynamics. Experiments on Seoul's Airbnb dataset show that our method reduces both average RMSE and MAE by approximately 48% compared to conventional baselines, including traditional statistical and machine learning models. Our framework not only improves forecasting accuracy but also offers practical insights for detecting oversupplied regions and supporting data-driven urban policy decisions.

---

## 29. Breaking the Passive Learning Trap: An Active Perception Strategy for Human Motion Prediction

**论文链接:** [http://arxiv.org/abs/2511.14237v1](http://arxiv.org/abs/2511.14237v1)

**作者:** Juncheng Hu, Zijian Zhang, Zeyu Wang, Guoyu Wang, Yingji Li, Kedi Lyu

**发布时间:** 2025-11-18

**备注:** 8 pages, 3 figures

### GPT解析

### 总结

本文提出了一种主动感知策略(APS)用于3D人体运动预测，通过商空间表示和辅助学习目标解决现有方法过度依赖隐式建模和被动学习的问题，显著提升了预测性能。

### 背景

3D人体运动预测是人工智能对人类行为精细理解和认知的重要体现，但当前方法存在过度依赖隐式网络建模、陷入被动学习陷阱的问题。

### 目的

为了解决现有方法过度依赖隐式建模、获取冗余单调坐标信息、缺乏主动引导学习机制的问题，提出主动感知策略(APS)。

### 方法

利用商空间表示显式编码运动特性，引入辅助学习目标加强时空建模；设计数据感知模块将姿态投影到商空间解耦运动几何与坐标冗余；通过联合编码切向量和Grassmann投影实现几何维度缩减、语义解耦和动态约束执行；引入网络感知模块通过恢复学习主动学习时空依赖关系；故意掩码关节或注入噪声构建辅助监督信号；设计辅助学习网络主动适应和从扰动信息中学习。

### 主要发现

APS方法与模型无关，可集成到不同预测模型中增强主动感知能力；实验结果表明APS取得了新的最先进水平，在H3.6M上超越现有方法16.3%，在CMU Mocap上超越13.9%，在3DPW上超越10.1%。

### 结论

APS通过主动感知策略有效解决了现有3D人体运动预测方法的问题，显著提高了预测性能，为未来研究提供了新思路。

### 翻译

三维人体运动预测是人工智能代理对人类行为进行细粒度理解和认知的重要体现。当前方法过度依赖时空关系和运动特性的隐式网络建模，陷入被动学习陷阱，导致获取冗余和单调的三维坐标信息，同时缺乏主动引导的显式学习机制。为克服这些问题，我们提出了一种用于人体运动预测的主动感知策略(APS)，利用商空间表示来显式编码运动特性，同时引入辅助学习目标来加强时空建模。具体而言，我们首先设计了一个数据感知模块，将姿态投影到商空间，解耦运动几何与坐标冗余。通过联合编码切向量和Grassmann投影，该模块同时实现了几何维度缩减、语义解耦和动态约束执行，从而有效表征运动姿态。此外，我们引入了一个网络感知模块，通过恢复学习主动学习时空依赖关系。该模块故意掩码特定关节或注入噪声来构建辅助监督信号。设计了一个专门的辅助学习网络来主动适应和从扰动信息中学习。值得注意的是，APS与模型无关，可以与不同的预测模型集成以增强主动感知能力。实验结果表明，我们的方法取得了新的最先进水平，大幅超越现有方法：在H3.6M上超越16.3%，在CMU Mocap上超越13.9%，在3DPW上超越10.1%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决3D人体运动预测中的'被动学习陷阱'问题。当前方法过度依赖隐式网络建模时空关系，导致获取的3D坐标信息冗余且单调，缺乏主动引导的学习机制。这个问题很重要，因为准确的人体运动预测在具身智能、人机交互和自动驾驶等领域有广泛应用，而被动学习限制了预测精度，特别是在处理复杂和非周期性运动时。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的两个主要局限(数据感知局限和网络感知局限)来设计解决方案。他们创建了主动感知策略(APS)，包含数据感知模块(DPM)和网络感知模块(NPM)。DPM利用商空间表示解耦运动几何与坐标冗余，NPM通过故意破坏输入数据并要求模型恢复来强制学习鲁棒的时空关系。作者借鉴了商空间表示理论、生成对抗网络训练机制、时空图注意力技术以及现有的姿态预测网络架构，但创新性地将它们整合形成新的主动感知框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是打破被动学习陷阱，通过主动引导的显式学习机制增强模型对时空关系和运动特性的理解。整体流程包括：1)数据感知模块将3D姿态映射到商空间，使用切空间和Grassmann投影捕捉运动特性；2)网络感知模块通过时空增强组件有选择地破坏输入数据(掩盖关节或添加噪声)，然后由时空学习组件使用图注意力机制学习并恢复被破坏的数据；3)通过复合损失函数和WGAN-GP进行模型优化，使预测结果更接近真实运动。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出模型无关的主动感知策略，从被动学习转向主动学习；2)数据感知模块利用商空间表示解耦运动几何与坐标冗余；3)网络感知模块通过故意破坏输入数据构建辅助监督信号；4)APS可与不同预测模型集成。相比之前工作，本文实现了学习范式转变(从'黑盒端到端拟合'到'解耦表示'+'对抗细化')，革新了表示方式(不再直接处理3D坐标)，创新了训练机制(引入辅助学习任务)，并在三个基准数据集上实现了显著性能提升(16.3%-10.1%)。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种主动感知策略，通过商空间表示和对抗训练机制，有效打破了人体运动预测中的被动学习陷阱，显著提升了预测精度，特别是在处理复杂和非周期性运动时。'}


### 论文摘要

Forecasting 3D human motion is an important embodiment of fine-grained understanding and cognition of human behavior by artificial agents. Current approaches excessively rely on implicit network modeling of spatiotemporal relationships and motion characteristics, falling into the passive learning trap that results in redundant and monotonous 3D coordinate information acquisition while lacking actively guided explicit learning mechanisms. To overcome these issues, we propose an Active Perceptual Strategy (APS) for human motion prediction, leveraging quotient space representations to explicitly encode motion properties while introducing auxiliary learning objectives to strengthen spatio-temporal modeling. Specifically, we first design a data perception module that projects poses into the quotient space, decoupling motion geometry from coordinate redundancy. By jointly encoding tangent vectors and Grassmann projections, this module simultaneously achieves geometric dimension reduction, semantic decoupling, and dynamic constraint enforcement for effective motion pose characterization. Furthermore, we introduce a network perception module that actively learns spatio-temporal dependencies through restorative learning. This module deliberately masks specific joints or injects noise to construct auxiliary supervision signals. A dedicated auxiliary learning network is designed to actively adapt and learn from perturbed information. Notably, APS is model agnostic and can be integrated with different prediction models to enhance active perceptual. The experimental results demonstrate that our method achieves the new state-of-the-art, outperforming existing methods by large margins: 16.3% on H3.6M, 13.9% on CMU Mocap, and 10.1% on 3DPW.

---

## 30. FreeMusco: Motion-Free Learning of Latent Control for Morphology-Adaptive Locomotion in Musculoskeletal Characters

**论文链接:** [http://arxiv.org/abs/2511.14205v1](http://arxiv.org/abs/2511.14205v1)

**作者:** Minkwan Kim, Yoonsang Lee

**发布时间:** 2025-11-18

**备注:** SIGGRAPH Asia 2025

### GPT解析

### 总结

FreeMusco是一种无需运动数据的框架，能够联合学习肌肉骨骼角色的潜在表示和控制策略，实现能量感知和形态自适应的运动，能够泛化到不同形态并实现多种下游任务。

### 背景

在肌肉骨骼角色运动控制领域，特别是在没有运动数据的情况下实现自适应运动是一个挑战。

### 目的

提出一种名为FreeMusco的框架，能够在没有运动数据的情况下学习肌肉骨骼角色的潜在表示和控制策略。

### 方法

利用肌肉骨骼模型作为强先验，使用基于模型的强化学习，引入时间平均损失公式捕捉自然步态的周期结构，并通过随机化目标姿势和能量水平来鼓励行为多样性。

### 主要发现

该框架能够实现能量感知和形态自适应的运动，泛化到人类、非人类和合成形态，产生不同的能量高效策略，并实现目标导航和路径跟随等下游任务。

### 结论

FreeMusco展示了一种在没有运动捕捉的情况下实现多功能和自适应运动控制的新方向，为在数据收集不切实际或不可能的情况下模拟角色运动提供了新方法。

### 翻译

我们提出了FreeMusco，一种无需运动数据的框架，能够联合学习肌肉骨骼角色的潜在表示和控制策略。通过利用肌肉骨骼模型作为强先验，我们的方法能够在没有运动数据的情况下实现能量感知和形态自适应的运动。该框架能够泛化到人类、非人类和合成形态，其中不同的能量高效策略自然出现——例如，Chimanoid中的四足步态与Humanoid中的双足步态。潜在空间和相应的控制策略是从零开始构建的，无需演示，并支持目标导航和路径跟随等下游任务——据我们所知，这是首个提供此类能力的无需运动数据的方法。FreeMusco通过基于模型的强化学习学习多样且物理合理的运动行为，由结合了控制、平衡和生物力学项的运动目标引导。为了更好地捕捉自然步态的周期结构，我们引入了时间平均损失公式，它在一个时间窗口内比较模拟状态和目标状态，而非逐帧比较。我们通过在训练过程中随机化目标姿势和能量水平来进一步鼓励行为多样性，使运动在运行时能够在形式和强度上灵活调整。这些结果共同证明，无需运动捕捉也能产生多功能和自适应的运动控制，为在数据收集不切实际或不可能的情况下模拟角色运动提供了新方向。


### 论文摘要

We propose FreeMusco, a motion-free framework that jointly learns latent representations and control policies for musculoskeletal characters. By leveraging the musculoskeletal model as a strong prior, our method enables energy-aware and morphology-adaptive locomotion to emerge without motion data. The framework generalizes across human, non-human, and synthetic morphologies, where distinct energy-efficient strategies naturally appear--for example, quadrupedal gaits in Chimanoid versus bipedal gaits in Humanoid. The latent space and corresponding control policy are constructed from scratch, without demonstration, and enable downstream tasks such as goal navigation and path following--representing, to our knowledge, the first motion-free method to provide such capabilities. FreeMusco learns diverse and physically plausible locomotion behaviors through model-based reinforcement learning, guided by the locomotion objective that combines control, balancing, and biomechanical terms. To better capture the periodic structure of natural gait, we introduce the temporally averaged loss formulation, which compares simulated and target states over a time window rather than on a per-frame basis. We further encourage behavioral diversity by randomizing target poses and energy levels during training, enabling locomotion to be flexibly modulated in both form and intensity at runtime. Together, these results demonstrate that versatile and adaptive locomotion control can emerge without motion capture, offering a new direction for simulating movement in characters where data collection is impractical or impossible.

---

## 31. Hierarchical Semantic Learning for Multi-Class Aorta Segmentation

**论文链接:** [http://arxiv.org/abs/2511.14187v1](http://arxiv.org/abs/2511.14187v1)

**作者:** Pengcheng Shi

**发布时间:** 2025-11-18

**备注:** Accepted by MICCAI 2024 Workshop AortaSeg

### GPT解析

### 总结

本文提出了一种基于课程学习和分形softmax的层次语义学习方法，用于主动脉分割，解决了现有方法忽略解剖层次关系和类别不平衡的问题，显著提高了分割准确性和效率。

### 背景

主动脉作为人体最大的动脉，容易发生夹层、动脉瘤和动脉粥样硬化等病理变化，需要及时干预。涉及分支血管的微创修复需要详细的三维解剖分析。

### 目的

解决现有主动脉分割方法中忽略层次解剖关系和类别不平衡的问题，提高分割准确性和临床实用性。

### 方法

采用课程学习策略，利用新型分形softmax进行层次语义学习，受人类认知启发，通过从简单到复杂的方式分解复杂结构，并使用两阶段推理策略实现加速。

### 主要发现

在验证集上，层次语义损失使nnU-Net ResEnc M的Dice分数提高了11.65%；在测试集上，所提模型比基线模型高出5.6%的Dice分数；两阶段推理策略实现了最多五倍的加速。

### 结论

该框架显著提高了分割准确性和效率，适合实时临床应用，相关代码已在GitHub上公开。

### 翻译

主动脉作为人体最大的动脉，容易发生夹层、动脉瘤和动脉粥样硬化等病理变化，通常需要及时干预。涉及分支血管的微创修复需要详细的三维解剖分析。现有方法往往忽略了解剖层次关系，同时难以处理血管结构中固有的严重类别不平衡问题。我们通过利用新型分形softmax进行层次语义学习的课程学习策略解决了这些挑战。受人类认知启发，我们的方法通过从简单到复杂的方式分解复杂结构，逐步学习解剖约束。课程学习框架通过首先为主导类别建立稳健的特征表示，再处理罕见但解剖学上关键的结构，自然地解决了类别不平衡问题，显著加速了多类别场景中模型的收敛。我们的两阶段推理策略实现了高达五倍的加速，提高了临床实用性。在第50个epoch的验证集上，我们的层次语义损失使nnU-Net ResEnc M的Dice分数提高了11.65%。在测试集上，所提出的模型比基线模型高出5.6%的Dice分数。实验结果表明分割准确性和效率有显著提高，使该框架适合实时临床应用。该挑战的实时代码已在以下公开：https://github.com/PengchengShi1220/AortaSeg24。分形softmax的代码将在https://github.com/PengchengShi1220/fractal-softmax上提供。


### 论文摘要

The aorta, the body's largest artery, is prone to pathologies such as dissection, aneurysm, and atherosclerosis, which often require timely intervention. Minimally invasive repairs involving branch vessels necessitate detailed 3D anatomical analysis. Existing methods often overlook hierarchical anatomical relationships while struggling with severe class imbalance inherent in vascular structures. We address these challenges with a curriculum learning strategy that leverages a novel fractal softmax for hierarchical semantic learning. Inspired by human cognition, our approach progressively learns anatomical constraints by decomposing complex structures from simple to complex components. The curriculum learning framework naturally addresses class imbalance by first establishing robust feature representations for dominant classes before tackling rare but anatomically critical structures, significantly accelerating model convergence in multi-class scenarios. Our two-stage inference strategy achieves up to fivefold acceleration, enhancing clinical practicality. On the validation set at epoch 50, our hierarchical semantic loss improves the Dice score of nnU-Net ResEnc M by 11.65%. The proposed model demonstrates a 5.6% higher Dice score than baselines on the test set. Experimental results show significant improvements in segmentation accuracy and efficiency, making the framework suitable for real-time clinical applications. The implementation code for this challenge entry is publicly available at: https://github.com/PengchengShi1220/AortaSeg24. The code for fractal softmax will be available at https://github.com/PengchengShi1220/fractal-softmax.

---

## 32. PAVE: An End-to-End Dataset for Production Autonomous Vehicle Evaluation

**论文链接:** [http://arxiv.org/abs/2511.14185v1](http://arxiv.org/abs/2511.14185v1)

**作者:** Xiangyu Li, Chen Wang, Yumao Liu, Dengbo He, Jiahao Zhang, Ke Ma

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了第一个完全由自动驾驶模式在真实世界收集的端到端基准数据集，用于评估自动驾驶车辆的真实行为安全性。

### 背景

现有自动驾驶数据集(如KITTI, nuScenes, Waymo)均由人类驾驶模式或未识别驾驶模式收集，仅能作为自动驾驶车辆感知和预测的早期训练。

### 目的

评估黑盒控制下自动驾驶车辆的真实行为安全性。

### 方法

创建并收集了市场上多个量产自动驾驶车型超过100小时的自然驾驶数据，分割成32,727个关键帧，每个包含四个同步摄像头图像和高精度GNSS/IMU数据，提供车辆轨迹和周围环境详细注释，并采用端到端运动规划模型进行安全性评估。

### 主要发现

该数据集包含丰富的场景级属性，如驾驶员意图、区域类型、照明、天气等，端到端运动规划模型在自动驾驶帧上的平均位移误差为1.4米。

### 结论

该数据集每周新增超过10小时数据，为自动驾驶车辆驾驶行为分析和安全评估研究提供了可持续的基础。

### 翻译

大多数现有的自动驾驶数据集(如KITTI, nuScenes和Waymo感知数据集)是通过人类驾驶模式或未识别驾驶模式收集的，仅能作为自动驾驶车辆(AVs)感知和预测的早期训练。为了评估黑盒控制下AVs的真实行为安全性，我们提出了第一个完全由自动驾驶模式在真实世界收集的端到端基准数据集。该数据集包含市场上多个量产自动驾驶车型超过100小时的自然驾驶数据。我们将原始数据分割成32,727个关键帧，每个关键帧包含四个同步的摄像头图像和高精度GNSS/IMU数据(0.8厘米定位精度)。对于每个关键帧，提供过去6秒和未来5秒的20Hz车辆轨迹，以及周围车辆、行人、交通灯和交通标志的详细2D注释。这些关键帧具有丰富的场景级属性，包括驾驶员意图、区域类型(覆盖高速公路、城市道路和居民区)、照明(白天、夜间或黄昏)、天气(晴朗或雨天)、路面(铺砌或未铺砌)、交通和弱势道路用户(VRU)密度、交通灯和交通标志(警告、禁止和指示)。为了评估AVs的安全性，我们采用了一种端到端运动规划模型，在自动驾驶帧上的平均位移误差(ADE)为1.4米。该数据集每周新增超过10小时数据，为AV驾驶行为分析和安全评估研究提供了可持续的基础。


### 论文摘要

Most existing autonomous-driving datasets (e.g., KITTI, nuScenes, and the Waymo Perception Dataset), collected by human-driving mode or unidentified driving mode, can only serve as early training for the perception and prediction of autonomous vehicles (AVs). To evaluate the real behavioral safety of AVs controlled in the black box, we present the first end-to-end benchmark dataset collected entirely by autonomous-driving mode in the real world. This dataset contains over 100 hours of naturalistic data from multiple production autonomous-driving vehicle models in the market. We segment the original data into 32,727 key frames, each consisting of four synchronized camera images and high-precision GNSS/IMU data (0.8 cm localization accuracy). For each key frame, 20 Hz vehicle trajectories spanning the past 6 s and future 5 s are provided, along with detailed 2D annotations of surrounding vehicles, pedestrians, traffic lights, and traffic signs. These key frames have rich scenario-level attributes, including driver intent, area type (covering highways, urban roads, and residential areas), lighting (day, night, or dusk), weather (clear or rain), road surface (paved or unpaved), traffic and vulnerable road users (VRU) density, traffic lights, and traffic signs (warning, prohibition, and indication). To evaluate the safety of AVs, we employ an end-to-end motion planning model that predicts vehicle trajectories with an Average Displacement Error (ADE) of 1.4 m on autonomous-driving frames. The dataset continues to expand by over 10 hours of new data weekly, thereby providing a sustainable foundation for research on AV driving behavior analysis and safety evaluation.

---

## 33. Look-Ahead Reasoning on Learning Platforms

**论文链接:** [http://arxiv.org/abs/2511.14745v1](http://arxiv.org/abs/2511.14745v1)

**作者:** Haiqing Zhu, Tijana Zrnic, Celestine Mendler-Dünner

**发布时间:** 2025-11-18

**备注:** accepted to NeurIPS 2025

### GPT解析

### 总结

这篇论文研究了学习平台上用户的策略性行为，引入了前瞻性推理和集体推理的概念，分析了用户如何通过level-k思考和协调行动来影响平台预测，并探讨了这些行为对均衡结果的影响。

### 背景

学习平台的优化标准通常反映设计者的优先级，而非用户的优先级。用户可能采取策略性行为来获得更有利的结果，挑战平台的预测。过去的研究主要关注用户对已部署模型的策略性响应，而忽略了其他用户行为的影响。

### 目的

研究用户在学习平台上的策略性行为，特别是引入前瞻性推理和集体推理的概念，分析用户如何通过思考未来预测和协调行动来影响平台结果，并探讨这些行为对均衡的影响。

### 方法

作者形式化了level-k思考概念，研究用户如何通过向前看一步来超越同行。同时，研究集体推理，用户通过优化他们对模型的集体影响来协调行动。通过对比集体行为与自私行为，分析协调的益处和限制。

### 主要发现

虽然向均衡收敛的速度加快，但均衡保持不变，高水平思考对个人没有长期好处；集体推理通过协调行动产生新的对齐概念；学习者和用户效用之间的对齐成为关键概念。

### 结论

研究揭示了用户在学习平台上策略性行为的复杂性，特别是前瞻性推理和集体推理的影响。虽然高水平思考加速了均衡收敛，但没有改变均衡结果。集体行动引入了新的对齐概念，这可能是未来研究的重要方向。

### 翻译

在许多学习平台上，指导模型训练的优化标准反映了设计者的优先级，而非他们所影响的个体的优先级。因此，用户可能会采取策略性行为以获得更有利的结果，有效地挑战平台的预测。虽然过去的研究已经研究了学习平台上用户的策略性行为，但主要集中在用户对已部署模型的策略性响应上，而没有考虑其他用户的行为。相比之下，前瞻性推理考虑到用户行动是相互关联的，并且在规模上会影响未来的预测。在这个框架内，我们首先形式化了行为经济学中的level-k思考概念，用户通过向前看一步来超越他们的同行。我们表明，虽然向均衡的收敛速度加快，但均衡保持不变，从长远来看，高水平思考对个人没有好处。然后，我们关注集体推理，用户通过优化他们对模型的集体影响来采取协调行动。通过对比集体与自私行为，我们描述了协调的益处和限制；学习者和用户效用之间对齐的新概念成为一个关键概念。我们讨论了与几个相关数学框架的联系，包括策略分类、预测性预测和算法集体行动。


### 论文摘要

On many learning platforms, the optimization criteria guiding model training reflect the priorities of the designer rather than those of the individuals they affect. Consequently, users may act strategically to obtain more favorable outcomes, effectively contesting the platform's predictions. While past work has studied strategic user behavior on learning platforms, the focus has largely been on strategic responses to a deployed model, without considering the behavior of other users. In contrast, look-ahead reasoning takes into account that user actions are coupled, and -- at scale -- impact future predictions. Within this framework, we first formalize level-$k$ thinking, a concept from behavioral economics, where users aim to outsmart their peers by looking one step ahead. We show that, while convergence to an equilibrium is accelerated, the equilibrium remains the same, providing no benefit of higher-level reasoning for individuals in the long run. Then, we focus on collective reasoning, where users take coordinated actions by optimizing through their joint impact on the model. By contrasting collective with selfish behavior, we characterize the benefits and limits of coordination; a new notion of alignment between the learner's and the users' utilities emerges as a key concept. We discuss connections to several related mathematical frameworks, including strategic classification, performative prediction, and algorithmic collective action.

---

## 34. Graph Neural Networks for Vehicular Social Networks: Trends, Challenges, and Opportunities

**论文链接:** [http://arxiv.org/abs/2511.14720v1](http://arxiv.org/abs/2511.14720v1)

**作者:** Elham Binshaflout, Aymen Hamrouni, Hakim Ghazzai

**发布时间:** 2025-11-18

**备注:** Submitted for IEEE Transactions on Intelligent Transportation Systems (T-ITS)

### GPT解析

### 总结

这篇论文综述了图神经网络(GNNs)在车辆社交网络(VSNs)中的应用，是首个专门针对这一领域的全面回顾。GNNs通过利用交通相关数据为VSN应用提供了有前景的解决方案。

### 背景

图神经网络已成为建模复杂互联数据的有力工具，特别适合智能交通系统应用。然而，专门针对GNNs在车辆社交网络中应用的综述尚属空白。

### 目的

系统分类和分析GNNs在VSN相关任务中的应用，提供定量见解和综合关键要点，检查可用数据集，并概述推进基于GNN的VSN应用所需的研究方向。

### 方法

利用欧几里得和非欧几里得交通相关数据(包括交通模式、道路使用者和天气条件)，按照交通流和轨迹预测、交通预测、信号控制、驾驶辅助、路由问题和连接管理等任务对现有研究进行系统分类和分析。

### 主要发现

GNNs在提高特定任务或子VSN图上的准确性、鲁棒性和实时性能方面显示出强大潜力，但缺乏对包含所有功能组件的完整、独立VSN建模的研究。

### 结论

随着数据的日益丰富和图学习的不断进步，GNNs有望在未来大规模和完全集成的VSN应用中发挥核心作用。

### 翻译

图神经网络(GNNs)已成为建模复杂互联数据的有力工具，使其特别适合广泛的智能交通系统(ITS)应用。这篇综述首次专门针对GNNs在车辆社交网络(VSNs)中的使用进行了全面回顾。通过利用欧几里得和非欧几里得交通相关数据，包括交通模式、道路使用者和天气条件，GNNs为分析和增强VSN应用提供了有前景的解决方案。该综述根据与VSN相关的主要任务(包括交通流和轨迹预测、交通预测、信号控制、驾驶辅助、路由问题和连接管理)对现有研究进行了系统分类和分析。它进一步提供了定量见解并综合了文献综述的关键要点。此外，该综述检查了可用数据集，并概述了推进基于GNN的VSN应用所需的研究方向。研究结果表明，尽管GNNs在提高特定任务或子VSN图上的准确性、鲁棒性和实时性能方面显示出强大潜力，但仍然缺乏对包含所有功能组件的完整、独立VSN建模的研究。随着数据的日益丰富和图学习的不断进步，GNNs有望在未来大规模和完全集成的VSN应用中发挥核心作用。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as powerful tools for modeling complex, interconnected data, making them particularly well suited for a wide range of Intelligent Transportation System (ITS) applications. This survey presents the first comprehensive review dedicated specifically to the use of GNNs within Vehicular Social Networks (VSNs). By leveraging both Euclidean and non-Euclidean transportation-related data, including traffic patterns, road users, and weather conditions, GNNs offer promising solutions for analyzing and enhancing VSN applications. The survey systematically categorizes and analyzes existing studies according to major VSN-related tasks, including traffic flow and trajectory prediction, traffic forecasting, signal control, driving assistance, routing problem, and connectivity management. It further provides quantitative insights and synthesizes key takeaways derived from the literature review. Additionally, the survey examines the available datasets and outlines open research directions needed to advance GNN-based VSN applications. The findings indicate that, although GNNs demonstrate strong potential for improving the accuracy, robustness, and real-time performances of on task-specific or sub-VSN graphs, there remains a notable absence of studies that model a complete, standalone VSN encompassing all functional components. With the increasing availability of data and continued progress in graph learning, GNNs are expected to play a central role in enabling future large-scale and fully integrated VSN applications.

---

## 35. Nonparametric Uniform Inference in Binary Classification and Policy Values

**论文链接:** [http://arxiv.org/abs/2511.14700v1](http://arxiv.org/abs/2511.14700v1)

**作者:** Nan Liu andYanbo Liu, Yuya Sasaki, Yuanyuan Wan

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文开发了成本敏感二元分类中非参数均匀推断的方法，该方法涵盖最大得分估计、预测效用最大化行动和政策学习。通过引入严格凸替代损失函数点识别代表性非参数政策函数，实现对最优分类政策和最优政策值的推断，简化了经验实现并建立了渐近正态性。

### 背景

成本敏感二元分类问题包括最大得分估计、预测效用最大化行动和政策学习，这些问题以收敛速度慢和非标准极限行为著称，即使在参数框架下点识别的情况下也是如此；在非参数设置下，可能面临识别失败的问题。

### 目的

解决成本敏感二元分类中的非参数均匀推断问题，克服收敛速度慢和非标准极限行为的挑战，避免识别失败，并简化经验实现。

### 方法

引入一个严格凸的替代损失函数，该函数点识别一个代表性的非参数政策函数；估计这个替代政策，对最优分类政策和最优政策值进行推断；建立最优政策值的根号n渐近正态性，推导最优分类政策在标准非参数速率下的高斯近似。

### 主要发现

所提出的方法可以实现高斯推断，相对于直接处理原始分类问题大大简化了经验实现；建立了最优政策值的根号n渐近正态性；推导了最优分类政策在标准非参数速率下的高斯近似；广泛的模拟研究证实了理论发现。

### 结论

通过引入严格凸替代损失函数点识别代表性非参数政策函数，成功解决了成本敏感二元分类中的非参数均匀推断问题，实现了简化经验实现和建立渐近正态性的目标。

### 翻译

我们开发了成本敏感二元分类中非参数均匀推断的方法，这一框架涵盖了最大得分估计、预测效用最大化行动和政策学习。这些问题以收敛速度慢和非标准极限行为而闻名，即使在点识别的参数框架下也是如此。在非参数设置中，它们可能进一步面临识别失败。为应对这些挑战，我们引入了一个严格凸的替代损失函数，该函数点识别一个代表性的非参数政策函数。然后我们估计这个替代政策，对最优分类政策和最优政策值进行推断。这种方法实现了高斯推断，相对于直接处理原始分类问题大大简化了经验实现。特别是，我们建立了最优政策值的根号n渐近正态性，并推导了最优分类政策在标准非参数速率下的高斯近似。广泛的模拟研究证实了理论发现。我们将我们的方法应用于国家JTPA研究，对最优治疗分配政策及其相关福利进行推断。


### 论文摘要

We develop methods for nonparametric uniform inference in cost-sensitive binary classification, a framework that encompasses maximum score estimation, predicting utility maximizing actions, and policy learning. These problems are well known for slow convergence rates and non-standard limiting behavior, even under point identified parametric frameworks. In nonparametric settings, they may further suffer from failures of identification. To address these challenges, we introduce a strictly convex surrogate loss that point-identifies a representative nonparametric policy function. We then estimate this surrogate policy to conduct inference on both the optimal classification policy and the optimal policy value. This approach enables Gaussian inference, substantially simplifying empirical implementation relative to working directly with the original classification problem. In particular, we establish root-$n$ asymptotic normality for the optimal policy value and derive a Gaussian approximation for the optimal classification policy at the standard nonparametric rate. Extensive simulation studies corroborate the theoretical findings. We apply our method to the National JTPA Study to conduct inference on the optimal treatment assignment policy and its associated welfare.

---

## 36. Deviations from the Isobaric Multiplet Mass Equation due to threshold states

**论文链接:** [http://arxiv.org/abs/2511.14674v1](http://arxiv.org/abs/2511.14674v1)

**作者:** R. J. Charity, J. Okołowicz, M. Płoszajczak, L. G. Sobotka, K. W. Brown

**发布时间:** 2025-11-18

**备注:** 7 pages, 3 figures

### GPT解析

### 总结

研究完成了A=16同位旋五重态中自旋/宇称Jπ=0+和2+状态的研究，发现其质量作为同位旋投影的函数偏离二次行为，表明存在超出两体力预期的同位旋破坏。

### 背景

关于A=16原子核同位旋五重态的研究，特别是0+和2+自旋/宇称状态的质量与同位旋投影的关系。

### 目的

研究这些状态的质量行为，检测可能存在的同位旋破坏，并理解其物理机制。

### 方法

使用壳模型嵌入连续谱（SMEC）进行理论预测，并与实验观测结果进行比较。

### 主要发现

1. 质量作为同位旋投影的函数偏离二次行为，表明存在同位旋破坏；2. 这种偏离在2+状态中最为明显；3. 同位旋破坏与质子富集核的开放量子系统特性有关；4. 16Ne中的0+和2+状态以及16F中的2+状态是阈值共振，存在s波与连续谱的耦合；5. 8C的基态也显示连续谱耦合，但通过p波耦合。

### 结论

同位旋五重态中观察到的偏离二次行为的现象是由连续谱耦合引起的，这为理解开放量子系统中的核结构提供了新视角。

### 翻译

最近的研究完成了A=16同位旋五重态中自旋/宇称Jπ=0+和2+状态的研究。它们的质量作为同位旋投影的函数显示出偏离二次行为的证据，表明存在超出两体力预期的同位旋破坏。这种偏离在2+状态中最为明显。壳模型嵌入连续谱（SMEC）的预测使我们能够解释这种同位旋破坏与五重质子富集成员的开放量子系统特性导致的核结构修改有关。特别是，16Ne中的0+和2+状态以及16F中的2+状态是阈值共振，位于质子衰变阈值上方，预期存在s波与连续谱的耦合。这些阈值状态与多重态其余成员二次行为的偏离使我们能够获得关于连续谱耦合能量修正大小和能量依赖性的信息。对于8C的基态，也观察到连续谱耦合，但这次是通过p波耦合。


### 论文摘要

Recent studies have completed the A=16 isospin quintets for states with spin/parity Jπ =0+ and 2+. The dependence of their masses as a function of isospin projection shows evidence for deviations from quadratic behavior indicating isospin violation beyond the expectation from two- body forces. The deviation is most pronounced for the 2+ states. Predictions from the Shell Model Embedded in the Continuum (SMEC) allow us to explain that this isospin violation is associated with a modification of the nuclear structure due to the open-quantum-system nature of the proton- rich members of the quintet. In particular, the 0+ and 2+ states in 16Ne and the 2+ state in 16F are threshold resonances located just above a proton-decay threshold where s-wave coupling to the continuum is expected. The measured deviations of these threshold states from the quadratic behavior of the remaining members of the multiplets makes it possible to obtain information on the magnitude and the energy dependence of the continuum-coupling energy correction. Continuum coupling is also indicated for the ground state of 8C, but this time through p-wave coupling.

---

## 37. M-CALLM: Multi-level Context Aware LLM Framework for Group Interaction Prediction

**论文链接:** [http://arxiv.org/abs/2511.14661v1](http://arxiv.org/abs/2511.14661v1)

**作者:** Diana Romero, Xin Gao, Daniel Khalkhali, Salma Elmalaki

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究展示了大型语言模型如何利用多级上下文信息预测协作混合现实环境中的群体协调模式，通过构建M-CALLM框架，将多模态传感器流转换为分层上下文，显著提高了预测性能。

### 背景

在协作混合现实环境中预测群体协调模式具有挑战性，传统统计模型存在性能上限。

### 目的

探索将个体行为特征、群体结构特性和时间动态编码为自然语言，使大型语言模型突破统计模型的性能限制。

### 方法

构建M-CALLM框架，将多模态传感器流转换为分层上下文；评估零样本提示、少样本学习和监督微调三种范式；在干预模式和模拟模式下与统计基线模型比较；在16个群体（64名参与者，约25小时）数据上进行测试。

### 主要发现

上下文感知的大型语言模型在对话预测上达到96%的准确率，比LSTM基线提高3.2倍，延迟低于35毫秒；但在模拟模式下性能下降83%；对话依赖时间模式，邻近性受益于群体结构（+6%），共享注意力完全失败（0%召回率）。

### 结论

该研究为构建智能协作感知系统提供了新思路，这些系统能够平衡语义推理能力和基本约束。

### 翻译

本文探讨了大型语言模型如何利用多级上下文信息来预测协作混合现实环境中的群体协调模式。我们证明将个体行为特征、群体结构特性和时间动态编码为自然语言，使大型语言模型能够突破统计模型的性能上限。我们构建了M-CALLM框架，将多模态传感器流转换为大型语言模型预测的分层上下文，并在干预模式（实时预测）和模拟模式（自回归预测）下，通过三种范式（零样本提示、少样本学习和监督微调）与统计基线进行比较。在16个群体（64名参与者，约25小时）的直接比较表明，上下文感知的大型语言模型在对话预测上达到96%的准确率，比LSTM基线提高3.2倍，同时保持低于35毫秒的延迟。然而，模拟模式显示由于级联错误导致性能下降83%。对不同模态性能的深入分析表明，对话依赖于时间模式，邻近性受益于群体结构（+6%），而共享注意力完全失败（0%召回率），暴露了架构局限性。我们希望这项工作能够激发构建智能协作感知系统的新思路，这些系统能够平衡语义推理能力与基本约束。


### 论文摘要

This paper explores how large language models can leverage multi-level contextual information to predict group coordination patterns in collaborative mixed reality environments. We demonstrate that encoding individual behavioral profiles, group structural properties, and temporal dynamics as natural language enables LLMs to break through the performance ceiling of statistical models. We build M-CALLM, a framework that transforms multimodal sensor streams into hierarchical context for LLM-based prediction, and evaluate three paradigms (zero-shot prompting, few-shot learning, and supervised fine-tuning) against statistical baselines across intervention mode (real-time prediction) and simulation mode (autoregressive forecasting) Head-to-head comparison on 16 groups (64 participants, ~25 hours) demonstrates that context-aware LLMs achieve 96% accuracy for conversation prediction, a 3.2x improvement over LSTM baselines, while maintaining sub-35ms latency. However, simulation mode reveals brittleness with 83% degradation due to cascading errors. Deep-dive into modality-specific performance shows conversation depends on temporal patterns, proximity benefits from group structure (+6%), while shared attention fails completely (0% recall), exposing architectural limitations. We hope this work spawns new ideas for building intelligent collaborative sensing systems that balance semantic reasoning capabilities with fundamental constraints.

---

## 38. Real-time time-dependent density functional theory for high-energy density physics

**论文链接:** [http://arxiv.org/abs/2511.14643v1](http://arxiv.org/abs/2511.14643v1)

**作者:** Alina Kononov, Minh Nguyen, Andrew D. Baczewski

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文探讨了高能量密度(HED)系统的电子响应特性及其在行星结构、聚变靶标演化和实验室天体物理学诊断中的应用。实时时间依赖密度泛函理论(TDDFT)被证明是一种多功能的建模框架，能够准确预测HED材料的动态响应特性。

### 背景

高能量密度(HED)系统的电子响应特性影响行星结构，驱动聚变靶标演化，并支持实验室天体物理学中的诊断技术。

### 目的

展示实时时间依赖密度泛函理论(TDDFT)在预测HED材料动态响应方面的应用价值，并提供实用教程和未来发展方向。

### 方法

使用实时TDDFT作为建模框架，能够准确预测HED材料的动态响应，包括自由-自由、束缚-自由和束缚-束缚贡献，无需人为状态划分；可以捕获集体和非集体行为；在线性响应区域及其之外都适用。

### 主要发现

实时TDDFT为HED系统提供了一个多功能的建模框架，能够准确预测动态响应特性，无需人为状态划分，适用于多种响应类型。

### 结论

实时TDDFT是研究HED系统电子响应特性的强大工具，本文提供了理论回顾、实用教程和未来发展方向，为HED科学研究提供了新的计算方法。

### 翻译

高能量密度(HED)系统的电子响应特性影响行星结构，驱动聚变靶标的演化，并支撑实验室天体物理学中的诊断技术。实时时间依赖密度泛函理论(TDDFT)提供了一种多功能的建模框架，能够准确预测HED材料的动态响应——包括自由-自由、束缚-自由和束缚-束缚贡献，无需人为状态划分；能够捕获集体和非集体行为；在线性响应区域及其之外都适用。我们回顾了应用于HED系统的实时TDDFT理论形式，提供了计算相关响应特性（动态结构因子、电导率和阻止本领）的实用教程，并评论了这一强大计算方法在服务HED科学中进一步发展的途径。


### 论文摘要

Electronic response properties of high-energy density (HED) systems influence planetary structure, drive evolution of fusion targets, and underpin diagnostics in laboratory astrophysics. Real-time time-dependent density functional theory (TDDFT) offers a versatile modeling framework capable of accurately predicting the dynamic response of HED materials -- including free-free, bound-free, and bound-bound contributions without requiring ad hoc state partitioning; capturing both collective and non-collective behavior; and applicable within the linear-response regime and beyond. We review the theoretical formalism of real-time TDDFT as applied to HED systems, provide a practical tutorial for computing relevant response properties (dynamic structure factors, conductivity, and stopping power), and comment on avenues for further development of this powerful computational method in service of HED science.

---

## 39. Graded strength of comparative illusions is explained by Bayesian inference

**论文链接:** [http://arxiv.org/abs/2511.14642v1](http://arxiv.org/abs/2511.14642v1)

**作者:** Yuhan Zhang, Erxiao Wang, Cory Shain

**发布时间:** 2025-11-18

**备注:** 49 pages, 7 figures

### GPT解析

### 总结

本研究探讨了语言处理中的比较错觉(CI)现象，并通过贝叶斯推断在有噪声信道上的理论模型来解释这种现象。

### 背景

语言处理和视觉处理一样，容易受到错觉的影响，人们会系统性地误感知刺激。比较错觉(CI)是一个例子，如'更多的学生去过俄罗斯，而不是我'，尽管这种比较在本质上是无意义的，但理解者倾向于判断句子为可接受的。

### 目的

通过直接预测错觉强度的定量模型来验证有噪声信道理论，该模型通过统计语言模型与人类行为数据的创新综合推导得出。

### 方法

利用贝叶斯推断在有噪声信道上的理论框架，通过统计语言模型与人类行为数据的创新综合，构建了合理解释的后验概率定量模型。

### 主要发现

该模型不仅解释了CI效应强度的精细差异，还解释了先前无法解释的由代词与完整名词短语than从句主语引起的效果。

### 结论

这些发现支持了句子理解的有噪声信道理论，并通过实证验证预测证明了该理论的有效性。这一结果结合了在错觉和非错觉背景中信道处理的相关证据，支持有噪声信道推断作为多样化语言处理现象的统一计算层理论。

### 翻译

与视觉处理一样，语言处理也容易受到错觉的影响，人们会系统性地误感知刺激。在一个这样的案例中——比较错觉(CI)，例如'更多的学生去过俄罗斯，而不是我'——理解者倾向于判断句子为可接受的，尽管其潜在的比较是无意义的。先前的研究认为这种现象可以通过贝叶斯推断在有噪声的信道上来解释：句子解释的后验概率与该解释的先验概率和被腐败为观察到的(CI)句子的可能性成正比。初步的行为研究通过评估CI句子的有限替代解释集，并证明理解者倾向于选择更可能被腐败为错觉句子的解释，从而支持了这一主张。在本研究中，我们通过直接预测错觉强度的定量模型来复制并显著超越早期工作，该模型是通过对合理解释的后验概率的定量模型推导得出的，我们通过统计语言模型与人类行为数据的创新综合得出了该模型。我们的模型不仅解释了CI效应强度的精细差异，还解释了先前无法解释的由代词与完整名词短语than从句主语引起的效果。这些发现通过证明该理论对比较错觉做出了新的实证验证预测，支持了句子理解的有噪声信道理论。这一结果结合了在错觉和非错觉背景中信道处理的相关证据，支持有噪声信道推断作为多样化语言处理现象的统一计算层理论。


### 论文摘要

Like visual processing, language processing is susceptible to illusions in which people systematically misperceive stimuli. In one such case--the comparative illusion (CI), e.g., More students have been to Russia than I have--comprehenders tend to judge the sentence as acceptable despite its underlying nonsensical comparison. Prior research has argued that this phenomenon can be explained as Bayesian inference over a noisy channel: the posterior probability of an interpretation of a sentence is proportional to both the prior probability of that interpretation and the likelihood of corruption into the observed (CI) sentence. Initial behavioral work has supported this claim by evaluating a narrow set of alternative interpretations of CI sentences and showing that comprehenders favor interpretations that are more likely to have been corrupted into the illusory sentence. In this study, we replicate and go substantially beyond this earlier work by directly predicting the strength of illusion with a quantitative model of the posterior probability of plausible interpretations, which we derive through a novel synthesis of statistical language models with human behavioral data. Our model explains not only the fine gradations in the strength of CI effects, but also a previously unexplained effect caused by pronominal vs. full noun phrase than-clause subjects. These findings support a noisy-channel theory of sentence comprehension by demonstrating that the theory makes novel predictions about the comparative illusion that bear out empirically. This outcome joins related evidence of noisy channel processing in both illusory and non-illusory contexts to support noisy channel inference as a unified computational-level theory of diverse language processing phenomena.

---

## 40. Precise, efficient and flexible modeling of crystallizing elastomers based on physics-augmented neural networks

**论文链接:** [http://arxiv.org/abs/2511.14553v1](http://arxiv.org/abs/2511.14553v1)

**作者:** Konrad Friedrichs, Franz Dammaß, Karl A. Kalina, Markus Kästner

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了一种精确高效的物理增强神经网络(PANN)来模拟天然橡胶中的应变诱导结晶现象。该方法基于双势框架，使用神经网络构建自由能和耗散势，确保了物理 desirable 属性如客观性、材料对称性和热力学一致性。

### 背景

天然橡胶在应变过程中会发生结晶现象，这对材料的力学性能有重要影响。传统模型在准确描述这一复杂现象时存在局限性。

### 目的

开发一种能够精确描述天然橡胶应变诱导结晶行为的模型，同时确保物理一致性和计算效率。

### 方法

采用基于神经网络的自由能和耗散势的双势框架，引入拉格朗日乘数和Karush-Kuhn-Tucker条件来确保结晶度的有界性。基于增量变分框架推导时间离散形式的控制方程，并实现有限元计算。

### 主要发现

所提出的PANN模型能够准确预测应力和结晶度的演化，适用于填充和非填充天然橡胶。模型在材料点级别和缺口试样中的场分布预测方面均表现出色。

### 结论

物理增强神经网络为模拟天然橡胶应变诱导结晶提供了精确且高效的方法，具有良好的预测能力和灵活性。

### 翻译

我们提出了一种精确高效的物理增强神经网络(PANN)来模拟天然橡胶(NR)中的应变诱导结晶。该方法基于双势框架，类似于广义标准材料(GSM)的概念。为了描述材料行为，采用了基于神经网络的自由能和耗散势。结晶度的演化从这两个势导出，类似于经典的GSM型方程。引入两个额外的拉格朗日乘数以及相应的Karush-Kuhn-Tucker条件，以确保结晶度的有界性，使其可以被解释为浓度型变量。基于神经网络的势能确保了所有物理 desirable 属性。最重要的是，客观性、材料对称性和热力学一致性自动得到满足。此外，还提出了基于增量变分框架的时间离散形式控制模型方程的替代推导方法，这也为有限元实现提供了基础。我们使用文献中的三种不同实验数据集展示了PANN的预测能力，同时考虑了材料点级别的应力和结晶度演化以及缺口试样中的相应场分布。此外，我们证明了该模型可以灵活地用于填充和非填充NR。


### 论文摘要

We propose a precise and efficient physics-augmented neural network (PANN) to model strain-induced crystallization in natural rubber (NR). The approach is based on a two potential framework, similar to the concept of generalized standard materials (GSMs). To describe the material behavior, neural network-based free energy and dissipation potentials are employed. The evolution of crystallinity is derived from the two potentials and resembles a classical GSM-type equation. Two additional Lagrange multipliers together with the corresponding Karush-Kuhn-Tucker conditions are introduced to ensure boundedness of the crystallinity, such that it can be interpreted as a variable of concentration type. The neural network-based potentials ensure all physically desirable properties by construction. Most importantly, objectivity, material symmetry, and thermodynamic consistency are automatically fulfilled. In addition, an alternative derivation of the governing model equations in time-discrete form is presented based on an incremental variational framework, which also serves as the basis for a finite element implementation. We demonstrate the predictive capability of the PANN using three different experimental data sets from literature, considering both stress and crystallinity evolution at material point level as well as the corresponding field distributions in a notched specimen. Moreover, we demonstrate that our model can be flexibly employed for both unfilled and filled NR.

---

## 41. Integral Bayesian symbolic regression for optimal discovery of governing equations from scarce and noisy data

**论文链接:** [http://arxiv.org/abs/2511.14388v1](http://arxiv.org/abs/2511.14388v1)

**作者:** Oriol Cabanas-Tirapu, Sergio Cobo-Lopez, Savannah E. Sanchez, Forest L. Rohwer, Marta Sales-Pardo, Roger Guimerà

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究介绍了一种积分贝叶斯符号回归方法，可直接从原始时间序列数据学习控制方程，无需手动假设或易出错的导数估计。即使在有噪声或稀少数据的情况下，该方法也能稳健识别控制方程，并在细菌生长实验中发现优于经典模型的新方程。

### 背景

理解系统如何随时间演变需要发现控制其行为的微分方程。从实验数据自动学习这些方程具有挑战性，特别是当数据有噪声或有限时，现有方法在估计未观测导数方面存在困难。

### 目的

引入一种积分贝叶斯符号回归方法，直接从原始时间序列数据学习控制方程，避免手动假设和易出错的导数估计。

### 方法

通过采样符号微分方程的空间并通过数值积分评估它们，即使在有噪声或稀少数据的情况下，也能稳健地识别控制方程。

### 主要发现

在合成基准测试中准确恢复真实模型；在所有噪声条件下对系统动力学做出准最优预测；应用于多种物种和基质的细菌生长实验，发现了新的生长方程，在准确捕捉微生物增殖的所有阶段（包括延迟期、指数期和饱和期）方面优于经典模型；揭示了生长动力学的微妙变化，如双斜升或非规范转变。

### 结论

该方法提供了一种更深入、数据驱动的微生物生理学理解，能够发现标准方法难以捕捉的复杂动态行为模式。

### 翻译

理解系统如何随时间演变通常需要发现控制其行为的微分方程。当数据有噪声或有限时，从实验数据自动学习这些方程具有挑战性，现有方法在估计未观测导数方面尤其困难。在此，我们引入了一种积分贝叶斯符号回归方法，可直接从原始时间序列数据学习控制方程，无需手动假设或易出错的导数估计。通过采样符号微分方程的空间并通过数值积分评估它们，我们的方法即使在有噪声或稀少数据的情况下也能稳健地识别控制方程。我们表明，这种方法在合成基准测试中准确恢复真实模型，并且在所有噪声条件下对系统动力学做出准最优预测。将此方法应用于多种物种和基质的细菌生长实验，我们发现了新的生长方程，在准确捕捉微生物增殖的所有阶段（包括延迟期、指数期和饱和期）方面优于经典模型。与标准方法不同，我们的方法揭示了生长动力学的微妙变化，如双斜升或非规范转变，从而提供了对微生物生理学更深入、数据驱动的理解。


### 论文摘要

Understanding how systems evolve over time often requires discovering the differential equations that govern their behavior. Automatically learning these equations from experimental data is challenging when the data are noisy or limited, and existing approaches struggle, in particular, with the estimation of unobserved derivatives. Here, we introduce an integral Bayesian symbolic regression method that learns governing equations directly from raw time-series data, without requiring manual assumptions or error-prone derivative estimation. By sampling the space of symbolic differential equations and evaluating them via numerical integration, our method robustly identifies governing equations even from noisy or scarce data. We show that this approach accurately recovers ground-truth models in synthetic benchmarks, and that it makes quasi-optimal predictions of system dynamics for all noise regimes. Applying this method to bacterial growth experiments across multiple species and substrates, we discover novel growth equations that outperform classical models in accurately capturing all phases of microbial proliferation, including lag, exponential, and saturation. Unlike standard approaches, our method reveals subtle shifts in growth dynamics, such as double ramp-ups or non-canonical transitions, offering a deeper, data-driven understanding of microbial physiology.

---

## 42. Magnetic Fields in the Shapley Supercluster Core with POSSUM: Challenging Model Predictions

**论文链接:** [http://arxiv.org/abs/2511.14377v1](http://arxiv.org/abs/2511.14377v1)

**作者:** D. Alonso-López, S. P. O'Sullivan, A. Bonafede, L. M. Böss, C. Stuardi, E. Osinga, C. S. Anderson, C. L. Van Eck, E. Carretti, J. L. West, T. Akahori, K. Dolag, S. Giacintucci, A. Khadir, Y. K. Ma, S. Malik, N. McClure-Griffiths, L. Rudnick, B. A. Seidel, S. Tiwari, T. Venturi

**发布时间:** 2025-11-18

**备注:** 23 pages, 15 figures. Accepted for publication in A&A

### GPT解析

### 总结

本研究使用法拉第旋转测量网格研究Shapley超星系团核心区域的磁化等离子体特性，通过分析旋转测量散射来约束气体密度和磁场特性，并与模型和模拟结果进行比较。

### 背景

法拉第旋转测量网格提供了一种追踪宇宙环境中磁化等离子体的敏感手段。Shapley超星系团核心区域包含两个星系团（A3558和A3562）以及它们之间的两个星系群，红移约为0.048。

### 目的

通过结合旋转测量网格数据与热Sunyaev-Zeldovich效应数据，确定气体密度、磁场特性及其相关性，从而约束Shapley超星系团核心区域的磁场特性。

### 方法

研究结合了来自POSSUM初步调查和Planck的热Sunyaev-Zeldovich效应数据，分析了旋转测量散射及其与最近星系团/星系群距离的关系，并将观测结果与半解析高斯随机场模型和宇宙磁流体动力学模拟进行比较。

### 主要发现

1) 在每平方度36个旋转测量的天空密度下，在SSC区域检测到30.5±4.6 rad/m²的过量旋转测量散射；2) 与模型比较发现，星系群和星系团中的平均磁场强度为1-3微高斯；3) 从所有物体0.3-1.8 r₅₀₀范围的数据得出的旋转测量散射剖面比预期的更平坦；4) 与SSC结构最匹配的宇宙磁流体动力学模拟与星系团间区域中尺度小于约0.8 r₅₀₀的湍流速度放大磁场的情况最为一致。

### 结论

密集的旋转测量网格和POSSUM提供的精度使得研究人员能够探测SSC星系团和星系群内及其r₅₀₀范围内的磁化气体。比预期更平坦的旋转测量散射剖面揭示了将观测结果与宇宙磁流体动力学模拟即使在最现实的预测中与相互作用的星系团外围区域协调一致的重大挑战。

### 翻译

法拉第旋转测量网格提供了一种敏感的方法，可以广泛追踪宇宙环境中的磁化等离子体。我们研究了来自Shapley超星系团核心的旋转测量信号，以约束气体的磁场特性。SSC区域包含两个星系团A3558和A3562，以及它们之间的两个星系群，红移约为0.048。我们将旋转测量网格数据与分别来自POSSUM初步调查和Planck的热Sunyaev-Zeldovich效应数据相结合。为了稳健地确定气体密度、其磁场特性及其相关性，我们研究了SSC区域的旋转测量散射及其作为与最近星系团/星系群距离函数的行为。我们将观测结果与半解析高斯随机场模型和更现实的宇宙磁流体动力学模拟进行比较。在每平方度36个旋转测量的天空密度下，我们在SSC区域检测到30.5±4.6 rad/m²的过量旋转测量散射。与模型比较，我们发现星系群和星系团中的平均磁场强度为1-3微高斯。从所有物体0.3-1.8 r₅₀₀范围的数据得出的旋转测量散射剖面比模型预期的更平坦，η<0.5更受支持。尽管存在这种差异，我们发现与SSC结构最匹配的宇宙磁流体动力学模拟与星系团间区域中尺度小于约0.8 r₅₀₀的湍流速度放大磁场的情况最为一致。POSSUM提供的密集旋转测量网格和精度使我们能够探测SSC星系团和星系群内及其r₅₀₀范围内的磁化气体。比预期更平坦的旋转测量散射剖面揭示了将观测结果与宇宙磁流体动力学模拟即使在最现实的预测中与相互作用的星系团外围区域协调一致的重大挑战。


### 论文摘要

Faraday Rotation Measure (RM) Grids provide a sensitive means to trace magnetized plasma across a wide range of cosmic environments. We study the RM signal from the Shapley Supercluster Core (SSC), in order to constrain the magnetic field properties of the gas. The SSC region consists of two galaxy clusters A3558 and A3562, and two galaxy groups between them, at $z\simeq 0.048$. We combine RM Grid data with thermal Sunyaev-Zeldovich effect data, obtained from the POSSUM pilot survey, and Planck, respectively. To robustly determine the gas density, its magnetic field properties, and their correlation, we study the RM scatter in the SSC region and its behavior as a function of distance to the nearest cluster/group. We compare observational results with semi-analytic Gaussian random field models and more realistic cosmological MHD simulations. With a sky-density of 36 RMs/deg$^{2}$, we detect an excess RM scatter of $30.5\pm 4.6 \, \mathrm{rad/m^2}$ in the SSC region. Comparing with models, we find an average magnetic field strength of 1-3 $μ$G (in the groups and clusters). The RM scatter profile, derived from data ranging from 0.3-1.8 $r_{500}$ for all objects, is systematically flatter than expected compared to models, with $η<0.5$ being favored. Despite this discrepancy, we find that cosmological MHD simulations matched to the SSC structure most closely align with scenarios where the magnetic field is amplified by the turbulent velocity in the intercluster regions on scales $\lesssim 0.8\,r_{500}$. The dense RM grid and precision provided by POSSUM allows us to probe magnetized gas in the SSC clusters and groups on scales within and beyond their $r_{500}$. Flatter-than-expected RM scatter profiles reveal a significant challenge in reconciling observations with even the most realistic predictions from cosmological MHD simulations in the outskirts of interacting clusters.

---

## 43. Multi-Timescale Model Predictive Control for Slow-Fast Systems

**论文链接:** [http://arxiv.org/abs/2511.14311v1](http://arxiv.org/abs/2511.14311v1)

**作者:** Lukas Schroth, Daniel Morton, Amon Lahr, Daniele Gammelli, Andrea Carron, Marco Pavone

**发布时间:** 2025-11-18

### GPT解析

### 总结

该论文提出了一种针对快速采样控制的多时间尺度模型预测控制方案，通过结合敏感性指数衰减原理，显著提高了计算效率。

### 背景

模型预测控制已成为约束控制的主要方法，但在结合长预测范围和高保真模型时，实时求解优化问题具有挑战性。

### 目的

受敏感性指数衰减结果的启发，旨在开发一种针对具有快速和慢速动态系统的多时间尺度MPC方案，以提高计算效率。

### 方法

提出的方法通过切换到仅捕捉慢速主导动态的简化模型，并指数级增加积分步长，沿着预测范围逐步减少模型细节，从而提高计算效率。

### 主要发现

在三个实际相关的机器人控制问题中，该方法实现了高达一个数量级的速度提升。

### 结论

多时间尺度MPC方案能够在保持控制性能的同时，显著提高计算效率。

### 翻译

模型预测控制(MPC)已成为约束控制的主要方法，使各种应用实现自主性。虽然在MPC中模型保真度至关重要，但当结合长预测范围与捕捉短期动态和长期行为的高保真模型时，实时求解相应的优化问题仍然具有挑战性。受敏感性指数衰减(EDS)结果的启发，该结果表明在特定条件下，建模误差的影响会沿着预测范围呈指数级下降，本文提出了一种针对快速采样控制的多时间尺度MPC方案。针对具有快速和慢速动态的系统，所提出的方法通过i)切换到仅捕捉慢速主导动态的简化模型和ii)指数级增加积分步长，沿着预测范围逐步减少模型细节，从而提高计算效率。我们在仿真中评估了三个实际相关的机器人控制问题，观察到速度提高了高达一个数量级。


### 论文摘要

Model Predictive Control (MPC) has established itself as the primary methodology for constrained control, enabling autonomy across diverse applications. While model fidelity is crucial in MPC, solving the corresponding optimization problem in real time remains challenging when combining long horizons with high-fidelity models that capture both short-term dynamics and long-term behavior. Motivated by results on the Exponential Decay of Sensitivities (EDS), which imply that, under certain conditions, the influence of modeling inaccuracies decreases exponentially along the prediction horizon, this paper proposes a multi-timescale MPC scheme for fast-sampled control. Tailored to systems with both fast and slow dynamics, the proposed approach improves computational efficiency by i) switching to a reduced model that captures only the slow, dominant dynamics and ii) exponentially increasing integration step sizes to progressively reduce model detail along the horizon. We evaluate the method on three practically motivated robotic control problems in simulation and observe speed-ups of up to an order of magnitude.

---

## 44. Unified Multimodal Vessel Trajectory Prediction with Explainable Navigation Intention

**论文链接:** [http://arxiv.org/abs/2511.14265v1](http://arxiv.org/abs/2511.14265v1)

**作者:** Rui Zhang, Chao Li, Kezhong Liu, Chen Wang, Bolong Zheng, Hongbo Jiang

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种统一的MTP框架，整合可解释的导航意图，解决了现有船舶轨迹预测方法场景适用性有限和可解释性不足的问题。

### 背景

船舶轨迹预测是智能海事系统的基础，在复杂海事环境中短期预测快速行为变化的多模态轨迹预测(MTP)是一个有前景的研究领域。

### 目的

解决现有船舶MTP方法场景适用性有限和可解释性不足的挑战，提出一个统一的MTP框架。

### 方法

将导航意图分为持续性和短暂性两类；从历史轨迹构建持续性意图树；使用条件变分自编码器(CVAE)建模动态短暂意图；使用非局部注意力机制保持全局场景一致性。

### 主要发现

在真实AIS数据集上的实验证明了该方法在不同场景下的广泛适用性，在ADE和FDE方面取得了显著改进，同时通过明确揭示每个预测轨迹背后的导航意图提高了可解释性。

### 结论

所提出的统一MTP框架通过整合可解释的导航意图，有效提高了船舶轨迹预测的场景适用性和可解释性。

### 翻译

船舶轨迹预测是智能海事系统的基础。在这一领域中，复杂海事环境中快速行为变化的短期预测已将多模态轨迹预测(MTP)确立为一个有前景的研究领域。然而，现有的船舶MTP方法存在场景适用性有限和可解释性不足的问题。为应对这些挑战，我们提出了一个统一的MTP框架，整合了可解释的导航意图，我们将这些意图分为持续性和短暂性两类。我们的方法从历史轨迹构建持续性意图树，使用条件变分自编码器(CVAE)建模动态短暂意图，同时使用非局部注意力机制保持全局场景一致性。在真实自动识别系统(AIS)数据集上的实验证明了我们方法在不同场景下的广泛适用性，在ADE和FDE方面取得了显著改进。此外，我们的方法通过明确揭示每个预测轨迹背后的导航意图，提高了可解释性。


### 论文摘要

Vessel trajectory prediction is fundamental to intelligent maritime systems. Within this domain, short-term prediction of rapid behavioral changes in complex maritime environments has established multimodal trajectory prediction (MTP) as a promising research area. However, existing vessel MTP methods suffer from limited scenario applicability and insufficient explainability. To address these challenges, we propose a unified MTP framework incorporating explainable navigation intentions, which we classify into sustained and transient categories. Our method constructs sustained intention trees from historical trajectories and models dynamic transient intentions using a Conditional Variational Autoencoder (CVAE), while using a non-local attention mechanism to maintain global scenario consistency. Experiments on real Automatic Identification System (AIS) datasets demonstrates our method's broad applicability across diverse scenarios, achieving significant improvements in both ADE and FDE. Furthermore, our method improves explainability by explicitly revealing the navigational intentions underlying each predicted trajectory.

---

## 45. Entropic uncertainty under indefinite causal order and input-output direction

**论文链接:** [http://arxiv.org/abs/2511.14192v1](http://arxiv.org/abs/2511.14192v1)

**作者:** Göktuğ Karpat

**发布时间:** 2025-11-18

**备注:** 15 pages, 5 figures

### GPT解析

### 总结

该研究探讨了当记忆量子比特经历噪声动力学时记忆辅助熵不确定关系的行为，特别关注量子开关和量子时间翻转这两种高阶控制过程，发现它们可以显著减少总熵不确定性，表明不确定因果顺序和输入输出方向可作为减轻噪声影响的资源。

### 背景

熵不确定关系量化了量子测量可预测性的极限。当被测系统与量子记忆相关联时，这些极限由记忆辅助熵不确定关系(MA-EUR)描述。

### 目的

研究当记忆量子比特经历通过高阶控制过程实现的噪声动力学时，MA-EUR的行为表现。

### 方法

研究设置中，控制量子比特是执行测量的系统本身，目标量子比特作为噪声量子存储器。针对Pauli通道，将其输入量子开关和量子时间翻转，并与直接应用进行比较。

### 主要发现

与直接应用Pauli通道相比，通过量子开关和量子时间翻转可以显著减少总熵不确定性。

### 结论

不确定因果顺序和输入输出方向可以作为资源，在MA-EUR及其应用背景下减轻噪声的影响。

### 翻译

熵不确定关系量化了量子测量可预测性的极限。当被测系统与量子记忆相关联时，这些极限由记忆辅助熵不确定关系(MA-EUR)描述。我们检查了当记忆量子比特经历通过高阶控制过程实现的噪声动力学时MA-EUR的行为，即量子开关和量子时间翻转。我们考虑了一种设置，其中控制量子比特是执行测量的系统本身，而目标量子比特作为噪声量子存储器。专注于Pauli通道，我们表明将它们输入量子开关和量子时间翻转可以与直接应用相比显著减少总熵不确定性。我们的结果表明，不确定因果顺序和输入输出方向可以作为资源，在MA-EUR及其应用背景下减轻噪声的影响。


### 论文摘要

Entropic uncertainty relations quantify the limits on the predictability of quantum measurements. When the measured system is correlated with a quantum memory, these limits are described by the memory-assisted entropic uncertainty relation (MA-EUR). We examine the behavior of MA-EUR when the memory qubit undergoes noisy dynamics implemented via high-order controlled processes, namely, the quantum switch and the quantum time-flip. We consider a setting in which the control qubit is the very system on which the measurements are performed, while the target qubit serves as a noisy quantum memory. Focusing on Pauli channels, we show that feeding them into the quantum switch and the quantum time-flip can significantly reduce the total entropic uncertainty as compared to their direct application. Our results reveal that indefinite causal order and input-output direction can serve as resources to mitigate the effects of noise in the context of MA-EUR and its applications.

---

## 46. Pressure-Induced B1 to B2 Phase Transition in CeN Studied by ab initio Correlation Matrix Renormalization Theory Calculations

**论文链接:** [http://arxiv.org/abs/2511.14085v1](http://arxiv.org/abs/2511.14085v1)

**作者:** Jianhua Zhang, Jun Liu, Yongxin Yao, Kai-Ming Ho, Cai-Zhuang Wang

**发布时间:** 2025-11-18

### GPT解析

### 总结

应用关联矩阵重整化理论研究铈氮化物在压力下的结构和电子特性变化

### 背景

铈氮化物(CeN)在压力下可能发生从B1(NaCl型)相到B2(CsCl型)相的结构相变

### 目的

使用关联矩阵重整化理论(CMRT)描述和预测铈氮化物在压力下的结构和电子特性变化

### 方法

应用关联矩阵重整化理论(CMRT)分析铈氮化物在压力下的电子结构和相变行为

### 主要发现

1) B1相的电子态密度在费米能级处有尖锐的4f准粒子共振峰，费米能级下有两个由强杂化形成的子带；2) 压缩下发生B1到B2相的一级相变，体积收缩约11%；3) 相变过程中4f谱权重变宽，轨道占据增加，与导态杂化增强，表明从部分局域到更巡游的4f行为转变；4) 理论预测与实验观察结果吻合

### 结论

关联矩阵重整化理论为稀土化合物中关联驱动的结构和电子转变提供了无参数的准确描述和预测

### 翻译

我们将关联矩阵重整化理论(CMRT)应用于压力下的铈氮化物(CeN)。对于B1(NaCl型)相，CMRT给出的状态方程与常压实验一致。它产生的电子态密度(DOS)特征是在费米能级处有一个尖锐的4f准粒子共振峰，费米能级以下有两个由局域Ce-4f电子与巡游Ce-5d和N-2p电子强杂化形成的子带，这与XPS实验一致。在压缩下，CMRT预测了从B1到B2(CsCl型)相的一级相变，体积收缩约11%。在相变过程中，4f谱权重变宽，4f轨道占据增加，与导态的杂化增强，表明从部分局域到更巡游的4f行为的交叉。这些特征与实验观察结果极好地吻合，证明CMRT为稀土化合物中关联驱动的结构和电子转变提供了无参数的描述和预测。


### 论文摘要

We apply correlation matrix renormalization theory (CMRT) to cerium nitride (CeN) under pressure. For B1 (NaCl-type) phase, CMRT gives an equation of state consistent with ambient pressure experiments. It produces electronic density-of-state (DOS) characterized by a sharp 4f quasi-particle resonance peak pinned at the Fermi level and two subbands formed by strong hybridization between the localized Ce-4f electrons and the itinerant Ce-5d and N-2p electrons below the Fermi level, consistent with XPS experiments. Upon compression, CMRT predicts a first-order B1 to B2 (CsCl-type) transition with ~11% volume collapse. Across the transition, the 4f spectral weight broadens, the 4f orbital occupancy increases, and the hybridization with conduction states enhances, signaling a crossover from partially localized to more itinerant 4f behavior. These features are in excellent agreement with experimental observations, demonstrating that CMRT provides a parameter-free description and prediction of correlation-driven structural and electronic transitions in rare-earth compounds.

---

## 47. Flood-LDM: Generalizable Latent Diffusion Models for rapid and accurate zero-shot High-Resolution Flood Mapping

**论文链接:** [http://arxiv.org/abs/2511.14033v1](http://arxiv.org/abs/2511.14033v1)

**作者:** Sun Han Neo, Sachith Seneviratne, Herath Mudiyanselage Viraj Vidura Herath, Abhishek Saha, Sanka Rasnayaka, Lucy Amanda Marshall

**发布时间:** 2025-11-18

**备注:** Accepted for publication at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026

### GPT解析

### 总结

本研究提出了一种基于潜在扩散模型的新型洪水地图超分辨率方法，旨在实现高精度洪水预测的同时显著减少计算时间，使其适用于实时洪水风险管理。

### 背景

洪水预测对应急规划和响应至关重要。传统物理模型计算密集，不适用于实时大规模应用；现有卷积神经网络方法虽精度和速度良好，但对未见过区域泛化能力有限。

### 目的

开发一种高效生成高保真度洪水地图的方法，减少推理时间，保持高精度，并增强模型在新区域的泛化能力和可解释性。

### 方法

利用潜在扩散模型对粗网格洪水地图进行超分辨率处理，结合物理信息输入解决机器学习中的黑盒行为问题，提高模型可解释性。

### 主要发现

潜在扩散模型显著减少生成高保真度洪水地图的计算时间而不损失精度；扩散模型在不同物理位置表现出优异泛化能力；迁移学习可加速对新地理区域的适应。

### 结论

基于扩散模型的洪水地图超分辨率方法能够高效生成高精度洪水预测结果，适用于实时洪水风险管理，同时具有良好的泛化能力和可解释性。

### 翻译

洪水预测对于应急规划和响应以减轻人类和经济损失至关重要。传统的基于物理的水动力模型使用需要细网格离散化的数值方法生成高分辨率洪水地图；这些方法计算密集，不适用于实时大规模应用。虽然最近的研究应用卷积神经网络进行洪水地图超分辨率并取得了良好的精度和速度，但它们对未见过区域的泛化能力有限。在本文中，我们提出了一种新方法，利用潜在扩散模型对粗网格洪水地图进行超分辨率处理，旨在实现与细网格洪水地图相当的精度，同时显著减少推理时间。实验结果表明，潜在扩散模型显著减少了生成高保真度洪水地图所需的计算时间，同时不损失精度，使其能够用于实时洪水风险管理。此外，扩散模型在不同物理位置表现出优异的泛化能力，迁移学习可以进一步加速对新地理区域的适应。我们的方法还结合了物理信息输入，解决了机器学习中常见的黑盒行为局限性，从而提高了可解释性。代码可在 https://github.com/neosunhan/flood-diff 获取。


### 论文摘要

Flood prediction is critical for emergency planning and response to mitigate human and economic losses. Traditional physics-based hydrodynamic models generate high-resolution flood maps using numerical methods requiring fine-grid discretization; which are computationally intensive and impractical for real-time large-scale applications. While recent studies have applied convolutional neural networks for flood map super-resolution with good accuracy and speed, they suffer from limited generalizability to unseen areas. In this paper, we propose a novel approach that leverages latent diffusion models to perform super-resolution on coarse-grid flood maps, with the objective of achieving the accuracy of fine-grid flood maps while significantly reducing inference time. Experimental results demonstrate that latent diffusion models substantially decrease the computational time required to produce high-fidelity flood maps without compromising on accuracy, enabling their use in real-time flood risk management. Moreover, diffusion models exhibit superior generalizability across different physical locations, with transfer learning further accelerating adaptation to new geographic regions. Our approach also incorporates physics-informed inputs, addressing the common limitation of black-box behavior in machine learning, thereby enhancing interpretability. Code is available at https://github.com/neosunhan/flood-diff.

---

## 48. Production Mechanisms of isoscalar pseudotensor mesons in pion and kaon induced reactions

**论文链接:** [http://arxiv.org/abs/2511.14011v1](http://arxiv.org/abs/2511.14011v1)

**作者:** Dong-En Lu, Li-Ming Wang

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究针对标量赝张量介子的理论空白，首次提供了四种预测介子的产生截面全面理论预测，为实验搜索提供关键指导。

### 背景

BESIII合作组最近观测到X(2600)共振态，重新激发了人们对轻介子谱学特别是赝张量介子的兴趣，但标量赝张量介子谱中多个预测的径向激发态仍未被观测。

### 目的

研究四种标量赝张量介子($η_2(4D)$, $η'_2(2D)$, $η'_2(3D)$, 和 $η'_2(4D)$)的产生机制。

### 方法

采用有效拉格朗日量方法结合雷吉轨迹唯象学来描述反应的高能行为。

### 主要发现

$η_2(4D)$态在介子诱导($π^- p 	o η_2 n$)反应中更为突出，而$η'_2(2D, 3D, 4D)$态则在介子诱导($K^- p 	o η_2 Λ$)过程中优先产生；这些态的总截面在特定束流动量处表现出特征性峰，为它们的发现提供了清晰的动力学窗口。

### 结论

这项研究为这些预测的介子的产生截面提供了首次全面的理论预测，为未来的实验搜索提供了关键指导。

### 翻译

BESIII合作组最近对X(2600)共振态的观测重新激发了人们对轻介子谱学的兴趣，特别是赝张量态。然而，在研究不充分的标量赝张量介子谱中存在显著的理论空白，其中几个预测的径向激发态仍未被观测到。为了解决这个问题，我们研究了四种这样的状态($η_2(4D)$, $η'_2(2D)$, $η'_2(3D)$, 和 $η'_2(4D)$)的产生，采用有效拉格朗日量方法结合雷吉轨迹唯象学来描述反应的高能行为。我们的计算显示出明显的产生选择性：$η_2(4D)$态在介子诱导($π^- p 	o η_2 n$)反应中更为突出，而$η'_2(2D, 3D, 4D)$态则在介子诱导($K^- p 	o η_2 Λ$)过程中优先产生。此外，这些态的总截面在特定束流动量处表现出特征性峰，为它们的发现提供了清晰的动力学窗口。这项研究为这些预测的介子的产生截面提供了首次全面的理论预测，为未来的实验搜索提供了关键指导。


### 论文摘要

The recent observation of the X(2600) resonance by the BESIII Collaboration has motivated a renewed interest in the spectroscopy of light mesons, particularly pseudotensor states. However, a significant theoretical gap exists in the poorly explored spectrum of isoscalar pseudotensor mesons, where several predicted radial excitations remain unobserved. To address this, we investigate the production of four such states ($η_2(4D)$, $η'_2(2D)$, $η'_2(3D)$, and $η'_2(4D)$) employing an effective Lagrangian approach combined with Regge trajectory phenomenology to describe the high-energy behavior of the reactions. Our calculations reveal a distinct production selectivity: the $η_2(4D)$ state is more prominent in pion induced ($π^- p \to η_2 n$) reactions, whereas the $η'_2(2D, 3D, 4D)$ states are preferentially produced in kaon induced ($K^- p \to η_2 Λ$) processes. Moreover, the total cross sections for these states exhibit characteristic peaks at specific beam momenta, providing clear kinematic windows for their discovery. This study provides the first comprehensive theoretical predictions for the production cross sections of these predicted mesons, offering crucial guidance for future experimental searches.

---

## 49. Universal Routing of Light via Optical Thermodynamics

**论文链接:** [http://arxiv.org/abs/2511.13968v1](http://arxiv.org/abs/2511.13968v1)

**作者:** Hediyeh M. Dinani, Georgios G. Pyrialakos, Abraham M. Berman Bradley, Monika Monika, Huizhong Ren, Mahmoud A. Selim, Ulf Peschel, Demetrios N. Christodoulides, Mercedeh Khajavikhan

**发布时间:** 2025-11-17

**DOI:** 10.1038/s41566-025-01756-4

### GPT解析

### 总结

本研究展示了在非线性多模系统中，通过光学热力学框架实现光束局域化的现象，无论光从哪个输入端口进入，都能汇聚到紧密局域化的基态。

### 背景

理解和利用复杂非线性系统的动力学是当前科学和技术努力的核心。在光学领域，光在非线性多模环境中的演化是一个难题，其混沌特性常常阻碍预测性洞察。

### 目的

展示一个反直觉的光学过程，其中光被发射到精心设计的非线性阵列的任何输入端口，都能普遍地汇聚到一个紧密局域化的基态。

### 方法

通过部署熵原理，在适当配置的非线性时间合成网格晶格中进行实验验证。

### 主要发现

这种现象在保守线性排列中是完全无法实现的，源于晶格结构和动能与非线性哈密顿量分量之间的相互作用，导致类似焦耳-汤姆逊的膨胀和模式热化两个光学热过程。实验中，光学温度接近零，导致光凝聚在单个斑点上，不受初始激发位置影响。

### 结论

这种效应为应用光学热力学原理实现新的光学功能开辟了新途径，如全光光束转向、多路复用和高功率非线性光束整形，同时增进对多模非线性系统中光与物质相互作用的物理理解。

### 翻译

理解和利用复杂非线性系统的动力学是当今科学和技术努力的核心。在光学领域，光在非线性多模环境中的演化是一个难题，因为其混沌演化常常阻碍预测性洞察。最近，提出了一种光学热力学框架，能够系统地预测并利用这些系统的复杂行为。在本工作中，通过部署熵原理，我们展示了一个反直觉的光学过程，其中光被发射到精心设计的非线性阵列的任何输入端口，都能普遍地汇聚到一个紧密局域化的基态，这种响应在保守线性排列中是完全无法实现的。这种现象源于晶格结构和动能与非线性哈密顿量分量的展开方式之间的相互作用，导致两个光学热过程：类似焦耳-汤姆逊的膨胀和模式热化。实验中，在适当配置的非线性时间合成网格晶格中，光学温度接近零，导致光凝聚在单个斑点上，无论初始激发位置如何。这里展示的效应为应用光学热力学原理实现新的光学功能开辟了新途径，如全光光束转向、多路复用和高功率非线性光束整形，同时也增进对多模非线性系统中光与物质相互作用的显著物理学的理解。


### 论文摘要

Understanding and exploiting the dynamics of complex nonlinear systems is nowadays at the core of a broad range of scientific and technological endeavors. Within the optical domain, light evolution in a nonlinear multimode environment presents a formidable problem, as its chaotic evolution often hinders predictive insights. Recently, an optical thermodynamic framework has been put forward that, in a systematic manner, can not only predict but also harness the intricate behavior of these systems. In this work, by deploying entropic principles, we demonstrate a counterintuitive optical process in which light, launched into any input port of a judiciously designed nonlinear array, universally channels into a tightly localized ground state, a response that is completely unattainable in linear conservative arrangements. This phenomenon arises from the interplay between lattice structure and the way the kinetic and nonlinear Hamiltonian components unfold, leading to two optical thermal processes: a Joule-Thomson-like expansion followed by mode thermalization. Experimentally, this effect is demonstrated in properly configured nonlinear time-synthetic mesh lattices, where the optical temperature approaches near zero, causing light to condense at a single spot, regardless of the initial excitation position. The effect demonstrated here opens new avenues for applying the principles of optical thermodynamics in realizing novel optical functionalities, such as all-optical beam steering, multiplexing, and nonlinear beam shaping in high-power regimes, while also offering a greater understanding of the remarkable physics of light-matter interactions in multimode nonlinear systems.

---

## 50. Anisotropic Dielectric Function of Graphite Probed by Far- and Near-Field Spectroscopies

**论文链接:** [http://arxiv.org/abs/2511.13964v1](http://arxiv.org/abs/2511.13964v1)

**作者:** A. Toksumakov, G. Ermolaev, D. Grudinin, A. Slavich, N. Pak, G. Tikhonowski, A. Vyshnevyy, G. Tselikov, A. Arsenin, V. Volkov

**发布时间:** 2025-11-17

**备注:** 13 pages, 5 figures

### GPT解析

### 总结

该研究解决了石墨光学常数长期存在的不一致性问题，建立了一套新的、自洽的光学常数，为碳基纳米光子学提供了重要参考。

### 背景

石墨作为革命性技术的基石材料，在能源储存和二维材料领域具有重要作用，但其在范德华异质结构中预测工程化光学行为的能力受到报道的光学常数持续差异的严重限制。

### 目的

解决石墨光学常数长期存在的不一致性问题，建立一套新的、自洽的光学常数。

### 方法

采用多模态方法，结合远场光谱椭偏法、纳米尺度近场光学探测(s-SNOM)和微反射光谱技术。

### 主要发现

建立了新的、自洽的光学常数(n和k)，涵盖了从紫外到近红外光谱范围内的面内和面外晶向。

### 结论

这项工作提供了一套统一的光学常数，解决了现有文献中的不一致问题，为碳基纳米光子学领域中光与物质相互作用的定量建模和工程化提供了必要的参考基础。

### 翻译

石墨是革命性技术的基石材料，从能源储存到整个二维材料领域都发挥着核心作用。尽管其具有基础性地位，但在范德华异质结构中预测工程化光学行为的能力受到报道的光学常数持续差异的严重限制。我们通过采用结合远场光谱椭偏法、纳米尺度近场光学探测(s-SNOM)和微反射光谱技术的多模态方法，解决了这一长期存在的不一致性问题。我们建立了一套新的、自洽的光学常数(n和k)，涵盖了从紫外到近红外光谱范围内的面内和面外晶向。这项工作提供了一套统一的光学常数，解决了现有文献中的不一致问题。通过建立这一明确的参考标准，我们为碳基纳米光子学不断发展的领域中光与物质相互作用的定量建模和工程化提供了必要的基础。


### 论文摘要

Graphite is a cornerstone material for revolutionary technologies, from energy storage to the entire field of two-dimensional materials. Despite its foundational role, the predictive power required for engineering emergent optical behavior in van der Waals heterostructures is severely constrained by persistent discrepancies in reported optical constants. We resolve this long-standing ambiguity by deploying a multi-modal approach that synergizes far-field spectroscopic ellipsometry with nanoscale near-field optical probing (s-SNOM) and micro-reflectance spectroscopy. We have established a new, self-consistent set of optical constants (n and k) for both in-plane and out-of-plane crystallographic directions across the ultraviolet-to-near-infrared spectrum. This work presents a unified set of optical constants that addresses inconsistencies in the existing literature. By establishing this definitive reference, we provide the essential foundation for the quantitative modeling and engineering of light-matter interactions in the evolving landscape of carbon-based nanophotonics.

---

## 51. Rapid Design and Fabrication of Body Conformable Surfaces with Kirigami Cutting and Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.13941v1](http://arxiv.org/abs/2511.13941v1)

**作者:** Jyotshna Bali, Jinyang Li, Jie Chen, Suyi Li

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究结合剪纸艺术原理和数据驱动建模，开发了一种个性化、快速且低成本的设计和制造流程，用于创建贴合膝关节表面的材料。通过3D扫描、机器学习和反向设计方法，成功制造出贴合度超过75%的个性化剪纸补丁，仅需一个工作日完成从扫描到交付的全过程。

### 背景

当前需要一种能够贴合动态变形身体表面的个性化可穿戴设备制造方法，特别是针对膝关节等复杂关节区域。

### 目的

开发一种个性化、快速且低成本的设计和制造流程，用于创建围绕膝关节的可贴合身体表面的材料。

### 方法

1. 对人体受试者前膝关节表面进行3D扫描；2. 提取两个关节角度之间的皮肤变形参数；3. 使用有限元分析数据构建机器学习模型；4. 应用高斯过程回归模型关联剪纸切割长度与应变参数；5. 基于协方差矩阵自适应进化策略进行反向设计；6. 使用激光切割制造最终产品。

### 主要发现

1. 高斯过程回归模型预测准确度达99.6%；2. 整个流程从扫描到交付仅需一个工作日；3. 个性化补丁成功贴合超过75%的皮肤面积；4. 开发的抗冲击泡沫补丁既能贴合动态运动又能提供关节保护。

### 结论

该设计和制造框架具有通用性，可扩展到其他变形身体表面，为创建个性化防护装备、透气粘合剂和贴合身体电子产品奠定了基础。

### 翻译

通过整合剪纸切割原理和数据驱动建模，本研究旨在开发一种个性化、快速且低成本的设计和制造流程，用于创建围绕膝关节的可贴合身体表面。该过程始于对人体受试者前膝关节表面的3D扫描，随后提取两个关节角度之间相应的皮肤变形，包括纵向应变和泊松比。同时，使用实验校准的有限元分析的大量模拟数据构建机器学习模型。该模型采用高斯过程回归将剪纸切割长度与产生的纵向应变和泊松比相关联。凭借0.996的R²得分，高斯过程回归在预测剪纸大变形方面优于其他模型。最后，基于协方差矩阵自适应进化策略的反向设计方法用于生成复制膝盖扫描观察到的平面皮肤变形的剪纸补丁设计。该流程应用于三个人体受试者， resulting kirigami knee patches were fabricated using rapid laser cutting, requiring only a business day from knee scanning to kirigami patch delivery. 低成本、个性化的剪纸补丁成功贴合了所有受试者超过75%的皮肤区域，为广泛的可穿戴设备奠定了基础。该研究通过一个抗冲击的剪纸泡沫补丁展示了其潜力，该补丁不仅贴合动态膝关节运动，还能提供关节抗冲击保护。最后，所提出的设计和制造框架具有通用性，可以扩展到其他变形的身体表面，使创建个性化可穿戴设备成为可能，如防护装备、透气粘合剂和贴合身体的电子产品。


### 论文摘要

By integrating the principles of kirigami cutting and data-driven modeling, this study aims to develop a personalized, rapid, and low-cost design and fabrication pipeline for creating body-conformable surfaces around the knee joint. The process begins with 3D scanning of the anterior knee surface of human subjects, followed by extracting the corresponding skin deformation between two joint angles in terms of longitudinal strain and Poisson's ratio. In parallel, a machine learning model is constructed using extensive simulation data from experimentally calibrated finite element analysis. This model employs Gaussian Process (GP) regression to relate kirigami cut lengths to the resulting longitudinal strain and Poisson's ratio. With an R2 score of 0.996, GP regression outperforms other models in predicting kirigami's large deformations. Finally, an inverse design approach based on the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is used to generate kirigami patch designs that replicate the in-plane skin deformation observed from the knee scans. This pipeline was applied to three human subjects, and the resulting kirigami knee patches were fabricated using rapid laser cutting, requiring only a business day from knee scanning to kirigami patch delivery. The low-cost, personalized kirigami patches successfully conformed to over 75 percent of the skin area across all subjects, establishing a foundation for a wide range of wearable devices. The study demonstrates this potential through an impact-resistant kirigami foam patch, which not only conforms to dynamic knee motion but also provides joint protection against impact. Finally, the proposed design and fabrication framework is generalizable and can be extended to other deforming body surfaces, enabling the creation of personalized wearables such as protective gear, breathable adhesives, and body-conformable electronics.

---

## 52. Thermodynamic Origin of the Tully-Fisher Relation in Dark Matter Dominated Galaxies: A Theoretical-Empirical Derivation

**论文链接:** [http://arxiv.org/abs/2511.13929v1](http://arxiv.org/abs/2511.13929v1)

**作者:** V. K. Oikonomou

**发布时间:** 2025-11-17

**备注:** Working paper. The arXiv entry contains an extended appendix with all the SPARC galaxies. The journal version will not include the appendix, only some characteristic galaxies will be presented

### GPT解析

### 总结

本研究引入了具有尺度依赖状态方程的自相互作用暗物质概念，通过双参数模型与174个星系数据进行对比，成功拟合了100个暗物质主导的矮星系旋转曲线，解决了尖核-核心问题，并半理论半经验地推导出Tully-Fisher关系。

### 背景

暗物质研究面临尖核-核心问题，需要新的暗物质模型来解释星系观测数据。

### 目的

开发一种具有尺度依赖状态方程的自相互作用暗物质模型，以更好地解释星系旋转曲线和解决现有暗物质模型的局限性。

### 方法

构建了一个有效的双参数模型，状态方程形式为P(r)=K(r)(ρ(r)/ρ_*)^γ(r)，其中K(r)=K_0×(1+r/r_c)^{-p}，并与SPARC数据库中的174个星系数据进行对比分析。

### 主要发现

1) 100个暗物质主导的矮星系、低光度、低表面亮度螺旋星系的旋转曲线被完美拟合；2) 该模型解决了尖核-核心问题；3) 半理论半经验地产生了经典Tully-Fisher关系和重子Tully-Fisher关系；4) 发现了相关性K_0∼V_max^2和K_0∼M_b^{0.5}；5) 对于暗物质主导星系，理论上发现K_0∼V_flat^2。

### 结论

尺度依赖的自相互作用暗物质模型能够很好地解释星系观测数据，特别是对于暗物质主导的矮星系，并且自然地导出了重要的星系尺度关系，为理解暗物质性质提供了新视角。

### 翻译

在这项工作中，我们引入了具有尺度依赖状态方程的自相互作用暗物质概念，在这种背景下，暗物质是碰撞性的，其状态方程是半径依赖的，形式为P(r)=K(r)(ρ(r)/ρ_*)^γ(r)。我们将有效的双参数模型与来自SPARC数据库的174个星系进行了对比，发现100个星系的旋转曲线可以被该模型完美拟合。这些星系是暗物质主导的，主要是矮星系、低光度和低表面亮度的螺旋星系。我们证明了尺度依赖的自相互作用暗物质解决了暗物质主导星系的尖核-核心问题。更重要的是，当考虑这100个可行的矮星系、低表面亮度和低光度星系时，尺度依赖的SIDM模型半理论半经验地产生了经典的Tully-Fisher关系和重子Tully-Fisher关系。熵函数K(r)的行为被假定为K(r)=K_0×(1+r/r_c)^{-p}。旋转曲线的完美拟合对应于近等温且处于流体静力平衡的暗物质晕，这自然预测了相关性K_0∼V_max^2。这种相关性在数据上得到经验证实，并且我们从数据中经验上也发现了L∼K_0^2，因此经典的Tully-Fisher关系被半理论半经验地再现。我们进行了同样的任务，理论上发现对于暗物质主导星系，K_0∼V_flat^2，这也从数据中得到经验证实，同时相关性K_0∼M_b^{0.5}也成立，因此重子Tully-Fisher定律以半理论半经验的方式自然出现。


### 论文摘要

In this work we introduce the concept of self-interacting dark matter with scale-dependent equation of state, in the context of which dark matter is collisional and its equation of state is radius-dependent and has the form $P(r)=K(r)\left(\frac{ρ(r)}{ρ_{\star}}\right)^{γ(r)}$. We confronted the effectively 2-parameter model with 174 galaxies from the SPARC data, and we found that the rotation curves of 100 galaxies can be perfectly fitted by the model. These galaxies are dark matter dominated, mostly dwarfs, low-luminosity and low-surface-brightness spiral galaxies. We demonstrate that scale-dependent self-interacting dark matter solves the cusp-core issue for dark matter dominated galaxies. More importantly, the structure of the scale-dependent SIDM model produces in a semi-theoretically and semi-empirically way the canonical Tully-Fisher and the baryonic Tully-Fisher relations when these 100 viable dwarfs, low-surface-brightness and low-luminosity galaxies are taken into account. The behavior of the entropy function $K(r)$ is assumed to be $K(r)=K_0\times\left(1+\frac{r}{r_c}\right)^{-p}$. The perfect fits of the rotation curves come for a nearly isothermal and virialized dark matter halo, which naturally predicts the correlation $K_0\sim V_{\mathrm{max}}^2$. This correlation holds true empirically as confirmed by the data and we also find empirically $L\sim K_0^2$ from the data, thus the canonical Tully-Fisher relation is reproduced semi-theoretically and semi-empirically. We perform the same task and we find theoretically, for dark matter dominated galaxies, that $K_0\sim V_{\mathrm{flat}}^2$ which is also confirmed empirically from the data, along with the correlation $K_0\sim M_b^{0.5}$, hence the baryonic Tully-Fisher law naturally emerges in a semi-theoretical and semi-empirical manner.

---

## 53. Computational study of irrational rotations via exact discontinuity tracking

**论文链接:** [http://arxiv.org/abs/2511.13879v1](http://arxiv.org/abs/2511.13879v1)

**作者:** Hannah Kravitz

**发布时间:** 2025-11-17

**备注:** 22 pages, 9 figures in main body, 10 figures in appendix

### GPT解析

### 总结

该论文提出了一种新的计算算法，用于精确计算无理数旋转的差异总和D_N(x,ρ)及其概率密度函数(pdf)，克服了数值不稳定性和采样方法的局限性，实现了从O(N^2)到O(N)的时间复杂度提升，并首次能够在O(N log N)时间内计算精确pdf及其关键性质。

### 背景

差异总和D_N(x,ρ)对于无理数旋转已引起数学家一个多世纪的兴趣。历史上研究主要在'几乎处处'或渐近意义上进行，而对于有限N的D_N越来越受到关注，因为它依赖于ρ的Diophantine性质，具有非平凡特性。D_N在N上相对于连分数收敛的商是周期性的，对于某些无理数增长迅速。

### 目的

稳定计算差异总和D_N(x,ρ)以形成关于其性质的猜想，并克服计算该和的确切值及其对应的概率密度函数(pdf)的困难，包括数值不稳定性和采样方法无法捕捉其跳跃不连续性的问题。

### 方法

提出了一种新的计算算法，通过不连续性完全定义了偏差函数及其相关的pdf，允许以O(N)时间和最小存储量计算D_N(x,ρ)到机器精度，显著提高了计算能力。

### 主要发现

算法首次能够在O(N log N)时间内直接计算精确pdf到机器精度，并能够计算偏差的关键性质：无穷范数(支持的一半)、二范数平方(pdf的方差)和pdf的峰度。算法能够产生清晰、精确的图形，有助于发展数学直觉和快速测试猜想。

### 结论

当ρ被有理数p_n/q_n良好近似时，当N=kq_n时，pdf表现出可预测的尖梯形模式。这些形状随着k的增加而退化，退化速度取决于p_n/q_n对ρ的近似程度。

### 翻译

无理数旋转的差异总和D_N(x,ρ)已引起数学家一个多世纪的兴趣。虽然历史上在'几乎处处'或渐近意义上进行研究，但对于有限N的D_N越来越因其依赖于ρ的Diophantine性质的非平凡特性而受到关注。D_N在N上相对于连分数收敛的商是周期性的，对于某些无理数增长迅速。因此，稳定计算这个和对于形成关于其性质的猜想是必要的。然而，由于数值不稳定性和采样方法无法捕捉其跳跃不连续性，计算该和的确切值及其对应的概率密度函数(pdf)非常困难。本文提出了一种新的计算算法，通过不连续性完全定义了偏差函数及其相关的pdf。这允许以O(N)时间和最小存储量计算D_N(x,ρ)到机器精度。这一计算能力的巨大改进(从O(N^2)的朴素版本)使得首次能够在O(N log N)时间内直接计算精确pdf到机器精度，并计算偏差的关键性质：无穷范数(支持的一半)、二范数平方(pdf的方差)和pdf的峰度。算法的一个关键优势在于它能够产生清晰、精确的图形，有助于发展数学直觉和快速测试猜想。例如，提出一个新猜想：当ρ被有理数p_n/q_n良好近似时，当N=kq_n时，pdf表现出可预测的尖梯形模式。这些形状随着k的增加而退化，退化速度取决于p_n/q_n对ρ的近似程度。


### 论文摘要

The discrepancy sum $D_N(x,ρ)$ for irrational rotations has been of interest to mathematicians for over a century. While historically studied in an ``almost-everywhere'' or asymptotic sense, $D_N$ for finite N is increasingly an object of interest for its nontrivial properties that depend on the Diophantine properties of $ρ$. This behavior is periodic in N with respect to the quotients of the continued fraction convergents, which grow quickly for some irrationals. Thus the stable computation of the sum is necessary for forming conjectures about its properties. However, computing the exact value of the sum and its corresponding probability density function (pdf) is notoriously difficult due to numerical instability in the sum itself and the failure of sampling methods to capture its jump discontinuities. This paper presents a novel computational algorithm that fully defines the discrepancy function and its associated pdf through its discontinuities. This allows the calculation of $D_N(x,ρ)$ to machine precision with minimal storage in O(N) time. This vast improvement in computability over the O(N^2) naive version enables, for the first time, the direct computation of the exact pdf up to machine precision in O(N log N) time, and with it, key properties of the discrepancy: $ \|D_N \|_{\infty}$ (half of the support of the pdf), $ \|D_N \|_{2}^2$ (the variance of the pdf), and the kurtosis of the pdf. A key strength of the algorithm lies in its ability to produce clear, exact figures, allowing the development of mathematical intuition and the quick testing of conjectures. As an example, a newly conjectured pattern is presented: when $ρ$ is well-approximated by rational $\frac{p_n}{q_n}$, the pdf exhibits a predictable spiked-trapezoidal pattern when $N=kq_n$. These shapes degrade as $k$ increases, at a speed depending on how well $\frac{p_n}{q_n}$ approximates $ρ$.

---

## 54. HMC: Learning Heterogeneous Meta-Control for Contact-Rich Loco-Manipulation

**论文链接:** [http://arxiv.org/abs/2511.14756v1](http://arxiv.org/abs/2511.14756v1)

**作者:** Lai Wei, Xuanbin Peng, Ri-Zhao Qiu, Tianshu Huang, Xuxin Cheng, Xiaolong Wang

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种异构元控制（HMC）框架，用于机器人自适应地结合多种控制模态，解决了纯位置控制器在处理复杂交互动力学时的局限性。

### 背景

从真实世界的机器人演示中学习对于与复杂现实世界环境交互具有前景，但交互动力学复杂性和变异性常常导致纯位置控制器在处理接触或变化负载时遇到困难。

### 目的

提出一种异构元控制（HMC）框架，用于适应性地拼接多种控制模态：位置、阻抗和混合力-位置控制，以提高机器人在复杂环境中的交互能力。

### 方法

1) 引入HMC-Controller接口，用于在扭矩空间中连续混合来自不同控制配置文件的动作；2) 提出HMC-Policy，将不同控制器统一为异构架构；3) 采用专家混合风格的路由，从大规模仅位置数据和细粒度力感知演示中学习。

### 主要发现

在真实人形机器人上的实验表明，在合规桌面擦拭和抽屉打开等具有挑战性的任务上，相比基线方法有超过50%的相对改进，证明了HMC框架的有效性。

### 结论

HMC框架能够有效地整合多种控制模态，显著提高机器人在复杂环境中的交互性能。

### 翻译

从真实世界的机器人演示中学习对于与复杂现实世界环境交互具有前景。然而，交互动力学的复杂性和变异性常常导致纯位置控制器在处理接触或变化负载时遇到困难。为此，我们提出了一种用于操作-移动的异构元控制（HMC）框架，能够自适应地拼接多种控制模态：位置、阻抗和混合力-位置控制。我们首先引入了一个HMC-Controller接口，用于在扭矩空间中连续混合来自不同控制配置文件的动作。HMC-Controller促进了遥操作和政策部署。然后，为了学习鲁棒的力感知策略，我们提出HMC-Policy，将不同控制器统一为异构架构。我们采用专家混合风格的路由，从大规模仅位置数据和细粒度力感知演示中学习。在真实人形机器人上的实验表明，在合规桌面擦拭和抽屉打开等具有挑战性的任务上，相比基线方法有超过50%的相对改进，证明了HMC的有效性。


### 论文摘要

Learning from real-world robot demonstrations holds promise for interacting with complex real-world environments. However, the complexity and variability of interaction dynamics often cause purely positional controllers to struggle with contacts or varying payloads. To address this, we propose a Heterogeneous Meta-Control (HMC) framework for Loco-Manipulation that adaptively stitches multiple control modalities: position, impedance, and hybrid force-position. We first introduce an interface, HMC-Controller, for blending actions from different control profiles continuously in the torque space. HMC-Controller facilitates both teleoperation and policy deployment. Then, to learn a robust force-aware policy, we propose HMC-Policy to unify different controllers into a heterogeneous architecture. We adopt a mixture-of-experts style routing to learn from large-scale position-only data and fine-grained force-aware demonstrations. Experiments on a real humanoid robot show over 50% relative improvement vs. baselines on challenging tasks such as compliant table wiping and drawer opening, demonstrating the efficacy of HMC.

---

## 55. LAUD: Integrating Large Language Models with Active Learning for Unlabeled Data

**论文链接:** [http://arxiv.org/abs/2511.14738v1](http://arxiv.org/abs/2511.14738v1)

**作者:** Tzu-Hsuan Chou, Chun-Nan Chou

**发布时间:** 2025-11-18

**备注:** 7 pages and one figure

### GPT解析

### 总结

该研究提出了LAUD学习框架，将大型语言模型与主动学习相结合，用于处理无标记数据集，通过零样本学习构建初始标签集来缓解冷启动问题，实验证明其在商品名称分类任务上优于传统方法。

### 背景

大型语言模型展现出超越预训练数据的泛化能力，微调后性能可达人类水平甚至更高。然而现实场景中缺乏标记数据常阻碍从业者获得高性能模型，迫使他们依赖繁琐、低效的试错式提示方法。

### 目的

为缓解缺乏标记数据的问题，提出一种有效训练和优化大型语言模型的解决方案。

### 方法

提出LAUD学习框架，整合大型语言模型与主动学习技术，利用零样本学习构建初始标签集，解决冷启动问题。

### 主要发现

实验结果表明，使用LAUD衍生的语言模型在商品名称分类任务上表现优于使用零样本或少样本学习的语言模型。

### 结论

LAUD框架为标记数据有限的情况提供了一种有效解决方案，能显著提升大型语言模型在分类任务上的性能。

### 翻译

大型语言模型(LLMs)已经展现出超越其预训练数据的出色泛化能力，并且微调LLMs可以将性能提升到人类水平甚至更高。然而，在现实场景中，缺乏标记数据常常阻止从业者获得高性能的模型，迫使他们严重依赖于通常繁琐、低效且依赖于试错的提示方法。为了缓解缺乏标记数据的问题，我们提出了一个将大型语言模型与主动学习相结合用于无标记数据集的学习框架(LAUD)。LAUD通过零样本学习构建初始标签集来缓解冷启动问题。实验结果表明，从LAUD衍生的语言模型在商品名称分类任务上优于使用零样本或少样本学习的语言模型，证明了LAUD的有效性。


### 论文摘要

Large language models (LLMs) have shown a remarkable ability to generalize beyond their pre-training data, and fine-tuning LLMs can elevate performance to human-level and beyond. However, in real-world scenarios, lacking labeled data often prevents practitioners from obtaining well-performing models, thereby forcing practitioners to highly rely on prompt-based approaches that are often tedious, inefficient, and driven by trial and error. To alleviate this issue of lacking labeled data, we present a learning framework integrating LLMs with active learning for unlabeled dataset (LAUD). LAUD mitigates the cold-start problem by constructing an initial label set with zero-shot learning. Experimental results show that LLMs derived from LAUD outperform LLMs with zero-shot or few-shot learning on commodity name classification tasks, demonstrating the effectiveness of LAUD.

---

## 56. ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents

**论文链接:** [http://arxiv.org/abs/2511.14584v1](http://arxiv.org/abs/2511.14584v1)

**作者:** Ankush Kadu, Ashwanth Krishnan

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文介绍了一种名为ReflexGrad的新型架构，通过整合三种互补机制实现了真正的零样本泛化，无需任务特定示例、微调或硬编码相似度度量。

### 背景

在强化学习和决策领域中，让智能体从经验中学习并能在不同任务间泛化，而无需针对特定任务进行训练，仍然是一个基本挑战。最近的方法分别探索了情景记忆、基于梯度的提示优化和分层任务分解，但这些方法的协同整合潜力尚未被探索。

### 目的

开发一种能够实现真正零样本泛化的系统，通过纯LLM语义推理，无需任务特定示例、微调或硬编码相似度度量。

### 方法

ReflexGrad架构紧密耦合三种互补机制：(1) 基于LLM的分层TODO分解用于战略规划；(2) 历史感知因果反思分析近期行动模式以识别失败根本原因并实现试验内学习；(3) 基于梯度的优化用于系统性改进。

### 主要发现

在ALFWorld基准任务上，ReflexGrad在试验0上实现了67%的零样本成功率，无需任何先前的任务经验或演示；通过经验分析确定了稳定收敛和有效跨任务转移(67%到78%改进)的基础架构机制。

### 结论

互补学习机制的协同整合能够实现接近先前少样本基线的鲁棒零样本泛化。

### 翻译

让智能体能够从经验中学习并在不同任务间泛化而无需针对特定任务进行训练，这仍然是强化学习和决策领域的一个基本挑战。虽然最近的方法分别探索了情景记忆、基于梯度的提示优化和分层任务分解，但它们协同整合的潜力仍未被探索。我们介绍了ReflexGrad，一种新型架构，它紧密耦合了三种互补机制：基于LLM的分层TODO分解用于战略规划；历史感知因果反思分析近期行动模式以识别失败根本原因并实现试验内学习；基于梯度的优化用于系统性改进。与依赖少样本演示的先前工作不同，我们的系统通过纯LLM语义推理实现了真正的零样本泛化，无需任务特定示例、微调或硬编码相似度度量。在ALFWorld基准任务上的评估显示，ReflexGrad在试验0上实现了67%的零样本成功率，无需任何先前的任务经验或演示，在首次接触时建立了有效性能。通过经验分析，我们确定了稳定收敛和有效跨任务转移的基础架构机制。我们的工作证明了互补学习机制的协同整合能够实现接近先前工作少样本基线的鲁棒零样本泛化。


### 论文摘要

Enabling agents to learn from experience and generalize across diverse tasks without task-specific training remains a fundamental challenge in reinforcement learning and decision-making. While recent approaches have explored episodic memory (Reflexion), gradient-based prompt optimization (TextGrad),and hierarchical task decomposition independently, their potential for synergistic integration remains unexplored. We introduce ReflexGrad, a novel architecture that tightly couples three complementary mechanisms: (1) LLM-based hierarchical TODO decomposition for strategic planning, (2) history-aware causal reflection that analyzes recent action patterns to identify failure root causes and enable within-trial learning, and (3) gradient-based optimization for systematic improvement. Unlike prior work relying on few-shot demonstrations, our system achieves true zero-shot generalization through pure LLM semantic reasoning,requiring no task-specific examples, fine-tuning, or hardcoded similarity metrics. Evaluated on ALFWorld benchmark tasks, ReflexGrad demonstrates 67% zero-shot success rate on Trial 0 without any prior task experience or demonstrations, establishing effective performance on first exposure. Through empirical analysis, we identify the architectural mechanisms underlying stable convergence (zero action loops) and effective cross-task transfer (67% to 78% improvement).Our work demonstrates that synergistic integration of complementary learning mechanisms enables robust zero-shot generalization that approaches few-shot baselines from prior work.

---

## 57. Mitigating Label Length Bias in Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.14385v1](http://arxiv.org/abs/2511.14385v1)

**作者:** Mario Sanz-Guerrero, Katharina von der Wense

**发布时间:** 2025-11-18

**备注:** Accepted to AACL 2025 (Main)

### GPT解析

### 总结

该研究针对大型语言模型在预测候选选项时存在的标签偏差问题，特别是多标记类别标签引起的'标签长度偏差'问题，提出了一种名为归一化上下文校准(NCC)的新方法。该方法在完整标签级别上进行预测归一化和校准，在多个数据集和模型上表现出色，F1值提升最高达10%。NCC不仅适用于传统分类任务，还能扩展到多项选择题回答等更广泛的任务，并且与上下文学习结合时表现出更好的鲁棒性和可靠性。

### 背景

大型语言模型(LLMs)是强大的零样本和少样本学习者。然而，当在一组候选选项中进行预测时，LLMs会受到标签偏差的影响，而现有的校准方法忽略了多标记类别标签引起的偏差。

### 目的

解决所谓的'标签长度偏差'问题，其中不同长度的标签被不一致地处理，即使在标准长度归一化之后。

### 方法

提出了归一化上下文校准(NCC)，这是一种在完整标签级别上归一化和校准预测的有效方法。

### 主要发现

NCC在多个数据集和模型上比先前方法取得了统计上显著的改进，F1值提升高达10%。此外，NCC将偏差缓解扩展到更广泛的任务，如多项选择题回答。与上下文学习结合时，NCC对少样本示例选择不那么敏感，需要更少的示例就能获得有竞争力的性能，并产生更可靠的置信度估计。

### 结论

缓解完整标签偏差对于提高基于LLM方法的性能和鲁棒性很重要，特别是在现实世界的应用中，因为类别标签自然地由多个标记组成。

### 翻译

大型语言模型(LLMs)是强大的零样本和少样本学习者。然而，当在一组候选选项中进行预测时，LLMs会受到标签偏差的影响，而现有的校准方法忽略了多标记类别标签引起的偏差。我们解决了一个称为标签长度偏差的问题，其中不同长度的标签被不一致地处理，即使在标准长度归一化之后。为了缓解这一问题，我们提出了归一化上下文校准(NCC)，这是一种在完整标签级别上归一化和校准预测的有效方法。NCC在多个数据集和模型上比先前方法取得了统计上显著的改进，F1值提升高达10%。此外，NCC将偏差缓解扩展到更广泛的任务，如多项选择题回答。我们的分析表明，当与上下文学习结合时，NCC对少样本示例选择不那么敏感，需要更少的示例就能获得有竞争力的性能，并产生更可靠的置信度估计。这些发现强调了缓解完整标签偏差对于提高基于LLM方法的性能和鲁棒性的重要性，特别是在现实世界的应用中，因为类别标签自然地由多个标记组成。


### 论文摘要

Large language models (LLMs) are powerful zero- and few-shot learners. However, when predicting over a set of candidate options, LLMs suffer from label biases, and existing calibration methods overlook biases arising from multi-token class labels. We tackle an issue we call label length bias, where labels of different lengths are treated inconsistently, even after standard length normalization. To mitigate it, we propose normalized contextual calibration (NCC), an effective method that normalizes and calibrates predictions at the full-label level. NCC achieves statistically significant improvements over prior approaches across multiple datasets and models, with gains of up to 10% F1. Moreover, NCC extends bias mitigation to broader tasks such as multiple-choice question answering. Our analysis shows that, when combined with in-context learning, NCC is less sensitive to few-shot example selection, requires fewer examples for competitive performance, and produces more reliable confidence estimates. These findings highlight the importance of mitigating full-label biases to improve the performance and robustness of LLM-based methods, particularly in real-world applications where class labels naturally consist of multiple tokens.

---

## 58. Weight Variance Amplifier Improves Accuracy in High-Sparsity One-Shot Pruning

**论文链接:** [http://arxiv.org/abs/2511.14282v1](http://arxiv.org/abs/2511.14282v1)

**作者:** Vincent-Daniel Yun, Junhyuk Jo, Sunwoo Lee

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种方差放大正则化器(VAR)，用于提高深度神经网络在剪枝后的鲁棒性，同时避免增加额外计算量，解决了现有剪枝鲁棒优化器计算成本高的问题。

### 背景

深度神经网络在视觉识别任务中表现出色但参数量大限制了实际应用；一次性剪枝是一种有效的模型压缩策略，但使用标准目标函数训练的模型在激进剪枝后精度会显著下降；现有的剪枝鲁棒优化器如SAM和CrAM虽能缓解精度下降但会增加额外计算量。

### 目的

提出一种不增加额外计算量的正则化方法，以提高深度神经网络在剪枝后的鲁棒性。

### 方法

提出方差放大正则化器(VAR)，在训练过程中有意增加模型参数的方差；通过促进权重分布中的高方差参数来减轻剪枝带来的负面影响；并提供VAR收敛行为的理论分析。

### 主要发现

参数方差越大，剪枝鲁棒性越强；VAR通过利用这一特性提高了模型的剪枝鲁棒性；实验结果证明了VAR在剪枝鲁棒性方面的优越性。

### 结论

VAR是一种有效的正则化方法，能够在不增加额外计算量的情况下显著提高深度神经网络在剪枝后的鲁棒性，解决了现有方法的局限性。

### 翻译

深度神经网络在视觉识别任务中取得了卓越的性能，但它们的大量参数使它们在现实世界应用中实用性较差。最近，一次性剪枝已成为一种在不进行额外训练的情况下减少模型大小的有效策略。然而，使用标准目标函数训练的模型在激进剪枝后往往会遭受显著的精度下降。一些现有的剪枝鲁棒优化器，如SAM和CrAM，通过引导模型朝向参数空间的更平坦区域来缓解这种精度下降，但它们不可避免地会产生不可忽略的额外计算。我们提出了一种方差放大正则化器(VAR)，它在训练过程中故意增加模型参数的方差。我们的研究揭示了一个有趣的发现：具有较高方差的参数表现出更强的剪枝鲁棒性。VAR通过在权重分布中促进这种方差来利用这一特性，从而减轻剪枝的不利影响。我们进一步提供了其收敛行为的理论分析，并通过大量实证结果证明了VAR的优越剪枝鲁棒性。


### 论文摘要

Deep neural networks achieve outstanding performance in visual recognition tasks, yet their large number of parameters makes them less practical for real-world applications. Recently, one-shot pruning has emerged as an effective strategy for reducing model size without additional training. However, models trained with standard objective functions often suffer a significant drop in accuracy after aggressive pruning. Some existing pruning-robust optimizers, such as SAM, and CrAM, mitigate this accuracy drop by guiding the model toward flatter regions of the parameter space, but they inevitably incur non-negligible additional computations. We propose a Variance Amplifying Regularizer (VAR) that deliberately increases the variance of model parameters during training. Our study reveals an intriguing finding that parameters with higher variance exhibit greater pruning robustness. VAR exploits this property by promoting such variance in the weight distribution, thereby mitigating the adverse effects of pruning. We further provide a theoretical analysis of its convergence behavior, supported by extensive empirical results demonstrating the superior pruning robustness of VAR.

---

## 59. PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval

**论文链接:** [http://arxiv.org/abs/2511.14130v1](http://arxiv.org/abs/2511.14130v1)

**作者:** Chun Chet Ng, Jia Yu Lim, Wei Zeng Low

**发布时间:** 2025-11-18

**备注:** 3rd-place solution for the ACM ICAIF 2025 Agentic Retrieval Grand Challenge

### GPT解析

### 总结

PRISM是一种无需训练的框架，通过整合改进的系统提示、上下文学习和轻量级多智能体系统，有效解决了金融信息检索问题。该框架在FinAgentBench数据集的文档排序和块排序任务上表现良好，NDCG@5达到0.71818，适用于实际生产环境。

### 背景

随着大语言模型的快速发展，金融信息检索已成为关键工业应用。从冗长的金融文件中提取任务相关信息对运营和分析决策至关重要。

### 目的

通过FinAgentBench数据集正式化金融信息检索问题，包含文档排序和块排序两个任务，并开发一种高效解决方案。

### 方法

提出PRISM框架，整合三个关键组件：1)改进的系统提示提供精确任务指令；2)上下文学习提供语义相关的少样本示例；3)轻量级多智能体系统建模协调评分行为。该框架无需训练，采用模块化、仅推理设计。

### 主要发现

PRISM的最佳配置在受限验证集上实现了0.71818的NDCG@5分数。研究表明各组件之间存在协同作用，且该框架在生产规模金融检索中可行且稳健。

### 结论

PRISM是一种实用、高效的金融信息检索解决方案，其模块化设计使其适用于实际用例。源代码已公开发布。

### 翻译

随着大语言模型（LLMs）的快速发展，金融信息检索已成为关键工业应用。从冗长的金融文件中提取任务相关信息对运营和分析决策至关重要。FinAgentBench数据集通过文档排序和块排序两个任务正式化了这一问题。我们提出了PRISM，一种无需训练的框架，整合了改进的系统提示、上下文学习（ICL）和轻量级多智能体系统。每个组件都经过广泛检查以揭示它们的协同作用：提示工程提供精确的任务指令，ICL提供语义相关的少样本示例，多智能体系统建模协调评分行为。我们的最佳配置在受限验证集上实现了0.71818的NDCG@5。我们进一步证明了PRISM在生产规模金融检索中可行且稳健。其模块化、仅推理设计使其适用于实际用例。源代码已发布在https://bit.ly/prism-ailens。


### 论文摘要

With the rapid progress of large language models (LLMs), financial information retrieval has become a critical industrial application. Extracting task-relevant information from lengthy financial filings is essential for both operational and analytical decision-making. The FinAgentBench dataset formalizes this problem through two tasks: document ranking and chunk ranking. We present PRISM, a training-free framework that integrates refined system prompting, in-context learning (ICL), and a lightweight multi-agent system. Each component is examined extensively to reveal their synergies: prompt engineering provides precise task instructions, ICL supplies semantically relevant few-shot examples, and the multi-agent system models coordinated scoring behaviour. Our best configuration achieves an NDCG@5 of 0.71818 on the restricted validation split. We further demonstrate that PRISM is feasible and robust for production-scale financial retrieval. Its modular, inference-only design makes it practical for real-world use cases. The source code is released at https://bit.ly/prism-ailens.

---

## 60. Zero-Training Task-Specific Model Synthesis for Few-Shot Medical Image Classification

**论文链接:** [http://arxiv.org/abs/2511.14082v1](http://arxiv.org/abs/2511.14082v1)

**作者:** Yao Qin, Yangyang Yan, YuanChao Yang, Jinhua Pang, Huanyong Bi, Yuan Liu, HaiHua Wang

**发布时间:** 2025-11-18

### GPT解析

### 总结

这项研究提出了一种名为零训练任务特定模型合成(ZS-TMS)的新方法，通过语义引导参数合成器(SGPS)，利用大规模预训练生成引擎直接合成特定任务分类器的参数，仅需少量输入（如单张示例图像和相应临床文本描述）即可生成轻量级高效分类器，无需任务特定训练或微调。该方法在少样本分类任务中取得了最先进的结果，特别适用于数据极其有限的罕见疾病诊断。

### 背景

深度学习模型在医学图像分析中取得了显著成功，但它们根本上受制于需要大规模、精心标注的数据集。这种对'大数据'的依赖是医学领域的关键瓶颈，因为患者数据本质上难以获取，专家标注成本高昂，特别是对于罕见疾病，样本数量本身就稀缺。

### 目的

克服医学深度学习对大规模标注数据的依赖，特别是解决罕见疾病样本稀缺的问题，实现无需大量数据训练的高效医学图像分析模型。

### 方法

提出零训练任务特定模型合成(ZS-TMS)范式，利用大规模预训练生成引擎直接合成特定任务分类器的全部参数。开发的语义引导参数合成器(SGPS)接收最少的多模态任务信息（如单张示例图像和相应临床文本描述），直接生成轻量级高效分类器（如EfficientNet-V2）的权重，无需任务特定训练或微调即可部署进行推理。

### 主要发现

在基于ISIC 2018皮肤病变数据集和自定义罕见疾病数据集的挑战性少样本分类基准上进行了广泛评估。结果表明，SGPS建立了新的最先进水平，显著优于先进的少样本和零样本学习方法，特别是在1样本和5样本分类的超低数据情况下。

### 结论

这项工作为AI诊断工具的快速开发和部署铺平了道路，特别适用于数据极其有限的罕见疾病'长尾'问题。

### 翻译

深度学习模型在医学图像分析中取得了显著成功，但它们根本上受制于需要大规模、精心标注的数据集。这种对'大数据'的依赖是医学领域的关键瓶颈，因为患者数据本质上难以获取，专家标注成本高昂，特别是对于罕见疾病，样本数量本身就稀缺。为了克服这一基本挑战，我们提出了一种新范式：零训练任务特定模型合成(ZS-TMS)。我们的方法不是调整现有模型或训练新模型，而是利用大规模预训练生成引擎直接合成特定任务分类器的全部参数。我们的框架——语义引导参数合成器(SGPS)——接收最少的多模态任务信息，如单张示例图像和相应临床文本描述，直接合成特定任务分类器的全部参数。生成引擎解释这些输入，为轻量级高效分类器（如EfficientNet-V2）生成权重，无需任何任务特定训练或微调即可立即部署进行推理。我们在基于ISIC 2018皮肤病变数据集和自定义罕见疾病数据集的挑战性少样本分类基准上进行了广泛评估。我们的结果表明，SGPS建立了新的最先进水平，显著优于先进的少样本和零样本学习方法，特别是在1样本和5样本分类的超低数据情况下。这项工作为AI诊断工具的快速开发和部署铺平了道路，特别适用于数据极其有限的罕见疾病'长尾'问题。


### 论文摘要

Deep learning models have achieved remarkable success in medical image analysis but are fundamentally constrained by the requirement for large-scale, meticulously annotated datasets. This dependency on "big data" is a critical bottleneck in the medical domain, where patient data is inherently difficult to acquire and expert annotation is expensive, particularly for rare diseases where samples are scarce by definition. To overcome this fundamental challenge, we propose a novel paradigm: Zero-Training Task-Specific Model Synthesis (ZS-TMS). Instead of adapting a pre-existing model or training a new one, our approach leverages a large-scale, pre-trained generative engine to directly synthesize the entire set of parameters for a task-specific classifier. Our framework, the Semantic-Guided Parameter Synthesizer (SGPS), takes as input minimal, multi-modal task information as little as a single example image (1-shot) and a corresponding clinical text description to directly synthesize the entire set of parameters for a task-specific classifier.   The generative engine interprets these inputs to generate the weights for a lightweight, efficient classifier (e.g., an EfficientNet-V2), which can be deployed for inference immediately without any task-specific training or fine-tuning. We conduct extensive evaluations on challenging few-shot classification benchmarks derived from the ISIC 2018 skin lesion dataset and a custom rare disease dataset. Our results demonstrate that SGPS establishes a new state-of-the-art, significantly outperforming advanced few-shot and zero-shot learning methods, especially in the ultra-low data regimes of 1-shot and 5-shot classification. This work paves the way for the rapid development and deployment of AI-powered diagnostic tools, particularly for the long tail of rare diseases where data is critically limited.

---

## 61. Meta-SimGNN: Adaptive and Robust WiFi Localization Across Dynamic Configurations and Diverse Scenarios

**论文链接:** [http://arxiv.org/abs/2511.14076v1](http://arxiv.org/abs/2511.14076v1)

**作者:** Qiqi Xiao, Ziqi Ye, Yinghui He, Jianwei Liu, Guanding Yu

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出Meta-SimGNN，一种结合图神经网络与元学习的新型WiFi定位系统，旨在解决设备配置变化对定位系统的影响，提高定位的泛化能力和鲁棒性。

### 背景

现有基于深度学习的定位研究通过元学习解决场景依赖性问题，但这些研究主要关注环境布局变化，忽视了设备配置（如带宽、接入点数量和使用天线数量）变化的影响。设备配置变化会影响信道状态信息的维度，损害神经网络可用性。

### 目的

解决设备配置变化对WiFi定位系统的影响，提高定位系统的泛化能力和鲁棒性，使系统能够适应不同设备配置和场景。

### 方法

1. 提出Meta-SimGNN系统，结合图神经网络和元学习；2. 引入细粒度CSI图构建方案，将每个接入点视为图节点以适应AP数量变化；3. 提出幅度-相位融合方法（利用幅度和相位构建CSI图像提高数据可靠性）和特征提取方法（提取维度一致特征解决带宽和天线数量变化问题）；4. 开发相似性引导的元学习策略，通过比较新场景与历史场景相似性确定微调初始参数。

### 主要发现

在不同场景下的商用WiFi设备上的大量实验结果表明，Meta-SimGNN在定位泛化能力和准确性方面优于基线方法。

### 结论

Meta-SimGNN通过结合图神经网络和元学习，有效解决了设备配置变化对WiFi定位系统的影响，提高了定位系统的泛化能力和鲁棒性。

### 翻译

为促进基于深度学习的定位实用性，现有研究通过元学习旨在解决场景依赖性问题。然而，这些研究主要关注环境布局的变化，而忽略了设备配置变化的影响，如带宽、接入点数量和使用天线数量。与环境变化不同，设备配置的变化会影响信道状态信息的维度，从而损害神经网络的可用性。为解决这一问题，我们提出了Meta-SimGNN，一种将图神经网络与元学习相结合的新型WiFi定位系统，以提高定位泛化能力和鲁棒性。首先，我们引入了一种细粒度的CSI图构建方案，其中每个接入点被视为图节点，以适应接入点数量的变化。为了结构化每个节点的特征，我们提出了一种幅度-相位融合方法和一种特征提取方法。前者利用幅度和相位构建CSI图像，提高数据可靠性；后者提取维度一致的特征，解决带宽和天线数量的变化问题。其次，开发了一种相似性引导的元学习策略，以增强在不同场景中的适应性。通过比较新场景与历史场景的相似性确定微调阶段的初始模型参数，促进模型快速适应新的定位场景。在不同场景下的商用WiFi设备上的大量实验结果表明，Meta-SimGNN在定位泛化能力和准确性方面优于基线方法。


### 论文摘要

To promote the practicality of deep learning-based localization, existing studies aim to address the issue of scenario dependence through meta-learning. However, these studies primarily focus on variations in environmental layouts while overlooking the impact of changes in device configurations, such as bandwidth, the number of access points (APs), and the number of antennas used. Unlike environmental changes, variations in device configurations affect the dimensionality of channel state information (CSI), thereby compromising neural network usability. To address this issue, we propose Meta-SimGNN, a novel WiFi localization system that integrates graph neural networks with meta-learning to improve localization generalization and robustness. First, we introduce a fine-grained CSI graph construction scheme, where each AP is treated as a graph node, allowing for adaptability to changes in the number of APs. To structure the features of each node, we propose an amplitude-phase fusion method and a feature extraction method. The former utilizes both amplitude and phase to construct CSI images, enhancing data reliability, while the latter extracts dimension-consistent features to address variations in bandwidth and the number of antennas. Second, a similarity-guided meta-learning strategy is developed to enhance adaptability in diverse scenarios. The initial model parameters for the fine-tuning stage are determined by comparing the similarity between the new scenario and historical scenarios, facilitating rapid adaptation of the model to the new localization scenario. Extensive experimental results over commodity WiFi devices in different scenarios show that Meta-SimGNN outperforms the baseline methods in terms of localization generalization and accuracy.

---

## 62. Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding

**论文链接:** [http://arxiv.org/abs/2511.13924v1](http://arxiv.org/abs/2511.13924v1)

**作者:** Qingyang Yan, Guangyao Chen, Yixiong Zou

**发布时间:** 2025-11-17

**备注:** AAAI 2026 (Oral)

### GPT解析

### 总结

本文提出了一种基于课程学习的相对策略优化(CuRPO)方法，用于解决强化学习微调的CoT推理在视觉定位任务中可能降低性能的问题，特别是在CoT输出冗长或复杂时。CuRPO利用CoT长度和gIoU奖励作为复杂度指标，将训练数据从简单到复杂进行结构化，在多个数据集上表现出色，具有高效率和鲁棒性。

### 背景

Chain-of-Thought (CoT) prompting通过明确生成中间推理步骤，在NLP和计算机视觉任务中显示出显著潜力。然而，基于强化学习的CoT微调在视觉定位任务中可能降低性能，特别是当CoT输出变得冗长或复杂时。此外，数据集大小增加并不总是提高性能，因为数据复杂度不同。

### 目的

提出一种新的训练策略来解决基于强化学习的CoT推理在视觉定位任务中可能降低性能的问题，特别是在CoT输出冗长或复杂的情况下。

### 方法

提出基于课程学习的相对策略优化(CuRPO)，利用CoT长度和广义交并比(gIoU)奖励作为复杂度指标，将训练数据从简单到复杂进行结构化，逐步提高训练难度。

### 主要发现

基于强化学习的CoT推理在视觉定位任务中可能降低性能，特别是当CoT输出变得冗长或复杂时；数据集大小增加并不总是提高性能，因为数据复杂度不同；CuRPO在多个数据集上优于现有方法，在RefCOCO上最高提升+12.52 mAP，并在少样本学习场景下表现出强大的定位性能。

### 结论

CuRPO是一种有效的训练策略，能够解决CoT推理在视觉定位任务中的性能下降问题，特别是在处理复杂和模糊的文本描述时表现出色，具有高效率和鲁棒性。

### 翻译

Chain-of-Thought (CoT) 提示最近通过显式生成中间推理步骤，在各种NLP和计算机视觉任务中显示出显著的前景。然而，我们发现基于强化学习(RL)的微调CoT推理在视觉定位任务中可能适得其反地降低性能，特别是当CoT输出变得冗长或复杂时。此外，我们的分析显示，数据集大小的增加并不总是提高性能，因为数据复杂度各不相同。受这些发现的启发，我们提出了基于课程学习的相对策略优化(CuRPO)，这是一种新颖的训练策略，利用CoT长度和广义交并比(gIoU)奖励作为复杂度指标，逐步将训练数据从简单到更复杂的例子进行结构化。在RefCOCO、RefCOCO+、RefCOCOg和LISA数据集上的广泛实验证明了我们方法的有效性。CuRPO始终优于现有方法，包括Visual-RFT，在RefCOCO上最高提升+12.52 mAP。此外，CuRPO表现出卓越的效率和鲁棒性，即使在少样本学习场景下也能提供强大的定位性能，特别有利于具有模糊复杂文本描述的任务。代码已发布在https://github.com/qyoung-yan/CuRPO。


### 论文摘要

Chain-of-Thought (CoT) prompting has recently shown significant promise across various NLP and computer vision tasks by explicitly generating intermediate reasoning steps. However, we find that reinforcement learning (RL)-based fine-tuned CoT reasoning can paradoxically degrade performance in Visual Grounding tasks, particularly as CoT outputs become lengthy or complex. Additionally, our analysis reveals that increased dataset size does not always enhance performance due to varying data complexities. Motivated by these findings, we propose Curriculum-based Relative Policy Optimization (CuRPO), a novel training strategy that leverages CoT length and generalized Intersection over Union (gIoU) rewards as complexity indicators to progressively structure training data from simpler to more challenging examples. Extensive experiments on RefCOCO, RefCOCO+, RefCOCOg, and LISA datasets demonstrate the effectiveness of our approach. CuRPO consistently outperforms existing methods, including Visual-RFT, with notable improvements of up to +12.52 mAP on RefCOCO. Moreover, CuRPO exhibits exceptional efficiency and robustness, delivering strong localization performance even in few-shot learning scenarios, particularly benefiting tasks characterized by ambiguous and intricate textual descriptions.The code is released on https://github.com/qyoung-yan/CuRPO.

---

## 63. SQL-to-Text Generation with Weighted-AST Few-Shot Prompting

**论文链接:** [http://arxiv.org/abs/2511.13907v1](http://arxiv.org/abs/2511.13907v1)

**作者:** Sriom Chakrabarti, Chuangtao Ma, Arijit Khan, Sebastian Link

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了一种Weighted-AST retrieval with prompting架构，通过整合结构化查询表示和大型语言模型提示技术，有效解决了SQL查询转换为自然语言描述时语义保持的问题。

### 背景

SQL-to-Text生成旨在将结构化SQL查询转换为自然语言描述，帮助非技术用户理解复杂数据库操作。尽管大型语言模型最近展示了有希望的结果，但现有方法往往无法保持SQL查询的确切语义，特别是当存在多种可能的正确表述时。

### 目的

解决当前方法在SQL查询语义保持方面的局限性，确保生成的自然语言描述既流畅又忠实于原始查询逻辑。

### 方法

提出Weighted-AST retrieval with prompting架构，使用基于带学习特征权重的抽象语法树(AST)的相似度指标来检索语义相关的示例作为少样本提示，结合结构感知提示技术确保生成的描述质量。

### 主要发现

在Spider、SParC和CoSQL三个基准数据集上的实验表明，该方法在执行准确度(EX)上比当前基线高出最多17.24%，在精确匹配(EM)方面表现更优，人类评估显示具有更一致的语义保真度，同时保持了有竞争力的运行时性能。

### 结论

Weighted-AST提示是一种可扩展且有效的方法，用于从结构化数据库查询中推导自然语言解释。

### 翻译

SQL到文本生成旨在将结构化的SQL查询转换为自然语言描述，从而使非技术用户能够理解复杂的数据库操作。尽管大型语言模型最近展示了有希望的结果，但当前方法往往无法保持SQL查询的确切语义，特别是当存在多种可能的正确表述时。为了解决这个问题，我们的工作提出了带提示的加权AST检索，这是一种结合结构化查询表示和LLM提示的架构。该方法使用基于带学习特征权重的抽象语法树(AST)的相似度指标，将语义相关的示例检索为少样本提示。我们的结构感知提示技术确保生成的描述既流畅又忠实于原始查询逻辑。在Spider、SparC和CoSQL三个基准数据集上进行的大量实验表明，我们的方法在执行准确度(EX)上比当前基线高出最多17.24%，在精确匹配(EM)方面表现更优，并且在人类评估时提供了更一致的语义保真度，同时保持了有竞争力的运行时性能。这些结果表明，加权AST提示是一种可扩展且有效的方法，用于从结构化数据库查询中推导自然语言解释。


### 论文摘要

SQL-to-Text generation aims at translating structured SQL queries into natural language descriptions, thereby facilitating comprehension of complex database operations for non-technical users. Although large language models (LLMs) have recently demonstrated promising results, current methods often fail to maintain the exact semantics of SQL queries, particularly when there are multiple possible correct phrasings. To address this problem, our work proposes Weighted-AST retrieval with prompting, an architecture that integrates structural query representations and LLM prompting. This method retrieves semantically relevant examples as few-shot prompts using a similarity metric based on an Abstract Syntax Tree (AST) with learned feature weights. Our structure-aware prompting technique ensures that generated descriptions are both fluent and faithful to the original query logic. Numerous experiments on three benchmark datasets - Spider, SParC, and CoSQL show that our method outperforms the current baselines by up to +17.24% in execution Accuracy (EX), performs superior in Exact Match (EM) and provides more consistent semantic fidelity when evaluated by humans, all while preserving competitive runtime performance. These results demonstrate that Weighted-AST prompting is a scalable and effective method for deriving natural language explanations from structured database queries.

---

## 64. Weakly Supervised Ephemeral Gully Detection In Remote Sensing Images Using Vision Language Models

**论文链接:** [http://arxiv.org/abs/2511.13891v1](http://arxiv.org/abs/2511.13891v1)

**作者:** Seyed Mohamad Ali Tousi, John A. Lory, G. N. DeSouza

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出首个用于检测农业田地中临时性冲沟的弱监督学习管道，利用视觉语言模型减少手动标记工作量，并创建了首个相关数据集。实验结果表明该方法优于传统方法。

### 背景

临时性冲沟是农业田地中最令人担忧的土壤侵蚀现象之一。其短暂的时间周期增加了使用传统计算机视觉和遥感方法自动检测的难度。此外，由于缺乏精确标记数据且难以生成准确标记数据，使用机器学习自动检测临时性冲沟的方法仅限于零样本方法，而这类方法难以实施。

### 目的

克服现有技术挑战，提出首个用于检测临时性冲沟的弱监督学习管道，减少手动标记的劳动强度，并创建首个用于从遥感图像半监督检测临时性冲沟的数据集。

### 方法

该方法基于遥感技术，利用视觉语言模型(VLMs)减少手动标记工作量。具体包括：1) 利用VLM预训练中嵌入的知识；2) 采用教师-学生模型架构，教师从VLMs提供的噪声标签中学习，学生通过弱监督使用教师生成的标签和感知噪声的损失函数进行学习。

### 主要发现

研究团队创建了首个用于半监督检测临时性冲沟的数据集，包含由科学家标记的多个位置和大量未标记位置，代表13年间获取的18,000多张高分辨率遥感图像。实验结果表明，当使用弱监督训练学生模型时，该方法比VLMs和标签模型本身表现更优。

### 结论

该弱监督学习方法为临时性冲沟的自动检测提供了有效解决方案，减少了手动标记的工作量，并提供了首个用于此类任务的数据集。代码和数据集已公开可用。

### 翻译

在土壤侵蚀问题中，临时性冲沟是农业田地中最令人担忧的现象之一。它们短暂的时间周期增加了使用传统计算机视觉方法和遥感技术自动检测它们的难度。此外，由于缺乏精确标记数据且难以生成准确标记数据，使用机器学习自动检测临时性冲沟的方法仅限于零样本方法，而这些方法难以实施。为了克服这些挑战，我们提出了首个用于检测临时性冲沟的弱监督管道。我们的方法依赖于遥感技术，并利用视觉语言模型(VLMs)来显著减少手动标记的劳动密集型任务。为此，该方法利用：1) VLM预训练中嵌入的知识；2) 教师-学生模型，其中教师从VLMs提供的噪声标签中学习，学生通过弱监督使用教师生成的标签和感知噪声的损失函数进行学习。我们还提供了首个用于从遥感图像半监督检测临时性冲沟的数据集。该数据集包含由一组土壤和植物科学家标记的多个位置，以及大量未标记位置。该数据集代表了13年间获取的超过18,000张高分辨率遥感图像。我们的实验结果通过比较显示，当使用弱监督训练学生模型时，我们的方法比VLMs和标签模型本身表现更优，证明了我们方法的有效性。本工作的代码和数据集已公开可用。


### 论文摘要

Among soil erosion problems, Ephemeral Gullies are one of the most concerning phenomena occurring in agricultural fields. Their short temporal cycles increase the difficulty in automatically detecting them using classical computer vision approaches and remote sensing. Also, due to scarcity of and the difficulty in producing accurate labeled data, automatic detection of ephemeral gullies using Machine Learning is limited to zero-shot approaches which are hard to implement. To overcome these challenges, we present the first weakly supervised pipeline for detection of ephemeral gullies. Our method relies on remote sensing and uses Vision Language Models (VLMs) to drastically reduce the labor-intensive task of manual labeling. In order to achieve that, the method exploits: 1) the knowledge embedded in the VLM's pretraining; 2) a teacher-student model where the teacher learns from noisy labels coming from the VLMs, and the student learns by weak supervision using teacher-generate labels and a noise-aware loss function. We also make available the first-of-its-kind dataset for semi-supervised detection of ephemeral gully from remote-sensed images. The dataset consists of a number of locations labeled by a group of soil and plant scientists, as well as a large number of unlabeled locations. The dataset represent more than 18,000 high-resolution remote-sensing images obtained over the course of 13 years. Our experimental results demonstrate the validity of our approach by showing superior performances compared to VLMs and the label model itself when using weak supervision to train an student model. The code and dataset for this work are made publicly available.

---

## 65. PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos

**论文链接:** [http://arxiv.org/abs/2511.12935v2](http://arxiv.org/abs/2511.12935v2)

**作者:** Dianbing Xi, Guoyuan An, Jingsen Zhu, Zhijian Liu, Yuan Liu, Ruiyuan Zhang, Jiayuan Lu, Yuchi Huo, Rui Wang

**发布时间:** 2025-11-17

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出PFAvatar（姿态融合头像），一种从每日穿搭(OOTD)照片中重建高质量3D头像的新方法，能够处理多样化姿势、遮挡和复杂背景。方法分为两个阶段：微调姿态感知扩散模型和蒸馏神经辐射场表示的3D头像。该方法只需5分钟即可完成个性化，比之前方法快48倍。

### 背景

现有3D头像重建方法通常将图像分割为资产（如服装、配饰）进行3D组装，容易产生不一致性。基于网格的表示方法存在分辨率依赖的离散化和错误的遮挡几何问题。

### 目的

开发一种从真实世界OOTD相册中生成实用3D头像的方法，解决现有方法在重建保真度、细节保留和对遮挡/截断的鲁棒性方面的局限性。

### 方法

方法包含两个阶段：1)从少量OOTD示例微调姿态感知扩散模型，避免分解直接建模全身外观，整合预训练ControlNet和条件先验保持损失(CPPL)；2)引入基于NeRF的头像表示，通过规范化SMPL-X空间采样和多分辨率3D-SDS优化，保留高频纹理并正确处理遮挡。

### 主要发现

实验表明，PFAvatar在重建保真度、细节保留和对遮挡/截断的鲁棒性方面优于最先进方法。重建的3D头像支持虚拟试穿、动画和人类视频重演等下游应用。

### 结论

PFAvatar推进了从真实世界OOTD相册中实用3D头像生成的发展，具有多功能性和实际价值。

### 翻译

我们提出了PFAvatar（姿态融合头像），一种新方法，可以从每日穿搭(OOTD)照片中重建高质量的3D头像，这些照片展示了多样化的姿势、遮挡和复杂背景。我们的方法包含两个阶段：(1)从少量OOTD示例微调姿态感知扩散模型，以及(2)蒸馏由神经辐射场(NeRF)表示的3D头像。在第一阶段，与之前将图像分割为资产（例如服装、配饰）进行3D组装的方法不同，后者容易出现不一致性，我们避免了分解并直接建模全身外观。通过整合用于姿态估计的预训练ControlNet和新的条件先验保持损失(CPPL)，我们的方法实现了端到端的细节学习，同时减少了少样本训练中的语言漂移。我们的方法只需5分钟即可完成个性化，比之前的方法实现了48倍的加速。在第二阶段，我们引入了一种基于NeRF的头像表示，通过规范化的SMPL-X空间采样和多分辨率3D-SDS进行优化。与基于网格的表示相比，后者容易受到分辨率依赖的离散化和错误的遮挡几何问题的影响，我们的连续辐射场可以通过透射保留高频纹理（例如头发）并正确处理遮挡。实验证明，PFAvatar在重建保真度、细节保留以及对遮挡/截断的鲁棒性方面优于最先进的方法，推动了从真实世界OOTD相册中实用3D头像生成的发展。此外，重建的3D头像支持虚拟试穿、动画和人类视频重演等下游应用，进一步证明了我们方法的多功能性和实际价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决从'每日穿搭'(OOTD)照片中重建高质量3D个性化头像的问题。这个问题在现实中很重要，因为人们每天都会拍摄大量这样的照片，包含丰富的个人外观信息；而在研究中，现有方法通常需要主体完全可见和精确相机校准，无法处理OOTD照片中的遮挡和截断问题。解决这一问题可以应用于虚拟试衣、动画、人体视频重演等多种下游应用，大大降低3D头像创建的门槛。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法如PuzzleAvatar的局限性，包括分割不一致、不支持姿势可控图像生成、训练时间长等问题。他们设计了两阶段方法：第一阶段(ControlBooth)提出姿势感知扩散模型，避免分解直接建模全身外观，使用ControlNet进行姿势估计并引入条件先验保持损失(CPPL)；第二阶段(BoothAvatar)从扩散模型中提炼NeRF表示的3D头像。该方法借鉴了ControlNet、Stable Diffusion、SDS和NeRF等现有工作，但进行了创新性整合和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是避免图像分解直接建模全身外观，使用姿势感知扩散模型结合姿势信息和文本描述进行端到端学习，并用NeRF而非网格表示3D头像以更好地处理遮挡和高频细节。流程分为两阶段：第一阶段ControlBooth预处理图像，预测姿势并生成文本描述，使用重建损失和CPPL微调扩散模型；第二阶段BoothAvatar从微调模型中提炼NeRF头像，使用SMPL-X空间采样和3D-SDS优化，引入局部几何约束和多分辨率采样策略提高质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)姿势感知扩散模型避免图像分解，直接建模全身外观；2)引入条件先验保持损失(CPPL)减少语言漂移；3)使用NeRF而非网格表示3D头像，更好处理遮挡和高频细节；4)3D一致的SDS优化确保3D一致性；5)局部几何约束保持精细结构。相比PuzzleAvatar，避免了分割不一致问题，训练时间从4小时缩短到5分钟，支持姿势可控生成；相比传统网格方法，NeRF能更好处理遮挡和复杂拓扑，保留更多细节。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PFAvatar通过姿势感知扩散模型和NeRF表示，实现了从日常穿搭照片中快速重建高质量、细节丰富且姿势可控的3D个性化头像，显著提升了重建质量和效率。'}


### 论文摘要

We propose PFAvatar (Pose-Fusion Avatar), a new method that reconstructs high-quality 3D avatars from Outfit of the Day(OOTD) photos, which exhibit diverse poses, occlusions, and complex backgrounds. Our method consists of two stages: (1) fine-tuning a pose-aware diffusion model from few-shot OOTD examples and (2) distilling a 3D avatar represented by a neural radiance field (NeRF). In the first stage, unlike previous methods that segment images into assets (e.g., garments, accessories) for 3D assembly, which is prone to inconsistency, we avoid decomposition and directly model the full-body appearance. By integrating a pre-trained ControlNet for pose estimation and a novel Condition Prior Preservation Loss (CPPL), our method enables end-to-end learning of fine details while mitigating language drift in few-shot training. Our method completes personalization in just 5 minutes, achieving a 48x speed-up compared to previous approaches. In the second stage, we introduce a NeRF-based avatar representation optimized by canonical SMPL-X space sampling and Multi-Resolution 3D-SDS. Compared to mesh-based representations that suffer from resolution-dependent discretization and erroneous occluded geometry, our continuous radiance field can preserve high-frequency textures (e.g., hair) and handle occlusions correctly through transmittance. Experiments demonstrate that PFAvatar outperforms state-of-the-art methods in terms of reconstruction fidelity, detail preservation, and robustness to occlusions/truncations, advancing practical 3D avatar generation from real-world OOTD albums. In addition, the reconstructed 3D avatar supports downstream applications such as virtual try-on, animation, and human video reenactment, further demonstrating the versatility and practical value of our approach.

---

## 66. Exploring Transferability of Self-Supervised Learning by Task Conflict Calibration

**论文链接:** [http://arxiv.org/abs/2511.13787v1](http://arxiv.org/abs/2511.13787v1)

**作者:** Huijie Guo, Jingyao Wang, Peizheng Guo, Xingchen Shen, Changwen Zheng, Wenwen Qiang

**发布时间:** 2025-11-16

### GPT解析

### 总结

本研究探讨了自监督学习的可迁移性，提出了任务冲突校准方法TC²，通过元学习范式构建多个SSL任务，解决了任务冲突问题，提高了模型表示的可迁移性。

### 背景

自监督学习（SSL）的可迁移性研究，即从一个任务中学到的表示支持另一个任务目标的能力。

### 目的

回答两个核心问题：SSL的表示可迁移性是什么，以及如何有效建模这种可迁移性。

### 方法

提出任务冲突校准（TC²）方法，包括分割批次创建多个SSL任务、使用因子提取网络生成因果生成因子、使用权重提取网络分配样本权重、利用数据重构和正交性确保有效性，并通过双层优化框架整合到训练过程中。

### 主要发现

引入任务级信息虽然提高了可迁移性，但仍受到任务冲突的阻碍；TC²方法能有效减轻这种冲突影响。

### 结论

TC²方法在多个下游任务上验证了能提高SSL模型的可迁移性。

### 翻译

本文通过解决两个核心问题探讨了自监督学习的可迁移性：（i）SSL的表示可迁移性是什么，以及（ii）如何有效建模这种可迁移性。可迁移性被定义为从一个任务中学到的表示支持另一个任务目标的能力。受元学习范式的启发，我们在每个训练批次中构建多个SSL任务来明确建模可迁移性。基于实证证据和因果分析，我们发现虽然引入任务级信息提高了可迁移性，但仍受到任务冲突的阻碍。为解决这个问题，我们提出了任务冲突校准（TC²）方法来减轻任务冲突的影响。具体来说，它首先分割批次创建多个SSL任务，注入任务级信息。然后，它使用因子提取网络为所有任务生成因果生成因子，使用权重提取网络为每个样本分配专门权重，采用数据重构、正交性和稀疏性确保有效性。最后，TC²在SSL训练期间校准样本表示，并通过双层优化框架整合到流程中，以提高学习表示的可迁移性。在多个下游任务上的实验结果表明，我们的方法一致地提高了SSL模型的可迁移性。


### 论文摘要

In this paper, we explore the transferability of SSL by addressing two central questions: (i) what is the representation transferability of SSL, and (ii) how can we effectively model this transferability? Transferability is defined as the ability of a representation learned from one task to support the objective of another.   Inspired by the meta-learning paradigm, we construct multiple SSL tasks within each training batch to support explicitly modeling transferability. Based on empirical evidence and causal analysis, we find that although introducing task-level information improves transferability, it is still hindered by task conflict. To address this issue, we propose a Task Conflict Calibration (TC$^2$) method to alleviate the impact of task conflict. Specifically, it first splits batches to create multiple SSL tasks, infusing task-level information. Next, it uses a factor extraction network to produce causal generative factors for all tasks and a weight extraction network to assign dedicated weights to each sample, employing data reconstruction, orthogonality, and sparsity to ensure effectiveness. Finally, TC$^2$ calibrates sample representations during SSL training and integrates into the pipeline via a two-stage bi-level optimization framework to boost the transferability of learned representations. Experimental results on multiple downstream tasks demonstrate that our method consistently improves the transferability of SSL models.

---

## 67. 3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology

**论文链接:** [http://arxiv.org/abs/2511.14613v1](http://arxiv.org/abs/2511.14613v1)

**作者:** Mohammad Vali Sanian, Arshia Hemmat, Amirhossein Vahidi, Jonas Maaskola, Jimmy Tsz Hang Lee, Stanislaw Makarchuk, Yeliz Demirci, Nana-Jane Chipampe, Omer Bayraktar, Lassi Paavolainen, Mohammad Lotfollahi

**发布时间:** 2025-11-18

**备注:** 11 pages

### GPT解析

### 总结

本文提出了HoloTea，一种3D感知的流匹配框架，可以从H&E染色图像中推断点水平的基因表达，并利用相邻切片信息提高3D转录组学分析的准确性和可扩展性。

### 背景

大多数预测算法独立处理组织切片并忽略3D结构，而现有的3D感知方法不是生成式的且扩展性不佳。

### 目的

开发一种可扩展且稳健的3D组织转录组学分析方法，以全面理解组织结构并提供对人类生物学和疾病更深入的见解。

### 方法

提出HoloTea框架，在共享特征空间中检索相邻切片上形态对应的点，将跨切片上下文融合到轻量级ControlNet中，引入结合ZINB先验和空间经验先验的3D一致先验条件，并使用全局注意力块实现与幻灯片中点数量成线性复杂度的3D H&E处理。

### 主要发现

在跨越不同组织类型和分辨率的三种空间转录组学数据集上，HoloTea相比2D和3D基线方法，在3D表达准确性和泛化能力方面都有一致的提升。

### 结论

HoloTea有望促进精确的3D虚拟组织的创建，最终加速生物标志物的发现并加深我们对疾病的理解。

### 翻译

可扩展且稳健的3D组织转录组学图谱能够实现对组织结构的全面理解，并为人类生物学和疾病提供更深入的见解。大多数直接从组织学推断空间转录组(ST)的预测算法独立处理每个切片并忽略3D结构，而现有的3D感知方法不是生成式的且扩展性不佳。我们提出了全息组织表达填充与分析(HoloTea)，这是一种3D感知的流匹配框架，可以从H&E染色图像中推断点水平的基因表达，同时明确使用相邻切片的信息。我们的核心思想是在共享特征空间中检索相邻幻灯片上形态上对应的点，并将这种跨切片上下文融合到轻量级ControlNet中，使条件能够遵循解剖连续性。为了更好地捕捉数据的计数特性，我们为流匹配引入了一个3D一致的先验条件，该先验结合了学习的零膨胀负二项式(ZINB)先验和从相邻切片构建的空间经验先验。全局注意力块引入了与幻灯片中点数量成线性关系的3D H&E，能够在大型3D ST数据集上进行训练和推理。在跨越不同组织类型和分辨率的三种空间转录组学数据集上，HoloTea相比2D和3D基线方法，在3D表达准确性和泛化能力方面都有一致的提升。我们期望HoloTea能够促进精确的3D虚拟组织的创建，最终加速生物标志物的发现并加深我们对疾病的理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从连续的组织学切片(H&E染色)生成三维空间转录组数据，重建组织体积的分子结构问题。这个问题很重要，因为现有空间转录组技术主要在2D切片上操作，视野有限，无法全面研究垂直细胞间通信、3D组织生态位和跨越多个平面的形态分子结构。通过解决这个问题，可以成本效益高地分析3D组织生态位，加速生物标志物发现，加深对健康和疾病中组织结构的理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了两个主要研究方向：整个切片生成建模与流匹配(如STFlow)和概率分配框架(如ST-Assign)，发现它们在3D重建方面存在两个关键差距：缺乏跨切片的3D一致性和未针对基因计数数据的特性调整起始分布。基于这些观察，作者设计了HoloTea方法，借鉴了流匹配、ControlNet、ZINB分布和全局集合注意力等现有技术，通过相邻切片条件化、z感知先验和可扩展全局注意机制来解决这些问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'HoloTea的核心思想是通过利用相邻切片的形态学信息提高3D一致性，具体包括：1)从相邻切片检索形态学对应的候选点；2)将这些上下文注入生成模型；3)使用生物学兼容的先验处理基因计数数据；4)使用可扩展的全局注意机制。整体流程包括：相邻切片条件化(检索候选点、计算相似性、构建相邻令牌、通过ControlNet注入)、z感知先验(预训练ZINB参数预测器)、全局集合注意力(两阶段全局上下文注入)和推理过程(初始化、时间步进、状态更新)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)相邻切片条件化方案，通过余弦相似性和ControlNet注入促进解剖连续性；2)z感知的生物学兼容先验，包括ZINB和空间-经验先验；3)稳定可扩展的全局注意力，与斑点数量线性扩展；4)显式3D一致性。相比STFlow，HoloTea扩展到连续切片区域并利用3D条件；相比ST-Assign，完全基于H&E不依赖外部参考，且计算成本线性增长；相比其他3D方法，避免了内存爆炸问题并在多数据集上表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HoloTea提出了一种创新的3D感知流匹配框架，通过利用相邻切片形态学信息和生物学兼容先验，实现了从连续组织学切片高效准确地生成三维空间转录组数据，为理解组织结构和疾病机制提供了强大工具。'}


### 论文摘要

A scalable and robust 3D tissue transcriptomics profile can enable a holistic understanding of tissue organization and provide deeper insights into human biology and disease. Most predictive algorithms that infer ST directly from histology treat each section independently and ignore 3D structure, while existing 3D-aware approaches are not generative and do not scale well. We present Holographic Tissue Expression Inpainting and Analysis (HoloTea), a 3D-aware flow-matching framework that imputes spot-level gene expression from H&E while explicitly using information from adjacent sections. Our key idea is to retrieve morphologically corresponding spots on neighboring slides in a shared feature space and fuse this cross section context into a lightweight ControlNet, allowing conditioning to follow anatomical continuity. To better capture the count nature of the data, we introduce a 3D-consistent prior for flow matching that combines a learned zero-inflated negative binomial (ZINB) prior with a spatial-empirical prior constructed from neighboring sections. A global attention block introduces 3D H&E scaling linearly with the number of spots in the slide, enabling training and inference on large 3D ST datasets. Across three spatial transcriptomics datasets spanning different tissue types and resolutions, HoloTea consistently improves 3D expression accuracy and generalization compared to 2D and 3D baselines. We envision HoloTea advancing the creation of accurate 3D virtual tissues, ultimately accelerating biomarker discovery and deepening our understanding of disease.

---

## 68. Feedback and Star Formation Efficiency in High-Mass Star-Forming Regions

**论文链接:** [http://arxiv.org/abs/2511.14557v1](http://arxiv.org/abs/2511.14557v1)

**作者:** Birka Zimmermann, Stefanie Walch, Seamus D. Clarke, Richard Wünsch, Andre Klepitko

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究通过全面模拟探索大质量恒星形成的参数空间，研究恒星反馈对恒星形成效率的影响，发现SFE在35%到57%之间，且恒星反馈是决定最终SFE的关键因素。

### 背景

为了推进对大质量恒星形成的理解，需要进行全面模拟以探索相关参数空间，并纳入足够物理以与观测数据比较。

### 目的

研究大质量恒星形成过程，特别是恒星反馈对恒星形成效率的影响。

### 方法

使用FLASH代码模拟孤立、秒差距尺度湍流核心的引力坍缩，将恒星建模为汇点；模拟包括电离辐射、辐射压力、非电离辐射和尘埃加热；独立计算尘埃、气体和辐射温度；初始条件基于ALMAGAL观测；评估恒星反馈并比较不同类型辐射的影响；检查空间分辨率和不同初始条件的影响。

### 主要发现

电离辐射阻止质量向恒星聚集，直接辐射压力增强HII区域扩张，非电离辐射加热抑制碎片化；更高分辨率导致更多恒星形成于高密度环境，使电离辐射保持更长时间，允许持续吸积，提高SFE；更平坦的密度分布、更高维里参数和更高金属丰度促进碎片化，可能通过减缓最 massive 恒星生长和延迟反馈来提高SFE；SFE在35%到57%之间。

### 结论

恒星反馈决定了最终的恒星形成效率。

### 翻译

为了推进我们对大质量恒星形成的理解，执行一套全面的模拟来探索相关参数空间并纳入足够物理以与观测数据进行比较是至关重要的。我们使用FLASH代码模拟孤立、秒差距尺度湍流核心的引力坍缩，将恒星建模为汇点。我们的模拟包括来自恒星的电离辐射及相关辐射压力，以及非电离辐射及其尘埃加热，同时包含自洽化学，以捕获新兴超致密HII区域的特性。尘埃、气体和辐射温度被独立计算。初始条件基于ALMAGAL观测。我们评估恒星反馈，比较电离辐射和辐射压力。电离辐射最终阻止质量向汇点聚集，而直接辐射压力增强HII区域的扩张。来自非电离辐射的加热抑制了碎片化。我们检查空间分辨率的影响，发现更高分辨率导致更多汇点，位于更高密度环境中。因此，电离辐射保持被困更长时间，允许持续吸积并产生更高的整体恒星形成效率(SFE)。我们探索了不同初始条件的影响，包括核心密度分布、维里参数和金属丰度。我们的参数研究表明，更平坦的密度分布、更高的维里参数和增加的金属丰度促进碎片化，可能通过减缓最 massive 恒星生长和延迟恒星反馈的起始来提高SFE。总体而言，我们发现SFE在35%到57%之间。恒星反馈决定了最终的SFE。


### 论文摘要

To advance our understanding of massive star formation, it is essential to perform a comprehensive suite of simulations that explore the relevant parameter space and include enough physics to enable a comparison with observational data. We simulate the gravitational collapse of isolated, parsec-scale turbulent cores using the FLASH code, modelling stars as sink particles. Our simulations incorporate ionizing radiation and the associated radiation pressure from stellar sources, and non-ionizing radiation and its dust heating, along with self-consistent chemistry, to capture the properties of emerging ultra-compact HII regions. Dust, gas, and radiation temperature are computed independently. The initial conditions are informed by ALMAGAL observations. We assess stellar feedback, comparing ionizing radiation and radiation pressure. Ionizing radiation ultimately halts mass accretion on to sink particles, while direct radiation pressure enhances the expansion of HII regions. Heating from non-ionizing radiation suppresses fragmentation. We examine the effect of spatial resolution, finding that higher resolution leads to more sink particles which are situated in environments with higher densities. As a result, ionizing radiation remains trapped longer, allowing continued accretion and yielding a higher overall star formation efficiency (SFE). We explore the impact of varying initial conditions, including the core density profile, virial parameter, and metallicity. Our parameter study reveals that a flatter density profile, higher virial parameter, and increased metallicity promote fragmentation, potentially enhancing the SFE by slowing the growth of the most massive stars and delaying the onset of stellar feedback. Overall, we find SFEs between 35% and 57%. Stellar feedback dictates the final SFE.

---

## 69. ArchMap: Arch-Flattening and Knowledge-Guided Vision Language Model for Tooth Counting and Structured Dental Understanding

**论文链接:** [http://arxiv.org/abs/2511.14336v1](http://arxiv.org/abs/2511.14336v1)

**作者:** Bohan Zhang, Yiyi Miao, Taoyu Wu, Tong Chen, Ji Jiang, Zhuoxiao Li, Zhe Tang, Limin Yu, Jionglong Su

**发布时间:** 2025-11-18

### GPT解析

### 总结

ArchMap是一种无需训练且知识引导的框架，通过几何标准化和本体引导的多模态推理，实现了对口腔3D扫描的稳健结构化理解，在数字正畸学中表现出色。

### 背景

现有深度学习方法依赖特定模态训练、大量标注数据和受控扫描条件，限制了跨设备泛化和临床部署；原始口腔网格存在弓形姿势变化、不完整几何结构和缺乏纹理等问题，使统一语义解释极具挑战性。

### 目的

解决现有方法的局限性，提出一种稳健的结构化口腔理解框架，能够在真实临床环境中有效工作。

### 方法

提出ArchMap框架，包含几何感知的弓形展平模块将原始3D网格标准化为空间对齐的多视图投影；构建牙科知识库编码层次化牙齿本体、牙列阶段策略和临床语义，以约束符号推理空间。

### 主要发现

在1060例正畸病例验证中，ArchMap在牙齿计数、解剖分区、牙列阶段分类和临床状况识别（拥挤、缺牙、假牙和龋齿）方面表现稳健；相比监督流程和VLM基线，具有更高准确性、减少语义漂移和优越稳定性。

### 结论

作为完全无需训练的系统，ArchMap证明了几何标准化与本体引导多模态推理相结合，为现代数字正畸学中3D口腔扫描的结构化分析提供了实用且可扩展的解决方案。

### 翻译

口腔3D扫描的结构化理解对数字正畸学至关重要。然而，现有的深度学习方法严重依赖特定模态的训练、大量标注的数据集和受控的扫描条件，这限制了跨设备的泛化能力，并阻碍了在真实临床工作流程中的部署。此外，原始的口腔网格在弓形姿势上存在显著差异，由遮挡或牙齿接触导致的不完整几何结构，以及缺乏纹理线索，使得统一的语义解释极具挑战性。为解决这些局限性，我们提出了ArchMap，一种无需训练且知识引导的框架，用于稳健的结构化口腔理解。ArchMap首先引入了一种几何感知的弓形展平模块，将原始3D网格标准化为空间对齐、保持连续性的多视图投影。然后，我们构建了一个牙科知识库，编码层次化的牙齿本体、牙列阶段策略和临床语义，以约束符号推理空间。我们在1060例正畸前后的病例上验证了ArchMap，证明了其在牙齿计数、解剖分区、牙列阶段分类以及临床状况（如拥挤、缺牙、假牙和龋齿）识别方面的稳健性能。与监督流程和提示VLM基线相比，ArchMap实现了更高的准确性、减少的语义漂移，以及在稀疏或伪影条件下的优越稳定性。作为一个完全无需训练的系统，ArchMap证明了将几何标准化与本体引导的多模态推理相结合，为现代数字正畸学中3D口腔扫描的结构化分析提供了实用且可扩展的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决口腔3D扫描数据结构化理解的问题，特别是现有深度学习方法依赖特定模态训练、大量标注数据和受控扫描条件，导致跨设备泛化能力差、难以在真实临床工作流程中部署的问题。这个问题在数字正畸学中非常重要，因为口腔3D扫描的结构化理解是诊断、治疗计划和疗效评估的基础，解决这些问题能推动数字正畸技术的发展和应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法对特定模态训练和大量标注数据的依赖限制了临床应用，然后思考如何在不依赖大量训练数据的情况下解决口腔3D扫描的几何变化和不完整性问题。他们借鉴了现有工作中的几何处理技术、多视图渲染方法、视觉-语言模型在牙科的应用(如DentVLM)以及监督和弱监督学习方法，但创新性地将这些技术组合成一个无需训练的框架，通过几何标准化与知识引导推理相结合的方式解决核心挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合几何标准化与本体引导的多模态推理，实现无需训练即可对口腔3D扫描进行结构化理解。整体实现流程分为三部分：1)几何感知的弓形展平，将3D口腔网格转换为空间对齐的多视图2D投影；2)构建牙齿知识库(DKB)，编码层次化牙齿本体、萌发阶段和临床语义；3)模式约束的视觉-语言推理，通过五个层次推理阶段(牙齿计数、解剖分类、大小分类、萌发阶段确定、临床条件识别)实现结构化输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)几何感知的弓形展平策略，将3D牙齿网格转换为保持连续性的多视图2D投影；2)特定任务的牙齿知识库(DKB)，提供符号约束和任务先验；3)零训练、模式约束的视觉-语言推理框架。相比之前的工作，ArchMap无需大量标注数据或特定模态训练，能处理口腔网格的几何变化和不完整性，提供统一、可解释且临床一致的输出，在稀疏或伪影条件下表现出更好的稳定性和准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ArchMap通过几何弓形展平与牙齿知识库引导的视觉-语言推理，实现了无需训练即可对口腔3D扫描进行准确、稳定的结构化牙科理解，为数字正畸提供了实用且可扩展的解决方案。'}


### 论文摘要

A structured understanding of intraoral 3D scans is essential for digital orthodontics. However, existing deep-learning approaches rely heavily on modality-specific training, large annotated datasets, and controlled scanning conditions, which limit generalization across devices and hinder deployment in real clinical workflows. Moreover, raw intraoral meshes exhibit substantial variation in arch pose, incomplete geometry caused by occlusion or tooth contact, and a lack of texture cues, making unified semantic interpretation highly challenging. To address these limitations, we propose ArchMap, a training-free and knowledge-guided framework for robust structured dental understanding. ArchMap first introduces a geometry-aware arch-flattening module that standardizes raw 3D meshes into spatially aligned, continuity-preserving multi-view projections. We then construct a Dental Knowledge Base (DKB) encoding hierarchical tooth ontology, dentition-stage policies, and clinical semantics to constrain the symbolic reasoning space. We validate ArchMap on 1060 pre-/post-orthodontic cases, demonstrating robust performance in tooth counting, anatomical partitioning, dentition-stage classification, and the identification of clinical conditions such as crowding, missing teeth, prosthetics, and caries. Compared with supervised pipelines and prompted VLM baselines, ArchMap achieves higher accuracy, reduced semantic drift, and superior stability under sparse or artifact-prone conditions. As a fully training-free system, ArchMap demonstrates that combining geometric normalization with ontology-guided multimodal reasoning offers a practical and scalable solution for the structured analysis of 3D intraoral scans in modern digital orthodontics.

---

## 70. Let Language Constrain Geometry: Vision-Language Models as Semantic and Spatial Critics for 3D Generation

**论文链接:** [http://arxiv.org/abs/2511.14271v1](http://arxiv.org/abs/2511.14271v1)

**作者:** Weimin Bai, Yubo Li, Weijian Luo, Zeqiang Lai, Yequan Wang, Wenzheng Chen, He Sun

**发布时间:** 2025-11-18

### GPT解析

### 总结

VLM3D是一个通用框架，将大型视觉语言模型重新用作可微分的语义和空间评判器，通过双查询评判信号解决文本到3D生成中的语义对齐和空间理解问题。

### 背景

文本到3D生成技术发展迅速，但最先进的模型（包括基于优化和前馈架构的模型）仍面临两个基本限制：难以处理粗略语义对齐和缺乏稳健的3D空间理解。

### 目的

解决现有文本到3D生成模型在语义对齐和空间理解方面的局限性，提高生成质量和准确性。

### 方法

提出VLM3D框架，利用大型视觉语言模型作为可微分的语义和空间评判器，核心是从VLM的是/否对数几率中派生的双查询评判信号，评估语义保真度和几何一致性。

### 主要发现

该指导信号在两种范式中有效：(1)作为基于优化管道的奖励目标，在标准基准上显著优于现有方法；(2)作为前馈管道的测试时指导模块，能纠正严重空间错误。

### 结论

VLM3D为将VLM丰富的、以语言为基础的语义和空间理解注入各种3D生成管道提供了一条原则性和可推广的路径。

### 翻译

文本到3D生成技术已取得快速发展，然而最先进的模型（包括基于优化和前馈架构的模型）仍面临两个基本限制。首先，它们难以处理粗略的语义对齐，常常无法捕捉精细的提示细节。其次，它们缺乏稳健的3D空间理解，导致几何不一致以及在部件组装和空间关系上的灾难性失败。为应对这些挑战，我们提出了VLM3D，这是一个通用框架，将大型视觉语言模型（VLMs）重新用作强大的、可微分的语义和空间评判器。我们的核心贡献是来自VLM的是/否对数几率的双查询评判信号，用于评估语义保真度和几何一致性。我们展示了该指导信号在两种不同范式中的通用性：(1)作为基于优化管道的奖励目标，VLM3D在标准基准上显著优于现有方法。(2)作为前馈管道的测试时指导模块，它主动引导SOTA原生3D模型的迭代采样过程，以纠正严重的空间错误。VLM3D为将VLM丰富的、以语言为基础的语义和空间理解注入各种3D生成管道建立了一条原则性和可推广的路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决文本到3D生成中的两个核心问题：一是语义对齐问题，即模型难以捕捉提示中的细粒度细节；二是空间理解问题，即模型缺乏稳健的3D空间理解能力，导致几何不一致和部件组装失败。这些问题在现实中很重要，因为3D内容创作在游戏、电影、AR/VR等领域有广泛应用需求，而当前生成的3D模型经常存在语义错误（如缺少关键元素）和几何问题（如部件悬浮、断裂），严重限制了3D生成技术的实用性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别出现有文本到3D生成方法的局限性：基于优化的方法继承2D扩散模型的弱语义基础和 poor空间意识；前馈方法受限于3D训练数据的复杂性（通常是单一对象而非多对象场景）。然后，作者利用大型视觉语言模型(VLM)的优势，它们具有丰富的、基于语言理解的语义和空间关系理解能力。作者设计了一个双查询批评信号，基于VLM的'Yes/No'对数几率，同时评估语义保真度和几何一致性。该方法借鉴了现有的基于优化的文本到3D生成方法（如SDS）、前馈模型（如Hunyuan3D）、视觉语言模型（如Qwen2.5-VL）以及人类偏好优化的方法（如DreamReward）。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将大型视觉语言模型(VLM)重新用作可微分的语义和空间批评家，通过设计双查询批评信号（同时评估语义保真度和几何一致性），并灵活集成到两种不同的生成范式中。整体实现流程分为两种：1）对于基于优化的管道，将VLM3D作为奖励目标集成到SDS损失中，使用动态调度平衡VLM奖励和SDS损失，通过反向传播直接优化3D表示；2）对于前馈模型，将VLM3D作为测试时指导模块，在迭代采样过程中使用VLM梯度修改采样方向，无需重新训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出VLM3D框架，首次建立将VLM的丰富、基于语言的理解注入到3D生成管道的通用路径；2）设计双查询批评信号，同时评估语义保真度和几何/空间一致性；3）展示批评信号在两种不同生成范式中的通用性。相比之前工作的不同之处在于：相比传统的CLIP风格编码器，VLM提供更细粒度的语义理解和空间推理能力；相比2D中心的扩散模型先验，VLM提供更丰富的语义和空间理解；相比仅使用人类反馈训练的3D奖励模型，VLM提供更广泛的语义和空间知识；同时解决了语义对齐和空间理解两个核心问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VLM3D创新性地将大型视觉语言模型转化为可微分的语义和空间批评家，通过双查询设计显著提升了文本到3D生成的语义准确性和几何一致性，并在两种不同生成范式中展示了其通用性。'}


### 论文摘要

Text-to-3D generation has advanced rapidly, yet state-of-the-art models, encompassing both optimization-based and feed-forward architectures, still face two fundamental limitations. First, they struggle with coarse semantic alignment, often failing to capture fine-grained prompt details. Second, they lack robust 3D spatial understanding, leading to geometric inconsistencies and catastrophic failures in part assembly and spatial relationships. To address these challenges, we propose VLM3D, a general framework that repurposes large vision-language models (VLMs) as powerful, differentiable semantic and spatial critics. Our core contribution is a dual-query critic signal derived from the VLM's Yes or No log-odds, which assesses both semantic fidelity and geometric coherence. We demonstrate the generality of this guidance signal across two distinct paradigms: (1) As a reward objective for optimization-based pipelines, VLM3D significantly outperforms existing methods on standard benchmarks. (2) As a test-time guidance module for feed-forward pipelines, it actively steers the iterative sampling process of SOTA native 3D models to correct severe spatial errors. VLM3D establishes a principled and generalizable path to inject the VLM's rich, language-grounded understanding of both semantics and space into diverse 3D generative pipelines.

---

## 71. LLM-Aligned Geographic Item Tokenization for Local-Life Recommendation

**论文链接:** [http://arxiv.org/abs/2511.14221v1](http://arxiv.org/abs/2511.14221v1)

**作者:** Hao Jiang, Guoquan Wang, Donglin Zhou, Sheng Yu, Yang Zeng, Wencong Zeng, Kun Gai, Guorui Zhou

**发布时间:** 2025-11-18

### GPT解析

### 总结

LGSID是一种创新的面向本地生活推荐的LLM对齐地理项目标记化框架，通过结合强化学习对齐和分层地理项目标记化策略，有效捕捉项目之间的细粒度空间特征和现实世界距离感知。

### 背景

大型语言模型（LLMs）的最新进展通过增强传统基于ID的方法的语义泛化能力，提升了基于文本的推荐。然而，在本地生活服务等特定领域任务中，简单将位置信息注入提示无法捕捉项目之间的细粒度空间特征和现实世界距离感知。

### 目的

提出LGSID框架，解决本地生活服务推荐中空间特征捕捉不足的问题，提高推荐系统的性能。

### 方法

框架包含两个关键组件：1）基于强化学习的地理LLM对齐模块，通过训练列表级奖励模型和引入G-DPO算法注入空间知识；2）分层地理项目标记化策略，主要标记源自离散空间和内容属性，残余标记使用对齐的LLM的地理表示向量优化。

### 主要发现

在快手真实世界行业数据集上的大量实验表明，LGSID持续优于最先进的判别性和生成性推荐模型，消融研究、可视化和案例研究进一步验证了其有效性。

### 结论

LGSID框架有效地解决了本地生活服务推荐中空间特征捕捉不足的问题，通过结合强化学习对齐和分层地理项目标记化，显著提高了推荐系统的性能。

### 翻译

大型语言模型（LLMs）的最新进展通过增强传统基于ID的方法的语义泛化能力，提升了基于文本的推荐。基于文本的方法通常通过提示设计编码项目文本信息，并通过项目标记化生成离散语义ID。然而，在本地生活服务等特定领域任务中，简单将位置信息注入提示无法捕捉项目之间的细粒度空间特征和现实世界距离感知。为此，我们提出了LGSID，一个面向本地生活推荐的LLM对齐地理项目标记化框架。该框架包含两个关键组件：（1）基于强化学习的地理LLM对齐，和（2）分层地理项目标记化。在基于强化学习的对齐模块中，我们首先训练一个列表级奖励模型来捕捉项目之间的现实世界空间关系。然后，我们引入了一种新的G-DPO算法，该算法使用预训练的奖励模型将泛化的空间知识和协作信号注入LLMs，同时保留其语义理解能力。此外，我们提出了一种分层地理项目标记化策略，其中主要标记源自离散的空间和内容属性，而残余标记则使用对齐的LLM的地理表示向量进行优化。在快手真实世界行业数据集上的大量实验表明，LGSID持续优于最先进的判别性和生成性推荐模型。消融研究、可视化和案例研究进一步验证了其有效性。


### 论文摘要

Recent advances in Large Language Models (LLMs) have enhanced text-based recommendation by enriching traditional ID-based methods with semantic generalization capabilities. Text-based methods typically encode item textual information via prompt design and generate discrete semantic IDs through item tokenization. However, in domain-specific tasks such as local-life services, simply injecting location information into prompts fails to capture fine-grained spatial characteristics and real-world distance awareness among items. To address this, we propose LGSID, an LLM-Aligned Geographic Item Tokenization Framework for Local-life Recommendation. This framework consists of two key components: (1) RL-based Geographic LLM Alignment, and (2) Hierarchical Geographic Item Tokenization. In the RL-based alignment module, we initially train a list-wise reward model to capture real-world spatial relationships among items. We then introduce a novel G-DPO algorithm that uses pre-trained reward model to inject generalized spatial knowledge and collaborative signals into LLMs while preserving their semantic understanding. Furthermore, we propose a hierarchical geographic item tokenization strategy, where primary tokens are derived from discrete spatial and content attributes, and residual tokens are refined using the aligned LLM's geographic representation vectors. Extensive experiments on real-world Kuaishou industry datasets show that LGSID consistently outperforms state-of-the-art discriminative and generative recommendation models. Ablation studies, visualizations, and case studies further validate its effectiveness.

---

## 72. Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution

**论文链接:** [http://arxiv.org/abs/2511.14210v1](http://arxiv.org/abs/2511.14210v1)

**作者:** N Dinesh Reddy, Sudeep Pillai

**发布时间:** 2025-11-18

### GPT解析

### 总结

Orion是一个多模态视觉智能体框架，可以接收任何模态输入并生成任何模态输出，专为视觉AI任务设计并取得了最先进结果。

### 背景

传统视觉-语言模型仅能生成描述性输出，无法满足复杂多步骤视觉工作流的需求。

### 目的

开发能够执行复杂多步骤视觉工作流的智能体框架，实现从被动视觉理解到主动、工具驱动视觉智能的转变。

### 方法

Orion使用具有多种工具调用能力的智能体框架，整合目标检测、关键点定位、全景分割、光学字符识别和几何分析等专业计算机视觉工具，结合神经感知与符号执行实现自主视觉推理。

### 主要发现

Orion在MMMU、MMBench、DocVQA和MMLongBench等基准测试上取得了具有竞争力的性能，将单一视觉-语言模型扩展为生产级视觉智能。

### 结论

Orion通过结合神经感知与符号执行，实现了自主视觉推理，标志着视觉AI从被动理解向主动、工具驱动智能的转变。

### 翻译

我们引入Orion，这是一个可以接收任何模态并生成任何模态的视觉智能体框架。使用具有多种工具调用能力的智能体框架，Orion专为视觉AI任务设计，并取得了最先进的结果。与仅生成描述性输出的传统视觉-语言模型不同，Orion协调了一系列专业的计算机视觉工具套件，包括目标检测、关键点定位、全景分割、光学字符识别和几何分析，以执行复杂的多步骤视觉工作流。该系统在MMMU、MMBench、DocVQA和MMLongBench上取得了具有竞争力的性能，同时将单一的视觉-语言模型扩展为生产级的视觉智能。通过结合神经感知与符号执行，Orion实现了自主视觉推理，标志着从被动视觉理解向主动、工具驱动的视觉智能的转变。


### 论文摘要

We introduce Orion, a visual agent framework that can take in any modality and generate any modality. Using an agentic framework with multiple tool-calling capabilities, Orion is designed for visual AI tasks and achieves state-of-the-art results. Unlike traditional vision-language models that produce descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition, and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance on MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic vision-language models to production-grade visual intelligence. By combining neural perception with symbolic execution, Orion enables autonomous visual reasoning, marking a transition from passive visual understanding to active, tool-driven visual intelligence.

---

## 73. Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning Framework with Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.14120v1](http://arxiv.org/abs/2511.14120v1)

**作者:** Hao Zhen, Yunxiang Yang, Jidong J. Yang

**发布时间:** 2025-11-18

**备注:** 23 pages, 4 figures, 3 tables

### GPT解析

### 总结

本文提出了MP-PVIR框架，通过四个阶段处理多视图视频流，将行人-车辆事故转化为结构化诊断报告，实现了对行人行为认知阶段的精确分析。

### 背景

行人-车辆事故仍是城市安全重大挑战，行人占全球交通死亡人数20%以上。现有视频系统可检测事故但难以分析行为认知阶段，视觉-语言模型通常单独处理视频缺乏时间结构或多视图集成。

### 目的

开发一个统一框架，系统处理多视图视频流生成结构化诊断报告，深入理解行人-车辆事故的行为认知阶段。

### 方法

MP-PVIR框架包含四个阶段：(1)事件触发多视图视频采集；(2)行人行为阶段分割；(3)阶段特定多视图推理；(4)分层综合和诊断推理。使用TG-VLM进行行为阶段分割(mIoU=0.4881)和PhaVR-VLM进行阶段感知多视图分析(标题分数33.063，问答准确率64.70%)，最后由大语言模型生成综合报告。

### 主要发现

MP-PVIR能将多视图视频数据有效转化为可操作见解，在Woven Traffic Safety数据集上表现优异，推进了AI驱动的交通安全分析。

### 结论

MP-PVIR框架成功将多视图视频数据转化为可操作的见解，为车辆-基础设施协作系统的AI驱动的交通安全分析提供了新方法。

### 翻译

行人-车辆事故仍然是城市安全的关键挑战，行人占全球交通死亡人数的20%以上。尽管现有的基于视频的系统可以检测事故何时发生，但它们对这些事件如何在不同行人行为认知阶段展开提供很少见解。最近的视觉-语言模型在视频理解方面显示出强大潜力，但它们通常单独处理视频，没有明确的时间结构或多视图集成。本文引入了多视图阶段感知行人-车辆事故推理(MP-PVIR)，这是一个统一框架，通过四个阶段将多视图视频流系统处理为结构化诊断报告：(1)事件触发的多视图视频采集，(2)行人行为阶段分割，(3)阶段特定的多视图推理，以及(4)分层综合和诊断推理。该框架通过将事故自动分割为认知阶段，在每个阶段执行同步的多视图分析，并将结果合成为具有针对性预防策略的因果链来操作化行为理论。特别是，两个专门的VLM支持MP-PVIR流程：TG-VLM用于行为阶段分割(mIoU = 0.4881)和PhaVR-VLM用于阶段感知多视图分析，在问答上达到33.063的标题分数和高达64.70%的准确率。最后，使用专用的大语言模型生成详细的综合报告，包括场景理解、行为解释、因果推理和预防建议。在Woven Traffic Safety数据集上的评估表明，MP-PVIR有效地将多视图视频数据转化为可操作的见解，推进了车辆-基础设施协作系统的AI驱动的交通安全分析。


### 论文摘要

Pedestrian-vehicle incidents remain a critical urban safety challenge, with pedestrians accounting for over 20% of global traffic fatalities. Although existing video-based systems can detect when incidents occur, they provide little insight into how these events unfold across the distinct cognitive phases of pedestrian behavior. Recent vision-language models (VLMs) have shown strong potential for video understanding, but they remain limited in that they typically process videos in isolation, without explicit temporal structuring or multi-view integration. This paper introduces Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning (MP-PVIR), a unified framework that systematically processes multi-view video streams into structured diagnostic reports through four stages: (1) event-triggered multi-view video acquisition, (2) pedestrian behavior phase segmentation, (3) phase-specific multi-view reasoning, and (4) hierarchical synthesis and diagnostic reasoning. The framework operationalizes behavioral theory by automatically segmenting incidents into cognitive phases, performing synchronized multi-view analysis within each phase, and synthesizing results into causal chains with targeted prevention strategies. Particularly, two specialized VLMs underpin the MP-PVIR pipeline: TG-VLM for behavioral phase segmentation (mIoU = 0.4881) and PhaVR-VLM for phase-aware multi-view analysis, achieving a captioning score of 33.063 and up to 64.70% accuracy on question answering. Finally, a designated large language model is used to generate comprehensive reports detailing scene understanding, behavior interpretation, causal reasoning, and prevention recommendations. Evaluation on the Woven Traffic Safety dataset shows that MP-PVIR effectively translates multi-view video data into actionable insights, advancing AI-driven traffic safety analytics for vehicle-infrastructure cooperative systems.

---

## 74. Studying AC-LGAD strip sensors from laser and testbeam measurements

**论文链接:** [http://arxiv.org/abs/2511.14095v1](http://arxiv.org/abs/2511.14095v1)

**作者:** Danush Shekar, Shirsendu Nanda, Zhenyu Ye, Ryan Heller, Artur Apresyan

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究构建了一套系统来表征和测量AC耦合低增益雪崩二极管(AC-LGADs)的空间和时间分辨率，使用1060nm激光源进行电荷沉积，并与120 GeV质子束的结果进行比较。

### 背景

4D跟踪探测器在即将进行的对撞机实验中将发挥重要作用，需要评估半导体传感器的性能。

### 目的

建立一套测量系统来评估AC-LGADs的空间和时间分辨率，开发一种校准方法，并通过模拟研究理解时间分辨率的各个影响因素。

### 方法

使用1060nm激光源进行初始电荷沉积，采用标定方法；与120 GeV质子束的结果进行比较；使用Silvaco TCAD和Weightfield2进行模拟研究。

### 主要发现

尽管激光和质子束的电荷沉积机制不同，但校准后两种方法测得的空间和时间分辨率具有兼容性。

### 结论

这项工作提供了一种评估半导体传感器性能的方法，可以补充束流测试测量并加速研发工作。

### 翻译

本文介绍了用于表征和测量AC耦合低增益雪崩二极管(AC-LGADs)空间和时间分辨率的装置设置，使用1060nm激光源通过标定方法沉积初始电荷。结果与120 GeV质子束获得的结果进行了比较。尽管激光和质子束在电荷沉积机制上存在差异，但校准后两种源的空间和时间分辨率被发现是兼容的。随着4D跟踪探测器在即将进行的对撞机实验中预计将发挥重要作用，我们预见这项工作可以作为评估半导体传感器性能的一种方式，可以补充束流测试测量并加速研发工作。此外，使用Silvaco TCAD和Weightfield2进行的模拟研究旨在理解AC-LGAD传感器中影响总时间分辨率的各个因素。


### 论文摘要

This paper presents the setup assembled to characterize and measure the spatial and timing resolutions of AC-coupled Low Gain Avalanche Diodes (AC-LGADs), using a 1060 nm laser source to deposit initial charges with a defined calibration methodology. The results were compared to those obtained with a 120 GeV proton beam. Despite the differences in the charge deposition mechanism between the laser and proton beam, the spatial and temporal resolutions were found to be compatible between the two sources after calibration. With 4D tracking detectors expected to play a vital role in upcoming collider experiments, we foresee this work as a way to evaluate the performance of semiconductor sensors that can augment testbeam measurements and accelerate R$\&$D efforts. Additionally, simulation studies using Silvaco TCAD and Weightfield2 were carried out to understand the various contributing factors to the total time resolution in AC-LGAD sensors, measured using the laser source.

---

## 75. Error-Driven Scene Editing for 3D Grounding in Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.14086v1](http://arxiv.org/abs/2511.14086v1)

**作者:** Yue Zhang, Zun Wang, Han Lin, Jialu Li, Jianing Yang, Yonatan Bitton, Idan Szpektor, Mohit Bansal

**发布时间:** 2025-11-18

**备注:** Code: https://github.com/zhangyuejoslin/Deer-3D

### GPT解析

### 总结

本文提出了一种名为DEER-3D的错误驱动框架，通过3D场景编辑生成精确的视觉反事实来解决3D大语言模型在语言与3D环境视觉和空间元素关联方面的局限性，显著提高了模型的 grounding 准确性。

### 背景

尽管3D大语言模型(3D-LLMs)近期有所进展，但它们在将语言准确关联到3D环境中的视觉和空间元素方面仍然有限。这种局限性部分源于训练数据侧重于语言推理而非空间理解，由于3D资源稀缺，导致固有的基础偏差未得到解决。

### 目的

解决3D-LLMs在语言与3D环境视觉和空间元素准确关联方面的局限性，通过生成精确的视觉反事实来减轻这些偏差，无需昂贵的场景重建或大规模3D数据收集。

### 方法

提出3D场景编辑作为关键机制，并引入DEER-3D框架，这是一个遵循'分解、诊断评估、编辑和再训练'工作流程的错误驱动框架。当识别到3D-LLM的基础失败时，框架首先诊断精确的谓词级错误，然后执行最小化的、与谓词对齐的3D场景编辑（如重新着色或重新定位），产生针对性的反事实监督用于迭代模型微调。

### 主要发现

在多个3D基础和场景理解任务的基准测试中评估了编辑流程，通过迭代改进，在所有评估的数据集上 consistently 证明了改进，显著提高了模型的 grounding 准确性。

### 结论

DEER-3D强调了有针对性的、错误驱动的场景编辑在弥合3D LLMs中的语言推理能力与空间基础方面的有效性。

### 翻译

尽管3D大语言模型近期有所进展，但它们在将语言准确关联到3D环境中的视觉和空间元素方面仍然有限。这种局限性部分源于训练数据侧重于语言推理而非空间理解，由于3D资源稀缺，导致固有的基础偏差未得到解决。为解决这一问题，我们提出3D场景编辑作为关键机制，通过细粒度空间操作生成精确的视觉反事实来减轻这些偏差，无需昂贵的场景重建或大规模3D数据收集。此外，为了使这些编辑具有针对性并直接解决模型的特定弱点，我们引入了DEER-3D，这是一个遵循'分解、诊断评估、编辑和再训练'工作流程的错误驱动框架，而非像传统方法那样广泛或随机地增强数据。具体来说，在识别到3D-LLM的基础失败后，我们的框架首先诊断精确的谓词级错误（如属性或空间关系）。然后执行最小化的、与谓词对齐的3D场景编辑，如重新着色或重新定位，为迭代模型微调产生针对性的反事实监督，显著提高了基础准确性。我们在多个3D基础和场景理解任务的基准测试中评估了我们的编辑流程，通过迭代改进，在所有评估的数据集上 consistently 证明了改进。DEER-3D强调了有针对性的、错误驱动的场景编辑在弥合3D LLMs中的语言推理能力与空间基础方面的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D大语言模型(3D-LLMs)无法准确将语言对应到3D环境中的视觉和空间元素的问题。这个问题很重要，因为准确的3D语言定位能力对具身AI和机器人操作至关重要，而当前模型在细粒度视觉细节定位、空间关系解释方面存在困难，倾向于依赖语言先验而非真实几何证据，导致定位错误。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了3D-LLMs的训练限制，发现主要问题在于数据集偏差，而现有方法主要关注文本增强无法解决视觉偏差问题。因此提出通过3D场景编辑生成精确的视觉反事实来减轻偏差。借鉴了反事实数据增强、错误驱动学习和3D场景编辑等现有工作，但将其应用到3D领域，并设计了针对性的编辑策略而非随机增强。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过错误驱动的3D场景编辑生成精确的视觉反事实样本，针对性地编辑3D场景而非随机增强，并通过迭代重新训练逐步提高模型定位准确性。整体流程遵循DEER-3D框架：1)分解：将语言查询分解为原子谓词；2)诊断评估：精确定位谓词级别的错误；3)编辑：执行复制-替换操作，修改目标属性；4)重新训练：集成反事实示例，迭代改进模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出3D场景编辑作为关键机制；引入DEER-3D错误驱动框架；针对性地编辑3D场景；通过谓词级别3D场景编辑生成针对性反事实监督；迭代重新训练机制。相比之前工作，DEER-3D从视觉侧而非文本侧解决限制；将反事实增强扩展到3D领域；通过3D视觉域循环执行实例级校正；需要大规模、可重现和谓词隔离的干预而非用户驱动编辑。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了DEER-3D框架，通过针对性的视觉反事实编辑显著提高了3D大语言模型在语言到3D场景定位任务中的性能，解决了现有方法无法解决的视觉偏差问题。'}


### 论文摘要

Despite recent progress in 3D-LLMs, they remain limited in accurately grounding language to visual and spatial elements in 3D environments. This limitation stems in part from training data that focuses on language reasoning rather than spatial understanding due to scarce 3D resources, leaving inherent grounding biases unresolved. To address this, we propose 3D scene editing as a key mechanism to generate precise visual counterfactuals that mitigate these biases through fine-grained spatial manipulation, without requiring costly scene reconstruction or large-scale 3D data collection. Furthermore, to make these edits targeted and directly address the specific weaknesses of the model, we introduce DEER-3D, an error-driven framework following a structured "Decompose, Diagnostic Evaluation, Edit, and Re-train" workflow, rather than broadly or randomly augmenting data as in conventional approaches. Specifically, upon identifying a grounding failure of the 3D-LLM, our framework first diagnoses the exact predicate-level error (e.g., attribute or spatial relation). It then executes minimal, predicate-aligned 3D scene edits, such as recoloring or repositioning, to produce targeted counterfactual supervision for iterative model fine-tuning, significantly enhancing grounding accuracy. We evaluate our editing pipeline across multiple benchmarks for 3D grounding and scene understanding tasks, consistently demonstrating improvements across all evaluated datasets through iterative refinement. DEER-3D underscores the effectiveness of targeted, error-driven scene editing in bridging linguistic reasoning capabilities with spatial grounding in 3D LLMs.

---

## 76. CORE: Compact Object-centric REpresentations as a New Paradigm for Token Merging in LVLMs

**论文链接:** [http://arxiv.org/abs/2511.14072v1](http://arxiv.org/abs/2511.14072v1)

**作者:** Jingyu Lei, Gaoang Wang, Der-Horng Lee

**发布时间:** 2025-11-18

### GPT解析

### 总结

CORE是一种新的视觉标记压缩范式，通过分割解码器生成对象掩码作为语义先验，指导视觉标记合并为紧凑的对象中心表示，并通过质心引导排序机制保留位置信息。

### 背景

大型视觉语言模型(LVLMs)因视觉标记随图像分辨率二次增长而面临高昂计算和内存成本；现有标记压缩方法缺乏高级语义理解，导致次优合并、信息冗余或上下文丢失。

### 目的

解决现有视觉标记压缩方法的局限性，提出一种新的视觉标记压缩范式。

### 方法

引入CORE (Compact Object-centric REpresentations)，利用高效的分割解码器生成对象掩码，作为高级语义先验指导视觉标记合并，并通过质心引导排序机制恢复合并标记的连贯空间顺序。

### 主要发现

CORE在六个固定速率压缩的权威基准测试上建立新SOTA；在自适应速率设置中实现显著效率提升；极端压缩下(仅保留2.2%视觉标记)仍保持97.4%基线性能。

### 结论

以对象为中心的表示在高效和有效的LVLM处理中具有优越性。

### 翻译

大型视觉语言模型(LVLMs)通常由于视觉标记随图像分辨率二次增长而面临高昂的计算和内存成本。现有的标记压缩方法虽然多样，但通常缺乏高级语义理解，导致次优合并、信息冗余或上下文丢失。为解决这些限制，我们引入CORE(紧凑对象中心表示)，一种新的视觉标记压缩范式。CORE利用高效的分割解码器生成对象掩码，这些掩码作为高级语义先验，指导视觉标记合并为紧凑的对象中心表示集合。此外，一种新颖的质心引导排序机制恢复了合并标记的连贯空间顺序，保留了重要的位置信息。大量实验表明，CORE不仅在六个固定速率压缩的权威基准测试上建立了新的最先进水平，还在自适应速率设置中实现了显著的效率提升。即使在极端压缩下，仅保留所有视觉标记的2.2%，CORE仍能保持97.4%的基线性能。我们的工作证明了对象中心表示在高效和有效的LVLM处理中的优越性。


### 论文摘要

Large Vision-Language Models (LVLMs) usually suffer from prohibitive computational and memory costs due to the quadratic growth of visual tokens with image resolution. Existing token compression methods, while varied, often lack a high-level semantic understanding, leading to suboptimal merges, information redundancy, or context loss. To address these limitations, we introduce CORE (Compact Object-centric REpresentations), a new paradigm for visual token compression. CORE leverages an efficient segmentation decoder to generate object masks, which serve as a high-level semantic prior to guide the merging of visual tokens into a compact set of object-centric representations. Furthermore, a novel centroid-guided sorting mechanism restores a coherent spatial order to the merged tokens, preserving vital positional information. Extensive experiments show that CORE not only establishes a new state-of-the-art on six authoritative benchmarks for fixed-rate compression, but also achieves dramatic efficiency gains in adaptive-rate settings. Even under extreme compression, after aggressively retaining with only 2.2% of all visual tokens, CORE still maintains 97.4% of baseline performance. Our work demonstrates the superiority of object-centric representations for efficient and effective LVLM processing.

---

## 77. RISE: Single Static Radar-based Indoor Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.14019v1](http://arxiv.org/abs/2511.14019v1)

**作者:** Kaichen Zhou, Laura Dodds, Sayed Saad Afzal, Fadel Adib

**发布时间:** 2025-11-18

### GPT解析

### 总结

RISE是一个创新的单静态雷达室内场景理解系统，利用多路径反射来提高空间分辨率，实现隐私保护的室内布局重建和物体检测。

### 背景

室内场景理解仍然是一个基本开放问题。光学传感器如RGB和LiDAR提供高空间保真度，但在室内环境中存在严重遮挡和隐私风险。毫米波雷达保护隐私并能穿透障碍物，但其固有的低空间分辨率使得可靠的几何推理困难。

### 目的

介绍RISE，这是第一个用于单静态雷达室内场景理解的基准和系统，同时针对布局重建和物体检测。

### 方法

RISE基于多路径反射编码丰富几何线索的见解，提出双角度多路径增强来建模到达角和离开角，恢复二次反射并揭示不可见结构。使用模拟到现实的分层扩散框架将碎片化的雷达响应转化为完整的布局重建和物体检测。

### 主要发现

基准包含50,000帧，跨越100个真实室内轨迹，是首个专注于雷达室内场景理解的大规模数据集。实验表明，与最先进的布局重建相比，RISE将Chamfer距离减少了60%，实现首个基于毫米波雷达的物体检测，达到58%的IoU。

### 结论

这些结果确立了RISE作为使用单静态雷达进行几何感知和隐私保护的室内场景理解的新基础。

### 翻译

稳健且保护隐私的室内场景理解仍然是一个基本开放问题。虽然光学传感器如RGB和LiDAR提供高空间保真度，但它们在室内环境中存在严重遮挡并引入隐私风险。相比之下，毫米波雷达保护隐私并能穿透障碍物，但其固有的低空间分辨率使得可靠的几何推理困难。我们介绍了RISE，这是首个用于单静态雷达室内场景理解的基准和系统，同时针对布局重建和物体检测。RISE基于多路径反射（传统上被视为噪声）编码丰富几何线索的关键见解。为利用这一点，我们提出了双角度多路径增强，明确建模到达角和离开角，以恢复二次（幽灵）反射并揭示不可见结构。在这些增强观测的基础上，一个模拟到现实的分层扩散框架将碎片化的雷达响应转化为完整的布局重建和物体检测。我们的基准包含跨越100个真实室内轨迹收集的50,000帧，形成了首个专注于雷达室内场景理解的大规模数据集。大量实验表明，与最先进的布局重建相比，RISE将Chamfer距离减少了60%（降至16厘米），并提供了首个基于毫米波雷达的物体检测，达到58%的IoU。这些结果确立了RISE作为使用单静态雷达进行几何感知和隐私保护的室内场景理解的新基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文旨在解决如何使用单个静态毫米波雷达进行室内场景理解的问题，包括布局重建和物体检测。这个问题在现实中很重要，因为室内场景理解是智能家居、虚拟现实和增强现实应用的基础，而现有的光学传感器(如RGB相机和LiDAR)存在严重遮挡问题且侵犯隐私，无线信号解决方案要么分辨率低，要么需要移动设备增加部署复杂度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过观察毫米波雷达的镜面反射导致有限可见区域的问题，意识到传统被视为噪声的多径反射实际上编码了丰富的几何信息。设计方法借鉴了无线信号室内重建、多径SLAM技术和扩散模型在视觉领域的应用，创新性地利用人体移动引入多径效应，开发双角度多径增强(BAME)模块和模拟到现实的分层扩散(SRHD)框架，将碎片化的雷达信号转化为完整的场景理解。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用人体移动产生的多径效应揭示环境结构，将传统视为噪声的多径反射转化为有价值的几何信息。整体流程包括：1)双角度多径增强(BAME)模块恢复被抑制的鬼影路径；2)多径反转模块估计反射器几何；3)模拟到现实的分层扩散(SRHD)框架将碎片化观测转化为完整布局和物体检测；4)空间一致性优化确保物理有效性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个单静态雷达室内场景理解系统；2)双角度多径增强(BAME)模块显式建模AOA和AOD；3)模拟到现实的分层扩散框架(SRHD)实现完整场景理解；4)大规模数据集支持。相比之前工作，RISE解决了盲点和鬼影可见性问题，不仅可重建墙壁等大表面，还能识别家具等小物体，布局重建准确率提高60%，并首次实现了基于毫米波雷达的物体检测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RISE创新性地利用单静态毫米波雷达通过人体移动产生的多径效应，实现了隐私保护的室内场景理解，首次同时完成了高精度的布局重建和物体检测任务。'}


### 论文摘要

Robust and privacy-preserving indoor scene understanding remains a fundamental open problem. While optical sensors such as RGB and LiDAR offer high spatial fidelity, they suffer from severe occlusions and introduce privacy risks in indoor environments. In contrast, millimeter-wave (mmWave) radar preserves privacy and penetrates obstacles, but its inherently low spatial resolution makes reliable geometric reasoning difficult.   We introduce RISE, the first benchmark and system for single-static-radar indoor scene understanding, jointly targeting layout reconstruction and object detection. RISE is built upon the key insight that multipath reflections, traditionally treated as noise, encode rich geometric cues. To exploit this, we propose a Bi-Angular Multipath Enhancement that explicitly models Angle-of-Arrival and Angle-of-Departure to recover secondary (ghost) reflections and reveal invisible structures. On top of these enhanced observations, a simulation-to-reality Hierarchical Diffusion framework transforms fragmented radar responses into complete layout reconstruction and object detection.   Our benchmark contains 50,000 frames collected across 100 real indoor trajectories, forming the first large-scale dataset dedicated to radar-based indoor scene understanding. Extensive experiments show that RISE reduces the Chamfer Distance by 60% (down to 16 cm) compared to the state of the art in layout reconstruction, and delivers the first mmWave-based object detection, achieving 58% IoU. These results establish RISE as a new foundation for geometry-aware and privacy-preserving indoor scene understanding using a single static radar.

---

## 78. Scene Graph-Guided Generative AI Framework for Synthesizing and Evaluating Industrial Hazard Scenarios

**论文链接:** [http://arxiv.org/abs/2511.13970v1](http://arxiv.org/abs/2511.13970v1)

**作者:** Sanjay Acharjee, Abir Khan Ratul, Diego Patino, Md Nazmus Sakib

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种创新的场景图引导的生成式AI框架，通过分析OSHA事故报告并使用场景图指导图像生成，解决了获取工作场所危险场景真实图像的困难问题。引入的VQA Graph Score证明了生成数据的高质量和实用性。

### 背景

训练视觉模型准确检测工作场所危险需要可能导致事故的不安全条件的真实图像。然而，获取此类数据集很困难，因为几乎不可能在事故发生时捕捉到事故触发场景。

### 目的

克服获取危险场景真实图像的局限性，提出一种新的场景图引导的生成式AI框架，用于合成基于历史职业安全与健康管理局(OSHA)事故报告的逼真危险场景图像。

### 方法

使用GPT-4o分析OSHA叙述，提取结构化的危险推理；将推理转换为对象级场景图，捕捉理解风险所需的空间和上下文关系；这些场景图引导文本到图像扩散模型，生成构图准确的危险场景；引入视觉问答(VQA)框架评估生成数据的真实性和语义保真度。

### 主要发现

VQA Graph Score在四个最先进的生成模型上，基于熵验证的表现优于CLIP和BLIP指标，证实了其更高的判别敏感性。

### 结论

所提出的框架能够合成逼真的危险场景图像，可用于训练视觉模型检测工作场所危险，解决了数据获取困难的问题。

### 翻译

训练视觉模型准确检测工作场所危险需要可能导致事故的不安全条件的真实图像。然而，获取此类数据集很困难，因为几乎不可能在事故发生时捕捉到事故触发场景。为了克服这一限制，本研究提出了一种新颖的场景图引导的生成式AI框架，它基于历史职业安全与健康管理局(OSHA)事故报告合成逼真的危险场景图像。使用GPT-4o分析OSHA叙述，提取结构化的危险推理，并将其转换为对象级场景图，捕捉理解风险所必需的空间和上下文关系。这些图表指导文本到图像扩散模型生成构图准确的危险场景。为了评估生成数据的真实性和语义保真度，引入了视觉问答(VQA)框架。在四个最先进的生成模型上，基于熵验证的VQA Graph Score优于CLIP和BLIP指标，证实了其更高的判别敏感性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是训练视觉模型检测工作场所危险时缺乏真实图像数据的问题。现实中，获取事故发生前的真实图像几乎不可能，因为这些事件是突然且罕见的。这个问题很重要，因为每年美国有约260万职业伤害和疾病报告，每96分钟就有一名工人因工作相关事件死亡。随着自动化和AI的发展，工作场所风险不断变化，传统安全方法难以应对，而缺乏视觉数据限制了能够推理复杂环境中潜在危险的智能安全系统的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到获取真实事故前视觉数据几乎不可能，转而考虑利用OSHA的事故记录来生成逼真的危险场景图像。他们意识到仅合成视觉真实的图像不够，还需确保这些图像准确反映真实的危险情况。设计方法时，他们借鉴了生成式AI（如GANs和扩散模型）、场景图表示法和视觉语言模型等技术。特别是，他们利用场景图作为中间表示，将文本描述的结构化危险信息转换为图像生成可用的视觉表征，同时引入视觉问答框架评估生成数据的保真度。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用场景图作为中间表示，将文本形式的OSHA事故报告转换为结构化的视觉表征，然后利用生成式AI合成逼真的工业危险场景图像，并通过创新的评估方法确保生成图像的准确性。整体流程包括：1) 使用GPT-4o分析OSHA事故报告并分类；2) 对危险理由进行语义嵌入和聚类，识别主要危险原型；3) 使用LLaMA 3将文本描述转换为场景图；4) 将场景图转换为自然语言提示引导图像生成；5) 将场景图转换为可验证断言，使用视觉语言模型评估生成图像的准确性；6) 计算图级合规分数，确保生成图像准确反映危险场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 场景图引导的生成式AI框架，从OSHA事故叙述中生成语义准确的危险场景图像；2) 基于视觉问答的VQA Graph Score评估指标，比CLIP和BLIP等传统指标更具判别力；3) 完整、可扩展的数据集和工具集。相比之前的工作，这种方法不仅生成图像，还确保图像在构图和语义上准确反映危险场景；它使用场景图作为中间表示，而不是直接从文本生成图像；评估方法专门针对危险场景设计，而不是使用通用图像质量指标；它系统地转换历史事故报告为合成数据，而不是仅依赖有限的真实数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于场景图的生成式AI框架，能够从OSHA事故报告中合成逼真的工业危险场景图像，并通过创新的视觉问答评估方法确保生成图像的语义和构图准确性，为训练智能危险检测系统提供了宝贵的合成数据资源。'}


### 论文摘要

Training vision models to detect workplace hazards accurately requires realistic images of unsafe conditions that could lead to accidents. However, acquiring such datasets is difficult because capturing accident-triggering scenarios as they occur is nearly impossible. To overcome this limitation, this study presents a novel scene graph-guided generative AI framework that synthesizes photorealistic images of hazardous scenarios grounded in historical Occupational Safety and Health Administration (OSHA) accident reports. OSHA narratives are analyzed using GPT-4o to extract structured hazard reasoning, which is converted into object-level scene graphs capturing spatial and contextual relationships essential for understanding risk. These graphs guide a text-to-image diffusion model to generate compositionally accurate hazard scenes. To evaluate the realism and semantic fidelity of the generated data, a visual question answering (VQA) framework is introduced. Across four state-of-the-art generative models, the proposed VQA Graph Score outperforms CLIP and BLIP metrics based on entropy-based validation, confirming its higher discriminative sensitivity.

---

## 79. A Brain Wave Encodes a Thousand Tokens: Modeling Inter-Cortical Neural Interactions for Effective EEG-based Emotion Recognition

**论文链接:** [http://arxiv.org/abs/2511.13954v1](http://arxiv.org/abs/2511.13954v1)

**作者:** Nilay Kumar, Priyansh Bhandari, G. Maragatham

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为RBTransformer的新型神经网络架构，用于基于脑电图(EEG)的情绪识别，该模型能够捕捉脑区域间的动态互动，显著提高了情绪识别的准确性。

### 背景

人类情绪难以通过文字表达且常被抽象化，而EEG信号能更直接地反映情绪脑活动。虽然深度学习模型已能处理EEG信号进行高精度情绪识别，但现有方法忽略了不同脑区域间的动态互动，这对理解情绪如何随时间展开和演变至关重要。

### 目的

开发一种基于Transformer的神经网络架构，建模脑的皮层间神经动力学，在潜在空间中更好地捕捉结构化的神经交互，以实现更有效的基于EEG的情绪识别。

### 方法

将EEG信号转换为带微分熵(BDE)标记，通过电极身份嵌入保留空间来源，使用连续的皮层间多头注意力块构建电极×电极注意力矩阵，使模型能够学习皮层间神经依赖关系，最后将特征传递给分类头获得预测结果。

### 主要发现

在SEED、DEAP和DREAMER三个数据集上进行的实验表明，RBTransformer在受试者依赖设置下，在效价、唤醒度和支配度三个维度上，无论二元还是多分类设置，均优于所有先前最先进的方法。

### 结论

通过建模脑区域间的动态互动，RBTransformer能够更准确地捕捉情绪的神经基础，从而显著提高基于EEG的情绪识别性能。

### 翻译

人类情绪难以通过文字表达，且在表达过程中常被抽象化；然而，脑电图(EEG)信号可以提供更直接的视角来观察情绪的脑活动。最近的研究表明，深度学习模型可以处理这些信号以高精度执行情绪识别。然而，许多现有方法忽略了不同脑区域之间的动态互动，这对于理解情绪如何随时间展开和演变至关重要，可能有助于更准确的情绪识别。为此，我们提出了RBTransformer，一种基于Transformer的神经网络架构，在潜在空间中建模大脑的皮层间神经动力学，以更好地捕捉结构化的神经交互，实现有效的基于EEG的情绪识别。首先，EEG信号被转换为带微分熵(BDE)标记，然后通过电极身份嵌入以保留空间来源。这些标记通过连续的皮层间多头注意力块进行处理，构建电极×电极注意力矩阵，使模型能够学习皮层间神经依赖关系。然后将生成的特征传递给分类头以获得最终预测。我们在SEED、DEAP和DREAMER数据集上进行了广泛实验，特别是在受试者依赖设置下，在效价、唤醒度和支配度(对DEAP和DREAMER)三个维度上，在二元和多分类设置下进行。结果表明，所提出的RBTransformer在所有三个数据集上，在所有三个维度和两种分类设置下，均优于所有先前最先进的方法。源代码可在以下网址获取：https://github.com/nnilayy/RBTransformer。


### 论文摘要

Human emotions are difficult to convey through words and are often abstracted in the process; however, electroencephalogram (EEG) signals can offer a more direct lens into emotional brain activity. Recent studies show that deep learning models can process these signals to perform emotion recognition with high accuracy. However, many existing approaches overlook the dynamic interplay between distinct brain regions, which can be crucial to understanding how emotions unfold and evolve over time, potentially aiding in more accurate emotion recognition. To address this, we propose RBTransformer, a Transformer-based neural network architecture that models inter-cortical neural dynamics of the brain in latent space to better capture structured neural interactions for effective EEG-based emotion recognition. First, the EEG signals are converted into Band Differential Entropy (BDE) tokens, which are then passed through Electrode Identity embeddings to retain spatial provenance. These tokens are processed through successive inter-cortical multi-head attention blocks that construct an electrode x electrode attention matrix, allowing the model to learn the inter-cortical neural dependencies. The resulting features are then passed through a classification head to obtain the final prediction. We conducted extensive experiments, specifically under subject-dependent settings, on the SEED, DEAP, and DREAMER datasets, over all three dimensions, Valence, Arousal, and Dominance (for DEAP and DREAMER), under both binary and multi-class classification settings. The results demonstrate that the proposed RBTransformer outperforms all previous state-of-the-art methods across all three datasets, over all three dimensions under both classification settings. The source code is available at: https://github.com/nnilayy/RBTransformer.

---

## 80. What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations

**论文链接:** [http://arxiv.org/abs/2511.13900v1](http://arxiv.org/abs/2511.13900v1)

**作者:** Mihir Gupte, Eshan Dixit, Muhammad Tayyab, Arun Adiththan

**发布时间:** 2025-11-17

**备注:** To be submitted for publication

### GPT解析

### 总结

本研究针对大型语言模型在长距离上下文利用方面的'lost-in-the-middle'现象，提出了GM-Extract基准数据集和评估系统，研究了不同数据表示方式对模型检索性能的影响，并评估了多种缓解技术的效果。

### 背景

大型语言模型(LLMs)在有效利用长距离上下文方面能力下降，即'lost-in-the-middle'现象，这对基于检索的LLM应用构成了重大挑战。

### 目的

研究这一现象在实际应用环境中的影响，设计评估LLM在检索控制变量性能的基准数据集，并分析不同缓解技术的效果。

### 方法

引入GM-Extract基准数据集；提出使用空间检索能力指标(文档指标)和语义检索能力指标(变量提取指标)的评估系统；对7-8B参数模型在键值提取和问答任务上进行评估；将缓解方法分为黑盒和白盒两类并测试其效果。

### 主要发现

仅通过改变上下文窗口中的数据表示方式，检索性能就会发生显著变化；模型间存在清晰性能模式，与困惑度分数相关；缓解技术的功效非常微妙，在某些情况下提高性能，在另一些情况下反而导致负面影响。

### 结论

评估揭示了不同缓解策略成功和失败的场景，为理解这些策略在实际环境中的效用提供了全面认识。

### 翻译

大型语言模型有效利用长距离上下文的能力逐渐减弱——即'lost-in-the-middle'现象——对基于检索的LLM应用构成了重大挑战。为在实际应用环境中研究这一现象的影响，我们引入了GM-Extract，这是一个精心设计的新型基准数据集，用于评估LLM在检索控制变量方面的性能。为准确诊断失败模式，我们提出了一个简单而优雅的评估系统，使用两个不同的指标：一个用于空间检索能力(文档指标)，另一个用于语义检索能力(变量提取指标)。我们在两个多文档任务(键值提取和问答)上对7-8B参数模型进行了系统评估，证明仅通过改变上下文窗口中数据的表示方式，检索性能就会发生显著变化。虽然未一致观察到明显的U型曲线，但我们的分析揭示了模型间的清晰性能模式，我们进一步将其与困惑度分数相关联。此外，我们对缓解方法进行了文献综述，将其分为两种不同的方法：黑盒方法和白盒方法。然后我们将这些技术应用到我们的基准测试中，发现它们的功效非常微妙。我们的评估突出了这些策略成功提高性能的场景，以及它们导致负面影响的意外情况，为理解它们在实际环境中的效用提供了全面认识。


### 论文摘要

The diminishing ability of large language models (LLMs) to effectively utilize long-range context-the "lost-in-the-middle" phenomenon-poses a significant challenge in retrieval-based LLM applications. To study the impact of this phenomenon in a real-world application setting, we introduce GM-Extract, a novel benchmark dataset meticulously designed to evaluate LLM performance on retrieval of control variables. To accurately diagnose failure modes, we propose a simple yet elegant evaluation system using two distinct metrics: one for spatial retrieval capability (Document Metric) and the other for semantic retrieval capability (Variable Extraction Metric). We conduct a systematic evaluation of 7-8B parameter models on two multi-document tasks (key-value extraction and question-answering), demonstrating a significant change in retrieval performance simply by altering how the data is represented in the context window. While a distinct U-shaped curve was not consistently observed, our analysis reveals a clear pattern of performance across models, which we further correlate with perplexity scores. Furthermore, we perform a literature survey of mitigation methods, which we categorize into two distinct approaches: black-box and white-box methods. We then apply these techniques to our benchmark, finding that their efficacy is highly nuanced. Our evaluation highlights scenarios where these strategies successfully improve performance, as well as surprising cases where they lead to a negative impact, providing a comprehensive understanding of their utility in a practical context.

---

## 81. VLMs Guided Interpretable Decision Making for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.13881v1](http://arxiv.org/abs/2511.13881v1)

**作者:** Xin Hu, Taotao Jing, Renran Tian, Zhengming Ding

**发布时间:** 2025-11-17

**备注:** Accepted by WACV 2026

### GPT解析

### 总结

本研究提出了一种新方法，将视觉语言模型从直接决策生成器转变为语义增强器，通过多模态交互架构和后处理细化模块，提高了自动驾驶决策的可靠性和可解释性。

### 背景

最近的自动驾驶研究探索了在视觉问答框架中使用视觉语言模型进行直接驾驶决策，但这些方法通常依赖手工设计的提示，性能不稳定，限制了其在真实场景中的鲁棒性和泛化能力。

### 目的

评估最先进的开源视觉语言模型在高层次决策任务中的表现，并识别它们提供可靠、上下文感知决策能力的局限性。

### 方法

提出新方法，将视觉语言模型的角色转变为语义增强器，利用其场景理解能力为现有视觉基准添加结构化、语言丰富的场景描述，并引入融合视觉和语言特征的多模态交互架构，同时设计后处理细化模块提高预测可靠性。

### 主要发现

在两个自动驾驶基准测试上的广泛实验表明，该方法取得了最先进的性能。

### 结论

为将视觉语言模型集成到可靠且可解释的自动驾驶系统中提供了有前景的方向。

### 翻译

自动驾驶的最新进展探索了在视觉问答框架中使用视觉语言模型进行直接驾驶决策。然而，这些方法通常依赖手工设计的提示，且性能不稳定，限制了它们在真实场景中的鲁棒性和泛化能力。在本工作中，我们使用自视角视觉输入评估了最先进的开源视觉语言模型在高层次决策任务上的表现，并发现它们提供可靠、上下文感知决策的能力存在关键局限性。受这些观察的启发，我们提出了一种新方法，将视觉语言模型的角色从直接决策生成器转变为语义增强器。具体来说，我们利用它们强大的通用场景理解能力，通过结构化、语言丰富的场景描述来增强现有的视觉基准测试。基于这种增强表示，我们引入了一个融合视觉和语言特征的多模态交互架构，用于更准确的决策和可解释的文本解释。此外，我们设计了一个后处理细化模块，利用视觉语言模型提高预测可靠性。在两个自动驾驶基准测试上的广泛实验表明，我们的方法取得了最先进的性能，为将视觉语言模型集成到可靠且可解释的自动驾驶系统中提供了有前景的方向。


### 论文摘要

Recent advancements in autonomous driving (AD) have explored the use of vision-language models (VLMs) within visual question answering (VQA) frameworks for direct driving decision-making. However, these approaches often depend on handcrafted prompts and suffer from inconsistent performance, limiting their robustness and generalization in real-world scenarios. In this work, we evaluate state-of-the-art open-source VLMs on high-level decision-making tasks using ego-view visual inputs and identify critical limitations in their ability to deliver reliable, context-aware decisions. Motivated by these observations, we propose a new approach that shifts the role of VLMs from direct decision generators to semantic enhancers. Specifically, we leverage their strong general scene understanding to enrich existing vision-based benchmarks with structured, linguistically rich scene descriptions. Building on this enriched representation, we introduce a multi-modal interactive architecture that fuses visual and linguistic features for more accurate decision-making and interpretable textual explanations. Furthermore, we design a post-hoc refinement module that utilizes VLMs to enhance prediction reliability. Extensive experiments on two autonomous driving benchmarks demonstrate that our approach achieves state-of-the-art performance, offering a promising direction for integrating VLMs into reliable and interpretable AD systems.

---

## 82. GRLoc: Geometric Representation Regression for Visual Localization

**论文链接:** [http://arxiv.org/abs/2511.13864v1](http://arxiv.org/abs/2511.13864v1)

**作者:** Changyang Li, Xuejian Ma, Lixiang Liu, Zhan Li, Qingan Yan, Yi Xu

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种名为几何表示回归(GRR)的新方法，作为绝对姿态回归(APR)的替代方案，通过显式预测解耦的几何表示来提高视觉定位的泛化能力。

### 背景

绝对姿态回归(APR)已成为视觉定位的有效方法，但传统APR模型作为黑盒直接从图像回归6自由度姿态，倾向于记忆训练视图而非理解3D场景几何。

### 目的

提出一种基于几何的替代方法，将APR重新表述为从图像直接回归底层3D表示的逆向过程，以提高模型的泛化能力和几何理解。

### 方法

受新颖视图合成启发，提出几何表示回归(GRR)范式，模型明确预测两个解耦的几何表示：射线束方向估计相机旋转，对应点图估计相机平移，最后通过可微确定性求解器恢复最终姿态。

### 主要发现

旋转和平移预测的明确解耦显著提升了性能，在7-Scenes和Cambridge Landmarks数据集上实现了最先进的结果。

### 结论

将逆向渲染过程建模是通往可泛化绝对姿态估计的更稳健路径，显式几何表示比直接回归姿态更有效。

### 翻译

绝对姿态回归(APR)已成为视觉定位的一种引人注目的范式。然而，APR模型通常作为黑盒操作，直接从查询图像回归6自由度姿态，这可能导致模型记忆训练视图而非理解3D场景几何。在这项工作中，我们提出了一种基于几何的替代方案。受新颖视图合成的启发，该方案从中间几何表示渲染图像，我们将APR重新表述为其逆向过程，直接从图像回归底层3D表示，并将这种范式命名为几何表示回归(GRR)。我们的模型明确预测世界坐标系中的两个解耦几何表示：(1)估计相机旋转的射线束方向，(2)估计相机平移的对应点图。然后使用可微确定性求解器从这些几何组件恢复最终的6自由度相机姿态。这种将学习到的视觉到几何映射与最终姿态计算分离的解耦方法在网络中引入了强几何先验。我们发现，旋转和平移预测的明确解耦可显著提升性能。我们在7-Scenes和Cambridge Landmarks数据集上展示了最先进的性能，验证了将逆向渲染过程建模是通往可泛化绝对姿态估计的更稳健路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉定位中的绝对姿态回归问题。现有方法直接从图像回归6自由度姿态，但往往倾向于记忆训练视图而非理解3D场景几何，导致泛化能力差。这个问题在现实中很重要，因为视觉定位是增强现实、机器人导航和自动驾驶等应用的基础，而理解场景几何结构对于构建更鲁棒、可解释的定位系统至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从新颖视图合成（NVS）方法如NeRF和3DGS获得启发，这些方法将渲染过程分解为明确的几何步骤。作者提出将这个过程反转：不是从3D表示渲染图像（正向过程），而是从图像回归底层的3D表示（反向过程），称为几何表示回归（GRR）。作者设计了双分支网络，一个分支预测光线方向用于旋转估计，另一个分支预测3D点图用于平移估计。借鉴了NVS的几何分解思想、APR的端到端框架以及领域对抗训练技术来桥接合成和真实数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将绝对姿态估计重新表述为几何表示回归任务，预测两种解耦的几何表示（光线方向场和3D点图），然后通过确定性求解器恢复最终姿态。整体流程：1)输入图像划分为N×N网格；2)双分支网络分别提取特征；3)光线分支预测世界空间光线方向，点分支预测3D点；4)使用Kabsch算法从光线方向估计旋转，从3D点图估计平移；5)训练时使用3DGS生成合成数据，并通过领域对抗训练提高泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出几何表示回归(GRR)新范式；2)预测解耦的几何表示(光线方向和3D点图)；3)设计双分支网络分别处理旋转和平移；4)使用确定性姿态求解器；5)结合领域自适应训练。不同之处：与传统APR不同，GRLoc回归中间几何表示而非直接回归姿态；与SCR不同，预测更高层次的几何表示；与RPR不同，无需检索参考图像；与PPR不同，专注于初始姿态估计；与CRR不同，动机是学习3D几何而非解决隐私问题，且解耦了旋转和平移预测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了几何表示回归范式，通过解耦预测光线方向和3D点图等几何表示，并使用确定性求解器恢复相机姿态，实现了比现有绝对姿态回归方法更准确、可解释且泛化能力更强的视觉定位系统。'}


### 论文摘要

Absolute Pose Regression (APR) has emerged as a compelling paradigm for visual localization. However, APR models typically operate as black boxes, directly regressing a 6-DoF pose from a query image, which can lead to memorizing training views rather than understanding 3D scene geometry. In this work, we propose a geometrically-grounded alternative. Inspired by novel view synthesis, which renders images from intermediate geometric representations, we reformulate APR as its inverse that regresses the underlying 3D representations directly from the image, and we name this paradigm Geometric Representation Regression (GRR). Our model explicitly predicts two disentangled geometric representations in the world coordinate system: (1) a ray bundle's directions to estimate camera rotation, and (2) a corresponding pointmap to estimate camera translation. The final 6-DoF camera pose is then recovered from these geometric components using a differentiable deterministic solver. This disentangled approach, which separates the learned visual-to-geometry mapping from the final pose calculation, introduces a strong geometric prior into the network. We find that the explicit decoupling of rotation and translation predictions measurably boosts performance. We demonstrate state-of-the-art performance on 7-Scenes and Cambridge Landmarks datasets, validating that modeling the inverse rendering process is a more robust path toward generalizable absolute pose estimation.

---

## 83. Robust galaxy image decompositions with Differential Evolution optimisation and the problem of classical bulges in and beyond the nearby Universe

**论文链接:** [http://arxiv.org/abs/2511.13823v1](http://arxiv.org/abs/2511.13823v1)

**作者:** Dimitri A. Gadotti

**发布时间:** 2025-11-17

**备注:** Accepted for publication in MNRAS (31 pages, including 14 figures, 7 tables and 3 appendices); Fig. A1 will appear in full resolution in the journal version (allowing to better visualise central details in the isophotal contours and ellipse fits)

### GPT解析

### 总结

该研究通过二维分解技术分析星系结构，探讨了邻近宇宙内外观测差异的原因，发现分辨率不足可能导致对经典凸起的错误识别。

### 背景

二维分解技术是推导星系恒星结构物理特性的有力工具，但现有拟合算法存在局限。邻近宇宙外的研究显示盘状星系中经典凸起比例高于邻近星系研究，后者表明经典凸起比例较小，可能挑战合并驱动的星系形成理论。

### 目的

理解邻近宇宙内外星系观测差异的根本原因，特别是关于经典凸起比例的不一致。

### 方法

使用TIMER项目的16个邻近星系样本，应用不同分解方法，分析原始图像和人工红移图像，采用差分进化算法进行测量。

### 主要发现

差分进化算法能准确测量结构特性且主观干预少，正确识别出核盘而非经典凸起；当物理空间分辨率不足时，对光度凸起Sérsic指数的系统高估会导致错误判断经典凸起的存在。

### 结论

分辨率不足导致的系统高估可能是观测差异的根本原因，这一问题即使在Euclid、HST和JWST等先进设施的数据中也可能存在。

### 翻译

通过对星系进行二维分解已被证明是推导星系中恒星结构物理特性的有力技术。然而，大多数研究使用的拟合算法容易陷入局部最小值，或涉及主观选择。此外，当应用于邻近宇宙以外的样本时，关于盘状星系中经典凸起比例的结果与邻近星系的研究不一致。后者的研究表明经典凸起比例较小，可能挑战了我们以合并驱动的星系形成观点。因此，理解邻近宇宙内外观测之间的差异至关重要。在本文中，我使用TIMER项目的16个邻近星系样本，这些星系先前已被证明不包含经典凸起，并应用不同的分解方法，使用原始图像和人工红移图像进行分解。我表明差分进化算法能够提供准确的物理特性测量，几乎不需要主观干预，正确指示出核盘的存在（而非经典凸起）。然而，我也表明当物理空间分辨率不充分时，对光度凸起Sérsic指数的系统高估会导致错误得出存在经典凸起的结论。我讨论了这如何可能是上述差异的根本原因，并指出即使使用来自Euclid、HST和JWST等设施的数据，这个问题也可能存在。


### 论文摘要

Deconstructing galaxies through two-dimensional decompositions has been shown to be a powerful technique to derive the physical properties of stellar structures in galaxies. However, most studies employ fitting algorithms that are prone to be trapped in local minima, or involve subjective choices. Furthermore, when applied on samples beyond the nearby Universe, results on the fraction of classical bulges in disc galaxies do not agree with studies on nearby galaxies. The latter studies point to a small fraction of classical bulges, possibly challenging our merger-driven picture of galaxy formation. Therefore, understanding the discrepancy between observations in and beyond the nearby Universe is of paramount importance. In this paper, I use a sample of 16 nearby galaxies drawn from the TIMER project, which previously have been shown to not host classical bulges, and perform decompositions applying different methodologies and employing the original images as well as artificially redshifted images. I show that the Differential Evolution algorithm is able to provide accurate measurements of structural properties with little subjective intervention, correctly indicating the presence of nuclear discs (not classical bulges). However, I also show that when the physical spatial resolution is not adequate, a systematic overestimation of the photometric bulge Sérsic index leads to the false conclusion of the presence of classical bulges. I discuss how this may be the root cause of the discrepancy mentioned above, and point out how this issue may be a problem even with data from facilities such as Euclid, HST and JWST.

---

## 84. Decoupling Scene Perception and Ego Status: A Multi-Context Fusion Approach for Enhanced Generalization in End-to-End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.13079v2](http://arxiv.org/abs/2511.13079v2)

**作者:** Jiacheng Tang, Mingyue Feng, Jiachao Liu, Yaonong Wang, Jian Pu

**发布时间:** 2025-11-17

**备注:** Accepted to AAAI 2026 (Oral)

### GPT解析

### 总结

该研究提出AdaptiveAD架构，通过双分支结构分离场景感知和自身状态处理，解决了自动驾驶系统过度依赖自身状态的问题，提升了系统的泛化能力和鲁棒性。

### 背景

面向规划的自动驾驶模块化设计虽已显著推进端到端系统，但现有架构过度依赖自身状态，限制了泛化和鲁棒的场景理解能力。

### 目的

解决现有自动驾驶架构过度依赖自身状态的问题，提高系统的泛化能力和鲁棒场景理解能力。

### 方法

提出AdaptiveAD架构，采用双分支结构：一个分支基于多任务学习进行场景驱动推理（故意从BEV编码器省略自身状态），另一个分支仅基于规划任务进行自身状态驱动推理；通过场景感知融合模块整合两个分支的决策；引入路径注意力机制和两个辅助任务（BEV单向蒸馏和自回归在线映射）确保解耦不损害多任务学习。

### 主要发现

现有架构的根本问题是设计上允许自身状态被轻易用作捷径，特别是在BEV编码器中过早融合自身状态导致强先验信息主导下游规划模块；AdaptiveAD显著减轻了对自身状态的过度依赖，在多种场景下展现出令人印象深刻的泛化能力。

### 结论

AdaptiveAD实现了最先进的开环规划性能，显著减轻了对自身状态的过度依赖，并在多种场景下展现出令人印象深刻的泛化能力，为自动驾驶系统设计提供了新的架构级解决方案。

### 翻译

面向规划的自动驾驶模块化设计已显著推进端到端系统。然而，现有架构仍受限于对自身状态的过度依赖，阻碍了泛化和鲁棒的场景理解。我们确定根本原因在于这些架构中的固有设计允许自身状态被轻易用作捷径。具体来说，在BEV编码器上游过早融合自身状态，允许这种强先验信息流主导下游规划模块。为应对这一挑战，我们提出AdaptiveAD，一种基于多上下文融合策略的架构级解决方案。其核心是一个双分支结构，明确分离场景感知和自身状态。一个分支基于多任务学习执行场景驱动推理，但故意从BEV编码器中省略自身状态，而另一个分支仅基于规划任务执行自身状态驱动推理。然后，场景感知融合模块自适应地整合两个分支的互补决策，形成最终规划轨迹。为确保这种解耦不损害多任务学习，我们引入了ego-BEV交互的路径注意力机制，并添加了两个针对性辅助任务：BEV单向蒸馏和自回归在线映射。在nuScenes数据集上的广泛评估表明，AdaptiveAD实现了最先进的开环规划性能。重要的是，它显著减轻了对自身状态的过度依赖，并在多种场景下展现出令人印象深刻的泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决端到端自动驾驶系统中的'自我状态过度依赖'问题，即模型倾向于'按惯性驾驶'而非'按视觉驾驶'。这个问题在现实中至关重要，因为它关系到自动驾驶系统的安全性和可靠性。当系统过度依赖车辆自身运动状态时，在紧急情况或复杂场景中可能导致致命的轨迹规划；在数据集偏差情况下难以处理转弯等复杂场景；当车辆状态信息不准确时系统性能会急剧下降。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析发现现有架构中自我状态与场景感知的过早融合是问题的根源。他们设计了一个双分支架构：一个分支专注于场景驱动的推理（排除自我状态影响），另一个分支专注于自我状态驱动的推理。最后通过场景感知融合模块自适应整合决策。该方法借鉴了VAD框架、因果一致性研究、知识蒸馏技术和多任务学习框架，但创新在于从架构层面而非数据或表示层面解决问题，通过物理分离信息流来抑制捷径学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多上下文融合策略显式分离场景感知和自我状态，使模型能同时考虑环境约束和车辆运动惯性。整体流程包括：1)多上下文决策生成：场景驱动分支排除自我状态影响，自我状态驱动分支保留车辆状态信息，两者都使用路径注意力机制；2)多上下文决策融合：通过场景感知初始化、上下文对齐和自适应融合整合两个分支的决策；3)辅助任务正则化：BEV单向蒸馏提升感知质量，自回归在线映射确保地图感知与规划行动的一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)架构级别的双分支解决方案，显式分离场景感知和自我状态；2)路径注意力机制，沿假设未来路径引导采样；3)BEV单向蒸馏，使用自我状态驱动的BEV增强场景驱动分支特征；4)自回归在线映射，建立规划到感知的反馈循环。相比先前工作，AdaptiveAD从架构而非数据或表示层面解决问题；通过物理分离信息流而非简单处理输入；在各种场景和噪声条件下展现出更强的鲁棒性和泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AdaptiveAD通过双分支架构和多上下文融合策略显式分离场景感知与自我状态，显著提升了端到端自动驾驶系统在复杂场景中的泛化能力和鲁棒性，同时抑制了模型对车辆自身运动状态的过度依赖。'}


### 论文摘要

Modular design of planning-oriented autonomous driving has markedly advanced end-to-end systems. However, existing architectures remain constrained by an over-reliance on ego status, hindering generalization and robust scene understanding. We identify the root cause as an inherent design within these architectures that allows ego status to be easily leveraged as a shortcut. Specifically, the premature fusion of ego status in the upstream BEV encoder allows an information flow from this strong prior to dominate the downstream planning module. To address this challenge, we propose AdaptiveAD, an architectural-level solution based on a multi-context fusion strategy. Its core is a dual-branch structure that explicitly decouples scene perception and ego status. One branch performs scene-driven reasoning based on multi-task learning, but with ego status deliberately omitted from the BEV encoder, while the other conducts ego-driven reasoning based solely on the planning task. A scene-aware fusion module then adaptively integrates the complementary decisions from the two branches to form the final planning trajectory. To ensure this decoupling does not compromise multi-task learning, we introduce a path attention mechanism for ego-BEV interaction and add two targeted auxiliary tasks: BEV unidirectional distillation and autoregressive online mapping. Extensive evaluations on the nuScenes dataset demonstrate that AdaptiveAD achieves state-of-the-art open-loop planning performance. Crucially, it significantly mitigates the over-reliance on ego status and exhibits impressive generalization capabilities across diverse scenarios.

---

## 85. GAEA: Experiences and Lessons Learned from a Country-Scale Environmental Digital Twin

**论文链接:** [http://arxiv.org/abs/2511.13807v1](http://arxiv.org/abs/2511.13807v1)

**作者:** Andreas Kamilaris, Chirag Padubidri, Asfa Jamil, Arslan Amin, Indrajit Kalita, Jyoti Harti, Savvas Karatsiolis, Aytac Guley

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文描述了在塞浦路斯岛部署国家级环境数字孪生系统三年后的经验和教训。

### 背景

在塞浦路斯岛上部署了一个名为GAEA的环境数字孪生系统，该系统已运行三年。

### 目的

展示地理空间分析和环境数字孪生在大规模应用中的能力、潜力、当前和未来挑战。

### 方法

部署并运行了一个包含27个环境地理空间服务的数字孪生系统GAEA。

### 主要发现

GAEA系统适用于城市规划者、政策制定者、农民、房主、房地产和林业专业人士，以及拥有房产组合的保险公司和银行等多种用户群体。

### 结论

地理空间分析和环境数字孪生在大规模应用中具有显著的力量和潜力，同时也面临当前和未来的挑战。

### 翻译

本文描述了在塞浦路斯岛上部署国家级环境数字孪生三年后的经验和教训。这个名为GAEA的数字孪生包含27个环境地理空间服务，适合城市规划者、政策制定者、农民、房主、房地产和林业专业人士，以及拥有房产组合的保险公司和银行使用。本文证明了地理空间分析和环境数字孪生在大规模应用中的力量、潜力、当前和未来挑战。


### 论文摘要

This paper describes the experiences and lessons learned after the deployment of a country-scale environmental digital twin on the island of Cyprus for three years. This digital twin, called GAEA, contains 27 environmental geospatial services and is suitable for urban planners, policymakers, farmers, property owners, real-estate and forestry professionals, as well as insurance companies and banks that have properties in their portfolio. This paper demonstrates the power, potential, current and future challenges of geospatial analytics and environmental digital twins on a large scale.

---

## 86. Zero-shot Synthetic Video Realism Enhancement via Structure-aware Denoising

**论文链接:** [http://arxiv.org/abs/2511.14719v1](http://arxiv.org/abs/2511.14719v1)

**作者:** Yifan Wang, Liya Ji, Zhanghan Ke, Harry Yang, Ser-Nam Lim, Qifeng Chen

**发布时间:** 2025-11-18

**备注:** Project Page: https://wyf0824.github.io/Video_Realism_Enhancement/

### GPT解析

### 总结

提出一种增强合成视频真实感的方法，能够以照片级真实感重新渲染来自模拟器的合成视频，保持多级结构在空间和时间域中的一致性。

### 背景

合成视频在真实感方面存在提升空间，需要一种方法能够在保持结构和语义一致性的同时增强视觉效果。

### 目的

开发一种零样本框架，能够在不进一步微调的情况下，基于扩散视频基础模型增强合成视频的真实感，同时保持原始视频的多级结构。

### 方法

通过辅助模型估计合成视频的结构感知信息（深度图、语义图、边缘图），将生成/去噪过程条件化于这些信息，确保增强视频在结构和语义层面与原始视频一致。

### 主要发现

实验表明该方法在结构一致性方面优于现有基线方法，同时保持了最先进的照片级真实感质量。

### 结论

该方法是一种简单、通用且强大的增强合成视频真实感的方法，能够在不损失原始视频结构和语义信息的情况下显著提升视觉真实感。

### 翻译

我们提出了一种增强合成视频真实感的方法，能够以照片级真实感的方式重新渲染来自模拟器的合成视频。我们的真实感增强方法是一种零样本框架，专注于在空间和时间域中将合成视频的多级结构保留到增强后的视频中，基于扩散视频基础模型构建且无需进一步微调。具体而言，我们通过辅助模型对合成视频的估计结构感知信息（如深度图、语义图和边缘图）进行有效修改，使生成/去噪过程条件化于这些信息，而不是从模拟器提取信息。这种指导确保增强的视频在结构和语义层面与原始合成视频保持一致。我们的方法是一种简单而通用且强大的增强合成视频真实感的方法：实验表明，我们的方法在保持原始视频结构一致性方面优于现有基线，同时维持了最先进的照片级真实感质量。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决如何提升合成视频（如自动驾驶模拟器生成的视频）的真实感问题。这个问题在自动驾驶领域尤为重要，因为自动驾驶需要大量数据进行训练，特别是罕见的长尾场景，但真实世界数据的收集和标注成本高昂。合成数据虽然可扩展可控，但与真实世界存在明显的'域差距'，影响模型性能。特别是对于安全关键的小物体（如交通信号灯、路标），现有方法往往无法准确保持其语义细节。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有GAN-based方法存在分辨率低、时间一致性差和语义不一致的问题，以及基于扩散模型的方法忽视了源视频视觉内容的问题。他们借鉴了Cosmos-Transfer世界模型作为基础框架，结合了DDIM Inversion技术通过确定性方式将源视频映射到初始潜在表示，并应用Classifier-Free Guidance来选择性地修改视觉风格。方法设计充分利用了多模态条件（深度图、语义图、边缘图）作为空间控制输入，确保在提升真实感的同时保持原始结构和语义信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'结构感知去噪'，利用合成视频的多级结构信息作为指导，在不破坏原始结构和语义内容的前提下提升视频真实感，同时采用零样本框架避免领域特定微调。整体流程分为三步：1) 源模拟和条件设置：生成视频描述，提取反转提示和正提示，以及空间条件图；2) 确定性潜在反转：通过DDIM反转将合成视频编码为结构感知的潜在表示；3) 结构感知去噪：从反转的潜在表示开始，使用正提示进行去噪，同时保持空间条件不变以确保结构一致性。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个零样本结构感知去噪框架，无需领域特定后训练；2) 简单有效的反转-生成范式，结合DDIM反转和多条件ControlNet；3) 针对小物体的评估协议，确保对语义保真度的严格评估。相比之前工作，该方法更注重保持源视频的视觉内容（颜色、光照），能更准确重现安全关键的语义细节；避免了其他视频生成方法的局限性（如内容生成与时间动力学解耦）；在保持小物体（如交通灯、路标）的颜色和结构方面表现更佳，且在不同环境条件下保持更好的时间一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一种基于结构感知去噪的零样本框架，能够在保持合成视频结构和语义一致性的同时，显著提升其照片级真实感，特别适用于自动驾驶等需要高保真合成数据的领域。'}


### 论文摘要

We propose an approach to enhancing synthetic video realism, which can re-render synthetic videos from a simulator in photorealistic fashion. Our realism enhancement approach is a zero-shot framework that focuses on preserving the multi-level structures from synthetic videos into the enhanced one in both spatial and temporal domains, built upon a diffusion video foundational model without further fine-tuning. Specifically, we incorporate an effective modification to have the generation/denoising process conditioned on estimated structure-aware information from the synthetic video, such as depth maps, semantic maps, and edge maps, by an auxiliary model, rather than extracting the information from a simulator. This guidance ensures that the enhanced videos are consistent with the original synthetic video at both the structural and semantic levels. Our approach is a simple yet general and powerful approach to enhancing synthetic video realism: we show that our approach outperforms existing baselines in structural consistency with the original video while maintaining state-of-the-art photorealism quality in our experiments.

---

## 87. \textit{FLARE}: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning

**论文链接:** [http://arxiv.org/abs/2511.14715v1](http://arxiv.org/abs/2511.14715v1)

**作者:** Abolfazl Younesi, Leon Kiss, Zahra Najafabadi Samani, Juan Aznar Poveda, Thomas Fahringer

**发布时间:** 2025-11-18

**备注:** Under Review

### GPT解析

### 总结

FLARE是一种自适应的基于声誉的联邦学习防御框架，将客户端可靠性评估从二元决策转变为连续的多维度信任评估，能够有效抵御拜占庭攻击、数据投毒和自适应对抗行为。

### 背景

联邦学习(FL)能够在保护数据隐私的同时实现协作模型训练，但容易受到恶意客户端的攻击，如拜占庭攻击、数据投毒或自适应对抗行为，导致模型完整性受损。

### 目的

解决现有防御机制依赖静态阈值和二元分类，无法适应现实部署中不断变化的客户端行为的问题，提出一种能够适应动态威胁环境的防御框架。

### 方法

FLARE框架包含四个核心组件：多维度声誉评分系统、自校准自适应阈值机制、基于声誉的加权聚合与软排除、以及局部差分隐私(LDP)机制。同时引入了一种高度隐蔽的统计模仿(SM)攻击作为基准对手。

### 主要发现

在MNIST、CIFAR-10和SVHN数据集上的实验表明，FLARE在各种攻击类型下保持高模型准确度，比现有拜占庭鲁棒方法收敛更快，将鲁棒性提高高达16%，并将模型收敛保持在非攻击基线的30%以内，同时以最小计算开销实现强大的恶意客户端检测。

### 结论

FLARE通过连续、多维度的信任评估和自适应防御机制，有效提升了联邦学习系统在面对复杂攻击时的鲁棒性和收敛性能，为实际部署中的安全联邦学习提供了可行解决方案。

### 翻译

联邦学习(FL)能够在保护数据隐私的同时实现协作模型训练。然而，它仍然容易受到恶意客户端的攻击，这些客户端通过拜占庭攻击、数据投毒或自适应对抗行为损害模型完整性。现有防御机制依赖于静态阈值和二元分类，无法适应现实部署中不断变化的客户端行为。我们提出了FLARE，一种自适应的基于声誉的框架，将客户端可靠性评估从二元决策转变为连续的、多维度的信任评估。FLARE集成了：多维度声誉评分、自校准自适应阈值机制、基于声誉的加权聚合与软排除、以及局部差分隐私(LDP)机制。我们还引入了一种高度隐蔽的统计模仿(SM)攻击作为基准对手。在MNIST、CIFAR-10和SVHN上对100个客户端的广泛实验表明，FLARE在各种攻击类型下保持高模型准确度，并且比现有的拜占庭鲁棒方法收敛更快。FLARE将鲁棒性提高高达16%，并将模型收敛保持在非攻击基线的30%以内，同时以最小的计算开销实现强大的恶意客户端检测性能。


### 论文摘要

Federated learning (FL) enables collaborative model training while preserving data privacy. However, it remains vulnerable to malicious clients who compromise model integrity through Byzantine attacks, data poisoning, or adaptive adversarial behaviors. Existing defense mechanisms rely on static thresholds and binary classification, failing to adapt to evolving client behaviors in real-world deployments. We propose FLARE, an adaptive reputation-based framework that transforms client reliability assessment from binary decisions to a continuous, multi-dimensional trust evaluation. FLARE integrates: (i) a multi-dimensional reputation score capturing performance consistency, statistical anomaly indicators, and temporal behavior, (ii) a self-calibrating adaptive threshold mechanism that adjusts security strictness based on model convergence and recent attack intensity, (iii) reputation-weighted aggregation with soft exclusion to proportionally limit suspicious contributions rather than eliminating clients outright, and (iv) a Local Differential Privacy (LDP) mechanism enabling reputation scoring on privatized client updates. We further introduce a highly evasive Statistical Mimicry (SM) attack, a benchmark adversary that blends honest gradients with synthetic perturbations and persistent drift to remain undetected by traditional filters. Extensive experiments with 100 clients on MNIST, CIFAR-10, and SVHN demonstrate that FLARE maintains high model accuracy and converges faster than state-of-the-art Byzantine-robust methods under diverse attack types, including label flipping, gradient scaling, adaptive attacks, ALIE, and SM. FLARE improves robustness by up to 16% and preserves model convergence within 30% of the non-attacked baseline, while achieving strong malicious-client detection performance with minimal computational overhead. https://github.com/Anonymous0-0paper/FLARE

---

## 88. Seeing Beyond the Image: ECG and Anatomical Knowledge-Guided Myocardial Scar Segmentation from Late Gadolinium-Enhanced Images

**论文链接:** [http://arxiv.org/abs/2511.14702v1](http://arxiv.org/abs/2511.14702v1)

**作者:** Farheen Ramzan, Yusuf Kiberu, Nikesh Jathanna, Meryem Jabrane, Vicente Grau, Shahnaz Jamil-Copley, Richard H. Clayton, Chen, Chen

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种新颖的多模态框架，通过整合心电图(ECG)导出的电生理信息和AHA-17图谱的解剖先验信息，实现了从晚期钆增强(LGE)心脏MRI中更准确的心肌瘢痕分割。

### 背景

从LGE心脏MRI中准确分割心肌瘢痕对于评估组织活力至关重要，但由于对比度变化和成像伪影而具有挑战性。ECG信号提供互补的生理信息，因为传导异常可以帮助定位或提示瘢痕心肌区域。

### 目的

开发一种整合生理和解剖知识的多模态框架，实现生理一致的LGE基础瘢痕分割，提高分割准确性。

### 方法

提出时间感知特征融合(TAFF)机制，根据ECG和LGE-MRI的采集时间差异动态加权和融合特征，将电生理信息与解剖先验信息有效结合。

### 主要发现

在临床数据集上评估，与最先进的仅图像基线方法(nnU-Net)相比，将瘢痕分割的平均Dice分数从0.6149提高到0.8463，精确度达0.9115，灵敏度达0.9043。

### 结论

整合生理和解剖知识使模型能够超越图像限制，为稳健和生理基础的心脏瘢痕分割设定了新方向。

### 翻译

从晚期钆增强(LGE)心脏MRI中准确分割心肌瘢痕对于评估组织活力至关重要，但由于对比度变化和成像伪影而具有挑战性。心电图(ECG)信号提供互补的生理信息，因为传导异常可以帮助定位或提示瘢痕心肌区域。在本工作中，我们提出了一种新颖的多模态框架，整合ECG导出的电生理信息和来自AHA-17图谱的解剖先验信息，用于生理一致的LGE基础瘢痕分割。由于ECG和LGE-MRI不是同时采集的，我们引入了时间感知特征融合(TAFF)机制，根据采集时间差异动态加权和融合特征。我们的方法在临床数据集上进行了评估，与最先进的仅图像基线(nnU-Net)相比取得了显著增益，将瘢痕的平均Dice分数从0.6149提高到0.8463，并在精确度(0.9115)和灵敏度(0.9043)方面都取得了高性能。这些结果表明，整合生理和解剖知识使模型能够'超越图像'，为稳健和生理基础的心脏瘢痕分割设定了新方向。


### 论文摘要

Accurate segmentation of myocardial scar from late gadolinium enhanced (LGE) cardiac MRI is essential for evaluating tissue viability, yet remains challenging due to variable contrast and imaging artifacts. Electrocardiogram (ECG) signals provide complementary physiological information, as conduction abnormalities can help localize or suggest scarred myocardial regions. In this work, we propose a novel multimodal framework that integrates ECG-derived electrophysiological information with anatomical priors from the AHA-17 atlas for physiologically consistent LGE-based scar segmentation. As ECGs and LGE-MRIs are not acquired simultaneously, we introduce a Temporal Aware Feature Fusion (TAFF) mechanism that dynamically weights and fuses features based on their acquisition time difference. Our method was evaluated on a clinical dataset and achieved substantial gains over the state-of-the-art image-only baseline (nnU-Net), increasing the average Dice score for scars from 0.6149 to 0.8463 and achieving high performance in both precision (0.9115) and sensitivity (0.9043). These results show that integrating physiological and anatomical knowledge allows the model to "see beyond the image", setting a new direction for robust and physiologically grounded cardiac scar segmentation.

---

## 89. HyMAD: A Hybrid Multi-Activity Detection Approach for Border Surveillance and Monitoring

**论文链接:** [http://arxiv.org/abs/2511.14698v1](http://arxiv.org/abs/2511.14698v1)

**作者:** Sriram Srinivasan, Srinivasan Aruchamy, Siva Ram Krisha Vadali

**发布时间:** 2025-11-18

**备注:** Multi-label seismic signal classification using novel attention-based feature fusion. Submitting to cs.CV due to relevance to general pattern recognition and time-frequency (spectrogram) analysis

### GPT解析

### 总结

该研究提出了一种名为HyMAD的深度神经网络架构，用于解决地震传感技术在边境监控中同时发生的多种活动难以区分的问题，实现了对人类入侵、动物移动和车辆行驶等活动的准确识别。

### 背景

地震传感技术作为边境监控的有前途解决方案，传感器埋于地下难以被入侵者检测或破坏，相比可见的摄像头或围栏更有效。然而，地震信号复杂且噪声大，准确检测和区分同时发生的重叠活动（如人类入侵、动物移动和车辆行驶）仍是一大挑战。

### 目的

开发一种能够准确识别和区分同时发生的多种地震活动的技术，提高边境监控系统的可靠性和有效性。

### 方法

提出HyMAD（混合多活动检测）深度神经网络架构，基于时空特征融合，集成使用SincNet提取的频谱特征和RNN建模的时间依赖性，采用自注意力层增强模内表示，并使用跨模态融合模块实现地震事件的鲁棒多标签分类。

### 主要发现

在从边境监控实地录音构建的数据集上评估，HyMAD展示了处理涉及人类、动物和车辆的复杂同时活动场景的泛化能力，取得了具有竞争力的性能。

### 结论

HyMAD提供了一个模块化框架，可扩展基于地震的活动识别技术，应用于实际安全领域，有效解决了同时发生活动难以区分的问题，提高了边境监控系统的可靠性。

### 翻译

地震传感技术已成为边境监控和监测的一种有前途的解决方案；通常埋在地下的地震传感器体积小且不易被发现，使入侵者难以检测、规避或破坏。与高度可见的摄像头或围栏相比，这显著提高了它们的有效性。然而，由于地震信号的复杂性和噪声特性，准确检测和区分同时发生的重叠活动（如人类入侵、动物移动和车辆行驶）仍然是一个重大挑战。正确识别同时发生的活动至关重要，因为无法分离它们可能导致错误分类、漏检以及对情况的不完整理解，从而降低监控系统的可靠性。为了解决这个问题，我们提出了HyMAD（混合多活动检测），一种基于时空特征融合的深度神经架构。该框架集成了使用SincNet提取的频谱特征和由循环神经网络（RNN）建模的时间依赖性。此外，HyMAD采用自注意力层来增强模内表示，并使用跨模态融合模块实现地震事件的鲁棒多标签分类。我们在从边境监控和监测背景下收集的实地录音构建的数据集上评估了我们的方法，展示了其在处理涉及人类、动物和车辆的复杂、同时活动场景时的泛化能力。我们的方法取得了具有竞争力的性能，并为扩展基于地震的活动识别提供了模块化框架，适用于实际安全应用。


### 论文摘要

Seismic sensing has emerged as a promising solution for border surveillance and monitoring; the seismic sensors that are often buried underground are small and cannot be noticed easily, making them difficult for intruders to detect, avoid, or vandalize. This significantly enhances their effectiveness compared to highly visible cameras or fences. However, accurately detecting and distinguishing between overlapping activities that are happening simultaneously, such as human intrusions, animal movements, and vehicle rumbling, remains a major challenge due to the complex and noisy nature of seismic signals. Correctly identifying simultaneous activities is critical because failing to separate them can lead to misclassification, missed detections, and an incomplete understanding of the situation, thereby reducing the reliability of surveillance systems. To tackle this problem, we propose HyMAD (Hybrid Multi-Activity Detection), a deep neural architecture based on spatio-temporal feature fusion. The framework integrates spectral features extracted with SincNet and temporal dependencies modeled by a recurrent neural network (RNN). In addition, HyMAD employs self-attention layers to strengthen intra-modal representations and a cross-modal fusion module to achieve robust multi-label classification of seismic events. e evaluate our approach on a dataset constructed from real-world field recordings collected in the context of border surveillance and monitoring, demonstrating its ability to generalize to complex, simultaneous activity scenarios involving humans, animals, and vehicles. Our method achieves competitive performance and offers a modular framework for extending seismic-based activity recognition in real-world security applications.

---

## 90. SkillGen: Learning Domain Skills for In-Context Sequential Decision Making

**论文链接:** [http://arxiv.org/abs/2511.14670v1](http://arxiv.org/abs/2511.14670v1)

**作者:** Ruomeng Ding, Wei Cheng, Minglai Shao, Chen Zhao

**发布时间:** 2025-11-18

### GPT解析

### 总结

SkillGen是一种基于技能的ICL框架，通过构建以动作为中心的领域级图，识别高效用动作并检索步骤级技能，生成细粒度、上下文感知的提示，在顺序决策任务中取得了显著的性能提升。

### 背景

大型语言模型(LLMs)通过上下文学习(ICL)越来越多地应用于顺序决策-making，但其效果对提示质量高度敏感。有效提示应满足三个原则：关注决策关键信息、提供步骤级粒度、最小化对专家标注的依赖。

### 目的

解决现有ICL方法无法同时满足有效提示的三个原则的问题，提出一种能够同时满足这些原则的ICL框架。

### 方法

SkillGen框架从采样轨迹中构建以动作为中心、领域级别的图，通过时间差分信用分配识别高效用动作，并检索步骤级技能以生成细粒度、上下文感知的提示。

### 主要发现

理论分析表明，关注高效用段支持任务可识别性，并为更有效的ICL提示设计提供指导。实验显示SkillGen在ALFWorld、BabyAI和ScienceWorld上平均提高进度率5.9%-16.5%。

### 结论

SkillGen框架有效解决了现有ICL方法的局限性，通过关注高效用动作和技能，提供了更有效的提示设计方法，在顺序决策任务中取得了显著的性能提升。

### 翻译

大型语言模型(LLMs)越来越多地通过上下文学习(ICL)应用于顺序决策-making，但其效果对提示质量高度敏感。有效的提示应满足三个原则：关注决策关键信息，提供步骤级粒度，并通过标签效率最小化对专家标注的依赖。然而，现有的ICL方法往往无法同时满足这三个标准。受这些挑战的启发，我们引入了SkillGen，一种用于结构化顺序推理的基于技能的ICL框架。它从采样轨迹中构建以动作为中心、领域级别的图，通过时间差分信用分配识别高效用动作，并检索步骤级技能以生成细粒度、上下文感知的提示。我们进一步提出了一个理论分析，表明关注高效用段支持任务可识别性，并为更有效的ICL提示设计提供指导。在ALFWorld、BabyAI和ScienceWorld上使用开源和专有LLMs的实验表明，SkillGen取得了一致的提升，平均提高各模型的进度率5.9%-16.5%。


### 论文摘要

Large language models (LLMs) are increasingly applied to sequential decision-making through in-context learning (ICL), yet their effectiveness is highly sensitive to prompt quality. Effective prompts should meet three principles: focus on decision-critical information, provide step-level granularity, and minimize reliance on expert annotations through label efficiency. However, existing ICL methods often fail to satisfy all three criteria simultaneously. Motivated by these challenges, we introduce SkillGen, a skill-based ICL framework for structured sequential reasoning. It constructs an action-centric, domain-level graph from sampled trajectories, identifies high-utility actions via temporal-difference credit assignment, and retrieves step-wise skills to generate fine-grained, context-aware prompts. We further present a theoretical analysis showing that focusing on high-utility segments supports task identifiability and informs more effective ICL prompt design. Experiments on ALFWorld, BabyAI, and ScienceWorld, using both open-source and proprietary LLMs, show that SkillGen achieves consistent gains, improving progress rate by 5.9%-16.5% on average across models.

---

## 91. Estimation of Spatial and Temporal Autoregressive Effects using LASSO - An Example of Hourly Particulate Matter Concentrations

**论文链接:** [http://arxiv.org/abs/2511.14666v1](http://arxiv.org/abs/2511.14666v1)

**作者:** Elkanah Nyabuto, Philipp Otto, Yarema Okhrin

**发布时间:** 2025-11-18

**备注:** 27 pages, 11 figures, 4 tables. Under revision at Environmetrics

### GPT解析

### 总结

本研究提出了一种使用最小绝对收缩和选择算子(LASSO)估计时空自回归面板数据模型中时空效应的方法，并通过蒙特卡洛模拟验证了其有效性，最终应用于德国巴伐利亚地区PM10浓度的分析。

### 背景

时空数据在环境监测等领域广泛存在，但如何有效估计其中的时空效应是一个挑战。传统的时空自回归模型包含空间/时间变化的外生回归项、时间自回归项和具有未知权重矩阵的空间自回归项。

### 目的

开发一种约束惩罚最大似然估计器，用于估计时空自回归面板数据模型中的权重矩阵和其他参数，同时提高模型的可解释性。

### 方法

使用LASSO技术估计时空自回归面板数据模型中的参数，通过蒙特卡洛模拟评估方法性能，并将该方法应用于德国巴伐利亚地区2005年至2020年的小时颗粒物浓度(PM10)数据分析。

### 主要发现

蒙特卡洛模拟显示该方法性能良好，准确性随时间点数量增加而提高；LASSO技术能够有效区分有意义和无意义的关系；应用于PM10数据分析发现部分监测站点具有高度空间依赖性，邻近站点浓度相互影响显著。

### 结论

LASSO技术通过产生稀疏权重矩阵(将一些权重收缩为零)提高了时空模型的性能和可解释性，有效捕捉了巴伐利亚地区PM浓度在监测站点间的依赖关系。

### 翻译

我们提出了一种使用最小绝对收缩和选择算子(LASSO)估计时空自回归面板数据模型中时空效应的方法。我们假设时空面板来自单变量随机过程，数据遵循时空自回归过程，包括空间/时间变化的外生回归项、时间自回归项和具有未知权重矩阵的空间自回归项。目标是使用约束惩罚最大似然估计器估计该权重矩阵和其他参数。蒙特卡洛模拟显示性能良好，准确性随时间点数量增加而提高。LASSO技术还能一致地区分空间权重和其他参数中有意义的关系(非零)和无意义的关系(现有零)。这种正则化估计程序应用于德国巴伐利亚地区2005年至2020年的小时颗粒物浓度(PM10)。结果显示一些站点具有高度空间依赖性，导致邻近监测站PM10浓度的影响更大。LASSO技术被证明通过将一些权重收缩为零产生稀疏权重矩阵，从而提高了巴伐利亚地区测量站点间PM浓度依赖性的可解释性。


### 论文摘要

We present an estimation procedure of spatial and temporal effects in spatiotemporal autoregressive panel data models using the Least Absolute Shrinkage and Selection Operator, LASSO (Tibshirani, 1996). We assume that the spatiotemporal panel is drawn from a univariate random process and that the data follows a spatiotemporal autoregressive process which includes a regressive term with space-/ time-varying exogenous regressor, a temporal autoregressive term and a spatial autoregressive term with an unknown weights matrix. The aim is to estimate this weight matrix alongside other parameters using a constraint penalised maximum likelihood estimator. Monte Carlo simulations showed a good performance with the accuracy increasing with an increasing number of time points. The use of the LASSO technique also consistently distinguishes between meaningful relationships (non-zeros) from those that are not (existing zeros) in both the spatial weights and other parameters. This regularised estimation procedure is applied to hourly particulate matter concentrations (PM10) in the Bavaria region, Germany for the years 2005 to 2020. Results show some stations with a high spatial dependency, resulting in a greater influence of PM10 concentrations in neighbouring monitoring stations. The LASSO technique proved to produce a sparse weights matrix by shrinking some weights to zero, hence improving the interpretability of the PM concentration dependencies across measurement stations in Bavaria

---

## 92. Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms

**论文链接:** [http://arxiv.org/abs/2511.14654v1](http://arxiv.org/abs/2511.14654v1)

**作者:** Marius Dubosc, Yann Fischer, Zacharie Auray, Nicolas Boutry, Edwin Carlinet, Michael Atlan, Thierry Geraud

**发布时间:** 2025-11-18

**备注:** 5 pages, 3 figures, 1 table. Submitted to ISBI2026

### GPT解析

### 总结

本研究提出了一种简单而有效的方法，通过时间分辨预处理使传统U-Net分割模型能够利用时间动态信息，在Doppler holography视网膜成像中实现动脉和静脉的准确分割，性能可与复杂模型相媲美。

### 背景

Doppler holography是一种新兴的视网膜成像技术，能够高时间分辨率地捕捉血液流动的动态行为，为视网膜血液动力学的定量评估提供了可能。

### 目的

解决传统视网膜动脉和静脉分割方法仅关注空间信息而忽略时间信息的问题，使分割方法能够充分利用Doppler holography数据中的时间动态特性。

### 方法

提出一种基于标准分割架构的方法，通过纳入专用脉冲分析管道衍生的特征，使传统U-Net模型能够利用时间动态信息进行动脉-静脉分割。

### 主要发现

时间分辨预处理可以使传统U-Net分割模型实现与更复杂的基于注意力或迭代模型的相当性能；时间分辨预处理能够释放深度学习在Doppler holography中的全部潜力。

### 结论

时间分辨预处理技术为视网膜血液动力学的定量探索开辟了新视角，相关数据集已公开可供研究使用。

### 翻译

多普勒全息术是一种新兴的视网膜成像技术，能够以高时间分辨率捕捉血液流动的动态行为，实现视网膜血液动力学的定量评估。这需要准确分割视网膜动脉和静脉，但传统分割方法仅关注空间信息，忽略了全息数据的丰富时间信息。在本工作中，我们提出了一种简单而有效的方法，使用标准分割架构对时间多普勒全息图进行动脉-静脉分割。通过纳入专用脉冲分析管道衍生的特征，我们的方法使传统U-Net能够利用时间动态，并实现与更复杂的基于注意力或迭代模型的相当性能。这些发现表明，时间分辨预处理可以释放深度学习在多普勒全息术中的全部潜力，为视网膜血液动力学的定量探索开辟新视角。数据集可在https://huggingface.co/datasets/DigitalHolography/公开获取。


### 论文摘要

Doppler holography is an emerging retinal imaging technique that captures the dynamic behavior of blood flow with high temporal resolution, enabling quantitative assessment of retinal hemodynamics. This requires accurate segmentation of retinal arteries and veins, but traditional segmentation methods focus solely on spatial information and overlook the temporal richness of holographic data. In this work, we propose a simple yet effective approach for artery-vein segmentation in temporal Doppler holograms using standard segmentation architectures. By incorporating features derived from a dedicated pulse analysis pipeline, our method allows conventional U-Nets to exploit temporal dynamics and achieve performance comparable to more complex attention- or iteration-based models. These findings demonstrate that time-resolved preprocessing can unlock the full potential of deep learning for Doppler holography, opening new perspectives for quantitative exploration of retinal hemodynamics. The dataset is publicly available at https://huggingface.co/datasets/DigitalHolography/

---

## 93. Fusing Biomechanical and Spatio-Temporal Features for Fall Prediction: Characterizing and Mitigating the Simulation-to-Reality Gap

**论文链接:** [http://arxiv.org/abs/2511.14620v1](http://arxiv.org/abs/2511.14620v1)

**作者:** Md Fokhrul Islam, Sajeda Al-Hammouri, Christopher J. Arellano, Kavan Hazeli, Heman Shakeri

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种名为BioST-GCN的双流模型，结合姿势和生物力学信息来预测老年人跌倒，但在模拟数据和现实世界之间存在显著性能差距。

### 背景

跌倒是导致老年人受伤和丧失独立性的主要原因。基于视觉的跌倒预测系统提供非侵入性解决方案，但可用跌倒数据的稀缺阻碍了其发展。

### 目的

开发一种改进的跌倒预测模型，并解决模拟数据与现实世界之间的差距问题。

### 方法

提出生物力学时空图卷积网络(BioST-GCN)，这是一种双流模型，使用交叉注意力融合机制结合姿势和生物力学信息。

### 主要发现

模型在模拟数据集上比基础ST-GCN模型提高5.32%和2.91%的F1分数；时空注意力机制提供了可解释性；在模拟数据上完全监督达到89.0%的F1分数，但零样本泛化到未见过的受试者下降到35.9%；这种差距在患有糖尿病或体弱的老年人中更为明显。

### 结论

需要弥合模拟和现实数据之间的差距，提出了个性化策略和隐私保护的数据管道以实现现实世界验证，以便开发针对弱势老年人群体的有效跌倒预测系统。

### 翻译

跌倒是导致老年人受伤和丧失独立性的主要原因。基于视觉的跌倒预测系统提供了一种非侵入性解决方案，可在撞击前几秒预测跌倒，但其发展受到可用跌倒数据稀缺的阻碍。本研究提出了生物力学时空图卷积网络(BioST-GCN)，这是一种双流模型，使用交叉注意力融合机制结合姿势和生物力学信息。我们的模型在模拟的MCF-UA特技演员和MUVIM数据集上分别比基础ST-GCN模型提高了5.32%和2.91%的F1分数。ST-GCN流中的时空注意力机制通过识别关键关节和时间阶段提供了可解释性。然而，严重的模拟-现实差距仍然存在。虽然我们的模型在模拟数据上完全监督达到89.0%的F1分数，但零样本泛化到未见过的受试者下降到35.9%。这种性能下降可能是由于模拟数据中的偏差，如'意图跌倒'的线索。对于患有糖尿病或体弱的老年人，这种差距因他们独特的运动学特征而加剧。为解决此问题，我们提出了个性化策略，并倡导隐私保护的数据管道以实现现实世界验证。我们的研究结果强调了弥合模拟和现实数据之间差距的紧迫性，以便为弱势老年人群开发有效的跌倒预测系统。


### 论文摘要

Falls are a leading cause of injury and loss of independence among older adults. Vision-based fall prediction systems offer a non-invasive solution to anticipate falls seconds before impact, but their development is hindered by the scarcity of available fall data. Contributing to these efforts, this study proposes the Biomechanical Spatio-Temporal Graph Convolutional Network (BioST-GCN), a dual-stream model that combines both pose and biomechanical information using a cross-attention fusion mechanism. Our model outperforms the vanilla ST-GCN baseline by 5.32% and 2.91% F1-score on the simulated MCF-UA stunt-actor and MUVIM datasets, respectively. The spatio-temporal attention mechanisms in the ST-GCN stream also provide interpretability by identifying critical joints and temporal phases. However, a critical simulation-reality gap persists. While our model achieves an 89.0% F1-score with full supervision on simulated data, zero-shot generalization to unseen subjects drops to 35.9%. This performance decline is likely due to biases in simulated data, such as `intent-to-fall' cues. For older adults, particularly those with diabetes or frailty, this gap is exacerbated by their unique kinematic profiles. To address this, we propose personalization strategies and advocate for privacy-preserving data pipelines to enable real-world validation. Our findings underscore the urgent need to bridge the gap between simulated and real-world data to develop effective fall prediction systems for vulnerable elderly populations.

---

## 94. OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.14582v1](http://arxiv.org/abs/2511.14582v1)

**作者:** Keda Tao, Kele Shao, Bohan Yu, Weiqiang Wang, Jian liu, Huan Wang

**发布时间:** 2025-11-18

**备注:** Code Link: https://github.com/KD-TAO/OmniZip

### GPT解析

### 总结

本文提出了OmniZip，一种无需训练的、音频引导的视听标记压缩框架，用于优化多模态标记表示和加速推理，解决了Omnimodal大语言模型处理音频视频标记序列时的计算瓶颈问题。

### 背景

Omnimodal大语言模型在统一的音频视频理解方面吸引了越来越多的研究关注，但处理音频视频标记序列造成了显著的计算瓶颈。现有的标记压缩方法还不能满足联合压缩多模态标记的需求。

### 目的

填补现有标记压缩方法无法满足联合压缩多模态标记需求的空白，提出一种优化多模态标记表示和加速推理的框架。

### 方法

OmniZip首先识别重要的音频标记，然后计算每个时间组的音频保留分数以捕获信息密度，从而动态引导视频标记修剪并保留由跨模态相似性增强的音频锚点提示。对于每个时间窗口，OmniZip使用交错时空方案压缩视频标记。

### 主要发现

大量的实证结果表明，OmniZip比其他最先进的对应方法实现了3.42倍的推理加速和1.4倍的内存减少，同时在不进行训练的情况下保持了性能。

### 结论

OmniZip是一种有效的解决方案，能够解决Omnimodal大语言模型在处理音频视频标记序列时的计算瓶颈问题，无需训练即可实现显著的推理加速和内存减少。

### 翻译

Omnimodal大语言模型最近在统一的音频视频理解方面吸引了越来越多的研究关注，然而处理音频视频标记序列造成了显著的计算瓶颈。现有的标记压缩方法尚未满足联合压缩多模态标记的这一新兴需求。为了填补这一空白，我们提出了OmniZip，一种无需训练的、音频引导的视听标记压缩框架，用于优化多模态标记表示和加速推理。具体来说，OmniZip首先识别重要的音频标记，然后计算每个时间组的音频保留分数以捕获信息密度，从而动态引导视频标记修剪并保留由跨模态相似性增强的音频锚点提示。对于每个时间窗口，OmniZip使用交错时空方案压缩视频标记。大量的实证结果表明了OmniZip的优势——它比其他最先进的对应方法实现了3.42倍的推理加速和1.4倍的内存减少，同时在不进行训练的情况下保持了性能。


### 论文摘要

Omnimodal large language models (OmniLLMs) have attracted increasing research attention of late towards unified audio-video understanding, wherein processing audio-video token sequences creates a significant computational bottleneck, however. Existing token compression methods have yet to accommodate this emerging need of jointly compressing multimodal tokens. To bridge this gap, we present OmniZip, a training-free, audio-guided audio-visual token-compression framework that optimizes multimodal token representation and accelerates inference. Specifically, OmniZip first identifies salient audio tokens, then computes an audio retention score for each time group to capture information density, thereby dynamically guiding video token pruning and preserving cues from audio anchors enhanced by cross-modal similarity. For each time window, OmniZip compresses the video tokens using an interleaved spatio-temporal scheme. Extensive empirical results demonstrate the merits of OmniZip - it achieves 3.42X inference speedup and 1.4X memory reduction over other top-performing counterparts, while maintaining performance with no training.

---

## 95. Estimating differential pistons for the Extremely Large Telescope using focal plane imaging and a residual network

**论文链接:** [http://arxiv.org/abs/2511.14577v1](http://arxiv.org/abs/2511.14577v1)

**作者:** P. Janin-Potiron, M. Gray, B. Neichel, M. Dumont, J. -F. Sauvage, C. T. Heritier, P. Jouve, R. Fetick, T. Fusco

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种使用深度学习模型从2x2夏克-哈特曼波前传感器图像中估计极大望远镜M4镜面花瓣间差分活塞的方法，通过时间平均显著提高了估计精度，且方法在实际应用中具有鲁棒性。

### 背景

极大望远镜即将投入运行，优化其成像性能至关重要。差分活塞效应源于自适应光学控制回路、热机械效应或其他因素，会显著降低图像质量，损害望远镜整体性能。

### 目的

提出一种使用2x2夏克-哈特曼波前传感器图像估计ELT M4镜面花瓣间差分活塞的方法，并评估该方法在不同观测条件和噪声源下的局限性。

### 方法

在数值模拟设置中提出方法，使用基于ResNet架构的深度学习模型，在模拟数据集上训练神经网络来估计差分活塞，评估方法在不同条件下的鲁棒性，包括斯特雷尔比变化、多色性和探测器噪声，使用均方根误差量化性能。

### 主要发现

该方法能从2x2 SH-WFS图像中提取差分活塞信息；帧的时间平均使差分活塞信号从湍流引起的散斑场中显现，显著改善了RMSE计算；更好的视宁度带来更高准确性；与单色情况相比，多色性仅使性能下降不到5%；探测器噪声不是主要限制因素，主要限制来自于需要足够的散斑平均；该方法也适用于其他类型的输入图像。

### 结论

该方法为估计ELT M4镜面花瓣间的差分活塞提供了一种有效途径，通过时间平均可以提高估计精度，方法在实际应用中具有鲁棒性，对多种条件不敏感。

### 翻译

随着极大望远镜接近运营状态，优化其成像性能至关重要。源于自适应光学控制回路、热机械效应或其他因素的差分活塞显著降低了图像质量，并对望远镜的整体性能造成不利影响。在数值模拟设置中，我们提出了一种使用ELT层析成像AO模式中常用的2x2夏克-哈特曼波前传感器(SH-WFS)图像来估计ELT M4镜面花瓣间差分活塞的方法。我们旨在通过评估该方法对不同观测条件和噪声源的敏感性来确定其局限性。我们使用基于ResNet架构的深度学习模型，在模拟数据集上训练神经网络(NN)来估计差分活塞。我们评估了该方法在各种条件下的鲁棒性，包括斯特雷尔比的变化、多色性和探测器噪声。性能使用估计的差分活塞像差的均方根误差(RMSE)进行量化。该方法展示了从2x2 SH-WFS图像中提取差分活塞信息的能力。帧的时间平均使差分活塞信号从湍流引起的散斑场中显现出来，并显著改善了RMSE计算。正如预期，更好的视宁度条件提高了准确性。与单色情况相比，多色性仅使性能下降不到5%。在实际场景中，探测器噪声不是限制因素，主要限制来自于需要足够的散斑平均。该网络也被证明适用于除2x2 SH-WFS数据以外的输入图像。


### 论文摘要

As the Extremely Large Telescope (ELT) approaches operational status, optimising its imaging performance is critical. A differential piston, arising from either the adaptive optics (AO) control loop, thermomechanical effects, or other sources, significantly degrades the image quality and is detrimental to the telescope's overall performance. In a numerical simulation set-up, we propose a method for estimating the differential piston between the petals of the ELT's M4 mirror using images from a 2x2 Shack-Hartmann wavefront sensor (SH-WFS), commonly used in the ELT's tomographic AO mode. We aim to identify the limitations of this approach by evaluating its sensitivity to various observing conditions and sources of noise. Using a deep learning model based on a ResNet architecture, we trained a neural network (NN) on simulated datasets to estimate the differential piston. We assessed the robustness of the method under various conditions, including variations in Strehl ratio, polychromaticity, and detector noise. The performance was quantified using the root mean square error (RMSE) of the estimated differential piston aberration. This method demonstrates the ability to extract differential piston information from 2x2 SH-WFS images. Temporal averaging of frames makes the differential piston signal emerge from the turbulence-induced speckle field and leads to a significant improvement in the RMSE calculation. As expected, better seeing conditions result in improved accuracy. Polychromaticity only degrades the performance by less than 5% compared to the monochromatic case. In a realistic scenario, detector noise is not a limiting factor, as the primary limitation rather arises from the need for sufficient speckle averaging. The network was also shown to be applicable to input images other than the 2x2 SH-WFS data.

---

## 96. DeepBlip: Estimating Conditional Average Treatment Effects Over Time

**论文链接:** [http://arxiv.org/abs/2511.14545v1](http://arxiv.org/abs/2511.14545v1)

**作者:** Haorui Ma, Dennis Frauen, Stefan Feuerriegel

**发布时间:** 2025-11-18

**备注:** 42 pages

### GPT解析

### 总结

这篇论文提出了DeepBlip，这是第一个用于结构嵌套均值模型(SNMMs)的神经网络框架。通过双重优化技巧克服了SNMMs无法进行端到端训练的限制，能够同时学习所有blip函数，整合序列神经网络捕捉时间依赖性，正确调整时间变化的混杂因素产生无偏估计，并在多个临床数据集上达到最先进的性能。

### 背景

结构嵌套均值模型(SNMMs)是一种估计随时间变化的治疗效果的原则性方法，其特殊优势是将治疗序列的联合效应分解为局部的、特定时间点的'blip效应'，提高了可解释性并允许高效评估最佳治疗策略。然而，由于SNMMs固有的序列g-估计方案，缺乏相应的神经框架，阻止了端到端的基于梯度的训练。

### 目的

开发第一个用于SNMMs的神经网络框架，克服序列g-估计方案对端到端训练的限制，实现所有blip函数的同时学习，并正确调整时间变化的混杂因素以产生无偏估计。

### 方法

作者提出了DeepBlip，这是一种新颖的双重优化技巧，使所有blip函数能够同时学习。DeepBlip无缝集成了序列神经网络(如LSTM或transformer)来捕获复杂的时间依赖性。该方法通过设计正确调整了时间变化的混杂因素，使用Neyman-正交损失函数确保对模型误规格的鲁棒性。

### 主要发现

DeepBlip成功应用于SNMMs，克服了序列g-估计方案的训练限制。该方法能够正确调整时间变化的混杂因素，产生无偏估计，并且对模型误规格具有鲁棒性。在各种临床数据集上的评估显示，DeepBlip达到了最先进的性能。

### 结论

DeepBlip是第一个用于SNMMs的神经框架，通过双重优化技巧克服了训练限制，实现了所有blip函数的同时学习。该方法能够捕捉复杂的时间依赖性，产生无偏估计，并对模型误规格具有鲁棒性，在多个临床数据集上展示了最先进的性能。

### 翻译

结构嵌套均值模型(SNMMs)是一种估计随时间变化的治疗效果的原则性方法。SNMMs的一个特殊优势是将治疗序列的联合效应分解为局部的、特定时间点的'blip效应'。这种分解通过增量效应提高了可解释性，并能够在不重新计算的情况下高效地评估最佳治疗策略的离线效果。然而，由于SNMMs固有的序列g-估计方案，缺乏神经框架，这阻止了端到端的基于梯度的训练。在这里，我们提出了DeepBlip，这是第一个用于SNMMs的神经框架，它通过一种新颖的双重优化技巧克服了这一限制，使所有blip函数能够同时学习。我们的DeepBlip无缝集成了序列神经网络(如LSTM或transformer)来捕获复杂的时间依赖性。通过设计，我们的方法正确调整了随时间变化的混杂因素以产生无偏估计，并且其Neyman-正交损失函数确保了对 nuisance 模型误规格的鲁棒性。最后，我们在各种临床数据集上评估了我们的DeepBlip，它达到了最先进的性能。


### 论文摘要

Structural nested mean models (SNMMs) are a principled approach to estimate the treatment effects over time. A particular strength of SNMMs is to break the joint effect of treatment sequences over time into localized, time-specific ``blip effects''. This decomposition promotes interpretability through the incremental effects and enables the efficient offline evaluation of optimal treatment policies without re-computation. However, neural frameworks for SNMMs are lacking, as their inherently sequential g-estimation scheme prevents end-to-end, gradient-based training. Here, we propose DeepBlip, the first neural framework for SNMMs, which overcomes this limitation with a novel double optimization trick to enable simultaneous learning of all blip functions. Our DeepBlip seamlessly integrates sequential neural networks like LSTMs or transformers to capture complex temporal dependencies. By design, our method correctly adjusts for time-varying confounding to produce unbiased estimates, and its Neyman-orthogonal loss function ensures robustness to nuisance model misspecification. Finally, we evaluate our DeepBlip across various clinical datasets, where it achieves state-of-the-art performance.

---

## 97. A Bayesian INLA-SPDE Approach to Spatio-Temporal Point-Grid Fusion with Change-of-Support and Misaligned Covariates

**论文链接:** [http://arxiv.org/abs/2511.14535v1](http://arxiv.org/abs/2511.14535v1)

**作者:** Weiyue Zheng, Andrew Elliott, Claire Miller, Marian Scott

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种时空数据融合框架，用于处理在不同空间支持上观测的点数据和网格数据，能够同时融合多个数据源，处理协变量不对齐问题，并进行时空预测。

### 背景

现有方法难以有效融合不同空间和时间分辨率、不同测量误差的多个数据源，以及处理协变量不对齐的问题。

### 目的

开发一个层次模型，联合融合同一变量在不同空间和时间分辨率下的多个数据源，并通过同一数据融合框架整合不对齐的协变量。

### 方法

使用具有Matérn-SPDE先验的潜在高斯场提供连续空间表示，特定于源的观测算子处理支持变化和协变量不对齐问题，结合时间依赖性实现时空预测，采用集成嵌套拉普拉斯近似和随机偏微分方程方法进行推断和预测。

### 主要发现

通过模拟实验验证了框架的效用，探索了方法随时间点数量和数据/协变量可用性变化的稳定性和性能，点数据和网格数据融合相比单一数据源模型有所改进。

### 结论

该框架成功应用于苏格兰Angus的Elliot Water流域的土壤制图，融合了现场传感器数据、对齐和不对齐的协变量、卫星数据和海拔数据，生成了具有不确定性的每日高分辨率地图。

### 翻译

我们提出了一种时空数据融合框架，用于处理在不同空间支持上观测的点数据和网格数据。具有Matérn-SPDE先验的潜在高斯场提供连续空间表示，而特定于源的观测算子将观测映射到点测量和网格平均值，解决了支持变化和协变量不对齐问题。此外，结合时间依赖性能够在未知位置和时间点进行预测。使用集成嵌套拉普拉斯近似和随机偏微分方程方法进行推断和预测，实现快速计算和不确定性量化。我们的贡献是：一个层次模型，联合融合同一变量在不同空间和时间分辨率下的多个数据源和不同测量误差，以及一个实际实现，通过同一数据融合框架整合不对齐的协变量，允许不同的协变量支持。我们通过校准到真实传感器密度和空间覆盖范围的模拟实验证明了该框架的效用。使用模拟框架，我们探索了方法随时间点数量和数据/协变量可用性变化的稳定性和性能，通过点数据和网格数据融合展示了相比单一数据源模型的改进。我们将该框架应用于苏格兰Angus的Elliot Water流域的土壤制图。我们融合了现场传感器数据、对齐和不对齐的协变量、卫星数据和海拔数据，生成了具有不确定性的每日高分辨率地图。


### 论文摘要

We propose a spatio-temporal data-fusion framework for point data and gridded data with variables observed on different spatial supports. A latent Gaussian field with a Matérn-SPDE prior provides a continuous space representation, while source-specific observation operators map observations to both point measurements and gridded averages, addressing change-of-support and covariate misalignment. Additionally incorporating temporal dependence enables prediction at unknown locations and time points. Inference and prediction are performed using the Integrated Nested Laplace Approximation and the Stochastic Partial Differential Equations approach, which delivers fast computation with uncertainty quantification. Our contributions are: a hierarchical model that jointly fuses multiple data sources of the same variable under different spatial and temporal resolutions and measurement errors, and a practical implementation that incorporates misaligned covariates via the same data fusion framework allowing differing covariate supports. We demonstrate the utility of this framework via simulations calibrated to realistic sensor densities and spatial coverage. Using the simulation framework, we explore the stability and performance of the approach with respect to the number of time points and data/covariate availability, demonstrating gains over single-source models through point and gridded data fusion. We apply our framework to soil moisture mapping in the Elliot Water catchment (Angus, Scotland). We fuse in-situ sensor data with aligned and misaligned covariates, satellite data and elevation data to produce daily high resolution maps with uncertainty.

---

## 98. Spatio-temporal Hawkes point processes: statistical inference and simulation strategies

**论文链接:** [http://arxiv.org/abs/2511.14509v1](http://arxiv.org/abs/2511.14509v1)

**作者:** Alba Bernabeu, Jorge Mateu

**发布时间:** 2025-11-18

### GPT解析

### 总结

时空霍克斯点过程是一类特殊的随机点过程，用于建模自激行为，其中事件的发生会增加其他事件发生的概率，能够处理时空现象中随机和确定性成分之间的复杂相互关系。

### 背景

尽管时空霍克斯点过程在实践中被广泛使用，但缺乏通用和统一的正式化方法，每篇论文都提出了对这些随机机制的不同观点。

### 目的

实现两种模拟技术和三种统一且自洽的推断技术，这些技术在时空霍克斯过程的实际建模中被广泛使用，并评估这些方法的实际性能。

### 方法

实现了两种模拟技术和三种统一、自洽的推断技术，同时为可重复性提供了有用的代码。

### 主要发现

评估了所实现的模拟和推断技术在时空霍克斯点过程建模中的实际性能。

### 结论

通过提供统一的框架和实用代码，促进了时空霍克斯点过程在实际应用中的标准化和可重复性。

### 翻译

时空霍克斯点过程是一类特别有趣的随机点过程，用于建模自激行为，在这种情况下，一个事件的发生会增加其他事件发生的概率。这些过程能够处理时空现象中随机和确定性成分之间的复杂相互关系。然而，尽管在实践中被广泛使用，但没有普遍且统一的正式化方法，每篇论文都提出了对这些随机机制的不同观点。考虑到这一点，我们实现了两种模拟技术和三种统一、自洽的推断技术，这些技术在时空霍克斯过程的实际建模中被广泛使用。此外，我们评估了这些方法的实际性能，同时为可重复性提供了有用的代码。


### 论文摘要

Spatio-temporal Hawkes point processes are a particularly interesting class of stochastic point processes for modeling self-exciting behavior, in which the occurrence of one event increases the probability of other events occurring. These processes are able to handle complex interrelationships between stochastic and deterministic components of spatio-temporal phenomena. However, despite its widespread use in practice, there is no common and unified formalism and every paper proposes different views of these stochastic mechanisms. With this in mind, we implement two simulation techniques and three unified, self-consistent inference techniques, which are widely used in the practical modeling of spatio-temporal Hawkes processes. Furthermore, we provide an evaluation of the practical performance of these methods, while providing useful code for reproducibility.

---

## 99. Hybrid Modeling of Photoplethysmography for Non-invasive Monitoring of Cardiovascular Parameters

**论文链接:** [http://arxiv.org/abs/2511.14452v1](http://arxiv.org/abs/2511.14452v1)

**作者:** Emanuele Palumbo, Sorawit Saengkyongam, Maria R. Cervera, Jens Behrmann, Andrew C. Miller, Guillermo Sapiro, Christina Heinze-Deml, Antoine Wehenkel

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了一种混合方法，通过结合血液动力学模拟和未标记的临床数据，直接从非侵入性的光电容积描记信号估计关键心血管生物标志物，如每搏输出量和心输出量。实验证明该方法能有效检测这些生物标志物的波动，并在监测时间变化方面优于传统监督方法。

### 背景

连续心血管监测在精准健康中具有重要作用，但关键心脏生物标志物需要侵入性测量，如动脉压力波形。虽然光电容积描记作为非侵入性替代方法在医院常规收集，但从PPG预测关键心脏生物标志物仍是一个开放性挑战，且标注的PPG测量数据稀缺。

### 目的

开发一种能够从非侵入性的光电容积描记信号中准确估计关键心血管生物标志物的方法，避免侵入性测量的需要，同时克服标注数据稀缺的问题。

### 方法

研究提出了一种混合方法，结合血液动力学模拟和未标记的临床数据。该方法包括：(1)一个在配对PPG-APW数据上训练的条件变分自编码器；(2)一个在标记的模拟APW段上训练的心脏生物标志物条件密度估计器。

### 主要发现

实验结果表明，该方法能够有效检测心输出量和每搏输出量的波动，并且在监测这些生物标志物的时间变化方面优于监督基线方法。

### 结论

该混合方法为从非侵入性光电容积描记信号估计关键心血管生物标志物提供了有效途径，避免了侵入性测量的需要，同时解决了标注数据稀缺的问题，在精准健康监测领域具有潜在应用价值。

### 翻译

连续心血管监测可以在精准健康中发挥关键作用。然而，一些感兴趣的基本心脏生物标志物，包括每搏输出量和心输出量，需要侵入性测量，例如动脉压力波形。作为非侵入性替代方案，光电容积描记测量在医院环境中常规收集。不幸的是，从PPG而非动脉压力波形预测关键心脏生物标志物仍然是一个开放的挑战，进一步受到标注光电容积描记测量数据稀缺的困扰。作为解决方案，我们提出了一种混合方法，使用血液动力学模拟和未标记的临床数据直接从光电容积描记信号估计心血管生物标志物。我们的混合模型结合了一个在配对光电容积描记-动脉压力波形数据上训练的条件变分自编码器，和一个在标记的模拟动脉压力波形段上训练的心脏生物标志物条件密度估计器。作为关键结果，我们的实验证明，所提出的方法能够检测心输出量和每搏输出量的波动，并且在监测这些生物标志物的时间变化方面优于监督基线。


### 论文摘要

Continuous cardiovascular monitoring can play a key role in precision health. However, some fundamental cardiac biomarkers of interest, including stroke volume and cardiac output, require invasive measurements, e.g., arterial pressure waveforms (APW). As a non-invasive alternative, photoplethysmography (PPG) measurements are routinely collected in hospital settings. Unfortunately, the prediction of key cardiac biomarkers from PPG instead of APW remains an open challenge, further complicated by the scarcity of annotated PPG measurements. As a solution, we propose a hybrid approach that uses hemodynamic simulations and unlabeled clinical data to estimate cardiovascular biomarkers directly from PPG signals. Our hybrid model combines a conditional variational autoencoder trained on paired PPG-APW data with a conditional density estimator of cardiac biomarkers trained on labeled simulated APW segments. As a key result, our experiments demonstrate that the proposed approach can detect fluctuations of cardiac output and stroke volume and outperform a supervised baseline in monitoring temporal changes in these biomarkers.

---

## 100. Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding

**论文链接:** [http://arxiv.org/abs/2511.14446v1](http://arxiv.org/abs/2511.14446v1)

**作者:** Hong Gao, Yiming Bao, Xuezhen Tu, Yutong Xu, Yue Jin, Yiyang Mu, Bin Zhong, Linan Yue, Min-Ling Zhang

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了Agentic Video Intelligence (AVI)框架，一个灵活且无需训练的系统，通过系统级设计和优化模拟人类视频理解能力，解决了现有视频理解方法的局限性。

### 背景

视频理解不仅需要视觉识别还需要复杂推理，现有的Vision-Language Models主要采用单次处理方式，不支持证据回顾和迭代改进；而基于代理的方法虽然支持长期推理，但依赖昂贵的专有模型或需要大量的代理RL训练。

### 目的

克服现有视频理解方法的局限性，提出一个灵活且无需训练的框架，能够模拟人类视频理解过程，实现更好的视频理解性能和可解释性。

### 方法

AVI引入三个关键创新：1) 受人类启发的三阶段推理过程（检索-感知-回顾）；2) 通过实体图组织的结构化视频知识库和多粒度集成工具；3) 开源模型集成，结合推理LLMs与轻量级基础CV模型和VLM。

### 主要发现

在LVBench、VideoMME-Long、LongVideoBench和Charades-STA上的实验表明，AVI实现了具有竞争力的性能，同时提供了更好的可解释性。

### 结论

AVI是一个灵活且无需训练的框架，通过系统级设计和优化能够有效模拟人类视频理解过程，解决了现有方法在证据回顾、迭代改进和依赖专有模型方面的局限性。

### 翻译

视频理解不仅需要视觉识别，还需要复杂推理。虽然Vision-Language Models展示了令人印象深刻的能力，但它们通常以单次处理方式处理视频，对证据回顾和迭代改进的支持有限。最近出现的基于代理的方法虽然能够进行长期推理，但它们要么严重依赖昂贵的专有模型，要么需要大量的代理RL训练。为了克服这些局限性，我们提出了Agentic Video Intelligence (AVI)，这是一个灵活且无需训练的框架，能够通过系统级设计和优化模拟人类视频理解能力。AVI引入了三个关键创新：(1) 受人类启发的三阶段推理过程（检索-感知-回顾），确保充分的全球探索和专注的局部分析；(2) 通过实体图组织的结构化视频知识库，以及多粒度集成工具，构成代理的交互环境；(3) 开源模型集成，结合推理LLMs与轻量级基础CV模型和VLM，消除对专有API或RL训练的依赖。在LVBench、VideoMME-Long、LongVideoBench和Charades-STA上的实验表明，AVI实现了具有竞争力的性能，同时提供了更好的可解释性。


### 论文摘要

Video understanding requires not only visual recognition but also complex reasoning. While Vision-Language Models (VLMs) demonstrate impressive capabilities, they typically process videos largely in a single-pass manner with limited support for evidence revisit and iterative refinement. While recently emerging agent-based methods enable long-horizon reasoning, they either depend heavily on expensive proprietary models or require extensive agentic RL training. To overcome these limitations, we propose Agentic Video Intelligence (AVI), a flexible and training-free framework that can mirror human video comprehension through system-level design and optimization. AVI introduces three key innovations: (1) a human-inspired three-phase reasoning process (Retrieve-Perceive-Review) that ensures both sufficient global exploration and focused local analysis, (2) a structured video knowledge base organized through entity graphs, along with multi-granularity integrated tools, constituting the agent's interaction environment, and (3) an open-source model ensemble combining reasoning LLMs with lightweight base CV models and VLM, eliminating dependence on proprietary APIs or RL training. Experiments on LVBench, VideoMME-Long, LongVideoBench, and Charades-STA demonstrate that AVI achieves competitive performance while offering superior interpretability.

---

## 101. Learning to See Through a Baby's Eyes: Early Visual Diets Enable Robust Visual Intelligence in Humans and Machines

**论文链接:** [http://arxiv.org/abs/2511.14440v1](http://arxiv.org/abs/2511.14440v1)

**作者:** Yusen Cai, Bhargava Satya Nunna, Qing Lin, Mengmi Zhang

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究婴儿视觉发展模式并提出CATDiet和CombDiet两种自监督学习方法，模拟婴儿视觉发展的不同阶段，提高了模型的鲁棒性和性能，为理解机器视觉智能提供了新框架。

### 背景

新生儿以低清晰度、颜色退化和时间连续的视觉感知世界，随着发育逐渐提高。这种分阶段的'视觉饮食'具有生态优势。

### 目的

探索婴儿视觉发展分阶段的生态优势，并将这些模式应用于自监督学习模型，以提升视觉智能系统的鲁棒性和性能。

### 方法

提出CATDiet方法，在物体为中心的视频上训练自监督学习模型，模拟婴儿视觉约束：从灰度到颜色(C)、从模糊到清晰(A)、保持时间连续性(T)。建立十个数据集的全面基准测试。提出CombDiet方法，先用CATDiet初始化，再进行标准训练。

### 主要发现

所有CATDiet变体在物体识别方面表现出增强的鲁棒性；模型展现出与生物学对齐的发展模式，包括类似于猕猴V1中突触密度的神经可塑性变化；CombDiet在物体识别和深度感知任务上优于标准自监督学习。

### 结论

婴儿早期视觉体验的发展过程为理解机器视觉智能的出现提供了强大的逆向工程框架，所有代码、数据和模型将公开。

### 翻译

新生儿以低清晰度、颜色退化和时间连续的视觉感知世界，随着婴儿发育，视觉能力逐渐提高。为了探索这种分阶段'视觉饮食'的生态优势，我们在物体为中心的视频上训练自监督学习模型，并施加模拟婴儿视觉的约束条件：从灰度到颜色(C)、从模糊到清晰(A)，以及保持时间连续性(T)—统称为CATDiet。为了评估，我们在十个数据集上建立了全面的基准测试，涵盖干净和损坏的图像识别、纹理-形状线索冲突测试、轮廓识别、深度顺序分类和视觉悬崖范式。所有CATDiet变体在物体识别方面表现出增强的鲁棒性，尽管仅使用物体为中心的视频进行训练。值得注意的是，模型还表现出与生物学对齐的发展模式，包括类似于猕猴V1中突触密度的神经可塑性变化，以及类似于婴儿视觉悬崖反应的行为。基于这些见解，CombDiet在标准训练前使用CATDiet初始化自监督学习，同时保持时间连续性。在物体为中心或婴儿头戴式视频上训练后，CombDiet在领域内和领域外的物体识别和深度感知任务上都优于标准自监督学习。总之，这些结果表明，早期婴儿视觉体验的发展过程为理解机器视觉智能的出现提供了强大的逆向工程框架。所有代码、数据和模型都将公开。


### 论文摘要

Newborns perceive the world with low-acuity, color-degraded, and temporally continuous vision, which gradually sharpens as infants develop. To explore the ecological advantages of such staged "visual diets", we train self-supervised learning (SSL) models on object-centric videos under constraints that simulate infant vision: grayscale-to-color (C), blur-to-sharp (A), and preserved temporal continuity (T)-collectively termed CATDiet. For evaluation, we establish a comprehensive benchmark across ten datasets, covering clean and corrupted image recognition, texture-shape cue conflict tests, silhouette recognition, depth-order classification, and the visual cliff paradigm. All CATDiet variants demonstrate enhanced robustness in object recognition, despite being trained solely on object-centric videos. Remarkably, models also exhibit biologically aligned developmental patterns, including neural plasticity changes mirroring synaptic density in macaque V1 and behaviors resembling infants' visual cliff responses. Building on these insights, CombDiet initializes SSL with CATDiet before standard training while preserving temporal continuity. Trained on object-centric or head-mounted infant videos, CombDiet outperforms standard SSL on both in-domain and out-of-domain object recognition and depth perception. Together, these results suggest that the developmental progression of early infant visual experience offers a powerful reverse-engineering framework for understanding the emergence of robust visual intelligence in machines. All code, data, and models will be publicly released.

---

## 102. ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

**论文链接:** [http://arxiv.org/abs/2511.14349v1](http://arxiv.org/abs/2511.14349v1)

**作者:** Junfu Pu, Teng Wang, Yixiao Ge, Yuying Ge, Chen Li, Ying Shan

**发布时间:** 2025-11-18

**备注:** Project Page: https://arcchapter.github.io/index_en.html

### GPT解析

### 总结

ARC-Chapter是首个在百万级长视频章节上训练的大规模视频章节化模型，具有双语、时间定位和分层章节注释的特点，显著提升了视频内容结构化的性能。

### 背景

长视频（如讲座、播客、纪录片）数量激增，对高效内容结构化的需求增加，但现有方法受限于小规模训练和简短粗糙的注释，难以泛化到长视频中的细微转换。

### 目的

提出首个在百万级长视频章节上训练的大规模视频章节化模型ARC-Chapter，解决长视频内容结构化的挑战。

### 方法

通过结构化流程整理双语英中章节数据集，统一ASR转录、场景文本、视觉标题为多级注释，并设计新的评估指标GRACE，包含多对一段落重叠和语义相似性。

### 主要发现

ARC-Chapter显著优于之前最佳方法，F1分数提高14.0%，SODA分数提高11.3%，在下游任务（如YouCook2上的密集视频字幕）上表现出良好的可转移性。

### 结论

ARC-Chapter建立了新的最先进水平，大幅提升了视频章节化性能，为长视频内容结构化提供了有效解决方案。

### 翻译

随着一小时长视频（如讲座、播客、纪录片）的激增，对高效内容结构化的需求日益增强。然而，现有方法受限于小规模训练和通常简短粗糙的注释，限制了在长视频中细微转换的泛化能力。我们引入了ARC-Chapter，这是首个在百万级长视频章节上训练的大规模视频章节化模型，具有双语、时间定位和分层章节注释的特点。为实现这一目标，我们通过一个结构化流程整理了双语英中章节数据集，该流程将ASR转录、场景文本和视觉标题统一为多级注释，从简短标题到长摘要。我们证明了数据规模和标签强度的扩展能带来明显的性能提升。此外，我们设计了一个名为GRACE的新评估指标，它包含多对一段落重叠和语义相似性，更好地反映了真实世界章节化的灵活性。大量实验表明，ARC-Chapter以显著优势建立了新的最先进水平，在F1分数上比之前最佳方法提高了14.0%，在SODA分数上提高了11.3%。此外，ARC-Chapter表现出优秀的可转移性，在YouCook2上的密集视频字幕等下游任务上提高了最先进水平。


### 论文摘要

The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring. However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos. We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations. To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries. We demonstrate clear performance improvements with data scaling, both in data volume and label intensity. Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility. Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by 14.0% in F1 score and 11.3% in SODA score. Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.

---

## 103. Model-Based Clustering of Football Event Sequences: A Marked Spatio-Temporal Point Process Mixture Approach

**论文链接:** [http://arxiv.org/abs/2511.14297v1](http://arxiv.org/abs/2511.14297v1)

**作者:** Koffi Amezouwui, Brigitte Gelein, Matthieu Marbac, Anthony Sorel

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种新的足球事件数据混合模型，通过聚类整个控球过程来揭示其时间、序列和空间结构。

### 背景

现有足球数据分析方法通常关注单个事件，缺乏对整个控球过程的系统分析。

### 目的

开发一种能够分析足球比赛中控球事件数据并揭示其时间、序列和空间特征的混合模型。

### 方法

提出一种混合模型，将控球建模为标记时空点过程：事件类型遵循具有吸收状态的有限马尔可夫链，事件时间遵循条件Gamma过程，空间位置通过截断布朗运动演变；从模型参数中提取控球速度、事件数量和空间动态等指标；通过广义期望最大化算法进行参数估计。

### 主要发现

应用在38场法甲联赛(2020/2021)的StatsBomb数据上，该方法揭示了圣埃蒂安面对的不同防守控球模式。

### 结论

与以往关注单个事件的方法不同，这种混合结构能够对完整控球进行有原则的聚类，支持战术分析和逼真虚拟训练环境的未来发展。

### 翻译

我们提出了一种用于足球事件数据的新型混合模型，该模型将整个控球过程聚类以揭示其时间、序列和空间结构。每个混合组件将控球建模为标记时空点过程：事件类型遵循具有丢球吸收状态的有限马尔可夫链，事件时间遵循考虑离散性的条件Gamma过程，空间位置通过截断布朗运动演变。为了辅助解释，我们从模型参数中派生总结指标，捕捉控球速度、事件数量和空间动态。参数通过广义期望最大化算法通过最大似然进行估计。应用于38场法甲比赛(2020/2021)的StatsBomb数据，我们的方法揭示了圣埃蒂安所面临的不同防守控球模式。与以往关注单个事件的方法不同，我们的混合结构能够对完整控球进行有原则的聚类，支持战术分析和未来逼真虚拟训练环境的发展。


### 论文摘要

We propose a novel mixture model for football event data that clusters entire possessions to reveal their temporal, sequential, and spatial structure. Each mixture component models possessions as marked spatio-temporal point processes: event types follow a finite Markov chain with an absorbing state for ball loss, event times follow a conditional Gamma process to account for dispersion, and spatial locations evolve via truncated Brownian motion. To aid interpretation, we derive summary indicators from model parameters capturing possession speed, number of events, and spatial dynamics. Parameters are estimated through maximum likelihood via Generalized Expectation-Maximization algorithm. Applied to StatsBomb data from 38 Ligue 1 matches (2020/2021), our approach uncovers distinct defensive possession patterns faced by Stade Rennais. Unlike previous approaches focusing on individual events, our mixture structure enables principled clustering of full possessions, supporting tactical analysis and the future development of realistic virtual training environments.

---

## 104. InstantViR: Real-Time Video Inverse Problem Solver with Distilled Diffusion Prior

**论文链接:** [http://arxiv.org/abs/2511.14208v1](http://arxiv.org/abs/2511.14208v1)

**作者:** Weimin Bai, Suzhe Xu, Yiwei Ren, Jinhua Hao, Ming Sun, Wenzheng Chen, He Sun

**发布时间:** 2025-11-18

### GPT解析

### 总结

InstantViR是一种基于预训练视频扩散先验的超快速视频重建框架，通过将双向视频扩散模型提炼为因果自回归学生模型，实现单次前向传递直接恢复视频，在保持高质量的同时大幅提升处理速度。

### 背景

视频逆流问题是流媒体、远程呈现和AR/VR的基础，需要高感知质量与严格延迟并存。现有基于扩散的方法要么存在临时伪影，要么迭代采样速度过慢，无法满足实时需求。

### 目的

开发一种既能保持高质量视频重建性能，又能满足实时处理需求的视频逆问题解决方案。

### 方法

将强大的双向视频扩散模型(教师)提炼为因果自回归学生模型，在单次前向传递中直接映射退化视频到恢复版本；使用先验驱动提炼方法，仅需教师模型和已知退化算子；用高效LeanVAE替换原始VAE，实现低延迟潜在空间处理。

### 主要发现

InstantViR在流媒体随机修复、高斯去模糊和超分辨率任务中匹配或超越基线质量，在NVIDIA A100 GPU上运行速度超过35 FPS，比迭代视频扩散求解器快达100倍。

### 结论

基于扩散的视频重建可与实时、交互式、可编辑、流媒体场景兼容，使高质量视频恢复成为现代视觉系统的实用组件。

### 翻译

视频逆流问题是流媒体、远程呈现和AR/VR的基础，其中高感知质量必须与严格的延迟限制并存。基于扩散的先验目前提供最先进的重建，但现有方法要么使用临时正则化器调整图像扩散模型(导致临时伪影)，要么依赖原生视频扩散模型，其迭代后验采样对于实时使用来说太慢。我们引入InstantViR，一个由预训练视频扩散先验驱动的超快速视频重建的推理框架。我们将强大的双向视频扩散模型(教师)提炼为因果自回归学生，该模型在单次前向传递中将退化视频直接映射到其恢复版本，继承教师的强时序建模能力，同时完全消除迭代测试时间优化。这种提炼是先验驱动的：它只需要教师扩散模型和已知退化算子，不依赖外部配对的清洁/噪声视频数据。为提高吞吐量，我们通过创新的教师空间正则化提炼方案，用高效LeanVAE替换视频扩散骨干VAE，实现低延迟潜在空间处理。在流媒体随机修复、高斯去模糊和超分辨率中，InstantViR匹配或超越基于扩散的基线重建质量，同时在NVIDIA A100 GPU上以超过35 FPS运行，比迭代视频扩散求解器加速达100倍。这些结果表明，基于扩散的视频重建与实时、交互式、可编辑、流媒体场景兼容，使高质量视频恢复成为现代视觉系统的实用组件。


### 论文摘要

Video inverse problems are fundamental to streaming, telepresence, and AR/VR, where high perceptual quality must coexist with tight latency constraints. Diffusion-based priors currently deliver state-of-the-art reconstructions, but existing approaches either adapt image diffusion models with ad hoc temporal regularizers - leading to temporal artifacts - or rely on native video diffusion models whose iterative posterior sampling is far too slow for real-time use. We introduce InstantViR, an amortized inference framework for ultra-fast video reconstruction powered by a pre-trained video diffusion prior. We distill a powerful bidirectional video diffusion model (teacher) into a causal autoregressive student that maps a degraded video directly to its restored version in a single forward pass, inheriting the teacher's strong temporal modeling while completely removing iterative test-time optimization. The distillation is prior-driven: it only requires the teacher diffusion model and known degradation operators, and does not rely on externally paired clean/noisy video data. To further boost throughput, we replace the video-diffusion backbone VAE with a high-efficiency LeanVAE via an innovative teacher-space regularized distillation scheme, enabling low-latency latent-space processing. Across streaming random inpainting, Gaussian deblurring and super-resolution, InstantViR matches or surpasses the reconstruction quality of diffusion-based baselines while running at over 35 FPS on NVIDIA A100 GPUs, achieving up to 100 times speedups over iterative video diffusion solvers. These results show that diffusion-based video reconstruction is compatible with real-time, interactive, editable, streaming scenarios, turning high-quality video restoration into a practical component of modern vision systems.

---

## 105. Experimental realization of a full-band wave antireflection based on temporal taper metamaterials

**论文链接:** [http://arxiv.org/abs/2511.14190v1](http://arxiv.org/abs/2511.14190v1)

**作者:** Haonan Hou, Kai Peng, Yangkai Wang, Jiarui Wang, Xudong Zhang, Ren Wang, Hao Hu, Jiang Xiong

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究实验实现了'时间锥'概念，这是一种具有全频带抗反射特性的时间超材料技术，解决了现有时间抗反射模型在高频处的反射问题，为未来光子系统提供了新的阻抗匹配方法。

### 背景

时间超材料作为额外自由度的引入为波控制和操控开辟了新途径。基于时间超材料的抗反射涂层作为一种创新方法，避免了额外的空间插入。然而，现有基于破坏性干涉机制的时间抗反射模型在高频处存在周期性强反射，限制了可实现带宽。

### 目的

提出并实验实现'时间锥'概念，解决现有时间抗反射模型在高频处的反射问题，实现宽带抗反射特性。

### 方法

设计并实验验证了一种基于电压控制变容器的1D时间超材料，实现了时间锥的概念。

### 主要发现

时间锥具有近乎全频带的抗反射特性，与渐变时变组件具有良好的兼容性，能够使系统免于空间匹配插入，并实现各种终端负载的敏捷阻抗匹配。

### 结论

基于时间锥的宽带抗反射技术是一种有前景的方法，适用于未来光子系统，能够解决现有时间抗反射模型的带宽限制问题。

### 翻译

时间可以作为额外的自由度引入，因此时间超材料如今为波控制和操控开辟了新途径。在这些进步中，基于时间超材料的抗反射涂层最近出现了一种创新方法，本质上避免了额外的空间插入。然而，先前具有有限插入时间过渡部分的时间抗反射模型依赖于破坏性干涉机制，在高频处表现出残留的周期性强反射，从根本上限制了可实现的带宽。在这项工作中，'时间锥'的概念，即具有近乎全频带抗反射特性和与渐变时变组件良好兼容性的传统空间锥的时间对应物，已被实验实现。基于电压控制变容器的1D时间超材料已进行设计并实验验证。基于时间锥的宽带抗反射使系统免于空间匹配插入，并能够实现各种终端负载的敏捷阻抗匹配，使其成为未来光子系统中有前景的方法。


### 论文摘要

As time can be introduced as an additional degree of freedom, temporal metamaterials nowadays open up new avenues for wave control and manipulation. Among these advancements, temporal metamaterial-based antireflection coatings have recently emerged as an innovative method that inherently avoids additional spatial insertions. However, prior temporal antireflection models with finite inserted temporal transition sections that rely on the destructive interference mechanism exhibit residual periodic strong reflections at high frequencies, fundamentally limiting the achievable bandwidth. In this work, the concept of "temporal taper", the temporal counterpart of a conventional spatial taper with a nearly full-band antireflection feature and good compatibility with gradual time-varying components, has been experimentally realized. A 1D temporal metamaterial base on voltage-controlled varactors has been designed experimentally validated. The temporal taper based broadband antireflection exempts the system from spatial matching insertions, and enables agile impedance matching for various terminal loads, positioning it as a promising approach in future photonic systems.

---

## 106. Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation

**论文链接:** [http://arxiv.org/abs/2511.14186v1](http://arxiv.org/abs/2511.14186v1)

**作者:** Zhaoyu Liu, Kan Jiang, Murong Ma, Zhe Hou, Yun Lin, Jin Song Dong

**发布时间:** 2025-11-18

**备注:** The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)

### GPT解析

### 总结

本文提出了统一多实体图网络(UMEG-Net)用于少样本精确事件检测，通过整合人体骨架和运动物体关键点，结合高级GCN和多尺度时间移位模块，以及多模态蒸馏技术，在有限标注数据条件下实现了强大的性能。

### 背景

精确事件检测(PES)是体育分析的关键组成部分，但因快速连续、运动模糊和细微视觉差异而极具挑战性。现有方法依赖大型标记数据集和领域特定训练，在少样本条件下表现不佳，而获取大型数据集在实际中非常困难。

### 目的

开发一个能够有效处理少样本条件下精确事件检测的统一多实体图网络，解决现有方法在数据有限情况下的局限性。

### 方法

UMEG-Net将人体骨架和运动特定物体关键点整合到统一图中，采用基于高级GCN和多尺度时间移位的高效时空提取模块，并通过多模态蒸馏将基于关键点图的知识转移到视觉表示中。

### 主要发现

该方法在有限标注数据条件下实现了强大性能，在少样本设置中显著优于基线模型，为少样本PES提供了可扩展且有效的解决方案。

### 结论

UMEG-Net为少样本精确事件检测提供了一个可扩展且有效的解决方案，克服了现有方法在数据有限条件下的局限性，代码已公开可用。

### 翻译

精确事件检测(PES)旨在识别精确时刻的细粒度事件，已成为体育分析的关键组成部分。由于事件快速连续、运动模糊和细微视觉差异，这一任务特别具有挑战性。因此，大多数现有方法依赖领域特定的端到端训练和大型标记数据集，并且常常由于仅依赖像素或姿态输入而在少样本条件下表现不佳。然而，获取大型标记数据集在实际中非常困难。我们提出了一种用于少样本PES的统一多实体图网络(UMEG-Net)。UMEG-Net将人体骨架和运动特定物体关键点整合到统一图中，并具有基于高级GCN和多尺度时间移位的高效时空提取模块。为进一步提高性能，我们采用多模态蒸馏将基于关键点图的知识转移到视觉表示中。我们的方法在有限标注数据条件下实现了强大的性能，在少样本设置中显著优于基线模型，为少样本PES提供了可扩展且有效的解决方案。代码已在https://github.com/LZYAndy/UMEG-Net公开可用。


### 论文摘要

Precise event spotting (PES) aims to recognize fine-grained events at exact moments and has become a key component of sports analytics. This task is particularly challenging due to rapid succession, motion blur, and subtle visual differences. Consequently, most existing methods rely on domain-specific, end-to-end training with large labeled datasets and often struggle in few-shot conditions due to their dependence on pixel- or pose-based inputs alone. However, obtaining large labeled datasets is practically hard. We propose a Unified Multi-Entity Graph Network (UMEG-Net) for few-shot PES. UMEG-Net integrates human skeletons and sport-specific object keypoints into a unified graph and features an efficient spatio-temporal extraction module based on advanced GCN and multi-scale temporal shift. To further enhance performance, we employ multimodal distillation to transfer knowledge from keypoint-based graphs to visual representations. Our approach achieves robust performance with limited labeled data and significantly outperforms baseline models in few-shot settings, providing a scalable and effective solution for few-shot PES. Code is publicly available at https://github.com/LZYAndy/UMEG-Net.

---

## 107. DoGCLR: Dominance-Game Contrastive Learning Network for Skeleton-Based Action Recognition

**论文链接:** [http://arxiv.org/abs/2511.14179v1](http://arxiv.org/abs/2511.14179v1)

**作者:** Yanshan Li, Ke Ma, Miaomiao Wei, Linhui Dai

**发布时间:** 2025-11-18

**备注:** 14 pages, 7 figures, journal

### GPT解析

### 总结

本文提出了一种基于博弈论的自监督对比学习框架DoGCLR，用于解决现有骨架动作识别方法中的运动信息丢失和非最优负样本选择问题。该框架通过时空双权重定位机制和基于熵的主导策略，显著提升了动作识别的准确率。

### 背景

现有的基于骨架的自监督对比学习方法通常统一处理所有骨架区域，并采用先进先出(FIFO)队列存储负样本，导致运动信息丢失和非最优负样本选择。

### 目的

解决现有方法中的运动信息丢失和非最优负样本选择问题，提高骨架动作识别的准确性和鲁棒性。

### 方法

提出DoGCLR(Dominance-Game Contrastive Learning network)框架，将正负样本构建建模为动态主导博弈；采用时空双权重定位机制识别关键运动区域并引导区域增强；使用基于熵的主导策略管理内存库，保留高熵负样本，替换低熵负样本。

### 主要发现

在NTU RGB+D 60 X-Sub/X-View上达到81.1%/89.4%准确率，在NTU RGB+D 120 X-Sub/X-Set上达到71.2%/75.5%准确率，分别超越最先进方法0.1%、2.7%、1.1%和2.3%；在PKU-MMD Part II上实现1.9%的准确率提升，展现强鲁棒性。

### 结论

DoGCLR通过博弈论方法实现了语义保持和判别强度的平衡，在多个数据集上取得最先进性能，特别是在更具挑战性场景中表现优异。

### 翻译

现有的基于骨架的自监督对比学习方法通常统一处理所有骨架区域，并采用先进先出(FIFO)队列存储负样本，这导致运动信息丢失和非最优负样本选择。为解决这些挑战，本文提出了基于骨架动作识别的主导博弈对比学习网络(DoGCLR)，一种基于博弈论的自监督框架。DoGCLR将正负样本的构建建模为动态主导博弈，其中两种样本类型相互作用以平衡语义保持和判别强度。具体而言，时空双权重定位机制识别关键运动区域并引导区域增强，同时增强运动多样性和保持语义。并行地，基于熵的主导策略通过保留高熵(困难)负样本和替换低熵(弱)负样本来管理内存库，确保持续接触有信息的对比信号。在NTU RGB+D和PKU-MMD数据集上进行了大量实验。在NTU RGB+D 60 X-Sub/X-View上，DoGCLR分别达到81.1%/89.4%的准确率，在NTU RGB+D 120 X-Sub/X-Set上，DoGCLR分别达到71.2%/75.5%的准确率，分别超越了最先进方法0.1%、2.7%、1.1%和2.3%。在PKU-MMD Part I/Part II上，DoGCLR的表现与最先进方法相当，并在Part II上实现了1.9%的更高准确率，突显了其在更具挑战性场景下的强大鲁棒性。


### 论文摘要

Existing self-supervised contrastive learning methods for skeleton-based action recognition often process all skeleton regions uniformly, and adopt a first-in-first-out (FIFO) queue to store negative samples, which leads to motion information loss and non-optimal negative sample selection. To address these challenges, this paper proposes Dominance-Game Contrastive Learning network for skeleton-based action Recognition (DoGCLR), a self-supervised framework based on game theory. DoGCLR models the construction of positive and negative samples as a dynamic Dominance Game, where both sample types interact to reach an equilibrium that balances semantic preservation and discriminative strength. Specifically, a spatio-temporal dual weight localization mechanism identifies key motion regions and guides region-wise augmentations to enhance motion diversity while maintaining semantics. In parallel, an entropy-driven dominance strategy manages the memory bank by retaining high entropy (hard) negatives and replacing low-entropy (weak) ones, ensuring consistent exposure to informative contrastive signals. Extensive experiments are conducted on NTU RGB+D and PKU-MMD datasets. On NTU RGB+D 60 X-Sub/X-View, DoGCLR achieves 81.1%/89.4% accuracy, and on NTU RGB+D 120 X-Sub/X-Set, DoGCLR achieves 71.2%/75.5% accuracy, surpassing state-of-the-art methods by 0.1%, 2.7%, 1.1%, and 2.3%, respectively. On PKU-MMD Part I/Part II, DoGCLR performs comparably to the state-of-the-art methods and achieves a 1.9% higher accuracy on Part II, highlighting its strong robustness on more challenging scenarios.

---

## 108. AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2511.14148v1](http://arxiv.org/abs/2511.14148v1)

**作者:** Yuhua Jiang, Shuang Cheng, Yan Ding, Feifei Gao, Biqing Qi

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为AsyncVLA的新型视觉-语言-动作模型，通过异步流匹配技术引入时间灵活性，实现动作生成中的自我纠正能力。该模型在机器人操作基准测试中表现出色，数据效率高，在一般具身评估中达到最先进性能。

### 背景

视觉-语言-动作模型已成为构建通用机器人的强大范式。然而，传统通过流匹配生成动作的VLA模型通常依赖同步流匹配，缺乏动作上下文感知和异步自我纠正能力，在长时程任务中不稳定，单个动作错误可能导致级联失败。

### 目的

解决传统VLA模型在长时程任务中的不稳定性问题，通过引入时间灵活性和自我纠正能力，提高模型在复杂任务中的表现。

### 方法

提出异步流匹配VLA框架：在异步流匹配中引入时间灵活性；通过动作上下文感知以非均匀时间表生成动作标记；引入置信度评估器使模型在执行前选择性地优化不准确的动作标记；提出同步流匹配和异步流匹配的统一训练程序，提高KV缓存利用率。

### 主要发现

在机器人操作基准测试上，AsyncVLA展现出数据高效性和自我纠正能力；由于异步生成特性，在一般具身评估中取得了最先进的结果。

### 结论

AsyncVLA通过异步流匹配和自我纠正机制显著提高了VLA模型在长时程任务中的稳定性和性能，为构建更强大的通用机器人提供了新解决方案。

### 翻译

视觉-语言-动作模型最近已成为构建通用机器人的强大范式。然而，传统通过流匹配生成动作的VLA模型通常依赖于刚性和统一的时间调度，即同步流匹配。由于缺乏动作上下文感知和异步自我纠正能力，同步流匹配在长时程任务中变得不稳定，因为单个动作错误可能会级联导致失败。在这项工作中，我们提出了异步流匹配VLA，这是一个新颖的框架，它在异步流匹配中引入时间灵活性，并实现了动作生成中的自我纠正。AsyncVLA突破了VLA模型中原始同步流匹配的限制，通过以非均匀时间表和动作上下文感知生成动作标记。此外，我们的方法引入了置信度评估器来提取初始生成动作的置信度，使模型能够在执行前选择性地优化不准确的动作标记。我们提出了同步流匹配和异步流匹配的统一训练程序，使单一模型具备两种模式，提高了KV缓存利用率。在机器人操作基准测试上的大量实验表明，AsyncVLA是数据高效的，并展现出自我纠正能力。由于异步生成特性，AsyncVLA在一般具身评估中取得了最先进的结果。


### 论文摘要

Vision-language-action (VLA) models have recently emerged as a powerful paradigm for building generalist robots. However, traditional VLA models that generate actions through flow matching (FM) typically rely on rigid and uniform time schedules, i.e., synchronous FM (SFM). Without action context awareness and asynchronous self-correction, SFM becomes unstable in long-horizon tasks, where a single action error can cascade into failure. In this work, we propose asynchronous flow matching VLA (AsyncVLA), a novel framework that introduces temporal flexibility in asynchronous FM (AFM) and enables self-correction in action generation. AsyncVLA breaks from the vanilla SFM in VLA models by generating the action tokens in a non-uniform time schedule with action context awareness. Besides, our method introduces the confidence rater to extract confidence of the initially generated actions, enabling the model to selectively refine inaccurate action tokens before execution. Moreover, we propose a unified training procedure for SFM and AFM that endows a single model with both modes, improving KV-cache utilization. Extensive experiments on robotic manipulation benchmarks demonstrate that AsyncVLA is data-efficient and exhibits self-correction ability. AsyncVLA achieves state-of-the-art results across general embodied evaluations due to its asynchronous generation in AFM. Our code is available at https://github.com/YuhuaJiang2002/AsyncVLA.

---

## 109. SMART: Shot-Aware Multimodal Video Moment Retrieval with Audio-Enhanced MLLM

**论文链接:** [http://arxiv.org/abs/2511.14143v1](http://arxiv.org/abs/2511.14143v1)

**作者:** An Yu, Weiheng Lu, Jian Li, Zhenfei Zhang, Yunhang Shen, Felix X. -F. Ye, Ming-Ching Chang

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了SMART框架，通过整合音频线索和镜头级别时间结构，改进了视频时刻检索任务，在基准测试上取得了显著性能提升。

### 背景

视频时刻检索是视频理解中的任务，旨在基于自然语言查询在未修剪视频中定位特定时间段。现有方法主要依赖粗糙时间理解和单一视觉模态，限制了在复杂视频上的性能。

### 目的

解决现有视频时刻检索方法在复杂视频上性能受限的问题，通过整合多模态信息提高检索准确性。

### 方法

提出SMART框架，一个基于多模态大语言模型的系统，整合音频线索和镜头级别时间结构，结合音频和视觉特征丰富多模态表示，应用镜头感知的令牌压缩减少冗余并保留细粒度时间细节，并改进提示设计以更好地利用视听线索。

### 主要发现

在Charades-STA和QVHighlights基准测试上，SMART相比最先进方法取得显著改进，包括R1@0.5提高1.61%和R1@0.7提高2.59%。

### 结论

通过整合音频信息和改进的时间结构理解，SMART框架有效提升了视频时刻检索的准确性和性能。

### 翻译

视频时刻检索是视频理解中的一个任务，旨在基于自然语言查询在未修剪的视频中定位特定的时间段。尽管最近使用传统技术和多模态大语言模型在视频时刻检索方面取得了进展，但大多数现有方法仍然依赖于粗糙的时间理解和单一的视觉模态，这限制了它们在复杂视频上的性能。为此，我们引入了SMART(镜头感知的多模态音频增强时间段检索)，这是一个基于MLLM的框架，它整合了音频线索并利用了镜头级别的时间结构。SMART通过结合音频和视觉特征来丰富多模态表示，同时应用镜头感知的令牌压缩，该方法在每个镜头中选择性地保留高信息令牌，以减少冗余并保留细粒度的时间细节。我们还改进了提示设计，以更好地利用视听线索。在Charades-STA和QVHighlights上的评估显示，SMART相比于最先进的方法取得了显著改进，包括在Charades-STA上R1@0.5提高了1.61%，R1@0.7提高了2.59%。


### 论文摘要

Video Moment Retrieval is a task in video understanding that aims to localize a specific temporal segment in an untrimmed video based on a natural language query. Despite recent progress in moment retrieval from videos using both traditional techniques and Multimodal Large Language Models (MLLM), most existing methods still rely on coarse temporal understanding and a single visual modality, limiting performance on complex videos. To address this, we introduce \textit{S}hot-aware \textit{M}ultimodal \textit{A}udio-enhanced \textit{R}etrieval of \textit{T}emporal \textit{S}egments (SMART), an MLLM-based framework that integrates audio cues and leverages shot-level temporal structure. SMART enriches multimodal representations by combining audio and visual features while applying \textbf{Shot-aware Token Compression}, which selectively retains high-information tokens within each shot to reduce redundancy and preserve fine-grained temporal details. We also refine prompt design to better utilize audio-visual cues. Evaluations on Charades-STA and QVHighlights show that SMART achieves significant improvements over state-of-the-art methods, including a 1.61\% increase in R1@0.5 and 2.59\% gain in R1@0.7 on Charades-STA.

---

## 110. MalRAG: A Retrieval-Augmented LLM Framework for Open-set Malicious Traffic Identification

**论文链接:** [http://arxiv.org/abs/2511.14129v1](http://arxiv.org/abs/2511.14129v1)

**作者:** Xiang Luo, Chang Liu, Gang Xiong, Chen Yang, Gaopeng Gou, Yaochen Ren, Zhen Li

**发布时间:** 2025-11-18

**备注:** 13 pages, 13 figures. Intended for submission to IEEE Transactions on Information Forensics and Security (TIFS)

### GPT解析

### 总结

MalRAG是首个基于大语言模型的检索增强框架，用于开放集恶意流量识别，通过构建全面的流量知识库、自适应检索和提示工程，实现对已知类别和新发现恶意流量的细粒度识别。

### 背景

网络安全中入侵检测系统标记的可疑流量细粒度识别至关重要。网络威胁不断演变，发现新型恶意流量与识别已知类别同样关键。现有深度学习方法通常依赖特定任务架构，限制了可迁移性，且需要对每个数据集单独调优。

### 目的

开发一种能够同时识别已知类别和新型恶意流量的开放集恶意流量识别方法，提高方法可迁移性，减少对特定数据集的调优需求。

### 方法

MalRAG框架包含：1)构建多视角流量数据库，从内容、结构和时间角度挖掘恶意流量；2)覆盖增强检索算法，跨视图查询组合最可能候选；3)流量感知自适应修剪，基于相似度分数选择候选子集；4)开发指导提示，集成任务指令、证据引用和决策指导。

### 主要发现

MalRAG在多样化真实世界数据集上，在已知类别细粒度识别和新型恶意流量发现方面均取得最先进结果。消融分析表明，MalRAG有效利用LLM能力，且不依赖特定LLM即可实现开放集恶意流量识别。

### 结论

MalRAG通过冻结LLM并依靠综合流量知识构建、自适应检索和提示工程，实现了对已知和新型恶意流量的高效识别，解决了现有方法的局限性。

### 翻译

网络安全中细粒度识别IDS标记的可疑流量至关重要。实际上，网络威胁不断演变，发现新型恶意流量与识别已知类别一样成为关键需求。最近的研究通过深度模型推动了这一目标，但它们通常依赖于特定任务架构，限制了可迁移性并需要对每个数据集进行调整。在本文中，我们介绍了MalRAG，这是第一个用于开放集恶意流量识别的LLM驱动检索增强框架。MalRAG冻结LLM并通过全面的流量知识构建、自适应检索和提示工程运行。具体而言，我们通过从内容、结构和时间角度挖掘先前的恶意流量来构建多视角流量数据库。此外，我们引入了覆盖增强检索算法，跨这些视图查询以组合最可能的候选，从而提高正确证据的包含率。然后，我们采用流量感知自适应修剪，基于流量感知相似度分数选择这些候选的变量子集，抑制不匹配项并产生可靠的检索证据。此外，我们开发了一套指导提示，将任务指令、证据引用和决策指导与检索到的证据集成，以提高LLM性能。在多样化的真实世界数据集和设置下，MalRAG在已知类别的细粒度识别和新型恶意流量发现方面都取得了最先进的结果。消融和深入分析进一步表明，MalRAG有效利用了LLM的能力，同时实现了开放集恶意流量识别而不依赖于特定LLM。


### 论文摘要

Fine-grained identification of IDS-flagged suspicious traffic is crucial in cybersecurity. In practice, cyber threats evolve continuously, making the discovery of novel malicious traffic a critical necessity as well as the identification of known classes. Recent studies have advanced this goal with deep models, but they often rely on task-specific architectures that limit transferability and require per-dataset tuning. In this paper we introduce MalRAG, the first LLM driven retrieval-augmented framework for open-set malicious traffic identification. MalRAG freezes the LLM and operates via comprehensive traffic knowledge construction, adaptive retrieval, and prompt engineering. Concretely, we construct a multi-view traffic database by mining prior malicious traffic from content, structural, and temporal perspectives. Furthermore, we introduce a Coverage-Enhanced Retrieval Algorithm that queries across these views to assemble the most probable candidates, thereby improving the inclusion of correct evidence. We then employ Traffic-Aware Adaptive Pruning to select a variable subset of these candidates based on traffic-aware similarity scores, suppressing incorrect matches and yielding reliable retrieved evidence. Moreover, we develop a suite of guidance prompts where task instruction, evidence referencing, and decision guidance are integrated with the retrieved evidence to improve LLM performance. Across diverse real-world datasets and settings, MalRAG delivers state-of-the-art results in both fine-grained identification of known classes and novel malicious traffic discovery. Ablation and deep-dive analyses further show that MalRAG effective leverages LLM capabilities yet achieves open-set malicious traffic identification without relying on a specific LLM.

---

## 111. ARC Is a Vision Problem!

**论文链接:** [http://arxiv.org/abs/2511.14761v1](http://arxiv.org/abs/2511.14761v1)

**作者:** Keya Hu, Ali Cy, Linlu Qiu, Xiaoman Delores Ding, Runqian Wang, Yeyin Eva Zhu, Jacob Andreas, Kaiming He

**发布时间:** 2025-11-18

**备注:** Technical Report. Project webpage: https://github.com/lillian039/VARC

### GPT解析

### 总结

本研究提出了Vision ARC（VARC）框架，将抽象推理语料库（ARC）问题重新定义为图像到图像的翻译问题，使用视觉架构处理视觉谜题任务，在ARC-1基准测试上达到60.4%的准确率，显著优于其他从零开始训练的方法，结果与领先的大型语言模型相当。

### 背景

ARC旨在促进对抽象推理的研究，这是人类智能的基本方面。现有研究通常将ARC视为语言导向问题，使用大型语言模型或循环推理模型解决，尽管ARC中的谜题任务本质上是视觉的。

### 目的

在视觉范式中构建ARC，将其视为图像到图像的翻译问题，融入视觉先验，并应用标准视觉架构来解决抽象推理问题。

### 方法

在'画布'上表示输入使其可像自然图像处理；应用标准视觉架构如Vision Transformer进行图像到图像映射；仅使用ARC数据进行从头训练；通过测试时训练推广到未见任务。

### 主要发现

提出的VARC框架在ARC-1基准上达到60.4%准确率，显著优于其他从零开始训练的方法；结果与领先大型语言模型相当；缩小了与人类平均性能的差距。

### 结论

通过将ARC视为视觉问题并使用视觉架构，取得了与领先语言模型相当的结果，为抽象推理研究提供了新视角。

### 翻译

抽象推理语料库（ARC）旨在促进对抽象推理的研究，这是人类智能的基本方面。处理ARC的常见方法将其视为语言导向问题，通过大型语言模型（LLMs）或循环推理模型解决。然而，尽管ARC中的谜题任务本质上是视觉的，现有研究很少从视觉中心的角度处理这个问题。在这项工作中，我们在视觉范式中构建ARC，将其视为图像到图像的翻译问题。为了融入视觉先验，我们在'画布'上表示输入，可以像自然图像一样处理。这样我们自然可以应用标准视觉架构，如普通Vision Transformer（ViT），来执行图像到图像映射。我们的模型仅使用ARC数据进行从头训练，并通过测试时训练推广到未见任务。我们的框架称为Vision ARC（VARC），在ARC-1基准上达到60.4%的准确率，显著优于同样从零开始训练的现有方法。我们的结果与领先的大型语言模型具有竞争力，并缩小了与人类平均性能的差距。


### 论文摘要

The Abstraction and Reasoning Corpus (ARC) is designed to promote research on abstract reasoning, a fundamental aspect of human intelligence. Common approaches to ARC treat it as a language-oriented problem, addressed by large language models (LLMs) or recurrent reasoning models. However, although the puzzle-like tasks in ARC are inherently visual, existing research has rarely approached the problem from a vision-centric perspective. In this work, we formulate ARC within a vision paradigm, framing it as an image-to-image translation problem. To incorporate visual priors, we represent the inputs on a "canvas" that can be processed like natural images. It is then natural for us to apply standard vision architectures, such as a vanilla Vision Transformer (ViT), to perform image-to-image mapping. Our model is trained from scratch solely on ARC data and generalizes to unseen tasks through test-time training. Our framework, termed Vision ARC (VARC), achieves 60.4% accuracy on the ARC-1 benchmark, substantially outperforming existing methods that are also trained from scratch. Our results are competitive with those of leading LLMs and close the gap to average human performance.

---

## 112. UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.14760v1](http://arxiv.org/abs/2511.14760v1)

**作者:** Rui Tian, Mingfei Gao, Haiming Gang, Jiasen Lu, Zhe Gan, Yinfei Yang, Zuxuan Wu, Afshin Dehghan

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出了UniGen-1.5，一个统一的多模态大语言模型，用于高级图像理解、生成和编辑。该模型在UniGen基础上进行了全面改进，增强了图像理解和生成能力，并解锁了强大的图像编辑能力。

### 背景

多模态大语言模型在图像处理领域的发展，需要同时具备图像理解、生成和编辑能力的统一模型。

### 目的

开发一个统一的多模态大语言模型，增强图像理解和生成能力，并解锁强大的图像编辑能力。

### 方法

1. 增强模型架构和训练流程；2. 提出统一的强化学习策略，通过共享奖励模型同时改进图像生成和编辑；3. 引入轻量级的编辑指令对齐阶段，提高编辑指令理解能力。

### 主要发现

在GenEval上获得0.89的整体分数；在ImgEdit上获得4.31的整体分数；超越了最先进的模型如BAGEL；达到了专有模型如GPT-Image-1的可比性能。

### 结论

UniGen-1.5在图像理解、生成和编辑方面展现出强大的能力，具有与专有模型相媲美的性能，同时保持开放性。

### 翻译

我们提出了UniGen-1.5，一个用于高级图像理解、生成和编辑的统一多模态大语言模型。基于UniGen，我们全面增强了模型架构和训练流程，以加强图像理解和生成能力，同时解锁强大的图像编辑能力。特别是，我们提出了一种统一的强化学习策略，通过共享的奖励模型共同改进图像生成和图像编辑。为了进一步提高图像编辑性能，我们提出了一个轻量级的编辑指令对齐阶段，显著提高了解析编辑指令的能力，这对于强化学习训练的成功至关重要。实验结果表明，UniGen-1.5展现出具有竞争力的理解和生成性能。具体而言，UniGen-1.5在GenEval和ImgEdit上分别获得0.89和4.31的整体分数，超越了最先进的模型如BAGEL，并达到了专有模型如GPT-Image-1的可比性能。


### 论文摘要

We present UniGen-1.5, a unified multimodal large language model (MLLM) for advanced image understanding, generation and editing. Building upon UniGen, we comprehensively enhance the model architecture and training pipeline to strengthen the image understanding and generation capabilities while unlocking strong image editing ability. Especially, we propose a unified Reinforcement Learning (RL) strategy that improves both image generation and image editing jointly via shared reward models. To further enhance image editing performance, we propose a light Edit Instruction Alignment stage that significantly improves the editing instruction comprehension that is essential for the success of the RL training. Experimental results show that UniGen-1.5 demonstrates competitive understanding and generation performance. Specifically, UniGen-1.5 achieves 0.89 and 4.31 overall scores on GenEval and ImgEdit that surpass the state-of-the-art models such as BAGEL and reaching performance comparable to proprietary models such as GPT-Image-1.

---

## 113. Vision Large Language Models Are Good Noise Handlers in Engagement Analysis

**论文链接:** [http://arxiv.org/abs/2511.14749v1](http://arxiv.org/abs/2511.14749v1)

**作者:** Alexander Vedernikov, Puneet Kumar, Haoyu Chen, Tapio Seppänen, Xiaobai Li

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种利用视觉大语言模型(VLMs)改进视频数据集中参与度识别的方法，通过问卷提取行为线索、数据分类和课程学习策略，在多个基准测试上取得了优于先前最先进方法的结果。

### 背景

视频数据集中的参与度识别与传统图像分类任务不同，特别受到主观标签和噪声的限制，这些因素限制了模型性能。

### 目的

克服参与度标签中主观性和噪声带来的挑战，提高视频数据集中参与度识别的准确性。

### 方法

提出一个利用视觉大语言模型(VLMs)的框架来精炼注释并指导训练过程；使用问卷提取行为线索并将数据分为高可靠性和低可靠性子集；引入结合课程学习和软标签细化的训练策略，逐步纳入模糊样本同时调整监督以反映不确定性；在精炼的高可靠性子集上训练传统计算机视觉模型，并结合课程策略进行增强。

### 主要发现

在精炼的高可靠性子集上训练的传统计算机视觉模型，结合课程策略后表现出改进；使用VLMs解决标签主观性问题带来了好处；该方法在EngageNet(六种特征设置中的三种，最大改进+1.21%)和DREAMS/PAFE上分别获得+0.22/+0.06的F1增益，超过了先前的最先进方法。

### 结论

通过VLMs解决标签主观性问题并采用课程学习和软标签细化策略，可以有效提高视频数据集中参与度识别的性能，在多个基准测试上取得了优于先前最先进方法的结果。

### 翻译

视频数据集中的参与度识别与传统图像分类任务不同，特别受到主观标签和噪声的限制，这些因素限制了模型性能。为了克服参与度标签中主观性和噪声带来的挑战，我们提出了一种利用视觉大语言模型(VLMs)来精炼注释并指导训练过程的框架。我们的框架使用问卷提取行为线索并将数据分为高可靠性和低可靠性子集。我们还引入了一种结合课程学习和软标签细化的训练策略，逐步纳入模糊样本同时调整监督以反映不确定性。我们证明，在精炼的高可靠性子集上训练的传统计算机视觉模型，结合我们的课程策略后表现出改进，突显了使用VLMs解决标签主观性的好处。这种方法在参与度基准测试上超过了先前的最先进方法，如在EngageNet(六种特征设置中的三种，最大改进+1.21%)和DREAMS/PAFE上分别获得+0.22/+0.06的F1增益。


### 论文摘要

Engagement recognition in video datasets, unlike traditional image classification tasks, is particularly challenged by subjective labels and noise limiting model performance. To overcome the challenges of subjective and noisy engagement labels, we propose a framework leveraging Vision Large Language Models (VLMs) to refine annotations and guide the training process. Our framework uses a questionnaire to extract behavioral cues and split data into high- and low-reliability subsets. We also introduce a training strategy combining curriculum learning with soft label refinement, gradually incorporating ambiguous samples while adjusting supervision to reflect uncertainty. We demonstrate that classical computer vision models trained on refined high-reliability subsets and enhanced with our curriculum strategy show improvements, highlighting benefits of addressing label subjectivity with VLMs. This method surpasses prior state of the art across engagement benchmarks such as EngageNet (three of six feature settings, maximum improvement of +1.21%), and DREAMS / PAFE with F1 gains of +0.22 / +0.06.

---

## 114. AdamHD: Decoupled Huber Decay Regularization for Language Model Pre-Training

**论文链接:** [http://arxiv.org/abs/2511.14721v1](http://arxiv.org/abs/2511.14721v1)

**作者:** Fu-Ming Guo, Yingfang Fan

**发布时间:** 2025-11-18

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: GPU-Accelerated and Scalable Optimization (ScaleOpt)

### GPT解析

### 总结

本文提出AdamHuberDecay，一种替代AdamW的优化器，通过使用Huber正则化替代ℓ2惩罚，解决了现有优化器对参数过度惩罚的问题，在预训练大型transformer模型中展现出更优的性能和效率。

### 背景

自适应优化器与解耦权重衰减（如AdamW）是预训练大型基于transformer的生成模型的标准方法。然而，权重衰减中嵌入的ℓ2惩罚是二次方的，导致所有参数以相同速率向原点靠近，使更新容易受到罕见但极端梯度方向的影响，并且常常对条件良好的坐标过度惩罚。

### 目的

提出一种替代AdamW的优化方法，解决ℓ2惩罚带来的问题，提高训练效率和模型性能。

### 方法

提出AdamHuberDecay，作为AdamW的替代方案，将ℓ2惩罚替换为解耦的平滑Huber正则化器。这种更新方法在参数幅度低于阈值δ时进行二次衰减，超过δ后进行线性（类似ℓ1）衰减。推导了闭式解耦Huber衰减步骤，并展示了如何以O(1)的额外成本将其与任何Adam系列优化器集成。

### 主要发现

AdamHuberDec具有以下特性：(a)有界的正则化梯度，(b)对每个坐标二阶矩缩放的不变性，(c)对过度增长的权重施加更强的稀疏性压力。在GPT-2和GPT-3预训练上的实验表明：(a)墙钟时间收敛速度提高10-15%，(b)验证困惑度降低最多4个点，(c)在下游任务上提供2.5-4.7%的性能提升，(d)产生明显更稀疏的权重直方图，在幅度剪枝后转化为20-30%的内存节省。消融研究证实了对异常值梯度和大批量训练的鲁棒性。

### 结论

AdamHuberDecay为下一代基础生成transformer的训练提供了一条更简单、更有效且更具弹性的路径，无需额外调参即可获得显著性能提升。

### 翻译

具有解耦权重衰减的自适应优化器（如AdamW）是预训练大型基于transformer的生成模型的实际标准。然而，嵌入在权重衰减中的ℓ2惩罚的二次特性使所有参数以相同速率向原点靠近，使更新容易受到罕见但极端梯度方向的影响，并且常常对条件良好的坐标过度惩罚。我们提出AdamHuberDecay，作为AdamW的替代方案，用解耦的平滑Huber正则化器替代ℓ2惩罚。由此产生的更新在参数幅度低于阈值δ时进行二次衰减，一旦超过δ则进行线性（类似ℓ1）衰减，从而产生：(i)有界的正则化梯度，(ii)对每个坐标二阶矩缩放的不变性，以及(iii)对过度增长的权重施加更强的稀疏性压力。我们推导了闭式解耦Huber衰减步骤，并展示了如何以O(1)的额外成本将其与任何Adam系列优化器集成。在GPT-2和GPT-3预训练上的大量实验表明，AdamHuberDecay（a）在墙钟时间上收敛速度提高10-15%，（b）将验证困惑度降低最多4个点，（c）在下游任务上提供2.5-4.7%的性能提升，以及（d）产生明显更稀疏的权重直方图，在幅度剪枝后转化为20-30%的内存节省，而无需调整AdamW默认网格之外的衰减系数。消融研究证实了对异常值梯度和大批量训练的鲁棒性，同时理论分析在有噪声更新下限制了预期参数范数。因此，AdamHuberDecay为下一代基础生成transformer的训练提供了一条更简单、更合理且更高效的路径。


### 论文摘要

Adaptive optimizers with decoupled weight decay, such as AdamW, are the de facto standard for pre-training large transformer-based generative models. Yet the quadratic nature of the $\ell_2$ penalty embedded in weight decay drives all parameters toward the origin at the same rate, making the update vulnerable to rare but extreme gradient directions and often over-penalizing well-conditioned coordinates. We propose AdamHuberDecay, a drop-in replacement for AdamW that substitutes the $\ell_2$ penalty with a decoupled smooth Huber regularizer. The resulting update decays parameters quadratically while their magnitude remains below a threshold $δ$, and linearly ($\ell_1$-like) once they exceed $δ$, yielding (i) bounded regularization gradients, (ii) invariance to per-coordinate second-moment rescaling, and (iii) stronger sparsity pressure on overgrown weights.   We derive the closed-form decoupled Huber decay step and show how to integrate it with any Adam-family optimizer at $O(1)$ extra cost. Extensive experiments on GPT-2 and GPT-3 pre-training demonstrate that AdamHuberDecay (a) converges 10-15% faster in wall-clock time, (b) reduces validation perplexity by up to 4 points, (c) delivers performance improvements of 2.5-4.7% across downstream tasks, and (d) yields visibly sparser weight histograms that translate into 20-30% memory savings after magnitude pruning, without tuning the decay coefficient beyond the default grid used for AdamW. Ablations confirm robustness to outlier gradients and large-batch regimes, together with theoretical analyses that bound the expected parameter norm under noisy updates. AdamHuberDecay therefore provides a simple, principled path toward more efficient and resilient training of next-generation foundational generative transformers.

---

## 115. FreeSwim: Revisiting Sliding-Window Attention Mechanisms for Training-Free Ultra-High-Resolution Video Generation

**论文链接:** [http://arxiv.org/abs/2511.14712v1](http://arxiv.org/abs/2511.14712v1)

**作者:** Yunfeng Wu, Jiayi Song, Zhenxiong Tan, Zihao He, Songhua Liu

**发布时间:** 2025-11-18

**备注:** 13 pages, 8 figures

### GPT解析

### 总结

本文提出了一种无需训练的方法，利用预训练的视频扩散Transformer来合成超高清视频，解决了传统Transformer视频生成器中注意力机制计算成本高的问题。

### 背景

现代基于Transformer的视频生成器中的注意力机制具有二次方的时间和内存复杂度，使得对超高清视频进行端到端训练变得极其昂贵。

### 目的

引入一种无需训练的方法，利用在原始尺度上预训练的视频扩散Transformer来合成更高分辨率的视频，而无需任何额外的训练或适应。

### 方法

核心是向内滑动窗口注意力机制，结合双路径管道和交叉注意力覆盖策略，确保局部注意力产生的语义内容被具有完整感受野的分支引导，同时采用交叉注意力缓存策略提高效率。

### 主要发现

该方法在无需训练的范式中能够提供具有精细视觉细节和高效性的超高清视频，在VBench上实现了优于基于训练的替代方案的性能。

### 结论

该方法成功解决了超高清视频生成中的计算效率问题，同时保持了高质量的结果，无需额外训练即可实现。

### 翻译

现代基于Transformer的视频生成器中注意力机制的二次方时间和内存复杂度，使得对超高清视频进行端到端训练变得异常昂贵。受此限制启发，我们引入了一种无需训练的方法，利用在其原始尺度上预训练的视频扩散Transformer来合成更高分辨率的视频，无需任何额外的训练或适应。我们方法的核心是一个向内滑动窗口注意力机制，它源于一个关键观察：保持每个查询令牌的训练尺度感受野对于保持视觉保真度和细节至关重要。然而，简单的局部窗口注意力不幸地经常导致内容重复，并且在生成结果中表现出缺乏全局连贯性。为了克服这一挑战，我们设计了一个双路径管道，用一种新的交叉注意力覆盖策略来支持窗口注意力，使局部注意力产生的语义内容能够被具有完整感受野的另一分支引导，从而确保整体一致性。此外，为了提高效率，我们对该分支采用了交叉注意力缓存策略，以避免频繁计算完整的3D注意力。大量实验表明，我们的方法在无需训练的范式中能够提供具有精细视觉细节和高效性的超高清视频。同时，它在VBench上实现了优于基于训练的替代方案的性能，同时保持了或提高了效率。代码可在以下网址获取：https://github.com/WillWu111/FreeSwim


### 论文摘要

The quadratic time and memory complexity of the attention mechanism in modern Transformer based video generators makes end-to-end training for ultra high resolution videos prohibitively expensive. Motivated by this limitation, we introduce a training-free approach that leverages video Diffusion Transformers pretrained at their native scale to synthesize higher resolution videos without any additional training or adaptation. At the core of our method lies an inward sliding window attention mechanism, which originates from a key observation: maintaining each query token's training scale receptive field is crucial for preserving visual fidelity and detail. However, naive local window attention, unfortunately, often leads to repetitive content and exhibits a lack of global coherence in the generated results. To overcome this challenge, we devise a dual-path pipeline that backs up window attention with a novel cross-attention override strategy, enabling the semantic content produced by local attention to be guided by another branch with a full receptive field and, therefore, ensuring holistic consistency. Furthermore, to improve efficiency, we incorporate a cross-attention caching strategy for this branch to avoid the frequent computation of full 3D attention. Extensive experiments demonstrate that our method delivers ultra-high-resolution videos with fine-grained visual details and high efficiency in a training-free paradigm. Meanwhile, it achieves superior performance on VBench, even compared to training-based alternatives, with competitive or improved efficiency. Codes are available at: https://github.com/WillWu111/FreeSwim

---

## 116. Attention via Synaptic Plasticity is All You Need: A Biologically Inspired Spiking Neuromorphic Transformer

**论文链接:** [http://arxiv.org/abs/2511.14691v1](http://arxiv.org/abs/2511.14691v1)

**作者:** Kallol Mondal, Ankush Kumar

**发布时间:** 2025-11-18

**备注:** 21 Pages, 5 Figures, 3 Table

### GPT解析

### 总结

这篇论文提出了一种名为S²TDPT的神经形态Transformer模型，通过使用脉冲时间依赖可塑性(STDP)实现自注意力机制，显著降低了能耗并提高了模型的可解释性。

### 背景

注意力机制是大脑选择性地关注特定方面而忽略不相关信息的能力，这启发了现代Transformer中的注意力机制。虽然Transformer支撑了像GPT这样的大型语言模型，但其训练和推理过程消耗大量能源。大脑注意力源自神经回路，而Transformer注意力则依赖于点积相似性。神经形态计算，特别是脉冲神经网络(SNNs)，为能效智能提供了大脑启发的路径，但当前基于脉冲的Transformer注意力仍存在非神经形态的问题。

### 目的

解决当前基于脉冲的Transformer中注意力层仍非神经形态的问题，包括：依赖点积相似性不适合事件驱动脉冲；保持注意力矩阵受冯·诺依曼瓶颈限制；仍与大脑式计算有差异。目标是开发一种更高效、更硬件友好且可解释的神经形态注意力模型。

### 方法

作者提出了S²TDPT(Spiking STDP Transformer)，一种通过脉冲时间依赖可塑性(STDP)实现自注意力的神经形态Transformer，将查询-键相关性嵌入到突触权重中。STDP是大脑记忆和学习的一个核心机制，在神经形态设备中被广泛研究，自然支持内存计算和非冯·诺依曼硬件。

### 主要发现

在CIFAR-10和CIFAR-100上，模型仅使用四个时间步就分别实现了94.35%和78.08%的准确率；在CIFAR-100上仅消耗0.49 mJ的能源，比标准ANN Transformer减少了88.47%的能源消耗；Grad-CAM显示模型关注语义相关区域，提高了可解释性。

### 结论

S²TDPT展示了如何通过生物启发的注意力机制产生能效高、硬件友好且可解释的神经形态模型，为未来的高效人工智能计算提供了新的方向。

### 翻译

注意力是大脑选择性地关注几个特定方面而忽略不相关信息的能力。这一生物原理启发了现代Transformer中的注意力机制。Transformer现在支撑着像GPT这样的大型语言模型，但代价是巨大的训练和推理能源消耗，导致大量的碳足迹。虽然大脑注意力源自神经回路，但Transformer注意力依赖于点积相似性来加权输入序列中的元素。神经形态计算，特别是脉冲神经网络(SNNs)，为能效智能提供了大脑启发的路径。尽管最近有基于脉冲的Transformer注意力研究工作，但核心注意力层仍然是非神经形态的。当前的脉冲注意力(i)依赖于点积或元素相似性，适合浮点运算，不适合事件驱动的脉冲；(ii)保持注意力矩阵，受到冯·诺依曼瓶颈的限制，限制了内存计算；(iii)仍然与大脑式计算有差异。为了解决这些问题，我们提出了脉冲STDP Transformer(S²TDPT)，一种通过脉冲时间依赖可塑性(STDP)实现自注意力的神经形态Transformer，将查询-键相关性嵌入到突触权重中。STDP是大脑记忆和学习的一个核心机制，在神经形态设备中被广泛研究，自然支持内存计算和非冯·诺依曼硬件。在CIFAR-10和CIFAR-100上，我们的模型仅使用四个时间步就实现了94.35%和78.08%的准确率，在CIFAR-100上仅消耗0.49 mJ的能源，比标准ANN Transformer减少了88.47%的能源消耗。Grad-CAM显示模型关注语义相关区域，提高了可解释性。总体而言，S²TDPT展示了如何通过生物启发的注意力机制产生能效高、硬件友好且可解释的神经形态模型。


### 论文摘要

Attention is the brain's ability to selectively focus on a few specific aspects while ignoring irrelevant ones. This biological principle inspired the attention mechanism in modern Transformers. Transformers now underpin large language models (LLMs) such as GPT, but at the cost of massive training and inference energy, leading to a large carbon footprint. While brain attention emerges from neural circuits, Transformer attention relies on dot-product similarity to weight elements in the input sequence. Neuromorphic computing, especially spiking neural networks (SNNs), offers a brain-inspired path to energy-efficient intelligence. Despite recent work on attention-based spiking Transformers, the core attention layer remains non-neuromorphic. Current spiking attention (i) relies on dot-product or element-wise similarity suited to floating-point operations, not event-driven spikes; (ii) keeps attention matrices that suffer from the von Neumann bottleneck, limiting in-memory computing; and (iii) still diverges from brain-like computation. To address these issues, we propose the Spiking STDP Transformer (S$^{2}$TDPT), a neuromorphic Transformer that implements self-attention through spike-timing-dependent plasticity (STDP), embedding query--key correlations in synaptic weights. STDP, a core mechanism of memory and learning in the brain and widely studied in neuromorphic devices, naturally enables in-memory computing and supports non-von Neumann hardware. On CIFAR-10 and CIFAR-100, our model achieves 94.35\% and 78.08\% accuracy with only four timesteps and 0.49 mJ on CIFAR-100, an 88.47\% energy reduction compared to a standard ANN Transformer. Grad-CAM shows that the model attends to semantically relevant regions, enhancing interpretability. Overall, S$^{2}$TDPT illustrates how biologically inspired attention can yield energy-efficient, hardware-friendly, and explainable neuromorphic models.

---

## 117. Giant enhancement of attosecond tunnel ionization competes with disorder-driven decoherence in silicon

**论文链接:** [http://arxiv.org/abs/2511.14678v1](http://arxiv.org/abs/2511.14678v1)

**作者:** D. N. Purschke, D. Vick, A. Cárdenas, N. Haram, P. Bastani, S. Gholam-Mirzaei, S. Mokhtari, V. Jelic, J. Chen, J. Canlas, J. Tordiff, Md. W. Rahman, A. Yu. Naumov, D. M. Villeneuve, A. Staudte, M. Salomons, R. E. F. Silva, Á. Jiménez-Galán, G. Vampa

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究了硅在晶体到非晶相变过程中的高次谐波产生，发现光谱显著重塑，低阶谐波增强而高阶谐波被抑制。通过量子动力学建模，发现非晶相中隧道电离产量增强250倍以上，无序引起的退相干在约六个晶格范围内阻尼电子-空穴极化。HHG光谱学揭示了传统探针无法发现的残余有序性，并观察到非晶硅岛的快速靶向激光退火。

### 背景

高次谐波产生(HHG)是一种强场现象，对隧道电离和固体中电子-空穴对的相干传输的阿秒动力学敏感。虽然固体HHG的基础已经建立，但对亚循环时间尺度上退相干本质的深入理解仍然难以捉摸。此外，对纳米尺度电离进行控制的需求日益增长。

### 目的

研究硅晶体从晶态到非晶态的结构相变过程中的HHG，探索HHG作为结构无序探测手段的潜力，寻找控制纳米尺度电离的新方法。

### 方法

研究硅在晶体到非晶结构相变过程中的高次谐波产生；建模实空间量子动力学；观察无序引起的退相干；使用HHG光谱学探测残余有序性；观察非晶硅岛的快速和靶向非共振激光退火。

### 主要发现

在硅从晶体到非晶的结构相变过程中，观察到光谱的显著重塑；非晶相中低阶谐波产量增强，同时高阶谐波被抑制；非晶相中的隧道电离产量增强了250倍以上；无序引起的退相干在约六个晶格范围内阻尼电子-空穴极化；HHG光谱学揭示了传统探针无法发现的残余有序性；观察到非晶硅岛的快速和靶向非共振激光退火。

### 结论

研究结果为强场现象中的阿秒退相干提供了独特的见解；建立了HHG光谱学作为结构无序探测手段；为光波纳米电子学的新机遇铺平了道路。

### 翻译

高次谐波产生(HHG)是一种强场现象，对隧道电离和固体中电子-空穴对的相干传输的阿秒动力学敏感。虽然固体HHG的基础已经建立，但对亚循环时间尺度上退相干本质的深入理解仍然难以捉摸。此外，对纳米尺度电离进行控制的需求日益增长。我们研究了硅在晶体到非晶结构相变过程中的HHG，观察到光谱的显著重塑，非晶相中低阶谐波产量增强，同时高阶谐波被抑制。对实空间量子动力学的建模将我们的观察结果与非晶相中隧道电离产量的巨大增强(>250倍)联系起来，并发现无序引起的退相干在约六个晶格范围内阻尼电子-空穴极化。HHG光谱学还揭示了传统探针无法发现的残余有序性。最后，我们观察到非晶硅岛的快速和靶向非共振激光退火。我们的研究结果为强场现象中的阿秒退相干提供了独特的见解，建立了HHG光谱学作为结构无序的探测手段，并为光波纳米电子学的新机遇铺平了道路。


### 论文摘要

High-harmonic generation (HHG) is a strong-field phenomenon that is sensitive to the attosecond dynamics of tunnel ionization and coherent transport of electron-hole pairs in solids. While the foundations of solid HHG have been established, a deep understanding into the nature of decoherence on sub-cycle timescales remains elusive. Furthermore, there is a growing need for tools to control ionization at the nanoscale. Here, we study HHG in silicon along a crystalline-to-amorphous (c-Si to a-Si) structural phase transition and observe a dramatic reshaping of the spectrum, with enhanced lower-order harmonic yield accompanied by quenching of the higher-order harmonics. Modelling the real-space quantum dynamics links our observations to a giant enhancement (>250 times) of tunnel ionization yield in the amorphous phase and a disorder-induced decoherence that damps the electron-hole polarization over approximately six lattice sites. HHG spectroscopy also reveals remnant order that was not apparent with conventional probes. Finally, we observe a rapid and targeted non-resonant laser annealing of amorphous silicon islands. Our results offer a unique insight into attosecond decoherence in strong-field phenomena, establish HHG spectroscopy as a probe of structural disorder, and pave the way for new opportunities in lightwave nanoelectronics.

---

## 118. FHIRconnect: Towards a seamless integration of openEHR and FHIR

**论文链接:** [http://arxiv.org/abs/2511.14618v1](http://arxiv.org/abs/2511.14618v1)

**作者:** Severin Kohler, Jordi Piera Jiménez, Michael Anywar, Lars Fuhrmann, Heather Leslie, Maximilian Meixner, Julian Saß, Florian Kärcher, Diego Boscá, Birger Haarbrandt, Michael Marschollek, Roland Eils

**发布时间:** 2025-11-18

**备注:** 27 pages, 4 figures

### GPT解析

### 总结

FHIRconnect是一种创新的领域特定语言和开源转换引擎，解决了openEHR和HL7 FHIR之间医疗保健互操作性的挑战，通过三层架构实现了标准化双向数据交换，并提供了开源工具促进社区驱动的映射标准化。

### 背景

openEHR和HL7 FHIR之间的医疗保健互操作性仍然具有挑战性，这主要是由于它们在数据建模方法上的根本差异以及缺乏标准化的转换机制。

### 目的

开发FHIRconnect，一种新的领域特定语言和开源转换引擎，实现openEHR和FHIR之间的标准化、双向数据交换。

### 方法

采用三层架构解决关键互操作性差距，利用基于国际原型的国际基础同时支持本地定制，实现了65%的映射重用。使用该框架，成功将24个国际原型映射到七个临床领域的15个FHIR配置文件。

### 主要发现

FHIRconnect成功实现了跨七个临床领域的24个国际原型到15个FHIR配置文件的映射，通过三层架构和基于国际原型的映射方法，实现了65%的映射重用率。

### 结论

FHIRconnect的主要贡献包括首个用于openEHR-FHIR转换的全面领域特定语言与正式规范、开源执行引擎openFHIR，以及涵盖高影响力临床原型的可访问映射库。这些组件建立了社区驱动映射标准化的技术基础，减少对定制ETL解决方案的依赖，推进基于开放标准的医疗保健IT系统的语法和语义互操作性。

### 翻译

由于openEHR和HL7 FHIR在数据建模方法上的根本差异以及缺乏标准化转换机制，两者之间的医疗保健互操作性仍然具有挑战性。本文提出了FHIRconnect，一种新的领域特定语言和开源转换引擎，能够实现openEHR和FHIR之间的标准化、双向数据交换。我们的方法通过三层架构解决了关键的互操作性差距，通过利用基于国际原型的国际基础同时支持本地定制，实现了65%的映射重用。使用此框架，FHIRconnect成功将24个国际原型映射到七个临床领域的15个FHIR配置文件。主要贡献包括首个用于openEHR-FHIR转换的具有正式规范的全面领域特定语言、一个开源执行引擎（openFHIR），以及一个涵盖高影响力临床原型的可访问映射库。这些组件共同建立了社区驱动映射标准化的技术基础，减少了对定制ETL解决方案的依赖，并推进了基于开放标准的医疗保健IT系统的语法和语义互操作性。


### 论文摘要

Healthcare interoperability between openEHR and HL7 FHIR remains challenging due to fundamental differences in their data modeling approaches and the absence of standardized transformation mechanisms. This paper presents FHIRconnect, a novel domain-specific language and open-source transformation engine that enables standardized, bidirectional data exchange between openEHR and FHIR. Our approach addresses critical interoperability gaps through a triple-layered architecture that achieves 65% mapping reuse across projects by leveraging international archetype-based foundations while supporting local customizations. Using this framework, FHIRconnect successfully mapped 24 international archetypes to 15 FHIR profiles across seven clinical domains. Key contributions include the first comprehensive DSL for openEHR-FHIR transformation with a formal specification, an open-source execution engine (openFHIR), and an accessible mapping library covering high-impact clinical archetypes. Together, these components establish the technical basis for community-driven mapping standardization, reducing reliance on custom ETL solutions and advancing syntactic and semantic interoperability in healthcare IT systems built on open standards.

---

## 119. Rate-Distortion Guided Knowledge Graph Construction from Lecture Notes Using Gromov-Wasserstein Optimal Transport

**论文链接:** [http://arxiv.org/abs/2511.14595v1](http://arxiv.org/abs/2511.14595v1)

**作者:** Yuan An, Ruhma Hashmi, Michelle Rogers, Jane Greenberg, Brian K. Smith

**发布时间:** 2025-11-18

**备注:** Accepted in the 5th Workshop on Knowledge Graphs and Big Data in Conjunction with IEEE Big Data 2025

### GPT解析

### 总结

本文提出了一种基于率失真理论和最优输运几何学的知识图谱构建和优化框架，用于将教育材料转换为高质量知识图谱，进而生成优质多选题。

### 背景

面向任务的知识图谱使AI驱动的学习辅助系统能自动生成高质量多选题，但将非结构化教育材料（如讲义和幻灯片）转换为能捕捉关键教学内容的知识图谱仍很困难。

### 目的

开发一个框架，将教育材料转换为紧凑且信息保留的知识图谱，用于生成高质量多选题。

### 方法

框架将讲座内容建模为度量测度空间捕捉语义和结构关系；使用Fused Gromov-Wasserstein耦合对齐候选知识图谱量化语义失真；通过知识图谱大小表达率项反映复杂性和紧凑性；采用细化操作符最小化率失真拉格朗日函数产生优化知识图谱。

### 主要发现

应用于数据科学讲座的原型生成了可解释的率失真曲线，表明从优化后的知识图谱生成的多选题在十五个质量标准上优于从原始讲义生成的多选题。

### 结论

该研究为个性化AI辅助教育中的信息论知识图谱优化奠定了理论基础。

### 翻译

面向任务的知识图谱使AI驱动的学习辅助系统能够自动生成高质量的多选题。然而，将非结构化的教育材料（如讲义和幻灯片）转换为能够捕捉关键教学内容的知识图谱仍然很困难。我们提出了一种基于率失真理论和最优输运几何学的知识图谱构建和优化框架。在该框架中，讲座内容被建模为度量测度空间，捕捉语义和结构关系，同时使用Fused Gromov-Wasserstein耦合对齐候选知识图谱，以量化语义失真。率项通过知识图谱的大小表达，反映复杂性和紧凑性。细化操作符（添加、合并、分割、移除、重连）最小化率失真拉格朗日函数，产生紧凑且信息保留的知识图谱。我们应用于数据科学讲座的原型生成了可解释的率失真曲线，并表明从优化后的知识图谱生成的多选题在十五个质量标准上始终优于从原始讲义生成的多选题。本研究为个性化AI辅助教育中的信息论知识图谱优化奠定了理论基础。


### 论文摘要

Task-oriented knowledge graphs (KGs) enable AI-powered learning assistant systems to automatically generate high-quality multiple-choice questions (MCQs). Yet converting unstructured educational materials, such as lecture notes and slides, into KGs that capture key pedagogical content remains difficult. We propose a framework for knowledge graph construction and refinement grounded in rate-distortion (RD) theory and optimal transport geometry. In the framework, lecture content is modeled as a metric-measure space, capturing semantic and relational structure, while candidate KGs are aligned using Fused Gromov-Wasserstein (FGW) couplings to quantify semantic distortion. The rate term, expressed via the size of KG, reflects complexity and compactness. Refinement operators (add, merge, split, remove, rewire) minimize the rate-distortion Lagrangian, yielding compact, information-preserving KGs. Our prototype applied to data science lectures yields interpretable RD curves and shows that MCQs generated from refined KGs consistently surpass those from raw notes on fifteen quality criteria. This study establishes a principled foundation for information-theoretic KG optimization in personalized and AI-assisted education.

---

## 120. Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks

**论文链接:** [http://arxiv.org/abs/2511.14592v1](http://arxiv.org/abs/2511.14592v1)

**作者:** Xianhui Meng, Yuchen Zhang, Zhijian Huang, Zheng Lu, Ziling Ji, Yaoyao Yin, Hongyuan Zhang, Guangfeng Jiang, Yandan Lin, Long Chen, Hangjun Ye, Li Zhang, Jun Liu, Xiaoshuai Hao

**发布时间:** 2025-11-18

### GPT解析

### 总结

这篇论文介绍了DSBench，第一个全面评估视觉-语言模型(VLMs)在自动驾驶安全关键场景中表现的基准测试，包括外部环境风险和驾驶行为安全两大类，共10个主要类别和28个子类别。研究通过评估各种VLMs发现它们在复杂安全情况下性能显著下降，并提出了一个包含98K个安全场景实例的数据集来提高VLMs的安全性能。

### 背景

视觉-语言模型(VLMs)在自动驾驶领域展现出巨大潜力，但它们在安全关键场景中的适用性尚未得到充分探索，存在安全隐患。这主要是因为缺乏能够同时评估外部环境风险和舱内驾驶行为安全的全面基准测试。

### 目的

开发一个全面的驾驶安全基准测试(DSBench)，以统一方式评估VLMs对各种安全风险的认知能力，填补现有评估体系的空白。

### 方法

创建了DSBench基准测试，包含外部环境风险和驾驶行为安全两大类，细分为10个主要类别和28个子类别。构建了一个包含98K个安全场景实例的数据集，专注于舱内和外部安全场景。对各种主流开源和闭源VLMs进行了广泛评估，并使用构建的数据集对现有VLMs进行了微调。

### 主要发现

各种主流VLMs在复杂安全关键情况下表现出显著的性能下降；微调专门的安全数据集可以显著提高现有VLMs的安全性能；当前VLMs在自动驾驶安全关键场景中存在明显的安全缺陷。

### 结论

DSBench基准测试为评估和改进VLMs在自动驾驶安全关键场景中的表现提供了全面工具。通过微调专门的安全数据集，可以显著提高VLMs的安全性能，为推进自动驾驶技术铺平道路。基准测试工具包、代码和模型检查点将公开提供，促进该领域的发展。

### 翻译

视觉-语言模型(VLMs)在自动驾驶方面展现出巨大潜力，但它们在安全关键场景中的适用性在很大程度上尚未被探索，引发了安全方面的担忧。这一问题源于缺乏能够同时评估外部环境风险和舱内驾驶行为安全的全面基准测试。为了弥合这一关键差距，我们引入了DSBench，这是第一个全面的驾驶安全基准测试，旨在统一方式评估VLMs对各种安全风险的认知能力。DSBench包含两大类：外部环境风险和舱内驾驶行为安全，细分为10个主要类别和总共28个子类别。这项全面评估涵盖了广泛的场景，确保对VLMs在安全关键环境中的表现进行彻底评估。通过各种主流开源和闭源VLMs的广泛评估，发现在复杂的安全关键情况下性能显著下降，突显了紧迫的安全问题。为了解决这个问题，我们构建了一个包含98K个实例的大型数据集，专注于舱内和外部安全场景，显示微调这个数据集可以显著提高现有VLMs的安全性能，为推进自动驾驶技术铺平道路。基准测试工具包、代码和模型检查点将公开提供。


### 论文摘要

Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible.

---

## 121. SweeperBot: Making 3D Browsing Accessible through View Analysis and Visual Question Answering

**论文链接:** [http://arxiv.org/abs/2511.14567v1](http://arxiv.org/abs/2511.14567v1)

**作者:** Chen Chen, Cuong Nguyen, Alexa Siu, Dingzeyu Li, Nadir Weibel

**发布时间:** 2025-11-18

**备注:** 28 pages, 16 figures, this article has been accepted for publication in the International Journal of Human-Computer Interaction (IJHCI), published by Taylor and Francis

### GPT解析

### 总结

SweeperBot是一个创新的系统，它通过结合最佳视图选择技术和基础模型，使屏幕阅读器用户能够通过视觉问答来探索和比较3D模型。专家审查和调查研究表明该系统对盲人和低视力用户有效且实用。

### 背景

屏幕阅读器(SR)用户访问3D模型仍然具有挑战性。虽然现有的3D查看器允许创作者提供替代文本，但它们通常缺乏关于3D模型的足够细节。

### 目的

基于一项形成性研究，本文介绍了SweeperBot，这是一个使SR用户能够利用视觉问答来探索和比较3D模型的系统。

### 方法

SweeperBot通过结合最佳视图选择技术与基于生成和识别的基础模型的优势来回答SR用户的视觉问题。

### 主要发现

一项针对10名有屏幕阅读器经验的盲人和低视力(BLV)用户的专家审查表明，使用SweeperBot协助BLV用户探索和比较3D模型是可行的。SweeperBot生成的描述质量通过一项针对30名视力正常参与者的第二次调查研究得到了验证。

### 结论

SweeperBot系统有效地帮助盲人和低视力用户通过视觉问答方式探索和比较3D模型，提高了3D模型的可访问性。

### 翻译

屏幕阅读器(SR)用户访问3D模型仍然具有挑战性。虽然现有的3D查看器允许创作者提供替代文本，但它们通常缺乏关于3D模型的足够细节。基于一项形成性研究，本文介绍了SweeperBot，这是一个使SR用户能够利用视觉问答来探索和比较3D模型的系统。SweeperBot通过结合最佳视图选择技术与基于生成和识别的基础模型的优势来回答SR用户的视觉问题。一项针对10名有屏幕阅读器经验的盲人和低视力(BLV)用户的专家审查表明，使用SweeperBot协助BLV用户探索和比较3D模型是可行的。SweeperBot生成的描述质量通过一项针对30名视力正常参与者的第二次调查研究得到了验证。


### 论文摘要

Accessing 3D models remains challenging for Screen Reader (SR) users. While some existing 3D viewers allow creators to provide alternative text, they often lack sufficient detail about the 3D models. Grounded on a formative study, this paper introduces SweeperBot, a system that enables SR users to leverage visual question answering to explore and compare 3D models. SweeperBot answers SR users' visual questions by combining an optimal view selection technique with the strength of generative- and recognition-based foundation models. An expert review with 10 Blind and Low-Vision (BLV) users with SR experience demonstrated the feasibility of using SweeperBot to assist BLV users in exploring and comparing 3D models. The quality of the descriptions generated by SweeperBot was validated by a second survey study with 30 sighted participants.

---

## 122. DecNefLab: A Modular and Interpretable Simulation Framework for Decoded Neurofeedback

**论文链接:** [http://arxiv.org/abs/2511.14555v1](http://arxiv.org/abs/2511.14555v1)

**作者:** Alexander Olza, Roberto Santana, David Soto

**发布时间:** 2025-11-18

### GPT解析

### 总结

DecNefLab是一个模块化和可解释的模拟框架，将DecNef形式化为机器学习问题，通过虚拟实验室环境帮助研究人员建模、分析和理解神经反馈动态，使用潜变量生成模型作为模拟参与者，直接观察内部认知状态，系统评估不同协议设计和受试者特征对学习的影响。

### 背景

DecNef是一种蓬勃发展的非侵入性脑调制方法，在神经医学和认知神经科学中有广泛应用，但研究进展受限于受试者依赖的学习变异性、依赖间接措施量化进展以及实验的高成本和时间需求。

### 目的

提出DecNefLab框架，将DecNef形式化为机器学习问题，为研究人员提供虚拟实验室环境，以建模、分析和理解神经反馈动态。

### 方法

DecNefLab作为一个模块化和可解释的模拟框架，使用潜变量生成模型作为模拟参与者，允许直接观察内部认知状态，系统评估不同协议设计和受试者特征如何影响学习过程。

### 主要发现

该框架可以重现DecNef学习的经验现象，识别DecNef反馈无法诱导学习的条件，并在人体实施前通过计算机模拟指导设计更稳健可靠的DecNef协议。

### 结论

DecNefLab弥合了计算建模和认知神经科学之间的差距，为方法创新、稳健的协议设计提供了原则性基础，促进对基于DecNef的脑调制的更深入理解。

### 翻译

解码神经反馈（DecNef）是一种蓬勃发展的非侵入性脑调制方法，在神经医学和认知神经科学中有广泛应用。然而，DecNef研究的进展仍然受限于受试者依赖的学习变异性、依赖间接措施来量化进展以及实验的高成本和时间需求。我们提出了DecNefLab，一个模块化和可解释的模拟框架，将DecNef形式化为机器学习问题。除了提供虚拟实验室外，DecNefLab使研究人员能够建模、分析和理解神经反馈动态。使用潜变量生成模型作为模拟参与者，DecNefLab允许直接观察内部认知状态，并系统评估不同协议设计和受试者特征如何影响学习。我们展示了这种方法如何能够（i）重现DecNef学习的经验现象，（ii）识别DecNef反馈无法诱导学习的条件，以及（iii）在人体实施前，通过计算机模拟指导设计更稳健可靠的DecNef协议。总之，DecNefLab弥合了计算建模和认知神经科学之间的差距，为方法创新、稳健的协议设计提供了原则性基础，并最终促进对基于DecNef的脑调制的更深入理解。


### 论文摘要

Decoded Neurofeedback (DecNef) is a flourishing non-invasive approach to brain modulation with wide-ranging applications in neuromedicine and cognitive neuroscience. However, progress in DecNef research remains constrained by subject-dependent learning variability, reliance on indirect measures to quantify progress, and the high cost and time demands of experimentation.   We present DecNefLab, a modular and interpretable simulation framework that formalizes DecNef as a machine learning problem. Beyond providing a virtual laboratory, DecNefLab enables researchers to model, analyze and understand neurofeedback dynamics. Using latent variable generative models as simulated participants, DecNefLab allows direct observation of internal cognitive states and systematic evaluation of how different protocol designs and subject characteristics influence learning.   We demonstrate how this approach can (i) reproduce empirical phenomena of DecNef learning, (ii) identify conditions under which DecNef feedback fails to induce learning, and (iii) guide the design of more robust and reliable DecNef protocols in silico before human implementation.   In summary, DecNefLab bridges computational modeling and cognitive neuroscience, offering a principled foundation for methodological innovation, robust protocol design, and ultimately, a deeper understanding of DecNef-based brain modulation.

---

## 123. Secondary electron topographical contrast formation in scanning transmission electron microscopy

**论文链接:** [http://arxiv.org/abs/2511.14491v1](http://arxiv.org/abs/2511.14491v1)

**作者:** Evgenii Vlasov, Wouter Heyvaert, Tom Stoops, Sandra Van Aert, Johan Verbeeck, Sara Bals

**发布时间:** 2025-11-18

**DOI:** 10.1016/j.ultramic.2025.114278

### GPT解析

### 总结

本文探讨了二次电子(SE)成像在扫描透射电子显微镜(STEM)中的应用价值，并提出了一种新的物理模型来解决SE成像对比度解释的难题。

### 背景

二次电子(SE)成像为传统的扫描透射电子显微镜(STEM)提供了表面敏感的、伪3D形貌信息的互补能力。然而，由于发射的SE与TEM物镜磁场之间的复杂相互作用，这类图像的对比度解释仍然是经验性的。

### 目的

提出一个分析物理模型，该模型考虑了SE发射的物理以及发射的SE与磁场之间的相互作用，从而实现更可靠的图像解释。

### 方法

开发一个考虑SE发射物理和发射SE与磁场相互作用的解析物理模型。

### 主要发现

所提出的模型能够实现更可靠的图像解释，并为新型3D表面重建算法奠定基础。

### 结论

通过考虑SE发射的物理和磁场相互作用，可以更准确地解释SE成像的对比度，这为未来的3D表面重建提供了可能性。

### 翻译

二次电子(SE)成像通过提供表面敏感的、伪3D形貌信息，为传统的扫描透射电子显微镜(STEM)提供了强大的互补能力。然而，由于发射的SE与TEM物镜磁场之间的复杂相互作用，此类图像的对比度解释仍然是经验性的。在这里，我们提出了一个分析物理模型，该模型考虑了SE发射的物理以及发射的SE与磁场之间的相互作用。这使得能够进行更可靠的图像解释，并可能为新型3D表面重建算法奠定基础。


### 论文摘要

Secondary electron (SE) imaging offers a powerful complementary capabilities to conventional scanning transmission electron microscopy (STEM) by providing surface-sensitive, pseudo-3D topographic information. However, contrast interpretation of such images remains empirical due to complex interactions of emitted SE with the magnetic field in the objective field of TEM. Here, we propose an analytical physical model that takes into account the physics of SE emission and interaction of the emitted SEs with magnetic field. This enables more reliable image interpretation and potentially lay the foundation for novel 3D surface reconstruction algorithms.

---

## 124. Segmentation-Aware Latent Diffusion for Satellite Image Super-Resolution: Enabling Smallholder Farm Boundary Delineation

**论文链接:** [http://arxiv.org/abs/2511.14481v1](http://arxiv.org/abs/2511.14481v1)

**作者:** Aditi Agarwal, Anjali Jain, Nikita Saxena, Ishan Deshpande, Michal Kazmierski, Abigail Annkah, Nadav Sherman, Karthikeyan Shanmugam, Alok Talekar, Vaibhav Rajan

**发布时间:** 2025-11-18

### GPT解析

### 总结

SEED-SR是一种新的农田边界划分方法，通过结合条件潜在扩散模型和多光谱多源地理空间基础模型，在分割感知的潜在空间中执行超分辨率，而非传统的像素空间方法，实现了20倍比例因子的分割图生成。

### 背景

通过卫星图像分割划定农田边界是农业应用的基础步骤，但小农户农场的准确边界划分需要高分辨率(HR)图像，而HR图像重访频率低（如每年一次）。为支持更频繁的监测，可将HR图像作为参考与重访频率更高的低分辨率(LR)图像结合，但现有方法存在局限性。

### 目的

开发一种能够有效结合高分辨率和低分辨率卫星图像的方法，实现更准确、更频繁的农田边界监测，特别是针对小农户农场。

### 方法

提出SEED-SR方法，结合条件潜在扩散模型和大尺度多光谱、多源地理空间基础模型，创新性地在分割感知的潜在空间中执行超分辨率，而非传统像素空间方法。

### 主要发现

SEED-SR能够生成20倍比例因子的分割图，在两个大型真实数据集上，实例分割和语义分割指标分别比基于最先进Ref-SR方法的方法提高了25.5和12.9。

### 结论

通过在分割感知的潜在空间中执行超分辨率，SEED-SR有效解决了现有Ref-SR方法的局限性，实现了更准确、更高效的农田边界划分，为农业应用提供了更频繁的监测能力。

### 翻译

通过卫星图像分割来划定农田边界是许多农业应用的基本步骤。对于小农户农场，准确的边界划分需要使用高分辨率(HR)图像，而这些图像的重访频率很低（例如每年一次）。为了支持更频繁的（次）季节性监测，可以将HR图像作为参考(ref)与重访频率更高（例如每周一次）的低分辨率(LR)图像结合，使用基于参考的超分辨率(Ref-SR)方法。然而，当前的Ref-SR方法优化感知质量并平滑了下游任务所需的关键特征，无法满足此任务的大规模因子需求。此外，先前的两步方法（先超分辨率再分割）没有有效利用多样化的卫星源作为输入。我们通过一种新方法SEED-SR解决了这些问题，它结合了条件潜在扩散模型和大尺度多光谱、多源地理空间基础模型。我们的关键创新是绕过像素空间的显式超分辨率任务，而是在分割感知的潜在空间中执行超分辨率。这种独特的方法使我们能够以前所未有的20倍比例因子生成分割图，在两个大型真实数据集上的严格实验表明，与基于最先进Ref-SR方法的方法相比，实例分割和语义分割指标分别提高了25.5和12.9。


### 论文摘要

Delineating farm boundaries through segmentation of satellite images is a fundamental step in many agricultural applications. The task is particularly challenging for smallholder farms, where accurate delineation requires the use of high resolution (HR) imagery which are available only at low revisit frequencies (e.g., annually). To support more frequent (sub-) seasonal monitoring, HR images could be combined as references (ref) with low resolution (LR) images -- having higher revisit frequency (e.g., weekly) -- using reference-based super-resolution (Ref-SR) methods. However, current Ref-SR methods optimize perceptual quality and smooth over crucial features needed for downstream tasks, and are unable to meet the large scale-factor requirements for this task. Further, previous two-step approaches of SR followed by segmentation do not effectively utilize diverse satellite sources as inputs. We address these problems through a new approach, $\textbf{SEED-SR}$, which uses a combination of conditional latent diffusion models and large-scale multi-spectral, multi-source geo-spatial foundation models. Our key innovation is to bypass the explicit SR task in the pixel space and instead perform SR in a segmentation-aware latent space. This unique approach enables us to generate segmentation maps at an unprecedented 20$\times$ scale factor, and rigorous experiments on two large, real datasets demonstrate up to $\textbf{25.5}$ and $\textbf{12.9}$ relative improvement in instance and semantic segmentation metrics respectively over approaches based on state-of-the-art Ref-SR methods.

---

## 125. Watchdogs and Oracles: Runtime Verification Meets Large Language Models for Autonomous Systems

**论文链接:** [http://arxiv.org/abs/2511.14435v1](http://arxiv.org/abs/2511.14435v1)

**作者:** Angelo Ferrando

**发布时间:** 2025-11-18

**DOI:** 10.4204/EPTCS.436.8

**备注:** In Proceedings FMAS 2025, arXiv:2511.13245

### GPT解析

### 总结

本文提出运行时验证(RV)和大语言模型(LLMs)的共生集成方法，以提高自主系统在包含学习组件和开放环境中的安全性和可信度。RV可作为LLM驱动自主系统的护栏，而LLMs可通过辅助规范捕获、支持预期推理和处理不确定性来扩展RV的能力。

### 背景

自主系统在包含学习组件和开放环境时，确保其安全性和可信度特别困难。形式化方法能提供强保证但依赖完整模型和静态假设。运行时验证通过监控运行时执行来补充形式化方法，预测性变体则可预测潜在违规。大语言模型擅长自然语言转换与模式识别，但易出错且缺乏形式化保证。

### 目的

论证RV和LLMs共生集成的必要性，展示两种技术如何相互补充，讨论其与现有调查和路线图的不同之处，分析挑战和认证含义，并确定未来可靠自主性的研究方向。

### 方法

提出RV和LLMs的共生集成方法：RV作为LLM驱动自主系统的护栏，提供运行时监控和预测；LLMs则通过辅助规范捕获、支持预期推理和处理不确定性来扩展RV的能力。

### 主要发现

1. RV和LLMs的共生集成可相互补充提高系统可靠性；2. RV可作为LLM驱动自主系统的护栏；3. LLMs可扩展RV的能力；4. 这种相互强化方法与现有调查和路线图不同；5. 此集成面临挑战并具有认证含义。

### 结论

通过RV和LLMs的共生集成，可构建更安全可靠的自主系统，特别是在包含学习组件和开放环境的情况下。这种相互强化方法为解决自主系统安全性和可信度挑战提供了新思路。

### 翻译

当自主系统包含学习组件和开放环境时，确保其安全性和可信度尤其困难。形式化方法提供了强有力的保证，但依赖于完整的模型和静态假设。运行时验证(RV)通过监控运行时的执行来补充它们，而在其预测性变体中，则通过预测潜在的违规行为。同时，大语言模型(LLMs)擅长将自然语言转换为形式化工件并识别数据中的模式，但它们仍然容易出错且缺乏形式化保证。这篇愿景论文论证了RV和LLMs的共生集成。RV可以作为LLM驱动自主系统的护栏，而LLMs可以通过辅助规范捕获、支持预期推理和帮助处理不确定性来扩展RV的能力。我们概述了这种相互强化与现有调查和路线图的不同之处，讨论了挑战和认证含义，并确定了未来可靠自主性的研究方向。


### 论文摘要

Assuring the safety and trustworthiness of autonomous systems is particularly difficult when learning-enabled components and open environments are involved. Formal methods provide strong guarantees but depend on complete models and static assumptions. Runtime verification (RV) complements them by monitoring executions at run time and, in its predictive variants, by anticipating potential violations. Large language models (LLMs), meanwhile, excel at translating natural language into formal artefacts and recognising patterns in data, yet they remain error-prone and lack formal guarantees. This vision paper argues for a symbiotic integration of RV and LLMs. RV can serve as a guardrail for LLM-driven autonomy, while LLMs can extend RV by assisting specification capture, supporting anticipatory reasoning, and helping to handle uncertainty. We outline how this mutual reinforcement differs from existing surveys and roadmaps, discuss challenges and certification implications, and identify future research directions towards dependable autonomy.

---

## 126. Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition

**论文链接:** [http://arxiv.org/abs/2511.14391v1](http://arxiv.org/abs/2511.14391v1)

**作者:** Fabian Schmidt, Noushiq Mohammed Kayilan Abdul Nazar, Markus Enzweiler, Abhinav Valada

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究提出TLS-Assist框架，一种用于增强基于大型语言模型的自动驾驶代理的模块化冗余层，专注于交通灯和标志识别，显著提高了驾驶性能并减少了违规行为。

### 背景

大型语言模型(LLMs)越来越多地用于自动驾驶的决策和规划，展现出有前途的推理能力和泛化能力，但当前基于LLM的驾驶代理缺乏强制执行交通规则的明确机制，难以可靠检测小型安全关键物体。

### 目的

解决现有LLM驾驶代理在交通规则执行和安全关键物体检测方面的局限性，提高自动驾驶系统的安全性和可靠性。

### 方法

提出TLS-Assist框架，将交通灯和标志的检测结果转换为结构化自然语言消息并注入到LLM输入中，强制模型对安全关键线索的明确关注。该框架即插即用，与模型无关，支持单视图和多视图摄像头设置。

### 主要发现

在CARLA的LangAuto基准测试中，TLS-Assist相对于LMDrive提高了最多14%的驾驶性能，相对于BEVDriver提高了7%，同时减少了交通灯和标志的违规行为。

### 结论

TLS-Assist有效解决了基于LLM的自动驾驶代理在交通规则执行和安全物体检测方面的不足，通过模块化冗余层显著提高了系统的安全性和性能。

### 翻译

大型语言模型(LLMs)越来越多地用于自动驾驶的决策和规划，展现出有前途的推理能力和泛化到各种交通情况的潜力。然而，当前基于LLM的驾驶代理缺乏强制执行交通规则的明确机制，通常难以可靠地检测小型安全关键物体，如交通灯和标志。为了解决这一限制，我们引入了TLS-Assist，这是一个模块化冗余层，通过明确的交通灯和标志识别功能增强了基于LLM的自动驾驶代理。TLS-Assist将检测结果转换为结构化自然语言消息，并注入到LLM输入中，强制对安全关键线索的明确关注。该框架即插即用，与模型无关，支持单视图和多视图摄像头设置。我们在CARLA的LangAuto基准测试的闭环设置中评估了TLS-Assist。结果表明，相对于LMDrive提高了最多14%的驾驶性能，相对于BEVDriver提高了7%，同时减少了交通灯和标志的违规行为。我们在https://github.com/iis-esslingen/TLS-Assist上公开发布了代码和模型。


### 论文摘要

Large Language Models (LLMs) are increasingly used for decision-making and planning in autonomous driving, showing promising reasoning capabilities and potential to generalize across diverse traffic situations. However, current LLM-based driving agents lack explicit mechanisms to enforce traffic rules and often struggle to reliably detect small, safety-critical objects such as traffic lights and signs. To address this limitation, we introduce TLS-Assist, a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition. TLS-Assist converts detections into structured natural language messages that are injected into the LLM input, enforcing explicit attention to safety-critical cues. The framework is plug-and-play, model-agnostic, and supports both single-view and multi-view camera setups. We evaluate TLS-Assist in a closed-loop setup on the LangAuto benchmark in CARLA. The results demonstrate relative driving performance improvements of up to 14% over LMDrive and 7% over BEVDriver, while consistently reducing traffic light and sign infractions. We publicly release the code and models on https://github.com/iis-esslingen/TLS-Assist.

---

## 127. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.14386v1](http://arxiv.org/abs/2511.14386v1)

**作者:** Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文研究了物理对抗样本(PAEs)对自动驾驶中立体匹配模型的影响，提出了一种基于纹理的3D物理对抗攻击方法，通过全局伪装纹理和新的立体匹配渲染模块实现高效且隐蔽的攻击效果。

### 背景

深度神经网络模型在自动驾驶感知中已被证明容易受到对抗样本的攻击，但已知的攻击通常利用2D贴片，并且主要针对单目感知。基于立体的双目深度估计中的物理对抗样本有效性尚未被充分探索。

### 目的

研究物理对抗样本对立体匹配模型的影响，并提出一种针对自动驾驶场景中立体匹配模型的纹理启动物理对抗攻击方法。

### 方法

提出使用全局伪装纹理而非局部2D贴片的3D PAE；提出新的3D立体匹配渲染模块处理相机视差效应；提出新的融合攻击通过细粒度PAE优化将目标无缝融入环境。

### 主要发现

提出的PAEs能够成功欺骗立体模型，产生错误的深度信息，且在隐蔽性和致命性上显著优于现有的隐藏攻击。

### 结论

所提出的3D物理对抗攻击方法通过全局伪装纹理和立体匹配渲染模块，能够在不同视角下保持视觉一致性和攻击有效性，显著提升了攻击的隐蔽性和致命性。

### 翻译

尽管用于实现自动驾驶感知的深度神经网络模型已被证明容易受到对抗样本的攻击，但已知的攻击通常利用2D贴片，并且主要针对单目感知。因此，物理对抗样本(PAEs)在基于立体的双目深度估计中的有效性很大程度上尚未被探索。为此，我们提出了第一种针对自动驾驶背景下立体匹配模型的纹理启动物理对抗攻击。我们的方法采用具有全局伪装纹理的3D PAE，而非基于局部2D贴片的方法，确保立体相机在不同视角下都能保持视觉一致性和攻击有效性。为了应对这些相机的视差效应，我们还提出了一个新的3D立体匹配渲染模块，使PAE能够与双目视觉中的真实世界位置和方向对齐。我们进一步提出了一种新颖的融合攻击，通过细粒度PAE优化将目标无缝融入环境。与无法与背景无缝融合的现有隐藏攻击相比，它在隐蔽性和致命性上有了显著提升。大量评估表明，我们的PAEs能够成功欺骗立体模型，产生错误的深度信息。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决对自动驾驶系统中基于立体匹配的双目深度估计模型的物理对抗攻击问题。这个问题很重要，因为现有的对抗攻击主要针对单目深度估计模型，使用2D贴片，在立体视觉系统中效果有限。而立体视觉系统利用双目视差原理对物理世界有更准确的深度感知，在自动驾驶中越来越重要。如果这些系统可以被有效欺骗，可能导致严重的安全事故，如车辆碰撞或路径规划错误。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：大多数对抗攻击使用2D贴片，主要针对单目深度估计，在立体视觉系统中效果有限。然后借鉴了现有的物理对抗攻击思想，但针对立体视觉的特殊性进行了改进：借鉴了单目深度估计的对抗攻击思想，但扩展到3D空间；借鉴了3D物体检测和渲染技术，用于精确对齐；借鉴了期望变换(EoT)技术提高对抗样本的鲁棒性。作者设计了一个新的立体对齐3D渲染模块，确保生成的对抗纹理在双目视角下保持一致性，并提出了'合并攻击'的新方法，通过精细化的纹理优化使目标物体与背景无缝融合。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D全局伪装纹理而非局部2D贴片，确保在不同视角下都能有效攻击；设计立体对齐的3D渲染模块，使对抗纹理与真实世界物体位置和方向对齐；提出合并攻击方法，通过精细化的对抗纹理优化使目标物体无缝融入环境。整体实现流程包括：1) 问题定义：生成可物理部署的对抗纹理；2) 立体对齐3D渲染：使用3D物体检测获取目标物体姿态和尺寸，参数化立体视角配置；3) 合并攻击纹理生成：边界深度提取、区域分割、纹理优化；4) 损失函数设计：合并损失+非可打印性损失+总变化损失；5) 训练和优化：使用可微分渲染管道和Adam优化器训练对抗纹理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次提出针对自动驾驶场景中立体匹配模型的纹理型物理对抗攻击，使用3D全局伪装纹理；2) 设计了新的立体对齐3D渲染模块，确保对抗纹理在物理世界中可实现；3) 提出了新颖的合并攻击方法，实现目标物体与背景的无缝融合。相比之前的工作，不同之处在于：攻击方式从2D贴片变为3D全局纹理；适用场景从单目深度估计扩展到立体视觉系统；攻击效果在各种视角下都保持有效；隐蔽性显著提高，实现了与背景的无缝融合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文首次提出了一种针对自动驾驶立体视觉系统的3D纹理型物理对抗攻击方法，通过立体对齐的3D渲染和精细化的纹理优化，实现了目标物体与背景的无缝融合，有效欺骗了立体深度估计模型并引发了严重的系统级安全问题。'}


### 论文摘要

Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

---

## 128. O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model

**论文链接:** [http://arxiv.org/abs/2511.14368v1](http://arxiv.org/abs/2511.14368v1)

**作者:** Rishi Gupta, Mukilan Karuppasamy, Shyam Marjit, Aditay Tripathi, Anirban Chakraborty

**发布时间:** 2025-11-18

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该研究提出了一种新的数据集和模型来解决大型视觉语言模型在理解手绘草图方面的局限性，显著提升了模型在多种视觉任务上的表现。

### 背景

大型视觉语言模型在现实世界应用中日益普及，但在解释抽象视觉输入方面能力有限，特别是难以理解手绘草图这种难以用文本描述的概念的直观表达方式。

### 目的

解决大型视觉语言模型在理解手绘草图方面的瓶颈问题，提高模型对抽象视觉输入的理解能力。

### 方法

提出两个关键贡献：一是创建了一个新的、大规模的图像-草图-指令三元组数据集，用于促进预训练和指令微调；二是基于该数据集训练了一个名为O3SLM的大型视觉语言模型。

### 主要发现

在多种基于草图的任务（包括目标定位、计数、图像检索和视觉问答）上评估表明，O3SLM取得了最先进的性能，显著优于现有的大型视觉语言模型在草图理解和推理方面的表现。

### 结论

通过新的数据集和O3SLM模型，有效解决了大型视觉语言模型在理解抽象视觉输入方面的局限性，为未来研究提供了新的方向。

### 翻译

虽然大型视觉语言模型在现实世界应用中日益部署，但它们解释抽象视觉输入的能力仍然有限。具体来说，它们难以理解手绘草图，而这种模态为表达难以用文本描述的概念提供了直观方式。我们确定的主要瓶颈是缺乏一个大规模数据集，该数据集能够同时建模草图、逼真图像和相应的自然语言指令。为此，我们提出了两个关键贡献：(1) 一个新的、大规模的图像-草图-指令三元组数据集，旨在促进预训练和指令微调；(2) O3SLM，一个在该数据集上训练的大型视觉语言模型。在多个基于草图的任务上的全面评估：(a) 目标定位，(b) 计数，(c) 图像检索（即SBIR和细粒度SBIR），以及(d) 视觉问答(VQA)；同时结合三个现有的草图数据集，即QuickDraw!、Sketchy和Tu Berlin，以及我们生成的SketchVCL数据集，表明O3SLM取得了最先进的性能，在草图理解和推理方面显著优于现有的大型视觉语言模型。


### 论文摘要

While Large Vision Language Models (LVLMs) are increasingly deployed in real-world applications, their ability to interpret abstract visual inputs remains limited. Specifically, they struggle to comprehend hand-drawn sketches, a modality that offers an intuitive means of expressing concepts that are difficult to describe textually. We identify the primary bottleneck as the absence of a large-scale dataset that jointly models sketches, photorealistic images, and corresponding natural language instructions. To address this, we present two key contributions: (1) a new, large-scale dataset of image-sketch-instruction triplets designed to facilitate both pretraining and instruction tuning, and (2) O3SLM, an LVLM trained on this dataset. Comprehensive evaluations on multiple sketch-based tasks: (a) object localization, (b) counting, (c) image retrieval i.e., (SBIR and fine-grained SBIR), and (d) visual question answering (VQA); while incorporating the three existing sketch datasets, namely QuickDraw!, Sketchy, and Tu Berlin, along with our generated SketchVCL dataset, show that O3SLM achieves state-of-the-art performance, substantially outperforming existing LVLMs in sketch comprehension and reasoning.

---

## 129. Silhouette-to-Contour Registration: Aligning Intraoral Scan Models with Cephalometric Radiographs

**论文链接:** [http://arxiv.org/abs/2511.14343v1](http://arxiv.org/abs/2511.14343v1)

**作者:** Yiyi Miao, Taoyu Wu, Ji Jiang, Tong Chen, Zhe Tang, Zhengyong Jiang, Angelos Stefanidis, Limin Yu, Jionglong Su

**发布时间:** 2025-11-18

### GPT解析

### 总结

论文提出DentalSCR方法，一种姿势稳定、轮廓引导的框架，用于解决口腔内扫描模型与侧位头颅X线片之间的3D-2D配准问题，克服传统基于强度配准方法的局限性，实现准确且可解释的轮廓到轮廓配准。

### 背景

在正畸诊断中，口腔内扫描模型与侧位头颅X线片之间的可靠3D-2D配准至关重要。然而传统基于强度的配准方法在真实临床条件下表现不佳，因为头颅X线片存在投影放大、几何失真、低对比度牙冠以及获取相关的变化等因素，导致配准不稳定或解剖学上不合理。

### 目的

解决传统配准方法在临床条件下的局限性，开发一种稳健的3D-2D配准框架，能够处理真实世界头颅X线片的挑战，提供高保真、临床可检查的配准结果。

### 方法

1) 构建U-Midline牙轴建立统一的跨弓解剖坐标系，稳定初始化并标准化投影几何；2) 使用基于表面的DRR公式结合冠状轴视角和高斯飞溅技术生成类似X光片的投影；3) 将配准公式化为2D相似性变换，在分层粗到细的调度下使用对称双向Chamfer距离进行优化，实现大捕获范围和亚像素级轮廓一致性。

### 主要发现

在34个专家标注的临床病例上评估显示：标志点误差显著减少，特别是在后牙区域；下颌骨上的分布更加紧密；曲线级别的Chamfer距离低且Hausdorff距离受控，表明配准质量高且稳定。

### 结论

DentalSCR能够稳健地处理真实世界的头颅X线片，并提供高保真、临床可检查的3D-2D配准效果，显著优于传统基线方法，解决了临床条件下的配准挑战。

### 翻译

口腔内扫描模型与侧位头颅X线片之间的可靠3D-2D配准对正畸诊断至关重要，然而传统的基于强度的配准方法在真实临床条件下表现不佳，在这些条件下，头颅X线片表现出投影放大、几何失真、低对比度牙冠以及获取相关的变化。这些因素阻碍了基于外观的相似性度量的稳定性，并常常导致收敛失败或解剖学上不合理的配准。为了解决这些局限性，我们提出了DentalSCR，一种姿势稳定、轮廓引导的框架，用于准确且可解释的轮廓到轮廓配准。我们的方法首先构建U-Midline牙轴以建立统一的跨弓解剖坐标系，从而稳定初始化并标准化各病例的投影几何。使用这个参考框架，我们通过基于表面的DRR公式结合冠状轴视角和高斯飞溅技术生成类似X光片的投影，保留临床源-物体-探测器放大并强调外部轮廓。然后，配准被公式化为2D相似性变换，在分层粗到细的调度下使用对称双向Chamfer距离进行优化，实现大捕获范围和亚像素级轮廓一致性。我们在34个专家标注的临床病例上评估了DentalSCR。实验结果表明，标志点误差显著减少，特别是在后牙区域，下颌骨上的分布更加紧密，曲线级别的Chamfer距离低且Hausdorff距离受控。这些发现表明，DentalSCR能够稳健地处理真实世界的头颅X线片，并提供高保真、临床可检查的3D-2D配准效果，优于传统基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决口腔内扫描模型与头颅侧位定位放射线照片之间的3D-2D对齐问题。这个问题在正畸诊断中非常重要，因为可靠的配准有助于将3D牙科模型整合到头颅测量坐标系中，实现可重复测量和一致的3D-2D可视化，但临床X射线的投影放大、几何失真以及牙冠轮廓的低对比度等因素传统方法难以处理。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别出现有方法的局限性：基于强度的方法对临床条件敏感，基于几何的方法在细小结构中不稳定。他们采用'分析-合成'范式，构建统一坐标系(UMDA)稳定初始化，模拟真实放射线几何生成投影，并使用轮廓距离场作为优化目标。方法借鉴了医学影像配准的三种范式、牙科几何的先验知识、头颅侧位放射线照片的自动分析技术以及Chamfer距离等几何度量方法。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用轮廓到轮廓的几何配准而非强度匹配，通过统一坐标系稳定初始化，模拟真实放射线几何，并使用对称距离度量。整体流程包括：1)构建U-中线牙轴(UMDA)建立解剖坐标系；2)使用冠状轴透视和高斯飞溅渲染生成类似放射线的投影；3)通过2D相似变换和对称Chamfer距离进行轮廓到轮廓的配准，采用分层粗到细优化策略提高精度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入U-中线牙轴(UMDA)稳定几何优化；2)模拟真实放射线几何缩小合成与获取图像差距；3)使用对称双向Chamfer距离和分层优化平衡捕获范围与精度。相比之前的工作，该方法不依赖图像强度而专注几何匹配，显式建模临床放射线几何，使用统一坐标系提高一致性，避免单向匹配偏差，并通过分层优化提高鲁棒性和准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DentalSCR通过引入统一解剖坐标系、模拟真实放射线几何和对称轮廓距离度量，实现了口腔内扫描模型与头颅侧位放射线照片之间的高精度、鲁棒配准，显著提升了正畸诊断中模型与影像的对齐准确性。'}


### 论文摘要

Reliable 3D-2D alignment between intraoral scan (IOS) models and lateral cephalometric radiographs is critical for orthodontic diagnosis, yet conventional intensity-driven registration methods struggle under real clinical conditions, where cephalograms exhibit projective magnification, geometric distortion, low-contrast dental crowns, and acquisition-dependent variation. These factors hinder the stability of appearance-based similarity metrics and often lead to convergence failures or anatomically implausible alignments. To address these limitations, we propose DentalSCR, a pose-stable, contour-guided framework for accurate and interpretable silhouette-to-contour registration. Our method first constructs a U-Midline Dental Axis (UMDA) to establish a unified cross-arch anatomical coordinate system, thereby stabilizing initialization and standardizing projection geometry across cases. Using this reference frame, we generate radiograph-like projections via a surface-based DRR formulation with coronal-axis perspective and Gaussian splatting, which preserves clinical source-object-detector magnification and emphasizes external silhouettes. Registration is then formulated as a 2D similarity transform optimized with a symmetric bidirectional Chamfer distance under a hierarchical coarse-to-fine schedule, enabling both large capture range and subpixel-level contour agreement. We evaluate DentalSCR on 34 expert-annotated clinical cases. Experimental results demonstrate substantial reductions in landmark error-particularly at posterior teeth-tighter dispersion on the lower jaw, and low Chamfer and controlled Hausdorff distances at the curve level. These findings indicate that DentalSCR robustly handles real-world cephalograms and delivers high-fidelity, clinically inspectable 3D--2D alignment, outperforming conventional baselines.

---

## 130. When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling

**论文链接:** [http://arxiv.org/abs/2511.14334v1](http://arxiv.org/abs/2511.14334v1)

**作者:** Alessio Pellegrino, Jacopo Mauro

**发布时间:** 2025-11-18

### GPT解析

### 总结

本研究探讨了大型语言模型在自动生成约束编程模型方面的能力及其局限性，发现其成功可能主要来自数据污染而非真正的推理能力。

### 背景

优化和约束编程领域的长期目标是能够用自然语言描述问题并自动获得可执行、高效的模型。大型语言模型似乎使这一愿景更接近实现，在自动生成经典基准模型方面显示出令人印象深刻的结果。

### 目的

检验大型语言模型的明显成功是否主要来自于数据污染而非真正的推理能力，即许多标准CP问题可能包含在这些模型的训练数据中。

### 方法

系统地重新表述和扰动一组著名的CSPLib问题，保留结构同时修改上下文并引入误导性元素，然后比较三个代表性LLMs在原始和修改描述下生成的模型。

### 主要发现

大型语言模型可以生成语法有效且语义合理的模型，但在上下文和语言变化下，它们的性能急剧下降，显示出浅层理解和对措辞的敏感性。

### 结论

大型语言模型在自动生成约束编程模型方面的能力可能主要依赖于训练数据中的相似问题，而非真正的推理能力，当问题表述发生变化时，性能会显著下降。

### 翻译

优化和约束编程领域长期以来的目标是能够用自然语言描述问题并自动获得可执行、高效的模型。大型语言模型似乎使这一愿景更接近实现，在自动生成经典基准模型方面显示出令人印象深刻的结果。然而，这种明显的成功可能主要来自于数据污染而非真正的推理能力：许多标准CP问题可能包含在这些模型的训练数据中。为了检验这一假设，研究人员系统地重新表述和扰动了一组著名的CSPLib问题，保留了它们的结构同时修改了上下文并引入了误导性元素。然后，他们比较了三个代表性LLMs在原始和修改描述下生成的模型。他们的定性分析表明，虽然LLMs可以生成语法有效且语义合理的模型，但在上下文和语言变化下，它们的性能急剧下降，显示出浅层理解和对措辞的敏感性。


### 论文摘要

One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording.

---

## 131. Dual-Variable Force Characterisation method for Human-Robot Interaction in Wearable Robotics

**论文链接:** [http://arxiv.org/abs/2511.14327v1](http://arxiv.org/abs/2511.14327v1)

**作者:** Felipe Ballen-Moreno, Pasquale Ferrentino, Milan Amighi, Bram Vanderborght, Tom Verstraten

**发布时间:** 2025-11-18

**备注:** 36 pages, 10 figures, submitted and under-review in Journal of the Mechanical Behavior of Biomedical Materials

### GPT解析

### 总结

该研究提出了一种双变量表征方法，用于改进可穿戴机器人与人体物理相互作用的建模和评估，解决了现有单变量方法的局限性。

### 背景

可穿戴机器人与人的物理相互作用对安全性和舒适性至关重要，但该相互作用在运动复杂性和软组织非线性行为方面具有挑战性。现有表征方法通常只依赖单一自由度的单一拟合变量，限制了多自由度交互场景的应用。

### 目的

开发一种双变量表征方法，包括法向力和切向力，以识别可靠的材料参数并评估单变量拟合对力和力矩响应的影响，从而提高可穿戴机器人交互模型的准确性。

### 方法

引入涉及法向力和切向力的双变量表征方法，通过分析不同场景和材料模型下的归一化均方误差(NMSE)，评估双变量表征的优势和必要性。

### 主要发现

分析表明，在表征过程中纳入两个变量（法向力和切向力）能够显著提高模型准确性，通过比较不同场景下的归一化均方误差证明了双变量方法的优势。

### 结论

该双变量表征方法为模拟用户与可穿戴机器人物理相互作用中的袖带和人体肢体提供了更可靠的框架，有助于开发更安全、舒适的可穿戴机器人系统。

### 翻译

理解可穿戴机器人的物理相互作用对于确保安全性和舒适性至关重要。然而，这种相互作用在两个关键方面很复杂：(1) 涉及的运动，以及(2) 软组织的非线性行为。已经采取了多种方法来更好地理解这种相互作用并改善物理界面或袖带的定量指标。由于这两个主题密切相关，有限建模和软组织表征为理解袖带引起的压力分布和剪切应力提供了有价值的见解。尽管如此，现有的表征方法通常只依赖于沿单一自由度的单一拟合变量，考虑到与可穿戴机器人的相互作用通常涉及多个自由度，这限制了它们的适用性。为解决这一局限性，本研究引入了一种双变量表征方法，包括法向力和切向力，旨在识别可靠的材料参数并评估单变量拟合对力和力矩响应的影响。通过分析不同场景和材料模型下的归一化均方误差(NMSE)，该方法证明了在表征过程中纳入两个变量的重要性，为尽可能接近水平的模拟提供了基础，重点关注用户与可穿戴机器人物理相互作用中的袖带和人体肢体。


### 论文摘要

Understanding the physical interaction with wearable robots is essential to ensure safety and comfort. However, this interaction is complex in two key aspects: (1) the motion involved, and (2) the non-linear behaviour of soft tissues. Multiple approaches have been undertaken to better understand this interaction and to improve the quantitative metrics of physical interfaces or cuffs. As these two topics are closely interrelated, finite modelling and soft tissue characterisation offer valuable insights into pressure distribution and shear stress induced by the cuff. Nevertheless, current characterisation methods typically rely on a single fitting variable along one degree of freedom, which limits their applicability, given that interactions with wearable robots often involve multiple degrees of freedom. To address this limitation, this work introduces a dual-variable characterisation method, involving normal and tangential forces, aimed at identifying reliable material parameters and evaluating the impact of single-variable fitting on force and torque responses. This method demonstrates the importance of incorporating two variables into the characterisation process by analysing the normalized mean square error (NMSE) across different scenarios and material models, providing a foundation for simulation at the closest possible level, with a focus on the cuff and the human limb involved in the physical interaction between the user and the wearable robot.

---

## 132. SAM-Fed: SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.14302v1](http://arxiv.org/abs/2511.14302v1)

**作者:** Sahar Nasirihaghighi, Negin Ghamsarian, Yiping Li, Marcel Breeuwer, Raphael Sznitman, Klaus Schoeffmann

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了SAM-Fed框架，一种利用高容量分割基础模型指导轻量级客户端训练的联邦半监督学习方法，通过双重知识蒸馏和自适应一致性机制改进像素级监督，在医学图像分割任务中表现优异。

### 背景

医学图像分割在临床上非常重要，但数据隐私和专家标注成本限制了标记数据的可用性。联邦半监督学习(FSSL)提供了解决方案，但面临两个挑战：伪标签可靠性依赖于局部模型的强度，而客户端设备由于计算资源有限，通常需要紧凑或异构架构，这降低了伪标签的质量和稳定性。

### 目的

开发一种联邦半监督框架，解决现有方法在医学图像分割中面临的伪标签质量和稳定性问题，同时适应客户端设备的计算限制。

### 方法

提出SAM-Fed框架，利用高容量分割基础模型指导轻量级客户端训练，结合双重知识蒸馏和自适应一致性机制来优化像素级监督。

### 主要发现

在皮肤病变和息肉分割任务上，无论是在同质还是异质设置中，SAM-Fed都持续优于现有的最先进FSSL方法。

### 结论

SAM-Fed有效解决了联邦半监督学习中伪标签质量和稳定性问题，同时适应了客户端设备的计算限制，是一种有前景的医学图像分割解决方案。

### 翻译

医学图像分割在临床上很重要，但数据隐私和专家标注的成本限制了标记数据的可用性。联邦半监督学习(FSSL)提供了解决方案，但面临两个挑战：伪标签可靠性依赖于局部模型的强度，而客户端设备由于计算资源有限，通常需要紧凑或异构架构。这些限制降低了伪标签的质量和稳定性，而大型模型虽然更准确，却无法在客户端设备上进行训练或用于常规推理。我们提出了SAM-Fed，一种联邦半监督框架，利用高容量分割基础模型在训练过程中指导轻量级客户端。SAM-Fed结合了双重知识蒸馏和自适应一致性机制来改进像素级监督。在皮肤病变和息肉分割任务上的实验，无论是在同质还是异质设置中，都表明SAM-Fed持续优于最先进的FSSL方法。


### 论文摘要

Medical image segmentation is clinically important, yet data privacy and the cost of expert annotation limit the availability of labeled data. Federated semi-supervised learning (FSSL) offers a solution but faces two challenges: pseudo-label reliability depends on the strength of local models, and client devices often require compact or heterogeneous architectures due to limited computational resources. These constraints reduce the quality and stability of pseudo-labels, while large models, though more accurate, cannot be trained or used for routine inference on client devices. We propose SAM-Fed, a federated semi-supervised framework that leverages a high-capacity segmentation foundation model to guide lightweight clients during training. SAM-Fed combines dual knowledge distillation with an adaptive agreement mechanism to refine pixel-level supervision. Experiments on skin lesion and polyp segmentation across homogeneous and heterogeneous settings show that SAM-Fed consistently outperforms state-of-the-art FSSL methods.

---

## 133. Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion

**论文链接:** [http://arxiv.org/abs/2511.14301v1](http://arxiv.org/abs/2511.14301v1)

**作者:** Eric Xue, Ruiyi Zhang, Zijun Zhang, Pengtao Xie

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了SteganoBackdoor，一种针对Transformer模型的隐蔽后门攻击方法，利用自然语言隐写术技术，将语义触发器转换为隐写载体，实现高攻击成功率且难以被防御系统检测。

### 背景

Transformer模型作为NLP基础，易受通过污染数据引入的后门攻击。当前研究集中在风格化伪影触发器攻击，而非更实用的语义触发器攻击，导致防御研究与实际威胁模型脱节。

### 目的

开发一种隐蔽的后门攻击方法，使模型响应语义触发器（如特定名称或实体），同时保持高攻击成功率和强规避能力，揭示当前防御系统的盲点。

### 方法

SteganoBackdoor利用自然语言隐写术的无害特性，采用梯度引导的数据优化过程，将语义触发器种子转换为隐写载体。这些载体嵌入高后门负载，保持文本流畅性，且与原始触发器没有表示相似性。

### 主要发现

在多种实验环境下，SteganoBackdoor以比先前方法低一个数量级的数据污染率实现了超过99%的攻击成功率，并对全面的数据级防御保持无与伦比的规避能力。

### 结论

SteganoBackdoor揭示了一种实用且隐蔽的攻击方式，突显了当前防御系统中的紧急盲点，强调了对抗数据防御和真实世界威胁建模的紧迫性。

### 翻译

Transformer模型是自然语言处理(NLP)应用的基础，但仍然容易受到通过污染数据引入的后门攻击，这些攻击在训练过程中植入了隐藏行为。为了增强防止此类妥协的能力，最近的研究集中在设计越来越隐蔽的攻击，以测试现有防御，将后门行为与风格化伪载或令牌级扰动触发器配对。然而，这种趋势分散了对更困难且更现实的案例的关注：使模型响应语义触发器，如特定名称或实体，成功的后门可能操纵部署系统中与真实人物或事件相关的输出。受这种日益扩大的差距的激励，我们引入了SteganoBackdoor，将隐蔽技术带回与实用威胁模型一致。利用自然语言隐写术的无害特性，SteganoBackdoor应用梯度引导的数据优化过程，将语义触发器种子转换为嵌入高后门负载的隐写载体，保持流畅性，并且与触发器没有表示相似性。在各种实验设置中，SteganoBackdoor以比先前方法低一个数量级的数据污染率实现了超过99%的攻击成功率，同时对全面的数据级防御保持无与伦比的规避能力。通过揭示这种实用且隐蔽的攻击，SteganoBackdoor突显了当前防御中的一个紧急盲点，并要求立即关注对抗数据防御和真实世界威胁建模。


### 论文摘要

Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.

---

## 134. Dental3R: Geometry-Aware Pairing for Intraoral 3D Reconstruction from Sparse-View Photographs

**论文链接:** [http://arxiv.org/abs/2511.14315v1](http://arxiv.org/abs/2511.14315v1)

**作者:** Yiyi Miao, Taoyu Wu, Tong Chen, Ji Jiang, Zhe Tang, Zhengyong Jiang, Angelos Stefanidis, Limin Yu, Jionglong Su

**发布时间:** 2025-11-18

### GPT解析

### 总结

该研究提出了Dental3R，一种无需姿态估计的图引导管道，能够从稀疏口腔照片中进行稳健、高保真的3D重建，解决了传统口腔扫描在远程正畸中不可行的问题。

### 背景

口腔内3D重建是数字正畸的基础，但传统口腔内扫描方法无法用于远程正畸。3D高斯溅射(3DGS)虽在新型视图合成方面有潜力，但应用于标准临床三联（未摆位的前部和双侧颊部照片）时面临挑战，主要因为口腔内设置中大的视基线、不一致光照和镜面表面会导致同时姿态和几何估计不稳定。

### 目的

解决稀疏视图光度监督导致的频率偏差问题，避免过度平滑的重建失去关键诊断细节；开发一种能够从稀疏口腔照片中生成高质量3D重建的方法，适用于远程正畸场景。

### 方法

提出Dental3R方法，包含几何感知配对策略(GAPS)来智能选择高价值图像对的紧凑子图，提高几何初始化稳定性并减少内存使用；使用小波正则化目标训练3DGS模型，通过离散小波变换强制限制保真度，保留精细釉质边界和邻面边缘，同时抑制高频伪影。

### 主要发现

在950个临床病例的大规模数据集和195个病例的额外基于视频的测试集上验证了该方法；实验结果表明Dental3R能够有效处理稀疏、未摆位的输入；在牙齿咬合可视化的新型视图合成质量方面优于现有最先进方法。

### 结论

Dental3R是一种有效的方法，可以从稀疏口腔照片中进行高质量3D重建，特别适用于远程正畸应用，能够克服传统方法的局限性并提供更准确的诊断信息。

### 翻译

口腔内3D重建是数字正畸的基础，然而传统的口腔内扫描方法对于远程正畸来说不可行，远程正畸通常依赖于稀疏的智能手机图像。虽然3D高斯溅射(3DGS)在新型视图合成方面显示出潜力，但其应用于标准临床三联（未摆位的前部和双侧颊部照片）具有挑战性。口腔内设置中常见的大的视基线、不一致光照和镜面表面可以同时破坏姿态和几何估计的稳定性。此外，稀疏视图光度监督常常导致频率偏差，产生过度平滑的重建，失去关键诊断细节。为了解决这些限制，我们提出了Dental3R，一种无姿态、图引导的管道，用于从稀疏口腔照片中进行稳健、高保真的重建。我们的方法首先构建几何感知配对策略(GAPS)，智能选择高价值图像对的紧凑子图。GAPS专注于对应匹配，从而提高几何初始化的稳定性并减少内存使用。基于恢复的姿态和点云，我们使用小波正则化目标训练3DGS模型。通过使用离散小波变换强制限制保真度，我们的方法保留了精细的釉质边界和邻面边缘，同时抑制高频伪影。我们在一个包含950个临床病例的大规模数据集和一个包含195个病例的额外基于视频的测试集上验证了我们的方法。实验结果表明，Dental3R能够有效处理稀疏、未摆位的输入，并在牙齿咬合可视化的新型视图合成质量方面实现了优于最先进方法的效果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏视角的口腔内照片中进行高质量3D重建的问题。这个问题在现实中很重要，因为传统口腔扫描设备需要专业设备和操作环境，无法满足远程正畸学的需求。而远程正畸学依赖于智能手机拍摄的稀疏照片，现有方法在处理这些照片时面临大视角基线、光照不一致和镜面表面等问题，导致重建结果不稳定且丢失重要诊断细节。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，特别是3D高斯溅射(3DGS)在处理口腔照片时的三个主要失败模式：内存计算量大、相机恢复不稳定和频率偏差导致细节丢失。基于这些分析，作者设计了一个无需姿态的图引导流程，借鉴了DUSt3R（用于从两个未标记视图回归密集点图）和3DGS（用于高效辐射场合成）等技术，但针对口腔环境进行了专门改进，提出了几何感知配对策略(GAPS)和小波正则化方法来解决现有问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'Dental3R的核心思想包括两部分：1) 几何感知配对策略(GAPS)，通过构建混合顺序几何视图图并选择最优图像对，提高姿态稳定性和减少内存使用；2) 小波正则化目标，通过离散小波变换保留精细的釉质边界和邻间边缘，同时抑制高频伪影。整体流程是：首先用GAPS策略生成图像对，然后利用立体密集重建模型回归点云和相机姿态，接着用点云初始化3D高斯，最后在优化过程中结合小波约束以确保几何一致性和细节保留。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 几何感知配对策略(GAPS)，优化图像对选择以平衡局部可靠性和全局刚性；2) 小波正则化目标，替代传统光度监督以保留重要诊断细节；3) 专门针对口腔环境的设计，处理大视角基线、光照不一致和镜面表面等挑战。相比之前的工作，Dental3R无需已知相机姿态，通过GAPS策略解决了内存和稳定性问题，使用小波正则化解决了频率偏差问题，特别适合稀疏视角的口腔重建场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Dental3R提出了一种创新的几何感知配对策略和小波正则化方法，实现了从稀疏视角的口腔内照片中进行高质量、高保真的3D重建，为远程正畸学提供了实用解决方案。'}


### 论文摘要

Intraoral 3D reconstruction is fundamental to digital orthodontics, yet conventional methods like intraoral scanning are inaccessible for remote tele-orthodontics, which typically relies on sparse smartphone imagery. While 3D Gaussian Splatting (3DGS) shows promise for novel view synthesis, its application to the standard clinical triad of unposed anterior and bilateral buccal photographs is challenging. The large view baselines, inconsistent illumination, and specular surfaces common in intraoral settings can destabilize simultaneous pose and geometry estimation. Furthermore, sparse-view photometric supervision often induces a frequency bias, leading to over-smoothed reconstructions that lose critical diagnostic details. To address these limitations, we propose \textbf{Dental3R}, a pose-free, graph-guided pipeline for robust, high-fidelity reconstruction from sparse intraoral photographs. Our method first constructs a Geometry-Aware Pairing Strategy (GAPS) to intelligently select a compact subgraph of high-value image pairs. The GAPS focuses on correspondence matching, thereby improving the stability of the geometry initialization and reducing memory usage. Building on the recovered poses and point cloud, we train the 3DGS model with a wavelet-regularized objective. By enforcing band-limited fidelity using a discrete wavelet transform, our approach preserves fine enamel boundaries and interproximal edges while suppressing high-frequency artifacts. We validate our approach on a large-scale dataset of 950 clinical cases and an additional video-based test set of 195 cases. Experimental results demonstrate that Dental3R effectively handles sparse, unposed inputs and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art methods.

---

## 135. NeuralSSD: A Neural Solver for Signed Distance Surface Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.14283v1](http://arxiv.org/abs/2511.14283v1)

**作者:** Zi-Chen Xi, Jiahui Huang, Hao-Xiang Chen, Francis Williams, Qun-Ce Xu, Tai-Jiang Mu, Shi-Min Hu

**发布时间:** 2025-11-18

**备注:** Under review

### GPT解析

### 总结

本文提出了一种名为NeuralSSD的广义方法，用于从点云数据重建三维隐式表面。该方法通过新的能量方程和卷积网络实现了高质量的表面重建，并在多个数据集上取得了最先进的结果。

### 背景

现有的隐式场参数化方法缺乏确保表面与输入点云数据紧密贴合的明确机制，影响了重建表面的质量。隐式方法因其能够准确表示形状以及在处理拓扑变化时的鲁棒性而被优先选择。

### 目的

开发一种能够从点云数据重建更高质量和更准确表面的方法，解决现有隐式场参数化的局限性，确保重建表面紧密贴合原始输入点。

### 方法

基于神经Galerkin方法提出NeuralSSD求解器，引入一种平衡点云信息可靠性的新型能量方程，以及一种学习三维信息的新型卷积网络，以实现更好的优化结果。

### 主要发现

NeuralSSD能够从点云中推断有价值的归纳偏置，实现高度准确和稳定的表面重建。在各种具有挑战性的数据集上评估，包括ShapeNet和Matterport数据集，该方法在表面重建精度和泛化能力方面取得了最先进的结果。

### 结论

NeuralSSD是一种有效的三维表面重建方法，通过结合新型能量方程和卷积网络，解决了现有方法的局限性，实现了高质量的表面重建，具有良好的准确性和稳定性。

### 翻译

我们提出了一种广义方法NeuralSSD，用于从广泛可用的点云数据重建三维隐式表面。NeuralSSD是基于神经Galerkin方法的求解器，旨在从输入点云中重建更高质量和更准确的表面。由于隐式方法能够准确表示形状以及在处理拓扑变化时的鲁棒性，因此被优先选择。然而，现有隐式场的参数化缺乏确保表面与输入数据紧密贴合的明确机制。为解决这个问题，我们提出了一种平衡点云信息可靠性的新型能量方程。此外，我们引入了一种学习三维信息的新型卷积网络，以实现更好的优化结果。这种方法确保重建的表面紧密贴合原始输入点，并从点云中推断出有价值的归纳偏置，从而实现高度准确和稳定的表面重建。NeuralSSD在各种具有挑战性的数据集上进行了评估，包括ShapeNet和Matterport数据集，并在表面重建精度和泛化能力方面取得了最先进的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从点云数据重建高质量三维隐式表面的问题。这个问题在现实中很重要，因为点云数据广泛用于激光扫描、深度相机等场景，但实际采集的点云通常是稀疏、有噪声且不完整的。高质量的表面重建对机器人导航、医学成像、虚拟现实等应用至关重要，直接影响下游任务和用户体验，尤其是在传感器受限、存在遮挡和数据不完整的情况下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统SDF/变分方法使用手工正则化，难以调整且不适应数据；学习方法要么需要场景特定优化，要么缺乏保证紧贴输入点的机制。作者借鉴了神经Galerkin方法、SSD和SPSR的工作，设计了一个统一的混合框架，结合自适应稀疏卷积网络和闭式SDF求解器。通过多层卷积处理非均匀点云分布，并引入点-体素注意力机制保留细粒度几何细节，从而平衡了数据保真度和学习到的先验知识。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合神经网络的表示能力和变分优化的精确拟合能力，通过一个混合框架重建高质量三维表面。整体流程包括：1)接收原始点云输入；2)构建多尺度稀疏体素层次结构；3)使用自适应稀疏卷积网络预测空间变化的基函数和法线，应用多层卷积和点-体素注意力机制；4)通过闭式求解器解决变分问题，计算基函数系数；5)使用双marching cubes等技术从隐式表面表示中构建网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的混合框架，结合自适应稀疏卷积预测器和闭式SDF求解器；2)新型能量公式，结合法线对齐、Hessian正则化和点约束；3)自适应稀疏神经网络，在多尺度体素支架上推断法线和基函数；4)多层卷积机制处理非均匀点云分布；5)点-体素注意力机制保留细粒度几何细节。相比传统方法，NeuralSSD使用学习先验和数据自适应正则化；相比纯学习方法，通过变分优化强制执行点级保真度；相比其他混合方法，使用基于Hessian的正则化器捕获更详细的表面，同时保持高效性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NeuralSSD通过结合神经网络的表示能力和变分优化的精确拟合，提出了一种统一的混合框架，能够从稀疏、有噪声的点云中重建高质量、防水且保留细节的三维表面，显著提高了表面重建的准确性和鲁棒性。'}


### 论文摘要

We proposed a generalized method, NeuralSSD, for reconstructing a 3D implicit surface from the widely-available point cloud data. NeuralSSD is a solver-based on the neural Galerkin method, aimed at reconstructing higher-quality and accurate surfaces from input point clouds. Implicit method is preferred due to its ability to accurately represent shapes and its robustness in handling topological changes. However, existing parameterizations of implicit fields lack explicit mechanisms to ensure a tight fit between the surface and input data. To address this, we propose a novel energy equation that balances the reliability of point cloud information. Additionally, we introduce a new convolutional network that learns three-dimensional information to achieve superior optimization results. This approach ensures that the reconstructed surface closely adheres to the raw input points and infers valuable inductive biases from point clouds, resulting in a highly accurate and stable surface reconstruction. NeuralSSD is evaluated on a variety of challenging datasets, including the ShapeNet and Matterport datasets, and achieves state-of-the-art results in terms of both surface reconstruction accuracy and generalizability.

---

## 136. ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes

**论文链接:** [http://arxiv.org/abs/2511.12977v2](http://arxiv.org/abs/2511.12977v2)

**作者:** Yixuan Yang, Luyang Xie, Zhen Luo, Zixiang Zhao, Tongsheng Ding, Mingqi Gao, Feng Zheng

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了ArtiWorld，一个能够从场景描述中自动识别可关节化物体并转换为可交互URDF模型的管道，无需手动转换，节省劳动力和成本。

### 背景

构建交互式模拟器和可扩展的机器人学习环境需要大量关节化资产，但大多数现有3D资产是刚性的，手动转换为关节化物体极其耗费劳动力和成本。

### 目的

自动识别场景中的可关节化物体并直接将其转换为关节化资产。

### 方法

提出ArtiWorld场景感知管道，从文本场景描述中定位候选可关节化物体并重建保留原始几何形状的URDF模型。核心是Arti4URDF，利用3D点云、大型语言模型先验知识和面向URDF的提示设计，快速将刚性物体转换为交互式关节化物体。

### 主要发现

在3D模拟物体、完整3D模拟场景和真实世界扫描场景三个级别评估中，ArtiWorld始终优于现有方法，取得最先进性能，同时保留物体几何形状并正确捕获交互性。

### 结论

为直接从现有3D资产构建交互式、机器人就绪的模拟环境提供了实际路径。

### 翻译

构建交互式模拟器和可扩展的机器人学习环境需要大量关节化资产。然而，大多数模拟中的现有3D资产是刚性的，手动将它们转换为关节化物体极其耗费劳动力和成本。这自然引出了一个问题：我们能否自动识别场景中的可关节化物体并直接将其转换为关节化资产？在本文中，我们提出了ArtiWorld，一个场景感知的管道，它从文本场景描述中定位候选可关节化物体，并重建保留原始几何形状的可执行URDF模型。该管道的核心是Arti4URDF，它利用3D点云、大型语言模型的先验知识和面向URDF的提示设计，快速将刚性物体转换为基于URDF的交互式关节化物体，同时保持其3D形状。我们在三个级别评估了ArtiWorld：3D模拟物体、完整的3D模拟场景和真实世界扫描场景。在所有三种设置中，我们的方法始终优于现有方法，并取得了最先进的性能，同时保留了物体几何形状并正确捕获了物体交互性，以产生可用的基于URDF的关节化模型。这为直接从现有3D资产构建交互式、机器人就绪的模拟环境提供了实际路径。代码和数据将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何自动识别场景中可关节化的物体并直接转换为可交互的关节化资产的问题。这个问题很重要，因为交互式模拟器和机器人学习环境需要大量关节化资产，而现有的3D资产大多是静态的，手动转换它们为关节化物体非常耗时且成本高昂。自动化这一过程可以大大降低创建交互式机器人环境的门槛，促进机器人学习的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有场景生成方法（如Holodeck、MesaTask）和物体级关节化建模方法（如CAGE、Articulate-Anything）的局限性，发现它们要么未明确处理可关节化物体，要么难以保持原始几何形状。基于这些观察，作者设计了ArtiWorld管道和Arti4URDF模型，将3D点云特征与大型语言模型结合，直接推理部件关系和关节参数。该方法借鉴了PartNet-Mobility和PhysXNet数据集、ULIP点云编码器以及PointLLM等将点云与LLM对齐的方法，但专注于关节化建模而非识别。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D物体的几何特征（通过点云编码）与大型语言模型相结合，直接推理部件之间的关系、关节类型和位置，并生成完整的URDF文件。通过结构化推理链将无结构的几何信息转换为符号化的运动学规范。整体流程包括：1) 从场景描述中识别可关节化物体；2) 对物体点云进行采样和编码；3) 将编码后的点云标记与结构化提示输入Arti4URDF模型；4) 模型预测关节类型、轴线和限制；5) 生成可执行的URDF模型并重建为交互式场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) ArtiWorld框架，实现场景级别的可关节化物体识别和转换；2) Arti4URDF模型，将3D几何特征嵌入LLM直接生成URDF文件；3) 点云提示引导的分割策略，解决部件分解问题；4) 全面的三级评估体系。相比之前工作，ArtiWorld在保持原始几何形状方面优于URDFormer，在关节预测准确性上显著优于Articulate-Anything，并通过结构化推理链减少了URDF-Anything的多任务学习错误。此外，ArtiWorld首次在场景级别实现了可关节化物体的识别和转换，使整个模拟场景变得可交互。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ArtiWorld通过结合大型语言模型与3D点云分析，实现了从静态3D场景到交互式关节化环境的自动化转换，显著降低了创建机器人学习环境的成本和复杂性。'}


### 论文摘要

Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.

---

## 137. STONE: Pioneering the One-to-N Backdoor Threat in 3D Point Cloud

**论文链接:** [http://arxiv.org/abs/2511.11210v2](http://arxiv.org/abs/2511.11210v2)

**作者:** Dongmei Shan, Wei Lian, Chongxia Wang

**发布时间:** 2025-11-14

**备注:** 15 pages, 4 figures

### GPT解析

### 总结

STONE是首个实现一对N后门攻击的框架，通过可配置球形触发器使单个触发器能够控制多个输出标签，为3D模型中的一对N映射提供了理论基础和实际实现。

### 背景

后门攻击对深度学习构成严重威胁，特别是在自动驾驶和机器人等安全敏感的3D领域。现有的3D点云攻击仅限于静态的一对一模式，而更灵活的一对N后门威胁尚未得到探索，缺乏理论和实践基础。

### 目的

引入STONE框架，通过可配置球形触发器实现一对N后门威胁，为3D模型中的一对N映射建立理论基础。

### 方法

STONE框架使用参数化的空间属性创建动态密钥空间，使单个触发器能够控制多个输出标签。通过神经切线核分析为STONE提供理论基础，这是首次为3D模型中的一对N映射提供正式基础。

### 主要发现

广泛的评估显示高攻击成功率（高达100%），同时在干净数据上没有准确率损失。

### 结论

这项工作为3D视觉中的多目标威胁建立了基础基准，对保障未来智能系统至关重要。

### 翻译

后门攻击对深度学习构成严重威胁，特别是在自动驾驶和机器人等安全敏感的3D领域。尽管这些攻击威力强大，但现有的3D点云攻击仅限于静态的一对一模式，使得更灵活的一对N后门威胁在很大程度上未被探索，且缺乏理论和实践基础。我们通过引入STONE（球形触发一对N后门使能框架）解决了这一问题，这是首个通过可配置球形触发器实现这一威胁的框架。其参数化的空间属性创建了动态密钥空间，使单个触发器能够控制多个输出标签。理论上，我们通过神经切线核分析为STONE奠定基础，首次为3D模型中的一对N映射提供了正式基础。经验上，广泛的评估显示高攻击成功率（高达100%），同时在干净数据上没有准确率损失。这项工作为3D视觉中的多目标威胁建立了基础基准，对保障未来智能系统至关重要。


### 论文摘要

Backdoor attacks pose a critical threat to deep learning, especially in safety-sensitive 3D domains such as autonomous driving and robotics. Despite their potency, existing attacks on 3D point clouds are limited to a static one-to-one paradigm, leaving the more flexible one-to-N backdoor threat largely unexplored and without a theoretical or practical foundation. We address this by introducing STONE (Spherical Trigger One-to-N Backdoor Enabling), the first framework that instantiates this threat through a configurable spherical trigger. Its parameterizable spatial properties create a dynamic key space, enabling a single trigger to control multiple output labels. Theoretically, we ground STONE through Neural Tangent Kernel (NTK) analysis, providing the first formal basis for one-to-N mappings in 3D models. Empirically, extensive evaluations show high attack success rate (up to 100\%) with no loss in clean-data accuracy. This work establishes a foundational benchmark for multi-target threats in 3D vision, crucial for securing future intelligent systems.

---

## 138. SparseST: Exploiting Data Sparsity in Spatiotemporal Modeling and Prediction

**论文链接:** [http://arxiv.org/abs/2511.14753v1](http://arxiv.org/abs/2511.14753v1)

**作者:** Junfeng Wu, Hadjer Benmeziane, Kaoutar El Maghraoui, Liu Liu, Yinan Wang

**发布时间:** 2025-11-18

### GPT解析

### 总结

时空数据挖掘在复杂物理系统中有广泛应用，但ConvLSTM等高性能模型计算成本高，不适合边缘设备。现有高效AI方法主要减少模型冗余，但忽略了数据和特征冗余问题。本文提出SparseST框架，利用数据稀疏性开发高效时空模型，并通过多目标复合损失函数平衡模型性能与计算效率。

### 背景

时空数据挖掘在交通、制造、医疗等复杂物理系统中有广泛应用。ConvLSTM及其变体在各种STDM应用中表现优异，但计算成本高，不适合资源有限的边缘设备。随着边缘计算需求增长，需要高效AI方法降低计算成本同时保持性能。

### 目的

减少时空数据挖掘的计算成本，同时保持模型性能，为从业者提供根据计算资源限制和下游任务性能要求调整模型的实用指导。

### 方法

开发SparseST框架，利用数据稀疏性开发高效时空模型；设计多目标复合损失函数，探索和近似模型性能与计算效率之间的帕累托前沿。

### 主要发现

时空数据和特征中存在大量冗余，引入不必要的计算负担；通过利用数据稀疏性可以提高时空模型的效率；多目标复合损失函数可以在模型性能和计算效率之间提供平衡。

### 结论

SparseST框架开创性地利用数据稀疏性而非模型稀疏性来提高时空模型效率，为边缘计算环境下的时空数据挖掘提供了新思路。

### 翻译

时空数据挖掘(STDM)在各类复杂物理系统(CPS)中有广泛应用，如交通、制造、医疗等。在所有提出的方法中，卷积长短期记忆网络(ConvLSTM)已被证明可以在不同应用中具有泛化和可扩展性，并且其多种变体在各种STDM应用中取得了最先进的性能。然而，ConvLSTM及其变体计算量大，使其在计算资源有限的边缘设备上不适用。随着CPS中边缘计算的新兴需求，高效AI对于减少计算成本同时保持模型性能至关重要。常见的高效AI方法旨在减少模型容量中的冗余(如模型剪枝、压缩等)。然而，时空数据挖掘自然需要广泛的模型容量，因为时空数据中嵌入的依赖关系复杂且难以捕捉，这限制了模型冗余的减少。相反，数据和特征中存在大量冗余，引入了不必要的计算负担，这在现有研究中很大程度上被忽视。因此，我们开发了名为SparseST的新颖框架，开创性地利用数据稀疏性来开发高效的时空模型。此外，我们通过设计多目标复合损失函数，探索和近似模型性能与计算效率之间的帕累托前沿，为从业者提供了根据计算资源限制和下游任务性能要求调整模型的实用指南。


### 论文摘要

Spatiotemporal data mining (STDM) has a wide range of applications in various complex physical systems (CPS), i.e., transportation, manufacturing, healthcare, etc. Among all the proposed methods, the Convolutional Long Short-Term Memory (ConvLSTM) has proved to be generalizable and extendable in different applications and has multiple variants achieving state-of-the-art performance in various STDM applications. However, ConvLSTM and its variants are computationally expensive, which makes them inapplicable in edge devices with limited computational resources. With the emerging need for edge computing in CPS, efficient AI is essential to reduce the computational cost while preserving the model performance. Common methods of efficient AI are developed to reduce redundancy in model capacity (i.e., model pruning, compression, etc.). However, spatiotemporal data mining naturally requires extensive model capacity, as the embedded dependencies in spatiotemporal data are complex and hard to capture, which limits the model redundancy. Instead, there is a fairly high level of data and feature redundancy that introduces an unnecessary computational burden, which has been largely overlooked in existing research. Therefore, we developed a novel framework SparseST, that pioneered in exploiting data sparsity to develop an efficient spatiotemporal model. In addition, we explore and approximate the Pareto front between model performance and computational efficiency by designing a multi-objective composite loss function, which provides a practical guide for practitioners to adjust the model according to computational resource constraints and the performance requirements of downstream tasks.

---

## 139. Deep-Learning Based Super-Resolution Functional Ultrasound Imaging of Transient Brain-Wide Neurovascular Activity on a Microscopic Scale

**论文链接:** [http://arxiv.org/abs/2511.14071v1](http://arxiv.org/abs/2511.14071v1)

**作者:** Yang Cai, Shaoyuan Yan, Long Xu, Yanfeng Zhu, Bo Li, Kailiang Xu

**发布时间:** 2025-11-18

### GPT解析

### 总结

研究开发了一种名为超分辨率功能性超声（SR-fUS）的新技术，通过深度学习超分辨率重建方法突破了传统功能性超声成像的空间分辨率限制，实现了微观尺度上的全脑神经影像。

### 背景

微观尺度的全脑瞬时神经影像对脑研究至关重要，但当前的成像方式难以满足时空分辨率要求。功能性超声（fUS）通过红细胞后散射实现神经血管成像，但受衍射极限空间分辨率的限制。

### 目的

开发超分辨率功能性超声（SR-fUS）技术，突破传统功能性超声成像的空间分辨率限制，实现红细胞动力学的高分辨率重建。

### 方法

SR-fUS利用超声定位显微镜（ULM）数据，结合红细胞径向波动和不确定性驱动损失，实现超分辨率重建。

### 主要发现

SR-fUS达到了25微米的空间分辨率和10毫秒的时间分辨率；成功成像了疼痛刺激诱导的大鼠脑中瞬时血液动力学反应；通过与双光子显微镜的比较研究，验证了SR-fUS在胡须刺激期间皮层微血管中的准确性。

### 结论

SR-fUS技术成功突破了传统功能性超声成像的空间分辨率限制，为微观尺度的全脑神经影像提供了新的可能，特别是在研究瞬时血液动力学反应方面具有重要应用价值。

### 翻译

微观尺度的全脑瞬时神经影像对脑研究至关重要，但当前的成像方式难以满足时空分辨率要求。功能性超声（fUS）通过红细胞后散射实现神经血管成像，但受衍射极限空间分辨率的限制。我们假设基于深度学习的超分辨率重建可以突破这一限制，引入超分辨率功能性超声（SR-fUS），该技术利用超声定位显微镜（ULM）数据实现红细胞动力学的超分辨率重建。通过结合红细胞径向波动和不确定性驱动损失，SR-fUS实现了25微米的空间分辨率和10毫秒的时间分辨率。SR-fUS被应用于成像疼痛刺激诱导的大鼠脑中瞬时血液动力学反应。通过与双光子显微镜的比较研究，进一步验证了SR-fUS在胡须刺激期间皮层微血管中的准确性。


### 论文摘要

Transient brain-wide neuroimaging on a microscopic scale is pivotal for brain research, yet current modalities face challenges in meeting such spatiotemporal requirements. Functional ultrasound (fUS) enables transient neurovascular imaging through red blood cell backscattering, but suffers from diffraction-limited spatial resolution. We hypothesize that deep learning-based super-resolution reconstruction can break through this limitation, introducing super-resolution functional ultrasound (SR-fUS) which leverages ultrasound localization microscopy (ULM) data to achieve super-resolution reconstruction of red blood cell dynamics. By incorporating red blood cell radial fluctuations with uncertainty-driven loss, SR-fUS achieves 25-μm spatial and 10-ms temporal resolution. SR-fUS was applied to image transient hemodynamic responses induced by pain stimulation in rat brains. SR-fUS accuracy in cortical microvasculature during whisker stimulation was further validated by a comparative study with two-photon microscopy.

---

## 140. EchoAgent: Guideline-Centric Reasoning Agent for Echocardiography Measurement and Interpretation

**论文链接:** [http://arxiv.org/abs/2511.13948v1](http://arxiv.org/abs/2511.13948v1)

**作者:** Matin Daghyani, Lyuyang Wang, Nima Hashemi, Bassant Medhat, Baraa Abdelsamad, Eros Rojas Velez, XiaoXiao Li, Michael Y. C. Tsang, Christina Luong, Teresa S. M. Tsang, Purang Abolmaesumi

**发布时间:** 2025-11-17

**备注:** 12 pages, Under Review

### GPT解析

### 总结

EchoAgent是一种新型框架，通过在大型语言模型控制下协调专门的视觉工具，实现了心脏超声视频的结构化、可解释自动化。该系统能够执行时间定位、空间测量和临床解释，并通过测量可行性预测模型实现自主工具选择，为心脏超声分析提供了可信AI的新方向。

### 背景

当前深度学习模型无法满足心脏超声解释所需的视频级别推理和基于指南的测量分析需求，需要开发新的方法来实现心脏超声视频的结构化和可解释自动化。

### 目的

开发一个能够支持视频级别推理和基于指南的测量分析的心脏超声自动化框架，解决现有深度学习模型在心脏超声解释方面的局限性。

### 方法

EchoAgent在大型语言模型控制下协调专门的视觉工具执行时间定位、空间测量和临床解释；开发了测量可行性预测模型确定解剖结构在每个帧中是否可可靠测量；创建了多样化的、经过临床验证的视频-查询对基准用于评估。

### 主要发现

尽管增加了时空视频分析的复杂性，EchoAgent仍能获得准确、可解释的结果；输出基于视觉证据和临床指南，支持透明度和可追溯性。

### 结论

通过特定任务工具和完整的视频级别自动化，实现心脏超声视频分析的智能体式、与指南一致的推理是可行的；EchoAgent为心脏超声中的可信AI设定了新的方向。

### 翻译

目的：心脏超声解释需要视频级别的推理和基于指南的测量分析，而当前的心脏超声深度学习模型不支持这一点。我们提出了EchoAgent，一个能够在此领域实现结构化、可解释自动化的框架。方法：EchoAgent在大型语言模型控制下协调专门的视觉工具，执行时间定位、空间测量和临床解释。一个关键贡献是测量可行性预测模型，它确定解剖结构在每个帧中是否可可靠测量，从而实现自主工具选择。我们创建了一个多样化的、经过临床验证的视频-查询对基准用于评估。结果：尽管增加了时空视频分析的复杂性，EchoAgent仍能获得准确、可解释的结果。输出基于视觉证据和临床指南，支持透明度和可追溯性。结论：这项工作证明了通过特定任务工具和完整的视频级别自动化，实现心脏超声视频分析的智能体式、与指南一致的推理是可行的。EchoAgent为心脏超声中的可信AI设定了新的方向。


### 论文摘要

Purpose: Echocardiographic interpretation requires video-level reasoning and guideline-based measurement analysis, which current deep learning models for cardiac ultrasound do not support. We present EchoAgent, a framework that enables structured, interpretable automation for this domain. Methods: EchoAgent orchestrates specialized vision tools under Large Language Model (LLM) control to perform temporal localization, spatial measurement, and clinical interpretation. A key contribution is a measurement-feasibility prediction model that determines whether anatomical structures are reliably measurable in each frame, enabling autonomous tool selection. We curated a benchmark of diverse, clinically validated video-query pairs for evaluation. Results: EchoAgent achieves accurate, interpretable results despite added complexity of spatiotemporal video analysis. Outputs are grounded in visual evidence and clinical guidelines, supporting transparency and traceability. Conclusion: This work demonstrates the feasibility of agentic, guideline-aligned reasoning for echocardiographic video analysis, enabled by task-specific tools and full video-level automation. EchoAgent sets a new direction for trustworthy AI in cardiac ultrasound.

---

## 141. MAT-MPNN: A Mobility-Aware Transformer-MPNN Model for Dynamic Spatiotemporal Prediction of HIV Diagnoses in California, Florida, and New England

**论文链接:** [http://arxiv.org/abs/2511.13797v1](http://arxiv.org/abs/2511.13797v1)

**作者:** Zhaoxuan Wang, Weichen Kang, Yutian Han, Lingyuan Zhao, Bo Li

**发布时间:** 2025-11-17

**备注:** 21 pages, 20 figures,1 table. Preprint

### GPT解析

### 总结

研究提出了一种创新的MAT-MPNN框架，通过结合Transformer和改进的图生成器，有效捕捉了HIV传播的时空复杂性，并在多个地区的预测中表现出色。

### 背景

艾滋病病毒(HIV)已成为全球重大健康挑战数十年，预测HIV诊断一直是关键研究领域。然而，传统的消息传递神经网络模型依赖于固定的二元邻接矩阵，仅能表示地理邻接关系，无法捕捉非连续县份之间的相互作用。

### 目的

提出一种深度学习架构来预测县级HIV诊断率，提高预测准确性和校准效果。

### 方法

提出了移动感知Transformer-消息传递神经网络(MAT-MPNN)框架，结合Transformer编码器提取的时间特征和通过移动图生成器(MGG)捕获的空间关系。MGG结合了地理和人口统计信息，改进了传统邻接矩阵。研究区域包括加利福尼亚州、佛罗里达州和新英格兰地区。

### 主要发现

与最佳性能的混合基线模型Transformer MPNN相比，MAT-MPNN在佛罗里达州将均方预测误差降低27.9%，在加利福尼亚州降低39.1%，在新英格兰地区降低12.5%；在佛罗里达州将预测模型选择准则提高7.7%，在加利福尼亚州提高3.5%，在新英格兰地区提高3.9。与空间变化自回归模型相比，MAT-MPNN在佛罗里达州和新英格兰地区表现更好，在加利福尼亚州表现相当。

### 结论

应用移动感知动态空间结构显著提高了时空流行病学预测的准确性和校准效果。

### 翻译

人类免疫缺陷病毒（HIV）数十年来已成为全球重大健康挑战，预测HIV诊断继续成为关键研究领域。然而，捕捉HIV传播复杂的时空依赖关系仍然具有挑战性。传统的消息传递神经网络（MPNN）模型依赖于固定的二元邻接矩阵，该矩阵仅编码地理邻接关系，无法表示不连续县份之间的相互作用。我们的研究提出了一种深度学习架构，即移动感知Transformer-消息传递神经网络（MAT-MPNN）框架，用于预测加利福尼亚州、佛罗里达州和新英格兰地区的县级HIV诊断率。该模型结合了Transformer编码器提取的时间特征和通过移动图生成器（MGG）捕获的空间关系。MGG通过结合地理和人口统计信息改进了传统邻接矩阵。与最佳性能的混合基线模型Transformer MPNN相比，MAT-MPNN在佛罗里达州将均方预测误差（MSPE）降低了27.9%，在加利福尼亚州降低了39.1%，在新英格兰地区降低了12.5%，并将预测模型选择准则（PMCC）分别提高了7.7%、3.5%和3.9%。MAT-MPNN在佛罗里达州和新英格兰地区的表现也优于空间变化自回归（SVAR）模型，在加利福尼亚州表现相当。这些结果表明，应用移动感知动态空间结构显著提高了时空流行病学预测的准确性和校准。


### 论文摘要

Human Immunodeficiency Virus (HIV) has posed a major global health challenge for decades, and forecasting HIV diagnoses continues to be a critical area of research. However, capturing the complex spatial and temporal dependencies of HIV transmission remains challenging. Conventional Message Passing Neural Network (MPNN) models rely on a fixed binary adjacency matrix that only encodes geographic adjacency, which is unable to represent interactions between non-contiguous counties. Our study proposes a deep learning architecture Mobility-Aware Transformer-Message Passing Neural Network (MAT-MPNN) framework to predict county-level HIV diagnosis rates across California, Florida, and the New England region. The model combines temporal features extracted by a Transformer encoder with spatial relationships captured through a Mobility Graph Generator (MGG). The MGG improves conventional adjacency matrices by combining geographic and demographic information. Compared with the best-performing hybrid baseline, the Transformer MPNN model, MAT-MPNN reduced the Mean Squared Prediction Error (MSPE) by 27.9% in Florida, 39.1% in California, and 12.5% in New England, and improved the Predictive Model Choice Criterion (PMCC) by 7.7%, 3.5%, and 3.9%, respectively. MAT-MPNN also achieved better results than the Spatially Varying Auto-Regressive (SVAR) model in Florida and New England, with comparable performance in California. These results demonstrate that applying mobility-aware dynamic spatial structures substantially enhances predictive accuracy and calibration in spatiotemporal epidemiological prediction.

---

## 142. CCSD: Cross-Modal Compositional Self-Distillation for Robust Brain Tumor Segmentation with Missing Modalities

**论文链接:** [http://arxiv.org/abs/2511.14599v1](http://arxiv.org/abs/2511.14599v1)

**作者:** Dongqing Xie, Yonghuang Wu, Zisheng Ai, Jun Min, Zhencun Jiang, Shaojin Geng, Lei Wang

**发布时间:** 2025-11-18

**备注:** 9 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为跨模态组合自蒸馏（CCSD）的新框架，用于解决在多模态MRI脑肿瘤分割中常见的模态缺失问题，该框架在各种缺失模态场景下均达到了最先进的性能。

### 背景

从多模态MRI中准确分割脑肿瘤对临床诊断和治疗规划至关重要。然而，在现实临床环境中，经常缺少一个或多个模态，这对基于深度学习的分割模型构成了重大挑战，严重影响了模型性能和泛化能力。

### 目的

提出一种能够灵活处理任意输入模态组合的新方法，以解决在缺少模态情况下的脑肿瘤分割问题。

### 方法

提出了一种新颖的跨模态组合自蒸馏（CCSD）框架，该框架采用共享-特定编码器-解码器架构，并包含两种自蒸馏策略：分层模态自蒸馏机制，在模态层次间转移知识以减少语义差异；渐进式模态组合蒸馏方法，通过在训练期间模拟逐渐的模态缺失来增强对缺失模态的鲁棒性。

### 主要发现

在公共脑肿瘤分割基准上的大量实验表明，CCSD在各种缺失模态场景下都达到了最先进的性能，具有很强的泛化能力和稳定性。

### 结论

CCSD框架有效地解决了多模态MRI中脑肿瘤分割面临的模态缺失问题，提高了模型在现实临床环境中的适用性。

### 翻译

从多模态MRI中准确分割脑肿瘤对临床诊断和治疗规划至关重要。虽然整合来自各种MRI序列的互补信息是常见做法，但在现实临床环境中一个或多个模态的频繁缺失构成了重大挑战，严重损害了基于深度学习的分割模型的性能和泛化能力。为应对这一挑战，我们提出了一种新颖的跨模态组合自蒸馏（CCSD）框架，可以灵活处理任意输入模态组合。CCSD采用共享-特定编码器-解码器架构，并融入两种自蒸馏策略：（i）分层模态自蒸馏机制，在模态层次间转移知识以减少语义差异；（ii）渐进式模态组合蒸馏方法，通过在训练期间模拟逐渐的模态缺失来增强对缺失模态的鲁棒性。在公共脑肿瘤分割基准上的大量实验证明，CCSD在各种缺失模态场景下均实现了最先进的性能，具有很强的泛化能力和稳定性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态MRI脑肿瘤分割中某些MRI模态缺失时模型性能下降的问题。在临床实践中，由于运动伪影、设备问题或协议不匹配，经常无法获取所有四种MRI模态(FLAIR、T1、T1c、T2)，这会严重影响深度学习模型的分割效果。这个问题很重要，因为它限制了深度学习模型在真实临床环境中的应用可靠性和实用性，可能导致医生对AI辅助诊断的信任度降低。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有处理缺失模态方法的局限性：模态增强方法会产生伪影；特征空间工程方法假设训练和测试时缺失模式一致；架构工程方法通常需要复杂结构。作者借鉴了知识蒸馏和自蒸馏的思想，但发现现有方法要么计算开销大，要么独立处理每个模态子集。作者的关键洞察是应通过自蒸馏框架内的相互知识交换来统一不同模态组合，充分利用模态组合的结构特性，设计了共享-特定编码器-解码器架构和两种自蒸馏策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过两种自蒸馏策略实现跨模态知识转移，使模型能处理任意模态组合。具体流程：1)使用共享编码器和模态特定编码器分别提取共同特征和独有特征，通过组合层融合；2)对缺失模态，将其输入设为零，用共享特征表示；3)分层模态自蒸馏(HMSD)：从完整模态向不同大小的模态子集转移知识，减少语义差异；4)递减模态组合蒸馏(DMCD)：按关键性顺序逐步移除模态，模拟渐进式丢失，通过顺序蒸馏增强鲁棒性；5)结合分割损失和两种蒸馏损失共同训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)灵活的共享-特定架构，可处理任意模态组合；2)分层模态自蒸馏，通过层级结构实现平滑知识转移；3)递减模态组合蒸馏，模拟最坏情况下的数据丢失。相比之前工作不同：不是固定教师-学生配对，而是在单一网络内自蒸馏；不是直接映射而是渐进式知识转移；不依赖模态合成避免伪影；不假设一致缺失模式；不依赖复杂结构；能更好处理关键模态缺失的极端情况。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种跨模态组合自蒸馏框架，通过分层模态自蒸馏和递减模态组合蒸馏两种策略，有效解决了多模态脑肿瘤分割中模态缺失问题，显著提高了模型在任意模态组合下的鲁棒性和性能。'}


### 论文摘要

The accurate segmentation of brain tumors from multi-modal MRI is critical for clinical diagnosis and treatment planning. While integrating complementary information from various MRI sequences is a common practice, the frequent absence of one or more modalities in real-world clinical settings poses a significant challenge, severely compromising the performance and generalizability of deep learning-based segmentation models. To address this challenge, we propose a novel Cross-Modal Compositional Self-Distillation (CCSD) framework that can flexibly handle arbitrary combinations of input modalities. CCSD adopts a shared-specific encoder-decoder architecture and incorporates two self-distillation strategies: (i) a hierarchical modality self-distillation mechanism that transfers knowledge across modality hierarchies to reduce semantic discrepancies, and (ii) a progressive modality combination distillation approach that enhances robustness to missing modalities by simulating gradual modality dropout during training. Extensive experiments on public brain tumor segmentation benchmarks demonstrate that CCSD achieves state-of-the-art performance across various missing-modality scenarios, with strong generalization and stability.

---

## 143. CLO: Efficient LLM Inference System with CPU-Light KVCache Offloading via Algorithm-System Co-Design

**论文链接:** [http://arxiv.org/abs/2511.14510v1](http://arxiv.org/abs/2511.14510v1)

**作者:** Jiawei Yi, Ping Gong, Youhui Bai, Jiaqi Ruan, Shengnan Wang, Pengcheng Wang, Haibo Wang, Weiguang Wang, Xia Zhu, Feng Wu, Cheng Li

**发布时间:** 2025-11-18

### GPT解析

### 总结

论文提出CLO，一种CPU轻量级KVCache卸载系统，通过算法和系统协同设计解决百万token级别LLM推理中的CPU瓶颈问题，显著提高解码吞吐量9.3%-66.6%。

### 背景

百万token级别的大语言模型(LLM)增长暴露了推理系统的可扩展性限制，其中KVCache主导内存使用和数据传输开销。现有卸载系统忽视了CPU瓶颈的三个方面：(1)CPU端细粒度动态缓存管理的巨大开销；(2)CPU端密集收集操作导致的PCIe带宽利用率差；(3)以CPU为中心的粗粒度同步引入的GPU运行时气泡。

### 目的

设计一个CPU轻量级的KVCache卸载系统，通过算法-系统协同设计解决现有系统中的CPU瓶颈问题，提高LLM推理效率。

### 方法

提出CLO系统，包含：(1)粗粒度头级近似GPU缓存策略，缓存管理成本可忽略不计；(2)数据预取与GPU持久缓存的无缝结合，降低传输开销；(3)零拷贝传输引擎充分利用PCIe带宽，以及GPU中心同步方法消除GPU停顿。

### 主要发现

在两个广泛使用的LLM上评估，CLO实现了与最先进系统相当的准确性，同时显著减少CPU开销，完全利用PCIe带宽，将解码吞吐量提高了9.3%-66.6%。

### 结论

算法-系统协同设计对于现代GPU平台上内存受限的LLM推理至关重要，CLO通过解决CPU瓶颈问题有效提高了LLM推理系统的性能。

### 翻译

百万token级别的大语言模型(LLM)增长暴露了推理系统的可扩展性限制，其中KVCache主导内存使用和数据传输开销。最近的卸载系统将KVCache迁移到CPU内存，并采用top-k注意力机制来减少从CPU传输的数据量，同时应用系统级优化如GPU缓存和预取来降低传输开销。然而，它们忽视了CPU瓶颈的三个方面：(1)CPU端执行的细粒度动态缓存管理的巨大开销；(2)CPU端密集收集操作导致的PCIe带宽利用率差，造成显著传输开销；(3)以CPU为中心的粗粒度同步引入的GPU运行时气泡。为应对这些挑战，我们提出了CLO，一种通过算法-系统协同设计的CPU轻量级KVCache卸载系统。CLO的特点包括：(1)粗粒度头级近似GPU缓存策略，缓存管理成本可忽略不计；(2)数据预取与GPU持久缓存的无缝结合，降低传输开销；(3)零拷贝传输引擎充分利用PCIe带宽，以及GPU中心同步方法消除GPU停顿。在两个广泛使用的LLM上的评估表明，CLO实现了与最先进系统相当的准确性，同时显著减少CPU开销，完全利用PCIe带宽，从而将解码吞吐量提高了9.3%-66.6%。我们的研究结果表明，算法-系统协同设计对于现代GPU平台上内存受限的LLM推理至关重要。我们在https://github.com/CommediaJW/CLO开源了CLO。


### 论文摘要

The growth of million-token LLMs exposes the scalability limits of inference systems, where the KVCache dominates memory usage and data transfer overhead. Recent offloading systems migrate the KVCache to CPU memory and incorporate top-k attention to reduce the volume of data transferred from the CPU, while further applying system-level optimizations such as on-GPU caching and prefetching to lower transfer overhead. However, they overlook the CPU bottleneck in three aspects: (1) substantial overhead of fine-grained dynamic cache management performed on the CPU side, (2) significant transfer overhead from poor PCIe bandwidth utilization caused by heavy gathering operations at the CPU side, and (3) GPU runtime bubbles introduced by coarse-grained CPU-centric synchronization. To address these challenges, we propose CLO, a CPU-light KVCache offloading system via algorithm-system co-design. CLO features: (1) a coarse-grained head-wise approximate on-GPU caching strategy with negligible cache management cost, (2) seamless combination of data prefetching and on-GPU persistent caching for lower transfer overhead, (3) a zero-copy transfer engine to fully exploit PCIe bandwidth, and a GPU-centric synchronization method to eliminate GPU stalls. Evaluation on two widely-used LLMs demonstrates that CLO achieves comparable accuracy to state-of-the-art systems, while substantially minimizing CPU overhead, fully utilizing PCIe bandwidth, thus improving decoding throughput by 9.3%-66.6%. Our results highlight that algorithm-system co-design is essential for memory-constrained LLM inference on modern GPU platforms. We open source CLO at https://github.com/CommediaJW/CLO.

---

## 144. GloTok: Global Perspective Tokenizer for Image Reconstruction and Generation

**论文链接:** [http://arxiv.org/abs/2511.14184v1](http://arxiv.org/abs/2511.14184v1)

**作者:** Xuan Zhao, Zhongyu Zhang, Yuge Huang, Yuxi Mi, Guodong Mu, Shouhong Ding, Jun Wang, Rizen Guo, Shuigeng Zhou

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为GloTok的图像标记化方法，通过利用全局关系信息建模更均匀的语义分布，提高图像重建和生成质量，且训练过程中无需直接访问预训练模型。

### 背景

现有图像标记化方法利用预训练视觉模型的语义特征进行监督扩展潜在表示分布，但采用局部监督方法限制了语义分布的均匀性。VA-VAE已证明更均匀的特征分布能带来更好的生成性能。

### 目的

引入一种新的标记化方法，利用全局关系信息建模更均匀的语义分布，提高图像重建和生成质量，同时避免训练过程中直接访问预训练模型。

### 方法

提出GloTok方法，包括：1)基于码本直方图关系学习的方法，将预训练模型的全局数据集语义转移到语义码本；2)设计残差学习模块恢复细粒度细节，最小化量化引起的重建误差。

### 主要发现

GloTok能提供分布更均匀的语义潜在表示，有利于自回归模型训练生成高质量图像，且训练过程中无需直接访问预训练模型。

### 结论

在ImageNet-1k基准上的实验表明，该方法达到了最先进的重建性能和生成质量。

### 翻译

现有的最先进的图像标记化方法利用预训练视觉模型的多样化语义特征进行额外监督，以扩展潜在表示的分布，从而提高图像重建和生成的质量。这些方法采用局部监督方法进行语义监督，这限制了语义分布的均匀性。然而，VA-VAE证明更均匀的特征分布能带来更好的生成性能。在这项工作中，我们引入了全局视角标记器(GloTok)，它利用全局关系信息来建模标记化特征的更均匀语义分布。具体而言，我们提出了一种基于码本的直方图关系学习方法，将预训练模型在整个数据集上建模的语义转移到语义码本中。然后，我们设计了一个残差学习模块，用于恢复细粒度细节，以最小化量化引起的重建误差。通过上述设计，GloTok提供了分布更均匀的语义潜在表示，这有利于自回归(AR)模型的训练，用于生成高质量图像，而在训练过程中不需要直接访问预训练模型。在标准ImageNet-1k基准上的实验清楚地表明，我们提出的方法达到了最先进的重建性能和生成质量。


### 论文摘要

Existing state-of-the-art image tokenization methods leverage diverse semantic features from pre-trained vision models for additional supervision, to expand the distribution of latent representations and thereby improve the quality of image reconstruction and generation. These methods employ a locally supervised approach for semantic supervision, which limits the uniformity of semantic distribution. However, VA-VAE proves that a more uniform feature distribution yields better generation performance. In this work, we introduce a Global Perspective Tokenizer (GloTok), which utilizes global relational information to model a more uniform semantic distribution of tokenized features. Specifically, a codebook-wise histogram relation learning method is proposed to transfer the semantics, which are modeled by pre-trained models on the entire dataset, to the semantic codebook. Then, we design a residual learning module that recovers the fine-grained details to minimize the reconstruction error caused by quantization. Through the above design, GloTok delivers more uniformly distributed semantic latent representations, which facilitates the training of autoregressive (AR) models for generating high-quality images without requiring direct access to pre-trained models during the training process. Experiments on the standard ImageNet-1k benchmark clearly show that our proposed method achieves state-of-the-art reconstruction performance and generation quality.

---

## 145. A Patient-Independent Neonatal Seizure Prediction Model Using Reduced Montage EEG and ECG

**论文链接:** [http://arxiv.org/abs/2511.14110v1](http://arxiv.org/abs/2511.14110v1)

**作者:** Sithmini Ranasingha, Agasthi Haputhanthri, Hansa Marasinghe, Nima Wickramasinghe, Kithmin Wickremasinghe, Jithangi Wanigasinghe, Chamira U. S. Edussooriya, Joshua P. Kulasingham

**发布时间:** 2025-11-18

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

本研究提出了一种基于卷积神经网络的新生儿癫痫早期预测模型，通过区分脑电图的间歇期和发作前期状态来实现癫痫预测。

### 背景

新生儿易患癫痫且常导致神经损伤，但临床表现微妙易被误诊，导致未经治疗的癫痫活动延长和脑损伤。连续视频脑电图监测是金标准但昂贵且需要专业知识。

### 目的

开发一种患者独立的癫痫早期预测模型，能够在癫痫发作前30分钟进行预测，以提高诊断准确性和及时性。

### 方法

构建基于卷积神经网络的模型，使用多通道脑电图和心电图信号提取的梅尔频率倒谱系数矩阵作为输入特征。在赫尔辛基新生儿脑电图数据集上使用10折交叉验证进行训练和验证。

### 主要发现

模型达到平均准确率97.52%、灵敏度98.31%、特异性96.39%、F1分数97.95%；结合心电图使F1分数提高1.42%，加入注意力机制额外提高0.5%；使用SHAP方法解释模型并提供头皮图定位癫痫病灶。

### 结论

该模型具有在新生儿重症监护室进行最小监督部署的潜力，能及时可靠预测癫痫，并通过迁移学习展示强大泛化能力。

### 翻译

新生儿极易患癫痫，常导致短期或长期神经功能障碍。然而，新生儿癫痫的临床表现微妙，常导致误诊，增加了未经治疗的癫痫活动延长和随后的脑损伤风险。连续视频脑电图监测是癫痫检测的金标准，但这是一项昂贵的评估，需要专业知识和时间。在本研究中，我们提出了一种基于卷积神经网络的模型，通过区分脑电图的间歇期和发作前期状态来实现新生儿癫痫的早期预测。我们的模型是患者独立的，能够在多个主体间泛化，并使用从多通道脑电图和心电图信号中提取的梅尔频率倒谱系数矩阵作为输入特征。在赫尔辛基新生儿脑电图数据集上使用10折交叉验证进行训练和验证，所提出的模型平均准确率达到97.52%，灵敏度98.31%，特异性96.39%，F1分数97.95%，能够在癫痫发作前30分钟进行准确预测。结合心电图与脑电图使F1分数提高了1.42%，而加入注意力机制则额外提高了0.5%。为提高透明度，我们整合了SHAP作为可解释人工智能方法来解释模型，并使用头皮图提供癫痫病灶的定位。总体结果表明，该模型具有在新生儿重症监护室进行最小监督部署的潜力，能够及时可靠地预测新生儿癫痫，同时通过迁移学习展示了在未见主体上的强大泛化能力。


### 论文摘要

Neonates are highly susceptible to seizures, often leading to short or long-term neurological impairments. However, clinical manifestations of neonatal seizures are subtle and often lead to misdiagnoses. This increases the risk of prolonged, untreated seizure activity and subsequent brain injury. Continuous video electroencephalogram (cEEG) monitoring is the gold standard for seizure detection. However, this is an expensive evaluation that requires expertise and time. In this study, we propose a convolutional neural network-based model for early prediction of neonatal seizures by distinguishing between interictal and preictal states of the EEG. Our model is patient-independent, enabling generalization across multiple subjects, and utilizes mel-frequency cepstral coefficient matrices extracted from multichannel EEG and electrocardiogram (ECG) signals as input features. Trained and validated on the Helsinki neonatal EEG dataset with 10-fold cross-validation, the proposed model achieved an average accuracy of 97.52%, sensitivity of 98.31%, specificity of 96.39%, and F1-score of 97.95%, enabling accurate seizure prediction up to 30 minutes before onset. The inclusion of ECG alongside EEG improved the F1-score by 1.42%, while the incorporation of an attention mechanism yielded an additional 0.5% improvement. To enhance transparency, we incorporated SHapley Additive exPlanations (SHAP) as an explainable artificial intelligence method to interpret the model and provided localization of seizure focus using scalp plots. The overall results demonstrate the model's potential for minimally supervised deployment in neonatal intensive care units, enabling timely and reliable prediction of neonatal seizures, while demonstrating strong generalization capability across unseen subjects through transfer learning.

---

## 146. LogPurge: Log Data Purification for Anomaly Detection via Rule-Enhanced Filtering

**论文链接:** [http://arxiv.org/abs/2511.14062v1](http://arxiv.org/abs/2511.14062v1)

**作者:** Shenglin Zhang, Ziang Chen, Zijing Que, Yilun Liu, Yongqian Sun, Sicheng Wei, Dan Pei, Hailin Li

**发布时间:** 2025-11-18

### GPT解析

### 总结

论文提出了一种名为LogPurge的成本感知、规则增强的净化框架，用于自动选择足够的正常日志序列子集来训练异常检测模型。该方法通过两阶段过滤算法实现，首先使用大语言模型去除聚集的异常模式并增强系统规则，然后采用分治策略将剩余污染区域分解为更小子问题进行净化。

### 背景

日志异常检测对于识别系统故障和预防安全漏洞至关重要，影响服务可靠性、性能优化和数据库日志分析等领域。现代方法依赖在干净日志序列上训练深度学习模型，但获取干净数据需要昂贵繁琐的人工标注，现有自动清洗方法未能充分整合日志的特定特征和实际语义。

### 目的

提出一种成本感知、规则增强的日志净化框架LogPurge，能够自动从污染的日志序列中选择足够的正常日志序列子集来训练异常检测模型，解决获取干净日志数据的高成本问题。

### 方法

LogPurge采用两阶段过滤算法：第一阶段使用大语言模型去除聚集的异常模式并增强系统规则；第二阶段采用分治策略将剩余污染区域分解为更小子问题，使每个子问题都能通过第一阶段程序有效净化。

### 主要发现

实验在两个公共数据集和一个工业数据集上进行，方法平均能去除98.74%的异常同时保留82.39%的正常样本。相比最新的无监督日志样本选择算法，在公共数据集上F-1分数分别提高了35.7%和84.11%，在私有数据集上实现了149.72%的F-1改进。

### 结论

LogPurge框架能有效解决日志异常检测中干净数据获取困难的问题，通过结合大语言模型和分治策略，显著提高了异常检测模型的性能，证明了该方法的有效性。

### 翻译

日志异常检测对于识别系统故障和预防安全漏洞至关重要，它能在大量日志数据中检测不规则模式，影响服务可靠性、性能优化和数据库日志分析等领域。现代日志异常检测方法依赖于在干净、无异常的日志序列上训练深度学习模型。然而，获取这样的干净日志数据需要昂贵且繁琐的人工标注，现有的自动清洗方法在其净化过程中未能完全整合日志的特定特征和实际语义。在本文中，我们提出了一种成本感知、规则增强的净化框架LogPurge，它能自动从污染的日志序列中选择足够的正常日志序列子集来训练异常检测模型。我们的方法涉及两阶段过滤算法：在第一阶段，我们使用大语言模型去除聚集的异常模式并增强系统规则，以提高对系统日志的理解；在第二阶段，我们采用分治策略将剩余的污染区域分解为更小的子问题，使每个子问题都能通过第一阶段程序有效净化。我们在两个公共数据集和一个工业数据集上进行的实验表明，我们的方法平均能显著去除98.74%的异常，同时保留82.39%的正常样本。与最新的无监督日志样本选择算法相比，我们的方法在公共数据集上实现了35.7%和84.11%的F-1分数提升，在私有数据集上实现了令人印象深刻的149.72%的F-1改进，证明了我们方法的有效性。


### 论文摘要

Log anomaly detection, which is critical for identifying system failures and preempting security breaches, detects irregular patterns within large volumes of log data, and impacts domains such as service reliability, performance optimization, and database log analysis. Modern log anomaly detection methods rely on training deep learning models on clean, anomaly-free log sequences. However, obtaining such clean log data requires costly and tedious human labeling, and existing automatic cleaning methods fail to fully integrate the specific characteristics and actual semantics of logs in their purification process. In this paper, we propose a cost-aware, rule-enhanced purification framework, LogPurge, that automatically selects a sufficient subset of normal log sequences from contamination log sequences to train a anomaly detection model. Our approach involves a two-stage filtering algorithm: In the first stage, we use a large language model (LLM) to remove clustered anomalous patterns and enhance system rules to improve LLM's understanding of system logs; in the second stage, we utilize a divide-and-conquer strategy that decomposes the remaining contaminated regions into smaller subproblems, allowing each to be effectively purified through the first stage procedure. Our experiments, conducted on two public datasets and one industrial dataset, show that our method significantly removes an average of 98.74% of anomalies while retaining 82.39% of normal samples. Compared to the latest unsupervised log sample selection algorithms, our method achieves F-1 score improvements of 35.7% and 84.11% on the public datasets, and an impressive 149.72% F-1 improvement on the private dataset, demonstrating the effectiveness of our approach.

---

## 147. SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics

**论文链接:** [http://arxiv.org/abs/2511.14049v1](http://arxiv.org/abs/2511.14049v1)

**作者:** Semen Leontev

**发布时间:** 2025-11-18

**备注:** 64 pages, 5 figures, 15 tables

### GPT解析

### 总结

本文介绍SmallML，一个贝叶斯迁移学习框架，能够在小数据集（50-200个观测值）上实现企业级预测精度，解决了中小企业因规模小而无法应用AI的问题。

### 背景

中小企业占美国企业的99.9%，但由于其运营规模与现代机器学习的数据需求不匹配，他们被系统性地排除在AI应用之外。

### 目的

开发一种能够在小数据条件下实现高精度预测的方法，使中小企业能够利用机器学习技术。

### 方法

研究团队开发了一个三层架构：第一层使用基于SHAP的程序从公共记录中提取信息先验；第二层在多个中小企业之间实现分层池化，使用自适应收缩平衡群体模式与实体特定特征；第三层提供具有有限样本覆盖保证的共形集，实现分布无关的不确定性量化。

### 主要发现

在客户流失数据验证中，每个企业100个观测值的情况下达到96.7%+/-4.2%的AUC，比独立逻辑回归提高了24.2个百分点；共形预测在90%目标下达到92%的实证覆盖率；在标准CPU硬件上训练完成时间为33分钟。

### 结论

SmallML使之前被机器学习排除在外的3300万美国中小企业能够获得企业级预测，解决了AI民主化的关键差距。

### 翻译

中小企业占美国企业的99.9%，但由于其运营规模与现代机器学习的数据需求不匹配，他们被系统性地排除在AI应用之外。本文介绍了SmallML，一个贝叶斯迁移学习框架，能够在数据集小至50-200个观测值的情况下实现企业级预测精度。我们开发了一个三层架构，整合了迁移学习、分层贝叶斯建模和共形预测。第一层使用基于SHAP的程序从22,673条公共记录中提取信息先验，将梯度提升的知识迁移到逻辑回归。第二层在J=5-50个中小企业之间实现分层池化，使用自适应收缩，平衡群体模式与实体特定特征。第三层提供具有有限样本覆盖保证的共形集P(y in C(x)) >= 1-alpha，实现分布无关的不确定性量化。在客户流失数据上的验证显示，每个企业100个观测值的情况下达到96.7%+/-4.2%的AUC，比独立逻辑回归（72.5%+/-8.1%）提高了24.2个百分点，p < 0.000001。共形预测在90%目标下达到92%的实证覆盖率。在标准CPU硬件上训练完成时间为33分钟。通过使之前被机器学习排除在外的3300万美国中小企业能够获得企业级预测，SmallML解决了AI民主化的关键差距。


### 论文摘要

Small and medium-sized enterprises (SMEs) represent 99.9% of U.S. businesses yet remain systematically excluded from AI due to a mismatch between their operational scale and modern machine learning's data requirements. This paper introduces SmallML, a Bayesian transfer learning framework achieving enterprise-level prediction accuracy with datasets as small as 50-200 observations.   We develop a three-layer architecture integrating transfer learning, hierarchical Bayesian modeling, and conformal prediction. Layer 1 extracts informative priors from 22,673 public records using a SHAP-based procedure transferring knowledge from gradient boosting to logistic regression. Layer 2 implements hierarchical pooling across J=5-50 SMEs with adaptive shrinkage, balancing population patterns with entity-specific characteristics. Layer 3 provides conformal sets with finite-sample coverage guarantees P(y in C(x)) >= 1-alpha for distribution-free uncertainty quantification.   Validation on customer churn data demonstrates 96.7% +/- 4.2% AUC with 100 observations per business -- a +24.2 point improvement over independent logistic regression (72.5% +/- 8.1%), with p < 0.000001. Conformal prediction achieves 92% empirical coverage at 90% target. Training completes in 33 minutes on standard CPU hardware. By enabling enterprise-grade predictions for 33 million U.S. SMEs previously excluded from machine learning, SmallML addresses a critical gap in AI democratization.   Keywords: Bayesian transfer learning, hierarchical models, conformal prediction, small-data analytics, SME machine learning

---

## 148. Learning Skill-Attributes for Transferable Assessment in Video

**论文链接:** [http://arxiv.org/abs/2511.13993v1](http://arxiv.org/abs/2511.13993v1)

**作者:** Kumar Ashutosh, Kristen Grauman

**发布时间:** 2025-11-17

**备注:** NeurIPS 2025, Project webpage: https://vision.cs.utexas.edu/projects/CrossTrainer/

### GPT解析

### 总结

这篇论文介绍了一种名为CrossTrainer的新方法，用于从视频中评估技能。它通过发现跨运动边界的技能属性，并利用多模态语言模型生成反馈，解决了当前模型在长尾运动领域缺乏专家监督的问题。该方法在跨运动和运动内设置中均表现优异，比现有技术提高了高达60%的性能。

### 背景

从视频中评估技能需要评估一个人身体表现的质量并解释如何改进。当前的模型专门针对个别运动，并且在长尾运动领域缺乏专家级监督，导致成本高且稀缺。

### 目的

缩小当前模型与实际需求之间的差距，探索可迁移的视频表示用于技能评估。

### 方法

提出了CrossTrainer方法，发现跨运动边界的技能属性，如平衡、控制和手部定位，然后训练多模态语言模型为新颖视频生成可操作的反馈和熟练度水平。

### 主要发现

在多个数据集上验证了新模型，在跨运动（迁移）和运动内（领域内）设置中均取得了高达60%的相对提升。通过抽象出表明人类技能的共享行为，所提出的视频表示比多种现有技术具有更好的泛化能力。

### 结论

通过抽象出表明人类技能的共享行为，所提出的视频表示方法丰富了当今的多模态大型语言模型，并显著提升了性能。

### 翻译

从视频中评估技能需要对一个人的身体表现质量进行评分并解释如何改进。当今的模型专门针对某项特定运动，并且在长尾运动领域缺乏专家级监督，导致成本高昂且稀缺。为缩小这一差距，我们探索了可迁移的视频表示用于技能评估。我们的CrossTrainer方法发现了诸如平衡、控制和手部定位等技能属性——这些属性的含义超越了任何给定运动的边界，然后训练一个多模态语言模型为新颖视频生成可操作的反馈，例如'提高双手以产生更多力量'，以及其熟练度水平，例如早期专家。我们在多个数据集上验证了新模型，在跨运动（迁移）和运动内（领域内）设置中，相对于最先进技术实现了高达60%的提升。通过抽象出表明人类技能的共享行为，所提出的视频表示比多种现有技术具有更好的泛化能力，丰富了当今的多模态大型语言模型。


### 论文摘要

Skill assessment from video entails rating the quality of a person's physical performance and explaining what could be done better. Today's models specialize for an individual sport, and suffer from the high cost and scarcity of expert-level supervision across the long tail of sports. Towards closing that gap, we explore transferable video representations for skill assessment. Our CrossTrainer approach discovers skill-attributes, such as balance, control, and hand positioning -- whose meaning transcends the boundaries of any given sport, then trains a multimodal language model to generate actionable feedback for a novel video, e.g., "lift hands more to generate more power" as well as its proficiency level, e.g., early expert. We validate the new model on multiple datasets for both cross-sport (transfer) and intra-sport (in-domain) settings, where it achieves gains up to 60% relative to the state of the art. By abstracting out the shared behaviors indicative of human skill, the proposed video representation generalizes substantially better than an array of existing techniques, enriching today's multimodal large language models.

---

## 149. Hybrid Convolution Neural Network Integrated with Pseudo-Newton Boosting for Lumbar Spine Degeneration Detection

**论文链接:** [http://arxiv.org/abs/2511.13877v1](http://arxiv.org/abs/2511.13877v1)

**作者:** Pandiyaraju V, Abishek Karthik, Jaspin K, Kannan A, Jaime Lloret

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种增强型模型架构，用于对腰椎退行性病变进行分类，采用混合方法结合EfficientNet和VGG19以及自定义组件。

### 背景

在医学图像，特别是高维医学图像背景下，传统迁移学习方法存在局限性，通常忽略了详细的解剖学特征。

### 目的

开发一种新的模型架构来提高腰椎退行性病变分类的性能，克服传统迁移学习方法的约束。

### 方法

提出了一种混合架构，整合EfficientNet和VGG19，并加入伪牛顿提升层和稀疏诱导特征缩减层。伪牛顿提升层能够智能调整特征权重，特别关注详细的解剖学特征；稀疏诱导层则去除学习特征中的冗余，产生简洁而鲁棒的病理表示。

### 主要发现

该架构显著提高了性能，与基线模型EfficientNet相比，达到了0.9的精确度、0.861的召回率、0.88的F1分数、0.18的损失值和88.1%的准确率。

### 结论

该工作克服了传统迁移学习方法在医学图像高维环境中的限制，为医学图像自动诊断工具的发展做出了贡献。

### 翻译

本文提出了一种新的增强型模型架构，用于对腰椎退行性病变进行分类，同时采用混合方法，将EfficientNet和VGG19与自定义设计的组件相结合。所提出的模型与传统迁移学习方法不同，它集成了伪牛顿提升层和稀疏诱导特征缩减层，形成多层次框架，进一步改进了特征选择和表示。伪牛顿提升层对特征权重进行智能调整，特别关注详细的解剖学特征，这些特征在迁移学习设置中大多被忽略。此外，稀疏诱导层去除学习特征中的冗余，为腰椎脊柱病理产生简洁而鲁棒的表示。这种架构是新颖的，它克服了传统迁移学习方法中的限制，特别是在医学图像的高维背景下，并实现了显著的性能提升，与基线模型EfficientNet相比，达到了0.9的精确度、0.861的召回率、0.88的F1分数、0.18的损失值和88.1%的准确率。这项工作将介绍架构、预处理流程和实验结果。这些结果有助于医学图像自动诊断工具的发展。


### 论文摘要

This paper proposes a new enhanced model architecture to perform classification of lumbar spine degeneration with DICOM images while using a hybrid approach, integrating EfficientNet and VGG19 together with custom-designed components. The proposed model is differentiated from traditional transfer learning methods as it incorporates a Pseudo-Newton Boosting layer along with a Sparsity-Induced Feature Reduction Layer that forms a multi-tiered framework, further improving feature selection and representation. The Pseudo-Newton Boosting layer makes smart variations of feature weights, with more detailed anatomical features, which are mostly left out in a transfer learning setup. In addition, the Sparsity-Induced Layer removes redundancy for learned features, producing lean yet robust representations for pathology in the lumbar spine. This architecture is novel as it overcomes the constraints in the traditional transfer learning approach, especially in the high-dimensional context of medical images, and achieves a significant performance boost, reaching a precision of 0.9, recall of 0.861, F1 score of 0.88, loss of 0.18, and an accuracy of 88.1%, compared to the baseline model, EfficientNet. This work will present the architectures, preprocessing pipeline, and experimental results. The results contribute to the development of automated diagnostic tools for medical images.

---

## 150. Inferring Planet and Disk Parameters from Protoplanetary Disk Images Using a Variational Autoencoder

**论文链接:** [http://arxiv.org/abs/2511.13840v1](http://arxiv.org/abs/2511.13840v1)

**作者:** Sayed Shafaat Mahmud, Sayantan Auddy, Neal Turner, Jeffrey S. Bary

**发布时间:** 2025-11-17

**备注:** 29 Pages, 20 Figures, Accepted at the Astrophysical Journal

### GPT解析

### 总结

本文提出了VADER框架，一种基于变分自编码器的生成式机器学习方法，用于从原行星盘的尘埃连续谱观测图像中推断行星和盘参数。

### 背景

许多原行星盘的尘埃连续谱观测显示有环和间隙，这些现象被广泛解释为行星正在形成的证据。

### 目的

开发第一个使用基于变分自编码器(VAE)的生成式机器学习(ML)框架，从原行星盘图像中推断行星和盘参数。

### 方法

创建名为VADER的框架，使用从FARGO3D流体动力学模拟生成的合成尘埃连续谱发射图像进行训练，并通过蒙特卡洛辐射转移计算进行后处理。VADER能够推断多达三个嵌入行星的质量以及盘参数：粘性α、气体-尘埃比、斯托克斯数和翘曲指数，并返回每个量的完整后验分布。

### 主要发现

VADER能够以高结构相似性重建盘形态，在不同行星质量下准确恢复行星参数，可靠预测盘参数。应用于23个原行星盘的ALMA观测，返回的嵌入行星质量估计为0.3-2个木星质量，与大多数已发表值在统计上一致，推断的盘参数与当前文献一致。

### 结论

一旦训练完成，VAE可以在几分钟内完成完整的后验参数推断，为大规模ALMA调查提供了足够的计算速度和统计严谨性。基于VAE的模型成为从盘结构推断嵌入行星质量和全局盘参数及其相关不确定性的强大工具。

### 翻译

许多原行星盘的尘埃连续谱观测揭示了环和间隙，这些被广泛解释为行星正在形成的证据。在这里，我们提出了第一个框架，使用基于变分自编码器的生成式机器学习从这类图像中推断行星和盘参数。这个新框架被称为VADER。我们在尘埃连续谱发射的合成图像上训练VADER，这些图像是从流体动力学模拟生成的，并通过辐射转移计算进行后处理。VADER推断出多达三个嵌入行星的质量以及盘参数：粘性α、气体-尘埃比、斯托克斯数和翘曲指数。VADER返回每个量的完整后验分布。我们证明VADER能够以高结构相似性重建盘形态，在不同行星质量下准确恢复行星参数，并可靠预测盘参数。应用于二十三个原行星盘的尘埃连续谱图像，我们的模型返回的嵌入行星质量估计为0.3-2个木星质量，在大多数情况下与已发表值在统计上一致，并推断出与当前文献一致的盘参数。一旦训练完成，VAE可以在几分钟内完成完整的后验参数推断，为大规模调查提供了足够的计算速度和统计严谨性。这些结果建立了基于VAE的模型作为从盘结构推断嵌入行星质量和全局盘参数及其相关不确定性的强大工具。


### 论文摘要

Dust-continuum observations of many protoplanetary disks reveal rings and gaps that are widely interpreted as evidence of ongoing planet formation. Here we present the first framework for inferring planet and disk parameters from such images using variational autoencoder (VAE) based generative machine learning (ML). The new framework is called VADER (Variational Autoencoder for Disks with Embedded Rings). We train VADER on synthetic images of dust continuum emission, generated from \texttt{FARGO3D} hydrodynamic simulations post-processed with Monte Carlo radiative transfer calculations. VADER infers the masses of up to three embedded planets as well as the disk parameters viscous $α$, dust-to-gas ratio, Stokes number, and flaring index. VADER returns a full posterior distribution for each of these quantities. We demonstrate that VADER reconstructs disk morphologies with high structural similarity (index $>$ 0.99), accurately recovers planet parameters with $R^2 > 0.9$ across planet masses, and reliably predicts disk parameters. Applied to ALMA dust continuum images of 23 protoplanetary disks, our model returns mass estimates for embedded planets of 0.3-2~$M_{\mathrm{Jup}}$ that agree to within $1σ$ of published values in most cases, and infers disk parameters consistent with current literature. Once trained, the VAE performs full posterior parameter inference in a matter of minutes, offering statistical rigor with enough computational speed for application to large-scale ALMA surveys. These results establish VAE-based models as powerful tools for inferring from disk structure the masses of embedded planets and the global disk parameters, with their associated uncertainties.

---

## 151. Systematic Evaluation of Time-Frequency Features for Binaural Sound Source Localization

**论文链接:** [http://arxiv.org/abs/2511.13487v2](http://arxiv.org/abs/2511.13487v2)

**作者:** Davoud Shariat Panah, Alessandro Ragano, Dan Barry, Jan Skoglund, Andrew Hines

**发布时间:** 2025-11-17

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本研究系统评估了双耳声源定位中的时频特征设计，研究了特征选择对模型性能的影响。研究发现精心设计的特征组合通常优于增加模型复杂度，最佳特征组合能使低复杂度CNN模型实现有竞争力的性能。

### 背景

双耳声源定位是声音处理领域的重要任务，特征选择对模型在不同条件下的性能有显著影响。

### 目的

系统评估时频特征设计对双耳声源定位的影响，研究特征选择如何影响模型性能，确定最佳特征组合。

### 方法

使用卷积神经网络(CNN)模型测试基于幅度(幅度谱图、双耳电平差ILD)和基于相位(相位谱图、双耳相位差IPD)的各种特征组合，在域内和域外数据上进行评估，使用了不匹配的头部相关传递函数(HRTF)。

### 主要发现

精心选择特征组合通常优于增加模型复杂性；对于域内SSL，ILD+IPD这样的双特征集已足够；要推广到多样化内容，需要结合通道谱图与ILD和IPD的更丰富输入；使用最佳特征集，低复杂度CNN模型实现了有竞争力的性能。

### 结论

特征设计在双耳声源定位中至关重要，为特定领域和通用定位提供了实用指导。

### 翻译

本研究对双耳声源定位(SSL)中的时频特征设计进行了系统评估，重点关注特征选择如何影响模型在不同条件下的性能。我们研究了使用各种基于幅度(幅度谱图、双耳电平差ILD)和基于相位(相位谱图、双耳相位差IPD)的特征组合的卷积神经网络(CNN)模型的性能。在具有不匹配的头部相关传递函数(HRTF)的域内和域外数据上的评估表明，精心选择特征组合通常优于增加模型复杂性。虽然像ILD+IPD这样的双特征集足以用于域内SSL，但推广到多样化内容需要结合通道谱图与ILD和IPD的更丰富输入。使用最佳特征集，我们的低复杂度CNN模型实现了具有竞争力的性能。我们的研究结果强调了特征设计在双耳SSL中的重要性，并为特定领域和通用定位提供了实用指导。


### 论文摘要

This study presents a systematic evaluation of time-frequency feature design for binaural sound source localization (SSL), focusing on how feature selection influences model performance across diverse conditions. We investigate the performance of a convolutional neural network (CNN) model using various combinations of amplitude-based features (magnitude spectrogram, interaural level difference - ILD) and phase-based features (phase spectrogram, interaural phase difference - IPD). Evaluations on in-domain and out-of-domain data with mismatched head-related transfer functions (HRTFs) reveal that carefully chosen feature combinations often outperform increases in model complexity. While two-feature sets such as ILD + IPD are sufficient for in-domain SSL, generalization to diverse content requires richer inputs combining channel spectrograms with both ILD and IPD. Using the optimal feature sets, our low-complexity CNN model achieves competitive performance. Our findings underscore the importance of feature design in binaural SSL and provide practical guidance for both domain-specific and general-purpose localization.

---

## 152. SLAM-AGS: Slide-Label Aware Multi-Task Pretraining Using Adaptive Gradient Surgery in Computational Cytology

**论文链接:** [http://arxiv.org/abs/2511.14639v1](http://arxiv.org/abs/2511.14639v1)

**作者:** Marco Acerbis, Swarnadip Chatterjee, Christophe Avenel, Joakim Lindblad

**发布时间:** 2025-11-18

**备注:** 5 pages, 2 figures, Submitted to ISBI2026

### GPT解析

### 总结

SLAM-AGS是一种幻灯片标签感知的多任务预训练框架，解决了计算细胞学中实例级标签不可靠和检出率极低的挑战，通过联合优化弱监督相似性目标和自监督对比目标，并应用自适应梯度手术稳定学习，显著提高了下游任务性能。

### 背景

计算细胞学面临两个主要挑战：实例级标签不可靠且获取成本极高，以及检出率极低。

### 目的

开发SLAM-AGS框架，解决上述挑战，提高下游任务性能，特别是在低检出率情况下的表现。

### 方法

SLAM-AGS框架联合优化两个目标：在幻灯片负样本块上的弱监督相似性目标和在幻灯片正样本块上的自监督对比目标。应用自适应梯度手术来稳定学习，解决冲突的任务梯度并防止模型崩溃。将预训练编码器集成到基于注意力的多实例学习聚合器中，进行包级预测和注意力引导的最异常实例检索。

### 主要发现

在公开的骨髓细胞学数据集上，从10%到0.5%的模拟检出率下，SLAM-AGS在包级F1分数和前400个阳性细胞检索方面优于其他预训练方法，在低检出率下提升最大。解决梯度干扰可以实现稳定的预训练和更好的下游任务性能。

### 结论

SLAM-AGS框架能够有效解决计算细胞学的挑战，特别是在低检出率情况下表现优异，研究团队已公开完整实现和评估框架。

### 翻译

计算细胞学面临两大挑战：一是实例级标签不可靠且获取成本过高，二是检出率极低。我们提出了SLAM-AGS，这是一种幻灯片标签感知的多任务预训练框架，联合优化了幻灯片负样本块上的弱监督相似性目标和幻灯片正样本块上的自监督对比目标，从而在下游任务上取得更好的性能。为稳定学习，我们应用自适应梯度手术来解决冲突的任务梯度并防止模型崩溃。我们将预训练编码器集成到基于注意力的多实例学习聚合器中，用于包级预测和包内最异常实例的注意力引导检索。在公开的骨髓细胞学数据集上，从10%到0.5%的模拟检出率下，SLAM-AGS在包级F1分数和前400个阳性细胞检索方面优于其他预训练方法，在低检出率下提升最大，这表明解决梯度干扰可以实现稳定的预训练和更好的下游任务性能。为便于复现，我们分享了完整的实现和评估框架作为开源代码：https://github.com/Ace95/SLAM-AGS。


### 论文摘要

Computational cytology faces two major challenges: i) instance-level labels are unreliable and prohibitively costly to obtain, ii) witness rates are extremely low. We propose SLAM-AGS, a Slide-Label-Aware Multitask pretraining framework that jointly optimizes (i) a weakly supervised similarity objective on slide-negative patches and (ii) a self-supervised contrastive objective on slide-positive patches, yielding stronger performance on downstream tasks. To stabilize learning, we apply Adaptive Gradient Surgery to tackle conflicting task gradients and prevent model collapse. We integrate the pretrained encoder into an attention-based Multiple Instance Learning aggregator for bag-level prediction and attention-guided retrieval of the most abnormal instances in a bag. On a publicly available bone-marrow cytology dataset, with simulated witness rates from 10% down to 0.5%, SLAM-AGS improves bag-level F1-Score and Top 400 positive cell retrieval over other pretraining methods, with the largest gains at low witness rates, showing that resolving gradient interference enables stable pretraining and better performance on downstream tasks. To facilitate reproducibility, we share our complete implementation and evaluation framework as open source: https://github.com/Ace95/SLAM-AGS.

---

## 153. Failure to Mix: Large language models struggle to answer according to desired probability distributions

**论文链接:** [http://arxiv.org/abs/2511.14630v1](http://arxiv.org/abs/2511.14630v1)

**作者:** Ivy Yuqian Yang, David Yu Zhang

**发布时间:** 2025-11-18

**备注:** 13 pages, 6 figures. Code and reproducibility package: https://github.com/BiostateAIresearch/failure-to-mix

### GPT解析

### 总结

现代大型语言模型无法按照指定的概率分布生成输出，表现出类似阶函数的行为，倾向于生成概率略高的选项，这会影响科学思想生成等需要概率性探索的任务。

### 背景

科学思想的生成和选择需要遵循目标概率分布进行探索，而当前AI基准测试有客观正确答案，通过这些基准对大型语言模型进行强化学习训练会抑制概率性探索。

### 目的

研究大型语言模型是否能够按照指定的概率分布生成输出。

### 方法

进行系统性实验，要求大型语言模型按照简单的概率分布生成输出。

### 主要发现

所有测试的现代大型语言模型都无法遵循指定的概率分布。例如，要求输出'1'的概率为49%时，模型几乎100%时间输出'0'。这种类似阶函数的行为倾向于生成概率略高的输出，甚至超过了模型内置的强偏见。

### 结论

现代大型语言模型在概率分布遵循方面存在严重缺陷，这可能会影响其在需要概率性探索的任务上的表现。

### 翻译

科学思想的生成和选择需要遵循目标概率分布进行探索。相比之下，当前AI基准测试有客观正确的答案，通过这些基准对大型语言模型进行强化学习训练会抑制概率性探索。在这里，我们进行了系统性实验，要求大型语言模型按照简单的概率分布生成输出，发现所有测试的现代大型语言模型都完全无法遵循这些分布。例如，要求输出'1'的概率为49%时，模型几乎100%时间输出'0'。这种类似阶函数的行为倾向于生成概率略高的输出，甚至超过了模型内置的强偏见。


### 论文摘要

Scientific idea generation and selection requires exploration following a target probability distribution. In contrast, current AI benchmarks have objectively correct answers, and training large language models (LLMs) via reinforcement learning against these benchmarks discourages probabilistic exploration. Here, we conducted systematic experiments requesting LLMs to produce outputs following simple probabilistic distributions, and found that all modern LLMs tested grossly fail to follow the distributions. For example, requesting a binary output of "1" 49% of the time produces an answer of "0" nearly 100% of the time. This step function-like behavior of near-exclusively generating the output with marginally highest probability even overrules even strong in-built LLM biases.

---

## 154. Gradient-Based Join Ordering

**论文链接:** [http://arxiv.org/abs/2511.14482v1](http://arxiv.org/abs/2511.14482v1)

**作者:** Tim Schwabe, Maribel Acosta

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种基于梯度的数据库查询连接顺序优化方法，通过将离散查询计划连续化，使用图神经网络作为成本模型，实现了比传统方法更高效且可扩展的查询优化。

### 背景

连接顺序选择是数据库查询优化的核心问题，属于NP难问题。传统方法将其视为基于成本模型的离散组合搜索，但面临高计算复杂性和有限可扩展性的挑战。

### 目的

解决传统连接顺序优化方法的高计算复杂性和有限可扩展性问题，开发一种更高效、可扩展的查询优化方法。

### 方法

当成本模型可微分时，将查询计划连续松弛为表示计划叠加的软邻接矩阵。结合Gumbel-Softmax参数化的邻接矩阵和强制计划有效性的可微分约束，在松弛空间内进行基于梯度的搜索。使用学习的图神经网络作为成本模型。

### 主要发现

在两种不同的图数据集上，这种基于梯度的方法能够找到与传统的离散局部搜索方法相当甚至更低成本的计划。此外，该方法的运行时间随查询大小线性增长，而经典方法的运行时间则是二次方或指数增长。

### 结论

这种基于梯度的连接顺序优化方法为未来更有效和高效的查询优化器开辟了新的可能性。

### 翻译

连接顺序选择是数据库查询中评估连接（合取、二元操作符）的最有效序列选择的NP难问题。由于查询执行的性能严重依赖于这一选择，连接顺序位于查询优化的核心。传统方法将此问题视为基于成本模型的离散组合搜索，但它们常常面临高计算复杂性和有限的可扩展性问题。我们表明，当成本模型可微分时，查询计划可以连续松弛为表示计划叠加的软邻接矩阵。这种连续松弛，结合邻接矩阵的Gumbel-Softmax参数化和强制计划有效性的可微分约束，使得能够在该松弛空间内进行基于梯度的计划搜索。使用学习的图神经网络作为成本模型，我们在两种不同的图数据集上证明，与传统离散局部搜索方法相比，这种基于梯度的方法能够找到相当甚至更低成本的计划。此外，我们 empirically 证明，与经典方法的二次方或指数运行时间相比，这种方法的运行时间随查询大小线性增长。我们相信这种迈向基于梯度的连接顺序优化的第一步可以在未来引领更有效和高效的查询优化器。


### 论文摘要

Join ordering is the NP-hard problem of selecting the most efficient sequence in which to evaluate joins (conjunctive, binary operators) in a database query. As the performance of query execution critically depends on this choice, join ordering lies at the core of query optimization. Traditional approaches cast this problem as a discrete combinatorial search over binary trees guided by a cost model, but they often suffer from high computational complexity and limited scalability. We show that, when the cost model is differentiable, the query plans can be continuously relaxed into a soft adjacency matrix representing a superposition of plans. This continuous relaxation, together with a Gumbel-Softmax parameterization of the adjacency matrix and differentiable constraints enforcing plan validity, enables gradient-based search for plans within this relaxed space. Using a learned Graph Neural Network as the cost model, we demonstrate that this gradient-based approach can find comparable and even lower-cost plans compared to traditional discrete local search methods on two different graph datasets. Furthermore, we empirically show that the runtime of this approach scales linearly with query size, in contrast to quadratic or exponential runtimes of classical approaches. We believe this first step towards gradient-based join ordering can lead to more effective and efficient query optimizers in the future.

---

## 155. EBind: a practical approach to space binding

**论文链接:** [http://arxiv.org/abs/2511.14229v1](http://arxiv.org/abs/2511.14229v1)

**作者:** Jim Broadbent, Felix Cohen, Frederik Hvilshøj, Eric Landau, Eren Sasoglu

**发布时间:** 2025-11-18

### GPT解析

### 总结

EBind是一种简化空间绑定的方法，通过关注两个核心组件（每个模态的单个编码器和高质量数据）实现了在单GPU上几小时内训练出最先进模型的能力，而不是需要多天。一个简单的18亿参数的多模态模型可以比它大4到17倍的模型表现更好。

### 背景

当前空间绑定方法复杂且计算资源密集，需要多GPU和多天训练才能达到最佳性能。

### 目的

简化空间绑定过程，减少计算资源需求，使模型能在单GPU上快速训练，同时保持或提高性能。

### 方法

提出EBind方法，这是一种简单、数据为中心、参数高效的方法，用于绑定多个对比模型的嵌入空间。使用三个互补数据源精心策划的数据集：i) 670万个通过最先进的检索模型获取的全自动多模态五元组；ii) 100万由人工标注为负匹配、部分匹配或正匹配的多样化三元组；iii) 340万个已有标题的数据项。

### 主要发现

一个简单的18亿参数的图像-文本-视频-音频-3D模型可以比它大4到17倍的模型表现更好。通过13种不同的评估证明了每个数据源的价值。

### 结论

通过简化空间绑定的核心组件和使用高质量数据，可以在显著减少计算资源的情况下实现最先进的模型性能。由于现有基准的局限性，还引入了第一个高质量的、共识标注的音频与PC之间的零样本分类基准。

### 翻译

我们通过关注两个核心组件（每个模态的单个编码器和高质量数据）来简化空间绑定，使模型能够在单GPU上几小时内训练出最先进的模型，而不是需要多天。我们提出了EBind，这是一种简单、数据为中心且参数高效的方法，用于绑定多个对比模型的嵌入空间。我们证明了一个简单的18亿参数的图像-文本-视频-音频-3D模型可以比它大4到17倍的模型表现更好。实现这一点的关键是三个互补数据源精心策划的数据集：i) 670万个通过最先进的检索模型获取的全自动多模态五元组；ii) 100万由人工标注为负匹配、部分匹配或正匹配的多样化三元组；iii) 340万个已有标题的数据项。我们使用13种不同的评估来展示每个数据源的价值。由于现有基准的局限性，我们进一步引入了第一个高质量的、共识标注的音频与PC之间的零样本分类基准。与相关工作相比，我们将开源我们的代码、模型权重和数据集。


### 论文摘要

We simplify space binding by focusing on two core components, a single encoder per modality and high-quality data; enabling training state-of-the-art models on a single GPU in a few hours as opposed to multiple days. We present EBind, an Easy, data-centric, and parameter-efficient method to Bind the embedding spaces of multiple contrastive models. We demonstrate that a simple 1.8B-parameter image-text-video-audio-3D model can outperform models 4 to 17x the size. The key to achieving this is a carefully curated dataset of three complementary data sources: i) 6.7M fully-automated multimodal quintuples sourced via SOTA retrieval models, ii) 1M diverse, semi-automated triples annotated by humans as negative, partial, or positive matches, and iii) 3.4M pre-existing captioned data items. We use 13 different evaluations to demonstrate the value of each data source. Due to limitations with existing benchmarks, we further introduce the first high-quality, consensus-annotated zero-shot classification benchmark between audio and PCs. In contrast to related work, we will open-source our code, model weights, and datasets.

---

## 156. Learning Representation and Synergy Invariances: A Povable Framework for Generalized Multimodal Face Anti-Spoofing

**论文链接:** [http://arxiv.org/abs/2511.14157v1](http://arxiv.org/abs/2511.14157v1)

**作者:** Xun Lin, Shuai Wang, Yi Yu, Zitong Yu, Jiale Zhou, Yizhong Liu, Xiaochun Cao, Alex Kot, Yefeng Zheng

**发布时间:** 2025-11-18

### GPT解析

### 总结

多模态人脸活体检测方法在跨领域部署时性能下降严重，比单模态FAS更严重。论文提出了RiSe框架，解决了模态表示不变风险和模态协同不变风险两个问题，通过AsyIRM和MMSD方法实现了最先进的跨领域性能。

### 背景

多模态人脸活体检测方法在跨领域部署时性能下降严重，比单模态FAS更严重。这是由于两个被忽视的风险影响跨模态泛化：模态表示不变风险和模态协同不变风险。

### 目的

解决多模态人脸活体检测在跨领域部署时性能下降的问题，具体是解决模态表示不变风险和模态协同不变风险。

### 方法

提出名为RiSe（Multimodal Representation and Synergy Invariance Learning）的框架，包含Asymmetric Invariant Risk Minimization (AsyIRM)和Multimodal Synergy Disentanglement (MMSD)两个组件。AsyIRM处理表示风险，学习球面决策边界；MMSD处理协同风险，通过跨样本混合和解缠增强泛化特征。

### 主要发现

FAS中固有的类别不对称性（多样化的欺骗 vs 紧凑的真实）扩大了泛化误差的上限，这种效应在多模态设置中被进一步放大；模型过度拟合领域特定的跨模态相关性，导致无法泛化到目标领域中的未知攻击。

### 结论

RiSe框架通过理论分析和实验验证，实现了最先进的跨领域性能。

### 翻译

多模态人脸活体检测(FAS)方法整合多种视觉模态，但在部署到未见领域时性能下降甚至比单模态FAS更严重。这主要是由于两个影响跨模态泛化的被忽视风险。首先是模态表示不变风险，即表示在领域变化下是否保持泛化能力。理论上证明，FAS中固有的类别不对称性（多样化的欺骗 vs 紧凑的真实）扩大了泛化误差的上限，这一效应在多模态设置中被进一步放大。其次是模态协同不变风险，模型过度拟合特定领域的跨模态相关性。这种虚假协同无法泛化到目标领域中的未知攻击，导致性能下降。为解决这些问题，我们提出了一个可证明的框架，即多模态表示和协同不变学习(RiSe)。对于表示风险，RiSe引入非对称不变风险最小化(AsyIRM)，在径向空间学习不变球面决策边界以适应不对称分布，同时在角度空间保留领域线索。对于协同风险，RiSe采用多模态协同解缠(MMSD)，这是一种自监督任务，通过跨样本混合和解缠增强内在的、可泛化的模态特征。理论分析和实验验证了RiSe的有效性，它实现了最先进的跨领域性能。


### 论文摘要

Multimodal Face Anti-Spoofing (FAS) methods, which integrate multiple visual modalities, often suffer even more severe performance degradation than unimodal FAS when deployed in unseen domains. This is mainly due to two overlooked risks that affect cross-domain multimodal generalization. The first is the modal representation invariant risk, i.e., whether representations remain generalizable under domain shift. We theoretically show that the inherent class asymmetry in FAS (diverse spoofs vs. compact reals) enlarges the upper bound of generalization error, and this effect is further amplified in multimodal settings. The second is the modal synergy invariant risk, where models overfit to domain-specific inter-modal correlations. Such spurious synergy cannot generalize to unseen attacks in target domains, leading to performance drops. To solve these issues, we propose a provable framework, namely Multimodal Representation and Synergy Invariance Learning (RiSe). For representation risk, RiSe introduces Asymmetric Invariant Risk Minimization (AsyIRM), which learns an invariant spherical decision boundary in radial space to fit asymmetric distributions, while preserving domain cues in angular space. For synergy risk, RiSe employs Multimodal Synergy Disentanglement (MMSD), a self-supervised task enhancing intrinsic, generalizable modal features via cross-sample mixing and disentanglement. Theoretical analysis and experiments verify RiSe, which achieves state-of-the-art cross-domain performance.

---

## 157. BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition

**论文链接:** [http://arxiv.org/abs/2511.14097v1](http://arxiv.org/abs/2511.14097v1)

**作者:** Weijia Fan, Qiufu Li, Jiajun Wen, Xiaoyang Peng

**发布时间:** 2025-11-18

**备注:** [AAAI-2026] code: https://github.com/wakinghours-github/BCE3S

### GPT解析

### 总结

本文提出了一种基于二元交叉熵的三元协同学习方法BCE3S，用于解决长尾识别任务中的特征学习和分类器不平衡问题，在多个数据集上实现了最先进的性能。

### 背景

在长尾识别任务中，期望头部和尾部类别都具有高的类内紧凑性和类间可分离性，同时所有分类器向量之间应保持平衡的可分离性。然而，现有基于交叉熵损失的方法难以学习理想特征，且在Softmax分母中耦合不平衡分类器向量，放大了不平衡效应。

### 目的

解决现有LTR方法中基于交叉熵损失函数的局限性，提出一种新方法来学习具有更好特性的特征，提高类内紧凑性和类间可分离性，同时平衡分类器向量间的可分离性。

### 方法

提出基于二元交叉熵的三元协同学习方法BCE3S，包含三个组成部分：1) 基于BCE的联合学习，优化分类器和样本特征；2) 基于BCE的对比学习，提高特征类内紧凑性；3) 基于BCE的均匀学习，平衡分类器向量间可分离性并增强特征特性。

### 主要发现

使用BCE3S训练的LTR模型实现了样本特征间更高的紧凑性和可分离性，同时平衡了分类器的可分离性，在CIFAR10-LT、CIFAR100-LT、ImageNet-LT和iNaturalist2018等数据集上取得了最先进的性能。

### 结论

BCE3S通过三元协同学习机制，有效解决了长尾识别任务中的特征学习和分类器不平衡问题，显著提高了模型性能。

### 翻译

对于长尾识别任务，我们期望头部类别和尾部类别都具有高的类内紧凑性和类间可分离性，同时所有分类器向量之间也应保持平衡的可分离性。现有的基于交叉熵损失的长尾识别方法不仅难以学习具有理想特性的特征，而且在Softmax的分母中耦合了不平衡的分类器向量，放大了长尾识别中的不平衡效应。在本文中，对于LTR任务，我们提出了一种基于二元交叉熵的三元协同学习方法，称为BCE3S，它包含三个组成部分：(1) 基于BCE的联合学习优化分类器和样本特征，通过在多个Sigmoid中解耦特征与不平衡分类器向量之间的度量，实现了比基于CE的联合学习更好的特征紧凑性和可分离性；(2) 基于BCE的对比学习进一步提高特征的类内紧凑性；(3) 基于BCE的均匀学习平衡分类器向量之间的可分离性，并通过与联合学习的交互式结合增强特征特性。广泛的实验表明，使用BCE3S训练的LTR模型不仅实现了样本特征间更高的紧凑性和可分离性，还平衡了分类器的可分离性，在CIFAR10-LT、CIFAR100-LT、ImageNet-LT和iNaturalist2018等各种长尾数据集上实现了最先进的性能。


### 论文摘要

For long-tailed recognition (LTR) tasks, high intra-class compactness and inter-class separability in both head and tail classes, as well as balanced separability among all the classifier vectors, are preferred. The existing LTR methods based on cross-entropy (CE) loss not only struggle to learn features with desirable properties but also couple imbalanced classifier vectors in the denominator of its Softmax, amplifying the imbalance effects in LTR. In this paper, for the LTR, we propose a binary cross-entropy (BCE)-based tripartite synergistic learning, termed BCE3S, which consists of three components: (1) BCE-based joint learning optimizes both the classifier and sample features, which achieves better compactness and separability among features than the CE-based joint learning, by decoupling the metrics between feature and the imbalanced classifier vectors in multiple Sigmoid; (2) BCE-based contrastive learning further improves the intra-class compactness of features; (3) BCE-based uniform learning balances the separability among classifier vectors and interactively enhances the feature properties by combining with the joint learning. The extensive experiments show that the LTR model trained by BCE3S not only achieves higher compactness and separability among sample features, but also balances the classifier's separability, achieving SOTA performance on various long-tailed datasets such as CIFAR10-LT, CIFAR100-LT, ImageNet-LT, and iNaturalist2018.

---

## 158. GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards

**论文链接:** [http://arxiv.org/abs/2511.14045v1](http://arxiv.org/abs/2511.14045v1)

**作者:** Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种针对强化学习与可验证奖励(RLVR)的新型成员推理攻击DIBA，该攻击利用模型行为变化而非答案记忆来推断训练数据成员关系，在多种场景下表现出显著优越性。

### 背景

大语言模型(LLMs)上的成员推理攻击(MIAs)在模型训练各阶段构成重大隐私风险。RLVR的最新进展为LLM训练带来范式转变，但其on-policy特性引入了独特的隐私泄露模式：训练依赖自生成响应而非固定真实输出。

### 目的

审计RLVR训练中基于行为变化的新型隐私风险，开发专门针对RLVR的成员推理攻击框架。

### 方法

提出Divergence-in-Behavior Attack (DIBA)，首个专为RLVR设计的成员推理框架。该方法将重点从记忆转向行为变化，利用模型行为在两个轴上的可测量变化：优势侧改进(如正确性增益)和对数侧发散(如策略漂移)。

### 主要发现

DIBA显著优于现有基线，实现约0.8的AUC和数量级更高的TPR@0.1%FPR。该方法在分布内、跨数据集、跨算法、黑盒场景以及视觉-语言模型扩展中均表现出优越性，且在适度防御措施下保持稳健。

### 结论

首次系统分析了RLVR中的隐私漏洞，表明即使在缺乏明确监督的情况下，训练数据暴露也可以通过行为痕迹可靠地推断出来。

### 翻译

在大语言模型(LLMs)上的成员推理攻击(MIAs)对模型训练的各个阶段构成重大隐私风险。强化学习与可验证奖励(RLVR)的最新进展为LLM训练带来了深刻的范式转变，特别是在复杂推理任务方面。然而，RLVR的on-policy特性引入了一种独特的隐私泄露模式：由于训练依赖于自生成的响应而没有固定的真实输出，成员推理现在必须确定给定提示(独立于任何特定响应)是否在微调过程中被使用。这创造了一种威胁，其中泄露不是来自答案记忆。为了审计这种新型隐私风险，我们提出了Divergence-in-Behavior Attack (DIBA)，这是第一个专门为RLVR设计的成员推理框架。DIBA将重点从记忆转向行为变化，利用模型行为在两个轴上的可测量变化：优势侧改进(例如，正确性增益)和对数侧发散(例如，策略漂移)。通过全面评估，我们证明DIBA显著优于现有基线，实现约0.8的AUC和数量级更高的TPR@0.1%FPR。我们在多种设置下验证了DIBA的优越性，包括分布内、跨数据集、跨算法、黑盒场景以及对视觉-语言模型的扩展。此外，我们的攻击在适度的防御措施下仍然保持稳健。据我们所知，这是第一个系统分析RLVR中隐私漏洞的工作，揭示了即使在缺乏明确监督的情况下，训练数据暴露也可以通过行为痕迹可靠地推断出来。


### 论文摘要

Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.   To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.   To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.

---

## 159. Training-free Detection of AI-generated images via Cropping Robustness

**论文链接:** [http://arxiv.org/abs/2511.14030v1](http://arxiv.org/abs/2511.14030v1)

**作者:** Sungik Choi, Hankook Lee, Moontae Lee

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为WaRPAD的无需训练的AI生成图像检测算法，利用自监督模型在不同分辨率下产生一致表示的特点来检测AI生成的图像。

### 背景

随着视觉生成模型的快速发展，AI生成图像检测变得至关重要。传统方法通常针对特定数据集训练检测器，需要先验数据知识。

### 目的

研究一种无需训练的方法，利用自监督模型且不需要先验数据知识，提出一种基于自监督模型的无需训练的AI生成图像检测算法。

### 方法

定义基础分数函数量化图像嵌入对Haar小波分解提取的高频方向扰动的敏感性；将图像缩放到模型输入大小的倍数并分割成小块，计算每个小块的基础分数；通过平均所有小块的分数获得最终检测分数。

### 主要发现

WaRPAD在不同分辨率和域的真实数据集以及23种不同生成模型生成的图像上验证；方法一致具有竞争力，对测试时干扰表现出强鲁棒性；适用于各种自监督模型，因为对RandomResizedCrop的不变性是自监督模型的常见训练方案。

### 结论

WaRPAD是一种有效的无需训练的AI生成图像检测方法，基于自监督模型，适用于多种场景和模型。

### 翻译

AI生成图像检测随着视觉生成模型的快速发展而变得至关重要。我们研究了一种无需训练的方法，利用自监督模型且不需要先验数据知识。这些使用RandomResizedCrop等增强方法预训练的模型，能够学习在不同分辨率下产生一致的表示。受此启发，我们提出了WaRPAD，一种基于自监督模型的无需训练的AI生成图像检测算法。由于图像中邻域像素差异对调整大小操作高度敏感，WaRPAD首先定义了一个基础分数函数，量化图像嵌入对通过Haar小波分解提取的高频方向扰动的敏感性。为模拟对裁剪增强的鲁棒性，我们将图像缩放到模型输入大小的倍数，将其分成更小的块，并计算每个块的基础分数。最终的检测分数是通过平均所有块的分数获得的。我们在不同分辨率和域的真实数据集以及23种不同生成模型生成的图像上验证了WaRPAD。我们的方法一致具有竞争力，并表现出对测试时干扰的强鲁棒性。此外，由于对RandomResizedCrop的不变性是自监督模型的常见训练方案，我们证明WaRPAD适用于各种自监督模型。


### 论文摘要

AI-generated image detection has become crucial with the rapid advancement of vision-generative models. Instead of training detectors tailored to specific datasets, we study a training-free approach leveraging self-supervised models without requiring prior data knowledge. These models, pre-trained with augmentations like RandomResizedCrop, learn to produce consistent representations across varying resolutions. Motivated by this, we propose WaRPAD, a training-free AI-generated image detection algorithm based on self-supervised models. Since neighborhood pixel differences in images are highly sensitive to resizing operations, WaRPAD first defines a base score function that quantifies the sensitivity of image embeddings to perturbations along high-frequency directions extracted via Haar wavelet decomposition. To simulate robustness against cropping augmentation, we rescale each image to a multiple of the models input size, divide it into smaller patches, and compute the base score for each patch. The final detection score is then obtained by averaging the scores across all patches. We validate WaRPAD on real datasets of diverse resolutions and domains, and images generated by 23 different generative models. Our method consistently achieves competitive performance and demonstrates strong robustness to test-time corruptions. Furthermore, as invariance to RandomResizedCrop is a common training scheme across self-supervised models, we show that WaRPAD is applicable across self-supervised models.

---

## 160. Self-Supervised Compression and Artifact Correction for Streaming Underwater Imaging Sonar

**论文链接:** [http://arxiv.org/abs/2511.13922v1](http://arxiv.org/abs/2511.13922v1)

**作者:** Rongsheng Qian, Chi Xu, Xiaoqiang Ma, Hao Fang, Yili Jin, William I. Atlas, Jiangchuan Liu

**发布时间:** 2025-11-17

**备注:** Accepted to WACV 2026

### GPT解析

### 总结

SCOPE是一个自监督框架，能够同时进行声纳图像的压缩和伪影校正，解决了实时成像声纳中带宽有限和声纳特定伪影的问题。该系统已在实际环境中部署，用于支持鲑鱼计数和环境监测。

### 背景

实时成像声纳在光学传感不可靠的水下环境中已成为重要的监测工具。然而，其广泛应用受到两个相互关联的挑战限制：上行链路带宽极其有限，以及严重的声纳特定伪影（斑点、运动模糊、混响、声学阴影）影响高达98%的帧。

### 目的

开发一个自监督框架，能够同时进行压缩和伪影校正，不需要干净的-噪声对或合成假设，以解决实时成像声纳中的带宽和伪影问题。

### 方法

SCOPE结合了两种技术：1)自适应码本压缩（ACC），学习针对声纳定制的频率编码的潜在表示；2)频率感知多尺度分割（FAMS），将帧分解为低频结构和稀疏高频动态，同时抑制快速波动的伪影。此外，采用对冲训练策略，使用无标签生成的低通代理对来引导频率感知学习。

### 主要发现

在数月的现场ARIS声纳数据评估中，SCOPE实现了0.77的结构相似性指数（SSIM），比先前的自监督去噪基线提高了40%。在低至0.0118 bpp的比特率下工作，将上行链路带宽减少了80%以上，同时提高了下游检测性能。系统实时运行：嵌入式GPU上编码时间为3.1毫秒，服务器端完整多层解码时间为97毫秒。已在太平洋西北部的三条河流中部署数月。

### 结论

学习频率结构的潜在表示使得在实际部署条件下能够实现实用的低比特率声纳流传输，同时保留信号细节。

### 翻译

实时成像声纳已成为光学传感不可靠的水下环境监测的重要工具。其更广泛的应用受到两个相互关联的挑战的限制：极其有限的上行链路带宽和严重的声纳特定伪影（斑点、运动模糊、混响、声学阴影），这些伪影影响高达98%的帧。我们提出了SCOPE，一个自监督框架，能够同时执行压缩和伪影校正，不需要干净的-噪声对或合成假设。SCOPE结合了（i）自适应码本压缩（ACC），它学习针对声纳定制的频率编码的潜在表示，以及（ii）频率感知多尺度分割（FAMS），它将帧分解为低频结构和稀疏高频动态，同时抑制快速波动的伪影。对冲训练策略进一步使用无标签生成的低通代理对来引导频率感知学习。在数月的现场ARIS声纳数据评估中，SCOPE实现了0.77的结构相似性指数（SSIM），比先前的自监督去噪基线提高了40%，在低至0.0118 bpp的比特率下工作。它在减少80%以上的上行链路带宽的同时提高了下游检测性能。该系统实时运行，嵌入式GPU上的编码时间为3.1毫秒，服务器端完整多层解码时间为97毫秒。SCOPE已在太平洋西北部的三条河流中部署数月，用于支持野生环境中的实时鲑鱼计数和环境监测。结果表明，学习频率结构的潜在表示使得在实际部署条件下能够实现实用的低比特率声纳流传输，同时保留信号细节。


### 论文摘要

Real-time imaging sonar has become an important tool for underwater monitoring in environments where optical sensing is unreliable. Its broader use is constrained by two coupled challenges: highly limited uplink bandwidth and severe sonar-specific artifacts (speckle, motion blur, reverberation, acoustic shadows) that affect up to 98% of frames. We present SCOPE, a self-supervised framework that jointly performs compression and artifact correction without clean-noise pairs or synthetic assumptions. SCOPE combines (i) Adaptive Codebook Compression (ACC), which learns frequency-encoded latent representations tailored to sonar, with (ii) Frequency-Aware Multiscale Segmentation (FAMS), which decomposes frames into low-frequency structure and sparse high-frequency dynamics while suppressing rapidly fluctuating artifacts. A hedging training strategy further guides frequency-aware learning using low-pass proxy pairs generated without labels. Evaluated on months of in-situ ARIS sonar data, SCOPE achieves a structural similarity index (SSIM) of 0.77, representing a 40% improvement over prior self-supervised denoising baselines, at bitrates down to <= 0.0118 bpp. It reduces uplink bandwidth by more than 80% while improving downstream detection. The system runs in real time, with 3.1 ms encoding on an embedded GPU and 97 ms full multi-layer decoding on the server end. SCOPE has been deployed for months in three Pacific Northwest rivers to support real-time salmon enumeration and environmental monitoring in the wild. Results demonstrate that learning frequency-structured latents enables practical, low-bitrate sonar streaming with preserved signal details under real-world deployment conditions.

---

## 161. SAE-MCVT: A Real-Time and Scalable Multi-Camera Vehicle Tracking Framework Powered by Edge Computing

**论文链接:** [http://arxiv.org/abs/2511.13904v1](http://arxiv.org/abs/2511.13904v1)

**作者:** Yuqiang Lin, Sam Lockyer, Florian Stanek, Markus Zarbock, Adrian Evans, Wenbin Li, Nic Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

提出SAE-MCVT，首个可扩展的实时多摄像头车辆跟踪框架，解决了现有研究忽视实时性和可扩展性的问题

### 背景

在现代智能交通系统中，摄像头是关键组件，多摄像头车辆跟踪(MCVT)能生成车辆轨迹，支持异常检测、交通密度估计和嫌疑车辆跟踪等应用

### 目的

开发一个兼顾实时性能和可扩展性的MCVT框架，以满足城市规模应用的需求

### 方法

SAE-MCVT系统包含边缘设备和中央工作站。边缘设备处理实时RTSP视频流，执行目标检测、跟踪、地理映射和特征提取，仅将车辆位置和深度外观特征等轻量级元数据传输至中央工作站。中央工作站基于相邻摄像头间时空关系约束进行跨摄像头关联，这些关系通过自监督摄像头链接模型学习

### 主要发现

在RoundaboutHD数据集上，SAE-MCVT能够在2K 15 FPS视频流上保持实时运行，并达到61.2的IDF1分数

### 结论

据所知，这是首个适合城市规模部署的可扩展实时MCVT框架

### 翻译

在现代智能交通系统中，摄像头是关键组件，因为它们能够为多个利益相关者提供有价值的信息。中心任务是多摄像头车辆跟踪(MCVT)，它生成车辆轨迹并支持异常检测、交通密度估计和嫌疑车辆跟踪等应用。然而，大多数现有的MCVT研究强调准确性而忽视了实时性能和可扩展性。这两个方面对于实际部署至关重要，并且随着摄像头数量的增加，在城市规模应用中变得越来越具有挑战性。为了解决这个问题，我们提出了SAE-MCVT，这是第一个可扩展的实时MCVT框架。该系统包含多个边缘设备，每个设备与一个中央工作站单独交互。在边缘端，实时RTSP视频流被序列化并通过包括目标检测、目标跟踪、地理映射和特征提取等模块进行处理。只有轻量级的元数据——车辆位置和深度外观特征——被传输到中央工作站。在中央端，在相邻摄像头之间的时空关系约束下计算跨摄像头关联，这些关系通过自监督的摄像头链接模型学习。在RoundaboutHD数据集上的实验表明，SAE-MCVT在2K 15 FPS视频流上保持实时运行，并达到61.2的IDF1分数。据我们所知，这是第一个适合城市规模部署的可扩展实时MCVT框架。


### 论文摘要

In modern Intelligent Transportation Systems (ITS), cameras are a key component due to their ability to provide valuable information for multiple stakeholders. A central task is Multi-Camera Vehicle Tracking (MCVT), which generates vehicle trajectories and enables applications such as anomaly detection, traffic density estimation, and suspect vehicle tracking. However, most existing studies on MCVT emphasize accuracy while overlooking real-time performance and scalability. These two aspects are essential for real-world deployment and become increasingly challenging in city-scale applications as the number of cameras grows. To address this issue, we propose SAE-MCVT, the first scalable real-time MCVT framework. The system includes several edge devices that interact with one central workstation separately. On the edge side, live RTSP video streams are serialized and processed through modules including object detection, object tracking, geo-mapping, and feature extraction. Only lightweight metadata -- vehicle locations and deep appearance features -- are transmitted to the central workstation. On the central side, cross-camera association is calculated under the constraint of spatial-temporal relations between adjacent cameras, which are learned through a self-supervised camera link model. Experiments on the RoundaboutHD dataset show that SAE-MCVT maintains real-time operation on 2K 15 FPS video streams and achieves an IDF1 score of 61.2. To the best of our knowledge, this is the first scalable real-time MCVT framework suitable for city-scale deployment.

---

## 162. AnaCP: Toward Upper-Bound Continual Learning via Analytic Contrastive Projection

**论文链接:** [http://arxiv.org/abs/2511.13880v1](http://arxiv.org/abs/2511.13880v1)

**作者:** Saleh Momeni, Changnan Xiao, Bing Liu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了一种名为AnaCP(Analytic Contrastive Projection)的新方法，用于解决类增量学习中的灾难性遗忘问题，通过结合预训练模型和增量特征适应，实现了与联合训练相当的性能。

### 背景

类增量学习(CIL)是持续学习中的一个核心设置，模型需要学习一系列包含不同类别的任务。传统CIL方法不利用预训练模型，会遭遇灾难性遗忘问题；而集成预训练模型的方法虽然效率高，但无法持续调整特征表示以适应新任务，导致性能受限。

### 目的

解决现有类增量学习方法中无法持续适应特征表示的问题，消除梯度更新导致的灾难性遗忘，同时保持解析分类器的效率。

### 方法

提出AnaCP(Analytic Contrastive Projection)方法，保留解析分类器的效率，同时实现增量特征适应，无需基于梯度的训练，从而避免灾难性遗忘。

### 主要发现

实验表明，AnaCP不仅优于现有基线方法，还达到了联合训练的准确率水平，而联合训练被视为类增量学习的性能上限。

### 结论

AnaCP方法成功地解决了类增量学习中的灾难性遗忘问题，通过结合预训练模型和增量特征适应，实现了与联合训练相当的性能，为类增量学习提供了新的解决方案。

### 翻译

本文研究了类增量学习(CIL)问题，这是持续学习中的一个核心设置，模型学习一系列任务，每个任务包含不同的类别集合。不利用预训练模型(PTMs)的传统CIL方法由于需要增量学习特征表示和分类器而遭受灾难性遗忘(CF)。将PTM集成到CIL中最近导致了高效的方法，它们将PTM作为固定特征提取器与解析分类器结合，取得了最先进的性能。然而，它们仍然面临一个主要局限：无法持续调整特征表示以最好地适应CIL任务，导致次优性能。为此，我们提出了AnaCP(Analytic Contrastive Projection)，一种新方法，它在保留解析分类器效率的同时，实现增量特征适应而无需基于梯度的训练，从而消除了梯度更新导致的CF。我们的实验表明，AnaCP不仅优于现有基线方法，还达到了被视为CIL上限的联合训练的准确率水平。


### 论文摘要

This paper studies the problem of class-incremental learning (CIL), a core setting within continual learning where a model learns a sequence of tasks, each containing a distinct set of classes. Traditional CIL methods, which do not leverage pre-trained models (PTMs), suffer from catastrophic forgetting (CF) due to the need to incrementally learn both feature representations and the classifier. The integration of PTMs into CIL has recently led to efficient approaches that treat the PTM as a fixed feature extractor combined with analytic classifiers, achieving state-of-the-art performance. However, they still face a major limitation: the inability to continually adapt feature representations to best suit the CIL tasks, leading to suboptimal performance. To address this, we propose AnaCP (Analytic Contrastive Projection), a novel method that preserves the efficiency of analytic classifiers while enabling incremental feature adaptation without gradient-based training, thereby eliminating the CF caused by gradient updates. Our experiments show that AnaCP not only outperforms existing baselines but also achieves the accuracy level of joint training, which is regarded as the upper bound of CIL.

---

## 163. Dual Origins of Rapid Flare Ribbon Downflows in an X9-class Solar Flare

**论文链接:** [http://arxiv.org/abs/2511.13862v1](http://arxiv.org/abs/2511.13862v1)

**作者:** Ryan J. French, William H. Ashfield, Cole A. Tamburri, Maria D. Kazachenko, Marie Dominique, Marcel Corchado Albelo, Amir Caspi

**发布时间:** 2025-11-17

**备注:** 14 pages, 6 figures, accepted for publication to ApJ

### GPT解析

### 总结

本研究在2024年10月3日的X9级太阳耀斑中检测到速度为150-217 km/s的快速下落流，这些下落流持续超过15分钟，分为两个不同阶段，表明多种机制导致耀斑带等离子向下加速。

### 背景

研究使用IRIS Si IV 1402.77 nm测量数据，结合ASO-S/HXI和LYRA Lyman-alpha观测，分析太阳耀斑中的快速等离子体运动现象。

### 目的

探究太阳耀斑中快速下落流的物理机制及其在不同阶段的行为特征，理解耀斑带等离子体的动力学过程。

### 方法

结合IRIS Si IV光谱数据、ASO-S/HXI硬X射线成像、LYRA Lyman-alpha测量以及硬X射线光谱分析，并使用机器学习K-means聚类方法量化谱线轮廓变化。

### 主要发现

1. 快速下落流分为两个阶段：第一阶段与高能发射同步，由色球层凝聚引起；第二阶段持续下落但高能发射已恢复背景水平，反映日冕雨现象。2. Si IV多普勒速度在整个演化中表现出50秒周期的准周期性脉动，可能与磁拱中的MHD振荡有关。

### 结论

太阳耀斑中的快速下落流由多种机制共同作用，初始阶段与色球层凝聚相关，后期阶段与日冕雨有关，且整个过程中存在准周期性脉动，表明磁拱中的MHD振荡可能影响等离子体运动。

### 翻译

我们在2024年10月3日的X9级太阳耀斑的IRIS Si IV 1402.77 nm测量中检测到150-217 km/s的快速下落流。这些快速红移值在耀斑开始后持续超过15分钟，可分为两个不同的行为阶段，表明多种机制导致耀斑带等子的向下加速。快速下落的第一阶段与ASO-S/HXI和LYRA Lyman-alpha测量的发射峰值同步，表明色球层下落流（最大红移176 km/s）是由太阳耀斑中色球层凝聚与能量释放引起的。在事件后期，尽管磁通量率降为零，高能HXR和Lyman-alpha测量恢复到背景水平，但强烈的Si IV耀斑带下落流仍然持续（最大值217 km/s），这反映了耀斑诱导的日冕雨在耀斑带足点的下落。硬X射线光谱分析支持这一场景，显示初始下落流阶段有强烈的非热发射，到第二阶段下降到接近背景水平。尽管这些不同的耀斑带行为阶段，Si IV多普勒速度在整个15分钟的耀斑演化中表现出准周期性脉动，周期恒定约为50秒（与环长度无关）。我们推断这些脉动可能是由磁拱中的MHD振荡引起的。最后，我们使用机器学习K-means聚类方法来量化快速下落流阶段的谱线轮廓变化。


### 论文摘要

We detect rapid downflows of 150-217 km/s in IRIS Si IV 1402.77 nm measurements of an X9-class solar flare on 2024 October 3rd. The fast redshift values persist for over 15 minutes from flare onset, and can be split into two distinct stages of behavior, suggesting that multiple mechanisms are responsible for the downwards acceleration of flare ribbon plasma. The first stage of rapid downflows are synchronized with peaks in emission from the Advanced Space-based Solar Observatory Hard X-ray Imager (ASO-S/HXI) and Large Yield Radiometer (LYRA) Lyman-alpha measurements, indicative that the chromospheric downflows (with a maximum redshift of 176 km/s) result from chromospheric condensations associated with impulsive energy release in the solar flare. Later in the event, strong Si IV flare ribbon downflows persist (to a maximum value of 217 km/s), despite the magnetic flux rate falling to zero, and high-energy HXR and Lyman-alpha measurements returning to background levels. This is reflective of downflows in the flare ribbon footpoints of flare-induced coronal rain. Hard X-ray spectral analysis supports this scenario, revealing strong non-thermal emission during the initial downflow stage, falling near background levels by the second stage. Despite these distinct and contrasting stages of ribbon behavior, Si IV Doppler velocities exhibit quasi-periodic pulsations with a constant ~50 s period across the 15-minute flare evolution (independently of loop length). We deduce that these pulsations are likely caused by MHD oscillations in the magnetic arcade. Finally, we utilize machine learning K-means clustering methods to quantify line profile variations during the stages of rapid downflows.

---

## 164. 论文ID: 2511.14744v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.14744v1.json'

---

## 165. Hyperbolic Graph Embeddings Reveal the Host-Pathogen Interactome

**论文链接:** [http://arxiv.org/abs/2511.14669v1](http://arxiv.org/abs/2511.14669v1)

**作者:** Xiaoqiong Xia, Cesar de la Fuente-Nunez

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文介绍了一种名为ApexPPI的深度学习框架，利用双曲空间表示蛋白质网络，整合多模态生物数据预测病原体与宿主蛋白质相互作用。该方法比传统方法更准确，识别出数千种高置信度相互作用，并通过AlphaFold 3验证了预测准确性，为宿主-病原体相互作用提供了全面图谱，有助于新治疗方法发现。

### 背景

感染取决于病原体和宿主蛋白质之间的相互作用，但全面绘制这些相互作用具有挑战性且劳动密集。许多生物网络具有层次化、无标度结构。

### 目的

开发一种能够准确预测病原体与宿主蛋白质相互作用的计算方法，以克服传统实验方法的局限性。

### 方法

创建名为ApexPPI的深度学习框架，在双曲黎曼空间中表示蛋白质网络以捕获层次化、无标度特征。整合多模态生物数据（蛋白质序列、基因扰动实验和互补相互作用网络），通过多任务双曲图神经网络预测相互作用。

### 主要发现

将蛋白质特征映射到双曲空间比以往方法在预测宿主-病原体相互作用方面具有更高准确性。从数千万种可能的蛋白质对中，识别出数千种高置信度相互作用，包括许多涉及人G蛋白偶联受体的相互作用。使用AlphaFold 3结构建模验证了数十种预测复合物的准确性。

### 结论

全面的宿主-病原体蛋白质相互作用图谱为发现新治疗方法提供了资源，展示了先进人工智能如何能够解开复杂生物系统。

### 翻译

感染取决于病原体和宿主蛋白质之间的相互作用，但全面绘制这些相互作用具有挑战性且劳动密集。许多生物网络具有层次化、无标度结构，因此我们开发了一个名为ApexPPI的深度学习框架，它在双曲黎曼空间中表示蛋白质网络以捕获这些特征。我们的模型整合了多模态生物数据（蛋白质序列、基因扰动实验和互补相互作用网络），通过多任务双曲图神经网络预测病原体和宿主蛋白质之间的可能相互作用。将蛋白质特征映射到双曲空间比以往方法在预测宿主-病原体相互作用方面具有更高的准确性。从数千万种可能的蛋白质对中，我们的模型确定了数千种高置信度的相互作用，包括许多涉及人G蛋白偶联受体的相互作用。我们使用AlphaFold 3结构建模验证了数十种这些预测的复合物，支持了我们预测的准确性。这张全面的宿主-病原体蛋白质相互作用图谱为发现新治疗方法提供了资源，并展示了先进的人工智能如何能够解开复杂的生物系统。


### 论文摘要

Infections depend on interactions between pathogen and host proteins, but comprehensively mapping these interactions is challenging and labor intensive. Many biological networks have hierarchical, scale-free structure, so we developed a deep learning framework, ApexPPI, that represents protein networks in hyperbolic Riemannian space to capture these features. Our model integrates multimodal biological data (protein sequences, gene perturbation experiments, and complementary interaction networks) to predict likely interactions between pathogen and host proteins through multi-task hyperbolic graph neural networks. Mapping protein features into hyperbolic space led to much higher accuracy than previous methods in predicting host-pathogen interactions. From tens of millions of possible protein pairs, our model identified thousands of high-confidence interactions, including many involving human G-protein-coupled receptors (GPCRs). We validated dozens of these predicted complexes using AlphaFold 3 structural modeling, supporting the accuracy of our predictions. This comprehensive map of host-pathogen protein interactions provides a resource for discovering new treatments and illustrates how advanced AI can unravel complex biological systems.

---

## 166. A Neuro-Symbolic Framework for Reasoning under Perceptual Uncertainty: Bridging Continuous Perception and Discrete Symbolic Planning

**论文链接:** [http://arxiv.org/abs/2511.14533v1](http://arxiv.org/abs/2511.14533v1)

**作者:** Jiahao Wu, Shengwen Yu

**发布时间:** 2025-11-18

**备注:** 29 pages, 10 figures, 12 tables

### GPT解析

### 总结

该研究提出了一种神经符号框架，通过连接连续感知和离散符号推理来解决AI系统中的不确定性处理问题，在机器人任务中表现出色，并提供了理论保证。

### 背景

在AI系统中，连接连续感知信号和离散符号推理是一个基本挑战，特别是在不确定性条件下运行的系统。

### 目的

提出一个神经符号框架，明确建模和传播从感知到规划的不确定性，为这两个抽象层次之间提供有原则的连接。

### 方法

结合基于transformer的感知前端和图神经网络（GNN）关系推理，从视觉观测中提取概率符号状态；使用不确定性感知的符号规划器，在置信度低时主动收集信息；在桌面机器人操作中作为具体应用展示框架的有效性；使用概率图形模型分析校准不确定性和规划收敛之间的定量联系。

### 主要发现

翻译器处理了10,047个PyBullet生成的场景（3-10个物体），输出具有校准置信度的概率谓词（总体F1=0.68）；系统在简单堆叠、深度堆叠和清除+堆叠基准测试中分别实现了94%/90%/88%的成功率（平均90.7%）；超越了最强的POMDP基线10-14分，同时规划时间在15毫秒内；建立了校准不确定性和规划收敛之间的定量联系，提供了经验验证的理论保证。

### 结论

该框架是通用目的的，可以应用于任何需要从感知输入到符号规划的不确定性感知推理的领域。

### 翻译

连接连续感知信号和离散符号推理是在不确定性条件下运行的AI系统中的一个基本挑战。我们提出了一个神经符号框架，明确建模和传播从感知到规划的不确定性，为这两个抽象层次之间提供了有原则的连接。我们的方法将基于transformer的感知前端与图神经网络（GNN）关系推理相结合，从视觉观测中提取概率符号状态，并使用一个不确定性感知的符号规划器，在置信度低时主动收集信息。我们在桌面机器人操作中作为具体应用展示了框架的有效性：翻译器处理了10,047个PyBullet生成的场景（3-10个物体），并输出具有校准置信度的概率谓词（总体F1=0.68）。当嵌入规划器时，系统在简单堆叠、深度堆叠和清除+堆叠基准测试中分别实现了94%/90%/88%的成功率（平均90.7%），超过了最强的POMDP基线10-14分，同时规划时间在15毫秒内。概率图形模型分析建立了校准不确定性和规划收敛之间的定量联系，提供了经验验证的理论保证。该框架是通用目的的，可以应用于任何需要从感知输入到符号规划的不确定性感知推理的领域。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何将连续的感知信号与离散的符号推理桥接起来，特别是在感知不确定性的情况下进行推理。这个问题在现实世界中非常重要，因为机器人、自动驾驶汽车等AI系统必须在不确定性环境下操作，而传统方法要么假设完美的感知（不现实），要么完全在感知层面操作（缺乏可解释性和泛化能力）。现有神经符号方法通常假设确定的符号状态，不处理不确定性；而POMDP方法虽然处理不确定性，但面临计算瓶颈和缺乏符号抽象的挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别现有方法的局限性来设计新方法：纯神经方法缺乏可解释性，经典符号规划器假设完美感知，现有神经符号方法不处理不确定性，POMDP面临计算瓶颈和符号抽象差距。作者借鉴了神经符号AI、POMDP、信息收集、图神经网络和Transformer等现有工作，但创新性地将这些技术结合，特别关注不确定性建模和传播。作者设计了结合Transformer感知前端与图神经网络的关系推理，以及不确定性感知的符号规划器，在置信度低时主动收集信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 明确建模和传播从感知到规划的不确定性；2) 使用概率符号表示而非确定的符号状态；3) 当置信度低于阈值时触发信息收集动作；4) 考虑谓词之间的依赖关系。整体流程包括：1) 神经符号转换器处理视觉输入输出概率符号状态；2) 不确定性处理将谓词分类；3) 符号规划器基于确定谓词生成动作序列；4) 闭环执行包括感知、预测、规划和执行，必要时触发信息收集。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：理论贡献(校准-收敛链接、依赖感知的不确定性建模、最优阈值选择)、方法贡献(不确定性感知符号规划框架、混合Transformer-GNN架构、关系特定自适应阈值、不确定性驱动的信息收集)和实证贡献(理论验证、全面评估、可重现性)。相比之前工作，本文的主要不同是：1) 明确处理感知不确定性而非假设确定性；2) 将计算复杂度从O(|O|^d)降低到O(b^d)；3) 在可解释符号谓词上操作而非原始像素；4) 提供不确定性校准与规划收敛的定量联系。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种神经符号框架，通过明确建模和传播从感知到规划的不确定性，桥接了连续感知与离散符号推理之间的差距，并在机器人操作任务中实现了高效、可解释的决策制定。'}


### 论文摘要

Bridging continuous perceptual signals and discrete symbolic reasoning is a fundamental challenge in AI systems that must operate under uncertainty. We present a neuro-symbolic framework that explicitly models and propagates uncertainty from perception to planning, providing a principled connection between these two abstraction levels. Our approach couples a transformer-based perceptual front-end with graph neural network (GNN) relational reasoning to extract probabilistic symbolic states from visual observations, and an uncertainty-aware symbolic planner that actively gathers information when confidence is low. We demonstrate the framework's effectiveness on tabletop robotic manipulation as a concrete application: the translator processes 10,047 PyBullet-generated scenes (3--10 objects) and outputs probabilistic predicates with calibrated confidences (overall F1=0.68). When embedded in the planner, the system achieves 94\%/90\%/88\% success on Simple Stack, Deep Stack, and Clear+Stack benchmarks (90.7\% average), exceeding the strongest POMDP baseline by 10--14 points while planning within 15\,ms. A probabilistic graphical-model analysis establishes a quantitative link between calibrated uncertainty and planning convergence, providing theoretical guarantees that are validated empirically. The framework is general-purpose and can be applied to any domain requiring uncertainty-aware reasoning from perceptual input to symbolic planning.

---

## 167. Certified Signed Graph Unlearning

**论文链接:** [http://arxiv.org/abs/2511.14168v1](http://arxiv.org/abs/2511.14168v1)

**作者:** Junpeng Zhao, Lin Li, Kaixi Hu, Kaize Shi, Jingling Yuan

**发布时间:** 2025-11-18

### GPT解析

### 总结

本文提出了一种名为认证符号图遗忘(CSGU)的新方法，解决了符号图神经网络中隐私保护的关键挑战，通过三阶段方法在保留符号信息的同时提供可证明的隐私保证。

### 背景

有符号图通过正边和负边建模复杂关系，有广泛现实应用。由于数据敏感性，选择性删除机制对隐私保护至关重要。现有图遗忘方法专为传统GNN设计，忽略了符号图的独特异构特性，应用于SGNNs时会失去关键符号信息，导致模型效用和遗忘效果下降。

### 目的

开发一种能够保留符号图关键信息并提供可证明隐私保证的遗忘方法，解决现有方法在SGNNs上的局限性，同时保持模型效用和遗忘效果。

### 方法

提出认证符号图遗忘(CSGU)，采用三阶段方法：(1)通过三角结构高效识别最小影响邻域；(2)应用社会学理论量化节点重要性，实现最佳隐私预算分配；(3)执行重要性加权的参数更新，实现经过认证的修改同时最小化效用降级。

### 主要发现

广泛实验证明CSGU优于现有方法，在SGNNs上实现了卓越的效用保持和遗忘效果，解决了现有方法在符号图处理中的关键问题。

### 结论

CSGU通过保留关键符号信息并应用社会学原理，实现了符号图神经网络中更好的隐私-效用权衡，为敏感图数据的隐私保护提供了有效解决方案。

### 翻译

有符号图通过正边和负边建模复杂关系，具有广泛的现实应用。鉴于此类数据的敏感性，选择性删除机制已成为隐私保护的关键。虽然图遗忘允许从图神经网络(GNNs)中移除特定数据的影响，但现有方法是为传统GNN设计的，忽略了符号图的独特异构特性。当应用于符号图神经网络(SGNNs)时，这些方法会失去关键的符号信息，降低模型效用和遗忘效果。为解决这些挑战，我们提出了认证符号图遗忘(CSGU)，在提供可证明的隐私保证的同时，保留SGNNs背后的社会学原理。CSGU采用三阶段方法：(1)通过三角结构高效识别最小影响邻域；(2)应用社会学理论量化节点重要性，实现最佳隐私预算分配；(3)执行重要性加权的参数更新，实现经过认证的修改同时最小化效用降级。大量实验证明，CSGU优于现有方法，在SGNNs的效用保持和遗忘效果方面均实现了卓越性能。


### 论文摘要

Signed graphs model complex relationships through positive and negative edges, with widespread real-world applications. Given the sensitive nature of such data, selective removal mechanisms have become essential for privacy protection. While graph unlearning enables the removal of specific data influences from Graph Neural Networks (GNNs), existing methods are designed for conventional GNNs and overlook the unique heterogeneous properties of signed graphs. When applied to Signed Graph Neural Networks (SGNNs), these methods lose critical sign information, degrading both model utility and unlearning effectiveness. To address these challenges, we propose Certified Signed Graph Unlearning (CSGU), which provides provable privacy guarantees while preserving the sociological principles underlying SGNNs. CSGU employs a three-stage method: (1) efficiently identifying minimal influenced neighborhoods via triangular structures, (2) applying sociological theories to quantify node importance for optimal privacy budget allocation, and (3) performing importance-weighted parameter updates to achieve certified modifications with minimal utility degradation. Extensive experiments demonstrate that CSGU outperforms existing methods, achieving superior performance in both utility preservation and unlearning effectiveness on SGNNs.

---

## 168. Complex-Weighted Convolutional Networks: Provable Expressiveness via Complex Diffusion

**论文链接:** [http://arxiv.org/abs/2511.13937v1](http://arxiv.org/abs/2511.13937v1)

**作者:** Cristina López Amado, Tassilo Schwarz, Yu Tian, Renaud Lambiotte

**发布时间:** 2025-11-17

**备注:** 19 pages, 6 figures. Learning on Graphs Conference 2025

### GPT解析

### 总结

本文提出了一种复加权图神经网络框架，通过为图边分配复数来驱动扩散过程，解决了传统GNNs的过平滑和对异质图性能不佳的问题。

### 背景

图神经网络(GNNs)在多种应用中取得了显著成功，但它们仍然受到过平滑和对异质图性能不佳的限制。

### 目的

解决GNNs面临的过平滑和对异质图性能不佳的挑战。

### 方法

引入了一个新颖的框架，为图赋予复加权结构，为每条边分配一个复数驱动扩散过程；证明了这种扩散的高度表达性；提出了复加权卷积网络(CWCN)，直接从数据中学习适合的复加权结构，同时用可学习矩阵和非线性激活来丰富扩散过程。

### 主要发现

复加权扩散为增强GNN表达性提供了原则性且通用的机制，任何节点分类任务都可以在复随机游走的稳态中解决。

### 结论

CWCN简单易实现，不需要额外超参数，在基准数据集上取得了有竞争力的性能，为既具有理论基础又具有实际有效性的模型开辟了新途径。

### 翻译

图神经网络(GNNs)已在各种应用中取得了显著成功，但它们仍然受到过平滑和对异质图性能不佳的限制。为应对这些挑战，我们引入了一种新颖的框架，为图赋予复加权结构，为每条边分配一个复数来驱动扩散过程，将随机游走扩展到复数域。我们证明这种扩散具有高度表达性：通过适当选择复权重，任何节点分类任务都可以在复随机游走的稳态中解决。基于这一见解，我们提出了复加权卷积网络(CWCN)，它直接从数据中学习适合的复加权结构，同时用可学习矩阵和非线性激活来丰富扩散过程。CWCN易于实现，除标准GNNs外不需要额外超参数，并在基准数据集上取得了有竞争力的性能。我们的结果表明，复加权扩散为增强GNN表达性提供了原则性且通用的机制，为既具有理论基础又具有实际有效性的模型开辟了新途径。


### 论文摘要

Graph Neural Networks (GNNs) have achieved remarkable success across diverse applications, yet they remain limited by oversmoothing and poor performance on heterophilic graphs. To address these challenges, we introduce a novel framework that equips graphs with a complex-weighted structure, assigning each edge a complex number to drive a diffusion process that extends random walks into the complex domain. We prove that this diffusion is highly expressive: with appropriately chosen complex weights, any node-classification task can be solved in the steady state of a complex random walk. Building on this insight, we propose the Complex-Weighted Convolutional Network (CWCN), which learns suitable complex-weighted structures directly from data while enriching diffusion with learnable matrices and nonlinear activations. CWCN is simple to implement, requires no additional hyperparameters beyond those of standard GNNs, and achieves competitive performance on benchmark datasets. Our results demonstrate that complex-weighted diffusion provides a principled and general mechanism for enhancing GNN expressiveness, opening new avenues for models that are both theoretically grounded and practically effective.

---

## 169. Fairness-Aware Graph Representation Learning with Limited Demographic Information

**论文链接:** [http://arxiv.org/abs/2511.13540v2](http://arxiv.org/abs/2511.13540v2)

**作者:** Zichong Wang, Zhipeng Yin, Liping Yang, Jun Zhuang, Rui Yu, Qingzhao Kong, Wenbin Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了FairGLite框架，一种在有限人口统计信息情况下减轻图神经网络偏见的新方法，通过代理生成机制、一致性嵌入策略和自适应置信度策略实现公平性，同时保持模型效用。

### 背景

确保图神经网络公平性对促进可信且社会责任感的机器学习系统至关重要。虽然近年提出多种公平图学习方法，但大多假设可完全获取人口统计信息，这在实践中因隐私、法律或监管限制而难以实现。

### 目的

开发一个公平图学习框架，在有限人口统计信息条件下减轻图学习中的偏见问题。

### 方法

提出由部分人口统计数据引导的代理生成机制；设计在人口统计组间强制节点嵌入保持一致性的策略；开发基于预测置信度动态调整节点对公平性和效用贡献的自适应置信度策略；提供理论分析证明框架在群体公平性指标上可实现可证明上界。

### 主要发现

通过在多个数据集和公平图学习框架上的大量实验，验证了FairGLite框架在减轻偏见和保持模型效用方面的有效性。

### 结论

FairGLite框架能够在有限人口统计信息情况下有效减轻图学习中的偏见，同时保持模型效用，为实际应用提供了可行的公平图学习解决方案。

### 翻译

确保图神经网络中的公平性对于促进可信且具有社会责任感的机器学习系统至关重要。为此，近年来提出了许多公平图学习方法。然而，它们中的大多数假设可以完全获取人口统计信息，由于隐私、法律或监管限制，这一要求在实践中很少能够满足。为此，本文引入了一种新颖的公平图学习框架，可在有限人口统计信息的情况下减轻图学习中的偏见。具体而言，我们提出了一种由部分人口统计数据引导的机制来生成人口统计信息的代理，并设计了一种在人口统计组之间强制节点嵌入保持一致性的策略。此外，我们开发了一种自适应置信度策略，根据预测置信度动态调整每个节点对公平性和效用的贡献。我们进一步提供了理论分析，证明我们的框架FairGLite在群体公平性指标上可实现可证明的上界，为偏见减轻提供了正式保证。通过在多个数据集和公平图学习框架上的大量实验，我们证明了该框架在减轻偏见和保持模型效用方面的有效性。


### 论文摘要

Ensuring fairness in Graph Neural Networks is fundamental to promoting trustworthy and socially responsible machine learning systems. In response, numerous fair graph learning methods have been proposed in recent years. However, most of them assume full access to demographic information, a requirement rarely met in practice due to privacy, legal, or regulatory restrictions. To this end, this paper introduces a novel fair graph learning framework that mitigates bias in graph learning under limited demographic information. Specifically, we propose a mechanism guided by partial demographic data to generate proxies for demographic information and design a strategy that enforces consistent node embeddings across demographic groups. In addition, we develop an adaptive confidence strategy that dynamically adjusts each node's contribution to fairness and utility based on prediction confidence. We further provide theoretical analysis demonstrating that our framework, FairGLite, achieves provable upper bounds on group fairness metrics, offering formal guarantees for bias mitigation. Through extensive experiments on multiple datasets and fair graph learning frameworks, we demonstrate the framework's effectiveness in both mitigating bias and maintaining model utility.

---

