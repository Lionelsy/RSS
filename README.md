# 今日论文推荐 - 2025-10-02

共 22 篇论文

---

## 1. GenAI Models Capture Urban Science but Oversimplify Complexity

**论文链接:** [http://arxiv.org/abs/2505.13803v3](http://arxiv.org/abs/2505.13803v3)

**作者:** Yecheng Zhang, Rong Zhao, Zimu Huang, Xinyu Wang, Yue Ma, Ying Long

**发布时间:** 2025-05-20

**备注:** 31 pages, 16 figures

### GPT解析

### 总结

本研究引入AI4US框架，系统评估生成式人工智能模型在城市科学数据生成中的表现，发现模型能重现核心理论模式但存在数据多样性不足、参数偏差等问题，并提出后校准方法提高数据保真度。

### 背景

生成式人工智能模型越来越多地用于科学数据生成，但它们与城市科学中的经验知识的一致性仍不清楚。

### 目的

引入AI4US框架，系统评估领先的GenAI模型，测试它们在生成符号性和感知性城市数据时的保真度。

### 方法

对于符号领域，将生成的数据与基础城市理论进行比较基准测试；对于感知领域，验证模型的视觉判断与人类基准的一致性，并利用生成控制能力进行城市感知的因果实验。

### 主要发现

GenAI模型能够重现核心理论模式，但生成的数据存在多样性差、系统性参数偏差等问题，可通过提示工程改进。

### 结论

引入使用最优传输的后校准程序，生成具有更高保真度的合成符号数据集。

### 翻译

生成式人工智能(GenAI)模型越来越多地用于科学数据生成，但它们与城市科学中的经验知识的一致性仍不清楚。在此，我们引入AI4US(城市科学人工智能)框架，通过测试生成符号性和感知性城市数据时的保真度，系统评估领先的GenAI模型。对于符号领域，我们将生成的数据与关于规模、空间和形态的基础城市理论进行基准测试。对于感知领域，我们验证模型的视觉判断与人类基准的一致性，并关键性地利用它们的生成控制能力进行城市感知的因果实验。我们的发现表明，虽然GenAI模型能够重现核心理论模式，但生成的数据存在关键限制：多样性差、系统性参数偏差，以及需要通过提示工程改进。为解决这些问题，我们引入了一种使用最优传输的后校准程序，该程序生成的合成符号数据集具有明显更高的保真度。


### 论文摘要

Generative artificial intelligence (GenAI) models are increasingly used for scientific data generation, yet their alignment with empirical knowledge in urban science remains unclear. Here, we introduce AI4US (Artificial Intelligence for Urban Science), a framework that systematically evaluates leading GenAI models by testing their fidelity in generating both symbolic and perceptual urban data. For the symbolic domain, we benchmark generated data against foundational urban theories concerning scale, space, and morphology. For the perceptual domain, we validate the models' visual judgments against human benchmarks and, critically, leverage their generative control to conduct in causal experiments on urban perception. Our findings show that while GenAI models reproduce core theoretical patterns, the generated data exhibit crucial limitations: poor diversity, systematic parametric deviations, and improvement from prompt engineering. To address this, we introduce a post-hoc calibration procedure using optimal transport, which produces synthetic symbolic datasets with demonstrably higher fidelity.

---

## 2. TTT3R: 3D Reconstruction as Test-Time Training

**论文链接:** [http://arxiv.org/abs/2509.26645v1](http://arxiv.org/abs/2509.26645v1)

**作者:** Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen

**发布时间:** 2025-09-30

**备注:** Page: https://rover-xingyu.github.io/TTT3R Code:  https://github.com/Inception3D/TTT3R

### GPT解析

### 总结

本文提出了一种名为TTT3R的测试时训练方法，通过优化循环神经网络在3D重建中的内存更新机制，显著提高了长度泛化能力。

### 背景

现代循环神经网络由于线性时间复杂度已成为3D重建的竞争性架构，但其在超出训练上下文长度时性能显著下降，表现出有限的长度泛化能力。

### 目的

从测试时训练的角度重新审视3D重建基础模型，将其设计视为一个在线学习问题，以提高长度泛化能力。

### 方法

利用内存状态与传入观测之间的对齐置信度，推导出内存更新的闭式学习率，以平衡保留历史信息和适应新观测。

### 主要发现

TTT3R这种无需训练的干预方法显著提高了长度泛化能力，在全局姿态估计方面比基线方法提高了2倍，同时仅需6GB GPU内存即可处理数千张图像并以20FPS运行。

### 结论

TTT3R是一种有效的测试时训练方法，能够显著提升3D重建中的长度泛化能力，同时保持高效的计算性能。

### 翻译

现代循环神经网络由于其线性时间复杂度已成为3D重建的竞争性架构。然而，当应用于超出训练上下文长度的场景时，它们的性能显著下降，显示出有限的长度泛化能力。在这项工作中，我们从测试时训练的角度重新审视了3D重建基础模型，将其设计视为一个在线学习问题。基于这一观点，我们利用内存状态与传入观测之间的对齐置信度，推导出内存更新的闭式学习率，以平衡保留历史信息和适应新观测。这种无需训练的干预方法(称为TTT3R)显著提高了长度泛化能力，在全局姿态估计方面比基线方法提高了2倍，同时仅需6GB GPU内存处理数千张图像即可达到20FPS的运行速度。代码可在https://rover-xingyu.github.io/TTT3R获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决3D重建模型在处理长序列图像时的'长度泛化'问题。当输入图像数量超过训练时见过的序列长度时，现有模型性能会显著下降。这个问题在现实中非常重要，因为实际应用通常需要处理任意数量的图像，而全注意力方法虽然性能好但计算和内存成本随序列长度呈二次增长，RNN方法虽然内存效率高但难以泛化到长序列。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从测试时训练角度重新审视3D重建模型，将其视为在线学习问题。他们分析了RNN模型长度泛化不佳的原因，借鉴了现代RNN在语言任务中的表现，以及DeltaNet、TTT和Titans等将递归更新视为在线学习的工作。作者发现CUT3R可以解释为测试时训练机制，但简单地延长训练序列会导致效率低下，因此设计了基于置信度引导的闭式状态更新规则TTT3R。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将状态更新视为测试时训练过程，利用内存状态与传入观测之间的对齐置信度来推导内存更新的学习率，平衡保留历史信息和适应新观测。整体流程是：1)将图像转换为标记；2)用新信息更新前一个状态；3)从更新状态中检索信息输出标记；4)提取3D点图；5)关键创新是引入基于对齐置信度的每标记学习率；6)使用闭式状态更新规则增强CUT3R的长度泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)从测试时训练角度重新审视3D重建模型；2)利用对齐置信度推导学习率；3)提出简单有效的闭式状态更新规则TTT3R；4)实现训练免费干预，无需微调；5)高效处理长序列。相比全注意力方法，TTT3R计算和内存成本为线性而非二次；相比其他RNN方法，显著改善了长度泛化；相比显式内存方法，内存成本保持恒定且无需额外训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种名为TTT3R的测试时训练方法，通过置信度引导的状态更新规则，显著提高了3D重建模型处理长序列图像的能力，同时保持了高效的计算效率和内存使用。'}


### 论文摘要

Modern Recurrent Neural Networks have become a competitive architecture for 3D reconstruction due to their linear-time complexity. However, their performance degrades significantly when applied beyond the training context length, revealing limited length generalization. In this work, we revisit the 3D reconstruction foundation models from a Test-Time Training perspective, framing their designs as an online learning problem. Building on this perspective, we leverage the alignment confidence between the memory state and incoming observations to derive a closed-form learning rate for memory updates, to balance between retaining historical information and adapting to new observations. This training-free intervention, termed TTT3R, substantially improves length generalization, achieving a $2\times$ improvement in global pose estimation over baselines, while operating at 20 FPS with just 6 GB of GPU memory to process thousands of images. Code available in https://rover-xingyu.github.io/TTT3R

---

## 3. TAP: Two-Stage Adaptive Personalization of Multi-task and Multi-Modal Foundation Models in Federated Learning

**论文链接:** [http://arxiv.org/abs/2509.26524v1](http://arxiv.org/abs/2509.26524v1)

**作者:** Seohyun Lee, Wenzhi Fang, Dong-Jun Han, Seyyedali Hosseinalipour, Christopher G. Brinton

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种名为TAP的两阶段自适应个性化方法，用于解决联邦学习中个性化模型的问题，特别是在客户端数据、任务和模态都异构的环境中。

### 背景

联邦学习(FL)在去中心化训练中表现出色，但产生的最终模型不一定适合每个客户端的需求。虽然已有大量工作用于创建个性化模型(PFL)，但较少关注通过微调具有多任务和多模态特性的基础模型来实现个性化。

### 目的

解决文献中缺乏理解的问题：如何在客户端不仅在数据上，而且在任务和模态上异构的环境下微调和个性化具有多任务和多模态特性的基础模型。

### 方法

提出TAP（两阶段自适应个性化）方法：(i)利用客户端和服务器之间的不匹配模型架构，在有利于客户端本地任务时选择性地执行替换操作；(ii)进行联邦学习后的知识蒸馏，以捕获有益的通用知识而不损害个性化。同时引入服务器模型在其模态-任务对架构下的首次收敛分析。

### 主要发现

随着模态-任务对数量的增加，服务器模型满足所有任务的能力会下降。通过大量实验，证明了所提出算法在各种数据集和任务上的有效性。

### 结论

TAP方法在比较多种基线时表现出色，实现代码已公开可用。

### 翻译

联邦学习(FL)尽管在去中心化方式下训练多个模型表现出色，但已被证明产生的最终模型不一定适合每个客户端的需求。虽然已有大量工作用于创建定制的个性化模型(称为个性化联邦学习PFL)，但较少关注通过微调具有多任务和多模态特性的基础模型来实现个性化。此外，现有文献缺乏对如何在不仅在数据上，而且在任务和模态上异构的客户端环境中微调和个性化此类模型的理解。为解决这一文献空白，我们提出了TAP（两阶段自适应个性化），它(i)利用客户端和服务器之间的不匹配模型架构，在有利于客户端本地任务时选择性地执行替换操作，以及(ii)进行联邦学习后的知识蒸馏，以捕获有益的通用知识而不损害个性化。我们还引入了服务器模型在其模态-任务对架构下的首次收敛分析，并证明随着模态-任务对数量的增加，其满足所有任务的能力会下降。通过大量实验，我们证明了与多种基线相比，所提出算法在各种数据集和任务上的有效性。实现代码已在https://github.com/lee3296/TAP公开。


### 论文摘要

Federated Learning (FL), despite demonstrating impressive capabilities in the training of multiple models in a decentralized manner, has been shown to produce a final model not necessarily well-suited to the needs of each client. While extensive work has been conducted on how to create tailored personalized models, called Personalized Federated Learning (PFL), less attention has been given to personalization via fine-tuning of foundation models with multi-task and multi-modal properties. Moreover, there exists a lack of understanding in the literature on how to fine-tune and personalize such models in a setting that is heterogeneous across clients not only in data, but also in tasks and modalities. To address this gap in the literature, we propose TAP (Two-Stage Adaptive Personalization), which (i) leverages mismatched model architectures between the clients and server to selectively conduct replacement operations when it benefits a client's local tasks and (ii) engages in post-FL knowledge distillation for capturing beneficial general knowledge without compromising personalization. We also introduce the first convergence analysis of the server model under its modality-task pair architecture, and demonstrate that as the number of modality-task pairs increases, its ability to cater to all tasks suffers. Through extensive experiments, we demonstrate the effectiveness of our proposed algorithm across a variety of datasets and tasks in comparison to a multitude of baselines. Implementation code is publicly available at https://github.com/lee3296/TAP.

---

## 4. Commmunication-Efficient and Accurate Approach for Aggregation in Federated Low-Rank Adaptation

**论文链接:** [http://arxiv.org/abs/2509.26399v2](http://arxiv.org/abs/2509.26399v2)

**作者:** Le-Tuan Nguyen, Minh-Duong Nguyen, Seon-Geun Jeong, Dung D. Le, Quoc-Viet Pham

**发布时间:** 2025-09-30

**备注:** 34 pages, 4 figures, 11 tables

### GPT解析

### 总结

FLoRA-NA是一种新的联邦低秩适应方法，通过服务器上的本地LoRA矩阵估计聚合矩阵，解决了现有FedLoRA方法中不精确更新导致的局部-全局泛化差距和通信开销问题。

### 背景

基础模型快速发展，分布式环境下的微调需求增加，FedLoRA方法受到关注，但现有方法面临不精确更新带来的挑战。

### 目的

解决现有FedLoRA方法中因不精确更新导致的局部-全局泛化差距和大量通信开销问题，提高可扩展性和有效性。

### 方法

FLoRA-NA利用服务器上的本地LoRA矩阵估计聚合矩阵Â和B̂，分发给客户端进行本地更新，最小化理想更新与实际更新之间的差异，不增加额外通信成本。

### 主要发现

实验结果显示FLoRA-NA在各种任务（自然语言理解、数学推理、代码解决能力）上实现了最先进的全局性能，同时保持低通信开销。

### 结论

FLoRA-NA实现了通信效率，弥合了本地个性化与全局泛化之间的差距，解决了先前个性化FedLoRA方法的关键局限性。

### 翻译

随着基础模型的快速出现和分布式环境下微调需求的增加，联邦低秩适应(FedLoRA)最近获得了显著关注。尽管潜力巨大，但现有FedLoRA方法因不精确更新而面临显著挑战。现有方法试图缓解这一问题，但它们常常引入局部-全局泛化差距并产生大量通信开销，限制了其可扩展性和有效性。为解决这些局限性，我们提出了FLoRA-NA（具有几乎准确估计的联邦低秩聚合）。FLoRA-NA利用服务器上的本地LoRA矩阵来估计聚合矩阵Â和B̂，然后将它们分发给客户端进行本地更新。这种替代的聚合矩阵最小化了理想∇W̄ = ∑[u=1 to U] B_u A_u和实际更新∇Ŵ = B̂Â之间的差异，而不增加比普通FedLoRA更多的通信成本。通过这种方式，FLoRA-NA实现了通信效率，并弥合了本地个性化与全局泛化之间的差距，解决了先前个性化FedLoRA方法的一个关键局限性。我们在各种任务上进行了广泛评估，包括使用各种基础模型进行自然语言理解、数学推理和代码解决能力。实验结果一致表明，FLoRA-NA实现了最先进的全局性能，同时保持低通信开销。


### 论文摘要

With the rapid emergence of foundation models and the increasing need for fine-tuning across distributed environments, Federated Low-Rank Adaptation (FedLoRA) has recently gained significant attention. Despite enormous potential, current FedLoRA methods face notable challenges due to inexact updates. Existing approaches have attempted to mitigate this issue, but they often introduce a \emph{local-global generalization gap} and incur \emph{substantial communication overhead}, limiting their scalability and effectiveness. To address these limitations, we propose \textbf{F}ederated \textbf{Lo}w-\textbf{R}ank \textbf{A}ggregation with \textbf{N}early \textbf{A}ccurate Estimation (FLoRA-NA). FLoRA-NA leverages the local LoRA matrices on the server to estimate the aggregated matrices $\hat{A}$ and $\hat{B}$, which are then distributed to clients for local updates. This surrogated aggregated matrices minimizes the divergence between ideal $\nabla \Bar{W} = \sum^{U}_{u=1}B_u A_u$ and practical updates $\nabla \hat{W} = \hat{B}\hat{A}$ without adding communication cost beyond vanilla FedLoRA. By doing so, FLoRA-NA achieves communication efficiency and bridges the gap between local personalization and global generalization, addressing a key limitation of prior personalized FedLoRA approaches. We conduct extensive evaluations across diverse tasks, including natural language understanding, mathematical reasoning, and code-solving ability using various foundation models. Experimental results consistently demonstrate that FLoRA-NA achieves state-of-the-art global performance while maintaining low communication overhead.

---

## 5. Are neural scaling laws leading quantum chemistry astray?

**论文链接:** [http://arxiv.org/abs/2509.26397v1](http://arxiv.org/abs/2509.26397v1)

**作者:** Siwoo Lee, Adji Bousso Dieng

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究探讨了神经网络缩放定律在量子化学模型中的应用，评估了不同规模模型在预测H2分子键解离能量方面的表现，发现仅靠扩展模型规模和数据量不足以构建可靠的量子化学模型。

### 背景

神经网络缩放定律推动机器学习社区在各个领域训练越来越大的基础模型，以确保高精度和可迁移的表示用于外推任务。

### 目的

测试通过扩展模型容量和训练数据规模能否构建可靠的量子化学模型，特别是在预测分子键解离能量方面的表现。

### 方法

通过扩展量子化学计算中的模型容量和训练数据规模，评估模型对中性H2分子键解离能量的预测能力。研究人员比较了仅使用稳定结构训练的模型与包含压缩和拉伸几何结构训练的模型的预测效果。

### 主要发现

1. 无论数据集大小或模型容量如何，仅训练稳定结构的模型无法定性地重现H2能量曲线；2. 只有当压缩和拉伸的几何结构明确包含在训练中时，预测才能大致呈现正确的形状；3. 即使在最大、最多样化的数据集上训练的基础模型，在简单的双原子分子上也表现出严重失败；4. 这些模型无法重现两个裸质子的排斥能量曲线，表明它们未能学习电子结构理论中的基本库仑定律。

### 结论

仅靠扩展模型规模和数据量不足以构建可靠的量子化学模型，需要在训练数据中包含更广泛的分子构型，以确保模型能够学习基本的物理原理。

### 翻译

神经网络缩放定律正在推动机器学习社区在各个领域训练越来越大的基础模型，确保高精度和可迁移的表示用于外推任务。我们通过扩展量子化学计算中的模型容量和训练数据规模来测试这一承诺。作为泛化任务，我们评估了模型对中性H2(最简单的分子)键解离能量的预测。我们发现，无论数据集大小或模型容量如何，仅训练稳定结构的模型无法甚至定性地重现H2能量曲线。只有当压缩和拉伸的几何结构明确包含在训练中时，预测才能大致呈现正确的形状。然而，即使在包含解离双原子分子的最大、最多样化的数据集上训练的最大基础模型，在简单的双原子分子上也表现出严重失败。最显著的是，它们无法重现两个裸质子的平凡排斥能量曲线，这揭示了它们在学习电子结构理论中的基本库仑定律方面的失败。这些结果表明，仅靠扩展规模不足以构建可靠的量子化学模型。


### 论文摘要

Neural scaling laws are driving the machine learning community toward training ever-larger foundation models across domains, assuring high accuracy and transferable representations for extrapolative tasks. We test this promise in quantum chemistry by scaling model capacity and training data from quantum chemical calculations. As a generalization task, we evaluate the resulting models' predictions of the bond dissociation energy of neutral H$_2$, the simplest possible molecule. We find that, regardless of dataset size or model capacity, models trained only on stable structures fail dramatically to even qualitatively reproduce the H$_2$ energy curve. Only when compressed and stretched geometries are explicitly included in training do the predictions roughly resemble the correct shape. Nonetheless, the largest foundation models trained on the largest and most diverse datasets containing dissociating diatomics exhibit serious failures on simple diatomic molecules. Most strikingly, they cannot reproduce the trivial repulsive energy curve of two bare protons, revealing their failure to learn the basic Coulomb's law involved in electronic structure theory. These results suggest that scaling alone is insufficient for building reliable quantum chemical models.

---

## 6. How Far Do Time Series Foundation Models Paint the Landscape of Real-World Benchmarks ?

**论文链接:** [http://arxiv.org/abs/2509.26347v1](http://arxiv.org/abs/2509.26347v1)

**作者:** Lujun Li, Lama Sleem, Yiqun Wang, Yangjie Xu, Niccolò Gentile, Radu State

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究提出了一种新的基准测试方法，通过从真实视频中提取时间信号来连接合成和真实数据，并开发了REAL-V-TSFM数据集。实验表明，当前先进的时间序列基础模型在传统基准测试上表现良好，但在真实世界数据上泛化能力有限。

### 背景

当前对时间序列基础模型(TSFMs)的评估主要集中在合成基准测试上，真实世界泛化能力没有得到充分检验。

### 目的

提出一种新的基准测试方法，连接合成和真实数据；从真实视频中提取时间信号，反映日常时间动态；开发名为REAL-V-TSFM的新数据集。

### 方法

使用光流从真实视频中提取时间信号；构建反映日常时间动态的数据集；在三种最先进的TSFMs上进行零样本预测实验。

### 主要发现

尽管在传统基准测试上表现出色，但这些模型在所提出的数据集上主要表现出性能下降，表明这些基础模型的泛化能力有限。

### 结论

强调需要以数据为中心的基准测试和多样化的模型结构来推进TSFMs向真正的通用性发展，同时进一步验证了基于视频的时间序列数据提取管道的有效性。

### 翻译

最近对时间序列基础模型(TSFMs)的评估强调了合成基准测试，使得真实世界的泛化能力没有得到充分检验。这项工作提出了一种新的基准测试方法，通过使用光流从真实视频中提取时间信号并反映日常时间动态的数据集，连接了合成和真实数据。基于此流程，我们引入了REAL-V-TSFM，这是一个旨在捕获从真实视频中获得的丰富且多样化的时间序列的新数据集。在最先进的TSFMs上进行零样本预测的实验结果表明，尽管在传统基准测试上表现出色，但这些模型在所提出的数据集上主要表现出性能下降，表明这些基础模型的泛化能力有限。这些发现强调了以数据为中心的基准测试和多样化模型结构的迫切需要，以推动TSFMs向真正的通用性发展，同时进一步验证了我们基于视频的时间序列数据提取管道的有效性。


### 论文摘要

Recent evaluations of time-series foundation models (TSFMs) have emphasized synthetic benchmarks, leaving real-world generalization less thoroughly examined. This work proposes a novel benchmarking approach that bridges synthetic and realistic data by extracting temporal signals from real-world video using optical flow and curating datasets reflecting everyday temporal dynamics. Building upon this pipeline, we introduce REAL-V-TSFM, a novel dataset designed to capture rich and diverse time series derived from real-world videos. Experimental results on three state-of-the-art of TSFMs under zero-shot forecasting shows that, despite strong performance on conventional benchmarks, these models predominantly exhibit performance degradation on the proposed dataset, indicating limited generalizability in these foundation models. These findings highlight the urgent need for data-centric benchmarking and diverse model structure to advance TSFMs toward genuine universality, while further validating the effectiveness of our video-based time series data extraction pipeline.

---

## 7. NeuroTTT: Bridging Pretraining-Downstream Task Misalignment in EEG Foundation Models via Test-Time Training

**论文链接:** [http://arxiv.org/abs/2509.26301v1](http://arxiv.org/abs/2509.26301v1)

**作者:** Suli Wang, Yangshen Deng, Zhenghua Bao, Xinyu Zhan, Yiqun Duan

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究提出了一种两阶段对齐策略NeuroTTT，解决了EEG基础模型在预训练与下游任务间的对齐问题以及跨受试者分布偏移问题，显著提升了模型在多样化BCI任务中的性能和鲁棒性。

### 背景

大规模EEG信号基础模型为通用脑机接口应用提供了前景，但常面临预训练目标与下游任务不匹配及跨受试者分布偏移的挑战。

### 目的

通过两阶段对齐策略弥合通用预训练与特定EEG解码任务之间的差距，提高模型在多样化BCI任务中的准确性和鲁棒性。

### 方法

第一阶段提出NeuroTTT领域特定自监督微调范式，使用任务相关自监督目标对齐重要EEG特征；第二阶段在推理时纳入测试时训练，包括自监督测试时训练和预测熵最小化，持续校准模型到新输入。

### 主要发现

NeuroTTT首次统一了领域调优自监督与测试时训练在大规模EEG基础模型中的应用，在想象语音、压力检测和运动想象等BCI任务中显著提升了性能，超越了传统微调和适应方法。

### 结论

两阶段对齐策略有效解决了EEG基础模型面临的挑战，将CBraMod和LaBraM等骨干网络的性能提升至更高水平，实现了最先进的性能表现。

### 翻译

大规模EEG信号基础模型为通用脑机接口(BCI)应用提供了有前景的路径，但它们常常面临预训练目标和下游任务之间的不匹配问题，以及显著的跨受试者分布偏移。本文通过引入两阶段对齐策略解决了这些挑战，弥合了通用预训练和特定EEG解码任务之间的差距。首先，我们提出了NeuroTTT：一种领域特定的自监督微调范式，使用任务相关的自监督目标增强基础模型，将潜在表示对齐到重要的频谱、空间和时间EEG特征，而无需额外的标记数据。其次，我们在推理时纳入测试时训练(TTT)，对单个未标记的测试样本进行自监督测试时训练，并进行预测熵最小化(Tent)，仅更新归一化统计，以持续将模型校准到每个新输入。我们的方法，据我们所知，是第一个将领域调优自监督与测试时训练统一应用于大规模EEG基础模型的方法，在多样化的BCI任务(想象语音、压力检测、运动想象)中显著提高了鲁棒性和准确性。使用CBraMod和LaBraM作为骨干网络，我们的方法将它们的性能提升到更高水平。在三个不同任务上的结果表明，所提出的对齐策略实现了最先进的性能，优于传统的微调和适应方法。我们的代码可在https://github.com/wsl2000/NeuroTTT获取。


### 论文摘要

Large-scale foundation models for EEG signals offer a promising path to generalizable brain-computer interface (BCI) applications, but they often suffer from misalignment between pretraining objectives and downstream tasks, as well as significant cross-subject distribution shifts. This paper addresses these challenges by introducing a two-stage alignment strategy that bridges the gap between generic pretraining and specific EEG decoding tasks. First, we propose NeuroTTT: a domain-specific self-supervised fine-tuning paradigm that augments the foundation model with task-relevant self-supervised objectives, aligning latent representations to important spectral, spatial, and temporal EEG features without requiring additional labeled data. Second, we incorporate test-time training (TTT) at inference, we perform (i) self-supervised test-time training on individual unlabeled test samples and (ii) prediction entropy minimization (Tent), which updates only normalization statistics to continually calibrate the model to each new input on the fly. Our approach, which, to our knowledge, is the first to unify domain-tuned self-supervision with test-time training in large-scale EEG foundation models, yields substantially improved robustness and accuracy across diverse BCI tasks (imagined speech, stress detection, motor imagery). Using CBraMod and LaBraM as backbones, our method pushes their performance to a markedly higher level. Results on three diverse tasks demonstrate that the proposed alignment strategy achieves state-of-the-art performance, outperforming conventional fine-tuning and adaptation methods. Our code is available at https://github.com/wsl2000/NeuroTTT.

---

## 8. Accelerating Transformers in Online RL

**论文链接:** [http://arxiv.org/abs/2509.26137v1](http://arxiv.org/abs/2509.26137v1)

**作者:** Daniil Zelezetsky, Alexey K. Kovalev, Aleksandr I. Panov

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种基于Transformer的强化学习模型训练方法，通过两阶段训练策略解决了Transformer在模型无关在线强化学习中的不稳定性问题，显著提高了训练效率和稳定性。

### 背景

基于Transformer的模型在强化学习领域扩展了机器人任务的可能性，但在模型无关的在线RL实现中带来了挑战，现有学习算法因Transformer模型的不稳定性而难以实施。

### 目的

开发一种方法使Transformer能够在模型无关的在线强化学习中更稳定、更快地进行训练，克服其不稳定性问题。

### 方法

提出一种两阶段算法：第一阶段使用更简单稳定的Accelerator策略作为Transformer的教练，通过行为克隆训练Transformer；第二阶段让预训练好的Transformer以完全在线方式与环境交互。

### 主要发现

该算法加速了Transformer的性能，实现了更稳定快速的在线训练；在基于状态和图像的ManiSkill环境及MuJoCo任务上验证了效果；将基于图像环境的训练时间减少最多一半；将离线方法所需的回放缓冲区大小减少到1-2万，显著降低计算需求。

### 结论

所提出的算法成功解决了Transformer在强化学习中的稳定性问题，同时提高了训练效率，减少了计算资源需求，为基于Transformer的强化学习模型提供了实用解决方案。

### 翻译

基于Transformer的模型在强化学习中的出现扩展了机器人任务的可能性，但在其实现过程中，特别是在模型无关的在线强化学习中，同时带来了一系列挑战。由于Transformer模型的不稳定性，一些现有的学习算法难以与其实施。在本文中，我们提出了一种使用Accelerator策略作为Transformer训练器的方法。Accelerator是一个更简单、更稳定的模型，在算法的第一阶段独立与环境交互，同时通过行为克隆训练Transformer。在第二阶段，预训练好的Transformer开始以完全在线的方式与环境交互。因此，这种模型无关的算法在性能上加速了Transformer，并帮助它以更稳定、更快的方式进行在线训练。通过对基于状态和图像的ManiSkill环境以及在MDP和POMDP设置下的MuJoCo任务进行实验，我们表明应用我们的算法不仅能够实现Transformer的稳定训练，还将基于图像的环境的训练时间减少了最多一半。此外，它将离线方法所需的回放缓冲区大小减少到1-2万，显著降低了整体计算需求。


### 论文摘要

The appearance of transformer-based models in Reinforcement Learning (RL) has expanded the horizons of possibilities in robotics tasks, but it has simultaneously brought a wide range of challenges during its implementation, especially in model-free online RL. Some of the existing learning algorithms cannot be easily implemented with transformer-based models due to the instability of the latter. In this paper, we propose a method that uses the Accelerator policy as a transformer's trainer. The Accelerator, a simpler and more stable model, interacts with the environment independently while simultaneously training the transformer through behavior cloning during the first stage of the proposed algorithm. In the second stage, the pretrained transformer starts to interact with the environment in a fully online setting. As a result, this model-free algorithm accelerates the transformer in terms of its performance and helps it to train online in a more stable and faster way. By conducting experiments on both state-based and image-based ManiSkill environments, as well as on MuJoCo tasks in MDP and POMDP settings, we show that applying our algorithm not only enables stable training of transformers but also reduces training time on image-based environments by up to a factor of two. Moreover, it decreases the required replay buffer size in off-policy methods to 10-20 thousand, which significantly lowers the overall computational demands.

---

## 9. Agent-based code generation for the Gammapy framework

**论文链接:** [http://arxiv.org/abs/2509.26110v1](http://arxiv.org/abs/2509.26110v1)

**作者:** Dmitriy Kostunin, Vladimir Sotnikov, Sergo Golovachev, Abhay Mehta, Tim Lukas Holch, Elisa Jones

**发布时间:** 2025-09-30

**备注:** ICRC2025 proceedings PoS(ICRC2025)753

### GPT解析

### 总结

本研究针对大型语言模型在生成专业科学库代码方面的挑战，开发了一个能够在受控环境中编写、执行和验证代码的智能体，并以Gammapy库为例进行了实现，同时提供了网络演示和基准测试套件。

### 背景

大型语言模型在软件代码生成方面是现代人工智能最成功的应用之一。基础模型对于有文档、示例和强大社区支持的流行框架非常有效，但专业科学库通常缺乏这些资源，且可能暴露不稳定API，使模型难以处理。

### 目的

解决大型语言模型在处理缺乏文档和社区支持的专业科学库时的局限性，特别是针对Gammapy库，开发一个能够有效编写、执行和验证代码的智能体系统。

### 方法

开发一个能够在受控环境中编写、执行和验证代码的智能体系统，专门针对Gammapy库的特点和需求进行设计，并提供一个网络演示和基准测试套件来评估其性能。

### 主要发现

研究表明，通过专门的智能体系统，可以克服大型语言模型在处理缺乏文档的专业科学库时的局限性，有效生成、执行和验证针对Gammapy库的代码。

### 结论

这项工作为大型语言模型在专业科学库代码生成方面的应用提供了新的解决方案，通过开发专门的智能体系统，解决了数据有限或过时情况下模型训练的挑战，并展示了当前成果和未来发展方向。

### 翻译

使用大型语言模型生成软件代码是现代人工智能最成功的应用之一。基础模型对于有文档、示例和强大社区支持的流行框架非常有效。相比之下，专业科学库通常缺乏这些资源，并且可能暴露正在积极开发中的不稳定API，使得在有限或过时数据上训练的模型难以处理。我们通过开发一个能够在受控环境中编写、执行和验证代码的智能体来解决Gammapy库的这些问题。我们提出了一个最小的网络演示和配套的基准测试套件。这项工作总结了设计，报告了当前状态，并概述了下一步计划。


### 论文摘要

Software code generation using Large Language Models (LLMs) is one of the most successful applications of modern artificial intelligence. Foundational models are very effective for popular frameworks that benefit from documentation, examples, and strong community support. In contrast, specialized scientific libraries often lack these resources and may expose unstable APIs under active development, making it difficult for models trained on limited or outdated data. We address these issues for the Gammapy library by developing an agent capable of writing, executing, and validating code in a controlled environment. We present a minimal web demo and an accompanying benchmarking suite. This contribution summarizes the design, reports our current status, and outlines next steps.

---

## 10. GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data

**论文链接:** [http://arxiv.org/abs/2509.26016v1](http://arxiv.org/abs/2509.26016v1)

**作者:** Lubian Bai, Xiuyuan Zhang, Siqi Zhang, Zepeng Zhang, Haoyu Wang, Wei Qin, Shihong Du

**发布时间:** 2025-09-30

**备注:** NeurIPS 2025

### GPT解析

### 总结

本研究提出了GeoLink，一个多模态框架，利用OpenStreetMap数据增强遥感基础模型，在预训练和下游任务阶段实现多模态协同，提升地理空间智能性能。

### 背景

地理空间智能的进步需要整合地面地理空间数据与遥感模型，但遥感与OpenStreetMap数据之间存在模态差距（数据结构、内容和空间粒度差异），且大多数现有遥感基础模型仅关注图像数据。

### 目的

开发一个多模态框架，利用OSM数据增强RS基础模型，在预训练和下游任务阶段实现有效协同，提升模型对复杂地理场景的适应性。

### 方法

GeoLink框架使用OSM数据衍生的多粒度学习信号增强RS自监督预训练，通过跨模态空间相关性引导信息交互；引入图像掩码重建实现稀疏输入的高效预训练；为下游任务生成单模态和多模态细粒度编码，支持土地覆盖分类和城市功能区映射等多种应用。

### 主要发现

预训练过程中整合OSM数据可提高RS图像编码器性能；在下游任务中融合RS和OSM数据可增强模型对复杂地理场景的适应性；空间相关性在实现有效的多模态地理空间数据整合中起关键作用。

### 结论

多模态协同在推进高级地理空间人工智能方面具有显著潜力，代码、检查点和使用示例已在GitHub平台发布。

### 翻译

将地面地理空间数据（如OpenStreetMap）与丰富的地理上下文整合到遥感基础模型中，对推进地理空间智能和支持广泛任务至关重要。然而，遥感和OSM数据之间的模态差距（包括数据结构、内容和空间粒度的差异）使得有效协同极具挑战性，且大多数现有RS基础模型仅关注图像。为此，本研究提出了GeoLink，一个多模态框架，利用OSM数据在预训练和下游任务阶段增强RS基础模型。具体而言，GeoLink利用OSM数据衍生的多粒度学习信号增强RS自监督预训练，通过跨模态空间相关性引导信息交互和协作。它还引入图像掩码重建，以实现稀疏输入的高效预训练。对于下游任务，GeoLink生成单模态和多模态细粒度编码，支持从土地覆盖分类等常见RS解译任务到城市功能区映射等更全面的地理任务等多种应用。大量实验表明，在预训练过程中整合OSM数据可提高RS图像编码器的性能，而在下游任务中融合RS和OSM数据可提高模型对复杂地理场景的适应性。这些结果强调了多模态协同在推进高级地理空间人工智能方面的潜力。此外，我们发现空间相关性在实现有效的多模态地理空间数据整合中起着关键作用。代码、检查点和使用示例已在https://github.com/bailubin/GeoLink_NeurIPS2025发布。


### 论文摘要

Integrating ground-level geospatial data with rich geographic context, like OpenStreetMap (OSM), into remote sensing (RS) foundation models (FMs) is essential for advancing geospatial intelligence and supporting a broad spectrum of tasks. However, modality gap between RS and OSM data, including differences in data structure, content, and spatial granularity, makes effective synergy highly challenging, and most existing RS FMs focus on imagery alone. To this end, this study presents GeoLink, a multimodal framework that leverages OSM data to enhance RS FM during both the pretraining and downstream task stages. Specifically, GeoLink enhances RS self-supervised pretraining using multi-granularity learning signals derived from OSM data, guided by cross-modal spatial correlations for information interaction and collaboration. It also introduces image mask-reconstruction to enable sparse input for efficient pretraining. For downstream tasks, GeoLink generates both unimodal and multimodal fine-grained encodings to support a wide range of applications, from common RS interpretation tasks like land cover classification to more comprehensive geographic tasks like urban function zone mapping. Extensive experiments show that incorporating OSM data during pretraining enhances the performance of the RS image encoder, while fusing RS and OSM data in downstream tasks improves the FM's adaptability to complex geographic scenarios. These results underscore the potential of multimodal synergy in advancing high-level geospatial artificial intelligence. Moreover, we find that spatial correlation plays a crucial role in enabling effective multimodal geospatial data integration. Code, checkpoints, and using examples are released at https://github.com/bailubin/GeoLink_NeurIPS2025

---

## 11. Towards Reliable and Holistic Visual In-Context Learning Prompt Selection

**论文链接:** [http://arxiv.org/abs/2509.25989v1](http://arxiv.org/abs/2509.25989v1)

**作者:** Wenxiao Wu, Jing-Hao Xue, Chengming Xu, Chen Liu, Xinwei Sun, Changxin Gao, Nong Sang, Yanwei Fu

**发布时间:** 2025-09-30

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种改进的视觉上下文学习方法RH-Partial2Global，解决了现有方法在上下文示例选择中的假设不足和随机采样问题，通过基于jackknife conformal预测和覆盖设计的采样策略提升了性能。

### 背景

视觉上下文学习（VICL）已成为适应视觉基础模型到新任务的重要方法，它通过利用上下文示例中的上下文信息，可以表述为潜在候选者的全局排序问题。

### 目的

解决现有VICL方法（如Partial2Global和VPR）中相似度优先假设缺乏充分依据，以及Partial2Global依赖随机采样导致比较不完整和冗余的问题。

### 方法

提出RH-Partial2Global方法，采用jackknife conformal预测指导策略构建可靠的替代集合，以及基于覆盖设计的采样方法确保成对偏好的全面均匀覆盖。

### 主要发现

RH-Partial2Global在各种视觉任务上表现出色，显著优于Partial2Global方法。

### 结论

通过改进上下文示例选择策略，RH-Partial2Global为视觉上下文学习提供了更可靠和全面的方法。

### 翻译

视觉上下文学习（VICL）已成为一种突出的方法，通过有效利用上下文示例中嵌入的上下文信息，使视觉基础模型能够适应新任务，这可以表述为潜在候选者的全局排序问题。当前的VICL方法，如Partial2Global和VPR，基于相似度优先假设，即与查询图像视觉上更相似的图像作为更好的上下文示例。这一基础假设虽然直观，但缺乏对其选择最优上下文示例功效的充分依据。此外，Partial2Global从一系列随机采样的成对偏好预测中构建其全局排序。这种对随机采样的依赖可能导致比较的不完整覆盖和冗余采样，从而进一步损害最终的全局排序。为解决这些问题，本文引入了Partial2Global的增强变体，专为VICL中上下文示例的可靠和整体选择而设计。我们提出的方法RH-Partial2Global利用jackknife conformal预测指导策略构建可靠的替代集合，以及基于覆盖设计的采样方法确保成对偏好的全面和均匀覆盖。大量实验证明，RH-Partial2Global实现了卓越的性能，并在各种视觉任务上优于Partial2Global。


### 论文摘要

Visual In-Context Learning (VICL) has emerged as a prominent approach for adapting visual foundation models to novel tasks, by effectively exploiting contextual information embedded in in-context examples, which can be formulated as a global ranking problem of potential candidates. Current VICL methods, such as Partial2Global and VPR, are grounded in the similarity-priority assumption that images more visually similar to a query image serve as better in-context examples. This foundational assumption, while intuitive, lacks sufficient justification for its efficacy in selecting optimal in-context examples. Furthermore, Partial2Global constructs its global ranking from a series of randomly sampled pairwise preference predictions. Such a reliance on random sampling can lead to incomplete coverage and redundant samplings of comparisons, thus further adversely impacting the final global ranking. To address these issues, this paper introduces an enhanced variant of Partial2Global designed for reliable and holistic selection of in-context examples in VICL. Our proposed method, dubbed RH-Partial2Global, leverages a jackknife conformal prediction-guided strategy to construct reliable alternative sets and a covering design-based sampling approach to ensure comprehensive and uniform coverage of pairwise preferences. Extensive experiments demonstrate that RH-Partial2Global achieves excellent performance and outperforms Partial2Global across diverse visual tasks.

---

## 12. Bringing Emerging Architectures to Sequence Labeling in NLP

**论文链接:** [http://arxiv.org/abs/2509.25918v1](http://arxiv.org/abs/2509.25918v1)

**作者:** Ana Ezquerro, Carlos Gómez-Rodríguez, David Vilares

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究探讨了预训练Transformer编码器作为序列标注的主导方法，并评估了其他替代架构在不同复杂度标注任务上的适应能力，发现这些架构在简单环境中的优势不一定能跨语言泛化或处理复杂结构化任务。

### 背景

预训练Transformer编码器是序列标注的主导方法。虽然xLSTMs、结构化状态空间模型、扩散模型和对抗学习等替代架构在语言建模中显示出潜力，但很少被应用于序列标注，且大多在平面或简化任务上。

### 目的

研究这些替代架构如何适应不同结构复杂度、标签空间和标记依赖性的标注任务，评估范围涵盖多种语言。

### 方法

评估多种替代架构（xLSTMs、结构化状态空间模型、扩散模型和对抗学习）在不同复杂度标注任务上的表现，这些任务在结构复杂度、标签空间和标记依赖性方面各不相同，并在多种语言环境下进行测试。

### 主要发现

在简单环境中观察到的替代架构的强大性能并不总是能很好地跨语言或数据集泛化，也不适用于更复杂的结构化任务。

### 结论

替代架构在语言建模中表现出的优势不一定能直接转移到序列标注领域，特别是在跨语言泛化和处理复杂结构化任务方面。

### 翻译

预训练Transformer编码器是序列标注的主导方法。虽然一些替代架构——如xLSTMs、结构化状态空间模型、扩散模型和对抗学习——在语言建模中显示出潜力，但很少被应用于序列标注，而且大多是在平面或简化的任务上。我们研究了这些架构如何适应不同结构复杂度、标签空间和标记依赖性的标注任务，评估范围涵盖多种语言。我们发现，在简单环境中观察到的强大性能并不总是能很好地跨语言或数据集泛化，也不适用于更复杂的结构化任务。


### 论文摘要

Pretrained Transformer encoders are the dominant approach to sequence labeling. While some alternative architectures-such as xLSTMs, structured state-space models, diffusion models, and adversarial learning-have shown promise in language modeling, few have been applied to sequence labeling, and mostly on flat or simplified tasks. We study how these architectures adapt across tagging tasks that vary in structural complexity, label space, and token dependencies, with evaluation spanning multiple languages. We find that the strong performance previously observed in simpler settings does not always generalize well across languages or datasets, nor does it extend to more complex structured tasks.

---

## 13. PatchEAD: Unifying Industrial Visual Prompting Frameworks for Patch-Exclusive Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2509.25856v1](http://arxiv.org/abs/2509.25856v1)

**作者:** Po-Han Huang, Jeng-Lin Li, Po-Hsuan Huang, Ming-Ching Chang, Wei-Chao Chen

**发布时间:** 2025-09-30

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为PatchEAD的统一补丁专用的工业异常检测框架，实现了无需训练的异常检测，兼容多种基础模型，并在小样本和批量零样本性能上优于先前工作。

### 背景

工业异常检测越来越多地依赖基础模型，以实现强大的分布外泛化和实际部署中的快速适应。然而，过去研究主要关注文本提示调整，而视觉方面被分散成每个基础模型特定的处理步骤。

### 目的

解决视觉提示处理分散的问题，提出一个统一的补丁专用框架，实现无需训练的异常检测，并兼容多种基础模型。

### 方法

构建视觉提示技术，包括对齐模块和前景掩码，开发PatchEAD框架，实现训练-free的异常检测方法。

### 主要发现

尽管没有使用文本特征，PatchEAD在少样本和批量零样本性能上优于先前工作；研究还考察了骨干结构和预训练特性如何影响补丁相似鲁棒性，为选择和配置基础模型提供实用指导。

### 结论

一个统一的仅补丁框架可以实现快速、校准轻量级的部署，无需精心设计的文本提示。

### 翻译

工业异常检测越来越多地依赖基础模型，旨在实现强大的分布外泛化和实际部署中的快速适应。值得注意的是，过去研究主要关注文本提示调整，而视觉方面被分散成每个基础模型特定的处理步骤。我们旨在通过提出一个统一的补丁专用框架PatchEAD来解决这一限制，实现无需训练的异常检测，兼容多种基础模型。该框架构建了视觉提示技术，包括对齐模块和前景掩码。我们的实验表明，尽管没有使用文本特征，但在少样本和批量零样本性能上优于先前工作。我们的研究进一步考察了骨干结构和预训练特性如何影响补丁相似鲁棒性，为选择和配置基础模型用于实际视觉检查提供实用指导。这些结果证实，一个统一的仅补丁框架可以实现快速、校准轻量级的部署，无需精心设计的文本提示。


### 论文摘要

Industrial anomaly detection is increasingly relying on foundation models, aiming for strong out-of-distribution generalization and rapid adaptation in real-world deployments. Notably, past studies have primarily focused on textual prompt tuning, leaving the intrinsic visual counterpart fragmented into processing steps specific to each foundation model. We aim to address this limitation by proposing a unified patch-focused framework, Patch-Exclusive Anomaly Detection (PatchEAD), enabling training-free anomaly detection that is compatible with diverse foundation models. The framework constructs visual prompting techniques, including an alignment module and foreground masking. Our experiments show superior few-shot and batch zero-shot performance compared to prior work, despite the absence of textual features. Our study further examines how backbone structure and pretrained characteristics affect patch-similarity robustness, providing actionable guidance for selecting and configuring foundation models for real-world visual inspection. These results confirm that a well-unified patch-only framework can enable quick, calibration-light deployment without the need for carefully engineered textual prompts.

---

## 14. Kairos: Towards Adaptive and Generalizable Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.25826v1](http://arxiv.org/abs/2509.25826v1)

**作者:** Kun Feng, Shaocheng Lan, Yuchen Fang, Wenchao He, Lintao Ma, Xingyu Lu, Kan Ren

**发布时间:** 2025-09-30

### GPT解析

### 总结

Kairos是一种灵活的时间序列基础模型框架，通过动态分块标记器和实例自适应位置嵌入解决了时间序列异构信息密度的挑战，在零样本场景下表现出色。

### 背景

时间序列基础模型已成为时间序列分析的强大范式，通过大规模预训练驱动。然而，时间序列本质上具有异构信息密度，受系统状态和信号复杂度影响，在零样本场景下存在显著建模挑战。

### 目的

克服当前TSFMs依赖非自适应处理管道的局限性，解决固定大小分块和传统位置编码无法适应时间序列动态特性的问题。

### 方法

提出Kairos框架，集成动态分块标记器和实例自适应位置嵌入，自适应选择标记化粒度并定制位置编码。在包含超过3000亿个时间点的大规模PreSTS语料库上训练，采用多分块预测策略。

### 主要发现

Kairos在GIFT-Eval和Time-Series-Library两个零样本基准测试上实现卓越性能，使用更少的参数就能在各种任务中持续优于已建立的方法。

### 结论

Kairos能有效处理时间序列的异构信息密度问题，通过自适应处理提高了零样本场景下的性能，为时间序列分析提供了更灵活、更强大的框架。

### 翻译

时间序列基础模型已成为时间序列分析的强大范式，通过在多样化数据语料库上进行大规模预训练来驱动。然而，时间序列本质上具有异构信息密度，受系统状态和信号复杂度影响，在零样本场景下存在显著建模挑战。当前TSFMs依赖非自适应处理管道，无法捕捉时间序列的动态特性。例如，常见标记化策略如固定大小分块强制执行刚性的观测粒度，限制了它们适应不同信息密度的能力。同样，传统位置编码施加统一的时间尺度，难以建模不同序列间的多样周期性和趋势。为克服这些限制，我们提出Kairos，一个灵活的TSFM框架，集成了动态分块标记器和实例自适应位置嵌入。Kairos自适应选择标记化粒度，并根据每个时间序列实例的独特特性定制位置编码。在包含超过3000亿个时间点的大规模可预测性分层时间序列(PreSTS)语料库上训练，并在推理阶段采用多分块预测策略，Kairos在两个常见零样本基准测试上以更少的参数实现了卓越性能，在各种任务中持续优于已建立的方法。项目页面为https://foundation-model-research.github.io/Kairos。


### 论文摘要

Time series foundation models (TSFMs) have emerged as a powerful paradigm for time series analysis, driven by large-scale pretraining on diverse data corpora. However, time series inherently exhibit heterogeneous information density over time, influenced by system states and signal complexity, presenting significant modeling challenges especially in a zero-shot scenario. Current TSFMs rely on non-adaptive processing pipelines that fail to capture this dynamic nature. For example, common tokenization strategies such as fixed-size patching enforce rigid observational granularity, limiting their ability to adapt to varying information densities. Similarly, conventional positional encodings impose a uniform temporal scale, making it difficult to model diverse periodicities and trends across series. To overcome these limitations, we propose Kairos, a flexible TSFM framework that integrates a dynamic patching tokenizer and an instance-adaptive positional embedding. Kairos adaptively selects tokenization granularity and tailors positional encodings to the unique characteristics of each time series instance. Trained on a large-scale Predictability-Stratified Time Series (PreSTS) corpus comprising over 300 billion time points and adopting a multi-patch prediction strategy in the inference stage, Kairos achieves superior performance with much fewer parameters on two common zero-shot benchmarks, GIFT-Eval and the Time-Series-Library benchmark, consistently outperforming established methods across diverse tasks. The project page is at https://foundation-model-research.github.io/Kairos .

---

## 15. Adapting SAM with Dynamic Similarity Graphs for Few-Shot Parameter-Efficient Small Dense Object Detection: A Case Study of Chickpea Pods in Field Conditions

**论文链接:** [http://arxiv.org/abs/2509.25805v1](http://arxiv.org/abs/2509.25805v1)

**作者:** Xintong Jiang, Yixue Liu, Mohamed Debbagh, Yu Tian, Valerio Hoyos-Villegas, Viacheslav Adamchuk, Shangpeng Sun

**发布时间:** 2025-09-30

**备注:** 23 pages, 11 figures, 4 tables

### GPT解析

### 总结

本研究提出了一种动态相似性图自适应(DSGA)模块，结合低秩自适应(LoRA)方法，在极端数据约束条件下调整Segment Anything Model(SAM)，用于复杂农业环境中小密集对象的前景和实例分割。该方法仅需400万可训练参数(占原始SAM的4.26%)，在鹰嘴豆荚数据集上表现出色，结构度量提升17.31%，自适应F度量提升62.36%，并在豆荚计数任务中实现了0.8987的调整R平方值。

### 背景

基础模型在农业计算机视觉任务中的参数高效微调(PEFT)面临挑战，主要源于有限的训练数据和复杂的田间条件。

### 目的

开发一种方法，在极端数据约束条件下调整SAM模型，实现对复杂农业环境中小密集对象的前景和实例分割。

### 方法

引入动态相似性图自适应(DSGA)模块，通过动态相似性图构建、可学习的多项式衰减初始化权重排名机制和自适应局部特征聚合，建立强大的空间和动态相似性表示。将基于图的特征自适应与低秩自适应(LoRA)相结合，创建互补优化框架。

### 主要发现

在2、4、8和10个样本条件下，DSGA与LoRA结合在多个指标上实现优越性能，随样本数量增加性能提升。与基线SAM微调相比，结构度量改进17.31%，自适应F度量提升62.36%。Grad-CAM和t-SNE分析验证了框架在特征区分方面的有效性。在10到120个豆荚的图像上实现了准确计数，调整R平方值为0.8987。

### 结论

DSGA与LoRA结合的方法能在农业计算机视觉任务中实现高效的参数微调，在有限数据和复杂条件下表现优异，具有自动化农业监测应用的实用价值。

### 翻译

Parameter-Efficient Fine-Tuning (PEFT) of foundation models for agricultural computer vision tasks remains challenging due to limited training data and complex field conditions. This study introduces a Dynamic Similarity-based Graph Adaptation (DSGA) module to adapt the Segment Anything Model (SAM) under extreme data constraints for precise foreground and instance segmentation of small dense objects in complex agricultural environments. Through dynamic similarity graph construction with a learnable polynomial decay-initialized weight ranking mechanism and adaptive local feature aggregation, DSGA establishes robust spatial and dynamic similarity representation with only 4.00M trainable parameters, which is 4.26% of the original SAM. Integrating this graph-based feature adaptation with Low-Rank Adaptation (LoRA) creates a complementary optimization framework that effectively captures both local and global dependencies in image embeddings while preserving model stability and parameter efficiency. Experimental results on a challenging chickpea pod dataset demonstrated that DSGA with LoRA achieved superior performance across multiple metrics evaluated under 2, 4, 8 and 10 shots, with progressive performance gains as shot count increased. Quantitative metrics showed a 17.31% improvement in Structure-measure and a 62.36% gain in adaptive F-measure compared to the baseline SAM fine-tuning. Comprehensive ablation studies and visualization analyses through Grad-CAM and t-SNE validated the framework's effectiveness in feature discrimination. The proposed adaptation demonstrated practical utility for automated agricultural monitoring applications, achieving accurate pod-counting with an adjusted R-squared of 0.8987 for images with 10 to 120 pods under challenging field conditions.


### 论文摘要

Parameter-Efficient Fine-Tuning (PEFT) of foundation models for agricultural computer vision tasks remains challenging due to limited training data and complex field conditions. This study introduces a Dynamic Similarity-based Graph Adaptation (DSGA) module to adapt the Segment Anything Model (SAM) under extreme data constraints for precise foreground and instance segmentation of small dense objects in complex agricultural environments. Through dynamic similarity graph construction with a learnable polynomial decay-initialized weight ranking mechanism and adaptive local feature aggregation, DSGA establishes robust spatial and dynamic similarity representation with only 4.00M trainable parameters, which is 4.26% of the original SAM. Integrating this graph-based feature adaptation with Low-Rank Adaptation (LoRA) creates a complementary optimization framework that effectively captures both local and global dependencies in image embeddings while preserving model stability and parameter efficiency. Experimental results on a challenging chickpea pod dataset demonstrated that DSGA with LoRA achieved superior performance across multiple metrics evaluated under 2, 4, 8 and 10 shots, with progressive performance gains as shot count increased. Quantitative metrics showed a 17.31% improvement in Structure-measure and a 62.36% gain in adaptive F-measure compared to the baseline SAM fine-tuning. Comprehensive ablation studies and visualization analyses through Grad-CAM and t-SNE validated the framework's effectiveness in feature discrimination. The proposed adaptation demonstrated practical utility for automated agricultural monitoring applications, achieving accurate pod-counting with an adjusted R-squared of 0.8987 for images with 10 to 120 pods under challenging field conditions.

---

## 16. Dolphin v1.0 Technical Report

**论文链接:** [http://arxiv.org/abs/2509.25748v2](http://arxiv.org/abs/2509.25748v2)

**作者:** Taohan Weng, Chi zhang, Chaoran Yan, Siya Liu, Xiaoyang Liu, Yalun Wu, Boyang Wang, Boyan Wang, Jiren Ren, Kaiwen Yan, Jinze Yu, Kaibing Hu, Henan Liu, Haoyun Zheng, Zhenyu Liu, Duo Zhang, Xiaoqing Guo, Anjie Le, Hongcheng Guo

**发布时间:** 2025-09-30

### GPT解析

### 总结

研究人员开发了Dolphin v1.0和Dolphin R1，这是首个大规模多模态超声基础模型，通过三阶段训练策略和200万规模的多模态数据集，成功解决了超声成像中的操作者依赖、图像噪声等挑战，在多项超声任务上实现了突破性性能。

### 背景

超声在现代医学中至关重要，但面临操作者依赖、图像噪声和实时扫描等挑战，阻碍了AI的整合。大型多模态模型在其他医学影像领域表现出色，但在处理超声的复杂性方面存在困难。

### 目的

解决超声成像中的挑战，开发首个将多样化临床任务统一在单一视觉语言框架中的大规模多模态超声基础模型。

### 方法

整理200万规模的多模态数据集，结合教科书知识、公共数据、合成样本和通用语料库；采用三阶段训练策略：领域专用预训练、指令驱动对齐和基于强化学习的微调；通过超声特定的奖励强化学习增强模型性能。

### 主要发现

Dolphin v1.0在分类、检测、回归和报告生成方面提供可靠性能；Dolphin R1在U2-Bench上达到0.5835的U2分数，是第二最佳模型的两倍多；推理增强的训练显著提高了诊断准确性、一致性和可解释性。

### 结论

推理增强的训练对于高风险医疗AI非常重要，能够显著提高诊断准确性、一致性和可解释性，为超声成像的AI应用开辟了新途径。

### 翻译

超声在现代医学中至关重要，但面临操作者依赖、图像噪声和实时扫描等挑战，阻碍了人工智能的整合。虽然大型多模态模型在其他医学影像领域表现出色，但在处理超声的复杂性方面存在困难。为此，我们引入了Dolphin v1.0和其推理增强版本Dolphin R1——首个将多样化临床任务统一在单一视觉语言框架中的大规模多模态超声基础模型。为应对超声的变异性与噪声，我们整理了一个200万规模的多模态数据集，结合了教科书知识、公共数据、合成样本和通用语料库。这确保了鲁棒的感知、泛化和临床适应性。Dolphin系列采用三阶段训练策略：领域专用预训练、指令驱动对齐和基于强化学习的微调。Dolphin v1.0在分类、检测、回归和报告生成方面提供可靠性能。Dolphin R1通过超声特定的奖励强化学习，增强了诊断推理、推理透明度和可解释性。在U2-Bench上对八项超声任务的评估中，Dolphin R1达到0.5835的U2分数——超过第二最佳模型的两倍，创造了新的最先进水平。Dolphin v1.0也表现出竞争力，验证了统一框架的有效性。比较显示，推理增强的训练显著提高了诊断准确性、一致性和可解释性，突显了其对高风险医疗AI的重要性。


### 论文摘要

Ultrasound is crucial in modern medicine but faces challenges like operator dependence, image noise, and real-time scanning, hindering AI integration. While large multimodal models excel in other medical imaging areas, they struggle with ultrasound's complexities. To address this, we introduce Dolphin v1.0 (V1) and its reasoning-augmented version, Dolphin R1-the first large-scale multimodal ultrasound foundation models unifying diverse clinical tasks in a single vision-language framework.To tackle ultrasound variability and noise, we curated a 2-million-scale multimodal dataset, combining textbook knowledge, public data, synthetic samples, and general corpora. This ensures robust perception, generalization, and clinical adaptability.The Dolphin series employs a three-stage training strategy: domain-specialized pretraining, instruction-driven alignment, and reinforcement-based refinement. Dolphin v1.0 delivers reliable performance in classification, detection, regression, and report generation. Dolphin R1 enhances diagnostic inference, reasoning transparency, and interpretability through reinforcement learning with ultrasound-specific rewards.Evaluated on U2-Bench across eight ultrasound tasks, Dolphin R1 achieves a U2-score of 0.5835-over twice the second-best model (0.2968) setting a new state of the art. Dolphin v1.0 also performs competitively, validating the unified framework. Comparisons show reasoning-enhanced training significantly improves diagnostic accuracy, consistency, and interpretability, highlighting its importance for high-stakes medical AI.

---

## 17. Understanding Generative Recommendation with Semantic IDs from a Model-scaling View

**论文链接:** [http://arxiv.org/abs/2509.25522v1](http://arxiv.org/abs/2509.25522v1)

**作者:** Jingzhe Liu, Liam Collins, Jiliang Tang, Tong Zhao, Neil Shah, Clark Mingxuan Ju

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究探讨了生成推荐系统(GR)中的模型扩展问题，发现基于语义ID(SID)的GR方法在扩展时存在明显瓶颈，而直接使用大型语言模型(LLM)作为推荐器(LLM-as-RS)展现出更好的扩展特性和更高的性能。

### 背景

生成模型的最新进展为推荐系统带来了生成推荐(GR)这一新范式，它试图统一丰富的项目语义和协同过滤信号。基于语义ID(SID)的方法是其中一种流行实现，它通过量化模态编码器的嵌入来表示项目。

### 目的

研究基于SID的GR在模型扩展时的性能瓶颈，并寻找具有更好扩展行为的GR模型。

### 方法

重新审视直接使用大型语言模型(LLMs)作为推荐器(LLM-as-RS)的范式，并通过实验比较不同模型规模(从4400万到140亿参数)下基于SID的GR和LLM-as-RS的性能表现。

### 主要发现

基于SID的GR在扩大模型规模时性能迅速饱和；SID编码项目语义信息的能力有限是基本瓶颈；LLM-as-RS具有优越的扩展特性，性能比基于SID的GR提高高达20%；随着LLMs规模扩大，它们捕捉协同过滤信息的能力也随之提高。

### 结论

基于SID的GR存在内在的扩展限制，而LLM-as-RS被视为构建GR基础模型的一个有前景的发展方向。

### 翻译

生成模型的最新进展已允许推荐系统(RS)出现一个有前景的范式，称为生成推荐(GR)，它试图统一丰富的项目语义和协同过滤信号。一种流行的现代方法是使用语义ID(SIDs)，这些是从模态编码器(如大型语言或视觉模型)的嵌入量化的离散代码，在自回归用户交互序列建模设置中表示项目(称为基于SID的GR)。尽管其他领域的生成模型表现出明确的扩展规律，我们的研究揭示基于SID的GR在扩展模型时显示出明显的瓶颈。特别是，当我们扩大每个组件(模态编码器、量化标记器和RS本身)时，基于SID的GR的性能迅速饱和。在本工作中，我们确定SID编码项目语义信息的能力有限是基本瓶颈之一。受此观察启发，作为获得具有更好扩展行为的GR模型的初步努力，我们重新审视了另一种直接使用大型语言模型(LLMs)作为推荐器的GR范式(称为LLM-as-RS)。我们的实验表明，LLM-as-RS范式具有优越的模型扩展特性，并通过扩展实现了比基于SID的GR最佳可达性能高20%的改进。我们还挑战了LLMs难以捕捉协同过滤信息的普遍观点，表明随着LLMs规模扩大，它们建模用户项目交互的能力有所提高。我们对从4400万到140亿参数的不同模型规模的基于SID的GR和LLMs的分析强调了基于SID的GR的内在扩展限制，并将LLM-as-RS定位为GR基础模型的一个有前景的发展方向。


### 论文摘要

Recent advancements in generative models have allowed the emergence of a promising paradigm for recommender systems (RS), known as Generative Recommendation (GR), which tries to unify rich item semantics and collaborative filtering signals. One popular modern approach is to use semantic IDs (SIDs), which are discrete codes quantized from the embeddings of modality encoders (e.g., large language or vision models), to represent items in an autoregressive user interaction sequence modeling setup (henceforth, SID-based GR). While generative models in other domains exhibit well-established scaling laws, our work reveals that SID-based GR shows significant bottlenecks while scaling up the model. In particular, the performance of SID-based GR quickly saturates as we enlarge each component: the modality encoder, the quantization tokenizer, and the RS itself. In this work, we identify the limited capacity of SIDs to encode item semantic information as one of the fundamental bottlenecks. Motivated by this observation, as an initial effort to obtain GR models with better scaling behaviors, we revisit another GR paradigm that directly uses large language models (LLMs) as recommenders (henceforth, LLM-as-RS). Our experiments show that the LLM-as-RS paradigm has superior model scaling properties and achieves up to 20 percent improvement over the best achievable performance of SID-based GR through scaling. We also challenge the prevailing belief that LLMs struggle to capture collaborative filtering information, showing that their ability to model user-item interactions improves as LLMs scale up. Our analyses on both SID-based GR and LLMs across model sizes from 44M to 14B parameters underscore the intrinsic scaling limits of SID-based GR and position LLM-as-RS as a promising path toward foundation models for GR.

---

## 18. Can Molecular Foundation Models Know What They Don't Know? A Simple Remedy with Preference Optimization

**论文链接:** [http://arxiv.org/abs/2509.25509v1](http://arxiv.org/abs/2509.25509v1)

**作者:** Langzhou He, Junyou Zhu, Fangxin Wang, Junhua Liu, Haoyan Xu, Yue Zhao, Philip S. Yu, Qitian Wu

**发布时间:** 2025-09-29

### GPT解析

### 总结

Mole-PAIR是一种即插即用模块，可提高分子基础模型在分布外数据上的可靠性，通过偏好优化方法改善分布外检测能力。

### 背景

分子基础模型正在快速推进科学发现，但在分布外样本上的不可靠性限制了其在药物发现和蛋白质设计等高风险领域的应用。

### 目的

解决分子基础模型的化学幻觉问题，提高其在分布外数据上的可靠性。

### 方法

提出Molecular Preference-Aligned Instance Ranking (Mole-PAIR)模块，将分布外检测问题表述为对分布内和分布外样本之间估计的分布外亲和力的偏好优化，通过成对学习目标实现这一目标，优化AUROC指标。

### 主要发现

在五个真实世界分子数据集上的实验表明，该方法显著提高了现有分子基础模型的分布外检测能力，在大小、支架和测定分布偏移下，AUROC分别提高了45.8%、43.9%和24.3%。

### 结论

Mole-PAIR是一个简单、灵活的模块，可以通过低成本的后训练有效提高分子基础模型在分布外数据上的可靠性。

### 翻译

分子基础模型正在快速推进科学发现，但其在分布外样本上的不可靠性严重限制了其在药物发现和蛋白质设计等高风险领域的应用。一个关键的失败模式是化学幻觉，即模型对未知分子做出高置信度但完全错误的预测。为了应对这一挑战，我们引入了分子偏好对齐实例排序（Mole-PAIR），这是一个简单即插即用的模块，可以灵活集成到现有基础模型中，通过经济高效的后训练提高其在分布外数据上的可靠性。具体来说，我们的方法将分布外检测问题表述为对分布内和分布外样本之间估计的分布外亲和力的偏好优化，通过成对学习目标实现这一目标。我们表明，这个目标本质上优化了AUROC，该指标衡量模型如何一致地对分布内和分布外样本进行排序。在五个真实世界分子数据集上的广泛实验表明，我们的方法显著提高了现有分子基础模型的分布外检测能力，在大小、支架和测定分布偏移下，AUROC分别提高了45.8%、43.9%和24.3%。


### 论文摘要

Molecular foundation models are rapidly advancing scientific discovery, but their unreliability on out-of-distribution (OOD) samples severely limits their application in high-stakes domains such as drug discovery and protein design. A critical failure mode is chemical hallucination, where models make high-confidence yet entirely incorrect predictions for unknown molecules. To address this challenge, we introduce Molecular Preference-Aligned Instance Ranking (Mole-PAIR), a simple, plug-and-play module that can be flexibly integrated with existing foundation models to improve their reliability on OOD data through cost-effective post-training. Specifically, our method formulates the OOD detection problem as a preference optimization over the estimated OOD affinity between in-distribution (ID) and OOD samples, achieving this goal through a pairwise learning objective. We show that this objective essentially optimizes AUROC, which measures how consistently ID and OOD samples are ranked by the model. Extensive experiments across five real-world molecular datasets demonstrate that our approach significantly improves the OOD detection capabilities of existing molecular foundation models, achieving up to 45.8%, 43.9%, and 24.3% improvements in AUROC under distribution shifts of size, scaffold, and assay, respectively.

---

## 19. Guided Diffusion for the Discovery of New Superconductors

**论文链接:** [http://arxiv.org/abs/2509.25186v1](http://arxiv.org/abs/2509.25186v1)

**作者:** Pawan Prakash, Jason B. Gibson, Zhongwei Li, Gabriele Di Gianluca, Juan Esquivel, Eric Fuemmeler, Benjamin Geisler, Jung Soo Kim, Adrian Roitberg, Ellad B. Tadmor, Mingjie Liu, Stefano Martiniani, Gregory R. Stewart, James J. Hamlin, Peter J. Hirschfeld, Richard G. Hennig

**发布时间:** 2025-09-29

**备注:** 13 pages, 5 figures, 1 table

### GPT解析

### 总结

本研究提出了一个引导扩散框架来加速新型超导体的发现。通过预训练和微调DiffCSP基础模型，结合多阶段筛选过程，研究人员从20万个结构中筛选出773个具有高临界温度的候选超导体，并通过实验验证了计算结果。

### 背景

材料科学中具有特定期望性质（如高温超导性）的材料逆设计是一个巨大挑战，因为化学和结构空间非常广阔。

### 目的

加速新型超导体的发现。

### 方法

提出一个引导扩散框架；使用Alexandria Database预训练DiffCSP基础模型；在7,183个具有第一性原理衍生标签的超导体上进行微调；采用无分类器引导，采样200,000个结构；通过多阶段筛选过程结合机器学习和密度泛函理论(DFT)计算评估稳定性和电子特性。

### 主要发现

获得34,027个独特候选材料；识别出773个具有DFT计算临界温度大于5K的候选材料；生成模型展示了有效的属性驱动设计能力；计算结果通过实验合成和表征得到验证。

### 结论

这种端到端工作流程加速了超导体发现，同时强调了预测和合成实验可实现材料的挑战。

### 翻译

具有特定期望性质（如高温超导性）的材料逆设计在材料科学中代表着一项巨大挑战，这源于化学和结构空间的广阔性。我们提出了一个引导扩散框架来加速新型超导体的发现。DiffCSP基础模型在Alexandria Database上进行了预训练，并在7,183个具有第一性原理衍生标签的超导体上进行了微调。采用无分类器引导，我们采样了20万个结构，产生了34,027个独特候选材料。结合机器学习和密度泛函理论(DFT)计算评估稳定性和电子特性的多阶段筛选过程，确定了773个具有DFT计算临界温度大于5K的候选材料。值得注意的是，我们的生成模型展示了有效的属性驱动设计能力。我们的计算结果通过本工作中进行的实验合成和表征得到验证，这突显了在化学领域稀疏探索中的挑战。这种端到端工作流程加速了超导体发现，同时强调了预测和合成实验可实现材料的挑战。


### 论文摘要

The inverse design of materials with specific desired properties, such as high-temperature superconductivity, represents a formidable challenge in materials science due to the vastness of chemical and structural space. We present a guided diffusion framework to accelerate the discovery of novel superconductors. A DiffCSP foundation model is pretrained on the Alexandria Database and fine-tuned on 7,183 superconductors with first principles derived labels. Employing classifier-free guidance, we sample 200,000 structures, which lead to 34,027 unique candidates. A multistage screening process that combines machine learning and density functional theory (DFT) calculations to assess stability and electronic properties, identifies 773 candidates with DFT-calculated $T_\mathrm{c}>5$ K. Notably, our generative model demonstrates effective property-driven design. Our computational findings were validated against experimental synthesis and characterization performed as part of this work, which highlighted challenges in sparsely charted chemistries. This end-to-end workflow accelerates superconductor discovery while underscoring the challenge of predicting and synthesizing experimentally realizable materials.

---

## 20. PAN: Pillars-Attention-Based Network for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2509.15935v2](http://arxiv.org/abs/2509.15935v2)

**作者:** Ruan Bispo, Dane Mitrev, Letizia Mariotti, Clément Botty, Denver Humphrey, Anthony Scanlan, Ciarán Eising

**发布时间:** 2025-09-19

### GPT解析

### 总结

本文提出了一种新型高效的基于鸟瞰视图(BEV)的3D目标检测算法，结合摄像头和雷达数据，通过优化雷达特征处理和简化网络结构，实现了高性能与快速推理的平衡。

### 背景

摄像头-雷达融合是摄像头-激光雷达融合在恶劣天气和光照条件下实时3D目标检测的稳健低成本替代方案，但现有文献中很少有工作专注于开发能充分利用雷达点云优势（如精确距离估计和速度信息）的新架构。

### 目的

开发一种新型高效的3D目标检测算法，充分利用雷达点云的优势，提高检测性能同时减少推理时间。

### 方法

引入新型骨干网络将雷达柱面特征映射到嵌入维度；使用自注意力机制建模雷达点间依赖关系；用简化卷积层替代基于FPN的卷积层以减少推理时间；在特征融合前充分利用雷达优势。

### 主要发现

该方法在3D目标检测上达到新最先进水平；使用ResNet-50时NDS指标达58.2；在nuScenes数据集上为同类模型设置新的推理时间基准。

### 结论

通过新型骨干网络、自注意力机制和简化卷积层的结合，所提方法在保持高性能的同时显著减少推理时间，为摄像头-雷达融合的3D目标检测设立了新基准。

### 翻译

摄像头-雷达融合为摄像头-激光雷达融合提供了一种稳健且低成本的替代方案，用于在恶劣天气和光照条件下的实时3D目标检测任务。然而，目前文献中很少有工作关注这种模态，更重要的是，很少有工作开发新的架构来探索雷达点云的优势，如精确的距离估计和速度信息。因此，这项工作提出了一种新型且高效的3D目标检测算法，使用鸟瞰视图(BEV)中的摄像头和雷达。我们的算法在将特征融合到检测头之前利用了雷达的优势。引入了一种新的骨干网络，将雷达柱面特征映射到嵌入维度。自注意力机制使骨干网络能够建模雷达点之间的依赖关系。我们使用简化的卷积层替代基于PointPillars架构中基于FPN的卷积层，主要目标是减少推理时间。我们的结果表明，通过这种修改，我们的方法在3D目标检测问题上达到了新的最先进水平，使用ResNet-50时在NDS指标上达到58.2，同时在同一类别上为nuScenes数据集上的推理时间设置了新的基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D物体检测中相机-雷达融合方法在恶劣天气和光照条件下的稳健性和效率问题。这个问题在自动驾驶领域至关重要，因为现有的相机-激光雷达融合方案虽然精度高，但成本昂贵且在恶劣条件下性能不佳，而相机-雷达融合提供了一种更经济且在恶劣条件下更稳健的替代方案，但目前很少有工作充分探索雷达点云的优势。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到单一传感器在自动驾驶环境中的局限性，特别是恶劣天气条件下的性能问题。他们选择相机-雷达组合是因为雷达在恶劣条件下更稳健且成本低于激光雷达。作者借鉴了PointPillars作为基础架构，BEV表示方法，ResNet作为相机特征提取器，以及CenterPoint作为检测头。同时，他们引入了注意力机制等Transformer架构中的核心概念，并改进了雷达特征提取方法，设计出PAN网络。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在特征融合前充分利用雷达点云的优势（精确距离和速度信息），并通过自注意力机制建模雷达点之间的依赖关系，同时简化网络结构以提高推理速度。整体流程包括：1)使用ResNet和PAN分别提取相机和雷达特征；2)通过雷达辅助视图转换将相机特征转换为鸟瞰图；3)使用多模态特征聚合模块融合雷达和相机的鸟瞰图特征；4)最后使用CenterPoint检测头进行3D物体检测。PAN backbone特别处理雷达点云的稀疏性，通过支柱特征提取、自注意力机制和简化卷积来提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)PAN Backbone引入基于支柱的自注意力机制建模雷达点依赖关系；2)简化卷积结构减少推理时间；3)在特征融合前充分利用雷达特征；4)实现实时性能(约30 FPS)。相比之前的工作，PAN更充分地利用了雷达的距离和速度信息，在保持高精度的同时显著提高了推理速度，特别是在恶劣天气条件(雨、夜)下表现更佳，相比基线模型CRN推理时间减少约43%，在nuScenes数据集上达到58.2的NDS指标。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAN提出了一种基于支柱-注意力的相机-雷达融合网络，通过充分利用雷达的距离和速度信息以及自注意力机制，在保持实时性能的同时，显著提高了3D物体检测的准确性和恶劣天气条件下的稳健性。'}


### 论文摘要

Camera-radar fusion offers a robust and low-cost alternative to Camera-lidar fusion for the 3D object detection task in real-time under adverse weather and lighting conditions. However, currently, in the literature, it is possible to find few works focusing on this modality and, most importantly, developing new architectures to explore the advantages of the radar point cloud, such as accurate distance estimation and speed information. Therefore, this work presents a novel and efficient 3D object detection algorithm using cameras and radars in the bird's-eye-view (BEV). Our algorithm exploits the advantages of radar before fusing the features into a detection head. A new backbone is introduced, which maps the radar pillar features into an embedded dimension. A self-attention mechanism allows the backbone to model the dependencies between the radar points. We are using a simplified convolutional layer to replace the FPN-based convolutional layers used in the PointPillars-based architectures with the main goal of reducing inference time. Our results show that with this modification, our approach achieves the new state-of-the-art in the 3D object detection problem, reaching 58.2 of the NDS metric for the use of ResNet-50, while also setting a new benchmark for inference time on the nuScenes dataset for the same category.

---

## 21. Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies

**论文链接:** [http://arxiv.org/abs/2509.25822v2](http://arxiv.org/abs/2509.25822v2)

**作者:** Jing Wang, Weiting Peng, Jing Tang, Zeyu Gong, Xihua Wang, Bo Tao, Li Cheng

**发布时间:** 2025-09-30

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS  2025)

### GPT解析

### 总结

本文提出了一种名为Action-Guided Diffusion Policy (DP-AG)的统一表征学习方法，通过概率潜在动力学明确建模感知和行动之间的动态相互作用，解决了现有模仿学习方法中感知与行动解耦的问题。

### 背景

现有的模仿学习方法将感知和行动解耦，忽略了人类在自适应行为中自然利用的感官表征和行动执行之间的因果互惠关系。

### 目的

弥合感知和行动之间的差距，引入一种统一的表征学习方法，明确建模感知和行动之间的动态相互作用。

### 方法

提出了Action-Guided Diffusion Policy (DP-AG)，通过变分推理将潜在观测编码为高斯后验，使用行动引导的SDE演化这些潜在表示，引入循环一致的对比损失促进感知和行动之间的双向学习，并强制执行双向一致的转换。

### 主要发现

DP-AG在模拟基准测试和真实世界UR5操作任务中显著优于最先进的方法；推导了行动引导SDE的变分下界，证明对比目标增强了潜在和行动轨迹的连续性。

### 结论

DP-AG为弥合生物适应性和人工策略学习之间的差距提供了有希望的一步。

### 翻译

现有的模仿学习方法将感知和行动解耦，这忽视了人类在自适应行为中自然利用的感官表征和行动执行之间的因果互惠关系。为了弥合这一差距，我们引入了行动引导扩散策略(DP-AG)，这是一种统一的表征学习方法，通过概率潜在动力学明确建模感知和行动之间的动态相互作用。DP-AG通过变分推理将潜在观测编码为高斯后验，并使用行动引导的SDE演化它们，其中扩散策略噪声预测的向量-雅可比积(VJP)作为结构化随机力驱动潜在更新。为了促进感知和行动之间的双向学习，我们引入了一个循环一致的对比损失，该损失将噪声预测的梯度流组织成一个连贯的感知-行动循环，强制在潜在更新和行动细化中实现双向一致的转换。理论上，我们推导了行动引导SDE的变分下界，并证明对比目标增强了潜在和行动轨迹的连续性。实验上，DP-AG在模拟基准测试和真实世界UR5操作任务中显著优于最先进的方法。因此，我们的DP-AG为弥合生物适应性和人工策略学习提供了有希望的一步。


### 论文摘要

Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action-Guided Diffusion Policy (DP-AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP-AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle-consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception-action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP-AG significantly outperforms state-of-the-art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP-AG offers a promising step toward bridging biological adaptability and artificial policy learning.

---

## 22. Learning to Interact in World Latent for Team Coordination

**论文链接:** [http://arxiv.org/abs/2509.25550v2](http://arxiv.org/abs/2509.25550v2)

**作者:** Dongsu Lee, Daehee Lee, Yaru Niu, Honguk Woo, Amy Zhang, Ding Zhao

**发布时间:** 2025-09-29

### GPT解析

### 总结

这项研究提出了IWoL（交互世界潜在表示）框架，通过直接建模通信协议，构建可学习表示空间来促进多智能体强化学习中的团队协调，实现去中心化执行和隐式协调，同时避免显式消息传递的缺点。

### 背景

在多智能体强化学习中构建有效的团队协调表示具有挑战性，这源于多智能体交互产生的复杂动态以及局部观察导致的不完整信息。

### 目的

开发一种能够联合捕获智能体间关系和特定任务世界信息的表示学习框架，促进团队协调，同时避免显式消息传递的固有缺点。

### 方法

提出IWoL框架，构建可学习表示空间，通过直接建模通信协议来联合捕获智能体间关系和特定任务世界信息，该表示既可作为智能体的隐式潜在变量，也可作为通信的显式消息。

### 主要发现

在四个具有挑战性的MARL基准测试中，IWoL为团队协调提供了简单而强大的关键，且能与现有MARL算法结合进一步提升性能。

### 结论

IWoL框架通过隐式协调和去中心化执行，有效解决了多智能体系统中的团队协调问题，同时避免了显式消息传递的缺点，并能与现有算法结合提升性能。

### 翻译

这项工作提出了一种新颖的表示学习框架——交互世界潜在表示（IWoL），以促进多智能体强化学习（MARL）中的团队协调。由于多智能体交互产生的复杂动态和局部观察导致的不完整信息，为团队协调构建有效的表示是一个具有挑战性的问题。我们的关键见解是构建一个可学习的表示空间，通过直接建模通信协议，联合捕获智能体间关系和特定任务的世界信息。我们维持这种表示完全去中心化的执行和隐式协调，同时避免了显式消息传递的固有缺点，例如决策速度较慢、容易受到恶意攻击者的攻击以及对带宽限制的敏感性。实际上，我们的表示不仅可以作为每个智能体的隐式潜在变量，也可以作为通信的显式消息。在四个具有挑战性的MARL基准测试中，我们评估了两种变体，并证明IWoL为团队协调提供了一种简单而强大的关键。此外，我们展示了我们的表示可以与现有的MARL算法结合，以进一步提高它们的性能。


### 论文摘要

This work presents a novel representation learning framework, interactive world latent (IWoL), to facilitate team coordination in multi-agent reinforcement learning (MARL). Building effective representation for team coordination is a challenging problem, due to the intricate dynamics emerging from multi-agent interaction and incomplete information induced by local observations. Our key insight is to construct a learnable representation space that jointly captures inter-agent relations and task-specific world information by directly modeling communication protocols. This representation, we maintain fully decentralized execution with implicit coordination, all while avoiding the inherent drawbacks of explicit message passing, e.g., slower decision-making, vulnerability to malicious attackers, and sensitivity to bandwidth constraints. In practice, our representation can be used not only as an implicit latent for each agent, but also as an explicit message for communication. Across four challenging MARL benchmarks, we evaluate both variants and show that IWoL provides a simple yet powerful key for team coordination. Moreover, we demonstrate that our representation can be combined with existing MARL algorithms to further enhance their performance.

---

