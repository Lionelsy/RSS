# 今日论文推荐 - 2025-10-31

共 37 篇论文

---

## 1. UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling

**论文链接:** [http://arxiv.org/abs/2510.24106v2](http://arxiv.org/abs/2510.24106v2)

**作者:** Junhong Zou, Zhenxu Sun, Yueqing Wang, Wei Qiu, Zhaoxiang Zhang, Zhen Lei, Xiangyu Zhu

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文提出了一种名为UniField的方法，通过整合多个子领域的空气动力学数据进行联合训练，解决了数据稀缺性问题，实现了更通用的流场表示。

### 背景

表面压力场的空气动力学模拟对许多工程问题至关重要。深度神经网络已成为传统计算流体力学(CFD)模拟的高效替代方案，但数据稀缺性限制了神经网络的应用。

### 目的

整合多个子领域的空气动力学数据进行联合训练，以学习更通用的流场表示，解决数据稀缺性问题。

### 方法

整合五个涵盖汽车、火车、飞机和一般形状等不同领域的数据集。提出UniField方法，采用领域无关的Transformer模块提取通用点云特征，并定制领域特定的流条件适配器以适应不同子领域的流信息。

### 主要发现

尽管不同子领域的空气动力学数据遵循不同方程，但联合训练的模型比单独训练的模型表现更好，表明这些数据相互补充，帮助模型学习更好的流场表示。

### 结论

UniField作为通用流场表示模型具有潜力，为神经网络在空气动力学分析中的更广泛应用奠定了基础。

### 翻译

物体表面压力场的空气动力学模拟对许多工程问题至关重要。近年来，深度神经网络已成为传统计算成本高昂的CFD模拟的高效替代方案，用于建模表面压力场。然而，数据稀缺性仍然是一个基本挑战，限制了神经网络的应用。为了解决这一限制，我们提出整合多个子领域的空气动力学数据进行联合训练，以学习更通用的流场表示。我们整合了五个涵盖不同领域的数据集，包括汽车、火车、飞机和一般形状。面对不同领域间的显著数据差异，我们提出了UniField，它采用领域无关的Transformer模块提取通用点云特征，并定制领域特定的流条件适配器以适应不同子领域的流信息。尽管不同子领域的空气动力学数据通常遵循不同的方程，但我们比较了在所有数据上联合训练的模型与在单个数据集上分别训练的模型，发现联合训练的模型通常表现更好。这表明这些数据相互补充，帮助模型学习更好的流场表示。这些结果突显了UniField作为通用流场表示模型的潜力，为神经网络在空气动力学分析中的更广泛应用奠定了基础。


### 论文摘要

Aerodynamic simulation of the surface pressure field around objects is crucial for many engineering problems. In recent years, deep neural networks have emerged as an efficient alternative to traditional, computationally expensive CFD simulations for modeling surface pressure fields. However, data scarcity remains a fundamental challenge, limiting the application of neural networks. To address this limitation, we propose to integrate aerodynamic data from multiple subfields and conduct joint training to learn more general field representations. We consolidate five different datasets covering various fields, including automobiles, trains, aircraft, and general shapes. Facing significant data differences across different domains, we propose UniField, which employs a domain-agnostic Transformer module to extract general point cloud features and customizes domain-specific flow-conditioned adapters to adapt to the flow information in different subfields. Despite the fact that aerodynamic data from different subfields are typically governed by different equations, we compare models trained jointly on all data with those trained separately on individual datasets and find that the jointly-trained model commonly demonstrates better performance. This indicates that these data complement each other to help the model learn better flow field representations. These results highlight the potential of UniField as a universal flow field representation model and lay the foundation for broader applications of neural networks in aerodynamic analysis.

---

## 2. Linearized Optimal Transport for Analysis of High-Dimensional Point-Cloud and Single-Cell Data

**论文链接:** [http://arxiv.org/abs/2510.22033v2](http://arxiv.org/abs/2510.22033v2)

**作者:** Tianxiang Wang, Yingtong Ke, Dhananjay Bhaskar, Smita Krishnaswamy, Alexander Cloninger

**发布时间:** 2025-10-24

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

该研究提出了一种线性最优传输(LOT)框架，用于处理单细胞技术生成的高维点云数据，实现了预测准确性、可解释性和生成建模的统一。

### 背景

单细胞技术生成高维点云数据，能详细表征复杂患者状态和治疗反应。每个患者由不规则点云而非简单向量表示，难以直接量化和比较个体间生物学差异。现有非线性方法虽准确但如同黑箱，生物学可解释性差。

### 目的

开发一种既能保持预测准确性又具有生物学可解释性的方法，解决单细胞点云数据分析中的挑战。

### 方法

适配线性最优传输(LOT)框架，将不规则点云嵌入固定维度欧几里得空间，同时保留分布结构。这种嵌入提供有原则的线性表示，形成患者间的配准，支持直接比较细胞分布。

### 主要发现

LOT实现了：(i) COVID-19患者状态的准确且可解释的分类，分类器权重映射回特定标志物和空间区域；(ii) 患者来源类器官的合成数据生成。LOT形心产生平均细胞谱，支持药物相互作用测试。

### 结论

LOT作为统一框架连接了预测性能、可解释性和生成建模，通过将异质点云转换为结构化嵌入，为理解高维生物系统中的免疫变异和治疗效应开辟新机会。

### 翻译

单细胞技术生成细胞的高维点云，能够详细表征复杂的患者状态和治疗反应。然而每个患者由不规则点云而非简单向量表示，使得难以直接量化和比较个体间的生物学差异。诸如核方法和神经网络等非线性方法能达到预测准确性，但如同黑箱，提供很少的生物学可解释性。为解决这些局限性，我们将线性最优传输(LOT)框架适配到这一场景，将不规则点云嵌入到固定维度的欧几里得空间，同时保留分布结构。这种嵌入提供了有原则的线性表示，保留了最优传输几何，同时支持下游分析。它还形成了任意两个患者之间的配准，使其能够直接比较细胞分布。在此空间中，LOT实现了：(i) COVID-19患者状态的准确且可解释的分类，其中分类器权重映射回驱动预测的特定标志物和空间区域；(ii) 患者来源类器官的合成数据生成，利用LOT嵌入的线性性。LOT形心产生平均细胞谱，代表组合条件或样本，支持药物相互作用测试。这些结果共同确立了LOT作为连接预测性能、可解释性和生成建模的统一框架。通过将异质点云转换为可直接追溯到原始数据的结构化嵌入，LOT为理解高维生物系统中的免疫变异和治疗效应开辟了新机会。


### 论文摘要

Single-cell technologies generate high-dimensional point clouds of cells, enabling detailed characterization of complex patient states and treatment responses. Yet each patient is represented by an irregular point cloud rather than a simple vector, making it difficult to directly quantify and compare biological differences between individuals. Nonlinear methods such as kernels and neural networks achieve predictive accuracy but act as black boxes, offering little biological interpretability.   To address these limitations, we adapt the Linear Optimal Transport (LOT) framework to this setting, embedding irregular point clouds into a fixed-dimensional Euclidean space while preserving distributional structure. This embedding provides a principled linear representation that preserves optimal transport geometry while enabling downstream analysis. It also forms a registration between any two patients, enabling direct comparison of their cellular distributions. Within this space, LOT enables: (i) \textbf{accurate and interpretable classification} of COVID-19 patient states, where classifier weights map back to specific markers and spatial regions driving predictions; and (ii) \textbf{synthetic data generation} for patient-derived organoids, exploiting the linearity of the LOT embedding. LOT barycenters yield averaged cellular profiles representing combined conditions or samples, supporting drug interaction testing.   Together, these results establish LOT as a unified framework that bridges predictive performance, interpretability, and generative modeling. By transforming heterogeneous point clouds into structured embeddings directly traceable to the original data, LOT opens new opportunities for understanding immune variation and treatment effects in high-dimensional biological systems.

---

## 3. Clone Deterministic 3D Worlds with Geometrically-Regularized World Models

**论文链接:** [http://arxiv.org/abs/2510.26782v1](http://arxiv.org/abs/2510.26782v1)

**作者:** Zaishuo Xia, Yukuan Lu, Xinyi Li, Yifan Xu, Yubei Chen

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出了一种几何正则化世界模型(GRWM)，通过改进表示学习来提高世界模型的性能，特别是在长期预测任务中表现出更好的保真度和稳定性。

### 背景

世界模型是一种内部模型，用于模拟世界的发展，基于过去的观察和行动预测智能体及其环境的未来。准确的世界模型对智能体在复杂环境中有效思考、规划和推理至关重要。然而，当前世界模型在长期范围内表现脆弱且会退化。

### 目的

研究仅通过改进表示学习是否能显著提高世界模型的性能，并构建一个能够完全克隆并拟合确定性3D世界的模型。

### 方法

提出几何正则化世界模型(GRWM)，强制自然感觉轨迹上的连续点在潜在表示空间中保持接近，从而学习与环境真实拓扑紧密对齐的潜在表示。

### 主要发现

GRWM显著提高了确定性3D环境和长期预测任务中的滚动保真度和稳定性，其优势源于学习具有优越几何结构的潜在流形。GRWM是即插即用的，只需最小架构修改，可随轨迹长度扩展，且兼容各种潜在生成主干网络。

### 结论

改进表示学习是构建健壮世界模型的直接且有用的途径，无需扩大动态模块即可提供可靠的长期预测。

### 翻译

世界模型是一种内部模型，用于模拟世界的发展。基于过去的观察和行动，它预测智能体及其环境的未来。准确的世界模型对于智能体在复杂动态环境中有效思考、规划和推理至关重要。尽管进展迅速，但当前世界模型仍然脆弱，且在长期范围内会退化。我们认为，一个主要原因是表示质量：外部输入（如图像）是高维度的，且有损或纠缠的潜在表示使动态学习变得不必要地困难。因此，我们研究仅通过改进表示学习是否能显著提高世界模型的性能。在本文中，我们通过解决一个基本但尚未解决的问题，朝着构建真正准确的世界模型迈出了一步：构建一个能够完全克隆并拟合确定性3D世界的模型。我们提出了几何正则化世界模型(GRWM)，强制自然感觉轨迹上的连续点在潜在表示空间中保持接近。这种方法产生了显著改进的潜在表示，与环境真实拓扑紧密对齐。GRWM是即插即用的，只需要最小的架构修改，可随轨迹长度扩展，并且兼容各种潜在生成主干网络。在确定性3D环境和长期预测任务中，GRWM显著提高了滚动保真度和稳定性。分析表明，其优势源于学习具有优越几何结构的潜在流形。这些发现支持一个明确的结论：改进表示学习是构建健壮世界模型的直接且有用的途径，无需扩大动态模块即可提供可靠的长期预测。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是如何构建能够准确克隆确定性3D世界的世界模型，特别是解决长期预测中误差累积导致轨迹偏离现实的问题。这个问题在现实中非常重要，因为世界模型是强化学习、机器人规划和游戏内容生成等应用的核心工具，而当前世界模型在长期预测方面表现脆弱，无法满足需要精确模拟环境的实际应用需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过实验发现，当世界模型直接使用真实底层状态时预测效果极佳，但使用标准VAE的潜在空间时性能急剧下降，这使他们认识到表示质量是主要瓶颈。他们借鉴了对比学习(特别是时序对比学习)和几何正则化在3D物体表示学习中的应用，将其扩展到3D环境建模领域。具体设计上，他们提出了结合时序上下文架构和时序对比正则化的GRWM方法，确保潜在空间结构与环境的真实状态流形一致。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过改进表示学习来提升世界模型的性能，使潜在空间结构与环境的真实状态流形保持一致。整体实现流程包括：1)使用因果编码器处理连续观察序列生成潜在表示；2)结合重构损失、KL散度项、时序缓慢损失(确保连续状态在潜在空间中接近)和潜在均匀性损失(防止特征坍塌)进行训练；3)将训练好的潜在表示作为输入提供给各种动力学模型进行状态转换学习。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)重新定义问题，从开放世界生成转向确定性环境的精确复制；2)提出'表示质量优先'的观点，与当前侧重改进动力学模型的主流思路不同；3)引入几何正则化方法，确保潜在空间结构与环境的真实拓扑一致；4)设计插件式组件，可无缝集成到现有模型中；5)显著提高长期轨迹预测的稳定性和准确性。相比之前工作，GRWM更注重表示质量而非复杂动力学模型，结合了时序上下文和几何正则化，且完全无监督，不需要真实状态标签。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '通过几何正则化改进潜在表示学习，GRWM显著提高了确定性3D世界模型的长期预测准确性和稳定性，证明了表示质量对构建可靠世界模型的关键作用。'}


### 论文摘要

A world model is an internal model that simulates how the world evolves. Given past observations and actions, it predicts the future of both the embodied agent and its environment. Accurate world models are essential for enabling agents to think, plan, and reason effectively in complex, dynamic settings. Despite rapid progress, current world models remain brittle and degrade over long horizons. We argue that a central cause is representation quality: exteroceptive inputs (e.g., images) are high-dimensional, and lossy or entangled latents make dynamics learning unnecessarily hard. We therefore ask whether improving representation learning alone can substantially improve world-model performance. In this work, we take a step toward building a truly accurate world model by addressing a fundamental yet open problem: constructing a model that can fully clone and overfit to a deterministic 3D world. We propose Geometrically-Regularized World Models (GRWM), which enforces that consecutive points along a natural sensory trajectory remain close in latent representation space. This approach yields significantly improved latent representations that align closely with the true topology of the environment. GRWM is plug-and-play, requires only minimal architectural modification, scales with trajectory length, and is compatible with diverse latent generative backbones. Across deterministic 3D settings and long-horizon prediction tasks, GRWM significantly increases rollout fidelity and stability. Analyses show that its benefits stem from learning a latent manifold with superior geometric structure. These findings support a clear takeaway: improving representation learning is a direct and useful path to robust world models, delivering reliable long-horizon predictions without enlarging the dynamics module.

---

## 4. UniTok-Audio: A Unified Audio Generation Framework via Generative Modeling on Discrete Codec Tokens

**论文链接:** [http://arxiv.org/abs/2510.26372v1](http://arxiv.org/abs/2510.26372v1)

**作者:** Chengwei Liu, Haoyin Yan, Shaofei Xue, Xiaotao Liang, Yinghao Liu, Zheng Xue, Gang Song, Boyang Zhou

**发布时间:** 2025-10-30

**备注:** 21 pages, 3 figures

### GPT解析

### 总结

本文提出了UniTok-Audio框架，解决了音频生成模型在质量和泛化能力方面的挑战，实现了统一的音频生成任务处理。

### 背景

生成式建模在文本、图像和音频领域取得显著成功，但音频生成模型仍面临音频质量和跨任务泛化能力的挑战，导致开发冗余、性能不一致和扩展性有限。

### 目的

提出UniTok-Audio，一个可扩展且可扩展的统一音频生成任务框架，解决现有音频生成模型的碎片化问题。

### 方法

1) 提取条件的连续特征，以自回归方式生成目标音频的离散令牌；2) 使用特殊任务标识符令牌统一不同任务的学习模式；3) 开发包含声学和语义分支的双流音频编解码器实现高保真波形重构。

### 主要发现

UniTok-Audio在五个时间对齐任务（语音恢复、目标说话人提取、语音分离、语音转换和语言查询音频源分离）上与最先进的特定任务或多任务系统相比具有竞争力的性能。

### 结论

UniTok-Audio提供了一个统一的音频生成框架，解决了现有模型的碎片化问题，作者将开源代码库以促进未来研究。

### 翻译

生成式建模最近在文本、图像和音频领域取得了显著成功，展示了统一表征学习的强大能力。然而，音频生成模型在音频质量和跨任务泛化能力方面仍面临挑战。这种碎片化导致了冗余的开发工作、不一致的性能和有限的扩展性。为了解决这些问题，我们提出了UniTok-Audio，一个可扩展且可扩展的统一音频生成任务框架。具体而言，1) UniTok-Audio以自回归方式提取条件的连续特征，生成目标音频的离散令牌；2) 特殊的任务标识符令牌在单一框架中统一了多种任务的学习模式；3) 开发了包含声学和语义分支的双流音频编解码器，用于高保真波形重构。实验结果表明，UniTok-Audio在五个时间对齐任务（语音恢复、目标说话人提取、语音分离、语音转换和语言查询音频源分离）上与最先进的特定任务或多任务系统相比具有竞争力的性能。为了促进未来的研究，我们将开源我们的代码库。我们工作的演示页面可以在以下网址找到：https://alibaba.github.io/unified-audio。


### 论文摘要

Generative modeling has recently achieved remarkable success across text, image, and audio domains, demonstrating powerful capabilities for unified representation learning. However, audio generation models still face challenges in terms of audio quality and generalization ability across tasks. This fragmentation results in redundant development efforts, inconsistent performance, and limited extensibility. To address these issues, we propose \textbf{UniTok-Audio}, a scalable and extensible framework for unified audio generation tasks. Specifically, 1) UniTok-Audio extracts continuous feature of conditions to generates discrete tokens of target audio in an autoregressive manner; 2) a special task identifier token unifies different learning patterns of multiple tasks in a single framework; 3) a dual-stream audio codec involving acoustic and semantic branch is developed for high-fidelity waveform reconstruction. Experimental results demonstrate that UniTok-Audio achieves competitive performance in comparation with state-of-the-art task-specific or multi-task systems across five time-aligned tasks: speech restoration, target speaker extraction, speech separation, voice conversion, and language-queried audio source separation. To foster future research, we will open-source our codebase. The demo page of our work can be found here: https://alibaba.github.io/unified-audio.

---

## 5. Understanding Hardness of Vision-Language Compositionality from A Token-level Causal Lens

**论文链接:** [http://arxiv.org/abs/2510.26302v1](http://arxiv.org/abs/2510.26302v1)

**作者:** Ziliang Chen, Tianang Xiao, Jusheng Zhang, Yongsen Zheng, Xipeng Chen

**发布时间:** 2025-10-30

### GPT解析

### 总结

该研究针对CLIP模型在组合推理方面的局限性提出了一种基于标记的因果表示学习框架，揭示了CLIP在处理对象、属性和关系组合时的脆弱性根源，并为改进模型提供了理论指导。

### 背景

CLIP模型通过在共享嵌入空间中对齐图像和文本实现了强大的跨模态泛化能力，但在处理对象、属性和关系的组合推理方面持续失败，其行为类似于词袋匹配器。先前的因果解释通常将文本建模为单个向量，掩盖了标记级别的结构，无法解释提示敏感性和对困难样本失败等核心现象。

### 目的

解决现有CLIP模型在组合推理方面的局限性，提供对标记级别结构的更好理解，并解释模型在提示敏感性和困难样本处理上的失败原因。

### 方法

提出了一种基于标记的因果表示学习（CRL）框架，该框架基于顺序的语言标记结构因果模型（SCM）。将块可识别性理论扩展到标记化文本，证明CLIP的对比目标可以在句子级和标记级SCM下恢复模态不变潜在变量。

### 主要发现

揭示了CLIP的组合脆弱性源于'组合不可识别性'现象。存在伪最优文本编码器，它们可以实现完美的模态不变对齐，但对SWAP、REPLACE和ADD操作在原子概念上明显不敏感，因此无法区分正确的标题和困难样本。分析还表明语言侧的不可识别性与视觉侧的失败通过模态差距相联系，迭代组合运算会增加问题难度。

### 结论

标记级别的因果表示学习框架能够解释CLIP在组合推理方面的失败，这些发现为改进负样本挖掘策略和提高模型组合推理能力提供了理论依据。

### 翻译

对比语言-图像预训练（CLIP）通过在共享嵌入空间中对齐图像和文本，提供了强大的跨模态泛化能力，但在处理对象、属性和关系的组合推理方面持续失败，其行为常常类似于词袋匹配器。先前的因果解释通常将文本建模为单个向量，掩盖了标记级别的结构，无法解释提示敏感性和对困难样本失败等核心现象。我们通过基于标记的因果表示学习（CRL）框架解决了这一差距，该框架基于顺序的语言标记结构因果模型（SCM）。我们的理论将块可识别性扩展到标记化文本，证明CLIP的对比目标可以在句子级和标记级SCM下恢复模态不变的潜在变量。关键是，标记粒度为CLIP的组合脆弱性提供了第一个原则性解释：组合不可识别性。我们展示了伪最优文本编码器的存在，这些编码器可以实现完美的模态不变对齐，但对SWAP、REPLACE和ADD操作在原子概念上明显不敏感，因此尽管与真正最优编码器优化相同的训练目标，却无法区分正确的标题和困难样本。分析进一步将语言侧的不可识别性与视觉侧的失败通过模态差距联系起来，并展示了迭代组合运算如何增加难度，从而改进了负样本挖掘策略。


### 论文摘要

Contrastive Language-Image Pre-training (CLIP) delivers strong cross modal generalization by aligning images and texts in a shared embedding space, yet it persistently fails at compositional reasoning over objects, attributes, and relations often behaving like a bag-of-words matcher. Prior causal accounts typically model text as a single vector, obscuring token-level structure and leaving core phenomena-such as prompt sensitivity and failures on hard negatives unexplained. We address this gap with a token-aware causal representation learning (CRL) framework grounded in a sequential, language-token SCM. Our theory extends block identifiability to tokenized text, proving that CLIP's contrastive objective can recover the modal-invariant latent variable under both sentence-level and token-level SCMs. Crucially, token granularity yields the first principled explanation of CLIP's compositional brittleness: composition nonidentifiability. We show the existence of pseudo-optimal text encoders that achieve perfect modal-invariant alignment yet are provably insensitive to SWAP, REPLACE, and ADD operations over atomic concepts, thereby failing to distinguish correct captions from hard negatives despite optimizing the same training objective as true-optimal encoders. The analysis further links language-side nonidentifiability to visual-side failures via the modality gap and shows how iterated composition operators compound hardness, motivating improved negative mining strategies.

---

## 6. ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs

**论文链接:** [http://arxiv.org/abs/2510.26178v1](http://arxiv.org/abs/2510.26178v1)

**作者:** Yanran Tang, Ruihong Qiu, Xue Li, Zi Huang

**发布时间:** 2025-10-30

### GPT解析

### 总结

这篇论文提出了ReaKase-8B框架，通过整合法律事实、法律问题、法律关系三元组和法律推理，显著提高了法律案例检索的性能。

### 背景

法律案例检索(LCR)是现实世界法律决策的基石，现有方法主要依赖传统的词汇模型和预训练语言模型编码法律文本，但忽略了法律实体间的关系以及法律事实和法律问题如何导致司法决策的推理过程。

### 目的

将法律关系信息和关键推理过程整合到精确的案例嵌入中，以提高案例检索的准确性，并提出ReaKase-8B框架有效利用这些信息进行法律案例检索。

### 方法

ReaKase-8B设计了上下文法律案例表示学习范式，使用微调的大语言模型，整合提取的法律事实、法律问题、法律关系三元组和法律推理信息。

### 主要发现

在COLIEE 2022和COLIEE 2023两个基准数据集上的实验表明，知识和推理增强的嵌入显著提高了检索性能，超越了基线模型。

### 结论

集成法律推理到法律案例检索系统中具有巨大潜力，ReaKase-8B框架展示了这种整合的有效性，代码已在GitHub发布。

### 翻译

该摘要已按要求翻译为中文，提取了论文的核心要素，包括背景、目的、方法、发现和结论，并以JSON格式组织。


### 论文摘要

Legal case retrieval (LCR) is a cornerstone of real-world legal decision making, as it enables practitioners to identify precedents for a given query case. Existing approaches mainly rely on traditional lexical models and pretrained language models to encode the texts of legal cases. Yet there are rich information in the relations among different legal entities as well as the crucial reasoning process that uncovers how legal facts and legal issues can lead to judicial decisions. Such relational reasoning process reflects the distinctive characteristics of each case that can distinguish one from another, mirroring the real-world judicial process. Naturally, incorporating such information into the precise case embedding could further enhance the accuracy of case retrieval. In this paper, a novel ReaKase-8B framework is proposed to leverage extracted legal facts, legal issues, legal relation triplets and legal reasoning for effective legal case retrieval. ReaKase-8B designs an in-context legal case representation learning paradigm with a fine-tuned large language model. Extensive experiments on two benchmark datasets from COLIEE 2022 and COLIEE 2023 demonstrate that our knowledge and reasoning augmented embeddings substantially improve retrieval performance over baseline models, highlighting the potential of integrating legal reasoning into legal case retrieval systems. The code has been released on https://github.com/yanran-tang/ReaKase-8B.

---

## 7. Bridging the Gap Between Molecule and Textual Descriptions via Substructure-aware Alignment

**论文链接:** [http://arxiv.org/abs/2510.26157v1](http://arxiv.org/abs/2510.26157v1)

**作者:** Hyuntae Park, Yeachan Kim, SangKeun Lee

**发布时间:** 2025-10-30

**备注:** EMNLP 2025 (main)

### GPT解析

### 总结

本文介绍了MolBridge，一种基于亚结构感知对齐的新型分子-文本学习框架，通过增强分子亚结构和化学短语之间的细粒度对齐，有效提升了分子表示学习的性能。

### 背景

分子和文本表示学习越来越受到关注，因为它有潜力增强对化学信息的理解。然而，现有模型往往难以捕捉分子及其描述之间的细微差异。

### 目的

解决现有模型缺乏学习分子亚结构和化学短语之间细粒度对齐能力的问题，提高分子表示学习的准确性。

### 方法

通过从分子亚结构和化学短语中衍生的额外对齐信号来增强原始分子-描述对，采用亚结构感知对比学习，并结合自我完善机制来过滤有噪声的对齐信号。

### 主要发现

实验结果表明，MolBridge能够有效捕获细粒度对应关系，并在广泛的分子基准测试中优于最先进的基线模型。

### 结论

亚结构感知对齐在分子-文本学习中具有重要意义，MolBridge框架为分子表示学习提供了有效解决方案。

### 翻译

分子和文本表示学习因其增强化学信息理解的潜力而日益受到关注。然而，现有模型往往难以捕捉分子及其描述之间的细微差异，因为它们缺乏学习分子亚结构和化学短语之间细粒度对齐的能力。为解决这一局限性，我们引入了MolBridge，一种基于亚结构感知对齐的新型分子-文本学习框架。具体而言，我们通过从分子亚结构和化学短语中衍生的额外对齐信号来增强原始分子-描述对。为了有效学习这些丰富的对齐信息，MolBridge采用亚结构感知对比学习，并结合一种自我完善机制来过滤有噪声的对齐信号。实验结果表明，MolBridge能够有效捕获细粒度对应关系，并在广泛的分子基准测试中优于最先进的基线模型，突显了亚结构感知对齐在分子-文本学习中的重要性。


### 论文摘要

Molecule and text representation learning has gained increasing interest due to its potential for enhancing the understanding of chemical information. However, existing models often struggle to capture subtle differences between molecules and their descriptions, as they lack the ability to learn fine-grained alignments between molecular substructures and chemical phrases. To address this limitation, we introduce MolBridge, a novel molecule-text learning framework based on substructure-aware alignments. Specifically, we augment the original molecule-description pairs with additional alignment signals derived from molecular substructures and chemical phrases. To effectively learn from these enriched alignments, MolBridge employs substructure-aware contrastive learning, coupled with a self-refinement mechanism that filters out noisy alignment signals. Experimental results show that MolBridge effectively captures fine-grained correspondences and outperforms state-of-the-art baselines on a wide range of molecular benchmarks, highlighting the significance of substructure-aware alignment in molecule-text learning.

---

## 8. Learning Geometry: A Framework for Building Adaptive Manifold Models through Metric Optimization

**论文链接:** [http://arxiv.org/abs/2510.26068v1](http://arxiv.org/abs/2510.26068v1)

**作者:** Di Zhang

**发布时间:** 2025-10-30

**备注:** 9 pages

### GPT解析

### 总结

这篇论文提出了一种新颖的机器学习范式，超越了传统的参数优化方法。它将模型本身视为可变形的几何实体，通过优化流形上的度量张量场来动态塑造模型空间的几何结构。

### 背景

传统的机器学习方法在固定的几何空间内搜索最优参数，而本文提出了一种不同的思路。

### 目的

提出一种新的机器学习范式，通过优化度量张量场来动态调整模型的几何结构，从而提高模型的表示能力和泛化能力。

### 方法

构建了一个变分框架，其损失函数平衡了数据保真度和流形的内在几何复杂度。为解决这个无限维优化问题的计算挑战，引入了一种基于离散微分几何的实用方法，将连续流形离散化为三角形网格，并通过边长参数化度量张量。

### 主要发现

理论分析揭示了该框架与广义相对论中的爱因斯坦-希尔伯特作用之间的深刻类比，为'数据驱动几何'概念提供了优雅的物理解释。即使拓扑结构固定，度量优化也比固定几何的模型具有更强的表达能力。

### 结论

这项工作为构建能够自主进化其几何和拓扑结构的完全动态'元学习器'奠定了坚实基础，并在科学模型发现和鲁棒表示学习等领域具有广阔的应用前景。

### 翻译

本文提出了一种超越传统参数优化的机器学习新范式。与在固定几何空间内搜索最优参数的传统方法不同，我们的核心思想是将模型本身视为可变形的几何实体。具体而言，我们在具有预定义拓扑的流形上优化度量张量场，从而动态塑造模型空间的几何结构。为此，我们构建了一个变分框架，其损失函数仔细平衡了数据保真度与流形的内在几何复杂性。前者确保模型能有效解释观测数据，而后者则作为正则化项，对过度弯曲或不规则的几何结构进行惩罚，以鼓励更简单的模型并防止过拟合。为解决这个无限维优化问题的计算挑战，我们引入了一种基于离散微分几何的实用方法：将连续流形离散化为三角形网格，并通过边长参数化度量张量，从而能够使用自动微分工具进行高效优化。理论分析揭示了我们的框架与广义相对论中爱因斯坦-希尔伯特作用之间的深刻类比，为'数据驱动几何'概念提供了优雅的物理解释。我们进一步认为，即使拓扑结构固定，度量优化也比具有固定几何的模型具有更强的表达能力。这项工作为构建能够自主进化其几何和拓扑结构的完全动态'元学习器'奠定了坚实基础，并指向了科学模型发现和鲁棒表示学习等领域的广泛应用前景。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决传统机器学习方法的局限性：模型在固定的几何空间中寻找最优参数，而非优化模型空间本身的几何结构。当数据的本质几何结构是非欧几里得、弯曲或具有复杂拓扑时，传统方法可能导致效率低下、泛化能力弱或难以解释。这个问题的重要性在于，它限制了模型对数据内在结构的捕捉能力，阻碍了机器学习在处理复杂数据结构时的表现，同时也限制了我们对学习过程本质的理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的思考始于对传统机器学习局限性的反思，提出不仅应优化参数点，还应优化塑造空间的几何结构本身。他借鉴了信息几何的启发，将参数化的概率分布视为微分流形，但超越了传统信息几何中静态度量的观念。作者设计了在固定拓扑流形上优化度量张量场的理论框架，构建了平衡数据保真度和几何复杂性的变分框架，并引入基于离散微分几何的实用计算方法。这项工作借鉴了信息几何、几何深度学习、生成模型和离散微分几何等领域的研究成果，但提出了全新的整合视角。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将机器学习模型视为可塑的几何实体，其学习过程表现为流形自身度量结构的自我优化，通过优化流形上的度量张量场来动态塑造模型空间的几何结构。实现流程包括：1)问题形式化，给定固定拓扑流形和观测数据集；2)设计变分框架，包含数据保真度项和几何复杂性项；3)将连续流形离散化为三角形网格，通过边长参数化度量张量；4)使用基于自动微分的优化算法进行迭代优化，通过投影梯度更新确保满足三角形不等式约束。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)从参数学习到结构学习的范式转变；2)将流形上的度量张量场本身作为优化目标；3)构建平衡数据保真度和几何复杂性的变分框架；4)基于离散微分几何的实用计算方法；5)揭示框架与广义相对论中爱因斯坦-希尔伯特作用的深刻联系。相比之前的工作，本文超越了传统信息几何中静态度量的观念，不同于几何深度学习中的固定几何假设，区别于传统生成模型中预设的简单潜在空间，也不同于传统非线性降维中固定的描述性流形，使几何结构成为学习过程的核心部分。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "这篇论文提出了一种机器学习新范式，通过优化流形上的度量张量场使模型能够动态塑造自身的几何结构，为构建完全自适应的'元学习器'奠定了理论基础。"}


### 论文摘要

This paper proposes a novel paradigm for machine learning that moves beyond traditional parameter optimization. Unlike conventional approaches that search for optimal parameters within a fixed geometric space, our core idea is to treat the model itself as a malleable geometric entity. Specifically, we optimize the metric tensor field on a manifold with a predefined topology, thereby dynamically shaping the geometric structure of the model space. To achieve this, we construct a variational framework whose loss function carefully balances data fidelity against the intrinsic geometric complexity of the manifold. The former ensures the model effectively explains observed data, while the latter acts as a regularizer, penalizing overly curved or irregular geometries to encourage simpler models and prevent overfitting. To address the computational challenges of this infinite-dimensional optimization problem, we introduce a practical method based on discrete differential geometry: the continuous manifold is discretized into a triangular mesh, and the metric tensor is parameterized by edge lengths, enabling efficient optimization using automatic differentiation tools. Theoretical analysis reveals a profound analogy between our framework and the Einstein-Hilbert action in general relativity, providing an elegant physical interpretation for the concept of "data-driven geometry". We further argue that even with fixed topology, metric optimization offers significantly greater expressive power than models with fixed geometry. This work lays a solid foundation for constructing fully dynamic "meta-learners" capable of autonomously evolving their geometry and topology, and it points to broad application prospects in areas such as scientific model discovery and robust representation learning.

---

## 9. Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis

**论文链接:** [http://arxiv.org/abs/2510.26014v1](http://arxiv.org/abs/2510.26014v1)

**作者:** Hyeonjun Lee, Hyungseob Shin, Gunhee Nam, Hyeonsoo Lee

**发布时间:** 2025-10-29

**备注:** Accepted to NeurIPS 2025 workshop Learning from Time Series for  Health (TS4H)

### GPT解析

### 总结

本文提出了一种用于离散时间生存分析的双重专家混合框架，有效解决了患者异质性建模和风险预测适应性的挑战，在乳腺癌数据集上表现出色。

### 背景

生存分析是建模直到感兴趣事件发生的时间的任务，广泛应用于临床和生物医学研究。主要挑战是如何建模患者异质性，同时使风险预测适应个体特征和时间动态性。

### 目的

开发一个双重专家混合框架，用于离散时间生存分析，以有效处理患者异质性和时间动态性。

### 方法

提出双重专家混合框架，结合特征编码器专家混合用于亚组感知的表示学习，以及风险专家混合利用患者特征和时间嵌入来捕获时间动态性。该设计可灵活集成到现有的基于深度学习的生存分析管道中。

### 主要发现

在METABRIC和GBSG乳腺癌数据集上，该方法持续提高了性能，将时间依赖的C-index提升了最多0.04（在测试集上），当整合到Consurv框架中时获得了进一步的提升。

### 结论

双重MoE框架能够有效处理患者异质性并适应时间动态性，在乳腺癌数据集上表现优异，且可以与现有框架结合使用。

### 翻译

生存分析是一项建模直到感兴趣事件发生的时间的任务，广泛应用于临床和生物医学研究。一个关键挑战是建模患者异质性，同时使风险预测适应个体特征和时间动态性。我们提出了一个用于离散时间生存分析的双重专家混合框架。我们的方法结合了一个特征编码器专家混合，用于亚组感知的表示学习，以及一个风险专家混合，利用患者特征和时间嵌入来捕获时间动态性。这种双重MoE设计可以灵活地与现有的基于深度学习的生存分析管道集成。在METABRIC和GBSG乳腺癌数据集上，我们的方法持续提高了性能，将时间依赖的C-index提升了最多0.04（在测试集上），当整合到Consurv框架中时获得了进一步的提升。


### 论文摘要

Survival analysis is a task to model the time until an event of interest occurs, widely used in clinical and biomedical research. A key challenge is to model patient heterogeneity while also adapting risk predictions to both individual characteristics and temporal dynamics. We propose a dual mixture-of-experts (MoE) framework for discrete-time survival analysis. Our approach combines a feature-encoder MoE for subgroup-aware representation learning with a hazard MoE that leverages patient features and time embeddings to capture temporal dynamics. This dual-MoE design flexibly integrates with existing deep learning based survival pipelines. On METABRIC and GBSG breast cancer datasets, our method consistently improves performance, boosting the time-dependent C-index up to 0.04 on the test sets, and yields further gains when incorporated into the Consurv framework.

---

## 10. Contrastive Predictive Coding Done Right for Mutual Information Estimation

**论文链接:** [http://arxiv.org/abs/2510.25983v1](http://arxiv.org/abs/2510.25983v1)

**作者:** J. Jon Ryu, Pavan Yeddanapudi, Xiangxiang Xu, Gregory W. Wornell

**发布时间:** 2025-10-29

**备注:** 26 pages, 5 figures

### GPT解析

### 总结

本文指出InfoNCE作为互信息估计器的局限性，并提出改进方法InfoNCE-anchor，通过引入辅助锚点类实现更准确的互信息估计。研究还揭示了对比表示学习的真正价值在于学习结构化密度比，而非准确估计互信息。

### 背景

InfoNCE目标最初用于对比表示学习，已成为互信息(MI)估计的流行选择，尽管它与MI的联系是间接的。

### 目的

证明为什么InfoNCE不应被视为有效的MI估计器，并引入一个称为InfoNCE-anchor的简单修改，用于准确的MI估计。

### 方法

通过引入一个辅助锚点类来修改InfoNCE，实现了一致的密度比估计，并产生了一个偏差显著减少的即插即用MI估计器。此外，使用适当的评分规则将框架推广，当使用对数评分时，InfoNCE-anchor作为特例被恢复。这个公式将一系列对比目标（包括NCE、InfoNCE和f-散度变体）统一在一个单一的原则性框架下。

### 主要发现

使用对数评分的InfoNCE-anchor能实现最准确的MI估计；然而，在自监督表示学习实验中，锚点并未提高下游任务性能。

### 结论

对比表示学习受益的不是准确的MI估计本身，而是结构化密度比的学习。

### 翻译

InfoNCE目标最初引入用于对比表示学习，尽管它与互信息(MI)的联系是间接的，但已成为MI估计的流行选择。在本文中，我们证明了为什么不应将InfoNCE视为有效的MI估计器，并介绍了一个简单的修改，我们称之为InfoNCE-anchor，用于准确的MI估计。我们的修改引入了一个辅助锚点类，实现了一致的密度比估计，并产生了一个偏差显著减少的即插即用MI估计器。除此之外，我们使用适当的评分规则推广了我们的框架，当使用对数评分时，InfoNCE-anchor作为特例被恢复。这个公式在单一原则性框架下统一了广泛的对比目标，包括NCE、InfoNCE和f-散度变体。从经验上看，我们发现使用对数评分的InfoNCE-anchor实现了最准确的MI估计；然而，在自监督表示学习实验中，我们发现锚点并未提高下游任务性能。这些发现证实了对比表示学习受益的不是准确的MI估计本身，而是结构化密度比的学习。


### 论文摘要

The InfoNCE objective, originally introduced for contrastive representation learning, has become a popular choice for mutual information (MI) estimation, despite its indirect connection to MI. In this paper, we demonstrate why InfoNCE should not be regarded as a valid MI estimator, and we introduce a simple modification, which we refer to as InfoNCE-anchor, for accurate MI estimation. Our modification introduces an auxiliary anchor class, enabling consistent density ratio estimation and yielding a plug-in MI estimator with significantly reduced bias. Beyond this, we generalize our framework using proper scoring rules, which recover InfoNCE-anchor as a special case when the log score is employed. This formulation unifies a broad spectrum of contrastive objectives, including NCE, InfoNCE, and $f$-divergence variants, under a single principled framework. Empirically, we find that InfoNCE-anchor with the log score achieves the most accurate MI estimates; however, in self-supervised representation learning experiments, we find that the anchor does not improve the downstream task performance. These findings corroborate that contrastive representation learning benefits not from accurate MI estimation per se, but from the learning of structured density ratios.

---

## 11. HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series

**论文链接:** [http://arxiv.org/abs/2510.25785v1](http://arxiv.org/abs/2510.25785v1)

**作者:** Simon A. Lee, Cyrus Tanade, Hao Zhou, Juhyeon Lee, Megha Thukral, Minji Han, Rachel Choi, Md Sazzad Hissain Khan, Baiying Lu, Migyeong Gwak, Mehrab Bin Morshed, Viswam Nathan, Md Mahbubur Rahman, Li Zhu, Subramaniam Venkatraman, Sharanya Arcot Desai

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了一种名为HiMAE的自监督学习方法，用于分析可穿戴传感器产生的生理时间序列数据，探索时间分辨率对预测效用的影响。

### 背景

可穿戴传感器提供了丰富的生理时间序列数据，但支配这些数据预测效用的基本原理尚不清楚。

### 目的

测试时间分辨率作为表征学习基本轴的假设，即不同的临床和行为结果依赖于不同尺度的结构。

### 方法

引入HiMAE（分层掩码自编码器），一种结合掩码自编码和分层卷积编码器-解码器的自监督框架。

### 主要发现

HiMAE能产生多分辨率嵌入，系统评估哪些时间尺度携带预测信号；在分类、回归和生成基准测试中优于最先进模型；体积小几个数量级；可在智能手表上实现亚毫秒级推理，支持真正的边缘计算。

### 结论

HiMAE既是高效的自监督学习方法，也是发现可穿戴健康数据中尺度敏感结构的有效工具。

### 翻译

可穿戴传感器提供了丰富的生理时间序列，但支配其预测效用的原理仍然不清楚。我们假设时间分辨率是表征学习的基本轴，不同的临床和行为结果依赖于不同尺度的结构。为了测试这一分辨率假设，我们引入了HiMAE（分层掩码自编码器），这是一种结合掩码自编码和分层卷积编码器-解码器的自监督框架。HiMAE产生多分辨率嵌入，能够系统评估哪些时间尺度携带预测信号，将分辨率从超参数转变为可解释性的探针。在分类、回归和生成基准测试中，HiMAE始终优于将尺度压缩的最先进基础模型，同时体积小几个数量级。HiMAE是一种高效的表征学习器，足够紧凑，可以在手表上完全运行，在智能手表类CPU上实现亚毫秒级推理，实现真正的边缘推理。总之，这些贡献使HiMAE既成为高效的自监督学习方法，也成为发现可穿戴健康中尺度敏感结构的发现工具。


### 论文摘要

Wearable sensors provide abundant physiological time series, yet the principles governing their predictive utility remain unclear. We hypothesize that temporal resolution is a fundamental axis of representation learning, with different clinical and behavioral outcomes relying on structure at distinct scales. To test this resolution hypothesis, we introduce HiMAE (Hierarchical Masked Autoencoder), a self supervised framework that combines masked autoencoding with a hierarchical convolutional encoder decoder. HiMAE produces multi resolution embeddings that enable systematic evaluation of which temporal scales carry predictive signal, transforming resolution from a hyperparameter into a probe for interpretability. Across classification, regression, and generative benchmarks, HiMAE consistently outperforms state of the art foundation models that collapse scale, while being orders of magnitude smaller. HiMAE is an efficient representation learner compact enough to run entirely on watch, achieving sub millisecond inference on smartwatch class CPUs for true edge inference. Together, these contributions position HiMAE as both an efficient self supervised learning method and a discovery tool for scale sensitive structure in wearable health.

---

## 12. Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education

**论文链接:** [http://arxiv.org/abs/2510.26402v1](http://arxiv.org/abs/2510.26402v1)

**作者:** Vikrant Sahu, Gagan Raj Gupta, Raghav Borikar, Nitin Mane

**发布时间:** 2025-10-30

### GPT解析

### 总结

Autograder+是一个创新的自动评分系统，通过AI驱动的反馈、语义聚类和交互式可视化，将自动评分从纯总结性过程转变为形成性学习体验，减轻教师工作量的同时支持有针对性的教学并促进更强的学习成果。

### 背景

编程教育的快速增长已经超过了传统评估工具的发展，使教师难以提供有意义、可扩展的反馈。传统的自动评分器虽然高效，但作为黑盒系统仅返回通过/失败结果，很少提供关于学生思维或学习需求的见解。

### 目的

将自动评分从纯总结性过程转变为形成性学习体验，为教师提供更有效的评估工具，同时为学生提供更有价值的反馈。

### 方法

引入两个关键功能：1) 使用微调的大语言模型自动生成反馈；2) 可视化学生代码提交以发现学习模式。模型经过精心筛选的学生代码和专家反馈进行微调，确保教育对齐和上下文感知的指导。系统支持提示池，允许教师通过选择的提示模板指导反馈风格。

### 主要发现

在来自多个编程任务的600多个学生提交的评估中，系统生成的反馈与教师评论具有很强的语义一致性。基于1000个带注释的提交训练的对比学习代码嵌入能够基于功能和方法的相似性将解决方案分组为有意义的集群。

### 结论

通过整合AI驱动的反馈、语义聚类和交互式可视化，Autograder+减轻了教师的工作量，同时支持有针对性的教学并促进更强的学习成果。

### 翻译

编程教育的快速增长已经超过了传统评估工具的发展，使教师难以提供有意义、可扩展的反馈。传统的自动评分器虽然高效，但作为黑盒系统仅返回通过/失败结果，很少提供关于学生思维或学习需求的见解。Autograder+旨在将自动评分从纯总结性过程转变为形成性学习体验。它引入了两个关键功能：使用微调的大语言模型自动生成反馈，以及可视化学生代码提交以发现学习模式。该模型经过精心筛选的学生代码和专家反馈进行微调，确保教育对齐、上下文感知的指导。在来自多个编程任务的600多个学生提交的评估中，系统生成的反馈与教师评论具有很强的语义一致性。对于可视化功能，基于1000个带注释的提交训练的对比学习代码嵌入能够基于功能和方法的相似性将解决方案分组为有意义的集群。该系统还支持提示池，允许教师通过选择的提示模板指导反馈风格。通过整合AI驱动的反馈、语义聚类和交互式可视化，Autograder+减轻了教师的工作量，同时支持有针对性的教学并促进更强的学习成果。


### 论文摘要

The rapid growth of programming education has outpaced traditional assessment tools, leaving faculty with limited means to provide meaningful, scalable feedback. Conventional autograders, while efficient, act as black-box systems that simply return pass/fail results, offering little insight into student thinking or learning needs.   Autograder+ is designed to shift autograding from a purely summative process to a formative learning experience. It introduces two key capabilities: automated feedback generation using a fine-tuned Large Language Model, and visualization of student code submissions to uncover learning patterns. The model is fine-tuned on curated student code and expert feedback to ensure pedagogically aligned, context-aware guidance.   In evaluation across 600 student submissions from multiple programming tasks, the system produced feedback with strong semantic alignment to instructor comments. For visualization, contrastively learned code embeddings trained on 1,000 annotated submissions enable grouping solutions into meaningful clusters based on functionality and approach. The system also supports prompt-pooling, allowing instructors to guide feedback style through selected prompt templates.   By integrating AI-driven feedback, semantic clustering, and interactive visualization, Autograder+ reduces instructor workload while supporting targeted instruction and promoting stronger learning outcomes.

---

## 13. Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.26241v1](http://arxiv.org/abs/2510.26241v1)

**作者:** Shiho Matta, Lis Kanashiro Pereira, Peitao Han, Fei Cheng, Shigeru Kitazawa

**发布时间:** 2025-10-30

**备注:** 10 pages

### GPT解析

### 总结

研究揭示了当前视觉语言模型在视频时间信息理解上的显著缺陷，提出了一个简单但有效的评估方法——时间方向判断(AoT)。

### 背景

现代视觉语言模型在许多多模态任务中表现出色，但它们对视频中时间信息的理解仍然薄弱，并且这一点尚未得到充分评估。

### 目的

探究视觉语言模型在理解视频时间信息方面的差距，通过判断视频播放方向(正向或反向)的挑战来评估其时间推理能力。

### 方法

引入AoT-PsyPhyBENCH基准测试，使用经过心理物理学验证的刺激和行为基线，测试VLMs能否推断自然视频中的时间方向，并对开源和专有模型进行全面评估。

### 主要发现

大多数模型表现接近随机水平，即使在物理不可逆过程和因果手动操作上表现最好的模型也远低于人类准确度，而人类几乎能立即识别这些内容。

### 结论

当前多模态系统存在基本差距：虽然捕捉了丰富的视觉语义相关性，但缺乏时间连续性和因果理解所需的归纳偏置。

### 翻译

现代视觉语言模型在许多多模态任务中表现出色，但它们对视频中时间信息的理解仍然薄弱，且这一点尚未得到充分评估。我们通过一个看似简单但具有揭示性的挑战来探究这一差距：判断时间方向(AoT)，即判断短视频片段是正向播放还是反向播放。我们引入了AoT-PsyPhyBENCH，这是一个经过心理物理学验证的基准测试，用于测试VLMs能否使用与人类相同的刺激和行为基线来推断自然视频中的时间方向。我们对开源和专有的、推理和非推理的VLMs进行了全面评估，发现大多数模型表现接近随机水平，即使在物理不可逆过程(如自由落体、扩散/爆炸)和因果手动操作(分割/添加)上表现最好的模型也远低于人类的准确度，而人类几乎能立即识别这些内容。这些结果突显了当前多模态系统的一个基本差距：虽然它们捕捉了丰富的视觉语义相关性，但缺乏时间连续性和因果理解所需的归纳偏置。我们发布了AoT-PsyPhyBENCH的代码和数据，以鼓励VLMs在物理和时间推理能力方面的进一步发展。


### 论文摘要

Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and, crucially, under-evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best lag far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.

---

## 14. STAR: A Privacy-Preserving, Energy-Efficient Edge AI Framework for Human Activity Recognition via Wi-Fi CSI in Mobile and Pervasive Computing Environments

**论文链接:** [http://arxiv.org/abs/2510.26148v1](http://arxiv.org/abs/2510.26148v1)

**作者:** Kexing Liu

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出了一种名为STAR的边缘AI优化框架，用于在低功耗嵌入式设备上实现实时、节能的人类活动识别(HAR)系统。该系统通过简化的神经网络、自适应信号处理和硬件感知优化，在保持高准确率的同时显著提高了计算效率。

### 背景

人类活动识别(HAR)通过Wi-Fi信道状态信息(CSI)提供了一种保护隐私、无需接触的感知方法，适用于智能家居、健康监测和移动物联网系统。然而，现有方法存在计算效率低下、高延迟和资源受限环境中可行性有限的问题。

### 目的

开发STAR(Sensing Technology for Activity Recognition)框架，在低功耗嵌入式设备上实现实时、节能的HAR，解决现有方法在资源受限环境中的局限性。

### 方法

集成轻量级神经网络架构、自适应信号处理和硬件感知联合优化；使用简化的基于门控循环单元(GRU)的循环神经网络，比传统LSTM减少33%参数；采用多阶段预处理管道(中值滤波、8阶巴特沃斯低通滤波和经验模态分解)；在配备NPU的Rockchip RV1126处理器上实现，并与ESP32-S3 CSI采集模块接口。

### 主要发现

在七类活动上的平均识别准确率达93.52%，人体存在检测准确率达99.11%；使用仅97.6k参数的紧凑模型；INT8量化推理以33 MHz速度运行，仅占用8% CPU利用率，比CPU执行快六倍；系统具有亚秒级响应延迟和低功耗。

### 结论

STAR系统确保了实时、隐私保护的HAR，为移动和普适计算环境提供了实用、可扩展的解决方案，有效解决了传统方法在资源受限嵌入式环境中的局限性。

### 翻译

人类活动识别(HAR)通过Wi-Fi信道状态信息(CSI)提供了一种保护隐私、无需接触的感知方法，适用于智能家居、健康监测和移动物联网系统。然而，现有方法常遇到计算效率低下、高延迟以及在资源受限的嵌入式移动边缘环境中可行性有限的问题。本文提出了STAR(Sensing Technology for Activity Recognition)，这是一个边缘AI优化的框架，集成了轻量级神经网络架构、自适应信号处理和硬件感知的联合优化，以在低功耗嵌入式设备上实现实时、节能的HAR。STAR采用简化的基于门控循环单元(GRU)的循环神经网络，比传统LSTM模型减少33%的模型参数，同时保持有效的时间建模能力。采用结合中值滤波、8阶巴特沃斯低通滤波和经验模态分解(EMD)的多阶段预处理管道，用于去噪CSI振幅数据并提取时空特征。对于设备上部署，STAR在配备嵌入式神经处理单元(NPU)的Rockchip RV1126处理器上实现，与基于ESP32-S3的CSI采集模块接口。实验结果显示，在七类活动上的平均识别准确率为93.52%，人体存在检测为99.11%，使用紧凑的97.6k参数模型。INT8量化推理以33 MHz的处理速度运行，仅占用8%的CPU利用率，比基于CPU的执行速度快六倍。凭借亚秒级响应延迟和低功耗，该系统确保了实时、隐私保护的HAR，为移动和普适计算环境提供了实用、可扩展的解决方案。


### 论文摘要

Human Activity Recognition (HAR) via Wi-Fi Channel State Information (CSI) presents a privacy-preserving, contactless sensing approach suitable for smart homes, healthcare monitoring, and mobile IoT systems. However, existing methods often encounter computational inefficiency, high latency, and limited feasibility within resource-constrained, embedded mobile edge environments. This paper proposes STAR (Sensing Technology for Activity Recognition), an edge-AI-optimized framework that integrates a lightweight neural architecture, adaptive signal processing, and hardware-aware co-optimization to enable real-time, energy-efficient HAR on low-power embedded devices. STAR incorporates a streamlined Gated Recurrent Unit (GRU)-based recurrent neural network, reducing model parameters by 33% compared to conventional LSTM models while maintaining effective temporal modeling capability. A multi-stage pre-processing pipeline combining median filtering, 8th-order Butterworth low-pass filtering, and Empirical Mode Decomposition (EMD) is employed to denoise CSI amplitude data and extract spatial-temporal features. For on-device deployment, STAR is implemented on a Rockchip RV1126 processor equipped with an embedded Neural Processing Unit (NPU), interfaced with an ESP32-S3-based CSI acquisition module. Experimental results demonstrate a mean recognition accuracy of 93.52% across seven activity classes and 99.11% for human presence detection, utilizing a compact 97.6k-parameter model. INT8 quantized inference achieves a processing speed of 33 MHz with just 8% CPU utilization, delivering sixfold speed improvements over CPU-based execution. With sub-second response latency and low power consumption, the system ensures real-time, privacy-preserving HAR, offering a practical, scalable solution for mobile and pervasive computing environments.

---

## 15. EgoExo-Con: Exploring View-Invariant Video Temporal Understanding

**论文链接:** [http://arxiv.org/abs/2510.26113v1](http://arxiv.org/abs/2510.26113v1)

**作者:** Minjoon Jung, Junbin Xiao, Junghyun Kim, Byoung-Tak Zhang, Angela Yao

**发布时间:** 2025-10-30

**备注:** project page:  \url{https://minjoong507.github.io/projects/EgoExo-Con/}

### GPT解析

### 总结

本研究探讨了视频大语言模型在不同视角下捕捉同一事件时的时间理解一致性问题，提出了EgoExo-Con基准测试和View-GRPO强化学习框架来解决现有模型的局限性。

### 背景

视频大语言模型在不同视角下捕捉同一事件时可能存在时间理解不一致的问题。

### 目的

研究视频大语言模型在不同视角下捕捉同一事件时是否能保持一致的时间理解能力，并提出改进方法。

### 方法

引入EgoExo-Con基准测试，包含全面同步的第一人称和第三人称视角视频对及人类优化的自然语言查询，强调时间验证和时间定位两个任务，并提出View-GRPO强化学习框架来增强特定视角的时间推理并促进跨视角一致性理解。

### 主要发现

现有Video-LLMs存在两个关键限制：(1)模型通常无法保持一致性，结果远差于单视角表现；(2)当使用双视角同步视频微调时，模型显示出一致性改进，但表现往往不如单视角训练的模型。

### 结论

View-GRPO方法在提高跨视角一致性方面优于简单的SFT和GRPO，所有研究资源将公开可用。

### 翻译

当视频从不同视角捕捉同一事件时，视频大语言模型能否实现一致的时间理解？为研究此问题，我们引入了EgoExo-Con(一致性)基准，该基准包含全面同步的第一人称和第三人称视频对以及人类优化的自然语言查询。EgoExo-Con强调两个时间理解任务：时间验证和时间定位。它不仅评估正确性，还评估跨视角的一致性。我们的分析揭示了现有Video-LLMs的两个关键局限：(1)模型通常无法保持一致性，结果远差于其单视角表现。(2)当使用双视角同步视频进行微调时，模型显示出一致性改进，但往往表现不如单视角训练的模型。为改进，我们提出了View-GRPO，一种新的强化学习框架，能有效增强特定视角的时间推理，同时鼓励跨视角的一致性理解。我们的方法在提高跨视角一致性方面优于简单的SFT和GRPO。所有资源将公开提供。


### 论文摘要

Can Video-LLMs achieve consistent temporal understanding when videos capture the same event from different viewpoints? To study this, we introduce EgoExo-Con (Consistency), a benchmark of comprehensively synchronized egocentric and exocentric video pairs with human-refined queries in natural language. EgoExo-Con emphasizes two temporal understanding tasks: Temporal Verification and Temporal Grounding. It evaluates not only correctness but consistency across viewpoints. Our analysis reveals two critical limitations of existing Video-LLMs: (1) models often fail to maintain consistency, with results far worse than their single-view performances. (2) When naively finetuned with synchronized videos of both viewpoints, the models show improved consistency but often underperform those trained on a single view. For improvements, we propose View-GRPO, a novel reinforcement learning framework that effectively strengthens view-specific temporal reasoning while encouraging consistent comprehension across viewpoints. Our method demonstrates its superiority over naive SFT and GRPO, especially for improving cross-view consistency. All resources will be made publicly available.

---

## 16. Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders

**论文链接:** [http://arxiv.org/abs/2510.26027v1](http://arxiv.org/abs/2510.26027v1)

**作者:** Ali Rasekh, Erfan Bagheri Soula, Omid Daliran, Simon Gottschalk, Mohsen Fayyaz

**发布时间:** 2025-10-29

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本文提出了一种改进的视频大语言模型架构，通过在视觉编码器中引入堆叠时序注意力模块，解决了当前模型在理解视频时序动态方面的局限性，显著提升了时序推理能力和动作识别性能。

### 背景

尽管多模态大语言模型(MLLMs)已取得显著进展，但理解视频中复杂的时序动态仍然是一个主要挑战。当前视频大语言模型(Video-LLM)架构在时序理解方面存在关键局限性，难以处理需要详细理解动作序列和时间进展的任务。

### 目的

提出一种改进的Video-LLM架构，增强模型对视频时序动态的理解能力，特别是在动作序列和时间进展方面的理解。

### 方法

在视觉编码器中直接引入堆叠的时序注意力模块，在视觉编码器中融入时序注意力，使模型能够更好地捕捉动作进展和帧之间的关系，在将视觉令牌传递给LLM之前增强时序理解。

### 主要发现

该方法显著提高了时序推理能力，在视频问答任务中优于现有模型，特别是在动作识别方面。在VITATECS、MVBench和Video-MME等基准测试中，性能提升了高达5.5%。

### 结论

通过增强视觉编码器的时序结构，解决了Video-LLMs在视频理解方面的关键差距，为视频时序理解提供了新的解决方案。

### 翻译

尽管多模态大语言模型(MLLMs)取得了显著进展，但理解视频中复杂的时序动态仍然是一个主要挑战。我们的实验表明，当前视频大语言模型(Video-LLM)架构在时序理解方面存在关键局限性，难以处理需要详细理解动作序列和时间进展的任务。在这项工作中，我们提出了一种Video-LLM架构，在视觉编码器中直接引入堆叠的时序注意力模块。这种设计在视觉编码器中融入了时序注意力，使模型能够在将视觉令牌传递给LLM之前更好地捕捉动作进展和帧之间的关系。我们的结果表明，这种方法显著提高了时序推理能力，并在视频问答任务中优于现有模型，特别是在动作识别方面。我们在VITATECS、MVBench和Video-MME等基准测试中提高了高达5.5%。通过增强视觉编码器的时序结构，我们解决了Video-LLMs在视频理解方面的关键差距。项目页面和代码可在以下网址获取：https://alirasekh.github.io/STAVEQ2/。


### 论文摘要

Despite significant advances in Multimodal Large Language Models (MLLMs), understanding complex temporal dynamics in videos remains a major challenge. Our experiments show that current Video Large Language Model (Video-LLM) architectures have critical limitations in temporal understanding, struggling with tasks that require detailed comprehension of action sequences and temporal progression. In this work, we propose a Video-LLM architecture that introduces stacked temporal attention modules directly within the vision encoder. This design incorporates a temporal attention in vision encoder, enabling the model to better capture the progression of actions and the relationships between frames before passing visual tokens to the LLM. Our results show that this approach significantly improves temporal reasoning and outperforms existing models in video question answering tasks, specifically in action recognition. We improve on benchmarks including VITATECS, MVBench, and Video-MME by up to +5.5%. By enhancing the vision encoder with temporal structure, we address a critical gap in video understanding for Video-LLMs. Project page and code are available at: https://alirasekh.github.io/STAVEQ2/.

---

## 17. From Queries to Insights: Agentic LLM Pipelines for Spatio-Temporal Text-to-SQL

**论文链接:** [http://arxiv.org/abs/2510.25997v1](http://arxiv.org/abs/2510.25997v1)

**作者:** Manu Redd, Tao Zhe, Dongjie Wang

**发布时间:** 2025-10-29

**DOI:** 10.1145/3764915.3770724

**备注:** 8 pages, 5 figures, GeoGenAgent'25 - ACM SIGSPATIAL

### GPT解析

### 总结

研究提出了一种基于代理的管道系统，用于处理复杂的空间和时间自然语言查询，显著提高了查询准确性和用户友好性。

### 背景

现有的自然语言到SQL系统在处理真实空间和时间查询时存在困难，需要将模糊的用户表述与特定模式类别匹配、处理时间推理并选择适当输出。

### 目的

开发能够处理复杂空间和时间查询的系统，支持缺乏SQL专业知识、详细模式知识或提示技能的用户。

### 方法

构建了一个基于代理的管道，通过基于Mistral的ReAct代理对基础文本到SQL模型(llama-3-sqlcoder-8b)进行编排，使代理能够通过模式检查、SQL生成、执行和可视化工具来规划、分解和调整查询。

### 主要发现

在纽约和东京签到数据集的35个自然语言查询评估中，代理系统准确率达到91.4%，而基础模型仅为28.6%，并通过地图、图表和结构化的自然语言摘要显著增强了可用性。

### 结论

代理编排而非更强的SQL生成器本身是构建交互式地理空间助手的有前途的基础。

### 翻译

自然语言到SQL系统有望使结构化数据访问民主化，使用户无需学习SQL即可查询数据库。然而，现有系统在处理现实空间时间查询方面存在困难，成功需要将模糊的用户表述与特定模式类别对齐、处理时间推理并选择适当输出。我们提出了一种基于代理的管道，通过基于Mistral的ReAct代理的编排，扩展了一个基础文本到SQL模型(llama-3-sqlcoder-8b)。该代理可以通过模式检查、SQL生成、执行和可视化工具来规划、分解和调整查询。我们在纽约和东京签到数据集上的35个自然语言查询进行了评估，涵盖了空间、时间和多数据集推理。代理的准确率显著高于基础模型，达到91.4%对28.6%，并通过地图、图表和结构化的自然语言摘要增强了可用性。关键的是，我们的设计支持了更自然的人机数据库交互，支持缺乏SQL专业知识、详细模式知识或提示技能的用户。我们得出结论，代理编排而非更强的SQL生成器本身，是交互式地理空间助手的有前途的基础。


### 论文摘要

Natural-language-to-SQL (NL-to-SQL) systems hold promise for democratizing access to structured data, allowing users to query databases without learning SQL. Yet existing systems struggle with realistic spatio-temporal queries, where success requires aligning vague user phrasing with schema-specific categories, handling temporal reasoning, and choosing appropriate outputs. We present an agentic pipeline that extends a naive text-to-SQL baseline (llama-3-sqlcoder-8b) with orchestration by a Mistral-based ReAct agent. The agent can plan, decompose, and adapt queries through schema inspection, SQL generation, execution, and visualization tools. We evaluate on 35 natural-language queries over the NYC and Tokyo check-in dataset, covering spatial, temporal, and multi-dataset reasoning. The agent achieves substantially higher accuracy than the naive baseline 91.4% vs. 28.6% and enhances usability through maps, plots, and structured natural-language summaries. Crucially, our design enables more natural human-database interaction, supporting users who lack SQL expertise, detailed schema knowledge, or prompting skill. We conclude that agentic orchestration, rather than stronger SQL generators alone, is a promising foundation for interactive geospatial assistants.

---

## 18. Enhancing Underwater Object Detection through Spatio-Temporal Analysis and Spatial Attention Networks

**论文链接:** [http://arxiv.org/abs/2510.25797v1](http://arxiv.org/abs/2510.25797v1)

**作者:** Sai Likhith Karri, Ansh Saxena

**发布时间:** 2025-10-29

### GPT解析

### 总结

该研究评估了时空建模和空间注意力机制在水下物体检测中的有效性，比较了标准YOLOv5、T-YOLOv5及其与CBAM结合的变体性能。

### 背景

水下物体检测在动态海洋环境中面临挑战，如突然运动、部分遮挡和逐渐运动等。

### 目的

研究时空建模和空间注意力机制如何提高水下物体检测的准确性和鲁棒性。

### 方法

分两个阶段进行，第一阶段评估T-YOLOv5与标准YOLOv5的性能比较；第二阶段开发添加了卷积块注意力模块(CBAM)的T-YOLOv5增强版本。

### 主要发现

T-YOLOv5和T-YOLOv5与CBAM的变体在mAP@50-95指标上分别达到0.813和0.811，显著优于标准YOLOv5的0.563。

### 结论

T-YOLOv5相比标准模型显著提高了检测可靠性，而T-YOLOv5与CBAM在具有挑战性的场景中进一步提高了性能，但在简单场景中会损失一些准确性。

### 翻译

该研究检验了时空建模和空间注意力机制在深度学习模型中用于水下物体检测的有效性。具体而言，在第一阶段，评估了增强时序的YOLOv5变体T-YOLOv5与标准YOLOv5的性能比较。在第二阶段，通过添加卷积块注意力模块(CBAM)开发了T-YOLOv5的增强版本。研究表明，CBAM如何通过时序建模提高了在动态海洋环境中的检测准确性，特别是在突然运动、部分遮挡和逐渐运动的条件下。测试结果显示，YOLOv5达到了0.563的mAP@50-95，而T-YOLOv5和带有CBAM的T-YOLOv5分别以0.813和0.811的mAP@50-95分数表现更优，突显了它们在检测复杂物体方面的卓越准确性和泛化能力。研究结果表明，与标准模型相比，T-YOLOv5显著提高了检测可靠性，而带有CBAM的T-YOLOv5在具有挑战性的场景中进一步提高了性能，尽管在简单场景中会损失一些准确性。


### 论文摘要

This study examines the effectiveness of spatio-temporal modeling and the integration of spatial attention mechanisms in deep learning models for underwater object detection. Specifically, in the first phase, the performance of temporal-enhanced YOLOv5 variant T-YOLOv5 is evaluated, in comparison with the standard YOLOv5. For the second phase, an augmented version of T-YOLOv5 is developed, through the addition of a Convolutional Block Attention Module (CBAM). By examining the effectiveness of the already pre-existing YOLOv5 and T-YOLOv5 models and of the newly developed T-YOLOv5 with CBAM. With CBAM, the research highlights how temporal modeling improves detection accuracy in dynamic marine environments, particularly under conditions of sudden movements, partial occlusions, and gradual motion. The testing results showed that YOLOv5 achieved a mAP@50-95 of 0.563, while T-YOLOv5 and T-YOLOv5 with CBAM outperformed with mAP@50-95 scores of 0.813 and 0.811, respectively, highlighting their superior accuracy and generalization in detecting complex objects. The findings demonstrate that T-YOLOv5 significantly enhances detection reliability compared to the standard model, while T-YOLOv5 with CBAM further improves performance in challenging scenarios, although there is a loss of accuracy when it comes to simpler scenarios.

---

## 19. Dynamic Context-Aware Scene Reasoning Using Vision-Language Alignment in Zero-Shot Real-World Scenarios

**论文链接:** [http://arxiv.org/abs/2510.26580v1](http://arxiv.org/abs/2510.26580v1)

**作者:** Manjunath Prasad Holenarasipura Rajiv, B. M. Vidyavathi

**发布时间:** 2025-10-30

**备注:** Preprint under review at IEEE Transactions on Pattern Analysis and  Machine Intelligence (TPAMI), 2025

### GPT解析

### 总结

该研究提出了一种动态上下文感知场景推理框架，利用视觉语言对齐解决零样本现实世界场景问题，通过结合视觉Transformer和大语言模型显著提高了复杂环境中的场景理解准确性。

### 背景

在现实世界环境中，AI系统经常面临没有标记数据的陌生场景，这给传统场景理解模型带来重大挑战。无法在未见过的上下文中进行泛化限制了基于视觉的应用程序在动态、非结构化环境中的部署。

### 目的

使智能系统能够在没有特定任务先验训练的情况下推断并适应新环境。

### 方法

提出的方法集成了预训练的视觉Transformer和大语言模型，将视觉语义与自然语言描述对齐增强上下文理解能力。动态推理模块通过结合全局场景线索和由语言先验引导的对象级交互来优化预测。

### 主要发现

在COCO、Visual Genome和Open Images等零样本基准上的实验表明，在复杂且未见过的环境中，场景理解准确性比基线模型提高了高达18%。由于视觉和语言的协同融合，系统在模糊或杂乱的场景中表现出强大的性能。

### 结论

该框架为上下文感知推理提供了一种可扩展和可解释的方法，推动了动态现实世界环境中的零样本泛化。

### 翻译

在现实世界环境中，AI系统经常面临没有标记数据的陌生场景，这给传统的场景理解模型带来了重大挑战。无法在未见过的上下文中进行泛化限制了基于视觉的应用程序在动态、非结构化环境中的部署。这项工作引入了一种动态上下文感知场景推理框架，利用视觉语言对齐来解决零样本现实世界场景问题。目标是使智能系统能够在没有特定任务先验训练的情况下推断并适应新环境。提出的方法集成了预训练的视觉Transformer和大语言模型，将视觉语义与自然语言描述对齐，增强上下文理解能力。动态推理模块通过结合全局场景线索和由语言先验引导的对象级交互来优化预测。在COCO、Visual Genome和Open Images等零样本基准上的广泛实验表明，在复杂且未见过的环境中，场景理解准确性比基线模型提高了高达18%。结果还显示，由于视觉和语言的协同融合，在模糊或杂乱的场景中表现出强大的性能。该框架为上下文感知推理提供了一种可扩展和可解释的方法，推动了动态现实世界环境中的零样本泛化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决AI系统在零样本(real-world)场景下进行动态上下文感知场景推理的问题。传统场景理解模型在面对没有标记数据的新环境时无法有效泛化，这限制了视觉应用在自动驾驶、机器人导航、监控等动态、非结构化环境中的部署。这个问题在现实中非常重要，因为真实世界环境往往是复杂、多变且缺乏标记数据的，AI系统需要能够理解和适应从未见过的新场景而不需要针对每个特定任务重新训练。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过整合预训练的视觉转换器(visual transformers)和大语言模型(large language models)来设计这个方法。他们借鉴了多项现有工作，包括Context VLM用于自动驾驶安全、结合开放世界检测器和大视觉语言模型的零样本目标识别、基于图的视觉语言模型调整方法、利用CLIP模型的动态场景恢复框架等。作者在这些工作的基础上，提出了一个动态推理模块，该模块利用全局场景线索和对象级交互，由语言先验指导，从而实现更有效的场景理解。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过视觉-语言对齐来实现动态上下文感知的场景推理，使AI系统能够在没有任务特定监督的情况下理解和适应新环境。整体实现流程包括：1)使用视觉编码器(如CLIP-ViT)提取丰富的视觉特征；2)使用语言编码器(如GPT或BERT变体)建模语义先验；3)采用基于注意力的跨模态融合机制对齐视觉和语言；4)引入上下文精炼单元，建模对象级交互和全局场景语义；5)在零样本场景下评估模型性能，通过计算视觉和文本嵌入的余弦相似度来进行场景理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出动态上下文感知场景推理框架，整合预训练视觉转换器和大语言模型；2)引入动态推理模块，利用全局场景线索和对象级交互；3)在多个零样本基准数据集上证明模型泛化和适应性；4)在模糊或杂乱场景中表现出强大性能；5)提供可扩展和可解释的上下文感知推理方法。相比之前的工作，这个方法的主要不同在于：结合了动态推理能力和视觉-语言对齐；不依赖任务特定监督；能处理复杂、模糊场景；具有更好的泛化能力和可解释性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的动态上下文感知场景推理框架，通过视觉-语言对齐实现了在没有任务特定监督的情况下理解和适应新环境的能力，显著提升了AI系统在复杂、模糊和杂乱场景中的表现。'}


### 论文摘要

In real-world environments, AI systems often face unfamiliar scenarios without labeled data, creating a major challenge for conventional scene understanding models. The inability to generalize across unseen contexts limits the deployment of vision-based applications in dynamic, unstructured settings. This work introduces a Dynamic Context-Aware Scene Reasoning framework that leverages Vision-Language Alignment to address zero-shot real-world scenarios. The goal is to enable intelligent systems to infer and adapt to new environments without prior task-specific training. The proposed approach integrates pre-trained vision transformers and large language models to align visual semantics with natural language descriptions, enhancing contextual comprehension. A dynamic reasoning module refines predictions by combining global scene cues and object-level interactions guided by linguistic priors. Extensive experiments on zero-shot benchmarks such as COCO, Visual Genome, and Open Images demonstrate up to 18% improvement in scene understanding accuracy over baseline models in complex and unseen environments. Results also show robust performance in ambiguous or cluttered scenes due to the synergistic fusion of vision and language. This framework offers a scalable and interpretable approach for context-aware reasoning, advancing zero-shot generalization in dynamic real-world settings.

---

## 20. AgriGS-SLAM: Orchard Mapping Across Seasons via Multi-View Gaussian Splatting SLAM

**论文链接:** [http://arxiv.org/abs/2510.26358v1](http://arxiv.org/abs/2510.26358v1)

**作者:** Mirko Usuelli, David Rapado-Rincon, Gert Kootstra, Matteo Matteucci

**发布时间:** 2025-10-30

### GPT解析

### 总结

AgriGS-SLAM是一种结合视觉和激光雷达的SLAM框架，利用多相机3D高斯溅射技术实现果园环境的实时3D场景理解，克服了重复几何结构、季节变化和风吹 foliage 运动的挑战。

### 背景

果园中的自主机器人需要实时3D场景理解，尽管存在重复的行列几何结构、季节性外观变化和风吹 foliage 运动。

### 目的

开发一种能够处理果园环境特殊挑战的实时3D场景理解系统。

### 方法

结合直接激光雷达里程计和闭环检测与多相机3D高斯溅射渲染；通过互补视角的批量栅格化恢复遮挡下的果园结构；在关键帧之间执行统一的梯度驱动地图生命周期；基于概率激光雷达深度一致性项进行姿态优化并加强几何-外观耦合。

### 主要发现

在苹果和梨果园的休眠期、开花期和收获期测试；跨季节和站点提供比先进3DGS-SLAM基线更清晰、更稳定的重建和轨迹；在拖拉机上保持实时性能。

### 结论

虽然演示于果园监测，但该方法可应用于需要鲁棒多模态感知的其他户外领域。

### 翻译

果园中的自主机器人需要实时3D场景理解，尽管存在重复的行列几何结构、季节性外观变化和风吹 foliage 运动。我们提出了AgriGS-SLAM，这是一种结合直接激光雷达里程计和闭环检测与多相机3D高斯溅射渲染的视觉-激光雷达SLAM框架。通过互补视角的批量栅格化恢复遮挡下的果园结构，同时在关键帧之间执行统一的梯度驱动地图生命周期以保留精细细节并限制内存使用。姿态优化由基于概率激光雷达的深度一致性项引导，通过相机投影反向传播以加强几何-外观耦合。我们在苹果和梨果园的休眠期、开花期和收获期使用标准化轨迹协议部署了该系统，评估训练视图和新颖视图合成以减少3DGS评估中的过拟合。跨季节和站点，AgriGS-SLAM比最近的先进3DGS-SLAM基线提供更清晰、更稳定的重建和更稳定的轨迹，同时在拖拉机上保持实时性能。虽然演示于果园监测，但该方法可应用于需要鲁棒多模态感知的其他户外领域。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决果园环境中自主机器人的实时3D场景理解问题，特别是在面对季节性外观变化、重复的行几何结构和风吹引起的叶片运动等挑战时的SLAM系统适应性。这个问题很重要，因为全球人口增长和劳动力短缺增加了对自主农业技术的需求，果园机器人需要准确的3D重建能力来执行导航、收获、喷洒和修剪等任务，同时农民需要即时反馈来调整田间操作，农业数字孪子也需要物理世界与数字表示之间的持续同步。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有SLAM系统在农业环境中的局限性：视觉-only方法在重复作物模式和植被运动下会失败，而激光雷达-only系统在几何稀疏性方面受限。他们借鉴了神经渲染的最新进展，特别是3D高斯泼溅(3DGS)的显式点表示和高效光栅化特性，适合大型场景SLAM的增量特性。方法设计上结合了直接激光雷达里程计(DLO)作为前端，因子图作为后端，并扩展了3DGS到多视图设置以处理果园遮挡，同时参考了现有的多视图优化和闭环检测技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合视觉和激光雷达两种传感器的优势，使用多摄像头系统提供互补视角解决遮挡问题，利用3D高斯泼溅进行实时场景表示，设计梯度驱动的地图生命周期管理，以及使用多模态损失函数联合优化场景表示和机器人定位。整体流程包括：SLAM前端使用直接激光雷达里程计估计运动；SLAM后端维护关键帧姿势的因子图；多视图3D高斯泼溅部分进行增量映射、内存管理和泼溅生命周期操作；优化循环调度各种操作；最后通过多模态损失函数结合光度一致性和几何一致性进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 实时视觉-激光雷达3DGS-SLAM系统；2) 跨季节适用性基准评估方法；3) 统一的梯度驱动的3DGS-SLAM地图和定位优化；4) 支持多摄像头设置的户外3DGS-SLAM框架。相比之前工作，该方法专门针对果园环境设计，使用多摄像头系统而非单摄像头，结合激光雷达里程计和闭环检测而非仅依赖视觉或激光雷达，设计了特定的地图生命周期管理策略，使用多模态损失函数结合光度和几何一致性，并在真实果园的不同生长阶段进行了广泛测试。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AgriGS-SLAM通过结合多视图3D高斯泼溅与激光雷达里程计和闭环检测，实现了果园环境中的实时、跨季节精确地图构建和机器人定位，解决了季节性变化、重复结构和遮挡带来的挑战。'}


### 论文摘要

Autonomous robots in orchards require real-time 3D scene understanding despite repetitive row geometry, seasonal appearance changes, and wind-driven foliage motion. We present AgriGS-SLAM, a Visual--LiDAR SLAM framework that couples direct LiDAR odometry and loop closures with multi-camera 3D Gaussian Splatting (3DGS) rendering. Batch rasterization across complementary viewpoints recovers orchard structure under occlusions, while a unified gradient-driven map lifecycle executed between keyframes preserves fine details and bounds memory. Pose refinement is guided by a probabilistic LiDAR-based depth consistency term, back-propagated through the camera projection to tighten geometry-appearance coupling. We deploy the system on a field platform in apple and pear orchards across dormancy, flowering, and harvesting, using a standardized trajectory protocol that evaluates both training-view and novel-view synthesis to reduce 3DGS overfitting in evaluation. Across seasons and sites, AgriGS-SLAM delivers sharper, more stable reconstructions and steadier trajectories than recent state-of-the-art 3DGS-SLAM baselines while maintaining real-time performance on-tractor. While demonstrated in orchard monitoring, the approach can be applied to other outdoor domains requiring robust multimodal perception.

---

## 21. A Three-Stage Bayesian Transfer Learning Framework to Improve Predictions in Data-Scarce Domains

**论文链接:** [http://arxiv.org/abs/2510.26541v1](http://arxiv.org/abs/2510.26541v1)

**作者:** Aidan Furlong, Robert Salko, Xingang Zhao, Xu Wu

**发布时间:** 2025-10-30

**备注:** Submitted to Engineering Applications of Artificial Intelligence

### GPT解析

### 总结

该研究提出了一种全监督的三阶段框架（staged B-DANN），结合参数迁移和共享潜在空间适应，用于解决数据稀缺领域的机器学习问题。该方法通过确定性特征提取、对抗性优化和贝叶斯微调三个阶段，提高了预测准确性和泛化能力，同时提供不确定性估计。

### 背景

机器学习在工程领域的应用持续增长，深度神经网络因其性能和可访问性被广泛采用，但需要大量高质量数据集。实验数据通常稀疏、嘈杂或不足以构建稳健的数据驱动模型。迁移学习利用数据丰富的源领域来辅助数据稀缺目标领域的学习，但传统参数迁移在领域差异较大时性能下降，领域对抗神经网络(DANNs)在半监督设置下可以处理更大的领域偏移，但训练不稳定且缺乏不确定性量化能力。

### 目的

开发一种能够处理领域差异、提供不确定性估计并提高数据稀缺领域预测准确性和泛化能力的机器学习方法。

### 方法

提出了一种全监督的三阶段框架：阶段性贝叶斯领域对抗神经网络(staged B-DANN)。第一阶段在源领域训练确定性特征提取器；第二阶段使用DANN对抗性地优化特征提取器；第三阶段在适应的特征提取器上构建贝叶斯神经网络，用于在目标领域微调，处理条件偏移并提供校准的不确定性估计。

### 主要发现

在合成基准测试中，该方法显著优于标准迁移技术；应用于预测矩形通道中的临界热通量时，利用管状实验数据作为源领域，结果显示staged B-DANN方法可以提高预测准确性和泛化能力。

### 结论

staged B-DANN方法可以改进预测准确性和泛化能力，并提供不确定性估计，可能有助于核工程等其他领域的应用。

### 翻译

机器学习在工程领域的应用持续增长，以支持广泛的应用。在这些方法中，深度神经网络因其性能和可访问性被广泛采用，但它们需要大量高质量的数据集。实验数据通常稀疏、嘈杂或不足以构建稳健的数据驱动模型。迁移学习利用数据丰富的相关源领域来辅助数据稀缺目标领域的学习，已显示出有效性。参数迁移（重用预训练权重）很常见，但在大的领域偏移下性能会下降。领域对抗神经网络(DANNs)通过学习领域不变的表示来解决这个问题，从而在半监督设置下提高较大领域偏移的迁移效果。然而，DANNs在训练过程中可能不稳定，且缺乏原生的不确定性量化手段。本研究引入了一种全监督的三阶段框架——阶段性贝叶斯领域对抗神经网络(staged B-DANN)，它结合了参数迁移和共享潜在空间适应。在第一阶段，在源领域训练确定性特征提取器。然后在第二阶段使用DANN对抗性地优化该特征提取器。在第三阶段，在适应的特征提取器上构建贝叶斯神经网络，用于在目标领域进行微调，以处理条件偏移并产生校准的不确定性估计。首先在合成基准测试中验证了这种阶段性B-DANN方法，结果表明它显著优于标准迁移技术。然后将其应用于预测矩形通道中的临界热通量任务，利用管状实验数据作为源领域。本研究结果表明，阶段性B-DANN方法可以提高预测准确性和泛化能力，可能有助于核工程的其他领域。


### 论文摘要

The use of ML in engineering has grown steadily to support a wide array of applications. Among these methods, deep neural networks have been widely adopted due to their performance and accessibility, but they require large, high-quality datasets. Experimental data are often sparse, noisy, or insufficient to build resilient data-driven models. Transfer learning, which leverages relevant data-abundant source domains to assist learning in data-scarce target domains, has shown efficacy. Parameter transfer, where pretrained weights are reused, is common but degrades under large domain shifts. Domain-adversarial neural networks (DANNs) help address this issue by learning domain-invariant representations, thereby improving transfer under greater domain shifts in a semi-supervised setting. However, DANNs can be unstable during training and lack a native means for uncertainty quantification. This study introduces a fully-supervised three-stage framework, the staged Bayesian domain-adversarial neural network (staged B-DANN), that combines parameter transfer and shared latent space adaptation. In Stage 1, a deterministic feature extractor is trained on the source domain. This feature extractor is then adversarially refined using a DANN in Stage 2. In Stage 3, a Bayesian neural network is built on the adapted feature extractor for fine-tuning on the target domain to handle conditional shifts and yield calibrated uncertainty estimates. This staged B-DANN approach was first validated on a synthetic benchmark, where it was shown to significantly outperform standard transfer techniques. It was then applied to the task of predicting critical heat flux in rectangular channels, leveraging data from tube experiments as the source domain. The results of this study show that the staged B-DANN method can improve predictive accuracy and generalization, potentially assisting other domains in nuclear engineering.

---

## 22. Analysis of the Robustness of an Edge Detector Based on Cellular Automata Optimized by Particle Swarm

**论文链接:** [http://arxiv.org/abs/2510.26509v1](http://arxiv.org/abs/2510.26509v1)

**作者:** Vinícius Ferraria, Eurico Ruivo

**发布时间:** 2025-10-30

### GPT解析

### 总结

该研究开发了一种基于二维元胞自动机的可适应边缘检测器，通过元启发式方法和迁移学习技术进行优化。研究分析了优化阶段搜索空间扩展的影响以及检测器在不同图像集上的适应性。

### 背景

边缘检测是图像处理中提取相关信息的重要任务，但现有检测器存在难以检测松散边缘和缺乏上下文信息等问题。

### 目的

分析扩大优化阶段搜索空间的影响，以及检测器在识别自然图像集及其专门子集边缘时的适应性稳健性。

### 方法

开发了一种由二维元胞自动机描述并通过元启发式方法结合迁移学习技术优化的可适应检测器。

### 主要发现

扩大优化阶段的搜索空间对所选图像集并不有效；模型能够适应不同输入，但迁移学习技术未带来显著改进。

### 结论

所提出的检测器具有良好的适应性，但扩大搜索空间和迁移学习技术未能显著提高性能。

### 翻译

边缘检测任务在图像处理中至关重要，旨在从图像中提取相关信息。此任务中存在一个反复出现的问题，即某些检测器的弱点，例如难以检测松散边缘以及缺乏从特定问题中提取相关信息的上下文。为解决这些弱点并使检测器适应图像特性，研究人员开发了一种由二维元胞自动机描述并通过元启发式方法结合迁移学习技术优化的可适应检测器。本研究旨在分析扩大优化阶段搜索空间的影响，以及检测器在识别自然图像集及其从同一图像集中提取的专门子集边缘时的适应性稳健性。获得的结果证明，对于所选图像集，扩大优化阶段的搜索空间并不有效。研究还通过一系列实验和验证技术分析了模型的适应性，发现无论采用何种验证方法，模型都能适应输入，并且应用于模型的迁移学习技术未显示出显著改进。


### 论文摘要

The edge detection task is essential in image processing aiming to extract relevant information from an image. One recurring problem in this task is the weaknesses found in some detectors, such as the difficulty in detecting loose edges and the lack of context to extract relevant information from specific problems. To address these weaknesses and adapt the detector to the properties of an image, an adaptable detector described by two-dimensional cellular automaton and optimized by meta-heuristic combined with transfer learning techniques was developed. This study aims to analyze the impact of expanding the search space of the optimization phase and the robustness of the adaptability of the detector in identifying edges of a set of natural images and specialized subsets extracted from the same image set. The results obtained prove that expanding the search space of the optimization phase was not effective for the chosen image set. The study also analyzed the adaptability of the model through a series of experiments and validation techniques and found that, regardless of the validation, the model was able to adapt to the input and the transfer learning techniques applied to the model showed no significant improvements.

---

## 23. Applications of Machine Learning in Polymer Materials: Property Prediction, Material Design, and Systematic Processes

**论文链接:** [http://arxiv.org/abs/2510.26100v1](http://arxiv.org/abs/2510.26100v1)

**作者:** Hongtao Guo Shuai Li Shu Li

**发布时间:** 2025-10-30

**备注:** 55 pages, 6 tables, 9 figures, a systematic review on the research  progress and application prospects of machine learning in polymer materials

### GPT解析

### 总结

本文系统综述了机器学习技术在聚合物材料领域的研究进展和应用前景，介绍了基本技术、应用方法、当前挑战和未来趋势。

### 背景

机器学习技术在聚合物材料领域发展迅速，显著加速了材料预测和设计，但其复杂性也给传统领域研究者带来理解和应用困难。聚合物材料研发面临结构复杂性和传统试错方法局限性等挑战。

### 目的

应对聚合物材料研发中的挑战，解决机器学习方法的复杂性给传统领域研究者带来的理解和应用困难，促进机器学习技术在聚合物材料领域的有效应用。

### 方法

分析聚合物材料研发中的固有挑战；介绍分子描述符、特征表示、数据标准化和清洗等关键技术；记录高质量聚合物数据库；构建高可靠性机器学习模型；实施实验验证、模型评估和优化方法。

### 主要发现

机器学习在聚合物性能预测和材料设计中发挥关键作用；传统机器学习、深度学习和迁移学习等算法有具体应用；数据驱动设计策略包括反向设计、高通量虚拟筛选和多目标优化。

### 结论

当前研究面临数据质量和模型泛化能力等技术挑战；未来发展趋势包括多尺度建模、物理信息机器学习、标准化数据共享和可解释机器学习。

### 翻译

本文系统综述了机器学习技术在聚合物材料领域的研究进展和应用前景。目前，机器学习方法在聚合物材料研究中发展迅速；尽管它们显著加速了材料预测和设计，但其复杂性也给传统领域研究者的理解和应用带来了困难。针对上述问题，本文首先分析了聚合物材料研发中的固有挑战，包括结构复杂性和传统试错方法的局限性。为解决这些问题，它重点介绍了分子描述符和特征表示、数据标准化和清洗等关键基础技术，并记录了多个高质量的聚合物数据库。随后，它详细阐述了机器学习在聚合物性能预测和材料设计中的关键作用，涵盖了传统机器学习、深度学习和迁移学习等算法的具体应用；进一步深入阐述了数据驱动设计策略，如反向设计、高通量虚拟筛选和多目标优化。本文还系统介绍了构建高可靠性机器学习模型的完整过程，总结了有效的实验验证、模型评估和优化方法。最后，它总结了当前研究中的技术挑战，如数据质量和模型泛化能力，并展望了包括多尺度建模、物理信息机器学习、标准化数据共享和可解释机器学习在内的未来发展趋势。


### 论文摘要

This paper systematically reviews the research progress and application prospects of machine learning technologies in the field of polymer materials. Currently, machine learning methods are developing rapidly in polymer material research; although they have significantly accelerated material prediction and design, their complexity has also caused difficulties in understanding and application for researchers in traditional fields. In response to the above issues, this paper first analyzes the inherent challenges in the research and development of polymer materials, including structural complexity and the limitations of traditional trial-and-error methods. To address these problems, it focuses on introducing key basic technologies such as molecular descriptors and feature representation, data standardization and cleaning, and records a number of high-quality polymer databases. Subsequently, it elaborates on the key role of machine learning in polymer property prediction and material design, covering the specific applications of algorithms such as traditional machine learning, deep learning, and transfer learning; further, it deeply expounds on data-driven design strategies, such as reverse design, high-throughput virtual screening, and multi-objective optimization. The paper also systematically introduces the complete process of constructing high-reliability machine learning models and summarizes effective experimental verification, model evaluation, and optimization methods. Finally, it summarizes the current technical challenges in research, such as data quality and model generalization ability, and looks forward to future development trends including multi-scale modeling, physics-informed machine learning, standardized data sharing, and interpretable machine learning.

---

## 24. Detecting Anomalies in Machine Learning Infrastructure via Hardware Telemetry

**论文链接:** [http://arxiv.org/abs/2510.26008v1](http://arxiv.org/abs/2510.26008v1)

**作者:** Ziji Chen, Steven Chien, Peng Qian, Noa Zilberman

**发布时间:** 2025-10-29

**备注:** 12 pages, 9 figures, submitted to nsdi 26

### GPT解析

### 总结

本文提出了一种名为System-X的系统级优化方法，它采用以硬件为中心的思路，仅依赖硬件信号而非工作负载知识进行优化，成功识别了网络和系统配置问题，加速了DeepSeek模型5.97%。

### 背景

现代机器学习已成为一个紧密结合的全栈生态系统，许多用户依赖云提供商提供弹性、隔离和成本高效的资源。然而，这些平台即服务使用虚拟化，导致运营商对用户的工作负载了解有限，阻碍了资源优化。

### 目的

论证工作负载知识对系统级优化不是必需的，并提出一种仅依赖硬件信号的优化方法。

### 方法

提出System-X系统，采用以硬件为中心的方法，仅依赖运营商完全可访问的硬件信号。通过从系统收集低级信号，使用无监督学习管道检测异常。该管道通过分析各种硬件平台上30多种流行的ML模型开发，确保能够适应新兴工作负载和未知部署模式。

### 主要发现

使用System-X成功识别了网络和系统配置问题，加速了DeepSeek模型5.97%。

### 结论

系统级优化可以通过仅依赖硬件信号来实现，无需了解具体的工作负载细节。

### 翻译

现代机器学习已发展成为一个紧密结合的全栈生态系统，结合了硬件、软件、网络和应用。许多用户依赖云提供商提供弹性、隔离和成本高效的资源。不幸的是，这些平台即服务使用虚拟化，这意味着运营商对用户的工作负载了解有限。这阻碍了运营商的资源优化，而这对确保成本效率和最小化执行时间至关重要。在本文中，我们认为工作负载知识对系统级优化不是必需的。我们提出了System-X，它采用以硬件为中心的方法，仅依赖硬件信号——这些信号完全可被运营商访问。使用从系统收集的低级信号，System-X通过无监督学习管道检测异常。该管道是通过分析各种硬件平台上30多种流行的ML模型开发的，确保了对新兴工作负载和未知部署模式的适应性。使用System-X，我们成功识别了网络和系统配置问题，将DeepSeek模型加速了5.97%。


### 论文摘要

Modern machine learning (ML) has grown into a tightly coupled, full-stack ecosystem that combines hardware, software, network, and applications. Many users rely on cloud providers for elastic, isolated, and cost-efficient resources. Unfortunately, these platforms as a service use virtualization, which means operators have little insight into the users' workloads. This hinders resource optimizations by the operator, which is essential to ensure cost efficiency and minimize execution time. In this paper, we argue that workload knowledge is unnecessary for system-level optimization. We propose System-X, which takes a \emph{hardware-centric} approach, relying only on hardware signals -- fully accessible by operators. Using low-level signals collected from the system, System-X detects anomalies through an unsupervised learning pipeline. The pipeline is developed by analyzing over 30 popular ML models on various hardware platforms, ensuring adaptability to emerging workloads and unknown deployment patterns. Using System-X, we successfully identified both network and system configuration issues, accelerating the DeepSeek model by 5.97%.

---

## 25. Unsupervised local learning based on voltage-dependent synaptic plasticity for resistive and ferroelectric synapses

**论文链接:** [http://arxiv.org/abs/2510.25787v1](http://arxiv.org/abs/2510.25787v1)

**作者:** Nikhil Garg, Ismael Balafrej, Joao Henrique Quintino Palhares, Laura Bégon-Lours, Davide Florini, Donato Francesco Falcone, Tommaso Stecconi, Valeria Bragaglia, Bert Jan Offrein, Jean-Michel Portal, Damien Querlioz, Yann Beilliard, Dominique Drouin, Fabien Alibart

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了一种基于忆阻器件的电压依赖性突触可塑性(VDSP)学习方法，用于解决边缘计算设备上AI部署的能耗问题。该方法实现了低功耗的无监督学习，在MNIST模式识别任务上取得了超过83%的准确率，并针对不同类型的忆阻器件进行了适应性调整和鲁棒性优化。

### 背景

边缘计算设备上部署人工智能面临显著的能耗和功能性挑战。这些设备需要低功耗且能够实时适应的学习机制。基于纳米尺度电阻存储器的内存计算技术可能在这些边缘设备上执行AI工作负载方面发挥关键作用。

### 目的

研究旨在开发一种高效的无监督和局部学习方法，基于赫布原理在忆阻突触中实现，使AI能够在边缘设备上低功耗运行，同时保持高性能。

### 方法

研究引入了电压依赖性突触可塑性(VDSP)作为基于赫布原理的忆阻突触无监督和局部学习方法。这种方法无需复杂脉冲整形电路即可实现在线学习，展示了如何将VDSP适应到三种具有不同开关特性的忆阻器件：TiO₂、基于HfO₂的金属氧化物丝状突触和基于HfZrO₄的铁电隧道结(FTJ)。通过系统级模拟验证了这些器件在脉冲神经网络中的无监督学习能力。

### 主要发现

1. 所有测试的忆阻器件在使用200个神经元的情况下，在基于MNIST的模式识别任务上实现了超过83%的准确率，达到最先进性能。
2. VDSP方法成功适应了三种不同类型的忆阻器件，证明了其通用性。
3. 研究评估了器件变异性(如开关阈值和高低电阻状态水平比率)对性能的影响，并提出了增强系统鲁棒性的缓解策略。

### 结论

VDSP方法为边缘计算设备上的AI部署提供了一种高效、低功耗的学习解决方案，无需复杂电路即可实现无监督学习。该方法在不同类型的忆阻器件上表现出色，并通过针对器件变异性的缓解策略增强了系统鲁棒性，为边缘AI应用提供了实用可行的技术路径。

### 翻译

在边缘计算设备上部署人工智能面临与能耗和功能相关的重大挑战。这些设备可以从大脑启发的学习机制中极大受益，允许在低功耗条件下进行实时适应。使用纳米尺度电阻存储器的内存计算可能在使这些边缘设备上执行AI工作负载方面发挥关键作用。在本研究中，我们引入了电压依赖性突触可塑性(VDSP)作为一种基于赫布原理的忆阻突触中无监督和局部学习的高效方法。这种方法无需脉冲时间依赖可塑性(STDP)通常需要的复杂脉冲整形电路即可实现在线学习。我们展示了如何将VDSP advantageous地适应到三种具有不同开关特性的忆阻器件：TiO₂、基于HfO₂的金属氧化物丝状突触和基于HfZrO₄的铁电隧道结(FTJ)。进行了包含这些器件的脉冲神经网络系统级模拟，以验证基于MNIST的模式识别任务上的无监督学习，达到了最先进的性能。结果表明所有设备在使用200个神经元的情况下都实现了超过83%的准确率。此外，我们评估了器件变异性的影响，如开关阈值和高低电阻状态水平比率，并提出了增强鲁棒性的缓解策略。


### 论文摘要

The deployment of AI on edge computing devices faces significant challenges related to energy consumption and functionality. These devices could greatly benefit from brain-inspired learning mechanisms, allowing for real-time adaptation while using low-power. In-memory computing with nanoscale resistive memories may play a crucial role in enabling the execution of AI workloads on these edge devices. In this study, we introduce voltage-dependent synaptic plasticity (VDSP) as an efficient approach for unsupervised and local learning in memristive synapses based on Hebbian principles. This method enables online learning without requiring complex pulse-shaping circuits typically necessary for spike-timing-dependent plasticity (STDP). We show how VDSP can be advantageously adapted to three types of memristive devices (TiO$_2$, HfO$_2$-based metal-oxide filamentary synapses, and HfZrO$_4$-based ferroelectric tunnel junctions (FTJ)) with disctinctive switching characteristics. System-level simulations of spiking neural networks incorporating these devices were conducted to validate unsupervised learning on MNIST-based pattern recognition tasks, achieving state-of-the-art performance. The results demonstrated over 83% accuracy across all devices using 200 neurons. Additionally, we assessed the impact of device variability, such as switching thresholds and ratios between high and low resistance state levels, and proposed mitigation strategies to enhance robustness.

---

## 26. 论文ID: 2510.23639v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.23639v2.json'

---

## 27. HEIR: Learning Graph-Based Motion Hierarchies

**论文链接:** [http://arxiv.org/abs/2510.26786v1](http://arxiv.org/abs/2510.26786v1)

**作者:** Cheng Zheng, William Koch, Baiang Li, Felix Heide

**发布时间:** 2025-10-30

**备注:** Code link: https://github.com/princeton-computational-imaging/HEIR

### GPT解析

### 总结

本文提出了一种通用的层次运动建模方法，通过图神经网络直接从数据中学习结构化的、可解释的运动关系，克服了传统方法依赖手动定义层次结构和固定运动基元的局限性，在多种运动类型上展现出优越性能。

### 背景

运动的层次结构存在于计算机视觉、图形学和机器人学等多个研究领域，复杂动力学通常源于简单运动组件之间的协调相互作用。现有方法通常依赖于手动定义的或启发式的层次结构，具有固定的运动基元，限制了它们在不同任务间的泛化能力。

### 目的

提出一种通用的层次运动建模方法，直接从数据中学习结构化的、可解释的运动关系，适用于广泛的以运动为中心的任务。

### 方法

使用基于图的层次结构表示观察到的运动，将全局绝对运动分解为父继承模式和局部运动残差；将层次推断制定为可微的图学习问题，其中顶点表示基本运动，有向边通过图神经网络捕获学习的父子依赖关系；在一维平移运动、二维旋转运动和动态三维场景变形三个示例上评估该方法。

### 主要发现

实验结果表明，该方法在一维和二维情况下成功重建了内在的运动层次结构；与基线相比，在动态3D高斯飞溅场景中产生了更真实、可解释的变形效果。

### 结论

该方法提供了一种适应性强、数据驱动的层次建模范式，适用于广泛的以运动为中心的任务，通过学习而非预设层次结构实现了更好的泛化能力和可解释性。

### 翻译

运动层次结构存在于包括计算机视觉、图形学和机器人学在内的研究领域，其中复杂动力学通常源于简单运动组件之间的协调相互作用。现有方法对这类动力学进行建模通常依赖于手动定义的或启发式的层次结构，具有固定的运动基元，限制了它们在不同任务间的泛化能力。在本工作中，我们提出了一种通用的层次运动建模方法，直接从数据中学习结构化的、可解释的运动关系。我们的方法使用基于图的层次结构来表示观察到的运动，明确地将全局绝对运动分解为父继承模式和局部运动残差。我们将层次推断制定为可微的图学习问题，其中顶点表示基本运动，有向边通过图神经网络捕获学习的父子依赖关系。我们在三个示例上评估了我们的层次重建方法：一维平移运动、二维旋转运动和通过高斯飞溅的动态三维场景变形。实验结果表明，我们的方法在一维和二维情况下重建了内在的运动层次结构，与基线相比，在动态3D高斯飞溅场景中产生了更真实、可解释的变形。通过提供一种适应性强、数据驱动的层次建模范式，我们的方法适用于广泛的以运动为中心的任务。项目页面：https://light.princeton.edu/HEIR/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决运动层次结构的自动建模问题，现有方法通常依赖手动定义的层次结构或固定的运动基元，限制了跨任务泛化能力。这个问题在计算机视觉、图形学和机器人学等多个领域都很重要，因为复杂运动往往源于简单运动组件间的协调，层次结构能帮助理解、生成、预测和控制运动，解决多尺度依赖关系和组合爆炸问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法（手动定义模板或非可解释神经模块）的局限性，注意到不同研究领域面临共同挑战，需要自适应选择合适抽象层次的方法。他们借鉴了图神经网络思想用于学习边权重和父子关系，使用Gumbel-Softmax技巧处理离散层次结构的可微分采样，还参考了层次运动表示（如骨骼定义关系）和3D场景变形（如NeMF、MovingParts）等领域的现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于图的层次结构表示运动，将全局绝对运动分解为父继承模式和局部运动残差，将层次推断转化为可微分图学习问题。流程包括：1)构建邻近有向图，顶点表示运动元素；2)通过图注意力层计算边权重；3)使用Gumbel-Softmax采样层次结构；4)沿层次结构累积相对速度重建绝对运动；5)通过重建损失和正则化项训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新包括：直接从数据学习可解释运动关系；使用图结构显式分解运动为父继承和局部残差；将层次推断转化为可微分图学习问题；适用于多种运动类型。相比之前工作，不依赖手动定义层次或固定基元，提供可解释结构，不假设特定领域或维度，在3D场景变形中产生更真实结果，具有更好泛化能力和可解释性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于图的层次运动学习方法HEIR，能够直接从数据中学习可解释的运动层次结构，有效分解复杂运动为父继承模式和局部残差，并在多种运动建模任务中展现出优越的性能。'}


### 论文摘要

Hierarchical structures of motion exist across research fields, including computer vision, graphics, and robotics, where complex dynamics typically arise from coordinated interactions among simpler motion components. Existing methods to model such dynamics typically rely on manually-defined or heuristic hierarchies with fixed motion primitives, limiting their generalizability across different tasks. In this work, we propose a general hierarchical motion modeling method that learns structured, interpretable motion relationships directly from data. Our method represents observed motions using graph-based hierarchies, explicitly decomposing global absolute motions into parent-inherited patterns and local motion residuals. We formulate hierarchy inference as a differentiable graph learning problem, where vertices represent elemental motions and directed edges capture learned parent-child dependencies through graph neural networks. We evaluate our hierarchical reconstruction approach on three examples: 1D translational motion, 2D rotational motion, and dynamic 3D scene deformation via Gaussian splatting. Experimental results show that our method reconstructs the intrinsic motion hierarchy in 1D and 2D cases, and produces more realistic and interpretable deformations compared to the baseline on dynamic 3D Gaussian splatting scenes. By providing an adaptable, data-driven hierarchical modeling paradigm, our method offers a formulation applicable to a broad range of motion-centric tasks. Project Page: https://light.princeton.edu/HEIR/

---

## 28. Graph Guided Modulo Recovery of EEG Signals

**论文链接:** [http://arxiv.org/abs/2510.26756v1](http://arxiv.org/abs/2510.26756v1)

**作者:** Soujanya Hazra, Sanjay Ghosh

**发布时间:** 2025-10-30

**备注:** 5 pages, 1 figure, and 2 tables

### GPT解析

### 总结

该研究提出了一种基于图神经网络的GraphUnwrapNet方法，用于解决脑电图(EEG)信号模数采样恢复问题。通过将EEG信号表示为有组织的图结构，并引入预估计引导的特征注入模块，有效提升了在信号折叠边界处的恢复稳定性。实验结果表明，该方法优于传统优化技术，并与当前深度学习模型具有竞争性。

### 背景

脑电图(EEG)在不同人之间常表现出显著变异性，这种波动会干扰可靠信号采集并可能导致信号失真或削波。

### 目的

开发一种有效的方法从模数采样的折叠观测中恢复原始EEG信号，解决这一高度不适定问题。

### 方法

提出GraphUnwrapNet，一种基于图神经网络的方法，将EEG信号表示为有组织的图结构，并引入预估计引导的特征注入模块，提供粗略的折叠指示器以增强恢复稳定性。

### 主要发现

在STEW数据集上的实验表明，与传统优化技术相比有持续提升，与当前深度学习模型相比具有竞争性的准确性。

### 结论

基于图的方法在鲁棒模数EEG恢复方面具有显著潜力。

### 翻译

脑电图(EEG)通常表现出显著的个体间变异性。这种波动会干扰可靠的信号采集并可能导致失真或削波。模数采样现在是解决这个问题的有前途的方法，通过折叠信号而不是使它们饱和。从折叠观测中恢复原始波形是一个高度不适定的问题。在本工作中，我们提出了一种基于图神经网络的方法，称为GraphUnwrapNet，用于EEG信号的模数恢复。我们的核心思想是将EEG信号表示为一个有组织的图，其通道和时间连接建立了潜在的相互依赖关系。我们的一个关键贡献是引入了一个预估计引导的特征注入模块，提供粗略的折叠指示器，增强在折叠边界处的恢复稳定性。这种设计将结构信息与折叠先验集成到一个统一的框架中。我们在同时任务脑电图工作负荷(STEW)数据集上进行了全面的实验。结果表明与传统优化技术相比有持续的提升，与当前深度学习模型相比具有竞争性的准确性。我们的发现强调了基于图的方法在鲁棒模数EEG恢复方面的潜力。


### 论文摘要

Electroencephalography (EEG) often shows significant variability among people. This fluctuation disrupts reliable acquisition and may result in distortion or clipping. Modulo sampling is now a promising solution to this problem, by folding signals instead of saturating them. Recovery of the original waveform from folded observations is a highly ill-posed problem. In this work, we propose a method based on a graph neural network, referred to as GraphUnwrapNet, for the modulo recovery of EEG signals. Our core idea is to represent an EEG signal as an organized graph whose channels and temporal connections establish underlying interdependence. One of our key contributions is in introducing a pre-estimation guided feature injection module to provide coarse folding indicators that enhance stability during recovery at wrap boundaries. This design integrates structural information with folding priors into an integrated framework. We performed comprehensive experiments on the Simultaneous Task EEG Workload (STEW) dataset. The results demonstrate consistent enhancements over traditional optimization techniques and competitive accuracy relative to current deep learning models. Our findings emphasize the potential of graph-based methodology for robust modulo EEG recovery.

---

## 29. Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras

**论文链接:** [http://arxiv.org/abs/2510.26614v1](http://arxiv.org/abs/2510.26614v1)

**作者:** Christoffer Koo Øhrstrøm, Ronja Güldenring, Lazaros Nalpantidis

**发布时间:** 2025-10-30

### GPT解析

### 总结

论文提出了一种针对事件相机的tokenization方法，称为Spiking Patches，能够保留事件流的异步性和空间稀疏性特性，同时保持高准确性，并且推理速度比传统方法更快。

### 背景

现有的事件表示方法（如帧或体素）虽然是同步的且降低了空间稀疏性，但能产生高准确性。

### 目的

发现一种能够保留事件相机独特属性的事件表示方法。

### 方法

提出Spiking Patches tokenizer，专门为事件相机设计，能够保留事件流的异步性和空间稀疏性。

### 主要发现

使用GNN、PCN和Transformer在手势识别和物体检测任务上评估，Spiking Patches的token比基于体素的token推理速度快3.4倍，比基于帧的token推理速度快10.4倍，在保持相同准确性的同时，在某些情况下甚至超越它们，手势识别绝对改进最高达3.8，物体检测绝对改进最高达1.4。

### 结论

tokenization是事件视觉领域的新方向，标志着保留事件相机属性方法的发展。

### 翻译

我们提出事件的tokenization并展示了一个tokenizer，Spiking Patches，专门为事件相机设计。给定异步和空间稀疏的事件流，我们的目标是发现保留这些属性的事件表示。先前的工作将事件表示为帧或体素。然而，虽然这些表示能产生高准确性，但帧和体素都是同步的，降低了空间稀疏性。Spiking Patches提供了保留事件相机独特属性的方法，我们在实验中证明这不会牺牲准确性。我们使用GNN、PCN和Transformer在手势识别和物体检测任务上评估我们的tokenizer。来自Spiking Patches的token比基于体素的token推理速度快3.4倍，比基于帧的token推理速度快10.4倍。我们在保持相同准确性的同时实现了这一点，在某些情况下甚至超越它们，手势识别绝对改进最高达3.8，物体检测绝对改进最高达1.4。因此，tokenization构成了事件视觉领域的一个新方向，标志着保留事件相机属性方法的发展。


### 论文摘要

We propose tokenization of events and present a tokenizer, Spiking Patches, specifically designed for event cameras. Given a stream of asynchronous and spatially sparse events, our goal is to discover an event representation that preserves these properties. Prior works have represented events as frames or as voxels. However, while these representations yield high accuracy, both frames and voxels are synchronous and decrease the spatial sparsity. Spiking Patches gives the means to preserve the unique properties of event cameras and we show in our experiments that this comes without sacrificing accuracy. We evaluate our tokenizer using a GNN, PCN, and a Transformer on gesture recognition and object detection. Tokens from Spiking Patches yield inference times that are up to 3.4x faster than voxel-based tokens and up to 10.4x faster than frames. We achieve this while matching their accuracy and even surpassing in some cases with absolute improvements up to 3.8 for gesture recognition and up to 1.4 for object detection. Thus, tokenization constitutes a novel direction in event-based vision and marks a step towards methods that preserve the properties of event cameras.

---

## 30. UnifiedFL: A Dynamic Unified Learning Framework for Equitable Federation

**论文链接:** [http://arxiv.org/abs/2510.26350v1](http://arxiv.org/abs/2510.26350v1)

**作者:** Furkan Pala, Islem Rekik

**发布时间:** 2025-10-30

### GPT解析

### 总结

论文提出UnifiedFL，一种动态联邦学习框架，用于处理具有不同神经网络架构和非相同分布数据的客户端之间的协作训练，通过图神经网络优化异构本地网络，实验证明在多个基准测试中表现优越。

### 背景

联邦学习作为关键范式，允许多个客户端在不共享原始数据的情况下协作训练模型，支持隐私保护应用。然而，关于具有不同神经网络架构和非相同分布数据集的客户端之间的协作训练研究仍然很少。

### 目的

解决现有联邦学习框架在支持根本不同架构客户端、处理数据统计异质性和领域断裂问题上的局限性，提高模型在不同测试域上的泛化能力。

### 方法

提出UnifiedFL框架，将异构本地网络表示为有向模型图中的节点和边，通过共享图神经网络优化。引入通用GNN参数化所有架构、基于客户端参数之间欧几里得距离的距离驱动聚类，以及平衡收敛性和多样性的两层聚合策略。

### 主要发现

现有联邦学习方法只能支持单一模型家族内的变体，假设共享全局架构，无法适应不同网络类型；现有方法通常只处理统计异质性，忽略领域断裂问题；当客户端使用不同架构、具有非相同分布数据并遇到不同测试域时，当前方法表现不佳。

### 结论

UnifiedFL在MedMNIST分类和海马体分割基准测试中表现出优越性能，代码和数据可在https://github.com/basiralab/UnifiedFL获取。

### 翻译

联邦学习（FL）已成为一种关键范式，使多个客户端能够在不共享原始数据的情况下协作训练模型，在放射学和病理学等领域支持隐私保护应用。然而，关于具有根本不同神经网络架构和非相同分布数据集的客户端之间的协作训练研究仍然很少。现有的联邦学习框架面临几个局限性。尽管声称支持架构异构性，但大多数联邦学习方法只容忍单一模型家族内的变体（例如，更浅、更深或更宽的CNN），仍然假设共享全局架构，无法适应客户端部署不同网络类型（例如，CNN、GNN、MLP）的联邦。此外，现有方法通常只处理统计异质性，而忽略了领域断裂问题，即每个客户端的数据分布与测试时面临的数据分布明显不同，从而削弱了模型的泛化能力。当客户端使用不同架构、具有非相同分布数据并遇到不同的测试域时，当前方法表现不佳。为解决这些挑战，我们提出UnifiedFL，一种动态联邦学习框架，将异构本地网络表示为有向模型图中的节点和边，并通过共享的图神经网络（GNN）进行优化。UnifiedFL引入了（i）通用GNN参数化所有架构，（ii）基于客户端参数之间欧几里得距离的距离驱动聚类，以及（iii）平衡收敛性和多样性的两层聚合策略。在MedMNIST分类和海马体分割基准测试中进行的实验证明了UnifiedFL的优越性能。代码和数据：https://github.com/basiralab/UnifiedFL


### 论文摘要

Federated learning (FL) has emerged as a key paradigm for collaborative model training across multiple clients without sharing raw data, enabling privacy-preserving applications in areas such as radiology and pathology. However, works on collaborative training across clients with fundamentally different neural architectures and non-identically distributed datasets remain scarce. Existing FL frameworks face several limitations. Despite claiming to support architectural heterogeneity, most recent FL methods only tolerate variants within a single model family (e.g., shallower, deeper, or wider CNNs), still presuming a shared global architecture and failing to accommodate federations where clients deploy fundamentally different network types (e.g., CNNs, GNNs, MLPs). Moreover, existing approaches often address only statistical heterogeneity while overlooking the domain-fracture problem, where each client's data distribution differs markedly from that faced at testing time, undermining model generalizability. When clients use different architectures, have non-identically distributed data, and encounter distinct test domains, current methods perform poorly. To address these challenges, we propose UnifiedFL, a dynamic federated learning framework that represents heterogeneous local networks as nodes and edges in a directed model graph optimized by a shared graph neural network (GNN). UnifiedFL introduces (i) a common GNN to parameterize all architectures, (ii) distance-driven clustering via Euclidean distances between clients' parameters, and (iii) a two-tier aggregation policy balancing convergence and diversity. Experiments on MedMNIST classification and hippocampus segmentation benchmarks demonstrate UnifiedFL's superior performance. Code and data: https://github.com/basiralab/UnifiedFL

---

## 31. From Embedding to Control: Representations for Stochastic Multi-Object Systems

**论文链接:** [http://arxiv.org/abs/2510.26344v1](http://arxiv.org/abs/2510.26344v1)

**作者:** Xiaoyuan Cheng, Yiming Yang, Wei Jiang, Chenyang Yuan, Zhuo Sun, Yukun Hu

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出图可控嵌入（GCE）框架，用于学习线性控制下的随机多对象动力学系统。该框架基于希尔伯特空间嵌入，将受控随机动力学的概率分布嵌入到再生核希尔伯特空间中，保留非线性表达能力的同时支持线性操作。GCE采用平均场近似技术捕获对象间依赖关系，并通过图神经网络构建适应动态交互模式的核特征，能够推广到未见过的拓扑结构。

### 背景

具有多个相互作用的随机非线性动力学系统的精确建模和控制是一个具有挑战性的问题。非均匀交互和随机拓扑结构使得这一任务更加困难，需要开发新的方法来处理这些复杂情况。

### 目的

研究如何实现具有多个相互作用的随机非线性动力学系统的精确建模和有效控制，特别是应对非均匀交互和随机拓扑带来的挑战。

### 方法

提出了图可控嵌入（GCE）框架，这是一种基于希尔伯特空间嵌入的通用方法。GCE将受控随机动力学的概率分布直接嵌入到再生核希尔伯特空间（RKHS）中，允许在保留非线性表达能力的同时进行线性操作。该方法采用平均场近似技术来捕获对象间依赖关系，并通过整合图神经网络构建数据相关的核特征，使其能够适应动态交互模式并推广到未见过的拓扑结构。

### 主要发现

1. GCE提供了关于存在性、收敛性和适用性的理论保证；2. 平均场近似技术能够有效捕获对象间依赖关系，实现低样本复杂度；3. 构建的核特征能够适应动态交互模式，仅用有限训练实例即可推广到未见拓扑；4. GCE可无缝扩展到不同大小和拓扑的多对象系统；5. 希尔伯特空间的线性支持简单而有效的控制算法来合成最优序列。

### 结论

图可控嵌入（GCE）框架为随机多对象动力学系统的建模和控制提供了一种有效方法。通过结合希尔伯特空间嵌入、平均场近似和图神经网络，GCE能够处理非均匀交互和随机拓扑带来的挑战，并在物理系统、机器人和电力系统实验中展现出优越性能，特别是在分布内和少样本测试中优于其他嵌入方法。

### 翻译

本文研究了如何在具有多个相互作用的随机非线性动力学系统中实现精确建模和有效控制。然而，非均匀交互和随机拓扑使这一任务具有挑战性。我们通过提出图可控嵌入（GCE）来解决这些挑战，这是一个用于学习线性控制下随机多对象动力学的通用框架。具体来说，GCE建立在希尔伯特空间嵌入的基础上，允许将受控随机动力学的概率分布直接嵌入到再生核希尔伯特空间（RKHS）中，这使其RKHS中的线性操作能够保留非线性表达能力。我们提供了关于GCE的存在性、收敛性和适用性的理论保证。值得注意的是，采用平均场近似技术来有效捕获对象间依赖关系，并实现可证明的低样本复杂度。通过整合图神经网络，我们构建了能够适应动态交互模式的数据相关核特征，并且仅用有限的训练实例就能推广到未见过的拓扑结构。GCE可以无缝扩展到不同大小和拓扑的多对象系统。利用希尔伯特空间的线性，GCE还支持简单而有效的控制算法来合成最优序列。在物理系统、机器人和电力系统上的实验验证了GCE的有效性，并在分布内和少样本测试中，相比各种有竞争力的嵌入方法都展现出一致的性能提升。


### 论文摘要

This paper studies how to achieve accurate modeling and effective control in stochastic nonlinear dynamics with multiple interacting objects. However, non-uniform interactions and random topologies make this task challenging. We address these challenges by proposing \textit{Graph Controllable Embeddings} (GCE), a general framework to learn stochastic multi-object dynamics for linear control. Specifically, GCE is built on Hilbert space embeddings, allowing direct embedding of probability distributions of controlled stochastic dynamics into a reproducing kernel Hilbert space (RKHS), which enables linear operations in its RKHS while retaining nonlinear expressiveness. We provide theoretical guarantees on the existence, convergence, and applicability of GCE. Notably, a mean field approximation technique is adopted to efficiently capture inter-object dependencies and achieve provably low sample complexity. By integrating graph neural networks, we construct data-dependent kernel features that are capable of adapting to dynamic interaction patterns and generalizing to even unseen topologies with only limited training instances. GCE scales seamlessly to multi-object systems of varying sizes and topologies. Leveraging the linearity of Hilbert spaces, GCE also supports simple yet effective control algorithms for synthesizing optimal sequences. Experiments on physical systems, robotics, and power grids validate GCE and demonstrate consistent performance improvement over various competitive embedding methods in both in-distribution and few-shot tests

---

## 32. A Survey of Heterogeneous Graph Neural Networks for Cybersecurity Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.26307v1](http://arxiv.org/abs/2510.26307v1)

**作者:** Laura Jiang, Reza Ryan, Qian Li, Nasim Ferdosian

**发布时间:** 2025-10-30

**备注:** 37 pages, 4 figures, 86 references. Submitted to Journal of Computer  Security (under review)

### GPT解析

### 总结

这篇论文提供了对网络安全中基于异构图神经网络(HGNN)的异常检测方法的全面综述，建立了分类法，分析了代表性模型，回顾了基准数据集和评估指标，并确定了未来研究方向。

### 背景

异常检测在网络安全中是关键任务，需要识别内部威胁、访问违规和协调攻击。基于图的方法在建模实体交互方面变得越来越重要，但大多数依赖于同质和静态结构，这限制了它们捕捉现实环境中异构性和时间演化的能力。

### 目的

解决基于HGNN的异常检测研究分散、缺乏比较评估和标准化基准的问题，为该领域建立结构化的基础。

### 方法

引入按异常类型和图动力学对方法进行分类的分类法，分析代表性模型并将其映射到关键网络安全应用，回顾常用基准数据集和评估指标，强调其优缺点。

### 主要发现

确定了与建模、数据和部署相关的主要开放挑战，概述了未来研究的有希望的方向。

### 结论

该综述旨在为推进基于HGNN的异常检测建立结构化的基础，使其可扩展、可解释且实际可部署。

### 翻译

异常检测是网络安全中的关键任务，其中识别内部威胁、访问违规和协调攻击对于确保系统弹性至关重要。基于图的方法在建模实体交互方面变得越来越重要，但大多数依赖于同质和静态结构，这限制了它们捕捉现实环境中异构性和时间演化的能力。异构图神经网络已成为一种有前景的异常检测范式，通过整合类型感知变换和关系敏感聚合，能够对复杂的网络数据进行更具表现力的建模。然而，当前关于基于HGNN的异常检测研究仍然分散，建模策略多样，比较评估有限，且缺乏标准化基准。为了解决这一差距，我们对网络安全中基于HGNN的异常检测方法进行了全面综述。我们引入了一个按异常类型和图动力学对方法进行分类的分类法，分析了代表性模型，并将它们映射到关键的网络安全应用。我们还回顾了常用的基准数据集和评估指标，强调了它们的优缺点。最后，我们确定了与建模、数据和部署相关的主要开放挑战，并概述了未来研究的有希望的方向。本综述旨在为推进基于HGNN的异常检测建立结构化的基础，朝着可扩展、可解释和实际可部署的解决方案发展。


### 论文摘要

Anomaly detection is a critical task in cybersecurity, where identifying insider threats, access violations, and coordinated attacks is essential for ensuring system resilience. Graph-based approaches have become increasingly important for modeling entity interactions, yet most rely on homogeneous and static structures, which limits their ability to capture the heterogeneity and temporal evolution of real-world environments. Heterogeneous Graph Neural Networks (HGNNs) have emerged as a promising paradigm for anomaly detection by incorporating type-aware transformations and relation-sensitive aggregation, enabling more expressive modeling of complex cyber data. However, current research on HGNN-based anomaly detection remains fragmented, with diverse modeling strategies, limited comparative evaluation, and an absence of standardized benchmarks. To address this gap, we provide a comprehensive survey of HGNN-based anomaly detection methods in cybersecurity. We introduce a taxonomy that classifies approaches by anomaly type and graph dynamics, analyze representative models, and map them to key cybersecurity applications. We also review commonly used benchmark datasets and evaluation metrics, highlighting their strengths and limitations. Finally, we identify key open challenges related to modeling, data, and deployment, and outline promising directions for future research. This survey aims to establish a structured foundation for advancing HGNN-based anomaly detection toward scalable, interpretable, and practically deployable solutions.

---

## 33. Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion

**论文链接:** [http://arxiv.org/abs/2510.26067v1](http://arxiv.org/abs/2510.26067v1)

**作者:** Chi Zhang, Mingrui Li, Wenzhe Tong, Xiaonan Huang

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出了一种形态感知的强化学习框架，通过将图神经网络集成到Soft Actor-Critic算法中，解决了张力完整性机器人的运动控制问题。

### 背景

张力完整性机器人结合刚性杆和弹性缆索，具有高弹性和可部署性，但因其欠驱动和高度耦合的动力学特性，在运动控制方面面临重大挑战。

### 目的

开发一种强化学习方法，利用机器人的结构先验知识，提高张力完整性机器人的运动控制性能。

### 方法

将机器人的物理拓扑表示为图，使用基于图神经网络(GNN)的策略捕捉组件间的耦合关系，并将其集成到Soft Actor-Critic (SAC)算法中。

### 主要发现

该方法在物理三杆张力完整性机器人上得到验证，在直线跟踪和双向转向等运动任务中表现出优异的样本效率、对噪声和刚度变化的鲁棒性以及改进的轨迹精度；学习到的策略可以直接从模拟转移到硬件，无需微调。

### 结论

将结构先验知识整合到强化学习中对于张力完整性机器人控制具有显著优势，能够实现更高效、更稳定的控制策略。

### 翻译

张力完整性机器人结合刚性杆和弹性缆索，提供高弹性和可部署性，但由于其欠驱动和高度耦合的动力学特性，给运动控制带来了重大挑战。本文引入了一种形态感知的强化学习框架，将图神经网络(GNN)集成到Soft Actor-Critic (SAC)算法中。通过将机器人的物理拓扑表示为图，所提出的基于GNN的策略捕捉了组件之间的耦合关系，实现了比传统多层感知器(MLP)策略更快且更稳定的学习。该方法在物理三杆张力完整性机器人上得到了验证，包括直线跟踪和双向转向在内的三种运动原语。它显示出优异的样本效率、对噪声和刚度变化的鲁棒性，以及改进的轨迹精度。值得注意的是，学习到的策略可以直接从模拟转移到硬件而无需微调，实现了稳定的真实世界运动。这些结果表明，将结构先验知识整合到强化学习中对于张力完整性机器人控制具有优势。


### 论文摘要

Tensegrity robots combine rigid rods and elastic cables, offering high resilience and deployability but posing major challenges for locomotion control due to their underactuated and highly coupled dynamics. This paper introduces a morphology-aware reinforcement learning framework that integrates a graph neural network (GNN) into the Soft Actor-Critic (SAC) algorithm. By representing the robot's physical topology as a graph, the proposed GNN-based policy captures coupling among components, enabling faster and more stable learning than conventional multilayer perceptron (MLP) policies. The method is validated on a physical 3-bar tensegrity robot across three locomotion primitives, including straight-line tracking and bidirectional turning. It shows superior sample efficiency, robustness to noise and stiffness variations, and improved trajectory accuracy. Notably, the learned policies transfer directly from simulation to hardware without fine-tuning, achieving stable real-world locomotion. These results demonstrate the advantages of incorporating structural priors into reinforcement learning for tensegrity robot control.

---

## 34. Data-driven Projection Generation for Efficiently Solving Heterogeneous Quadratic Programming Problems

**论文链接:** [http://arxiv.org/abs/2510.26061v1](http://arxiv.org/abs/2510.26061v1)

**作者:** Tomoharu Iwata, Futoshi Futami

**发布时间:** 2025-10-30

### GPT解析

### 总结

提出一种数据驱动的框架，通过针对特定实例的投影减少高维二次规划问题的变量数量，使用图神经网络生成定制化投影，高效解决二次规划问题。

### 背景

二次规划问题在高维情况下求解复杂度高，传统方法面临计算挑战。

### 目的

开发一种能够高效解决高维二次规划问题的方法，通过减少变量数量降低计算复杂度，同时保证解的质量。

### 方法

设计基于图神经网络的模型生成定制化投影；使用双层优化训练模型，内层优化在给定投影下解决QP问题，外层优化更新模型参数；开发高效算法计算参数梯度，无需通过求解器反向传播；提供神经网络生成投影矩阵解决QP问题的泛化能力理论分析。

### 主要发现

方法能产生高质量可行解并减少计算时间；即使对未见过的QP问题也能生成高质量解决方案；实验结果优于现有方法。

### 结论

所提出的数据驱动框架通过针对特定实例的投影和图神经网络模型，能高效解决高维二次规划问题，在保证解质量的同时显著减少计算时间。

### 翻译

我们提出了一种数据驱动的框架，通过使用针对特定实例的投影来减少高维二次规划问题中的变量数量，从而高效解决二次规划问题。我们设计了一个基于图神经网络的模型，为每个二次规划实例生成定制化投影，使我们能够即使对于未见过的也能产生高质量解。该模型在异构QP上进行训练，以最小化在投影解上评估的期望目标值。这被表述为一个双层优化问题；内层优化在给定投影下使用QP求解器解决QP问题，而外层优化更新模型参数。我们开发了一种高效算法来解决这个双层优化问题，计算参数梯度时无需通过求解器进行反向传播。我们提供了使用神经网络生成的投影矩阵解决QP问题的泛化能力理论分析。实验结果表明，我们的方法产生了高质量可行解并减少了计算时间，优于现有方法。


### 论文摘要

We propose a data-driven framework for efficiently solving quadratic programming (QP) problems by reducing the number of variables in high-dimensional QPs using instance-specific projection. A graph neural network-based model is designed to generate projections tailored to each QP instance, enabling us to produce high-quality solutions even for previously unseen problems. The model is trained on heterogeneous QPs to minimize the expected objective value evaluated on the projected solutions. This is formulated as a bilevel optimization problem; the inner optimization solves the QP under a given projection using a QP solver, while the outer optimization updates the model parameters. We develop an efficient algorithm to solve this bilevel optimization problem, which computes parameter gradients without backpropagating through the solver. We provide a theoretical analysis of the generalization ability of solving QPs with projection matrices generated by neural networks. Experimental results demonstrate that our method produces high-quality feasible solutions with reduced computation time, outperforming existing methods.

---

## 35. Robust GNN Watermarking via Implicit Perception of Topological Invariants

**论文链接:** [http://arxiv.org/abs/2510.25934v1](http://arxiv.org/abs/2510.25934v1)

**作者:** Jipeng Li, Yannning Shen

**发布时间:** 2025-10-29

### GPT解析

### 总结

论文提出了一种名为InvGNN-WM的图神经网络水印技术，它不依赖后门触发器，而是将所有权与模型对图不变性的隐式感知联系起来，实现了黑盒验证且对任务影响微小的水印方法。

### 背景

图神经网络是有价值的知识产权，但现有水印技术大多依赖后门触发器，这些触发器在常见模型编辑下会被破坏，导致所有权模糊。

### 目的

开发一种无需触发器、支持黑盒验证且对任务影响微小的图神经网络水印技术，以解决现有水印技术的局限性。

### 方法

使用轻量级头在所有者私有的载体集上预测归一化代数连通性；使用敏感解码器输出比特；使用校准阈值控制误报率。

### 主要发现

在多种节点和图分类数据集和主干网络上，InvGNN-WM保持了清洁准确率，同时比基线方法产生更高的水印准确率；该方法在非结构化剪枝、微调和后训练量化条件下保持强健性；纯知识蒸馏会削弱水印，而带有水印损失的知识蒸馏可以恢复水印。

### 结论

InvGNN-WM提供了不可感知性和鲁棒性的保证，精确移除该水印被证明是NP完全问题。

### 翻译

图神经网络(GNNs)是有价值的知识产权，但许多水印依赖于后门触发器，这些触发器在常见的模型编辑下会被破坏并导致所有权模糊。我们提出了InvGNN-WM，它将所有权与模型对图不变性的隐式感知联系起来，实现了无需触发器、黑盒验证且对任务影响微小的水印。轻量级头在所有者私有的载体集上预测归一化代数连通性；敏感解码器输出比特，校准阈值控制误报率。在多样化的节点和图分类数据集及主干网络上，InvGNN-WM匹配清洁准确率，同时比基于触发器和压缩的基线产生更高的水印准确率。它在非结构化剪枝、微调和后训练量化下保持强健性；普通知识蒸馏(KD)会削弱水印，而带有水印损失的KD(KD+WM)可恢复它。我们提供了不可感知性和鲁棒性的保证，并证明精确移除是NP完全的。


### 论文摘要

Graph Neural Networks (GNNs) are valuable intellectual property, yet many watermarks rely on backdoor triggers that break under common model edits and create ownership ambiguity. We present InvGNN-WM, which ties ownership to a model's implicit perception of a graph invariant, enabling trigger-free, black-box verification with negligible task impact. A lightweight head predicts normalized algebraic connectivity on an owner-private carrier set; a sign-sensitive decoder outputs bits, and a calibrated threshold controls the false-positive rate. Across diverse node and graph classification datasets and backbones, InvGNN-WM matches clean accuracy while yielding higher watermark accuracy than trigger- and compression-based baselines. It remains strong under unstructured pruning, fine-tuning, and post-training quantization; plain knowledge distillation (KD) weakens the mark, while KD with a watermark loss (KD+WM) restores it. We provide guarantees for imperceptibility and robustness, and we prove that exact removal is NP-complete.

---

## 36. Attention Augmented GNN RNN-Attention Models for Advanced Cybersecurity Intrusion Detection

**论文链接:** [http://arxiv.org/abs/2510.25802v1](http://arxiv.org/abs/2510.25802v1)

**作者:** Jayant Biradar, Smit Shah, Tanmay Naik

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了一种新型混合深度学习架构，结合图神经网络、循环神经网络和多头注意力机制，显著提升了网络安全入侵检测能力。

### 背景

现代网络安全环境需要实时入侵检测系统，且需要将计算资源集中在高影响安全事件上。UNSW-NB15数据集包含多样的网络流量模式，为研究提供了基础。

### 目的

开发一种能够有效捕获网络流量中的空间依赖性和时间动态性的入侵检测系统，提高检测复杂攻击模式的能力。

### 方法

提出混合深度学习架构，结合图神经网络(GNNs)捕获空间依赖性、循环神经网络(RNNs)进行序列分析，以及多头注意力机制提高模型可解释性和特征选择。

### 主要发现

实验证明，与传统机器学习方法和独立深度学习模型相比，该混合模型在准确率、精确率、召回率和F1分数等多个评估指标上表现更优。特别是在检测高级持续性威胁(APTs)、分布式拒绝服务(DDoS)攻击和零日漏洞等复杂攻击模式方面表现出色。

### 结论

该混合模型是复杂网络环境中下一代网络安全应用的有前景解决方案。

### 翻译

在本文中，我们提出了一种新型混合深度学习架构，协同结合图神经网络(GNNs)、循环神经网络(RNNs)和多头注意力机制，显著提升了网络安全入侵检测能力。通过利用包含多样化网络流量模式的UNSW-NB15综合数据集，我们的方法有效地通过图结构关系捕获空间依赖性，并通过网络事件的序列分析捕获时间动态性。集成的注意力机制提供了提高模型可解释性和增强特征选择的双重好处，使网络安全分析师能够将计算资源集中在高影响安全事件上——这是现代实时入侵检测系统的关键要求。我们广泛的实验评估表明，与传统机器学习方法和独立的深度学习模型相比，所提出的混合模型在多个评估指标上实现了优越的性能，包括准确率、精确率、召回率和F1分数。该模型在检测高级持续性威胁(APTs)、分布式拒绝服务(DDoS)攻击和零日漏洞等复杂攻击模式方面表现出特别强的性能，使其成为复杂网络环境中下一代网络安全应用的有前景解决方案。


### 论文摘要

In this paper, we propose a novel hybrid deep learning architecture that synergistically combines Graph Neural Networks (GNNs), Recurrent Neural Networks (RNNs), and multi-head attention mechanisms to significantly enhance cybersecurity intrusion detection capabilities. By leveraging the comprehensive UNSW-NB15 dataset containing diverse network traffic patterns, our approach effectively captures both spatial dependencies through graph structural relationships and temporal dynamics through sequential analysis of network events. The integrated attention mechanism provides dual benefits of improved model interpretability and enhanced feature selection, enabling cybersecurity analysts to focus computational resources on high-impact security events -- a critical requirement in modern real-time intrusion detection systems. Our extensive experimental evaluation demonstrates that the proposed hybrid model achieves superior performance compared to traditional machine learning approaches and standalone deep learning models across multiple evaluation metrics, including accuracy, precision, recall, and F1-score. The model achieves particularly strong performance in detecting sophisticated attack patterns such as Advanced Persistent Threats (APTs), Distributed Denial of Service (DDoS) attacks, and zero-day exploits, making it a promising solution for next-generation cybersecurity applications in complex network environments.

---

## 37. SHA-256 Infused Embedding-Driven Generative Modeling of High-Energy Molecules in Low-Data Regimes

**论文链接:** [http://arxiv.org/abs/2510.25788v1](http://arxiv.org/abs/2510.25788v1)

**作者:** Siddharth Verma, Alankar Alankar

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了一种结合LSTM网络和注意力GNN的新方法用于高能分子生成和属性预测，通过创新的嵌入空间构建策略实现了67.5%的有效性和37.5%的新颖性，成功识别出37种新型超爆炸物。

### 背景

高能材料在推进和防御领域至关重要，但其发现受限于实验数据和测试设施的有限获取。

### 目的

开发一种新的方法来发现高能分子，特别是高能爆炸材料。

### 方法

结合长短期记忆网络(LSTM)进行分子生成，使用注意力图神经网络(GNN)进行属性预测，提出了一种创新的嵌入空间构建策略，整合固定的SHA-256嵌入和部分可训练表示，在学习开始前重塑分子输入空间，不依赖预训练。

### 主要发现

生成器达到67.5%的有效性和37.5%的新颖性；生成的库相对于训练集的平均Tanimoto系数为0.214，表明框架能够生成多样化的化学空间；识别出37种新型超爆炸物，预测爆速超过9公里/秒。

### 结论

这种新方法能够有效发现新型高能材料，特别是超爆炸物，为高能材料的研究提供了新的途径。

### 翻译

高能材料(HEMs)对于推进和防御领域至关重要，但其发现受限于实验数据和测试设施的有限获取。这项工作通过结合用于分子生成的长短期记忆网络(LSTM)和用于属性预测的注意力图神经网络(GNN)，提出了一种针对高能分子的新方法。我们提出了一种变革性的嵌入空间构建策略，整合固定的SHA-256嵌入和部分可训练表示。与传统的正则化技术不同，这改变了表示基础本身，在学习开始前重塑了分子输入空间。无需依赖预训练，生成器实现了67.5%的有效性和37.5%的新颖性。生成的库相对于训练集的平均Tanimoto系数为0.214，表明该框架能够生成多样化的化学空间。我们识别出37种新型超爆炸物，预测爆速超过9公里/秒。


### 论文摘要

High-energy materials (HEMs) are critical for propulsion and defense domains, yet their discovery remains constrained by experimental data and restricted access to testing facilities. This work presents a novel approach toward high-energy molecules by combining Long Short-Term Memory (LSTM) networks for molecular generation and Attentive Graph Neural Networks (GNN) for property predictions. We propose a transformative embedding space construction strategy that integrates fixed SHA-256 embeddings with partially trainable representations. Unlike conventional regularization techniques, this changes the representational basis itself, reshaping the molecular input space before learning begins. Without recourse to pretraining, the generator achieves 67.5% validity and 37.5% novelty. The generated library exhibits a mean Tanimoto coefficient of 0.214 relative to training set signifying the ability of framework to generate a diverse chemical space. We identified 37 new super explosives higher than 9 km/s predicted detonation velocity.

---

