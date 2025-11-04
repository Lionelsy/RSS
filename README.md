# 今日论文推荐 - 2025-11-04

共 4 篇论文

---

## 1. Hybrid-Task Meta-Learning: A GNN Approach for Scalable and Transferable Bandwidth Allocation

**论文链接:** [http://arxiv.org/abs/2401.10253v3](http://arxiv.org/abs/2401.10253v3)

**作者:** Xin Hao, Changyang She, Phee Lep Yeoh, Yuhong Liu, Branka Vucetic, Yonghui Li

**发布时间:** 2023-12-23

### GPT解析

### 总结

本文提出了一种基于深度学习的带宽分配策略，具有可扩展性和可转移性特点。通过使用图神经网络和混合任务元学习算法，实现了在不同通信场景下的高效带宽分配。

### 背景

随着用户数量和通信场景的多样化，传统的带宽分配方法面临可扩展性和泛化能力不足的挑战。

### 目的

开发一种既可随用户数量扩展，又能适应不同通信场景（如非平稳无线信道、不同服务质量要求和动态可用资源）的带宽分配策略。

### 方法

1. 使用图神经网络(GNN)表示带宽分配策略，确保参数数量不随用户数量变化；2. 开发混合任务元学习(HML)算法，在元训练阶段使用不同通信场景训练GNN初始参数；3. 在元测试阶段使用少量样本对未见过的通信场景进行微调。

### 主要发现

1. HML方法比现有基准提高初始性能8.79%，样本效率提高73%；2. 微调后的GNN策略以更低推理复杂度获得接近最优策略的奖励；3. HML比最优迭代算法减少约200到2000倍的计算时间。

### 结论

基于GNN和HML的带宽分配策略在性能、效率和计算复杂度方面均优于现有方法，为实际通信系统中的资源分配提供了有效解决方案。

### 翻译

在本文中，我们开发了一种基于深度学习的带宽分配策略，该策略：1)随用户数量可扩展；2)可转移到不同的通信场景，如非平稳无线信道、不同的服务质量要求和动态可用资源。为了支持可扩展性，带宽分配策略由图神经网络(GNN)表示，其训练参数数量不随用户数量变化。为了实现GNN的泛化能力，我们开发了一种混合任务元学习(HML)算法，在元训练期间使用不同的通信场景训练GNN的初始参数。接下来，在元测试期间，使用少量样本对未见过的通信场景微调GNN。仿真结果表明，与现有基准相比，我们的HML方法可以提高初始性能8.79%，样本效率提高73%。微调后，我们的次优GNN策略与使用迭代优化获得的最优策略相比，可以以低得多的推理复杂度获得几乎相同的奖励。数值结果验证，与最优迭代算法相比，我们的HML可以减少约200到2000倍的计算时间。


### 论文摘要

In this paper, we develop a deep learning-based bandwidth allocation policy that is: 1) scalable with the number of users and 2) transferable to different communication scenarios, such as non-stationary wireless channels, different quality-of-service (QoS) requirements, and dynamically available resources. To support scalability, the bandwidth allocation policy is represented by a graph neural network (GNN), with which the number of training parameters does not change with the number of users. To enable the generalization of the GNN, we develop a hybrid-task meta-learning (HML) algorithm that trains the initial parameters of the GNN with different communication scenarios during meta-training. Next, during meta-testing, a few samples are used to fine-tune the GNN with unseen communication scenarios. Simulation results demonstrate that our HML approach can improve the initial performance by 8.79%, and sample efficiency by 73%, compared with existing benchmarks. After fine-tuning, our near-optimal GNN-based policy can achieve close to the same reward with much lower inference complexity compared to the optimal policy obtained using iterative optimization. Numerical results validate that our HML can reduce the computation time by approximately 200 to 2000 times than the optimal iterative algorithm.

---

## 2. Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks

**论文链接:** [http://arxiv.org/abs/2510.25760v2](http://arxiv.org/abs/2510.25760v2)

**作者:** Xu Zheng, Zihao Dongfang, Lutao Jiang, Boyuan Zheng, Yulong Guo, Zhenquan Zhang, Giuliano Albanese, Runyi Yang, Mengjiao Ma, Zixin Zhang, Chenfei Liao, Dingcheng Zhen, Yuanhuiyi Lyu, Yuqian Fu, Bin Ren, Linfeng Zhang, Danda Pani Paudel, Nicu Sebe, Luc Van Gool, Xuming Hu

**发布时间:** 2025-10-29

### GPT解析

### 总结

这篇论文对大型模型在多模态空间推理任务方面进行了全面综述，分类了多模态大型语言模型的最新进展，并介绍了开放基准进行评估。

### 背景

人类具有通过视觉和声音等多模态观察理解空间的空间推理能力。大型多模态推理模型通过学习感知和推理，在多样化的空间任务中展现出有前景的性能，但系统综述和公开可用的基准仍然有限。

### 目的

提供对大型模型多模态空间推理任务的全面回顾，分类多模态大型语言模型(MLLMs)的最新进展，并介绍用于评估的开放基准。

### 方法

从概述一般空间推理开始，重点关注训练后技术、可解释性和架构。研究空间关系推理、场景和布局理解、3D空间中的视觉问答和定位。回顾具身AI进展，包括视觉语言导航和动作模型。考虑音频和第一人称视频等新兴模态。

### 主要发现

多模态大型模型在空间推理任务中展现出有前景的性能，但仍需要更多系统研究和公开基准来评估这些模型的能力。

### 结论

这篇综述为多模态空间推理这一不断发展的领域奠定了坚实基础并提供了见解。相关更新信息、代码和开放基准的实现可在GitHub上获取。

### 翻译

人类具有空间推理能力，使他们能够通过视觉和声音等多模态观察理解空间。大型多模态推理模型通过学习感知和推理扩展了这些能力，在多样化的空间任务中展现出有前景的性能。然而，这些模型的系统综述和公开可用的基准仍然有限。在这篇综述中，我们提供了对大型模型多模态空间推理任务的全面回顾，分类了多模态大型语言模型(MLLMs)的最新进展，并介绍了用于评估的开放基准。我们首先概述一般空间推理，重点关注训练后技术、可解释性和架构。除了传统的2D任务外，我们还研究了空间关系推理、场景和布局理解，以及3D空间中的视觉问答和定位。我们还回顾了具身AI的进展，包括视觉语言导航和动作模型。此外，我们还考虑了音频和第一人称视频等新兴模态，这些模态通过新型传感器促进新的空间理解。我们相信这篇综述为多模态空间推理这一不断发展的领域奠定了坚实基础并提供了见解。关于这篇综述的更新信息、代码和开放基准的实现可在https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning上找到。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态空间推理领域缺乏系统性回顾和公开可用基准的问题。这个问题很重要，因为人类具有通过视觉、声音等多模态感知理解空间的能力，而大型多模态模型虽已展现出色性能，但缺乏系统评估和比较标准，阻碍了该领域的快速发展。空间推理对导航、物体关系理解和复杂场景交互等实际应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过构建分类框架（如图2所示）组织多模态空间推理研究，从一般多模态空间推理到3D空间推理，再到具身AI和新兴模态。作者借鉴了多模态模型、空间推理和具身AI等领域的现有工作，同时发现前人研究存在空白，如Wang等人专注于单模态任务，Ke等人未深入多模态空间推理，Bi等人未提供系统评估框架。作者通过系统性文献回顾和基准构建填补了这些空白。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过系统回顾和基准测试促进多模态空间推理研究发展。流程包括：1)明确定义多模态空间推理任务和评估协议；2)构建分类框架涵盖2D到3D、静态到动态、传统到新兴模态；3)全面回顾文献，包括后训练技术、可解释性、架构设计等；4)开发开放基准评估模型性能；5)提供代码和实现资源促进进一步研究。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提供多模态空间推理的系统性全面回顾；2)构建详细分类框架（图2）覆盖广泛任务；3)引入开放基准标准化评估；4)整合空间推理与具身AI；5)提供资源促进研究。相比前人工作，本文更全面系统，不仅涵盖2D到3D任务，还整合新兴模态和具身AI，并提供实用评估基准，而非仅关注单一领域或理论实现。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性的文献回顾、分类框架构建和开放基准引入，为多模态空间推理领域提供了坚实基础和评估标准，促进了该领域的研究发展和实际应用。'}


### 论文摘要

Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.

---

## 3. Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.27629v2](http://arxiv.org/abs/2510.27629v2)

**作者:** Boyi Wei, Zora Che, Nathaniel Li, Udari Madhushani Sehwag, Jasper Götting, Samira Nedungadi, Julian Michael, Summer Yue, Dan Hendrycks, Peter Henderson, Zifan Wang, Seth Donoughe, Mantas Mazeika

**发布时间:** 2025-10-31

**备注:** 17 Pages, 5 figures

### GPT解析

### 总结

该研究提出了一种名为eval的框架，用于评估旨在减少生物基础模型双重使用能力的程序的鲁棒性。研究通过序列建模、突变效应预测和毒力预测三个角度评估模型对病毒的理解能力，发现当前的数据过滤方法可能不够有效，被排除的知识可以通过微调恢复，且双重使用信号可能已存在于预训练表示中。

### 背景

开放权重生物基础模型呈现双重使用困境。这些模型有加速科学研究和药物开发的巨大潜力，但也可能被恶意行为者用于开发更致命的生物武器。当前的方法主要关注在预训练过程中过滤生物危害数据，但其有效性尚不明确。

### 目的

解决当前过滤生物危害数据方法的有效性不明确问题，特别是针对可能微调这些模型进行恶意使用的有决心的行为者。提出一个框架来评估旨在减少生物基础模型双重使用能力的程序的鲁棒性。

### 方法

提出了名为eval的框架，通过三个角度评估模型的病毒理解能力：序列建模、突变效应预测和毒力预测。

### 主要发现

当前过滤实践可能不是特别有效；在某些情况下，被排除的知识可以通过微调快速恢复，并在序列建模中表现出更广泛的泛化能力；双重使用信号可能已经存在于预训练表示中，可以通过简单的线性探测来引出。

### 结论

数据过滤作为独立程序面临挑战，强调需要对开放权重生物基础模型进行更深入的安全和安保策略研究。

### 翻译

开放权重生物基础模型呈现双重使用困境。虽然这些模型在加速科学研究和药物开发方面展现出巨大潜力，但也可能被恶意行为者用于开发更致命的生物武器。为了减轻这些模型带来的风险，当前的方法主要关注在预训练过程中过滤生物危害数据。然而，这种方法的有效性仍不明确，特别是针对可能微调这些模型进行恶意使用的有决心的行为者。为了解决这一空白，我们提出了eval框架，用于评估旨在减少生物基础模型双重使用能力的程序的鲁棒性。eval通过序列建模、突变效应预测和毒力预测三个角度评估模型对病毒的理解能力。我们的结果表明，当前的过滤实践可能不是特别有效：在某些情况下，被排除的知识可以通过微调快速恢复，并在序列建模中表现出更广泛的泛化能力。此外，双重使用信号可能已经存在于预训练表示中，可以通过简单的线性探测来引出。这些发现强调了数据过滤作为独立程序所面临的挑战，突显了对开放权重生物基础模型进行更深入的安全和安保策略研究的必要性。


### 论文摘要

Open-weight bio-foundation models present a dual-use dilemma. While holding great promise for accelerating scientific research and drug development, they could also enable bad actors to develop more deadly bioweapons. To mitigate the risk posed by these models, current approaches focus on filtering biohazardous data during pre-training. However, the effectiveness of such an approach remains unclear, particularly against determined actors who might fine-tune these models for malicious use. To address this gap, we propose \eval, a framework to evaluate the robustness of procedures that are intended to reduce the dual-use capabilities of bio-foundation models. \eval assesses models' virus understanding through three lenses, including sequence modeling, mutational effects prediction, and virulence prediction. Our results show that current filtering practices may not be particularly effective: Excluded knowledge can be rapidly recovered in some cases via fine-tuning, and exhibits broader generalizability in sequence modeling. Furthermore, dual-use signals may already reside in the pretrained representations, and can be elicited via simple linear probing. These findings highlight the challenges of data filtering as a standalone procedure, underscoring the need for further research into robust safety and security strategies for open-weight bio-foundation models.

---

## 4. Image Hashing via Cross-View Code Alignment in the Age of Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.27584v2](http://arxiv.org/abs/2510.27584v2)

**作者:** Ilyass Moummad, Kawtar Zaher, Hervé Goëau, Alexis Joly

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了CroVCA（Cross-View Code Alignment），一种简单统一的原则，用于学习在不同语义对齐视图中保持一致的二进制码。通过HashCoder实现，该方法仅需5个训练周期就能达到最先进的结果，在16位时表现尤为出色，具有高效率、适应性和广泛适用性。

### 背景

大规模检索需要紧凑且具有判别性的表示。基础模型提供强大的视觉和多模态嵌入，但这些高维空间中的最近邻搜索计算成本高。哈希提供了一种高效的替代方案，但现有方法通常依赖复杂流程、多目标项、针对单一学习范式设计以及长时间训练。

### 目的

引入CroVCA，一种简单统一的原则，用于学习在不同语义对齐视图中保持一致的二进制码，实现高效、快速、准确的哈希方法。

### 方法

使用单个二元交叉熵损失强制对齐，编码率最大化作为反崩溃正则化器促进平衡和多样化的码。设计了HashCoder，一个轻量级的MLP哈希网络，带有最终的批归一化层以强制平衡的码。HashCoder可用作冻结嵌入上的探测头，或通过LoRA微调有效地适配编码器。

### 主要发现

CroVCA在基准测试中取得了最先进的结果，只需5个训练周期。在16位时表现特别好，例如在COCO上的无监督哈希在不到2分钟内完成，在ImageNet100上的监督哈希在单个GPU上约3分钟内完成。

### 结论

CroVCA展示了其效率、适应性和广泛的适用性。

### 翻译

高效的大规模检索需要既紧凑又具有判别性的表示。基础模型提供了强大的视觉和多模态嵌入，但这些高维空间中的最近邻搜索计算成本高昂。哈希通过使用二进制码实现快速汉明距离搜索，提供了一种高效的替代方案，然而现有方法通常依赖复杂流程、多目标项、针对单一学习范式的设计以及长时间的训练。我们引入了CroVCA（Cross-View Code Alignment），这是一种学习在不同语义对齐视图中保持一致的二进制码的简单统一原则。单个二元交叉熵损失强制对齐，而编码率最大化作为反崩溃正则化器促进平衡和多样化的码。为此，我们设计了HashCoder，一个带有最终批归一化层以强制平衡码的轻量级MLP哈希网络。HashCoder可以用作冻结嵌入上的探测头，或通过LoRA微调有效地适配编码器。在基准测试中，CroVCA仅需5个训练周期就取得了最先进的结果。在16位时，它表现特别好——例如，在单个GPU上，COCO上的无监督哈希在不到2分钟内完成，ImageNet100上的监督哈希在约3分钟内完成。这些结果凸显了CroVCA的效率、适应性和广泛的适用性。


### 论文摘要

Efficient large-scale retrieval requires representations that are both compact and discriminative. Foundation models provide powerful visual and multimodal embeddings, but nearest neighbor search in these high-dimensional spaces is computationally expensive. Hashing offers an efficient alternative by enabling fast Hamming distance search with binary codes, yet existing approaches often rely on complex pipelines, multi-term objectives, designs specialized for a single learning paradigm, and long training times. We introduce CroVCA (Cross-View Code Alignment), a simple and unified principle for learning binary codes that remain consistent across semantically aligned views. A single binary cross-entropy loss enforces alignment, while coding-rate maximization serves as an anti-collapse regularizer to promote balanced and diverse codes. To implement this, we design HashCoder, a lightweight MLP hashing network with a final batch normalization layer to enforce balanced codes. HashCoder can be used as a probing head on frozen embeddings or to adapt encoders efficiently via LoRA fine-tuning. Across benchmarks, CroVCA achieves state-of-the-art results in just 5 training epochs. At 16 bits, it particularly well-for instance, unsupervised hashing on COCO completes in under 2 minutes and supervised hashing on ImageNet100 in about 3 minutes on a single GPU. These results highlight CroVCA's efficiency, adaptability, and broad applicability.

---

