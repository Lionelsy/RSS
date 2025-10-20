# 今日论文推荐 - 2025-10-20

共 36 篇论文

---

## 1. Hypergraph Contrastive Sensor Fusion for Multimodal Fault Diagnosis in Induction Motors

**论文链接:** [http://arxiv.org/abs/2510.15547v1](http://arxiv.org/abs/2510.15547v1)

**作者:** Usman Ali, Ali Zia, Waqas Ali, Umer Ramzan, Abdul Rehman, Muhammad Tayyab Chaudhry, Wei Xiang

**发布时间:** 2025-10-17

**备注:** Submitted to IEEE Sensors Journal

### GPT解析

### 总结

本文提出了一种多模态超图对比注意力网络(MM-HCAN)，用于感应电机的鲁棒故障诊断，实现了高达99.82%的准确率，具有强大的跨域泛化能力和噪声鲁棒性。

### 背景

可靠的感应电机故障诊断对工业安全和运营连续性至关重要，但传统方法难以捕捉复杂的多模态信号关系，局限于单模态数据或单一故障类型，在嘈杂或跨域条件下性能下降。

### 目的

开发一个统一的故障诊断框架，解决多模态传感器融合问题，实现轴承、定子和转子故障的同时诊断，提高诊断系统的泛化能力和鲁棒性。

### 方法

提出多模态超图对比注意力网络(MM-HCAN)，首次将对比学习整合到专为多模态传感器融合设计的超图拓扑中，实现模态内和模态间依赖关系的联合建模，增强超越欧几里得嵌入空间的泛化能力。

### 主要发现

在三个真实世界基准测试中，MM-HCAN实现了高达99.82%的准确率，具有强大的跨域泛化能力和对噪声的鲁棒性，消融研究验证了每个组件的有效贡献。

### 结论

MM-HCAN为全面的多故障诊断提供了可扩展和鲁棒的解决方案，支持工业环境中的预测维护和资产寿命延长。

### 翻译

可靠的感应电机故障诊断对工业安全和运营连续性至关重要，可减轻昂贵的意外停机时间。传统方法往往难以捕捉复杂的多模态信号关系，局限于单模态数据或单一故障类型，并在嘈杂或跨域条件下表现出性能下降。本文提出了多模态超图对比注意力网络(MM-HCAN)，一个用于鲁棒故障诊断的统一框架。据我们所知，MM-HCAN首次将对比学习整合到专为多模态传感器融合设计的超图拓扑中，实现了模态内和模态间依赖关系的联合建模，并增强了超越欧几里得嵌入空间的泛化能力。该模型支持轴承、定子和转子故障的同时诊断，满足了工程上对整合诊断能力的需求。在三个真实世界基准测试中评估，MM-HCAN实现了高达99.82%的准确率，具有强大的跨域泛化能力和对噪声的鲁棒性，证明了其适合实际部署。消融研究验证了每个组件的贡献。MM-HCAN为全面的多故障诊断提供了可扩展和鲁棒的解决方案，支持工业环境中的预测维护和资产寿命延长。


### 论文摘要

Reliable induction motor (IM) fault diagnosis is vital for industrial safety and operational continuity, mitigating costly unplanned downtime. Conventional approaches often struggle to capture complex multimodal signal relationships, are constrained to unimodal data or single fault types, and exhibit performance degradation under noisy or cross-domain conditions. This paper proposes the Multimodal Hypergraph Contrastive Attention Network (MM-HCAN), a unified framework for robust fault diagnosis. To the best of our knowledge, MM-HCAN is the first to integrate contrastive learning within a hypergraph topology specifically designed for multimodal sensor fusion, enabling the joint modelling of intra- and inter-modal dependencies and enhancing generalisation beyond Euclidean embedding spaces. The model facilitates simultaneous diagnosis of bearing, stator, and rotor faults, addressing the engineering need for consolidated di- agnostic capabilities. Evaluated on three real-world benchmarks, MM-HCAN achieves up to 99.82% accuracy with strong cross-domain generalisation and resilience to noise, demonstrating its suitability for real-world deployment. An ablation study validates the contribution of each component. MM-HCAN provides a scalable and robust solution for comprehensive multi-fault diagnosis, supporting predictive maintenance and extended asset longevity in industrial environments.

---

## 2. MCA: Modality Composition Awareness for Robust Composed Multimodal Retrieval

**论文链接:** [http://arxiv.org/abs/2510.15543v1](http://arxiv.org/abs/2510.15543v1)

**作者:** Qiyu Wu, Shuyang Cui, Satoshi Hayakawa, Wei-Yao Wang, Hiromi Wakaki, Yuki Mitsufuji

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种模态组合感知框架，以提高多模态大语言模型作为统一编码器时的鲁棒性，解决了传统对比学习训练的统一编码器容易学习模态捷径的问题。

### 背景

多模态检索支持跨模态内容检索，应用广泛。单独编码器方法如CLIP通过对比学习对齐模态特定嵌入，而多模态大语言模型(MLLMs)实现了处理组合输入的统一编码器。

### 目的

解决统一编码器使用传统对比学习训练时容易学习模态捷径的问题，提高分布转移下的检索鲁棒性。

### 方法

提出模态组合感知框架，包含偏好损失强制多模态嵌入优于单模态对应部分，以及组合正则化目标将多模态嵌入与单模态部分组成的原型对齐，明确建模组合表示与单模态部分间的结构关系。

### 主要发现

在各种基准测试上，该框架在分布外检索方面表现提升，证明了模态组合感知是利用MLLM作为统一编码器时鲁棒组合多模态检索的有效原则。

### 结论

模态组合感知框架能有效提高多模态检索的鲁棒性，特别是在分布转移情况下，为多模态检索提供了新的有效原则。

### 翻译

多模态检索寻求跨模态（如文本或图像）检索相关内容，支持从AI搜索到内容生成的应用。尽管像CLIP这样的单独编码器方法通过对比学习对齐模态特定嵌入取得了成功，但最近的多模态大语言模型(MLLMs)实现了可以直接处理组合输入的统一编码器。虽然灵活且先进，我们发现使用传统对比学习训练的统一编码器容易学习模态捷径，导致在分布转移下鲁棒性差。我们提出了一种模态组合感知框架来缓解这一问题。具体而言，偏好损失强制多模态嵌入优于其单模态对应部分，而组合正则化目标将多模态嵌入与其单模态部分组成的原型对齐。这些目标明确建模了组合表示与其单模态对应部分之间的结构关系。在各种基准测试上的实验显示分布外检索有所提升，突显了模态组合感知作为利用MLLM作为统一编码器时鲁棒组合多模态检索的有效原则。


### 论文摘要

Multimodal retrieval, which seeks to retrieve relevant content across modalities such as text or image, supports applications from AI search to contents production. Despite the success of separate-encoder approaches like CLIP align modality-specific embeddings with contrastive learning, recent multimodal large language models (MLLMs) enable a unified encoder that directly processes composed inputs. While flexible and advanced, we identify that unified encoders trained with conventional contrastive learning are prone to learn modality shortcut, leading to poor robustness under distribution shifts. We propose a modality composition awareness framework to mitigate this issue. Concretely, a preference loss enforces multimodal embeddings to outperform their unimodal counterparts, while a composition regularization objective aligns multimodal embeddings with prototypes composed from its unimodal parts. These objectives explicitly model structural relationships between the composed representation and its unimodal counterparts. Experiments on various benchmarks show gains in out-of-distribution retrieval, highlighting modality composition awareness as a effective principle for robust composed multimodal retrieval when utilizing MLLMs as the unified encoder.

---

## 3. Large Reasoning Embedding Models: Towards Next-Generation Dense Retrieval Paradigm

**论文链接:** [http://arxiv.org/abs/2510.14321v2](http://arxiv.org/abs/2510.14321v2)

**作者:** Jianting Tang, Dongshuai Li, Tao Wen, Fuyu Lv, Dan Ou, Linli Xu

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了大型推理嵌入模型(LREM)，通过将推理过程整合到表示学习中，解决了现有嵌入模型在处理困难查询时的性能下降问题，显著提高了电子商务搜索系统的检索准确性。

### 背景

在现代电子商务搜索系统中，密集检索已成为不可或缺的组成部分。主流嵌入模型已从BERT转向大型语言模型(LLMs)以获得更准确的文本建模，但这些模型仍采用直接嵌入方法，语义准确性不足。现有模型通过对比学习实现语义对齐，但倾向于捕捉训练数据中的统计共现模式，偏向浅层词汇和语义匹配，导致对与目标物品词汇差异大的困难查询性能显著下降。

### 目的

解决现有嵌入模型在处理困难查询时的性能下降问题，通过整合推理过程到表示学习中，提高检索准确性，弥合原始查询和目标物品之间的语义差距。

### 方法

提出大型推理嵌入模型(LREM)，创新性地将推理过程整合到表示学习中。对于困难查询，LREM首先进行推理以深入理解原始查询，然后生成推理增强的查询嵌入用于检索。采用两阶段训练过程：第一阶段在精心策划的查询-思维链-物品三元组上使用SFT和InfoNCE损失优化LLM，建立初步推理和嵌入能力；第二阶段通过强化学习进一步优化推理轨迹。

### 主要发现

推理过程有效地弥合了原始查询和目标物品之间的语义差距，显著提高了检索准确性。大量的离线和在线实验验证了LREM的有效性。

### 结论

LREM已被部署在中国最大的电子商务平台上，自2025年8月起开始应用。

### 翻译

在现代电子商务搜索系统中，密集检索已成为不可或缺的组成部分。通过计算查询和物品(产品)嵌入之间的相似性，它能够从大规模存储库中高效地选择候选产品。随着大型语言模型(LLMs)的突破，主流嵌入模型已经逐渐从BERT转向LLMs以实现更准确的文本建模。然而，这些模型仍然采用直接嵌入方法，嵌入的语义准确性仍然不足。因此，对比学习被大量使用来实现正对之间的紧密语义对齐。结果，这类模型往往会捕捉训练数据中的统计共现模式，使其偏向于浅层词汇和语义匹配。对于与目标物品表现出显著词汇差异的困难查询，性能会显著下降。在这项工作中，我们提出了大型推理嵌入模型(LREM)，创新性地将推理过程整合到表示学习中。对于困难查询，LREM首先进行推理以实现对原始查询的深入理解，然后生成推理增强的查询嵌入用于检索。这个推理过程有效地弥合了原始查询和目标物品之间的语义差距，显著提高了检索准确性。具体来说，我们采用两阶段训练过程：第一阶段在精心策划的查询-思维链-物品三元组上使用SFT和InfoNCE损失优化LLM，建立初步推理和嵌入能力；第二阶段通过强化学习(RL)进一步优化推理轨迹。大量的离线和在线实验验证了LREM的有效性，使其自2025年8月起被部署在中国最大的电子商务平台上。


### 论文摘要

In modern e-commerce search systems, dense retrieval has become an indispensable component. By computing similarities between query and item (product) embeddings, it efficiently selects candidate products from large-scale repositories. With the breakthroughs in large language models (LLMs), mainstream embedding models have gradually shifted from BERT to LLMs for more accurate text modeling. However, these models still adopt direct-embedding methods, and the semantic accuracy of embeddings remains inadequate. Therefore, contrastive learning is heavily employed to achieve tight semantic alignment between positive pairs. Consequently, such models tend to capture statistical co-occurrence patterns in the training data, biasing them toward shallow lexical and semantic matches. For difficult queries exhibiting notable lexical disparity from target items, the performance degrades significantly. In this work, we propose the Large Reasoning Embedding Model (LREM), which novelly integrates reasoning processes into representation learning. For difficult queries, LREM first conducts reasoning to achieve a deep understanding of the original query, and then produces a reasoning-augmented query embedding for retrieval. This reasoning process effectively bridges the semantic gap between original queries and target items, significantly improving retrieval accuracy. Specifically, we adopt a two-stage training process: the first stage optimizes the LLM on carefully curated Query-CoT-Item triplets with SFT and InfoNCE losses to establish preliminary reasoning and embedding capabilities, and the second stage further refines the reasoning trajectories via reinforcement learning (RL). Extensive offline and online experiments validate the effectiveness of LREM, leading to its deployment on China's largest e-commerce platform since August 2025.

---

## 4. BLIP3o-NEXT: Next Frontier of Native Image Generation

**论文链接:** [http://arxiv.org/abs/2510.15857v1](http://arxiv.org/abs/2510.15857v1)

**作者:** Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, Tianyi Zhou, Junnan Li, Silvio Savarese, Caiming Xiong, Ran Xu

**发布时间:** 2025-10-17

### GPT解析

### 总结

BLIP3o-NEXT是一个全开源的基础模型，统一了文本到图像生成和图像编辑功能，采用自回归+扩散架构，在多种基准测试中表现优于现有模型。

### 背景

原生图像生成领域不断发展，需要能够同时处理图像生成和编辑的统一架构，以及探索影响模型性能的关键因素。

### 目的

开发一个先进的全开源基础模型，推动原生图像生成的前沿发展，并探索影响模型性能的关键见解。

### 方法

采用自回归+扩散架构，自回归模型基于多模态输入生成离散图像令牌，其隐藏状态作为扩散模型的条件信号生成高保真图像；结合后训练和数据引擎提高指令遵循能力和图像一致性。

### 主要发现

1. 架构选择对性能影响有限，有效扩展和快速推理是关键；2. 强化学习可进一步推动原生图像生成；3. 图像编辑具有挑战性，但可通过后训练和数据引擎改进；4. 数据质量和规模决定模型性能上限。

### 结论

BLIP3o-NEXT通过创新的架构设计和关键见解的应用，实现了图像生成和编辑的新水平，在多种基准测试中表现优异。

### 翻译

我们提出了BLIP3o-NEXT，这是BLIP3系列中的一个全开源基础模型，推动了原生图像生成的下一个前沿。BLIP3o-NEXT将文本到图像生成和图像编辑统一在一个架构中，展示了强大的图像生成和编辑能力。在开发最先进的原生图像生成模型过程中，我们确定了四个关键见解：(1) 大多数架构选择产生可比的性能；只要架构能有效扩展并支持快速推理，就可以被认为是有效的；(2) 强化学习的成功应用可以进一步推动原生图像生成的前沿；(3) 图像编辑仍然具有挑战性，但通过后训练和数据引擎可以显著提高指令遵循能力和生成图像与参考图像之间的一致性；(4) 数据质量和规模仍然是决定模型性能上限的决定性因素。基于这些见解，BLIP3o-NEXT采用自回归+扩散架构，其中自回归模型首先基于多模态输入生成离散图像令牌，然后其隐藏状态被用作扩散模型的条件信号以生成高保真图像。该架构结合了自回归模型的推理强度和指令遵循能力以及扩散模型的精细细节渲染能力，实现了新的连贯性和真实感水平。对各种文本到图像和图像编辑基准的广泛评估表明，BLIP3o-NEXT优于现有模型。


### 论文摘要

We present BLIP3o-NEXT, a fully open-source foundation model in the BLIP3 series that advances the next frontier of native image generation. BLIP3o-NEXT unifies text-to-image generation and image editing within a single architecture, demonstrating strong image generation and image editing capabilities. In developing the state-of-the-art native image generation model, we identify four key insights: (1) Most architectural choices yield comparable performance; an architecture can be deemed effective provided it scales efficiently and supports fast inference; (2) The successful application of reinforcement learning can further push the frontier of native image generation; (3) Image editing still remains a challenging task, yet instruction following and the consistency between generated and reference images can be significantly enhanced through post-training and data engine; (4) Data quality and scale continue to be decisive factors that determine the upper bound of model performance. Building upon these insights, BLIP3o-NEXT leverages an Autoregressive + Diffusion architecture in which an autoregressive model first generates discrete image tokens conditioned on multimodal inputs, whose hidden states are then used as conditioning signals for a diffusion model to generate high-fidelity images. This architecture integrates the reasoning strength and instruction following of autoregressive models with the fine-detail rendering ability of diffusion models, achieving a new level of coherence and realism. Extensive evaluations of various text-to-image and image-editing benchmarks show that BLIP3o-NEXT achieves superior performance over existing models.

---

## 5. SpeechLLMs for Large-scale Contextualized Zero-shot Slot Filling

**论文链接:** [http://arxiv.org/abs/2510.15851v1](http://arxiv.org/abs/2510.15851v1)

**作者:** Kadri Hacioglu, Manjunath K E, Andreas Stolcke

**发布时间:** 2025-10-17

**备注:** 13 pages, EMNLP 2025

### GPT解析

### 总结

本研究探讨了基于语音的大型语言模型（speech LLMs）在口语理解（SLU）槽填充任务中的应用，通过创建任务上界、识别性能差距并提出改进措施，显著提升了模型性能。

### 背景

槽填充是口语理解的关键子任务，传统实现方式为级联的语音识别后跟一个或多个自然语言理解组件。新兴的语音大型语言模型为理解任务提供了更统一、生成式和遵循指令的新途径，具有数据效率、计算效率和零样本能力。

### 目的

创建槽填充任务的经验上界，确定性能、鲁棒性和泛化差距，并提出改进措施以缩小与上界结果的差距。

### 方法

通过改进训练数据、架构和训练策略来提升模型性能，并评估这些改进措施的有效性。

### 主要发现

每项改进措施都显著提高了模型性能，同时研究还指出了实际应用中面临的挑战，并为利用这些新兴模型提供了经验指导。

### 结论

基于语音的大型语言模型为槽填充任务提供了新的有效途径，但仍需解决实践挑战，进一步优化以实现更好的性能和泛化能力。

### 翻译

槽填充是口语理解（SLU）中的一个关键子任务，传统实现方式为级联的语音识别后跟一个或多个自然语言理解（NLU）组件。最近出现的基于语音的大型语言模型（speech LLMs），它整合了语音和文本基础模型，为以更统一、生成式和遵循指令的方式实现语音理解任务开辟了新途径，同时承诺具有数据和计算效率，具有零样本能力，可推广到未见过的槽标签。我们通过为槽填充任务创建经验上界，确定性能、鲁棒性和泛化差距，并提出改进训练数据、架构和训练策略的建议来缩小与上界结果的差距。我们证明这些措施中的每一项都显著提高了性能，同时突出了实践挑战，并为利用这些新兴模型提供了经验指导和见解。


### 论文摘要

Slot filling is a crucial subtask in spoken language understanding (SLU), traditionally implemented as a cascade of speech recognition followed by one or more natural language understanding (NLU) components. The recent advent of speech-based large language models (speechLLMs), which integrate speech and textual foundation models, has opened new avenues for achieving speech understanding tasks in a more unified, generative, and instruction-following manner while promising data and compute efficiency with zero-shot abilities, generalizing to unseen slot labels. We address the slot-filling task by creating an empirical upper bound for the task, identifying performance, robustness, and generalization gaps, and proposing improvements to the training data, architecture, and training strategies to narrow the gap with the upper bound result. We show that each of these measures improve performance substantially, while highlighting practical challenges and providing empirical guidance and insights for harnessing these emerging models.

---

## 6. PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training

**论文链接:** [http://arxiv.org/abs/2510.15596v1](http://arxiv.org/abs/2510.15596v1)

**作者:** Alicia Golden, Michael Kuchnik, Samuel Hsia, Zachary DeVito, Gu-Yeon Wei, David Brooks, Carole-Jean Wu

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文研究了大规模模型训练（超过数万个GPU）中的性能变异性问题，提出了PRISM性能建模框架，考虑了大规模分布式训练的随机性质，为训练时间提供概率保证的量化度量。

### 背景

大规模模型训练中，训练过程中的中断是必然发生的随机事件，随着训练规模扩大和GPU在受限环境下运行，动态运行时变异会变得更加频繁。在64k GPU规模下，已观察到9%的GPU时间变异性，GEMM工作负载上GPU性能最高有14%的变异。

### 目的

理解性能变异性的潜在原因，并探索分布式训练的设计和优化空间，提出一种能考虑训练随机性质的性能建模框架。

### 方法

提出PRISM性能建模框架，其核心是统计方法，为训练时间提供概率保证的量化度量。使用该框架探索并行化方法到下一代训练系统的设计和优化空间，并通过真实系统测量进行验证。

### 主要发现

1) PRISM框架的训练时间预测准确率为20.8%的Kolmogorov-Smirnov距离；2) 根据计算节点放置的不同，可获得高达1.26倍的性能提升潜力；3) 优化通信内核（如AllGather和ReduceScatter）对最小化训练步骤时间变异贡献最大。

### 结论

PRISM框架能够有效建模和优化大规模分布式训练中的性能变异性，通过考虑并行化策略对变异的敏感性，可以显著提高训练效率。

### 翻译

数万个GPU以上的大规模模型训练是一个未知领域。在这种规模下，训练过程中的中断不是是否会发生的问题，而是何时会发生的问题——这是一种降低训练生产力的随机过程。随着训练规模扩大和GPU在越来越受限的功率和热应力环境下运行，动态运行时变异将变得越来越频繁。在64k GPU规模下，我们已经观察到前沿基础模型训练有9%的GPU时间变异性。为了理解变异性的潜在原因，我们分析了各种平台上大规模的GPU微基准测试，显示在GEMM工作负载上，GPU性能最高有14%的变异，这取决于训练硬件和部署环境。受我们的分析和围绕性能变异性的广阔设计空间的启发，我们提出了PRISM——一个考虑大规模分布式训练随机性质的性能建模框架。PRISM的核心是一种统计方法，为训练时间提供概率保证的量化度量。使用PRISM，我们探索了分布式训练的设计和优化空间，从并行化方法到下一代训练系统。PRISM通过真实系统测量进行了验证，显示出训练时间预测准确率为20.8%的Kolmogorov-Smirnov距离。使用PRISM，我们证明，如果考虑并行化策略对变异的敏感性，根据计算节点放置的不同，可获得高达1.26倍的性能提升潜力。此外，我们使用PRISM识别了为减少性能变异而优化的内核，并预测了在变异被放大的大规模作业中减速的概率。我们发现优化通信内核，如AllGather和ReduceScatter，对最小化训练步骤时间变异贡献最大。


### 论文摘要

Large model training beyond tens of thousands of GPUs is an uncharted territory. At such scales, disruptions to the training process are not a matter of if, but a matter of when -- a stochastic process degrading training productivity. Dynamic runtime variation will become increasingly more frequent as training scales up and GPUs are operated in increasingly power-limited and thermally-stressed environments. At the 64k GPU scale, we already observed 9% GPU time variability for frontier foundation model training. To understand potential causes of variability, we analyze GPU microbenchmarks at scale across a variety of platforms, showing up to 14% variation in GPU performance on GEMM workloads depending on training hardware and deployed environment.   Motivated by our analysis and the large design space around performance variability, we present PRISM -- a performance modeling framework that considers the stochastic nature of the large-scale distributed training. The core of PRISM is the statistical method that provides a quantifiable measure for probabilistic guarantees on training time. Using PRISM, we explore the design and optimization space of distributed training, from parallelization methods to next-generation training systems. PRISM is validated with real-system measurement, showing training time prediction accuracy with 20.8% Kolmogorov-Smirnov distance. Using PRISM, we demonstrate that, depending on computation node placement, up to 1.26x performance improvement potential is available if we factor in sensitivities of parallelization strategies to variation. In addition, we use PRISM to identify kernels to optimize for reducing performance variability and predict probability of slow-down for large-scale jobs where variation is magnified. We find optimizing communication kernels, such as AllGather and ReduceScatter, contribute most to minimizing variability in training step time.

---

## 7. VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.15530v1](http://arxiv.org/abs/2510.15530v1)

**作者:** Zehao Ni, Yonghao He, Lingfeng Qian, Jilei Mao, Fa Fu, Wei Sui, Hu Su, Junran Peng, Zhipeng Wang, Bin He

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究提出了一种名为VO-DP的视觉单视图扩散策略学习方法，利用预训练的视觉基础模型实现语义和几何特征的有效融合，在机器人操作任务中表现出色，特别是在真实世界任务中显著优于基于点云的方法。

### 背景

在模仿学习中，基于视觉运动扩散策略学习是机器人操作的主要方向之一。大多数这类方法依赖点云作为观察输入，并通过点云特征学习构建场景表示，从而实现显著精度。然而，现有文献对纯视觉解决方案的深入探索不足，尽管这些方案具有巨大潜力。

### 目的

探索一种仅依赖视觉的单视图扩散策略学习方法(VO-DP)，以充分利用视觉基础模型的能力，实现语义和几何特征的有效融合，并评估其在模拟和真实世界任务中的性能。

### 方法

提出了一种名为VO-DP的视觉单视图扩散策略学习方法，利用预训练的视觉基础模型实现语义和几何特征的有效融合。具体包括：利用VGGT的中间特征，融合DINOv2的语义特征和交替注意力块的几何特征，通过交叉注意力融合特征，并使用CNN进行空间压缩形成策略头输入。

### 主要发现

1) 在模拟任务中，VO-DP平均成功率达到64.6%，与DP3(64.0%)相当，远高于DP(34.8%)；2) 在真实世界任务中，VO-DP达到87.9%的成功率，显著优于DP3(67.5%)和DP(11.2%)；3) VO-DP在颜色、大小、背景和光照等变化条件下表现出高度稳定性；4) VO-DP在真实世界任务中的性能明显优于基于点云的方法DP3。

### 结论

VO-DP是一种有效的视觉单视图扩散策略学习方法，在模拟和真实世界任务中均表现出色，特别是在真实环境中显著优于现有方法。该方法为机器人操作领域提供了一种纯视觉解决方案，展示了视觉基础模型在机器人学习中的巨大潜力。

### 翻译

在模仿学习的背景下，基于视觉运动的扩散策略学习是机器人操作的主要方向之一。大多数这些方法依赖点云作为观察输入，并通过点云特征学习构建场景表示，从而实现显著精度。然而，现有文献对纯视觉解决方案的深入探索不足，尽管这些方案具有巨大潜力。在本文中，我们提出了一种视觉单视图扩散策略学习方法(VO-DP)，利用预训练的视觉基础模型实现语义和几何特征的有效融合。我们利用VGGT的中间特征，融合DINOv2的语义特征和交替注意力块的几何特征。特征通过交叉注意力融合，并通过CNN进行空间压缩，形成策略头的输入。大量实验证明，VO-DP不仅显著优于纯视觉基线DP，而且与基于点云的方法DP3表现出不同的性能趋势：在模拟任务中，VO-DP平均成功率达到64.6%，与DP3的64.0%相当，远高于DP的34.8%；而在真实世界任务中，它达到87.9%，以显著优势分别超过DP3的67.5%和DP的11.2%。进一步的鲁棒性评估证实，VO-DP在颜色、大小、背景和光照等变化条件下保持高度稳定。最后，我们开源了一个机器人操作训练库。该库基于Accelerate构建，支持多机多GPU并行训练和混合精度训练。它兼容DP、DP3和VO-DP等视觉运动策略，并支持RoboTwin模拟器。


### 论文摘要

In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.

---

## 8. PFGS: Pose-Fused 3D Gaussian Splatting for Complete Multi-Pose Object Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.15386v1](http://arxiv.org/abs/2510.15386v1)

**作者:** Ting-Yu Yen, Yu-Sheng Chiu, Shih-Hsuan Hung, Peter Wonka, Hung-Kuo Chu

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究介绍了一种名为PFGS的姿态感知3D高斯飞溅框架，用于从多姿态图像捕获中重建完整物体。该方法通过迭代融合辅助姿态的图像到主姿态的统一3DGS表示中，结合全局和局部配准策略，有效解决了现有方法在重建遮挡或自遮挡区域时的不完整问题。

### 背景

3D高斯飞溅的最新进展已实现了从多视图图像生成高质量、实时的novel-view synthesis。然而，大多数现有方法假设物体以单一静态姿态被捕获，导致重建不完整，缺失了被遮挡或自遮挡区域。此外，最近的3D基础模型在提高配准鲁棒性和效率方面取得了进展，但仍受限于高内存需求和次优准确性。

### 目的

解决从多姿态图像捕获中重建完整物体的实际挑战，克服现有方法在重建遮挡区域时的不完整性问题，并解决基础模型在配准过程中的高内存需求和次优准确性问题。

### 方法

PFGS姿态感知3DGS框架，给定物体在一个主姿态和几个辅助姿态的图像，迭代地将每个辅助集融合到主姿态的统一3DGS表示中。采用姿态感知融合策略，结合全局和局部配准来有效合并视图并优化3DGS模型。通过更智能地将基础模型整合到配准过程中来克服挑战：利用背景特征进行每姿态相机姿态估计，并使用基础模型进行跨姿态配准。

### 主要发现

实验结果表明，PFGS在定性和定量评估中始终优于强大的基线方法，产生更完整的重建和更高保真度的3DGS模型。

### 结论

PFGS通过智能整合基础模型到配准过程中，结合了两种方法的优势，同时解决了背景不一致问题，实现了从多姿态图像捕获中重建完整物体的目标。

### 翻译

最近的3D高斯飞溅(3DGS)进展已经能够从多视图图像实现高质量、实时的novel-view synthesis。然而，大多数现有方法假设物体以单一静态姿态被捕获，导致重建不完整，缺失了被遮挡或自遮挡区域。我们引入了PFGS，一种姿态感知的3DGS框架，解决了从多姿态图像捕获中重建完整物体的实际挑战。给定物体在一个主姿态和几个辅助姿态的图像，PFGS迭代地将每个辅助集融合到主姿态的统一3DGS表示中。我们的姿态感知融合策略结合了全局和局部配准，以有效合并视图并优化3DGS模型。虽然最近的3D基础模型进展提高了配准的鲁棒性和效率，但仍受限于高内存需求和次优准确性。PFGS通过更智能地将它们整合到配准过程中克服了这些挑战：它利用背景特征进行每姿态相机姿态估计，并使用基础模型进行跨姿态配准。这种设计结合了两种方法的优势，同时解决了背景不一致问题。实验结果表明，PFGS在定性和定量评估中始终优于强大的基线方法，产生更完整的重建和更高保真度的3DGS模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从多姿态图像捕获中重建完整3D物体的问题。现有的3D高斯泼溅方法假设物体在单一静态姿态下被捕获，导致重建不完整，特别是会错过被物体自身遮挡的区域。这个问题在现实中非常重要，因为完整3D重建对虚拟现实、增强现实、机器人技术和数字孪生等应用至关重要，而单一姿态无法获取物体的全部表面信息，特别是在自遮挡的情况下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有3D高斯泼溅方法在处理多姿态物体重建时的局限性，并识别出多姿态重建面临的技术挑战：物体姿态变化导致无法使用传统SfM技术、跨姿态变化破坏对应估计、合并独立重建模型会引入伪影。作者借鉴了3D基础模型（如Fast3R）来提高配准鲁棒性，采用轮廓共识融合策略对齐不同姿态相机，并使用背景特征进行姿态估计。他们设计了一个三阶段管道（全局配准、局部配准、3DGS模型完成）来解决这些问题，并通过实验验证了方法的有效性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用姿态感知融合策略，将不同姿态捕获的图像有效合并到一个统一的3D高斯泼溅表示中，结合全局和局部配准技术处理多姿态重建挑战。整体流程包括：1）预处理阶段构建初始3DGS并估计相机姿态；2）全局配准阶段选择混合姿态图像、使用3D基础模型估计姿态、通过两阶段轮廓共识融合对齐坐标系；3）局部配准阶段使用轮廓引导和光度目标优化进一步精炼对齐；4）3DGS模型完成阶段使用平衡采样策略微调模型；5）迭代过程逐步融入更多辅助姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出PFGS框架实现多姿态3D高斯泼溅物体的增量重建；2）设计有效的全局配准方法对齐不同姿态图像集；3）提出混合姿态图像选择策略确保几何一致性；4）开发两阶段轮廓共识融合策略统一坐标系；5）提出平衡采样策略处理图像数量不平衡问题。相比之前工作，PFGS能处理物体姿态变化（不同于传统SfM），解决了3D基础模型的内存和精度问题，专门针对多姿态重建优化（不同于其他3D高斯泼溅方法），不依赖顺序输入结构（不同于在线重建方法），专注于合并不同姿态的部分重建（不同于物体聚焦方法）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PFGS通过姿态感知融合策略，将多姿态图像捕获合并为完整3D高斯泼溅表示，解决了传统方法在处理物体自遮挡和跨姿态变化时的局限性，实现了更完整、更高保真度的3D物体重建。'}


### 论文摘要

Recent advances in 3D Gaussian Splatting (3DGS) have enabled high-quality, real-time novel-view synthesis from multi-view images. However, most existing methods assume the object is captured in a single, static pose, resulting in incomplete reconstructions that miss occluded or self-occluded regions. We introduce PFGS, a pose-aware 3DGS framework that addresses the practical challenge of reconstructing complete objects from multi-pose image captures. Given images of an object in one main pose and several auxiliary poses, PFGS iteratively fuses each auxiliary set into a unified 3DGS representation of the main pose. Our pose-aware fusion strategy combines global and local registration to merge views effectively and refine the 3DGS model. While recent advances in 3D foundation models have improved registration robustness and efficiency, they remain limited by high memory demands and suboptimal accuracy. PFGS overcomes these challenges by incorporating them more intelligently into the registration process: it leverages background features for per-pose camera pose estimation and employs foundation models for cross-pose registration. This design captures the best of both approaches while resolving background inconsistency issues. Experimental results demonstrate that PFGS consistently outperforms strong baselines in both qualitative and quantitative evaluations, producing more complete reconstructions and higher-fidelity 3DGS models.

---

## 9. Symmetric Entropy-Constrained Video Coding for Machines

**论文链接:** [http://arxiv.org/abs/2510.15347v1](http://arxiv.org/abs/2510.15347v1)

**作者:** Yuxiao Sun, Yao Zhao, Meiqin Liu, Chao Yao, Jian Jin, Weisi Lin

**发布时间:** 2025-10-17

**备注:** This paper is prepared to submit to the IEEE Transactions

### GPT解析

### 总结

本文提出了一种对称熵约束的视频编码框架(SEC-VCM)，通过建立视频编解码器与视觉主干之间的对称对齐，优化机器视觉系统的视频编码效果。

### 背景

视频传输越来越多地服务于机器视觉系统(MVS)而非人类视觉系统(HVS)，视频编码为机器(VCM)已成为关键研究课题。现有VCM方法通常将编解码器绑定到特定下游模型，需要重新训练或有监督数据，限制了多任务场景中的泛化能力。

### 目的

提出一种对称熵约束的视频编码框架用于机器(SEC-VCM)，建立视频编解码器与视觉主干之间的对称对齐，使编解码器能够利用视觉主干的表示能力保留语义信息并丢弃机器视觉系统无关的信息。

### 方法

1) 提出对称熵约束的视频编码框架(SEC-VCM)；2) 建立视频编解码器与视觉主干之间的对称对齐；3) 采用双向熵约束(BiEC)机制确保视频解码和视觉主干编码过程的对称性；4) 通过语义-像素双路径融合(SPDF)模块将像素级先验注入到最终重建中。

### 主要发现

实验结果表明，该框架在速率-任务性能方面达到了最先进水平，与VTM相比，在视频实例分割(节省37.41%比特率)、视频对象分割(29.83%)、目标检测(46.22%)和多目标跟踪(44.94%)任务上实现了显著的比特率节省。

### 结论

SEC-VCM框架通过建立视频编解码器与视觉主干之间的对称对齐，有效利用了视觉主干的表示能力，保留了语义信息并丢弃了机器视觉系统无关的信息，显著提高了机器导向的重建质量。

### 翻译

随着视频传输越来越多地服务于机器视觉系统(MVS)而非人类视觉系统(HVS)，视频编码为机器(VCM)已成为关键研究课题。现有的VCM方法通常将编解码器绑定到特定下游模型，需要重新训练或有监督数据，从而限制了多任务场景中的泛化能力。最近，统一的VCM框架采用视觉主干(VB)和视觉基础模型(VFM)来支持单个编解码器完成多个视频理解任务。它们主要利用VB/VFM来保持语义一致性或抑制非语义信息，但很少探索如何在VB/VFM指导下直接将视频编码与理解联系起来。因此，我们提出了面向机器的对称熵约束视频编码框架(SEC-VCM)。它在视频编解码器和视觉主干之间建立了对称对齐，使编解码器能够利用视觉主干的表示能力来保留语义并丢弃机器视觉系统无关的信息。具体而言，双向熵约束(BiEC)机制通过抑制条件熵确保视频解码和视觉主干编码过程的对称性。这有助于编解码器明确处理对机器视觉系统有益的语义信息，同时压缩无用信息。此外，语义-像素双路径融合(SPDF)模块将像素级先验注入到最终重建中。通过语义-像素融合，它抑制了对机器视觉系统有害的伪影，并提高了机器导向的重建质量。实验结果表明，我们的框架在速率-任务性能方面达到了最先进水平，与VTM相比，在视频实例分割(节省37.41%)、视频对象分割(29.83%)、目标检测(46.22%)和多目标跟踪(44.94%)任务上实现了显著的比特率节省。我们将发布我们的代码。


### 论文摘要

As video transmission increasingly serves machine vision systems (MVS) instead of human vision systems (HVS), video coding for machines (VCM) has become a critical research topic. Existing VCM methods often bind codecs to specific downstream models, requiring retraining or supervised data and thus limiting generalization in multi-task scenarios. Recently, unified VCM frameworks have employed visual backbones (VB) and visual foundation models (VFM) to support multiple video understanding tasks with a single codec. They mainly utilize VB/VFM to maintain semantic consistency or suppress non-semantic information, but seldom explore how to directly link video coding with understanding under VB/VFM guidance. Hence, we propose a Symmetric Entropy-Constrained Video Coding framework for Machines (SEC-VCM). It establishes a symmetric alignment between the video codec and VB, allowing the codec to leverage VB's representation capabilities to preserve semantics and discard MVS-irrelevant information. Specifically, a bi-directional entropy-constraint (BiEC) mechanism ensures symmetry between the process of video decoding and VB encoding by suppressing conditional entropy. This helps the codec to explicitly handle semantic information beneficial for MVS while squeezing useless information. Furthermore, a semantic-pixel dual-path fusion (SPDF) module injects pixel-level priors into the final reconstruction. Through semantic-pixel fusion, it suppresses artifacts harmful to MVS and improves machine-oriented reconstruction quality. Experimental results show our framework achieves state-of-the-art (SOTA) in rate-task performance, with significant bitrate savings over VTM on video instance segmentation (37.41%), video object segmentation (29.83%), object detection (46.22%), and multiple object tracking (44.94%). We will release our code.

---

## 10. Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition

**论文链接:** [http://arxiv.org/abs/2510.15280v1](http://arxiv.org/abs/2510.15280v1)

**作者:** Fan Liu, Jindong Han, Tengfei Lyu, Weijia Zhang, Zhe-Rui Yang, Lu Dai, Cancheng Liu, Hao Liu

**发布时间:** 2025-10-17

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文探讨基础模型(FMs)如GPT-4和AlphaFold如何推动科学研究向新范式转变，提出三阶段框架描述这一演变过程，并回顾当前应用、识别风险与未来方向。

### 背景

基础模型(FMs)如GPT-4和AlphaFold正在改变科学研究格局，它们不仅加速假设生成、实验设计和结果解释等任务，还引发了一个根本性问题：FMs是仅仅增强现有科学方法，还是重新定义科学研究的进行方式？

### 目的

支持科学界理解基础模型(FMs)的变革作用，促进对科学发现未来的反思，并通过提出的三阶段框架帮助理解FMs如何催化科学研究的范式转变。

### 方法

提出一个三阶段框架描述基础模型(FMs)在科学中的演变：(1)元科学整合阶段，FMs增强传统范式中的工作流程；(2)混合人机共创阶段，FMs成为问题制定、推理和发现的积极合作者；(3)自主科学发现阶段，FMs作为独立运行，能够在最少人类干预下生成新科学知识。

### 主要发现

基础模型(FMs)正在催化向新科学范式的转变，通过三个阶段逐步演变：从增强传统工作流程，到成为人机共创的积极合作者，最终实现自主科学发现。作者还回顾了FMs在现有科学范式中的应用和新兴能力，并确定了相关风险和未来发展方向。

### 结论

基础模型(FMs)不仅仅是增强现有科学方法论的工具，而是正在重新定义科学研究的进行方式，推动科学向新范式转变。科学社区需要理解这一变革性作用，并反思科学发现的未来。

### 翻译

基础模型(FMs)如GPT-4和AlphaFold正在重塑科学研究格局。除了加速假设生成、实验设计和结果解释等任务外，它们还引发了一个更根本的问题：FMs仅仅是增强现有科学方法论，还是重新定义了科学研究的进行方式？在本文中，我们认为FMs正在催化向新科学范式的转变。我们引入了一个三阶段框架来描述这一演变：(1)元科学整合，FMs增强传统范式中的工作流程；(2)混合人机共创，FMs成为问题制定、推理和发现的积极合作者；(3)自主科学发现，FMs作为独立运行，能够在最少人类干预下生成新科学知识。通过这一视角，我们回顾了FMs在现有科学范式中的应用和新兴能力。我们进一步确定了FMs赋能的科学发现的风险和未来方向。这篇立场论文旨在支持科学界理解FMs的变革作用，并促进对科学发现未来的反思。我们的项目可在https://github.com/usail-hkust/Awesome-Foundation-Models-for-Scientific-Discovery获取。


### 论文摘要

Foundation models (FMs), such as GPT-4 and AlphaFold, are reshaping the landscape of scientific research. Beyond accelerating tasks such as hypothesis generation, experimental design, and result interpretation, they prompt a more fundamental question: Are FMs merely enhancing existing scientific methodologies, or are they redefining the way science is conducted? In this paper, we argue that FMs are catalyzing a transition toward a new scientific paradigm. We introduce a three-stage framework to describe this evolution: (1) Meta-Scientific Integration, where FMs enhance workflows within traditional paradigms; (2) Hybrid Human-AI Co-Creation, where FMs become active collaborators in problem formulation, reasoning, and discovery; and (3) Autonomous Scientific Discovery, where FMs operate as independent agents capable of generating new scientific knowledge with minimal human intervention. Through this lens, we review current applications and emerging capabilities of FMs across existing scientific paradigms. We further identify risks and future directions for FM-enabled scientific discovery. This position paper aims to support the scientific community in understanding the transformative role of FMs and to foster reflection on the future of scientific discovery. Our project is available at https://github.com/usail-hkust/Awesome-Foundation-Models-for-Scientific-Discovery.

---

## 11. Reflections from Research Roundtables at the Conference on Health, Inference, and Learning (CHIL) 2025

**论文链接:** [http://arxiv.org/abs/2510.15217v1](http://arxiv.org/abs/2510.15217v1)

**作者:** Emily Alsentzer, Marie-Laure Charpignon, Bill Chen, Niharika D'Souza, Jason Fries, Yixing Jiang, Aparajita Kashyap, Chanwoo Kim, Simon Lee, Aishwarya Mandyam, Ashery Christopher Mbilinyi, Nikita Mehandru, Nitish Nagesh, Brighton Nuwagira, Emma Pierson, Arvind Pillai, Akane Sano, Tanveer Syeda-Mahmood, Shashank Yadav, Elias Adhanom, Muhammad Umar Afza, Amelia Archer, Suhana Bedi, Vasiliki Bikia, Trenton Chang, George H. Chen, Winston Chen, Erica Chiang, Edward Choi, Octavia Ciora, Paz Dozie-Nnamah, Shaza Elsharief, Matthew Engelhard, Ali Eshragh, Jean Feng, Josh Fessel, Scott Fleming, Kei Sen Fong, Thomas Frost, Soham Gadgil, Judy Gichoya, Leeor Hershkovich, Sujeong Im, Bhavya Jain, Vincent Jeanselme, Furong Jia, Qixuan, Jin, Yuxuan Jin, Daniel Kapash, Geetika Kapoor, Behdokht Kiafar, Matthias Kleiner, Stefan Kraft, Annika Kumar, Daeun Kyung, Zhongyuan Liang, Joanna Lin, Qianchu, Liu, Chang Liu, Hongzhou Luan, Chris Lunt, Leopoldo Julían Lechuga López, Matthew B. A. McDermott, Shahriar Noroozizadeh, Connor O'Brien, YongKyung Oh, Mixail Ota, Stephen Pfohl, Meagan Pi, Tanmoy Sarkar Pias, Emma Rocheteau, Avishaan Sethi, Toru Shirakawa, Anita Silver, Neha Simha, Kamile Stankeviciute, Max Sunog, Peter Szolovits, Shengpu Tang, Jialu Tang, Aaron Tierney, John Valdovinos, Byron Wallace, Will Ke Wang, Peter Washington, Jeremy Weiss, Daniel Wolfe, Emily Wong, Hye Sun Yun, Xiaoman Zhang, Xiao Yu Cindy Zhang, Hayoung Jeong, Kaveri A. Thakoor

**发布时间:** 2025-10-17

### GPT解析

### 总结

第六届健康、推理和学习年度会议(CHIL 2025)于2025年6月在美国加州大学伯克利分校举行，会议举办了8个研究圆桌会议，促进机器学习与医疗保健交叉领域的协作讨论。

### 背景

会议由健康学习与推理协会(AHLI)主办，于2025年6月25-27日在美国加州大学伯克利分校举行。

### 目的

促进机器学习和医疗保健交叉领域的协作小组对话，讨论关键挑战、探索新兴机会、构思可行方向。

### 方法

举办研究圆桌会议，每个圆桌会议由高级和初级主席共同主持，强调开放交流、智力好奇心和包容性参与。

### 主要发现

会议涵盖了8个主题：'可解释性、可解释性和透明度'、'不确定性、偏见和公平性'、'因果关系'、'领域适应'、'基础模型'、'从小型医疗数据学习'、'多模态方法'和'可扩展、可转化的医疗保健解决方案'。

### 结论

通过8个由19位圆桌主席主持的圆桌会议，会议促进了该领域的严格讨论、机会探索和方向构思。

### 翻译

第六届健康、推理和学习年度会议(CHIL 2025)由健康学习与推理协会(AHLI)主办，于2025年6月25-27日在美国加州大学伯克利分校举行。作为今年计划的一部分，我们举办了研究圆桌会议，以促进机器学习和医疗保健交叉领域关键及时话题的协作小组对话。每个圆桌会议由高级和初级主席团队主持，他们促进了开放交流、智力好奇心和包容性参与。会议强调对关键挑战的严格讨论、新兴机会的探索以及该领域可行方向的集体构思。总共有19位圆桌主席主持了8个圆桌会议，主题包括'可解释性、可解释性和透明度'、'不确定性、偏见和公平性'、'因果关系'、'领域适应'、'基础模型'、'从小型医疗数据学习'、'多模态方法'和'可扩展、可转化的医疗保健解决方案'。


### 论文摘要

The 6th Annual Conference on Health, Inference, and Learning (CHIL 2025), hosted by the Association for Health Learning and Inference (AHLI), was held in person on June 25-27, 2025, at the University of California, Berkeley, in Berkeley, California, USA. As part of this year's program, we hosted Research Roundtables to catalyze collaborative, small-group dialogue around critical, timely topics at the intersection of machine learning and healthcare. Each roundtable was moderated by a team of senior and junior chairs who fostered open exchange, intellectual curiosity, and inclusive engagement. The sessions emphasized rigorous discussion of key challenges, exploration of emerging opportunities, and collective ideation toward actionable directions in the field. In total, eight roundtables were held by 19 roundtable chairs on topics of "Explainability, Interpretability, and Transparency," "Uncertainty, Bias, and Fairness," "Causality," "Domain Adaptation," "Foundation Models," "Learning from Small Medical Data," "Multimodal Methods," and "Scalable, Translational Healthcare Solutions."

---

## 12. Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection

**论文链接:** [http://arxiv.org/abs/2510.15202v1](http://arxiv.org/abs/2510.15202v1)

**作者:** Denis Janiak, Jakub Binkowski, Tomasz Kajdanowicz

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究探讨了分布外(OOD)检测中马氏距离方法的可靠性问题，分析了表示几何和规范化对性能的影响，并提出了一种新的径向缩放ℓ2规范化方法。

### 背景

OOD检测对于深度学习模型的可靠部署至关重要，而马氏距离方法虽被广泛使用，但其表示几何和规范化对性能的影响尚未被充分理解，这可能限制其下游应用。

### 目的

解决对马氏距离方法中表示几何和规范化影响理解不足的问题，通过全面的实证研究探索这些因素与OOD性能的关系。

### 方法

研究进行了跨不同图像基础模型、数据集和距离规范化方案的实证分析，定义了数据表示的理想几何形状，分析了规范化对OOD性能的影响，并提出了径向缩放ℓ2规范化方法。

### 主要发现

基于马氏距离的方法并非普遍可靠；光谱和内在维度指标可以准确预测模型的OOD性能；规范化对OOD性能有显著影响。

### 结论

提出的径向缩放ℓ2规范化方法通过引入可调参数控制特征空间的径向几何，系统性收缩或扩展表示以显著提高OOD检测性能，为设计更有效和可靠的深度学习模型提供了新见解。

### 翻译

该研究探讨了分布外(OOD)检测中马氏距离方法的可靠性问题，分析了表示几何和规范化对性能的影响，并提出了一种新的径向缩放ℓ2规范化方法。OOD检测对于深度学习模型的可靠部署至关重要，而马氏距离方法虽被广泛使用，但其表示几何和规范化对性能的影响尚未被充分理解，这可能限制其下游应用。研究旨在解决对马氏距离方法中表示几何和规范化影响理解不足的问题，通过全面的实证研究探索这些因素与OOD性能的关系。研究进行了跨不同图像基础模型、数据集和距离规范化方案的实证分析，定义了数据表示的理想几何形状，分析了规范化对OOD性能的影响，并提出了径向缩放ℓ2规范化方法。主要发现包括：基于马氏距离的方法并非普遍可靠；光谱和内在维度指标可以准确预测模型的OOD性能；规范化对OOD性能有显著影响。结论是，提出的径向缩放ℓ2规范化方法通过引入可调参数控制特征空间的径向几何，系统性收缩或扩展表示以显著提高OOD检测性能，为设计更有效和可靠的深度学习模型提供了新见解。


### 论文摘要

Out-of-distribution (OOD) detection is critical for the reliable deployment of deep learning models. hile Mahalanobis distance methods are widely used, the impact of representation geometry and normalization on their performance is not fully understood, which may limit their downstream application. To address this gap, we conducted a comprehensive empirical study across diverse image foundation models, datasets, and distance normalization schemes. First, our analysis shows that Mahalanobis-based methods aren't universally reliable. Second, we define the ideal geometry for data representations and demonstrate that spectral and intrinsic-dimensionality metrics can accurately predict a model's OOD performance. Finally, we analyze how normalization impacts OOD performance. Building upon these studies, we propose radially scaled $\ell_2$ normalization, a method that generalizes the standard $\ell_2$ normalization recently applied to Mahalanobis-based OOD detection. Our approach introduces a tunable parameter to directly control the radial geometry of the feature space, systematically contracting or expanding representations to significantly improve OOD detection performance. By bridging the gap between representation geometry, normalization, and OOD performance, our findings offer new insights into the design of more effective and reliable deep learning models.

---

## 13. The Economics of AI Foundation Models: Openness, Competition, and Governance

**论文链接:** [http://arxiv.org/abs/2510.15200v1](http://arxiv.org/abs/2510.15200v1)

**作者:** Fasheng Xu, Xiaoyu Wang, Wei Chen, Karen Xie

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究分析了基础模型生态系统中开放性的战略选择及其经济影响，发现开放性具有双重效应，并揭示了现有开发者的最优开放性策略与数据飞轮效应强度呈非单调关系，形成了'开放性陷阱'现象。

### 背景

基础模型(FM)生态系统中'开放性'的战略选择已成为一个关键问题，但其背后的经济驱动因素尚未被充分探索。

### 目的

分析开放性如何影响AI价值链中的竞争，包括现有开发者、下游部署者和新进入开发者之间的互动关系。

### 方法

构建了一个两期博弈论模型，研究开放性对竞争格局的影响。

### 主要发现

开放性具有双重效应：增强知识溢出到新进入者，同时通过'数据飞轮效应'增强现有开发者优势；现有开发者的最优开放性策略与数据飞轮效应强度呈非单调关系；中等数据飞轮效应下，现有开发者会战略性地限制开放性；形成了'开放性陷阱'，即透明度要求可能适得其反；垂直整合和政府补贴等干预措施也可能无效。

### 结论

通过建模开发者对竞争和监管压力的战略反应，为分析复杂且快速发展的FM生态系统中的竞争和设计有效政策提供了稳健的框架。

### 翻译

基础模型生态系统中'开放性'的战略选择已成为一个关键问题。虽然这一选择引发了激烈辩论，但其背后的经济驱动因素尚未得到充分探索。我们构建了一个两期博弈论模型，分析开放性如何影响AI价值链中的竞争，涉及现有开发者、下游部署者和新进入开发者。开放性产生双重效应：它增强了知识溢出到新进入者，但也通过'数据飞轮效应'增强了现有开发者的优势，即更大的用户参与度进一步降低了部署者未来的微调成本。我们的分析显示，现有开发者的第一期最优开放性强度与数据飞轮效应强度呈非单调关系。当数据飞轮效应较弱或非常强时，现有开发者偏好更高水平的开放性；然而，在中等范围内，它会战略性地限制开放性以损害新进入者的学习。这种动态导致了'开放性陷阱'，这是一个关键的政策悖论，即透明度要求可能适得其反，消除企业的战略灵活性，减少投资并降低福利。我们扩展了模型，表明其他常见的干预措施可能同样无效。例如，垂直整合只有在数据飞轮效应足够强到克服潜在更高效竞争对手的损失时才有利于生态系统。同样，旨在促进采用的政府补贴可能被现有开发者通过战略性的价格和开放性调整完全获取，使价值链的其他部分处境更差。通过建模开发者对竞争和监管压力的战略反应，我们为在复杂且快速发展的FM生态系统中分析竞争和设计有效政策提供了稳健的框架。


### 论文摘要

The strategic choice of model "openness" has become a defining issue for the foundation model (FM) ecosystem. While this choice is intensely debated, its underlying economic drivers remain underexplored. We construct a two-period game-theoretic model to analyze how openness shapes competition in an AI value chain, featuring an incumbent developer, a downstream deployer, and an entrant developer. Openness exerts a dual effect: it amplifies knowledge spillovers to the entrant, but it also enhances the incumbent's advantage through a "data flywheel effect," whereby greater user engagement today further lowers the deployer's future fine-tuning cost. Our analysis reveals that the incumbent's optimal first-period openness is surprisingly non-monotonic in the strength of the data flywheel effect. When the data flywheel effect is either weak or very strong, the incumbent prefers a higher level of openness; however, for an intermediate range, it strategically restricts openness to impair the entrant's learning. This dynamic gives rise to an "openness trap," a critical policy paradox where transparency mandates can backfire by removing firms' strategic flexibility, reducing investment, and lowering welfare. We extend the model to show that other common interventions can be similarly ineffective. Vertical integration, for instance, only benefits the ecosystem when the data flywheel effect is strong enough to overcome the loss of a potentially more efficient competitor. Likewise, government subsidies intended to spur adoption can be captured entirely by the incumbent through strategic price and openness adjustments, leaving the rest of the value chain worse off. By modeling the developer's strategic response to competitive and regulatory pressures, we provide a robust framework for analyzing competition and designing effective policy in the complex and rapidly evolving FM ecosystem.

---

## 14. Hyperparameter Optimization and Reproducibility in Deep Learning Model Training

**论文链接:** [http://arxiv.org/abs/2510.15164v1](http://arxiv.org/abs/2510.15164v1)

**作者:** Usman Afzaal, Ziyu Su, Usama Sajjad, Hao Lu, Mostafa Rezapour, Metin Nafi Gurcan, Muhammad Khalid Khan Niazi

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究解决了病理学基础模型训练中的可重复性挑战，调查了软件随机性、硬件非确定性和超参数报告不一致性问题，并通过系统评估不同超参数设置和数据增强策略的影响，提供了实用规则来指导未来开发可重复的数字病理学基础模型。

### 背景

基础模型训练在病理学领域面临可重复性挑战，这些挑战通常由软件随机性、硬件非确定性以及超参数报告不一致性所阻碍。

### 目的

调查这些问题，通过系统评估不同超参数设置和数据增强策略对模型性能的影响。

### 方法

在QUILT-1M数据集上训练了一个CLIP模型，并在三个下游病理学数据集（PatchCamelyon、LC25000-Lung和LC25000-Colon）上系统评估了不同超参数设置和数据增强策略的影响。

### 主要发现

• 图像裁剪策略中，中等程度的随机裁剪比更激进或更保守的设置表现更好；• 不带局部损失的分布式训练提高了模型稳定性；• 较低的学习率在所有数据集上均降低了模型性能；• 结肠组织数据集提供了最可重复的基准测试结果。

### 结论

计算病理学中的可重复性不仅依赖于透明的文档记录，还依赖于精心选择的实验配置，并提供了实用规则来指导未来开发可重复的数字病理学基础模型的工作。

### 翻译

可重复性仍然是病理学基础模型训练中的一个关键挑战，常常受到软件随机性、硬件非确定性和不一致的超参数报告的阻碍。为了调查这些问题，我们在QUILT-1M数据集上训练了一个CLIP模型，并系统评估了不同超参数设置和数据增强策略在三个下游病理学数据集（PatchCamelyon、LC25000-Lung和LC25000-Colon）上的影响。尽管不同运行之间存在变异性，我们确定了明显的趋势：RandomResizedCrop值为0.7-0.8比更激进(0.6)或保守(0.9)的设置表现更好，不带局部损失的分布式训练提高了稳定性，学习率低于5.0e-5在所有数据集上一致降低了性能。LC25000(Colon)数据集始终提供了最可重复的基准。这些发现强调，计算病理学中的可重复性不仅依赖于透明的文档记录，还依赖于精心选择的实验配置，我们提供了实用规则来指导未来开发可重复的数字病理学基础模型的工作。


### 论文摘要

Reproducibility remains a critical challenge in foundation model training for histopathology, often hindered by software randomness, hardware non-determinism, and inconsistent hyperparameter reporting. To investigate these issues, we trained a CLIP model on the QUILT-1M dataset and systematically evaluated the impact of different hyperparameter settings and augmentation strategies across three downstream histopathology datasets (PatchCamelyon, LC25000-Lung, and LC25000-Colon). Despite variability across runs, we identified clear trends: RandomResizedCrop values of 0.7-0.8 outperformed more aggressive (0.6) or conservative (0.9) settings, distributed training without local loss improved stability, and learning rates below 5.0e-5 consistently degraded performance across all datasets. The LC25000 (Colon) dataset consistently provided the most reproducible benchmark. These findings highlight that reproducibility in computational pathology depends not only on transparent documentation but also on carefully chosen experimental configurations, and we provide practical rules to guide future efforts in developing reproducible foundation models for digital pathology.

---

## 15. MOBIUS: Big-to-Mobile Universal Instance Segmentation via Multi-modal Bottleneck Fusion and Calibrated Decoder Pruning

**论文链接:** [http://arxiv.org/abs/2510.15026v1](http://arxiv.org/abs/2510.15026v1)

**作者:** Mattia Segu, Marta Tintore Gazulla, Yongqin Xian, Luc Van Gool, Federico Tombari

**发布时间:** 2025-10-16

**备注:** ICCV 2025

### GPT解析

### 总结

MOBIUS是一个高效的基础模型家族，专为通用实例分割设计，能够在保持高性能的同时大幅减少计算需求，实现了从高端加速器到移动硬件的跨设备部署。

### 背景

扩大模型规模和训练数据推动了基础模型在实例级感知方面的发展，在目标检测和分割任务中取得了最先进的性能，但高计算成本限制了它们在资源受限平台上的应用。

### 目的

研究现有架构在实现高效边缘部署方面的局限性，同时不牺牲性能；引入MOBIUS基础模型家族，实现帕累托最优的缩放，支持跨设备部署。

### 方法

提出瓶颈像素解码器用于高效多尺度多模态融合；提出语言引导的不确定性校准损失用于自适应解码器剪枝；提出简化的统一训练策略。

### 主要发现

MOBIUS将像素和transformer解码器的FLOPs分别减少了高达55%和75%，同时仅用三分之一训练迭代次数就保持了最先进的性能。

### 结论

MOBIUS为高性能计算平台和移动设备上的高效分割建立了新的基准。

### 翻译

扩大模型规模和训练数据推动了基础模型在实例级感知方面的发展，在目标检测和分割任务中实现了最先进的领域内和零样本性能。然而，它们的高计算成本限制了在资源受限平台上的应用。我们首先研究了现有架构在实现高效边缘部署而不牺牲性能方面的局限性。然后我们引入了MOBIUS，一个用于通用实例分割的基础模型家族，设计为帕累托最优的缩放，支持从高端加速器到移动硬件的跨设备部署。为了减少训练和推理需求，我们提出了：(i)用于高效多尺度多模态融合的瓶颈像素解码器，(ii)用于自适应解码器剪枝的语言引导不确定性校准损失，以及(iii)简化的统一训练策略。与权衡准确性和减少复杂性的高效基线不同，MOBIUS将像素和transformer解码器的FLOPs分别减少了高达55%和75%，同时在仅三分之一的训练迭代次数内保持了最先进的性能。MOBIUS为高性能计算平台和移动设备上的高效分割建立了新基准。


### 论文摘要

Scaling up model size and training data has advanced foundation models for instance-level perception, achieving state-of-the-art in-domain and zero-shot performance across object detection and segmentation. However, their high computational cost limits adoption on resource-constrained platforms. We first examine the limitations of existing architectures in enabling efficient edge deployment without compromising performance. We then introduce MOBIUS, a family of foundation models for universal instance segmentation, designed for Pareto-optimal downscaling to support deployment across devices ranging from high-end accelerators to mobile hardware. To reduce training and inference demands, we propose: (i) a bottleneck pixel decoder for efficient multi-scale and multi-modal fusion, (ii) a language-guided uncertainty calibration loss for adaptive decoder pruning, and (iii) a streamlined, unified training strategy. Unlike efficient baselines that trade accuracy for reduced complexity, MOBIUS reduces pixel and transformer decoder FLOPs by up to 55% and 75%, respectively, while maintaining state-of-the-art performance in just a third of the training iterations. MOBIUS establishes a new benchmark for efficient segmentation on both high-performance computing platforms and mobile devices.

---

## 16. Vision-Centric Activation and Coordination for Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.14349v2](http://arxiv.org/abs/2510.14349v2)

**作者:** Yunnan Wang, Fan Lu, Kecheng Zheng, Ziyuan Huang, Ziqiang Li, Wenjun Zeng, Xin Jin

**发布时间:** 2025-10-16

### GPT解析

### 总结

VaCo是一种优化多模态大语言模型(MLLMs)表示的新方法，通过整合多个视觉基础模型(VFMs)的视觉中心激活和协调，显著提高了MLLMs在视觉理解方面的性能。

### 背景

多模态大语言模型(MLLMs)通过整合视觉编码器的图像特征与LLMs展示先进理解能力，但主流MLLMs仅受文本token的下一词预测监督，忽略了分析能力所需的关键视觉中心信息。

### 目的

解决主流MLLMs忽略视觉中心信息的问题，提高MLLMs的视觉理解能力和分析能力。

### 方法

引入VaCo方法，包括：1)视觉判别对齐整合VFMs提取的任务感知特征；2)在MLLMs中融入可学习的模块化任务查询(MTQs)和视觉对齐层(VALs)；3)设计令牌网关掩码(TGM)协调多个VFMs之间的表示冲突。

### 主要发现

大量实验表明，VaCo在各种基准测试中显著提高了不同MLLMs的性能，展示了其在视觉理解方面的优越能力。

### 结论

VaCo通过整合多个VFMs的视觉特征，有效解决了主流MLLMs忽略视觉中心信息的问题，提高了MLLMs的综合性能。

### 翻译

多模态大语言模型(MLLMs)整合视觉编码器中的图像特征与LLMs，展示了先进的理解能力。然而，主流MLLMs仅受文本token的下一词预测监督，忽略了分析能力所需的关键视觉中心信息。为解决这一困境，我们引入了VaCo，它通过多个视觉基础模型(VFMs)的视觉中心激活和协调来优化MLLM表示。VaCo引入视觉判别对齐来整合从VFMs提取的任务感知特征，从而统一MLLM中文本和视觉输出的优化。具体而言，我们将可学习的模块化任务查询(MTQs)和视觉对齐层(VALs)整合到MLLMs中，在多样化VFMs的监督下激活特定的视觉信号。为协调VFMs之间的表示冲突，精心设计的令牌网关掩码(TGM)限制了多组MTQs之间的信息流动。大量实验表明，VaCo在各种基准测试中显著提高了不同MLLMs的性能，展示了其在视觉理解方面的卓越能力。


### 论文摘要

Multimodal large language models (MLLMs) integrate image features from visual encoders with LLMs, demonstrating advanced comprehension capabilities. However, mainstream MLLMs are solely supervised by the next-token prediction of textual tokens, neglecting critical vision-centric information essential for analytical abilities. To track this dilemma, we introduce VaCo, which optimizes MLLM representations through Vision-Centric activation and Coordination from multiple vision foundation models (VFMs). VaCo introduces visual discriminative alignment to integrate task-aware perceptual features extracted from VFMs, thereby unifying the optimization of both textual and visual outputs in MLLMs. Specifically, we incorporate the learnable Modular Task Queries (MTQs) and Visual Alignment Layers (VALs) into MLLMs, activating specific visual signals under the supervision of diverse VFMs. To coordinate representation conflicts across VFMs, the crafted Token Gateway Mask (TGM) restricts the information flow among multiple groups of MTQs. Extensive experiments demonstrate that VaCo significantly improves the performance of different MLLMs on various benchmarks, showcasing its superior capabilities in visual comprehension.

---

## 17. UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos

**论文链接:** [http://arxiv.org/abs/2510.15018v1](http://arxiv.org/abs/2510.15018v1)

**作者:** Mingxuan Liu, Honglin He, Elisa Ricci, Wayne Wu, Bolei Zhou

**发布时间:** 2025-10-16

**备注:** Technical report. Project page: https://urbanverseproject.github.io/

### GPT解析

### 总结

UrbanVerse是一个数据驱动的真实到仿真系统，将众包城市旅游视频转换为具有物理感知能力的交互式仿真场景，包含10万多个带注释的城市3D资产库和自动场景提取管道，显著提高了城市AI代理训练的效果和导航策略的泛化能力。

### 背景

城市中的实体AI代理（如配送机器人、四足机器人等）日益增多，它们需要在混乱的城市街道中导航以提供最后一公里连接。训练这类代理需要多样化的、高保真的城市环境，但现有的人工制作或程序生成的仿真场景要么缺乏可扩展性，要么无法捕捉真实世界的复杂性。

### 目的

引入UrbanVerse，一个数据驱动的真实到仿真系统，将众包的城市旅游视频转换为具有物理感知能力的交互式仿真场景。

### 方法

UrbanVerse包括两个部分：UrbanVerse-100K（包含10万多个带注释的城市3D资产库，具有语义和物理属性）和UrbanVerse-Gen（一个自动管道，从视频中提取场景布局，并使用检索到的资产创建度量尺度的3D仿真）。在IsaacSim中运行，提供来自24个国家的160个高质量构建场景和10个艺术家设计的测试场景的精选基准。

### 主要发现

UrbanVerse场景保留了真实世界的语义和布局，实现了与人工制作场景相当的人评估真实感。在城市导航中，在UrbanVerse中训练的策略显示出扩展幂律和强大的泛化能力，与先前的方法相比，在仿真中成功率提高了6.3%，在零样本仿真到真实迁移中提高了30.1%，仅用两次干预就完成了300米的真实世界任务。

### 结论

UrbanVerse系统能够有效地将真实世界的城市环境转化为高质量的仿真场景，为训练城市AI代理提供了有效工具，并显著提高了导航策略的性能和泛化能力。

### 翻译

城市实体AI代理，从配送机器人到四足机器人，正日益增多地遍布我们的城市，在混乱的街道上导航以提供最后一公里的连接。训练此类代理需要多样化、高保真的城市环境来扩展规模，然而现有的人工制作或程序生成的仿真场景要么缺乏可扩展性，要么无法捕捉真实世界的复杂性。我们引入了UrbanVerse，一个数据驱动的真实到仿真系统，将众包的城市旅游视频转换为具有物理感知能力的交互式仿真场景。UrbanVerse包括：(i) UrbanVerse-100K，一个包含10万多个带注释的城市3D资产库，具有语义和物理属性，以及(ii) UrbanVerse-Gen，一个自动管道，从视频中提取场景布局，并使用检索到的资产创建度量尺度的3D仿真。在IsaacSim中运行，UrbanVerse提供了来自24个国家的160个高质量构建场景，以及10个艺术家设计的测试场景的精选基准。实验表明，UrbanVerse场景保留了真实世界的语义和布局，实现了与人工制作场景相当的人评估真实感。在城市导航中，在UrbanVerse中训练的策略显示出扩展幂律和强大的泛化能力，与先前的方法相比，在仿真中成功率提高了6.3%，在零样本仿真到真实迁移中提高了30.1%，仅用两次干预就完成了300米的真实世界任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决城市环境中具身AI体（如配送机器人、四足机器人等）训练所需的高保真、多样化城市环境不足的问题。这个问题很重要，因为随着城市中微型移动系统的兴起，这些AI体需要能够泛化到各种复杂的真实世界环境中，但现有的模拟场景要么缺乏可扩展性，要么无法捕捉真实世界的复杂性和动态变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过认识到需要两个关键元素来构建解决方案：大规模3D资产数据库和自动化场景生成管道。他们借鉴了现有工作如Objaverse等3D资产库，但解决了其中资产质量、相关性和标注不足的问题；同时结合了MASt3R、YoloWorld、SAM2等多种技术来构建UrbanVerse-Gen管道，实现从未校准视频中提取场景信息。还参考了数字孪生概念，但扩展到城市街道级别场景生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个数据驱动的真实到模拟系统，将真实世界城市旅游视频转换为具有物理感知的交互式模拟场景，保留真实世界的语义、布局和物理特性。整体流程分为两大部分：1) UrbanVerse-100K资产数据库：收集和标注102,530个高质量3D城市对象，每个带有33种属性；2) UrbanVerse-Gen管道：从视频中提取场景信息，检索匹配资产，并在IsaacSim中生成可交互的数字孪生场景，包括场景蒸馏、资产匹配和场景组装三个阶段。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) UrbanVerse-100K大规模高质量标注资产库；2) UrbanVerse-Gen从未校准视频中自动生成高保真城市场景的管道；3) 跨24个国家的160场景训练库。相比之前工作，不同之处在于：解决了现有模拟器场景真实性不足、资产多样性有限和物理标注缺乏的问题；超越了仅使用被动数据训练的方法，通过交互式模拟实现更好的障碍物避免；扩展了数字孪生概念到城市街道级别，而非仅限于室内环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UrbanVerse通过将众源城市旅游视频转换为具有物理感知的交互式模拟场景，解决了城市环境中具身AI体训练所需的高保真、可扩展城市环境问题，实现了真实世界分布的保真度并显著提升了策略在模拟到真实世界零样本迁移中的性能。'}


### 论文摘要

Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.

---

## 18. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.15430v1](http://arxiv.org/abs/2510.15430v1)

**作者:** Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了Learning to Detect (LoD)框架，用于有效检测大型视觉语言模型中的未知越狱攻击，解决了现有检测方法的泛化性和效率问题。

### 背景

尽管进行了广泛的对齐努力，大型视觉语言模型(LVLMs)仍然容易受到越狱攻击，带来严重的安全风险。

### 目的

克服现有检测方法的局限性，提出一种能够准确检测未知越狱攻击的通用框架。

### 方法

提出Learning to Detect (LoD)框架，将重点从特定攻击学习转向特定任务学习。该框架包括多模态安全概念激活向量模块用于安全导向的表征学习和安全模式自动编码器模块用于无监督攻击分类。

### 主要发现

大量实验表明，该方法在各种未知攻击上实现了持续更高的检测AUROC，同时提高了效率。

### 结论

LoD框架通过转变学习方式，有效解决了现有检测方法在泛化性和效率方面的局限性，能够更准确、高效地检测未知攻击。

### 翻译

尽管进行了广泛的对齐努力，大型视觉语言模型(LVLMs)仍然容易受到越狱攻击，带来严重的安全风险。为解决这个问题，现有的检测方法要么学习特定攻击的参数，这阻碍了它们对未见攻击的泛化能力；要么依赖于启发式原则，这限制了准确性和效率。为克服这些局限性，我们提出了Learning to Detect (LoD)框架，一种通过将重点从特定攻击学习转向特定任务学习来准确检测未知越狱攻击的通用框架。该框架包括用于安全导向表征学习的多模态安全概念激活向量模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在各种未知攻击上实现了持续更高的检测AUROC，同时提高了效率。代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB获取。


### 论文摘要

Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

---

## 19. Large-scale User Game Lifecycle Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.15412v1](http://arxiv.org/abs/2510.15412v1)

**作者:** Yanjie Gou, Jiangming Liu, Kouying Xue, Yi Hua

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种用户游戏生命周期(UGL)方法来解决游戏推荐系统中的稀疏性和不平衡性问题，通过创新策略显著提升了游戏广告和推荐的性能。

### 背景

视频游戏产业快速发展，需要在线游戏平台开发有效的广告和推荐系统。现有的推荐系统表示学习方法不适合游戏场景，主要面临游戏稀疏性和不平衡性挑战。

### 目的

解决游戏推荐系统中的游戏稀疏性和不平衡性问题，提升游戏广告和推荐的性能。

### 方法

引入用户游戏生命周期(UGL)丰富用户行为；提出两种创新策略操纵用户行为以提取短期和长期兴趣；采用逆概率掩码策略处理游戏不平衡问题。

### 主要发现

UGL表示显著提升了模型性能，离线实验显示游戏广告AUC平均增加1.83%，游戏内物品推荐AUC增加0.5%；在线实验显示游戏广告CVR平均增加21.67%，游戏内物品推荐ARPU增加0.82%。

### 结论

UGL表示方法能有效解决游戏稀疏性和不平衡性问题，显著提升游戏广告和推荐的性能表现。

### 翻译

视频游戏生产的快速发展需要为在线游戏平台开发有效的广告和推荐系统。向用户推荐和宣传游戏取决于捕捉他们对游戏的兴趣。然而，为处理推荐系统中的数十亿物品而设计的现有表示学习方法不适合游戏广告和推荐。这主要是由于游戏稀疏性，其中仅有的几百个游戏不足以进行大规模用户表示学习，以及游戏不平衡性，其中用户行为被少数热门游戏主导。为解决稀疏性问题，我们引入了用户游戏生命周期(UGL)，旨在丰富用户在游戏中的行为。此外，我们提出了两种创新策略，旨在操纵用户行为以更有效地提取短期和长期兴趣。为解决游戏不平衡挑战，我们提出了用于UGL表示学习的逆概率掩码策略。离线和在线实验结果表明，UGL表示通过在游戏广告中平均实现1.83%的离线AUC增长和21.67%的在线CVR增长，以及在游戏内物品推荐中平均实现0.5%的离线AUC增长和0.82%的在线ARPU增长，显著增强了模型性能。


### 论文摘要

The rapid expansion of video game production necessitates the development of effective advertising and recommendation systems for online game platforms. Recommending and advertising games to users hinges on capturing their interest in games. However, existing representation learning methods crafted for handling billions of items in recommendation systems are unsuitable for game advertising and recommendation. This is primarily due to game sparsity, where the mere hundreds of games fall short for large-scale user representation learning, and game imbalance, where user behaviors are overwhelmingly dominated by a handful of popular games. To address the sparsity issue, we introduce the User Game Lifecycle (UGL), designed to enrich user behaviors in games. Additionally, we propose two innovative strategies aimed at manipulating user behaviors to more effectively extract both short and long-term interests. To tackle the game imbalance challenge, we present an Inverse Probability Masking strategy for UGL representation learning. The offline and online experimental results demonstrate that the UGL representations significantly enhance model by achieving a 1.83% AUC offline increase on average and a 21.67% CVR online increase on average for game advertising and a 0.5% AUC offline increase and a 0.82% ARPU online increase for in-game item recommendation.

---

## 20. Towards Robust Zero-Shot Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.15382v1](http://arxiv.org/abs/2510.15382v1)

**作者:** Kexin Zheng, Lauriane Teyssier, Yinan Zheng, Yu Luo, Xiayuan Zhan

**发布时间:** 2025-10-17

**备注:** Neurips 2025, 36 pages, 18 figures

### GPT解析

### 总结

本文提出了一种名为BREEZE的新型零样本强化学习框架，该框架基于前向-后向表示(FB)方法进行升级，通过引入行为正则化、任务条件扩散模型和基于注意力的表达架构，解决了现有方法中建模表达能力不足和离分布动作导致的外推误差问题，显著提升了学习稳定性、策略提取能力和表示学习质量。

### 背景

零样本强化学习的最新发展为学习预训练通用策略开辟了新途径，这些策略可以以零样本方式适应任意新任务。尽管流行的前向-后向表示(FB)及相关方法在零样本RL中显示出潜力，但它们的建模存在局限性。

### 目的

解决现有零样本RL方法中建模表达能力不足、离分布(OOD)动作在离线学习期间引起的外推误差导致有偏表示、最终造成次优性能的问题。

### 方法

作者提出了BREEZE(具有表达能力增强的行为正则化零样本RL)，这是一个升级的基于FB的框架，主要包括：1)在零样本RL策略学习中引入行为正则化，将策略优化转化为稳定的样本内学习范式；2)使用任务条件扩散模型提取策略，生成高质量和多模态的动作分布；3)采用基于注意力的表达架构进行表示建模，以捕捉环境动力学之间的复杂关系。

### 主要发现

在ExORL和D4RL Kitchen上的大量实验表明，BREEZE实现了最佳或接近最佳的性能，并且比先前的离线零样本RL方法表现出更强的鲁棒性。

### 结论

BREEZE通过结合行为正则化、任务条件扩散模型和基于注意力的表达架构，有效解决了现有零样本RL方法的局限性，显著提升了学习稳定性、策略提取能力和表示学习质量，为零样本强化学习领域提供了新的解决方案。

### 翻译

最近零样本强化学习(RL)的发展为学习预训练通用策略开辟了新途径，这些策略可以以零样本方式适应任意新任务。虽然流行的前向-后向表示(FB)及相关方法在零样本RL中显示出潜力，但我们通过实验发现它们的建模缺乏表达能力，并且离分布(OOD)动作在离线学习期间引起的外推误差有时会导致有偏表示，最终导致次优性能。为解决这些问题，我们提出了BREEZE(具有表达能力增强的行为正则化零样本RL)，这是一个升级的基于FB的框架，同时增强学习稳定性、策略提取能力和表示学习质量。BREEZE在零样本RL策略学习中引入行为正则化，将策略优化转化为稳定的样本内学习范式。此外，BREEZE使用任务条件扩散模型提取策略，能够在零样本RL设置中生成高质量和多模态的动作分布。而且，BREEZE采用基于注意力的表达架构进行表示建模，以捕捉环境动力学之间的复杂关系。在ExORL和D4RL Kitchen上的大量实验表明，BREEZE实现了最佳或接近最佳的性能，并且比先前的离线零样本RL方法表现出更强的鲁棒性。官方实现可在https://github.com/Whiterrrrr/BREEZE获取。


### 论文摘要

The recent development of zero-shot reinforcement learning (RL) has opened a new avenue for learning pre-trained generalist policies that can adapt to arbitrary new tasks in a zero-shot manner. While the popular Forward-Backward representations (FB) and related methods have shown promise in zero-shot RL, we empirically found that their modeling lacks expressivity and that extrapolation errors caused by out-of-distribution (OOD) actions during offline learning sometimes lead to biased representations, ultimately resulting in suboptimal performance. To address these issues, we propose Behavior-REgularizEd Zero-shot RL with Expressivity enhancement (BREEZE), an upgraded FB-based framework that simultaneously enhances learning stability, policy extraction capability, and representation learning quality. BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm. Additionally, BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings. Moreover, BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics. Extensive experiments on ExORL and D4RL Kitchen demonstrate that BREEZE achieves the best or near-the-best performance while exhibiting superior robustness compared to prior offline zero-shot RL methods. The official implementation is available at: https://github.com/Whiterrrrr/BREEZE.

---

## 21. DCMIL: A Progressive Representation Learning of Whole Slide Images for Cancer Prognosis Analysis

**论文链接:** [http://arxiv.org/abs/2510.14403v2](http://arxiv.org/abs/2510.14403v2)

**作者:** Chao Tu, Kun Huang, Jie Zhang, Qianjin Feng, Yu Zhang, Zhenyuan Ning

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种名为DCMIL（双课程对比多实例学习）的简单到难渐进式表示学习方法，用于高效处理全切片图像（WSIs）进行癌症预后预测，无需密集标注，可直接将千兆像素级WSIs转化为结果预测。

### 背景

计算病理学是一个新兴学科，旨在利用全切片图像量化形态异质性并开发癌症客观预后模型，但受千兆像素级输入的计算瓶颈和密集手动标注稀缺性的阻碍，当前方法常忽略多倍率WSIs上的细粒度信息和肿瘤微环境变异。

### 目的

开发一种高效处理WSIs的癌症预后预测方法，解决现有方法的局限性，特别是计算瓶颈和标注稀缺问题。

### 方法

提出DCMIL（dual-curriculum contrastive multi-instance learning）方法，这是一种简单到难渐进式表示学习技术，不依赖密集标注，可直接处理千兆像素级WSIs。

### 主要发现

在十二种癌症类型（5,954名患者，1,254万张图像块）上的实验显示，DCMIL优于标准WSI预后模型，能识别预后显著区域，提供不确定性估计，捕捉正常与肿瘤组织形态差异，并可能产生新生物学见解。

### 结论

DCMIL方法有效解决了计算病理学中的关键挑战，为癌症预后预测提供了强大工具，所有代码已在GitHub公开。

### 翻译

蓬勃发展的计算病理学学科展现出利用全切片图像（WSIs）量化形态异异性并为人类癌症开发客观预后模型的潜力。然而，千兆像素级输入的计算瓶颈和密集手动标注的稀缺性阻碍了其进展。当前方法常常忽略多倍率WSIs上的细粒度信息和肿瘤微环境的变异。在此，我们提出了一种简单到难渐进式表示学习，称为双课程对比多实例学习（DCMIL），以高效处理WSIs用于癌症预后。该模型不依赖密集标注，能够直接将千兆像素级WSIs转化为结果预测。在十二种癌症类型（5,954名患者，1,254万张图像块）上的大量实验表明，DCMIL优于标准的基于WSI的预后模型。此外，DCMIL能够识别细粒度的预后显著区域，提供稳健的实例不确定性估计，并捕捉正常组织和肿瘤组织之间的形态差异，有可能产生新的生物学见解。所有代码已在https://github.com/tuuuc/DCMIL公开。


### 论文摘要

The burgeoning discipline of computational pathology shows promise in harnessing whole slide images (WSIs) to quantify morphological heterogeneity and develop objective prognostic modes for human cancers. However, progress is impeded by the computational bottleneck of gigapixel-size inputs and the scarcity of dense manual annotations. Current methods often overlook fine-grained information across multi-magnification WSIs and variations in tumor microenvironments. Here, we propose an easy-to-hard progressive representation learning, termed dual-curriculum contrastive multi-instance learning (DCMIL), to efficiently process WSIs for cancer prognosis. The model does not rely on dense annotations and enables the direct transformation of gigapixel-size WSIs into outcome predictions. Extensive experiments on twelve cancer types (5,954 patients, 12.54 million tiles) demonstrate that DCMIL outperforms standard WSI-based prognostic models. Additionally, DCMIL identifies fine-grained prognosis-salient regions, provides robust instance uncertainty estimation, and captures morphological differences between normal and tumor tissues, with the potential to generate new biological insights. All codes have been made publicly accessible at https://github.com/tuuuc/DCMIL.

---

## 22. CausalVerse: Benchmarking Causal Representation Learning with Configurable High-Fidelity Simulations

**论文链接:** [http://arxiv.org/abs/2510.14049v2](http://arxiv.org/abs/2510.14049v2)

**作者:** Guangyi Chen, Yunlong Deng, Peiyuan Zhu, Yan Li, Yifan Shen, Zijian Li, Kun Zhang

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文引入了一个新的因果表示学习(CRL)基准，使用高保真模拟视觉数据，包含约20万张图像和300万视频帧，涵盖四个领域的24个子场景，提供对底层因果结构的灵活访问，评估了不同范式的代表性CRL方法，并提供了经验见解。

### 背景

因果表示学习旨在揭示数据生成过程并识别潜在的因果变量和关系，但其评估具有固有挑战性，因为需要已知的真实因果变量和因果结构。现有评估通常依赖简单的合成数据集或真实世界任务的下游性能，面临真实性和评估精度之间的两难困境。

### 目的

引入一个新的CRL基准，使用高保真模拟视觉数据，既保持真实的视觉复杂性，又能访问真实的因果生成过程，提供全面的测试平台以弥合严格评估和实际应用之间的差距。

### 方法

创建了一个包含约20万张图像和300万视频帧的数据集，涵盖四个领域(静态图像生成、动态物理模拟、机器人操作和交通情况分析)的24个子场景，从静态到动态设置，从简单到复杂结构，从单智能体到多智能体交互。提供对底层因果结构的灵活访问，允许用户修改或配置它们以符合CRL中的假设要求。

### 主要发现

该基准提供了一个全面的测试平台，有望弥合严格评估和实际应用之间的差距；评估了不同范式的代表性CRL方法；提供了经验见解，帮助实践者和新手选择或扩展适当的CRL框架以解决特定类型的现实问题。

### 结论

该基准有助于解决特定类型的现实问题，这些问题可以从CRL视角中受益；提供了项目页面和数据集的访问链接。

### 翻译

因果表示学习(CRL)旨在揭示数据生成过程并识别潜在的因果变量和关系，其评估由于需要已知的真实因果变量和因果结构而仍然具有固有挑战性。现有评估通常要么依赖简单的合成数据集，要么依赖真实世界任务的下游性能，普遍面临真实性和评估精度之间的两难困境。在本文中，我们引入了一个使用高保真模拟视觉数据的新CRL基准，这些数据既保持了真实的视觉复杂性，更重要的是，可以访问真实的因果生成过程。该数据集包含四个领域(静态图像生成、动态物理模拟、机器人操作和交通情况分析)中24个子场景的约20万张图像和300万视频帧。这些场景从静态到动态设置，从简单到复杂结构，从单智能体到多智能体交互，提供了一个全面的测试平台，有望弥合严格评估和实际应用之间的差距。此外，我们提供对底层因果结构的灵活访问，允许用户修改或配置它们以符合CRL中的假设要求，如可用的领域标签、时间依赖性或干预历史。利用这个基准，我们评估了不同范式的代表性CRL方法，并提供了经验见解，帮助实践者和新手选择或扩展适当的CRL框架，以适当解决可以从CRL视角中受益的特定类型的现实问题。欢迎访问我们的：项目页面：https://causal-verse.github.io/，数据集：https://huggingface.co/CausalVerse。


### 论文摘要

Causal Representation Learning (CRL) aims to uncover the data-generating process and identify the underlying causal variables and relations, whose evaluation remains inherently challenging due to the requirement of known ground-truth causal variables and causal structure. Existing evaluations often rely on either simplistic synthetic datasets or downstream performance on real-world tasks, generally suffering a dilemma between realism and evaluative precision. In this paper, we introduce a new benchmark for CRL using high-fidelity simulated visual data that retains both realistic visual complexity and, more importantly, access to ground-truth causal generating processes. The dataset comprises around 200 thousand images and 3 million video frames across 24 sub-scenes in four domains: static image generation, dynamic physical simulations, robotic manipulations, and traffic situation analysis. These scenarios range from static to dynamic settings, simple to complex structures, and single to multi-agent interactions, offering a comprehensive testbed that hopefully bridges the gap between rigorous evaluation and real-world applicability. In addition, we provide flexible access to the underlying causal structures, allowing users to modify or configure them to align with the required assumptions in CRL, such as available domain labels, temporal dependencies, or intervention histories. Leveraging this benchmark, we evaluated representative CRL methods across diverse paradigms and offered empirical insights to assist practitioners and newcomers in choosing or extending appropriate CRL frameworks to properly address specific types of real problems that can benefit from the CRL perspective. Welcome to visit our: Project page:https://causal-verse.github.io/, Dataset:https://huggingface.co/CausalVerse.

---

## 23. Attn-JGNN: Attention Enhanced Join-Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.15583v1](http://arxiv.org/abs/2510.15583v1)

**作者:** Jixin Zhang, Yong Lai

**发布时间:** 2025-10-17

### GPT解析

### 总结

提出了一种用于解决#SAT问题的注意力增强连接图神经网络(Attn-JGNN)模型，显著提高了求解准确性。

### 背景

#SAT问题是计算机科学中的重要问题，涉及计算布尔公式满足元的数量，现有神经网络方法可能存在求解精度不足的问题。

### 目的

提高解决#SAT问题的准确性，通过引入注意力机制来优化连接图神经网络模型。

### 方法

受迭代连接图传播算法启发，使用树分解将CNF公式编码为连接图，在连接图上进行迭代消息传递，通过学习分区函数近似模型数量，并在连接图的簇内和簇间应用注意力机制，使模型更关注关键变量和簇，减少冗余计算。

### 主要发现

注意力机制使Attn-JGNN能够在概率推理中更关注关键变量和簇，减少冗余计算，实验表明该模型比其他神经网络方法取得了更好的结果。

### 结论

Attn-JGNN模型通过结合注意力机制和连接图神经网络，有效提高了#SAT问题的求解准确性。

### 翻译

我们提出了一种用于解决#SAT问题的注意力增强连接图神经网络(Attn-JGNN)模型，显著提高了求解准确性。受迭代连接图传播算法启发，Attn-JGNN使用树分解将CNF公式编码为连接图，然后在连接图上进行迭代消息传递，最后通过学习分区函数来近似模型数量。为了进一步提高求解准确性，我们在连接图的簇内和簇间应用注意力机制，这使得Attn-JGNN能够在概率推理中更关注关键变量和簇，并减少冗余计算。最后，我们的实验表明，Attn-JGNN模型比其他神经网络方法取得了更好的结果。


### 论文摘要

We propose an Attention Enhanced Join-Graph Neural Networks(Attn-JGNN) model for solving #SAT problems, which significantly improves the solving accuracy. Inspired by the Iterative Join Graph Propagation (IJGP) algorithm, Attn-JGNN uses tree decomposition to encode the CNF formula into a join-graph, then performs iterative message passing on the join-graph, and finally approximates the model number by learning partition functions. In order to further improve the accuracy of the solution, we apply the attention mechanism in and between clusters of the join-graphs, which makes Attn-JGNN pay more attention to the key variables and clusters in probabilistic inference, and reduces the redundant calculation. Finally, our experiments show that our Attn-JGNN model achieves better results than other neural network methods.

---

## 24. Fault Cause Identification across Manufacturing Lines through Ontology-Guided and Process-Aware FMEA Graph Learning with LLMs

**论文链接:** [http://arxiv.org/abs/2510.15428v1](http://arxiv.org/abs/2510.15428v1)

**作者:** Sho Okazaki, Kohei Kaminishi, Takuma Fujiu, Yusheng Wang, Jun Ota

**发布时间:** 2025-10-17

### GPT解析

### 总结

本研究提出了一种过程感知框架，结合制造领域概念化与图神经网络推理，提高FMEA知识在不同制造生产线间的可重用性，有效解决了故障原因识别的挑战。

### 背景

自动化制造线中的故障原因识别具有挑战性，主要由于系统复杂性、频繁重新配置以及现有FMEA知识的有限可重用性。FMEA工作表包含宝贵的专家见解，但由于自然语言变异性、术语不一致和工艺差异，在异构生产线之间的重用受到阻碍。

### 目的

解决FMEA知识在不同生产线间重用的限制，提高故障原因识别的准确性和可靠性。

### 方法

提出一个过程感知框架，首先通过本体引导的大语言模型提取，将多个生产线的FMEA工作表转换为统一的知识图谱；其次，使用带有过程感知评分函数的关系图卷积网络学习尊重语义关系和顺序流程的嵌入；最后，使用链接预测推断和排序与目标生产线流程一致的候选故障原因。

### 主要发现

在汽车压力传感器装配线上的案例研究表明，所提出的方法优于最先进的检索增强生成基线（F1@20 = 0.267）和RGCN方法（0.400），实现了故障原因识别的最佳性能（0.523）。消融研究证实了LLM驱动的领域概念化和过程感知学习的贡献。

### 结论

所提出的框架显著提高了FMEA知识在异构生产线之间的可转移性，支持操作人员更可靠地诊断故障，为未来智能制造中的领域自适应LLM应用铺平道路。

### 翻译

在自动化生产线中，由于系统复杂性、频繁重新配置以及现有故障模式与影响分析知识的有限可重用性，故障原因识别具有挑战性。尽管FMEA工作表包含宝贵的专家见解，但由于自然语言变异性、术语不一致和工艺差异，它们在异构生产线之间的重用受到阻碍。为解决这些限制，本研究提出了一种过程感知框架，通过结合制造领域概念化与图神经网络推理，提高FMEA的可重用性。首先，通过本体引导的大语言模型提取，将多个生产线的FMEA工作表转换为统一的知识图谱，捕获领域概念如动作、状态、组件和参数。其次，使用带有过程感知评分函数的关系图卷积网络学习尊重语义关系和顺序流程的嵌入。最后，使用链接预测来推断和排序与目标生产线流程一致的候选故障原因。在汽车压力传感器装配线上的案例研究表明，所提出的方法优于最先进的检索增强生成基线（F1@20 = 0.267）和RGCN方法（0.400），实现了故障原因识别的最佳性能（0.523）。消融研究证实了LLM驱动的领域概念化和过程感知学习的贡献。这些结果表明，所提出的框架显著提高了FMEA知识在异构生产线之间的可转移性，从而支持操作人员更可靠地诊断故障，并为未来智能制造中领域自适应LLM的应用铺平道路。


### 论文摘要

Fault cause identification in automated manufacturing lines is challenging due to the system's complexity, frequent reconfigurations, and the limited reusability of existing Failure Mode and Effects Analysis (FMEA) knowledge. Although FMEA worksheets contain valuable expert insights, their reuse across heterogeneous lines is hindered by natural language variability, inconsistent terminology, and process differences. To address these limitations, this study proposes a process-aware framework that enhances FMEA reusability by combining manufacturing-domain conceptualization with graph neural network (GNN) reasoning. First, FMEA worksheets from multiple manufacturing lines are transformed into a unified knowledge graph through ontology-guided large language model (LLM) extraction, capturing domain concepts such as actions, states, components, and parameters. Second, a Relational Graph Convolutional Network (RGCN) with the process-aware scoring function learns embeddings that respect both semantic relationships and sequential process flows. Finally, link prediction is employed to infer and rank candidate fault causes consistent with the target line's process flow.   A case study on automotive pressure sensor assembly lines demonstrates that the proposed method outperforms a state-of-the-art retrieval-augmented generation (RAG) baseline (F1@20 = 0.267) and an RGCN approach (0.400), achieving the best performance (0.523) in fault cause identification. Ablation studies confirm the contributions of both LLM-driven domain conceptualization and process-aware learning. These results indicate that the proposed framework significantly improves the transferability of FMEA knowledge across heterogeneous lines, thereby supporting operators in diagnosing failures more reliably and paving the way for future domain-adaptive LLM applications in smart manufacturing.

---

## 25. Geometric Mixture Models for Electrolyte Conductivity Prediction

**论文链接:** [http://arxiv.org/abs/2510.15403v1](http://arxiv.org/abs/2510.15403v1)

**作者:** Anyi Li, Jiacheng Cen, Songyou Li, Mingze Li, Yang Yu, Wenbing Huang

**发布时间:** 2025-10-17

### GPT解析

### 总结

本研究提出了GeoMix框架，用于准确预测电解质系统中的离子电导率，解决了缺乏高质量基准和混合系统几何建模不足的挑战。

### 背景

电解质系统中离子电导率的准确预测对科学和技术应用至关重要，但当前研究面临两个基本挑战：(1)缺乏高质量标准化基准，(2)对混合系统中几何结构和分子间相互作用的建模不足。

### 目的

解决现有研究的局限性，建立新的电解质研究基准，并提供通用的几何学习框架以推进混合系统建模。

### 方法

重新组织和增强CALiSol和DiffMix电解质数据集，加入分子的几何图表示；提出GeoMix框架，保持Set-SE(3)等变性；设计几何交互网络(GIN)作为专门为分子间几何消息传递的等变模块。

### 主要发现

GeoMix在两个数据集上都一致优于多种基线方法，证明了跨分子几何相互作用和等变消息传递对准确属性预测的重要性。

### 结论

该工作不仅为电解质研究建立了新基准，还提供了通用的几何学习框架，可应用于能源材料、药物开发等领域的混合系统建模。

### 翻译

电解质系统中离子电导率的准确预测对推进众多科学和技术应用至关重要。尽管已取得显著进展，但当前研究面临两个基本挑战：(1)缺乏高质量标准化基准，(2)对混合系统中几何结构和分子间相互作用的建模不足。为解决这些局限性，我们首先通过加入分子的几何图表示来重新组织和增强CALiSol和DiffMix电解质数据集。然后，我们提出了GeoMix，一种新型几何感知框架，保留了混合系统的重要但具有挑战性的Set-SE(3)等变性特性。GeoMix的核心是几何交互网络(GIN)，一个专门为分子间几何消息传递设计的等变模块。全面的实验证明，GeoMix在两个数据集上都一致优于多种基线方法(包括MLPs、GNNs和几何GNNs)，验证了跨分子几何相互作用和等变消息传递对准确属性预测的重要性。这项工作不仅为电解质研究建立了新基准，还提供了通用的几何学习框架，推进了能源材料、药物开发等领域中混合系统的建模。


### 论文摘要

Accurate prediction of ionic conductivity in electrolyte systems is crucial for advancing numerous scientific and technological applications. While significant progress has been made, current research faces two fundamental challenges: (1) the lack of high-quality standardized benchmarks, and (2) inadequate modeling of geometric structure and intermolecular interactions in mixture systems. To address these limitations, we first reorganize and enhance the CALiSol and DiffMix electrolyte datasets by incorporating geometric graph representations of molecules. We then propose GeoMix, a novel geometry-aware framework that preserves Set-SE(3) equivariance-an essential but challenging property for mixture systems. At the heart of GeoMix lies the Geometric Interaction Network (GIN), an equivariant module specifically designed for intermolecular geometric message passing. Comprehensive experiments demonstrate that GeoMix consistently outperforms diverse baselines (including MLPs, GNNs, and geometric GNNs) across both datasets, validating the importance of cross-molecular geometric interactions and equivariant message passing for accurate property prediction. This work not only establishes new benchmarks for electrolyte research but also provides a general geometric learning framework that advances modeling of mixture systems in energy materials, pharmaceutical development, and beyond.

---

## 26. Backdoor or Manipulation? Graph Mixture of Experts Can Defend Against Various Graph Adversarial Attacks

**论文链接:** [http://arxiv.org/abs/2510.15333v1](http://arxiv.org/abs/2510.15333v1)

**作者:** Yuyuan Feng, Bin Ma, Enyan Dai

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种基于专家混合(MoE)架构的统一框架，用于防御图神经网络中的多种对抗性攻击，包括后门攻击、边操纵和节点注入攻击。

### 背景

研究表明图神经网络容易受到多种对抗性攻击，包括操纵、节点注入和后门攻击。然而，现有防御方法通常只针对单一类型的攻击，缺乏能够同时防御多种威胁的统一方法。

### 目的

设计一个可扩展的统一框架，能够同时防御后门、边操纵和节点注入等多种图对抗攻击。

### 方法

利用专家混合(MoE)架构的灵活性，提出基于互信息的逻辑多样性损失，鼓励专家关注不同邻域结构；引入鲁棒性感知的路由器，识别扰动模式并将受扰节点路由到鲁棒专家。

### 主要发现

在各种对抗设置下的广泛实验表明，该方法在抵御多种图对抗攻击方面表现出优越的鲁棒性。

### 结论

所提出的框架能够有效防御多种图对抗攻击，为图神经网络的安全提供了统一解决方案。

### 翻译

大量研究已经强调了图神经网络(GNNs)容易受到对抗性攻击的脆弱性，包括操纵、节点注入以及最近出现的后门攻击威胁。然而，现有的防御方法通常只关注单一类型的攻击，缺乏同时防御多种威胁的统一方法。在本工作中，我们利用专家混合(MoE)架构的灵活性，设计了一个可扩展的统一框架，用于防御后门、边操纵和节点注入攻击。具体来说，我们提出了一种基于互信息的逻辑多样性损失，鼓励各个专家在决策过程中关注不同的邻域结构，从而确保在局部结构受到扰动时，有足够数量的专家不受影响。此外，我们引入了一种鲁棒性感知的路由器，能够识别扰动模式并将受扰动的节点自适应地路由到相应的鲁棒专家。在各种对抗设置下进行的广泛实验表明，我们的方法在抵御多种图对抗攻击方面持续表现出优越的鲁棒性。


### 论文摘要

Extensive research has highlighted the vulnerability of graph neural networks (GNNs) to adversarial attacks, including manipulation, node injection, and the recently emerging threat of backdoor attacks. However, existing defenses typically focus on a single type of attack, lacking a unified approach to simultaneously defend against multiple threats. In this work, we leverage the flexibility of the Mixture of Experts (MoE) architecture to design a scalable and unified framework for defending against backdoor, edge manipulation, and node injection attacks. Specifically, we propose an MI-based logic diversity loss to encourage individual experts to focus on distinct neighborhood structures in their decision processes, thus ensuring a sufficient subset of experts remains unaffected under perturbations in local structures. Moreover, we introduce a robustness-aware router that identifies perturbation patterns and adaptively routes perturbed nodes to corresponding robust experts. Extensive experiments conducted under various adversarial settings demonstrate that our method consistently achieves superior robustness against multiple graph adversarial attacks.

---

## 27. Spatiotemporal Traffic Prediction in Distributed Backend Systems via Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.15215v1](http://arxiv.org/abs/2510.15215v1)

**作者:** Zhimin Qiu, Feng Liu, Yuxiao Wang, Chenrui Hu, Ziyu Cheng, Di Wu

**发布时间:** 2025-10-17

### GPT解析

### 总结

该论文提出了一种基于图神经网络的建模方法，用于解决分布式后端系统中的交通预测问题。通过将系统抽象为图结构，结合图卷积机制和门控循环结构，实现了空间结构与时间演化的联合建模，显著提高了预测的准确性和鲁棒性。

### 背景

传统模型在捕捉分布式后端系统中的复杂依赖性和动态特征方面存在局限性，需要更先进的建模方法来提高交通预测的准确性。

### 目的

提出一种基于图神经网络的建模方法，克服传统模型的限制，提高分布式后端系统中交通预测的准确性和鲁棒性。

### 方法

将系统抽象为包含节点和边的图，节点特征表示流量和资源状态，邻接关系描述服务交互；使用图卷积机制实现节点特征的多阶传播和聚合；采用门控循环结构动态建模历史序列；通过时空联合建模模块融合图表示与时间依赖性；使用解码器生成未来流量预测；使用均方误差进行模型训练。

### 主要发现

所提出的方法在不同预测范围和模型深度下实现了稳定的性能和低误差，显著提高了分布式后端系统中交通预测的准确性和鲁棒性。

### 结论

图神经网络在复杂系统建模中具有巨大潜力，能够有效捕捉分布式后端系统中的复杂依赖性和动态特征。

### 翻译

本文解决了分布式后端系统中的交通预测问题，并提出了一种基于图神经网络的建模方法，以克服传统模型在捕捉复杂依赖性和动态特征方面的局限性。该系统被抽象为一个包含节点和边的图，其中节点特征表示流量和资源状态，邻接关系描述服务交互。图卷积机制实现了节点特征的多阶传播和聚合，而门控循环结构动态建模历史序列，从而将空间结构与时间演化相结合。时空联合建模模块进一步融合图表示与时间依赖性，解码器生成未来流量预测。模型使用均方误差进行训练，以最小化与实际值的偏差。基于公共分布式系统日志的实验构建了节点特征、拓扑和序列的组合输入，并使用MSE、RMSE、MAE和MAPE指标将所提出的方法与主流基线进行比较。结果表明，所提出的方法在不同预测范围和模型深度下实现了稳定的性能和低误差，显著提高了分布式后端系统中交通预测的准确性和鲁棒性，验证了图神经网络在复杂系统建模中的潜力。


### 论文摘要

This paper addresses the problem of traffic prediction in distributed backend systems and proposes a graph neural network based modeling approach to overcome the limitations of traditional models in capturing complex dependencies and dynamic features. The system is abstracted as a graph with nodes and edges, where node features represent traffic and resource states, and adjacency relations describe service interactions. A graph convolution mechanism enables multi order propagation and aggregation of node features, while a gated recurrent structure models historical sequences dynamically, thus integrating spatial structures with temporal evolution. A spatiotemporal joint modeling module further fuses graph representation with temporal dependency, and a decoder generates future traffic predictions. The model is trained with mean squared error to minimize deviations from actual values. Experiments based on public distributed system logs construct combined inputs of node features, topology, and sequences, and compare the proposed method with mainstream baselines using MSE, RMSE, MAE, and MAPE. Results show that the proposed method achieves stable performance and low error across different prediction horizons and model depths, significantly improving the accuracy and robustness of traffic forecasting in distributed backend systems and verifying the potential of graph neural networks in complex system modeling.

---

## 28. Structural Generalization for Microservice Routing Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.15210v1](http://arxiv.org/abs/2510.15210v1)

**作者:** Chenrui Hu, Ziyu Cheng, Di Wu, Yuxiao Wang, Feng Liu, Zhimin Qiu

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种基于图神经网络的端到端优化框架，用于微服务系统中的智能路由，旨在提高复杂拓扑结构下路由决策效率和整体系统性能。

### 背景

微服务系统中的智能路由问题，需要处理复杂拓扑结构下的路由决策和系统性能优化。

### 目的

提高路由决策效率和整体系统性能，特别是在复杂拓扑结构下。更好地评估路径质量，捕获服务通信中的不稳定性和瓶颈风险。

### 方法

将微服务之间的调用关系建模为图，服务节点和通信链路作为图的节点和边；使用多维特征作为输入，包括节点状态、链路延迟和调用频率；采用多层图神经网络进行高阶信息聚合和结构建模；模型为每个候选服务路径输出分数，用于指导动态路由决策；引入边感知注意力机制提高模型评估路径质量的能力；在不同网络深度、拓扑密度和服务规模下进行系统性分析。

### 主要发现

提出的方法在多个关键指标上优于现有主流策略；能够有效处理高度动态和并发的微服务环境；展示了强大的性能、鲁棒性和结构泛化能力。

### 结论

基于图神经网络的端到端优化框架在微服务系统智能路由方面表现优异，能够提高路由决策效率和系统整体性能。

### 翻译

这篇论文专注于微服务系统中的智能路由，并提出了一种基于图神经网络的端到端优化框架。目标是提高复杂拓扑结构下路由决策效率和整体系统性能。该方法将微服务之间的调用关系建模为图。在该图中，服务节点和通信链路被视为图的节点和边。使用节点状态、链路延迟和调用频率等多维特征作为输入。采用多层图神经网络执行高阶信息聚合和结构建模。模型为每个候选服务路径输出分数。然后使用这些分数来指导动态路由决策。为了提高模型评估路径质量的能力，引入了边感知注意力机制。该机制帮助模型更准确地捕获服务通信中的不稳定性和瓶颈风险。论文还对该模型在不同网络深度、拓扑密度和服务规模下的性能进行了系统性分析。从路由准确性、预测误差和系统稳定性等方面评估了该方法的有效性。实验结果表明，该方法在多个关键指标上优于现有的主流策略。它能够有效处理高度动态和并发的微服务环境，并表现出强大的性能、鲁棒性和结构泛化能力。


### 论文摘要

This paper focuses on intelligent routing in microservice systems and proposes an end-to-end optimization framework based on graph neural networks. The goal is to improve routing decision efficiency and overall system performance under complex topologies. The method models invocation relationships among microservices as a graph. In this graph, service nodes and communication links are treated as graph nodes and edges. Multi-dimensional features such as node states, link latency, and call frequency are used as input. A multi-layer graph neural network is employed to perform high-order information aggregation and structural modeling. The model outputs a score for each candidate service path. These scores are then used to guide dynamic routing decisions. To improve the model's ability to assess path quality, an edge-aware attention mechanism is introduced. This mechanism helps the model capture instability and bottleneck risks in service communications more accurately. The paper also conducts a systematic analysis of the model's performance under different network depths, topology densities, and service scales. It evaluates the effectiveness of the method in terms of routing accuracy, prediction error, and system stability. Experimental results show that the proposed method outperforms existing mainstream strategies across multiple key metrics. It handles highly dynamic and concurrent microservice environments effectively and demonstrates strong performance, robustness, and structural generalization.

---

## 29. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs

**论文链接:** [http://arxiv.org/abs/2510.15188v1](http://arxiv.org/abs/2510.15188v1)

**作者:** Ahmed Aly, Essam Mansour, Amr Youssef

**发布时间:** 2025-10-16

### GPT解析

### 总结

OCR-APT是一种创新的APT检测系统，通过结合图神经网络和大型语言模型，实现了更准确、更可解释的攻击检测和故事重建，解决了现有系统高误报率和粗粒度警报的问题。

### 背景

高级持续性威胁(APTs)是隐蔽的网络攻击，通常能逃避系统级审计日志的检测。现有系统将异常检测应用于这些图，但常有高误报率和粗粒度警报，且依赖节点属性导致虚假关联，降低了检测的鲁棒性和可靠性。

### 目的

开发能够生成准确、类人的完整攻击叙述的系统，解决高误报率和粗粒度警报问题，提供更鲁棒的异常检测，并生成可解释的最终报告。

### 方法

引入OCR-APT系统，使用图神经网络(GNNs)进行子图异常检测，学习节点周围的行为模式而非脆弱属性；然后使用大型语言模型(LLMs)迭代处理检测到的子图重建多阶段攻击故事，每个阶段在继续前进行验证以减少幻觉。

### 主要发现

在DARPA TC3、OpTC和NODLINK数据集上的评估显示，OCR-APT在检测准确率和警报可解释性方面优于最先进的系统，且能重建全面捕获攻击故事的类人报告。

### 结论

OCR-APT系统有效解决了现有APT检测系统的局限性，通过结合GNNs和LLMs，提供了更准确和可解释的APT检测与攻击故事重建能力。

### 翻译

高级持续性威胁(APTs)是隐蔽的网络攻击，通常能逃避系统级审计日志中的检测。来源图将这些日志建模为连接的实体和事件，揭示了线性日志表示中遗漏的关系。现有系统将这些图应用于异常检测，但常常遭受高误报率和粗粒度警报的困扰。它们对文件路径或IP等节点属性的依赖导致虚假关联，降低了检测的鲁棒性和可靠性。为了完全理解攻击的进展和影响，安全分析师需要能够生成准确、类人的完整攻击叙述的系统。为解决这些挑战，我们引入了OCR-APT，一个用于APT检测和类人攻击故事重建的系统。OCR-APT使用图神经网络(GNNs)进行子图异常检测，学习节点周围的行为模式，而不是依赖文件路径或IP等脆弱属性。这种方法导致更鲁棒的异常检测。然后，它使用大型语言模型(LLMs)迭代处理检测到的子图，重建多阶段攻击故事。每个阶段在继续前进行验证，减少幻觉并确保可解释的最终报告。我们在DARPA TC3、OpTC和NODLINK数据集上的评估表明，OCR-APT在检测准确率和警报可解释性方面优于最先进的系统。此外，OCR-APT重建的类人报告全面捕获了攻击故事。


### 论文摘要

Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.

---

## 30. A Comprehensive Evaluation of Graph Neural Networks and Physics Informed Learning for Surrogate Modelling of Finite Element Analysis

**论文链接:** [http://arxiv.org/abs/2510.15750v1](http://arxiv.org/abs/2510.15750v1)

**作者:** Nayan Kumar Singh

**发布时间:** 2025-10-16

**备注:** 14 pages, 6 figures, 5 tables. Code available  at:https://github.com/SinghNayanKumar/DL-surrogate-modelling

### GPT解析

### 总结

本研究评估了图神经网络(GNNs)和3D U-Nets作为参数化I型梁有限元分析(FEA)替代模型的性能，引入了基于Navier-Cauchy方程的物理信息神经网络(PINN)框架，并证明课程学习策略对稳定训练至关重要。研究发现GNNs整体优于U-Net，MPNN和图变换器达到最高精度，PINN框架显著提高了泛化能力。

### 背景

有限元分析(FEA)是产品设计生命周期中不可或缺的部分，但计算成本高，不适合许多设计优化问题。深度学习模型可能是一个很好的解决方案。

### 目的

评估图神经网络(GNNs)和3D U-Nets作为FEA替代模型的性能，引入物理信息神经网络(PINN)框架，研究课程学习策略对训练稳定性的影响。

### 方法

使用图神经网络(GNNs)和3D U-Nets作为参数化I型梁FEA的替代模型，实现基于Navier-Cauchy方程的PINN框架，采用课程学习策略：先在数据上预训练，再进行物理信息微调。

### 主要发现

GNNs整体上优于U-Net；最差的GNN模型(GCN)相对L2误差为8.7%，而最好的U-Net模型得分为13.0%；MPNN和图变换器达到最高精度，相对L2得分分别为3.5%和2.6%；PINN框架显著提高了泛化能力，在高信号任务上减少了高达11.3%的误差；图变换器是最准确的模型但推理速度较慢；MPNN-PINN模型在性能和效率之间提供了最佳平衡。

### 结论

图神经网络是有限元分析的有效替代方案；课程学习策略对稳定训练至关重要；物理信息神经网络框架提高了模型的泛化能力；MPNN-PINN模型在预测性能、模型大小和推理速度之间取得了良好平衡。

### 翻译

虽然有限元分析(FEA)是产品设计生命周期中不可或缺的部分，但分析计算成本高，使其不适合许多设计优化问题。深度学习模型可以是一个很好的解决方案。然而，选择能够以高精度模拟FEA的架构是一个挑战。本文提出了对图神经网络(GNNs)和3D U-Nets作为参数化I型梁FEA替代模型的综合评估。我们引入了一个由Navier-Cauchy方程控制的物理信息神经网络(PINN)框架，以强制执行物理定律。关键的是，我们证明课程学习策略——先在数据上预训练，再进行物理信息微调——对于稳定训练至关重要。我们的结果表明，GNNs从根本上优于U-Net。即使在GNNs中最差的GCN框架也实现了8.7%的相对L2误差，而在U-Net中最好的框架(使用高分辨率数据训练的带注意力机制的U-Net)获得了13.0%的得分。在基于图的架构中，消息传递神经网络(MPNN)和图变换器达到了最高的准确性，分别实现了3.5%和2.6%的相对L2得分。包含物理基本定律(PINN)显著提高了泛化能力，在高信号任务上减少了高达11.3%的误差。虽然图变换器是最准确的模型，但在推理时比第二好的模型MPNN-PINN慢37.5%。PINN增强的MPNN(MPNN-PINN)提供了最实用的解决方案。它在预测性能、模型大小和推理速度之间提供了良好的平衡。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决有限元分析(FEA)计算成本高、不适合实时应用和设计优化的问题。这个问题在现实中很重要，因为FEA虽然广泛应用于产品设计，但计算时间长，难以用于需要快速反馈的场景，如数字孪生或多次迭代的设计优化。传统替代方法如降阶模型存在线性假设、侵入性要求等局限性，难以处理复杂的非线性物理现象。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了FEA的计算瓶颈和传统替代方法的局限性，认识到需要更灵活、非线性、非侵入性的解决方案。他们借鉴了现有工作中的图神经网络处理网格数据的方法，以及物理信息神经网络(PINN)嵌入物理定律的思路。在此基础上，作者设计了系统性比较多种GNN架构的方案，并创新性地采用课程学习策略来稳定PINN训练，通过两阶段训练(数据驱动预训练+物理信息微调)解决了训练不稳定的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用图神经网络(GNN)直接处理FEA的非结构化网格，并通过物理信息神经网络(PINN)嵌入物理定律来提高泛化能力。整体流程包括：1)使用gmsh和DOLFINx生成I梁的FEA数据，创建低信号和高信号数据集；2)实现多种GNN架构(GCN、GAT、MPNN、图变换器)和3D U-Net基线；3)设计节点特征编码和消息传递机制；4)将纳维-柯西方程嵌入损失函数，采用课程学习和损失权重退火稳定训练；5)使用多种指标评估模型性能和计算效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)系统性比较多种GNN架构，证明GNN在FEA替代模型中的优越性；2)成功将物理定律嵌入GNN，显著提高泛化能力；3)提出稳健的PINN训练策略，通过课程学习和损失权重退火解决训练不稳定问题；4)进行全面的性能-效率分析，识别最优实用架构。相比之前工作，本文提供了更全面的架构比较，解决了PINN训练不稳定问题，并考虑了实际部署中的模型大小和推理速度等实用因素。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统比较多种图神经网络架构并成功集成物理信息学习，为有限元分析提供了一个高效、准确且泛化能力强的替代模型，同时提出了稳定的训练策略和实用的部署指南。'}


### 论文摘要

Although Finite Element Analysis (FEA) is an integral part of the product design lifecycle, the analysis is computationally expensive, making it unsuitable for many design optimization problems. The deep learning models can be a great solution. However, selecting the architecture that emulates the FEA with great accuracy is a challenge. This paper presents a comprehensive evaluation of graph neural networks (GNNs) and 3D U-Nets as surrogates for FEA of parametric I-beams. We introduce a Physics-Informed Neural Network (PINN) framework, governed by the Navier Cauchy equations, to enforce physical laws. Crucially, we demonstrate that a curriculum learning strategy, pretraining on data followed by physics informed fine tuning, is essential for stabilizing training. Our results show that GNNs fundamentally outperform the U-Net. Even the worst performer among GNNs, the GCN framework, achieved a relative L2 error of 8.7% while the best framework among U Net, U Net with attention mechanism trained on high resolution data, achieved 13.0% score. Among the graph-based architectures, the Message Passing Neural Networks (MPNN) and Graph Transformers achieved the highest accuracy, achieving a relative L2 score of 3.5% and 2.6% respectively. The inclusion of physics fundamental laws (PINN) significantly improved the generalization, reducing error by up to 11.3% on high-signal tasks. While the Graph Transformer is the most accurate model, it is more 37.5% slower during inference when compared to second best model, MPNN PINN. The PINN enhanced MPNN (MPNN PINN) provides the most practical solution. It offers a good compromise between predictive performance, model size, and inference speed.

---

## 31. Leveraging Teleconnections with Physics-Informed Graph Attention Networks for Long-Range Extreme Rainfall Forecasting in Thailand

**论文链接:** [http://arxiv.org/abs/2510.12328v3](http://arxiv.org/abs/2510.12328v3)

**作者:** Kiattikun Chobtham, Kanoksri Sarinnapakorn, Kritanai Torsri, Prattana Deeprasertkul, Jirawan Kamma

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了一种结合物理信息的图神经网络与极值分析技术，用于改进泰国地区的站点降雨预测，特别是在极端事件预报方面取得了显著成果。

### 背景

准确的降雨预报，尤其是极端事件的预报，在气候学和地球系统中仍是一个重大挑战。传统方法在泰国地区的站点降雨预测中存在局限性。

### 目的

开发一种新型预测模型，提高泰国地区站点降雨预测的准确性，特别是针对极端事件的预测能力，并提供高分辨率地图以支持长期水资源管理决策。

### 方法

结合物理信息的图神经网络与极值分析技术，利用站点图的表示捕捉复杂时空模式，通过遥相关提供可解释性；预处理影响区域降雨的气候指标；应用基于图注意力网络和长短期记忆网络的模型；使用空间季节感知广义帕累托分布方法进行阈值超限映射以解决极端问题。

### 主要发现

实验证明该方法在大多数地区优于成熟基线模型，包括易发生极端事件的区域；与最先进技术保持强竞争力；与业务预报系统SEAS5相比，显著改进了极端事件的预测；能够提供支持决策的高分辨率地图。

### 结论

所提出的方法是改进降雨预测特别是极端事件预测的实用增强工具，可为长期水资源管理提供决策支持。

### 翻译

准确的降雨预报，特别是对极端事件的预报，在气候学和地球系统中仍然是一个重大挑战。本文提出了结合物理信息的图神经网络与极值分析技术的新方法，以改进泰国地区的站点降雨预测。该模型利用站点图的表示来捕捉复杂的时空模式，并通过遥相关提供可解释性。我们预处理可能影响区域降雨的相关气候指标。提出的基于图注意力网络和长短期记忆网络的模型，使用简单地形降水物理公式推导的初始边特征应用注意力机制，嵌入随后由LSTM层处理。为解决极端问题，我们使用新颖的空间季节感知广义帕累托分布方法进行阈值超限映射，克服了传统机器学习模型的局限性。实验证明，我们的方法在大多数地区都优于成熟的基线模型，包括易发生极端事件的区域，并与最先进技术保持强竞争力。与业务预报系统SEAS5相比，我们的实际应用改进了极端事件的预测，并提供了实用的增强功能，可以生成支持长期水资源管理决策的高分辨率地图。


### 论文摘要

Accurate rainfall forecasting, particularly for extreme events, remains a significant challenge in climatology and the Earth system. This paper presents novel physics-informed Graph Neural Networks (GNNs) combined with extreme-value analysis techniques to improve gauge-station rainfall predictions across Thailand. The model leverages a graph-structured representation of gauge stations to capture complex spatiotemporal patterns, and it offers explainability through teleconnections. We preprocess relevant climate indices that potentially influence regional rainfall. The proposed Graph Attention Network with Long Short-Term Memory (Attention-LSTM) applies the attention mechanism using initial edge features derived from simple orographic-precipitation physics formulation. The embeddings are subsequently processed by LSTM layers. To address extremes, we perform Peak-Over-Threshold (POT) mapping using the novel Spatial Season-aware Generalized Pareto Distribution (GPD) method, which overcomes limitations of traditional machine-learning models. Experiments demonstrate that our method outperforms well-established baselines across most regions, including areas prone to extremes, and remains strongly competitive with the state of the art. Compared with the operational forecasting system SEAS5, our real-world application improves extreme-event prediction and offers a practical enhancement to produce high-resolution maps that support decision-making in long-term water management.

---

## 32. FreqPDE: Rethinking Positional Depth Embedding for Multi-View 3D Object Detection Transformers

**论文链接:** [http://arxiv.org/abs/2510.15385v1](http://arxiv.org/abs/2510.15385v1)

**作者:** Haisheng Su, Junjie Zhang, Feixiang Song, Sanping Zhou, Wei Wu, Nanning Zheng, Junchi Yan

**发布时间:** 2025-10-17

**备注:** Accepted to ICCV2025

### GPT解析

### 总结

本文提出了一种名为FreqPDE的新方法，用于从多视图2D图像中准确检测3D物体，解决了现有方法中深度预测质量不佳的问题。

### 背景

从多视图2D图像中准确检测3D物体是自动驾驶领域的一项具有挑战性但至关重要的任务。当前方法依赖深度预测来恢复空间信息，但存在深度不连续和小物体不清晰等问题。

### 目的

开发一种能够提供更准确深度信息的方法，解决现有深度预测中的边界不连续和小物体不清晰问题，并考虑跨视图一致性和尺度不变性。

### 方法

提出频率感知位置深度嵌入（FreqPDE），包含三个主要模块：频率感知空间金字塔编码器（FSPE）构建特征金字塔；跨视图尺度不变深度预测器（CSDP）估计像素级深度分布；位置深度编码器（PDE）生成3D深度感知特征。同时采用混合深度监督进行互补深度学习。

### 主要发现

现有深度预测方法存在物体边界深度不连续和小物体不清晰的问题，主要由投影点稀疏监督和高级图像特征使用引起。跨视图一致性和尺度不变性在先前方法中被忽视。

### 结论

在nuScenes数据集上的广泛实验证明了所提出FreqPDE方法的有效性和优越性，能够显著提升3D物体检测的准确性。

### 翻译

从多视图2D图像中准确检测3D物体是自动驾驶领域一项具有挑战性但至关重要的任务。当前方法通过整合深度预测来恢复物体查询解码的空间信息，这需要在训练阶段使用LiDAR点进行显式监督。然而，预测的深度质量仍然不理想，如物体边界深度不连续和小物体不清晰，主要由投影点的稀疏监督和使用高级图像特征进行深度预测引起。此外，先前的方法也忽视了跨视图一致性和尺度不变性。本文引入了频率感知位置深度嵌入（FreqPDE）来为2D图像特征赋予空间信息，用于3D检测transformer解码器，这通过三个主要模块实现：频率感知空间金字塔编码器（FSPE）结合不同级别的高频边缘线索和低级语义构建特征金字塔；跨视图尺度不变深度预测器（CSDP）使用跨视图和高效通道注意力机制估计像素级深度分布；位置深度编码器（PDE）结合2D图像特征和3D位置嵌入，为查询解码生成3D深度感知特征。同时采用混合深度监督，从度量和分布方面进行互补深度学习。在nuScenes数据集上进行的广泛实验证明了所提出方法的有效性和优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从多视角2D图像中准确检测3D物体时深度预测质量不佳的问题，包括物体边界深度不连续、小物体不清晰等。这个问题在自动驾驶领域至关重要，因为准确的3D物体检测是确保自动驾驶系统安全感知周围环境的关键，而仅使用摄像头的方法比基于LiDAR的方法成本更低，但恢复3D空间信息是一个具有挑战性的病态问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前深度预测方法的三个主要缺陷：仅使用高级图像特征导致细节丢失、投影点云稀疏监督导致学习不完整、以及忽略了跨视图一致性和尺度不变性。基于这些问题，作者设计了FreqPDE方法，包含三个核心模块：频率感知空间金字塔编码器(FSPE)、跨视图尺度不变深度预测器(CSDP)和位置深度编码器(PDE)。该方法借鉴了现有工作，如BEVFormer和StreamPETR等基于Transformer的3D检测方法，以及BEVDepth等深度预测方法，同时引入了FreqFusion等频率域学习的方法，但进行了创新性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用频率感知的位置深度嵌入为2D视觉特征提供高质量的空间信息，通过结合高频边缘细节和低频全局语义提高深度预测质量，并采用混合深度监督进行互补学习。整体流程包括：1)FSPE模块构建特征金字塔，结合不同级别的高频边缘线索和低频语义；2)CSDP模块进行分层深度预测，使用跨视图注意力和高效通道注意力确保一致性和尺度不变性；3)PDE模块结合2D图像特征和3D位置嵌入生成深度感知特征；4)采用混合深度监督，结合稀疏LiDAR和密集伪深度图；5)使用Transformer解码器生成最终检测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将频率域信息引入多视图3D检测，同时利用高频和低频信息；2)FSPE模块通过低频语义提取和高频边界增强保留更多细节；3)CSDP模块引入跨视图注意力和相机感知通道注意力解决一致性和尺度不变性问题；4)混合深度监督结合稀疏LiDAR和密集伪深度图进行互补学习。相比之前工作，不同之处在于：之前方法主要使用单一频率信息，忽略了跨视图一致性和尺度不变性，且监督方式单一；而FreqPDE同时利用高频和低频信息，解决了跨视图一致性和尺度不变性问题，并引入了混合深度监督。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FreqPDE通过引入频率感知的位置深度嵌入和混合深度监督，显著提高了多视角3D物体检测的准确性，特别是在处理远距离和小物体方面。'}


### 论文摘要

Detecting 3D objects accurately from multi-view 2D images is a challenging yet essential task in the field of autonomous driving. Current methods resort to integrating depth prediction to recover the spatial information for object query decoding, which necessitates explicit supervision from LiDAR points during the training phase. However, the predicted depth quality is still unsatisfactory such as depth discontinuity of object boundaries and indistinction of small objects, which are mainly caused by the sparse supervision of projected points and the use of high-level image features for depth prediction. Besides, cross-view consistency and scale invariance are also overlooked in previous methods. In this paper, we introduce Frequency-aware Positional Depth Embedding (FreqPDE) to equip 2D image features with spatial information for 3D detection transformer decoder, which can be obtained through three main modules. Specifically, the Frequency-aware Spatial Pyramid Encoder (FSPE) constructs a feature pyramid by combining high-frequency edge clues and low-frequency semantics from different levels respectively. Then the Cross-view Scale-invariant Depth Predictor (CSDP) estimates the pixel-level depth distribution with cross-view and efficient channel attention mechanism. Finally, the Positional Depth Encoder (PDE) combines the 2D image features and 3D position embeddings to generate the 3D depth-aware features for query decoding. Additionally, hybrid depth supervision is adopted for complementary depth learning from both metric and distribution aspects. Extensive experiments conducted on the nuScenes dataset demonstrate the effectiveness and superiority of our proposed method.

---

## 33. ERNet: Efficient Non-Rigid Registration Network for Point Sequences

**论文链接:** [http://arxiv.org/abs/2510.15800v1](http://arxiv.org/abs/2510.15800v1)

**作者:** Guangzhao He, Yuxi Xiao, Zhen Xu, Xiaowei Zhou, Sida Peng

**发布时间:** 2025-10-17

**备注:** Accepted to ICCV 2025. Project Page: https://guangzhaohe.com/ernet

### GPT解析

### 总结

论文提出了一种名为ERNet的高效前馈模型，用于解决将物体形状注册到经历非刚性变形的点云序列的挑战。该方法通过两阶段流程预测变形图序列，能有效处理嘈杂和部分输入，并利用时间信息实现准确和一致的序列注册。

### 背景

将物体形状注册到经历非刚性变形的点云序列是一个长期存在的挑战。主要困难来自两个方面：目标函数非凸性导致的局部极小值（特别是在嘈杂或部分输入情况下）阻碍了准确和鲁棒的变形估计；长序列中的误差累积导致跟踪失败。

### 目的

解决非刚性变形点云序列注册中的挑战，特别是局部极小值问题和误差累积问题，同时提高处理效率。

### 方法

采用可扩展的数据驱动方法，提出ERNet模型，这是一种在大变形数据集上训练的高效前馈模型。关键设计是通过两阶段流程预测变形图序列：首先估计帧级粗略图节点实现鲁棒初始化，然后在滑动窗口方式下随时间细化它们的轨迹。该方法能有效处理嘈杂和部分输入，同时利用时间信息进行准确和一致的序列注册。

### 主要发现

在Deforming Things4D和D-FAUST数据集上，所提出的方法优于之前的先进方法；与之前最好的方法相比，实现了4倍以上的速度提升，显著提高了处理效率。

### 结论

ERNet模型有效解决了非刚性变形点云序列注册中的挑战，在准确性和效率方面都表现出色。

### 翻译

将物体形状注册到经历非刚性变形的点云序列是一个长期存在的挑战。关键困难源于两个因素：(i)由于目标函数的非凸性，特别是在嘈杂或部分输入的情况下，存在局部极小值，这阻碍了准确和鲁棒的变形估计；(ii)长序列中的误差累积导致跟踪失败。为应对这些挑战，我们采用可扩展的数据驱动方法，并提出了ERNet，一种在大变形数据集上训练的高效前馈模型。它旨在处理嘈杂和部分输入，同时有效利用时间信息进行准确和一致的序列注册。我们设计的关键是通过两阶段流程预测变形图序列，首先估计帧级粗略图节点以实现鲁棒初始化，然后在滑动窗口方式下随时间细化它们的轨迹。大量实验表明，我们提出的方法(i)在Deforming Things4D和D-FAUST数据集上都优于之前的先进方法，(ii)与之前最好的方法相比实现了4倍以上的速度提升，提供了显著的效率改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决将物体形状注册到一系列经历非刚性变形的点云上的问题。这个问题在计算机视觉和机器人领域至关重要，因为它涉及动态重建、场景理解和机器人操作等广泛应用。传统方法容易陷入局部最优解，特别是在处理噪声或部分输入的点云时，导致变形估计不准确；同时，长序列中的误差累积会导致跟踪失败，限制了这些方法在实际应用中的有效性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统优化方法容易陷入局部最优，基于神经变形场的方法难以泛化到噪声或部分输入，而预测密集对应关系的方法计算复杂度高。因此，作者设计了一个高效的前馈模型，采用数据驱动方法在大型变形数据集上训练。方法借鉴了变形图表示、三平面编码器、滑动窗口策略和局部刚性假设等现有技术，但创新性地将它们组合成一个两阶段管道：首先估计帧级粗略图节点，然后在滑动窗口方式下细化节点轨迹，实现了鲁棒且时间一致的序列配准。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用变形图作为非刚性变形的紧凑表示，通过两阶段策略（粗略匹配和时空细化）实现高效配准，并利用局部刚性假设推断节点变换属性。整体流程包括：1) 使用三平面编码器将源点云和目标点云序列编码为特征；2) 从源点云采样节点，通过节点到帧匹配初始化节点位置；3) 使用时空变换器在滑动窗口中细化节点轨迹；4) 利用局部刚性假设和Procrustes分析估计节点变换；5) 使用径向基函数线性混合皮肤将变形图转换为密集变形场，完成配准。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 高效的前馈序列非刚性注册架构，采用两阶段策略提高鲁棒性和时间一致性；2) 变形图回归作为非刚性注册的高效表示，平衡了表达能力和计算效率；3) 利用局部刚性假设推断节点变换属性，避免直接预测高维非线性变换的困难。相比传统优化方法，ERNet不易陷入局部最优且能处理噪声输入；相比基于神经变形场的方法，它不需要每帧优化，效率更高；相比预测密集对应关系的方法，它计算效率更高，更适合处理长序列。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ERNet通过引入基于变形图的两阶段预测策略，实现了高效、准确且时间一致的非刚性点云序列配准，在保持高精度的同时实现了超过4倍的速度提升，显著优于现有方法。'}


### 论文摘要

Registering an object shape to a sequence of point clouds undergoing non-rigid deformation is a long-standing challenge. The key difficulties stem from two factors: (i) the presence of local minima due to the non-convexity of registration objectives, especially under noisy or partial inputs, which hinders accurate and robust deformation estimation, and (ii) error accumulation over long sequences, leading to tracking failures. To address these challenges, we introduce to adopt a scalable data-driven approach and propose ERNet, an efficient feed-forward model trained on large deformation datasets. It is designed to handle noisy and partial inputs while effectively leveraging temporal information for accurate and consistent sequential registration. The key to our design is predicting a sequence of deformation graphs through a two-stage pipeline, which first estimates frame-wise coarse graph nodes for robust initialization, before refining their trajectories over time in a sliding-window fashion. Extensive experiments show that our proposed approach (i) outperforms previous state-of-the-art on both the DeformingThings4D and D-FAUST datasets, and (ii) achieves more than 4x speedup compared to the previous best, offering significant efficiency improvement.

---

## 34. MRASfM: Multi-Camera Reconstruction and Aggregation through Structure-from-Motion in Driving Scenes

**论文链接:** [http://arxiv.org/abs/2510.15467v1](http://arxiv.org/abs/2510.15467v1)

**作者:** Lingfeng Xuan, Chang Nie, Yiqing Xu, Zhe Liu, Yanzi Miao, Hesheng Wang

**发布时间:** 2025-10-17

**备注:** 8 pages, 11 figures

### GPT解析

### 总结

本文提出了一种名为MRASfM的多相机重建和聚合运动结构框架，专门针对驾驶场景中的运动结构(SfM)问题，解决了姿态估计不可靠、道路表面重建异常值过多以及重建效率低等挑战。

### 背景

Structure from Motion (SfM)估计相机姿态并重建点云，是各种任务的基础。然而，将SfM应用于多相机系统捕捉的驾驶场景存在显著困难，包括不可靠的姿态估计、道路表面重建中过多的异常值以及低重建效率。

### 目的

为了解决这些限制，提出专门为驾驶场景设计的多相机重建和聚合运动结构(MRASfM)框架。

### 方法

MRASfM通过以下方法解决挑战：1)在注册过程中利用多相机系统内的固定空间关系提高姿态估计可靠性；2)采用平面模型有效去除道路表面重建中的错误点；3)在捆绑调整中将多相机集视为单一单元减少优化变量提高效率；4)通过场景关联和组装模块以从粗到细的方式实现多场景聚合。

### 主要发现

在实际车辆上部署多相机系统验证了MRASfM在不同场景中的泛化能力和在具有挑战性条件下的鲁棒性。在公共数据集上的大规模验证结果显示，MRASfM达到了最先进的性能，实现了较低的绝对姿态误差。

### 结论

MRASfM框架有效地解决了多相机系统在驾驶场景中应用SfM时面临的主要挑战，提高了姿态估计的可靠性、道路表面重建的质量和整体重建效率。

### 翻译

运动结构(SfM)估计相机姿态并重建点云，形成各种任务的基础。然而，将SfM应用于由多相机系统捕捉的驾驶场景存在显著困难，包括不可靠的姿态估计、道路表面重建中过多的异常值以及低重建效率。为了解决这些限制，我们提出了一种专门为驾驶场景设计的多相机重建和聚合运动结构(MRASfM)框架。MRASfM通过在注册过程中利用多相机系统内的固定空间关系来提高相机姿态估计的可靠性。为了提高道路表面重建的质量，我们的框架采用平面模型有效去除三角测量道路表面中的错误点。此外，在捆绑调整(BA)中将多相机集视为单一单元有助于减少优化变量以提高效率。此外，MRASfM通过场景关联和组装模块以从粗到细的方式实现多场景聚合。我们在实际车辆上部署了多相机系统，以验证MRASfM在各种场景中的泛化能力以及在具有挑战性条件下的鲁棒性。此外，在公共数据集上的大规模验证结果显示了MRASfM的最先进性能，实现了较低的绝对姿态误差。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决将传统SfM技术应用于多相机系统捕获的驾驶场景时的三大挑战：不可靠的姿态估计、道路表面重建中过多的离群点和低重建效率。这个问题在现实中很重要，因为准确的驾驶场景重建是高清地图构建和新视角合成等关键下游任务的基础，而传统vSLAM存在累积漂移问题，直接应用SfM又面临上述挑战，限制了自动驾驶系统对环境的精确感知和自身定位能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先深入分析了传统SfM在驾驶场景中的局限性，然后有选择地借鉴了现有工作：参考了多相机vSLAM方法如BAMF-SLAM和MAVIS，但意识到它们需要精确校准；借鉴了COLMAP和MMA等SfM方法，但发现它们在处理多相机系统时有局限；受MCSfM启发但希望进一步提高效率。作者针对性地设计了解决方案：利用多相机固定空间关系提高姿态估计可靠性，应用平面模型过滤道路离群点，将相机集视为统一单位提高效率，并设计场景聚合模块整合碎片化场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将多相机系统视为刚性单元，利用相机间的固定空间关系作为先验约束，结合语义信息提高重建质量，通过分层优化提高效率，并以粗到细方式整合碎片化场景。整体流程分为单场景重建和多场景聚合：单场景重建包括多相机对应点搜索、先验初始化和迭代重建（相机集注册、语义辅助三角测量、相机集BA）；多场景聚合包括场景关联（使用GNSS定位）和场景组装（粗组装和精组装，通过SfM优化变换矩阵）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 相机集注册模块，将多相机视为刚性单元提高姿态估计鲁棒性，不同于传统方法忽略或简单利用相机间关系；2) 相机集BA模块，优化车辆姿态和内部相对姿态而非单个相机姿态，显著减少优化变量；3) 语义辅助三角测量，使用平面模型过滤道路离群点，专门处理道路表面特殊挑战；4) 多场景聚合模块，以粗到细方式整合无共享图像的碎片化场景，突破了传统SfM聚合方法的限制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MRASfM通过将多相机系统视为刚性单元、结合语义信息过滤离群点、优化重建流程以及设计多场景聚合方法，实现了驾驶场景中更准确、高效和鲁棒的三维重建。'}


### 论文摘要

Structure from Motion (SfM) estimates camera poses and reconstructs point clouds, forming a foundation for various tasks. However, applying SfM to driving scenes captured by multi-camera systems presents significant difficulties, including unreliable pose estimation, excessive outliers in road surface reconstruction, and low reconstruction efficiency. To address these limitations, we propose a Multi-camera Reconstruction and Aggregation Structure-from-Motion (MRASfM) framework specifically designed for driving scenes. MRASfM enhances the reliability of camera pose estimation by leveraging the fixed spatial relationships within the multi-camera system during the registration process. To improve the quality of road surface reconstruction, our framework employs a plane model to effectively remove erroneous points from the triangulated road surface. Moreover, treating the multi-camera set as a single unit in Bundle Adjustment (BA) helps reduce optimization variables to boost efficiency. In addition, MRASfM achieves multi-scene aggregation through scene association and assembly modules in a coarse-to-fine fashion. We deployed multi-camera systems on actual vehicles to validate the generalizability of MRASfM across various scenes and its robustness in challenging conditions through real-world applications. Furthermore, large-scale validation results on public datasets show the state-of-the-art performance of MRASfM, achieving 0.124 absolute pose error on the nuScenes dataset.

---

## 35. Integrating Product Coefficients for Improved 3D LiDAR Data Classification (Part II)

**论文链接:** [http://arxiv.org/abs/2510.15219v1](http://arxiv.org/abs/2510.15219v1)

**作者:** Patricia Medina, Rasika Karkare

**发布时间:** 2025-10-17

**备注:** 16 pages, 6 figures, 5 tables

### GPT解析

### 总结

这项研究扩展了之前关于使用乘积系数增强3D LiDAR点云分类的工作，展示了将乘积系数与自编码器表示和KNN分类器结合可以带来性能提升。

### 背景

研究基于之前的工作，该工作引入了乘积系数（一种测度论描述符）来补充原始的LiDAR空间特征。

### 目的

探索将乘积系数与自编码器表示和KNN分类器结合的方法，以提升LiDAR分类性能。

### 方法

结合乘积系数与自编码器表示和KNN分类器，并与基于PCA的基线方法以及早期框架进行比较。还研究了逐级添加乘积系数的效果。

### 主要发现

将乘积系数与自编码器表示和KNN分类器结合，在PCA基线和早期框架上带来了一致的性能提升。逐级添加乘积系数显示出更丰富的系数集合系统性地改善了类别可分离性和整体准确性。

### 结论

结合分层乘积系数特征与自编码器对提升LiDAR分类性能具有重要价值。

### 翻译

这项工作扩展了我们之前关于使用乘积系数增强3D LiDAR点云分类的研究，乘积系数是一种补充原始空间LiDAR特征的测度论描述符。在这里，我们展示了将乘积系数与自编码器表示和KNN分类器相结合，在基于PCA的基线和我们早期的框架上都能带来一致的性能提升。我们还研究了逐级添加乘积系数的影响，揭示了一个明显的趋势：更丰富的系数集合系统性地改善了类别可分离性和整体准确性。结果强调了将分层乘积系数特征与自编码器相结合以进一步提升LiDAR分类性能的价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是改进3D LiDAR点云数据的分类性能。LiDAR技术广泛应用于数字高程模型更新、冰川和滑坡监测、海岸线分析和城市发展等领域，而将3D LiDAR点准确分类为语义类别（如植被、人造结构和水体）是这些应用中的关键步骤。提高分类准确率对于环境监测、城市规划、灾害评估等实际应用具有重要意义，特别是在气候变化研究中，如森林生长和碳封存能力的评估。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在之前工作[5]中引入了乘积系数作为度量论描述符，通过在原始空间Lidar特征基础上添加这些系数来增强分类性能。本研究进一步扩展了这个框架，借鉴了自编码器在表示学习方面的优势，用自编码器替代了之前工作使用的主成分分析(PCA)。作者认识到线性变换(如PCA)在捕获复杂特征依赖关系和减少冗余方面的局限性，因此引入了非线性表示学习方法。实验设计包括比较不同维度减少方法(PCA、自编码器、Nystroem)和不同分类器(KNN、随机森林)的性能，以验证新方法的有效性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '方法的核心思想是通过结合乘积系数和自编码器来增强3D LiDAR点云分类性能。乘积系数是基于度量论的特征，能够捕捉点云数据的局部结构信息，超越原始空间坐标。自编码器则学习非线性表示，能够更有效地捕获复杂特征依赖关系并减少冗余。整体实现流程包括：1) 特征生成：计算每个数据点周围局部邻域内的乘积系数，生成七个新特征；2) 特征标准化：将生成的特征标准化到单位立方体[0,1]^3；3) 维度减少：使用PCA或自编码器减少特征维度；4) 分类：使用KNN或随机森林分类器进行分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 引入自编码器替代PCA进行非线性表示学习，能够更有效地捕获复杂特征依赖关系；2) 系统地评估了不同级别乘积系数对分类性能的影响，发现更丰富的系数集能系统性地提高类别可分性和整体准确性；3) 实验证明结合乘积系数和自编码器的框架在分类准确率和F1分数上持续优于基于PCA的基线和之前的框架。相比之前的工作，主要不同在于使用了自编码器进行非线性表示学习，而不是使用PCA进行线性变换，以及更系统地评估了不同级别乘积系数的影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过结合基于度量论的乘积系数和自编码器非线性表示学习，显著提高了3D LiDAR点云分类的性能，为地理空间数据分析提供了一个更强大的框架。'}


### 论文摘要

This work extends our previous study on enhancing 3D LiDAR point-cloud classification with product coefficients \cite{medina2025integratingproductcoefficientsimproved}, measure-theoretic descriptors that complement the original spatial Lidar features. Here, we show that combining product coefficients with an autoencoder representation and a KNN classifier delivers consistent performance gains over both PCA-based baselines and our earlier framework. We also investigate the effect of adding product coefficients level by level, revealing a clear trend: richer sets of coefficients systematically improve class separability and overall accuracy. The results highlight the value of combining hierarchical product-coefficient features with autoencoders to push LiDAR classification performance further.

---

## 36. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model

**论文链接:** [http://arxiv.org/abs/2510.12276v2](http://arxiv.org/abs/2510.12276v2)

**作者:** Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了一种名为空间强制(SF)的对齐策略，用于增强视觉-语言-动作(VLA)模型的空间理解能力，无需依赖明确的3D输入或深度估计器。

### 背景

大多数VLA模型构建于仅基于2D数据预训练的视觉语言模型上，缺乏准确的空间感知能力，影响在3D物理世界中的操作。现有解决方案面临传感器噪声、硬件异构性和深度覆盖不完整等挑战。

### 目的

提出一种方法使VLA模型能够获得空间理解能力，而不依赖于明确的3D输入或深度估计器。

### 方法

提出空间强制(SF)对齐策略，通过将VLA的中间视觉嵌入与预训练的3D基础模型生成的几何表示对齐，隐式强制VLA模型发展空间理解能力。

### 主要发现

SF通过在中间层强制对齐，引导VLA编码更丰富的空间表示，提高动作精确度。实验表明SF取得了最先进的结果，超越了基于2D和3D的VLA，将训练速度提高了最多3.8倍，并提高了各种机器人任务的数据效率。

### 结论

SF是一种简单有效的对齐策略，能够使VLA模型获得空间理解能力，提高性能和训练效率。

### 翻译

视觉-语言-动作(VLA)模型最近在使机器人能够遵循语言指令和执行精确动作方面显示出强大的潜力。然而，大多数VLA构建于仅基于2D数据预训练的视觉语言模型上，这些模型缺乏准确的空间感知能力，阻碍了它们在3D物理世界中的操作能力。现有解决方案尝试整合明确的3D传感器输入，如深度图或点云，但由于传感器噪声、硬件异构性和现有数据集中的深度覆盖不完整，这些方法面临挑战。从2D图像估计3D线索的替代方法也受限于深度估计器的有限性能。我们提出了空间强制(SF)，这是一种简单而有效的对齐策略，隐式强制VLA模型发展空间理解能力，而不依赖于明确的3D输入或深度估计器。SF将VLA的中间视觉嵌入与预训练的3D基础模型生成的几何表示对齐。通过在中间层强制对齐，SF引导VLA编码更丰富的空间表示，提高动作精确度。在模拟和真实环境中的大量实验表明，SF取得了最先进的结果，超越了基于2D和3D的VLA。SF进一步将训练速度提高了最多3.8倍，并提高了各种机器人任务的数据效率。项目页面位于https://spatial-forcing.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉-语言-动作模型(VLA)缺乏准确空间感知能力的问题，因为这些模型大多仅基于2D数据预训练，无法有效适应3D物理世界。这个问题在现实中很重要，因为机器人操作需要在3D世界中整合语义推理和精确控制；在研究中重要是因为现有解决方案要么依赖昂贵的3D传感器(面临噪声和硬件兼容性问题)，要么从2D图像估计3D信息(受限于深度估计器性能)，而本文方法提供了一个无需这些依赖的替代方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过深度探测实验观察到当前VLA模型的视觉嵌入缺乏有意义的空间结构，然后提出一种隐式增强空间理解能力的方法。他们借鉴了表示监督领域的进展，特别是表示对齐策略，利用预训练的3D基础模型VGGT提供丰富的空间表示作为监督信号。作者还参考了Huang等人的工作，发现监督相对较深但不是最深的层(第24层)效果最佳，因为太浅的层可能无法获得足够的空间信息，而太深的层则会丢失视觉特定特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将VLA模型的中间视觉嵌入与预训练3D基础模型产生的几何表示进行对齐，隐式地强制模型发展空间理解能力，无需显式3D输入或深度估计器。实现流程是：1)输入多视角图像到VGGT模型生成空间表示；2)将VLA的中间视觉嵌入与这些空间表示进行对齐；3)使用余弦相似度作为对齐目标函数；4)选择第24层进行监督；5)将对齐损失与动作生成损失结合；6)推理阶段与标准VLA模型操作相同，无额外计算开销。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出Spatial Forcing方法，通过隐式对齐增强空间感知；2)不依赖显式3D传感器输入或深度估计器；3)通过中间层对齐引导模型编码丰富空间表示；4)实现训练加速(最高3.8倍)和数据效率提升。相比之前工作，不同之处在于：与显式3D输入方法相比，SF无需额外传感器；与从2D估计3D的方法相比，不受深度估计器限制；与现有表示监督方法不同，SF专注于空间表示对齐，特别针对VLA的空间感知提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Spatial Forcing方法，通过将视觉-语言-动作模型的中间视觉嵌入与预训练3D基础模型的空间表示进行隐式对齐，显著提升了模型的空间感知能力、训练效率和数据利用率，无需依赖显式3D传感器输入或深度估计器。'}


### 论文摘要

Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators. We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision. Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at https://spatial-forcing.github.io/

---

