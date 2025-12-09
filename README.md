# 今日论文推荐 - 2025-12-09

共 141 篇论文

---

## 1. Unison: A Fully Automatic, Task-Universal, and Low-Cost Framework for Unified Understanding and Generation

**论文链接:** [http://arxiv.org/abs/2512.07747v1](http://arxiv.org/abs/2512.07747v1)

**作者:** Shihao Zhao, Yitong Chen, Zeyinzi Jiang, Bojia Zi, Shaozhe Hao, Yu Liu, Chaojie Mao, Kwan-Yee K. Wong

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了一种名为Unison的多模态统一理解和生成模型，采用两阶段方案，以极低训练成本实现多种任务自动化处理。

### 背景

多模态学习中存在两种主流方法：一种是基于自回归范式训练transformer，需大量数据和计算资源；另一种是两阶段方案，连接预训练模型进行对齐微调，训练成本较低但任务覆盖有限或生成质量差。两种方法均缺乏解析输入元信息的能力，需手动参数配置。

### 目的

提出一种采用两阶段方案同时保留预训练模型能力的模型，以极低训练成本覆盖多种多模态理解任务和生成任务，实现自动解析用户意图、确定任务类型和提取参数，使多模态任务完全自动化。

### 方法

提出名为Unison的模型，采用两阶段方案保留预训练模型能力，实现自动解析用户意图、自动确定目标任务类型和自动提取相关参数，无需人工干预。

### 主要发现

在仅50万训练样本和50 GPU小时的低成本设置下，模型能够准确自动识别任务并提取相关参数，在多种理解和生成任务上取得了优越的性能。

### 结论

Unison模型成功实现了多模态任务的统一理解和生成，以极低的训练成本实现了高性能，并实现了任务的完全自动化，无需人工干预。

### 翻译

统一理解和生成是多模态学习中一个极具吸引力的研究方向。存在两种方法：一种是通过自回归范式训练transformer，另一种是采用连接预训练理解和生成模型进行对齐微调的两阶段方案。前者需要普通研究者难以负担的大量数据和计算资源。尽管后者的训练成本较低，但现有工作往往存在任务覆盖有限或生成质量差的问题。两种方法都缺乏解析输入元信息（如任务类型、图像分辨率、视频持续时间等）的能力，需要繁琐且不智能的手动参数配置。在本文中，我们提出了Unison，它采用两阶段方案同时保留预训练模型的能力。以极低的训练成本，我们涵盖了多种多模态理解任务，包括文本、图像和视频理解，以及多样的生成任务，如文本到视觉内容生成、编辑、可控生成和基于IP的参考生成。我们还赋予模型自动解析用户意图、确定目标任务类型并准确提取相应任务所需元信息的能力。这使得各种多模态任务能够完全自动化，无需人工干预。实验表明，在仅50万训练样本和50 GPU小时的低成本设置下，我们的模型能够准确自动地识别任务并提取相关参数，并在多种理解和生成任务上取得优越性能。


### 论文摘要

Unified understanding and generation is a highly appealing research direction in multimodal learning. There exist two approaches: one trains a transformer via an auto-regressive paradigm, and the other adopts a two-stage scheme connecting pre-trained understanding and generative models for alignment fine-tuning. The former demands massive data and computing resources unaffordable for ordinary researchers. Though the latter requires a lower training cost, existing works often suffer from limited task coverage or poor generation quality. Both approaches lack the ability to parse input meta-information (such as task type, image resolution, video duration, etc.) and require manual parameter configuration that is tedious and non-intelligent. In this paper, we propose Unison which adopts the two-stage scheme while preserving the capabilities of the pre-trained models well. With an extremely low training cost, we cover a variety of multimodal understanding tasks, including text, image, and video understanding, as well as diverse generation tasks, such as text-to-visual content generation, editing, controllable generation, and IP-based reference generation. We also equip our model with the ability to automatically parse user intentions, determine the target task type, and accurately extract the meta-information required for the corresponding task. This enables full automation of various multimodal tasks without human intervention. Experiments demonstrate that, under a low-cost setting of only 500k training samples and 50 GPU hours, our model can accurately and automatically identify tasks and extract relevant parameters, and achieve superior performance across a variety of understanding and generation tasks.

---

## 2. HLTCOE Evaluation Team at TREC 2025: VQA Track

**论文链接:** [http://arxiv.org/abs/2512.07738v1](http://arxiv.org/abs/2512.07738v1)

**作者:** Dengjia Zhang, Charles Weng, Katherine Guerrerio, Yi Lu, Kenton Murray, Alexander Martin, Reno Kriz, Benjamin Van Durme

**发布时间:** 2025-12-08

**备注:** 7 pages, 1 figure

### GPT解析

### 总结

HLTCOE团队开发了一种列表式学习框架，通过结合生成建模和判别式排序来改进VQA任务中的答案生成，提高了语义精度和排序一致性

### 背景

HLTCOE评估团队参与了TREC VQA的答案生成(AG)任务

### 目的

开发一个列表式学习框架，旨在提高答案生成中的语义精度和排序一致性

### 方法

给定视频-问题对，基础多模态模型首先生成多个候选答案，然后使用一种新颖的带排序权值的掩码指针交叉熵损失函数训练的模型对这些候选答案进行重新排序。该目标函数整合了基于指针的候选选择、依赖排序的加权和以及在词汇限制下的掩码交叉熵，实现了稳定且可解释的列表式优化。通过结合生成建模和判别式排序，该方法产生连贯的细粒度答案列表

### 主要发现

实验显示在准确性和排序稳定性方面有持续提升，对于需要时间推理和语义消歧的问题尤其有效

### 结论

这种列表式学习框架能够有效提高答案生成的质量和一致性

### 翻译

HLTCOE评估团队参与了TREC VQA的答案生成(AG)任务，为此我们开发了一个列表式学习框架，旨在提高答案生成中的语义精度和排序一致性。给定一个视频-问题对，基础多模态模型首先生成多个候选答案，然后使用一种新颖的带排序权值的掩码指针交叉熵损失函数训练的模型对这些候选答案进行重新排序。该目标函数整合了基于指针的候选选择、依赖排序的加权和以及在词汇限制下的掩码交叉熵，实现了稳定且可解释的列表式优化。通过结合生成建模与判别式排序，我们的方法产生了连贯的细粒度答案列表。实验显示在准确性和排序稳定性方面有持续提升，特别是对于需要时间推理和语义消歧的问题。


### 论文摘要

The HLTCOE Evaluation team participated in TREC VQA's Answer Generation (AG) task, for which we developed a listwise learning framework that aims to improve semantic precision and ranking consistency in answer generation. Given a video-question pair, a base multimodal model first generates multiple candidate answers, which are then reranked using a model trained with a novel Masked Pointer Cross-Entropy Loss with Rank Weights. This objective integrates pointer-based candidate selection, rank-dependent weighting, and masked cross-entropy under vocabulary restriction, enabling stable and interpretable listwise optimization. By bridging generative modeling with discriminative ranking, our method produces coherent, fine-grained answer lists. Experiments reveal consistent gains in accuracy and ranking stability, especially for questions requiring temporal reasoning and semantic disambiguation.

---

## 3. Unified Video Editing with Temporal Reasoner

**论文链接:** [http://arxiv.org/abs/2512.07469v1](http://arxiv.org/abs/2512.07469v1)

**作者:** Xiangpeng Yang, Ji Xie, Yiyuan Yang, Yan Huang, Min Xu, Qiang Wu

**发布时间:** 2025-12-08

**备注:** Project Page: https://videocof.github.io/

### GPT解析

### 总结

VideoCoF是一种创新的视频编辑方法，通过引入受思维链启发的帧链方法，解决了专家模型与统一模型之间的冲突。它通过强制执行'观察、推理、然后编辑'的过程，实现了无需用户掩码的精确指令到区域对齐和细粒度视频编辑，并利用RoPE对齐策略确保运动对齐和长度外推，以极小的数据成本达到了最先进的性能。

### 背景

现有的视频编辑方法面临一个关键的权衡：专家模型虽然精确，但依赖于特定任务的先验知识（如掩码），这阻碍了方法的统一性；另一方面，统一的时序上下文学习模型无需掩码，但缺乏明确的空间线索，导致指令到区域的映射能力弱且定位不精确。

### 目的

解决专家模型与统一模型之间的冲突，提出一种无需用户提供的掩码即可实现精确指令到区域对齐和细粒度视频编辑的方法。

### 方法

提出了VideoCoF，一种受思维链推理启发的创新帧链方法。VideoCoF强制执行'观察、推理、然后编辑'的过程，通过要求视频扩散模型先生成推理标记（编辑区域隐变量），再生成目标视频标记。此外，引入了一种RoPE对齐策略，利用这些推理标记确保运动对齐，并实现超出训练持续时间的长度外推。

### 主要发现

仅使用5万对视频数据的最小数据成本，VideoCoF在VideoCoF-Bench上达到了最先进的性能；显式推理步骤消除了用户提供掩码的需要；实现了精确的指令到区域对齐和细粒度视频编辑；RoPE对齐策略确保了运动对齐并支持长度外推。

### 结论

VideoCoF解决了现有视频编辑方法中的关键权衡问题，通过引入显式推理步骤，实现了无需掩码的精确视频编辑，并且方法高效有效。

### 翻译

现有的视频编辑方法面临一个关键的权衡：专家模型提供精确性但依赖于特定任务的先验知识如掩码，阻碍了统一性；相反，统一的时序上下文学习模型无需掩码但缺乏明确的空间线索，导致弱指令到区域映射和不精确的定位。为解决这一冲突，我们提出了VideoCoF，一种受思维链推理启发的创新帧链方法。VideoCoF通过强制视频扩散模型首先预测推理标记（编辑区域隐变量）然后生成目标视频标记，执行'观察、推理、然后编辑'的程序。这个显式推理步骤消除了用户提供掩码的需要，同时实现了精确的指令到区域对齐和细粒度视频编辑。此外，我们引入了一种RoPE对齐策略，利用这些推理标记确保运动对齐并实现超出训练持续时间的长度外推。我们证明，仅使用5万对视频数据的最小数据成本，VideoCoF在VideoCoF-Bench上达到了最先进的性能，验证了我们方法的高效性和有效性。我们的代码、权重和数据可在https://github.com/knightyxp/VideoCoF获取。


### 论文摘要

Existing video editing methods face a critical trade-off: expert models offer precision but rely on task-specific priors like masks, hindering unification; conversely, unified temporal in-context learning models are mask-free but lack explicit spatial cues, leading to weak instruction-to-region mapping and imprecise localization. To resolve this conflict, we propose VideoCoF, a novel Chain-of-Frames approach inspired by Chain-of-Thought reasoning. VideoCoF enforces a ``see, reason, then edit" procedure by compelling the video diffusion model to first predict reasoning tokens (edit-region latents) before generating the target video tokens. This explicit reasoning step removes the need for user-provided masks while achieving precise instruction-to-region alignment and fine-grained video editing. Furthermore, we introduce a RoPE alignment strategy that leverages these reasoning tokens to ensure motion alignment and enable length extrapolation beyond the training duration. We demonstrate that with a minimal data cost of only 50k video pairs, VideoCoF achieves state-of-the-art performance on VideoCoF-Bench, validating the efficiency and effectiveness of our approach. Our code, weight, data are available at https://github.com/knightyxp/VideoCoF.

---

## 4. Venus: An Efficient Edge Memory-and-Retrieval System for VLM-based Online Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.07344v1](http://arxiv.org/abs/2512.07344v1)

**作者:** Shengyuan Ye, Bei Ouyang, Tianyi Qian, Liekang Zeng, Mu Yuan, Xiaowen Chu, Weijie Hong, Xu Chen

**发布时间:** 2025-12-08

**备注:** Accepted by IEEE International Conference on Computer Communications 2026

### GPT解析

### 总结

该研究提出了Venus系统，一种用于高效在线视频理解的设备端内存和检索系统，通过边缘-云解耦架构显著降低了系统开销，同时保持了高推理准确性。

### 背景

视觉-语言模型(VLMs)在在线视频理解应用中展现出强大的多模态理解能力，但现有研究过于关注提升VLMs的推理能力，而忽略了部署限制，导致实际部署中系统开销过大。

### 目的

设计一种高效的视频理解系统，解决VLMs在实际部署中的系统开销问题，实现实时响应并保持高推理准确性。

### 方法

提出Venus系统，采用边缘-云解耦架构，将内存构建和关键帧检索从云端下沉到边缘，分两个阶段运行：1)摄入阶段：通过场景分割和聚类处理流式视频，构建分层内存；2)查询阶段：使用基于阈值的渐进式采样算法进行关键帧选择，平衡系统成本和推理准确性。

### 主要发现

Venus与最先进方法相比，在总响应延迟上实现了15x-131x的加速，能够在几秒内实现实时响应，同时保持相当甚至更优的推理准确性。

### 结论

Venus通过创新的边缘-云解耦架构和高效的内存检索机制，成功解决了VLMs在实时视频理解应用中的系统开销问题，为实际部署提供了可行方案。

### 翻译

视觉-语言模型(VLMs)已经展示了令人印象深刻的多模态理解能力，并被部署在越来越多的在线视频理解应用中。虽然最近的工作广泛探索了在这些情况下提升VLMs的推理能力，但部署约束被忽视，导致在实际部署中系统开销过大。为解决这个问题，我们提出了Venus，一个用于高效在线视频理解的设备端内存和检索系统。Venus提出了一种边缘-云解耦架构，将内存构建和关键帧检索从云端下沉到边缘，分两个阶段运行。在摄入阶段，Venus通过场景分割和聚类持续处理流式边缘视频，选中的关键帧使用多模态嵌入模型进行嵌入，构建分层内存以实现高效存储和检索。在查询阶段，Venus从内存索引传入的查询，并采用基于阈值的渐进式采样算法进行关键帧选择，提高多样性并自适应平衡系统成本和推理准确性。我们的广泛评估表明，与最先进方法相比，Venus在总响应延迟上实现了15x-131x的加速，能够在几秒内实现实时响应，同时保持相当甚至更优的推理准确性。


### 论文摘要

Vision-language models (VLMs) have demonstrated impressive multimodal comprehension capabilities and are being deployed in an increasing number of online video understanding applications. While recent efforts extensively explore advancing VLMs' reasoning power in these cases, deployment constraints are overlooked, leading to overwhelming system overhead in real-world deployments. To address that, we propose Venus, an on-device memory-and-retrieval system for efficient online video understanding. Venus proposes an edge-cloud disaggregated architecture that sinks memory construction and keyframe retrieval from cloud to edge, operating in two stages. In the ingestion stage, Venus continuously processes streaming edge videos via scene segmentation and clustering, where the selected keyframes are embedded with a multimodal embedding model to build a hierarchical memory for efficient storage and retrieval. In the querying stage, Venus indexes incoming queries from memory, and employs a threshold-based progressive sampling algorithm for keyframe selection that enhances diversity and adaptively balances system cost and reasoning accuracy. Our extensive evaluation shows that Venus achieves a 15x-131x speedup in total response latency compared to state-of-the-art methods, enabling real-time responses within seconds while maintaining comparable or even superior reasoning accuracy.

---

## 5. ContextAnyone: Context-Aware Diffusion for Character-Consistent Text-to-Video Generation

**论文链接:** [http://arxiv.org/abs/2512.07328v1](http://arxiv.org/abs/2512.07328v1)

**作者:** Ziyang Mai, Yu-Wing Tai

**发布时间:** 2025-12-08

### GPT解析

### 总结

ContextAnyone是一个上下文感知扩散框架，能够从文本和单个参考图像生成角色一致的视频，解决了现有方法无法保持发型、服装和体型等更广泛上下文线索的问题。

### 背景

文本到视频生成技术发展迅速，但在不同场景中保持角色身份一致性仍然是一个主要挑战。现有的个性化方法通常只关注面部身份，而无法保持更广泛的上下文线索，如发型、服装和体型等，这些对视觉连贯性至关重要。

### 目的

提出一个名为ContextAnyone的上下文感知扩散框架，实现从文本和单个参考图像生成角色一致的视频。

### 方法

联合重建参考图像和生成新的视频帧，使模型能够完全感知和利用参考信息；通过创新的Emphasize-Attention模块将参考信息有效整合到基于DiT的扩散主干中，选择性地强化参考感知特征并防止帧间的身份漂移；使用双重引导损失结合扩散和参考重建目标，以增强外观保真度；提出的Gap-RoPE位置嵌入将参考和视频令牌分离，以稳定时间建模。

### 主要发现

实验表明，ContextAnyone在身份一致性和视觉质量方面优于现有的参考到视频方法，能够生成连贯且保持上下文的角色视频，涵盖不同的动作和场景。

### 结论

ContextAnyone框架能够有效地解决文本到视频生成中角色身份一致性的问题，通过综合考虑面部特征和更广泛的上下文线索，生成高质量的角色一致视频。

### 翻译

文本到视频（T2V）生成技术发展迅速，但在不同场景中保持角色身份一致性仍然是一个主要挑战。现有的个性化方法通常只关注面部身份，但无法保持更广泛的上下文线索，如发型、服装和体型等，这些对视觉连贯性至关重要。我们提出了ContextAnyone，一个上下文感知的扩散框架，能够从文本和单个参考图像实现角色一致的视频生成。我们的方法联合重建参考图像和生成新的视频帧，使模型能够完全感知和利用参考信息。参考信息通过创新的Emphasize-Attention模块有效整合到基于DiT的扩散主干中，该模块选择性地强化参考感知特征并防止帧间的身份漂移。双重引导损失结合了扩散和参考重建目标，以增强外观保真度，而提出的Gap-RoPE位置嵌入将参考和视频令牌分离，以稳定时间建模。实验表明，ContextAnyone在身份一致性和视觉质量方面优于现有的参考到视频方法，能够生成连贯且保持上下文的角色视频，涵盖不同的动作和场景。项目页面：https://github.com/ziyang1106/ContextAnyone


### 论文摘要

Text-to-video (T2V) generation has advanced rapidly, yet maintaining consistent character identities across scenes remains a major challenge. Existing personalization methods often focus on facial identity but fail to preserve broader contextual cues such as hairstyle, outfit, and body shape, which are critical for visual coherence. We propose \textbf{ContextAnyone}, a context-aware diffusion framework that achieves character-consistent video generation from text and a single reference image. Our method jointly reconstructs the reference image and generates new video frames, enabling the model to fully perceive and utilize reference information. Reference information is effectively integrated into a DiT-based diffusion backbone through a novel Emphasize-Attention module that selectively reinforces reference-aware features and prevents identity drift across frames. A dual-guidance loss combines diffusion and reference reconstruction objectives to enhance appearance fidelity, while the proposed Gap-RoPE positional embedding separates reference and video tokens to stabilize temporal modeling. Experiments demonstrate that ContextAnyone outperforms existing reference-to-video methods in identity consistency and visual quality, generating coherent and context-preserving character videos across diverse motions and scenes. Project page: \href{https://github.com/ziyang1106/ContextAnyone}{https://github.com/ziyang1106/ContextAnyone}.

---

## 6. NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.07218v1](http://arxiv.org/abs/2512.07218v1)

**作者:** Feng Liang, Weixin Zeng, Runhao Zhao, Xiang Zhao

**发布时间:** 2025-12-08

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

该论文提出了Neuro-Symbolic Temporal Reasoning (NeSTR)框架，结合结构化符号表示与混合反思推理，以增强大型语言模型在时间推理方面的能力。

### 背景

大型语言模型在自然语言处理任务中表现出色，但在复杂时间约束下的时间推理仍然是一个主要挑战。

### 目的

解决现有符号方法和反思机制在时间推理中的局限性，提高大型语言模型对时间信息的理解和应用能力。

### 方法

NeSTR框架通过符号编码保留明确的时间关系，通过验证强制逻辑一致性，并通过反身反思纠正有缺陷的推理。

### 主要发现

NeSTR在各种时间问答基准上实现了优越的零样本性能，并在无需微调的情况下持续改进时间推理能力。

### 结论

神经符号集成在增强大型语言模型时间理解方面具有显著优势。

### 翻译

大型语言模型已在各种自然语言处理任务中展现出卓越的性能。然而，时间推理，特别是在复杂时间约束下的推理，仍然是一个重大挑战。为此，现有方法探索了符号方法，它明确编码时间结构，以及反思机制，它通过多步推理修正推理错误。尽管如此，符号方法通常未能充分利用LLM的推理能力，而反思方法通常缺乏结构化的时间表示，这可能导致不一致或幻觉推理。因此，即使有正确的时间上下文，LLM仍可能误解或误用时间相关信息，导致不完整或不准确的答案。为解决这些局限性，本文提出了神经符号时间推理（NeSTR），一种将结构化符号表示与混合反思推理相结合的新框架，以增强LLM推理的时间敏感性。NeSTR通过符号编码保留明确的时间关系，通过验证强制逻辑一致性，并通过反身反思纠正有缺陷的推理。在多种时间问答基准上的广泛实验表明，NeSTR实现了优越的零样本性能，并在无需微调的情况下持续改进时间推理，展示了神经符号集成在增强大型语言模型时间理解方面的优势。


### 论文摘要

Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, temporal reasoning, particularly under complex temporal constraints, remains a major challenge. To this end, existing approaches have explored symbolic methods, which encode temporal structure explicitly, and reflective mechanisms, which revise reasoning errors through multi-step inference. Nonetheless, symbolic approaches often underutilize the reasoning capabilities of LLMs, while reflective methods typically lack structured temporal representations, which can result in inconsistent or hallucinated reasoning. As a result, even when the correct temporal context is available, LLMs may still misinterpret or misapply time-related information, leading to incomplete or inaccurate answers. To address these limitations, in this work, we propose Neuro-Symbolic Temporal Reasoning (NeSTR), a novel framework that integrates structured symbolic representations with hybrid reflective reasoning to enhance the temporal sensitivity of LLM inference. NeSTR preserves explicit temporal relations through symbolic encoding, enforces logical consistency via verification, and corrects flawed inferences using abductive reflection. Extensive experiments on diverse temporal question answering benchmarks demonstrate that NeSTR achieves superior zero-shot performance and consistently improves temporal reasoning without any fine-tuning, showcasing the advantage of neuro-symbolic integration in enhancing temporal understanding in large language models.

---

## 7. NeuroABench: A Multimodal Evaluation Benchmark for Neurosurgical Anatomy Identification

**论文链接:** [http://arxiv.org/abs/2512.06921v1](http://arxiv.org/abs/2512.06921v1)

**作者:** Ziyang Song, Zelin Zang, Xiaofan Ye, Boqiang Xu, Long Bai, Jinlin Wu, Hongliang Ren, Hongbin Liu, Jiebo Luo, Zhen Lei

**发布时间:** 2025-12-07

**备注:** Accepted by IEEE ICIA 2025

### GPT解析

### 总结

该研究介绍了Neurosurgical Anatomy Benchmark (NeuroABench)，这是首个专门用于评估神经外科领域解剖理解的多模态基准测试。研究评估了当前最先进的多模态大语言模型在解剖识别任务上的表现，并与神经外科住院医师的表现进行了比较。

### 背景

多模态大语言模型在手术视频理解方面显示出巨大潜力，能够提高零样本性能和促进人机交互，为推进外科教育和辅助提供坚实基础。然而，现有研究和数据集主要关注手术程序和工作流程的理解，而对解剖理解的关键作用关注有限。在临床实践中，外科医生严重依赖精确的解剖理解来解释、回顾和学习手术视频。

### 目的

为了填补这一空白，作者引入了Neurosurgical Anatomy Benchmark (NeuroABench)，这是第一个专门创建用于评估神经外科领域解剖理解的多模态基准测试。

### 方法

NeuroABench包含9小时注释的神经外科视频，涵盖89种不同的程序，并使用新颖的多模态注释流程开发，具有多个审查周期。该基准测试评估68个临床解剖结构的识别，为评估模型性能提供了严格和标准化的框架。研究在超过10个最先进的多模态大语言模型上进行了实验，并提取了一个数据集子集，与四位神经外科住院医师进行了信息测试。

### 主要发现

在多模态大语言模型上的实验显示出显著的局限性，表现最好的模型在解剖识别任务中只达到40.87%的准确率。神经外科住院医师测试中，表现最好的学生达到56%的准确率，最低分为28%，平均分为46.5%。虽然表现最好的多模态大语言模型与得分最低的学生表现相当，但仍显著落后于该组的平均表现。

### 结论

这一比较既强调了多模态大语言模型在解剖理解方面的进展，也强调了在实现人类水平性能方面仍存在巨大差距。

### 翻译

多模态大语言模型在手术视频理解方面已显示出巨大潜力。随着零样本性能的提升和更有效的人机交互，它们为推进外科教育和辅助提供了坚实的基础。然而，现有研究和数据集主要关注对手术程序和工作流程的理解，而对解剖理解的关键作用关注有限。在临床实践中，外科医生严重依赖精确的解剖理解来解释、回顾和学习手术视频。为了填补这一空白，我们引入了神经外科解剖基准测试(NeuroABench)，这是首个专门创建用于评估神经外科领域解剖理解的多模态基准测试。NeuroABench包含9小时注释的神经外科视频，涵盖89种不同的程序，并使用新颖的多模态注释流程开发，具有多个审查周期。该基准测试评估68个临床解剖结构的识别，为评估模型性能提供了严格和标准化的框架。在超过10个最先进的多模态大语言模型上的实验揭示了显著的局限性，表现最好的模型在解剖识别任务中仅达到40.87%的准确率。为了进一步评估该基准测试，我们从数据集中提取了一个子集，并与四位神经外科住院医师进行了信息测试。结果显示，表现最好的学生达到56%的准确率，最低分为28%，平均分为46.5%。虽然表现最好的多模态大语言模型与得分最低的学生表现相当，但仍显著落后于该组的平均表现。这一比较既强调了多模态大语言模型在解剖理解方面的进展，也强调了在实现人类水平性能方面仍存在的巨大差距。


### 论文摘要

Multimodal Large Language Models (MLLMs) have shown significant potential in surgical video understanding. With improved zero-shot performance and more effective human-machine interaction, they provide a strong foundation for advancing surgical education and assistance. However, existing research and datasets primarily focus on understanding surgical procedures and workflows, while paying limited attention to the critical role of anatomical comprehension. In clinical practice, surgeons rely heavily on precise anatomical understanding to interpret, review, and learn from surgical videos. To fill this gap, we introduce the Neurosurgical Anatomy Benchmark (NeuroABench), the first multimodal benchmark explicitly created to evaluate anatomical understanding in the neurosurgical domain. NeuroABench consists of 9 hours of annotated neurosurgical videos covering 89 distinct procedures and is developed using a novel multimodal annotation pipeline with multiple review cycles. The benchmark evaluates the identification of 68 clinical anatomical structures, providing a rigorous and standardized framework for assessing model performance. Experiments on over 10 state-of-the-art MLLMs reveal significant limitations, with the best-performing model achieving only 40.87% accuracy in anatomical identification tasks. To further evaluate the benchmark, we extract a subset of the dataset and conduct an informative test with four neurosurgical trainees. The results show that the best-performing student achieves 56% accuracy, with the lowest scores of 28% and an average score of 46.5%. While the best MLLM performs comparably to the lowest-scoring student, it still lags significantly behind the group's average performance. This comparison underscores both the progress of MLLMs in anatomical understanding and the substantial gap that remains in achieving human-level performance.

---

## 8. Less Is More, but Where? Dynamic Token Compression via LLM-Guided Keyframe Prior

**论文链接:** [http://arxiv.org/abs/2512.06866v1](http://arxiv.org/abs/2512.06866v1)

**作者:** Yulin Li, Haokun Gui, Ziyang Fan, Junjie Wang, Bin Kang, Bin Chen, Zhuotao Tian

**发布时间:** 2025-12-07

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

DyToK是一种无需训练的视频令牌压缩方法，通过利用VLLMs的固有注意力机制实现动态令牌压缩，优先选择语义丰富的帧同时抑制冗余，显著提高了视频大型语言模型的效率而不牺牲准确性。

### 背景

视频大型语言模型(VLLMs)在视频理解方面取得了显著进展，但面临由于长视频视觉标记序列导致的二次计算增长带来的效率瓶颈。现有关键帧采样方法虽能提高时间建模效率，但在特征编码前引入额外计算成本，且二进制帧选择范式并非最优。

### 目的

提出一种动态令牌压缩方法，解决VLLMs处理长视频时的效率瓶颈问题，实现效率与准确性的更好平衡。

### 方法

提出DyToK（Dynamic Token compression via LLM-guided Keyframe prior），一种无需训练的范式，通过利用VLLMs的固有注意力机制实现动态令牌压缩。利用VLLM注意力层自然编码的查询条件关键帧先验，动态调整每帧令牌保留比率，优先选择语义丰富的帧同时抑制冗余。

### 主要发现

DyToK实现了最先进的效率-准确性权衡，与现有压缩方法（如VisionZip和FastV）即插即用兼容，在保持多个VLLMs（如LLaVA-OneVision和Qwen2.5-VL）准确性的同时，推理速度提高了4.3倍。

### 结论

DyToK是一种有效的视频令牌压缩方法，能够显著提高视频大型语言模型的处理效率而不牺牲准确性，且具有良好的兼容性。

### 翻译

视频大型语言模型(VLLMs)的最新进展已实现了显著的视频理解能力，但由于长视频的视觉标记序列导致计算量呈二次增长，仍面临关键的效率瓶颈。虽然现有关键帧采样方法可以提高时间建模效率，但在特征编码前引入了额外的计算成本，并且二进制帧选择范式被发现并非最优。因此，在这项工作中，我们提出了DyToK（Dynamic Token compression via LLM-guided Keyframe prior），一种无需训练的范式，通过利用VLLMs的固有注意力机制实现动态令牌压缩。我们的分析表明，VLLM注意力层自然编码了查询条件关键帧先验，通过DyToK动态调整每帧令牌保留比率，优先选择语义丰富的帧同时抑制冗余。大量实验证明，DyToK实现了最先进的效率-准确性权衡。DyToK与现有压缩方法（如VisionZip和FastV）即插即用兼容，在保持多个VLLMs（如LLaVA-OneVision和Qwen2.5-VL）准确性的同时，推理速度提高了4.3倍。代码可在https://github.com/yu-lin-li/DyToK获取。


### 论文摘要

Recent advances in Video Large Language Models (VLLMs) have achieved remarkable video understanding capabilities, yet face critical efficiency bottlenecks due to quadratic computational growth with lengthy visual token sequences of long videos. While existing keyframe sampling methods can improve temporal modeling efficiency, additional computational cost is introduced before feature encoding, and the binary frame selection paradigm is found suboptimal. Therefore, in this work, we propose Dynamic Token compression via LLM-guided Keyframe prior (DyToK), a training-free paradigm that enables dynamic token compression by harnessing VLLMs' inherent attention mechanisms. Our analysis reveals that VLLM attention layers naturally encoding query-conditioned keyframe priors, by which DyToK dynamically adjusts per-frame token retention ratios, prioritizing semantically rich frames while suppressing redundancies. Extensive experiments demonstrate that DyToK achieves state-of-the-art efficiency-accuracy tradeoffs. DyToK shows plug-and-play compatibility with existing compression methods, such as VisionZip and FastV, attaining 4.3x faster inference while preserving accuracy across multiple VLLMs, such as LLaVA-OneVision and Qwen2.5-VL. Code is available at https://github.com/yu-lin-li/DyToK .

---

## 9. MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.06810v1](http://arxiv.org/abs/2512.06810v1)

**作者:** Yueqian Wang, Songxiang Liu, Disong Wang, Nuo Xu, Guanglu Wan, Huishuai Zhang, Dongyan Zhao

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文提出了一种新颖的文本到文本的主动交互方法，通过多轮强化学习训练的模型MMDuet2，能够在视频播放过程中自主决定何时响应，无需精确的响应时间标注，在ProactiveVideoQA基准测试上取得了最先进的性能。

### 背景

视频多模态大语言模型（Video MLLMs）的最新进展显著提升了视频理解和多模态交互能力，但大多数现有系统采用轮流对话方式，模型只能在用户发言后回复。

### 目的

开发一种能够在视频播放过程中主动决定何时回复的模型，以提升实时应用中的交互体验。

### 方法

提出一种基于多轮强化学习的训练方法，使模型能够根据对话历史和当前帧的视觉上下文自主决定是否回应；在包含5.2万个视频和两种类型对话的数据集上通过SFT和RL训练模型MMDuet2。

### 主要发现

MMDuet2在响应时机和质量方面优于现有的主动Video MLLM基线，在ProactiveVideoQA基准测试上取得了最先进的性能。

### 结论

所提出的方法解决了先前方法中的困难，如手动调整响应决策阈值和标注精确回复时间的问题，实现了更好的主动交互性能。

### 翻译

视频多模态大语言模型（Video MLLMs）的最新进展显著提升了视频理解和多模态交互能力。虽然大多数现有系统采用轮流对话方式，模型只能在用户发言后回复，但在视频播放过程中主动决定何时回复，对于实时应用来说是一个有前景但具有挑战性的方向。在这项工作中，我们提出了一种新颖的文本到文本的主动交互方法，模型能够根据对话历史和当前帧的视觉上下文，自主决定在每个回合是否回应。为了克服先前方法中的困难，如手动调整响应决策阈值和标注精确回复时间，我们引入了一种基于多轮强化学习的训练方法，鼓励及时准确的响应，而无需精确的响应时间标注。我们在包含5.2万个视频和两种类型对话的数据集上通过SFT和RL训练了我们的模型MMDuet2。实验结果表明，MMDuet2在响应时机和质量方面优于现有的主动Video MLLM基线，在ProactiveVideoQA基准测试上取得了最先进的性能。


### 论文摘要

Recent advances in video multimodal large language models (Video MLLMs) have significantly enhanced video understanding and multi-modal interaction capabilities. While most existing systems operate in a turn-based manner where the model can only reply after user turns, proactively deciding when to reply during video playback presents a promising yet challenging direction for real-time applications. In this work, we propose a novel text-to-text approach to proactive interaction, where the model autonomously determines whether to respond or remain silent at each turn based on dialogue history and visual context up to current frame of an streaming video. To overcome difficulties in previous methods such as manually tuning response decision thresholds and annotating precise reply times, we introduce a multi-turn RL based training method that encourages timely and accurate responses without requiring precise response time annotations. We train our model MMDuet2 on a dataset of 52k videos with two types of dialogues via SFT and RL. Experimental results demonstrate that MMDuet2 outperforms existing proactive Video MLLM baselines in response timing and quality, achieving state-of-the-art performance on the ProactiveVideoQA benchmark.

---

## 10. 1 + 1 > 2: Detector-Empowered Video Large Language Model for Spatio-Temporal Grounding and Reasoning

**论文链接:** [http://arxiv.org/abs/2512.06673v1](http://arxiv.org/abs/2512.06673v1)

**作者:** Shida Gao, Feng Xue, Xiangfeng Wang, Anlong Ming, Teng Long, Yihua Shao, Haozhe Wang, Zhaowen Lin, Wei Wang, Nicu Sebe

**发布时间:** 2025-12-07

### GPT解析

### 总结

DEViL是一种结合视频LLM和开放词汇检测器的方法，通过参考语义令牌连接两者，并使用管状挖掘时间正则化来解决自回归空间解码中的错误累积和漂移问题，在时空定位和推理任务中表现出色。

### 背景

时空定位和推理旨在根据用户查询定位视频中事件的时间段和空间区域，并推理因果关系、时间顺序和动作关系等语义。当前多模态大语言模型主要将边界框视为文本令牌并自回归生成，但这种方法会导致输出序列过长，空间错误随时间累积，定位结果逐渐漂移。

### 目的

解决当前多模态大语言模型在时空定位和推理任务中自回归空间解码导致的空间错误累积和定位结果漂移问题。

### 方法

提出DEViL(Detector-Empowered Video LLM)，将视频LLM与开放词汇检测器(OVD)耦合，通过参考语义令牌(RST)连接两者，RST既作为控制信号又作为OVD文本嵌入的替代品。此外，在OVD内提出管状挖掘时间正则化(TTReg)，促使OVD为目标对象生成时间上一致的查询。

### 主要发现

DEViL在各种细粒度视频理解任务中表现出色，特别是在STVG和GroundedVQA任务上。

### 结论

通过结合视频LLM和开放词汇检测器，并使用参考语义令牌和管状挖掘时间正则化，DEViL有效解决了自回归空间解码中的错误累积和漂移问题，提升了时空定位和推理的性能。

### 翻译

时空定位和推理旨在根据用户查询定位视频中事件的时间段和空间区域，同时推理因果关系、时间顺序和动作关系等语义。为实现这一目标，当前多模态大语言模型主要将边界框视为文本令牌并自回归生成它们。然而，这种自回归空间解码会导致输出序列非常长，使空间错误随时间累积，导致定位结果在视频中逐渐漂移。为解决这一问题，我们提出了DEViL(Detector-Empowered Video LLM)，它将视频LLM与开放词汇检测器(OVD)耦合。具体而言，MLLM和检测器通过参考语义令牌(RST)连接，该令牌将用户查询提炼为丰富的语义表示。与仅作为空间提示或分割器开关的令牌不同，RST既作为控制信号又作为OVD文本嵌入的替代品，使指代理解和空间定位能够端到端学习。此外，我们在OVD内提出了管状挖掘时间正则化(TTReg)，它促使OVD为目标对象生成时间上一致的查询，从而确保有效的时间关联。实验表明，DEViL在各种细粒度视频理解任务中表现出色，特别是在STVG和GroundedVQA任务上。代码将在https://github.com/gaostar123/DeViL上发布。


### 论文摘要

Spatio-temporal grounding and reasoning aims to locate the temporal segment and spatial region of an event in a video given a user query, while also reasoning about semantics such as causality, temporal order, and action relationships. To achieve this, current MLLMs primarily treats bounding boxes as text tokens and generates them autoregressively. However, such autoregressive spatial decoding leads to very-long output sequences, causing spatial errors to accumulated over time and the localization results to progressively drift across a video. To address this, we present a Detector-Empowered Video LLM, short for DEViL, which couples a Video LLM with an open-vocabulary detector (OVD). Specifically, the MLLM and detector are connected via a reference-semantic token (RST) that distills the user query into a rich semantic representation. Unlike tokens that merely serve as spatial prompts or segmentor switches, the RST functions as both a control signal and a replacement for the OVD's text embedding, enabling end-to-end learning of both referential understanding and spatial localization. Furthermore, we propose a tube-mined temporal regularization (TTReg) within OVD, which drives the OVD to generate temporally-consistent queries for target objects, thereby ensuring effective temporal association. Experiments demonstrate that DEViL achieves strong performance across various fine-grained video understanding tasks, particularly STVG and GroundedVQA. Code will be released on https://github.com/gaostar123/DeViL.

---

## 11. MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.06581v1](http://arxiv.org/abs/2512.06581v1)

**作者:** Yuhao Su, Anwesa Choudhuri, Zhongpai Gao, Benjamin Planche, Van Nguyen Nguyen, Meng Zheng, Yuhan Shen, Arun Innanje, Terrence Chen, Ehsan Elhamifar, Ziyan Wu

**发布时间:** 2025-12-06

### GPT解析

### 总结

这篇论文提出了MedVidBench基准数据集和MedGRPO强化学习框架，用于改进医学视频理解中的视觉-语言模型性能。MedVidBench包含531,850个视频-指令对，涵盖8个医学来源，而MedGRPO通过跨数据集奖励归一化和医学LLM评估器解决了多数据集训练中的奖励不平衡问题。

### 背景

大型视觉-语言模型在医学视频理解方面存在困难，因为这类任务需要空间精确性、时间推理能力和临床语义理解。

### 目的

创建一个大规模的医学视频理解基准数据集；解决多数据集训练中奖励不平衡导致训练崩溃的问题；提出一个稳健的训练方法，促进视觉-语言模型在医学领域的发展。

### 方法

构建MedVidBench基准数据集：包含531,850个视频-指令对，覆盖8个医学来源，包括视频、片段和帧级别的任务，通过严格的质量保证流程构建；提出MedGRPO框架：包含两个关键创新——跨数据集奖励归一化，将每个数据集的中等性能映射到共同奖励值；医学LLM评估器，通过比较相似性评分评估标题质量，从五个临床维度进行评估。

### 主要发现

在MedVidBench上进行监督微调的Qwen2.5-VL-7B模型在所有任务上都优于GPT-4.1和Gemini-2.5-Flash；MedGRPO框架在基础线和定位、标题生成任务上进一步改进了性能。

### 结论

该研究为医学领域的视觉-语言模型建立了基础基准和稳健的训练方法论，促进了该领域的发展。

### 翻译

大型视觉-语言模型在医学视频理解方面存在困难，这类任务需要空间精确性、时间推理能力和临床语义理解。为此，我们首先引入了MedVidBench，这是一个大规模基准，包含来自8个医学来源的531,850个视频-指令对，涵盖视频、片段和帧级别任务，通过专家引导提示和双模型验证的严格质量保证流程进行策划。虽然在MedVidBench上进行监督微调带来了显著提升，但由于数据集间奖励规模不平衡，标准的强化学习(RL)方法失败，这导致优化不稳定并引发训练崩溃。为克服这一问题，我们引入了MedGRPO，这是一个用于平衡多数据集训练的新型RL框架，具有两个关键创新：(1)跨数据集奖励归一化，将每个数据集的中等性能映射到共同奖励值，确保无论难度如何都能实现公平优化；(2)医学LLM评估器，通过比较相似性评分从五个临床维度评估标题质量。在MedVidBench上对Qwen2.5-VL-7B进行监督微调在所有任务上都显著优于GPT-4.1和Gemini-2.5-Flash，证明了MedVidBench的有效性，而我们的MedGRPO框架在基础线上进一步改进了定位和标题生成任务。我们的工作为医学领域视觉-语言模型的进展建立了基础基准和稳健的训练方法论。我们的项目网站可在https://yuhaosu.github.io/MedGRPO/获取。


### 论文摘要

Large vision-language models struggle with medical video understanding, where spatial precision, temporal reasoning, and clinical semantics are critical. To address this, we first introduce \textbf{MedVidBench}, a large-scale benchmark of 531,850 video-instruction pairs across 8 medical sources spanning video, segment, and frame-level tasks, curated through a rigorous quality assurance pipeline with expert-guided prompting and dual-model validation. While supervised fine-tuning on MedVidBench yields noticeable gains, standard Reinforcement Learning (RL) fails due to imbalanced reward scales across datasets, which destabilizes optimization and leads to training collapse. To overcome this, we introduce \textbf{MedGRPO}, a novel RL framework for balanced multi-dataset training with two key innovations: (1) \emph{cross-dataset reward normalization} that maps each dataset's median performance to a common reward value, ensuring fair optimization regardless of difficulty, and (2) a \emph{medical LLM judge} that evaluates caption quality on five clinical dimensions through comparative similarity scoring. Supervised fine-tuning Qwen2.5-VL-7B on MedVidBench substantially outperforms GPT-4.1 and Gemini-2.5-Flash across all tasks, demonstrating MedVidBench's efficacy, while our MedGRPO framework further improves upon the SFT baseline across grounding and captioning tasks. Our work establishes a foundational benchmark and robust training methodology for advancing vision-language models in medical domains. Our project website is available at https://yuhaosu.github.io/MedGRPO/.

---

## 12. Exploiting Spatiotemporal Properties for Efficient Event-Driven Human Pose Estimation

**论文链接:** [http://arxiv.org/abs/2512.06306v1](http://arxiv.org/abs/2512.06306v1)

**作者:** Haoxian Zhou, Chuanzhi Xu, Langyi Chen, Haodong Chen, Yuk Ying Chung, Qiang Qu, Xaoming Chen, Weidong Cai

**发布时间:** 2025-12-06

### GPT解析

### 总结

该研究提出了一种基于点云框架的事件流人体姿态估计方法，通过设计事件时间切片卷积模块和事件切片序列化模块，结合边缘增强技术，在保持事件信号高时间分辨率的同时提高了人体姿态估计性能。

### 背景

人体姿态估计关注预测人体关键点以分析人体运动。事件相机提供高时间分辨率和低延迟，能够在具有挑战性的条件下实现稳健估计。然而，大多数现有方法将事件流转换为密集事件帧，增加了计算量并牺牲了事件信号的高时间分辨率。

### 目的

基于点云框架利用事件流的时空特性，以增强人体姿态估计性能。

### 方法

设计事件时间切片卷积模块捕获事件切片之间的短期依赖关系，结合事件切片序列化模块进行结构化时间建模，并在基于点云的事件表示中应用边缘增强，以在稀疏事件条件下增强空间边缘信息。

### 主要发现

在DHP19数据集上的实验表明，提出的方法在PointNet、DGCNN和Point Transformer三个代表性的点云骨干网络上均提高了性能。

### 结论

通过直接利用事件流的时空特性而不转换为密集帧，该方法能够在保持事件相机高时间分辨率优势的同时，提高人体姿态估计的准确性。

### 翻译

人体姿态估计关注预测人体关键点以分析人体运动。事件相机提供高时间分辨率和低延迟，能够在具有挑战性的条件下实现稳健估计。然而，大多数现有方法将事件流转换为密集事件帧，这增加了额外计算量并牺牲了事件信号的高时间分辨率。在这项工作中，我们旨在基于点云框架利用事件流的时空特性，以增强人体姿态估计性能。我们设计了事件时间切片卷积模块来捕获事件切片之间的短期依赖关系，并将其与事件切片序列化模块结合进行结构化时间建模。我们还在基于点云的事件表示中应用边缘增强，以在稀疏事件条件下增强空间边缘信息，从而进一步提高性能。在DHP19数据集上的实验表明，我们提出的方法在PointNet、DGCNN和Point Transformer三个代表性的点云骨干网络上均提高了性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何高效利用事件相机进行人体姿态估计的问题。现有方法通常将事件流转换为密集事件帧，这增加了计算量并牺牲了事件信号的高时间分辨率特性。这个问题很重要，因为事件相机能在低光、高动态范围等极端条件下工作，克服了传统相机的局限，而高效利用这些数据可以带来更准确、更实时的人体姿态估计，在动作识别、人机交互等领域有广泛应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：大多数方法将事件流转换为密集事件帧，破坏了事件数据的稀疏性并增加计算量；同时，静态身体部位不产生事件导致难以识别。作者借鉴了Chen等人的事件点云表示方法和Xu等人的Sobel边缘增强技术，在此基础上设计了ETSC模块捕捉短期时间依赖，ES-Seq模块组织时间序列，以及Sobel边缘增强模块提升空间信息。作者通过观察相邻时间切片包含重要运动线索，提出显式建模时间依赖关系可以解决静态部位识别问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用事件数据的时空特性，通过点云框架而非密集帧来增强人体姿态估计，解决静态部位难以识别的问题。整体流程包括：1)将事件流转换为5维光栅化事件点云(x,y,tavg,pacc,ecnt)；2)应用Sobel边缘增强增强空间边缘信息；3)通过ES-Seq模块将点云组织成时间切片序列；4)使用ETSC模块捕捉跨切片的时间依赖关系；5)融合时空特征并通过骨干网络(如PointNet)进行姿态估计；6)通过三角测量获得最终3D姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)ETSC模块：捕捉事件切片间的短期依赖关系，针对超短序列优化的扩张卷积设计；2)ES-Seq模块：将无结构事件点组织成结构化时间序列；3)Sobel边缘增强：在稀疏事件条件下增强空间边缘信息。相比之前工作，本文不将事件流转换为密集帧，保留了数据稀疏性和高时间分辨率；相比Chen等人的点云方法，增加了显式时间建模；相比其他稀疏表示方法，保留了更好的空间几何关系建模能力，并通过时间切片和边缘增强提供更丰富的时空特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合时间建模和空间边缘增强的事件点云框架，通过ETSC和ES-Seq模块以及Sobel边缘增强，在不牺牲事件数据稀疏性和高时间分辨率的前提下，显著提高了事件驱动的人体姿态估计性能。'}


### 论文摘要

Human pose estimation focuses on predicting body keypoints to analyze human motion. Event cameras provide high temporal resolution and low latency, enabling robust estimation under challenging conditions. However, most existing methods convert event streams into dense event frames, which adds extra computation and sacrifices the high temporal resolution of the event signal. In this work, we aim to exploit the spatiotemporal properties of event streams based on point cloud-based framework, designed to enhance human pose estimation performance. We design Event Temporal Slicing Convolution module to capture short-term dependencies across event slices, and combine it with Event Slice Sequencing module for structured temporal modeling. We also apply edge enhancement in point cloud-based event representation to enhance spatial edge information under sparse event conditions to further improve performance. Experiments on the DHP19 dataset show our proposed method consistently improves performance across three representative point cloud backbones: PointNet, DGCNN, and Point Transformer.

---

## 13. Beyond Lux thresholds: a systematic pipeline for classifying biologically relevant light contexts from wearable data

**论文链接:** [http://arxiv.org/abs/2512.06181v1](http://arxiv.org/abs/2512.06181v1)

**作者:** Yanuo Zhou

**发布时间:** 2025-12-05

**备注:** 16 pages, 8 figures. Reproducible pipeline for classifying biologically light from wearable spectral data. Manuscript in preparation for journal submission

### GPT解析

### 总结

该研究建立并验证了一种可重复的流程和设计规则，用于从可穿戴光谱数据中区分自然光与人工光，取得了高准确率，并在GitHub和Zenodo上公开了所有代码和配置文件。

### 背景

可穿戴光谱仪能够对生物相关光进行现场量化，但用于上下文分类的可重复流程尚未明确指定。

### 目的

建立并验证一个基于受试者评估的、可重复的流程和可行的设计规则，用于从可穿戴光谱数据中区分自然光与人工光。

### 方法

分析来自26名参与者的ActLumus记录，每个参与者以10秒的采样频率至少监测7天，并配以每日暴露日记。流程包括：域选择、以10为底的对数变换、排除总强度的L2归一化、小时级medoid聚合、正弦/余弦小时编码和MLP分类器，在受试者交叉验证下进行评估。

### 主要发现

提出的序列在自然与人工光分类任务上取得了高性能，代表性配置达到了AUC = 0.938（准确率88%）。相比之下，室内与室外分类由于光谱重叠和类别不平衡，仅达到可行性水平（最佳AUC约0.75）。阈值基线在数据上表现不足，表明需要超越简单照度截止的光谱-时间建模方法。

### 结论

研究提供了一个可重复、可审核的基线流程和设计规则，用于在受试者泛化下的上下文光分类。所有代码、配置文件和衍生工件都将公开存档，以支持后续研究的重用和基准测试。

### 翻译

背景：可穿戴光谱仪能够对生物相关光进行现场量化，但用于上下文分类的可重复流程尚未明确指定。目的：建立并验证一个基于受试者评估的、可重复的流程和可行的设计规则，用于从可穿戴光谱数据中区分自然光与人工光。方法：我们分析了来自26名参与者的ActLumus记录，每个参与者以10秒的采样频率至少监测7天，并配以每日暴露日记。该流程固定了以下顺序：域选择、以10为底的对数变换、排除总强度的L2归一化（避免亮度捷径）、小时级medoid聚合、正弦/余弦小时编码和MLP分类器，在受试者交叉验证下进行评估。结果：提出的序列在主要任务上 consistently 取得了高性能，代表性配置在保留的受试者分割上达到了自然与人工分类的AUC = 0.938（准确率88%）。相比之下，由于光谱重叠和类别不平衡，室内与室外分类仍处于可行性水平（最佳AUC约0.75；在没有上下文传感器的情况下多数类别崩溃）。阈值基线在我们的数据上不足，支持需要超越照度截止的光谱-时间建模。结论：我们提供了一个可重复、可审核的基线流程和设计规则，用于在受试者泛化下的上下文光分类。所有代码、配置文件和衍生工件都将公开存档（GitHub + Zenodo DOI），以支持重用和基准测试。


### 论文摘要

Background: Wearable spectrometers enable field quantification of biologically relevant light, yet reproducible pipelines for contextual classification remain under-specified.   Objective: To establish and validate a subject-wise evaluated, reproducible pipeline and actionable design rules for classifying natural vs. artificial light from wearable spectral data.   Methods: We analysed ActLumus recordings from 26 participants, each monitored for at least 7 days at 10-second sampling, paired with daily exposure diaries. The pipeline fixes the sequence: domain selection, log-base-10 transform, L2 normalisation excluding total intensity (to avoid brightness shortcuts), hour-level medoid aggregation, sine/cosine hour encoding, and MLP classifier, evaluated under participant-wise cross-validation.   Results: The proposed sequence consistently achieved high performance on the primary task, with representative configurations reaching AUC = 0.938 (accuracy 88%) for natural vs. artificial classification on the held-out subject split. In contrast, indoor vs. outdoor classification remained at feasibility level due to spectral overlap and class imbalance (best AUC approximately 0.75; majority-class collapse without contextual sensors). Threshold baselines were insufficient on our data, supporting the need for spectral-temporal modelling beyond illuminance cut-offs.   Conclusions: We provide a reproducible, auditable baseline pipeline and design rules for contextual light classification under subject-wise generalisation. All code, configuration files, and derived artefacts will be openly archived (GitHub + Zenodo DOI) to support reuse and benchmarking.

---

## 14. Inferring Compositional 4D Scenes without Ever Seeing One

**论文链接:** [http://arxiv.org/abs/2512.05272v1](http://arxiv.org/abs/2512.05272v1)

**作者:** Ahmet Berke Gokmen, Ajad Chhatkuli, Luc Van Gool, Danda Pani Paudel

**发布时间:** 2025-12-04

**备注:** Project page: https://github.com/insait-institute/COM4D

### GPT解析

### 总结

本文提出了COM4D方法，能够从单目视频中一致地联合预测多个4D/3D物体的结构和时空配置，无需4D组合训练数据，并在相关任务中达到最先进水平。

### 背景

现实世界场景通常由多个静态和动态物体组成，捕获它们的4维结构、组成和时空配置非常困难。现有方法通常一次只关注一个物体，并依赖于特定类别的参数化形状模型，这可能导致场景配置不一致，且仅限于已建模的物体类别。

### 目的

提出一种名为COM4D（Compositional 4D）的方法，能够一致地联合预测4D/3D物体的结构和时空配置，仅使用静态多物体或动态单物体的监督。

### 方法

通过在2D视频输入上精心设计空间和时间注意力的训练来实现。训练分为两部分：从物体组成中学习和从视频中的单个物体动态中学习，从而避免对4D组合训练数据的依赖。推理时，注意力混合机制结合这些独立学习到的注意力，通过交替进行空间和时间推理，重建完整且持久的4D场景。

### 主要发现

COM4D在现有的4D物体和组合3D重建的独立问题中提供了最先进的结果，尽管它完全是数据驱动的。

### 结论

COM4D是一种有效的方法，能够从单目视频中重建包含多个相互作用的物体的完整4D场景，不需要4D组合训练数据。

### 翻译

现实世界中的场景通常由几个静态和动态物体组成。在野外捕获它们的4维结构、组成和时空配置，虽然非常有趣，但同样非常困难。因此，现有工作通常一次只关注一个物体，同时依赖于针对动态物体的某些特定类别的参数化形状模型。除了仅限于已建模的物体类别外，这还可能导致场景配置不一致。我们提出了COM4D（Compositional 4D）方法，该方法一致地联合预测4D/3D物体的结构和时空配置，仅使用静态多物体或动态单物体监督。我们通过在2D视频输入上精心设计空间和时间注意力的训练来实现这一点。训练被分解为一方面从物体组成中学习，另一方面从整个视频中的单个物体动态中学习，从而完全避免了对4D组合训练数据的依赖。在推理时，我们提出的注意力混合机制结合了这些独立学习到的注意力，不需要任何4D组合示例。通过在空间和时间推理之间交替进行，COM4D可以直接从单目视频中重建完整且持久的4D场景，其中包含多个相互作用的物体。此外，尽管COM4D完全数据驱动，但在现有的4D物体和组合3D重建的独立问题中提供了最先进的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从单目视频中重建包含多个静态和动态物体的完整4D场景（3D空间+时间维度）的问题。这个问题在现实中很重要，因为真实世界场景通常由多个静态和动态物体组成，能够捕捉它们的4D结构、组成和时空配置在计算机视觉、增强现实、机器人导航等领域有广泛应用价值。在研究中也很重要，因为现有方法通常一次只关注一个物体或依赖特定类别的参数化模型，导致场景配置不一致且难以泛化，同时真实世界多物体4D场景数据非常稀少，限制了该领域的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到直接获取真实世界组合4D训练数据的困难，因此采用了一种不同的方法：从两个容易获取的独立来源学习注意力机制——静态多物体观察用于空间结构，单物体动画用于时间动态。在推理时，通过物理假设（每个时间点所有场景元素暂时是静态的，动态通过向前传播物体状态展开）统一这些独立学习的能力。该方法借鉴了TripoSG的图像到网格生成模型、Diffusion Transformer架构、VAE形状表示以及Diffusion Forcing训练方法，但通过创新的注意力解析和混合策略，实现了从未见过组合4D数据的情况下重建复杂场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过注意力解析训练一个单一的DiT模型理解空间组成和动态，然后在推理时通过注意力混合机制将独立学习的能力组合起来生成复杂4D场景。整体流程：1)注意力解析：交替从3D-FRONT和DeformingThings数据集采样，为DiT块分配互补角色（偶数块处理空间关系，奇数块捕获时间依赖）；2)注意力混合：推理时偶数空间块处理全局场景布局，奇数时间块单独处理每个动态物体的运动和变形；3)训练过程：两阶段训练，第一阶段微调模型，第二阶段启用扩散强制以增强时间一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)注意力解析策略，将空间和时间推理从互补数据源中解耦；2)注意力混合机制，在推理时统一独立学习的注意力实现组合4D场景重建；3)统一的注意力框架，能泛化到多样化场景并实现多交互物体的持久4D重建。相比之前工作，该方法不需要直接监督组合4D训练数据、不依赖特定类别的参数化形状模型、不需要测试时优化、能处理多个静态和动态物体的交互场景，并在保持物体分解的同时重建完整4D场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'COM4D通过创新的注意力解析和混合方法，首次实现了从未见过组合4D训练数据的情况下从单目视频中重建多个静态和动态物体的完整持久4D场景，并在相关任务上达到最先进性能。'}


### 论文摘要

Scenes in the real world are often composed of several static and dynamic objects. Capturing their 4-dimensional structures, composition and spatio-temporal configuration in-the-wild, though extremely interesting, is equally hard. Therefore, existing works often focus on one object at a time, while relying on some category-specific parametric shape model for dynamic objects. This can lead to inconsistent scene configurations, in addition to being limited to the modeled object categories. We propose COM4D (Compositional 4D), a method that consistently and jointly predicts the structure and spatio-temporal configuration of 4D/3D objects using only static multi-object or dynamic single object supervision. We achieve this by a carefully designed training of spatial and temporal attentions on 2D video input. The training is disentangled into learning from object compositions on the one hand, and single object dynamics throughout the video on the other, thus completely avoiding reliance on 4D compositional training data. At inference time, our proposed attention mixing mechanism combines these independently learned attentions, without requiring any 4D composition examples. By alternating between spatial and temporal reasoning, COM4D reconstructs complete and persistent 4D scenes with multiple interacting objects directly from monocular videos. Furthermore, COM4D provides state-of-the-art results in existing separate problems of 4D object and composed 3D reconstruction despite being purely data-driven.

---

## 15. VideoMem: Enhancing Ultra-Long Video Understanding via Adaptive Memory Management

**论文链接:** [http://arxiv.org/abs/2512.04540v1](http://arxiv.org/abs/2512.04540v1)

**作者:** Hongbo Jin, Qingyuan Wang, Wenhao Zhang, Yang Liu, Sijie Cheng

**发布时间:** 2025-12-04

### GPT解析

### 总结

VideoMem是一种新颖的框架，通过自适应内存管理将长视频理解作为顺序生成任务处理，显著提升了超长视频理解能力。

### 背景

超长视频理解是一个开放挑战，现有视觉语言模型(VLMs)因上下文长度有限和长期记忆保留效率低，在处理超长视频内容时表现不佳。

### 目的

解决现有VLMs在超长视频理解上的局限性，避免构建外部知识库和检索增强生成系统带来的巨大存储和计算开销。

### 方法

提出VideoMem框架，动态更新全局内存缓冲区以保留关键信息并丢弃冗余内容；集成渐进式分组相对策略优化(PRPO)算法，包含渐进状态传播(PSP)和时间级联奖励(TCR)两个核心模块。

### 主要发现

VideoMem在多种超长视频理解任务的基准测试中显著优于现有的开源模型。

### 结论

VideoMem通过自适应内存管理和创新的训练算法，有效解决了超长视频理解的挑战，为该领域提供了新的解决方案。

### 翻译

超长视频理解仍然是一个开放的挑战，因为现有的视觉语言模型(VLMs)由于上下文长度有限和长期记忆保留效率低，在这样的内容上表现不佳。为解决这一问题，最近的工作试图构建外部知识库和相应的检索增强生成(RAG)系统，但这些方法带来了巨大的存储和计算开销。在本文中，我们提出了VideoMem，一个新颖的框架，它通过自适应内存管理开创性地将长视频理解建模为顺序生成任务。具体来说，VideoMem动态更新全局内存缓冲区，该缓冲区自适应地保留视频时间线上的关键信息，同时丢弃冗余内容。为了高效训练VLMs处理此类长期任务，VideoMem集成了渐进式分组相对策略优化(PRPO)算法，配备了两个核心模块：渐进状态传播(PSP)自适应地保留有效当前状态，将它们传播到下一个回滚步骤，并逐渐缩小模型探索空间。时间级联奖励(TCR)进一步缓解了奖励稀疏性，提高了样本利用率并加速了收敛。大量实验表明，VideoMem在多种超长视频理解任务的基准测试中显著优于现有的开源模型。


### 论文摘要

Ultra long video understanding remains an open challenge, as existing vision language models (VLMs) falter on such content due to limited context length and inefficient long term memory retention. To address this, recent works have attempted to construct external knowledge bases and corresponding retrieval agumented generation (RAG) systems, yet these incur enormous storage and computational overhead. In this paper, we propose VideoMem, a novel framework that pioneers models long video understanding as a sequential generation task via adaptive memory management. Specifically, VideoMem dynamically updates a global memory buffer, which adaptively retains critical information while discarding redundant content across the video timeline. To efficiently train VLMs for such long-term tasks, VideoMem integrates the Progressive Grouped Relative Policy Optimization (PRPO) algorithm, equipped with two core modules: Progressive State Propagation (PSP) adaptively retains valid current states, propagates them to the next rollout step, and gradually narrows the model exploration space. Temporal Cascading Reward (TCR) further alleviates reward sparsity, improving sample utilization and accelerating convergence. Extensive experiments demonstrate that VideoMem significantly outperforms existing open-source models across diverse benchmarks for ultra-long video understanding tasks.

---

## 16. PhyVLLM: Physics-Guided Video Language Model with Motion-Appearance Disentanglement

**论文链接:** [http://arxiv.org/abs/2512.04532v1](http://arxiv.org/abs/2512.04532v1)

**作者:** Yu-Wei Zhan, Xin Wang, Hong Chen, Tongtong Feng, Wei Feng, Ren Wang, Guangyao Li, Qing Li, Wenwu Zhu

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了PhyVLLM，一种物理引导的视频语言框架，通过解耦视觉外观和物体运动，并利用神经常微分方程模块建模物理动态，显著提升了Video LLMs在物理推理和一般视频理解任务上的性能。

### 背景

Video LLMs在多种视频语言任务中表现出色，但在需要更深层次物理动态理解的情况下往往失败，这主要是因为它们依赖于基于外观的匹配。

### 目的

解决将物理运动建模纳入Video LLMs时面临的三个关键挑战：运动信号与外观变化纠缠、有效运动建模需要连续时间和物理动态表示、收集物理属性标注成本高昂且不切实际。

### 方法

提出PhyVLLM框架，通过双分支编码器解耦视觉外观和物体运动；集成神经常微分方程模块生成可微分的物理动态表示；将运动感知表示投影到预训练LLM的标记空间；采用自监督方式建模物体运动的连续演化，避免对显式物理标签的需求。

### 主要发现

PhyVLLM在物理推理和一般视频理解任务上都显著优于最先进的Video LLMs，突出了整合显式物理建模的优势。

### 结论

将显式物理建模整合到Video LLMs中可以有效提升模型在需要物理理解的任务上的性能，同时保持模型原有的多模态能力。

### 翻译

视频大语言模型在广泛的各种视频语言任务中表现出色。然而，它们在需要更深层次理解物理动态的场景中往往失败。这一限制主要源于它们对基于外观匹配的依赖。将物理运动建模纳入视频理解至关重要，但带来了三个关键挑战：(1)运动信号通常与外观变化纠缠在一起，难以提取清晰的物理线索；(2)有效的运动建模不仅需要连续时间的运动表示，还需要捕捉物理动态；(3)为物理属性收集准确的标注既昂贵又不切实际。为解决这些问题，我们提出了PhyVLLM，一种物理引导的视频语言框架，明确将物理运动整合到Video LLMs中。具体来说，PhyVLLM通过双分支编码器解耦视觉外观和物体运动。为了随时间建模物理动态，我们集成了神经常微分方程模块，生成可微分的物理动态表示。所得到的运动感知表示被投影到预训练LLM的标记空间中，使模型能够进行物理推理，同时不损害模型原有的多模态能力。为避免对显式物理标签的需求，PhyVLLM采用自监督方式来建模物体运动的连续演化。实验结果表明，PhyVLLM在物理推理和一般视频理解任务上都显著优于最先进的Video LLMs，突出了整合显式物理建模的优势。


### 论文摘要

Video Large Language Models (Video LLMs) have shown impressive performance across a wide range of video-language tasks. However, they often fail in scenarios requiring a deeper understanding of physical dynamics. This limitation primarily arises from their reliance on appearance-based matching. Incorporating physical motion modeling is crucial for deeper video understanding, but presents three key challenges: (1) motion signals are often entangled with appearance variations, making it difficult to extract clean physical cues; (2) effective motion modeling requires not only continuous-time motion representations but also capturing physical dynamics; and (3) collecting accurate annotations for physical attributes is costly and often impractical. To address these issues, we propose PhyVLLM, a physical-guided video-language framework that explicitly incorporates physical motion into Video LLMs. Specifically, PhyVLLM disentangles visual appearance and object motion through a dual-branch encoder. To model physical dynamics over time, we incorporate a Neural Ordinary Differential Equation (Neural ODE) module, which generates differentiable physical dynamic representations. The resulting motion-aware representations are projected into the token space of a pretrained LLM, enabling physics reasoning without compromising the model's original multimodal capabilities. To circumvent the need for explicit physical labels, PhyVLLM employs a self-supervised manner to model the continuous evolution of object motion. Experimental results demonstrate that PhyVLLM significantly outperforms state-of-the-art Video LLMs on both physical reasoning and general video understanding tasks, highlighting the advantages of incorporating explicit physical modeling.

---

## 17. StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios

**论文链接:** [http://arxiv.org/abs/2512.04451v1](http://arxiv.org/abs/2512.04451v1)

**作者:** Yifei Wang, Zhenkai Li, Tianwen Qian, Huanran Zheng, Zheng Wang, Yuqian Fu, Xiaoling Wang

**发布时间:** 2025-12-04

### GPT解析

### 总结

StreamEQA是首个针对具身场景中流式视频问答的基准测试，从具身和流式两个维度评估模型能力，揭示了现有模型在此方面的不足，旨在推动相关研究。

### 背景

具身智能正在向实际部署发展，持续感知和推理流式视觉输入的能力变得至关重要。智能体需要保持对环境的情境感知，理解与周围实体的交互，并根据过去的观察、当前情境和预期未来事件动态规划行动。

### 目的

为了促进具身智能中流式视频理解的研究，引入首个专为具身场景中流式视频问答设计的基准测试StreamEQA。

### 方法

StreamEQA从两个正交维度评估多模态大语言模型：具身维度(感知、交互、规划)和流式维度(向后、实时、向前推理)。基于156个独立长视频，定义42个任务，通过混合流程生成约21K个带精确时间戳的问答对。

### 主要发现

对13个最先进视频-LLMs的评估显示，尽管这些模型在传统基准测试上表现出色，但在具身场景的流式视频理解方面仍然存在困难。

### 结论

StreamEQA将促进具身应用中流式视频理解的研究发展。

### 翻译

随着具身智能向实际部署发展，持续感知和推理流式视觉输入的能力变得至关重要。在这样的环境中，智能体必须保持对环境的情境感知，理解与周围实体的交互，并根据过去的观察、当前情境和预期未来事件动态规划行动。为了促进这一方向的进展，我们引入了StreamEQA，这是首个专为具身场景中流式视频问答设计的基准测试。StreamEQA沿着两个正交维度评估现有的多模态大语言模型：具身维度和流式维度。在具身维度上，我们将问题分为三个层次：感知、交互和规划，这些层次逐步评估模型识别细粒度视觉细节、推理智能体-对象交互和执行高级目标导向推理的能力。对于流式维度，问题分为向后、实时和向前推理，每种模式依赖不同的时间上下文。基于156个独立的长视频，StreamEQA定义了42个任务，并通过结合自动生成和人工优化的混合流程生成了约21K个带有精确时间戳的问答对。对13个最先进的视频-LLMs的评估显示，尽管这些模型在传统基准测试上表现出色，但在具身场景的流式视频理解方面仍然存在困难。我们希望StreamEQA能够促进具身应用中流式视频理解的研究。


### 论文摘要

As embodied intelligence advances toward real-world deployment, the ability to continuously perceive and reason over streaming visual inputs becomes essential. In such settings, an agent must maintain situational awareness of its environment, comprehend the interactions with surrounding entities, and dynamically plan actions informed by past observations, current contexts, and anticipated future events. To facilitate progress in this direction, we introduce StreamEQA, the first benchmark designed for streaming video question answering in embodied scenarios. StreamEQA evaluates existing MLLMs along two orthogonal dimensions: Embodied and Streaming. Along the embodied dimension, we categorize the questions into three levels: perception, interaction, and planning, which progressively assess a model's ability to recognize fine-grained visual details, reason about agent-object interactions, and perform high-level goal-directed reasoning. For the streaming dimension, questions are divided into backward, real-time, and forward reasoning, with each mode relying on a distinct temporal context. Built upon 156 independent long videos, StreamEQA defines 42 tasks and generates approximately 21K question-answer pairs with precise timestamps through a hybrid pipeline combining automated generation and human refinement. Evaluations of 13 state-of-the-art video-LLMs reveal that, despite strong performance on conventional benchmarks, these models still struggle with streaming video understanding in embodied scenarios. We hope StreamEQA will catalyze research on streaming video understanding for embodied applications.

---

## 18. TempR1: Improving Temporal Understanding of MLLMs via Temporal-Aware Multi-Task Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.03963v2](http://arxiv.org/abs/2512.03963v2)

**作者:** Tao Wu, Li Yang, Gen Zhan, Yabin Zhang, Yiting Liao, Junlin Li, Deliang Fu, Li Zhang, Limin Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究提出了TempR1框架，通过时间感知的多任务强化学习方法增强多模态大语言模型的时间理解能力，在多个基准测试中取得了最先进性能。

### 背景

增强多模态大语言模型的时间理解对长视频分析至关重要，可支持时间定位、动作检测和时间敏感问答等任务。现有强化学习方法通常局限于有限任务类型和数据，限制了泛化能力。

### 目的

提出一个时间感知的多任务强化学习框架，系统地加强多模态大语言模型的时间理解能力，使其能够适应多样化的时间理解场景。

### 方法

整合多任务语料库使模型接触多样化时间结构和语义；基于Group Relative Policy Optimization算法实现跨任务优化；将时间任务分为三种预测区间与真实实例对应类型；为每种类型设计特定定位奖励，捕捉细粒度时间依赖关系。

### 主要发现

TempR1在多个基准测试中取得最先进性能；跨互补任务的联合优化产生强大协同效应，同时增强泛化能力和单任务性能。

### 结论

TempR1为多模态大语言模型中的时间推理建立了可扩展且原则性的新范式。

### 翻译

增强多模态大语言模型的时间理解对于推进长视频分析至关重要，能够支持时间定位、动作检测和时间敏感问答等任务。虽然强化学习最近被探索用于改善时间推理，但现有方法通常局限于有限的任务类型和数据，限制了它们在多样化时间理解场景中的泛化能力。为应对这一挑战，我们提出了TempR1，一个时间感知的多任务强化学习框架，系统地加强多模态大语言模型的时间理解能力。我们整理了一个多任务语料库，使模型接触多样化的时间结构和语义，并基于组相对策略优化算法实现稳定有效的跨任务优化。具体来说，我们将时间任务分为预测区间与真实实例之间的三种对应类型，并为每种类型设计特定的定位奖励，使TempR1能够捕捉细粒度的时间依赖关系并适应不同的时间模式。大量实验证明，TempR1在多个基准测试中取得了最先进的性能。此外，跨互补任务的联合优化产生了强大的协同效应，同时增强了泛化能力和单任务性能，为多模态大语言模型中的时间推理建立了可扩展且原则性的范式。


### 论文摘要

Enhancing the temporal understanding of Multimodal Large Language Models (MLLMs) is essential for advancing long-form video analysis, enabling tasks such as temporal localization, action detection, and time-sensitive question answering. While reinforcement learning (RL) has recently been explored for improving temporal reasoning, existing approaches are often confined to limited task types and data, restricting their generalization across diverse temporal understanding scenarios. To address this challenge, we present TempR1, a temporal-aware multi-task reinforcement learning framework that systematically strengthens MLLMs' temporal comprehension. We curate a multi-task corpus that exposes the model to diverse temporal structures and semantics, and build upon the Group Relative Policy Optimization (GRPO) algorithm to achieve stable and effective cross-task optimization. Specifically, we categorize temporal tasks into three correspondence types between predicted intervals and ground-truth instances, and design tailored localization rewards for each, enabling TempR1 to capture fine-grained temporal dependencies and adapt to different temporal patterns. Extensive experiments demonstrate that TempR1 attains state-of-the-art performance across multiple benchmarks. Moreover, its joint optimization over complementary tasks yields a strong synergistic effect, enhancing both generalization and single-task performance, establishing a scalable and principled paradigm for temporal reasoning in MLLMs.

---

## 19. A Strong View-Free Baseline Approach for Single-View Image Guided Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2506.15747v2](http://arxiv.org/abs/2506.15747v2)

**作者:** Fangzhou Lin, Zilin Dai, Rigved Sanku, Songlin Hou, Kazunori D Yamada, Haichong K. Zhang, Ziming Zhang

**发布时间:** 2025-06-18

**备注:** 7 pages, 2 figures

### GPT解析

### 总结

这篇论文探索了单视图图像引导在点云补全任务中的必要性，提出了一种不依赖视图的强基线方法，仅使用部分点云作为输入，通过注意力机制和多流信息融合，有效提升了点云补全的性能，实验证明该方法优于现有最先进技术。

### 背景

单视图图像引导的点云补全（SVIPC）任务旨在借助单视图图像从不完整的输入点云中重建完整的点云。虽然之前的研究已经证明了这种多模态方法的有效性，但图像引导的基本必要性在很大程度上尚未被检验。

### 目的

探索图像引导在SVIPC任务中的必要性，并提出一个不依赖视图的强基线方法。

### 方法

提出了一种基于注意力的多分支编码器-解码器网络，仅将部分点云作为输入，不依赖视图。使用由交叉注意力和自注意力层驱动的层次自融合机制，有效地跨多个流集成信息，丰富特征表示并增强网络捕获几何结构的能力。

### 主要发现

在ShapeNet-ViPC数据集上进行的大量实验和消融研究表明，不依赖视图的框架优于最先进的SVIPC方法。

### 结论

研究结果为SVIPC中多模态学习的发展提供了新的见解，演示代码将在https://github.com/Zhang-VISLab上提供。

### 翻译

单视图图像引导的点云补全（SVIPC）任务旨在借助单视图图像从不完整的输入点云中重建完整的点云。虽然之前的研究已经证明了这种多模态方法的有效性，但图像引导的基本必要性在很大程度上尚未被检验。为此，我们提出了一种基于注意力的多分支编码器-解码器网络作为SVIPC的强基线方法，该方法仅将部分点云作为输入，不依赖视图。我们的层次自融合机制由交叉注意力和自注意力层驱动，有效地跨多个流集成信息，丰富了特征表示并增强了网络捕获几何结构的能力。在ShapeNet-ViPC数据集上进行的大量实验和消融研究表明，我们的不依赖视图框架优于最先进的SVIPC方法。我们希望研究结果能为SVIPC中多模态学习的发展提供新的见解。我们的演示代码将在https://github.com/Zhang-VISLab上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要质疑并探索单视图图像引导在点云补全任务中的必要性。这个问题很重要，因为现有的多模态方法依赖图像输入增加了系统复杂度和计算成本，且图像质量、视角和校准问题会影响性能；在某些场景下（如缺乏纹理的3D形状或模糊图像），图像可能无法提供足够的几何信息；探索仅使用点云的方法可以简化系统架构，提高实际应用中的可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先回顾点云补全研究，发现大多数方法使用编码器-解码器架构，注意到多模态方法使用图像作为额外输入但质疑其必要性。作者意识到现有研究很少探索单视图图像引导的必要性，从而提出'无视图'方法。设计时借鉴了PointNet++作为编码器基础，受SnowflakeNet启发但保留中间特征，采用类似XMFNet的解码器架构，并使用注意力机制融合多分支特征。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'自融合'：通过多分支编码器处理部分点云，使用交叉注意力和自注意力机制融合不同分支特征，不依赖外部图像。实现流程：1)输入不完整点云；2)3D编码器提取多尺度层次特征；3)多分支独立处理点云；4)自融合网络通过交叉注意力和自注意力融合特征；5)融合特征连接；6)解码器重建完整点云；7)使用Chamfer Distance作为损失函数。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)提出首个'无视图'强基线方法挑战图像引导的必要性；2)设计基于注意力的多分支自融合机制；3)层次化自融合机制通过交叉注意力和自注意力捕获几何结构；4)证明仅用点云数据就能达到甚至超过多模态方法性能。不同之处：无需图像输入；使用注意力机制而非简单融合策略；保留中间特征表示；在多类别点云补全上取得更好性能。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于注意力的多分支自融合网络，证明仅使用部分点云数据作为输入，无需任何图像引导，就能在点云补全任务上达到甚至超过当前最先进的多模态方法的性能，为点云补全领域提供了新的见解和强基线。'}


### 论文摘要

The single-view image guided point cloud completion (SVIPC) task aims to reconstruct a complete point cloud from a partial input with the help of a single-view image. While previous works have demonstrated the effectiveness of this multimodal approach, the fundamental necessity of image guidance remains largely unexamined. To explore this, we propose a strong baseline approach for SVIPC based on an attention-based multi-branch encoder-decoder network that only takes partial point clouds as input, view-free. Our hierarchical self-fusion mechanism, driven by cross-attention and self-attention layers, effectively integrates information across multiple streams, enriching feature representations and strengthening the networks ability to capture geometric structures. Extensive experiments and ablation studies on the ShapeNet-ViPC dataset demonstrate that our view-free framework performs superiorly to state-of-the-art SVIPC methods. We hope our findings provide new insights into the development of multimodal learning in SVIPC. Our demo code will be available at https://github.com/Zhang-VISLab.

---

## 20. Online Segment Any 3D Thing as Instance Tracking

**论文链接:** [http://arxiv.org/abs/2512.07599v1](http://arxiv.org/abs/2512.07599v1)

**作者:** Hanshi Wang, Zijian Cai, Jin Gao, Yiwei Zhang, Weiming Hu, Ke Wang, Zhipeng Zhang

**发布时间:** 2025-12-08

**备注:** NeurIPS 2025, Code is at https://github.com/AutoLab-SAI-SJTU/AutoSeg3D

### GPT解析

### 总结

本文提出了一种名为AutoSeg3D的方法，将在线3D分割重新概念化为实例跟踪问题，通过对象查询实现时间信息传播，显著提升了具身智能体的环境感知能力。

### 背景

在线、实时和细粒度的3D分割是具身智能体感知和理解操作环境的基本能力。现有方法使用预定义对象查询从视觉基础模型(VFMs)聚合语义信息，但忽视了感知过程中的时间维度。

### 目的

为了进一步释放具身代理的时间环境感知能力，将在线3D分割重新概念化为实例跟踪问题。

### 方法

利用对象查询进行时间信息传播，其中长期实例关联促进特征和对象身份的连贯性，短期实例更新丰富即时观察。同时引入空间一致性学习减轻VFMs的碎片化问题，为时间学习提供更全面的实例信息。

### 主要发现

稀疏对象查询促进的时间信息交换和一致性学习增强了空间理解，避免了密集时间点云交互的计算负担。该方法在ScanNet200上比ESAM高出2.8 AP，并在多个数据集上取得一致提升。

### 结论

将3D分割视为实例跟踪问题并利用对象查询进行时间信息传播，有效提升了在线3D分割的性能，建立了新的最先进水平。

### 翻译

在线、实时和细粒度的3D分割构成了具身智能体感知和理解其操作环境的基本能力。最近的进展采用预定义的对象查询来聚合从视觉基础模型(VFMs)输出的语义信息，这些信息被提升到3D点云中，通过查询间交互促进空间信息传播。然而，感知本质上是一个动态过程，使得时间理解在这些主流的基于查询的管道中成为一个被忽视的关键维度。因此，为了进一步释放具身代理的时间环境感知能力，我们的工作将在线3D分割重新概念化为一个实例跟踪问题(AutoSeg3D)。我们的核心策略是利用对象查询进行时间信息传播，其中长期实例关联促进特征和对象身份的连贯性，而短期实例更新丰富了即时观察。考虑到具身机器人中的视角变化通常会导致跨帧的部分物体可见性，这种机制有助于模型开发超越不完整瞬时视图的整体物体理解。此外，我们引入空间一致性学习来减轻VFMs固有的碎片化问题，为增强长期和短期时间学习提供更全面的实例信息。这些稀疏对象查询促进的时间信息交换和一致性学习不仅增强了空间理解，还避免了密集时间点云交互带来的计算负担。我们的方法建立了新的最先进水平，在ScanNet200上比ESAM高出2.8 AP，并在ScanNet、SceneNN和3RScan数据集上取得了一致的提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在线、实时和细粒度的3D分割问题中缺乏时间维度理解的问题。当前方法使用预定义的对象查询聚合2D视觉基础模型的输出到3D点云，但忽视了时间维度的连贯性。这个问题在现实中很重要，因为自主机器人等具身智能体需要在动态环境中实时理解和分割3D场景，缺乏时间连贯性会导致过度分割，影响系统的实时性和鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类和机器人感知是动态过程的本质出发，借鉴了多目标跟踪（MOT）、视频实例分割（如VisTR）和3D检测模型（如Sparse4D）等现有工作。这些方法通过空间连续性和外观相似性实现跨帧检测的一致性。作者受大脑互补学习系统启发，将框架分解为长期记忆（LTM）用于实例关联和短期记忆（STM）用于实例更新，并设计了三个轻量级协同模块：LTM、STM和空间一致性学习（SCL）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将在线3D实例分割重新定义为实例跟踪问题，通过引入时间维度增强分割的连贯性。整体流程包括：1) 使用视觉前端模块生成初始分割；2) 通过SCL模块处理过度分割问题，合并高亲和力掩码片段；3) 通过STM模块注入短期时间上下文，使用距离感知注意力过滤背景噪声；4) 通过LTM模块进行长期实例关联，处理长时间遮挡后的实例重新识别。这三个模块协同工作，实现时间连贯的3D分割。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 将在线3D分割重新定义为实例跟踪问题；2) 设计三个协同模块（LTM、STM、SCL）；3) 空间一致性学习机制，包括学习掩码集成和实例一致性掩码监督；4) 距离感知的短期记忆机制。相比之前工作（如ESAM），本文显式建模时间维度，专门处理VFM的过度分割问题，使用双分支解码器增强鲁棒性，并通过稀疏对象查询提高计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种将在线3D实例分割重新定义为实例跟踪问题的新方法，通过三个协同模块显著提高了具身智能体在动态环境中的3D分割性能和鲁棒性，同时保持了实时效率。'}


### 论文摘要

Online, real-time, and fine-grained 3D segmentation constitutes a fundamental capability for embodied intelligent agents to perceive and comprehend their operational environments. Recent advancements employ predefined object queries to aggregate semantic information from Vision Foundation Models (VFMs) outputs that are lifted into 3D point clouds, facilitating spatial information propagation through inter-query interactions. Nevertheless, perception is an inherently dynamic process, rendering temporal understanding a critical yet overlooked dimension within these prevailing query-based pipelines. Therefore, to further unlock the temporal environmental perception capabilities of embodied agents, our work reconceptualizes online 3D segmentation as an instance tracking problem (AutoSeg3D). Our core strategy involves utilizing object queries for temporal information propagation, where long-term instance association promotes the coherence of features and object identities, while short-term instance update enriches instant observations. Given that viewpoint variations in embodied robotics often lead to partial object visibility across frames, this mechanism aids the model in developing a holistic object understanding beyond incomplete instantaneous views. Furthermore, we introduce spatial consistency learning to mitigate the fragmentation problem inherent in VFMs, yielding more comprehensive instance information for enhancing the efficacy of both long-term and short-term temporal learning. The temporal information exchange and consistency learning facilitated by these sparse object queries not only enhance spatial comprehension but also circumvent the computational burden associated with dense temporal point cloud interactions. Our method establishes a new state-of-the-art, surpassing ESAM by 2.8 AP on ScanNet200 and delivering consistent gains on ScanNet, SceneNN, and 3RScan datasets.

---

## 21. A graph generation pipeline for critical infrastructures based on heuristics, images and depth data

**论文链接:** [http://arxiv.org/abs/2512.07269v1](http://arxiv.org/abs/2512.07269v1)

**作者:** Mike Diessner, Yannick Tarant

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究提出了一种基于摄影测量的图生成管道，用于创建关键基础设施的虚拟表示，比传统的激光扫描方法更经济高效。

### 背景

物理关键基础设施（如水厂或能源厂）的虚拟表示用于模拟和数字孪生，以确保服务的弹性和连续性。传统方法需要使用激光扫描仪获取3D点云，这些方法成本高昂且需要专业知识。

### 目的

开发一种更经济高效的图生成方法，用于创建关键基础设施的虚拟表示，替代昂贵的激光扫描方法。

### 方法

使用双目相机生成的RGB图像和深度数据，通过深度学习进行目标检测和实例分割，并采用用户定义的启发式或规则来推断对象之间的关系，构建图结构。

### 主要发现

两个液压系统的测试结果表明，该方法生成的图接近真实情况，同时具有灵活性，可针对特定应用进行定制，且透明度高，适用于关键基础设施的高风险决策场景。

### 结论

基于摄影测量的图生成管道是一种更经济、灵活且透明的方法，适用于关键基础设施的虚拟表示和决策支持。

### 翻译

物理关键基础设施（如水厂或能源厂）的虚拟表示用于模拟和数字孪生，以确保其服务的弹性和连续性。这些模型通常需要来自激光扫描仪的3D点云，这些点云获取成本高昂且需要专业知识才能使用。在本文中，我们提出了一种基于摄影测量的图生成管道。该管道使用双目相机生成的RGB图像和深度数据，检测相关对象并预测它们之间的关系。这种更经济有效的方法使用深度学习进行目标检测和对象实例分割，并采用用户定义的启发式或规则来推断对象之间的关系。两个液压系统的结果表明，该策略可以生成接近真实情况的图，同时其灵活性允许该方法针对特定应用进行定制，并且其透明度使其能够用于关键基础设施所需的高风险决策。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决为关键基础设施（如水力或能源设施）创建虚拟表示的问题。目前这类模型通常需要使用昂贵的激光扫描仪获取3D点云，成本高且需要专业知识。这个问题很重要，因为关键基础设施是国家和社会生存运作的核心，其中断会导致供应短缺、健康风险、经济混乱等严重后果。准确高效的虚拟模型对于监控、维护和优化这些基础设施，以及为紧急情况做准备至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到激光扫描方法成本高昂且需要专业知识，因此寻求更经济实惠的替代方案。他们选择了摄影测量法（RGB图像和深度数据）作为数据源，结合深度学习进行目标检测和实例分割，并采用用户定义的启发式规则推断对象关系。作者借鉴了现有工作，如使用YOLOv8模型（基于COCO数据集预训练）、DBSCAN算法进行聚类、针孔相机模型进行2D到3D投影，但创新点在于将这些技术组合成一个专门针对关键基础设施图生成的完整流程，并强调方法的透明度和可解释性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用摄影测量法结合深度学习和启发式规则，以经济实惠的方式生成关键基础设施的图表示。整体流程包括：1)数据收集：使用立体相机系统获取RGB图像和深度数据；2)目标检测：非管道对象使用YOLOv8姿态模型，管道对象使用YOLOv8实例分割模型；3)对象匹配和清理：跨图像匹配相同对象，去除噪声；4)端点估计：预测对象的连接点；5)图生成：基于端点距离创建初始图，应用规则细化；6)结果评估：与真实值比较分析差异。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用经济实惠的摄影测量法替代昂贵的激光扫描；2)结合深度学习与启发式规则，平衡了性能与可解释性；3)区分处理管道和非管道对象，提高准确性；4)提供可定制的规则系统，适应不同应用场景；5)强调透明度和可解释性，支持高风险决策。相比之前工作，不同之处在于：数据源从点云转向图像处理；技术路线从纯深度学习转向混合方法；关系推断从黑盒模型转向可解释规则；专门针对关键基础设施应用；创新性地区分了管道和非管道对象的处理方式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于摄影测量、深度学习和启发式规则的经济实惠且可解释的图生成方法，用于创建关键基础设施的虚拟表示，支持模拟和数字孪生应用，同时保持足够的准确性和透明度以支持高风险决策。'}


### 论文摘要

Virtual representations of physical critical infrastructures, such as water or energy plants, are used for simulations and digital twins to ensure resilience and continuity of their services. These models usually require 3D point clouds from laser scanners that are expensive to acquire and require specialist knowledge to use. In this article, we present a graph generation pipeline based on photogrammetry. The pipeline detects relevant objects and predicts their relation using RGB images and depth data generated by a stereo camera. This more cost-effective approach uses deep learning for object detection and instance segmentation of the objects, and employs user-defined heuristics or rules to infer their relations. Results of two hydraulic systems show that this strategy can produce graphs close to the ground truth while its flexibility allows the method to be tailored to specific applications and its transparency qualifies it to be used in the high stakes decision-making that is required for critical infrastructures.

---

## 22. Object Pose Distribution Estimation for Determining Revolution and Reflection Uncertainty in Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.07211v1](http://arxiv.org/abs/2512.07211v1)

**作者:** Frederik Hagelskjær, Dimitrios Arapis, Steffen Madsen, Thorbjørn Mosekjær Iversen

**发布时间:** 2025-12-08

**备注:** 8 pages, 8 figures, 5 tables, ICCR 2025

### GPT解析

### 总结

本文提出了一种基于神经网络的新方法，仅使用3D无颜色数据估计物体姿态不确定性，解决了工业环境中颜色信息缺失的问题，并在实际拣选场景中进行了验证。

### 背景

物体姿态估计对机器人感知至关重要，但传统方法通常提供单一姿态估计，无法捕捉视觉模糊引起的不确定性。现有姿态分布方法严重依赖颜色信息，而这在工业环境中通常不可用。

### 目的

开发一种仅使用3D无颜色数据估计物体姿态不确定性的神经网络方法，这是首个不依赖RGB输入而利用深度学习进行姿态分布估计的方法。

### 方法

提出基于神经网络的姿态不确定性估计方法，仅使用3D无颜色数据，在具有不同几何模糊性的物体的实际拣选场景中进行验证。当前实现专注于反射和旋转对称性，但框架可扩展到完整的SE(3)姿态分布估计。

### 主要发现

仅使用3D无颜色数据可以有效估计物体姿态不确定性，该方法在具有不同几何模糊性的物体上表现良好。

### 结论

提出了一种创新的姿态不确定性估计方法，解决了工业环境中颜色信息缺失的问题，源代码可在opde3d.github.io获取。

### 翻译

物体姿态估计对机器人感知至关重要，通常提供单一姿态估计。然而，单一估计无法捕捉视觉模糊引起的姿态不确定性，这可能导致不可靠的行为。现有的姿态分布方法严重依赖颜色信息，这在工业环境中通常不可用。我们提出了一种新颖的基于神经网络的方法，仅使用3D无颜色数据估计物体姿态不确定性。据我们所知，这是首个不依赖RGB输入而利用深度学习进行姿态分布估计的方法。我们在具有不同几何模糊性的物体的实际拣选场景中验证了我们的方法。我们当前的实现专注于反射和旋转对称性，但该框架可扩展到完整的SE(3)姿态分布估计。源代码可在opde3d.github.io获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在3D点云数据中估计物体姿态分布的问题，特别是处理由于物体对称性（如圆柱形物体）导致的旋转和反射不确定性。这个问题在工业制造中非常重要，因为传统姿态估计只提供单一姿态预测，无法捕捉视觉模糊性，可能导致机器人抓取失败，损坏物体、设备或机器人。在工业环境中，3D传感器比RGB图像更常用，因为颜色信息可能不可靠或受光照条件影响，因此需要不依赖颜色信息的方法来处理姿态不确定性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了单一姿态估计的局限性，认识到现有方法大多依赖RGB图像，但在工业环境中3D数据更常用。他们借鉴了SpyroPose方法，这是一个基于RGB图像的姿态分布估计方法，并将其适应到3D点云数据。作者设计了特征聚合器来结合空间信息和特征嵌入，使用神经网络评估不同姿态的可能性。他们限制了搜索空间为旋转和反射，简化了问题并生成完整的旋转不确定性直方图。方法针对工业环境进行了调整，不依赖颜色信息，使方法独立于光照条件，专注于实际工业应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用神经网络估计物体姿态的概率分布，而不只是单一姿态，专注于处理由于物体对称性导致的旋转和反射不确定性，仅使用3D点云数据不依赖颜色信息。整体流程包括：1)从CAD模型采样关键点；2)使用类似PointNet的结构(如DGCNN)对场景进行编码；3)在编码点云中找到与变换后关键点最近的点；4)通过特征聚合器结合空间信息和特征嵌入；5)使用多层感知机评估姿态可能性；6)训练时应用数据增强；7)推理时获取初始姿态并计算姿态分布；8)根据任务需求定义不确定性接受策略。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个仅使用3D点云的姿态分布估计方法，不依赖RGB输入；2)设计的特征聚合器结合空间信息和特征嵌入；3)专注于旋转和反射不确定性，适合处理圆柱形物体；4)在实际工业场景(bin picking)中验证方法。相比之前工作，不同之处在于：输入数据类型不同(仅使用3D点云而非RGB图像)；应用场景不同(针对工业环境和对称物体)；技术实现不同(使用关键点最近邻采样而非图像投影)；不确定性处理不同(专注于旋转和反射而非完整6自由度)；验证方式不同(测试实际抓取性能而非仅姿态精度)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的基于神经网络的3D点云处理方法，首次在没有RGB信息的情况下估计物体姿态分布，有效解决了工业环境中由物体对称性引起的旋转和反射不确定性问题，提高了机器人系统在视觉模糊场景中的可靠性和鲁棒性。'}


### 论文摘要

Object pose estimation is crucial to robotic perception and typically provides a single-pose estimate. However, a single estimate cannot capture pose uncertainty deriving from visual ambiguity, which can lead to unreliable behavior. Existing pose distribution methods rely heavily on color information, often unavailable in industrial settings.   We propose a novel neural network-based method for estimating object pose uncertainty using only 3D colorless data. To the best of our knowledge, this is the first approach that leverages deep learning for pose distribution estimation without relying on RGB input. We validate our method in a real-world bin picking scenario with objects of varying geometric ambiguity. Our current implementation focuses on symmetries in reflection and revolution, but the framework is extendable to full SE(3) pose distribution estimation. Source code available at opde3d.github.io

---

## 23. Hierarchical Image-Guided 3D Point Cloud Segmentation in Industrial Scenes via Multi-View Bayesian Fusion

**论文链接:** [http://arxiv.org/abs/2512.06882v1](http://arxiv.org/abs/2512.06882v1)

**作者:** Yu Zhu, Naoya Chiba, Koichi Hashimoto

**发布时间:** 2025-12-07

**备注:** Accepted to BMVC 2025 (Sheffield, UK, Nov 24-27, 2025). Supplementary video and poster available upon request

### GPT解析

### 总结

本文提出了一种分层的图像引导三维分割框架，用于解决工业环境中密集布局和多尺度物体的三维分割问题，有效处理遮挡和结构复杂性。

### 背景

可靠的三维分割对于理解具有密集布局和多尺度物体的复杂场景（如工业环境）至关重要。在这些场景中，严重的遮挡会削弱物体之间的几何边界，而物体尺度的巨大差异会导致端到端模型无法准确捕捉粗略和精细细节。

### 目的

解决现有三维分割方法的局限性，包括基于三维点的方法需要昂贵标注，以及图像引导方法存在跨视图语义不一致的问题。

### 方法

提出了一种分层的图像引导三维分割框架，从实例级别到部分级别逐步细化分割。实例分割通过渲染俯视图图像并投影SAM生成的掩码到三维点云实现；部分分割则通过渲染多视图图像、应用二维分割和后投影，并进行贝叶斯更新融合以确保跨视图语义一致性。

### 主要发现

在真实工厂数据上的实验表明，该方法能有效处理遮挡和结构复杂性，实现持续的高类别平均交并比分数。在公共数据集上的额外评估确认了该框架的泛化能力。

### 结论

该框架具有鲁棒性、标注效率和适应不同三维环境的能力，为工业环境中的三维分割提供了有效解决方案。

### 翻译

可靠的三维分割对于理解具有密集布局和多尺度物体的复杂场景至关重要，这在工业环境中很常见。在这种情况下，严重的遮挡会削弱物体之间的几何边界，而物体尺度的巨大差异将导致端到端模型无法准确捕捉粗略和精细细节。现有的基于三维点的方法需要昂贵的标注，而图像引导的方法通常存在跨视图的语义不一致。为了应对这些挑战，我们提出了一种分层的图像引导三维分割框架，从实例级别到部分级别逐步细化分割。实例分割涉及渲染俯视图图像并将由YOLO-World提示的SAM生成的掩码投影回三维点云。随后通过渲染从上一阶段获得的每个实例的多视图图像，在每个视图应用相同的二维分割和后投影过程，然后进行贝叶斯更新融合以确保跨视图的语义一致性。在真实工厂数据上的实验表明，我们的方法能有效处理遮挡和结构复杂性，实现持续的高类别平均交并比分数。在公共数据集上的额外评估确认了我们框架的泛化能力，突显了其鲁棒性、标注效率和适应不同三维环境的能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决工业场景中3D点云分割的两大挑战：一是物体间的遮挡会削弱几何边界，二是物体尺度的巨大差异导致端到端模型无法同时捕获粗粒度和细粒度细节。这个问题在现实中非常重要，因为工业环境中的3D场景理解对于机器人操作和数字孪生构建等应用至关重要。精确识别物体实例及其结构是这些应用的核心需求，而工业环境由于其紧密连接的组件和复杂的空间布局，对3D分割尤其具有挑战性，通常导致视觉混乱和边界模糊，难以准确分离各个部件。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析现有方法的局限性来设计他们的方法。他们注意到现有的基于3D点的方法需要大量手动标注且难以泛化到大规模场景，而基于图像的方法虽然受益于基础模型如SAM和YOLO-World，但不同视图间存在语义不一致问题。作者借鉴了现有的2D检测和分割模型，但通过分层架构和多视图贝叶斯融合解决了这些局限性。他们设计了一个两阶段框架：实例级分割和部件级分割，每个阶段都采用'检测然后分割'的策略，结合了YOLO-World的类别提示和SAM的几何感知掩码生成能力。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过分层架构和多视图贝叶斯融合，从粗粒度到细粒度逐步细化分割结果，同时确保跨视图的语义一致性。整体流程分为两个阶段：1)实例级分割：渲染顶视图图像，使用YOLO-World检测物体并生成提示，SAM生成2D掩码后投影回3D点云；2)部件级分割：对每个实例渲染多视图图像，在每个视图应用相同的2D分割和后投影过程，然后通过贝叶斯更新融合确保跨视图语义一致性。系统还采用自适应渲染策略，根据物体大小和点密度计算点半径，以处理工业环境中的尺度变化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分层图像引导框架，将3D分解为实例到部件和检测到分割的阶段；2)贝叶斯更新融合机制，解决跨视图不一致问题；3)基于视觉基础模型的模块化管道，实现低成本高精度分割。相比之前的工作，这个方法不需要昂贵的3D标注，解决了图像引导方法中的语义不一致问题，能同时处理大物体和细粒度组件，特别是在多尺度条件下，且能快速适应新场景和任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种分层图像引导的3D点云分割框架，通过多视图贝叶斯融合有效解决了工业场景中遮挡和结构复杂性导致的分割挑战，实现了高精度、低成本的3D场景理解。'}


### 论文摘要

Reliable 3D segmentation is critical for understanding complex scenes with dense layouts and multi-scale objects, as commonly seen in industrial environments. In such scenarios, heavy occlusion weakens geometric boundaries between objects, and large differences in object scale will cause end-to-end models fail to capture both coarse and fine details accurately. Existing 3D point-based methods require costly annotations, while image-guided methods often suffer from semantic inconsistencies across views. To address these challenges, we propose a hierarchical image-guided 3D segmentation framework that progressively refines segmentation from instance-level to part-level. Instance segmentation involves rendering a top-view image and projecting SAM-generated masks prompted by YOLO-World back onto the 3D point cloud. Part-level segmentation is subsequently performed by rendering multi-view images of each instance obtained from the previous stage and applying the same 2D segmentation and back-projection process at each view, followed by Bayesian updating fusion to ensure semantic consistency across views. Experiments on real-world factory data demonstrate that our method effectively handles occlusion and structural complexity, achieving consistently high per-class mIoU scores. Additional evaluations on public dataset confirm the generalization ability of our framework, highlighting its robustness, annotation efficiency, and adaptability to diverse 3D environments.

---

## 24. EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy

**论文链接:** [http://arxiv.org/abs/2512.06684v1](http://arxiv.org/abs/2512.06684v1)

**作者:** Yumeng He, Zanwei Zhou, Yekun Zheng, Chen Liang, Yunbo Wang, Xiaokang Yang

**发布时间:** 2025-12-07

### GPT解析

### 总结

EMGauss是一个创新的3D重建框架，基于高斯溅射技术，将切片到3D重建问题重新定义为3D动态场景渲染问题，解决了体积电子显微镜中各向异性结构的重建挑战。

### 背景

体积电子显微镜(vEM)可实现生物结构的纳米级3D成像，但受限于获取权衡导致的各向异性和有限的轴向分辨率。现有深度学习方法依赖横向先验恢复各向同性，但对形态各向异性结构效果不佳。

### 目的

提出EMGauss框架，绕过基于各向同性方法的局限性，为体积电子显微镜提供有效的3D重建解决方案。

### 方法

将切片到3D重建重新构建为基于高斯溅射的3D动态场景渲染问题，将轴向切片进展建模为2D高斯点云的时间演变；引入Teacher-Student自举机制，使用未观察切片上的高置信度预测作为伪监督信号增强数据稀疏区域的保真度。

### 主要发现

与基于扩散和GAN的重建方法相比，EMGauss显著提高了插值质量，实现了连续切片合成，且无需大规模预训练。

### 结论

EMGauss不仅适用于体积电子显微镜，还可能为不同成像领域提供可推广的切片到3D重建解决方案。

### 翻译

体积电子显微镜(vEM)能够实现生物结构的纳米级3D成像，但仍受限于获取权衡导致的各向异性体积和有限的轴向分辨率。现有的深度学习方法试图利用横向先验来恢复各向同性，但它们的假设对于形态各向异性的结构不成立。我们提出了EMGauss，一个从平面扫描的2D切片进行3D重建的通用框架，应用于vEM，它绕过了基于各向同性方法的固有局限性。我们的关键创新是将切片到3D重建重新构建为基于高斯溅射的3D动态场景渲染问题，其中轴向切片的进展被建模为2D高斯点云的时间演变。为了增强数据稀疏区域的保真度，我们整合了一个Teacher-Student自举机制，使用未观察切片上的高置信度预测作为伪监督信号。与基于扩散和GAN的重建方法相比，EMGauss显著提高了插值质量，实现了连续切片合成，并消除了对大规模预训练的需求。除了vEM之外，它还可能为不同成像领域提供通用的切片到3D解决方案。


### 论文摘要

Volume electron microscopy (vEM) enables nanoscale 3D imaging of biological structures but remains constrained by acquisition trade-offs, leading to anisotropic volumes with limited axial resolution. Existing deep learning methods seek to restore isotropy by leveraging lateral priors, yet their assumptions break down for morphologically anisotropic structures. We present EMGauss, a general framework for 3D reconstruction from planar scanned 2D slices with applications in vEM, which circumvents the inherent limitations of isotropy-based approaches. Our key innovation is to reframe slice-to-3D reconstruction as a 3D dynamic scene rendering problem based on Gaussian splatting, where the progression of axial slices is modeled as the temporal evolution of 2D Gaussian point clouds. To enhance fidelity in data-sparse regimes, we incorporate a Teacher-Student bootstrapping mechanism that uses high-confidence predictions on unobserved slices as pseudo-supervisory signals. Compared with diffusion- and GAN-based reconstruction methods, EMGauss substantially improves interpolation quality, enables continuous slice synthesis, and eliminates the need for large-scale pretraining. Beyond vEM, it potentially provides a generalizable slice-to-3D solution across diverse imaging domains.

---

## 25. Vision-Guided Grasp Planning for Prosthetic Hands in Unstructured Environments

**论文链接:** [http://arxiv.org/abs/2512.06517v1](http://arxiv.org/abs/2512.06517v1)

**作者:** Shifa Sulaiman, Akash Bachhar, Ming Shen, Simon Bøgh

**发布时间:** 2025-12-06

### GPT解析

### 总结

该论文提出了一种用于假肢手的视觉引导抓取算法，整合了感知、规划和控制功能，实现了在动态环境中的灵巧操作能力。

### 背景

假肢技术的最新进展越来越注重通过智能控制系统提高灵活性和自主性，基于视觉的方法为假肢手在动态环境中与各种物体进行更自然交互提供了有希望的结果。

### 目的

开发一种视觉引导抓取算法，使假肢手能够更自然地与不同物体交互，提高假肢的灵活性和自主性。

### 方法

通过安装在设备上的摄像头捕捉场景，使用基于包围体层次结构的视觉算法分割目标物体并定义边界框；采用快速探索随机树星算法生成候选轨迹并计算抓取接触点；基于轨迹与物体点云间的最小欧氏距离确定指尖末端姿态；每个手指抓取姿态独立确定；使用阻尼最小二乘逆运动学求解器计算关节角度并传输给执行器；这种模块化流程支持逐指抓取规划和实时适应性。

### 主要发现

提出的算法在模拟环境中得到验证，并在Linker Hand O7平台上成功进行了实验集成，证明了其在非结构化环境中的实用性和有效性。

### 结论

该模块化管道能够支持逐指抓取规划，并在非结构化环境中提供实时适应性，为假肢技术的发展提供了新的方向。

### 翻译

假肢技术的最新进展越来越注重通过智能控制系统提高灵活性和自主性。基于视觉的方法为假肢手在动态环境中与各种物体进行更自然交互提供了有希望的结果。在此基础上，本文提出了一种用于假肢手的视觉引导抓取算法，整合了感知、规划和控制功能，实现灵巧操作。安装在设备上的摄像头捕捉场景，采用基于包围体层次结构的视觉算法分割要抓取的物体并定义其边界框。通过使用快速探索随机树星算法生成候选轨迹来计算抓取接触点，并基于这些轨迹与物体点云之间的最小欧氏距离选择指尖末端姿态。每个手指抓取姿态独立确定，实现自适应、特定物体的配置。使用基于阻尼最小二乘的逆运动学求解器计算相应的关节角度，随后传输给手指执行器执行。这种模块化流程支持逐指抓取规划，并在非结构化环境中支持实时适应性。所提出的方法在模拟中得到了验证，并在Linker Hand O7平台上进行了实验集成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决假肢手在非结构化环境中通过视觉引导进行有效抓取规划的问题。这个问题很重要，因为它能帮助假肢用户更自然、自主地处理日常物体，提高生活质量和独立性。当前假肢技术面临噪声点云处理、实时嵌入式计算和将几何感知转换为可行手指运动等挑战，解决这些问题将使假肢手能够更好地适应复杂、动态的日常环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，如简化感知牺牲几何保真度、将手视为单一实体忽略单指约束等。他们设计了一个模块化的面向单指抓取流水线，紧密耦合感知、规划和控制。具体设计包括使用BVH-AABB表示进行物体分割，RRT*算法生成候选轨迹，以及DLS方法解决逆运动学问题。作者借鉴了多项现有工作，如Morales的视觉引导流水线、Saxena的立体视觉抓取系统等，但解决了它们的局限性，如处理简单几何形状的限制、缺乏闭环控制等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是模块化、面向单指的抓取规划，每个手指独立计算末端姿态，使用BVH-AABB表示进行紧密3D物体分割，建立从感知到执行的端到端视觉驱动流水线。整体流程分为：1)感知阶段 - RGB-D传感器获取点云，预处理后使用BVH算法分割物体；2)规划阶段 - 用RRT*算法生成候选轨迹，选择最小化轨迹到表面距离的指尖姿态；3)控制阶段 - 通过DLS逆运动学求解器计算关节角度并传输给执行器；4)安全机制 - 失败时触发重新规划，支持未来触觉传感集成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)面向单指、轨迹感知的抓取规划，实现自适应物体特定抓取；2)BVH-AABB感知用于紧密3D分割，提高接触定位精度；3)端到端视觉到执行流水线，提供低延迟路径；4)模块化可扩展架构，在真实假肢硬件上验证。相比之前工作，该方法提供了更好的几何保真度，提高了对不规则形状的适应性，通过在线轨迹生成提供更大灵活性，并在实际硬件而非仅仿真中验证，还提供了完整的闭环控制流水线。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于视觉引导的模块化抓取规划方法，通过单指轨迹感知和BVH-AABB感知，使假肢手能够在非结构化环境中实现高精度、自适应的物体抓取。'}


### 论文摘要

Recent advancements in prosthetic technology have increasingly focused on enhancing dexterity and autonomy through intelligent control systems. Vision-based approaches offer promising results for enabling prosthetic hands to interact more naturally with diverse objects in dynamic environments. Building on this foundation, the paper presents a vision-guided grasping algorithm for a prosthetic hand, integrating perception, planning, and control for dexterous manipulation. A camera mounted on the set up captures the scene, and a Bounding Volume Hierarchy (BVH)-based vision algorithm is employed to segment an object for grasping and define its bounding box. Grasp contact points are then computed by generating candidate trajectories using Rapidly-exploring Random Tree Star algorithm, and selecting fingertip end poses based on the minimum Euclidean distance between these trajectories and the objects point cloud. Each finger grasp pose is determined independently, enabling adaptive, object-specific configurations. Damped Least Square (DLS) based Inverse kinematics solver is used to compute the corresponding joint angles, which are subsequently transmitted to the finger actuators for execution. This modular pipeline enables per-finger grasp planning and supports real-time adaptability in unstructured environments. The proposed method is validated in simulation, and experimental integration on a Linker Hand O7 platform.

---

## 26. Representation Learning for Point Cloud Understanding

**论文链接:** [http://arxiv.org/abs/2512.06058v1](http://arxiv.org/abs/2512.06058v1)

**作者:** Siming Yan

**发布时间:** 2025-12-05

**备注:** 181 pages

### GPT解析

### 总结

该论文研究3D数据获取与利用技术，聚焦点云基元分割的监督表示学习、自监督学习方法和2D到3D的迁移学习三个领域。

### 背景

随着技术快速发展，3D数据获取和利用在计算机视觉、机器人技术和地理空间分析等领域日益普及。3D数据通过3D扫描仪、LiDAR和RGB-D相机获取，提供丰富的几何、形状和尺度信息，与2D图像结合可为机器提供全面的环境理解。

### 目的

研究点云基元分割的监督表示学习、自监督学习方法和从2D到3D的迁移学习三个主要领域，提高机器对3D环境的理解能力。

### 方法

整合预训练的2D模型来支持3D网络训练，不简单地将2D数据转换，而是通过有效整合2D知识显著提高3D理解能力。

### 主要发现

广泛的实验验证了所提出方法的有效性，这些方法通过有效整合2D知识，有望推进点云表示学习的发展。

### 结论

所提出的方法在点云表示学习领域具有潜力，通过整合2D知识有效提升了机器对3D环境的理解能力。

### 翻译

随着技术的快速进步，3D数据获取和利用已在计算机视觉、机器人技术和地理空间分析等多个领域变得越来越普遍。通过3D扫描仪、LiDAR和RGB-D相机等方法获取的3D数据，提供了丰富的几何、形状和尺度信息。当与2D图像结合时，3D数据为机器提供了对其环境的全面理解，有益于自动驾驶、机器人、遥感和医疗等应用。本论文聚焦于三个主要领域：点云基元分割的监督表示学习、自监督学习方法以及从2D到3D的迁移学习。我们的方法整合了预训练的2D模型来支持3D网络训练，显著提高了3D理解能力，而不只是简单转换2D数据。大量实验验证了我们方法的有效性，展示了它们通过有效整合2D知识来推进点云表示学习的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何有效学习和理解3D点云数据的问题。这个问题非常重要，因为随着3D数据采集技术的普及，点云已成为计算机视觉、机器人、地理空间分析和自动驾驶等领域的关键数据形式。点云包含丰富的几何、形状和尺度信息，能够帮助机器全面理解环境。然而，点云的无序性、稀疏性和不规则性给处理带来了挑战，有效的表示学习方法对于点云分类、分割和目标检测等任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从三个方向设计了解决方案：1) 对于监督表示学习，作者注意到传统几何启发式方法（如RANSAC）与深度学习方法各有优势，因此提出HPNet模型结合两者；2) 对于自监督学习，作者识别出点云自编码器面临的采样变化问题，设计了非对称点云自编码器；3) 对于迁移学习，作者利用多视图表示作为2D和3D之间的桥梁。作者确实借鉴了现有工作，如SPFN、ParseNet、MAE等，但进行了创新性改进，特别是在特征表示、模型架构和训练策略方面。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '论文提出了三个主要方法：1) HPNet的核心思想是结合传统几何启发式与深度学习方法，使用混合点描述符，流程包括密集描述符模块计算点特征、光谱嵌入模块转换几何关系为点描述符、聚类模块组合特征并输出分割结果；2) 隐式自编码器的核心思想是使用隐函数作为解码器输出减少采样变化影响，流程包括编码器处理点云、解码器使用隐函数表示3D形状、训练重建底层3D形状；3) 多视图表示迁移学习的核心思想是利用多视图作为2D和3D桥梁，流程包括特征编码网络、点云特征体积投影、2D知识转移模块和多视图一致性模块。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '论文的关键创新点包括：1) HPNet的混合点描述符和自适应权重组合；2) 隐式自编码器解决采样变化问题，学习与采样无关的表示；3) 掩码3D特征预测专注于恢复高阶特征而非点位置；4) 多视图表示迁移学习有效整合2D知识提升3D理解。相比之前工作，这些创新在特征表示方式上更全面，模型架构设计更合理，训练策略更有效，知识迁移方式更直接，显著提升了点云表示学习的效果和应用能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过结合传统几何方法与深度学习、设计新型自监督学习架构和有效的2D到3D知识迁移策略，显著提升了点云表示学习的效果和应用能力。'}


### 论文摘要

With the rapid advancement of technology, 3D data acquisition and utilization have become increasingly prevalent across various fields, including computer vision, robotics, and geospatial analysis. 3D data, captured through methods such as 3D scanners, LiDARs, and RGB-D cameras, provides rich geometric, shape, and scale information. When combined with 2D images, 3D data offers machines a comprehensive understanding of their environment, benefiting applications like autonomous driving, robotics, remote sensing, and medical treatment. This dissertation focuses on three main areas: supervised representation learning for point cloud primitive segmentation, self-supervised learning methods, and transfer learning from 2D to 3D. Our approach, which integrates pre-trained 2D models to support 3D network training, significantly improves 3D understanding without merely transforming 2D data. Extensive experiments validate the effectiveness of our methods, showcasing their potential to advance point cloud representation learning by effectively integrating 2D knowledge.

---

## 27. When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.07684v1](http://arxiv.org/abs/2512.07684v1)

**作者:** Zihan Chen, Lanyu Yu

**发布时间:** 2025-12-08

**备注:** 10 pages

### GPT解析

### 总结

该研究提出了一个基于图神经网络(GNN)的框架，用于检测数字社区中的三种不文明行为(毒性、攻击性和人身攻击)，结合文本内容和评论关系结构，性能优于现有大型语言模型且推理成本更低。

### 背景

在线不文明行为已成为数字社区中普遍且持续存在的问题，给用户带来重大社会和心理负担。现有平台的审核和自动检测方法在准确性和效率方面往往表现有限。

### 目的

解决现有在线不文明行为检测方法的局限性，开发一种更准确、更高效的检测框架。

### 方法

提出图神经网络(GNN)框架，将用户评论表示为节点，评论间的文本相似性定义边，使网络能同时从语言内容和评论关系结构中学习。引入动态调整的注意力机制，在信息聚合过程中自适应平衡节点和拓扑特征。

### 主要发现

提出的架构在多个指标上优于12个最先进的大型语言模型，同时需要显著更低的推理成本。结构上下文在检测在线不文明行为中起关键作用，纯文本LLM范式在行为预测方面存在局限性。

### 结论

结构上下文对检测在线不文明行为至关重要，研究团队将公开所有数据集和比较输出，以支持进一步研究和可重复性。

### 翻译

在线不文明行为已成为数字社区中广泛且持续存在的问题，给用户带来了重大的社会和心理负担。尽管许多平台试图通过审核和自动检测来遏制不文明行为，但现有方法的准确性和效率往往有限。为应对这一挑战，我们提出了一种图神经网络(GNN)框架，用于检测英语维基百科社区中的三种不文明行为(即毒性、攻击性和人身攻击)。我们的模型将每个用户评论表示为节点，评论之间的文本相似性定义边，使网络能够同时从语言内容和评论之间的关系结构中学习。我们还引入了一种动态调整的注意力机制，在信息聚合过程中自适应地平衡节点和拓扑特征。实证评估表明，我们提出的架构在多个指标上优于12个最先进的大型语言模型(LLMs)，同时需要显著更低的推理成本。这些发现突显了结构上下文在检测在线不文明行为中的关键作用，并解决了纯文本LLM范式在行为预测中的局限性。所有数据集和比较输出将在我们的存储库中公开，以支持进一步研究和可重复性。


### 论文摘要

Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility.

---

## 28. 论文ID: 2512.07450v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.07450v1.json'

---

## 29. Scalable Formal Verification of Incremental Stability in Large-Scale Systems Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.07448v1](http://arxiv.org/abs/2512.07448v1)

**作者:** Ahan Basu, Mahathi Anand, Pushpak Jagtap

**发布时间:** 2025-12-08

### GPT解析

### 总结

该研究提出了一种基于图神经网络的分布式框架，用于验证具有未知动态和已知互连结构的大规模系统的增量稳定性。

### 背景

大规模系统的稳定性验证是一个重要挑战，特别是当系统动态未知但互连结构已知时。

### 目的

开发一种能够以数据驱动方式验证大规模系统增量稳定性的新方法。

### 方法

构建子系统的局部增量李雅普诺夫函数，通过图神经网络合成这些函数，并将其组合为互连系统的全局李雅普诺夫函数，最后利用训练好的神经网络的Lipschitz边界获得形式正确性保证。

### 主要发现

所提出的方法能够有效地验证大规模非线性系统的增量稳定性，并通过两个非线性案例研究得到了验证。

### 结论

基于图神经网络的分布式框架为具有未知动态和已知互连结构的大规模系统增量稳定性验证提供了一种有效方法。

### 翻译

这项工作提出了一种新颖的分布式框架，利用图神经网络来验证具有未知动态和已知互连结构的大规模系统的增量稳定性。我们提出的方法依赖于为子系统构建局部增量李雅普诺夫函数，然后将它们组合起来以获得互连系统的适当李雅普诺夫函数。图神经网络以数据驱动方式合成这些函数。然后，通过利用训练好的神经网络的Lipschitz边界获得形式正确性保证。最后，通过两个非线性案例研究验证了我们方法的有效性。


### 论文摘要

This work proposes a novel distributed framework for verifying the incremental stability of large-scale systems with unknown dynamics and known interconnection structures using graph neural networks. Our proposed approach relies on the construction of local incremental Lyapunov functions for subsystems, which are then composed together to obtain a suitable Lyapunov function for the interconnected system. Graph neural networks are used to synthesize these functions in a data-driven fashion. The formal correctness guarantee is then obtained by leveraging Lipschitz bounds of the trained neural networks. Finally, the effectiveness of our approach is validated through two nonlinear case studies.

---

## 30. Mitigating Bias in Graph Hyperdimensional Computing

**论文链接:** [http://arxiv.org/abs/2512.07433v1](http://arxiv.org/abs/2512.07433v1)

**作者:** Yezi Liu, William Youngwoo Chung, Yang Ni, Hanning Chen, Mohsen Imani

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究探讨了图超维计算（Graph HDC）中的公平性问题，提出了一种名为FairGHDC的公平感知训练框架，有效减轻了数据表示和决策规则中的偏见，同时保持了计算效率和准确性。

### 背景

图超维计算（HDC）是一种有前途的认知任务范式，它使用称为超向量的高维向量模拟类脑计算。尽管HDC在图结构数据上表现出鲁棒性和效率，但其公平性影响在很大程度上尚未被探索。

### 目的

研究图HDC中的公平性问题，其中数据表示和决策规则中的偏见可能导致对不同群体的不平等待遇，并提出一个公平性感知训练框架FairGHDC来减轻这些偏见。

### 方法

FairGHDC引入了一个基于差距的人口公平正则化器导出的偏差校正项，并将其转换为标量公平因子，用于缩放真实标签类别超向量的更新。这使得可以直接在超向量空间中进行去偏，而无需修改图编码器或需要反向传播。

### 主要发现

超向量编码和基于相似性的分类可能传播甚至放大偏见。实验结果表明，FairGHDC显著减少了人口公平和公平机会差距，同时保持了与标准GNN和公平感知GNN相当的准确性，并在GPU上实现了约10倍于基线的训练速度提升。

### 结论

FairGHDC框架有效地解决了图HDC中的公平性问题，同时保持了其计算效率和准确性优势，是一种有前途的公平图学习方法。

### 翻译

图超维计算（HDC）已成为认知任务中有前景的范式，它使用称为超向量的高维向量模拟类脑计算。尽管HDC在图结构数据上提供了鲁棒性和效率，但其公平性影响在很大程度上仍未被探索。在本文中，我们研究图HDC中的公平性问题，其中数据表示和决策规则中的偏见可能导致对不同群体的不平等待遇。我们展示了超向量编码和基于相似性的分类如何传播甚至放大这些偏见，并提出了一种公平性感知的训练框架FairGHDC来减轻它们。FairGHDC引入了一个基于差距的人口公平正则化器导出的偏差校正项，并将其转换为标量公平因子，用于缩放真实标签类别超向量的更新。这使得可以直接在超向量空间中进行去偏，而无需修改图编码器或需要反向传播。在六个基准数据集上的实验结果表明，FairGHDC显著减少了人口公平和公平机会差距，同时保持了与标准GNN和公平感知GNN相当的准确性。同时，FairGHDC保留了HDC的计算优势，在GPU上实现了比GNN和公平感知GNN基线高约一个数量级（约10倍）的训练速度提升。


### 论文摘要

Graph hyperdimensional computing (HDC) has emerged as a promising paradigm for cognitive tasks, emulating brain-like computation with high-dimensional vectors known as hypervectors. While HDC offers robustness and efficiency on graph-structured data, its fairness implications remain largely unexplored. In this paper, we study fairness in graph HDC, where biases in data representation and decision rules can lead to unequal treatment of different groups. We show how hypervector encoding and similarity-based classification can propagate or even amplify such biases, and we propose a fairness-aware training framework, FairGHDC, to mitigate them. FairGHDC introduces a bias correction term, derived from a gap-based demographic-parity regularizer, and converts it into a scalar fairness factor that scales the update of the class hypervector for the ground-truth label. This enables debiasing directly in the hypervector space without modifying the graph encoder or requiring backpropagation. Experimental results on six benchmark datasets demonstrate that FairGHDC substantially reduces demographic-parity and equal-opportunity gaps while maintaining accuracy comparable to standard GNNs and fairness-aware GNNs. At the same time, FairGHDC preserves the computational advantages of HDC, achieving up to about one order of magnitude ($\approx 10\times$) speedup in training time on GPU compared to GNN and fairness-aware GNN baselines.

---

## 31. E-PCN: Jet Tagging with Explainable Particle Chebyshev Networks Using Kinematic Features

**论文链接:** [http://arxiv.org/abs/2512.07420v1](http://arxiv.org/abs/2512.07420v1)

**作者:** Md Raqibul Islam, Adrita Khan, Mir Sazzat Hossain, Choudhury Ben Yamin Siddiqui, Md. Zakir Hossan, Tanjib Khan, M. Arshad Momen, Amin Ahsan Ali, AKM Mahbubur Rahman

**发布时间:** 2025-12-08

**备注:** 25 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种名为可解释粒子切比雪夫网络（E-PCN）的新型图神经网络，用于高能对撞机实验中的喷注分类任务。该方法通过整合多种运动学变量并利用梯度加权类激活映射技术，实现了高性能且可解释的喷注分类。

### 背景

在高能对撞机实验中，识别和分类准直粒子喷注对于数据解释至关重要。虽然深度学习技术已提高喷注分类能力，但通常缺乏可解释性，限制了物理学家对分类结果的理解。

### 目的

开发一种既具有高性能又具备物理可解释性的喷注分类方法，使研究人员能够理解哪些特征对分类决策起主导作用。

### 方法

E-PCN是一种扩展自粒子切比雪夫网络（PCN）的图神经网络。它为每个喷注构建四个图表示，分别由角分离、横向动量、动量分数和不变质量平方四种运动学变量加权。通过梯度加权类激活映射（Grad-CAM）技术，确定哪些变量对分类结果影响最大。

### 主要发现

分析表明，角分离和横向动量共同贡献了约76%的分类决策（分别为40.72%和35.67%），而动量分数和不变质量贡献了剩余的24%。在JetClass数据集上，E-PCN实现了94.67%的宏准确率、96.78%的宏AUC和86.79%的宏AUPR，显著优于基线PCN模型。

### 结论

E-PCN不仅提高了喷注分类的性能，还提供了物理可解释的特征学习能力，使研究人员能够理解分类决策背后的物理意义，代表了高能物理数据分析的重要进步。

### 翻译

高能对撞机实验中准直粒子喷注的识别和分类对于解释数据至关重要。虽然深度学习已提高喷注分类能力，但通常缺乏可解释性。我们提出了可解释粒子切比雪夫网络（E-PCN），一种扩展自粒子切比雪夫网络（PCN）的图神经网络。E-PCN通过为每个喷注构建四个图表示来整合运动学变量，每个图由不同变量加权：角分离、横向动量、动量分数和不变质量平方。我们使用梯度加权类激活映射概念来确定哪些运动学变量主导分类结果。分析显示，角分离和横向动量共同贡献约76%的分类决策（分别为40.72%和35.67%），动量分数和不变质量贡献剩余的24%。在包含10个信号类的JetClass数据集上评估，E-PCN实现了94.67%的宏准确率、96.78%的宏AUC和86.79%的宏AUPR，分别比基线PCN实现提高2.36%、4.13%和24.88%，同时展示了可物理解释的特征学习能力。


### 论文摘要

The identification and classification of collimated particle sprays, or jets, are essential for interpreting data from high-energy collider experiments. While deep learning has improved jet classification, it often lacks interpretability. We introduce the Explainable Particle Chebyshev Network (E-PCN), a graph neural network extending the Particle Chebyshev Network (PCN). E-PCN integrates kinematic variables into jet classification by constructing four graph representations per jet, each weighted by a distinct variable: angular separation ($Δ$), transverse momentum ($k_T$), momentum fraction ($z$), and invariant mass squared ($m^2$). We use the concept of Gradient-weighted Class Activation Mapping (Grad-CAM) to determine which kinematic variables dominate classification outcomes. Analysis reveals that angular separation and transverse momentum collectively account for approximately 76% of classification decisions (40.72% and 35.67%, respectively), with momentum fraction and invariant mass contributing the remaining 24%. Evaluated on the JetClass dataset with 10 signal classes, E-PCN achieves a macro-accuracy of 94.67%, macro-AUC of 96.78%, and macro-AUPR of 86.79%, representing improvements of 2.36%, 4.13%, and 24.88% respectively over the baseline PCN implementation, while demonstrating physically interpretable feature learning.

---

## 32. Recent advancements in the tau reconstruction and identification techniques in CMS

**论文链接:** [http://arxiv.org/abs/2512.07387v1](http://arxiv.org/abs/2512.07387v1)

**作者:** Andrea Cardini

**发布时间:** 2025-12-08

**备注:** Plenary talk presented at the 18th International Workshop on Tau Lepton Physics (TAU2025)

### GPT解析

### 总结

该论文介绍了CMS实验中Tau轻子强子衰变重建和识别的最新进展，包括在线和离线层面的改进，以及这些改进对物理研究的潜在影响。

### 背景

Tau轻子在希格斯玻色子研究和超越标准模型物理研究中扮演关键角色，特别是在当前大型强子对撞机及其高亮度升级阶段。

### 目的

展示CMS实验中Tau轻子强子衰变重建和识别技术的最新发展，提高对真实Tau衰变与其他粒子喷注的区分能力。

### 方法

采用基于深度卷积神经网络和领域适应的Tau识别算法，在高 level trigger使用简化版本；探索基于图神经网络和粒子变换器的替代方法；开发使用图神经网络重建和识别位移Tau轻子的专门技术。

### 主要发现

新算法显著提高了真实强子Tau衰变与误识别的夸克和胶子喷注、电子和μ子的区分能力；展示了两种算法使用Run 3早期数据的性能和校准结果。

### 结论

许多涉及Tau轻子的CMS物理分析将从这些改进中受益，有助于提高希格斯玻色子研究和超越标准模型物理研究的精确度。

### 翻译

Tau轻子在当前大型强子对撞机及其高亮度升级中的希格斯玻色子研究和超越标准模型物理研究中起着至关重要的作用。本次报告介绍了CMS实验中Tau轻子强子衰变重建和识别的最新进展，包括在线和离线层面。为早期Run 3数据收集期部署的Tau识别算法基于具有领域适应的深度卷积神经网络，显著提高了真实强子Tau衰变与误识别的夸克和胶子喷注、电子和μ子的区分能力。在实时数据收集中，使用算法的简化版本在高 level trigger选择包含Tau轻子的事例。报告展示了使用Run 3早期数据的两种算法的性能和校准情况。许多涉及Tau轻子的CMS物理分析预计将从这些改进中受益。还介绍了基于喷注味别并结合图神经网络和粒子变换器的识别强子Tau的替代方法。此外，讨论了使用图神经网络重建和识别来自长寿命粒子衰变的位移Tau轻子的专门技术。


### 论文摘要

Tau leptons play a crucial role in studies of the Higgs boson and searches for Beyond the Standard Model physics at the present LHC and in its high luminosity upgrade. This talk presents the latest advancements in the reconstruction and identification of hadronic decays of tau leptons at the CMS experiment, both at the online and offline levels. The tau identification algorithm deployed for the early Run 3 data-taking period, based on a deep convolutional neural network with domain adaptation, showcases significantly improved discrimination of genuine hadronic tau decays against mis-identified quark and gluon jets, electrons, and muons. During live data-taking, a simplified version of the algorithm is used to select events with tau leptons at the High Level Trigger (HLT). The performance and calibration of both algorithms using early Run 3 data are presented. Many CMS physics analyses involving tau leptons are expected to benefit from these improvements. Alternative approaches to identify hadronic taus combined with jet flavour, based on graph neural networks and particle transformers, are also covered. Additionally, the dedicated techniques used to reconstruct and identify displaced tau leptons originating from long-lived particle decays using graph neural networks are discussed.

---

## 33. On the Impact of Graph Neural Networks in Recommender Systems: A Topological Perspective

**论文链接:** [http://arxiv.org/abs/2512.07384v1](http://arxiv.org/abs/2512.07384v1)

**作者:** Daniele Malitesta, Claudio Pomo, Vito Walter Anelli, Alberto Carlo Maria Mancino, Alejandro Bellogín, Tommaso Di Noia

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了一种以拓扑为中心的视角来理解基于图神经网络(GNN)的推荐系统，强调用户-物品图的结构特性与GNN架构设计的交互对模型性能的关键影响。

### 背景

在推荐系统中，用户-物品交互可建模为二分图，这促使了图神经网络的快速应用。尽管GNN在实践中表现优于传统协同过滤方法，但其系统优势的原因尚未被完全理解。

### 目的

建立一种拓扑为中心的视角来理解基于GNN的推荐系统，强调对模型性能的理解应考虑用户-物品图的结构特性及其与GNN架构的交互。

### 方法

引入正式分类法提炼十一种代表性GNN推荐方法的共同模式；定义并重新解释推荐数据集的十三个经典和拓扑特性；分析GNN架构如何编码这些特性；构建连接数据集特性与模型行为的解释框架。

### 主要发现

基于GNN的推荐系统性能与其底层拓扑结构密切相关；不同GNN架构以不同方式编码数据集的拓扑特性；数据集的拓扑特性可解释和预测模型行为。

### 结论

通过拓扑基础重新框架了基于GNN的推荐；概述了下一代拓扑感知推荐系统在理论、数据中心和评估方面的开放性挑战。

### 翻译

在推荐系统中，用户-物品交互可以建模为二分图，其中用户节点和物品节点通过无向边连接。这种基于图的视角推动了图神经网络(GNNs)的快速采用，它们通常优于协同过滤(CF)方法，如潜在因子模型、深度神经网络和生成策略。然而，尽管它们在实证上取得了成功，GNN相比其他CF方法具有系统优势的原因仍只有部分被理解。这本专著推进了基于GNN推荐的以拓扑为中心的视角。我们认为，对这些模型性能的全面理解应考虑用户-物品图的结构特性及其与GNN架构设计的交互。为支持这一观点，我们引入了一个正式的分类法，提炼了十一种代表性基于GNN的推荐方法的共同建模模式，并将它们整合为一个统一的概念流程。我们进一步正式化了推荐数据集的十三个经典和拓扑特性，并通过图机器学习的视角重新解释它们。利用这些定义，我们分析了所考虑的基于GNN的推荐架构，以评估它们如何以及在何种程度上编码这些特性。基于这一分析，我们推导出一个解释性框架，将可测量的数据集特性与模型行为和性能联系起来。总而言之，这本专著通过其拓扑基础重新框架了基于GNN的推荐，并概述了下一代拓扑感知推荐系统在理论、数据中心和评估方面的开放性挑战。


### 论文摘要

In recommender systems, user-item interactions can be modeled as a bipartite graph, where user and item nodes are connected by undirected edges. This graph-based view has motivated the rapid adoption of graph neural networks (GNNs), which often outperform collaborative filtering (CF) methods such as latent factor models, deep neural networks, and generative strategies. Yet, despite their empirical success, the reasons why GNNs offer systematic advantages over other CF approaches remain only partially understood. This monograph advances a topology-centered perspective on GNN-based recommendation. We argue that a comprehensive understanding of these models' performance should consider the structural properties of user-item graphs and their interaction with GNN architectural design. To support this view, we introduce a formal taxonomy that distills common modeling patterns across eleven representative GNN-based recommendation approaches and consolidates them into a unified conceptual pipeline. We further formalize thirteen classical and topological characteristics of recommendation datasets and reinterpret them through the lens of graph machine learning. Using these definitions, we analyze the considered GNN-based recommender architectures to assess how and to what extent they encode such properties. Building on this analysis, we derive an explanatory framework that links measurable dataset characteristics to model behavior and performance. Taken together, this monograph re-frames GNN-based recommendation through its topological underpinnings and outlines open theoretical, data-centric, and evaluation challenges for the next generation of topology-aware recommender systems.

---

## 34. Edge-Aware Graph Attention Model for Structural Optimization of High Entropy Carbides

**论文链接:** [http://arxiv.org/abs/2512.07358v1](http://arxiv.org/abs/2512.07358v1)

**作者:** Neethu Mohan Mangalassery, Abhishek Kumar Singh

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了一种边缘感知图注意力模型，这是一种物理信息图神经网络，用于高效预测高熵体系的弛豫原子结构，具有轻量级架构和低计算开销的特点。

### 背景

预测化学复杂材料的弛豫原子结构，特别是高熵体系，仍然是一个主要的计算挑战。传统的第一性原理方法对于高熵系统来说变得异常昂贵。

### 目的

开发一种高效、准确的方法来预测高熵体系的弛豫原子结构，替代传统但计算成本高昂的第一性原理方法。

### 方法

作者提出了边缘感知图注意力模型，采用化学和几何信息描述符，通过多头自注意力机制利用节点和边特征自适应地加权相邻原子，学习不依赖于全局方向或位置的复杂化学和结构关系。

### 主要发现

作者在涵盖二元到高熵碳化物成分的系统数据集上训练和评估了该模型，证明了其准确性、收敛效率和可转移性。

### 结论

该模型提供了一个快速、可扩展且经济高效的第一性原理替代方案，具有刚性变换不变性和领域感知的注意力机制，能够加速熵稳定材料的发现和筛选。

### 翻译

预测化学复杂材料的弛豫原子结构仍然是一个主要的计算挑战，特别是对于传统第一性原理方法变得异常昂贵的高熵系统。我们引入了边缘感知图注意力模型，这是一种专为预测高熵体系弛豫原子结构而设计的物理信息图神经网络。边缘感知图注意力模型采用化学和几何信息描述符，这些描述符捕捉了原子性质和局部结构环境。为了有效捕获原子相互作用，我们的模型集成了多头自注意力机制，该机制使用节点和边特征自适应地加权相邻原子。这种边缘感知注意力框架学习复杂的化学和结构关系，不依赖于全局方向或位置。我们在碳化物系统数据集上训练和评估了边缘感知GAT模型，该数据集涵盖了从二元到高熵碳化物成分，并证明了其准确性、收敛效率和可转移性。该架构轻量级，计算开销非常小，使其非常适合大规模材料筛选。通过提供刚性变换不变性并利用领域感知的注意力机制，我们的模型为DFT提供了一个快速、可扩展且经济高效的替代方案，从而加速了熵稳定材料的发现和筛选。


### 论文摘要

Predicting relaxed atomic structures of chemically complex materials remains a major computational challenge, particularly for high-entropy systems where traditional first-principles methods become prohibitively expensive. We introduce the edge-aware graph attention model, a physics-informed graph neural network tailored for predicting relaxed atomic structures of high-entropy systems. the edge-aware graph attention model employs chemically and geometrically informed descriptors that capture both atomic properties and local structural environments. To effectively capture atomic interactions, our model integrates a multi-head self-attention mechanism that adaptively weighs neighbouring atoms using both node and edge features. This edge-aware attention framework learn complex chemical and structural relationships independent of global orientation or position. We trained and evaluated the edge-aware GAT model on a dataset of carbide systems, spanning binary to high-entropy carbide compositions, and demonstrated its accuracy, convergence efficiency, and transferability. The architecture is lightweight, with a very low computational footprint, making it highly suitable for large-scale materials screening. By providing invariance to rigid-body transformations and leveraging domain-informed attention mechanisms, our model delivers a fast, scalable, and cost-effective alternative to DFT, enabling accelerated discovery and screening of entropy-stabilised materials.

---

## 35. Towards a Relationship-Aware Transformer for Tabular Data

**论文链接:** [http://arxiv.org/abs/2512.07310v1](http://arxiv.org/abs/2512.07310v1)

**作者:** Andrei V. Konstantinov, Valerii A. Zuev, Lev V. Utkin

**发布时间:** 2025-12-08

### GPT解析

### 总结

这篇论文提出了一种基于改进注意力机制的解决方案，用于在表格数据的深度学习模型中考虑样本间的依赖关系。

### 背景

表格数据的深度学习模型通常不允许对样本之间施加外部依赖关系图，而图神经网络只考虑相邻节点，难以应用于稀疏图。

### 目的

提出解决方案，通过改进的注意力机制考虑数据点之间可能的关系，特别是在处理效应估计等任务中。

### 方法

提出基于改进注意力机制的几种解决方案，通过在注意力矩阵中添加项来考虑数据点之间可能的关系。

### 主要发现

在合成和真实数据集上的回归任务中，以及IHDP数据集上的处理效应估计任务中，将提出的模型相互之间以及与梯度提升决策树进行了比较。

### 结论

论文提出的解决方案能够有效处理表格数据中的样本间依赖关系，在多种任务中表现良好。

### 翻译

表格数据的深度学习模型通常不允许对样本之间施加外部依赖关系图，这对于处理效应估计等任务中考虑相关性很有用。图神经网络只考虑相邻节点，使它们难以应用于稀疏图。本文提出了几种基于改进注意力机制的解决方案，通过在注意力矩阵中添加项来考虑数据点之间可能的关系。我们的模型在合成和真实数据集上的回归任务中相互之间以及与梯度提升决策树进行了比较，同时在IHDP数据集上的处理效应估计任务中也进行了比较。


### 论文摘要

Deep learning models for tabular data typically do not allow for imposing a graph of external dependencies between samples, which can be useful for accounting for relatedness in tasks such as treatment effect estimation. Graph neural networks only consider adjacent nodes, making them difficult to apply to sparse graphs. This paper proposes several solutions based on a modified attention mechanism, which accounts for possible relationships between data points by adding a term to the attention matrix. Our models are compared with each other and the gradient boosting decision trees in a regression task on synthetic and real-world datasets, as well as in a treatment effect estimation task on the IHDP dataset.

---

## 36. Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection

**论文链接:** [http://arxiv.org/abs/2512.07246v1](http://arxiv.org/abs/2512.07246v1)

**作者:** Mengqi Wang, Jianwei Wang, Qing Liu, Xiwei Xu, Zhenchang Xing, Liming Zhu, Wenjie Zhang

**发布时间:** 2025-12-08

**备注:** 14 pages, 8 figures

### GPT解析

### 总结

本文提出了一种基于大型语言模型的表格数据错误检测方法，通过构建决策树集成框架解决了现有方法可解释性差和鲁棒性不足的问题。

### 背景

错误检测(ED)对于确保表格数据质量至关重要。当前最先进的ED方法利用大型语言模型(LLM)的知识和能力来标记单元格是否错误，但这些方法存在可解释性和鲁棒性方面的局限。

### 目的

解决现有LLM作为标签器的方法缺乏可解释性和鲁棒性的问题，提出一种新的LLM作为诱导器的框架，用于构建可解释且鲁棒的错误检测系统。

### 方法

提出TreeED和ForestED两种方法：TreeED基于提示查询LLM诱导决策树骨架，包含规则节点(简单验证)、GNN节点(复杂模式)和叶节点(最终决策)；ForestED使用不确定性采样获取多个行子集，为每个子集构建决策树，并通过期望最大化算法优化共识预测。

### 主要发现

通过大量实验证明，所提出的方法准确、可解释且鲁棒，比最佳基线平均F1分数提高了16.1%。

### 结论

LLM-as-an-inducer框架能有效解决现有错误检测方法的可解释性和鲁棒性问题，在表格数据错误检测任务中取得了显著性能提升。

### 翻译

错误检测(ED)旨在识别表格数据中不正确或不一致的单元格值，对于确保数据质量很重要。最近最先进的ED方法利用预训练知识和大型语言模型(LLM)中嵌入的语义能力来直接标记单元格是否错误。然而，这种LLM作为标签器的管道(1)依赖于黑盒、隐式决策过程，因此无法提供检测结果的可解释性，并且(2)对提示高度敏感，由于模型的内在随机性导致输出不一致，因此缺乏鲁棒性。为解决这些局限性，我们提出了一种LLM作为诱导器的框架，采用LLM诱导用于ED的决策树(称为TreeED)，并进一步集成多个此类树进行共识检测(称为ForestED)，从而提高可解释性和鲁棒性。具体来说，基于从数据上下文、决策树规范和输出要求中获取的提示，TreeED查询LLM以诱导决策树骨架，其根到叶的决策路径指定了评估给定样本的逐步过程。每个树包含三种类型的节点：(1)执行简单验证检查(如格式或范围)的规则节点，(2)捕获复杂模式(如函数依赖关系)的图神经网络(GNN)节点，以及(3)输出最终决策类型(错误或干净)的叶节点。此外，ForestED使用基于不确定性的采样获取多个行子集，使用TreeED为每个子集构建决策树。然后，它利用基于期望最大化-最大化的算法联合估计树可靠性并优化共识ED预测。大量实验证明我们的方法准确、可解释且鲁棒，比最佳基线平均F1分数提高了16.1%。


### 论文摘要

Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability embedded in large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline (1) relies on the black box, implicit decision process, thus failing to provide explainability for the detection results, and (2) is highly sensitive to prompts, yielding inconsistent outputs due to inherent model stochasticity, therefore lacking robustness. To address these limitations, we propose an LLM-as-an-inducer framework that adopts LLM to induce the decision tree for ED (termed TreeED) and further ensembles multiple such trees for consensus detection (termed ForestED), thereby improving explainability and robustness. Specifically, based on prompts derived from data context, decision tree specifications and output requirements, TreeED queries the LLM to induce the decision tree skeleton, whose root-to-leaf decision paths specify the stepwise procedure for evaluating a given sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks (e.g., format or range), (2) Graph Neural Network (GNN) nodes that capture complex patterns (e.g., functional dependencies), and (3) leaf nodes that output the final decision types (error or clean). Furthermore, ForestED employs uncertainty-based sampling to obtain multiple row subsets, constructing a decision tree for each subset using TreeED. It then leverages an Expectation-Maximization-based algorithm that jointly estimates tree reliability and optimizes the consensus ED prediction. Extensive xperiments demonstrate that our methods are accurate, explainable and robust, achieving an average F1-score improvement of 16.1% over the best baseline.

---

## 37. Benchmarking Deep Neural Networks for Modern Recommendation Systems

**论文链接:** [http://arxiv.org/abs/2512.07000v1](http://arxiv.org/abs/2512.07000v1)

**作者:** Abderaouf Bahi, Ibtissem Gasmi

**发布时间:** 2025-12-07

### GPT解析

### 总结

本研究评估了七种不同神经网络架构在三个数据集上的推荐系统表现

### 背景

研究在零售电商、亚马逊产品和Netflix奖三种数据集上部署了CNN、RNN、GNN、自编码器、Transformer、NCF和Siamese Networks七种神经网络架构

### 目的

评估这些神经网络架构在推荐系统中的有效性，并通过混合方法提升推荐系统的性能

### 方法

使用准确率、召回率、F1分数和推荐多样性等指标评估各模型表现

### 主要发现

GNN擅长处理电商环境中的复杂项目关系；RNN能有效捕捉Netflix等平台的时间动态；Siamese Networks对推荐多样化有贡献，特别是在零售环境中

### 结论

建议采用混合方法结合各种模型优势，以更好地满足用户偏好并适应数字平台不断变化的需求

### 翻译

本文研究了在零售电商、亚马逊产品和Netflix奖三种不同数据集上部署CNN、RNN、GNN、自编码器、Transformer、NCF和Siamese Networks七种不同的神经网络架构。通过准确率、召回率、F1分数和推荐多样性等指标评估其有效性。结果表明，GNN特别擅长管理电商环境中的复杂项目关系，而RNN在捕捉Netflix等平台必需的时间动态方面有效。Siamese Networks因其对推荐多样化的贡献而受到重视，特别是在零售环境中。尽管有这些好处，但仍存在计算需求高、依赖大量数据以及平衡准确性和多样性推荐等挑战。研究通过建议结合各种模型优势的混合方法，为推荐系统的发展提供参考，以更好地满足用户偏好并适应当代数字平台不断变化的需求。


### 论文摘要

This paper examines the deployment of seven different neural network architectures CNN, RNN, GNN, Autoencoder, Transformer, NCF, and Siamese Networks on three distinct datasets: Retail E-commerce, Amazon Products, and Netflix Prize. It evaluates their effectiveness through metrics such as accuracy, recall, F1-score, and diversity in recommendations. The results demonstrate that GNNs are particularly adept at managing complex item relationships in e-commerce environments, whereas RNNs are effective in capturing the temporal dynamics that are essential for platforms such as Netflix.. Siamese Networks are emphasized for their contribution to the diversification of recommendations, particularly in retail settings. Despite their benefits, issues like computational demands, reliance on extensive data, and the challenge of balancing accurate and diverse recommendations are addressed. The study seeks to inform the advancement of recommendation systems by suggesting hybrid methods that merge the strengths of various models to better satisfy user preferences and accommodate the evolving demands of contemporary digital platforms.

---

## 38. Can We Go Beyond Visual Features? Neural Tissue Relation Modeling for Relational Graph Analysis in Non-Melanoma Skin Histology

**论文链接:** [http://arxiv.org/abs/2512.06949v1](http://arxiv.org/abs/2512.06949v1)

**作者:** Shravan Venkatraman, Muthu Subash Kavitha, Joe Dhanith P R, V Manikandarajan, Jia Wu

**发布时间:** 2025-12-07

**备注:** 19 pages, 5 figures, 2 tables

### GPT解析

### 总结

本文提出了一种名为神经组织关系建模(NTRM)的新型分割框架，通过结合CNN和图神经网络来建模组织间的空间和功能关系，显著提升了皮肤癌诊断中组织病理学图像分割的准确性。

### 背景

组织病理学图像分割对于皮肤癌诊断中组织结构的描绘至关重要，但建模空间上下文和组织间关系仍然是一个挑战，特别是在有重叠或形态相似组织的区域。当前基于CNN的方法主要基于视觉纹理操作，通常将组织视为独立区域，未能编码生物上下文。

### 目的

开发一种能够建模组织和空间功能关系的新型分割框架，以解决当前方法在处理边界密集区域和相似组织时的局限性。

### 方法

提出神经组织关系建模(NTRM)框架，该框架通过组织级图神经网络增强CNN，在预测区域上构建图，通过消息传递传播上下文信息，并通过空间投影细化分割。NTRM明确编码组织间依赖性，能够在边界密集区域实现结构连贯的预测。

### 主要发现

在基准组织病理学非黑色素瘤皮肤癌分割数据集上，NTRM优于最先进的方法，实现了比评估方法中表现最好的模型高4.9%至31.25%的Dice相似系数。实验表明，关系建模为更上下文感知和可解释的组织学分割提供了原则性路径。

### 结论

与缺乏组织级结构意识的局部感受野架构相比，关系建模为更上下文感知和可解释的组织学分割提供了原则性路径，代码已公开在GitHub上。

### 翻译

组织病理学图像分割对于界定皮肤癌诊断中的组织结构至关重要，但建模空间上下文和组织间关系仍然是一个挑战，特别是在有重叠或形态相似组织的区域。当前基于卷积神经网络(CNN)的方法主要基于视觉纹理操作，通常将组织视为独立区域，未能编码生物上下文。为此，我们引入神经组织关系建模(NTRM)，一种新颖的分割框架，通过组织级图神经网络增强CNN，以跨组织类型建模空间和功能关系。NTRM在预测区域上构建图，通过消息传递传播上下文信息，并通过空间投影细化分割。与先前方法不同，NTRM明确编码组织间依赖性，能够在边界密集区域实现结构连贯的预测。在基准组织病理学非黑色素瘤皮肤癌分割数据集上，NTRM优于最先进的方法，实现了比评估方法中表现最好的模型高4.9%至31.25%的稳健Dice相似系数。我们的实验表明，与缺乏组织级结构意识的局部感受野架构相比，关系建模为更上下文感知和可解释的组织学分割提供了原则性路径。我们的代码可在https://github.com/shravan-18/NTRM获取。


### 论文摘要

Histopathology image segmentation is essential for delineating tissue structures in skin cancer diagnostics, but modeling spatial context and inter-tissue relationships remains a challenge, especially in regions with overlapping or morphologically similar tissues. Current convolutional neural network (CNN)-based approaches operate primarily on visual texture, often treating tissues as independent regions and failing to encode biological context. To this end, we introduce Neural Tissue Relation Modeling (NTRM), a novel segmentation framework that augments CNNs with a tissue-level graph neural network to model spatial and functional relationships across tissue types. NTRM constructs a graph over predicted regions, propagates contextual information via message passing, and refines segmentation through spatial projection. Unlike prior methods, NTRM explicitly encodes inter-tissue dependencies, enabling structurally coherent predictions in boundary-dense zones. On the benchmark Histopathology Non-Melanoma Skin Cancer Segmentation Dataset, NTRM outperforms state-of-the-art methods, achieving a robust Dice similarity coefficient that is 4.9\% to 31.25\% higher than the best-performing models among the evaluated approaches. Our experiments indicate that relational modeling offers a principled path toward more context-aware and interpretable histological segmentation, compared to local receptive-field architectures that lack tissue-level structural awareness. Our code is available at https://github.com/shravan-18/NTRM.

---

## 39. ExPUFFIN: Thermodynamic Consistent Viscosity Prediction in an Extended Path-Unifying Feed-Forward Interfaced Network

**论文链接:** [http://arxiv.org/abs/2512.06927v1](http://arxiv.org/abs/2512.06927v1)

**作者:** Carine Menezes Rebello, Ulderico Di Caprio, Jenny Steen-Hansen, Bruno Rodrigues, Erbet Almeida Costa, Anderson Rapello dos Santos, Flora Esposito, Mumin Enis Leblebici, Idelfonso B. R. Nogueira

**发布时间:** 2025-12-07

### GPT解析

### 总结

研究提出了ExPUFFIN，一个扩展版的Path-unifying Feed-Forward Interfaced Network，这是一个混合图神经网络框架，可以直接从分子图预测纯烃的温度依赖性粘度，并通过在输出层引入机制归纳偏置确保热力学一致性。

### 背景

准确预测液体粘度对过程设计和模拟至关重要，但对新型分子仍具挑战性。传统基团贡献模型难以处理异构体区分、大分子和参数可用性问题，而纯数据驱动的图神经网络需要大量数据集且可解释性有限，且缺乏热力学一致性。

### 目的

开发一种能够准确预测纯烃温度依赖性粘度的方法，确保预测具有热力学一致性，并提高模型的准确性、鲁棒性和可转移性。

### 方法

提出ExPUFFIN混合图神经网络框架，直接从分子图预测纯烃温度依赖性粘度；分子信息以图结构形式给出，编码为图卷积网络，并映射到基于两个热物理相关性的归纳偏置神经元：三参数Andrade型方程和四参数经验粘度-温度关系；将这些模型的准确性与纯数据驱动的预测进行比较。

### 主要发现

基于Andrade的ExPUFFIN变体将RMSE比纯数据驱动基线降低37%，产生平滑、物理一致的粘度-温度曲线内插和外推；经验ExPUFFIN模型提供可比的准确性同时保持稳健趋势；在GNN输出中嵌入基于物理的结构可提高准确性、鲁棒性和可转移性。

### 结论

该方法能够可靠预测复杂烃分子的粘度，且易于扩展到其他特性和更广泛的化学领域；在GNN输出中嵌入基于物理的结构可显著提升模型性能。

### 翻译

准确预测液体粘度对过程设计和模拟至关重要，但对新型分子而言仍然具有挑战性。传统的基团贡献模型难以处理异构体区分、大分子和参数可用性问题，而纯数据驱动的图神经网络需要大量数据集且可解释性有限。即使可行应用，纯数据驱动模型在预测中缺乏热力学一致性，并非可靠解决方案。本研究引入了ExPUFFIN，即Path-unifying Feed-Forward Interfaced Network的扩展版本，它是一个混合图神经网络框架，可以直接从分子图预测纯烃的温度依赖性粘度，同时在输出层强制执行机制归纳偏置以确保热力学一致性。分子信息以图结构形式给出，编码为图卷积网络，并映射到基于两个热物理相关性的归纳偏置神经元：一个三参数Andrade型方程和一个四参数经验粘度-温度关系。将这些模型的准确性与纯数据驱动的预测进行了比较。基于Andrade的ExPUFFIN变体将RMSE比纯数据驱动基线降低了37%，并产生平滑、物理一致的粘度-温度曲线内插和外推，这些特性在纯数据驱动模型中并未观察到。经验ExPUFFIN模型提供可比的准确性同时保持稳健趋势。总体而言，在GNN输出中嵌入基于物理的结构提高了准确性、鲁棒性和可转移性，能够可靠预测复杂烃分子的粘度。该方法易于扩展到其他特性和更广泛的化学领域。


### 论文摘要

Accurate prediction of liquid viscosity is essential for process design and simulation, yet remains challenging for novel molecules.Conventional group-contribution models struggle with isomer discrimination, large molecules, and parameter availability, while purely data-driven graph neural networks (GNNs) demand large datasets and offer limited interpretability. Even when feasible to be applied, purely data-driven models lack thermodynamic consistency in their predictions and are not a reliable solution.This work introduces ExPUFFIN, an extended version of the Path-unifying Feed-Forward Interfaced Network, consisting of a hybrid GNN-based framework that directly predicts temperature-dependent viscosities of pure hydrocarbons from molecular graphs, while enforcing mechanistic inductive biases in the output layer to ensure thermodynamic consistency. Molecular information is given as graph structures, encoded as a graph convolutional network, and mapped to an inductive bias neuron based on two thermophysical correlations:a three-parameter Andrade-type equation and a four-parameter empirical viscosity-temperature relation. The accuracy of these models is compared with a solely data-driven prediction. The Andrade-based ExPUFFIN variant reduces RMSE compared to the purely data-driven baseline of 37 percent and yields smooth, physically consistent interpolation and extrapolation of viscosity-temperature curves, properties that are not observed in purely data-driven models. The empirical ExPUFFIN model provides comparable accuracy while retaining robust trends. Overall, embedding physics-based structure in GNN outputs improves accuracy, robustness, and transferability, enabling reliable viscosity predictions for complex hydrocarbon molecules. The approach is readily extendable to other properties and significantly broader chemical domains.

---

## 40. Measuring Over-smoothing beyond Dirichlet energy

**论文链接:** [http://arxiv.org/abs/2512.06782v1](http://arxiv.org/abs/2512.06782v1)

**作者:** Weiqi Guan, Zihao Shi

**发布时间:** 2025-12-07

**备注:** 17 pages, 1 figure

### GPT解析

### 总结

本文提出了一种基于高阶特征导数能量的节点相似度度量家族，解决了传统Dirichlet能量仅能捕捉一阶特征导数的局限性，并通过理论分析和实验验证了其在图神经网络过平滑问题研究中的应用价值。

### 背景

Dirichlet能量是量化过平滑问题的常用指标，但它仅限于捕捉一阶特征导数，存在固有局限性。

### 目的

提出一种基于高阶特征导数能量的节点相似度度量的一般化家族，以克服传统Dirichlet能量的局限性。

### 方法

通过严格的理论分析建立了这些度量之间的关系，确定了在连续热扩散和离散聚合算子下Dirichlet能量的衰减率。

### 主要发现

过平滑衰减率与图拉普拉斯谱间隙之间存在内在联系；基于注意力的图神经网络在所提出的度量标准下表现出过平滑问题。

### 结论

基于高阶特征导数的度量提供了更全面的过平滑分析框架，并揭示了基于注意力的图神经网络存在的问题。

### 翻译

虽然Dirichlet能量作为量化过平滑的普遍指标，但它本质上仅限于捕捉一阶特征导数。为了解决这一局限性，我们提出了一种基于高阶特征导数能量的节点相似度度量的一般化家族。通过这些度量之间关系的严格理论分析，我们建立了在连续热扩散和离散聚合算子下Dirichlet能量的衰减率。此外，我们的分析揭示了过平滑衰减率与图拉普拉斯谱间隙之间的内在联系。最后，实证结果表明，在所提出的度量标准下评估时，基于注意力的图神经网络(GNNs)会受到过平滑问题的影响。


### 论文摘要

While Dirichlet energy serves as a prevalent metric for quantifying over-smoothing, it is inherently restricted to capturing first-order feature derivatives. To address this limitation, we propose a generalized family of node similarity measures based on the energy of higher-order feature derivatives. Through a rigorous theoretical analysis of the relationships among these measures, we establish the decay rates of Dirichlet energy under both continuous heat diffusion and discrete aggregation operators. Furthermore, our analysis reveals an intrinsic connection between the over-smoothing decay rate and the spectral gap of the graph Laplacian. Finally, empirical results demonstrate that attention-based Graph Neural Networks (GNNs) suffer from over-smoothing when evaluated under these proposed metrics.

---

## 41. Crystallographic Texture-Generalizable Orientation-Aware Interaction-Based Deep Material Network for Polycrystal Modeling and Texture Evolution

**论文链接:** [http://arxiv.org/abs/2512.06779v1](http://arxiv.org/abs/2512.06779v1)

**作者:** Ting-Ju Wei, Tung-Huan Su, Chuin-Shan Chen

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文提出了一种改进的机器学习框架TACS-GNN-ODMN，解决了原有ODMN模型需要针对不同晶体学纹理重新训练的限制，显著提高了模型的泛化能力，同时保持物理可解释性和预测准确性。

### 背景

机器学习通过构建高效的替代模型显著推进了材料建模。取向感知的基于交互的深度材料网络(ODMN)是一种能够从线性弹性刚度数据学习多晶微观结构内在几何-力学关系并预测非线性力学响应和纹理演化的框架，但需要针对每种不同的晶体学纹理重新训练，限制了其适用性。

### 目的

解决ODMN需要针对每种不同晶体学纹理重新训练的限制，提高模型的泛化能力，使其能够准确预测不同纹理下的材料行为。

### 方法

引入TACS-GNN-ODMN框架，集成两个关键组件：(i)纹理自适应聚类和采样(TACS)方案，用于初始化纹理相关参数；(ii)图神经网络(GNN)，用于预测应力平衡相关参数。

### 主要发现

TACS-GNN-ODMN能够准确预测不同纹理下的非线性响应和纹理演化，与直接数值模拟(DNS)结果高度一致。通过消除纹理特定重新训练的需求，同时保留了物理可解释性，显著增强了模型的泛化能力。

### 结论

TACS-GNN-ODMN大幅提升了ODMN的泛化能力，为多尺度模拟和下一代材料设计提供了强大而高效的替代模型。

### 翻译

机器学习通过构建能够实现高计算效率而不牺牲预测精度的替代模型，显著推进了材料建模。取向感知的基于交互的深度材料网络(ODMN)是这样一个框架，其中一组材料节点代表晶体学纹理，分层交互网络基于希尔-曼德尔条件强制这些节点之间的应力平衡。仅使用线弹性刚度数据，ODMN就能学习多晶微观结构内的内在几何-力学关系，使其能够高保真地预测非线性力学响应和纹理演化。然而，由于需要针对每种不同的晶体学纹理重新训练，其适用性仍然受到限制。为解决这一限制，我们引入了TACS-GNN-ODMN框架，该框架集成了(i)用于初始化纹理相关参数的纹理自适应聚类和采样(TACS)方案，以及(ii)用于预测应力平衡相关参数的图神经网络(GNN)。所提出的框架能够准确预测不同纹理下的非线性响应和纹理演化，与直接数值模拟(DNS)结果高度一致。通过消除纹理特定重新训练的需求，同时保留物理可解释性，TACS-GNN-ODMN显著增强了ODMN的泛化能力，为多尺度模拟和下一代材料设计提供了强大而高效的替代模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多晶体材料建模中现有方法的局限性，特别是原始ODMN模型需要针对每种不同的晶体织构重新训练的问题。这个问题在材料科学和工程领域非常重要，因为多晶体材料在工程应用中非常普遍，织构演化对材料的力学性能有重要影响，而传统计算方法（如直接数值模拟DNS）计算成本高，难以用于大规模工程设计和优化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了原始ODMN的局限性（需要针对每种织构重新训练），然后思考如何提高模型的泛化能力。他们设计了两个关键组件：TACS（Texture-Adaptive Clustering and Sampling）方案用于初始化织构相关参数，和GNN（Graph Neural Network）用于预测应力平衡相关参数。作者借鉴了多项现有工作，包括ODMN框架作为基础、TACS方法用于织构表示、图神经网络处理微观结构拓扑关系，以及Hill-Mandel条件和平均理论作为物理基础。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合TACS方案和GNN，创建一个能够泛化到不同织构分布的多晶体建模框架，同时保持物理可解释性和计算效率。整体流程分为两个阶段：离线训练阶段使用TACS采样代表性取向，构建微观结构图，通过GNN学习应力平衡参数，联合训练整个框架；在线预测阶段对未见过的RVE构建微观结构图，使用训练好的GNN预测参数，初始化织构参数，构建特定ODMN模型，最后快速预测力学响应和织构演化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) TACS-GNN-ODMN框架首次将TACS和GNN与ODMN结合，实现不同织构分布的泛化；2) 保持物理可解释性，区别于纯数据驱动的深度学习方法；3) 高效性，相比DNS加速比超过200倍；4) 准确性，预测精度与DNS高度一致。相比原始ODMN，主要不同在于不需要针对每种织构重新训练，能处理未见过的织构分布，通过GNN更好捕捉晶粒间相互作用，同时保持物理可解释性和计算效率的平衡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合纹理自适应聚类采样和图神经网络的深度材料网络框架，实现了对不同织构分布多晶体材料的高效、准确且物理可解释的力学响应和织构演化预测，无需针对每种织构重新训练。'}


### 论文摘要

Machine learning has significantly advanced materials modeling by enabling surrogate models that achieve high computational efficiency without compromising predictive accuracy. The Orientation-aware Interaction-based Deep Material Network (ODMN) is one such framework, in which a set of material nodes represents crystallographic textures, and a hierarchical interaction network enforces stress equilibrium among these nodes based on the Hill-Mandel condition. Using only linear elastic stiffness data, ODMN learns the intrinsic geometry-mechanics relationships within polycrystalline microstructures, allowing it to predict nonlinear mechanical responses and texture evolution with high fidelity. However, its applicability remains limited by the need to retrain for each distinct crystallographic texture. To address this limitation, we introduce the TACS-GNN-ODMN framework, which integrates (i) a Texture-Adaptive Clustering and Sampling (TACS) scheme for initializing texture-related parameters and (ii) a Graph Neural Network (GNN) for predicting stress-equilibrium-related parameters. The proposed framework accurately predicts nonlinear responses and texture evolution across diverse textures, showing close agreement with direct numerical simulations (DNS). By eliminating the requirement for texture-specific retraining while preserving physical interpretability, TACS-GNN-ODMN substantially enhances the generalization capability of ODMN, offering a robust and efficient surrogate model for multiscale simulations and next-generation materials design.

---

## 42. Multi-Scale Protein Structure Modelling with Geometric Graph U-Nets

**论文链接:** [http://arxiv.org/abs/2512.06752v1](http://arxiv.org/abs/2512.06752v1)

**作者:** Chang Liu, Vivian Li, Linus Leong, Vladimir Radenkovic, Pietro Liò, Chaitanya K. Joshi

**发布时间:** 2025-12-07

**备注:** Presented at Machine Learning in Structural Biology, 2025. Open-source code: https://github.com/VirtualProteins/GNN_UNet

### GPT解析

### 总结

本文提出了一种名为几何图U-网络的新型模型，通过递归地粗化和细化蛋白质图来学习多尺度表示，解决了传统几何图神经网络无法捕获蛋白质层次相互作用的问题。

### 背景

几何图神经网络和Transformer已成为学习3D蛋白质结构的先进技术，但它们依赖消息传递机制，无法捕获控制蛋白质功能的层次相互作用，如全局结构域和长距离变构调节。

### 目的

作者认为网络架构本身应该反映这种生物层次结构，因此提出了一种能够捕获蛋白质层次相互作用的新型模型。

### 方法

引入了几何图U-网络，这是一种新型模型，通过递归地粗化和细化蛋白质图来学习多尺度表示，理论上比标准几何GNN更具表现力。

### 主要发现

在蛋白质折叠分类任务上，几何U-网络显著优于不变和等变基线模型，展示了学习定义蛋白质折叠的全局结构模式的能力。

### 结论

该工作为设计能够学习生物分子多尺度结构的几何深度学习架构提供了理论基础。

### 翻译

几何图神经网络和Transformer已成为学习3D蛋白质结构的先进技术。然而，它们对消息传递的依赖使它们无法捕获控制蛋白质功能的层次相互作用，如全局结构域和长距离变构调节。在这项工作中，我们认为网络架构本身应该反映这种生物层次结构。我们引入了几何图U-网络，这是一种新型模型，通过递归地粗化和细化蛋白质图来学习多尺度表示。我们从理论上证明这种层次设计比标准几何GNN更具表现力。经验上，在蛋白质折叠分类任务中，几何U-网络显著优于不变和等变基线模型，展示了它们学习定义蛋白质折叠的全局结构模式的能力。我们的工作为设计能够学习生物分子多尺度结构的几何深度学习架构提供了理论基础。


### 论文摘要

Geometric Graph Neural Networks (GNNs) and Transformers have become state-of-the-art for learning from 3D protein structures. However, their reliance on message passing prevents them from capturing the hierarchical interactions that govern protein function, such as global domains and long-range allosteric regulation. In this work, we argue that the network architecture itself should mirror this biological hierarchy. We introduce Geometric Graph U-Nets, a new class of models that learn multi-scale representations by recursively coarsening and refining the protein graph. We prove that this hierarchical design can theoretically more expressive than standard Geometric GNNs. Empirically, on the task of protein fold classification, Geometric U-Nets substantially outperform invariant and equivariant baselines, demonstrating their ability to learn the global structural patterns that define protein folds. Our work provides a principled foundation for designing geometric deep learning architectures that can learn the multi-scale structure of biomolecules.

---

## 43. Learning Thermoelectric Transport from Crystal Structures via Multiscale Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2512.06697v1](http://arxiv.org/abs/2512.06697v1)

**作者:** Yuxuan Zeng, Wei Cao, Yijing Zuo, Fang Lyu, Wenhao Xie, Tan Peng, Yue Hou, Ling Miao, Ziyu Wang, Jing Shi

**发布时间:** 2025-12-07

### GPT解析

### 总结

研究提出了一种专门用于估计无机热电晶体电子输运系数的图神经网络模型，通过多尺度编码晶体结构和物理化学性质，实现了最先进的性能并具有出色的外推能力。

### 背景

图神经网络擅长从图结构数据中提取潜在模式，特别适用于晶体表征学习，而热电晶体的电子输运性质是材料科学中的重要研究方向。

### 目的

开发一个专门的GNN模型来准确估计无机热电晶体的电子输运系数，并探索其物理机制。

### 方法

设计了一种多尺度编码模型，涵盖全局、原子、键和角级别；将GNN与从头算计算结合；从全局和原子角度进行可解释性分析，追踪输运行为的起源。

### 主要发现

模型在基准数据集上实现了最先进性能并具有显著外推能力；成功识别出具有优异电子输运特性的化合物；模型的决策过程揭示了潜在的物理模式。

### 结论

该研究为计算机辅助材料设计提供了新见解，展示了GNN在材料发现和性质预测中的潜力。

### 翻译

图神经网络（GNNs）被设计用于从图结构数据中提取潜在模式，使其特别适用于晶体表征学习。在此，我们提出了一个专门用于估计无机热电晶体电子输运系数的GNN模型。该模型以多尺度方式编码晶体结构和物理化学性质，包括全局、原子、键和角级别。它在基准数据集上实现了最先进的性能，并具有显著的外推能力。通过将我们提出的GNN与从头算计算相结合，我们成功识别出具有优异电子输运特性的化合物，并从全局和原子角度进一步进行了可解释性分析，追溯了它们不同输运行为的起源。有趣的是，模型的决策过程自然揭示了潜在的物理模式，为计算机辅助材料设计提供了新的见解。


### 论文摘要

Graph neural networks (GNNs) are designed to extract latent patterns from graph-structured data, making them particularly well suited for crystal representation learning. Here, we propose a GNN model tailored for estimating electronic transport coefficients in inorganic thermoelectric crystals. The model encodes crystal structures and physicochemical properties in a multiscale manner, encompassing global, atomic, bond, and angular levels. It achieves state-of-the-art performance on benchmark datasets with remarkable extrapolative capability. By combining the proposed GNN with \textit{ab initio} calculations, we successfully identify compounds exhibiting outstanding electronic transport properties and further perform interpretability analyses from both global and atomic perspectives, tracing the origins of their distinct transport behaviors. Interestingly, the decision process of the model naturally reveals underlying physical patterns, offering new insights into computer-assisted materials design.

---

## 44. Learning-based Link Prediction Methods Integrating Network Topological Features and Embedding Representations

**论文链接:** [http://arxiv.org/abs/2512.06677v1](http://arxiv.org/abs/2512.06677v1)

**作者:** Zi-Xuan Jin, Jun-Fan Yi, Ke-Ke Shang

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文提出了一种名为TELP的集成链接预测模型，结合网络拓扑特征和嵌入表示，通过多阶段架构捕获局部连接模式和全局结构，并使用多种机器学习模型的集成来提高预测性能和鲁棒性。在九个基准网络上的实验表明，TELP优于传统方法和主流图神经网络模型。

### 背景

链接预测是复杂网络拓扑分析的前沿任务，旨在基于观察到的节点和结构信息推断节点对之间潜在链接的存在。

### 目的

提出一个集成链接预测模型(TELP)，结合网络拓扑特征和嵌入表示，以克服传统启发式方法在学习节点属性和深度结构模式方面的局限性，以及基于学习方法的弱可解释性和有限泛化能力。

### 方法

TELP采用多阶段架构：通过选择同质和异质拓扑特征来捕获局部连接模式；生成Node2Vec嵌入并与拓扑特征融合，形成多维表示；部署逻辑回归、随机森林和XGBoost模型的集成来最大化预测性能和鲁棒性。

### 主要发现

在九个经典基准网络上的实验表明，TELP与传统启发式方法和主流图神经网络模型相比，实现了优越的AUC和AP性能。消融研究进一步证实，特征融合和集成策略对于最佳性能是必不可少的。

### 结论

TELP模型通过结合拓扑特征和嵌入表示，以及使用集成学习方法，有效地解决了链接预测中的挑战，并在多个基准测试中表现出色。

### 翻译

链接预测作为复杂网络拓扑分析的前沿任务，旨在基于观察到的节点和结构信息推断节点对之间潜在链接的存在。我们提出了一种集成链接预测模型TELP，该模型整合了网络拓扑特征和嵌入表示，旨在克服传统启发式方法在捕捉节点属性和深度结构模式方面的局限性，以及基于学习方法的弱可解释性和有限泛化能力。TELP利用多阶段架构。通过网络类型感知的同质和异质拓扑特征选择来捕获局部连接模式，这也提高了可解释性。为了融入全局结构，生成了Node2Vec嵌入并与这些拓扑特征融合，形成全面的多维表示。在此基础上，部署了逻辑回归、随机森林和XGBoost模型的集成，以最大化预测性能和鲁棒性。在九个经典基准网络上的实验表明，TELP与传统启发式方法和主流图神经网络模型相比，实现了优越的AUC和AP性能，而消融研究进一步证实，特征融合和集成策略对于最佳性能是必不可少的。


### 论文摘要

Link prediction, as a frontier task in complex network topology analysis, aims to infer the existence of latent links between node pairs based on observed nodes and structural information. We propose an ensemble link prediction model that integrates network topology features and embedding representations (TELP), designed to overcome the limitations of conventional heuristic methods in capturing node attributes and deep structural patterns, as well as the weak interpretability and limited generalization of learning-based approaches. TELP leverages a multi-stage architecture. Local connectivity patterns are captured through network-type-aware selection of homogeneous and heterogeneous topology features, which also promotes interpretability. To incorporate global structure, Node2Vec embeddings are generated and fused with these topology features, resulting in comprehensive multi-dimensional representations. Building on this enriched feature space, an ensemble of logistic regression, random forest, and XGBoost models is deployed to maximize predictive performance and robustness. Experiments on nine classical benchmark networks demonstrate that TELP achieves superior AUC and AP performance compared with traditional heuristic approaches and mainstream graph neural network models, while ablation studies further confirm that feature fusion and ensemble strategies are essential for optimal performance.

---

## 45. The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News

**论文链接:** [http://arxiv.org/abs/2512.06638v1](http://arxiv.org/abs/2512.06638v1)

**作者:** Isha Karn, David Jensen

**发布时间:** 2025-12-07

**备注:** Preprint. Approximately 15 pages, 5 figures, 3 tables

### GPT解析

### 总结

研究指出，用于假新闻检测的两个常用基准数据集(GossipCop和PolitiFact)存在图拓扑结构过于简单的问题，无法有效评估图神经网络(GNNs)利用传播结构的能力。

### 背景

图神经网络(GNNs)被广泛用于通过建模社交媒体上新闻文章的内容和传播结构来检测假新闻。

### 目的

评估常用基准数据集对测试GNNs利用传播结构能力的适用性，并比较GNNs与结构不可知多层感知器(MLP)的性能差异。

### 方法

系统性地将五种GNN架构与使用相同节点特征的MLP进行基准测试；进行受控实验，包括节点特征洗牌和边缘结构随机化；分析数据集的图拓扑结构特征；在合成数据集上进行对比实验。

### 主要发现

MLP的性能与GNN相匹配或仅略逊一筹，性能差距通常在1-2%以内；特征洗牌导致性能崩溃，但边缘随机化下性能保持稳定；超过75%的节点距离根节点只有一跳，结构多样性极低；在结构信息量大的合成数据集上，GNN显著优于MLP。

### 结论

广泛使用的基准数据集并不能有效地测试建模结构特征的效用，这促使开发具有更丰富、更多样图拓扑结构的数据集。

### 翻译

图神经网络(GNNs)被广泛用于通过建模社交媒体上新闻文章的内容和传播结构来检测假新闻。我们表明，两个最常用的基准数据集——GossipCop和PolitiFact——并不适合评估使用传播结构的模型的效用。具体来说，这些数据集表现出浅层的、自我中心的图拓扑结构，几乎无法区分不同的建模方法。我们系统地将五种GNN架构与使用相同节点特征的结构不可知多层感知器(MLP)进行了基准测试。我们表明MLP的性能与GNN相匹配或仅略逊一筹，性能差距通常在1-2%以内，并且置信区间重叠。为了分离这些数据集中结构的贡献，我们进行了受控实验，其中节点特征被洗牌或边缘结构被随机化。我们发现性能在特征洗牌下崩溃，但在边缘随机化下保持稳定。这表明结构在这些基准测试中起的作用微乎其微。结构分析进一步揭示，超过75%的节点距离根节点只有一跳，表现出极小的结构多样性。相比之下，在节点特征有噪声且结构信息量大的合成数据集上，GNN显著优于MLP。这些发现提供了强有力的证据，表明广泛使用的基准数据集并不能有意义地测试建模结构特征的效用，它们促使开发具有更丰富、更多样图拓扑结构的数据集。


### 论文摘要

Graph neural networks (GNNs) are widely used for the detection of fake news by modeling the content and propagation structure of news articles on social media. We show that two of the most commonly used benchmark data sets - GossipCop and PolitiFact - are poorly suited to evaluating the utility of models that use propagation structure. Specifically, these data sets exhibit shallow, ego-like graph topologies that provide little or no ability to differentiate among modeling methods. We systematically benchmark five GNN architectures against a structure-agnostic multilayer perceptron (MLP) that uses the same node features. We show that MLPs match or closely trail the performance of GNNs, with performance gaps often within 1-2% and overlapping confidence intervals. To isolate the contribution of structure in these datasets, we conduct controlled experiments where node features are shuffled or edge structures randomized. We find that performance collapses under feature shuffling but remains stable under edge randomization. This suggests that structure plays a negligible role in these benchmarks. Structural analysis further reveals that over 75% of nodes are only one hop from the root, exhibiting minimal structural diversity. In contrast, on synthetic datasets where node features are noisy and structure is informative, GNNs significantly outperform MLPs. These findings provide strong evidence that widely used benchmarks do not meaningfully test the utility of modeling structural features, and they motivate the development of datasets with richer, more diverse graph topologies.

---

## 46. Hierarchical geometric deep learning enables scalable analysis of molecular dynamics

**论文链接:** [http://arxiv.org/abs/2512.06520v1](http://arxiv.org/abs/2512.06520v1)

**作者:** Zihan Pengmei, Spencer C. Guo, Chatipat Lorpaiboon, Aaron R. Dinner

**发布时间:** 2025-12-06

**备注:** 17 pages, 12 figures

### GPT解析

### 总结

该研究通过聚合局部信息的方法，在保持原子级细节的同时显著减少内存和运行时间需求，使得能够在单GPU上在几分钟内分析包含数千个残基的蛋白质-核酸复合物的分子动力学模拟。

### 背景

分子动力学模拟可生成复杂系统的原子级详细轨迹，但缺乏良好定量描述符时分析具有挑战性。图神经网络(GNNs)虽可避免手动特征工程，但在处理数百以上残基的生物分子系统时受限，难以捕获长程相互作用细节，且大型图计算需大量内存和运行时间。

### 目的

开发一种方法，能够高效分析大型生物分子系统的动力学模拟，克服传统GNN方法在处理大型系统时的局限性。

### 方法

通过聚合局部信息来减少内存和运行时间需求，同时保持原子级细节。

### 主要发现

1. 该方法使单GPU可在几分钟内分析包含数千残基的蛋白质-核酸复合物模拟；2. 对于数百残基系统，该方法提高了分析性能和结果可解释性。

### 结论

聚合局部信息的方法在不牺牲原子级细节的情况下，显著提高了大型生物分子系统动力学分析的计算效率，为研究大型蛋白质-核酸复合物提供了新可能。

### 翻译

分子动力学模拟可以生成复杂系统的原子级详细轨迹，但当系统缺乏良好的定量描述符时，分析这些动力学可能具有挑战性。图神经网络通过在表示空间相邻原子的节点之间传递消息，有望避免手动特征工程，但在分析包含数百个以上残基的生物分子系统时受到限制，这是由于难以通过消息传递捕获长程相互作用的细节，以及与大型图相关的内存和运行时间要求。我们展示了如何聚合局部信息以减少内存和运行时间需求，同时不牺牲原子级细节。我们证明了这种方法使得能够在单GPU上在几分钟内分析包含数千个残基的蛋白质-核酸复合物的模拟。对于包含数百个残基的系统，有足够的数据进行定量比较，我们表明该方法提高了性能和可解释性。


### 论文摘要

Molecular dynamics simulations can generate atomically detailed trajectories of complex systems, but analyzing these dynamics can be challenging when systems lack well-established quantitative descriptors (features). Graph neural networks (GNNs) in which messages are passed between nodes that represent atoms that are spatial neighbors promise to obviate manual feature engineering, but the use of GNNs with biomolecular systems of more than a few hundred residues has been limited in the context of analyzing dynamics by both difficulties in capturing the details of long-range interactions with message passing and the memory and runtime requirements associated with large graphs. Here, we show how local information can be aggregated to reduce memory and runtime requirements without sacrificing atomic detail. We demonstrate that this approach opens the door to analyzing simulations of protein-nucleic acid complexes with thousands of residues on single GPUs within minutes. For systems with hundreds of residues, for which there are sufficient data to make quantitative comparisons, we show that the approach improves performance and interpretability.

---

## 47. DDFI: Diverse and Distribution-aware Missing Feature Imputation via Two-step Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.06356v1](http://arxiv.org/abs/2512.06356v1)

**作者:** Yifan Song, Fenglin Yu, Yihong Luo, Xingjian Tao, Siya Qiu, Kai Han, Jing Tang

**发布时间:** 2025-12-06

### GPT解析

### 总结

本研究提出了DDFI方法，结合特征传播和基于图的掩码自编码器，解决图神经网络中节点特征不完整的问题，通过Co-Label Linking算法和两步表示生成过程，提高了在非连通图、过平滑问题和归纳任务中的性能。

### 背景

现实场景中节点特征不完整是普遍存在的问题，例如网络用户属性部分私有，导致图神经网络性能显著下降。特征传播方法虽有良好表现，但在非完全连通图、过平滑问题和归纳任务特征分布偏移方面存在局限性。

### 目的

解决特征传播方法在处理非连通图、过平滑问题和归纳任务特征分布偏移方面的局限性，提高图神经网络在节点特征缺失情况下的性能。

### 方法

提出DDFI方法，结合特征传播和基于图的掩码自编码器；设计Co-Label Linking算法，随机连接相同标签节点增强连通图性能；开发两步表示生成过程，通过MAE重建特征减少分布偏移并增强特征多样性；收集Sailing数据集包含自然缺失特征用于评估。

### 主要发现

在六个公共数据集和Sailing数据集上的实验表明，DDFI在直推式和归纳式设置下均优于最先进方法，有效解决了特征传播方法的三大问题。

### 结论

DDFI通过创新性地结合特征传播和掩码自编码器，以及Co-Label Linking算法和两步表示生成过程，显著提高了图神经网络在处理不完整节点特征时的性能，特别是在非连通图、过平滑问题和归纳任务方面。

### 翻译

不完整的节点特征在现实场景中无处不在，例如，网络用户的属性可能部分私有，这导致图神经网络（GNNs）的性能显著下降。特征传播（FP）是一种众所周知的方法，在图的缺失节点特征插补方面表现良好，但它仍存在以下三个问题：1）它难以处理非完全连通的图；2）插补的特征面临过平滑问题；3）FP专为直推式任务设计，忽略了归纳任务中的特征分布偏移。为了应对这些挑战，我们引入了DDFI，这是一种多样化且分布感知的缺失特征插补方法，以非平凡的方式将特征传播与基于图的掩码自编码器（MAE）相结合。它首先设计了一个简单而有效的算法，即Co-Label Linking（CLL），随机连接训练集中具有相同标签的节点，以提高在具有大量连通分量的图上的性能。然后，我们在推理阶段开发了一种新颖的两步表示生成过程。具体而言，在推理过程中不直接使用FP插补的特征作为输入，DDFI通过整个MAE进一步重建特征，以减少归纳任务中的特征分布偏移并增强节点特征的多样性。同时，由于现有的图特征插补方法仅通过手动屏蔽特征来模拟缺失场景进行评估，我们收集了一个名为Sailing的新数据集，其中包含来自航行记录的自然缺失特征，以帮助更好地评估有效性。在六个公共数据集和Sailing上进行的大量实验表明，DDFI在直推式和归纳式设置下都优于最先进的方法。


### 论文摘要

Incomplete node features are ubiquitous in real-world scenarios, e.g., the attributes of web users may be partly private, which causes the performance of Graph Neural Networks (GNNs) to decline significantly. Feature propagation (FP) is a well-known method that performs well for imputation of missing node features on graphs, but it still has the following three issues: 1) it struggles with graphs that are not fully connected, 2) imputed features face the over-smoothing problem, and 3) FP is tailored for transductive tasks, overlooking the feature distribution shift in inductive tasks. To address these challenges, we introduce DDFI, a Diverse and Distribution-aware Missing Feature Imputation method that combines feature propagation with a graph-based Masked AutoEncoder (MAE) in a nontrivial manner. It first designs a simple yet effective algorithm, namely Co-Label Linking (CLL), that randomly connects nodes in the training set with the same label to enhance the performance on graphs with numerous connected components. Then we develop a novel two-step representation generation process at the inference stage. Specifically, instead of directly using FP-imputed features as input during inference, DDFI further reconstructs the features through the whole MAE to reduce feature distribution shift in the inductive tasks and enhance the diversity of node features. Meanwhile, since existing feature imputation methods for graphs only evaluate by simulating the missing scenes with manually masking the features, we collect a new dataset called Sailing from the records of voyages that contains naturally missing features to help better evaluate the effectiveness. Extensive experiments conducted on six public datasets and Sailing show that DDFI outperforms the state-of-the-art methods under both transductive and inductive settings.

---

## 48. LLM-Upgraded Graph Reinforcement Learning for Carbon-Aware Job Scheduling in Smart Manufacturing

**论文链接:** [http://arxiv.org/abs/2512.06351v1](http://arxiv.org/abs/2512.06351v1)

**作者:** Zhiying Yang, Fang Liu, Wei Zhang, Xin Lou, Malcolm Yoke Hean Low, Boon Ping Gan

**发布时间:** 2025-12-06

### GPT解析

### 总结

本文介绍了Luca，一个基于大型语言模型升级的图强化学习框架，用于碳感知柔性作业车间调度。

### 背景

智能制造系统中的动态和可持续调度面临挑战，需要一种能够同时考虑能源效率和调度及时性的方法。

### 目的

开发一种能够优化完工时间和碳排放双重目标的调度框架，以支持可持续制造目标。

### 方法

Luca框架整合了图神经网络和大型语言模型，通过精心设计的内部提示策略生成融合嵌入，然后使用深度强化学习策略网络处理这些嵌入，生成实时调度决策。框架采用双重目标奖励函数，鼓励能源效率和调度及时性。

### 主要发现

在合成数据集上，与最佳对比算法相比，Luca平均实现了4.1%的完工时间降低，最高可达12.2%，同时保持相同的排放水平。在公共数据集上，Luca在完工时间和排放方面都观察到额外收益。

### 结论

Luca在智能制造中的碳感知调度是有效且实用的。

### 翻译

本文介绍了Luca，一个基于大型语言模型升级的图强化学习框架，用于碳感知柔性作业车间调度。Luca通过整合图神经网络和大型语言模型，由精心设计的内部提示策略指导，生成能够捕捉最新调度状态结构特征和上下文语义的融合嵌入。这种表达性嵌入随后由深度强化学习策略网络处理，生成针对最小化完工时间和碳排放目标优化的实时调度决策。为支持可持续目标，Luca采用双重目标奖励函数，鼓励能源效率和调度及时性。在合成和公共数据集上的实验结果表明，Luca持续优于对比算法。例如，在合成数据集上，与最佳对比算法相比，它平均实现4.1%的完工时间降低，最高可达12.2%，同时保持相同的排放水平。在公共数据集上，观察到完工时间和排放方面的额外收益。这些结果表明，Luca在智能制造中的碳感知调度是有效且实用的。


### 论文摘要

This paper presents \textsc{Luca}, a \underline{l}arge language model (LLM)-\underline{u}pgraded graph reinforcement learning framework for \underline{c}arbon-\underline{a}ware flexible job shop scheduling. \textsc{Luca} addresses the challenges of dynamic and sustainable scheduling in smart manufacturing systems by integrating a graph neural network and an LLM, guided by a carefully designed in-house prompting strategy, to produce a fused embedding that captures both structural characteristics and contextual semantics of the latest scheduling state. This expressive embedding is then processed by a deep reinforcement learning policy network, which generates real-time scheduling decisions optimized for both makespan and carbon emission objectives. To support sustainability goals, \textsc{Luca} incorporates a dual-objective reward function that encourages both energy efficiency and scheduling timeliness. Experimental results on both synthetic and public datasets demonstrate that \textsc{Luca} consistently outperforms comparison algorithms. For instance, on the synthetic dataset, it achieves an average of 4.1\% and up to 12.2\% lower makespan compared to the best-performing comparison algorithm while maintaining the same emission level. On public datasets, additional gains are observed for both makespan and emission. These results demonstrate that \textsc{Luca} is effective and practical for carbon-aware scheduling in smart manufacturing.

---

## 49. Multimodal Graph Neural Networks for Prognostic Modeling of Brain Network Reorganization

**论文链接:** [http://arxiv.org/abs/2512.06303v1](http://arxiv.org/abs/2512.06303v1)

**作者:** Preksha Girish, Rachana Mysore, Kiran K. N., Hiranmayee R., Shipra Prashanth, Shrey Kumar

**发布时间:** 2025-12-06

**备注:** 5 pages, 2 figures. IEEE conference-style format

### GPT解析

### 总结

该研究提出了一种多模态图神经网络框架，整合结构MRI、扩散张量成像和功能MRI数据，建模大脑网络时空重组，生成可解释生物标志物，并预测个体认知下降风险。

### 背景

理解大脑网络的动态重组对于预测认知下降、神经进展和临床结果的个体差异至关重要。

### 目的

开发一种能够整合多种神经影像数据、建模大脑网络时空重组、生成可解释生物标志物并预测个体认知下降风险的方法。

### 方法

提出多模态图神经网络框架，将脑区表示为节点，结构和功能连接表示为边形成纵向脑图；使用嵌入基于图循环网络的分数随机微分算子捕捉时间演化；通过注意力机制融合多模态信息并生成可解释生物标志物；组合这些生物标志物形成复合预后指数。

### 主要发现

在纵向神经影像数据集上验证了方法的预测准确性和可解释性；数学严谨的多模态基于图方法能够从现有成像数据中推导出临床上有意义的生物标志物；无需收集新数据即可实现目标。

### 结论

多模态图神经网络框架为理解大脑网络动态重组提供了有效工具，能够生成有临床意义的生物标志物，预测认知下降风险，且不依赖于新的数据收集。

### 翻译

理解大脑网络的动态重组对于预测认知下降、神经进展和临床结果的个体差异至关重要。这项工作提出了一种多模态图神经网络框架，整合了结构MRI、扩散张量成像和功能MRI，用于建模大脑网络的时空重组。脑区被表示为节点，结构和功能连接被表示为边，为每个受试者形成纵向脑图。时间演化通过嵌入基于图循环网络的分数随机微分算子来捕捉，使能够建模网络动力学的长期依赖性和随机波动。注意力机制融合多模态信息并生成可解释的生物标志物，包括网络能量熵、图曲率、分数记忆指数和模态特异性注意力分数。这些生物标志物被组合成一个复合预后指数，以量化网络不稳定或认知下降的个体风险。在纵向神经影像数据集上的实验证明了预测准确性和可解释性。结果强调了数学严谨的多模态基于图方法从现有成像数据推导临床上有意义生物标志物的潜力，而无需收集新数据。


### 论文摘要

Understanding the dynamic reorganization of brain networks is critical for predicting cognitive decline, neurological progression, and individual variability in clinical outcomes. This work proposes a multimodal graph neural network framework that integrates structural MRI, diffusion tensor imaging, and functional MRI to model spatiotemporal brain network reorganization. Brain regions are represented as nodes and structural and functional connectivity as edges, forming longitudinal brain graphs for each subject. Temporal evolution is captured via fractional stochastic differential operators embedded within graph-based recurrent networks, enabling the modeling of long-term dependencies and stochastic fluctuations in network dynamics. Attention mechanisms fuse multimodal information and generate interpretable biomarkers, including network energy entropy, graph curvature, fractional memory indices, and modality-specific attention scores. These biomarkers are combined into a composite prognostic index to quantify individual risk of network instability or cognitive decline. Experiments on longitudinal neuroimaging datasets demonstrate both predictive accuracy and interpretability. The results highlight the potential of mathematically rigorous, multimodal graph-based approaches for deriving clinically meaningful biomarkers from existing imaging data without requiring new data collection.

---

## 50. Back to Author Console Empowering GNNs for Domain Adaptation via Denoising Target Graph

**论文链接:** [http://arxiv.org/abs/2512.06236v1](http://arxiv.org/abs/2512.06236v1)

**作者:** Haiyang Yu, Meng-Chieh Lee, Xiang song, Qi Zhu, Christos Faloutsos

**发布时间:** 2025-12-06

### GPT解析

### 总结

本文提出了一种名为GraphDeT的框架，通过在目标图上添加辅助边去噪任务来增强图神经网络在域适应场景下的节点分类性能，实验证明该方法在处理时间和区域域图偏移时优于现有基线方法。

### 背景

在图域适应中，当图数据在不同时间或从不同区域收集时，经常会出现结构域偏移，导致图神经网络在目标图上表现不佳。

### 目的

利用源图和目标图结构以及源标签来增强图神经网络在目标图上的泛化能力。

### 方法

提出GraphDeT框架，将辅助的边去噪任务整合到GNN训练中，并通过理论分析将该辅助任务与图的泛化界限联系起来，证明其可以施加约束从而提高泛化能力。

### 主要发现

简单地在目标图上加入一个用于去噪图边的辅助损失函数可以非常有效地增强GNN在目标图上的性能，这种辅助任务能够收紧泛化界限并提高模型的泛化能力。

### 结论

GraphDeT框架在处理时间和区域域图偏移方面表现出优于现有基线方法的性能，为图域适应中的节点分类任务提供了有效的解决方案。

### 翻译

我们探索了图域适应背景下的节点分类任务，该任务利用源图和目标图结构以及源标签来增强图神经网络在目标图上的泛化能力。结构域偏移经常发生，特别是当图数据在不同时间或从不同区域收集时，导致GNN在目标图上表现不佳。令人惊讶的是，我们发现简单地在目标图上添加一个用于去噪图边的辅助损失函数可以极大地增强GNN在目标图上的性能。基于这一见解，我们提出了GraphDeT框架，该框架将这个辅助边任务整合到域适应下的节点分类GNN训练中。我们的理论分析将这个辅助边任务与图的泛化界限联系起来，证明这种辅助任务可以施加约束从而收紧界限并提高泛化能力。实验结果表明，在处理时间和区域域图偏移方面，与现有基线方法相比具有优越的性能。


### 论文摘要

We explore the node classification task in the context of graph domain adaptation, which uses both source and target graph structures along with source labels to enhance the generalization capabilities of Graph Neural Networks (GNNs) on target graphs. Structure domain shifts frequently occur, especially when graph data are collected at different times or from varying areas, resulting in poor performance of GNNs on target graphs. Surprisingly, we find that simply incorporating an auxiliary loss function for denoising graph edges on target graphs can be extremely effective in enhancing GNN performance on target graphs. Based on this insight, we propose our framework, GraphDeT, a framework that integrates this auxiliary edge task into GNN training for node classification under domain adaptation. Our theoretical analysis connects this auxiliary edge task to the graph generalization bound with -distance, demonstrating such auxiliary task can imposes a constraint which tightens the bound and thereby improves generalization. The experimental results demonstrate superior performance compared to the existing baselines in handling both time and regional domain graph shifts.

---

## 51. Enhancing Urban Sensing Utility with Sensor-enabled Vehicles and Easily Accessible Data

**论文链接:** [http://arxiv.org/abs/2512.07124v1](http://arxiv.org/abs/2512.07124v1)

**作者:** Hui Zhong, Qing-Long Lu, Qiming Zhang, Hongliang Lu, Xinhu Zheng

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究提出了一种自适应框架，用于增强配备传感器的车辆的感知效用，通过整合异构开源数据和时空权重优化来提高城市数据收集效率。

### 背景

城市感知对智慧城市发展至关重要，现代车辆技术使车辆从单纯交通工具转变为有价值的数据收集传感器。车辆感知技术因其灵活性、成本效益和时空覆盖范围广而被认为是具有前景的技术，但优化感知策略以平衡时空覆盖、减少冗余和应对预算限制仍是一大挑战。

### 目的

开发一个自适应框架，增强配备传感器的车辆的感知效用，通过整合异构开源数据，利用时空权重优化车辆选择和感知覆盖，以提高城市数据收集效率。

### 方法

开发了一种基于熵的车辆选择策略（Improved OptiFleet），旨在最大化感知效用同时最小化冗余。使用中国广州320辆配备传感器的车辆两个月内的真实空气质量数据验证框架性能。

### 主要发现

所提出的方法优于基线策略，在减少车队规模的情况下，可提供高达5%更高的感知效用。研究还强调了动态城市数据在优化移动感知策略中的关键作用。

### 结论

车辆感知技术是智慧城市发展的重要工具，所提出的自适应框架能够有效优化车辆感知策略，提高城市数据收集效率。

### 翻译

城市感知对智慧城市的发展至关重要，能够为城市管理提供监测、计算和决策支持。得益于车辆技术的进步，现代车辆正从单纯的交通工具转变为有价值的城市数据收集传感器，有潜力改善交通拥堵、交通可持续性和基础设施检查。基于车辆的感知技术因其灵活性、成本效益和广泛的时空覆盖而日益被认为是一种有前景的技术。然而，优化感知策略以平衡时空覆盖、减少冗余和应对预算限制仍然是一个关键挑战。本研究提出了一种自适应框架，用于增强配备传感器的车辆的感知效用。通过整合异构开源数据，该框架利用时空权重来优化各种城市背景下的车辆选择和感知覆盖。开发了一种基于熵的车辆选择策略（Improved OptiFleet），以在最小化冗余的同时最大化感知效用。该框架使用中国广州320辆配备传感器的车辆在两个月内收集的真实空气质量数据进行了验证。关键发现表明，所提出的方法优于基线策略，在减少车队规模的情况下可提供高达5%更高的感知效用，同时也强调了动态城市数据在优化移动感知策略中的关键作用。


### 论文摘要

Urban sensing is essential for the development of smart cities, enabling monitoring, computing, and decision-making for urban management.Thanks to the advent of vehicle technologies, modern vehicles are transforming from solely mobility tools to valuable sensors for urban data collection, and hold the potential of improving traffic congestion, transport sustainability, and infrastructure inspection.Vehicle-based sensing is increasingly recognized as a promising technology due to its flexibility, cost-effectiveness, and extensive spatiotemporal coverage. However, optimizing sensing strategies to balance spatial and temporal coverage, minimize redundancy, and address budget constraints remains a key challenge.This study proposes an adaptive framework for enhancing the sensing utility of sensor-equipped vehicles.By integrating heterogeneous open-source data, the framework leverages spatiotemporal weighting to optimize vehicle selection and sensing coverage across various urban contexts.An entropy-based vehicle selection strategy, \texttt{Improved OptiFleet}, is developed to maximize sensing utility while minimizing redundancy.The framework is validated using real-world air quality data from 320 sensor-equipped vehicles operating in Guangzhou, China, over two months.Key findings show that the proposed method outperforms baseline strategies, providing up to 5\% higher sensing utility with reduced fleet sizes, and also highlights the critical role of dynamic urban data in optimizing mobile sensing strategies.

---

## 52. Offload Rethinking by Cloud Assistance for Efficient Environmental Sound Recognition on LPWANs

**论文链接:** [http://arxiv.org/abs/2502.15285v3](http://arxiv.org/abs/2502.15285v3)

**作者:** Le Zhang, Quanling Zhao, Run Wang, Shirley Bian, Onat Gungor, Flavio Ponzina, Tajana Rosing

**发布时间:** 2025-02-21

**备注:** Accepted by The 23rd ACM Conference on Embedded Networked Sensor Systems (SenSys '25)

### GPT解析

### 总结

ORCA是一种创新的资源高效云辅助环境声音识别系统，专为无电池设备设计，通过低功耗广域网络运行，实现了广域音频传感应用中的高效声音识别。

### 背景

基于学习的环境声音识别是生物研究和城市规模传感系统中超低功耗环境监测的关键方法。现有系统面临资源有限和偏远地区能量收集供电的挑战，设备端声音识别因资源限制准确率低，而云卸载策略则受高通信成本阻碍。

### 目的

开发ORCA系统，解决资源受限设备在低功耗广域网络上进行环境声音识别的问题，提高识别准确率同时降低通信成本和能耗。

### 方法

提出云辅助策略，结合基于自注意力的云子频谱特征选择方法，促进高效设备端推理，解决LPWAN上三个关键挑战：高通信成本和低数据速率、动态无线信道条件、不可靠卸载。

### 主要发现

在能量收集的无电池微控制器上实现ORCA，并在真实世界城市声音测试床中评估。结果显示ORCA比最先进方法节能高达80倍，延迟减少220倍，同时保持可比的准确率。

### 结论

ORCA有效解决了资源受限设备上环境声音识别的准确率与能耗之间的矛盾，为广域音频传感应用提供了高效可行的解决方案。

### 翻译

基于学习的环境声音识别已成为生物研究和城市规模传感系统中超低功耗环境监测的关键方法。这些系统通常在资源有限的情况下运行，并由偏远地区收集的能量供电。设备端声音识别的最新工作因资源限制而准确率低，而云卸载策略则因高通信成本而受阻。在本工作中，我们介绍了ORCA，这是一种新颖的资源高效云辅助环境声音识别系统，运行在无电池设备上，通过低功耗广域网络，针对广域音频传感应用。我们提出了一种云辅助策略，弥补了设备端推理的低准确率，同时最小化了云卸载的通信成本。通过利用基于自注意力的云子频谱特征选择方法促进高效的设备端推理，ORCA解决了LPWAN上资源受限云卸载的三个关键挑战：1)高通信成本和低数据速率，2)动态无线信道条件，3)不可靠的卸载。我们在一个能量收集的无电池微控制器上实现了ORCA，并在真实世界城市声音测试床中对其进行了评估。我们的结果表明，ORCA在保持可比准确率的同时，比最先进的方法节能高达80倍，延迟减少220倍。


### 论文摘要

Learning-based environmental sound recognition has emerged as a crucial method for ultra-low-power environmental monitoring in biological research and city-scale sensing systems. These systems usually operate under limited resources and are often powered by harvested energy in remote areas. Recent efforts in on-device sound recognition suffer from low accuracy due to resource constraints, whereas cloud offloading strategies are hindered by high communication costs. In this work, we introduce ORCA, a novel resource-efficient cloud-assisted environmental sound recognition system on batteryless devices operating over the Low-Power Wide-Area Networks (LPWANs), targeting wide-area audio sensing applications. We propose a cloud assistance strategy that remedies the low accuracy of on-device inference while minimizing the communication costs for cloud offloading. By leveraging a self-attention-based cloud sub-spectral feature selection method to facilitate efficient on-device inference, ORCA resolves three key challenges for resource-constrained cloud offloading over LPWANs: 1) high communication costs and low data rates, 2) dynamic wireless channel conditions, and 3) unreliable offloading. We implement ORCA on an energy-harvesting batteryless microcontroller and evaluate it in a real world urban sound testbed. Our results show that ORCA outperforms state-of-the-art methods by up to $80 \times$ in energy savings and $220 \times$ in latency reduction while maintaining comparable accuracy.

---

## 53. Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection

**论文链接:** [http://arxiv.org/abs/2303.14311v1](http://arxiv.org/abs/2303.14311v1)

**作者:** Anurag Ghosh, N. Dinesh Reddy, Christoph Mertz, Srinivasa G. Narasimhan

**发布时间:** 2023-03-25

**备注:** CVPR 2023 Accepted Paper, 21 pages, 16 Figures

### GPT解析

### 总结

本文提出了一种可学习的几何引导先验方法，利用3D场景的粗糙几何结构来重新采样图像，以提高小目标和远处目标的检测性能，同时保持实时性和效率。

### 背景

实时高效感知对自主导航和城市规模传感至关重要。除了架构改进外，流式感知方法利用自适应采样提高了实时检测性能。

### 目的

提出一种可学习的几何引导先验，结合3D场景的粗糙几何信息（地面平面和上方平面）来重新采样图像，实现高效的目标检测。

### 方法

开发了一种结合3D场景几何信息的可学习先验方法，通过重新采样图像来优化目标检测过程，特别关注小目标和远处目标的检测。

### 主要发现

该方法显著提高了小目标和远处目标的检测性能；在自主导航场景中，小目标检测率提高+39%，实时性能提高+63%；在固定交通摄像头场景中，能在其他方法无法实现的图像尺度上检测小目标，比朴素下采样方法提高195%，比最先进技术提高63%。

### 结论

几何引导先验方法能有效提高小目标和远处目标的检测性能，同时在延迟和内存使用方面也更有效率，适用于自主导航和城市规模传感应用。

### 翻译

实时高效感知对自主导航和城市规模传感至关重要。与架构改进不同，流式感知方法利用自适应采样提高了实时检测性能。在这项工作中，我们提出了一种可学习的几何引导先验，结合3D场景的粗糙几何结构（地面平面和上方平面）来重新采样图像，以实现高效的目标检测。这显著提高了小目标和远处目标的检测性能，同时在延迟和内存方面也更有效率。对于自主导航，使用相同的检测器和尺度，我们的方法将小目标的检测率提高了+4.1 AP_S或+39%，实时性能提高了+5.3 sAP_S或+63%。对于固定交通摄像头，我们的方法能够在其他方法无法实现的图像尺度上检测小目标。在相同尺度下，我们的方法比朴素下采样方法提高小目标检测195%（+12.5 AP_S），比最先进技术提高63%（+4.2 AP_S）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在资源受限的边缘设备上实现高效实时物体检测的问题。在自动驾驶和城市规模感知应用中，需要处理大量图像数据（如交通摄像头每天捕获50万帧图像），但边缘设备资源有限，无法直接处理高分辨率图像。传统均匀下采样虽然节省资源，但会严重影响小物体和远距离物体的检测准确性，这对自动驾驶安全和城市监控至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类视觉系统获取灵感，人类能通过场景几何的高级语义高效识别物体。作者观察到大多数感兴趣的物体存在于两个平面区域（地面和上方平面），它们在图像中的大小遵循几何关系。作者借鉴了神经变形机制用于图像分类和检测的工作，但指出Fovea等方法的端到端训练显著性网络在物体检测中失败，转而使用启发式方法效果不佳。因此，作者设计了基于双平面透视先验的方法，通过学习几何参数实现非均匀采样，放大远处区域以提高小物体检测性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于学习的双平面透视先验，利用3D场景的几何知识（地面平面和上方平面）重新采样图像，实现高效物体检测。整体流程包括：1)参数化两个平面模型，学习几何参数；2)从鸟瞰图定义显著性函数并映射到相机视图；3)结合地面和上方平面的显著性图；4)使用显著性引导进行图像变形；5)根据场景类型获取消失点（固定摄像头缓存、自动驾驶回归或一般情况计算）；6)使用伪标签学习几何参数，使模型适应新领域。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双平面透视先验同时处理地面和上方物体；2)端到端学习几何参数而非使用启发式方法；3)基于几何关系的非均匀采样策略；4)多场景适应性和新视角泛化能力；5)伪标签学习方法减少对真实标签的依赖。相比之前工作，本文的几何先验专门针对检测任务，而非Fovea等方法使用的固定先验或基于前一帧的先验；相比传统均匀下采样，本文方法显著提高了小物体检测性能；相比其他自适应采样方法，本文关注空间采样而非仅尺度选择。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于双平面透视先验的可学习图像重采样方法，通过利用场景几何知识显著提高了资源受限边缘设备上小物体的检测性能，同时保持了实时性。'}


### 论文摘要

Real-time efficient perception is critical for autonomous navigation and city scale sensing. Orthogonal to architectural improvements, streaming perception approaches have exploited adaptive sampling improving real-time detection performance. In this work, we propose a learnable geometry-guided prior that incorporates rough geometry of the 3D scene (a ground plane and a plane above) to resample images for efficient object detection. This significantly improves small and far-away object detection performance while also being more efficient both in terms of latency and memory. For autonomous navigation, using the same detector and scale, our approach improves detection rate by +4.1 $AP_{S}$ or +39% and in real-time performance by +5.3 $sAP_{S}$ or +63% for small objects over state-of-the-art (SOTA). For fixed traffic cameras, our approach detects small objects at image scales other methods cannot. At the same scale, our approach improves detection of small objects by 195% (+12.5 $AP_{S}$) over naive-downsampling and 63% (+4.2 $AP_{S}$) over SOTA.

---

## 54. In-Context and Few-Shots Learning for Forecasting Time Series Data based on Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.07705v1](http://arxiv.org/abs/2512.07705v1)

**作者:** Saroj Gopali, Bipin Chhetri, Deepika Giri, Sima Siami-Namini, Akbar Siami Namin

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究调查了使用大型语言模型进行时间序列数据预测的性能，并将其与现有方法和Google的TimesFM基础模型进行比较。研究发现TimesFM表现最佳，而OpenAI的o4-mini在零样本学习下也表现良好。

### 背景

现有时间序列数据建模方法包括ARIMA、基于Transformer的模型、LSTM和TCN，其中深度学习模型如LSTM和TCN已显示出良好效果。随着预训练基础模型(如LLMs和Google的TimesFM)的发展，研究它们是否能超越现有方法具有重要意义。

### 目的

调查基础模型在分析和预测时间序列数据方面是否能够超越现有建模方法，并研究使用LLM模型进行时间序列数据预测的性能。

### 方法

研究特定应用领域基础模型训练中的上下文学习方法，探索通过上下文学习、零样本学习和少样本学习训练LLM，使用OpenAI的o4-mini、Gemini 2.5 Flash Lite、TimesFM以及TCN和LSTM网络进行时间序列预测并进行比较。

### 主要发现

TimesFM表现最佳，具有最低的RMSE值和有竞争力的推理时间。OpenAI的o4-mini基于零样本学习也表现出良好的性能。

### 结论

预训练的时间序列基础模型是实时预测的一个有前途的方向，能够实现准确且可扩展的部署，且只需要最小的模型调整。

### 翻译

现有时间序列数据建模和预测的数据驱动方法包括ARIMA(自回归积分移动平均)、基于Transformer的模型、LSTM(长短期记忆)和TCN(时间卷积网络)。这些方法，特别是基于深度学习的模型如LSTM和TCN，在预测时间序列数据方面已显示出优异的结果。随着利用预训练基础模型(如大型语言模型LLMs)以及Google最近的时间序列数据基础模型TimesFM的发展，研究这些基础模型是否能够超越现有建模方法在分析和预测时间序列数据方面的表现引起了研究兴趣。本文研究了使用LLM模型进行时间序列数据预测的性能。我们研究了在特定应用领域基础模型训练中的上下文学习方法。更具体地说，本文探索了通过上下文学习、零样本学习和少样本学习训练LLM，并使用OpenAI的o4-mini和Gemini 2.5 Flash Lite以及Google最近的基于Transformer的TimesFM(一个时间序列特定的基础模型)，连同两种深度学习模型(即TCN和LSTM网络)进行时间序列预测。研究结果表明，TimesFM具有最佳的整体性能，最低的RMSE值和有竞争力的推理时间。此外，OpenAI的o4-mini基于零样本学习也表现出良好的性能。这些研究结果强调了预训练的时间序列基础模型作为实时预测的一个有前途的方向，能够实现准确且可扩展的部署，只需最小的模型调整。


### 论文摘要

Existing data-driven approaches in modeling and predicting time series data include ARIMA (Autoregressive Integrated Moving Average), Transformer-based models, LSTM (Long Short-Term Memory) and TCN (Temporal Convolutional Network). These approaches, and in particular deep learning-based models such as LSTM and TCN, have shown great results in predicting time series data. With the advancement of leveraging pre-trained foundation models such as Large Language Models (LLMs) and more notably Google's recent foundation model for time series data, {\it TimesFM} (Time Series Foundation Model), it is of interest to investigate whether these foundation models have the capability of outperforming existing modeling approaches in analyzing and predicting time series data.   This paper investigates the performance of using LLM models for time series data prediction. We investigate the in-context learning methodology in the training of LLM models that are specific to the underlying application domain. More specifically, the paper explores training LLMs through in-context, zero-shot and few-shot learning and forecasting time series data with OpenAI {\tt o4-mini} and Gemini 2.5 Flash Lite, as well as the recent Google's Transformer-based TimesFM, a time series-specific foundation model, along with two deep learning models, namely TCN and LSTM networks. The findings indicate that TimesFM has the best overall performance with the lowest RMSE value (0.3023) and the competitive inference time (266 seconds). Furthermore, OpenAI's o4-mini also exhibits a good performance based on Zero Shot learning.   These findings highlight pre-trained time series foundation models as a promising direction for real-time forecasting, enabling accurate and scalable deployment with minimal model adaptation.

---

## 55. Neural Compress-and-Forward for the Primitive Diamond Relay Channel

**论文链接:** [http://arxiv.org/abs/2512.07662v1](http://arxiv.org/abs/2512.07662v1)

**作者:** Ozan Aygün, Ezgi Ozyilkan, Elza Erkip

**发布时间:** 2025-12-08

**备注:** Accepted to 2025 59th Asilomar Conference on Signals, Systems, and Computers

### GPT解析

### 总结

研究钻石中继信道中基于神经网络的压缩转发方法，特别是在无感知中继情况下如何实现双中继系统的有效分布式压缩。

### 背景

钻石中继信道是协作通信的典型模型，其中源节点通过两个并行中继与目标节点通信。每个中继观察到源信号的有噪声版本，并通过正交、无噪、有限速率的链路将压缩描述转发到目标节点。

### 目的

将单中继信道的神经压缩转发方法扩展到双中继情况，实现无需中继间协调的分布式压缩，并在无感知中继条件下保持性能。

### 方法

提出使用基于学习的量化器在中继处分别压缩观测值，这些量化器能够远程协作利用输入相关性。目标节点联合解码源消息。方案使用有限阶调制进行端到端训练。

### 主要发现

仿真结果表明，所提出的神经压缩转发方案接近已知理论边界，证明神经CF可以扩展到多中继系统同时保持性能和可解释性。

### 结论

学习型量化器可以在没有中继间协调的情况下实现有效的分布式压缩，符合Berger-Tung风格编码，为多中继系统提供了可行的解决方案。

### 翻译

钻石中继信道是一种源节点通过两个并行中继与目标节点通信的协作通信典型模型。我们关注原始变体，其中每个中继观察到源信号的有噪声版本，并通过正交、无噪、有限速率的链路将压缩描述转发到目标节点。压缩转发(CF)在这种设置中特别有效，尤其是在中继无法访问源编码簿的无感知中继情况下。虽然单中继信道的神经CF方法已被研究，但将其扩展到双中继情况并非易事，因为它需要完全分布式压缩且没有任何中继间协调。我们证明中继处的基于学习的量化器可以通过远程但协作的方式操作，利用输入相关性，实现符合Berger-Tung风格编码的有效分布式压缩。每个中继使用一次性学习的量化器分别压缩其观测值，目标节点联合解码源消息。仿真结果表明，所提出的方案使用有限阶调制进行端到端训练，操作接近已知理论边界。这些结果表明神经CF可以扩展到多中继系统，同时保持性能和可解释性。


### 论文摘要

The diamond relay channel, where a source communicates with a destination via two parallel relays, is one of the canonical models for cooperative communications. We focus on the primitive variant, where each relay observes a noisy version of the source signal and forwards a compressed description over an orthogonal, noiseless, finite-rate link to the destination. Compress-and-forward (CF) is particularly effective in this setting, especially under oblivious relaying where relays lack access to the source codebook. While neural CF methods have been studied in single-relay channels, extending them to the two-relay case is non-trivial, as it requires fully distributed compression without any inter-relay coordination. We demonstrate that learning-based quantizers at the relays can harness input correlations by operating remote, yet in a collaborative fashion, enabling effective distributed compression in line with Berger-Tung-style coding. Each relay separately compresses its observation using a one-shot learned quantizer, and the destination jointly decodes the source message. Simulation results show that the proposed scheme, trained end-to-end with finite-order modulation, operates close to the known theoretical bounds. These results demonstrate that neural CF can scale to multi-relay systems while maintaining both performance and interpretability.

---

## 56. Complementary Learning Approach for Text Classification using Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.07583v1](http://arxiv.org/abs/2512.07583v1)

**作者:** Navid Asgari, Benjamin M. Cole

**发布时间:** 2025-12-08

**DOI:** 10.2139/ssrn.5577090

**备注:** 67 pages

### GPT解析

### 总结

本研究提出了一种结构化方法，以经济高效和精简的方式利用大型语言模型，结合学者和机器的优势，弥补各自不足。

### 背景

大型语言模型(LLMs)在定量研究中的应用存在成本和效率问题，需要更有效的人机协作方法。

### 目的

开发一种方法，使学者能够以低成本方式利用LLMs的优势，同时管理其固有弱点。

### 方法

通过思维链和少样本学习提示，将定性研究中的合著团队最佳实践扩展到定量研究的人机团队，允许人类使用溯因推理和自然语言质疑机器和人类决策。

### 主要发现

该方法使学者能够使用精心设计的低成本技术管理LLMs的弱点，并能有效分析人机评分差异。

### 结论

该方法为人机协作在定量研究中提供了新的可能性，使人类能够有效利用LLMs的能力同时保持对过程的控制。

### 翻译

本研究提出了一种结构化方法，以经济高效和精简的方式利用大型语言模型，整合学者和机器的优势，同时弥补各自的不足。我们的方法通过计算机科学中的思维链和少样本学习提示促进，将定性研究中合著团队的最佳实践扩展到定量研究中的人机团队。这允许人类使用溯因推理和自然语言来质疑机器和人类所做的决策。我们的方法展示了学者如何使用精心设计的低成本技术来管理LLMs的固有弱点。我们演示了如何使用该方法来分析1,934份宣布制药联盟的新闻稿(1990-2017年)中的人机评分差异。


### 论文摘要

In this study, we propose a structured methodology that utilizes large language models (LLMs) in a cost-efficient and parsimonious manner, integrating the strengths of scholars and machines while offsetting their respective weaknesses. Our methodology, facilitated through a chain of thought and few-shot learning prompting from computer science, extends best practices for co-author teams in qualitative research to human-machine teams in quantitative research. This allows humans to utilize abductive reasoning and natural language to interrogate not just what the machine has done but also what the human has done. Our method highlights how scholars can manage inherent weaknesses OF LLMs using careful, low-cost techniques. We demonstrate how to use the methodology to interrogate human-machine rating discrepancies for a sample of 1,934 press releases announcing pharmaceutical alliances (1990-2017).

---

## 57. The Meta-Learning Gap: Combining Hydra and Quant for Large-Scale Time Series Classification

**论文链接:** [http://arxiv.org/abs/2512.06666v1](http://arxiv.org/abs/2512.06666v1)

**作者:** Urav Maniar

**发布时间:** 2025-12-07

**备注:** Link to the repository: https://github.com/urav06/research

### GPT解析

### 总结

时间序列分类面临准确性和计算效率之间的权衡，研究通过结合互补范式的两种高效算法来获得集成方法的好处同时保持计算可行性

### 背景

时间序列分类面临准确性和计算效率之间的基本权衡。全面的集成方法如HIVE-COTE 2.0达到了最先进的准确性，但在UCR基准上需要340小时的训练时间，对于大规模数据集不切实际

### 目的

研究是否可以通过结合来自互补范式的两种高效算法来获得集成方法的好处，同时保持计算可行性

### 方法

结合Hydra（竞争卷积核）和Quant（分层区间分位数）两种算法，在六种集成配置中进行评估，在10个大规模MONSTER数据集（7,898至1,168,774个训练实例）上评估性能

### 主要发现

最强的配置将平均准确性从0.829提高到0.836，在10个数据集中的7个上取得成功；预测组合集成仅捕捉到11%的理论预言潜能，揭示了显著的元学习优化差距；特征级联方法通过学习新的决策边界超过了预言界限；预测级互补性与集成增益显示出适度的相关性

### 结论

挑战已经从确保算法不同转变为如何有效地组合它们；当前的元学习策略难以利用预言分析确认存在的互补性；改进的组合策略有可能在各种时间序列分类应用中将集成收益提高两到三倍

### 翻译

时间序列分类面临准确性和计算效率之间的基本权衡。虽然全面的集成方法如HIVE-COTE 2.0实现了最先进的准确性，但它们在UCR基准上需要340小时的训练时间，对于大规模数据集来说不切实际。我们研究是否可以通过结合来自互补范式的两种高效算法来获得集成方法的好处，同时保持计算可行性。在六种集成配置中结合Hydra（竞争卷积核）和Quant（分层区间分位数），我们在10个大规模MONSTER数据集（7,898至1,168,774个训练实例）上评估性能。我们最强的配置将平均准确性从0.829提高到0.836，在10个数据集中的7个上取得成功。然而，预测组合集成仅捕捉到11%的理论预言潜能，揭示了显著的元学习优化差距。特征级联方法通过学习新的决策边界超过了预言界限，而预测级互补性与集成增益显示出适度的相关性。核心发现：挑战已经从确保算法不同转变为如何有效地组合它们。当前的元学习策略难以利用预言分析确认存在的互补性。改进的组合策略有可能在各种时间序列分类应用中将集成收益提高两到三倍。


### 论文摘要

Time series classification faces a fundamental trade-off between accuracy and computational efficiency. While comprehensive ensembles like HIVE-COTE 2.0 achieve state-of-the-art accuracy, their 340-hour training time on the UCR benchmark renders them impractical for large-scale datasets. We investigate whether targeted combinations of two efficient algorithms from complementary paradigms can capture ensemble benefits while maintaining computational feasibility. Combining Hydra (competing convolutional kernels) and Quant (hierarchical interval quantiles) across six ensemble configurations, we evaluate performance on 10 large-scale MONSTER datasets (7,898 to 1,168,774 training instances). Our strongest configuration improves mean accuracy from 0.829 to 0.836, succeeding on 7 of 10 datasets. However, prediction-combination ensembles capture only 11% of theoretical oracle potential, revealing a substantial meta-learning optimization gap. Feature-concatenation approaches exceeded oracle bounds by learning novel decision boundaries, while prediction-level complementarity shows moderate correlation with ensemble gains. The central finding: the challenge has shifted from ensuring algorithms are different to learning how to combine them effectively. Current meta-learning strategies struggle to exploit the complementarity that oracle analysis confirms exists. Improved combination strategies could potentially double or triple ensemble gains across diverse time series classification applications.

---

## 58. POLARIS: Is Multi-Agentic Reasoning the Next Wave in Engineering Self-Adaptive Systems?

**论文链接:** [http://arxiv.org/abs/2512.04702v2](http://arxiv.org/abs/2512.04702v2)

**作者:** Divyansh Pandey, Vyakhya Gupta, Prakhar Singhal, Karthik Vaidhyanathan

**发布时间:** 2025-12-04

**备注:** Accepted as a short paper at SEAMS 2026

### GPT解析

### 总结

本文介绍了POLARIS，一个三层多智能体自适应框架，用于处理现代软件生态系统中的不确定性挑战，超越了传统的反应性自适应方法。

### 背景

现代软件生态系统的规模、复杂性、互连性和自主性不断增长，引入了前所未有的不确定性，挑战了传统自适应的基础。现有方法难以推广到新环境或协调分布式子系统的响应，无法应对新出现的未知未知。

### 目的

开发一个能够处理不确定性、从过去经验学习并预见变化的框架，使系统能够保持有弹性的目标导向行为。

### 方法

POLARIS是一个三层多智能体自适应框架：(1)低延迟适配器层用于监控和安全执行；(2)透明推理层使用工具感知、可解释的智能体生成和验证计划；(3)元层记录经验并随时间元学习改进的自适应策略。

### 主要发现

POLARIS通过共享知识和预测模型处理不确定性，从过去行动中学习，并发展其策略。在SWIM和SWITCH两个自适应示例上的初步评估显示，POLARIS持续优于最先进的基线。

### 结论

POLARIS标志着向自适应3.0的转变，类似于软件3.0范式，系统不仅从环境中学习，还推理和演化自己的自适应过程，不断改进以应对新挑战。

### 翻译

现代软件生态系统的规模、复杂性、互连性和自主性不断增长，引入了前所未有的不确定性，挑战了传统自适应的基础。现有方法，通常是基于规则的控制器或孤立的学习组件，难以推广到新环境或协调分布式子系统的响应，无法应对新出现的未知未知。关于自适应2.0的最新讨论强调AI与自适应系统之间的平等伙伴关系，将学习驱动的智能与自适应控制相结合，以实现预测性和主动性。在此基础上，我们引入了POLARIS，一个三层多智能体自适应框架，超越了反应性自适应。POLARIS集成了：(1)用于监控和安全执行的低延迟适配器层；(2)透明推理层，使用工具感知、可解释的智能体生成和验证计划；(3)元层，记录经验并随时间元学习改进的自适应策略。通过共享知识和预测模型，POLARIS处理不确定性，从过去行动中学习，并发展其策略，使系统能够预见变化并保持有弹性的目标导向行为。在SWIM和SWITCH两个自适应示例上的初步评估显示，POLARIS持续优于最先进的基线。我们认为这标志着向自适应3.0的转变，类似于软件3.0：一个范式，系统不仅从环境中学习，还推理和演化自己的自适应过程，不断改进以应对新挑战。


### 论文摘要

The growing scale, complexity, interconnectivity, and autonomy of modern software ecosystems introduce unprecedented uncertainty, challenging the foundations of traditional self-adaptation. Existing approaches, typically rule-driven controllers or isolated learning components, struggle to generalize to novel contexts or coordinate responses across distributed subsystems, leaving them ill-equipped for emergent unknown unknowns. Recent discussions on Self-Adaptation 2.0 emphasize an equal partnership between AI and adaptive systems, merging learning-driven intelligence with adaptive control for predictive and proactive behavior. Building on this foundation, we introduce POLARIS, a three-layer multi-agentic self-adaptation framework that advances beyond reactive adaptation. POLARIS integrates: (1) a low-latency Adapter layer for monitoring and safe execution, (2) a transparent Reasoning layer that generates and verifies plans using tool-aware, explainable agents, and (3) a Meta layer that records experiences and meta-learns improved adaptation policies over time. Through shared knowledge and predictive models, POLARIS handles uncertainty, learns from past actions, and evolves its strategies, enabling systems that anticipate change and maintain resilient, goal-directed behavior. Preliminary evaluation on two self-adaptive exemplars, SWIM and SWITCH, shows that POLARIS consistently outperforms state-of-the-art baselines. We argue this marks a shift toward Self-Adaptation 3.0, akin to Software 3.0: a paradigm where systems not only learn from their environment but also reason about and evolve their own adaptation processes, continuously improving to meet novel challenges.

---

## 59. Distribution Matching Variational AutoEncoder

**论文链接:** [http://arxiv.org/abs/2512.07778v1](http://arxiv.org/abs/2512.07778v1)

**作者:** Sen Ye, Jianning Pei, Mengde Xu, Shuyang Gu, Chunyu Wang, Liwei Wang, Han Hu

**发布时间:** 2025-12-08

### GPT解析

### 总结

这项研究提出了Distribution-Matching VAE（DMVAE），一种新的视觉生成模型，通过明确对齐编码器的潜在分布与任意参考分布，超越了传统VAE的高斯先验限制，实现了更好的图像重建和建模效率。

### 背景

大多数视觉生成模型在应用扩散或自回归建模前会将图像压缩到潜在空间。然而，现有方法如VAE和基础模型对齐编码器隐式约束了潜在空间，而没有明确塑造其分布，使得不清楚哪种类型的分布最适合建模。

### 目的

引入DMVAE，通过分布匹配约束明确将编码器的潜在分布与任意参考分布对齐，从而研究哪种潜在分布更有利于建模，并提高图像生成质量。

### 方法

提出Distribution-Matching VAE（DMVAE），通过分布匹配约束将编码器的潜在分布与任意参考分布对齐。这种方法超越了传统VAE的高斯先验限制，能够与自监督特征、扩散噪声或其他先验衍生的分布对齐。

### 主要发现

自监督学习（SSL）衍生的分布在重建保真度和建模效率之间提供了良好的平衡，在ImageNet上仅用64个训练周期就达到了3.2的gFID。

### 结论

选择合适的潜在分布结构（通过分布级对齐实现）而不是依赖固定先验，是弥合易建模潜在变量和高保真图像合成之间差距的关键。

### 翻译

大多数视觉生成模型在应用扩散或自回归建模前会将图像压缩到潜在空间。然而，现有方法如VAE和基础模型对齐编码器隐式约束了潜在空间，而没有明确塑造其分布，使得不清楚哪种类型的分布最适合建模。我们引入Distribution-Matching VAE（DMVAE），它通过分布匹配约束将编码器的潜在分布与任意参考分布明确对齐。这超越了传统VAE的高斯先验限制，能够与自监督特征、扩散噪声或其他先验衍生的分布对齐。通过DMVAE，我们可以系统地研究哪些潜在分布更有利于建模，我们发现自监督学习衍生的分布在重建保真度和建模效率之间提供了良好的平衡，在ImageNet上仅用64个训练周期就达到了3.2的gFID。我们的结果表明，选择合适的潜在分布结构（通过分布级对齐实现）而不是依赖固定先验，是弥合易建模潜在变量和高保真图像合成之间差距的关键。代码可在https://github.com/sen-ye/dmvae获取。


### 论文摘要

Most visual generative models compress images into a latent space before applying diffusion or autoregressive modelling. Yet, existing approaches such as VAEs and foundation model aligned encoders implicitly constrain the latent space without explicitly shaping its distribution, making it unclear which types of distributions are optimal for modeling. We introduce \textbf{Distribution-Matching VAE} (\textbf{DMVAE}), which explicitly aligns the encoder's latent distribution with an arbitrary reference distribution via a distribution matching constraint. This generalizes beyond the Gaussian prior of conventional VAEs, enabling alignment with distributions derived from self-supervised features, diffusion noise, or other prior distributions. With DMVAE, we can systematically investigate which latent distributions are more conducive to modeling, and we find that SSL-derived distributions provide an excellent balance between reconstruction fidelity and modeling efficiency, reaching gFID equals 3.2 on ImageNet with only 64 training epochs. Our results suggest that choosing a suitable latent distribution structure (achieved via distribution-level alignment), rather than relying on fixed priors, is key to bridging the gap between easy-to-model latents and high-fidelity image synthesis. Code is avaliable at https://github.com/sen-ye/dmvae.

---

## 60. PVeRA: Probabilistic Vector-Based Random Matrix Adaptation

**论文链接:** [http://arxiv.org/abs/2512.07703v1](http://arxiv.org/abs/2512.07703v1)

**作者:** Leo Fillioux, Enzo Ferrante, Paul-Henry Cournède, Maria Vakalopoulou, Stergios Christodoulidis

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了PVeRA，一种VeRA适配器的概率版本，在VTAB-1k基准测试中优于VeRA和其他适配器。

### 背景

大型基础模型近年来出现并在各种任务上推动性能边界，但训练或微调这些模型需要大量数据集和计算资源，而这些资源通常稀缺且昂贵。

### 目的

开发一种计算效率高的解决方案，允许大型模型在少量数据和计算能力下进行微调，以解决资源稀缺问题。

### 方法

提出PVeRA，一种VeRA适配器的概率版本，通过概率方式修改VeRA的低秩矩阵，允许处理输入中的固有模糊性，并支持训练和测试期间使用不同的采样配置。

### 主要发现

在VTAB-1k基准和七种适配器上的全面评估显示，PVeRA优于VeRA和其他适配器。

### 结论

PVeRA是一种有效的参数高效适应方法，能够处理输入中的固有模糊性，并提供更好的性能。

### 翻译

大型基础模型在过去几年中已经出现，正在推动各种任务的性能边界。训练甚至微调此类模型需要大量数据集和计算资源，而这些资源通常稀缺且昂贵。适应方法通过允许此类模型在少量数据和计算能力下进行微调，提供了一种计算效率高的解决方案来应对这些限制。这是通过向冻结的主干网络添加新的可训练模块来实现的，这些模块只有一小部分可训练参数，并且只在新型任务上训练这些模块。最近，VeRA适配器被证明在参数高效适应方面表现出色，它利用一对冻结的随机低秩矩阵，这些矩阵在所有层间共享。在本文中，我们提出了PVeRA，即VeRA适配器的概率版本，它以概率方式修改VeRA的低秩矩阵。这种修改自然地允许处理输入中的固有模糊性，并允许在训练和测试期间使用不同的采样配置。我们在VTAB-1k基准和七种适配器上进行了全面评估，PVeRA优于VeRA和其他适配器。我们提供用于使用PVeRA训练模型和对所有适配器进行基准测试的代码：https://github.com/leofillioux/pvera。


### 论文摘要

Large foundation models have emerged in the last years and are pushing performance boundaries for a variety of tasks. Training or even finetuning such models demands vast datasets and computational resources, which are often scarce and costly. Adaptation methods provide a computationally efficient solution to address these limitations by allowing such models to be finetuned on small amounts of data and computing power. This is achieved by appending new trainable modules to frozen backbones with only a fraction of the trainable parameters and fitting only these modules on novel tasks. Recently, the VeRA adapter was shown to excel in parameter-efficient adaptations by utilizing a pair of frozen random low-rank matrices shared across all layers. In this paper, we propose PVeRA, a probabilistic version of the VeRA adapter, which modifies the low-rank matrices of VeRA in a probabilistic manner. This modification naturally allows handling inherent ambiguities in the input and allows for different sampling configurations during training and testing. A comprehensive evaluation was performed on the VTAB-1k benchmark and seven adapters, with PVeRA outperforming VeRA and other adapters. Our code for training models with PVeRA and benchmarking all adapters is available https://github.com/leofillioux/pvera.

---

## 61. Incorporating Structure and Chord Constraints in Symbolic Transformer-based Melodic Harmonization

**论文链接:** [http://arxiv.org/abs/2512.07627v1](http://arxiv.org/abs/2512.07627v1)

**作者:** Maximos Kaliakatsos-Papakostas, Konstantinos Soiledis, Theodoros Tsamis, Dimos Makris, Vassilis Katsouros, Emilios Cambouropoulos

**发布时间:** 2025-12-08

**DOI:** 10.5281/zenodo.16948248

**备注:** Proceedings of the 6th Conference on AI Music Creativity (AIMC 2025), Brussels, Belgium, September 10th-12th

### GPT解析

### 总结

本文研究在旋律和声中纳入预定义和弦约束的问题，提出了一种名为B*的算法来强制预训练的Transformer模型在正确位置满足和弦约束。

### 背景

Transformer架构在生成符号音乐方面具有显著优势，目前许多研究正在探索如何利用这些架构来满足用户对生成内容的偏好。

### 目的

研究在特定位置提供期望和弦作为输入时，如何使自回归Transformer模型在生成的和声中融入该和弦。

### 方法

提出了一种名为B*的算法，结合了束搜索、A*搜索和回溯技术，强制预训练的Transformer满足和弦约束，确保在正确的节拍位置和正确的节拍内。

### 主要发现

该算法在最坏情况下具有指数复杂度，是一种暴力搜索方法，但它提供了许多改进的可能性。

### 结论

本文是首次尝试强调该问题的困难性，并提出了一种允许引入启发式方法的算法框架，为未来研究奠定了基础。

### 翻译

Transformer架构在生成符号音乐方面具有显著优势；关于如何利用这些架构来满足用户对生成内容的偏好，正从多个方面进行研究。本文研究在旋律和声中纳入预定义和弦约束的问题，即在特定位置提供期望和弦作为输入，自回归Transformer模型需要在生成的和声中融入该和弦。文章讨论了涉及此类约束的特殊性，并提出了一种称为B*的算法来解决此任务。该算法结合了束搜索和A*搜索以及回溯技术，强制预训练的Transformer满足和弦约束，确保在正确的节拍位置和正确的节拍内。该算法在最坏情况下具有指数复杂度，是一种暴力搜索方法；然而，本文是首次尝试强调该问题的困难性，并提出了一种算法，由于它允许引入启发式方法，因此提供了许多改进的可能性。


### 论文摘要

Transformer architectures offer significant advantages regarding the generation of symbolic music; their capabilities for incorporating user preferences toward what they generate is being studied under many aspects. This paper studies the inclusion of predefined chord constraints in melodic harmonization, i.e., where a desired chord at a specific location is provided along with the melody as inputs and the autoregressive transformer model needs to incorporate the chord in the harmonization that it generates. The peculiarities of involving such constraints is discussed and an algorithm is proposed for tackling this task. This algorithm is called B* and it combines aspects of beam search and A* along with backtracking to force pretrained transformers to satisfy the chord constraints, at the correct onset position within the correct bar. The algorithm is brute-force and has exponential complexity in the worst case; however, this paper is a first attempt to highlight the difficulties of the problem and proposes an algorithm that offers many possibilities for improvements since it accommodates the involvement of heuristics.

---

## 62. Time Series Foundation Models for Process Model Forecasting

**论文链接:** [http://arxiv.org/abs/2512.07624v1](http://arxiv.org/abs/2512.07624v1)

**作者:** Yongbo Yu, Jari Peeperkorn, Johannes De Smedt, Jochen De Weerdt

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究评估了时间序列基础模型(TSFMs)在过程模型预测(PMF)任务上的应用效果，发现TSFMs在预测过程控制流结构演变方面表现优于传统模型，即使零样本使用也能取得良好效果。

### 背景

过程模型预测(PMF)旨在通过建模直接跟随关系的时态动力学来预测控制流结构如何随时间演变，补充了专注于单个案例前缀的预测过程监控。先前研究表明机器学习和深度学习模型仅比统计基线提供适度改进，主要由于DF时间序列的稀疏性和异质性。

### 目的

研究时间序列基础模型(TSFMs)作为PMF的替代方法，评估其在预测过程控制流结构演变方面的效果，并比较零样本使用和微调变体的性能差异。

### 方法

使用从真实生活事件日志中导出的DF时间序列，比较TSFMs的零样本使用(无需额外训练)和针对PMF特定数据微调的变体，使用MAE和RMSE作为评估指标。

### 主要发现

TSFMs通常比在同一日志上从头开始训练的传统和专门模型产生更低的预测误差，表明了从非过程领域迁移时态结构的能力。虽然微调可以进一步提高准确性，但收益通常较小，且在较小或更复杂的数据集上可能消失，因此零样本使用仍然是一个强大的默认选择。

### 结论

TSFMs在过程相关时间序列上展示了良好的泛化能力和数据效率，据我们所知，这是首次对PMF的时间基础模型进行的系统评估。

### 翻译

过程模型预测(PMF)旨在通过建模直接跟随关系的时态动力学来预测过程控制流结构如何随时间演变，这补充了专注于单个案例前缀的预测过程监控。先前的基准测试表明，机器学习和深度学习模型仅比统计基线提供适度改进，这主要是由于DF时间序列的稀疏性和异质性。我们研究了时间序列基础模型(TSFMs)作为PMF的替代方案，这是一种针对通用时间序列的大型预训练模型。使用从真实生活事件日志中导出的DF时间序列，我们比较了TSFMs的零样本使用(无需额外训练)和针对PMF特定数据微调的变体。TSFMs通常比在同一日志上从头开始训练的传统和专门模型产生更低的预测误差(MAE和RMSE)，这表明了从非过程领域迁移时态结构的能力。虽然微调可以进一步提高准确性，但收益通常较小，并且在较小或更复杂的数据集上可能会消失，因此零样本使用仍然是一个强大的默认选择。我们的研究强调了TSFMs在过程相关时间序列上的泛化能力和数据效率，据我们所知，这为PMF提供了时间基础模型的首次系统评估。


### 论文摘要

Process Model Forecasting (PMF) aims to predict how the control-flow structure of a process evolves over time by modeling the temporal dynamics of directly-follows (DF) relations, complementing predictive process monitoring that focuses on single-case prefixes. Prior benchmarks show that machine learning and deep learning models provide only modest gains over statistical baselines, mainly due to the sparsity and heterogeneity of the DF time series. We investigate Time Series Foundation Models (TSFMs), large pre-trained models for generic time series, as an alternative for PMF. Using DF time series derived from real-life event logs, we compare zero-shot use of TSFMs, without additional training, with fine-tuned variants adapted on PMF-specific data. TSFMs generally achieve lower forecasting errors (MAE and RMSE) than traditional and specialized models trained from scratch on the same logs, indicating effective transfer of temporal structure from non-process domains. While fine-tuning can further improve accuracy, the gains are often small and may disappear on smaller or more complex datasets, so zero-shot use remains a strong default. Our study highlights the generalization capability and data efficiency of TSFMs for process-related time series and, to the best of our knowledge, provides the first systematic evaluation of temporal foundation models for PMF.

---

## 63. Metric-Fair Prompting: Treating Similar Samples Similarly

**论文链接:** [http://arxiv.org/abs/2512.07608v1](http://arxiv.org/abs/2512.07608v1)

**作者:** Jing Wang, Jie Shen, Xing Niu, Tong Zhang, Jeremy Weiss

**发布时间:** 2025-12-08

### GPT解析

### 总结

提出了一种名为'Metric-Fair Prompting'的公平感知提示框架，用于指导大型语言模型在公平性约束下做决策。

### 背景

在多项选择医疗问答应用中，每个{(问题, 选项)}对被视为一个二元实例，标签为正确或不正确。

### 目的

促进个体公平性，使相似实例得到相似对待，并提高大型语言模型在高风险临床多项选择题上的准确性。

### 方法

使用NLP嵌入计算问题相似性，在相似问题的联合对中解决问题；执行全局决策协议，提取决定性临床特征，将每个(问题, 选项)映射为置信度分数，并施加Lipschitz风格约束使相似输入获得相似分数。

### 主要发现

在MedQA (US)基准上评估，Metric-Fair Prompting比标准单项提示表现更好，证明公平引导、置信度导向的推理可以提高LLM在高风险临床多项选择题上的准确性。

### 结论

公平引导、置信度导向的推理可以增强大型语言模型在高风险临床多项选择题上的准确性。

### 翻译

我们引入了'Metric-Fair Prompting'，一种公平感知的提示框架，指导大型语言模型在公平性约束下做决策。在多项选择医疗问答应用中，每个{(问题, 选项)}对被视为一个二元实例，标签为正确或不正确。为了促进个体公平性——相似实例得到相似对待——我们使用NLP嵌入计算问题相似性，并在相似问题的联合对中解决问题，而不是孤立地解决。提示执行全局决策协议：提取决定性临床特征，将每个(问题, 选项)映射为作为置信度的分数，并施加Lipschitz风格约束，使相似输入获得相似分数，从而产生一致的输出。在MedQA (US)基准上评估，Metric-Fair Prompting显示出比标准单项提示更好的性能，证明公平引导、置信度导向的推理可以提高LLM在高风险临床多项选择题上的准确性。


### 论文摘要

We introduce \emph{Metric-Fair Prompting}, a fairness-aware prompting framework that guides large language models (LLMs) to make decisions under metric-fairness constraints. In the application of multiple-choice medical question answering, each {(question, option)} pair is treated as a binary instance with label $+1$ (correct) or $-1$ (incorrect). To promote {individual fairness}~--~treating similar instances similarly~--~we compute question similarity using NLP embeddings and solve items in \emph{joint pairs of similar questions} rather than in isolation. The prompt enforces a global decision protocol: extract decisive clinical features, map each \((\text{question}, \text{option})\) to a score $f(x)$ that acts as confidence, and impose a Lipschitz-style constraint so that similar inputs receive similar scores and, hence, consistent outputs. Evaluated on the {MedQA (US)} benchmark, Metric-Fair Prompting is shown to improve performance over standard single-item prompting, demonstrating that fairness-guided, confidence-oriented reasoning can enhance LLM accuracy on high-stakes clinical multiple-choice questions.

---

## 64. LongCat-Image Technical Report

**论文链接:** [http://arxiv.org/abs/2512.07584v1](http://arxiv.org/abs/2512.07584v1)

**作者:** Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, Jie Hu

**发布时间:** 2025-12-08

### GPT解析

### 总结

LongCat-Image是一个开创性的开源双语(中英文)图像生成基础模型，通过严格的数据筛选策略和紧凑设计解决了多语言文本渲染、照片级真实感、部署效率和开发者可访问性等挑战，在文本渲染、中文字符处理、模型效率和图像编辑方面均达到最先进水平。

### 背景

当前主流图像生成模型在多语言文本渲染、照片级真实感、部署效率和开发者可访问性方面存在核心挑战。

### 目的

开发一个开源双语(中英文)图像生成基础模型，解决现有模型在多语言文本渲染、照片级真实感、部署效率和开发者可访问性方面的挑战。

### 方法

通过在预训练、中训练和SFT阶段采用严格的数据筛选策略，在RL阶段协调使用筛选的奖励模型；采用仅60亿参数的紧凑型核心扩散模型设计；建立了最全面的开源生态系统，发布多个模型版本和完整的训练工具链。

### 主要发现

1) 建立了新的最先进水平(SOTA)，提供卓越的文本渲染能力和显著的照片级真实感，显著提升美学质量；2) 为中文字符渲染设定新行业标准，支持复杂和罕见字符，在覆盖率和准确性上优于主要开源和商业解决方案；3) 模型效率显著，比领域内常见的大型MoE架构小得多，确保最少的VRAM使用和快速推理，显著降低部署成本；4) 在图像编辑方面表现优异，在标准基准测试中实现SOTA结果，具有更好的编辑一致性。

### 结论

LongCat-Image的开源性将为开发者和研究人员提供强有力的支持，推动视觉内容创作的前沿发展。

### 翻译

我们介绍了LongCat-Image，这是一个开创性的开源双语(中英文)图像生成基础模型，旨在解决当前主流模型中普遍存在的多语言文本渲染、照片级真实感、部署效率和开发者可访问性等核心挑战。1) 我们通过在预训练、中训练和SFT阶段采用严格的数据筛选策略，并在RL阶段协调使用筛选的奖励模型来实现这一目标。这一策略使该模型成为新的最先进水平(SOTA)，提供卓越的文本渲染能力和显著的照片级真实感，并显著提升了美学质量。2) 值得注意的是，它为中文字符渲染设定了新的行业标准。通过支持甚至复杂和罕见的字符，它在覆盖率上优于主要的开源和商业解决方案，同时实现了更高的准确性。3) 该模型通过其紧凑设计实现了显著效率。仅拥有60亿参数的核心扩散模型，它明显小于领域内常见的近200亿或更大的Mixture-of-Experts(MoE)架构。这确保了最少的VRAM使用和快速推理，显著降低了部署成本。除了生成功能外，LongCat-Image在图像编辑方面也表现出色，在标准基准测试中实现了SOTA结果，与其他开源作品相比具有更好的编辑一致性。4) 为了完全赋能社区，我们建立了迄今为止最全面的开源生态系统。我们不仅发布了用于文本到图像和图像编辑的多个模型版本，包括中训练和后训练阶段的检查点，还发布了整个训练流程的工具链。我们相信，LongCat-Image的开源性将为开发者和研究人员提供强有力的支持，推动视觉内容创作的前沿发展。


### 论文摘要

We introduce LongCat-Image, a pioneering open-source and bilingual (Chinese-English) foundation model for image generation, designed to address core challenges in multilingual text rendering, photorealism, deployment efficiency, and developer accessibility prevalent in current leading models. 1) We achieve this through rigorous data curation strategies across the pre-training, mid-training, and SFT stages, complemented by the coordinated use of curated reward models during the RL phase. This strategy establishes the model as a new state-of-the-art (SOTA), delivering superior text-rendering capabilities and remarkable photorealism, and significantly enhancing aesthetic quality. 2) Notably, it sets a new industry standard for Chinese character rendering. By supporting even complex and rare characters, it outperforms both major open-source and commercial solutions in coverage, while also achieving superior accuracy. 3) The model achieves remarkable efficiency through its compact design. With a core diffusion model of only 6B parameters, it is significantly smaller than the nearly 20B or larger Mixture-of-Experts (MoE) architectures common in the field. This ensures minimal VRAM usage and rapid inference, significantly reducing deployment costs. Beyond generation, LongCat-Image also excels in image editing, achieving SOTA results on standard benchmarks with superior editing consistency compared to other open-source works. 4) To fully empower the community, we have established the most comprehensive open-source ecosystem to date. We are releasing not only multiple model versions for text-to-image and image editing, including checkpoints after mid-training and post-training stages, but also the entire toolchain of training procedure. We believe that the openness of LongCat-Image will provide robust support for developers and researchers, pushing the frontiers of visual content creation.

---

## 65. Tessellation GS: Neural Mesh Gaussians for Robust Monocular Reconstruction of Dynamic Objects

**论文链接:** [http://arxiv.org/abs/2512.07381v1](http://arxiv.org/abs/2512.07381v1)

**作者:** Shuohan Tao, Boyao Zhou, Hanzhang Tu, Yuwang Wang, Yebin Liu

**发布时间:** 2025-12-08

### GPT解析

### 总结

Tessellation GS是一种创新的2D高斯泼溅方法，通过基于网格面的约束和自适应细分策略，显著改善了动态场景重建质量，特别是在稀疏视图条件下。

### 背景

3D高斯泼溅技术能从带姿态的图像序列实现高度逼真的场景重建，但由于其各向异性特性，在视角外推方面存在困难，导致过拟合和泛化能力差，特别是在稀疏视图和动态场景重建中表现不佳。

### 目的

提出一种名为Tessellation GS的结构化2D高斯泼溅方法，该方法基于网格面锚定，用于从单个连续移动或静态摄像机重建动态场景。

### 方法

将2D高斯约束在局部区域内，通过网格面上的分层神经特征推断它们的属性；高斯细分由感知细节的损失函数驱动的自适应面细分策略引导；同时利用重建基础模型的先验来初始化高斯变形，使基于优化的方法能够从单个静态摄像机重建一般动态对象。

### 主要发现

该方法优于之前的最佳方法，在外观和网格重建任务上，LPIPS降低了29.1%，Chamfer距离降低了49.2%。

### 结论

Tessellation GS有效解决了传统3D高斯泼溅在动态场景重建中的局限性，特别是从单个静态摄像机重建一般动态对象的挑战。

### 翻译

3D高斯泼溅技术能够从带有姿态的图像序列实现高度逼真的场景重建，但由于其各向异性特性，在视角外推方面存在困难，导致过拟合和泛化能力差，特别是在稀疏视图和动态场景重建中表现不佳。我们提出了Tessellation GS，一种基于网格面锚定的结构化2D高斯泼溅方法，用于从单个连续移动或静态摄像机重建动态场景。我们的方法将2D高斯约束在局部区域内，并通过网格面上的分层神经特征推断它们的属性。高斯细分由感知细节的损失函数驱动的自适应面细分策略引导。此外，我们利用重建基础模型的先验来初始化高斯变形，使得基于优化的方法能够从单个静态摄像机重建一般动态对象，这对于以前的方法来说极具挑战性。我们的方法优于之前的最佳方法，在外观和网格重建任务上，LPIPS降低了29.1%，Chamfer距离降低了49.2%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D高斯泼溅技术在动态场景重建中视角外推能力差的问题，导致过拟合和泛化能力弱。这个问题在现实中很重要，因为通常只能从有限角度（如单目相机）观察动态物体，而现有方法在新视角下渲染效果差，限制了虚拟现实、增强现实和数字人等应用中高质量动态物体重建的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D GS方法的局限性，发现缺乏几何约束导致高斯点自由延伸造成过拟合。他们借鉴了网格表示方法的优势，将高斯点锚定在网格面上以提供几何约束；同时利用大型重建模型(LRM)获取粗略几何先验。设计了两阶段优化流程：第一阶段提取运动和几何信息，第二阶段进行结构化2D GS重建。借鉴了3D GS、Mesh-based GS、LRM和变形模型等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出结构化2D高斯泼溅方法，将高斯点锚定在网格面上，使用层次化神经特征推断属性，并通过自适应面细分策略引导高斯细分。整体流程分两阶段：第一阶段用LRM获取每帧粗略几何网格，建立帧间对应关系并优化变形场；第二阶段在规范网格上初始化结构化2D高斯，通过网格-高斯四叉树自适应控制高斯密度，联合优化变形模型和高斯属性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：结构化2D高斯锚定在网格面上避免视角过拟合；自适应面细分策略通过细节感知损失函数自动添加高斯；利用LRM先验初始化高斯变形提高鲁棒性；两阶段优化流程。相比传统3D GS，减少了视角过拟合；相比基于模板方法，不需类别特定模板且能处理拓扑变化；相比其他动态重建方法，在稀疏视图和自然相机运动下表现更好；相比DG-Mesh等提供了更高保真度和更低内存负担。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Tessellation GS通过将结构化2D高斯点锚定在网格面上并利用自适应细分策略，实现了从单目视频中鲁棒重建动态物体，显著提高了新视角渲染质量并减少了几何和外观重建误差。'}


### 论文摘要

3D Gaussian Splatting (GS) enables highly photorealistic scene reconstruction from posed image sequences but struggles with viewpoint extrapolation due to its anisotropic nature, leading to overfitting and poor generalization, particularly in sparse-view and dynamic scene reconstruction. We propose Tessellation GS, a structured 2D GS approach anchored on mesh faces, to reconstruct dynamic scenes from a single continuously moving or static camera. Our method constrains 2D Gaussians to localized regions and infers their attributes via hierarchical neural features on mesh faces. Gaussian subdivision is guided by an adaptive face subdivision strategy driven by a detail-aware loss function. Additionally, we leverage priors from a reconstruction foundation model to initialize Gaussian deformations, enabling robust reconstruction of general dynamic objects from a single static camera, previously extremely challenging for optimization-based methods. Our method outperforms previous SOTA method, reducing LPIPS by 29.1% and Chamfer distance by 49.2% on appearance and mesh reconstruction tasks.

---

## 66. Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning

**论文链接:** [http://arxiv.org/abs/2512.07374v1](http://arxiv.org/abs/2512.07374v1)

**作者:** Yezi Liu, Hanning Chen, Wenjun Huang, Yang Ni, Mohsen Imani

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文介绍了一种名为Recover-to-Forget (R2F)的新型框架，用于在大语言模型中进行高效的知识遗忘，无需完整重新训练或访问原始训练数据。

### 背景

在大基础模型（如LLMs）中进行unlearning对于实现动态知识更新、强制执行数据删除权利和纠正模型行为至关重要。然而，现有方法通常需要对完整模型进行微调或访问原始训练数据，限制了可扩展性和实用性。

### 目的

开发一种可扩展且实用的unlearning方法，不需要完整重新训练或访问内部参数，同时保持模型的一般性能。

### 方法

R2F框架通过从低秩LoRA适配器更新重建全模型梯度方向来实现高效unlearning。使用多个释义提示计算相对于LoRA参数的梯度，训练梯度解码器近似全模型梯度，并在代理模型上训练后转移到目标模型。

### 主要发现

提供了跨模型泛化的理论分析，证明R2F实现了有效的unlearning同时保留模型一般性能，实验结果表明它是预训练LLMs中unlearning的可扩展轻量级替代方案。

### 结论

R2F框架提供了一种高效、可扩展的unlearning方法，不需要完整重新训练或访问原始训练数据，同时保持模型的一般性能。

### 翻译

在大基础模型（如LLMs）中进行unlearning对于实现动态知识更新、强制执行数据删除权利和纠正模型行为至关重要。然而，现有的unlearning方法通常需要对完整模型进行微调或访问原始训练数据，这限制了它们的可扩展性和实用性。在这项工作中，我们引入了Recover-to-Forget (R2F)，一种基于从低秩LoRA适配器更新重建全模型梯度方向的高效LLMs unlearning的新型框架。我们不是通过完整模型进行反向传播，而是使用多个释义提示计算相对于LoRA参数的梯度，并训练一个梯度解码器来近似相应的全模型梯度。为确保适用于更大或黑盒模型，解码器在代理模型上训练并转移到目标模型。我们提供了跨模型泛化的理论分析，并证明我们的方法实现了有效的unlearning，同时保留了模型的一般性能。实验结果表明，R2F为预训练LLMs中的unlearning提供了一种可扩展且轻量级的替代方案，不需要完整重新训练或访问内部参数。


### 论文摘要

Unlearning in large foundation models (e.g., LLMs) is essential for enabling dynamic knowledge updates, enforcing data deletion rights, and correcting model behavior. However, existing unlearning methods often require full-model fine-tuning or access to the original training data, which limits their scalability and practicality. In this work, we introduce Recover-to-Forget (R2F), a novel framework for efficient unlearning in LLMs based on reconstructing full-model gradient directions from low-rank LoRA adapter updates. Rather than performing backpropagation through the full model, we compute gradients with respect to LoRA parameters using multiple paraphrased prompts and train a gradient decoder to approximate the corresponding full-model gradients. To ensure applicability to larger or black-box models, the decoder is trained on a proxy model and transferred to target models. We provide a theoretical analysis of cross-model generalization and demonstrate that our method achieves effective unlearning while preserving general model performance. Experimental results demonstrate that R2F offers a scalable and lightweight alternative for unlearning in pretrained LLMs without requiring full retraining or access to internal parameters.

---

## 67. VFM-VLM: Vision Foundation Model and Vision Language Model based Visual Comparison for 3D Pose Estimation

**论文链接:** [http://arxiv.org/abs/2512.07215v1](http://arxiv.org/abs/2512.07215v1)

**作者:** Md Selim Sarowar, Sungho Kim

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文对基于CLIP和基于DINOv2的方法在3D姿态估计（手部物体抓取场景）方面进行了全面的视觉比较，评估了它们在6D物体姿态估计任务上的表现。

### 背景

视觉基础模型（VFMs）和视觉语言模型（VLMs）通过提供丰富的语义和几何表征彻底改变了计算机视觉领域。

### 目的

对基于CLIP和基于DINOv2的方法在3D姿态估计（手部物体抓取场景）方面进行全面的视觉比较。

### 方法

评估这两种模型在6D物体姿态估计任务上的表现，通过语言基础提供语义理解和密集几何特征。研究者在基准数据集上进行了大量实验。

### 主要发现

基于CLIP的方法在语义一致性方面表现更好，而基于DINOv2的方法则表现出增强的几何精度和具有竞争力的性能。

### 结论

分析为选择适用于机器人操作、抓取和拾取应用的视觉模型提供了见解。

### 翻译

视觉基础模型（VFMs）和视觉语言模型（VLMs）通过提供丰富的语义和几何表征彻底改变了计算机视觉。本文对基于CLIP和基于DINOv2的方法在手部物体抓取场景中的3D姿态估计进行了全面的视觉比较。我们在6D物体姿态估计任务上评估了这两种模型，并展示了它们的互补优势：CLIP通过语言基础在语义理解方面表现出色，而DINOv2提供更优越的密集几何特征。通过在基准数据集上的大量实验，我们表明基于CLIP的方法实现了更好的语义一致性，而基于DINOv2的方法则表现出增强的几何精度和具有竞争力的性能。我们的分析为选择适用于机器人操作、抓取和拾取应用的视觉模型提供了见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要比较了基于CLIP（视觉语言模型）和DINOv2（视觉基础模型）两种方法在3D姿态估计（特别是手部物体抓取场景中的6D物体姿态估计）中的性能差异。这个问题很重要，因为机器人操作需要同时具备语义理解和几何精度，而现有深度学习方法缺乏机器人操作所需的上下文知识。理解这两种模型的互补优势有助于为机器人抓取应用选择合适的视觉模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到从纯几何方法向语义感知系统转变的必要性，注意到两种新兴的自监督学习范式：DINOv2通过自蒸馏捕获视觉特征，CLIP通过对比学习联合视觉语言表示。他们设计了两种不同架构：基于CLIP的强调语义理解，基于DINOv2的强调密集几何特征。该方法借鉴了现有的视觉语言模型、自监督学习、PnP算法和ICP算法等技术，并将它们应用于3D姿态估计任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心是比较两种不同类型的视觉模型（CLIP擅长语义理解，DINOv2擅长几何精度）在3D姿态估计中的表现，探索它们的互补优势。基于CLIP的流程包括：特征提取→语义增强→跨模态融合→姿态回归。基于DINOv2的流程包括：密集特征提取→关键点检测→几何推理→优化。两种方法都使用标准评估指标（ADD、ADD-S、旋转误差、平移误差）进行评估，并通过可视化分析预测质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：系统比较CLIP和DINOv2在3D姿态估计中的性能；通过2D关键点投影和3D边界框进行视觉分析；提供语义与几何精度之间权衡的见解。相比之前工作，这项研究同时评估了两种不同类型的模型，充分探索了它们在3D姿态估计中的应用，强调了互补优势，并提出了结合两者的混合方法方向，而之前工作可能更专注于单一模型或特定应用场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统比较CLIP和DINOv2两种视觉模型在3D姿态估计中的表现，揭示了语义理解和几何精度之间的权衡，为机器人抓取应用提供了选择合适视觉模型的指导，并提出了结合两者优势的混合方法方向。'}


### 论文摘要

Vision Foundation Models (VFMs) and Vision Language Models (VLMs) have revolutionized computer vision by providing rich semantic and geometric representations. This paper presents a comprehensive visual comparison between CLIP based and DINOv2 based approaches for 3D pose estimation in hand object grasping scenarios. We evaluate both models on the task of 6D object pose estimation and demonstrate their complementary strengths: CLIP excels in semantic understanding through language grounding, while DINOv2 provides superior dense geometric features. Through extensive experiments on benchmark datasets, we show that CLIP based methods achieve better semantic consistency, while DINOv2 based approaches demonstrate competitive performance with enhanced geometric precision. Our analysis provides insights for selecting appropriate vision models for robotic manipulation and grasping, picking applications.

---

## 68. PlantBiMoE: A Bidirectional Foundation Model with SparseMoE for Plant Genomes

**论文链接:** [http://arxiv.org/abs/2512.07113v1](http://arxiv.org/abs/2512.07113v1)

**作者:** Kepeng Lin, Qizhe Zhang, Rui Wang, Xuehai Hu, Wei Xu

**发布时间:** 2025-12-08

**备注:** 6 pages, 5 figures, accept to BIBM

### GPT解析

### 总结

本文提出了一种名为PlantBiMoE的轻量级且表达能力强的植物基因组语言模型，通过整合双向Mamba和稀疏专家混合框架解决了现有方法的局限性，在多项任务上表现优异。

### 背景

理解植物基因组的基本语言规则是计算生物学中的一个基本挑战。近期进展如AgroNT和PDLLMs虽取得一定进展，但存在参数量过大和无法有效建模DNA双链性质的问题。

### 目的

开发一种能够有效捕获DNA双链结构依赖性且计算效率高的植物基因组语言模型，解决现有方法的局限性。

### 方法

提出PlantBiMoE模型，集成了双向Mamba和稀疏专家混合(SparseMoE)框架。在修改后的植物基因组基准测试(MPGB)上进行了评估，该基准整合了11个代表性任务中的31个数据集，输入序列长度从50到6,000 bp。

### 主要发现

PlantBiMoE在31个数据集中的20个上取得了最佳性能，与现有模型相比平均表现最佳，能有效表示植物基因组序列。

### 结论

PlantBiMoE可作为各种基因组任务的强大计算工具，对植物基因组学、基因编辑和合成生物学做出了实质性贡献。代码可在GitHub上获取：https://github.com/HUST-Keep-Lin/PlantBiMoE

### 翻译

理解植物基因组的基本语言规则仍然是计算生物学中的一个基本挑战。尽管最近的进展包括AgroNT和PDLLMs取得了显著进展，但它们分别存在参数量过大和有限地建模DNA双链性质的能力。为解决这些局限性，我们提出了PlantBiMoE，一种轻量级且表达能力强的植物基因组语言模型，集成了双向Mamba和稀疏专家混合框架。双向Mamba使模型能够有效捕获正向和反向DNA链之间的结构依赖性，而稀疏专家混合显著减少了活动参数的数量，提高了计算效率而不牺牲建模能力。我们在修改后的植物基因组基准测试(MPGB)上评估和测试了我们的模型，这是一个增强的基因组基准，整合了11个代表性任务中的31个数据集，输入序列长度从50到6,000 bp不等。实验结果表明，PlantBiMoE在31个数据集中的20个上取得了最佳性能，与现有模型相比平均表现最佳。总之，所有上述结果表明，我们的模型可以有效表示植物基因组序列，作为各种基因组任务的强大计算工具，同时对植物基因组学、基因编辑和合成生物学做出了实质性贡献。代码可在以下网址获取：https://github.com/HUST-Keep-Lin/PlantBiMoE


### 论文摘要

Understanding the underlying linguistic rules of plant genomes remains a fundamental challenge in computational biology. Recent advances including AgroNT and PDLLMs have made notable progress although, they suffer from excessive parameter size and limited ability to model the bidirectional nature of DNA strands respectively. To address these limitations, we propose PlantBiMoE, a lightweight and expressive plant genome language model that integrates bidirectional Mamba and a Sparse Mixture-of-Experts (SparseMoE) framework. The bidirectional Mamba enables the model to effectively capture structural dependencies across both the forward and reverse DNA strands, while SparseMoE significantly reduces the number of active parameters, improving computational efficiency without sacrificing modeling capacity. We evaluated and tested our model on the Modified Plants Genome Benchmark (MPGB), an enhanced genomic benchmark, which consolidates 31 datasets across 11 representative tasks, with input sequence lengths ranging from 50 to 6,000 bp. Experimental results demonstrate that PlantBiMoE achieves the best performance on 20 out of 31 datasets and the average best when comparing with existing models. In summary, all above results demonstrate that our model can effectively represent plant genomic sequences, serving as a robust computational tool for diverse genomic tasks, while making substantive contributions to plant genomics, gene editing, and synthetic biology. The code is available at: https://github.com/HUST-Keep-Lin/PlantBiMoE

---

## 69. Evaluating and Preserving High-level Fidelity in Super-Resolution

**论文链接:** [http://arxiv.org/abs/2512.07037v1](http://arxiv.org/abs/2512.07037v1)

**作者:** Josep M. Rocafort, Shaolin Su, Javier Vazquez-Corral, Alexandra Gomez-Villa

**发布时间:** 2025-12-07

### GPT解析

### 总结

本研究探讨了图像超分辨率模型中高层保真度测量的重要性，构建了首个SR模型保真度数据集，评估了现有模型的表现，分析了质量指标与保真度的相关性，并展示了通过保真度反馈微调模型可同时提升语义保真度和感知质量。

### 背景

当前图像超分辨率模型在重建细节和产生视觉上令人满意的输出方面效果显著，但其过强的生成能力可能导致图像内容改变，这种高层级变化易被人类识别但在现有低层级图像质量指标中研究不足。

### 目的

建立测量高层保真度的重要性，作为揭示生成式SR模型可靠性的补充标准；构建首个带保真度分数的SR数据集；评估现有模型在保持高层保真度方面的表现；分析质量指标与保真度的相关性；探索通过保真度反馈优化模型的潜力。

### 方法

构建首个带有不同SR模型保真度分数的注释数据集；评估最先进SR模型在高层保真度方面的表现；分析现有图像质量指标与保真度测量的相关性；利用基础模型解决高层级任务；通过保真度反馈微调SR模型。

### 主要发现

现有SR模型在保持高层保真度方面存在不足；现有图像质量指标与保真度测量相关性有限；基础模型在解决高层级任务上表现更好；通过保真度反馈微调可同时提升语义保真度和感知质量。

### 结论

测量高层保真度对评估和优化SR模型具有重要意义；提出的保真度标准在模型评估和优化中具有潜在价值；研究将公开数据集、代码和模型以促进领域发展。

### 翻译

最近的图像超分辨率(SR)模型在重建细节和提供视觉上令人满意的输出方面取得了令人印象深刻的效果。然而，过强的生成能力有时会产生幻觉，从而改变图像内容，尽管获得了高视觉质量。这种高层级的变化很容易被人类识别，但在现有的低层级图像质量指标中尚未得到充分研究。在本文中，我们确立了测量SR模型高层保真度的重要性，作为揭示生成式SR模型可靠性的补充标准。我们构建了首个带有不同SR模型保真度分数的注释数据集，并评估了最先进的SR模型在保持高层保真度方面的实际表现。基于该数据集，我们分析了现有图像质量指标与保真度测量的相关性，并进一步展示了这种高层级任务可以更好地由基础模型解决。最后，通过基于我们的保真度反馈微调SR模型，我们证明了语义保真度和感知质量都可以得到提高，展示了我们提出的标准在模型评估和优化中的潜在价值。我们将在论文接受后发布数据集、代码和模型。


### 论文摘要

Recent image Super-Resolution (SR) models are achieving impressive effects in reconstructing details and delivering visually pleasant outputs. However, the overpowering generative ability can sometimes hallucinate and thus change the image content despite gaining high visual quality. This type of high-level change can be easily identified by humans yet not well-studied in existing low-level image quality metrics. In this paper, we establish the importance of measuring high-level fidelity for SR models as a complementary criterion to reveal the reliability of generative SR models. We construct the first annotated dataset with fidelity scores from different SR models, and evaluate how state-of-the-art (SOTA) SR models actually perform in preserving high-level fidelity. Based on the dataset, we then analyze how existing image quality metrics correlate with fidelity measurement, and further show that this high-level task can be better addressed by foundation models. Finally, by fine-tuning SR models based on our fidelity feedback, we show that both semantic fidelity and perceptual quality can be improved, demonstrating the potential value of our proposed criteria, both in model evaluation and optimization. We will release the dataset, code, and models upon acceptance.

---

## 70. Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.06999v1](http://arxiv.org/abs/2512.06999v1)

**作者:** Zihao Wang, Ruibin Yuan, Ziqi Geng, Hengjia Li, Xingwei Qu, Xinyi Li, Songye Chen, Haoying Fu, Roger B. Dannenberg, Kejun Zhang

**发布时间:** 2025-12-07

**DOI:** 10.1145/3746027.3758148

**备注:** Accepted to ACMMM 2025 oral

### GPT解析

### 总结

该论文提出了一种无参考、多维度的歌唱评估生态系统，包括Sing-MD数据集、VocalVerse混合架构和H-TPR基准，解决了现有系统依赖参考曲目和简化复杂表演的问题。

### 背景

自动歌唱评估对于教育和娱乐领域至关重要，但现有系统存在两个基本限制：依赖参考曲目抑制创造性表达，以及将复杂表演简化为仅基于音高和节奏的非诊断性分数。

### 目的

从判别性评估转向描述性评估，创建一个无参考、多维度的完整评估生态系统。

### 方法

1) 引入Sing-MD数据集，由专家在四个维度（呼吸控制、音色质量、情感表达和声乐技巧）上标注；2) 提出VocalVerse高效混合架构，利用轻量级声学编码器建模全局表演特征和长期依赖关系；3) 建立H-TPR基准，评估模型生成感知有效排序的能力。

### 主要发现

专家在标注Sing-MD数据集时存在显著的不一致性，这挑战了传统基于准确性的指标的有效性。

### 结论

通过提出的生态系统，可以实现更全面、更准确的歌唱评估，不再局限于简单的音高和节奏匹配，而是考虑多维度的表演特征。

### 翻译

自动歌唱评估对教育和娱乐至关重要。然而，现有系统面临两个基本限制：依赖参考曲目，这抑制了创造性表达；以及将复杂的表演简化为仅基于音高和节奏的非诊断性分数。我们主张从判别性评估转向描述性评估，创建一个无参考、多维度的完整评估生态系统。首先，我们介绍了Sing-MD，这是一个由专家在四个维度上标注的大规模数据集：呼吸控制、音色质量、情感表达和声乐技巧。我们的分析显示专家之间存在显著的标注不一致性，挑战了传统基于准确性的指标的有效性。其次，为解决多模态大语言模型在分析完整歌曲时的内存限制，我们提出了VocalVerse。这种高效的混合架构利用轻量级声学编码器来建模全局表演特征和长期依赖关系。第三，为解决自动指标的不足，我们建立了H-TPR（人机分层感知排序）基准，该基准评估模型生成感知有效排序的能力，而不是预测有噪声的真实分数。


### 论文摘要

Automated singing assessment is crucial for education and entertainment. However, existing systems face two fundamental limitations: reliance on reference tracks, which stifles creative expression, and the simplification of complex performances into non-diagnostic scores based solely on pitch and rhythm. We advocate for a shift from discriminative to descriptive evaluation, creating a complete ecosystem for reference-free, multi-dimensional assessment. First, we introduce Sing-MD, a large-scale dataset annotated by experts across four dimensions: breath control, timbre quality, emotional expression, and vocal technique. Our analysis reveals significant annotation inconsistencies among experts, challenging the validity of traditional accuracy-based metrics. Second, addressing the memory limitations of Multimodal Large Language Models (MLLMs) in analyzing full-length songs, we propose VocalVerse. This efficient hybrid architecture leverages a lightweight acoustic encoder to model global performance features and long-term dependencies. Third, to address automated metric shortcomings, we establish the H-TPR (Human-in-the-loop Tiered Perceptual Ranking) benchmark, which evaluates a model's ability to generate perceptually valid rankings rather than predicting noisy ground-truth scores.

---

## 71. JT-DA: Enhancing Data Analysis with Tool-Integrated Table Reasoning Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.06859v1](http://arxiv.org/abs/2512.06859v1)

**作者:** Ce Chi, Xing Wang, Zhendong Wang, Xiaofan Liu, Ce Li, Zhiyan Song, Chen Zhao, Kexin Yang, Boshen Shi, Jingjing Yang, Chao Deng, Junlan Feng

**发布时间:** 2025-12-07

### GPT解析

### 总结

本研究提出了JT-DA-8B（九天数据分析师8B），一个专门为复杂表格推理任务设计的大型语言模型。通过构建包含34个表格推理任务的综合训练语料库，并采用监督微调和强化学习方法进行优化，该模型在表格推理任务中表现出色。同时提出了一个四阶段表格推理工作流程以提高模型性能。

### 背景

表格推理场景中缺乏高质量监督数据，这限制了相关模型的发展和应用。

### 目的

开发一个专门用于复杂表格推理任务的大型语言模型，解决表格推理场景中缺乏高质量监督的问题。

### 方法

1. 构建包含34个表格推理任务的综合训练语料库，聚合29个公开表格问答数据集和300万张表格；2. 提出自动流程生成涉及推理模式的多步分析任务；3. 基于开源JT-Coder-8B模型进行训练；4. 利用基于大语言模型的评分和工作流对齐过滤提炼高质量数据；5. 采用监督微调和强化学习优化模型；6. 提出四阶段表格推理工作流程：表格预处理、表格感知、工具集成推理和提示工程。

### 主要发现

实验结果表明，JT-DA-8B在各种表格推理任务中实现了强大的性能，证明了数据为中心的生成和工作流驱动的优化方法的有效性。

### 结论

JT-DA-8B通过高质量数据构建和系统的工作流程优化，成为一个高效的表格推理专用大型语言模型。

### 翻译

在这项工作中，我们提出了JT-DA-8B（九天数据分析师8B），这是一个专门为各种现实场景中复杂表格推理任务设计的大型语言模型。为解决表格推理场景中缺乏高质量监督的问题，我们通过聚合29个公开表格问答数据集和300万张表格，构建了一个包含34个定义明确的表格推理任务的全面且多样化的训练语料库。我们提出了一个自动流程，用于生成涉及推理模式的真实多步分析任务。该模型基于开源的JT-Coder-8B模型进行训练，这是一个从头训练的80亿参数仅解码器基础模型。在训练阶段，我们利用基于大语言模型的评分和工作流对齐过滤来提炼高质量、以表格为中心的数据。同时采用监督微调（SFT）和强化学习（RL）来优化我们的模型。随后，提出了一个四阶段表格推理工作流程，包括表格预处理、表格感知、工具集成推理和提示工程，以提高模型的可解释性和执行准确性。实验结果表明，JT-DA-8B在各种表格推理任务中实现了强大的性能，证明了数据为中心的生成和工作流驱动的优化的有效性。


### 论文摘要

In this work, we present JT-DA-8B (JiuTian Data Analyst 8B), a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse training corpus with 34 well-defined table reasoning tasks, by aggregating 29 public table QA datasets and 3 million tables. An automatic pipeline is proposed to generate realistic multi-step analytical tasks involving reasoning patterns. The model is trained upon open-source JT-Coder-8B model, an 8B-parameter decoder-only foundation model trained from scratch. In the training stage, we leverage LLM-based scoring and workflow-aligned filtering to distill high-quality, table-centric data. Both supervised fine-tuning (SFT) and Reinforcement learning (RL) are adopted to optimize our model. Afterwards, a four-stage table reasoning workflow is proposed, including table preprocessing, table sensing, tool-integrated reasoning, and prompt engineering, to improve model interpretability and execution accuracy. Experimental results show that JT-DA-8B achieves strong performance in various table reasoning tasks, demonstrating the effectiveness of data-centric generation and workflow-driven optimization.

---

## 72. Generalized Geometry Encoding Volume for Real-time Stereo Matching

**论文链接:** [http://arxiv.org/abs/2512.06793v1](http://arxiv.org/abs/2512.06793v1)

**作者:** Jiaxin Liu, Gangwei Xu, Xianqi Wang, Chengliang Zhang, Xin Yang

**发布时间:** 2025-12-07

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为GGEV的新型实时立体匹配网络，通过深度感知特征和动态成本聚合模块实现了强泛化能力，在多个基准测试上超越了现有方法。

### 背景

实时立体匹配方法主要关注领域内性能提升，但往往忽略了现实应用中泛化能力的重要性。最近的立体基础模型利用单眼基础模型(MFMs)提高泛化能力，但通常存在大量的推理延迟。

### 目的

解决泛化能力和推理速度之间的权衡问题，提出一种能够实现强泛化能力的实时立体匹配网络。

### 方法

提出广义几何编码体积(GGEV)。首先提取深度感知特征，将领域不变的结构先验作为成本聚合的指导；然后引入深度感知动态成本聚合(DDCA)模块，将这些先验自适应地合并到每个视差假设中，有效增强未见场景中的脆弱匹配关系。

### 主要发现

实验结果表明，GGEV在零样本泛化能力方面超越了所有现有的实时方法，并在KITTI 2012、KITTI 2015和ETH3D基准测试上取得了最先进的性能。

### 结论

GGEV通过两个轻量级且互补的步骤构建了具有强泛化能力的广义几何编码体积，成功解决了泛化能力和推理速度之间的权衡问题。

### 翻译

实时立体匹配方法主要关注增强领域内性能，但常常忽略了现实应用中泛化能力的关键重要性。相比之下，最近的立体基础模型利用单眼基础模型(MFMs)来提高泛化能力，但通常遭受大量的推理延迟。为了解决这种权衡问题，我们提出了广义几何编码体积(GGEV)，一种能够实现强泛化能力的新型实时立体匹配网络。我们首先提取深度感知特征，将领域不变的结构先验作为成本聚合的指导。随后，我们引入了深度感知动态成本聚合(DDCA)模块，将这些先验自适应地合并到每个视差假设中，有效增强了未见场景中的脆弱匹配关系。这两个步骤都是轻量级且互补的，导致构建了一个具有强泛化能力的广义几何编码体积。实验结果表明，我们的GGEV在零样本泛化能力方面超越了所有现有的实时方法，并在KITTI 2012、KITTI 2015和ETH3D基准测试上取得了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是实时立体匹配方法在保持推理速度的同时提高泛化能力的问题。这个问题很重要，因为立体匹配是3D重建、自动驾驶和机器人导航等应用的基础技术，这些场景对算法的泛化能力和推理延迟都有严格要求，而现有方法要么泛化能力不足，要么无法满足实时性要求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有实时立体匹配方法的局限性：它们依赖清晰明确的匹配线索，在遮挡、无纹理区域等挑战性区域表现不佳；同时分析了使用单目基础模型的方法虽然泛化能力强但推理延迟大的问题。基于这些分析，作者设计了GGEV，借鉴了单目基础模型(如Depth Anything V2)提取深度特征的能力，以及动态卷积核的思想，但创新性地将这些技术与立体匹配的成本聚合过程结合，实现了轻量且高效的特征融合与成本聚合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过深度感知的特征来指导立体匹配的成本聚合过程，利用单目基础模型提取的深度结构先验增强脆弱的匹配关系。整体流程分为四个阶段：1)多线索特征提取，包括纹理特征和深度特征的提取与融合；2)成本体积构建，使用纹理特征创建相关体积；3)深度感知动态成本聚合，根据视差假设与深度特征的亲和力生成动态卷积核；4)深度感知迭代细化，使用GRU逐步优化视差图并利用深度特征辅助恢复全分辨率结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)广义几何编码体积(GGEV)，轻量整合深度先验提高泛化；2)深度感知动态成本聚合(DDCA)模块，自适应生成动态卷积核；3)强大的零样本泛化能力，即使仅在合成数据上训练也能在真实世界表现良好。相比之前的工作，GGEV不依赖昂贵的骨干网络或复杂的迭代机制，而是通过轻量级的深度特征提取和动态卷积实现高效且泛化的成本聚合，解决了实时性与泛化能力之间的权衡问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了GGEV，一种实时立体匹配网络，通过深度感知的特征自适应指导成本聚合，实现了在保持实时性的同时显著提高在未见场景中的泛化能力。'}


### 论文摘要

Real-time stereo matching methods primarily focus on enhancing in-domain performance but often overlook the critical importance of generalization in real-world applications. In contrast, recent stereo foundation models leverage monocular foundation models (MFMs) to improve generalization, but typically suffer from substantial inference latency. To address this trade-off, we propose Generalized Geometry Encoding Volume (GGEV), a novel real-time stereo matching network that achieves strong generalization. We first extract depth-aware features that encode domain-invariant structural priors as guidance for cost aggregation. Subsequently, we introduce a Depth-aware Dynamic Cost Aggregation (DDCA) module that adaptively incorporates these priors into each disparity hypothesis, effectively enhancing fragile matching relationships in unseen scenes. Both steps are lightweight and complementary, leading to the construction of a generalized geometry encoding volume with strong generalization capability. Experimental results demonstrate that our GGEV surpasses all existing real-time methods in zero-shot generalization capability, and achieves state-of-the-art performance on the KITTI 2012, KITTI 2015, and ETH3D benchmarks.

---

## 73. Foundation Model for Polycrystalline Material Informatics

**论文链接:** [http://arxiv.org/abs/2512.06770v1](http://arxiv.org/abs/2512.06770v1)

**作者:** Ting-Ju Wei, Chuin-Shan Chen

**发布时间:** 2025-12-07

### GPT解析

### 总结

该研究提出了一个三维多晶体基础模型，通过大规模自监督预训练学习基于体素微观结构的物理结构化表示，并评估了其在两个不同物理特性下游任务中的性能。

### 背景

在数据稀缺的科学环境中，标记的微观结构有限，而物理一致的泛化对于材料设计至关重要。

### 目的

开发一个能够学习微观结构物理表示的基础模型，并评估其在不同物理特性下游任务中的性能和泛化能力。

### 方法

使用包含10万个FCC微观结构的数据集进行预训练，这些微观结构的晶体学方向跨越纹理包络；采用掩码策略强制模型从不完整空间信息中推断潜在特征；将预训练编码器与方向感知的基于交互的深度材料网络(ODMN)耦合以推断完整网络参数集。

### 主要发现

在均质化刚度预测任务中，预训练编码器在所有掩码比例下都优于非预训练基线；在非线性响应建模任务中，模型能准确预测未见过的微观结构的应力-应变关系；预训练编码器在两项任务中都表现出显著更强的泛化能力。

### 结论

所提出的框架具有很强的可转移性，适合数据稀缺的科学设置，为与实验衍生的微观结构集成提供了可扩展途径，为实际材料设计中的微观结构-属性推理提供了新基础。

### 翻译

我们提出了一个三维多晶体基础模型，通过大规模自监督预训练学习基于体素微观结构的物理结构化表示。编码器在包含10万个FCC微观结构的数据集上进行训练，这些微观结构的晶体学方向跨越纹理包络，使用掩码策略强制模型从不完整的空间信息中推断潜在特征。通过两个具有不同物理特性的下游任务评估学习表示的质量。(i)均质化刚度预测：预训练编码器在所有掩码比例下都优于非预训练基线。(ii)非线性响应建模：将编码器与方向感知的基于交互的深度材料网络(ODMN)耦合，以推断完整的网络参数集，从而能够准确预测以前未见过的微观结构的应力-应变关系。在两项任务中，预训练编码器都表现出显著更强的泛化能力。这些结果强调了所提出框架的强可转移性及其适合数据稀缺科学设置的特性，在这些设置中标记的微观结构有限，物理一致的泛化至关重要。该基础模型为与实验衍生的微观结构集成提供了可扩展的途径，为实际材料设计中的微观结构-属性推理提供了新基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多晶材料微观结构与宏观性能之间关系的建模问题。这个问题很重要，因为多晶材料是现代工程应用的基础，其宏观力学性能由微观结构（特别是晶体学纹理）决定。准确预测这种关系对材料设计和性能优化至关重要，但在数据稀缺环境中，传统方法难以有效处理这种复杂关系。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了自然语言处理领域基础模型（如BERT和GPT）的成功经验，将其应用到材料科学领域。作者首先认识到晶体学纹理对控制多晶材料各向异性力学响应的关键作用，然后构建大规模合成数据集，使用层次单纯形采样遍历FCC晶体的完整纹理包络，开发3D掩码自编码器架构学习纹理感知的潜在表示。作者还参考了掩码自编码器在材料科学中的应用、深度材料网络架构以及现有机器学习方法在多晶系统中的应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过大规模自监督预训练学习多晶材料的物理结构化表示，使用掩码策略迫使模型从不完整空间信息中推断潜在特征，然后将预训练编码器转移到不同物理特性的下游任务。整体流程包括：1)构建包含100,000个FCC微观结构的数据集；2)自监督预训练阶段，通过掩码部分体素块并重建来学习特征；3)下游任务应用，包括均匀化刚度预测和非线性响应建模，分别使用线性回归头和与ODMN网络耦合的方式。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次为多晶微观结构设计的基础模型；2)构建系统遍历纹理包络的大规模数据集；3)开发3D体素基础模型学习纹理感知表示；4)在两个不同物理特性的下游任务上验证模型；5)展示强泛化能力。相比之前工作，不同之处在于：大多数研究局限于二维系统；解决了传统DMN架构参数与特定微观结构相关的局限；针对数据稀缺环境提供解决方案；为实验衍生微观结构集成提供可扩展途径。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文开发了一种基于大规模自监督学习的3D多晶基础模型，通过学习微观结构的物理结构化表示，实现了对多晶材料宏观力学性能（包括线弹性和非线性响应）的准确预测，为数据稀缺环境下的材料设计提供了新途径。'}


### 论文摘要

We present a 3D polycrystal foundation model that learns a physically structured representation of voxel-based microstructures through large-scale self-supervised pretraining. The encoder is trained on a dataset of 100,000 FCC microstructures whose crystallographic orientations span the texture hull, using a masking strategy that forces the model to infer latent features from incomplete spatial information. The quality of the learned representation is evaluated through two downstream tasks with distinct physical characteristics. (i) Homogenized stiffness prediction: the pretrained encoder consistently outperforms the non-pretrained baseline across all masking ratios. (ii) Nonlinear response modeling: the encoder is coupled with an orientation-aware interaction-based deep material network (ODMN) to infer complete sets of network parameters, enabling accurate stress-strain predictions for previously unseen microstructures. In both tasks, the pretrained encoder demonstrates markedly stronger generalization capability. These results underscore the strong transferability of the proposed framework and its suitability for data-scarce scientific settings, where labeled microstructures are limited and physics-consistent generalization is essential. The foundation model provides a scalable route toward integration with experimentally derived microstructures, offering a new basis for microstructure-property reasoning in practical materials design.

---

## 74. OSM+: Billion-Level Open Street Map Data Processing System for City-wide Experiments

**论文链接:** [http://arxiv.org/abs/2512.06743v1](http://arxiv.org/abs/2512.06743v1)

**作者:** Guanjie Zheng, Ziyang Su, Yiheng Wang, Yuhang Luo, Hongwei Zhang, Xuanhe Zhou, Linghe Kong, Fan Wu, Wen Ling

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文介绍了一个通过分布式计算处理OpenStreetMap数据构建的全球十亿顶点道路网络图数据集(OSM+)，该数据集具有高可访问性和可用性，并展示了其在交通预测、城市边界检测和交通政策控制三个应用场景中的使用。

### 背景

道路网络数据可提供丰富的城市信息，是各种城市研究的基础。然而，处理大规模全球道路网络数据需要大量计算资源，且处理结果难以统一，影响下游任务的测试。

### 目的

处理OpenStreetMap数据，构建结构化的全球十亿顶点道路网络图数据集，提高数据集的可访问性和可用性，为城市研究和交通分析提供基础。

### 方法

使用云服务上的5000个核心进行分布式计算处理OpenStreetMap数据，发布开源且可全球下载的数据集，提供开放的图结构和易于使用的空间查询接口。

### 主要发现

1) 交通预测：发布包含31个城市的新基准测试集，提供更大的空间覆盖范围和更全面的算法评估；2) 交通政策控制：发布6个城市的新数据集，规模更大，带来千级多智能体协调的新挑战；3) 数据转换器促进多模态时空数据集成，加速地理空间基础模型训练和科学发现进程。

### 结论

OSM+数据集及其相关工具为城市研究和交通分析提供了强大基础，推动了算法在更大规模网络上的扩展性，加速了科学发现的进程。

### 翻译

道路网络数据可以提供丰富的城市信息，从而成为各种城市研究的基础。然而，处理大规模全球道路网络数据需要大量计算资源，且处理结果可能难以统一以测试下游任务。因此，本文中，我们通过云服务上5000个核心的分布式计算处理OpenStreetMap数据，并发布了一个结构化的全球十亿顶点道路网络图数据集，具有高可访问性（开源且可全球下载）和可用性（开放的图结构和易于使用的空间查询接口）。为展示该数据集的易用性，我们提出了三个说明性用例，包括交通预测、城市边界检测和交通政策控制，并对这三个任务进行了广泛实验。1) 对于研究充分的交通预测任务，我们发布了一个包含31个城市的新基准（交通数据已处理并与我们发布的OSM+道路网络数据集结合），比以往常用的数据集提供更大的空间覆盖范围和更全面的算法评估。这一新基准将推动算法从数百个道路网络交叉路口扩展到数千个交叉路口的可扩展性。2) 对于需要与道路网络交互的更高级交通政策控制任务，我们发布了一个包含6个城市的新数据集，规模远大于之前的数据集。这为千级多智能体协调带来了新的挑战。3) 随着OSM+数据集的发布，数据转换器的发布促进了多模态时空数据的集成，用于地理空间基础模型的训练，从而加速了揭示引人入胜的科学洞察的过程。


### 论文摘要

Road network data can provide rich information about cities and thus become the base for various urban research. However, processing large volume world-wide road network data requires intensive computing resources and the processed results might be different to be unified for testing downstream tasks. Therefore, in this paper, we process the OpenStreetMap data via a distributed computing of 5,000 cores on cloud services and release a structured world-wide 1-billion-vertex road network graph dataset with high accessibility (opensource and downloadable to the whole world) and usability (open-box graph structure and easy spatial query interface). To demonstrate how this dataset can be utilized easily, we present three illustrative use cases, including traffic prediction, city boundary detection and traffic policy control, and conduct extensive experiments for these three tasks. (1) For the well-investigated traffic prediction tasks, we release a new benchmark with 31 cities (traffic data processed and combined with our released OSM+ road network dataset), to provide much larger spatial coverage and more comprehensive evaluation of compared algorithms than the previously frequently-used datasets. This new benchmark will push the algorithms on their scalability from hundreds of road network intersections to thousands of intersections. (2) While for the more advanced traffic policy control task which requires interaction with the road network, we release a new 6 city datasets with much larger scale than the previous datasets. This brings new challenge for thousand-scale multi-agent coordination. (3) Along with the OSM+ dataset, the release of data converters facilitates the integration of multimodal spatial-temporal data for geospatial foundation model training, thereby expediting the process of uncovering compelling scientific insights. PVLDB Reference Forma

---

## 75. Protecting Bystander Privacy via Selective Hearing in LALMs

**论文链接:** [http://arxiv.org/abs/2512.06380v1](http://arxiv.org/abs/2512.06380v1)

**作者:** Xiao Zhan, Guangzhi Sun, Jose Such, Phil Woodland

**发布时间:** 2025-12-06

**备注:** Dataset: https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench

### GPT解析

### 总结

本文提出了SH-Bench基准测试和BPFT训练方法，用于评估和改善音频语言模型的选择性聆听能力，解决旁观者隐私泄露问题。

### 背景

大型音频语言模型(LALMs)在现实世界部署中会无意捕获附近旁观者的语音，带来隐私风险，而现有的基准和防御措施大多忽略了这一点。

### 目的

开发首个评估'选择性聆听'能力的基准测试，即模型专注于预期主要说话者同时拒绝处理意外旁观者语音信息的能力。

### 方法

构建SH-Bench基准包含3,968个多说话者音频混合(真实世界和合成场景)及77k多项选择题；提出选择性效能(SE)指标；开发旁观者隐私微调(BPFT)训练流程，教导模型拒绝旁观者相关查询而不损害主要说话者理解。

### 主要发现

当前先进LALMs存在大量隐私泄露，音频理解能力强并不能转化为对旁观者隐私的保护；BPFT方法显著提升性能，SE最多提高15.9%，超过Gemini 2.5 Pro。

### 结论

SH-Bench和BPFT为音频基础模型中的旁观者隐私测量和改进提供了首个系统性框架，表明选择性聆听是可学习的，但在当前LALMs中远未实现。

### 翻译

大型音频语言模型(LALMs)越来越多地部署在现实环境中，它们不可避免地会捕获附近非预期旁观者的语音，这带来了现有基准和防御措施大多忽视的隐私风险。我们引入了SH-Bench，这是首个旨在评估选择性聆听的基准：模型专注于预期的主要说话者，同时拒绝处理或揭示关于意外旁观者语音信息的能力。SH-Bench包含3,968个多说话者音频混合，涵盖真实世界和合成场景，配对77k个多项选择题，用于测试模型在一般和选择性操作模式下的表现。我们提出了选择性效能(SE)这一统一指标，捕捉多说话者理解和旁观者隐私保护。我们对最先进的开源和专有LALMs的评估显示存在大量隐私泄露，强大的音频理解能力无法转化为对旁观者隐私的选择性保护。为缓解这一差距，我们引入了旁观者隐私微调(BPFT)，这是一种训练流程，教导模型拒绝与旁观者相关的查询，同时不降低主要说话者的理解能力。BPFT带来了显著提升，使SE比Gemini 2.5 Pro提高最多15.9%，表明选择性聆听是可以学习的，但在当前的LALMs中远未实现。SH-Bench和BPFT为音频基础模型中的旁观者隐私测量和改进提供了首个系统性框架。


### 论文摘要

Large audio language models (LALMs) are increasingly deployed in real-world settings where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences largely overlook. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures spanning both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. We propose Selective Efficacy (SE), a unified metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LALMs reveals substantial privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we introduce Bystander Privacy Fine-Tuning (BPFT), a training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. BPFT yields substantial gains which improve SE by up to 15.9% over Gemini 2.5 Pro, demonstrating that selective hearing is learnable but far from achieved in current LALMs. SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio foundation models.

---

## 76. SpectraIrisPAD: Leveraging Vision Foundation Models for Spectrally Conditioned Multispectral Iris Presentation Attack Detection

**论文链接:** [http://arxiv.org/abs/2512.06103v1](http://arxiv.org/abs/2512.06103v1)

**作者:** Raghavendra Ramachandra, Sushma Venkatesh

**发布时间:** 2025-12-05

**备注:** Accepted in IEEE T-BIOM

### GPT解析

### 总结

虹膜识别作为最准确的生物识别模态之一，面临呈现攻击的脆弱性问题，本文提出了SpectraIrisPAD多光谱虹膜攻击检测框架和MSIrPAD数据集，实验证明该方法在检测各种攻击方面具有卓越的鲁棒性和泛化能力。

### 背景

虹膜识别被广泛认为是准确的生物识别模态，但在现实应用中的广泛部署引发了其在呈现攻击面前的脆弱性担忧，有效的呈现攻击检测对确保虹膜生物识别系统的完整性和安全至关重要。

### 目的

开发一种有效的多光谱虹膜呈现攻击检测方法，以提高虹膜识别系统的安全性和鲁棒性。

### 方法

提出SpectraIrisPAD框架，利用DINOv2视觉变换器主干网络，配备可学习的光谱位置编码、令牌融合和对比学习技术，提取具有区分性的波段特定特征；同时创建MSIrPAD数据集，包含在五个不同NIR波长下捕获的18,848张虹膜图像，涵盖八种不同的PAI类别。

### 主要发现

SpectraIrisPAD在所有性能指标上都持续优于几种最先进的基线方法，在检测各种呈现攻击方面表现出卓越的鲁棒性和泛化能力。

### 结论

多光谱成像技术结合深度学习方法可以有效提升虹膜呈现攻击检测的性能，SpectraIrisPAD框架为提高虹膜识别系统的安全性提供了有效解决方案。

### 翻译

虹膜识别被广泛认为是生物识别模态中最准确的一种。然而，它在现实应用中的广泛部署引发了对其在呈现攻击面前脆弱性的重大担忧。因此，有效的呈现攻击检测对于确保虹膜生物识别系统的完整性和安全至关重要。虽然传统虹膜识别系统主要在近红外光谱下运行，但跨越多个NIR波段的多光谱成像可以提供互补的反射信息，增强PAD方法的泛化能力。在这项工作中，我们提出了SpectraIrisPAD，一种新颖的基于深度学习的多光谱虹膜PAD框架。SpectraIrisPAD利用配备了可学习光谱位置编码、令牌融合和对比学习的DINOv2视觉变换器主干网络，提取具有区分性的、波段特定的特征，有效区分真实样本和各种伪造伪影。此外，我们引入了一个新的综合数据集多光谱虹膜PAD(MSIrPAD)，包含多种PAI，使用在五个不同的NIR波长(800nm、830nm、850nm、870nm和980nm)下运行的自定义设计多光谱虹膜传感器捕获。该数据集包含18,848张虹膜图像，涵盖八种不同的PAI类别，包括五种纹理隐形眼镜、打印攻击和基于显示器的攻击。我们在未见过的攻击评估协议下进行了全面实验，以评估所提出方法的泛化能力。SpectraIrisPAD在所有性能指标上都持续优于几种最先进的基线方法，表现出卓越的鲁棒性和泛化能力，能够检测广泛的呈现攻击。


### 论文摘要

Iris recognition is widely recognized as one of the most accurate biometric modalities. However, its growing deployment in real-world applications raises significant concerns regarding its vulnerability to Presentation Attacks (PAs). Effective Presentation Attack Detection (PAD) is therefore critical to ensure the integrity and security of iris-based biometric systems. While conventional iris recognition systems predominantly operate in the near-infrared (NIR) spectrum, multispectral imaging across multiple NIR bands provides complementary reflectance information that can enhance the generalizability of PAD methods. In this work, we propose \textbf{SpectraIrisPAD}, a novel deep learning-based framework for robust multispectral iris PAD. The SpectraIrisPAD leverages a DINOv2 Vision Transformer (ViT) backbone equipped with learnable spectral positional encoding, token fusion, and contrastive learning to extract discriminative, band-specific features that effectively distinguish bona fide samples from various spoofing artifacts. Furthermore, we introduce a new comprehensive dataset Multispectral Iris PAD (\textbf{MSIrPAD}) with diverse PAIs, captured using a custom-designed multispectral iris sensor operating at five distinct NIR wavelengths (800\,nm, 830\,nm, 850\,nm, 870\,nm, and 980\,nm). The dataset includes 18,848 iris images encompassing eight diverse PAI categories, including five textured contact lenses, print attacks, and display-based attacks. We conduct comprehensive experiments under unseen attack evaluation protocols to assess the generalization capability of the proposed method. SpectraIrisPAD consistently outperforms several state-of-the-art baselines across all performance metrics, demonstrating superior robustness and generalizability in detecting a wide range of presentation attacks.

---

## 77. Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm

**论文链接:** [http://arxiv.org/abs/2512.05511v1](http://arxiv.org/abs/2512.05511v1)

**作者:** Chuang Yu, Jinmiao Zhao, Yunpeng Liu, Yaokun Li, Xiujun Shu, Yuanhao Feng, Bo Wang, Yimian Dai, Xiangyu Yue

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究首次将视觉基础模型引入单帧红外小目标检测任务，提出了基础驱动的高效范式(FDEP)，包含语义调制融合模块和协作优化隐式自蒸馏策略，并构建了整体评估指标，显著提升了检测性能。

### 背景

大规模视觉基础模型在多个视觉领域表现出强大泛化能力，但在单帧红外小目标检测方面的潜力尚未被充分探索。

### 目的

首次将视觉基础模型的冻结表征引入SIRST任务，提出一种基础驱动的高效范式，可无缝适配现有编码器-解码器方法，显著提高精度而不增加推理开销。

### 方法

设计语义调制融合模块实现全局语义先验与任务特定特征的动态对齐和深度融合；提出协作优化隐式自蒸馏策略通过参数共享实现隐式语义转移；构建整体SIRST评估指标进行多阈值积分评估。

### 主要发现

配备FDEP框架的SIRST检测网络在多个公共数据集上取得了最先进的性能。

### 结论

FDEP框架有效提升了SIRST检测性能，代码已公开在GitHub上。

### 翻译

虽然大规模视觉基础模型(VFMs)在多样化的视觉领域表现出强大的泛化能力，但它们在单帧红外小目标(SIRST)检测方面的潜力很大程度上尚未被探索。为了填补这一空白，我们首次将来自VFMs的冻结表征系统性地引入SIRST任务，并提出了一种基础驱动的高效范式(FDEP)，它可以无缝适配现有的基于编码器-解码器的方法，显著提高精度而无需额外的推理开销。具体来说，设计了一个语义调制融合模块(SAMF)，以实现VFM的全局语义先验与任务特定特征的动态对齐和深度融合。同时，为了避免VFM引入的推理时间负担，我们提出了一个基于协作优化的隐式自蒸馏(CO-ISD)策略，通过参数共享和同步反向传播实现主分支和轻量分支之间的隐式语义转移。此外，为了统一分散的评估体系，我们构建了一个整体SIRST评估(HSE)指标，在像素级置信度和目标级鲁棒性上进行多阈值积分评估，为公平的模型比较提供了稳定而全面的基础。大量实验表明，配备我们FDEP框架的SIRST检测网络在多个公共数据集上取得了最先进的(SOTA)性能。我们的代码可在https://github.com/YuChuang1205/FDEP-Framework获取。


### 论文摘要

While large-scale visual foundation models (VFMs) exhibit strong generalization across diverse visual domains, their potential for single-frame infrared small target (SIRST) detection remains largely unexplored. To fill this gap, we systematically introduce the frozen representations from VFMs into the SIRST task for the first time and propose a Foundation-Driven Efficient Paradigm (FDEP), which can seamlessly adapt to existing encoder-decoder-based methods and significantly improve accuracy without additional inference overhead. Specifically, a Semantic Alignment Modulation Fusion (SAMF) module is designed to achieve dynamic alignment and deep fusion of the global semantic priors from VFMs with task-specific features. Meanwhile, to avoid the inference time burden introduced by VFMs, we propose a Collaborative Optimization-based Implicit Self-Distillation (CO-ISD) strategy, which enables implicit semantic transfer between the main and lightweight branches through parameter sharing and synchronized backpropagation. In addition, to unify the fragmented evaluation system, we construct a Holistic SIRST Evaluation (HSE) metric that performs multi-threshold integral evaluation at both pixel-level confidence and target-level robustness, providing a stable and comprehensive basis for fair model comparison. Extensive experiments demonstrate that the SIRST detection networks equipped with our FDEP framework achieve state-of-the-art (SOTA) performance on multiple public datasets. Our code is available at https://github.com/YuChuang1205/FDEP-Framework

---

## 78. Everything is Context: Agentic File System Abstraction for Context Engineering

**论文链接:** [http://arxiv.org/abs/2512.05470v1](http://arxiv.org/abs/2512.05470v1)

**作者:** Xiwei Xu, Robert Mao, Quan Bai, Xuewu Gu, Yechao Li, Liming Zhu

**发布时间:** 2025-12-05

**备注:** Submitted

### GPT解析

### 总结

该论文提出了一种基于文件系统抽象的上下文工程方法，用于解决生成式AI系统中的知识管理问题，并通过开源AIGNE框架实现了可验证的上下文工程管道。

### 背景

生成式AI通过引入基础模型作为预训练子系统重塑了软件系统设计，新兴挑战已从模型微调转向上下文工程，即系统如何捕获、构建和治理外部知识、记忆、工具和人类输入以实现可信推理。

### 目的

提出一种受Unix'一切皆文件'概念启发的文件系统抽象，为上下文工程提供持久的、受治理的基础设施，以管理异构上下文工件并提高系统的可追溯性和问责制。

### 方法

在开源AIGNE框架中实现文件系统抽象，通过统一挂载、元数据和访问控制来管理上下文工件；构建了包含上下文构建器、加载器和评估器的可验证上下文工程管道，在令牌限制下组装、交付和验证上下文。

### 主要发现

随着GenAI成为决策支持的积极参与者，人类在作为策展人、验证者和共同推理者方面发挥着核心作用；所提出的架构为负责任和以人为中心的AI协作建立了可重用的基础。

### 结论

通过具有记忆的代理和基于MCP的GitHub助手两个示例展示了该架构的应用价值；在AIGNE框架中的实现表明该方法可支持开发者和工业环境中的可验证、可维护和行业就绪的GenAI系统。

### 翻译

生成式AI(GenAI)通过引入基础模型作为预训练子系统重塑了软件系统设计，重新定义了架构和操作。新兴挑战不再是模型微调，而是上下文工程——系统如何捕获、构建和治理外部知识、记忆、工具和人类输入以实现可信推理。现有的提示工程、检索增强生成(RAG)和工具集成等实践仍然碎片化，产生临时性工件，限制了可追溯性和问责制。本文提出了一个受Unix'一切皆文件'概念启发的文件系统抽象用于上下文工程。该抽象通过统一挂载、元数据和访问控制，为管理异构上下文工件提供了持久的、受治理的基础设施。在开源AIGNE框架中实现的架构，实现了包含上下文构建器、加载器和评估器的可验证上下文工程管道，在令牌限制下组装、交付和验证上下文。随着GenAI成为决策支持的积极参与者，人类在作为策展人、验证者和共同推理者方面发挥着核心作用。所提出的架构为负责任和以人为中心的AI协作建立了可重用的基础，通过两个示例进行了演示：一个具有记忆的代理和一个基于MCP的GitHub助手。在AIGNE框架中的实现展示了如何在该架构中操作化，支持开发者和工业环境中的可验证、可维护和行业就绪的GenAI系统。


### 论文摘要

Generative AI (GenAI) has reshaped software system design by introducing foundation models as pre-trained subsystems that redefine architectures and operations. The emerging challenge is no longer model fine-tuning but context engineering-how systems capture, structure, and govern external knowledge, memory, tools, and human input to enable trustworthy reasoning. Existing practices such as prompt engineering, retrieval-augmented generation (RAG), and tool integration remain fragmented, producing transient artefacts that limit traceability and accountability. This paper proposes a file-system abstraction for context engineering, inspired by the Unix notion that 'everything is a file'. The abstraction offers a persistent, governed infrastructure for managing heterogeneous context artefacts through uniform mounting, metadata, and access control. Implemented within the open-source AIGNE framework, the architecture realises a verifiable context-engineering pipeline, comprising the Context Constructor, Loader, and Evaluator, that assembles, delivers, and validates context under token constraints. As GenAI becomes an active collaborator in decision support, humans play a central role as curators, verifiers, and co-reasoners. The proposed architecture establishes a reusable foundation for accountable and human-centred AI co-work, demonstrated through two exemplars: an agent with memory and an MCP-based GitHub assistant. The implementation within the AIGNE framework demonstrates how the architecture can be operationalised in developer and industrial settings, supporting verifiable, maintainable, and industry-ready GenAI systems.

---

## 79. SMamDiff: Spatial Mamba for Stochastic Human Motion Prediction

**论文链接:** [http://arxiv.org/abs/2512.00355v1](http://arxiv.org/abs/2512.00355v1)

**作者:** Junqiao Fan, Pengfei Liu, Haocong Rao

**发布时间:** 2025-11-29

### GPT解析

### 总结

本研究提出了一种名为SMamDiff的空间Mamba扩散模型，通过两种创新设计确保在单阶段扩散模型中实现时空一致性，用于人类运动预测。该模型在单阶段概率HMP方法中实现了最先进的结果，同时比多阶段基线使用更少的延迟和内存。

### 背景

随着智能房间感知和服务机器人的广泛部署，人类运动预测(HMP)对于安全、主动的辅助至关重要。然而，现有的HMP方法要么产生忽略不确定性的单一确定性预测，要么依赖牺牲运动可能性的概率模型。扩散模型改善了准确性和多样性之间的权衡，但通常依赖于对边缘部署成本高昂的多阶段流水线。

### 目的

本研究旨在确保在单阶段扩散模型中保持时空一致性，用于人类运动预测。

### 方法

作者提出了SMamDiff，一种基于空间Mamba的扩散模型，包含两种新颖设计：(1)残差DCT运动编码，在时域DCT之前减去最后观察到的姿态，减少第一个DC分量主导作用，突出信息性更高的频率线索，使模型学习关节如何移动而不是它们的位置；(2)火柴人绘制空间Mamba模块，按有序的、逐个关节的方式处理关节，使后来的关节依赖于先前的关节，以诱导长程、跨关节依赖关系。

### 主要发现

在Human3.6M和HumanEva数据集上，这些一致性机制在单阶段概率HMP方法中实现了最先进的结果，同时比多阶段扩散基线使用更少的延迟和内存。

### 结论

SMamDiff模型通过创新的残差DCT运动编码和火柴人绘制空间Mamba模块，成功地在单阶段扩散模型中实现了时空一致性，为人类运动预测提供了高效且准确的解决方案。

### 翻译

随着智能房间感知和服务机器人的广泛部署，人类运动预测对于安全、主动的辅助至关重要。然而，许多现有的HMP方法要么产生忽略不确定性的单一确定性预测，要么依赖牺牲运动可能性的概率模型。扩散模型改善了准确性和多样性之间的权衡，但通常依赖于对边缘部署成本高昂的多阶段流水线。本研究聚焦于如何在HMP的单阶段扩散模型中确保时空一致性。我们引入了SMamDiff，一种基于空间Mamba的扩散模型，包含两种新颖设计：(i)残差DCT运动编码，在时域DCT之前减去最后观察到的姿态，减少第一个DC分量主导作用，突出信息性更高的频率线索，使模型学习关节如何移动而不是它们的位置；(ii)火柴人绘制空间Mamba模块，按有序的、逐个关节的方式处理关节，使后来的关节依赖于先前的关节，以诱导长程、跨关节依赖关系。在Human3.6M和HumanEva上，这些一致性机制在单阶段概率HMP方法中实现了最先进的结果，同时比多阶段扩散基线使用更少的延迟和内存。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决人类运动预测(HMP)中的时空一致性问题，特别是在单一阶段扩散模型中如何保持这种一致性。这个问题在现实世界中非常重要，因为随着智能房间感知和服务机器人的广泛部署，准确预测人类运动对于安全、主动的辅助至关重要。现有的方法要么产生单一确定性预测(忽略不确定性)，要么使用牺牲运动可能性的概率模型，而扩散模型虽然能平衡准确性和多样性，但通常依赖多阶段管道，对边缘部署成本太高。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：确定性方法忽略不确定性，VAE和GAN方法产生不现实的运动，而多阶段扩散模型部署效率低。作者注意到在频率域中，人体运动表示的光谱高度不平衡，DC分量主导导致模型学习平均姿势而非关节动态。设计上借鉴了扩散模型、DCT变换和GCN等现有工作，创新性地提出残差-DCT运动编码(减去最后观察姿势减少DC分量主导)和火柴人绘制的空间Mamba模块(按顺序处理关节建立依赖关系)。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在残差-DCT域中进行单阶段端到端扩散，使用类似火柴人绘制的空间Mamba模块建模关节关系，并通过K多样性训练目标提高样本多样性。整体流程：1)输入历史运动序列；2)构建残差序列并应用DCT变换；3)在残差-DCT域执行扩散过程(训练学习分布，推理从噪声去噪)；4)应用逆变换恢复时域预测；5)空间Mamba按顺序处理关节确保一致性；6)使用K多样性训练目标优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三个：1)残差-DCT运动编码：减去最后观察姿势减少DC分量主导，突出高频运动线索；2)火柴人绘制的空间Mamba模块：按顺序处理关节，建立跨关节依赖关系；3)K多样性训练目标：产生多个并行预测，缓解模式崩溃。相比之前工作，SMamDiff是单阶段端到端的(减少延迟内存)，残差-DCT使时间运动更自然，空间Mamba能捕获远程关节依赖，在保持时空一致性的同时实现了更高精度和真实感。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SMamDiff通过残差-DCT表示和火柴人绘制的空间Mamba模块，在单阶段扩散模型中实现了高精度、高真实感的人类运动预测，同时保持了时空一致性和部署效率。'}


### 论文摘要

With intelligent room-side sensing and service robots widely deployed, human motion prediction (HMP) is essential for safe, proactive assistance. However, many existing HMP methods either produce a single, deterministic forecast that ignores uncertainty or rely on probabilistic models that sacrifice kinematic plausibility. Diffusion models improve the accuracy-diversity trade-off but often depend on multi-stage pipelines that are costly for edge deployment. This work focuses on how to ensure spatial-temporal coherence within a single-stage diffusion model for HMP. We introduce SMamDiff, a Spatial Mamba-based Diffusion model with two novel designs: (i) a residual-DCT motion encoding that subtracts the last observed pose before a temporal DCT, reducing the first DC component ($f=0$) dominance and highlighting informative higher-frequency cues so the model learns how joints move rather than where they are; and (ii) a stickman-drawing spatial-mamba module that processes joints in an ordered, joint-by-joint manner, making later joints condition on earlier ones to induce long-range, cross-joint dependencies. On Human3.6M and HumanEva, these coherence mechanisms deliver state-of-the-art results among single-stage probabilistic HMP methods while using less latency and memory than multi-stage diffusion baselines.

---

## 80. mmPred: Radar-based Human Motion Prediction in the Dark

**论文链接:** [http://arxiv.org/abs/2512.00345v1](http://arxiv.org/abs/2512.00345v1)

**作者:** Junqiao Fan, Haocong Rao, Jiarui Zhang, Jianfei Yang, Lihua Xie

**发布时间:** 2025-11-29

**备注:** This paper is accepted by AAAI-2026

### GPT解析

### 总结

本研究首次将毫米波雷达引入人体运动预测领域，提出了一种名为mmPred的扩散框架，解决了雷达信号中的镜面反射和多径效应问题，通过双域历史运动表示和全局骨架关系Transformer实现了高精度的人体运动预测。

### 背景

现有基于RGB-D相机的人体运动预测方法对光照条件敏感且存在隐私问题，限制了其在消防和医疗等现实世界应用中的使用。

### 目的

开发一种基于毫米波雷达的人体运动预测方法，以提高鲁棒性和保护隐私，同时解决雷达信号中的噪声和时序不一致问题。

### 方法

提出mmPred扩散框架，引入双域历史运动表示，包含时域姿态细化分支学习精细细节和频域主导运动分支捕捉全局趋势，并设计全局骨架关系Transformer作为扩散骨干建模关节间合作。

### 主要发现

mmPred在mmBody和mm-Fi数据集上分别比现有方法提高了8.6%和22%，达到了最先进的性能。

### 结论

毫米波雷达结合mmPred框架为人体运动预测提供了一种鲁棒且隐私保护的解决方案，在现实世界应用中具有巨大潜力。

### 翻译

现有基于RGB-D相机的人体运动预测方法对光照条件敏感且引发隐私问题，限制了其在消防和医疗等现实世界应用中的使用。受毫米波雷达的鲁棒性和隐私保护特性启发，本研究首次将雷达引入HMP作为一种新型传感模态。然而，雷达信号常受镜面反射和多径效应影响，导致测量结果噪声大且时间不一致，如身体部位漏检。为解决这些雷达特有问题，我们提出了mmPred，这是首个针对雷达HMP的扩散框架。mmPred引入双域历史运动表示来指导生成过程，结合时域姿态细化(TPR)分支学习精细细节和频域主导运动(FDM)分支捕捉全局运动趋势并抑制帧级不一致。此外，我们设计了全局骨架关系Transformer(GST)作为扩散骨干，建模全局关节间合作，使受损关节能动态聚合其他关节的信息。大量实验表明，mmPred达到了最先进的性能，在mmBody和mm-Fi上分别比现有方法高出8.6%和22%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于毫米波雷达的人体运动预测问题，特别是在黑暗等不利环境下的应用。这个问题很重要，因为现有基于RGB-D相机的方法对光照条件敏感且存在隐私问题，限制了它们在消防、医疗保健等现实场景中的应用。毫米波雷达能穿透烟雾、不受光照影响且保护隐私，为这些挑战场景提供了理想解决方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了毫米波雷达的优势（不受光照影响、保护隐私）和挑战（信号噪声大、时间不一致），然后借鉴了扩散模型在生成真实人体运动方面的潜力。设计上，作者引入了双域历史运动表示（时域姿态精炼TPR和频域主导运动FDM）来克服雷达信号问题，并设计了全局骨架关系Transformer（GST）作为扩散骨干。作者借鉴了雷达点云处理技术、扩散模型在人体运动生成中的应用、频域分析（如DCT变换）以及Transformer架构在建模关节关系方面的应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用毫米波雷达在不利环境下的优势，结合扩散模型能力，通过双域表示和全局骨架关系Transformer实现更准确的人体运动预测。整体流程包括：1) 双域历史运动估计（TPR分支处理时域姿态细节，FDM分支提取频域主导运动趋势）；2) 跨域历史融合生成条件嵌入；3) 使用GST作为扩散骨干在频域进行未来姿态预测；4) 分阶段训练（先训练FDM，再训练扩散模型）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个专门针对雷达的人体运动预测框架；2) 双域历史运动表示（结合时域细节和频域趋势）；3) 全局骨架关系Transformer（GST）作为扩散骨干，建模关节间关系；4) 频域扩散模型处理雷达噪声。相比之前工作，mmPred使用毫米波雷达而非RGB-D相机作为输入，专门处理雷达特有的噪声和时序不一致问题，采用双域表示而非仅时域操作，以及设计了专门的GST架构处理关节间关系。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'mmPred首次将毫米波雷达与扩散模型结合，通过双域历史表示和全局骨架关系Transformer，实现了在黑暗等不利环境下更准确、更鲁棒的人体运动预测。'}


### 论文摘要

Existing Human Motion Prediction (HMP) methods based on RGB-D cameras are sensitive to lighting conditions and raise privacy concerns, limiting their real-world applications such as firefighting and healthcare. Motivated by the robustness and privacy-preserving nature of millimeter-wave (mmWave) radar, this work introduces radar as a novel sensing modality for HMP, for the first time. Nevertheless, radar signals often suffer from specular reflections and multipath effects, resulting in noisy and temporally inconsistent measurements, such as body-part miss-detection. To address these radar-specific artifacts, we propose mmPred, the first diffusion-based framework tailored for radar-based HMP. mmPred introduces a dual-domain historical motion representation to guide the generation process, combining a Time-domain Pose Refinement (TPR) branch for learning fine-grained details and a Frequency-domain Dominant Motion (FDM) branch for capturing global motion trends and suppressing frame-level inconsistency. Furthermore, we design a Global Skeleton-relational Transformer (GST) as the diffusion backbone to model global inter-joint cooperation, enabling corrupted joints to dynamically aggregate information from others. Extensive experiments show that mmPred achieves state-of-the-art performance, outperforming existing methods by 8.6% on mmBody and 22% on mm-Fi.

---

## 81. MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction

**论文链接:** [http://arxiv.org/abs/2511.22181v1](http://arxiv.org/abs/2511.22181v1)

**作者:** Maitrayee Keskar, Mohan Trivedi, Ross Greer

**发布时间:** 2025-11-27

**备注:** 8 pages, 3 figures, 4 tables

### GPT解析

### 总结

论文提出了一种基于视觉的自动驾驶轨迹规划方法MTR-VP，使用ViT编码器从图像和过去车辆状态生成上下文嵌入，并通过交叉注意力结合意图信息。在Waymo数据集上的评估表明，多轨迹预测优于单轨迹预测，但Transformer方法在有效结合视觉和运动特征方面存在挑战。

### 背景

自动驾驶需要准确预测和规划车辆轨迹。现有的MTR方法通过定位代理意图和迭代优化运动来提供多模态轨迹预测的基础，但主要依赖地图特征而非视觉特征。

### 目的

开发一种能够有效结合视觉信息和运动特征的轨迹规划方法，替代基于地图的特征，提高自动驾驶系统的性能。

### 方法

提出MTR-VP方法，使用ViT编码器处理原始图像和过去运动状态生成上下文嵌入，通过交叉注意力机制结合意图信息，而非使用可学习意图查询。在Waymo端到端驾驶数据集上进行评估，使用消融研究分析架构。

### 主要发现

基于Transformer的方法在有效结合视觉特征和运动特征以产生有用的场景上下文嵌入方面并不有效；即使使用CLIP和DINOv2等基础模型增强意图嵌入，这一问题仍然存在；预测多个未来的分布而非单个未来轨迹可以提高规划性能。

### 结论

虽然基于Transformer的方法在多轨迹预测方面表现良好，但在有效融合视觉和运动特征方面仍有改进空间。未来的研究应关注如何更好地结合这两种模态的信息。

### 翻译

我们提出了一种用于自动驾驶的轨迹规划方法，学习基于图像的上下文嵌入，这些嵌入与运动预测框架和基于规划的意图输入相一致。在我们的方法中，ViT编码器将原始图像和过去的运动状态作为输入，并被训练生成上下文嵌入，受最近MTR（运动Transformer）编码器生成的嵌入启发，有效地用学习到的视觉表示替代基于地图的特征。MTR通过定位代理意图和通过运动查询对迭代优化运动为多模态轨迹预测提供了坚实基础；我们将我们的方法命名为MTR-VP（基于视觉的规划运动Transformer），而不是使用MTR解码器中的可学习意图查询，我们在意图和上下文嵌入之间使用交叉注意力，这些上下文嵌入反映了从驾驶场景和过去车辆状态编码的信息组合。我们在Waymo端到端驾驶数据集上评估了我们的方法，该数据集需要使用先前的摄像头图像、代理姿态历史和路由目标来预测代理在鸟瞰图坐标中未来5秒的轨迹。我们通过消融研究分析了我们的架构，移除了输入图像和多轨迹输出。我们的结果表明，用于结合视觉特征和运动特征（如过去轨迹特征）的基于Transformer的方法在将两种模式结合以产生有用的场景上下文嵌入方面并不有效，即使当意图嵌入通过CLIP和DINOv2的场景上下文基础模型表示增强时也是如此，但预测多个未来的分布而不是单个未来轨迹可以提高规划性能。


### 论文摘要

We present a method for trajectory planning for autonomous driving, learning image-based context embeddings that align with motion prediction frameworks and planning-based intention input. Within our method, a ViT encoder takes raw images and past kinematic state as input and is trained to produce context embeddings, drawing inspiration from those generated by the recent MTR (Motion Transformer) encoder, effectively substituting map-based features with learned visual representations. MTR provides a strong foundation for multimodal trajectory prediction by localizing agent intent and refining motion iteratively via motion query pairs; we name our approach MTR-VP (Motion Transformer for Vision-based Planning), and instead of the learnable intention queries used in the MTR decoder, we use cross attention on the intent and the context embeddings, which reflect a combination of information encoded from the driving scene and past vehicle states. We evaluate our methods on the Waymo End-to-End Driving Dataset, which requires predicting the agent's future 5-second trajectory in bird's-eye-view coordinates using prior camera images, agent pose history, and routing goals. We analyze our architecture using ablation studies, removing input images and multiple trajectory output. Our results suggest that transformer-based methods that are used to combine the visual features along with the kinetic features such as the past trajectory features are not effective at combining both modes to produce useful scene context embeddings, even when intention embeddings are augmented with foundation-model representations of scene context from CLIP and DINOv2, but that predicting a distribution over multiple futures instead of a single future trajectory boosts planning performance.

---

## 82. Optimization of Deep Learning Models for Dynamic Market Behavior Prediction

**论文链接:** [http://arxiv.org/abs/2511.19090v1](http://arxiv.org/abs/2511.19090v1)

**作者:** Shenghan Zhao, Yuzhen Lin, Ximeng Yang, Qiaochu Lu, Haozhong Xue, Gaozhe Jiang

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究提出了一种混合序列模型用于电子商务交易中的多时段需求预测，结合了多尺度时间卷积、门控循环模块和时间感知自注意力机制，在多个基准模型上显示出更好的准确性和鲁棒性。

### 背景

金融科技的兴起见证了深度学习模型在预测消费者行为方面的应用激增，这一趋势在增强贷款策略和提高市场效率方面显示出巨大潜力。

### 目的

研究电子商务交易中的多时段需求预测，专注于零售市场行为，明确预测目标：每个SKU的日需求（或收入）预测，预测时段为H=1、7、14天。

### 方法

提出一种混合序列模型，结合多尺度时间卷积、门控循环模块和时间感知自注意力机制。使用标准回归损失训练，在MAE、RMSE、sMAPE、MASE和Theil's U_2指标下评估，采用严格时间分割防止信息泄露。与ARIMA/Prophet、LSTM/GRU、LightGBM和Transformer预测器（TFT、Informer、Autoformer、N-BEATS）进行基准比较。

### 主要发现

结果显示，与基准模型相比，所提出的模型在准确性和鲁棒性方面都有持续的提升，特别是在高峰/节假日期间。通过消融实验和统计显著性测试确保了改进的可靠性。

### 结论

该研究提出了一种有效的混合序列模型用于电子商务交易中的多时段需求预测，并在多个基准上显示出优越的性能。研究还提供了实现细节以促进结果的可复现性。

### 翻译

金融科技的兴起见证了深度学习模型在预测消费者行为方面的应用激增，这一趋势在增强贷款策略和提高市场效率方面显示出巨大潜力。我们使用UCI Online Retail II数据集研究了电子商务交易中的多时段需求预测。与之前混合了金融贷款叙述和零售数据的版本不同，我们专注于零售市场行为，并明确定义了预测目标：每个SKU的日需求（或收入）预测，预测时段为H=1、7、14天。我们提出了一种混合序列模型，结合了多尺度时间卷积、门控循环模块和时间感知自注意力。模型使用标准回归损失进行训练，并在MAE、RMSE、sMAPE、MASE和Theil's U_2指标下评估，采用严格的时间分割以防止信息泄露。我们与ARIMA/Prophet、LSTM/GRU、LightGBM和最先进的Transformer预测器（TFT、Informer、Autoformer、N-BEATS）进行了基准比较。结果显示，与基准模型相比，所提出的模型在准确性和鲁棒性方面都有持续的提升，特别是在高峰/节假日期间。我们还提供了消融实验和统计显著性测试以确保改进的可靠性，并发布了实现细节以促进可复现性。


### 论文摘要

The advent of financial technology has witnessed a surge in the utilization of deep learning models to anticipate consumer conduct, a trend that has demonstrated considerable potential in enhancing lending strategies and bolstering market efficiency. We study multi-horizon demand forecasting on e-commerce transactions using the UCI Online Retail II dataset. Unlike prior versions of this manuscript that mixed financial-loan narratives with retail data, we focus exclusively on retail market behavior and define a clear prediction target: per SKU daily demand (or revenue) for horizons H=1,7,14. We present a hybrid sequence model that combines multi-scale temporal convolutions, a gated recurrent module, and time-aware self-attention. The model is trained with standard regression losses and evaluated under MAE, RMSE, sMAPE, MASE, and Theil's U_2 with strict time-based splits to prevent leakage. We benchmark against ARIMA/Prophet, LSTM/GRU, LightGBM, and state-of-the-art Transformer forecasters (TFT, Informer, Autoformer, N-BEATS). Results show consistent accuracy gains and improved robustness on peak/holiday periods. We further provide ablations and statistical significance tests to ensure the reliability of improvements, and we release implementation details to facilitate reproducibility.

---

## 83. Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj

**论文链接:** [http://arxiv.org/abs/2511.18248v1](http://arxiv.org/abs/2511.18248v1)

**作者:** Wei Zhen Teoh

**发布时间:** 2025-11-23

**备注:** 9 pages, 3 figures, accepted to the AI4TS Workshop at AAAI 2026

### GPT解析

### 总结

本文提出了一种名为CausalTraj的基于时间和因果的似然模型，用于联合预测多个交互智能体的轨迹，解决了现有模型在评估和优化中忽视联合合理性的问题。

### 背景

联合预测多个交互智能体的轨迹是体育分析和其他涉及复杂群体动态领域的核心挑战。准确预测能够实现逼真的模拟和对比赛演变的战略理解。

### 目的

开发一个能够生成共同可能的多智能体轨迹预测的模型，并改进评估方法以更好地衡量模型的集体建模能力。

### 方法

提出CausalTraj，一个基于时间的因果、似然模型，并强调使用联合指标来评估跨智能体的联合准确性，而不仅仅是每个智能体的单独准确性。

### 主要发现

在NBA SportVU、Basketball-U和Football-U数据集上评估，CausalTraj实现了具有竞争力的每个智能体准确性，并在联合指标上取得了最佳记录的结果，同时产生了质量上连贯且逼真的比赛演变。

### 结论

CausalTraj模型在多智能体轨迹预测方面表现优异，特别是在联合预测方面，能够生成连贯且逼真的比赛演变场景。

### 翻译

联合预测多个交互智能体的轨迹是体育分析和其他涉及复杂群体动态领域的核心挑战。准确预测能够实现逼真的模拟和对比赛演变的战略理解。大多数现有模型仅基于每个智能体的准确性指标进行评估，这些指标独立评估每个智能体的最佳预测。然而，这些指标忽略了模型是否学习到哪些预测轨迹能够共同形成合理的多智能体未来。许多最先进模型主要基于这些指标进行设计和优化，导致它们在联合预测方面表现不佳，也无法在团队体育中生成连贯、可解释的多智能体场景。我们提出了CausalTraj，一个基于时间的因果、似然模型，旨在生成共同可能的多智能体轨迹预测。为了更好地评估集体建模能力，我们强调联合指标，这些指标衡量在最佳生成的场景样本中跨智能体的联合准确性。在NBA SportVU、Basketball-U和Football-U数据集上评估，CausalTraj实现了具有竞争力的每个智能体准确性，并在联合指标上取得了最佳记录的结果，同时产生了质量上连贯且逼真的比赛演变。


### 论文摘要

Jointly forecasting trajectories of multiple interacting agents is a core challenge in sports analytics and other domains involving complex group dynamics. Accurate prediction enables realistic simulation and strategic understanding of gameplay evolution. Most existing models are evaluated solely on per-agent accuracy metrics (minADE, minFDE), which assess each agent independently on its best-of-k prediction. However these metrics overlook whether the model learns which predicted trajectories can jointly form a plausible multi-agent future. Many state-of-the-art models are designed and optimized primarily based on these metrics. As a result, they may underperform on joint predictions and also fail to generate coherent, interpretable multi-agent scenarios in team sports. We propose CausalTraj, a temporally causal, likelihood-based model that is built to generate jointly probable multi-agent trajectory forecasts. To better assess collective modeling capability, we emphasize joint metrics (minJADE, minJFDE) that measure joint accuracy across agents within the best generated scenario sample. Evaluated on the NBA SportVU, Basketball-U, and Football-U datasets, CausalTraj achieves competitive per-agent accuracy and the best recorded results on joint metrics, while yielding qualitatively coherent and realistic gameplay evolutions.

---

## 84. SM2ITH: Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control

**论文链接:** [http://arxiv.org/abs/2511.17798v1](http://arxiv.org/abs/2511.17798v1)

**作者:** Francesco D'Orazio, Sepehr Samavi, Xintong Du, Siqi Zhou, Giuseppe Oriolo, Angela P. Schoellig

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究提出了SM$^2$ITH框架，通过双层优化将分层任务模型预测控制与交互式人类运动预测相结合，使移动操作机器人能够在动态人类中心环境中安全高效地执行任务。

### 背景

现有的基于优化的方法如HTMPC虽然能高效执行多任务并保持严格任务优先级，但主要应用于静态或结构化场景，难以适应动态人类中心环境。

### 目的

开发一种能够预测人类对机器人行为反应的框架，使移动操作机器人能够在动态人类中心环境中安全高效地工作。

### 方法

提出SM$^2$ITH框架，结合HTMPC与交互式人类运动预测，通过双层优化同时考虑机器人和人类动力学，实现安全高效的协调。

### 主要发现

交互式预测使移动操作机器人能够实现安全高效的协调，表现优于依赖加权目标或开环人类模型的基线方法。

### 结论

所提出的SM$^2$ITH框架能够有效处理动态人类中心环境中的移动操作任务，为机器人与人类的安全协作提供了新思路。

### 翻译

移动操作机器人在以人为中心的环境中执行复杂的导航和操作任务序列。虽然最近的基于优化的方法如分层任务模型预测控制能够高效执行多任务并保持严格的任务优先级，但它们目前主要应用于静态或结构化场景。将这些方法扩展到动态的以人为中心的环境中需要能够预测人类对机器人行为反应的预测模型。本文提出了通过任务分层双层模型预测控制实现交互式人类预测的安全移动操作SM$^2$ITH，这是一个结合HTMPC与交互式人类运动预测的统一框架，通过双层优化同时考虑机器人和人类动力学。该框架在两种不同的移动操作机器人Stretch 3和Ridgeback-UR10上进行了验证，涵盖三种实验设置：具有不同导航和操作优先级的交付任务、具有不同人类运动预测模型的顺序拾放任务、以及涉及对抗性人类行为的交互。我们的结果突显了交互式预测如何实现安全高效的协调，优于依赖加权目标或开环人类模型的基线方法。


### 论文摘要

Mobile manipulators are designed to perform complex sequences of navigation and manipulation tasks in human-centered environments. While recent optimization-based methods such as Hierarchical Task Model Predictive Control (HTMPC) enable efficient multitask execution with strict task priorities, they have so far been applied mainly to static or structured scenarios. Extending these approaches to dynamic human-centered environments requires predictive models that capture how humans react to the actions of the robot. This work introduces Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control (SM$^2$ITH), a unified framework that combines HTMPC with interactive human motion prediction through bilevel optimization that jointly accounts for robot and human dynamics. The framework is validated on two different mobile manipulators, the Stretch 3 and the Ridgeback-UR10, across three experimental settings: (i) delivery tasks with different navigation and manipulation priorities, (ii) sequential pick-and-place tasks with different human motion prediction models, and (iii) interactions involving adversarial human behavior. Our results highlight how interactive prediction enables safe and efficient coordination, outperforming baselines that rely on weighted objectives or open-loop human models.

---

## 85. Hidden markov model to predict tourists visited place

**论文链接:** [http://arxiv.org/abs/2511.19465v1](http://arxiv.org/abs/2511.19465v1)

**作者:** Theo Demessance, Chongke Bi, Sonia Djebali, Guillaume Guerard

**发布时间:** 2025-11-21

**DOI:** 10.1109/MDM52706.2021.00041

### GPT解析

### 总结

本文提出了一种基于社交网络数据分析的方法，通过机器学习语法推断算法来理解和学习游客移动，并预测其未来行动。该方法产生的隐马尔可夫模型灵活且可用新数据编辑，在巴黎的案例研究中证明了其有效性。

### 背景

社交网络正成为分析游客行为的流行方式，游客在旅行期间分享评论和照片产生的海量数据为建模其旅程和分析行为提供了可能。

### 目的

预测游客的下一个行动，以理解旅游需求和改进决策支持，从而提升旅游营销效果。

### 方法

基于社交网络数据分析，使用机器学习语法推断算法，将算法适应到大数据环境中，生成表示游客移动的隐马尔可夫模型。

### 主要发现

成功将语法推断算法适应到大数据环境中；产生的隐马尔可夫模型能够表示游客移动，且模型灵活、可用新数据编辑。

### 结论

通过在法国巴黎的案例研究，证明了所提出方法论的有效性。

### 翻译

如今，社交网络正成为分析游客行为的流行方式，这得益于游客在这些网络上停留时留下的数字痕迹。游客在旅行期间倾向于分享评论和照片，产生了大量数据，这使得能够建模他们的旅程并分析他们的行为。预测游客的下一个行动在旅游营销中起着关键作用，有助于理解需求和改进决策支持。在本文中，我们提出了一种基于社交网络数据分析的方法，以理解和学习游客的移动，并预测未来的行动。该方法依赖于机器学习语法推断算法。本文的主要贡献是将语法推断算法适应到大数据环境中。我们的方法产生了一个表示一组游客移动的隐马尔可夫模型。隐马尔可夫模型灵活且可以用新数据编辑。选择法国首都巴黎来证明所提出方法的有效性。


### 论文摘要

Nowadays, social networks are becoming a popular way of analyzing tourist behavior, thanks to the digital traces left by travelers during their stays on these networks. The massive amount of data generated; by the propensity of tourists to share comments and photos during their trip; makes it possible to model their journeys and analyze their behavior. Predicting the next movement of tourists plays a key role in tourism marketing to understand demand and improve decision support. In this paper, we propose a method to understand and to learn tourists' movements based on social network data analysis to predict future movements. The method relies on a machine learning grammatical inference algorithm. A major contribution in this paper is to adapt the grammatical inference algorithm to the context of big data. Our method produces a hidden Markov model representing the movements of a group of tourists. The hidden Markov model is flexible and editable with new data. The capital city of France, Paris is selected to demonstrate the efficiency of the proposed methodology.

---

## 86. RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis

**论文链接:** [http://arxiv.org/abs/2511.17045v2](http://arxiv.org/abs/2511.17045v2)

**作者:** Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan Hou, Zhihang Zhong, Xiao Sun

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026 (Oral)

### GPT解析

### 总结

RacketVision是一个新的数据集和基准，用于推进体育分析中的计算机视觉研究，涵盖乒乓球、网球和羽毛球。该数据集首次提供大规模、细粒度的球拍姿态标注和传统球位置标注，支持三个相互关联的任务：细粒度球跟踪、关节式球拍姿态估计和预测性球轨迹预测。

### 背景

在体育分析领域，特别是涉及球拍类运动的计算机视觉研究需要更精细的数据支持。现有数据集可能缺乏对球拍姿态的细粒度标注，限制了复杂人体-物体交互的研究。

### 目的

创建一个包含乒乓球、网球和羽毛球数据的新数据集；提供大规模、细粒度的球拍姿态标注和传统球位置标注；支持三个相互关联的研究任务；为动态物体跟踪、条件运动预测和体育多模态分析提供研究资源。

### 方法

构建RacketVision数据集，包含乒乓球、网球和羽毛球的大规模细粒度标注；评估现有基线方法；实现多模态融合，特别是使用CrossAttention机制来整合球拍姿态特征。

### 主要发现

简单地将球拍姿态特征连接起来会降低性能；CrossAttention机制对于释放球拍姿态特征的价值至关重要；使用CrossAttention机制的轨迹预测结果超越了强大的单模态基线。

### 结论

RacketVision提供了一个多功能资源和未来研究的强有力起点，特别是在动态物体跟踪、条件运动预测和体育多模态分析领域。

### 翻译

我们介绍了RacketVision，这是一个推进体育分析中计算机视觉的新型数据集和基准，涵盖乒乓球、网球和羽毛球。该数据集首次提供大规模、细粒度的球拍姿态标注以及传统的球位置标注，使研究复杂的人体-物体交互成为可能。它旨在解决三个相互关联的任务：细粒度球跟踪、关节式球拍姿态估计和预测性球轨迹预测。我们对已建立的基线评估揭示了多模态融合的关键见解：虽然简单连接球拍姿态特征会降低性能，但CrossAttention机制对于释放其价值至关重要，导致轨迹预测结果超越了强大的单模态基线。RacketVision为未来在动态物体跟踪、条件运动预测和体育多模态分析方面的研究提供了多功能资源和强有力的起点。项目页面位于https://github.com/OrcustD/RacketVision


### 论文摘要

We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at https://github.com/OrcustD/RacketVision

---

## 87. Lane-Frame Quantum Multimodal Driving Forecasts for the Trajectory of Autonomous Vehicles

**论文链接:** [http://arxiv.org/abs/2511.17675v1](http://arxiv.org/abs/2511.17675v1)

**作者:** Navneet Singh, Shiva Raj Pokhrel

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了一种紧凑的混合量子架构，用于自动驾驶轨迹预测，能够在计算和延迟限制下提供准确且校准的多模态预测结果。该模型通过结合量子注意力编码器、量子前馈堆栈和基于傅里叶的解码器，在Waymo开放运动数据集上实现了优于运动学基线的性能。

### 背景

自动驾驶的轨迹预测需要在严格的计算和延迟约束下提供准确且校准的多模态未来预测，这对传统方法提出了挑战。

### 目的

开发一种紧凑的混合量子架构，使其与道路场景结构保持一致，通过在以自我为中心的车道对齐框架中操作，并预测运动学基线的残差校正来实现准确的轨迹预测。

### 方法

结合了受Transformer启发的量子注意力编码器（9个量子比特）、参数精简的量子前馈堆栈（64层，约1200个可训练角度）和基于傅里叶的解码器，使用浅层纠缠和相位叠加在单次前向传播中生成16个轨迹假设，模态置信度来自潜在频谱。所有电路参数通过同时扰动随机近似（SPSA）进行训练。

### 主要发现

在Waymo开放运动数据集上，模型在预测2.0秒范围内的16个模型中，实现了最小平均位移误差1.94米和最小最终位移误差3.56米，一致性地优于运动学基线，降低了漏报率并具有强召回率。

### 结论

车道框架中的残差学习、截断傅里叶解码、浅层纠缠和基于频谱的排名将容量集中在重要位置，从而在现代自动驾驶基准上的小型浅层量子电路中产生稳定的优化和可靠的多模态预测。

### 翻译

自动驾驶的轨迹预测必须在严格的计算和延迟限制下提供准确且校准的多模态未来预测。我们提出了一种紧凑的混合量子架构，通过在以自我为中心的车道对齐框架中操作，并预测运动学基线的残差校正而不是绝对姿态，使量子归纳偏差与道路场景结构保持一致。该模型结合了受Transformer启发的量子注意力编码器（9个量子比特）、参数精简的量子前馈堆栈（64层，约1200个可训练角度）和基于傅里叶的解码器，使用浅层纠缠和相位叠加在单次前向传播中生成16个轨迹假设，模态置信度来自潜在频谱。所有电路参数均通过同时扰动随机近似（SPSA）进行训练，避免了通过非解析组件的反向传播。在Waymo开放运动数据集中，该模型在预测2.0秒范围内的16个模型中，实现了1.94米的最小平均位移误差和3.56米的最小最终位移误差，一致性地优于运动学基线，降低了漏报率并具有强召回率。消融实验确认，车道框架中的残差学习、截断傅里叶解码、浅层纠缠和基于频谱的排名将容量集中在重要位置，从而在现代自动驾驶基准上的小型浅层量子电路中产生稳定的优化和可靠的多模态预测。


### 论文摘要

Trajectory forecasting for autonomous driving must deliver accurate, calibrated multi-modal futures under tight compute and latency constraints. We propose a compact hybrid quantum architecture that aligns quantum inductive bias with road-scene structure by operating in an ego-centric, lane-aligned frame and predicting residual corrections to a kinematic baseline instead of absolute poses. The model combines a transformer-inspired quantum attention encoder (9 qubits), a parameter-lean quantum feedforward stack (64 layers, ${\sim}1200$ trainable angles), and a Fourier-based decoder that uses shallow entanglement and phase superposition to generate 16 trajectory hypotheses in a single pass, with mode confidences derived from the latent spectrum. All circuit parameters are trained with Simultaneous Perturbation Stochastic Approximation (SPSA), avoiding backpropagation through non-analytic components. In the Waymo Open Motion Dataset, the model achieves minADE (minimum Average Displacement Error) of \SI{1.94}{m} and minFDE (minimum Final Displacement Error) of \SI{3.56}{m} in the $16$ models predicted over the horizon of \SI{2.0}{s}, consistently outperforming a kinematic baseline with reduced miss rates and strong recall. Ablations confirm that residual learning in the lane frame, truncated Fourier decoding, shallow entanglement, and spectrum-based ranking focus capacity where it matters, yielding stable optimization and reliable multi-modal forecasts from small, shallow quantum circuits on a modern autonomous-driving benchmark.

---

## 88. gr-Orbit-Toolkit: A Python-Based Software for Simulating and Visualizing Relativistic Orbits

**论文链接:** [http://arxiv.org/abs/2511.19442v1](http://arxiv.org/abs/2511.19442v1)

**作者:** Milagros Delgado, Wladimir E. Banda-Barragán

**发布时间:** 2025-11-13

**DOI:** 10.1007/978-3-032-08366-1_20

**备注:** Author's accepted manuscript of a chapter published in Information and Communication Technologies (TICEC 2025 Proceedings), Communications in Computer and Information Science, vol. 2707, Springer Nature Switzerland AG, 2026. The final authenticated version is available at: https://doi.org/10.1007/978-3-032-08366-1_20 15 pages, 4 figures

### GPT解析

### 总结

本文介绍了一个名为gr-orbit-toolkit的Python软件，用于模拟经典和相对论场景下的轨道运动，旨在促进STEM领域的教学和研究。

### 背景

专用模拟软件对STEM领域的教学和研究至关重要；物理教学在使用数字孪生技术时更加有效；现代高级编程语言的发展为物理研究提供了便利。

### 目的

开发一个基于Python的软件来模拟经典和广义相对论场景下的轨道运动。

### 方法

展示经典和相对论轨道加速度的常微分方程，采用后牛顿方法处理相对论部分；通过数值积分这些方程，使用欧拉和龙格-库塔方法模拟小型物体围绕大质量物体的轨道；研究以太阳或黑洞为中心的双体模型样本。

### 主要发现

经典和相对论预测的轨道运动在接近中心大质量物体的史瓦西半径时显著不同；经典力学适用于远离中心大质量物体的轨道运动，而广义相对论适用于强引力场区域；代码能够捕获相对论轨道进动；收敛分析表明该工具包在数值上是稳健的。

### 结论

gr-orbit-toolkit旨在促进广义相对论的教学和研究，在公共代码库中提供了全面的用户和开发者指南。

### 翻译

为科学、技术、工程和数学领域的教学和研究创建专用模拟软件至关重要。当使用数字孪生技术伴随理论课时，物理教学可以更加有效。物理研究因现代高级编程语言的到来而大大受益，这些语言促进了用户友好代码的实现。在这里，我们报告了我们自己基于Python的软件gr-orbit-toolkit，用于模拟经典和广义相对论场景下的轨道。首先，我们展示了经典和相对论轨道加速度的常微分方程。对于后者，我们采用后牛顿方法。其次，我们描述了我们的算法，该算法通过数值积分这些ODEs，使用欧拉和龙格-库塔方法模拟小型物体围绕大质量物体的轨道。然后，我们研究了一系列以太阳或黑洞为中心的双体模型样本。我们的模拟证实，经典和相对论ODEs预测的轨道运动在接近中心大质量物体的史瓦西半径时截然不同。经典力学解释了远离中心大质量物体的轨道运动，但需要广义相对论来研究靠近大质量物体的物体运动。我们对不同偏心率物体的研究表明，我们的代码捕获了相对论轨道进动。我们的收敛分析表明该工具包在数值上是稳健的。我们的gr-orbit-toolkit旨在促进广义相对论的教学和研究，因此在公共代码库中提供了全面的用户和开发者指南。


### 论文摘要

Creating software dedicated to simulation is essential for teaching and research in Science, Technology, Engineering, and Mathematics (STEM). Physics lecturing can be more effective when digital twins are used to accompany theory classes. Research in physics has greatly benefited from the advent of modern, high-level programming languages, which facilitate the implementation of user-friendly code. Here, we report our own Python-based software, the gr-orbit-toolkit, to simulate orbits in classical and general relativistic scenarios. First, we present the ordinary differential equations (ODEs) for classical and relativistic orbital accelerations. For the latter, we follow a post-Newtonian approach. Second, we describe our algorithm, which numerically integrates these ODEs to simulate the orbits of small-sized objects orbiting around massive bodies by using Euler and Runge-Kutta methods. Then, we study a set of sample two-body models with either the Sun or a black hole in the center. Our simulations confirm that the orbital motions predicted by classical and relativistic ODEs drastically differ for bodies near the Schwarzschild radius of the central massive body. Classical mechanics explains the orbital motion of objects far away from a central massive body, but general relativity is required to study objects moving at close proximity to a massive body, where the gravitational field is strong. Our study on objects with different eccentricities confirms that our code captures relativistic orbital precession. Our convergence analysis shows the toolkit is numerically robust. Our gr-orbit-toolkit aims at facilitating teaching and research in general relativity, so a comprehensive user and developer guide is provided in the public code repository.

---

## 89. VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction

**论文链接:** [http://arxiv.org/abs/2511.10203v1](http://arxiv.org/abs/2511.10203v1)

**作者:** Stephane Da Silva Martins, Emanuel Aldea, Sylvie Le Hégarat-Mascle

**发布时间:** 2025-11-13

**备注:** Paper accepted at WACV 2026

### GPT解析

### 总结

提出VISTA，一种用于多智能体轨迹预测的递归目标条件Transformer，结合跨注意力融合模块、社交令牌注意力机制和成对注意力图，实现了更准确且符合社会规范的轨迹预测。

### 背景

多智能体轨迹预测对在密集、交互环境中运行的自主系统至关重要，但现有方法通常无法同时捕捉智能体的长期目标和精细的社会交互。

### 目的

开发一种能够联合捕捉智能体长期目标和精细社会交互的多智能体轨迹预测方法，以生成更真实的多智能体未来轨迹。

### 方法

VISTA结合了三个关键组件：(i)跨注意力融合模块，整合长期意图与过去运动；(ii)社交令牌注意力机制，灵活建模智能体间的交互；(iii)成对注意力图，在推理时使社会影响模式可解释。

### 主要发现

在MADRAS基准测试和SDD上，VISTA实现了最先进的准确性并显著减少了碰撞率。在MADRAS上，VISTA将强基线的平均碰撞率从2.14%降低到0.03%；在SDD上，它实现了零碰撞，同时提高了ADE、FDE和minFDE指标。

### 结论

VISTA生成的轨迹符合社会规范、具有目标意识且可解释，对安全关键的自主系统具有应用前景。

### 翻译

多智能体轨迹预测对在密集、交互环境中运行的自主系统至关重要。现有方法通常无法同时捕捉智能体的长期目标和精细的社会交互，导致不切实际的多智能体未来预测。我们提出VISTA，一种用于多智能体轨迹预测的递归目标条件Transformer。VISTA结合了(i)整合长期意图与过去运动的跨注意力融合模块，(ii)灵活建模智能体间交互的社交令牌注意力机制，以及(iii)在推理时使社会影响模式可解释的成对注意力图。我们的模型将单智能体目标条件预测转变为连贯的多智能体预测框架。除了标准的位移指标外，我们还评估轨迹碰撞率作为联合现实性的度量。在MADRAS基准测试和SDD上，VISTA实现了最先进的准确性并显著减少了碰撞。在MADRAS上，它将强基线的平均碰撞率从2.14%降低到0.03%，在SDD上实现了零碰撞，同时提高了ADE、FDE和minFDE。这些结果表明VISTA生成的轨迹符合社会规范、具有目标意识且可解释，对安全关键的自主系统具有应用前景。


### 论文摘要

Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems.

---

## 90. Shared Spatial Memory Through Predictive Coding

**论文链接:** [http://arxiv.org/abs/2511.04235v2](http://arxiv.org/abs/2511.04235v2)

**作者:** Zhengru Fang, Yu Guo, Jingjing Wang, Yuang Zhang, Haonan An, Yinhai Wang, Yuguang Fang

**发布时间:** 2025-11-06

**备注:** We have prepared the open-source code and video demonstration pages: 1. Code: github.com/fangzr/SSM-PC 2. Demo: fangzr.github.io/SSM-PC/index.html

### GPT解析

### 总结

本文提出了一种多智能体预测编码框架，通过最小化智能体间的不确定性来解决多智能体系统中的协调挑战，展示了在带宽受限环境下的卓越性能。

### 背景

多智能体系统中构建一致共享空间记忆是一个关键挑战，部分可观察性和有限带宽常常导致协调中的灾难性故障。

### 目的

引入一种多智能体预测编码框架，将协调表述为智能体之间相互不确定性的最小化，并学习高效通信策略。

### 方法

通过信息瓶颈目标促使智能体学习通信对象、内容和时机；采用类似网格细胞的度量作为自定位的内部空间编码；发展带宽高效的通信机制和专门编码伙伴位置的神经群体；利用分层强化学习策略主动探索以减少联合不确定性。

### 主要发现

在Memory-Maze基准测试中，当带宽从128位/步减少到4位/步时，成功率从73.5%平缓下降到64.4%，而全广播基线从67.6%急剧下降到28.6%，表明该方法对带宽限制具有极强的适应性。

### 结论

研究结果为复杂社交表征如何从统一的预测驱动中涌现提供了理论原则和生物 plausible 的基础，促进了集体智能的发展。

### 翻译

构建一致共享的空间记忆是多智能体系统中的一个关键挑战，其中部分可观察性和有限带宽常常导致协调中的灾难性故障。我们引入了一种多智能体预测编码框架，将协调表述为智能体之间相互不确定性的最小化。通过信息瓶颈目标，该框架促使智能体学习不仅要与谁通信、通信什么，还要学习何时。在该框架的基础是一种类似网格细胞的度量，作为自定位的内部空间编码，通过自监督运动预测自发产生。基于这种内部空间编码，智能体逐渐发展出带宽高效的通信机制和专门编码伙伴位置的神经群体 - 海马体社交位置细胞的人工类似物。这些社交表征被进一步用于分层强化学习策略，主动探索以减少联合不确定性。在Memory-Maze基准测试中，我们的方法对带宽限制表现出极强的适应性：当带宽从128位/步减少到4位/步时，成功率从73.5%平缓下降到64.4%，而全广播基线从67.6%崩溃到28.6%。我们的研究结果为复杂社交表征如何从统一的预测驱动中涌现提供了理论原则和生物 plausible 的基础，从而导致了集体智能的产生。


### 论文摘要

Constructing a consistent shared spatial memory is a critical challenge in multi-agent systems, where partial observability and limited bandwidth often lead to catastrophic failures in coordination. We introduce a multi-agent predictive coding framework that formulates coordination as the minimization of mutual uncertainty among agents. Through an information bottleneck objective, this framework prompts agents to learn not only who and what to communicate but also when. At the foundation of this framework lies a grid-cell-like metric as internal spatial coding for self-localization, emerging spontaneously from self-supervised motion prediction. Building upon this internal spatial code, agents gradually develop a bandwidth-efficient communication mechanism and specialized neural populations that encode partners' locations-an artificial analogue of hippocampal social place cells (SPCs). These social representations are further utilized by a hierarchical reinforcement learning policy that actively explores to reduce joint uncertainty. On the Memory-Maze benchmark, our approach shows exceptional resilience to bandwidth constraints: success degrades gracefully from 73.5% to 64.4% as bandwidth shrinks from 128 to 4 bits/step, whereas a full-broadcast baseline collapses from 67.6% to 28.6%. Our findings establish a theoretically principled and biologically plausible basis for how complex social representations emerge from a unified predictive drive, leading to collective intelligence.

---

## 91. Trapped Fermions Through Kolmogorov-Arnold Wavefunctions

**论文链接:** [http://arxiv.org/abs/2512.07800v1](http://arxiv.org/abs/2512.07800v1)

**作者:** Paulo F. Bedaque, Jacob Cigliano, Hersh Kumar, Srijit Paul, Suryansh Rajawat

**发布时间:** 2025-12-08

**备注:** 18 pages, 6 figures

### GPT解析

### 总结

研究了一种使用Kolmogorov-Arnold网络构建神经网络波函数的变分蒙特卡洛框架，用于处理一维自旋-1/2费米子混合系统

### 背景

需要精确模拟一维自旋-1/2费米子混合系统的量子行为

### 目的

开发一种能够高精度模拟费米子系统量子行为的计算方法

### 方法

使用Kolmogorov-Arnold网络构建通用神经网络波函数变分蒙特卡洛框架，结合系统化迁移学习和波函数短程行为优化

### 主要发现

方法在亚百分精度下与精确结果一致；能够准确描述吸引相互作用中的配对效应；在杂质情况下与已知结果相符

### 结论

该方法通过结合迁移学习和波函数短程行为优化，显著提高了计算效率，同时保持了高精度

### 翻译

我们研究了使用Kolmogorov-Arnold网络构建通用神经网络波函数变分假设的、用于捕获一维自旋-1/2费米子混合系统的变分蒙特卡洛框架。该方法原则上可以达到任意精度，仅受蒙特卡洛采样限制，并在亚百分精度下与精确结果进行了验证。对于吸引相互作用，它捕获了配对效应，在杂质情况下与已知结果一致。我们提出了一种基于网络参数数量的系统化迁移学习方法，允许针对目标精度进行高效训练。通过将波函数的短程行为纳入波函数假设中而不引入偏差，我们大大提高了方法的效率。


### 论文摘要

We investigate a variational Monte Carlo framework for trapped one-dimensional mixture of spin-$\frac{1}{2}$ fermions using Kolmogorov-Arnold networks (KANs) to construct universal neural-network wavefunction ansätze. The method can, in principle, achieve arbitrary accuracy, limited only by the Monte Carlo sampling and was checked against exact results at sub-percent precision. For attractive interactions, it captures pairing effects, and in the impurity case it agrees with known results. We present a method of systematic transfer learning in the number of network parameters, allowing for efficient training for a target precision. We vastly increase the efficiency of the method by incorporating the short-distance behavior of the wavefunction into the ansätz without biasing the method.

---

## 92. Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement

**论文链接:** [http://arxiv.org/abs/2512.07611v1](http://arxiv.org/abs/2512.07611v1)

**作者:** Yongsheng Lian

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究系统比较了三种强化学习算法（PPO、GRPO和DAPO）在提高大型语言模型复杂推理能力方面的效果，并通过受控迁移学习评估提供了实用的训练指导。

### 背景

大型语言模型的复杂推理能力可以通过强化学习算法进行提升，但不同算法的效果和适用条件尚不明确。

### 目的

系统比较三种强化学习算法（PPO、GRPO和DAPO）在提高大型语言模型推理能力方面的效果，并提供实用的训练参数指导。

### 方法

采用受控迁移学习评估方法，首先让模型在专门的Countdown Game上进行微调，然后在一系列通用推理基准测试上进行评估，并进行参数分析。

### 主要发现

1) RL训练的模型在所有任务上都优于基线模型，但改进程度因基准测试而异；2) 在GRPO和DAPO中增加组大小会导致训练更稳定且准确率更高；3) KL惩罚系数的影响是非单调的；4) DAPO中的动态采样组件不会提高性能，禁用它时DAPO表现最佳。

### 结论

不同强化学习算法在提高大型语言模型推理能力方面有不同表现，参数设置对性能有显著影响，应根据具体任务和算法特点调整训练参数。

### 翻译

本研究提出了三种强化学习算法（PPO、GRPO和DAPO）用于提高大型语言模型复杂推理能力的系统比较。我们的主要贡献是一种受控的迁移学习评估：模型首先在专门的Countdown Game上进行微调，然后在一套通用推理基准测试上进行评估。在所有任务中，RL训练的模型都优于其对应的基线模型，尽管改进程度因基准测试而异。我们的参数分析为基于RL的LLM训练提供了实用指导。在GRPO和DAPO中增加组大小会导致训练动态更稳定且准确率更高，而KL惩罚系数的影响是非单调的。此外，我们发现DAPO中的动态采样组件不会提高性能；事实上，禁用DS时DAPO能达到最佳整体结果。


### 论文摘要

This study presents a systematic comparison of three Reinforcement Learning (RL) algorithms (PPO, GRPO, and DAPO) for improving complex reasoning in large language models (LLMs). Our main contribution is a controlled transfer-learning evaluation: models are first fine-tuned on the specialized Countdown Game and then assessed on a suite of general-purpose reasoning benchmarks. Across all tasks, RL-trained models outperform their corresponding base models, although the degree of improvement differs by benchmark.   Our parametric analysis offers practical guidance for RL-based LLM training. Increasing the group size in GRPO and DAPO leads to more stable training dynamics and higher accuracy, while the impact of the KL-penalty coefficient is non-monotonic. Additionally, we find that the Dynamic Sampling (DS) component in DAPO does not improve performance; in fact, the best overall results are achieved with DAPO when DS is disabled.

---

## 93. XMCQDPT2-Fidelity Transfer-Learning Potentials and a Wavepacket Oscillation Model with Power-Law Decay for Ultrafast Photodynamics

**论文链接:** [http://arxiv.org/abs/2512.07537v1](http://arxiv.org/abs/2512.07537v1)

**作者:** Ivan V. Dudakov, Pavel M. Radzikovitsky, Dmitry S. Popov, Denis A. Firsov, Vadim V. Korolev, Daniil N. Chistikov, Vladimir E. Bochenkov, Anastasia V. Bochenkova

**发布时间:** 2025-12-08

**备注:** 26 pages, 4 tables, 4 figures

### GPT解析

### 总结

本研究开发了一种基于机器学习的原子间势(MLIPs)方法，用于准确模拟光化学反应，克服了生成高水平量子化学数据的高成本障碍。通过迁移学习、多态学习和Δ学习等技术，该方法实现了多态多参考微扰理论精度，成功应用于亚胺阳离子的光解离研究，揭示了其完整的反应景观和竞争性衰减通道。研究还引入了波包振荡模型来解释群体动力学，并验证了MLIP-不确定性修正的重要性。

### 背景

理论化学的核心追求之一是准确模拟光化学反应，这类反应通过锥形交叉点的非绝热跃迁控制。机器学习已成为构建所需势能面的变革性工具，但将其应用于激发态面临基本障碍：生成高水平量子化学数据的成本高昂。

### 目的

开发一种机器学习方法来克服光化学反应模拟中的数据生成成本问题，实现高精度的势能面构建，并应用于亚胺阳离子的光解离研究，揭示其完整的反应动力学机制。

### 方法

1. 开发机器学习原子间势(MLIPs)，通过迁移学习、多态学习和Δ学习技术实现多态多参考微扰理论精度
2. 将方法应用于亚胺阳离子的S₂光激发研究
3. 使用XMCQDPT2/SA(3)-CASSCF(12,12)电子结构描述
4. 引入MLIP-不确定性修正基于模型集合预测
5. 开发波包振荡模型来解释群体动力学

### 主要发现

1. 不同的MLIP模型对群体动力学有不同影响，与其性能相关
2. MLIP-不确定性修正使不同方法达成一致，验证了这一指标对可靠动力学的重要性
3. 波包振荡模型定量重现了超快衰减，建立了量子跃迁概率和经典速率常数之间的直接联系
4. 动力学拟合产生了通道特定寿命，支持了通过新型σπ*/S₀锥形交叉点介导的光化学途径

### 结论

机器学习方法可以有效克服光化学反应模拟中的数据生成成本问题，实现高精度的势能面构建。MLIP-不确定性修正对于确保动力学模拟的可靠性至关重要。波包振荡模型为解释群体动力学提供了机制透明的框架，能够直接从第一性原理模拟中提取态特定寿命。这些发现为理解和预测光化学反应提供了新的工具和见解。

### 翻译

理论化学的核心追求之一是准确模拟光化学反应，这类反应通过锥形交叉点的非绝热跃迁控制。机器学习已成为构建所需势能面的变革性工具，但将其应用于激发态面临基本障碍：生成高水平量子化学数据的成本高昂。我们通过开发机器学习原子间势(MLIPs)克服了这一挑战，这些势能通过各种技术（如迁移学习、多态学习和Δ学习）实现了多态多参考微扰理论精度。将这种方法应用于亚胺阳离子，我们揭示了其在S₂光激发后的完整光解离景观。全面的XMCQDPT2/SA(3)-CASSCF(12,12)电子结构描述捕获了所有竞争性衰减通道，包括S₁分支到光异构化和直接H₂损失途径。我们的结果表明，群体动力学通常取决于MLIP模型，与其性能相关。同时，基于模型集合预测引入的MLIP-不确定性修正使不同方法达成一致，验证了这一指标对可靠动力学的重要性。为了解释群体动力学，我们引入了波包振荡模型 - 一种机制透明、幂律动力学框架，可直接从第一性原理模拟中提取态特定寿命。该模型定量重现了超快衰减，在量子跃迁概率和经典速率常数之间建立了直接联系。动力学拟合产生了通道特定寿命，支持了最近通过新型σπ*/S₀锥形交叉点介导的光化学途径。


### 论文摘要

A central pursuit in theoretical chemistry is the accurate simulation of photochemical reactions, which are governed by nonadiabatic transitions through conical intersections. Machine learning has emerged as a transformative tool for constructing the necessary potential energy surfaces, but applying it to excited states faces a fundamental barrier: the cost of generating high-level quantum chemistry data. We overcome this challenge by developing machine-learning interatomic potentials (MLIPs) that achieve multi-state multi-reference perturbation theory accuracy through various techniques, such as transfer, multi-state, and $Δ$-learning. Applied to the methaniminium cation, our highest-fidelity transfer-learning model uncovers its complete photodissociation landscape following S$_2$ photoexcitation. The comprehensive XMCQDPT2/SA(3)-CASSCF(12,12) electronic structure description captures all competing decay channels, including S$_1$ branching into photoisomerization and direct H$_2$-loss pathways. Our results show that the population dynamics generally depends on the MLIP model, correlating with its performance. At the same time, the introduction of MLIP-uncertainty corrections based on the predictions of an ensemble of models brings different approaches into agreement, validating this metric as essential for reliable dynamics. To interpret the population dynamics, we introduce a wavepacket oscillation model - a mechanistically transparent, power-law kinetics framework that extracts state-specific lifetimes directly from first-principles simulations. The model quantitatively reproduces the ultrafast decay, creating a direct link between quantum transition probabilities and classical rate constants. The kinetic fits yield channel-specific lifetimes, supporting the recently discovered photochemical pathway mediated by a novel $σπ^*/S_0$ conical intersection.

---

## 94. Dictionary-Based Contrastive Learning for GNSS Jamming Detection

**论文链接:** [http://arxiv.org/abs/2512.07512v1](http://arxiv.org/abs/2512.07512v1)

**作者:** Zawar Hussain, Arslan Majal, Aamir Hussain Chughtai, Talha Nadeem

**发布时间:** 2025-12-08

### GPT解析

### 总结

该研究提出了一种基于字典的对比学习（DBCL）框架，用于GNSS干扰检测，结合迁移学习、对比表示学习和模型压缩技术，实现了在资源有限的嵌入式接收器上的实时可靠干扰检测。

### 背景

GNSS信号在导航、运输和工业网络中应用广泛，但其极低接收功率使其极易受到射频干扰和故意干扰。虽然现代数据驱动方法提供了强大表示能力，但在资源有限的嵌入式设备上进行实时可靠的干扰检测仍面临高计算和内存需求的挑战。

### 目的

开发一种能够在资源有限的嵌入式接收器上实现实时且可靠的GNSS干扰检测的方法，同时减少计算和内存需求并保持高准确性。

### 方法

提出基于字典的对比学习（DBCL）框架，整合迁移学习、对比表示学习和模型压缩技术；结合调整后的对比和基于字典的损失函数增强特征可分离性；应用结构化剪枝和知识蒸馏减少模型复杂度。

### 主要发现

在多种数据条件下的评估表明，该算法持续优于CNN、MobileViT和ResNet-18架构；框架显著减少内存占用和推理延迟，适合实时、低功耗GNSS干扰检测。

### 结论

DBCL框架有效解决了GNSS干扰检测中的实时性和资源限制问题，通过结合多种先进技术实现了在嵌入式设备上的高效干扰检测，同时保持高准确性。

### 翻译

全球导航卫星系统（GNSS）信号在导航、运输和工业网络应用中至关重要。然而，它们的极低接收功率使其极易受到射频干扰（RFI）和故意干扰的影响。现代数据驱动方法为这类应用提供了强大的表示能力，但在资源有限的嵌入式接收器上进行实时可靠的干扰检测仍然是一个关键挑战，这是因为传统学习范式的计算和内存需求很高。为解决这些挑战，这项工作提出了一种用于GNSS干扰检测的基于字典的对比学习（DBCL）框架，该框架整合了迁移学习、对比表示学习和模型压缩技术。该框架结合了调整后的对比和基于字典的损失函数，以增强在低数据条件下的特征可分离性，并应用结构化剪枝和知识蒸馏来减少模型复杂度，同时保持高准确性。在多种数据条件下的广泛评估表明，所提出的算法在性能上持续优于现代CNN、MobileViT和ResNet-18架构。该框架显著减少了内存占用和推理延迟，证实了其在嵌入式平台上进行实时、低功耗GNSS干扰检测的适用性。


### 论文摘要

Global Navigation Satellite System (GNSS) signals are fundamental in applications across navigation, transportation, and industrial networks. However, their extremely low received power makes them highly vulnerable to radio-frequency interference (RFI) and intentional jamming. Modern data-driven methods offer powerful representational power for such applications, however real-time and reliable jamming detection on resource-limited embedded receivers remains a key challenge due to the high computational and memory demands of the conventional learning paradigm. To address these challenges, this work presents a dictionary-based contrastive learning (DBCL) framework for GNSS jamming detection that integrates transfer learning, contrastive representation learning, and model compression techniques. The framework combines tuned contrastive and dictionary-based loss functions to enhance feature separability under low-data conditions and applies structured pruning and knowledge distillation to reduce model complexity while maintaining high accuracy. Extensive evaluation across varying data regimes demonstrate that the proposed algorithm consistently outperforms modern CNN, MobileViT, and ResNet-18 architectures. The framework achieves a substantial reduction in memory footprint and inference latency, confirming its suitability for real-time, low-power GNSS interference detection on embedded platforms.

---

## 95. Reevaluating Automated Wildlife Species Detection: A Reproducibility Study on a Custom Image Dataset

**论文链接:** [http://arxiv.org/abs/2512.07305v1](http://arxiv.org/abs/2512.07305v1)

**作者:** Tobias Abraham Haider

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究重现了Carl等人关于使用预训练Google Inception-ResNet-v2模型自动检测欧洲野生哺乳动物物种的研究，评估了其方法的可复现性和泛化能力。

### 背景

Carl等人评估了预训练的Google Inception-ResNet-v2模型在相机陷阱图像中自动检测欧洲野生哺乳动物物种的表现。

### 目的

评估Carl等人研究方法的可复现性和泛化能力。

### 方法

从零开始重新实现实验，使用公开可用的资源和包含90个物种900张图像的不同数据集，经过最小预处理后进行分类。

### 主要发现

获得了62%的整体分类准确率，与原始报告中71%的准确率相近；每个类别的表现差异很大，宏F1得分为0.28，突显了当标签与ImageNet类别不完全对齐时泛化的局限性。

### 结论

预训练的卷积神经网络可以为野生动物物种识别提供实用的基线，但也需要物种特定的适应或迁移学习来获得一致的高质量预测。

### 翻译

本研究重新审视了Carl等人的研究结果，他们评估了预训练的Google Inception-ResNet-v2模型用于在相机陷阱图像中自动检测欧洲野生哺乳动物物种。为了评估其方法的可复现性和泛化能力，我们使用公开可用的资源和包含90个物种900张图像的不同数据集，从头开始重新实现了实验。经过最小预处理后，我们获得了62%的整体分类准确率，与原始工作中报告的71%基本一致，尽管数据集有所不同。与原始研究一样，每类性能差异很大，宏F1得分为0.28，突显了当标签不直接与ImageNet类别对齐时泛化的局限性。我们的结果证实，预训练的卷积神经网络可以为野生动物物种识别提供实用的基线，但也强化了需要物种特定的适应或迁移学习来实现一致的高质量预测。


### 论文摘要

This study revisits the findings of Carl et al., who evaluated the pre-trained Google Inception-ResNet-v2 model for automated detection of European wild mammal species in camera trap images. To assess the reproducibility and generalizability of their approach, we reimplemented the experiment from scratch using openly available resources and a different dataset consisting of 900 images spanning 90 species. After minimal preprocessing, we obtained an overall classification accuracy of 62%, closely aligning with the 71% reported in the original work despite differences in datasets. As in the original study, per-class performance varied substantially, as indicated by a macro F1 score of 0.28,highlighting limitations in generalization when labels do not align directly with ImageNet classes. Our results confirm that pretrained convolutional neural networks can provide a practical baseline for wildlife species identification but also reinforce the need for species-specific adaptation or transfer learning to achieve consistent, high-quality predictions.

---

## 96. RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.06811v1](http://arxiv.org/abs/2512.06811v1)

**作者:** Xiang Lin, Weixin Li, Shu Guo, Lihong Wang, Di Huang

**发布时间:** 2025-12-07

**备注:** Accepted by AAAI 2026(Oral)

### GPT解析

### 总结

本文提出了一种名为RMAdapter的新型多模态适配器，用于解决预训练视觉-语言模型在少样本场景下面临的任务特定适应与泛化能力平衡问题。

### 背景

预训练视觉-语言模型（如CLIP）已成为多模态迁移学习的重要工具，但在少样本场景下微调这些模型存在显著挑战，当前研究主要集中在基于提示的适应方法，而基于适配器的方法研究不足。

### 目的

解决少样本场景下微调视觉-语言模型时平衡任务特定适应和泛化能力的挑战，并探索基于适配器的适应方法。

### 方法

提出基于重建的多模态适配器（RMAdapter），采用双分支架构：适应分支通过参数高效微调注入任务特定知识，重建分支通过将潜在空间特征重建回原始特征空间来保留通用知识。通过计算每层的重建损失和共享投影模块保持轻量级，并加入一致性约束来调节判别性和泛化能力之间的权衡。

### 主要发现

RMAdapter在三个代表性任务上（泛化到新类别、泛化到新目标数据集和领域泛化）均优于最先进的方法，且不依赖数据增强或重复提示设计。

### 结论

RMAdapter有效地平衡了任务特定知识和通用知识，通过双分支架构和精心设计实现了轻量级计算，同时保持了优异的性能。

### 翻译

预训练视觉-语言模型（如CLIP）已成为多模态迁移学习中的基本工具。然而，在少样本场景下微调VLMs在平衡获得模型的特定任务适应性和泛化能力方面存在重大挑战。同时，当前研究主要集中于基于提示的适应方法，而基于适配器的方法研究不足，显示出明显的性能差距。为应对这些挑战，我们引入了一种新颖的基于重建的多模态适配器（RMAdapter），它利用双分支架构。与传统单分支适配器不同，RMAdapter包括：（1）一个适应分支，通过参数高效微调注入任务特定知识；（2）一个重建分支，通过将潜在空间特征重建回原始特征空间来保留通用知识。这种设计促进了通用知识和任务特定知识之间的动态平衡。重要的是，尽管RMAdapter引入了额外的重建分支，但经过精心优化以保持轻量级。通过在每层本地计算重建损失和共享投影模块，整体计算开销保持最小。还加入了一致性约束以更好地调节判别性和泛化能力之间的权衡。我们在三个代表性任务上全面评估了RMAdapter的有效性：泛化到新类别、泛化到新目标数据集和领域泛化。在不依赖数据增强或重复提示设计的情况下，我们的RMAdapter在所有评估指标上都一致优于最先进的方法。


### 论文摘要

Pre-trained Vision-Language Models (VLMs), \textit{e.g.} CLIP, have become essential tools in multimodal transfer learning. However, fine-tuning VLMs in few-shot scenarios poses significant challenges in balancing task-specific adaptation and generalization in the obtained model. Meanwhile, current researches have predominantly focused on prompt-based adaptation methods, leaving adapter-based approaches underexplored and revealing notable performance gaps. To address these challenges, we introduce a novel Reconstruction-based Multimodal Adapter (RMAdapter), which leverages a dual-branch architecture. Unlike conventional single-branch adapters, RMAdapter consists of: (1) an adaptation branch that injects task-specific knowledge through parameter-efficient fine-tuning, and (2) a reconstruction branch that preserves general knowledge by reconstructing latent space features back into the original feature space. This design facilitates a dynamic balance between general and task-specific knowledge. Importantly, although RMAdapter introduces an additional reconstruction branch, it is carefully optimized to remain lightweight. By computing reconstruction loss locally at each layer and sharing projection modules, the overall computational overhead is kept minimal. A consistency constraint is also incorporated to better regulate the trade-off between discriminability and generalization. We comprehensively evaluate the effectiveness of RMAdapter on three representative tasks: generalization to new categories, generalization to new target datasets, and domain generalization. Without relying on data augmentation or duplicate prompt designs, our RMAdapter consistently outperforms state-of-the-art approaches across all evaluation metrics.

---

## 97. A Patient-Doctor-NLP-System to contest inequality for less privileged

**论文链接:** [http://arxiv.org/abs/2512.06734v1](http://arxiv.org/abs/2512.06734v1)

**作者:** Subrit Dikshit, Ritu Tiwari, Priyank Jain

**发布时间:** 2025-12-07

**备注:** 19 pages, 6 figures

### GPT解析

### 总结

本研究提出了一种名为PDFTEMRA的紧凑型transformer架构，通过集成模型蒸馏、频域调制、集成学习和随机激活模式，在保持语言理解性能的同时显著降低计算需求，特别适合资源受限的医疗环境应用。

### 背景

迁移学习促进了大型语言模型在主流NLP用例中的快速发展，但在资源受限的现实医疗环境中训练和部署这些大型模型仍然具有挑战性。

### 目的

解决视觉障碍用户和低资源语言(如印地语)使用者在农村环境中需要医疗援助时支持有限的问题。

### 方法

提出PDFTEMRA架构，在针对印地语和可访问性场景定制的医疗问答和咨询数据集上训练和评估，并与标准NLP最先进模型基线进行比较。

### 主要发现

PDFTEMRA实现了与标准模型相当的性能，但计算需求显著降低，表明其适用于可访问、包容性、低资源医疗NLP应用。

### 结论

PDFTEMRA是一种有效的方法，可以在保持语言理解性能的同时降低计算成本，特别适合资源受限的医疗环境。

### 翻译

迁移学习(TL)加速了大型语言模型(LLMs)在主流自然语言处理(NLP)用例中的快速发展和应用。然而，在资源受限的现实医疗环境中训练和部署这些大型LLM仍然具有挑战性。本研究解决了视觉障碍用户和低资源语言(如印地语)使用者在农村环境中需要医疗援助时支持有限的问题。我们提出了PDFTEMRA(高性能蒸馏频率变换器集成模型与随机激活)，一种基于transformer的紧凑型架构，集成了模型蒸馏、频域调制、集成学习和随机激活模式，以降低计算成本同时保持语言理解性能。该模型在针对印地语和可访问性场景定制的医疗问答和咨询数据集上进行了训练和评估，并将其性能与标准NLP最先进模型基线进行了比较。结果表明，PDFTEMRA实现了相当的性能但计算需求显著降低，表明其适用于可访问、包容性、低资源医疗NLP应用。


### 论文摘要

Transfer Learning (TL) has accelerated the rapid development and availability of large language models (LLMs) for mainstream natural language processing (NLP) use cases. However, training and deploying such gigantic LLMs in resource-constrained, real-world healthcare situations remains challenging. This study addresses the limited support available to visually impaired users and speakers of low-resource languages such as Hindi who require medical assistance in rural environments. We propose PDFTEMRA (Performant Distilled Frequency Transformer Ensemble Model with Random Activations), a compact transformer-based architecture that integrates model distillation, frequency-domain modulation, ensemble learning, and randomized activation patterns to reduce computational cost while preserving language understanding performance. The model is trained and evaluated on medical question-answering and consultation datasets tailored to Hindi and accessibility scenarios, and its performance is compared against standard NLP state-of-the-art model baselines. Results demonstrate that PDFTEMRA achieves comparable performance with substantially lower computational requirements, indicating its suitability for accessible, inclusive, low-resource medical NLP applications.

---

## 98. Diagnosis-based mortality prediction for intensive care unit patients via transfer learning

**论文链接:** [http://arxiv.org/abs/2512.06511v1](http://arxiv.org/abs/2512.06511v1)

**作者:** Mengqi Xu, Subha Maity, Joel Dubin

**发布时间:** 2025-12-06

### GPT解析

### 总结

研究评估了迁移学习方法在考虑诊断异质性的重症监护室死亡率预测中的应用，发现迁移学习方法优于传统方法。

### 背景

在重症监护室中，重症疾病的基础原因在不同诊断间存在显著差异，但考虑诊断异质性的预测模型尚未得到系统研究。

### 目的

评估针对特定诊断的死亡率预测的迁移学习方法，以填补研究空白。

### 方法

应用基于GLM和XGBoost的模型到eICU协作研究数据库，并进行比较分析。

### 主要发现

迁移学习始终优于仅使用特定诊断数据训练的模型和单独使用APACHE IVa评分的模型；迁移学习的校准性能优于在合并数据上训练的模型；Youden截断值是比传统0.5更适合二分类结果的决定阈值；迁移学习在各种截断标准下保持一致的高预测性能。

### 结论

迁移学习方法对于考虑诊断异质性的重症监护室死亡率预测有效，优于传统预测方法。

### 翻译

在重症监护室中，重症疾病的基础原因在不同诊断间存在显著差异，但考虑到诊断异质性的预测模型尚未得到系统研究。为解决这一空白，我们评估了针对特定诊断的死亡率预测的迁移学习方法，并将基于GLM和XGBoost的模型应用于eICU协作研究数据库。我们的结果表明，迁移学习始终优于仅使用特定诊断数据训练的模型和单独使用著名的ICU疾病严重程度评分(APACHE IVa)的模型，同时其校准性能也优于在合并数据上训练的模型。我们的研究还表明，Youden截断值是比传统0.5更适合二分类结果的决定阈值，且迁移学习在各种截断标准下保持一致的高预测性能。


### 论文摘要

In the intensive care unit, the underlying causes of critical illness vary substantially across diagnoses, yet prediction models accounting for diagnostic heterogeneity have not been systematically studied. To address the gap, we evaluate transfer learning approaches for diagnosis-specific mortality prediction and apply both GLM- and XGBoost-based models to the eICU Collaborative Research Database. Our results demonstrate that transfer learning consistently outperforms models trained only on diagnosis-specific data and those using a well-known ICU severity-of-illness score, i.e., APACHE IVa, alone, while also achieving better calibration than models trained on the pooled data. Our findings also suggest that the Youden cutoff is a more appropriate decision threshold than the conventional 0.5 for binary outcomes, and that transfer learning maintains consistently high predictive performance across various cutoff criteria.

---

## 99. Deep learning for autism detection using clinical notes: A comparison of transfer learning for a transparent and black-box approach

**论文链接:** [http://arxiv.org/abs/2512.06161v1](http://arxiv.org/abs/2512.06161v1)

**作者:** Gondy Leroy, Prakash Bisht, Sai Madhuri Kandula, Nell Maltman, Sydney Rice

**发布时间:** 2025-12-05

**DOI:** 10.1016/j.artmed.2025.103318

**备注:** 9 pages

### GPT解析

### 总结

本研究提出了一种基于BioBERT的透明可解释机器学习方法，用于自闭症谱系障碍(ASD)的诊断，通过分析非结构化临床文本和行为描述映射到诊断标准，实现了高准确率的ASD自动诊断。

### 背景

自闭症谱系障碍是一种复杂的神经发育状况，其患病率上升导致诊断过程需求增加。现有机器学习模型大多作为黑盒操作，通常只在单一数据集上训练，限制了其泛化能力。

### 目的

开发一种透明且可解释的机器学习方法，提高ASD诊断模型的泛化能力和临床实用性。

### 方法

使用BioBERT语言模型分析非结构化临床文本，训练模型标记行为描述并映射到诊断标准，然后分配最终诊断标签。采用两个不同真实世界数据集评估迁移学习能力，比较顺序训练和混合训练策略，并与黑盒方法进行对比。

### 主要发现

透明模型表现优异，混合数据训练策略达到最佳结果（97%敏感性，98%特异性）；顺序训练导致性能下降；黑盒模型表现较差（90%敏感性，96%特异性）；透明方法整体优于黑盒方法；混合数据训练应作为首选方法。

### 结论

该研究为神经发育诊断中更可信、可泛化且具有临床实用性的AI工具开发奠定了基础，提高了ASD诊断的自动化和准确性。

### 翻译

自闭症谱系障碍(ASD)是一种复杂的神经发育状况，其患病率上升对冗长的诊断过程提出了越来越高的要求。机器学习(ML)在自动化ASD诊断方面显示出潜力，但大多数现有模型作为黑盒运行，并且通常只在单一数据集上训练，限制了它们的泛化能力。在本研究中，我们引入了一种透明且可解释的ML方法，利用最先进的语言模型BioBERT来分析非结构化临床文本。该模型被训练用于标记行为描述并将其映射到诊断标准，然后这些标准用于分配最终标签（ASD或非ASD）。我们使用两个不同的真实世界数据集评估迁移学习能力（将知识转移到新数据的能力）。我们在数据集上顺序训练和混合训练，并比较了最佳模型及其转移到新数据的能力。我们还创建了一种黑盒方法并重复了这一迁移过程以进行比较。我们的透明模型表现出强大的性能，混合数据训练策略产生了最佳结果（97%敏感性，98%特异性）。跨数据集的顺序训练导致性能略有下降，突显了训练数据顺序的重要性。黑盒模型在顺序训练或混合数据训练时表现较差（90%敏感性，96%特异性）。总体而言，我们的透明方法优于黑盒方法。在训练时混合数据产生了稍好的性能，并且在实际可行时应作为首选方法。这项工作为神经发育诊断中更可信、可泛化且具有临床实用性的AI工具铺平了道路。


### 论文摘要

Autism spectrum disorder (ASD) is a complex neurodevelopmental condition whose rising prevalence places increasing demands on a lengthy diagnostic process. Machine learning (ML) has shown promise in automating ASD diagnosis, but most existing models operate as black boxes and are typically trained on a single dataset, limiting their generalizability. In this study, we introduce a transparent and interpretable ML approach that leverages BioBERT, a state-of-the-art language model, to analyze unstructured clinical text. The model is trained to label descriptions of behaviors and map them to diagnostic criteria, which are then used to assign a final label (ASD or not). We evaluate transfer learning, the ability to transfer knowledge to new data, using two distinct real-world datasets. We trained on datasets sequentially and mixed together and compared the performance of the best models and their ability to transfer to new data. We also created a black-box approach and repeated this transfer process for comparison. Our transparent model demonstrated robust performance, with the mixed-data training strategy yielding the best results (97 % sensitivity, 98 % specificity). Sequential training across datasets led to a slight drop in performance, highlighting the importance of training data order. The black-box model performed worse (90 % sensitivity, 96 % specificity) when trained sequentially or with mixed data. Overall, our transparent approach outperformed the black-box approach. Mixing datasets during training resulted in slightly better performance and should be the preferred approach when practically possible. This work paves the way for more trustworthy, generalizable, and clinically actionable AI tools in neurodevelopmental diagnostics.

---

## 100. Comparative Analysis of Autonomous and Systematic Control Strategies for Hole-Doped Hubbard Clusters: Reinforcement Learning versus Physics-Guided Design

**论文链接:** [http://arxiv.org/abs/2512.06095v1](http://arxiv.org/abs/2512.06095v1)

**作者:** Shivanshu Dwivedi, Kalum Palandage

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究比较了两种控制量子点阵列中电子关联的方法：物理引导设计和深度强化学习，发现RL代理在优化效率和准确性方面表现出色。

### 背景

量子点阵列中的电子关联工程需要导航高维、非凸参数空间，其中空穴掺杂从根本上改变了物理性质。

### 目的

比较两种控制范式（物理引导设计和自主深度强化学习）在单空穴半满Hubbard模型上的性能，评估RL在复杂量子系统优化中的有效性。

### 方法

使用系统的物理引导设计和具有几何感知神经架构的自主深度强化学习方法，在五个三维晶格（从四面体到面心立方结构）上对单空穴半满Hubbard模型进行基准测试。

### 主要发现

RL代理实现了人类竞争级别的准确性和百分之九十五点五的任务成功率，样本效率比网格搜索高三个到四个数量级，优于其他黑盒优化方法，并通过迁移学习实现了百分之九十一的少样本泛化到未见过的几何结构。

### 结论

自主深度强化学习是复杂量子系统快速优化和非明显策略发现的可行且高效的框架。

### 翻译

在量子点阵列中工程化电子关联需要导航高维、非凸参数空间，其中空穴掺杂从根本上改变了物理性质。我们提出了对单空穴半满Hubbard模型两种控制范式的比较研究：(i)系统的物理引导设计和(ii)具有几何感知神经架构的自主深度强化学习。虽然系统分析揭示了关键设计原则，如用于捕获移动空穴的场诱导局域化，但对于优化来说计算上不可行。我们展示了一个自主RL代理，在从四面体到面心立方结构的五个三维晶格上进行基准测试，实现了人类竞争级别的准确性和百分之九十五点五的保留任务成功率。该代理的样本效率比网格搜索高三个到四个数量级，并优于其他黑盒优化方法。迁移学习实现了百分之九十一的少样本泛化到未见过的几何结构。这项工作确立了自主RL作为复杂量子系统快速优化和非明显策略发现的可行且高效的框架。


### 论文摘要

Engineering electron correlations in quantum dot arrays demands navigation of high-dimensional, non-convex parameter spaces where hole doping fundamentally alters the physics. We present a comparative study of two control paradigms for the one-hole, half-filled Hubbard model: (i) systematic physics-guided design and (ii) autonomous deep reinforcement learning with geometry-aware neural architectures. While systematic analysis reveals key design principles, such as field-induced localization for trapping the mobile hole, it becomes computationally intractable for optimization. We show that an autonomous RL agent, benchmarked across five 3D lattices from tetrahedron to FCC, achieves human-competitive accuracy (R^2 > 0.97) and 95.5 percent success on held-out tasks. The agent is 3-4 orders of magnitude more sample-efficient than grid search and outperforms other black-box optimization methods. Transfer learning yields 91 percent few-shot generalization to unseen geometries. This work establishes autonomous RL as a viable and highly efficient framework for rapid optimization and non-obvious strategy discovery in complex quantum systems.

---

## 101. High-Throughput Unsupervised Profiling of the Morphology of 316L Powder Particles for Use in Additive Manufacturing

**论文链接:** [http://arxiv.org/abs/2512.06012v1](http://arxiv.org/abs/2512.06012v1)

**作者:** Emmanuel Akeweje, Conall Kirk, Chi-Wai Chan, Denis Dowling, Mimi Zhang

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究开发了一种自动化的机器学习框架，用于高通量表征金属粉末形态，解决了传统粉末表征方法通量低且无法捕捉工业批次不均匀性的问题。通过评估三种聚类管道，发现傅里叶描述符结合k均值方法最为有效，为SLM工艺中的粉末实时监测提供了新途径。

### 背景

选择性激光熔化是一种粉末床增材制造技术，其零件质量严重依赖于原料粉末的形态。然而，传统的粉末表征方法通量低且定性，无法捕捉工业规模批次的不均匀性，限制了工艺优化和质量控制。

### 目的

提出一个自动化的机器学习框架，通过高通量成像与形状提取和聚类相结合，大规模分析金属粉末形态，为SLM工艺提供快速、准确的粉末质量评估方法。

### 方法

开发并评估了三种聚类管道：自编码器管道、形状描述符管道和函数数据管道。在一个包含约126,000张粉末图像(直径0.5-102微米)的数据集上进行测试，使用内部有效性指标评估各管道性能。

### 主要发现

傅里叶描述符+k均值聚类管道被识别为最有效的方法，在标准桌面工作站上实现了每粒子亚毫秒级运行时间，同时获得了最低的戴维斯-鲍尔丁指数和最高的卡林斯基-哈拉巴兹分数，表明其在准确性和效率上的优势。

### 结论

这种无监督学习框架能够快速、自动地评估粉末形态，并支持跟踪重复使用周期中的形态演变，为SLM工作流程中的原料实时监测提供了可能，有助于提高零件质量和工艺效率。

### 翻译

选择性激光熔化是一种粉末床增材制造技术，其零件质量严重依赖于原料粉末形态。然而，传统的粉末表征方法通量低且定性，无法捕捉工业规模批次的不均匀性。我们提出了一种自动化的机器学习框架，将高通量成像与形状提取和聚类相结合，用于大规模分析金属粉末形态。我们开发并评估了三种聚类管道：自编码器管道、形状描述符管道和函数数据管道。在一个包含约126,000张粉末图像(直径0.5-102微米)的数据集上，内部有效性指标识别出傅里叶描述符+k均值管道是最有效的，在标准桌面工作站上实现了每粒子亚毫秒级运行时间，同时获得了最低的戴维斯-鲍尔丁指数和最高的卡林斯基-哈拉巴兹分数。尽管本研究侧重于建立形态聚类框架，但产生的形状组为未来研究其与流动性、堆积密度和SLM零件质量的关系奠定了基础。总体而言，这种无监督学习框架能够快速、自动地评估粉末形态，并支持跟踪重复使用周期中的形态演变，为SLM工作流程中的原料实时监测提供了途径。


### 论文摘要

Selective Laser Melting (SLM) is a powder-bed additive manufacturing technique whose part quality depends critically on feedstock morphology. However, conventional powder characterization methods are low-throughput and qualitative, failing to capture the heterogeneity of industrial-scale batches. We present an automated, machine learning framework that couples high-throughput imaging with shape extraction and clustering to profile metallic powder morphology at scale. We develop and evaluate three clustering pipelines: an autoencoder pipeline, a shape-descriptor pipeline, and a functional-data pipeline. Across a dataset of approximately 126,000 powder images (0.5-102 micrometer diameter), internal validity metrics identify the Fourier-descriptor + k-means pipeline as the most effective, achieving the lowest Davies-Bouldin index and highest Calinski-Harabasz score while maintaining sub-millisecond runtime per particle on a standard desktop workstation. Although the present work focuses on establishing the morphological-clustering framework, the resulting shape groups form a basis for future studies examining their relationship to flowability, packing density, and SLM part quality. Overall, this unsupervised learning framework enables rapid, automated assessment of powder morphology and supports tracking of shape evolution across reuse cycles, offering a path toward real-time feedstock monitoring in SLM workflows.

---

## 102. Non-Negative Matrix Factorization Using Non-Von Neumann Computers

**论文链接:** [http://arxiv.org/abs/2512.00675v1](http://arxiv.org/abs/2512.00675v1)

**作者:** Ajinkya Borle, Charles Nicholas, Uchenna Chukwu, Mohammad-Ali Miri, Nicholas Chancellor

**发布时间:** 2025-11-30

**备注:** 14 pages, 5 figures, 6 tables and 1 appendix

### GPT解析

### 总结

该研究探索了使用基于能量的优化方法解决非负矩阵分解(NMF)问题，适用于非冯·诺依曼架构的机器。研究者使用Dirac-3设备评估了他们的方法，提出了QUBO模型和四次模型两种形式。实验结果表明，对于非负实数矩阵，融合方法优于单独使用Scikit-learn；对于非负整数矩阵，Dirac-3在大多数情况下优于Google的CP-SAT求解器。

### 背景

非负矩阵分解(NMF)是一种应用于无监督学习的矩阵分解问题，其一般形式及许多变体本质上都是NP难的。

### 目的

探索如何使用基于能量的优化方法解决NMF问题，使其适用于具有非冯·诺依曼架构的特定机器。

### 方法

使用基于熵计算范式的Dirac-3设备，提出了两种模型：(i)适合Ising机器的二元无约束二次优化模型(QUBO)，和(ii)允许实数和整数变量的四次模型，适合Dirac-3这样的机器。对于实数矩阵，采用融合方法；对于整数矩阵，与Google的CP-SAT求解器进行比较。

### 主要发现

当前设备无法解决大型NMF问题，但初步实验结果很有希望。对于非负实数矩阵，融合方法(先使用Dirac-3再结合Scikit-learn)在重构矩阵误差方面优于单独使用Scikit-learn。对于非负整数矩阵，Dirac-3在串行处理中大多数情况下优于CP-SAT求解器。

### 结论

未来研究可能会识别出熵计算和其他非冯·诺依曼架构可以提供明显优势的问题领域和变体。

### 翻译

非负矩阵分解(NMF)是一种矩阵分解问题，应用于无监督学习。该问题的一般形式(及其许多变体)本质上都是NP难的。在我们的工作中，我们探索了如何使用适合某些具有非冯·诺依曼架构机器的基于能量的优化方法来解决此问题。我们使用了由Quantum Computing Inc.制造的基于熵计算范式的Dirac-3设备来评估我们的方法。我们的公式包括：(i)适合Ising机器的二次无约束二元优化模型(QUBO)，以及允许实数和整数变量的四次公式(适合Dirac-3这样的机器)。尽管当前设备无法解决大型NMF问题，我们初步实验的结果足够有希望，值得进一步研究。对于非负实数矩阵，我们观察到，先使用Dirac-3，然后将结果作为初始因子矩阵输入到Scikit-learn的NMF程序的融合方法，在重构矩阵误差方面优于单独使用Scikit-learn的NMF程序（使用默认参数）。对于非负整数矩阵的实验，我们将Dirac-3设备与Google的CP-SAT求解器（在Or-Tools包中）进行了比较，发现对于串行处理，Dirac-3在大多数情况下优于CP-SAT。我们相信，该领域的未来工作可能会识别出熵计算（和其他非冯·诺依曼架构）可以提供明显优势的问题领域和变体。


### 论文摘要

Non-negative matrix factorization (NMF) is a matrix decomposition problem with applications in unsupervised learning. The general form of this problem (along with many of its variants) is NP-hard in nature. In our work, we explore how this problem could be solved with an energy-based optimization method suitable for certain machines with non-von Neumann architectures. We used the Dirac-3, a device based on the entropy computing paradigm and made by Quantum Computing Inc., to evaluate our approach. Our formulations consist of (i) a quadratic unconstrained binary optimization model (QUBO, suitable for Ising machines) and a quartic formulation that allows for real-valued and integer variables (suitable for machines like the Dirac-3). Although current devices cannot solve large NMF problems, the results of our preliminary experiments are promising enough to warrant further research. For non-negative real matrices, we observed that a fusion approach of first using Dirac-3 and then feeding its results as the initial factor matrices to Scikit-learn's NMF procedure outperforms Scikit-learn's NMF procedure on its own, with default parameters in terms of the error in the reconstructed matrices. For our experiments on non-negative integer matrices, we compared the Dirac-3 device to Google's CP-SAT solver (inside the Or-Tools package) and found that for serial processing, Dirac-3 outperforms CP-SAT in a majority of the cases. We believe that future work in this area might be able to identify domains and variants of the problem where entropy computing (and other non-von Neumann architectures) could offer a clear advantage.

---

## 103. Multivariate Gaussian Representation Learning for Medical Action Evaluation

**论文链接:** [http://arxiv.org/abs/2511.10060v2](http://arxiv.org/abs/2511.10060v2)

**作者:** Luming Yang, Haoxian Liu, Siqing Li, Alper Yilmaz

**发布时间:** 2025-11-13

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该研究引入了CPREval-6k医疗动作基准数据集和GaussMedAct多变量高斯编码框架，通过自适应时空表示学习改进医疗动作分析，实现了高精度和实时推理。

### 背景

医疗视觉中的细粒度动作评估面临独特挑战，包括缺乏全面的数据集、严格的精度要求以及对快速动作的时空动态建模不足。

### 目的

为了支持医疗动作评估模型的发展与评估，该研究旨在引入一个全面的医疗动作基准数据集并提出一个能有效建模快速医疗动作的框架。

### 方法

1) 引入CPREval-6k数据集：包含6,372个专家标注的视频，具有22个临床标签；2) 提出GaussMedAct框架，包括多变量高斯表示(将联合运动投影到时缩多维空间，分解动作为自适应3D高斯令牌)和混合空间编码(采用笛卡尔和向量双流策略，利用关节和骨骼特征)；3) 通过各向异性协方差建模保持运动语义，同时保持对时空噪声的鲁棒性。

### 主要发现

1) 在基准测试上实现92.1%的Top-1准确率，支持实时推理；2) 相比基线方法，准确率提高5.9%，同时计算量仅为10%；3) 跨数据集实验证实了该方法在鲁棒性方面的优越性。

### 结论

GaussMedAct框架通过自适应时空表示学习有效解决了医疗动作分析中的挑战，在准确率和效率方面都表现出色，为医疗视觉中的细粒度动作评估提供了新解决方案。

### 翻译

医疗视觉中的细粒度动作评估面临独特挑战，由于缺乏全面的数据集、严格的精度要求以及对快速动作的时空动态建模不足。为了支持发展和评估，我们引入了CPREval-6k，这是一个多视图、多标签的医疗动作基准，包含6,372个专家标注的视频和22个临床标签。利用这个数据集，我们提出了GaussMedAct，一个多变量高斯编码框架，通过自适应时空表示学习来推进医疗动作分析。多变量高斯表示将联合运动投影到时缩多维空间，并将动作分解为自适应3D高斯作为令牌。这些令牌通过各向异性协方差建模保持运动语义，同时保持对时空噪声的鲁棒性。混合空间编码采用笛卡尔和向量双流策略，有效利用关节和骨骼形式的骨骼信息。所提出的方法在基准测试上实现了92.1%的Top-1准确率，支持实时推理，相比基线方法准确率提高5.9%，同时仅使用10%的FLOPs。跨数据集实验证实了我们的方法在鲁棒性方面的优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决医疗动作（特别是心肺复苏CPR）的精确评估问题。在现实中，这个问题极其重要，因为心脏骤停每年仅在美国就导致超过436,000人死亡，而高质量的CPR可以将生存率提高一倍。然而，当前人工评估的准确率仅为74.8%，且现有计算机视觉系统无法捕捉厘米级的运动偏差（如5厘米的按压深度）和毫秒级的频率变化，直接影响抢救效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行思考：RGB方法缺乏解剖建模能力且计算延迟高，骨架方法则丢弃运动语义且易受噪声影响。作者借鉴了多个领域的工作：计算机图形学中的高斯飞溅技术、概率模型中的高斯混合模型、时空兴趣点概念，以及心理学研究中关于点光显示传达动作的发现。基于这些，作者设计了结合多元高斯表示和混合空间编码的新方法，同时捕捉绝对位置和相对运动信息，实现鲁棒的医疗动作评估。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是用多元高斯分布来建模人体动作的时空动态，将动作分解为自适应的3D高斯作为令牌，通过各向异性协方差建模保留运动语义，同时保持对噪声的鲁棒性。整体流程是：输入视频经过姿态估计得到关键点→将关键点分为笛卡尔空间（绝对位置）和向量空间（相对运动）两个流→两个流分别通过多元高斯表示模块生成高斯表示→特征融合后生成动作令牌张量→用于下游任务（如CPR错误分类和评估报告生成）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 创建了CPREVAL-6K最大的多视角临床CPR数据集；2) 提出了GAUSSMEDACT端到端框架；3) 设计了多元高斯表示(MGR)方法；4) 开发了混合空间编码(HSE)双流架构。相比之前的工作，不同之处在于：RGB方法虽特征丰富但缺乏解剖建模且计算成本高；骨架方法则丢弃运动语义；本文方法通过高斯建模避免了这些问题，同时捕捉绝对和相对运动信息，实现了92.1%的Top-1准确率，仅需10%的计算量，且对噪声更鲁棒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过多元高斯表示学习和混合空间编码，结合新的临床数据集，解决了医疗动作评估中的精度和效率问题，为提高心脏骤停患者生存率提供了创新工具。'}


### 论文摘要

Fine-grained action evaluation in medical vision faces unique challenges due to the unavailability of comprehensive datasets, stringent precision requirements, and insufficient spatiotemporal dynamic modeling of very rapid actions. To support development and evaluation, we introduce CPREval-6k, a multi-view, multi-label medical action benchmark containing 6,372 expert-annotated videos with 22 clinical labels. Using this dataset, we present GaussMedAct, a multivariate Gaussian encoding framework, to advance medical motion analysis through adaptive spatiotemporal representation learning. Multivariate Gaussian Representation projects the joint motions to a temporally scaled multi-dimensional space, and decomposes actions into adaptive 3D Gaussians that serve as tokens. These tokens preserve motion semantics through anisotropic covariance modeling while maintaining robustness to spatiotemporal noise. Hybrid Spatial Encoding, employing a Cartesian and Vector dual-stream strategy, effectively utilizes skeletal information in the form of joint and bone features. The proposed method achieves 92.1% Top-1 accuracy with real-time inference on the benchmark, outperforming baseline by +5.9% accuracy with only 10% FLOPs. Cross-dataset experiments confirm the superiority of our method in robustness.

---

## 104. Weighted Contrastive Learning for Anomaly-Aware Time-Series Forecasting

**论文链接:** [http://arxiv.org/abs/2512.07569v1](http://arxiv.org/abs/2512.07569v1)

**作者:** Joel Ekstrand, Tor Mattsson, Zahra Taghiyarrenani, Slawomir Nowaczyk, Jens Lundström, Mikael Lindén

**发布时间:** 2025-12-08

### GPT解析

### 总结

提出了一种名为加权对比适应（WECA）的方法，用于在异常条件下提高多元时间序列预测的可靠性，同时保持正常情况下的性能。

### 背景

在异常条件下对多元时间序列进行可靠预测在应用中至关重要，例如ATM现金物流，其中突然的需求变化可能会扰乱运营。现代深度预测器在正常数据上实现高精度，但在分布变化发生时往往会失败。

### 目的

开发一种能够在异常条件下提高预测可靠性的方法，同时保持正常情况下的性能。

### 方法

提出了加权对比适应（WECA），这是一种加权对比目标，使正常和异常增强表示保持一致，在保持良性变化一致性的同时保留异常相关信息。

### 主要发现

在全国ATM交易数据集上的评估显示，与正常训练的基线相比，WECA在受异常影响的数据上的SMAPE提高了6.1个百分点，而在正常数据上的性能下降可以忽略不计。

### 结论

WECA增强了异常情况下的预测可靠性，同时没有牺牲常规运营期间的性能。

### 翻译

在异常条件下对多元时间序列进行可靠预测在应用中至关重要，例如ATM现金物流，其中突然的需求变化可能会扰乱运营。现代深度预测器在正常数据上实现高精度，但在分布变化发生时往往会失败。我们提出了加权对比适应（WECA），这是一种加权对比目标，使正常和异常增强表示保持一致，在保持良性变化一致性的同时保留异常相关信息。在全国ATM交易数据集上的评估显示，与正常训练的基线相比，WECA在受异常影响的数据上的SMAPE提高了6.1个百分点，而在正常数据上的性能下降可以忽略不计。这些结果表明，WECA增强了异常情况下的预测可靠性，同时没有牺牲常规运营期间的性能。


### 论文摘要

Reliable forecasting of multivariate time series under anomalous conditions is crucial in applications such as ATM cash logistics, where sudden demand shifts can disrupt operations. Modern deep forecasters achieve high accuracy on normal data but often fail when distribution shifts occur. We propose Weighted Contrastive Adaptation (WECA), a Weighted contrastive objective that aligns normal and anomaly-augmented representations, preserving anomaly-relevant information while maintaining consistency under benign variations. Evaluations on a nationwide ATM transaction dataset with domain-informed anomaly injection show that WECA improves SMAPE on anomaly-affected data by 6.1 percentage points compared to a normally trained baseline, with negligible degradation on normal data. These results demonstrate that WECA enhances forecasting reliability under anomalies without sacrificing performance during regular operations.

---

## 105. DGGAN: Degradation Guided Generative Adversarial Network for Real-time Endoscopic Video Enhancement

**论文链接:** [http://arxiv.org/abs/2512.07253v1](http://arxiv.org/abs/2512.07253v1)

**作者:** Handing Xu, Zhenguo Nie, Tairan Peng, Huimin Pan, Xin-Jun Liu

**发布时间:** 2025-12-08

**备注:** 18 pages, 8 figures, and 7 tables

### GPT解析

### 总结

该研究提出了一种退化感知框架用于内窥镜视频增强，通过跨帧传播退化表示实现实时高质量增强，解决了现有方法计算需求过高的问题。

### 背景

内窥镜手术依赖于术中视频，图像质量是手术安全和有效性的决定性因素。然而，内窥镜视频常因不均匀照明、组织散射、遮挡和运动模糊而降质，这会掩盖重要解剖细节并增加手术难度。

### 目的

解决现有深度学习方法计算需求过高，无法满足实时手术使用的问题，实现实时高质量的内窥镜视频增强。

### 方法

提出一种退化感知框架，使用对比学习从图像中提取退化表示，引入融合机制用这些表示调制图像特征指导单帧增强模型，并通过循环一致性约束训练模型以提高鲁棒性和泛化能力。

### 主要发现

实验证明，该框架在性能和效率之间取得了优于现有最先进方法的平衡，退化感知建模对实时内窥镜视频增强非常有效。

### 结论

隐式学习和传播退化表示为临床应用提供了实用的途径。

### 翻译

内窥镜手术依赖于术中视频，这使得图像质量成为手术安全和有效性的决定性因素。然而，内窥镜视频常因不均匀照明、组织散射、遮挡和运动模糊而降质，这掩盖了关键的解剖细节并使手术操作复杂化。尽管基于深度学习方法在图像增强方面显示出潜力，但大多数现有方法的计算需求仍然过高，无法满足实时手术使用的需求。为应对这一挑战，我们提出了一种用于内窥镜视频增强的退化感知框架，通过跨帧传播退化表示实现实时高质量增强。在我们的框架中，退化表示首先使用对比学习从图像中提取。然后我们引入了一种融合机制，用这些表示调制图像特征，指导单帧增强模型，该模型通过退化图像和恢复图像之间的循环一致性约束进行训练，以提高鲁棒性和泛化能力。实验证明，与几种最先进的方法相比，我们的框架在性能和效率之间取得了更好的平衡。这些结果突显了退化感知建模在实时内窥镜视频增强中的有效性。尽管如此，我们的方法表明，隐式学习和传播退化表示为临床应用提供了实用的途径。


### 论文摘要

Endoscopic surgery relies on intraoperative video, making image quality a decisive factor for surgical safety and efficacy. Yet, endoscopic videos are often degraded by uneven illumination, tissue scattering, occlusions, and motion blur, which obscure critical anatomical details and complicate surgical manipulation. Although deep learning-based methods have shown promise in image enhancement, most existing approaches remain too computationally demanding for real-time surgical use. To address this challenge, we propose a degradation-aware framework for endoscopic video enhancement, which enables real-time, high-quality enhancement by propagating degradation representations across frames. In our framework, degradation representations are first extracted from images using contrastive learning. We then introduce a fusion mechanism that modulates image features with these representations to guide a single-frame enhancement model, which is trained with a cycle-consistency constraint between degraded and restored images to improve robustness and generalization. Experiments demonstrate that our framework achieves a superior balance between performance and efficiency compared with several state-of-the-art methods. These results highlight the effectiveness of degradation-aware modeling for real-time endoscopic video enhancement. Nevertheless, our method suggests that implicitly learning and propagating degradation representation offer a practical pathway for clinical application.

---

## 106. Self-Supervised Learning on Molecular Graphs: A Systematic Investigation of Masking Design

**论文链接:** [http://arxiv.org/abs/2512.07064v1](http://arxiv.org/abs/2512.07064v1)

**作者:** Jiannan Yang, Veronika Thost, Tengfei Ma

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究通过统一概率框架分析分子表示学习中的自监督学习，揭示掩码策略设计选择的有效性，为开发更有效的分子图自监督学习方法提供指导。

### 背景

自监督学习在分子表示学习中起核心作用，但许多基于掩码的预训练创新被作为启发式方法引入，缺乏原则性评估，不清楚哪些设计选择真正有效。

### 目的

将整个预训练-微调工作流程统一到概率框架中，实现掩码策略的透明比较和更深入的理解。

### 方法

在严格控制环境下研究三个核心设计维度：掩码分布、预测目标和编码器架构；使用信息论度量评估预训练信号的信息量，并将其与下游性能联系起来。

### 主要发现

对于常见节点级预测任务，复杂掩码分布并不比均匀采样带来一致好处；预测目标的选择及其与编码器架构的协同作用更为关键；转向语义更丰富的预测目标可带来显著下游性能提升，特别是与图Transformer编码器配对时。

### 结论

这些见解为开发更有效的分子图自监督学习方法提供了实用指导。

### 翻译

自监督学习在分子表示学习中起着核心作用。然而，许多最近基于掩码的预训练创新被作为启发式方法引入，缺乏原则性评估，模糊了哪些设计选择真正有效。这项工作将整个预训练-微调工作流程统一到一个概率框架中，实现了掩码策略的透明比较和更深入的理解。基于这一形式化，我们在严格控制的设置下对三个核心设计维度进行了对照研究：掩码分布、预测目标和编码器架构。我们进一步采用信息论度量来评估预训练信号的信息量，并将其与经验基准的下游性能联系起来。我们的发现揭示了一个令人惊讶的见解：对于常见的节点级预测任务，复杂的掩码分布并不比均匀采样带来一致的好处。相反，预测目标的选择及其与编码器架构的协同作用要关键得多。特别是，转向语义更丰富的预测目标会带来显著的下游性能提升，特别是当与具有表现力的图Transformer编码器配对时。这些见解为开发更有效的分子图自监督方法提供了实用指导。


### 论文摘要

Self-supervised learning (SSL) plays a central role in molecular representation learning. Yet, many recent innovations in masking-based pretraining are introduced as heuristics and lack principled evaluation, obscuring which design choices are genuinely effective. This work cast the entire pretrain-finetune workflow into a unified probabilistic framework, enabling a transparent comparison and deeper understanding of masking strategies. Building on this formalism, we conduct a controlled study of three core design dimensions: masking distribution, prediction target, and encoder architecture, under rigorously controlled settings. We further employ information-theoretic measures to assess the informativeness of pretraining signals and connect them to empirically benchmarked downstream performance. Our findings reveal a surprising insight: sophisticated masking distributions offer no consistent benefit over uniform sampling for common node-level prediction tasks. Instead, the choice of prediction target and its synergy with the encoder architecture are far more critical. Specifically, shifting to semantically richer targets yields substantial downstream improvements, particularly when paired with expressive Graph Transformer encoders. These insights offer practical guidance for developing more effective SSL methods for molecular graphs.

---

## 107. Selective Masking based Self-Supervised Learning for Image Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.06981v1](http://arxiv.org/abs/2512.06981v1)

**作者:** Yuemin Wang, Ian Stavness

**发布时间:** 2025-12-07

### GPT解析

### 总结

本文提出了一种用于语义分割的新型自监督学习方法，使用选择性掩码图像重建作为预训练任务。该方法替代了大多数掩码图像建模预训练方法中使用的随机掩码增强，通过迭代步骤选择重建损失最高的图像块进行掩码，利用已训练模型的知识。

### 背景

大多数掩码图像建模预训练方法使用随机掩码增强，而语义分割任务通常需要大量标注数据。

### 目的

提出一种新的自监督学习方法用于语义分割，使用选择性掩码图像重建作为预训练任务，以提高下游分割性能。

### 方法

提出的选择性掩码方法替代了随机掩码增强，通过迭代步骤选择重建损失最高的图像块进行掩码，利用已训练模型的知识进行图像重建预训练。

### 主要发现

在Pascal VOC和Cityscapes两个通用数据集以及Nassar 2020和Sugarbeets 2016两个杂草分割数据集上，选择性掩码方法比传统随机掩码方法和监督ImageNet预训练在下游分割精度上分别提高了2.9%和2.5%。此外，该方法显著提高了表现最差类别的准确率，并且使用相同的预训练和下游数据集对于低预算自监督预训练效果最好。

### 结论

选择性掩码图像重建方法为改进端到端语义分割工作流程提供了有效且实用的解决方案，特别适用于需要有限模型容量以满足推理速度和计算资源要求的情况。

### 翻译

本文提出了一种用于语义分割的新型自监督学习方法，使用选择性掩码图像重建作为预训练任务。我们提出的方法替代了大多数掩码图像建模预训练方法中使用的随机掩码增强。提出的选择性掩码方法通过将图像重建预训练分解为迭代步骤，选择性地掩码重建损失最高的图像块，以利用已训练模型的知识。我们在两个通用数据集（Pascal VOC和Cityscapes）和两个杂草分割数据集（Nassar 2020和Sugarbeets 2016）上证明，我们提出的选择性掩码方法在下游分割精度上优于传统的随机掩码方法和监督ImageNet预训练，对于通用数据集提高了2.9%，对于杂草分割数据集提高了2.5%。此外，我们发现我们的选择性掩码方法显著提高了表现最差类别的准确率。最后，我们表明使用相同的预训练和下游数据集对于低预算自监督预训练能获得最佳结果。我们提出的选择性掩码图像重建方法为改进端到端语义分割工作流程提供了有效且实用的解决方案，特别适用于需要有限模型容量以满足推理速度和计算资源要求的情况。


### 论文摘要

This paper proposes a novel self-supervised learning method for semantic segmentation using selective masking image reconstruction as the pretraining task. Our proposed method replaces the random masking augmentation used in most masked image modelling pretraining methods. The proposed selective masking method selectively masks image patches with the highest reconstruction loss by breaking the image reconstruction pretraining into iterative steps to leverage the trained model's knowledge. We show on two general datasets (Pascal VOC and Cityscapes) and two weed segmentation datasets (Nassar 2020 and Sugarbeets 2016) that our proposed selective masking method outperforms the traditional random masking method and supervised ImageNet pretraining on downstream segmentation accuracy by 2.9% for general datasets and 2.5% for weed segmentation datasets. Furthermore, we found that our selective masking method significantly improves accuracy for the lowest-performing classes. Lastly, we show that using the same pretraining and downstream dataset yields the best result for low-budget self-supervised pretraining. Our proposed Selective Masking Image Reconstruction method provides an effective and practical solution to improve end-to-end semantic segmentation workflows, especially for scenarios that require limited model capacity to meet inference speed and computational resource requirements.

---

## 108. Patronus: Identifying and Mitigating Transferable Backdoors in Pre-trained Language Models

**论文链接:** [http://arxiv.org/abs/2512.06899v1](http://arxiv.org/abs/2512.06899v1)

**作者:** Tianhang Zhao, Wei Du, Haodong Zhao, Sufeng Duan, Gongshen Liu

**发布时间:** 2025-12-07

**备注:** Work in progress

### GPT解析

### 总结

该论文提出了Patronus框架，用于防御预训练语言模型中的可转移后门攻击，通过利用触发器对参数转移的输入侧不变性，结合多触发对比搜索算法和双阶段缓解策略，实现了高精度的后门检测和攻击防御。

### 背景

可转移后门对预训练语言模型供应链构成严重威胁，而现有防御研究处于初期阶段，主要依赖输出特征空间的异常检测，但存在微调修改模型参数导致防御失效的缺陷。

### 目的

解决下游任务微调导致的模型参数修改和输出分布变化问题，使防御方法能够适应参数变化并保持有效性。

### 方法

提出Patronus框架，利用触发器对参数转移的输入侧不变性；引入多触发对比搜索算法连接梯度优化与对比学习；采用双阶段缓解策略结合实时输入监控和对抗训练模型净化。

### 主要发现

在15个PLMs和10个任务上的实验表明，Patronus实现了≥98.7%的后门检测召回率，将攻击成功率降低到干净设置的水平，显著优于所有最先进的基线方法。

### 结论

Patronus通过创新的输入侧不变性方法和多触发对比搜索算法，有效解决了可转移后门在参数变化环境下的防御难题，为预训练语言模型的安全提供了可靠保障。

### 翻译

可转移后门对预训练语言模型(PLMs)供应链构成严重威胁，但防御研究仍处于初期阶段，主要依靠在输出特征空间中检测异常。我们确定了一个关键缺陷：在下游任务上进行微调会不可避免地修改模型参数，从而改变输出分布，使得预先计算的防御措施失效。为解决此问题，我们提出了Patronus，一个利用触发器对参数转移的输入侧不变性的新框架。为克服离散文本优化的收敛挑战，Patronus引入了多触发对比搜索算法，有效连接了基于梯度的优化与对比学习目标。此外，我们采用双阶段缓解策略，结合实时输入监控和通过对抗训练进行模型净化。在15个PLMs和10个任务上的广泛实验表明，Patronus实现了≥98.7%的后门检测召回率，并将攻击成功率降低到干净设置的水平，在所有设置中都显著优于所有最先进的基线方法。代码可在https://github.com/zth855/Patronus获取。


### 论文摘要

Transferable backdoors pose a severe threat to the Pre-trained Language Models (PLMs) supply chain, yet defensive research remains nascent, primarily relying on detecting anomalies in the output feature space. We identify a critical flaw that fine-tuning on downstream tasks inevitably modifies model parameters, shifting the output distribution and rendering pre-computed defense ineffective. To address this, we propose Patronus, a novel framework that use input-side invariance of triggers against parameter shifts. To overcome the convergence challenges of discrete text optimization, Patronus introduces a multi-trigger contrastive search algorithm that effectively bridges gradient-based optimization with contrastive learning objectives. Furthermore, we employ a dual-stage mitigation strategy combining real-time input monitoring with model purification via adversarial training. Extensive experiments across 15 PLMs and 10 tasks demonstrate that Patronus achieves $\geq98.7\%$ backdoor detection recall and reduce attack success rates to clean settings, significantly outperforming all state-of-the-art baselines in all settings. Code is available at https://github.com/zth855/Patronus.

---

## 109. CMV-Fuse: Cross Modal-View Fusion of AMR, Syntax, and Knowledge Representations for Aspect Based Sentiment Analysis

**论文链接:** [http://arxiv.org/abs/2512.06679v1](http://arxiv.org/abs/2512.06679v1)

**作者:** Smitha Muthya Sudheendra, Mani Deep Cherukuri, Jaideep Srivastava

**发布时间:** 2025-12-07

### GPT解析

### 总结

CMV-Fuse是一个跨模态视角融合框架，通过整合多种语言学视角来模拟人类语言处理过程，显著提升了基于方面的情感分析性能。

### 背景

自然语言理解需要整合从表层句法到深层语义再到世界知识的多种互补视角，而当前的情感分析系统通常只利用孤立的语言学视角，忽视了人类自然利用的结构表示之间的复杂相互作用。

### 目的

开发一个能够系统性地结合多种语言学视角的框架，以更接近人类处理语言的方式提升情感分析能力。

### 方法

提出CMV-Fuse框架，整合抽象意义表示、成分句法分析、依存句法和语义注意力四种语言学视角，并增强外部知识集成；通过分层门控注意力融合机制和结构感知多视角对比学习来捕捉精细结构模式和广泛上下文理解。

### 主要发现

在标准基准测试上，CMV-Fuse与强基线相比取得了实质性改进，分析揭示了每种语言学视角如何为更强大的情感分析做出贡献。

### 结论

通过系统性地整合多种互补的语言学视角，CMV-Fuse能够更全面地捕捉情感分析中的复杂模式，使情感分析系统更接近人类处理语言的方式。

### 翻译

自然语言理解本质上依赖于整合从表层句法到深层语义再到世界知识的多种互补视角。然而，当前的基于方面的情感分析系统通常利用孤立的语言学视角，从而忽视了人类自然利用的结构表示之间的复杂相互作用。我们提出了CMV-Fuse，一个跨模态视角融合框架，通过系统性地结合多种语言学视角来模拟人类语言处理过程。我们的方法系统性地编排四种语言学视角：抽象意义表示、成分句法分析、依存句法和语义注意力，并增强外部知识集成。通过在局部句法、中间语义和全局知识层面的分层门控注意力融合，CMV-Fuse能够捕捉精细的结构模式和广泛的上下文理解。一种新颖的结构感知多视角对比学习机制确保互补表示之间的一致性，同时保持计算效率。大量实验表明，在标准基准测试上，与强基线相比取得了实质性改进，分析揭示了每种语言学视角如何为更强大的情感分析做出贡献。


### 论文摘要

Natural language understanding inherently depends on integrating multiple complementary perspectives spanning from surface syntax to deep semantics and world knowledge. However, current Aspect-Based Sentiment Analysis (ABSA) systems typically exploit isolated linguistic views, thereby overlooking the intricate interplay between structural representations that humans naturally leverage. We propose CMV-Fuse, a Cross-Modal View fusion framework that emulates human language processing by systematically combining multiple linguistic perspectives. Our approach systematically orchestrates four linguistic perspectives: Abstract Meaning Representations, constituency parsing, dependency syntax, and semantic attention, enhanced with external knowledge integration. Through hierarchical gated attention fusion across local syntactic, intermediate semantic, and global knowledge levels, CMV-Fuse captures both fine-grained structural patterns and broad contextual understanding. A novel structure aware multi-view contrastive learning mechanism ensures consistency across complementary representations while maintaining computational efficiency. Extensive experiments demonstrate substantial improvements over strong baselines on standard benchmarks, with analysis revealing how each linguistic view contributes to more robust sentiment analysis.

---

## 110. Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts

**论文链接:** [http://arxiv.org/abs/2512.06652v1](http://arxiv.org/abs/2512.06652v1)

**作者:** Xiaolei Lu, Shamim Nemati

**发布时间:** 2025-12-07

### GPT解析

### 总结

这项工作提出了一种自适应测试时训练(AdaTTT)框架，通过自监督学习和部分最优传输技术，提高了ICU患者侵入性机械通气预测模型的泛化能力和鲁棒性。

### 背景

准确预测ICU患者是否需要侵入性机械通气对及时干预和资源分配至关重要。然而，不同机构的患者人群、临床实践和电子健康记录系统的差异导致领域偏移，降低了预测模型的泛化性能。

### 目的

引入自适应测试时训练(AdaTTT)框架，专门用于ICU环境中基于电子健康记录的侵入性机械通气预测，以提高模型在部署时的适应能力。

### 方法

推导测试时预测误差的信息论界限，引入自监督学习框架进行重建和掩码特征建模，通过动态掩码策略强调关键特征，并结合原型学习和部分最优传输技术提高对领域偏移的鲁棒性。

### 主要发现

在多中心ICU队列上的实验表明，该框架在不同的测试时适应基准上具有竞争力的分类性能。

### 结论

AdaTTT框架能够有效缓解领域偏移问题，提高ICU患者侵入性机械通气预测的准确性，为临床决策提供更可靠的工具。

### 翻译

准确预测重症监护室患者需要侵入性机械通气的需求对于及时干预和资源分配至关重要。然而，不同机构中患者人群、临床实践和电子健康记录系统的差异引入了领域偏移，导致预测模型在部署过程中的泛化性能下降。测试时训练已成为一种有前景的方法，通过在没有标记目标域数据的情况下在推理过程中动态适应模型来缓解此类偏移。在这项工作中，我们引入了自适应测试时训练(AdaTTT)，这是一种增强的TTT框架，专门用于ICU环境中基于EHR的IMV预测。我们首先推导测试时预测误差的信息论界限，并证明它受主任务和辅助任务之间不确定性的约束。为了增强它们的对齐，我们引入了一个具有预训练任务的自监督学习框架：重建和掩码特征建模，通过强调对主任务关键特征的动态掩码策略进行优化。此外，为了提高对领域偏移的鲁棒性，我们结合了原型学习，并采用部分最优传输(POT)进行灵活的部分特征对齐，同时保持临床上有意义的患者表示。在多中心ICU队列上的实验表明，在不同的测试时适应基准上具有竞争力的分类性能。


### 论文摘要

Accurate prediction of the need for invasive mechanical ventilation (IMV) in intensive care units (ICUs) patients is crucial for timely interventions and resource allocation. However, variability in patient populations, clinical practices, and electronic health record (EHR) systems across institutions introduces domain shifts that degrade the generalization performance of predictive models during deployment. Test-Time Training (TTT) has emerged as a promising approach to mitigate such shifts by adapting models dynamically during inference without requiring labeled target-domain data. In this work, we introduce Adaptive Test-Time Training (AdaTTT), an enhanced TTT framework tailored for EHR-based IMV prediction in ICU settings. We begin by deriving information-theoretic bounds on the test-time prediction error and demonstrate that it is constrained by the uncertainty between the main and auxiliary tasks. To enhance their alignment, we introduce a self-supervised learning framework with pretext tasks: reconstruction and masked feature modeling optimized through a dynamic masking strategy that emphasizes features critical to the main task. Additionally, to improve robustness against domain shifts, we incorporate prototype learning and employ Partial Optimal Transport (POT) for flexible, partial feature alignment while maintaining clinically meaningful patient representations. Experiments across multi-center ICU cohorts demonstrate competitive classification performance on different test-time adaptation benchmarks.

---

## 111. Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation

**论文链接:** [http://arxiv.org/abs/2512.06105v1](http://arxiv.org/abs/2512.06105v1)

**作者:** Junwen Zheng, Xinran Xu, Li Rong Wang, Chang Cai, Lucinda Siyun Tan, Dingyuan Wang, Hong Liang Tey, Xiuyi Fan

**发布时间:** 2025-12-05

**备注:** AAAI-26-AIA

### GPT解析

### 总结

本文提出了一个名为CEFM的跨模态可解释框架，用于黑色素瘤分类。该框架利用对比学习将临床诊断标准（ABC规则）映射到Vision Transformer嵌入空间，实现临床语义与视觉特征对齐，并通过自然语言生成生成结构化文本解释，提高了模型决策的透明度。

### 背景

深度学习在黑色素瘤分类中已展现出专家级性能，成为临床皮肤病学中的有力工具。然而，模型的不透明性和缺乏可解释性仍然是临床采用的主要障碍，临床医生难以信任黑盒模型的决策过程。

### 目的

解决深度学习模型在黑色素瘤分类中的可解释性问题，建立模型决策过程与临床医生理解之间的桥梁，提高临床医生对AI系统的信任度。

### 方法

提出CEFM框架，使用对比学习作为核心机制。具体包括：使用双投影头将临床诊断标准（ABC规则：不对称性、边界和颜色）映射到Vision Transformer嵌入空间；将临床语义与视觉特征对齐；通过自然语言生成将对齐表示转换为结构化文本解释；创建原始图像数据与临床解释之间的透明链接。

### 主要发现

在公共数据集上实现了92.79%的准确率和0.961的AUC；在多个可解释性指标上取得了显著改进；学习到的嵌入的空间排列与临床医生应用ABC规则的方式一致；有效连接了高性能分类与临床信任之间的差距。

### 结论

CEFM框架成功解决了黑色素瘤分类中深度学习模型的可解释性问题，通过将临床诊断标准与视觉特征对齐并生成结构化解释，提高了模型的透明度和临床医生的信任度，为深度学习在临床皮肤病学中的应用提供了新的可能性。

### 翻译

深度学习在黑色素瘤分类中已展现出专家级性能，使其成为临床皮肤病学中的有力工具。然而，模型的不透明性和缺乏可解释性仍然是临床采用的重大障碍，因为临床医生通常难以信任黑盒模型的决策过程。为解决这一差距，我们提出了一个用于黑色素瘤的跨模态可解释框架（CEFM），利用对比学习作为实现可解释性的核心机制。具体而言，CEFM使用双投影头将黑色素瘤诊断的临床标准（即不对称性、边界和颜色）映射到Vision Transformer嵌入空间，从而使临床语义与视觉特征对齐。对齐的表示随后通过自然语言生成转换为结构化文本解释，在原始图像数据和临床解释之间创建了透明链接。在公共数据集上的实验显示，准确率达到92.79%，AUC为0.961，同时在多个可解释性指标上取得了显著改进。定性分析进一步表明，学习到的嵌入的空间排列与临床医生应用ABC规则的方式一致，有效连接了高性能分类与临床信任之间的差距。


### 论文摘要

Deep learning has demonstrated expert-level performance in melanoma classification, positioning it as a powerful tool in clinical dermatology. However, model opacity and the lack of interpretability remain critical barriers to clinical adoption, as clinicians often struggle to trust the decision-making processes of black-box models. To address this gap, we present a Cross-modal Explainable Framework for Melanoma (CEFM) that leverages contrastive learning as the core mechanism for achieving interpretability. Specifically, CEFM maps clinical criteria for melanoma diagnosis-namely Asymmetry, Border, and Color (ABC)-into the Vision Transformer embedding space using dual projection heads, thereby aligning clinical semantics with visual features. The aligned representations are subsequently translated into structured textual explanations via natural language generation, creating a transparent link between raw image data and clinical interpretation. Experiments on public datasets demonstrate 92.79% accuracy and an AUC of 0.961, along with significant improvements across multiple interpretability metrics. Qualitative analyses further show that the spatial arrangement of the learned embeddings aligns with clinicians' application of the ABC rule, effectively bridging the gap between high-performance classification and clinical trust.

---

## 112. Physics-Guided Deepfake Detection for Voice Authentication Systems

**论文链接:** [http://arxiv.org/abs/2512.06040v1](http://arxiv.org/abs/2512.06040v1)

**作者:** Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

**发布时间:** 2025-12-04

### GPT解析

### 总结

该论文提出了一种结合物理引导的深度伪造检测和不确定性感知边缘学习的框架，用于应对网络边缘语音认证系统面临的深度伪造攻击和控制平面投毒威胁。

### 背景

网络边缘部署的语音认证系统面临双重威胁：复杂的深度伪造合成攻击和分布式联邦学习协议中的控制平面投毒。

### 目的

开发一个能够同时抵御深度伪造攻击和控制平面投毒的语音认证框架，解决网络语音认证的完整威胁模型。

### 方法

提出一个框架，融合可解释的物理特征（建模声道动力学）和自监督学习模块的表示，通过多模态集成架构处理，并使用贝叶斯集成提供不确定性估计。

### 主要发现

结合基于物理的音频样本特性评估和不确定性估计，使框架能够对先进的深度伪造攻击和复杂的控制平面投毒保持鲁棒性。

### 结论

所提出的框架有效解决了网络语音认证系统面临的完整威胁模型，提高了安全性。

### 翻译

在网络边缘部署的语音认证系统面临双重威胁：a) 复杂的深度伪造合成攻击和 b) 分布式联邦学习协议中的控制平面投毒。我们提出了一个将物理引导的深度伪造检测与边缘学习中的不确定性感知相结合的框架。该框架融合了建模声道动力学的可解释物理特征和来自自监督学习模块的表示。然后这些表示通过多模态集成架构处理，接着是提供不确定性估计的贝叶斯集成。纳入基于物理的音频样本特性评估和不确定性估计，使我们提出的框架能够对先进的深度伪造攻击和复杂的控制平面投毒保持鲁棒性，解决了网络语音认证的完整威胁模型。


### 论文摘要

Voice authentication systems deployed at the network edge face dual threats: a) sophisticated deepfake synthesis attacks and b) control-plane poisoning in distributed federated learning protocols. We present a framework coupling physics-guided deepfake detection with uncertainty-aware in edge learning. The framework fuses interpretable physics features modeling vocal tract dynamics with representations coming from a self-supervised learning module. The representations are then processed via a Multi-Modal Ensemble Architecture, followed by a Bayesian ensemble providing uncertainty estimates. Incorporating physics-based characteristics evaluations and uncertainty estimates of audio samples allows our proposed framework to remain robust to both advanced deepfake attacks and sophisticated control-plane poisoning, addressing the complete threat model for networked voice authentication.

---

## 113. Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2512.07760v1](http://arxiv.org/abs/2512.07760v1)

**作者:** Menglin Wang, Xiaojin Gong, Jiachen Li, Genlin Ji

**发布时间:** 2025-12-08

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该论文提出了一种解决无监督可见光-红外人员重识别中跨模态关联挑战的方法，通过模态感知Jaccard距离和'分割-对比'策略实现了更可靠的跨模态关联和模态不变表示学习。

### 背景

无监督可见光-红外人员重识别(USVI-ReID)旨在不依赖任何标注的情况下匹配可见光和红外摄像头中的同一个人。由于可见光和红外模态之间存在显著差异，估计可靠的跨模态关联成为主要挑战。

### 目的

通过挖掘和关注可见光-红外模态偏差，从两个方面解决跨模态学习：偏差缓解的全局关联和模态不变表示学习。

### 方法

提出模态感知Jaccard距离来缓解模态差异引起的距离偏差，并通过全局聚类估计更可靠的跨模态关联；设计'分割-对比'策略获得模态特定的全局原型，在全局关联指导下对这些原型进行对齐，实现模态不变但具有ID辨别力的表示学习。

### 主要发现

尽管方法概念简单，但在基准VI-ReID数据集上获得了最先进的性能，显著优于现有方法。

### 结论

该方法通过解决模态偏差和改进跨模态关联，有效提升了无监督可见光-红外人员重识别的性能。

### 翻译

无监督可见光-红外人员重识别(USVI-ReID)旨在不依赖任何标注的情况下匹配可见光和红外摄像头中的同一个人。鉴于可见光和红外模态之间存在显著差异，估计可靠的跨模态关联成为USVI-ReID中的主要挑战。现有方法通常采用最优传输来关联模态内聚类，这容易传播局部聚类错误，同时忽略了全局实例级关系。通过挖掘和关注可见光-红外模态偏差，本文从两个方面解决跨模态学习：偏差缓解的全局关联和模态不变表示学习。受单模态重识别中摄像头感知距离校正的启发，我们提出了模态感知Jaccard距离来缓解由模态差异引起的距离偏差，从而可以通过全局聚类估计更可靠的跨模态关联。为进一步改进跨模态表示学习，设计了'分割-对比'策略来获得模态特定的全局原型。通过在全局关联指导下显式对这些原型进行对齐，可以实现模态不变但具有ID辨别力的表示学习。虽然方法概念简单，但在基准VI-ReID数据集上获得了最先进的性能，并以显著优势优于现有方法，验证了其有效性。


### 论文摘要

Unsupervised visible-infrared person re-identification (USVI-ReID) aims to match individuals across visible and infrared cameras without relying on any annotation. Given the significant gap across visible and infrared modality, estimating reliable cross-modality association becomes a major challenge in USVI-ReID. Existing methods usually adopt optimal transport to associate the intra-modality clusters, which is prone to propagating the local cluster errors, and also overlooks global instance-level relations. By mining and attending to the visible-infrared modality bias, this paper focuses on addressing cross-modality learning from two aspects: bias-mitigated global association and modality-invariant representation learning. Motivated by the camera-aware distance rectification in single-modality re-ID, we propose modality-aware Jaccard distance to mitigate the distance bias caused by modality discrepancy, so that more reliable cross-modality associations can be estimated through global clustering. To further improve cross-modality representation learning, a `split-and-contrast' strategy is designed to obtain modality-specific global prototypes. By explicitly aligning these prototypes under global association guidance, modality-invariant yet ID-discriminative representation learning can be achieved. While conceptually simple, our method obtains state-of-the-art performance on benchmark VI-ReID datasets and outperforms existing methods by a significant margin, validating its effectiveness.

---

## 114. Mapping Still Matters: Coarse-Graining with Machine Learning Potentials

**论文链接:** [http://arxiv.org/abs/2512.07692v1](http://arxiv.org/abs/2512.07692v1)

**作者:** Franz Görlich, Julija Zavadlav

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究系统探讨了映射选择如何影响等变机器学习势能(MLPs)学习到的表示，为构建可转移的粗粒度(CG)模型提供了实用指导。

### 背景

粗粒度建模使分子模拟能够达到全原子方法无法实现的时间和长度尺度。对于经典CG模型，映射选择是准确性和可转移性的主要决定因素。机器学习势能的出现为构建能够学习任何映射的真实平均力势的CG模型提供了新机会。

### 目的

系统研究映射选择如何影响等变MLP学习到的表示

### 方法

通过研究液态己烷、氨基酸和多聚丙氨酸来系统调查映射选择对等变MLP表示学习的影响

### 主要发现

当键合和非键相互作用的长度尺度重叠时，可能会出现非物理键置换；正确编码物种和保持立体化学至关重要，因为忽略任何一个都会引入非物理对称性

### 结论

研究结果为选择与现代架构兼容的CG映射提供了实用指导，指导可转移CG模型的发展

### 翻译

粗粒度(CG)建模使分子模拟能够达到全原子方法无法实现的时间和长度尺度。对于经典CG模型，映射的选择，即原子如何分组到CG位点，是准确性和可转移性的主要决定因素。同时，机器学习势能(MLPs)的出现为构建CG模型提供了新机会，理论上可以学习任何映射的真实平均力势。在本工作中，我们通过研究液态己烷、氨基酸和多聚丙氨酸，系统调查了映射选择如何影响等变MLP学习到的表示。我们发现，当键合和非键相互作用的长度尺度重叠时，可能会出现非物理键置换。我们还证明，正确编码物种和保持立体化学至关重要，因为忽略任何一个都会引入非物理对称性。我们的研究结果为选择与现代架构兼容的CG映射提供了实用指导，并指导了可转移CG模型的发展。


### 论文摘要

Coarse-grained (CG) modeling enables molecular simulations to reach time and length scales inaccessible to fully atomistic methods. For classical CG models, the choice of mapping, that is, how atoms are grouped into CG sites, is a major determinant of accuracy and transferability. At the same time, the emergence of machine learning potentials (MLPs) offers new opportunities to build CG models that can in principle learn the true potential of the mean force for any mapping. In this work, we systematically investigate how the choice of mapping influences the representations learned by equivariant MLPs by studying liquid hexane, amino acids, and polyalanine. We find that when the length scales of bonded and nonbonded interactions overlap, unphysical bond permutations can occur. We also demonstrate that correctly encoding species and maintaining stereochemistry are crucial, as neglecting either introduces unphysical symmetries. Our findings provide practical guidance for selecting CG mappings compatible with modern architectures and guide the development of transferable CG models.

---

## 115. Multi-Domain Motion Embedding: Expressive Real-Time Mimicry for Legged Robots

**论文链接:** [http://arxiv.org/abs/2512.07673v1](http://arxiv.org/abs/2512.07673v1)

**作者:** Matthias Heyrman, Chenhao Li, Victor Klemm, Dongho Kang, Stelian Coros, Marco Hutter

**发布时间:** 2025-12-08

**备注:** 15 pages

### GPT解析

### 总结

MDME是一种统一结构化和非结构化特征嵌入的运动表示方法，使用基于小波的编码器和概率嵌入，能够从最小输入集生成丰富的参考运动表示，实现跨多样运动风格和形态的改进泛化，并在人形和四足平台上实现了复杂轨迹的准确实时再现，无需任务特定调整或在线重定位。

### 背景

有效的运动表示对实现机器人实时模仿表达性行为至关重要，但现有运动控制器通常忽略运动中的固有模式。之前的表示学习方法没有尝试同时捕捉人类和动物运动中的结构化周期性模式和非常规变化。

### 目的

开发一种能够统一结构化和非结构化特征嵌入的运动表示方法，以捕捉运动中的结构化周期性模式和非常规变化，从而实现机器人对复杂运动的实时、准确模仿。

### 方法

提出多域运动嵌入(MDME)，使用基于小波的编码器和概率嵌入并行地统一结构化和非结构化特征的嵌入，生成丰富的参考运动表示。通过在机器人控制策略上使用学习到的嵌入来评估MDME在无重定位的实时运动模仿方面的表现。

### 主要发现

MDME在重建保真度和对未见运动的泛化能力方面优于先前的方法；MDME可以在人形和四足平台上准确再现复杂轨迹；MDME可以通过零样本部署实时再现新颖的运动风格；无需任务特定调整或在线重定位即可实现高质量运动模仿。

### 结论

MDME是一种可扩展的实时机器人模仿的通用且结构感知的基础，能够从最小输入集生成丰富的运动表示，实现跨多样运动风格和形态的改进泛化。

### 翻译

有效的运动表示对于使机器人能够实时模仿表达性行为至关重要，然而现有的运动控制器通常忽略运动中的固有模式。之前在表示学习方面的努力没有尝试同时捕捉人类和动物运动中的结构化周期性模式和非常规变化。为此，我们提出了多域运动嵌入(MDME)，这是一种运动表示，它使用基于小波的编码器和概率嵌入并行地统一结构化和非结构化特征的嵌入。这从最小输入集生成了参考运动的丰富表示，实现了跨多样运动风格和形态的改进泛化。我们通过在机器人控制策略上以学习到的嵌入为条件，在无重定位的实时运动模仿方面评估了MDME，展示了在人形和四足平台上复杂轨迹的准确再现。我们的比较研究证实，MDME在重建保真度和对未见运动的泛化能力方面优于先前的方法。此外，我们证明了MDME可以通过零样本部署实时再现新颖的运动风格，消除了任务特定调整或在线重定位的需要。这些结果将MDME定位为可扩展的实时机器人模仿的通用且结构感知的基础。


### 论文摘要

Effective motion representation is crucial for enabling robots to imitate expressive behaviors in real time, yet existing motion controllers often ignore inherent patterns in motion. Previous efforts in representation learning do not attempt to jointly capture structured periodic patterns and irregular variations in human and animal movement. To address this, we present Multi-Domain Motion Embedding (MDME), a motion representation that unifies the embedding of structured and unstructured features using a wavelet-based encoder and a probabilistic embedding in parallel. This produces a rich representation of reference motions from a minimal input set, enabling improved generalization across diverse motion styles and morphologies. We evaluate MDME on retargeting-free real-time motion imitation by conditioning robot control policies on the learned embeddings, demonstrating accurate reproduction of complex trajectories on both humanoid and quadruped platforms. Our comparative studies confirm that MDME outperforms prior approaches in reconstruction fidelity and generalizability to unseen motions. Furthermore, we demonstrate that MDME can reproduce novel motion styles in real-time through zero-shot deployment, eliminating the need for task-specific tuning or online retargeting. These results position MDME as a generalizable and structure-aware foundation for scalable real-time robot imitation.

---

## 116. Dual-Stream Cross-Modal Representation Learning via Residual Semantic Decorrelation

**论文链接:** [http://arxiv.org/abs/2512.07568v1](http://arxiv.org/abs/2512.07568v1)

**作者:** Xuecheng Li, Weikuan Jia, Alisher Kurbonaliev, Qurbonaliev Alisher, Khudzhamkulov Rustam, Ismoilov Shuhratjon, Eshmatov Javhariddin, Yuanjie Zheng

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了一种双流残差语义解相关网络(DSRSD-Net)，用于解决多模态学习中的模态主导、冗余信息耦合和虚假跨模态相关性问题，通过解耦模态特定和模态共享信息提高模型的可解释性和鲁棒性。

### 背景

跨模态学习已成为整合图像、文本和结构化属性等多源异构信息的基本范式，但多模态表示常面临模态主导、冗余信息耦合和虚假跨模态相关性等问题，导致泛化能力差且可解释性有限，特别是高方差模态会掩盖较弱但语义重要的信号。

### 目的

解决多模态表示中的模态主导、冗余信息耦合和虚假跨模态相关性问题，提高模型泛化能力和可解释性，使模型能够理解哪些模态实际驱动预测，并保持当某些模态有噪声或缺失时的鲁棒性。

### 方法

提出双流残差语义解相关网络(DSRSD-Net)，包含三个主要组件：(1)双流表示学习模块通过残差投影分离模态内(私有)和模态间(共享)的潜在因子；(2)残差语义对齐头使用对比和回归风格目标的组合将不同模态的共享因子映射到公共空间；(3)解相关和正交性损失正则化共享空间的协方差结构，同时强制执行共享和私有流之间的正交性，从而抑制跨模态冗余并防止特征坍塌。

### 主要发现

在两个大规模教育基准测试上的实验结果表明，DSRSD-Net在后续步骤预测和最终结果预测方面持续优于强单模态、早期融合、晚期融合和协同注意力基线。

### 结论

DSRSD-Net是一种简单而有效的框架，通过解耦模态特定和模态共享信息，成功解决了多模态学习中的关键挑战，提高了模型的可解释性和鲁棒性，并在预测任务上表现优于多种基线方法。

### 翻译

跨模态学习已成为整合图像、文本和结构化属性等多源异构信息的基本范式。然而，多模态表示常常遭受模态主导、冗余信息耦合和虚假跨模态相关性等问题，导致次优泛化和有限的解释性。特别是，高方差模态往往掩盖较弱但语义重要的信号，而简单的融合策略以不受控的方式纠缠模态共享和模态特定因子。这使得难以理解哪些模态实际驱动预测，并在某些模态有噪声或缺失时保持鲁棒性。为应对这些挑战，我们提出了一种双流残差语义解相关网络(DSRSD-Net)，这是一个简单而有效的框架，通过残差分解和显式语义解相关约束解耦模态特定和模态共享信息。DSRSD-Net引入：(1)双流表示学习模块，通过残差投影分离模态内(私有)和模态间(共享)的潜在因子；(2)残差语义对齐头，使用对比和回归风格目标的组合将不同模态的共享因子映射到公共空间；(3)解相关和正交性损失，正则化共享空间的协方差结构，同时强制执行共享和私有流之间的正交性，从而抑制跨模态冗余并防止特征坍塌。在两个大规模教育基准上的实验结果表明，DSRSD-Net在后续步骤预测和最终结果预测方面持续优于强单模态、早期融合、晚期融合和协同注意力基线。


### 论文摘要

Cross-modal learning has become a fundamental paradigm for integrating heterogeneous information sources such as images, text, and structured attributes. However, multimodal representations often suffer from modality dominance, redundant information coupling, and spurious cross-modal correlations, leading to suboptimal generalization and limited interpretability. In particular, high-variance modalities tend to overshadow weaker but semantically important signals, while naïve fusion strategies entangle modality-shared and modality-specific factors in an uncontrolled manner. This makes it difficult to understand which modality actually drives a prediction and to maintain robustness when some modalities are noisy or missing. To address these challenges, we propose a Dual-Stream Residual Semantic Decorrelation Network (DSRSD-Net), a simple yet effective framework that disentangles modality-specific and modality-shared information through residual decomposition and explicit semantic decorrelation constraints. DSRSD-Net introduces: (1) a dual-stream representation learning module that separates intra-modal (private) and inter-modal (shared) latent factors via residual projection; (2) a residual semantic alignment head that maps shared factors from different modalities into a common space using a combination of contrastive and regression-style objectives; and (3) a decorrelation and orthogonality loss that regularizes the covariance structure of the shared space while enforcing orthogonality between shared and private streams, thereby suppressing cross-modal redundancy and preventing feature collapse. Experimental results on two large-scale educational benchmarks demonstrate that DSRSD-Net consistently improves next-step prediction and final outcome prediction over strong single-modality, early-fusion, late-fusion, and co-attention baselines.

---

## 117. 论文ID: 2512.07309v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.07309v1.json'

---

## 118. RVLF: A Reinforcing Vision-Language Framework for Gloss-Free Sign Language Translation

**论文链接:** [http://arxiv.org/abs/2512.07273v1](http://arxiv.org/abs/2512.07273v1)

**作者:** Zhi Rao, Yucheng Zhou, Benjia Zhou, Yiqing Huang, Sergio Escalera, Jun Wan

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究提出了一种名为RVLF的三阶段强化视觉语言框架，解决了无词汇表手语翻译中的手语表示不足和句子级语义不匹配问题，结合了大视觉语言模型和强化学习技术，显著提高了翻译质量。

### 背景

无词汇表手语翻译面临两个关键挑战：手语表示不足无法捕捉细微的视觉线索，以及当前基于LLM的方法存在句子级语义不匹配问题，限制了翻译质量。

### 目的

解决手语表示不足和句子级语义不匹配问题，提高无词汇表手语翻译的质量。

### 方法

提出RVLF框架，构建专门为手语设计的大视觉语言模型，结合强化学习；引入语义表示学习机制融合基于骨架的运动线索和DINOv2提取的视觉特征；使用指令微调获得SLT-SFT基线；引入基于GRPO的优化策略，结合BLEU和ROUGE奖励函数微调模型得到SLT-GRPO。

### 主要发现

RVLF框架在多个数据集上显著提高了BLEU-4分数：CSL-Daily(+5.1)、PHOENIX-2014T(+1.11)、How2Sign(+1.4)和OpenASL(+1.61)；首次将GRPO引入手语翻译；实验验证了GRPO优化在提高翻译质量和语义一致性方面的有效性。

### 结论

RVLF框架通过专门设计的手语视觉语言模型和基于GRPO的强化学习优化，有效解决了无词汇表手语翻译中的关键挑战，显著提高了翻译质量，无需在大型外部手语数据集上进行预训练。

### 翻译

无词汇表手语翻译（SLT）受到两个关键挑战的阻碍：手语表示不足无法捕捉细微的视觉线索，以及当前基于LLM的方法中的句子级语义不匹配限制了翻译质量。为解决这些问题，我们提出了一个三阶段的强化视觉语言框架（RVLF）。我们构建了一个专门为手语设计的大视觉语言模型，然后将其与强化学习结合，以自适应地提高翻译性能。首先，为了充分表示手语，RVLF引入了一种有效的语义表示学习机制，该机制融合了基于骨架的运动线索和通过DINOv2提取的语义丰富的视觉特征，随后通过指令微调获得强大的SLT-SFT基线。然后，为了改善句子级语义不匹配，我们引入了一种基于GRPO的优化策略，使用结合翻译忠实度和句子完整性的奖励函数对SLT-SFT模型进行微调，得到优化的SLT-GRPO模型。我们的概念上简单的框架在无词汇表手语翻译设置下取得了显著提升，无需在任何大型外部手语数据集上进行预训练，分别在多个数据集上提高了BLEU-4分数。据我们所知，这是首次将GRPO纳入手语翻译的研究。大量的实验和消融研究验证了基于GRPO的优化在提高翻译质量和语义一致性方面的有效性。


### 论文摘要

Gloss-free sign language translation (SLT) is hindered by two key challenges: **inadequate sign representation** that fails to capture nuanced visual cues, and **sentence-level semantic misalignment** in current LLM-based methods, which limits translation quality. To address these issues, we propose a three-stage **r**einforcing **v**ision-**l**anguage **f**ramework (**RVLF**). We build a large vision-language model (LVLM) specifically designed for sign language, and then combine it with reinforcement learning (RL) to adaptively enhance translation performance. First, for a sufficient representation of sign language, RVLF introduces an effective semantic representation learning mechanism that fuses skeleton-based motion cues with semantically rich visual features extracted via DINOv2, followed by instruction tuning to obtain a strong SLT-SFT baseline. Then, to improve sentence-level semantic misalignment, we introduce a GRPO-based optimization strategy that fine-tunes the SLT-SFT model with a reward function combining translation fidelity (BLEU) and sentence completeness (ROUGE), yielding the optimized model termed SLT-GRPO. Our conceptually simple framework yields substantial gains under the gloss-free SLT setting without pre-training on any external large-scale sign language datasets, improving BLEU-4 scores by +5.1, +1.11, +1.4, and +1.61 on the CSL-Daily, PHOENIX-2014T, How2Sign, and OpenASL datasets, respectively. To the best of our knowledge, this is the first work to incorporate GRPO into SLT. Extensive experiments and ablation studies validate the effectiveness of GRPO-based optimization in enhancing both translation quality and semantic consistency.

---

## 119. ReLKD: Inter-Class Relation Learning with Knowledge Distillation for Generalized Category Discovery

**论文链接:** [http://arxiv.org/abs/2512.07229v1](http://arxiv.org/abs/2512.07229v1)

**作者:** Fang Zhou, Zhiqiang Chen, Martin Pavlovski, Yizhong Zhang

**发布时间:** 2025-12-08

**备注:** Accepted to the Main Track of the 28th European Conference on Artificial Intelligence (ECAI 2025). To appear in the proceedings published by IOS Press (DOI: 10.3233/FAIA413)

### GPT解析

### 总结

论文提出了ReLKD框架，用于解决广义类别发现问题，通过有效利用隐含的类别间关系来增强新类别的分类能力。

### 背景

广义类别发现(GCD)面临在仅有已知类别标签的情况下，对包含已知和未知类别的未标记数据进行分类的挑战。先前研究通常将每个类别独立处理，忽视了类别间的内在关系，而直接获取这些关系在现实场景中非常困难。

### 目的

解决获取类别间关系的困难，提出一个能够有效利用隐含类别间关系并利用这些知识增强新类别分类的框架。

### 方法

ReLKD框架包含三个关键模块：目标粒度模块用于学习判别性表示，粗粒度模块用于捕获层次化的类别关系，蒸馏模块用于将粗粒度模块的知识转移到目标粒度模块以优化其表示学习。

### 主要发现

在四个数据集上的大量实验证明了ReLKD的有效性，特别是在标记数据有限的情况下表现突出。

### 结论

ReLKD框架成功地解决了广义类别发现中利用类别间关系的挑战，能够有效提升新类别的分类性能。

### 翻译

论文提出了ReLKD，一个端到端框架，能够有效利用隐含的类别间关系，并利用这些知识来增强新类别的分类。ReLKD包含三个关键模块：一个用于学习判别性表示的目标粒度模块，一个用于捕获层次化类别关系的粗粒度模块，以及一个用于将粗粒度模块知识转移到目标粒度模块以优化其表示学习的蒸馏模块。在四个数据集上的大量实验证明了ReLKD的有效性，特别是在标记数据有限的情况下。ReLKD的代码可在https://github.com/ZhouF-ECNU/ReLKD获取。


### 论文摘要

Generalized Category Discovery (GCD) faces the challenge of categorizing unlabeled data containing both known and novel classes, given only labels for known classes. Previous studies often treat each class independently, neglecting the inherent inter-class relations. Obtaining such inter-class relations directly presents a significant challenge in real-world scenarios. To address this issue, we propose ReLKD, an end-to-end framework that effectively exploits implicit inter-class relations and leverages this knowledge to enhance the classification of novel classes. ReLKD comprises three key modules: a target-grained module for learning discriminative representations, a coarse-grained module for capturing hierarchical class relations, and a distillation module for transferring knowledge from the coarse-grained module to refine the target-grained module's representation learning. Extensive experiments on four datasets demonstrate the effectiveness of ReLKD, particularly in scenarios with limited labeled data. The code for ReLKD is available at https://github.com/ZhouF-ECNU/ReLKD.

---

## 120. Dual Refinement Cycle Learning: Unsupervised Text Classification of Mamba and Community Detection on Text Attributed Graph

**论文链接:** [http://arxiv.org/abs/2512.07100v1](http://arxiv.org/abs/2512.07100v1)

**作者:** Hong Wang, Yinglong Zhang, Hanhan Guo, Xuewen Xia, Xing Xu

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了双重精炼周期学习(DRCL)框架，一种完全无监督的方法，用于解决预训练语言模型难以在实际文本属性网络中部署以及社区检测方法忽略文本语义的问题。

### 背景

预训练语言模型具有强大的文本理解能力，但由于严重依赖标记数据，难以在实际文本属性网络中部署。同时，社区检测方法通常忽略文本语义，限制了它们在内容组织、推荐和风险监控等下游应用中的实用性。

### 目的

提出一种完全无监督的框架，解决在没有标签或类别定义的实际场景中的问题，克服预训练语言模型和社区检测方法的局限性。

### 方法

提出双重精炼周期学习(DRCL)框架，通过预热初始化整合结构和语义信息，包含基于GCN的社区检测模块(GCN-CDM)和文本语义建模模块(TSMM)之间的双向精炼周期，两个模块迭代交换伪标签，使语义提示增强结构聚类，结构模式指导文本表示学习，无需人工监督。

### 主要发现

在多个文本属性图数据集上，DRCL持续改进发现社区的结构和语义质量；仅使用DRCL的社区信号训练的基于Mamba的分类器实现了与监督模型相当的准确性。

### 结论

DRCL展示了在标记数据稀缺或成本高昂的大规模系统中部署的潜力。

### 翻译

预训练语言模型提供强大的文本理解能力，但由于严重依赖标记数据，仍然难以在实际文本属性网络中部署。同时，社区检测方法通常忽略文本语义，限制了它们在内容组织、推荐和风险监控等下游应用中的实用性。为了克服这些限制，我们提出了双重精炼周期学习(DRCL)，这是一种完全无监督的框架，专为没有标签或类别定义的实际场景而设计。DRCL通过预热初始化以及基于GCN的社区检测模块(GCN-CDM)和文本语义建模模块(TSMM)之间的双向精炼周期来整合结构和语义信息。两个模块迭代交换伪标签，使语义提示能够增强结构聚类，结构模式能够指导文本表示学习，无需人工监督。在多个文本属性图数据集上，DRCL持续改进了发现社区的结构和语义质量。此外，仅从DRCL的社区信号训练的基于Mamba的分类器实现了与监督模型相当的准确性，证明了它在标记数据稀缺或成本高昂的大规模系统中部署的潜力。


### 论文摘要

Pretrained language models offer strong text understanding capabilities but remain difficult to deploy in real-world text-attributed networks due to their heavy dependence on labeled data. Meanwhile, community detection methods typically ignore textual semantics, limiting their usefulness in downstream applications such as content organization, recommendation, and risk monitoring. To overcome these limitations, we present Dual Refinement Cycle Learning (DRCL), a fully unsupervised framework designed for practical scenarios where no labels or category definitions are available.   DRCL integrates structural and semantic information through a warm-start initialization and a bidirectional refinement cycle between a GCN-based Community Detection Module (GCN-CDM) and a Text Semantic Modeling Module (TSMM). The two modules iteratively exchange pseudo-labels, allowing semantic cues to enhance structural clustering and structural patterns to guide text representation learning without manual supervision.   Across several text-attributed graph datasets, DRCL consistently improves the structural and semantic quality of discovered communities. Moreover, a Mamba-based classifier trained solely from DRCL's community signals achieves accuracy comparable to supervised models, demonstrating its potential for deployment in large-scale systems where labeled data are scarce or costly.

---

## 121. Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation

**论文链接:** [http://arxiv.org/abs/2512.06883v1](http://arxiv.org/abs/2512.06883v1)

**作者:** Zhongtao Rao, Peilin Zhou, Dading Chong, Zhiwei Chen, Shoujin Wang, Nan Tang

**发布时间:** 2025-12-07

### GPT解析

### 总结

该研究提出了一种名为SDA的轻量级框架，用于解决多模态推荐系统中的表示不对齐和梯度冲突问题，通过结合跨模态结构对齐和模态解耦适应两个组件，显著提升了推荐性能。

### 背景

多模态推荐通过利用视觉和文本信号提高准确性，其成功取决于学习高质量的跨模态表示。大型视觉语言模型（LVLMs）为跨模态表示学习提供了有前景的骨干网络，但将其应用于推荐系统仍面临挑战。

### 目的

解决LVLMs在推荐应用中的两个主要挑战：(1) 表示不对齐问题，即项目数据与通用预训练之间的领域差距导致嵌入空间不对齐；(2) 微调过程中的梯度冲突问题，即共享适配器导致干扰和判别能力不足。

### 方法

提出SDA框架，包含两个组件：1) 跨模态结构对齐（CMSA）：使用模内结构作为软教师对齐嵌入；2) 模态解耦适应（MoDA）：通过专业化的、门控的低秩路径解耦梯度流，缓解梯度冲突。

### 主要发现

在三个公开的Amazon数据集上，SDA能够无缝集成到现有的多模态和顺序推荐器中，在Hit@10上平均提升6.15%，在NDCG@10上平均提升8.64%。对于长尾项目，SDA实现了高达12.83%和18.70%的性能提升，同时具有最小的推理开销。

### 结论

SDA框架有效解决了LVLMs在推荐系统应用中的表示不对齐和梯度冲突问题，显著提升了推荐性能，特别是在处理长尾项目方面表现突出，且计算开销小。

### 翻译

多模态推荐通过利用视觉和文本信号提高准确性，其成功很大程度上取决于学习高质量的跨模态表示。最近大型视觉语言模型（LVLMs）的进展提供了统一的跨模态表示学习，使其成为有前景的骨干网络。然而，将LVLMs应用于推荐仍然面临挑战：(i) 表示不对齐，即项目数据与通用预训练之间的领域差距导致嵌入空间不对齐；(ii) 微调过程中的梯度冲突，即共享适配器导致干扰和判别能力不足。为解决这些问题，我们提出了SDA，一个用于结构化和解耦适应的轻量级框架，它集成了两个组件：跨模态结构对齐（CMSA）和模态解耦适应（MoDA）。CMSA使用模内结构作为软教师对齐嵌入，而MoDA通过专业化的、门控的低秩路径解耦梯度流来缓解梯度冲突。在三个公开的Amazon数据集上的实验表明，SDA能够无缝集成到现有的多模态和顺序推荐器中，在Hit@10上平均提升6.15%，在NDCG@10上平均提升8.64%。对于长尾项目，它还实现了高达12.83%和18.70%的性能提升，同时具有最小的推理开销。我们的代码和完整的实验结果可在https://github.com/RaoZhongtao/SDA获取。


### 论文摘要

Multimodal recommendation enhances accuracy by leveraging visual and textual signals, and its success largely depends on learning high-quality cross-modal representations. Recent advances in Large Vision-Language Models (LVLMs) offer unified multimodal representation learning, making them a promising backbone. However, applying LVLMs to recommendation remains challenging due to (i) representation misalignment, where domain gaps between item data and general pre-training lead to unaligned embedding spaces, and (ii) gradient conflicts during fine-tuning, where shared adapters cause interference and a lack of discriminative power. To address this, we propose SDA, a lightweight framework for Structural and Disentangled Adaptation, which integrates two components: Cross-Modal Structural Alignment (CMSA) and Modality-Disentangled Adaptation. CMSA aligns embeddings using intra-modal structures as a soft teacher, while MoDA mitigates gradient conflicts via expertized, gated low-rank paths to disentangle gradient flows. Experiments on three public Amazon datasets show SDA integrates seamlessly with existing multimodal and sequential recommenders, yielding average gains of 6.15% in Hit@10 and 8.64% in NDCG@10. It also achieves up to 12.83% and 18.70% gains on long-tail items with minimal inference overhead. Our code and full experimental results are available at https://github.com/RaoZhongtao/SDA.

---

## 122. Learning Invariant Graph Representations Through Redundant Information

**论文链接:** [http://arxiv.org/abs/2512.06154v1](http://arxiv.org/abs/2512.06154v1)

**作者:** Barproda Halder, Pasan Dissanayake, Sanghamitra Dutta

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究引入信息论中的部分信息分解(PID)工具，提出冗余引导的不变图学习(RIG)框架，通过最大化冗余信息并隔离虚假和因果子图，实现不同分布偏移下的OOD泛化。

### 背景

学习不变图表示用于OOD泛化具有挑战性，因为学习的表示往往保留虚假成分。

### 目的

解决不变表示学习中保留虚假成分的问题，通过引入PID超越经典信息论度量。

### 方法

提出名为RIG的多级优化框架，通过估计冗余信息的下界并最大化它以及附加目标，同时隔离虚假和因果子图。

### 主要发现

现有仅依赖经典信息论度量的方法存在局限性，需要专注于目标Y在虚假子图和不变子图间共享的冗余信息。

### 结论

在合成和真实图数据集上的实验证明了RIG框架的泛化能力。

### 翻译

学习不变图表示用于分布外(OOD)泛化仍然具有挑战性，因为学习的表示往往保留虚假成分。为应对这一挑战，本研究引入了信息论中的一个新工具——部分信息分解(PID)，它超越了经典信息论度量。我们识别出仅依赖经典信息论度量的现有不变表示学习方法的局限性，促使需要精确关注通过PID获得的虚假子图Gs和不变子图Gc之间关于目标Y的冗余信息。接下来，我们提出了一个名为冗余引导的不变图学习(RIG)的新多级优化框架，该框架最大化冗余信息，同时隔离虚假和因果子图，实现不同分布偏移下的OOD泛化。我们的方法依赖于估计冗余信息的下界（这本身需要优化）以及最大化它和其他附加目标。在合成和真实图数据集上的实验证明了我们提出的RIG框架的泛化能力。


### 论文摘要

Learning invariant graph representations for out-of-distribution (OOD) generalization remains challenging because the learned representations often retain spurious components. To address this challenge, this work introduces a new tool from information theory called Partial Information Decomposition (PID) that goes beyond classical information-theoretic measures. We identify limitations in existing approaches for invariant representation learning that solely rely on classical information-theoretic measures, motivating the need to precisely focus on redundant information about the target $Y$ shared between spurious subgraphs $G_s$ and invariant subgraphs $G_c$ obtained via PID. Next, we propose a new multi-level optimization framework that we call -- Redundancy-guided Invariant Graph learning (RIG) -- that maximizes redundant information while isolating spurious and causal subgraphs, enabling OOD generalization under diverse distribution shifts. Our approach relies on alternating between estimating a lower bound of redundant information (which itself requires an optimization) and maximizing it along with additional objectives. Experiments on both synthetic and real-world graph datasets demonstrate the generalization capabilities of our proposed RIG framework.

---

## 123. DisentangleFormer: Spatial-Channel Decoupling for Multi-Channel Vision

**论文链接:** [http://arxiv.org/abs/2512.04314v1](http://arxiv.org/abs/2512.04314v1)

**作者:** Jiashu Liao, Pietro Liò, Marc de Kamps, Duygu Sarikaya

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了DisentangleFormer架构，通过空间-通道解耦实现鲁棒的多通道视觉表示，解决了Vision Transformers在联合处理空间和通道维度时导致的纠缠表示问题。

### 背景

Vision Transformers存在基本限制：标准自注意力机制联合处理空间和通道维度，导致纠缠表示，妨碍对结构依赖和语义依赖的独立建模。这一问题在高光谱成像中尤为突出，其中通道捕获不同的生物物理或生化线索。

### 目的

提出DisentangleFormer架构，通过有原则的空间-通道解耦实现鲁棒的多通道视觉表示，以独立建模结构和语义线索。

### 方法

受信息论原理启发，采用并行设计，包含三个核心组件：(1)并行解耦：独立处理空间标记和通道标记流；(2)压缩标记增强器：自适应校准模块，动态融合空间和通道流；(3)多尺度FFN：通过多尺度局部上下文补充全局注意力。

### 主要发现

在多个高光谱基准测试上，DisentangleFormer实现了最先进性能，在Indian Pine、Pavia University、Houston和BigEarthNet等数据集上持续优于现有模型，同时在ImageNet上保持竞争力，计算成本降低17.8%。

### 结论

DisentangleFormer通过空间和通道解耦有效解决了Vision Transformers的基本限制，在高光谱成像任务中表现优异且计算效率更高。

### 翻译

视觉变换器面临一个基本限制：标准自注意力机制联合处理空间和通道维度，导致纠缠的表示，这妨碍了对结构依赖和语义依赖的独立建模。这个问题在高光谱成像中尤其突出，从卫星高光谱遥感到红外病理成像，其中通道捕获不同的生物物理或生化线索。我们提出了DisentangleFormer架构，通过有原则的空间-通道解耦实现鲁棒的多通道视觉表示。受解相关表示学习的信息论原理启发，我们的并行设计能够独立建模结构和语义线索，同时最小化空间和通道流之间的冗余。我们的设计整合了三个核心组件：(1)并行解耦：独立处理空间标记和通道标记流，实现跨空间和光谱维度的解相关特征学习；(2)压缩标记增强器：自适应校准模块，动态融合空间和通道流；(3)多尺度FFN：通过多尺度局部上下文补充全局注意力，以捕获细粒度的结构和语义依赖。在多个高光谱基准测试上的广泛实验表明，DisentangleFormer实现了最先进的性能，在Indian Pine、Pavia University和Houston等大型BigEarthNet遥感数据集以及红外病理数据集上持续优于现有模型。此外，它在ImageNet上保持竞争力的准确率，同时将计算成本降低了17.8%。代码将在接受后公开提供。


### 论文摘要

Vision Transformers face a fundamental limitation: standard self-attention jointly processes spatial and channel dimensions, leading to entangled representations that prevent independent modeling of structural and semantic dependencies. This problem is especially pronounced in hyperspectral imaging, from satellite hyperspectral remote sensing to infrared pathology imaging, where channels capture distinct biophysical or biochemical cues. We propose DisentangleFormer, an architecture that achieves robust multi-channel vision representation through principled spatial-channel decoupling. Motivated by information-theoretic principles of decorrelated representation learning, our parallel design enables independent modeling of structural and semantic cues while minimizing redundancy between spatial and channel streams. Our design integrates three core components: (1) Parallel Disentanglement: Independently processes spatial-token and channel-token streams, enabling decorrelated feature learning across spatial and spectral dimensions, (2) Squeezed Token Enhancer: An adaptive calibration module that dynamically fuses spatial and channel streams, and (3) Multi-Scale FFN: complementing global attention with multi-scale local context to capture fine-grained structural and semantic dependencies. Extensive experiments on hyperspectral benchmarks demonstrate that DisentangleFormer achieves state-of-the-art performance, consistently outperforming existing models on Indian Pine, Pavia University, and Houston, the large-scale BigEarthNet remote sensing dataset, as well as an infrared pathology dataset. Moreover, it retains competitive accuracy on ImageNet while reducing computational cost by 17.8% in FLOPs. The code will be made publicly available upon acceptance.

---

## 124. Benchmarking CXR Foundation Models With Publicly Available MIMIC-CXR and NIH-CXR14 Datasets

**论文链接:** [http://arxiv.org/abs/2512.06014v1](http://arxiv.org/abs/2512.06014v1)

**作者:** Jiho Shin, Dominic Marshall, Matthieu Komorowski

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究对两个医学图像基础模型在不同数据集上的表现进行了比较评估，发现它们各有优势，强调了标准化评估医学基础模型的必要性。

### 背景

最近的医学图像表示学习基础模型表现强劲，但它们在不同数据集上的比较行为尚未被充分探索。

### 目的

对两个大规模胸部X光(CXR)嵌入模型(CXR-Foundation (ELIXR v2.0)和MedImageInsight)在公共MIMIC-CR和NIH ChestX-ray14数据集上进行基准测试。

### 方法

使用统一的预处理流程和固定的下游分类器评估每个模型，确保可复现的比较；直接从预训练编码器提取嵌入，在多个疾病标签上训练轻量级LightGBM分类器，报告平均AUROC和F1分数及95%置信区间。

### 主要发现

MedImageInsight在大多数任务上实现了略高的性能，而CXR-Foundation表现出强大的跨数据集稳定性；对MedImageInsight嵌入的无监督聚类揭示了一致的疾病特定结构，与定量结果一致。

### 结论

结果强调了标准化评估医学基础模型的必要性，并为未来的多模态和临床集成研究建立了可复现的基线。

### 翻译

最近的基础模型在医学图像表示学习中展示了强大的性能，然而它们在不同数据集上的比较行为仍未被充分探索。这项工作在公共的MIMIC-CR和NIH ChestX-ray14数据集上对两个大规模胸部X光(CXR)嵌入模型(CXR-Foundation (ELIXR v2.0)和MedImageInsight)进行了基准测试。每个模型都使用统一的预处理流程和固定的下游分类器进行评估，以确保可复现的比较。我们直接从预训练编码器中提取嵌入，在多个疾病标签上训练轻量级LightGBM分类器，并报告了平均AUROC和F1分数及其95%置信区间。MedImageInsight在大多数任务上实现了略高的性能，而CXR-Foundation表现出强大的跨数据集稳定性。对MedImageInsight嵌入的无监督聚类进一步揭示了一致的疾病特定结构，与定量结果一致。这些结果强调了标准化评估医学基础模型的必要性，并为未来的多模态和临床集成研究建立了可复现的基线。


### 论文摘要

Recent foundation models have demonstrated strong performance in medical image representation learning, yet their comparative behaviour across datasets remains underexplored. This work benchmarks two large-scale chest X-ray (CXR) embedding models (CXR-Foundation (ELIXR v2.0) and MedImagelnsight) on public MIMIC-CR and NIH ChestX-ray14 datasets. Each model was evaluated using a unified preprocessing pipeline and fixed downstream classifiers to ensure reproducible comparison. We extracted embeddings directly from pre-trained encoders, trained lightweight LightGBM classifiers on multiple disease labels, and reported mean AUROC, and F1-score with 95% confidence intervals. MedImageInsight achieved slightly higher performance across most tasks, while CXR-Foundation exhibited strong cross-dataset stability. Unsupervised clustering of MedImageIn-sight embeddings further revealed a coherent disease-specific structure consistent with quantitative results. The results highlight the need for standardised evaluation of medical foundation models and establish reproducible baselines for future multimodal and clinical integration studies.

---

## 125. SpatialDreamer: Incentivizing Spatial Reasoning via Active Mental Imagery

**论文链接:** [http://arxiv.org/abs/2512.07733v1](http://arxiv.org/abs/2512.07733v1)

**作者:** Meng Cao, Xingyu Li, Xue Liu, Ian Reid, Xiaodan Liang

**发布时间:** 2025-12-08

### GPT解析

### 总结

论文提出了SpatialDreamer框架，通过强化学习和几何策略优化(GeoPO)提升多模态大语言模型在复杂空间推理任务上的表现，实现了类人主动空间心理模拟。

### 背景

多模态大语言模型在场景理解方面虽有进展，但在需要心理模拟的复杂空间推理任务上表现仍然有限。当前方法主要依赖被动观察空间数据，无法内化主动的心理意象过程。

### 目的

解决当前多模态大语言模型在空间推理任务上的局限性，实现类人主动空间心理模拟能力。

### 方法

提出SpatialDreamer强化学习框架，通过主动探索、世界模型视觉想象和基于证据的推理实现空间推理闭环过程；针对长程推理任务缺乏细粒度奖励监督的问题，提出几何策略优化(GeoPO)，引入树结构采样和具有几何一致性约束的步骤级奖励估计。

### 主要发现

广泛的实验表明SpatialDreamer在多个具有挑战性的基准测试中取得了极具竞争力的结果。

### 结论

SpatialDreamer代表了多模态大语言模型在类人主动空间心理模拟方面的关键进展。

### 翻译

尽管多模态大语言模型(MLLMs)在场景理解方面取得了进展，但在需要心理模拟的复杂空间推理任务上的表现仍然显著有限。当前方法通常依赖对空间数据的被动观察，无法内化主动的心理意象过程。为了弥补这一差距，我们提出了SpatialDreamer，一个强化学习框架，通过主动探索、通过世界模型进行视觉想象和基于证据的推理的闭环过程实现空间推理。为解决长程推理任务中缺乏细粒度奖励监督的问题，我们提出了几何策略优化(GeoPO)，它引入了树结构采样和具有几何一致性约束的步骤级奖励估计。大量实验表明，SpatialDreamer在多个具有挑战性的基准测试中取得了极具竞争力的结果，标志着MLLMs类人主动空间心理模拟的关键性进展。


### 论文摘要

Despite advancements in Multi-modal Large Language Models (MLLMs) for scene understanding, their performance on complex spatial reasoning tasks requiring mental simulation remains significantly limited. Current methods often rely on passive observation of spatial data, failing to internalize an active mental imagery process. To bridge this gap, we propose SpatialDreamer, a reinforcement learning framework that enables spatial reasoning through a closedloop process of active exploration, visual imagination via a world model, and evidence-grounded reasoning. To address the lack of fine-grained reward supervision in longhorizontal reasoning tasks, we propose Geometric Policy Optimization (GeoPO), which introduces tree-structured sampling and step-level reward estimation with geometric consistency constraints. Extensive experiments demonstrate that SpatialDreamer delivers highly competitive results across multiple challenging benchmarks, signifying a critical advancement in human-like active spatial mental simulation for MLLMs.

---

## 126. Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2512.07472v1](http://arxiv.org/abs/2512.07472v1)

**作者:** Siyu Xu, Zijian Wang, Yunke Wang, Chenghao Xia, Tao Huang, Chang Xu

**发布时间:** 2025-12-08

### GPT解析

### 总结

本研究提出了一种名为可供性场干预(AFI)的轻量级混合框架，通过3D空间可供性场(SAFs)作为插件来增强Vision-Language-Action(VLA)模型在分布变化场景下的鲁棒性，解决了VLA模型在场景变化时陷入'记忆陷阱'的问题。

### 背景

Vision-Language-Action (VLA) 模型在机器人操作中表现出色，能够直接将视觉观察和语言指令映射到动作。然而，这些模型在分布变化的情况下表现脆弱：当测试场景变化时，VLA往往会重现记忆的轨迹，而不是适应更新后的场景。

### 目的

解决VLA模型在分布变化场景下的脆弱性问题，防止模型陷入'记忆陷阱'，增强模型在陌生环境中的适应能力。

### 方法

提出可供性场干预(AFI)框架，使用3D空间可供性场(SAFs)作为按需插件来指导VLA行为。系统通过本体感受检测记忆陷阱，将机器人重新定位到高可供性区域，提出可供性驱动的路径点锚定VLA生成的动作，并使用基于SAF的评分器选择最优轨迹。

### 主要发现

实验表明，该方法在不同VLA主干网络下，在真实机器人平台上的分布外场景中平均提高了23.5%的性能，在LIBERO-Pro基准测试上提高了20.2%，有效增强了VLA对分布变化的鲁棒性。

### 结论

可供性场干预(AFI)框架通过提供明确的3D空间理解，显著提升了VLA模型在分布变化场景下的鲁棒性和适应性，解决了记忆陷阱问题。

### 翻译

Vision-Language-Action (VLA) 模型通过直接将视觉观察和语言指令映射到动作，在机器人操作中展现出卓越性能。然而，它们在分布变化的情况下仍然表现脆弱：当测试场景发生变化时，VLA往往会重现记忆的轨迹，而不是适应更新后的场景，我们将这种失败模式称为'记忆陷阱'。这一局限性源于端到端设计，缺乏明确的3D空间推理能力，无法在陌生环境中可靠地识别可操作区域。为了弥补这种空间理解的缺失，3D空间可供性场(SAFs)可以提供几何表示，突出显示物理上可行的交互区域，提供明确的线索指示机器人应该接近或避免的区域。因此，我们提出了可供性场干预(AFI)，这是一个轻量级混合框架，使用SAFs作为按需插件来指导VLA行为。我们的系统通过本体感受检测记忆陷阱，将机器人重新定位到最近的高可供性区域，并提出可供性驱动的路径点来锚定VLA生成的动作。然后，基于SAF的评分器选择具有最高累积可供性的轨迹。大量实验证明，我们的方法在不同VLA主干网络下，在真实机器人平台上的分布外场景中平均提高了23.5%，在LIBERO-Pro基准测试上提高了20.2%，验证了其在增强VLA对分布变化的鲁棒性方面的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决Vision-Language-Action (VLA)模型在机器人操作中的'记忆陷阱'问题，即当测试场景发生变化时，VLA模型往往会重复记忆中的轨迹而非适应新环境，导致任务失败。这个问题在现实中非常重要，因为机器人需要在各种未知和变化的环境中工作，而不仅仅是复制训练场景中的动作。记忆陷阱限制了VLA模型在实际应用中的鲁棒性和泛化能力，使其难以应对真实世界中的各种变化和挑战。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了VLA模型的问题根源在于其端到端设计，缺乏明确的3D空间推理能力。他们借鉴了3D空间affordance fields (SAFs)的概念，这些可以提供几何表示，突出显示物理上可行的交互区域。作者设计了一个混合框架Affordance Field Intervention (AFI)，使用SAFs作为按需插件来引导VLA行为。该方法包括三个主要步骤：(1)通过本体感受检测记忆陷阱；(2)将机器人重新定位到最近的高affordance区域；(3)提出affordance驱动的路径点作为VLA生成动作的锚点。作者还参考了VLM-based规划方法，但克服了这些方法的局限性，如不可靠的动作计划和任务特定的提示工程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过引入3D空间affordance fields (SAFs)作为VLA模型的插件，帮助模型在分布外(OOD)场景中'逃脱'记忆陷阱。SAFs提供了明确的3D空间推理能力，指导机器人适应变化的环境。整体实现流程包括：1) 构建空间affordance field (SAF)：通过视觉语言模型识别任务相关对象，构建目标引导场和障碍物回避场；2) 记忆陷阱检测：监控机器人执行状态，当末端执行器位移低且与目标距离远时触发；3) 干预过程：历史回滚到高affordance位置，分层探索采样中间路径点，使用VLA生成动作候选，并选择累积affordance成本最低的轨迹执行。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 提出'记忆陷阱'概念，明确描述VLA在分布外场景中的失败模式；2) 设计Affordance Field Intervention (AFI)框架，将SAFs作为按需插件集成；3) 开发记忆陷阱检测机制，通过本体感受监控；4) 提出分层探索策略，结合SAF空间推理和VLA任务能力；5) 设计SAF-based评分器评估动作候选。相比之前工作，本文主要不同在于：不是从头训练新模型，而是对现有VLA进行轻量级干预；结合数据驱动政策和可解释几何规划；无需额外演示数据或微调即可提升鲁棒性；在真实和模拟环境中均显示显著性能提升。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过引入3D空间affordance fields作为插件干预，有效解决了VLA模型在分布外场景中的记忆陷阱问题，显著提升了机器人在变化环境中的操作鲁棒性，无需重新训练模型或额外数据。'}


### 论文摘要

Vision-Language-Action (VLA) models have shown great performance in robotic manipulation by mapping visual observations and language instructions directly to actions. However, they remain brittle under distribution shifts: when test scenarios change, VLAs often reproduce memorized trajectories instead of adapting to the updated scene, which is a failure mode we refer to as the "Memory Trap". This limitation stems from the end-to-end design, which lacks explicit 3D spatial reasoning and prevents reliable identification of actionable regions in unfamiliar environments. To compensate for this missing spatial understanding, 3D Spatial Affordance Fields (SAFs) can provide a geometric representation that highlights where interactions are physically feasible, offering explicit cues about regions the robot should approach or avoid. We therefore introduce Affordance Field Intervention (AFI), a lightweight hybrid framework that uses SAFs as an on-demand plug-in to guide VLA behavior. Our system detects memory traps through proprioception, repositions the robot to recent high-affordance regions, and proposes affordance-driven waypoints that anchor VLA-generated actions. A SAF-based scorer then selects trajectories with the highest cumulative affordance. Extensive experiments demonstrate that our method achieves an average improvement of 23.5% across different VLA backbones ($π_{0}$ and $π_{0.5}$) under out-of-distribution scenarios on real-world robotic platforms, and 20.2% on the LIBERO-Pro benchmark, validating its effectiveness in enhancing VLA robustness to distribution shifts.

---

## 127. STRinGS: Selective Text Refinement in Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2512.07230v1](http://arxiv.org/abs/2512.07230v1)

**作者:** Abhinav Raundhal, Gaurav Behera, P J Narayanan, Ravi Kiran Sarvadevabhatla, Makarand Tapaswi

**发布时间:** 2025-12-08

**备注:** Accepted to WACV 2026. Project Page, see https://STRinGS-official.github.io

### GPT解析

### 总结

论文提出STRinGS，一个文本感知的选择性细化框架，用于解决3D高斯溅射(3DGS)在重建文本细节时的问题，提高文本区域的可读性和准确性。

### 背景

文本作为标志、标签或指令是真实场景中的关键元素，能够传达重要的上下文信息。然而，3D表示方法如3D高斯溅射(3DGS)在保持细粒度文本细节方面存在困难，文本元素的微小重建错误可能导致显著的语义损失。

### 目的

解决3DGS在文本区域重建中的问题，提高文本区域的可读性和准确性。

### 方法

STRinGS分别处理文本和非文本区域，先细化文本区域，然后将它们与非文本区域合并进行全场景优化。作者还引入了文本可读性度量指标OCR字符错误率(CER)来评估文本区域的效能。

### 主要发现

STRinGS在仅7K次迭代后，相比3DGS实现了63.6%的相对改进，能够产生清晰、可读的文本，即使在具有挑战性的配置下也是如此。

### 结论

STRinGS方法和STRinGS-360数据集一起推动了文本丰富环境中的3D场景理解边界，为更强大的文本感知重建方法铺平了道路。

### 翻译

文本作为标志、标签或指令是真实场景中的关键元素，因为它们可以传达重要的上下文信息。像3D高斯溅射(3DGS)这样的3D表示方法在保持细粒度文本细节的同时难以实现高视觉保真度。文本元素重建中的小错误可能导致显著的语义损失。我们提出了STRinGS，一个文本感知的选择性细化框架，用于解决3DGS重建中的这个问题。我们的方法分别处理文本和非文本区域，先细化文本区域，然后将它们与非文本区域合并进行全场景优化。STRinGS即使在具有挑战性的配置下也能产生清晰、可读的文本。我们引入了文本可读性度量OCR字符错误率(CER)来评估文本区域的效能。STRinGS在仅7K次迭代后相比3DGS实现了63.6%的相对改进。我们还引入了一个包含多样化文本场景的精选数据集STRinGS-360，用于评估3D重建中的文本可读性。我们的方法和数据集一起推动了文本丰富环境中3D场景理解的边界，为更强大的文本感知重建方法铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D高斯溅射(3DGS)在重建场景中的文本细节时表现不佳的问题。文本作为现实场景中的关键元素(如标志、标签、说明)能传达重要信息，文本重建中的小错误可能导致重大语义损失。这个问题在自动驾驶(解读路标)、VR(改善用户体验)、机器人技术(物体识别)等领域都很重要，但传统3D重建方法和评估指标难以有效处理文本细节。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3DGS在处理高频细节(如文本)时的局限性，分析了现有方法如Mip-Splatting、3DGS-MCMC等虽然提高了整体场景质量但未专门针对文本优化。作者设计了一个两阶段框架：第一阶段隔离并重建文本区域，第二阶段进行全场景优化。作者借鉴了COLMAP进行SfM处理、Hi-SAM进行文本分割、3DGS的优化策略等现有工作，但将其整合到一个专门针对文本重建的新框架中，并进行了针对性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将文本和非文本区域分开处理，采用两阶段优化策略。流程包括：1)预处理阶段使用COLMAP和Hi-SAM获取点云和文本掩码；2)3D文本分割将点云分为文本和非文本点；3)第一阶段仅优化文本区域，进行密集化处理并锁定位置参数；4)第二阶段结合文本和非文本区域进行全场景优化，使用差异化学习率策略；5)输出最终结果，实现增强文本可读性同时保持整体场景质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门针对文本的3DGS框架；2)两阶段选择性优化策略；3)文本感知的密集化方法；4)差异化学习率策略；5)STRinGS-360文本丰富数据集；6)OCR-CER文本可读性评估指标。相比之前工作，STRinGS专门针对语义区域(文本)而非整体视觉质量，将文本和非文本区域分开处理，能在早期训练阶段实现高质量文本重建，同时保持整体场景质量和训练效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'STRinGS通过引入专门针对文本的选择性细化框架和相应的评估方法，显著提高了3D场景中文本的重建质量和可读性，同时保持了整体场景的保真度和训练效率。'}


### 论文摘要

Text as signs, labels, or instructions is a critical element of real-world scenes as they can convey important contextual information. 3D representations such as 3D Gaussian Splatting (3DGS) struggle to preserve fine-grained text details, while achieving high visual fidelity. Small errors in textual element reconstruction can lead to significant semantic loss. We propose STRinGS, a text-aware, selective refinement framework to address this issue for 3DGS reconstruction. Our method treats text and non-text regions separately, refining text regions first and merging them with non-text regions later for full-scene optimization. STRinGS produces sharp, readable text even in challenging configurations. We introduce a text readability measure OCR Character Error Rate (CER) to evaluate the efficacy on text regions. STRinGS results in a 63.6% relative improvement over 3DGS at just 7K iterations. We also introduce a curated dataset STRinGS-360 with diverse text scenarios to evaluate text readability in 3D reconstruction. Our method and dataset together push the boundaries of 3D scene understanding in text-rich environments, paving the way for more robust text-aware reconstruction methods.

---

## 128. START: Spatial and Textual Learning for Chart Understanding

**论文链接:** [http://arxiv.org/abs/2512.07186v1](http://arxiv.org/abs/2512.07186v1)

**作者:** Zhuoming Liu, Xiaofeng Gao, Feiyang Niu, Qiaozi Gao, Liu Liu, Robinson Piramuthu

**发布时间:** 2025-12-08

**备注:** WACV2026 Camera Ready

### GPT解析

### 总结

START是一个结合空间和文本学习的图表理解框架，通过图表元素定位和图表到代码生成来增强多模态大语言模型对图表的理解。研究团队提出了START-Dataset数据集和CS-Bench评估基准，实验证明START在多个评估中表现优异，明显超越了之前的最先进方法。

### 背景

图表理解对于在现实场景（如分析科学论文和技术报告）中部署多模态大语言模型至关重要。与自然图像不同，图表结合了结构化的视觉布局（空间属性）和基础数据表示（文本属性），理解这两方面对于精确、细粒度的图表推理至关重要。

### 目的

提出START（Spatial and Textual learning for chART understanding）框架，加强多模态大语言模型对图表视觉布局和数据细节的理解，解决现有方法无法处理的挑战。

### 方法

引入图表元素定位和图表到代码生成两种技术；提出START-Dataset，通过新颖的数据生成管道生成，该管道首先利用多模态大语言模型将真实图表图像转换为可执行的图表代码，恢复基础数据表示同时保留真实世界图表的视觉分布，然后使用大语言模型进化代码以确定图表元素位置；提出图表空间理解基准(Chart Spatial understanding Benchmark, CS-Bench)。

### 主要发现

START在模型规模和基准测试上对基础模型带来了一致的提升，明显超越了之前的最先进方法。

### 结论

START框架有效提升了多模态大语言模型对图表的理解能力，代码、数据和模型将公开可用，为图表理解研究提供了新资源。

### 翻译

图表理解对于在现实场景中部署多模态大语言模型至关重要，例如分析科学论文和技术报告。与自然图像不同，图表结合了结构化的视觉布局（空间属性）和基础数据表示（文本属性）——理解这两方面对于精确、细粒度的图表推理至关重要。受此观察启发，我们提出了START，即用于图表理解的空间和文本学习。具体而言，我们引入了图表元素定位和图表到代码生成，以加强多模态大语言模型对图表视觉布局和数据细节的理解。为了促进空间和文本学习，我们提出了START-Dataset，它通过新颖的数据生成管道生成，该管道首先利用多模态大语言模型将真实图表图像转换为可执行的图表代码，恢复基础数据表示同时保留真实世界图表的视觉分布。然后我们使用大语言模型进化代码，以确定捕获图表视觉结构的图表元素位置，解决现有方法无法处理的挑战。为了评估模型理解图表空间结构的能力，我们提出了图表空间理解基准，填补了全面图表理解评估的关键空白。利用空间和文本学习，START在各种模型规模和基准测试中都对基础模型带来了一致的提升，并明显超越了之前的最先进方法。代码、数据和模型将公开可用。


### 论文摘要

Chart understanding is crucial for deploying multimodal large language models (MLLMs) in real-world scenarios such as analyzing scientific papers and technical reports. Unlike natural images, charts pair a structured visual layout (spatial property) with an underlying data representation (textual property) -- grasping both is essential for precise, fine-grained chart reasoning. Motivated by this observation, we propose START, the Spatial and Textual learning for chART understanding. Specifically, we introduce (i) chart-element grounding and (ii) chart-to-code generation to strengthen an MLLM's understanding of both chart visual layout and data details. To facilitate spatial and textual learning, we propose the START-Dataset generated with a novel data-generation pipeline that first leverages an MLLM to translate real chart images into executable chart code, recovering the underlying data representation while preserving the visual distribution of real-world charts. We then evolve the code with a Large Language Model (LLM) to ascertain the positions of chart elements that capture the chart's visual structure, addressing challenges that existing methods cannot handle. To evaluate a model's ability to understand chart spatial structures, we propose the Chart Spatial understanding Benchmark (CS-Bench), filling a critical gap in comprehensive chart understanding evaluation. Leveraging spatial and textual learning, START delivers consistent gains across model sizes and benchmarks over the base models and surpasses prior state-of-the-art by a clear margin. Code, data and models will be publicly available.

---

## 129. A Large-Scale Multimodal Dataset and Benchmarks for Human Activity Scene Understanding and Reasoning

**论文链接:** [http://arxiv.org/abs/2512.07136v1](http://arxiv.org/abs/2512.07136v1)

**作者:** Siyang Jiang, Mu Yuan, Xiang Ji, Bufang Yang, Zeyu Liu, Lilin Xu, Yang Li, Yuting He, Liran Dong, Wenrui Lu, Zhenyu Yan, Xiaofan Jiang, Wei Gao, Hongkai Chen, Guoliang Xing

**发布时间:** 2025-12-08

### GPT解析

### 总结

这篇论文介绍了CUHK-X，一个大规模多模态数据集和基准测试套件，用于人类动作识别(HAR)、人类动作理解(HAU)和人类动作推理(HARn)。该数据集包含58,445个样本，覆盖40种动作，由30名参与者在两个室内环境中执行。作者提出了一种基于提示的场景创建方法以提高描述一致性，实验报告显示平均准确率分别为76.52%(HAR)、40.76%(HAU)和70.25%(HARn)。

### 背景

多模态人类动作识别利用互补传感器进行活动分类。近期大型语言模型的发展催生了人类动作理解和推理新任务，但大多数大型视觉语言模型在处理深度、IMU和mmWave等非RGB模态时存在困难，且现有HAR数据集缺乏细粒度动作动态信息。

### 目的

解决现有多模态人类动作分析中的数据限制问题，特别是非RGB模态数据和细粒度动作动态信息的缺乏，创建支持HAR、HAU和HARn任务的数据集和基准测试套件。

### 方法

考虑两种真实配对类型：数据标签(离散类别)和数据标题(文本描述)。提出基于提示的场景创建方法，利用大型语言模型生成逻辑连接的活动序列，然后进行人工验证。CUHK-X包含58,445个样本，涵盖40种动作。

### 主要发现

实验报告显示，CUHK-X在三个基准测试上的平均准确率分别为：76.52%(HAR)、40.76%(HAU)和70.25%(HARn)，表明该数据集能有效支持多模态人类动作分析的不同任务。

### 结论

CUHK-X为多模态人类动作分析提供了新资源，特别是对需要细粒度动作动态理解的HAU和HARn任务，旨在促进社区应用和发展密集型数据学习方法。

### 翻译

多模态人类动作识别(HAR)利用互补传感器进行活动分类。除了识别功能外，近期大型语言模型(LLMs)的进步使得详细描述和因果推理成为可能，从而催生了新任务：人类动作理解(HAU)和人类动作推理(HARn)。然而，大多数LLMs，特别是大型视觉语言模型(LVLMs)，在处理深度、IMU和mmWave等非RGB模态时存在困难，因为缺乏大规模数据-标题资源。现有HAR数据集主要提供粗略的数据-标签注释，不足以捕获HAU和HARn所需的细粒度动作动态。我们考虑了两种真实配对类型：(1)数据标签(离散类别)和(2)数据标题(文本描述)。简单地从标签生成标题通常缺乏逻辑和时空一致性。我们介绍了CUHK-X，一个用于HAR、HAU和HARn的大规模多模态数据集和基准测试套件。CUHK-X包含58,445个样本，涵盖40种动作，由30名参与者在两个室内环境中执行。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决人类活动理解(HAU)和人类活动推理(HARn)任务中缺乏大规模多模态数据集的问题。这个问题很重要，因为现有的大多数大型语言模型难以处理非RGB模态数据(如深度、IMU或mmWave)，而现有HAR数据集只提供粗粒度标签，不足以描述HAU和HARn所需的详细动作动态。这个问题在现实中对医疗保健、智能家居和隐私保护等领域至关重要，例如在阿尔茨海默病管理中需要连贯理解患者行为，智能家居系统需要预测用户行为来优化环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有数据集的局限性，发现现有数据集要么缺乏模态多样性，要么只提供粗粒度标签。他们借鉴了美国时间使用调查(ATUS)的活动分类框架和多个现有数据集中的高频动作，采用'真实优先'(GT-first)的数据收集策略。设计上，他们提出了基于提示的场景创建方法，利用LLMs生成逻辑连贯的活动序列，并通过人类检查确保物理可行性和时间逻辑。这种方法结合了多种现有技术，如基于模板的生成和人类验证，形成了一个综合性的解决方案。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个包含七种同步模态(RGB、深度、红外、热成像、骨骼、IMU和mmWave)的大规模多模态数据集，并使用基于提示的场景创建方法生成逻辑一致的活动描述。整体流程包括：1)基于ATUS和跨数据集频率分析进行动作选择，确定40种代表性动作；2)利用LLMs将选定动作连接成连贯的场景描述；3)通过语言丰富化和人类检查增强字幕质量；4)在两个真实室内环境中收集数据，确保多模态精确同步；5)建立三个基准测试(HAR、HAU、HARn)包含六个任务，评估各种模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建了CUHK-X数据集，包含58,445个样本、七种模态和40种日常活动；2)提出了基于提示的场景创建方法，解决字幕时空一致性问题；3)建立了三个基准测试包含六个任务。相比之前工作，CUHK-X提供了更全面的模态覆盖(七种而非一两种)，包含详细的字幕描述支持高级理解任务，采用真实优先策略确保数据质量，并提供全面的基准测试。现有数据集要么模态单一，要么标签粗糙，无法支持HAU和HARn任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一个名为CUHK-X的大规模多模态数据集和基准测试，包含七种传感器模态、40种日常活动和六个评估任务，为人类活动识别、理解和推理研究提供了重要资源。'}


### 论文摘要

Multimodal human action recognition (HAR) leverages complementary sensors for activity classification. Beyond recognition, recent advances in large language models (LLMs) enable detailed descriptions and causal reasoning, motivating new tasks: human action understanding (HAU) and human action reasoning (HARn). However, most LLMs, especially large vision language models (LVLMs), struggle with non-RGB modalities such as depth, IMU, and mmWave due to the lack of large-scale data-caption resources. Existing HAR datasets mainly provide coarse data-label annotations, which are insufficient to capture fine-grained action dynamics needed for HAU and HARn. We consider two ground-truth pair types: (1) data label (discrete category) and (2) data caption (textual description). Naively generating captions from labels often lacks logical and spatiotemporal consistency. We introduce CUHK-X, a large-scale multimodal dataset and benchmark suite for HAR, HAU, and HARn. CUHK-X contains 58,445 samples covering 40 actions performed by 30 participants across two indoor environments. To improve caption consistency, we propose a prompt-based scene creation method that leverages LLMs to generate logically connected activity sequences, followed by human validation. CUHK-X includes three benchmarks with six evaluation tasks. Experiments report average accuracies of 76.52% (HAR), 40.76% (HAU), and 70.25% (HARn). CUHK-X aims to enable the community to apply and develop data-intensive learning methods for robust, multimodal human activity analysis. Project page and code: https://openaiotlab.github.io/CUHK-X/ and https://github.com/openaiotlab/CUHK-X.

---

## 130. Stitch and Tell: A Structured Multimodal Data Augmentation Method for Spatial Understanding

**论文链接:** [http://arxiv.org/abs/2512.06769v1](http://arxiv.org/abs/2512.06769v1)

**作者:** Hang Yin, Xiaomin He, PeiWen Yuan, Yiwei Li, Jiayi Shi, Wenxiao Fan, Shaoxiong Feng, Kan Li

**发布时间:** 2025-12-07

### GPT解析

### 总结

该论文提出了一种名为SiTe的方法，用于解决视觉-语言模型中的空间幻觉问题。这种方法通过在数据中注入结构化的空间监督，无需昂贵的先进模型或人工标注，就能有效提高模型的空间理解能力，同时保持或提高通用视觉-语言能力。

### 背景

现有的视觉-语言模型经常遭受空间幻觉的困扰，即生成关于图像中物体相对位置的错误描述。问题主要源于图像和文本之间的不对称特性。

### 目的

为了增强视觉-语言模型的空间理解能力，作者提出了一种简单、无需标注、即插即用的方法。

### 方法

提出了一种名为'Stitch and Tell'(SiTe)的方法，它通过沿空间轴拼接图像并基于拼接图像的布局生成具有空间意识的描述或问答对，来构建拼接的图像-文本对，从而向数据中注入结构化的空间监督，无需依赖昂贵的先进模型或人工参与。

### 主要发现

在三种架构（LLaVA-v1.5-7B、LLaVA-Qwen2-1.5B和HALVA-7B）、两个训练数据集和八个基准上评估SiTe，实验表明SiTe提高了空间理解任务，如MME_Position和Spatial-MM，同时保持或提高了通用视觉-语言基准的性能，包括COCO-QA和MMBench。

### 结论

明确向训练数据中注入具有空间意识的结构是减轻空间幻觉和改善空间理解的有效方法，同时保留通用视觉-语言能力。

### 翻译

现有的视觉-语言模型常常遭受空间幻觉的困扰，即生成关于图像中物体相对位置的错误描述。我们认为这个问题主要源于图像和文本之间的不对称特性。为了增强视觉-语言模型的空间理解能力，我们提出了一种简单、无需标注、即插即用的方法，名为'Stitch and Tell'（简称SiTe），它将结构化的空间监督注入到数据中。它通过沿空间轴拼接图像并基于拼接图像的布局生成具有空间意识的描述或问答对来构建拼接的图像-文本对，无需依赖昂贵的先进模型或人工参与。我们在三种架构、两个训练数据集和八个基准上评估了SiTe。实验表明，SiTe提高了空间理解任务，同时保持或提高了通用视觉-语言基准的性能。我们的研究结果表明，明确向训练数据中注入具有空间意识的结构是减轻空间幻觉和改善空间理解的有效方法，同时保留通用视觉-语言能力。


### 论文摘要

Existing vision-language models often suffer from spatial hallucinations, i.e., generating incorrect descriptions about the relative positions of objects in an image. We argue that this problem mainly stems from the asymmetric properties between images and text. To enrich the spatial understanding ability of vision-language models, we propose a simple, annotation-free, plug-and-play method named $\text{Stitch and Tell}$ (abbreviated as SiTe), which injects structured spatial supervision into data. It constructs stitched image-text pairs by stitching images along a spatial axis and generating spatially-aware captions or question answer pairs based on the layout of stitched image, without relying on costly advanced models or human involvement. We evaluate SiTe across three architectures including LLaVA-v1.5-7B, LLaVA-Qwen2-1.5B and HALVA-7B, two training datasets, and eight benchmarks. Experiments show that SiTe improves spatial understanding tasks such as $\text{MME}_{\text{Position}}$ (+5.50%) and Spatial-MM (+4.19%), while maintaining or improving performance on general vision-language benchmarks including COCO-QA (+1.02%) and MMBench (+4.76%). Our findings suggest that explicitly injecting spatially-aware structure into training data offers an effective way to mitigate spatial hallucinations and improve spatial understanding, while preserving general vision-language capabilities.

---

## 131. Physics-Grounded Attached Shadow Detection Using Approximate 3D Geometry and Light Direction

**论文链接:** [http://arxiv.org/abs/2512.06179v1](http://arxiv.org/abs/2512.06179v1)

**作者:** Shilin Hu, Jingyi Xu, Sagnik Das, Dimitris Samaras, Hieu Le

**发布时间:** 2025-12-05

### GPT解析

### 总结

该论文提出了一种新的阴影检测框架，专门解决了现有方法忽视的附着阴影检测问题，通过迭代几何-光照推理过程显著提高了附着阴影检测性能。

### 背景

附着阴影发生在遮挡物表面因自遮挡而无法接收光照的区域，对定义物体三维结构和增强场景理解至关重要。然而现有阴影检测方法主要针对投射阴影，缺乏专门的数据集和模型用于检测附着阴影。

### 目的

引入一个能够同时检测投射阴影和附着阴影的框架，解决现有方法中缺乏专门针对附着阴影检测的问题。

### 方法

提出联合框架，包含阴影检测模块（分别预测两种阴影类型）和光照估计模块（从检测到的阴影推断光照方向）。结合估计的光照方向和表面法线推导几何一致的局部映射，识别可能自遮挡的区域，并将该映射反馈以优化阴影预测，形成闭环推理过程。构建了包含1458张图像的数据集，分别标注了投射阴影和附着阴影。

### 主要发现

迭代几何-光照推理显著改善了附着阴影的检测，误报率（BER）降低了至少33%，同时保持了整体和投射阴影的强性能。

### 结论

该方法通过结合几何和光照信息实现了更好的阴影分割，有效解决了附着阴影检测的问题。

### 翻译

附着阴影发生在遮挡物表面，由于自遮挡而光照无法到达的区域。它们对于定义物体的三维结构和增强场景理解至关重要。然而现有的阴影检测方法主要针对投射阴影，没有专门的数据集或模型用于检测附着阴影。为解决这一差距，我们引入了一个框架，通过推理阴影与场景光照和几何形状的相互关系来联合检测投射阴影和附着阴影。我们的系统包含一个阴影检测模块，分别预测两种阴影类型，以及一个光照估计模块，从检测到的阴影推断光照方向。估计的光照方向结合表面法线，使我们能够推导出几何一致的局部映射，识别可能自遮挡的区域。然后将该局部映射反馈以优化阴影预测，形成闭环推理过程，迭代改进阴影分割和光照估计。为了训练我们的方法，我们构建了一个包含1458张图像的数据集，分别标注了投射阴影和附着阴影，从而能够训练和定量评估两种阴影。实验结果表明，这种迭代几何-光照推理显著改善了附着阴影的检测，误报率至少降低了33%，同时保持了整体和投射阴影的强性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是附着阴影(attached shadows)的检测问题。现有阴影检测方法主要关注投影阴影(cast shadows)，而忽略了附着阴影。这个问题很重要，因为附着阴影对于定义物体的三维结构和增强场景理解至关重要，两种阴影类型共同编码了场景结构和照明的互补线索，有助于计算机视觉中的场景理解和物体识别。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于物理原理思考，观察到附着阴影是由表面方向和光线方向相互作用形成的，表面背离光源的区域会处于阴影中。他们设计了一个双模块架构：阴影检测模块预测投影和附着阴影，光估计模块推断光线方向。这两个模块形成封闭反馈循环，通过多次迭代相互改进。作者借鉴了传统照明推理方法和现代深度学习技术，阴影检测模块基于SILT方法改进，光估计模块基于ConvNeXt-S架构，并利用现有数据集构建了自己的专门数据集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用物理原理建立封闭反馈循环系统，通过联合学习阴影检测和光照估计来同时检测两种阴影。流程如下：1)初始化时阴影检测模块接收图像和法线图预测阴影；2)光估计模块使用预测阴影和法线估计光线方向；3)基于估计光线方向和表面法线生成部分附着阴影图；4)将此图反馈给阴影检测模块改进预测；5)多次迭代运行，逐步提高阴影检测和光照估计的准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门针对附着阴影检测的框架；2)基于物理的迭代推理过程；3)首个包含投影和附着阴影单独标注的新数据集；4)双模块架构设计。相比之前工作，本文明确区分并分别检测两种阴影类型，利用表面几何和光线方向的物理关系，引入迭代反馈机制，并构建了专门的数据集，显著提高了附着阴影检测性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了首个基于物理原理的迭代框架，通过联合学习阴影检测和光照估计实现了对投影阴影和附着阴影的准确区分和检测，并构建了首个专门针对这两种阴影类型的数据集。'}


### 论文摘要

Attached shadows occur on the surface of the occluder where light cannot reach because of self-occlusion. They are crucial for defining the three-dimensional structure of objects and enhancing scene understanding. Yet existing shadow detection methods mainly target cast shadows, and there are no dedicated datasets or models for detecting attached shadows. To address this gap, we introduce a framework that jointly detects cast and attached shadows by reasoning about their mutual relationship with scene illumination and geometry. Our system consists of a shadow detection module that predicts both shadow types separately, and a light estimation module that infers the light direction from the detected shadows. The estimated light direction, combined with surface normals, allows us to derive a geometry-consistent partial map that identifies regions likely to be self-occluded. This partial map is then fed back to refine shadow predictions, forming a closed-loop reasoning process that iteratively improves both shadow segmentation and light estimation. In order to train our method, we have constructed a dataset of 1,458 images with separate annotations for cast and attached shadows, enabling training and quantitative evaluation of both. Experimental results demonstrate that this iterative geometry-illumination reasoning substantially improves the detection of attached shadows, with at least 33% BER reduction, while maintaining strong full and cast shadow performance.

---

## 132. BeLLA: End-to-End Birds Eye View Large Language Assistant for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.06096v1](http://arxiv.org/abs/2512.06096v1)

**作者:** Karthik Mohan, Sonam Singh, Amit Arvind Kale

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种名为BeLLA的端到端架构，将统一的360°鸟瞰图表示与大型语言模型连接，用于自动驾驶中的问答任务。该方法在需要更强空间推理能力的任务上表现优异。

### 背景

Vision-Language模型和多模态语言模型在自动驾驶研究中的快速发展已经显著改变了研究格局，使更丰富的场景理解、上下文感知推理和更可解释的决策成为可能。然而，现有工作往往依赖于无法利用多摄像头系统空间结构的单视图编码器，或在聚合的多视图特征上操作，这些特征缺乏统一的空间表示。

### 目的

提出一种能够有效利用多摄像头系统空间结构并建立统一空间表示的方法，以增强自动驾驶系统中的场景理解和推理能力。

### 方法

提出了BeLLA，一种端到端架构，将统一的360°BEV表示与大型语言模型连接，用于自动驾驶中的问答任务。

### 主要发现

在NuScenes-QA和DriveLM两个基准测试中，BeLLA在需要更强空间推理能力的问题上持续优于现有方法，在涉及物体相对位置和附近物体行为理解的问题上实现了高达+9.3%的绝对提升。在其他类别的问题上，BeLLA也具有竞争力。

### 结论

BeLLA通过将统一的360°BEV表示与大型语言模型结合，有效解决了现有方法在空间推理方面的局限性，显著提高了自动驾驶系统在问答任务中的表现。

### 翻译

Vision-Language模型和多模态语言模型在自动驾驶研究中的快速发展已经显著改变了研究格局，使更丰富的场景理解、上下文感知推理和更可解释的决策成为可能。然而，现有工作往往依赖于无法利用多摄像头系统空间结构的单视图编码器，或在聚合的多视图特征上操作，这些特征缺乏统一的空间表示，使得对以自我为中心的方向、物体关系和更广泛上下文的推理变得更加困难。因此，我们提出了BeLLA，一种端到端架构，将统一的360°BEV表示与大型语言模型连接，用于自动驾驶中的问答任务。我们主要使用两个基准测试评估我们的工作——NuScenes-QA和DriveLM，在需要更强空间推理能力的问题上，如涉及物体相对位置和附近物体行为理解的问题，BeLLA持续优于现有方法，在某些任务中实现了高达+9.3%的绝对提升。在其他类别的问题上，BeLLA也具有竞争力，展示了处理多样化问题的能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何有效结合多摄像头系统的空间结构与大型语言模型进行自动驾驶场景问答的问题。这个问题很重要，因为在自动驾驶中准确识别周围车辆和行人的位置及轨迹对安全决策至关重要，而现有方法要么无法利用多摄像头的空间结构，要么缺乏统一的场景表示，难以进行空间推理。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，然后借鉴了自动驾驶领域广泛采用的鸟瞰图(BEV)表示方法，以及多模态基础模型和视觉-语言模型(VLMs)的工作。他们设计了两阶段训练流程：先用冻结的BEV编码器处理多视图图像并通过投影器对齐文本描述，然后微调语言模型用于驾驶导向的问答任务。这种方法结合了BEV的空间表示能力和LLM的语言理解能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将多摄像头系统的360度鸟瞰图表示与大型语言模型相结合，创建一个能基于统一场景表示回答自动驾驶问题的端到端架构。实现流程分为两阶段：1)预训练阶段，用冻结的BEV编码器处理图像并通过投影器将BEV特征转换为语言模型可理解的嵌入，对齐空间特征与文本描述；2)微调阶段，保持BEV编码器冻结，更新投影器和语言模型参数，使模型能够回答基于BEV场景上下文的驾驶相关问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出BeLLA架构，通过BEV特征将多摄像头输入与语言理解连接；2)将BEV表示压缩为单个令牌并引入BEV-文本预对齐阶段；3)在需要空间推理的问答任务上显著提升性能。相比之前工作，BeLLA不同于传统单视图编码器方法，能够利用多摄像头空间结构；也不同于多视图特征聚合方法，提供统一的场景表示；还直接将原始BEV特征投影到LLM输入空间，而非转换为结构化格式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BeLLA通过将鸟瞰图空间表示与大型语言模型相结合，创造了一个端到端的自动驾驶问答系统，显著提升了在需要空间推理任务上的性能，同时保持了在广泛问题类型上的竞争力。'}


### 论文摘要

The rapid development of Vision-Language models (VLMs) and Multimodal Language Models (MLLMs) in autonomous driving research has significantly reshaped the landscape by enabling richer scene understanding, context-aware reasoning, and more interpretable decision-making. However, a lot of existing work often relies on either single-view encoders that fail to exploit the spatial structure of multi-camera systems or operate on aggregated multi-view features, which lack a unified spatial representation, making it more challenging to reason about ego-centric directions, object relations, and the wider context. We thus present BeLLA, an end-to-end architecture that connects unified 360° BEV representations with a large language model for question answering in autonomous driving. We primarily evaluate our work using two benchmarks - NuScenes-QA and DriveLM, where BeLLA consistently outperforms existing approaches on questions that require greater spatial reasoning, such as those involving relative object positioning and behavioral understanding of nearby objects, achieving up to +9.3% absolute improvement in certain tasks. In other categories, BeLLA performs competitively, demonstrating the capability of handling a diverse range of questions.

---

## 133. Towards Cross-View Point Correspondence in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.04686v2](http://arxiv.org/abs/2512.04686v2)

**作者:** Yipu Wang, Yuheng Ji, Yuyang Liu, Enshen Zhou, Ziqiang Yang, Yuxuan Tian, Ziheng Qin, Yue Liu, Huajie Tan, Cheng Chi, Zhiyuan Ma, Daniel Dajun Zeng, Xiaolong Zheng

**发布时间:** 2025-12-04

### GPT解析

### 总结

研究提出了跨视角点对应(CVPC)任务和CrossPoint-Bench基准测试，构建了CrossPoint-378K数据集，并开发了CroPond模型，显著提升了跨视角对应能力，为具身AI和空间理解研究提供了基础。

### 背景

跨视角对应能力是空间理解和具身AI的基本能力，但视觉语言模型(VLMs)目前仍未能实现这一能力，特别是在精确的点级对应方面，这对精确的效用性交互至关重要。

### 目的

提出跨视角点对应任务和CrossPoint-Bench基准测试，受人类认知过程'感知'、'推理'和'对应'的启发，构建层次化设计基准测试，以解决跨视角对应问题。

### 方法

构建CrossPoint-378K数据集，包含900个场景中的37.8万个问答对，专注于可操作的效用性区域；提出CroPond模型，在该数据集上进行训练；设计层次化基准测试评估跨视角对应能力。

### 主要发现

最先进模型(如Gemini-2.5-Pro)与人类表现差距显著，总体准确率差距超过54.65%，暴露了从粗粒度判断到细粒度坐标预测转变的挑战；CroPond在CrossPoint-Bench上取得最先进性能，比Gemini-2.5-Pro高出39.7%的准确率。

### 结论

CroPond为推进跨视角对应研究提供了坚实基础，相关基准测试、数据集和模型已在GitHub公开，可供后续研究使用。

### 翻译

跨视角对应是空间理解和具身AI的基本能力。然而，视觉语言模型(VLMs)仍远未实现这一能力，特别是在实现精确的点级对应方面，这对精确的效用性交互至关重要。因此，我们提出了跨视角点对应(CVPC)任务和CrossPoint-Bench，这是一个受人类认知过程'感知'、'推理'和'对应'启发的、具有层次化设计的全面基准测试。我们的评估显示，最先进的模型(如Gemini-2.5-Pro)仍远落后于人类，总体准确率差距超过54.65%，暴露了从粗粒度判断到细粒度坐标预测转变的挑战。为解决这个问题，我们构建了CrossPoint-378K数据集，包含900个场景中的37.8万个问答对，专注于更好地反映现实世界操作和交互场景的可操作效用性区域。此外，我们提出了在CrossPoint-378K数据集上训练的CroPond模型。我们的CroPond在CrossPoint-Bench上取得了最先进的性能，比Gemini-2.5-Pro高出39.7%的准确率，为推进跨视角对应研究提供了基础。该基准测试、数据集和模型可在https://github.com/WangYipu2002/CrossPoint公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型(VLMs)在不同视角间建立精确点级几何对应的能力不足问题，特别是在从粗粒度判断过渡到细粒度坐标预测方面的瓶颈。这个问题很重要，因为跨视角对应能力是空间理解和具身AI的基础，对于机器人导航、抓取和多智能体协作等关键任务至关重要，也是连接视觉理解和物理执行的桥梁，能让机器人更好地理解空间布局并执行实际任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受人类认知过程'感知-推理-对应'的启发，将跨视角对应分解为三个阶段：指令条件定位、可见性推理和对应。他们设计了包含四个评估维度的分层基准测试，系统评估整个推理链。数据集构建借鉴了现有的图像筛选、掩码生成和跨视角映射技术，模型训练则采用了监督微调与多源数据联合训练策略。作者借鉴了多视角理解、空间理解和数据构建等领域的现有研究成果，但针对点级对应这一特定挑战进行了创新设计。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是让模型能够跨越不同视角建立精确的点级几何对应关系，而非仅限于粗粒度区域对应；通过分层评估体系系统测试模型能力；专注于可操作区域而非静态区域；结合专门构建的跨视角对应数据与现有空间理解数据进行训练。整体流程包括：1)定义CVPC任务并分解为三个阶段；2)构建CrossPoint-Bench基准测试，包含四个评估任务；3)通过图像采样、区域分割、跨视角配对和问答生成构建CrossPoint-378K数据集；4)基于Qwen2.5-VL开发CroPond模型并进行多源联合训练；5)在多个基准测试和应用场景中评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统定义跨视角点对应(CVPC)任务，要求精确坐标预测；2)提出首个全面的CVPC基准测试CrossPoint-Bench，采用分层设计；3)构建首个大规模CVPC数据集CrossPoint-378K，专注于可操作区域；4)开发强基线模型CroPond，显著提升性能。相比之前工作，不同之处在于：任务粒度从多选题提升到点级定位；增强了语义相关性，专注于可操作区域；系统评估了模型对尺度变化和遮挡的鲁棒性；数据集明确围绕可操作区域构建；模型性能大幅提升，CroPond准确率达76.8%，远超现有模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统定义跨视角点对应任务、提出首个全面基准测试和数据集、开发强基线模型，显著提升了视觉语言模型在不同视角间建立精确点级几何对应的能力，为具身AI和空间理解研究奠定了重要基础。'}


### 论文摘要

Cross-view correspondence is a fundamental capability for spatial understanding and embodied AI. However, it is still far from being realized in Vision-Language Models (VLMs), especially in achieving precise point-level correspondence, which is crucial for precise affordance interaction. So we propose the Cross-View Point Correspondence (CVPC) task and CrossPoint-Bench, a comprehensive benchmark with hierarchical design, inspired by the human cognitive process of "perceive", "reason", and "correspond". Our evaluation shows the state-of-the-art models (e.g., Gemini-2.5-Pro) still fall far behind humans, with a gap of over 54.65% in overall accuracy, exposing a challenge in transitioning from coarse-grained judgement to fine-grained coordinate prediction. To address this problem, we construct CrossPoint-378K, a dataset with 378K question-answering pairs across 900 scenes, focused on actionable affordance regions that better reflect real-world manipulation and interaction scenarios. Furthermore, we propose CroPond that trained on the CrossPoint-378K dataset. Our CroPond achieves state-of-the-art performance on CrossPoint-Bench, surpassing Gemini-2.5-Pro by 39.7% accuracy, which offers a foundation for advancing future work on cross-view correspondence. The benchmark, dataset, and model are publicly available at https://github.com/WangYipu2002/CrossPoint.

---

## 134. Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding

**论文链接:** [http://arxiv.org/abs/2512.02487v1](http://arxiv.org/abs/2512.02487v1)

**作者:** Yerim Jeon, Miso Lee, WonJun Moon, Jae-Pil Heo

**发布时间:** 2025-12-02

### GPT解析

### 总结

论文提出了一种名为3D-SLIM的新型掩码策略，用于改进3D场景语言理解中的注意力机制，解决了现有方法中的两个基本冲突，并在各种3D场景语言任务中取得了显著的性能提升。

### 背景

最近的研究利用大型语言模型进行3D推理，但现有方法采用的标准解码器依赖因果注意力掩码，在3D场景理解中引入了两个基本冲突：无序3D对象之间的顺序偏差和受限制的对象-指令注意力，阻碍了特定任务的推理。

### 目的

为了克服现有方法的局限性，提出3D-SLIM掩码策略，用适应3D场景空间结构的自适应注意力掩码替代因果掩码，使模型能够根据对象的空间关系处理对象，同时被用户的任务指导。

### 方法

3D-SLIM引入两个关键组件：几何自适应掩码，基于空间密度而非标记顺序约束注意力；以及指令感知掩码，使对象标记能够直接访问指令上下文。该设计简单，不需要架构修改，也不添加额外参数。

### 主要发现

3D-SLIM在多种3D场景语言任务中产生了显著的性能提升，在多个基准测试和LLM基线上的广泛实验验证了其有效性，强调了解码器设计在3D多模态推理中的关键作用。

### 结论

通过引入3D-SLIM掩码策略，成功解决了现有3D场景语言理解方法中的两个基本冲突，改进了模型对3D场景的处理能力，同时保持了模型的简洁性和效率。

### 翻译

最近在3D场景语言理解方面的进展利用大型语言模型进行3D推理，通过将它们的一般推理能力转移到3D多模态上下文中。然而，现有方法通常采用语言建模中的标准解码器，这些解码器依赖因果注意力掩码。这种设计在3D场景理解中引入了两个基本冲突：无序3D对象之间的顺序偏差和受限制的对象-指令注意力，阻碍了特定任务的推理。为了克服这些限制，我们提出了3D空间语言指令掩码，这是一种有效的掩码策略，用适应3D场景空间结构的自适应注意力掩码替代了因果掩码。我们的3D-SLIM引入了两个关键组件：几何自适应掩码，基于空间密度而非标记顺序约束注意力；以及指令感知掩码，使对象标记能够直接访问指令上下文。这种设计使模型能够根据对象的空间关系处理对象，同时被用户的任务指导。3D-SLIM简单，不需要架构修改，也不添加额外参数，却在各种3D场景语言任务中产生了显著的性能提升。在多个基准测试和LLM基线上的广泛实验验证了其有效性，并强调了解码器设计在3D多模态推理中的关键作用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D场景-语言理解中标准语言模型解码器的两个基本冲突：一是因果掩码对顺序无关的3D对象引入了不必要的顺序依赖；二是限制了对象令牌和指令令牌之间的交互，阻碍了任务特定推理。这个问题很重要，因为3D场景-语言理解是机器人导航和具身智能等应用的基础，统一的多模态框架能处理多种任务，而现有研究忽视了解码器架构与3D数据不匹配的问题，解决它可以提高AI系统在复杂3D环境中的理解和推理能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别现有方法的两个冲突开始思考，意识到标准语言模型解码器的因果掩码与3D数据的本质特性不匹配。他们借鉴了人类解释3D场景的方式：通过空间邻近性分组对象，并根据语言线索关注相关区域。基于这些洞察，设计了两种掩码机制。该方法借鉴了现有的对象中心3D LLM框架（如Chat-Scene、Inst3D-LMM），利用了现有的3D场景表示方法，采用了标准的交叉熵损失作为训练目标，并在多个基准测试上进行了评估。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是替换标准语言模型解码器中的因果掩码，设计一种适应3D场景空间结构的自适应注意力掩码，使模型能够基于对象的空间关系处理它们，同时被用户的任务引导。整体流程包括：1)输入处理，将3D场景分解为对象提议并形成多模态序列；2)Geometry-adaptive Mask实现，计算局部密度，确定自适应邻居范围；3)Instruction-aware Mask实现，允许对象令牌关注指令令牌；4)使用交叉熵损失进行训练；5)应用新掩码策略处理输入并生成输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)识别了现有框架中的两个基本冲突；2)提出了3D-SLIM掩码策略，无需修改架构或添加参数；3)设计了两种专门掩码组件。相比之前的工作，不同之处在于：研究重点从输入表示转向解码器设计；用自适应注意力掩码替代标准因果掩码；考虑3D场景的空间结构和对象间关系；允许对象和指令令牌之间的直接交互，而非强制处理完所有对象后才考虑指令。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了3D-SLIM，一种简单有效的自适应注意力掩码策略，通过解决现有语言模型解码器在3D场景理解中的顺序偏差和对象-指令交互限制问题，显著提升了大型语言模型在3D场景-语言理解任务中的空间推理能力，无需修改架构或添加额外参数。'}


### 论文摘要

Recent advances in 3D scene-language understanding have leveraged Large Language Models (LLMs) for 3D reasoning by transferring their general reasoning ability to 3D multi-modal contexts. However, existing methods typically adopt standard decoders from language modeling, which rely on a causal attention mask. This design introduces two fundamental conflicts in 3D scene understanding: sequential bias among order-agnostic 3D objects and restricted object-instruction attention, hindering task-specific reasoning. To overcome these limitations, we propose 3D Spatial Language Instruction Mask (3D-SLIM), an effective masking strategy that replaces the causal mask with an adaptive attention mask tailored to the spatial structure of 3D scenes. Our 3D-SLIM introduces two key components: a Geometry-adaptive Mask that constrains attention based on spatial density rather than token order, and an Instruction-aware Mask that enables object tokens to directly access instruction context. This design allows the model to process objects based on their spatial relationships while being guided by the user's task. 3D-SLIM is simple, requires no architectural modifications, and adds no extra parameters, yet it yields substantial performance improvements across diverse 3D scene-language tasks. Extensive experiments across multiple benchmarks and LLM baselines validate its effectiveness and underscore the critical role of decoder design in 3D multi-modal reasoning.

---

## 135. Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration

**论文链接:** [http://arxiv.org/abs/2512.02458v1](http://arxiv.org/abs/2512.02458v1)

**作者:** Zhongyi Cai, Yi Du, Chen Wang, Yu Kong

**发布时间:** 2025-12-02

### GPT解析

### 总结

该研究关注连续具身任务中的空间知识重用问题，提出了SEER-Bench基准测试和3DSPMR方法来解决这一挑战。

### 背景

现有室内具身任务研究通常要求智能体主动探索未知环境并推理场景以实现特定目标，但在实际应用中，智能体往往面临连续任务，每个新子任务在前一个任务完成后开始，且某些子任务可能不可行。

### 目的

研究连续具身任务中如何重用先前探索积累的空间知识来支持后续推理和探索这一尚未被充分探索但具有重要实际意义的具身AI挑战。

### 方法

引入SEER-Bench（连续具身探索与推理基准测试），包含具身问答(EQA)和具身多模态导航(EMN)两个经典任务；提出3DSPMR（3D空间记忆推理）方法，利用已探索区域的关系、视觉和几何线索来增强多模态大型语言模型(MLLMs)在连续具身任务中的能力。

### 主要发现

大量实验验证了3DSPMR在连续EQA和EMN任务上都取得了显著的性能提升，首次将几何信息明确整合到基于MLLM的空间理解和推理中。

### 结论

该研究通过整合几何信息提高了MLLM在连续具身任务中的空间理解和推理能力，解决了空间知识重用的核心挑战。

### 翻译

现有关于室内具身任务的研究通常要求智能体主动探索未知环境并推理场景以实现特定目标。然而，在实际部署时，智能体经常面临连续任务，其中每个新子任务在前一个任务完成后开始，且某些子任务可能不可行，例如寻找不存在的物体。与单任务设置相比，核心挑战在于重用先前探索积累的空间知识来支持后续的推理和探索。在本工作中，我们研究了这一尚未被充分探索但具有重要实际意义的具身AI挑战。为评估这一挑战，我们引入了SEER-Bench，一个新的连续具身探索与推理基准测试，包含两个经典具身任务：具身问答(EQA)和具身多模态导航(EMN)。基于SEER-Bench，我们提出了3DSPMR，一种3D空间记忆推理方法，利用已探索区域的关系、视觉和几何线索来增强多模态大型语言模型(MLLMs)在连续具身任务中的推理和探索能力。据我们所知，这是首次将几何信息明确整合到基于MLLM的空间理解和推理中的工作。大量实验验证了3DSPMR在连续EQA和EMN任务上都取得了显著的性能提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决智能体在连续具身任务中如何有效重用先前探索积累的空间知识，并识别不可行任务的问题。这个问题很重要，因为现实世界中的智能体往往需要处理一系列连续任务，而不是孤立的单个任务，且某些任务可能实际上不可行（如搜索不存在的物体）。现有方法在处理这类连续任务时面临记忆设计、过度自信和无法识别不可行任务等挑战，限制了智能体在实际应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有基准测试主要关注单一任务场景，缺乏对连续任务和不可行任务的研究。他们提出了SEER-Bench基准测试，并针对MLLMs在空间理解和推理方面的局限性，设计了统一的空间记忆方法。作者借鉴了场景图方法编码对象关系、自我中心视觉记忆保留过去观察、3D语义地图构建等现有工作，但创新性地将这些互补模态整合到统一的3D空间记忆中，并结合了几何信息来增强MLLMs的空间理解能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个统一的空间记忆，整合全局关系、局部视觉和几何信息，以增强MLLMs在连续具身任务中的空间理解和推理能力。整体流程包括：1) 构建统一空间记忆（全局场景图、局部关键帧记忆和几何覆盖图）；2) 具身推理框架（自适应推理模块和几何检查机制）；3) 几何-语义探索模块（前沿检测、多因素评分、目标选择和导航）。这种方法通过几何信息连接全局和局部表示，使智能体能更有效地重用空间知识并识别不可行任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出SEER-Bench基准测试，首次关注连续具身探索和推理；2) 提出统一的多线索空间记忆，整合关系、视觉和几何信息；3) 开发3DSPMR方法，包含自适应推理模块、几何检查机制和几何-语义探索模块。相比之前工作，本文首次将几何信息明确纳入基于MLLM的空间理解和推理，解决了现有方法在连续任务中的三个主要挑战，并引入了新的评估指标（SSR和SSPL）来更好地评估长时程性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了SEER-Bench基准测试和3DSPMR方法，通过整合关系、视觉和几何信息的统一空间记忆，显著提升了智能体在连续具身任务中的推理可靠性、任务可行性和探索效率。'}


### 论文摘要

Existing research on indoor embodied tasks typically requires agents to actively explore unknown environments and reason about the scene to achieve a specific goal. However, when deployed in real life, agents often face sequential tasks, where each new sub-task follows the completion of the previous one, and certain sub-tasks may be infeasible, such as searching for a non-existent object. Compared with the single-task setting, the core challenge lies in reusing spatial knowledge accumulated from previous explorations to support subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. To evaluate this challenge, we introduce SEER-Bench, a new Sequential Embodied Exploration and Reasoning Benchmark encompassing encompassing two classic embodied tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). Building on SEER-Bench, we propose 3DSPMR, a 3D SPatial Memory Reasoning approach that exploits relational, visual, and geometric cues from explored regions to augment Multi-Modal Large Language Models (MLLMs) for reasoning and exploration in sequential embodied tasks. To the best of our knowledge, this is the first work to explicitly incorporate geometric information into MLLM-based spatial understanding and reasoning. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.

---

## 136. HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild

**论文链接:** [http://arxiv.org/abs/2512.02450v1](http://arxiv.org/abs/2512.02450v1)

**作者:** Valentin Bieri, Marie-Julie Rakotosaona, Keisuke Tateno, Francis Engelmann, Leonidas Guibas

**发布时间:** 2025-12-02

**备注:** NeurIPS 2025 (Datasets and Benchmarks Track) Project Page: https://houselayout3d.github.io

### GPT解析

### 总结

这篇论文介绍了HouseLayout3D真实世界基准数据集和MultiFloor3D无需训练基线方法，用于解决现有3D布局估计模型在处理多层建筑时的局限性。

### 背景

当前3D布局估计模型主要在简单单房间或单楼层的人工合成数据集上训练，无法原生处理大型多层建筑，需要分割场景为单独楼层，这移除了理解连接多级别结构所需的全局空间上下文。

### 目的

引入支持完整建筑规模布局估计的真实世界基准数据集，包括多层和建筑复杂空间，并提出有效方法解决现有模型的局限性。

### 方法

提出HouseLayout3D真实世界基准数据集和MultiFloor3D无需训练基线方法，后者利用最新场景理解技术处理多层建筑布局估计。

### 主要发现

MultiFloor3D方法已在新的HouseLayout3D基准和先前数据集上优于现有3D布局估计模型，突显了该领域需要进一步研究。

### 结论

通过真实世界基准和简单有效方法，论文展示了现有模型的局限并指明了未来研究方向，数据和代码已公开。

### 翻译

当前的3D布局估计模型主要在包含简单单房间或单楼层环境的人工合成数据集上进行训练。因此，它们无法原生处理大型多层建筑，需要在处理前将场景分割为单独的楼层，这移除了对理解连接多个级别的楼梯等结构至关重要的全局空间上下文。在这项工作中，我们介绍了HouseLayout3D，一个真实世界的基准数据集，旨在支持向完整的建筑规模布局估计进展，包括多层和建筑复杂空间。我们还提出了MultiFloor3D，一个简单的无需训练的基线方法，它利用最新的场景理解方法，并且已经在我们的基准和先前数据集上优于现有的3D布局估计模型，突显了进一步研究这个方向的必要性。数据和代码可在https://houselayout3d.github.io获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决当前3D布局估计模型无法处理大型多楼层建筑的问题。现有模型主要在简单单房间或单层环境的合成数据集上训练，需要将场景分割成单独楼层处理，这会移除推理楼梯等结构所必需的全局空间上下文。这个问题很重要，因为空间感知周围3D布局是许多感知算法和机器人系统的关键要求，缺乏对大型多楼层建筑的处理能力限制了3D布局预测在真实世界场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有数据集缺乏大型多楼层建筑的多样性，因此创建了HOUSELAYOUT3D基准数据集，包含真实世界扫描和详细手动标注。他们设计了MULTIFLOOR3D方法，这是一种无需训练的基线方法，结合了现代3D场景重建和布局拟合策略。该方法借鉴了现有工作，如使用DN-Splatter进行3D网格重建，OneFormer进行2D分割，Hov-SG进行房间分割，以及DBSCAN进行窗口检测，但将这些技术整合成一个专门针对多楼层建筑布局估计的新流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合现代3D场景重建和布局拟合策略，创建一个简单但有效的无需训练的方法来处理大型多楼层建筑的3D布局估计。整体实现流程分为四个阶段：1) 从RGB图像重建3D网格；2) 使用预训练的2D分割模型提取主要结构元素形成布局骨架；3) 使用几何和语义信息优化布局骨架，修复孔洞和未观察区域，推断更完整的布局原型；4) 将布局原型转换为场景图，然后转换为最终的3D布局。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) HOUSELAYOUT3D数据集，第一个用于大型多楼层建筑3D布局估计的基准数据集；2) MULTIFLOOR3D方法，一种无需训练的基线方法；3) 能够处理复杂多楼层结构，保留全局空间上下文；4) 提供详细的建筑结构标注，包括墙壁、地板、天花板、楼梯、窗户和门。相比之前的工作，本文的方法不需要将场景分割成单独楼层处理，能够处理楼梯等连接多个楼层的结构，并且在真实世界多楼层建筑上表现更好，而现有方法主要在简单单房间或单层环境的合成数据集上训练，缺乏多样性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了HOUSELAYOUT3D基准数据集和无需训练的MULTIFLOOR3D方法，解决了现有3D布局估计模型无法处理大型多楼层建筑的问题，显著提升了在复杂建筑结构中的布局估计性能。'}


### 论文摘要

Current 3D layout estimation models are primarily trained on synthetic datasets containing simple single room or single floor environments. As a consequence, they cannot natively handle large multi floor buildings and require scenes to be split into individual floors before processing, which removes global spatial context that is essential for reasoning about structures such as staircases that connect multiple levels. In this work, we introduce HouseLayout3D, a real world benchmark designed to support progress toward full building scale layout estimation, including multiple floors and architecturally intricate spaces. We also present MultiFloor3D, a simple training free baseline that leverages recent scene understanding methods and already outperforms existing 3D layout estimation models on both our benchmark and prior datasets, highlighting the need for further research in this direction. Data and code are available at: https://houselayout3d.github.io.

---

## 137. OpenREAD: Reinforced Open-Ended Reasoning for End-to-End Autonomous Driving with LLM-as-Critic

**论文链接:** [http://arxiv.org/abs/2512.01830v2](http://arxiv.org/abs/2512.01830v2)

**作者:** Songyan Zhang, Wenhui Huang, Zhan Chen, Chua Jiahao Collister, Qihang Huang, Chen Lv

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为OpenREAD的开放式推理强化视觉语言模型框架，用于自动驾驶领域。该框架通过构建大规模思维链注释和使用Qwen3大语言模型作为强化微调中的评判者，实现了从高层推理到低层轨迹规划的端到端强化微调，显著提升了自动驾驶系统在推理和规划任务上的性能。

### 背景

目前，两阶段微调策略（包括监督微调SFT和强化微调RFT）在推进知识驱动的自动驾驶范式方面显示出巨大潜力。然而，SFT的学习性质限制了推理的泛化能力，从而限制了驾驶性能的充分发挥。同时，现有的RFT方法主要应用于下游任务，因为场景理解是一个开放性问题，相应的奖励难以量化。

### 目的

为了解决现有方法的局限性，研究旨在提出一个能够实现从高层推理到低层轨迹规划的全谱系端到端RFT的框架，使系统能够更好地处理开放性问题并提升整体性能。

### 方法

作者提出了OpenREAD框架，具体方法包括：1) 在开源驾驶相关知识数据集上构建大规模思维链(CoT)注释；2) 使用强大的Qwen3大语言模型作为RFT中的评判者，对开放性问题进行奖励建模，量化推理质量；3) 实现联合端到端RFT，同时提升上游和下游任务性能。

### 主要发现

广泛的实验证实，联合端到端RFT在上游和下游任务中都带来了显著改进，使OpenREAD在推理和规划基准测试上达到了最先进的性能。

### 结论

OpenREAD框架成功地解决了自动驾驶领域中开放性推理和奖励量化的问题，通过端到端的强化微调策略，实现了从高层推理到低层轨迹规划的全面优化，为自动驾驶系统的发展提供了新的思路和方法。

### 翻译

最近，两阶段微调策略，例如通过监督微调获取必要的驾驶知识，再通过强化微调进一步决策和规划，在推进知识驱动的自动驾驶范式方面显示出巨大潜力。然而，监督微调的学习性质仍然限制了推理的泛化能力，从而限制了驾驶性能的充分发挥。同时，当前的强化微调方法主要应用于下游任务，因为场景理解是一个开放性问题，相应的奖励难以量化。为了解决这些局限性，我们提出了OpenREAD，一个基于开放式推理强化视觉语言模型的自动驾驶框架，实现了从高层推理到低层轨迹规划的全谱系端到端强化微调。具体而言，我们首先在开源驾驶相关知识数据集上构建大规模思维链注释，并使用强大的Qwen3大语言模型作为强化微调中的评判者，在奖励建模过程中对开放性问题的推理质量进行量化。大量实验证实，联合端到端强化微调在上游和下游任务中都带来了显著改进，使OpenREAD在推理和规划基准测试上达到了最先进的性能。


### 论文摘要

Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.

---

## 138. SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge

**论文链接:** [http://arxiv.org/abs/2512.01629v2](http://arxiv.org/abs/2512.01629v2)

**作者:** Yumeng He, Ying Jiang, Jiayin Lu, Yin Yang, Chenfanfu Jiang

**发布时间:** 2025-12-01

**备注:** Project page: https://heyumeng.com/SPARK/index.html. 17 pages, 7 figures

### GPT解析

### 总结

SPARK是一个创新的框架，能够从单张RGB图像重建物理一致的运动学部分级关节式3D对象，结合了视觉语言模型、生成式扩散变压器和可微分渲染技术，生成高质量模拟就绪的资产。

### 背景

关节式3D对象对具身AI、机器人和交互式场景理解至关重要，但创建模拟就绪的资产仍然劳动密集，需要专家对部分层次结构和运动结构进行建模。

### 目的

介绍SPARK框架，用于从单个RGB图像重建物理一致的运动学部分级关节式对象，简化创建过程。

### 方法

首先利用VLMs提取粗略的URDF参数并生成部分级参考图像，然后将部分图像指导和推断的结构图集成到生成式扩散变压器中合成一致形状，最后结合可微分前向运动学和可微分渲染来优化关节参数。

### 主要发现

广泛的实验表明，SPARK能够跨不同类别生成高质量的、模拟就绪的关节式资产，使机器人操作和交互建模等下游应用成为可能。

### 结论

SPARK框架有效地解决了从单张RGB图像重建关节式3D对象的挑战，生成的资产质量高，可用于各种应用场景。

### 翻译

关节式3D对象对具身AI、机器人和交互式场景理解至关重要，但创建模拟就绪的资产仍然劳动密集，需要专家对部分层次结构和运动结构进行建模。我们介绍了SPARK，一个用于从单个RGB图像重建物理一致、运动学部分级关节式对象的框架。给定输入图像，我们首先利用VLMs提取粗略的URDF参数并生成部分级参考图像。然后我们将部分图像指导和推断的结构图集成到生成式扩散变压器中，以合成一致的部件和关节式对象的完整形状。为了进一步优化URDF参数，我们结合了可微分前向运动学和可微分渲染，在VLM生成的开放状态监督下优化关节类型、轴和原点。大量实验表明，SPARK能够跨不同类别生成高质量、模拟就绪的关节式资产，使机器人操作和交互建模等下游应用成为可能。项目页面：https://heyumeng.com/SPARK/index.html。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从单个RGB图像重建物理一致的运动学零件级关节化物体的问题。这个问题在具身AI、机器人和交互场景理解中非常重要，因为创建仿真就绪的关节化3D资产目前仍然劳动密集，需要专家对零件层次结构和运动结构进行建模。高质量、可交互的3D资产对于机器人操作、动画和仿真应用至关重要，但现有方法要么生成融合的难以重用的形状，要么生成的零件缺乏运动学一致性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到关节化3D物体重建的挑战：传统方法劳动密集，现有生成模型产生融合形状难以重用，零件级生成缺乏运动学一致性。作者借鉴了多项现有工作：使用VLM提取URDF参数和生成参考图像，采用扩散变换器合成一致形状，应用可微分前向运动学和渲染优化关节参数。设计思路是先利用VLM获取结构信息，再通过DiT生成几何形状，最后优化关节参数，形成完整流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视觉-语言模型(VLM)的先验知识与扩散变换器(DiT)结合，利用VLM提取的结构信息指导高质量关节化物体的生成。整体流程分为三阶段：1)VLM引导的结构推理：提取粗略URDF参数，生成零件参考图像，构建结构图；2)零件-关节化物体生成：使用DiT在多级注意力机制指导下生成3D零件网格并组装；3)关节优化：通过可微分前向运动学和渲染优化关节参数。最后应用纹理完成整个重建过程。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出结合VLM先验与扩散变换器的新颖框架，同时生成高质量几何和准确URDF参数；2)引入零件图像指导和多级注意力机制实现一致的多零件合成；3)开发关节优化组件在VLM监督下 refine运动学参数。相比之前工作，SPARK无需专家建模，仅需单张图像输入，生成的零件具有运动学一致性且可直接用于仿真，解决了融合形状难以重用和零件分割忽略运动结构的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPARK通过整合视觉-语言模型先验、扩散变换器和可微分优化，实现了从单张图像中高质量重建具有准确运动学参数的仿真就绪关节化物体，为具身AI和机器人应用提供了重要工具。'}


### 论文摘要

Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling. Project page: https://heyumeng.com/SPARK/index.html.

---

## 139. DAGLFNet: Deep Feature Attention Guided Global and Local Feature Fusion for Pseudo-Image Point Cloud Segmentation

**论文链接:** [http://arxiv.org/abs/2510.10471v2](http://arxiv.org/abs/2510.10471v2)

**作者:** Chuang Chen, Yi Lin, Bo Wang, Jing Hu, Xi Wu, Wenyi Ge

**发布时间:** 2025-10-12

### GPT解析

### 总结

该研究提出了一种名为DAGLFNet的基于伪图像的语义分割框架，用于解决环境感知系统中处理非结构化点云并提取结构化语义信息的挑战。该方法通过三个关键组件提高了特征判别性，并在SemanticKITTI和nuScenes数据集上取得了优异的性能。

### 背景

环境感知系统对于高精度地图绘制和自主导航至关重要，LiDAR作为核心传感器提供准确的3D点云数据。有效处理非结构化点云同时提取结构化语义信息仍然是一个重大挑战。近年来，许多基于伪图像的表示方法出现，通过融合3D点云和2D网格来平衡效率和性能。

### 目的

解决伪图像表示与原始3D信息之间的根本不一致问题，这种不一致严重损害了2D-3D特征融合，导致特征判别性差。提出DAGLFNet框架以提取判别性特征。

### 方法

DAGLFNet包含三个关键组件：1) 全局-局部特征融合编码(GL-FFE)模块，增强集合内局部特征相关性并捕获全局上下文信息；2) 多分支特征提取(MB-FE)网络，捕获更丰富的邻域信息并提高轮廓特征的判别性；3) 基于深度特征引导注意力的特征融合(FFDFA)机制，优化跨通道特征融合精度。

### 主要发现

实验评估表明，DAGLFNet在SemanticKITTI和nuScenes的验证集上分别实现了69.9%和78.7%的平均交并比(mIoU)分数，证明了该方法在准确性和效率之间实现了良好的平衡。

### 结论

DAGLFNet成功解决了伪图像表示与原始3D信息之间的不一致问题，通过三个关键组件有效提高了特征判别性，为环境感知系统提供了一种高效且准确的解决方案。

### 翻译

环境感知系统对于高精度地图绘制和自主导航至关重要，LiDAR作为核心传感器提供准确的3D点云数据。有效处理非结构化点云同时提取结构化语义信息仍然是一个重大挑战。近年来，许多基于伪图像的表示方法出现，通过融合3D点云和2D网格来平衡效率和性能。然而，伪图像表示与原始3D信息之间的根本不一致严重损害了2D-3D特征融合，这是信息融合的主要障碍，导致特征判别性差。本研究提出了DAGLFNet，一种基于伪图像的语义分割框架，旨在提取判别性特征。它包含三个关键组件：首先，全局-局部特征融合编码(GL-FFE)模块，用于增强集合内局部特征相关性并捕获全局上下文信息；其次，多分支特征提取(MB-FE)网络，用于捕获更丰富的邻域信息并提高轮廓特征的判别性；第三，基于深度特征引导注意力的特征融合(FFDFA)机制，用于优化跨通道特征融合精度。实验评估表明，DAGLFNet在SemanticKITTI和nuScenes的验证集上分别实现了69.9%和78.7%的平均交并比(mIoU)分数。该方法在准确性和效率之间实现了良好的平衡。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云语义分割中伪图像表示方法存在的问题，包括深度冲突、边界模糊和特征退化等问题。这个问题在现实中非常重要，因为LiDAR是自动驾驶和高精度地图绘制的核心传感器，而点云语义分割是机器人环境感知和自主导航的关键技术。高效准确地处理点云数据对于自动驾驶汽车理解周围环境、做出安全决策至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有LiDAR点云语义分割方法的优缺点，指出基于点的方法计算密集，基于体素的方法内存消耗大，而基于范围视图的方法虽然平衡了效率和性能但存在深度冲突和细节丢失问题。针对这些局限，作者设计了DAGLFNet框架，借鉴了现有工作中的范围视图表示、注意力机制和多分支特征提取等思想，但创新性地引入深度信息作为动态调制因子，专门针对点云特性进行了改进设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过全局-局部特征融合捕获上下文和局部几何关系，通过多分支特征提取扩大感受野并增强边界特征，通过深度特征引导的注意力机制优化特征融合。整体流程包括：1)特征编码阶段对点云分组并提取点级和组级特征；2)图像特征提取阶段通过MB-FE网络处理组级特征；3)特征更新阶段通过深度引导的融合机制更新特征；4)融合头模块聚合多阶段特征生成最终语义预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)GL-FFE模块同时捕获全局上下文和局部几何关系；2)MB-FE网络通过三并行分支扩大感受野并增强边界特征；3)FFDFA机制利用深度信息作为权重约束提高特征融合精度。相比之前的工作，该方法不仅处理投影点还保留原始点云信息，解决了深度冲突问题；通过深度感知的注意力机制而非简单特征拼接，更有效处理稀疏和遮挡区域；专注于单一数据源避免了多源数据对齐挑战。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DAGLFNet通过全局-局部特征融合、多分支特征提取和深度特征引导的注意力机制，有效解决了LiDAR点云伪图像表示中的深度冲突、边界模糊和特征退化问题，实现了高效且准确的点云语义分割。'}


### 论文摘要

Environmental perception systems are crucial for high-precision mapping and autonomous navigation, with LiDAR serving as a core sensor providing accurate 3D point cloud data. Efficiently processing unstructured point clouds while extracting structured semantic information remains a significant challenge. In recent years, numerous pseudo-image-based representation methods have emerged to balance efficiency and performance by fusing 3D point clouds with 2D grids. However, the fundamental inconsistency between the pseudo-image representation and the original 3D information critically undermines 2D-3D feature fusion, posing a primary obstacle for coherent information fusion and leading to poor feature discriminability. This work proposes DAGLFNet, a pseudo-image-based semantic segmentation framework designed to extract discriminative features. It incorporates three key components: first, a Global-Local Feature Fusion Encoding (GL-FFE) module to enhance intra-set local feature correlation and capture global contextual information; second, a Multi-Branch Feature Extraction (MB-FE) network to capture richer neighborhood information and improve the discriminability of contour features; and third, a Feature Fusion via Deep Feature-guided Attention (FFDFA) mechanism to refine cross-channel feature fusion precision. Experimental evaluations demonstrate that DAGLFNet achieves mean Intersection-over-Union (mIoU) scores of 69.9% and 78.7% on the validation sets of SemanticKITTI and nuScenes, respectively. The method achieves an excellent balance between accuracy and efficiency.

---

## 140. RangeSAM: On the Potential of Visual Foundation Models for Range-View represented LiDAR segmentation

**论文链接:** [http://arxiv.org/abs/2509.15886v3](http://arxiv.org/abs/2509.15886v3)

**作者:** Paul Julius Kühn, Duc Anh Nguyen, Arjan Kuijper, Holger Graf, Saptarshi Neil Sinha

**发布时间:** 2025-09-19

### GPT解析

### 总结

本研究探讨了使用视觉基础模型SAM2作为LiDAR点云范围视图分割的骨干网络，通过优化编码器架构实现了高效准确的3D点云分割。

### 背景

点云分割对自动驾驶和3D场景理解至关重要，但现有的体素法和点法虽能捕获精细几何信息，却存在计算成本高、内存访问不规则和实时效率有限等问题。相比之下，范围视图方法相对未被充分探索，但可以利用成熟的2D语义分割技术实现快速准确预测。

### 目的

研究当前最先进的视觉基础模型SAM2是否可以作为LiDAR点云范围视图分割的强大骨干网络，探索VFMs作为3D感知通用骨干的可行性。

### 方法

提出了第一个适应SAM2进行3D分割的范围视图框架，结合高效的2D特征提取和标准投影/反投影操作。对编码器进行了三种架构修改：(1)强调LiDAR范围图像中水平空间依赖性的新模块；(2)针对球形投影几何特性的定制配置；(3)专门设计用于捕获范围视图伪图像中独特空间模式和断点的调整机制。

### 主要发现

该方法在SemanticKITTI数据集上取得了有竞争力的性能，同时受益于2D为中心的管道的速度、可扩展性和部署简便性，证明了范围视图分割方法使用VFMs可以取得有前景的结果。

### 结论

该研究强调了VFMs作为3D感知通用骨干的可行性，为统一的基础模型驱动的LiDAR分割开辟了道路，展示了将先进2D技术应用于3D感知任务的潜力。

### 翻译

点云分割对自动驾驶和3D场景理解至关重要。虽然最近的体素法和点法由于它们与深度架构的兼容性和捕获精细几何的能力而主导研究，但它们通常带来高计算成本、不规则的内存访问和有限的实时效率。相比之下，范围视图方法虽然相对未被充分探索，但可以利用成熟的2D语义分割技术进行快速准确的预测。受视觉基础模型(VFMs)在字幕、零样本识别和多模态任务中的快速进展启发，我们研究了SAM2（当前最先进的分割VFM）是否可以作为LiDAR点云在范围视图中的分割的强大骨干。我们提出了，据我们所知，第一个将SAM2适应3D分割的范围视图框架，结合高效的2D特征提取和标准投影/反投影来操作点云。为了优化SAM2的范围视图表示，我们对编码器实现了几种架构修改：(1)一个强调LiDAR范围图像中固有水平空间依赖性的新模块；(2)针对球形投影几何特性定制的配置；(3)编码器骨干中专门设计用于捕获范围视图伪图像中存在的独特空间模式和断点的调整机制。我们的方法在SemanticKITTI上取得了有竞争力的性能，同时受益于2D为中心的管道的速度、可扩展性和部署简便性。这项工作强调了VFMs作为3D感知通用骨干的可行性，并为统一的基础模型驱动的LiDAR分割开辟了道路。结果让我们得出结论，使用VFMs的范围视图分割方法带来了有前景的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云分割的计算效率和准确性问题。当前主流的点云分割方法（如基于体素和点的方法）虽然性能好，但存在高计算成本、不规则内存访问和有限的运行效率问题。这个问题在自动驾驶和3D场景理解中至关重要，因为高效准确的点云分割是实现实时环境感知和决策的基础。范围视图方法虽然潜力巨大但研究不足，而视觉基础模型在2D领域表现出色，探索其在3D领域的应用具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了点云分割的挑战和范围视图方法的潜力，注意到视觉基础模型在2D领域的强大能力。他们选择SAM2作为基础模型，因为它是当前最先进的VFM分割模型。设计过程中，作者借鉴了SAM2架构、Receptive Field Blocks(RFB)特征解码方法、k-NN插值后处理技术和多种数据增强策略。同时，作者进行了创新性调整，设计了新的Stem模块强调水平空间依赖，自定义了Hiera块配置以适应球形投影的几何特性，并调整了窗口注意力机制以捕捉范围视图中的独特空间模式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型(SAM2)的强大分割能力，通过将3D点云投影为2D范围视图表示，将3D问题转化为2D问题，并设计特定的架构修改以适应范围视图数据的特性。整体流程包括：1)预处理将无序LiDAR扫描转换为范围视图表示；2)模型处理包括Stem模块、基于Hiera主干的编码器和RFB解码器；3)后处理通过k-NN插值进行标签传播；4)训练使用复合损失函数和数据增强策略提高模型泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首创RangeSAM框架，首次将SAM2模型应用于3D点云分割；2)架构创新，包括新的Stem模块、自定义Hiera块配置和调整的窗口注意力机制；3)方法论创新，通过投影/反投影结合2D特征提取与点云处理；4)训练策略创新，采用多数据集训练优于传统2D预训练。相比之前的工作，RangeSAM专注于范围视图表示而非体素或点表示，利用成熟的2D分割技术而非复杂3D操作，直接在3D数据上训练而非依赖2D预训练，并应用最新的视觉基础模型而非从头训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RangeSAM首次将视觉基础模型SAM2成功应用于范围视图表示的LiDAR点云分割，通过创新的架构设计和训练策略，实现了与最先进方法相竞争的性能，同时保持了2D方法的效率和可扩展性。'}


### 论文摘要

Point cloud segmentation is central to autonomous driving and 3D scene understanding. While voxel- and point-based methods dominate recent research due to their compatibility with deep architectures and ability to capture fine-grained geometry, they often incur high computational cost, irregular memory access, and limited real-time efficiency. In contrast, range-view methods, though relatively underexplored - can leverage mature 2D semantic segmentation techniques for fast and accurate predictions. Motivated by the rapid progress in Visual Foundation Models (VFMs) for captioning, zero-shot recognition, and multimodal tasks, we investigate whether SAM2, the current state-of-the-art VFM for segmentation tasks, can serve as a strong backbone for LiDAR point cloud segmentation in the range view. We present , to our knowledge, the first range-view framework that adapts SAM2 to 3D segmentation, coupling efficient 2D feature extraction with standard projection/back-projection to operate on point clouds. To optimize SAM2 for range-view representations, we implement several architectural modifications to the encoder: (1) a novel module that emphasizes horizontal spatial dependencies inherent in LiDAR range images, (2) a customized configuration of tailored to the geometric properties of spherical projections, and (3) an adapted mechanism in the encoder backbone specifically designed to capture the unique spatial patterns and discontinuities present in range-view pseudo-images. Our approach achieves competitive performance on SemanticKITTI while benefiting from the speed, scalability, and deployment simplicity of 2D-centric pipelines. This work highlights the viability of VFMs as general-purpose backbones for 3D perception and opens a path toward unified, foundation-model-driven LiDAR segmentation. Results lets us conclude that range-view segmentation methods using VFMs leads to promising results.

---

## 141. PointVDP: Learning View-Dependent Projection by Fireworks Rays for 3D Point Cloud Segmentation

**论文链接:** [http://arxiv.org/abs/2507.06618v2](http://arxiv.org/abs/2507.06618v2)

**作者:** Yang Chen, Yueqi Duan, Haowen Sun, Ziwei Wang, Jiwen Lu, Yap-Peng Tan

**发布时间:** 2025-07-09

**备注:** This version needs major revision

### GPT解析

### 总结

本文提出视图依赖投影(VDP)方法，通过数据驱动的3D到2D映射和颜色正则化优化，实现了高效的点云分割，在保持竞争力的同时降低了计算成本。

### 背景

现有基于投影的方法在复杂场景中使用视图无关的投影，依赖直线生成直接射线或使用向上曲线减少遮挡，但这种视图无关性限制了投影射线只能使用预设参数，无法捕获不同视图平面上的足够投影多样性。

### 目的

设计从3D点分布生成数据驱动投影的框架，通过预测受烟花自适应行为启发的射线，生成信息量大的单图像输入；构建颜色正则化优化框架，最大化投影图像中2D空间利用率。

### 方法

提出视图依赖投影(VDP)方法，设计高效的3D到2D映射，动态适应视图变化的空间几何；构建颜色正则化，强调语义像素中的重要特征，抑制黑色像素中的非语义特征；开发名为PointVDP的轻量级投影方法。

### 主要发现

VDP方法在边际计算成本下开发轻量级投影，在S3DIS和ScanNet基准测试上取得了具有竞争力的结果。

### 结论

PointVDP提供了一种资源高效的语义理解解决方案，能够在保持分割性能的同时降低计算开销。

### 翻译

在本文中，我们提出视图依赖投影(VDP)以促进点云分割，设计高效的3D到2D映射，动态适应视图变化的空间几何。现有的基于投影的方法在复杂场景中利用视图无关投影，依赖直线生成直接射线或向上曲线来减少遮挡。然而，它们的视图无关性提供的投影射线仅限于人类设置预定义的参数，限制了点的感知能力，无法在不同视图平面上捕获足够的投影多样性。虽然每个视图平面通常使用多个投影来增强空间多样性，但投影冗余导致图像处理中过度的计算开销和低效率。为解决这些限制，我们设计了VDP框架，从3D点分布生成数据驱动的投影，通过预测受烟花自适应行为启发的射线，产生信息量大的单图像输入。此外，我们构建颜色正则化来优化框架，强调语义像素中的基本特征并抑制黑色像素中的非语义特征，从而最大化投影图像中的2D空间利用率。因此，我们的方法PointVDP在边际计算成本下开发了轻量级投影。在S3DIS和ScanNet基准上的实验表明，我们的方法取得了具有竞争力的结果，为语义理解提供了资源高效的解决方案。


### 论文摘要

In this paper, we propose view-dependent projection (VDP) to facilitate point cloud segmentation, designing efficient 3D-to-2D mapping that dynamically adapts to the spatial geometry from view variations. Existing projection-based methods leverage view-independent projection in complex scenes, relying on straight lines to generate direct rays or upward curves to reduce occlusions. However, their view independence provides projection rays that are limited to pre-defined parameters by human settings, restricting point awareness and failing to capture sufficient projection diversity across different view planes. Although multiple projections per view plane are commonly used to enhance spatial variety, the projected redundancy leads to excessive computational overhead and inefficiency in image processing. To address these limitations, we design a framework of VDP to generate data-driven projections from 3D point distributions, producing highly informative single-image inputs by predicting rays inspired by the adaptive behavior of fireworks. In addition, we construct color regularization to optimize the framework, which emphasizes essential features within semantic pixels and suppresses the non-semantic features within black pixels, thereby maximizing 2D space utilization in a projected image. As a result, our approach, PointVDP, develops lightweight projections in marginal computation costs. Experiments on S3DIS and ScanNet benchmarks show that our approach achieves competitive results, offering a resource-efficient solution for semantic understanding.

---

