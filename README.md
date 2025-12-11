# 今日论文推荐 - 2025-12-11

共 59 篇论文

---

## 1. HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2512.09928v1](http://arxiv.org/abs/2512.09928v1)

**作者:** Minghui Lin, Pengxiang Ding, Shu Wang, Zifeng Zhuang, Yang Liu, Xinyang Tong, Wenxuan Song, Shangke Lyu, Siteng Huang, Donglin Wang

**发布时间:** 2025-12-10

**备注:** Project page: https://hifvla.github.io Github: https://github.com/OpenHelix-Team/HiF-VLA

### GPT解析

### 总结

该研究提出了HiF-VLA框架，通过利用运动表示进行双向时间推理，解决了VLA模型中的时间近视问题，提高了长视野任务中的连贯性，并在多个基准测试和实际机器人操作任务中表现出色。

### 背景

Vision-Language-Action (VLA)模型最近通过将视觉和语言线索转化为动作，使机器人操作成为可能。然而，大多数VLA模型假设马尔可夫性质，仅依赖当前观察，因此受到时间近视的影响，导致长视野连贯性下降。

### 目的

解决VLA模型中的时间近视问题，提高长视野任务中的连贯性，通过利用运动表示进行双向时间推理。

### 方法

提出HiF-VLA（Hindsight, Insight, and Foresight for VLAs）统一框架，该框架利用运动进行双向时间推理。HiF-VLA通过后验先验编码过去动态，通过前瞻推理预测未来运动，并通过后验调制的联合专家整合两者，实现'边思考边行动'的长视野操作范式。

### 主要发现

HiF-VLA在LIBERO-Long和CALVIN ABC-D基准测试上超越了强大的基线模型，同时几乎没有增加推理延迟。此外，HiF-VLA在实际的长视野机器人操作任务中取得了显著改进，证明了其在实际机器人环境中的广泛有效性。

### 结论

HiF-VLA框架通过利用运动表示进行双向时间推理，有效解决了VLA模型中的时间近视问题，提高了长视野任务中的连贯性，并在多个基准测试和实际应用中表现出色。

### 翻译

视觉语言动作模型最近通过将视觉和语言线索转化为动作，使机器人操作成为可能。然而，大多数VLA模型假设马尔可夫性质，仅依赖当前观察，因此受到时间近视的影响，导致长视野连贯性下降。在本研究中，我们将运动视为更紧凑和信息丰富的时间上下文和世界动态表示，它捕获状态间变化同时过滤静态像素级噪声。基于这一想法，我们提出了HiF-VLA（用于VLA的后验、洞察和前瞻），一个利用运动进行双向时间推理的统一框架。HiF-VLA通过后验先验编码过去动态，通过前瞻推理预测未来运动，并通过后验调制的联合专家整合两者，实现长视野操作的'边思考边行动'范式。因此，HiF-VLA在LIBERO-Long和CALVIN ABC-D基准测试上超越了强大的基线模型，同时几乎没有增加推理延迟。此外，HiF-VLA在实际的长视野机器人操作任务中取得了显著改进，证明了其在实际机器人环境中的广泛有效性。


### 论文摘要

Vision-Language-Action (VLA) models have recently enabled robotic manipulation by grounding visual and linguistic cues into actions. However, most VLAs assume the Markov property, relying only on the current observation and thus suffering from temporal myopia that degrades long-horizon coherence. In this work, we view motion as a more compact and informative representation of temporal context and world dynamics, capturing inter-state changes while filtering static pixel-level noise. Building on this idea, we propose HiF-VLA (Hindsight, Insight, and Foresight for VLAs), a unified framework that leverages motion for bidirectional temporal reasoning. HiF-VLA encodes past dynamics through hindsight priors, anticipates future motion via foresight reasoning, and integrates both through a hindsight-modulated joint expert to enable a ''think-while-acting'' paradigm for long-horizon manipulation. As a result, HiF-VLA surpasses strong baselines on LIBERO-Long and CALVIN ABC-D benchmarks, while incurring negligible additional inference latency. Furthermore, HiF-VLA achieves substantial improvements in real-world long-horizon manipulation tasks, demonstrating its broad effectiveness in practical robotic settings.

---

## 2. ChronusOmni: Improving Time Awareness of Omni Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.09841v1](http://arxiv.org/abs/2512.09841v1)

**作者:** Yijing Chen, Yihan Wu, Kaisi Guan, Yuchen Ren, Yuyue Wang, Ruihua Song, Liyun Ru

**发布时间:** 2025-12-10

**备注:** Code available at https://github.com/YJCX330/Chronus/

### GPT解析

### 总结

本研究提出了ChronusOmni模型，一个增强时间感知能力的全方位大语言模型，特别针对显性和隐性的视听时间定位任务，通过结合文本时间戳标记与视听表示，并使用强化学习加强时间推理，在时间定位任务上取得了显著性能提升。

### 背景

时间感知是全方位大语言模型的基本能力，对理解长视频和回答复杂问题至关重要。先前方法主要针对视觉语言场景，专注于明确的时间定位问题，但往往未能充分利用音频模态，忽略了跨模态的隐式时间定位，尽管这类关系在现实场景中很普遍。

### 目的

设计一个能够增强显性和隐性的视听时间定位的时间感知能力的全方位大语言模型。

### 方法

1) 提出ChronusOmni模型，在每时间单元将基于文本的时间戳标记与视觉和音频表示交错，实现跨模态的统一时间建模；2) 结合强化学习与专门设计的奖励函数，强制正确的时间顺序并加强细粒度的时间推理；3) 构建ChronusAV数据集，一个时间准确、模态完整且跨模态对齐的数据集，用于支持视听时间定位任务的训练和评估。

### 主要发现

实验结果表明，ChronusOmni在ChronusAV数据集上取得了最先进的性能，提高了30%以上，并在大多数其他时间定位基准指标上取得了最佳结果，突显了模型在跨模态方面的强时间感知能力。

### 结论

ChronusOmni模型通过有效结合文本时间戳标记与视听表示，并利用强化学习加强时间推理，成功增强了全方位大语言模型的时间感知能力，特别是在视听时间定位任务上取得了显著进展。

### 翻译

时间感知是全方位大语言模型的基本能力，特别是对于理解长视频和回答复杂问题至关重要。先前的方法主要针对视觉语言场景，并专注于明确的时间定位问题，如识别视觉事件发生的时间或确定特定时间发生的事件。然而，这些方法往往未能充分利用音频模态，并且忽略了跨模态的隐式时间定位，例如识别角色说话时的视觉内容，或确定视觉事件发生时所说的话，尽管这些跨模态时间关系在现实场景中很普遍。在本文中，我们提出了ChronusOmni，一个旨在增强显性和隐性的视听时间定位的时间感知能力的全方位大语言模型。首先，我们在每时间单元将基于文本的时间戳标记与视觉和音频表示交错，实现跨模态的统一时间建模。其次，为了强制正确的时间顺序并加强细粒度的时间推理，我们结合了强化学习与专门设计的奖励函数。此外，我们构建了ChronusAV，一个时间准确、模态完整且跨模态对齐的数据集，用于支持视听时间定位任务的训练和评估。实验结果表明，ChronusOmni在ChronusAV上取得了最先进的性能，提高了30%以上，并在大多数其他时间定位基准指标上取得了最佳结果。这突显了我们的模型在跨模态方面的强时间感知能力，同时保留了通用的视频和音频理解能力。


### 论文摘要

Time awareness is a fundamental ability of omni large language models, especially for understanding long videos and answering complex questions. Previous approaches mainly target vision-language scenarios and focus on the explicit temporal grounding questions, such as identifying when a visual event occurs or determining what event happens at aspecific time. However, they often make insufficient use of the audio modality, and overlook implicit temporal grounding across modalities--for example, identifying what is visually present when a character speaks, or determining what is said when a visual event occurs--despite such cross-modal temporal relations being prevalent in real-world scenarios. In this paper, we propose ChronusOmni, an omni large language model designed to enhance temporal awareness for both explicit and implicit audiovisual temporal grounding. First, we interleave text-based timestamp tokens with visual and audio representations at each time unit, enabling unified temporal modeling across modalities. Second, to enforce correct temporal ordering and strengthen fine-grained temporal reasoning, we incorporate reinforcement learning with specially designed reward functions. Moreover, we construct ChronusAV, a temporally-accurate, modality-complete, and cross-modal-aligned dataset to support the training and evaluation on audiovisual temporal grounding task. Experimental results demonstrate that ChronusOmni achieves state-of-the-art performance on ChronusAV with more than 30% improvement and top results on most metrics upon other temporal grounding benchmarks. This highlights the strong temporal awareness of our model across modalities, while preserving general video and audio understanding capabilities.

---

## 3. Composing Concepts from Images and Videos via Concept-prompt Binding

**论文链接:** [http://arxiv.org/abs/2512.09824v1](http://arxiv.org/abs/2512.09824v1)

**作者:** Xianghao Kong, Zeyu Zhang, Yuwei Guo, Zhuoran Zhao, Songchun Zhang, Anyi Rao

**发布时间:** 2025-12-10

**备注:** Project page: https://refkxh.github.io/BiCo_Webpage/

### GPT解析

### 总结

本文提出了Bind & Compose方法，通过视觉概念与提示令牌绑定，实现灵活的视觉概念组合，解决了现有方法在复杂概念提取和图像视频概念组合方面的不足。

### 背景

视觉概念组合旨在将图像和视频中的不同元素整合到单一、连贯的视觉输出中，但当前方法在准确提取复杂视觉概念和灵活组合图像与视频概念方面仍有不足。

### 目的

开发一个一次性方法，通过将视觉概念与相应的提示令牌绑定，并使用来自各种源的绑定令牌组合目标提示，实现灵活的视觉概念组合。

### 方法

采用分层绑定器结构进行交叉注意力条件，将视觉概念编码为提示令牌；设计多样化-吸收机制，使用额外吸收令牌消除概念无关细节的影响；提出时间解耦策略，将视频概念训练解耦为两个阶段，使用双分支绑定器结构进行时间建模。

### 主要发现

评估结果表明，Bind & Compose在概念一致性、提示保真度和运动质量方面优于现有方法，为视觉创意开辟了新的可能性。

### 结论

Bind & Compose方法有效解决了视觉概念组合中的挑战，实现了更准确的概念提取和更灵活的概念组合。

### 翻译

视觉概念组合旨在将图像和视频中的不同元素整合到单一、连贯的视觉输出中，但在准确提取复杂视觉概念和灵活组合图像与视频概念方面仍存在不足。我们引入了Bind & Compose，这是一种一次性方法，通过将视觉概念与相应的提示令牌绑定，并使用来自各种源的绑定令牌组合目标提示，从而实现灵活的视觉概念组合。它采用分层绑定器结构，在扩散模型中进行交叉注意力条件处理，将视觉概念编码为相应的提示令牌，以准确分解复杂视觉概念。为提高概念-令牌绑定准确性，我们设计了多样化-吸收机制，使用额外的吸收令牌在多样化提示训练时消除概念无关细节的影响。为增强图像和视频概念之间的兼容性，我们提出了时间解耦策略，将视频概念的训练过程解耦为两个阶段，并使用双分支绑定器结构进行时间建模。评估表明，我们的方法在概念一致性、提示保真度和运动质量方面优于现有方法，为视觉创意开辟了新的可能性。


### 论文摘要

Visual concept composition, which aims to integrate different elements from images and videos into a single, coherent visual output, still falls short in accurately extracting complex concepts from visual inputs and flexibly combining concepts from both images and videos. We introduce Bind & Compose, a one-shot method that enables flexible visual concept composition by binding visual concepts with corresponding prompt tokens and composing the target prompt with bound tokens from various sources. It adopts a hierarchical binder structure for cross-attention conditioning in Diffusion Transformers to encode visual concepts into corresponding prompt tokens for accurate decomposition of complex visual concepts. To improve concept-token binding accuracy, we design a Diversify-and-Absorb Mechanism that uses an extra absorbent token to eliminate the impact of concept-irrelevant details when training with diversified prompts. To enhance the compatibility between image and video concepts, we present a Temporal Disentanglement Strategy that decouples the training process of video concepts into two stages with a dual-branch binder structure for temporal modeling. Evaluations demonstrate that our method achieves superior concept consistency, prompt fidelity, and motion quality over existing approaches, opening up new possibilities for visual creativity.

---

## 4. Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder

**论文链接:** [http://arxiv.org/abs/2512.09626v1](http://arxiv.org/abs/2512.09626v1)

**作者:** Yousef Azizi Movahed, Fatemeh Ziaeetabar

**发布时间:** 2025-12-10

**备注:** Code available at: https://github.com/YousefAMovahed/beyond-sequences-hoi-benchmark

### GPT解析

### 总结

研究通过结构化数据处理流程和轻量级架构实现了手部-物体交互状态的精确分类，最终准确率达到97.60%

### 背景

可靠预测手部-物体交互中的人类意图是计算机视觉领域的开放挑战

### 目的

研究基础子问题：精细分类原子交互状态，即'接近'、'抓取'和'持有'

### 方法

引入结构化数据处理流程将MANIAC数据集原始视频转换为27,476个统计运动特征向量；比较静态分类器(MLPs)与时间模型(RNNs)

### 主要发现

当设置双向RNN序列长度为1时，网络转变为高容量静态特征编码器，显著提高准确性；优化模型成功克服最具挑战性的'抓取'类别，实现0.90的平衡F1分数

### 结论

为使用结构化、可解释特征和轻量级架构的低级手部-物体交互识别提供了新基准

### 翻译

Reliably predicting human intent in hand-object interactions is an open challenge for computer vision. Our research concentrates on a fundamental sub-problem: the fine-grained classification of atomic interaction states, namely 'approaching', 'grabbing', and 'holding'. To this end, we introduce a structured data engineering process that converts raw videos from the MANIAC dataset into 27,476 statistical-kinematic feature vectors. Each vector encapsulates relational and dynamic properties from a short temporal window of motion. Our initial hypothesis posited that sequential modeling would be critical, leading us to compare static classifiers (MLPs) against temporal models (RNNs). Counter-intuitively, the key discovery occurred when we set the sequence length of a Bidirectional RNN to one (seq_length=1). This modification converted the network's function, compelling it to act as a high-capacity static feature encoder. This architectural change directly led to a significant accuracy improvement, culminating in a final score of 97.60%. Of particular note, our optimized model successfully overcame the most challenging transitional class, 'grabbing', by achieving a balanced F1-score of 0.90. These findings provide a new benchmark for low-level hand-object interaction recognition using structured, interpretable features and lightweight architectures.


### 论文摘要

Reliably predicting human intent in hand-object interactions is an open challenge for computer vision. Our research concentrates on a fundamental sub-problem: the fine-grained classification of atomic interaction states, namely 'approaching', 'grabbing', and 'holding'. To this end, we introduce a structured data engineering process that converts raw videos from the MANIAC dataset into 27,476 statistical-kinematic feature vectors. Each vector encapsulates relational and dynamic properties from a short temporal window of motion. Our initial hypothesis posited that sequential modeling would be critical, leading us to compare static classifiers (MLPs) against temporal models (RNNs). Counter-intuitively, the key discovery occurred when we set the sequence length of a Bidirectional RNN to one (seq_length=1). This modification converted the network's function, compelling it to act as a high-capacity static feature encoder. This architectural change directly led to a significant accuracy improvement, culminating in a final score of 97.60%. Of particular note, our optimized model successfully overcame the most challenging transitional class, 'grabbing', by achieving a balanced F1-score of 0.90. These findings provide a new benchmark for low-level hand-object interaction recognition using structured, interpretable features and lightweight architectures.

---

## 5. Video-QTR: Query-Driven Temporal Reasoning Framework for Lightweight Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.09354v1](http://arxiv.org/abs/2512.09354v1)

**作者:** Xinkui Zhao, Zuxin Wang, Yifan Zhang, Guanjie Cheng, Yueshen Xu, Shuiguang Deng, Chang Liu, Naibo Wang, Jianwei Yin

**发布时间:** 2025-12-10

### GPT解析

### 总结

论文提出了一种名为Video-QTR的轻量级框架，通过查询驱动的动态感知资源分配，解决了多模态大语言模型在长视频理解中的计算效率问题，显著减少了输入帧消耗并保持了最先进的性能。

### 背景

多模态大语言模型(MLLMs)的快速发展扩展了视觉语言推理能力，但将其应用于长视频理解仍面临计算密集的挑战。密集帧编码产生过多视觉标记，导致高内存消耗、冗余计算和可扩展性有限。

### 目的

解决多模态大语言模型在长视频理解中的计算效率问题，减少密集帧编码带来的资源消耗，提高模型在实际应用中的可扩展性。

### 方法

提出Video-QTR(Query-Driven Temporal Reasoning)框架，将视频理解重新定义为查询引导的推理过程。根据查询的语义意图动态分配感知资源，在推理和感知之间创建自适应反馈循环，而非编码每一帧。

### 主要发现

在五个基准测试(MSVD-QA, Activity Net-QA, Movie Chat, 和 Video MME)上的实验表明，Video-QTR实现了最先进的性能，同时减少了高达73%的输入帧消耗。

### 结论

查询驱动的时间推理为视频理解提供了高效且可扩展的解决方案，通过动态分配感知资源而非处理每一帧，显著提高了计算效率。

### 翻译

多模态大语言模型(MLLMs)的快速发展显著扩展了视觉语言推理的范围，使统一系统能够解释和描述复杂的视觉内容。然而，将这些模型应用于长视频理解仍然计算密集。密集帧编码会产生过多的视觉标记，导致高内存消耗、冗余计算和实际应用中的可扩展性有限。这种低效性突显了传统'先处理再推理'范式的关键局限性，该范式在语义推理之前会详尽地分析视觉流。为应对这一挑战，我们引入了Video-QTR(基于查询的时间推理)，这是一个轻量级框架，将视频理解重新定义为查询引导的推理过程。与编码每一帧不同，Video-QTR根据查询的语义意图动态分配感知资源，在推理和感知之间创建自适应反馈循环。在五个基准测试(MSVD-QA, Activity Net-QA, Movie Chat, 和 Video MME)上的广泛实验表明，Video-QTR在减少高达73%的输入帧消耗的同时，实现了最先进的性能。这些结果证实了查询驱动的时间推理为视频理解提供了高效且可扩展的解决方案。


### 论文摘要

The rapid development of multimodal large-language models (MLLMs) has significantly expanded the scope of visual language reasoning, enabling unified systems to interpret and describe complex visual content. However, applying these models to long-video understanding remains computationally intensive. Dense frame encoding generates excessive visual tokens, leading to high memory consumption, redundant computation, and limited scalability in real-world applications. This inefficiency highlights a key limitation of the traditional process-then-reason paradigm, which analyzes visual streams exhaustively before semantic reasoning. To address this challenge, we introduce Video-QTR (Query-Driven Temporal Reasoning), a lightweight framework that redefines video comprehension as a query-guided reasoning process. Instead of encoding every frame, Video-QTR dynamically allocates perceptual resources based on the semantic intent of the query, creating an adaptive feedback loop between reasoning and perception. Extensive experiments across five benchmarks: MSVD-QA, Activity Net-QA, Movie Chat, and Video MME demonstrate that Video-QTR achieves state-of-the-art performance while reducing input frame consumption by up to 73%. These results confirm that query-driven temporal reasoning provides an efficient and scalable solution for video understanding.

---

## 6. What Happens When: Learning Temporal Orders of Events in Videos

**论文链接:** [http://arxiv.org/abs/2512.08979v1](http://arxiv.org/abs/2512.08979v1)

**作者:** Daechul Ahn, Yura Choi, Hyeonbeom Choi, Seongwon Cho, San Kim, Jonghyun Choi

**发布时间:** 2025-12-05

**备注:** WACV 2026

### GPT解析

### 总结

该研究探讨了视频大大多模态模型(VLMMs)在捕捉事件时间顺序方面的局限性，提出了新的基准测试VECTOR和改进方法MECOT，以增强模型的时间理解能力。

### 背景

视频大大多模态模型(VLMMs)在视频理解方面已展现出令人印象深刻的性能，但它们准确捕捉多个事件时间顺序的能力仍然有待探索。

### 目的

为了评估VLMMs的时间理解能力，提出了VECTOR基准测试，旨在明确评估模型识别事件顺序的能力。

### 方法

提出了MECOT方法，包括：(1)基于详细的逐个事件视频描述训练模型，(2)在推理时使用思维链提示来增强时间意识。

### 主要发现

通过实验发现，即使视频帧被打乱，VLMMs在现有基准测试上仍表现良好，表明它们可能依赖典型场景的先验知识而非准确的事件顺序；在VECTOR基准测试上，各种VLMMs经常无法理解事件的顺序。

### 结论

MECOT在VECTOR上优于之前的方法，并提高了现有视频基准测试的性能，表明时间理解的有效性。

### 翻译

视频大大多模态模型(VLMMs)在视频理解方面已展现出令人印象深刻的性能，但它们准确捕捉多个事件时间顺序的能力仍然有待探索。我们有趣地观察到，通过全面的实验，即使视频帧被打乱，模型在现有基准测试上仍表现良好。这表明VLMMs不一定必须依赖视觉事件的准确顺序处理，而是可能依赖对典型场景的先验知识来回答问题。为了评估VLMMs的时间理解能力，我们提出了VECTOR基准测试，旨在明确评估模型识别事件顺序的能力。在这个基准测试上，我们观察到各种VLMMs通常无法理解事件的顺序。为此，我们提出了MECOT(基于思维链的多事件指令微调)，它(1)基于详细的逐个事件视频描述训练模型，(2)在推理时使用思维链提示来增强时间意识。MECOT在VECTOR上优于之前的方法，同时提高了现有视频基准测试的性能，表明时间理解的有效性。我们发布了代码、模型和数据集。


### 论文摘要

Video Large Multimodal Models (VLMMs) have shown impressive performance in video understanding, yet their ability to accurately capture the temporal order of multiple events remains underexplored. We interestingly observe that, even when video frames are scrambled, models perform very well on the existing benchmarks by comprehensive experiments. This implies that VLMMs may not necessarily rely on accurate sequential processing of visual events, but instead depend on prior knowledge of typical scenarios to answer the question. To benchmark temporal understanding capabilities in VLMMs, we propose VECTOR, designed to explicitly assess a model's ability to identify the temporal order of events. On this benchmark, we observe that various VLMMs often fail to understand the orders of events. To address this, we propose MECOT (Multi-Event instruction fine-tuning with Chain-of-Thought), which (1) trains models on detailed, event-by-event video descriptions and (2) using chain-of-thought prompts at inference to enhance temporal awareness. MECOT outperforms prior arts on VECTOR as well as improving performance on existing video benchmarks, implying effectiveness of temporal understanding. We release our code, model and datasets.

---

## 7. Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for Sleep Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.09591v1](http://arxiv.org/abs/2512.09591v1)

**作者:** Magnus Ruud Kjaer, Rahul Thapa, Gauri Ganjoo, Hyatt Moore, Poul Joergen Jennum, Brandon M. Westover, James Zou, Emmanuel Mignot, Bryan He, Andreas Brink-Kjaer

**发布时间:** 2025-12-10

### GPT解析

### 总结

该研究介绍了Stanford Sleep Bench，一个大型的多导睡眠图数据集，用于评估自监督表示学习在睡眠分析基础模型中的应用，并解决了睡眠基础模型发展中的两个关键限制。

### 背景

多导睡眠图(PSG)作为睡眠分析的金标准测试，产生大量多模态临床数据，为利用自监督表示学习(SSRL)预训练基础模型增强睡眠分析提供了机会。

### 目的

解决睡眠基础模型发展的两个关键限制：缺乏共享数据集和基准，以及缺乏对SSRL方法在睡眠相关任务上的系统评估。

### 方法

创建Stanford Sleep Bench数据集，包含17,467条记录(总计163,000+小时)，涵盖13种临床疾病预测任务和标准睡眠相关任务，系统评估多种SSRL预训练方法在四个下游任务上的表现。

### 主要发现

多种预训练方法在睡眠分期、呼吸暂停诊断和年龄估计方面表现相当；但对于死亡率和疾病预测，对比学习方法显著优于其他方法且收敛更快。

### 结论

Stanford Sleep Bench及其配套资源(预训练模型权重、训练管道和评估代码)的发布将促进睡眠研究的可重复性和发展。

### 翻译

多导睡眠图(PSG)是睡眠分析的金标准测试，产生大量多模态临床数据，为利用自监督表示学习(SSRL)预训练基础模型增强睡眠分析提供了机会。然而，睡眠基础模型的进展受两个关键限制阻碍：(1)缺乏包含多样化任务的共享数据集和基准用于训练和评估，(2)缺乏对SSRL方法在睡眠相关任务上的系统评估。为解决这些差距，我们引入Stanford Sleep Bench，这是一个大规模PSG数据集，包含来自一家主要睡眠诊所的17,467条记录，总计超过163,000小时，包括13种临床疾病预测任务以及标准的睡眠相关任务，如睡眠分期、呼吸暂停诊断和年龄估计。我们在Stanford Sleep Bench上系统评估了SSRL预训练方法，评估了四个下游任务的表现：睡眠分期、呼吸暂停诊断、年龄估计以及疾病和死亡率预测。我们的结果显示，多种预训练方法在睡眠分期、呼吸暂停诊断和年龄估计方面取得了可比的性能。然而，对于死亡率和疾病预测，对比学习方法显著优于其他方法，同时在预训练过程中收敛更快。为了促进可重复性和推进睡眠研究，我们将发布Stanford Sleep Bench以及预训练模型权重、训练管道和评估代码。


### 论文摘要

Polysomnography (PSG), the gold standard test for sleep analysis, generates vast amounts of multimodal clinical data, presenting an opportunity to leverage self-supervised representation learning (SSRL) for pre-training foundation models to enhance sleep analysis. However, progress in sleep foundation models is hindered by two key limitations: (1) the lack of a shared dataset and benchmark with diverse tasks for training and evaluation, and (2) the absence of a systematic evaluation of SSRL approaches across sleep-related tasks. To address these gaps, we introduce Stanford Sleep Bench, a large-scale PSG dataset comprising 17,467 recordings totaling over 163,000 hours from a major sleep clinic, including 13 clinical disease prediction tasks alongside canonical sleep-related tasks such as sleep staging, apnea diagnosis, and age estimation. We systematically evaluate SSRL pre-training methods on Stanford Sleep Bench, assessing downstream performance across four tasks: sleep staging, apnea diagnosis, age estimation, and disease and mortality prediction. Our results show that multiple pretraining methods achieve comparable performance for sleep staging, apnea diagnosis, and age estimation. However, for mortality and disease prediction, contrastive learning significantly outperforms other approaches while also converging faster during pretraining. To facilitate reproducibility and advance sleep research, we will release Stanford Sleep Bench along with pretrained model weights, training pipelines, and evaluation code.

---

## 8. Transport Novelty Distance: A Distributional Metric for Evaluating Material Generative Models

**论文链接:** [http://arxiv.org/abs/2512.09514v1](http://arxiv.org/abs/2512.09514v1)

**作者:** Paul Hagemann, Simon Müller, Janine George, Philipp Benner

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文引入了一种名为传输新颖性距离（TNovD）的新型评估指标，用于同时评估材料生成模型的质量和新颖性。该指标基于最优传输理论和图神经网络，通过对比学习训练，能够有效检测记忆化和低质量材料数据，并且具有领域通用性，可扩展到图像和分子等其他领域。

### 背景

生成式机器学习的最新进展为新材料的设计和发现开辟了新可能性。随着模型变得越来越复杂，对严格且有意义的评估指标的需求也在增加。现有的评估方法通常无法捕捉生成结构的质量和新颖性，限制了评估真实生成性能的能力。

### 目的

引入传输新颖性距离（TNovD）来评估用于材料发现的生成模型，同时考虑生成材料的质量和新颖性。

### 方法

基于最优传输理论的思想，TNovD使用训练集和生成集特征之间的耦合，通过阈值细化为质量和记忆模式。特征通过图神经网络从晶体结构生成，该网络通过对比学习进行训练，以区分材料、它们的增强版本和不同大小的超胞。在典型的玩具实验（包括记忆、噪声注入和晶格变形）以及MP20验证集和WBM替换数据集上评估了该指标。

### 主要发现

TNovD能够检测出记忆化和低质量的材料数据，并对几种流行的材料生成模型进行了性能基准测试，证明了其有效性。

### 结论

TNovD框架最初是为材料领域设计的，但它与领域无关，可以适应其他领域，如图像和分子，具有广泛的应用前景。

### 翻译

生成机器学习的最新进展为新材料的设计和发现开辟了新的可能性。然而，随着这些模型变得越来越复杂，对严格且有意义的评估指标的需求也在增长。现有的评估方法通常无法捕捉生成结构的质量和新颖性，限制了评估真实生成性能的能力。在本文中，我们引入了传输新颖性距离（TNovD）来共同评估用于材料发现的生成模型，基于生成材料的质量和新颖性。基于最优传输理论的思想，TNovD使用训练集和生成集特征之间的耦合，通过阈值细化为质量和记忆模式。特征是通过图神经网络从晶体结构生成的，该网络通过对比学习进行训练，以区分材料、它们的增强版本和不同大小的超胞。我们在典型的与晶体结构预测相关的玩具实验中评估了我们提出的指标，包括记忆、噪声注入和晶格变形。此外，我们在MP20验证集和WBM替换数据集上验证了TNovD，证明它能够检测出记忆化和低质量的材料数据。我们还对几种流行的材料生成模型进行了性能基准测试。虽然最初是为材料领域引入的，但我们的TNovD框架与领域无关，可以适应其他领域，如图像和分子。


### 论文摘要

Recent advances in generative machine learning have opened new possibilities for the discovery and design of novel materials. However, as these models become more sophisticated, the need for rigorous and meaningful evaluation metrics has grown. Existing evaluation approaches often fail to capture both the quality and novelty of generated structures, limiting our ability to assess true generative performance. In this paper, we introduce the Transport Novelty Distance (TNovD) to judge generative models used for materials discovery jointly by the quality and novelty of the generated materials. Based on ideas from Optimal Transport theory, TNovD uses a coupling between the features of the training and generated sets, which is refined into a quality and memorization regime by a threshold. The features are generated from crystal structures using a graph neural network that is trained to distinguish between materials, their augmented counterparts, and differently sized supercells using contrastive learning. We evaluate our proposed metric on typical toy experiments relevant for crystal structure prediction, including memorization, noise injection and lattice deformations. Additionally, we validate the TNovD on the MP20 validation set and the WBM substitution dataset, demonstrating that it is capable of detecting both memorized and low-quality material data. We also benchmark the performance of several popular material generative models. While introduced for materials, our TNovD framework is domain-agnostic and can be adapted for other areas, such as images and molecules.

---

## 9. DMP-TTS: Disentangled multi-modal Prompting for Controllable Text-to-Speech with Chained Guidance

**论文链接:** [http://arxiv.org/abs/2512.09504v1](http://arxiv.org/abs/2512.09504v1)

**作者:** Kang Yin, Chunyu Qiang, Sirui Zhao, Xiaopeng Wang, Yuzhe Liang, Pengfei Cai, Tong Xu, Chen Zhang, Enhong Chen

**发布时间:** 2025-12-10

### GPT解析

### 总结

DMP-TTS是一种基于潜在扩散变换器的可控文本转语音系统，实现了说话人音色和说话风格的独立控制。

### 背景

可控文本转语音(TTS)系统在独立控制说话人音色和说话风格方面面临重大挑战，这些属性之间常常存在纠缠问题。

### 目的

开发一个能够独立操纵说话人音色和说话风格的TTS系统，解决属性间的纠缠问题。

### 方法

提出DMP-TTS框架，包含基于CLAP的风格编码器对齐音频和文本提示，链式无分类器指导实现细粒度控制，以及表示对齐方法稳定训练并加速收敛。

### 主要发现

DMP-TTS比开源基线系统提供更强的风格可控性，同时保持竞争性的语音可懂度和自然度。

### 结论

DMP-TTS通过明确解耦和多模态提示技术，成功实现了说话人音色和说话风格的独立控制，提升了TTS系统的可控性。

### 翻译

可控文本转语音(TTS)系统在实现说话人音色和说话风格的独立控制方面面临重大挑战，这些属性之间常常存在纠缠。我们提出了DMP-TTS，一个具有明确解耦和多模态提示的潜在扩散变换器(DiT)框架。基于CLAP的风格编码器(Style-CLAP)通过对比学习和多任务监督，将参考音频和描述性文本的提示对齐到共享空间。为了在推理过程中实现细粒度控制，我们引入了链式无分类器指导(cCFG)，通过分层条件dropout训练，能够独立调整内容、音色和风格指导强度。此外，我们采用表示对齐(REPA)方法，将预训练Whisper模型的声学-语义特征蒸馏到中间DiT表示中，稳定训练并加速收敛。实验表明，DMP-TTS比开源基线系统提供更强的风格可控性，同时保持竞争性的可懂度和自然度。代码和演示将在https://y61329697.github.io/DMP-TTS/提供。


### 论文摘要

Controllable text-to-speech (TTS) systems face significant challenges in achieving independent manipulation of speaker timbre and speaking style, often suffering from entanglement between these attributes. We present DMP-TTS, a latent Diffusion Transformer (DiT) framework with explicit disentanglement and multi-modal prompting. A CLAP-based style encoder (Style-CLAP) aligns cues from reference audio and descriptive text in a shared space and is trained with contrastive learning plus multi-task supervision on style attributes. For fine-grained control during inference, we introduce chained classifier-free guidance (cCFG) trained with hierarchical condition dropout, enabling independent adjustment of content, timbre, and style guidance strengths. Additionally, we employ Representation Alignment (REPA) to distill acoustic-semantic features from a pretrained Whisper model into intermediate DiT representations, stabilizing training and accelerating convergence. Experiments show that DMP-TTS delivers stronger style controllability than open-source baselines while maintaining competitive intelligibility and naturalness. Code and demos will be available at https://y61329697.github.io/DMP-TTS/.

---

## 10. StateSpace-SSL: Linear-Time Self-supervised Learning for Plant Disease Detectio

**论文链接:** [http://arxiv.org/abs/2512.09492v1](http://arxiv.org/abs/2512.09492v1)

**作者:** Abdullah Al Mamun, Miaohua Zhang, David Ahmedt-Aristizabal, Zeeshan Hayder, Mohammad Awrangjeb

**发布时间:** 2025-12-10

**备注:** Accepted to AAAI workshop (AgriAI 2026)

### GPT解析

### 总结

StateSpace-SSL是一种创新的线性时间自监督学习框架，专为植物疾病检测设计，使用视觉Mamba状态空间编码器通过方向扫描有效捕捉叶面病变的连续性模式，显著优于传统的CNN和变换器方法。

### 背景

自监督学习在植物疾病检测中具有吸引力，可利用大量未标记的叶片图像，但现有SSL方法基于CNN或视觉变换器，与农业图像匹配度不高。CNN难以捕捉沿叶结构连续发展的疾病模式，而变换器引入高分辨率补丁的二次注意力成本。

### 目的

解决现有SSL方法的局限性，提出更适合农业图像的SSL框架，有效捕捉植物叶片上的疾病模式。

### 方法

提出StateSpace-SSL框架，采用视觉Mamba状态空间编码器，通过在叶表面方向扫描建模长范围病变连续性，使用原型驱动的教师-学生目标对齐多视图表示，促进稳定和病变感知的特征学习。

### 主要发现

在三个公开植物疾病数据集上的实验表明，StateSpace-SSL在各种评估指标上一致优于基于CNN和变换器的SSL基线，定性分析确认其学习到紧凑、病变聚焦的特征图。

### 结论

线性状态空间建模在自监督植物疾病表征学习中具有明显优势，StateSpace-SSL能有效解决农业图像中疾病模式捕捉的挑战。

### 翻译

自监督学习（SSL）在植物疾病检测中具有吸引力，因为它可以利用大量未标记的叶片图像，然而大多数现有的SSL方法构建在CNN或视觉变换器上，这些方法与农业图像的匹配度不高。基于CNN的SSL难以捕捉沿叶结构连续发展的疾病模式，而基于变换器的SSL引入了来自高分辨率补丁的二次注意力成本。为解决这些局限性，我们提出了StateSpace-SSL，一种线性时间的SSL框架，它采用视觉Mamba状态空间编码器，通过在叶表面方向扫描来建模长范围的病变连续性。原型驱动的教师-学生目标对齐多视图表示，鼓励从标记数据中学习稳定且病变感知的特征。在三个公开可用的植物疾病数据集上的实验表明，StateSpace-SSL在各种评估指标上一致优于基于CNN和变换器的SSL基线。定性分析进一步确认它学习到紧凑的、病变聚焦的特征图，突显了线性状态空间建模在自监督植物疾病表征学习中的优势。


### 论文摘要

Self-supervised learning (SSL) is attractive for plant disease detection as it can exploit large collections of unlabeled leaf images, yet most existing SSL methods are built on CNNs or vision transformers that are poorly matched to agricultural imagery. CNN-based SSL struggles to capture disease patterns that evolve continuously along leaf structures, while transformer-based SSL introduces quadratic attention cost from high-resolution patches. To address these limitations, we propose StateSpace-SSL, a linear-time SSL framework that employs a Vision Mamba state-space encoder to model long-range lesion continuity through directional scanning across the leaf surface. A prototype-driven teacher-student objective aligns representations across multiple views, encouraging stable and lesion-aware features from labelled data. Experiments on three publicly available plant disease datasets show that StateSpace-SSL consistently outperforms the CNN- and transformer-based SSL baselines in various evaluation metrics. Qualitative analyses further confirm that it learns compact, lesion-focused feature maps, highlighting the advantage of linear state-space modelling for self-supervised plant disease representation learning.

---

## 11. Self-Supervised Learning with Gaussian Processes

**论文链接:** [http://arxiv.org/abs/2512.09322v1](http://arxiv.org/abs/2512.09322v1)

**作者:** Yunshan Duan, Sinead Williamson

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了一种名为高斯过程自监督学习(GPSSL)的新方法，利用高斯过程模型进行表示学习，解决了传统自监督学习方法在生成相似样本对和不确定性量化方面的局限性，并在多种数据集上的分类和回归任务中表现出色。

### 背景

自监督学习(SSL)是一种无需标记样本监督即可学习数据潜在结构的机器学习范式。当前大多数SSL方法依赖生成相似观测样本对的能力来确保表示空间的平滑性，但这对许多类型的数据具有挑战性。此外，这些方法缺乏不确定性量化考虑，在样本外预测设置中表现可能不佳。

### 目的

为了解决传统SSL方法的局限性，作者提出了一种新方法GPSSL，旨在提高表示学习的效果，并解决不确定性量化和样本外预测问题。

### 方法

作者提出的高斯过程自监督学习(GPSSL)是一种利用高斯过程(GP)模型进行表示学习的新方法。该方法在高斯过程先验上施加表示约束，并通过最小化鼓励信息丰富表示的损失函数获得广义贝叶斯后验。GP固有的协方差函数自然地将相似单元的表示拉在一起，作为使用明确正样本的替代方案。

### 主要发现

研究表明GPSSL与核主成分分析(kernel PCA)和基于神经网络的流行SSL方法VICReg密切相关，但与这两种方法不同，GPSSL允许后验不确定性可以传播到下游任务。在各种数据集上的分类和回归任务实验表明，GPSSL在准确性、不确定性和误差控制方面均优于传统方法。

### 结论

GPSSL作为一种创新的SSL方法，通过利用高斯过程的特性，成功解决了传统SSL方法在样本对生成和不确定性量化方面的局限性，并在多种任务中表现出优越性能，为自监督学习领域提供了新的思路和解决方案。

### 翻译

自监督学习(SSL)是一种机器学习范式，其中模型无需从标记样本的显式监督即可学习理解数据的基本结构。从SSL获得的表示已被证明对许多下游任务有用，包括聚类和线性分类等。为确保表示空间的平滑性，大多数SSL方法依赖于生成与给定实例相似的观测样本对的能力。然而，对于许多类型的数据，生成这些样本对可能具有挑战性。此外，这些方法缺乏不确定性量化考虑，在样本外预测设置中可能表现不佳。为解决这些局限性，我们提出了高斯过程自监督学习(GPSSL)，一种利用高斯过程(GP)模型进行表示学习的新方法。在表示上施加GP先验，并通过最小化鼓励信息丰富表示的损失函数获得广义贝叶斯后验。GP固有的协方差函数自然地将相似单元的表示拉在一起，作为使用明确正样本的替代方案。我们表明GPSSL与核主成分分析和流行的基于神经网络的SSL方法VICReg密切相关，但与这两种方法不同，GPSSL允许可以传播到下游任务的后验不确定性。在各种数据集上考虑分类和回归任务的实验表明，GPSSL在准确性、不确定性和误差控制方面优于传统方法。


### 论文摘要

Self supervised learning (SSL) is a machine learning paradigm where models learn to understand the underlying structure of data without explicit supervision from labeled samples. The acquired representations from SSL have demonstrated useful for many downstream tasks including clustering, and linear classification, etc. To ensure smoothness of the representation space, most SSL methods rely on the ability to generate pairs of observations that are similar to a given instance. However, generating these pairs may be challenging for many types of data. Moreover, these methods lack consideration of uncertainty quantification and can perform poorly in out-of-sample prediction settings. To address these limitations, we propose Gaussian process self supervised learning (GPSSL), a novel approach that utilizes Gaussian processes (GP) models on representation learning. GP priors are imposed on the representations, and we obtain a generalized Bayesian posterior minimizing a loss function that encourages informative representations. The covariance function inherent in GPs naturally pulls representations of similar units together, serving as an alternative to using explicitly defined positive samples. We show that GPSSL is closely related to both kernel PCA and VICReg, a popular neural network-based SSL method, but unlike both allows for posterior uncertainties that can be propagated to downstream tasks. Experiments on various datasets, considering classification and regression tasks, demonstrate that GPSSL outperforms traditional methods in terms of accuracy, uncertainty quantification, and error control.

---

## 12. Contrastive Learning for Semi-Supervised Deep Regression with Generalized Ordinal Rankings from Spectral Seriation

**论文链接:** [http://arxiv.org/abs/2512.09267v1](http://arxiv.org/abs/2512.09267v1)

**作者:** Ce Wang, Weihang Dai, Hanru Bai, Xiaomeng Li

**发布时间:** 2025-12-10

### GPT解析

### 总结

本研究提出了一种扩展的对比回归方法，能够在半监督设置中有效利用未标记数据，通过构建特征相似度矩阵并应用谱排序算法来恢复样本的序数关系，从而减少对标注数据的依赖，提升回归模型的表示能力和性能。

### 背景

对比学习方法通过在特征空间中强制执行标签距离关系来提高回归模型的表示能力，但这些方法高度依赖标签信息来正确恢复特征的序数关系，限制了它们在半监督回归中的应用。

### 目的

扩展对比回归方法，使其能够在半监督设置中使用未标记数据，减少对昂贵标注数据的依赖，同时保持或提升模型的性能。

### 方法

在小批量中同时使用标记和未标记样本构建特征相似度矩阵；通过谱排序算法恢复未标记样本的序数排序；利用标记样本提供正则化，使排序更可靠；使用动态规划算法选择稳健特征构建矩阵；将恢复的序数关系用于未标记样本的对比学习和预测监督。

### 主要发现

所提方法在各种数据集上提供了理论保证和经验验证；能够超越现有的最先进半监督深度回归方法；允许更多数据用于特征表示学习，从而获得更稳健的结果。

### 结论

该方法成功扩展了对比回归到半监督场景，有效利用未标记数据，减少标注依赖，提升了回归性能，相关代码已在GitHub平台发布。

### 翻译

对比学习方法强制特征空间中的标签距离关系以提高回归模型的表示能力。然而，这些方法高度依赖标签信息来正确恢复特征的序数关系，限制了它们在半监督回归中的应用。在本工作中，我们将对比回归方法扩展到允许在半监督设置中使用未标记数据，从而减少对昂贵标注的依赖。特别是，我们在小批量中同时使用标记和未标记样本构建特征相似度矩阵以反映样本间关系，如果误差水平在特定范围内，可以通过谱排序算法恢复未标记样本的准确序数排序。引入标记样本通过真实标签信息的指导提供序数排序的正则化，使排序更可靠。为了减少特征扰动，我们进一步利用动态规划算法为矩阵构建选择稳健特征。恢复的序数关系随后用于未标记样本的对比学习，从而允许更多数据用于特征表示学习，实现更稳健的结果。序数排序也可用于监督未标记样本的预测，作为额外的训练信号。我们在各种数据集上提供理论保证和实验验证，表明我们的方法能够超越现有的最先进半监督深度回归方法。我们的代码已发布在https://github.com/xmed-lab/CLSS。


### 论文摘要

Contrastive learning methods enforce label distance relationships in feature space to improve representation capability for regression models. However, these methods highly depend on label information to correctly recover ordinal relationships of features, limiting their applications to semi-supervised regression. In this work, we extend contrastive regression methods to allow unlabeled data to be used in the semi-supervised setting, thereby reducing the dependence on costly annotations. Particularly we construct the feature similarity matrix with both labeled and unlabeled samples in a mini-batch to reflect inter-sample relationships, and an accurate ordinal ranking of involved unlabeled samples can be recovered through spectral seriation algorithms if the level of error is within certain bounds. The introduction of labeled samples above provides regularization of the ordinal ranking with guidance from the ground-truth label information, making the ranking more reliable. To reduce feature perturbations, we further utilize the dynamic programming algorithm to select robust features for the matrix construction. The recovered ordinal relationship is then used for contrastive learning on unlabeled samples, and we thus allow more data to be used for feature representation learning, thereby achieving more robust results. The ordinal rankings can also be used to supervise predictions on unlabeled samples, serving as an additional training signal. We provide theoretical guarantees and empirical verification through experiments on various datasets, demonstrating that our method can surpass existing state-of-the-art semi-supervised deep regression methods. Our code have been released on https://github.com/xmed-lab/CLSS.

---

## 13. PolyLingua: Margin-based Inter-class Transformer for Robust Cross-domain Language Detection

**论文链接:** [http://arxiv.org/abs/2512.08143v2](http://arxiv.org/abs/2512.08143v2)

**作者:** Ali Lotfi Rezaabad, Bikram Khanal, Shashwat Chaurasia, Lu Zeng, Dezhi Hong, Hossein Bashashati, Thomas Butler, Megan Ganji

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文介绍了PolyLingua，一种基于Transformer的轻量级语言识别模型，用于多语言系统中的语言检测和细粒度语言分类，在保持高准确性的同时适用于计算和延迟受限的环境。

### 背景

语言识别是聊天机器人和虚拟助手等多语言系统中的关键第一步，其错误可能导致后续环节的失败。现有工具存在局限性：开源工具如LangDetect和FastText速度快但准确性低，而大型语言模型虽有效但成本过高，难以满足低延迟或资源有限的需求。

### 目的

开发一种轻量级语言识别模型，能够处理具有挑战性的情况（如音乐请求中歌曲标题与用户语言不同的情况），同时保持高准确性和低资源消耗。

### 方法

PolyLingua采用基于Transformer的轻量级架构，使用两级对比学习框架，结合实例级分离和类级对齐，并应用自适应边界技术，为相关语言生成紧凑且良好分离的嵌入表示。

### 主要发现

在Amazon Massive数据集上达到99.25%的F1分数，在Song数据集上达到98.15%的F1分数，性能超越Sonnet 3.5模型，同时参数量仅为后者的十分之一。

### 结论

PolyLingua是一种高效的语言识别解决方案，在保持高准确性的同时具有轻量级特性，特别适合计算和延迟受限的环境，能有效处理语言识别中的挑战性情况。

### 翻译

语言识别是聊天机器人和虚拟助手等多语言系统中的关键第一步，能够实现语言和文化上准确的用户体验。此阶段的错误可能导致后续环节的失败，因此对准确性要求很高。然而，现有的语言识别工具在某些关键情况下（如歌曲标题与用户语言不同的情况）表现不佳。开源工具如LangDetect和FastText速度快但准确性较低，而大型语言模型虽然有效，但在低延迟或资源有限的环境中往往成本过高。我们引入了PolyLingua，一种基于Transformer的轻量级模型，用于领域内语言检测和细粒度语言分类。它采用两级对比学习框架，结合实例级分离和类级对齐，并使用自适应边界，即使对于密切相关的语言也能产生紧凑且良好分离的嵌入。在两个具有挑战性的数据集——Amazon Massive（多语言数字助手话语）和Song数据集（频繁代码切换的音乐请求）——上评估，PolyLingua分别实现了99.25%和98.15%的F1分数，超越了Sonnet 3.5，同时使用的参数量仅为后者的十分之一，使其成为计算和延迟受限环境的理想选择。


### 论文摘要

Language identification is a crucial first step in multilingual systems such as chatbots and virtual assistants, enabling linguistically and culturally accurate user experiences. Errors at this stage can cascade into downstream failures, setting a high bar for accuracy. Yet, existing language identification tools struggle with key cases -- such as music requests where the song title and user language differ. Open-source tools like LangDetect, FastText are fast but less accurate, while large language models, though effective, are often too costly for low-latency or low-resource settings. We introduce PolyLingua, a lightweight Transformer-based model for in-domain language detection and fine-grained language classification. It employs a two-level contrastive learning framework combining instance-level separation and class-level alignment with adaptive margins, yielding compact and well-separated embeddings even for closely related languages. Evaluated on two challenging datasets -- Amazon Massive (multilingual digital assistant utterances) and a Song dataset (music requests with frequent code-switching) -- PolyLingua achieves 99.25% F1 and 98.15% F1, respectively, surpassing Sonnet 3.5 while using 10x fewer parameters, making it ideal for compute- and latency-constrained environments.

---

## 14. Explainable Fundus Image Curation and Lesion Detection in Diabetic Retinopathy

**论文链接:** [http://arxiv.org/abs/2512.08986v1](http://arxiv.org/abs/2512.08986v1)

**作者:** Anca Mihai, Adrian Groza

**发布时间:** 2025-12-06

### GPT解析

### 总结

本文提出了一种质量控制框架，用于确保糖尿病视网膜病变诊断中使用的眼底图像数据质量，以提高AI模型的准确性和可靠性。

### 背景

糖尿病视网膜病变(DR)影响长期糖尿病患者，若不及时诊断可导致视力丧失。眼底摄影可捕捉视网膜结构和异常，表明疾病阶段。人工智能可辅助临床医生识别病变，但需要高质量标注数据集。

### 目的

解决视网膜图像采集和人工标注中可能出现的错误问题，确保只有高质量数据用于AI模型的评估和训练。

### 方法

提出一种质量控制框架：1)使用基于可解释特征的分类器过滤不合适图像，特征通过图像处理和对比学习提取；2)增强图像并进行标注，使用基于深度学习的辅助；3)通过计算标注者间一致性确定标注可用性。

### 主要发现

视网膜结构的复杂性导致图像采集和病变解释中可能出现错误，需要质量控制框架确保数据质量。

### 结论

所提出的质量控制框架可确保使用高标准数据进行AI模型的训练和评估，提高糖尿病视网膜病变诊断的准确性。

### 翻译

糖尿病视网膜病变(DR)影响长期糖尿病患者。若无早期诊断，DR可导致视力丧失。眼底摄影可捕捉视网膜结构和表明疾病阶段的异常。人工智能可支持临床医生识别这些病变，减少人工工作量，但模型需要高质量标注数据集。由于视网膜结构复杂，图像采集和病变解释中可能出现错误。我们提出了一种质量控制框架，确保只有高质量数据用于评估和AI训练。首先，使用基于可解释特征的分类器过滤不合适图像，特征通过图像处理和对比学习提取。然后，图像被增强并进行标注，使用基于深度学习的辅助。最后，使用推导公式计算标注者间的一致性，确定标注的可用性。


### 论文摘要

Diabetic Retinopathy (DR) affects individuals with long-term diabetes. Without early diagnosis, DR can lead to vision loss. Fundus photography captures the structure of the retina along with abnormalities indicative of the stage of the disease. Artificial Intelligence (AI) can support clinicians in identifying these lesions, reducing manual workload, but models require high-quality annotated datasets. Due to the complexity of retinal structures, errors in image acquisition and lesion interpretation of manual annotators can occur. We proposed a quality-control framework, ensuring only high-standard data is used for evaluation and AI training. First, an explainable feature-based classifier is used to filter inadequate images. The features are extracted both using image processing and contrastive learning. Then, the images are enhanced and put subject to annotation, using deep-learning-based assistance. Lastly, the agreement between annotators calculated using derived formulas determines the usability of the annotations.

---

## 15. Log NeRF: Comparing Spaces for Learning Radiance Fields

**论文链接:** [http://arxiv.org/abs/2512.09375v1](http://arxiv.org/abs/2512.09375v1)

**作者:** Sihe Chen, Luv Verma, Bruce A. Maxwell

**发布时间:** 2025-12-10

**备注:** The 36th British Machine Vision Conference

### GPT解析

### 总结

本研究探讨了在Neural Radiance Fields (NeRF)中使用不同颜色空间对渲染效果的影响，发现对数RGB(log RGB)空间能够提高渲染质量、增强场景鲁棒性，特别是在低光照条件下表现优异。

### 背景

NeRF在新型视图合成方面已取得显著成果，通常使用sRGB图像进行监督，但很少关注网络学习辐射场表示时所使用的颜色空间。

### 目的

验证对数RGB空间是否能使NeRF学习场景外观更紧凑、更有效的表示，以提高渲染质量和鲁棒性。

### 方法

使用GoPro相机拍摄约30个视频，确保线性数据恢复；在不同颜色空间(线性、sRGB、GPLog和对数RGB)下训练NeRF模型，将网络输出转换为通用颜色空间后再进行渲染和损失计算；通过定量和定性评估比较不同颜色空间的表现。

### 主要发现

使用对数RGB颜色空间可提高渲染质量，在不同场景中表现出更强的鲁棒性，在低光照条件下表现特别良好，且使用相同位深的输入图像即可实现这些改进。

### 结论

对数RGB空间为NeRF提供了更紧凑、更有效的场景外观表示方式，这一优势在不同网络大小和NeRF变体上具有泛化性和稳定性。

### 翻译

神经辐射场(NeRF)在新型视图合成方面取得了显著成果，通常使用sRGB图像进行监督。然而，很少关注网络学习辐射场表示时所使用的颜色空间。受双照明二色反射(BIDR)模型启发，该模型表明对数变换可以简化光照和反射率的分离，我们假设对数RGB空间能使NeRF学习场景外观更紧凑、更有效的表示。为此，我们使用GoPro相机拍摄约30个视频，通过逆编码确保线性数据恢复。我们在不同颜色空间解释下训练NeRF模型：线性、sRGB、GPLog和对数RGB，通过将每个网络输出转换为通用颜色空间后再进行渲染和损失计算，强制在不同颜色空间中进行表示学习。定性和定量评估表明，使用对数RGB颜色空间可提高渲染质量，在不同场景中表现出更强的鲁棒性，在低光照条件下表现特别良好，同时使用相同位深的输入图像。对不同网络大小和NeRF变体的进一步分析确认了对数空间优势的泛化性和稳定性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要研究不同颜色空间对神经辐射场(NeRF)学习效果的影响。这个问题很重要，因为目前大多数NeRF研究使用sRGB图像进行监督，但很少关注颜色空间对辐射场学习的影响。颜色空间选择可能影响NeRF的学习效率、表示紧凑性、鲁棒性和动态范围，特别是在低光条件下对重建质量有显著影响。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到双光源二色反射(BIDR)模型的启发，该模型表明对数变换可以简化光照和反射率的分离。基于这一理论，作者假设在log-RGB空间中，NeRF能够学习到场景外观更紧凑、更有效的表示。他们借鉴了BiLaRF NeRF模型作为实验基础，并参考了GoPro相机的GPLog编码技术和RawNeRF使用原始HDR数据的思想，但表明不需要更高位深就能获得高质量结果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在log-RGB颜色空间中学习神经辐射场，利用log空间中反射率和光照项可以分离的特性，使网络能够更有效地表示场景外观，特别是在低光条件下。整体流程包括：1)使用GoPro相机捕获视频；2)将GPLog编码转换为线性RGB并提取相机姿态；3)在BiLaRF架构基础上添加表示空间变换、sRGB变换和输入数据变换；4)比较四种颜色空间(GPLog、线性RGB、sRGB和TrueLog)在NeRF训练中的表现；5)使用PSNR、颜色渲染、深度图可视化等多种指标评估结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统研究了颜色空间对NeRF学习的影响；2)提出在log-RGB空间学习NeRF可显著提高重建质量，特别是在低光条件下；3)展示了如何控制NeRF网络的隐式表示空间；4)证明了log RGB的特殊性，类似变换无法带来相同好处；5)贡献了新的GPLog编码格式的NeRF视频及处理流程。相比之前工作，大多数NeRF研究假设在sRGB空间学习，而RawNeRF虽使用原始HDR数据但需要更高位深，本文方法无需更高位深就能获得高质量结果，并提供了理论解释说明为什么log空间更适合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文证明了在log-RGB颜色空间中学习神经辐射场能够显著提高重建质量、增强低光条件下的表现、提高模型鲁棒性，并提供了控制NeRF隐式表示空间的方法和理论基础。'}


### 论文摘要

Neural Radiance Fields (NeRF) have achieved remarkable results in novel view synthesis, typically using sRGB images for supervision. However, little attention has been paid to the color space in which the network is learning the radiance field representation. Inspired by the BiIlluminant Dichromatic Reflection (BIDR) model, which suggests that a logarithmic transformation simplifies the separation of illumination and reflectance, we hypothesize that log RGB space enables NeRF to learn a more compact and effective representation of scene appearance. To test this, we captured approximately 30 videos using a GoPro camera, ensuring linear data recovery through inverse encoding. We trained NeRF models under various color space interpretations linear, sRGB, GPLog, and log RGB by converting each network output to a common color space before rendering and loss computation, enforcing representation learning in different color spaces. Quantitative and qualitative evaluations demonstrate that using a log RGB color space consistently improves rendering quality, exhibits greater robustness across scenes, and performs particularly well in low light conditions while using the same bit-depth input images. Further analysis across different network sizes and NeRF variants confirms the generalization and stability of the log space advantage.

---

## 16. Efficiency-Aware Computational Intelligence for Resource-Constrained Manufacturing Toward Edge-Ready Deployment

**论文链接:** [http://arxiv.org/abs/2512.09319v1](http://arxiv.org/abs/2512.09319v1)

**作者:** Qianyu Zhou

**发布时间:** 2025-12-10

**备注:** 2025, University of Connecticut

### GPT解析

### 总结

该论文开发了一个基于效率的计算框架，用于解决工业信息物理系统中的数据挑战和资源限制问题，实现了数据稀少、物理感知和部署就绪的智能系统。

### 背景

工业信息物理系统面临异构感知、随机动态和变化的过程条件，产生的数据常常不完整、未标记、不平衡且存在领域偏移。高保真数据集成本高、保密性强且获取缓慢，而边缘设备面临延迟、带宽和能量的严格限制。

### 目的

开发一个效率导向的计算框架，实现现代制造环境中的数据稀少、物理感知和部署就绪的智能，解决多模态和多尺度工业场景中的核心瓶颈。

### 方法

采用生成策略缓解数据稀缺和不平衡；使用半监督学习整合未标记信息减少标注需求；通过物理感知的表示学习增强可解释性；利用空间感知的基于图的代理建模提供复杂过程的高效近似；实施边缘云协作压缩方案支持资源约束下的实时分析；结合领域特定检索增强零样本视觉语言推理能力。

### 主要发现

这些方法共同解决了多模态和多尺度工业场景中的核心瓶颈，零样本视觉语言推理结合领域特定检索使系统能够在未见场景中进行泛化评估。

### 结论

这些发展建立了数据高效和资源感知智能的统一范式，弥合了实验室学习与工业部署之间的差距，支持多样化制造系统中的可靠决策。

### 翻译

工业信息物理系统在异构感知、随机动态和变化的过程条件下运行，产生的数据往往不完整、未标记、不平衡且存在领域偏移。高保真数据集仍然成本高昂、保密性强且获取缓慢，而边缘设备面临延迟、带宽和能量的严格限制。这些因素限制了集中式深度学习的实用性，阻碍了可靠数字孪生的发展，并增加了安全关键应用中错误逃逸的风险。受这些挑战的启发，本论文开发了一个基于效率的计算框架，使现代制造环境能够实现数据稀少、物理感知和部署就绪的智能。研究推进了能够共同解决多模态和多尺度工业场景核心瓶颈的方法。生成策略缓解了数据稀缺和不平衡，而半监督学习整合了未标记信息以减少标注和仿真需求。物理感知的表示学习增强了可解释性，并改善了小数据条件下的状态监测。空间感知的基于图的代理建模提供了复杂过程的高效近似，边缘云协作压缩方案支持资源约束下的实时信号分析。本论文还通过领域特定检索增强的零样本视觉语言推理扩展了视觉理解，使系统能够在先前未见过的场景中进行泛化评估。这些发展共同建立了数据高效和资源感知智能的统一范式，弥合了实验室学习与工业部署之间的差距，支持多样化制造系统中的可靠决策。


### 论文摘要

Industrial cyber physical systems operate under heterogeneous sensing, stochastic dynamics, and shifting process conditions, producing data that are often incomplete, unlabeled, imbalanced, and domain shifted. High-fidelity datasets remain costly, confidential, and slow to obtain, while edge devices face strict limits on latency, bandwidth, and energy. These factors restrict the practicality of centralized deep learning, hinder the development of reliable digital twins, and increase the risk of error escape in safety-critical applications. Motivated by these challenges, this dissertation develops an efficiency grounded computational framework that enables data lean, physics-aware, and deployment ready intelligence for modern manufacturing environments. The research advances methods that collectively address core bottlenecks across multimodal and multiscale industrial scenarios. Generative strategies mitigate data scarcity and imbalance, while semi-supervised learning integrates unlabeled information to reduce annotation and simulation demands. Physics-informed representation learning strengthens interpretability and improves condition monitoring under small-data regimes. Spatially aware graph-based surrogate modeling provides efficient approximation of complex processes, and an edge cloud collaborative compression scheme supports real-time signal analytics under resource constraints. The dissertation also extends visual understanding through zero-shot vision language reasoning augmented by domain specific retrieval, enabling generalizable assessment in previously unseen scenarios. Together, these developments establish a unified paradigm of data efficient and resource aware intelligence that bridges laboratory learning with industrial deployment, supporting reliable decision-making across diverse manufacturing systems.

---

## 17. Dual Refinement Cycle Learning: Unsupervised Text Classification of Mamba and Community Detection on Text Attributed Graph

**论文链接:** [http://arxiv.org/abs/2512.07100v2](http://arxiv.org/abs/2512.07100v2)

**作者:** Hong Wang, Yinglong Zhang, Hanhan Guo, Xuewen Xia, Xing Xu

**发布时间:** 2025-12-08

### GPT解析

### 总结

本文提出了一种名为双重精炼周期学习(DRCL)的完全无监督框架，整合了结构和语义信息，解决了预训练语言模型依赖标记数据和传统社区检测忽略文本语义的问题。

### 背景

预训练语言模型虽有强大文本理解能力，但因依赖标记数据难以在实际文本属性网络中部署；同时，社区检测方法通常忽略文本语义，限制了其在内容组织、推荐和风险监控等下游应用中的实用性。

### 目的

克服现有方法的局限性，设计适用于没有标签或类别定义实际场景的完全无监督框架，实现文本属性网络中的有效社区检测。

### 方法

DRCL通过预热初始化以及基于GCN的社区检测模块(GCN-CDM)和文本语义建模模块(TSMM)之间的双向精炼周期整合结构和语义信息。两个模块迭代交换伪标签，使语义线索增强结构聚类，结构模式指导文本表示学习，无需人工监督。

### 主要发现

在多个文本属性图数据集上，DRCL持续改进了发现社区的结构和语义质量；仅从DRCL的社区信号训练的基于Mamba的分类器实现了与监督模型相当的准确性。

### 结论

DRCL展示了在标记数据稀缺或成本高昂的大规模系统中部署的潜力，为实际应用中的文本属性网络社区检测提供了有效解决方案。

### 翻译

预训练语言模型提供强大的文本理解能力，但由于严重依赖标记数据，仍然难以在实际文本属性网络中部署。同时，社区检测方法通常忽略文本语义，限制了它们在内容组织、推荐和风险监控等下游应用中的实用性。为了克服这些限制，我们提出了双重精炼周期学习(DRCL)，这是一种完全无监督的框架，专为没有标签或类别定义的实际场景设计。DRCL通过预热初始化以及基于GCN的社区检测模块(GCN-CDM)和文本语义建模模块(TSMM)之间的双向精炼周期，整合结构和语义信息。两个模块迭代交换伪标签，允许语义线索增强结构聚类，结构模式指导文本表示学习，无需人工监督。在多个文本属性图数据集上，DRCL持续改进了发现社区的结构和语义质量。此外，仅从DRCL的社区信号训练的基于Mamba的分类器实现了与监督模型相当的准确性，证明了它在标记数据稀缺或成本高昂的大规模系统中部署的潜力。代码可在https://github.com/wuanghoong/DRCL.git获取。


### 论文摘要

Pretrained language models offer strong text understanding capabilities but remain difficult to deploy in real-world text-attributed networks due to their heavy dependence on labeled data. Meanwhile, community detection methods typically ignore textual semantics, limiting their usefulness in downstream applications such as content organization, recommendation, and risk monitoring. To overcome these limitations, we present Dual Refinement Cycle Learning (DRCL), a fully unsupervised framework designed for practical scenarios where no labels or category definitions are available. DRCL integrates structural and semantic information through a warm-start initialization and a bidirectional refinement cycle between a GCN-based Community Detection Module (GCN-CDM) and a Text Semantic Modeling Module (TSMM). The two modules iteratively exchange pseudo-labels, allowing semantic cues to enhance structural clustering and structural patterns to guide text representation learning without manual supervision. Across several text-attributed graph datasets, DRCL consistently improves the structural and semantic quality of discovered communities. Moreover, a Mamba-based classifier trained solely from DRCL's community signals achieves accuracy comparable to supervised models, demonstrating its potential for deployment in large-scale systems where labeled data are scarce or costly. The code is available at https://github.com/wuanghoong/DRCL.git.

---

## 18. Generative Point Cloud Registration

**论文链接:** [http://arxiv.org/abs/2512.09407v1](http://arxiv.org/abs/2512.09407v1)

**作者:** Haobo Jiang, Jin Xie, Jian Yang, Liang Yu, Jianmin Zheng

**发布时间:** 2025-12-10

**备注:** 14 pages, 9 figures

### GPT解析

### 总结

提出了一种新颖的3D配准范式——生成式点云配准，结合先进的2D生成模型与3D匹配任务以提高配准性能，引入了Match-ControlNet模型，并在3DMatch和ScanNet数据集上验证了有效性。

### 背景

3D点云配准是计算机视觉中的重要任务，传统方法可能存在几何和纹理一致性的挑战，而2D生成模型在图像生成方面取得了显著进展，但尚未充分应用于3D配准任务。

### 目的

将先进的2D生成模型与3D匹配任务结合，生成跨视图一致的图像对，促进几何-颜色特征融合，实现稳健的3D点云配准。

### 方法

提出生成式3D配准范式，开发Match-ControlNet模型利用ControlNet的深度条件生成能力，通过耦合条件去噪方案和耦合提示指导，确保生成图像对具有2D-3D几何一致性和跨视图纹理一致性。

### 主要发现

生成的图像对能有效促进几何-颜色特征融合；Match-ControlNet能生成具有2D-3D几何一致性的图像；耦合条件去噪方案和提示指导能促进跨视图特征交互；该生成式3D配准范式可无缝集成到各种配准方法中以提高性能。

### 结论

提出的生成式3D配准范式是通用的，可以增强各种配准方法的性能，在3DMatch和ScanNet数据集上的实验验证了该方法的有效性。

### 翻译

在这篇论文中，我们提出了一种新颖的3D配准范式——生成式点云配准，它将先进的2D生成模型与3D匹配任务相结合以提高配准性能。我们的核心思想是生成与源点云和目标点云良好对齐的跨视图一致的图像对，从而促进几何-颜色特征融合以实现稳健的匹配。为确保高质量的匹配，生成的图像对应同时具有2D-3D几何一致性和跨视图纹理一致性。为此，我们引入了Match-ControlNet，一种匹配特定的可控2D生成模型。具体而言，它利用ControlNet的深度条件生成能力，生成与点云导出的深度图几何对齐的图像，确保2D-3D几何一致性。此外，通过引入耦合条件去噪方案和耦合提示指导，Match-ControlNet进一步促进了跨视图特征交互，引导纹理一致性生成。我们的生成式3D配准范式是通用的，可以无缝集成到各种配准方法中以增强其性能。在3DMatch和ScanNet数据集上的大量实验验证了我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云配准中的挑战，特别是在低重叠度、噪声点等困难场景下的鲁棒性问题。这个问题很重要，因为点云配准是3D重建、LiDAR SLAM和物体定位等下游应用的核心技术，而现实世界中的挑战限制了现有方法在更广泛场景中的应用。现有方法主要依赖几何信息，缺乏颜色和纹理信息的辅助，而研究表明颜色信息能显著提高匹配质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到RGB-D配准研究的启发，思考如何在没有RGB数据的情况下利用颜色信息增强几何描述符。他们借鉴了ControlNet的深度条件生成能力，预训练大视觉模型（如DINOv2和Stable Diffusion）的特征提取能力，以及现有颜色点云配准方法（如ColorPCR）的融合策略。通过生成式AI模型，他们设计了一种新范式，为纯几何点云生成配对的RGB图像，以补充几何信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过生成模型为点云对生成配对的RGB图像，提供丰富的颜色信息来补充几何特征，同时确保生成的图像对具有2D-3D几何一致性和跨视图纹理一致性。整体流程包括：1)使用Match-ControlNet生成与点云对应的图像对；2)通过零样本几何-颜色特征融合或XYZ-RGB融合将颜色信息与几何特征结合；3)使用增强后的特征进行对应关系估计和姿态估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出生成式点云配准新范式；2)开发Match-ControlNet实现匹配特定的成对图像生成，确保几何和纹理一致性；3)提出零样本几何-颜色融合机制；4)框架具有通用性和即插即用性。相比之前工作，不同之处在于：传统方法仅使用几何信息，而本文引入生成颜色信息；实现了成对而非单图像生成；确保了生成的图像对同时具有几何和纹理一致性；适用于纯几何点云场景，无需实际RGB图像。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种生成式点云配准新范式，通过合成与点云几何结构一致的成对RGB图像并将颜色信息与几何特征融合，显著提高了在挑战场景下的点云配准性能。'}


### 论文摘要

In this paper, we propose a novel 3D registration paradigm, Generative Point Cloud Registration, which bridges advanced 2D generative models with 3D matching tasks to enhance registration performance. Our key idea is to generate cross-view consistent image pairs that are well-aligned with the source and target point clouds, enabling geometry-color feature fusion to facilitate robust matching. To ensure high-quality matching, the generated image pair should feature both 2D-3D geometric consistency and cross-view texture consistency. To achieve this, we introduce Match-ControlNet, a matching-specific, controllable 2D generative model. Specifically, it leverages the depth-conditioned generation capability of ControlNet to produce images that are geometrically aligned with depth maps derived from point clouds, ensuring 2D-3D geometric consistency. Additionally, by incorporating a coupled conditional denoising scheme and coupled prompt guidance, Match-ControlNet further promotes cross-view feature interaction, guiding texture consistency generation. Our generative 3D registration paradigm is general and could be seamlessly integrated into various registration methods to enhance their performance. Extensive experiments on 3DMatch and ScanNet datasets verify the effectiveness of our approach.

---

## 19. FunPhase: A Periodic Functional Autoencoder for Motion Generation via Phase Manifolds

**论文链接:** [http://arxiv.org/abs/2512.09423v1](http://arxiv.org/abs/2512.09423v1)

**作者:** Marco Pegoraro, Evan Atherton, Bruno Roy, Aliasghar Khani, Arianna Rampini

**发布时间:** 2025-12-10

### GPT解析

### 总结

该论文介绍了一种名为FunPhase的功能周期性自编码器，用于学习人体运动的相位流形，通过函数空间公式替代离散时间解码，实现任意时间分辨率的平滑轨迹采样。

### 背景

学习自然人体运动具有挑战性，因为空间几何和时间动态之间存在强耦合。现有将运动嵌入到相位流形的方法虽然有效，但缺乏可扩展性且局限于特定设置。

### 目的

开发一种新的方法来学习人体运动的相位流形，解决现有方法在可扩展性和应用范围上的局限性，实现更灵活、更通用的运动建模。

### 方法

提出FunPhase，一种功能周期性自编码器，学习运动的相位流形，并用函数空间公式替代离散时间解码，支持超分辨率和部分身体运动完成等下游任务，具有跨骨架和数据集的泛化能力。

### 主要发现

FunPhase模型比先前的周期性自编码器基线实现了显著更低的重建误差，同时支持更广泛的应用范围，并且与最先进的运动生成方法性能相当。

### 结论

FunPhase通过函数空间公式和周期性自编码器的创新应用，解决了人体运动建模中的关键挑战，提供了一种更灵活、更通用的方法来处理运动预测和生成任务。

### 翻译

学习自然人体运动仍然具有挑战性，因为空间几何和时间动态之间存在强耦合。将运动嵌入到相位流形（捕获局部周期性的潜在空间）中已被证明对运动预测有效；然而，现有方法缺乏可扩展性，仍局限于特定设置。我们引入了FunPhase，一种功能周期性自编码器，它学习运动的相位流形，并用函数空间公式替代离散时间解码，从而能够以任意时间分辨率采样平滑轨迹。FunPhase支持超分辨率和部分身体运动完成等下游任务，具有跨骨架和数据集的泛化能力，并将运动预测和生成统一在单个可解释流形中。我们的模型比先前的周期性自编码器基线实现了显著更低的重建误差，同时支持更广泛的应用范围，并且与最先进的运动生成方法性能相当。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自然运动学习中的空间几何与时间动态强耦合问题，以及现有方法缺乏可扩展性和局限于特定设置的问题。这个问题在现实中很重要，因为运动生成是计算机视觉、图形学、机器人和人机交互的核心能力，应用包括虚拟角色动画、视频游戏和具身智能体等，能有效自动化传统耗时的动画制作过程，同时实现行为建模、仿真和数据增强。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有的相位表示方法（如DeepPhase）在运动预测方面有效但缺乏可扩展性，于是想扩展学习到的流形框架使其与现代生成模型兼容。他们借鉴了DeepPhase的相位流形概念和功能生成框架（FunDiff）的思想，保留了相位分解的可解释性，同时采用Perceiver架构进行编码解码，并使用扩散模型进行运动生成。FunPhase本质上是将DeepPhase的周期性自编码器扩展到功能空间，使其能够处理任意时间分辨率和不同骨骼结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将运动表示为连续的时空函数而非离散帧序列，学习一个相位流形来捕获运动的周期性结构。整体流程分为两个阶段：首先是周期函数自编码器（FunPhase），它使用Perceiver-based编码器处理关节旋转和根位置，通过周期分解将潜在空间分解为周期组件，再用功能解码器重建运动；其次是相位扩散，在学习的相位流形上操作扩散模型，对周期参数进行域变换使其与高斯扩散兼容，使用扩散变换器处理1D潜在序列，最后通过采样和逆变换生成新运动。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出功能周期性自编码器FunPhase；将运动重建为时空函数同时保持有意义的相位流形；支持多种下游任务如运动超分辨率和部分身体运动补全；引入在相位流形上操作的神经运动控制器；使用功能空间中的扩散实现先进运动生成。相比DeepPhase，FunPhase是功能性的，可处理任意时间分辨率，更可扩展且不依赖固定骨骼；相比其他运动生成模型，它使用函数空间而非基于帧的表示，允许变速解码和改进稳定性；相比FunDiff，它专门针对运动数据，结合相位流形学习和功能解码器。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FunPhase通过将运动表示为带有相位流形的连续时空函数，统一了解释性运动预测和生成，实现了跨骨骼和数据集的高保真度、物理合理的运动生成。'}


### 论文摘要

Learning natural body motion remains challenging due to the strong coupling between spatial geometry and temporal dynamics. Embedding motion in phase manifolds, latent spaces that capture local periodicity, has proven effective for motion prediction; however, existing approaches lack scalability and remain confined to specific settings. We introduce FunPhase, a functional periodic autoencoder that learns a phase manifold for motion and replaces discrete temporal decoding with a function-space formulation, enabling smooth trajectories that can be sampled at arbitrary temporal resolutions. FunPhase supports downstream tasks such as super-resolution and partial-body motion completion, generalizes across skeletons and datasets, and unifies motion prediction and generation within a single interpretable manifold. Our model achieves substantially lower reconstruction error than prior periodic autoencoder baselines while enabling a broader range of applications and performing on par with state-of-the-art motion generation methods.

---

## 20. From Forecast to Action: Uncertainty-Aware UAV Deployment for Ocean Drifter Recovery

**论文链接:** [http://arxiv.org/abs/2512.09260v1](http://arxiv.org/abs/2512.09260v1)

**作者:** Jingeun Kim, Yong-Hyuk Kim, Yourim Yoon

**发布时间:** 2025-12-10

**备注:** This paper has been accepted by CIKM 2025 STIntelligence Workshop

### GPT解析

### 总结

论文提出了一种新颖的预测-优化框架，用于海上搜索行动，将轨迹预测与无人机部署优化相结合，形成端到端解决方案。

### 背景

传统的静态部署方法存在局限性，先前的研究没有解决轨迹预测与部署优化的端到端集成问题。

### 目的

开发一种结合轨迹预测和空间优化的方法，用于提升海上搜索和救援行动的效率。

### 方法

使用大型语言模型预测漂流物轨迹，采用基于高斯的粒子采样建模空间不确定性，动态调整无人机探测半径，并使用元启发式算法优化无人机部署。

### 主要发现

在韩国海岸线真实数据上的实验表明，该方法（特别是为此问题设计的修复机制）显著优于随机搜索基线。

### 结论

该工作为智能海上救援引入了轨迹预测和空间优化的实用且鲁棒的集成方法。

### 翻译

我们提出了一种用于海上搜索行动的新型预测-优化框架，将轨迹预测与无人机部署优化相结合——这是一种先前研究中未解决的端到端方法。大型语言模型预测漂流物的轨迹，并使用基于高斯的粒子采样对空间不确定性进行建模。与传统的静态部署方法不同，我们根据距离动态调整无人机探测半径，并使用元启发式算法优化其部署。在韩国海岸线真实数据上的实验表明，我们的方法（特别是为此问题设计的修复机制）显著优于随机搜索基线。这项工作为智能海上救援引入了轨迹预测和空间优化的实用且鲁棒的集成。


### 论文摘要

We present a novel predict-then-optimize framework for maritime search operations that integrates trajectory forecasting with UAV deployment optimization-an end-to-end approach not addressed in prior work. A large language model predicts the drifter's trajectory, and spatial uncertainty is modeled using Gaussian-based particle sampling. Unlike traditional static deployment methods, we dynamically adapt UAV detection radii based on distance and optimize their placement using meta-heuristic algorithms. Experiments on real-world data from the Korean coastline demonstrate that our method, particularly the repair mechanism designed for this problem, significantly outperforms the random search baselines. This work introduces a practical and robust integration of trajectory prediction and spatial optimization for intelligent maritime rescue.

---

## 21. LUMOS: Large User MOdels for User Behavior Prediction

**论文链接:** [http://arxiv.org/abs/2512.08957v1](http://arxiv.org/abs/2512.08957v1)

**作者:** Dhruv Nigam

**发布时间:** 2025-11-28

### GPT解析

### 总结

本文介绍了一种名为LUMOS的基于Transformer的架构，用于大规模用户行为预测。该架构通过联合学习多个任务，消除了对特定任务模型和手动特征工程的依赖，仅使用原始用户活动数据。LUMOS引入了交叉注意力机制和多模态标记化技术，实验表明其在多个任务上优于传统模型，并带来了可衡量的业务影响。

### 背景

大规模用户行为预测仍然是在线B2C平台面临的关键挑战。传统方法严重依赖于特定任务模型和特定领域的特征工程，这既耗时又计算密集，需要领域专业知识，因此难以扩展。

### 目的

开发一种能够有效预测用户行为且可扩展的解决方案，消除对特定任务模型和手动特征工程的依赖，并提高预测性能。

### 方法

LUMOS是一种基于Transformer的架构，通过联合学习多个任务消除特定任务模型和手动特征工程；引入交叉注意力机制，利用未来已知事件（如节假日、促销等）来条件化预测；采用多模态标记化技术，结合用户交易、事件上下文和静态用户人口统计属性，通过专门的嵌入路径处理这些数据。

### 主要发现

在包含2.5亿用户产生的2750亿用户活动令牌的生产数据集上，LUMOS相比传统特定任务模型具有优越性能；在5个具有既定基线的任务中，二元分类任务的ROC-AUC平均提高0.025，回归任务的MAPE平均减少4.6%；在线A/B测试验证了这些改进转化为可衡量的业务影响，日活跃用户增加3.15%。

### 结论

LUMOS为大规模用户行为预测提供了一个有效且可扩展的解决方案，通过消除对特定任务模型和手动特征工程的依赖，显著提高了预测性能，并带来了可衡量的业务价值。

### 翻译

大规模用户行为预测仍然是在线B2C平台面临的关键挑战。传统方法严重依赖于特定任务模型和特定领域的特征工程。这既耗时又计算密集，需要领域专业知识，因此难以扩展。我们提出了LUMOS（大型用户模型系列），这是一种基于Transformer的架构，它通过仅使用原始用户活动数据联合学习多个任务，消除了特定任务模型和手动特征工程。LUMOS引入了一种新颖的交叉注意力机制，将预测基于未来已知事件（如节假日、促销等）进行条件化，使模型能够预测复杂的行为模式，如'即将到来的节假日将如何影响用户参与度？'。该架构还采用多模态标记化技术，将用户交易、事件上下文和静态用户人口统计属性结合起来，通过专门的嵌入路径处理成丰富的表示。通过对包含2.5亿用户产生的2750亿用户活动令牌的生产数据集进行广泛实验，我们证明LUMOS相比传统特定任务模型实现了优越的性能。在5个具有既定基线的任务中，二元分类任务的ROC-AUC平均提高0.025，回归任务的MAPE平均减少4.6%。在线A/B测试验证了这些改进转化为可衡量的业务影响，日活跃用户增加3.15%。


### 论文摘要

User behavior prediction at scale remains a critical challenge for online B2C platforms. Traditional approaches rely heavily on task-specific models and domain-specific feature engineering. This is time-consuming, computationally expensive, and requires domain expertise and therefore not scalable. We present LUMOS (Large User MOdel Series), a transformer-based architecture that eliminates task-specific models and manual feature engineering by learning multiple tasks jointly using only raw user activity data. LUMOS introduces a novel cross-attention mechanism that conditions predictions on future known events (e.g., holidays, sales, etc.), enabling the model to predict complex behaviour patterns like "how will upcoming holidays affect user engagement?" The architecture also employs multi-modal tokenization, combining user transactions, event context, and static user demographic attributes into rich representations processed through specialized embedding pathways.   Through extensive experiments on a production dataset spanning 275 billion user activity tokens from 250 million users, we demonstrate that LUMOS achieves superior performance compared to traditional task-specific models. Across 5 tasks with established baselines, we achieve an average improvement of 0.025 in ROC-AUC for binary classification tasks and 4.6\% reduction in MAPE for regression tasks. Online A/B testing validates these improvements translate to measurable business impact with a 3.15\% increase in Daily Active Users.

---

## 22. Analysis of Dirichlet Energies as Over-smoothing Measures

**论文链接:** [http://arxiv.org/abs/2512.09890v1](http://arxiv.org/abs/2512.09890v1)

**作者:** Anna Bison, Alessandro Sperduti

**发布时间:** 2025-12-10

### GPT解析

### 总结

本研究分析了两种图拉普拉斯矩阵（非归一化和归一化）诱导的狄利克雷能量作为过平滑度量函数的区别，解决了图神经网络中监控动态过程的模糊性问题。

### 背景

在图神经网络研究中，存在两种常用的过平滑度量函数：由非归一化图拉普拉斯矩阵和归一化图拉普拉斯矩阵诱导的狄利克雷能量。这些度量函数在评估和监控图神经网络性能时被广泛使用，但它们之间的区别和适用性尚未被充分理解。

### 目的

明确两种图拉普拉斯矩阵诱导的狄利克雷能量作为过平滑度量函数之间的区别，解决在监控图神经网络动态过程中的模糊性问题，并提供选择与图神经网络架构谱兼容的度量的指导原则。

### 方法

通过形式化两种图拉普拉斯矩阵（非归一化和归一化）诱导的狄利克雷能量的基本谱特性，分析它们在作为节点相似度度量时的公理化性质，特别是验证它们是否满足Rusch等人提出的节点相似度度量的公理化定义。

### 主要发现

1) 归一化图拉普拉斯矩阵诱导的狄利克雷能量不满足Rusch等人提出的节点相似度度量的公理化定义；2) 两种图拉普拉斯矩阵诱导的狄利克雷能量具有不同的谱特性；3) 选择与图神经网络架构谱兼容的度量对于准确监控动态过程至关重要。

### 结论

在评估图神经网络的过平滑问题时，研究者需要根据图神经网络架构的谱特性选择适当的度量函数。明确理解这些区别有助于更准确地监控和优化图神经网络的动态过程。

### 翻译

我们分析了两种常用作过平滑度量的函数之间的区别：由非归一化图拉普拉斯矩阵和归一化图拉普拉斯矩阵诱导的狄利克雷能量。我们证明了后者不满足Rusch等人提出的节点相似度度量的公理化定义。通过形式化这两种定义的基本谱特性，我们强调了选择与图神经网络架构谱兼容的度量所需的关键区别，从而解决了监控动态过程中的模糊性问题。


### 论文摘要

We analyze the distinctions between two functionals often used as over-smoothing measures: the Dirichlet energies induced by the unnormalized graph Laplacian and the normalized graph Laplacian. We demonstrate that the latter fails to satisfy the axiomatic definition of a node-similarity measure proposed by Rusch \textit{et al.} By formalizing fundamental spectral properties of these two definitions, we highlight critical distinctions necessary to select the metric that is spectrally compatible with the GNN architecture, thereby resolving ambiguities in monitoring the dynamics.

---

## 23. M3Net: A Multi-Metric Mixture of Experts Network Digital Twin with Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.09797v1](http://arxiv.org/abs/2512.09797v1)

**作者:** Blessed Guda, Carlee Joe-Wong

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了一种名为M3Net的多指标专家混合网络数字孪生模型，用于提高网络性能预测的准确性。

### 背景

5G/6G网络技术发展带来连接设备数量大幅增加，使网络管理复杂化，且相关应用对延迟和可靠性等指标有严格且多样化的性能要求。

### 目的

开发一种能够平衡准确性和可扩展性的网络性能预测方法，解决传统网络建模方法的局限性，并超越现有只关注单一性能指标的网络数字孪生模型。

### 方法

提出M3Net，一种基于图神经网络架构的多指标专家混合网络数字孪生模型，能够从扩展的网络状态数据集中估计多种性能指标。

### 主要发现

M3Net显著提高了流延迟预测的准确性，将MAPE从20.06%降低到17.39%，同时在抖动和丢包率预测方面分别实现了66.47%和78.7%的准确率。

### 结论

M3Net是一种有效的解决方案，能够处理多种性能指标，提高网络性能预测的准确性，为5G/6G网络管理提供了新的可能性。

### 翻译

5G/6G网络技术的兴起有望实现自动驾驶车辆和虚拟现实等应用，导致连接设备数量大幅增加，并必然使网络管理复杂化。更糟糕的是，这些应用通常对延迟和可靠性等指标有着严格但又多样化的性能要求。因此，最近许多工作都集中在开发网络性能预测能力上。然而，传统的网络建模方法，如离散事件模拟器和仿真，往往无法平衡准确性和可扩展性。由机器学习增强的网络数字孪生(NDTs)通过创建物理网络的虚拟副本进行实时模拟和分析，提供了一种可行的解决方案。然而，最先进的模型尚未达到完整的NDT水平，因为它们通常只关注单一性能指标或模拟网络数据。我们介绍了M3Net，这是一种多指标专家混合(MoE) NDT，它使用图神经网络架构从扩展的网络状态数据集中估计各种场景下的多种性能指标。我们证明，M3Net通过将MAPE(平均绝对百分比误差)从20.06%降低到17.39%，显著提高了流延迟预测的准确性，同时在每个流的抖动和丢包方面分别实现了66.47%和78.7%的准确率。


### 论文摘要

The rise of 5G/6G network technologies promises to enable applications like autonomous vehicles and virtual reality, resulting in a significant increase in connected devices and necessarily complicating network management. Even worse, these applications often have strict, yet heterogeneous, performance requirements across metrics like latency and reliability. Much recent work has thus focused on developing the ability to predict network performance. However, traditional methods for network modeling, like discrete event simulators and emulation, often fail to balance accuracy and scalability. Network Digital Twins (NDTs), augmented by machine learning, present a viable solution by creating virtual replicas of physical networks for real- time simulation and analysis. State-of-the-art models, however, fall short of full-fledged NDTs, as they often focus only on a single performance metric or simulated network data. We introduce M3Net, a Multi-Metric Mixture-of-experts (MoE) NDT that uses a graph neural network architecture to estimate multiple performance metrics from an expanded set of network state data in a range of scenarios. We show that M3Net significantly enhances the accuracy of flow delay predictions by reducing the MAPE (Mean Absolute Percentage Error) from 20.06% to 17.39%, while also achieving 66.47% and 78.7% accuracy on jitter and packets dropped for each flow

---

## 24. Physics-Aware Heterogeneous GNN Architecture for Real-Time BESS Optimization in Unbalanced Distribution Systems

**论文链接:** [http://arxiv.org/abs/2512.09780v1](http://arxiv.org/abs/2512.09780v1)

**作者:** Aoxiang Ma, Salah Ghamizi, Jun Cao, Pedro Rodriguez

**发布时间:** 2025-12-10

**备注:** 5 pages, 2 figures, 3 tables

### GPT解析

### 总结

该研究提出了一种基于异构图神经网络和物理信息损失函数的方法，用于三相不平衡配电网中的电池储能系统调度，通过嵌入详细的三相电网信息并整合电池约束条件，实现了高精度的网络状态变量预测和可靠的约束合规调度。

### 背景

电池储能系统在维持三相不平衡配电网电压稳定和实现最优调度方面变得越来越重要。然而，现有的深度学习方法通常缺乏明确的三相表示，难以准确建模特定相位的动态并执行操作约束，导致不可行的调度解决方案。

### 目的

开发一种能够准确建模三相不平衡配电网动态并确保操作约束合规的电池储能系统调度方法。

### 方法

将详细的三相电网信息（包括相电压、不平衡负载和BESS状态）嵌入到异构图节点中，应用多种GNN架构（GCN、GAT、GraphSAGE、GPS）联合预测网络状态变量，并使用物理信息损失函数在训练过程中通过软惩罚方式整合关键电池约束（SoC和C-rate限制）。

### 主要发现

在CIGRE 18-bus配电系统上的实验验证表明，嵌入-损失方法实现了低预测误差，各GNN架构的母线电压MSE分别为：GCN(6.92e-07)、GAT(1.21e-06)、GPS(3.29e-05)和SAGE(9.04e-07)；物理信息方法确保几乎为零的SoC和C-rate约束违反。

### 结论

通过嵌入详细的三相电网信息和使用物理信息损失函数，所提出的方法能够准确预测网络状态变量并确保操作约束合规，为三相不平衡配电网中的电池储能系统调度提供了可靠解决方案。

### 翻译

电池储能系统在维持三相不平衡配电网电压稳定和实现最优调度方面变得越来越重要。然而，现有的深度学习方法通常缺乏明确的三相表示，难以准确建模特定相位的动态并执行操作约束，导致不可行的调度解决方案。本文证明，通过将详细的三相电网信息（包括相电压、不平衡负载和BESS状态）嵌入到异构图节点中，多种GNN架构可以高精度地联合预测网络状态变量。此外，物理信息损失函数在训练过程中通过软惩罚方式整合了关键电池约束。在CIGRE 18-bus配电系统上的实验验证表明，这种嵌入-损失方法实现了低预测误差，各GNN架构的母线电压MSE分别为：GCN(6.92e-07)、GAT(1.21e-06)、GPS(3.29e-05)和SAGE(9.04e-07)。重要的是，物理信息方法确保几乎为零的SoC和C-rate约束违反，证实了其对可靠、约束合规调度的有效性。


### 论文摘要

Battery energy storage systems (BESS) have become increasingly vital in three-phase unbalanced distribution grids for maintaining voltage stability and enabling optimal dispatch. However, existing deep learning approaches often lack explicit three-phase representation, making it difficult to accurately model phase-specific dynamics and enforce operational constraints--leading to infeasible dispatch solutions. This paper demonstrates that by embedding detailed three-phase grid information--including phase voltages, unbalanced loads, and BESS states--into heterogeneous graph nodes, diverse GNN architectures (GCN, GAT, GraphSAGE, GPS) can jointly predict network state variables with high accuracy. Moreover, a physics-informed loss function incorporates critical battery constraints--SoC and C-rate limits--via soft penalties during training. Experimental validation on the CIGRE 18-bus distribution system shows that this embedding-loss approach achieves low prediction errors, with bus voltage MSEs of 6.92e-07 (GCN), 1.21e-06 (GAT), 3.29e-05 (GPS), and 9.04e-07 (SAGE). Importantly, the physics-informed method ensures nearly zero SoC and C-rate constraint violations, confirming its effectiveness for reliable, constraint-compliant dispatch.

---

## 25. Graph-Based Bayesian Optimization for Quantum Circuit Architecture Search with Uncertainty Calibrated Surrogates

**论文链接:** [http://arxiv.org/abs/2512.09586v1](http://arxiv.org/abs/2512.09586v1)

**作者:** Prashant Kumar Choudhary, Nouhaila Innan, Muhammad Shafique, Rajeev Singh

**发布时间:** 2025-12-10

**备注:** 17 pages, 13 figures

### GPT解析

### 总结

本文提出了一种自动化框架，用于发现和优化变分量子电路，使用基于图的贝叶斯优化结合图神经网络作为代理模型，在网络安全数据集上验证了其有效性。

### 背景

量子电路设计是实际量子机器学习在处理复杂、真实世界数据时面临的关键瓶颈。

### 目的

开发一个自动化框架，能够发现和优化变分量子电路(VQCs)，以解决量子机器学习中的电路设计难题。

### 方法

使用基于图的贝叶斯优化，结合图神经网络(GNN)作为代理模型；将电路表示为图，并通过基于代理不确定性的预期改进获取函数进行变异和选择；在下一代防火墙遥感和网络物联网(NF-ToN-IoT-V2)网络安全数据集上评估候选电路；使用混合量子的经典变分分类器；在量子嵌入前进行特征选择和缩放；与基于MLP的代理模型、随机搜索和贪婪GNN选择进行比较。

### 主要发现

GNN引导的优化器能够找到具有更低复杂度的电路；与所有基线相比，具有竞争性或更好的分类准确率；通过标准量子噪声通道（包括幅度阻尼、相位阻尼、热弛豫、退极化和读出位翻转噪声）评估了其鲁棒性。

### 结论

该实现是完全可复现的，包含时间基准测试和最佳发现电路的导出，为自动化量子电路发现提供了可扩展和可解释的途径。

### 翻译

量子电路设计是实际量子机器学习在复杂、真实世界数据上应用的关键瓶颈。我们提出了一种自动化框架，使用基于图的贝叶斯优化和图神经网络(GNN)代理模型来发现和优化变分量子电路(VQCs)。电路被表示为图，并通过基于代理不确定性的预期改进获取函数进行变异和选择，使用蒙特卡洛dropout。候选电路在经过特征选择和缩放以用于量子嵌入后，使用混合量子的经典变分分类器在下一代防火墙遥感和网络物联网(NF-ToN-IoT-V2)网络安全数据集上进行评估。我们将我们的流水线与基于MLP的代理模型、随机搜索和贪婪GNN选择进行比较。GNN引导的优化器始终能够找到比所有基线更低复杂度的电路，并且具有竞争性或更好的分类准确率。通过标准量子噪声通道（包括幅度阻尼、相位阻尼、热弛豫、退极化和读出位翻转噪声）进行了鲁棒性评估。该实现是完全可复现的，包含时间基准测试和最佳发现电路的导出，为自动化量子电路发现提供了可扩展和可解释的途径。


### 论文摘要

Quantum circuit design is a key bottleneck for practical quantum machine learning on complex, real-world data. We present an automated framework that discovers and refines variational quantum circuits (VQCs) using graph-based Bayesian optimization with a graph neural network (GNN) surrogate. Circuits are represented as graphs and mutated and selected via an expected improvement acquisition function informed by surrogate uncertainty with Monte Carlo dropout. Candidate circuits are evaluated with a hybrid quantum-classical variational classifier on the next generation firewall telemetry and network internet of things (NF-ToN-IoT-V2) cybersecurity dataset, after feature selection and scaling for quantum embedding. We benchmark our pipeline against an MLP-based surrogate, random search, and greedy GNN selection. The GNN-guided optimizer consistently finds circuits with lower complexity and competitive or superior classification accuracy compared to all baselines. Robustness is assessed via a noise study across standard quantum noise channels, including amplitude damping, phase damping, thermal relaxation, depolarizing, and readout bit flip noise. The implementation is fully reproducible, with time benchmarking and export of best found circuits, providing a scalable and interpretable route to automated quantum circuit discovery.

---

## 26. Advancing Text Classification with Large Language Models and Neural Attention Mechanisms

**论文链接:** [http://arxiv.org/abs/2512.09444v1](http://arxiv.org/abs/2512.09444v1)

**作者:** Ning Lyu, Yuxi Wang, Feng Chen, Qingyuan Zhang

**发布时间:** 2025-12-10

### GPT解析

### 总结

本研究提出了一种基于大型语言模型的文本分类算法，通过文本编码、上下文表示建模、基于注意力的增强、特征聚合和分类预测等框架，有效解决了传统方法在捕获长距离依赖、理解上下文语义和处理类别不平衡方面的局限性。

### 背景

传统文本分类方法在捕获长距离依赖、理解上下文语义和处理类别不平衡方面存在局限性。

### 目的

开发一种基于大型语言模型的文本分类算法，以解决传统方法的局限性，提高分类性能。

### 方法

提出了一个包含文本编码、上下文表示建模、基于注意力的增强、特征聚合和分类预测的框架。在表示阶段，通过大规模预训练语言模型获取深度语义嵌入，并应用注意力机制增强关键特征的选择性表示。在聚合阶段，结合全局和加权策略生成鲁棒的文本级向量。在分类阶段，使用全连接层和Softmax输出预测类别分布，采用交叉熵损失优化模型参数。

### 主要发现

与包括循环神经网络、图神经网络和Transformers在内的多个基线模型相比，所提出的方法在所有指标上都表现更好，特别是在召回率和AUC上有显著提升。超参数和数据条件的敏感性实验表明，适当的模型配置对性能有显著影响，模型在不同条件下显示出良好的适应性和稳定性。

### 结论

所提出的文本分类方法不仅实现了有效的性能提升，还通过系统分析验证了其在复杂数据环境下的鲁棒性和适用性。

### 翻译

本研究提出了一种基于大型语言模型的文本分类算法，旨在解决传统方法在捕获长距离依赖、理解上下文语义和处理类别不平衡方面的局限性。该框架包括文本编码、上下文表示建模、基于注意力的增强、特征聚合和分类预测。在表示阶段，通过大规模预训练语言模型获取深度语义嵌入，并应用注意力机制增强关键特征的选择性表示。在聚合阶段，结合全局和加权策略生成鲁棒的文本级向量。在分类阶段，使用全连接层和Softmax输出预测类别分布，采用交叉熵损失优化模型参数。比较实验引入了包括循环神经网络、图神经网络和Transformers在内的多个基线模型，并在精确率、召回率、F1分数和AUC上进行了评估。结果表明，所提出的方法在所有指标上都优于现有模型，特别是在召回率和AUC上有显著提升。此外，还对超参数和数据条件进行了敏感性实验，包括隐藏维度对AUC的影响以及类别不平衡比率对召回率的影响。研究结果证明，适当的模型配置对性能有显著影响，并揭示了模型在不同条件下的适应性和稳定性。总体而言，所提出的文本分类方法不仅实现了有效的性能提升，还通过系统分析验证了其在复杂数据环境下的鲁棒性和适用性。


### 论文摘要

This study proposes a text classification algorithm based on large language models, aiming to address the limitations of traditional methods in capturing long-range dependencies, understanding contextual semantics, and handling class imbalance. The framework includes text encoding, contextual representation modeling, attention-based enhancement, feature aggregation, and classification prediction. In the representation stage, deep semantic embeddings are obtained through large-scale pretrained language models, and attention mechanisms are applied to enhance the selective representation of key features. In the aggregation stage, global and weighted strategies are combined to generate robust text-level vectors. In the classification stage, a fully connected layer and Softmax output are used to predict class distributions, and cross-entropy loss is employed to optimize model parameters. Comparative experiments introduce multiple baseline models, including recurrent neural networks, graph neural networks, and Transformers, and evaluate them on Precision, Recall, F1-Score, and AUC. Results show that the proposed method outperforms existing models on all metrics, with especially strong improvements in Recall and AUC. In addition, sensitivity experiments are conducted on hyperparameters and data conditions, covering the impact of hidden dimensions on AUC and the impact of class imbalance ratios on Recall. The findings demonstrate that proper model configuration has a significant effect on performance and reveal the adaptability and stability of the model under different conditions. Overall, the proposed text classification method not only achieves effective performance improvement but also verifies its robustness and applicability in complex data environments through systematic analysis.

---

## 27. BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.09385v1](http://arxiv.org/abs/2512.09385v1)

**作者:** Uisang Lee, Changhoon Chung, Junmo Lee, Soo-Mook Moon

**发布时间:** 2025-12-10

**备注:** This paper is accepted to AAAI 2026

### GPT解析

### 总结

BugSweeper是一个端到端的深度学习框架，可以直接从源代码检测智能合约漏洞，无需人工规则设计。它使用函数级抽象语法图表示Solidity函数，并通过两阶段图神经网络进行分析，显著优于现有检测方法。

### 背景

以太坊的快速增长使得快速准确地检测智能合约漏洞变得更为重要。现有的基于机器学习的方法虽然有一定前景，但许多仍然依赖领域专家设计的基于规则的预处理方法。

### 目的

引入BugSweeper，一个端到端的深度学习框架，可以直接从源代码检测漏洞，无需人工工程。

### 方法

BugSweeper将每个Solidity函数表示为函数级抽象语法图，结合抽象语法树与增强的控制流和数据流语义。然后使用两阶段图神经网络分析：第一阶段过滤语法图中的噪声，第二阶段进行高级推理以检测各种漏洞。

### 主要发现

对现实世界合约的广泛实验表明，BugSweeper显著优于所有最先进的检测方法。

### 结论

通过消除手工制作规则的需要，BugSweeper提供了强大、自动化和可扩展的解决方案，用于保护智能合约，无需依赖安全专家。

### 翻译

以太坊的快速增长使得快速准确地检测智能合约漏洞变得更为重要。虽然基于机器学习的方法已经显示出一些前景，但许多仍然依赖领域专家设计的基于规则的预处理方法。基于规则的预处理方法通常丢弃源代码中的关键上下文，可能导致某些漏洞被忽略，并且对新出现的威胁适应性有限。我们引入了BugSweeper，一个端到端的深度学习框架，可以直接从源代码检测漏洞，无需人工工程。BugSweeper将每个Solidity函数表示为函数级抽象语法图，这是一种将抽象语法树与增强的控制流和数据流语义相结合的新图结构。然后，我们的两阶段图神经网络分析这些图。第一阶段的GNN过滤语法图中的噪声，而第二阶段的GNN进行高级推理以检测各种漏洞。对现实世界合约的广泛实验表明，BugSweeper显著优于所有最先进的检测方法。通过消除对手工规则的需求，我们的方法为保护智能合约提供了一个强大、自动化和可扩展的解决方案，无需任何安全专家的参与。


### 论文摘要

The rapid growth of Ethereum has made it more important to quickly and accurately detect smart contract vulnerabilities. While machine-learning-based methods have shown some promise, many still rely on rule-based preprocessing designed by domain experts. Rule-based preprocessing methods often discard crucial context from the source code, potentially causing certain vulnerabilities to be overlooked and limiting adaptability to newly emerging threats. We introduce BugSweeper, an end-to-end deep learning framework that detects vulnerabilities directly from the source code without manual engineering. BugSweeper represents each Solidity function as a Function-Level Abstract Syntax Graph (FLAG), a novel graph that combines its Abstract Syntax Tree (AST) with enriched control-flow and data-flow semantics. Then, our two-stage Graph Neural Network (GNN) analyzes these graphs. The first-stage GNN filters noise from the syntax graphs, while the second-stage GNN conducts high-level reasoning to detect diverse vulnerabilities. Extensive experiments on real-world contracts show that BugSweeper significantly outperforms all state-of-the-art detection methods. By removing the need for handcrafted rules, our approach offers a robust, automated, and scalable solution for securing smart contracts without any dependence on security experts.

---

## 28. Branching Strategies Based on Subgraph GNNs: A Study on Theoretical Promise versus Practical Reality

**论文链接:** [http://arxiv.org/abs/2512.09355v1](http://arxiv.org/abs/2512.09355v1)

**作者:** Junru Zhou, Yicheng Wang, Pan Li

**发布时间:** 2025-12-10

### GPT解析

### 总结

这篇论文研究了子图GNNs作为MILP中'学习分支'的理论中间地带，发现虽然理论上它们能提供更好的分支决策，但在实践中计算成本过高，表明未来研究应关注效率与表达能力的平衡。

### 背景

图神经网络(GNNs)已成为混合整数线性规划(MILP)中'学习分支'的一种有前景的方法。标准消息传递GNNs(MPNNs)效率高但表达力有限，而高阶GNNs表达能力强但计算成本过高。

### 目的

研究子图GNNs作为理论上的中间地带，探索其在MILP分支决策中的应用潜力。

### 方法

证明锚定节点的子图GNNs(表达能力低于3-WL)足以近似强分支分数，并在四个基准数据集上进行实证评估，比较其与MPNNs和启发式方法的性能。

### 主要发现

锚定节点的子图GNNs理论上能提供更好的分支决策，但其O(n)的复杂度开销导致严重的内存瓶颈和更慢的求解时间。计算成本超过了决策质量上的收益。

### 结论

对于MILP分支，当前表达性GNNs的计算成本超过了其在决策质量上的收益，未来研究必须关注效率与表达能力的平衡。

### 翻译

图神经网络(GNNs)已成为混合整数线性规划(MILP)中'学习分支'的一种有前景的方法。虽然标准消息传递GNNs(MPNNs)效率高，但理论上缺乏充分表示MILP结构的能力。相反，高阶GNNs(如2-FGNNs)表达能力强，但计算上过于昂贵。在本工作中，我们研究子图GNNs作为理论上的中间地带。关键的是，虽然先前工作[Chen等人，2025]证明了具有3-WL表达能力的GNNs可以近似强分支，但我们证明了更精确的结果：表达能力严格低于3-WL[Zhang等人，2023]的锚定节点子图GNNs足以近似强分支分数。然而，我们在四个基准数据集上的广泛实证评估揭示了理论与实际之间的鲜明对比。虽然理论上锚定节点的子图GNNs能提供更好的分支决策，但其O(n)的复杂度开销导致比MPNNs和启发式方法更严重的内存瓶颈和更慢的求解时间。我们的结果表明，对于MILP分支，当前表达性GNNs的计算成本超过了其在决策质量上的收益，表明未来研究必须关注保持效率的表达能力。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as a promising approach for ``learning to branch'' in Mixed-Integer Linear Programming (MILP). While standard Message-Passing GNNs (MPNNs) are efficient, they theoretically lack the expressive power to fully represent MILP structures. Conversely, higher-order GNNs (like 2-FGNNs) are expressive but computationally prohibitive. In this work, we investigate Subgraph GNNs as a theoretical middle ground. Crucially, while previous work [Chen et al., 2025] demonstrated that GNNs with 3-WL expressive power can approximate Strong Branching, we prove a sharper result: node-anchored Subgraph GNNs whose expressive power is strictly lower than 3-WL [Zhang et al., 2023] are sufficient to approximate Strong Branching scores. However, our extensive empirical evaluation on four benchmark datasets reveals a stark contrast between theory and practice. While node-anchored Subgraph GNNs theoretically offer superior branching decisions, their $O(n)$ complexity overhead results in significant memory bottlenecks and slower solving times than MPNNs and heuristics. Our results indicate that for MILP branching, the computational cost of expressive GNNs currently outweighs their gains in decision quality, suggesting that future research must focus on efficiency-preserving expressivity.

---

## 29. Understanding the Failure Modes of Transformers through the Lens of Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.09182v1](http://arxiv.org/abs/2512.09182v1)

**作者:** Hunjae Lee

**发布时间:** 2025-12-09

### GPT解析

### 总结

本研究从图神经网络理论视角分析了Transformer（特别是仅解码器Transformer）的失败模式，揭示了其与GNN在信息传播瓶颈方面的相似性，并尝试将现有解决方案统一在更理论化的框架下。

### 背景

Transformer架构主导现代大型语言模型，尽管表现优异但仍存在问题和失败模式，导致意外的失败模式和可预测的不对称性能下降。

### 目的

通过图神经网络理论研究Transformer的多种失败模式，弥合观察到的失败模式与该领域普遍缺乏的理论理解之间的差距。

### 方法

将深度学习和Transformer视为可学习信息混合和传播的机制，将模型失败模式研究视为信息传播瓶颈研究，应用GNN理论框架分析Transformer的因果性质及其在信息传播中创造的几何特性。

### 主要发现

许多GNN面临的问题同样存在于Transformer中；仅解码器Transformer的因果性质导致信息传播中可预测且可能具有破坏性的失败模式；现有的Transformer解决方案往往是临时性的，由直觉驱动而非基于理论动机。

### 结论

本文尝试将Transformer中观察到的失败模式与该领域普遍缺乏的理论理解联系起来，提供对解决方案有效原因、实际解决的问题以及如何进一步改进以针对特定Transformer失败模式的见解。

### 翻译

Transformer架构，更具体地说是仅解码器Transformer，主导着现代大型语言模型架构。尽管它们已被证明工作得异常好，但并非没有问题，导致了意外的失败模式和可预测的不对称性能下降。本文通过图神经网络理论视角研究了这些观察到的Transformer失败模式。我们首先论证了深度学习的大部分内容，包括Transformer，是关于可学习的信息混合和传播。这使得模型失败模式研究成为信息传播瓶颈的研究。这自然引向了GNN理论，其中已有关于信息传播瓶颈和模型理论失败模式的丰富文献。然后我们论证了许多GNN面临的问题同样存在于Transformer中。此外，我们分析了仅解码器Transformer的因果性质如何在信息传播中创造有趣的几何特性，导致可预测且可能具有破坏性的失败模式。最后，我们观察到Transformer研究中现有的解决方案往往是临时性的，由直觉驱动而非基于理论动机。因此，我们在更理论化的视角下统一了许多这样的解决方案，提供了为什么它们有效、它们实际解决了什么问题以及如何进一步改进以针对特定Transformer失败模式的见解。总体而言，本文试图弥合Transformer中观察到的失败模式与该领域普遍缺乏的理论理解之间的差距。


### 论文摘要

Transformers and more specifically decoder-only transformers dominate modern LLM architectures. While they have shown to work exceptionally well, they are not without issues, resulting in surprising failure modes and predictably asymmetric performance degradation. This article is a study of many of these observed failure modes of transformers through the lens of graph neural network (GNN) theory. We first make the case that much of deep learning, including transformers, is about learnable information mixing and propagation. This makes the study of model failure modes a study of bottlenecks in information propagation. This naturally leads to GNN theory, where there is already a rich literature on information propagation bottlenecks and theoretical failure modes of models. We then make the case that many issues faced by GNNs are also experienced by transformers. In addition, we analyze how the causal nature of decoder-only transformers create interesting geometric properties in information propagation, resulting in predictable and potentially devastating failure modes. Finally, we observe that existing solutions in transformer research tend to be ad-hoc and driven by intuition rather than grounded theoretical motivation. As such, we unify many such solutions under a more theoretical perspective, providing insight into why they work, what problem they are actually solving, and how they can be further improved to target specific failure modes of transformers. Overall, this article is an attempt to bridge the gap between observed failure modes in transformers and a general lack of theoretical understanding of them in this space.

---

## 30. AI-Driven Expansion and Application of the Alexandria Database

**论文链接:** [http://arxiv.org/abs/2512.09169v1](http://arxiv.org/abs/2512.09169v1)

**作者:** Théo Cavignac, Jonathan Schmidt, Pierre-Paul De Breuck, Antoine Loew, Tiago F. T. Cerqueira, Hai-Chen Wang, Anton Bochkarev, Yury Lysogorskiy, Aldo H. Romero, Ralf Drautz, Silvana Botti, Miguel A. L. Marques

**发布时间:** 2025-12-09

### GPT解析

### 总结

提出了一种新型多阶段计算材料发现工作流程，在识别热力学稳定性化合物方面达到99%成功率，比之前方法提高三倍

### 背景

计算材料发现领域需要更高效的方法来识别稳定材料

### 目的

开发一种高成功率的多阶段工作流程，用于计算材料发现

### 方法

结合Matra-Genoa生成模型、Orb-v2通用机器学习原子间势和ALIGNN图神经网络进行能量预测，生成候选结构并用DFT验证

### 主要发现

工作流程成功识别出130万个DFT验证的化合物，扩展ALEXANDRIA数据库至580万个结构，预测的结构无序率与实验数据库匹配，揭示了空间群分布、配位环境和相稳定性网络中的基本模式

### 结论

该工作流程显著提高了材料发现的效率和准确性，发布的完整数据集和模型为材料科学领域提供了有价值的资源

### 翻译

我们提出了一种新的多阶段计算材料发现工作流程，在识别热力学稳定性在100 meV/原子以内的化合物方面达到了99%的成功率，比之前的方法提高了三倍。通过结合Matra-Genoa生成模型、Orb-v2通用机器学习原子间势和ALIGNN图神经网络进行能量预测，我们生成了1.19亿个候选结构，并将130万个DFT验证的化合物添加到ALEXANDRIA数据库中，包括7.4万个新的稳定材料。扩展后的ALEXANDRIA数据库现在包含580万个结构，其中17.5万个化合物位于凸包上。预测的结构无序率（37-43%）与实验数据库匹配，与其他最近的人工智能生成数据集不同。分析揭示了空间群分布、配位环境和相稳定性网络中的基本模式，包括凸包连接性的次线性标度。我们发布了完整的数据集，包括包含力和应力的sAlex25，包含1400万个非平衡结构，用于训练通用力场。我们证明在此数据上微调GRACE模型可以提高基准测试准确性。所有数据、模型和工作流程都根据知识共享许可免费提供。


### 论文摘要

We present a novel multi-stage workflow for computational materials discovery that achieves a 99% success rate in identifying compounds within 100 meV/atom of thermodynamic stability, with a threefold improvement over previous approaches. By combining the Matra-Genoa generative model, Orb-v2 universal machine learning interatomic potential, and ALIGNN graph neural network for energy prediction, we generated 119 million candidate structures and added 1.3 million DFT-validated compounds to the ALEXANDRIA database, including 74 thousand new stable materials. The expanded ALEXANDRIA database now contains 5.8 million structures with 175 thousand compounds on the convex hull. Predicted structural disorder rates (37-43%) match experimental databases, unlike other recent AI-generated datasets. Analysis reveals fundamental patterns in space group distributions, coordination environments, and phase stability networks, including sub-linear scaling of convex hull connectivity. We release the complete dataset, including sAlex25 with 14 million out-of-equilibrium structures containing forces and stresses for training universal force fields. We demonstrate that fine-tuning a GRACE model on this data improves benchmark accuracy. All data, models, and workflows are freely available under Creative Commons licenses.

---

## 31. Graph Deep Learning for Intracranial Aneurysm Blood Flow Simulation and Risk Assessment

**论文链接:** [http://arxiv.org/abs/2512.09013v1](http://arxiv.org/abs/2512.09013v1)

**作者:** Paul Garnier, Pablo Jeken-Rico, Vincent Lannelongue, Chiara Faitini, Aurèle Goetz, Lea Chanvillard, Ramy Nemer, Jonathan Viquerat, Ugo Pelissier, Philippe Meliga, Jacques Sédat, Thomas Liebig, Yves Chau, Elie Hachem

**发布时间:** 2025-12-09

### GPT解析

### 总结

该研究提出了一种图神经网络替代模型，可以从血管几何结构直接重现全场血流动力学，每个心脏周期不到一分钟，无需网格特定校准，能够推广到未见过的患者几何形状和流入条件。

### 背景

颅内动脉瘤是全球范围内神经疾病发病率和死亡率的主要原因，其破裂风险与局部血流动力学特性（特别是壁剪切应力和振荡剪切指数）密切相关。传统计算流体动力学模拟准确但速度慢且需要专业知识，而临床成像替代方案如4D Flow MRI虽然可以直接进行体内测量，但空间分辨率不足且非常昂贵。

### 目的

开发一种能够弥合传统CFD模拟和临床成像之间差距的替代模型，实现从血管几何结构直接重现全场血流动力学，并在临床实践中实现近实时分析。

### 方法

提出了一种图神经网络替代模型，使用患者特异性动脉瘤的高保真模拟数据集进行训练，架构结合了图变换器和自回归预测，能够模拟血流、壁剪切应力和振荡剪切指数。

### 主要发现

该模型能够在不到一分钟的时间内从血管几何结构重现全场血流动力学，准确模拟血流动力学参数，并能在未见过的患者几何形状和流入条件下泛化，无需网格特定校准。

### 结论

该研究将高保真模拟从专家专属研究工具转变为可部署的数据驱动决策支持系统，整个流程在患者成像后几分钟内提供高分辨率血流动力学预测，无需计算专家参与，标志着向实时、床边动脉瘤分析迈出了重要一步。

### 翻译

颅内动脉瘤仍然是全球范围内神经疾病发病率和死亡率的主要原因，其破裂风险与局部血流动力学特性，特别是壁剪切应力和振荡剪切指数密切相关。传统的计算流体动力学模拟提供了准确的见解，但速度极慢且需要专业知识。临床成像替代方案如4D Flow MRI提供了直接的体内测量，但其空间分辨率仍然不足以捕捉驱动内皮重塑和破裂风险的精细剪切模式，同时极其不切实际且昂贵。我们提出了一种图神经网络替代模型，通过从血管几何结构直接重现全场血流动力学，每个心脏周期不到一分钟，弥合了这一差距。该模型在患者特异性动脉瘤的高保真模拟数据集上进行训练，我们的架构结合了图变换器和自回归预测，能够准确模拟血流、壁剪切应力和振荡剪切指数。该模型可以推广到未见过的患者几何形状和流入条件，无需网格特定校准。除了加速模拟外，我们的框架为临床可解释的血流动力学预测奠定了基础。通过实现与现有成像流程集成的近实时推理，它允许直接与医院相图评估进行比较，并通过物理基础的高分辨率流场扩展它们。这项工作将高保真模拟从专家专属研究工具转变为可部署的数据驱动决策支持系统。我们的完整流程在患者成像后几分钟内提供高分辨率血流动力学预测，无需计算专家参与，标志着向实时、床边动脉瘤分析迈出了重要一步。


### 论文摘要

Intracranial aneurysms remain a major cause of neurological morbidity and mortality worldwide, where rupture risk is tightly coupled to local hemodynamics particularly wall shear stress and oscillatory shear index. Conventional computational fluid dynamics simulations provide accurate insights but are prohibitively slow and require specialized expertise. Clinical imaging alternatives such as 4D Flow MRI offer direct in-vivo measurements, yet their spatial resolution remains insufficient to capture the fine-scale shear patterns that drive endothelial remodeling and rupture risk while being extremely impractical and expensive.   We present a graph neural network surrogate model that bridges this gap by reproducing full-field hemodynamics directly from vascular geometries in less than one minute per cardiac cycle. Trained on a comprehensive dataset of high-fidelity simulations of patient-specific aneurysms, our architecture combines graph transformers with autoregressive predictions to accurately simulate blood flow, wall shear stress, and oscillatory shear index. The model generalizes across unseen patient geometries and inflow conditions without mesh-specific calibration. Beyond accelerating simulation, our framework establishes the foundation for clinically interpretable hemodynamic prediction. By enabling near real-time inference integrated with existing imaging pipelines, it allows direct comparison with hospital phase-diagram assessments and extends them with physically grounded, high-resolution flow fields.   This work transforms high-fidelity simulations from an expert-only research tool into a deployable, data-driven decision support system. Our full pipeline delivers high-resolution hemodynamic predictions within minutes of patient imaging, without requiring computational specialists, marking a step-change toward real-time, bedside aneurysm analysis.

---

## 32. Solving Oversmoothing in GNNs via Nonlocal Message Passing: Algebraic Smoothing and Depth Scalability

**论文链接:** [http://arxiv.org/abs/2512.08475v2](http://arxiv.org/abs/2512.08475v2)

**作者:** Weiqi Guan, Junlin He

**发布时间:** 2025-12-09

**备注:** 18 pages, 4 figures

### GPT解析

### 总结

本研究探讨了Layer Normalization (LN)放置与过平滑现象之间的关系，发现Pre-LN架构避免过平滑但受深度诅咒影响，而Post-LN架构绕过深度诅咒但经历过平滑。提出了一种基于Post-LN的新方法，通过诱导代数平滑来同时避免这两种问题，无需额外参数即可支持更深的网络并提高性能。

### 背景

Layer Normalization (LN)的放置与过平滑现象之间的关系尚未被充分探索。存在一个关键的两难困境：Pre-LN架构避免了过平滑，但受到深度诅咒的影响；而Post-LN架构绕过了深度诅咒，但经历了过平滑问题。

### 目的

解决LN放置与过平滑现象之间的两难困境，提出一种基于Post-LN的新方法，诱导代数平滑，避免过平滑和深度诅咒。

### 方法

提出一种基于Post-LN的新方法，该方法诱导代数平滑，不需要额外参数。

### 主要发现

在五个基准测试中的实证结果表明，该方法支持更深的网络（最多256层）并提高性能，不需要额外参数。

### 结论

该方法通过诱导代数平滑，解决了LN放置与过平滑现象之间的两难困境，同时避免了过平滑和深度诅咒。

### 翻译

Layer Normalization (LN)放置与过平滑现象之间的关系仍未被充分探索。我们确定了一个关键的两难困境：Pre-LN架构避免了过平滑，但受到深度诅咒的影响，而Post-LN架构绕过了深度诅咒，但经历了过平滑。为解决这一问题，我们提出了一种基于Post-LN的新方法，它诱导代数平滑，避免了过平滑且不受深度诅咒影响。在五个基准测试中的实证结果表明，我们的方法支持更深的网络（最多256层）并提高性能，无需额外参数。主要贡献：理论表征：分析LN动力学及其对过平滑和深度诅咒的影响。有原则的解决方案：一种参数高效的方法，诱导代数平滑并避免过平滑和深度诅咒。经验验证：广泛的实验展示了该方法在更深GNN中的有效性。


### 论文摘要

The relationship between Layer Normalization (LN) placement and the oversmoothing phenomenon remains underexplored. We identify a critical dilemma: Pre-LN architectures avoid oversmoothing but suffer from the curse of depth, while Post-LN architectures bypass the curse of depth but experience oversmoothing.   To resolve this, we propose a new method based on Post-LN that induces algebraic smoothing, preventing oversmoothing without the curse of depth. Empirical results across five benchmarks demonstrate that our approach supports deeper networks (up to 256 layers) and improves performance, requiring no additional parameters.   Key contributions:   Theoretical Characterization: Analysis of LN dynamics and their impact on oversmoothing and the curse of depth.   A Principled Solution: A parameter-efficient method that induces algebraic smoothing and avoids oversmoothing and the curse of depth.   Empirical Validation: Extensive experiments showing the effectiveness of the method in deeper GNNs.

---

## 33. HPM-KD: Hierarchical Progressive Multi-Teacher Framework for Knowledge Distillation and Efficient Model Compression

**论文链接:** [http://arxiv.org/abs/2512.09886v1](http://arxiv.org/abs/2512.09886v1)

**作者:** Gustavo Coelho Haase, Paulo Henrique Dourado da Silva

**发布时间:** 2025-12-10

**备注:** 9 pages

### GPT解析

### 总结

本文提出了一种名为HPM-KD的知识蒸馏框架，解决了传统知识蒸馏技术的四个主要局限性：超参数敏感性、大教师到小学生的能力差距、多教师场景下的次优协调以及计算资源利用效率低下。HPM-KD集成了六个协同组件，实现了自动化超参数调优、渐进式蒸馏链、注意力加权的多教师集成、元学习温度调度、并行处理流水线和共享优化内存。实验表明，HPM-KD能在保持85%准确率的同时实现10-15倍的模型压缩，消除手动调参需求，并通过并行化减少30-40%的训练时间。

### 背景

知识蒸馏(KD)已成为一种有前景的模型压缩技术，但面临四个关键局限性：(1)对超参数敏感，需要大量手动调优；(2)从非常大的教师模型蒸馏到小型学生模型时存在能力差距；(3)多教师场景下的协调次优；(4)计算资源利用效率低下。

### 目的

解决传统知识蒸馏技术的四个主要局限性，提高模型压缩效率并减少人工干预。

### 方法

提出HPM-KD框架，包含六个协同组件：(i)基于元学习的自适应配置管理器，消除手动超参数调优；(ii)具有自动确定中间模型的渐进式蒸馏链；(iii)学习动态每样本权重的注意力加权多教师集成；(iv)在整个训练过程中自适应调整温度的元学习温度调度器；(v)具有智能负载平衡的并行处理流水线；(vi)用于跨实验重用的共享优化内存。

### 主要发现

在CIFAR-10、CIFAR-100和表格数据集上的实验表明，HPM-KD能在保持85%准确率的同时实现10-15倍的模型压缩，消除手动调优需求，并通过并行化减少30-40%的训练时间。消融研究确认了每个组件的独立贡献(0.10-0.98个百分点)。

### 结论

HPM-KD框架有效解决了传统知识蒸馏技术的关键局限性，显著提高了模型压缩效率和易用性，同时保持了模型性能。该框架已作为开源DeepBridge库的一部分提供。

### 翻译

知识蒸馏(KD)已成为一种有前景的模型压缩技术，但面临关键局限性：(1)对超参数敏感，需要大量手动调优；(2)从非常大的教师模型蒸馏到小型学生模型时存在能力差距；(3)多教师场景下的协调次优；(4)计算资源利用效率低下。我们提出HPM-KD框架，集成了六个协同组件：(i)通过元学习的自适应配置管理器，消除手动超参数调优；(ii)具有自动确定中间模型的渐进式蒸馏链；(iii)学习动态每样本权重的注意力加权多教师集成；(iv)在整个训练过程中自适应调整温度的元学习温度调度器；(v)具有智能负载平衡的并行处理流水线；(vi)用于跨实验重用的共享优化内存。在CIFAR-10、CIFAR-100和表格数据集上的实验表明，HPM-KD在保持85%准确率的同时实现10-15倍的模型压缩，消除手动调优需求，并通过并行化减少30-40%的训练时间。消融研究确认了每个组件的独立贡献(0.10-0.98个百分点)。HPM-KD作为开源DeepBridge库的一部分提供。


### 论文摘要

Knowledge Distillation (KD) has emerged as a promising technique for model compression but faces critical limitations: (1) sensitivity to hyperparameters requiring extensive manual tuning, (2) capacity gap when distilling from very large teachers to small students, (3) suboptimal coordination in multi-teacher scenarios, and (4) inefficient use of computational resources. We present \textbf{HPM-KD}, a framework that integrates six synergistic components: (i) Adaptive Configuration Manager via meta-learning that eliminates manual hyperparameter tuning, (ii) Progressive Distillation Chain with automatically determined intermediate models, (iii) Attention-Weighted Multi-Teacher Ensemble that learns dynamic per-sample weights, (iv) Meta-Learned Temperature Scheduler that adapts temperature throughout training, (v) Parallel Processing Pipeline with intelligent load balancing, and (vi) Shared Optimization Memory for cross-experiment reuse. Experiments on CIFAR-10, CIFAR-100, and tabular datasets demonstrate that HPM-KD: achieves 10x-15x compression while maintaining 85% accuracy retention, eliminates the need for manual tuning, and reduces training time by 30-40% via parallelization. Ablation studies confirm independent contribution of each component (0.10-0.98 pp). HPM-KD is available as part of the open-source DeepBridge library.

---

## 34. PathCo-LatticE: Pathology-Constrained Lattice-Of Experts Framework for Fully-supervised Few-Shot Cardiac MRI Segmentation

**论文链接:** [http://arxiv.org/abs/2512.09779v1](http://arxiv.org/abs/2512.09779v1)

**作者:** Mohamed Elbayumi, Mohammed S. M. Elbaz

**发布时间:** 2025-12-10

### GPT解析

### 总结

PathCo-LatticE是一种完全监督的少样本学习框架，通过病理引导的合成监督替代未标记数据，在心脏MRI分割任务中实现了强大的零样本泛化能力，无需目标域微调即可优于现有方法。

### 背景

少样本学习可缓解心脏MRI分割中的数据稀缺问题，但传统方法依赖半监督技术，对域移位和验证偏差敏感，限制了零样本泛化能力。

### 目的

提出PathCo-LatticE框架，用病理引导的合成监督替代未标记数据，实现完全监督的少样本学习，提高零样本泛化能力。

### 方法

1) 虚拟患者引擎：从稀疏临床锚点建模连续疾病轨迹，合成生理合理的3D队列数据；2) 自强化交错验证：使用渐进式挑战性合成样本在线评估模型，消除对真实验证数据的需求；3) 动态专家网格：在病理感知拓扑中组织专业网络，根据输入激活最相关专家，实现零样本泛化。

### 主要发现

PathCo-LatticE优于四种最先进FSL方法4.2-11% Dice分数；仅从7个标记锚点开始性能优异；仅用19个标记锚点就接近全监督性能；在四个供应商间表现出优越的调和能力；可泛化到未见过的病理。

### 结论

PathCo-LatticE通过病理引导的合成监督替代未标记数据，实现了完全监督的少样本学习，在心脏MRI分割任务中提供了强大的零样本泛化能力，代码将公开可用。

### 翻译

少样本学习(FSL)缓解了心脏MRI分割中的数据稀缺问题，但通常依赖对域移位和验证偏差敏感的半监督技术，限制了零样本泛化能力。我们提出了PathCo-LatticE，一种完全监督的FSL框架，用病理引导的合成监督替代未标记数据。首先，我们的虚拟患者引擎从稀疏临床锚点建模连续的潜在疾病轨迹，使用生成模型合成生理上合理的、完全标记的3D队列。其次，自强化交错验证(SIV)提供无泄漏协议，使用渐进式具有挑战性的合成样本在线评估模型，消除对真实验证数据的需求。最后，动态专家网格(LoE)在病理感知拓扑中组织专业网络，并根据输入激活最相关的专家，实现强大的零样本泛化能力，无需目标域微调。我们在严格的分布外(OOD)设置中评估了PathCo-LatticE，所有锚点和严重程度统计信息来自单一源域(ACDC)，并在多中心、多供应商的M&Ms数据集上进行零样本测试。PathCo-LatticE优于四种最先进的FSL方法4.2-11% Dice分数，仅从7个标记锚点开始，仅用19个标记锚点就接近全监督性能(差距在1% Dice以内)。该方法在四个供应商间表现出优越的调和能力，并可泛化到未见过的病理。[代码将公开提供]。


### 论文摘要

Few-shot learning (FSL) mitigates data scarcity in cardiac MRI segmentation but typically relies on semi-supervised techniques sensitive to domain shifts and validation bias, restricting zero-shot generalizability. We propose PathCo-LatticE, a fully supervised FSL framework that replaces unlabeled data with pathology-guided synthetic supervision. First, our Virtual Patient Engine models continuous latent disease trajectories from sparse clinical anchors, using generative modeling to synthesize physiologically plausible, fully labeled 3D cohorts. Second, Self-Reinforcing Interleaved Validation (SIV) provides a leakage-free protocol that evaluates models online with progressively challenging synthetic samples, eliminating the need for real validation data. Finally, a dynamic Lattice-of-Experts (LoE) organizes specialized networks within a pathology-aware topology and activates the most relevant experts per input, enabling robust zero-shot generalization to unseen data without target-domain fine-tuning. We evaluated PathCo-LatticE in a strict out-of-distribution (OOD) setting, deriving all anchors and severity statistics from a single-source domain (ACDC) and performing zero-shot testing on the multi-center, multi-vendor M&Ms dataset. PathCo-LatticE outperforms four state-of-the-art FSL methods by 4.2-11% Dice starting from only 7 labeled anchors, and approaches fully supervised performance (within 1% Dice) with only 19 labeled anchors. The method shows superior harmonization across four vendors and generalization to unseen pathologies. [Code will be made publicly available].

---

## 35. Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback

**论文链接:** [http://arxiv.org/abs/2512.09366v1](http://arxiv.org/abs/2512.09366v1)

**作者:** Dimitra Maoutsa

**发布时间:** 2025-12-10

**备注:** 10 pages, 2 figures; accepted & presented at NeurIPS 2025 workshop Symmetry and Geometry in Neural Representations (NeurReps)

### GPT解析

### 总结

本研究提出了一种元学习框架，用于发现循环网络中结构化信用分配的局部学习规则，这些网络使用稀疏反馈进行训练。

### 背景

生物神经网络使用局部突触可塑性从稀疏、延迟的反馈中学习复杂行为，但支持结构化信用分配的机制仍然不清楚。相比之下，解决类似任务的人工循环网络通常依赖于生物上不合理的学习规则或手工制作的局部更新。

### 目的

探索能够支持从延迟强化学习中学习的局部可塑性规则空间，以实现更接近生物机制的神经网络学习。

### 方法

提出一个元学习框架，在任务执行期间交替进行局部新赫布式样更新，并通过学习进行切线传播的外部循环来优化可塑性参数。

### 主要发现

得到的三因子学习规则仅使用局部信息和延迟奖励就能实现长时间尺度的信用分配，为循环电路学习提供了基于生物学的新见解。

### 结论

这种方法为理解生物神经网络如何处理延迟奖励提供了新的机制理解，有助于开发更接近生物特性的学习算法。

### 翻译

生物神经网络使用局部突触可塑性从稀疏、延迟的反馈中学习复杂行为，但支持结构化信用分配的机制仍然不清楚。相比之下，解决类似任务的人工循环网络通常依赖于生物上不合理的学习规则或手工制作的局部更新。能够支持从延迟强化学习中学习的局部可塑性规则空间在很大程度上仍未被探索。在这里，我们提出了一个元学习框架，用于发现使用稀疏反馈训练的循环网络中结构化信用分配的局部学习规则。我们的方法在任务执行期间交替进行局部新赫布式样更新，并通过学习进行切线传播的外部循环来优化可塑性参数。 resulting three-factor learning rules enable long-timescale credit assignment using only local information and delayed rewards, offering new insights into biologically grounded mechanisms for learning in recurrent circuits.


### 论文摘要

Biological neural networks learn complex behaviors from sparse, delayed feedback using local synaptic plasticity, yet the mechanisms enabling structured credit assignment remain elusive. In contrast, artificial recurrent networks solving similar tasks typically rely on biologically implausible global learning rules or hand-crafted local updates. The space of local plasticity rules capable of supporting learning from delayed reinforcement remains largely unexplored. Here, we present a meta-learning framework that discovers local learning rules for structured credit assignment in recurrent networks trained with sparse feedback. Our approach interleaves local neo-Hebbian-like updates during task execution with an outer loop that optimizes plasticity parameters via \textbf{tangent-propagation through learning}. The resulting three-factor learning rules enable long-timescale credit assignment using only local information and delayed rewards, offering new insights into biologically grounded mechanisms for learning in recurrent circuits.

---

## 36. 论文ID: 2512.08606v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.08606v2.json'

---

## 37. LISN: Language-Instructed Social Navigation with VLM-based Controller Modulating

**论文链接:** [http://arxiv.org/abs/2512.09920v1](http://arxiv.org/abs/2512.09920v1)

**作者:** Junting Chen, Yunchuan Li, Panfeng Jiang, Jiacheng Du, Zixuan Chen, Chenrui Tie, Jiajun Deng, Lin Shao

**发布时间:** 2025-12-10

**备注:** 8 pages

### GPT解析

### 总结

本文提出了LISN-Bench，第一个基于模拟的语言指令社交导航基准测试，以及Social-Nav-Modulator系统，用于解决移动机器人在社交环境中导航时遵循人类指令的问题。

### 背景

为了实现人机共存，社交感知导航对移动机器人至关重要。然而现有研究主要关注路径效率和行人碰撞避免，这些只是社交导航的基本要素，机器人还需遵循用户指令，对齐任务目标和社会规范。

### 目的

开发首个包含指令遵循和场景理解的标准化社交导航基准测试，并提出一种能够理解并执行人类指令的社交导航系统。

### 方法

提出Social-Nav-Modulator，一个快速-慢速分层系统，其中视觉语言模型代理调整成本图和控制器参数。该方法将底层动作生成与较慢的VLM循环解耦，减少对高频VLM推理的依赖，同时提高动态避免和感知适应性。

### 主要发现

该方法平均成功率达到91.3%，比最具竞争力的基线高出63%，在人群中跟随人和导航时严格避免指令禁止区域等挑战性任务中表现尤为突出。

### 结论

LISN-Bench为社交导航研究提供了新的基准，Social-Nav-Modulator系统能有效处理复杂的社交导航场景，显著提高了机器人在社交环境中的导航能力。

### 翻译

为实现人机共存，社交感知导航对移动机器人至关重要。然而，该领域现有研究主要关注路径效率和行人碰撞避免，这些虽然必要但仅代表社交导航的一部分。除了这些基础外，机器人还必须遵守用户指令，使其行动与人类表达的任务目标和社会规范保持一致。在这项工作中，我们提出了LISN-Bench，这是第一个基于模拟的语言指令社交导航基准测试。基于Rosnav-Arena 3.0构建，它是首个包含跨多样化场景的指令遵循和场景理解的标准化社交导航基准。为解决这一任务，我们进一步提出了Social-Nav-Modulator，一个快速-慢速分层系统，其中VLM代理调整成本图和控制器参数。将底层动作生成与较慢的VLM循环解耦，减少对高频VLM推理的依赖，同时提高动态避免和感知适应性。我们的方法平均成功率达到91.3%，比最具竞争力的基线高出63%，大部分改进出现在具有挑战性的任务中，如人群中跟随人和导航时严格避免指令禁止区域。项目网站位于：https://social-nav.github.io/LISN-project/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有社会导航研究只关注路径效率和碰撞避免，而忽视了对人类语言指令的理解和遵守的问题。这个问题很重要，因为在真实人机共存环境中，机器人不仅需要安全导航，还需要理解并执行人类的具体指令，遵守社会规范，比如在医院环境中跟随特定人员、避开特定区域等，这对机器人真正融入人类社会至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有社会导航基准测试的局限性，缺乏对指令遵循和场景理解的评估。同时发现大型视觉语言模型(VLM)虽然能理解多模态信息，但推理速度慢，难以满足实时控制需求。作者借鉴了SFW-SAC工作，利用VLM动态调整规划器参数，同时保持低级控制系统的快速反应能力。这种设计既利用了VLM的高级语义理解能力，又确保了系统的实时性和安全性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个快速-慢速分层系统，将基于VLM的高级语义推理与低级反应控制解耦。慢速系统使用VLM分析视觉和语言输入，通过调用预定义工具生成参数和视觉标记；快速系统基于社会力模型(SFM)和动态社会成本图层，利用慢速系统生成的参数实时生成控制指令。这种设计允许机器人展示复杂社会行为，同时保持实时碰撞避免能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) LISN-Bench：首个基于模拟的语言指令社会导航基准测试，包含多种场景和标准化评估指标；2) Social-Nav-Modulator：分层框架，解耦VLM推理和低级控制；3) 通过工具调用机制将VLM的高级理解转化为具体控制参数。相比之前工作，该方法不依赖顺序模拟模型，支持连续实时控制；不是简单适应障碍物，而是用VLM动态调整规划器参数；不是将VLM用于端到端导航，而是作为参数调制器，兼顾语义理解和实时控制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LISN论文提出了首个语言指令社会导航基准测试和分层控制框架，通过解耦VLM语义推理和低级反应控制，实现了在复杂动态环境中既符合社会规范又保持安全高效的机器人导航。'}


### 论文摘要

Towards human-robot coexistence, socially aware navigation is significant for mobile robots. Yet existing studies on this area focus mainly on path efficiency and pedestrian collision avoidance, which are essential but represent only a fraction of social navigation. Beyond these basics, robots must also comply with user instructions, aligning their actions to task goals and social norms expressed by humans. In this work, we present LISN-Bench, the first simulation-based benchmark for language-instructed social navigation. Built on Rosnav-Arena 3.0, it is the first standardized social navigation benchmark to incorporate instruction following and scene understanding across diverse contexts. To address this task, we further propose Social-Nav-Modulator, a fast-slow hierarchical system where a VLM agent modulates costmaps and controller parameters. Decoupling low-level action generation from the slower VLM loop reduces reliance on high-frequency VLM inference while improving dynamic avoidance and perception adaptability. Our method achieves an average success rate of 91.3%, which is greater than 63% than the most competitive baseline, with most of the improvements observed in challenging tasks such as following a person in a crowd and navigating while strictly avoiding instruction-forbidden regions. The project website is at: https://social-nav.github.io/LISN-project/

---

## 38. UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision

**论文链接:** [http://arxiv.org/abs/2512.09583v1](http://arxiv.org/abs/2512.09583v1)

**作者:** Alberto Rota, Mert Kiray, Mert Asim Karaoglu, Patrick Ruhkamp, Elena De Momi, Nassir Navabm, Benjamin Busam

**发布时间:** 2025-12-10

### GPT解析

### 总结

UnReflectAnything是一种仅使用RGB的框架，能够从单张图像中去除高光，适用于自然和外科图像领域，在多个基准测试中取得了与最先进方法相竞争的性能。

### 背景

高光会扭曲物体外观，模糊纹理细节，并在自然和外科图像中阻碍几何推理，这些问题在非朗伯表面和非均匀光照条件下尤为严重。

### 目的

开发一种能够从单张图像中去除高光的RGB-only框架，通过预测高光图和无反射漫反射重建来解决高光问题。

### 方法

使用冻结的视觉Transformer编码器提取多尺度特征，轻量级头部定位镜面区域，令牌级修复模块恢复损坏的特征块，并引入虚拟高光合成流程，利用单目几何、菲涅耳感知着色和随机光照渲染物理合理的高光，使模型能够在任意RGB图像上训练。

### 主要发现

UnReflectAnything能够在自然和外科领域有效泛化，处理非朗伯表面和非均匀照明产生的严重高光问题，并在多个基准测试中取得与最先进方法相竞争的性能。

### 结论

UnReflectAnything是一种有效的单图像高光去除方法，能够恢复无高光的漫反射图像，适用于多种应用场景。

### 翻译

高光会扭曲外观，模糊纹理，并在自然和外科影像中阻碍几何推理。我们提出了UnReflectAnything，一种仅使用RGB的框架，通过预测高光图和无反射漫反射重建从单张图像中去除高光。该模型使用冻结的视觉Transformer编码器提取多尺度特征，轻量级头部定位镜面区域，以及令牌级修复模块在生成最终漫反射图像之前恢复损坏的特征块。为克服配对监督的缺乏，我们引入了虚拟高光合成流程，使用单目几何、菲涅耳感知着色和随机光照渲染物理合理的高光，使模型能够在具有正确几何结构的任意RGB图像上进行训练。UnReflectAnything能够在自然和外科领域泛化，其中非朗伯表面和非均匀照明会创建严重高光，并在几个基准测试中取得了与最先进结果相竞争的性能。项目页面：https://alberto-rota.github.io/UnReflectAnything/


### 论文摘要

Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery. We present UnReflectAnything, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction. The model uses a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a token-level inpainting module that restores corrupted feature patches before producing the final diffuse image. To overcome the lack of paired supervision, we introduce a Virtual Highlight Synthesis pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting which enables training on arbitrary RGB images with correct geometric structure. UnReflectAnything generalizes across natural and surgical domains where non-Lambertian surfaces and non-uniform lighting create severe highlights and it achieves competitive performance with state-of-the-art results on several benchmarks. Project Page: https://alberto-rota.github.io/UnReflectAnything/

---

## 39. SIP: Site in Pieces- A Dataset of Disaggregated Construction-Phase 3D Scans for Semantic Segmentation and Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.09062v1](http://arxiv.org/abs/2512.09062v1)

**作者:** Seongyong Kim, Yong Kwon Cho

**发布时间:** 2025-12-09

### GPT解析

### 总结

论文介绍了SIP（Site in Pieces）数据集，该数据集反映了建筑工地上LiDAR数据采集的实际约束，包括室内和室外场景，使用地面激光扫描仪捕获，并进行了点级别注释。该数据集包含结构组件和细长临时物体，如脚手架、MEP管道和剪刀式升降机，并提供了可适应的类配置，简化在现代3D深度学习框架中的采用。

### 背景

准确的建筑工地3D场景解释对于进度监测、安全评估和数字孪生发展至关重要。LiDAR在建筑中被广泛使用，因为它比基于相机的系统具有优势，并且在杂乱和动态变化条件下表现可靠。然而，大多数用于3D感知的公共数据集来自密集融合扫描，具有均匀采样和完全可见性，这些条件不能反映真实的建筑工地。现场数据通常作为孤立的单站点LiDAR视图收集，受到安全要求、有限访问和持续操作的限制。

### 目的

创建一个反映建筑工地上LiDAR采集实际约束的数据集，解决现有数据集中径向密度衰减、碎片几何和视点相关可见性等特征代表不足的问题。

### 方法

创建SIP数据集，包括室内和室外场景，使用地面激光扫描仪捕获，并使用针对建筑环境定制的分类法进行点级别注释：A. 建筑环境，B. 施工作业，C. 场地周围环境。建立扫描协议、注释工作流程和质量控制程序，确保数据集的一致性。

### 主要发现

建筑工地上的LiDAR数据采集受到多种因素限制，导致数据具有径向密度衰减、碎片几何和视点相关可见性等特征。现有数据集未能充分代表这些特征，SIP数据集通过保留真实世界感知特性，能够实现强大的基准测试。

### 结论

SIP数据集通过提供保留真实世界感知特性的现场数据，能够实现强大的基准测试，并为推进面向建筑的3D视觉任务做出贡献。该数据集是公开可用的，并附带一个支持Git仓库，提供可适应的类配置，简化在现代3D深度学习框架中的采用。

### 翻译

准确的建筑工地3D场景解释对于进度监测、安全评估和数字孪生发展至关重要。LiDAR在建筑中被广泛使用，因为它比基于相机的系统具有优势，并且在杂乱和动态变化条件下表现可靠。然而，大多数用于3D感知的公共数据集来自密集融合扫描，具有均匀采样和完全可见性，这些条件不能反映真实的建筑工地。现场数据通常作为孤立的单站点LiDAR视图收集，受到安全要求、有限访问和持续操作的限制。这些因素导致径向密度衰减、碎片几何和视点相关可见性等特征，这些特征在现有数据集中尚未得到充分代表。本文介绍了SIP（Site in Pieces）数据集，该数据集创建于反映建筑工地上LiDAR采集的实际约束。SIP提供了使用地面激光扫描仪捕获的室内和室外场景，并使用针对建筑环境定制的分类法进行了点级别注释：A. 建筑环境，B. 施工作业，C. 场地周围环境。该数据集包括结构组件和细长临时物体，如脚手架、MEP管道和剪刀式升降机，其中由遮挡和碎片几何引起的稀疏性使得分割特别具有挑战性。扫描协议、注释工作流程和质量控制程序为数据集建立了一致的基础。SIP是公开可用的，并附带一个支持Git仓库，提供可适应的类配置，简化在现代3D深度学习框架中的采用。通过提供保留真实世界感知特性的现场数据，SIP能够实现强大的基准测试，并为推进面向建筑的3D视觉任务做出贡献。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的问题是现有3D感知数据集无法准确反映真实建筑工地的激光雷达扫描条件。现实中，建筑工地扫描受安全限制、有限访问和持续施工影响，导致数据具有遮挡、不完整几何和密度不均匀等特征；而研究中的现有数据集多是理想化的多视图融合扫描，无法代表真实环境，限制了建筑场景理解算法在实际应用中的效果，而准确的3D场景理解对施工进度监控、安全评估和数字孪生开发至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到建筑工地扫描的独特特征和挑战，然后分析现有数据集（如S3DIS、ScanNet、SemanticKITTI等）的局限性，发现它们都是多视图融合扫描，无法反映真实工地条件。作者借鉴了现有数据集的标注方法和分类体系，采用了点云处理工具（如CloudCompare、Open3D），并参考了数据集组织结构，但在数据采集策略上故意避免多视图融合，以保留单站扫描的真实特性，专门针对建筑环境设计了一套分类体系和采集流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建SIP数据集，专门针对建筑工地的激光雷达扫描条件，保留单站扫描的原始特征（遮挡、不完整几何、密度不均匀），并提供详细的语义标注。实现流程包括：1)使用Faro激光雷达采集40个单站场景（27室内+13室外）；2)将原始格式转换为ASCII格式并保留多种点属性；3)进行均匀随机下采样保持原始密度不平衡；4)人工标注23个语义类别（建成环境、施工操作、场地周围）；5)严格质量控制确保标注准确性；6)提供完整的数据集基础设施和工具支持现代3D深度学习框架。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)真实建筑工地扫描条件的数据集，保留单站扫描的原始局限性和挑战；2)建筑特定的语义标注集，包含23个针对建筑环境定制的类别；3)可重用的数据集基础设施，支持多种3D感知框架；4)支持高级研究方向如遮挡鲁棒分割和建筑机器人感知。相比之前工作，SIP采用单站扫描而非多视图融合，专注于包含临时结构和设备的建筑环境，具有径向密度衰减和碎片化几何等真实特征，分类体系也针对建筑场景定制，而非通用环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SIP数据集通过提供反映真实建筑工地扫描条件的单站激光雷达扫描及其详细语义标注，填补了现有3D感知数据集在建筑场景理解和语义分割方面的空白，为开发能够在真实工地环境下鲁棒运行的3D视觉算法提供了宝贵的资源。'}


### 论文摘要

Accurate 3D scene interpretation in active construction sites is essential for progress monitoring, safety assessment, and digital twin development. LiDAR is widely used in construction because it offers advantages over camera-based systems, performing reliably in cluttered and dynamically changing conditions. Yet most public datasets for 3D perception are derived from densely fused scans with uniform sampling and complete visibility, conditions that do not reflect real construction sites. Field data are often collected as isolated single-station LiDAR views, constrained by safety requirements, limited access, and ongoing operations. These factors lead to radial density decay, fragmented geometry, and view-dependent visibility-characteristics that remain underrepresented in existing datasets. This paper presents SIP, Site in Pieces, a dataset created to reflect the practical constraints of LiDAR acquisition during construction. SIP provides indoor and outdoor scenes captured with a terrestrial LiDAR scanner and annotated at the point level using a taxonomy tailored to construction environments: A. Built Environment, B. Construction Operations, and C. Site Surroundings. The dataset includes both structural components and slender temporary objects such as scaffolding, MEP piping, and scissor lifts, where sparsity caused by occlusion and fragmented geometry make segmentation particularly challenging. The scanning protocol, annotation workflow, and quality control procedures establish a consistent foundation for the dataset. SIP is openly available with a supporting Git repository, offering adaptable class configurations that streamline adoption within modern 3D deep learning frameworks. By providing field data that retain real-world sensing characteristics, SIP enables robust benchmarking and contributes to advancing construction-oriented 3D vision tasks.

---

## 40. Mind to Hand: Purposeful Robotic Control via Embodied Reasoning

**论文链接:** [http://arxiv.org/abs/2512.08580v2](http://arxiv.org/abs/2512.08580v2)

**作者:** Peijun Tang, Shangjin Xie, Binyan Sun, Baifu Huang, Kuncheng Luo, Haotian Yang, Weiqi Jin, Jianan Wang

**发布时间:** 2025-12-09

**备注:** 49 pages, 25 figures

### GPT解析

### 总结

本研究介绍Lumo-1，一种通用的视觉-语言-行动模型，成功将机器人的推理能力与行动能力统一起来，通过三阶段预训练管道和强化学习优化，在具身视觉语言推理和真实世界机器人任务中表现出色。

### 背景

人类行动依赖于上下文和意图，推理在其中扮演核心角色。尽管大规模互联网数据使AI系统具备了广泛的推理能力，但这些能力在物理行动中的基础应用仍然是一个重大挑战。

### 目的

开发一种通用的视觉-语言-行动（VLA）模型，将机器人的推理能力（'mind'）与行动能力（'hand'）统一起来，实现具身推理和行动的整合。

### 方法

构建一个三阶段预训练管道：(1)在精选的视觉语言数据上继续VLM预训练，增强具身推理技能；(2)在跨具身机器人数据和视觉语言数据上进行联合训练；(3)在Astribot S1上收集的轨迹上进行带推理过程的行动训练。最后整合强化学习来完善推理-行动一致性，形成语义推理和运动控制的闭环。

### 主要发现

大量实验表明，Lumo-1在具身视觉语言推理方面取得了显著的性能提升。真实世界评估显示，Lumo-1在广泛的挑战性机器人任务中超越了强大的基线模型，能够很好地泛化到新物体和新环境，在长时程任务中表现尤为出色，并能响应需要推理策略、概念和空间的人类自然指令。

### 结论

Lumo-1成功地将机器人的推理能力与行动能力统一起来，实现了具身视觉语言推理的显著提升，并在真实世界任务中表现出强大的泛化能力和性能，为通用机器人控制提供了重要进展。

### 翻译

人类行动依赖于上下文和意图，推理在其中扮演核心角色。尽管互联网规模的数据使AI系统具备了广泛的推理能力，但这些能力在物理行动中的基础应用仍然是一个重大挑战。我们引入Lumo-1，一个通用的视觉-语言-行动（VLA）模型，将机器人的推理能力（'mind'）与行动能力（'hand'）统一起来。我们的方法基于预训练的视觉语言模型（VLMs）的通用多模态推理能力，逐步扩展到具身推理和行动预测，最终实现结构化推理和推理-行动对齐。这构成了一个三阶段预训练管道：(1)在精选的视觉语言数据上继续VLM预训练，增强规划、空间理解和轨迹预测等具身推理技能；(2)在跨具身机器人数据和视觉语言数据上进行联合训练；(3)在Astribot S1（一个具有类人灵巧性和敏捷性的双臂移动机械臂）上收集的轨迹上进行带推理过程的行动训练。最后，我们整合强化学习来进一步完善推理-行动一致性，并在语义推理和运动控制之间形成闭环。大量实验表明，Lumo-1在具身视觉语言推理方面取得了显著的性能提升，这是通用机器人控制的关键组成部分。真实世界的评估进一步显示，Lumo-1在广泛的挑战性机器人任务中超越了强大的基线模型，能够很好地泛化到新物体和新环境，在长时程任务中表现尤为出色，并能响应需要推理策略、概念和空间的人类自然指令。


### 论文摘要

Humans act with context and intention, with reasoning playing a central role. While internet-scale data has enabled broad reasoning capabilities in AI systems, grounding these abilities in physical action remains a major challenge. We introduce Lumo-1, a generalist vision-language-action (VLA) model that unifies robot reasoning ("mind") with robot action ("hand"). Our approach builds upon the general multi-modal reasoning capabilities of pre-trained vision-language models (VLMs), progressively extending them to embodied reasoning and action prediction, and ultimately towards structured reasoning and reasoning-action alignment. This results in a three-stage pre-training pipeline: (1) Continued VLM pre-training on curated vision-language data to enhance embodied reasoning skills such as planning, spatial understanding, and trajectory prediction; (2) Co-training on cross-embodiment robot data alongside vision-language data; and (3) Action training with reasoning process on trajectories collected on Astribot S1, a bimanual mobile manipulator with human-like dexterity and agility. Finally, we integrate reinforcement learning to further refine reasoning-action consistency and close the loop between semantic inference and motor control. Extensive experiments demonstrate that Lumo-1 achieves significant performance improvements in embodied vision-language reasoning, a critical component for generalist robotic control. Real-world evaluations further show that Lumo-1 surpasses strong baselines across a wide range of challenging robotic tasks, with strong generalization to novel objects and environments, excelling particularly in long-horizon tasks and responding to human-natural instructions that require reasoning over strategy, concepts and space.

---

## 41. Incorporating Fairness in Neighborhood Graphs for Fair Spectral Clustering

**论文链接:** [http://arxiv.org/abs/2512.09810v1](http://arxiv.org/abs/2512.09810v1)

**作者:** Adithya K Moorthy, V Vijaya Saradhi, Bhanu Prasad

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了一种用于构建公平图的新方法，通过在图形成过程中强制执行人口统计均等，实现更公平的谱聚类结果，无需修改聚类算法本身。

### 背景

图聚类在无监督学习中扮演关键角色，但传统方法通过不公平的图构建可能低估某些群体，延续偏见。常用的kNN和ε邻域图会在敏感群体中传播不同的边缘影响，导致有偏的聚类结果。

### 目的

解决谱聚类预处理中的关键差距，证明图构建中的拓扑公平性对于实现公平聚类结果至关重要。

### 方法

提出在邻域选择步骤的最早阶段纳入公平约束的新方法，将敏感特征的比例表示纳入局部图结构，同时保持几何一致性。确保每个敏感群体在每节点的邻域中得到代表。

### 主要发现

在图构建中提供每个敏感群体在每节点邻域中的表示，会导致更公平的谱聚类结果，因为图的拓扑特征自然反映公平的群体比例。图构建中的拓扑公平性本身有助于实现更公平的谱聚类结果，无需更改聚类算法本身。

### 结论

通过在三个合成数据集、七个真实世界表格数据集和三个真实世界图像数据集上的 thorough 实验，证明了公平图构建方法在图聚类任务中超越了当前基线。

### 翻译

图聚类在无监督学习方法中如谱聚类中起着关键作用，但传统方法通过不公平的图构建可能低估某些群体而延续偏见。本研究介绍了构建公平k近邻和公平ε邻域图的新方法，在图形成过程中主动强制执行人口统计均等。通过在邻域选择步骤的最早阶段纳入公平约束，我们的方法将敏感特征的比例表示纳入局部图结构，同时保持几何一致性。我们的工作解决了公平谱聚类预处理中的关键差距，证明了图构建中的拓扑公平性对于实现公平聚类结果至关重要。常用的kNN和ε邻域图等图构建方法会在敏感群体中传播不同的边缘影响，导致有偏的聚类结果。在每个节点的邻域中提供每个敏感群体的代表，会导致更公平的谱聚类结果，因为图的拓扑特征自然反映公平的群体比例。本研究通过说明图构建中的拓扑公平性如何自然促进更公平的谱聚类结果，无需更改聚类算法本身，填补了公平无监督学习中的一个重要缺陷。在三个合成数据集、七个真实世界表格数据集和三个真实世界图像数据集上的 thorough 实验证明，我们的公平图构建方法在图聚类任务中超越了当前基线。


### 论文摘要

Graph clustering plays a pivotal role in unsupervised learning methods like spectral clustering, yet traditional methods for graph clustering often perpetuate bias through unfair graph constructions that may underrepresent some groups. The current research introduces novel approaches for constructing fair k-nearest neighbor (kNN) and fair epsilon-neighborhood graphs that proactively enforce demographic parity during graph formation. By incorporating fairness constraints at the earliest stage of neighborhood selection steps, our approaches incorporate proportional representation of sensitive features into the local graph structure while maintaining geometric consistency.Our work addresses a critical gap in pre-processing for fair spectral clustering, demonstrating that topological fairness in graph construction is essential for achieving equitable clustering outcomes. Widely used graph construction methods like kNN and epsilon-neighborhood graphs propagate edge based disparate impact on sensitive groups, leading to biased clustering results. Providing representation of each sensitive group in the neighborhood of every node leads to fairer spectral clustering results because the topological features of the graph naturally reflect equitable group ratios. This research fills an essential shortcoming in fair unsupervised learning, by illustrating how topological fairness in graph construction inherently facilitates fairer spectral clustering results without the need for changes to the clustering algorithm itself. Thorough experiments on three synthetic datasets, seven real-world tabular datasets, and three real-world image datasets prove that our fair graph construction methods surpass the current baselines in graph clustering tasks.

---

## 42. Improved Physics-Driven Neural Network to Solve Inverse Scattering Problems

**论文链接:** [http://arxiv.org/abs/2512.09333v1](http://arxiv.org/abs/2512.09333v1)

**作者:** Yutong Du, Zicheng Liu, Bo Wu, Jingwei Kou, Hang Li, Changyou Li, Yali Zong, Bo Qi

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了一种改进的物理驱动神经网络框架用于解决电磁逆散射问题

### 背景

电磁逆散射问题的求解面临挑战

### 目的

开发一种稳定、高效且准确的电磁逆散射问题求解器

### 方法

引入高斯局部化振荡抑制窗口激活函数；开发动态散射子区域识别策略；集成迁移学习

### 主要发现

新激活函数稳定收敛并实现轻量级准确网络；动态识别策略自适应细化计算域并降低成本；迁移学习扩展求解器适用性

### 结论

所提求解器在重建精度、鲁棒性和效率上优于现有最先进方法

### 翻译

本文提出了一种改进的物理驱动神经网络框架用于解决电磁逆散射问题。引入了一种新的高斯局部化振荡抑制窗口激活函数来稳定收敛并实现轻量级但准确的网络架构。进一步开发了动态散射子区域识别策略以自适应细化计算域，防止漏检并降低计算成本。此外，集成了迁移学习以扩展求解器在实际场景中的适用性，将迭代算法的物理可解释性与神经网络的实时推理能力相结合。数值模拟和实验结果表明，与现有最先进方法相比，所提出的求解器在重建精度、鲁棒性和效率方面表现出色。


### 论文摘要

This paper presents an improved physics-driven neural network (IPDNN) framework for solving electromagnetic inverse scattering problems (ISPs). A new Gaussian-localized oscillation-suppressing window (GLOW) activation function is introduced to stabilize convergence and enable a lightweight yet accurate network architecture. A dynamic scatter subregion identification strategy is further developed to adaptively refine the computational domain, preventing missed detections and reducing computational cost. Moreover, transfer learning is incorporated to extend the solver's applicability to practical scenarios, integrating the physical interpretability of iterative algorithms with the real-time inference capability of neural networks. Numerical simulations and experimental results demonstrate that the proposed solver achieves superior reconstruction accuracy, robustness, and efficiency compared with existing state-of-the-art methods.

---

## 43. Understanding Mental States in Active and Autonomous Driving with EEG

**论文链接:** [http://arxiv.org/abs/2512.09190v1](http://arxiv.org/abs/2512.09190v1)

**作者:** Prithila Angkan, Paul Hungler, Ali Etemad

**发布时间:** 2025-12-09

**备注:** 15 Pages, 13 Figures and 3 Tables. This work has been submitted to IEEE Transaction for possible publication

### GPT解析

### 总结

本研究首次通过脑电图比较了主动驾驶和自动驾驶模式下的认知负荷、疲劳、效价和唤醒度差异，发现尽管两种模式在复杂度趋势上相似，但心理状态强度和神经激活存在明显差异，且迁移学习实验证实了两种模式间的分布转移。

### 背景

理解驾驶员在主动驾驶和自动驾驶模式下的心理状态差异对于设计安全的人车界面至关重要。

### 目的

通过脑电图（EEG）比较两种驾驶模式下认知负荷、疲劳、效价和唤醒度的差异。

### 方法

使用31名参与者在三种不同复杂度水平下执行相同任务的数据，分析时间模式、任务复杂度效应和通道激活差异。

### 主要发现

两种模式在复杂度水平上表现出相似趋势，但心理状态强度和潜在神经激活有显著差异；主动驾驶和自动驾驶之间存在明显的分布转移；迁移学习实验证实模型在两种模式间难以互相推广；这种分布转移主要归因于运动参与和注意力需求的差异；尽管自动驾驶整体皮层激活降低，参与者仍表现出与干预准备、情绪反应和单调相关的心理状态波动。

### 结论

开发下一代自动驾驶车辆驾驶员监控系统时，需要特定场景的数据和模型。

### 翻译

理解驾驶员在主动驾驶和自动驾驶模式下的心理状态差异对于设计安全的人车界面至关重要。本文首次通过脑电图比较了两种驾驶模式下的认知负荷、疲劳、效价和唤醒度。使用来自31名参与者在三种不同复杂度水平下执行相同任务的数据，我们分析了时间模式、任务复杂度效应和通道激活差异。我们的发现表明，尽管两种模式在复杂度水平上表现出相似趋势，但心理状态的强度和潜在的神经激活存在显著差异，表明主动驾驶和自动驾驶之间存在明显的分布转移。迁移学习实验证实，在主动驾驶数据上训练的模型难以推广到自动驾驶，反之亦然。我们将这种分布转移主要归因于两种驾驶模式中运动参与和注意力需求的差异，这导致了不同的空间和时间脑电图激活模式。尽管自动驾驶导致整体皮层激活降低，参与者仍表现出与干预准备、任务诱发情绪反应和单调相关被动疲劳相关的认知负荷、疲劳、效价和唤醒度的可测量波动。这些结果强调，在为自动驾驶车辆开发下一代驾驶员监控系统时，需要特定场景的数据和模型。


### 论文摘要

Understanding how driver mental states differ between active and autonomous driving is critical for designing safe human-vehicle interfaces. This paper presents the first EEG-based comparison of cognitive load, fatigue, valence, and arousal across the two driving modes. Using data from 31 participants performing identical tasks in both scenarios of three different complexity levels, we analyze temporal patterns, task-complexity effects, and channel-wise activation differences. Our findings show that although both modes evoke similar trends across complexity levels, the intensity of mental states and the underlying neural activation differ substantially, indicating a clear distribution shift between active and autonomous driving. Transfer-learning experiments confirm that models trained on active driving data generalize poorly to autonomous driving and vice versa. We attribute this distribution shift primarily to differences in motor engagement and attentional demands between the two driving modes, which lead to distinct spatial and temporal EEG activation patterns. Although autonomous driving results in lower overall cortical activation, participants continue to exhibit measurable fluctuations in cognitive load, fatigue, valence, and arousal associated with readiness to intervene, task-evoked emotional responses, and monotony-related passive fatigue. These results emphasize the need for scenario-specific data and models when developing next-generation driver monitoring systems for autonomous vehicles.

---

## 44. Quantum Algorithm for Estimating Ollivier-Ricci Curvature

**论文链接:** [http://arxiv.org/abs/2512.09822v1](http://arxiv.org/abs/2512.09822v1)

**作者:** Nhat A. Nghiem, Linh Nguyen, Tuan K. Do, Tzu-Chieh Wei, Trung V. Phan

**发布时间:** 2025-12-10

### GPT解析

### 总结

介绍了一种用于计算Ollivier Ricci曲率的量子算法，该算法在特定问题上能实现比经典方法更快的指数级加速。

### 背景

Ollivier Ricci曲率是图和一般度量空间上通过最优输运定义的Ricci曲率的离散类比，已在金融网络信号脆弱性分析和组合量子力学等领域有应用。

### 目的

开发一种量子算法来计算Ollivier Ricci曲率，以解决几何问题并实现比经典方法更高效的计算。

### 方法

提出一种量子算法，对于给定点云和成对距离作为输入的问题，能够在两类特定问题上实现指数级加速。

### 主要发现

该量子算法可以在两类特定问题上实现比已知最佳经典方法更快的指数级加速。

### 结论

这项工作朝着能够提供实际价值同时指导基本理论的几何问题量子算法又迈进了一步。

### 翻译

我们介绍了一种用于计算Ollivier Ricci曲率的量子算法，这是图和一般度量空间上通过最优输运定义的Ricci曲率的离散类比。这种曲率已有应用范围从金融网络中的信号脆弱性到作为组合量子力学中的基本量。对于给定点云和成对距离作为输入的问题，我们证明我们的算法可以在两类特定问题上实现比已知最佳经典方法更快的指数级加速。我们的工作是朝着能够提供实际价值同时指导基本理论的几何问题量子算法又迈进了一步。


### 论文摘要

We introduce a quantum algorithm for computing the Ollivier Ricci curvature, a discrete analogue of the Ricci curvature defined via optimal transport on graphs and general metric spaces. This curvature has seen applications ranging from signaling fragility in financial networks to serving as basic quantities in combinatorial quantum gravity. For inputs given as a point cloud with pairwise distances, we show that our algorithm can achieve an exponential speedup over the best-known classical methods for two particular classes of problem. Our work is another step toward quantum algorithms for geometrical problems that are capable of delivering practical value while also informing fundamental theory.

---

## 45. 论文ID: 2512.09668v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.09668v1.json'

---

## 46. Super4DR: 4D Radar-centric Self-supervised Odometry and Gaussian-based Map Optimization

**论文链接:** [http://arxiv.org/abs/2512.09608v1](http://arxiv.org/abs/2512.09608v1)

**作者:** Zhiheng Li, Weihua Wang, Qiang Shen, Yichen Zhao, Zheng Fang

**发布时间:** 2025-12-10

**备注:** 17 pages, 20 figures

### GPT解析

### 总结

Super4DR是一种4D雷达中心框架，用于基于学习的里程计估计和高斯地图优化，在传统SLAM系统表现不佳的环境中表现优异。

### 背景

传统使用视觉或LiDAR数据的SLAM系统在弱光和恶劣天气条件下表现不佳。虽然4D雷达适合这类环境，但其稀疏和嘈杂的点云妨碍了准确的里程计估计，同时雷达地图结构模糊且不完整。

### 目的

提出一个名为Super4DR的4D雷达中心框架，用于基于学习的里程计估计和高斯地图优化，以克服传统方法的局限性。

### 方法

设计了一个集群感知的里程计网络，将聚类雷达点的对象级线索用于帧间匹配，并采用分层自监督机制克服异常值；使用3D高斯作为中间表示，结合雷达特定的增长策略、选择性分离和多视图正则化，以恢复模糊地图区域和未检测区域。

### 主要发现

Super4DR相比先前的自监督方法实现了67%的性能提升，几乎达到了有监督里程计的性能水平，缩小了与LiDAR的地图质量差距，同时支持多模态图像渲染。

### 结论

Super4DR是一种有效的4D雷达中心框架，能够在传统SLAM系统表现不佳的环境中提供良好的里程计估计和地图优化性能。

### 翻译

传统使用视觉或LiDAR数据的SLAM系统在弱光和恶劣天气条件下常常表现不佳。虽然4D雷达适合这类环境，但其稀疏和嘈杂的点云妨碍了准确的里程计估计，而雷达地图结构模糊且不完整。因此，我们提出了Super4DR，一种4D雷达中心框架，用于基于学习的里程计估计和高斯地图优化。首先，我们设计了一个集群感知的里程计网络，将聚类雷达点的对象级线索用于帧间匹配，同时采用分层自监督机制，通过时空一致性、知识转移和特征对比来克服异常值。其次，我们提出使用3D高斯作为中间表示，结合雷达特定的增长策略、选择性分离和多视图正则化，以恢复模糊的地图区域和那些基于图像纹理未检测到的区域。实验表明，Super4DR相比先前的自监督方法实现了67%的性能提升，几乎达到了有监督里程计的性能水平，缩小了与LiDAR的地图质量差距，同时支持多模态图像渲染。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的问题是：在恶劣天气条件（如低光照、雨雪、烟雾等）下，传统视觉和LiDAR SLAM系统性能下降的问题。4D雷达虽然适合这些环境，但其稀疏和嘈杂的点云导致难以进行准确的里程计估计，同时基于4D雷达构建的地图存在模糊和不完整的结构问题。这个问题在现实中非常重要，因为自动驾驶、机器人导航等应用需要在各种环境条件下可靠运行。在研究中，它推动了SLAM技术在极端条件下的进步，拓展了感知系统的应用边界。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了4D雷达数据的特性（稀疏、嘈杂、块状分布）和现有方法的局限性：基于几何的方法依赖密集点云，监督学习需要昂贵标签，自监督算法过度依赖刚性几何约束，而现有方法主要关注姿态精度忽视地图质量。作者借鉴了多项现有技术：3D高斯溅射作为地图表示、DBSCAN聚类处理点云分布、自监督学习中的对比学习和知识蒸馏、视觉基础模型提供深度先验。作者的创新在于将这些技术整合到一个统一框架中，并针对4D雷达特性进行了改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是以4D雷达为主要传感器，通过集群感知网络进行自监督里程计估计，并利用3D高斯表示优化地图质量。整体流程分为两阶段：1）里程计估计：对雷达点云进行聚类和下采样，通过点-簇特征编码器提取特征，使用多层次自监督损失（簇加权距离、列占用、教师引导等）训练网络估计姿态；2）地图优化：将初始雷达地图转换为3D高斯表示，应用深度辅助地面完成和几何感知的densification策略，使用选择性分离处理天空浮点物，通过多视图正则化优化高斯属性，最后将优化后的高斯转换回点云生成完整地图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）集群感知里程计网络：利用对象级而非点级匹配，引入簇加权距离和列占用损失，采用教师引导机制；2）高斯地图优化：首次将3D高斯溅射应用于4D雷达地图，提出深度辅助地面完成和几何感知densification，实现选择性分离和多视图正则化；3）统一框架：首次集成雷达里程计和地图优化，实现自监督训练和多模态渲染。相比之前工作，Super4DR同时关注里程计精度和地图质量，采用更适合雷达数据的对象级匹配，利用高斯作为中间表示，并引入视觉先验辅助雷达地图重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Super4DR首次提出了一种集成4D雷达自监督里程计和高斯地图优化的统一框架，显著提高了恶劣环境下的定位精度和地图质量，同时实现了多模态图像渲染能力。'}


### 论文摘要

Conventional SLAM systems using visual or LiDAR data often struggle in poor lighting and severe weather. Although 4D radar is suited for such environments, its sparse and noisy point clouds hinder accurate odometry estimation, while the radar maps suffer from obscure and incomplete structures. Thus, we propose Super4DR, a 4D radar-centric framework for learning-based odometry estimation and gaussian-based map optimization. First, we design a cluster-aware odometry network that incorporates object-level cues from the clustered radar points for inter-frame matching, alongside a hierarchical self-supervision mechanism to overcome outliers through spatio-temporal consistency, knowledge transfer, and feature contrast. Second, we propose using 3D gaussians as an intermediate representation, coupled with a radar-specific growth strategy, selective separation, and multi-view regularization, to recover blurry map areas and those undetected based on image texture. Experiments show that Super4DR achieves a 67% performance gain over prior self-supervised methods, nearly matches supervised odometry, and narrows the map quality disparity with LiDAR while enabling multi-modal image rendering.

---

## 47. REASAN: Learning Reactive Safe Navigation for Legged Robots

**论文链接:** [http://arxiv.org/abs/2512.09537v1](http://arxiv.org/abs/2512.09537v1)

**作者:** Qihao Yuan, Ziyu Cao, Ming Cao, Kailai Li

**发布时间:** 2025-12-10

**备注:** 8 pages

### GPT解析

### 总结

提出了一种基于单一LiDAR传感器的模块化端到端框架，用于腿式机器人在复杂动态环境中的反应式导航

### 背景

复杂动态环境中的腿式机器人导航面临挑战，需要实时、鲁棒的安全导航系统

### 目的

开发一种轻量级、高效的导航系统，能够在复杂环境中实现单机器人和多机器人的实时反应式导航

### 方法

设计四个模拟训练模块：三个强化学习策略（运动控制、安全防护、导航）和一个基于Transformer的外感受估计器；采用模块化分解简化复杂任务；使用标准强化学习训练方法，结合奖励塑造和课程设计

### 主要发现

通过消融实验验证了设计选择的正确性；与现有方法相比，在具有挑战性的导航任务中表现出更强的鲁棒性；REASAN系统实现了完全板载和实时反应式导航

### 结论

REASAN系统能够有效处理复杂环境中的导航挑战，无需依赖启发式或复杂的策略切换机制

### 翻译

我们提出了一种新颖的模块化端到端框架，用于在复杂动态环境中使用单一光检测和测距传感器进行腿式反应式导航。该系统包含四个模拟训练模块：三个用于运动控制、安全防护和导航的强化学习策略，以及一个处理原始点云输入的基于Transformer的外感受估计器。这种复杂腿式运动控制任务的模块化分解使得能够使用简单架构的轻量级神经网络，通过标准的强化学习实践进行训练，采用有针对性的奖励塑造和课程设计，而不依赖启发式或复杂的策略切换机制。我们进行了全面的消融实验来验证我们的设计选择，并证明与现有方法相比，在具有挑战性的导航任务中具有更强的鲁棒性。由此产生的反应式安全导航系统在单机器人和多机器人设置中，在复杂环境中实现了完全板载和实时的反应式导航。我们在https://github.com/ASIG-X/REASAN发布了我们的训练和部署代码。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决足式机器人在复杂动态环境中的反应式安全导航问题。这个问题很重要，因为足式机器人具有通用移动性优势，在搜救、物流、娱乐等领域有广泛应用前景，但在日常以人为中心的环境中导航仍面临挑战，包括高性能运动控制、处理动态障碍物、传感器集成和资源限制等问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过模块化设计解决复杂导航问题，将任务分解为专门模块。他们分析了现有方法的局限性，包括传统导航方法在动态环境中效率低下，以及现有学习方法处理复杂场景的不足。作者借鉴了强化学习、Transformer架构和点云处理技术，但创新性地组合这些技术，设计了四个模块：运动控制策略、安全屏蔽策略、导航策略和外感受估计器，采用顺序训练方式逐步构建完整系统。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将复杂足式导航任务分解为多个专门模块，每个模块负责特定功能，实现轻量级神经网络和简单架构，避免使用启发式方法或复杂策略切换机制。整体流程是：1) LiDAR传感器获取环境信息；2) 外感受估计器处理原始点云生成射线表示；3) 导航策略根据目标位置和外感受信息生成速度命令；4) 安全屏蔽策略将导航速度命令转换为安全速度命令；5) 运动控制策略根据安全速度命令生成关节目标位置；6) 内部PD控制器执行这些位置命令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 模块化设计，将复杂任务分解为三个RL策略和一个外感受估计器；2) 基于Transformer的外感受估计器，直接从原始LiDAR扫描处理动态障碍物感知；3) 简化训练流程，使用标准RL实践和针对性奖励设计；4) 实现完全机载和实时反应式导航。相比之前工作，REASAN不依赖中间表示或基于优化的解决方案，避免了启发式方法，能处理更复杂的反应行为如绕行障碍物和逃离死胡同。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REASAN提出了一种模块化端到端框架，通过轻量级神经网络和标准强化学习实践，实现了足式机器人在复杂动态环境中的完全机载、实时反应式安全导航。'}


### 论文摘要

We present a novel modularized end-to-end framework for legged reactive navigation in complex dynamic environments using a single light detection and ranging (LiDAR) sensor. The system comprises four simulation-trained modules: three reinforcement-learning (RL) policies for locomotion, safety shielding, and navigation, and a transformer-based exteroceptive estimator that processes raw point-cloud inputs. This modular decomposition of complex legged motor-control tasks enables lightweight neural networks with simple architectures, trained using standard RL practices with targeted reward shaping and curriculum design, without reliance on heuristics or sophisticated policy-switching mechanisms. We conduct comprehensive ablations to validate our design choices and demonstrate improved robustness compared to existing approaches in challenging navigation tasks. The resulting reactive safe navigation (REASAN) system achieves fully onboard and real-time reactive navigation across both single- and multi-robot settings in complex environments. We release our training and deployment code at https://github.com/ASIG-X/REASAN.

---

## 48. FUSER: Feed-Forward MUltiview 3D Registration Transformer and SE(3)$^N$ Diffusion Refinement

**论文链接:** [http://arxiv.org/abs/2512.09373v1](http://arxiv.org/abs/2512.09373v1)

**作者:** Haobo Jiang, Jin Xie, Jian Yang, Liang Yu, Jianmin Zheng

**发布时间:** 2025-12-10

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

本文提出了FUSER，首个前馈多视图点云配准transformer，以及FUSER-DF，一个SE(3)^N扩散细化框架，用于高效准确的多视图点云配准。

### 背景

传统的多视图点云配准依赖广泛的成对匹配来构建姿态图进行全局同步，这种方法计算量大，且在缺乏整体几何约束时本质上是不适定的。

### 目的

开发一种计算效率高且准确的多视图点云配准方法，避免传统方法的计算负担和不适定性问题。

### 方法

FUSER通过将每个扫描编码为低分辨率超点特征，利用稀疏3D CNN保留绝对平移线索，并通过几何交替注意力模块进行高效的扫描内和扫描间推理；FUSER-DF则基于FUSER构建SE(3)^N扩散细化框架，通过去噪纠正FUSER的估计。

### 主要发现

FUSER能够在不进行成对估计的情况下直接预测全局姿态；FUSER-DF能够通过去噪纠正FUSER的估计；在多个数据集上实验表明该方法实现了优越的配准精度和计算效率。

### 结论

所提出的方法在多视图点云配准任务中表现出色，在保持高精度的同时显著提高了计算效率。

### 翻译

多视图点云的配准传统上依赖于广泛的成对匹配来构建姿态图以实现全局同步，这种方法计算量大，并且在缺乏整体几何约束时本质上是不适定的。本文提出了FUSER，这是第一个前馈多视图配准transformer，它在统一的紧凑潜在空间中联合处理所有扫描，直接预测全局姿态，无需任何成对估计。为了保持可处理性，FUSER通过稀疏3D CNN将每个扫描编码为低分辨率超点特征，保留绝对平移线索，并通过几何交替注意力模块执行高效的扫描内和扫描间推理。特别地，我们将现成基础模型的2D注意力先验转移到3D特征交互和几何一致性的增强上。基于FUSER，我们进一步引入了FUSER-DF，一个SE(3)^N扩散细化框架，通过在联合SE(3)^N空间中进行去噪来纠正FUSER的估计。FUSER充当多视图配准模型来构建去噪器，并推导出先验条件化的SE(3)^N变分下界用于去噪监督。在3DMatch、ScanNet和ArkitScenes上的大量实验表明，我们的方法实现了优越的配准精度和卓越的计算效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多视角点云配准问题。传统方法依赖于大量两两配对构建姿态图并进行全局同步，这种方法计算成本高、缺乏全局几何约束、对离群值敏感且误差会累积。这个问题在3D场景重建、AR/VR和具身AI等应用中至关重要，因为它是这些领域的基础技术，直接影响最终重建质量和应用性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统两阶段配准范式的局限性，包括缺乏全局上下文、对离群值敏感、计算开销大和强归纳偏见。基于这些分析，他们设计了一种前馈式多视角配准范式，直接预测全局姿态。FUSER模型包含三个主要组件：绝对几何编码器（使用稀疏3D CNN）、几何交替注意力模块（交替进行扫描内和扫描间消息传递）和全局姿态预测器。作者借鉴了VGGT的交替注意力机制（但移除了参考标记以确保排列等变性），并将2D基础模型（π3）的注意力先验迁移到3D点云处理中，还基于SE(3)扩散模型思想提出了SE(3)^N扩散细化框架FUSER-DF。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过统一的紧凑潜在空间联合处理所有扫描，直接预测全局姿态，无需任何两两估计，同时利用绝对几何编码保留位置信息，使用交替注意力机制进行多扫描推理。整体流程为：1) 输入一组无序、部分重叠的点云扫描；2) 使用稀疏3D CNN将每个扫描编码为低分辨率超点特征；3) 通过几何交替注意力模块交替进行扫描内和扫描间的消息传递；4) 直接回归每个扫描的全局姿态；5) 可选地使用FUSER-DF进行SE(3)^N扩散细化，进一步优化姿态估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出首个前馈式多视角配准Transformer，直接预测全局姿态；2) 设计绝对几何编码器保留绝对空间线索；3) 提出几何交替注意力模块结合2D基础模型注意力先验；4) 引入SE(3)^N扩散细化框架FUSER-DF。相比之前的工作，FUSER避免了传统两阶段范式，无需大量两两配对，计算效率从分钟级提升到秒级，通过全局推理减少误差累积，并采用完全数据驱动的方式减少对手工设计的依赖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了FUSER，首个前馈式多视角3D配准Transformer，通过在紧凑潜在空间中联合处理所有扫描直接预测全局姿态，并引入SE(3)^N扩散细化框架FUSER-DF，显著提高了配准精度和计算效率。'}


### 论文摘要

Registration of multiview point clouds conventionally relies on extensive pairwise matching to build a pose graph for global synchronization, which is computationally expensive and inherently ill-posed without holistic geometric constraints. This paper proposes FUSER, the first feed-forward multiview registration transformer that jointly processes all scans in a unified, compact latent space to directly predict global poses without any pairwise estimation. To maintain tractability, FUSER encodes each scan into low-resolution superpoint features via a sparse 3D CNN that preserves absolute translation cues, and performs efficient intra- and inter-scan reasoning through a Geometric Alternating Attention module. Particularly, we transfer 2D attention priors from off-the-shelf foundation models to enhance 3D feature interaction and geometric consistency. Building upon FUSER, we further introduce FUSER-DF, an SE(3)$^N$ diffusion refinement framework to correct FUSER's estimates via denoising in the joint SE(3)$^N$ space. FUSER acts as a surrogate multiview registration model to construct the denoiser, and a prior-conditioned SE(3)$^N$ variational lower bound is derived for denoising supervision. Extensive experiments on 3DMatch, ScanNet and ArkitScenes demonstrate that our approach achieves the superior registration accuracy and outstanding computational efficiency.

---

## 49. ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation

**论文链接:** [http://arxiv.org/abs/2512.09364v1](http://arxiv.org/abs/2512.09364v1)

**作者:** Shengchao Zhou, Jiehong Lin, Jiahui Liu, Shizhen Zhao, Chirui Chang, Xiaojuan Qi

**发布时间:** 2025-12-10

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为ASSIST-3D的适应型3D场景合成管道，用于解决类别无关3D实例分割中的数据合成问题，通过三个关键创新增强了模型的泛化能力。

### 背景

类别无关3D实例分割需要分割所有物体实例（包括未见过的物体）而不依赖语义类别，当前方法因缺乏标注3D场景数据或有噪声的2D分割而难以泛化。现有3D场景合成方法无法同时满足几何多样性、上下文复杂性和布局合理性这三个关键要求。

### 目的

开发一个适应型3D场景合成管道（ASSIST-3D），用于合成适当的数据以增强类别无关3D实例分割模型的泛化能力。

### 方法

ASSIST-3D包含三个关键创新：1)从广泛3D CAD资产集合中进行异构物体选择，通过随机采样最大化几何和上下文多样性；2)通过LLM引导的空间推理结合深度优先搜索生成合理的场景布局；3)通过多视图RGB-D图像渲染和融合构建真实点云，模拟真实传感器数据获取过程。

### 主要发现

在ScanNetV2、ScanNet++和S3DIS基准上的实验表明，使用ASSIST-3D生成数据训练的模型显著优于现有方法。比较结果显示该专用管道优于现有3D场景合成方法。

### 结论

ASSIST-3D是一个有效的3D场景合成解决方案，能够生成高质量、多样化的合成数据，成功解决了类别无关3D实例分割中的数据稀缺问题，显著提升了模型性能。

### 翻译

类别无关3D实例分割处理具有挑战性的任务，即分割所有物体实例（包括以前未见过的实例），而不依赖语义类别。当前方法由于缺乏标注的3D场景数据或有噪声的2D分割而难以泛化。虽然合成数据生成提供了一个有前景的解决方案，但现有的3D场景合成方法无法同时满足几何多样性、上下文复杂性和布局合理性，这些对任务都至关重要。为满足这些需求，我们提出了一个用于类别无关3D实例分割的适应型3D场景合成管道，称为ASSIST-3D，用于合成适当的数据以增强模型泛化能力。具体而言，ASSIST-3D具有三个关键创新，包括1)从广泛的3D CAD资产集合中进行异构物体选择，在物体采样中引入随机性以最大化几何和上下文多样性；2)通过LLM引导的空间推理结合深度优先搜索进行场景布局生成，以实现合理的物体放置；以及3)通过多视图RGB-D图像渲染和融合从合成场景构建真实的点云，紧密模拟真实世界传感器数据获取。在ScanNetV2、ScanNet++和S3DIS基准上的实验表明，使用ASSIST-3D生成数据训练的模型显著优于现有方法。进一步的比较凸显了我们专用管道优于现有3D场景合成方法的优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决类别无关的3D实例分割中的数据稀缺问题。这个问题在现实中很重要，因为自动驾驶、机器人导航和虚拟现实等领域需要系统能够识别和分割各种未知物体，而真实世界中存在大量未在训练数据中出现过的物体类别，同时3D数据采集和标注成本高昂，导致真实世界3D场景数据稀缺。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性，认识到数据多样性是提升类别无关3D实例分割泛化能力的关键因素。他们借鉴了现有工作中的LLM空间推理、DFS布局策略和多视图点云构建技术，但针对现有3D场景合成方法的不足进行了创新改进，设计了专门流程来同时满足几何多样性、上下文复杂性和布局合理性三个关键原则。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过精心设计的3D场景合成流程，生成高质量的、多样化的合成3D场景数据，解决真实世界3D数据稀缺问题。整体流程分三阶段：1)异构物体选择：从大规模3D模型库随机采样物体，确保多样性；2)场景布局生成：利用LLM推断物体空间关系，结合DFS策略合理放置物体；3)真实点云构建：模拟真实传感器数据获取过程，通过多视图RGB-D图像渲染和融合生成真实点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)专门为类别无关3D实例分割设计的3D场景合成流程；2)同时满足几何多样性、上下文复杂性和布局合理性三个原则；3)模拟真实传感器获取过程的点云构建方法。相比之前工作，与Holodeck不同在于通过随机采样确保多样性和避免偏向常见物体；与RandomRooms不同在于使用LLM确保布局合理性；与基于2D基础模型的方法不同在于直接生成3D数据，避免2D分割错误和多视图融合问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ASSIST-3D提出了一种创新的3D场景合成方法，通过同时确保几何多样性、上下文复杂性和布局合理性，生成高质量的合成数据，显著提升了类别无关3D实例分割模型的泛化能力。'}


### 论文摘要

Class-agnostic 3D instance segmentation tackles the challenging task of segmenting all object instances, including previously unseen ones, without semantic class reliance. Current methods struggle with generalization due to the scarce annotated 3D scene data or noisy 2D segmentations. While synthetic data generation offers a promising solution, existing 3D scene synthesis methods fail to simultaneously satisfy geometry diversity, context complexity, and layout reasonability, each essential for this task. To address these needs, we propose an Adapted 3D Scene Synthesis pipeline for class-agnostic 3D Instance SegmenTation, termed as ASSIST-3D, to synthesize proper data for model generalization enhancement. Specifically, ASSIST-3D features three key innovations, including 1) Heterogeneous Object Selection from extensive 3D CAD asset collections, incorporating randomness in object sampling to maximize geometric and contextual diversity; 2) Scene Layout Generation through LLM-guided spatial reasoning combined with depth-first search for reasonable object placements; and 3) Realistic Point Cloud Construction via multi-view RGB-D image rendering and fusion from the synthetic scenes, closely mimicking real-world sensor data acquisition. Experiments on ScanNetV2, ScanNet++, and S3DIS benchmarks demonstrate that models trained with ASSIST-3D-generated data significantly outperform existing methods. Further comparisons underscore the superiority of our purpose-built pipeline over existing 3D scene synthesis approaches.

---

## 50. Magic Gems: A Polyhedral Framework for Magic Squares

**论文链接:** [http://arxiv.org/abs/2512.09170v1](http://arxiv.org/abs/2512.09170v1)

**作者:** Kyle Elliott Mathewson

**发布时间:** 2025-12-09

**备注:** Connecting Combinatorics, Geometry, and Linear Algebra. 8 figures, ancillary code included. Interactive visualization: https://kylemath.github.io/MagicGemWebpage/

### GPT解析

### 总结

本文介绍了Magic Gems，一种将魔方表示为三维多面体的几何表示方法，揭示了魔方约束与统计结构之间的联系，并通过协方差能量泛函证明了魔方的特性。

### 背景

魔方是一种古老的数学结构，但其在几何表示和统计特性方面的研究尚不充分。

### 目的

开发一种将魔方表示为三维几何对象的方法，揭示魔方的统计特性，并提供一种规范化的几何表示方法。

### 方法

通过将n×n魔方映射到以坐标网格为中心的坐标系中，将单元格值作为垂直位移，构建点云，其凸包定义了Magic Gem。引入协方差能量泛函，并通过穷举法和大规模采样分析魔方的特性。

### 主要发现

1. 魔方中位置和值之间的协方差为零；2. 协方差能量泛函的零点恰好是魔方；3. 魔方是孤立的局部最小值；4. Magic Gems表示在二面体对称性D_4下是不变的。

### 结论

Magic Gems提供了一种将魔方表示为三维几何对象的方法，揭示了魔方的统计特性，并为魔方的等价类提供了规范化的几何表示。

### 翻译

我们引入Magic Gems，这是一种将魔方表示为三维多面体的几何表示。通过将n×n魔方映射到以坐标网格为中心的坐标系中，将单元格值作为垂直位移，我们构建了一个点云，其凸包定义了Magic Gem。这揭示了魔方约束与统计结构之间的联系：我们证明了魔方中位置和值之间的协方差为零。我们引入了协方差能量泛函——行、列和对角线指示变量协方差的平方和，并证明对于n=3（通过穷举法），其零点恰好是魔方。对n=4,5（4.6亿多种排列）进行的大规模采样提供了强有力的数值证据，表明这一特性可以扩展到更高阶。扰动分析表明魔方是孤立的局部最小值。该表示在二面体对称性D_4下是不变的，为等价类产生了规范几何对象。


### 论文摘要

We introduce Magic Gems, a geometric representation of magic squares as three-dimensional polyhedra. By mapping an n x n magic square onto a centered coordinate grid with cell values as vertical displacements, we construct a point cloud whose convex hull defines the Magic Gem. This reveals a connection between magic square constraints and statistical structure: we prove that magic squares have vanishing covariances between position and value. We introduce a covariance energy functional -- the sum of squared covariances with row, column, and diagonal indicator variables -- and prove for n=3 (via exhaustive enumeration) that its zeros are precisely the magic squares. Large-scale sampling for n=4,5 (460+ million arrangements) provides strong numerical evidence that this characterization extends to larger orders. Perturbation analysis demonstrates that magic squares are isolated local minima. The representation is invariant under dihedral symmetry D_4, yielding canonical geometric objects for equivalence classes.

---

## 51. FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.09872v1](http://arxiv.org/abs/2512.09872v1)

**作者:** Khurram Khalil, Khaza Anuarul Hoque

**发布时间:** 2025-12-10

**备注:** Accepted in IEEE HOST 2026

### GPT解析

### 总结

本文提出FlipLLM，一种强化学习框架，用于高效发现大型语言模型和视觉模型中的位翻转攻击漏洞。该框架结合灵敏度引导的层剪枝和Q-learning，能够快速识别关键位集合，仅需翻转5-7个关键位即可导致模型性能急剧下降。

### 背景

生成式人工智能模型如大型语言模型和大型视觉模型虽表现优异，但容易受到基于硬件的位翻转攻击威胁。现有BFA发现方法缺乏泛化能力且难以扩展，无法在合理时间内分析现代基础模型的复杂参数空间。

### 目的

开发一种可扩展且自适应的方法来探索语言和多模态基础模型的BFA脆弱性，为全面的硬件安全评估提供新途径。

### 方法

提出FlipLLM，一种与架构无关的强化学习框架，将BFA发现表述为顺序决策问题。结合灵敏度引导的层剪枝和Q-learning算法，高效识别可导致灾难性故障的最小高影响力位集合。

### 主要发现

FlipLLM比现有方法快2.5倍识别关键位；翻转5个关键位可将LLaMA 3.1 8B准确率从69.9%降至0.2%；翻转7个关键位可将LLaVA的VQA分数从78%降至几乎0%；对识别的位位置应用标准硬件保护机制可完全缓解BFA影响。

### 结论

FlipLLM提供了首个可扩展和自适应的探索语言和多模态基础模型BFA脆弱性的方法，为全面硬件安全评估铺平道路，具有指导硬件级防御的实用价值。

### 翻译

生成式人工智能模型，如大型语言模型和大型视觉模型，展现出最先进的性能，但仍然容易受到基于硬件的威胁，特别是位翻转攻击。现有的BFA发现方法缺乏泛化能力且难以扩展，通常无法在合理时间内分析现代基础模型的巨大参数空间和复杂相互依赖关系。本文提出了FlipLLM，一种与架构无关的强化学习框架，将BFA发现表述为顺序决策问题。FlipLLM结合了灵敏度引导的层剪枝和Q-learning，以高效识别能够导致灾难性故障的最小、高影响力位集合。我们通过将FlipLLM应用于多样化的模型集和数据集来证明其有效性和泛化能力。我们的结果表明，FlipLLM能够比最先进方法快2.5倍识别易受位翻转攻击影响的关键位。进一步分析表明，对识别的位位置应用标准硬件保护机制可完全缓解BFA影响，展示了我们的框架在指导硬件级防御方面的实用价值。


### 论文摘要

Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.

---

## 52. Seeing Soil from Space: Towards Robust and Scalable Remote Soil Nutrient Analysis

**论文链接:** [http://arxiv.org/abs/2512.09576v1](http://arxiv.org/abs/2512.09576v1)

**作者:** David Seu, Nicolas Longepe, Gabriel Cioltea, Erik Maidik, Calin Andrei

**发布时间:** 2025-12-10

**备注:** 23 pages, 13 figures, 13 tables

### GPT解析

### 总结

该研究开发了一个结合遥测数据和环境协变量的混合建模系统，用于精确估计农田土壤属性，包括有机碳、总氮、有效磷、交换性钾和pH值。系统在欧洲多样化农田土壤条件下表现良好，特别是在有机碳和总氮的预测上，并通过严格的验证框架证明了其稳健性和实用性。

### 背景

环境变量越来越多地影响农业决策，但可获取且可扩展的土壤评估工具仍然有限。

### 目的

提出一个稳健且可扩展的建模系统，用于估计农田土壤属性，包括土壤有机碳、总氮、有效磷、交换性钾和pH值。

### 方法

采用混合建模方法，结合间接和直接土壤建模技术；使用从辐射传输模型派生的可解释物理信息协变量和基础模型的复杂非线性嵌入；在覆盖欧洲农田土壤多样化成土气候区的协调数据集上验证；采用严格的验证框架，包括空间阻塞、分层分割和统计上不同的训练-测试集。

### 主要发现

模型在土壤有机碳和总氮上获得最高精度，这种性能在未见过的位置上保持稳定；土壤有机碳的平均绝对误差为5.12 g/kg，一致性相关系数为0.77；总氮的平均绝对误差为0.44 g/kg，一致性相关系数为0.77；通过一致性校准，在目标置信水平下达到90%的覆盖率。

### 结论

本研究通过应用可扩展、数据驱动的土壤分析框架促进了农业的数字化转型，该框架可扩展到需要定量土壤评估的相关领域，如碳市场。

### 翻译

环境变量正越来越多地影响农业决策，然而可获取且可扩展的土壤评估工具仍然有限。本研究提出了一种稳健且可扩展的建模系统，用于估计农田土壤属性，包括土壤有机碳、总氮、有效磷、交换性钾和pH值，使用遥测数据和环境协变量。该系统采用混合建模方法，结合通过代理和驱动因素间接建模土壤的方法与直接光谱建模。我们通过使用从辐射传输模型派生的可解释物理信息协变量和基础模型的复杂非线性嵌入，扩展了当前方法。我们在覆盖欧洲农田土壤多样化成土气候区的协调数据集上验证了该系统。评估采用严格的验证框架，强制执行空间阻塞、分层分割和统计上不同的训练-测试集，这使得评估更加困难，并为未见区域产生更现实的误差估计。模型在土壤有机碳和总氮上获得了最高精度。这种性能在未见过的位置上保持稳定，无论是在空间交叉验证还是独立测试集下。土壤有机碳的平均绝对误差为5.12 g/kg，一致性相关系数为0.77；总氮的平均绝对误差为0.44 g/kg，一致性相关系数为0.77。我们还通过一致性校准评估了不确定性，在目标置信水平下达到90%的覆盖率。本研究通过应用可扩展、数据驱动的土壤分析框架促进了农业的数字化转型，该框架可扩展到需要定量土壤评估的相关领域，如碳市场。


### 论文摘要

Environmental variables are increasingly affecting agricultural decision-making, yet accessible and scalable tools for soil assessment remain limited. This study presents a robust and scalable modeling system for estimating soil properties in croplands, including soil organic carbon (SOC), total nitrogen (N), available phosphorus (P), exchangeable potassium (K), and pH, using remote sensing data and environmental covariates. The system employs a hybrid modeling approach, combining the indirect methods of modeling soil through proxies and drivers with direct spectral modeling. We extend current approaches by using interpretable physics-informed covariates derived from radiative transfer models (RTMs) and complex, nonlinear embeddings from a foundation model. We validate the system on a harmonized dataset that covers Europes cropland soils across diverse pedoclimatic zones. Evaluation is conducted under a robust validation framework that enforces strict spatial blocking, stratified splits, and statistically distinct train-test sets, which deliberately make the evaluation harder and produce more realistic error estimates for unseen regions. The models achieved their highest accuracy for SOC and N. This performance held across unseen locations, under both spatial cross-validation and an independent test set. SOC obtained a MAE of 5.12 g/kg and a CCC of 0.77, and N obtained a MAE of 0.44 g/kg and a CCC of 0.77. We also assess uncertainty through conformal calibration, achieving 90 percent coverage at the target confidence level. This study contributes to the digital advancement of agriculture through the application of scalable, data-driven soil analysis frameworks that can be extended to related domains requiring quantitative soil evaluation, such as carbon markets.

---

## 53. Representation Invariance and Allocation: When Subgroup Balance Matters

**论文链接:** [http://arxiv.org/abs/2512.09496v1](http://arxiv.org/abs/2512.09496v1)

**作者:** Anissa Alloula, Charles Jones, Zuzanna Wakefield-Skorniewska, Francesco Quinzan, Bartłomiej Papież

**发布时间:** 2025-12-10

### GPT解析

### 总结

本研究探讨了训练数据中不同人口统计群体代表性不平衡对模型性能的影响，提出了'潜在分离假设'，并通过理论分析和实证验证，证明预训练模型潜在空间中子群体间的分离程度决定了模型对子群体表示的依赖性，为数据收集和平衡决策提供了指导。

### 背景

训练数据中不同人口统计群体的代表性不平衡对模型跨群体泛化能力构成挑战。传统观点认为平衡子群体表示能优化性能，但最近的实证结果与此矛盾，表明不平衡数据分布可能提高子群体性能或对子群体性能无影响。

### 目的

系统研究子群体分配对模型性能的影响，通过改变训练数据组成来表征子群体性能对数据平衡的敏感性，并提出并验证'潜在分离假设'。

### 方法

对四个视觉和语言模型进行系统性研究，改变训练数据组成以分析子群体性能对数据平衡的敏感性；提出'潜在分离假设'并进行形式化；提供理论分析；通过实证验证假设；展示基础模型微调的实际应用。

### 主要发现

部分微调模型对子群体表示的依赖性取决于预训练模型潜在空间中子群体之间的分离程度；不平衡的数据分布在某些情况下可提高子群体性能；在某些情况下，训练过程中缺少整个子群体对子群体性能没有影响；对潜在子群体分离的定量分析可为数据收集和平衡决策提供信息。

### 结论

预训练模型潜在空间中子群体间的分离程度是决定模型对子群体表示依赖性的关键因素；通过分析潜在子群体分离，可以优化数据收集和平衡策略，从而提高模型在不同人口统计群体上的性能。

### 翻译

训练数据中人口统计群体的代表性不平衡对模型跨群体泛化能力构成挑战。标准做法假设平衡子群体表示能优化性能。然而，最近的实证结果与此假设相矛盾：在某些情况下，不平衡的数据分布实际上提高了子群体性能，而在其他情况下，训练过程中缺少整个子群体对子群体性能没有影响。我们对四个视觉和语言模型进行了子群体分配的系统性研究，通过改变训练数据组成来表征子群体性能对数据平衡的敏感性。我们提出了'潜在分离假设'，该假设认为部分微调模型对子群体表示的依赖性取决于预训练模型潜在空间中子群体之间的分离程度。我们形式化了这一假设，提供了理论分析，并通过实证验证了它。最后，我们展示了基础模型微调的实际应用，证明对潜在子群体分离的定量分析可以为数据收集和平衡决策提供信息。


### 论文摘要

Unequal representation of demographic groups in training data poses challenges to model generalisation across populations. Standard practice assumes that balancing subgroup representation optimises performance. However, recent empirical results contradict this assumption: in some cases, imbalanced data distributions actually improve subgroup performance, while in others, subgroup performance remains unaffected by the absence of an entire subgroup during training. We conduct a systematic study of subgroup allocation across four vision and language models, varying training data composition to characterise the sensitivity of subgroup performance to data balance. We propose the latent separation hypothesis, which states that a partially fine-tuned model's dependence on subgroup representation is determined by the degree of separation between subgroups in the latent space of the pre-trained model. We formalise this hypothesis, provide theoretical analysis, and validate it empirically. Finally, we present a practical application to foundation model fine-tuning, demonstrating that quantitative analysis of latent subgroup separation can inform data collection and balancing decisions.

---

## 54. Scene-agnostic Hierarchical Bimanual Task Planning via Visual Affordance Reasoning

**论文链接:** [http://arxiv.org/abs/2512.09310v1](http://arxiv.org/abs/2512.09310v1)

**作者:** Kwang Bin Lee, Jiho Kang, Sung-Hee Lee

**发布时间:** 2025-12-10

**备注:** 8 pages, 4 figures

### GPT解析

### 总结

提出了一种统一的场景无关双手任务规划框架，整合了三个关键模块，使智能体能够在复杂未知环境中执行语义上有意义且物理可行的双手协调任务。

### 背景

具身智能体在开放环境中需要将高级指令转化为具体可执行的行为，通常需要双手协调。虽然基础模型提供了强大语义推理能力，但现有机器人任务规划器主要局限于单手操作，未能解决场景无关环境中双手操作固有的空间、几何和协调挑战。

### 目的

开发一个统一的场景无关双手任务规划框架，桥接高级推理与3D具体化的双手执行能力。

### 方法

整合三个关键模块：1)视觉点定位分析场景图像检测对象并生成交互点；2)双手子目标规划器基于空间相邻性和可达性推理产生紧凑子目标；3)交互点驱动的双手提示将子目标与技能库绑定，实例化满足约束条件的同步动作序列。

### 主要发现

该框架使智能体能规划语义上有意义、物理可行且可并行化的双手行为；实验表明产生连贯可行且紧凑的双手计划；能泛化到杂乱场景无需重新训练，展示场景无关约束推理的鲁棒性。

### 结论

该框架成功解决了场景无关环境中的双手任务规划问题，将高级语义推理与具体双手执行有效结合，适用于复杂未知环境。

### 翻译

在开放环境中运行的具身智能体必须将高级指令转化为具体、可执行的行为，通常需要双手协调使用。虽然最近的基础模型提供了强大的语义推理能力，但现有的机器人任务规划器主要仍是单手的，未能解决场景无关环境中双手操作固有的空间、几何和协调挑战。我们提出了一个统一的场景无关双手任务规划框架，它将高级推理与3D具体化的双手执行联系起来。我们的方法整合了三个关键模块。视觉点定位分析单场景图像以检测相关对象并生成与世界对齐的交互点。双手子目标规划器基于空间相邻性和跨对象可达性进行推理，产生紧凑的、运动中性的子目标，利用双手协调行动的机会。交互点驱动的双手提示将这些子目标与结构化技能库绑定，实例化满足手部状态和约束条件的同步单手或双手动作序列。这些模块共同使智能体能够在杂乱、未见过的场景中规划语义上有意义、物理上可行且可并行化的双手行为。实验表明，它产生连贯、可行且紧凑的双手计划，并能泛化到杂乱场景而无需重新训练，展示了双手任务场景无关约束推理的鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决场景无关的双臂任务规划问题，即如何让机器人在开放环境中将高级指令转化为协调的双手操作行为。这个问题很重要，因为现实世界中许多任务需要双手配合完成（如一只手扶着容器，另一只手倒液体），而现有机器人系统大多只支持单手操作，无法处理双臂操作中固有的空间、几何和协调挑战，限制了机器人在复杂环境中的应用能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（如单手规划为主、缺乏场景泛化能力）进行思考，设计了一个模块化框架。他们借鉴了多个领域的工作：1) 基础模型（如LLMs和VLMs）提供语义推理能力；2) 视觉提示方法（如Set-of-Marks范式）用于场景理解；3) 传统符号规划方法（如PDDL）用于任务分解；4) 机器人操作中的关键点抽象技术用于细粒度交互。作者将这些元素整合创新，形成了三个核心模块：视觉点接地、双臂子目标规划和交互点驱动的双臂提示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过视觉特性推理和分层规划，实现场景无关的双臂协调操作。整体流程分为三阶段：1) 预处理阶段（视觉点接地）：分析场景图像，检测对象并生成交互点，构建空间邻接图；2) 任务规划阶段：双臂子目标规划器确定操作顺序和合并机会，交互点驱动的双臂提示将子目标转化为具体动作序列；3) 执行阶段：通过参数化操作符同步执行双手动作，并更新场景状态。这种方法使机器人能在未见过的新环境中高效规划双手操作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的场景无关双臂任务规划框架；2) 视觉点接地模块实现细粒度场景理解；3) 双臂子目标规划器通过合并规则提高效率；4) 交互点驱动的双臂提示确保动作可行性。相比之前工作，本文首次系统解决双臂操作问题，而非单手操作；强调场景无关性，无需针对新场景重新训练；采用分层结构而非端到端方法；整合视觉特性推理与技能库，确保物理可行性和语义一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种统一的场景无关双臂任务规划框架，通过视觉特性推理和分层规划，使机器人能在开放环境中高效、协调地执行双手操作，无需针对特定场景重新训练。'}


### 论文摘要

Embodied agents operating in open environments must translate high-level instructions into grounded, executable behaviors, often requiring coordinated use of both hands. While recent foundation models offer strong semantic reasoning, existing robotic task planners remain predominantly unimanual and fail to address the spatial, geometric, and coordination challenges inherent to bimanual manipulation in scene-agnostic settings. We present a unified framework for scene-agnostic bimanual task planning that bridges high-level reasoning with 3D-grounded two-handed execution. Our approach integrates three key modules. Visual Point Grounding (VPG) analyzes a single scene image to detect relevant objects and generate world-aligned interaction points. Bimanual Subgoal Planner (BSP) reasons over spatial adjacency and cross-object accessibility to produce compact, motion-neutralized subgoals that exploit opportunities for coordinated two-handed actions. Interaction-Point-Driven Bimanual Prompting (IPBP) binds these subgoals to a structured skill library, instantiating synchronized unimanual or bimanual action sequences that satisfy hand-state and affordance constraints. Together, these modules enable agents to plan semantically meaningful, physically feasible, and parallelizable two-handed behaviors in cluttered, previously unseen scenes. Experiments show that it produces coherent, feasible, and compact two-handed plans, and generalizes to cluttered scenes without retraining, demonstrating robust scene-agnostic affordance reasoning for bimanual tasks.

---

## 55. From SAM to DINOv2: Towards Distilling Foundation Models to Lightweight Baselines for Generalized Polyp Segmentation

**论文链接:** [http://arxiv.org/abs/2512.09307v1](http://arxiv.org/abs/2512.09307v1)

**作者:** Shivanshu Agnihotri, Snehashis Majhi, Deepak Ranjan Nayak, Debesh Jha

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了一种名为Polyp-DiFoM的新型蒸馏框架，将大规模视觉基础模型的丰富表示转移到轻量级分割基础模型中，实现了结肠镜息肉分割任务的高效准确部署，在五个基准数据集上显著优于基础模型和最先进模型，同时计算开销减少近9倍。

### 背景

结肠镜检查中息肉分割对结直肠癌早期检测至关重要，但息肉在大小、形状和颜色方面变化显著且具有隐蔽性，使分割具有挑战性。轻量级基础模型如U-Net等虽然易于部署且计算成本低，但难以处理上述问题；而大规模视觉基础模型如SAM等在自然图像领域表现优异，但直接迁移到医学影像任务面临数据集稀缺和领域知识缺乏的障碍。

### 目的

弥合基础模型与轻量级分割模型之间的差距，提出一种蒸馏框架将基础模型的丰富表示转移到轻量级分割基础模型中，实现临床环境中的高效准确部署。

### 方法

提出Polyp-DiFoM蒸馏框架，将基础模型的语义先验知识注入到U-Net和U-Net++等标准架构中，并进行频域编码以增强蒸馏效果和提高泛化能力。在Kvasir-SEG、CVC-ClinicDB、ETIS、ColonDB和CVC-300五个基准数据集上进行了广泛实验。

### 主要发现

Polyp-DiFoM在五个基准数据集上显著优于各自的基础模型和最先进的模型，同时计算开销减少了近9倍。

### 结论

Polyp-DiFoM成功地将基础模型的知识转移到轻量级模型中，实现了高效且准确的临床部署，代码已公开可用。

### 翻译

在结肠镜检查中准确的息肉分割对于结直肠癌的早期检测至关重要，但由于息肉在大小、形状和颜色方面的显著变化以及其隐蔽性，这仍然具有挑战性。虽然U-Net、U-Net++和PraNet等轻量级基础模型在易于部署和低计算成本方面具有优势，但它们难以处理上述问题，导致分割性能有限。相比之下，SAM、DINOv2、OneFormer和Mask2Former等大规模视觉基础模型在自然图像领域表现出了令人印象深刻的泛化性能。然而，它们直接迁移到医学影像任务（如结肠镜息肉分割）并不简单，主要原因是缺乏大规模数据集和领域特定知识。为了弥合这一差距，我们提出了一个新颖的蒸馏框架Polyp-DiFoM，将基础模型的丰富表示转移到轻量级分割基础模型中，允许在临床环境中高效准确地部署。特别地，我们将基础模型的语义先验知识注入到U-Net和U-Net++等标准架构中，并进行频域编码以增强蒸馏效果，验证了它们的泛化能力。我们在Kvasir-SEG、CVC-ClinicDB、ETIS、ColonDB和CVC-300五个基准数据集上进行了大量实验。值得注意的是，Polyp-DiFoM始终显著优于各自的基础模型以及最先进的模型，同时计算开销减少了近9倍。代码可在https://github.com/lostinrepo/PolypDiFoM获取。


### 论文摘要

Accurate polyp segmentation during colonoscopy is critical for the early detection of colorectal cancer and still remains challenging due to significant size, shape, and color variations, and the camouflaged nature of polyps. While lightweight baseline models such as U-Net, U-Net++, and PraNet offer advantages in terms of easy deployment and low computational cost, they struggle to deal with the above issues, leading to limited segmentation performance. In contrast, large-scale vision foundation models such as SAM, DINOv2, OneFormer, and Mask2Former have exhibited impressive generalization performance across natural image domains. However, their direct transfer to medical imaging tasks (e.g., colonoscopic polyp segmentation) is not straightforward, primarily due to the scarcity of large-scale datasets and lack of domain-specific knowledge. To bridge this gap, we propose a novel distillation framework, Polyp-DiFoM, that transfers the rich representations of foundation models into lightweight segmentation baselines, allowing efficient and accurate deployment in clinical settings. In particular, we infuse semantic priors from the foundation models into canonical architectures such as U-Net and U-Net++ and further perform frequency domain encoding for enhanced distillation, corroborating their generalization capability. Extensive experiments are performed across five benchmark datasets, such as Kvasir-SEG, CVC-ClinicDB, ETIS, ColonDB, and CVC-300. Notably, Polyp-DiFoM consistently outperforms respective baseline models significantly, as well as the state-of-the-art model, with nearly 9 times reduced computation overhead. The code is available at https://github.com/lostinrepo/PolypDiFoM.

---

## 56. FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.09282v1](http://arxiv.org/abs/2512.09282v1)

**作者:** Xiang Chen, Jinshan Pan, Jiangxin Dong, Jian Yang, Jinhui Tang

**发布时间:** 2025-12-10

**备注:** Project page: https://lowlevelcv.com/

### GPT解析

### 总结

本文提出了FoundIR-v2，一个基于扩散的高容量图像恢复基础模型，通过数据均衡调度范式优化不同任务的数据混合比例，并引入MoE驱动的调度器实现任务自适应的扩散先验分配，在50多个子任务中实现了优于现有方法的性能。

### 背景

近期研究表明，预训练数据的规模和质量提升推动了图像恢复基础模型的显著进步。

### 目的

研究发现不同恢复任务的数据混合比例是决定全能图像恢复模型整体性能的关键因素。

### 方法

提出了FoundIR-v2模型，采用数据均衡调度范式动态优化不同任务的混合训练数据集比例，利用数据混合定律确保数据集组成平衡；引入MoE驱动的调度器到生成预训练中，为每个恢复任务灵活分配任务自适应的扩散先验。

### 主要发现

通过数据均衡调度和MoE驱动的调度器，模型能够处理更广泛现实场景中的50多个子任务，并取得优于现有方法的性能。

### 结论

FoundIR-v2模型通过优化数据混合比例和引入任务自适应的扩散先验，在广泛的现实场景子任务中实现了全面的性能提升。

### 翻译

近期研究表明，预训练数据的规模和质量提升推动了图像恢复基础模型的显著进步。在这项工作中，我们发现来自不同恢复任务的数据混合比例也是一个关键因素，直接决定了全能图像恢复模型的总体性能。为此，我们提出了一个高容量的基于扩散的图像恢复基础模型FoundIR-v2，它采用数据均衡调度范式来动态优化来自不同任务的混合训练数据集的比例。通过利用数据混合定律，我们的方法确保了平衡的数据集组成，使模型能够在不同任务中实现一致的泛化和全面的性能。此外，我们在生成预训练中引入了一个有效的由专家混合(MoE)驱动的调度器，为每个恢复任务灵活分配任务自适应的扩散先验，考虑到不同任务表现出的不同退化形式和水平。大量实验证明，我们的方法可以解决更广泛现实场景中的50多个子任务，并取得了优于最先进方法的性能。


### 论文摘要

Recent studies have witnessed significant advances in image restoration foundation models driven by improvements in the scale and quality of pre-training data. In this work, we find that the data mixture proportions from different restoration tasks are also a critical factor directly determining the overall performance of all-in-one image restoration models. To this end, we propose a high-capacity diffusion-based image restoration foundation model, FoundIR-v2, which adopts a data equilibrium scheduling paradigm to dynamically optimize the proportions of mixed training datasets from different tasks. By leveraging the data mixing law, our method ensures a balanced dataset composition, enabling the model to achieve consistent generalization and comprehensive performance across diverse tasks. Furthermore, we introduce an effective Mixture-of-Experts (MoE)-driven scheduler into generative pre-training to flexibly allocate task-adaptive diffusion priors for each restoration task, accounting for the distinct degradation forms and levels exhibited by different tasks. Extensive experiments demonstrate that our method can address over 50 sub-tasks across a broader scope of real-world scenarios and achieves favorable performance against state-of-the-art approaches.

---

## 57. GLACIA: Instance-Aware Positional Reasoning for Glacial Lake Segmentation via Multimodal Large Language Model

**论文链接:** [http://arxiv.org/abs/2512.09251v1](http://arxiv.org/abs/2512.09251v1)

**作者:** Lalit Maurya, Saurabh Kaushik, Beth Tellman

**发布时间:** 2025-12-10

### GPT解析

### 总结

该论文提出了GLACIA框架，首次将大型语言模型与分割能力相结合，用于冰川湖监测，能够生成准确的分割掩码和空间推理输出，有效支持灾害准备和政策制定。

### 背景

冰川湖监测对于减轻预期的冰川湖溃决洪水风险具有重要意义。然而，基于卷积神经网络和视觉变换器的现有分割方法仅限于像素级预测，缺乏高级全局场景语义和人类可解释的推理能力。

### 目的

解决现有分割方法缺乏高级全局场景语义和人类可解释推理的问题，开发能够同时提供准确分割掩码和空间推理输出的框架，以支持冰川环境下的灾害准备和政策制定。

### 方法

提出GLACIA框架，将大型语言模型与分割能力相结合。构建了冰川湖位置推理(GLake-Pos)数据集管道，提供多样化的、空间基础问答对，以克服遥感领域中缺乏实例感知位置推理数据的问题。

### 主要发现

比较评估表明，GLACIA(mIoU: 87.30)超越了基于CNNs的最先进方法(mIoU: 78.55 - 79.01)、ViTs(mIoU: 69.27 - 81.75)、地理基础模型(mIoU: 76.37 - 87.10)以及基于推理的分割方法(mIoU: 60.12 - 75.66)。

### 结论

该方法通过促进自然语言交互，支持在快速变化的冰川环境下的直观灾害准备和知情决策制定，从而实现更高效和可解释的决策制定。代码已在GitHub上发布。

### 翻译

冰川湖监测对于减轻预期的冰川湖溃决洪水风险具有重要意义。然而，基于卷积神经网络和视觉变换器的现有分割方法仍局限于像素级预测，缺乏高级全局场景语义和人类可解释的推理。为解决这一问题，我们引入了GLACIA，这是第一个将大型语言模型与分割能力相结合的框架，能够生成准确的分割掩码和相应的空间推理输出。我们构建了冰川湖位置推理数据集管道，提供多样化的、空间基础的问答对，旨在克服遥感领域中缺乏实例感知位置推理数据的问题。比较评估表明，GLACIA超越了基于CNNs的最先进方法、ViTs、地理基础模型以及基于推理的分割方法。我们的方法通过促进自然语言交互，支持在快速变化的冰川环境下的直观灾害准备和知情决策制定，从而实现更高效和可解释的决策制定。代码已在GitHub上发布。


### 论文摘要

Glacial lake monitoring bears great significance in mitigating the anticipated risk of Glacial Lake Outburst Floods. However, existing segmentation methods based on convolutional neural networks (CNNs) and Vision Transformers (ViTs), remain constrained to pixel-level predictions, lacking high-level global scene semantics and human-interpretable reasoning. To address this, we introduce GLACIA (\textbf{G}lacial \textbf{LA}ke segmentation with \textbf{C}ontextual \textbf{I}nstance \textbf{A}wareness), the first framework that integrates large language models with segmentation capabilities to produce both accurate segmentation masks and corresponding spatial reasoning outputs. We construct the Glacial Lake Position Reasoning (GLake-Pos) dataset pipeline, which provides diverse, spatially grounded question-answer pairs designed to overcome the lack of instance-aware positional reasoning data in remote sensing. Comparative evaluation demonstrate that GLACIA (mIoU: 87.30) surpasses state-of-the-art method based on CNNs (mIoU: 78.55 - 79.01), ViTs (mIoU: 69.27 - 81.75), Geo-foundation models (mIoU: 76.37 - 87.10), and reasoning based segmentation methods (mIoU: 60.12 - 75.66). Our approach enables intuitive disaster preparedness and informed policy-making in the context of rapidly changing glacial environments by facilitating natural language interaction, thereby supporting more efficient and interpretable decision-making. The code is released on https://github.com/lalitmaurya47/GLACIA

---

## 58. LLMs for Analog Circuit Design Continuum (ACDC)

**论文链接:** [http://arxiv.org/abs/2512.09199v1](http://arxiv.org/abs/2512.09199v1)

**作者:** Yasaman Esfandiari, Jocelyn Rego, Austin Meyer, Jonathan Gallagher, Mia Levy

**发布时间:** 2025-12-09

### GPT解析

### 总结

本研究探讨了大型语言模型在模拟电路设计领域的适用性和一致性，重点关注人工智能辅助设计中人类仍参与其中的情况。

### 背景

大型语言模型和transformer架构在多种自然语言任务中展现出令人印象深刻的推理和生成能力，但它们在现实世界工程领域的可靠性和稳健性很大程度上尚未被探索，限制了它们在以人为中心的工作流程中的实际应用价值。

### 目的

研究LLMs在模拟电路设计任务中的适用性和一致性，该任务需要特定领域的推理、遵循物理约束和结构化表示，并专注于人工智能辅助设计，其中人类仍然参与其中。

### 方法

研究不同的数据表示如何影响模型行为，并比较较小模型(如T5、GPT-2)与更大的基础模型(如Mistral-7B、GPT-oss-20B)在不同训练条件下的表现。

### 主要发现

发现了关键的可靠性挑战，包括对数据格式的敏感性、生成设计的不稳定性以及未见过的电路配置的泛化能力有限。

### 结论

这些发现提供了关于LLMs作为增强人类在复杂工程任务中能力的工具的局限性和潜力的早期证据，为设计可靠、可部署的基础模型用于结构化、现实世界的应用提供了见解。

### 翻译

大型语言模型(LLMs)和transformer架构在多种自然语言任务中展现出令人印象深刻的推理和生成能力。然而，它们在现实世界工程领域的可靠性和稳健性很大程度上尚未被探索，限制了它们在以人为中心的工作流程中的实际应用价值。在这项工作中，我们研究了LLMs在模拟电路设计中的适用性和一致性——这是一个需要特定领域推理、遵循物理约束和结构化表示的任务——专注于人工智能辅助设计，其中人类仍然参与其中。我们研究了不同的数据表示如何影响模型行为，并比较了较小模型(如T5、GPT-2)与更大的基础模型(如Mistral-7B、GPT-oss-20B)在不同训练条件下的表现。我们的结果突出了关键的可靠性挑战，包括对数据格式的敏感性、生成设计的不稳定性以及未见过的电路配置的泛化能力有限。这些发现提供了关于LLMs作为增强人类在复杂工程任务中能力的工具的局限性和潜力的早期证据，为设计可靠、可部署的基础模型用于结构化、现实世界的应用提供了见解。


### 论文摘要

Large Language Models (LLMs) and transformer architectures have shown impressive reasoning and generation capabilities across diverse natural language tasks. However, their reliability and robustness in real-world engineering domains remain largely unexplored, limiting their practical utility in human-centric workflows. In this work, we investigate the applicability and consistency of LLMs for analog circuit design -- a task requiring domain-specific reasoning, adherence to physical constraints, and structured representations -- focusing on AI-assisted design where humans remain in the loop. We study how different data representations influence model behavior and compare smaller models (e.g., T5, GPT-2) with larger foundation models (e.g., Mistral-7B, GPT-oss-20B) under varying training conditions. Our results highlight key reliability challenges, including sensitivity to data format, instability in generated designs, and limited generalization to unseen circuit configurations. These findings provide early evidence on the limits and potential of LLMs as tools to enhance human capabilities in complex engineering tasks, offering insights into designing reliable, deployable foundation models for structured, real-world applications.

---

## 59. Digital Modeling of Spatial Pathway Activity from Histology Reveals Tumor Microenvironment Heterogeneity

**论文链接:** [http://arxiv.org/abs/2512.09003v1](http://arxiv.org/abs/2512.09003v1)

**作者:** Ling Liao, Changhuei Yang, Maxim Artyomov, Mark Watson, Adam Kepecs, Haowen Zhou, Alexey Sergushichev, Richard Cote

**发布时间:** 2025-12-09

### GPT解析

### 总结

研究人员开发了一种计算框架，可通过常规组织学图像预测肿瘤微环境中的信号通路活性，特别适用于研究TGFb信号通路，为整合图像分析和空间转录组学数据提供了新途径。

### 背景

空间转录组学(Spatial transcriptomics, ST)能够同时映射组织形态和空间分辨的基因表达，为研究肿瘤微环境异质性提供了独特机会。

### 目的

引入一个计算框架，可直接从苏木精-伊红(H&E)染色组织学图像预测空间通路活性，在微观分辨率(55和100微米)下进行预测。

### 方法

使用从计算病理学基础模型中提取的图像特征，在三个独立的乳腺癌和肺癌空间转录组学数据集上进行测试。

### 主要发现

TGFb信号通路是三个数据集中预测最准确的通路；在87-88%的可靠预测案例中，生成的空间TGFb活性图反映了肿瘤和相邻非肿瘤区域之间的预期对比；线性和非线性预测模型表现相似，表明图像特征可能与通路活性呈线性关系。

### 结论

从常规病理学图像中提取的特征可以恢复空间连贯且生物学可解释的通路模式，为在肿瘤微环境研究中整合基于图像的推理与空间转录组学信息提供了一种可扩展的策略。

### 翻译

空间转录组学(Spatial transcriptomics, ST)能够同时映射组织形态和空间分辨的基因表达，为研究肿瘤微环境异质性提供了独特机会。在此，我们介绍了一个计算框架，可直接从苏木精-伊红染色组织学图像在微观分辨率55和100微米处预测空间通路活性。使用从计算病理学基础模型派生的图像特征，我们发现TGFb信号通路是三个独立的乳腺癌和肺癌空间转录组学数据集中预测最准确的通路。在87-88%的可靠预测案例中，生成的空间TGFb活性图反映了肿瘤和相邻非肿瘤区域之间的预期对比，这与TGFb在调节肿瘤微环境相互作用中的已知作用一致。值得注意的是，线性和非线性预测模型表现相似，表明图像特征可能与通路活性呈主要线性关系，或者非线性结构相对于测量噪声较小。这些发现证明，从常规病理学图像中提取的特征可以恢复空间连贯且生物学可解释的通路模式，为在肿瘤微环境研究中整合基于图像的推理与空间转录组学信息提供了一种可扩展策略。


### 论文摘要

Spatial transcriptomics (ST) enables simultaneous mapping of tissue morphology and spatially resolved gene expression, offering unique opportunities to study tumor microenvironment heterogeneity. Here, we introduce a computational framework that predicts spatial pathway activity directly from hematoxylin-and-eosin-stained histology images at microscale resolution 55 and 100 um. Using image features derived from a computational pathology foundation model, we found that TGFb signaling was the most accurately predicted pathway across three independent breast and lung cancer ST datasets. In 87-88% of reliably predicted cases, the resulting spatial TGFb activity maps reflected the expected contrast between tumor and adjacent non-tumor regions, consistent with the known role of TGFb in regulating interactions within the tumor microenvironment. Notably, linear and nonlinear predictive models performed similarly, suggesting that image features may relate to pathway activity in a predominantly linear fashion or that nonlinear structure is small relative to measurement noise. These findings demonstrate that features extracted from routine histopathology may recover spatially coherent and biologically interpretable pathway patterns, offering a scalable strategy for integrating image-based inference with ST information in tumor microenvironment studies.

---

