# 今日论文推荐 - 2025-10-09

共 65 篇论文

---

## 1. AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs

**论文链接:** [http://arxiv.org/abs/2510.07293v1](http://arxiv.org/abs/2510.07293v1)

**作者:** Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xiaoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng Gu, Xiangqi Jin, Jiabing Yang, Kai Li, Zhifei Liu, Weijia Li, Cunxiang Wang, Conghui He, Linfeng Zhang

**发布时间:** 2025-10-08

**备注:** 26 pages, 23 figures, the code is available at  \url{https://github.com/DabDans/AudioMarathon}

### GPT解析

### 总结

本文介绍了AudioMarathon基准测试，用于评估大型音频语言模型在长音频处理方面的理解和推理效率，针对长音频处理中的主要挑战。

### 背景

处理长格式音频是大型音频语言模型(LALMs)面临的主要挑战。这些模型在处理注意力机制的二次方成本和建模长程时间依赖方面存在困难。现有的音频基准测试大多基于短片段构建，没有在现实的长上下文设置中评估模型。

### 目的

为了解决长音频评估的研究空白，引入AudioMarathon基准测试，旨在评估模型在长音频处理方面的理解和推理效率。

### 方法

AudioMarathon建立在三个支柱上：长上下文音频输入(90.0-300.0秒，对应2,250-7,500个音频标记)；语音、声音和音乐的完整领域覆盖；需要多跳推理的复杂推理任务。评估了最先进的LALMs，并研究了加速技术，分析了标记剪枝和KV缓存清除之间的权衡。

### 主要发现

随着音频长度的增加，最先进的LALMs性能明显下降。研究结果表明当前LALMs之间存在较大差距，突显了更好的时间推理和内存高效架构的需求。

### 结论

AudioMarathon将推动音频和多模态研究社区开发能够解决复杂音频任务的高级音频理解模型。

### 翻译

处理长格式音频是大型音频语言模型(LALMs)的主要挑战。这些模型在注意力机制的二次方成本和建模长程时间依赖方面存在困难。现有的音频基准测试大多基于短片段构建，没有在现实的长上下文设置中评估模型。为了解决这一研究空白，我们引入了AudioMarathon基准测试，旨在评估模型在长音频处理方面的理解和推理效率。AudioMarathon建立在三个支柱上：持续时间从90.0到300.0秒的长上下文音频输入，对应编码的2,250到7,500个音频标记；语音、声音和音乐的完整领域覆盖；以及需要多跳推理的复杂推理任务。我们评估了最先进的LALMs，并观察到随着音频长度的增加，性能明显下降。我们还研究了加速技术，分析了标记剪枝和KV缓存清除之间的权衡。结果表明当前LALMs之间存在较大差距，突显了更好的时间推理和内存高效架构的需求。我们相信AudioMarathon将推动音频和多模态研究社区开发能够解决复杂音频任务的高级音频理解模型。


### 论文摘要

Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks.

---

## 2. Online Generic Event Boundary Detection

**论文链接:** [http://arxiv.org/abs/2510.06855v1](http://arxiv.org/abs/2510.06855v1)

**作者:** Hyungrok Jung, Daneul Kim, Seunggyun Lim, Jeany Son, Jonghyun Choi

**发布时间:** 2025-10-08

**备注:** ICCV 2025

### GPT解析

### 总结

本文介绍了在线通用事件边界检测(On-GEBD)任务，提出了一种名为Estimator的新框架，包含一致性事件预测器(CEA)和在线边界鉴别器(OBD)两个关键组件。实验表明，该框架优于所有基线模型，并在标准数据集上达到与离线方法相当的性能。

### 背景

通用事件边界检测(GEBD)旨在从人类感知角度解释长视频，但现有方法需要处理完整视频帧才能预测，与人类在线实时处理数据的方式不同。

### 目的

引入在线通用事件边界检测(On-GEBD)新任务，旨在立即检测流视频中通用事件的边界，弥合当前方法与人类感知方式的差距。

### 方法

提出受事件分割理论(EST)启发的Estimator框架，包含两个组件：CEA基于先前的帧生成反映当前事件动态的未来帧预测；OBD测量预测误差并使用统计测试自适应调整阈值，以捕捉多样化的细微事件转换。

### 主要发现

实验结果表明，Estimator优于从最近的在线视频理解模型改编的所有基线模型，并在Kinetics-GEBD和TAPOS数据集上实现了与先前离线GEBD方法相当的性能。

### 结论

Estimator框架能够有效在线检测视频中的事件边界，性能接近离线方法，解决了当前GEBD方法无法实时处理视频的问题。

### 翻译

通用事件边界检测(GEBD)旨在从人类感知的角度解释长视频。然而，当前的GEBD方法需要处理完整的视频帧才能做出预测，与人类在线和实时处理数据的方式不同。为了弥合这一差距，我们引入了一个新任务——在线通用事件边界检测(On-GEBD)，旨在立即检测流视频中通用事件的边界。该任务面临在无法访问未来帧的情况下，实时识别细微的、无分类的事件变化的独特挑战。为了应对这些挑战，我们提出了一个受事件分割理论(EST)启发的新型On-GEBD框架Estimator，该理论解释了人类如何通过利用预测信息和实际信息之间的差异将进行中的活动分割为事件。我们的框架包含两个关键组件：一致性事件预测器(CEA)和在线边界鉴别器(OBD)。具体而言，CEA仅基于先前的帧生成反映当前事件动态的未来帧预测。然后，OBD测量预测误差，并使用过去误差的统计测试自适应调整阈值，以捕捉多样化的、细微的事件转换。实验结果表明，Estimator优于从最近的在线视频理解模型改编的所有基线模型，并在Kinetics-GEBD和TAPOS数据集上实现了与先前离线GEBD方法相当的性能。


### 论文摘要

Generic Event Boundary Detection (GEBD) aims to interpret long-form videos through the lens of human perception. However, current GEBD methods require processing complete video frames to make predictions, unlike humans processing data online and in real-time. To bridge this gap, we introduce a new task, Online Generic Event Boundary Detection (On-GEBD), aiming to detect boundaries of generic events immediately in streaming videos. This task faces unique challenges of identifying subtle, taxonomy-free event changes in real-time, without the access to future frames. To tackle these challenges, we propose a novel On-GEBD framework, Estimator, inspired by Event Segmentation Theory (EST) which explains how humans segment ongoing activity into events by leveraging the discrepancies between predicted and actual information. Our framework consists of two key components: the Consistent Event Anticipator (CEA), and the Online Boundary Discriminator (OBD). Specifically, the CEA generates a prediction of the future frame reflecting current event dynamics based solely on prior frames. Then, the OBD measures the prediction error and adaptively adjusts the threshold using statistical tests on past errors to capture diverse, subtle event transitions. Experimental results demonstrate that Estimator outperforms all baselines adapted from recent online video understanding models and achieves performance comparable to prior offline-GEBD methods on the Kinetics-GEBD and TAPOS datasets.

---

## 3. From Captions to Keyframes: Efficient Video Summarization via Caption- and Context-Aware Frame Scoring

**论文链接:** [http://arxiv.org/abs/2510.06509v1](http://arxiv.org/abs/2510.06509v1)

**作者:** Shih-Yao Lin, Sibendu Paul, Caren Chen

**发布时间:** 2025-10-07

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

该研究提出了一种高效的视频语言理解方法，通过选择保留语义和上下文信息的小帧集来处理长视频。

### 背景

视频语言理解需要从长视频中提取关键信息，但全帧处理效率低下，需要更有效的方法。

### 目的

开发一个能够识别最具信息量帧的框架，用于视频检索、字幕生成和视频语言推理等下游任务。

### 方法

提出了KeyScore多模态帧评分框架，结合字幕和视觉上下文估计帧级重要性；同时引入STACFP（时空自适应聚类帧提案）生成紧凑多样的帧候选。

### 主要发现

该方法相比全帧推理可实现高达99%的帧减少，在MSRVTT、MSVD和DiDeMo数据集上显著优于标准8帧编码器。

### 结论

强调视觉和文本信号之间的多模态对齐可实现可扩展、高效且基于字幕的视频理解，无需明确的视频摘要步骤。

### 翻译

高效的视频语言理解需要选择保留长视频中语义和上下文信息的小帧集。我们提出了KeyScore，一个多模态帧评分框架，联合利用字幕和视觉上下文来估计帧级重要性。通过结合语义相似性、时间多样性和上下文下降影响，KeyScore识别出用于检索、字幕生成和视频语言推理等下游任务的最具信息量的帧。为补充KeyScore，我们引入了STACFP（用于帧提案的时空自适应聚类），它为长视频生成紧凑且多样的帧候选。这些模块共同实现了相比全帧推理高达99%的帧减少，并在MSRVTT、MSVD和DiDeMo上显著优于标准8帧编码器。我们的结果表明，强调视觉和文本信号之间的多模态对齐可实现可扩展、高效且基于字幕的视频理解，无需明确的视频摘要。


### 论文摘要

Efficient video-language understanding requires selecting a small set of frames that retain semantic and contextual information from long videos. We propose KeyScore, a multimodal frame scoring framework that jointly leverages captions and visual context to estimate frame-level importance. By combining semantic similarity, temporal diversity, and contextual drop impact, KeyScore identifies the most informative frames for downstream tasks such as retrieval, captioning, and video-language reasoning. To complement KeyScore, we introduce STACFP (Spatio-Temporal Adaptive Clustering for Frame Proposals), which generates compact and diverse frame candidates for long-form videos. Together, these modules achieve up to 99\% frame reduction compared to full-frame inference and substantially outperform standard 8-frame encoders on MSRVTT, MSVD, and DiDeMo. Our results demonstrate that emphasizing multimodal alignment between visual and textual signals enables scalable, efficient, and caption-grounded video understanding -- without explicit video summarization.

---

## 4. BlockGPT: Spatio-Temporal Modelling of Rainfall via Frame-Level Autoregression

**论文链接:** [http://arxiv.org/abs/2510.06293v1](http://arxiv.org/abs/2510.06293v1)

**作者:** Cristian Meo, Varun Sarathchandran, Avijit Majhi, Shao Hung, Carlo Saccardi, Ruben Imhoff, Roberto Deidda, Remko Uijlenhoet, Justin Dauwels

**发布时间:** 2025-10-07

### GPT解析

### 总结

BlockGPT是一种新型生成式自回归transformer模型，使用批量标记化方法预测降水图，在准确性和推理速度方面均优于现有方法。

### 背景

降水预测是复杂的时空建模任务，对减轻极端天气影响至关重要。短期降水预报需要准确且计算高效的实时模型。

### 目的

解决现有方法（如基于标记的自回归模型和扩散模型）的局限性，开发一种既准确又计算高效的降水预测模型。

### 方法

BlockGPT采用批量标记化方法，在每个时间步预测完整的二维降水场，通过帧内自注意力和帧间因果注意力实现时空因子化，作为视频预测的模型无关范式实现。

### 主要发现

在KNMI和SEVIR两个数据集上评估，BlockGPT比NowcastingGPT和DiffCast+Phydnet等基线模型实现更高准确性、更好的事件定位，推理速度提高高达31倍。

### 结论

BlockGPT在降水预报任务中表现出色，在准确性、事件定位和推理速度方面均优于现有方法。

### 翻译

预测降水图是一个高度复杂的时空建模任务，对于减轻极端天气事件的影响至关重要。短期降水预报或临近预报需要不仅准确而且计算高效的模型以实现实时应用。当前方法如基于标记的自回归模型往往存在错误的归纳偏置和缓慢的推理，而扩散模型可能计算密集。为解决这些限制，我们引入了BlockGPT，一种使用批量标记化方法的生成式自回归transformer，在每个时间步预测完整的二维场。作为视频预测的模型无关范式，BlockGPT通过在每帧内使用自注意力并在帧间使用因果注意力来分解时空；在本工作中，我们将其实例化用于降水临近预报。我们在两个降水数据集上评估BlockGPT，即荷兰的KNMI和美国的SEVIR，并将其与最先进的基线模型进行比较，包括基于标记的和基于扩散的模型。结果表明，BlockGPT实现了更高的准确性，通过分类度量的指标测量的事件定位，以及比可比基线模型快高达31倍的推理速度。


### 论文摘要

Predicting precipitation maps is a highly complex spatiotemporal modeling task, critical for mitigating the impacts of extreme weather events. Short-term precipitation forecasting, or nowcasting, requires models that are not only accurate but also computationally efficient for real-time applications. Current methods, such as token-based autoregressive models, often suffer from flawed inductive biases and slow inference, while diffusion models can be computationally intensive. To address these limitations, we introduce BlockGPT, a generative autoregressive transformer using batched tokenization (Block) method that predicts full two-dimensional fields (frames) at each time step. Conceived as a model-agnostic paradigm for video prediction, BlockGPT factorizes space-time by using self-attention within each frame and causal attention across frames; in this work, we instantiate it for precipitation nowcasting. We evaluate BlockGPT on two precipitation datasets, viz. KNMI (Netherlands) and SEVIR (U.S.), comparing it to state-of-the-art baselines including token-based (NowcastingGPT) and diffusion-based (DiffCast+Phydnet) models. The results show that BlockGPT achieves superior accuracy, event localization as measured by categorical metrics, and inference speeds up to 31x faster than comparable baselines.

---

## 5. Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models

**论文链接:** [http://arxiv.org/abs/2510.05034v2](http://arxiv.org/abs/2510.05034v2)

**作者:** Yolo Yunlong Tang, Jing Bi, Pinxin Liu, Zhenyu Pan, Zhangyun Tan, Qianxiang Shen, Jiani Liu, Hang Hua, Junjia Guo, Yunzhong Xiao, Chao Huang, Zhiyuan Wang, Susan Liang, Xinyi Liu, Yizhi Song, Yuhe Nie, Jia-Xing Zhong, Bozheng Li, Daiqing Qi, Ziyun Zeng, Ali Vosoughi, Luchuan Song, Zeliang Zhang, Daiki Shimada, Han Liu, Jiebo Luo, Chenliang Xu

**发布时间:** 2025-10-06

**备注:** The 1st version

### GPT解析

### 总结

这篇综述首次全面考察了视频大多模态模型(Video-LMMs)的后训练方法，提供了统一框架来推进视频理解能力。

### 背景

视频理解是计算机视觉领域最具挑战性的前沿，需要模型推理复杂时空关系、长期依赖和多模态证据。最近出现的Video-LMMs集成了视觉编码器和基于解码器的强大语言模型，在视频理解任务中展现出卓越能力。

### 目的

填补文献中关于将Video-LMMs从基础感知系统转变为复杂推理引擎的后训练阶段研究的碎片化状态，提供首个全面的后训练方法考察。

### 方法

涵盖三个基本支柱：使用思维链的监督微调、可验证目标的强化学习、通过增强推理计算实现的测试时缩放。提供结构化分类法阐明这些技术的角色和相互连接，分析代表性方法并综合关键设计原则和评估协议，同时整理必要的基准、数据集和指标。

### 主要发现

视频理解面临独特挑战如时间定位、时空基础、长视频效率和多模态证据整合；识别出奖励设计、可扩展性和成本性能优化方面的关键开放性挑战。

### 结论

为研究人员和从业人员提供推进Video-LMM能力的统一框架，相关资源和更新可在指定GitHub仓库获取。

### 翻译

视频理解代表了计算机视觉中最具挑战性的前沿，需要模型推理复杂的时空关系、长期依赖和多模态证据。最近出现的视频大多模态模型(Video-LMMs)将视觉编码器与基于解码器的强大语言模型相结合，在视频理解任务中展示了卓越能力。然而，将这些模型从基础感知系统转变为复杂推理引擎的关键阶段——后训练，在文献中仍然分散。这篇综述首次对Video-LMMs的后训练方法进行了全面考察，涵盖三个基本支柱：使用思维链的监督微调、可验证目标的强化学习、通过增强推理计算实现的测试时缩放。我们提出了结构化分类法，阐明了这些技术的作用、相互连接和视频特定适应，解决了时间定位、时空基础、长视频效率和多模态证据集成等独特挑战。通过系统分析代表性方法，我们综合了关键设计原则、见解和评估协议，同时确定了奖励设计、可扩展性和成本性能优化方面的关键开放挑战。我们还整理了必要的基准、数据集和指标，以促进对后训练效果的严格评估。这篇综述旨在为研究人员和从业人员提供推进Video-LMM能力的统一框架。额外的资源和更新保存在：https://github.com/yunlong10/Awesome-Video-LMM-Post-Training


### 论文摘要

Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training

---

## 6. GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection (Preprint)

**论文链接:** [http://arxiv.org/abs/2510.07285v1](http://arxiv.org/abs/2510.07285v1)

**作者:** Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Qi Hu, Yan Li, Chang Liu

**发布时间:** 2025-10-08

**备注:** This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

### GPT解析

### 总结

本研究提出了一种名为GTCN-G的新型深度学习框架，用于解决网络入侵检测系统中的网络威胁复杂性和数据类别不平衡问题。该框架结合了门控时序卷积网络和图卷积网络的优势，并通过残差学习机制提高对少数类恶意活动的检测敏感性。

### 背景

现代入侵检测系统面临两大挑战：网络威胁日益复杂以及流量数据中的类别不平衡问题。图神经网络擅长建模拓扑结构，时序卷积网络能够捕捉时间序列依赖关系，但将两者有机结合并同时解决数据不平衡问题的框架仍然是一个开放性挑战。

### 目的

开发一种能够有效整合图神经网络和时序卷积网络优势，同时明确解决数据不平衡问题的入侵检测框架，以提高对罕见恶意活动（少数类）的检测敏感性和整体检测性能。

### 方法

提出了一种名为门控时序卷积网络与图（GTCN-G）的深度学习框架，该方法：1）融合门控时序卷积网络（G-TCN）用于从网络流量中提取分层时间特征；2）使用图卷积网络（GCN）从底层图结构中学习；3）集成通过图注意力网络（GAT）实现的残差学习机制，通过残差连接保留原始特征信息，以缓解类别不平衡问题。

### 主要发现

在UNSW-NB15和ToN-IoT两个公开基准数据集上的广泛实验表明，所提出的GTCN-G模型在二分类和多分类任务中都实现了最先进的性能，显著优于现有的基线模型。

### 结论

GTCN-G框架成功结合了图神经网络和时序卷积网络的优势，并通过残差学习机制有效解决了数据不平衡问题，为现代入侵检测系统提供了高效解决方案。

### 翻译

网络威胁日益复杂和流量数据中固有的类别不平衡对现代入侵检测系统构成了严峻挑战。虽然图神经网络在建模拓扑结构方面表现出色，时序卷积网络擅长捕捉时间序列依赖关系，但能够协同整合两者并同时明确解决数据不平衡问题的框架仍然是一个开放性挑战。本文引入了一种名为门控时序卷积网络与图的新型深度学习框架，旨在克服这些局限性。我们的模型独特地融合了门控时序卷积网络用于从网络流量中提取分层时间特征，以及设计用于从底层图结构中学习的图卷积网络。核心创新在于集成了通过图注意力网络实现的残差学习机制。该机制通过残差连接保留原始特征信息，这对于缓解类别不平衡问题并增强对罕见恶意活动的检测敏感性至关重要。我们在两个公开基准数据集上进行了大量实验以验证我们的方法。实验结果表明，所提出的模型实现了最先进的性能，在二分类和多分类任务中都显著优于现有基线模型。


### 论文摘要

The escalating complexity of network threats and the inherent class imbalance in traffic data present formidable challenges for modern Intrusion Detection Systems (IDS). While Graph Neural Networks (GNNs) excel in modeling topological structures and Temporal Convolutional Networks (TCNs) are proficient in capturing time-series dependencies, a framework that synergistically integrates both while explicitly addressing data imbalance remains an open challenge. This paper introduces a novel deep learning framework, named Gated Temporal Convolutional Network and Graph (GTCN-G), engineered to overcome these limitations. Our model uniquely fuses a Gated TCN (G-TCN) for extracting hierarchical temporal features from network flows with a Graph Convolutional Network (GCN) designed to learn from the underlying graph structure. The core innovation lies in the integration of a residual learning mechanism, implemented via a Graph Attention Network (GAT). This mechanism preserves original feature information through residual connections, which is critical for mitigating the class imbalance problem and enhancing detection sensitivity for rare malicious activities (minority classes). We conducted extensive experiments on two public benchmark datasets, UNSW-NB15 and ToN-IoT, to validate our approach. The empirical results demonstrate that the proposed GTCN-G model achieves state-of-the-art performance, significantly outperforming existing baseline models in both binary and multi-class classification tasks.

---

## 7. GNN-enhanced Traffic Anomaly Detection for Next-Generation SDN-Enabled Consumer Electronics

**论文链接:** [http://arxiv.org/abs/2510.07109v1](http://arxiv.org/abs/2510.07109v1)

**作者:** Guan-Yan Yang, Farn Wang, Kuo-Hui Yeh

**发布时间:** 2025-10-08

**备注:** This paper has been accepted for publication in IEEE Transactions on  Consumer Electronics. 10 pages, 6 figures

### GPT解析

### 总结

该研究提出了一种基于图神经网络的网络异常检测框架(GNN-NAD)，用于解决消费电子产品在物联网环境中面临的安全威胁问题。通过整合软件定义网络和计算优先网络架构，该框架能够有效检测网络异常并提供整体网络安全视图。

### 背景

消费电子产品连接到物联网容易受到DDoS和基于网络的攻击，这些攻击可能导致设备功能受损和远程劫持。现有基于深度学习的流量异常检测系统在传统网络环境中表现良好，但往往过于复杂且依赖静态基础设施，需要手动配置和管理。

### 目的

解决现有检测系统的局限性，提出一个可扩展的网络模型，整合软件定义网络和计算优先网络，用于增强下一代消费电子产品网络的安全性和效率。

### 方法

提出了一种基于图神经网络的网络异常检测框架(GNN-NAD)，该框架整合了基于软件定义网络的消费电子产品网络并支持计算优先网络架构。GNN-NAD融合静态的、漏洞感知的攻击图与动态流量特征，框架核心是用于图表示学习的GNN模型(GSAGE)，结合随机森林分类器，形成GSAGE+RF设计。

### 主要发现

在消费电子产品环境上的实验评估显示，GNN-NAD在准确性、召回率、精确度和F1分数等指标上表现出色。即使在样本量较小的情况下，其性能也超过了当前的网络异常检测方法。

### 结论

这项工作通过提出GNN-NAD框架，有效推进了下一代智能消费电子产品网络的安全性和效率，为物联网环境下的设备安全提供了新的解决方案。

### 翻译

消费电子产品连接到物联网容易受到各种攻击，包括DDoS和基于网络的威胁，这些攻击可能损害其功能并实现远程劫持。这些漏洞允许攻击者利用消费电子产品进行更广泛的系统攻击，同时恶意代码可以在消费电子产品网络中传播，导致设备故障。现有的基于深度学习的流量异常检测系统在传统网络环境中表现出高准确性，但往往过于复杂且依赖静态基础设施，需要手动配置和管理。为解决这些限制，我们提出了一种可扩展的网络模型，整合了软件定义网络和计算优先网络，用于下一代消费电子产品网络。在该网络模型中，我们提出了一个基于图神经网络的网络异常检测框架(GNN-NAD)，该框架整合了基于软件定义网络的消费电子产品网络并支持计算优先网络架构。GNN-NAD独特地融合了静态的、漏洞感知的攻击图与动态流量特征，提供网络安全的整体视图。框架的核心是一个用于图表示学习的GNN模型(GSAGE)，随后是随机森林分类器。这种设计相比现有的特征选择方法表现出优越性能。在消费电子产品环境上的实验评估显示，GNN-NAD在准确性、召回率、精确度和F1分数等指标上表现出色，即使在小样本情况下，也超过了当前网络异常检测方法的性能。这项工作推进了下一代智能消费电子产品网络的安全性和效率。


### 论文摘要

Consumer electronics (CE) connected to the Internet of Things are susceptible to various attacks, including DDoS and web-based threats, which can compromise their functionality and facilitate remote hijacking. These vulnerabilities allow attackers to exploit CE for broader system attacks while enabling the propagation of malicious code across the CE network, resulting in device failures. Existing deep learning-based traffic anomaly detection systems exhibit high accuracy in traditional network environments but are often overly complex and reliant on static infrastructure, necessitating manual configuration and management. To address these limitations, we propose a scalable network model that integrates Software-defined Networking (SDN) and Compute First Networking (CFN) for next-generation CE networks. In this network model, we propose a Graph Neural Networks-based Network Anomaly Detection framework (GNN-NAD) that integrates SDN-based CE networks and enables the CFN architecture. GNN-NAD uniquely fuses a static, vulnerability-aware attack graph with dynamic traffic features, providing a holistic view of network security. The core of the framework is a GNN model (GSAGE) for graph representation learning, followed by a Random Forest (RF) classifier. This design (GSAGE+RF) demonstrates superior performance compared to existing feature selection methods. Experimental evaluations on CE environment reveal that GNN-NAD achieves superior metrics in accuracy, recall, precision, and F1 score, even with small sample sizes, exceeding the performance of current network anomaly detection methods. This work advances the security and efficiency of next-generation intelligent CE networks.

---

## 8. Revisiting Node Affinity Prediction in Temporal Graphs

**论文链接:** [http://arxiv.org/abs/2510.06940v1](http://arxiv.org/abs/2510.06940v1)

**作者:** Krishna Sri Ipsit Mantri, Or Feldman, Moshe Eliasof, Chaim Baskin

**发布时间:** 2025-10-08

**备注:** preprint

### GPT解析

### 总结

本研究提出了一种名为NAViS的节点亲和力预测模型，通过利用启发式方法和状态空间模型之间的等价性，并引入新型损失函数，在TGB数据集上实现了优于现有技术和启发式方法的性能。

### 背景

节点亲和力预测是时序图学习中的常见任务，广泛应用于社交网络、金融网络和推荐系统等领域。尽管最近的研究尝试将先进的动态链接属性预测模型适应到节点亲和力预测中，但简单的启发式方法（如持久预测或移动平均）却表现更好。

### 目的

分析当前时序图神经网络在节点亲和力预测训练中面临的挑战，并提出适当的解决方案以提升预测性能。

### 方法

开发NAViS（使用虚拟状态的节点亲和力预测模型），通过结合提出的解决方案，利用启发式方法和状态空间模型之间的等价性，并引入一种专门用于节点亲和力预测的新型损失函数来解决训练难题。

### 主要发现

在TGB数据集上的评估表明，NAViS模型性能优于现有最先进的方法，包括各种启发式方法。研究源代码已在GitHub上公开分享。

### 结论

NAViS模型通过创新的虚拟状态方法和新型损失函数，成功解决了时序图神经网络在节点亲和力预测中的训练挑战，实现了比现有技术和简单启发式方法更优的性能。

### 翻译

节点亲和力预测是时序图学习中的常见任务，广泛应用于社交网络、金融网络、推荐系统等多个领域。最近的研究通过将最先进的动态链接属性预测模型适应到节点亲和力预测中来解决这一任务。然而，简单的启发式方法，如持久预测或移动平均，却优于这些模型。在本研究中，我们分析了当前时序图神经网络在节点亲和力预测训练中的挑战，并提出了适当的解决方案。结合这些解决方案，我们开发了NAViS——使用虚拟状态的节点亲和力预测模型，通过利用启发式方法和状态空间模型之间的等价性。虽然前景良好，但训练NAViS并不简单。因此，我们进一步引入了一种用于节点亲和力预测的新型损失函数。我们在TGB上评估了NAViS，结果表明它优于最先进的方法，包括启发式方法。我们的源代码可在https://github.com/orfeld415/NAVIS获取。


### 论文摘要

Node affinity prediction is a common task that is widely used in temporal graph learning with applications in social and financial networks, recommender systems, and more. Recent works have addressed this task by adapting state-of-the-art dynamic link property prediction models to node affinity prediction. However, simple heuristics, such as Persistent Forecast or Moving Average, outperform these models. In this work, we analyze the challenges in training current Temporal Graph Neural Networks for node affinity prediction and suggest appropriate solutions. Combining the solutions, we develop NAViS - Node Affinity prediction model using Virtual State, by exploiting the equivalence between heuristics and state space models. While promising, training NAViS is non-trivial. Therefore, we further introduce a novel loss function for node affinity prediction. We evaluate NAViS on TGB and show that it outperforms the state-of-the-art, including heuristics. Our source code is available at https://github.com/orfeld415/NAVIS

---

## 9. MoRE-GNN: Multi-omics Data Integration with a Heterogeneous Graph Autoencoder

**论文链接:** [http://arxiv.org/abs/2510.06880v1](http://arxiv.org/abs/2510.06880v1)

**作者:** Zhiyu Wang, Sonia Koszut, Pietro Liò, Francesco Ceccarelli

**发布时间:** 2025-10-08

### GPT解析

### 总结

多组学单细胞数据整合面临高维性和复杂跨模态关系的挑战，研究提出MoRE-GNN方法有效解决了这一问题

### 背景

多组学单细胞数据整合具有挑战性，主要因为数据具有高维性和复杂的跨模态关系

### 目的

开发一种能够有效整合多组学单细胞数据的方法，捕捉生物学上有意义的关系

### 方法

提出MoRE-GNN（多组学关系边图神经网络），这是一种异构图自编码器，结合图卷积和注意力机制，可以直接从数据动态构建关系图

### 主要发现

在六个公开数据集上的评估表明，MoRE-GNN能够捕获生物学上有意义的关系，优于现有方法，特别是在强跨模态相关性的设置中；此外，学习到的表示允许准确的下游跨模态预测

### 结论

尽管性能可能因数据集复杂性而异，但MoRE-GNN为推进多组学整合提供了一个自适应、可扩展和可解释的框架

### 翻译

多组学单细胞数据的整合由于其高维性和复杂的跨模态关系而仍然具有挑战性。为了解决这个问题，我们引入了MoRE-GNN（多组学关系边图神经网络），这是一种异构图自编码器，结合了图卷积和注意力机制，可以直接从数据中动态构建关系图。在六个公开可用数据集上的评估表明，MoRE-GNN能够捕获生物学上有意义的关系，并且优于现有方法，特别是在具有强跨模态相关性的设置中。此外，学习到的表示允许进行准确的下游跨模态预测。尽管性能可能因数据集的复杂性而有所不同，但MoRE-GNN为推进多组学整合提供了一个自适应、可扩展和可解释的框架。


### 论文摘要

The integration of multi-omics single-cell data remains challenging due to high-dimensionality and complex inter-modality relationships. To address this, we introduce MoRE-GNN (Multi-omics Relational Edge Graph Neural Network), a heterogeneous graph autoencoder that combines graph convolution and attention mechanisms to dynamically construct relational graphs directly from data. Evaluations on six publicly available datasets demonstrate that MoRE-GNN captures biologically meaningful relationships and outperforms existing methods, particularly in settings with strong inter-modality correlations. Furthermore, the learned representations allow for accurate downstream cross-modal predictions. While performance may vary with dataset complexity, MoRE-GNN offers an adaptive, scalable and interpretable framework for advancing multi-omics integration.

---

## 10. Towards Generalization of Graph Neural Networks for AC Optimal Power Flow

**论文链接:** [http://arxiv.org/abs/2510.06860v1](http://arxiv.org/abs/2510.06860v1)

**作者:** Olayiwola Arowolo, Jochen L. Cremer

**发布时间:** 2025-10-08

**备注:** Pre-print has been submitted for review

### GPT解析

### 总结

提出了一种混合异构消息传递神经网络（HH-MPNN），用于解决交流最优潮流（ACOPF）在大规模电力系统中的计算效率和拓扑适应性问题。

### 背景

ACOPF对于大规模电力系统计算成本很高，传统求解器需要过长的求解时间。机器学习方法虽能提供计算加速，但在可扩展性和拓扑适应性方面存在问题，需要昂贵的重新训练。

### 目的

实现跨电网规模的可扩展性，并使模型能够适应拓扑变化。

### 方法

提出混合异构消息传递神经网络（HH-MPNN），将母线、发电机、负载、并联电抗器、输电线路和变压器建模为不同的节点或边类型，结合可扩展的变压器模型处理长距离依赖关系。

### 主要发现

在14到2000母线的电网上，HH-MPNN在默认拓扑上实现了小于1%的最优性差距；零样本应用于数千个未见过的拓扑，实现了小于3%的最优性差距；在较小电网上预训练能改善较大电网的结果；与内点求解器相比，计算速度提升达到1000倍到10000倍。

### 结论

这些结果推动了实时电力系统操作的实用、可泛化的机器学习发展。

### 翻译

交流最优潮流（ACOPF）对于大规模电力系统计算成本很高，传统求解器需要过长的求解时间。机器学习方法虽能提供计算加速，但在可扩展性和拓扑适应性方面存在问题，需要昂贵的重新训练。为了实现跨电网规模的可扩展性和适应拓扑变化，我们提出了一种混合异构消息传递神经网络（HH-MPNN）。HH-MPNN将母线、发电机、负载、并联电抗器、输电线路和变压器建模为不同的节点或边类型，结合可扩展的变压器模型处理长距离依赖关系。在14到2000母线的电网上，HH-MPNN在默认拓扑上实现了小于1%的最优性差距。零样本应用于数千个未见过的拓扑，HH-MPNN实现了小于3%的最优性差距，尽管仅在默认拓扑上训练。在较小电网上预训练也能改善较大电网的结果。与内点求解器相比，计算速度提升达到1000倍到10000倍。这些结果推动了实时电力系统操作的实用、可泛化的机器学习发展。


### 论文摘要

AC Optimal Power Flow (ACOPF) is computationally expensive for large-scale power systems, with conventional solvers requiring prohibitive solution times. Machine learning approaches offer computational speedups but struggle with scalability and topology adaptability without expensive retraining. To enable scalability across grid sizes and adaptability to topology changes, we propose a Hybrid Heterogeneous Message Passing Neural Network (HH-MPNN). HH-MPNN models buses, generators, loads, shunts, transmission lines and transformers as distinct node or edge types, combined with a scalable transformer model for handling long-range dependencies. On grids from 14 to 2,000 buses, HH-MPNN achieves less than 1% optimality gap on default topologies. Applied zero-shot to thousands of unseen topologies, HH-MPNN achieves less than 3% optimality gap despite training only on default topologies. Pre-training on smaller grids also improves results on a larger grid. Computational speedups reach 1,000x to 10,000x compared to interior point solvers. These results advance practical, generalizable machine learning for real-time power system operations.

---

## 11. Soft-Evidence Fused Graph Neural Network for Cancer Driver Gene Identification across Multi-View Biological Graphs

**论文链接:** [http://arxiv.org/abs/2510.06290v1](http://arxiv.org/abs/2510.06290v1)

**作者:** Bang Chen, Lijun Guo, Houli Fan, Wentao He, Rong Zhang

**发布时间:** 2025-10-07

**备注:** 8pages

### GPT解析

### 总结

本文提出了Soft-Evidence Fusion Graph Neural Network (SEFGNN)框架，用于在多个生物网络中识别癌症驱动基因(CDGs)。该框架在决策层面而非特征层面融合多网络信息，使用Dempster-Shafer理论进行不确定性感知融合，并引入软证据平滑(SES)模块提高排名稳定性。实验表明SEFGNN在三个癌症数据集上优于现有方法，且在发现新型CDGs方面具有潜力。

### 背景

识别癌症驱动基因(CDGs)对理解癌症机制和开发靶向治疗至关重要。图神经网络(GNNs)已被用于通过捕获生物相互作用网络中的模式来识别CDGs，但大多数方法仅依赖单一蛋白质-蛋白质相互作用(PPI)网络，忽略了其他生物网络的互补信息。一些研究通过特征对齐整合多网络，但这种方法假设跨网络基因关系一致，可能忽略网络异质性并引入冲突信息。

### 目的

开发一个能够在多个网络上识别CDGs的新框架，解决现有方法在网络异质性和冲突信息处理上的不足，通过在决策层面而非特征层面融合多网络信息，提高CDG识别准确性和发现新CDGs的能力。

### 方法

提出Soft-Evidence Fusion Graph Neural Network (SEFGNN)框架，将每个生物网络视为独立证据源，在决策层面使用Dempster-Shafer理论(DST)进行不确定性感知融合，并引入软证据平滑(SES)模块来减轻DST过度自信的风险，提高排名稳定性同时保留判别性能。

### 主要发现

在三个癌症数据集上的实验表明，SEFGNN始终优于最先进的基线方法；SEFGNN在发现新型CDGs方面展现出强大潜力；通过在决策层面融合多网络信息，能够更好地处理网络异质性；SES模块有效提高了排名稳定性而不影响判别性能。

### 结论

SEFGNN是一个有效的多网络CDG识别框架，通过决策层面融合多网络信息，能够更好地处理网络异质性和冲突信息；引入的SES模块有效提高了方法稳定性；该方法在发现新型CDGs方面具有应用前景。

### 翻译

识别癌症驱动基因(CDGs)对于理解癌症机制和开发靶向治疗至关重要。图神经网络(GNNs)最近已被用于通过捕获生物相互作用网络中的模式来识别CDGs。然而，大多数基于GNN的方法依赖于单一的蛋白质-蛋白质相互作用(PPI)网络，忽略了其他生物网络中的互补信息。一些研究通过将特征与一致性约束对齐来整合多个网络，学习统一的基因表示用于CDG识别。然而，这种表示级别的融合通常假设跨网络中的基因关系是一致的，这可能忽略网络异质性并引入冲突信息。为解决这一问题，我们提出了Soft-Evidence Fusion Graph Neural Network (SEFGNN)，这是一个在多个网络上识别CDGs的新型框架，在决策层面而非特征层面进行融合。SEFGNN不强制特征级别的一致性，而是将每个生物网络视为独立的证据源，并使用Dempster-Shafer理论(DST)在决策层面进行不确定性感知融合。为了减轻DST的过度自信风险，我们进一步引入了软证据平滑(SES)模块，它在保持判别性能的同时提高了排名稳定性。在三个癌症数据集上的实验表明，SEFGNN始终优于最先进的基线方法，并在发现新型CDGs方面展现出强大潜力。


### 论文摘要

Identifying cancer driver genes (CDGs) is essential for understanding cancer mechanisms and developing targeted therapies. Graph neural networks (GNNs) have recently been employed to identify CDGs by capturing patterns in biological interaction networks. However, most GNN-based approaches rely on a single protein-protein interaction (PPI) network, ignoring complementary information from other biological networks. Some studies integrate multiple networks by aligning features with consistency constraints to learn unified gene representations for CDG identification. However, such representation-level fusion often assumes congruent gene relationships across networks, which may overlook network heterogeneity and introduce conflicting information. To address this, we propose Soft-Evidence Fusion Graph Neural Network (SEFGNN), a novel framework for CDG identification across multiple networks at the decision level. Instead of enforcing feature-level consistency, SEFGNN treats each biological network as an independent evidence source and performs uncertainty-aware fusion at the decision level using Dempster-Shafer Theory (DST). To alleviate the risk of overconfidence from DST, we further introduce a Soft Evidence Smoothing (SES) module that improves ranking stability while preserving discriminative performance. Experiments on three cancer datasets show that SEFGNN consistently outperforms state-of-the-art baselines and exhibits strong potential in discovering novel CDGs.

---

## 12. Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2510.07316v1](http://arxiv.org/abs/2510.07316v1)

**作者:** Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Sida Peng, Xin Yang

**发布时间:** 2025-10-08

**备注:** NeurIPS 2025. Project page: https://pixel-perfect-depth.github.io/

### GPT解析

### 总结

本文提出了Pixel-Perfect Depth模型，一种基于像素空间扩散生成的单目深度估计模型，可直接生成高质量无飞像素的点云，避免了传统VAE压缩带来的边缘伪影问题。

### 背景

当前生成式深度估计模型通过微调Stable Diffusion实现，但需要使用VAE将深度图压缩到潜在空间，这会在边缘和细节处引入飞像素伪影，影响点云质量。

### 目的

开发一种直接在像素空间进行扩散生成的深度估计模型，避免VAE诱导的伪影，生成高质量、无飞像素的点云，并提高效率和准确性。

### 方法

提出两种新颖设计：1) 语义提示扩散变换器(Sp-DiT)，将视觉基础模型的语义表示整合到DiT中以提示扩散过程；2) 级联DiT设计，逐步增加token数量以提高效率和准确性。

### 主要发现

该模型在五个基准测试的所有已发布生成模型中取得了最佳性能，并且在边缘感知点云评估中显著优于所有其他模型，证明了其有效性和优越性。

### 结论

Pixel-Perfect Depth模型通过直接在像素空间进行扩散生成，成功避免了VAE诱导的伪影，结合语义提示和级联设计，实现了高质量深度估计和点云生成，为单目深度估计提供了新的有效方法。

### 翻译

本文提出了Pixel-Perfect Depth，一种基于像素空间扩散生成的单目深度估计模型，可以从估计的深度图中生成高质量、无飞像素的点云。当前生成式深度估计模型通过微调Stable Diffusion实现，取得了令人印象深刻的性能。然而，它们需要使用VAE将深度图压缩到潜在空间，这不可避免地在边缘和细节处引入飞像素。我们的模型直接在像素空间执行扩散生成，避免了VAE引起的伪影。为了克服与像素空间生成相关的高复杂度，我们引入了两种新颖设计：1) 语义提示扩散变换器(Sp-DiT)，将视觉基础模型的语义表示整合到DiT中以提示扩散过程，从而保持全局语义一致性的同时增强细粒度视觉细节；2) 级联DiT设计，逐步增加token数量以进一步提高效率和准确性。我们的模型在五个基准测试的所有已发布生成模型中取得了最佳性能，并且在边缘感知点云评估中显著优于所有其他模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决单目深度估计中的'飞像素'问题，即深度图转换为点云时在物体边缘和细节处产生的几何伪影。这个问题很重要，因为深度估计是3D重建、新视角合成和机器人操作等应用的基础，飞像素问题限制了这些技术在自由视角广播、机器人操作和沉浸式内容创建等实际场景中的应用效果。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析发现现有生成式深度模型使用VAE压缩深度图会导致飞像素问题，尝试直接在像素空间进行扩散生成。但发现这种方法计算复杂且难以优化，于是借鉴高分辨率图像生成研究，认识到主要困难在于建模全局图像结构。基于此，作者设计了Pixel-Perfect Depth框架，整合了视觉基础模型的语义提示和级联变压器设计。作者借鉴了扩散模型（特别是Flow Matching）、Transformer架构（DiT）和多种视觉基础模型（如DINOv2、Depth Anything v2）等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是直接在像素空间进行扩散生成，避免使用VAE压缩，同时通过语义提示和级联设计解决像素空间扩散的计算和优化挑战。整体流程：1)输入图像；2)用视觉基础模型提取高级语义表示；3)对语义进行归一化；4)采用Flow Matching进行扩散生成，早期使用大补丁尺寸建模全局结构，后期增加标记数生成细节；5)通过迭代去噪将噪声转换为最终深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)直接在像素空间进行扩散生成，避免VAE压缩导致的飞像素；2)Semantics-Prompted DiT整合高级语义提示，保留全局结构同时增强细节；3)Cascade DiT Design采用渐进式补丁大小策略，提高效率和准确性；4)提出边缘感知点云评估指标。相比之前工作，不依赖预训练Stable Diffusion，从头训练仍能实现优越性能；与判别式模型相比保留边缘清晰度；与其他生成式模型相比避免VAE伪影；采用纯Transformer架构，无卷积层。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Pixel-Perfect Depth通过直接在像素空间进行语义提示的扩散生成，结合级联变压器设计，实现了高质量且无飞像素的单目深度估计，显著优于现有生成式模型在点云质量方面的表现。'}


### 论文摘要

This paper presents Pixel-Perfect Depth, a monocular depth estimation model based on pixel-space diffusion generation that produces high-quality, flying-pixel-free point clouds from estimated depth maps. Current generative depth estimation models fine-tune Stable Diffusion and achieve impressive performance. However, they require a VAE to compress depth maps into latent space, which inevitably introduces \textit{flying pixels} at edges and details. Our model addresses this challenge by directly performing diffusion generation in the pixel space, avoiding VAE-induced artifacts. To overcome the high complexity associated with pixel-space generation, we introduce two novel designs: 1) Semantics-Prompted Diffusion Transformers (SP-DiT), which incorporate semantic representations from vision foundation models into DiT to prompt the diffusion process, thereby preserving global semantic consistency while enhancing fine-grained visual details; and 2) Cascade DiT Design that progressively increases the number of tokens to further enhance efficiency and accuracy. Our model achieves the best performance among all published generative models across five benchmarks, and significantly outperforms all other models in edge-aware point cloud evaluation.

---

## 13. WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.07313v1](http://arxiv.org/abs/2510.07313v1)

**作者:** Zezhong Qian, Xiaowei Chi, Yuming Li, Shizun Wang, Zhiyuan Qin, Xiaozhu Ju, Sirui Han, Shanghang Zhang

**发布时间:** 2025-10-08

### GPT解析

### 总结

WristWorld是首个仅从锚定视角生成腕部视角视频的4D世界模型，通过结合几何和跨视角先验解决极端视角转换问题，在多个数据集上实现了最先进的视频生成效果，并提高了VLA性能。

### 背景

腕部视角观察对VLA模型至关重要，能捕捉精细的手物交互增强操作性能，但大规模数据集很少包含此类记录，导致丰富的锚定视角与稀缺的腕部视角之间存在显著差距。现有世界模型无法仅从锚定视角生成腕部视角视频。

### 目的

开发一种能够仅从锚定视角生成腕部视角视频的模型，解决锚定视角与腕部视角之间的差距问题，提升VLA模型的操作性能。

### 方法

WristWorld在两个阶段运行：1)重建阶段，扩展VGGT并融入空间投影一致性损失，估计几何一致的腕部视角姿态和4D点云；2)生成阶段，使用视频生成模型从重建视角合成时间连贯的腕部视角视频。

### 主要发现

在Droid、Calvin和Franka Panda数据集上的实验展示了最先进的视频生成效果，具有优越的空间一致性；同时提高了VLA性能，在Calvin上将平均任务完成长度提高了3.81%，缩小了42.4%的锚定-腕部视角差距。

### 结论

WristWorld成功解决了仅从锚定视角生成腕部视角视频的挑战，通过结合几何和跨视角先验，实现了时间连贯且空间一致的腕部视角视频生成，显著提升了VLA模型的操作性能。

### 翻译

腕部视角观察对VLA模型至关重要，因为它们捕捉了精细的手物交互，直接增强操作性能。然而，大规模数据集很少包含此类记录，导致丰富的锚定视角与稀缺的腕部视角之间存在显著差距。现有的世界模型无法弥合这一差距，因为它们需要腕部视角的第一帧，因此无法仅从锚定视角生成腕部视角视频。面对这一差距，最近的视觉几何模型如VGGT出现，具有几何和跨视角先验，使得解决极端视角转换成为可能。受这些见解启发，我们提出了WristWorld，这是首个仅从锚定视角生成腕部视角视频的4D世界模型。WristWorld在两个阶段运行：(i)重建，扩展VGGT并融入我们的空间投影一致性损失，以估计几何一致的腕部视角姿态和4D点云；(ii)生成，使用我们的视频生成模型从重建的视角合成时间连贯的腕部视角视频。在Droid、Calvin和Franka Panda上的实验展示了最先进的视频生成，具有优越的空间一致性，同时提高了VLA性能，在Calvin上将平均任务完成长度提高了3.81%，并缩小了42.4%的锚定-腕部视角差距。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从丰富的外部视角(anchor views)数据生成稀缺的手腕视角(wrist-view)视频数据。这个问题很重要，因为手腕视角能直接捕捉精细的手-物体交互，对提高机器人操作性能至关重要；而收集大规模手腕视角数据成本高昂，需要额外传感器和精确校准；现有世界模型无法填补这一视角差距，因为它们通常需要手腕视角的第一帧作为条件。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了手腕视角数据稀缺但重要的问题，发现现有世界模型无法从外部视角生成手腕视角视频。他们受到视觉几何模型(如VGGT)和基于扩散的视频生成方法的启发，设计了一个两阶段框架：重建阶段扩展VGGT并融入空间投影一致性(SPC)损失来估计几何一致的手腕姿态和4D点云；生成阶段使用扩散模型结合重建的几何条件和CLIP编码的外部视角语义来合成时间连贯的手腕视角视频。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个4D世界模型，能够仅从外部视角生成手腕视角视频，同时确保几何一致性和时间连贯性。整体流程分为两阶段：1)重建阶段：使用扩展的VGGT模型估计手腕相机姿态，应用SPC损失监督几何一致性，重建3D场景并投影到手腕视角形成条件图；2)生成阶段：基于扩散的视频生成模型以手腕视角投影条件和CLIP编码的外部视角特征为条件，生成时间连贯且几何一致的手腕视角视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个能仅从外部视角生成手腕视角视频的4D世界模型；2)专门的手腕头部设计和空间投影一致性(SPC)损失确保几何一致性；3)条件图生成提供结构指导；4)结合CLIP编码的外部视角语义弥补全局语义缺失。相比之前的工作，WristWorld不需要手腕视角的第一帧作为条件，同时兼顾几何一致性和时间连贯性，可作为即插即用模块扩展单视角世界模型，并在多个数据集上证明了优越性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'WristWorld提出了一种创新的两阶段4D世界模型，能够仅从外部视角生成几何一致且时间连贯的手腕视角视频，显著提升了视觉-语言-动作模型的操作性能，并可作为即插即用模块扩展现有单视角世界模型为多视角能力。'}


### 论文摘要

Wrist-view observations are crucial for VLA models as they capture fine-grained hand-object interactions that directly enhance manipulation performance. Yet large-scale datasets rarely include such recordings, resulting in a substantial gap between abundant anchor views and scarce wrist views. Existing world models cannot bridge this gap, as they require a wrist-view first frame and thus fail to generate wrist-view videos from anchor views alone. Amid this gap, recent visual geometry models such as VGGT emerge with geometric and cross-view priors that make it possible to address extreme viewpoint shifts. Inspired by these insights, we propose WristWorld, the first 4D world model that generates wrist-view videos solely from anchor views. WristWorld operates in two stages: (i) Reconstruction, which extends VGGT and incorporates our Spatial Projection Consistency (SPC) Loss to estimate geometrically consistent wrist-view poses and 4D point clouds; (ii) Generation, which employs our video generation model to synthesize temporally coherent wrist-view videos from the reconstructed perspective. Experiments on Droid, Calvin, and Franka Panda demonstrate state-of-the-art video generation with superior spatial consistency, while also improving VLA performance, raising the average task completion length on Calvin by 3.81% and closing 42.4% of the anchor-wrist view gap.

---

## 14. MV-Performer: Taming Video Diffusion Model for Faithful and Synchronized Multi-view Performer Synthesis

**论文链接:** [http://arxiv.org/abs/2510.07190v1](http://arxiv.org/abs/2510.07190v1)

**作者:** Yihao Zhi, Chenghong Li, Hongjie Liao, Xihe Yang, Zhengwentai Sun, Jiahao Chang, Xiaodong Cun, Wensen Feng, Xiaoguang Han

**发布时间:** 2025-10-08

**备注:** Accepted by SIGGRAPH Asia 2025 conference track

### GPT解析

### 总结

本文提出了MV-Performer框架，一种用于从单目全身捕获中创建同步新视角视频的创新方法，专注于人体中心的4D新视角合成，实现了360度视角变化。

### 背景

视频生成领域的最新突破表明，视频扩散模型可以作为隐式的4D新视角合成器，但当前方法主要集中在前视图的相机轨迹重定向，难以生成360度视角变化。

### 目的

专注于人体为中心的子领域，提出MV-Performer框架，用于从单目全身捕获中创建同步的新视角视频，实现360度合成。

### 方法

充分利用MVHumanNet数据集，引入信息丰富的条件信号，使用从定向部分点云渲染的相机依赖法线图减轻观察歧义，提出多视角人体为中心的视频扩散模型融合不同信息源，并提供针对野外视频案例的鲁棒推理程序。

### 主要发现

在三个数据集上的大量实验证明了MV-Performer的最先进有效性和鲁棒性，为人体中心的4D新视角合成建立了强大的模型。

### 结论

MV-Performer成功解决了当前方法在360度视角变化方面的局限性，在人体中心的4D新视角合成方面表现出色。

### 翻译

最近在视频生成领域的突破，由大规模数据集和扩散技术驱动，表明视频扩散模型可以作为隐式的4D新视角合成器。然而，当前方法主要集中在前视图内重定向相机轨迹，同时难以生成360度视角变化。在本文中，我们专注于人体为中心的子领域，提出了MV-Performer，一种用于从单目全身捕获中创建同步新视角视频的创新框架。为了实现360度合成，我们充分利用了MVHumanNet数据集并引入了信息丰富的条件信号。具体来说，我们使用从定向部分点云渲染的相机依赖法线图，有效减轻了可见和不可见观察之间的歧义。为了保持生成视频的同步性，我们提出了一个多视角人体为中心的视频扩散模型，融合参考视频、部分渲染和不同视角的信息。此外，我们为野外视频案例提供了鲁棒的推理程序，大大减轻了不完善单目深度估计引起的伪影。在三个数据集上的大量实验证明了我们MV-Performer的最先进有效性和鲁棒性，为人体中心的4D新视角合成建立了强大的模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单目视频中生成同步且一致的360度多视角人体表演视频的问题。这个问题在现实和研究中很重要，因为现有的多视角人体重建通常需要昂贵复杂的多视角相机系统，限制了应用范围。而单目输入的多视角合成可以大大降低数据采集成本，在媒体创作、虚拟现实、电影制作、自由视角视频和沉浸式体验等领域有广泛应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：当前方法难以处理360度视角变化，基于扩散模型的方法在处理大运动和保持时间细节方面有局限，而基于深度扭曲的方法在大视角变化时会产生不准确效果。作者借鉴了视频扩散模型（特别是WAN2.1）、基于深度的扭曲范式、多视角注意力和参考注意力机制等现有工作，但针对人体场景的特殊性进行了改进：使用MVHumanNet数据集，提出相机依赖法线图条件解决大视角歧义，设计多视角人体视频扩散模型确保同步性，并提供鲁棒推理过程减轻深度估计误差。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视频扩散模型作为强大生成器，结合基于深度的扭曲处理视角变化，引入相机依赖法线图作为条件信号区分可见和不可见区域，并通过多视角和参考注意力机制确保生成视频的同步性和一致性。整体流程：1)输入单目正面视频；2)使用MegaSaM和Sapiens估计深度和法线；3)对齐并优化深度；4)渲染部分几何条件和法线图；5)使用3D VAE编码特征；6)通过修改的DiT模型进行去噪，包含参考注意力和同步注意力；7)输出多视角同步视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个将单目视频转换为密集多视角视频的生成框架；2)提出由法线图指导的多视角视频扩散模型，实现大视角变化下的合成；3)相机依赖法线图条件解决大视角歧义；4)鲁棒推理过程减轻深度估计误差。相比之前工作：1)专门针对人体场景使用MVHumanNet数据集；2)使用显式几何先验而非隐式相机嵌入；3)通过注意力机制确保多视角同步性；4)针对不完善深度估计提供优化过程；5)实现真正的360度全方位合成而非小范围视角变化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MV-Performer通过创新的相机依赖法线图条件和多视角注意力机制，首次实现了从单目视频中生成同步且一致的360度多视角人体表演视频，为沉浸式VR/AR和自由视角视频应用提供了强大的新工具。'}


### 论文摘要

Recent breakthroughs in video generation, powered by large-scale datasets and diffusion techniques, have shown that video diffusion models can function as implicit 4D novel view synthesizers. Nevertheless, current methods primarily concentrate on redirecting camera trajectory within the front view while struggling to generate 360-degree viewpoint changes. In this paper, we focus on human-centric subdomain and present MV-Performer, an innovative framework for creating synchronized novel view videos from monocular full-body captures. To achieve a 360-degree synthesis, we extensively leverage the MVHumanNet dataset and incorporate an informative condition signal. Specifically, we use the camera-dependent normal maps rendered from oriented partial point clouds, which effectively alleviate the ambiguity between seen and unseen observations. To maintain synchronization in the generated videos, we propose a multi-view human-centric video diffusion model that fuses information from the reference video, partial rendering, and different viewpoints. Additionally, we provide a robust inference procedure for in-the-wild video cases, which greatly mitigates the artifacts induced by imperfect monocular depth estimation. Extensive experiments on three datasets demonstrate our MV-Performer's state-of-the-art effectiveness and robustness, setting a strong model for human-centric 4D novel view synthesis.

---

## 15. The Stage Comes to You: A Real-Time Tele-Immersive System with 3D Point Clouds and Vibrotactile Feedback

**论文链接:** [http://arxiv.org/abs/2510.07009v1](http://arxiv.org/abs/2510.07009v1)

**作者:** Takahiro Matsumoto, Takahiro Kusabuka, Hiroshi Chigira, Kazuhiko Murasaki, Kakagu Komazaki, Masafumi Suzuki, Masakatsu Aoki

**发布时间:** 2025-10-08

**备注:** 2 pages, 1 figure. Accepted for presentation at SIGGRAPH Asia 2025  Posters. The final version will appear in the ACM Digital Library

### GPT解析

### 总结

一种低延迟的远程沉浸式娱乐系统，能够传输3D点云和脚步振动，创造舞台存在的感受。

### 背景

远程沉浸式娱乐需要解决实时传输、渲染和触觉反馈的技术挑战。

### 目的

开发一种低延迟的远程沉浸式娱乐系统，使远程观众能够获得接近现场表演的体验。

### 方法

在快速变化的灯光下捕获动态点云，使用可穿戴加速度计感知脚步振动，通过总延迟小于100毫秒的系统进行处理、传输和渲染，使用大型3D LED墙和振动地板为远程观众提供视觉和触觉反馈。

### 主要发现

系统成功实现了小于100毫秒的总延迟，在相距20公里的地点间进行了有效连接，观众能够观看现场舞蹈表演并与表演者实时互动而无明显延迟。

### 结论

该系统能够有效创造远程沉浸式娱乐体验，使远程观众获得接近现场表演的感受。

### 翻译

我们提出了一种低延迟的远程沉浸式娱乐系统，该系统传输3D点云和表演者的脚步振动，创造舞台存在的感受。移动的表演者和周围环境在快速变化的灯光下被捕获为动态点云，然后在总延迟小于100毫秒的情况下进行处理、传输和渲染。在高环境噪声下，脚步振动通过可穿戴加速度计感知。实时视觉和触觉流被传送到远程场地，在那里大型3D LED墙和高效的振动地板环绕数十名观众。在2025年世博会上进行的公开试验连接了相距20公里的地点：观众观看了现场舞蹈表演，与表演者交谈，没有明显延迟。


### 论文摘要

We present a low-latency tele-immersive entertainment system that streams 3D point clouds and performers' footstep vibrations, creating the sense that the stage is present. Moving performers and their surroundings are captured as dynamic point clouds under rapidly changing lighting, then processed, transmitted, and rendered within a total latency of less than 100 ms. Under high ambient noise, footstep vibrations are sensed by wearable accelerometers. Real-time visual and haptic streams are delivered to a remote venue, where a large 3D LED wall and a vibration-efficient haptic floor envelop dozens of spectators. A public trial at Expo 2025 linked sites 20 km apart: visitors watched a live dance show and conversed with performers without noticeable delay.

---

## 16. Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion

**论文链接:** [http://arxiv.org/abs/2510.06687v1](http://arxiv.org/abs/2510.06687v1)

**作者:** Jie Luo, Yuxuan Jiang, Xin Jin, Mingyu Liu, Yihui Fan

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种多模态语义分割方法，通过整合光场和激光雷达数据来解决自动驾驶中遮挡等复杂条件下的场景理解挑战。作者创建了首个结合光场和点云数据的多模态语义分割数据集，并提出了Mlpfseg网络，包含特征补全和深度感知模块，实验表明该方法比单一模态方法表现更优。

### 背景

语义分割是自动驾驶场景理解的基础，但在遮挡等复杂条件下仍面临显著挑战。光场和激光雷达模态提供了互补的视觉和空间线索，有助于鲁棒感知，但它们的有效融合受到视角多样性有限和模态差异固有问题的阻碍。

### 目的

解决多模态数据（光场和激光雷达）在语义分割中的有效融合问题，特别是在处理遮挡等复杂场景时，提高自动驾驶系统的场景理解能力。

### 方法

作者提出了首个整合光场数据和点云数据的多模态语义分割数据集。基于此数据集，设计了Mlpfseg网络（多模态光场点云融合分割网络），包含特征补全模块和深度感知模块。特征补全模块通过点云特征图的差分重建解决点云和图像像素之间的密度不匹配问题；深度感知模块通过强化注意力分数来提高对遮挡物体的分割效果。

### 主要发现

该方法比仅使用图像的分割方法高出1.71%的平均交并比(mIoU)，比仅使用点云的分割方法高出2.38% mIoU，证明了其有效性。

### 结论

通过整合光场和激光雷达数据，并采用特征补全和深度感知技术，可以显著提高在复杂条件（如遮挡）下的语义分割性能，为自动驾驶场景理解提供了更有效的解决方案。

### 翻译

语义分割作为自动驾驶场景理解的基石，但在遮挡等复杂条件下仍面临重大挑战。光场和激光雷达模态提供了互补的视觉和空间线索，有利于鲁棒感知；然而，它们的有效融合受到视角多样性有限和固有模态差异的阻碍。为解决这些挑战，本文提出了首个整合光场数据和点云数据的多模态语义分割数据集。基于此数据集，我们提出了多模态光场点云融合分割网络(Mlpfseg)，结合特征补全和深度感知，同时对相机图像和激光雷达点云进行分割。特征补全模块通过执行点云特征图的差分重建，解决点云与图像像素之间的密度不匹配问题，增强这些模态的融合。深度感知模块通过强化注意力分数来提高对遮挡物体的分割效果。我们的方法比仅图像分割高出1.71%平均交并比(mIoU)，比仅点云分割高出2.38% mIoU，证明了其有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决语义分割在复杂条件（如遮挡）下面临的挑战，特别是如何有效融合光场和激光雷达两种模态数据以提高分割准确性。这个问题在自动驾驶领域至关重要，因为准确的场景理解是安全驾驶的基础，而遮挡物体和小物体在实际交通场景中很常见，现有单一模态方法（图像或点云）各有局限性，难以有效处理这些情况。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括单一模态数据的不足和多模态融合的不充分。他们创建了首个结合光场和激光雷达的多模态数据集TrafficScene，使用3×3相机阵列提供更丰富的视角信息。在设计Mlpfseg网络时，借鉴了HRNet-48用于图像特征提取、SPVCNN用于点云特征提取，以及Mseg3D的多尺度特征融合方法。同时创新性地设计了点-像素特征融合模块解决密度不匹配问题，以及深度差异感知模块处理遮挡问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过同时分割光场图像和激光雷达点云两种模态数据，充分利用它们的优势互补，并解决点云稀疏性和遮挡感知问题。整体流程包括：1)双分支特征提取（光场图像分支用HRNet-48，点云分支用SPVCNN）；2)点-像素特征融合模块(PFFM)将点云特征投影到图像并进行插值填补；3)深度差异感知模块(DDPM)通过比较预测和真实深度图识别遮挡区域；4)融合特征输入分割头进行最终预测；5)计算多任务损失函数优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个多模态光场-激光雷达语义分割数据集TrafficScene，提供全视角标注；2)Mlpfseg网络架构，首次实现同时分割两种模态；3)点-像素特征融合模块(PFFM)，解决点云稀疏性问题；4)深度差异感知模块(DDPM)，专门处理遮挡物体。相比之前工作，不同之处在于：现有多模态方法通常只针对单一模态进行分割，且未充分考虑点云稀疏性和遮挡问题；而本文方法通过同时处理两种模态和专门设计的创新模块，显著提高了被遮挡物体和小物体的分割准确性，实验显示mIoU比之前方法提高1.71-2.38。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了首个结合光场图像和激光雷达点云的多模态语义分割数据集和网络，通过创新的特征融合和深度感知模块，显著提高了在复杂交通场景下特别是被遮挡物体和小物体的语义分割准确性。'}


### 论文摘要

Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; however, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential reconstruction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image-only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness.

---

## 17. Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation

**论文链接:** [http://arxiv.org/abs/2510.06582v1](http://arxiv.org/abs/2510.06582v1)

**作者:** Fei Zhang, Rob Chancia, Josie Clapp, Amirhossein Hassanzadeh, Dimah Dera, Richard MacKenzie, Jan van Aardt

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种半自动的、不确定性感知的管道，用于地面激光扫描点云的语义分割，通过球形投影、特征增强、集成学习和目标标注相结合，减少标注工作量同时保持高准确性，并构建了Mangrove3D数据集。

### 背景

准确的地面激光扫描点云语义分割受到昂贵的手动标注限制，这限制了大规模应用的可能性。

### 目的

开发减少标注工作量的半自动管道，构建红树林森林的语义分割TLS数据集，并评估数据效率和特征重要性。

### 方法

将3D点投影到2D球形网格，使用多源特征丰富像素，训练集成分割网络生成伪标签和不确定性地图，通过不确定性地图指导模糊区域标注，将2D输出投影回3D，开发三层可视化套件，构建Mangrove3D数据集，评估数据效率和特征重要性，进行跨数据集测试验证泛化能力。

### 主要发现

性能在约12个标注扫描后趋于饱和，几何特征贡献最大，紧凑的九通道堆叠几乎捕获所有判别能力，平均交并比稳定在约0.76，特征增强策略在跨数据集测试中具有泛化能力。

### 结论

研究贡献包括不确定性感知的TLS标注管道与可视化工具、Mangrove3D数据集、数据效率和特征重要性的经验指导，为生态监测等领域的可扩展高质量TLS点云分割提供支持。

### 翻译

地面激光扫描点云的准确语义分割受到昂贵的手动标注限制。我们提出了一种半自动的、不确定性感知的管道，结合球形投影、特征增强、集成学习和目标标注，以减少标注工作量同时保持高准确性。我们的方法将3D点投影到2D球形网格，使用多源特征丰富像素，并训练集成分割网络生成伪标签和不确定性地图，后者指导模糊区域标注。2D输出被投影回3D，产生由三层可视化套件支持的密集标注点云，用于快速分类和审阅者指导。使用此管道，我们构建了Mangrove3D数据集。我们进一步评估数据效率和特征重要性，解决需要多少标注数据以及哪些特征最重要的问题。结果表明，性能在约12个标注扫描后趋于饱和，几何特征贡献最大，紧凑的九通道堆叠几乎捕获所有判别能力，平均交并比稳定在约0.76。最后，我们通过跨数据集测试确认了特征增强策略的泛化能力。研究贡献包括不确定性感知的TLS标注管道、Mangrove3D数据集、数据效率和特征重要性的经验指导，以及公开的数据集和处理脚本。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决地面激光扫描(TLS)点云语义分割中高质量手动标注成本高昂的问题。这个问题在生态监测和林业研究中至关重要，因为准确的点云分割是提取树木指标、生物量和栖息地特征的基础，但生态场景中严重的遮挡、不规则几何和交织的树结构使得手动标注极其困难和耗时，限制了自动化分析在林业和环境监测中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法的局限性：3D标注工具主要针对城市环境，球面投影方法主要用于自动驾驶，主动学习和自训练方法主要在RGB丰富的室内数据集上应用，特征融合方法依赖多传感器数据。在此基础上，作者借鉴了球面投影、主动学习、自训练和特征融合等现有工作，但进行了专门改进：扩展为多通道特征表示，专为没有配准图像的TLS设计，结合主动学习和自训练策略，并创建了专门针对红树林生态系统的数据集和三层次可视化工具。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将3D点云投影到2D球面网格降低标注难度，使用多源特征丰富表示增强模型理解，利用集成学习产生伪标签和不确定性图指导标注人员关注不确定性高的区域，结合主动学习和自训练减少人工标注工作量同时保持高准确性。整体流程分为三阶段：1)球面投影和可视化：将3D点云投影到2D球面网格，组织多通道特征，创建三层次可视化工具；2)混合标注：使用少量手动标注训练集成模型，生成分割掩码和不确定性图，高不确定性区域手动精化，高置信度预测保留为伪标签；3)3D空间反向投影和精化：将2D分割掩码反向投影到3D点云，应用几何平滑和特征驱动修复，创建紧凑虚拟球体用于快速检查。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)不确定性感知的TLS标注流程，首次将主动学习和自训练结合应用于全分辨率TLS球面投影图；2)多通道特征丰富表示，专为TLS设计不依赖外部图像；3)三层次可视化工具，加速数据分类和标注指导；4)创建了首个红树林TLS数据集Mangrove3D。相比之前工作的不同：与现有3D标注工具相比，专为复杂的TLS生态场景设计；与球面投影方法相比，扩展为多通道特征表示；与主动学习和自训练相比，针对TLS场景的挑战进行了改进；与特征融合方法相比，仅使用LiDAR衍生特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种半自动、不确定性感知的标注流程，通过球面投影、特征丰富、集成学习和目标标注相结合，显著降低了TLS点云语义分割的标注工作量，同时保持了高准确性，并创建了首个红树林TLS数据集Mangrove3D。'}


### 论文摘要

Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D.   Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at https://fz-rit.github.io/through-the-lidars-eye/.

---

## 18. Novel point cloud registration approach for noninvasive patient specific estimation of leaflet strain from 3D images of heart valves

**论文链接:** [http://arxiv.org/abs/2510.06578v1](http://arxiv.org/abs/2510.06578v1)

**作者:** Wensi Wu, Matthew Daemer, Jeffrey A. Weiss, Alison M. Pouch, Matthew A. Jolley

**发布时间:** 2025-10-08

### GPT解析

### 总结

本研究提出了一种新的特征跟踪框架，用于量化房室瓣的瓣膜应变，相比现有方法具有更高的准确性和鲁棒性。

### 背景

心脏瓣膜疾病很常见，是心力衰竭的主要病因。瓣膜应变是评估瓣膜病理学发生和发展的潜在指标，但目前缺乏稳健且可推广的无创方法从临床图像中量化瓣膜应变。

### 目的

开发一种新的特征跟踪框架，用于使用三维超声心动图图像量化房室瓣的瓣膜应变。

### 方法

提出了一种新的特征跟踪框架，用于使用儿科和成人患者的三维超声心动图图像量化房室瓣的瓣膜应变，并通过有限元基准验证其准确性。

### 主要发现

该方法在评估心脏瓣膜的解剖变形和应变方面表现出比其他基于点的方法更高的准确性，能够在无需参数调整的情况下稳健跟踪高度可变形态的瓣膜跨阶段变形。第一主应变的中位数和四分位范围大于0.5与瓣膜膨出（脱垂）相关。

### 结论

进一步研究心脏瓣膜疾病的生物力学特征可能有助于提高瓣膜疾病的预后评估和纵向评估。

### 翻译

心脏瓣膜疾病很常见，是心力衰竭的主要病因。瓣膜应变是评估瓣膜病理学发生和发展的潜在指标。然而，从临床获取的患者图像中无创量化瓣膜应变的方法仍然有限。在本工作中，我们提出了一种新的特征跟踪框架，用于使用儿科和成人患者的三维超声心动图图像量化房室瓣的瓣膜应变。与其他基于点的方法相比，我们的方法在评估心脏瓣膜的解剖变形和应变方面表现出更高的准确性，并通过有限元基准验证。此外，我们的方法无需参数调整即可稳健地跟踪高度可变形态的瓣膜跨阶段变形。我们的分析显示，第一主应变的中位数和四分位范围大于0.5与瓣膜膨出（脱垂）相关。进一步研究心脏瓣膜疾病的生物力学特征可能有助于提高瓣膜疾病的预后评估和纵向评估。


### 论文摘要

Valvular heart disease is prevalent and a major contributor to heart failure. Valve leaflet strain is a promising metric for evaluating the mechanics underlying the initiation and progression of valvular pathology. However, robust and generalizable methods for noninvasively quantifying valvular strain from clinically acquired patient images remain limited. In this work, we present a novel feature-tracking framework for quantifying leaflet strain in atrioventricular valves using 3D echocardiographic images of pediatric and adult patients. Our method demonstrated superior accuracy in the assessment of anatomical deformation and strain of heart valves compared to other point-based approaches, as verified against a finite element benchmark. Further, our approach can robustly track inter-phase deformation of valves across highly variable morphologies without parameter tuning. Our analysis revealed that a median and interquartile range of the 1st principal strain greater than 0.5 is associated with leaflet billow (prolapse). Further investigation of the biomechanical signatures of heart valve disease has the potential to enhance prognostic assessment and longitudinal evaluation of valvular disease.

---

## 19. Terrain-Aided Navigation Using a Point Cloud Measurement Sensor

**论文链接:** [http://arxiv.org/abs/2510.06470v1](http://arxiv.org/abs/2510.06470v1)

**作者:** Abdülbaki Şanlan, Fatih Erol, Murad Abu-Khalaf, Emre Koyuncu

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究探讨了点云测量在地形辅助导航中的应用，通过比较两种测量模型并研究其可观测性特性，证明了点云测量优于雷达高度计，且选择哪种模型取决于计算资源。

### 背景

地形辅助导航领域，以及惯性导航系统可能存在的精度问题。

### 目的

通过探索生成有用的测量创新误差的方法，辅助惯性导航系统，实现有效的非线性状态估计。

### 方法

比较了两种涉及数字地形高程模型扫描的测量模型：a)基于典型光线投射的模型，从给定姿态返回预测的点云测量；b)一种计算量较小不需要光线投射的滑动网格模型。此外，还研究了这两种测量模型的高度可观测性特性，并与雷达高度计进行了性能比较。

### 主要发现

点云测量优于雷达高度计的使用，且应使用的点云测量模型取决于计算资源。

### 结论

点云测量优于雷达高度计，而应使用的点云测量模型取决于可用的计算资源。

### 翻译

我们研究了点云测量在地形辅助导航中的使用。我们的目标是通过探索生成有用的测量创新误差的方法来辅助惯性导航系统，实现有效的非线性状态估计。我们比较了两种涉及数字地形高程模型扫描的测量模型：a)一种基于从给定姿态出发的典型光线投射的模型，返回该姿态的预测点云测量；b)另一种计算量较小不需要光线投射的模型，我们在此称为滑动网格。除了需要姿态外，它还需要点云测量本身的模式，并返回预测的点云测量。我们进一步研究了这两种测量模型的高度可观测性特性。作为基线，我们将点云测量的性能与雷达高度计的使用进行了比较，并显示了精度的提高。我们最后得出结论，点云测量优于雷达高度计的使用，而应使用的点云测量模型取决于计算资源。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用点云测量传感器来辅助惯性导航系统，提高在GPS拒绝环境下的导航精度问题。这个问题在现实中非常重要，因为GPS信号在许多环境（如城市峡谷、室内、军事区域）可能不可用或被干扰，而传统地形辅助导航主要依赖雷达高度计，在高倾斜角等情况下精度有限，无法满足现代自主导航系统的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了地形辅助导航的历史发展，从早期的TERCOM和SITAN系统到现代的粒子滤波应用。他们借鉴了[10]和[12]中使用点云测量进行状态估计的基本思路，但创新性地提出了两种不同的测量模型：射线投射和滑动网格。作者认识到传统扩展卡尔曼滤波难以处理高度非线性的测量模型，因此选择使用边缘化粒子滤波器来估计误差状态。他们还借鉴了现有的运动模型（如Pinson模型）和状态分解方法，将总状态导航动力学分解为标称状态和误差状态。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用点云测量提供丰富的地形信息，通过比较实际点云测量和预测点云测量生成误差信号，并使用边缘化粒子滤波器估计和校正惯性导航系统的误差状态。创新性地提出了滑动网格方法，避免计算密集的射线投射，显著提高效率。整体实现流程包括：1)获取IMU数据和点云测量；2)建立运动模型并分解状态；3)对每个粒子预测点云测量；4)计算预测与实际测量的误差；5)更新粒子权重；6)估计误差状态；7)与标称状态结合得到改进的导航解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出滑动网格方法避免计算密集的射线投射；2)使用较小尺寸的点云进行地形辅助导航；3)采用边缘化粒子滤波器直接估计误差状态；4)证明点云测量可观测高度信息，无需额外气压高度计；5)全面评估不同IMU等级下的性能。相比之前工作，本文的方法比传统雷达高度计提供更丰富的地形信息；相比[10]使用的大型机载激光扫描仪，使用更小的点云；相比[12]的可导向激光测量，使用固定模式点云；相比传统扩展卡尔曼滤波，更好地处理高度非线性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于点云测量的高效地形辅助导航方法，通过创新的滑动网格算法和边缘化粒子滤波器，显著提高了GPS拒绝环境下的导航精度，特别是在高倾斜角和复杂地形条件下。'}


### 论文摘要

We investigate the use of a point cloud measurement in terrain-aided navigation. Our goal is to aid an inertial navigation system, by exploring ways to generate a useful measurement innovation error for effective nonlinear state estimation. We compare two such measurement models that involve the scanning of a digital terrain elevation model: a) one that is based on typical ray-casting from a given pose, that returns the predicted point cloud measurement from that pose, and b) another computationally less intensive one that does not require raycasting and we refer to herein as a sliding grid. Besides requiring a pose, it requires the pattern of the point cloud measurement itself and returns a predicted point cloud measurement. We further investigate the observability properties of the altitude for both measurement models. As a baseline, we compare the use of a point cloud measurement performance to the use of a radar altimeter and show the gains in accuracy. We conclude by showing that a point cloud measurement outperforms the use of a radar altimeter, and the point cloud measurement model to use depends on the computational resources

---

## 20. Human Action Recognition from Point Clouds over Time

**论文链接:** [http://arxiv.org/abs/2510.05506v2](http://arxiv.org/abs/2510.05506v2)

**作者:** James Dickens

**发布时间:** 2025-10-07

### GPT解析

### 总结

该论文提出了一种基于3D视频的人类动作识别新方法，结合点云技术与稀疏卷积网络，实现了从密集3D数据中识别人体动作。

### 背景

当前人类动作识别研究主要集中在骨骼动作识别和基于视频的方法。随着消费级深度传感器和激光雷达设备的普及，利用密集3D数据进行动作识别成为一个新的机会。

### 目的

开发一种新的方法，利用3D视频进行动作识别，作为骨骼动作识别和视频方法的'第三种方式'。

### 方法

提出一个处理流程，从场景背景中分割人体点云，跟踪个体随时间变化，并进行身体部位分割；支持来自深度传感器和单目深度估计的点云；核心是一种新的3D动作识别骨干网络，结合基于点云的技术和应用于体素映射点云序列的稀疏卷积网络；集成辅助点特征，包括表面法线、颜色、红外强度和身体部位解析标签，以提高识别准确性。

### 主要发现

在NTU RGB-D 120数据集上的评估表明，该方法与现有的骨骼动作识别算法具有竞争力；将基于传感器和估计的深度输入以集成方式结合，当考虑不同受试者进行训练和测试时，准确率达到89.3%；这种方法优于之前的点云动作识别方法。

### 结论

提出的3D动作识别方法是一种有效的新方法，能够利用密集3D数据进行动作识别，并且在性能上优于现有方法。

### 翻译

最近的人类动作识别研究主要集中在骨骼动作识别和基于视频的方法。随着消费级深度传感器和激光雷达设备的日益普及，利用密集3D数据进行动作识别，发展第三种方法的机会正在增加。本文提出了一种从3D视频识别动作的新方法，引入了一个流程，从场景背景中分割人体点云，随时间跟踪个体，并执行身体部位分割。该方法支持来自深度传感器和单目深度估计的点云。所提出的HAR框架的核心是一种用于3D动作识别的新型骨干网络，它将基于点云的技术与应用于体素映射点云序列的稀疏卷积网络相结合。实验集成了辅助点特征，包括表面法线、颜色、红外强度和身体部位解析标签，以提高识别准确性。在NTU RGB-D 120数据集上的评估表明，该方法与现有的骨骼动作识别算法具有竞争力。此外，在集成设置中同时结合基于传感器和估计的深度输入，当考虑不同受试者进行训练和测试时，这种方法达到89.3%的准确率，超越了之前的点云动作识别方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决从3D点云序列识别人类动作的问题。随着消费级深度传感器和激光雷达设备的普及，利用3D点云数据进行动作识别成为一种有前景的'第三种方式'，补充了传统的基于视频和骨骼的方法。这个问题很重要，因为动作识别在监控、老年人跌倒检测、体育分析和自动驾驶等领域有广泛应用价值，而现有方法各有局限性：视频方法需要降低帧尺寸限制了精细动作识别，骨骼方法依赖可能出错的关键点估计。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，然后设计了适用于两种场景的流程：深度传感器获取的真实点云和通过单目深度估计从RGB视频生成的点云。方法借鉴了多项现有工作：使用M2FP模型进行人体分割，ByteTrack算法进行跟踪，迭代最远点采样进行点采样，DBSCAN进行去噪，以及Depth-Anything v2进行深度估计。在网络架构上借鉴了PointNet的T-Net模块和ResNet的残差设计。作者的创新在于将这些技术整合到一个完整的点云动作识别流程中，并设计了新的SP-HP-ConvoT骨干网络。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是直接利用3D点云序列进行动作识别，结合点处理技术和稀疏卷积网络的优势，并融合多种辅助特征提高识别精度。整体流程分为三部分：1) 点云获取：对深度传感器输入进行人体分割、3D投影和去噪；对RGB输入先进行单目深度估计再进行相同处理；2) 特征提取：计算表面法线等特征，使用迭代最远点采样固定点数；3) 动作识别：使用T-Net嵌入、体素映射、稀疏CNN骨干（包括MS-TCN层）、全局池化和全连接分类器进行最终分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 完整的点云处理流程，包含人体分割、跟踪和身体部位分割；2) 同时支持深度传感器和单目深度估计两种输入场景；3) 新的SP-HP-ConvoT骨干网络，结合点处理和稀疏卷积；4) 多模态特征融合，包括表面法线、颜色等；5) 高效处理多人场景。相比之前工作，不同之处在于：提供了更精确的人体和身体部位分割，结合了点处理和稀疏卷积方法，支持从RGB估计点云，不依赖可能出错的关键点估计，保留了更丰富的3D空间信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种新颖的从点云序列识别人类动作的方法，通过结合点处理技术和稀疏卷积网络，并适应两种输入场景，在NTU RGB-D 120数据集上实现了与现有骨骼动作识别算法相竞争的性能。'}


### 论文摘要

Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods. With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way. This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation. The method supports point clouds from both depth sensors and monocular depth estimation. At the core of the proposed HAR framework is a novel backbone for 3D action recognition, which combines point-based techniques with sparse convolutional networks applied to voxel-mapped point cloud sequences. Experiments incorporate auxiliary point features including surface normals, color, infrared intensity, and body part parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D 120 dataset demonstrates that the method is competitive with existing skeletal action recognition algorithms. Moreover, combining both sensor-based and estimated depth inputs in an ensemble setup, this approach achieves 89.3% accuracy when different human subjects are considered for training and testing, outperforming previous point cloud action recognition methods.

---

## 21. ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning

**论文链接:** [http://arxiv.org/abs/2510.05070v2](http://arxiv.org/abs/2510.05070v2)

**作者:** Siheng Zhao, Yanjie Ze, Yue Wang, C. Karen Liu, Pieter Abbeel, Guanya Shi, Rocky Duan

**发布时间:** 2025-10-06

**备注:** 9 pages, 8 figures

### GPT解析

### 总结

ResMimic是一种双阶段残差学习框架，用于从人类运动数据中学习精确且富有表现力的人形机器人全身运动控制，解决了现有通用运动跟踪方法在运动操作中缺乏精确性和物体意识的问题。

### 背景

人形全身运动操作对于日常服务和仓库任务具有变革性潜力，但现有的通用运动跟踪(GMT)方法虽然能重现多样的人类运动，却缺乏运动操作所需的精确性和物体意识。

### 目的

开发一种从人类运动数据中学习精确且富有表现力的人形机器人控制方法，以实现高效的运动操作任务。

### 方法

ResMimic采用双阶段残差学习框架：第一阶段使用大规模纯人类运动数据训练的GMT策略作为基础生成类人全身运动；第二阶段学习高效但精确的残差策略优化GMT输出，改进运动并加入物体交互。还设计了三种辅助训练技术：基于点云的物体跟踪奖励、接触奖励和基于课程的虚拟物体控制器。

### 主要发现

在仿真和真实Unitree G1人形机器人上的评估结果显示，ResMimic与强基线相比在任务成功率、训练效率和鲁棒性方面有显著提升。

### 结论

ResMimic框架有效地解决了人形机器人运动操作中的精确性和物体意识问题，为机器人在日常服务和仓库任务中的应用提供了新的可能性。

### 翻译

人形全身运动操作承诺为日常服务和仓库任务带来变革性能力。虽然最近的通用运动跟踪(GMT)进展使人形机器人能够重现多样的人类运动，但这些策略缺乏运动操作所需的精确性和物体意识。为此，我们引入ResMimic，一种从人类运动数据中进行精确且富有表现力的人形机器人控制的双阶段残差学习框架。首先，在大型纯人类运动数据上训练的GMT策略作为生成类人全身运动的任务不可知基础。然后，学习一个高效但精确的残差策略来优化GMT输出，改进运动并整合物体交互。为进一步促进高效训练，我们设计了(i)基于点云的物体跟踪奖励，用于更平滑的优化；(ii)接触奖励，鼓励精确的人形身体-物体交互；以及(iii)基于课程的虚拟物体控制器，以稳定早期训练。我们在仿真和真实的Unitree G1人形机器人上评估了ResMimic。结果表明，与强基线相比，ResMimic在任务成功率、训练效率和鲁棒性方面有显著提升。视频可在https://resmimic.github.io/获取。


### 论文摘要

Humanoid whole-body loco-manipulation promises transformative capabilities for daily service and warehouse tasks. While recent advances in general motion tracking (GMT) have enabled humanoids to reproduce diverse human motions, these policies lack the precision and object awareness required for loco-manipulation. To this end, we introduce ResMimic, a two-stage residual learning framework for precise and expressive humanoid control from human motion data. First, a GMT policy, trained on large-scale human-only motion, serves as a task-agnostic base for generating human-like whole-body movements. An efficient but precise residual policy is then learned to refine the GMT outputs to improve locomotion and incorporate object interaction. To further facilitate efficient training, we design (i) a point-cloud-based object tracking reward for smoother optimization, (ii) a contact reward that encourages accurate humanoid body-object interactions, and (iii) a curriculum-based virtual object controller to stabilize early training. We evaluate ResMimic in both simulation and on a real Unitree G1 humanoid. Results show substantial gains in task success, training efficiency, and robustness over strong baselines. Videos are available at https://resmimic.github.io/ .

---

## 22. Platonic Transformers: A Solid Choice For Equivariance

**论文链接:** [http://arxiv.org/abs/2510.03511v2](http://arxiv.org/abs/2510.03511v2)

**作者:** Mohammad Mohaiminul Islam, Rishabh Anand, David R. Wessels, Friso de Kruiff, Thijs P. Kuipers, Rex Ying, Clara I. Sánchez, Sharvaree Vadgama, Georg Bökman, Erik J. Bekkers

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种名为柏拉图Transformer的新型模型，通过引入柏拉图立体对称群参考帧的注意力机制，解决了传统Transformers缺乏几何对称性归纳偏置的问题，同时保持了模型的高效性和灵活性。

### 背景

Transformers模型虽然应用广泛，但缺乏科学和计算机视觉中常见的几何对称性的归纳偏置。现有的等变方法通常通过复杂、计算密集型设计牺牲了Transformers原本的高效性和灵活性。

### 目的

开发一种方法，使Transformers能够处理几何对称性问题，同时不损失其原有的效率和灵活性优势。

### 方法

引入柏拉图Transformer，通过定义相对于柏拉图立体对称群参考帧的注意力机制，诱导出有原则的权重共享方案，使模型对连续平移和柏拉图对称性都具有等变性，同时保持标准Transformer的架构和计算成本。

### 主要发现

所提出的注意力机制在形式上等价于动态群卷积，揭示了模型能够学习自适应几何滤波器，并 enables一个高度可扩展的线性时间卷积变体。

### 结论

在计算机视觉(CIFAR-10)、3D点云(ScanObjectNN)和分子性质预测(QM9, OMol25)等多样化基准测试中，柏拉图Transformer通过利用几何约束实现了具有竞争力的性能，且没有额外计算成本。

### 翻译

虽然Transformer应用广泛，但缺乏科学和计算机视觉中常见的几何对称性的归纳偏置。现有的等变方法通常通过复杂、计算密集型设计牺牲了使Transformers如此有效的高效性和灵活性。我们引入柏拉图Transformer来解决这一权衡问题。通过定义相对于柏拉图立体对称群参考帧的注意力，我们的方法诱导出一种有原则的权重共享方案。这使得模型对连续平移和柏拉图对称性都具有等变性，同时保留了标准Transformer的精确架构和计算成本。此外，我们证明这种注意力在形式上等价于动态群卷积，这揭示了模型学习自适应几何滤波器，并 enables一个高度可扩展的线性时间卷积变体。在计算机视觉(CIFAR-10)、3D点云(ScanObjectNN)和分子性质预测(QM9, OMol25)等多样化基准测试中，柏拉图Transformer通过利用这些几何约束实现了具有竞争力的性能，且没有额外成本。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决Transformer模型缺乏处理几何对称性的归纳偏置问题。这一问题在科学和计算机视觉领域至关重要，因为物理、分子化学和3D计算机视觉等领域的数据具有固有的几何对称性。现有的等变方法虽然能处理对称性，但往往牺牲了Transformer的高效性和灵活性，导致难以扩展到大规模应用。没有几何对称性处理能力的模型在处理科学数据时效果不佳，限制了模型在分子预测、3D点云处理等任务中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到Transformer通过位置编码可以处理平移不变性，但无法处理旋转等更复杂的几何变换。他们受Rotary Position Embeddings (RoPE)启发，RoPE通过相对位置编码使Transformer具有平移等变性。作者将这一思路扩展到处理更复杂的几何变换，特别是旋转和反射。他们借鉴了RoPE的位置编码机制、群等变神经网络的群论处理方法以及多头自注意力机制的设计，但创新性地将柏拉图立体(Platonic solids)的对称性群作为参考帧，实现了在保持Transformer计算效率的同时引入几何对称性处理能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入柏拉图立体对称性群作为参考帧，在这些参考帧上并行计算注意力，并通过权重共享机制使模型能够从任何方向学习通用的注意力模式。整体实现流程包括：1) 特征提升：将输入特征提升为群函数，使其相对于每个参考帧有定义；2) 等变线性变换：使用群卷积代替标准矩阵乘法，确保线性层等变；3) 注意力计算：在多个参考帧上并行计算注意力，每个帧使用RoPE处理相对位置；4) 输出组合：将所有参考帧的输出组合，得到最终结果。这种方法保持了标准Transformer的计算图和计算成本，同时引入了几何对称性处理能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Platonic Transformer架构：首次将柏拉图立体对称性群集成到Transformer中；2) 权重共享机制：通过群卷积实现权重共享，在不增加计算成本的情况下引入几何偏置；3) 等变性注意力：实现了对连续平移和离散旋转/反射的等变性；4) 动态群卷积等价性：证明了RoPE注意力等价于动态群卷积，并提出了线性时间复杂度的变体；5) 计算效率：保持了标准Transformer的计算图和计算成本。相比之前的工作，它不像传统等变网络需要复杂的计算操作，不像混合架构需要打破对称性，不像不变注意力机制牺牲特征表示，也不像帧平均方法需要为每个帧元素单独前向传播。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Platonic Transformer通过引入柏拉图立体对称性群的参考帧和权重共享机制，在保持标准Transformer计算效率的同时，实现了对几何变换的等变性处理，解决了对称性感知与可扩展性之间的长期权衡问题。'}


### 论文摘要

While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost.

---

## 23. Unified Unsupervised Anomaly Detection via Matching Cost Filtering

**论文链接:** [http://arxiv.org/abs/2510.03363v2](http://arxiv.org/abs/2510.03363v2)

**作者:** Zhe Zhang, Mingxiu Cai, Gaochang Wu, Jing Zhang, Lingqiao Liu, Dacheng Tao, Tianyou Chai, Xiatian Zhu

**发布时间:** 2025-10-03

**备注:** 63 pages (main paper and supplementary material), 39 figures, 58  tables

### GPT解析

### 总结

本文提出统一成本过滤(UCF)框架，通过可学习过滤模块减轻匹配噪声并突出细微异常，在单模态和多模态UAD场景中均取得最先进结果。

### 背景

无监督异常检测(UAD)在工业检测和医疗分析等领域有广泛应用，但因隐私问题和冷启动约束，异常样本稀缺。现有方法主要进行图像或特征级匹配，但匹配噪声被忽视，限制了检测能力。从单模态RGB UAD扩展到多模态场景，但这些研究方向相互隔离，阻碍了全面理解和知识转移。

### 目的

从匹配角度提出统一单模态和多模态UAD的方法，开发通用后优化框架提升各种UAD模型性能。

### 方法

构建测试样本与来自相同或不同模态正常样本间的匹配成本体积，使用测试样本的多层注意力引导的可学习过滤模块减轻匹配噪声并突出细微异常。

### 主要发现

在22个多样化基准测试上，UCF能有效增强各种UAD方法，在单模态(RGB)和多模态(RGB-3D, RGB-Text)UAD场景中持续实现新的最先进结果。

### 结论

UCF是适用于各种UAD模型的通用后优化方法，能够有效提升异常检测性能，代码和模型将在指定网址发布。

### 翻译

无监督异常检测(UAD)旨在仅使用正常训练数据识别图像和像素级别的异常，在工业检测和医疗分析等领域有广泛应用，但因隐私问题和冷启动约束，异常样本稀缺。现有方法，无论是基于重建（恢复正常对应物）还是基于嵌入（预训练表示），根本上都进行图像或特征级匹配来生成异常图。然而，匹配噪声在很大程度上被忽视，限制了它们的检测能力。除了早期基于单模态RGB的UAD关注外，最近的进展扩展到多模态场景，例如RGB-3D和RGB-Text，这得益于点云传感和视觉语言模型。尽管面临共同挑战，但这些研究方向在很大程度上仍然相互隔离，阻碍了全面理解和知识转移。在本文中，我们从匹配角度倡导统一单模态和多模态设置的UAD。在这一洞察下，我们提出了统一成本过滤(UCF)，这是一个通用的后优化细化框架，用于细化任何UAD模型的异常成本体积。成本体积是通过将测试样本与来自相同或不同模态的正常样本进行匹配构建的，然后是测试样本的多层注意力引导的可学习过滤模块，减轻匹配噪声并突出细微异常。在22个多样化基准测试上的全面实验证明了UCF在增强各种UAD方法方面的有效性，在单模态(RGB)和多模态(RGB-3D, RGB-Text)UAD场景中持续实现新的最先进结果。代码和模型将在https://github.com/ZHE-SAPI/CostFilter-AD发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决无监督异常检测(UAD)中的匹配噪声问题。这个问题在现实中非常重要，因为UAD在工业检测、医疗分析等领域有广泛应用，但这些领域中异常样本由于隐私问题和冷启动约束而稀缺。匹配噪声会导致模糊的边界、假阳性和假阴性，特别是对于细微缺陷、低对比度或接近正常的区域，严重影响检测的准确性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者从匹配角度重新概念化单模态和多模态UAD，明确指出被忽视的匹配噪声问题。他们借鉴了立体匹配、深度估计、光流估计和光场渲染等领域的'匹配成本过滤'概念，将UAD重构为特征提取、异常成本体积构建和异常成本体积过滤的三步范式。UCF设计为通用后处理框架，可集成到各种UAD方法中。作者还使用了注意力机制、残差连接等深度学习技术，并引入多模板匹配和类感知适配器来增强检测能力。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将异常检测重新概念化为匹配成本过滤过程，通过构建和过滤异常成本体积来抑制匹配噪声，同时保留边缘结构和细微异常。整体流程包括：1)准备参考模板(重建RGB模板、3D点云或文本提示)；2)提取输入和模板的多层特征；3)构建异常成本体积(通过相似度匹配)；4)使用3D U-Net和双流注意力引导(输入特征和初始异常图)过滤成本体积；5)生成最终异常图。训练时使用合成异常掩码，推理时与基线响应加权结合。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)重新概念化UAD，明确解决匹配噪声问题；2)提出三步范式(特征提取、成本体积构建和过滤)；3)设计通用UCF框架；4)引入双流注意力引导机制；5)采用多模板匹配策略；6)添加类感知适配器。相比之前工作，UCF明确处理匹配噪声问题，提供统一框架适用于所有模态，作为通用插件可集成到各种方法中，在22个基准测试上取得state-of-the-art结果，并能更好保留边缘结构和检测细微异常。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的统一成本过滤(UCF)框架通过构建和过滤异常成本体积有效抑制了匹配噪声，显著提升了单模态和多模态无监督异常检测的性能，为UAD领域提供了一个统一且通用的解决方案。'}


### 论文摘要

Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB-3D and RGB-Text, enabled by point cloud sensing and vision-language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB-3D, RGB-Text) UAD scenarios. Code and models will be released at https://github.com/ZHE-SAPI/CostFilter-AD.

---

## 24. TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs

**论文链接:** [http://arxiv.org/abs/2510.06747v1](http://arxiv.org/abs/2510.06747v1)

**作者:** I-Fan Lin, Faegheh Hasibi, Suzan Verberne

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种无需训练和标签的短文本聚类方法，可应用于任何现有嵌入器。该方法基于迭代向量更新，通过代表性文本构建稀疏向量，并在LLM指导下进行迭代优化。

### 背景

在面向客户的聊天机器人场景中，公司需要处理大量用户话语并根据其意图进行聚类。在这些商业环境中，通常没有标记数据可用，且聚类数量未知。

### 目的

开发一种无需训练和标签的短文本聚类方法，适用于任何现有嵌入器，能够在没有预先了解聚类或标签的情况下进行聚类，并且具有可扩展性和低资源适应性。

### 方法

提出一种基于迭代向量更新的方法：首先基于代表性文本构建稀疏向量，然后通过LLM指导进行迭代优化这些向量。

### 主要发现

1) 该方法在不预先了解聚类或标签的情况下，能够达到与使用对比学习的最先进方法相当或更好的结果；2) 在不同数据集和小型LLM上的实验表明，该方法与模型无关，可应用于任何嵌入器、小型LLM和不同聚类方法；3) 该方法可以扩展到大型数据集，降低LLM的计算成本。

### 结论

这种低资源、可适应的设置以及方法的可扩展性，使其比现有的聚类方法更符合实际场景。

### 翻译

在本文中，我们提出了一种无需训练和标签的短文本聚类方法，可以应用于任何现有的嵌入器。在面向客户的聊天机器人背景下，公司需要处理大量用户话语，这些话语需要根据其意图进行聚类。在这些商业环境中，通常没有标记数据可用，且聚类数量未知。我们的方法基于迭代向量更新：它基于代表性文本构建稀疏向量，然后通过LLM指导迭代优化这些向量。我们的方法在不假设预先了解聚类或标签的情况下，能够达到与使用对比学习的最先进方法相当或更好的结果。在不同数据集和小型LLM上的实验表明，我们的方法与模型无关，可以应用于任何嵌入器、小型LLM和不同的聚类方法。我们还表明，我们的方法可以扩展到大型数据集，降低LLM的计算成本。这种低资源、可适应的设置以及方法的可扩展性，使其比现有的聚类方法更符合实际场景。


### 论文摘要

In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods.

---

## 25. TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics

**论文链接:** [http://arxiv.org/abs/2510.07181v1](http://arxiv.org/abs/2510.07181v1)

**作者:** Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang

**发布时间:** 2025-10-08

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

TIGeR是一种新框架，将视觉语言模型从感知估计器转变为几何计算机，通过外部工具实现厘米级精度的几何计算，在几何推理基准测试上达到SOTA性能，并在现实机器人操作任务中表现出色。

### 背景

视觉语言模型在空间推理方面显示出显著能力，但本质上仅限于定性精度，缺乏机器人技术所需的计算精度。当前方法未能利用深度传感器和相机校准的度量线索，而是将几何问题降级为无法实现机器人操作所需厘米级精度的模式识别任务。

### 目的

提出一种新框架，使视觉语言模型从感知估计器转变为几何计算机，通过外部工具使模型能够生成和执行精确的几何计算，实现机器人操作所需的厘米级精度。

### 方法

TIGeR(工具集成几何推理)框架不试图在神经网络内部化复杂的几何操作，而是赋予模型识别几何推理需求、合成适当计算代码并调用专门库进行精确计算的能力。研究团队引入了TIGeR-300K数据集，包含点变换、姿态估计、轨迹生成和空间兼容性验证，并采用两阶段训练管道：监督微调(SFT)和强化微调(RFT)，配合分层奖励设计。

### 主要发现

TIGeR在几何推理基准测试上实现了SOTA性能，在现实世界机器人操作任务中表现出厘米级精度。

### 结论

TIGeR框架成功将视觉语言模型从感知估计器转变为几何计算机，通过外部工具实现了精确的几何计算，解决了视觉语言模型在机器人应用中的精度限制问题。

### 翻译

视觉语言模型在空间推理方面显示出显著能力，但它们本质上仅限于定性精度，缺乏机器人技术所需的计算精度。当前方法未能利用深度传感器和相机校准的度量线索，而是将几何问题降级为无法实现机器人操作所需厘米级精度的模式识别任务。我们提出了TIGeR(工具集成几何推理)，这是一种新框架，通过使视觉语言模型能够通过外部工具生成和执行精确的几何计算，将其从感知估计器转变为几何计算机。TIGeR不试图在神经网络内部化复杂的几何操作，而是赋予模型识别几何推理需求、合成适当计算代码并调用专门库进行精确计算的能力。为支持这一范式，我们引入了TIGeR-300K，这是一个全面的面向工具调用的数据集，涵盖点变换、姿态估计、轨迹生成和空间兼容性验证，包含工具调用序列和中间计算。通过结合监督微调(SFT)和我们提出的分层奖励设计的强化微调(RFT)的两阶段训练管道，TIGeR在几何推理基准测试上实现了SOTA性能，同时在现实世界机器人操作任务中表现出厘米级精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决视觉语言模型(VLMs)在精确几何计算方面的局限性，使其无法满足机器人操作所需的厘米级精度。这个问题很重要，因为现实世界中的机器人需要精确计算3D位置、距离和姿态等几何信息，而现有VLM只能提供'左侧'、'可到达'等定性评估，无法支持机器人完成精确操作任务，如物体抓取、放置和路径规划等。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到VLMs在几何推理方面的根本局限，因此提出不要试图在神经网络内部复杂化几何运算，而是让模型识别几何推理需求，生成适当代码并调用外部工具执行。他们借鉴了工具集成推理(TIR)和空间理解领域的工作，但创新性地提出了新的基于过程的奖励函数和两阶段训练流程(SFT+RFT)，结合了模板合成数据和大模型重写数据，确保几何概念的全面覆盖和模型的泛化能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将VLM从感知估计器转变为几何计算机，通过外部工具实现精确几何计算。整体流程包括：1)工具分类(视觉感知工具和几何计算工具)；2)数据生成(TIGeR-300K数据集，包含模板合成和大模型重写两种数据)；3)两阶段训练(监督微调SFT初始化工具使用能力，强化微调RFT使用分层奖励函数提升精度)；4)推理过程(模型识别几何需求，生成代码，调用工具执行计算，返回结果)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)概念与方法创新，将VLM转变为几何计算机；2)TIGeR-300K数据集，专为几何推理设计；3)两阶段SFT→RFT训练流程和分层奖励设计。相比之前工作，不同之处在于：不试图在神经网络内部复杂化几何运算；提出新的基于过程的奖励函数；结合模板合成和大模型重写数据；在真实机器人任务中实现厘米级精度；支持跨视点的统一推理，即使多视角相机没有联合校准也能保持一致的数值推理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TIGeR通过工具集成几何推理框架，使视觉语言模型能够从感知估计器转变为精确的几何计算机，实现了厘米级精度的机器人操作能力。'}


### 论文摘要

Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.

---

## 26. Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction

**论文链接:** [http://arxiv.org/abs/2510.04759v2](http://arxiv.org/abs/2510.04759v2)

**作者:** Chi Yan, Dan Xu

**发布时间:** 2025-10-06

**备注:** Project Page: https://yanchi-3dv.github.io/PG-Occ

### GPT解析

### 总结

本文提出了一种名为PG-Occ的创新渐进式高斯Transformer框架，用于实现开放词汇的3D占用预测，解决了稀疏表示难以捕捉小物体而密集表示计算开销大的权衡问题。

### 背景

3D占用预测任务近年来在基于视觉的自动驾驶系统中扮演关键角色。传统方法局限于固定语义类别，而最近的方法转向预测文本对齐的特征以实现开放词汇查询，但存在稀疏和密集表示的权衡问题。

### 目的

开发一种能够有效捕捉细粒度场景细节同时保持计算效率的3D占用预测框架，实现开放词汇场景理解。

### 方法

提出PG-Occ框架，采用渐进式在线密集化策略逐步增强3D高斯表示；引入各向异性感知采样策略结合时空融合，自适应分配不同尺度和阶段高斯的感受野，实现更有效的特征聚合和场景信息捕获。

### 主要发现

通过广泛评估，PG-Occ实现了最先进的性能，比之前表现最好的方法相对提高了14.3%的mIoU。

### 结论

PG-Over框架成功解决了文本对齐场景建模中的权衡问题，能够实现更精确和详细的场景理解，代码和预训练模型将在项目页面发布。

### 翻译

3D占用预测任务近年来取得了显著进展，在基于视觉的自动驾驶系统中发挥着关键作用。虽然传统方法局限于固定的语义类别，但最近的方法已转向预测文本对齐的特征，以实现真实场景中的开放词汇文本查询。然而，在文本对齐的场景建模中存在权衡：稀疏高斯表示难以捕捉场景中的小物体，而密集表示则会带来显著的计算开销。为解决这些限制，我们提出了PG-Occ，一种创新的渐进式高斯Transformer框架，可实现开放词汇的3D占用预测。我们的框架采用渐进式在线密集化，这是一种前馈策略，能够逐步增强3D高斯表示以捕获细粒度的场景细节。通过迭代增强表示，框架实现越来越精确和详细的场景理解。另一个关键贡献是引入了各向异性感知采样策略结合时空融合，能够自适应地为不同尺度和阶段的高斯分配感受野，实现更有效的特征聚合和更丰富的场景信息捕获。通过广泛评估，我们证明PG-Occ实现了最先进的性能，比之前表现最好的方法相对提高了14.3%的mIoU。代码和预训练模型将在发布时在我们的项目页面上提供：https://yanchi-3dv.github.io/PG-Occ

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D占用预测中的开放词汇问题，即让系统能够识别和定位任意文本提示描述的物体，而不仅限于预定义的语义类别。这个问题在自动驾驶和 embodied intelligence 领域至关重要，因为现实世界中存在大量未预定义的物体类别，系统能够通过文本指令灵活识别这些物体对于安全导航和交互至关重要。此外，现有方法在捕捉小物体细节和计算效率之间存在权衡，限制了实际应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统方法受限于预定义类别，而开放词汇方法在稀疏表示难以捕捉小物体和密集表示计算开销大之间存在权衡。他们借鉴了3D高斯溅射技术用于场景表示，Transformer架构用于特征聚合，以及CLIP等开放词汇视觉-语言模型进行文本对齐。设计思路是采用渐进式方法，从粗到细逐步增强场景表示，同时通过非对称自注意力机制确保训练稳定性，利用各向异性感知采样提高特征提取效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过渐进式高斯建模和各向异性感知采样，实现高效的开放词汇3D占用预测，将场景表示为可扩展的文本对齐特征高斯点集。整体流程包括：1)初始化阶段从伪深度图生成初始高斯；2)渐进式高斯建模，包含基础层和多个渐进层，每层使用渐进式在线密度化添加新高斯；3)各向异性感知特征采样，根据高斯形状和方向进行特征提取；4)使用2D监督训练，结合深度和特征损失；5)最终将高斯表示转换为密集3D占用场，支持任意文本查询的语义定位。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)渐进式高斯Transformer框架，通过迭代增强表示捕捉细粒度细节；2)各向异性感知采样策略，根据高斯特性自适应分配感受野；3)非对称自注意力机制，防止新高斯干扰已优化高斯。相比之前工作，PG- adaptively 扩展高斯查询数量而非使用固定数量，能够更好地建模复杂场景；保持高斯表示的稀疏性降低计算开销；实现前馈推理无需离线优化；更有效地平衡开放词汇能力和计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PG-Occ通过渐进式高斯建模和各向异性感知采样，首次实现了高效且高精度的开放词汇3D占用预测，突破了预定义语义类别的限制，显著提升了自动驾驶系统的环境感知能力。'}


### 论文摘要

The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: https://yanchi-3dv.github.io/PG-Occ

---

## 27. HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.07210v1](http://arxiv.org/abs/2510.07210v1)

**作者:** Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种名为HyPlan的新型混合学习辅助规划方法，用于解决自动驾驶汽车在部分可观察交通环境中的无碰撞导航问题。

### 背景

自动驾驶汽车需要在部分可观察的交通环境中安全导航，这面临着挑战。

### 目的

开发一种能够安全且高效导航的方法，减少执行时间而不影响驾驶安全。

### 方法

HyPlan结合了多智能体行为预测、深度强化学习（使用近端策略优化）和近似在线部分可观察马尔可夫决策过程规划（使用基于启发式置信度的垂直剪枝）。

### 主要发现

在CARLA-CTS2基准测试中，HyPlan在包含行人的关键交通场景测试中，比选定的相关基线方法导航更安全，并且比考虑的其他在线POMDP规划器显著更快。

### 结论

HyPlan是一种有效的混合方法，能够在保证安全的同时提高导航效率。

### 翻译

我们提出了一种名为HyPlan的新型混合学习辅助规划方法，用于解决自动驾驶汽车在部分可观察交通环境中的无碰撞导航问题。HyPlan结合了多智能体行为预测方法、使用近端策略优化的深度强化学习和基于启发式置信度的垂直剪枝的近似在线部分可观察马尔可夫决策过程规划，以减少执行时间而不影响驾驶安全。我们在CARLA-CTS2基准测试中对包含行人的关键交通场景进行的实验性能分析表明，HyPlan可能比选定的相关基线导航更安全，并且比考虑的其他在线POMDP规划器显著更快。


### 论文摘要

We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.

---

## 28. Does Physics Knowledge Emerge in Frontier Models?

**论文链接:** [http://arxiv.org/abs/2510.06251v1](http://arxiv.org/abs/2510.06251v1)

**作者:** Ieva Bagdonaviciute, Vibhav Vineet

**发布时间:** 2025-10-03

**备注:** 8 pages, 7 figures. Preprint

### GPT解析

### 总结

研究前沿视觉-语言模型在物理动态理解和预测方面的能力，发现尽管这些模型在视觉感知和一般推理方面表现出色，但在物理动态理解方面存在明显不足，感知和物理推理能力呈碎片化状态，未能有效结合形成因果理解。

### 背景

前沿的视觉-语言模型(VLMs)在视觉感知和一般推理方面显示出强大结果，但它们理解和预测物理动态的能力仍然不清楚。

### 目的

评估前沿VLMs在物理模拟任务中的表现，探究感知能力与物理推理能力之间的关系，揭示当前VLMs在物理动态理解方面的局限性。

### 方法

在三个物理模拟数据集(CLEVRER、Physion和Physion++)上对六个前沿VLMs进行基准测试，评估任务包括预测结果或假设替代情况；设计诊断子测试分离感知(物体、颜色、遮挡物)与物理推理(运动预测、空间关系)能力；分析感知/物理推理性能与评估准确性之间的相关性。

### 主要发现

模型在感知或物理推理方面的出色表现并不一致地转化为预测或反事实评估的更高准确性；感知和物理推理技能之间存在弱相关性；当前VLMs的感知和物理推理能力仍然是碎片化的，未能结合成因果理解。

### 结论

当前VLMs存在一个核心局限：感知和物理推理技能未能有效结合形成因果理解，这表明需要开发能够更紧密地绑定感知和推理的架构。

### 翻译

前沿视觉-语言模型(VLMs)在视觉感知和一般推理方面显示出强大的结果，但它们理解和预测物理动态的能力仍然不清楚。我们在三个物理模拟数据集 - CLEVRER、Physion和Physion++上对六个前沿VLMs进行了基准测试，其中评估任务测试模型是否可以预测结果或对替代情况提出假设。为了进行更深入的探究，我们设计了诊断子测试，将感知(物体、颜色、遮挡物)与物理推理(运动预测、空间关系)分离出来。直观地来看，更强的诊断性能应该支持更高的评估准确性。然而，我们的分析揭示了弱相关性：在感知或物理推理方面表现出色的模型并不总是在预测或反事实评估中表现更好。这种违反直觉的差距暴露了当前VLMs的一个核心局限：感知和物理技能仍然是碎片化的，未能结合成因果理解，这强调了需要开发能够更紧密地绑定感知和推理的架构。


### 论文摘要

Leading Vision-Language Models (VLMs) show strong results in visual perception and general reasoning, but their ability to understand and predict physical dynamics remains unclear. We benchmark six frontier VLMs on three physical simulation datasets - CLEVRER, Physion, and Physion++ - where the evaluation tasks test whether a model can predict outcomes or hypothesize about alternative situations. To probe deeper, we design diagnostic subtests that isolate perception (objects, colors, occluders) from physics reasoning (motion prediction, spatial relations). Intuitively, stronger diagnostic performance should support higher evaluation accuracy. Yet our analysis reveals weak correlations: models that excel at perception or physics reasoning do not consistently perform better on predictive or counterfactual evaluation. This counterintuitive gap exposes a central limitation of current VLMs: perceptual and physics skills remain fragmented and fail to combine into causal understanding, underscoring the need for architectures that bind perception and reasoning more tightly.

---

## 29. OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects

**论文链接:** [http://arxiv.org/abs/2510.06952v1](http://arxiv.org/abs/2510.06952v1)

**作者:** Bing Li, Wuqi Wang, Yanan Zhang, Jingzheng Li, Haigen Min, Wei Feng, Xingyu Zhao, Jie Zhang, Qing Guo

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种文本到3D对抗生成方法(Phy3DAdvGen)，能够生成对LiDAR检测器不可见的3D物体模型，并在物理环境中实现，有效揭示了自动驾驶系统中LiDAR检测器的安全漏洞。

### 背景

LiDAR 3D目标检测器是自动驾驶的基础，未能检测到物体可能带来严重的安全风险。开发有效的3D对抗攻击对于彻底测试这些检测系统并在实际部署前发现其漏洞至关重要。

### 目的

引入文本到3D对抗生成方法，实现物理上可实现的攻击，生成对LiDAR检测器真正不可见的3D物体模型，并且可以在现实世界中轻松实现。

### 方法

通过系统研究行人3D模型的拓扑、连通性和强度对检测的影响，提出物理感知的文本到3D对抗生成方法，迭代优化文本提示生成LiDAR不可见的行人，并基于包含13个真实物体3D模型的物体池生成3D物体以确保物理可实现性。

### 主要发现

该方法能够生成3D行人，在CARLA模拟环境和物理环境中都能规避六种最先进的LiDAR 3D检测器，突显了安全关键应用中的漏洞。

### 结论

物理上可实现的对抗攻击方法能够有效测试LiDAR检测系统，揭示其安全漏洞，强调了在实际部署前进行此类测试的重要性。

### 翻译

基于LiDAR的3D目标检测器是自动驾驶的基础，未能检测到物体会带来严重的安全风险。开发有效的3D对抗攻击对于彻底测试这些检测系统并在实际部署前暴露其漏洞至关重要。然而，现有的向3D点添加优化扰动的对抗攻击有两个关键局限：它们很少导致物体完全消失，并且在物理环境中难以实现。我们引入了文本到3D对抗生成方法，这是一种新颖的方法，能够实现物理上可实现的攻击，生成对LiDAR检测器真正不可见的3D物体模型，并且可以在现实世界中轻松实现。具体而言，我们进行了第一个实证研究，通过操纵行人3D模型的拓扑、连通性和强度，并将行人与多个物体结合，系统研究了影响检测漏洞的因素。基于这些见解，我们提出了物理感知的文本到3D对抗生成(Phy3DAdvGen)方法，通过迭代优化动词、物体和姿势来生成对LiDAR不可见的行人。为确保物理可实现性，我们构建了一个包含13个真实物体3D模型的综合物体池，并限制Phy3DAdvGen基于该集合中的物体组合生成3D物体。大量实验表明，我们的方法能够生成3D行人，在CARLA模拟环境和物理环境中都能规避六种最先进的(SOTA)LiDAR 3D检测器，从而突显了安全关键应用中的漏洞。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成物理上可实现的3D对抗样本，使这些样本能够欺骗LiDAR 3D目标检测器，让检测器无法检测到这些物体。这个问题在现实中非常重要，因为自动驾驶系统中LiDAR检测是基础组件，如果无法检测到行人等物体可能导致严重安全事故；在研究中，开发有效的对抗攻击能帮助测试系统漏洞并提高鲁棒性，而现有方法很少能实现物体完全消失且难以在物理环境中部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先在CARLA仿真环境中系统研究了影响LiDAR检测漏洞的因素，发现物体组合对检测率影响最大。他们借鉴了文本到3D生成模型(如基于高斯溅射的LGM)和对抗攻击领域的工作，但创新性地将这些技术结合。设计方法包括构建动词-物体-姿势(VOP)文本提示空间，通过对抗优化寻找能生成LiDAR不可见物体的提示，并使用物理对象池确保物理可实现性，最终形成Phy3DAdvGen框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过优化文本提示来生成物理可实现且能规避LiDAR检测的3D物体。整体流程包括：1) 构建VOP文本提示；2) 使用检测器置信度作为目标进行对抗优化；3) 将优化后的提示输入文本到3D生成模型(如LGM)生成3D物体；4) 通过物理对象池和多视图渲染实现物理部署，调整真实物体的位置、旋转和缩放以模拟生成的对抗物体。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次系统研究3D物体在LiDAR场景中的可检测性因素；2) 提出Phy3DAdvGen框架，通过优化离散文本组件生成对抗性3D物体；3) 使用物理对象池确保现实世界可行性；4) 在模拟和物理环境中验证攻击效果。相比之前工作，本文方法不直接扰动点云或优化网格几何，而是通过文本提示生成对抗内容，提供了更大的优化空间、更好的物理可实现性和更强的攻击效果。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于文本提示优化的物理可实现3D对抗生成方法，能生成对LiDAR检测器完全不可见的物体，并在真实环境中验证了其有效性，揭示了自动驾驶感知系统的重要安全漏洞。'}


### 论文摘要

LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation (Phy3DAdvGen) that systematically optimizes text prompts by iteratively refining verbs, objects, and poses to produce LiDAR-invisible pedestrians. To ensure physical realizability, we construct a comprehensive object pool containing 13 3D models of real objects and constrain Phy3DAdvGen to generate 3D objects based on combinations of objects in this set. Extensive experiments demonstrate that our approach can generate 3D pedestrians that evade six state-of-the-art (SOTA) LiDAR 3D detectors in both CARLA simulation and physical environments, thereby highlighting vulnerabilities in safety-critical applications.

---

## 30. MolGA: Molecular Graph Adaptation with Pre-trained 2D Graph Encoder

**论文链接:** [http://arxiv.org/abs/2510.07289v1](http://arxiv.org/abs/2510.07289v1)

**作者:** Xingtong Yu, Chang Zhou, Xinming Zhang, Yuan Fang

**发布时间:** 2025-10-08

**备注:** Under review

### GPT解析

### 总结

本文提出了MolGA方法，通过灵活整合多样的分子领域知识，将预训练的2D图编码器适应到下游分子应用中，有效解决了现有方法在整合分子知识方面的局限性。

### 背景

分子图表示学习在化学和生物医学研究中广泛应用。虽然预训练的2D图编码器表现出强大性能，但它们忽视了与亚分子实例（原子和键）相关的丰富分子领域知识。而分子预训练方法虽将此类知识纳入预训练目标，但通常采用针对特定类型知识的设计，缺乏整合分子中多样知识的灵活性。

### 目的

提出一种方法，能够重用广泛可用且经验证的预训练2D编码器，同时在下游适应过程中整合分子领域知识，提供一种更实用的替代方案。

### 方法

作者提出了MolGA方法，包括：1)分子对齐策略，弥合预训练拓扑表示与领域知识表示之间的差距；2)条件适应机制，生成特定于实例的标记，实现分子领域知识的细粒度整合。

### 主要发现

作者在十一个公共数据集上进行了广泛的实验，证明了MolGA的有效性。

### 结论

MolGA通过灵活整合多样的分子领域知识，有效地将预训练的2D图编码器适应到下游分子应用中，是一种更实用的方法。

### 翻译

分子图表示学习在化学和生物医学研究中得到广泛应用。虽然预训练的2D图编码器已展现出强大的性能，但它们忽视了与亚分子实例（原子和键）相关的丰富分子领域知识。尽管分子预训练方法将此类知识纳入其预训练目标，但它们通常采用针对特定类型知识的设计，缺乏整合分子中存在的多样知识的灵活性。因此，重用广泛可用且经验证的预训练2D编码器，同时在下游适应过程中整合分子领域知识，提供了一种更实用的替代方案。在这项工作中，我们提出了MolGA，它通过灵活整合多样的分子领域知识，将预训练的2D图编码器适应到下游分子应用中。首先，我们提出了一个分子对齐策略，弥合预训练拓扑表示与领域知识表示之间的差距。其次，我们引入了一个条件适应机制，生成特定于实例的标记，以实现分子领域知识在下游任务中的细粒度整合。最后，我们在十一个公共数据集上进行了广泛的实验，证明了MolGA的有效性。


### 论文摘要

Molecular graph representation learning is widely used in chemical and biomedical research. While pre-trained 2D graph encoders have demonstrated strong performance, they overlook the rich molecular domain knowledge associated with submolecular instances (atoms and bonds). While molecular pre-training approaches incorporate such knowledge into their pre-training objectives, they typically employ designs tailored to a specific type of knowledge, lacking the flexibility to integrate diverse knowledge present in molecules. Hence, reusing widely available and well-validated pre-trained 2D encoders, while incorporating molecular domain knowledge during downstream adaptation, offers a more practical alternative. In this work, we propose MolGA, which adapts pre-trained 2D graph encoders to downstream molecular applications by flexibly incorporating diverse molecular domain knowledge. First, we propose a molecular alignment strategy that bridge the gap between pre-trained topological representations with domain-knowledge representations. Second, we introduce a conditional adaptation mechanism that generates instance-specific tokens to enable fine-grained integration of molecular domain knowledge for downstream tasks. Finally, we conduct extensive experiments on eleven public datasets, demonstrating the effectiveness of MolGA.

---

## 31. Resolution scaling governs DINOv3 transfer performance in chest radiograph classification

**论文链接:** [http://arxiv.org/abs/2510.07191v1](http://arxiv.org/abs/2510.07191v1)

**作者:** Soroosh Tayebi Arasteh, Mina Shaigan, Christiane Kuhl, Jakob Nikolas Kather, Sven Nebelung, Daniel Truhn

**发布时间:** 2025-10-08

### GPT解析

### 总结

本研究评估了DINOv3自监督学习模型在胸部X光检查中的应用效果，与DINOv2和ImageNet初始化方法进行了比较，确定了最佳输入分辨率为512x512像素，并发现ConvNeXt-B骨干网络表现优于ViT-B/16。

### 背景

自监督学习在视觉表征学习方面取得了进展，但其在胸部X光检查这种具有精细发现的高容量成像模式中的价值尚不明确。Meta的DINOv3通过Gram锚定的自蒸馏扩展了早期的SSL模型，但这些设计选择是否改善了胸部X光检查的迁移学习尚未经过系统测试。

### 目的

评估DINOv3在胸部X光检查中的应用效果，与DINOv2和ImageNet初始化进行比较，确定最佳输入分辨率，以及评估不同骨干网络的表现。

### 方法

在七个数据集(超过814,000个样本)上进行了基准测试，评估了ViT-B/16和ConvNeXt-B两个骨干网络，分析了224x224、512x512和1024x1024像素的图像，还评估了来自7B模型的冻结特征，主要结果是标签的平均AUROC。

### 主要发现

在224x224分辨率下，DINOv3和DINOv2在成人数据集上表现相当；将分辨率提高到512x512时，DINOv3一致优于DINOv2和ImageNet；在儿科队列中，不同初始化方法之间没有差异；在所有设置中，ConvNeXt-B都优于ViT-B/16；使用冻结特征的模型性能低于完全微调的骨干网络；扩展到1024x1024并没有进一步提高准确性；分辨率相关的增益在边界依赖性和小焦点异常方面最为明显。

### 结论

在胸部X光检查中，更高的输入分辨率对于利用现代自监督模型的益处至关重要；512x512像素代表了一个实用的上限，其中DINOv3初始化的ConvNeXt-B网络提供最强性能；临床应用上，支持在512x512分辨率下使用微调的中型骨干网络进行胸部X光解释，预计在检测细微或边界为中心的病变时会有最大收益。

### 翻译

自监督学习在视觉表征学习方面取得了进展，但其在胸部X光检查这种具有精细发现的高容量成像模式中的价值尚不明确。Meta的DINOv3通过Gram锚定的自蒸馏扩展了早期的SSL模型。这些设计选择是否改善了胸部X光检查的迁移学习尚未经过系统测试。我们在七个数据集上对DINOv3与DINOv2和ImageNet初始化进行了基准测试（样本量超过814,000）。评估了两个代表性的骨干网络：ViT-B/16和ConvNeXt-B。分析了224x224、512x512和1024x1024像素的图像。我们还评估了来自7B模型的冻结特征。主要结果是标签的平均AUROC。在224x224分辨率下，DINOv3和DINOv2在成人数据集上表现相当。将分辨率提高到512x512时，DINOv3一致优于DINOv2和ImageNet。相比之下，儿科队列的结果显示不同初始化方法之间没有差异。在所有设置中，ConvNeXt-B都优于ViT-B/16。使用冻结DINOv3-7B特征的模型性能低于完全微调的86-89M参数骨干网络，突显了领域适应的重要性。扩展到1024x1024并没有进一步提高准确性。分辨率相关的增益在边界依赖性和小焦点异常方面最为明显。在胸部X光检查中，更高的输入分辨率对于利用现代自监督模型的益处至关重要。512x512像素代表了一个实用的上限，其中DINOv3初始化的ConvNeXt-B网络提供最强性能，而更大的输入提供的成本回报最小。从临床角度来看，这些发现支持在512x512分辨率下使用微调的中型骨干网络进行胸部X光解释，预计在检测与急诊和重症监护环境相关的细微或边界为中心的病变时会有最大收益。


### 论文摘要

Self-supervised learning (SSL) has advanced visual representation learning, but its value in chest radiography, a high-volume imaging modality with fine-grained findings, remains unclear. Meta's DINOv3 extends earlier SSL models through Gram-anchored self-distillation. Whether these design choices improve transfer learning for chest radiography has not been systematically tested. We benchmarked DINOv3 against DINOv2 and ImageNet initialization across seven datasets (n>814,000). Two representative backbones were evaluated: ViT-B/16 and ConvNeXt-B. Images were analyzed at 224x224, 512x512, and 1024x1024 pixels. We additionally assessed frozen features from a 7B model. The primary outcome was mean AUROC across labels. At 224x224, DINOv3 and DINOv2 achieved comparable performance on adult datasets. Increasing resolution to 512x512 yielded consistent improvements for DINOv3 over both DINOv2 and ImageNet. In contrast, results in pediatric cohort showed no differences across initializations. Across all settings, ConvNeXt-B outperformed ViT-B/16. Models using frozen DINOv3-7B features underperformed relative to fully finetuned 86-89M-parameter backbones, highlighting the importance of domain adaptation. Scaling to 1024x1024 did not further improve accuracy. Resolution-related gains were most evident for boundary-dependent and small focal abnormalities. In chest radiography, higher input resolution is critical for leveraging the benefits of modern self-supervised models. 512x512 pixels represent a practical upper limit where DINOv3-initialized ConvNeXt-B networks provide the strongest performance, while larger inputs offer minimal return on cost. Clinically, these findings support use of finetuned, mid-sized backbones at 512x512 for chest radiograph interpretation, with the greatest gains expected in detecting subtle or boundary-centered lesions relevant to emergency and critical care settings.

---

## 32. Bridged Clustering for Representation Learning: Semi-Supervised Sparse Bridging

**论文链接:** [http://arxiv.org/abs/2510.07182v1](http://arxiv.org/abs/2510.07182v1)

**作者:** Patrick Peixuan Ye, Chen Shani, Ellen Vitercik

**发布时间:** 2025-10-08

### GPT解析

### 总结

Bridged Clustering是一种半监督框架，通过独立聚类输入和输出数据，并使用少量配对示例学习聚类间的稀疏桥梁，实现高效预测。

### 背景

传统半监督学习方法未充分利用输出数据，而密集传输方法缺乏稀疏性和可解释性。

### 目的

提出一种从任意未配对的输入X和输出Y数据集中学习预测器的半监督框架。

### 方法

首先独立地对X和Y进行聚类，然后使用少量配对示例学习聚类间的稀疏、可解释桥梁；推理时将新输入分配到最近输入聚类，返回链接输出聚类的质心作为预测值。

### 主要发现

方法明确利用仅输出数据，保持稀疏且可解释的对齐；在有界的错误聚类和错误桥接率下成为有效预测器；在低监督设置中与最先进方法具有竞争力。

### 结论

Bridged Clustering是一种有效的半监督学习方法，能够充分利用未配对的输入和输出数据，同时保持模型的可解释性和效率。

### 翻译

我们引入Bridged Clustering，一种半监督框架，可从任意未配对的输入X和输出Y数据集中学习预测器。我们的方法首先独立地对X和Y进行聚类，然后仅使用少量配对示例学习聚类之间的稀疏、可解释的桥梁。在推理时，新输入x被分配到其最近的输入聚类，链接输出聚类的质心被返回作为预测值ŷ。与传统SSL不同，Bridged Clustering明确利用仅输出数据，与密集传输方法不同，它保持稀疏且可解释的对齐。通过理论分析，我们表明在有界的错误聚类和错误桥接率下，我们的算法成为有效且高效的预测器。从经验上看，我们的方法与最先进方法具有竞争力，同时保持简单、模型无关和在低监督设置中的高标签效率。


### 论文摘要

We introduce Bridged Clustering, a semi-supervised framework to learn predictors from any unpaired input $X$ and output $Y$ dataset. Our method first clusters $X$ and $Y$ independently, then learns a sparse, interpretable bridge between clusters using only a few paired examples. At inference, a new input $x$ is assigned to its nearest input cluster, and the centroid of the linked output cluster is returned as the prediction $\hat{y}$. Unlike traditional SSL, Bridged Clustering explicitly leverages output-only data, and unlike dense transport-based methods, it maintains a sparse and interpretable alignment. Through theoretical analysis, we show that with bounded mis-clustering and mis-bridging rates, our algorithm becomes an effective and efficient predictor. Empirically, our method is competitive with SOTA methods while remaining simple, model-agnostic, and highly label-efficient in low-supervision settings.

---

## 33. Unified Molecule Pre-training with Flexible 2D and 3D Modalities: Single and Paired Modality Integration

**论文链接:** [http://arxiv.org/abs/2510.07035v1](http://arxiv.org/abs/2510.07035v1)

**作者:** Tengwei Song, Min Wu, Yuan Fang

**发布时间:** 2025-10-08

**DOI:** 10.1145/3746252.3761084

**备注:** CIKM 2025

### GPT解析

### 总结

FlexMol是一种灵活的分子预训练框架，能够学习统一的分子表示并支持单模态输入，克服了现有方法在缺少某种模态数据时的局限性

### 背景

分子表示学习在药物发现和材料设计等应用中发挥关键作用，现有方法需要成对的2D和3D分子数据进行预训练以捕捉全面的分子结构信息

### 目的

开发一种能够处理单模态输入的分子表示学习方法，解决现有方法在某种模态数据不可用或计算成本高昂时的局限性

### 方法

提出FlexMol框架，为2D和3D分子数据使用单独模型，通过参数共享提高计算效率，利用解码器生成缺失模态特征，实现多阶段连续学习过程

### 主要发现

FlexMol在多种分子属性预测任务上表现出色，实验证明其对不完整数据也有效

### 结论

FlexMol成功实现了统一分子表示学习并支持单模态输入，为分子表示学习提供了更灵活的解决方案

### 翻译

分子表示学习在推进药物发现和材料设计等应用中发挥着关键作用。现有工作利用分子信息的2D和3D模态进行预训练，旨在捕捉全面的结构和几何洞察。然而，这些方法需要成对的2D和3D分子数据来有效训练模型并防止其坍缩为单一模态，这在某种模态不可用或计算成本高昂的场景中存在限制。为了克服这一限制，我们提出了FlexMol，一个灵活的分子预训练框架，它学习统一的分子表示，同时支持单模态输入。具体而言，受视觉语言模型中统一结构的启发，我们的方法为2D和3D分子数据使用单独的模型，利用参数共享提高计算效率，并使用解码器生成缺失模态的特征。这使得多阶段连续学习过程成为可能，在训练期间两种模态协同贡献，同时确保在推理期间只有一种模态可用时的鲁棒性。大量实验表明，FlexMol在广泛的分子属性预测任务上实现了卓越的性能，我们也 empirically 证明了其在不完整数据上的有效性。我们的代码和数据可在 https://github.com/tewiSong/FlexMol 获取。


### 论文摘要

Molecular representation learning plays a crucial role in advancing applications such as drug discovery and material design. Existing work leverages 2D and 3D modalities of molecular information for pre-training, aiming to capture comprehensive structural and geometric insights. However, these methods require paired 2D and 3D molecular data to train the model effectively and prevent it from collapsing into a single modality, posing limitations in scenarios where a certain modality is unavailable or computationally expensive to generate. To overcome this limitation, we propose FlexMol, a flexible molecule pre-training framework that learns unified molecular representations while supporting single-modality input. Specifically, inspired by the unified structure in vision-language models, our approach employs separate models for 2D and 3D molecular data, leverages parameter sharing to improve computational efficiency, and utilizes a decoder to generate features for the missing modality. This enables a multistage continuous learning process where both modalities contribute collaboratively during training, while ensuring robustness when only one modality is available during inference. Extensive experiments demonstrate that FlexMol achieves superior performance across a wide range of molecular property prediction tasks, and we also empirically demonstrate its effectiveness with incomplete data. Our code and data are available at https://github.com/tewiSong/FlexMol.

---

## 34. Relational Database Distillation: From Structured Tables to Condensed Graph Data

**论文链接:** [http://arxiv.org/abs/2510.06980v1](http://arxiv.org/abs/2510.06980v1)

**作者:** Xinyi Gao, Jingxi Zhang, Lijian Chen, Tong Chen, Lizhen Cui, Hongzhi Yin

**发布时间:** 2025-10-08

### GPT解析

### 总结

这项研究提出了关系数据库蒸馏(RDD)问题，旨在将大规模关系数据库压缩为紧凑的异构图，同时保留预测能力。通过保留多模态列信息和主-外键关系，并设计了基于核脊回归的目标函数，该方法显著减少了数据大小，同时保持了在分类和回归任务上的竞争性能。

### 背景

关系数据库(RDBs)支撑着全球大多数数据管理系统，其中信息被结构化为多个相互依赖的表格。最近的进展利用图表示学习来捕获表格间的复杂关系作为多跳依赖，但这些方法由于数据库的巨大规模和表格间密集消息传递的计算负担，仍然面临存储开销过大和训练时间过长的问题。

### 目的

解决关系数据库在预测任务中面临的存储和计算效率问题，通过蒸馏方法将大规模关系数据库压缩为紧凑的异构图，同时保留足够的预测能力，使图模型能够高效训练和使用。

### 方法

提出了关系数据库蒸馏(RDD)方法，将大规模RDBs蒸馏为紧凑的异构图；通过节点特征保留多模态列信息；通过异构边编码主-外键关系，维护数据完整性和关系结构；设计基于核脊回归和伪标签的目标函数，避免传统低效的双级蒸馏框架，确保蒸馏后的图能适应多样化的下游任务。

### 主要发现

在多个真实世界RDBs上的大量实验表明，该解决方案显著减少了数据大小，同时在分类和回归任务上保持了竞争性能，为RDBs的可扩展学习创造了有效途径。

### 结论

关系数据库蒸馏方法能够有效解决大规模关系数据库在预测任务中的存储和计算效率问题，通过将数据库压缩为紧凑的异构图同时保留关键信息，为处理大规模关系数据库提供了可行的解决方案。

### 翻译

关系数据库(RDBs)支撑着全球大多数数据管理系统，其中信息被结构化为多个相互依赖的表格。为了有效利用RDBs中的知识进行预测任务，最近的进展利用图表示学习来捕获表格间的复杂关系作为多跳依赖。尽管取得了最先进的性能，但由于数据库的巨大规模和表格间密集消息传递的计算负担，这些方法仍然受到存储开销过大和训练时间过长的限制。为了缓解这些问题，我们提出了并研究了关系数据库蒸馏(RDD)问题。具体而言，我们旨在将大规模RDBs蒸馏为紧凑的异构图，同时保留训练基于图模型的预测能力所需的效用。多模态列信息通过节点特征保留，主-外键关系通过异构边编码，从而维护数据完整性和关系结构。为确保在不采用传统低效的双级蒸馏框架的情况下适应多样化的下游任务，我们进一步设计了基于核脊回归和伪标签的目标函数，为蒸馏后的图生成高质量特征。在多个真实世界RDBs上的大量实验表明，我们的解决方案显著减少了数据大小，同时在分类和回归任务上保持了竞争性能，为RDBs的可扩展学习创造了有效途径。


### 论文摘要

Relational databases (RDBs) underpin the majority of global data management systems, where information is structured into multiple interdependent tables. To effectively use the knowledge within RDBs for predictive tasks, recent advances leverage graph representation learning to capture complex inter-table relations as multi-hop dependencies. Despite achieving state-of-the-art performance, these methods remain hindered by the prohibitive storage overhead and excessive training time, due to the massive scale of the database and the computational burden of intensive message passing across interconnected tables. To alleviate these concerns, we propose and study the problem of Relational Database Distillation (RDD). Specifically, we aim to distill large-scale RDBs into compact heterogeneous graphs while retaining the predictive power (i.e., utility) required for training graph-based models. Multi-modal column information is preserved through node features, and primary-foreign key relations are encoded via heterogeneous edges, thereby maintaining both data fidelity and relational structure. To ensure adaptability across diverse downstream tasks without engaging the traditional, inefficient bi-level distillation framework, we further design a kernel ridge regression-guided objective with pseudo-labels, which produces quality features for the distilled graph. Extensive experiments on multiple real-world RDBs demonstrate that our solution substantially reduces the data size while maintaining competitive performance on classification and regression tasks, creating an effective pathway for scalable learning with RDBs.

---

## 35. Learning Global Representation from Queries for Vectorized HD Map Construction

**论文链接:** [http://arxiv.org/abs/2510.06969v1](http://arxiv.org/abs/2510.06969v1)

**作者:** Shoumeng Qiu, Xinrun Li, Yang Long, Xiangyang Xue, Varun Ojha, Jian Pu

**发布时间:** 2025-10-08

**备注:** 16 pages

### GPT解析

### 总结

本文提出了一种名为MapGR的新型架构，用于在线构建高清矢量地图，通过全局表示学习提高自动驾驶系统中地图构建的准确性和效率。

### 背景

在线构建高清地图是现代自动驾驶系统的基石。当前最先进方法基于DETR框架，将其作为实例检测问题处理，但这些方法依赖于独立的可学习对象查询，导致主要采用局部查询视角，忽视了高清地图中固有的全局表示。

### 目的

提出一种能够学习和利用查询中全局表示的架构，以改进高清地图的构建质量。

### 方法

提出MapGR（Global Representation learning for HD Map construction）架构，包含两个协同模块：1）全局表示学习模块，通过精心设计的整体分割任务使查询分布与全局地图更好对齐；2）全局表示指导模块，为每个查询提供显式的全局级上下文信息，促进优化过程。

### 主要发现

在nuScenes和Argoverse2数据集上的评估验证了MapGR方法的有效性，与最先进的基线相比，在平均精度均值方面有显著提高。

### 结论

通过引入全局表示学习机制，MapGR成功解决了传统方法中忽视全局表示的问题，显著提升了高清地图构建的性能。

### 翻译

在线构建高清矢量地图是现代自动驾驶系统的基石。最先进的方法，特别是基于DETR框架的方法，将其作为实例检测问题来处理。然而，它们依赖于独立的、可学习的对象查询，导致主要采用局部查询视角，忽视了高清地图中固有的全局表示。在这项工作中，我们提出了MapGR（高清地图的全局表示学习），这是一种旨在学习和利用查询中全局表示的架构。我们的方法引入了两个协同模块：全局表示学习模块，通过精心设计的整体分割任务，鼓励所有查询的分布更好地与全局地图对齐；以及全局表示指导模块，为每个单独的查询提供显式的全局级上下文信息，促进其优化。在nuScenes和Argoverse2数据集上的评估验证了我们方法的有效性，与最先进的基线相比，平均精度均值有显著提高。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在线高清地图（HD Map）构建中的全局表示学习问题。当前基于DETR框架的方法主要将地图构建视为实例检测问题，但它们依赖于独立的、可学习的对象查询，导致查询主要具有局部视角，忽略了高清地图中固有的全局表示。这个问题在现实中非常重要，因为在线高清地图构建是现代自动驾驶系统的基石，它为安全高效的导航提供高精度的地图元素感知。与传统高清地图相比，在线高清地图能更好地适应不断变化的道路条件，如施工区域、车道修改和意外障碍。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有DETR-like框架在HD地图构建中的局限性，特别是它们处理'连续分布'（HD地图在BEV空间中的空间连续性）时的不足，而DETR框架主要针对'独立分布'（独立目标的空间分布）进行了优化。作者发现现有方法通过手动实例分区来应用DETR框架，但这会导致信息损失并忽略全局结构信息。因此，作者提出从所有对象查询中直接学习全局HD地图表示，然后利用这个表示促进每个单独查询的学习。该方法借鉴了DETR框架及其变体（如Deformable DETR、Conditional DETR等）以及HD地图构建领域的现有工作（如MapTR、VectorMapNet等），但创新性地引入了全局表示学习思想。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是从所有查询中学习全局地图表示，而不是仅仅关注单个实例，并通过全局表示来指导局部查询的优化，使每个查询在优化的同时保持全局视角。整体实现流程包括：1)输入处理：从多视角图像提取特征并投影到BEV空间；2)查询处理：使用多层Transformer解码器解码实例预测；3)全局表示学习(GRL)模块：将查询聚合为全局表示并计算与地面真实地图的损失；4)全局表示指导(GRG)模块：将全局信息融入每个查询并使用融合后的查询进行最终预测；5)输出：生成向量化的高清地图元素。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出基于全局表示学习的HD地图构建方法MapGR；2)设计全局表示学习(GRL)模块，增强查询的全局分布学习；3)提出全局表示指导(GRG)模块，通过全局信息指导单个查询优化。相比之前工作的不同：1)采用全局视角而非局部视角，考虑了地图元素间的空间关系和结构依赖性；2)通过全局表示将梯度传播到所有实例查询，而非仅成功匹配的预测；3)学习全局地图表示而非仅实例级表示；4)设计为即插即用模块，可与现有方法如MapTR系列兼容；5)专门针对HD地图的'连续分布'特性进行优化，而非DETR框架主要针对的'独立分布'。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过从查询中学习全局表示来增强高清地图构建的方法，引入了全局表示学习和全局表示指导两个互补模块，显著提升了地图构建的准确性和一致性，同时保持了计算效率。'}


### 论文摘要

The online construction of vectorized high-definition (HD) maps is a cornerstone of modern autonomous driving systems. State-of-the-art approaches, particularly those based on the DETR framework, formulate this as an instance detection problem. However, their reliance on independent, learnable object queries results in a predominantly local query perspective, neglecting the inherent global representation within HD maps. In this work, we propose \textbf{MapGR} (\textbf{G}lobal \textbf{R}epresentation learning for HD \textbf{Map} construction), an architecture designed to learn and utilize a global representations from queries. Our method introduces two synergistic modules: a Global Representation Learning (GRL) module, which encourages the distribution of all queries to better align with the global map through a carefully designed holistic segmentation task, and a Global Representation Guidance (GRG) module, which endows each individual query with explicit, global-level contextual information to facilitate its optimization. Evaluations on the nuScenes and Argoverse2 datasets validate the efficacy of our approach, demonstrating substantial improvements in mean Average Precision (mAP) compared to leading baselines.

---

## 36. Angular Constraint Embedding via SpherePair Loss for Constrained Clustering

**论文链接:** [http://arxiv.org/abs/2510.06907v1](http://arxiv.org/abs/2510.06907v1)

**作者:** Shaojie Zhang, Ke Chen

**发布时间:** 2025-10-08

**备注:** Accepted by NeurIPS 2025, 6 Figures and 1 Table in Main text, 18  Figures and 5 Tables in Appendices

### GPT解析

### 总结

本文提出了一种名为SpherePair的新型角度约束嵌入方法，用于深度受约束聚类(DCC)，解决了现有方法在端到端建模中受限于锚点或难以学习判别性欧几里得嵌入的问题。

### 背景

受约束聚类通过成对约束整合领域知识，但现有深度受约束聚类方法要么受限于端到端建模中的锚点，要么难以学习判别性欧几里得嵌入，限制了它们的可扩展性和实际应用性。

### 目的

为了避免现有DCC方法的缺陷，提出一种新颖的角度约束嵌入方法，实现更有效的聚类表示学习。

### 方法

使用SpherePair损失和几何公式，该方法忠实地编码成对约束，产生在角度空间中聚类友好的嵌入，有效分离表示学习与聚类过程。

### 主要发现

SpherePair能够在无冲突情况下保留成对关系，无需指定确切聚类数量，可推广到未见数据，能快速推断聚类数量，并有严格理论保证支持。

### 结论

与最先进DCC方法在各种基准上的比较评估及理论见解的经验验证，证实了SpherePair的优越性能、可扩展性和实际有效性。

### 翻译

受约束聚类通过成对约束整合领域知识。然而，现有的深度受约束聚类方法要么受限于端到端建模中的锚点，要么难以学习判别性欧几里得嵌入，限制了它们的可扩展性和实际应用性。为了避免各自的缺陷，我们提出了一种新颖的DCC角度约束嵌入方法，称为SpherePair。使用SpherePair损失和几何公式，我们的方法忠实地编码成对约束，并导致在角度空间中聚类友好的嵌入，有效地将表示学习与聚类分离。SpherePair在没有冲突的情况下保留成对关系，无需指定确切的聚类数量，可以推广到未见数据，能够快速推断聚类数量，并有严格的理论保证支持。与最先进的DCC方法在各种基准上的比较评估，以及对理论见解的经验验证，证实了其优越的性能、可扩展性和整体实际有效性。代码可在我们的仓库获取。


### 论文摘要

Constrained clustering integrates domain knowledge through pairwise constraints. However, existing deep constrained clustering (DCC) methods are either limited by anchors inherent in end-to-end modeling or struggle with learning discriminative Euclidean embedding, restricting their scalability and real-world applicability. To avoid their respective pitfalls, we propose a novel angular constraint embedding approach for DCC, termed SpherePair. Using the SpherePair loss with a geometric formulation, our method faithfully encodes pairwise constraints and leads to embeddings that are clustering-friendly in angular space, effectively separating representation learning from clustering. SpherePair preserves pairwise relations without conflict, removes the need to specify the exact number of clusters, generalizes to unseen data, enables rapid inference of the number of clusters, and is supported by rigorous theoretical guarantees. Comparative evaluations with state-of-the-art DCC methods on diverse benchmarks, along with empirical validation of theoretical insights, confirm its superior performance, scalability, and overall real-world effectiveness. Code is available at \href{https://github.com/spherepaircc/SpherePairCC/tree/main}{our repository}.

---

## 37. 论文ID: 2510.06842v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.06842v1.json'

---

## 38. Dual Goal Representations

**论文链接:** [http://arxiv.org/abs/2510.06714v1](http://arxiv.org/abs/2510.06714v1)

**作者:** Seohong Park, Deepinder Mann, Sergey Levine

**发布时间:** 2025-10-08

### GPT解析

### 总结

这篇论文提出了一种用于目标条件强化学习的双重目标表示方法，该方法通过状态与其他所有状态的时间距离关系来表征状态，具有理论优势并能提升目标到达性能。

### 背景

目标条件强化学习(GCRL)领域需要更有效的状态表示方法，以捕捉环境动态并提高目标到达性能。

### 目的

开发一种基于状态间时间距离关系的双重目标表示方法，该方法能够捕捉环境内在动态，提供足够信息以恢复最优目标到达策略，并过滤外部噪声。

### 方法

提出双重目标表示，将状态表征为'从所有其他状态的时间距离的集合'；基于此概念开发了一种可结合任何现有GCRL算法的目标表示学习方法。

### 主要发现

双重目标表示具有两个重要理论特性：1)仅依赖于环境的内在动态，对原始状态表示不变；2)包含足够信息以恢复最优目标到达策略，同时能过滤外部噪声。在OGBench任务套件上的20个基于状态和像素的实验中，双重目标表示一致提高了离线目标到达性能。

### 结论

双重目标表示是一种有效的目标条件强化学习方法，能够提升各种任务中的目标到达性能，且可与现有GCRL算法结合使用。

### 翻译

在这项工作中，我们为目标条件强化学习(GCRL)引入了双重目标表示。双重目标表示将状态定义为'从所有其他状态的时间距离的集合'；换句话说，它通过状态与其他每个状态的关系（以时间距离测量）来编码状态。这种表示提供了几个有吸引力的理论特性。首先，它仅依赖于环境的内在动态，且对原始状态表示不变。其次，它包含足够的信息来恢复最优的目标到达策略，同时能够过滤外部噪声。基于这一概念，我们开发了一种实用的目标表示学习方法，可以与任何现有的GCRL算法结合。通过在OGBench任务套件上的多样化实验，我们经验性地表明，双重目标表示在20个基于状态和像素的任务中一致地提高了离线目标到达性能。


### 论文摘要

In this work, we introduce dual goal representations for goal-conditioned reinforcement learning (GCRL). A dual goal representation characterizes a state by "the set of temporal distances from all other states"; in other words, it encodes a state through its relations to every other state, measured by temporal distance. This representation provides several appealing theoretical properties. First, it depends only on the intrinsic dynamics of the environment and is invariant to the original state representation. Second, it contains provably sufficient information to recover an optimal goal-reaching policy, while being able to filter out exogenous noise. Based on this concept, we develop a practical goal representation learning method that can be combined with any existing GCRL algorithm. Through diverse experiments on the OGBench task suite, we empirically show that dual goal representations consistently improve offline goal-reaching performance across 20 state- and pixel-based tasks.

---

## 39. Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer

**论文链接:** [http://arxiv.org/abs/2510.06691v1](http://arxiv.org/abs/2510.06691v1)

**作者:** Jing-Zong Zhang, Shuang Guo, Li-Lin Zhu, Lingxiao Wang, Guo-Liang Ma

**发布时间:** 2025-10-08

**备注:** 10 pages, 5 figures, accepted at the NeurIPS 2025 workshop "Machine  Learning and the Physical Sciences"

### GPT解析

### 总结

该研究提出了一种基于Transformer的自编码器方法，通过两阶段训练范式从重离子碰撞数据中提取信息特征，显著提高了分类准确率并捕获了复杂非线性相关性。

### 背景

高能核物理的核心挑战是从重离子碰撞的高维最终态数据中提取信息特征，传统方法依赖选定可观测量，可能遗漏数据中细微但物理相关的结构。

### 目的

开发一种能够从重离子碰撞数据中提取信息丰富特征的方法，以实现可靠的下游分析和物理现象研究。

### 方法

引入基于Transformer的自编码器，采用两阶段训练范式：1)自监督预训练，从无标签HIC数据学习潜在表示；2)监督微调，适应特定物理任务。应用主成分分析和SHAP解释评估特征质量。

### 主要发现

1)该方法在大和小碰撞系统分类中显著优于PointNet；2)自编码器捕获了超越单个可观测量的复杂非线性相关性；3)产生的特征具有强判别力和解释力。

### 结论

两阶段框架作为HIC中特征学习的通用和稳健基础，为分析夸克-胶子等离子体特性和其他新兴现象提供了更强大的分析工具。

### 翻译

高能核物理中的一个核心挑战是从重离子碰撞的高维最终态数据中提取信息特征，以实现可靠的下游分析。传统方法通常依赖于选定的可观测量，可能会遗漏数据中细微但物理相关的结构。为此，我们引入了一种基于Transformer的自编码器，采用两阶段范式进行训练：自监督预训练 followed by 监督微调。预训练编码器直接从无标签的HIC数据中学习潜在表示，提供紧凑且信息丰富的特征空间，可以适应不同的物理任务。作为案例研究，我们将该方法应用于区分大碰撞系统和小碰撞系统，其分类准确率显著高于PointNet。主成分分析和SHAP解释进一步表明，该自编码器捕获了超越单个可观测量的复杂非线性相关性，产生了具有强判别力和解释力的特征。这些结果建立了我们的两阶段框架作为HIC中特征学习的通用和稳健基础，为分析夸克-胶子等离子体特性和其他新兴现象打开了更强大分析的大门。实现在GitHub上公开：https://github.com/Giovanni-Sforza/MaskPoint-AMPT。


### 论文摘要

A central challenge in high-energy nuclear physics is to extract informative features from the high-dimensional final-state data of heavy-ion collisions (HIC) in order to enable reliable downstream analyses. Traditional approaches often rely on selected observables, which may miss subtle but physically relevant structures in the data. To address this, we introduce a Transformer-based autoencoder trained with a two-stage paradigm: self-supervised pre-training followed by supervised fine-tuning. The pretrained encoder learns latent representations directly from unlabeled HIC data, providing a compact and information-rich feature space that can be adapted to diverse physics tasks. As a case study, we apply the method to distinguish between large and small collision systems, where it achieves significantly higher classification accuracy than PointNet. Principal component analysis and SHAP interpretation further demonstrate that the autoencoder captures complex nonlinear correlations beyond individual observables, yielding features with strong discriminative and explanatory power. These results establish our two-stage framework as a general and robust foundation for feature learning in HIC, opening the door to more powerful analyses of quark--gluon plasma properties and other emergent phenomena. The implementation is publicly available at https://github.com/Giovanni-Sforza/MaskPoint-AMPT.

---

## 40. The Effect of Label Noise on the Information Content of Neural Representations

**论文链接:** [http://arxiv.org/abs/2510.06401v1](http://arxiv.org/abs/2510.06401v1)

**作者:** Ali Hussaini Umar, Franky Kevin Nando Tezoh, Jean Barbier, Santiago Acevedo, Alessandro Laio

**发布时间:** 2025-10-07

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本研究探讨了标签噪声对深度学习模型隐藏表示的影响，发现隐藏表示的信息内容随网络参数数量变化呈现双下降行为，且过参数化网络的表示对标签噪声具有鲁棒性。

### 背景

在监督分类任务中，模型训练用于预测数据点标签，但现实数据集中的标签常因标注错误而存在噪声。虽然标签噪声对深度学习模型性能的影响已被广泛研究，但它对网络隐藏表示的影响仍不明确。

### 目的

填补标签噪声对网络隐藏表示影响的研究空白，通过系统比较隐藏表示来理解这一问题。

### 方法

使用信息不平衡（Information Imbalance）作为条件互值的计算高效代理，系统比较不同情况下的隐藏表示。

### 主要发现

1) 隐藏表示的信息内容随网络参数数量变化呈现双下降行为；2) 在欠参数化情况下，噪声标签学习的表示比干净标签学习的更具信息量；3) 在过参数化情况下，两种表示具有相同信息量；4) 过参数化网络的表示对标签噪声具有鲁棒性；5) 交叉熵损失可减少倒数第二层与softmax前层间的信息不平衡；6) 随机标签学习的表示比随机特征表现更差。

### 结论

过参数化网络的表示对标签噪声具有鲁棒性，这为理解分类任务中的泛化提供了新视角。训练随机标签使网络超越懒学习，权重会适应以编码标签信息。

### 翻译

在监督分类任务中，模型被训练来预测每个数据点的标签。在现实世界的数据集中，这些标签常常由于标注错误而存在噪声。虽然标签噪声对深度学习模型性能的影响已被广泛研究，但它对网络隐藏表示的影响仍知之甚少。我们通过使用信息不平衡（一种条件互值的计算高效代理）系统比较隐藏表示来填补这一知识空白。通过这种分析，我们观察到隐藏表示的信息内容作为网络参数数量的函数呈现出双下降行为，类似于测试误差的行为。我们进一步证明，在欠参数化情况下，使用噪声标签学习的表示比使用干净标签学习的表示更具信息量，而在过参数化情况下，这些表示具有相同的信息量。我们的结果表明，过参数化网络的表示对标签噪声具有鲁棒性。我们还发现在过参数化情况下，使用交叉熵损失时，倒数第二层和softmax前层之间的信息不平衡会减少。这为理解分类任务中的泛化提供了新视角。将我们的分析扩展到从随机标签学习的表示，我们表明这些表示比随机特征表现更差。这表明在随机标签上的训练使网络远远超出懒学习，因为权重会适应以编码标签信息。


### 论文摘要

In supervised classification tasks, models are trained to predict a label for each data point. In real-world datasets, these labels are often noisy due to annotation errors. While the impact of label noise on the performance of deep learning models has been widely studied, its effects on the networks' hidden representations remain poorly understood. We address this gap by systematically comparing hidden representations using the Information Imbalance, a computationally efficient proxy of conditional mutual information. Through this analysis, we observe that the information content of the hidden representations follows a double descent as a function of the number of network parameters, akin to the behavior of the test error. We further demonstrate that in the underparameterized regime, representations learned with noisy labels are more informative than those learned with clean labels, while in the overparameterized regime, these representations are equally informative. Our results indicate that the representations of overparameterized networks are robust to label noise. We also found that the information imbalance between the penultimate and pre-softmax layers decreases with cross-entropy loss in the overparameterized regime. This offers a new perspective on understanding generalization in classification tasks. Extending our analysis to representations learned from random labels, we show that these perform worse than random features. This indicates that training on random labels drives networks much beyond lazy learning, as weights adapt to encode labels information.

---

## 41. Type and Complexity Signals in Multilingual Question Representations

**论文链接:** [http://arxiv.org/abs/2510.06304v1](http://arxiv.org/abs/2510.06304v1)

**作者:** Robin Kokot, Wessel Poelman

**发布时间:** 2025-10-07

**备注:** Workshop on Multilingual Representation Learning at EMNLP 2025

### GPT解析

### 总结

这项研究探讨了多语言transformer模型如何表示问题的形态句法特性，通过引入QTC数据集和扩展探测方法，比较了不同模型的表现，并分析了上下文表示与统计基线的优劣。

### 背景

多语言transformer模型在处理不同语言问题时，其表示形态句法特性的能力尚不清楚，需要系统性的研究。

### 目的

研究多语言transformer模型对问题形态句法特性的表示能力，评估不同方法在问题分类和复杂性分析中的表现。

### 方法

引入了包含七种语言的问题类型和复杂性(QTC)数据集，标注了类型信息和多种复杂性指标；扩展了探测方法，使用选择性控制处理回归标签；比较了冻结Glot500-m模型的分层探测、子词TF-IDF基线和微调模型的性能。

### 主要发现

在具有显式标记的语言中，统计特征能有效分类问题；神经探测能更好地捕获细粒度的结构性复杂性模式；研究结果帮助评估了上下文表示何时优于统计基线，以及参数更新对预训练语言信息可用性的影响。

### 结论

多语言transformer模型能够有效表示问题的形态句法特性，但不同方法在不同语言和任务上有各自的优势，需要根据具体需求选择合适的方法。

### 翻译

这项工作研究了多语言transformer模型如何表示问题的形态句法特性。我们引入了包含七种语言句子的问题类型和复杂性(QTC)数据集，并标注了类型信息和复杂性指标，包括依赖长度、树深度和词汇密度。我们的评估扩展了探测方法，使用选择性控制来处理回归标签，以量化泛化能力的提升。我们比较了在冻结的Glot500-m表示上的分层探测与子词TF-IDF基线以及微调模型的性能。结果表明，在具有显式标记的语言中，统计特征能有效分类问题，而神经探测能更好地捕获细粒度的结构性复杂性模式。我们使用这些结果来评估上下文表示何时优于统计基线，以及参数更新是否减少预训练语言信息的可用性。


### 论文摘要

This work investigates how a multilingual transformer model represents morphosyntactic properties of questions. We introduce the Question Type and Complexity (QTC) dataset with sentences across seven languages, annotated with type information and complexity metrics including dependency length, tree depth, and lexical density. Our evaluation extends probing methods to regression labels with selectivity controls to quantify gains in generalizability. We compare layer-wise probes on frozen Glot500-m (Imani et al., 2023) representations against subword TF-IDF baselines, and a fine-tuned model. Results show that statistical features classify questions effectively in languages with explicit marking, while neural probes capture fine-grained structural complexity patterns better. We use these results to evaluate when contextual representations outperform statistical baselines and whether parameter updates reduce the availability of pre-trained linguistic information.

---

## 42. Evaluating Fundus-Specific Foundation Models for Diabetic Macular Edema Detection

**论文链接:** [http://arxiv.org/abs/2510.07277v1](http://arxiv.org/abs/2510.07277v1)

**作者:** Franco Javier Arellano, José Ignacio Orlando

**发布时间:** 2025-10-08

**备注:** Accepted for publication at SIPAIM 2025

### GPT解析

### 总结

本研究比较了基础模型和传统CNN在糖尿病性黄斑水肿检测中的性能，发现轻量级CNN在大多数情况下表现更好。

### 背景

糖尿病性黄斑水肿是糖尿病患者视力丧失的主要原因。虽然深度学习在自动检测方面有潜力，但受限于标注数据的稀缺性。基础模型被视为替代方案，但其在DME检测中的效果尚不明确。

### 目的

系统比较不同基础模型和标准迁移学习方法在DME检测任务上的表现，特别是比较RETFound、FLAIR和EfficientNet-B0三种模型。

### 方法

在IDRiD、MESSIDOR-2和OCT-and-Eye-Fundus-Images三个数据集上，采用不同的训练方案和评估设置对RETFound、FLAIR和EfficientNet-B0进行比较。

### 主要发现

尽管基础模型规模较大，但并不始终优于微调的CNN。EfficientNet-B0在大多数评估中表现最佳，RETFound仅在OEFI数据集上表现良好，而FLAIR在零样本设置下表现出色。

### 结论

基础模型可能不适合精细的眼科任务如DME检测，即使在微调后也是如此，轻量级CNN在数据稀缺环境中仍是强大的基线。

### 翻译

糖尿病性黄斑水肿(DME)是糖尿病视网膜病变(DR)患者视力丧失的主要原因。虽然深度学习在从眼底图像自动检测这种情况方面显示出有希望的结果，但由于标注数据的有限可用性，其应用仍然具有挑战性。基础模型(FM)已成为一种替代解决方案。然而，目前尚不清楚它们是否特别能够应对DME检测。在本文中，我们系统地比较了不同的基础模型和标准的迁移学习方法用于此任务。具体来说，我们在IDRiD、MESSIDOR-2和OCT-and-Eye-Fundus-Images(OEFI)上，比较了两种最流行的视网膜图像基础模型--RETFound和FLAIR，以及一个EfficientNet-B0骨干网络，采用不同的训练方案和评估设置。结果表明，尽管规模较大，基础模型在该任务上并不始终优于微调的CNN。特别是，EfficientNet-B0在大多数评估设置中，根据ROC曲线下面积和精确度/召回率曲线排名第一或第二，而RETFound仅在OEFI中显示出有希望的结果。另一方面，FLAIR展示了有竞争力的零样本性能，在适当提示时实现了显著的AUC-PR分数。这些发现表明，基础模型可能不是精细的眼科任务(如DME检测)的良好工具，即使在微调之后，这表明在数据稀缺环境中，轻量级CNN仍然是强大的基线。


### 论文摘要

Diabetic Macular Edema (DME) is a leading cause of vision loss among patients with Diabetic Retinopathy (DR). While deep learning has shown promising results for automatically detecting this condition from fundus images, its application remains challenging due the limited availability of annotated data. Foundation Models (FM) have emerged as an alternative solution. However, it is unclear if they can cope with DME detection in particular. In this paper, we systematically compare different FM and standard transfer learning approaches for this task. Specifically, we compare the two most popular FM for retinal images--RETFound and FLAIR--and an EfficientNet-B0 backbone, across different training regimes and evaluation settings in IDRiD, MESSIDOR-2 and OCT-and-Eye-Fundus-Images (OEFI). Results show that despite their scale, FM do not consistently outperform fine-tuned CNNs in this task. In particular, an EfficientNet-B0 ranked first or second in terms of area under the ROC and precision/recall curves in most evaluation settings, with RETFound only showing promising results in OEFI. FLAIR, on the other hand, demonstrated competitive zero-shot performance, achieving notable AUC-PR scores when prompted appropriately. These findings reveal that FM might not be a good tool for fine-grained ophthalmic tasks such as DME detection even after fine-tuning, suggesting that lightweight CNNs remain strong baselines in data-scarce environments.

---

## 43. Chem-NMF: Multi-layer $α$-divergence Non-Negative Matrix Factorization for Cardiorespiratory Disease Clustering, with Improved Convergence Inspired by Chemical Catalysts and Rigorous Asymptotic Analysis

**论文链接:** [http://arxiv.org/abs/2510.06632v1](http://arxiv.org/abs/2510.06632v1)

**作者:** Yasaman Torabi, Shahram Shirani, James P. Reilly

**发布时间:** 2025-10-08

### GPT解析

### 总结

本研究提出了一种名为Chem-NMF的新型非负矩阵分解方法，通过引入有界因子解决多层架构中的收敛性问题，并首次从物理化学角度分析NMF算法的收敛行为。

### 背景

非负矩阵分解(NMF)是一种无监督学习方法，在音频处理、生物医学信号分析和图像识别等领域提供低秩表示。将α-散度融入NMF公式可提高优化灵活性，但扩展到多层架构时确保收敛性存在挑战。

### 目的

解决NMF在多层架构中的收敛性问题，并从物理化学角度对NMF算法的收敛行为进行严格分析。

### 方法

提出了一种名为Chem-NMF的新方法，受到化学反应中能垒玻尔兹曼概率的启发，引入了有界因子来稳定收敛。从数学上证明了渐进收敛结果，并将其应用于真实数据。

### 主要发现

实验结果表明，所提出的算法在生物医学信号上聚类准确率提高了5.6%±2.7%，在人脸图像上提高了11.1%±7.2%（平均值±标准差）。

### 结论

Chem-NMF方法有效解决了NMF在多层架构中的收敛性问题，并首次从物理化学角度对NMF算法的收敛行为进行了严格分析。

### 翻译

非负矩阵分解(NMF)是一种无监督学习方法，在音频处理、生物医学信号分析和图像识别等各个领域提供低秩表示。在NMF公式中融入α-散度提高了优化的灵活性，然而将这些方法扩展到多层架构时，确保收敛性存在挑战。为此，我们受化学反应中能垒玻尔兹曼概率的启发，引入了一种新方法来理论上进行收敛分析，称为Chem-NMF，它包含一个稳定收敛的有界因子。据我们所知，这是首次从物理化学角度对NMF算法的收敛行为进行严格分析的研究。我们从数学上证明了渐进收敛结果，然后展示了它们如何应用于真实数据。实验结果表明，所提出的算法在生物医学信号上的聚类准确率提高了5.6%±2.7%，在人脸图像上提高了11.1%±7.2%（平均值±标准差）。


### 论文摘要

Non-Negative Matrix Factorization (NMF) is an unsupervised learning method offering low-rank representations across various domains such as audio processing, biomedical signal analysis, and image recognition. The incorporation of $\alpha$-divergence in NMF formulations enhances flexibility in optimization, yet extending these methods to multi-layer architectures presents challenges in ensuring convergence. To address this, we introduce a novel approach inspired by the Boltzmann probability of the energy barriers in chemical reactions to theoretically perform convergence analysis. We introduce a novel method, called Chem-NMF, with a bounding factor which stabilizes convergence. To our knowledge, this is the first study to apply a physical chemistry perspective to rigorously analyze the convergence behaviour of the NMF algorithm. We start from mathematically proven asymptotic convergence results and then show how they apply to real data. Experimental results demonstrate that the proposed algorithm improves clustering accuracy by 5.6% $\pm$ 2.7% on biomedical signals and 11.1% $\pm$ 7.2% on face images (mean $\pm$ std).

---

## 44. Self-supervised Physics-guided Model with Implicit Representation Regularization for Fast MRI Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.06611v1](http://arxiv.org/abs/2510.06611v1)

**作者:** Jingran Xu, Yuanyuan Liu, Yanjie Zhu

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了一种名为UnrollINR的新型零样本自监督MRI重建框架，能够在不依赖外部训练数据的情况下实现特定扫描的MRI重建，结合了深度展开结构和隐式神经表示的优势，在高加速率下表现出色。

### 背景

磁共振成像（MRI）是重要的临床诊断工具，但其广泛应用受到扫描时间长的限制。快速MRI重建技术通过从欠采样的k空间数据中重建高保真MR图像来有效减少采集时间。近年来，基于深度学习的方法在该领域取得了显著进展，特别是在难以获取完全采样数据的场景中，自监督和无监督学习方法证明特别有价值。

### 目的

开发一种不依赖外部训练数据的零样本自监督MRI重建框架，提高MRI重建的性能和可解释性。

### 方法

提出名为UnrollINR的新型零样本自监督重建框架，采用物理引导的展开迭代重建架构，并将隐式神经表示（INR）作为正则化先验来有效约束解空间。通过结合深度展开结构与INR的强大隐式表示能力，增强了模型的解释性和重建性能。

### 主要发现

实验结果表明，即使在10的高加速率下，UnrollINR相比监督学习方法也能实现优越的重建性能，验证了所提出方法的优越性。

### 结论

UnrollINR框架通过结合深度展开结构和隐式神经表示，实现了无需外部训练数据的MRI重建，在高加速率下仍能保持高质量的重建效果。

### 翻译

磁共振成像（MRI）是重要的临床诊断工具，但其广泛应用受到扫描时间长的限制。快速MRI重建技术通过从欠采样的k空间数据中重建高保真MR图像来有效减少采集时间。近年来，基于深度学习的方法在该领域取得了显著进展，特别是在难以获取完全采样数据的场景中，自监督和无监督学习方法证明特别有价值。本文提出了一种名为UnrollINR的新型零样本自监督重建框架，它能够在不依赖外部训练数据的情况下实现特定扫描的MRI重建。该方法采用物理引导的展开迭代重建架构，并将隐式神经表示（INR）作为正则化先验来有效约束解空间。通过结合深度展开结构与INR的强大隐式表示能力，增强了模型的解释性和重建性能。实验结果表明，即使在10的高加速率下，UnrollINR相比监督学习方法也能实现优越的重建性能，验证了所提出方法的优越性。


### 论文摘要

Magnetic Resonance Imaging (MRI) is a vital clinical diagnostic tool, yet its widespread application is limited by prolonged scan times. Fast MRI reconstruction techniques effectively reduce acquisition duration by reconstructing high-fidelity MR images from undersampled k-space data. In recent years, deep learning-based methods have demonstrated remarkable progress in this field, with self-supervised and unsupervised learning approaches proving particularly valuable in scenarios where fully sampled data are difficult to obtain. This paper proposes a novel zero-shot self-supervised reconstruction framework named UnrollINR, which enables scan-specific MRI reconstruction without relying on external training data. The method adopts a physics-guided unrolled iterative reconstruction architecture and introduces Implicit Neural Representation (INR) as a regularization prior to effectively constrain the solution space. By combining a deep unrolled structure with the powerful implicit representation capability of INR, the model's interpretability and reconstruction performance are enhanced. Experimental results demonstrate that even at a high acceleration rate of 10, UnrollINR achieves superior reconstruction performance compared to the supervised learning method, validating the superiority of the proposed method.

---

## 45. TransFIRA: Transfer Learning for Face Image Recognizability Assessment

**论文链接:** [http://arxiv.org/abs/2510.06353v1](http://arxiv.org/abs/2510.06353v1)

**作者:** Allen Tu, Kartik Narayan, Joshua Gleason, Jennifer Xu, Matthew Meyn, Tom Goldstein, Vishal M. Patel

**发布时间:** 2025-10-07

**备注:** Project Page: https://transfira.github.io/

### GPT解析

### 总结

TransFIRA是一个轻量级且无需注释的框架，将人脸图像可识别性直接嵌入到嵌入空间中，通过类中心相似度和类中心角分离度定义可识别性，实现了最先进的验证准确性，并扩展到人体识别领域。

### 背景

在不受约束环境中进行人脸识别时，需应对姿态、模糊、光照和遮挡的极端变化，传统视觉质量指标无法预测输入对编码器的可识别性，现有FIQA方法通常依赖视觉启发式、人工注释或计算密集型生成流程，导致预测与编码器决策几何分离。

### 目的

引入TransFIRA框架，将可识别性直接嵌入到嵌入空间中，提供一种无需注释的可识别性评估方法。

### 方法

TransFIRA通过三个方面的进步实现目标：(i)使用类中心相似度(CCS)和类中心角分离度(CCAS)定义可识别性；(ii)采用可识别性感知的聚合策略，无需外部标签或特定训练；(iii)扩展到人脸以外的领域，包括编码器可解释性和可识别性感知的人体识别评估。

### 主要发现

TransFIRA在BRIAR和IJB-C数据集上实现了最先进的验证准确性，与真实可识别性的相关性几乎翻倍，在人脸识别和人体识别方面均表现出色，且在跨数据集转移下具有鲁棒性。

### 结论

TransFIRA作为一个统一的、几何驱动的可识别性评估框架，特定于编码器、准确、可解释且可跨模态扩展，显著提高了FIQA的准确性、可解释性和应用范围。

### 翻译

在不受约束的环境（如监控、视频和网络图像）中进行人脸识别必须应对姿态、模糊、光照和遮挡的极端变化，传统的视觉质量指标无法预测输入对部署的编码器是否真正可识别。现有的FIQA方法通常依赖视觉启发式、人工注释或计算密集型的生成流程，使其预测与编码器的决策几何分离。我们引入TransFIRA（人脸图像可识别性评估的迁移学习），一个轻量级且无需注释的框架，将可识别性直接嵌入到嵌入空间中。TransFIRA提供了三个方面的进步：(i)通过类中心相似度和类中心角分离度定义可识别性，产生第一个自然、决策边界对齐的过滤和加权标准；(ii)一种可识别性感知的聚合策略，在BRIAR和IJB-C上实现了最先进的验证准确性，同时与真实可识别性的相关性几乎翻倍，无需外部标签、启发式或特定于骨干网络的训练；以及(iii)超越人脸的新扩展，包括基于编码器的可解释性，揭示降级和主题特定因素如何影响可识别性，以及第一个可识别性感知的人体识别评估。实验验证了TransFIRA在人脸识别方面取得了最先进的结果，在人体识别方面表现出色，并且在跨数据集转移下具有鲁棒性。这些贡献共同确立了TransFIRA作为一个统一的、几何驱动的可识别性评估框架——特定于编码器、准确、可解释且可跨模态扩展——显著提高了FIQA的准确性、可解释性和范围。


### 论文摘要

Face recognition in unconstrained environments such as surveillance, video, and web imagery must contend with extreme variation in pose, blur, illumination, and occlusion, where conventional visual quality metrics fail to predict whether inputs are truly recognizable to the deployed encoder. Existing FIQA methods typically rely on visual heuristics, curated annotations, or computationally intensive generative pipelines, leaving their predictions detached from the encoder's decision geometry. We introduce TransFIRA (Transfer Learning for Face Image Recognizability Assessment), a lightweight and annotation-free framework that grounds recognizability directly in embedding space. TransFIRA delivers three advances: (i) a definition of recognizability via class-center similarity (CCS) and class-center angular separation (CCAS), yielding the first natural, decision-boundary--aligned criterion for filtering and weighting; (ii) a recognizability-informed aggregation strategy that achieves state-of-the-art verification accuracy on BRIAR and IJB-C while nearly doubling correlation with true recognizability, all without external labels, heuristics, or backbone-specific training; and (iii) new extensions beyond faces, including encoder-grounded explainability that reveals how degradations and subject-specific factors affect recognizability, and the first recognizability-aware body recognition assessment. Experiments confirm state-of-the-art results on faces, strong performance on body recognition, and robustness under cross-dataset shifts. Together, these contributions establish TransFIRA as a unified, geometry-driven framework for recognizability assessment -- encoder-specific, accurate, interpretable, and extensible across modalities -- significantly advancing FIQA in accuracy, explainability, and scope.

---

## 46. Scalable deep fusion of spaceborne lidar and synthetic aperture radar for global forest structural complexity mapping

**论文链接:** [http://arxiv.org/abs/2510.06299v1](http://arxiv.org/abs/2510.06299v1)

**作者:** Tiago de Conto, John Armston, Ralph Dubayah

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种融合GEDI激光雷达和多模态SAR数据的深度学习框架，用于生成全球高分辨率的森林结构复杂性地图。

### 背景

森林结构复杂性指标整合多个冠层属性为单一值，反映栖息地质量和生态系统功能。GEDI星载激光雷达虽能绘制温带和热带森林结构复杂性，但其稀疏采样限制了连续高分辨率制图。

### 目的

开发一种可扩展的深度学习框架，融合GEDI观测与多模态SAR数据，生成全球高分辨率的森林结构复杂性地图。

### 方法

使用调整的EfficientNetV2架构，在1.3亿多个GEDI足迹上训练，融合GEDI观测与多模态SAR数据集，生成25米分辨率的全覆盖森林结构复杂性地图。

### 主要发现

模型实现高性能(全局R2=0.82)，参数少于40万个；能在不同生物群落和时间周期内产生准确预测并校准不确定性；已生成2015-2022年的全球多时段森林结构复杂性数据集；可通过迁移学习扩展预测其他森林结构变量。

### 结论

该方法支持全球森林结构动力学的连续监测，为气候变化背景下的生物多样性保护和生态系统管理提供工具，且计算成本较低。

### 翻译

森林结构复杂性指标将多个冠层属性整合为一个单一值，反映栖息地质量和生态系统功能。全球生态系统动力学调查(GEDI)的星载激光雷达已经能够绘制温带和热带森林的结构复杂性图，但其稀疏采样限制了连续高分辨率制图。我们提出了一种可扩展的深度学习框架，融合GEDI观测与多模态合成孔径雷达(SAR)数据集，生成全球高分辨率(25米)的森林结构复杂性全覆盖地图。我们调整的EfficientNetV2架构，在超过1.3亿个GEDI足迹上训练，实现了高性能(全局R2=0.82)，参数少于40万个，使其成为可访问的工具，使研究人员能够在任何规模上处理数据集，而无需专门的计算基础设施。该模型在生物群落和时间周期内产生准确的预测，并校准不确定性估计，保留了精细尺度的空间模式。它已被用于生成2015年至2022年的全球多时段森林结构复杂性数据集。通过迁移学习，该框架可以扩展以预测其他森林结构变量，计算成本最小。这种方法支持全球森林结构动力学的连续、多时段监测，并为气候变化背景下的生物多样性和生态系统管理工作提供工具。


### 论文摘要

Forest structural complexity metrics integrate multiple canopy attributes into a single value that reflects habitat quality and ecosystem function. Spaceborne lidar from the Global Ecosystem Dynamics Investigation (GEDI) has enabled mapping of structural complexity in temperate and tropical forests, but its sparse sampling limits continuous high-resolution mapping. We present a scalable, deep learning framework fusing GEDI observations with multimodal Synthetic Aperture Radar (SAR) datasets to produce global, high-resolution (25 m) wall-to-wall maps of forest structural complexity. Our adapted EfficientNetV2 architecture, trained on over 130 million GEDI footprints, achieves high performance (global R2 = 0.82) with fewer than 400,000 parameters, making it an accessible tool that enables researchers to process datasets at any scale without requiring specialized computing infrastructure. The model produces accurate predictions with calibrated uncertainty estimates across biomes and time periods, preserving fine-scale spatial patterns. It has been used to generate a global, multi-temporal dataset of forest structural complexity from 2015 to 2022. Through transfer learning, this framework can be extended to predict additional forest structural variables with minimal computational cost. This approach supports continuous, multi-temporal monitoring of global forest structural dynamics and provides tools for biodiversity conservation and ecosystem management efforts in a changing climate.

---

## 47. Empirical Comparison of Membership Inference Attacks in Deep Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.05753v2](http://arxiv.org/abs/2510.05753v2)

**作者:** Yuxuan Bai, Gauri Pradhan, Marlon Tobaben, Antti Honkela

**发布时间:** 2025-10-07

**备注:** 30 pages, 13 figures, published in TMLR  https://openreview.net/forum?id=UligTUCgdt

### GPT解析

### 总结

该研究比较了不同成员推理攻击(MIAs)在迁移学习环境中的性能，帮助从业者识别用于隐私风险评估的最有效攻击方法。研究发现没有单一的MIA能捕捉迁移学习模型的所有隐私风险，不同攻击方法在不同情况下效果各异。

### 背景

随着大规模基础模型的出现，机器学习训练范式正从从头训练转向迁移学习，这使得在敏感应用中使用小型领域特定数据集进行高效训练成为可能。成员推理攻击(MIAs)可用于评估机器学习模型的隐私泄露风险。

### 目的

通过比较多样化MIA在迁移学习设置中的性能，帮助从业者识别用于隐私风险评估的最有效攻击方法。

### 方法

比较不同类型的成员推理攻击(MIAs)在迁移学习环境中的表现，研究攻击效力与训练数据量的关系，以及不同攻击方法在不同数据集上的效果差异。

### 主要发现

1. 基于分数的MIA攻击效力随训练数据增加而降低；2. 没有单一MIA能捕捉迁移学习模型的所有隐私风险；3. 似然比攻击(LiRA)在大多数实验场景中表现优异，但逆海森攻击(IHA)在针对使用PatchCamelyon数据集在高数据环境下微调的模型时更为有效。

### 结论

在评估迁移学习模型的隐私风险时，需要考虑多种MIA方法，因为不同的攻击方法在不同情况下可能更有效，没有一种攻击方法能全面捕捉所有隐私风险。

### 翻译

随着强大大规模基础模型的出现，训练范式正日益从从头开始训练转向迁移学习。这使得在敏感应用中使用典型的小型领域特定数据集进行高效训练成为可能。成员推理攻击(MIAs)为机器学习模型提供了隐私泄露的经验估计。然而，先前针对通过迁移学习微调的模型的MIA评估仅依赖于一小部分可能的攻击方法。我们通过比较多样化MIA在迁移学习环境中的性能来解决这个问题，帮助从业者识别用于隐私风险评估的最有效攻击。我们发现，对于基于分数的MIA，攻击效力随训练数据的增加而降低。我们发现没有单一的MIA能捕捉通过迁移学习训练的模型的所有隐私风险。尽管似然比攻击(LiRA)在大多数实验场景中表现优异，但逆海森攻击(IHA)被证明在针对使用PatchCamelyon数据集在高数据环境下微调的模型时更为有效。


### 论文摘要

With the emergence of powerful large-scale foundation models, the training paradigm is increasingly shifting from from-scratch training to transfer learning. This enables high utility training with small, domain-specific datasets typical in sensitive applications. Membership inference attacks (MIAs) provide an empirical estimate of the privacy leakage by machine learning models. Yet, prior assessments of MIAs against models fine-tuned with transfer learning rely on a small subset of possible attacks. We address this by comparing performance of diverse MIAs in transfer learning settings to help practitioners identify the most efficient attacks for privacy risk evaluation. We find that attack efficacy decreases with the increase in training data for score-based MIAs. We find that there is no one MIA which captures all privacy risks in models trained with transfer learning. While the Likelihood Ratio Attack (LiRA) demonstrates superior performance across most experimental scenarios, the Inverse Hessian Attack (IHA) proves to be more effective against models fine-tuned on PatchCamelyon dataset in high data regime.

---

## 48. Temporal Prompting Matters: Rethinking Referring Video Object Segmentation

**论文链接:** [http://arxiv.org/abs/2510.07319v1](http://arxiv.org/abs/2510.07319v1)

**作者:** Ci-Siang Lin, Min-Hung Chen, I-Jieh Liu, Chien-Yi Wang, Sifei Liu, Yu-Chiang Frank Wang

**发布时间:** 2025-10-08

### GPT解析

### 总结

本研究提出了一个名为时间提示生成和选择(Tenet)的框架，用于解决视频对象分割任务，通过分解任务因素并利用现成的检测器和跟踪器生成时间提示，实现了高效模型适应。

### 背景

现有的大多数视频对象分割方法需要密集的掩码注释进行端到端训练，计算量大且可扩展性差。

### 目的

重新思考视频对象分割问题，探究该任务的关键因素，并开发一种更高效的方法。

### 方法

将RVOS任务分解为指代、视频和分割三个因素，提出Tenet框架解决指代和视频因素，利用现成的对象检测器和跟踪器生成与指代语句相关的时间提示，并通过提示偏好学习评估提示质量。

### 主要发现

通过使用生成的时间提示指导基础分割模型，能够为被指代对象生成高质量掩码，实现模型向视频对象分割的高效适应。

### 结论

Tenet框架在RVOS基准测试上表现出有效性，提供了一种更高效的视频对象分割方法。

### 翻译

引用视频对象分割旨在分割视频中由查询语句指代的对象。大多数现有方法需要密集的掩码注释进行端到端训练，这可能导致计算量大且可扩展性差。在本工作中，我们重新思考了RVOS问题，旨在探究该任务的关键。基于现有的基础分割模型，我们将RVOS任务分解为指代、视频和分割因素，并提出时间提示生成和选择(Tenet)框架来解决指代和视频因素，同时将分割问题留给基础模型。为了高效地将基于图像的基础分割模型适应到视频对象分割中，我们利用现成的对象检测器和跟踪器生成与指代语句相关的时间提示。虽然可以生成高质量的时间提示，但无法从置信度分数中轻易识别。为解决此问题，我们提出了提示偏好学习来评估生成的时间提示的质量。通过使用这些提示指导基于图像的基础分割模型，我们能够为被指代对象生成高质量掩码，实现模型向视频对象分割的高效适应。在RVOS基准测试上的实验证明了Tenet框架的有效性。


### 论文摘要

Referring Video Object Segmentation (RVOS) aims to segment the object referred to by the query sentence in the video. Most existing methods require end-to-end training with dense mask annotations, which could be computation-consuming and less scalable. In this work, we rethink the RVOS problem and aim to investigate the key to this task. Based on existing foundation segmentation models, we decompose the RVOS task into referring, video, and segmentation factors, and propose a Temporal Prompt Generation and Selection (Tenet) framework to address the referring and video factors while leaving the segmentation problem to foundation models. To efficiently adapt image-based foundation segmentation models to referring video object segmentation, we leverage off-the-shelf object detectors and trackers to produce temporal prompts associated with the referring sentence. While high-quality temporal prompts could be produced, they can not be easily identified from confidence scores. To tackle this issue, we propose Prompt Preference Learning to evaluate the quality of the produced temporal prompts. By taking such prompts to instruct image-based foundation segmentation models, we would be able to produce high-quality masks for the referred object, enabling efficient model adaptation to referring video object segmentation. Experiments on RVOS benchmarks demonstrate the effectiveness of the Tenet framework.

---

## 49. MoRe: Monocular Geometry Refinement via Graph Optimization for Cross-View Consistency

**论文链接:** [http://arxiv.org/abs/2510.07119v1](http://arxiv.org/abs/2510.07119v1)

**作者:** Dongki Jung, Jaehoon Choi, Yonghan Lee, Sungmin Eum, Heesung Kwon, Dinesh Manocha

**发布时间:** 2025-10-08

### GPT解析

### 总结

本文提出了MoRe，一种无需训练的单目几何细化方法，用于提高跨视图一致性和实现尺度对齐，通过图优化框架解决单目几何先验的尺度模糊性问题。

### 背景

单目3D基础模型为感知任务提供了可扩展的解决方案，使其在更广泛的3D视觉应用中具有吸引力。

### 目的

提高单目3D重建的跨视图一致性，实现尺度对齐，并改善3D重建和新视图合成的质量，特别是在稀疏视图渲染场景中。

### 方法

提出MoRe方法，通过帧间特征匹配建立对应关系，采用基于图的优化框架，利用单目基础模型估计的3D点和表面法线进行局部平面近似，而非简单的最小二乘优化。

### 主要发现

MoRe不仅提高了3D重建的质量，还改善了新视图合成，特别是在稀疏视图渲染场景中表现更为突出。

### 结论

MoRe方法有效解决了单目几何先验中的尺度模糊性问题，同时保留了底层3D结构，为单目3D视觉应用提供了改进的解决方案。

### 翻译

单目3D基础模型为感知任务提供了可扩展的解决方案，使其在更广泛的3D视觉应用中具有吸引力。在本文中，我们提出了MoRe，一种无需训练的单目几何细化方法，旨在提高跨视图一致性和实现尺度对齐。为了诱导帧间关系，我们的方法采用帧间特征匹配来建立对应关系。我们不是在这些匹配点上应用简单的最小二乘优化，而是制定了一个基于图的优化框架，使用单目基础模型估计的3D点和表面法线进行局部平面近似。这种方法解决了单目几何先验中固有的尺度模糊性问题，同时保留了底层3D结构。我们进一步证明，MoRe不仅提高了3D重建，还改善了新视图合成，特别是在稀疏视图渲染场景中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目3D基础模型在不同视角之间的一致性问题（cross-view consistency）。由于单目相机固有的尺度模糊性，不同视角预测的点云往往无法对齐，导致3D重建和新视图合成质量不佳。这个问题在机器人、AR/VR、自动驾驶等领域非常重要，因为这些领域广泛应用单目相机，而多视角一致性是3D场景理解和高质量重建的关键挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有单目3D基础模型的局限性，特别是多视角之间的尺度不一致问题。他们发现端到端方法缺乏模块化，难以集成外部传感器，而简单最小二乘优化容易引入噪声。因此，他们设计了两阶段方法：首先进行初始仿射变换对齐，然后引入基于图的优化使用局部平面近似来细化对齐。该方法借鉴了MoGe的并行化对齐求解器、MadPose的尺度参数化方法，以及Rossi等人的局部平面优化技术，并利用DKM等图像匹配算法建立稠密对应关系。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过基于图的优化提升单目3D点云的跨视角一致性，利用局部平面近似和表面法线信息来优化点云对齐，而不是简单地最小化对应点之间的欧氏距离。整体流程包括：1) 使用单目基础模型预测点云和法线；2) 通过图像匹配建立像素对应关系；3) 应用仿射变换进行初始对齐；4) 构建图结构并定义多种几何约束（帧内和跨帧平面约束、kNN约束、视线一致性约束等）；5) 使用多尺度策略优化点云位置；6) 输出优化后的跨视角一致点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 模块化框架，将点云对齐与完整3D场景重建解耦，允许灵活集成传统几何方法和外部传感器；2) 基于图的优化方法，使用局部平面近似和表面法线进行联合优化，减少噪声影响；3) 多种跨视角几何约束，确保优化后点云在多视角间保持一致性；4) 直接优化点云坐标而非深度图，允许在三个方向上调整。相比之前的工作，MoRe解决了端到端方法缺乏模块化的问题，提高了简单最小二乘优化的精度，并超越了单目方法如MoGe的尺度模糊性限制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MoRe通过创新的基于图优化和局部平面近似的跨视角对齐方法，有效解决了单目3D基础模型的尺度模糊性问题，实现了模块化、高精度的多视角一致3D重建，并在稀疏视图场景下显著提升了新视图合成的质量。'}


### 论文摘要

Monocular 3D foundation models offer an extensible solution for perception tasks, making them attractive for broader 3D vision applications. In this paper, we propose MoRe, a training-free Monocular Geometry Refinement method designed to improve cross-view consistency and achieve scale alignment. To induce inter-frame relationships, our method employs feature matching between frames to establish correspondences. Rather than applying simple least squares optimization on these matched points, we formulate a graph-based optimization framework that performs local planar approximation using the estimated 3D points and surface normals estimated by monocular foundation models. This formulation addresses the scale ambiguity inherent in monocular geometric priors while preserving the underlying 3D structure. We further demonstrate that MoRe not only enhances 3D reconstruction but also improves novel view synthesis, particularly in sparse view rendering scenarios.

---

## 50. Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report

**论文链接:** [http://arxiv.org/abs/2510.07092v1](http://arxiv.org/abs/2510.07092v1)

**作者:** Riccardo Mereu, Aidan Scannell, Yuxin Hou, Yi Zhao, Aditya Jitta, Antonio Dominguez, Luigi Acerbi, Amos Storkey, Paul Chang

**发布时间:** 2025-10-08

**备注:** 6 pages, 3 figures, 1X world model challenge technical report

### GPT解析

### 总结

本研究介绍了1X World Model Challenge，一个专注于人形机器人交互的开源基准，包含采样和压缩两个赛道。研究团队在采样赛道上适配了视频生成模型Wan-2.2 TI2V-5B，在压缩赛道上训练了时空Transformer模型，并在两个任务中均获得第一名。

### 背景

World models是AI和机器人学中的强大范式，使智能体能够通过预测视觉观察或紧凑的潜在状态来推理未来。然而，缺乏针对真实世界人形机器人交互的开源基准。

### 目的

创建并参与1X World Model Challenge，提供一个人形机器人交互的开源基准测试，包含两个互补赛道：采样（预测未来图像帧）和压缩（预测未来离散潜在代码）。

### 方法

采样赛道：将视频生成基础模型Wan-2.2 TI2V-5B适应到视频状态条件下的未来帧预测，使用AdaLN-Zero条件化机器人状态，并通过LoRA进行微调。压缩赛道：从头开始训练Spatio-Temporal Transformer模型。

### 主要发现

采样任务达到23.0 dB PSNR，压缩任务达到Top-500 CE为6.6386，在两个挑战中都获得第一名。

### 结论

所提出的模型在两个互补的任务上都表现出色，证明了World models在真实世界人形机器人交互中的有效性。

### 翻译

世界模型是AI和机器人学中的强大范式，使智能体能够通过预测视觉观察或紧凑的潜在状态来推理未来。1X World Model Challenge引入了一个真实世界人形机器人交互的开源基准，包含两个互补赛道：采样，专注于预测未来图像帧；压缩，专注于预测未来离散潜在代码。对于采样赛道，我们将视频生成基础模型Wan-2.2 TI2V-5B适应到视频状态条件下的未来帧预测。我们使用AdaLN-Zero将视频生成条件化为机器人状态，并使用LoRA进一步微调模型。对于压缩赛道，我们从零开始训练了一个时空Transformer模型。我们的模型在采样任务中达到23.0 dB PSNR，在压缩任务中达到Top-500 CE为6.6386，在两个挑战中都获得了第一名。


### 论文摘要

World models are a powerful paradigm in AI and robotics, enabling agents to reason about the future by predicting visual observations or compact latent states. The 1X World Model Challenge introduces an open-source benchmark of real-world humanoid interaction, with two complementary tracks: sampling, focused on forecasting future image frames, and compression, focused on predicting future discrete latent codes. For the sampling track, we adapt the video generation foundation model Wan-2.2 TI2V-5B to video-state-conditioned future frame prediction. We condition the video generation on robot states using AdaLN-Zero, and further post-train the model using LoRA. For the compression track, we train a Spatio-Temporal Transformer model from scratch. Our models achieve 23.0 dB PSNR in the sampling task and a Top-500 CE of 6.6386 in the compression task, securing 1st place in both challenges.

---

## 51. Revisiting Mixout: An Overlooked Path to Robust Finetuning

**论文链接:** [http://arxiv.org/abs/2510.06982v1](http://arxiv.org/abs/2510.06982v1)

**作者:** Masih Aminbeidokhti, Heitor Rapela Medeiros, Eric Granger, Marco Pedersoli

**发布时间:** 2025-10-08

### GPT解析

### 总结

研究微调视觉基础模型时提高准确性与鲁棒性的平衡方法，提出GMixout技术。

### 背景

微调视觉基础模型通常可以提高领域内准确性，但会降低在分布偏移情况下的鲁棒性。

### 目的

重新审视Mixout随机正则化方法，通过单次运行、权重共享的隐式集合视角分析，提出改进方法以提高模型在分布偏移下的鲁棒性。

### 方法

通过单次运行、权重共享的隐式集合视角重新审视Mixout，发现控制鲁棒性的三个关键杠杆：掩码锚点、重采样频率和掩码稀疏性，并基于此提出GMixout方法，用指数移动平均快照替换固定锚点，并通过显式的重采样频率超参数调节掩码周期。采用稀疏核实现，仅更新一小部分参数，无需推理时间开销。

### 主要发现

控制鲁棒性的三个关键杠杆是掩码锚点、重采样频率和掩码稀疏性；GMixout在提高领域内准确性的同时，能增强模型在分布偏移下的鲁棒性。

### 结论

GMixout在多个基准测试中表现优异，在提高领域内准确性的同时，超越了Model Soups和强参数高效微调基线在分布偏移下的表现，且可在消费级GPU上训练。

### 翻译

微调视觉基础模型通常可以提高领域内准确性，但会以在分布偏移下降低鲁棒性为代价。我们通过单次运行、权重共享的隐式集合的视角重新审视了Mixout，这是一种随机正则化方法，间歇性地用其预训练参考替换微调权重。这种视角揭示了控制鲁棒性的三个关键杠杆：掩码锚点、重采样频率和掩码稀疏性。在分析指导下，我们引入了GMixout，它(i)用适应训练过程中变化的指数移动平均快照替换固定锚点，以及(ii)通过显式的重采样频率超参数调节掩码周期。我们的稀疏核实现仅更新一小部分参数，无需推理时间开销， enabling在消费级GPU上训练。在涵盖协变量偏移、损坏和类别不平衡的基准测试中，包括ImageNet / ImageNet-LT、DomainNet、iWildCam和CIFAR100-C，GMixout在提高领域内准确性方面始终超越零样本性能，同时在分布偏移下超越了Model Soups和强参数高效微调基线。


### 论文摘要

Finetuning vision foundation models often improves in-domain accuracy but comes at the cost of robustness under distribution shift. We revisit Mixout, a stochastic regularizer that intermittently replaces finetuned weights with their pretrained reference, through the lens of a single-run, weight-sharing implicit ensemble. This perspective reveals three key levers that govern robustness: the \emph{masking anchor}, \emph{resampling frequency}, and \emph{mask sparsity}. Guided by this analysis, we introduce GMixout, which (i) replaces the fixed anchor with an exponential moving-average snapshot that adapts during training, and (ii) regulates masking period via an explicit resampling-frequency hyperparameter. Our sparse-kernel implementation updates only a small fraction of parameters with no inference-time overhead, enabling training on consumer-grade GPUs. Experiments on benchmarks covering covariate shift, corruption, and class imbalance, ImageNet / ImageNet-LT, DomainNet, iWildCam, and CIFAR100-C, GMixout consistently improves in-domain accuracy beyond zero-shot performance while surpassing both Model Soups and strong parameter-efficient finetuning baselines under distribution shift.

---

## 52. VA-Adapter: Adapting Ultrasound Foundation Model to Echocardiography Probe Guidance

**论文链接:** [http://arxiv.org/abs/2510.06809v1](http://arxiv.org/abs/2510.06809v1)

**作者:** Teng Wang, Haojun Jiang, Yuxuan Wang, Zhenguo Sun, Shiji Song, Gao Huang

**发布时间:** 2025-10-08

### GPT解析

### 总结

该研究提出了一种名为视觉-动作适配器(VA-Adapter)的参数高效方法，将超声基础模型知识应用于探头引导任务，帮助初级超声医师获取高质量心脏超声图像。

### 背景

超声心动图是检测心脏疾病的关键工具，但心脏超声操作难度极高，导致高技能人员短缺，患者无法及时接受检查服务。获取高质量超声图像是准确诊断的前提。

### 目的

将基础模型从大量数据集中学习到的医学知识适应到探头引导任务中，为初级超声医师提供实时操作建议，以获取高质量的超声图像。

### 方法

设计了一个参数高效的视觉-动作适配器(VA-Adapter)，使基础模型的图像编码器能够编码视觉-动作序列，提高引导性能。该适配器具有内置的顺序推理能力和紧凑设计，仅微调一小部分参数即可使预训练的超声基础模型学习精确的探头调整策略。

### 主要发现

大量实验表明，VA-Adapter能够超越现有的强探头引导模型。

### 结论

VA-Adapter有效解决了心脏超声操作难度高、专业人员短缺的问题，通过提供实时操作建议，帮助初级超声医师获取高质量超声图像。

### 翻译

超声心动图是检测心脏疾病的关键工具。最近，超声基础模型在心脏超声图像分析方面表现出显著能力。然而，获取高质量超声图像是准确诊断的前提。由于心脏超声操作难度极高，缺乏高技能人员，导致患者无法及时接受检查服务。在本文中，我们旨在将基础模型从大量数据集中学习到的医学知识适应到探头引导任务中，该任务旨在为初级超声医师提供实时操作建议，以获取高质量的超声图像。此外，受专家根据过去探索优化行动决策的实践启发，我们精心设计了一个参数高效的视觉-动作适配器(VA-Adapter)，使基础模型的图像编码器能够编码视觉-动作序列，从而提高引导性能。凭借紧凑设计中的内置顺序推理能力，VA-Adapter使预训练的超声基础模型仅通过微调一小部分参数即可学习精确的探头调整策略。大量实验证明，VA-Adapter能够超越强大的探头引导模型。我们的代码将在接受后发布。


### 论文摘要

Echocardiography is a critical tool for detecting heart diseases. Recently, ultrasound foundation models have demonstrated remarkable capabilities in cardiac ultrasound image analysis. However, obtaining high-quality ultrasound images is a prerequisite for accurate diagnosis. Due to the exceptionally high operational difficulty of cardiac ultrasound, there is a shortage of highly skilled personnel, which hinders patients from receiving timely examination services. In this paper, we aim to adapt the medical knowledge learned by foundation models from vast datasets to the probe guidance task, which is designed to provide real-time operational recommendations for junior sonographers to acquire high-quality ultrasound images. Moreover, inspired by the practice where experts optimize action decisions based on past explorations, we meticulously design a parameter-efficient Vision-Action Adapter (VA-Adapter) to enable foundation model's image encoder to encode vision-action sequences, thereby enhancing guidance performance. With built-in sequential reasoning capabilities in a compact design, the VA-Adapter enables a pre-trained ultrasound foundation model to learn precise probe adjustment strategies by fine-tuning only a small subset of parameters. Extensive experiments demonstrate that the VA-Adapter can surpass strong probe guidance models. Our code will be released after acceptance.

---

## 53. UniFField: A Generalizable Unified Neural Feature Field for Visual, Semantic, and Spatial Uncertainties in Any Scene

**论文链接:** [http://arxiv.org/abs/2510.06754v1](http://arxiv.org/abs/2510.06754v1)

**作者:** Christian Maurer, Snehal Jauhri, Sophie Lueth, Georgia Chalvatzaki

**发布时间:** 2025-10-08

**备注:** Project website: https://sites.google.com/view/uniffield

### GPT解析

### 总结

UniFField是一种统一的不确定性感知神经特征场，结合视觉、语义和几何特征，能够预测每种模态的不确定性，适用于机器人任务执行和决策。

### 背景

对3D场景进行全面的可视化、几何和语义理解对机器人任务执行至关重要，特别是在非结构化和复杂环境中。机器人需要评估感知信息的可靠性以做出稳健决策。现有的3D神经特征场方法有两个关键局限：通常是场景特定的，且缺乏对预测不确定性的建模能力。

### 目的

开发一种统一的不确定性感知神经特征场，结合多种特征模态并预测不确定性，使机器人能够零样本应用于新环境，并做出稳健决策。

### 方法

提出UniFField，一种统一的神经特征场，结合视觉、语义和几何特征。使用基于体素的特征表示，随着机器人探索场景逐步整合RGB-D图像，同时更新不确定性估计。

### 主要发现

不确定性估计能够准确描述模型在场景重建和语义特征预测中的预测误差。利用特征预测及其相应的不确定性成功完成了主动物体搜索任务，展示了稳健决策的能力。

### 结论

UniFField提供了一种有效的方法，使机器人能够在复杂环境中进行3D场景理解，并通过不确定性预测做出更稳健的决策。

### 翻译

对3D场景进行全面的可视化、几何和语义理解对于机器人任务的成功执行至关重要，特别是在非结构化和复杂环境中。此外，为了做出稳健决策，机器人有必要评估感知信息的可靠性。尽管最近3D神经特征场的进展使机器人能够利用预训练基础模型的特征来完成语言引导的操控和导航等任务，但现有方法存在两个关键局限：(i)它们通常是场景特定的，(ii)它们缺乏对其预测不确定性的建模能力。我们提出了UniFField，一种统一的不确定性感知神经特征场，在单一可泛化表示中结合视觉、语义和几何特征，同时预测每种模态的不确定性。我们的方法可以零样本应用于任何新环境，随着机器人探索场景逐步将RGB-D图像整合到我们的基于体素的特征表示中，同时更新不确定性估计。我们评估了不确定性估计，以准确描述模型在场景重建和语义特征预测中的预测误差。此外，我们成功利用特征预测及其相应的不确定性，使用移动操作器机器人完成了主动物体搜索任务，展示了稳健决策的能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决两个关键问题：1) 现有的3D神经特征场通常是场景特定的，无法很好地泛化到新场景；2) 现有方法缺乏对其预测的不确定性建模的能力。这两个问题在机器人技术中至关重要，因为机器人需要在未结构化和复杂的环境中执行任务，并需要评估感知信息的可靠性以做出稳健决策，特别是在部分可观察的环境中，理解预测的不确定性对机器人的主动感知和探索至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了多个现有工作的思想：参考了NeRF和Gaussian Splatting等3D神经表示方法，但解决了它们无法增量添加观察的问题；学习了通用特征场的工作，这些工作可以在多个场景上预训练并零样本应用到新场景；参考了增量神经表示的工作，随时间聚合信息；借鉴了贝叶斯方法如Dropout和集成方法用于不确定性建模；采用了类似GeFF的方法学习通用语义先验，并利用基于体素的特征表示。通过整合这些现有方法的思想并添加不确定性建模能力，设计了UniFField。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的神经特征场，能够结合视觉、语义和几何特征在一个表示中，预测每种模态的不确定性，支持零样本应用到任何新环境，并允许增量更新。实现流程：1) 构建统一特征场：从RGB-D图像提取特征并反投影到3D空间，创建图像特征体积和TSDF体积，添加不确定性指标；2) 解码统一特征场：构建三个解码网络映射到RGB值、语义特征、TSDF值及不确定性；3) 使用可微体积渲染将3D预测投影到2D进行训练；4) 使用不确定性感知的监督训练模型；5) 推理时直接构建新场景特征场，通过新RGB-D帧的统一特征体积的运行平均进行增量更新。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 通用统一神经特征场，结合多种模态在一个表示中；2) 能够预测每种模态的不确定性，使在部分可观察设置中进行稳健决策；3) 支持增量更新，适合连续探索场景的机器人；4) 设计了使用不确定性感知UniFField进行主动对象搜索任务的方法。相比之前工作，UniFField的不同在于：不仅处理视觉和几何信息，还整合语义信息；明确建模预测的不确定性；支持零样本应用和增量更新；能够处理不同类型的不确定性（认知和偶然不确定性）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniFField是一种通用统一的神经特征场，能够结合视觉、语义和几何特征并预测每种模态的不确定性，使机器人能够在任何场景中进行稳健的决策和主动探索，而无需针对每个场景进行优化。'}


### 论文摘要

Comprehensive visual, geometric, and semantic understanding of a 3D scene is crucial for successful execution of robotic tasks, especially in unstructured and complex environments. Additionally, to make robust decisions, it is necessary for the robot to evaluate the reliability of perceived information. While recent advances in 3D neural feature fields have enabled robots to leverage features from pretrained foundation models for tasks such as language-guided manipulation and navigation, existing methods suffer from two critical limitations: (i) they are typically scene-specific, and (ii) they lack the ability to model uncertainty in their predictions. We present UniFField, a unified uncertainty-aware neural feature field that combines visual, semantic, and geometric features in a single generalizable representation while also predicting uncertainty in each modality. Our approach, which can be applied zero shot to any new environment, incrementally integrates RGB-D images into our voxel-based feature representation as the robot explores the scene, simultaneously updating uncertainty estimation. We evaluate our uncertainty estimations to accurately describe the model prediction errors in scene reconstruction and semantic feature prediction. Furthermore, we successfully leverage our feature predictions and their respective uncertainty for an active object search task using a mobile manipulator robot, demonstrating the capability for robust decision-making.

---

## 54. LLM Company Policies and Policy Implications in Software Organizations

**论文链接:** [http://arxiv.org/abs/2510.06718v1](http://arxiv.org/abs/2510.06718v1)

**作者:** Ranim Khojah, Mazen Mohamad, Linda Erlenhov, Francisco Gomes de Oliveira Neto, Philipp Leitner

**发布时间:** 2025-10-08

**备注:** Accepted at IEEE Software Special Issue on AIware in the Foundation  Models Era

### GPT解析

### 总结

研究探讨了软件组织中采用大型语言模型(LLM)聊天机器人相关的风险及制定明确政策的必要性，调查了11家公司的政策制定过程及影响因素

### 背景

软件组织在采用大型语言模型(LLM)聊天机器人时面临相关风险

### 目的

帮助管理者将聊天机器人安全地集成到开发工作流程中

### 方法

调查11家公司如何创建聊天机器人相关政策及影响这些政策的因素

### 主要发现

软件组织采用LLM聊天机器人存在风险，需要制定明确政策；11家公司的政策创建方式及影响因素

### 结论

需要明确的政策来管理LLM聊天机器人的采用，以确保安全集成到开发工作流程中

### 翻译

软件组织中采用大型语言模型(LLM)聊天机器人相关的风险凸显了明确政策的必要性。我们研究了11家公司如何制定这些政策以及影响这些政策的因素，旨在帮助管理者将聊天机器人安全地集成到开发工作流程中。


### 论文摘要

The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows.

---

## 55. RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training

**论文链接:** [http://arxiv.org/abs/2510.06710v1](http://arxiv.org/abs/2510.06710v1)

**作者:** Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, Zhihao Liu, Kang Chen, Wenhao Tang, Quanlu Zhang, Weinan Zhang, Chao Yu, Yu Wang

**发布时间:** 2025-10-08

**备注:** This is the technical report of the RLinf Team, focusing on the  algorithm side. For the system-level design, please refer to  arXiv:2509.15965. The open-sourced code link: https://github.com/RLinf/RLinf

### GPT解析

### 总结

本文介绍了RLinf-VLA，一个统一且高效的框架，用于可扩展的VLA模型强化学习训练。该框架通过灵活的资源分配设计解决了在强化学习和视觉-语言-动作模型训练中整合渲染、训练和推理的挑战，实现了训练速度的显著提升。它支持多种VLA架构、强化学习算法和模拟器，在模拟环境中表现出色，并在真实机器人部署中显示出比监督微调更好的泛化能力。

### 背景

视觉和语言基础模型的进步显著推动了多模态理解、推理和生成能力的发展，激发了将此类能力扩展到具身环境中的兴趣，通过视觉-语言-动作模型实现。然而，大多数VLA模型仍使用监督微调进行训练，这种方法在分布变化下泛化能力有限，因为存在错误累积问题。强化学习提供了一个有前景的替代方案，但现有尝试仍然零散，缺乏统一的比较平台。

### 目的

为了解决VLA模型训练中缺乏统一框架的问题，作者引入了RLinf-VLA，这是一个统一且高效的框架，用于VLA模型的可扩展强化学习训练。该系统旨在提供一个灵活的资源分配设计，解决在强化学习和视觉-语言-动作模型训练中整合渲染、训练和推理的挑战。

### 方法

RLinf-VLA采用了一种高度灵活的资源分配设计，特别是对于GPU并行化的模拟器，实现了一种新颖的混合细粒度流水线分配模式，实现了1.61倍至1.88倍的训练速度提升。通过统一的接口，RLinf-VLA无缝支持多种VLA架构、多种强化学习算法以及各种模拟器。

### 主要发现

在模拟环境中，统一模型在130个LIBERO任务上达到了98.11%的成功率，在25个ManiSkill任务上达到了97.66%的成功率。研究还总结了一套将强化学习应用于VLA训练的最佳实践。在真实世界的Franka机器人上，强化学习训练的策略比使用监督微调训练的策略表现出更强的泛化能力。

### 结论

作者将RLinf-VLA视为加速和标准化具身智能研究的基础。该框架提供了一个统一的平台，使研究人员能够公平和系统地比较不同模型架构和算法设计，从而推动该领域的发展。

### 翻译

视觉和语言基础模型的最新进展显著提升了多模态理解、推理和生成能力，激发了通过视觉-语言-动作模型将此类能力扩展到具身环境中的兴趣。然而，大多数VLA模型仍然使用监督微调进行训练，由于错误累积问题，这种方法在分布变化下难以泛化。强化学习通过交互直接优化任务性能，提供了一个有前景的替代方案。为了解决这一差距，我们引入了RLinf-VLA，一个用于VLA模型可扩展强化学习训练的统一高效框架。该系统采用了一种高度灵活的资源分配设计，解决了在强化学习和视觉-语言-动作模型训练中整合渲染、训练和推理的挑战。特别是，对于GPU并行化的模拟器，RLinf-VLA实现了一种新颖的混合细粒度流水线分配模式，实现了1.61倍至1.88倍的训练速度提升。通过统一的接口，RLinf-VLA无缝支持多种VLA架构、多种强化学习算法以及各种模拟器。在模拟环境中，统一模型在130个LIBERO任务上达到了98.11%的成功率，在25个ManiSkill任务上达到了97.66%的成功率。除了实证性能外，我们的研究还总结了一套将强化学习应用于VLA训练的最佳实践。此外，我们在真实世界的Franka机器人上进行了初步部署，发现强化学习训练的策略比使用监督微调训练的策略表现出更强的泛化能力。我们将RLinf-VLA视为加速和标准化具身智能研究的基础。


### 论文摘要

Recent progress in vision and language foundation models has significantly advanced multimodal understanding, reasoning, and generation, inspiring a surge of interest in extending such capabilities to embodied settings through vision-language-action (VLA) models. Yet, most VLA models are still trained with supervised fine-tuning (SFT), which struggles to generalize under distribution shifts due to error accumulation. Reinforcement learning (RL) offers a promising alternative by directly optimizing task performance through interaction, but existing attempts remain fragmented and lack a unified platform for fair and systematic comparison across model architectures and algorithmic designs. To address this gap, we introduce RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models. The system adopts a highly flexible resource allocation design that addresses the challenge of integrating rendering, training, and inference in RL+VLA training. In particular, for GPU-parallelized simulators, RLinf-VLA implements a novel hybrid fine-grained pipeline allocation mode, achieving a 1.61x-1.88x speedup in training. Through a unified interface, RLinf-VLA seamlessly supports diverse VLA architectures (e.g., OpenVLA, OpenVLA-OFT), multiple RL algorithms (e.g., PPO, GRPO), and various simulators (e.g., ManiSkill, LIBERO). In simulation, a unified model achieves 98.11\% across 130 LIBERO tasks and 97.66\% across 25 ManiSkill tasks. Beyond empirical performance, our study distills a set of best practices for applying RL to VLA training and sheds light on emerging patterns in this integration. Furthermore, we present preliminary deployment on a real-world Franka robot, where RL-trained policies exhibit stronger generalization than those trained with SFT. We envision RLinf-VLA as a foundation to accelerate and standardize research on embodied intelligence.

---

## 56. POME: Post Optimization Model Edit via Muon-style Projection

**论文链接:** [http://arxiv.org/abs/2510.06627v1](http://arxiv.org/abs/2510.06627v1)

**作者:** Yong Liu, Di Fu, Yang Luo, Zirui Zhu, Minhao Cheng, Cho-Jui Hsieh, Yang You

**发布时间:** 2025-10-08

### GPT解析

### 总结

论文介绍了一种名为POME（后优化模型编辑）的新算法，它仅使用预训练和微调检查点即可提升大型语言模型的性能，无需额外数据或进一步优化。

### 背景

大型语言模型在微调后可能仍有性能提升空间，特别是在不引入额外计算资源的情况下。

### 目的

开发一种简单的方法来提高已微调的大型语言模型的性能，同时保持与现有训练框架的兼容性。

### 方法

对微调权重与预训练权重之间的差异ΔW应用μ子样投影，使用截断奇异值分解（SVD）平衡主导更新方向的影响，并修剪代表噪声的小奇异值。作为一种简单的后处理步骤，POME与训练流程完全解耦。

### 主要发现

POME在GSM8K上平均性能提升+2.5%，在代码生成上提升+1.0%。该方法适用于从7B基础模型到72B RLHF指令模型的广泛范围。

### 结论

POME是一种实用、零成本的微调流程增强方法，对任何优化器或分布式框架都是通用兼容的，代码已公开在GitHub上。

### 翻译

我们引入了后优化模型编辑（POME），一种新算法，它仅使用预训练和微调检查点即可增强大型语言模型的性能，无需额外数据或进一步优化。核心思想是对ΔW（微调权重与预训练权重之间的差异）应用μ子样投影。这种投影使用截断奇异值分解（SVD）来平衡主导更新方向的影响，并修剪小的奇异值，这些值通常代表噪声。作为一种简单的后处理步骤，POME与训练流程完全解耦。它需要零修改且不产生任何开销，使其与任何优化器或分布式框架普遍兼容。POME提供了一致的性能提升，在GSM8K上平均性能提高+2.5%，在代码生成上提高+1.0%。它的广泛适用性——从7B基础模型到72B RLHF指令模型——使其成为任何微调流程的实用、零成本增强。代码可在https://github.com/NUS-HPC-AI-Lab/POME获取。


### 论文摘要

We introduce Post-Optimization Model Edit (POME), a new algorithm that enhances the performance of fine-tuned large language models using only their pretrained and fine-tuned checkpoints, without requiring extra data or further optimization. The core idea is to apply a muon-style projection to $\Delta W$, the difference between the fine-tuned and pretrained weights. This projection uses truncated singular value decomposition (SVD) to equalize the influence of dominant update directions and prune small singular values, which often represent noise. As a simple post-processing step, POME is completely decoupled from the training pipeline. It requires zero modifications and imposes no overhead, making it universally compatible with any optimizer or distributed framework. POME delivers consistent gains, boosting average performance by +2.5\% on GSM8K and +1.0\% on code generation. Its broad applicability -- from 7B foundation models to 72B RLHF-instructed models -- establishes it as a practical, zero-cost enhancement for any fine-tuning pipeline. Code is available at https://github.com/NUS-HPC-AI-Lab/POME.

---

## 57. Cluster Paths: Navigating Interpretability in Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.06541v1](http://arxiv.org/abs/2510.06541v1)

**作者:** Nicholas M. Kroeger, Vincent Bindschaedler

**发布时间:** 2025-10-08

### GPT解析

### 总结

论文提出了cluster paths方法，通过聚类神经网络激活并使用序列表示输入来提高深度学习模型的可解释性。该方法通过多个指标评估，并在各种任务上展示了有效性和可扩展性，能够揭示模型决策背后的视觉概念。

### 背景

现代深度神经网络在视觉任务中取得了令人印象深刻的性能，但它们的决策过程仍然不透明，存在不必要的信任风险、未检测到的偏差和意外故障。

### 目的

开发一种后验可解释性方法，使深度神经网络的决策过程更加透明，帮助理解模型如何做出决策以及识别潜在问题。

### 方法

在选定层对激活进行聚类，并将每个输入表示为其聚类ID序列。引入四个评估指标：路径复杂性（认知负荷）、加权路径纯度（类别对齐）、决策对齐保真度（预测保真度）和路径一致性（对扰动的稳定性）。将cluster paths扩展到concept paths，通过在大型语言模型上提示最小路径差异来推导。

### 主要发现

在虚假线索的CIFAR-10实验中，cluster paths识别基于颜色的捷径；在五类CelebA发色任务中，实现了90%的保真度，并在高斯噪声下保持96%的一致性；能够揭示多网络深度的视觉概念，如调色板、纹理或对象上下文；可以作为有效的分布外(OOD)检测器。

### 结论

cluster paths可以扩展到大型视觉模型，同时生成简洁且人类可读的解释，帮助理解模型决策过程并识别潜在问题。

### 翻译

虽然现代深度神经网络在视觉任务中取得了令人印象深刻的性能，但它们的决策过程仍然不透明，存在不必要的信任风险、未检测到的偏差和意外故障。我们提出了cluster paths，一种后验可解释性方法，它在选定层对激活进行聚类，并将每个输入表示为其聚类ID序列。为了评估这些cluster paths，我们引入了四个指标：路径复杂性（认知负荷）、加权路径纯度（类别对齐）、决策对齐保真度（预测保真度）和路径一致性（对扰动的稳定性）。在虚假线索的CIFAR-10实验中，cluster paths识别基于颜色的捷径，并在线索被移除时失效。在五类CelebA发色任务中，它们实现了90%的保真度，并在高斯噪声下保持96%的一致性，同时不牺牲准确性。扩展到ImageNet上预训练的Vision Transformer时，我们将cluster paths扩展为从大型语言模型提示中派生的concept paths。最后，我们表明cluster paths可以作为一种有效的分布外(OOD)检测器，在模型产生过度自信的预测之前可靠地标记异常样本。cluster paths在多个网络深度上揭示了视觉概念，如调色板、纹理或对象上下文，证明了cluster paths可以扩展到大型视觉模型，同时生成简洁且人类可读的解释。


### 论文摘要

While modern deep neural networks achieve impressive performance in vision tasks, they remain opaque in their decision processes, risking unwarranted trust, undetected biases and unexpected failures. We propose cluster paths, a post-hoc interpretability method that clusters activations at selected layers and represents each input as its sequence of cluster IDs. To assess these cluster paths, we introduce four metrics: path complexity (cognitive load), weighted-path purity (class alignment), decision-alignment faithfulness (predictive fidelity), and path agreement (stability under perturbations). In a spurious-cue CIFAR-10 experiment, cluster paths identify color-based shortcuts and collapse when the cue is removed. On a five-class CelebA hair-color task, they achieve 90% faithfulness and maintain 96% agreement under Gaussian noise without sacrificing accuracy. Scaling to a Vision Transformer pretrained on ImageNet, we extend cluster paths to concept paths derived from prompting a large language model on minimal path divergences. Finally, we show that cluster paths can serve as an effective out-of-distribution (OOD) detector, reliably flagging anomalous samples before the model generates over-confident predictions. Cluster paths uncover visual concepts, such as color palettes, textures, or object contexts, at multiple network depths, demonstrating that cluster paths scale to large vision models while generating concise and human-readable explanations.

---

## 58. PuzzlePlex: Benchmarking Foundation Models on Reasoning and Planning with Puzzles

**论文链接:** [http://arxiv.org/abs/2510.06475v1](http://arxiv.org/abs/2510.06475v1)

**作者:** Yitao Long, Yuru Jiang, Hongjun Liu, Yilun Zhao, Jingchen Sun, Yiqiu Shen, Chen Zhao, Arman Cohan, Dennis Shasha

**发布时间:** 2025-10-07

### GPT解析

### 总结

这项研究引入了PuzzlePlex基准测试，用于评估基础模型在推理和规划方面的能力及其在复杂环境中的可扩展性。该研究包含15种不同类型的谜题，支持扩展性，并开发了细粒度性能指标。研究发现推理模型在基于指令的设置中表现更佳，而基于代码的执行虽有挑战但更具可扩展性。

### 背景

基础模型在复杂、动态环境中的推理和规划能力及其可扩展性需要更全面的评估。

### 目的

开发一个基准测试来评估基础模型的推理、规划和泛化能力，并研究它们在不同设置下的表现和扩展极限。

### 方法

引入PuzzlePlex基准测试，包含15种不同类型的谜题（确定性和随机性游戏，单人及双人场景）；实现定制化游戏策略；开发细粒度性能指标；在基于指令和基于代码的两种设置下评估前沿基础模型。

### 主要发现

推理模型在基于指令的设置中表现优于其他模型；基于代码的执行面临更大挑战，但提供了可扩展且高效的替代方案。

### 结论

PuzzlePlex基准测试能够实现有针对性的评估，并指导基础模型在推理、规划和泛化方面的未来改进。

### 翻译

这项工作研究了基础模型在推理和规划方面的能力及其在复杂、动态环境中的可扩展性。我们引入了PuzzlePlex，一个通过多样化谜题集来评估这些能力的基准测试。PuzzlePlex包含15种类型的谜题，包括不同难度的确定性和随机性游戏，以及单人场景和双人场景。PuzzlePlex框架为每个游戏提供了全面的环境，并支持扩展性，可以根据基础模型的发展生成更具挑战性的实例。此外，我们实现了定制化的游戏策略进行比较。基于这个基准测试，我们开发了细粒度的指标来衡量性能，并对前沿基础模型在两种设置下进行了深入分析：基于指令的和基于代码的。此外，我们系统地研究了它们的扩展极限。我们的研究结果表明，在基于指令的设置中，推理模型优于其他模型，而基于代码的执行则面临更大挑战，但提供了可扩展且高效的替代方案。PuzzlePlex能够实现有针对性的评估，并指导基础模型在推理、规划和泛化方面的未来改进。


### 论文摘要

This work investigates the reasoning and planning capabilities of foundation models and their scalability in complex, dynamic environments. We introduce PuzzlePlex, a benchmark designed to assess these capabilities through a diverse set of puzzles. PuzzlePlex consists of 15 types of puzzles, including deterministic and stochastic games of varying difficulty, as well as single-player and two-player scenarios. The PuzzlePlex framework provides a comprehensive environment for each game, and supports extensibility to generate more challenging instances as foundation models evolve. Additionally, we implement customized game-playing strategies for comparison. Building on this benchmark, we develop fine-grained metrics to measure performance and conduct an in-depth analysis of frontier foundation models across two settings: instruction-based and code-based. Furthermore, we systematically investigate their scaling limits. Our findings show that reasoning models outperform others in instruction-based settings, while code-based execution presents greater challenges but offers a scalable and efficient alternative. PuzzlePlex enables targeted evaluation and guides future improvements in reasoning, planning, and generalization for foundation models.

---

## 59. Nearly Instance-Optimal Parameter Recovery from Many Trajectories via Hellinger Localization

**论文链接:** [http://arxiv.org/abs/2510.06434v1](http://arxiv.org/abs/2510.06434v1)

**作者:** Eliot Shekhtman, Yichen Zhou, Ingvar Ziemann, Nikolai Matni, Stephen Tu

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文通过Hellinger局部化框架显著扩展了多轨迹设置中实例最优率的应用范围，提出了一种控制路径测度层面平方Hellinger距离的方法，并在参数空间中进行局部化，实现了在广泛条件下的实例最优边界。该方法在四个不同的案例研究中得到了验证，边界几乎匹配了来自渐近正态性的实例最优率，显著优于标准简化方法。

### 背景

从时间相关数据学习是现代机器学习的核心方面，但在多轨迹设置中（数据由许多时间索引随机过程的独立实现组成），序列学习的理解仍然不完整。这种设置反映了现代大型基础模型的训练管道，并提供了在不需要典型混合假设的情况下进行学习的可能性。

### 目的

解决多轨迹设置中实例最优边界的问题。目前只有带相关协变量的最小二乘回归具有实例最优边界，对于更一般的模型或损失函数，现有方法要么简化为i.i.d.学习，要么使用单轨迹结果，都存在有效样本量扩展的限制。

### 方法

通过Hellinger局部化框架进行最大似然估计。首先通过简化为i.i.d.学习来控制路径测度层面的平方Hellinger距离，然后在参数空间中作为二次形式进行局部化，由轨迹Fisher信息加权。

### 主要发现

在广泛的条件下，产生了与完整数据预算成比例的实例最优边界。在四个不同的案例研究中（简单马尔可夫链混合、非高斯噪声下的相关线性回归、具有非单调激活的广义线性模型以及线性注意力序列模型），边界几乎匹配了来自渐近正态性的实例最优率，显著优于标准简化方法。

### 结论

该工作显著扩展了多轨迹设置中实例最优率的应用范围，为序列学习提供了更好的理论保证，特别是在处理更一般的模型或损失函数时。

### 翻译

从时间相关数据学习是现代机器学习的核心方面。然而，我们对序列学习的理解仍然不完整，特别是在多轨迹设置中，数据由许多时间索引随机过程的独立实现组成。这种重要的设置既反映了现代大型基础模型的训练管道，又提供了在不采用单轨迹情况下典型混合假设的情况下进行学习的可能性。然而，目前只有带相关协变量的最小二乘回归具有实例最优边界；对于更一般的模型或损失函数，唯一广泛适用的保证要么简化为独立同分布学习，有效样本量仅随轨迹数量扩展，要么在每条单独轨迹混合时使用现有的单轨迹结果，有效样本量随总数据预算除以混合时间而扩展。在这项工作中，我们通过Hellinger局部化框架显著扩展了多轨迹设置中实例最优率的应用范围，这是最大似然估计的一种通用方法。我们的方法首先通过简化为独立同分布学习来控制路径测度层面的平方Hellinger距离，然后在参数空间中作为由轨迹Fisher信息加权的二次形式进行局部化。这产生了在广泛条件下与完整数据预算成比例的实例最优边界。我们在四个不同的案例研究中实现了我们的框架：简单马尔可夫链混合、非高斯噪声下的相关线性回归、具有非单调激活的广义线性模型以及线性注意力序列模型。在所有情况下，我们的边界几乎匹配了来自渐近正态性的实例最优率，显著优于标准简化方法。


### 论文摘要

Learning from temporally-correlated data is a core facet of modern machine learning. Yet our understanding of sequential learning remains incomplete, particularly in the multi-trajectory setting where data consists of many independent realizations of a time-indexed stochastic process. This important regime both reflects modern training pipelines such as for large foundation models, and offers the potential for learning without the typical mixing assumptions made in the single-trajectory case. However, instance-optimal bounds are known only for least-squares regression with dependent covariates; for more general models or loss functions, the only broadly applicable guarantees result from a reduction to either i.i.d. learning, with effective sample size scaling only in the number of trajectories, or an existing single-trajectory result when each individual trajectory mixes, with effective sample size scaling as the full data budget deflated by the mixing-time.   In this work, we significantly broaden the scope of instance-optimal rates in multi-trajectory settings via the Hellinger localization framework, a general approach for maximum likelihood estimation. Our method proceeds by first controlling the squared Hellinger distance at the path-measure level via a reduction to i.i.d. learning, followed by localization as a quadratic form in parameter space weighted by the trajectory Fisher information. This yields instance-optimal bounds that scale with the full data budget under a broad set of conditions. We instantiate our framework across four diverse case studies: a simple mixture of Markov chains, dependent linear regression under non-Gaussian noise, generalized linear models with non-monotonic activations, and linear-attention sequence models. In all cases, our bounds nearly match the instance-optimal rates from asymptotic normality, substantially improving over standard reductions.

---

## 60. Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.06419v1](http://arxiv.org/abs/2510.06419v1)

**作者:** Mert Kayaalp, Caner Turkmen, Oleksandr Shchur, Pedro Mercado, Abdul Fatir Ansari, Michael Bohlke-Schneider, Bernie Wang

**发布时间:** 2025-10-07

### GPT解析

### 总结

研究探索了构建较小预训练预测模型组合作为单一大型模型的替代方案，通过集成或模型选择技术实现竞争性性能且使用更少参数。

### 背景

时间序列基础模型是否越大越好存在疑问，需要探索替代方案。

### 目的

探索构建预测模型组合的策略，评估其在大型基准测试上的性能。

### 方法

构建较小预训练预测模型组合，应用集成或模型选择技术，使用更少参数实现高性能。

### 主要发现

专门模型组合始终优于独立训练的通用模型组合；对基础模型进行训练后处理是创建多样化专业模型的高效方法；集成和模型选择比测试时微调更计算高效。

### 结论

构建较小专业模型的组合可以在保持竞争力的同时减少参数使用，提高计算效率。

### 翻译

对于时间序列基础模型，越大总是越好吗？带着这个问题，我们探索了训练单一大型整体模型的替代方案：构建一组较小、预训练的预测模型组合。通过对这些模型组合应用集成或模型选择，我们使用更少的参数在大型基准测试上实现了竞争性性能。我们探索了设计此类模型组合的策略，发现专门模型的组合始终优于独立训练的通用模型组合。值得注意的是，我们证明了训练后处理基础模型是创建足够多样化专业模型的一种计算效率高的方法，并提供证据表明集成和模型选择比测试时微调更计算高效。


### 论文摘要

Is bigger always better for time series foundation models? With the question in mind, we explore an alternative to training a single, large monolithic model: building a portfolio of smaller, pretrained forecasting models. By applying ensembling or model selection over these portfolios, we achieve competitive performance on large-scale benchmarks using much fewer parameters. We explore strategies for designing such portfolios and find that collections of specialist models consistently outperform portfolios of independently trained generalists. Remarkably, we demonstrate that post-training a base model is a compute-effective approach for creating sufficiently diverse specialists, and provide evidences that ensembling and model selection are more compute-efficient than test-time fine-tuning.

---

## 61. Geometry-Aware Backdoor Attacks: Leveraging Curvature in Hyperbolic Embeddings

**论文链接:** [http://arxiv.org/abs/2510.06397v1](http://arxiv.org/abs/2510.06397v1)

**作者:** Ali Baheri

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究揭示了非欧几里得基础模型中存在的一种特定于几何的脆弱性，即边界驱动的不对称性可以被后门触发器利用。作者分析了这一问题，提出了几何自适应触发器，并评估了不同任务和架构下的表现，同时指出了防御方法的局限性。

### 背景

非欧几里得基础模型越来越多地将表示放置在双曲几何等弯曲空间中。这种几何结构引入了新的安全挑战。

### 目的

探究非欧几里得基础模型中的安全漏洞，特别是边界驱动的不对称性如何被后门触发器利用，并提出相应的防御方法。

### 方法

作者通过理论分析形式化了边界附近微小输入变化导致模型表示空间大幅变化的现象，提出了几何自适应触发器，并在多种任务和架构上进行了实证评估。

### 主要发现

1. 在非欧几里得模型的边界附近，微小输入变化会导致模型表示空间中不成比例的大幅变化；2. 传统输入空间检测器难以检测这些边界附近的微妙变化；3. 通过沿半径向内拉点的方法可以抑制触发器，但会牺牲模型在同一方向上的有用敏感性；4. 攻击成功率随接近边界而增加，而传统检测器则减弱，与理论趋势一致。

### 结论

非欧几里得模型中存在一种特定于几何的脆弱性，理解这种脆弱性对于设计和理解防御方法的局限至关重要。提出的几何自适应触发器展示了这种漏洞的实际影响。

### 翻译

非欧几里得基础模型越来越多地将表示放置在双曲几何等弯曲空间中。我们表明，这种几何结构会产生一种边界驱动的不对称性，后门触发器可以加以利用。在边界附近，微小的输入变化对标准的输入空间检测器来说可能看起来很微妙，但在模型的表示空间中会产生不成比例的大幅变化。我们的分析形式化了这种效应，并揭示了防御方法的一个局限性：通过沿半径向内拉点的方法可以抑制此类触发器，但代价是在同一方向上牺牲了模型的有用敏感性。基于这些见解，我们提出了一种简单的几何自适应触发器，并在不同任务和架构上进行了评估。实验表明，攻击成功率随着接近边界而增加，而传统检测器则减弱，这与理论趋势一致。这些结果共同揭示了非欧几里得模型中的一种特定于几何的脆弱性，并为设计和理解防御的局限提供了基于分析的指导。


### 论文摘要

Non-Euclidean foundation models increasingly place representations in curved spaces such as hyperbolic geometry. We show that this geometry creates a boundary-driven asymmetry that backdoor triggers can exploit. Near the boundary, small input changes appear subtle to standard input-space detectors but produce disproportionately large shifts in the model's representation space. Our analysis formalizes this effect and also reveals a limitation for defenses: methods that act by pulling points inward along the radius can suppress such triggers, but only by sacrificing useful model sensitivity in that same direction. Building on these insights, we propose a simple geometry-adaptive trigger and evaluate it across tasks and architectures. Empirically, attack success increases toward the boundary, whereas conventional detectors weaken, mirroring the theoretical trends. Together, these results surface a geometry-specific vulnerability in non-Euclidean models and offer analysis-backed guidance for designing and understanding the limits of defenses.

---

## 62. Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

**论文链接:** [http://arxiv.org/abs/2510.06377v1](http://arxiv.org/abs/2510.06377v1)

**作者:** Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, Jure Leskovec

**发布时间:** 2025-10-07

**备注:** preprint; under review

### GPT解析

### 总结

本文提出了关系Transformer(RT)架构，能够在多样化的关系数据库上预训练并直接应用于未见过的数据集和任务，无需特定微调或上下文示例检索。

### 背景

预训练transformer模型可以通过零样本提示适应新的序列建模任务，但关系领域仍缺乏跨数据集和任务的迁移架构。核心挑战在于关系数据的多样性，包括不同的异构模式、图结构和功能依赖。

### 目的

开发一种能够预训练并在多样化关系数据库上应用，直接迁移到未见数据集和任务的架构，无需任务或数据集特定的微调。

### 方法

关系Transformer(RT)架构：(1)使用表格/列元数据对单元格进行标记化，(2)通过掩码标记预测进行预训练，(3)利用新的关系注意机制处理列、行和主-外键链接。

### 主要发现

在RelBench数据集上预训练后，RT在二元分类任务上达到完全监督AUROC的94%，仅使用2200万参数模型的一次前向传递，而270亿参数的LLM仅达到84%；微调能以高样本效率达到最先进结果；RT的零样本迁移利用了任务-表格上下文、关系注意模式和模式语义。

### 结论

关系Transformer为关系数据的基础模型提供了实用路径。

### 翻译

预训练transformer模型可以通过零样本提示轻松适应新的序列建模任务，但关系领域仍然缺乏能够在不同数据集和任务间迁移的架构。核心挑战在于关系数据的多样性，包括不同的异构模式、图结构和功能依赖。在本文中，我们提出了关系Transformer(RT)架构，可以在多样化的关系数据库上进行预训练，并直接应用于未见过的数据集和任务，无需任务或数据集特定的微调，或检索上下文示例。RT(i)使用表格/列元数据对单元格进行标记化，(ii)通过掩码标记预测进行预训练，(iii)利用一种新的关系注意机制处理列、行和主-外键链接。在RelBench数据集（涵盖流失和销售预测等任务）上预训练后，RT在二元分类任务上实现了强大的零样本性能，平均达到完全监督AUROC的94%，仅使用2200万参数模型的一次前向传递，而270亿参数的大语言模型仅达到84%。微调能以高样本效率达到最先进的结果。实验表明，RT的零样本迁移利用了任务-表格上下文、关系注意模式和模式语义。总体而言，RT为关系数据的基础模型提供了实用路径。


### 论文摘要

Pretrained transformers readily adapt to new sequence modeling tasks via zero-shot prompting, but relational domains still lack architectures that transfer across datasets and tasks. The core challenge is the diversity of relational data, with varying heterogeneous schemas, graph structures and functional dependencies. In this paper, we present the Relational Transformer (RT) architecture, which can be pretrained on diverse relational databases and directly applied to unseen datasets and tasks without task- or dataset-specific fine-tuning, or retrieval of in-context examples. RT (i) tokenizes cells with table/column metadata, (ii) is pretrained via masked token prediction, and (iii) utilizes a novel \textit{Relational Attention} mechanism over columns, rows, and primary-foreign key links. Pretrained on RelBench datasets spanning tasks such as churn and sales forecasting, RT attains strong zero-shot performance, averaging 94% of fully supervised AUROC on binary classification tasks with a single forward pass of a 22M parameter model, as opposed to 84% for a 27B LLM. Fine-tuning yields state-of-the-art results with high sample efficiency. Our experiments show that RT's zero-shot transfer harnesses task-table context, relational attention patterns and schema semantics. Overall, RT provides a practical path toward foundation models for relational data.

---

## 63. EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA

**论文链接:** [http://arxiv.org/abs/2510.06371v1](http://arxiv.org/abs/2510.06371v1)

**作者:** Firoj Alam, Ali Ezzat Shahroor, Md. Arid Hasan, Zien Sheikh Ali, Hunzalah Hassan Bhatti, Mohamed Bayan Kmainasi, Shammur Absar Chowdhury, Basel Mousi, Fahim Dalvi, Nadir Durrani, Natasa Milic-Frayling

**发布时间:** 2025-10-07

**备注:** Multimodal Foundation Models, Large Language Models, Native,  Multilingual, Language Diversity, Contextual Understanding, Culturally  Informed

### GPT解析

### 总结

研究提出了一个名为EverydayMMQA的框架，用于创建大规模、基于文化的数据集，以解决多模态模型在需要文化背景知识的问答任务中的局限性，并开发了OASIS数据集作为该框架的实例。

### 背景

大规模多模态模型在视觉问答等任务上表现出色，但当查询需要基于文化、日常知识，特别是在低资源语言和代表性不足的语言中时，这些模型常常失败。

### 目的

弥合多模态模型在文化背景知识问答方面的差距，引入日常多模态多语言问答(EverydayMMQA)框架，用于创建大规模、基于文化的口语和视觉问答(SVQA)数据集。

### 方法

使用EverydayMMQA框架开发了OASIS多模态数据集，整合语音、图像和文本；包含约92万张图像和1480万个问答对，其中370万个口语问题；支持四种输入组合：仅语音、仅文本、语音+图像、文本+图像；专注于英语和阿拉伯语变体，涵盖18个国家；数据集内容反映多样化、真实世界的情境。

### 主要发现

OASIS测试模型在超越对象识别的任务上，涉及语用、常识和文化感知推理；对四个闭源模型、三个开源模型和一个微调模型进行了基准测试。

### 结论

EverydayMMQA和OASIS一起提供了一个基准和训练数据集，用于构建在文化背景下处理全面日常任务的多模态大语言模型；框架和数据集将对社区公开可用。

### 翻译

大规模多模态模型在视觉问答(VQA)等任务上取得强劲成果，但当查询需要基于文化的日常知识，特别是在低资源和代表性不足的语言中时，它们往往会失败。为了弥合这一差距，我们引入了日常多模态多语言问答(EverydayMMQA)，这是一个用于创建大规模、基于文化的口语和视觉问答(SVQA)数据集的框架。使用此框架，我们开发了OASIS，一个整合语音、图像和文本的多模态数据集。OASIS包含约92万张图像和1480万个问答对，其中370万个口语问题，支持四种独特的输入组合：仅语音、仅文本、语音+图像和文本+图像。专注于英语和阿拉伯语变体，18个国家，数据集内容经过策划以反映多样化、真实世界的情境。OASIS测试模型在涉及语用、常识和文化感知推理的超越对象识别的任务。我们对四个闭源模型、三个开源模型和一个微调模型进行了基准测试。EverydayMMQA和OASIS一起提供了一个基准和训练数据集，用于构建在文化背景下处理全面日常任务的多模态大语言模型。该框架和数据集将向社区公开。


### 论文摘要

Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they often fail when queries require culturally grounded, everyday knowledge, particularly in low-resource and underrepresented languages. To bridge this gap, we introduce Everyday Multimodal and Multilingual QA (EverydayMMQA), a framework for creating large-scale, culturally-grounded datasets for spoken and visual question answering (SVQA). Using this framework, we developed OASIS, a multimodal dataset integrating speech, images, and text. With over ~0.92M images and 14.8M QA pairs, OASIS contains 3.7M spoken questions, enabling four unique input combinations: speech-only, text-only, speech+image, and text+image. Focused on English and Arabic varieties, 18 countries, the dataset content is curated to reflect diverse, real-world situations. OASIS tests models on tasks beyond object recognition that involve pragmatic, commonsense, and culturally aware reasoning. We benchmarked four closed-source models, three open-source models, and one fine-tuned model. EverydayMMQA and OASIS together provide a benchmark and training dataset for building multimodal LLMs for a comprehensive set of everyday tasks within cultural contexts. The framework and dataset will be made publicly available to the community.

---

## 64. Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks

**论文链接:** [http://arxiv.org/abs/2510.06349v1](http://arxiv.org/abs/2510.06349v1)

**作者:** Moein E. Samadi, Andreas Schuppert

**发布时间:** 2025-10-07

### GPT解析

### 总结

该论文探讨了基础模型在现实世界应用中的局限性，特别是在动态复杂系统决策方面的挑战，并提出了基于小型智能体网络(SANs)的去中心化架构作为替代方案。

### 背景

基础模型迅速推动AI发展，但其决策是否超越人类策略尚不明确。尽管AI发展速度极快，但在许多与日常生活和社会相关的应用领域(如重症监护中动态发展疾病的诊断和治疗)只显示出适度的进展。适应复杂系统到动态环境是共同挑战，需要可靠、自适应的建模。

### 目的

开发具有最少数据和有限机制知识的自适应AI模型方法，使AI能在复杂动态系统中展现出明显的优越性，然后才能承担更广泛的决策角色。

### 方法

提出一种由交互式小型智能体网络(SANs)组成的去中心化架构，专注于代表系统专门子结构的智能体，每个智能体仅覆盖系统功能的一个子集。通过群体学习实现自适应决策。

### 主要发现

维度灾难是高效自适应的基本障碍，单一基础模型在克服这一障碍方面面临概念上的限制。多样化群体中的群体学习可以使自适应SANs在动态环境中提供比单一基础模型更好的决策，但代价是细节的可重复性降低。

### 结论

对于需要处理复杂动态系统的应用，基于SANs的去中心化架构可能比单一基础模型更有效，尽管在细节可重复性方面有所牺牲。

### 翻译

基础模型迅速推动了人工智能的发展，引发了一个问题：它们的决策最终是否会超越人类在现实世界中的策略。人工智能发展的指数级甚至可能是超指数级的速度使得此类分析变得困难。然而，许多对日常生活和社会重要的应用领域迄今为止只显示出适度的进展；一个突出的例子是在重症监护中诊断和治疗动态发展的疾病。共同的挑战是将复杂系统适应动态环境。有效的策略必须在由强相互作用功能组成的系统中优化结果，同时避免共享的副作用；这需要可靠、自适应的建模。这些任务与构建高度复杂系统的数字孪生相一致，而这些系统的机制并未被完全或定量理解。因此，开发具有最少数据和有限机制知识的方法来构建自适应AI模型至关重要。由于这一挑战超出了医学范畴，AI在承担更广泛的决策角色之前，应该在这些环境中展示出明显的优越性。我们将维度灾难视为高效自适应的基本障碍，并论证单一的基础模型在克服它方面面临概念上的限制。作为替代方案，我们提出了一种由交互式小型智能体网络(SANs)组成的去中心化架构。我们专注于代表系统专门子结构的智能体，其中每个智能体仅覆盖系统功能的一个子集。基于SANs学习行为的数学结果和现有应用的证据，我们论证多样化群体中的群体学习可以使自适应SANs在动态环境中提供比单一基础模型更好的决策，尽管代价是细节的可重复性降低。


### 论文摘要

Foundation models have rapidly advanced AI, raising the question of whether their decisions will ultimately surpass human strategies in real-world domains. The exponential, and possibly super-exponential, pace of AI development makes such analysis elusive. Nevertheless, many application areas that matter for daily life and society show only modest gains so far; a prominent case is diagnosing and treating dynamically evolving disease in intensive care.   The common challenge is adapting complex systems to dynamic environments. Effective strategies must optimize outcomes in systems composed of strongly interacting functions while avoiding shared side effects; this requires reliable, self-adaptive modeling. These tasks align with building digital twins of highly complex systems whose mechanisms are not fully or quantitatively understood. It is therefore essential to develop methods for self-adapting AI models with minimal data and limited mechanistic knowledge. As this challenge extends beyond medicine, AI should demonstrate clear superiority in these settings before assuming broader decision-making roles.   We identify the curse of dimensionality as a fundamental barrier to efficient self-adaptation and argue that monolithic foundation models face conceptual limits in overcoming it. As an alternative, we propose a decentralized architecture of interacting small agent networks (SANs). We focus on agents representing the specialized substructure of the system, where each agent covers only a subset of the full system functions. Drawing on mathematical results on the learning behavior of SANs and evidence from existing applications, we argue that swarm-learning in diverse swarms can enable self-adaptive SANs to deliver superior decision-making in dynamic environments compared with monolithic foundation models, though at the cost of reduced reproducibility in detail.

---

## 65. Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding

**论文链接:** [http://arxiv.org/abs/2510.06308v1](http://arxiv.org/abs/2510.06308v1)

**作者:** Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jinbin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu, Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang, Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiaohong Liu, Yu Qiao, Yihao Liu

**发布时间:** 2025-10-07

**备注:** 33 pages, 13 figures, 10 tables

### GPT解析

### 总结

Lumina-DiMOO是一个开源的基础模型，用于无缝的多模态生成和理解。

### 背景

现有的统一多模态模型在处理效率和任务支持方面存在局限。

### 目的

开发一个能够高效处理多种模态输入输出并支持广泛多模态任务的基础模型。

### 方法

使用全离散扩散建模来处理各种模态的输入和输出，区别于之前的自回归(AR)或混合AR-扩散范式。

### 主要发现

实现了比先前AR或混合AR-扩散范式更高的采样效率；能够支持广泛的多模态任务，包括文本到图像生成、图像到图像生成（如图像编辑、主体驱动生成和图像修复等）以及图像理解；在多个基准测试上实现了最先进的性能，超越了现有的开源统一多模态模型。

### 结论

为促进多模态和离散扩散模型研究的进一步发展，将代码和模型检查点发布给社区。

### 翻译

我们介绍了Lumina-DiMOO，这是一个用于无缝多模态生成和理解的开源基础模型。Lumina-DiMOO通过使用全离散扩散建模来处理各种模态的输入和输出，从而区别于之前的统一模型。这种创新方法使Lumina-DiMOO能够比先前的自回归(AR)或混合AR-扩散范式实现更高的采样效率，并能够支持广泛的多模态任务，包括文本到图像生成、图像到图像生成（例如图像编辑、主体驱动生成和图像修复等），以及图像理解。Lumina-DiMOO在多个基准测试上实现了最先进的性能，超越了现有的开源统一多模态模型。为了促进多模态和离散扩散模型研究的进一步发展，我们将代码和模型检查点发布给社区。项目页面：https://synbol.github.io/Lumina-DiMOO。


### 论文摘要

We introduce Lumina-DiMOO, an open-source foundational model for seamless multi-modal generation and understanding. Lumina-DiMOO sets itself apart from prior unified models by utilizing a fully discrete diffusion modeling to handle inputs and outputs across various modalities. This innovative approach allows Lumina-DiMOO to achieve higher sampling efficiency compared to previous autoregressive (AR) or hybrid AR-Diffusion paradigms and adeptly support a broad spectrum of multi-modal tasks, including text-to-image generation, image-to-image generation (e.g., image editing, subject-driven generation, and image inpainting, etc.), as well as image understanding. Lumina-DiMOO achieves state-of-the-art performance on multiple benchmarks, surpassing existing open-source unified multi-modal models. To foster further advancements in multi-modal and discrete diffusion model research, we release our code and checkpoints to the community. Project Page: https://synbol.github.io/Lumina-DiMOO.

---

