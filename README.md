# 今日论文推荐 - 2025-11-05

共 116 篇论文

---

## 1. Dynamic Reflections: Probing Video Representations with Text Alignment

**论文链接:** [http://arxiv.org/abs/2511.02767v1](http://arxiv.org/abs/2511.02767v1)

**作者:** Tyler Zhu, Tengda Han, Leonidas Guibas, Viorica Pătrăucean, Maks Ovsjanikov

**发布时间:** 2025-11-04

**备注:** 21 pages, 12 figures

### GPT解析

### 总结

本研究进行了首次全面的视频-文本表征对齐研究，探索现代视频和语言编码器的能力，并提出了参数化的测试时缩放定律。

### 背景

多模态表征对齐已被证明可以提供不同编码器在跨数据类型结构相似性和下游能力方面的见解。虽然图像与文本的对齐已取得显著进展，但视频数据的时序特性在这一背景下尚未得到充分探索。

### 目的

进行首次全面的视频-文本表征对齐研究，探究现代视频和语言编码器的性能和能力。

### 方法

研究视频-文本表征对齐，探究不同视觉和文本数据丰富度对跨模态对齐的影响，分析语义对齐与下游任务性能的关联，以及时序推理与跨模态对齐的关系。

### 主要发现

1) 跨模态对齐高度依赖于测试时提供的视觉和文本数据的丰富程度；2) 提出的参数化测试时缩放定律具有显著的预测能力；3) 与文本编码器的强对齐可能与通用视频表征和理解相关；4) 时序推理与跨模态对齐的关联为视觉和语言模型提供了挑战性测试平台。

### 结论

将视频-文本对齐引入为一种信息丰富的零样本方法，用于探测不同编码器在时空数据上的表征能力。

### 翻译

不同模态表征的对齐最近已被证明能够提供关于不同编码器在跨数据类型结构相似性和下游能力方面的见解。虽然在对齐图像与文本方面已取得重大进展，但视频数据的时序特性在此背景下仍 largely 未经探索。在本工作中，我们进行了首次全面的视频-文本表征对齐研究，探究现代视频和语言编码器的能力。我们的发现揭示了几个关键见解。首先，我们证明跨模态对齐高度依赖于测试时提供的视觉（静态图像与多帧视频）和文本（单一标题与集合）数据的丰富程度，特别是使用最先进的视频编码器时。我们提出了捕捉这种行为的参数化测试时缩放定律，并显示出与经验观察相比显著的预测能力。其次，我们研究了语义对齐与语义和非语义下游任务性能之间的相关性，提供了初步证据，表明与文本编码器的强对齐可能与通用视频表征和理解相关。最后，我们将时序推理与跨模态对齐相关联，为视觉和语言模型提供了具有挑战性的测试平台。总体而言，我们的工作将视频-文本对齐引入为一种信息丰富的零样本方法，用于探测不同编码器在时空数据上的表征能力。项目页面可在 https://video-prh.github.io/ 找到。


### 论文摘要

The alignment of representations from different modalities has recently been shown to provide insights on the structural similarities and downstream capabilities of different encoders across diverse data types. While significant progress has been made in aligning images with text, the temporal nature of video data remains largely unexplored in this context. In this work, we conduct the first comprehensive study of video-text representation alignment, probing the capabilities of modern video and language encoders. Our findings reveal several key insights. First, we demonstrate that cross-modal alignment highly depends on the richness of both visual (static images vs. multi-frame videos) and text (single caption vs. a collection) data provided at test time, especially when using state-of-the-art video encoders. We propose parametric test-time scaling laws that capture this behavior and show remarkable predictive power against empirical observations. Secondly, we investigate the correlation between semantic alignment and performance on both semantic and non-semantic downstream tasks, providing initial evidence that strong alignment against text encoders may be linked to general-purpose video representation and understanding. Finally, we correlate temporal reasoning with cross-modal alignment providing a challenging test-bed for vision and language models. Overall, our work introduces video-text alignment as an informative zero-shot way to probe the representation power of different encoders for spatio-temporal data. Project page can be found at https://video-prh.github.io/

---

## 2. From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics

**论文链接:** [http://arxiv.org/abs/2511.02427v1](http://arxiv.org/abs/2511.02427v1)

**作者:** Nicolas Schuler, Lea Dewald, Nick Baldig, Jürgen Graf

**发布时间:** 2025-11-04

**备注:** 15 pages, 6 figures, 1 table; accepted for AI-2025 Forty-fifth SGAI  International Conference on Artificial Intelligence CAMBRIDGE, ENGLAND 16-18  DECEMBER 2025

### GPT解析

### 总结

该研究调查了先进的视觉语言模型(VLMs)在场景解释和动作识别任务上的能力，特别关注适合在移动机器人背景下部署到边缘设备的小型VLMs。

### 背景

视频理解、场景解释和常识推理是使智能体能够解释视觉信息、感知环境、互动并做出决策的关键任务。大型语言模型和视觉语言模型在这些领域取得了显著进展，支持特定领域应用和零样本开放词汇任务，但计算复杂性限制了它们在边缘设备和移动机器人中的应用。

### 目的

调查最先进的VLMs在场景解释和动作识别任务上的能力，特别关注能在移动机器人背景下部署到边缘设备的小型VLMs。

### 方法

提出一个评估管道，并在多样化数据集上进行测试，包括各种真实世界城市景观、校园和室内场景。

### 主要发现

讨论了小型模型在边缘设备上的潜力，特别强调了挑战、弱点、固有模型偏差以及获取信息的应用。

### 结论

小型VLMs在边缘设备上部署具有潜力，但仍面临计算效率、准确性和推理时间之间的权衡挑战。

### 翻译

视频理解、场景解释和常识推理是非常具有挑战性的任务，它们使智能体能够解释视觉信息，允许智能体在其环境中感知、互动并做出理性决策。大型语言模型和视觉语言模型近年来在这些领域显示出显著的进步，使特定领域的应用以及零样本开放词汇任务成为可能，并能够结合多个领域。然而，所需的计算复杂性对它们在边缘设备和移动机器人环境中的应用提出了挑战，特别是在考虑准确性和推理时间之间的权衡时。在本文中，我们调查了最先进的VLMs在场景解释和动作识别任务上的能力，特别关注能够在移动机器人背景下部署到边缘设备的小型VLMs。所提出的管道在各种包含不同真实世界城市景观、校园和室内场景的多样化数据集上进行了评估。实验评估讨论了这些小型模型在边缘设备上的潜力，特别强调了挑战、弱点、固有模型偏差以及获取信息的应用。补充材料可通过以下存储库获取：https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何在边缘设备上部署小型视觉语言模型(VLMs)进行零样本场景解释的问题，特别是在移动机器人应用中。这个问题很重要，因为移动机器人需要在动态环境中自主运行，而视觉常识推理对机器人理解环境和做出决策至关重要。边缘设备上的本地解决方案对于无法保证外部服务可用性的场景特别重要，同时零样本能力允许开放域使用，不受限于预定义的动作集合，更接近真实世界场景的复杂性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了视觉常识推理的不同方法，特别是利用大型语言模型(LLMs)和视觉语言模型(VLMs)的优势。他们研究了各种模型，特别关注适用于边缘设备的小型模型(sVLMs)。作者借鉴了现有工作如ViCor(结合LLMs和VLMs的优势)、VLMaps(结合视觉语言特征与3D重建)等，设计了一个混合架构，结合本地边缘设备处理和云支持的优势，以平衡计算能力、隐私保护和实时性需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在边缘设备上使用小型VLM进行场景解释，同时保持隐私并利用零样本能力处理开放词汇场景。整体实现流程是：1)边缘设备上的小型VLM生成最近时间间隔内图像序列的文本描述；2)生成的描述被分解为名词，用于提示零样本分割和跟踪；3)使用Grounded DINO和SAM进行零样本目标检测和分割；4)生成的描述可用于各种下游任务，如与本地或云端LLMs进行进一步推理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)专注于在边缘设备上部署小型VLMs进行零样本场景解释；2)提出结合本地边缘设备和云支持的混合架构；3)在多样化的真实世界数据集上评估小型VLMs的能力；4)研究不同场景域(校园室内、校园室外和城市)的性能差异；5)提出语义引导的分割方法，专注于描述中重要的元素。相比之前的工作，这种方法不依赖外部服务器进行所有处理，保护隐私；使用小型模型而非大型模型，更适合边缘计算环境；关注真实世界场景而非受控环境；提供了在移动机器人背景下应用VLMs的全面评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文评估并展示了小型视觉语言模型在边缘设备上进行零样本场景解释的可行性，为移动机器人在真实世界环境中提供了一种隐私保护的本地解决方案。'}


### 论文摘要

Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/

---

## 3. M3PD Dataset: Dual-view Photoplethysmography (PPG) Using Front-and-rear Cameras of Smartphones in Lab and Clinical Settings

**论文链接:** [http://arxiv.org/abs/2511.02349v1](http://arxiv.org/abs/2511.02349v1)

**作者:** Jiankai Tang, Tao Zhang, Jia Li, Yiru Zhang, Mingyu Zhang, Kegang Wang, Yuming Hao, Bolin Wang, Haiyang Li, Xingyao Wang, Yuanchun Shi, Yuntao Wang, Sichong Qian

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究提出了一种基于智能手机的双视图光电容积描记术方法，通过融合面部和指尖视频数据，提高了心率监测的准确性和鲁棒性。

### 背景

便携式生理监测对心血管疾病的早期检测和管理至关重要，但当前方法通常需要专业设备限制了可及性，或者要求患者保持不切实际的姿势。基于智能手机的视频光电容积描记术虽提供了便捷的无创替代方案，但仍面临运动伪影、光照变化和单视图限制等可靠性挑战。

### 目的

解决现有方法的局限性，引入首个公开可用的双视图移动光电容积描记术数据集，并提出一种融合双视图数据的新方法，提高心血管患者心率监测的准确性和鲁棒性。

### 方法

构建M3PD数据集，包含60名参与者（47名心血管患者）通过前置和后置智能手机摄像头同时采集的面部和指尖同步视频。基于此双视图设置，提出F3Mamba模型，通过基于Mamba的时间建模融合面部和指尖视图。

### 主要发现

F3Mamba模型将心率误差比现有单视图基线降低了21.9%至30.2%，同时在具有挑战性的现实场景中提高了监测的鲁棒性。

### 结论

双视图移动光电容积描记术结合先进的融合算法可有效提高心血管患者心率监测的准确性和可靠性，为便携式心血管健康监测提供了新思路。

### 翻译

便携式生理监测对心血管疾病的早期检测和管理至关重要，但当前方法通常需要专业设备限制了可及性，或者要求患者保持不切实际的姿势。基于智能手机的视频光电容积描记术提供了便捷的无创替代方案，但仍面临由运动伪影、光照变化和单视图限制引起的可靠性挑战。很少有研究证明其在心血管患者中的可靠应用，且缺乏广泛使用的开放数据集用于跨设备准确性评估。为解决这些限制，我们引入了M3PD数据集，这是首个公开可用的双视图移动光电容积描记术数据集，包含通过前置和后置智能手机摄像头同时采集的60名参与者（包括47名心血管患者）的面部和指尖同步视频。基于这种双视图设置，我们进一步提出F3Mamba，通过基于Mamba的时间建模融合面部和指尖视图。该模型将心率误差比现有单视图基线降低了21.9%至30.2%，同时在具有挑战性的现实场景中提高了鲁棒性。数据和代码：https://github.com/Health-HCI-Group/F3Mamba。


### 论文摘要

Portable physiological monitoring is essential for early detection and management of cardiovascular disease, but current methods often require specialized equipment that limits accessibility or impose impractical postures that patients cannot maintain. Video-based photoplethysmography on smartphones offers a convenient noninvasive alternative, yet it still faces reliability challenges caused by motion artifacts, lighting variations, and single-view constraints. Few studies have demonstrated reliable application to cardiovascular patients, and no widely used open datasets exist for cross-device accuracy. To address these limitations, we introduce the M3PD dataset, the first publicly available dual-view mobile photoplethysmography dataset, comprising synchronized facial and fingertip videos captured simultaneously via front and rear smartphone cameras from 60 participants (including 47 cardiovascular patients). Building on this dual-view setting, we further propose F3Mamba, which fuses the facial and fingertip views through Mamba-based temporal modeling. The model reduces heart-rate error by 21.9 to 30.2 percent over existing single-view baselines while improving robustness in challenging real-world scenarios. Data and code: https://github.com/Health-HCI-Group/F3Mamba.

---

## 4. LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning

**论文链接:** [http://arxiv.org/abs/2511.01448v1](http://arxiv.org/abs/2511.01448v1)

**作者:** Zhengjun Huang, Zhoujin Tian, Qintian Guo, Fangyuan Zhang, Yingli Zhou, Di Jiang, Xiaofang Zhou

**发布时间:** 2025-11-03

### GPT解析

### 总结

LiCoMemory是一种端到端代理记忆框架，通过引入CogniGraph轻量级分层图解决大型语言模型记忆限制问题，在长期对话任务中表现出色。

### 背景

大型语言模型(LLM)代理具有出色的对话和推理能力，但受限于上下文窗口小和缺乏持久性记忆。

### 目的

解决现有外部记忆架构中扁平、纠缠结构导致的冗余表示、非结构化检索以及效率和准确性下降问题。

### 方法

提出LiCoMemory框架，引入CogniGraph轻量级分层图，利用实体和关系作为语义索引层，采用时间和层次感知搜索与集成重排序进行自适应知识检索。

### 主要发现

在LoCoMo和LongMemEval基准测试上，LiCoMemory在时间推理、多会话一致性和检索效率方面优于基线模型，并显著降低了更新延迟。

### 结论

LiCoMemory有效解决了大型语言模型的记忆限制问题，提高了长期对话任务中的性能和效率。

### 翻译

大型语言模型(LLM)代理展现出卓越的对话和推理能力，但仍然受限于有限的上下文窗口和持久性记忆的缺乏。最近的工作通过外部记忆架构解决这些限制，通常采用基于图的表示，但大多数采用扁平、纠缠的结构，将语义与拓扑交织在一起，导致冗余表示、非结构化检索以及效率和准确性的下降。为解决这些问题，我们提出了LiCoMemory，一个用于实时更新和检索的端到端代理记忆框架，它引入了CogniGraph，一种利用实体和关系作为语义索引层的轻量级分层图，并采用时间和层次感知搜索与集成重排序进行自适应和连贯的知识检索。在长期对话基准LoCoMo和LongMemEval上的实验表明，LiCoMemory不仅在时间推理、多会话一致性和检索效率方面优于既定的基线，而且显著降低了更新延迟。我们的官方代码和数据可在https://github.com/EverM0re/LiCoMemory获取。


### 论文摘要

Large Language Model (LLM) agents exhibit remarkable conversational and reasoning capabilities but remain constrained by limited context windows and the lack of persistent memory. Recent efforts address these limitations via external memory architectures, often employing graph-based representations, yet most adopt flat, entangled structures that intertwine semantics with topology, leading to redundant representations, unstructured retrieval, and degraded efficiency and accuracy. To resolve these issues, we propose LiCoMemory, an end-to-end agentic memory framework for real-time updating and retrieval, which introduces CogniGraph, a lightweight hierarchical graph that utilizes entities and relations as semantic indexing layers, and employs temporal and hierarchy-aware search with integrated reranking for adaptive and coherent knowledge retrieval. Experiments on long-term dialogue benchmarks, LoCoMo and LongMemEval, show that LiCoMemory not only outperforms established baselines in temporal reasoning, multi-session consistency, and retrieval efficiency, but also notably reduces update latency. Our official code and data are available at https://github.com/EverM0re/LiCoMemory.

---

## 5. DeepSpecs: Expert-Level Questions Answering in 5G

**论文链接:** [http://arxiv.org/abs/2511.01305v1](http://arxiv.org/abs/2511.01305v1)

**作者:** Aman Ganapathy Manvattira, Yifei Xu, Ziyue Dang, Songwu Lu

**发布时间:** 2025-11-03

### GPT解析

### 总结

DeepSpecs是一个通过结构化和时间推理增强的RAG系统，通过三个元数据库（SpecDB、ChangeDB和TDocDB）解决5G标准文档中的交叉引用和演变问题，显著提高了回答5G规范专业级问题的能力。

### 背景

5G技术为数十亿用户提供移动互联网接入，回答关于5G规范的专业级问题需要浏览数千页交叉引用的标准文档。现有的检索增强生成(RAG)框架依赖语义相似性，无法可靠地解决交叉引用或对规范演变进行推理。

### 目的

开发一个能够处理5G标准文档复杂性的系统，解决现有RAG框架在处理交叉引用和规范演变方面的局限性。

### 方法

提出DeepSpecs系统，使用三个元数据库：SpecDB（条款对齐的规范文本）、ChangeDB（行级版本差异）和TDocDB（标准化会议文档）。通过元数据查找递归检索引用条款解决交叉引用，通过挖掘变化并链接到变更请求来跟踪规范演变。整理了两个5G问答数据集：573条专家注释的真实世界问题和350条演变问题。

### 主要发现

DeepSpecs在多个LLM后端上优于基础模型和最先进的电信RAG系统。消融研究证实明确的交叉引用解决和演变感知检索显著提高了答案质量，强调了建模5G标准的结构和时间特性的价值。

### 结论

DeepSpecs通过结构化和时间推理有效解决了5G标准文档的复杂性，显著提高了回答关于5G规范的专业级问题的能力。

### 翻译

5G技术为数十亿用户提供了移动互联网接入。回答关于5G规范的专业级问题需要浏览数千页交叉引用的标准文档，这些标准在不同版本中不断演变。现有的检索增强生成(RAG)框架，包括电信特定方法，依赖语义相似性，无法可靠地解决交叉引用或对规范演变进行推理。我们提出了DeepSpecs，一个通过结构化和时间推理增强的RAG系统，通过三个丰富的元数据库：SpecDB（条款对齐的规范文本）、ChangeDB（行级版本差异）和TDocDB（标准化会议文档）。DeepSpecs通过元数据查找递归检索引用的条款，明确解决交叉引用，并通过挖掘变化并将它们链接到记录设计原理的变更请求来跟踪规范演变。我们整理了两个5G问答数据集：573条来自从业者论坛和教育资源的专家注释的真实世界问题，以及350条从已批准变更请求中演变而来的问题。在多个LLM后端上，DeepSpecs优于基础模型和最先进的电信RAG系统；消融研究证实明确的交叉引用解决和演变感知检索显著提高了答案质量，强调了建模5G标准的结构和时间特性的价值。


### 论文摘要

5G technology enables mobile Internet access for billions of users. Answering expert-level questions about 5G specifications requires navigating thousands of pages of cross-referenced standards that evolve across releases. Existing retrieval-augmented generation (RAG) frameworks, including telecom-specific approaches, rely on semantic similarity and cannot reliably resolve cross-references or reason about specification evolution. We present DeepSpecs, a RAG system enhanced by structural and temporal reasoning via three metadata-rich databases: SpecDB (clause-aligned specification text), ChangeDB (line-level version diffs), and TDocDB (standardization meeting documents). DeepSpecs explicitly resolves cross-references by recursively retrieving referenced clauses through metadata lookup, and traces specification evolution by mining changes and linking them to Change Requests that document design rationale. We curate two 5G QA datasets: 573 expert-annotated real-world questions from practitioner forums and educational resources, and 350 evolution-focused questions derived from approved Change Requests. Across multiple LLM backends, DeepSpecs outperforms base models and state-of-the-art telecom RAG systems; ablations confirm that explicit cross-reference resolution and evolution-aware retrieval substantially improve answer quality, underscoring the value of modeling the structural and temporal properties of 5G standards.

---

## 6. KAT-GNN: A Knowledge-Augmented Temporal Graph Neural Network for Risk Prediction in Electronic Health Records

**论文链接:** [http://arxiv.org/abs/2511.01249v1](http://arxiv.org/abs/2511.01249v1)

**作者:** Kun-Wei Lin, Yu-Chen Kuo, Hsin-Yao Wang, Yi-Ju Tseng

**发布时间:** 2025-11-03

**备注:** 10 pages, 3 figures

### GPT解析

### 总结

这篇论文提出了KAT-GNN（知识增强型时序图神经网络）框架，用于基于电子健康记录的临床风险预测，通过整合临床知识和时序动态，在冠状动脉疾病预测和住院死亡率预测任务中取得了最先进的性能。

### 背景

使用电子健康记录进行临床风险预测对于及时干预和临床决策支持至关重要。然而，建模异构和不规则的时序EHR数据存在重大挑战。

### 目的

开发一个能够有效整合临床知识和时序动态的图神经网络框架，用于提高临床风险预测的准确性。

### 方法

KAT-GNN首先从EHR中构建特定模态的患者图，然后使用两种知识来源增强这些图：（1）来自SNOMED CT的本体驱动边；（2）从EHR中提取的共现先验。随后，采用时间感知transformer来捕捉图编码的患者表示中的纵向动态。

### 主要发现

KAT-GNN在冠状动脉疾病预测中达到最先进性能（AUROC: 0.9269 ± 0.0029），在MIMIC-III（AUROC: 0.9230 ± 0.0070）和MIMIC-IV（AUROC: 0.8849 ± 0.0089）的死亡率预测中也表现出色，持续优于GRASP和RETAIN等基线模型。消融研究证实，基于知识的增强和时序建模组件都是性能提升的重要贡献者。

### 结论

将临床知识整合到图表示中，结合时间感知注意力机制，为跨不同临床任务和数据集的风险预测提供了一种有效且可推广的方法。

### 翻译

使用电子健康记录进行临床风险预测对于促进及时干预和临床决策支持至关重要。然而，建模异构和不规则的时序EHR数据存在重大挑战。我们提出了KAT-GNN（知识增强型时序图神经网络），一种基于图的框架，整合临床知识和时序动态用于风险预测。KAT-GNN首先从EHR中构建特定模态的患者图，然后使用两种知识来源增强这些图：（1）来自SNOMED CT的本体驱动边；（2）从EHR中提取的共现先验。随后，采用时间感知transformer来捕捉图编码的患者表示中的纵向动态。KAT-GNN在三个不同的数据集和任务上进行了评估：使用长庚研究数据库进行冠状动脉疾病预测，以及使用MIMIC-III和MIMIC-IV数据集进行住院死亡率预测。KAT-GNN在CAD预测中达到最先进性能，在MIMIC-III和MIMIC-IV的死亡率预测中表现出色，持续优于GRASP和RETAIN等基线模型。消融研究证实，基于知识的增强和时序建模组件都是性能提升的重要贡献者。这些发现表明，将临床知识整合到图表示中，结合时间感知注意力机制，为跨不同临床任务和数据集的风险预测提供了一种有效且可推广的方法。


### 论文摘要

Clinical risk prediction using electronic health records (EHRs) is vital to facilitate timely interventions and clinical decision support. However, modeling heterogeneous and irregular temporal EHR data presents significant challenges. We propose \textbf{KAT-GNN} (Knowledge-Augmented Temporal Graph Neural Network), a graph-based framework that integrates clinical knowledge and temporal dynamics for risk prediction. KAT-GNN first constructs modality-specific patient graphs from EHRs. These graphs are then augmented using two knowledge sources: (1) ontology-driven edges derived from SNOMED CT and (2) co-occurrence priors extracted from EHRs. Subsequently, a time-aware transformer is employed to capture longitudinal dynamics from the graph-encoded patient representations. KAT-GNN is evaluated on three distinct datasets and tasks: coronary artery disease (CAD) prediction using the Chang Gung Research Database (CGRD) and in-hospital mortality prediction using the MIMIC-III and MIMIC-IV datasets. KAT-GNN achieves state-of-the-art performance in CAD prediction (AUROC: 0.9269 $\pm$ 0.0029) and demonstrated strong results in mortality prediction in MIMIC-III (AUROC: 0.9230 $\pm$ 0.0070) and MIMIC-IV (AUROC: 0.8849 $\pm$ 0.0089), consistently outperforming established baselines such as GRASP and RETAIN. Ablation studies confirm that both knowledge-based augmentation and the temporal modeling component are significant contributors to performance gains. These findings demonstrate that the integration of clinical knowledge into graph representations, coupled with a time-aware attention mechanism, provides an effective and generalizable approach for risk prediction across diverse clinical tasks and datasets.

---

## 7. Fleming-VL: Towards Universal Medical Visual Reasoning with Multimodal LLMs

**论文链接:** [http://arxiv.org/abs/2511.00916v1](http://arxiv.org/abs/2511.00916v1)

**作者:** Yan Shu, Chi Liu, Robin Chen, Derek Li, Bryan Dai

**发布时间:** 2025-11-02

### GPT解析

### 总结

Fleming-VL是一个统一的端到端框架，用于跨异构模态的综合医学视觉理解。通过三种关键策略解决了医学数据异质性和格式不一致的挑战，并在多个基准测试上取得了最先进的性能。

### 背景

多模态大语言模型(MLLMs)在通用领域表现出色，研究人员正致力于赋予其医学对话能力。然而，医学数据具有异质性，包含2D图像、3D体积扫描和时序视频序列等多种模态，这些模态间的领域差距和数据格式不一致阻碍了统一医学MLLMs的发展。

### 目的

解决医学数据异质性和格式不一致的挑战，开发一个统一的端到端框架，用于跨异构模态的综合医学视觉理解。

### 方法

从数据角度出发通过三种策略：(1)整合自然域和医学特定领域的长上下文数据扩大预训练；(2)补充稀有医学数据（包括整体视频分析和代表性不足的2D模态）；(3)扩展评估框架，纳入3D体积和视频理解基准。通过监督微调(SFT)和组相对策略优化(GRPO)开发了多种模型规模的Fleming-VL。

### 主要发现

Fleming-VL在多个基准测试上取得了最先进的性能，包括医学VQA、视频问答和3D医学图像理解。

### 结论

Fleming-VL成功解决了医学数据异质性和格式不一致的挑战，为跨模态医学视觉理解提供了统一框架。作者公开发布了Fleming-VL以促进医学AI的透明、可复现和可审计的进展。

### 翻译

多模态大语言模型(MLLMs)已在各种通用领域场景中展现出显著的有效性，如视觉问答和图像描述。最近，研究人员越来越专注于赋予MLLMs医学对话能力，这对临床应用具有重要前景。然而，医学数据由于其异质性而呈现独特挑战——包含多种模态，包括2D图像、3D体积扫描和时序视频序列。这些模态之间的显著领域差距和数据格式不一致阻碍了统一医学MLLMs的发展。为应对这些挑战，我们提出了Fleming-VL，一个用于跨异构模态综合医学视觉理解的统一端到端框架。Fleming-VL从数据角度通过三种关键策略解决这个问题：(1)通过整合自然域和医学特定领域的长上下文数据扩大预训练；(2)通过补充稀有医学数据（包括整体视频分析和代表性不足的2D模态，如超声和皮肤镜图像）来完善微调；(3)扩展现有评估框架，纳入3D体积和视频理解基准。通过监督微调(SFT)和组相对策略优化(GRPO)，我们开发了多种模型规模的Fleming-VL。大量实验表明，Fleming-VL在多个基准测试上取得了最先进的性能，包括医学VQA、视频问答和3D医学图像理解。我们公开发布Fleming-VL，以促进医学AI的透明、可复现和可审计的进展。


### 论文摘要

Multimodal Large Language Models (MLLMs) have demonstrated remarkable effectiveness in various general-domain scenarios, such as visual question answering and image captioning. Recently, researchers have increasingly focused on empowering MLLMs with medical conversational abilities, which hold significant promise for clinical applications. However, medical data presents unique challenges due to its heterogeneous nature -- encompassing diverse modalities including 2D images, 3D volumetric scans, and temporal video sequences. The substantial domain gap and data format inconsistencies across these modalities have hindered the development of unified medical MLLMs. To address these challenges, we propose Fleming-VL, a unified end-to-end framework for comprehensive medical visual understanding across heterogeneous modalities. Fleming-VL tackles this problem from a data-centric perspective through three key strategies: (1) scaling up pretraining by integrating long-context data from both natural and medical-specific domains; (2) complementing fine-tuning with rare medical data, including holistic video analysis and underrepresented 2D modalities such as ultrasound and dermoscopy images; (3) extending existing evaluation frameworks to incorporate 3D volumetric and video understanding benchmarks. Through supervised fine-tuning (SFT) and group relative policy optimization (GRPO), we develop Fleming-VL in multiple model scales. Extensive experiments demonstrate that Fleming-VL achieves state-of-the-art performance across multiple benchmarks, including medical VQA, video QA, and 3D medical image understanding. We publicly release Fleming-VL to promote transparent, reproducible, and auditable progress in medical AI.

---

## 8. A Systematic Review of Spatio-Temporal Statistical Models: Theory, Structure, and Applications

**论文链接:** [http://arxiv.org/abs/2511.00422v1](http://arxiv.org/abs/2511.00422v1)

**作者:** Isabella Habereder, Thomas Kneib, Isao Echizen, Timo Spinde

**发布时间:** 2025-11-01

### GPT解析

### 总结

这是一项关于时空数据统计模型的系统文献综述，提出了时空模型结构的分类方案，并分析了不同领域中的应用情况和模型特点。

### 背景

具有时空属性的数据在许多研究领域普遍存在，分析时空关系的统计模型被广泛应用。现有的综述要么专注于特定领域，要么专注于特定模型类型，缺乏全面的、跨学科的综合概述。

### 目的

为了解决现有综述的局限性，作者旨在提出时空模型结构的分类方案，并突出它们在常见领域的应用。

### 方法

作者遵循PRISMA指南进行了系统文献综述，搜索了两个数据库，时间跨度为2021-2025年，确定了83篇符合标准的出版物。

### 主要发现

层次模型是最常使用的模型；大多数模型包含加性成分以考虑时空依赖性；不同应用领域的首选模型结构不同；研究工作主要集中在少数特定学科，尽管时空数据具有更广泛的相关性；可重复性仍然有限。

### 结论

作者的综述不仅为跨学科比较模型结构提供了灵感，还强调了提高透明度、可访问性和跨领域知识转移的机会。

### 翻译

具有时空属性的数据在许多研究领域普遍存在，分析时空关系的统计模型被广泛应用。现有的综述要么专注于特定领域，要么专注于特定模型类型，造成缺乏全面的、跨学科综合概述的空白。为解决这一问题，我们遵循PRISMA指南进行了系统文献综述，搜索了两个数据库中2021-2025年的文献，确定了83篇符合我们标准的出版物。我们提出了时空模型结构的分类方案，并突出了它们在常见领域中的应用：流行病学、生态学、公共卫生、经济学和犯罪学。尽管不同领域的任务有所不同，但许多模型具有相似之处。我们发现层次模型是最常使用的，大多数模型包含加性成分以考虑时空依赖性。应用领域的首选模型结构各不相同。我们还注意到，尽管时空数据具有更广泛的相关性，但研究工作主要集中在少数特定学科。此外，我们发现可重复性仍然有限。因此，我们的综述不仅为跨学科比较模型结构提供了灵感，还强调了提高透明度、可访问性和跨领域知识转移的机会。


### 论文摘要

Data with spatial-temporal attributes are prevalent across many research fields, and statistical models for analyzing spatio-temporal relationships are widely used. Existing reviews focus either on specific domains or model types, creating a gap in comprehensive, cross-disciplinary overviews. To address this, we conducted a systematic literature review following the PRISMA guidelines, searched two databases for the years 2021-2025, and identified 83 publications that met our criteria. We propose a classification scheme for spatio-temporal model structures and highlight their application in the most common fields: epidemiology, ecology, public health, economics, and criminology. Although tasks vary by domain, many models share similarities. We found that hierarchical models are the most frequently used, and most models incorporate additive components to account for spatial-temporal dependencies. The preferred model structures differ among fields of application. We also observe that research efforts are concentrated in only a few specific disciplines, despite the broader relevance of spatio-temporal data. Furthermore, we notice that reproducibility remains limited. Our review, therefore, not only offers inspiration for comparing model structures in an interdisciplinary manner but also highlights opportunities for greater transparency, accessibility, and cross-domain knowledge transfer.

---

## 9. LongCat-Flash-Omni Technical Report

**论文链接:** [http://arxiv.org/abs/2511.00279v1](http://arxiv.org/abs/2511.00279v1)

**作者:** Meituan LongCat Team, Bairui Wang, Bayan, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, Chen Chen, Chengxu Yang, Chengzuo Yang, Cong Han, Dandan Peng, Delian Ruan, Detai Xin, Disong Wang, Dongchao Yang, Fanfan Liu, Fengjiao Chen, Fengyu Yang, Gan Dong, Gang Huang, Gang Xu, Guanglu Wan, Guoqiang Tan, Guoqiao Yu, Haibo Qiu, Hao Lu, Hongbo Liu, Hongyu Xiang, Jiaheng Wu, Jian Yang, Jiaxing Liu, Jing Huang, Jingang Wang, Jinrui Ding, Juchao Jiang, Jun Kuang, Jun Wang, Junhui Mei, Ke Ding, Kefeng Zhang, Lei Chen, Liang Shi, Limeng Qiao, Liming Zheng, Lin Ma, Liuyang Guo, Liya Ma, Luying Sun, Man Gao, Mengshen Zhu, Miao Cao, Minliang Lin, Nuo Xu, Peng Shi, Qi Zhang, Qian Fang, Qian Wang, Qian Yang, Quanxiu Wang, Rongxiang Weng, Rongxin Guo, Ruoxuan Liang, Senbin Yang, Shanbo Xu, Shanglin Lei, Shengze Ye, Shimin Chen, Shuaiqi Chen, Shujie Hu, Shuo Li, Siqi Yang, Siyu Xu, Siyu Ren, Song Li, Songxiang Liu, Tianhao Bai, Tianye Dai, Wei Hong, Wei Wang, Weixiao Zhao, Wengang Cao, Wenlong Zhu, Wenlong He, Xi Su, Xi Nan, Xiaohan Zhao, Xiaohao Wang, Xiaoyu Zhao, Xiaoyu Wang, Xiaoyu Li, Xin Pan, Xin Chen, Xiusong Sun, Xu Xiang, Xudong Xing, Xuezhi Cao, Xunliang Cai, Yang Yang, Yanli Tan, Yao Yao, Yerui Sun, Yi Chen, Yifan Lu, Yin Gong, Yining Zhang, Yitian Chen, Yiyang Gan, Yuchen Tang, Yuchen Xie, Yueqian Wang, Yuewen Zheng, Yufei Zhang, Yufeng Zhong, Yulei Qian, Yuqi Peng, Yuwei Jiang, Zeyang Hu, Zheng Zhang, Zhengkun Tian, Zhiqing Hong, Zhixiong Zeng, Zhuqi Mi, Ziran Li, Ziwen Wang, Ziyi Zhao, Ziyuan Zhuang, Zizhe Zhao

**发布时间:** 2025-10-31

### GPT解析

### 总结

LongCat-Flash-Omni是一个最先进的开源多模态模型，具有5600亿参数，擅长实时音频视觉交互。它采用课程启发的渐进式训练策略，从简单到复杂的模态序列建模任务过渡，在保持强大单模态能力的同时获得全面的多模态能力。

### 背景

基于LongCat-Flash模型，该模型采用高性能的捷径连接专家混合架构，具有零计算专家。LongCat-Flash-Omni集成了高效的多模态感知和语音重建模块。

### 目的

开发一个能够实现低延迟实时音频视觉交互的大型多模态模型，同时保持高性能和效率。

### 方法

采用课程启发的渐进式训练策略；基于LongCat-Flash的捷径连接MoE架构；集成高效多模态感知和语音重建模块；开发模态解耦并行化方案来管理大规模多模态训练中的数据和模型异质性。

### 主要发现

尽管模型庞大（5600亿参数，其中270亿被激活），仍能实现低延迟实时音频视觉交互；模态解耦并行化方案效率高，能维持文本-only训练超过90%的吞吐量；在多模态基准测试中取得开源模型的最先进性能；在文本、图像、视频理解以及音频理解和生成等多种模态特定任务上具有高度竞争力。

### 结论

LongCat-Flash-Omni是一个高效的大型多模态模型，通过创新的训练策略和架构设计，实现了实时音频视觉交互，并在各种任务上取得了优异性能。研究团队开源了该模型，以促进未来的研究和社区发展。

### 翻译

我们介绍了LongCat-Flash-Omni，这是一个最先进的开源多模态模型，具有5600亿参数，擅长实时音频视觉交互。通过采用课程启发的渐进式训练策略，从简单到复杂的模态序列建模任务过渡，LongCat-Flash-Omni在保持强大单模态能力的同时获得了全面的多模态能力。基于采用高性能捷径连接专家混合架构且具有零计算专家的LongCat-Flash，LongCat-Flash-Omni集成了高效的多模态感知和语音重建模块。尽管其庞大的5600亿参数（其中270亿被激活），LongCat-Flash-Omni仍实现了低延迟的实时音频视觉交互。在训练基础设施方面，我们开发了一种模态解耦并行化方案，专门用于管理大规模多模态训练中固有的数据和模型异质性。这种创新方法通过维持文本-only训练超过90%的吞吐量，展示了卓越的效率。广泛的评估表明，LongCat-Flash-Omni在开源模型的多模态基准测试中取得了最先进的性能。此外，它在广泛的模态特定任务上提供了高度竞争性的结果，包括文本、图像和视频理解，以及音频理解和生成。我们全面概述了模型架构设计、训练流程和数据策略，并开源了该模型，以促进社区未来的研究和开发。


### 论文摘要

We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community.

---

## 10. FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.00141v1](http://arxiv.org/abs/2511.00141v1)

**作者:** Janghoon Cho, Jungsoo Lee, Munawar Hayat, Kyuwoong Hwang, Fatih Porikli, Sungha Choi

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了FLoC框架，一种基于设施定位函数的高效视觉标记压缩方法，用于解决长视频理解中视觉标记过多导致的扩展性问题，在保证接近最优性能的同时显著减少视觉标记数量。

### 背景

最近的长视频理解研究利用了大型多模态模型(LMMs)先进的视觉-语言推理能力，推动了专门处理长视频序列的视频-LMMs的发展。然而，这些模型的扩展性受到长视频序列产生的大量视觉标记的严重限制。

### 目的

解决长视频理解中视觉标记过多导致的模型扩展性问题，开发一种高效的方法来压缩视觉标记，同时保持模型的性能。

### 方法

提出FLoC框架，基于设施定位函数的视觉标记压缩方法，通过懒贪婪算法快速选择紧凑且具有代表性和多样性的视觉标记子集，在预定义的视觉标记数量预算内工作。该方法无需训练，与模型和查询无关。

### 主要发现

在Video-MME、MLVU和LongVideoBench等大规模基准测试上的广泛评估表明，该框架始终优于最近的压缩技术，展示了其在解决长视频理解关键挑战方面的有效性和稳健性，以及处理速度方面的效率。

### 结论

FLoC框架为长视频理解提供了一个通用的解决方案，能够无缝集成到各种视频-LLMs和现有工作流程中，显著减少视觉标记数量而不牺牲性能。

### 翻译

最近的长视频理解研究利用了大型多模态模型(LMMs)先进的视觉-语言推理能力，推动了专门处理长视频序列的视频-LMMs的发展。然而，这些模型的扩展性受到长视频序列产生的大量视觉标记的严重限制。为应对这一挑战，本文提出了FLoC，一种基于设施定位函数的高效视觉标记压缩框架，这是一种在视觉标记数量预定义预算内快速选择紧凑但高度代表性和多样性视觉标记子集的原则性方法。通过集成懒贪婪算法，我们的方法通过快速选择紧凑的标记子集实现了显著的效率提升，在保证接近最优性能的同时大幅减少了视觉标记数量。值得注意的是，我们的方法无需训练，与模型和查询无关，提供了一个通用的解决方案，可以无缝集成到各种视频-LLMs和现有工作流程中。在Video-MME、MLVU和LongVideoBench等大规模基准上的广泛评估表明，我们的框架始终优于最近的压缩技术，这不仅突显了其在解决长视频理解关键挑战方面的有效性和稳健性，也展示了其在处理速度方面的效率。


### 论文摘要

Recent studies in long video understanding have harnessed the advanced visual-language reasoning capabilities of Large Multimodal Models (LMMs), driving the evolution of video-LMMs specialized for processing extended video sequences. However, the scalability of these models is severely limited by the overwhelming volume of visual tokens generated from extended video sequences. To address this challenge, this paper proposes FLoC, an efficient visual token compression framework based on the facility location function, a principled approach that swiftly selects a compact yet highly representative and diverse subset of visual tokens within a predefined budget on the number of visual tokens. By integrating the lazy greedy algorithm, our method achieves remarkable efficiency gains by swiftly selecting a compact subset of tokens, drastically reducing the number of visual tokens while guaranteeing near-optimal performance. Notably, our approach is training-free, model-agnostic, and query-agnostic, providing a versatile solution that seamlessly integrates with diverse video-LLMs and existing workflows. Extensive evaluations on large-scale benchmarks, such as Video-MME, MLVU, and LongVideoBench, demonstrate that our framework consistently surpasses recent compression techniques, highlighting not only its effectiveness and robustness in addressing the critical challenges of long video understanding, but also its efficiency in processing speed.

---

## 11. AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency

**论文链接:** [http://arxiv.org/abs/2511.00107v1](http://arxiv.org/abs/2511.00107v1)

**作者:** Piyushkumar Patel

**发布时间:** 2025-10-30

### GPT解析

### 总结

该研究提出了MOVAI框架，解决了文本到视频生成中的时间一致性、组合理解和精细控制问题，通过创新的场景解析、注意力机制和视频细化模块实现了高质量视频生成。

### 背景

文本到视频生成是生成式人工智能的关键前沿，但现有方法在保持时间一致性、组合理解和精细控制视觉叙事方面存在困难。

### 目的

开发一个名为MOVAI的新型分层框架，用于高保真文本到视频合成，结合组合场景理解和时间感知扩散模型。

### 方法

MOVAI框架包含三个关键创新：(1)组合场景解析器(CSP)将文本描述分解为带有时间注释的分层场景图；(2)时间-空间注意力机制(TSAM)确保帧间连贯运动动态同时保持空间细节；(3)渐进式视频细化(PVR)模块通过多尺度时间推理迭代提高视频质量。

### 主要发现

在标准基准上的实验表明，MOVAI实现了最先进的性能，相比现有方法，LPIPS指标提高15.3%，FVD指标提高12.7%，用户偏好研究提高18.9%。

### 结论

MOVAI框架在生成具有真实时间动态和精细语义控制的多对象复杂场景方面表现出特别优势。

### 翻译

文本到视频生成已成为生成式人工智能的关键前沿，然而现有方法在保持时间一致性、组合理解和精细控制视觉叙事方面存在困难。我们提出了MOVAI（多模态原始视频AI），这是一种创新的分层框架，将组合场景理解与时间感知扩散模型相结合，用于高保真文本到视频合成。我们的方法引入了三个关键创新：(1)组合场景解析器(CSP)，将文本描述分解为带有时间注释的分层场景图；(2)时间-空间注意力机制(TSAM)，确保帧间连贯的运动动态同时保持空间细节；(3)渐进式视频细化(PVR)模块，通过多尺度时间推理迭代提高视频质量。在标准基准上的广泛实验表明，MOVAI实现了最先进的性能，与现有方法相比，在LPIPS指标上提高15.3%，在FVD指标上提高12.7%，在用户偏好研究中提高18.9%。我们的框架在生成具有真实时间动态和精细语义控制的多对象复杂场景方面表现出特别优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决文本到视频生成中的时间一致性、组合理解和精细控制问题。这些问题很重要，因为视频不仅是图像集合，而是复杂的时间叙事，需要确保物体在帧间保持一致、动作流畅自然，同时能准确理解文本描述中的复杂场景关系。现有方法常出现闪烁、物体变形、控制不足等问题，限制了视频生成技术的实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到视频生成比图像生成更具挑战性，需要从根本上设计时间建模，而非简单地在图像生成基础上添加时间维度。他们借鉴了文本到图像生成中的扩散模型技术（如Stable Diffusion），视频生成中的3D卷积和Transformer架构（如CogVideo），以及组合理解中的场景图生成技术。基于这些现有工作，作者设计了MOVAI框架，从基础层面构建专门用于时间视觉叙事的系统。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将文本描述分解为层次化场景图，同时建模帧内空间关系和跨帧时间依赖，通过多尺度时间推理渐进式提高视频质量。整体流程分为四个阶段：1)文本输入处理：使用BERT编码器将文本转换为密集嵌入；2)场景理解：CSP模块通过图神经网络生成包含对象、关系和时间约束的场景图；3)注意力处理：TSAM模块通过空间、时间和跨模态三种注意力机制确保生成一致性；4)视频生成：PVR模块通过三个分辨率级别迭代细化视频质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三个：1)组合场景解析器(CSP)：将文本分解为层次化场景图，提供精细控制；2)时间-空间注意力机制(TSAM)：统一建模空间和时间关系，保持物体一致性并确保流畅运动；3)渐进式视频细化(PVR)：多阶段细化过程提高生成稳定性。相比之前工作，MOVAI不是简单地将图像生成方法扩展到视频，而是专为时间视觉叙事设计的系统，解决了现有方法在复杂场景中表现不佳的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MOVAI通过创新的组合场景解析、时间-空间注意力和渐进式视频细化框架，显著提高了文本到视频生成的时间一致性和视觉质量，为复杂场景中的高质量视频生成提供了新方法。'}


### 论文摘要

Text to video generation has emerged as a critical frontier in generative artificial intelligence, yet existing approaches struggle with maintaining temporal consistency, compositional understanding, and fine grained control over visual narratives. We present MOVAI (Multimodal Original Video AI), a novel hierarchical framework that integrates compositional scene understanding with temporal aware diffusion models for high fidelity text to video synthesis. Our approach introduces three key innovations: (1) a Compositional Scene Parser (CSP) that decomposes textual descriptions into hierarchical scene graphs with temporal annotations, (2) a Temporal-Spatial Attention Mechanism (TSAM) that ensures coherent motion dynamics across frames while preserving spatial details, and (3) a Progressive Video Refinement (PVR) module that iteratively enhances video quality through multi-scale temporal reasoning. Extensive experiments on standard benchmarks demonstrate that MOVAI achieves state-of-the-art performance, improving video quality metrics by 15.3% in LPIPS, 12.7% in FVD, and 18.9% in user preference studies compared to existing methods. Our framework shows particular strength in generating complex multi-object scenes with realistic temporal dynamics and fine-grained semantic control.

---

## 12. MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning

**论文链接:** [http://arxiv.org/abs/2511.00085v1](http://arxiv.org/abs/2511.00085v1)

**作者:** Peilin Tan, Chuanqi Shi, Dian Tu, Liang Xie

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了MaGNet，一种基于Mamba双超图网络的股票趋势预测方法，通过三个关键创新解决了现有方法在捕捉时间依赖性和动态股票间互动方面的局限性。

### 背景

股票趋势预测对盈利交易策略和投资组合管理至关重要，但由于市场波动性、复杂时间动态和股票间多维关系的存在，这一任务极具挑战性。现有方法难以有效捕捉时间依赖性和动态股票间互动，常忽略市场横截面影响，依赖静态相关性，对节点和边采用统一处理，并混淆多样化关系。

### 目的

开发一种新型股票预测模型，有效捕捉时间依赖性和动态股票间互动，提高预测准确性和投资回报。

### 方法

MaGNet包含三个关键创新：(1) MAGE块：利用双向Mamba和自适应门控机制进行上下文时间建模，集成稀疏专家混合层动态适应不同市场条件，使用多头注意力捕获全局依赖；(2) 特征级和股票级二维时空注意力模块：实现多元特征精确融合和跨股票依赖关系，桥接时间建模与关系推理；(3) 双超图框架：包含时间约束超图(TCH)捕获细粒度因果依赖，以及全局概率超图(GPH)建模市场范围模式，实现多尺度关系学习。

### 主要发现

在六个主要股票指数上的广泛实验表明，MaGNet在预测性能和投资回报方面均优于最先进方法，且具有出色的风险管理能力。

### 结论

MaGNet通过创新的架构设计有效解决了股票趋势预测中的关键挑战，能够处理市场波动性、复杂时间动态和股票间多维关系，为交易策略和投资组合管理提供了更可靠的预测工具。

### 翻译

股票趋势预测对盈利交易策略和投资组合管理至关重要，但由于市场波动性、复杂的时间动态和股票间的多维关系，这一任务仍然具有挑战性。现有方法难以有效捕捉时间依赖性和动态的股票间互动，常常忽略市场横截面影响，依赖静态相关性，对节点和边采用统一处理，并混淆多样化关系。这项工作引入了MaGNet，一种用于股票预测的新型Mamba双超图网络，整合了三个关键创新：(1) MAGE块，利用双向Mamba和自适应门控机制进行上下文时间建模，并集成稀疏专家混合层以动态适应不同市场条件，同时使用多头注意力捕获全局依赖；(2) 特征级和股票级二维时空注意力模块实现多元特征的精确融合和跨股票依赖关系，有效增强信息量同时保留内在数据结构，桥接时间建模与关系推理；(3) 双超图框架包括时间约束超图(TCH)，通过时间约束捕获细粒度因果依赖，以及全局概率超图(GPH)，通过软超边分配和Jensen-Shannon散度加权机制建模市场范围模式，共同分离局部时间影响与瞬时全局结构，实现多尺度关系学习。在六个主要股票指数上的广泛实验表明，MaGNet在预测性能和投资回报方面均优于最先进方法，并具有出色的风险管理能力。代码可在https://github.com/PeilinTime/MaGNet获取。


### 论文摘要

Stock trend prediction is crucial for profitable trading strategies and portfolio management yet remains challenging due to market volatility, complex temporal dynamics and multifaceted inter-stock relationships. Existing methods struggle to effectively capture temporal dependencies and dynamic inter-stock interactions, often neglecting cross-sectional market influences, relying on static correlations, employing uniform treatments of nodes and edges, and conflating diverse relationships. This work introduces MaGNet, a novel Mamba dual-hyperGraph Network for stock prediction, integrating three key innovations: (1) a MAGE block, which leverages bidirectional Mamba with adaptive gating mechanisms for contextual temporal modeling and integrates a sparse Mixture-of-Experts layer to enable dynamic adaptation to diverse market conditions, alongside multi-head attention for capturing global dependencies; (2) Feature-wise and Stock-wise 2D Spatiotemporal Attention modules enable precise fusion of multivariate features and cross-stock dependencies, effectively enhancing informativeness while preserving intrinsic data structures, bridging temporal modeling with relational reasoning; and (3) a dual hypergraph framework consisting of the Temporal-Causal Hypergraph (TCH) that captures fine-grained causal dependencies with temporal constraints, and Global Probabilistic Hypergraph (GPH) that models market-wide patterns through soft hyperedge assignments and Jensen-Shannon Divergence weighting mechanism, jointly disentangling localized temporal influences from instantaneous global structures for multi-scale relational learning. Extensive experiments on six major stock indices demonstrate MaGNet outperforms state-of-the-art methods in both superior predictive performance and exceptional investment returns with robust risk management capabilities. Codes available at: https://github.com/PeilinTime/MaGNet.

---

## 13. UniLION: Towards Unified Autonomous Driving Model with Linear Group RNNs

**论文链接:** [http://arxiv.org/abs/2511.01768v1](http://arxiv.org/abs/2511.01768v1)

**作者:** Zhe Liu, Jinghua Hou, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, Xiang Bai

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一个名为UniLION的统一自动驾驶模型，能够高效处理大规模LiDAR点云、高分辨率多视角图像和事件时间序列数据，基于线性群组RNN算子。该模型作为单一通用架构，可无缝支持多种专业变体，并在多项核心任务上取得具有竞争力的甚至最先进的性能。

### 背景

Transformer模型在各个领域表现出色，但其二次注意力机制在处理长序列数据时引入了显著的计算开销。

### 目的

开发一种统一的多模态自动驾驶模型，能够处理多种数据类型，并在3D感知、预测和规划等核心任务上保持高性能，同时简化多模态和多任务自动驾驶系统的设计。

### 方法

提出了UniLION模型，基于线性群组RNN算子，能够高效处理大规模LiDAR点云、高分辨率多视角图像和事件时间序列。该模型作为单一架构支持多种专业变体（仅LiDAR、时间LiDAR、多模态、多模态时间融合配置），无需显式的时间或多模态融合模块。

### 主要发现

UniLION在3D感知（3D目标检测、3D目标跟踪、3D占用预测、BEV地图分割）、预测（运动预测）和规划（端到端规划）等广泛核心任务中持续提供具有竞争力和最先进的性能。

### 结论

这种统一范式自然地简化了多模态和多任务自动驾驶系统的设计，同时保持卓越性能，为自动驾驶中3D基础模型的发展提供了新视角。

### 翻译

尽管Transformer已在各个领域展现出卓越的能力，但其二次注意力机制在处理长序列数据时引入了显著的计算开销。在本文中，我们提出了一个统一的自动驾驶模型UniLION，它基于线性群组RNN算子（即对分组特征执行线性RNN），能够高效处理大规模LiDAR点云、高分辨率多视角图像和事件时间序列。值得注意的是，UniLION作为一个单一的多功能架构，可以无缝支持多种专业变体（即仅LiDAR、时间LiDAR、多模态和多模态时间融合配置），而无需显式的时间或多模态融合模块。此外，UniLION在广泛的核心任务中持续提供具有竞争力和最先进的性能，包括3D感知（例如3D目标检测、3D目标跟踪、3D占用预测、BEV地图分割）、预测（例如运动预测）和规划（例如端到端规划）。这种统一范式自然地简化了多模态和多任务自动驾驶系统的设计，同时保持卓越性能。最终，我们希望UniLION能为自动驾驶中3D基础模型的发展提供新的视角。代码可在https://github.com/happinesslz/UniLION获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中多模态数据（激光雷达点云和摄像头图像）和时间序列信息的统一处理问题。现实中，自动驾驶需要同时处理来自不同传感器的异构数据和时间维度的信息，而现有方法通常需要复杂的融合模块和顺序依赖的架构，导致系统复杂、计算效率低。解决这个问题可以简化自动驾驶系统设计，提高计算效率，增强系统鲁棒性和适应性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到Transformer的二次方复杂度不适合处理长序列数据，而线性RNN具有线性计算复杂度的优势。他们借鉴了之前的工作LION（基于线性RNN的3D目标检测），并扩展到更广泛的自动驾驶任务。作者还参考了BEV表示方法和多模态融合技术，但通过线性组RNN的创新应用，消除了对显式融合模块的需求，实现了更简洁高效的统一架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用线性组RNN作为统一架构的基础，通过直接标记级连接将多模态（激光雷达和图像）和时间信息统一处理，消除显式的融合模块，生成紧凑的BEV特征表示。整体流程包括：1) 编码阶段处理激光雷达点云和多视角图像；2) 使用3D稀疏窗口分区将输入体素分组；3) 通过UniLION块进行特征交互；4) 使用自回归体素生成策略增强特征；5) 通过BEV主干网络生成统一表示；6) 并行执行各种下游任务（检测、跟踪等）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一异构输入，通过直接标记连接集成多模态和时间信息；2) 统一模型架构，实现不同输入格式的参数共享；3) 统一输出表示，将多模态时间信息压缩为紧凑BEV特征；4) 在多种任务上实现竞争性或最先进性能。相比之前工作，UniLION消除了显式融合模块，使用线性RNN降低计算复杂度，支持单一模型处理多种配置，实现并行多任务学习而非顺序依赖，提高了系统灵活性和故障容错能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniLION提出了一种基于线性组RNN的统一自动驾驶模型框架，能够高效处理多模态和时间序列数据，在保持高性能的同时显著简化了系统架构并提高了计算效率。'}


### 论文摘要

Although transformers have demonstrated remarkable capabilities across various domains, their quadratic attention mechanisms introduce significant computational overhead when processing long-sequence data. In this paper, we present a unified autonomous driving model, UniLION, which efficiently handles large-scale LiDAR point clouds, high-resolution multi-view images, and even temporal sequences based on the linear group RNN operator (i.e., performs linear RNN for grouped features). Remarkably, UniLION serves as a single versatile architecture that can seamlessly support multiple specialized variants (i.e., LiDAR-only, temporal LiDAR, multi-modal, and multi-modal temporal fusion configurations) without requiring explicit temporal or multi-modal fusion modules. Moreover, UniLION consistently delivers competitive and even state-of-the-art performance across a wide range of core tasks, including 3D perception (e.g., 3D object detection, 3D object tracking, 3D occupancy prediction, BEV map segmentation), prediction (e.g., motion prediction), and planning (e.g., end-to-end planning). This unified paradigm naturally simplifies the design of multi-modal and multi-task autonomous driving systems while maintaining superior performance. Ultimately, we hope UniLION offers a fresh perspective on the development of 3D foundation models in autonomous driving. Code is available at https://github.com/happinesslz/UniLION

---

## 14. Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System

**论文链接:** [http://arxiv.org/abs/2511.00096v1](http://arxiv.org/abs/2511.00096v1)

**作者:** Shangyu Lou

**发布时间:** 2025-10-30

**备注:** Accepted to The 3rd ACM SIGSPATIAL International Workshop on Advances  in Urban AI (UrbanAI'25)

### GPT解析

### 总结

Urban-MAS是一个基于大语言模型的多智能体系统框架，通过三种专门设计的智能体提高城市预测的准确性和可靠性，在零样本设置下进行以人为中心的城市预测任务。

### 背景

Urban AI在以人为中心的城市任务方面取得进展，大语言模型虽能整合多模态输入处理城市异构数据，但在特定领域任务上表现欠佳。

### 目的

介绍Urban-MAS框架，用于在零样本设置下进行以人为中心的城市预测，解决单一大语言模型在城市特定任务上的局限性。

### 方法

Urban-MAS包含三种智能体：预测因素引导智能体（优先考虑关键预测因素指导知识提取）、可靠城市信息提取智能体（通过比较输出、验证一致性提高鲁棒性）、多城市信息推理智能体（跨维度整合多源信息进行预测）。

### 主要发现

在东京、米兰和西雅图的实验中，Urban-MAS相比单-LLM基线显著减少预测误差；消融研究表明预测因素引导智能体对提高预测性能最为关键。

### 结论

Urban-MAS被定位为可扩展的以人为中心的城市AI预测范式，代码已在GitHub开源。

### 翻译

城市人工智能（Urban AI）已推进了以人为中心的城市任务，如感知预测和人类动态。大语言模型可以整合多模态输入以处理复杂城市系统中的异构数据，但在特定领域任务上往往表现不佳。Urban-MAS是一个基于大语言模型的多智能体系统（MAS）框架，用于在零样本设置下进行以人为中心的城市预测。它包括三种智能体类型：预测因素引导智能体，优先考虑关键预测因素以指导知识提取，增强压缩城市知识在大语言模型中的有效性；可靠城市信息提取智能体，通过比较多个输出、验证一致性并在发生冲突时重新提取来提高鲁棒性；多城市信息推理智能体，跨维度整合提取的多源信息进行预测。在东京、米兰和西雅图的运行量预测和城市感知实验表明，Urban-MAS与单-LLM基线相比显著减少了误差。消融研究表明预测因素引导智能体对提高预测性能最为关键，使Urban-MAS成为以人为中心的城市AI预测的可扩展范式。代码可在项目网站获取：https://github.com/THETUREHOOHA/UrbanMAS


### 论文摘要

Urban Artificial Intelligence (Urban AI) has advanced human-centered urban tasks such as perception prediction and human dynamics. Large Language Models (LLMs) can integrate multimodal inputs to address heterogeneous data in complex urban systems but often underperform on domain-specific tasks. Urban-MAS, an LLM-based Multi-Agent System (MAS) framework, is introduced for human-centered urban prediction under zero-shot settings. It includes three agent types: Predictive Factor Guidance Agents, which prioritize key predictive factors to guide knowledge extraction and enhance the effectiveness of compressed urban knowledge in LLMs; Reliable UrbanInfo Extraction Agents, which improve robustness by comparing multiple outputs, validating consistency, and re-extracting when conflicts occur; and Multi-UrbanInfo Inference Agents, which integrate extracted multi-source information across dimensions for prediction. Experiments on running-amount prediction and urban perception across Tokyo, Milan, and Seattle demonstrate that Urban-MAS substantially reduces errors compared to single-LLM baselines. Ablation studies indicate that Predictive Factor Guidance Agents are most critical for enhancing predictive performance, positioning Urban-MAS as a scalable paradigm for human-centered urban AI prediction. Code is available on the project website:https://github.com/THETUREHOOHA/UrbanMAS

---

## 15. OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning

**论文链接:** [http://arxiv.org/abs/2511.02205v1](http://arxiv.org/abs/2511.02205v1)

**作者:** Kevin Valencia, Thilina Balasooriya, Xihaier Luo, Shinjae Yoo, David Keetae Park

**发布时间:** 2025-11-04

**备注:** 25 pages, 12 figures, 8 tables

### GPT解析

### 总结

OmniField是一种连续感知框架，能够处理真实世界实验数据的多模态时空学习挑战，通过学习基于可用模态条件化的连续神经场并迭代融合跨模态上下文，实现统一的重建、插值、预测和跨模态预测功能。

### 背景

真实世界实验数据的多模态时空学习面临两个主要挑战：单模态测量数据稀疏、不规则且带有噪声，但跨模态之间存在相关性；可用模态集合随空间和时间变化，导致可用记录减少。

### 目的

提出一种能够适应任意模态子集并处理稀疏、不规则、带噪声数据的框架，用于多模态时空学习。

### 方法

提出OmniField框架，该框架学习基于可用模态条件化的连续神经场，并通过多模态串扰块架构与迭代跨模态细化相结合，在解码器前对齐信号，无需网格化或代理预处理即可实现统一功能。

### 主要发现

OmniField在评估中始终优于八个强大的多模态时空基线；即使在严重的模拟传感器噪声下，性能仍接近清洁输入水平，显示出对损坏测量的强鲁棒性。

### 结论

OmniField是一种有效解决多模态时空学习中数据稀疏、不规则、带噪声以及模态集合变化等挑战的框架。

### 翻译

真实世界实验数据的多模态时空学习受两个挑战限制：单模态测量稀疏、不规则且带有噪声(QA/QC伪影)，但跨模态相关；可用模态集合随空间和时间变化，除非模型能够在训练和测试时适应任意子集，否则会缩小可用记录。我们提出了OmniField，一种连续感知框架，学习基于可用模态条件化的连续神经场，并迭代融合跨模态上下文。多模态串扰块架构与迭代跨模态细化相结合，在解码器前对齐信号，无需网格化或代理预处理即可实现统一的重建、插值、预测和跨模态预测。广泛评估显示，OmniField始终优于八个强大的多模态时空基线。在严重的模拟传感器噪声下，性能仍接近清洁输入水平，突显了对损坏测量的鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态时空学习中的两个关键挑战：1) 同一模态的测量数据稀疏、不规则且含噪声，但不同模态间相互关联；2) 可用模态集在空间和时间上变化，缩小可用记录规模。这些问题在气候科学、空气污染研究、材料科学等多个领域都至关重要，因为现有方法要么依赖数据预处理引入系统副作用，要么使用模型方法但假设固定观测算子，难以处理真实世界中的传感器数据不完整性和噪声问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：数据预处理引入平滑偏差和不确定性崩溃，而模型方法假设固定观测算子在实际情况中不成立。基于此，作者设计了一个连续感知的框架OmniField，它基于条件神经场(CNFs)原理，扩展了SCENT的工作。借鉴了多模态融合(MIA)、神经场表示(NeRF)和算子学习(FNO)等现有方法，但针对科学数据的特殊性进行了改进，特别是处理稀疏、不规则、噪声数据的能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是学习一个连续感知的多模态条件神经场，能够处理稀疏、不规则、噪声数据，并适应不同模态的可用性。整体流程采用编码器-处理器-解码器架构：1) 编码器将不规则观测转换为固定长度表示；2) 处理器融合坐标编码与上下文摘要，形成条件神经场；3) 解码器生成各模态预测。关键组件包括高斯傅里叶特征(GFF)和正弦初始化解决低频偏差，多模态串扰(MCT)块实现跨模态信息交换，迭代跨模态精炼(ICMR)渐进对齐信号，以及灵活模态 fusion处理模态缺失情况。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 连续感知的多模态条件神经场框架；2) 高斯傅里叶特征和正弦初始化解决低频偏差；3) 多模态串扰(MCT)块实现跨模态信息交换；4) 迭代跨模态精炼(ICMR)渐进信号对齐；5) 灵活模态融合处理模态缺失。相比之前工作，OmniField扩展了SCENT的多模态能力，不同于PROSE-FD的PDE算子学习，区别于MIA的双层优化方法，且无需网格化预处理，能更好处理科学数据中的稀疏性和噪声问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OmniField通过引入连续感知的多模态条件神经场、多模态串扰块和迭代跨模态精炼机制，有效解决了科学实验中多模态时空学习的稀疏性、不规则性和噪声挑战，实现了在无需网格化或代理预处理的情况下，对多种任务的鲁棒统一处理。'}


### 论文摘要

Multimodal spatiotemporal learning on real-world experimental data is constrained by two challenges: within-modality measurements are sparse, irregular, and noisy (QA/QC artifacts) but cross-modally correlated; the set of available modalities varies across space and time, shrinking the usable record unless models can adapt to arbitrary subsets at train and test time. We propose OmniField, a continuity-aware framework that learns a continuous neural field conditioned on available modalities and iteratively fuses cross-modal context. A multimodal crosstalk block architecture paired with iterative cross-modal refinement aligns signals prior to the decoder, enabling unified reconstruction, interpolation, forecasting, and cross-modal prediction without gridding or surrogate preprocessing. Extensive evaluations show that OmniField consistently outperforms eight strong multimodal spatiotemporal baselines. Under heavy simulated sensor noise, performance remains close to clean-input levels, highlighting robustness to corrupted measurements.

---

## 16. Bayesian full waveform inversion with learned prior using deep convolutional autoencoder

**论文链接:** [http://arxiv.org/abs/2511.02737v1](http://arxiv.org/abs/2511.02737v1)

**作者:** Shuhua Hu, Mrinal K Sen, Zeyu Zhao, Abdelrahman Elmeliegy, Shuo Zhang

**发布时间:** 2025-11-04

**备注:** 16 pages, 19 figures, 2 tables

### GPT解析

### 总结

本文提出了一种结合深度卷积自编码器和贝叶斯全波形反演的方法，通过降低模型维度和优化计算过程，实现了更高效的速度模型重建和不确定性评估。

### 背景

全波形反演(FWI)可以用贝叶斯框架表达，其中相关的不确定性由后验概率分布(PPD)捕获。然而，使用基于采样的方法如马尔可夫链蒙特卡洛(MCMC)解决贝叶斯FWI在计算上非常困难，因为模型空间的维度极高。

### 目的

为了缓解计算困难，作者开发了一种深度卷积自编码器(CAE)作为反演的学习先验，以提高计算效率。

### 方法

1) 使用CAE将详细的地下速度模型压缩为低维潜在表示；2) 采用自适应梯度MCMC算法，通过基于自动微分的FWI在潜在空间中高效计算梯度；3) 实现迁移学习策略，通过反演过程中的在线微调，使框架能够适应原始训练集中未表示的速度结构。

### 主要发现

使用合成数据的数值实验表明，与传统MCMC方法相比，该方法能以更高的效率重建速度模型并评估不确定性。

### 结论

结合深度学习和贝叶斯方法的创新框架有效解决了高维模型空间中的计算挑战，实现了更高效的全波形反演。

### 翻译

全波形反演(FWI)可以用贝叶斯框架表达，其中相关的不确定性由后验概率分布(PPD)捕获。实际上，使用基于采样的方法如马尔可夫链蒙特卡洛(MCMC)解决贝叶斯FWI在计算上非常困难，因为模型空间的维度极高。为了缓解这一困难，我们开发了一种深度卷积自编码器(CAE)，作为反演的学习先验。CAE将详细的地下速度模型压缩为低维潜在表示，实现了比传统降维方法更有效且地质一致性的模型简化。反演过程采用自适应梯度MCMC算法，通过基于自动微分的FWI在潜在空间中高效计算梯度。此外，我们通过反演过程中的在线微调实现了迁移学习策略，使框架能够适应原始训练集中未表示的速度结构。使用合成数据的数值实验表明，与传统MCMC方法相比，该方法能以更高的效率重建速度模型并评估不确定性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决贝叶斯全波形反演(Bayesian FWI)的计算效率问题。传统基于采样的贝叶斯方法(如MCMC)因模型空间维度极高而计算需求巨大，难以实际应用。这个问题很重要，因为贝叶斯框架能自然包含不确定性，提供更全面的地下结构成像结果，而传统方法只给出单一最优解，无法评估不确定性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到贝叶斯FWI面临维度灾难问题后，借鉴了深度学习中的自编码器技术，特别是卷积自编码器(CAE)的降维能力。他们结合了贝叶斯推理框架、MCMC采样方法、自动微分技术和迁移学习策略，提出在低维潜在空间中进行采样以提高效率。该方法确实借鉴了大量现有工作，包括贝叶斯理论基础、自编码器应用、MCMC方法和自动微分技术等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用深度卷积自编码器将高维地下速度模型压缩到低维潜在空间，在这个低维空间中进行贝叶斯MCMC采样，解决维度灾难问题。流程包括：1)训练阶段：收集地下速度模型数据，训练CAE；2)反演阶段：将初始模型映射到潜在空间，执行自适应梯度MCMC采样，使用自动微分计算梯度；3)对于分布外情况，先进行解码器在线微调，再进行采样；4)结果分析：收集后验样本，计算统计量和评估不确定性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用深度卷积自编码器作为学习先验，优于传统参数化方法；2)在潜在空间中进行自适应梯度MCMC采样，大幅降低计算复杂度；3)提出在线微调的迁移学习策略，解决分布外反演问题；4)使用正弦激活函数提高性能。相比之前工作，本文更紧密地结合自编码器与贝叶斯框架，提出完整流程，证明了CAE在保持地质特征方面的优势，并解决了分布外反演挑战。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合深度卷积自编码器与贝叶斯全波形反演的创新方法，通过在低维潜在空间中进行高效采样，显著提高了地下结构成像的计算效率，同时能够量化反演结果的不确定性。'}


### 论文摘要

Full waveform inversion (FWI) can be expressed in a Bayesian framework, where the associated uncertainties are captured by the posterior probability distribution (PPD). In practice, solving Bayesian FWI with sampling-based methods such as Markov chain Monte Carlo (MCMC) is computationally demanding because of the extremely high dimensionality of the model space. To alleviate this difficulty, we develop a deep convolutional autoencoder (CAE) that serves as a learned prior for the inversion. The CAE compresses detailed subsurface velocity models into a low-dimensional latent representation, achieving more effective and geologically consistent model reduction than conventional dimension reduction approaches. The inversion procedure employs an adaptive gradient-based MCMC algorithm enhanced by automatic differentiation-based FWI to compute gradients efficiently in the latent space. In addition, we implement a transfer learning strategy through online fine-tuning during inversion, enabling the framework to adapt to velocity structures not represented in the original training set. Numerical experiments with synthetic data show that the method can reconstruct velocity models and assess uncertainty with improved efficiency compared to traditional MCMC methods.

---

## 17. Unsupervised Learning for Industrial Defect Detection: A Case Study on Shearographic Data

**论文链接:** [http://arxiv.org/abs/2511.02541v1](http://arxiv.org/abs/2511.02541v1)

**作者:** Jessica Plassmann, Nicolas Schuler, Georg von Freymann, Michael Schuth

**发布时间:** 2025-11-04

**备注:** 15 pages, 6 figures, 1 table; accepted for AI-2025 Forty-fifth SGAI  International Conference on Artificial Intelligence CAMBRIDGE, ENGLAND 16-18  DECEMBER 2025

### GPT解析

### 总结

本研究探索了无监督学习方法在剪切散斑图像自动异常检测中的应用，比较了三种深度学习模型，发现师生特征匹配方法在分类鲁棒性和缺陷定位方面表现最佳，为工业环境中的高效无损检测提供了新思路。

### 背景

剪切散斑测量是一种高灵敏度、全场检测能力的无损检测方法，可用于检测表面下缺陷。然而，由于其需要专家解读，在工业应用中的推广受到限制。

### 目的

减少对标记数据和人工评估的依赖，探索无监督学习方法用于剪切散斑图像中的自动异常检测，提高工业应用中的检测效率和可扩展性。

### 方法

评估了三种架构：全连接自编码器、卷积自编码器和师生特征匹配模型。所有模型仅使用无缺陷数据进行训练。开发了一个使用具有可重复缺陷模式的自定义试样的受控数据集，定义了两个训练子集：一个只包含无畸变、无缺陷样本，另一个额外包含全局变形但无缺陷的数据。评估包括二元分类和空间缺陷定位。

### 主要发现

师生特征匹配方法实现了卓越的分类鲁棒性和精确的定位能力。与自编码器模型相比，它表现出更好的特征表示可分性，通过t-SNE嵌入可视化。使用YOLOv8作为参考基准验证了定位质量。

### 结论

无监督深度学习在工业环境中具有可扩展性和标签效率，为剪切散斑检测提供了有前景的解决方案，特别是在减少对专家解读的依赖方面。

### 翻译

剪切散斑测量是一种用于检测表面下缺陷的无损检测方法，具有高灵敏性和全场检测能力。然而，由于其需要专家解读，在工业应用中的推广仍然有限。为了减少对标记数据和人工评估的依赖，本研究探索了无监督学习方法用于剪切散斑图像中的自动异常检测。评估了三种架构：全连接自编码器、卷积自编码器和师生特征匹配模型。所有模型仅使用无缺陷数据进行训练。开发了一个使用具有可重复缺陷模式的自定义试样的受控数据集，实现了在理想和现实变形条件下系统获取剪切散斑测量。定义了两个训练子集：一个只包含无畸变、无缺陷样本，另一个额外包含全局变形但无缺陷的数据。后者通过包含可能掩盖局部异常的变形引起的条纹图案来模拟实际检测条件。从二元分类和师生模型的缺陷定位方面评估了模型。结果表明，师生方法实现了卓越的分类鲁棒性和精确的定位能力。与基于自编码器的模型相比，它表现出更好的特征表示可分性，通过t-SNE嵌入可视化。此外，在标记缺陷数据上训练的YOLOv8模型作为定位质量的参考基准。这项研究强调了无监督深度学习在工业环境中可扩展、标签高效的剪切散斑检测的潜力。


### 论文摘要

Shearography is a non-destructive testing method for detecting subsurface defects, offering high sensitivity and full-field inspection capabilities. However, its industrial adoption remains limited due to the need for expert interpretation. To reduce reliance on labeled data and manual evaluation, this study explores unsupervised learning methods for automated anomaly detection in shearographic images. Three architectures are evaluated: a fully connected autoencoder, a convolutional autoencoder, and a student-teacher feature matching model. All models are trained solely on defect-free data. A controlled dataset was developed using a custom specimen with reproducible defect patterns, enabling systematic acquisition of shearographic measurements under both ideal and realistic deformation conditions. Two training subsets were defined: one containing only undistorted, defect-free samples, and one additionally including globally deformed, yet defect-free, data. The latter simulates practical inspection conditions by incorporating deformation-induced fringe patterns that may obscure localized anomalies. The models are evaluated in terms of binary classification and, for the student-teacher model, spatial defect localization. Results show that the student-teacher approach achieves superior classification robustness and enables precise localization. Compared to the autoencoder-based models, it demonstrates improved separability of feature representations, as visualized through t-SNE embeddings. Additionally, a YOLOv8 model trained on labeled defect data serves as a reference to benchmark localization quality. This study underscores the potential of unsupervised deep learning for scalable, label-efficient shearographic inspection in industrial environments.

---

## 18. Enhancing Phenotype Discovery in Electronic Health Records through Prior Knowledge-Guided Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2511.02102v1](http://arxiv.org/abs/2511.02102v1)

**作者:** Melanie Mayer, Kimberly Lactaoen, Gary E. Weissman, Blanca E. Himes, Rebecca A. Hubbard

**发布时间:** 2025-11-03

**备注:** Submitted to JAMIA; preprint is the author's original version. Github  repo: https://github.com/mm4963/prior-guided-EHR-phenotyping/tree/main

### GPT解析

### 总结

本研究开发了一种结合领域特定知识的贝叶斯潜在类别框架，用于改进基于电子健康记录的无监督学习表型发现方法。通过信息先验引导聚类向临床相关亚组发展，在哮喘患者数据中成功识别出与2型炎症特征相关的'T2高炎症'亚型。

### 背景

传统的基于电子健康记录的无监督学习方法在表型发现方面显示出前景，但这些方法通常忽略了现有的临床信息，限制了结果的可解释性。在缺乏明确表型定义的异质性疾病研究中，需要能够整合临床知识的方法。

### 目的

旨在通过整合领域特定知识来提高EHR衍生表型的临床意义，并展示该方法在识别与2型炎症特征相关的哮喘亚型方面的效用。

### 方法

开发了一个框架，通过信息先验将临床知识整合到贝叶斯潜在类别模型中，引导无监督聚类向临床相关的亚组方向发展。该方法能够建模缺失值，考虑潜在的随机缺失模式，并提供患者级别的表型分配概率及其不确定性。研究者在包含44,642名成年哮喘患者的大型哮喘EHR队列中应用了该模型，为2型炎症相关特征指定了信息先验，为其他临床变量指定了弱信息先验。

### 主要发现

使用2017年1月至2024年2月的就诊数据，研究者发现了表型分配的双峰后验分布，表明存在明显的类别分离。T2炎症信息类(38.7%)的特征是嗜酸性粒细胞水平和过敏标志物升高，以及高医疗利用率和药物使用，尽管后者的变量只有弱信息先验。这些模式表明存在一个'未控制的T2高'亚型。

### 结论

贝叶斯潜在类别建模方法支持在缺乏明确表型定义的异质性疾病研究中进行假设生成和队列识别，展示了如何通过整合临床知识提高EHR数据分析的临床相关性。

### 翻译

目标：利用电子健康记录数据进行无监督学习在表型发现方面显示出前景，但通常方法忽略了现有的临床信息，限制了可解释性。我们将贝叶斯潜在类别框架具体化用于表型分析，整合领域特定知识以提高EHR衍生表型的临床意义，并通过识别与2型炎症特征相关的哮喘亚型来说明其效用。材料与方法：我们通过信息先验将临床知识整合到贝叶斯潜在类别模型中，引导无监督聚类向临床相关的亚组方向发展。该方法对缺失值进行建模，考虑潜在的随机缺失模式，并提供患者级别的表型分配概率及其不确定性。使用可重用且灵活的代码，我们将该模型应用于大型哮喘EHR队列，为T2炎症相关特征指定信息先验，为其他临床变量指定弱信息先验，让数据 informing 后验分布。结果与结论：使用2017年1月至2024年2月间44,642名成年哮喘患者的就诊数据，我们发现表型分配的后验分布呈双峰，表明存在明显的类别分离。T2炎症信息类(38.7%)的特征是嗜酸性粒细胞水平和过敏标志物升高，以及高医疗利用率和药物使用，尽管后者的变量只有弱信息先验。这些模式表明存在一个'未控制的T2高'亚型。这展示了我们的贝叶斯潜在类别建模方法如何支持在缺乏明确表型定义的异质性疾病EHR研究中进行假设生成和队列识别。


### 论文摘要

Objectives: Unsupervised learning with electronic health record (EHR) data has shown promise for phenotype discovery, but approaches typically disregard existing clinical information, limiting interpretability. We operationalize a Bayesian latent class framework for phenotyping that incorporates domain-specific knowledge to improve clinical meaningfulness of EHR-derived phenotypes and illustrate its utility by identifying an asthma sub-phenotype informed by features of Type 2 (T2) inflammation.   Materials and methods: We illustrate a framework for incorporating clinical knowledge into a Bayesian latent class model via informative priors to guide unsupervised clustering toward clinically relevant subgroups. This approach models missingness, accounting for potential missing-not-at-random patterns, and provides patient-level probabilities for phenotype assignment with uncertainty. Using reusable and flexible code, we applied the model to a large asthma EHR cohort, specifying informative priors for T2 inflammation-related features and weakly informative priors for other clinical variables, allowing the data to inform posterior distributions.   Results and Conclusion: Using encounter data from January 2017 to February 2024 for 44,642 adult asthma patients, we found a bimodal posterior distribution of phenotype assignment, indicating clear class separation. The T2 inflammation-informed class (38.7%) was characterized by elevated eosinophil levels and allergy markers, plus high healthcare utilization and medication use, despite weakly informative priors on the latter variables. These patterns suggest an "uncontrolled T2-high" sub-phenotype. This demonstrates how our Bayesian latent class modeling approach supports hypothesis generation and cohort identification in EHR-based studies of heterogeneous diseases without well-established phenotype definitions.

---

## 19. Machine and Deep Learning for Indoor UWB Jammer Localization

**论文链接:** [http://arxiv.org/abs/2511.01819v1](http://arxiv.org/abs/2511.01819v1)

**作者:** Hamed Fard, Mahsa Kholghi, Benedikt Groß, Gerhard Wunder

**发布时间:** 2025-11-03

**备注:** Accepted at the 20th International Conference on Risks and Security  of Internet and Systems (CRiSIS 2025, Gatineau-Canada,  https://crisis2025.uqo.ca/). The paper will soon be published as  post-proceedings in Springer's LNCS

### GPT解析

### 总结

该研究解决了超宽带(UWB)定位系统易受干扰攻击的问题，提出了一种域对抗ConvNeXt自编码器(A-CNT)方法，能够在室内布局变化的情况下实现鲁棒的干扰器定位。

### 背景

超宽带(UWB)定位能提供厘米级精度，但容易受到干扰攻击，对智能建筑中的资产跟踪和入侵检测构成安全风险。尽管机器学习和深度学习方法已改进标签定位，但在单个房间内定位恶意干扰器以及应对变化的室内布局方面研究不足。

### 目的

研究如何在室内布局变化的情况下实现鲁棒的干扰器定位，解决域偏移问题。

### 方法

引入两个新的UWB数据集（原始和修改后的房间配置），建立全面的ML/DL基线，使用多种分类和回归指标评估性能。提出域对抗ConvNeXt自编码器(A-CNT)，利用梯度反转层对齐跨域的CIR衍生特征。

### 主要发现

在源数据集上，随机森林达到最高的F1-macro分数0.95，XGBoost达到最低的平均欧几里得误差20.16厘米。但在修改后的房间布局中，XGBoost的平均误差增加十倍至207.99厘米。A-CNT框架将平均误差降低到34.67厘米，比非对抗性迁移学习提高77%，比最佳基线提高83%，使30厘米内的样本比例恢复到0.56。

### 结论

对抗特征对齐使得尽管环境变化，室内干扰器定位能够保持鲁棒性和可转移性。

### 翻译

超宽带(UWB)定位提供厘米级精度但容易受到干扰攻击，对智能建筑中的资产跟踪和入侵检测构成安全风险。虽然机器学习(ML)和深度学习(DL)方法已改进标签定位，但在单个房间内定位恶意干扰器以及应对变化的室内布局方面 largely unexplored。研究引入两个新的UWB数据集，分别在原始和修改后的房间配置下收集，建立全面的ML/DL基线。使用多种分类和回归指标严格评估性能。在源数据集上，随机森林达到最高的F1-macro分数0.95，XGBoost达到最低的平均欧几里得误差20.16厘米。但在修改后的房间布局中部署源训练的模型导致性能严重下降，XGBoost的平均误差增加十倍至207.99厘米，显示出显著的域偏移。为缓解这种退化，提出域对抗ConvNeXt自编码器(A-CNT)，利用梯度反转层对齐跨域的CIR衍生特征。A-CNT框架通过将平均误差降低到34.67厘米恢复定位性能，比非对抗性迁移学习提高77%，比最佳基线提高83%，使30厘米内的样本比例恢复到0.56。总体结果表明，尽管环境变化，对抗特征对齐 enables robust and transferable indoor jammer localization。代码和数据集可在 https://github.com/afbf4c8996f/Jammer-Loc 获取。


### 论文摘要

Ultra-wideband (UWB) localization delivers centimeter-scale accuracy but is vulnerable to jamming attacks, creating security risks for asset tracking and intrusion detection in smart buildings. Although machine learning (ML) and deep learning (DL) methods have improved tag localization, localizing malicious jammers within a single room and across changing indoor layouts remains largely unexplored. Two novel UWB datasets, collected under original and modified room configurations, are introduced to establish comprehensive ML/DL baselines. Performance is rigorously evaluated using a variety of classification and regression metrics. On the source dataset with the collected UWB features, Random Forest achieves the highest F1-macro score of 0.95 and XGBoost achieves the lowest mean Euclidean error of 20.16 cm. However, deploying these source-trained models in the modified room layout led to severe performance degradation, with XGBoost's mean Euclidean error increasing tenfold to 207.99 cm, demonstrating significant domain shift. To mitigate this degradation, a domain-adversarial ConvNeXt autoencoder (A-CNT) is proposed that leverages a gradient-reversal layer to align CIR-derived features across domains. The A-CNT framework restores localization performance by reducing the mean Euclidean error to 34.67 cm. This represents a 77 percent improvement over non-adversarial transfer learning and an 83 percent improvement over the best baseline, restoring the fraction of samples within 30 cm to 0.56. Overall, the results demonstrate that adversarial feature alignment enables robust and transferable indoor jammer localization despite environmental changes. Code and dataset available at https://github.com/afbf4c8996f/Jammer-Loc

---

## 20. An Open-Access Benchmark of Statistical and Machine-Learning Anomaly Detection Methods for Battery Applications

**论文链接:** [http://arxiv.org/abs/2511.01745v1](http://arxiv.org/abs/2511.01745v1)

**作者:** Mei-Chin Pang, Suraj Adhikari, Takuma Kasahara, Nagihiro Haba, Saneyuki Ohno

**发布时间:** 2025-11-03

### GPT解析

### 总结

OSBAD是一个开源基准测试平台，用于电池应用中的异常检测框架，通过测试15种不同算法实现异常检测方法的系统性比较，提出的特征转换工作流程和贝叶斯优化管道提高了异常检测性能，具有跨化学泛化能力，为电池安全分析提供了重要工具和方法。

### 背景

电池安全在从消费电子产品到电动汽车和飞机的各种应用中至关重要，未被发现的异常可能引发安全隐患或导致昂贵的停机时间。

### 目的

提出一个名为OSBAD的开源基准测试平台，用于电池应用中的异常检测框架，实现不同数据集上异常检测方法的系统性比较。

### 方法

对15种不同的算法进行基准测试，包括统计方法、基于距离的方法和无监督机器学习方法；展示基于物理和统计信息的特征转换工作流程，通过将集体异常分解为点异常提高异常可分离性；提出贝叶斯优化管道，基于迁移学习和回归代理实现自动超参数调优。

### 主要发现

通过涵盖液态和固态化学成分的数据集验证，证明了OSBAD的跨化学泛化能力，能够识别不同电化学系统中的异常情况；物理和统计信息驱动的特征工程以及概率超参数调优的模型选择对推进关键能源系统的可信数据驱动诊断具有重要意义。

### 结论

通过向社区提供开源可复现的异常检测工作流程的基准测试数据库，OSBAD为开发安全、可扩展和可转移的电池分析异常检测工具建立了统一基础；强调了物理和统计信息驱动的特征工程以及概率超参数调优的模型选择在推进关键能源系统可信数据驱动诊断中的重要性。

### 翻译

电池安全在从消费电子产品到电动汽车和飞机的各种应用中至关重要，未被发现的异常可能引发安全隐患或导致昂贵的停机时间。在本研究中，我们提出了OSBAD作为电池应用中异常检测框架的开源基准。通过对涵盖统计、基于距离和无监督机器学习方法在内的15种不同算法进行基准测试，OSBAD实现了跨异构数据集异常检测方法的系统性比较。此外，我们展示了如何通过基于物理和统计信息的特征转换工作流程，通过将集体异常分解为点异常来提高异常可分离性。为解决无监督异常检测中因标签不完整导致的主要瓶颈，我们提出了一种基于迁移学习和回归代理的贝叶斯优化管道，实现自动超参数调优。通过在涵盖液态和固态化学成分的数据集上进行验证，我们进一步证明了OSBAD的跨化学泛化能力，能够识别不同电化学系统中的不规则性。通过向社区提供包含开源可复现异常检测工作流程的基准测试数据库，OSBAD为开发电池分析中安全、可扩展和可转移的异常检测工具建立了统一基础。这项研究强调了物理和统计信息驱动的特征工程以及概率超参数调优的模型选择在推进关键能源系统可信数据驱动诊断中的重要性。


### 论文摘要

Battery safety is critical in applications ranging from consumer electronics to electric vehicles and aircraft, where undetected anomalies could trigger safety hazards or costly downtime. In this study, we present OSBAD as an open-source benchmark for anomaly detection frameworks in battery applications. By benchmarking 15 diverse algorithms encompassing statistical, distance-based, and unsupervised machine-learning methods, OSBAD enables a systematic comparison of anomaly detection methods across heterogeneous datasets. In addition, we demonstrate how a physics- and statistics-informed feature transformation workflow enhances anomaly separability by decomposing collective anomalies into point anomalies. To address a major bottleneck in unsupervised anomaly detection due to incomplete labels, we propose a Bayesian optimization pipeline that facilitates automated hyperparameter tuning based on transfer-learning and regression proxies. Through validation on datasets covering both liquid and solid-state chemistries, we further demonstrate the cross-chemistry generalization capability of OSBAD to identify irregularities across different electrochemical systems. By making benchmarking database with open-source reproducible anomaly detection workflows available to the community, OSBAD establishes a unified foundation for developing safe, scalable, and transferable anomaly detection tools in battery analytics. This research underscores the significance of physics- and statistics-informed feature engineering as well as model selection with probabilistic hyperparameter tuning, in advancing trustworthy, data-driven diagnostics for safety-critical energy systems.

---

## 21. Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning

**论文链接:** [http://arxiv.org/abs/2511.01502v1](http://arxiv.org/abs/2511.01502v1)

**作者:** Mengtan Zhang, Zizhan Guo, Hongbo Zhao, Yi Feng, Zuyi Xiong, Yue Wang, Shaoyi Du, Hanli Wang, Rui Fan

**发布时间:** 2025-11-03

**备注:** 18 pages, 14 figures

### GPT解析

### 总结

本文提出了一种名为DiMoDE的深度和自我运动联合学习框架，通过区分性处理运动组件，利用各自刚性流的几何规律来改善深度和自我运动估计，在多个数据集上取得了最先进的表现。

### 背景

无监督学习深度和自我运动这两个基础3D感知任务近年来取得了显著进展。然而，大多数方法将自我运动视为辅助任务，要么混合所有运动类型，要么在监督中排除与深度无关的旋转运动，限制了强几何约束的融入，降低了在多样化条件下的可靠性和鲁棒性。

### 目的

引入对运动组件的区分性处理，利用各自刚性流的几何规律来改善深度和自我运动估计，提高系统在多样化条件下的可靠性和鲁棒性。

### 方法

给定连续视频帧，网络首先对齐源相机和目标相机的光轴和成像平面。帧之间的光流通过这些对齐进行变换，并量化偏差以对每个自我运动组件单独施加几何约束。这些对齐进一步将联合学习过程重新表述为共轴和共面形式，其中深度和每个平移分量可以通过闭合形式的几何关系相互推导。

### 主要发现

通过区分性处理运动组件并利用各自刚性流的几何规律，DiMoDE框架在多个公共数据集和新收集的多样化真实世界数据集上实现了最先进的表现，特别是在具有挑战性的条件下表现优异。

### 结论

DiMoDE是一种通用的深度和自我运动联合学习框架，通过区分性处理运动组件和利用几何约束，显著提高了深度和自我运动估计的准确性和鲁棒性，特别是在具有挑战性的条件下。

### 翻译

无监督学习深度和自我运动这两个基础3D感知任务近年来取得了显著进展。然而，大多数方法将自我运动视为辅助任务，要么混合所有运动类型，要么在监督中排除与深度无关的旋转运动。这种设计限制了强几何约束的融入，降低了在多样化条件下的可靠性和鲁棒性。本研究引入了对运动组件的区分性处理，利用各自刚性流的几何规律来改善深度和自我运动估计。给定连续视频帧，网络首先对齐源相机和目标相机的光轴和成像平面。帧之间的光流通过这些对齐进行变换，并量化偏差以对每个自我运动组件单独施加几何约束，实现更精细的优化。这些对齐进一步将联合学习过程重新表述为共轴和共面形式，其中深度和每个平移分量可以通过闭合形式的几何关系相互推导，引入互补约束以提高深度鲁棒性。DiMoDE是一种融合了这些设计的通用深度和自我运动联合学习框架，在多个公共数据集和新收集的多样化真实世界数据集上取得了最先进的表现，特别是在具有挑战性的条件下。我们的源代码将在发表后于mias.group/DiMoDE公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决无监督学习深度和自运动估计中运动组件处理不当的问题。现有方法要么不加区分地混合所有运动类型，要么排除旋转运动，阻碍了强几何约束的融入，限制了模型在复杂环境下的可靠性。这个问题很重要，因为深度和自运动估计是3D感知的基础，在自动驾驶、机器人导航等领域有广泛应用，而更准确鲁棒的估计能提高系统在各种实际环境中的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析不同运动类型（旋转、切向平移、径向平移）产生的刚性流动差异，发现旋转产生不规则深度无关流动，而平移产生规则但不同的深度相关流动。现有方法混合这些流动导致监督信号质量下降。作者借鉴了现有无监督学习方法（如SfMLearner、Monodepth2）使用光流作为监督信号的思想，以及相机校准和几何变换原理，但创新性地应用于区分不同运动组件的处理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是区分性处理运动组件，利用每种组件固有的几何规律性：旋转产生不规则流动，切向平移产生平行流动，径向平移产生朝向/远离主点的流动。实现流程包括：1)将自运动分解为旋转、切向平移和径向平移；2)用旋转对齐光轴消除不规则流动；3)用平移对齐成像平面生成规则流动；4)施加双重几何约束（优化PoseNet和DepthNet）；5)整合所有约束进行联合训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次明确区分三种运动组件并分别处理；2)通过光轴和成像平面对齐将学习重新表述为同轴和共面形式；3)引入双重几何约束分别优化PoseNet和DepthNet；4)提出兼容多种架构的DiMoDE统一框架。相比之前工作，本文不混合处理所有运动类型或完全排除旋转，而是区分切向和径向平移，引入更高层次的几何约束而非仅依赖像素级光度一致性，显著提高了模型在复杂环境下的鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过区分性处理运动组件并引入双重几何约束，提出了一种改进的无监督深度和自运动联合学习框架DiMoDE，显著提高了模型在不同环境条件下的准确性和鲁棒性。'}


### 论文摘要

Unsupervised learning of depth and ego-motion, two fundamental 3D perception tasks, has made significant strides in recent years. However, most methods treat ego-motion as an auxiliary task, either mixing all motion types or excluding depth-independent rotational motions in supervision. Such designs limit the incorporation of strong geometric constraints, reducing reliability and robustness under diverse conditions. This study introduces a discriminative treatment of motion components, leveraging the geometric regularities of their respective rigid flows to benefit both depth and ego-motion estimation. Given consecutive video frames, network outputs first align the optical axes and imaging planes of the source and target cameras. Optical flows between frames are transformed through these alignments, and deviations are quantified to impose geometric constraints individually on each ego-motion component, enabling more targeted refinement. These alignments further reformulate the joint learning process into coaxial and coplanar forms, where depth and each translation component can be mutually derived through closed-form geometric relationships, introducing complementary constraints that improve depth robustness. DiMoDE, a general depth and ego-motion joint learning framework incorporating these designs, achieves state-of-the-art performance on multiple public datasets and a newly collected diverse real-world dataset, particularly under challenging conditions. Our source code will be publicly available at mias.group/DiMoDE upon publication.

---

## 22. A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation

**论文链接:** [http://arxiv.org/abs/2511.01404v1](http://arxiv.org/abs/2511.01404v1)

**作者:** Xiaoyu Liu, Yiqing Wu, Ruidong Han, Fuzhen Zhuang, Xiang Li, Wei Lin

**发布时间:** 2025-11-03

**备注:** Accepted by CIKM'25

### GPT解析

### 总结

该研究针对多领域推荐系统中的数据不平衡问题提出了解决方案，通过软分区半监督协同迁移学习(SSCTL)方法显著提升了推荐效果。

### 背景

多领域推荐(MDR)在工业实践中至关重要，共享-特定架构被广泛用于捕获共享和独特属性，但不同领域间数据不平衡导致模型性能问题。

### 目的

解决多领域推荐中因数据不平衡导致的两个关键问题：主导领域数据过多导致模型偏向，以及非主导领域数据稀疏导致过拟合。

### 方法

提出软分区半监督协同迁移学习(SSCTL)方法，通过生成动态参数解决主导问题，利用主导领域实例的带权伪标签增强非主导领域数据以对抗过拟合。

### 主要发现

在线实验表明，该方法在各个领域均取得显著改进，GMV增长0.54%至2.90%，CTR提升0.22%至1.69%。

### 结论

SSCTL方法有效解决了多领域推荐中的数据不平衡问题，提升了非主导领域的推荐性能，具有实际应用价值。

### 翻译

在工业实践中，多领域推荐(MDR)起着关键作用。共享-特定架构在工业解决方案中被广泛使用，通过共享和特定参数来捕获共享和独特属性。然而，由于不同领域间数据不平衡，这些模型面临两个关键问题：(1)主导问题：主导领域的数据使模型性能偏向，忽略了非主导领域。(2)过拟合问题：非主导领域的数据稀疏导致特定参数过拟合。为应对这些挑战，我们提出了用于多领域推荐的软分区半监督协同迁移学习(SSCTL)。SSCTL生成动态参数来解决主导问题，从而将重点转向非主导领域的样本。为了对抗过拟合，它利用主导领域实例的带权伪标签来增强非主导领域数据。我们进行了全面的在线和离线实验来验证所提出方法的有效性。在线测试在各种领域都取得了显著改进，GMV增长了0.54%至2.90%，CTR提升了0.22%至1.69%。


### 论文摘要

In industrial practice, Multi-domain Recommendation (MDR) plays a crucial role. Shared-specific architectures are widely used in industrial solutions to capture shared and unique attributes via shared and specific parameters. However, with imbalanced data across different domains, these models face two key issues: (1) Overwhelming: Dominant domain data skews model performance, neglecting non-dominant domains. (2) Overfitting: Sparse data in non-dominant domains leads to overfitting in specific parameters. To tackle these challenges, we propose Soft-partitioned Semi-supervised Collaborative Transfer Learning (SSCTL) for multi-domain recommendation. SSCTL generates dynamic parameters to address the overwhelming issue, thus shifting focus towards samples from non-dominant domains. To combat overfitting, it leverages pseudo-labels with weights from dominant domain instances to enhance non-dominant domain data. We conduct comprehensive experiments, both online and offline, to validate the efficacy of our proposed method. Online tests yielded significant improvements across various domains, with increases in GMV ranging from 0.54% to 2.90% and enhancements in CTR ranging from 0.22% to 1.69%.

---

## 23. Embodiment Transfer Learning for Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2511.01224v1](http://arxiv.org/abs/2511.01224v1)

**作者:** Chengmeng Li, Yaxin Peng

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究提出了一种名为ET-VLA的新型框架，通过合成继续预训练(SCP)和具身思维图技术，解决了VLA模型在多机器人协作方面的挑战，显著提升了模型在多embodiment环境中的性能。

### 背景

Vision-language-action (VLA)模型已在机器人学习领域取得显著进展，能够在大规模、跨embodiment数据上训练并针对特定机器人进行微调。然而，最先进的自回归VLA模型在多机器人协作方面表现不佳。

### 目的

开发一种高效有效的框架，将预训练的VLA模型转移到多机器人系统，解决多机器人协作问题。

### 方法

ET-VLA框架的核心是合成继续预训练(SCP)，使用合成生成的数据使模型适应新的embodiment，避免真实人类演示，降低数据收集成本。SCP使模型学习正确动作和精确动作令牌数量，随后在目标embodiment数据上微调。此外，提出具身思维图技术，将子任务表述为节点，使VLA模型在任务执行中区分各embodiment的功能和角色。

### 主要发现

在模拟基准测试和三种不同双臂机器人的真实机器人上验证了方法有效性。ET-VLA在六个真实世界任务上的表现比OpenVLA高出53.2%。

### 结论

将开源所有代码，支持社区推进用于机器人学习的VLA模型发展。

### 翻译

视觉-语言-动作(VLA)模型显著推进了机器人学习，使能够在大规模、跨embodiment数据上训练并为特定机器人进行微调。然而，最先进的自回归VLA模型在多机器人协作方面存在困难。我们引入了embodiment迁移学习，称为ET-VLA，这是一种新型框架，用于高效有效地将预训练的VLA模型转移到多机器人系统。ET-VLA的核心是合成继续预训练(SCP)，它使用合成的生成数据来使模型适应新的embodiment，绕过对真实人类演示的需求，降低数据收集成本。SCP使模型能够学习正确的动作和精确的动作令牌数量。在SCP之后，模型在目标embodiment数据上进行微调。为了进一步提高模型在多embodiment上的性能，我们提出了具身思维图技术，一种新颖的方法，将每个子任务表述为一个节点，使VLA模型能够在任务执行过程中区分每个embodiment的功能和角色。我们的研究考虑了双臂机器人，作为多机器人的一个简单版本来验证我们的方法。我们在模拟基准测试和覆盖三种不同双臂embodiment的真实机器人上验证了我们方法的有效性。特别是，我们提出的ET-VLA在六个真实世界任务上可以比OpenVLA高出53.2%。我们将开源所有代码，支持社区推进用于机器人学习的VLA模型。


### 论文摘要

Vision-language-action (VLA) models have significantly advanced robotic learning, enabling training on large-scale, cross-embodiment data and fine-tuning for specific robots. However, state-of-the-art autoregressive VLAs struggle with multi-robot collaboration. We introduce embodiment transfer learning, denoted as ET-VLA, a novel framework for efficient and effective transfer of pre-trained VLAs to multi-robot. ET-VLA's core is Synthetic Continued Pretraining (SCP), which uses synthetically generated data to warm up the model for the new embodiment, bypassing the need for real human demonstrations and reducing data collection costs. SCP enables the model to learn correct actions and precise action token numbers. Following SCP, the model is fine-tuned on target embodiment data. To further enhance the model performance on multi-embodiment, we present the Embodied Graph-of-Thought technique, a novel approach that formulates each sub-task as a node, that allows the VLA model to distinguish the functionalities and roles of each embodiment during task execution. Our work considers bimanual robots, a simple version of multi-robot to verify our approaches. We validate the effectiveness of our method on both simulation benchmarks and real robots covering three different bimanual embodiments. In particular, our proposed ET-VLA \space can outperform OpenVLA on six real-world tasks over 53.2%. We will open-source all codes to support the community in advancing VLA models for robot learning.

---

## 24. STELLAR-koff: A Transfer Learning Model for Protein-Ligand Dissociation Rate Constant Prediction Based on Interaction Landscape

**论文链接:** [http://arxiv.org/abs/2511.01171v1](http://arxiv.org/abs/2511.01171v1)

**作者:** Jingyuan Li

**发布时间:** 2025-11-03

### GPT解析

### 总结

该研究开发了一种名为STELLAR-koff的图神经网络模型，用于预测蛋白质-配体解离速率常数，通过迁移学习将多个配体构象转化为相互作用景观，扩展了数据集，并在测试中表现出优越性能。

### 背景

成功的药物设计关键在于正确理解蛋白质-配体相互作用，目前已有许多预测热力学性质的深度学习模型，但缺乏成熟的预测动力学性质的模型，主要原因是缺乏动力学数据。

### 目的

开发一个预测蛋白质-配体解离速率常数的模型，解决缺乏动力学数据的问题。

### 方法

开发名为STELLAR-koff的图神经网络模型，使用迁移学习将蛋白质内多个配体构象转化为蛋白质-配体相互作用景观，扩展PDBbind koff数据集从680到1197个条目，通过五折交叉验证和外部集测试模型性能。

### 主要发现

STELLAR-koff在五折交叉验证中达到0.729的皮尔逊相关系数，性能超越或与大多数已发表的预测方法相当；在外部集上对未见蛋白质的预测表现出强大性能，特别是在粘着斑激酶上达到0.838的皮尔逊相关系数；在周期依赖性激酶上的实验验证证明了其在真实药物发现场景中的有效性。

### 结论

该研究为预测蛋白质-配体解离速率常数提供了有效工具，为该领域的未来发展提供了新的见解。

### 翻译

成功药物设计的关键在于正确理解蛋白质-配体相互作用。在当前知识框架下，这些相互作用可以从热力学和动力学两个角度进行描述。近年来，许多深度学习模型 emerged 用于预测蛋白质-配体相互作用的热力学性质。然而，目前缺乏成熟的预测动力学性质的模型，主要原因是缺乏动力学数据。为解决这个问题，我们开发了一个名为STELLAR-koff（基于结构的迁移学习用于配体活性回归）的图神经网络模型来预测蛋白质-配体解离速率常数。与传统的蛋白质-配体性质预测模型不同，后者通常使用单一复合物构象作为输入，STELLAR-koff采用迁移学习将蛋白质内多个配体构象转化为蛋白质-配体相互作用景观，并将这种景观作为模型的主要输入。此外，我们将PDBbind koff数据集从680个扩展到1197个条目，并使用增强的数据集进行模型训练和测试。通过五折交叉验证测试时，STELLAR-koff达到0.729的皮尔逊相关系数，性能超越或与大多数已发表的预测方法相当。在外部集测试中，STELLAR-koff在对未见蛋白质的预测中表现出强大性能，特别是在粘着斑激酶上达到了0.838的皮尔逊相关系数。在周期依赖性激酶上的实验验证也证明了STELLAR-koff在真实药物发现场景中的有效性。我们相信这项研究为预测蛋白质-配体解离速率常数提供了有效工具，并为该领域的未来发展提供了新的见解。


### 论文摘要

The key to successful drug design lies in the correct comprehension of protein-ligand interactions. Within the current knowledge paragm, these interactions can be described from both thermodynamic and kinetic perspectives. In recent years, many deep learning models have emerged for predicting the thermodynamic properties of protein-ligand interactions. However, there is currently no mature model for predicting kinetic properties, primarily due to lack of kinetic data. To tackle this problem, we have developed a graph neural network model called STELLAR-koff (Structure-based TransfEr Learning for Ligand Activity Regression) to predict protein-ligand dissociation rate constant. Unlike traditional protein-ligand property prediction models, which typically use a single complex conformation as input, STELLAR-koff employs transfer learning to transform multiple ligand conformations within the protein into a protein ligand interaction landscape, and uses this landscape as the primary input for the model. In addition, we expanded the PDBbind koff dataset from 680 to 1,197 entries and employed the augmented dataset for model training and testing. When tested through five-fold cross-validation, STELLAR-koff achieved Pearson correlation coefficient of 0.729 surpassing or being on pair with most of the published prediction methods. Tested on external set, STELLAR-koff demonstrated strong predictive performance on unseen protein, achieving a Pearson of 0.838 on the focal adhesion kinase in particular. Experimental validation on cyclin-dependent kinase also demonstrated the effectiveness of STELLAR-koff in real drug discovering scenarios. We believe this study provides an effective tool for predicting protein-ligand dissociation rate constant and offers new insight for the future development of this field.

---

## 25. Few-Shot Multimodal Medical Imaging: A Theoretical Framework

**论文链接:** [http://arxiv.org/abs/2511.01140v1](http://arxiv.org/abs/2511.01140v1)

**作者:** Md Talha Mohsin, Ismail Abdulrashid

**发布时间:** 2025-11-03

**备注:** 6 Pages

### GPT解析

### 总结

本文提出了一种统一的理论框架，用于描述低资源医学影像条件下的学习和推理，旨在解决医学影像领域数据稀缺问题。

### 背景

医学影像依赖于大型标记数据集，但在临床环境中这些数据集不易获取，从业者面临数据有限、数据系统碎片化和数据集不平衡等结构障碍。

### 目的

为数据稀缺情况下医学影像学习方法提供坚实的理论基础，解释为什么某些方法成功或失败。

### 方法

在少样本条件下形式化学习目标并计算样本复杂度约束；基于PAC学习和PAC-Bayesian理论解释多模态集成如何促进泛化和量化不确定性；提出解释稳定性形式化度量。

### 主要发现

多模态集成可以在数据稀缺条件下促进泛化和量化不确定性；解释稳定性度量可以为低数据条件下的可解释性提供保证。

### 结论

所提出的框架通过统一描述样本效率、不确定量化和可解释性，为构建可靠、数据高效的医学影像诊断系统奠定了理论基础。

### 翻译

医学成像在很大程度上依赖于大型标记数据集。但不幸的是，在临床环境中它们并不总是容易获取。此外，许多从业者经常面临各种结构障碍，如数据可用性有限、数据系统碎片化和数据集不平衡。这些障碍通常导致诊断不确定性增加、某些疾病表现不足、模型鲁棒性降低和诊断决策有偏见。为应对这些挑战，转移学习、元学习和多模态融合等方法已取得长足进展。然而，在数据稀缺的情况下，它们成功或失败的原因仍缺乏坚实的理论依据。为解决这一差距，我们提出了一个统一的理论框架，用于描述低资源医学影像条件下的学习和推理。我们首先在少样本条件下形式化学习目标，并计算样本复杂度约束，以估计实现临床可靠精度所需的最小数据量。然后基于PAC学习和PAC-Bayesian理论的思想，我们解释多模态集成如何促进泛化，以及在稀疏监督下量化不确定性。我们进一步提出了解释稳定性的形式化度量，为低数据条件下的可解释性提供保证。总之，所提出的框架通过在统一理论设定中共同描述样本效率、不确定量化和可解释性，为构建可靠、数据高效的诊断系统奠定了基础。


### 论文摘要

Medical imaging relies heavily on large, labeled datasets. But, unfortunately, they are not always easily accessible in clinical settings. Additionally, many practitioners often face various structural obstacles like limited data availability, fragmented data systems, and unbalanced datasets. These barriers often lead to the increased diagnostic uncertainty, underrepresentation of certain conditions, reduced model robustness, and biased diagnostic decisions. In response to these challenges, approaches such as transfer learning, meta-learning, and multimodal fusion have made great strides. However, they still need a solid theoretical justification for why they succeed or fail in situations where data is scarce. To address this gap, we propose a unified theoretical framework that characterizes learning and inference under low-resource medical imaging conditions. We first formalize the learning objective under few-shot conditions and compute sample complexity constraints to estimate the smallest quantity of data needed to achieve clinically reliable accuracy. Then based on ideas from PAC-learning and PAC-Bayesian theory, we explain how multimodal integration encourages generalization and quantifies uncertainty under sparse supervision. We further propose a formal metric for explanation stability, offering interpretability guarantees under low-data conditions. Taken together, the proposed framework establishes a principled foundation for constructing dependable, data-efficient diagnostic systems by jointly characterizing sample efficiency, uncertainty quantification, and interpretability in a unified theoretical setting.

---

## 26. Learning an Efficient Optimizer via Hybrid-Policy Sub-Trajectory Balance

**论文链接:** [http://arxiv.org/abs/2511.00543v1](http://arxiv.org/abs/2511.00543v1)

**作者:** Yunchuan Guan, Yu Liu, Ke Zhou, Hui Li, Sen Jia, Zhiqi Shen, Ziyang Wang, Xinglin Zhang, Tao Chen, Jenq-Neng Hwang, Lei Li

**发布时间:** 2025-11-01

### GPT解析

### 总结

本文提出了一种名为Lo-Hp的解耦两阶段权重生成框架，解决了生成式模型在神经网络权重生成中的过耦合和长时程问题，提高了灵活性和效率。

### 背景

生成式模型的最新进展使得神经网络能够在不依赖基于梯度优化的情况下生成权重，但当前方法受限于过耦合和长时程问题。过耦合将权重生成与特定任务目标紧密绑定，限制了学习优化器的灵活性；长时程问题因缺乏局部约束导致推理效率低下和准确性不足。

### 目的

解决现有权重生成方法的过耦合和长时程问题，提高框架的灵活性和推理效率，特别是在需要频繁权重更新的任务中。

### 方法

提出Lo-Hp框架，一个解耦的两阶段权重生成方法，通过学习各种优化策略增强灵活性。采用混合策略子轨迹平衡目标，结合在线学习和离线学习来捕获局部优化策略。

### 主要发现

理论上证明了仅学习局部优化策略可以解决长时程问题，同时增强全局最优权重的生成。在需要频繁权重更新的任务中验证了Lo-Hp的优越准确性和推理效率。

### 结论

Lo-Hp在迁移学习、小样本学习、领域泛化和大型语言模型适应等需要频繁权重更新的任务中表现出色，具有更高的准确性和推理效率。

### 翻译

生成式建模的最新进展使神经网络能够在不依赖基于梯度优化的情况下生成权重。然而，当前方法受限于过耦合和长时程问题。前者将权重生成与特定任务目标紧密绑定，从而限制了学习优化器的灵活性。后者因缺乏局部约束导致推理效率低下和准确性不足。在本文中，我们提出了Lo-Hp，一个解耦的两阶段权重生成框架，通过学习各种优化策略来提高灵活性。它采用混合策略子轨迹平衡目标，结合在线学习和离线学习来捕获局部优化策略。理论上，我们证明了仅学习局部优化策略可以解决长时程问题，同时增强全局最优权重的生成。此外，我们在需要频繁权重更新的任务（如迁移学习、小样本学习、领域泛化和大型语言模型适应）中验证了Lo-Hp的优越准确性和推理效率。


### 论文摘要

Recent advances in generative modeling enable neural networks to generate weights without relying on gradient-based optimization. However, current methods are limited by issues of over-coupling and long-horizon. The former tightly binds weight generation with task-specific objectives, thereby limiting the flexibility of the learned optimizer. The latter leads to inefficiency and low accuracy during inference, caused by the lack of local constraints. In this paper, we propose Lo-Hp, a decoupled two-stage weight generation framework that enhances flexibility through learning various optimization policies. It adopts a hybrid-policy sub-trajectory balance objective, which integrates on-policy and off-policy learning to capture local optimization policies. Theoretically, we demonstrate that learning solely local optimization policies can address the long-horizon issue while enhancing the generation of global optimal weights. In addition, we validate Lo-Hp's superior accuracy and inference efficiency in tasks that require frequent weight updates, such as transfer learning, few-shot learning, domain generalization, and large language model adaptation.

---

## 27. Transfer Learning for Onboard Cloud Segmentation in Thermal Earth Observation: From Landsat to a CubeSat Constellation

**论文链接:** [http://arxiv.org/abs/2511.00357v1](http://arxiv.org/abs/2511.00357v1)

**作者:** Niklas Wölki, Lukas Kondmann, Christian Mollière, Martin Langer, Julia Gottfriedsen, Martin Werner

**发布时间:** 2025-11-01

**备注:** This work was presented at the TerraBytes Workshop at the 42nd  International Conference on Machine Learning. This version is not part of the  official ICML proceedings

### GPT解析

### 总结

该研究解决了CubeSat任务中热红外云分割的挑战，通过迁移学习和轻量级架构实现了准确、高效的在轨云分割，支持实时决策。

### 背景

CubeSat任务在热红外地球观测中的云分割是一个关键但研究不足的任务。这些卫星受限于硬件，通常只能依赖单一热红外波段，且缺乏足够的标记数据，使得传统云掩膜技术不可行。

### 目的

解决CubeSat任务中热红外云分割的挑战，通过迁移学习方法实现准确、高效的热红外云分割，支持实时决策。

### 方法

将迁移学习应用于FOREST-2 CubeSat的热红外云分割任务，使用带有轻量级MobileNet编码器的UNet模型。在公共的Landsat-7云覆盖评估数据集上预训练模型，然后在联合训练设置中使用少量任务特定样本进行微调。最后将模型转换为TensorRT引擎，在NVIDIA Jetson Nano上实现全图像推理。

### 主要发现

通过迁移学习方法，宏F1分数从仅使用FOREST-2基线的0.850提高到0.877。利用公共数据集和轻量级架构可以在轨上实现准确、高效的热红外云分割，全图像推理时间不到5秒。

### 结论

利用公共数据集和轻量级架构可以在轨上实现准确、高效的热红外云分割，支持数据受限地球观测任务中的实时决策。

### 翻译

机载云分割是热红外地球观测中的一个关键但研究不足的任务，特别是对于受限于有限硬件和光谱信息的CubeSat任务。CubeSat通常依赖单一热红外波段，且缺乏足够的标记数据，这使得传统的云掩膜技术不可行。这项工作通过将迁移学习应用于FOREST-2 CubeSat的热红外云分割来解决这些挑战，使用带有轻量级MobileNet编码器的UNet模型。我们在公共的Landsat-7云覆盖评估数据集上预训练模型，然后在联合训练设置中使用少量任务特定样本进行微调，使宏F1分数从仅使用FOREST-2基线的0.850提高到0.877。我们将模型转换为TensorRT引擎，并在NVIDIA Jetson Nano上演示了全图像推理，时间不到5秒。这些结果表明，利用公共数据集和轻量级架构可以在轨上实现准确、高效的热红外云分割，支持数据受限地球观测任务中的实时决策。


### 论文摘要

Onboard cloud segmentation is a critical yet underexplored task in thermal Earth observation (EO), particularly for CubeSat missions constrained by limited hardware and spectral information. CubeSats often rely on a single thermal band and lack sufficient labeled data, making conventional cloud masking techniques infeasible. This work addresses these challenges by applying transfer learning to thermal cloud segmentation for the FOREST-2 CubeSat, using a UNet with a lightweight MobileNet encoder. We pretrain the model on the public Landsat-7 Cloud Cover Assessment Dataset and fine-tune it with a small set of mission-specific samples in a joint-training setup, improving the macro F1 from 0.850 to 0.877 over FOREST-2-only baselines. We convert the model to a TensorRT engine and demonstrate full-image inference in under 5 seconds on an NVIDIA Jetson Nano. These results show that leveraging public datasets and lightweight architectures can enable accurate, efficient thermal-only cloud masking on-orbit, supporting real-time decision-making in data-limited EO missions.

---

## 28. FedReplay: A Feature Replay Assisted Federated Transfer Learning Framework for Efficient and Privacy-Preserving Smart Agriculture

**论文链接:** [http://arxiv.org/abs/2511.00269v1](http://arxiv.org/abs/2511.00269v1)

**作者:** Long Li, Jiajia Li, Dong Chen, Lina Pu, Haibo Yao, Yanbo Huang

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种结合冻结的CLIP视觉变换器和轻量级变换器分类器的联邦学习框架，用于解决智能农业中的准确分类问题，同时处理隐私保护和非IID数据分布的挑战。

### 背景

准确分类在智能农业中扮演关键角色，但传统集中式训练引发隐私问题，标准联邦学习难以处理非IID数据且通信成本高。

### 目的

开发一种联邦学习框架，解决数据隐私和非IID数据分布问题，同时降低通信成本。

### 方法

利用预训练的CLIP ViT特征提取能力，避免从头训练大规模模型；将联邦更新限制在紧凑分类器上减少传输开销；共享1%的CLIP特征表示对齐不同参与者的类别表示，同时保护隐私。

### 主要发现

在农业分类任务上达到86.6%的准确率，比基线联邦学习方法高出4倍以上。

### 结论

结合视觉-语言模型特征与联邦学习能有效实现隐私保护和可扩展的农业智能。

### 翻译

准确的分类在智能农业中起着关键作用，支持作物监测、果实识别和病虫害检测等应用。然而，传统的集中式训练通常需要大规模数据收集，这引发了隐私问题，而标准的联邦学习难以处理非独立同分布数据并产生高通信成本。为解决这些挑战，我们提出了一种结合冻结的对比语言-图像预训练视觉变换器和轻量级变换器分类器的联邦学习框架。通过利用预训练的CLIP视觉变换器的强大特征提取能力，该框架避免了从头开始训练大规模模型，并将联邦更新限制在紧凑的分类器上，从而显著减少了传输开销。此外，为了减轻由非IID数据分布导致的性能下降，从所有类别中共享一小部分(1%)的CLIP提取的特征表示。这些共享的特征无法逆向还原为原始图像，确保了隐私保护，同时使不同参与者的类别表示保持一致。在农业分类任务上的实验结果表明，所提出的方法达到了86.6%的准确率，比基线联邦学习方法高出4倍以上。这证明了将视觉-语言模型特征与联邦学习相结合的有效性和效率，为隐私保护和可扩展的农业智能提供了新方法。


### 论文摘要

Accurate classification plays a pivotal role in smart agriculture, enabling applications such as crop monitoring, fruit recognition, and pest detection. However, conventional centralized training often requires large-scale data collection, which raises privacy concerns, while standard federated learning struggles with non-independent and identically distributed (non-IID) data and incurs high communication costs. To address these challenges, we propose a federated learning framework that integrates a frozen Contrastive Language-Image Pre-training (CLIP) vision transformer (ViT) with a lightweight transformer classifier. By leveraging the strong feature extraction capability of the pre-trained CLIP ViT, the framework avoids training large-scale models from scratch and restricts federated updates to a compact classifier, thereby reducing transmission overhead significantly. Furthermore, to mitigate performance degradation caused by non-IID data distribution, a small subset (1%) of CLIP-extracted feature representations from all classes is shared across clients. These shared features are non-reversible to raw images, ensuring privacy preservation while aligning class representation across participants. Experimental results on agricultural classification tasks show that the proposed method achieve 86.6% accuracy, which is more than 4 times higher compared to baseline federated learning approaches. This demonstrates the effectiveness and efficiency of combining vision-language model features with federated learning for privacy-preserving and scalable agricultural intelligence.

---

## 29. Melanoma Classification Through Deep Ensemble Learning and Explainable AI

**论文链接:** [http://arxiv.org/abs/2511.00246v1](http://arxiv.org/abs/2511.00246v1)

**作者:** Wadduwage Shanika Perera, ABM Islam, Van Vung Pham, Min Kyung An

**发布时间:** 2025-10-31

**DOI:** 10.5220/0012575400003657

**备注:** Publisher-formatted version provided under CC BY-NC-ND 4.0 license.  Original source produced by SciTePress

### GPT解析

### 总结

本研究提出了一种结合可解释人工智能技术的机器学习模型，用于黑色素瘤的早期检测，以提高诊断的可靠性和信任度。

### 背景

黑色素瘤是最具侵袭性和致命性的皮肤癌之一，如果在早期不被发现和治疗会导致死亡。人工智能技术，特别是深度学习已被开发用于帮助皮肤科医生早期检测黑色素瘤，并取得了高准确率。然而，深度学习模型的黑盒操作导致缺乏可靠性和信任度。

### 目的

开发一种可靠的机器学习模型，通过集成学习和可解释人工智能技术来提高黑色素瘤早期检测的准确性和可信度。

### 方法

提出了一种机器学习模型，使用三种最先进的深度迁移学习网络的集成学习，并结合可解释人工智能技术来解释预测的基础。

### 主要发现

通过集成三种最先进的深度迁移学习网络和可解释人工智能技术，可以提高黑色素瘤检测的准确性和可靠性。

### 结论

可解释人工智能技术可以解决深度学习模型黑盒操作导致的可靠性和信任度问题，为医疗诊断领域提供更可靠的AI辅助诊断工具。

### 翻译

黑色素瘤是最具侵袭性和致命性的皮肤癌之一，如果在早期不被发现和治疗会导致死亡。人工智能技术最近已被开发用于帮助皮肤科医生早期检测黑色素瘤，基于深度学习的系统能够以高准确率检测这些病变。然而，整个社区必须克服可解释性的限制，才能在医疗诊断领域从深度学习中获得最大收益。由于深度学习模型决策中的黑盒操作缺陷，结果缺乏可靠性和信任度。然而，可解释人工智能(XAI)可以通过解释AI系统的预测来解决这个问题。本文提出了一种机器学习模型，使用三种最先进的深度迁移学习网络的集成学习，并利用XAI技术解释预测的基础，确保预测的可靠性。


### 论文摘要

Melanoma is one of the most aggressive and deadliest skin cancers, leading to mortality if not detected and treated in the early stages. Artificial intelligence techniques have recently been developed to help dermatologists in the early detection of melanoma, and systems based on deep learning (DL) have been able to detect these lesions with high accuracy. However, the entire community must overcome the explainability limit to get the maximum benefit from DL for diagnostics in the healthcare domain. Because of the black box operation's shortcomings in DL models' decisions, there is a lack of reliability and trust in the outcomes. However, Explainable Artificial Intelligence (XAI) can solve this problem by interpreting the predictions of AI systems. This paper proposes a machine learning model using ensemble learning of three state-of-the-art deep transfer Learning networks, along with an approach to ensure the reliability of the predictions by utilizing XAI techniques to explain the basis of the predictions.

---

## 30. An Efficient and Generalizable Transfer Learning Method for Weather Condition Detection on Ground Terminals

**论文链接:** [http://arxiv.org/abs/2511.00211v1](http://arxiv.org/abs/2511.00211v1)

**作者:** Wenxuan Zhang, Peng Hu

**发布时间:** 2025-10-31

**DOI:** 10.1109/TAES.2024.3496857

### GPT解析

### 总结

这篇论文提出了一种高效的迁移学习方法，用于检测地面终端组件上的天气相关条件，如积雪和潮湿等，以提高低地球轨道卫星互联网系统在恶劣天气条件下的性能和可靠性。该方法在检测多种天气条件下表现出色，且具有很好的泛化能力。

### 背景

随着低地球轨道卫星星座在卫星互联网中的广泛应用，为农村和偏远地区提供了无处不在的连接能力。然而，天气事件对卫星互联网的性能和可靠性有显著影响。雪、雨等不良天气会严重干扰卫星天线等关键地面终端组件的性能，破坏LEO卫星与地面站之间的空间-地面链路条件。

### 目的

研究需要基于地区的天气预报以及对地面终端组件上精细化天气条件的检测能力，以支持卫星互联网的故障诊断和缓解，确保系统的可靠性。然而，目前缺乏有效的解决方案，且在实际部署中需要考虑解决方案的有效性和泛化能力。

### 方法

论文讨论了一种高效的迁移学习(TL)方法，使地面组件能够本地检测代表性的天气相关条件。该方法可以检测由不良和典型天气事件引起的积雪、潮湿等条件。

### 主要发现

所提出的迁移学习方法在检测多种天气条件下表现出色，与典型的深度学习方法如YOLOv7、YOLOv9、Faster R-CNN和R-YOLO相比具有优越性能。此外，该方法还显示出能够泛化到各种场景的优势。

### 结论

该迁移学习方法为解决天气对卫星互联网地面终端组件的影响提供了有效解决方案，能够提高卫星互联网在恶劣天气条件下的可靠性和性能，且具有良好的泛化能力，适用于各种实际部署场景。

### 翻译

随着低地球轨道(LEO)卫星星座在卫星互联网中的日益普及，为农村和偏远地区提供了无处不在的连接能力。然而，天气事件对卫星互联网的性能和可靠性有重大影响。雪、雨等不良天气事件会显著干扰卫星互联网关键地面终端组件（如卫星天线）的性能和运行，严重破坏LEO卫星与地面站之间的空间-地面链路条件。这一挑战不仅需要基于地区的天气预报，还需要对地面终端组件上的精细化天气条件进行精细检测。这种能力可以帮助卫星互联网进行故障诊断和缓解，确保可靠性，但相应的解决方案仍然缺乏，更不用说在实际部署中必不可少的有效性和泛化能力。本文讨论了一种高效的迁移学习(TL)方法，使地面组件能够本地检测代表性的天气相关条件。所提出的方法可以检测由不良和典型天气事件引起的积雪、潮湿等条件，与典型的深度学习方法（如YOLOv7、YOLOv9、Faster R-CNN和R-YOLO）相比表现出优越性能。我们的迁移学习方法还显示出能够泛化到各种场景的优势。


### 论文摘要

The increasing adoption of satellite Internet with low-Earth-orbit (LEO) satellites in mega-constellations allows ubiquitous connectivity to rural and remote areas. However, weather events have a significant impact on the performance and reliability of satellite Internet. Adverse weather events such as snow and rain can disturb the performance and operations of satellite Internet's essential ground terminal components, such as satellite antennas, significantly disrupting the space-ground link conditions between LEO satellites and ground stations. This challenge calls for not only region-based weather forecasts but also fine-grained detection capability on ground terminal components of fine-grained weather conditions. Such a capability can assist in fault diagnostics and mitigation for reliable satellite Internet, but its solutions are lacking, not to mention the effectiveness and generalization that are essential in real-world deployments. This paper discusses an efficient transfer learning (TL) method that can enable a ground component to locally detect representative weather-related conditions. The proposed method can detect snow, wet, and other conditions resulting from adverse and typical weather events and shows superior performance compared to the typical deep learning methods, such as YOLOv7, YOLOv9, Faster R-CNN, and R-YOLO. Our TL method also shows the advantage of being generalizable to various scenarios.

---

## 31. Transfer learning discovery of molecular modulators for perovskite solar cells

**论文链接:** [http://arxiv.org/abs/2511.00204v1](http://arxiv.org/abs/2511.00204v1)

**作者:** Haoming Yan, Xinyu Chen, Yanran Wang, Zhengchao Luo, Weizheng Huang, Hongshuai Wang, Peng Chen, Yuzhi Zhang, Weijie Sun, Jinzhuo Wang, Qihuang Gong, Rui Zhu, Lichen Zhao

**发布时间:** 2025-10-31

### GPT解析

### 总结

本研究开发了一种基于预训练深度神经网络的化学信息迁移学习框架，用于预测分子调节剂对钙钛矿太阳能电池功率转换效率的影响，并通过实验验证了该方法的有效性，实现了26.91%的高效率。

### 背景

钙钛矿太阳能电池的有效分子调节剂发现对推进其发展至关重要，但化学空间的庞大以及耗时昂贵的实验筛选阻碍了研究进程。同时，机器学习在加速材料发现方面有潜力，但由于数据稀缺和传统定量结构-性质关系模型的限制，将其应用于钙钛矿太阳能电池仍面临挑战。

### 目的

开发一种能够高效预测分子调节剂对钙钛矿太阳能电池性能影响的机器学习方法，以加速新型分子调节剂的发现过程。

### 方法

应用基于预训练深度神经网络的化学信息迁移学习框架，通过系统性地对多种分子表示进行基准测试，建立预测模型，并利用可解释性技术可视化学习到的化学表示，最后对筛选出的候选分子进行实验验证。

### 主要发现

该框架能够低成本、高通量地筛选79,043种商业可用分子，并准确预测分子调节剂对钙钛矿太阳能电池功率转换效率的影响。实验验证表明，该框架确定的前分子调节剂显著提高了电池性能，实现了26.91%的冠军功率转换效率。

### 结论

基于预训练深度神经网络的化学信息迁移学习框架为钙钛矿太阳能电池分子调节剂的发现提供了有效工具，克服了传统方法中的数据稀缺和模型限制问题，显著加速了材料发现进程。

### 翻译

有效分子调节剂的发现对推进钙钛矿太阳能电池的发展至关重要，但研究过程受到化学空间庞大以及耗时昂贵的实验筛选的阻碍。同时，机器学习在加速材料发现方面具有巨大潜力。然而，由于数据稀缺和传统定量结构-性质关系模型的局限性，将机器学习应用于钙钛矿太阳能电池仍是一个重大挑战。在此，我们应用了一种基于预训练深度神经网络的化学信息迁移学习框架，能够高精度地预测分子调节剂对钙钛矿太阳能电池功率转换效率的影响。通过对多种分子表示进行系统性基准测试建立了该框架，能够以低成本和高通量方式对79,043种商业可用分子进行虚拟筛选。此外，我们利用可解释性技术可视化学习到的化学表示，并对得到的调节剂-钙钛矿相互作用进行实验表征。该框架确定的前分子调节剂随后通过实验验证，在钙钛矿太阳能电池中实现了显著提高的冠军功率转换效率26.91%。


### 论文摘要

The discovery of effective molecular modulators is essential for advancing perovskite solar cells (PSCs), but the research process is hindered by the vastness of chemical space and the time-consuming and expensive trial-and-error experimental screening. Concurrently, machine learning (ML) offers significant potential for accelerating materials discovery. However, applying ML to PSCs remains a major challenge due to data scarcity and limitations of traditional quantitative structure-property relationship (QSPR) models. Here, we apply a chemical informed transfer learning framework based on pre-trained deep neural networks, which achieves high accuracy in predicting the molecular modulator's effect on the power conversion efficiency (PCE) of PSCs. This framework is established through systematical benchmarking of diverse molecular representations, enabling lowcost and high-throughput virtual screening over 79,043 commercially available molecules. Furthermore, we leverage interpretability techniques to visualize the learned chemical representation and experimentally characterize the resulting modulator-perovskite interactions. The top molecular modulators identified by the framework are subsequently validated experimentally, delivering a remarkably improved champion PCE of 26.91% in PSCs.

---

## 32. Retrieval-Augmented Multimodal Depression Detection

**论文链接:** [http://arxiv.org/abs/2511.01892v1](http://arxiv.org/abs/2511.01892v1)

**作者:** Ruibo Hou, Shiyu Teng, Jiaqing Liu, Shurong Chai, Yinhao Li, Lanfen Lin, Yen-Wei Chen

**发布时间:** 2025-10-29

**备注:** Accepted in IEEE EMBC 2025

### GPT解析

### 总结

本研究提出了一种新颖的检索增强生成（RAG）框架，用于抑郁症检测，通过情感提示作为辅助模态提高情感表示和可解释性。

### 背景

多模态深度学习通过整合文本、音频和视频信号在抑郁症检测中显示出潜力。然而，现有利用情感分析增强情感理解的方法存在计算成本高、领域不匹配和静态知识限制的问题。

### 目的

解决现有抑郁症检测方法中计算成本高、领域不匹配和静态知识限制的问题。

### 方法

提出检索增强生成（RAG）框架，给定抑郁症相关文本，从情感数据集中检索语义相关的情感内容，使用大型语言模型生成情感提示作为辅助模态，以丰富情感表示并提高可解释性。

### 主要发现

在AVEC 2019数据集上的实验表明，该方法实现了最先进的性能，CCC达到0.593，MAE达到3.95，超过了之前的迁移学习和多任务学习基线。

### 结论

检索增强生成框架能有效解决现有方法在抑郁症检测中的问题，提高性能和可解释性。

### 翻译

多模态深度学习通过整合文本、音频和视频信号在抑郁症检测中显示出潜力。最近的研究利用情感分析来增强情感理解，但存在计算成本高、领域不匹配和静态知识限制的问题。为解决这些问题，我们提出了一种新颖的检索增强生成（RAG）框架。给定一个与抑郁症相关的文本，我们的方法从情感数据集中检索语义上相关的情感内容，并使用大型语言模型生成情感提示作为辅助模态。这个提示丰富了情感表示并提高了可解释性。在AVEC 2019数据集上的实验表明，我们的方法实现了最先进的性能，CCC达到0.593，MAE达到3.95，超过了之前的迁移学习和多任务学习基线。


### 论文摘要

Multimodal deep learning has shown promise in depression detection by integrating text, audio, and video signals. Recent work leverages sentiment analysis to enhance emotional understanding, yet suffers from high computational cost, domain mismatch, and static knowledge limitations. To address these issues, we propose a novel Retrieval-Augmented Generation (RAG) framework. Given a depression-related text, our method retrieves semantically relevant emotional content from a sentiment dataset and uses a Large Language Model (LLM) to generate an Emotion Prompt as an auxiliary modality. This prompt enriches emotional representation and improves interpretability. Experiments on the AVEC 2019 dataset show our approach achieves state-of-the-art performance with CCC of 0.593 and MAE of 3.95, surpassing previous transfer learning and multi-task learning baselines.

---

## 33. Multi-Mapcher: Loop Closure Detection-Free Heterogeneous LiDAR Multi-Session SLAM Leveraging Outlier-Robust Registration for Autonomous Vehicles

**论文链接:** [http://arxiv.org/abs/2511.00635v1](http://arxiv.org/abs/2511.00635v1)

**作者:** Hyungtae Lim, Daebeom Kim, Hyun Myung

**发布时间:** 2025-11-01

**备注:** 13 pages, 12 figures

### GPT解析

### 总结

本研究提出了一种名为Multi-Mapcher的新型多会话同步定位与地图构建框架，通过大规模地图到地图配准实现会话间初始对齐，克服了传统方法对回环检测的依赖，实验证明该方法在各种LiDAR传感器条件下表现优异且速度更快。

### 背景

随着各种3D光探测和测距（LiDAR）传感器被引入市场，使用异构LiDAR传感器进行多会话同步定位与地图构建的研究已经积极展开。现有的MSS方法大多依赖于回环检测来实现会话间的对齐；然而，由于不同会话中使用的传感器在密度和视场（FoV）上存在差异，回环检测的性能可能会降低。

### 目的

挑战现有方法严重依赖检测模块的范式，提出一种使用大规模地图到地图配准实现会话间初始对齐的新型MSS框架，这通常被认为不可行。

### 方法

利用抗异常值的3D点云配准技术进行会话间初始对齐，然后在假设初始对齐足够精确的情况下，通过基于半径搜索发现会话间的回环，最后采用基于锚节点的鲁棒姿态图优化来构建一致的全局地图。

### 主要发现

使用各种LiDAR传感器捕获会话时，新方法表现出显著更好的MSS性能，并且比最先进的方法更快。

### 结论

Multi-Mapcher框架通过创新地使用大规模地图到地图配准，有效解决了异构LiDAR传感器在多会话同步定位与地图构建中的挑战，代码已公开在https://github.com/url-kaist/multi-mapcher。

### 翻译

随着各种3D光探测和测距（LiDAR）传感器被引入市场，使用异构LiDAR传感器进行多会话同步定位与地图构建（MSS）的研究已经积极展开。现有的MSS方法大多依赖于回环检测来实现会话间的对齐；然而，由于不同会话中使用的传感器在密度和视场（FoV）上存在差异，回环检测的性能可能会降低。本研究挑战了现有方法严重依赖检测模块的范式，提出了一种名为Multi-Mapcher的新型MSS框架，该框架采用大规模地图到地图配准来实现会话间初始对齐，这通常被认为不可行，方法是利用抗异常值的3D点云配准。在假设会话间初始对齐足够精确的情况下，通过基于半径搜索来发现会话间的回环，然后采用基于锚节点的鲁棒姿态图优化来构建一致的全局地图。如我们的实验所示，我们的方法在使用各种LiDAR传感器捕获会话时表现出显著更好的MSS性能，并且比最先进的方法更快。我们的代码可在https://github.com/url-kaist/multi-mapcher获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多会话SLAM中，当使用不同类型激光雷达传感器时，现有方法依赖的回环检测模块性能下降导致会话间对齐失败的问题。这个问题很重要，因为随着市场上各种激光雷达传感器的出现，自动驾驶车辆和机器人可能配备不同类型的传感器，而现有方法在处理异构传感器数据时表现不佳，导致无法构建一致的全局地图，限制了长期地图管理和多机器人协作的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有MSS方法在异构激光雷达传感器上表现不佳，挑战了依赖回环检测的范式。他们借鉴了异常值鲁棒的点云注册方法（如Quatro），使用快速点特征直方图（FPFH）建立对应关系，并引入锚节点概念解决多会话轨迹参考帧不固定的问题。作者还借鉴了基于因子图的姿态图优化方法，但进行了改进以适应多会话场景，实现了地图到地图和扫描到扫描两个级别的鲁棒注册。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是不依赖传统回环检测模块，而是使用异常值鲁棒的点云注册实现会话间对齐。整体流程包括：1) 会话内SLAM处理；2) 地图到地图级别的会话间初始对齐，使用FPFH特征和鲁棒注册估计变换矩阵；3) 扫描到扫描级别的会话间回环闭合，通过半径搜索和截断均方误差过滤错误候选；4) 基于锚节点的姿态图优化；5) 构建一致的全局地图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 不依赖回环检测的LCD-free方法；2) 实现大规模地图到地图注册；3) 在地图和扫描两个级别使用鲁棒注册；4) 提出截断均方误差处理视场差异；5) 支持异构激光雷达传感器。相比之前工作，Multi-Mapcher在处理异构传感器数据时表现更好，不会因LCD性能下降导致失败；将2D地图合并扩展到3D；对部分重叠和环境变化更鲁棒；速度比现有方法快5-9倍。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Multi-Mapcher通过不依赖回环检测的鲁棒注册方法，实现了对异构激光雷达传感器捕获的多会话SLAM的精确对齐，构建了一致的全局地图，同时比现有方法更快且更鲁棒。'}


### 论文摘要

As various 3D light detection and ranging (LiDAR) sensors have been introduced to the market, research on multi-session simultaneous localization and mapping (MSS) using heterogeneous LiDAR sensors has been actively conducted. Existing MSS methods mostly rely on loop closure detection for inter-session alignment; however, the performance of loop closure detection can be potentially degraded owing to the differences in the density and field of view (FoV) of the sensors used in different sessions. In this study, we challenge the existing paradigm that relies heavily on loop detection modules and propose a novel MSS framework, called Multi-Mapcher, that employs large-scale map-to-map registration to perform inter-session initial alignment, which is commonly assumed to be infeasible, by leveraging outlier-robust 3D point cloud registration. Next, after finding inter-session loops by radius search based on the assumption that the inter-session initial alignment is sufficiently precise, anchor node-based robust pose graph optimization is employed to build a consistent global map. As demonstrated in our experiments, our approach shows substantially better MSS performance for various LiDAR sensors used to capture the sessions and is faster than state-of-the-art approaches. Our code is available at https://github.com/url-kaist/multi-mapcher.

---

## 34. MambaNetLK: Enhancing Colonoscopy Point Cloud Registration with Mamba

**论文链接:** [http://arxiv.org/abs/2511.00260v1](http://arxiv.org/abs/2511.00260v1)

**作者:** Linzhe Jiang, Jiayuan Huang, Sophia Bano, Matthew J. Clarkson, Zhehua Mao, Mobarak I. Hoque

**发布时间:** 2025-10-31

**备注:** 12 pages, 4 figures, 3 tables, IPCAI conference

### GPT解析

### 总结

本文提出了一种名为MambaNetLK的新型3D点云配准方法，以及一个名为C3VD-Raycasting-10k的大规模临床数据集，用于解决结肠镜引导中生物组织特征退化和术前术后域差异导致的配准稳定性问题。

### 背景

准确的3D点云配准是可靠的图像引导结肠镜检查的基础，直接影响病变定位、边缘评估和导航安全性。然而，生物组织具有重复纹理和局部均匀几何特征，导致特征退化，同时术前解剖和术中观察之间的显著域差异进一步降低了配准稳定性。

### 目的

解决临床关键挑战，引入一种针对内镜导航的新型3D配准方法，并创建高质量、临床基础的数据集以支持严格且可复现的基准测试。

### 方法

引入C3VD-Raycasting-10k数据集，包含10,014对从临床CT数据派生的几何对齐点云；提出MambaNetLK框架，通过将Mamba状态空间模型作为跨模态特征提取器集成到PointNetLK架构中增强其能力，有效捕获长程依赖关系，并使用Lucas-Kanade算法迭代实现对齐。

### 主要发现

在临床数据集C3VD-Raycasting-10k上，MambaNetLK实现了最先进的性能，与第二好的方法相比，中值旋转误差降低了56.04%，RMSE平移误差降低了26.19%；该模型在ModelNet40上表现出强大的泛化能力，并对初始姿态扰动具有优异的鲁棒性。

### 结论

MambaNetLK为外科导航中的3D配准提供了强大基础，基于SSM的全局表达特征提取器与大规模临床数据集的结合，使微创手术如结肠镜检查中的引导系统更加准确和可靠。

### 翻译

准确的3D点云配准是可靠的图像引导结肠镜检查的基础，直接影响病变定位、边缘评估和导航安全性。然而，生物组织具有重复纹理和局部均匀几何特征，导致特征退化，同时术前解剖和术中观察之间的显著域差异进一步降低了配准稳定性。为解决这些临床关键挑战，我们引入了一种针对内镜导航的新型3D配准方法和高质量、临床基础的数据集以支持严格且可复现的基准测试。我们引入了C3VD-Raycasting-10k，这是一个包含10,014对从临床CT数据派生的几何对齐点云的大规模基准数据集。我们提出了MambaNetLK，一种新的无对应点配准框架，通过将Mamba状态空间模型作为跨模态特征提取器集成到PointNetLK架构中来增强它。因此，所提出的框架以线性时间复杂度有效捕获长程依赖关系。对齐是通过使用Lucas-Kanade算法迭代实现的。在临床数据集C3VD-Raycasting-10k上，MambaNetLK与最先进的方法相比实现了最佳性能，中值旋转误差比第二好的方法降低了56.04%，RMSE平移误差降低了26.19%。该模型在ModelNet40上也表现出强大的泛化能力，并对初始姿态扰动具有优异的鲁棒性。MambaNetLK为外科导航中的3D配准提供了强大的基础。基于SSM的全局表达特征提取器与大规模临床数据集的结合，使结肠镜等微创手术中的引导系统更加准确和可靠。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决结肠镜检查中的3D点云配准问题，即如何准确地将手术中的实时内窥镜数据与术前的CT扫描模型进行对齐。这个问题非常重要，因为准确的配准直接关系到病变定位、边界评估和导航安全，而生物组织的重复纹理和局部均匀几何特性会导致特征退化，同时术前与术中数据间的域偏移会进一步降低配准稳定性，影响临床诊断和治疗效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于对现有方法局限性的分析进行设计：基于对应关系的方法在平滑、无纹理的器官表面存在特征退化；PointNetLK等无对应关系方法依赖MLP特征提取器，难以捕获长距离几何依赖；Transformer虽有长距离建模能力但在手术应用中受限。作者借鉴了PointNetLK的框架，但用Mamba状态空间模型替代了MLP特征提取器，并创建了新的临床数据集C3VD-Raycasting-10k来解决基准数据缺乏的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将Mamba状态空间模型整合到Lucas-Kanade配准流程中，通过将点云视为序列来有效捕获全局几何结构，避免显式点对应关系。整体流程：1)输入点云并进行序列化和位置编码；2)使用Mamba块处理位置感知特征序列，选择性传播或遗忘信息；3)通过MLP层和最大池化生成全局特征描述符；4)使用Lucas-Kanade算法迭代最小化源点和目标点云特征差异直至收敛；5)利用相机姿态生成匹配的源点和目标点云对。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)MambaNetLK框架，结合Mamba SSM和IC-LK实现长距离依赖建模；2)C3VD-Raycasting-10k临床数据集，提供10,014个视点匹配的点云对；3)高效捕获全局几何结构，克服MLP特征提取器局限；4)线性时间复杂度的长距离依赖建模。相比之前工作：避免了基于对应关系方法中的特征退化问题；比PointNetLK更好地捕获长距离几何依赖；比Transformer更高效且更适合手术应用；解决了缺乏3D配准基准数据集的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MambaNetLK通过整合Mamba状态空间模型和创建C3VD-Raycasting-10k临床数据集，显著提高了结肠镜检查中3D点云配准的准确性和鲁棒性，为微创手术中的实时临床导航系统奠定了坚实基础。'}


### 论文摘要

Accurate 3D point cloud registration underpins reliable image-guided colonoscopy, directly affecting lesion localization, margin assessment, and navigation safety. However, biological tissue exhibits repetitive textures and locally homogeneous geometry that cause feature degeneracy, while substantial domain shifts between pre-operative anatomy and intra-operative observations further degrade alignment stability. To address these clinically critical challenges, we introduce a novel 3D registration method tailored for endoscopic navigation and a high-quality, clinically grounded dataset to support rigorous and reproducible benchmarking. We introduce C3VD-Raycasting-10k, a large-scale benchmark dataset with 10,014 geometrically aligned point cloud pairs derived from clinical CT data. We propose MambaNetLK, a novel correspondence-free registration framework, which enhances the PointNetLK architecture by integrating a Mamba State Space Model (SSM) as a cross-modal feature extractor. As a result, the proposed framework efficiently captures long-range dependencies with linear-time complexity. The alignment is achieved iteratively using the Lucas-Kanade algorithm. On the clinical dataset, C3VD-Raycasting-10k, MambaNetLK achieves the best performance compared with the state-of-the-art methods, reducing median rotation error by 56.04% and RMSE translation error by 26.19% over the second-best method. The model also demonstrates strong generalization on ModelNet40 and superior robustness to initial pose perturbations. MambaNetLK provides a robust foundation for 3D registration in surgical navigation. The combination of a globally expressive SSM-based feature extractor and a large-scale clinical dataset enables more accurate and reliable guidance systems in minimally invasive procedures like colonoscopy.

---

## 35. A Novel Grouping-Based Hybrid Color Correction Algorithm for Color Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.02397v1](http://arxiv.org/abs/2511.02397v1)

**作者:** Kuo-Liang Chung, Ting-Chung Tang

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了一种基于分组的混合颜色校正算法用于彩色点云，根据点云重叠率自适应分组，并针对不同组别采用不同的颜色校正方法，通过大量测试验证了算法的有效性。

### 背景

彩色点云的颜色一致性校正是3D渲染和压缩应用中的基础且重要任务，而以往的颜色校正方法主要针对彩色图像，而非点云数据。

### 目的

提出一种基于分组的混合颜色校正算法，专门用于彩色点云的颜色一致性校正。

### 方法

1) 估计对齐后的源点云和目标点云之间的重叠率；2) 根据重叠率高低，将目标点分为两组(Gcl和Gmod)或三组(Gcl、Gmod和Gdist)；3) 对Gcl组使用K近邻双边插值方法；4) 对Gmod组使用结合KBI和直方图均衡化的方法；5) 对Gdist组使用直方图均衡化方法；6) 讨论算法的分组效应特性和消融研究。

### 主要发现

算法的颜色一致性校正效果已通过1086对测试彩色点云与最先进方法进行了验证，证明了其有效性。

### 结论

提出的基于分组的混合颜色校正算法能够有效实现彩色点云的颜色一致性校正，相关C++源代码已在GitHub平台公开。

### 翻译

彩色点云的颜色一致性校正是3D渲染和压缩应用中的一个基础而重要的任务。过去，大多数先前的颜色校正方法旨在校正彩色图像的颜色。本文的目的是提出一种基于分组的混合颜色校正算法用于彩色点云。我们的算法首先估计对齐后的源点云和目标点云之间的重叠率，然后根据估计的重叠率高低，自适应地将目标点分为两组，即近距离组Gcl和中距离组Gmod，或三组，即Gcl、Gmod和远距离组Gdist。为了校正Gcl中目标点的颜色，提出了一种基于K近邻的双边插值(KBI)方法。为了校正Gmod中目标点的颜色，提出了一种结合KBI和直方图均衡化(JKHE)的方法。对于Gdist中的目标点，提出了一种直方图均衡化(HE)方法进行颜色校正。最后，我们讨论了算法中无分组效应特性和消融研究。我们的算法在1086对测试彩色点云上与最先进方法进行了比较，证明了期望的颜色一致性校正效果。本算法的C++源代码可以从网站https://github.com/ivpml84079/Point-cloud-color-correction获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决彩色点云对齐后的颜色一致性问题。当两个彩色点云对齐时，由于采集设备参数、光照条件等因素差异，常会出现颜色不一致现象，导致视觉不协调。这个问题在3D视觉、自动驾驶、虚拟现实等领域非常重要，因为颜色不一致会影响点云压缩、人体姿态估计和点云渲染等应用的质量，最终影响合成3D场景的真实感和美观度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有颜色校正方法（如最近邻法、K近邻法、直方图匹配法等）的优缺点，发现这些方法主要针对彩色图像而非点云。作者借鉴了这些方法的基本思想，但针对点云特性进行了改进。作者观察到点云中点与周围点的关系不同于图像像素，因此考虑了点之间的空间距离关系，提出了基于重叠率的分组策略，并针对不同距离的组设计了专门的校正方法，从而形成了这个混合算法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是根据源点云和目标点云之间的重叠率，自适应地将目标点分为不同组，然后对不同组的点采用最适合的颜色校正方法。整体流程为：1)计算点云对齐的重叠率；2)根据重叠率决定将目标点分为两组或三组；3)对近距离组使用K近邻双边插值法校正颜色；4)对中等距离组使用联合K近邻双边插值法和直方图均衡化法校正颜色；5)对远距离组使用直方图均衡化法校正颜色；6)输出颜色校正后的结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出基于重叠率的自适应分组策略；2)针对不同距离组设计专门的校正方法；3)证明算法具有分组效应自由性质，确保组间边界颜色平滑；4)在大量点云对上验证了算法性能。相比之前工作，本文专注于点云而非图像，采用分组策略而非单一方法处理所有点，并在中等距离组中引入了动态权重调整机制，显著提高了颜色校正效果和视觉质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于分组混合的颜色校正算法，通过自适应分组和针对性校正方法，有效解决了彩色点云对齐后的颜色一致性问题，显著提升了合成3D场景的视觉质量。'}


### 论文摘要

Color consistency correction for color point clouds is a fundamental yet important task in 3D rendering and compression applications. In the past, most previous color correction methods aimed at correcting color for color images. The purpose of this paper is to propose a grouping-based hybrid color correction algorithm for color point clouds. Our algorithm begins by estimating the overlapping rate between the aligned source and target point clouds, and then adaptively partitions the target points into two groups, namely the close proximity group Gcl and the moderate proximity group Gmod, or three groups, namely Gcl, Gmod, and the distant proximity group Gdist, when the estimated overlapping rate is low or high, respectively. To correct color for target points in Gcl, a K-nearest neighbors based bilateral interpolation (KBI) method is proposed. To correct color for target points in Gmod, a joint KBI and the histogram equalization (JKHE) method is proposed. For target points in Gdist, a histogram equalization (HE) method is proposed for color correction. Finally, we discuss the grouping-effect free property and the ablation study in our algorithm. The desired color consistency correction benefit of our algorithm has been justified through 1086 testing color point cloud pairs against the state-of-the-art methods. The C++ source code of our algorithm can be accessed from the website: https://github.com/ivpml84079/Point-cloud-color-correction.

---

## 36. Self-Supervised Moving Object Segmentation of Sparse and Noisy Radar Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.02395v1](http://arxiv.org/abs/2511.02395v1)

**作者:** Leon Schwarzer, Matthias Zeller, Daniel Casado Herraez, Simon Dierl, Michael Heidingsfeld, Cyrill Stachniss

**发布时间:** 2025-11-04

**备注:** Accepted for publication at IEEE International Conference on  Intelligent Transportation Systems (ITSC 2025), 8 pages, 3 figures

### GPT解析

### 总结

这篇论文提出了一种自监督学习方法，用于稀疏和有噪声的雷达点云的运动目标分割。该方法采用两步式方法：首先使用基于聚类的对比自监督表示学习进行预训练，然后使用有限的标注数据进行监督微调。作者提出了一种新颖的基于聚类的对比损失函数，通过动态点移除进行聚类优化，使网络能够生成雷达数据的运动感知表示。通过自监督预训练，该方法在微调后提高了标签效率，并有效提升了最先进性能。

### 背景

运动目标分割对自动驾驶等自主移动系统的安全和可靠性至关重要，可以提高后续任务（如SLAM或路径规划）的可靠性和鲁棒性。虽然相机或LiDAR数据的分割已被广泛研究并取得良好成果，但通常需要累积时间序列来获得必要的时间上下文，从而增加了延迟。雷达传感器通过提供点的多普勒速度的直接测量值解决了单扫描运动目标分割的问题。然而，雷达点云通常稀疏且有噪声，使得在监督学习中使用的数据标注非常繁琐、耗时且成本高昂。

### 目的

解决雷达点云数据标注困难的问题，提出一种自监督方法，用于稀疏和有噪声的雷达点云的运动目标分割，减少对大量标注数据的依赖，提高分割效率和准确性。

### 方法

采用两步方法：1) 对比自监督表示学习：提出一种新颖的基于聚类的对比损失函数，通过动态点移除进行聚类优化，预训练网络以生成雷达数据的运动感知表示；2) 监督微调：使用有限的标注数据进行监督微调。

### 主要发现

所提出的方法在微调后提高了标签效率，通过自监督预训练有效提升了最先进性能。

### 结论

自监督学习方法可以有效解决雷达点云运动目标分割中标注数据不足的问题，通过预训练和微调的两步策略，能够在有限标注数据的情况下实现高性能的分割结果。

### 翻译

运动目标分割对于自动驾驶等自主移动系统的安全和可靠性至关重要，可以提高后续任务（如SLAM或路径规划）的可靠性和鲁棒性。虽然相机或LiDAR数据的分割已被广泛研究并取得良好成果，但通常需要累积时间序列来获得必要的时间上下文，从而增加了延迟。雷达传感器通过提供点的多普勒速度的直接测量值解决了这个问题，可用于单扫描运动目标分割。然而，雷达点云通常稀疏且有噪声，使得在监督学习中使用的数据标注非常繁琐、耗时且成本高昂。为解决这个问题，我们解决了稀疏和有噪声的雷达点云的自监督运动目标分割任务。我们采用对比自监督表示学习与后续使用有限标注数据的监督微调的两步方法。我们提出了一种新颖的基于聚类的对比损失函数，通过动态点移除进行聚类优化，预训练网络以生成雷达数据的运动感知表示。我们的方法在微调后提高了标签效率，通过自监督预训练有效提升了最先进性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决稀疏和有噪声的雷达点云上的移动物体分割问题。这个问题在现实中非常重要，因为移动物体分割对自动驾驶汽车等自主移动系统的安全和可靠性至关重要，能够提高后续任务如SLAM或路径规划的可靠性和鲁棒性。虽然相机和LiDAR数据的分割研究广泛，但它们需要积累时间序列数据，增加了系统延迟。雷达传感器可以通过多普勒速度直接测量实现单次扫描的移动物体分割，但雷达点云的稀疏性和噪声性使得数据标注非常困难、耗时且成本高昂。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的问题：相机和LiDAR需要时间序列增加延迟，雷达虽能单次扫描但数据质量差导致标注困难。作者借鉴了自监督学习思想，采用两步方法：先进行对比自监督表征学习预训练，再用有限标注数据监督微调。设计上采用学生-教师框架（自监督学习中常用），并基于雷达实例变换器（RIT）架构进行修改。核心创新是设计了新的对比损失函数，结合聚类和动态点移除（DPR）算法来生成伪标签，使网络学习运动感知的表征。作者还借鉴了LiDAR领域的自监督表征学习方法，但针对雷达数据的特性进行了调整。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用自监督学习减少对标注数据的依赖，设计专门的对比损失函数使网络学习能区分运动和静止物体的表征，结合空间信息和运动信息提高分割性能。整体流程包括：1)数据准备：使用包含坐标、多普勒速度和雷达横截面积的雷达点云；2)自监督预训练：构建学生-教师框架，使用HDBSCAN聚类，通过DPR算法细化聚类分离运动和静止点，计算对比损失拉近同类聚类、推远异类聚类；3)监督微调：添加MLP生成分割掩码，在少量标注数据上微调；4)评估：在View-of-Delft和RadarScenes数据集上用IoU指标评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的聚类对比损失函数，专门针对雷达数据设计，关注运动信息而非语义特征；2)自监督预训练框架首次应用于雷达点云的移动物体分割；3)两步训练方法提高标签效率。相比之前工作的不同：与相机/LiDAR方法不需时间序列减少延迟；与RaFlow等自监督方法相比在少量标注数据上表现更好；与现有监督方法在相同标注量下性能更高；与现有对比损失函数不同，专注于运动信息而非语义特征；结合聚类和算法方法生成伪标签而非依赖数据标注。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种新的自监督学习方法，通过设计专门的聚类对比损失函数，在少量标注数据的情况下显著提高了稀疏和噪声雷达点云上的移动物体分割性能。'}


### 论文摘要

Moving object segmentation is a crucial task for safe and reliable autonomous mobile systems like self-driving cars, improving the reliability and robustness of subsequent tasks like SLAM or path planning. While the segmentation of camera or LiDAR data is widely researched and achieves great results, it often introduces an increased latency by requiring the accumulation of temporal sequences to gain the necessary temporal context. Radar sensors overcome this problem with their ability to provide a direct measurement of a point's Doppler velocity, which can be exploited for single-scan moving object segmentation. However, radar point clouds are often sparse and noisy, making data annotation for use in supervised learning very tedious, time-consuming, and cost-intensive. To overcome this problem, we address the task of self-supervised moving object segmentation of sparse and noisy radar point clouds. We follow a two-step approach of contrastive self-supervised representation learning with subsequent supervised fine-tuning using limited amounts of annotated data. We propose a novel clustering-based contrastive loss function with cluster refinement based on dynamic points removal to pretrain the network to produce motion-aware representations of the radar data. Our method improves label efficiency after fine-tuning, effectively boosting state-of-the-art performance by self-supervised pretraining.

---

## 37. 3D Point Cloud Object Detection on Edge Devices for Split Computing

**论文链接:** [http://arxiv.org/abs/2511.02293v1](http://arxiv.org/abs/2511.02293v1)

**作者:** Taisuke Noguchi, Takuya Azumi

**发布时间:** 2025-11-04

**DOI:** 10.1109/RAGE62451.2024.00009

**备注:** 6 pages. This version includes minor lstlisting configuration  adjustments for successful compilation. No changes to content or layout.  Originally published at ACM/IEEE RAGE 2024

### GPT解析

### 总结

本研究利用分割计算技术解决了自动驾驶领域中深度学习模型在边缘设备上计算负担重、处理时间长和功耗高的问题，实验结果表明该方法能有效减少推理时间和边缘设备执行时间。

### 背景

自动驾驶技术领域正在快速发展，深度学习是其中的关键组成部分。特别是在感知领域，利用LiDAR收集的3D点云数据来运行深度神经网络模型进行3D物体检测。

### 目的

解决最先进的深度学习模型在边缘设备上处理时间长和功耗高的问题，通过分割计算减轻边缘设备的计算负担。

### 方法

采用分割计算，一种分布式机器学习推理方法，将计算任务分割处理，只传输深度神经网络模型的中间数据，从而减少边缘设备的计算负担和数据泄露风险。

### 主要发现

在体素化后分割，推理时间减少70.8%，边缘设备执行时间减少90.0%；在网络内分割，推理时间减少高达57.1%，边缘设备执行时间减少高达69.5%。

### 结论

分割计算能有效解决自动驾驶技术中边缘设备计算负担重的问题，显著减少处理时间和功耗，同时提高数据安全性。

### 翻译

自动驾驶技术领域正在快速发展，深度学习是其关键组成部分。特别是在感知领域，利用LiDAR收集的3D点云数据来运行深度神经网络模型进行3D物体检测。然而，这些最先进的模型复杂度高，导致边缘设备处理时间长和功耗增加。本研究旨在通过利用分割计算（一种分布式机器学习推理方法）来解决这些问题。分割计算旨在减轻边缘设备的计算负担，从而减少处理时间和功耗。此外，它仅传输深度神经网络模型的中间数据，从而最小化数据泄露风险。实验结果表明，在体素化后分割可将推理时间减少70.8%，边缘设备执行时间减少90.0%。在网络内分割时，推理时间可减少高达57.1%，边缘设备执行时间可减少高达69.5%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决边缘设备进行3D点云目标检测时计算负担过重的问题。在自动驾驶领域，LiDAR收集的3D点云数据需要通过复杂深度神经网络处理，但边缘设备计算能力有限，导致处理时间长、功耗高，影响实时性能和安全。这一问题重要是因为它关系到自动驾驶系统的可靠性和实用性，计算效率不足可能导致安全隐患，而轻量级模型又会降低检测准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了边缘设备处理复杂模型的局限性，考虑了轻量级模型和直接传输数据到服务器两种方案，但发现它们分别存在准确性和隐私问题。因此，作者借鉴了Split Computing这一分布式机器学习方法，将其应用于3D点云目标检测场景。作者使用了OpenPCDet作为检测工具箱，并参考了BottleFit等现有SC方法，同时选择了Voxel R-CNN作为基础模型，通过实验确定最佳分割点。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将深度神经网络模型在中间分割成两部分(head model和tail model)，head模型在边缘设备运行，tail模型在边缘服务器运行，只传输中间处理结果而非原始数据。实现流程为：1)边缘设备接收点云数据并预处理；2)模型在预设分割点处分割；3)边缘设备运行head模型；4)将head模型输出传输到边缘服务器；5)边缘服务器运行tail模型生成预测结果；6)将结果返回给边缘设备。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将Split Computing应用于3D点云目标检测；2)系统研究了不同分割点对性能的影响；3)提出选择分割点的两个关键标准(早期分割和最小输出数据大小)；4)通过实验验证了方法的有效性。相比之前工作，这种方法在保持高检测准确性的同时显著降低了边缘设备计算负担和执行时间，相比直接传输原始数据保护了隐私，相比轻量级模型维持了更好的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于Split Computing的3D点云目标检测方法，通过将深度学习模型分割并在边缘设备和边缘服务器间协同计算，显著降低了边缘设备的计算负担和执行时间，同时保护了数据隐私。'}


### 论文摘要

The field of autonomous driving technology is rapidly advancing, with deep learning being a key component. Particularly in the field of sensing, 3D point cloud data collected by LiDAR is utilized to run deep neural network models for 3D object detection. However, these state-of-the-art models are complex, leading to longer processing times and increased power consumption on edge devices. The objective of this study is to address these issues by leveraging Split Computing, a distributed machine learning inference method. Split Computing aims to lessen the computational burden on edge devices, thereby reducing processing time and power consumption. Furthermore, it minimizes the risk of data breaches by only transmitting intermediate data from the deep neural network model. Experimental results show that splitting after voxelization reduces the inference time by 70.8% and the edge device execution time by 90.0%. When splitting within the network, the inference time is reduced by up to 57.1%, and the edge device execution time is reduced by up to 69.5%.

---

## 38. LiDAR-VGGT: Cross-Modal Coarse-to-Fine Fusion for Globally Consistent and Metric-Scale Dense Mapping

**论文链接:** [http://arxiv.org/abs/2511.01186v1](http://arxiv.org/abs/2511.01186v1)

**作者:** Lijie Wang, Lianjie Guo, Ziyi Xu, Qianhao Wang, Fei Gao, Xieyuanli Chen

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种名为LiDAR-VGGT的新型框架，通过两阶段融合流程将激光雷达惯性里程计与VGGT模型紧密结合，实现了大规模彩色点云的有效重建，解决了现有方法在可扩展性和度量尺度方面的局限性。

### 背景

大规模彩色点云重建是机器人学中的重要任务，支持感知、导航和场景理解。然而，现有的激光雷达惯性视觉里程计(LIVO)对外部校准高度敏感，而3D视觉基础模型如VGGT在大规模环境中可扩展性有限且缺乏度量尺度。

### 目的

克服现有技术的局限性，提出一种能够生成密集、全局一致的彩色点云的新型框架。

### 方法

提出LiDAR-VGGT框架，通过两阶段由粗到细的融合流程紧密耦合激光雷达惯性里程计与VGGT模型。第一阶段采用预融合模块进行鲁棒的初始化细化，有效估计VGGT姿态和具有粗略度量尺度的点云；第二阶段通过后融合模块增强跨模态3D相似性变换，使用基于边界框的正则化减少传感器间视场不一致导致的尺度失真。

### 主要发现

在多个数据集上的实验表明，LiDAR-VGGT实现了密集、全局一致的彩色点云，性能优于基于VGGT的方法和LIVO基线。

### 结论

提出的LiDAR-VGGT框架有效解决了现有技术在点云重建中的局限性，并将新型彩色点云评估工具包实现为开源软件。

### 翻译

重建大规模彩色点云是机器人学中的重要任务，支持感知、导航和场景理解。尽管激光雷达惯性视觉里程计(LIVO)有所进步，但其性能对外部校准高度敏感。同时，3D视觉基础模型如VGGT在大规模环境中可扩展性有限且缺乏度量尺度。为克服这些限制，我们提出了LiDAR-VGGT，一种通过两阶段由粗到细的融合流程紧密耦合激光雷达惯性里程计与最先进的VGGT模型的新型框架：首先，具有鲁棒初始化细化的预融合模块有效估计了每个会话内具有粗略度量尺度的VGGT姿态和点云；然后，后融合模块增强跨模态3D相似性变换，使用基于边界框的正则化减少激光雷达和相机传感器之间视场不一致导致的尺度失真。在多个数据集上的大量实验表明，LiDAR-VGGT实现了密集、全局一致的彩色点云，性能优于基于VGGT的方法和LIVO基线。我们提出的新型彩色点云评估工具包实现将作为开源软件发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决两个问题：1) 传统LiDAR惯性视觉里程计(LIVO)方法对传感器外参校准敏感且点云稀疏；2) 3D视觉基础模型(如VGGT)在大环境中缺乏全局一致性和度量尺度。这个问题在机器人领域非常重要，因为准确的彩色点云重建是机器人感知、导航和场景理解的基础，对自主导航、多机器人协作和自动驾驶等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的优缺点：LIVO提供精确定位但对校准敏感且点云稀疏，VGGT能生成密集彩色点云但缺乏全局一致性和度量尺度。作者借鉴了LIVO的LiDAR-IMU融合获取真实尺度参考，借鉴VGGT的视觉几何变换生成密集重建，借鉴SLAM中的位图优化确保全局一致性。在此基础上，作者设计了新的两阶段粗到细融合框架：预融合模块使用LIVO初始化和校准VGGT，后融合模块通过增强的跨模态配准进一步优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将LiDAR-IMU视觉里程计与VGGT模型紧密耦合，利用LiDAR提供真实世界尺度信息解决VGGT缺乏度量尺度的问题，同时利用VGGT生成密集彩色点云的能力克服LiDAR点云稀疏的局限性。整体流程分为：1) 预融合模块：将长图像序列分成多个会话，独立使用VGGT处理，通过线性验证和尺度RANSAC精炼VGGT位姿，转换到世界坐标系；2) 后融合模块：使用基于边界框正则化的增强跨模态Sim(3)配准，将VGGT点云与LiDAR点云对齐，应用全局位图优化确保全局一致性；3) 彩色地图评估：提出四种评估指标评估重建质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 提出LiDAR-VGGT框架，首次将LiDAR与VGGT融合；2) 设计预融合模块，通过线性验证和尺度RANSAC精炼VGGT位姿；3) 引入基于边界框正则化的跨模态Sim(3)配准，解决视场不一致导致的尺度失真；4) 提出新的彩色点云评估工具。相比之前工作：1) 比纯LIVO方法生成更密集点云且对校准误差不那么敏感；2) 比纯VGGT方法提供真实度量尺度和更好全局一致性；3) 比其他融合方法直接处理RGB图像，效率更高；4) 专门评估彩色点云质量，不仅关注几何质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiDAR-VGGT通过创新的粗到细跨模态融合方法，成功将LiDAR的真实世界度量信息与VGGT的密集彩色重建能力相结合，实现了大规模、全局一致且具有准确度量尺度的彩色点云地图重建。'}


### 论文摘要

Reconstructing large-scale colored point clouds is an important task in robotics, supporting perception, navigation, and scene understanding. Despite advances in LiDAR inertial visual odometry (LIVO), its performance remains highly sensitive to extrinsic calibration. Meanwhile, 3D vision foundation models, such as VGGT, suffer from limited scalability in large environments and inherently lack metric scale. To overcome these limitations, we propose LiDAR-VGGT, a novel framework that tightly couples LiDAR inertial odometry with the state-of-the-art VGGT model through a two-stage coarse- to-fine fusion pipeline: First, a pre-fusion module with robust initialization refinement efficiently estimates VGGT poses and point clouds with coarse metric scale within each session. Then, a post-fusion module enhances cross-modal 3D similarity transformation, using bounding-box-based regularization to reduce scale distortions caused by inconsistent FOVs between LiDAR and camera sensors. Extensive experiments across multiple datasets demonstrate that LiDAR-VGGT achieves dense, globally consistent colored point clouds and outperforms both VGGT-based methods and LIVO baselines. The implementation of our proposed novel color point cloud evaluation toolkit will be released as open source.

---

## 39. GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies

**论文链接:** [http://arxiv.org/abs/2511.00998v1](http://arxiv.org/abs/2511.00998v1)

**作者:** Ziye Wang, Li Kang, Yiran Qin, Jiahua Ma, Zhanglin Peng, Lei Bai, Ruimao Zhang

**发布时间:** 2025-11-02

**备注:** Accepted by NeurIPS 2025. Project page:  https://ziyeeee.github.io/gaudp.io/

### GPT解析

### 总结

本文提出GauDP，一种新的高斯图像协同表示方法，用于在多智能体协作系统中实现可扩展的、感知感知的模仿学习。该方法通过从分散的RGB观测构建全局一致的3D高斯场，并动态将3D高斯属性重新分配给各智能体的局部视角，使智能体能够保持各自视角的同时从共享场景表示中查询任务关键特征。

### 背景

在具身多智能体系统中实现有效协调仍然是一个基本挑战，特别是在智能体必须平衡个体视角与全局环境感知的场景中。现有方法往往难以平衡细粒度的局部控制和全面的场景理解，导致可扩展性有限且协作质量受损。

### 目的

开发一种新的表示方法，使多智能体系统能够同时实现细粒度控制和全局连贯的行为，而无需额外的感知模式（如3D点云）。

### 方法

GauDP方法包括：1) 从分散的RGB观测构建全局一致的3D高斯场；2) 动态将3D高斯属性重新分配给每个智能体的局部视角；3) 使所有智能体能够从共享场景表示中自适应查询任务关键特征，同时保持各自的视角。

### 主要发现

在RoboFactory基准测试（包括多种多臂操作任务）上评估，GauDP方法比现有基于图像的方法表现出优越的性能，接近点云驱动方法的有效性，同时随着智能体数量的增加保持强大的可扩展性。

### 结论

GauDP提供了一种新的表示方法，能够在多智能体协作系统中实现细粒度控制和全局连贯的行为，无需额外的感知模式，且具有良好的可扩展性。

### 翻译

最近，在具身多智能体系统中实现有效协调仍然是一个基本挑战，特别是在智能体必须平衡个体视角与全局环境感知的场景中。现有方法往往难以平衡细粒度的局部控制和全面的场景理解，导致可扩展性有限且协作质量受损。在本文中，我们提出了GauDP，一种新的高斯图像协同表示方法，用于在多智能体协作系统中实现可扩展的、感知感知的模仿学习。具体来说，GauDP从分散的RGB观测构建全局一致的3D高斯场，然后动态将3D高斯属性重新分配给每个智能体的局部视角。这使得所有智能体能够在保持各自视角的同时从共享场景表示中自适应查询任务关键特征。这种设计实现了细粒度控制和全局连贯的行为，而无需额外的感知模式（如3D点云）。我们在RoboFactory基准测试上评估了GauDP，该测试包括多种多臂操作任务。我们的方法比现有基于图像的方法表现出优越的性能，接近点云驱动方法的有效性，同时随着智能体数量的增加保持强大的可扩展性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决具身多智能体系统中的有效协调问题，特别是在智能体需要平衡个体视角与全局环境感知的场景中。这个问题在现实中非常重要，因为许多实际应用（如工业装配、手术机器人和辅助家务任务）需要多个智能体协调工作。如果智能体之间不能有效协调，可能会导致碰撞或任务中断等灾难性失败。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了多智能体控制的两种主要范式：聚合所有智能体的局部观察或使用全局环境观察。发现第一种方法无法捕捉联合协作状态，第二种方法缺乏高分辨率细节。作者借鉴了3D高斯溅射技术用于3D场景重建，以及扩散策略框架用于动作生成。他们设计了一个统一的图像-高斯表示框架，通过3D高斯场构建全局一致表示，然后动态分配给各智能体。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过3D高斯表示融合局部和全局观察，使智能体在保持个体视角的同时，能从共享场景中查询任务关键特征。流程包括：1) 从各智能体的RGB图像构建全局3D高斯场；2) 动态将高斯属性重新分配给各智能体；3) 智能体从共享高斯表示中提取任务特征；4) 使用扩散策略基于融合特征预测动作。具体实现使用Noposplat网络重建3D高斯，通过交叉视图ViT解码器融合信息，并引入深度监督提高保真度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的图像-高斯表示框架；2) 动态表示选择机制；3) 选择性全局上下文分发；4) 像素级协同策略。相比之前工作，GauDP的不同之处在于：仅使用RGB输入就能达到接近点云方法的性能；能自然扩展到更多智能体；同时提供精细控制和全局一致行为；在RoboFactory基准测试中显著优于现有图像方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GauDP通过3D高斯-图像协同表示，使多智能体系统在仅使用RGB输入的情况下就能实现接近点云方法的协作性能，同时保持良好的可扩展性。'}


### 论文摘要

Recently, effective coordination in embodied multi-agent systems has remained a fundamental challenge, particularly in scenarios where agents must balance individual perspectives with global environmental awareness. Existing approaches often struggle to balance fine-grained local control with comprehensive scene understanding, resulting in limited scalability and compromised collaboration quality. In this paper, we present GauDP, a novel Gaussian-image synergistic representation that facilitates scalable, perception-aware imitation learning in multi-agent collaborative systems. Specifically, GauDP constructs a globally consistent 3D Gaussian field from decentralized RGB observations, then dynamically redistributes 3D Gaussian attributes to each agent's local perspective. This enables all agents to adaptively query task-critical features from the shared scene representation while maintaining their individual viewpoints. This design facilitates both fine-grained control and globally coherent behavior without requiring additional sensing modalities (e.g., 3D point cloud). We evaluate GauDP on the RoboFactory benchmark, which includes diverse multi-arm manipulation tasks. Our method achieves superior performance over existing image-based methods and approaches the effectiveness of point-cloud-driven methods, while maintaining strong scalability as the number of agents increases.

---

## 40. Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow

**论文链接:** [http://arxiv.org/abs/2511.00977v1](http://arxiv.org/abs/2511.00977v1)

**作者:** Kristiyan Sakalyan, Alessandro Palma, Filippo Guerranti, Fabian J. Theis, Stephan Günnemann

**发布时间:** 2025-11-02

**备注:** 37 pages, 15 figures, to appear in NeurIPS 2025

### GPT解析

### 总结

研究介绍了一种名为NicheFlow的基于流的生成模型，用于推断细胞微环境在时空数据中的演化轨迹。

### 背景

理解时空数据中细胞微环境的演化对于解析组织发育和疾病进展至关重要。当前模拟细胞进化的方法在单细胞水平上操作，忽略了组织中细胞状态的协调发育。

### 目的

开发一种能够推断细胞微环境在连续空间切片上的时间轨迹的方法，克服现有单细胞水平方法的局限性。

### 方法

NicheFlow是一种基于流的生成模型，通过将局部细胞邻域表示为点云，使用最优传输和变分流匹配联合建模细胞状态和空间坐标的演化。

### 主要发现

NicheFlow成功从多样化的时空数据集中恢复了全局空间架构和局部微环境组成，从胚胎发育到大脑发育。

### 结论

NicheFlow能够有效地建模细胞微环境的时空演化，为理解组织发育和疾病进展提供了新的工具。

### 翻译

理解时空数据中细胞微环境的演化对于解析组织发育和疾病进展至关重要。虽然像空间转录组学这样的实验技术现在能够在时空上实现组织组织的高分辨率映射，但当前模拟细胞进化的方法在单细胞水平上操作，忽略了组织中细胞状态的协调发育。我们介绍了NicheFlow，一种基于流的生成模型，用于推断细胞微环境在连续空间切片上的时间轨迹。通过将局部细胞邻域表示为点云，NicheFlow使用最优传输和变分流匹配联合建模细胞状态和空间坐标的演化。我们的方法成功从多样化的时空数据集中恢复了全局空间架构和局部微环境组成，从胚胎发育到大脑发育。


### 论文摘要

Understanding the evolution of cellular microenvironments in spatiotemporal data is essential for deciphering tissue development and disease progression. While experimental techniques like spatial transcriptomics now enable high-resolution mapping of tissue organization across space and time, current methods that model cellular evolution operate at the single-cell level, overlooking the coordinated development of cellular states in a tissue. We introduce NicheFlow, a flow-based generative model that infers the temporal trajectory of cellular microenvironments across sequential spatial slides. By representing local cell neighborhoods as point clouds, NicheFlow jointly models the evolution of cell states and spatial coordinates using optimal transport and Variational Flow Matching. Our approach successfully recovers both global spatial architecture and local microenvironment composition across diverse spatiotemporal datasets, from embryonic to brain development.

---

## 41. URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model

**论文链接:** [http://arxiv.org/abs/2511.00940v1](http://arxiv.org/abs/2511.00940v1)

**作者:** Zhe Li, Xiang Bai, Jieyu Zhang, Zhuangzhe Wu, Che Xu, Ying Li, Chengkai Hou, Shanghang Zhang

**发布时间:** 2025-11-02

**备注:** Accepted to the 39th Conference on Neural Information Processing  Systems (NeurIPS 2025)

### GPT解析

### 总结

本文提出了一种名为URDF-Anything的端到端自动重建框架，基于3D多模态大语言模型，用于构建关节物体的精确数字孪生，显著提高了几何分割、运动学参数预测和物理执行能力。

### 背景

构建关节物体的精确数字孪生对于机器人模拟训练和具身AI世界模型构建至关重要，但传统方法需要繁琐的手动建模或多阶段流程。

### 目的

开发一种端到端的自动重建框架，简化关节物体数字孪生的构建过程，提高其准确性和效率。

### 方法

提出URDF-Anything框架，基于3D多模态大语言模型，利用点云和文本多模态输入的自回归预测框架联合优化几何分割和运动学参数预测，实现专门的[SEG]令牌机制与点云特征交互。

### 主要发现

实验表明该方法在几何分割(mIoU提高17%)、运动学参数预测(平均误差减少29%)和物理执行能力(比基线提高50%)方面显著优于现有方法，且表现出优秀的泛化能力。

### 结论

URDF-Anything为构建机器人模拟的数字孪生提供了高效解决方案，显著提高了模拟到现实的迁移能力。

### 翻译

构建关节物体的精确数字孪生对于机器人模拟训练和具身AI世界模型构建至关重要，但传统上需要繁琐的手动建模或多阶段流程。在这项工作中，我们提出了URDF-Anything，一种基于3D多模态大语言模型的端到端自动重建框架。URDF-Anything利用基于点云和文本多模态输入的自回归预测框架，联合优化几何分割和运动学参数预测。它实现了一个专门的[SEG]令牌机制，直接与点云特征交互，实现细粒度的部件级分割，同时保持与运动学参数预测的一致性。在模拟和真实世界数据集上的实验表明，我们的方法在几何分割(mIoU提高17%)、运动学参数预测(平均误差减少29%)和物理执行能力(比基线提高50%)方面显著优于现有方法。值得注意的是，我们的方法表现出优秀的泛化能力，即使在训练集外的物体上也能良好表现。这项工作为构建机器人模拟的数字孪生提供了高效解决方案，显著提高了模拟到现实的迁移能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决铰接物体（如门、抽屉、剪刀等具有内部连接结构的物体）的数字孪生自动重建问题。传统方法需要繁琐的手动建模或多阶段处理流程，而本文提出的方法可以直接从视觉输入（单视图或多视图图像）自动生成功能性的URDF模型。这个问题在机器人仿真训练、具身AI世界模型构建、自动驾驶和交互式虚拟/增强现实环境中至关重要，因为这些应用需要精确的物体表示来进行准确的物理模拟和交互。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：之前的铰接物体重建方法要么依赖给定的网格资产库，要么涉及单独的部件分割阶段，无法实现端到端的处理。作者设计了一个基于3D多模态大语言模型（MLLM）的框架，利用其处理多模态输入的能力、大规模预训练获取的3D形状先验以及直接理解空间关系的能力。该方法借鉴了ShapeLLM作为骨干网络，并创新性地应用了LISA中的[SEG]标记机制来实现符号铰接结构与几何分割的同步预测。在点云生成方面，作者使用了DUSt3R（多视图）和LGM（单视图）等现有方法，但在整体框架上进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D多模态大语言模型的能力，通过特殊的[SEG]标记机制实现符号铰接结构与几何分割的联合预测，确保预测的运动学与重建几何之间的一致性。整体流程包括：1)输入表示：将视觉输入转换为3D点云；2)多模态铰接解析：3D MLLM联合预测部件分割和运动学参数；3)几何分割：通过[SEG]标记机制对点云进行精细分割；4)网格转换和URDF生成：将分割结果和运动学参数整合为标准URDF文件。这种方法实现了从原始视觉输入到可直接用于物理模拟的完整URDF模型的端到端处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个用于铰接物体重建的端到端3D MLLM框架；2)[SEG]标记机制实现几何分割和运动学参数的深度耦合与联合预测；3)端到端训练确保几何与运动学之间的一致性；4)强大的泛化能力，在训练集外物体上表现优异。相比之前的工作，本文直接使用原始3D点云作为输入而非简化表示（如OBB），采用端到端处理而非多阶段流水线，同时预测几何和运动学参数而非分别处理，并通过创新机制确保两者之间的一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了URDF-Anything，一种基于3D多模态大语言模型的端到端框架，能够从视觉输入自动重建铰接物体的功能URDF数字孪生，实现了几何分割和运动学参数的高精度联合预测，显著提升了模拟到现实转换的能力。'}


### 论文摘要

Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.

---

## 42. Persistence-Based Statistics for Detecting Structural Changes in High-Dimensional Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.00938v1](http://arxiv.org/abs/2511.00938v1)

**作者:** Toshiyuki Nakayama

**发布时间:** 2025-11-02

**备注:** 42 pages, 3 figures, under review

### GPT解析

### 总结

这篇论文研究了持久性统计在分布变化下的概率行为，并提出了一种用于检测高维随机点云结构变化的新非参数框架。

### 背景

持久性统计在拓扑数据分析中用于研究点云数据的结构特性，但在分布变化下的行为尚需深入研究。

### 目的

建立持久性统计在一般分布下的理论性质，并开发一种能够检测高维点云结构变化的统计方法。

### 方法

建立经典持久性统计量（总持久性和最大持久性）的矩界和紧性结果，引入基于持久性景观与Jensen-Shannon散度的标准化统计量，并证明其Hölder连续性。

### 主要发现

持久性统计在高斯混合模型中具有明确的尺度行为，所提出的统计量具有稳定性、尺度和位移不变性，能够通过置换测试进行非参数推断。

### 结论

该方法能够捕获制度转变和演化的几何复杂性，为随机持久性的理论理解和复杂高维系统中拓扑变化的检测提供了严格的统计基础。

### 翻译

我们研究了持久性统计在分布变化下的概率行为，并提出了一种用于检测高维随机点云结构变化的新非参数框架。我们首先在一般分布下建立了经典持久性统计量（总持久性和最大持久性）的矩界和紧性结果，并为高斯混合模型推导了明确的尺度行为。基于这些理论基础，我们引入了一种结合持久性景观和Jensen-Shannon散度的标准化统计量，并证明了它相对于输入点云扰动的Hölder连续性。所得的测度是稳定的、尺度和位移不变的，并通过置换测试适合非参数推断。使用去中心化治理数据的动态属性向量进行的数值说明展示了所提出的方法如何能够捕获制度转变和演化的几何复杂性。我们的结果为随机持久性的理论理解做出了贡献，并为复杂高维系统中拓扑变化的检测提供了严格的统计基础。


### 论文摘要

We study the probabilistic behavior of persistence statistics under distributional variability and propose a novel nonparametric framework for detecting structural changes in high-dimensional random point clouds. We first establish moment bounds and tightness results for classical persistence statistics - total and maximum persistence - under general distributions, with explicit scaling behavior derived for Gaussian mixture models. Building on these theoretical foundations, we introduce a normalized statistic based on persistence landscapes combined with the Jensen-Shannon divergence, and we prove its Holder continuity with respect to perturbations of input point clouds. The resulting measure is stable, scale- and shift-invariant, and suitable for nonparametric inference via permutation testing. A numerical illustration using dynamic attribute vectors from decentralized governance data demonstrates how the proposed method can capture regime shifts and evolving geometric complexity. Our results contribute to the theoretical understanding of random persistence and provide a rigorous statistical foundation for topological change-point detection in complex, high-dimensional systems.

---

## 43. Neural Green's Functions

**论文链接:** [http://arxiv.org/abs/2511.01924v1](http://arxiv.org/abs/2511.01924v1)

**作者:** Seungwoo Yoo, Kyeongmin Yeo, Jisung Hwang, Minhyuk Sung

**发布时间:** 2025-11-02

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文介绍了一种名为'Neural Green's Function'的神经网络解决方案算子，用于求解具有特征分解的线性偏微分方程。

### 背景

Green函数是线性偏微分方程的解算子，它们仅依赖于域的几何形状。

### 目的

设计一个能够模仿Green函数行为的神经网络，实现在不同不规则几何形状和源函数及边界条件下的泛化能力。

### 方法

Neural Green's Function从表示问题域的体积点云中提取逐点特征，并使用它们来预测解算子的分解，然后通过数值积分评估解。

### 主要发现

该框架对训练中使用的特定函数不敏感，能够实现稳健高效的泛化。在MCB数据集上对机械零件几何形状的稳态热分析中，Neural Green's Function优于最先进的神经算子，在五个形状类别上平均误差减少了13.9%，比需要计算密集网格化的数值求解器快350倍。

### 结论

Neural Green's Function是一种有效的神经网络解决方案算子，能够处理线性偏微分方程，并在不同几何形状和条件下实现良好的泛化性能。

### 翻译

我们引入了神经格林函数，一种用于线性偏微分方程的神经网络解算子，其微分算子允许特征分解。受格林函数的启发，线性偏微分方程的解算子仅依赖于域的几何形状，我们设计了神经格林函数来模仿它们的行为，实现在各种不规则几何形状、源函数和边界函数上的优越泛化能力。具体而言，神经格林函数从表示问题域的体积点云中提取逐点特征，并使用它们来预测解算子的分解，随后通过数值积分应用来评估解。与最近基于学习的解算子不同，这些解算子通常难以泛化到未见过的源函数或边界函数，我们的框架在设计上对训练中使用的特定函数不敏感，能够实现稳健和高效的泛化。在MCB数据集中机械零件几何形状的稳态热分析中，神经格林函数优于最先进的神经算子，在五个形状类别上平均误差减少了13.9%，而比需要计算密集网格化的数值求解器快350倍。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决线性偏微分方程(PDEs)求解的问题，特别是那些微分算子可以进行特征分解的线性PDEs（如泊松方程和双调和方程）。这个问题在现实和研究中非常重要，因为PDE在科学和工程领域有广泛应用，包括热分析、静电学、流体动力学和弹性力学等。传统数值求解方法（如有限元法）依赖于计算密集型的网格生成过程，这限制了在工程设计早期阶段的快速迭代评估。现有学习求解器虽然不需要网格，但往往难以推广到未见过的源函数和边界函数，当问题域、源函数和边界函数同时变化时表现不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到Green函数的启发，Green函数是线性PDE的解算子，仅依赖于问题域的几何形状而不依赖于特定的源函数或边界函数。作者基于线性PDE解的数学表达式，利用微分算子的特征分解性质，将解算子表示为特征向量和特征值的乘积。设计神经网络仅从域几何中提取特征，而不依赖于特定的源函数或边界函数。作者借鉴了Green函数理论、神经算子（如GNO、FNO）以及Transolver的网络架构，同时扩展了先前学习Green函数的工作（如Boullé和Teng等人的工作），使其能够处理更复杂的几何形状。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个仅从问题域几何中提取特征的神经网络，使用这些特征近似Green函数的特征分解，预测必要的微分量（如质量矩阵）以进行数值积分，从而实现解的计算。整体实现流程如下：1) 输入表示问题域几何的查询点；2) 使用神经网络（基于Transolver架构）从查询点坐标提取特征；3) 使用提取的特征构造神经Green函数，作为真实Green函数的近似；4) 预测每个顶点的质量值和算子的子矩阵；5) 通过基于Green函数的数值积分公式计算解；6) 通过最小化预测解与真实解之间的误差以及质量预测的正则化项来优化网络参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 仅从域几何提取特征，不将源函数和边界函数作为输入，使模型能够推广到未见过的函数；2) 利用Green函数的特征分解性质设计更有效的表示；3) 预测必要的微分量（如质量矩阵），使方法能够处理复杂几何形状；4) 专注于可以特征分解的线性PDE，如泊松方程和双调和方程。相比之前的工作，该方法比传统数值求解器快350倍（不需要网格生成），比PINNs不需要为每个问题实例重新训练，比神经算子（如GNO、FNO）能够更好地推广到未见过的源函数和边界函数，比先前学习Green函数的工作能够处理更复杂的几何形状且不需要为每个域重新训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种神经Green函数方法，通过仅从问题域几何中提取特征并近似Green函数的特征分解，实现了对线性偏微分方程的高效求解，能够推广到各种不规则几何形状和未见过的源函数、边界函数，且比传统数值求解器快350倍。'}


### 论文摘要

We introduce Neural Green's Function, a neural solution operator for linear partial differential equations (PDEs) whose differential operators admit eigendecompositions. Inspired by Green's functions, the solution operators of linear PDEs that depend exclusively on the domain geometry, we design Neural Green's Function to imitate their behavior, achieving superior generalization across diverse irregular geometries and source and boundary functions. Specifically, Neural Green's Function extracts per-point features from a volumetric point cloud representing the problem domain and uses them to predict a decomposition of the solution operator, which is subsequently applied to evaluate solutions via numerical integration. Unlike recent learning-based solution operators, which often struggle to generalize to unseen source or boundary functions, our framework is, by design, agnostic to the specific functions used during training, enabling robust and efficient generalization. In the steady-state thermal analysis of mechanical part geometries from the MCB dataset, Neural Green's Function outperforms state-of-the-art neural operators, achieving an average error reduction of 13.9\% across five shape categories, while being up to 350 times faster than a numerical solver that requires computationally expensive meshing.

---

## 44. Benchmarking individual tree segmentation using multispectral airborne laser scanning data: the FGI-EMIT dataset

**论文链接:** [http://arxiv.org/abs/2511.00653v1](http://arxiv.org/abs/2511.00653v1)

**作者:** Lassi Ruoppa, Tarmo Hietala, Verneri Seppänen, Josef Taher, Teemu Hakala, Xiaowei Yu, Antero Kukko, Harri Kaartinen, Juha Hyyppä

**发布时间:** 2025-11-01

**备注:** 39 pages, 9 figures

### GPT解析

### 总结

本研究引入了FGI-EMIT，首个用于单木分割的大规模多谱段机载激光扫描基准数据集，并比较了传统无监督算法与深度学习方法在树木分割任务上的性能表现。

### 背景

单木分割(LiDAR点云)是森林资源清查、碳监测和生物多样性评估的基础应用。传统方法采用无监督几何算法，近期转向监督深度学习。过去因缺乏大规模基准数据集限制了方法发展，尽管多光谱反射率能提高分割准确性，但多光谱LiDAR数据格式仍然有限。

### 目的

创建首个用于单木分割的大规模多谱段机载激光扫描基准数据集，并全面评估不同算法的性能。

### 方法

FGI-EMIT数据集在532、905和1,550 nm波长处捕获，包含1,561个手动标注的树木，特别关注小林下树木。研究评估了四种传统无监督算法和四种监督深度学习方法，其中无监督方法使用贝叶斯优化超参数，深度学习模型从头训练。

### 主要发现

无监督方法中Treeiso表现最佳，F1分数为52.7%；深度学习方法整体显著更好，最佳模型ForestFormer3D达到73.3%的F1分数。林下树木性能差异最显著，ForestFormer3D比Treeiso高出25.9个百分点。当前深度学习方法未能有效利用多谱段反射率信息，但单通道反射率可略微提高准确性，特别是对林下树木。即使点密度低至10点/m²，深度学习方法仍优于无监督算法。

### 结论

深度学习方法在树木分割任务上显著优于传统无监督方法，多光谱数据有提高分割准确性的潜力，但当前深度学习模型未能充分利用这些信息。即使在低点密度条件下，深度学习方法也保持优势。

### 翻译

从激光雷达点云中进行单木分割(ITS)对于森林资源清查、碳监测和生物多样性评估等应用至关重要。传统上，ITS通过无监督的几何算法实现，而最近的进展已转向监督深度学习(DL)。过去，由于缺乏大规模基准数据集，方法开发进展受限，尽管有证据表明多光谱(MS)反射率可以提高ITS的准确性，但新型数据格式(特别是多光谱激光雷达)的可用性至今仍然有限。本研究引入了FGI-EMIT，这是首个用于ITS的大规模多谱段机载激光扫描基准数据集。该数据集在532、905和1,550 nm波长处捕获，包含1,561个手动标注的树木，特别关注小林下树木。利用FGI-EMIT，我们全面评估了四种传统无监督算法和四种监督深度学习方法。无监督方法的超参数使用贝叶斯方法优化，而深度学习模型从头开始训练。在无监督方法中，Treeiso实现了最高的测试集F1分数，为52.7%。深度学习方法整体表现显著更好，最佳模型ForestFormer3D达到了73.3%的F1分数。林下树木观察到最显著的差异，ForestFormer3D比Treeiso高出25.9个百分点。消融研究表明，当多谱段反射率作为额外输入特征提供时，当前的基于深度学习的方法通常无法有效利用这些信息，尽管单通道反射率可以略微提高准确性，特别是对于林下树木。在不同点密度下的性能分析进一步表明，即使点密度低至10点/m²，深度学习方法仍然始终优于无监督算法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决个体树木分割(ITS)方法的评估和比较问题，特别是缺乏大规模多光谱激光扫描基准数据集的挑战。这个问题在现实中很重要，因为准确的树木分割是林业调查、碳监测和生物多样性评估的基础应用，而缺乏标准化的评估框架使得研究人员和从业者难以选择最适合特定应用的方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有数据集的局限性，如缺乏多光谱数据和高质量3D标注，然后设计了FGI-EMIT数据集。他们借鉴了FOR-Instance等数据集的经验，但增加了多光谱数据和更详细的标注；在评估方法上借鉴了计算机视觉领域的3D IoU指标；在超参数优化上使用了贝叶斯优化；在数据分割上采用了分层随机采样确保代表性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个大规模、高质量的多光谱激光扫描数据集，用于系统性地评估传统无监督算法和深度学习方法。整体流程包括：1)使用多波长激光扫描仪采集数据；2)预处理和合并数据；3)手动标注树木实例和语义信息；4)计算树木位置和高度；5)将树木分为四种类别；6)分割数据集为训练集和测试集；7)使用3D IoU指标评估多种方法的性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建了第一个大规模多光谱激光扫描个体树木分割基准数据集FGI-EMIT；2)进行了全面的性能比较，评估了四种传统方法和四种深度学习方法；3)首次研究了多光谱信息在深度学习方法中的利用。相比之前的工作，FGI-EMIT是第一个包含城市环境人工结构的多光谱数据集，并且提供了系统性的超参数优化和多种点云密度下的性能评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过创建首个大规模多光谱激光扫描个体树木分割基准数据集并进行全面的性能比较，证明了深度学习方法显著优于传统无监督算法，同时揭示了当前深度学习方法未能充分利用多光谱信息的局限性。'}


### 论文摘要

Individual tree segmentation (ITS) from LiDAR point clouds is fundamental for applications such as forest inventory, carbon monitoring and biodiversity assessment. Traditionally, ITS has been achieved with unsupervised geometry-based algorithms, while more recent advances have shifted toward supervised deep learning (DL). In the past, progress in method development was hindered by the lack of large-scale benchmark datasets, and the availability of novel data formats, particularly multispectral (MS) LiDAR, remains limited to this day, despite evidence that MS reflectance can improve the accuracy of ITS. This study introduces FGI-EMIT, the first large-scale MS airborne laser scanning benchmark dataset for ITS. Captured at wavelengths 532, 905, and 1,550 nm, the dataset consists of 1,561 manually annotated trees, with a particular focus on small understory trees. Using FGI-EMIT, we comprehensively benchmarked four conventional unsupervised algorithms and four supervised DL approaches. Hyperparameters of unsupervised methods were optimized using a Bayesian approach, while DL models were trained from scratch. Among the unsupervised methods, Treeiso achieved the highest test set F1-score of 52.7%. The DL approaches performed significantly better overall, with the best model, ForestFormer3D, attaining an F1-score of 73.3%. The most significant difference was observed in understory trees, where ForestFormer3D exceeded Treeiso by 25.9 percentage points. An ablation study demonstrated that current DL-based approaches generally fail to leverage MS reflectance information when it is provided as additional input features, although single channel reflectance can improve accuracy marginally, especially for understory trees. A performance analysis across point densities further showed that DL methods consistently remain superior to unsupervised algorithms, even at densities as low as 10 points/m$^2$.

---

## 45. Been There, Scanned That: Nostalgia-Driven LiDAR Compression for Self-Driving Cars

**论文链接:** [http://arxiv.org/abs/2511.00652v1](http://arxiv.org/abs/2511.00652v1)

**作者:** Ali Khalid, Jaiaid Mobin, Sumanth Rao Appala, Avinash Maurya, Stephany Berrio Perez, M. Mustafa Rafique, Fawad Ahmad

**发布时间:** 2025-11-01

### GPT解析

### 总结

本文提出了一种名为DejaView的3D点云数据压缩方法，通过利用自动驾驶车辆在大时间尺度上的数据冗余，实现了高效的数据压缩。

### 背景

自动驾驶车辆每天可产生数TB传感器数据，其中大量是由LiDAR等深度传感器生成的3D点云数据。这些数据需传输到云端用于机器学习模型训练或事故分析，但网络和存储成本高昂。

### 目的

减少自动驾驶车辆3D点云数据的网络传输和存储成本，提高数据压缩效率。

### 方法

DejaView利用自动驾驶车辆活动区域有限且主要行驶固定路线的特点，在更大的时间尺度（天和月）上寻找数据冗余，而非传统的帧间冗余。其核心是一个diff操作，将点云紧凑地表示为相对于过去3D数据的增量。

### 主要发现

使用两个月LiDAR数据的测试表明，DejaView的端到端实现可将点云压缩210倍，同时保持仅15厘米的重构误差。

### 结论

DejaView是一种有效的3D点云数据压缩方法，特别适用于自动驾驶车辆场景，能够显著降低数据存储和传输成本。

### 翻译

自动驾驶车辆每天可以产生数TB的传感器数据。其中很大一部分是由深度传感器（如LiDAR）产生的3D点云数据。这些数据必须传输到云存储，用于训练机器学习模型或进行分析，例如在发生事故时进行取证调查。为了减少网络和存储成本，本文介绍了DejaView。尽管先前的工作使用帧间冗余来压缩数据，但DejaView在更大的时间尺度（天和月）上搜索并利用冗余，以实现更有效的压缩。我们基于自动驾驶车辆的活动区域有限且主要每天行驶相同路线的洞察设计了DejaView。因此，车辆每天收集的3D数据可能与过去捕获的数据相似。为此，DejaView的核心是一个diff操作，将点云紧凑地表示为相对于过去3D数据的增量。使用两个月的LiDAR数据，DejaView的端到端实现可以在仅15厘米的重构误差下将点云压缩210倍。


### 论文摘要

An autonomous vehicle can generate several terabytes of sensor data per day. A significant portion of this data consists of 3D point clouds produced by depth sensors such as LiDARs. This data must be transferred to cloud storage, where it is utilized for training machine learning models or conducting analyses, such as forensic investigations in the event of an accident. To reduce network and storage costs, this paper introduces DejaView. Although prior work uses interframe redundancies to compress data, DejaView searches for and uses redundancies on larger temporal scales (days and months) for more effective compression. We designed DejaView with the insight that the operating area of autonomous vehicles is limited and that vehicles mostly traverse the same routes daily. Consequently, the 3D data they collect daily is likely similar to the data they have captured in the past. To capture this, the core of DejaView is a diff operation that compactly represents point clouds as delta w.r.t. 3D data from the past. Using two months of LiDAR data, an end-to-end implementation of DejaView can compress point clouds by a factor of 210 at a reconstruction error of only 15 cm.

---

## 46. Three-dimensional narrow volume reconstruction method with unconditional stability based on a phase-field Lagrange multiplier approach

**论文链接:** [http://arxiv.org/abs/2511.00508v1](http://arxiv.org/abs/2511.00508v1)

**作者:** Renjun Gao, Xiangjie Kong, Dongting Cai, Boyi Fu, Junxiang Yang

**发布时间:** 2025-11-01

**备注:** Preprint, 30+ pages; multiple figures and tables; code and data:  https://github.com/cfdyang521/C-3PO/tree/main; intended for submission to a  computational mathematics journal

### GPT解析

### 总结

提出了一种基于Allen-Cahn模型的有效点云重建算法，采用拉格朗日乘子法，通过增强的边缘检测函数重建窄壳结构。

### 背景

从点云重建物体在假肢、医学成像、计算机视觉等领域非常重要。

### 目的

开发一种有效的基于Allen-Cahn模型的重建算法。

### 方法

采用拉格朗日乘子法，利用物体的散乱数据点，通过求解增强有边缘检测函数的控制方程来重建窄壳；边缘检测函数基于无符号距离函数设计；使用Crank-Nicolson时间离散化和有限差分法近似空间运算。

### 主要发现

算法可以稳定和解耦地更新解；全离散方案被证明是无条件稳定的；复杂3D体积重建实验验证了算法的准确性、稳定性和有效性。

### 结论

分析了特定参数选择如何影响重建体积的细节和精细度；分享了计算代码和数据以便其他研究者理解和使用该算法。

### 翻译

从点云重建物体在假肢、医学成像、计算机视觉等领域至关重要。我们提出了一种基于Allen-Cahn模型的有效重建算法，采用拉格朗日乘子法。利用物体的散乱数据点，我们通过求解增强有从无符号距离函数导出的边缘检测函数的控制方程来重建窄壳。特别设计的边缘检测函数确保了能量稳定性。通过拉格朗日乘子技术重新表述控制方程并实施Crank-Nicolson时间离散化，我们可以以稳定和解耦的方式更新解。空间运算使用有限差分法近似，我们通过分析证明了全离散方案的无条件稳定性。包括重建《星球大战》中的字符等复杂3D体积在内的全面数值实验，验证了算法的准确性、稳定性和有效性。此外，我们分析了特定参数选择如何影响重建体积的细节和精细度。为了便于感兴趣的读者理解我们的算法，我们在https://github.com/cfdyang521/C-3PO/tree/main分享了计算代码和数据。


### 论文摘要

Reconstruction of an object from points cloud is essential in prosthetics, medical imaging, computer vision, etc. We present an effective algorithm for an Allen--Cahn-type model of reconstruction, employing the Lagrange multiplier approach. Utilizing scattered data points from an object, we reconstruct a narrow shell by solving the governing equation enhanced with an edge detection function derived from the unsigned distance function. The specifically designed edge detection function ensures the energy stability. By reformulating the governing equation through the Lagrange multiplier technique and implementing a Crank--Nicolson time discretization, we can update the solutions in a stable and decoupled manner. The spatial operations are approximated using the finite difference method, and we analytically demonstrate the unconditional stability of the fully discrete scheme. Comprehensive numerical experiments, including reconstructions of complex 3D volumes such as characters from \textit{Star Wars}, validate the algorithm's accuracy, stability, and effectiveness. Additionally, we analyze how specific parameter selections influence the level of detail and refinement in the reconstructed volumes. To facilitate the interested readers to understand our algorithm, we share the computational codes and data in https://github.com/cfdyang521/C-3PO/tree/main.

---

## 47. A Multimodal Dataset for Indoor Radio Mapping with 3D Point Clouds and RSSI

**论文链接:** [http://arxiv.org/abs/2511.00494v1](http://arxiv.org/abs/2511.00494v1)

**作者:** Ljupcho Milosheski, Kuon Akiyama, Blaž Bertalanič, Jernej Hribar, Ryoichi Shinkuma

**发布时间:** 2025-11-01

**备注:** 11 pages, 7 figures, 3 tables, under review to Nature Scientific Data

### GPT解析

### 总结

本文介绍了一个多模态数据集，结合高分辨率3D激光雷达扫描与Wi-Fi接收信号强度测量，用于研究室内无线信号传播特性，特别是在不同AP配置和人员存在情况下的动态环境影响。

### 背景

随着支持带宽密集型和延迟敏感型应用的智能设备增多，室内环境需要可靠的无线连接。准确的无线电环境图(REMs)估计对自适应网络规划和接入点优化至关重要。

### 目的

克服室内空间复杂性导致的真实REMs生成挑战，为数据驱动的无线建模研究提供资源，特别是在IEEE 802.11be(Wi-Fi 7)等新兴高频标准的背景下，促进高容量室内通信系统发展。

### 方法

创建并展示了一个多模态数据集，整合了高分辨率3D激光雷达扫描与Wi-Fi RSSI测量数据，在多房间室内环境中，20种不同AP配置下收集，包含无人和有人两种场景的测量数据。

### 主要发现

该数据集支持研究动态环境（如人员存在）对无线信号传播的影响，为室内无线通信系统优化提供了基础数据支持。

### 结论

所提出的数据集作为研究资源，有助于推动数据驱动的室内无线建模，特别是在高频标准下的通信系统优化，为开发稳健、高容量的室内通信系统奠定基础。

### 翻译

随着支持带宽密集型和延迟敏感型应用（如实时视频分析、智能感知和扩展现实XR）的智能设备数量不断增加，室内环境需要可靠的无线连接。在此，准确的无线电环境图(REMs)估计能够支持自适应无线网络规划和接入点(AP)部署优化。然而，由于室内空间的复杂性，生成真实的REMs仍然具有挑战性。为克服这一挑战，本文引入了一个多模态数据集，该数据集集成了高分辨率3D激光雷达扫描与在多房间室内环境中20种不同AP配置下收集的Wi-Fi接收信号强度指示器(RSSI)测量。该数据集捕获了两种测量场景：第一种是环境中无人存在的情况，第二种是有人存在的情况。因此，所提供的数据集支持研究动态环境对无线信号传播的影响。该资源旨在促进数据驱动的无线建模研究，特别是在IEEE 802.11be(Wi-Fi 7)等新兴高频标准的背景下，旨在推动稳健、高容量室内通信系统的发展。


### 论文摘要

The growing number of smart devices supporting bandwidth-intensive and latency-sensitive applications, such as real-time video analytics, smart sensing, and Extended Reality (XR), necessitates reliable wireless connectivity in indoor environments. Therein, accurate estimation of Radio Environment Maps (REMs) enables adaptive wireless network planning and optimization of Access Point (AP) placement. However, generating realistic REMs remains challenging due to the complexity of indoor spaces. To overcome this challenge, this paper introduces a multimodal dataset that integrates high-resolution 3D LiDAR scans with Wi-Fi Received Signal Strength Indicator (RSSI) measurements collected under 20 distinct AP configurations in a multi-room indoor environment. The dataset captures two measurement scenarios: the first without human presence in the environment, and the second with human presence. Thus, the presented dataset supports the study of dynamic environmental effects on wireless signal propagation. This resource is designed to facilitate research in data-driven wireless modeling, particularly in the context of emerging high-frequency standards such as IEEE 802.11be (Wi-Fi 7), and aims to advance the development of robust, high-capacity indoor communication systems.

---

## 48. VLM6D: VLM based 6Dof Pose Estimation based on RGB-D Images

**论文链接:** [http://arxiv.org/abs/2511.00120v1](http://arxiv.org/abs/2511.00120v1)

**作者:** Md Selim Sarowar, Sungho Kim

**发布时间:** 2025-10-31

**备注:** This paper has been accepted to IEIE( The Institute Of Electronics  and Information Engineering, South Korea) Fall,2025 Conference

### GPT解析

### 总结

VLM6D是一种新颖的双流架构，利用RGB-D输入中的视觉和几何数据实现鲁棒且精确的6D物体姿态估计，在具有挑战性的Occluded-LineMOD数据集上取得了新的SOTA性能。

### 背景

计算机视觉中精确计算6D物体姿态的主要挑战在于当前方法在从合成数据到真实环境的泛化方面存在困难，特别是在光照变化、无纹理物体和严重遮挡的情况下表现脆弱。

### 目的

提出VLM6D，一种新颖的双流架构，利用RGB-D输入中的视觉和几何数据的优势，实现鲁棒且精确的姿态估计。

### 方法

VLM6D框架集成了两个专门的编码器：一个强大的自监督视觉Transformer（DINOv2）处理RGB模态，利用其丰富的预训练视觉理解能力；一个PointNet++编码器处理深度数据衍生的3D点云，实现鲁棒的几何推理；这两个互补的特征流被有效融合，用于多任务预测头。

### 主要发现

通过全面实验，VLM6D在具有挑战性的Occluded-LineMOD数据集上取得了新的SOTA性能，验证了其优越的鲁棒性和准确性。

### 结论

VLM6D通过结合视觉和几何信息，成功解决了计算机视觉中6D物体姿态估计的挑战，特别是在处理复杂环境（如光照变化、无纹理物体和严重遮挡）时表现出色。

### 翻译

计算机视觉中的主要挑战是精确计算6D物体的姿态，然而许多当前方法仍然脆弱且难以从合成数据泛化到具有变化光照、无纹理物体和严重遮挡的真实世界情况。为解决这些限制，VLM6D是一种新颖的双流架构，它利用RGB-D输入中视觉和几何数据的独特优势进行鲁棒且精确的姿态估计。我们的框架独特地集成了两个专门的编码器：一个强大的自监督视觉Transformer（DINOv2）处理RGB模态，利用其丰富、预训练的视觉语法理解能力，实现对纹理和光照变化的显著抵抗力。同时，一个PointNet++编码器处理从深度数据衍生的3D点云，实现鲁棒的几何推理，即使在严重遮挡情况下典型的稀疏、碎片化数据中也能表现出色。这些互补的特征流被有效融合，为多任务预测头提供信息。我们通过全面实验证明，VLM6D在具有挑战性的Occluded-LineMOD上获得了新的SOTA性能，验证了其优越的鲁棒性和准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决计算机视觉中6D物体姿态估计的挑战，特别是处理真实世界中常见的问题如光照变化、无纹理物体和严重遮挡。这个问题非常重要，因为准确的6D姿态估计是机器人抓取、自动驾驶、增强现实和自动化装配等应用的基础能力，使机器能够感知、理解和与周围环境互动。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（在真实世界复杂情况下表现脆弱）设计了VLM6D。他们借鉴了双流架构思想，分别处理RGB和深度数据，但进行了创新：使用DINOv2处理RGB数据（利用其强大的视觉理解和泛化能力），使用PointNet++处理深度数据（处理几何信息）。作者还借鉴了密集对应点的思想，但通过双流架构和特征融合策略改进了计算效率和准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用双流架构，分别从RGB图像和深度数据中提取互补的特征信息（RGB提供视觉信息，深度提供几何信息），然后融合这些特征进行6D姿态估计。实现流程包括：1)输入处理（RGB图像调整归一化，深度图像转为点云）；2)双流特征提取（DINOv2处理RGB，PointNet++处理点云）；3)特征融合（连接特征向量并通过MLP处理）；4)多任务预测（旋转、平移、置信度和物体分类）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双流架构设计（结合DINOv2和PointNet++）；2)特征融合策略（晚期融合）；3)多任务预测头。相比之前工作（如RDPN6D），VLM6D具有更强的泛化能力（使用自监督学习）、更高效的计算（不需要预测密集坐标图）、更好的鲁棒性（在反射表面、无纹理物体和极端遮挡条件下表现更好），并且能够处理高达80%的物体遮挡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VLM6D通过创新性地结合自监督视觉Transformer和点云处理网络，实现了在复杂场景下（严重遮挡、光照变化、无纹理物体）更加鲁棒和准确的6D物体姿态估计。'}


### 论文摘要

The primary challenge in computer vision is precisely calculating the pose of 6D objects, however many current approaches are still fragile and have trouble generalizing from synthetic data to real-world situations with fluctuating lighting, textureless objects, and significant occlusions. To address these limitations, VLM6D, a novel dual-stream architecture that leverages the distinct strengths of visual and geometric data from RGB-D input for robust and precise pose estimation. Our framework uniquely integrates two specialized encoders: a powerful, self-supervised Vision Transformer (DINOv2) processes the RGB modality, harnessing its rich, pre-trained understanding of visual grammar to achieve remarkable resilience against texture and lighting variations. Concurrently, a PointNet++ encoder processes the 3D point cloud derived from depth data, enabling robust geometric reasoning that excels even with the sparse, fragmented data typical of severe occlusion. These complementary feature streams are effectively fused to inform a multi task prediction head. We demonstrate through comprehensive experiments that VLM6D obtained new SOTA performance on the challenging Occluded-LineMOD, validating its superior robustness and accuracy.

---

## 49. D$^2$GS: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.25173v2](http://arxiv.org/abs/2510.25173v2)

**作者:** Kejing Xia, Jidong Jia, Ke Jin, Yucai Bai, Li Sun, Dacheng Tao, Youjian Zhang

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出D²GS，一个无LiDAR的城市场景重建框架，能够产生比LiDAR更密集、更准确的几何先验，实验表明其性能优于现有方法，甚至超过使用真实LiDAR数据的方法。

### 背景

高斯散射在自动驾驶城市场景重建中潜力巨大，但当前方法依赖LiDAR和图像等多模态传感器。LiDAR提供的几何先验虽可减轻重建不适定性，但实践中获取准确LiDAR数据面临挑战：需要精确时空校准且不同传感器位置会导致重投影误差。

### 目的

避免获取准确LiDAR深度的困难，开发一种无LiDAR的城市场景重建框架，获得与LiDAR一样有效但更密集、更准确的几何先验。

### 方法

D²GS框架包含三个主要部分：首先，通过反向投影多视图度量深度预测初始化密集点云，并用渐进修剪策略优化以提高全局一致性；其次，通过深度增强器联合优化高斯几何和预测深度，利用深度基础模型的扩散先验增强高斯渲染的深度图；最后，约束道路区域内高斯的形状和法线属性以提高地面几何准确性。

### 主要发现

Waymo数据集上的大量实验表明，该方法持续优于最先进方法，产生更准确的几何，即使与使用真实LiDAR数据的方法相比也是如此。

### 结论

D²GS成功实现了无LiDAR的城市场景重建，能够产生比传统LiDAR方法更密集、更准确的几何先验，性能超越现有最先进方法。

### 翻译

最近，高斯散射在自动驾驶领域的城市场景重建中显示出巨大潜力。然而，当前的城市场景重建方法通常依赖于多模态传感器作为输入，即LiDAR和图像。尽管LiDAR点云提供的几何先验可以大大减轻重建中的不适定性，但在实践中获取准确的LiDAR数据仍然具有挑战性：i)需要LiDAR与其他传感器之间的精确时空校准，因为它们可能无法同时捕获数据；ii)当LiDAR和相机安装在不同位置时，空间错位会导致重投影误差。为了避免获取准确LiDAR深度的困难，我们提出了D²GS，一个无LiDAR的城市场景重建框架。在这项工作中，我们获得了与LiDAR一样有效但更密集、更准确的几何先验。首先，我们通过反向投影多视图度量深度预测来初始化密集点云。然后通过渐进修剪策略优化该点云以提高全局一致性。其次，我们通过深度增强器联合优化高斯几何和预测的密集度量深度。具体来说，我们利用来自深度基础模型的扩散先验来增强由高斯渲染的深度图。反过来，增强的深度在高斯训练期间提供更强的几何约束。最后，我们通过约束道路区域内高斯的形状和法线属性来提高地面几何的准确性。在Waymo数据集上的大量实验表明，我们的方法持续优于最先进的方法，产生更准确的几何，即使与使用真实LiDAR数据的方法相比也是如此。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶领域城市场景重建中对LiDAR传感器的依赖问题。这个问题很重要，因为获取LiDAR数据需要昂贵设备、专业车辆，且传感器间精确校准困难，同时LiDAR和相机安装在不同位置会导致重投影误差，这些都限制了实际应用的可扩展性和成本效益。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了LiDAR依赖带来的校准困难和数据获取成本问题，然后意识到需要替代LiDAR的密集几何先验。方法设计上，他们使用多视图深度预测初始化点云，通过渐进式修剪策略优化，借鉴了3DGS的高效特性、扩散模型的深度生成先验以及场景图表示方法来组织不同类型的Gaussian，并结合道路区域的强几何先验知识。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过仅使用相机输入创建不依赖LiDAR的城市街道场景重建框架，利用图像衍生的几何先验替代LiDAR点云。实现流程包括：1)使用多视图深度估计和渐进式修剪初始化紧凑Gaussian表示；2)使用基于扩散的深度增强器进行深度和Gaussian的联合优化；3)在场景图中引入专门的道路节点利用强几何先验显式建模地面平面；4)迭代优化Gaussian参数和深度表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全LiDAR-free的重建框架；2)渐进式修剪策略从密集点云获得紧凑表示；3)基于扩散的深度增强器实现深度和Gaussian的联合优化；4)专门的道路节点利用强几何先验。相比之前工作，D2GS消除了对LiDAR的依赖和校准误差，避免了单目深度估计的尺度模糊和多视图深度估计不适合动态场景的问题，并通过迭代优化提供了更密集、准确的深度监督。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'D2GS提出了一种不依赖LiDAR的城市街道场景重建框架，通过渐进式修剪、深度增强和道路节点优化，实现了比使用LiDAR数据更准确的几何重建。'}


### 论文摘要

Recently, Gaussian Splatting (GS) has shown great potential for urban scene reconstruction in the field of autonomous driving. However, current urban scene reconstruction methods often depend on multimodal sensors as inputs, \textit{i.e.} LiDAR and images. Though the geometry prior provided by LiDAR point clouds can largely mitigate ill-posedness in reconstruction, acquiring such accurate LiDAR data is still challenging in practice: i) precise spatiotemporal calibration between LiDAR and other sensors is required, as they may not capture data simultaneously; ii) reprojection errors arise from spatial misalignment when LiDAR and cameras are mounted at different locations. To avoid the difficulty of acquiring accurate LiDAR depth, we propose D$^2$GS, a LiDAR-free urban scene reconstruction framework. In this work, we obtain geometry priors that are as effective as LiDAR while being denser and more accurate. $\textbf{First}$, we initialize a dense point cloud by back-projecting multi-view metric depth predictions. This point cloud is then optimized by a Progressive Pruning strategy to improve the global consistency. $\textbf{Second}$, we jointly refine Gaussian geometry and predicted dense metric depth via a Depth Enhancer. Specifically, we leverage diffusion priors from a depth foundation model to enhance the depth maps rendered by Gaussians. In turn, the enhanced depths provide stronger geometric constraints during Gaussian training. $\textbf{Finally}$, we improve the accuracy of ground geometry by constraining the shape and normal attributes of Gaussians within road regions. Extensive experiments on the Waymo dataset demonstrate that our method consistently outperforms state-of-the-art methods, producing more accurate geometry even when compared with those using ground-truth LiDAR data.

---

## 50. Which LiDAR scanning pattern is better for roadside perception: Repetitive or Non-repetitive?

**论文链接:** [http://arxiv.org/abs/2511.00060v1](http://arxiv.org/abs/2511.00060v1)

**作者:** Zhiqi Qi, Runxin Zhao, Hanyang Zhuang, Chunxiang Wang, Ming Yang

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究探讨了不同LiDAR扫描模式对路边感知性能的影响，创建了'InfraLiDARs' Benchmark'数据集，比较了重复式和非重复式扫描LiDAR的性能，发现两者检测性能相当，非重复式LiDAR虽然感知范围有限但成本效益高

### 背景

基于LiDAR的路边感知是智能交通系统的基石，现有研究多关注LiDAR的最佳放置位置，而不同扫描模式对感知性能的影响研究不足

### 目的

系统研究基础设施背景下不同LiDAR扫描模式的差异，评估这些模式对3D目标检测算法性能的影响

### 方法

在CARLA仿真环境中创建'InfraLiDARs' Benchmark'数据集，使用同时运行的重复式和非重复式扫描LiDAR进行数据收集，进行统计分析并评估多种3D目标检测算法的性能

### 主要发现

非重复扫描LiDAR和128线重复扫描LiDAR在各种场景中检测性能相当；尽管非重复LiDAR感知范围有限，但因其价格低廉而具有成本效益；不同扫描模式产生不同点云分布，影响目标检测和环境理解效果

### 结论

为设置具有最佳LiDAR扫描模式和兼容算法的路边感知系统提供见解，适应不同路边应用需求，并公开数据集促进进一步研究

### 翻译

基于LiDAR的路边感知是先进智能交通系统(ITS)的基石。虽然已有大量研究解决了基础设施LiDAR的最佳放置问题，但不同LiDAR扫描模式对感知性能的深远影响尚未得到充分研究。各种扫描模式的固有特性——如传统重复式（机械/固态）与新兴的非重复式（如基于棱镜的系统）——导致在不同距离下产生不同的点云分布，这直接影响目标检测和整体环境理解的效果。为了系统性地研究基础设施背景下的这些差异，我们引入了'InfraLiDARs' Benchmark'，这是一个在CARLA仿真环境中精心收集的新数据集，使用了同时运行的基础设施LiDAR，展示了两种扫描模式。利用这个基准，我们对各种LiDAR扫描能力进行了全面的统计分析，并评估了这些不同模式对各种领先3D目标检测算法性能的影响。我们的研究揭示，非重复扫描LiDAR和128线重复扫描LiDAR在各种场景中表现出相当的检测性能。尽管非重复LiDAR的感知范围有限，但考虑到其低廉的价格，它是一种经济有效的选择。最终，这项研究为设置具有最佳LiDAR扫描模式和兼容算法的路边感知系统提供了见解，以适应不同的路边应用需求，并公开发布了'InfraLiDARs' Benchmark'数据集，以促进进一步的研究。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要想解决的问题是：对于路侧感知系统，重复扫描模式（repetitive）和非重复扫描模式（non-repetitive）的激光雷达（LiDAR）哪种性能更好？这个问题在现实中非常重要，因为LiDAR路侧感知是智能交通系统的基础，直接影响交通安全性、流量管理和自动驾驶能力；在研究中也很重要，因为虽然已有大量研究关注LiDAR部署位置，但不同扫描模式对感知性能的影响研究相对不足，而不同的扫描模式会导致不同距离下的点云分布差异，直接影响物体检测和环境理解的效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别了研究空白：虽然已有车载LiDAR研究，但路侧部署的不同扫描模式比较不足。然后设计了综合评估框架，包括统计基准和性能基准。他们在CARLA仿真环境中创建了专用数据集'InfraLiDARs' Benchmark'，收集了不同LiDAR扫描模式的数据。作者借鉴了现有的3D物体检测算法（如PointRCNN、PointPillars、PV-RCNN和DSVT），并参考了已有的LiDAR分类框架，但基于扫描模式而非传统分类方法。他们还参考了路侧感知研究，但专注于扫描模式而非传感器放置策略。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过系统比较重复扫描和非重复扫描两种LiDAR模式在路侧感知中的性能，为实际部署提供数据驱动的建议，并考虑成本效益因素。整体实现流程包括：1）在CARLA仿真环境中创建三种场景（高速公路、十字路口、弯道），使用四种LiDAR在相同位置和方向部署；2）进行统计基准测试，分析点云质量和不同距离下的检测能力；3）使用多种3D物体检测算法进行性能基准测试，采用整体AP分析、距离分段AP分析和高质量检测区域分析；4）综合比较结果，提出针对不同应用场景的最优配置建议。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1）创建了'InfraLiDARs' Benchmark'数据集，专为路侧感知设计；2）首次系统比较了路侧部署中重复与非重复扫描LiDAR的性能；3）提出了全面的评估框架，结合统计基准和性能基准；4）发现了非重复扫描LiDAR在远距离检测中的优势。相比之前的工作，本文的研究焦点不同（关注扫描模式而非车载LiDAR或传感器放置），使用的数据集不同（专为路侧感知设计），评估维度更全面（考虑距离分布和高质量检测区域），并引入了成本效益分析。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过创建专用数据集和综合评估框架，系统比较了重复扫描与非重复扫描LiDAR在路侧感知中的性能，发现非重复扫描LiDAR与128线重复扫描LiDAR具有相当的检测性能，且在远距离检测中表现更佳，为路侧感知系统的优化部署提供了数据驱动的建议。'}


### 论文摘要

LiDAR-based roadside perception is a cornerstone of advanced Intelligent Transportation Systems (ITS). While considerable research has addressed optimal LiDAR placement for infrastructure, the profound impact of differing LiDAR scanning patterns on perceptual performance remains comparatively under-investigated. The inherent nature of various scanning modes - such as traditional repetitive (mechanical/solid-state) versus emerging non-repetitive (e.g. prism-based) systems - leads to distinct point cloud distributions at varying distances, critically dictating the efficacy of object detection and overall environmental understanding. To systematically investigate these differences in infrastructure-based contexts, we introduce the "InfraLiDARs' Benchmark," a novel dataset meticulously collected in the CARLA simulation environment using concurrently operating infrastructure-based LiDARs exhibiting both scanning paradigms. Leveraging this benchmark, we conduct a comprehensive statistical analysis of the respective LiDAR scanning abilities and evaluate the impact of these distinct patterns on the performance of various leading 3D object detection algorithms. Our findings reveal that non-repetitive scanning LiDAR and the 128-line repetitive LiDAR were found to exhibit comparable detection performance across various scenarios. Despite non-repetitive LiDAR's limited perception range, it's a cost-effective option considering its low price. Ultimately, this study provides insights for setting up roadside perception system with optimal LiDAR scanning patterns and compatible algorithms for diverse roadside applications, and publicly releases the "InfraLiDARs' Benchmark" dataset to foster further research.

---

## 51. UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling

**论文链接:** [http://arxiv.org/abs/2510.24106v3](http://arxiv.org/abs/2510.24106v3)

**作者:** Junhong Zou, Zhenxu Sun, Wei Qiu, Zhaoxiang Zhang, Zhen Lei, Xiangyu Zhu

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了一种名为UniField的通用流场表示模型，通过整合多个子领域的空气动力学数据进行联合训练，解决了神经网络在空气动力学模拟中面临的数据稀缺问题。

### 背景

表面压力场的空气动力学模拟对许多工程问题至关重要。近年来，深度神经网络已成为传统计算流体动力学(CFD)模拟的高效替代方案，但数据稀缺性仍然是一个基本挑战，限制了神经网络的应用。

### 目的

为了解决数据稀缺的限制，作者提出整合多个子领域的空气动力学数据进行联合训练，以学习更通用的场域表示。

### 方法

作者整合了五个不同的数据集，涵盖汽车、火车、飞机和一般形状等多个领域。面对不同领域之间的显著数据差异，他们提出了UniField，它采用领域无关的Transformer模块提取通用的点云特征，并定制领域特定的流条件适配器来适应不同子领域的流信息。

### 主要发现

尽管不同子领域的空气动力学数据通常受不同方程支配，但作者发现，在所有数据上联合训练的模型通常比在单独数据集上分别训练的模型表现更好。这表明这些数据相互补充，帮助模型学习更好的流场表示。

### 结论

这些结果突显了UniField作为通用流场表示模型的潜力，并为神经网络在空气动力学分析中的更广泛应用奠定了基础。

### 翻译

物体表面压力场的空气动力学模拟对许多工程问题至关重要。近年来，深度神经网络已成为传统计算上昂贵的CFD模拟的高效替代方案，用于建模表面压力场。然而，数据稀缺仍然是一个基本挑战，限制了神经网络的应用。为了解决这一限制，我们提出整合多个子领域的空气动力学数据并进行联合训练，以学习更通用的场域表示。我们整合了五个不同的数据集，涵盖各种领域，包括汽车、火车、飞机和一般形状。面对不同领域之间的显著数据差异，我们提出了UniField，它采用领域无关的Transformer模块提取通用的点云特征，并定制领域特定的流条件适配器来适应不同子领域的流信息。尽管不同子领域的空气动力学数据通常受不同方程支配，但我们比较了在所有数据上联合训练的模型与在单独数据集上分别训练的模型，发现联合训练的模型通常表现出更好的性能。这表明这些数据相互补充，帮助模型学习更好的流场表示。这些结果突显了UniField作为通用流场表示模型的潜力，并为神经网络在空气动力学分析中的更广泛应用奠定了基础。


### 论文摘要

Aerodynamic simulation of the surface pressure field around objects is crucial for many engineering problems. In recent years, deep neural networks have emerged as an efficient alternative to traditional, computationally expensive CFD simulations for modeling surface pressure fields. However, data scarcity remains a fundamental challenge, limiting the application of neural networks. To address this limitation, we propose to integrate aerodynamic data from multiple subfields and conduct joint training to learn more general field representations. We consolidate five different datasets covering various fields, including automobiles, trains, aircraft, and general shapes. Facing significant data differences across different domains, we propose UniField, which employs a domain-agnostic Transformer module to extract general point cloud features and customizes domain-specific flow-conditioned adapters to adapt to the flow information in different subfields. Despite the fact that aerodynamic data from different subfields are typically governed by different equations, we compare models trained jointly on all data with those trained separately on individual datasets and find that the jointly-trained model commonly demonstrates better performance. This indicates that these data complement each other to help the model learn better flow field representations. These results highlight the potential of UniField as a universal flow field representation model and lay the foundation for broader applications of neural networks in aerodynamic analysis.

---

## 52. A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding

**论文链接:** [http://arxiv.org/abs/2511.02565v1](http://arxiv.org/abs/2511.02565v1)

**作者:** Jingyu Lu, Haonan Wang, Qixiang Zhang, Xiaomeng Li

**发布时间:** 2025-11-04

**备注:** 9 pages main text with 6 figures (excluding references),  supplementary material included

### GPT解析

### 总结

本文提出了一种名为视觉皮层流架构（VCFlow）的新型层次解码框架，能够无需针对特定受试者训练的情况下，从fMRI数据重建连续视觉体验，解决了跨受试者泛化困难和大脑信号复杂性的挑战。

### 背景

主题无关的脑解码技术在临床应用方面有很大潜力，但由于跨受试者泛化困难和大脑信号的复杂性，这一方向仍处于探索阶段。传统方法需要每个受试者超过12小时的数据和大量计算资源。

### 目的

开发一种新的解码框架，能够高效地重建视觉体验，减少对受试者特定数据的依赖，并提高计算效率。

### 方法

提出视觉皮层流架构（VCFlow），这是一种层次解码框架，明确模拟人类视觉系统的腹侧-背侧架构。通过解离和利用早期视觉皮层、腹侧和背侧流中的特征，捕获视觉重建所需的多样化和互补认知信息。同时引入特征级对比学习策略，增强提取受试者不变的语义表征，提高对未见受试者的适用性。

### 主要发现

VCFlow相比传统方法仅损失7%的准确率，但无需重新训练，每10秒即可生成重建的视频，提供了一种快速且临床可扩展的解决方案。

### 结论

VCFlow为视觉脑解码领域提供了一个高效、实用的解决方案，有望在临床应用中发挥作用。

### 翻译

主题无关的脑解码旨在无需针对特定受试者训练的情况下，从fMRI数据重建连续视觉体验，在临床应用方面有很大潜力。然而，由于跨受试者泛化困难和大脑信号的复杂性，这一方向仍处于探索阶段。在本工作中，我们提出了视觉皮层流架构（VCFlow），一种新的层次解码框架，明确模拟人类视觉系统的腹侧-背侧架构，以学习多维表征。通过解离和利用来自早期视觉皮层、腹侧和背侧流的特征，VCFlow捕获了视觉重建所需的多样化和互补认知信息。此外，我们引入了特征级对比学习策略，以增强提取受试者不变的语义表征，从而增强对未见受试者的主题无关适用性。与需要每个受试者超过12小时数据和大量计算的传统方法不同，VCFlow平均仅损失7%的准确率，无需重新训练，每10秒即可生成每个重建视频，提供了快速且临床可扩展的解决方案。论文接受后将发布源代码。


### 论文摘要

Subject-agnostic brain decoding, which aims to reconstruct continuous visual experiences from fMRI without subject-specific training, holds great potential for clinical applications. However, this direction remains underexplored due to challenges in cross-subject generalization and the complex nature of brain signals. In this work, we propose Visual Cortex Flow Architecture (VCFlow), a novel hierarchical decoding framework that explicitly models the ventral-dorsal architecture of the human visual system to learn multi-dimensional representations. By disentangling and leveraging features from early visual cortex, ventral, and dorsal streams, VCFlow captures diverse and complementary cognitive information essential for visual reconstruction. Furthermore, we introduce a feature-level contrastive learning strategy to enhance the extraction of subject-invariant semantic representations, thereby enhancing subject-agnostic applicability to previously unseen subjects. Unlike conventional pipelines that need more than 12 hours of per-subject data and heavy computation, VCFlow sacrifices only 7\% accuracy on average yet generates each reconstructed video in 10 seconds without any retraining, offering a fast and clinically scalable solution. The source code will be released upon acceptance of the paper.

---

## 53. Seeing Across Time and Views: Multi-Temporal Cross-View Learning for Robust Video Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.02564v1](http://arxiv.org/abs/2511.02564v1)

**作者:** Md Rashidunnabi, Kailash A. Hambarde, Vasco Lopes, Joao C. Neves, Hugo Proenca

**发布时间:** 2025-11-04

### GPT解析

### 总结

这篇论文提出了MTF-CVReID，一个针对视频跨视角行人重识别的参数高效框架，通过七个专门设计的模块解决了视角变化大、尺度差异和时间不一致性带来的挑战，在保持实时效率的同时实现了最先进的性能。

### 背景

视频跨视角行人重识别(如空中-地面监控)由于极端视角变化、尺度差异和时间不一致性仍然是一个开放问题。

### 目的

开发一个参数高效的框架，解决跨视角视频行人重识别中的挑战，同时保持实时计算效率。

### 方法

在ViT-B/16骨干网络上引入七个互补模块：CSFN(校正相机和视角偏差)、MRFH(实现跨高度的尺度稳定)、IAMM(强化持久的身份特征)、TDM(用于感知运动的短期时间编码)、IVFA(实现视角不变的表征对齐)、HTPL(捕获多尺度时间规律)和MVICL(使用对比学习范式强制跨视角身份一致性)。

### 主要发现

尽管只增加约200万个参数和0.7 GFLOPs，MTF-CVReID保持了实时效率(189 FPS)，在AG-VPReID基准测试的所有高度级别上实现了最先进性能，并在G2A-VReID和MARS数据集上表现出强大的跨数据集泛化能力。

### 结论

精心设计的基于适配器的模块可以在不牺牲计算效率的情况下显著增强跨视角鲁棒性和时间一致性。

### 翻译

基于视频的跨域视角行人重识别(例如，空中-地面监控)由于极端视角变化、尺度差异和时间不一致性仍然是一个开放问题。为了解决这些挑战，我们提出了MTF-CVReID，一个参数高效的框架，在ViT-B/16骨干网络上引入了七个互补模块。具体来说，我们包括：(1)跨流特征归一化(CSFN)来校正相机和视角偏差；(2)多分辨率特征调和(MRFH)用于跨高度的尺度稳定；(3)身份感知记忆模块(IAMM)来强化持久的身份特征；(4)时间动态建模(TDM)用于感知运动的短期时间编码；(5)跨视角特征对齐(IVFA)实现视角不变的表征对齐；(6)分层时间模式学习(HTPL)捕获多尺度时间规律；以及(7)多视角身份一致性学习(MVICL)，使用对比学习范式强制跨视角身份一致性。尽管比基线模型只增加了约200万个参数和0.7 GFLOPs，MTF-CVReID保持了实时效率(189 FPS)，并在AG-VPReID基准测试的所有高度级别上实现了最先进性能，同时在G2A-VReID和MARS数据集上具有强大的跨数据集泛化能力。这些结果表明，精心设计的基于适配器的模块可以在不牺牲计算效率的情况下显著增强跨视角鲁棒性和时间一致性。源代码可在https://github.com/MdRashidunnabi/MTF-CVReID获取。


### 论文摘要

Video-based person re-identification (ReID) in cross-view domains (for example, aerial-ground surveillance) remains an open problem because of extreme viewpoint shifts, scale disparities, and temporal inconsistencies. To address these challenges, we propose MTF-CVReID, a parameter-efficient framework that introduces seven complementary modules over a ViT-B/16 backbone. Specifically, we include: (1) Cross-Stream Feature Normalization (CSFN) to correct camera and view biases; (2) Multi-Resolution Feature Harmonization (MRFH) for scale stabilization across altitudes; (3) Identity-Aware Memory Module (IAMM) to reinforce persistent identity traits; (4) Temporal Dynamics Modeling (TDM) for motion-aware short-term temporal encoding; (5) Inter-View Feature Alignment (IVFA) for perspective-invariant representation alignment; (6) Hierarchical Temporal Pattern Learning (HTPL) to capture multi-scale temporal regularities; and (7) Multi-View Identity Consistency Learning (MVICL) that enforces cross-view identity coherence using a contrastive learning paradigm. Despite adding only about 2 million parameters and 0.7 GFLOPs over the baseline, MTF-CVReID maintains real-time efficiency (189 FPS) and achieves state-of-the-art performance on the AG-VPReID benchmark across all altitude levels, with strong cross-dataset generalization to G2A-VReID and MARS datasets. These results show that carefully designed adapter-based modules can substantially enhance cross-view robustness and temporal consistency without compromising computational efficiency. The source code is available at https://github.com/MdRashidunnabi/MTF-CVReID

---

## 54. CoCoVa: Chain of Continuous Vision-Language Thought for Latent Space Reasoning

**论文链接:** [http://arxiv.org/abs/2511.02360v1](http://arxiv.org/abs/2511.02360v1)

**作者:** Jizheng Ma, Xiaofei Zhou, Yanlong Song, Han Yan

**发布时间:** 2025-11-04

### GPT解析

### 总结

CoCoVa是一种新的视觉-语言模型框架，通过连续跨模态推理解决传统模型在处理高维视觉感知方面的局限性，在多个基准测试中表现出色。

### 背景

人类认知中存在难以用言语表达的隐性思维过程，使人类能以多种方式理解世界；而当代视觉-语言模型局限于离散的语言标记空间推理，限制了视觉感知的丰富高维特性。

### 目的

弥合人类隐性思维与当前视觉-语言模型之间的差距，提出利用连续跨模态推理处理多样化视觉-语言任务的新框架。

### 方法

CoCoVa的核心是迭代推理循环，使用潜在Q-Former作为动态推理引擎，通过跨模态融合优化潜在思维向量链；实现令牌选择机制识别显著视觉区域；结合对比学习和基于扩散的重构进行多任务训练，确保潜在表示与视觉和文本模态对齐。

### 主要发现

CoCoVa在准确率和令牌效率上优于强基线模型；1.5B主干模型在几乎所有基准测试中与7B-9B模型竞争或超越；扩展到7B LLM主干模型时仍保持竞争力；学习到的潜在空间捕捉了可解释和结构化的推理模式。

### 结论

CoCoVa成功弥合了离散语言处理与视觉理解连续性之间的表征差距，展示了桥接这一差距的潜力。

### 翻译

在人类认知中，存在许多难以言表且超越言语表达的思维过程，使我们能够以多种方式理解和与世界互动。然而，当代视觉-语言模型仍然局限于在离散且刚性的语言标记空间中进行推理，从而限制了视觉感知的丰富高维特性。为了弥合这一差距，我们提出了CoCoVa（连续视觉-语言思维链），一种利用连续跨模态推理处理多样化视觉-语言任务的新框架。CoCoVa的核心是一个迭代推理循环，其中一种新型的潜在Q-Former作为动态推理引擎，通过跨模态融合迭代优化潜在思维向量链。为了聚焦这一过程，令牌选择机制动态识别显著的视觉区域，模拟注意力焦点。为确保这些潜在思维保持基础，我们使用结合对比学习和基于扩散的重构的多任务目标训练模型，强制潜在表示与视觉和文本模态保持对齐。评估显示，CoCoVa在准确率和令牌效率方面优于强基线模型。使用1.5B主干模型时，它在几乎所有基准测试中与更大的7B-9B模型竞争或超越。当扩展到7B LLM主干模型时，它仍能与最先进模型保持竞争力。定性分析验证了学习到的潜在空间捕捉了可解释和结构化的推理模式，突显了CoCoVa弥合离散语言处理与视觉理解连续性之间表征差距的潜力。


### 论文摘要

In human cognition, there exist numerous thought processes that are tacit and beyond verbal expression, enabling us to understand and interact with the world in multiple ways. However, contemporary Vision-Language Models (VLMs) remain constrained to reasoning within the discrete and rigid space of linguistic tokens, thereby bottlenecking the rich, high-dimensional nature of visual perception. To bridge this gap, we propose CoCoVa (Chain of Continuous Vision-Language Thought), a novel framework for vision-language model that leverages continuous cross-modal reasoning for diverse vision-language tasks. The core of CoCoVa is an iterative reasoning cycle, where a novel Latent Q-Former (LQ-Former) acts as a dynamic reasoning engine, iteratively refining a chain of latent thought vectors through cross-modal fusion. To focus this process, a token selection mechanism dynamically identifies salient visual regions, mimicking attentional focus. To ensure these latent thoughts remain grounded, we train the model with a multi-task objective that combines contrastive learning and diffusion-based reconstruction, enforcing alignment between latent representations and both visual and textual modalities. Evaluations show CoCoVa improves accuracy and token efficiency over strong baselines. With a 1.5B backbone, it competes with or surpasses larger 7B-9B models on almost all benchmarks. When scaled to 7B LLM backbones, it remains competitive with state-of-the-art models. Qualitative analysis validates that learned latent space captures interpretable and structured reasoning patterns, highlighting the potential of CoCoVa to bridge the representational gap between discrete language processing and the continuous nature of visual understanding.

---

## 55. Probabilistic Graph Cuts

**论文链接:** [http://arxiv.org/abs/2511.02272v1](http://arxiv.org/abs/2511.02272v1)

**作者:** Ayoub Ghriss

**发布时间:** 2025-11-04

**备注:** 23 pages

### GPT解析

### 总结

该研究提出了一种统一的概率框架，用于图割的概率松弛，作为谱聚类的可微分替代方案，无需特征分解即可实现端到端和在线学习，为可扩展、可微分的图划分提供了严谨的数值稳定基础。

### 背景

现有的图割概率松弛方法主要集中在RatioCut上，缺乏通用保证和原则性梯度，限制了其在各种聚类和对比学习任务中的应用。

### 目的

开发一个覆盖广泛割类型（包括Normalized Cut）的统一概率框架，提供期望离散割的紧密解析上界，并建立可扩展、可微分图划分的严谨数值稳定基础。

### 方法

通过积分表示和高斯超几何函数构建统一的概率框架，提供具有闭式前向和反向传播的解析上界，实现可微分图划分。

### 主要发现

提出的统一框架覆盖了广泛的割类型，通过积分表示和高斯超几何函数提供了期望离散割的紧密解析上界，并具有闭式前向和反向传播。

### 结论

该研究为可扩展、可微分的图划分提供了严谨、数值稳定的基础，能够支持广泛的聚类和对比学习目标，克服了现有方法的局限性。

### 翻译

图割的概率松弛为谱聚类提供了不同的可微分替代方案，能够在不进行特征分解的情况下实现端到端和在线学习，但先前的工作主要集中在RatioCut上，缺乏通用保证和原则性梯度。我们提出了一个统一的概率框架，涵盖了广泛的割类型，包括Normalized Cut。我们的框架通过积分表示和具有闭式前向和反向传播的高斯超几何函数，为期望离散割提供了紧密的解析上界。这些结果共同为可扩展、可微分的图划分提供了一个严谨的、数值稳定的基础，涵盖了广泛的聚类和对比学习目标。


### 论文摘要

Probabilistic relaxations of graph cuts offer a differentiable alternative to spectral clustering, enabling end-to-end and online learning without eigendecompositions, yet prior work centered on RatioCut and lacked general guarantees and principled gradients. We present a unified probabilistic framework that covers a wide class of cuts, including Normalized Cut. Our framework provides tight analytic upper bounds on expected discrete cuts via integral representations and Gauss hypergeometric functions with closed-form forward and backward. Together, these results deliver a rigorous, numerically stable foundation for scalable, differentiable graph partitioning covering a wide range of clustering and contrastive learning objectives.

---

## 56. NSYNC: Negative Synthetic Image Generation for Contrastive Training to Improve Stylized Text-To-Image Translation

**论文链接:** [http://arxiv.org/abs/2511.01517v1](http://arxiv.org/abs/2511.01517v1)

**作者:** Serkan Ozturk, Samet Hicsonmez, Pinar Duygulu

**发布时间:** 2025-11-03

**备注:** Under review

### GPT解析

### 总结

本文提出了一种名为NSYNC的新型对比学习框架，通过生成负合成数据集与正真实图像结合进行对比训练，以提高大型文本到图像扩散模型的风格化能力。

### 背景

当前文本条件图像生成方法能生成逼真图像但无法捕捉特定风格，即使在目标风格数据集上微调也难以掌握风格特征。

### 目的

提高大型文本到图像扩散模型的风格化能力，使其能够更好地捕捉特定风格特征。

### 方法

NSYNC框架专注于生成负合成数据集，与正真实图像一起用于对比训练。同时处理负数据和正数据获得对应梯度，通过从正梯度中减去其在负梯度上的投影得到正交分量，基于此更新参数，消除正负数据中都存在的平凡属性，引导模型捕捉独特风格。

### 主要发现

在各种画家和插画师风格的实验中，NSYNC在定量和定性评估上都优于基线方法。

### 结论

NSYNC框架能有效提升文本到图像扩散模型的风格化能力，通过对比学习方法和负合成数据集的生成，使模型能够更好地捕捉特定风格特征。

### 翻译

当前文本条件图像生成方法输出看起来真实的图像，但它们无法捕捉特定风格。简单地在目标风格数据集上微调仍然难以掌握风格特征。在这项工作中，我们提出了一种新颖的对比学习框架来提高大型文本到图像扩散模型的风格化能力。受图像生成模型惊人进展的启发，我们在方法中利用了合成图像生成。通常，生成的合成数据依赖于任务，大多数情况下用于扩大可用的真实训练数据集。有了NSYNC，我们专注于生成负合成数据集，与真实正图像一起用于新颖的对比训练方案。在我们提出的训练设置中，我们将负数据与正数据一起前向传播，分别获得负梯度和正梯度。然后我们通过从正梯度中减去其在负梯度上的投影来获得正交分量，基于此更新参数。这个正交分量消除了正负数据中都存在的平凡属性，并引导模型捕捉更独特的风格。在各种画家和插画师风格上的实验表明，我们的方法在定量和定性上都优于基线方法。我们的代码可在https://github.com/giddyyupp/NSYNC获取。


### 论文摘要

Current text conditioned image generation methods output realistic looking images, but they fail to capture specific styles. Simply finetuning them on the target style datasets still struggles to grasp the style features. In this work, we present a novel contrastive learning framework to improve the stylization capability of large text-to-image diffusion models. Motivated by the astonishing advance in image generation models that makes synthetic data an intrinsic part of model training in various computer vision tasks, we exploit synthetic image generation in our approach. Usually, the generated synthetic data is dependent on the task, and most of the time it is used to enlarge the available real training dataset. With NSYNC, alternatively, we focus on generating negative synthetic sets to be used in a novel contrastive training scheme along with real positive images. In our proposed training setup, we forward negative data along with positive data and obtain negative and positive gradients, respectively. We then refine the positive gradient by subtracting its projection onto the negative gradient to get the orthogonal component, based on which the parameters are updated. This orthogonal component eliminates the trivial attributes that are present in both positive and negative data and directs the model towards capturing a more unique style. Experiments on various styles of painters and illustrators show that our approach improves the performance over the baseline methods both quantitatively and qualitatively. Our code is available at https://github.com/giddyyupp/NSYNC.

---

## 57. Embodied Cognition Augmented End2End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.01334v1](http://arxiv.org/abs/2511.01334v1)

**作者:** Ling Niu, Xiaoji Zheng, Han Wang, Chen Zheng, Ziyuan Yang, Bokui Chen, Jiangtao Gong

**发布时间:** 2025-11-03

**备注:** 24 pages,4 pages

### GPT解析

### 总结

本文提出了一种名为E³AD的新范式，通过在视觉特征提取网络和通用脑电图大模型之间进行对比学习，学习潜在的人类驾驶认知以增强端到端自动驾驶规划性能。

### 背景

基于视觉的端到端自动驾驶已成为新范式，但现有方法通常依赖于标签监督下训练的视觉特征提取网络，这种有限监督框架限制了驾驶模型的通用性和适用性。

### 目的

提出E³AD范式，通过对比学习整合人类驾驶认知，以增强端到端自动驾驶的规划性能。

### 方法

收集认知数据集用于对比学习；研究使用人类驾驶认知增强端到端规划的方法和机制；在公开数据集上使用流行驾驶模型作为基线；进行开环和闭环测试全面评估规划性能。

### 主要发现

E³AD范式显著增强了基线模型的端到端规划性能；消融研究验证了驾驶认知的贡献和对比学习过程的有效性。

### 结论

这是首个将人类驾驶认知整合到端到端自动驾驶规划中的工作；是具身认知数据融入端到端自动驾驶的初步尝试；为脑启发自动驾驶系统提供了有价值的见解；代码将在Github上提供。

### 翻译

近年来，基于视觉的端到端自动驾驶已成为一种新范式。然而，流行的端到端方法通常依赖于在标签监督下训练的视觉特征提取网络。这种有限监督框架限制了驾驶模型的通用性和适用性。在本文中，我们提出了一种称为E³AD的新范式，主张在视觉特征提取网络和通用脑电图大模型之间进行对比学习，以学习潜在的人类驾驶认知，从而增强端到端规划。在这项工作中，我们收集了用于上述对比学习过程的数据集。随后，我们研究了使用人类驾驶认知增强端到端规划的方法和潜在机制，在公开可用的自动驾驶数据集上使用流行的驾驶模型作为基线。进行了开环和闭环测试，以全面评估规划性能。实验结果表明，E³AD范式显著增强了基线模型的端到端规划性能。消融研究进一步验证了驾驶认知的贡献和对比学习过程的有效性。据我们所知，这是第一个将人类驾驶认知整合用于改进端到端自动驾驶规划的工作。它代表了将具身认知数据融入端到端自动驾驶的初步尝试，为未来的脑启发自动驾驶系统提供了有价值的见解。我们的代码将在Github上提供。


### 论文摘要

In recent years, vision-based end-to-end autonomous driving has emerged as a new paradigm. However, popular end-to-end approaches typically rely on visual feature extraction networks trained under label supervision. This limited supervision framework restricts the generality and applicability of driving models. In this paper, we propose a novel paradigm termed $E^{3}AD$, which advocates for comparative learning between visual feature extraction networks and the general EEG large model, in order to learn latent human driving cognition for enhancing end-to-end planning. In this work, we collected a cognitive dataset for the mentioned contrastive learning process. Subsequently, we investigated the methods and potential mechanisms for enhancing end-to-end planning with human driving cognition, using popular driving models as baselines on publicly available autonomous driving datasets. Both open-loop and closed-loop tests are conducted for a comprehensive evaluation of planning performance. Experimental results demonstrate that the $E^{3}AD$ paradigm significantly enhances the end-to-end planning performance of baseline models. Ablation studies further validate the contribution of driving cognition and the effectiveness of comparative learning process. To the best of our knowledge, this is the first work to integrate human driving cognition for improving end-to-end autonomous driving planning. It represents an initial attempt to incorporate embodied cognitive data into end-to-end autonomous driving, providing valuable insights for future brain-inspired autonomous driving systems. Our code will be made available at Github

---

## 58. ColMate: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval

**论文链接:** [http://arxiv.org/abs/2511.00903v1](http://arxiv.org/abs/2511.00903v1)

**作者:** Ahmed Masry, Megh Thakkar, Patrice Bechard, Sathwik Tejaswi Madhusudhan, Rabiul Awal, Shambhavi Mishra, Akshay Kalkunte Suresh, Srivatsava Daruru, Enamul Hoque, Spandana Gella, Torsten Scholak, Sai Rajeswar

**发布时间:** 2025-11-02

### GPT解析

### 总结

本文介绍了ColMate，一个专门针对多模态文档检索的模型，通过OCR预训练、自监督掩码对比学习和后期交互评分机制，改进了现有的文档检索方法，在基准测试中取得了更好的性能。

### 背景

检索增强生成在模型需要专业知识或最新数据访问时已被证明具有实用性。然而，现有的多模态文档检索方法通常复制仅为文本检索开发的技术，无论是在文档编码方式、训练目标定义还是相似度分数计算方面。

### 目的

解决现有多模态文档检索方法的局限性，弥合多模态表示学习与文档检索之间的差距。

### 方法

提出ColMate模型，它利用三种关键技术：基于OCR的预训练目标、自监督掩码对比学习目标，以及与多模态文档结构和视觉特征更相关的后期交互评分机制。

### 主要发现

ColMate在ViDoRe V2基准测试上比现有检索模型提高了3.61%，并且显示出对域外基准测试的更强泛化能力。

### 结论

ColMate是一个有效的文档检索模型，它专门针对多模态文档的特点进行了优化，能够提供比现有方法更好的性能和泛化能力。

### 翻译

检索增强生成已被证明在模型需要专业知识或访问最新数据时具有实用性。然而，现有的多模态文档检索方法通常复制仅为文本检索开发的技术，无论是在文档编码方式、训练目标定义还是相似度分数计算方面。为解决这些局限性，我们提出了ColMate，一个弥合多模态表示学习与文档检索之间差距的文档检索模型。ColMate利用了基于OCR的预训练目标、自监督掩码对比学习目标，以及一种与多模态文档结构和视觉特征更相关的后期交互评分机制。ColMate在ViDoRe V2基准测试上比现有检索模型提高了3.61%，显示出对域外基准测试的更强泛化能力。


### 论文摘要

Retrieval-augmented generation has proven practical when models require specialized knowledge or access to the latest data. However, existing methods for multimodal document retrieval often replicate techniques developed for text-only retrieval, whether in how they encode documents, define training objectives, or compute similarity scores. To address these limitations, we present ColMate, a document retrieval model that bridges the gap between multimodal representation learning and document retrieval. ColMate utilizes a novel OCR-based pretraining objective, a self-supervised masked contrastive learning objective, and a late interaction scoring mechanism more relevant to multimodal document structures and visual characteristics. ColMate obtains 3.61% improvements over existing retrieval models on the ViDoRe V2 benchmark, demonstrating stronger generalization to out-of-domain benchmarks.

---

## 59. TriCon-Fair: Triplet Contrastive Learning for Mitigating Social Bias in Pre-trained Language Models

**论文链接:** [http://arxiv.org/abs/2511.00854v1](http://arxiv.org/abs/2511.00854v1)

**作者:** Chong Lyu, Lin Li, Shiqing Wu, Jingling Yuan

**发布时间:** 2025-11-02

### GPT解析

### 总结

本文提出了一种名为TriCon-Fair的对比学习框架，用于解决大型语言模型中的社会偏见传播问题。该方法通过解耦损失函数，结合三元组和语言建模项，消除正负耦合，减少歧视性输出，同时保持强大的下游性能。

### 背景

大型语言模型的广泛应用引发了关于社会偏见传播的严重担忧，这可能导致有害和不公平的结果。现有的去偏见方法独立处理有偏见和无偏见的样本，忽略了它们之间的相互关系。

### 目的

解决现有去偏见方法中存在的隐藏负-正耦合问题，即对一组的改进无意中损害另一组，导致残余社会偏见持续存在。

### 方法

提出TriCon-Fair，一种对比学习框架，采用解耦损失函数，结合三元组和语言建模项，消除正负耦合。该方法为每个锚点分配明确的有偏见负样本和无偏见正样本，解耦推-拉动态，避免正负耦合，并联合优化语言建模目标以保持通用能力。

### 主要发现

实验结果表明，TriCon-Fair能够超越现有的去偏见基线，减少歧视性输出，同时保持强大的下游性能。

### 结论

TriCon-Fair为敏感的自然语言处理应用提供了一种实用且合乎伦理的解决方案。

### 翻译

大型语言模型日益增多的应用引发了关于社会偏见传播的严重担忧，这可能导致有害和不公平的结果。然而，现有的去偏见方法独立处理有偏见和无偏见的样本，从而忽略了它们之间的相互关系。这种疏忽导致了一种隐藏的负-正耦合，即对一组的改进无意中损害另一组，使残余社会偏见得以持续。在本文中，我们介绍了TriCon-Fair，一种对比学习框架，采用结合三元组和语言建模项的解耦损失函数来消除负-正耦合。我们的TriCon-Fair为每个锚点分配明确的有偏见负样本和无偏见正样本，解耦推-拉动态，避免负-正耦合，并联合优化语言建模(LM)目标以保持通用能力。实验结果表明，TriCon-Fair超越了现有的去偏见基线，减少了歧视性输出，同时保持强大的下游性能。这表明我们提出的TriCon-Fair为敏感的自然语言处理应用提供了一种实用且合乎伦理的解决方案。


### 论文摘要

The increasing utilization of large language models raises significant concerns about the propagation of social biases, which may result in harmful and unfair outcomes. However, existing debiasing methods treat the biased and unbiased samples independently, thus ignoring their mutual relationship. This oversight enables a hidden negative-positive coupling, where improvements for one group inadvertently compromise the other, allowing residual social bias to persist. In this paper, we introduce TriCon-Fair, a contrastive learning framework that employs a decoupled loss that combines triplet and language modeling terms to eliminate positive-negative coupling. Our TriCon-Fair assigns each anchor an explicitly biased negative and an unbiased positive, decoupling the push-pull dynamics and avoiding positive-negative coupling, and jointly optimizes a language modeling (LM) objective to preserve general capability. Experimental results demonstrate that TriCon-Fair reduces discriminatory output beyond existing debiasing baselines while maintaining strong downstream performance. This suggests that our proposed TriCon-Fair offers a practical and ethical solution for sensitive NLP applications.

---

## 60. Towards classification-based representation learning for place recognition on LiDAR scans

**论文链接:** [http://arxiv.org/abs/2511.00738v2](http://arxiv.org/abs/2511.00738v2)

**作者:** Maksim Konoplia, Dmitrii Khizbullin

**发布时间:** 2025-11-01

### GPT解析

### 总结

本文提出了一种将地点识别作为多类分类问题的替代方法，通过为LiDAR扫描分配离散位置标签并训练编码器-解码器模型来直接分类位置，在NuScenes数据集上验证了其与对比学习方法相当的竞争性能，同时具有更高的训练效率和稳定性。

### 背景

地点识别是自动驾驶中的关键任务，允许车辆使用传感器数据确定自身位置。现有方法大多依赖于对比学习。

### 目的

探索一种替代对比学习的方法，将地点识别作为多类分类问题处理。

### 方法

为LiDAR扫描分配离散位置标签，训练编码器-解码器模型直接分类每个扫描的位置。

### 主要发现

在NuScenes数据集上评估显示，该方法与基于对比学习的方法具有竞争性能，同时在训练效率和稳定性方面具有优势。

### 结论

将地点识别作为多类分类问题是一种有效的替代方法，具有与对比学习方法相当的性能，并且在训练效率和稳定性方面具有优势。

### 翻译

地点识别是自动驾驶中的一个关键任务，它允许车辆使用传感器数据来确定自己的位置。虽然大多数现有方法依赖于对比学习，但我们通过将地点识别构建为一个多类分类问题来探索一种替代方法。我们的方法为LiDAR扫描分配离散的位置标签，并训练一个编码器-解码器模型来直接分类每个扫描的位置。我们在NuScenes数据集上评估了这种方法，并表明它与基于对比学习的方法相比具有竞争力的性能，同时在训练效率和稳定性方面提供优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于激光雷达(LiDAR)扫描的地点识别问题，即根据传感器数据确定车辆位置。这个问题在现实中非常重要，因为在GPS信号不可靠的环境（如城市峡谷、隧道或恶劣天气）中，准确的定位对自动驾驶车辆的导航、地图绘制和安全决策至关重要。在研究中，这个问题也很重要，因为现有方法大多基于对比学习，需要复杂的负样本挖掘策略且训练效率低、稳定性差。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到人脸识别领域分类方法的启发，思考是否可以将地点识别问题重新定义为分类任务。作者注意到地点不像标准分类任务中的对象那样有明确边界，因此通过将连续位置离散化为网格单元来解决这一问题。他们借鉴了PointNet++作为骨干网络处理点云，采用了两塔(two-tower)架构分离索引构建和查询服务，并参考了LCPR的数据集划分策略。作者设计了掩码交叉熵损失函数，避免惩罚预测到正确位置相邻网格的预测，解决了空间相关位置之间的梯度冲突问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将地点识别问题从传统的对比学习框架重新定义为多类别分类问题，通过将连续位置离散化为网格单元，训练模型直接预测激光扫描对应的离散位置类别。整体流程包括：1)数据准备和划分；2)位置离散化，将连续坐标转换为网格坐标并分配唯一类别标签；3)构建编码器-解码器模型，使用PointNet++处理点云并添加分类头；4)使用掩码交叉熵损失进行训练，避免惩罚相邻位置的预测；5)评估时使用KNN搜索在预构建的嵌入数据库中查找最相似的样本，计算召回率指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将地点识别重新定义为多类别分类问题；2)提出新颖的位置离散化方法，使用3元组表示离散位置；3)设计掩码交叉熵损失解决空间相关位置的梯度冲突；4)避免对比学习中的复杂负样本挖掘，提高训练效率；5)通过整合多地图数据展示大规模训练可行性。相比之前的工作，不同之处在于训练目标（分类vs度量学习）、负样本处理（无需复杂挖掘）、模型架构（添加分类头）和训练稳定性（分类方法更稳定）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于分类的新型地点识别方法，通过位置离散化和掩码损失函数，为激光雷达扫描的地点识别提供了更高效、更稳定的训练框架。'}


### 论文摘要

Place recognition is a crucial task in autonomous driving, allowing vehicles to determine their position using sensor data. While most existing methods rely on contrastive learning, we explore an alternative approach by framing place recognition as a multi-class classification problem. Our method assigns discrete location labels to LiDAR scans and trains an encoder-decoder model to classify each scan's position directly. We evaluate this approach on the NuScenes dataset and show that it achieves competitive performance compared to contrastive learning-based methods while offering advantages in training efficiency and stability.

---

## 61. Reasoning Planning for Language Models

**论文链接:** [http://arxiv.org/abs/2511.00521v1](http://arxiv.org/abs/2511.00521v1)

**作者:** Bao Nguyen, Hieu Trung Nguyen, Ruifeng She, Xiaojin Fu, Viet Anh Nguyen

**发布时间:** 2025-11-01

**备注:** 29 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为EPIC的集成规划与对比学习框架，用于解决语言模型生成中选择合适推理方法的问题，通过理论分析和创新框架实现了准确性和计算效率的平衡。

### 背景

在语言模型生成中，为给定查询选择合适的推理方法仍然是一个关键挑战。现有方法通常生成多个候选响应并使用聚合策略选择输出答案，且往往假设更多的候选答案会带来更高的准确性。

### 目的

重新审视'更多候选答案意味着更高准确性'这一假设，通过严谨的理论分析，推导固定生成分布和候选大小下标准聚合方法的准确性界限。

### 方法

提出EPIC（Ensemble Planning with Contrastive learning）框架，学习一个共享的表示空间来捕捉模型推理能力和查询方法兼容性，并将概率界限作为正则化项纳入效用驱动的优化中，平衡准确性和计算成本。

### 主要发现

通过理论分析对现有假设有了更深入的理解；EPIC能够在各种数学推理任务中一致地选择最优推理方法；在提高准确性的同时减少了计算开销。

### 结论

EPIC框架有效地解决了语言模型生成中选择合适推理方法的挑战，通过理论分析和创新框架实现了准确性和计算效率的平衡。

### 翻译

为给定查询选择合适的推理方法在语言模型生成中仍然是一个关键挑战。现有方法通常生成多个候选响应并使用聚合策略来选择输出答案，常常假设更多的候选答案会带来更高的准确性。我们通过严谨的理论分析重新审视这一假设，推导了固定生成分布和候选大小下标准聚合方法的准确性界限。基于这些见解，我们引入了EPIC，一种集成规划与对比学习框架，用于学习一个共享的表示空间，该空间能够捕捉模型推理能力和查询方法兼容性。EPIC将我们的概率界限作为正则化项纳入到效用驱动的优化中，平衡准确性和计算成本。在各种数学推理任务上的实验表明，EPIC能够一致地选择最优推理方法，在提高准确性的同时减少计算开销。我们的代码可以在https://github.com/nguyenngocbaocmt02/EPIC找到。


### 论文摘要

Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at https://github.com/nguyenngocbaocmt02/EPIC.

---

## 62. ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training

**论文链接:** [http://arxiv.org/abs/2511.00446v1](http://arxiv.org/abs/2511.00446v1)

**作者:** Xin Yao, Haiyang Zhao, Yimin Chen, Jiawei Guo, Kecheng Huang, Ming Zhao

**发布时间:** 2025-11-01

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

这篇论文介绍了一种名为ToxicTextCLIP的框架，用于生成高质量的对抗性文本来攻击CLIP模型在预训练阶段。

### 背景

CLIP模型通过自监督对比学习对齐大规模网络数据中的图像-文本对推动了视觉-语言建模，但它依赖未筛选的互联网数据，面临数据投毒和后门风险。现有研究主要关注基于图像的攻击，而同样对CLIP训练至关重要的文本模态尚未被充分探索。

### 目的

开发一种针对CLIP预训练阶段的高质量对抗文本生成框架，解决背景不一致导致的语义错位和背景一致文本稀缺性问题。

### 方法

ToxicTextCLIP框架迭代应用两种技术：1) 背景感知选择器，优先选择与目标类别背景内容对齐的文本；2) 背景驱动增强器，生成语义连贯且多样化的投毒样本。

### 主要发现

在分类和检索任务上的实验表明，ToxicTextCLIP实现了高达95.83%的投毒成功率和98.68%的后门Hit@1，同时成功绕过了RoCLIP、CleanCLIP和SafeCLIP防御。

### 结论

ToxicTextCLIP是一种有效的针对CLIP预训练阶段的文本投毒框架，能够高效生成高质量的对抗文本。

### 翻译

对比语言图像预训练（CLIP）模型通过自监督对比学习对齐大规模网络数据中的图像-文本对，显著推动了视觉-语言建模的发展。然而，它对未筛选的互联网来源数据的依赖使其面临数据投毒和后门风险。虽然现有研究主要调查基于图像的攻击，但对CLIP训练同样至关重要的文本模态仍未被充分探索。在这项工作中，我们引入了ToxicTextCLIP，一个用于在预训练阶段针对CLIP生成高质量对抗文本的框架。该框架解决了两个关键挑战：由背景与目标类别不一致导致的语义错位，以及背景一致文本的稀缺性。为此，ToxicTextCLIP迭代应用：1) 一个背景感知选择器，优先选择与目标类别背景内容对齐的文本；2) 一个背景驱动增强器，生成语义连贯且多样化的投毒样本。在分类和检索任务上的广泛实验表明，ToxicTextCLIP实现了高达95.83%的投毒成功率和98.68%的后门Hit@1，同时绕过了RoCLIP、CleanCLIP和SafeCLIP防御。源代码可通过https://github.com/xinyaocse/ToxicTextCLIP/访问。


### 论文摘要

The Contrastive Language-Image Pretraining (CLIP) model has significantly advanced vision-language modeling by aligning image-text pairs from large-scale web data through self-supervised contrastive learning. Yet, its reliance on uncurated Internet-sourced data exposes it to data poisoning and backdoor risks. While existing studies primarily investigate image-based attacks, the text modality, which is equally central to CLIP's training, remains underexplored. In this work, we introduce ToxicTextCLIP, a framework for generating high-quality adversarial texts that target CLIP during the pre-training phase. The framework addresses two key challenges: semantic misalignment caused by background inconsistency with the target class, and the scarcity of background-consistent texts. To this end, ToxicTextCLIP iteratively applies: 1) a background-aware selector that prioritizes texts with background content aligned to the target class, and 2) a background-driven augmenter that generates semantically coherent and diverse poisoned samples. Extensive experiments on classification and retrieval tasks show that ToxicTextCLIP achieves up to 95.83% poisoning success and 98.68% backdoor Hit@1, while bypassing RoCLIP, CleanCLIP and SafeCLIP defenses. The source code can be accessed via https://github.com/xinyaocse/ToxicTextCLIP/.

---

## 63. Simple and Behavior-Driven Augmentation for Recommendation with Rich Collaborative Signals

**论文链接:** [http://arxiv.org/abs/2511.00436v2](http://arxiv.org/abs/2511.00436v2)

**作者:** Doyun Choi, Cheonwoo Lee, Jaemin Yoo

**发布时间:** 2025-11-01

**备注:** 10 pages. This paper is accepted at IEEE BigData 2025 (Short)

### GPT解析

### 总结

本文提出了一种名为SCAR(Simple Collaborative Augmentation for Recommendation)的简单协作增强方法，用于改进图协同过滤中的对比学习效果。该方法通过生成伪交互而非删除信息来增强数据视图，在四个基准数据集上表现出色，尤其在稀疏数据场景中效果显著。

### 背景

对比学习(CI)已被广泛用于增强图协同过滤(GCF)的性能以实现个性化推荐。数据增强在对比学习的成功中起着关键作用，先前的工作设计了去除用户和项目之间噪声交互的增强方法。

### 目的

提出一种简单而直观的增强方法SCAR，旨在最大化对比学习对图协同过滤的有效性，同时避免复杂增强模块的缺点。

### 方法

SCAR不删除信息，而是利用从用户-项目交互中提取的协作信号生成伪交互，然后将这些伪交互添加到现有交互中或用来替换现有交互，从而生成更鲁棒的数据表示。

### 主要发现

在四个基准数据集上的实验表明，SCAR在关键评估指标上优于之前的基于对比学习的图协同过滤方法以及其他最先进的自监督学习方法。SCAR在不同的超参数设置下表现出强大的鲁棒性，并且在稀疏数据场景中特别有效。

### 结论

通过避免定义噪声的模糊性问题，SCAR能够保留核心信息并生成更可靠的数据视图，同时避免了过于复杂的增强模块，是一种更有效且直观的数据增强方法。

### 翻译

对比学习(CI)已被广泛用于增强图协同过滤(GCF)的性能以实现个性化推荐。由于数据增强在对比学习的成功中起着关键作用，先前的工作设计了去除用户和项目之间噪声交互的增强方法，以生成有效的增强视图。然而，定义'噪声'的模糊性持续存在丢失核心信息和生成不可靠数据视图的风险，同时增加了增强的整体复杂性。在本文中，我们提出了用于推荐的简单协作增强(SCAR)，这是一种新颖而直观的增强方法，旨在最大化对比学习对图协同过滤的有效性。SCAR不删除信息，而是利用从用户-项目交互中提取的协作信号生成伪交互，然后将这些伪交互添加到现有交互中或用来替换现有交互。这产生了更鲁棒的表示，同时避免了过于复杂的增强模块的缺陷。我们在四个基准数据集上进行了实验，表明SCAR在关键评估指标上优于之前的基于对比学习的图协同过滤方法以及其他最先进的自监督学习方法。SCAR在不同的超参数设置下表现出强大的鲁棒性，并且在稀疏数据场景中特别有效。


### 论文摘要

Contrastive learning (CL) has been widely used for enhancing the performance of graph collaborative filtering (GCF) for personalized recommendation. Since data augmentation plays a crucial role in the success of CL, previous works have designed augmentation methods to remove noisy interactions between users and items in order to generate effective augmented views. However, the ambiguity in defining ''noisiness'' presents a persistent risk of losing core information and generating unreliable data views, while increasing the overall complexity of augmentation. In this paper, we propose Simple Collaborative Augmentation for Recommendation (SCAR), a novel and intuitive augmentation method designed to maximize the effectiveness of CL for GCF. Instead of removing information, SCAR leverages collaborative signals extracted from user-item interactions to generate pseudo-interactions, which are then either added to or used to replace existing interactions. This results in more robust representations while avoiding the pitfalls of overly complex augmentation modules. We conduct experiments on four benchmark datasets and show that SCAR outperforms previous CL-based GCF methods as well as other state-of-the-art self-supervised learning approaches across key evaluation metrics. SCAR exhibits strong robustness across different hyperparameter settings and is particularly effective in sparse data scenarios.

---

## 64. Mutual Information guided Visual Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.00028v1](http://arxiv.org/abs/2511.00028v1)

**作者:** Hanyang Chen, Yanchao Yang

**发布时间:** 2025-10-26

**备注:** Tech Report - Undergraduate Thesis - 2023

### GPT解析

### 总结

本文研究了基于互信息的数据选择方法，以提高表征学习在开放环境中的泛化能力。

### 背景

使用InfoNCE损失的表征学习方法能够有效减少人工标注，但数据选择和增强仍依赖人工假设或工程方法，可能不是最优的。例如，对比学习中的数据增强主要关注颜色抖动，旨在模拟真实世界的光照变化。

### 目的

研究基于从实际分布计算互信息来选择训练数据的潜力，使学习到的特征在开放环境中具有更好的泛化能力。

### 方法

考虑在自然扰动（如颜色变化和运动）下表现出高互信息的场景补丁作为对比损失学习的正样本，提出了一种基于互信息的数据增强方法。

### 主要发现

在多个基准测试上评估了所提出的互信息感知数据增强方法，证明了其有效性。

### 结论

基于互信息的数据选择方法是一个有前途的未来研究方向。

### 翻译

使用InfoNCE损失的表征学习方法已证明通过训练不变性神经网络特征提取器能够显著减少人工标注工作量。尽管不同变体的训练目标遵循数据与学习特征之间的信息最大化原则，但数据选择和增强仍然依赖人类假设或工程方法，这可能不是最优的。例如，对比学习中的数据增强主要关注颜色抖动，旨在模拟真实世界的光照变化。在本工作中，我们研究了基于从实际分布计算的互信息来选择训练数据的潜力，原则上，这应该使学习到的特征在开放环境中应用时具有更好的泛化能力。具体而言，我们将具有自然扰动（如颜色变化和运动）下表现出高互信息的场景补丁视为使用对比损失学习的正样本。我们在多个最先进的表征学习框架的几个基准上评估了所提出的互信息感知数据增强方法，证明了其有效性，并确立了其作为未来研究的有前途的方向。


### 论文摘要

Representation learning methods utilizing the InfoNCE loss have demonstrated considerable capacity in reducing human annotation effort by training invariant neural feature extractors. Although different variants of the training objective adhere to the information maximization principle between the data and learned features, data selection and augmentation still rely on human hypotheses or engineering, which may be suboptimal. For instance, data augmentation in contrastive learning primarily focuses on color jittering, aiming to emulate real-world illumination changes. In this work, we investigate the potential of selecting training data based on their mutual information computed from real-world distributions, which, in principle, should endow the learned features with better generalization when applied in open environments. Specifically, we consider patches attached to scenes that exhibit high mutual information under natural perturbations, such as color changes and motion, as positive samples for learning with contrastive loss. We evaluate the proposed mutual-information-informed data augmentation method on several benchmarks across multiple state-of-the-art representation learning frameworks, demonstrating its effectiveness and establishing it as a promising direction for future research.

---

## 65. Causal Graph Neural Networks for Healthcare

**论文链接:** [http://arxiv.org/abs/2511.02531v1](http://arxiv.org/abs/2511.02531v1)

**作者:** Munib Mesinovic, Max Buhlan, Tingting Zhu

**发布时间:** 2025-11-04

### GPT解析

### 总结

医疗人工智能系统在不同机构部署时经常失败，表现为性能下降和历史数据中歧视模式的延续。这种脆弱性部分源于学习统计关联而非因果机制。因果图神经网络通过结合生物医学数据的图表示和因果推理原理，解决了分布转移、歧视和不可解释性的三重危机，学习不变的机制而非虚假相关性。

### 背景

医疗人工智能系统在不同机构部署时经常失败，有记录显示性能下降和延续了历史数据中的歧视模式。这种脆弱性部分源于学习统计关联而非因果机制。

### 目的

解决医疗人工智能系统的分布转移、歧视和不可解释性三重危机，通过学习不变的机制而非虚假相关性来提高系统性能和公平性。

### 方法

因果图神经网络，结合结构因果模型、解纠缠的因果表征学习，以及图上的干预预测和反事实推理技术。

### 主要发现

因果图神经网络已应用于精神疾病诊断（脑网络分析）、癌症亚型分类（多组学因果整合）、连续生理监测（机械解释）和药物推荐（纠正处方偏见）。这些进展为患者特异性因果数字孪生奠定基础，结合大型语言模型进行假设生成和因果图神经网络进行机械验证。

### 结论

仍存在重大障碍，包括计算需求阻碍实时部署、验证挑战需要多模态证据三角测量、以及因果清洗风险。提出分层框架区分受因果启发的架构和因果验证的发现，并确定关键研究优先事项，以做出因果而非纯粹关联的声明。

### 翻译

医疗人工智能系统在不同机构部署时经常失败，有记录显示性能下降和延续了历史数据中的歧视模式。这种脆弱性部分源于学习统计关联而非因果机制。因果图神经网络通过结合生物医学数据的图表示和因果推理原理，解决了分布转移、歧视和不可解释性的三重危机，学习不变的机制而非虚假相关性。本综述审视了方法论基础，包括结构因果模型、解纠缠的因果表征学习，以及图上的干预预测和反事实推理技术。我们分析了具有临床价值的应用，包括通过脑网络分析的精神疾病诊断、通过多组学因果整合的癌症亚型分类、具有机械解释的连续生理监测，以及纠正处方偏见的药物推荐。这些进展为患者特异性因果数字孪生奠定基础，使计算机内临床实验成为可能，并整合大型语言模型进行假设生成和因果图神经网络进行机械验证。仍然存在重大障碍，包括计算需求阻碍实时部署、验证挑战需要超越交叉验证的多模态证据三角测量，以及因果清洗的风险（方法使用因果术语但缺乏严格的证据支持）。我们提出分层框架区分受因果启发的架构和因果验证的发现，并确定关键研究优先事项，以做出因果而非纯粹关联的声明。


### 论文摘要

Healthcare artificial intelligence systems routinely fail when deployed across institutions, with documented performance drops and perpetuation of discriminatory patterns embedded in historical data. This brittleness stems, in part, from learning statistical associations rather than causal mechanisms. Causal graph neural networks address this triple crisis of distribution shift, discrimination, and inscrutability by combining graph-based representations of biomedical data with causal inference principles to learn invariant mechanisms rather than spurious correlations. This Review examines methodological foundations spanning structural causal models, disentangled causal representation learning, and techniques for interventional prediction and counterfactual reasoning on graphs. We analyse applications demonstrating clinical value across psychiatric diagnosis through brain network analysis, cancer subtyping via multi-omics causal integration, continuous physiological monitoring with mechanistic interpretation, and drug recommendation correcting prescription bias. These advances establish foundations for patient-specific Causal Digital Twins, enabling in silico clinical experimentation, with integration of large language models for hypothesis generation and causal graph neural networks for mechanistic validation. Substantial barriers remain, including computational requirements precluding real-time deployment, validation challenges demanding multi-modal evidence triangulation beyond cross-validation, and risks of causal-washing where methods employ causal terminology without rigorous evidentiary support. We propose tiered frameworks distinguishing causally-inspired architectures from causally-validated discoveries and identify critical research priorities making causal rather than purely associational claims.

---

## 66. Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization

**论文链接:** [http://arxiv.org/abs/2511.02489v1](http://arxiv.org/abs/2511.02489v1)

**作者:** Tao Liu, Kan Ren, Qian Chen

**发布时间:** 2025-11-04

**备注:** 20 pages, Submitted to IEEE TIM

### GPT解析

### 总结

本文提出了一种基于目标检测的跨视图无人机定位框架，通过图神经网络处理图像间和图像内节点关系，有效解决了GNSS受限区域的定位问题，并在异构航空图像匹配中表现出色。

### 背景

随着低空经济的快速发展，无人机在巡逻系统中的测量和跟踪变得至关重要。然而，在GNSS受限区域，基于卫星的定位方法容易失效，需要新的定位解决方案。

### 目的

开发一个跨视图无人机定位框架，通过目标检测进行地图匹配，有效解决跨时间、跨视图、异构航空图像匹配问题，提高无人机在无卫星信号环境下的定位能力。

### 方法

利用现代目标检测从无人机和卫星图像中提取显著实例，集成图神经网络推理图像间和图像内节点关系，采用细粒度的基于图的节点相似度度量方法实现检索和定位。与传统图像检索方法和分类任务方法相比，避免了极坐标重投影、透视变换或生成对抗网络可能带来的错位、内容损失和真实性有限问题。

### 主要发现

在公共和真实世界数据集上的广泛实验表明，该方法能有效处理异构外观差异，具有良好泛化能力，适用于具有更大模态差距的场景，如红外-可见光图像匹配。

### 结论

该框架为GNSS受限区域的无人机定位提供了有效解决方案，相关数据集将在https://github.com/liutao23/ODGNNLoc.git公开，为后续研究提供支持。

### 翻译

随着低空经济的快速增长，无人机已成为巡逻系统中测量和跟踪的关键工具。然而，在GNSS受限区域，基于卫星的定位方法容易失效。本文提出了一种跨视图无人机定位框架，通过目标检测执行地图匹配，旨在有效解决跨时间、跨视图、异构航空图像匹配问题。在典型流程中，无人机视觉定位被表述为图像检索问题：提取特征构建定位地图，并通过将查询图像与具有已知姿态的参考数据库匹配来估计其姿态。由于公开的无人机定位数据集有限，许多方法将定位重新表述为分类任务，并依赖这些数据集中的场景标签来确保准确性。其他方法使用极坐标重投影、透视变换或生成对抗网络来减少跨域差异；然而，它们可能存在错位、内容损失和真实性有限的问题。相比之下，我们利用现代目标检测从无人机和卫星图像中准确提取显著实例，并集成图神经网络来推理图像间和图像内节点关系。使用细粒度的、基于图的节点相似度度量方法，我们的方法实现了强大的检索和定位性能。在公共和真实世界数据集上的广泛实验表明，我们的方法能有效处理异构外观差异，并具有良好的泛化能力，使其适用于具有更大模态差距的场景，如红外-可见光图像匹配。我们的数据集将在以下网址公开：https://github.com/liutao23/ODGNNLoc.git。


### 论文摘要

With the rapid growth of the low-altitude economy, UAVs have become crucial for measurement and tracking in patrol systems. However, in GNSS-denied areas, satellite-based localization methods are prone to failure. This paper presents a cross-view UAV localization framework that performs map matching via object detection, aimed at effectively addressing cross-temporal, cross-view, heterogeneous aerial image matching. In typical pipelines, UAV visual localization is formulated as an image-retrieval problem: features are extracted to build a localization map, and the pose of a query image is estimated by matching it to a reference database with known poses. Because publicly available UAV localization datasets are limited, many approaches recast localization as a classification task and rely on scene labels in these datasets to ensure accuracy. Other methods seek to reduce cross-domain differences using polar-coordinate reprojection, perspective transformations, or generative adversarial networks; however, they can suffer from misalignment, content loss, and limited realism. In contrast, we leverage modern object detection to accurately extract salient instances from UAV and satellite images, and integrate a graph neural network to reason about inter-image and intra-image node relationships. Using a fine-grained, graph-based node-similarity metric, our method achieves strong retrieval and localization performance. Extensive experiments on public and real-world datasets show that our approach handles heterogeneous appearance differences effectively and generalizes well, making it applicable to scenarios with larger modality gaps, such as infrared-visible image matching. Our dataset will be publicly available at the following URL: https://github.com/liutao23/ODGNNLoc.git.

---

## 67. Using ensemble learning with hybrid graph neural networks and transformers to predict traffic in cities

**论文链接:** [http://arxiv.org/abs/2511.02484v1](http://arxiv.org/abs/2511.02484v1)

**作者:** Ismail Zrigui, Samira Khoulji, Mohamed Larbi Kerkeb

**发布时间:** 2025-11-04

**DOI:** 10.5281/zenodo.17521951

### GPT解析

### 总结

本文提出了一种名为HybridST的混合架构，用于提高城市交通预测的准确性，特别是在复杂的多模式交通环境中。

### 背景

智能交通系统(ITS)在城市交通预测方面仍然存在困难，特别是在具有复杂时空动态的大规模多模式交通环境中。

### 目的

开发一种能够准确捕捉空间依赖性、长期时间模式和外部信号的新型交通预测模型。

### 方法

提出HybridST混合架构，整合图神经网络(GNNs)、多头时间序列Transformer和监督集成学习方法(XGBoost或RandomForest)，以综合捕捉空间依赖性、长期时间模式和包括天气、日历或控制状态在内的外部信号。

### 主要发现

在METR-LA、PEMS-BAY和Seattle Loop tree三个公开基准数据集上的实验表明，HybridST在MAE和RMSE等重要指标上始终优于经典基线模型(LSTM, GCN, DCRNN, PDFormer)，同时保持良好的可扩展性和易于理解的特点。

### 结论

HybridST框架为实时城市交通规划、能源优化和缓解拥堵策略提供了有前景的途径，特别适用于智慧城市和2030年世界杯等重大活动的交通管理。

### 翻译

智能交通系统(ITS)在城市交通预测方面仍然面临挑战，特别是在具有复杂时空动态的大规模多模式交通环境中。本文提出了HybridST，一种混合架构，整合了图神经网络(GNNs)、多头时间序列Transformer和监督集成学习方法(XGBoost或RandomForest)，以共同捕捉空间依赖性、长期时间模式和外部信号，包括天气、日历或控制状态。我们在METR-LA、PEMS-BAY和Seattle Loop tree三个公开基准数据集上测试了我们的模型。这些数据集涵盖了从高速公路传感器网络到车路协同感知的各种场景。实验结果表明，HybridST在MAE和RMSE等重要指标上始终优于经典基线模型(LSTM, GCN, DCRNN, PDFormer)，同时仍然保持高度可扩展性和易于理解的特点。所提出的框架为实时城市交通规划、能源优化和缓解拥堵策略提供了有前景的途径，特别是在智慧城市和2030年世界杯等重大活动的框架内。


### 论文摘要

Intelligent transportation systems (ITS) still have a hard time accurately predicting traffic in cities, especially in big, multimodal settings with complicated spatiotemporal dynamics. This paper presents HybridST, a hybrid architecture that integrates Graph Neural Networks (GNNs), multi-head temporal Transformers, and supervised ensemble learning methods (XGBoost or Random Forest) to collectively capture spatial dependencies, long-range temporal patterns, and exogenous signals, including weather, calendar, or control states. We test our model on the METR-LA, PEMS-BAY, and Seattle Loop tree public benchmark datasets. These datasets include situations ranging from freeway sensor networks to vehicle-infrastructure cooperative perception. Experimental results show that HybridST consistently beats classical baselines (LSTM, GCN, DCRNN, PDFormer) on important metrics like MAE and RMSE, while still being very scalable and easy to understand. The proposed framework presents a promising avenue for real-time urban mobility planning, energy optimization, and congestion alleviation strategies, especially within the framework of smart cities and significant events such as the 2030 FIFA World Cup.

---

## 68. Evolving Graph Learning for Out-of-Distribution Generalization in Non-stationary Environments

**论文链接:** [http://arxiv.org/abs/2511.02354v1](http://arxiv.org/abs/2511.02354v1)

**作者:** Qingyun Sun, Jiayi Luo, Haonan Yuan, Xingcheng Fu, Hao Peng, Jianxin Li, Philip S. Yu

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了一种名为EvoOOD的新型演化图学习框架，通过环境感知不变模式识别来解决动态图在分布偏移下的泛化能力问题。

### 背景

图神经网络在动态图的空间和时间模式利用方面表现出色，但现有GNN在分布偏移下泛化能力差，这在动态场景中是不可避免的。

### 目的

探索动态图生成在潜在非平稳环境中对分布外(OOD)泛化的影响，并提出一种用于OOD泛化的新型演化图学习框架。

### 方法

设计环境序列变分自编码器建模环境演化并推断环境分布；引入环境感知不变模式识别机制解决环境多样化问题；使用混合实例化环境样本对单个节点进行细粒度因果干预。

### 主要发现

该方法有助于区分用于OOD预测的时空不变模式，特别是在非平稳环境中；实验证明了EvoOOD在真实世界和合成动态数据集分布偏移下的优越性。

### 结论

据作者所知，这是首次从环境演化角度研究动态图OOD泛化问题。

### 翻译

图神经网络在利用动态图上的空间和时间模式方面表现出色。然而，现有的GNN在分布偏移下表现出较差的泛化能力，这在动态场景中是不可避免的。随着动态图生成在潜在非平稳环境中不断推进，探索它们对分布外(OOD)泛化的影响至关重要。本文通过环境感知不变模式识别，提出了一种用于OOD泛化的新型演化图学习框架(EvoOOD)。具体来说，我们首先设计了一个环境序列变分自编码器来建模环境演化并推断潜在环境分布。然后，我们引入了一种环境感知不变模式识别机制，通过推断的分布来解决环境多样化问题。最后，我们使用混合实例化环境样本对单个节点进行细粒度因果干预。这种方法有助于区分用于OOD预测的时空不变模式，特别是在非平稳环境中。实验结果证明了EvoOOD在真实世界和合成动态数据集分布偏移下的优越性。据我们所知，这是首次从环境演化角度研究动态图OOD泛化问题。


### 论文摘要

Graph neural networks have shown remarkable success in exploiting the spatial and temporal patterns on dynamic graphs. However, existing GNNs exhibit poor generalization ability under distribution shifts, which is inevitable in dynamic scenarios. As dynamic graph generation progresses amid evolving latent non-stationary environments, it is imperative to explore their effects on out-of-distribution (OOD) generalization. This paper proposes a novel Evolving Graph Learning framework for OOD generalization (EvoOOD) by environment-aware invariant pattern recognition. Specifically, we first design an environment sequential variational auto-encoder to model environment evolution and infer the underlying environment distribution. Then, we introduce a mechanism for environment-aware invariant pattern recognition, tailored to address environmental diversification through inferred distributions. Finally, we conduct fine-grained causal interventions on individual nodes using a mixture of instantiated environment samples. This approach helps to distinguish spatio-temporal invariant patterns for OOD prediction, especially in non-stationary environments. Experimental results demonstrate the superiority of EvoGOOD on both real-world and synthetic dynamic datasets under distribution shifts. To the best of our knowledge, it is the first attempt to study the dynamic graph OOD generalization problem from the environment evolution perspective.

---

## 69. Link prediction Graph Neural Networks for structure recognition of Handwritten Mathematical Expressions

**论文链接:** [http://arxiv.org/abs/2511.02288v1](http://arxiv.org/abs/2511.02288v1)

**作者:** Cuong Tuan Nguyen, Ngoc Tuan Nguyen, Triet Hoang Minh Dao, Huy Minh Nhat, Huy Truong Dinh

**发布时间:** 2025-11-04

**备注:** accepted for ICDAR2025-WML

### GPT解析

### 总结

该研究提出了一种基于图神经网络的手写数学表达式识别方法，通过将数学表达式建模为图结构，结合深度BLSTM网络和图神经网络进行识别和结构优化。

### 背景

手写数学表达式识别是一个具有挑战性的任务，需要同时识别符号并理解它们之间的空间关系。

### 目的

开发一种能够准确识别手写数学表达式并正确理解其结构的方法。

### 方法

将手写数学表达式建模为图，其中节点代表符号，边代表空间依赖关系；使用深度BLSTM网络进行符号分割、识别和空间关系分类，形成初始原始图；使用2D-CFG解析器生成所有可能的空间关系；应用基于GNN的链接预测模型优化结构，移除不必要的连接，形成最终的符号标记图。

### 主要发现

实验结果表明，所提出的方法在手写数学表达式结构识别方面具有良好的性能。

### 结论

基于图神经网络的方法在处理手写数学表达式识别任务中是有效的，能够准确识别符号并正确理解它们之间的空间关系。

### 翻译

我们提出了一种基于图神经网络的手写数学表达式识别方法，通过将手写数学表达式建模为图来表示，其中节点代表符号，边捕获空间依赖关系。使用深度BLSTM网络进行符号分割、识别和空间关系分类，形成初始原始图。然后，2D-CFG解析器生成所有可能的空间关系，而基于GNN的链接预测模型通过移除不必要的连接来优化结构，最终形成符号标记图。实验结果证明了我们方法的有效性，在手写数学表达式结构识别方面显示出良好的性能。


### 论文摘要

We propose a Graph Neural Network (GNN)-based approach for Handwritten Mathematical Expression (HME) recognition by modeling HMEs as graphs, where nodes represent symbols and edges capture spatial dependencies. A deep BLSTM network is used for symbol segmentation, recognition, and spatial relation classification, forming an initial primitive graph. A 2D-CFG parser then generates all possible spatial relations, while the GNN-based link prediction model refines the structure by removing unnecessary connections, ultimately forming the Symbol Label Graph. Experimental results demonstrate the effectiveness of our approach, showing promising performance in HME structure recognition.

---

## 70. PrivGNN: High-Performance Secure Inference for Cryptographic Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.02185v1](http://arxiv.org/abs/2511.02185v1)

**作者:** Fuyi Wang, Zekai Chen, Mingyuan Fan, Jianying Zhou, Lei Pan, Leo Yu Zhang

**发布时间:** 2025-11-04

**备注:** Accepted to FC'25

### GPT解析

### 总结

该研究设计并实现了一种名为SysName的轻量级密码学方案，用于在云环境中进行安全图神经网络推断，通过混合加法和函数秘密共享技术，显著提升了计算效率。

### 背景

图神经网络是分析和学习图结构数据的强大工具，可在多种服务中应用。但在隐私关键型云环境中部署这些服务需要开发安全推断协议来保护敏感的图结构数据。现有解决方案主要关注图像和文本数据的卷积模型，而保护图神经网络和图结构数据的安全挑战相对未被充分探索。

### 目的

开发一种轻量级密码学方案，用于在云环境中安全地进行图神经网络推断，保护敏感图结构数据的同时保持高效性和准确性。

### 方法

设计并实现SysName方案，在安全两方计算框架下混合使用加法秘密共享和函数秘密共享，基于一系列新颖的交互协议，优化了线性层和非线性层的计算效率。

### 主要发现

SysName与最先进解决方案相比，线性层速度提升1.5倍至1.7倍，非线性层速度提升2倍至15倍。在四个数据集上的实验表明，安全预测速度提升1.3倍至4.7倍，同时保持与明文图属性推断相当的准确性。

### 结论

SysName是一种高效、安全的图神经网络推断方案，能够在保护数据隐私的同时提供与明文计算相当的准确性和显著更快的计算速度。

### 翻译

图神经网络是分析和学习图结构数据的强大工具，促进广泛的服务应用。在隐私关键的云环境中部署此类服务需要开发安全推断协议以保护敏感的图结构数据。然而，现有的安全推断解决方案主要关注图像和文本数据的卷积模型，而保护图神经网络和图结构数据的挑战相对未被充分探索。在本工作中，我们设计、实现并评估了SysName，这是一种用于云中以图为中心推断的轻量级密码学方案。通过在安全两方计算中混合加法秘密共享和函数秘密共享，SysName基于一系列新颖的交互协议精心设计，与最先进解决方案相比，线性层速度提升1.5倍至1.7倍，非线性层速度提升2倍至15倍。提供了彻底的理论分析以证明SysName的正确性、安全性和轻量级特性。在四个数据集上的广泛实验表明，SysName具有卓越的效率，安全预测速度提升1.3倍至4.7倍，同时保持与明文图属性推断相当的准确性。


### 论文摘要

Graph neural networks (GNNs) are powerful tools for analyzing and learning from graph-structured (GS) data, facilitating a wide range of services. Deploying such services in privacy-critical cloud environments necessitates the development of secure inference (SI) protocols that safeguard sensitive GS data. However, existing SI solutions largely focus on convolutional models for image and text data, leaving the challenge of securing GNNs and GS data relatively underexplored. In this work, we design, implement, and evaluate $\sysname$, a lightweight cryptographic scheme for graph-centric inference in the cloud. By hybridizing additive and function secret sharings within secure two-party computation (2PC), $\sysname$ is carefully designed based on a series of novel 2PC interactive protocols that achieve $1.5\times \sim 1.7\times$ speedups for linear layers and $2\times \sim 15\times$ for non-linear layers over state-of-the-art (SotA) solutions. A thorough theoretical analysis is provided to prove $\sysname$'s correctness, security, and lightweight nature. Extensive experiments across four datasets demonstrate $\sysname$'s superior efficiency with $1.3\times \sim 4.7\times$ faster secure predictions while maintaining accuracy comparable to plaintext graph property inference.

---

## 71. Rethinking LLM Human Simulation: When a Graph is What You Need

**论文链接:** [http://arxiv.org/abs/2511.02135v1](http://arxiv.org/abs/2511.02135v1)

**作者:** Joseph Suh, Suhong Moon, Serina Chang

**发布时间:** 2025-11-03

**备注:** Code: https://github.com/schang-lab/gems

### GPT解析

### 总结

本研究提出了一种名为GEMS的基于图的轻量级模型，用于人类模拟任务，在保持与大型语言模型相当或更好准确性的同时，显著提高了效率、可解释性和透明度。

### 背景

大型语言模型越来越多地被用来模拟人类，应用范围从调查预测到决策制定。然而，这些模型通常体积庞大，计算资源需求高。

### 目的

探究在人类模拟任务中，是否可以使用更小、更专业的模型替代大型语言模型，特别是在个体在离散选项中做出选择的场景下。

### 方法

提出Graph-basEd Models for human Simulation (GEMS)框架，将离散选择模拟任务转化为图上的链接预测问题，利用关系知识并仅在需要时融入语言表示。使用图神经网络作为基础架构。

### 主要发现

在三个模拟数据集上的三种关键设置中评估显示，尽管图神经网络模型比大型语言模型小三个数量级，但它能够匹配或超越强大的大型语言模型基线模型的性能。GEMS在准确性、效率、可解释性和透明度方面均表现出色。

### 结论

基于图的建模作为大型语言模型用于人类模拟的轻量级替代方案具有显著前景，特别是在需要高效、可解释和透明的模拟系统的场景中。

### 翻译

大型语言模型正越来越多地被用来模拟人类，应用范围从调查预测到决策制定。然而，大型语言模型是否严格必要，还是更小、领域专用的模型就足够了？我们确定了一类模拟问题，即个体在离散选项中做出选择的问题，在这类问题中，图神经网络可以匹配甚至超越强大的大型语言模型基线模型，尽管小三个数量级。我们提出了基于图的人类模拟模型，它将离散选择模拟任务作为图上的链接预测问题，利用关系知识，仅在需要时才融入语言表示。在三个模拟数据集上的三种关键设置中的评估表明，GEMS实现了与大型语言模型相当或更好的准确性，同时具有更高的效率、可解释性和透明度，突显了基于图的建模作为大型语言模型用于人类模拟的轻量级替代方案的潜力。我们的代码可在https://github.com/schang-lab/gems获取。


### 论文摘要

Large language models (LLMs) are increasingly used to simulate humans, with applications ranging from survey prediction to decision-making. However, are LLMs strictly necessary, or can smaller, domain-grounded models suffice? We identify a large class of simulation problems in which individuals make choices among discrete options, where a graph neural network (GNN) can match or surpass strong LLM baselines despite being three orders of magnitude smaller. We introduce Graph-basEd Models for human Simulation (GEMS), which casts discrete choice simulation tasks as a link prediction problem on graphs, leveraging relational knowledge while incorporating language representations only when needed. Evaluations across three key settings on three simulation datasets show that GEMS achieves comparable or better accuracy than LLMs, with far greater efficiency, interpretability, and transparency, highlighting the promise of graph-based modeling as a lightweight alternative to LLMs for human simulation. Our code is available at https://github.com/schang-lab/gems.

---

## 72. Predicting Microbial Interactions Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.02038v1](http://arxiv.org/abs/2511.02038v1)

**作者:** Elham Gholamzadeh, Kajal Singla, Nico Scherf

**发布时间:** 2025-11-03

**备注:** 9 pages, 3 figures, NeurIPS 2025 Workshop New Perspectives in Graph  Machine Learning

### GPT解析

### 总结

本研究利用图神经网络(GNNs)预测微生物间的相互作用类型，包括二元相互作用(正/负)和复杂相互作用类型(如互利共生、竞争和寄生)，取得了80.44%的F1分数，显著优于传统XGBoost模型的72.76%。

### 背景

预测物种间相互作用是微生物生态学中的一个关键挑战，因为这些相互作用对微生物群落的结构和活性起着决定性作用。

### 目的

利用单培养生长能力、与其他物种的相互作用以及系统发育数据来预测微生物相互作用的负向或正向效应。

### 方法

使用包含超过7,500个相互作用的数据集(涉及两个分类群的20个物种在40种不同碳条件下的共培养)训练模型；构建微生物相互作用的边图，利用图神经网络(GNNs)预测相互作用模式。

### 主要发现

模型不仅能预测二元相互作用，还能分类更复杂的相互作用类型；初始结果F1分数达80.44%，显著优于文献中可比的传统XGBoost模型(72.76%)。

### 结论

图神经网络是预测微生物相互作用的强大工具，为理解微生物群落结构和功能提供了有效方法。

### 翻译

预测物种间相互作用是微生物生态学中的一个关键挑战，因为这些相互作用对微生物群落的结构和活性至关重要。在本工作中，我们使用单培养生长能力、与其他物种的相互作用以及系统发育数据来预测相互作用的负向或正向效应。更准确地说，我们使用了最大的可用成对相互作用数据集之一来训练我们的模型，包含在40种不同碳条件下共培养的两个分类群的20个物种之间的7,500多个相互作用，主要关注Nestor等人[28]的工作。在本工作中，我们提出图神经网络(GNNs)作为预测效应方向的有力分类器。我们构建成对微生物相互作用的边图，以利用单个共培养实验中的共享信息，并使用GNNs预测相互作用模式。我们的模型不仅可以预测二元相互作用(正/负)，还可以分类更复杂的相互作用类型，如互利共生、竞争和寄生。我们的初步结果令人鼓舞，F1分数达到80.44%。这显著优于文献中可比的方法，包括传统的极端梯度提升(XGBoost)模型，后者报告的F1分数为72.76%。


### 论文摘要

Predicting interspecies interactions is a key challenge in microbial ecology, as these interactions are critical to determining the structure and activity of microbial communities. In this work, we used data on monoculture growth capabilities, interactions with other species, and phylogeny to predict a negative or positive effect of interactions. More precisely, we used one of the largest available pairwise interaction datasets to train our models, comprising over 7,500 interactions be- tween 20 species from two taxonomic groups co-cultured under 40 distinct carbon conditions, with a primary focus on the work of Nestor et al.[28 ]. In this work, we propose Graph Neural Networks (GNNs) as a powerful classifier to predict the direction of the effect. We construct edge-graphs of pairwise microbial interactions in order to leverage shared information across individual co-culture experiments, and use GNNs to predict modes of interaction. Our model can not only predict binary interactions (positive/negative) but also classify more complex interaction types such as mutualism, competition, and parasitism. Our initial results were encouraging, achieving an F1-score of 80.44%. This significantly outperforms comparable methods in the literature, including conventional Extreme Gradient Boosting (XGBoost) models, which reported an F1-score of 72.76%.

---

## 73. HyperNQ: A Hypergraph Neural Network Decoder for Quantum LDPC Codes

**论文链接:** [http://arxiv.org/abs/2511.01741v1](http://arxiv.org/abs/2511.01741v1)

**作者:** Ameya S. Bhave, Navnil Choudhury, Kanad Basu

**发布时间:** 2025-11-03

**备注:** 6 pages, 4 figures, Submitted to the IEEE International Conference on  Communications (ICC 2026). Preprint version

### GPT解析

### 总结

HyperNQ是一种基于超图神经网络的量子低密度奇偶校验码解码器，通过利用超边捕获高阶稳定器约束，显著提高了量子错误纠正性能。

### 背景

量子计算需要有效的错误纠正策略来缓解噪声和退相干问题。量子低密度奇偶校验码通过支持恒定速率编码和稀疏奇偶校验结构，已成为可扩展量子错误纠正应用的有前途的解决方案。

### 目的

开发一种能够捕获高阶稳定器约束的QLDPC解码器，以解决传统解码方法如置信传播在短循环存在时收敛性差的问题，以及图神经网络仅限于成对交互而无法捕获高阶相关性的限制。

### 方法

提出HyperNQ，第一个基于超图神经网络的QLDPC解码器，利用超边捕获高阶稳定器约束，实现高度表达性和紧凑的解码。采用两阶段消息传递方案，并在伪阈值区域评估解码器性能。

### 主要发现

在伪阈值标记以下，HyperNQ将逻辑错误率比置信传播提高了最多84%，比基于图神经网络的策略提高了50%，展示了优于现有最先进解码器的性能。

### 结论

HyperNQ通过超图神经网络有效解决了传统QLDPC解码方法的局限性，为量子错误纠正提供了一种创新且高效的解决方案。

### 翻译

量子计算需要有效的错误纠正策略来缓解噪声和退相干。量子低密度奇偶校验码通过支持恒定速率编码和稀疏奇偶校验结构，已成为可扩展量子错误纠正应用的有前途的解决方案。然而，通过置信传播等传统方法解码QLDPC码在存在短循环时收敛性差。图神经网络等机器学习技术利用节点特征进行学习消息传递，但它们仅限于Tanner图上的成对交互，限制了捕获高阶相关性的能力。在这项工作中，我们提出了HyperNQ，这是第一个基于超图神经网络的QLDPC解码器，通过利用超边捕获高阶稳定器约束，从而实现高度表达性和紧凑的解码。我们使用两阶段消息传递方案，并在伪阈值区域评估解码器。在伪阈值标记以下，HyperNQ将逻辑错误率比BP提高了最多84%，比基于GNN的策略提高了50%，展示了优于现有最先进解码器的性能。


### 论文摘要

Quantum computing requires effective error correction strategies to mitigate noise and decoherence. Quantum Low-Density Parity-Check (QLDPC) codes have emerged as a promising solution for scalable Quantum Error Correction (QEC) applications by supporting constant-rate encoding and a sparse parity-check structure. However, decoding QLDPC codes via traditional approaches such as Belief Propagation (BP) suffers from poor convergence in the presence of short cycles. Machine learning techniques like Graph Neural Networks (GNNs) utilize learned message passing over their node features; however, they are restricted to pairwise interactions on Tanner graphs, which limits their ability to capture higher-order correlations. In this work, we propose HyperNQ, the first Hypergraph Neural Network (HGNN)- based QLDPC decoder that captures higher-order stabilizer constraints by utilizing hyperedges-thus enabling highly expressive and compact decoding. We use a two-stage message passing scheme and evaluate the decoder over the pseudo-threshold region. Below the pseudo-threshold mark, HyperNQ improves the Logical Error Rate (LER) up to 84% over BP and 50% over GNN-based strategies, demonstrating enhanced performance over the existing state-of-the-art decoders.

---

## 74. Panther: A Cost-Effective Privacy-Preserving Framework for GNN Training and Inference Services in Cloud Environments

**论文链接:** [http://arxiv.org/abs/2511.01654v1](http://arxiv.org/abs/2511.01654v1)

**作者:** Congcong Chen, Xinyu Liu, Kaifeng Huang, Lifei Wei, Yang Shi

**发布时间:** 2025-11-03

**备注:** Accepted for publication in IEEE Transactions on Services Computing  (TSC)

### GPT解析

### 总结

本文提出了Panther，一个在云环境中用于图神经网络训练和推理的经济有效的隐私保护框架，能够在保护隐私的同时显著降低计算和通信成本。

### 背景

图神经网络在多个领域有重大影响，随着用户向云计算迁移，在云环境中保护GNN隐私成为关键问题。现有隐私保护技术虽然存在，但计算和通信开销较高，导致财务成本高，限制了其广泛采用。

### 目的

保护GNN隐私同时降低额外经济成本，开发一个适用于云环境的经济有效的隐私保护框架。

### 方法

Panther利用四方计算异步执行安全数组访问协议，并随机填充GNN节点的邻居信息来保护隐私。

### 主要发现

与最先进方法相比，Panther将训练和推理时间分别平均减少75.28%和82.80%，通信开销分别减少52.61%和50.26%，在Google Cloud平台上估计为GNN训练和推理分别节省55.05%和59.00%的财务成本。

### 结论

Panther是一个有效的隐私保护框架，能够在保护GNN隐私的同时显著降低计算和通信成本，有望促进隐私保护GNN技术在云环境中的广泛应用。

### 翻译

图神经网络(GNNs)在交通状态预测、社交推荐、知识感知问答等领域已产生重大影响。随着越来越多用户转向云计算，在云环境中释放GNN能力的同时保护隐私已成为关键问题。具体而言，GNN的训练数据和推理数据需要防止被外部对手窃取。同时，云计算的经济成本是用户的另一个主要关注点。因此，尽管现有研究已提出云环境中GNN的隐私保护技术，但其额外的计算和通信开销仍然相对较高，导致高额财务成本，限制了用户中的广泛采用。为了在保护GNN隐私的同时降低额外财务成本，我们引入了Panther，这是一个在云环境中用于GNN训练和推理服务的经济有效的隐私保护框架。技术上，Panther利用四方计算异步执行安全数组访问协议，并随机填充GNN节点的邻居信息。我们证明了Panther可以保护GNN模型训练和推理的隐私。我们的评估显示，与最先进的方法相比，Panther分别将训练和推理时间平均减少75.28%和82.80%，通信开销平均减少52.61%和50.26%，这估计在Google Cloud平台上为GNN训练和推理过程分别节省55.05%和59.00%的财务成本（基于按需定价模型）。


### 论文摘要

Graph Neural Networks (GNNs) have marked significant impact in traffic state prediction, social recommendation, knowledge-aware question answering and so on. As more and more users move towards cloud computing, it has become a critical issue to unleash the power of GNNs while protecting the privacy in cloud environments. Specifically, the training data and inference data for GNNs need to be protected from being stolen by external adversaries. Meanwhile, the financial cost of cloud computing is another primary concern for users. Therefore, although existing studies have proposed privacy-preserving techniques for GNNs in cloud environments, their additional computational and communication overhead remain relatively high, causing high financial costs that limit their widespread adoption among users.   To protect GNN privacy while lowering the additional financial costs, we introduce Panther, a cost-effective privacy-preserving framework for GNN training and inference services in cloud environments. Technically, Panther leverages four-party computation to asynchronously executing the secure array access protocol, and randomly pads the neighbor information of GNN nodes. We prove that Panther can protect privacy for both training and inference of GNN models. Our evaluation shows that Panther reduces the training and inference time by an average of 75.28% and 82.80%, respectively, and communication overhead by an average of 52.61% and 50.26% compared with the state-of-the-art, which is estimated to save an average of 55.05% and 59.00% in financial costs (based on on-demand pricing model) for the GNN training and inference process on Google Cloud Platform.

---

## 75. IVGAE-TAMA-BO: A novel temporal dynamic variational graph model for link prediction in global food trade networks with momentum structural memory and Bayesian optimization

**论文链接:** [http://arxiv.org/abs/2511.01639v1](http://arxiv.org/abs/2511.01639v1)

**作者:** Sicheng Wang, Shuhao Chen, Jingran Zhou, Chengyi Tu

**发布时间:** 2025-11-03

**备注:** 26pages,6figures

### GPT解析

### 总结

这项研究开发了一种创新的动态图神经网络模型IVGAE-TAMA-BO，用于预测全球粮食贸易网络中的未来链接。该模型通过捕捉时间演变和结构依赖关系，显著提高了预测准确性，结合贝叶斯优化的模型在各种贸易场景下表现优异。

### 背景

全球粮食贸易在确保粮食安全和维持供应链稳定方面发挥着关键作用。然而，粮食贸易网络结构在政治、经济和环境因素影响下动态演变，使得建模和预测未来的贸易链接具有挑战性。

### 目的

有效捕捉粮食贸易网络中的时间模式，提高链接预测的准确性和稳健性。

### 方法

提出了IVGAE-TAMA-BO，一种新型的动态图神经网络。基于原始IVGAE框架，引入了Trade-Aware Momentum Aggregator (TAMA)来捕捉贸易网络的时间演变，联合建模短期波动和长期结构依赖关系。使用基于动量的结构记忆机制提高预测的稳定性和性能，并使用贝叶斯优化自动调整关键超参数。

### 主要发现

这是首次将动态图神经网络应用于粮食贸易网络领域，显著提高了预测性能。在五种作物特定数据集上的实验表明，IVGAE-TAMA明显优于静态IVGAE和其他动态基线，通过有效建模时间依赖性，贝叶斯优化进一步提升了IVGAE-TAMA-BO的性能。

### 结论

提出的框架是全球贸易网络结构预测的稳健且可扩展的解决方案，在粮食安全监测和政策决策支持方面具有强大的应用潜力。

### 翻译

全球粮食贸易在确保粮食安全和维持供应链稳定方面发挥着关键作用。然而，其网络结构在政治、经济和环境因素的影响下动态演变，使得建模和预测未来的贸易链接具有挑战性。因此，有效捕捉粮食贸易网络中的时间模式对于提高链接预测的准确性和稳健性至关重要。本研究引入了IVGAE-TAMA-BO，一种专为模拟不断演变的贸易结构和预测全球粮食贸易网络中未来链接而设计的新型动态图神经网络。据我们所知，这是首次将动态图神经网络应用于该领域，显著提高了预测性能。基于原始IVGAE框架，所提出的模型纳入了一个贸易感知动量聚合器(TAMA)来捕捉贸易网络的时间演变，联合建模短期波动和长期结构依赖关系。基于动量的结构记忆机制进一步提高了预测的稳定性和性能。此外，使用贝叶斯优化来自动调整关键超参数，增强在不同贸易场景下的泛化能力。在五种作物特定数据集上的广泛实验表明，IVGAE-TAMA通过有效建模时间依赖性，明显优于静态IVGAE和其他动态基线，而贝叶斯优化进一步提升了IVGAE-TAMA-BO的性能。这些结果表明，所提出的框架是全球贸易网络结构预测的稳健且可扩展的解决方案，在粮食安全监测和政策决策支持方面具有强大的应用潜力。


### 论文摘要

Global food trade plays a crucial role in ensuring food security and maintaining supply chain stability. However, its network structure evolves dynamically under the influence of geopolitical, economic, and environmental factors, making it challenging to model and predict future trade links. Effectively capturing temporal patterns in food trade networks is therefore essential for improving the accuracy and robustness of link prediction. This study introduces IVGAE-TAMA-BO, a novel dynamic graph neural network designed to model evolving trade structures and predict future links in global food trade networks. To the best of our knowledge, this is the first work to apply dynamic graph neural networks to this domain, significantly enhancing predictive performance. Building upon the original IVGAE framework, the proposed model incorporates a Trade-Aware Momentum Aggregator (TAMA) to capture the temporal evolution of trade networks, jointly modeling short-term fluctuations and long-term structural dependencies. A momentum-based structural memory mechanism further improves predictive stability and performance. In addition, Bayesian optimization is used to automatically tune key hyperparameters, enhancing generalization across diverse trade scenarios. Extensive experiments on five crop-specific datasets demonstrate that IVGAE-TAMA substantially outperforms the static IVGAE and other dynamic baselines by effectively modeling temporal dependencies, while Bayesian optimization further boosts performance in IVGAE-TAMA-BO. These results highlight the proposed framework as a robust and scalable solution for structural prediction in global trade networks, with strong potential for applications in food security monitoring and policy decision support.

---

## 76. Gated Fusion Enhanced Multi-Scale Hierarchical Graph Convolutional Network for Stock Movement Prediction

**论文链接:** [http://arxiv.org/abs/2511.01570v1](http://arxiv.org/abs/2511.01570v1)

**作者:** Xiaosha Xue, Peibo Duan, Zhipeng Liu, Qi Chu, Changsheng Zhang, Bin zhang

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种名为MS-HGFN(多尺度分层图融合网络)的新型模型，用于解决股票市场预测中的挑战，通过分层GNN模块和自上而下的门控方法，有效捕捉股票间的复杂关系和多尺度特征，实验表明该模型在预测准确性和稳定性方面表现优异。

### 背景

准确预测股票市场走势由于股票固有的波动性和股票间复杂的相互依赖关系仍然是一个巨大的挑战。

### 目的

克服现有多尺度图神经网络在股票预测中忽视的两个关键点：每个股票内部的细微属性模式以及多尺度采样中对粗粒度和细粒度特征的偏见注意力。

### 方法

提出MS-HGFN模型，包含一个分层GNN模块，通过在不同时间尺度上学习内部属性模式和外部属性特征形成动态图，并采用自上而下的门控方法促进多尺度时空特征的融合。

### 主要发现

使用美国和中国股票市场的真实数据集进行实验，MS-HGFN优于传统和先进模型，预测准确性提高了高达1.4%，在回报模拟中增强了稳定性。

### 结论

MS-HGFN通过有效捕捉股票间的复杂关系和多尺度特征，显著提升了股票市场预测的准确性和稳定性。

### 翻译

准确预测股票市场走势由于股票固有的波动性和股票间复杂的相互依赖关系仍然是一个巨大的挑战。虽然多尺度图神经网络在建模这些关系方面具有潜力，但它们经常忽视两个关键点：每个股票内部影响股票间相关性的细微的内部属性模式，以及在多尺度采样过程中对粗粒度和细粒度特征的偏见注意力。为了克服这些挑战，我们引入了MS-HGFN(多尺度分层图融合网络)。该模型具有一个分层GNN模块，通过在不同时间尺度上学习内部属性模式和外部属性特征来形成动态图，从而全面捕捉时空依赖关系。此外，自上而下的门控方法促进多尺度时空特征的整合，保留了关键的粗粒度和细粒度特征而不过多干扰。利用美国和中国股票市场的真实数据集进行的实验表明，MS-HGFN优于传统和先进模型，预测准确性提高了高达1.4%，并在回报模拟中增强了稳定性。代码可在https://anonymous.4open.science/r/MS-HGFN获取。


### 论文摘要

Accurately predicting stock market movements remains a formidable challenge due to the inherent volatility and complex interdependencies among stocks. Although multi-scale Graph Neural Networks (GNNs) hold potential for modeling these relationships, they frequently neglect two key points: the subtle intra-attribute patterns within each stock affecting inter-stock correlation, and the biased attention to coarse- and fine-grained features during multi-scale sampling. To overcome these challenges, we introduce MS-HGFN (Multi-Scale Hierarchical Graph Fusion Network). The model features a hierarchical GNN module that forms dynamic graphs by learning patterns from intra-attributes and features from inter-attributes over different time scales, thus comprehensively capturing spatio-temporal dependencies. Additionally, a top-down gating approach facilitates the integration of multi-scale spatio-temporal features, preserving critical coarse- and fine-grained features without too much interference. Experiments utilizing real-world datasets from U.S. and Chinese stock markets demonstrate that MS-HGFN outperforms both traditional and advanced models, yielding up to a 1.4% improvement in prediction accuracy and enhanced stability in return simulations. The code is available at https://anonymous.4open.science/r/MS-HGFN.

---

## 77. Efficient Curvature-aware Graph Network

**论文链接:** [http://arxiv.org/abs/2511.01443v1](http://arxiv.org/abs/2511.01443v1)

**作者:** Chaoqun Fei, Tinglve Zhou, Tianyong Hao, Yangyang Li

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究提出了一种名为有效电阻曲率的新型图曲率度量方法，解决了现有Ollivier-Ricci曲率计算复杂度高的问题，在保持相当几何表达能力的同时显著提高了计算效率。

### 背景

图曲率能为图神经网络提供几何先验，增强其建模复杂图结构的能力，特别是在结构感知、鲁棒性和理论可解释性方面。现有Ollivier-Ricci曲率虽具有强几何可解释性，但计算复杂度极高，限制了其在大型图数据集上的应用。

### 目的

开发一种计算效率更高且保持相当几何表达能力的图曲率度量方法，以替代计算复杂度高的Ollivier-Ricci曲率。

### 方法

提出有效电阻曲率，使用节点对之间的有效电阻而非最优传输距离来量化沿图边的消息传递难易度。

### 主要发现

有效电阻曲率显著优于Ollivier-Ricci曲率在计算效率方面，同时保持了相当的几何表达能力；理论证明了其低计算复杂度和对Ollivier-Ricci曲率的可替代性；实验表明其在多种GNN任务上实现了竞争性性能，同时大幅降低计算开销。

### 结论

有效电阻曲率为图神经网络提供了一种高效且具有竞争力的几何先验方法，解决了Ollivier-Ricci曲率在大规模图数据集上的应用限制。

### 翻译

图曲率为图神经网络(GNNs)提供几何先验，增强其建模复杂图结构的能力，特别是在结构感知、鲁棒性和理论可解释性方面。在现有方法中，Ollivier-Ricci曲率因其强几何可解释性而被广泛研究，能有效表征节点间的局部几何分布。然而，其极高的计算复杂度限制了其在大型图数据集上的适用性。为应对这一挑战，我们提出了一种新的图曲率度量方法——有效电阻曲率，它使用节点对之间的有效电阻而非最优传输距离来量化沿图边的消息传递难易度。该方法在计算效率上显著优于Ollivier-Ricci曲率，同时保持了相当的几何表达能力。理论上，我们证明了有效电阻曲率的低计算复杂度，并建立了其对Ollivier-Ricci曲率的可替代性。此外，在多种GNN任务上的广泛实验表明，我们的方法在大幅降低计算开销的同时，实现了与Ollivier-Ricci曲率相当的性能。


### 论文摘要

Graph curvature provides geometric priors for Graph Neural Networks (GNNs), enhancing their ability to model complex graph structures, particularly in terms of structural awareness, robustness, and theoretical interpretability. Among existing methods, Ollivier-Ricci curvature has been extensively studied due to its strong geometric interpretability, effectively characterizing the local geometric distribution between nodes. However, its prohibitively high computational complexity limits its applicability to large-scale graph datasets. To address this challenge, we propose a novel graph curvature measure--Effective Resistance Curvature--which quantifies the ease of message passing along graph edges using the effective resistance between node pairs, instead of the optimal transport distance. This method significantly outperforms Ollivier-Ricci curvature in computational efficiency while preserving comparable geometric expressiveness. Theoretically, we prove the low computational complexity of effective resistance curvature and establish its substitutability for Ollivier-Ricci curvature. Furthermore, extensive experiments on diverse GNN tasks demonstrate that our method achieves competitive performance with Ollivier-Ricci curvature while drastically reducing computational overhead.

---

## 78. 论文ID: 2511.01408v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.01408v1.json'

---

## 79. Diffusion-Based Solver for CNF Placement on the Cloud-Continuum

**论文链接:** [http://arxiv.org/abs/2511.01343v1](http://arxiv.org/abs/2511.01343v1)

**作者:** Álvaro Vázquez Rodríguez, Manuel Fernández-Veiga, Carlos Giraldo-Rodríguez

**发布时间:** 2025-11-03

**备注:** 7 pages, 7 figures. Presented at PE-WASUN'25 (IEEE International  Symposium on Performance Evaluation of Wireless Ad Hoc, Sensor, and  Ubiquitous Networks)

### GPT解析

### 总结

本文提出了一种基于去噪扩散概率模型(DDPM)的新理论框架，用于解决云原生网络功能(CNFs)在云连续体中的部署问题，实现了比传统方法更快且更有效的解决方案。

### 背景

云原生网络功能(CNFs)在云连续体(Cloud-Continuum)中的部署是当前5G和未来6G网络编排的核心挑战。这个过程涉及将互依赖的计算任务(结构化为服务功能链)放置在分布式云基础设施上，同时满足严格的资源、带宽和延迟约束。

### 目的

解决传统方法(包括混合整数非线性规划、启发式和强化学习)在可扩展性、约束处理和泛化能力方面的局限性。

### 方法

提出基于去噪扩散概率模型(DDPM)的新理论框架，将部署问题重新概念化为生成图到分配任务，将部署问题编码为异构图，训练图神经网络去噪器迭代细化有噪的CNF到云分配矩阵，并将特定约束的损失直接整合到损失函数中。

### 主要发现

在各种拓扑结构上进行了广泛评估，证实该模型始终产生可行解决方案，推理速度比MINLP求解器快几个数量级。

### 结论

基于扩散的生成模型在受约束的网络嵌入问题中显示出潜力，对分布式云原生网络功能的实际、可扩展编排产生影响。

### 翻译

将云原生网络功能(CNFs)跨云连续体(Cloud-Continuum)的部署代表当前5G和未来6G网络编排中的一个核心挑战。该过程涉及将互依赖的计算任务(结构化为服务功能链)放置在分布式云基础设施上，同时满足严格的资源、带宽和延迟约束。公认的是，包括混合整数非线性规划、启发式和强化学习在内的传统方法在可扩展性、约束处理和泛化能力方面存在局限性。在本研究中，提出了一种基于去噪扩散概率模型(DDPM)的新理论框架用于CNF部署。当前方法将重新概念化部署为生成图到分配任务，其中部署问题被编码为异构图，并且训练图神经网络去噪器来迭代细化有噪的CNF到云分配矩阵。该模型将特定约束的损失直接整合到损失函数中，从而使其能够学习可行的解空间。通过严谨和系统的方法实现了DDPM公式与结构组合约束的集成。已在各种拓扑结构上进行了广泛评估，证实该模型始终产生可行解决方案，推理速度比MINLP求解器快几个数量级。获得的结果证明了基于扩散的生成模型在受约束的网络嵌入问题中的潜力，对分布式云原生网络功能的实际、可扩展编排产生影响。


### 论文摘要

The placement of Cloud-Native Network Functions (CNFs) across the Cloud-Continuum represents a core challenge in the orchestration of current 5G and future 6G networks. The process involves the placement of interdependent computing tasks, structured as Service Function Chains, over distributed cloud infrastructures. This is achieved while satisfying strict resource, bandwidth and latency constraints. It is acknowledged that classical approaches, including mixed-integer nonlinear programming, heuristics and reinforcement learning are limited in terms of scalability, constraint handling and generalisation capacity. In the present study, a novel theoretical framework is proposed, which is based on Denoising Diffusion Probabilistic Models (DDPM) for CNF placement. The present approach proposes a reconceptualisation of placement as a generative graph to assignment task, where the placement problem is encoded as a heterogeneous graph, and a Graph Neural Network denoiser is trained to iteratively refine noisy CNF-to-cloud assignment matrices. The model incorporates constraint-specific losses directly into the loss function, thereby allowing it to learn feasible solution spaces. The integration of the DDPM formulation with structured combinatorial constraints is achieved through a rigorous and systematic approach. Extensive evaluations across diverse topologies have been conducted, which have confirmed that the model consistently produces feasible solutions with orders of magnitude faster inference than MINLP solvers. The results obtained demonstrate the potential of diffusion-based generative modelling for constrained network embedding problems, making an impact towards the practical, scalable orchestration of distributed Cloud-Native Network Functions.

---

## 80. Graph Neural Network-Based Semi-Supervised Open-Set Fault Diagnosis for Marine Machinery Systems

**论文链接:** [http://arxiv.org/abs/2511.01258v1](http://arxiv.org/abs/2511.01258v1)

**作者:** Chuyue Lou, M. Amine Atoui

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种半监督开集故障诊断(SOFD)框架，用于解决海洋机械系统中未知故障类型的检测问题，增强了深度学习模型在实际工业环境中的适用性。

### 背景

基于深度学习的海洋机械系统故障诊断方法在航运业受到关注，但现有研究假设训练和测试数据集的故障类别一致且已知，在受控环境下表现良好。然而实际应用中可能出现训练期间未见的故障类型，导致现有方法失效。

### 目的

解决未知故障类型导致现有故障诊断方法失效的问题，增强和扩展深度学习模型在开集故障诊断场景中的适用性。

### 方法

提出半监督开集故障诊断(SOFD)框架，包含可靠性子集构建过程，使用监督特征学习模型提取的多层融合特征表示选择未标记测试子集，将标记训练集和伪标记测试子集输入半监督诊断模型，学习判别性特征以实现已知故障分类和未知样本检测。

### 主要发现

在公共海事基准数据集上的实验结果证明所提出的SOFD框架具有有效性和优越性。

### 结论

SOFD框架能够有效处理海洋机械系统中的开集故障诊断问题，在工业应用中具有潜力，可应对实际环境中出现的未知故障类型。

### 翻译

最近，基于深度学习模型的海洋机械系统故障诊断方法在航运业引起了广泛关注。大多数现有研究假设训练和测试数据集中的故障类别是一致且已知的，这些方法在受控环境下表现良好。然而，在实践中，可能会出现先前未见或未知的故障类型（即训练期间不存在的分布外或开集观测），导致这些方法失效，并对它们在工业中的广泛部署构成重大挑战。为应对这一挑战，本文提出了一种半监督开集故障诊断(SOFD)框架，增强了深度学习模型在开集故障诊断场景中的适用性。该框架包括一个可靠性子集构建过程，使用监督特征学习模型提取的多层融合特征表示来选择未标记的测试子集。然后，将标记的训练集和伪标记的测试子集输入半监督诊断模型，学习每个类的判别性特征，实现对已知故障的准确分类和未知样本的有效检测。在公共海事基准数据集上的实验结果证明了所提出的SOFD框架的有效性和优越性。


### 论文摘要

Recently, fault diagnosis methods for marine machinery systems based on deep learning models have attracted considerable attention in the shipping industry. Most existing studies assume fault classes are consistent and known between the training and test datasets, and these methods perform well under controlled environment. In practice, however, previously unseen or unknown fault types (i.e., out-of-distribution or open-set observations not present during training) can occur, causing such methods to fail and posing a significant challenge to their widespread industrial deployment. To address this challenge, this paper proposes a semi-supervised open-set fault diagnosis (SOFD) framework that enhances and extends the applicability of deep learning models in open-set fault diagnosis scenarios. The framework includes a reliability subset construction process, which uses a multi-layer fusion feature representation extracted by a supervised feature learning model to select an unlabeled test subset. The labeled training set and pseudo-labeled test subset are then fed into a semi-supervised diagnosis model to learn discriminative features for each class, enabling accurate classification of known faults and effective detection of unknown samples. Experimental results on a public maritime benchmark dataset demonstrate the effectiveness and superiority of the proposed SOFD framework.

---

## 81. WindMiL: Equivariant Graph Learning for Wind Loading Prediction

**论文链接:** [http://arxiv.org/abs/2511.01226v1](http://arxiv.org/abs/2511.01226v1)

**作者:** Themistoklis Vargiemezis, Charilaos Kanatsoulis, Catherine Gorlé

**发布时间:** 2025-11-03

### GPT解析

### 总结

WindMiL是一种新的机器学习框架，通过结合系统数据集生成与对称感知的图神经网络，实现了建筑物风荷载的高效、可扩展和准确预测，解决了传统方法成本高的问题。

### 背景

准确预测建筑物风荷载对结构安全和可持续设计至关重要，但传统方法如风洞测试和大涡模拟(LES)成本过高，每个LES案例需要至少24小时计算时间，使得全面参数研究不可行。

### 目的

开发一种能够高效、准确预测建筑物风荷载的方法，降低计算成本，使大规模参数研究成为可能。

### 方法

构建大规模风荷载数据集，通过符号距离函数插值处理屋面几何形状，模拟462个不同形状和风向的案例；开发反射等变性图神经网络，确保在镜像几何形状下的物理一致性预测。

### 主要发现

WindMiL在插值和外推评估中取得高精度，表面压力系数的平均值和标准差的误差小于等于0.02；在反射测试评估中保持96%以上的命中率，比非等变性基线模型提高10%以上。

### 结论

WindMiL通过将系统数据集与等变性代理模型配对，实现了建筑物风荷载的高效、可扩展和准确预测，为结构安全和可持续设计提供了实用工具。

### 翻译

准确预测建筑物上的风荷载对于结构安全和可持续设计至关重要，然而传统方法如风洞测试和大涡模拟(LES)对于大规模探索来说成本过高。每个LES案例通常需要至少24小时的计算时间，这使得全面的参数研究不可行。我们提出了WindMiL，这是一种新的机器学习框架，结合了系统数据集生成与对称感知的图神经网络(GNNs)。首先，我们通过对屋面几何形状应用符号距离函数插值，并在不同形状和风向条件下模拟462个案例，构建了一个关于低层建筑物风荷载的大规模数据集。其次，我们开发了一种反射等变性GNN，确保在镜像几何形状下物理预测的一致性。在插值和外推评估中，WindMiL在表面压力系数的平均值和标准差方面都取得了高精度，并且在反射测试评估中保持准确，命中率保持在96%以上，而非等变性基线模型的命中率下降了10%以上。通过将系统数据集与等变性代理模型配对，WindMiL能够实现建筑物风荷载的高效、可扩展和准确的预测。


### 论文摘要

Accurate prediction of wind loading on buildings is crucial for structural safety and sustainable design, yet conventional approaches such as wind tunnel testing and large-eddy simulation (LES) are prohibitively expensive for large-scale exploration. Each LES case typically requires at least 24 hours of computation, making comprehensive parametric studies infeasible. We introduce WindMiL, a new machine learning framework that combines systematic dataset generation with symmetry-aware graph neural networks (GNNs). First, we introduce a large-scale dataset of wind loads on low-rise buildings by applying signed distance function interpolation to roof geometries and simulating 462 cases with LES across varying shapes and wind directions. Second, we develop a reflection-equivariant GNN that guarantees physically consistent predictions under mirrored geometries. Across interpolation and extrapolation evaluations, WindMiL achieves high accuracy for both the mean and the standard deviation of surface pressure coefficients (e.g., RMSE $\leq 0.02$ for mean $C_p$) and remains accurate under reflected-test evaluation, maintaining hit rates above $96\%$ where the non-equivariant baseline model drops by more than $10\%$. By pairing a systematic dataset with an equivariant surrogate, WindMiL enables efficient, scalable, and accurate predictions of wind loads on buildings.

---

## 82. An Interdisciplinary and Cross-Task Review on Missing Data Imputation

**论文链接:** [http://arxiv.org/abs/2511.01196v1](http://arxiv.org/abs/2511.01196v1)

**作者:** Jicong Fan

**发布时间:** 2025-11-03

### GPT解析

### 总结

这是一篇关于缺失数据插补方法的综合性综述，连接了统计基础与现代机器学习进展，涵盖了从传统到现代的各种插补方法，特别关注复杂数据类型和与下游任务的集成，并提出了未来研究方向。

### 背景

缺失数据是数据科学中的基本挑战，在医疗保健、生物信息学、社会科学、电子商务和工业监控等多个领域显著阻碍了分析和决策。尽管有数十年的研究和多种插补方法，但文献仍然分散在不同领域，缺乏将统计基础与现代机器学习进展联系起来的全面综合。

### 目的

系统性地回顾缺失数据的核心概念，包括缺失机制、单次与多次插补以及不同的插补目标；检查各领域的问题特征；提供插补方法的全面分类；研究插补与下游任务的集成；评估理论保证、基准资源和评估指标；确定关键挑战和未来方向。

### 方法

从经典技术（如回归、EM算法）到现代方法（如低秩和高秩矩阵补全、深度学习模型、大型语言模型）进行全面分类；特别关注复杂数据类型（张量、时间序列、流数据、图结构数据、分类数据和多模态数据）的插补方法；研究插补与下游任务（分类、聚类、异常检测）的集成方式，包括顺序管道和联合优化框架；评估理论保证、基准资源和评估指标。

### 主要发现

提供了从传统到现代的插补方法全面分类；特别关注了复杂数据类型的插补方法；探讨了插补与下游任务的集成方式；强调了模型选择和超参数优化的重要性；指出了隐私保护插补的日益重要性；提出了追求可泛化模型的方向。

### 结论

确定了关键挑战和未来方向，包括模型选择和超参数优化的重要性；通过联邦学习进行隐私保护插补的日益重要性；追求能够跨领域和数据类型适应的可泛化模型；为未来研究勾勒出路线图。

### 翻译

缺失数据是数据科学中的一个基本挑战，在医疗保健、生物信息学、社会科学、电子商务和工业监控等广泛领域显著阻碍了分析和决策。尽管有数十年的研究和众多的插补方法，但文献仍然分散在不同领域，迫切需要将统计基础与现代机器学习进展联系起来的全面综合。本工作系统性地回顾了核心概念，包括缺失机制、单次与多次插补以及不同的插补目标，并检查了各领域的问题特征。它提供了插补方法的全面分类，涵盖从经典技术（如回归、EM算法）到现代方法，如低秩和高秩矩阵补全、深度学习模型（自编码器、GAN、扩散模型、图神经网络）和大型语言模型。特别关注了复杂数据类型的方法，如张量、时间序列、流数据、图结构数据、分类数据和多模态数据。除了方法论，我们还研究了插补与下游任务（如分类、聚类和异常检测）的关键集成，检查了顺序管道和联合优化框架。该综述还评估了理论保证、基准资源和评估指标。最后，我们确定了关键挑战和未来方向，强调模型选择和超参数优化的重要性、通过联邦学习进行隐私保护插补的日益重要性，以及追求能够跨领域和数据类型适应的可泛化模型，从而为未来研究勾勒出路线图。


### 论文摘要

Missing data is a fundamental challenge in data science, significantly hindering analysis and decision-making across a wide range of disciplines, including healthcare, bioinformatics, social science, e-commerce, and industrial monitoring. Despite decades of research and numerous imputation methods, the literature remains fragmented across fields, creating a critical need for a comprehensive synthesis that connects statistical foundations with modern machine learning advances. This work systematically reviews core concepts-including missingness mechanisms, single versus multiple imputation, and different imputation goals-and examines problem characteristics across various domains. It provides a thorough categorization of imputation methods, spanning classical techniques (e.g., regression, the EM algorithm) to modern approaches like low-rank and high-rank matrix completion, deep learning models (autoencoders, GANs, diffusion models, graph neural networks), and large language models. Special attention is given to methods for complex data types, such as tensors, time series, streaming data, graph-structured data, categorical data, and multimodal data. Beyond methodology, we investigate the crucial integration of imputation with downstream tasks like classification, clustering, and anomaly detection, examining both sequential pipelines and joint optimization frameworks. The review also assesses theoretical guarantees, benchmarking resources, and evaluation metrics. Finally, we identify critical challenges and future directions, emphasizing model selection and hyperparameter optimization, the growing importance of privacy-preserving imputation via federated learning, and the pursuit of generalizable models that can adapt across domains and data types, thereby outlining a roadmap for future research.

---

## 83. GraphGeo: Multi-Agent Debate Framework for Visual Geo-localization with Heterogeneous Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.00908v1](http://arxiv.org/abs/2511.00908v1)

**作者:** Heng Zheng, Yuling Shi, Xiaodong Gu, Haochen You, Zijian Zhang, Lubin Gan, Hao Zhang, Wenjun Huang, Jin Huang

**发布时间:** 2025-11-02

### GPT解析

### 总结

本文提出GraphGeo，一种基于异构图神经网络的多代理辩论框架，用于视觉地理定位任务，通过结构化辩论将代理间的认知冲突转化为增强的定位精度。

### 背景

视觉地理定位需要广泛地理知识和复杂推理来确定无GPS元数据的图像位置；传统检索方法受限于数据库覆盖和质量；大型视觉-语言模型虽能直接从图像内容推理位置，但单个模型难以处理多样化地理区域和复杂场景；现有多代理系统统一处理所有代理交互，缺乏有效处理冲突预测的机制。

### 目的

提出GraphGeo框架，利用异构图神经网络和结构化辩论机制提高视觉地理定位的准确性和鲁棒性。

### 方法

使用类型化边建模多样化辩论关系，区分支持性协作、竞争性论证和知识转移；引入双重辩论机制，结合节点级细化和边级论证建模；采用跨层拓扑细化策略，实现图结构与代理表示的共同演化。

### 主要发现

在多个基准测试中，GraphGeo显著优于最先进的方法，有效将代理间的认知冲突转化为增强的地理定位精度。

### 结论

通过结构化辩论机制，GraphGeo能够有效处理代理间的认知冲突，显著提升视觉地理定位的性能，为多代理系统在复杂推理任务中的应用提供了新思路。

### 翻译

视觉地理定位需要广泛的地理知识和复杂的推理来确定没有GPS元数据的图像位置。传统检索方法受限于数据库覆盖范围和质量。最近的大型视觉-语言模型(LVLMs)能够直接从图像内容进行位置推理，但单个模型难以处理多样化的地理区域和复杂场景。现有的多代理系统通过模型协作提高性能，但统一处理所有代理交互，缺乏有效处理冲突预测的机制。我们提出GraphGeo，一个使用异构图神经网络进行视觉地理定位的多代理辩论框架。我们的方法通过类型化边建模多样化的辩论关系，区分支持性协作、竞争性论证和知识转移。我们引入了双重辩论机制，结合节点级细化和边级论证建模。跨层拓扑细化策略实现了图结构与代理表示的共同演化。在多个基准测试中的实验表明，GraphGeo显著优于最先进的方法。我们的框架通过结构化辩论将代理之间的认知冲突转化为增强的地理定位精度。


### 论文摘要

Visual geo-localization requires extensive geographic knowledge and sophisticated reasoning to determine image locations without GPS metadata. Traditional retrieval methods are constrained by database coverage and quality. Recent Large Vision-Language Models (LVLMs) enable direct location reasoning from image content, yet individual models struggle with diverse geographic regions and complex scenes. Existing multi-agent systems improve performance through model collaboration but treat all agent interactions uniformly. They lack mechanisms to handle conflicting predictions effectively. We propose \textbf{GraphGeo}, a multi-agent debate framework using heterogeneous graph neural networks for visual geo-localization. Our approach models diverse debate relationships through typed edges, distinguishing supportive collaboration, competitive argumentation, and knowledge transfer. We introduce a dual-level debate mechanism combining node-level refinement and edge-level argumentation modeling. A cross-level topology refinement strategy enables co-evolution between graph structure and agent representations. Experiments on multiple benchmarks demonstrate GraphGeo significantly outperforms state-of-the-art methods. Our framework transforms cognitive conflicts between agents into enhanced geo-localization accuracy through structured debate.

---

## 84. IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval

**论文链接:** [http://arxiv.org/abs/2511.00268v1](http://arxiv.org/abs/2511.00268v1)

**作者:** Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi

**发布时间:** 2025-10-31

**备注:** Accepted at EMNLP 2025 (Main)

### GPT解析

### 总结

论文提出了IL-PCR语料库，用于法规检索和先例检索两项任务的共同开发，并利用LLM重新排序方法取得了最佳性能。

### 背景

法律从业者经常需要识别和检索相关法规和先例案例，但研究人员至今独立处理这两项任务，忽略了它们之间的内在联系。

### 目的

解决法规检索和先例检索任务之间的研究缺口，开发可以利用两项任务依赖关系的模型。

### 方法

提出IL-PCR语料库作为共同测试平台，使用词汇模型、语义模型和基于GNN的集成模型进行实验，并开发基于LLM的重新排序方法。

### 主要发现

基于LLM的重新排序方法能够有效利用法规检索和先例检索任务之间的依赖关系，取得了最佳性能。

### 结论

通过IL-PCR语料库和基于LLM的重新排序方法，可以更好地同时处理法规检索和先例检索任务，提高法律检索的效率和准确性。

### 翻译

识别/检索给定法律情况下的相关法规和先例/判例是法律从业者常见的任务。迄今为止，研究人员独立处理这两项任务，为每个任务开发了完全不同的数据集和模型；然而，这两项检索任务本质上是相关的，例如，相似的案例往往会引用相似的法规（由于相似的事实情况）。在本文中，我们解决了这一研究缺口。我们提出了IL-PCR（印度法律先例和法规检索语料库），这是一个独特的语料库，为开发法规检索和先例检索两项任务的模型提供了共同测试平台，可以利用两项任务之间的依赖关系。我们使用多种基线模型对这两项任务进行了广泛实验，包括词汇模型、语义模型和基于GNN的集成模型。此外，为了利用两项任务之间的依赖关系，我们开发了一种基于LLM的重新排序方法，取得了最佳性能。


### 论文摘要

Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance.

---

## 85. MeixnerNet: Adaptive and Robust Spectral Graph Neural Networks with Discrete Orthogonal Polynomials

**论文链接:** [http://arxiv.org/abs/2511.00113v1](http://arxiv.org/abs/2511.00113v1)

**作者:** Huseyin Goksu

**发布时间:** 2025-10-30

### GPT解析

### 总结

MeixnerNet是一种新的谱图神经网络架构，使用离散正交多项式而非连续正交多项式，解决了连续滤波器应用于离散图结构的理论不匹配问题。

### 背景

Spectral GNNs通过在谱域定义图卷积实现了最先进的结果，而ChebyNet使用基于连续正交多项式的滤波器是一种常见方法。

### 目的

解决连续域滤波器应用于离散图结构造成的不匹配问题，提高模型性能并增强对超参数设置的鲁棒性。

### 方法

引入MeixnerNet，使用离散Meixner多项式作为滤波器基础，使多项式参数可学习，并通过结合Laplacian缩放和LayerNorm的技术解决数值不稳定性问题。

### 主要发现

在最佳设置下，MeixnerNet在三个基准测试中胜过ChebyNet两个；更重要的是，MeixnerNet对多项式次数的变化异常稳健，而ChebyNet对此超参数非常脆弱。

### 结论

使用离散正交多项式而非连续正交多项式可以解决理论不匹配问题，提高模型性能和对超参数的鲁棒性。

### 翻译

谱图神经网络通过在谱域定义图卷积取得了最先进的结果。ChebyNet推广的一种常见方法是使用基于连续正交多项式（如Chebyshev）的多项式滤波器。这造成了理论上的脱节，因为这些连续域滤波器被应用于本质上离散的图结构。我们假设这种不匹配可能导致次优性能和对超参数设置的脆弱性。在本文中，我们介绍了MeixnerNet，一种新的谱GNN架构，它采用离散正交多项式——特别是Meixner多项式。我们的模型使多项式的两个关键形状参数可学习，允许滤波器根据给定图的特定谱属性调整其多项式基。我们通过引入一种结合Laplacian缩放和每个基的LayerNorm的新稳定技术，克服了这些多项式显著的数值不稳定性。实验证明，在最佳设置下，MeixnerNet与强大的ChebyNet基线相比具有竞争性到优越的性能（在3个基准测试中赢得2个）。更重要的是，我们表明MeixnerNet对多项式次数的变化异常稳健，而ChebyNet对此超参数证明是非常脆弱的，在性能崩溃的地方MeixnerNet保持稳定。


### 论文摘要

Spectral Graph Neural Networks (GNNs) have achieved state-of-the-art results by defining graph convolutions in the spectral domain. A common approach, popularized by ChebyNet, is to use polynomial filters based on continuous orthogonal polynomials (e.g., Chebyshev). This creates a theoretical disconnect, as these continuous-domain filters are applied to inherently discrete graph structures. We hypothesize this mismatch can lead to suboptimal performance and fragility to hyperparameter settings.   In this paper, we introduce MeixnerNet, a novel spectral GNN architecture that employs discrete orthogonal polynomials -- specifically, the Meixner polynomials $M_k(x; \beta, c)$. Our model makes the two key shape parameters of the polynomial, beta and c, learnable, allowing the filter to adapt its polynomial basis to the specific spectral properties of a given graph. We overcome the significant numerical instability of these polynomials by introducing a novel stabilization technique that combines Laplacian scaling with per-basis LayerNorm.   We demonstrate experimentally that MeixnerNet achieves competitive-to-superior performance against the strong ChebyNet baseline at the optimal K = 2 setting (winning on 2 out of 3 benchmarks). More critically, we show that MeixnerNet is exceptionally robust to variations in the polynomial degree K, a hyperparameter to which ChebyNet proves to be highly fragile, collapsing in performance where MeixnerNet remains stable.

---

## 86. Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything

**论文链接:** [http://arxiv.org/abs/2511.02834v1](http://arxiv.org/abs/2511.02834v1)

**作者:** Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh

**发布时间:** 2025-11-04

**备注:** 16 pages, 7 figures, 14 tables. Under Review

### GPT解析

### 总结

Agent-Omni框架通过主代理系统协调现有基础模型，实现了无需重新训练的灵活多模态推理，在多种模态和全能任务上取得了最先进性能，特别是复杂跨模态推理任务，且具有模块化、可扩展和透明的特点。

### 背景

多模态大语言模型（MLLMs）已展现出强大的能力，但仍局限于固定的模态对，并且需要使用大型对齐数据集进行昂贵的微调。构建能够完全整合文本、图像、音频和视频的全能模型仍然不切实际，且缺乏强大的推理支持。

### 目的

提出一个Agent-Omni框架，通过主代理系统协调现有基础模型，实现灵活的多模态推理，无需重新训练。

### 方法

主代理解释用户意图，将子任务委托给特定模态的代理，并将它们的输出整合成连贯的响应。

### 主要发现

在文本、图像、音频、视频和全能基准上的广泛实验表明，Agent-Omni始终取得了最先进的性能，特别是在需要复杂跨模态推理的任务上。其基于代理的设计实现了专业基础模型的无缝集成，确保了对多样化输入的适应性，同时保持透明性和可解释性。

### 结论

该框架是模块化的且易于扩展，允许随着更强大模型的可用而进行未来改进。作者发布了开源实现，以支持对可扩展和可靠的全模态推理的持续研究。

### 翻译

多模态大语言模型（MLLMs）已展现出强大的能力，但仍局限于固定的模态对，并且需要使用大型对齐数据集进行昂贵的微调。构建能够完全整合文本、图像、音频和视频的全能模型仍然不切实际，且缺乏强大的推理支持。在本文中，我们提出了一个Agent-Omni框架，通过主代理系统协调现有基础模型，实现了无需重新训练的灵活多模态推理。主代理解释用户意图，将子任务委托给特定模态的代理，并将它们的输出整合成连贯的响应。在文本、图像、音频、视频和全能基准上的广泛实验表明，Agent-Omni始终取得了最先进的性能，特别是在需要复杂跨模态推理的任务上。其基于代理的设计实现了专业基础模型的无缝集成，确保了对多样化输入的适应性，同时保持透明性和可解释性。此外，该框架是模块化的且易于扩展，允许随着更强大模型的可用而进行未来改进。我们发布了开源实现，以支持对可扩展和可靠的全模态推理的持续研究。


### 论文摘要

Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning.

---

## 87. GeoCrossBench: Cross-Band Generalization for Remote Sensing

**论文链接:** [http://arxiv.org/abs/2511.02831v1](http://arxiv.org/abs/2511.02831v1)

**作者:** Hakob Tamazyan, Ani Vanyan, Alvard Barseghyan, Anna Khosrovyan, Evan Shelhamer, Hrant Khachatrian

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出了GeoCrossBench基准测试，用于评估遥感基础模型对新卫星的泛化能力，并开发了ChiViT模型来改善跨卫星性能。研究发现现有模型在分布内表现不佳，对无波段重叠卫星的泛化能力显著下降，但对额外波段有一定适应能力。仅微调最后一层线性层即可获得一致性能，表明该基准远未饱和。

### 背景

遥感卫星的数量和多样性随时间增长，而大多数标记数据来自较老的卫星。随着地球观测基础模型的扩展，支持新卫星的训练成本增加，模型对新卫星的泛化能力变得尤为重要。

### 目的

介绍GeoCrossBench基准测试，扩展流行的GeoBench基准，并开发新的评估协议来测试模型对无波段重叠卫星和具有额外波段卫星的泛化能力。同时开发ChiViT模型以改善跨卫星性能。

### 方法

创建GeoCrossBench基准测试，开发ChannelViT的自监督扩展ChiViT，并进行多项实验：评估分布内性能、评估对无波段重叠卫星的泛化能力、评估对具有额外波段卫星的泛化能力，以及测试仅微调最后一层线性层的性能。

### 主要发现

1) 即使最好的遥感基础模型在分布内设置下也不如通用模型如DINOv3；2) 当泛化到无波段重叠的新卫星时，所有模型性能下降2-4倍，ChiViT显著优于第二名DINOv3；3) 当测试时提供额外波段，所有测试模型性能平均下降5-25%；4) 仅使用所有波段标签微调最后一层线性层，可在所有卫星上获得相对一致的性能。

### 结论

该基准测试远未达到饱和。作者公开发布了代码和数据集，以鼓励开发具有更强跨卫星泛化能力的更面向未来的遥感模型。

### 翻译

随着遥感卫星的数量和多样性随时间增长，而绝大多数标记数据来自较老的卫星。随着地球观测基础模型的扩展，支持新卫星的(重新)训练成本也在增加，因此模型对新卫星的泛化能力变得越来越重要。在这项工作中，我们介绍了GeoCrossBench，这是流行的GeoBench基准的扩展，具有新的评估协议：它测试分布内性能；对无波段重叠卫星的泛化能力；以及对具有训练集之外额外波段卫星的泛化能力。我们还开发了ChannelViT的自监督扩展ChiViT，以改善其跨卫星性能。首先，我们表明即使最好的遥感基础模型(DOFA, TerraFM)在分布内设置下也不如通用模型如DINOv3。其次，当泛化到无波段重叠的新卫星时，所有模型性能下降2-4倍，而ChiViT显著优于第二名DINOv3。第三，当测试时提供额外波段，所有测试模型的性能平均下降5-25%。最后，我们表明仅使用所有波段标签微调这些模型的最后一层线性层，可以在所有卫星上获得相对一致的性能，这突显出该基准远未饱和。我们公开发布了代码和数据集，以鼓励开发具有更强跨卫星泛化能力的更面向未来的遥感模型。


### 论文摘要

The number and diversity of remote sensing satellites grows over time, while the vast majority of labeled data comes from older satellites. As the foundation models for Earth observation scale up, the cost of (re-)training to support new satellites grows too, so the generalization capabilities of the models towards new satellites become increasingly important. In this work we introduce GeoCrossBench, an extension of the popular GeoBench benchmark with a new evaluation protocol: it tests the in-distribution performance; generalization to satellites with no band overlap; and generalization to satellites with additional bands with respect to the training set. We also develop a self-supervised extension of ChannelViT, ChiViT, to improve its cross-satellite performance. First, we show that even the best foundation models for remote sensing (DOFA, TerraFM) do not outperform general purpose models like DINOv3 in the in-distribution setting. Second, when generalizing to new satellites with no band overlap, all models suffer 2-4x drop in performance, and ChiViT significantly outperforms the runner-up DINOv3. Third, the performance of all tested models drops on average by 5-25\% when given additional bands during test time. Finally, we show that fine-tuning just the last linear layer of these models using oracle labels from all bands can get relatively consistent performance across all satellites, highlighting that the benchmark is far from being saturated. We publicly release the code and the datasets to encourage the development of more future-proof remote sensing models with stronger cross-satellite generalization.

---

## 88. PLUTO-4: Frontier Pathology Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.02826v1](http://arxiv.org/abs/2511.02826v1)

**作者:** Harshith Padigela, Shima Nofallah, Atchuth Naveen Chilaparasetti, Ryun Han, Andrew Walker, Judy Shen, Chintan Shah, Blake Martin, Aashish Sood, Elliot Miller, Ben Glass, Andy Beck, Harsha Pokkalla, Syed Ashar Javed

**发布时间:** 2025-11-04

### GPT解析

### 总结

研究人员介绍了PLUTO-4，这是新一代病理基础模型，扩展了Pathology-Universal Transformer到前沿规模。该模型家族包含两种互补的Vision Transformer架构：PLUTO-4S（紧凑高效，适合多规模部署）和PLUTO-4G（前沿规模，最大化表示能力）。模型在大型多机构数据集上预训练，并在多种病理学任务上取得了最先进的性能。

### 背景

基础模型在大型病理图像语料库上的训练显示出在多种组织病理学任务中的强大迁移能力。

### 目的

开发下一代病理基础模型PLUTO-4，扩展Pathology-Universal Transformer到前沿规模，提升病理图像分析能力。

### 方法

提供两种互补的Vision Transformer架构：PLUTO-4S（使用FlexiViT设置和2D-RoPE嵌入）和PLUTO-4G（使用单一补丁大小训练）。模型使用基于DINOv2的自监督目标进行预训练，训练数据包含来自137,144名患者的551,164例WSI，跨越50个机构，覆盖60多种疾病类型和100多种染色方法。

### 主要发现

PLUTO-4在需要不同空间和生物学背景的任务上达到最先进性能，包括补丁级分类、分割和幻灯片级诊断。PLUTO-4S提供高吞吐量和稳健性能，适合实际部署；PLUTO-4G在多个病理学基准上建立新性能前沿，在皮肤病理学诊断中提高11%性能。

### 结论

PLUTO-4的多样化改进凸显了其作为转化研究和诊断用例骨干的潜力，能够改变现实世界应用。

### 翻译

在大型病理图像语料库上训练的基础模型已显示出在多种组织病理学任务中的强大迁移能力。基于这一进展，我们介绍了PLUTO-4，我们的下一代病理基础模型，将病理学通用转换器(PLUTO)扩展到前沿规模。我们在PLUTO-4家族中分享了两种互补的Vision Transformer架构：一个紧凑高效的PLUTO-4S模型，使用带有2D-RoPE嵌入的FlexiViT设置进行优化，适用于多规模部署；以及一个前沿规模的PLUTO-4G模型，使用单一补丁大小训练以最大化表示能力和稳定性。两个模型都使用从DINOv2衍生的自监督目标进行预训练，在包含来自137,144名患者的551,164例WSI的大型多机构语料库上训练，跨越50个机构，涵盖60多种疾病类型和100多种染色方法。在公共和内部基准上的全面评估表明，PLUTO-4在需要不同空间和生物学背景的任务上实现了最先进的性能，包括补丁级分类、分割和幻灯片级诊断。紧凑的PLUTO-4S为实际部署提供高吞吐量和稳健性能，而PLUTO-4G在多个病理学基准上建立了新的性能前沿，包括在皮肤病理学诊断中提高11%的性能。这些多样化的改进强调了PLUTO-4作为转化研究和诊断用例骨干的潜力，可以改变现实世界应用。


### 论文摘要

Foundation models trained on large-scale pathology image corpora have demonstrated strong transfer capabilities across diverse histopathology tasks. Building on this progress, we introduce PLUTO-4, our next generation of pathology foundation models that extend the Pathology-Universal Transformer (PLUTO) to frontier scale. We share two complementary Vision Transformer architectures in the PLUTO-4 family: a compact and efficient PLUTO-4S model optimized for multi-scale deployment using a FlexiViT setup with 2D-RoPE embeddings, and a frontier-scale PLUTO-4G model trained with a single patch size to maximize representation capacity and stability. Both models are pretrained using a self-supervised objective derived from DINOv2 on a large multi-institutional corpus containing 551,164 WSIs from 137,144 patients across over 50 institutions, spanning over 60 disease types and over 100 stains. Comprehensive evaluation across public and internal benchmarks demonstrates that PLUTO-4 achieves state-of-the-art performance on tasks requiring varying spatial and biological context, including patch-level classification, segmentation, and slide-level diagnosis. The compact PLUTO-4S provides high-throughput and robust performance for practical deployment, while PLUTO-4G establishes new performance frontiers across multiple pathology benchmarks, including an 11% improvement in dermatopathology diagnosis. These diverse improvements underscore PLUTO-4's potential to transform real-world applications as a backbone for translational research and diagnostic use cases.

---

## 89. TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.02802v1](http://arxiv.org/abs/2511.02802v1)

**作者:** Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Vinay Kumar Sankarapu

**发布时间:** 2025-11-04

### GPT解析

### 总结

TabTune是一个统一的开源库，通过单一接口标准化了表格基础模型的完整工作流程，解决了当前表格基础模型采用受限的问题。

### 背景

表格基础模型在结构化数据学习中是一个不断发展的范式，但由于异构预处理管道、碎片化API、不一致的微调程序以及缺乏针对部署导向指标的标准化评估，其采用仍然有限。

### 目的

提出TabTune库，标准化表格基础模型的完整工作流程，提供一致访问和评估能力。

### 方法

TabTune提供对七种最先进模型的一致访问，支持零样本推理、元学习、监督微调(SFT)和参数高效微调(PEFT)等多种适应策略，自动化模型感知预处理，管理架构异构性，并集成性能、校准和公平性评估模块。

### 主要发现

TabTune框架支持一致的基准测试，表格基础模型可以通过多种适应策略进行有效应用。

### 结论

TabTune库为表格基础模型的可扩展性和可重复性提供了解决方案，使研究人员和实践者能够一致地评估和部署这些模型。

### 翻译

表格基础模型代表了结构化数据学习中不断增长的范式，将大规模预训练的好处扩展到表格领域。然而，由于异构预处理管道、碎片化的API、不一致的微调程序以及缺乏针对部署导向指标(如校准和公平性)的标准化评估，其采用仍然有限。我们提出了TabTune，一个通过单一接口标准化表格基础模型完整工作流程的统一库。TabTune提供对七种最先进模型的一致访问，支持多种适应策略，包括零样本推理、元学习、监督微调(SFT)和参数高效微调(PEFT)。该框架自动化模型感知的预处理，在内部管理架构异构性，并集成了用于性能、校准和公平性的评估模块。TabTune是为可扩展性和可重复性而设计的，它能够对表格基础模型的适应策略进行一致的基准测试。该库是开源的，可在https://github.com/Lexsi-Labs/TabTune获取。


### 论文摘要

Tabular foundation models represent a growing paradigm in structured data learning, extending the benefits of large-scale pretraining to tabular domains. However, their adoption remains limited due to heterogeneous preprocessing pipelines, fragmented APIs, inconsistent fine-tuning procedures, and the absence of standardized evaluation for deployment-oriented metrics such as calibration and fairness. We present TabTune, a unified library that standardizes the complete workflow for tabular foundation models through a single interface. TabTune provides consistent access to seven state-of-the-art models supporting multiple adaptation strategies, including zero-shot inference, meta-learning, supervised fine-tuning (SFT), and parameter-efficient fine-tuning (PEFT). The framework automates model-aware preprocessing, manages architectural heterogeneity internally, and integrates evaluation modules for performance, calibration, and fairness. Designed for extensibility and reproducibility, TabTune enables consistent benchmarking of adaptation strategies of tabular foundation models. The library is open source and available at https://github.com/Lexsi-Labs/TabTune .

---

## 90. When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning

**论文链接:** [http://arxiv.org/abs/2511.02794v1](http://arxiv.org/abs/2511.02794v1)

**作者:** Chenyu Zhang, Minsol Kim, Shohreh Ghorbani, Jingyao Wu, Rosalind Picard, Patricia Maes, Paul Pu Liang

**发布时间:** 2025-11-04

**备注:** Accepted at the Multimodal Algorithmic Reasoning (MAR) Workshop,  NeurIPS 2025

### GPT解析

### 总结

本文提出了一种诊断多模态大语言模型推理过程的新方法，通过'模态破坏'概念分析模态间的交互关系，并开发了一种轻量级评估框架来识别贡献模态和破坏模态。

### 背景

多模态大语言模型(MLLMs)虽快速发展，但其推理过程仍不透明：不清楚哪个模态驱动预测，冲突如何解决，或何时一个信息流占主导地位。

### 目的

分析模态间动态交互关系，特别是高置信度单模态错误如何覆盖其他证据并误导融合结果，为多模态推理提供诊断支架。

### 方法

提出轻量级、模型无关的评估层，将每个模态视为独立代理产生候选标签和自我评估，通过简单融合机制聚合输出，区分贡献者(支持正确结果的模态)和破坏者(误导的模态)。

### 主要发现

在多模态情感识别基准案例研究中应用该方法，揭示了基础模型的系统性可靠性特征，帮助区分失败源于数据集伪影还是模型内在限制。

### 结论

该框架为多模态推理提供诊断支架，支持对融合动力学的原则性审计，为可能的模型改进干预措施提供指导。

### 翻译

尽管多模态大语言模型(MLLMs)迅速发展，它们的推理过程仍然不透明：通常不清楚哪个模态驱动预测，冲突如何解决，或者何时一个信息流占主导地位。在本文中，我们引入了模态破坏，这是一种诊断性失效模式，其中高置信度的单模态错误会覆盖其他证据并误导融合结果。为了分析此类动态过程，我们提出了一种轻量级、模型无关的评估层，将每个模态视为一个代理，产生候选标签和用于审计的简短自我评估。一个简单的融合机制聚合这些输出，暴露贡献者(支持正确结果的模态)和破坏者(误导的模态)。在基础模型的多模态情感识别基准案例研究中应用我们的诊断层，揭示了系统性的可靠性特征，提供了关于失败可能源于数据集伪影还是模型限制的见解。更广泛地说，我们的框架为多模态推理提供了诊断支架，支持对融合动力学的原则性审计，并为可能的干预措施提供信息。


### 论文摘要

Despite rapid growth in multimodal large language models (MLLMs), their reasoning traces remain opaque: it is often unclear which modality drives a prediction, how conflicts are resolved, or when one stream dominates. In this paper, we introduce modality sabotage, a diagnostic failure mode in which a high-confidence unimodal error overrides other evidence and misleads the fused result. To analyze such dynamics, we propose a lightweight, model-agnostic evaluation layer that treats each modality as an agent, producing candidate labels and a brief self-assessment used for auditing. A simple fusion mechanism aggregates these outputs, exposing contributors (modalities supporting correct outcomes) and saboteurs (modalities that mislead). Applying our diagnostic layer in a case study on multimodal emotion recognition benchmarks with foundation models revealed systematic reliability profiles, providing insight into whether failures may arise from dataset artifacts or model limitations. More broadly, our framework offers a diagnostic scaffold for multimodal reasoning, supporting principled auditing of fusion dynamics and informing possible interventions.

---

## 91. VidEmo: Affective-Tree Reasoning for Emotion-Centric Video Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.02712v1](http://arxiv.org/abs/2511.02712v1)

**作者:** Zhicheng Zhang, Weicheng Wang, Yongjie Zhu, Wenyu Qin, Pengfei Wan, Di Zhang, Jufeng Yang

**发布时间:** 2025-11-04

**备注:** 41 pages, 26 figures

### GPT解析

### 总结

该研究提出了一种新颖的情感线索引导推理框架和视频情感基础模型(VidEmo)，解决了视频中情感理解和预测的挑战，通过建立Emo-CFG数据集和两阶段调优方法，在15个面部感知任务中取得了优异性能。

### 背景

理解和预测视频中的情绪在近期研究中引起了广泛关注，得益于视频大型语言模型(VideoLLMs)的进步。尽管先进方法在视频情绪分析方面取得了进展，但情绪的本质特性（动态和线索依赖）仍带来重大挑战，使得难以以合理的理由理解复杂且不断变化的情绪状态。

### 目的

提出一种情感线索引导推理框架，统一基本属性感知、表达分析和高级情感理解；设计专门用于情感推理和指令跟随的视频情感基础模型(VidEmo)；建立情感理解任务的基础数据基础设施。

### 方法

提出情感线索引导推理框架，分阶段统一基本属性感知、表达分析和高级情感理解；开发视频情感基础模型(VidEmo)家族；采用两阶段调优过程：课程情感学习注入情感知识，情感树强化学习进行情感推理；建立包含210万多样化指令样本的情感细粒度数据集(Emo-CFG)，包括可解释的情感问答、细粒度字幕和相关理由。

### 主要发现

实验结果表明，该方法在15个面部感知任务中取得了具有竞争力的性能，树立了新的里程碑。Emo-CFG数据集为情感理解任务提供了必要资源，包括可解释的情感问答、细粒度字幕和相关理由。

### 结论

通过统一的框架和专门设计的模型，有效解决了视频中情感理解和预测的挑战。建立的数据基础设施和数据集为情感理解任务提供了重要资源，推动了该领域的发展。

### 翻译

理解和预测视频中的情绪在近期研究中引起了广泛关注，这得益于视频大型语言模型(VideoLLMs)的进步。尽管先进方法在视频情绪分析方面取得了进展，但情绪的本质特性仍带来重大挑战。情绪具有动态和线索依赖的特性，使得难以以合理的理由理解复杂且不断变化的情绪状态。为应对这些挑战，我们提出了一种新颖的情感线索引导推理框架，以分阶段的方式统一基本属性感知、表达分析和高级情感理解。我们方法的核心是一系列专为情感推理和指令跟随而设计的视频情感基础模型(VidEmo)。这些模型经历两阶段调优过程：首先进行课程情感学习以注入情感知识，然后进行情感树强化学习以进行情感推理。此外，我们建立了基础数据基础设施，并引入了一个以情感为中心的细粒度数据集(Emo-CFG)，包含210万多样化的基于指令的样本。Emo-CFG包括可解释的情感问答、细粒度字幕和相关理由，为推进情感理解任务提供了必要资源。实验结果表明，我们的方法取得了具有竞争力的性能，在15个面部感知任务中树立了新的里程碑。


### 论文摘要

Understanding and predicting emotion from videos has gathered significant attention in recent studies, driven by advancements in video large language models (VideoLLMs). While advanced methods have made progress in video emotion analysis, the intrinsic nature of emotions poses significant challenges. Emotions are characterized by dynamic and cues-dependent properties, making it difficult to understand complex and evolving emotional states with reasonable rationale. To tackle these challenges, we propose a novel affective cues-guided reasoning framework that unifies fundamental attribute perception, expression analysis, and high-level emotional understanding in a stage-wise manner. At the core of our approach is a family of video emotion foundation models (VidEmo), specifically designed for emotion reasoning and instruction-following. These models undergo a two-stage tuning process: first, curriculum emotion learning for injecting emotion knowledge, followed by affective-tree reinforcement learning for emotion reasoning. Moreover, we establish a foundational data infrastructure and introduce a emotion-centric fine-grained dataset (Emo-CFG) consisting of 2.1M diverse instruction-based samples. Emo-CFG includes explainable emotional question-answering, fine-grained captions, and associated rationales, providing essential resources for advancing emotion understanding tasks. Experimental results demonstrate that our approach achieves competitive performance, setting a new milestone across 15 face perception tasks.

---

## 92. Apriel-H1: Towards Efficient Enterprise Reasoning Models

**论文链接:** [http://arxiv.org/abs/2511.02651v1](http://arxiv.org/abs/2511.02651v1)

**作者:** Oleksiy Ostapenko, Luke Kumar, Raymond Li, Denis Kocetkov, Joel Lamy-Poirier, Shruthan Radhakrishna, Soham Parikh, Shambhavi Mishra, Sebastien Paquet, Srinivas Sunkara, Valérie Bécaert, Sathwik Tejaswi Madhusudhan, Torsten Scholak

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究提出了一种结合transformer注意力和状态空间模型(SSM)的混合架构，实现了比传统transformer模型更高的推理效率，同时保持了良好的推理性能。

### 背景

大型语言模型通过transformer架构和注意力机制实现了显著的推理能力，但transformers在注意力模块中具有二次时间和内存复杂度，且需要缓存键值状态，这严重限制了吞吐量和可扩展性。高推理吞吐量对智能体任务、长上下文推理等应用至关重要。

### 目的

开发一种混合LLM架构，结合transformer注意力和SSM序列混合器，实现高效的推理能力，同时保持较高的推理吞吐量。

### 方法

提出Apriel-H1系列混合LLMs，通过增量蒸馏从预训练推理transformer(Apriel-Nemotron-15B-Thinker)获得，逐步用线性Mamba块替换不关键的注意力层。发布了多种后蒸馏变体，分析了不同SSM与MHA比例对推理性能的影响，并在vLLM环境中测试了30/50混合变体的性能。

### 主要发现

蒸馏后的混合SSM-Transformer架构在生产环境中实现了超过2倍的更高推理吞吐量，同时推理性能仅最小程度下降。随着更多Mamba层替换MHA，推理性能会逐渐下降，但效率显著提升。

### 结论

混合SSM-Transformer架构能够在不显著损害推理质量的情况下，比预训练的transformer等效模型提供实质性的效率提升，为大型语言模型提供了一种有前途的替代方案。

### 翻译

大型语言模型(LLMs)通过具有注意力机制的transformer架构实现了显著的推理能力。然而，transformers在注意力模块(MHA)中具有二次时间和内存复杂度，并且在推理过程中需要缓存键值状态，这严重限制了吞吐量和可扩展性。高推理吞吐量对于智能体任务、长上下文推理、高请求负载下的高效部署以及更高效的测试时计算缩放至关重要。状态空间模型(SSMs)如Mamba通过具有固定大小隐藏状态的循环计算提供了具有线性推理复杂度和常量内存占用的有前途的替代方案。在本技术报告中，我们引入了Apriel-H1系列混合LLMs，结合了transformer注意力和SSM序列混合器，在150亿模型规模下实现高效推理。这些模型通过从预训练推理transformer(Apriel-Nemotron-15B-Thinker)进行增量蒸馏获得，逐步用线性Mamba块替换不太关键的注意力层。我们发布了多种后蒸馏变体，具有不同的SSM与MHA比例，并分析了当更多Mamba层替换MHA时推理性能如何下降。此外，我们发布了30/50混合变体，在推理轨迹的监督数据集上进一步微调，在生产就绪的vLLM环境中实现了超过2倍的更高推理吞吐量，且推理性能最小程度下降。这表明，蒸馏后的混合SSM-Transformer架构可以在不显著损害推理质量的情况下，比预训练的transformer等效模型提供实质性的效率提升。


### 论文摘要

Large Language Models (LLMs) achieve remarkable reasoning capabilities through transformer architectures with attention mechanisms. However, transformers suffer from quadratic time and memory complexity in the attention module (MHA) and require caching key-value states during inference, which severely limits throughput and scalability. High inference throughput is critical for agentic tasks, long-context reasoning, efficient deployment under high request loads, and more efficient test-time compute scaling.   State Space Models (SSMs) such as Mamba offer a promising alternative with linear inference complexity and a constant memory footprint via recurrent computation with fixed-size hidden states. In this technical report we introduce the Apriel-H1 family of hybrid LLMs that combine transformer attention and SSM sequence mixers for efficient reasoning at 15B model size. These models are obtained through incremental distillation from a pretrained reasoning transformer, Apriel-Nemotron-15B-Thinker, progressively replacing less critical attention layers with linear Mamba blocks.   We release multiple post-distillation variants of Apriel-H1-15B-Thinker with different SSM-to-MHA ratios and analyse how reasoning performance degrades as more Mamba layers replace MHA. Additionally, we release a 30/50 hybrid variant of Apriel-H1, further fine-tuned on a supervised dataset of reasoning traces, achieving over 2x higher inference throughput when deployed in the production-ready vLLM environment, with minimal degradation in reasoning performance. This shows that distilled hybrid SSM-Transformer architectures can deliver substantial efficiency gains over the pretrained transformer equivalent without substantially compromising the reasoning quality.

---

## 93. 论文ID: 2511.02622v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.02622v1.json'

---

## 94. Zero-Shot Multi-Animal Tracking in the Wild

**论文链接:** [http://arxiv.org/abs/2511.02591v1](http://arxiv.org/abs/2511.02591v1)

**作者:** Jan Frederik Meier, Timo Lüddecke

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了一种基于视觉基础模型的多动物跟踪框架，无需重新训练即可应用于新数据集，在多种物种和环境中表现出强大且一致的性能。

### 背景

多动物跟踪对于理解动物生态和行为至关重要，但由于栖息地、运动模式和物种外观的差异，这仍然是一个具有挑战性的任务。传统方法通常需要对每个应用场景进行大量的模型微调和启发式设计。

### 目的

探索最近的视觉基础模型在零样本多动物跟踪中的潜力，开发一个无需重新训练或超参数调整的通用跟踪框架。

### 方法

结合Grounding Dino目标检测器和Segment Anything Model 2 (SAM 2)跟踪器，以及精心设计的启发式方法，构建了一个可应用于新数据集的跟踪框架。

### 主要发现

在ChimpAct、Bird Flock Tracking、AnimalTrack和GMOT-40子集上的评估表明，该框架在多样物种和环境中表现出强大且一致的性能。

### 结论

所提出的基于视觉基础模型的跟踪框架能够在不进行重新训练的情况下适应新的数据集和应用场景，为多动物跟踪提供了新的解决方案。

### 翻译

多动物跟踪对于理解动物生态和行为至关重要。然而，由于栖息地、运动模式和物种外观的差异，这仍然是一项具有挑战性的任务。传统方法通常需要对每个应用场景进行大量的模型微调和启发式设计。在这项工作中，我们探索了最近的视觉基础模型在零样本多动物跟踪中的潜力。通过结合Grounding Dino目标检测器和Segment Anything Model 2 (SAM 2)跟踪器以及精心设计的启发式方法，我们开发了一个跟踪框架，可以应用于新数据集而无需重新训练或超参数调整。在ChimpAct、Bird Flock Tracking、AnimalTrack和GMOT-40子集上的评估表明，该框架在多样物种和环境中表现出强大且一致的性能。代码可在https://github.com/ecker-lab/SAM2-Animal-Tracking获取。


### 论文摘要

Multi-animal tracking is crucial for understanding animal ecology and behavior. However, it remains a challenging task due to variations in habitat, motion patterns, and species appearance. Traditional approaches typically require extensive model fine-tuning and heuristic design for each application scenario. In this work, we explore the potential of recent vision foundation models for zero-shot multi-animal tracking. By combining a Grounding Dino object detector with the Segment Anything Model 2 (SAM 2) tracker and carefully designed heuristics, we develop a tracking framework that can be applied to new datasets without any retraining or hyperparameter adaptation. Evaluations on ChimpAct, Bird Flock Tracking, AnimalTrack, and a subset of GMOT-40 demonstrate strong and consistent performance across diverse species and environments. The code is available at https://github.com/ecker-lab/SAM2-Animal-Tracking.

---

## 95. Resource-efficient Automatic Refinement of Segmentations via Weak Supervision from Light Feedback

**论文链接:** [http://arxiv.org/abs/2511.02576v1](http://arxiv.org/abs/2511.02576v1)

**作者:** Alix de Langlais, Benjamin Billot, Théo Aguilar Vidal, Marc-Olivier Gauci, Hervé Delingette

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了SCORE（基于区域评估的分割校正）框架，一种弱监督方法，用于改进医学图像分割结果。SCORE仅需轻量级反馈即可学习改进分割预测，无需密集的训练图像标注，在肱骨CT扫描上显著提升了初始分割性能。

### 背景

手动分割医学图像解剖区域虽然准确但劳动强度大且易变，推动了自动化方法的发展。虽然基础模型可实现多种解剖结构和成像模态的自动分割，但可能不总能达到临床准确性标准。现有分割改进方法要么需要大量用户交互，要么需要完全监督的分割进行训练。

### 目的

开发一种弱监督框架，能够仅使用轻量级反馈来改进分割预测，减少对密集训练标注的依赖。

### 方法

SCORE框架引入了一种新的损失函数，利用区域质量分数和过分割/欠分割错误标签，而不是依赖密集的训练图像标注。该方法在肱骨CT扫描上进行了验证。

### 主要发现

SCORE在肱骨CT扫描上显著改进了TotalSegmentator的初始预测，实现了与现有改进方法相当的性能，同时大大减少了监督需求和标注时间。

### 结论

SCORE提供了一种有效的弱监督分割改进方法，能够在不牺牲性能的情况下，显著减少对大量标注数据的依赖。

### 翻译

在医学图像分析中，解剖区域的划分是一项关键任务。手动分割虽然能实现高精度，但劳动强度大且容易产生变化，这促使了自动化方法的发展。最近，大量基础模型使得在多种解剖结构和成像模态上实现自动分割成为可能，但这些模型并不总能达到临床准确性标准。虽然分割改进策略可以提高性能，但当前方法依赖于大量用户交互或需要完全监督的分割进行训练。在此，我们提出了SCORE（基于区域评估的分割校正），一种弱监督框架，它仅使用训练过程中的轻量级反馈来学习改进掩膜预测。具体而言，SCORE不依赖密集的训练图像标注，而是引入了一种新的损失函数，利用区域质量分数和过分割/欠分割错误标签。我们在肱骨CT扫描上验证了SCORE，它显著改进了TotalSegmentator的初始预测，并实现了与现有改进方法相当的性能，同时大大减少了它们的监督需求和标注时间。我们的代码可在以下网址获取：https://gitlab.inria.fr/adelangl/SCORE。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何高效改进医学图像分割模型的预测结果问题。医学图像分割是临床诊断的关键任务，手动分割准确但耗时费力，而自动化基础模型虽提高了效率，但其预测结果往往达不到临床应用标准。现有的改进方法要么需要大量人工交互，要么需要完全标注的训练数据，限制了它们在临床环境中的广泛应用。因此，开发一种既能提高分割准确性又能减少标注负担的方法对推动医学图像分析的临床应用具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有分割改进方法的局限性：半自动方法需要大量人工交互，全自动方法则需要完全标注的地面真实数据。基于这一分析，作者提出使用'轻量反馈'（区域质量评分和错误类型标签）作为弱监督信号来训练改进网络。他们借鉴了3D U-Net网络架构、TotalSegmentator基础模型和Otsu阈值计算等方法，但创新性地设计了一个形态学启发的三部分损失函数，将区域级别的反馈转换为体素级别的校正指导。同时，作者还开发了专门的数据增强策略，提高模型对不同强度分割误差的鲁棒性。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用'轻量反馈'（区域质量评分和错误类型标签）作为弱监督信号来训练分割改进网络，而不需要完全标注的地面真实数据。整体流程包括：1)输入3D医学图像、基础模型生成的初始分割和结构轮廓概率图；2)使用3D U-Net作为改进网络；3)通过三部分损失函数进行训练：稳定性损失保持正确区域内部不变，扩张损失针对欠分割区域，收缩损失针对过分割区域；4)训练好的网络接收输入图像、初始分割和边界概率图，输出改进后的分割结果。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)弱监督训练策略，仅使用轻量反馈而非完全标注数据；2)形态学启发的三部分损失函数，将区域级反馈转换为体素级校正；3)边界先验整合，指导网络在解剖学合理位置进行校正；4)专门的数据增强策略，提高模型鲁棒性。相比之前的工作，SCORE完全自动化且不需要专家在推理过程中交互，训练标注时间减少了约95%，同时在多个测试集上达到了与完全监督方法相当的性能，为医学图像分割的临床应用提供了更实用的解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SCORE提出了一种创新的弱监督框架，仅通过轻量反馈就能高效训练分割改进模型，显著减少了标注需求同时达到了与完全监督方法相当的性能，为医学图像分割的临床应用提供了实用解决方案。'}


### 论文摘要

Delineating anatomical regions is a key task in medical image analysis. Manual segmentation achieves high accuracy but is labor-intensive and prone to variability, thus prompting the development of automated approaches. Recently, a breadth of foundation models has enabled automated segmentations across diverse anatomies and imaging modalities, but these may not always meet the clinical accuracy standards. While segmentation refinement strategies can improve performance, current methods depend on heavy user interactions or require fully supervised segmentations for training. Here, we present SCORE (Segmentation COrrection from Regional Evaluations), a weakly supervised framework that learns to refine mask predictions only using light feedback during training. Specifically, instead of relying on dense training image annotations, SCORE introduces a novel loss that leverages region-wise quality scores and over/under-segmentation error labels. We demonstrate SCORE on humerus CT scans, where it considerably improves initial predictions from TotalSegmentator, and achieves performance on par with existing refinement methods, while greatly reducing their supervision requirements and annotation time. Our code is available at: https://gitlab.inria.fr/adelangl/SCORE.

---

## 96. 论文ID: 2511.02503v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.02503v1.json'

---

## 97. Can Foundation Models Revolutionize Mobile AR Sparse Sensing?

**论文链接:** [http://arxiv.org/abs/2511.02215v1](http://arxiv.org/abs/2511.02215v1)

**作者:** Yiqin Zhao, Tian Guo

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究探讨了基础模型如何改变移动稀疏感知的格局，通过真实移动AR数据评估，发现基础模型在几何感知图像变换方面提供显著改进，证明了基于基础模型的稀疏感知的可扩展性及其在3D场景重建中的领先性能。

### 背景

移动感知系统长期以来在感知质量和效率之间面临基本权衡，这种权衡源于计算能力、功率和其他限制。稀疏感知作为一种关键策略，旨在获取和处理仅一部分传感器数据，以在这些限制下维持性能。

### 目的

探究基础模型是否能改变移动稀疏感知的格局。

### 方法

使用真实的移动AR数据进行评估，研究基础模型在几何感知图像变换方面的表现，这是实现准确重用跨帧信息的关键技术。

### 主要发现

基础模型在几何感知图像变换方面提供了显著改进；基于基础模型的稀疏感知具有可扩展性；在3D场景重建方面表现领先。

### 结论

研究揭示了将基础模型集成到移动稀疏感知系统中的前景和开放挑战的关键方面。

### 翻译

移动感知系统长期以来由于计算、功率和其他限制，在感知质量和效率之间面临基本权衡。稀疏感知作为一种旨在获取和处理仅一部分传感器数据的关键策略，一直是在这些限制下维持性能的重要方法。然而，现有的稀疏感知方法通常面临准确性降低的问题，因为缺失的空间和时间信息给许多感知系统带来了不确定性。在本研究中，我们探究基础模型是否能改变移动稀疏感知的格局。使用真实的移动AR数据进行评估，我们的研究表明基础模型在几何感知图像变换方面提供了显著改进，这是实现准确重用跨帧信息的关键技术。此外，我们的研究证明了基于基础模型的稀疏感知的可扩展性，并显示了其在3D场景重建中的领先性能。总体而言，我们的研究揭示了将基础模型集成到移动稀疏感知系统中的前景和开放挑战的关键方面。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决移动增强现实系统中的感知质量和效率之间的权衡问题。由于移动设备在计算能力、功耗等方面的限制，传统稀疏感知方法虽然减少了数据采集和处理量，但往往导致准确性下降。这个问题在现实中很重要，因为连续感知会消耗大量移动设备能源，影响设备续航和用户体验，而随着AR应用普及，如何在资源受限的移动设备上实现高效准确的感知变得至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到移动AR系统面临感知质量和效率的权衡问题，发现传统稀疏感知方法存在准确性下降的缺陷。他们注意到基础模型具有大规模预训练和强大泛化能力，可能从稀疏输入中提取有意义信息，从而解决传统稀疏感知问题。作者借鉴了现有稀疏感知技术、基础模型(如DINOv3和Metric3DV2)、几何感知图像变形技术和3D重建方法(如Poisson表面重建和ICP)，但将它们创新性地结合应用于移动AR稀疏感知场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用基础模型增强移动AR中的稀疏感知能力，通过基础模型从稀疏传感器数据中提取有意义信息，在减少感知频率的同时保持或提高感知质量。整体流程包括：1)使用基础模型从RGB图像估计深度图替代传统LiDAR；2)利用估计的深度图进行几何感知图像变形，实现跨帧信息重用；3)使用基础模型估计的深度图进行3D环境重建；4)分析不同稀疏感知策略下的信息重叠，优化控制策略。实验使用ScanNet++数据集，通过比较基础模型与LiDAR在图像变形和3D重建中的表现来验证方法效果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将基础模型应用于移动AR稀疏感知；2)利用基础模型估计深度图显著提高几何感知图像变形准确性；3)展示基础模型在低帧率条件下也能实现高质量3D重建；4)分析时空稀疏感知策略下的信息重叠，提出混合策略思路。相比之前工作，本文方法减少了对硬件传感器的依赖，通过更精确的深度估计提高了跨帧信息重用效果，并提出了考虑时间和空间两个维度的混合稀疏感知策略，而非传统基于时间间隔或单一维度的控制策略。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文展示了基础模型如何通过更精确的深度估计和跨帧信息重用，显著提升移动AR系统中的稀疏感知能力，实现了在减少感知频率的同时甚至提高3D重建质量的效果。'}


### 论文摘要

Mobile sensing systems have long faced a fundamental trade-off between sensing quality and efficiency due to constraints in computation, power, and other limitations. Sparse sensing, which aims to acquire and process only a subset of sensor data, has been a key strategy for maintaining performance under such constraints. However, existing sparse sensing methods often suffer from reduced accuracy, as missing information across space and time introduces uncertainty into many sensing systems. In this work, we investigate whether foundation models can change the landscape of mobile sparse sensing. Using real-world mobile AR data, our evaluations demonstrate that foundation models offer significant improvements in geometry-aware image warping, a central technique for enabling accurate reuse of cross-frame information. Furthermore, our study demonstrates the scalability of foundation model-based sparse sensing and shows its leading performance in 3D scene reconstruction. Collectively, our study reveals critical aspects of the promises and the open challenges of integrating foundation models into mobile sparse sensing systems.

---

## 98. Automated Reward Design for Gran Turismo

**论文链接:** [http://arxiv.org/abs/2511.02094v1](http://arxiv.org/abs/2511.02094v1)

**作者:** Michel Ma, Takuma Seno, Kaushik Subramanian, Peter R. Wurman, Peter Stone, Craig Sherstan

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究展示了如何利用基础模型在奖励函数空间中进行搜索，仅基于文本指令生成期望的强化学习代理，特别是在Gran Turismo 7赛车游戏中的应用。

### 背景

在强化学习代理设计中，设计师通过定义奖励函数传达期望行为，但将期望行为映射到奖励函数是一个困难的过程，特别是在自动驾驶赛车等复杂环境中。

### 目的

展示如何利用当前基础模型有效地搜索奖励函数空间，仅基于文本指令生成期望的强化学习代理。

### 方法

结合基于大型语言模型(LLM)的奖励生成、基于视觉语言模型(VLM)的偏好评估和人类反馈的系统。

### 主要发现

该系统能够生成与GT Sophy(冠军级强化学习赛车代理)具有竞争力的赛车代理，并能生成新颖的行为。

### 结论

为实际应用中的自动化奖励设计铺平了道路。

### 翻译

在设计强化学习(RL)代理时，设计师通过定义奖励函数来传达期望的代理行为——作为对代理行动的奖励或惩罚而给予的数值反馈。然而，将期望行为映射到奖励函数可能是一个困难的过程，特别是在自动驾驶赛车等复杂环境中。在本文中，我们展示了当前基础模型如何能够有效地搜索奖励函数空间，仅基于文本指令为Gran Turismo 7赛车游戏生成期望的强化学习代理。通过结合基于大型语言模型的奖励生成、基于视觉语言模型的偏好评估和人类反馈，我们展示了如何使用我们的系统生成与GT Sophy(冠军级强化学习赛车代理)具有竞争力的赛车代理，以及生成新颖的行为，为实际应用中的自动化奖励设计铺平了道路。


### 论文摘要

When designing reinforcement learning (RL) agents, a designer communicates the desired agent behavior through the definition of reward functions - numerical feedback given to the agent as reward or punishment for its actions. However, mapping desired behaviors to reward functions can be a difficult process, especially in complex environments such as autonomous racing. In this paper, we demonstrate how current foundation models can effectively search over a space of reward functions to produce desirable RL agents for the Gran Turismo 7 racing game, given only text-based instructions. Through a combination of LLM-based reward generation, VLM preference-based evaluation, and human feedback we demonstrate how our system can be used to produce racing agents competitive with GT Sophy, a champion-level RL racing agent, as well as generate novel behaviors, paving the way for practical automated reward design in real world applications.

---

## 99. Text-VQA Aug: Pipelined Harnessing of Large Multimodal Models for Automated Synthesis

**论文链接:** [http://arxiv.org/abs/2511.02046v1](http://arxiv.org/abs/2511.02046v1)

**作者:** Soham Joshi, Shwet Kamal Mishra, Viswanath Gopalakrishnan

**发布时间:** 2025-11-03

**备注:** First two authors contributed equally

### GPT解析

### 总结

该论文提出了一种自动化方法，用于基于场景文本创建大规模视觉问答数据集，整合多种模型和技术避免了繁琐的人工标注过程。

### 背景

创建大规模视觉问答(text-VQA)数据库需要熟练的人工标注，这既繁琐又具有挑战性。随着处理视觉和语言模态的基础模型的出现以及OCR系统的成熟，为解决这个问题提供了新的可能。

### 目的

建立一个端到端的管道，能够根据给定图像中的场景文本自动合成问答(QA)对，实现text-VQA数据集的自动化合成。

### 方法

提出一个自动化合成text-VQA数据集的管道，整合多种模型和算法，包括OCR检测和识别(文本定位)、感兴趣区域(ROI)检测、标题生成和问题生成，将这些组件整合成一个连贯的管道以自动合成和验证QA对。

### 主要发现

该管道能够生成可靠的QA对，并能根据场景文本数据的可用性进行扩展，成功创建了包含约72K个QA对、基于约44K张图像的大规模text-VQA数据集。

### 结论

据我们所知，这是第一个提出的管道，可以自动合成和验证大规模text-VQA数据集，为视觉问答领域提供了新的数据集构建方法。

### 翻译

为视觉问答任务创建大规模数据库，特别是针对场景中的文本数据(text-VQA)，需要熟练的人工标注，这既繁琐又具有挑战性。随着处理视觉和语言模态的基础模型的出现以及OCR系统的成熟，现在有必要建立一个端到端的管道，能够根据给定图像中的场景文本合成问答(QA)对。我们提出了一个用于text-VQA数据集自动合成的管道，可以生成可靠的QA对，并能根据场景文本数据的可用性进行扩展。我们提出的方法利用了多种模型和算法的能力，包括OCR检测和识别(文本定位)、感兴趣区域(ROI)检测、标题生成和问题生成。这些组件被整合成一个连贯的管道，以自动合成和验证QA对。据我们所知，这是第一个提出的管道，可以自动合成和验证包含约72K个QA对、基于约44K张图像的大规模text-VQA数据集。


### 论文摘要

Creation of large-scale databases for Visual Question Answering tasks pertaining to the text data in a scene (text-VQA) involves skilful human annotation, which is tedious and challenging. With the advent of foundation models that handle vision and language modalities, and with the maturity of OCR systems, it is the need of the hour to establish an end-to-end pipeline that can synthesize Question-Answer (QA) pairs based on scene-text from a given image. We propose a pipeline for automated synthesis for text-VQA dataset that can produce faithful QA pairs, and which scales up with the availability of scene text data. Our proposed method harnesses the capabilities of multiple models and algorithms involving OCR detection and recognition (text spotting), region of interest (ROI) detection, caption generation, and question generation. These components are streamlined into a cohesive pipeline to automate the synthesis and validation of QA pairs. To the best of our knowledge, this is the first pipeline proposed to automatically synthesize and validate a large-scale text-VQA dataset comprising around 72K QA pairs based on around 44K images.

---

## 100. Assessing the value of Geo-Foundational Models for Flood Inundation Mapping: Benchmarking models for Sentinel-1, Sentinel-2, and Planetscope for end-users

**论文链接:** [http://arxiv.org/abs/2511.01990v1](http://arxiv.org/abs/2511.01990v1)

**作者:** Saurabh Kaushik, Lalit Maurya, Elizabeth Tellman, ZhiJie Zhang

**发布时间:** 2025-11-03

### GPT解析

### 总结

该研究对多种地理基础模型(GFMs)与传统模型在洪水淹没制图方面进行了系统比较，发现GFMs特别是Clay模型在性能和效率方面均优于传统模型，即使在数据有限的情况下也能保持良好表现。

### 背景

地理基础模型(GFMs)能够快速可靠地从卫星影像中提取时空信息，通过利用位置和时间嵌入改进洪水淹没制图。然而，尚不清楚GFMs是否优于传统模型如U-Net，且缺乏对不同传感器和数据可用性场景的系统比较。

### 目的

评估三种GFMs(Prithvi 2.0、Clay V1.5、DOFA和UViT)与TransNorm、U-Net和Attention U-Net在PlanetScope、Sentinel-1和Sentinel-2上的表现，为用户提供模型选择指导。

### 方法

使用多种传感器数据进行模型比较，进行区域外交叉验证(跨五个区域)，进行少样本实验评估少量训练数据下的表现，并比较不同模型的计算时间和模型大小。

### 主要发现

1) 所有GFMs表现相当，不同传感器上最佳和最差模型间仅2-5%差异；2) Clay在PlanetScope和Sentinel-2上表现最佳，Prithvi在Sentinel-1上领先；3) 在交叉验证中，Clay在所有传感器上表现略优于其他模型；4) Clay在保留细节方面具有优势；5) 仅用五张训练图像，Clay表现优于其他模型；6) Clay计算效率更高，模型较小(2600万参数)，比Prithvi快约3倍，比DOFA快2倍。

### 结论

与先前发现相反，研究结果表明GFMs在洪水制图准确性方面比传统U-Net有小到中等程度的提升，同时计算成本和标注工作量更低。

### 翻译

地理基础模型(GFMs)能够快速可靠地从卫星影像中提取时空信息，通过利用位置和时间嵌入改进洪水淹没制图。尽管它们有潜力，但尚不清楚GFMs是否优于传统模型如U-Net。对不同传感器和数据可用性场景的系统比较仍然缺乏，这是指导用户选择模型的重要步骤。为此，我们评估了三种GFMs(Prithvi 2.0、Clay V1.5、DOFA和Prithvi的变体UViT)与TransNorm、U-Net和Attention U-Net在PlanetScope、Sentinel-1和Sentinel-2上的表现。我们观察到所有GFMs都具有竞争力，不同传感器上最佳和最差模型间仅2-5%的差异。Clay在PlanetScope(0.79 mIoU)和Sentinel-2(0.70)上表现最佳，而Prithvi在Sentinel-1上领先(0.57)。在跨五个区域的外交叉验证中，Clay在所有传感器上表现略好(PlanetScope: 0.72(0.04), Sentinel-2: 0.66(0.07), Sentinel-1: 0.51(0.08))，优于Prithvi和DOFA。在所有19个站点的外交叉验证中，Clay比U-Net提高了4%的准确率。视觉检查显示Clay在保留细节方面具有优势。少样本实验显示，仅用五张训练图像，Clay在PlanetScope上达到0.64 mIoU，优于Prithvi(0.24)和DOFA(0.35)。在计算时间方面，由于模型较小(2600万参数)，Clay是更好的选择，比Prithvi(6.5亿参数)快约3倍，比DOFA(4.1亿参数)快2倍。与先前发现相反，我们的研究结果表明，与传统U-Net相比，GFMs在洪水制图准确性方面有小到中等程度的提升，同时计算成本和标注工作量更低。


### 论文摘要

Geo-Foundational Models (GFMs) enable fast and reliable extraction of spatiotemporal information from satellite imagery, improving flood inundation mapping by leveraging location and time embeddings. Despite their potential, it remains unclear whether GFMs outperform traditional models like U-Net. A systematic comparison across sensors and data availability scenarios is still lacking, which is an essential step to guide end-users in model selection. To address this, we evaluate three GFMs, Prithvi 2.0, Clay V1.5, DOFA, and UViT (a Prithvi variant), against TransNorm, U-Net, and Attention U-Net using PlanetScope, Sentinel-1, and Sentinel-2. We observe competitive performance among all GFMs, with only 2-5% variation between the best and worst models across sensors. Clay outperforms others on PlanetScope (0.79 mIoU) and Sentinel-2 (0.70), while Prithvi leads on Sentinel-1 (0.57). In leave-one-region-out cross-validation across five regions, Clay shows slightly better performance across all sensors (mIoU: 0.72(0.04), 0.66(0.07), 0.51(0.08)) compared to Prithvi (0.70(0.05), 0.64(0.09), 0.49(0.13)) and DOFA (0.67(0.07), 0.64(0.04), 0.49(0.09)) for PlanetScope, Sentinel-2, and Sentinel-1, respectively. Across all 19 sites, leave-one-region-out cross-validation reveals a 4% improvement by Clay compared to U-Net. Visual inspection highlights Clay's superior ability to retain fine details. Few-shot experiments show Clay achieves 0.64 mIoU on PlanetScope with just five training images, outperforming Prithvi (0.24) and DOFA (0.35). In terms of computational time, Clay is a better choice due to its smaller model size (26M parameters), making it ~3x faster than Prithvi (650M) and 2x faster than DOFA (410M). Contrary to previous findings, our results suggest GFMs offer small to moderate improvements in flood mapping accuracy at lower computational cost and labeling effort compared to traditional U-Net.

---

## 101. Towards Robust Mathematical Reasoning

**论文链接:** [http://arxiv.org/abs/2511.01846v1](http://arxiv.org/abs/2511.01846v1)

**作者:** Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, Junehyuk Jung

**发布时间:** 2025-11-03

**备注:** EMNLP 2025 (main conference),  https://aclanthology.org/2025.emnlp-main.1794/

### GPT解析

### 总结

本文提出了IMO-Bench，一个针对国际数学奥林匹克竞赛(IMO)级别的高级推理评估基准套件，包含IMO-AnswerBench和IMO-Proof Bench两部分，用于评估基础模型的数学推理能力。该基准在Gemini Deep Think模型上取得了显著成果，并在IMO 2025上获得金牌表现。

### 背景

现有基础模型的数学推理能力评估存在局限性，要么过于简单，要么只关注获取简短正确答案，缺乏对高级数学推理能力的有效评估。

### 目的

开发一个针对国际数学奥林匹克竞赛(IMO)级别的高级推理评估基准，以更准确地评估基础模型的数学推理能力。

### 方法

构建了IMO-Bench评估套件，包括IMO-AnswerBench(测试400个多样化奥林匹克问题)和IMO-Proof Bench(评估证明写作能力，包含基础和高级IMO级别问题及详细评分指南)。此外，还构建了IMO-GradingBench，包含1000个人类评分的证明。

### 主要发现

Gemini Deep Think模型在IMO-AnswerBench上达到80.0%的准确率，在高级IMO-Proof Bench上达到65.7%，分别领先其他最佳非Gemini模型6.9%和42.4%。基于Gemini推理能力的自动评分器与人工评估有很好的相关性。

### 结论

IMO-Bench为评估高级数学推理能力提供了有效工具，IMO-GradingBench促进了长答案自动评估的发展，这些工具将帮助社区推进强大的数学推理能力。

### 翻译

找到正确的北极星指标对于推进基础模型的数学推理能力至关重要，特别是考虑到现有评估要么太简单，要么只关注获取正确的简短答案。为解决这些问题，我们提出了IMO-Bench，一个由顶级专家评审的高级推理基准套件，专门针对国际数学奥林匹克竞赛(IMO)的水平，这是年轻数学家最负盛名的平台。IMO-AnswerBench首先测试模型在400个多样化奥林匹克问题上的表现，这些问题有可验证的简短答案。IMO-Proof Bench是下一级别的证明写作能力评估，包含基础和高级IMO级别问题以及详细的评分指南，以促进自动评分。这些基准在我们的Gemini Deep Think在IMO 2025上取得历史性金牌表现(Luong和Lockhart, 2025)中发挥了关键作用。我们的模型在IMO-AnswerBench上达到80.0%，在高级IMO-Proof Bench上达到65.7%，分别大幅领先最佳非Gemini模型6.9%和42.4%。我们还表明，使用Gemini推理能力构建的自动评分器与人工评估有很好的相关性，并构建了IMO-GradingBench，包含1000个证明的人类评分，以促进长答案自动评估的进一步发展。我们希望IMO-Bench将帮助社区推进强大的数学推理能力，并在https://imobench.github.io/上发布。


### 论文摘要

Finding the right north-star metrics is highly critical for advancing the mathematical reasoning capabilities of foundation models, especially given that existing evaluations are either too easy or only focus on getting correct short answers. To address these issues, we present IMO-Bench, a suite of advanced reasoning benchmarks, vetted by a panel of top specialists and that specifically targets the level of the International Mathematical Olympiad (IMO), the most prestigious venue for young mathematicians. IMO-AnswerBench first tests models on 400 diverse Olympiad problems with verifiable short answers. IMO-Proof Bench is the next-level evaluation for proof-writing capabilities, which includes both basic and advanced IMO level problems as well as detailed grading guidelines to facilitate automatic grading. These benchmarks played a crucial role in our historic achievement of the gold-level performance at IMO 2025 with Gemini Deep Think (Luong and Lockhart, 2025). Our model achieved 80.0% on IMO-AnswerBench and 65.7% on the advanced IMO-Proof Bench, surpassing the best non-Gemini models by large margins of 6.9% and 42.4% respectively. We also showed that autograders built with Gemini reasoning correlate well with human evaluations and construct IMO-GradingBench, with 1000 human gradings on proofs, to enable further progress in automatic evaluation of long-form answers. We hope that IMO-Bench will help the community towards advancing robust mathematical reasoning and release it at https://imobench.github.io/.

---

## 102. How Far Are Surgeons from Surgical World Models? A Pilot Study on Zero-shot Surgical Video Generation with Expert Assessment

**论文链接:** [http://arxiv.org/abs/2511.01775v1](http://arxiv.org/abs/2511.01775v1)

**作者:** Zhen Chen, Qing Xu, Jinlin Wu, Biao Yang, Yuhao Zhai, Geng Guo, Jing Zhang, Yinlu Ding, Nassir Navab, Jiebo Luo

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究提出了SurgVeo基准测试和手术合理性金字塔(SPP)框架，用于评估手术视频生成模型。研究发现先进模型在视觉层面表现良好，但在理解手术操作、环境反馈和手术意图等深层知识方面存在明显不足。

### 背景

基础模型在视频生成领域展现出作为物理世界模拟模型的潜力，但在手术等高风险领域需要深度、专业的因果知识，而非通用物理规则，这一领域仍存在研究空白。

### 目的

系统解决手术视频生成模型评估的挑战，提出首个专家策划的手术视频生成模型评估基准SurgVeo，以及专门用于评估模型输出的四层框架SPP。

### 方法

基于SurgVeo基准，让先进Veo-3模型对腹腔镜和神经外科手术片段进行零样本预测，由四位认证外科医生根据SPP框架评估生成视频。

### 主要发现

研究揭示了明显的'合理性差距'：Veo-3在视觉感知合理性方面表现出色，但在器械操作合理性、环境反馈合理性和手术意图合理性等更高层次评估中严重失败。

### 结论

该研究提供了手术AI中视觉模仿与因果理解之间鸿沟的首次定量证据，为开发能够处理专业医疗领域复杂性的未来模型奠定了基础和路线图。

### 翻译

视频生成领域的基础模型作为模拟物理世界的潜在世界模型展现出卓越能力。然而，在手术等高风险领域的应用仍是一个关键未探索的空白，这些领域需要深度、专业的因果知识而非通用物理规则。为系统解决这一挑战，我们提出了SurgVeo，这是首个用于手术视频生成模型评估的专家策划基准，以及手术合理性金字塔(SPP)，这是一个新颖的四层框架，专门用于从基本外观到复杂手术策略评估模型输出。基于SurgVeo基准，我们让先进的Veo-3模型对来自腹腔镜和神经外科手术片段进行零样本预测任务。由四位认证外科医生组成的评估小组根据SPP评估生成的视频。我们的研究结果揭示了一个明显的'合理性差距'：虽然Veo-3在视觉感知合理性方面取得了卓越成就，但在SPP的更高层次上却严重失败，包括器械操作合理性、环境反馈合理性和手术意图合理性。这项工作提供了手术AI中视觉上令人信服的模仿与因果理解之间鸿沟的首次定量证据。我们从SurgVeo和SPP中获得的研究结果为开发能够处理专业、现实医疗领域复杂性的未来模型奠定了重要基础和路线图。


### 论文摘要

Foundation models in video generation are demonstrating remarkable capabilities as potential world models for simulating the physical world. However, their application in high-stakes domains like surgery, which demand deep, specialized causal knowledge rather than general physical rules, remains a critical unexplored gap. To systematically address this challenge, we present SurgVeo, the first expert-curated benchmark for video generation model evaluation in surgery, and the Surgical Plausibility Pyramid (SPP), a novel, four-tiered framework tailored to assess model outputs from basic appearance to complex surgical strategy. On the basis of the SurgVeo benchmark, we task the advanced Veo-3 model with a zero-shot prediction task on surgical clips from laparoscopic and neurosurgical procedures. A panel of four board-certified surgeons evaluates the generated videos according to the SPP. Our results reveal a distinct "plausibility gap": while Veo-3 achieves exceptional Visual Perceptual Plausibility, it fails critically at higher levels of the SPP, including Instrument Operation Plausibility, Environment Feedback Plausibility, and Surgical Intent Plausibility. This work provides the first quantitative evidence of the chasm between visually convincing mimicry and causal understanding in surgical AI. Our findings from SurgVeo and the SPP establish a crucial foundation and roadmap for developing future models capable of navigating the complexities of specialized, real-world healthcare domains.

---

## 103. AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling

**论文链接:** [http://arxiv.org/abs/2511.01747v1](http://arxiv.org/abs/2511.01747v1)

**作者:** Guangkun Nie, Gongzheng Tang, Yujie Xiao, Jun Li, Shun Huang, Deyun Zhang, Qinghao Zhao, Shenda Hong

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究提出了AnyPPG，一个在大型多源同步PPG-ECG数据上预训练的PPG基础模型，通过与ECG表示对齐学习有生理意义的特征，在多项生理分析和多器官疾病诊断任务中取得了最先进的性能，展示了PPG作为全面健康评估模式的潜力。

### 背景

PPG是一种非侵入式且易于获取的健康监测方式，可在临床环境外使用。然而，现有研究受限于标记数据的规模和多样性，这限制了模型的准确性、泛化能力以及更广泛应用的探索。

### 目的

研究通过整合基础模型技术，探索PPG进行全面健康档案构建的潜力。

### 方法

提出了AnyPPG，一个在大型、多源同步PPG-ECG数据上预训练的PPG基础模型。通过在共享空间中对齐PPG和ECG表示，AnyPPG从未标记的信号中学习有生理意义的特征。在多样化的下游任务中评估了其能力，包括传统的生理分析和全面的多器官疾病诊断。

### 主要发现

在跨越六个独立数据集的十一个生理分析任务中，AnyPPG取得了最先进的性能，与次优模型相比，回归任务平均提高了12.8%，分类任务平均提高了9.1%。在多器官疾病诊断中，AnyPPG展示了广泛的跨系统诊断潜力，在1,014个ICD-10三位数疾病类别中，13个达到0.8以上的AUC，137个超过0.7。除了在心血管疾病方面表现强劲外，AnyPPG在帕金森病和慢性肾病等非心血管状况中也显示出实质性的诊断价值。

### 结论

AnyPPG证明，通过与ECG进行生理对齐训练的PPG基础模型可以产生准确且稳健的信号表示。基于此能力，它强调了PPG作为评估全身和多器官健康模式的潜力。

### 翻译

背景：光电容积脉搏波描记术提供了一种非侵入式且易于获取的健康监测方式，可用于临床环境之外的健康监测。然而，现有研究受限于标记数据的规模和多样性，这限制了模型的准确性、泛化能力以及更广泛应用的探索。本研究通过整合基础模型技术，调查了PPG进行全面健康档案构建的潜力。方法：我们提出了AnyPPG，一个在大型、多源同步PPG-ECG数据上预训练的PPG基础模型。通过在共享空间中对齐PPG和ECG表示，AnyPPG从未标记的信号中学习有生理意义的特征。其能力在多样化的下游任务中得到了进一步评估，涵盖传统的生理分析和全面的多器官疾病诊断。结果：在跨越六个独立数据集的十一个生理分析任务中，AnyPPG取得了最先进的性能，与次优模型相比，回归任务平均提高了12.8%，分类任务平均提高了9.1%。在多器官疾病诊断中，AnyPPG展示了广泛的跨系统诊断潜力。在1,014个ICD-10三位数疾病类别中，13个达到0.8以上的AUC，137个超过0.7。除了在心力衰竭、瓣膜疾病和高血压等心血管疾病方面表现强劲外，AnyPPG在帕金森病和慢性肾病等非心血管状况中也显示出实质性的诊断价值。结论：AnyPPG证明，通过与ECG进行生理对齐训练的PPG基础模型可以产生准确且稳健的信号表示。基于此能力，它强调了PPG作为评估全身和多器官健康模式的潜力。


### 论文摘要

Background: Photoplethysmography (PPG) offers a noninvasive and accessible modality for health monitoring beyond clinical settings. However, existing studies are limited by the scale and diversity of labeled data, constraining model accuracy, generalizability, and the exploration of broader applications. This study investigates the potential of PPG for holistic health profiling through the integration of foundation model techniques.   Methods: We present AnyPPG, a PPG foundation model pretrained on large-scale, multi-source synchronized PPG-ECG data. By aligning PPG and ECG representations within a shared space, AnyPPG learns physiologically meaningful features from unlabeled signals. Its capability was further evaluated across a diverse set of downstream tasks, encompassing both conventional physiological analysis and comprehensive multi-organ disease diagnosis.   Results: Across eleven physiological analysis tasks spanning six independent datasets, AnyPPG achieved state-of-the-art performance, with average improvements of 12.8% in regression and 9.1% in classification tasks over the next-best model. In multi-organ disease diagnosis, AnyPPG demonstrated broad cross-system diagnostic potential. Among 1,014 ICD-10 three-digit disease categories, 13 achieved an AUC above 0.8 and 137 exceeded 0.7. Beyond strong performance in cardiovascular diseases such as heart failure, valvular disorders, and hypertension, AnyPPG also showed substantial diagnostic value for non-cardiovascular conditions, exemplified by Parkinson's disease (AUC = 0.78) and chronic kidney disease (AUC = 0.74).   Conclusions: AnyPPG demonstrates that a PPG foundation model trained through physiological alignment with ECG can produce accurate and robust signal representations. Building on this capability, it underscores the potential of PPG as a modality for comprehensive assessment of systemic and multi-organ health.

---

## 104. DINO-MX: A Modular & Flexible Framework for Self-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2511.01610v1](http://arxiv.org/abs/2511.01610v1)

**作者:** Mahmut Selman Gokmen, Cody Bumgardner

**发布时间:** 2025-11-03

### GPT解析

### 总结

DINO-MX是一个模块化、可扩展的视觉基础模型训练框架，结合了DINO、DINOv2和DINOv3的核心原则，支持多种Transformer架构和Hugging Face生态系统，通过多种训练策略和分布式训练方法实现了高效的自监督视觉模型训练。

### 背景

现有的视觉基础模型训练流程通常不够灵活、局限于特定领域或计算成本高，限制了它们在不同领域和资源环境中的可用性。

### 目的

开发一个模块化、可扩展的训练框架，结合DINO系列模型的核心原则，支持各种基于Transformer的架构，并与Hugging Face生态系统完全兼容，提高视觉基础模型的可用性和适应性。

### 方法

DINO-MX是一个统一配置驱动的训练框架，支持低秩自适应(LoRA)、层冻结、知识蒸馏等多种训练策略，通过分布式数据并行(DDP)和全分片数据并行(FSDP)支持分布式训练，适用于自然和专门的数据类型，包括单通道和多通道图像。

### 主要发现

在不同数据集上的实验结果表明，DINO-MX在显著降低计算成本的同时实现了具有竞争力的性能。此外，它提供了可解释性工具和标签引导的数据增强方法，可以改进基于注意力的定位，无需额外的检测或分割头。

### 结论

DINO-MX为开发、适应和基准测试自监督视觉模型提供了一个可重现且可扩展的基础，适用于各种研究和实际应用。

### 翻译

视觉基础模型(VFMs)通过自监督方法推动了表征学习的进步。然而，现有的训练流程通常不够灵活、局限于特定领域或计算成本高，这限制了它们在不同领域和资源环境中的可用性。DINO-MX是一个模块化且可扩展的训练框架，在统一的配置驱动系统中结合了DINO、DINOv2和DINOv3的核心原则。它支持各种基于Transformer的架构，并与Hugging Face生态系统完全兼容。该框架包括多种训练策略，如低秩自适应(LoRA)、层冻结和知识蒸馏，同时通过分布式数据并行(DDP)和全分片数据并行(FSDP)支持分布式训练。DINO-MX设计用于处理自然和专门的数据类型，包括单通道和多通道图像。在不同数据集上的实验结果表明，DINO-MX在显著降低计算成本的同时实现了具有竞争力的性能。此外，它提供了解释性工具和一种标签引导的数据增强方法，可以改进基于注意力的定位，而无需额外的检测或分割头。DINO-MX为开发、适应和基准测试自监督视觉模型提供了一个可重现且可扩展的基础，适用于各种研究和实际应用。


### 论文摘要

Vision Foundation Models (VFMs) have advanced representation learning through self-supervised methods. However, existing training pipelines are often inflexible, domain-specific, or computationally expensive, which limits their usability across different domains and resource settings. DINO-MX is a modular and extensible training framework that combines the core principles of DINO, DINOv2 and DINOv3 within a unified configuration-driven system. It supports a variety of transformer-based architectures and is fully compatible with the Hugging Face ecosystem. The framework includes multiple training strategies such as low-rank adaptation (LoRA), layer freezing, and knowledge distillation, along with support for distributed training through both Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP). DINO-MX is designed to work with both natural and specialized data types, including single- and multi-channel images. Experimental results on diverse datasets show that DINO-MX achieves competitive performance while significantly reducing computational costs. Additionally, it offers interpretability tools and a label-guided data augmentation method that improves attention-based localization without the need for extra detection or segmentation heads. DINO-MX provides a reproducible and scalable foundation for developing, adapting, and benchmarking self-supervised vision models across a range of research and real-world applications.

---

## 105. Analyzing Sustainability Messaging in Large-Scale Corporate Social Media

**论文链接:** [http://arxiv.org/abs/2511.01550v1](http://arxiv.org/abs/2511.01550v1)

**作者:** Ujjwal Sharma, Stevan Rudinac, Ana Mićković, Willemijn van Dolen, Marcel Worring

**发布时间:** 2025-11-03

### GPT解析

### 总结

这篇论文介绍了一个多模态分析流程，利用视觉和语言领域的大型基础模型来分析企业社交媒体内容，特别是与可持续发展相关的传播。

### 背景

企业在X平台(前Twitter)上的信息在不断变化、多模态且往往模糊不清，这给分析带来了挑战。

### 目的

分析企业社交媒体内容，揭示不同行业在可持续发展目标参与度上的差异、时间趋势以及企业信息、环境、社会、治理风险和消费者参与之间的关联。

### 方法

使用大型语言模型集合标注企业推文与可持续发展目标的一致性，并利用视觉语言模型在语义簇框架内分析视觉可持续性传播模式。

### 主要发现

揭示了不同行业在可持续发展目标参与度上的差异、时间趋势以及企业信息、环境、社会、治理风险和消费者参与之间的关联。

### 结论

自动标签生成和语义视觉聚类方法可广泛适用于其他领域，为大规模社交媒体分析提供了灵活框架。

### 翻译

在这项工作中，我们介绍了一个多模态分析流程，它利用视觉和语言领域的大型基础模型来分析企业社交媒体内容，重点关注与可持续发展相关的传播。针对企业在X平台(前Twitter)等平台上不断变化、多模态且往往模糊不清的企业信息传播所面临的挑战，我们采用大型语言模型集合来标注大量企业推文，使其与17个可持续发展目标的主题保持一致。这种方法避免了昂贵的、特定任务的标注需求，并探索了此类模型作为社交媒体数据的临时标注者的潜力，能够以可扩展的方式高效捕捉对可持续性主题的显性和隐性引用。作为文本分析的补充，我们在一个使用语义簇的视觉理解框架内利用视觉语言模型，揭示视觉可持续性传播中的模式。这种综合方法揭示了不同行业在可持续发展目标参与度上的差异、时间趋势以及企业信息、环境、社会、治理风险和消费者参与之间的关联。我们的方法——自动标签生成和语义视觉聚类——可广泛适用于其他领域，并为大规模社交媒体分析提供了一个灵活的框架。


### 论文摘要

In this work, we introduce a multimodal analysis pipeline that leverages large foundation models in vision and language to analyze corporate social media content, with a focus on sustainability-related communication. Addressing the challenges of evolving, multimodal, and often ambiguous corporate messaging on platforms such as X (formerly Twitter), we employ an ensemble of large language models (LLMs) to annotate a large corpus of corporate tweets on their topical alignment with the 17 Sustainable Development Goals (SDGs). This approach avoids the need for costly, task-specific annotations and explores the potential of such models as ad-hoc annotators for social media data that can efficiently capture both explicit and implicit references to sustainability themes in a scalable manner. Complementing this textual analysis, we utilize vision-language models (VLMs), within a visual understanding framework that uses semantic clusters to uncover patterns in visual sustainability communication. This integrated approach reveals sectoral differences in SDG engagement, temporal trends, and associations between corporate messaging, environmental, social, governance (ESG) risks, and consumer engagement. Our methods-automatic label generation and semantic visual clustering-are broadly applicable to other domains and offer a flexible framework for large-scale social media analysis.

---

## 106. Driving scenario generation and evaluation using a structured layer representation and foundational models

**论文链接:** [http://arxiv.org/abs/2511.01541v1](http://arxiv.org/abs/2511.01541v1)

**作者:** Arthur Hubert, Gamal Elghazaly, Raphaël Frank

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种结构化的五层模型，用于改进罕见驾驶场景的评估和生成，并结合大型基础模型使用数据增强策略生成新的驾驶场景。该模型为场景中的每个代理引入子类和特征，使用特定于层模型的嵌入进行比较，并评估了合成数据集的相关性。

### 背景

罕见且具有挑战性的驾驶场景对自动驾驶车辆的发展至关重要。由于这些场景难以遇到，使用生成模型模拟或生成它们是一种流行的方法。之前的研究已经尝试在层模型中结构化驾驶场景表示。

### 目的

提出一种结构化的五层模型，以提高罕见场景的评估和生成能力。研究并调整两个指标来评估合成数据集在结构化表示背景下的相关性。

### 方法

1. 提出结构化的五层模型改进驾驶场景表示；2. 结合大型基础模型采用数据增强策略生成新场景；3. 为每个代理引入子类和特征；4. 使用特定于层模型的嵌入进行比较；5. 研究多样性分数和原创性分数两个评估指标；6. 在不同生成设置下展示指标并进行合成视频的定性评估。

### 主要发现

论文展示了在不同生成设置下多样性和原创性分数的应用，以及从结构化场景描述生成的合成视频的定性评估结果。代码和扩展结果可在提供的GitHub链接获取。

### 结论

提出的结构化五层模型能够有效改进罕见驾驶场景的评估和生成，结合大型基础模型和数据增强策略可以生成高质量的合成驾驶场景。

### 翻译

罕见且具有挑战性的驾驶场景对自动驾驶车辆的发展至关重要。由于它们难以遇到，使用生成模型模拟或生成它们是一种流行方法。在之前尝试在层模型中结构化驾驶场景表示的基础上，我们提出了一种结构化的五层模型来改进罕见场景的评估和生成。我们使用该模型与大型基础模型结合，采用数据增强策略生成新的驾驶场景。与之前的表示方法不同，我们的结构为场景中的每个代理引入了子类和特征，使我们能够使用特定于我们层模型的嵌入来比较它们。我们研究并调整了两个指标来评估合成数据集在结构化表示背景下的相关性：多样性分数估计数据集中场景之间的差异程度，而原创性分数计算合成数据集与真实参考集的相似程度。本文展示了在不同生成设置下的这两个指标，以及对从结构化场景描述生成的合成视频的定性评估。代码和扩展结果可在https://github.com/Valgiz/5LMSG找到。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成和评估罕见或具有挑战性的驾驶场景（Edge Cases）的问题。这个问题在现实中非常重要，因为这类场景对自动驾驶系统的开发至关重要，但它们在真实驾驶数据中非常稀缺，难以遇到。通过模拟或生成这些罕见场景，研究人员可以测试和改进自动驾驶系统应对极端情况的能力，从而提高系统的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到罕见驾驶场景对自动驾驶开发的重要性，但同时也意识到这些场景难以获取。他们借鉴了Scholtes等人提出的6层模型(6LM)，但将其简化为5层模型(5LM)，去掉了与数字信息相关的第6层。作者利用大型语言模型(LLM)和基础模型来生成新的驾驶场景，采用数据增强策略。他们还研究并调整了两个指标来评估合成数据集的相关性：多样性分数和原创性分数。整个设计过程是基于对现有工作的分析和对自动驾驶场景生成需求的深入理解。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用结构化的五层模型(5LM)来表示驾驶场景，以便更好地生成和评估罕见场景，并利用大型语言模型和基础模型来增强现有真实场景数据集。整体实现流程包括：1)将驾驶场景分解为5个层次（道路结构、周围建筑、临时变化、动态对象、环境条件）；2)使用多模态大语言模型分析真实驾驶数据并根据5层模型格式化；3)通过两种策略生成新场景（非结构化编辑和结构化层编辑）；4)使用专门的评估指标（原创性分数和多样性分数）来评估生成场景的质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结构化的五层模型(5LM)提供标准化的驾驶场景表示，引入子类和特征使每个代理都能被详细描述；2)新的合成场景生成策略，通过编辑现有真实场景的特定组件创建边缘案例；3)文本评估方法，提出原创性分数和多样性分数来评估生成场景质量；4)结合视觉和文本生成，先生成结构化场景描述再转化为视觉表示。相比之前的工作，这种方法提供了更精细的场景控制、更全面的评估指标，并确保了生成场景的语义合理性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于结构化五层模型和基础模型的驾驶场景生成与评估方法，通过数据增强策略创建多样化且原创的罕见驾驶场景，为自动驾驶系统开发和测试提供了新的工具和评估框架。'}


### 论文摘要

Rare and challenging driving scenarios are critical for autonomous vehicle development. Since they are difficult to encounter, simulating or generating them using generative models is a popular approach. Following previous efforts to structure driving scenario representations in a layer model, we propose a structured five-layer model to improve the evaluation and generation of rare scenarios. We use this model alongside large foundational models to generate new driving scenarios using a data augmentation strategy. Unlike previous representations, our structure introduces subclasses and characteristics for every agent of the scenario, allowing us to compare them using an embedding specific to our layer-model. We study and adapt two metrics to evaluate the relevance of a synthetic dataset in the context of a structured representation: the diversity score estimates how different the scenarios of a dataset are from one another, while the originality score calculates how similar a synthetic dataset is from a real reference set. This paper showcases both metrics in different generation setup, as well as a qualitative evaluation of synthetic videos generated from structured scenario descriptions. The code and extended results can be found at https://github.com/Valgiz/5LMSG.

---

## 107. HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA

**论文链接:** [http://arxiv.org/abs/2511.01463v1](http://arxiv.org/abs/2511.01463v1)

**作者:** Lei Hu, Yongjing Ye, Shihong Xia

**发布时间:** 2025-11-03

**备注:** 10 pages, 5figures. The Thirty-Ninth Annual Conference on Neural  Information Processing Systems

### GPT解析

### 总结

这项研究提出了HMVLM模型，基于MoE LoRA策略的统一框架，用于解决人类运动与语言模型整合中的模态差异和灾难性遗忘问题，同时改进姿态表示方法。

### 背景

指令调优数据的扩展使基础语言模型能够表现出改进的指令遵循能力和在多样化下游任务上的卓越性能。语义丰富的3D人体运动正逐渐与这些基础模型集成，以增强多模态理解和跨模态生成能力。然而，人体运动和文本之间的模态差异引发了关于这种整合过程中灾难性遗忘的未解决问题。此外，开发能够在异构下游任务中保持泛化能力的自回归兼容姿态表示仍然是一个关键的技术障碍。

### 目的

解决人体运动与文本之间的模态差异导致的灾难性遗忘问题，以及开发能够在多样化下游任务中保持泛化能力的自回归兼容姿态表示。

### 方法

提出HMVLM（人类运动-视觉-语言模型），这是一个基于专家混合低秩适应（MoE LoRA）策略的统一框架。该框架利用门控网络根据输入提示动态分配LoRA专家权重，实现多任务的同步微调。为减轻指令调优过程中的灾难性遗忘，引入了一种新型零专家，用于保留预训练参数以处理一般语言任务。对于姿态表示，通过将人体划分为不同的关节组，实现了身体部位特定的标记化，提高了表示的空间分辨率。

### 主要发现

实验表明，该方法有效减轻了指令调优过程中的知识遗忘，并在多样化的人体运动下游任务上取得了卓越的性能。

### 结论

HMVLM模型通过MoE LoRA策略和零专家机制，成功解决了人体运动与语言模型整合中的灾难性遗忘问题，同时通过身体部位特定的标记化改进了姿态表示，为多模态理解和生成提供了有效解决方案。

### 翻译

指令调优数据的扩展使基础语言模型能够在多样化的下游任务上表现出改进的指令遵循能力和卓越性能。语义丰富的3D人体运动正逐渐与这些基础模型集成，以增强多模态理解和跨模态生成能力。然而，人体运动和文本之间的模态差异引发了关于这种整合过程中灾难性遗忘的未解决问题。此外，开发能够在异构下游任务中保持泛化能力的自回归兼容姿态表示仍然是一个关键的技术障碍。为解决这些问题，我们提出了HMVLM（人类运动-视觉-语言模型），这是一个基于专家混合低秩适应（MoE LoRA）策略的统一框架。该框架利用门控网络根据输入提示动态分配LoRA专家权重，实现多任务的同步微调。为减轻指令调优过程中的灾难性遗忘，我们引入了一种新型零专家，用于保留预训练参数以处理一般语言任务。对于姿态表示，我们通过将人体划分为不同的关节组，实现了身体部位特定的标记化，提高了表示的空间分辨率。实验表明，我们的方法有效减轻了指令调优过程中的知识遗忘，并在多样化的人体运动下游任务上取得了卓越的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决两个关键问题：1) 当基础语言模型集成人类运动模态时出现的灾难性遗忘问题，即模型在训练过程中遗忘原有的世界知识和语言能力；2) 如何开发能保留跨任务泛化能力的自回归兼容姿态表示问题。这些问题在现实中很重要，因为随着大型基础模型在多模态理解中的应用日益广泛，将人类运动（包含丰富语义和情感）集成到这些模型中变得至关重要，而遗忘原有知识会导致模型变成对话能力有限的任务特定系统，姿态表示的局限性则会限制虚拟现实、具身智能等多个领域的应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到监督指令微调会过度关注新标记导致遗忘原有知识，因此借鉴了MoE和LoRA技术设计MoE LoRA框架，并引入零专家来保留预训练参数。针对姿态表示问题，作者受图像处理中基于块标记化的启发，将人体划分为不同肢体部分分别编码。作者借鉴了多项现有工作：MoE架构和LoRA技术用于多任务微调，基于块的图像编码用于空间建模，VQ-VAE架构用于离散化运动序列，以及Transformer架构用于自回归生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想包括：1) 使用MoE LoRA框架通过门控网络动态分配专家权重实现多任务微调；2) 引入零专家保留预训练参数减轻灾难性遗忘；3) 采用基于身体部位的标记化提高表示空间分辨率。整体流程：首先处理输入指令和提示，通过门控网络分配专家权重；然后对模态特定输入进行投影对齐；接着根据权重动态组合LoRA专家；同时使用身体部位标记器将姿态和运动离散化为标记；最后基础模型和加权专家共同生成输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) MoE LoRA框架引入零专家机制保留基础模型知识；2) 基于身体部位的标记化方法提高姿态表示的空间分辨率；3) 统一框架支持多种人体相关下游任务。相比之前工作不同：与MotionGPT等模型相比，HMVLM通过MoE LoRA和零专家更好保留了对话能力；与传统运动标记化相比，同时考虑空间信息提高表示精细度；与单任务模型相比，作为统一多任务框架在多任务场景下仍保持良好性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了HMVLM框架，通过MoE LoRA和基于身体部位的标记化方法，有效解决了人类运动与语言模型集成中的灾难性遗忘和表示问题，同时支持多种人体相关任务的高效执行。'}


### 论文摘要

The expansion of instruction-tuning data has enabled foundation language models to exhibit improved instruction adherence and superior performance across diverse downstream tasks. Semantically-rich 3D human motion is being progressively integrated with these foundation models to enhance multimodal understanding and cross-modal generation capabilities. However, the modality gap between human motion and text raises unresolved concerns about catastrophic forgetting during this integration. In addition, developing autoregressive-compatible pose representations that preserve generalizability across heterogeneous downstream tasks remains a critical technical barrier. To address these issues, we propose the Human Motion-Vision-Language Model (HMVLM), a unified framework based on the Mixture of Expert Low-Rank Adaption(MoE LoRA) strategy. The framework leverages the gating network to dynamically allocate LoRA expert weights based on the input prompt, enabling synchronized fine-tuning of multiple tasks. To mitigate catastrophic forgetting during instruction-tuning, we introduce a novel zero expert that preserves the pre-trained parameters for general linguistic tasks. For pose representation, we implement body-part-specific tokenization by partitioning the human body into different joint groups, enhancing the spatial resolution of the representation. Experiments show that our method effectively alleviates knowledge forgetting during instruction-tuning and achieves remarkable performance across diverse human motion downstream tasks.

---

## 108. From Passive to Proactive: A Multi-Agent System with Dynamic Task Orchestration for Intelligent Medical Pre-Consultation

**论文链接:** [http://arxiv.org/abs/2511.01445v1](http://arxiv.org/abs/2511.01445v1)

**作者:** ChengZhang Yu, YingRu He, Hongyan Cheng, nuo Cheng, Zhixing Liu, Dongxu Mu, Zhangrui Shen, Zhanpeng Jin

**发布时间:** 2025-11-03

**备注:** 14pages, 7 figures, 7 tables

### GPT解析

### 总结

该研究介绍了一种分层多智能体框架，将被动医疗AI系统转变为主动询问智能体，通过自主任务编排优化预咨询流程，在提高效率的同时保证了临床质量。

### 背景

全球医疗系统面临患者数量增加和咨询时间有限的挑战，许多国家初级保健咨询平均不足5分钟。现有预咨询流程受限于被动交互模式和AI系统的上下文管理挑战。

### 目的

引入分层多智能体框架，将被动医疗AI系统转变为主动询问智能体，通过自主任务编排提升预咨询效率和质量。

### 方法

开发八智能体架构，具有集中控制机制，将预咨询分解为四个主要任务（分诊、现病史采集、既往史采集、主诉生成）及13个子任务。在1,372个中国医疗平台电子健康记录上评估，使用GPT-OSS 20B、Qwen3-8B、Phi4-14B等基础模型。

### 主要发现

框架在初级科室分诊准确率达87.0%，二级科室分类达80.5%；智能体驱动调度任务完成率98.2%，高于顺序处理的93.1%；临床质量评分平均为：主诉4.56分、现病史4.48分、既往史4.69分（5分制）；T2在12.7轮内完成，T3在16.9轮内完成。

### 结论

模型无关架构在不同基础模型上保持高性能，通过本地部署保护数据隐私，展示了自主AI系统增强临床预咨询效率和质量的可能性。

### 翻译

全球医疗系统正面临患者数量增加和咨询时间有限的严峻挑战，在许多国家，初级保健咨询的平均时间不足5分钟。虽然涵盖分诊和结构化病史采集的预咨询流程提供了潜在解决方案，但它们仍受限于现有AI系统中的被动交互模式和上下文管理挑战。本研究引入了一种分层多智能体框架，通过自主任务编排将被动的医疗AI系统转变为主动的询问智能体。我们开发了一个具有集中控制机制的八智能体架构，将预咨询分解为四个主要任务：分诊、现病史采集、既往史采集和主诉生成，其中T1-T3进一步细分为13个特定领域的子任务。在中国医疗平台的1,372个经过验证的电子健康记录上，使用多个基础模型（GPT-OSS 20B、Qwen3-8B、Phi4-14B）评估，该框架在初级科室分诊上达到87.0%的准确率，在二级科室分类上达到80.5%的准确率，使用智能体驱动的调度任务完成率达到98.2%，而顺序处理为93.1%。18名医师的临床质量评分平均为：主诉4.56分、现病史4.48分、既往史4.69分（5分制），T2在12.7轮内完成，T3在16.9轮内完成。与模型无关的架构在不同基础模型上保持了高性能，同时通过本地部署保护数据隐私，展示了自主AI系统增强临床环境中预咨询效率和质量的可能性。


### 论文摘要

Global healthcare systems face critical challenges from increasing patient volumes and limited consultation times, with primary care visits averaging under 5 minutes in many countries. While pre-consultation processes encompassing triage and structured history-taking offer potential solutions, they remain limited by passive interaction paradigms and context management challenges in existing AI systems. This study introduces a hierarchical multi-agent framework that transforms passive medical AI systems into proactive inquiry agents through autonomous task orchestration. We developed an eight-agent architecture with centralized control mechanisms that decomposes pre-consultation into four primary tasks: Triage ($T_1$), History of Present Illness collection ($T_2$), Past History collection ($T_3$), and Chief Complaint generation ($T_4$), with $T_1$--$T_3$ further divided into 13 domain-specific subtasks. Evaluated on 1,372 validated electronic health records from a Chinese medical platform across multiple foundation models (GPT-OSS 20B, Qwen3-8B, Phi4-14B), the framework achieved 87.0% accuracy for primary department triage and 80.5% for secondary department classification, with task completion rates reaching 98.2% using agent-driven scheduling versus 93.1% with sequential processing. Clinical quality scores from 18 physicians averaged 4.56 for Chief Complaints, 4.48 for History of Present Illness, and 4.69 for Past History on a 5-point scale, with consultations completed within 12.7 rounds for $T_2$ and 16.9 rounds for $T_3$. The model-agnostic architecture maintained high performance across different foundation models while preserving data privacy through local deployment, demonstrating the potential for autonomous AI systems to enhance pre-consultation efficiency and quality in clinical settings.

---

## 109. Towards General Auditory Intelligence: Large Multimodal Models for Machine Listening and Speaking

**论文链接:** [http://arxiv.org/abs/2511.01299v1](http://arxiv.org/abs/2511.01299v1)

**作者:** Siyin Wang, Zengrui Jin, Changli Tang, Qiujia Li, Bo Li, Chen Chen, Yuchen Hu, Wenyi Yu, Yixuan Li, Jimin Zhuang, Yudong Yang, Mingqiu Wang, Michael Han, Yifan Ding, Junwen Bai, Tom Ouyang, Shuo-yiin Chang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Guangzhi Sun, Zhehuai Chen, Ji Wu, Bowen Zhou, Yuxuan Wang, Tara Sainath, Yonghui Wu, Chao Zhang

**发布时间:** 2025-11-03

**备注:** 22 pages, 11 figures

### GPT解析

### 总结

这篇论文是一篇综述，探讨了在大语言模型和通用人工智能时代，计算机听觉如何超越传统范式，充分利用基础模型的能力，朝着更全面的理解、更自然的生成和更类人的交互方向发展。

### 背景

在大语言模型和通用人工智能时代，计算机听觉需要发展以适应新的技术环境。音频作为一种包含丰富语义、情感和上下文线索的模式，在实现自然化和具身机器智能方面发挥着关键作用。

### 目的

这篇综述旨在全面回顾近期将音频整合到LLMs中的进展，分析LLMs如何重塑音频感知和推理能力，探索音频和视觉模式的融合如何增强情境感知和跨模态推理，并确定构建音频原生AGI系统的关键挑战和未来方向。

### 方法

论文采用综述方法，分析近期在音频与LLMs整合方面的研究进展，特别关注四个关键领域：音频理解、音频生成、基于语音的交互和音频-视觉理解。论文分析了LLMs如何改变音频感知和推理，探索了多模态智能的边界。

### 主要发现

LLMs使系统能够在更深层次的语义水平上理解声音，生成富有表现力的音频输出，进行类人的口语交互，并且音频和视觉模式的融合增强了情境感知和跨模态推理，推动了多模态智能的边界。

### 结论

这篇综述不仅综合了现有研究，还确定了构建能够像人类一样通过声音感知、理解和交互的音频原生AGI系统的关键挑战和未来方向。

### 翻译

在大语言模型和通用人工智能时代，计算机听觉必须超越传统范式，充分利用基础模型的能力，朝着更全面的理解、更自然的生成和更类人的交互方向发展。音频作为一种富含语义、情感和上下文线索的模式，在实现自然化和具身机器智能方面发挥着至关重要的作用。这篇综述全面回顾了近期将音频整合到LLMs中的进展，重点关注四个关键领域：音频理解、音频生成、基于语音的交互和音频-视觉理解。我们分析了LLMs如何重塑音频感知和推理，使系统能够在更深层次的语义水平上理解声音，生成富有表现力的音频输出，并进行类人的口语交互。此外，我们还探索了音频和视觉模式的融合如何增强情境感知和跨模态推理，推动多模态智能的边界。这篇综述不仅综合了现有研究，还确定了构建能够像人类一样自然地通过声音感知、理解和交互的音频原生AGI系统的关键挑战和未来方向。


### 论文摘要

In the era of large language models (LLMs) and artificial general intelligence (AGI), computer audition must evolve beyond traditional paradigms to fully leverage the capabilities of foundation models, towards more comprehensive understanding, more natural generation and more human-like interaction. Audio, as a modality rich in semantic, emotional, and contextual cues, plays a vital role in achieving naturalistic and embodied machine intelligence. This survey provides a comprehensive review of recent progress in integrating audio into LLMs, with a focus on four key areas: audio comprehension, audio generation, speech-based interaction, and audio-visual understanding. We analyze how LLMs are reshaping audio perception and reasoning, enabling systems to understand sound at a deeper semantic level, generate expressive audio outputs, and engage in human-like spoken interaction. Furthermore, we explore how the fusion of audio and visual modalities enhances situational awareness and cross-modal reasoning, pushing the boundaries of multimodal intelligence. This survey not only synthesizes existing research but also identifies critical challenges and future directions for building audio-native AGI systems capable of perceiving, understanding, and interacting through sound as naturally as humans do.

---

## 110. Optimal Attention Temperature Enhances In-Context Learning under Distribution Shift

**论文链接:** [http://arxiv.org/abs/2511.01292v1](http://arxiv.org/abs/2511.01292v1)

**作者:** Samet Demir, Zafer Dogan

**发布时间:** 2025-11-03

**备注:** 26 pages, 6 figures

### GPT解析

### 总结

本研究探讨了在分布偏移条件下注意力温度对预训练Transformer模型上下文学习性能的影响，首次提供了理论和实证研究，证明了最优注意力温度可以最小化分布偏移导致的误差，提高了ICL的鲁棒性。

### 背景

预训练Transformer模型在上下文学习方面表现出色，但当预训练和测试数据间存在分布偏移时，ICL性能会急剧下降，这种情况在实际部署中越来越常见。虽然调整注意力温度可以提高Transformer性能，但在分布偏移条件下注意力温度对ICL的影响尚未被探索。

### 目的

提供关于在分布偏移条件下ICL注意力温度的首个理论和实证研究，探索最优注意力温度对提高ICL鲁棒性的作用。

### 方法

使用'线性化softmax'框架推导闭式泛化误差表达式，证明输入协方差变化和标签噪声对ICL的影响，并通过线性回归任务的模拟和GPT-2、LLaMA2-7B在问答基准上的大规模实验验证理论预测。

### 主要发现

输入协方差的变化或标签噪声会显著损害ICL性能；存在最优注意力温度可以最小化分布偏移条件下的误差；注意力温度是提高预训练Transformer中ICL鲁棒性的有效机制。

### 结论

注意力温度是提高预训练Transformer中ICL鲁棒性的原则性和强大机制，研究推进了理论理解，并为实践中选择注意力温度提供了可行的指导。

### 翻译

预训练Transformer在上下文学习方面表现出色，仅从少量例子中就能推断新任务。然而，当预训练和测试数据之间存在分布偏移时，它们的ICL性能可能会急剧下降，这种情况在实际部署中越来越常见。虽然最近的实证研究表明调整softmax中的注意力温度可以提高Transformer性能，但在分布偏移条件下，注意力温度在ICL中的作用仍未被探索。本文首次提供了关于在分布偏移条件下ICL注意力温度的理论和实证研究。使用简化但富有表现力的'线性化softmax'框架，我们推导出闭式泛化误差表达式，并证明输入协方差的变化或标签噪声会显著损害ICL，但存在最优注意力温度可以最小化这种误差。然后，我们通过线性回归任务的广泛模拟和GPT-2及LLaMA2-7B在问答基准上的大规模实验验证了我们的预测。我们的研究结果表明，注意力温度是提高预训练Transformer中ICL鲁棒性的原则性和强大机制，推进了理论理解，并为实践中选择注意力温度提供了可行的指导。


### 论文摘要

Pretrained Transformers excel at in-context learning (ICL), inferring new tasks from only a handful of examples. Yet, their ICL performance can degrade sharply under distribution shift between pretraining and test data, a regime increasingly common in real-world deployments. While recent empirical work hints that adjusting the attention temperature in the softmax can enhance Transformer performance, the attention temperature's role in ICL under distribution shift remains unexplored. This paper provides the first theoretical and empirical study of attention temperature for ICL under distribution shift. Using a simplified but expressive "linearized softmax" framework, we derive closed-form generalization error expressions and prove that shifts in input covariance or label noise substantially impair ICL, but that an optimal attention temperature exists which minimizes this error. We then validate our predictions through extensive simulations on linear regression tasks and large-scale experiments with GPT-2 and LLaMA2-7B on question-answering benchmarks. Our results establish attention temperature as a principled and powerful mechanism for improving the robustness of ICL in pretrained Transformers, advancing theoretical understanding and providing actionable guidance for selecting attention temperature in practice.

---

## 111. Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions

**论文链接:** [http://arxiv.org/abs/2511.01284v1](http://arxiv.org/abs/2511.01284v1)

**作者:** Karma Phuntsho, Abdullah, Kyungmi Lee, Ickjai Lee, Euijoon Ahn

**发布时间:** 2025-11-03

### GPT解析

### 总结

这是一篇关于Foundation models在医学图像分析中应用的综述文章，评估了将FMs适应医学成像特定需求的多种策略，包括监督微调、领域特定预训练等方法，并指出了新兴研究方向如持续学习、隐私保护方法等，为开发适应性、可信且临床整合的FMs提供了路线图。

### 背景

Foundation models已成为医学图像分析中的一种变革性范式，能够为广泛的临床任务和成像模式提供可泛化、任务无关的解决方案。它们从大规模数据中学习可迁移表示的能力，有望解决传统特定任务模型的局限性。然而，将FMs适应真实临床实践仍面临关键挑战，包括域偏移、高质量标注数据有限、计算需求大以及严格的隐私要求。

### 目的

本文对将FMs适应医学成像特定需求的策略进行了全面评估。

### 方法

检查了监督微调、领域特定预训练、参数高效微调、自监督学习、混合方法以及多模态或跨模态框架等适应FMs的方法，并对每种方法评估了报告的性能提升、临床适用性和局限性，同时确定了先前综述经常忽视的权衡和未解决的挑战。

### 主要发现

新兴研究方向包括：实现动态部署的持续学习；保护敏感数据的联邦和隐私保护方法；提高数据效率的混合自监督学习；结合合成生成与人工验证循环的数据中心管道；以及评估在真实临床变异性下的鲁棒泛化的系统基准测试。

### 结论

通过概述这些策略和相关研究差距，本综述为开发适应性、可信且临床整合的FMs提供了路线图，使其能够满足真实医学成像的需求。

### 翻译

基础模型已成为医学图像分析中的一种变革性范式，能够为广泛的临床任务和成像模式提供可泛化、任务无关的解决方案。它们从大规模数据中学习可迁移表示的能力，有望解决传统特定任务模型的局限性。然而，将FMs适应真实临床实践仍面临关键挑战，包括域偏移、高质量标注数据有限、计算需求大以及严格的隐私要求。本文对将FMs适应医学成像特定需求的策略进行了全面评估。我们检查了监督微调、领域特定预训练、参数高效微调、自监督学习、混合方法以及多模态或跨模态框架等方法。对于每种方法，我们评估了报告的性能提升、临床适用性和局限性，同时确定了先前综述经常忽视的权衡和未解决的挑战。除了这些已建立的技术外，我们还强调了旨在解决当前差距的新兴方向，包括实现动态部署的持续学习、保护敏感数据的联邦和隐私保护方法、提高数据效率的混合自监督学习、结合合成生成与人工验证循环的数据中心管道，以及评估在真实临床变异性下的鲁棒泛化的系统基准测试。通过概述这些策略和相关研究差距，本综述为开发适应性、可信且临床整合的FMs提供了路线图，使其能够满足真实医学成像的需求。


### 论文摘要

Foundation models (FMs) have emerged as a transformative paradigm in medical image analysis, offering the potential to provide generalizable, task-agnostic solutions across a wide range of clinical tasks and imaging modalities. Their capacity to learn transferable representations from large-scale data has the potential to address the limitations of conventional task-specific models. However, adaptation of FMs to real-world clinical practice remains constrained by key challenges, including domain shifts, limited availability of high-quality annotated data, substantial computational demands, and strict privacy requirements. This review presents a comprehensive assessment of strategies for adapting FMs to the specific demands of medical imaging. We examine approaches such as supervised fine-tuning, domain-specific pretraining, parameter-efficient fine-tuning, self-supervised learning, hybrid methods, and multimodal or cross-modal frameworks. For each, we evaluate reported performance gains, clinical applicability, and limitations, while identifying trade-offs and unresolved challenges that prior reviews have often overlooked. Beyond these established techniques, we also highlight emerging directions aimed at addressing current gaps. These include continual learning to enable dynamic deployment, federated and privacy-preserving approaches to safeguard sensitive data, hybrid self-supervised learning to enhance data efficiency, data-centric pipelines that combine synthetic generation with human-in-the-loop validation, and systematic benchmarking to assess robust generalization under real-world clinical variability. By outlining these strategies and associated research gaps, this review provides a roadmap for developing adaptive, trustworthy, and clinically integrated FMs capable of meeting the demands of real-world medical imaging.

---

## 112. Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play

**论文链接:** [http://arxiv.org/abs/2511.01261v1](http://arxiv.org/abs/2511.01261v1)

**作者:** Jiatong Shi, Jionghao Han, Yichen Lu, Santiago Pascual, Pengfei Wu, Chenye Cui, Shinji Watanabe, Chao Weng, Cong Zhou

**发布时间:** 2025-11-03

**备注:** 67 pages

### GPT解析

### 总结

Speech-DRAME是一个统一的框架，用于评估语音角色扮演系统，包含评估基准、微调评估模型和语音角色扮演基准三个层面贡献，通过原型评估和真实性评估两种互补策略提供全面的评估基础。

### 背景

角色扮演已成为生成模型的关键测试平台，从纯文本对话扩展到多模态交互。将角色扮演扩展到语音可以捕捉韵律、情感和表达方式，但也带来了新的评估挑战。

### 目的

提出Speech-DRAME框架，解决当前语音角色扮演评估中存在的问题，为语音角色扮演提供全面、可复制的评估基础。

### 方法

Speech-DRAME框架在三个层面做出贡献：(i)Speech-DRAME-EvalBench评估基准，包含双语人工注释数据和用于训练测试语音评估模型的协议，(ii)DRAME-Eval微调评估模型，显著优于零样本和少样本音频大语言模型，(iii)Speech-DRAME-RoleBench语音角色扮演基准，利用DRAME-Eval作为自动评估者。同时区分了原型评估和真实性评估两种互补的评估策略。

### 主要发现

与零样本音频大语言模型评估者相比，DRAME-Eval与人类评分的一致性更强，原型评估中的相关系数从0.480提高到0.629，真实性评估中从0.390提高到0.625。

### 结论

通过整合透明的基准资源、建模方法和系统级评估，Speech-DRAME为评估语音角色扮演提供了首个全面、可复制的基础。

### 翻译

角色扮演已成为生成模型的关键测试平台，从纯文本对话扩展到多模态交互。将角色扮演扩展到语音可以捕捉韵律、情感和表达方式，但也带来了新的评估挑战。当前的评估流程通常使用音频大语言模型作为零样本评估者，这些模型会忽略副语言线索，将多个方面合并为粗略的分数，并依赖无法反映现实世界角色的合成语音参考。我们提出了Speech-DRAME，一个统一框架，在三个层面做出贡献：(i)Speech-DRAME-EvalBench，一个包含双语人工注释数据和用于训练测试语音评估模型的协议的评估基准，(ii)DRAME-Eval，一个微调的评估模型，显著优于零样本和少样本音频大语言模型，(iii)Speech-DRAME-RoleBench，一个利用DRAME-Eval作为自动评估者来比较语音基础模型的语音角色扮演基准。Speech-DRAME区分了两种互补的评估策略：原型评估，一种自上而下的方法，衡量对广泛角色原型的遵循程度；真实性评估，一种基于真实人类语音的自下而上方法，强调细微的角色质量。与零样本音频大语言模型评估者相比，DRAME-Eval与人类评分的一致性更强（原型评估中的相关系数从0.480提高到0.629，真实性评估中从0.390提高到0.625）。通过整合透明的基准资源、建模方法和系统级评估，Speech-DRAME为评估语音角色扮演提供了首个全面、可复制的基础。


### 论文摘要

Role-play has become a key testbed for generative models, expanding from text-only dialogue to multimodal interaction. Extending role-play to speech captures prosody, emotion, and delivery, but also poses new evaluation challenges. Current pipelines often use audio large language models (ALLMs) as zero-shot judges, which miss paralinguistic cues, collapse multiple aspects into coarse scores, and rely on synthetic speech references that fail to reflect real-world roles. We present Speech-DRAME, a unified framework that contributes at three levels: (i) Speech-DRAME-EvalBench, an evaluation benchmark with bilingual human-annotated data and protocols for training and testing speech evaluation models (SEMs), (ii) DRAME-Eval, a fine-tuned evaluation model, which substantially outperforms zero-shot and few-shot ALLMs, and (iii) Speech-DRAME-RoleBench, a speech role-play benchmark that leverages DRAME-Eval as an automatic judge to compare speech foundation models (SFMs). Speech-DRAME distinguishes between two complementary evaluation strategies: Archetype Evaluation, a top-down approach measuring adherence to broad role archetypes, and Realism Evaluation, a bottom-up approach grounded in real human speech that emphasizes nuanced role quality. Compared to zero-shot ALLM judges, DRAME-Eval achieves stronger agreement with human ratings (Pearson correlation from 0.480 to 0.629 in archetypes, and 0.390 to 0.625 in realism). By integrating transparent benchmark resources, modeling approaches, and system-level evaluation, Speech-DRAME provides the first comprehensive, reproducible foundation for assessing spoken role-play.

---

## 113. VesSAM: Efficient Multi-Prompting for Segmenting Complex Vessel

**论文链接:** [http://arxiv.org/abs/2511.00981v1](http://arxiv.org/abs/2511.00981v1)

**作者:** Suzhong Fu, Rui Sun, Xuan Ding, Jingqi Dong, Yiming Yang, Yao Zhu, Min Chang Jordan Ren, Delin Deng, Angelica Aviles-Rivero, Shuguang Cui, Zhen Li

**发布时间:** 2025-11-02

### GPT解析

### 总结

VesSAM是一个专门针对2D血管分割的强大高效框架，通过结合卷积适配器、多提示编码器和轻量级解码器，解决了血管分割中的挑战，在各种成像模态上表现优异，且在分布外数据上具有良好的泛化能力。

### 背景

准确的血管分割对临床应用（如疾病诊断和手术规划）至关重要，但由于血管具有细小、分支结构和低纹理对比度，血管分割仍然具有挑战性。基础模型如Segment Anything Model (SAM)在通用分割方面显示出前景，但在血管结构上表现不佳。

### 目的

开发一个专门针对2D血管分割的强大而高效的框架，以克服现有方法的局限性。

### 方法

VesSAM框架包含三个关键组件：(1)卷积适配器增强局部纹理特征，(2)多提示编码器通过分层交叉注意力融合解剖学提示（包括骨架、分叉点和线段中点），(3)轻量级掩码解码器减少锯齿伪影。此外，还引入了自动化流程生成结构化多提示标注，并整理了包含5种成像模态8个数据集的多样化基准数据集。

### 主要发现

VesSAM在Dice和IoU指标上比最先进的基于PEFT的SAM变体高出10%和13%，与完全微调的方法相比实现了具有竞争力的性能，且参数显著减少。VesSAM在分布外（OoD）设置中表现良好，在平均OoD Dice和IoU上优于所有基线。

### 结论

VesSAM是一个专门为血管分割设计的有效框架，通过结合创新架构和提示机制，显著提升了血管分割的性能和泛化能力。

### 翻译

准确的血管分割对临床应用（如疾病诊断和手术规划）至关重要，但由于血管细小、分支结构和低纹理对比度，仍然具有挑战性。虽然像Segment Anything Model (SAM)这样的基础模型在通用分割方面显示出前景，但在血管结构上表现不佳。在这项工作中，我们提出了VesSAM，一个专门针对2D血管分割的强大而高效的框架。VesSAM集成了(1)卷积适配器增强局部纹理特征，(2)多提示编码器通过分层交叉注意力融合解剖学提示，包括骨架、分叉点和线段中点，以及(3)轻量级掩码解码器减少锯齿伪影。我们还引入了自动化流程生成结构化多提示标注，并整理了一个包含5种成像模态8个数据集的多样化基准数据集。实验结果表明，VesSAM在Dice和IoU指标上持续比最先进的基于PEFT的SAM变体高出10%和13%，与完全微调的方法相比实现了具有竞争力的性能，且参数显著减少。VesSAM在分布外（OoD）设置中泛化良好，在平均OoD Dice和IoU上优于所有基线。


### 论文摘要

Accurate vessel segmentation is critical for clinical applications such as disease diagnosis and surgical planning, yet remains challenging due to thin, branching structures and low texture contrast. While foundation models like the Segment Anything Model (SAM) have shown promise in generic segmentation, they perform sub-optimally on vascular structures. In this work, we present VesSAM, a powerful and efficient framework tailored for 2D vessel segmentation. VesSAM integrates (1) a convolutional adapter to enhance local texture features, (2) a multi-prompt encoder that fuses anatomical prompts, including skeletons, bifurcation points, and segment midpoints, via hierarchical cross-attention, and (3) a lightweight mask decoder to reduce jagged artifacts. We also introduce an automated pipeline to generate structured multi-prompt annotations, and curate a diverse benchmark dataset spanning 8 datasets across 5 imaging modalities. Experimental results demonstrate that VesSAM consistently outperforms state-of-the-art PEFT-based SAM variants by over 10% Dice and 13% IoU, and achieves competitive performance compared to fully fine-tuned methods, with significantly fewer parameters. VesSAM also generalizes well to out-of-distribution (OoD) settings, outperforming all baselines in average OoD Dice and IoU.

---

## 114. Actial: Activate Spatial Reasoning Ability of Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.01618v1](http://arxiv.org/abs/2511.01618v1)

**作者:** Xiaoyu Zhan, Wenxuan Huang, Hao Sun, Xinyu Fu, Changfeng Ma, Shaosheng Cao, Bohan Jia, Shaohui Lin, Zhenfei Yin, Lei Bai, Wanli Ouyang, Yuanqi Li, Jie Guo, Yanwen Guo

**发布时间:** 2025-11-03

### GPT解析

### 总结

本研究通过引入视角学习任务和Viewpoint-100K数据集，采用两阶段微调策略和混合冷启动初始化方法，有效提升了多模态大语言模型在3D推理任务中的空间推理能力，为未来机器人、自主系统和3D场景理解等领域的发展提供了支持。

### 背景

多模态大语言模型在2D视觉理解方面取得了显著进展，引起了人们对将其应用于复杂3D推理任务的兴趣。然而，尚不清楚这些模型是否能有效捕捉稳健现实世界性能所需的详细空间信息，特别是跨视图一致性，这是准确3D推理的关键要求。

### 目的

引入视角学习任务，旨在评估和改进多模态大语言模型的空间推理能力。

### 方法

提出Viewpoint-100K数据集，包含10万个以对象为中心的图像对，具有多样化的视角和相应的问题-答案对；采用两阶段微调策略：首先通过监督微调将基础知识注入基础模型，然后通过强化学习增强泛化能力；引入混合冷启动初始化方法，同时学习视角表示并保持连贯的推理思维。

### 主要发现

实验结果表明，该方法显著激活了多模态大语言模型的空间推理能力，提高了在领域内和领域外推理任务上的性能。

### 结论

研究结果强调了在多模态大语言模型中开发基础空间技能的价值，支持机器人、自主系统和3D场景理解方面的未来进步。

### 翻译

多模态大语言模型的最新进展显著提高了二维视觉理解能力，促使人们对其在复杂三维推理任务中的应用产生兴趣。然而，这些模型是否能有效捕捉稳健现实世界性能所需的详细空间信息，特别是跨视图一致性，这一准确三维推理的关键要求，仍不清楚。考虑到这一问题，我们引入了视角学习，这是一个旨在评估和改进多模态大语言模型空间推理能力的任务。我们提出了Viewpoint-100K数据集，包含10万个以对象为中心的图像对，具有多样化的视角和相应的问题-答案对。我们的方法采用两阶段微调策略：首先，通过在Viewpoint-100K上进行监督微调将基础知识注入基础多模态大语言模型，在多个任务上取得显著改进；其次，通过在更广泛的问题集上使用组相对策略优化算法的强化学习来增强泛化能力。此外，我们引入了一种混合冷启动初始化方法，旨在同时学习视角表示并保持连贯的推理思维。实验结果表明，我们的方法显著激活了多模态大语言模型的空间推理能力，提高了在领域内和领域外推理任务上的性能。我们的研究结果强调了在多模态大语言模型中开发基础空间技能的价值，支持机器人、自主系统和三维场景理解方面的未来发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型(MLLMs)无法有效捕捉3D空间推理所需的详细空间信息，特别是跨视图一致性的问题。这个问题很重要，因为虽然MLLMs在2D视觉理解方面进步显著，但在需要准确3D推理的现实应用中表现不佳，限制了机器人在复杂环境中的导航、自主系统的空间感知以及3D场景理解等关键应用的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到当前MLLMs主要在强调2D连续性的视频数据上训练，导致它们难以理解3D空间一致性。他们从计算机视觉领域的相机校准和立体匹配方法中获得启发，设计了简化的视角学习任务，将复杂的3D问题分解为简单的多选题。作者借鉴了参考帧(FoR)概念、Group Relative Policy Optimization算法、冷启动初始化和思维链(CoT)等现有技术，但将其创新性地应用于激活MLLMs的空间推理能力上。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过专门的视角学习任务激活MLLMs的空间推理能力，采用两阶段微调策略：先注入基础知识，再增强泛化能力。整体流程包括：1)创建Viewpoint-100K数据集，包含10万个对象中心的图像对和问答对；2)第一阶段使用监督微调(SFT)和混合冷启动初始化(90%真实数据+10%伪思维链)注入基础知识；3)第二阶段使用强化学习GRPO算法在SAT数据集上增强泛化能力，鼓励模型生成自己的推理链并应用已学空间知识。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出视角学习任务评估和改进MLLMs空间推理能力；2)创建Viewpoint-100K数据集；3)采用两阶段微调策略；4)设计混合冷启动初始化方法；5)将复杂3D问题简化为多选题。相比之前工作，本文更关注空间推理的基础任务(如视角估计)而非高级推理；不依赖额外3D信息而是激活模型内在能力；不仅提高特定任务性能，还增强领域外泛化能力；强调基础的3D感知能力而非仅关注高层次的推理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过视角学习任务和两阶段微调策略，成功激活了多模态大语言模型的空间推理能力，显著提升了模型在视觉和空间推理任务中的性能，特别是在领域外任务中的泛化能力。'}


### 论文摘要

Recent advances in Multimodal Large Language Models (MLLMs) have significantly improved 2D visual understanding, prompting interest in their application to complex 3D reasoning tasks. However, it remains unclear whether these models can effectively capture the detailed spatial information required for robust real-world performance, especially cross-view consistency, a key requirement for accurate 3D reasoning. Considering this issue, we introduce Viewpoint Learning, a task designed to evaluate and improve the spatial reasoning capabilities of MLLMs. We present the Viewpoint-100K dataset, consisting of 100K object-centric image pairs with diverse viewpoints and corresponding question-answer pairs. Our approach employs a two-stage fine-tuning strategy: first, foundational knowledge is injected to the baseline MLLM via Supervised Fine-Tuning (SFT) on Viewpoint-100K, resulting in significant improvements across multiple tasks; second, generalization is enhanced through Reinforcement Learning using the Group Relative Policy Optimization (GRPO) algorithm on a broader set of questions. Additionally, we introduce a hybrid cold-start initialization method designed to simultaneously learn viewpoint representations and maintain coherent reasoning thinking. Experimental results show that our approach significantly activates the spatial reasoning ability of MLLM, improving performance on both in-domain and out-of-domain reasoning tasks. Our findings highlight the value of developing foundational spatial skills in MLLMs, supporting future progress in robotics, autonomous systems, and 3D scene understanding.

---

## 115. PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model

**论文链接:** [http://arxiv.org/abs/2511.01571v1](http://arxiv.org/abs/2511.01571v1)

**作者:** Wenqi Liang, Gan Sun, Yao He, Jiahua Dong, Suyan Dai, Ivan Laptev, Salman Khan, Yang Cong

**发布时间:** 2025-11-03

**备注:** 17pages,7 figures, 5 tabels

### GPT解析

### 总结

PixelVLA是一种新型Vision-Language-Action模型，支持像素级推理和多模态提示，通过两阶段自动注释流程生成Pixel-160K数据集，实验显示其比OpenVLA提高操作成功率10.1%-17.8%，同时仅需1.5%的预训练成本。

### 背景

Vision-Language-Action models (VLAs)是学习通用视觉运动控制策略的有力工具，但当前VLAs主要在大规模图像-文本-动作数据上训练，存在两个关键限制：难以进行像素级场景理解，以及严重依赖文本提示，降低了在现实世界环境中的灵活性。

### 目的

解决当前VLAs在像素级场景理解和现实世界应用灵活性方面的限制，引入支持像素级推理和多模态提示（文本和视觉输入）的VLA模型。

### 方法

基于新的视觉运动指令调整框架，集成多尺度像素感知编码器和视觉提示编码器，提出两阶段自动注释流程生成Pixel-160K数据集，该数据集具有从现有机器人数据派生的像素级注释。

### 主要发现

在三个标准VLA基准测试和两个VLA模型变体上的实验表明，PixelVLA比OpenVLA提高操作成功率10.1%-17.8%，同时仅需OpenVLA 1.5%的预训练成本。

### 结论

PixelVLA可以集成到现有VLAs中，在复杂环境中实现更准确、高效和多功能的机器人控制，数据集和代码将作为开源发布。

### 翻译

视觉-语言-动作模型（VLAs）正成为学习通用视觉运动控制策略的有力工具。然而，当前的VLAs主要在大规模的图像-文本-动作数据上训练，并在两个方面存在局限：（i）它们难以进行像素级场景理解，（ii）它们严重依赖文本提示，这降低了它们在现实世界环境中的灵活性。为应对这些挑战，我们引入了PixelVLA，这是第一个支持像素级推理和多模态提示（文本和视觉输入）的VLA模型。我们的方法建立在一种新的视觉运动指令调整框架上，该框架集成了多尺度像素感知编码器和视觉提示编码器。为了有效训练PixelVLA，我们进一步提出了一种两阶段自动注释流程，生成Pixel-160K，这是一个从现有机器人数据派生的大型像素级注释数据集。在三个标准VLA基准测试和两个VLA模型变体上的实验表明，PixelVLA比OpenVLA提高操作成功率10.1%-17.8%，同时仅需其1.5%的预训练成本。这些结果表明，PixelVLA可以集成到现有VLAs中，在复杂环境中实现更准确、高效和多功能的机器人控制。数据集和代码将作为开源发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有视觉-语言-动作模型(VLAs)的两个关键限制：1)缺乏像素级场景理解能力，2)过度依赖文本提示而缺乏对视觉提示的灵活处理。这些问题在现实中很重要，因为像素级理解对于机器人在复杂环境中进行精确操作至关重要，而多模态提示能力则能增强人机交互的灵活性和适应性，使机器人能更好地应对现实世界中的多样化任务和场景变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有VLAs的局限性，然后借鉴了视觉指令调优在视觉语言模型中的成功经验。具体设计上，作者参考了OpenVLA和π0等VLA模型的基本架构，利用SAM模型进行图像分割，并采用LoRA技术进行模型微调。通过引入多尺度像素感知编码器、视觉提示编码器和连续动作解码器这三个核心组件，以及设计两阶段自动注释管道来生成高质量数据集，作者构建了PixelVLA模型，实现了像素级理解和多模态提示能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过像素级理解和多模态提示能力增强VLAs在复杂环境中的空间感知和操作精度。整体实现流程分为三部分：1)架构设计，包含视觉编码器、视觉提示编码器、多尺度像素感知编码器、LLM骨干和连续动作解码器；2)数据生成，通过两阶段自动注释管道(夹爪感知区域提案阶段和多模态对象分割阶段)创建Pixel-160K数据集；3)训练流程，包括连续动作训练阶段和像素级理解增强阶段，后者使用LoRA适配进行微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多尺度像素感知编码器，实现像素级场景理解；2)视觉提示编码器，支持点、线、区域和掩码等多种视觉提示；3)两阶段自动注释管道，生成高质量像素级标注数据集Pixel-160K；4)连续动作解码器，直接预测连续动作表示。相比之前工作，PixelVLA突破了传统VLAs仅处理图像级别信息和依赖文本提示的限制，实现了更精细的空间理解和更灵活的人机交互方式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PixelVLA通过引入像素级理解和多模态提示能力，显著提升了机器人在复杂环境中的操作精度和泛化能力，同时仅需1.5%的预训练成本就能实现10.1%~28.7%的操作成功率提升。'}


### 论文摘要

Vision-Language-Action models (VLAs) are emerging as powerful tools for learning generalizable visuomotor control policies. However, current VLAs are mostly trained on large-scale image-text-action data and remain limited in two key ways: (i) they struggle with pixel-level scene understanding, and (ii) they rely heavily on textual prompts, which reduces their flexibility in real-world settings. To address these challenges, we introduce PixelVLA, the first VLA model designed to support both pixel-level reasoning and multimodal prompting with text and visual inputs. Our approach is built on a new visuomotor instruction tuning framework that integrates a multiscale pixel-aware encoder with a visual prompting encoder. To train PixelVLA effectively, we further propose a two-stage automated annotation pipeline that generates Pixel-160K, a large-scale dataset with pixel-level annotations derived from existing robot data. Experiments on three standard VLA benchmarks and two VLA model variants show that PixelVLA improves manipulation success rates by 10.1%-17.8% over OpenVLA, while requiring only 1.5% of its pretraining cost. These results demonstrate that PixelVLA can be integrated into existing VLAs to enable more accurate, efficient, and versatile robot control in complex environments. The dataset and code will be released as open source.

---

## 116. Grounding Surgical Action Triplets with Instrument Instance Segmentation: A Dataset and Target-Aware Fusion Approach

**论文链接:** [http://arxiv.org/abs/2511.00643v1](http://arxiv.org/abs/2511.00643v1)

**作者:** Oluwatosin Alabi, Meng Wei, Charlie Budd, Tom Vercauteren, Miaojing Shi

**发布时间:** 2025-11-01

### GPT解析

### 总结

本研究提出了一种新的手术动作三元组空间定位方法，通过器械实例分割来实现空间定位的<器械、动词、目标>输出，并构建了相应的大规模数据集和评估基准。

### 背景

现有的手术动作三元组识别方法仅限于帧级分类学习，无法可靠地将动作与特定器械实例关联。之前的空间定位方法主要依赖类激活图，缺乏精确性和鲁棒性，无法满足详细的器械-组织交互分析需求。

### 目的

解决现有方法在手术动作三元组空间定位上的局限性，提出一种能够将动作与特定器械实例空间关联的统一框架。

### 方法

提出了'三元组分割'任务，构建了CholecTriplet-Seg大规模数据集（包含30,000+标注帧），并设计了TargetFusionNet架构，通过目标感知融合机制扩展Mask2Former，融合弱解剖先验与器械实例查询以提高解剖目标预测准确性。

### 主要发现

TargetFusionNet在识别、检测和三元组分割指标上均优于现有基线，证明强实例监督与弱目标先验相结合能显著提高手术动作理解的准确性和鲁棒性。

### 结论

三元组分割为手术动作三元组的空间定位建立了统一框架，所提出的基准和架构为更可解释的手术场景理解铺平了道路。

### 翻译

理解手术器械-组织交互不仅需要识别哪个器械在哪个解剖目标上执行什么动作，还需要将这些交互在手术场景中空间定位。现有的手术动作三元组识别方法仅限于从帧级分类学习，无法可靠地将动作与特定的器械实例关联起来。之前的空间定位尝试主要依赖于类激活图，但这些方法缺乏详细的器械-组织交互分析所需的精确性和鲁棒性。为解决这一差距，我们提出了通过器械实例分割来定位手术动作三元组，简称为三元组分割，这是一个新的统一任务，可以产生空间定位的<器械、动词、目标>输出。我们首先介绍了CholecTriplet-Seg数据集，这是一个包含超过30,000个标注帧的大规模数据集，将器械实例掩码与动作动词和解剖目标标注相关联，并建立了首个用于强监督、实例级三元组定位和评估的基准。为了学习三元组分割，我们提出了TargetFusionNet，这是一种新颖的架构，它通过目标感知融合机制扩展了Mask2Former，以解决通过将弱解剖先验与器械实例查询融合来准确预测解剖目标的挑战。在识别、检测和三元组分割指标上的评估表明，TargetFusionNet始终优于现有基线，证明强实例监督与弱目标先验相结合显著提高了手术动作理解的准确性和鲁棒性。三元组分割为空间定位手术动作三元组建立了统一框架。所提出的基准和架构为更可解释的手术场景理解铺平了道路。


### 论文摘要

Understanding surgical instrument-tissue interactions requires not only identifying which instrument performs which action on which anatomical target, but also grounding these interactions spatially within the surgical scene. Existing surgical action triplet recognition methods are limited to learning from frame-level classification, failing to reliably link actions to specific instrument instances.Previous attempts at spatial grounding have primarily relied on class activation maps, which lack the precision and robustness required for detailed instrument-tissue interaction analysis.To address this gap, we propose grounding surgical action triplets with instrument instance segmentation, or triplet segmentation for short, a new unified task which produces spatially grounded <instrument, verb, target> outputs.We start by presenting CholecTriplet-Seg, a large-scale dataset containing over 30,000 annotated frames, linking instrument instance masks with action verb and anatomical target annotations, and establishing the first benchmark for strongly supervised, instance-level triplet grounding and evaluation.To learn triplet segmentation, we propose TargetFusionNet, a novel architecture that extends Mask2Former with a target-aware fusion mechanism to address the challenge of accurate anatomical target prediction by fusing weak anatomy priors with instrument instance queries.Evaluated across recognition, detection, and triplet segmentation metrics, TargetFusionNet consistently improves performance over existing baselines, demonstrating that strong instance supervision combined with weak target priors significantly enhances the accuracy and robustness of surgical action understanding.Triplet segmentation establishes a unified framework for spatially grounding surgical action triplets. The proposed benchmark and architecture pave the way for more interpretable, surgical scene understanding.

---

