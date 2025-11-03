# 今日论文推荐 - 2025-11-03

共 59 篇论文

---

## 1. 论文ID: 2510.27629v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.27629v1.json'

---

## 2. Validity Is What You Need

**论文链接:** [http://arxiv.org/abs/2510.27628v1](http://arxiv.org/abs/2510.27628v1)

**作者:** Sebastian Benthall, Andrew Clark

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文讨论了Agentic AI的定义、特性和验证方法，提出Agentic AI是一种软件交付机制，类似于SaaS，使应用程序能够在复杂企业环境中自主工作。研究指出Agentic AI系统主要是应用程序而非基础模型，其成功依赖于用户验证，并强调在良好验证措施下，可用更简单模型替代复杂基础模型。

### 背景

AI代理在计算机科学领域早已被讨论和研究，但当前的Agentic AI系统是新的发展。大型语言模型(LLMs)等基础模型的进步推动了Agentic AI的发展。

### 目的

考虑其他Agentic AI的定义并提出一个新的现实主义定义。

### 方法

将Agentic AI定义为软件交付机制，类似于软件即服务(SaaS)，使应用程序能够在复杂的企业环境中自主工作。

### 主要发现

Agentic AI系统主要是应用程序而非基础模型，其成功依赖于最终用户和主要利益相关者的验证；验证应用程序的工具与技术评估基础模型的技术不同；在有良好验证措施的情况下，基础模型通常可以用更简单、更快、更可解释的模型替代。

### 结论

对于Agentic AI，有效性是关键需求，LLMs是可能实现这一目标的一种选择。

### 翻译

虽然AI代理在计算机科学领域早已被讨论和研究，但今天的Agentic AI系统是全新的。我们考虑了其他Agentic AI的定义，并提出了一个新的现实主义定义。Agentic AI是一种软件交付机制，类似于软件即服务(SaaS)，它使应用程序能够在复杂的企业环境中自主工作。大型语言模型(LLMs)等基础模型的最新进展推动了Agentic AI的发展。然而，我们注意到Agentic AI系统主要是应用程序，而非基础模型，因此其成功依赖于最终用户和主要利益相关者的验证。主要用户验证其应用程序所需的工具与技术用于评估基础模型的技术有很大不同。讽刺的是，在有良好验证措施的情况下，在许多情况下基础模型可以用更简单、更快、更可解释的模型来替代，这些模型处理核心逻辑。当涉及到Agentic AI时，有效性是您所需要的。LLMs是实现这一目标的一种选择。


### 论文摘要

While AI agents have long been discussed and studied in computer science, today's Agentic AI systems are something new. We consider other definitions of Agentic AI and propose a new realist definition. Agentic AI is a software delivery mechanism, comparable to software as a service (SaaS), which puts an application to work autonomously in a complex enterprise setting. Recent advances in large language models (LLMs) as foundation models have driven excitement in Agentic AI. We note, however, that Agentic AI systems are primarily applications, not foundations, and so their success depends on validation by end users and principal stakeholders. The tools and techniques needed by the principal users to validate their applications are quite different from the tools and techniques used to evaluate foundation models. Ironically, with good validation measures in place, in many cases the foundation models can be replaced with much simpler, faster, and more interpretable models that handle core logic. When it comes to Agentic AI, validity is what you need. LLMs are one option that might achieve it.

---

## 3. Image Hashing via Cross-View Code Alignment in the Age of Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.27584v1](http://arxiv.org/abs/2510.27584v1)

**作者:** Ilyass Moummad, Kawtar Zaher, Hervé Goëau, Alexis Joly

**发布时间:** 2025-10-31

### GPT解析

### 总结

论文提出了CroVCA（Cross-View Code Alignment）方法，一种简单统一的原则，用于学习在语义对齐视图中保持一致的二进制码，用于高效的大规模检索。

### 背景

高效的大规模检索需要既紧凑又有区分度的表示。基础模型提供强大的视觉和多模态嵌入，但这些高维空间中的最近邻搜索计算成本高。现有哈希方法通常依赖复杂的流程、多目标函数、专为单一学习范式设计且训练时间长。

### 目的

开发一种简单、高效且适应性强的方法来学习二进制码，用于快速的大规模检索，同时减少训练时间和计算资源需求。

### 方法

引入CroVCA原则，使用单个二元交叉熵损失强制语义对齐视图之间的码对齐，并采用编码率最大化作为防坍塌正则化子。设计了HashCoder，一种轻量级的MLP哈希网络，带有最终的批量归一化层以强制平衡的码。HashCoder可作为冻结嵌入上的探测头或通过LoRA微调适应编码器。

### 主要发现

CroVCA在基准测试中仅需5个训练周期就取得了最先进的结果。在16位时表现特别优异，例如在COCO上的无监督哈希在单GPU上不到2分钟完成，在ImageNet100上的监督哈希约3分钟完成。

### 结论

CroVCA提供了一种高效、适应性强的哈希方法，具有广泛的适用性，显著减少了训练时间和计算资源需求，同时保持了高质量的检索性能。

### 翻译

高效的大规模检索需要既紧凑又有区分度的表示。基础模型提供强大的视觉和多模态嵌入，但这些高维空间中的最近邻搜索计算成本高。哈希通过使用二进制码实现快速的汉明距离搜索，提供了一种高效的替代方案。然而，现有方法通常依赖复杂的流程、多目标函数、专为单一学习范式设计且训练时间长。我们介绍了CroVCA（Cross-View Code Alignment），一种简单统一的原则，用于学习在语义对齐视图中保持一致的二进制码。单个二元交叉熵损失强制对齐，而编码率最大化作为防坍塌正则化子促进平衡和多样化的码。为此，我们设计了HashCoder，一种轻量级的MLP哈希网络，带有最终的批量归一化层以强制平衡的码。HashCoder可用作冻结嵌入上的探测头，或通过LoRA微调高效适应编码器。在基准测试中，CroVCA仅需5个训练周期就取得了最先进的结果。在16位时，性能特别好，例如在COCO上的无监督哈希在单GPU上不到2分钟完成，在ImageNet100上的监督哈希约3分钟完成。这些结果突显了CroVCA的高效性、适应性和广泛适用性。


### 论文摘要

Efficient large-scale retrieval requires representations that are both compact and discriminative. Foundation models provide powerful visual and multimodal embeddings, but nearest neighbor search in these high-dimensional spaces is computationally expensive. Hashing offers an efficient alternative by enabling fast Hamming distance search with binary codes, yet existing approaches often rely on complex pipelines, multi-term objectives, designs specialized for a single learning paradigm, and long training times. We introduce CroVCA (Cross-View Code Alignment), a simple and unified principle for learning binary codes that remain consistent across semantically aligned views. A single binary cross-entropy loss enforces alignment, while coding-rate maximization serves as an anti-collapse regularizer to promote balanced and diverse codes. To implement this, we design HashCoder, a lightweight MLP hashing network with a final batch normalization layer to enforce balanced codes. HashCoder can be used as a probing head on frozen embeddings or to adapt encoders efficiently via LoRA fine-tuning. Across benchmarks, CroVCA achieves state-of-the-art results in just 5 training epochs. At 16 bits, it particularly well-for instance, unsupervised hashing on COCO completes in under 2 minutes and supervised hashing on ImageNet100 in about 3 minutes on a single GPU. These results highlight CroVCA's efficiency, adaptability, and broad applicability.

---

## 4. Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs

**论文链接:** [http://arxiv.org/abs/2510.27558v1](http://arxiv.org/abs/2510.27558v1)

**作者:** Sushil Samuel Dinesh, Shinkyu Park

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种无需领域特定训练的机器人操作框架，该框架利用预训练基础模型进行机器人操作，通过集成现成模型、多模态感知和通用推理模型实现稳健任务排序，并通过动态场景图提供空间感知和环境一致性推理能力。

### 背景

机器人操作领域通常需要领域特定的训练，而预训练基础模型在多个领域已显示出强大能力，但如何有效利用这些模型进行机器人操作尚不明确。

### 目的

开发一个无需领域特定训练的机器人操作框架，有效利用预训练基础模型进行机器人操作任务。

### 方法

集成现成的预训练基础模型，结合多模态感知和通用推理模型，动态维护场景图提供空间感知，通过场景图实现环境一致性推理，并在桌面机器人操作实验中评估框架性能。

### 主要发现

通过一系列桌面机器人操作实验评估，该框架展示了在现成基础模型之上直接构建机器人操作系统的潜力。

### 结论

该框架证明了利用预训练基础模型进行机器人操作的可行性，无需领域特定训练，为构建机器人操作系统提供了新思路。

### 翻译

本文提出了一个框架，该框架利用预训练的基础模型进行机器人操作，无需领域特定的训练。该框架集成现成的模型，结合来自基础模型的多模态感知和能够进行稳健任务排序的通用推理模型。在框架内动态维护的场景图提供了空间感知能力，并使对环境的一致推理成为可能。通过一系列桌面机器人操作实验对该框架进行了评估，结果凸显了在现成基础模型之上直接构建机器人操作系统的潜力。


### 论文摘要

This paper presents a framework that leverages pre-trained foundation models for robotic manipulation without domain-specific training. The framework integrates off-the-shelf models, combining multimodal perception from foundation models with a general-purpose reasoning model capable of robust task sequencing. Scene graphs, dynamically maintained within the framework, provide spatial awareness and enable consistent reasoning about the environment. The framework is evaluated through a series of tabletop robotic manipulation experiments, and the results highlight its potential for building robotic manipulation systems directly on top of off-the-shelf foundation models.

---

## 5. MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series

**论文链接:** [http://arxiv.org/abs/2510.27547v1](http://arxiv.org/abs/2510.27547v1)

**作者:** Xue Xia, Randall Balestriero, Tao Zhang, Yixin Zhou, Andrew Ding, Dev Saini, Lorenz Hurni

**发布时间:** 2025-10-31

### GPT解析

### 总结

MapSAM2是一个基于视觉基础模型的统一框架，用于自动分割历史地图图像和时间序列，通过将图像和时间序列视为视频处理，有效解决了历史地图分析中的风格多样性和数据稀缺性问题。

### 背景

历史地图是记录不同时间段地理特征的有价值档案，但由于其风格多样性和标注训练数据稀缺，自动分析面临挑战；从历史地图时间序列构建链接时空数据集更为耗时耗力，需要综合多张地图信息。

### 目的

开发一个统一框架自动分割历史地图图像和时间序列，支持建筑物年代测定、道路网络和聚落发展分析、环境变化研究等多种应用。

### 方法

MapSAM2基于视觉基础模型，使用少样本微调适应不同分割任务；关键创新是将历史地图图像和时间序列都视为视频处理，对于图像将图块组作为视频处理使内存注意力机制整合上下文线索，对于时间序列引入Siegfried建筑时间序列数据集并提出通过模拟时间变换从单年地图生成伪时间序列以减少标注成本。

### 主要发现

MapSAM2能有效学习时间关联，在有限监督或使用伪视频的情况下可准确分割和时间序列中的建筑物，提高了几何准确性，特别是对于区域特征。

### 结论

MapSAM2框架成功解决了历史地图图像和时间序列的自动分割问题，作者将发布数据集和代码以支持未来研究。

### 翻译

历史地图是记录不同时间段地理特征独特且有价值的档案。然而，由于其风格多样性和标注训练数据的稀缺性，历史地图图像的自动分析仍然是一个重大挑战。从历史地图时间序列构建链接时空数据集更加耗时耗力，因为它需要综合多张地图的信息。这类数据集对于建筑物年代测定、道路网络和聚落发展分析、环境变化研究等应用至关重要。我们提出了MapSAM2，一个用于自动分割历史地图图像和时间序列的统一框架。MapSAM2基于视觉基础模型，使用少样本微调适应不同的分割任务。我们的关键创新是将历史地图图像和时间序列都视为视频处理。对于图像，我们将一组图块作为视频处理，使内存注意力机制能够整合来自相似图块的上下文线索，从而提高了几何准确性，特别是对于区域特征。对于时间序列，我们引入了标注的Siegfried建筑时间序列数据集，并为了减少标注成本，提出了通过模拟常见的时间变换从单年地图生成伪时间序列。实验结果表明，MapSAM2能有效学习时间关联，并在有限监督或使用伪视频的情况下准确分割和时间序列中的建筑物。我们将发布我们的数据集和代码以支持未来研究。


### 论文摘要

Historical maps are unique and valuable archives that document geographic features across different time periods. However, automated analysis of historical map images remains a significant challenge due to their wide stylistic variability and the scarcity of annotated training data. Constructing linked spatio-temporal datasets from historical map time series is even more time-consuming and labor-intensive, as it requires synthesizing information from multiple maps. Such datasets are essential for applications such as dating buildings, analyzing the development of road networks and settlements, studying environmental changes etc. We present MapSAM2, a unified framework for automatically segmenting both historical map images and time series. Built on a visual foundation model, MapSAM2 adapts to diverse segmentation tasks with few-shot fine-tuning. Our key innovation is to treat both historical map images and time series as videos. For images, we process a set of tiles as a video, enabling the memory attention mechanism to incorporate contextual cues from similar tiles, leading to improved geometric accuracy, particularly for areal features. For time series, we introduce the annotated Siegfried Building Time Series Dataset and, to reduce annotation costs, propose generating pseudo time series from single-year maps by simulating common temporal transformations. Experimental results show that MapSAM2 learns temporal associations effectively and can accurately segment and link buildings in time series under limited supervision or using pseudo videos. We will release both our dataset and code to support future research.

---

## 6. Leveraging Generic Time Series Foundation Models for EEG Classification

**论文链接:** [http://arxiv.org/abs/2510.27522v1](http://arxiv.org/abs/2510.27522v1)

**作者:** Théo Gnassounou, Yessin Moakher, Shifeng Xie, Vasilii Feofanov, Ievgen Redko

**发布时间:** 2025-10-31

### GPT解析

### 总结

这项工作探索了时间序列基础模型在脑电图(EEG)任务中的应用潜力，发现即使是在非神经数据或合成信号上预训练的通用模型也能有效地迁移到EEG任务，并且性能超过特定于EEG的模型。

### 背景

时间序列基础模型正在成为强大的通用主干网络，但这些模型在特定领域生物医学信号（如脑电图EEG）方面的应用尚未得到充分探索。

### 目的

研究最近提出的时间序列分类基础模型在不同EEG任务上的适用性，包括运动想象分类和睡眠阶段预测。

### 方法

测试两种预训练方案：(a)在来自多个领域的异构真实世界时间序列上进行预训练，(b)在纯合成数据上进行预训练。

### 主要发现

两种变体都表现出强大的性能，一致优于广泛使用的卷积基线EEGNet和最新的特定于EEG的基础模型CBraMod，表明通用时间序列基础模型能有效迁移到EEG任务。

### 结论

利用跨领域预训练模型进行脑信号分析具有前景，EEG可能会从更广泛的时间序列文献的进步中受益。

### 翻译

时间序列基础模型正在作为强大的通用主干网络出现，然而它们在特定领域生物医学信号（如脑电图EEG）方面的潜力仍然相当未被探索。在这项工作中，我们研究了一种最近提出的时间序列分类基础模型在不同EEG任务（如运动想象分类和睡眠阶段预测）上的适用性。我们测试了两种预训练方案：(a)在来自多个领域的异构真实世界时间序列上预训练，(b)在纯合成数据上预训练。我们发现两种变体都表现出强大的性能，一致优于广泛使用的卷积基线EEGNet和最新的特定于EEG的基础模型CBraMod。这些结果表明，通用时间序列基础模型，即使是在非神经起源数据或合成信号上预训练的，也能有效地迁移到EEG。我们的发现强调了利用跨领域预训练模型进行脑信号分析的前景，表明EEG可能会从更广泛的时间序列文献的进步中受益。


### 论文摘要

Foundation models for time series are emerging as powerful general-purpose backbones, yet their potential for domain-specific biomedical signals such as electroencephalography (EEG) remains rather unexplored. In this work, we investigate the applicability a recently proposed time series classification foundation model, to a different EEG tasks such as motor imagery classification and sleep stage prediction. We test two pretraining regimes: (a) pretraining on heterogeneous real-world time series from multiple domains, and (b) pretraining on purely synthetic data. We find that both variants yield strong performance, consistently outperforming EEGNet, a widely used convolutional baseline, and CBraMod, the most recent EEG-specific foundation model. These results suggest that generalist time series foundation models, even when pretrained on data of non-neural origin or on synthetic signals, can transfer effectively to EEG. Our findings highlight the promise of leveraging cross-domain pretrained models for brain signal analysis, suggesting that EEG may benefit from advances in the broader time series literature.

---

## 7. Mitigating Semantic Collapse in Partially Relevant Video Retrieval

**论文链接:** [http://arxiv.org/abs/2510.27432v1](http://arxiv.org/abs/2510.27432v1)

**作者:** WonJun Moon, MinSeok Jung, Gilhan Park, Tae-Young Kim, Cheol-Ho Cho, Woojin Jun, Jae-Pil Heo

**发布时间:** 2025-10-31

**备注:** Accpeted to NeurIPS 2025. Code is available at  https://github.com/admins97/MSC_PRVR

### GPT解析

### 总结

本文提出了一种解决部分相关视频检索中语义崩塌问题的框架，通过文本相关性保持学习和跨分支视频对齐方法，显著提高了检索准确性。

### 背景

部分相关视频检索旨在检索与文本查询部分内容匹配的视频。现有方法将每个标注的文本-视频对视为正例，其他视为负例，忽略了单个视频内部和不同视频之间的丰富语义变化，导致同一视频中不同事件的嵌入空间压缩在一起，而不同视频中语义相似的查询和片段被分开。

### 目的

解决文本和视频嵌入空间中的语义崩塌问题，提高部分相关视频检索的准确性，特别是在处理包含多个不同事件的视频时。

### 方法

1) 文本相关性保持学习，保留基础模型编码的文本查询之间的语义关系；2) 跨分支视频对齐，一种对比对齐方法，解耦时间尺度上的分层视频表示；3) 保留顺序的令牌合并和自适应CBVA，生成内部连贯且相互区别的视频片段以增强对齐效果。

### 主要发现

提出的框架有效防止了语义崩塌，并在PRVR基准测试上显著提高了检索准确性。

### 结论

通过解决语义崩塌问题，该研究改进了部分相关视频检索的性能，特别是在处理包含多个不同事件的视频时。

### 翻译

部分相关视频检索(PRVPR)旨在检索与文本查询部分内容匹配的视频。现有方法将每个标注的文本-视频对视为正例，其他视为负例，忽略了单个视频内部和不同视频之间的丰富语义变化。因此，同一视频中不同事件的查询及其对应的视频片段段的嵌入空间压缩在一起，而不同视频中语义相似的查询和片段的嵌入空间被分开。这限制了视频包含多个、多样事件时的检索性能。本文解决了上述问题，称为文本和视频嵌入空间中的语义崩塌。我们首先引入文本相关性保持学习，保留基础模型编码的文本查询之间的语义关系。为解决视频嵌入中的崩塌问题，我们提出了跨分支视频对齐(CBVA)，一种对比对齐方法，解耦时间尺度上的分层视频表示。随后，我们引入保留顺序的令牌合并和自适应CBVA，通过生成内部连贯且相互区别的视频片段来增强对齐效果。在PRVR基准测试上的大量实验表明，我们的框架有效防止了语义崩塌并显著提高了检索准确性。


### 论文摘要

Partially Relevant Video Retrieval (PRVR) seeks videos where only part of the content matches a text query. Existing methods treat every annotated text-video pair as a positive and all others as negatives, ignoring the rich semantic variation both within a single video and across different videos. Consequently, embeddings of both queries and their corresponding video-clip segments for distinct events within the same video collapse together, while embeddings of semantically similar queries and segments from different videos are driven apart. This limits retrieval performance when videos contain multiple, diverse events. This paper addresses the aforementioned problems, termed as semantic collapse, in both the text and video embedding spaces. We first introduce Text Correlation Preservation Learning, which preserves the semantic relationships encoded by the foundation model across text queries. To address collapse in video embeddings, we propose Cross-Branch Video Alignment (CBVA), a contrastive alignment method that disentangles hierarchical video representations across temporal scales. Subsequently, we introduce order-preserving token merging and adaptive CBVA to enhance alignment by producing video segments that are internally coherent yet mutually distinctive. Extensive experiments on PRVR benchmarks demonstrate that our framework effectively prevents semantic collapse and substantially improves retrieval accuracy.

---

## 8. A Sensing Whole Brain Zebrafish Foundation Model for Neuron Dynamics and Behavior

**论文链接:** [http://arxiv.org/abs/2510.27366v1](http://arxiv.org/abs/2510.27366v1)

**作者:** Sam Fatehmanesh Vegas, Matt Thomson, James Gornet, David Prober

**发布时间:** 2025-10-31

### GPT解析

### 总结

研究团队开发了一种名为SBM的稀疏注意力全脑基础模型，用于斑马鱼幼虫，能够基于感觉刺激预测神经元放电概率，并将大脑状态与行为联系起来。该模型支持对复杂神经现象的快速、基于行为的探索。

### 背景

神经动力学支撑着从记忆到睡眠的各种行为，但识别高阶现象（如社交互动）的机制在实验上具有挑战性。现有的全脑模型往往无法扩展到单神经元分辨率，忽略了行为读出，或依赖于PCA/卷积管道，这些方法会错过长程、非线性相互作用。

### 目的

开发一种能够预测神经元放电概率并将大脑状态与行为联系起来的全脑模型，同时保持全脑规模和可解释性。

### 方法

研究团队引入了一种稀疏注意力全脑基础模型（SBM），该模型在神经元和时间上分解注意力，从而实现全脑规模和可解释性。模型与一个排列不变的行为头相结合，能够基于梯度合成引发目标行为的神经模式。

### 主要发现

在保留主体上，该模型实现了平均绝对误差小于0.02的校准预测和稳定的自回归滚动。通过排列不变的行为头，SBM能够基于梯度合成引发目标行为的神经模式。

### 结论

该框架支持对复杂神经现象的快速、基于行为的探索，为研究高阶神经现象提供了新的工具。

### 翻译

神经动力学支撑着从记忆到睡眠的各种行为，但识别高阶现象（如社交互动）的机制在实验上具有挑战性。现有的全脑模型往往无法扩展到单神经元分辨率，忽略行为读出，或依赖于PCA/卷积管道，这些方法会错过长程、非线性相互作用。我们引入了一种用于斑马鱼幼虫的稀疏注意力全脑基础模型（SBM），该模型基于感觉刺激预测神经元放电概率，并将大脑状态与行为联系起来。SBM在神经元和时间上分解注意力，从而实现全脑规模和可解释性。在保留主体上，它实现了平均绝对误差小于0.02的校准预测和稳定的自回归滚动。与一个排列不变的行为头相结合，SBM能够基于梯度合成引发目标行为的神经模式。该框架支持对复杂神经现象的快速、基于行为的探索。


### 论文摘要

Neural dynamics underlie behaviors from memory to sleep, yet identifying mechanisms for higher-order phenomena (e.g., social interaction) is experimentally challenging. Existing whole-brain models often fail to scale to single-neuron resolution, omit behavioral readouts, or rely on PCA/conv pipelines that miss long-range, non-linear interactions. We introduce a sparse-attention whole-brain foundation model (SBM) for larval zebrafish that forecasts neuron spike probabilities conditioned on sensory stimuli and links brain state to behavior. SBM factorizes attention across neurons and along time, enabling whole-brain scale and interpretability. On a held-out subject, it achieves mean absolute error <0.02 with calibrated predictions and stable autoregressive rollouts. Coupled to a permutation-invariant behavior head, SBM enables gradient-based synthesis of neural patterns that elicit target behaviors. This framework supports rapid, behavior-grounded exploration of complex neural phenomena.

---

## 9. Cross-Band Channel Impulse Response Prediction: Leveraging 3.5 GHz Channels for Upper Mid-Band

**论文链接:** [http://arxiv.org/abs/2510.27349v1](http://arxiv.org/abs/2510.27349v1)

**作者:** Fan-Hao Lin, Chi-Jui Sung, Chu-Hsiang Huang, Hui Chen, Chao-Kai Wen, Henk Wymeersch

**发布时间:** 2025-10-31

**备注:** 7 pages, 5 figures, 4 tables, this work has been submitted to IEEE  International Conference on Communications (ICC) 2026

### GPT解析

### 总结

本文提出了一种名为CIR-UNext的深度学习框架，用于6G网络中的跨频带信道预测，解决了中高频段信道预测的计算复杂性和数据采集成本高的问题。

### 背景

6G网络中，特别是在中高频段（FR3，7-24GHz），穿透损耗和阻塞问题严重。射线追踪方法虽然能提供高保真建模，但计算量大且高频数据采集成本高。

### 目的

开发一种高效且可扩展的跨频带信道预测方法，利用低频段数据预测高频段信道特性。

### 方法

提出CIR-UNext深度学习框架，利用丰富的3.5GHz信道脉冲响应（CIR）来预测7GHz的CIR。该框架结合基于射线追踪的数据集流程和注意力U-Net（AU-Net）变体进行增益和相位预测。

### 主要发现

AU-Net-Aux模型在未见过的复杂环境中实现了0.58dB的中值增益误差和0.27rad的相位预测误差。扩展的Channel2ComMap基础模型在MIMO-OFDM系统中的吞吐量预测表现优于现有方法。

### 结论

CIR-UNext为跨频带预测提供了高效且可扩展的解决方案，可应用于6G网络中的定位、波束管理、数字孪生和智能资源分配等领域。

### 翻译

精确的跨频带信道预测对6G网络至关重要，特别是在中高频段（FR3，7-24GHz），该频段的穿透损耗和阻塞问题严重。虽然射线追踪（RT）能提供高保真建模，但计算量大，且高频数据采集成本高。为应对这些挑战，我们提出了CIR-UNext，这是一个深度学习框架，旨在利用丰富的3.5GHz信道脉冲响应（CIR）来预测7GHz的CIR。该框架将基于RT的数据集流程与用于增益和相位预测的注意力U-Net（AU-Net）变体相结合。提出的AU-Net-Aux模型在未见过的复杂环境中实现了0.58dB的中值增益误差和0.27rad的相位预测误差。此外，我们将CIR-UNext扩展为一个基础模型Channel2ComMap，用于MIMO-OFDM系统中的吞吐量预测，显示出与现有方法相比的优越性能。总体而言，CIR-UNext为跨频带预测提供了高效且可扩展的解决方案，使定位、波束管理、数字孪生和智能资源分配等应用在6G网络中成为可能。


### 论文摘要

Accurate cross-band channel prediction is essential for 6G networks, particularly in the upper mid-band (FR3, 7--24 GHz), where penetration loss and blockage are severe. Although ray tracing (RT) provides high-fidelity modeling, it remains computationally intensive, and high-frequency data acquisition is costly. To address these challenges, we propose CIR-UNext, a deep learning framework designed to predict 7 GHz channel impulse responses (CIRs) by leveraging abundant 3.5 GHz CIRs. The framework integrates an RT-based dataset pipeline with attention U-Net (AU-Net) variants for gain and phase prediction. The proposed AU-Net-Aux model achieves a median gain error of 0.58 dB and a phase prediction error of 0.27 rad on unseen complex environments. Furthermore, we extend CIR-UNext into a foundation model, Channel2ComMap, for throughput prediction in MIMO-OFDM systems, demonstrating superior performance compared with existing approaches. Overall, CIR-UNext provides an efficient and scalable solution for cross-band prediction, enabling applications such as localization, beam management, digital twins, and intelligent resource allocation in 6G networks.

---

## 10. Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing

**论文链接:** [http://arxiv.org/abs/2510.27335v1](http://arxiv.org/abs/2510.27335v1)

**作者:** Yijia Wang, Yiqing Shen, Weiming Chen, Zhihai He

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种名为CIELR的新方法，通过将复杂用户指令转换为简单明确的编辑动作，避免了联合微调大型语言模型和扩散模型的高计算成本，实现了高效的复杂图像编辑。

### 背景

现有图像编辑方法在处理简单编辑指令时表现良好，但处理复杂指令时通常需要联合微调大型语言模型(LLMs)和扩散模型(DMs)，这涉及极高的计算复杂性和训练成本。

### 目的

解决现有方法在处理复杂图像编辑指令时的高计算成本问题，提出一种无需联合微调LLMs和DMs的新方法。

### 方法

CIELR方法首先利用基础模型构建输入图像的结构化语义表示，然后引入迭代更新机制逐步细化这一表示，获得图像场景的细粒度视觉表示，从而能够执行复杂且灵活的图像编辑任务。

### 主要发现

在SmartEdit Reasoning Scenario Set上的实验显示，该方法在PSNR指标上比之前的最先进方法高出9.955 dB，表明其在保持应保持一致区域方面的优越性。此外，作者还构建了包含86个图像样本的CIEBench基准，CIELR在该基准上也优于之前的方法。

### 结论

CIELR方法成功避免了联合微调LLMs和DMs的需要，通过将复杂指令分解为简单动作，实现了高效且高质量的复杂图像编辑，代码和数据集已在GitHub上公开。

### 翻译

现有的图像编辑方法可以很好地处理简单的编辑指令。为了处理复杂的编辑指令，它们通常需要联合微调大型语言模型(LLMs)和扩散模型(DMs)，这涉及非常高的计算复杂性和训练成本。为了解决这个问题，我们提出了一种新方法，称为CIELR(Complex Image Editing via LLM Reasoning)，它将复杂的用户指令转换为一组简单明确的编辑动作，消除了联合微调大型语言模型和扩散模型的需要。具体来说，我们首先使用基础模型构建输入图像的结构化语义表示。然后，我们引入一个迭代更新机制，可以逐步细化这一表示，获得图像场景的细粒度视觉表示。这使我们能够执行复杂且灵活的图像编辑任务。在SmartEdit Reasoning Scenario Set上的大量实验表明，我们的方法在PSNR指标上比之前的最先进方法高出9.955 dB，表明其在保持应保持一致区域方面的优越性。由于公共数据集中复杂推理图像编辑的样本有限，我们构建了一个名为CIEBench的基准，包含86个图像样本，以及一个专门用于基于推理的图像编辑的指标。CIELR在这个基准上也优于之前的方法。代码和数据集可在https://github.com/Jia-shao/Reasoning-Editing获取。


### 论文摘要

Existing image editing methods can handle simple editing instructions very well. To deal with complex editing instructions, they often need to jointly fine-tune the large language models (LLMs) and diffusion models (DMs), which involves very high computational complexity and training cost. To address this issue, we propose a new method, called \textbf{C}omplex \textbf{I}mage \textbf{E}diting via \textbf{L}LM \textbf{R}easoning (CIELR), which converts a complex user instruction into a set of simple and explicit editing actions, eliminating the need for jointly fine-tuning the large language models and diffusion models. Specifically, we first construct a structured semantic representation of the input image using foundation models. Then, we introduce an iterative update mechanism that can progressively refine this representation, obtaining a fine-grained visual representation of the image scene. This allows us to perform complex and flexible image editing tasks. Extensive experiments on the SmartEdit Reasoning Scenario Set show that our method surpasses the previous state-of-the-art by 9.955 dB in PSNR, indicating its superior preservation of regions that should remain consistent. Due to the limited number of samples of public datasets of complex image editing with reasoning, we construct a benchmark named CIEBench, containing 86 image samples, together with a metric specifically for reasoning-based image editing. CIELR also outperforms previous methods on this benchmark. The code and dataset are available at \href{https://github.com/Jia-shao/Reasoning-Editing}{https://github.com/Jia-shao/Reasoning-Editing}.

---

## 11. Fusion of Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis

**论文链接:** [http://arxiv.org/abs/2510.27237v1](http://arxiv.org/abs/2510.27237v1)

**作者:** Zhidong Yang, Xiuhui Shi, Wei Ba, Zhigang Song, Haijing Luan, Taiyuan Hu, Senlin Lin, Jiguang Wang, Shaohua Kevin Zhou, Rui Yan

**发布时间:** 2025-10-31

**备注:** 22 pages, 9 figures

### GPT解析

### 总结

本研究提出了一种名为FuseCPath的新型框架，用于融合异质性病理学基础模型，提升模型整体性能。该框架通过多视图聚类过滤代表性补丁、集群级重新嵌入策略融合补丁级模型，以及协作蒸馏策略融合幻灯片级模型，在多种癌症数据集上实现了最先进的性能。

### 背景

全幻灯片图像分析已成为计算病理学中日益重要的技术。最近的病理学基础模型在从WSI中提取有意义的补丁级或幻灯片级特征方面显示出显著优势。然而，当前病理学基础模型由于多样化的私有训练数据集和不同网络架构表现出显著异质性，导致使用不同基础模型提取的特征进行下游任务时性能不稳定。

### 目的

为了充分利用多种基础模型的优势，本研究旨在提出一种新型框架来融合异质性病理学基础模型，从而获得具有优越集成性能的模型。

### 方法

FuseCPath框架包含三个主要贡献：(i) 基于多视图聚类的方法，通过多个基础模型的嵌入来过滤出具有代表性的训练补丁；(ii) 集群级重新嵌入策略，在线捕获补丁级局部特征，以有效融合异质性补丁级基础模型；(iii) 协作蒸馏策略，探索基础模型之间的连接，以有效融合异质性幻灯片级基础模型。

### 主要发现

在癌症基因组图谱的肺癌、膀胱癌和结直肠癌数据集上进行的大量实验表明，所提出的FuseCPath在这些公共数据集的多个任务上实现了最先进的性能。

### 结论

FuseCPath框架通过有效融合异质性病理学基础模型，解决了当前病理学基础模型存在的异质性问题，为全幻灯片图像分析提供了更强大的工具，在多种癌症数据集上取得了优异的性能表现。

### 翻译

全幻灯片图像分析已成为计算病理学中日益重要的技术。最近的病理学基础模型进展在从全幻灯片图像中提取有意义的补丁级或幻灯片级特征表示方面显示出显著优势。然而，当前的病理学基础模型由于多样化的私有训练数据集和不同的网络架构表现出显著的异质性。这种异质性在使用不同基础模型提取的特征进行下游任务时会导致性能变化。为了充分利用多种基础模型的优势，在本工作中，我们提出了一种用于融合异质性病理学基础模型的新型框架，称为FuseCPath，产生了一个具有优越集成性能的模型。我们框架的主要贡献可以总结如下：(i) 为了保证训练补丁的代表性，我们提出了一种基于多视图聚类的方法，通过多个基础模型的嵌入来过滤出判别性补丁；(ii) 为了有效融合异质性补丁级基础模型，我们设计了一种集群级重新嵌入策略，在线捕获补丁级局部特征；(iii) 为了有效融合异质性幻灯片级基础模型，我们设计了一种协作蒸馏策略，探索幻灯片级基础模型之间的连接。在癌症基因组图谱的肺癌、膀胱癌和结直肠癌数据集上进行的大量实验表明，所提出的FuseCPath在这些公共数据集的多个任务上实现了最先进的性能。


### 论文摘要

Whole slide image (WSI) analysis has emerged as an increasingly essential technique in computational pathology. Recent advances in the pathological foundation models (FMs) have demonstrated significant advantages in deriving meaningful patch-level or slide-level feature representations from WSIs. However, current pathological FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the extracted features from different FMs in the downstream tasks. To fully explore the advantage of multiple FMs effectively, in this work, we propose a novel framework for the fusion of heterogeneous pathological FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: (i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the heterogeneous patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the heterogeneous slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs. Extensive experiments conducted on lung cancer, bladder cancer, and colorectal cancer datasets from The Cancer Genome Atlas (TCGA) have demonstrated that the proposed FuseCPath achieves state-of-the-art performance across multiple tasks on these public datasets.

---

## 12. MoRE: 3D Visual Geometry Reconstruction Meets Mixture-of-Experts

**论文链接:** [http://arxiv.org/abs/2510.27234v1](http://arxiv.org/abs/2510.27234v1)

**作者:** Jingnan Gao, Zhe Wang, Xianze Fang, Xingyu Ren, Zhuo Chen, Shengqi Liu, Yuhao Cheng, Jiangjing Lyu, Xiaokang Yang, Yichao Yan

**发布时间:** 2025-10-31

**备注:** Project Page: https://g-1nonly.github.io/MoRE_Website/, Code:  https://github.com/alibaba/Taobao3D

### GPT解析

### 总结

本文提出了MoRE，一种基于专家混合(MoE)架构的密集3D视觉基础模型，通过动态路由特征到任务特定专家，解决了3D模型扩展的挑战，实现了在各种几何任务中的最先进性能。

### 背景

语言和视觉领域的最新进展表明增加模型容量可提高任务表现；3D视觉几何重建中大规模训练也被证明有效，但由于几何监督复杂性和3D数据多样性，进一步扩展3D模型面临挑战。

### 目的

克服3D模型扩展的限制，提出一种提高可扩展性和适应性的密集3D视觉基础模型。

### 方法

提出MoRE模型，采用专家混合架构动态路由特征到任务特定专家；包含基于置信度的深度细化模块以稳定几何估计；集成密集语义特征与全局对齐的3D主干表示进行表面法线预测；使用定制的损失函数确保鲁棒学习。

### 主要发现

MoRE在多个基准测试中实现最先进性能，支持有效的下游应用且无需额外计算。

### 结论

MoRE通过专家混合架构解决了3D模型扩展挑战，提高了模型的可扩展性和适应性，并在真实世界条件下增强了鲁棒性。

### 翻译

近期语言和视觉领域的进展表明，扩展模型容量可以持续提高各种任务的表现。在3D视觉几何重建中，大规模训练同样被证明对学习通用表示有效。然而，由于几何监督的复杂性和3D数据的多样性，进一步扩展3D模型具有挑战性。为克服这些限制，我们提出了MoRE，一种基于专家混合(MoE)架构的密集3D视觉基础模型，它动态地将特征路由到任务特定的专家，使它们能够专业化于互补的数据方面，提高可扩展性和适应性。为提高真实世界条件下的鲁棒性，MoRE包含一个基于置信度的深度细化模块，用于稳定和细化几何估计。此外，它将密集语义特征与全局对齐的3D主干表示集成，用于高保真表面法线预测。MoRE还使用定制的损失函数进行优化，确保对各种输入和多种几何任务的鲁棒学习。大量实验表明，MoRE在多个基准测试中实现了最先进的性能，并支持有效的下游应用，无需额外计算。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D视觉几何重建领域的可扩展性问题。传统方法依赖场景特定优化，缺乏灵活性，难以在AR/VR、游戏内容创建、机器人和自动驾驶等需要强大几何先验和跨场景一致性的现实应用中发挥作用。随着模型和数据规模的扩大，3D重建面临几何监督复杂性和数据多样性的挑战，限制了性能提升。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D重建需要像语言和视觉领域那样的可扩展基础模型，但直接扩展3D模型面临特殊挑战。他们受混合专家模型(MoE)框架启发，该框架在大型语言模型中已证明能有效扩展神经网络。作者借鉴了VGGT作为基础架构，参考了DINOv2的特征提取技术，并受MoGe等单目几何估计模型的启发。他们设计了一个结合MoE的3D重建框架，通过动态路由特征到专家，实现不同场景的自适应表示学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将混合专家(MoE)框架引入3D视觉几何重建，创建一个密集的3D视觉基础模型，使模型能够动态路由特征到任务特定专家，学习不同场景的自适应和互补表示。整体流程包括：1)使用密集视觉Transformer骨干网络提取特征；2)实现MoE层，动态路由特征到专家；3)设计基于置信度的深度细化模块过滤噪声；4)融合全局3D特征与局部语义特征增强法线预测；5)使用多任务训练目标和自适应损失策略稳定训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将MoE框架系统应用于3D视觉几何重建；2)基于置信度的深度细化模块提高深度估计准确性；3)密集语义特征融合增强表面法线预测；4)多任务训练策略确保稳定学习。相比之前工作，MoRE通过MoE架构实现了更好的可扩展性和适应性，能处理各种3D场景，而传统方法通常针对特定场景优化；与其他前馈3D重建方法相比，MoRE能保持跨场景的高性能，并通过融合语义特征提高了细节表现。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MoRE通过将混合专家框架引入3D视觉几何重建，创建了一个可扩展且自适应的基础模型，能够通过动态路由特征到任务特定专家，实现高质量、鲁棒的3D几何预测，并融合语义特征提高表面法线估计的准确性。'}


### 论文摘要

Recent advances in language and vision have demonstrated that scaling up model capacity consistently improves performance across diverse tasks. In 3D visual geometry reconstruction, large-scale training has likewise proven effective for learning versatile representations. However, further scaling of 3D models is challenging due to the complexity of geometric supervision and the diversity of 3D data. To overcome these limitations, we propose MoRE, a dense 3D visual foundation model based on a Mixture-of-Experts (MoE) architecture that dynamically routes features to task-specific experts, allowing them to specialize in complementary data aspects and enhance both scalability and adaptability. Aiming to improve robustness under real-world conditions, MoRE incorporates a confidence-based depth refinement module that stabilizes and refines geometric estimation. In addition, it integrates dense semantic features with globally aligned 3D backbone representations for high-fidelity surface normal prediction. MoRE is further optimized with tailored loss functions to ensure robust learning across diverse inputs and multiple geometric tasks. Extensive experiments demonstrate that MoRE achieves state-of-the-art performance across multiple benchmarks and supports effective downstream applications without extra computation.

---

## 13. SpecAware: A Spectral-Content Aware Foundation Model for Unifying Multi-Sensor Learning in Hyperspectral Remote Sensing Mapping

**论文链接:** [http://arxiv.org/abs/2510.27219v1](http://arxiv.org/abs/2510.27219v1)

**作者:** Renjie Ji, Xue Wang, Chao Niu, Wen Zhang, Yong Mei, Kun Tan

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了SpecAware，一种新型的高光谱光谱内容感知基础模型，用于统一多传感器学习以进行高光谱成像映射。作者还构建了Hyper-400K数据集，包含超过40万个图像块。SpecAware采用两步超网络驱动编码过程，能够感知和解释不同场景和传感器中的空间-光谱特征，实验证明其在多个任务上表现优异。

### 背景

高光谱成像(HSI)是精细土地利用和土地覆盖(LULC)制图的重要工具，但其数据的内在异质性一直是开发通用模型的主要障碍。现有HSI基础模型忽略了传感器元属性的关键指导作用，且难以进行多传感器训练，限制了可迁移性。

### 目的

解决HSI基础模型中传感器元属性被忽略以及多传感器训练困难的问题，开发一个能够统一多传感器学习的高光谱成像映射基础模型。

### 方法

1. 提出SpecAware模型；2. 构建Hyper-400K数据集；3. 设计两步超网络驱动编码过程：元内容感知模块融合传感器元属性和图像内容生成条件输入，超嵌入模块通过样本条件超网络动态生成矩阵因子对进行通道编码。

### 主要发现

SpecAware能够感知和解释不同场景和传感器中的空间-光谱特征，自适应处理可变数量的光谱通道，建立统一的联合预训练框架。在六个数据集上的实验表明，SpecAware在土地覆盖语义分割分类、变化检测和场景分类方面表现优异。

### 结论

SpecAware通过考虑传感器元属性和采用两步超网络驱动编码过程，成功解决了HSI基础模型在多传感器训练和可迁移性方面的挑战，为高光谱成像映射提供了统一且高效的解决方案。

### 翻译

高光谱成像(HSI)是精细土地利用和土地覆盖(LULC)制图的重要工具。然而，HSI数据的内在异质性长期以来一直是通过联合训练开发通用模型的主要障碍。尽管HSI基础模型在不同下游任务中显示出潜力，但现有方法通常忽略了传感器元属性的关键指导作用，并且难以进行多传感器训练，限制了它们的可迁移性。为了解决这些挑战，我们提出了SpecAware，这是一种新型的高光谱光谱内容感知基础模型，用于统一多传感器学习以进行HSI映射。我们还构建了Hyper-400K数据集来促进这项研究，这是一个新的包含超过40万个来自不同机载AVIRIS传感器图像块的大规模高质量基准数据集。SpecAware的核心是一个两步超网络驱动的HSI数据编码过程。首先，我们设计了一个元内容感知模块，通过融合传感器元属性和其自身的图像内容，为每个HSI块生成一个独特的条件输入，针对每个样本的每个光谱波段定制。其次，我们设计了超嵌入模块，其中样本条件超网络动态生成一对用于通道编码的矩阵因子，包括自适应空间模式提取和潜在语义特征重投影。因此，SpecAware获得了感知和解释不同场景和传感器中空间-光谱特征的能力。这反过来又使SpecAware能够自适应处理可变数量的光谱通道，建立统一的联合预训练框架。在六个数据集上的广泛实验表明，SpecAware能够学习优越的特征表示，在土地覆盖语义分割分类、变化检测和场景分类方面表现出色。


### 论文摘要

Hyperspectral imaging (HSI) is a vital tool for fine-grained land-use and land-cover (LULC) mapping. However, the inherent heterogeneity of HSI data has long posed a major barrier to developing generalized models via joint training. Although HSI foundation models have shown promise for different downstream tasks, the existing approaches typically overlook the critical guiding role of sensor meta-attributes, and struggle with multi-sensor training, limiting their transferability. To address these challenges, we propose SpecAware, which is a novel hyperspectral spectral-content aware foundation model for unifying multi-sensor learning for HSI mapping. We also constructed the Hyper-400K dataset to facilitate this research, which is a new large-scale, high-quality benchmark dataset with over 400k image patches from diverse airborne AVIRIS sensors. The core of SpecAware is a two-step hypernetwork-driven encoding process for HSI data. Firstly, we designed a meta-content aware module to generate a unique conditional input for each HSI patch, tailored to each spectral band of every sample by fusing the sensor meta-attributes and its own image content. Secondly, we designed the HyperEmbedding module, where a sample-conditioned hypernetwork dynamically generates a pair of matrix factors for channel-wise encoding, consisting of adaptive spatial pattern extraction and latent semantic feature re-projection. Thus, SpecAware gains the ability to perceive and interpret spatial-spectral features across diverse scenes and sensors. This, in turn, allows SpecAware to adaptively process a variable number of spectral channels, establishing a unified framework for joint pre-training. Extensive experiments on six datasets demonstrate that SpecAware can learn superior feature representations, excelling in land-cover semantic segmentation classification, change detection, and scene classification.

---

## 14. FMint-SDE: A Multimodal Foundation Model for Accelerating Numerical Simulation of SDEs via Error Correction

**论文链接:** [http://arxiv.org/abs/2510.27173v1](http://arxiv.org/abs/2510.27173v1)

**作者:** Jiaxin Yuan, Haizhao Yang, Maria Cameron

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种名为FMint-SDE的多模态基础模型，用于快速准确地模拟动态系统，解决了传统数值积分器在准确性和效率之间的权衡问题，以及现有神经网络方法需要为每个案例单独训练模型的限制。

### 背景

快速准确地模拟动态系统是科学和工程领域的基本挑战。传统数值积分器通常在准确性和计算效率之间面临权衡，而现有的基于神经网络的方法通常需要为每种情况单独训练一个模型。

### 目的

克服传统方法和现有神经网络方法的局限性，引入一种新的多模态基础模型用于大规模微分方程模拟。

### 方法

提出了FMint-SDE（基于初始化的随机微分方程基础模型），它基于仅解码器的transformer架构，具有上下文学习能力。该模型利用数值和文本模态学习通用误差校正方案，使用传统求解器生成的粗解序列进行提示训练，从而实现对不同系统的广泛泛化。

### 主要发现

在涵盖分子动力学、机械系统、金融和生物学应用的一系列具有挑战性的SDE基准测试上，实验结果表明FMint-SDE相比经典求解器实现了更优的准确性-效率权衡。

### 结论

FMint-SDE作为动态系统通用仿真工具具有巨大潜力，能够实现快速准确的模拟。

### 翻译

快速准确地模拟动态系统是科学和工程领域的基本挑战。传统数值积分器通常在准确性和计算效率之间面临权衡，而现有的基于神经网络的方法通常需要为每种情况单独训练一个模型。为了克服这些限制，我们引入了一种新颖的多模态基础模型FMint-SDE（基于初始化的随机微分方程基础模型），用于大规模微分方程模拟。基于具有上下文学习能力的仅解码器transformer，FMint-SDE利用数值和文本模态学习通用误差校正方案。它使用传统求解器生成的粗解序列进行提示训练，从而实现对不同系统的广泛泛化。我们在一系列具有挑战性的SDE基准测试上评估了我们的模型，这些测试涵盖了分子动力学、机械系统、金融和生物学应用。实验结果表明，我们的方法相比经典求解器实现了更优的准确性-效率权衡，凸显了FMint-SDE作为动态系统通用仿真工具的潜力。


### 论文摘要

Fast and accurate simulation of dynamical systems is a fundamental challenge across scientific and engineering domains. Traditional numerical integrators often face a trade-off between accuracy and computational efficiency, while existing neural network-based approaches typically require training a separate model for each case. To overcome these limitations, we introduce a novel multi-modal foundation model for large-scale simulations of differential equations: FMint-SDE (Foundation Model based on Initialization for stochastic differential equations). Based on a decoder-only transformer with in-context learning, FMint-SDE leverages numerical and textual modalities to learn a universal error-correction scheme. It is trained using prompted sequences of coarse solutions generated by conventional solvers, enabling broad generalization across diverse systems. We evaluate our models on a suite of challenging SDE benchmarks spanning applications in molecular dynamics, mechanical systems, finance, and biology. Experimental results show that our approach achieves a superior accuracy-efficiency tradeoff compared to classical solvers, underscoring the potential of FMint-SDE as a general-purpose simulation tool for dynamical systems.

---

## 15. AD-SAM: Fine-Tuning the Segment Anything Vision Foundation Model for Autonomous Driving Perception

**论文链接:** [http://arxiv.org/abs/2510.27047v1](http://arxiv.org/abs/2510.27047v1)

**作者:** Mario Camarena, Het Patel, Fatemeh Nazari, Evangelos Papalexakis, Mohamadhossein Noruzoliaee, Jia Chen

**发布时间:** 2025-10-30

**备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems  (IEEE T-ITS)

### GPT解析

### 总结

本文提出了AD-SAM模型，这是一个针对自动驾驶领域语义分割任务优化的视觉基础模型。该模型通过双编码器和可变形解码器扩展了SAM模型，在道路场景的复杂空间和几何特性上表现出色。实验证明AD-SAM在多个基准测试中超越了现有模型，具有更好的分割精度、跨域泛化能力和数据效率。

### 背景

自动驾驶需要精确的语义分割来理解复杂的道路场景。现有的基础模型如SAM在自动驾驶领域的应用可能无法充分满足对空间和几何复杂性的需求。此外，自动驾驶领域需要模型能够高效学习并减少标注成本。

### 目的

开发一个专门针对自动驾驶场景优化的语义分割模型，提高分割精度、边界精确度，同时实现更好的泛化能力和数据效率，减少对大量标注数据的依赖。

### 方法

1. 双编码器结构：结合SAM预训练的Vision Transformer (ViT-H)的全局语义上下文和可训练的卷积深度学习主干网络（ResNet-50）的局部空间细节，生成多尺度融合表示。2. 可变形融合模块：对齐不同尺度和对象几何形状的异构特征。3. 可变形注意力解码器：执行渐进式多阶段细化。4. 混合损失函数：结合Focal、Dice、Lovasz-Softmax和Surface损失，提高语义类别平衡性、边界精确度和优化稳定性。

### 主要发现

1. 在Cityscapes和BDD100K基准测试中，AD-SAM的分割精度分别达到68.1 mIoU和59.5 mIoU，显著优于SAM、G-SAM和DeepLabV3。2. 在结构化和多样化的道路场景中分别领先最多22.9和19.2 mIoU。3. AD-SAM表现出强大的跨域泛化能力，保留得分为0.87（而SAM为0.76）。4. 学习速度更快且更稳定，在30-40个周期内收敛，是基准模型学习速度的两倍。5. 仅使用1000个样本，AD-SAM仍能保持0.607 mIoU，显示出高数据效率。

### 结论

针对基础模型进行有针对性的架构和优化增强，能够实现可靠且可扩展的自动驾驶感知。AD-SAM通过结合全局语义和局部空间信息，以及使用可变形注意力机制，显著提高了自动驾驶场景中的语义分割性能，同时减少了训练数据需求。

### 翻译

本文提出了自动驾驶分割一切模型（AD-SAM），这是一个针对自动驾驶（AD）中语义分割任务微调的视觉基础模型。AD-SAM通过双编码器和可变形解码器扩展了分割一切模型（SAM），以适应道路场景的空间和几何复杂性。双编码器通过结合SAM预训练的Vision Transformer (ViT-H)的全局语义上下文和可训练的卷积深度学习主干网络（即ResNet-50）的局部空间细节，产生多尺度融合表示。可变形融合模块对齐了不同尺度和对象几何形状之间的异构特征。解码器使用可变形注意力执行渐进式多阶段细化。训练由混合损失指导，该损失整合了Focal、Dice、Lovasz-Softmax和Surface损失，提高了语义类别平衡性、边界精确度和优化稳定性。在Cityscapes和伯克利深度驾驶100K（BDD100K）基准测试中的实验表明，AD-SAM在分割精度上超越了SAM、广义SAM（G-SAM）和深度学习基线（DeepLabV3）。它在Cityscapes上达到68.1平均交并比（mIoU），在BDD100K上达到59.5 mIoU，在结构化和多样化的道路场景中分别领先SAM、G-SAM和DeepLabV3最多22.9和19.2 mIoU。AD-SAM表现出强大的跨域泛化能力，保留得分为0.87（SAM为0.76），并且学习速度更快、更稳定，在30-40个周期内收敛，是基准模型学习速度的两倍。仅使用1000个样本，AD-SAM仍能保持0.607 mIoU，表明数据效率对于降低标注成本至关重要。这些结果证实，对基础模型进行有针对性的架构和优化增强能够实现可靠且可扩展的AD感知。


### 论文摘要

This paper presents the Autonomous Driving Segment Anything Model (AD-SAM), a fine-tuned vision foundation model for semantic segmentation in autonomous driving (AD). AD-SAM extends the Segment Anything Model (SAM) with a dual-encoder and deformable decoder tailored to spatial and geometric complexity of road scenes. The dual-encoder produces multi-scale fused representations by combining global semantic context from SAM's pretrained Vision Transformer (ViT-H) with local spatial detail from a trainable convolutional deep learning backbone (i.e., ResNet-50). A deformable fusion module aligns heterogeneous features across scales and object geometries. The decoder performs progressive multi-stage refinement using deformable attention. Training is guided by a hybrid loss that integrates Focal, Dice, Lovasz-Softmax, and Surface losses, improving semantic class balance, boundary precision, and optimization stability. Experiments on the Cityscapes and Berkeley DeepDrive 100K (BDD100K) benchmarks show that AD-SAM surpasses SAM, Generalized SAM (G-SAM), and a deep learning baseline (DeepLabV3) in segmentation accuracy. It achieves 68.1 mean Intersection over Union (mIoU) on Cityscapes and 59.5 mIoU on BDD100K, outperforming SAM, G-SAM, and DeepLabV3 by margins of up to +22.9 and +19.2 mIoU in structured and diverse road scenes, respectively. AD-SAM demonstrates strong cross-domain generalization with a 0.87 retention score (vs. 0.76 for SAM), and faster, more stable learning dynamics, converging within 30-40 epochs, enjoying double the learning speed of benchmark models. It maintains 0.607 mIoU with only 1000 samples, suggesting data efficiency critical for reducing annotation costs. These results confirm that targeted architectural and optimization enhancements to foundation models enable reliable and scalable AD perception.

---

## 16. GeoPep: A geometry-aware masked language model for protein-peptide binding site prediction

**论文链接:** [http://arxiv.org/abs/2510.27040v1](http://arxiv.org/abs/2510.27040v1)

**作者:** Dian Chen, Yunkai Chen, Tong Lin, Sijie Chen, Xiaolin Cheng

**发布时间:** 2025-10-30

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

GeoPep是一种创新的肽结合位点预测框架，通过迁移学习从ESM3蛋白质基础模型获取知识，有效解决了蛋白质-肽相互作用预测中的挑战。

### 背景

多模态方法整合蛋白质结构和序列在蛋白质-蛋白质界面预测中已取得显著成功，但由于肽的内在构象灵活性和结构数据有限，这些方法难以扩展到蛋白质-肽相互作用领域。

### 目的

开发一种能够克服肽构象灵活性和结构数据有限性问题的蛋白质-肽结合位点预测方法。

### 方法

GeoPep框架利用ESM3多模态蛋白质基础模型的迁移学习，微调其预学习的蛋白质-蛋白质结合表示，并与参数高效的神经网络架构集成，使用基于距离的损失函数训练模型，利用3D结构信息增强预测能力。

### 主要发现

全面评估表明GeoPep能够有效捕获稀疏和异质的结合模式，在蛋白质-肽结合位点预测方面显著优于现有方法。

### 结论

GeoPep为蛋白质-肽相互作用预测提供了一种有效的解决方案，成功克服了该领域的关键挑战。

### 翻译

整合蛋白质结构和序列的多模态方法在蛋白质-蛋白质界面预测中取得了显著成功。然而，由于肽的内在构象灵活性和结构数据有限，阻碍了结构感知模型的直接训练，将这些方法扩展到蛋白质-肽相互作用仍然具有挑战性。为解决这些局限性，我们引入了GeoPep，这是一种新颖的肽结合位点预测框架，它利用从ESM3（多模态蛋白质基础模型）的迁移学习。GeoPep微调ESM3从蛋白质-蛋白质结合中预学习的丰富表示，以解决蛋白质-肽结合数据有限的问题。微调的模型进一步与能够从稀疏数据中学习复杂模式的参数高效神经网络架构集成。此外，模型使用基于距离的损失函数进行训练，利用3D结构信息增强结合位点预测。全面评估表明，GeoPep通过有效捕获稀疏和异质的结合模式，在蛋白质-肽结合位点预测方面显著优于现有方法。


### 论文摘要

Multimodal approaches that integrate protein structure and sequence have achieved remarkable success in protein-protein interface prediction. However, extending these methods to protein-peptide interactions remains challenging due to the inherent conformational flexibility of peptides and the limited availability of structural data that hinder direct training of structure-aware models. To address these limitations, we introduce GeoPep, a novel framework for peptide binding site prediction that leverages transfer learning from ESM3, a multimodal protein foundation model. GeoPep fine-tunes ESM3's rich pre-learned representations from protein-protein binding to address the limited availability of protein-peptide binding data. The fine-tuned model is further integrated with a parameter-efficient neural network architecture capable of learning complex patterns from sparse data. Furthermore, the model is trained using distance-based loss functions that exploit 3D structural information to enhance binding site prediction. Comprehensive evaluations demonstrate that GeoPep significantly outperforms existing methods in protein-peptide binding site prediction by effectively capturing sparse and heterogeneous binding patterns.

---

## 17. MoME: Mixture of Visual Language Medical Experts for Medical Imaging Segmentation

**论文链接:** [http://arxiv.org/abs/2510.26996v1](http://arxiv.org/abs/2510.26996v1)

**作者:** Arghavan Rezvani, Xiangyi Yan, Anthony T. Wu, Kun Han, Pooya Khosravi, Xiaohui Xie

**发布时间:** 2025-10-30

### GPT解析

### 总结

MoME是一种用于医学图像分割的视觉语言专家混合模型，结合了多尺度视觉特征和文本嵌入，通过动态专家选择实现高性能的医学图像分割。

### 背景

大型语言模型中广泛使用的专家混合范式在医学视觉语言任务中的应用尚不充分，医学图像分析需要更有效的处理方法。

### 目的

将成功的MoE范式应用于医学图像分割任务，探索视觉语言模型在医学领域的新型集成方法，实现稳健的医学图像分析。

### 方法

提出MoME架构，有效利用针对医学图像复杂性定制的多尺度视觉特征，并结合文本嵌入实现动态专家选择。使用包含3410个CT扫描的10个数据集进行训练和测试。

### 主要发现

MoME在全面的医学图像分割基准测试中表现出强大的性能，在多个数据集上展示了具有竞争力的精确度。通过整合文本信息，模型性能得到显著提升。

### 结论

MoME探索了一种用于实现稳健医学图像分析结果的新型架构，证明了将基础模型和MoE范式应用于医学图像分割的有效性。

### 翻译

在这项研究中，我们提出了MoME，一种用于医学图像分割的视觉语言医学专家混合模型。MoME借鉴了大型语言模型中广泛使用的专家混合范式，应用于医学视觉语言任务。该架构通过有效利用针对医学图像复杂性定制的多尺度视觉特征，并结合文本嵌入，实现了动态专家选择。这项工作探索了视觉语言模型在该领域的新型集成方法。利用包含3410个CT扫描的10个数据集，MoME在全面的医学图像分割基准测试中表现出强大的性能。我们的方法探索了基础模型在医学图像中的应用，受益于MoE通过整合文本信息提高模型性能的已证实有效性。MoME在多个数据集上展示了具有竞争力的精确度，探索了一种用于实现稳健医学图像分析结果的新型架构。


### 论文摘要

In this study, we propose MoME, a Mixture of Visual Language Medical Experts, for Medical Image Segmentation. MoME adapts the successful Mixture of Experts (MoE) paradigm, widely used in Large Language Models (LLMs), for medical vision-language tasks. The architecture enables dynamic expert selection by effectively utilizing multi-scale visual features tailored to the intricacies of medical imagery, enriched with textual embeddings. This work explores a novel integration of vision-language models for this domain. Utilizing an assembly of 10 datasets, encompassing 3,410 CT scans, MoME demonstrates strong performance on a comprehensive medical imaging segmentation benchmark. Our approach explores the integration of foundation models for medical imaging, benefiting from the established efficacy of MoE in boosting model performance by incorporating textual information. Demonstrating competitive precision across multiple datasets, MoME explores a novel architecture for achieving robust results in medical image analysis.

---

## 18. NaviTrace: Evaluating Embodied Navigation of Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.26909v1](http://arxiv.org/abs/2510.26909v1)

**作者:** Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey

**发布时间:** 2025-10-30

**备注:** 9 pages, 6 figures, under review at IEEE conference

### GPT解析

### 总结

本文提出了NaviTrace，一个高质量的视觉问答基准测试，用于评估视觉语言模型在机器人导航任务中的表现。该基准测试包含1000多个场景和3000多条专家轨迹，使用语义感知轨迹分数评估了八个最先进的VLM，发现模型在空间定位和目标定位方面与人类性能存在差距。

### 背景

视觉语言模型在多种任务和场景中展现出前所未有的性能和泛化能力，将其集成到机器人导航系统中有助于构建通用机器人。然而，评估这些模型导航能力受到现实世界试验成本高、模拟过于简单以及基准有限等限制。

### 目的

开发一个高质量的基准测试来评估视觉语言模型在机器人导航任务中的能力，解决当前评估方法存在的问题。

### 方法

提出NaviTrace基准测试，模型接收指令和实体类型（人类、腿式机器人、轮式机器人、自行车），在图像空间输出二维导航轨迹。使用语义感知轨迹分数（结合动态时间规整距离、目标端点误差和基于逐像素语义的实体条件惩罚）进行评估，并与人类偏好进行对比。

### 主要发现

八个最先进的VLM在机器人导航任务中与人类性能存在持续差距，主要问题在于空间定位和目标定位能力不足。NaviTrace为现实世界机器人导航提供了可扩展且可重现的基准测试。

### 结论

NaviTrace基准测试填补了视觉语言模型在机器人导航评估方面的空白，为未来研究提供了可靠的评估工具，有助于推动通用机器人的发展。

### 翻译

视觉语言模型在各种任务和场景中展现出前所未有的性能和泛化能力。将这些基础模型集成到机器人导航系统中，为构建通用机器人开辟了途径。然而，评估这些模型的导航能力仍然受到现实世界试验成本高、模拟过于简单以及基准有限等限制。我们引入了NaviTrace，这是一个高质量的视觉问答基准测试，模型接收指令和实体类型（人类、腿式机器人、轮式机器人、自行车），必须在图像空间输出二维导航轨迹。在1000多个场景和3000多条专家轨迹中，我们使用新引入的语义感知轨迹分数系统地评估了八个最先进的VLM。该指标结合了动态时间规整距离、目标端点误差以及基于逐像素语义的实体条件惩罚，并与人类偏好相关。我们的评估显示，由于空间定位和目标定位能力差，模型与人类性能之间存在持续差距。NaviTrace为现实世界机器人导航建立了可扩展且可重现的基准测试。基准测试和排行榜可以在https://leggedrobotics.github.io/navitrace_webpage/找到。


### 论文摘要

Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at https://leggedrobotics.github.io/navitrace_webpage/.

---

## 19. Cognition Envelopes for Bounded AI Reasoning in Autonomous UAS Operations

**论文链接:** [http://arxiv.org/abs/2510.26905v1](http://arxiv.org/abs/2510.26905v1)

**作者:** Pedro Antonio Alarcón Granadeno, Arturo Miguel Bernal Russell, Sofia Nelson, Demetrius Hernandez, Maureen Petterson, Michael Murphy, Walter J. Scheirer, Jane Cleland-Huang

**发布时间:** 2025-10-30

**备注:** 10.5 pages, 9 figures

### GPT解析

### 总结

论文提出了认知包络(Cognition Envelopes)的概念，用于约束AI模型在网络物理系统中的决策，以应对模型引入的新类型错误。

### 背景

网络物理系统越来越多地依赖基础模型如大型语言模型(LLMs)和视觉语言模型(VLMs)，通过增强感知、推理和规划来提高自主性。

### 目的

解决基础模型引入的新类型错误(如幻觉、过度泛化和上下文不匹配)导致的错误决策问题。

### 方法

引入认知包络概念，建立推理边界来约束AI生成的决策，同时补充元认知和传统安全包络的使用，并为其制定实际指南和系统化流程进行定义、验证和保证。

### 主要发现

摘要中未明确提及具体研究发现。

### 结论

认知包络是管理AI模型决策风险的一种有效方法，需要系统化的流程来确保其有效性。

### 翻译

网络物理系统越来越多地依赖基础模型，如大型语言模型(LLMs)和视觉语言模型(VLMs)，通过增强感知、推理和规划来提高自主性。然而，这些模型也引入了新的错误类型，如幻觉、过度泛化和上下文不匹配，导致错误和有缺陷的决策。为了解决这个问题，我们引入了认知包络的概念，旨在建立推理边界，以约束AI生成的决策，同时补充元认知和传统安全包络的使用。与安全包络一样，认知包络需要其实际指南和系统化的流程来进行定义、验证和保证。


### 论文摘要

Cyber-physical systems increasingly rely on Foundational Models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) to increase autonomy through enhanced perception, inference, and planning. However, these models also introduce new types of errors, such as hallucinations, overgeneralizations, and context misalignments, resulting in incorrect and flawed decisions. To address this, we introduce the concept of Cognition Envelopes, designed to establish reasoning boundaries that constrain AI-generated decisions while complementing the use of meta-cognition and traditional safety envelopes. As with safety envelopes, Cognition Envelopes require practical guidelines and systematic processes for their definition, validation, and assurance.

---

## 20. Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for Time Series Classification

**论文链接:** [http://arxiv.org/abs/2510.26777v1](http://arxiv.org/abs/2510.26777v1)

**作者:** Andreas Auer, Daniel Klotz, Sebastinan Böck, Sepp Hochreiter

**发布时间:** 2025-10-30

**备注:** NeurIPS 2025 Workshop on Recent Advances in Time Series Foundation  Models (BERT2S)

### GPT解析

### 总结

本研究探讨了预训练的时间序列预测模型在分类任务上的表示有效性，挑战了任务特定预训练的必要性假设。

### 背景

最近关于时间序列基础模型的研究主要集中在预测任务上，不清楚这些模型学习到的表示有多大的通用性。

### 目的

研究冻结预训练的预测模型是否能为分类任务提供有效的表示，以及预测性能与分类性能之间的关系。

### 方法

比较不同的表示提取策略，并引入两种与模型无关的嵌入增强方法，通过实验评估预测模型在分类任务上的表现。

### 主要发现

最好的预测模型在分类任务上实现的准确率与或甚至超过了专门为分类预训练的最先进模型，且预测性能和分类性能之间存在正相关。

### 结论

学习预测可能是构建通用时间序列基础模型的有力途径，挑战了任务特定预训练的必要性假设。

### 翻译

最近关于时间序列基础模型的研究主要集中在预测上，这使得它们学习到的表示的通用性尚不清楚。在本研究中，我们考察冻结预训练的预测模型是否能为分类提供有效表示。为此，我们比较了不同的表示提取策略，并引入了两种与模型无关的嵌入增强方法。我们的实验表明，最好的预测模型实现的分类准确率与甚至超过了专门为分类预训练的最先进模型。此外，我们观察到预测性能和分类性能之间存在正相关。这些发现挑战了任务特定预训练的必要性假设，并表明学习预测可能是构建通用时间序列基础模型的有力途径。


### 论文摘要

Recent research on time series foundation models has primarily focused on forecasting, leaving it unclear how generalizable their learned representations are. In this study, we examine whether frozen pre-trained forecasting models can provide effective representations for classification. To this end, we compare different representation extraction strategies and introduce two model-agnostic embedding augmentations. Our experiments show that the best forecasting models achieve classification accuracy that matches or even surpasses that of state-of-the-art models pre-trained specifically for classification. Moreover, we observe a positive correlation between forecasting and classification performance. These findings challenge the assumption that task-specific pre-training is necessary, and suggest that learning to forecast may provide a powerful route toward constructing general-purpose time series foundation models.

---

## 21. Cross-Platform Evaluation of Reasoning Capabilities in Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.26732v1](http://arxiv.org/abs/2510.26732v1)

**作者:** J. de Curtò, I. de Zarzà, Pablo García, Jordi Cabot

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文对当代基础模型的推理能力进行了全面的跨平台评估，建立了不依赖特定基础设施的基准测试，涵盖高性能计算超级计算机、云平台和大学集群三种计算范式。

### 背景

当前基础模型在不同平台上的推理能力表现尚不明确，需要建立一个统一的评估框架。

### 目的

建立基础设施无关的基准测试，评估15个基础模型在8个学术领域79个问题上的推理能力，并探索不同计算环境下的性能差异。

### 方法

通过三个实验阶段进行评估：(1)基线建立：在MareNostrum 5上使用6个模型评估19个问题；(2)基础设施验证：在大学集群和Nebius AI Studio上重复基准测试；(3)扩展评估：在两个平台上对完整的79个问题进行全面评估。

### 主要发现

挑战了传统的扩展假设，确立了训练数据质量比模型大小更重要，为不同环境中的模型选择提供了可操作的指导原则。

### 结论

三种基础设施的方法和79个问题的基准测试使能够随着基础模型的演变进行推理能力的纵向跟踪。

### 翻译

本文对当代基础模型的推理能力进行了全面的跨平台评估，建立了一个不依赖特定基础设施的基准测试，涵盖高性能计算超级计算机、云平台和大学集群三种计算范式。我们评估了15个基础模型，涵盖物理、数学、化学、经济学、生物学、统计学、微积分和优化八个学术领域的79个问题。通过三个实验阶段：(1)基线建立：在MareNostrum 5上使用6个模型评估19个问题，建立方法和参考性能；(2)基础设施验证：在大学集群和Nebius AI Studio上重复19个问题的基准测试，确认与基础设施无关的可重复性；(3)扩展评估：在两个平台上对完整的79个问题进行全面评估，探索架构多样性方面的规模化泛化能力。研究挑战了传统的扩展假设，确立了训练数据质量比模型大小更重要，为教育、生产和研究环境中的模型选择提供了可操作的指导原则。三种基础设施的方法和79个问题的基准测试使能够随着基础模型的演变进行推理能力的纵向跟踪。


### 论文摘要

This paper presents a comprehensive cross-platform evaluation of reasoning capabilities in contemporary foundation models, establishing an infrastructure-agnostic benchmark across three computational paradigms: HPC supercomputing (MareNostrum 5), cloud platforms (Nebius AI Studio), and university clusters (a node with eight H200 GPUs).   We evaluate 15 foundation models across 79 problems spanning eight academic domains (Physics, Mathematics, Chemistry, Economics, Biology, Statistics, Calculus, and Optimization) through three experimental phases: (1) Baseline establishment: Six models (Mixtral-8x7B, Phi-3, LLaMA 3.1-8B, Gemma-2-9b, Mistral-7B, OLMo-7B) evaluated on 19 problems using MareNostrum 5, establishing methodology and reference performance; (2) Infrastructure validation: The 19-problem benchmark repeated on university cluster (seven models including Falcon-Mamba state-space architecture) and Nebius AI Studio (nine state-of-the-art models: Hermes-4 70B/405B, LLaMA 3.1-405B/3.3-70B, Qwen3 30B/235B, DeepSeek-R1, GPT-OSS 20B/120B) to confirm infrastructure-agnostic reproducibility; (3) Extended evaluation: Full 79-problem assessment on both university cluster and Nebius platforms, probing generalization at scale across architectural diversity.   The findings challenge conventional scaling assumptions, establish training data quality as more critical than model size, and provide actionable guidelines for model selection across educational, production, and research contexts. The tri-infrastructure methodology and 79-problem benchmark enable longitudinal tracking of reasoning capabilities as foundation models evolve.

---

## 22. LSM-MS2: A Foundation Model Bridging Spectral Identification and Biological Interpretation

**论文链接:** [http://arxiv.org/abs/2510.26715v1](http://arxiv.org/abs/2510.26715v1)

**作者:** Gabriel Asher, Devesh Shah, Amy A. Caudy, Luke Ferro, Lea Amar, Ana S. H. Costa, Thomas Patton, Niall O'Connor, Jennifer M. Campbell, Jack Geremia

**发布时间:** 2025-10-30

### GPT解析

### 总结

LSM-MS2是一个大规模深度学习基础模型，通过学习语义化学空间，显著提高了光谱识别的性能，特别是在异构化合物识别、复杂生物样本分析和低浓度条件下，并能直接进行生物学解释和临床预测。

### 背景

大多数质谱数据仍未被表征，导致大量生物和化学信息未被利用。机器学习的最新进展开始解决这个问题，特别是在串联质谱数据的光谱识别任务中。

### 目的

介绍LSM-MS2，一个大规模深度学习基础模型，旨在学习语义化学空间并解决质谱数据表征不足的问题。

### 方法

LSM-MS2是在数百万张光谱上训练的大规模深度学习基础模型，它学习语义化学空间，能够生成丰富的光谱嵌入。

### 主要发现

LSM-MS2在光谱识别方面取得了最先进的性能：在识别具有挑战性的异构化合物方面比现有方法提高30%的准确率；在复杂生物样本中产生42%的正确识别；在低浓度条件下保持稳健性；产生的光谱嵌入可直接从最少的下游数据进行生物学解释；成功区分不同的疾病状态并预测各种转化应用中的临床结果。

### 结论

LSM-MS2模型有效地解决了质谱数据表征不足的问题，为生物和化学信息的利用提供了新的可能性。

### 翻译

绝大多数质谱数据仍未被表征，导致其大量生物和化学信息未被利用。机器学习的最新进展开始解决这个问题，特别是在串联质谱数据的光谱识别任务方面。在此，我们介绍了LSM-MS2的最新一代，这是一个在数百万张光谱上训练的大规模深度学习基础模型，用于学习语义化学空间。LSM-MS2在光谱识别方面取得了最先进的性能，在识别具有挑战性的异构化合物方面比现有方法提高30%的准确率，在复杂生物样本中产生42%的正确识别，并在低浓度条件下保持稳健性。此外，LSM-MS2产生丰富的光谱嵌入，能够直接从最少的下游数据进行生物学解释，成功区分不同的疾病状态并预测各种转化应用中的临床结果。


### 论文摘要

A vast majority of mass spectrometry data remains uncharacterized, leaving much of its biological and chemical information untapped. Recent advances in machine learning have begun to address this gap, particularly for tasks such as spectral identification in tandem mass spectrometry data. Here, we present the latest generation of LSM-MS2, a large-scale deep learning foundation model trained on millions of spectra to learn a semantic chemical space. LSM-MS2 achieves state-of-the-art performance in spectral identification, improving on existing methods by 30% in accuracy of identifying challenging isomeric compounds, yielding 42% more correct identifications in complex biological samples, and maintaining robustness under low-concentration conditions. Furthermore, LSM-MS2 produces rich spectral embeddings that enable direct biological interpretation from minimal downstream data, successfully differentiating disease states and predicting clinical outcomes across diverse translational applications.

---

## 23. ProstNFound+: A Prospective Study using Medical Foundation Models for Prostate Cancer Detection

**论文链接:** [http://arxiv.org/abs/2510.26703v1](http://arxiv.org/abs/2510.26703v1)

**作者:** Paul F. R. Wilson, Mohamed Harmanani, Minh Nguyen Nhat To, Amoon Jamzad, Tarek Elghareb, Zhuoxin Guo, Adam Kinnaird, Brian Wodlinger, Purang Abolmaesumi, Parvin Mousavi

**发布时间:** 2025-10-30

### GPT解析

### 总结

该研究提出了ProstNFound+，一种基于医学基础模型的前列腺癌检测系统，并通过前瞻性验证了其在临床微超声图像中的有效性和实用性。

### 背景

医学基础模型为构建高性能诊断系统提供了新途径，但在前列腺癌微超声检测领域的临床应用尚未得到验证。

### 目的

开发并验证ProstNFound+系统，将医学基础模型应用于前列腺癌的微超声检测，并进行临床前瞻性验证。

### 方法

ProstNFound+整合了医学基础模型、适配器调优和嵌入前列腺癌特异性临床生物标志物的自定义提示编码器，生成癌症热图和风险评分。模型在多中心回顾性数据上训练，并在五年后新临床站点获取的数据上进行前瞻性评估，与标准临床评分方案(PRI-MUS和PI-RADS)进行对比。

### 主要发现

ProstNFound+在前瞻性数据上表现出强大的泛化能力，性能未出现下降；与临床评分高度一致；生成的热图具有可解释性，与活检证实的病变一致。

### 结论

ProstNFound+具有临床部署潜力，为专家驱动协议提供了可扩展且可解释的替代方案。

### 翻译

目的：医学基础模型为构建高性能诊断系统提供了一种途径。然而，这些模型在前列腺癌微超声检测中的临床应用尚未得到验证。我们提出了ProstNFound+，这是一个针对前列腺癌微超声检测的基础模型适应版本，并进行了首次前瞻性验证。方法：ProstNFound+整合了医学基础模型、适配器调优和嵌入前列腺癌特异性临床生物标志物的自定义提示编码器。该模型生成癌症热图和临床显著前列腺癌的风险评分。在多中心回顾性数据上训练后，模型在五年后从新临床站点获取的数据上进行前瞻性评估。模型预测与标准临床评分方案(PRI-MUS和PI-RADS)进行基准测试。结果：ProstNFound+在前瞻性数据上显示出强大的泛化能力，与回顾性评估相比没有性能下降。与临床评分高度一致，生成与活检证实的病变一致的可解释热图。结论：结果突显了ProstNFound+在临床部署方面的潜力，提供了可扩展且可解释的替代方案，替代专家驱动的协议。


### 论文摘要

Purpose: Medical foundation models (FMs) offer a path to build high-performance diagnostic systems. However, their application to prostate cancer (PCa) detection from micro-ultrasound ({\mu}US) remains untested in clinical settings. We present ProstNFound+, an adaptation of FMs for PCa detection from {\mu}US, along with its first prospective validation. Methods: ProstNFound+ incorporates a medical FM, adapter tuning, and a custom prompt encoder that embeds PCa-specific clinical biomarkers. The model generates a cancer heatmap and a risk score for clinically significant PCa. Following training on multi-center retrospective data, the model is prospectively evaluated on data acquired five years later from a new clinical site. Model predictions are benchmarked against standard clinical scoring protocols (PRI-MUS and PI-RADS). Results: ProstNFound+ shows strong generalization to the prospective data, with no performance degradation compared to retrospective evaluation. It aligns closely with clinical scores and produces interpretable heatmaps consistent with biopsy-confirmed lesions. Conclusion: The results highlight its potential for clinical deployment, offering a scalable and interpretable alternative to expert-driven protocols.

---

## 24. Aeolus: A Multi-structural Flight Delay Dataset

**论文链接:** [http://arxiv.org/abs/2510.26616v2](http://arxiv.org/abs/2510.26616v2)

**作者:** Lin Xu, Xinyun Yuan, Yuxuan Liang, Suwan Yin, Yuankai Wu

**发布时间:** 2025-10-30

### GPT解析

### 总结

介绍Aeolus，一个大规模多模态航班延误数据集，用于推进航班延误预测研究和支持表格数据基础模型的开发。

### 背景

现有领域数据集通常限于扁平表格结构，无法捕捉航班延误传播中固有的时空动态特性。

### 目的

解决现有数据集的局限性，提供能够捕捉航班延误传播特性的多模态数据集。

### 方法

提供三种对齐模态：(i) 包含丰富运营、气象和机场级别特征的表格数据集，涵盖超过500万次航班；(ii) 航班链模块，模拟沿连续航班航段的延误传播，捕捉上下游依赖关系；(iii) 航班网络图，编码共享的飞机、机组人员和机场资源连接，实现跨航班关系推理。

### 主要发现

数据集通过时间分割、全面特征和严格的防泄漏措施精心构建，支持真实且可复现的机器学习评估。

### 结论

Aeolus支持多种任务，包括回归、分类、时间结构建模和图学习，可作为表格、序列和图模态的统一基准，填补了领域特定建模和通用结构数据研究的关键空白。

### 翻译

我们介绍Aeolus，这是一个大规模多模态航班延误数据集，旨在推进航班延误预测研究并支持表格数据基础模型的开发。该领域现有数据集通常限于扁平表格结构，无法捕捉延误传播中固有的时空动态特性。Aeolus通过提供三种对齐模态解决了这一局限：(i) 包含丰富运营、气象和机场级别特征的表格数据集，涵盖超过500万次航班；(ii) 航班链模块，用于模拟沿连续航班航段的延误传播，捕捉上游和下游依赖关系；(iii) 航班网络图，编码共享的飞机、机组人员和机场资源连接，实现跨航班关系推理。该数据集通过时间分割、全面特征和严格的防泄漏措施精心构建，支持真实且可复现的机器学习评估。Aeolus支持多种任务，包括回归、分类、时间结构建模和图学习，可作为表格、序列和图模态的统一基准。我们发布了基线实验和预处理工具以促进采用。Aeolus填补了领域特定建模和通用结构数据研究的关键空白。我们的源代码和数据可在https://github.com/Flnny/Delay-data获取。


### 论文摘要

We introduce Aeolus, a large-scale Multi-modal Flight Delay Dataset designed to advance research on flight delay prediction and support the development of foundation models for tabular data. Existing datasets in this domain are typically limited to flat tabular structures and fail to capture the spatiotemporal dynamics inherent in delay propagation. Aeolus addresses this limitation by providing three aligned modalities: (i) a tabular dataset with rich operational, meteorological, and airportlevel features for over 50 million flights; (ii) a flight chain module that models delay propagation along sequential flight legs, capturing upstream and downstream dependencies; and (iii) a flight network graph that encodes shared aircraft, crew, and airport resource connections, enabling cross-flight relational reasoning. The dataset is carefully constructed with temporal splits, comprehensive features, and strict leakage prevention to support realistic and reproducible machine learning evaluation. Aeolus supports a broad range of tasks, including regression, classification, temporal structure modeling, and graph learning, serving as a unified benchmark across tabular, sequential, and graph modalities. We release baseline experiments and preprocessing tools to facilitate adoption. Aeolus fills a key gap for both domain-specific modeling and general-purpose structured data research.Our source code and data can be accessed at https://github.com/Flnny/Delay-data

---

## 25. Leveraging Foundation Models for Enhancing Robot Perception and Action

**论文链接:** [http://arxiv.org/abs/2510.26855v1](http://arxiv.org/abs/2510.26855v1)

**作者:** Reihaneh Mirjalili

**发布时间:** 2025-10-30

**备注:** Doctoral thesis

### GPT解析

### 总结

该论文研究了如何系统性地利用基础模型来增强机器人能力，特别是在非结构化环境中实现更有效的定位、交互和操作。研究围绕四个核心问题展开，共同构建了一个语义感知的机器人智能框架。

### 背景

基础模型在机器人领域的应用是一个新兴的研究方向，特别是在处理非结构化环境中的挑战时。

### 目的

探索基础模型如何被系统性地应用于机器人领域，以增强机器人在非结构化环境中的定位、交互和操作能力。

### 方法

研究围绕四个核心问题展开，每个问题解决机器人领域的一个基本挑战，共同构建一个语义感知的机器人智能框架。

### 主要发现

基础模型可以被系统性地利用来增强机器人在非结构化环境中的能力，包括更有效的定位、交互和操作。

### 结论

基础模型为增强机器人能力提供了新的途径，特别是在处理非结构化环境中的复杂任务时。

### 翻译

本论文研究了如何系统性地利用基础模型来增强机器人能力，使机器人在非结构化环境中能够实现更有效的定位、交互和操作。这项工作围绕四个核心问题展开，每个问题都解决了机器人领域的一个基本挑战，同时共同构建了一个语义感知的机器人智能框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何系统性地将基础模型（Foundation Models）集成到机器人系统中，以增强机器人在非结构化环境中的感知和动作能力。这个问题很重要，因为传统机器人系统依赖于特定任务模型，难以在真实世界的复杂、动态环境中泛化和适应。基础模型具有强大的语义理解和泛化能力，可以帮助机器人更好地解释复杂场景、适应新任务，并在变化的环境中灵活响应，最终实现更接近人类的灵活推理和行为。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统机器人系统在真实环境中的局限性，然后发现了基础模型（如GPT-3、CLIP等）的潜力。作者设计方法时借鉴了现有工作：在定位方面借鉴了语义信息增强方法，但使用了更强大的基础模型；在抓取方面借鉴了语言引导方法，但结合了大型语言模型和视觉语言模型；在分类方面借鉴了知识蒸馏技术；在视觉鲁棒性方面借鉴了视觉抽象方法。作者基于四个核心研究问题（定位、抓取、分类和操作）设计了四个方法，每个方法都利用基础模型的特定能力，并通过不同技术（如零样本推理、模型蒸馏）解决机器人面临的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '这篇论文提出了四个主要方法，每个针对不同机器人任务：1) FM-Loc使用基础模型生成语义图像描述符，通过CLIP检测对象、GPT-3生成房间标签、再次使用CLIP选择最可能的房间标签，最后计算查询与参考图像的语义相似度；2) Lan-grasp利用大型语言模型确定可抓取部分，视觉语言模型定位这些部分，并引入反馈机制动态调整抓取策略；3) VLM-Vac使用知识蒸馏将视觉语言模型能力转移到轻量级模型，并通过语言引导的经验回放实现持续学习；4) ARRO使用开放词汇分割和对象检测隔离任务相关组件，投影到虚拟背景上，减少视觉域变化的影响。整体流程都是先利用基础模型进行语义理解，然后根据具体任务设计相应的推理或决策机制。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) FM-Loc使用基础模型构建语义描述符，无需额外数据收集或微调；2) Lan-grasp结合语言模型和视觉模型进行语义抓取，引入反馈机制；3) VLM-Vac使用知识蒸馏和经验回放实现持续学习；4) ARRO提出视觉抽象方法提高视觉运动策略鲁棒性。相比之前工作，这些创新更系统地利用基础模型能力，注重语义理解和上下文感知，采用零样本推理等技术解决机器人挑战，并在真实环境中广泛评估，展示了实用性和有效性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性地将基础模型集成到机器人核心任务中，利用零样本推理、模型蒸馏和视觉抽象等技术，显著提高了机器人在非结构化环境中的语义理解、泛化能力和鲁棒性，为开发能够与人类在动态环境中可靠协作的自主机器人提供了新框架。'}


### 论文摘要

This thesis investigates how foundation models can be systematically leveraged to enhance robotic capabilities, enabling more effective localization, interaction, and manipulation in unstructured environments. The work is structured around four core lines of inquiry, each addressing a fundamental challenge in robotics while collectively contributing to a cohesive framework for semantics-aware robotic intelligence.

---

## 26. CYPRESS: Crop Yield Prediction via Regression on Prithvi's Encoder for Satellite Sensing

**论文链接:** [http://arxiv.org/abs/2510.26609v1](http://arxiv.org/abs/2510.26609v1)

**作者:** Shayan Nejadshamsi, Yuanyuan Zhang, Shadi Zaki, Brock Porth, Lysa Porth, Vahab Khoshdel

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文介绍了CYPRESS，一种基于深度学习的油菜籽产量预测模型，通过利用预训练的地理空间基础模型，将多时相卫星图像转换为高分辨率的像素级产量图，为精准农业提供了更实用的工具。

### 背景

准确及时的作物产量预测对全球粮食安全和现代农业管理至关重要，但传统方法缺乏精准农业所需的可扩展性和粒度。

### 目的

开发一种名为CYPRESS的深度学习模型，用于高分辨率、田间级别的油菜籽产量预测。

### 方法

CYPRESS利用预训练的大规模地理空间基础模型(Prithvi-EO-2.0-600M)，将其适配为连续回归任务，将多时相卫星图像转换为密集的像素级产量图。

### 主要发现

在加拿大草原综合数据集上评估，CYPRESS表现优于现有的基于深度学习的产量预测模型，证明了微调基础模型用于专业农业应用的有效性。

### 结论

CYPRESS提供连续、高分辨率的输出，比传统方法更实用；该工作弥合了大规模地球观测和农场决策之间的差距，为详细的农业监测提供了可扩展的解决方案。

### 翻译

准确及时的作物产量预测对全球粮食安全和现代农业管理至关重要。传统方法往往缺乏精准农业所需的可扩展性和粒度。本文介绍了CYPRESS（通过卫星传感的Prithvi编码器回归进行作物产量预测），这是一种专为高分辨率、田间级油菜籽产量预测设计的深度学习模型。CYPRESS利用预训练的大规模地理空间基础模型（Prithvi-EO-2.0-600M），并对其进行适配以用于连续回归任务，将多时相卫星图像转换为密集的像素级产量图。在加拿大草原综合数据集上的评估表明，CYPRESS优于现有的基于深度学习的产量预测模型，突显了微调基础模型用于专业农业应用的有效性。通过提供连续、高分辨率的输出，CYPRESS比传统的分类或县级聚合方法为精准农业提供了更实用的工具。这项工作验证了一种新颖的方法，弥合了大规模地球观测和农场决策之间的差距，为详细的农业监测提供了可扩展的解决方案。


### 论文摘要

Accurate and timely crop yield prediction is crucial for global food security and modern agricultural management. Traditional methods often lack the scalability and granularity required for precision farming. This paper introduces CYPRESS (Crop Yield Prediction via Regression on Prithvi's Encoder for Satellite Sensing), a deep learning model designed for high-resolution, intra-field canola yield prediction. CYPRESS leverages a pre-trained, large-scale geospatial foundation model (Prithvi-EO-2.0-600M) and adapts it for a continuous regression task, transforming multi-temporal satellite imagery into dense, pixel-level yield maps. Evaluated on a comprehensive dataset from the Canadian Prairies, CYPRESS demonstrates superior performance over existing deep learning-based yield prediction models, highlighting the effectiveness of fine-tuning foundation models for specialized agricultural applications. By providing a continuous, high-resolution output, CYPRESS offers a more actionable tool for precision agriculture than conventional classification or county-level aggregation methods. This work validates a novel approach that bridges the gap between large-scale Earth observation and on-farm decision-making, offering a scalable solution for detailed agricultural monitoring.

---

## 27. Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

**论文链接:** [http://arxiv.org/abs/2510.26585v1](http://arxiv.org/abs/2510.26585v1)

**作者:** Fulin Lin, Shaowen Chen, Ruishan Fang, Hongwei Wang, Tao Lin

**发布时间:** 2025-10-30

### GPT解析

### 总结

SupervisorAgent是一个轻量级和模块化的框架，用于运行时自适应监督，无需更改基础智能体的架构。它通过无LLM的自适应过滤器触发，在关键时刻进行干预，主动纠正错误，指导低效行为，并净化观测信息，从而显著减少多智能体系统中的令牌消耗，同时保持成功率。

### 背景

多智能体系统在处理复杂任务时表现出色，但随着操作复杂性的增加，它们的自主性增强会导致关键效率问题，如过度消耗令牌和因错误信息导致的失败。现有方法主要关注事后故障归因，缺乏主动的、实时干预来增强鲁棒性和效率。

### 目的

引入SupervisorAgent，一个轻量级和模块化的框架，用于运行时自适应监督，无需更改基础智能体的架构，以解决多智能体系统中的效率问题。

### 方法

SupervisorAgent由一个无LLM的自适应过滤器触发，在关键时刻进行干预，主动纠正错误，指导低效行为，并净化观测信息。

### 主要发现

在具有挑战性的GAIA基准测试中，SupervisorAgent将Smolagent框架的令牌消耗平均减少了29.45%，同时不损害其成功率。在五个额外的基准测试（数学推理、代码生成和问答）和各种最先进的基础模型上的广泛实验验证了我们方法的广泛适用性和鲁棒性。

### 结论

SupervisorAgent是一个有效的解决方案，可以解决多智能体系统中的效率问题，同时保持其性能，代码已开源。

### 翻译

虽然多智能体系统在复杂任务中表现出色，但随着操作复杂性的增加，它们的自主性增强常常导致关键效率问题，如过度消耗令牌和因错误信息导致的失败。现有方法主要关注事后故障归因，缺乏主动的、实时干预来增强鲁棒性和效率。为此，我们引入了SupervisorAgent，一个轻量级和模块化的框架，用于运行时自适应监督，无需更改基础智能体的架构。由无LLM的自适应过滤器触发，SupervisorAgent在关键时刻进行干预，主动纠正错误，指导低效行为，并净化观测信息。在具有挑战性的GAIA基准测试中，SupervisorAgent将Smolagent框架的令牌消耗平均减少了29.45%，同时不损害其成功率。在五个额外的基准测试（数学推理、代码生成和问答）和各种最先进的基础模型上的广泛实验验证了我们方法的广泛适用性和鲁棒性。代码可在https://github.com/LINs-lab/SupervisorAgent获取。


### 论文摘要

While Multi-Agent Systems (MAS) excel at complex tasks, their growing autonomy with operational complexity often leads to critical inefficiencies, such as excessive token consumption and failures arising from misinformation. Existing methods primarily focus on post-hoc failure attribution, lacking proactive, real-time interventions to enhance robustness and efficiency. To this end, we introduce SupervisorAgent, a lightweight and modular framework for runtime, adaptive supervision that operates without altering the base agent's architecture. Triggered by an LLM-free adaptive filter, SupervisorAgent intervenes at critical junctures to proactively correct errors, guide inefficient behaviors, and purify observations. On the challenging GAIA benchmark, SupervisorAgent reduces the token consumption of the Smolagent framework by an average of 29.45% without compromising its success rate. Extensive experiments across five additional benchmarks (math reasoning, code generation, and question answering) and various SoTA foundation models validate the broad applicability and robustness of our approach. The code is available at https://github.com/LINs-lab/SupervisorAgent.

---

## 28. MedSAE: Dissecting MedCLIP Representations with Sparse Autoencoders

**论文链接:** [http://arxiv.org/abs/2510.26411v1](http://arxiv.org/abs/2510.26411v1)

**作者:** Riccardo Renzulli, Colas Lepoutre, Enrico Cassano, Marco Grangetto

**发布时间:** 2025-10-30

### GPT解析

### 总结

研究通过应用医学稀疏自编码器(MedSAEs)提高医疗视觉中的机制可解释性，在保持高性能的同时增加了AI模型的透明度。

### 背景

人工智能在医疗保健领域需要既准确又可解释的模型，特别是在医疗视觉分析方面。

### 目的

通过将MedSAEs应用于MedCLIP的潜在空间来提高医疗视觉中的机制可解释性，MedCLIP是一种在胸部X光片和报告上训练的视觉-语言模型。

### 方法

提出一个结合相关性指标、熵分析和通过MedGEMMA基础模型进行自动神经元命名的评估框架，并在CheXpert数据集上进行实验。

### 主要发现

MedSAE神经元比原始MedCLIP特征实现了更高的单语义性和可解释性。

### 结论

研究结果连接了高性能医疗AI和透明度，为发展临床可靠表示提供了可扩展的步骤。

### 翻译

医疗保健中的人工智能需要准确且可解释的模型。我们通过将医学稀疏自编码器(MedSAEs)应用于MedCLIP的潜在空间，推进了医疗视觉中的机制可解释性，MedCLIP是一种在胸部X光片和报告上训练的视觉-语言模型。为了量化可解释性，我们提出一个结合相关性指标、熵分析和通过MedGEMMA基础模型进行自动神经元命名的评估框架。在CheXpert数据集上的实验表明，MedSAE神经元比原始MedCLIP特征具有更高的单语义性和可解释性。我们的研究结果连接了高性能医疗AI和透明度，为向临床可靠表示发展提供了可扩展的步骤。


### 论文摘要

Artificial intelligence in healthcare requires models that are accurate and interpretable. We advance mechanistic interpretability in medical vision by applying Medical Sparse Autoencoders (MedSAEs) to the latent space of MedCLIP, a vision-language model trained on chest radiographs and reports. To quantify interpretability, we propose an evaluation framework that combines correlation metrics, entropy analyzes, and automated neuron naming via the MedGEMMA foundation model. Experiments on the CheXpert dataset show that MedSAE neurons achieve higher monosemanticity and interpretability than raw MedCLIP features. Our findings bridge high-performing medical AI and transparency, offering a scalable step toward clinically reliable representations.

---

## 29. Towards Explainable and Reliable AI in Finance

**论文链接:** [http://arxiv.org/abs/2510.26353v1](http://arxiv.org/abs/2510.26353v1)

**作者:** Albi Isufaj, Pablo Mollá, Helmut Prendinger

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出了金融领域可解释和可靠AI的三种方法，包括Time-LLM模型使用提示避免错误预测、结合基础模型与可靠性估计器过滤不可靠预测，以及使用符号推理编码领域规则提供透明解释。

### 背景

金融预测越来越多地使用大型神经网络模型，但这些模型的透明度低，对信任和监管合规性提出了挑战。

### 目的

提出几种金融领域可解释和可靠AI的方法，以解决模型的透明度和可靠性问题。

### 方法

1) 描述Time-LLM（时间序列基础模型）如何使用提示来避免错误的方向性预测；2) 展示将时间序列预测的基础模型与可靠性估计器结合可以过滤不可靠的预测；3) 主张使用符号推理来编码领域规则，以提供透明的解释。

### 主要发现

这些方法强调只执行既可靠又可解释的预测；在股票和加密货币数据上的实验表明，该架构减少了误报并支持选择性执行。

### 结论

通过整合预测性能、可靠性估计和基于规则的推理，该框架推进了透明和可审计的金融AI系统。

### 翻译

金融预测越来越多地使用大型神经网络模型，但其不透明性对信任和监管合规性提出了挑战。我们提出了几种金融领域可解释和可靠AI的方法。首先，我们描述了Time-LLM（时间序列基础模型）如何使用提示来避免错误的方向性预测。其次，我们展示了将时间序列预测的基础模型与可靠性估计器结合可以过滤不可靠的预测。第三，我们主张使用符号推理来编码领域规则以提供透明的解释。这些方法强调只执行既可靠又可解释的预测。在股票和加密货币数据上的实验表明，该架构减少了误报并支持选择性执行。通过整合预测性能、可靠性估计和基于规则的推理，我们的框架推进了透明和可审计的金融AI系统。


### 论文摘要

Financial forecasting increasingly uses large neural network models, but their opacity raises challenges for trust and regulatory compliance. We present several approaches to explainable and reliable AI in finance. \emph{First}, we describe how Time-LLM, a time series foundation model, uses a prompt to avoid a wrong directional forecast. \emph{Second}, we show that combining foundation models for time series forecasting with a reliability estimator can filter our unreliable predictions. \emph{Third}, we argue for symbolic reasoning encoding domain rules for transparent justification. These approaches shift emphasize executing only forecasts that are both reliable and explainable. Experiments on equity and cryptocurrency data show that the architecture reduces false positives and supports selective execution. By integrating predictive performance with reliability estimation and rule-based reasoning, our framework advances transparent and auditable financial AI systems.

---

## 30. ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts

**论文链接:** [http://arxiv.org/abs/2510.26186v1](http://arxiv.org/abs/2510.26186v1)

**作者:** Jinho Choi, Hyesu Lim, Steffen Schneider, Jaegul Choo

**发布时间:** 2025-10-30

**备注:** Published in the Thirty-Ninth Conference on Neural Information  Processing Systems (NeurIPS 2025)

### GPT解析

### 总结

本文提出了ConceptScope框架，用于通过发现和量化人类可解释的概念来分析视觉数据集中的偏差。

### 背景

数据集偏差在机器学习数据集中普遍存在，但系统性识别这些偏差具有挑战性，因为需要昂贵的、细粒度的属性标注。

### 目的

开发一个可扩展且自动化的框架来分析视觉数据集，通过发现和量化人类可解释的概念来识别和量化数据集偏差。

### 方法

ConceptScope使用在视觉基础模型表示上训练的稀疏自编码器来发现和量化人类可解释的概念。根据概念与类标签的语义相关性和统计相关性，将概念分为目标、上下文和偏差类型，从而实现类级别的数据集特征描述、偏差识别和鲁棒性评估。

### 主要发现

ConceptScope能够捕获广泛的视觉概念，包括物体、纹理、背景、面部属性、情绪和动作。概念激活产生的空间归因与语义上有意义的图像区域一致。该框架能够可靠地检测已知偏差（如Waterbirds中的背景偏差）并发现先前未标注的偏差（如ImageNet中共现的物体）。

### 结论

ConceptScope为数据集审计和模型诊断提供了实用的工具。

### 翻译

数据集偏差在机器学习数据集无处不在，其中数据点偏向于某些概念。然而，在没有昂贵的细粒度属性标注的情况下，系统性地识别这些偏差具有挑战性。我们提出了ConceptScope，这是一个可扩展且自动化的框架，用于通过使用在视觉基础模型表示上训练的稀疏自编码器发现和量化人类可解释的概念来分析视觉数据集。ConceptScope根据概念与类标签的语义相关性和统计相关性将概念分为目标、上下文和偏差类型，从而通过基于概念的子分组实现类级别的数据集特征描述、偏差识别和鲁棒性评估。通过与标注数据集的比较，我们验证了ConceptScope能够捕获广泛的视觉概念，包括物体、纹理、背景、面部属性、情绪和动作。此外，我们表明概念激活产生的空间归因与语义上有意义的图像区域一致。ConceptScope可靠地检测了已知偏差（例如Waterbirds中的背景偏差）并发现了先前未标注的偏差（例如ImageNet中共现的物体），为数据集审计和模型诊断提供了实用的工具。


### 论文摘要

Dataset bias, where data points are skewed to certain concepts, is ubiquitous in machine learning datasets. Yet, systematically identifying these biases is challenging without costly, fine-grained attribute annotations. We present ConceptScope, a scalable and automated framework for analyzing visual datasets by discovering and quantifying human-interpretable concepts using Sparse Autoencoders trained on representations from vision foundation models. ConceptScope categorizes concepts into target, context, and bias types based on their semantic relevance and statistical correlation to class labels, enabling class-level dataset characterization, bias identification, and robustness evaluation through concept-based subgrouping. We validate that ConceptScope captures a wide range of visual concepts, including objects, textures, backgrounds, facial attributes, emotions, and actions, through comparisons with annotated datasets. Furthermore, we show that concept activations produce spatial attributions that align with semantically meaningful image regions. ConceptScope reliably detects known biases (e.g., background bias in Waterbirds) and uncovers previously unannotated ones (e.g, co-occurring objects in ImageNet), offering a practical tool for dataset auditing and model diagnostics.

---

## 31. Deep Neural Watermarking for Robust Copyright Protection in 3D Point Clouds

**论文链接:** [http://arxiv.org/abs/2510.27533v1](http://arxiv.org/abs/2510.27533v1)

**作者:** Khandoker Ashik Uz Zaman, Mohammad Zahangir Alam, Mohammed N. M. Ali, Mahdi H. Miraz

**发布时间:** 2025-10-31

**DOI:** 10.33166/AETiC.2025.05.002

### GPT解析

### 总结

本文提出了一种用于3D点云版权保护和所有权验证的鲁棒深度神经水印框架，通过奇异值分解将二进制水印嵌入到3D点云块中，并利用PointNet++神经网络架构实现水印的可靠提取。

### 背景

随着数字媒体中三维内容的快速增长，知识产权保护变得至关重要。与传统的图像或视频不同，3D点云在版权执行方面面临独特挑战，因为它们特别容易受到各种几何和非几何攻击的影响，这些攻击可以轻易降低或移除传统水印信号。

### 目的

解决3D点云在版权保护中的挑战，提出一种能够抵抗各种攻击的鲁棒深度神经水印框架，用于3D点云的版权保护和所有权验证。

### 方法

使用奇异值分解(SVD)将二进制水印嵌入到3D点云块的奇异值中，并利用PointNet++神经网络架构的深度学习提取能力。训练网络以在数据经过旋转、缩放、噪声、裁剪和信号失真等各种攻击后仍能可靠提取水印。

### 主要发现

在ModelNet40数据集上的验证表明，深度学习提取方法在具有挑战性的条件下显著优于传统的SVD技术。对于实验中最严重的几何失真——裁剪(70%)攻击，深度学习方法实现了0.83的位准确度和0.80的交并比(IoU)，而传统SVD方法仅实现了0.58的位准确度和0.26的IoU。

### 结论

该方法即使在严重失真条件下也能实现卓越的水印恢复并保持高保真度，证明了其在实际应用中的有效性。

### 翻译

随着数字媒体中三维内容的快速增长，知识产权保护变得至关重要。与传统的图像或视频不同，3D点云在版权执行方面面临独特挑战，因为它们特别容易受到各种几何和非几何攻击的影响，这些攻击可以轻易降低或移除传统水印信号。在本文中，我们通过提出一种用于3D点云版权保护和所有权验证的鲁棒深度神经水印框架来解决这些挑战。我们的方法使用奇异值分解(SVD)将二进制水印嵌入到3D点云块的奇异值中，并利用PointNet++神经网络架构的深度学习提取能力。网络经过训练，即使在数据经过旋转、缩放、噪声、裁剪和信号失真等各种攻击后也能可靠提取水印。我们使用公开的ModelNet40数据集验证了我们的方法，证明在具有挑战性的条件下，基于深度学习的提取显著优于传统的SVD技术。我们的实验评估表明，基于深度学习的提取方法显著优于现有的SVD方法，深度学习在裁剪(70%)攻击(实验中最严重的几何失真)下实现了0.83的位准确度和0.80的交并比(IoU)，而SVD仅实现了0.58的位准确度和0.26的IoU。这证明了我们的方法即使在严重失真条件下也能实现卓越的水印恢复并保持高保真度。


### 论文摘要

The protection of intellectual property has become critical due to the rapid growth of three-dimensional content in digital media. Unlike traditional images or videos, 3D point clouds present unique challenges for copyright enforcement, as they are especially vulnerable to a range of geometric and non-geometric attacks that can easily degrade or remove conventional watermark signals. In this paper, we address these challenges by proposing a robust deep neural watermarking framework for 3D point cloud copyright protection and ownership verification. Our approach embeds binary watermarks into the singular values of 3D point cloud blocks using spectral decomposition, i.e. Singular Value Decomposition (SVD), and leverages the extraction capabilities of Deep Learning using PointNet++ neural network architecture. The network is trained to reliably extract watermarks even after the data undergoes various attacks such as rotation, scaling, noise, cropping and signal distortions. We validated our method using the publicly available ModelNet40 dataset, demonstrating that deep learning-based extraction significantly outperforms traditional SVD-based techniques under challenging conditions. Our experimental evaluation demonstrates that the deep learning-based extraction approach significantly outperforms existing SVD-based methods with deep learning achieving bitwise accuracy up to 0.83 and Intersection over Union (IoU) of 0.80, compared to SVD achieving a bitwise accuracy of 0.58 and IoU of 0.26 for the Crop (70%) attack, which is the most severe geometric distortion in our experiment. This demonstrates our method's ability to achieve superior watermark recovery and maintain high fidelity even under severe distortions.

---

## 32. A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics

**论文链接:** [http://arxiv.org/abs/2510.27033v1](http://arxiv.org/abs/2510.27033v1)

**作者:** Simindokht Jahangard, Mehrzad Mohammadi, Abhinav Dhall, Hamid Rezatofighi

**发布时间:** 2025-10-30

### GPT解析

### 总结

提出了一种整合全景图像和3D点云信息的神经符号框架，结合神经感知与符号推理，用于解决视觉推理中的细粒度空间推理问题，在拥挤的人造环境中表现出优越性能和可靠性，同时保持轻量级设计。

### 背景

视觉推理，特别是空间推理，是一个具有挑战性的认知任务，需要理解物体关系及其在复杂环境中的交互。现有的视觉语言模型在感知任务上表现出色，但在细粒度空间推理方面存在困难，因为它们采用隐式、基于相关性的推理且仅依赖图像。

### 目的

提出一种新的神经符号框架，整合全景图像和3D点云信息，结合神经感知与符号推理，以明确建模空间和逻辑关系。

### 方法

框架由感知模块(用于检测实体和提取属性)和推理模块(构建结构化的场景图以支持精确、可解释的查询)组成。

### 主要发现

在JRDB-Reasoning数据集上评估，该方法在拥挤的人造环境中表现出优越的性能和可靠性，同时保持轻量级设计。

### 结论

该框架适用于机器人和具身AI应用。

### 翻译

视觉推理，特别是空间推理，是一项具有挑战性的认知任务，需要理解物体关系及其在复杂环境中的交互，尤其是在机器人领域。现有的视觉语言模型在感知任务上表现出色，但由于其隐式、基于相关性的推理和仅依赖图像的能力，在细粒度空间推理方面存在困难。我们提出了一种新的神经符号框架，整合全景图像和3D点云信息，结合神经感知与符号推理，以明确建模空间和逻辑关系。我们的框架由感知模块(用于检测实体和提取属性)和推理模块(构建结构化的场景图以支持精确、可解释的查询)组成。在JRDB-Reasoning数据集上的评估表明，该方法在拥挤的人造环境中表现出优越的性能和可靠性，同时保持轻量级设计，适用于机器人和具身AI应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人在复杂环境中进行细粒度空间推理的问题，特别是在理解物体和人类之间精确空间关系方面的挑战。这个问题很重要，因为空间推理是机器人导航和交互的基础能力，在人类建造的拥挤环境中，机器人需要准确理解多个实体之间的空间关系来完成各种任务，而现有模型在这方面存在明显不足，导致在机器人、导航和具身AI应用中可靠性不高。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视觉-语言模型(VLMs)在空间推理方面的局限性，包括其隐式推理方式、仅依赖2D图像而忽略3D信息、以及在相对人类定位等任务中的表现不佳。基于这些分析，作者设计了一个结合神经感知与符号推理的框架，借鉴了基础视觉-语言主干网络用于特征提取，以及场景图概念来表示实体关系。通过整合全景图像和3D点云信息，作者构建了一个能够进行显式几何和逻辑结构推理的系统，从而克服了现有模型的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合神经感知与符号推理，利用全景图像和3D点云信息进行更准确、可解释的空间推理，通过构建场景图作为中间推理层来支持精确查询。整体流程分为两个主要部分：感知部分和推理部分。感知部分包括特征提取模块(检测实体并提取属性)和投影模块(整合语义特征与几何关系)；推理部分是图搜索模块，包含句子解析(将查询转换为结构化表示)和搜索算法(在场景图上查找答案)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 多模态轻量级框架，整合全景图像和3D点云信息，参数少(1.3B)且适合机器人应用；2) 新颖的显式符号推理，通过构建场景图减少推理错误；3) 在拥挤环境中表现出优越的性能和可靠性。相比之前的工作，不同之处在于：结合了2D图像和3D点云信息而非仅依赖2D图像；使用显式符号推理结构而非隐式统计相关性；轻量级设计同时实现高级推理能力；特别擅长处理细粒度空间关系和复杂查询。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种轻量级多模态神经符号框架，通过整合全景图像和3D点云信息，结合神经感知与符号推理，显著提高了机器人在复杂环境中进行细粒度空间推理的能力，同时保持了模型的轻量化和可解释性。'}


### 论文摘要

Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications.

---

## 33. Active transfer learning for structural health monitoring

**论文链接:** [http://arxiv.org/abs/2510.27525v1](http://arxiv.org/abs/2510.27525v1)

**作者:** J. Poole, N. Dervilis, K. Worden, P. Gardner, V. Giglioni, R. S. Mills, A. J. Hughes

**发布时间:** 2025-10-31

**DOI:** 10.1016/j.ymssp.2025.113260

### GPT解析

### 总结

该研究提出了一种贝叶斯框架用于基于群体的结构健康监测中的域适应，结合主动采样策略提高数据效率，减少标记数据需求，降低结构运营成本。

### 背景

用于训练结构健康监测系统的数据通常昂贵且难以获取，特别是标记数据。基于群体的结构健康监测试图通过利用多个结构的数据解决这一问题，但不同结构的数据分布差异可能导致传统机器学习方法产生较大泛化误差。

### 目的

提出一种能够利用有限标记目标数据改进无监督域适应映射的贝叶斯框架，并将其与主动采样策略集成，指导检查选择最有信息量的观察结果进行标记。

### 方法

使用域适应技术对齐数据分布，提出贝叶斯框架用于群体结构健康监测中的域适应，集成主动采样策略指导检查，并在实验桥梁群体上评估该方法的有效性，包括多种损伤状态和环境条件的数据。

### 主要发现

结合迁移学习和主动学习可以在标记稀缺场景中提高学习分类模型的数据效率，对数据驱动的结构运营和维护有重要影响。

### 结论

通过采用所提出的方法，可以减少结构运营寿命内的检查次数，从而降低运营成本，同时保持有效的结构健康监测。

### 翻译

用于训练结构健康监测系统的数据通常昂贵且/或不切实际，特别是对于标记数据。基于群体的结构健康监测旨在通过利用来自多个结构的数据来解决这一限制。然而，来自不同结构的数据将遵循不同的分布，可能导致通过传统机器学习方法学习的模型产生较大的泛化误差。为了解决这个问题，可以采用迁移学习--以域适应的形式--来对齐数据分布。大多数先前的方法只考虑了无监督域适应，其中没有可用的标记目标数据；它们没有考虑如何将这些技术整合到在线框架中--随着在整个监测过程中获得标签而进行更新。本文提出了用于群体结构健康监测中域适应的贝叶斯框架，可以使用有限数量的标记目标数据改进无监督域适应映射。此外，该模型被集成到主动采样策略中，指导检查选择最有信息量的观察结果进行标记--从而进一步减少学习目标分类器所需的标记数据。该方法的有效性在一组实验桥梁群体上进行了评估。具体而言，该群体包括对应于多种损伤状态的数据，以及一套全面的环境条件。研究发现，结合迁移学习和主动学习可以在标记稀缺场景中提高学习分类模型时的数据效率。这一结果对数据驱动的结构运营和维护有影响，表明可以在结构的运营寿命内减少检查次数--从而降低运营成本。


### 论文摘要

Data for training structural health monitoring (SHM) systems are often expensive and/or impractical to obtain, particularly for labelled data. Population-based SHM (PBSHM) aims to address this limitation by leveraging data from multiple structures. However, data from different structures will follow distinct distributions, potentially leading to large generalisation errors for models learnt via conventional machine learning methods. To address this issue, transfer learning -- in the form of domain adaptation (DA) -- can be used to align the data distributions. Most previous approaches have only considered \emph{unsupervised} DA, where no labelled target data are available; they do not consider how to incorporate these technologies in an online framework -- updating as labels are obtained throughout the monitoring campaign. This paper proposes a Bayesian framework for DA in PBSHM, that can improve unsupervised DA mappings using a limited quantity of labelled target data. In addition, this model is integrated into an active sampling strategy to guide inspections to select the most informative observations to label -- leading to further reductions in the required labelled data to learn a target classifier. The effectiveness of this methodology is evaluated on a population of experimental bridges. Specifically, this population includes data corresponding to several damage states, as well as, a comprehensive set of environmental conditions. It is found that combining transfer learning and active learning can improve data efficiency when learning classification models in label-scarce scenarios. This result has implications for data-informed operation and maintenance of structures, suggesting a reduction in inspections over the operational lifetime of a structure -- and therefore a reduction in operational costs -- can be achieved.

---

## 34. pDANSE: Particle-based Data-driven Nonlinear State Estimation from Nonlinear Measurements

**论文链接:** [http://arxiv.org/abs/2510.27503v1](http://arxiv.org/abs/2510.27503v1)

**作者:** Anubhab Ghosh, Yonina C. Eldar, Saikat Chatterjee

**发布时间:** 2025-10-31

**备注:** 11 pages, 10 figures, under review at IEEE Transactions on Signal  Processing

### GPT解析

### 总结

本文提出了一种粒子数据驱动非线性状态估计方法(pDANSE)，用于处理状态转换模型未知的模型自由过程，通过循环神经网络和基于重参数化技巧的粒子采样方法处理非线性测量系统，实现了高效的状态估计。

### 背景

传统数据驱动非线性状态估计方法(DANSE)在处理线性测量系统时可获得状态后验的闭式解，但面对非线性测量系统时这种方法不再适用，需要开发新的解决方案。

### 目的

开发一种能够处理非线性测量系统的状态估计方法，避免使用计算密集的顺序蒙特卡洛(SMC)和祖先采样，并支持在有标签和无标签数据下分别进行半监督和无监督学习。

### 方法

使用循环神经网络(RNN)提供高斯先验参数描述模型自由过程状态，采用基于重参数化技巧的粒子采样方法处理非线性测量，估计状态后验的二阶统计量，并在随机Lorenz-63系统上验证性能。

### 主要发现

pDANSE能有效利用顺序测量避免计算密集方法，在立方非线性、相机模型非线性(无监督学习)以及半波整流非线性、笛卡尔到球面非线性(半监督学习)四种测量系统上均表现良好，性能与具有完整状态转换模型知识的粒子滤波器相当。

### 结论

pDANSE方法成功解决了模型自由过程的非线性测量状态估计问题，通过创新方法实现了高效准确的状态估计，为未知模型系统的状态估计提供了有效解决方案。

### 翻译

我们考虑设计一种数据驱动的非线性状态估计方法，该方法使用（带噪声的）非线性测量值来处理底层状态转换模型未知的过程。这样的过程被称为无模型过程。循环神经网络提供高斯先验的参数，该参数使用给定时间点的所有先前测量值来描述无模型过程的状态。在DANSE的情况下，测量系统是线性的，从而得到状态后验的闭式解。然而，非线性测量系统的存在使得闭式解不可行。相反，使用在时间点观察到的非线性测量值来计算状态后验的二阶统计量。我们使用基于重参数化技巧的粒子采样方法处理非线性测量，并估计状态后验的二阶统计量。所提出的方法被称为基于粒子的DANSE(pDANSE)。pDANSE的RNN有效利用顺序测量，避免了使用计算密集的顺序蒙特卡洛和/或祖先采样。我们描述了pDANSE的半监督学习方法，在没有标签数据的情况下过渡到无监督学习。使用随机Lorenz-63系统作为基准过程，我们实验性地展示了四种非线性测量系统的状态估计性能。我们探索了立方非线性和相机模型非线性，这里使用无监督学习；然后我们探索了半波整流非线性和笛卡尔到球面非线性，这里使用半监督学习。状态估计的性能被证明与具有完整Lorenz-63系统状态转换模型知识的粒子滤波器相比具有竞争力。


### 论文摘要

We consider the problem of designing a data-driven nonlinear state estimation (DANSE) method that uses (noisy) nonlinear measurements of a process whose underlying state transition model (STM) is unknown. Such a process is referred to as a model-free process. A recurrent neural network (RNN) provides parameters of a Gaussian prior that characterize the state of the model-free process, using all previous measurements at a given time point. In the case of DANSE, the measurement system was linear, leading to a closed-form solution for the state posterior. However, the presence of a nonlinear measurement system renders a closed-form solution infeasible. Instead, the second-order statistics of the state posterior are computed using the nonlinear measurements observed at the time point. We address the nonlinear measurements using a reparameterization trick-based particle sampling approach, and estimate the second-order statistics of the state posterior. The proposed method is referred to as particle-based DANSE (pDANSE). The RNN of pDANSE uses sequential measurements efficiently and avoids the use of computationally intensive sequential Monte-Carlo (SMC) and/or ancestral sampling. We describe the semi-supervised learning method for pDANSE, which transitions to unsupervised learning in the absence of labeled data. Using a stochastic Lorenz-$63$ system as a benchmark process, we experimentally demonstrate the state estimation performance for four nonlinear measurement systems. We explore cubic nonlinearity and a camera-model nonlinearity where unsupervised learning is used; then we explore half-wave rectification nonlinearity and Cartesian-to-spherical nonlinearity where semi-supervised learning is used. The performance of state estimation is shown to be competitive vis-\`a-vis particle filters that have complete knowledge of the STM of the Lorenz-$63$ system.

---

## 35. UNILocPro: Unified Localization Integrating Model-Based Geometry and Channel Charting

**论文链接:** [http://arxiv.org/abs/2510.27394v1](http://arxiv.org/abs/2510.27394v1)

**作者:** Yuhao Zhang, Guangjin Pan, Musa Furkan Keskin, Ossi Kaltiokallio, Mikko Valkama, Henk Wymeersch

**发布时间:** 2025-10-31

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本文提出了一种名为UNILocPro的统一定位框架，结合基于模型的定位和信道映射方法，用于处理混合视距/非视距场景。该框架通过自适应激活两种方法并使用多种损失函数进行无监督学习，显著提高了定位精度。同时提出了低复杂度的UNILoc实现，大幅降低训练复杂度而性能仅有轻微下降。

### 背景

在混合视距(LoS)/非视距(NLoS)场景中，单一的定位方法难以满足高精度需求。基于模型的定位方法和信道映射(Channel Charting, CC)方法各有优势，但单独使用时存在局限性。

### 目的

开发一种统一框架，有效结合基于模型的定位和信道映射方法，提高混合LoS/NLoS场景下的定位精度，同时降低训练复杂度。

### 方法

1. 提出UNILocPro框架，根据LoS/NLoS识别自适应激活基于模型和基于CC的方法；2. 利用基于模型方法的信息训练CC模型；3. 使用多种损失函数：成对距离损失、三元组损失(如果有时间戳)、基于LoS的损失和基于最优传输(OT)的损失；4. 提出低复杂度实现UNILoc，使用自生成标签训练CC模型，避免迭代Sinkhorn更新。

### 主要发现

1. 统一框架比单独的基于模型和基于CC的方法显著提高了定位精度；2. 带有时间戳的UNILocPro性能与完全监督的指纹识别相当，无需标记训练数据；3. UNILoc大幅降低训练复杂度，性能仅有轻微下降。

### 结论

UNILocPro框架有效结合了两种定位方法的优势，在混合LoS/NLoS场景中实现了高精度定位。UNLoc作为低复杂度实现，在实际应用中具有更好的实用性。

### 翻译

在本文中，我们提出了一种统一的定位框架（称为UNILocPro），该框架集成了基于模型的定位和信道映射（CC）方法，用于处理混合视距（LoS）/非视距（NLoS）场景。具体而言，基于LoS/NLoS识别，在基于模型和基于CC的方法之间进行自适应激活。针对无监督学习，利用基于模型方法获得的信息来训练CC模型，联合使用成对距离损失（涉及新的不相似度度量设计）、三元组损失（如果有时间戳）、基于LoS的损失和基于最优传输（OT）的损失，以保持全局几何结构。为了减少UNILocPro的训练复杂度，我们提出了一种低复杂度实现（称为UNILoc），其中CC模型使用通过单个预训练OT转换生成的自生成标签进行训练，避免了OT损失计算中涉及的迭代Sinkhorn更新。大量的数值实验表明，所提出的统一框架比基于模型和基于CC的方法显著提高了定位精度。值得注意的是，带有时间戳的UNILocPro性能与完全监督的指纹识别相当，尽管它不使用标记的训练数据。研究还表明，低复杂度的UNLoc可以显著减少训练复杂度，而性能仅有轻微下降。


### 论文摘要

In this paper, we propose a unified localization framework (called UNILocPro) that integrates model-based localization and channel charting (CC) for mixed line-of-sight (LoS)/non-line-of-sight (NLoS) scenarios. Specifically, based on LoS/NLoS identification, an adaptive activation between the model-based and CC-based methods is conducted. Aiming for unsupervised learning, information obtained from the model-based method is utilized to train the CC model, where a pairwise distance loss (involving a new dissimilarity metric design), a triplet loss (if timestamps are available), a LoS-based loss, and an optimal transport (OT)-based loss are jointly employed such that the global geometry can be well preserved. To reduce the training complexity of UNILocPro, we propose a low-complexity implementation (called UNILoc), where the CC model is trained with self-generated labels produced by a single pre-training OT transformation, which avoids iterative Sinkhorn updates involved in the OT-based loss computation. Extensive numerical experiments demonstrate that the proposed unified frameworks achieve significantly improved positioning accuracy compared to both model-based and CC-based methods. Notably, UNILocPro with timestamps attains performance on par with fully-supervised fingerprinting despite operating without labelled training data. It is also shown that the low-complexity UNILoc can substantially reduce training complexity with only marginal performance degradation.

---

## 36. Soft Task-Aware Routing of Experts for Equivariant Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.27222v1](http://arxiv.org/abs/2510.27222v1)

**作者:** Jaebyeong Jeon, Hyeonseo Jang, Jy-yong Sohn, Kibok Lee

**发布时间:** 2025-10-31

**备注:** NeurIPS 2025

### GPT解析

### 总结

这篇论文介绍了软任务感知路由（STAR）策略，用于解决等变表示学习和不变表示学习联合训练中的冗余特征学习问题。STAR通过将投影头建模为专家，促使它们专门捕捉共享信息或任务特定信息，从而减少冗余并提高模型效率。

### 背景

等变表示学习旨在捕捉输入变换在表示空间中引起的变异，而不变表示学习通过忽略这些变换来编码语义信息。最近研究表明，联合学习这两种表示对下游任务有益，通常通过使用单独的投影头实现。

### 目的

解决不变学习和等变学习联合训练中信息共享被忽略的问题，减少冗余特征学习，提高模型容量的利用效率。

### 方法

引入软任务感知路由（STAR）策略，将投影头建模为专家，促使它们专门捕捉共享信息或任务特定信息。

### 主要发现

STAR策略使不变嵌入和等变嵌入之间的标准相关性降低，减少了冗余特征学习。实验结果表明，STAR在各种迁移学习任务中实现了持续改进。

### 结论

STAR通过有效的任务感知路由策略，解决了等变和不变表示学习中的冗余问题，提高了模型效率，在各种迁移学习任务中展现了优越性能。

### 翻译

等变表示学习旨在捕捉输入变换在表示空间中引起的变异，而不变表示学习通过忽略这些变换来编码语义信息。最近研究表明，联合学习这两种表示通常对下游任务有益，通常通过使用单独的投影头实现。然而，这种设计忽略了不变学习和等变学习之间共享的信息，导致冗余的特征学习和模型容量的低效利用。为此，我们引入了软任务感知路由（STAR），这是一种针对投影头的路由策略，将它们建模为专家。STAR促使专家专门捕捉共享信息或任务特定信息，从而减少冗余的特征学习。我们通过观察不变嵌入和等变嵌入之间较低的标准相关性来验证这一效果。实验结果表明在各种迁移学习任务中都有持续改进。代码可在https://github.com/YonseiML/star获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决不变表示学习和等变表示学习之间的冗余特征学习问题。当使用两个独立投影头分别处理这两个任务时，它们会冗余地捕获共享信息，导致模型容量使用效率低下。这个问题很重要，因为不变和等变学习实际上是相互依赖的，而非完全独立，传统方法忽略这种依赖关系会导致模型性能受限，同时保留语义内容(不变性)和变换相关信息(等变性)对许多视觉任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先通过'陨石坑错觉'例子识别了不变和等变学习间的相互依赖关系，意识到传统双分支方法会导致冗余特征学习。为此，他们设计了两种形式的Soft Task-Aware Routing (STAR)：单一共享投影头(添加一个共享投影头提供共同嵌入)和MMoE投影(采用多门控混合专家架构动态分配专家)。作者借鉴了现有的混合专家框架，特别是MMoE架构，但创新性地将其仅限制在预训练阶段的投影头中，解决了传统MoE模型在跨任务转移中的可转移性问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过专家路由显式协调共享信息和任务特定信息，减少冗余特征学习。整体流程为：1)为输入图像生成两个增强视图；2)用共享编码器提取潜在表示；3)通过STAR投影模块生成不变和等变嵌入——SS版本使用三个专家(不变、等变和共享)输出相加，MMoE版本使用共享专家和任务特定路由器动态分配权重；4)等变学习通过预测器预测变换后的嵌入；5)使用对比损失训练模型；预训练后仅保留编码器用于下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)揭示不变和等变学习间的内在依赖性；2)提出STAR策略协调共享与任务特定信息；3)设计STAR的两种实现形式(单一共享投影和MMoE投影)；4)创新性地将MMoE限制在预训练阶段。相比EquiMod等传统方法，STAR显式建模任务间共享信息，减少冗余特征学习，能根据输入动态分配专家而非使用固定投影头，并在多种下游任务上取得更好性能，同时降低了不变和等变嵌入间的典型相关性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Soft Task-Aware Routing策略，通过专家路由显式协调不变和等变表示学习中的共享与任务特定信息，有效减少了冗余特征学习，并在多种下游任务上提升了表示学习的性能和泛化能力。'}


### 论文摘要

Equivariant representation learning aims to capture variations induced by input transformations in the representation space, whereas invariant representation learning encodes semantic information by disregarding such transformations. Recent studies have shown that jointly learning both types of representations is often beneficial for downstream tasks, typically by employing separate projection heads. However, this design overlooks information shared between invariant and equivariant learning, which leads to redundant feature learning and inefficient use of model capacity. To address this, we introduce Soft Task-Aware Routing (STAR), a routing strategy for projection heads that models them as experts. STAR induces the experts to specialize in capturing either shared or task-specific information, thereby reducing redundant feature learning. We validate this effect by observing lower canonical correlations between invariant and equivariant embeddings. Experimental results show consistent improvements across diverse transfer learning tasks. The code is available at https://github.com/YonseiML/star.

---

## 37. Functional Analysis of Loss-development Patterns in P&C Insurance

**论文链接:** [http://arxiv.org/abs/2510.27204v1](http://arxiv.org/abs/2510.27204v1)

**作者:** Arthur Charpentier, Qiheng Guo, Mike Ludkovski

**发布时间:** 2025-10-31

**备注:** 34 pages. Keywords: loss development; loss reserving; incremental  loss ratios; unsupervised learning; functional data; functional depth;  outlier detection; IBNR

### GPT解析

### 总结

这篇论文使用函数数据分析方法分析NAIC P计划损失三角形的损失发展，提出了一种基于偏最小二乘回归和函数自举的概率预测框架，能够提供更准确的函数预测区间。

### 背景

NAIC P计划损失三角形是保险业常用的损失发展分析工具，传统方法如链梯法在处理不确定性方面存在局限。

### 目的

研究增量损失比率的发展模式，识别异常曲线，并开发一种更准确的概率预测方法来估计未来损失发展。

### 方法

采用函数数据分析方法，将数据视为3300多条增量损失比率曲线；使用函数数据深度研究发展模式；提出基于偏最小二乘回归的函数模型来完成部分发展的曲线；结合函数自举量化不确定性。

### 主要发现

基于公司特定协变量可以识别发展模式的相似性和差异；能够识别异常的增量损失比率曲线；所提出的方法相比链梯法具有更好的概率评分。

### 结论

函数数据分析方法为损失发展分析提供了新的视角，所提出的概率预测框架能够提供更准确的函数预测区间，有助于保险公司更好地评估未来损失发展。

### 翻译

我们使用函数数据分析方法分析NAIC P计划损失三角形的损失发展。采用函数观点，我们的数据集包含24个事故年份中工人赔偿险种的3300多条增量损失比率曲线。依赖函数数据深度，我们首先基于公司特定协变量研究发展模式的相似性和差异，并识别异常的增量损失比率曲线。探索性发现激励了论文后半部分发展的概率预测框架。我们提出了一种函数模型，基于主成分分析得分的偏最小二乘回归来完成部分发展的增量损失比率曲线。结合上述方法与函数自举，使我们能够量化所有未来滞后的未来增量损失比率的不确定性。我们证明，与链梯法相比，我们的方法具有更好的概率评分，特别是能够提供准确的函数预测区间。


### 论文摘要

We analyze loss development in NAIC Schedule P loss triangles using functional data analysis methods. Adopting the functional viewpoint, our dataset comprises 3300+ curves of incremental loss ratios (ILR) of workers' compensation lines over 24 accident years. Relying on functional data depth, we first study similarities and differences in development patterns based on company-specific covariates, as well as identify anomalous ILR curves.   The exploratory findings motivate the probabilistic forecasting framework developed in the second half of the paper. We propose a functional model to complete partially developed ILR curves based on partial least squares regression of PCA scores. Coupling the above with functional bootstrapping allows us to quantify future ILR uncertainty jointly across all future lags. We demonstrate that our method has much better probabilistic scores relative to Chain Ladder and in particular can provide accurate functional predictive intervals.

---

## 38. Detecting Anomalies in Machine Learning Infrastructure via Hardware Telemetry

**论文链接:** [http://arxiv.org/abs/2510.26008v2](http://arxiv.org/abs/2510.26008v2)

**作者:** Ziji Chen, Steven W. D. Chien, Peng Qian, Noa Zilberman

**发布时间:** 2025-10-29

**备注:** 12 pages, 9 figures, submitted to nsdi 26

### GPT解析

### 总结

本文提出了Reveal系统，一种仅依赖硬件信号进行机器学习工作负载优化的方法，成功识别系统问题并加速模型性能。

### 背景

现代机器学习已发展为紧密结合的全栈生态系统，用户依赖云提供商获取弹性资源，但这些平台即服务使用虚拟化，导致运营商对用户工作负载了解有限。

### 目的

论证工作负载知识对于系统级优化不是必需的，提出一种仅依赖硬件信号的优化方法。

### 方法

提出Reveal系统，采用以硬件为中心的方法，使用从系统收集的低级信号，通过无监督学习流程检测异常。该流程基于30多种流行ML模型在各种硬件平台上的分析开发，确保对新兴工作负载的适应性。

### 主要发现

使用Reveal成功识别了网络和系统配置问题，加速了DeepSeek模型5.97%的性能。

### 结论

工作负载知识对于系统级优化不是必需的，通过仅依赖硬件信号，运营商可以实现有效的系统优化。

### 翻译

现代机器学习(ML)已发展为紧密结合的全栈生态系统，结合了硬件、软件、网络和应用。许多用户依赖云提供商提供弹性、隔离和成本高效的资源。不幸的是，这些平台即服务使用虚拟化，这意味着运营商对用户的工作负载了解有限。这阻碍了运营商进行资源优化，而资源优化对确保成本效率和最小化执行时间至关重要。在本文中，我们认为工作负载知识对于系统级优化不是必需的。我们提出了Reveal，它采用以硬件为中心的方法，仅依赖硬件信号-运营商完全可以访问。使用从系统收集的低级信号，Reveal通过无监督学习流程检测异常。该流程是通过分析各种硬件平台上超过30种流行ML模型开发的，确保对新兴工作负载和未知部署模式的适应性。使用Reveal，我们成功识别了网络和系统配置问题，将DeepSeek模型加速了5.97%。


### 论文摘要

Modern machine learning (ML) has grown into a tightly coupled, full-stack ecosystem that combines hardware, software, network, and applications. Many users rely on cloud providers for elastic, isolated, and cost-efficient resources. Unfortunately, these platforms as a service use virtualization, which means operators have little insight into the users' workloads. This hinders resource optimizations by the operator, which is essential to ensure cost efficiency and minimize execution time. In this paper, we argue that workload knowledge is unnecessary for system-level optimization. We propose Reveal, which takes a hardware-centric approach, relying only on hardware signals - fully accessible by operators. Using low-level signals collected from the system, Reveal detects anomalies through an unsupervised learning pipeline. The pipeline is developed by analyzing over 30 popular ML models on various hardware platforms, ensuring adaptability to emerging workloads and unknown deployment patterns. Using Reveal, we successfully identified both network and system configuration issues, accelerating the DeepSeek model by 5.97%.

---

## 39. ANCHOR: Integrating Adversarial Training with Hard-mined Supervised Contrastive Learning for Robust Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.27599v1](http://arxiv.org/abs/2510.27599v1)

**作者:** Samarup Bhattacharya, Anubhab Bhattacharya, Abir Chakraborty

**发布时间:** 2025-10-31

**备注:** 11 pages, 1 figure

### GPT解析

### 总结

本文提出了一种名为ANCHOR的框架，通过结合监督对比学习和硬正样本挖掘，提高神经网络模型对抗对抗性攻击的鲁棒性，同时保持较高的准确率。

### 背景

神经网络通过遵循梯度学习，逐步调整参数识别数据中的模式，但这种学习机制也使模型容易受到对抗性攻击，即通过微小、不可察觉的输入变化导致模型做出错误判断。

### 目的

开发一种能够学习更稳定、更有意义的模式表示的方法，使模型对对抗性攻击更加鲁棒，同时保持高准确率。

### 方法

提出ANCHOR框架，利用监督对比学习和显式硬正样本挖掘，使图像、其增强版本和扰动版本在嵌入空间中与同类图像聚类，同时与其他类别图像分离，从而专注于稳定模式而非脆弱梯度线索。

### 主要发现

在CIFAR-10数据集上，ANCHOR在干净和PGD-20攻击下的鲁棒准确性均优于标准对抗训练方法，表明结合对抗性指导与硬挖掘的对比监督有助于模型学习更结构化和鲁棒性的表示。

### 结论

结合对抗性指导和硬挖掘的对比监督可以有效缩小模型准确性和鲁棒性之间的差距，使神经网络能够更好地抵抗对抗性攻击。

### 翻译

神经网络改变了机器解释世界的方式。从根本上说，它们通过遵循梯度学习，逐步调整参数，直到识别出数据中最具判别性的模式。这一过程赋予了它们力量，但也打开了一个隐藏缺陷的大门。正是这些帮助模型学习的梯度，也可以用来产生微小、不可察觉的调整，导致模型完全改变其决策。这种调整被称为对抗性攻击。这些攻击通过向图像添加微小、不可察觉的变化来利用这一漏洞，这些变化虽然对人类眼睛来说是相同的，但会导致模型做出错误预测。在这项工作中，我们提出了对抗训练的对比性硬挖掘用于优化鲁棒性（ANCHOR）框架，该框架利用监督对比学习的力量，结合显式的硬正样本挖掘，使模型能够学习图像的表示，使图像、其增强版本和扰动版本在嵌入空间中与同一类别的其他图像聚类在一起，同时与其他类别的图像分离。这种对齐帮助模型专注于稳定、有意义的模式，而不是脆弱的梯度线索。在CIFAR-10上，我们的方法在干净和鲁棒准确性方面都取得了令人印象深刻的结果，在PGD-20（epsilon = 0.031）攻击下优于标准的对抗训练方法。我们的结果表明，将对抗性指导与硬挖掘的对比监督相结合，有助于模型学习更有结构和鲁棒性的表示，缩小了准确性和鲁棒性之间的差距。


### 论文摘要

Neural networks have changed the way machines interpret the world. At their core, they learn by following gradients, adjusting their parameters step by step until they identify the most discriminant patterns in the data. This process gives them their strength, yet it also opens the door to a hidden flaw. The very gradients that help a model learn can also be used to produce small, imperceptible tweaks that cause the model to completely alter its decision. Such tweaks are called adversarial attacks. These attacks exploit this vulnerability by adding tiny, imperceptible changes to images that, while leaving them identical to the human eye, cause the model to make wrong predictions. In this work, we propose Adversarially-trained Contrastive Hard-mining for Optimized Robustness (ANCHOR), a framework that leverages the power of supervised contrastive learning with explicit hard positive mining to enable the model to learn representations for images such that the embeddings for the images, their augmentations, and their perturbed versions cluster together in the embedding space along with those for other images of the same class while being separated from images of other classes. This alignment helps the model focus on stable, meaningful patterns rather than fragile gradient cues. On CIFAR-10, our approach achieves impressive results for both clean and robust accuracy under PGD-20 (epsilon = 0.031), outperforming standard adversarial training methods. Our results indicate that combining adversarial guidance with hard-mined contrastive supervision helps models learn more structured and robust representations, narrowing the gap between accuracy and robustness.

---

## 40. C-LEAD: Contrastive Learning for Enhanced Adversarial Defense

**论文链接:** [http://arxiv.org/abs/2510.27249v1](http://arxiv.org/abs/2510.27249v1)

**作者:** Suklav Ghosh, Sonal Kumar, Arijit Sur

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种利用对比学习进行对抗防御的新方法，通过同时优化模型参数和扰动，使网络学习鲁棒表示，实验结果表明该方法显著提高了模型对抗各种对抗性扰动的鲁棒性。

### 背景

深度神经网络在计算机视觉任务中取得了显著成功，但它们容易受到对抗性攻击，这种攻击只需对输入图像进行微小扰动就能导致错误预测。

### 目的

解决对抗性攻击问题，以便部署稳健的深度学习系统。

### 方法

提出了一种新颖的方法，利用对比学习进行对抗防御。该方法利用对比损失函数，通过同时使用干净和对抗性扰动的图像来增强分类模型的鲁棒性。通过同时优化模型参数和扰动，使网络学习对对抗攻击不太敏感的鲁棒表示。

### 主要发现

实验结果表明，模型对各种类型的对抗性扰动有显著改进。这表明对比损失有助于提取更具信息性和弹性的特征，有助于深度学习的对抗鲁棒性领域。

### 结论

对比学习在对抗防御方面是一个有前景的方向，能够提高模型对对抗攻击的鲁棒性。

### 翻译

深度神经网络在图像分类、分割和目标检测等计算机视觉任务中取得了显著成功。然而，它们容易受到对抗性攻击，这种攻击只需对输入图像进行微小扰动就能导致错误预测。解决这个问题对于部署稳健的深度学习系统至关重要。本文提出了一种新颖的方法，利用对比学习进行对抗防御，这是一个先前未被探索的领域。我们的方法利用对比损失函数，通过同时使用干净和对抗性扰动的图像来增强分类模型的鲁棒性。通过同时优化模型参数和扰动，我们的方法使网络学习对对抗攻击不太敏感的鲁棒表示。实验结果表明，模型对各种类型的对抗性扰动的鲁棒性有显著提高。这表明对比损失有助于提取更具信息性和弹性的特征，为深度学习的对抗鲁棒性领域做出了贡献。


### 论文摘要

Deep neural networks (DNNs) have achieved remarkable success in computer vision tasks such as image classification, segmentation, and object detection. However, they are vulnerable to adversarial attacks, which can cause incorrect predictions with small perturbations in input images. Addressing this issue is crucial for deploying robust deep-learning systems. This paper presents a novel approach that utilizes contrastive learning for adversarial defense, a previously unexplored area. Our method leverages the contrastive loss function to enhance the robustness of classification models by training them with both clean and adversarially perturbed images. By optimizing the model's parameters alongside the perturbations, our approach enables the network to learn robust representations that are less susceptible to adversarial attacks. Experimental results show significant improvements in the model's robustness against various types of adversarial perturbations. This suggests that contrastive loss helps extract more informative and resilient features, contributing to the field of adversarial robustness in deep learning.

---

## 41. IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.22706v3](http://arxiv.org/abs/2510.22706v3)

**作者:** Hao Li, Zhengyu Zou, Fangfu Liu, Xuanyang Zhang, Fangzhou Hong, Yukang Cao, Yushi Lan, Manyuan Zhang, Gang Yu, Dingwen Zhang, Ziwei Liu

**发布时间:** 2025-10-26

**备注:** https://github.com/lifuguan/IGGT_official

### GPT解析

### 总结

本文提出InstanceGrounded Geometry Transformer (IGGT)，一个端到端的大型统一transformer，用于统一空间重建和实例级上下文理解的知识。

### 背景

人类自然地将3D世界的几何结构和语义内容视为交织的维度，能够连贯准确地理解复杂场景。然而，大多数先前方法优先训练大型几何模型用于低级3D重建，并将高级空间理解孤立处理，忽视了这两个方面的关键互动，导致泛化能力有限和下游任务表现不佳。最近的尝试通过简单对齐3D模型与特定语言模型来缓解此问题，但限制了感知能力和下游任务适应性。

### 目的

开发一个能够统一空间重建和实例级上下文理解知识的端到端大型统一transformer模型。

### 方法

提出Instance Grounded Geometry Transformer (IGGT)，设计3D一致的对比学习策略，指导模型通过仅2D视觉输入编码具有几何结构和实例聚类的统一表示，支持将2D输入提升为具有明确不同对象实例的连贯3D场景。同时构建InsScene-15K数据集，包含高质量的RGB图像、姿态、深度图和3D一致的实例级掩码注释。

### 主要发现

通过统一几何结构和语义理解，可以改善3D场景分析的性能；仅通过2D视觉输入就能实现有效的3D重建和实例理解。

### 结论

IGGT模型能够有效统一几何结构和语义理解，通过3D一致的对比学习策略，仅从2D视觉输入就能生成具有明确对象实例的连贯3D场景。

### 翻译

人类自然地将3D世界的几何结构和语义内容视为交织的维度，能够连贯准确地理解复杂场景。然而，大多数先前方法优先训练大型几何模型用于低级3D重建，并将高级空间理解孤立处理，忽视了这两个3D场景分析基本方面之间的关键互动，从而限制了泛化能力，导致在下游3D理解任务中表现不佳。最近的尝试通过简单地将3D模型与特定语言模型对齐来缓解此问题，从而限制了感知能力并限制了下游任务的适应性。在本文中，我们提出了InstanceGrounded Geometry Transformer (IGGT)，一个端到端的大型统一transformer，用于统一空间重建和实例级上下文理解的知识。具体来说，我们设计了一种3D一致的对比学习策略，指导IGGT通过仅2D视觉输入编码具有几何结构和实例聚类的统一表示。这种表示支持将2D视觉输入一致地提升到具有明确不同对象实例的连贯3D场景。为促进此任务，我们进一步构建了InsScene-15K，一个通过新颖数据整理流程构建的大规模数据集，包含高质量的RGB图像、姿态、深度图和3D一致的实例级掩码注释。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景重建中几何结构和语义理解的分离问题。人类自然将几何结构和语义内容交织理解，而现有方法将这两者孤立处理，导致模型泛化能力差，在下游3D理解任务中表现不佳。这个问题在机器人操作、AR/VR和空间规划等应用中至关重要，这些应用需要同时理解场景的精确几何结构和丰富语义内容。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：分离处理几何重建和语义理解，或简单将3D模型与特定语言模型对齐，限制了模型适应性和性能。作者认为应通过联合训练将几何结构和实例级语义耦合，让模型自主学习两者关系。设计上借鉴了VGGT的统一Transformer架构，使用DINOv2提取特征，采用DPT-like架构进行密集预测，并利用SAM2进行数据标注。核心创新在于设计了3D一致的对比学习策略，确保跨视图的实例一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过联合训练将几何结构和实例级语义耦合，实现相互提升，并使用实例掩码作为桥接连接统一表示与各种视觉语言模型。整体流程：1) 构建InsScene-15K数据集；2) 使用大型统一变换器编码多视图图像为统一场景表示；3) 通过几何头部和实例头部分别预测几何结构和实例特征；4) 应用跨模态融合块增强实例特征的细粒度空间感知；5) 使用3D一致的对比学习策略训练模型；6) 通过实例掩码桥接策略支持下游任务如实例跟踪、开放词汇分割和QA场景定位。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 实例级几何-语义统一表示，通过联合训练实现相互提升；2) 3D一致的对比学习策略，确保跨视图实例一致性；3) 实例掩码桥接策略，实现与各种VLMs和LMMs的即插即用集成；4) 构建InsScene-15K大规模高质量数据集。相比之前工作：不同于分离处理几何和语义的方法，IGGT实现两者统一；不同于简单对齐特定语言模型的方法，IGT通过联合训练自主学习关系；不同于仅支持类别级特征的方法，IGT能区分同一类别中的不同实例；不绑定特定VLM，而是通过实例掩码灵活集成各种模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IGGT通过统一几何重建和实例级语义理解，并引入实例掩码桥接策略，实现了高质量的语义3D重建和灵活的下游任务支持，显著提升了3D场景理解的性能和泛化能力。'}


### 论文摘要

Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.

---

## 42. Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.27606v1](http://arxiv.org/abs/2510.27606v1)

**作者:** Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, Jiaqi Wang

**发布时间:** 2025-10-31

**备注:** preprint

### GPT解析

### 总结

Spatial-SSRL是一种自监督强化学习范式，通过从普通RGB或RGB-D图像中提取可验证信号，设计了五个捕捉2D和3D空间结构的预训练任务，显著提升了大型视觉语言模型的空间理解能力，无需昂贵的监督或专业工具。

### 背景

大型视觉语言模型(LVLMs)在空间理解方面存在弱点，现有的监督微调(SFT)和可验证奖励强化学习(RLVR)方法依赖昂贵的监督、专业工具或受限环境，限制了规模扩展。

### 目的

开发一种无需昂贵监督和专业工具的自强化学习范式，提升LVLMs的空间理解能力，同时保持通用视觉能力。

### 方法

提出Spatial-SSRL，一种自监督RL范式，自动设计五个预训练任务：打乱块重排序、翻转块识别、裁剪块修复、区域深度排序和相对3D位置预测，这些任务提供易于验证的真实答案，无需人工或LVLM注释。

### 主要发现

在七个空间理解基准测试中，Spatial-SSRL相比Qwen2.5-VL基线实现了显著提升：3B模型平均准确率提高4.63%，7B模型提高3.89%，同时保留了通用视觉能力。

### 结论

简单、内在的监督使大规模RLVR成为可能，为LVLMs提供更强的空间智能的实用途径，无需依赖昂贵的监督或专业工具。

### 翻译

空间理解仍然是大型视觉语言模型(LVLMs)的弱点。现有的监督微调(SFT)和最近的可验证奖励强化学习(RLVR)流程依赖于昂贵的监督、专业工具或受限环境，限制了规模扩展。我们引入了Spatial-SSRL，一种自监督RL范式，直接从普通RGB或RGB-D图像中派生可验证信号。Spatial-SSRL自动设计了五个捕捉2D和3D空间结构的预训练任务：打乱块重排序、翻转块识别、裁剪块修复、区域深度排序和相对3D位置预测。这些任务提供易于验证的真实答案，不需要人工或LVLM注释。在我们的任务上训练显著提高了空间推理能力，同时保留了通用视觉能力。在图像和视频设置下的七个空间理解基准测试中，Spatial-SSRL相比Qwen2.5-VL基线实现了平均准确率提升：3B模型提升4.63%，7B模型提升3.89%。我们的结果表明，简单、内在的监督使大规模RLVR成为可能，并为LVLMs提供更强的空间智能的实用途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大型视觉语言模型（LVLMs）在空间理解方面的不足问题。空间理解对LVLMs分析复杂真实世界场景至关重要，能够推理深度、距离、方位和相对物体位置，实现3D环境重建，并支持自动驾驶、机器人操作和具身导航等应用。尽管LVLMs在其他任务上表现优异，但其空间理解能力仍远低于人类水平，限制了它们在现实世界中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从现有方法的局限性出发，认识到监督微调（SFT）和可验证奖励强化学习（RLVR）依赖昂贵监督和专业工具的问题。他们借鉴了视觉自监督学习（SSL）的理念，认为普通RGB或RGB-D图像中固有的内在一致性信号可以自然地监督空间理解。作者设计了自监督任务作为可验证奖励函数，并使用组相对策略优化（GRPO）进行强化学习训练。这种方法结合了SSL的无监督特性和RLVR的优化优势，但创新性地将其应用于空间理解任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用图像内在结构作为自监督信号，生成可验证的奖励，通过强化学习优化LVLM的空间理解能力，无需人工标注或专业工具。整体流程包括：1) 设计五类自监督任务（三类无深度任务：打乱块重排、翻转块识别、裁剪块修复；两类基于深度任务：区域深度排序、相对3D位置预测）；2) 从COCO等数据集收集原始图像，自动构建Spatial-SSRL-81k数据集；3) 采用SFT冷启动后，使用GRPO进行强化学习训练，结合准确度奖励和格式奖励优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出Spatial-SSRL自监督强化学习新范式；2) 设计覆盖2D和3D空间结构的五类互补pretext任务；3) 构建完全自动生成的Spatial-SSRL-81k数据集；4) 结合SFT冷启动和GRPO优化。相比之前工作，这种方法不依赖昂贵标注或专业工具，避免了SFT的过拟合问题和RLVR的环境限制，同时将SSL从表示学习转移到行为优化，实现了更好的泛化能力和可扩展性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Spatial-SSRL通过图像内在结构自监督和强化学习，显著提升了LVLM的空间理解能力，同时保持通用视觉能力，且成本更低、可扩展性更强。'}


### 论文摘要

Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs.

---

## 43. NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding

**论文链接:** [http://arxiv.org/abs/2510.27481v1](http://arxiv.org/abs/2510.27481v1)

**作者:** Wei Xu, Cheng Wang, Dingkang Liang, Zongchuang Zhao, Xingyu Jiang, Peng Zhang, Xiang Bai

**发布时间:** 2025-10-31

**备注:** Accepted to NeurIPS 2025. Data and models are available at  https://github.com/H-EmbodVis/NAUTILUS

### GPT解析

### 总结

本研究构建了NautData数据集并提出视觉特征增强模块，开发了名为NAUTILUS的水下大语言模型，有效提高了水下场景理解能力。

### 背景

水下探索对了解地球和资源勘探、国家安全等应用至关重要，但水下场景理解需要多任务感知能力，而目前缺乏大规模水下多任务指令调整数据集。

### 目的

构建支持八种水下场景理解任务的大规模数据集，并开发能够提高水下场景理解鲁棒性的方法，解决水下图像退化挑战。

### 方法

构建包含145万张图像-文本对的NautData数据集；引入基于水下成像模型的物理先验，提出即插即用的视觉特征增强模块；将模块集成到LLaVA-1.5和Qwen2.5-VL基线模型中，构建NAUTILUS模型。

### 主要发现

VFE模块在NautData和公共水下数据集上实验证明有效，能够持续提高基线模型在大多数支持任务上的性能，确保了NAUTILUS在水下场景理解领域的优越性。

### 结论

通过大规模数据集构建和视觉特征增强模块提出，成功提高了水下场景理解性能，NAUTILUS模型在水下场景理解任务中表现优越。

### 翻译

水下探索为我们提供了了解地球的关键见解，并在资源勘探、国家安全等方面吸引越来越多的关注。我们研究水下场景理解方法，旨在实现水下探索的自动化。水下场景理解任务需要从多个粒度进行多任务感知。然而，缺乏大规模水下多任务指令调整数据集阻碍了这项研究的进展。为了弥补这一差距，我们构建了NautData，这是一个包含145万张图像-文本对的数据集，支持八种水下场景理解任务。它使水下场景理解模型的发展和全面评估成为可能。水下图像退化是一个广泛认可的问题，它干扰水下任务。为了提高水下场景理解的鲁棒性，我们引入了基于水下成像模型的物理先验，并提出了一种即插即用的视觉特征增强模块，该模块明确恢复了清晰的水下信息。我们将此模块集成到著名的基线模型LLaVA-1.5和Qwen2.5-VL中，构建了我们的水下大语言模型NAUTILUS。在NautData和公共水下数据集上进行的实验证明了VFE模块的有效性，持续提高了基线模型在大多数支持任务上的性能，从而确保了NAUTILUS在水下场景理解领域的优越性。数据和模型可在https://github.com/H-EmbodVis/NAUTILUS获取。


### 论文摘要

Underwater exploration offers critical insights into our planet and attracts increasing attention for its broader applications in resource exploration, national security, etc. We study the underwater scene understanding methods, which aim to achieve automated underwater exploration. The underwater scene understanding task demands multi-task perceptions from multiple granularities. However, the absence of large-scale underwater multi-task instruction-tuning datasets hinders the progress of this research. To bridge this gap, we construct NautData, a dataset containing 1.45 M image-text pairs supporting eight underwater scene understanding tasks. It enables the development and thorough evaluation of the underwater scene understanding models. Underwater image degradation is a widely recognized challenge that interferes with underwater tasks. To improve the robustness of underwater scene understanding, we introduce physical priors derived from underwater imaging models and propose a plug-and-play vision feature enhancement (VFE) module, which explicitly restores clear underwater information. We integrate this module into renowned baselines LLaVA-1.5 and Qwen2.5-VL and build our underwater LMM, NAUTILUS. Experiments conducted on the NautData and public underwater datasets demonstrate the effectiveness of the VFE module, consistently improving the performance of both baselines on the majority of supported tasks, thus ensuring the superiority of NAUTILUS in the underwater scene understanding area. Data and models are available at https://github.com/H-EmbodVis/NAUTILUS.

---

## 44. GeoFM: Enhancing Geometric Reasoning of MLLMs via Synthetic Data Generation through Formal Language

**论文链接:** [http://arxiv.org/abs/2510.27448v1](http://arxiv.org/abs/2510.27448v1)

**作者:** Yuhao Zhang, Dingxin Hu, Tinghao Yu, Hao Liu, Yiting Liu

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种名为GeoFM的新方法，用于合成高质量的几何数据，解决了多模态大语言模型在几何推理中的挑战。通过形式化语言探索度量空间内条件的组合，GeoFM能够生成多样化且正确的几何问题，实验证明其显著提升了模型在几何任务上的性能。

### 背景

多模态大语言模型在学术界和工业界因其处理多模态任务的能力而受到广泛关注。然而，由于高质量几何数据的稀缺，这些模型在数学几何推理方面面临挑战。

### 目的

解决多模态大语言模型在数学几何推理中的挑战，通过开发一种新的合成几何数据方法来提升模型性能。

### 方法

提出GeoFM方法，使用形式化语言探索度量空间内条件的组合，生成高保真度的几何问题，并通过符号引擎确保正确性。

### 主要发现

使用GeoFM合成数据训练的模型在MathVista几何问题解决任务上超越专有GPT-4o模型18.7%，在GeoQA上超越16.5%；在MathVista上超越领先开源模型5.7%，在GeoQA上超越2.7%。

### 结论

GeoFM方法能够生成高质量、多样化的几何数据，有效提升了模型在几何推理任务上的性能，显著优于现有方法。

### 翻译

多模态大语言模型在学术界和工业界因其处理多模态任务的能力而受到广泛关注。然而，由于高质量几何数据的稀缺，这些模型在数学几何推理方面面临挑战。为了解决这个问题，合成几何数据已成为一种必要策略。当前生成合成几何数据的方法包括重新表述或扩展现有问题，以及使用预定义规则和模板创建几何图像和问题。然而，这些方法往往产生的数据多样性不足或容易引入噪声。此外，现有方法合成的几何图像变化有限，与真实几何图差异显著。为了克服这些限制，我们提出了GeoFM，一种新的合成几何数据方法。GeoFM使用形式化语言探索度量空间内条件的组合，生成与原始问题不同但保持高保真度的几何问题，并通过符号引擎确保正确性。实验结果表明，我们的合成数据显著优于现有方法。使用我们数据训练的模型在MathVista的几何问题解决任务上超越专有GPT-4o模型18.7%，在GeoQA上超越16.5%。此外，它在MathVista上超越领先开源模型5.7%，在GeoQA上超越2.7%。


### 论文摘要

Multi-modal Large Language Models (MLLMs) have gained significant attention in both academia and industry for their capabilities in handling multi-modal tasks. However, these models face challenges in mathematical geometric reasoning due to the scarcity of high-quality geometric data. To address this issue, synthetic geometric data has become an essential strategy. Current methods for generating synthetic geometric data involve rephrasing or expanding existing problems and utilizing predefined rules and templates to create geometric images and problems. However, these approaches often produce data that lacks diversity or is prone to noise. Additionally, the geometric images synthesized by existing methods tend to exhibit limited variation and deviate significantly from authentic geometric diagrams. To overcome these limitations, we propose GeoFM, a novel method for synthesizing geometric data. GeoFM uses formal languages to explore combinations of conditions within metric space, generating high-fidelity geometric problems that differ from the originals while ensuring correctness through a symbolic engine. Experimental results show that our synthetic data significantly outperforms existing methods. The model trained with our data surpass the proprietary GPT-4o model by 18.7\% on geometry problem-solving tasks in MathVista and by 16.5\% on GeoQA. Additionally, it exceeds the performance of a leading open-source model by 5.7\% on MathVista and by 2.7\% on GeoQA.

---

## 45. Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs

**论文链接:** [http://arxiv.org/abs/2510.27517v1](http://arxiv.org/abs/2510.27517v1)

**作者:** Zherui Yang, Zhehao Li, Kangbo Lyu, Yixuan Li, Tao Du, Ligang Liu

**发布时间:** 2025-10-31

**备注:** NeurIPS 2025, poster

### GPT解析

### 总结

本研究提出了一种基于图神经网络(GNN)的稀疏近似逆(SPAI)预处理器方法，用于解决共轭梯度求解器在GPU上的并行化和长距离依赖问题，显著提高了求解速度和性能。

### 背景

共轭梯度(CG)求解器是求解对称正定线性系统Ax=b的常用方法，有效预处理器对快速收敛至关重要。传统预处理器依赖预设算法提供理论保证但限制数据优化能力，现有基于学习的方法利用GNN提高性能，但依赖不完全分解导致GPU并行化困难和长距离依赖建模挑战。

### 目的

开发一种GPU友好的学习型预处理器，特别是使用GNN构建稀疏近似逆(SPAI)预处理器，避免三角求解并减少每个CG步骤的计算量。

### 方法

使用GNN构建稀疏近似逆(SPAI)预处理器，每个CG步骤只需两次矩阵-向量乘积；利用矩阵-向量乘积的局部性与GNN局部传播机制的兼容性；引入基于统计的尺度不变损失函数，匹配CG收敛率取决于条件数而非绝对尺度的特性。

### 主要发现

在三个PDE导出数据集和一个合成数据集上评估，该方法在GPU上优于标准预处理器(对角线、IC和传统SPAI)及之前的基于学习预处理器；减少40%-53%求解时间(快68%-113%)，具有更好条件数和泛化性能。

### 结论

所提出的GNN构建的SPAI预处理器成功解决了现有方法在GPU并行化和长距离依赖建模方面的挑战，在GPU上实现了更快的求解时间和更好的性能，适用于广泛的应用场景。

### 翻译

共轭梯度求解器(CG)是求解对称正定线性系统Ax=b的常用方法，其中有效的预处理器对快速收敛至关重要。传统预处理器依赖于预设算法来提供严格的理论保证，同时限制了利用数据优化的能力。现有的基于学习的方法通常利用图神经网络(GNN)来提高性能和加速构建过程。然而，它们对不完全分解的依赖导致了重大挑战：相关的三角求解在实践中阻碍了GPU并行化，并引入了难以被GNN建模的长距离依赖关系。为解决这些问题，我们提出了一种基于学习的方法来生成GPU友好的预处理器，特别是使用GNN构建稀疏近似逆(SPAI)预处理器，避免了三角求解，每个CG步骤只需要两次矩阵-向量乘积。矩阵-向量乘积的局部性与GNN的局部传播机制兼容。GNN的灵活性也使我们的方法可以应用于广泛场景。此外，我们引入了一种基于统计的尺度不变损失函数，其设计匹配了CG的性质——收敛率取决于条件数，而不是A的绝对尺度，从而提高了学习到的预处理器的性能。在三个从PDE导出的数据集和一个合成数据集上的评估表明，我们的方法在GPU上优于标准预处理器(对角线、IC和传统SPAI)以及之前的基于学习的预处理器。我们在GPU上减少了40%-53%的求解时间(快68%-113%)，同时具有更好的条件数和优异的泛化性能。源代码可在https://github.com/Adversarr/LearningSparsePreconditioner4GPU获取。


### 论文摘要

The conjugate gradient solver (CG) is a prevalent method for solving symmetric and positive definite linear systems Ax=b, where effective preconditioners are crucial for fast convergence. Traditional preconditioners rely on prescribed algorithms to offer rigorous theoretical guarantees, while limiting their ability to exploit optimization from data. Existing learning-based methods often utilize Graph Neural Networks (GNNs) to improve the performance and speed up the construction. However, their reliance on incomplete factorization leads to significant challenges: the associated triangular solve hinders GPU parallelization in practice, and introduces long-range dependencies which are difficult for GNNs to model. To address these issues, we propose a learning-based method to generate GPU-friendly preconditioners, particularly using GNNs to construct Sparse Approximate Inverse (SPAI) preconditioners, which avoids triangular solves and requires only two matrix-vector products at each CG step. The locality of matrix-vector product is compatible with the local propagation mechanism of GNNs. The flexibility of GNNs also allows our approach to be applied in a wide range of scenarios. Furthermore, we introduce a statistics-based scale-invariant loss function. Its design matches CG's property that the convergence rate depends on the condition number, rather than the absolute scale of A, leading to improved performance of the learned preconditioner. Evaluations on three PDE-derived datasets and one synthetic dataset demonstrate that our method outperforms standard preconditioners (Diagonal, IC, and traditional SPAI) and previous learning-based preconditioners on GPUs. We reduce solution time on GPUs by 40%-53% (68%-113% faster), along with better condition numbers and superior generalization performance. Source code available at https://github.com/Adversarr/LearningSparsePreconditioner4GPU

---

## 46. Spectral Neural Graph Sparsification

**论文链接:** [http://arxiv.org/abs/2510.27474v1](http://arxiv.org/abs/2510.27474v1)

**作者:** Angelica Liguori, Ettore Ritacco, Pietro Sabatino, Annalisa Socievole

**发布时间:** 2025-10-31

### GPT解析

### 总结

论文提出了一种新的图表示学习框架——谱保持网络，通过生成简化的图作为原始图的忠实代理，使下游任务能够在降低计算成本的情况下进行。

### 背景

图是建模复杂系统的核心工具，应用于社交网络、分子化学和神经科学等领域。图神经网络已成为图学习的标准工具，但仍受限于固定结构的依赖和过平滑问题。

### 目的

提出谱保持网络框架，生成简化的图作为原始图的忠实代理，使社区检测、影响传播和信息扩散等下游任务能够在降低计算成本的情况下进行。

### 方法

谱保持网络引入两个关键组件：联合图进化层，同时变换图拓扑和节点特征矩阵，使结构和属性在层间自适应演化；谱一致性损失，通过强制保持图的谱特性和节点特征向量的一致性来正则化这些变换。

### 主要发现

通过节点级稀化分析评估了谱保持网络的有效性，使用成熟指标并与最先进方法进行基准测试，实验结果表明该方法具有优越的性能和明显优势。

### 结论

谱保持网络是一种有效的图表示学习方法，能够生成简化的图同时保持原始图的关键特性，在降低计算成本的同时，在下游任务中表现出色。

### 翻译

图是建模复杂系统的核心工具，应用于社交网络、分子化学和神经科学等领域。虽然图神经网络，特别是图卷积网络已成为图学习的标准工具，但它们仍然受限于对固定结构的依赖和过平滑问题。我们提出了谱保持网络，这是一种用于图表示学习的新框架，它生成简化的图作为原始图的忠实代理，使社区检测、影响传播和信息扩散等下游任务能够以降低的计算成本进行。谱保持网络引入了两个关键组件：联合图进化层和谱一致性损失。前者同时变换图拓扑和节点特征矩阵，使结构和属性在层间自适应演化，克服了静态邻域聚合的刚性。后者通过强制保持图的谱特性和节点特征向量的一致性来正则化这些变换。我们通过分析成熟的指标和与最先进方法进行基准测试，评估了谱保持网络在节点级稀化方面的有效性。实验结果表明我们的方法具有优越的性能和明显的优势。


### 论文摘要

Graphs are central to modeling complex systems in domains such as social networks, molecular chemistry, and neuroscience. While Graph Neural Networks, particularly Graph Convolutional Networks, have become standard tools for graph learning, they remain constrained by reliance on fixed structures and susceptibility to over-smoothing. We propose the Spectral Preservation Network, a new framework for graph representation learning that generates reduced graphs serving as faithful proxies of the original, enabling downstream tasks such as community detection, influence propagation, and information diffusion at a reduced computational cost. The Spectral Preservation Network introduces two key components: the Joint Graph Evolution layer and the Spectral Concordance loss. The former jointly transforms both the graph topology and the node feature matrix, allowing the structure and attributes to evolve adaptively across layers and overcoming the rigidity of static neighborhood aggregation. The latter regularizes these transformations by enforcing consistency in both the spectral properties of the graph and the feature vectors of the nodes. We evaluate the effectiveness of Spectral Preservation Network on node-level sparsification by analyzing well-established metrics and benchmarking against state-of-the-art methods. The experimental results demonstrate the superior performance and clear advantages of our approach.

---

## 47. Multi-Modal Feature Fusion for Spatial Morphology Analysis of Traditional Villages via Hierarchical Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.27208v1](http://arxiv.org/abs/2510.27208v1)

**作者:** Jiaxin Zhang, Zehong Zhu, Junye Deng, Yunqin Li, and Bowen Wang

**发布时间:** 2025-10-31

### GPT解析

### 总结

该研究提出了一种分层图神经网络模型，整合多源数据对村庄空间形态进行分析，解决了现有研究中的局限性，在多模态融合和分类任务上取得了显著性能提升。

### 背景

村庄区域在人地关系研究中具有重要性，但随着城市化发展，村庄空间特征逐渐消失，景观同质化问题突出。现有研究主要采用单一学科视角，过度依赖定性分析方法，且受限于数字基础设施不足和数据缺乏。

### 目的

为解决当前研究局限性，提出一种分层图神经网络模型，整合多源数据对村庄空间形态进行深入分析。

### 方法

提出包含两种节点（输入节点和通信节点）和两种边（静态输入边和动态通信边）的分层图神经网络模型。通过结合图卷积网络和图注意力网络，在两阶段特征更新机制下融合多模态特征，并引入关系池化机制，对17个子类型实施联合训练策略。

### 主要发现

实验结果表明，该方法在多模态融合和分类任务上比现有方法取得显著性能提升。所有子类型的联合优化使平均准确率/F1值从独立模型的0.71/0.83提升到0.82/0.90，其中地块任务带来了6%的提升。

### 结论

该方法为探索村庄空间格局和生成逻辑提供了科学依据。

### 翻译

村庄区域在研究人地关系中具有重要性。然而，随着城市化的推进，空间特征的逐渐消失和景观的同质化已成为突出问题。现有研究主要采用单一学科视角分析村庄空间形态及其影响因素，过度依赖定性分析方法。这些研究常受限于数字基础设施不足和数据缺乏。为解决当前研究局限性，本文提出了一种整合多源数据的分层图神经网络模型，对村庄空间形态进行深入分析。该框架包含两种节点类型（输入节点和通信节点）和两种边类型（静态输入边和动态通信边）。通过结合图卷积网络和图注意力网络，所提出的模型在两阶段特征更新机制下高效融合多模态特征。此外，基于现有的村庄空间形态分类原则，本文引入了关系池化机制，并对17个子类型实施了联合训练策略。实验结果表明，该方法在多模态融合和分类任务上比现有方法取得了显著的性能提升。此外，所有子类型的联合优化使平均准确率/F1值从独立模型的0.71/0.83提升到0.82/0.90，其中地块任务带来了6%的提升。我们的方法为探索村庄空间格局和生成逻辑提供了科学证据。


### 论文摘要

Villages areas hold significant importance in the study of human-land relationships. However, with the advancement of urbanization, the gradual disappearance of spatial characteristics and the homogenization of landscapes have emerged as prominent issues. Existing studies primarily adopt a single-disciplinary perspective to analyze villages spatial morphology and its influencing factors, relying heavily on qualitative analysis methods. These efforts are often constrained by the lack of digital infrastructure and insufficient data. To address the current research limitations, this paper proposes a Hierarchical Graph Neural Network (HGNN) model that integrates multi-source data to conduct an in-depth analysis of villages spatial morphology. The framework includes two types of nodes-input nodes and communication nodes-and two types of edges-static input edges and dynamic communication edges. By combining Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), the proposed model efficiently integrates multimodal features under a two-stage feature update mechanism. Additionally, based on existing principles for classifying villages spatial morphology, the paper introduces a relational pooling mechanism and implements a joint training strategy across 17 subtypes. Experimental results demonstrate that this method achieves significant performance improvements over existing approaches in multimodal fusion and classification tasks. Additionally, the proposed joint optimization of all sub-types lifts mean accuracy/F1 from 0.71/0.83 (independent models) to 0.82/0.90, driven by a 6% gain for parcel tasks. Our method provides scientific evidence for exploring villages spatial patterns and generative logic.

---

## 48. MDAS-GNN: Multi-Dimensional Spatiotemporal GNN with Spatial Diffusion for Urban Traffic Risk Forecasting

**论文链接:** [http://arxiv.org/abs/2510.27197v1](http://arxiv.org/abs/2510.27197v1)

**作者:** Ziyuan Gao

**发布时间:** 2025-10-31

### GPT解析

### 总结

本研究提出了一种基于多维注意力的空间扩散图神经网络(MDAS-GNN)，用于交通事故预测，整合了交通安全、基础设施和环境三个核心风险维度，在多个城市区域的数据集上表现出优越性能。

### 背景

交通事故是全球严重的公共健康挑战，每年导致超过135万人死亡。传统事故预测模型将道路段独立处理，无法捕捉城市交通网络中的复杂空间关系和时间依赖性。

### 目的

开发一种能够有效捕捉城市交通网络中复杂空间关系和时间依赖性的交通事故预测模型，为交通基础设施设计和城市规划提供数据支持。

### 方法

构建了MDAS-GNN模型，整合三个核心风险维度：交通安全、基础设施和环境风险。采用特定特征的空间扩散机制和多头时间注意力来捕捉不同时间跨度的依赖关系，并在英国交通部提供的事故数据上进行了验证。

### 主要发现

MDAS-GNN相比既定的基线方法表现出优越性能，在短期、中期和长期预测中都保持了较低的预测误差，特别是在长期预测方面表现出色。消融研究表明，集成的多维特征比单特征方法更有效，可将预测误差降低高达40%。

### 结论

该框架为土木工程师和城市规划者提供了先进的预测能力，支持交通基础设施设计，使决策者能够基于数据进行路网优化、基础设施资源改进和城市发展项目中的战略安全干预。

### 翻译

交通事故代表着一项关键的公共卫生挑战，每年在全球范围内造成超过135万人死亡。传统的事故预测模型将道路段独立处理，无法捕捉城市交通网络中复杂的空间关系和时间依赖性。本研究开发了MDAS-GNN，一种基于多维注意力的空间扩散图神经网络，整合了三个核心风险维度：交通安全、基础设施和环境风险。该框架采用特定特征的空间扩散机制和多头时间注意力来捕捉不同时间跨度的依赖关系。在英国交通部提供的伦敦中部、曼彻斯特南部和伯明翰东南部的事故数据评估中，MDAS-GNN相比既定的基线方法取得了优越的性能。该模型在短期、中期和长期预测中都保持了一致的低预测误差，特别是在长期预测方面具有特别优势。消融研究证实，集成的多维特征优于单特征方法，可将预测误差降低高达40%。该框架为土木工程师和城市规划者提供了先进的预测能力，用于交通基础设施设计，使他们能够为路网优化、基础设施资源改进和城市发展项目中的战略安全干预提供数据驱动的决策。


### 论文摘要

Traffic accidents represent a critical public health challenge, claiming over 1.35 million lives annually worldwide. Traditional accident prediction models treat road segments independently, failing to capture complex spatial relationships and temporal dependencies in urban transportation networks. This study develops MDAS-GNN, a Multi-Dimensional Attention-based Spatial-diffusion Graph Neural Network integrating three core risk dimensions: traffic safety, infrastructure, and environmental risk. The framework employs feature-specific spatial diffusion mechanisms and multi-head temporal attention to capture dependencies across different time horizons. Evaluated on UK Department for Transport accident data across Central London, South Manchester, and SE Birmingham, MDASGNN achieves superior performance compared to established baseline methods. The model maintains consistently low prediction errors across short, medium, and long-term periods, with particular strength in long-term forecasting. Ablation studies confirm that integrated multi-dimensional features outperform singlefeature approaches, reducing prediction errors by up to 40%. This framework provides civil engineers and urban planners with advanced predictive capabilities for transportation infrastructure design, enabling data-driven decisions for road network optimization, infrastructure resource improvements, and strategic safety interventions in urban development projects.

---

## 49. Relation-Aware Bayesian Optimization of DBMS Configurations Guided by Affinity Scores

**论文链接:** [http://arxiv.org/abs/2510.27145v1](http://arxiv.org/abs/2510.27145v1)

**作者:** Sein Kwon, Seulgi Baek, Hyunseo Yang, Youngwan Jo, Sanghyun Park

**发布时间:** 2025-10-31

**备注:** 13 pages

### GPT解析

### 总结

论文提出了RelTune框架，通过关系图表示参数依赖关系，并引入混合评分引导的贝叶斯优化方法，解决了现有DBMS参数自动调优方法的局限性，在多个DBMS和工作负载上实现了更快的收敛和更高的优化效率。

### 背景

数据库管理系统(DBMSs)对于管理大规模异构数据至关重要，其性能受配置参数影响。有效的参数调优对于适应不同工作负载、最大化吞吐量同时最小化延迟至关重要。

### 目的

解决现有自动配置优化方法的关键局限性，包括忽略参数间依赖关系、仅选择部分参数进行优化、贝叶斯优化依赖代理模型导致的预测不稳定和探索效率低下等问题。

### 方法

提出RelTune框架，将参数依赖关系表示为关系图，学习基于图神经网络(GNN)的潜在嵌入编码性能相关语义；引入混合评分引导的贝叶斯优化(HBO)，结合代理预测与亲和性评分，测量与先前高性能配置的接近度。

### 主要发现

在多个DBMS和工作负载上的实验结果表明，RelTune比传统基于贝叶斯优化的方法实现更快的收敛和更高的优化效率，在所有评估场景中取得了最先进的性能。

### 结论

RelTune框架有效解决了现有DBMS参数自动调优方法的关键局限性，通过考虑参数依赖关系和更全面的参数优化，实现了更优的性能。

### 翻译

数据库管理系统(DBMSs)是管理大规模异构数据的基础，其性能受配置参数的影响。有效调整这些参数对于适应不同的工作负载、最大化吞吐量同时最小化延迟至关重要。最近的研究集中在使用机器学习进行自动配置优化；然而，现有方法仍存在几个关键局限性。大多数调优框架忽略了参数之间的依赖关系，假设每个参数独立运作。这种简化限制了优化器利用参数间的关联效应，限制了其捕捉性能敏感交互的能力。此外，为降低高维搜索空间的复杂性，先前工作通常只选择前几个参数进行优化，忽略了其他对性能有重要贡献的参数。作为自动调优最常用的方法，贝叶斯优化(BO)也受限于其对代理模型的依赖，这可能导致预测不稳定和探索效率低下。为克服这些局限性，我们提出了RelTune，一种新框架，它将参数依赖关系表示为关系图，并学习基于图神经网络的潜在嵌入，编码性能相关的语义。RelTune进一步引入了混合评分引导的贝叶斯优化(HBO)，结合代理预测与亲和性评分，测量与先前高性能配置的接近度。在多个DBMS和工作负载上的实验结果表明，RelTune比传统的基于BO的方法实现更快的收敛和更高的优化效率，在所有评估场景中取得了最先进的性能。


### 论文摘要

Database Management Systems (DBMSs) are fundamental for managing large-scale and heterogeneous data, and their performance is critically influenced by configuration parameters. Effective tuning of these parameters is essential for adapting to diverse workloads and maximizing throughput while minimizing latency. Recent research has focused on automated configuration optimization using machine learning; however, existing approaches still exhibit several key limitations. Most tuning frameworks disregard the dependencies among parameters, assuming that each operates independently. This simplification prevents optimizers from leveraging relational effects across parameters, limiting their capacity to capture performancesensitive interactions. Moreover, to reduce the complexity of the high-dimensional search space, prior work often selects only the top few parameters for optimization, overlooking others that contribute meaningfully to performance. Bayesian Optimization (BO), the most common method for automatic tuning, is also constrained by its reliance on surrogate models, which can lead to unstable predictions and inefficient exploration. To overcome these limitations, we propose RelTune, a novel framework that represents parameter dependencies as a Relational Graph and learns GNN-based latent embeddings that encode performancerelevant semantics. RelTune further introduces Hybrid-Score-Guided Bayesian Optimization (HBO), which combines surrogate predictions with an Affinity Score measuring proximity to previously high-performing configurations. Experimental results on multiple DBMSs and workloads demonstrate that RelTune achieves faster convergence and higher optimization efficiency than conventional BO-based methods, achieving state-of-the-art performance across all evaluated scenarios.

---

## 50. A Cloud-Based Spatio-Temporal GNN-Transformer Hybrid Model for Traffic Flow Forecasting with External Feature Integration

**论文链接:** [http://arxiv.org/abs/2510.27039v1](http://arxiv.org/abs/2510.27039v1)

**作者:** Zhuo Zheng, Lingran Meng, Ziyu Lin

**发布时间:** 2025-10-30

### GPT解析

### 总结

本文提出了一种基于云的混合模型，结合时空图神经网络和Transformer架构，用于交通流量预测，该模型能有效捕捉复杂时空依赖关系并整合外部特征，在云平台上部署实现了可扩展性和实时适应性。

### 背景

准确的交通流量预测对智能交通系统的发展至关重要，支持交通信号优化、拥堵管理和路线规划等任务。传统模型往往无法有效捕捉大规模路网中的复杂时空依赖关系，特别是在天气、节假日和交通事故等外部因素的影响下。

### 目的

解决传统模型在捕捉大规模路网复杂时空依赖关系方面的不足，特别是在外部因素影响下的预测问题，开发一种能够有效整合空间和时间信息的混合模型，提高交通流量预测的准确性。

### 方法

提出一种基于云的混合模型，结合时空图神经网络和Transformer架构。该模型利用GNN在建模路网空间相关性方面的优势以及Transformer捕捉长期时间依赖关系的能力。通过特征融合整合外部上下文特征以提高预测准确性。模型部署在云计算平台上以实现可扩展性和实时适应性。

### 主要发现

实验评估显示，该模型在数据集上的表现优于基线方法，RMSE仅为17.92，MAE仅为10.53。这些发现表明混合GNN-Transformer方法为基于云的ITS应用提供了有效且可扩展的解决方案。

### 结论

混合GNN-Transformer方法为交通流量预测提供了方法论进步，并为拥堵缓解提供了实际应用价值，是一种有效且可扩展的基于云的ITS应用解决方案。

### 翻译

准确的交通流量预测对智能交通系统的发展至关重要，支持交通信号优化、拥堵管理和路线规划等任务。传统模型往往无法有效捕捉大规模路网中的复杂时空依赖关系，特别是在天气、节假日和交通事故等外部因素的影响下。为应对这一挑战，本文提出了一种基于云的混合模型，结合时空图神经网络和Transformer架构进行交通流量预测。该模型利用了GNN在建模路网空间相关性方面的优势以及Transformer捕捉长期时间依赖关系的能力。通过特征融合整合外部上下文特征以提高预测准确性。所提出的模型部署在云计算平台上以实现可扩展性和实时适应性。对数据集的实验评估显示，我们的模型优于基线方法，RMSE仅为17.92，MAE仅为10.53。这些发现表明，混合GNN-Transformer方法为基于云的ITS应用提供了有效且可扩展的解决方案，为交通流量预测提供了方法论进步，并为拥堵缓解提供了实际应用价值。


### 论文摘要

Accurate traffic flow forecasting is essential for the development of intelligent transportation systems (ITS), supporting tasks such as traffic signal optimization, congestion management, and route planning. Traditional models often fail to effectively capture complex spatial-temporal dependencies in large-scale road networks, especially under the influence of external factors such as weather, holidays, and traffic accidents. To address this challenge, this paper proposes a cloud-based hybrid model that integrates Spatio-Temporal Graph Neural Networks (ST-GNN) with a Transformer architecture for traffic flow prediction. The model leverages the strengths of GNNs in modeling spatial correlations across road networks and the Transformers' ability to capture long-term temporal dependencies. External contextual features are incorporated via feature fusion to enhance predictive accuracy. The proposed model is deployed on a cloud computing platform to achieve scalability and real-time adaptability. Experimental evaluation of the dataset shows that our model outperforms baseline methods (LSTM, TCN, GCN, pure Transformer) with an RMSE of only 17.92 and a MAE of only 10.53. These findings suggest that the hybrid GNN-Transformer approach provides an effective and scalable solution for cloud-based ITS applications, offering methodological advancements for traffic flow forecasting and practical implications for congestion mitigation.

---

## 51. Oral Tradition-Encoded NanyinHGNN: Integrating Nanyin Music Preservation and Generation through a Pipa-Centric Dataset

**论文链接:** [http://arxiv.org/abs/2510.26817v1](http://arxiv.org/abs/2510.26817v1)

**作者:** Jianbing Xiahou, Weixi Zhai, Xu Cui

**发布时间:** 2025-10-28

**备注:** 10 pages, 2 figures

### GPT解析

### 总结

本文提出了NanyinHGNN，一个用于生成南音乐器音乐的异构图网络模型。该模型通过构建以琵琶为中心的MIDI数据集和专门的标记化方法，将装饰音生成为异构图中的节点，结合图神经网络和规则系统生成真实的南音乐曲。

### 背景

南音是联合国教科文组织认可的无形文化遗产，它是一种以琵琶为中心的异声传统，核心旋律以传统记谱法记谱，而装饰音则通过口头传承，这对保存和当代创新都提出了挑战。

### 目的

解决南音音乐保存和创新中的挑战，特别是处理传统记谱与口头传承装饰音之间的差异。

### 方法

构建以琵琶为中心的MIDI数据集，开发NanyinTok作为专门的标记化方法，使用图转换器将符号序列转换为图结构，将装饰音生成重新定义为异构图中装饰音节点的创建，结合图神经网络生成旋律轮廓，并使用基于南音表演实践的规则系统完善装饰音。

### 主要发现

该模型成功生成了包含四种传统乐器的真实异声合奏，证明了将领域特定知识整合到模型架构中可以有效缓解民族音乐计算学中的数据稀缺挑战。

### 结论

整合领域特定知识到模型架构中可以有效解决民族音乐计算学中的数据稀缺问题，为南音等传统音乐的保存和创新提供了新方法。

### 翻译

我们提出了NanyinHGNN，一个用于生成南音乐器音乐的异构图网络模型。作为联合国教科文组织认可的无形文化遗产，南音遵循一种以琵琶为中心的异声传统，核心旋律以传统记谱法记谱，而装饰音则通过口头传承，这对保存和当代创新都提出了挑战。为解决这一问题，我们构建了一个以琵琶为中心的MIDI数据集，开发了NanyinTok作为专门的标记化方法，并使用图转换器将符号序列转换为图结构，以确保保留关键音乐特征。我们的主要创新是将装饰音生成为异构图中装饰音节点的创建。首先，图神经网络生成针对装饰音优化的旋律轮廓；然后，一个由南音表演实践指导的规则系统将这些轮廓完善为完整的装饰音，而在训练期间不需要明确的装饰音注释。实验结果表明，我们的模型成功生成了包含四种传统乐器的真实异声合奏。这些发现证明将领域特定知识整合到模型架构中可以有效缓解民族音乐计算学中的数据稀缺挑战。


### 论文摘要

We propose NanyinHGNN, a heterogeneous graph network model for generating Nanyin instrumental music. As a UNESCO-recognized intangible cultural heritage, Nanyin follows a heterophonic tradition centered around the pipa, where core melodies are notated in traditional notation while ornamentations are passed down orally, presenting challenges for both preservation and contemporary innovation. To address this, we construct a Pipa-Centric MIDI dataset, develop NanyinTok as a specialized tokenization method, and convert symbolic sequences into graph structures using a Graph Converter to ensure that key musical features are preserved. Our key innovation reformulates ornamentation generation as the creation of ornamentation nodes within a heterogeneous graph. First, a graph neural network generates melodic outlines optimized for ornamentations. Then, a rule-guided system informed by Nanyin performance practices refines these outlines into complete ornamentations without requiring explicit ornamentation annotations during training. Experimental results demonstrate that our model successfully generates authentic heterophonic ensembles featuring four traditional instruments. These findings validate that integrating domain-specific knowledge into model architecture can effectively mitigate data scarcity challenges in computational ethnomusicology.

---

## 52. Effect of Domain Generalization Techniques in Low Resource Systems

**论文链接:** [http://arxiv.org/abs/2510.27512v1](http://arxiv.org/abs/2510.27512v1)

**作者:** Mahi Aminu, Chisom Chibuike, Fatimo Adebanjo, Omokolade Awosanya, Samuel Oyeneye

**发布时间:** 2025-10-31

### GPT解析

### 总结

该研究探讨了资源有限情况下自然语言任务中的两种因果领域泛化技术，评估了它们在处理分布偏移问题上的有效性。

### 背景

机器学习模型通常假设训练和测试数据遵循相同分布，但现实中的分布偏移常导致这一假设失效。在资源有限的环境中，数据稀缺和领域多样性不足进一步阻碍了模型的稳健泛化能力。

### 目的

研究两种不同的因果领域泛化技术，提高资源有限自然语言任务中模型对分布偏移的鲁棒性。

### 方法

1) 因果数据增强(CDA)方法：自动生成反事实例子应用于NaijaSenti Twitter语料库的情感分类，通过添加语义等效的释义模拟受控分布偏移；2) 不变因果表征学习(ICRL)方法：使用DINER框架，将其适配到多语言情感分析任务中。

### 主要发现

两种方法都提高了对未见领域的鲁棒性：反事实数据增强在情感分类中带来了一致的跨域准确率提升；而使用DINER的因果表征学习在多语言情感分析中改善了分布外性能，尽管不同语言间的提升程度有所不同。

### 结论

因果领域泛化技术能有效提高资源有限情况下自然语言处理任务的跨域泛化能力，但不同方法在不同任务和语言中的效果存在差异。

### 翻译

机器学习模型通常假设训练和测试数据遵循相同分布，这一假设在现实场景中常因分布偏移而失效。这一问题在资源有限的环境中尤为突出，因为数据稀缺和有限的领域多样性阻碍了稳健的泛化。领域泛化(DG)方法通过学习跨领域保持不变的特征来解决这一挑战，通常使用因果机制提高模型的鲁棒性。在本研究中，我们考察了资源有限的自然语言任务中的两种不同因果DG技术。首先，我们研究了一种因果数据增强(CDA)方法，自动生成反事实例子以提高对虚假相关性的鲁棒性。我们将此方法应用于NaijaSenti Twitter语料库上的情感分类，通过添加语义等效的释义来模拟受控的分布偏移。其次，我们探索了使用DINER框架的不变因果表征学习(ICRL)方法，该方法最初用于消除基于方面的情感分析的偏见。我们将DINER适配到多语言环境中。我们的研究结果表明，两种方法都提高了对未见领域的鲁棒性：反事实数据增强在情感分类中带来了一致的跨域准确率提升，而使用DINER的因果表征学习在多语言情感分析中改善了分布外性能，尽管不同语言间的提升程度有所不同。


### 论文摘要

Machine learning models typically assume that training and test data follow the same distribution, an assumption that often fails in real-world scenarios due to distribution shifts. This issue is especially pronounced in low-resource settings, where data scarcity and limited domain diversity hinder robust generalization. Domain generalization (DG) approaches address this challenge by learning features that remain invariant across domains, often using causal mechanisms to improve model robustness. In this study, we examine two distinct causal DG techniques in low-resource natural language tasks. First, we investigate a causal data augmentation (CDA) approach that automatically generates counterfactual examples to improve robustness to spurious correlations. We apply this method to sentiment classification on the NaijaSenti Twitter corpus, expanding the training data with semantically equivalent paraphrases to simulate controlled distribution shifts. Second, we explore an invariant causal representation learning (ICRL) approach using the DINER framework, originally proposed for debiasing aspect-based sentiment analysis. We adapt DINER to a multilingual setting. Our findings demonstrate that both approaches enhance robustness to unseen domains: counterfactual data augmentation yields consistent cross-domain accuracy gains in sentiment classification, while causal representation learning with DINER improves out-of-distribution performance in multilingual sentiment analysis, albeit with varying gains across languages.

---

## 53. Interpretable Model-Aware Counterfactual Explanations for Random Forest

**论文链接:** [http://arxiv.org/abs/2510.27397v1](http://arxiv.org/abs/2510.27397v1)

**作者:** Joshua S. Harvey, Guanchao Feng, Sai Anusha Meesala, Tina Zhao, Dhagash Mehta

**发布时间:** 2025-10-31

**备注:** Presented at XAI-FIN-2025: International Joint Workshop on  Explainable AI in Finance: Achieving Trustworthy Financial Decision-Making;  November 15, 2025; Singapore

### GPT解析

### 总结

本文提出了一种基于随机森林的反事实解释方法，用于解决机器学习模型在受监管行业中解释性不足的问题。

### 背景

机器学习模型尽管具有强大的预测能力，但在受监管行业（如金融）中应用受限，因为它们提供解释的能力有限。现有的模型无关框架（如Shapley值）通常与所寻求的因果解释不一致。

### 目的

解决反事实案例搜索和解释的问题，生成更直观、可行且稀疏有用的解释。

### 方法

将反事实搜索和解释问题表述为相似性学习问题，利用随机森林预测模型本身学习的表示。找到反事实后，计算解释的特征重要性作为从原始实例到达反事实所穿越的随机森林分区的函数。

### 主要发现

该方法在MNIST手写数字数据集和德国信用数据集上生成的解释比Shapley值更稀疏和有用。

### 结论

基于随机森林的反事实解释方法能够提供更符合监管行业需求的解释性结果。

### 翻译

尽管机器学习模型具有巨大的预测能力，但由于它们提供解释的能力有限，通常不适合在金融等受监管行业应用。虽然像Shapley值这样的模型无关框架已被证明是方便且流行的，但它们很少符合通常所寻求的因果解释类型。反事实案例解释，即告知个人哪些情况需要不同才能导致结果变化，可能更直观和可行。然而，寻找合适的反事实案例是一个开放的挑战，同样解释哪些特征对结果变化最为关键也是如此。在这里，我们将反事实搜索和解释问题表述为相似性学习，利用随机森林预测模型本身学习的表示。一旦找到反事实，解释的特征重要性被计算为从原始实例到达它所穿越的随机森林分区的函数。我们在MNIST手写数字数据集和德国信用数据集上展示了这种方法，发现它生成的解释比Shapley值更稀疏和有用。


### 论文摘要

Despite their enormous predictive power, machine learning models are often unsuitable for applications in regulated industries such as finance, due to their limited capacity to provide explanations. While model-agnostic frameworks such as Shapley values have proved to be convenient and popular, they rarely align with the kinds of causal explanations that are typically sought after. Counterfactual case-based explanations, where an individual is informed of which circumstances would need to be different to cause a change in outcome, may be more intuitive and actionable. However, finding appropriate counterfactual cases is an open challenge, as is interpreting which features are most critical for the change in outcome. Here, we pose the question of counterfactual search and interpretation in terms of similarity learning, exploiting the representation learned by the random forest predictive model itself. Once a counterfactual is found, the feature importance of the explanation is computed as a function of which random forest partitions are crossed in order to reach it from the original instance. We demonstrate this method on both the MNIST hand-drawn digit dataset and the German credit dataset, finding that it generates explanations that are sparser and more useful than Shapley values.

---

## 54. Functional embeddings enable Aggregation of multi-area SEEG recordings over subjects and sessions

**论文链接:** [http://arxiv.org/abs/2510.27090v1](http://arxiv.org/abs/2510.27090v1)

**作者:** Sina Javadzadeh, Rahil Soroushmojdehi, S. Alireza Seyyed Mousavi, Mehrnaz Asadi, Sumiko Abe, Terence D. Sanger

**发布时间:** 2025-10-31

**备注:** Submitted to ICLR 2026

### GPT解析

### 总结

本研究提出了一种可扩展的表示学习框架，用于解决跨受试者颅内记录数据聚合的挑战。该框架通过学习电极的功能身份和建模区域间关系，实现了受试者无关的神经数据处理，为大规模跨受试者数据分析和预训练提供了新途径。

### 背景

跨受试者颅内记录数据聚合面临电极数量、位置和覆盖区域差异大的挑战。传统空间归一化方法（如MNI坐标）虽然提供了解剖参考，但往往无法捕捉真正的功能相似性，尤其在定位不精确时，即使匹配解剖坐标，不同个体的目标脑区域和神经动力学也可能存在显著差异。

### 目的

开发一种可扩展的表示学习框架，学习受试者无关的电极功能身份，并建模区域间关系，以实现跨受试者颅内神经数据的有效聚合和分析，特别是在缺乏严格任务结构和均匀传感器布局的情况下。

### 方法

研究提出了一种双阶段框架：(1) 使用具有对比目标的孪生编码器，从多区域局部场电位中学习每个电极的受试者无关功能身份，诱导对区域特定神经签名具有局部敏感性的嵌入几何结构；(2) 将这些嵌入标记化，用于变换器模型，使用可变数量的通道建模区域间关系。研究在包含20名受试者的基底节-丘脑区域数据集上评估了该方法，这些数据集来自灵活的休息/运动记录会话，具有异构的电极布局。

### 主要发现

学习到的功能空间支持准确的受试者内辨别，形成清晰、区域一致的聚类，并能零样本迁移到未见过的通道。变换器在功能标记上操作，无需受试者特定的头部或监督，能够捕获跨区域依赖关系并重建被屏蔽的通道，为下游解码提供了受试者无关的主干。

### 结论

该研究为颅内神经数据在缺乏严格任务结构和均匀传感器布局情况下的跨受试者聚合和预训练提供了一条新途径，有望促进大规模神经数据分析的发展。

### 翻译

跨受试者颅内记录数据聚合具有挑战性，因为电极数量、放置位置和覆盖区域差异很大。像MNI坐标这样的空间归一化方法提供了共享的解剖参考，但往往无法捕捉真正的功能相似性，尤其是在定位不精确时；即使在匹配的解剖坐标上，不同个体之间的目标脑区域和潜在的神经动力学也可能存在显著差异。我们提出了一种可扩展的表示学习框架，该框架(i)使用具有对比目标的孪生编码器，从多区域局部场电位中学习每个电极的受试者无关功能身份，诱导一种对区域特定神经签名具有局部敏感性的嵌入几何结构，以及(ii)将这些嵌入标记化，用于一个变换器，该变换器使用可变数量的通道建模区域间关系。我们在一个包含20名受试者的数据集上评估了该框架，这些数据集涵盖了基底节-丘脑区域，是在灵活的休息/运动记录会话期间收集的，具有异构的电极布局。学习到的功能空间支持准确的受试者内辨别，并形成清晰、区域一致的聚类；它可以零样本迁移到未见过的通道。变换器在功能标记上操作，无需受试者特定的头部或监督，能够捕获跨区域依赖关系，并重建被屏蔽的通道，为下游解码提供了受试者无关的主干。这些结果表明，在严格的任务结构和均匀传感器布局不可用的情况下，为颅内神经数据实现大规模、跨受试者聚合和预训练提供了一条途径。


### 论文摘要

Aggregating intracranial recordings across subjects is challenging since electrode count, placement, and covered regions vary widely. Spatial normalization methods like MNI coordinates offer a shared anatomical reference, but often fail to capture true functional similarity, particularly when localization is imprecise; even at matched anatomical coordinates, the targeted brain region and underlying neural dynamics can differ substantially between individuals. We propose a scalable representation-learning framework that (i) learns a subject-agnostic functional identity for each electrode from multi-region local field potentials using a Siamese encoder with contrastive objectives, inducing an embedding geometry that is locality-sensitive to region-specific neural signatures, and (ii) tokenizes these embeddings for a transformer that models inter-regional relationships with a variable number of channels. We evaluate this framework on a 20-subject dataset spanning basal ganglia-thalamic regions collected during flexible rest/movement recording sessions with heterogeneous electrode layouts. The learned functional space supports accurate within-subject discrimination and forms clear, region-consistent clusters; it transfers zero-shot to unseen channels. The transformer, operating on functional tokens without subject-specific heads or supervision, captures cross-region dependencies and enables reconstruction of masked channels, providing a subject-agnostic backbone for downstream decoding. Together, these results indicate a path toward large-scale, cross-subject aggregation and pretraining for intracranial neural data where strict task structure and uniform sensor placement are unavailable.

---

## 55. Incremental Human-Object Interaction Detection with Invariant Relation Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.27020v1](http://arxiv.org/abs/2510.27020v1)

**作者:** Yana Wei, Zeen Chi, Chongyu Wang, Yu Wu, Shipeng Yan, Yongfei Liu, Xuming He

**发布时间:** 2025-10-30

### GPT解析

### 总结

这篇论文提出了一个名为'无样本增量关系蒸馏'(IRD)的框架，用于解决开放环境中人类-物体交互(HOI)的增量检测问题，以应对动态环境中的交互漂移和零样本HOI组合检测挑战。

### 背景

在开放世界环境中，人类-物体交互(HOI)不断演变，这挑战了传统的封闭世界HOI检测模型。人类具有渐进式获取知识的能力，而现有的增量学习模型面临灾难性遗忘问题，以及交互漂移和零样本HOI组合检测的特殊挑战。

### 目的

开发能够识别动态环境中人类-物体关系的智能体，即增量HOI检测(IHOID)，以解决增量学习中的灾难性遗忘问题，以及交互漂移和零样本HOI组合检测的挑战。

### 方法

提出了一个无样本增量关系蒸馏(IRD)框架。IRD将物体和关系的学习解耦，并引入了两种独特的蒸馏损失，用于学习在不同HOI组合中共享相同关系的不变关系特征。

### 主要发现

在HICO-DET和V-COCO数据集上的大量实验表明，该方法在减轻遗忘、增强对交互漂移的鲁棒性以及零样本HOI的泛化能力方面优于最先进的基线方法。

### 结论

IRD框架有效地解决了开放世界环境中HOI检测的增量学习挑战，特别是在处理动态交互和零样本场景方面表现出色。

### 翻译

在开放世界环境中，人类-物体交互(HOI)不断演变，这对传统的封闭世界HOI检测模型提出了挑战。受人类渐进式获取知识能力的启发，我们探索了增量HOI检测(IHOID)，以开发能够识别此类动态环境中人类-物体关系的智能体。这种设置不仅面临增量学习中常见的灾难性遗忘问题，还面临由交互漂移和检测具有连续到达数据的零样本HOI组合带来的特殊挑战。因此，我们提出了一个新颖的无样本增量关系蒸馏(IRD)框架。IRD将物体和关系的学习解耦，并引入了两种独特的蒸馏损失，用于学习在不同HOI组合中共享相同关系的不变关系特征。在HICO-DET和V-COCO数据集上的大量实验证明了我们的方法在减轻遗忘、增强对交互漂移的鲁棒性以及零样本HOI的泛化能力方面优于最先进的基线方法。代码可在以下网址获取：https://github.com/weiyana/ContinualHOI


### 论文摘要

In open-world environments, human-object interactions (HOIs) evolve continuously, challenging conventional closed-world HOI detection models. Inspired by humans' ability to progressively acquire knowledge, we explore incremental HOI detection (IHOID) to develop agents capable of discerning human-object relations in such dynamic environments. This setup confronts not only the common issue of catastrophic forgetting in incremental learning but also distinct challenges posed by interaction drift and detecting zero-shot HOI combinations with sequentially arriving data. Therefore, we propose a novel exemplar-free incremental relation distillation (IRD) framework. IRD decouples the learning of objects and relations, and introduces two unique distillation losses for learning invariant relation features across different HOI combinations that share the same relation. Extensive experiments on HICO-DET and V-COCO datasets demonstrate the superiority of our method over state-of-the-art baselines in mitigating forgetting, strengthening robustness against interaction drift, and generalization on zero-shot HOIs. Code is available at \href{https://github.com/weiyana/ContinualHOI}{this HTTP URL}

---

## 56. FOCUS: Efficient Keyframe Selection for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.27280v1](http://arxiv.org/abs/2510.27280v1)

**作者:** Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, Yang You

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种名为FOCUS的关键帧选择方法，用于解决多模态大语言模型处理长视频时的标记预算问题，在处理不到2%视频帧的情况下实现了显著的准确性提升。

### 背景

多模态大语言模型将图像和视频帧表示为视觉标记，但扩展到长视频时标记预算会远超实际限制。现有方法要么均匀采样，要么使用小模型进行关键帧选择，但这些方法依赖预过滤且可能错过重要信息。

### 目的

开发一种无需训练、与模型无关的关键帧选择模块，在严格标记预算下选择与查询相关的帧，避免错过重要信息。

### 方法

FOCUS将关键帧选择表述为多臂老虎机中的组合纯探索问题，将短时间片段视为臂，使用经验均值和伯恩斯坦置信半径识别信息区域，同时保留对不确定区域的探索。采用两阶段探索-利用程序，先识别高价值时间区域，再在每个区域内选择最高分帧。

### 主要发现

在两个长视频问答基准测试中，FOCUS处理不到2%的视频帧实现了显著的准确性提升。对于超过20分钟的视频，在LongVideoBench上实现了11.9%的准确率提升。

### 结论

FOCUS作为关键帧选择方法的有效性得到验证，为MLLMs可扩展的长视频理解提供了一种简单通用的解决方案。

### 翻译

多模态大语言模型(MLLMs)将图像和视频帧表示为视觉标记。然而，从单图像扩展到数小时长的视频会使标记预算远超实际限制。因此，常用方法要么均匀下采样，要么使用较小的视觉语言模型进行基于检索评分的关键帧选择。然而，这些关键帧选择方法仍然依赖选择前的预过滤来降低推理成本，并可能错过最具信息量的时刻。我们提出FOCUS，即Frame-Optimistic Confidence Upper-bound Selection，这是一种无需训练、与模型无关的关键帧选择模块，在严格的标记预算下选择与查询相关的帧。FOCUS将关键帧选择表述为多臂老虎机中的组合纯探索问题：它将短时间片段视为臂，并使用经验均值和伯恩斯坦置信半径来识别信息丰富的区域，同时保留对不确定区域的探索。 resulting两阶段探索-利用程序从具有理论保证的顺序策略简化而来，首先识别高价值时间区域，然后在每个区域内选择得分最高的帧。在两个长视频问答基准测试中，FOCUS在处理不到2%的视频帧的同时提供了显著的准确性提升。对于超过20分钟的视频，它在LongVideoBench上实现了11.9%的准确率提升，证明了其作为关键帧选择方法的有效性，并为MLLMs可扩展的长视频理解提供了一种简单通用的解决方案。


### 论文摘要

Multimodal large language models (MLLMs) represent images and video frames as visual tokens. Scaling from single images to hour-long videos, however, inflates the token budget far beyond practical limits. Popular pipelines therefore either uniformly subsample or apply keyframe selection with retrieval-style scoring using smaller vision-language models. However, these keyframe selection methods still rely on pre-filtering before selection to reduce the inference cost and can miss the most informative moments.   We propose FOCUS, Frame-Optimistic Confidence Upper-bound Selection, a training-free, model-agnostic keyframe selection module that selects query-relevant frames under a strict token budget. FOCUS formulates keyframe selection as a combinatorial pure-exploration (CPE) problem in multi-armed bandits: it treats short temporal clips as arms, and uses empirical means and Bernstein confidence radius to identify informative regions while preserving exploration of uncertain areas. The resulting two-stage exploration-exploitation procedure reduces from a sequential policy with theoretical guarantees, first identifying high-value temporal regions, then selecting top-scoring frames within each region On two long-video question-answering benchmarks, FOCUS delivers substantial accuracy improvements while processing less than 2% of video frames. For videos longer than 20 minutes, it achieves an 11.9% gain in accuracy on LongVideoBench, demonstrating its effectiveness as a keyframe selection method and providing a simple and general solution for scalable long-video understanding with MLLMs.

---

## 57. Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation

**论文链接:** [http://arxiv.org/abs/2510.27080v1](http://arxiv.org/abs/2510.27080v1)

**作者:** Arnabh Borah, Md Tanvirul Alam, Nidhi Rastogi

**发布时间:** 2025-10-31

### GPT解析

### 总结

本研究引入了一个基于检索增强生成(RAG)的框架，用于增强大型语言模型在网络安全任务中的适应性和可靠性，特别是在知识保留和时间推理方面。

### 背景

安全应用越来越多地依赖大型语言模型进行网络威胁检测，但其不透明的推理限制了信任，特别是在需要特定领域知识的决策中。安全威胁迅速演变，要求模型不仅回忆历史事件，还要适应新漏洞和攻击模式。RAG在一般LLM应用中已显示出有效性，但在网络安全领域的潜力尚未得到充分探索。

### 目的

开发一个基于RAG的框架，使网络安全数据情境化，并增强大型语言模型在知识保留和时间推理方面的准确性。

### 方法

使用外部数据集和Llama-3-8B-Instruct模型，评估基准RAG和优化的混合检索方法，并在多个性能指标上进行比较分析。

### 主要发现

混合检索方法在增强大型语言模型对网络安全任务的适应性和可靠性方面显示出显著前景。

### 结论

基于RAG的框架可以有效提升大型语言模型在网络安全领域的应用能力，特别是在处理不断演变的安全威胁时。

### 翻译

安全应用越来越多地依赖大型语言模型(LLMs)进行网络威胁检测；然而，它们的不透明推理常常限制了信任，特别是在需要特定领域网络安全知识的决策中。由于安全威胁迅速演变，LLM不仅要回忆历史事件，还要适应新兴的漏洞和攻击模式。检索增强生成(RAG)已在一般LLM应用中显示出有效性，但其在网络安全领域的潜力仍未得到充分探索。在这项工作中，我们引入了一个基于RAG的框架，用于使网络安全数据情境化，并增强LLM在知识保留和时间推理方面的准确性。使用外部数据集和Llama-3-8B-Instruct模型，我们评估了基准RAG和优化的混合检索方法，并在多个性能指标上进行了比较分析。我们的研究结果强调了混合检索在增强LLM对网络安全任务的适应性和可靠性方面的前景。


### 论文摘要

Security applications are increasingly relying on large language models (LLMs) for cyber threat detection; however, their opaque reasoning often limits trust, particularly in decisions that require domain-specific cybersecurity knowledge. Because security threats evolve rapidly, LLMs must not only recall historical incidents but also adapt to emerging vulnerabilities and attack patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in general LLM applications, but its potential for cybersecurity remains underexplored. In this work, we introduce a RAG-based framework designed to contextualize cybersecurity data and enhance LLM accuracy in knowledge retention and temporal reasoning. Using external datasets and the Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid retrieval approach, and conduct a comparative analysis across multiple performance metrics. Our findings highlight the promise of hybrid retrieval in strengthening the adaptability and reliability of LLMs for cybersecurity tasks.

---

## 58. M^3Detection: Multi-Frame Multi-Level Feature Fusion for Multi-Modal 3D Object Detection with Camera and 4D Imaging Radar

**论文链接:** [http://arxiv.org/abs/2510.27166v1](http://arxiv.org/abs/2510.27166v1)

**作者:** Xiaozhi Li, Huijun Di, Jian Li, Feng Liu, Wei Liang

**发布时间:** 2025-10-31

**备注:** 16 pages, 9 figures

### GPT解析

### 总结

M^3Detection是一个统一的多帧3D物体检测框架，通过融合相机和4D成像雷达数据，实现多级特征融合，解决了单帧融合信息不完整的问题，在多帧检测中取得了最先进的效果。

### 背景

4D成像雷达在恶劣天气条件下提供稳健感知，相机传感器提供密集语义信息，两者互补融合对3D感知具有潜力。然而现有方法多限于单帧输入，无法捕捉完整场景，且图像退化和雷达稀疏性影响检测性能。

### 目的

解决多帧融合面临的两个挑战：实现跨帧和跨模态的稳健有效物体特征融合，以及减少冗余特征提取的计算成本。

### 方法

提出M^3Detection框架，在多模态数据上进行多级特征融合；利用基线检测器的中间特征和跟踪器生成参考轨迹；设计雷达信息引导的全局级间对象特征聚合模块对候选提案进行全局特征对齐；设计局部级间网格特征聚合模块扩展局部特征；使用轨迹级多帧时空推理模块编码跨帧交互并增强时间表示。

### 主要发现

在VoD和TJ4DRadSet数据集上的实验表明，M^3Detection实现了最先进的3D检测性能，验证了其在多帧检测与相机-4D成像雷达融合方面的有效性。

### 结论

M^3Detection通过多级特征融合和时空推理，有效解决了相机-雷达融合中的信息不完整问题，提高了检测性能，为多模态多帧3D感知提供了新的解决方案。

### 翻译

最近的4D成像雷达进展使得在恶劣天气条件下能够实现稳健的感知，而相机传感器则提供密集的语义信息。融合这些互补的模态在成本效益高的3D感知方面具有巨大潜力。然而，大多数现有的相机-雷达融合方法仅限于单帧输入，只能捕捉场景的部分视图。不完整的场景信息，加上图像退化和4D雷达的稀疏性，阻碍了整体检测性能。相比之下，多帧融合提供了更丰富的时空信息，但面临两个挑战：实现跨帧和跨模态的稳健有效物体特征融合，以及减少冗余特征提取的计算成本。因此，我们提出了M^3Detection，一个统一的多帧3D物体检测框架，在来自相机和4D成像雷达的多模态数据上进行多级特征融合。我们的框架利用基线检测器的中间特征并使用跟踪器生成参考轨迹，提高计算效率并为第二阶段提供更丰富的信息。在第二阶段，我们设计了一个雷达信息引导的全局级间对象特征聚合模块，对候选提案进行全局特征对齐，以及一个局部级间网格特征聚合模块，沿着参考轨迹扩展局部特征以增强细粒度物体表示。然后，聚合的特征通过轨迹级多帧时空推理模块进行处理，以编码跨帧交互并增强时间表示。在VoD和TJ4DRadSet数据集上的大量实验表明，M^3Detection实现了最先进的3D检测性能，验证了其在多帧检测与相机-4D成像雷达融合方面的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决单帧相机-雷达融合3D物体检测的局限性，即单帧输入只能捕捉场景部分视图，导致不完整场景信息、图像退化和4D雷达稀疏性影响检测性能。这个问题在自动驾驶领域至关重要，因为3D物体检测是自动驾驶的基础技术，而多帧信息能提供更丰富的时空上下文，提高检测准确性和鲁棒性，同时解决计算效率问题对实时系统至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了单帧检测的局限性及多帧融合的潜力与挑战，然后设计了两阶段框架：第一阶段利用基线检测器提取特征并生成初始检测结果和参考轨迹；第二阶段进行多帧多级特征融合。作者借鉴了现有BEV特征融合检测器（如BEVFusion）、两阶段检测框架（如MPPNet）、注意力机制和跟踪算法（如Immortal），但创新性地设计了多级特征融合策略来解决多模态多帧检测问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多帧多级特征融合提升相机和4D雷达多模态3D物体检测性能，利用丰富时空信息同时避免冗余特征提取。流程分两阶段：第一阶段使用基线检测器提取特征并生成轨迹；第二阶段包括三个模块：全局级间物体特征聚合(GOA)减轻跟踪不确定性，局部级间网格特征聚合(LGA)增强细粒度表示，轨迹级多帧时空推理(MSTR)建模时间特征交互，最终融合全局和局部特征进行检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的多帧3D物体检测框架M3Detection；2)多帧多级特征融合策略(GOA、LGA和MSTR模块)。相比之前工作，不同之处在于：避免了传统多帧方法的冗余特征提取；采用多级特征融合同时关注全局上下文和局部细节；专门针对相机和4D雷达多模态融合优化；通过多假设策略减轻跟踪不确定性影响；不仅进行场景级时间聚合，还进行物体级精细建模。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'M3Detection通过创新的多帧多级特征融合框架，有效整合了相机和4D成像雷达的互补信息，在保持计算效率的同时显著提高了3D物体检测的准确性和鲁棒性，为自动驾驶感知系统提供了更可靠的解决方案。'}


### 论文摘要

Recent advances in 4D imaging radar have enabled robust perception in adverse weather, while camera sensors provide dense semantic information. Fusing the these complementary modalities has great potential for cost-effective 3D perception. However, most existing camera-radar fusion methods are limited to single-frame inputs, capturing only a partial view of the scene. The incomplete scene information, compounded by image degradation and 4D radar sparsity, hinders overall detection performance. In contrast, multi-frame fusion offers richer spatiotemporal information but faces two challenges: achieving robust and effective object feature fusion across frames and modalities, and mitigating the computational cost of redundant feature extraction. Consequently, we propose M^3Detection, a unified multi-frame 3D object detection framework that performs multi-level feature fusion on multi-modal data from camera and 4D imaging radar. Our framework leverages intermediate features from the baseline detector and employs the tracker to produce reference trajectories, improving computational efficiency and providing richer information for second-stage. In the second stage, we design a global-level inter-object feature aggregation module guided by radar information to align global features across candidate proposals and a local-level inter-grid feature aggregation module that expands local features along the reference trajectories to enhance fine-grained object representation. The aggregated features are then processed by a trajectory-level multi-frame spatiotemporal reasoning module to encode cross-frame interactions and enhance temporal representation. Extensive experiments on the VoD and TJ4DRadSet datasets demonstrate that M^3Detection achieves state-of-the-art 3D detection performance, validating its effectiveness in multi-frame detection with camera-4D imaging radar fusion.

---

## 59. MLPerf Automotive

**论文链接:** [http://arxiv.org/abs/2510.27065v1](http://arxiv.org/abs/2510.27065v1)

**作者:** Radoyeh Shojaei, Predrag Djurdjevic, Mostafa El-Khamy, James Goel, Kasper Mecklenburg, John Owens, Pınar Muyan-Özçelik, Tom St. John, Jinho Suh, Arjun Suresh

**发布时间:** 2025-10-31

**备注:** 16 pages, 5 figures, 6 tables

### GPT解析

### 总结

MLPerf Automotive是首个用于评估汽车系统中AI加速部署的机器学习系统的标准化公共基准测试。

### 背景

现有的基准测试套件无法用于汽车系统，因为汽车工作负载具有独特的约束，包括安全性和实时处理能力，这些特性使它们区别于之前引入基准测试所针对的领域。

### 目的

解决汽车机器学习系统需要标准化性能评估方法的问题。

### 方法

通过MLCommons和自动驾驶计算联盟之间的合作开发，提供延迟和准确性指标以及评估协议，使不同硬件平台和软件实现之间能够进行一致且可重复的性能比较。

### 主要发现

第一版基准测试包括2D目标检测、2D语义分割和3D目标检测等汽车感知任务。描述了基准设计的方法论，包括任务选择、参考模型和提交规则。讨论了第一轮基准提交以及获取数据集和开发参考实现所涉及的挑战。

### 结论

MLPerf Automotive基准测试为汽车AI系统提供了标准化的评估方法，使不同平台和实现之间的性能比较成为可能。

### 翻译

我们提出了MLPerf Automotive，这是首个用于评估部署在汽车系统中用于AI加速的机器学习系统的标准化公共基准测试。该基准测试由MLCommons和自动驾驶计算联盟通过合作开发，解决了汽车机器学习系统需要标准化性能评估方法的需求。现有的基准测试套件无法用于这些系统，因为汽车工作负载具有独特的约束，包括安全性和实时处理能力，这些特性使它们区别于之前引入基准测试所针对的领域。我们的基准测试框架提供了延迟和准确性指标以及评估协议，使不同硬件平台和软件实现之间能够进行一致且可重复的性能比较。第一版基准测试包括2D目标检测、2D语义分割和3D目标检测等汽车感知任务。我们描述了基准设计背后的方法论，包括任务选择、参考模型和提交规则。我们还讨论了第一轮基准提交以及获取数据集和开发参考实现所涉及的挑战。我们的基准测试代码可在https://github.com/mlcommons/mlperf_automotive获取。


### 论文摘要

We present MLPerf Automotive, the first standardized public benchmark for evaluating Machine Learning systems that are deployed for AI acceleration in automotive systems. Developed through a collaborative partnership between MLCommons and the Autonomous Vehicle Computing Consortium, this benchmark addresses the need for standardized performance evaluation methodologies in automotive machine learning systems. Existing benchmark suites cannot be utilized for these systems since automotive workloads have unique constraints including safety and real-time processing that distinguish them from the domains that previously introduced benchmarks target. Our benchmarking framework provides latency and accuracy metrics along with evaluation protocols that enable consistent and reproducible performance comparisons across different hardware platforms and software implementations. The first iteration of the benchmark consists of automotive perception tasks in 2D object detection, 2D semantic segmentation, and 3D object detection. We describe the methodology behind the benchmark design including the task selection, reference models, and submission rules. We also discuss the first round of benchmark submissions and the challenges involved in acquiring the datasets and the engineering efforts to develop the reference implementations. Our benchmark code is available at https://github.com/mlcommons/mlperf_automotive.

---

