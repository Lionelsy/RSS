# 今日论文推荐 - 2025-10-01

共 69 篇论文

---

## 1. Generalized Fine-Grained Category Discovery with Multi-Granularity Conceptual Experts

**论文链接:** [http://arxiv.org/abs/2509.26227v1](http://arxiv.org/abs/2509.26227v1)

**作者:** Haiyang Zheng, Nan Pu, Wenjing Li, Nicu Sebe, Zhun Zhong

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了多粒度概念专家(MGCE)框架，通过自适应挖掘视觉概念和整合多粒度知识来解决广义类别发现(GCD)中的挑战，实现了在未知类别数量情况下的准确类别发现。

### 背景

广义类别发现(GCD)是一个开放世界问题，旨在利用部分标记类别的知识对未标记数据进行聚类。现有方法存在两个主要局限：无法利用视觉数据中的多粒度概念信息，以及假设训练时已知未标记类别的数量，这在实际场景中不切实际。

### 目的

解决现有GCD方法在利用多粒度概念信息和处理未知类别数量方面的局限性，提出一种能够在实际开放世界场景中有效工作的类别发现框架。

### 方法

提出多粒度概念专家(MGCE)框架，包含两个核心模块：(1)动态概念对比学习(DCCL)，交替进行概念挖掘和双层次表示学习；(2)多粒度专家协作学习(MECL)，引入不同粒度的专家并使用概念对齐矩阵实现跨专家协作。该框架能自动估计未标记数据中的类别数量。

### 主要发现

在九个细粒度视觉识别基准上的实验表明，MGCE达到最先进结果，特别是在新类别准确性方面。即使没有类别数量的先验知识，MGCE也优于需要知道类别确切数量的参数化方法，平均提高3.6%。

### 结论

MGCE框架通过有效利用多粒度概念信息和自适应估计类别数量，显著提升了广义类别发现性能，特别适用于实际开放世界场景，代码已公开。

### 翻译

广义类别发现(GCD)是一个开放世界问题，它通过利用部分标记类别的知识对未标记数据进行聚类。一个关键挑战是未标记数据可能包含已知和未知类别。现有方法受两个主要限制。首先，它们无法利用视觉数据中的多粒度概念信息，这限制了表示质量。其次，大多数假设训练时知道未标记类别的数量，这在实际场景中不切实际。为解决这些问题，我们提出了多粒度概念专家(MGCE)框架，该框架自适应挖掘视觉概念并整合多粒度知识进行准确的类别发现。MGCE包含两个模块：(1)动态概念对比学习(DCCL)，交替进行概念挖掘和双层次表示学习，联合优化特征学习和类别发现；(2)多粒度专家协作学习(MECL)，通过引入不同粒度的额外专家扩展单专家范式，并使用概念对齐矩阵实现有效的跨专家协作。重要的是，MGCE可以自动估计未标记数据中的类别数量，使其适用于实际的开放世界设置。在九个细粒度视觉识别基准上的大量实验表明，MGCE达到最先进的结果，特别是在新类别准确性方面。值得注意的是，即使没有类别数量的先验知识，MGCE也优于需要知道类别确切数量的参数化方法，平均提高3.6%。代码可在https://github.com/HaiyangZheng/MGCE获取。


### 论文摘要

Generalized Category Discovery (GCD) is an open-world problem that clusters unlabeled data by leveraging knowledge from partially labeled categories. A key challenge is that unlabeled data may contain both known and novel categories. Existing approaches suffer from two main limitations. First, they fail to exploit multi-granularity conceptual information in visual data, which limits representation quality. Second, most assume that the number of unlabeled categories is known during training, which is impractical in real-world scenarios. To address these issues, we propose a Multi-Granularity Conceptual Experts (MGCE) framework that adaptively mines visual concepts and integrates multi-granularity knowledge for accurate category discovery. MGCE consists of two modules: (1) Dynamic Conceptual Contrastive Learning (DCCL), which alternates between concept mining and dual-level representation learning to jointly optimize feature learning and category discovery; and (2) Multi-Granularity Experts Collaborative Learning (MECL), which extends the single-expert paradigm by introducing additional experts at different granularities and by employing a concept alignment matrix for effective cross-expert collaboration. Importantly, MGCE can automatically estimate the number of categories in unlabeled data, making it suitable for practical open-world settings. Extensive experiments on nine fine-grained visual recognition benchmarks demonstrate that MGCE achieves state-of-the-art results, particularly in novel-class accuracy. Notably, even without prior knowledge of category numbers, MGCE outperforms parametric approaches that require knowing the exact number of categories, with an average improvement of 3.6\%. Code is available at https://github.com/HaiyangZheng/MGCE.

---

## 2. Self-Supervised Anatomical Consistency Learning for Vision-Grounded Medical Report Generation

**论文链接:** [http://arxiv.org/abs/2509.25963v1](http://arxiv.org/abs/2509.25963v1)

**作者:** Longzhen Yang, Zhangkai Ni, Ying Wen, Yihang Liu, Lianghua He, Heng Tao Shen

**发布时间:** 2025-09-30

**DOI:** 10.1145/3746027.3754913

### GPT解析

### 总结

本文提出了一种名为SS-ACL（自监督解剖一致性学习）的新型无标注框架，用于视觉引导的医学报告生成。该框架通过将生成的报告与相应的解剖区域对齐，使用简单的文本提示，构建了分层解剖图，并引入区域级对比学习来增强样本间语义对齐。实验表明，该方法无需专家标注即可生成准确且视觉有基础的报告，在词汇准确性和临床功效上分别比最先进方法提高10%和25%，在零样本视觉定位上超越当前领先视觉基础模型8%。

### 背景

视觉引导的医学报告生成旨在产生临床准确的医学图像描述，但现有方法通常依赖单独训练的检测模块，需要大量专家标注，引入高标注成本。此外，不同数据集中的病理分布偏差限制了方法的泛化能力。

### 目的

解决现有方法的高标注成本和泛化能力有限的问题，提出一种无需专家标注的框架，能够生成准确且视觉有基础的医学报告，并提供可解释的视觉证据。

### 方法

提出Self-Supervised Anatomical Consistency Learning（SS-ACL）框架，该框架将生成的报告与相应的解剖区域对齐，使用简单的文本提示。构建了受人体解剖学启发的分层解剖图，递归重建细粒度解剖区域以强制样本内空间对齐，并引入基于解剖一致性的区域级对比学习来增强样本间语义对齐。这些对齐的嵌入作为报告生成的先验知识，使注意力图能够提供可解释的视觉证据。

### 主要发现

SS-ACL无需专家标注即可生成准确且视觉有基础的报告，在词汇准确性上比最先进方法提高10%，在临床功效上提高25%。在各种下游视觉任务上具有竞争力，在零样本视觉定位上超越当前领先视觉基础模型8%。

### 结论

SS-ACL是一种有效的无需标注的医学报告生成方法，能够生成准确且可解释的视觉证据支持的报告，在多种视觉任务上表现出色。

### 翻译

视觉引导的医学报告生成旨在产生临床准确的医学图像描述，基于明确的视觉证据，以提高可解释性并促进融入临床工作流程。然而，现有方法通常依赖于单独训练的检测模块，需要大量专家标注，引入了高标注成本，并由于数据集中的病理分布偏差而限制了泛化能力。为了解决这些挑战，我们提出了自监督解剖一致性学习（SS-ACL）——一种新颖且无需标注的框架，它使用简单的文本提示将生成的报告与相应的解剖区域对齐。SS-ACL构建了受人体解剖学自上而下包含结构启发的分层解剖图，按空间位置组织实体。它递归重建细粒度解剖区域，强制样本内空间对齐， inherently引导注意力图朝向文本提示的视觉相关区域。为了进一步增强异常识别的样本间语义对齐，SS-ACL引入了基于解剖一致性的区域级对比学习。这些对齐的嵌入作为报告生成的先验，使注意力图能够提供可解释的视觉证据。大量实验表明，SS-ACL在不依赖专家标注的情况下，（i）生成准确且视觉有基础的报告——在词汇准确性上比最先进方法提高10%，在临床功效上提高25%，并且（ii）在各种下游视觉任务上取得竞争性性能，在零样本视觉定位上超越当前领先的视觉基础模型8%。


### 论文摘要

Vision-grounded medical report generation aims to produce clinically accurate descriptions of medical images, anchored in explicit visual evidence to improve interpretability and facilitate integration into clinical workflows. However, existing methods often rely on separately trained detection modules that require extensive expert annotations, introducing high labeling costs and limiting generalizability due to pathology distribution bias across datasets. To address these challenges, we propose Self-Supervised Anatomical Consistency Learning (SS-ACL) -- a novel and annotation-free framework that aligns generated reports with corresponding anatomical regions using simple textual prompts. SS-ACL constructs a hierarchical anatomical graph inspired by the invariant top-down inclusion structure of human anatomy, organizing entities by spatial location. It recursively reconstructs fine-grained anatomical regions to enforce intra-sample spatial alignment, inherently guiding attention maps toward visually relevant areas prompted by text. To further enhance inter-sample semantic alignment for abnormality recognition, SS-ACL introduces a region-level contrastive learning based on anatomical consistency. These aligned embeddings serve as priors for report generation, enabling attention maps to provide interpretable visual evidence. Extensive experiments demonstrate that SS-ACL, without relying on expert annotations, (i) generates accurate and visually grounded reports -- outperforming state-of-the-art methods by 10\% in lexical accuracy and 25\% in clinical efficacy, and (ii) achieves competitive performance on various downstream visual tasks, surpassing current leading visual foundation models by 8\% in zero-shot visual grounding.

---

## 3. The Impact of Scaling Training Data on Adversarial Robustness

**论文链接:** [http://arxiv.org/abs/2509.25927v1](http://arxiv.org/abs/2509.25927v1)

**作者:** Marco Zimmerli, Andreas Plesner, Till Aczel, Roger Wattenhofer

**发布时间:** 2025-09-30

**备注:** Accepted at the workshop Reliable ML from Unreliable Data at NeurIPS  2025

### GPT解析

### 总结

研究训练数据特性对36种最先进视觉模型对抗鲁棒性的影响，发现数据质量和模型架构比单纯规模更重要。

### 背景

深度神经网络尽管在架构和训练范式上有所进步，但仍容易受到对抗样本的攻击。

### 目的

研究训练数据特性如何影响对抗鲁棒性，评估不同模型在各种攻击下的表现。

### 方法

在从120万到220亿张图像的数据集上训练36种视觉模型，采用监督、自监督和对比学习方法，并在六种黑盒攻击类别下评估：随机扰动、几何掩模、对象操作、图像损坏和风格转换。

### 主要发现

鲁棒性随数据量和模型大小呈对数缩放规律；数据量增加十倍，攻击成功率平均降低约3.2%；模型大小增加十倍，攻击成功率平均降低约13.4%；精选数据集上的自监督模型优于大规模非精选数据集上的模型；对抗微调改善了结构变化泛化但未改善颜色分布泛化；人类与机器视觉存在差距。

### 结论

虽然扩展规模可以提高鲁棒性，但数据质量、架构和训练目标在实现广泛对抗韧性方面比原始规模起着更决定性的作用。

### 翻译

尽管架构和训练范式有所进步，深度神经网络仍然容易受到对抗样本的攻击。我们研究了训练数据特性如何影响36种最先进视觉模型的对抗鲁棒性，这些模型涵盖监督、自监督和对比学习方法，在从120万到220亿张图像的数据集上训练。模型在六种黑盒攻击类别下进行评估：随机扰动、两种几何掩模、COCO对象操作、ImageNet-C损坏和ImageNet-R风格转换。鲁棒性随数据量和模型大小呈对数缩放规律：数据量增加十倍，攻击成功率平均降低约3.2%；模型大小增加十倍，攻击成功率平均降低约13.4%。值得注意的是，一些在精选数据集上训练的自监督模型(如DINOv2)优于在更大但不太精选的数据集上训练的模型，挑战了仅规模驱动鲁棒性的假设。ResNet50的对抗微调改善了结构变化的泛化能力，但没有改善颜色分布的泛化能力。人类评估显示人类视觉与机器视觉之间存在持续差距。这些结果表明，虽然扩展规模可以提高鲁棒性，但数据质量、架构和训练目标在实现广泛对抗韧性方面比原始规模起着更决定性的作用。


### 论文摘要

Deep neural networks remain vulnerable to adversarial examples despite advances in architectures and training paradigms. We investigate how training data characteristics affect adversarial robustness across 36 state-of-the-art vision models spanning supervised, self-supervised, and contrastive learning approaches, trained on datasets from 1.2M to 22B images. Models were evaluated under six black-box attack categories: random perturbations, two types of geometric masks, COCO object manipulations, ImageNet-C corruptions, and ImageNet-R style shifts. Robustness follows a logarithmic scaling law with both data volume and model size: a tenfold increase in data reduces attack success rate (ASR) on average by ~3.2%, whereas a tenfold increase in model size reduces ASR on average by ~13.4%. Notably, some self-supervised models trained on curated datasets, such as DINOv2, outperform others trained on much larger but less curated datasets, challenging the assumption that scale alone drives robustness. Adversarial fine-tuning of ResNet50s improves generalization across structural variations but not across color distributions. Human evaluation reveals persistent gaps between human and machine vision. These results show that while scaling improves robustness, data quality, architecture, and training objectives play a more decisive role than raw scale in achieving broad-spectrum adversarial resilience.

---

## 4. HiStyle: Hierarchical Style Embedding Predictor for Text-Prompt-Guided Controllable Speech Synthesis

**论文链接:** [http://arxiv.org/abs/2509.25842v1](http://arxiv.org/abs/2509.25842v1)

**作者:** Ziyu Zhang, Hanzhao Li, Jingbin Hu, Wenhao Li, Lei Xie

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究提出了HiStyle，一种两阶段风格嵌入预测器，用于提升可控文本转语音系统的风格控制能力。研究通过t-SNE分析揭示了风格嵌入的层次聚类模式，并设计了分层预测方法。

### 背景

可控语音合成指通过操控特定韵律和副语言学属性（如性别、音量、语速、音高和音高变化）来精确控制说话风格。随着先进生成模型（特别是大型语言模型和扩散模型）的集成，可控文本转语音系统已从基于标签的控制转向基于自然语言描述的控制，通常通过从文本提示中预测全局风格嵌入实现。

### 目的

解决当前直接预测风格嵌入方法忽略其潜在分布的问题，从而释放可控文本转语音系统的全部潜力。

### 方法

使用t-SNE分析可视化主流文本转语音系统的全局风格嵌入分布；提出HiStyle两阶段风格嵌入预测器，基于文本提示分层预测风格嵌入；结合对比学习对齐文本和音频嵌入空间；提出结合统计方法和人类听觉偏好的风格标注策略。

### 主要发现

文本转语音系统中的风格嵌入呈现清晰的层次聚类模式：嵌入首先按音色聚类，然后根据风格属性细分为更精细的聚类。

### 结论

综合实验表明，HiStyle应用于基础文本转语音模型时，比其他风格嵌入预测方法实现了显著更好的风格可控性，同时保持了高语音质量（自然度和可懂度）。

### 翻译

可控语音合成指通过操控特定的韵律和副语言学属性（如性别、音量、语速、音高和音高变化）来精确控制说话风格。随着先进生成模型（特别是大型语言模型和扩散模型）的集成，可控文本转语音系统已从基于标签的控制转向基于自然语言描述的控制，这通常通过从文本提示中预测全局风格嵌入来实现。然而，这种直接的预测方法忽略了风格嵌入的潜在分布，可能阻碍可控文本转语音系统的全部潜力。在本研究中，我们使用t-SNE分析和可视化各种主流文本转语音系统的全局风格嵌入分布，揭示了清晰的层次聚类模式：嵌入首先按音色聚类，然后根据风格属性细分为更精细的聚类。基于这一观察，我们提出了HiStyle，一个基于文本条件分层预测风格嵌入的两阶段风格嵌入预测器，并进一步结合对比学习来帮助对齐文本和音频嵌入空间。此外，我们提出了一种风格标注策略，利用统计方法和人类听觉偏好的互补优势，生成更准确和感知一致的文本提示用于风格控制。综合实验表明，当应用于基础文本转语音模型时，HiStyle比其他风格嵌入预测方法实现了显著更好的风格可控性，同时保持了高语音质量（自然度和可懂度）。音频样本可在https://anonymous.4open.science/w/HiStyle-2517/获取。


### 论文摘要

Controllable speech synthesis refers to the precise control of speaking style by manipulating specific prosodic and paralinguistic attributes, such as gender, volume, speech rate, pitch, and pitch fluctuation. With the integration of advanced generative models, particularly large language models (LLMs) and diffusion models, controllable text-to-speech (TTS) systems have increasingly transitioned from label-based control to natural language description-based control, which is typically implemented by predicting global style embeddings from textual prompts. However, this straightforward prediction overlooks the underlying distribution of the style embeddings, which may hinder the full potential of controllable TTS systems. In this study, we use t-SNE analysis to visualize and analyze the global style embedding distribution of various mainstream TTS systems, revealing a clear hierarchical clustering pattern: embeddings first cluster by timbre and subsequently subdivide into finer clusters based on style attributes. Based on this observation, we propose HiStyle, a two-stage style embedding predictor that hierarchically predicts style embeddings conditioned on textual prompts, and further incorporate contrastive learning to help align the text and audio embedding spaces. Additionally, we propose a style annotation strategy that leverages the complementary strengths of statistical methodologies and human auditory preferences to generate more accurate and perceptually consistent textual prompts for style control. Comprehensive experiments demonstrate that when applied to the base TTS model, HiStyle achieves significantly better style controllability than alternative style embedding predicting approaches while preserving high speech quality in terms of naturalness and intelligibility. Audio samples are available at https://anonymous.4open.science/w/HiStyle-2517/.

---

## 5. Less is More: Towards Simple Graph Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2509.25742v1](http://arxiv.org/abs/2509.25742v1)

**作者:** Yanan Zhao, Feng Ji, Jingyang Dai, Jiaze Ma, Wee Peng Tay

**发布时间:** 2025-09-30

**备注:** Submitted to ICLR 2026

### GPT解析

### 总结

本文提出了一种简单而有效的图对比学习(GCL)方法，通过聚合节点特征噪声和图拓扑结构特征，在异质图上取得了最先进的结果，同时无需复杂的数据增强和负采样技术。

### 背景

图对比学习在无监督图表示学习中显示出强大潜力，但在异质图(连接节点通常属于不同类的图)上的效果仍然有限。现有方法大多依赖复杂的数据增强方案、复杂的编码器或负采样技术。

### 目的

重新审视监督和无监督图学习的基础，探索在异质图这种具有挑战性的设置下，是否真的需要如此复杂的方法。

### 方法

发现原始节点特征和图结构自然为对比学习提供了两个互补视图。提出一个简单的GCL模型，使用GCN编码器捕获结构特征，使用MLP编码器隔离节点特征噪声，无需数据增强和负采样。

### 主要发现

在异质图基准测试上取得了最先进的结果，计算和内存开销最小。在同质图上在复杂性、可扩展性和鲁棒性方面具有优势。提供了该方法的理论依据。

### 结论

通过大量实验验证了该方法的有效性，包括对黑盒和白盒对抗攻击的鲁棒性评估，证明了该方法的实用性和鲁棒性。

### 翻译

图对比学习(GCL)在无监督图表示学习中显示出强大的潜力，然而它在异质图(连接节点通常属于不同类的图)上的有效性仍然有限。大多数现有方法依赖于复杂的数据增强方案、复杂的编码器或负采样，这引发了一个问题：在这种具有挑战性的设置下，是否真的需要如此复杂的方案。在这项工作中，我们重新审视了监督和无监督图学习的基础，并为GCL发现了一个简单而有效的原则：通过聚合节点特征噪声和从图拓扑导出的结构特征来缓解节点特征噪声。这一观察表明，原始节点特征和图结构自然为对比学习提供了两个互补视图。基于这一见解，我们提出了一个极其简单的GCL模型，使用GCN编码器捕获结构特征，使用MLP编码器隔离节点特征噪声。我们的设计既不需要数据增强也不需要负采样，却以最小的计算和内存开销在异质图基准测试上取得了最先进的结果，同时在复杂性、可扩展性和鲁棒性方面在同质图上也具有优势。我们为该方法提供了理论依据，并通过大量实验验证了其有效性，包括对黑盒和白盒对抗攻击的鲁棒性评估。


### 论文摘要

Graph Contrastive Learning (GCL) has shown strong promise for unsupervised graph representation learning, yet its effectiveness on heterophilic graphs, where connected nodes often belong to different classes, remains limited. Most existing methods rely on complex augmentation schemes, intricate encoders, or negative sampling, which raises the question of whether such complexity is truly necessary in this challenging setting. In this work, we revisit the foundations of supervised and unsupervised learning on graphs and uncover a simple yet effective principle for GCL: mitigating node feature noise by aggregating it with structural features derived from the graph topology. This observation suggests that the original node features and the graph structure naturally provide two complementary views for contrastive learning. Building on this insight, we propose an embarrassingly simple GCL model that uses a GCN encoder to capture structural features and an MLP encoder to isolate node feature noise. Our design requires neither data augmentation nor negative sampling, yet achieves state-of-the-art results on heterophilic benchmarks with minimal computational and memory overhead, while also offering advantages in homophilic graphs in terms of complexity, scalability, and robustness. We provide theoretical justification for our approach and validate its effectiveness through extensive experiments, including robustness evaluations against both black-box and white-box adversarial attacks.

---

## 6. ProbMed: A Probabilistic Framework for Medical Multimodal Binding

**论文链接:** [http://arxiv.org/abs/2509.25711v1](http://arxiv.org/abs/2509.25711v1)

**作者:** Yuan Gao, Sangwook Kim, Jianzhong You, Chris McIntosh

**发布时间:** 2025-09-30

**备注:** ICCV 2025

### GPT解析

### 总结

本研究提出了概率模态增强诊断（ProbMED）模型，通过概率对比学习解决医疗决策中多模态信息整合的问题，在多种医疗任务上表现优异。

### 背景

医疗决策需要整合多种医疗信息，从影像到临床叙述，这些医疗模态通常以多对多方式获取。然而，当前的医学视觉-语言预训练模型（Med-VLPMs）无法在模型训练和嵌入中直接考虑这种多对多映射。

### 目的

开发一种能够处理医疗信息多对多映射的多模态医学视觉-语言预训练模型，提高医疗决策的准确性。

### 方法

提出概率模态增强诊断（ProbMED），采用概率对比学习来建模嵌入的分布而非确定性估计。将胸部X光、心电图、超声心动图和临床文本四种模态对齐到统一的概率嵌入空间。使用带有Hellinger距离的InfoNCE损失整合跨模态分布，并引入概率合成采样损失改善模态内绑定。

### 主要发现

在13个医学数据集上的实验表明，ProbMED在跨模态检索、零样本和少样本分类方面优于当前Med-VLPMs。模型展示了多种模态的稳健整合能力，用于预后判断，并改进了医学模态内和模态间的绑定。

### 结论

ProbMED通过概率对比学习有效解决了医疗信息多对多映射的问题，能够更好地整合多种医疗模态，提高医疗决策的准确性。

### 翻译

医疗决策需要整合多种医疗信息，从影像到临床叙述。这些医疗模态通常以多对多方式获取。然而，当前的医学视觉-语言预训练模型（Med-VLPMs）无法在模型训练和嵌入中直接考虑这种多对多映射。为此，我们提出了概率模态增强诊断（ProbMED），这是一种多模态Med-VLPM，采用概率对比学习来建模嵌入的分布而非确定性估计。ProbMED将四种不同模态——胸部X光、心电图、超声心动图和临床文本——对齐到统一的概率嵌入空间。我们使用带有Hellinger距离的InfoNCE损失来整合跨模态分布。我们引入了概率合成采样损失，捕获模态特定的均值和方差，以改善模态内绑定。在13个医学数据集上的广泛实验表明，我们的模型在跨模态检索、零样本和少样本分类方面优于当前的Med-VLPMs。我们还展示了多种模态的稳健整合用于预后判断，显示出改进的医学模态内和模态间绑定。


### 论文摘要

Medical decision-making requires integrating diverse medical information, from imaging to clinical narratives. These medical modalities are often acquired in a many-to-many manner. However, current medical vision-language pretraining models (Med-VLPMs) fail to directly account for this many-to-many mapping in their model training and embeddings. To address this, we present Probabilistic Modality-Enhanced Diagnosis (ProbMED), a multimodal Med-VLPM that employs probabilistic contrastive learning to model distributions over embeddings rather than deterministic estimates. ProbMED aligns four distinct modalities -- chest X-rays, electrocardiograms, echocardiograms, and clinical text -- into a unified probabilistic embedding space. We use InfoNCE loss with Hellinger distance to integrate inter-modality distributions. We introduce a probabilistic synthetic sampling loss that captures modality-specific mean and variance to improve intra-modality binding. Extensive experiments across 13 medical datasets demonstrate that our model outperforms current Med-VLPMs in cross-modality retrieval, zero-shot, and few-shot classification. We also demonstrate the robust integration of multiple modalities for prognostication, showing improved intra- and inter-medical modality binding.

---

## 7. Generalized Contrastive Learning for Universal Multimodal Retrieval

**论文链接:** [http://arxiv.org/abs/2509.25638v1](http://arxiv.org/abs/2509.25638v1)

**作者:** Jungsoo Lee, Janghoon Cho, Hyojin Park, Munawar Hayat, Kyuwoong Hwang, Fatih Porikli, Sungha Choi

**发布时间:** 2025-09-30

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为广义对比学习(GCL)的新方法，用于解决跨模态检索模型在检索融合图像-文本模态时的性能下降问题。GCL无需创建新数据集，通过在mini-batch内跨所有模态执行对比学习，利用现有图像-标题配对数据集学习统一表示空间，在多个基准测试中提升了现有多模态检索模型的性能。

### 背景

跨模态检索模型(如CLIP)在检索由融合图像-文本模态组成的键(如同时包含图像和文本的维基百科页面)时表现出性能下降。尽管它们在一致的性能改进方面表现出色，但在处理这种多模态组合时效果不佳。

### 目的

开发一个统一的单检索模型，能够检索跨不同模态组合的键，以解决跨模态检索模型在处理融合模态时的性能下降问题。

### 方法

提出广义对比学习(GCL)，一种新的损失函数，通过在mini-batch内跨所有模态执行对比学习，利用现有的图像-标题配对数据集来学习统一的表示空间。这种方法避免了构建新数据集的需求，同时能够推广到未见过的模态组合。

### 主要发现

GCL方法在M-BEIR、MMEB和CoVR基准测试中，对即用型多模态检索模型(如VISTA、CLIP和TinyCLIP)展示了一致的性能改进，证明了其有效性和通用性。

### 结论

广义对比学习(GCL)是一种有效的多模态检索改进方法，它无需繁琐的新数据集策划，即可提高多模态检索性能，并且能够推广到未见过的模态组合。

### 翻译

尽管跨模态检索模型(如CLIP)持续展现出性能提升，但在检索由融合图像-文本模态组成的键(如同时包含图像和文本的维基百科页面)时，其性能会下降。为应对这一关键挑战，近期多模态检索研究致力于开发能够检索跨不同模态组合的键的统一单检索模型。常见方法是构建新的图像-文本三元组集合(例如，给定查询图像检索一对图像和文本)。然而，这种方法需要精心策划以确保数据集质量，且无法推广到未见过的模态组合。为克服这些局限，本文提出广义对比学习(GCL)，一种新的损失函数，无需繁琐的新数据集策划即可提高多模态检索性能。具体而言，GCL通过在mini-batch内跨所有模态执行对比学习，利用现有的图像-标题配对数据集学习统一表示空间。我们通过在M-BEIR、MMEB和CoVR基准测试中展示GCL对即用型多模态检索模型(如VISTA、CLIP和TinyCLIP)的一致性能改进，证明了GCL的有效性。


### 论文摘要

Despite their consistent performance improvements, cross-modal retrieval models (e.g., CLIP) show degraded performances with retrieving keys composed of fused image-text modality (e.g., Wikipedia pages with both images and text). To address this critical challenge, multimodal retrieval has been recently explored to develop a unified single retrieval model capable of retrieving keys across diverse modality combinations. A common approach involves constructing new composed sets of image-text triplets (e.g., retrieving a pair of image and text given a query image). However, such an approach requires careful curation to ensure the dataset quality and fails to generalize to unseen modality combinations. To overcome these limitations, this paper proposes Generalized Contrastive Learning (GCL), a novel loss formulation that improves multimodal retrieval performance without the burdensome need for new dataset curation. Specifically, GCL operates by enforcing contrastive learning across all modalities within a mini-batch, utilizing existing image-caption paired datasets to learn a unified representation space. We demonstrate the effectiveness of GCL by showing consistent performance improvements on off-the-shelf multimodal retrieval models (e.g., VISTA, CLIP, and TinyCLIP) using the M-BEIR, MMEB, and CoVR benchmarks.

---

## 8. Translation from Wearable PPG to 12-Lead ECG

**论文链接:** [http://arxiv.org/abs/2509.25480v1](http://arxiv.org/abs/2509.25480v1)

**作者:** Hui Ji, Wei Gao, Pengfei Zhou

**发布时间:** 2025-09-29

**备注:** 14 pages,10 figures

### GPT解析

### 总结

P2Es是一种创新的、具有人口感知能力的扩散框架，可以从PPG信号生成临床有效的12导联ECG，通过三个关键创新解决了现有技术的局限性。

### 背景

12导联心电图是心血管监测的金标准，比PPG具有更好的诊断粒度和特异性，但现有系统依赖于繁琐的多电极设置，限制了持续监测；而基于PPG的方法因缺乏导联间约束和时空依赖性建模不足，无法重建多导联ECG。

### 目的

开发一个能够从PPG信号生成临床有效的12导联ECG的框架，解决现有技术的局限性，实现更便捷的心血管监测。

### 方法

P2Es框架包含三个关键创新：1)在正向过程中引入频域模糊化和时域噪声干扰模拟真实信号失真；2)在反向过程中设计时域多尺度生成模块和频域去模糊；3)利用基于KNN的聚类结合对比学习为反向过程分配亲和矩阵，实现人口特定的ECG转换。

### 主要发现

大量实验结果表明，P2Es在12导联ECG重建方面优于基线模型。

### 结论

P2Es框架成功解决了从PPG信号生成临床有效的12导联ECG的技术挑战，为心血管监测提供了新的可能性。

### 翻译

12导联心电图是心血管监测的金标准，与光电容积描记法相比具有更好的诊断粒度和特异性。然而，现有的12导联心电图系统依赖于繁琐的多电极设置，限制了在 ambulatory settings 中的持续监测，而当前的基于PPG的方法由于缺乏导联间约束和对导联间时空依赖性建模不足，无法重建多导联心电图。为了弥补这一差距，我们引入了P2Es，一个创新的、具有人口感知能力的扩散框架，通过三个关键创新从PPG信号生成临床有效的12导联心电图。具体来说，在正向过程中，我们引入频域模糊化，然后进行时域噪声干扰，以模拟真实世界的信号失真。在反向过程中，我们设计了时域多尺度生成模块，然后进行频域去模糊。特别是，我们利用基于KNN的聚类结合对比学习为反向过程分配亲和矩阵，实现人口特定的ECG转换。大量实验结果表明，P2Es在12导联心电图重建方面优于基线模型。


### 论文摘要

The 12-lead electrocardiogram (ECG) is the gold standard for cardiovascular monitoring, offering superior diagnostic granularity and specificity compared to photoplethysmography (PPG). However, existing 12-lead ECG systems rely on cumbersome multi-electrode setups, limiting sustained monitoring in ambulatory settings, while current PPG-based methods fail to reconstruct multi-lead ECG due to the absence of inter-lead constraints and insufficient modeling of spatial-temporal dependencies across leads. To bridge this gap, we introduce P2Es, an innovative demographic-aware diffusion framework designed to generate clinically valid 12-lead ECG from PPG signals via three key innovations. Specifically, in the forward process, we introduce frequency-domain blurring followed by temporal noise interference to simulate real-world signal distortions. In the reverse process, we design a temporal multi-scale generation module followed by frequency deblurring. In particular, we leverage KNN-based clustering combined with contrastive learning to assign affinity matrices for the reverse process, enabling demographic-specific ECG translation. Extensive experimental results show that P2Es outperforms baseline models in 12-lead ECG reconstruction.

---

## 9. LUMA: Low-Dimension Unified Motion Alignment with Dual-Path Anchoring for Text-to-Motion Diffusion Model

**论文链接:** [http://arxiv.org/abs/2509.25304v1](http://arxiv.org/abs/2509.25304v1)

**作者:** Haozhe Jia, Wenshuo Chen, Yuqi Lin, Yang Yang, Lei Wang, Mang Ning, Bowen Tian, Songning Lai, Nanqian Jia, Yifan Chen, Yutao Yue

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为LUMA的文本到动作扩散模型，通过双路径锚定机制解决了现有模型中的语义对齐问题和运动学伪影。

### 背景

当前基于U-Net架构的扩散模型在文本到动作生成任务中表现良好，但仍存在语义对齐问题和运动学伪影。研究发现网络深层的严重梯度衰减是关键瓶颈，导致高级特征学习不足。

### 目的

解决网络深层的梯度衰减问题，提高文本到动作生成模型的语义对齐能力，减少运动学伪影，同时提高模型训练效率。

### 方法

提出LUMA（低维统一动作对齐）模型，采用双路径锚定机制：第一路径使用轻量级MoCLIP模型提供时域语义监督；第二路径从低频DCT分量中提取频域互补对齐信号；通过时间调制机制自适应融合这两个锚，使模型在去噪过程中从粗略对齐逐渐过渡到细粒度语义细化。

### 主要发现

在HumanML3D和KIT-ML数据集上，LUMA实现了最先进的性能，FID分数分别为0.035和0.123；与基线相比，LUMA加速了1.4倍的收敛速度。

### 结论

LUMA是一个高效且可扩展的高保真文本到动作生成解决方案，能够有效解决语义对齐问题和运动学伪影。

### 翻译

虽然当前基于U-Net架构的扩散模型在文本到动作生成任务中已显示出有希望的结果，但它们仍存在语义对齐问题和运动学伪影。通过分析，我们识别出网络深层的严重梯度衰减是关键瓶颈，导致高级特征学习不足。为解决这个问题，我们提出了LUMA（低维统一动作对齐），一种文本到动作扩散模型，它采用双路径锚定来增强语义对齐。第一条路径包含一个通过对比学习训练的轻量级MoCLIP模型，不依赖外部数据，提供时域语义监督。第二条路径在频域引入互补对齐信号，从富含语义内容的低频DCT分量中提取。这两个锚通过时间调制机制自适应融合，使模型在去噪过程中逐渐从粗略对齐过渡到细粒度语义细化。在HumanML3D和KIT-ML上的实验结果表明，LUMA实现了最先进的性能，FID分数分别为0.035和0.123。此外，与基线相比，LUMA加速了1.4倍的收敛速度，使其成为高保真文本到动作生成的高效可扩展解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决扩散模型在文本到运动生成任务中的语义不对齐和运动伪影问题。作者通过分析发现网络深层存在严重的梯度衰减现象，导致高级特征学习不足。这个问题在现实中很重要，因为文本到运动生成技术在动画、电影制作、虚拟现实和机器人等领域有广泛应用，而语义对齐问题直接影响生成质量，限制了这些技术的实用性和效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有扩散模型在文本到运动生成中的局限性，特别是网络深层的梯度衰减问题。他们设计了一个双路径统一语义对齐框架，借鉴了表示对齐(REPA)等工作的思想，但创新性地引入了时间锚点和频率锚点。时间锚点使用MoCLIP轻量级编码器提取运动语义，频率锚点利用DCT变换提取低频特征作为稳定监督信号。通过FiLM调制模块动态整合这两个锚点，解决了对外部预训练模型的依赖，使其更适合运动任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'LUMA的核心思想是通过双路径锚定机制增强语义对齐，解决网络深层梯度衰减问题。整体流程是：1)从U-Net下采样块提取低维特征；2)应用两个MLP将特征投影到时间语义和频率语义表示；3)使用FiLM网络根据时间步调制特征强度；4)计算时间锚点与MoCLIP特征的余弦相似度损失；5)计算频率锚点与运动DCT系数的均方误差损失；6)使用余弦退火调度渐进减少语义对齐损失影响；7)结合DDPM重构损失进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统分析扩散模型的梯度衰减问题；2)提出MoCLIP轻量级运动编码器，不依赖外部预训练模型；3)设计双路径时间感知语义锚定机制；4)实现更快的训练收敛和更高质量的运动生成。相比之前工作，LUMA的主要不同在于：不依赖外部大型预训练模型(如REPA依赖DINOv2)，引入双路径锚点(时间+频率)，通过FiLM调制实现时间感知的特征注入，解决了深层梯度衰减问题，同时保持了运动细节和语义一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LUMA通过双路径语义锚定机制解决了扩散模型在文本到运动生成中的深层梯度衰减问题，实现了更高质量的语义对齐和更快的训练收敛速度。'}


### 论文摘要

While current diffusion-based models, typically built on U-Net architectures, have shown promising results on the text-to-motion generation task, they still suffer from semantic misalignment and kinematic artifacts. Through analysis, we identify severe gradient attenuation in the deep layers of the network as a key bottleneck, leading to insufficient learning of high-level features. To address this issue, we propose \textbf{LUMA} (\textit{\textbf{L}ow-dimension \textbf{U}nified \textbf{M}otion \textbf{A}lignment}), a text-to-motion diffusion model that incorporates dual-path anchoring to enhance semantic alignment. The first path incorporates a lightweight MoCLIP model trained via contrastive learning without relying on external data, offering semantic supervision in the temporal domain. The second path introduces complementary alignment signals in the frequency domain, extracted from low-frequency DCT components known for their rich semantic content. These two anchors are adaptively fused through a temporal modulation mechanism, allowing the model to progressively transition from coarse alignment to fine-grained semantic refinement throughout the denoising process. Experimental results on HumanML3D and KIT-ML demonstrate that LUMA achieves state-of-the-art performance, with FID scores of 0.035 and 0.123, respectively. Furthermore, LUMA accelerates convergence by 1.4$\times$ compared to the baseline, making it an efficient and scalable solution for high-fidelity text-to-motion generation.

---

## 10. LayerLock: Non-collapsing Representation Learning with Progressive Freezing

**论文链接:** [http://arxiv.org/abs/2509.10156v3](http://arxiv.org/abs/2509.10156v3)

**作者:** Goker Erdogan, Nikhil Parthasarathy, Catalin Ionescu, Drew A. Hudson, Alexander Lerchner, Andrew Zisserman, Mehdi S. M. Sajjadi, Joao Carreira

**发布时间:** 2025-09-12

**备注:** ICCV 2025

### GPT解析

### 总结

LayerLock是一种简单有效的自监督视觉表征学习方法，通过渐进式层冻结从像素预测过渡到潜在预测。

### 背景

在视频掩码自编码(MAE)模型训练过程中，ViT层按照深度顺序收敛：浅层先收敛，深层后收敛。

### 目的

加速标准MAE训练，并开发一种简单可扩展的潜在预测方法，避免'表征崩溃'问题。

### 方法

LayerLock方法根据明确的训练进度表渐进式冻结模型，从像素预测过渡到潜在预测。

### 主要发现

ViT层在训练过程中按深度顺序收敛，这一观察可加速标准MAE，相同的进度表可用于避免'表征崩溃'的潜在预测。

### 结论

LayerLock方法应用于高达40亿参数的大模型，在4DS感知套件上的结果超过了非潜在掩码预测。

### 翻译

我们介绍了LayerLock，一种简单而有效的自监督视觉表征学习方法，它通过渐进式层冻结从像素预测过渡到潜在预测。首先，我们观察到在视频掩码自编码(MAE)模型训练过程中，ViT层按照其深度顺序收敛：浅层层先收敛，深层层后收敛。然后我们表明，这一观察可以通过在整个训练过程中根据明确的进度表渐进式冻结模型来加速标准MAE。此外，相同的进度表可用于一种简单且可扩展的潜在预测方法，不会遭受'表征崩溃'。我们将提出的方法LayerLock应用于高达40亿参数的大模型，在4DS感知套件上的结果超过了非潜在掩码预测。


### 论文摘要

We introduce LayerLock, a simple yet effective approach for self-supervised visual representation learning, that gradually transitions from pixel to latent prediction through progressive layer freezing. First, we make the observation that during training of video masked-autoencoding (MAE) models, ViT layers converge in the order of their depth: shallower layers converge early, deeper layers converge late. We then show that this observation can be exploited to accelerate standard MAE by progressively freezing the model according to an explicit schedule, throughout training. Furthermore, this same schedule can be used in a simple and scalable approach to latent prediction that does not suffer from "representation collapse". We apply our proposed approach, LayerLock, to large models of up to 4B parameters with results surpassing those of non-latent masked prediction on the 4DS perception suite.

---

## 11. MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2509.26642v1](http://arxiv.org/abs/2509.26642v1)

**作者:** Zhuoyang Liu, Jiaming Liu, Jiadong Xu, Nuowei Han, Chenyang Gu, Hao Chen, Kaichen Zhou, Renrui Zhang, Kai Chin Hsieh, Kun Wu, Zhengping Che, Jian Tang, Shanghang Zhang

**发布时间:** 2025-09-30

### GPT解析

### 总结

研究引入了多感官语言-动作(MLA)模型，通过协同感知异构感官模态和预测未来多感官目标促进物理世界建模。模型采用无需编码器的多模态对齐方案，将大语言模型作为感知模块直接解释2D图像、3D点云和触觉标记。同时设计了未来多感官生成后训练策略，增强模型对物理动态的理解。MLA在复杂接触密集任务中表现优异，比最先进方法分别提高12%和24%。

### 背景

视觉-语言-动作模型(VLA)通过继承视觉-语言模型(VLM)和学习动作生成，在机器人操作任务中展示了泛化能力。但大多数VLA模型仅关注解释视觉和语言生成动作，而忽略了机器人必须在空间-物理世界中感知和交互的本质需求，导致在实现复杂和接触密集控制时存在理解差距。

### 目的

解决VLA模型在理解机器人特定多感官信息方面的不足，引入能够协同感知异构感官模态并预测未来多感官目标以促进物理世界建模的多感官语言-动作(MLA)模型，提升机器人在复杂物理环境中的操作能力。

### 方法

1) 提出无需编码器的多模态对齐方案，将大语言模型本身作为感知模块，通过位置对应关系直接解释2D图像、3D点云和触觉标记；2) 设计未来多感官生成后训练策略，使MLA能够推理语义、几何和交互信息，为动作生成提供更稳健的条件。

### 主要发现

MLA模型在复杂、接触密集的真实世界任务中，比之前最先进的2D和3D VLA方法分别提高了12%和24%，同时展示了在未见配置上的泛化能力提升。

### 结论

MLA模型通过协同感知异构感官模态和预测未来多感官目标，有效提升了机器人在复杂物理环境中的操作能力。无需编码器的多模态对齐方案和未来多感官生成后训练策略共同增强了模型对物理动态的理解，为机器人在接触密集任务中的表现提供了显著改进。

### 翻译

视觉-语言-动作模型(VLA)通过继承视觉-语言模型(VLM)和学习动作生成，在机器人操作任务中展示了泛化能力。大多数VLA模型专注于解释视觉和语言以生成动作，而机器人必须在空间-物理世界中感知和交互。这种差距凸显了需要全面理解机器人特定的多感官信息，这对于实现复杂和接触密集的控制至关重要。为此，我们引入了一种多感官语言-动作(MLA)模型，该模型协同感知异构感官模态并预测未来多感官目标以促进物理世界建模。具体而言，为了增强感知表示，我们提出了一种无需编码器的多模态对齐方案，创新性地将大语言模型本身作为感知模块，通过位置对应关系直接解释2D图像、3D点云和触觉标记。为了进一步增强MLA对物理动态的理解，我们设计了一种未来多感官生成后训练策略，使MLA能够推理语义、几何和交互信息，为动作生成提供更稳健的条件。在评估中，MLA模型在复杂、接触密集的真实世界任务中，分别比之前最先进的2D和3D VLA方法提高了12%和24%，同时展示了在未见配置上的泛化能力提升。项目网站：https://sites.google.com/view/open-mla

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人操作中仅依赖2D图像和语言指令的局限性，使机器人能够全面理解和处理物理世界中的多模态信息（包括视觉、触觉和空间几何）。这个问题很重要，因为真实世界是三维物理环境，机器人需要多种感官信息来准确感知空间依赖性和物理动态，从而实现复杂和接触密集的操作任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视觉-语言-动作(VLA)模型的局限性，发现它们主要依赖2D图像，无法充分捕捉空间物理信息。因此，作者设计了两个关键创新：1)编码器自由的多模态对齐机制，直接利用大语言模型(LLM)作为感知模块；2)未来多感官生成后训练策略，增强物理动力学理解。作者借鉴了对比学习方法和世界知识预测策略，但创新性地应用于机器人多模态理解和预测任务，并采用了现有的LLaMA-2 7B模型作为基础进行改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过协同处理多种异构感官模态（视觉、触觉、空间几何）并预测它们未来的状态，来增强物理世界建模，改进机器人操作能力。整体流程包括：1)大规模预训练：在57万条轨迹的图像-动作配对数据集上预训练LLM；2)监督微调(SFT)：使用任务特定数据集微调，引入编码器自由多模态对齐机制；3)后训练：进行未来多感官生成后训练，使模型能够推理语义、几何和交互信息。架构上包括图像、点云、触觉标记化器，LLM骨干和未来预测解码器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)编码器自由的多模态对齐机制，直接利用LLM作为感知模块；2)未来多感官生成后训练策略，联合预测图像、点云和触觉的未来状态；3)渐进式训练流程，逐步集成感知、理解和动作生成能力。相比之前工作，MLA的主要不同在于：整合了2D视觉、3D几何和触觉信息；直接利用LLM而非模态特定编码器；不仅预测动作还预测未来多模态状态；在复杂现实任务中性能显著优于之前方法（高出12%-24%）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MLA通过整合多感官信息并预测其未来状态，显著提升了机器人在复杂物理环境中的操作能力和泛化性能。'}


### 论文摘要

Vision-language-action models (VLAs) have shown generalization capabilities in robotic manipulation tasks by inheriting from vision-language models (VLMs) and learning action generation. Most VLA models focus on interpreting vision and language to generate actions, whereas robots must perceive and interact within the spatial-physical world. This gap highlights the need for a comprehensive understanding of robotic-specific multisensory information, which is crucial for achieving complex and contact-rich control. To this end, we introduce a multisensory language-action (MLA) model that collaboratively perceives heterogeneous sensory modalities and predicts future multisensory objectives to facilitate physical world modeling. Specifically, to enhance perceptual representations, we propose an encoder-free multimodal alignment scheme that innovatively repurposes the large language model itself as a perception module, directly interpreting multimodal cues by aligning 2D images, 3D point clouds, and tactile tokens through positional correspondence. To further enhance MLA's understanding of physical dynamics, we design a future multisensory generation post-training strategy that enables MLA to reason about semantic, geometric, and interaction information, providing more robust conditions for action generation. For evaluation, the MLA model outperforms the previous state-of-the-art 2D and 3D VLA methods by 12% and 24% in complex, contact-rich real-world tasks, respectively, while also demonstrating improved generalization to unseen configurations. Project website: https://sites.google.com/view/open-mla

---

## 12. S$^3$E: Self-Supervised State Estimation for Radar-Inertial System

**论文链接:** [http://arxiv.org/abs/2509.25984v1](http://arxiv.org/abs/2509.25984v1)

**作者:** Shengpeng Wang, Yulong Xie, Qing Liao, Wei Wang

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种名为S³E的自监督状态估计器，利用雷达信号谱和惯性数据融合实现准确的状态估计，解决了雷达点云稀疏性、多路径效应和角度分辨率有限等问题。

### 背景

毫米波雷达在状态估计中因其经济性和在恶劣条件下的可靠性而受到广泛关注，但现有的定位解决方案通常依赖后处理的雷达点云作为地标点。

### 目的

解决雷达点云的内在稀疏性、多路径效应产生的鬼影点以及单线性雷达有限的角度分辨率等问题，提高状态估计性能。

### 方法

提出S³E自监督状态估计器，利用信息更丰富的雷达信号谱绕过稀疏点，融合互补的惯性信息实现准确定位；探索外部感知雷达和本体感知惯性传感器之间的关联；引入跨融合技术处理有限角度分辨率问题。

### 主要发现

实验结果表明，S³E方法在不依赖定位真实值监督的情况下实现了鲁棒且准确的性能。

### 结论

据作者所知，这是首次尝试通过互补的自监督方式融合雷达谱和惯性数据来实现状态估计。

### 翻译

毫米波雷达用于状态估计因其经济性和在恶劣条件下的可靠性而受到广泛关注。现有的定位解决方案通常依赖后处理的雷达点云作为地标点。然而，雷达点云的内在稀疏性、多路径效应产生的鬼影点以及单线性雷达有限的角度分辨率严重降低了状态估计性能。为解决这些问题，我们提出了S³E，一种自监督状态估计器，它利用信息更丰富的雷达信号谱来绕过稀疏点，并融合互补的惯性信息来实现准确定位。S³E充分探索了外部感知雷达和本体感知惯性传感器之间的关联，以实现互补效益。为了处理有限的角度分辨率，我们引入了一种新颖的跨融合技术，通过利用异构数据之间的细微旋转位移相关性来增强空间结构信息。实验结果表明，我们的方法在不依赖定位真实值监督的情况下实现了鲁棒且准确的性能。据我们所知，这是首次尝试通过互补的自监督方式融合雷达谱和惯性数据来实现状态估计。


### 论文摘要

Millimeter-wave radar for state estimation is gaining significant attention for its affordability and reliability in harsh conditions. Existing localization solutions typically rely on post-processed radar point clouds as landmark points. Nonetheless, the inherent sparsity of radar point clouds, ghost points from multi-path effects, and limited angle resolution in single-chirp radar severely degrade state estimation performance. To address these issues, we propose S$^3$E, a \textbf{S}elf-\textbf{S}upervised \textbf{S}tate \textbf{E}stimator that employs more richly informative radar signal spectra to bypass sparse points and fuses complementary inertial information to achieve accurate localization. S$^3$E fully explores the association between \textit{exteroceptive} radar and \textit{proprioceptive} inertial sensor to achieve complementary benefits. To deal with limited angle resolution, we introduce a novel cross-fusion technique that enhances spatial structure information by exploiting subtle rotational shift correlations across heterogeneous data. The experimental results demonstrate our method achieves robust and accurate performance without relying on localization ground truth supervision. To the best of our knowledge, this is the first attempt to achieve state estimation by fusing radar spectra and inertial data in a complementary self-supervised manner.

---

## 13. PinPoint3D: Fine-Grained 3D Part Segmentation from a Few Clicks

**论文链接:** [http://arxiv.org/abs/2509.25970v1](http://arxiv.org/abs/2509.25970v1)

**作者:** Bojun Zhang, Hangjian Ye, Hao Zheng, Jianzheng Huang, Zhengyu Lin, Zhenhong Guo, Feng Zheng

**发布时间:** 2025-09-30

**备注:** 15 pages, 12 figures, conference

### GPT解析

### 总结

PinPoint3D是一种新型交互式框架，用于精细粒度、多粒度3D分割，仅需少量用户点击即可生成精确的部件级掩码，解决了现有方法在稀疏点云和标注数据方面的局限性。

### 背景

精细的3D部件分割对具身AI系统执行复杂操作任务至关重要，但现有交互式分割方法主要局限于粗粒度的实例级目标，而非交互式方法难以处理稀疏的真实世界扫描数据且严重缺乏标注数据。

### 目的

解决现有交互式和非交互式3D分割方法的局限性，开发一种能够处理稀疏点云且不需要大量标注数据的精细粒度3D分割框架。

### 方法

引入PinPoint3D框架，其核心是一个新的3D数据合成管道，用于创建大规模、场景级且具有密集部件标注的数据集，使系统能够仅凭少量用户点击生成精确部件级掩码。

### 主要发现

PinPoint3D在首次点击设置下每个对象部件的平均IoU约为55.8%，仅需几次额外点击即可超过71.3%的IoU，与当前最先进的基线相比，在IoU和精度上提高了最多16%，在处理具有高效率的挑战性稀疏点云方面表现出色。

### 结论

PinPoint3D代表了在复杂3D环境中实现更细致精确的机器感知和交互的重要一步，为具身AI系统执行复杂操作任务提供了有效支持。

### 翻译

精细的3D部件分割对于使具身AI系统能够执行复杂操作任务（如与物体的特定功能部件交互）至关重要。然而，现有的交互式分割方法主要局限于粗粒度的实例级目标，而非交互式方法难以处理稀疏的真实世界扫描数据且严重缺乏标注数据。为解决这些限制，我们引入了PinPoint3D，一种用于精细粒度、多粒度3D分割的新型交互式框架，能够仅凭少量用户点击生成精确的部件级掩码。我们工作的一个关键组成部分是我们开发的一个新的3D数据合成管道，用于创建具有密集部件标注的大规模场景级数据集，克服了阻碍该领域进展的关键瓶颈。通过全面实验和用户研究，我们证明我们的方法显著优于现有方法，在首次点击设置下每个对象部件的平均IoU约为55.8%，仅需几次额外点击即可超过71.3%的IoU。与当前最先进的基线相比，PinPoint3D在IoU和精度上提高了最多16%，突显了其在处理具有高效率的挑战性稀疏点云方面的有效性。我们的工作代表了在复杂3D环境中实现更细致精确的机器感知和交互的重要一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云中细粒度部分分割的问题，特别是如何通过少量用户点击实现精确的部分级分割。这个问题在现实中非常重要，因为它能让具身AI系统(如服务机器人)识别并操作物体的特定功能部件(如抽屉把手)，而不仅仅是识别整个物体。这种精细的理解能力对于执行复杂操作任务至关重要，是AI系统实现类人水平交互的关键一步。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有交互式3D分割方法的局限性，如只能处理粗粒度实例级分割，以及非交互式方法在处理稀疏真实世界扫描数据时的不足。他们借鉴了AGILE3D的交互式训练范式和3D稀疏卷积网络作为特征提取骨干，并参考了PartField模型用于生成伪标签。在此基础上，作者设计了专门针对部分分割的框架，包括两级transformer解码器架构和目标注意力掩码机制，同时开发了新的数据合成方法来克服数据稀缺问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'PinPoint3D的核心思想是创建一个交互式框架，通过少量用户点击即可实现对3D点云的细粒度部分分割。整体流程包括：1)使用稀疏卷积网络提取点特征，并通过轻量级适配器模块调整特征维度；2)将用户点击编码为可学习的查询特征，保留空间和时间信息；3)通过两级transformer解码器处理场景-实例和实例-部分的分割；4)使用目标注意力掩码(TAM)约束部分查询在目标物体内操作；5)支持迭代细化，用户可添加更多点击提高分割精度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)专为细粒度部分分割设计的交互式框架，仅需少量点击即可生成精确部分掩码；2)全新的3D数据合成管道，创建大规模场景级数据集解决数据稀缺问题；3)直观的交互机制和用户界面；4)强大的跨数据集泛化能力。相比之前的工作，PinPoint3D专门处理部分级而非仅实例级分割，能够处理稀疏场景级点云，采用专门的部分级分支和注意力机制，并引入TAM确保部分查询在目标对象内操作，防止跨对象干扰。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PinPoint3D通过创新的交互式框架和数据合成方法，实现了仅需少量点击即可对3D点云进行精确细粒度部分分割，显著提升了具身AI系统对复杂环境中物体组件的理解和交互能力。'}


### 论文摘要

Fine-grained 3D part segmentation is crucial for enabling embodied AI systems to perform complex manipulation tasks, such as interacting with specific functional components of an object. However, existing interactive segmentation methods are largely confined to coarse, instance-level targets, while non-interactive approaches struggle with sparse, real-world scans and suffer from a severe lack of annotated data. To address these limitations, we introduce PinPoint3D, a novel interactive framework for fine-grained, multi-granularity 3D segmentation, capable of generating precise part-level masks from only a few user point clicks. A key component of our work is a new 3D data synthesis pipeline that we developed to create a large-scale, scene-level dataset with dense part annotations, overcoming a critical bottleneck that has hindered progress in this field. Through comprehensive experiments and user studies, we demonstrate that our method significantly outperforms existing approaches, achieving an average IoU of around 55.8% on each object part under first-click settings and surpassing 71.3% IoU with only a few additional clicks. Compared to current state-of-the-art baselines, PinPoint3D yields up to a 16% improvement in IoU and precision, highlighting its effectiveness on challenging, sparse point clouds with high efficiency. Our work represents a significant step towards more nuanced and precise machine perception and interaction in complex 3D environments.

---

## 14. LiDAR Point Cloud Colourisation Using Multi-Camera Fusion and Low-Light Image Enhancement

**论文链接:** [http://arxiv.org/abs/2509.25859v1](http://arxiv.org/abs/2509.25859v1)

**作者:** Pasindu Ranasinghe, Dibyayan Patra, Bikram Banerjee, Simit Raval

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究提出了一种新颖的、与硬件无关的方法，通过融合相机数据与LiDAR测量来增强空间理解，特别是在低光照条件下生成彩色点云。

### 背景

近年来，相机数据与LiDAR测量的融合已成为增强空间理解的有效方法，但传统方法在低光照条件下存在局限性。

### 目的

开发一种能够在低光照条件下可靠工作的彩色点云生成方法，提供完整的360度覆盖，并简化校准过程。

### 方法

提出一种硬件无关的方法，使用多个相机输入为机械式LiDAR生成彩色点云；集成低光照图像增强模块提高鲁棒性；通过初始校准确定相机内参，并自动计算LiDAR与相机间的几何变换；使用色彩校正确保相机馈送的一致性。

### 主要发现

系统在Velodyne Puck Hi-Res LiDAR和四相机配置测试中表现良好；优化后的软件实现了实时性能；即使在非常低的光照条件下也能可靠地进行彩色化，成功恢复原本无法检测到的场景细节。

### 结论

该方法有效解决了低光照条件下彩色点云生成的挑战，无需专门校准目标简化了设置，为空间理解提供了新的解决方案。

### 翻译

近年来，相机数据与LiDAR测量的融合已成为增强空间理解的有效方法。本研究引入了一种新颖的、与硬件无关的方法，使用多个相机输入为机械式LiDAR生成彩色点云，提供完整的360度覆盖。主要创新点在于其在低光照条件下的鲁棒性，这是通过在融合管道中集成低光照图像增强模块实现的。系统需要初始校准以确定相机内参，然后自动计算LiDAR和相机之间的几何变换，无需专门的校准目标，简化了设置。数据处理框架使用色彩校正以确保相机馈送的一致性，然后再进行融合。该算法使用Velodyne Puck Hi-Res LiDAR和四相机配置进行了测试。优化后的软件实现了实时性能，即使在非常低的光照条件下也能可靠地进行彩色化，成功恢复了原本无法检测到的场景细节。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云着色问题，特别是在低光照条件下的着色问题。LiDAR技术虽然能提供精确的距离测量，但其产生的点云是单色的，缺乏颜色信息，这限制了人类和算法对场景的解释能力。这个问题在自动驾驶中妨碍了对道路标志和交通信号的识别，在遥感应用中影响植被分类和地形分析，在地下矿井和夜间导航等低光照环境中，可靠的颜色信息对于安全监控和导航至关重要。完整的颜色信息可以提高点云的语义解释性，使下游任务更加准确有效。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过多角度思考设计了这个方法：首先意识到单摄像头无法提供完整覆盖，因此采用多摄像头系统；其次开发自动化校准方法消除对专门校准目标的需求；然后集成低光照增强模块解决低光照条件下的着色问题；同时设计颜色校正确保跨摄像头一致性；最后优化实现实时性能。该方法借鉴了多项现有工作：使用OpenCV进行摄像头内参校准；受Yoon等人方法启发开发外参校准；采用基于U-Net架构的模型进行低光照增强；使用标准点云处理技术如ICP算法、DBSCAN聚类等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多摄像头融合实现完整360度覆盖，结合自动化校准和低光照增强技术，确保在各种光照条件下都能获得高质量的颜色信息。整体流程分为两个阶段：初始校准阶段（包括摄像头内参校准、LiDAR-摄像头外参校准和颜色校正校准）和数据处理阶段（数据采集与同步、图像预处理包括模糊过滤、颜色校正和低光照增强、数据融合将LiDAR点投影到图像并分配颜色、输出着色点云）。通过向量化计算和C++实现确保实时性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 多摄像头实现完整360度覆盖，解决了单摄像头方法中大部分区域无法着色的问题；2) 自动化、无需目标的LiDAR-摄像头校准方法，简化了设置过程；3) 低光照条件下的增强性能，能在极低光照（0.5勒克斯）下恢复场景细节；4) 颜色校正确保跨摄像头一致性；5) 实时性能优化，实现10Hz处理速率。相比之前工作，不同之处在于：覆盖范围更完整（几乎100%点被着色vs 传统方法的有限覆盖）；校准更自动化（无需专门目标vs 传统方法需要）；低光照性能更好（能在极低光照下工作vs 传统方法表现差）；硬件通用性更强（可与不同摄像头配合使用）；实时性能更高（10Hz vs 许多现有方法的非实时处理）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文开发了一种结合多摄像头融合和低光照图像增强的端到端方法，实现了LiDAR点云在低光照条件下的完整360度实时着色，显著提高了点云的语义解释性并简化了部署过程。'}


### 论文摘要

In recent years, the fusion of camera data with LiDAR measurements has emerged as a powerful approach to enhance spatial understanding. This study introduces a novel, hardware-agnostic methodology that generates colourised point clouds from mechanical LiDAR using multiple camera inputs, providing complete 360-degree coverage. The primary innovation lies in its robustness under low-light conditions, achieved through the integration of a low-light image enhancement module within the fusion pipeline. The system requires initial calibration to determine intrinsic camera parameters, followed by automatic computation of the geometric transformation between the LiDAR and cameras, removing the need for specialised calibration targets and streamlining the setup. The data processing framework uses colour correction to ensure uniformity across camera feeds before fusion. The algorithm was tested using a Velodyne Puck Hi-Res LiDAR and a four-camera configuration. The optimised software achieved real-time performance and reliable colourisation even under very low illumination, successfully recovering scene details that would otherwise remain undetectable.

---

## 15. From Cheap Geometry to Expensive Physics: Elevating Neural Operators via Latent Shape Pretraining

**论文链接:** [http://arxiv.org/abs/2509.25788v1](http://arxiv.org/abs/2509.25788v1)

**作者:** Zhizhou Zhang, Youjia Wu, Kaixuan Zhang, Yanjia Wang

**发布时间:** 2025-09-30

### GPT解析

### 总结

论文提出了一种两阶段框架，利用无物理标记的几何设计数据来改进监督算子学习，提高PDE解决方案预测的准确性。

### 背景

工业设计评估通常依赖高保真的偏微分方程模拟，但这些模拟计算成本高，使得密集探索设计空间不切实际。算子学习已成为加速PDE解决方案预测的有前途的方法，但其有效性常受限于基于物理的标记数据稀缺。

### 目的

提出一个两阶段框架，更好地利用丰富的无物理资源，并在有限标记数据下改进监督算子学习。

### 方法

第一阶段，在几何重建任务上预训练自编码器，学习无需PDE标记的表达性潜在表示；第二阶段，以标准监督方式训练神经算子预测PDE解决方案，使用预训练的潜在嵌入作为输入而非原始点云。两个阶段都采用基于Transformer的架构来处理点云数据并无缝集成。

### 主要发现

在四个PDE数据集和三种最先进的基于Transformer的神经算子上，与直接在原始点云输入上训练的模型相比，该方法一致提高了预测准确性。

### 结论

来自无物理预训练的表示为数据高效的算子学习提供了强大的基础。

### 翻译

工业设计评估通常依赖于对控制偏微分方程的高保真模拟。虽然这些模拟准确，但计算成本高，使得密集探索设计空间不切实际。算子学习已成为加速PDE解决方案预测的有前途的方法；然而，其有效性常受限于基于物理的标记数据稀缺。与此同时，大量仅包含几何信息的候选设计方案很容易获得，但很大程度上未被充分利用。我们提出一个两阶段框架，以更好地利用这种丰富、无物理的资源，并在有限标记数据下改进监督算子学习。在第一阶段，我们在几何重建任务上预训练自编码器，学习无需PDE标记的表达性潜在表示。在第二阶段，以标准监督方式训练神经算子预测PDE解决方案，使用预训练的潜在嵌入作为输入而非原始点云。两个阶段都采用基于Transformer的架构来处理点云数据并无缝集成。在四个PDE数据集和三种最先进的基于Transformer的神经算子上，与直接在原始点云输入上训练的模型相比，我们的方法一致提高了预测准确性。这些结果表明，来自无物理预训练的表示为数据高效的算子学习提供了强大的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决工业设计评估中物理模拟计算成本高与标记数据稀缺的问题。在工业设计中，高保真度的物理模拟虽然准确但计算极其昂贵，使得密集探索设计空间不切实际；同时，算子学习虽能加速PDE解预测，但受限于基于物理的标记数据稀缺。大量只有几何信息(无物理标签)的候选设计数据丰富但未被充分利用，这一问题在工业研发中尤为重要，因为它直接影响产品设计的效率和成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到工业设计中的痛点：物理模拟计算成本高，而几何数据丰富但物理标记数据稀缺。观察到算子学习效果受限于标记数据稀缺，意识到大量几何数据未被充分利用。借鉴了计算机视觉和自然语言处理中的预训练技术(如BERT、MAE)，以及点云变分自编码器(VAE)用于3D重建的方法。作者设计了Transformer架构来处理点云数据并整合两个阶段，最终提出两阶段框架：第一阶段利用丰富的几何数据进行预训练学习潜在表示，第二阶段在有限物理标记数据上微调神经算子。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大量廉价的几何数据进行预训练，学习有意义的几何表示，然后在有限的物理标记数据上微调神经算子，提高数据效率和预测准确性。整体流程分为两个阶段：第一阶段是几何预训练，使用点云变分自编码器(VAE)进行占用率场(occupancy field)重建这一物理无关的代理任务，编码器将点云信息聚合到固定长度向量并投影到潜在空间，解码器预测查询点的占用值来重建几何；第二阶段是算子学习，使用预训练的潜在表示作为神经算子输入，在有限的物理标记数据上训练预测PDE解，训练过程中保持预训练编码器冻结。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)提出两阶段训练框架，第一阶段通过代理任务进行几何预训练，第二阶段微调神经算子；(2)引入占用率重建作为代理任务，允许在不规则几何上进行自监督表示学习；(3)充分利用无标签几何数据，解决物理模拟数据稀缺问题；(4)采用Transformer架构处理点云并整合两阶段。相比之前工作，该方法不依赖于特定网格结构或物理类型，适用性更广；不同于现有基于固定网格结构的预训练方法，以及需要大规模物理数据集的预训练方法，本文方法能够利用大量廉价的几何数据，无需物理标签，更具实用性和通用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种两阶段训练框架，通过利用大量廉价的几何数据进行物理无关的预训练，学习有意义的几何表示，然后在有限的物理标记数据上微调神经算子，显著提高了数据效率和预测准确性，解决了工业设计中物理模拟计算成本高和标记数据稀缺的问题。'}


### 论文摘要

Industrial design evaluation often relies on high-fidelity simulations of governing partial differential equations (PDEs). While accurate, these simulations are computationally expensive, making dense exploration of design spaces impractical. Operator learning has emerged as a promising approach to accelerate PDE solution prediction; however, its effectiveness is often limited by the scarcity of labeled physics-based data. At the same time, large numbers of geometry-only candidate designs are readily available but remain largely untapped. We propose a two-stage framework to better exploit this abundant, physics-agnostic resource and improve supervised operator learning under limited labeled data. In Stage 1, we pretrain an autoencoder on a geometry reconstruction task to learn an expressive latent representation without PDE labels. In Stage 2, the neural operator is trained in a standard supervised manner to predict PDE solutions, using the pretrained latent embeddings as inputs instead of raw point clouds. Transformer-based architectures are adopted for both the autoencoder and the neural operator to handle point cloud data and integrate both stages seamlessly. Across four PDE datasets and three state-of-the-art transformer-based neural operators, our approach consistently improves prediction accuracy compared to models trained directly on raw point cloud inputs. These results demonstrate that representations from physics-agnostic pretraining provide a powerful foundation for data-efficient operator learning.

---

## 16. VGGT-X: When VGGT Meets Dense Novel View Synthesis

**论文链接:** [http://arxiv.org/abs/2509.25191v1](http://arxiv.org/abs/2509.25191v1)

**作者:** Yang Liu, Chuanchen Luo, Zimo Tang, Junran Peng, Zhaoxiang Zhang

**发布时间:** 2025-09-29

**备注:** Project Page: https://dekuliutesla.github.io/vggt-x.github.io/

### GPT解析

### 总结

该研究解决了将3D基础模型应用于密集新视角合成的问题，提出了VGGT-X方法，解决了VRAM负担和输出质量问题，实现了与COLMAP初始化流程相当的性能，并在密集无COLMAP的新视角合成和姿态估计中达到了最先进水平。

### 背景

基于NeRF和3DGS的新视角合成虽然取得了显著进展，但仍依赖于从运动恢复结构(SfM)获取准确的3D属性，这种方法在低纹理或低重叠场景中效率低且不稳定。虽然最近的3DFMs在速度上有显著优势，但大多数验证仅限于稀疏视角设置。

### 目的

研究将3D基础模型扩展到密集视角合成场景，解决面临的VRAM负担急剧增加和输出不完美导致初始化敏感的3D训练性能下降这两个核心障碍。

### 方法

引入VGGT-X方法，包含三个关键组件：(1)内存高效的VGGT实现，可扩展到处理1000+图像；(2)自适应全局对齐技术，用于增强VGGT输出；(3)稳健的3DGS训练实践。

### 主要发现

实验表明，所提出的方法显著缩小了与COLMAP初始化流程的保真度差距，在密集无COLMAP的新视角合成和姿态估计任务中实现了最先进的结果。研究还分析了与COLMAP初始化渲染仍存在差距的原因。

### 结论

VGGT-X方法成功解决了3D基础模型在密集视角合成中的两个核心障碍，为3D基础模型和密集新视角合成的未来发展提供了有价值的见解和实践指导。

### 翻译

该论文研究了将3D基础模型(3DFMs)应用于密集新视角合成(NVS)的问题。尽管基于NeRF和3DGS的新视角合成取得了显著进展，但当前方法仍然依赖于从运动恢复结构(SfM)获取准确的3D属性（如相机姿态和点云），这在低纹理或低重叠捕获中通常速度慢且脆弱。最近的3DFMs相比传统流程显示出数量级的加速，并有在线NVS的巨大潜力。但大多数验证和结论仅限于稀疏视角设置。我们的研究表明，简单地将3DFMs扩展到密集视图会遇到两个基本障碍：VRAM负担急剧增加和输出不完美，导致初始化敏感的3D训练性能下降。为解决这些问题，我们引入了VGGT-X，包括一个可扩展到1000+图像的内存高效VGGT实现，用于增强VGGT输出的自适应全局对齐，以及稳健的3DGS训练实践。大量实验表明，这些措施显著缩小了与COLMAP初始化流程的保真度差距，在密集的COLMAP-free NVS和姿态估计中实现了最先进的结果。此外，我们分析了与COLMAP初始化渲染仍存在差距的原因，为3D基础模型和密集NVS的未来发展提供了见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何将3D基础模型（如VGGT）扩展到密集新视角合成（dense NVS）的问题。当处理大量图像（1000+）时，存在两个主要障碍：一是VRAM负担急剧增加，二是输出质量不完美。这个问题很重要，因为传统方法（如NeRF和3DGS）依赖准确相机姿态和点云，通常需要耗时的结构从运动（SfM）方法获取，而3D基础模型虽快但大多仅适用于稀疏视图。实现密集视图合成对于快速、可靠的3D重建和视角合成至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了两个主要障碍：VRAM负担和输出质量。为解决VRAM问题，他们设计了内存优化技术：消除冗余特征缓存、降低数值精度（BFloat16）、采用批处理帧操作。为解决输出质量问题，他们提出自适应全局对齐改进VGGT输出，并采用MCMC-3DGS提高鲁棒性，同时进行联合姿态优化。作者借鉴了多项现有工作：VGGT作为基础模型，XFeat用于特征匹配，MCMC-3DGS提供鲁棒训练策略，以及传统Bundle Adjustment的思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过内存优化使VGGT能处理大量图像，通过自适应全局对齐提高输出质量，并利用鲁棒的3DGS训练策略处理不完美初始化。整体流程：1）内存高效VGGT实现（消除冗余特征、降低精度、批处理帧操作）；2）相机参数全局对齐（使用XFeat匹配、自适应权重策略、自适应学习率）；3）不完美姿态下的3DGS训练（采用MCMC-3DGS、联合优化残差姿态、使用高置信度点云初始化）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）内存高效的VGGT实现，支持1000+图像推理；2）自适应全局对齐，基于极线距离分布的权重策略和自适应学习率；3）鲁棒的3DGS训练策略，结合MCMC和联合优化。相比之前工作的不同：相比原始VGGT，大幅降低内存需求并提高处理能力；相比其他3D基础模型，更高效处理密集视图；相比其他COLMAP-free方法，在渲染质量和姿态估计上取得更好结果；还分析了与COLMAP初始化的差距并提供改进方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VGGT-X通过内存优化的VGGT实现、自适应全局对齐和鲁棒的3DGS训练策略，成功将3D基础模型扩展到密集新视角合成，显著缩小了与COLMAP初始化方法的性能差距，实现了快速、可靠的完全COLMAP-free密集NVS系统。'}


### 论文摘要

We study the problem of applying 3D Foundation Models (3DFMs) to dense Novel View Synthesis (NVS). Despite significant progress in Novel View Synthesis powered by NeRF and 3DGS, current approaches remain reliant on accurate 3D attributes (e.g., camera poses and point clouds) acquired from Structure-from-Motion (SfM), which is often slow and fragile in low-texture or low-overlap captures. Recent 3DFMs showcase orders of magnitude speedup over the traditional pipeline and great potential for online NVS. But most of the validation and conclusions are confined to sparse-view settings. Our study reveals that naively scaling 3DFMs to dense views encounters two fundamental barriers: dramatically increasing VRAM burden and imperfect outputs that degrade initialization-sensitive 3D training. To address these barriers, we introduce VGGT-X, incorporating a memory-efficient VGGT implementation that scales to 1,000+ images, an adaptive global alignment for VGGT output enhancement, and robust 3DGS training practices. Extensive experiments show that these measures substantially close the fidelity gap with COLMAP-initialized pipelines, achieving state-of-the-art results in dense COLMAP-free NVS and pose estimation. Additionally, we analyze the causes of remaining gaps with COLMAP-initialized rendering, providing insights for the future development of 3D foundation models and dense NVS. Our project page is available at https://dekuliutesla.github.io/vggt-x.github.io/

---

## 17. Wavelet-Induced Rotary Encodings: RoPE Meets Graphs

**论文链接:** [http://arxiv.org/abs/2509.22259v2](http://arxiv.org/abs/2509.22259v2)

**作者:** Isaac Reid, Arijit Sehanobish, Cederik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E. Turner, Petar Veličković

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究提出了WIRE（小波诱导的旋转编码），一种将旋转位置编码（RoPE）扩展到图结构数据的新方法。WIRE比RoPE更通用，在网格图的特例中可以恢复RoPE，并具有多种理想的理论性质。实验证明在底层图结构重要的场景中，WIRE是有效的。

### 背景

旋转位置编码（RoPE）是一种在大型语言模型（LLMs）和视觉Transformer（ViTs）中广泛使用的流行算法，但在图结构数据上的应用有限。

### 目的

将RoPE算法扩展到图结构数据上，开发一种更通用的位置编码方法，适用于图神经网络和图结构数据处理。

### 方法

研究提出了WIRE（小波诱导的旋转编码），通过小波诱导的旋转编码将RoPE扩展到图结构数据。WIRE具有节点排序置换下的等变性、与线性注意力兼容，以及在特定假设下对图电阻距离的渐近依赖性等理论性质。

### 主要发现

1. WIRE比RoPE更通用，在网格图的特例中可以恢复RoPE；2. WIRE具有多种理想的理论性质，包括等变性、兼容性和渐近依赖性；3. 在识别单色子图、点云语义分割和图基准测试等任务中表现有效；4. WIRE在底层图结构重要的场景中特别有效。

### 结论

WIRE成功将RoPE扩展到图结构数据，提供了一种更通用的位置编码方法，适用于图神经网络和相关任务。实验证明在需要考虑图结构信息的应用中，WIRE是有效的解决方案。

### 翻译

我们介绍WIRE：小波诱导的旋转编码。WIRE将旋转位置编码（RoPE）——一种在大型语言模型和视觉Transformer中流行的算法——扩展到图结构数据。我们证明WIRE比RoPE更通用，在网格图的特例中可以恢复后者。WIRE还具有许多理想的理论性质，包括在节点排序置换下的等变性、与线性注意力的兼容性，以及在特定假设下对图电阻距离的渐近依赖性。我们在一系列合成和真实世界任务上测试了WIRE，包括识别单色子图、点云的语义分割以及更标准的图基准测试。我们发现当底层图结构很重要时，WIRE是有效的。


### 论文摘要

We introduce WIRE: Wavelet-Induced Rotary Encodings. WIRE extends Rotary Position Encodings (RoPE), a popular algorithm in LLMs and ViTs, to graph-structured data. We demonstrate that WIRE is more general than RoPE, recovering the latter in the special case of grid graphs. WIRE also enjoys a host of desirable theoretical properties, including equivariance under node ordering permutation, compatibility with linear attention, and (under select assumptions) asymptotic dependence on graph resistive distance. We test WIRE on a range of synthetic and real-world tasks, including identifying monochromatic subgraphs, semantic segmentation of point clouds, and more standard graph benchmarks. We find it to be effective in settings where the underlying graph structure is important.

---

## 18. MUSE-Explainer: Counterfactual Explanations for Symbolic Music Graph Classification Models

**论文链接:** [http://arxiv.org/abs/2509.26521v1](http://arxiv.org/abs/2509.26521v1)

**作者:** Baptiste Hilaire, Emmanouil Karystinaios, Gerhard Widmer

**发布时间:** 2025-09-30

**备注:** Accepted at the 17th International Symposium on Computer Music  Multidisciplinary Research (CMMR) 2025

### GPT解析

### 总结

这项研究提出了MUSE-Explainer方法，用于提高音乐分析中深度学习模型的可解释性，通过生成反事实解释来揭示音乐图神经网络模型的决策过程。

### 背景

在符号音乐分析中部署深度学习模型时，可解释性至关重要。然而，大多数研究强调模型性能而非解释能力。

### 目的

开发一种新方法，帮助揭示音乐图神经网络模型如何做出决策，并提供清晰、易于人类理解的解释。

### 方法

MUSE-Explainer通过生成反事实解释工作，即对音乐图谱进行小而有意义的修改，这些修改会改变模型的预测，同时确保结果在音乐上保持连贯。与现有方法不同，MUSE-Explainer根据音乐数据的结构定制解释，避免产生不切实际或令人困惑的输出。

### 主要发现

在音乐分析任务上评估该方法，表明它提供了可以通过Verovio等标准音乐工具可视化的直观见解。

### 结论

MUSE-Explainer为音乐分析中的深度学习模型提供了更好的可解释性，使研究人员能够理解模型如何做出音乐分析决策。

### 翻译

可解释性对于在符号音乐分析中部署深度学习模型至关重要，但大多数研究强调模型性能而非解释。为此，我们引入了MUSE-Explainer，一种新方法，它通过提供清晰、易于人类理解的解释，帮助揭示音乐图神经网络模型如何做出决策。我们的方法通过进行小的、有意义的音乐图谱修改来生成反事实解释，这些修改会改变模型的预测，同时确保结果在音乐上保持连贯。与现有方法不同，MUSE-Explainer根据音乐数据的结构定制其解释，避免产生不切实际或令人困惑的输出。我们在音乐分析任务上评估了我们的方法，并表明它提供了可以通过Verovio等标准音乐工具可视化的直观见解。


### 论文摘要

Interpretability is essential for deploying deep learning models in symbolic music analysis, yet most research emphasizes model performance over explanation. To address this, we introduce MUSE-Explainer, a new method that helps reveal how music Graph Neural Network models make decisions by providing clear, human-friendly explanations. Our approach generates counterfactual explanations by making small, meaningful changes to musical score graphs that alter a model's prediction while ensuring the results remain musically coherent. Unlike existing methods, MUSE-Explainer tailors its explanations to the structure of musical data and avoids unrealistic or confusing outputs. We evaluate our method on a music analysis task and show it offers intuitive insights that can be visualized with standard music tools such as Verovio.

---

## 19. Regression Language Models for Code

**论文链接:** [http://arxiv.org/abs/2509.26476v1](http://arxiv.org/abs/2509.26476v1)

**作者:** Yash Akhauri, Xingyou Song, Arissa Wongpanich, Bryan Lewandowski, Mohamed S. Abdelfattah

**发布时间:** 2025-09-30

### GPT解析

### 总结

研究提出了一种统一的回归语言模型(RLM)，能够直接从代码文本预测多种编程指标，包括内存占用、延迟和神经网络性能，无需复杂的特征工程。

### 背景

代码到指标回归是一个具有挑战性的任务，因为编程语言具有开放性特征。先前的方法依赖于复杂且特定领域的特征工程，限制了其通用性和效率。

### 目的

开发一种能够直接从代码文本预测多种数值指标的统一模型，无需针对不同任务进行特定的特征工程。

### 方法

提出并实现了一个单一的统一回归语言模型(RLM)，该模型基于T5Gemma初始化，具有300M参数，能够直接从代码文本预测多种指标。

### 主要发现

1) 300M参数的RLM在APPS编程竞赛提交中获得>0.9的Spearman秩；2) 单一模型在CodeNet的17种不同语言上获得>0.5的平均Spearman秩；3) 在五个经典NAS设计空间上获得0.46的最高平均Kendall-Tau，超越之前由图神经网络主导的结果；4) 能够同时预测多种硬件平台上的架构延迟。

### 结论

统一的回归语言模型能够有效解决代码到指标回归问题，无需复杂的特征工程，在多种编程语言和指标预测任务上表现出色，为编程分析和优化提供了新方法。

### 翻译

我们研究代码到指标回归：预测代码执行的数值结果，这是一个具有挑战性的任务，因为编程语言具有开放性特征。虽然先前的方法依赖于复杂且特定领域的特征工程，但我们表明，单一的统一回归语言模型(RLM)可以直接从文本同时预测(i)多种高级语言(如Python和C++)的代码内存占用，(ii)Triton GPU内核的延迟，以及(iii)以ONNX格式表示的训练神经网络的准确性和速度。特别是，一个相对较小的300M参数RLM(从T5Gemma初始化)在APPS的编程竞赛提交中获得>0.9的Spearman秩，并且单一统一模型在CodeNet的17种不同语言上获得>0.5的平均Spearman秩。此外，RLM在五个经典的NAS设计空间上获得了0.46的最高平均Kendall-Tau，这些空间之前由图神经网络主导，并且能够同时在众多硬件平台上预测架构延迟。


### 论文摘要

We study code-to-metric regression: predicting numeric outcomes of code executions, a challenging task due to the open-ended nature of programming languages. While prior methods have resorted to heavy and domain-specific feature engineering, we show that a single unified Regression Language Model (RLM) can simultaneously predict directly from text, (i) the memory footprint of code across multiple high-level languages such as Python and C++, (ii) the latency of Triton GPU kernels, and (iii) the accuracy and speed of trained neural networks represented in ONNX. In particular, a relatively small 300M parameter RLM initialized from T5Gemma, obtains > 0.9 Spearman-rank on competitive programming submissions from APPS, and a single unified model achieves > 0.5 average Spearman-rank across 17 separate languages from CodeNet. Furthermore, the RLM can obtain the highest average Kendall-Tau of 0.46 on five classic NAS design spaces previously dominated by graph neural networks, and simultaneously predict architecture latencies on numerous hardware platforms.

---

## 20. Graph Neural Network Acceleration on FPGAs for Fast Inference in Future Muon Triggers at HL-LHC

**论文链接:** [http://arxiv.org/abs/2509.26419v1](http://arxiv.org/abs/2509.26419v1)

**作者:** Martino Errico, Davide Fiacco, Stefano Giagu, Giuliano Gustavino, Valerio Ippolito, Graziella Russo

**发布时间:** 2025-09-30

**备注:** 5 pages, 2 figures Submission to SciPost for conference proceedings

### GPT解析

### 总结

本研究探索了机器学习方法在高亮度大型强子对撞机未来μ子触发中的应用，特别关注图神经网络在稀疏探测器数据处理中的优势。

### 背景

高亮度大型强子对撞机(HL-LHC)将达到比之前运行高7倍的亮度，产生更密集的事件和更大的占有率，这对触发算法提出了更高要求。

### 目的

开发下一代触发算法，使其在严格的延迟预算内保持可靠的选择能力。

### 方法

使用ATLAS μ子谱仪作为基准，采用卷积神经网络(CNN)作为参考，并引入图神经网络(GNN)作为稀疏探测器数据的自然模型进行对比研究。

### 主要发现

初步的单径迹研究表明，图神经网络能够以紧凑的架构实现高效率。

### 结论

图神经网络在μ子触发应用中展现出良好前景，特别是在考虑现场可编程门阵列(FPGA)部署的情况下。

### 翻译

高亮度大型强子对撞机(HL-LHC)将达到比之前运行高7倍的亮度，产生更密集的事件和更大的占有率。下一代触发算法必须在严格的延迟预算内保持可靠的选择。本研究探索了用于未来μ子触发的机器学习方法，使用ATLAS μ子谱仪作为基准。使用卷积神经网络(CNN)作为参考，同时引入图神经网络(GNN)作为稀疏探测器数据的自然模型。初步的单径迹研究表明，GNNs能够以紧凑的架构实现高效率，这是一个令人鼓舞的结果，特别是在考虑FPGA部署的情况下。


### 论文摘要

The High-Luminosity LHC (HL-LHC) will reach luminosities up to 7 times higher than the previous run, yielding denser events and larger occupancies. Next generation trigger algorithms must retain reliable selection within a strict latency budget. This work explores machine-learning approaches for future muon triggers, using the ATLAS Muon Spectrometer as a benchmark. A Convolutional Neural Network (CNN) is used as a reference, while a Graph Neural Network (GNN) is introduced as a natural model for sparse detector data. Preliminary single-track studies show that GNNs achieve high efficiency with compact architectures, an encouraging result in view of FPGA deployment.

---

## 21. MC-GNNAS-Dock: Multi-criteria GNN-based Algorithm Selection for Molecular Docking

**论文链接:** [http://arxiv.org/abs/2509.26377v1](http://arxiv.org/abs/2509.26377v1)

**作者:** Siyuan Cao, Hongxuan Wu, Jiabao Brad Wang, Yiliang Yuan, Mustafa Misir

**发布时间:** 2025-09-30

**备注:** Short paper. Preprint of a forthcoming conference contribution

### GPT解析

### 总结

本研究提出了MC-GNNAS-Dock系统，通过多标准评估、架构改进和排名感知损失函数三种关键改进，显著提升了分子对接算法选择的性能。

### 背景

分子对接是药物发现中预测配体-靶点相互作用的核心工具，但现有算法在不同上下文中的表现不一，没有单一算法能始终占优。基于图神经网络的算法选择框架如GNNAS-Dock被提出应对这一挑战。

### 目的

开发一个增强的系统MC-GNNAS-Dock，克服单一对接算法在不同场景下表现不一致的问题，提高算法选择的准确性和可靠性。

### 方法

MC-GNNAS-Dock系统包含三大改进：1)多标准评估结合结合姿态准确性和PoseBusters有效性检查；2)通过引入残差连接优化架构增强预测鲁棒性；3)整合排名感知损失函数强化排名学习。研究在包含约3200个PDBBind蛋白质-配体复合物的数据集上进行了大量实验。

### 主要发现

MC-GNNAS-Dock展现出一致优越的性能，在RMSD低于1埃(2埃)且通过PoseBuster有效性检查的复合标准下，比最佳单一求解器Uni-Mol Docking V2高出最多5.4%(3.4%)。

### 结论

MC-GNNAS-Dock通过综合多种评估标准和优化算法架构，显著提高了分子对接算法选择的性能，为药物发现提供了更可靠的预测工具。

### 翻译

分子对接是药物发现中用于预测配体-靶点相互作用的核心工具。尽管存在多种基于搜索和机器学习的方法，但没有单一的对接算法能在所有情况下持续占优，因为性能会随上下文变化。为应对这一挑战，已提出基于图神经网络的算法选择框架如GNNAS-Dock。本研究引入了一个增强系统MC-GNNAS-Dock，具有三项关键改进。首先，多标准评估将结合姿态准确性(RMSD)与PoseBusters的有效性检查相结合，提供更严格的评估。其次，通过包含残差连接的架构改进增强了预测鲁棒性。第三，整合了排名感知损失函数以强化排名学习。在包含约3200个来自PDBBind的蛋白质-配体复合物的精选数据集上进行了大量实验。MC-GNNAS-Dock表现出一致优越的性能，在RMSD低于1埃(2埃)且通过PoseBuster有效性检查的复合标准下，比最佳单一求解器(SBS) Uni-Mol Docking V2高出最多5.4%(3.4%)。


### 论文摘要

Molecular docking is a core tool in drug discovery for predicting ligand-target interactions. Despite the availability of diverse search-based and machine learning approaches, no single docking algorithm consistently dominates, as performance varies by context. To overcome this challenge, algorithm selection frameworks such as GNNAS-Dock, built on graph neural networks, have been proposed. This study introduces an enhanced system, MC-GNNAS-Dock, with three key advances. First, a multi-criteria evaluation integrates binding-pose accuracy (RMSD) with validity checks from PoseBusters, offering a more rigorous assessment. Second, architectural refinements by inclusion of residual connections strengthen predictive robustness. Third, rank-aware loss functions are incorporated to sharpen rank learning. Extensive experiments are performed on a curated dataset containing approximately 3200 protein-ligand complexes from PDBBind. MC-GNNAS-Dock demonstrates consistently superior performance, achieving up to 5.4% (3.4%) gains under composite criteria of RMSD below 1\AA{} (2\AA{}) with PoseBuster-validity compared to the single best solver (SBS) Uni-Mol Docking V2.

---

## 22. Ultra-Reliable Risk-Aggregated Sum Rate Maximization via Model-Aided Deep Learning

**论文链接:** [http://arxiv.org/abs/2509.26311v1](http://arxiv.org/abs/2509.26311v1)

**作者:** Hassaan Hashmi, Spyridon Pougkakiotis, Dionysis Kalogerias

**发布时间:** 2025-09-30

### GPT解析

### 总结

该研究针对多输入单输出(MISO)下行无线网络中的加权速率和最大化问题，特别关注用户速率可靠性，提出了一种基于条件风险价值(CVaR)的新方法，并设计了α鲁棒图神经网络(αRGNN)来优化性能。

### 背景

研究多输入单输出(MISO)下行无线网络中的加权速率和最大化问题，重点关注用户速率可靠性，并考虑信道衰落不确定性/风险带来的挑战。

### 目的

在信道衰落不确定性条件下最大化加权速率和，同时确保速率的超可靠性。

### 方法

引入风险聚合公式处理复杂的WSR最大化问题，利用CVaR作为函数强制执行速率可靠性；建立预编码问题与加权风险厌恶MSE问题之间的WMMSE类似等价性；设计定制的展开图神经网络策略函数近似(αRGNN)，训练以最大化来自不利信道实现的尾部速率。

### 主要发现

训练后的αRGNN完全消除了每个用户的深度速率衰落，显著且最优地减少了统计用户速率变异性，同时保持了足够的遍历性能。

### 结论

所提出的αRGNN方法能有效处理信道衰落不确定性，提高用户速率可靠性，优化网络性能。

### 翻译

我们考虑在多输入单输出(MISO)下行无线网络中最大化加权速率和的问题，重点关注用户速率可靠性。我们引入了一种复杂WSR最大化问题的风险聚合新公式，利用条件风险价值(CVaR)作为函数，强制执行在信道衰落不确定性/风险情况下的速率(超)可靠性。我们建立了所提出的预编码问题与加权风险厌恶MSE问题之间的WMMSE类似等价性，使我们能够设计定制的展开图神经网络(GNN)策略函数近似(PFA)，称为α鲁棒图神经网络(αRGNN)，训练以最大化来自不利无线信道实现(如深度衰落、衰减)导致的尾部(CVaR)速率。我们 empirically 证明训练后的αRGNN完全消除了每个用户的深度速率衰落，同时显著且最优地减少了统计用户速率变异性，并保持了足够的遍历性能。


### 论文摘要

We consider the problem of maximizing weighted sum rate in a multiple-input single-output (MISO) downlink wireless network with emphasis on user rate reliability. We introduce a novel risk-aggregated formulation of the complex WSR maximization problem, which utilizes the Conditional Value-at-Risk (CVaR) as a functional for enforcing rate (ultra)-reliability over channel fading uncertainty/risk. We establish a WMMSE-like equivalence between the proposed precoding problem and a weighted risk-averse MSE problem, enabling us to design a tailored unfolded graph neural network (GNN) policy function approximation (PFA), named {\alpha}-Robust Graph Neural Network ({\alpha}RGNN), trained to maximize lower-tail (CVaR) rates resulting from adverse wireless channel realizations (e.g., deep fading, attenuation). We empirically demonstrate that a trained {\alpha}RGNN fully eliminates per user deep rate fades, and substantially and optimally reduces statistical user rate variability while retaining adequate ergodic performance.

---

## 23. Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification

**论文链接:** [http://arxiv.org/abs/2509.26032v1](http://arxiv.org/abs/2509.26032v1)

**作者:** Xiaobao Wang, Ruoxiao Sun, Yujun Zhang, Bingdao Feng, Dongxiao He, Luzhi Wang, Di Jin

**发布时间:** 2025-09-30

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS  2025)

### GPT解析

### 总结

本文提出了一种名为DPSBA的干净标签后门框架，用于解决图神经网络中的后门攻击问题。DPSBA通过异常感知判别器引导的对抗性训练来学习分布内触发器，有效抑制了结构和语义异常，实现了高攻击成功率的同时显著提高隐蔽性。

### 背景

图神经网络在节点分类、链接预测和图分类等任务上表现出色，但容易受到后门攻击。图级别攻击面临在保持隐蔽性的同时操纵全局表示的挑战，而现有方法存在结构偏差和语义偏差导致的异常问题，使被污染的图容易被检测。

### 目的

开发一种能够有效抑制结构和语义异常的后门攻击方法，实现高攻击成功率的同时提高隐蔽性，在有效性和可检测性之间取得更好的平衡。

### 方法

提出DPSBA框架，通过异常感知判别器引导的对抗性训练来学习分布内触发器。这种方法针对现有图分类后门方法中的两种主要异常源：稀有子图触发器的结构偏差和标签翻转引起的语义偏差。

### 主要发现

DPSBA能够有效抑制结构和语义异常，在真实世界数据集上的实验表明，与最先进的基线相比，DPSBA在攻击有效性和隐蔽性之间实现了更好的平衡。

### 结论

DPSBA是一种改进的后门攻击方法，解决了现有图分类后门方法中的异常检测问题，实现了更优的攻击效果和隐蔽性，为研究图神经网络的安全性问题提供了新的思路。

### 翻译

图神经网络已在节点分类、链接预测和图分类等任务中展现出强大的性能，但仍容易受到后门攻击的影响，这些攻击在训练过程中植入难以察觉的触发器来控制预测结果。虽然节点级别攻击利用局部消息传递，但图级别攻击面临更困难的挑战，即在保持隐蔽性的同时操纵全局表示。我们确定了现有图分类后门方法中存在的两种主要异常源：稀有子图触发器的结构偏差和标签翻转引起的语义偏差，这些异常使得被污染的图容易被异常检测模型检测。为此，我们提出DPSBA，一种干净标签的后门框架，通过异常感知判别器引导的对抗性训练来学习分布内触发器。DPSBA有效抑制了结构和语义异常，在实现高攻击成功率的同时显著提高了隐蔽性。在真实世界数据集上的广泛实验验证了DPSBA与最先进的基线相比，在有效性和可检测性之间实现了更优的平衡。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated strong performance across tasks such as node classification, link prediction, and graph classification, but remain vulnerable to backdoor attacks that implant imperceptible triggers during training to control predictions. While node-level attacks exploit local message passing, graph-level attacks face the harder challenge of manipulating global representations while maintaining stealth. We identify two main sources of anomaly in existing graph classification backdoor methods: structural deviation from rare subgraph triggers and semantic deviation caused by label flipping, both of which make poisoned graphs easily detectable by anomaly detection models. To address this, we propose DPSBA, a clean-label backdoor framework that learns in-distribution triggers via adversarial training guided by anomaly-aware discriminators. DPSBA effectively suppresses both structural and semantic anomalies, achieving high attack success while significantly improving stealth. Extensive experiments on real-world datasets validate that DPSBA achieves a superior balance between effectiveness and detectability compared to state-of-the-art baselines.

---

## 24. Accelerated Discovery of High-\k{appa} Oxides with Physics-Based Factorized Machine Learning

**论文链接:** [http://arxiv.org/abs/2509.26022v1](http://arxiv.org/abs/2509.26022v1)

**作者:** Atsushi Takigawa, Shin Kiyohara, Yu Kumagai

**发布时间:** 2025-09-30

**备注:** 42 pages, 13 figures, 6 tables, submitted

### GPT解析

### 总结

研究团队提出了一种联合机器学习方法，通过结合等变图神经网络和预训练机器学习势，显著提高了对高κ材料介电张量离子贡献的预测准确性，并成功从大量候选材料中筛选出38种新型高κ氧化物。

### 背景

研究人员持续致力于探索结合高介电常数和宽带隙特性的下一代高κ材料，但基于机器学习的虚拟筛选面临挑战，主要由于预测介电张量离子贡献的准确性较低，而这部分对高κ材料的介电性能起主导作用。

### 目的

提高对介电张量离子贡献的预测准确性，从而更有效地筛选高κ材料。

### 方法

提出联合机器学习模型，使用等变图神经网络预测玻恩有效电荷，使用高度精确的预训练机器学习势预测声子特性，然后从这些量中解析计算离子介电张量。

### 主要发现

该方法显著提高了离子贡献的预测准确性，从超过8,000个候选材料中成功识别出38种新型高κ氧化物。

### 结论

联合机器学习方法可以有效提高高κ材料介电性能预测的准确性，特别是在离子贡献方面。

### 翻译

研究人员持续致力于探索结合高介电常数和宽带隙特性的下一代高κ材料。然而，基于机器学习的虚拟筛选仍然具有挑战性，主要由于预测介电张量的离子贡献时准确性较低，而这部分对高κ材料的介电性能起主导作用。作者在此提出了一种联合机器学习模型，该模型使用等变图神经网络预测玻恩有效电荷，使用高度精确的预训练机器学习势预测声子特性。离子介电张量然后从这些量中解析计算得出。这种方法显著提高了离子贡献的预测准确性。使用所提出的模型，我们成功地从超过8,000个候选材料中筛选出38种新型高κ氧化物。


### 论文摘要

Considerable effort continues to be devoted to the exploration of next-generation high-\k{appa} materials that combine a high dielectric constant with a wide band gap. However, machine learning (ML)-based virtual screening has remained challenging, primarily due to the low accuracy in predicting the ionic contribution to the dielectric tensor, which dominates the dielectric performance of high-\k{appa} materials. We here propose a joint ML model that predicts Born effective charges using an equivariant graph neural network, and phonon properties using a highly accurate pretrained ML potential. The ionic dielectric tensor is then computed analytically from these quantities. This approach significantly improves the accuracy of ionic contribution. Using the proposed model, we successfully identified 38 novel high-\k{appa} oxides from a screening pool of over 8,000 candidates.

---

## 25. HiFIRec: Towards High-Frequency yet Low-Intention Behaviors for Multi-Behavior Recommendation

**论文链接:** [http://arxiv.org/abs/2509.25755v1](http://arxiv.org/abs/2509.25755v1)

**作者:** Ruiqi Luo, Ran Jin, Zhenglong Li, Kaixi Hu, Xiaohui Tao, Lin Li

**发布时间:** 2025-09-30

### GPT解析

### 总结

HiFIRec是一种新的多行为推荐方法，通过差异化行为建模解决高频低意图行为带来的噪声问题，并在基准测试上取得了优于现有方法的性能。

### 背景

多行为推荐利用多种用户-物品交互解决数据稀疏性和冷启动问题，在医疗保健和电子商务等领域提供个性化服务。现有方法多使用图神经网络统一建模用户意图，但未能充分考虑不同行为间的异质性。

### 目的

提出HiFIRec方法，通过差异化行为建模纠正高频但低意图行为的影响，解决这些行为隐含的噪声信号和误导性频繁模式问题。

### 方法

HiFIRec通过分层抑制噪声信号，利用逐层邻域聚合提取邻域信息，并通过自适应跨层特征融合捕获用户意图；同时提出强度感知的非采样策略，动态调整负样本权重以纠正看似合理的频繁模式。

### 主要发现

高频低意图行为隐含的噪声信号和误导性频繁模式会阻碍用户意图的学习，需要特别处理。

### 结论

HiFIRec在两个基准测试上相对提高了HR@10指标4.21%-6.81%，证明了该方法在多行为推荐任务上的有效性。

### 翻译

多行为推荐利用多种类型的用户-物品交互来解决数据稀疏性和冷启动问题，在医疗保健和电子商务等领域提供个性化服务。大多数现有方法使用图神经网络以统一方式建模用户意图，这未能充分考虑不同行为之间的异质性。特别是，高频但低意图的行为可能隐含噪声信号，而看似合理但具有误导性的频繁模式会阻碍用户意图的学习。为此，本文提出了一种新的多行为推荐方法HiFIRec，通过差异化行为建模来纠正高频但低意图行为的影响。为了修正噪声信号，我们通过逐层邻域聚合提取邻域信息，并在各层中分层抑制噪声，进一步通过自适应跨层特征融合捕获用户意图。为了纠正看似合理的频繁模式，我们提出了一种强度感知的非采样策略，动态调整负样本的权重。在两个基准测试上的大量实验表明，HiFIRec相对HR@10指标比几种最先进的方法提高了4.21%-6.81%。


### 论文摘要

Multi-behavior recommendation leverages multiple types of user-item interactions to address data sparsity and cold-start issues, providing personalized services in domains such as healthcare and e-commerce. Most existing methods utilize graph neural networks to model user intention in a unified manner, which inadequately considers the heterogeneity across different behaviors. Especially, high-frequency yet low-intention behaviors may implicitly contain noisy signals, and frequent patterns that are plausible while misleading, thereby hindering the learning of user intentions. To this end, this paper proposes a novel multi-behavior recommendation method, HiFIRec, that corrects the effect of high-frequency yet low-intention behaviors by differential behavior modeling. To revise the noisy signals, we hierarchically suppress it across layers by extracting neighborhood information through layer-wise neighborhood aggregation and further capturing user intentions through adaptive cross-layer feature fusion. To correct plausible frequent patterns, we propose an intensity-aware non-sampling strategy that dynamically adjusts the weights of negative samples. Extensive experiments on two benchmarks show that HiFIRec relatively improves HR@10 by 4.21%-6.81% over several state-of-the-art methods.

---

## 26. Cooperative Autonomous Driving in Diverse Behavioral Traffic: A Heterogeneous Graph Reinforcement Learning Approach

**论文链接:** [http://arxiv.org/abs/2509.25751v1](http://arxiv.org/abs/2509.25751v1)

**作者:** Qi Liu, Xueyuan Li, Zirui Li, Juhui Gim

**发布时间:** 2025-09-30

**备注:** 7 pages, 5 figures and 4 tables

### GPT解析

### 总结

本文提出了一种结合专家系统的异质图强化学习框架，用于提高自动驾驶车辆在异质交通环境中的决策性能。

### 背景

异质交通环境和多样化的驾驶风格对自动驾驶车辆构成重大挑战，这些挑战源于其固有的复杂性和动态交互。

### 目的

提出一种异质图强化学习框架结合专家系统，以改善自动驾驶车辆的决策性能。

### 方法

引入异质图表示捕捉车辆间复杂交互；提出带有专家模型的异质图神经网络编码多样化车辆特征；利用双深度Q学习算法训练决策模型。

### 主要发现

在四向交叉路口案例研究中，涉及人类车辆的各种驾驶风格，所提方法在安全性、效率、稳定性和收敛率方面优于基线方法，同时保持良好实时性能。

### 结论

该框架能有效应对异质交通环境中的复杂挑战，提高自动驾驶车辆的决策能力。

### 翻译

在具有多样化驾驶风格的异质交通环境中导航，由于其固有的复杂性和动态交互，对自动驾驶车辆（AVs）构成了重大挑战。本文通过提出一种结合专家系统的异质图强化学习（GRL）框架来解决这一挑战，以提高自动驾驶车辆的决策性能。首先，引入异质图表示来捕捉车辆之间的复杂交互。然后，提出带有专家模型的异质图神经网络（HGNN-EM）来有效编码多样化的车辆特征，并产生由领域知识驱动的驾驶指令。此外，利用双深度Q学习（DDQN）算法来训练决策模型。在一个典型的四向交叉路口案例研究中，涉及人类车辆（HVs）的各种驾驶风格，证明了所提出的方法在安全性、效率和收敛率方面优于几种基线方法，同时保持良好的实时性能。


### 论文摘要

Navigating heterogeneous traffic environments with diverse driving styles poses a significant challenge for autonomous vehicles (AVs) due to their inherent complexity and dynamic interactions. This paper addresses this challenge by proposing a heterogeneous graph reinforcement learning (GRL) framework enhanced with an expert system to improve AV decision-making performance. Initially, a heterogeneous graph representation is introduced to capture the intricate interactions among vehicles. Then, a heterogeneous graph neural network with an expert model (HGNN-EM) is proposed to effectively encode diverse vehicle features and produce driving instructions informed by domain-specific knowledge. Moreover, the double deep Q-learning (DDQN) algorithm is utilized to train the decision-making model. A case study on a typical four-way intersection, involving various driving styles of human vehicles (HVs), demonstrates that the proposed method has superior performance over several baselines regarding safety, efficiency, stability, and convergence rate, all while maintaining favorable real-time performance.

---

## 27. Adaptive Graph Coarsening for Efficient GNN Training

**论文链接:** [http://arxiv.org/abs/2509.25706v1](http://arxiv.org/abs/2509.25706v1)

**作者:** Rostyslav Olshevskyi, Madeline Navarro, Santiago Segarra

**发布时间:** 2025-09-30

### GPT解析

### 总结

提出了一种自适应图粗化方法，可以在训练过程中同时学习图神经网络参数并通过K均值聚类合并节点，适用于大规模图处理和异质数据场景。

### 背景

随着现实世界图规模扩大，直接处理变得极具挑战性甚至不可行。针对大规模数据定制算法可能牺牲性能，因此考虑图减少来降低训练数据量。

### 目的

开发一种方法，在训练过程中同时训练图神经网络并通过基于节点嵌入的K均值聚类对节点分区来粗化图。

### 方法

提出一种方法，在训练过程中同时训练GNN和粗化其图，通过基于节点嵌入的K均值聚类对节点进行分区。与过去的图粗化工作不同，这种方法允许在训练过程中合并节点。

### 主要发现

该方法不需要将粗化作为预处理步骤，节点聚类可以适应学习任务而不仅依赖于图连通性和特征，适用于异质数据等具有挑战性的场景。

### 结论

在同质和异质节点分类数据集上验证了该方法。可视化节点嵌入及其对应聚类关系，表明粗化图在训练过程中能适应学习任务。

### 翻译

我们提出了一种自适应图粗化方法，在训练过程中通过K均值聚类联合学习图神经网络参数和合并节点。随着现实世界图的规模不断扩大，直接处理它们变得越来越具有挑战性，有时甚至不可行。针对大规模数据定制算法可能会牺牲性能，因此我们考虑图减少来减少训练中使用的数据量。特别是，我们提出了一种方法，通过基于节点嵌入的K均值聚类对节点进行分区，同时训练GNN和粗化其图。与过去的图粗化工作不同，我们的方法允许我们在训练过程中合并节点。这不仅避免了将粗化作为预处理步骤，而且我们的节点聚类可以适应学习任务，而不仅仅依赖于图连通性和特征。因此，我们的方法适用于其他方法具有挑战性的场景，如异质数据。我们在同质和异质节点分类数据集上验证了我们的方法。我们进一步可视化了节点嵌入及其对应聚类之间的关系，以说明我们的粗化图在训练过程中适应学习任务。


### 论文摘要

We propose an adaptive graph coarsening method to jointly learn graph neural network (GNN) parameters and merge nodes via K-means clustering during training. As real-world graphs grow larger, processing them directly becomes increasingly challenging and sometimes infeasible. Tailoring algorithms to large-scale data may sacrifice performance, so we instead consider graph reduction to decrease the amount of data used during training. In particular, we propose a method to simultaneously train a GNN and coarsen its graph by partitioning nodes via K-means clustering based on their embeddings. Unlike past graph coarsening works, our approach allows us to merge nodes during training. Not only does this preclude coarsening as a preprocessing step, but our node clusters can adapt to the learning task instead of relying solely on graph connectivity and features. Thus, our method is amenable to scenarios that are challenging for other methods, such as heterophilic data. We validate our approach on both homophilic and heterophilic node classification datasets. We further visualize relationships between node embeddings and their corresponding clusters to illustrate that our coarsened graph adapts to the learning task during training.

---

## 28. AttentionViG: Cross-Attention-Based Dynamic Neighbor Aggregation in Vision GNNs

**论文链接:** [http://arxiv.org/abs/2509.25570v1](http://arxiv.org/abs/2509.25570v1)

**作者:** Hakan Emre Gedik, Andrew Martin, Mustafa Munir, Oguzhan Baser, Radu Marculescu, Sandeep P. Chinchali, Alan C. Bovik

**发布时间:** 2025-09-29

**备注:** WACV submission. 13 pages, including the main text (8 pages),  references, and supplementary material

### GPT解析

### 总结

本研究提出了一种基于交叉注意力的节点-邻居特征聚合方法，并开发了AttentionViG架构，在图像识别任务中取得了最先进性能，同时保持了计算效率。

### 背景

Vision Graph Neural Networks (ViGs)在图像识别任务中展现出优于CNNs和ViTs的性能。ViG框架的重要组成部分是节点-邻居特征聚合方法，尽管已有多种图卷积方法（如Max-Relative、EdgeConv、GIN和GraphSAGE），但仍需要一个能够有效捕获复杂节点-邻居关系且不需要架构特定调整的通用聚合方法。

### 目的

开发一种能够有效捕获复杂节点-邻居关系且不需要架构特定调整的通用聚合方法，并构建一个基于此方法的视觉图神经网络架构。

### 方法

提出了一种基于交叉注意力的聚合方法，其中查询投影来自节点，而键投影来自其邻居；并引入了AttentionViG架构，使用这种交叉注意力聚合方案进行非局部消息传递。

### 主要发现

在ImageNet-1K基准上，AttentionViG取得了最先进的图像识别性能；在下游任务（包括MSCOCO 2017上的目标检测和实例分割，以及ADE20K上的语义分割）上也表现出良好的可迁移性；所提出的方法在保持计算效率的同时，实现了强大的性能。

### 结论

基于交叉注意力的聚合方法能够有效捕获复杂的节点-邻居关系，AttentionViG架构在图像识别和下游任务中实现了最先进的性能，同时保持了计算效率，为视觉图神经网络的发展提供了新的方向。

### 翻译

视觉图神经网络(ViGs)在图像识别任务中已经展现出与卷积神经网络(CNNs)和视觉Transformer(ViTs)相比有前景的性能。ViG框架的一个基本部分是节点-邻居特征聚合方法。尽管已经探索了各种图卷积方法，如Max-Relative、EdgeConv、GIN和GraphSAGE，但仍需要一个能够有效捕获复杂节点-邻居关系且不需要架构特定调整的通用聚合方法。为了解决这一差距，我们提出了一种基于交叉注意力的聚合方法，其中查询投影来自节点，而键投影来自其邻居。此外，我们引入了一种名为AttentionViG的新架构，它使用提出的交叉注意力聚合方案进行非局部消息传递。我们在ImageNet-1K基准上评估了AttentionViG的图像识别性能，并取得了最先进的性能。此外，我们还评估了其在下游任务上的可迁移性，包括MSCOCO 2017上的目标检测和实例分割，以及ADE20K上的语义分割。我们的结果表明，所提出的方法不仅实现了强大的性能，而且保持了效率，与先前的视觉GNN架构相比，在相当的计算量(FLOPs)下提供了具有竞争力的准确性。


### 论文摘要

Vision Graph Neural Networks (ViGs) have demonstrated promising performance in image recognition tasks against Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). An essential part of the ViG framework is the node-neighbor feature aggregation method. Although various graph convolution methods, such as Max-Relative, EdgeConv, GIN, and GraphSAGE, have been explored, a versatile aggregation method that effectively captures complex node-neighbor relationships without requiring architecture-specific refinements is needed. To address this gap, we propose a cross-attention-based aggregation method in which the query projections come from the node, while the key projections come from its neighbors. Additionally, we introduce a novel architecture called AttentionViG that uses the proposed cross-attention aggregation scheme to conduct non-local message passing. We evaluated the image recognition performance of AttentionViG on the ImageNet-1K benchmark, where it achieved SOTA performance. Additionally, we assessed its transferability to downstream tasks, including object detection and instance segmentation on MS COCO 2017, as well as semantic segmentation on ADE20K. Our results demonstrate that the proposed method not only achieves strong performance, but also maintains efficiency, delivering competitive accuracy with comparable FLOPs to prior vision GNN architectures.

---

## 29. Evaluating Foundation Models with Pathological Concept Learning for Kidney Cancer

**论文链接:** [http://arxiv.org/abs/2509.25552v1](http://arxiv.org/abs/2509.25552v1)

**作者:** Shangqi Gao, Sihan Wang, Yibo Gao, Boming Wang, Xiahai Zhuang, Anne Warren, Grant Stewart, James Jones, Mireia Crispin-Ortuzar

**发布时间:** 2025-09-29

**备注:** Best Paper Award at MICCAI AMAI 2025

### GPT解析

### 总结

该研究开发了一种针对肾癌的病理概念学习方法，评估了基础模型的翻译能力。通过TNM分期指南和病理报告构建病理概念，利用基础模型从全幻灯片图像中提取深度特征，构建病理图捕捉空间相关性，并训练图神经网络识别这些概念。该方法在肾癌生存分析中表现出色，具有良好可解释性和公平性，能准确识别低风险和高风险患者。

### 背景

基础模型在医学影像分析中展现出巨大潜力，但其在病理概念学习和临床应用中的翻译能力仍需评估。特别是在肾癌诊断和预后分析方面，需要更有效的方法整合病理信息和临床数据。

### 目的

评估基础模型在肾癌病理概念学习中的翻译能力，开发一种基于病理图和图神经网络的方法，用于肾癌生存分析和风险分层。

### 方法

1. 利用TNM分期指南和病理报告构建全面的肾癌病理概念；2. 使用基础模型从全幻灯片图像中提取深度特征；3. 构建病理图来捕捉空间相关性；4. 训练图神经网络识别这些病理概念；5. 在肾癌生存分析中验证方法的有效性。

### 主要发现

1. 基于基础模型的特征提取能有效捕捉肾癌病理特征；2. 病理图和图神经网络能准确识别肾癌相关病理概念；3. 该方法在肾癌生存分析中表现出色；4. 方法具有良好的可解释性和公平性，能准确区分低风险和高风险患者。

### 结论

该研究提出的方法能有效评估基础模型在肾癌病理概念学习中的翻译能力，为肾癌诊断和预后分析提供了新的工具。方法的可解释性和公平性特点使其具有临床应用潜力，源代码已公开以促进进一步研究和应用。

### 翻译

为了评估基础模型的翻译能力，我们开发了一种专注于肾癌的病理概念学习方法。通过利用TNM分期指南和病理报告，我们构建了肾癌的全面病理概念。然后，我们使用基础模型从全幻灯片图像中提取深度特征，构建病理图以捕捉空间相关性，并训练图神经网络来识别这些概念。最后，我们在肾癌生存分析中证明了这种方法的有效性，强调了其在识别低风险和高风险患者方面的可解释性和公平性。源代码已通过https://github.com/shangqigao/RadioPath发布。


### 论文摘要

To evaluate the translational capabilities of foundation models, we develop a pathological concept learning approach focused on kidney cancer. By leveraging TNM staging guidelines and pathology reports, we build comprehensive pathological concepts for kidney cancer. Then, we extract deep features from whole slide images using foundation models, construct pathological graphs to capture spatial correlations, and trained graph neural networks to identify these concepts. Finally, we demonstrate the effectiveness of this approach in kidney cancer survival analysis, highlighting its explainability and fairness in identifying low- and high-risk patients. The source code has been released by https://github.com/shangqigao/RadioPath.

---

## 30. AGNOMIN -- Architecture Agnostic Multi-Label Function Name Prediction

**论文链接:** [http://arxiv.org/abs/2509.25514v1](http://arxiv.org/abs/2509.25514v1)

**作者:** Yonatan Gizachew Achamyeleh, Tongtao Zhang, Joshua Hyunki Kim, Gabriel Garcia, Shih-Yuan Yu, Anton Kocheturov, Mohammad Abdullah Al Faruque

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为AGNOMIN的新型架构无关方法，用于在剥离二进制文件中进行多标签函数名预测。该方法结合了控制流图、函数调用图和动态学习的PCode特征，构建增强层次图，并使用层次图神经网络生成跨架构一致的函数表示。AGNOMIN还采用改进的基于Renée解码器的预测方法，在实际应用中表现出优异性能。

### 背景

函数名预测对于理解软件逆向工程中的剥离二进制文件至关重要，是进行后续漏洞分析和修补的关键步骤。然而，现有方法通常面临架构特定限制、数据稀缺和多样化命名约定等挑战。

### 目的

开发一种架构无关的方法，能够在剥离二进制文件中进行多标签函数名预测，以支持可扩展的安全评估。

### 方法

AGNOMIN构建了增强层次图（FEHGs），结合控制流图、函数调用图和动态学习的PCode特征。使用层次图神经网络处理这些结构，生成跨架构一致的函数表示。函数名预测采用基于Renée解码器的改进方法，加入基于注意力的头部层和算法改进。

### 主要发现

在9,000个跨三种架构的ELF可执行二进制文件上评估，AGNOMIN在测试数据集上的精确度提高高达27.17%，召回率提高55.86%。该方法能很好地泛化到未见过的架构，比最接近的基线方法高出5.89%的召回率。实际应用中，AGNOMIN成功帮助逆向工程师分析和修补跨不同架构的易受攻击二进制文件。

### 结论

AGNOMIN是一种有效的架构无关方法，能够在剥离二进制文件中进行多标签函数名预测，具有优异的性能和泛化能力，对安全评估具有实用价值。

### 翻译

函数名预测对于理解软件逆向工程中的剥离二进制文件至关重要，这是进行后续漏洞分析和修补的关键步骤。然而，现有方法通常面临架构特定限制、数据稀缺和多样化命名约定等挑战。我们提出了AGNOMIN，一种用于剥离二进制文件中多标签函数名预测的新型架构无关方法。AGNOMIN构建了增强层次图（FEHGs），结合了控制流图、函数调用图和动态学习的PCode特征。层次图神经网络处理这种增强结构，以生成跨架构一致的函数表示，这对可扩展的安全评估至关重要。对于函数名预测，AGNOMIN采用了一种基于Renée解码器的改进方法，增强了基于注意力的头部层和算法改进。我们在包含9,000个跨三种架构的ELF可执行二进制文件的全面数据集上评估了AGNOMIN，证明了其相比最先进方法的优越性能，在测试数据集上的精确度提高了高达27.17%，召回率提高了55.86%。此外，AGNOMIN能够很好地泛化到未见过的架构，比最接近的基线方法高出5.89%的召回率。AGNOMIN的实际效用已在安全黑客马拉松中得到验证，成功帮助逆向工程师分析和修补跨不同架构的易受攻击二进制文件。


### 论文摘要

Function name prediction is crucial for understanding stripped binaries in software reverse engineering, a key step for \textbf{enabling subsequent vulnerability analysis and patching}. However, existing approaches often struggle with architecture-specific limitations, data scarcity, and diverse naming conventions. We present AGNOMIN, a novel architecture-agnostic approach for multi-label function name prediction in stripped binaries. AGNOMIN builds Feature-Enriched Hierarchical Graphs (FEHGs), combining Control Flow Graphs, Function Call Graphs, and dynamically learned \texttt{PCode} features. A hierarchical graph neural network processes this enriched structure to generate consistent function representations across architectures, vital for \textbf{scalable security assessments}. For function name prediction, AGNOMIN employs a Ren\'ee-inspired decoder, enhanced with an attention-based head layer and algorithmic improvements.   We evaluate AGNOMIN on a comprehensive dataset of 9,000 ELF executable binaries across three architectures, demonstrating its superior performance compared to state-of-the-art approaches, with improvements of up to 27.17\% in precision and 55.86\% in recall across the testing dataset. Moreover, AGNOMIN generalizes well to unseen architectures, achieving 5.89\% higher recall than the closest baseline. AGNOMIN's practical utility has been validated through security hackathons, where it successfully aided reverse engineers in analyzing and patching vulnerable binaries across different architectures.

---

## 31. Scalable Boltzmann Generators for equilibrium sampling of large-scale materials

**论文链接:** [http://arxiv.org/abs/2509.25486v1](http://arxiv.org/abs/2509.25486v1)

**作者:** Maximilian Schebek, Jutta Rogal

**发布时间:** 2025-09-29

### GPT解析

### 总结

本研究提出了一种新的玻尔兹曼生成器架构，通过结合增强耦合流和图神经网络，解决了多体系统平衡分布采样中的可扩展性问题，实现了高效的大系统材料采样。

### 背景

玻尔兹曼生成器等生成模型因能够一次性产生无偏且无相关样本而受到广泛关注，但在扩展到大型系统时仍面临重大挑战。特别是在材料科学应用中，需要处理大规模原子系统。

### 目的

开发一种可扩展的玻尔兹曼生成器架构，专注于材料科学应用，能够高效采样大规模材料系统，克服现有方法的计算资源限制和采样效率问题。

### 方法

利用增强耦合流结合图神经网络，基于局部环境信息构建生成过程，同时保持基于能量的训练和快速推理能力。这种方法使生成过程更加高效，并且架构可转移到大系统尺寸。

### 主要发现

与先前架构相比，新模型训练速度显著提高，计算资源需求大幅减少，采样效率更优。架构成功应用于远超一千个原子的系统，包括Lennard-Jones晶体、mW水的冰相和硅的相图，能够产生高度精确的平衡系综和自由能数据。

### 结论

所提出的玻尔兹曼生成器架构有效解决了多体系统平衡分布采样的可扩展性问题，能够处理前所未有的大规模材料系统，为材料科学研究提供了强大的工具，特别是在需要精确平衡系综和自由能计算的应用中。

### 翻译

使用生成模型对多体系统的平衡分布进行采样，正如玻尔兹曼生成器首次展示的那样，因其能够一次性产生无偏且无相关的样本而引起了广泛关注。尽管这些模型在自然科学领域展现出巨大的潜力和令人印象深刻的结果，但将其扩展到大型系统仍然是一个重大挑战。在这项工作中，我们引入了一种玻尔兹曼生成器架构，专门解决了材料科学应用中的可扩展性瓶颈问题。我们结合增强耦合流和图神经网络，使生成过程基于局部环境信息，同时允许基于能量的训练和快速推理。与先前的架构相比，我们的模型训练速度显著提高，计算资源需求大幅减少，并且实现了更高的采样效率。关键的是，该架构可以转移到更大的系统尺寸，从而能够高效采样前所未有的大尺寸模拟单元的材料。我们通过将该方法应用于多种材料系统（包括Lennard-Jones晶体、mW水的冰相和硅的相图，系统规模远超一千个原子）来证明我们方法的潜力。训练好的玻尔兹曼生成器能够为各种晶体结构产生高度精确的平衡系综，以及各种系统尺寸下的亥姆霍兹自由能和吉布斯自由能，能够达到有限尺寸效应变得可忽略的规模。


### 论文摘要

The use of generative models to sample equilibrium distributions of many-body systems, as first demonstrated by Boltzmann Generators, has attracted substantial interest due to their ability to produce unbiased and uncorrelated samples in `one shot'. Despite their promise and impressive results across the natural sciences, scaling these models to large systems remains a major challenge. In this work, we introduce a Boltzmann Generator architecture that addresses this scalability bottleneck with a focus on applications in materials science. We leverage augmented coupling flows in combination with graph neural networks to base the generation process on local environmental information, while allowing for energy-based training and fast inference. Compared to previous architectures, our model trains significantly faster, requires far less computational resources, and achieves superior sampling efficiencies. Crucially, the architecture is transferable to larger system sizes, which allows for the efficient sampling of materials with simulation cells of unprecedented size. We demonstrate the potential of our approach by applying it to several materials systems, including Lennard-Jones crystals, ice phases of mW water, and the phase diagram of silicon, for system sizes well above one thousand atoms. The trained Boltzmann Generators produce highly accurate equilibrium ensembles for various crystal structures, as well as Helmholtz and Gibbs free energies across a range of system sizes, able to reach scales where finite-size effects become negligible.

---

## 32. GESA: Graph-Enhanced Semantic Allocation for Generalized, Fair, and Explainable Candidate-Role Matching

**论文链接:** [http://arxiv.org/abs/2509.25435v1](http://arxiv.org/abs/2509.25435v1)

**作者:** Rishi Ashish Shah, Shivaay Dhondiyal, Kartik Sharma, Sukriti Talwar, Saksham Jain, Sparsh Jain

**发布时间:** 2025-09-29

### GPT解析

### 总结

GESA是一个全面的框架，解决了候选人角色分配中的准确性、公平性和可解释性问题，在大型基准测试中表现出色。

### 背景

准确的、公平的、可解释的候选人角色分配是一个跨领域的基本挑战，包括企业招聘、学术录取、奖学金授予和志愿者安置系统。

### 目的

解决当前最先进方法在语义灵活性、人口统计学偏见、决策过程透明度和动态政策约束下可扩展性方面的问题。

### 方法

通过整合领域自适应的transformer嵌入、异构自监督图神经网络、对抗性去偏见机制、多目标遗传优化和可解释AI组件来构建GESA框架。

### 主要发现

在包含20,000个候选人档案和3,000个角色规范的基准测试中，GESA实现了94.5%的前3名分配准确率，37%的多样性表示改进，0.98的公平性分数，以及亚秒级的端到端延迟。

### 结论

GESA具有混合推荐能力和玻璃盒可解释性，适合在工业、学术界和非营利部门等多样化的国际环境中部署。

### 翻译

准确的、公平的、可解释的候选人角色分配代表了一个跨多个领域的基本挑战，包括企业招聘、学术录取、奖学金授予和志愿者安置系统。当前最先进的方法在语义灵活性、持续的人口统计学偏见、决策过程不透明以及在动态政策约束下可扩展性差等方面存在局限性。我们提出了GESA（Graph-Enhanced Semantic Allocation），一个全面的框架，通过整合领域自适应的transformer嵌入、异构自监督图神经网络、对抗性去偏见机制、多目标遗传优化和可解释AI组件来解决这些限制。我们在包含20,000个候选人档案和3,000个角色规范的大规模国际基准测试中的实验评估表明，GESA具有卓越的性能，前3名分配准确率达到94.5%，多样性表示提高了37%，人口统计学类别间的公平性得分为0.98，端到端延迟不到一秒。此外，GESA还具备混合推荐能力和玻璃盒可解释性，使其适合在工业、学术界和非营利部门等多样化的国际环境中部署。


### 论文摘要

Accurate, fair, and explainable allocation of candidates to roles represents a fundamental challenge across multiple domains including corporate hiring, academic admissions, fellowship awards, and volunteer placement systems. Current state-of-the-art approaches suffer from semantic inflexibility, persistent demographic bias, opacity in decision-making processes, and poor scalability under dynamic policy constraints. We present GESA (Graph-Enhanced Semantic Allocation), a comprehensive framework that addresses these limitations through the integration of domain-adaptive transformer embeddings, heterogeneous self-supervised graph neural networks, adversarial debiasing mechanisms, multi-objective genetic optimization, and explainable AI components. Our experimental evaluation on large-scale international benchmarks comprising 20,000 candidate profiles and 3,000 role specifications demonstrates superior performance with 94.5% top-3 allocation accuracy, 37% improvement in diversity representation, 0.98 fairness score across demographic categories, and sub-second end-to-end latency. Additionally, GESA incorporates hybrid recommendation capabilities and glass-box explainability, making it suitable for deployment across diverse international contexts in industry, academia, and non-profit sectors.

---

## 33. Leveraging Vulnerabilities in Temporal Graph Neural Networks via Strategic High-Impact Assaults

**论文链接:** [http://arxiv.org/abs/2509.25418v1](http://arxiv.org/abs/2509.25418v1)

**作者:** Dong Hyun Jeon, Lijing Zhu, Haifang Li, Pengze Li, Jingna Feng, Tiehang Duan, Houbing Herbert Song, Cui Tao, Shuteng Niu

**发布时间:** 2025-09-29

**DOI:** 10.1145/3746252.3761282

### GPT解析

### 总结

本文提出了一种针对时间图神经网络的新型攻击方法HIA，通过识别关键节点并采用混合扰动策略，有效降低了TGNN的准确性，揭示了当前模型的脆弱性。

### 背景

时间图神经网络已成为分析社交网络、通信系统和金融网络等关键应用中动态图的重要工具，但其对抗攻击的鲁棒性仍是一个重大挑战。

### 目的

开发一种能够战略性地针对最具影响力的节点和边的新型攻击方法，以最大化对TGNN的影响并暴露其脆弱性。

### 方法

提出高影响攻击(HIA)框架，利用数据驱动的代理模型识别结构上和动态上重要的节点，采用混合扰动策略结合战略边注入和有针对性边删除，同时最小化扰动数量以提高隐蔽性。

### 主要发现

HIA在五个真实世界数据集和四种TGNN架构上显著降低了链接预测任务的准确性，MRR最高下降35.55%，显著优于现有攻击方法。

### 结论

当前STDG模型存在基本脆弱性，迫切需要开发考虑结构和时间动态的鲁棒防御方法。

### 翻译

时间图神经网络已成为分析社交网络、通信系统和金融网络等关键应用中动态图不可或缺的工具。然而，TGNNs对抗攻击的鲁棒性，特别是利用时间维度进行的复杂攻击，仍然是一个重大挑战。现有的针对时空动态图的攻击方法通常依赖于简单、易检测的扰动(如随机边的添加/删除)，未能战略性地针对最具影响力的节点和边以实现最大影响。我们引入了高影响攻击(HIA)，这是一种专门设计用于克服这些限制并暴露TGNNs关键漏洞的新型受限黑盒攻击框架。HIA利用数据驱动的代理模型来识别结构上重要的节点(对网络连接性至关重要)和动态上重要的节点(对图的时序演变关键)。然后采用混合扰动策略，结合战略边注入(创建误导性连接)和有针对性的边删除(破坏关键路径)，最大化TGNN性能下降。重要的是，HIA最小化了扰动数量以提高隐蔽性，使其更难被检测。在五个真实世界数据集和四种代表性TGNN架构(TGN、JODIE、DySAT和TGAT)上的全面实验表明，HIA显著降低了TGNN在链接预测任务上的准确性，平均倒数排名(MRR)最高下降35.55%——比最先进的基线有显著改进。这些结果突显了当前STDG模型的基本脆弱性，并强调了需要考虑结构和时间动态的鲁棒防御的迫切性。


### 论文摘要

Temporal Graph Neural Networks (TGNNs) have become indispensable for analyzing dynamic graphs in critical applications such as social networks, communication systems, and financial networks. However, the robustness of TGNNs against adversarial attacks, particularly sophisticated attacks that exploit the temporal dimension, remains a significant challenge. Existing attack methods for Spatio-Temporal Dynamic Graphs (STDGs) often rely on simplistic, easily detectable perturbations (e.g., random edge additions/deletions) and fail to strategically target the most influential nodes and edges for maximum impact. We introduce the High Impact Attack (HIA), a novel restricted black-box attack framework specifically designed to overcome these limitations and expose critical vulnerabilities in TGNNs. HIA leverages a data-driven surrogate model to identify structurally important nodes (central to network connectivity) and dynamically important nodes (critical for the graph's temporal evolution). It then employs a hybrid perturbation strategy, combining strategic edge injection (to create misleading connections) and targeted edge deletion (to disrupt essential pathways), maximizing TGNN performance degradation. Importantly, HIA minimizes the number of perturbations to enhance stealth, making it more challenging to detect. Comprehensive experiments on five real-world datasets and four representative TGNN architectures (TGN, JODIE, DySAT, and TGAT) demonstrate that HIA significantly reduces TGNN accuracy on the link prediction task, achieving up to a 35.55% decrease in Mean Reciprocal Rank (MRR) - a substantial improvement over state-of-the-art baselines. These results highlight fundamental vulnerabilities in current STDG models and underscore the urgent need for robust defenses that account for both structural and temporal dynamics.

---

## 34. A Graph-based Hybrid Beamforming Framework for MIMO Cell-Free ISAC Networks

**论文链接:** [http://arxiv.org/abs/2509.25385v1](http://arxiv.org/abs/2509.25385v1)

**作者:** Yanan Du, Sai Xu, Jagmohan Chauhan

**发布时间:** 2025-09-29

### GPT解析

### 总结

这篇论文开发了一种基于图的多输入多输出无小区集成通信与感知网络的混合波束成形框架。

### 背景

多输入多输出无小区集成通信与感知网络需要同时实现通信和感知功能，并保持发射器与雷达接收器之间的物理分离。

### 目的

在功率约束下联合提高通信和感知性能。

### 方法

构建新型MIMO无小区ISAC网络模型，将多目标优化问题重新表述为单目标优化问题，开发基于多个图神经网络的方法实现混合波束成形，并设计基于FPGA的加速器减少推理延迟。

### 主要发现

数值模拟验证了所提出优化方法的通信和感知能力，实验评估证明了基于FPGA加速在GNN推理中的显著性能提升。

### 结论

所提出的框架能够有效实现通信与感知的协同工作，并通过FPGA加速器提高了推理效率。

### 翻译

这篇论文为多输入多输出无小区集成通信与感知网络开发了一种基于图的混合波束成形框架。具体来说，我们构建了一种新型的MIMO无小区ISAC网络模型。在该模型中，多个双功能基站发射器采用分布式混合波束成形，同时实现通信和感知，同时保持发射器与雷达接收器之间的物理分离。基于此模型，我们在功率约束下制定了一个多目标优化问题，以联合提高通信和感知性能。为解决这一问题，首先将问题重新表述为单目标优化问题。然后，开发了一种由多个图神经网络组成的方法，以实现具有完美或不完美信道状态信息的混合波束成形。一旦训练完成，神经网络模型可以分布在各个基站上，实现快速高效的推理。为进一步减少推理延迟，开发了一种定制化的基于现场可编程门阵列的加速器。数值模拟验证了所提出优化方法的通信和感知能力，而实验评估证明了GNN推理中基于FPGA加速的显著性能提升。


### 论文摘要

This paper develops a graph-based hybrid beamforming framework for multiple-input multiple-output (MIMO) cell-free integrated sensing and communication (ISAC) networks. Specifically, we construct a novel MIMO cell-free ISAC network model. In this model, multiple dual-function base station (BS) transmitters employ distributed hybrid beamforming to enable simultaneous communication and sensing, while maintaining physical separation between the transmitters and the radar receiver. Building on this model, we formulate a multi-objective optimization problem under a power constraint to jointly improve communication and sensing performance. To solve it, the problem is first reformulated as a single-objective optimization problem. Then, a graph-based method composed of multiple graph neural networks (GNNs) is developed to realize hybrid beamforming with either perfect or imperfect channel state information. Once trained, the neural network model can be deployed distributively across BSs, enabling fast and efficient inference. To further reduce inference latency, a custom field-programmable gate array (FPGA)-based accelerator is developed. Numerical simulations validate the communication and sensing capabilities of the proposed optimization approach, while experimental evaluations demonstrate remarkable performance gains of FPGA-based acceleration in GNN inference.

---

## 35. Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids

**论文链接:** [http://arxiv.org/abs/2509.25158v1](http://arxiv.org/abs/2509.25158v1)

**作者:** Ehimare Okoyomon, Arbel Yaniv, Christoph Goebel

**发布时间:** 2025-09-29

### GPT解析

### 总结

这篇论文研究了如何通过物理信息策略提高图神经网络在配电电网电压预测中的泛化能力。

### 背景

电压预测对维持电力系统稳定性至关重要但具有挑战性。基于机器学习的方法(尤其是图神经网络)能显著提高速度，但在有限或不完整数据训练时泛化能力较差。

### 目的

系统研究归纳偏置在提高模型可靠学习功率流能力方面的作用，评估三种物理信息策略的效果。

### 方法

评估三种物理信息策略：(1)功率流约束的损失函数，(2)复值神经网络，(3)基于残差的任务重构。使用包含多种低压和中压电网配置的ENGAGE数据集进行对照实验。

### 主要发现

研究提供了关于哪些模型假设能最有效地引导现代配电网络中可靠高效的电压预测的实用见解。

### 结论

通过物理信息策略可以提高图神经网络在电压预测中的泛化能力和可靠性。

### 翻译

配电电网中的电压预测是维持电力系统稳定性的关键但困难的任务。机器学习方法，特别是图神经网络，提供了显著的加速，但在有限或不完整数据上训练时泛化能力较差。在这项工作中，我们系统研究了归纳偏置在提高模型可靠学习功率流能力方面的作用。具体来说，我们评估了三种物理信息策略：(i)功率流约束的损失函数，(ii)复值神经网络，(iii)基于残差的任务重构。使用包含多种低压和中压电网配置的ENGAGE数据集，我们进行对照实验以分离每种归纳偏置的影响，并评估标准预测性能和分布外泛化能力。我们的研究提供了关于哪些模型假设最有效地引导现代配电网络中可靠高效的电压预测的实用见解。


### 论文摘要

Voltage prediction in distribution grids is a critical yet difficult task for maintaining power system stability. Machine learning approaches, particularly Graph Neural Networks (GNNs), offer significant speedups but suffer from poor generalization when trained on limited or incomplete data. In this work, we systematically investigate the role of inductive biases in improving a model's ability to reliably learn power flow. Specifically, we evaluate three physics-informed strategies: (i) power-flow-constrained loss functions, (ii) complex-valued neural networks, and (iii) residual-based task reformulation. Using the ENGAGE dataset, which spans multiple low- and medium-voltage grid configurations, we conduct controlled experiments to isolate the effect of each inductive bias and assess both standard predictive performance and out-of-distribution generalization. Our study provides practical insights into which model assumptions most effectively guide learning for reliable and efficient voltage prediction in modern distribution networks.

---

## 36. Transformer Classification of Breast Lesions: The BreastDCEDL_AMBL Benchmark Dataset and 0.92 AUC Baseline

**论文链接:** [http://arxiv.org/abs/2509.26440v1](http://arxiv.org/abs/2509.26440v1)

**作者:** Naomi Fridman, Anat Goldstein

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究开发了一个基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，以提高诊断特异性并减少不必要的活检。

### 背景

乳腺磁共振成像是癌症检测和治疗规划的关键工具，但其临床应用受到特异性的限制，导致高假阳率和不必要的活检。现有公共数据集缺乏良性病变的注释，限制了良恶性分类研究。

### 目的

开发一个基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，以解决区分良性和恶性病变的挑战。

### 方法

研究团队实现了SegFormer架构，通过语义分割量化恶性像素分布。创建了BreastDCEDL_AMBL数据集，包含88名患者和133个注释病变。训练集整合了超过1,200名患者的数据，验证了迁移学习方法。

### 主要发现

SegFormer架构在病变级别分类中达到0.92的AUC，在患者水平上达到100%的敏感性和67%的特异性，可能在不遗漏恶性肿瘤的情况下消除三分之一的非必要活检。模型产生可解释的空间预测，支持临床决策制定。

### 结论

公开发布的数据集、模型和评估协议为DCE-MRI病变分类提供了第一个标准化基准，促进了临床部署的方法学进步。

### 翻译

乳腺磁共振成像是癌症检测和治疗规划的关键工具，但其临床应用受到特异性的限制，导致高假阳率和不必要的活检。本研究介绍了一个基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，以解决区分良性和恶性病变的挑战。我们实现了SegFormer架构，在病变级别分类中达到0.92的AUC，在患者水平上达到100%的敏感性和67%的特异性，这可能在不遗漏恶性肿瘤的情况下消除三分之一的非必要活检。该模型通过语义分割量化恶性像素分布，产生可解释的空间预测，支持临床决策制定。


### 论文摘要

The error is caused by special characters that arXiv's system doesn't recognize. Here's the cleaned version with all problematic characters replaced: Breast magnetic resonance imaging is a critical tool for cancer detection and treatment planning, but its clinical utility is hindered by poor specificity, leading to high false-positive rates and unnecessary biopsies. This study introduces a transformer-based framework for automated classification of breast lesions in dynamic contrast-enhanced MRI, addressing the challenge of distinguishing benign from malignant findings. We implemented a SegFormer architecture that achieved an AUC of 0.92 for lesion-level classification, with 100% sensitivity and 67% specificity at the patient level - potentially eliminating one-third of unnecessary biopsies without missing malignancies. The model quantifies malignant pixel distribution via semantic segmentation, producing interpretable spatial predictions that support clinical decision-making. To establish reproducible benchmarks, we curated BreastDCEDL_AMBL by transforming The Cancer Imaging Archive's AMBL collection into a standardized deep learning dataset with 88 patients and 133 annotated lesions (89 benign, 44 malignant). This resource addresses a key infrastructure gap, as existing public datasets lack benign lesion annotations, limiting benign-malignant classification research. Training incorporated an expanded cohort of over 1,200 patients through integration with BreastDCEDL datasets, validating transfer learning approaches despite primary tumor-only annotations. Public release of the dataset, models, and evaluation protocols provides the first standardized benchmark for DCE-MRI lesion classification, enabling methodological advancement toward clinical deployment.

---

## 37. LTA-L2S: Lexical Tone-Aware Lip-to-Speech Synthesis for Mandarin with Cross-Lingual Transfer Learning

**论文链接:** [http://arxiv.org/abs/2509.25670v1](http://arxiv.org/abs/2509.25670v1)

**作者:** Kang Yang, Yifan Liang, Fangkun Liu, Zhenping Xie, Chengshi Zheng

**发布时间:** 2025-09-30

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本文提出了一种名为LTA-L2S的中文唇语到语音合成方法，通过跨语言迁移学习和流动匹配技术解决了中文L2S合成中的口型到音素映射复杂性和音调关键性问题，显著提高了语音可理解性和音调准确性。

### 背景

唇语到语音合成在中文领域面临重大挑战，主要因为中文存在复杂的口型到音素映射关系，且声调对语音可理解性至关重要。

### 目的

开发一种能够有效处理中文唇语到语音合成挑战的方法，特别是解决口型到音素映射复杂性和声调建模问题。

### 方法

1) 采用跨语言迁移学习策略，适应预先训练的英语音频-视觉自监督学习模型；2) 使用流动匹配模型生成F0轮廓，由ASR微调的SSL语音单元指导；3) 通过两阶段训练范式提升语音质量，流动匹配后网络完善粗略频谱图。

### 主要发现

大量实验表明，LTA-L2S在语音可理解性和音调准确性方面显著优于现有方法。

### 结论

LTA-L2S通过迁移学习利用英语领域知识，结合流动匹配技术有效解决了中文唇语到语音合成的关键挑战，是一种高效的方法。

### 翻译

中文唇语到语音合成是一个重大挑战，受到复杂的口型到音素映射和音调在可理解性中关键角色的阻碍。为解决这一问题，我们提出了词调感知唇语到语音（LTA-L2S）。为处理口型到音素复杂性，我们的模型通过跨语言迁移学习策略适应了预先训练的英语音频-视觉自监督学习模型。该策略不仅将从大量英语数据中学到的通用知识迁移到中文领域，还避免了从头训练此类模型的巨大成本。为专门建模词调并提高可理解性，我们进一步采用流动匹配模型生成F0轮廓。此生成过程由ASR微调的SSL语音单元指导，这些单元包含重要的超音段信息。整体语音质量通过两阶段训练范式得到提升，其中流动匹配后网络完善第一阶段产生的粗略频谱图。大量实验表明，LTA-L2S在语音可理解性和音调准确性方面都显著优于现有方法。


### 论文摘要

Lip-to-speech (L2S) synthesis for Mandarin is a significant challenge, hindered by complex viseme-to-phoneme mappings and the critical role of lexical tones in intelligibility. To address this issue, we propose Lexical Tone-Aware Lip-to-Speech (LTA-L2S). To tackle viseme-to-phoneme complexity, our model adapts an English pre-trained audio-visual self-supervised learning (SSL) model via a cross-lingual transfer learning strategy. This strategy not only transfers universal knowledge learned from extensive English data to the Mandarin domain but also circumvents the prohibitive cost of training such a model from scratch. To specifically model lexical tones and enhance intelligibility, we further employ a flow-matching model to generate the F0 contour. This generation process is guided by ASR-fine-tuned SSL speech units, which contain crucial suprasegmental information. The overall speech quality is then elevated through a two-stage training paradigm, where a flow-matching postnet refines the coarse spectrogram from the first stage. Extensive experiments demonstrate that LTA-L2S significantly outperforms existing methods in both speech intelligibility and tonal accuracy.

---

## 38. Capacity-Net-Based RIS Precoding Design without Channel Estimation for mmWave MIMO System

**论文链接:** [http://arxiv.org/abs/2509.25660v1](http://arxiv.org/abs/2509.25660v1)

**作者:** Chun-Yuan Huang, Po-Heng Chou, Wan-Jen Huang, Ying-Ren Chien, Yu Tsao

**发布时间:** 2025-09-30

**DOI:** 10.1109/PIMRC59610.2024.10817310

**备注:** 10 pages, 5 figures, and published in 2024 IEEE PIMRC

### GPT解析

### 总结

本文提出了Capacity-Net，一种用于RIS辅助毫米波MIMO系统的无监督学习方法，通过优化RIS相移因子提高系统速率，无需完整的信道状态信息。

### 背景

毫米波频段存在严重的信道衰落问题，传统优化算法需要完整准确的信道状态信息，但RIS主要由无源组件组成，使得获取完整CSI具有挑战性。

### 目的

开发一种无需完整CSI的无监督学习方法，通过优化RIS反射元件的相移因子来最大化RIS辅助毫米波MIMO系统中的可实现速率。

### 方法

提出Capacity-Net框架，建立接收到的导频信号、优化的RIS相移和由此产生的可实现速率之间的直接映射，而不是传统的信道估计方法。

### 主要发现

仿真结果表明，基于Capacity-Net的无监督学习方法优于传统基于信道估计的学习方法。

### 结论

Capacity-Net能够有效解决RIS辅助毫米波MIMO系统中CSI获取困难的问题，通过无监督学习技术提高系统性能。

### 翻译

本文提出了一种名为Capacity-Net的新型无监督学习方法，旨在最大化智能反射面辅助的毫米波多输入多输出系统中可实现的速率。为了对抗毫米波频段的严重信道衰落，我们优化了RIS中反射元件的相移因子以提高可实现的速率。然而，大多数优化算法严重依赖于完整且准确的信道状态信息，而由于RIS主要由无源组件组成，获取这些信息通常具有挑战性。为了克服这一挑战，我们利用接收到的导频信号提供的隐式CSI采用无监督学习技术。具体来说，评估无监督学习方法当前优化结果的可实现速率作为性能指标通常需要完美的CSI。与信道估计不同，Capacity-Net被提出用于建立接收到的导频信号、优化的RIS相移和由此产生的可实现速率之间的映射。仿真结果表明，与传统基于信道估计的学习方法相比，所提出的基于Capacity-Net的无监督学习方法具有优越性。


### 论文摘要

In this paper, we propose Capacity-Net, a novel unsupervised learning approach aimed at maximizing the achievable rate in reflecting intelligent surface (RIS)-aided millimeter-wave (mmWave) multiple input multiple output (MIMO) systems. To combat severe channel fading of the mmWave spectrum, we optimize the phase-shifting factors of the reflective elements in the RIS to enhance the achievable rate. However, most optimization algorithms rely heavily on complete and accurate channel state information (CSI), which is often challenging to acquire since the RIS is mostly composed of passive components. To circumvent this challenge, we leverage unsupervised learning techniques with implicit CSI provided by the received pilot signals. Specifically, it usually requires perfect CSI to evaluate the achievable rate as a performance metric of the current optimization result of the unsupervised learning method. Instead of channel estimation, the Capacity-Net is proposed to establish a mapping among the received pilot signals, optimized RIS phase shifts, and the resultant achievable rates. Simulation results demonstrate the superiority of the proposed Capacity-Net-based unsupervised learning approach over learning methods based on traditional channel estimation.

---

## 39. MetaChest: Generalized few-shot learning of patologies from chest X-rays

**论文链接:** [http://arxiv.org/abs/2509.25590v1](http://arxiv.org/abs/2509.25590v1)

**作者:** Berenice Montalvo-Lezama, Gibran Fuentes-Pineda

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了MetaChest数据集，用于解决医学图像分析中标注数据有限的问题，并通过实验评估了少样本学习方法在胸部X光片分类任务中的表现。

### 背景

标注数据的有限性是将深度学习方法应用于医学图像分析的主要挑战。少样本学习方法旨在从少量标记示例中识别新类别，但医学应用如胸部X光片分类通常需要同时学习新类别并利用已知类别的知识，这更符合广义少样本分类场景，而在此场景下少样本学习研究不足。

### 目的

开发一个适合医学图像分析的少样本学习方法，解决医学图像分析中标注数据有限的问题，并评估不同方法在少样本多标签分类任务中的表现。

### 方法

提出MetaChest数据集，包含来自四个公共数据库的479,215张胸部X光片，设计专门为标准少样本分类的元集分区，开发生成多标签片段的算法，评估标准迁移学习方法和ProtoNet扩展在多种少样本多标签分类任务上的表现。

### 主要发现

增加每个片段的类别数量和每个类别的训练示例数量可以提高分类性能；迁移学习方法持续优于专为少样本学习设计的ProtoNet扩展；更高分辨率的图像可提高准确性但增加计算成本；高效模型架构可在减少资源需求的情况下实现与更大模型相当的性能。

### 结论

研究证明了少样本学习在医学图像分析中的可行性，迁移学习方法在少样本医学图像分类任务中表现良好，模型效率和图像分辨率对性能有重要影响。

### 翻译

标注数据的有限性是将深度学习方法应用于医学图像分析的主要挑战。少样本学习方法旨在仅从少量标记示例中识别新类别。这些方法通常在标准少样本学习设置下研究，其中任务中的所有类别都是新的。然而，医学应用如胸部X光片的病理分类通常需要同时学习新类别并利用已知类别的知识，这更符合广义少样本分类场景。尽管其实际相关性，少样本学习在此背景下研究甚少。在本文中，我们提出MetaChest，这是一个从四个公共数据库收集的包含479,215张胸部X光片的大规模数据集。MetaChest包含专门为标准少样本分类设计的元集分区，以及用于生成多标签片段的算法。我们进行了广泛的实验，评估了标准迁移学习方法和ProtoNet扩展在各种少样本多标签分类任务上的表现。我们的结果表明，增加每个片段的类别数量和每个类别的训练示例数量可以提高分类性能。值得注意的是，尽管迁移学习方法并非专为少样本学习而设计，但它持续优于ProtoNet扩展。我们还表明，更高分辨率的图像可以提高准确性，但会增加计算成本，而高效的模型架构可以在显著减少资源需求的情况下实现与更大模型相当的性能。


### 论文摘要

The limited availability of annotated data presents a major challenge for applying deep learning methods to medical image analysis. Few-shot learning methods aim to recognize new classes from only a small number of labeled examples. These methods are typically studied under the standard few-shot learning setting, where all classes in a task are new. However, medical applications such as pathology classification from chest X-rays often require learning new classes while simultaneously leveraging knowledge of previously known ones, a scenario more closely aligned with generalized few-shot classification. Despite its practical relevance, few-shot learning has been scarcely studied in this context. In this work, we present MetaChest, a large-scale dataset of 479,215 chest X-rays collected from four public databases. MetaChest includes a meta-set partition specifically designed for standard few-shot classification, as well as an algorithm for generating multi-label episodes. We conduct extensive experiments evaluating both a standard transfer learning approach and an extension of ProtoNet across a wide range of few-shot multi-label classification tasks. Our results demonstrate that increasing the number of classes per episode and the number of training examples per class improves classification performance. Notably, the transfer learning approach consistently outperforms the ProtoNet extension, despite not being tailored for few-shot learning. We also show that higher-resolution images improve accuracy at the cost of additional computation, while efficient model architectures achieve comparable performance to larger models with significantly reduced resource requirements.

---

## 40. ClustRecNet: A Novel End-to-End Deep Learning Framework for Clustering Algorithm Recommendation

**论文链接:** [http://arxiv.org/abs/2509.25289v1](http://arxiv.org/abs/2509.25289v1)

**作者:** Mohammadreza Bakhtyari, Bogdan Mazoure, Renato Cordeiro de Amorim, Guillaume Rabusseau, Vladimir Makarenkov

**发布时间:** 2025-09-29

### GPT解析

### 总结

论文介绍了一种名为ClustRecNet的新型深度学习推荐框架，用于为给定数据集确定最合适的聚类算法，解决了无监督学习中聚类算法选择的长期挑战。

### 背景

在无监督学习中，聚类算法的选择一直是一个长期存在的挑战，传统的聚类有效性指标和现有的AutoML聚类推荐方法在性能上仍有提升空间。

### 目的

开发一个深度学习框架，能够自动为给定数据集推荐最合适的聚类算法，减少对手工设计的元特征和传统聚类有效性指标的依赖。

### 方法

构建了一个包含34,000个具有不同结构特性的合成数据集的综合数据仓库，使用10种流行聚类算法处理每个数据集，通过调整兰德指数评估聚类结果建立真实标签，设计了结合卷积、残差和注意力机制的神经网络架构，支持端到端训练学习数据集的紧凑表示。

### 主要发现

在合成和真实世界基准测试中，该深度学习模型始终优于传统CVIs和最先进的AutoML聚类推荐方法，在合成数据上相比Calinski-Harabasz指数实现了0.497的ARI提升，在真实数据上实现了15.3%的ARI增益。

### 结论

ClustRecNet框架能够有效解决聚类算法选择问题，通过深度学习技术自动推荐最适合的聚类算法，显著优于传统方法和现有的AutoML方法。

### 翻译

我们引入了ClustRecNet——一种基于深度学习(DL)的新型推荐框架，用于确定给定数据集最合适的聚类算法，解决了无监督学习中长期存在的聚类算法选择挑战。为了在此背景下实现监督学习，我们构建了一个包含34,000个具有不同结构特性的合成数据集的综合数据仓库。每个数据集都使用10种流行的聚类算法进行处理。通过调整兰德指数(ARI)评估得到的聚类结果，以建立真实标签，用于我们DL模型的训练和评估。所提出的网络架构集成了卷积、残差和注意力机制，以从输入数据中捕获局部和全局结构模式。这种设计支持端到端训练，学习数据集的紧凑表示，并能够直接推荐最合适的聚类算法，减少对手工设计的元特征和传统聚类有效性指标(CVIs)的依赖。


### 论文摘要

We introduce ClustRecNet - a novel deep learning (DL)-based recommendation framework for determining the most suitable clustering algorithms for a given dataset, addressing the long-standing challenge of clustering algorithm selection in unsupervised learning. To enable supervised learning in this context, we construct a comprehensive data repository comprising 34,000 synthetic datasets with diverse structural properties. Each of them was processed using 10 popular clustering algorithms. The resulting clusterings were assessed via the Adjusted Rand Index (ARI) to establish ground truth labels, used for training and evaluation of our DL model. The proposed network architecture integrates convolutional, residual, and attention mechanisms to capture both local and global structural patterns from the input data. This design supports end-to-end training to learn compact representations of datasets and enables direct recommendation of the most suitable clustering algorithm, reducing reliance on handcrafted meta-features and traditional Cluster Validity Indices (CVIs). Comprehensive experiments across synthetic and real-world benchmarks demonstrate that our DL model consistently outperforms conventional CVIs (e.g. Silhouette, Calinski-Harabasz, Davies-Bouldin, and Dunn) as well as state-of-the-art AutoML clustering recommendation approaches (e.g. ML2DAC, AutoCluster, and AutoML4Clust). Notably, the proposed model achieves a 0.497 ARI improvement over the Calinski-Harabasz index on synthetic data and a 15.3% ARI gain over the best-performing AutoML approach on real-world data.

---

## 41. Attention over Scene Graphs: Indoor Scene Representations Toward CSAI Classification

**论文链接:** [http://arxiv.org/abs/2509.26457v1](http://arxiv.org/abs/2509.26457v1)

**作者:** Artur Barros, Carlos Caetano, João Macedo, Jefersson A. dos Santos, Sandra Avila

**发布时间:** 2025-09-30

**备注:** British Machine Vision Conference (BMVC 2025), in the From Scene  Understanding to Human Modeling Workshop

### GPT解析

### 总结

提出了一种名为ASGRA的新型框架，通过场景图和图注意力网络进行室内场景分类和敏感内容分析

### 背景

室内场景分类是计算机视觉中的关键任务，应用于机器人技术和敏感内容分析（如儿童性虐待图像CSAI分类），但面临物体间复杂关系和空间布局的挑战

### 目的

开发一种能够有效处理室内场景分类和敏感内容分析的方法，同时提供可解释性和隐私保护

### 方法

ASGRA框架，在结构化图表示上操作而非原始像素；先将图像转换为场景图，然后使用图注意力网络进行推理，直接建模场景组件间的交互

### 主要发现

在Places8上达到81.27%的平衡准确率，超过基于图像的方法；在CSAI评估中达到74.27%的平衡准确率

### 结论

结构化场景表示是室内场景分类和CSAI分类的稳健范式

### 翻译

室内场景分类是计算机视觉中的一个关键任务，应用范围广泛，从机器人技术到敏感内容分析，如儿童性虐待图像（CSAI）分类。由于物体之间复杂的关系和空间布局，这个问题尤其具有挑战性。在这项工作中，我们提出了用于敏感内容分析的场景图注意力（ASGRA），这是一个新型框架，它在结构化图表示而非原始像素上运行。通过首先将图像转换为场景图，然后使用图注意力网络进行推理，ASGRA直接建模了场景组件之间的交互。这种方法提供了两个关键优势：（i）通过识别物体和关系提供内在可解释性，以及（ii）隐私保护，使得无需直接访问敏感图像即可进行模型训练。在Places8上，我们实现了81.27%的平衡准确率，超过了基于图像的方法。与执法部门合作的CSAI实际评估达到了74.27%的平衡准确率。我们的研究结果表明，结构化场景表示是室内场景分类和CSAI分类的稳健范式。代码已在https://github.com/tutuzeraa/ASGRA公开。


### 论文摘要

Indoor scene classification is a critical task in computer vision, with wide-ranging applications that go from robotics to sensitive content analysis, such as child sexual abuse imagery (CSAI) classification. The problem is particularly challenging due to the intricate relationships between objects and complex spatial layouts. In this work, we propose the Attention over Scene Graphs for Sensitive Content Analysis (ASGRA), a novel framework that operates on structured graph representations instead of raw pixels. By first converting images into Scene Graphs and then employing a Graph Attention Network for inference, ASGRA directly models the interactions between a scene's components. This approach offers two key benefits: (i) inherent explainability via object and relationship identification, and (ii) privacy preservation, enabling model training without direct access to sensitive images. On Places8, we achieve 81.27% balanced accuracy, surpassing image-based methods. Real-world CSAI evaluation with law enforcement yields 74.27% balanced accuracy. Our results establish structured scene representations as a robust paradigm for indoor scene classification and CSAI classification. Code is publicly available at https://github.com/tutuzeraa/ASGRA.

---

## 42. Seeing Space and Motion: Enhancing Latent Actions with Spatial and Dynamic Awareness for VLA

**论文链接:** [http://arxiv.org/abs/2509.26251v1](http://arxiv.org/abs/2509.26251v1)

**作者:** Zhejia Cai, Yandan Yang, Xinyuan Chang, Shiyi Liang, Ronghan Chen, Feng Xiong, Mu Xu, Ruqi Huang

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种改进的潜在动作框架Farsighted-LAM和基于其构建的端到端VLA框架SSM-VLA，解决了传统LAMs在空间理解和时间感知方面的瓶颈问题，通过几何感知空间编码、多尺度时间建模和明确推理提高了具身智能的鲁棒性和泛化能力。

### 背景

Latent Action Models (LAMs) 使 Vision-Language-Action (VLA) 系统能够从大规模未标注数据中学习语义动作表示，但存在两个主要瓶颈：1) 常用的端到端训练图像编码器空间理解能力差；2) 当输入帧距离较远时，LAMs表现脆弱，导致时间感知能力有限，这些因素阻碍了稳定清晰的动作建模。

### 目的

解决LAMs的空间理解差和时间感知有限的问题，提高动作建模的稳定性和清晰度，增强具身智能的鲁棒性和泛化能力。

### 方法

1) 提出Farsighted-LAM，具有几何感知空间编码和多尺度时间建模的潜在动作框架，能从连续帧中捕获结构先验和动态运动模式；2) 提出SSM-VLA，基于Farsighted-LAM的端到端VLA框架，集成结构感知与视觉思维链模块，明确推理环境动态，提高决策一致性和可解释性。

### 主要发现

在模拟和现实世界环境中的多个VLA任务上验证了SSM-VLA，达到了最先进的性能，证明了结合几何感知建模、时间一致性和明确推理策略的有效性。

### 结论

结合几何感知建模、时间一致性和明确推理的策略在增强具身智能的鲁棒性和泛化能力方面是有效的，为VLA系统的发展提供了新思路。

### 翻译

潜在动作模型(LAMs)使视觉-语言-动作(VLA)系统能够从大规模未标注数据中学习语义动作表示。然而，我们确定了LAMs的两个瓶颈：1) 通常采用的端到端训练图像编码器空间理解能力差；2) 当输入帧距离较远时，LAMs可能表现脆弱，导致时间感知能力有限。这些因素不可避免地阻碍了稳定和清晰的动作建模。为此，我们提出了Farsighted-LAM，这是一种具有几何感知空间编码和多尺度时间建模的潜在动作框架，能够从连续帧中捕获结构先验和动态运动模式。我们进一步提出了SSM-VLA，这是一个基于Farsighted-LAM构建的端到端VLA框架，它将结构感知与视觉思维链模块集成，以明确推理环境动态，提高决策一致性和可解释性。我们在模拟和现实世界环境中的多个VLA任务上验证了SSM-VLA，并取得了最先进的性能。我们的结果表明，我们结合几何感知建模、时间一致性和明确推理的策略在增强具身智能的鲁棒性和泛化能力方面是有效的。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有潜在动作模型（LAMs）的两个关键瓶颈：一是图像编码器缺乏空间理解能力，二是当输入帧距离较远时时间感知能力有限。这些问题在现实中很重要，因为它们导致机器人在复杂环境中做出不稳定和语义模糊的决策，限制了机器人在现实世界中的可靠应用和跨平台泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有LAM模型的局限性，针对性地提出了两个关键设计：1) 使用DINOv2特征进行几何感知的空间编码，捕捉结构先验知识；2) 通过连续帧序列进行多尺度时间建模，捕捉长期动态和精细运动转换。作者还借鉴了视觉期望增强工作（如VideoAgent、Seer、VPP等），这些方法使用视觉期望增强当前观察上下文，而本文进一步通过预测带有几何先验的未来视觉状态来增强动作指导。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 增强空间理解能力，使用DINOv2特征提取器获取富含几何和语义信息的视觉特征；2) 改进时间感知能力，通过处理多个未来关键帧捕获长期动态；3) 引入显式推理机制，在执行动作前模拟环境动态。整体流程分为三个阶段：1) 未来观察预测阶段，生成视觉思维链；2) 远见潜在动作建模阶段，整合空间和时间动态；3) 模块化动作块预测阶段，生成最终可执行动作。这三个阶段通过多模态协同注意力机制在一个统一架构中实现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Farsighted-LAM框架，使用DINOv2特征进行几何感知编码和多帧时间建模；2) SSM-VLA端到端框架，集成结构化感知与视觉思维链模块；3) 多模态重建损失，确保模型在学习表示中保持外观和几何一致性。相比之前的工作，本文方法在空间理解上从RGB编码转向几何感知特征，在时间感知上从两帧输入扩展到多帧序列，在推理机制上引入视觉思维链作为中间步骤，使潜在动作更加稳定和语义丰富。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了SSM-VLA框架，通过结合几何感知的空间编码、多尺度时间建模和显式视觉推理，显著提升了视觉-语言-动作系统在复杂环境中的空间理解、时间感知和决策能力，实现了最先进的性能并增强了机器人在现实世界中的泛化能力。'}


### 论文摘要

Latent Action Models (LAMs) enable Vision-Language-Action (VLA) systems to learn semantic action representations from large-scale unannotated data. Yet, we identify two bottlenecks of LAMs: 1) the commonly adopted end-to-end trained image encoder suffers from poor spatial understanding; 2) LAMs can be fragile when input frames are distant, leading to limited temporal perception. Such factors inevitably hinder stable and clear action modeling. To this end, we propose Farsighted-LAM, a latent action framework with geometry-aware spatial encoding and multi-scale temporal modeling, capturing structural priors and dynamic motion patterns from consecutive frames. We further propose SSM-VLA, an end-to-end VLA framework built upon Farsighted-LAM, which integrates structured perception with a visual Chain-of-Thought module to explicitly reason about environmental dynamics, enhancing decision consistency and interpretability. We validate SSM-VLA on multiple VLA tasks in both simulation and real-world settings, and achieve state-of-the-art performance. Our results demonstrate that our strategy of combining geometry-aware modeling, temporal coherence, and explicit reasoning is effective in enhancing the robustness and generalizability of embodied intelligence.

---

## 43. Neighbor-aware informal settlement mapping with graph convolutional networks

**论文链接:** [http://arxiv.org/abs/2509.26171v1](http://arxiv.org/abs/2509.26171v1)

**作者:** Thomas Hallopeau, Joris Guérin, Laurent Demagistri, Christovam Barcellos, Nadine Dessay

**发布时间:** 2025-09-30

**备注:** 10 pages, 3 figures, 2 tables. Accepted at the ECML PKDD 2025  Workshop on Machine Learning for Earth Observation

### GPT解析

### 总结

本文提出了一种基于图的框架，通过将空间单元与其邻居一起嵌入图结构，并使用图卷积网络进行分类，有效解决了传统方法忽略城市结构关系的问题，在里约热内卢的案例研究中表现出色。

### 背景

在快速发展的城市中，非正规定居点的映射对于解决城市规划、公共卫生和基础设施相关挑战至关重要。地理空间机器学习已成为从遥感数据中检测和绘制这些区域的关键工具，但现有方法通常将空间单元独立处理，忽略了城市结构的关系结构。

### 目的

提出一种基于图的框架，明确将局部地理上下文整合到非正规定居点的分类过程中。

### 方法

每个空间单元（单元格）与其相邻邻居一起嵌入图结构中，训练一个轻量级图卷积网络（GCN）来分类中心单元格是否属于非正规定居点。在里约热内卢的案例研究中进行实验，使用五个不同区域的空间交叉验证，确保在异质城市景观中的鲁棒性和泛化能力。

### 主要发现

该方法优于标准基线，比单个单元格分类的Kappa系数提高了17个百分点。基于图的建模优于简单连接相邻单元格的特征，证明了编码空间结构对城市场景理解的益处。

### 结论

基于图的框架能够有效整合空间上下文信息，提高非正规定居点分类的准确性，为城市规划和政策制定提供更可靠的数据支持。

### 翻译

绘制非正规定居点地图对于应对快速发展的城市中与城市规划、公共卫生和基础设施相关的挑战至关重要。地理空间机器学习已成为从遥感数据中检测和绘制这些区域的关键工具。然而，现有方法通常将空间单元独立处理，忽略了城市结构的关系结构。我们提出了一种基于图的框架，明确将局部地理上下文整合到分类过程中。每个空间单元（单元格）与其相邻邻居一起嵌入图结构中，训练一个轻量级图卷积网络（GCN）来分类中心单元格是否属于非正规定居点。在里约热内卢的案例研究中进行实验，使用五个不同区域的空间交叉验证，确保在异质城市景观中的鲁棒性和泛化能力。我们的方法优于标准基线，比单个单元格分类的Kappa系数提高了17个百分点。我们还表明，基于图的建模优于简单连接相邻单元格的特征，证明了编码空间结构对城市场景理解的益处。


### 论文摘要

Mapping informal settlements is crucial for addressing challenges related to urban planning, public health, and infrastructure in rapidly growing cities. Geospatial machine learning has emerged as a key tool for detecting and mapping these areas from remote sensing data. However, existing approaches often treat spatial units independently, neglecting the relational structure of the urban fabric. We propose a graph-based framework that explicitly incorporates local geographical context into the classification process. Each spatial unit (cell) is embedded in a graph structure along with its adjacent neighbors, and a lightweight Graph Convolutional Network (GCN) is trained to classify whether the central cell belongs to an informal settlement. Experiments are conducted on a case study in Rio de Janeiro using spatial cross-validation across five distinct zones, ensuring robustness and generalizability across heterogeneous urban landscapes. Our method outperforms standard baselines, improving Kappa coefficient by 17 points over individual cell classification. We also show that graph-based modeling surpasses simple feature concatenation of neighboring cells, demonstrating the benefit of encoding spatial structure for urban scene understanding.

---

## 44. Human-MME: A Holistic Evaluation Benchmark for Human-Centric Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2509.26165v1](http://arxiv.org/abs/2509.26165v1)

**作者:** Yuansen Liu, Haiming Tang, Jinlong Peng, Jiangning Zhang, Xiaozhong Ji, Qingdong He, Donghao Luo, Zhenye Gan, Junwei Zhu, Yunhang Shen, Chaoyou Fu, Chengjie Wang, Xiaobin Hu, Shuicheng Yan

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了Human-MME基准，用于全面评估多模态大语言模型在人类中心场景理解方面的能力，提供多样化的场景、渐进式的评估维度和高质量的注释。

### 背景

多模态大语言模型在视觉理解任务中取得显著进展，但对人类中心场景的理解能力很少被探索，主要缺乏考虑人类细粒度水平和因果推理能力的全面评估基准。

### 目的

提出Human-MME基准，为多模态大语言模型在人类中心场景理解方面提供更全面的评估框架。

### 方法

构建具有三个关键特征的Human-MME基准：1)人类场景多样性，涵盖4个主要视觉领域、15个次级领域和43个子领域；2)渐进式和多样化的评估维度，从细粒度感知到高维推理，包含8个维度和19,945个图像问题对；3)高质量注释和丰富数据范式，构建自动注释流程和人工标注平台。

### 主要发现

在17个最先进的多模态大语言模型上的实验有效暴露了它们在人类中心场景理解方面的局限性，为未来研究提供方向。

### 结论

Human-MME基准为评估多模态大语言模型在人类中心场景理解方面提供了全面框架，有助于推动该领域研究进展。

### 翻译

多模态大语言模型在视觉理解任务中已经显示出显著的进展。然而，它们对人类中心场景的理解能力很少被探索，主要是由于缺乏全面的评估基准，这些基准需要考虑人类导向的细粒度水平和更高维度的因果推理能力。鉴于人体物理复杂性和细粒度结构标注的困难，高质量评估基准面临严峻挑战。在本文中，我们提出了Human-MME，这是一个精心策划的基准，旨在为多模态大语言模型在人类中心场景理解方面提供更全面的评估。与其他现有基准相比，我们的工作提供了三个关键特征：1. 人类场景的多样性，涵盖4个主要视觉领域、15个次级领域和43个子领域，确保广泛的场景覆盖。2. 渐进式和多样化的评估维度，从人类导向的细粒度感知到更高维度的推理逐步评估基于人类的活动，包含八个维度、19,945个真实世界图像问题对和评估套件。3. 高质量注释和丰富的数据范式，构建自动注释流程和人工标注平台，支持严格的手动标注以促进精确可靠的模型评估。我们的基准通过构建选择、简答、定位、排序和判断问题组件及其复杂组合问题，将单目标理解扩展到多人和多图像相互理解。在17个最先进的多模态大语言模型上的广泛实验有效地暴露了它们的局限性，并指导未来的多模态大语言模型研究向更好的人类中心图像理解方向发展。所有数据和代码都可以在https://github.com/Yuan-Hou/Human-MME获取。


### 论文摘要

Multimodal Large Language Models (MLLMs) have demonstrated significant advances in visual understanding tasks. However, their capacity to comprehend human-centric scenes has rarely been explored, primarily due to the absence of comprehensive evaluation benchmarks that take into account both the human-oriented granular level and higher-dimensional causal reasoning ability. Such high-quality evaluation benchmarks face tough obstacles, given the physical complexity of the human body and the difficulty of annotating granular structures. In this paper, we propose Human-MME, a curated benchmark designed to provide a more holistic evaluation of MLLMs in human-centric scene understanding. Compared with other existing benchmarks, our work provides three key features: 1. Diversity in human scene, spanning 4 primary visual domains with 15 secondary domains and 43 sub-fields to ensure broad scenario coverage. 2. Progressive and diverse evaluation dimensions, evaluating the human-based activities progressively from the human-oriented granular perception to the higher-dimensional reasoning, consisting of eight dimensions with 19,945 real-world image question pairs and an evaluation suite. 3. High-quality annotations with rich data paradigms, constructing the automated annotation pipeline and human-annotation platform, supporting rigorous manual labeling to facilitate precise and reliable model assessment. Our benchmark extends the single-target understanding to the multi-person and multi-image mutual understanding by constructing the choice, short-answer, grounding, ranking and judgment question components, and complex questions of their combination. The extensive experiments on 17 state-of-the-art MLLMs effectively expose the limitations and guide future MLLMs research toward better human-centric image understanding. All data and code are available at https://github.com/Yuan-Hou/Human-MME.

---

## 45. EasyOcc: 3D Pseudo-Label Supervision for Fully Self-Supervised Semantic Occupancy Prediction Models

**论文链接:** [http://arxiv.org/abs/2509.26087v1](http://arxiv.org/abs/2509.26087v1)

**作者:** Seamie Hayes, Ganesh Sistu, Ciarán Eising

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种基于基础模型生成3D伪标签的自监督学习方法，用于语义占用预测任务，显著降低了计算成本并提高了性能。

### 背景

自监督模型在语义占用预测领域取得显著进展，利用复杂的损失计算策略弥补真实标签缺失。现有技术如新视图合成、跨视图渲染和深度估计虽能解决语义和深度模糊问题，但训练阶段计算成本和内存消耗高。

### 目的

减轻现有技术的计算成本和内存使用问题，提高语义占用预测性能。

### 方法

提出由Grounded-SAM和Metric3Dv2基础模型生成的3D伪真实标签，并利用时间信息进行标签密集化；开发了仅从标签学习的简化模型EasyOcc，避免复杂渲染策略。

### 主要发现

3D伪标签可轻松集成现有模型，在OccNeRF中mIoU从9.73提升至14.09(提高45%)；EasyOcc模型达到13.86 mIoU；整个场景评估时，EasyOcc达到7.71 mIoU，比之前最佳模型高出31%。

### 结论

基础模型、时间上下文和损失计算空间的选择对自监督学习进行全面场景理解至关重要。

### 翻译

自监督模型最近取得了显著进展，特别是在语义占用预测领域。这些模型利用复杂的损失计算策略来补偿真实标签的缺失。例如，已探索新视图合成、跨视图渲染和深度估计等技术来解决语义和深度模糊问题。然而，这些技术在训练阶段通常需要高计算成本和内存使用，特别是在新视图合成的情况下。为缓解这些问题，我们提出由基础模型Grounded-SAM和Metric3Dv2生成的3D伪真实标签，并利用时间信息进行标签密集化。我们的3D伪标签可以轻松集成到现有模型中，实现了显著的性能提升，在OccNeRF模型中实现时mIoU从9.73增加到14.09，提高了45%。这与该领域的早期进展形成对比，那些进展通常不容易转移到其他架构。此外，我们提出了一个简化模型EasyOcc，达到13.86 mIoU。该模型仅从我们的标签进行学习，避免了之前提到的复杂渲染策略。此外，我们的方法使模型能够在不应用相机掩码的情况下在整个场景上评估时达到最先进的性能，EasyOcc达到7.71 mIoU，比之前最好的模型高出31%。这些发现突显了基础模型、时间上下文和损失计算空间选择在自监督学习进行全面场景理解中的关键重要性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自监督语义占用预测模型中计算复杂度高和内存消耗大的问题。现有方法使用新视角合成等技术解决语义和深度模糊，但训练阶段需要高昂计算成本。这个问题在现实中很重要，特别是在自动驾驶领域，准确的语义占用预测对理解周围环境至关重要，但获取大量精确的3D标注数据困难和昂贵。自监督学习可减少对标注数据的依赖，降低训练成本，提高模型泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有自监督方法依赖复杂渲染技术，计算成本高；意识到基础模型能生成高质量2D伪标签但通常在2D空间使用；提出将2D伪标签转换为3D伪标签，直接在3D空间进行损失计算；利用时间信息增加标签密度但只限于静态对象避免重复。作者借鉴了Grounded-SAM进行语义分割、Metric3Dv2进行深度估计，并参考了SelfOcc、OccNeRF和GaussianOcc等现有自监督模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基础模型生成3D伪标签，直接在3D空间进行损失计算，避免复杂渲染技术，同时利用时间信息增加标签密度。流程包括：1)语义点云生成：从Grounded-SAM获取语义图，Metric3Dv2获取深度图，投影到3D空间并聚合；2)语义点云稠密化：使用过去13个时间样本增加密度，移除动态点避免重复；3)语义点云体素化：按0.4m³分辨率体素化，设置10点阈值确定占用；4)模型训练：使用3D伪标签作为监督，设计伪损失函数集成到现有模型或训练新模型EasyOcc。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)3D伪标签：使用基础模型生成3D伪标签直接在3D空间计算损失；2)无缝集成：3D伪标签可轻松集成现有模型提高性能；3)动态类别分割：显著提高动态对象预测性能；4)整体场景表示：使模型更好理解整个场景。不同之处：1)避免复杂渲染技术：之前方法依赖新视角合成等，本文直接3D空间计算损失；2)时间信息利用：只在训练阶段聚合时间样本，避免推理阶段额外计算；3)简化模型：提出EasyOcc模型，仅用3D伪标签训练，无需复杂渲染；4)全面场景理解：模型能更好理解被当前视角遮挡的区域。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种使用3D伪标签监督的自监督语义占用预测方法，通过简化训练过程并利用时间信息，显著提高了模型性能，特别是在整个场景理解方面。'}


### 论文摘要

Self-supervised models have recently achieved notable advancements, particularly in the domain of semantic occupancy prediction. These models utilize sophisticated loss computation strategies to compensate for the absence of ground-truth labels. For instance, techniques such as novel view synthesis, cross-view rendering, and depth estimation have been explored to address the issue of semantic and depth ambiguity. However, such techniques typically incur high computational costs and memory usage during the training stage, especially in the case of novel view synthesis. To mitigate these issues, we propose 3D pseudo-ground-truth labels generated by the foundation models Grounded-SAM and Metric3Dv2, and harness temporal information for label densification. Our 3D pseudo-labels can be easily integrated into existing models, which yields substantial performance improvements, with mIoU increasing by 45\%, from 9.73 to 14.09, when implemented into the OccNeRF model. This stands in contrast to earlier advancements in the field, which are often not readily transferable to other architectures. Additionally, we propose a streamlined model, EasyOcc, achieving 13.86 mIoU. This model conducts learning solely from our labels, avoiding complex rendering strategies mentioned previously. Furthermore, our method enables models to attain state-of-the-art performance when evaluated on the full scene without applying the camera mask, with EasyOcc achieving 7.71 mIoU, outperforming the previous best model by 31\%. These findings highlight the critical importance of foundation models, temporal context, and the choice of loss computation space in self-supervised learning for comprehensive scene understanding.

---

## 46. MUVLA: Learning to Explore Object Navigation via Map Understanding

**论文链接:** [http://arxiv.org/abs/2509.25966v1](http://arxiv.org/abs/2509.25966v1)

**作者:** Peilong Han, Fan Jia, Min Zhang, Yutao Qiu, Hongyao Tang, Yan Zheng, Tiancai Wang, Jianye Hao

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种名为MUVLA的地图理解视觉-语言-动作模型，专门用于物体导航任务。该模型通过语义地图抽象统一历史信息，并采用三阶段训练流程实现有效的探索策略。

### 背景

物体导航任务需要模型能够理解和处理空间信息，而现有的方法可能难以处理历史信息和多样化的演示数据。

### 目的

开发一个能够基于目标物体描述预测动作序列的模型，并通过奖励引导回报建模来放大监督信号，从而学习有效的探索行为。

### 方法

MUVLA模型将当前和历史观测以及语义地图作为输入，利用语义地图抽象统一历史信息。采用三阶段训练流程：学习地图级空间理解、从混合质量演示中模仿行为、奖励放大。通过基于密集短期进度信号的奖励引导回报建模来放大监督。

### 主要发现

在HM3D和Gibson基准测试上的实验表明，MUVLA实现了良好的泛化能力，即使从低质量或部分成功的轨迹中也能学习有效的探索行为。

### 结论

MUVLA能够将多样化的演示统一为强大的空间表示，并生成更合理的探索策略，从而在物体导航任务中表现出色。

### 翻译

在本文中，我们提出了MUVLA，一种专为物体导航设计的地图理解视觉-语言-动作模型。它利用语义地图抽象来统一和结构化历史信息，以紧凑和一致的形式编码空间上下文。MUVLA将当前和历史观测以及语义地图作为输入，并根据目标物体的描述预测动作序列。此外，它通过基于密集短期进度信号的奖励引导回报建模来放大监督，使模型能够开发详细的动作价值理解以实现奖励最大化。MUVLA采用三阶段训练流程：学习地图级空间理解、从混合质量演示中模仿行为、以及奖励放大。这种策略使MUVLA能够将多样化的演示统一为强大的空间表示，并生成更合理的探索策略。在HM3D和Gibson基准测试上的实验表明，MUVLA实现了良好的泛化能力，即使从低质量或部分成功的轨迹中也能学习有效的探索行为。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决物体导航问题，即如何让智能体在未知环境中自主有效地寻找特定物体。这个问题在现实中非常重要，因为它关系到家庭服务机器人、搜救机器人等自主系统的实用能力；在研究中也很重要，因为它挑战了智能体的感知、记忆和推理能力，且难以定义高质量的数据标准，现有方法要么过度依赖外部模块，要么难以从混合质量的数据中学习有效探索策略。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先区分了视觉-语言导航和物体导航的本质区别，认识到物体导航需要'独立探索'而非'遵循指令'。作者针对两个关键问题进行思考：如何统一表示历史信息，以及如何从混合质量数据中学习有效探索策略。设计上借鉴了现有工作：地图表示借鉴了MapNav，强化学习借鉴了离线RL和决策变换器，推理能力借鉴了大语言模型的链式思维方法，但针对物体导航任务进行了创新整合。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过语义地图抽象统一和结构化历史信息，将空间上下文编码为紧凑形式，使智能体能从混合质量数据中学习有效探索策略。整体流程分三阶段：1)地图理解阶段，构建语义地图并预训练空间理解能力；2)行为克隆阶段，从混合质量演示中模仿导航行为；3)奖励放大阶段，通过奖励引导训练提升探索效率。每阶段冻结不同模块，逐步优化模型能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的语义地图抽象，将冗余探索记忆结构化；2)三阶段训练流程，分别优化空间理解、行为模仿和奖励引导；3)奖励引导的返回建模，使用期望回归评估动作价值。相比之前工作，MUVLA不同于MapNav之处在于增加了奖励建模和链式思维推理；不同于传统方法在于能利用混合质量数据；不同于无需训练方法在于直接集成大语言模型到决策过程；不同于视觉-语言导航方法在于专注于独立探索而非指令跟随。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MUVLA通过语义地图抽象和三阶段训练流程，实现了从混合质量数据中学习高效物体导航策略的能力，显著提升了智能体在未知环境中寻找目标物体的性能和泛化能力。'}


### 论文摘要

In this paper, we present MUVLA, a Map Understanding Vision-Language-Action model tailored for object navigation. It leverages semantic map abstractions to unify and structure historical information, encoding spatial context in a compact and consistent form. MUVLA takes the current and history observations, as well as the semantic map, as inputs and predicts the action sequence based on the description of goal object. Furthermore, it amplifies supervision through reward-guided return modeling based on dense short-horizon progress signals, enabling the model to develop a detailed understanding of action value for reward maximization. MUVLA employs a three-stage training pipeline: learning map-level spatial understanding, imitating behaviors from mixed-quality demonstrations, and reward amplification. This strategy allows MUVLA to unify diverse demonstrations into a robust spatial representation and generate more rational exploration strategies. Experiments on HM3D and Gibson benchmarks demonstrate that MUVLA achieves great generalization and learns effective exploration behaviors even from low-quality or partially successful trajectories.

---

## 47. VLM-FO1: Bridging the Gap Between High-Level Reasoning and Fine-Grained Perception in VLMs

**论文链接:** [http://arxiv.org/abs/2509.25916v1](http://arxiv.org/abs/2509.25916v1)

**作者:** Peng Liu, Haozhan Shen, Chunxin Fang, Zhicheng Sun, Jiajia Liao, Tiancheng Zhao

**发布时间:** 2025-09-30

**备注:** 22 pages

### GPT解析

### 总结

VLM-FO1是一个新型框架，通过将对象感知从坐标生成转变为特征检索，解决了VLMs在细粒度感知任务上的局限性，实现了高性能且不损害基础模型的通用视觉理解能力。

### 背景

Vision-Language Models (VLMs) 在高层场景理解方面表现出色，但在需要精确本地化的细粒度感知任务上表现不佳。这是因为生成精确的数值坐标对以语言为中心的架构来说是一个具有挑战性的任务。

### 目的

引入VLM-FO1框架，通过重新构建面向对象的感知问题，从脆弱的坐标生成任务转变为强大的特征检索任务，克服VLMs在细粒度感知上的局限性。

### 方法

VLM-FO1作为一个即插即用模块，与任何预训练的VLM集成。它利用混合细粒度区域编码器(HFRE)，该编码器具有双视觉编码器，生成同时包含丰富语义和空间细节的区域标记。基于标记的引用系统使大型语言模型能够无缝推理并将语言锚定在这些特定的视觉区域中。

### 主要发现

实验表明，VLM-FO1在多样化的基准测试中取得了最先进的性能，展示了在对象接地、区域生成理解和视觉区域推理方面的卓越能力。两阶段训练策略确保了这些感知能力的提升不会损害基础模型的通用视觉理解能力。

### 结论

VLM-FO1为构建具有感知能力的VLMs建立了一种有效且灵活的范式，弥合了高层推理和细粒度视觉接地之间的差距。

### 翻译

视觉-语言模型(VLMs)擅长高层场景理解，但在需要精确本地化的细粒度感知任务上表现不佳。这种失败源于一个根本性的不匹配，因为生成精确的数值坐标对于以语言为中心的架构来说是一项具有挑战性的任务。在本文中，我们引入了VLM-FO1，一个新型框架，通过将面向对象的感知从脆弱的坐标生成问题重新构建为强大的特征检索任务，从而克服这一限制。我们的方法作为一个即插即用模块，可以与任何预训练的VLM集成。它利用混合细粒度区域编码器(HFRE)，该编码器具有双视觉编码器，生成同时包含丰富语义和空间细节的区域标记。然后，基于标记的引用系统使大型语言模型能够无缝推理并将语言锚定在这些特定的视觉区域中。实验表明，VLM-FO1在多样化的基准测试套件中取得了最先进的性能，展示了在对象接地、区域生成理解和视觉区域推理方面的卓越能力。重要的是，我们的两阶段训练策略确保了这些感知能力的提升不会损害基础模型的通用视觉理解能力。VLM-FO1为构建具有感知能力的VLMs建立了一种有效且灵活的范式，弥合了高层推理和细粒度视觉接地之间的差距。


### 论文摘要

Vision-Language Models (VLMs) excel at high-level scene understanding but falter on fine-grained perception tasks requiring precise localization. This failure stems from a fundamental mismatch, as generating exact numerical coordinates is a challenging task for language-centric architectures. In this paper, we introduce VLM-FO1, a novel framework that overcomes this limitation by reframing object-centric perception from a brittle coordinate generation problem into a robust feature retrieval task. Our method operates as a plug-and-play module that integrates with any pre-trained VLM. It leverages a Hybrid Fine-grained Region Encoder (HFRE), featuring a dual vision encoder, to generate powerful region tokens rich in both semantic and spatial detail. A token-based referencing system then enables the LLM to seamlessly reason about and ground language in these specific visual regions. Experiments show that VLM-FO1 achieves state-of-the-art performance across a diverse suite of benchmarks, demonstrating exceptional capabilities in object grounding, region generational understanding, and visual region reasoning. Crucially, our two-stage training strategy ensures that these perception gains are achieved without compromising the base model's general visual understanding capabilities. VLM-FO1 establishes an effective and flexible paradigm for building perception-aware VLMs, bridging the gap between high-level reasoning and fine-grained visual grounding.

---

## 48. Visual Jigsaw Post-Training Improves MLLMs

**论文链接:** [http://arxiv.org/abs/2509.25190v1](http://arxiv.org/abs/2509.25190v1)

**作者:** Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, Ziwei Liu

**发布时间:** 2025-09-29

### GPT解析

### 总结

研究提出了Visual Jigsaw，一种通用的自监督后训练框架，旨在增强多模态大语言模型的视觉理解能力，通过视觉排序任务实现无需额外视觉生成组件或人工标注的训练。

### 背景

基于强化学习的后训练已成为增强多模态大语言模型对齐和推理能力的强大范式，但当前后训练方法主要是以文本为中心的，密集视觉输入仅被用来提取用于文本推理的稀疏线索，现有方法仍依赖文本作为中间媒介或引入额外视觉生成设计。

### 目的

引入一个通用的自监督后训练框架，加强多模态大语言模型中的视觉理解能力，避免依赖文本中介或额外视觉生成组件。

### 方法

提出Visual Jigsaw框架，形式化为通用排序任务：视觉输入被分割、打乱，模型通过产生正确的自然语言排列来重建视觉信息，符合可验证奖励的强化学习，无需额外视觉生成组件或人工标注，在图像、视频和3D数据三种视觉模态上实例化。

### 主要发现

大量实验证明Visual Jigsaw在细粒度感知、时间推理和3D空间理解方面有显著改进。

### 结论

以视觉为中心的自监督任务在后训练多模态大语言模型中具有巨大潜力，可为未来视觉为中心的前置设计研究提供启发。

### 翻译

基于强化学习的后训练最近 emerged as a powerful paradigm for enhancing the alignment and reasoning capabilities of multimodal large language models (MLLMs). While vision-centric post-training is crucial for enhancing MLLMs' intrinsic understanding of visual signals, current post-training paradigms are predominantly text-centric, where dense visual inputs are only leveraged to extract sparse cues for text-based reasoning. There exist a few approaches in this direction, however, they often still rely on text as an intermediate mediator or introduce additional visual generative designs. In this work, we introduce Visual Jigsaw, a generic self-supervised post-training framework designed to strengthen visual understanding in MLLMs. Visual Jigsaw is formulated as a general ordering task: visual inputs are partitioned, shuffled, and the model must reconstruct the visual information by producing the correct permutation in natural language. This naturally aligns with reinforcement learning from verifiable rewards (RLVR), requires no additional visual generative components, and derives its supervisory signal automatically without any annotations. We instantiate Visual Jigsaw across three visual modalities, including images, videos, and 3D data. Extensive experiments demonstrate substantial improvements in fine-grained perception, temporal reasoning, and 3D spatial understanding. Our findings highlight the potential of self-supervised vision-centric tasks in post-training MLLMs and aim to inspire further research on vision-centric pretext designs. Project Page: https://penghao-wu.github.io/visual_jigsaw/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决多模态大语言模型（MLLMs）在视觉理解方面的问题。当前MLLM的后训练方法过度依赖文本，视觉信息仅被用作提取线索来支持文本推理，缺乏对视觉信号本身的深度理解。这个问题很重要，因为增强视觉理解能力能让AI系统在需要精细视觉感知的任务（如自动驾驶、医疗影像分析、机器人技术）中表现更好，使模型不仅能'看到'，还能真正'理解'视觉内容。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从自监督视觉表示学习的历史中汲取灵感，特别是拼图式任务（如重新排序打乱的图像块、恢复视频帧顺序）。他们认识到这类任务提供了结构排序信号，可作为重建任务的简化版本。作者设计了Visual Jigsaw，将视觉理解转化为排序问题：将视觉输入分割打乱，让模型预测正确顺序。这种方法借鉴了自监督学习中拼图任务的思想和强化学习可验证奖励框架，但创新点在于不需要额外视觉生成组件，与现有文本输出模型无缝集成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视觉理解转化为排序问题：将图像、视频或3D数据分割成若干部分，打乱顺序，让模型通过自然语言预测正确的原始排列。这种方法不需要像素级重建，也不需额外视觉组件。流程包括：1）任务定义（图像分割成块、视频分成片段、3D数据选取不同深度点）；2）奖励设计（完全正确得1分，部分正确按比例打折，无效得0）；3）使用GRPO算法训练模型；4）在多种视觉基准测试上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出Visual Jigsaw框架，轻量级且可验证的自监督任务；2）在图像、视频和3D三种模态上实现通用性；3）设计部分准确性的分级奖励函数。相比之前工作不同：1）不同于文本中心方法，专注于视觉信号本身；2）区别于需额外组件的重建方法，不改变模型架构；3）比现有拼图任务更复杂有效；4）使用强化学习而非监督微调，证明RL在泛化上更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Visual Jigsaw通过自监督的视觉排序任务，在不改变模型架构的情况下，显著提升了多模态大语言模型在图像、视频和3D数据上的细粒度感知、时间理解和空间推理能力。'}


### 论文摘要

Reinforcement learning based post-training has recently emerged as a powerful paradigm for enhancing the alignment and reasoning capabilities of multimodal large language models (MLLMs). While vision-centric post-training is crucial for enhancing MLLMs' intrinsic understanding of visual signals, current post-training paradigms are predominantly text-centric, where dense visual inputs are only leveraged to extract sparse cues for text-based reasoning. There exist a few approaches in this direction, however, they often still rely on text as an intermediate mediator or introduce additional visual generative designs. In this work, we introduce Visual Jigsaw, a generic self-supervised post-training framework designed to strengthen visual understanding in MLLMs. Visual Jigsaw is formulated as a general ordering task: visual inputs are partitioned, shuffled, and the model must reconstruct the visual information by producing the correct permutation in natural language. This naturally aligns with reinforcement learning from verifiable rewards (RLVR), requires no additional visual generative components, and derives its supervisory signal automatically without any annotations. We instantiate Visual Jigsaw across three visual modalities, including images, videos, and 3D data. Extensive experiments demonstrate substantial improvements in fine-grained perception, temporal reasoning, and 3D spatial understanding. Our findings highlight the potential of self-supervised vision-centric tasks in post-training MLLMs and aim to inspire further research on vision-centric pretext designs. Project Page: https://penghao-wu.github.io/visual_jigsaw/

---

## 49. PAD3R: Pose-Aware Dynamic 3D Reconstruction from Casual Videos

**论文链接:** [http://arxiv.org/abs/2509.25183v1](http://arxiv.org/abs/2509.25183v1)

**作者:** Ting-Hsuan Liao, Haowen Liu, Yiran Xu, Songwei Ge, Gengshan Yang, Jia-Bin Huang

**发布时间:** 2025-09-29

**备注:** SIGGRAPH Asia 2025. Project page:https://pad3r.github.io/

### GPT解析

### 总结

PAD3R是一种从 casually captured、未摆位的单目视频中重建可变形3D物体的方法，能够处理长视频序列、大幅物体变形、大范围相机移动和有限视角覆盖等挑战场景。

### 背景

现有方法难以处理长视频序列中出现的显著物体变形、大范围相机移动和有限视角覆盖等问题，这些因素通常对传统系统构成挑战。

### 目的

开发一种能够从 casually captured、未摆位的单目视频中重建可变形3D物体的方法，解决现有方法在处理复杂场景时的局限性。

### 方法

PAD3R的核心是训练一个个性化的物体中心姿态估计器，由预训练的图像到3D模型监督，指导可变形3D高斯表示的优化。优化过程通过整个输入视频的长期2D点跟踪进行正则化，结合生成先验和可微渲染技术，以类别无关的方式重建高保真度的关节式3D表示。

### 主要发现

通过大量的定性和定量结果表明，PAD3R具有鲁棒性，在具有挑战性的场景中泛化能力良好。

### 结论

PAD3R在动态场景理解和3D内容创作方面具有潜力，能够有效处理复杂场景下的3D重建问题。

### 翻译

我们提出了PAD3R，一种从 casually captured、未摆位的单目视频中重建可变形3D物体的方法。与现有方法不同，PAD3R能够处理长视频序列，其中包含显著的物体变形、大范围的相机移动和有限的视角覆盖，这些因素通常对传统系统构成挑战。其核心方法是训练一个个性化的、物体中心的姿态估计器，由预训练的图像到3D模型监督，指导可变形3D高斯表示的优化。优化过程通过整个输入视频的长期2D点跟踪进行进一步正则化。通过结合生成先验和可微渲染，PAD3R能够以类别无关的方式重建物体的高保真度、关节式3D表示。大量的定性和定量结果表明，PAD3R具有鲁棒性，并在具有挑战性的场景中泛化良好，突显了其在动态场景理解和3D内容创作方面的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从普通拍摄的单目视频中重建动态、可变形的3D对象的问题。这个问题在现实和研究中的重要性在于：1) 游戏开发、电影制作、增强现实/虚拟现实和机器人等领域都需要这种技术；2) 传统方法依赖专门传感器或特定类别模型，限制了在多样化真实环境中的应用；3) 从单目视频中重建动态3D对象是病态问题，存在多种可能的3D解释；4) 现有技术在处理复杂相机运动和对象变形时存在局限性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者设计这个方法时借鉴了多种现有技术：1) 使用图像到3D模型从关键帧生成静态3D表示；2) 基于DINO-v2框架训练个性化的对象中心相机姿态估计器；3) 采用变形3D高斯溅射技术建模动态场景；4) 使用SuGaR方法的混合高斯表示增强表面建模；5) 利用2D点跟踪提供运动线索。作者的创新在于将这些技术有机结合，分两个阶段解决问题：首先是对象中心相机姿态初始化，然后是动态高斯溅射重建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 利用生成扩散先验恢复静态3D表示，同时使用可微分渲染估计以对象为中心的相机姿态和时间变化的变形；2) 训练个性化的相机姿态估计器为后续4D重建提供关键初始化；3) 使用双向多块跟踪策略有效利用2D点跟踪的长期运动线索来正则化对象变形。整体流程分两阶段：第一阶段从视频中选关键帧，用图像到3D方法生成静态3D模型，训练PoseNet预测相机姿态；第二阶段初始化混合3D高斯表示，用神经皮肤变形模型建模动态运动，通过双向多块跟踪提供监督，最后优化变形网络和姿态网络。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有：1) 利用图像到3D模型学习对象中心相机姿态，为4D重建提供关键初始化；2) 引入多块策略有效利用2D点跟踪的长期运动线索正则化对象变形；3) 在真实世界和合成视频数据集上广泛实验，证明方法优于最先进技术。相比之前工作：1) 不依赖专门传感器或特定类别模型，适用于多样化对象；2) 不需要精确相机姿态和密集视角覆盖；3) 能处理真实世界视频中的复杂相机轨迹和动态对象运动；4) 不需要高质量多视图训练数据，能推广到分布外的真实世界视频。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAD3R提出了一种新颖的两阶段方法，通过结合生成先验和可微分渲染，实现了从普通视频中高质量重建动态3D对象，解决了现有方法在处理复杂相机运动和对象变形时的局限性。'}


### 论文摘要

We present PAD3R, a method for reconstructing deformable 3D objects from casually captured, unposed monocular videos. Unlike existing approaches, PAD3R handles long video sequences featuring substantial object deformation, large-scale camera movement, and limited view coverage that typically challenge conventional systems. At its core, our approach trains a personalized, object-centric pose estimator, supervised by a pre-trained image-to-3D model. This guides the optimization of deformable 3D Gaussian representation. The optimization is further regularized by long-term 2D point tracking over the entire input video. By combining generative priors and differentiable rendering, PAD3R reconstructs high-fidelity, articulated 3D representations of objects in a category-agnostic way. Extensive qualitative and quantitative results show that PAD3R is robust and generalizes well across challenging scenarios, highlighting its potential for dynamic scene understanding and 3D content creation.

---

## 50. GeoSketch: A Neural-Symbolic Approach to Geometric Multimodal Reasoning with Auxiliary Line Construction and Affine Transformation

**论文链接:** [http://arxiv.org/abs/2509.22460v2](http://arxiv.org/abs/2509.22460v2)

**作者:** Shichao Weng, Zhiqiang Wang, Yuhua Zhou, Rui Lu, Ting Liu, Zhiyang Teng, Xiaozhang Liu, Hanmeng Liu

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了GeoSketch，一个神经符号框架，将几何问题解决转化为交互式感知-推理-行动循环，包含三个模块：感知模块将图表抽象为结构化逻辑形式，符号推理模块应用几何定理决定演绎步骤，草图动作模块执行辅助线绘制等操作。通过监督微调和强化学习的两阶段训练方法，GeoSketch显著提高了几何问题解决能力。

### 背景

几何问题解决对多模态大语言模型提出了独特挑战，需要联合解释文本和图表，以及迭代的空间推理。现有方法将图表处理为静态图像，缺乏动态操作能力，而这正是人类几何推理的核心方面，包括辅助线构建和仿射变换。

### 目的

开发一个能够进行动态交互的几何问题解决框架，使模型能够像人类一样进行辅助线构建和仿射变换等操作，从而提高几何问题解决的准确性和成功率。

### 方法

GeoSketch是一个神经符号框架，包含三个主要模块：感知模块将图表抽象为结构化逻辑形式；符号推理模块应用几何定理决定下一个演绎步骤；草图动作模块执行绘制辅助线或应用变换等操作，形成闭环更新图表。训练采用两阶段流程：首先在2000个符号化整理的轨迹上进行监督微调，然后使用密集的符号奖励进行强化学习，以提高鲁棒性和策略探索。

### 主要发现

作者引入了GeoSketch基准测试，包含390个需要辅助构建或仿射变换的高质量几何问题。实验表明，与静态感知方法相比，GeoSketch显著提高了逐步推理准确性和问题解决成功率。

### 结论

GeoSketch通过统一分层决策、可执行视觉动作和符号验证，将多模态推理从静态解释推进到动态、可验证的交互，为解决复杂空间问题建立了新的基础。

### 翻译

几何问题解决（GPS）对多模态大语言模型（MLLMs）提出了独特挑战，不仅需要联合解释文本和图表，还需要迭代的空间推理。虽然现有方法将图表处理为静态图像，但它们缺乏动态操作能力——这是人类几何推理的核心方面，涉及辅助线构建和仿射变换。我们提出了GeoSketch，一个神经符号框架，将几何推理重新构建为交互式的感知-推理-行动循环。GeoSketch整合了：(1) 将图表抽象为结构化逻辑形式的感知模块，(2) 应用几何定理决定下一个演绎步骤的符号推理模块，以及(3) 执行绘制辅助线或应用变换等操作的草图动作模块，从而形成闭环更新图表。为了训练这个智能体，我们开发了一个两阶段流程：首先在2000个符号化整理的轨迹上进行监督微调，然后使用密集的符号奖励进行强化学习，以提高鲁壮性和策略探索。为了评估这一范式，我们引入了GeoSketch基准测试，这是一个包含390个需要辅助构建或仿射变换的高质量几何问题集合。在强大的MLLM基线上的实验表明，GeoSketch显著提高了逐步推理准确性和问题解决成功率，超越了静态感知方法。通过统一分层决策、可执行视觉动作和符号验证，GeoSketch将多模态推理从静态推进到动态、可验证的交互，为解决复杂空间问题建立了新的基础。


### 论文摘要

Geometric Problem Solving (GPS) poses a unique challenge for Multimodal Large Language Models (MLLMs), requiring not only the joint interpretation of text and diagrams but also iterative visuospatial reasoning. While existing approaches process diagrams as static images, they lack the capacity for dynamic manipulation - a core aspect of human geometric reasoning involving auxiliary line construction and affine transformations. We present GeoSketch, a neural-symbolic framework that recasts geometric reasoning as an interactive perception-reasoning-action loop. GeoSketch integrates: (1) a Perception module that abstracts diagrams into structured logic forms, (2) a Symbolic Reasoning module that applies geometric theorems to decide the next deductive step, and (3) a Sketch Action module that executes operations such as drawing auxiliary lines or applying transformations, thereby updating the diagram in a closed loop. To train this agent, we develop a two-stage pipeline: supervised fine-tuning on 2,000 symbolic-curated trajectories followed by reinforcement learning with dense, symbolic rewards to enhance robustness and strategic exploration. To evaluate this paradigm, we introduce the GeoSketch Benchmark, a high-quality set of 390 geometry problems requiring auxiliary construction or affine transformations. Experiments on strong MLLM baselines demonstrate that GeoSketch significantly improves stepwise reasoning accuracy and problem-solving success over static perception methods. By unifying hierarchical decision-making, executable visual actions, and symbolic verification, GeoSketch advances multimodal reasoning from static interpretation to dynamic, verifiable interaction, establishing a new foundation for solving complex visuospatial problems.

---

## 51. Leveraging Scene Context with Dual Networks for Sequential User Behavior Modeling

**论文链接:** [http://arxiv.org/abs/2509.26172v1](http://arxiv.org/abs/2509.26172v1)

**作者:** Xu Chen, Yunmeng Shu, Yuangang Pan, Jinsong Lan, Xiaoyong Zhu, Shuai Xiao, Haojin Zhu, Ivor W. Tsang, Bo Zheng

**发布时间:** 2025-09-30

**备注:** 12pages

### GPT解析

### 总结

本文提出了一种双序列预测网络(DSPnet)用于用户未来行为预测，通过捕捉用户对物品和场景的动态兴趣以及它们之间的相互作用，有效提高了预测准确性。该方法已在实际系统中部署并带来了业务指标的提升。

### 背景

建模用户序列行为对改善信息检索体验至关重要。场景特征作为应用内为提供特定功能而创建的子界面，是一个重要但常被忽视的上下文信息。不同场景具有不同功能和用户习惯，导致用户参与度存在显著差异。现有模型要么忽略场景特征，要么仅将其用作属性嵌入，无法有效捕捉动态兴趣和场景与物品间的相互作用。

### 目的

开发一种能够有效捕捉用户对物品和场景的动态兴趣以及它们之间相互作用的模型，以提高未来行为预测的准确性。

### 方法

提出双序列预测网络(DSPnet)，包括：1)两个并行网络分别学习用户对物品和场景的动态兴趣；2)序列特征增强模块捕捉相互作用以增强预测；3)条件对比正则化(CCR)损失函数捕获相似历史序列的不变性。

### 主要发现

理论分析表明DSPnet是学习场景和物品序列间联合关系的原则性方法。在公共基准和两个工业数据集上的实验验证了其有效性。该方法在线部署后带来CTR增加0.04个百分点，交易增长0.78%，GMV增长0.64%。

### 结论

DSPnet有效解决了现有模型在利用场景特征进行用户序列行为预测方面的不足，通过捕捉用户对物品和场景的动态兴趣及它们之间的相互作用，显著提高了预测性能，并在实际应用中带来了可衡量的业务收益。

### 翻译

为未来行为预测建模用户序列行为对于改善用户的信息检索体验至关重要。最近的研究强调整合上下文信息以提高预测性能的重要性。一个重要但通常被忽视的上下文信息是场景特征，我们将其定义为应用内的子界面，由开发者创建以提供特定功能，例如电子商务应用中的'text2product search'和'live'模块。不同场景表现出不同的功能和用户习惯，导致用户参与度在场景间存在显著差异。流行的序列行为模型要么忽略场景特征，要么仅将其用作属性嵌入，这在建模用户序列时无法有效捕捉动态兴趣以及场景与物品之间的相互作用。在这项工作中，我们提出了一个新颖的双序列预测网络(DSPnet)，以有效捕捉场景和物品之间的动态兴趣和相互作用，用于未来行为预测。DSPnet包含两个并行网络，专门用于学习用户对物品和场景的动态兴趣，以及一个序列特征增强模块，用于捕捉相互作用以增强未来行为预测。此外，我们引入了一种条件对比正则化(CCR)损失函数，以捕获相似历史序列的不变性。理论分析表明，DSPnet是学习场景和物品序列之间联合关系的原则性方法。在一个公共基准和两个收集的工业数据集上进行了广泛的实验。该方法已在我们的系统中在线部署，带来了CTR增加0.04个百分点，交易增长0.78%，GMV增长0.64%。代码可在匿名github上获取：https://anonymous.4open.science/r/DSPNet-ForPublish-2506/。


### 论文摘要

Modeling sequential user behaviors for future behavior prediction is crucial in improving user's information retrieval experience. Recent studies highlight the importance of incorporating contextual information to enhance prediction performance. One crucial but usually neglected contextual information is the scene feature which we define as sub-interfaces within an app, created by developers to provide specific functionalities, such as ``text2product search" and ``live" modules in e-commence apps. Different scenes exhibit distinct functionalities and usage habits, leading to significant distribution gap in user engagement across them. Popular sequential behavior models either ignore the scene feature or merely use it as attribute embeddings, which cannot effectively capture the dynamic interests and interplay between scenes and items when modeling user sequences. In this work, we propose a novel Dual Sequence Prediction networks (DSPnet) to effectively capture the dynamic interests and interplay between scenes and items for future behavior prediction. DSPnet consists of two parallel networks dedicated to learn users' dynamic interests over items and scenes, and a sequence feature enhancement module to capture the interplay for enhanced future behavior prediction. Further, we introduce a Conditional Contrastive Regularization (CCR) loss to capture the invariance of similar historical sequences. Theoretical analysis suggests that DSPnet is a principled way to learn the joint relationships between scene and item sequences. Extensive experiments are conducted on one public benchmark and two collected industrial datasets. The method has been deployed online in our system, bringing a 0.04 point increase in CTR, 0.78\% growth in deals, and 0.64\% rise in GMV. The codes are available at this anonymous github: \textcolor{blue}{https://anonymous.4open.science/r/DSPNet-ForPublish-2506/}.

---

## 52. Physics-Informed Learning for Human Whole-Body Kinematics Prediction via Sparse IMUs

**论文链接:** [http://arxiv.org/abs/2509.25704v1](http://arxiv.org/abs/2509.25704v1)

**作者:** Cheng Guo, Giuseppe L'Erario, Giulio Romualdi, Mattia Leonori, Marta Lorenzini, Arash Ajoudani, Daniele Pucci

**发布时间:** 2025-09-30

### GPT解析

### 总结

该研究提出了一种物理信息学习框架，仅使用5个惯性测量单元(IMU)来预测人类运动，整合领域知识到训练和推理中，实现了高精度、平滑过渡和良好泛化能力的人类运动预测。

### 背景

准确且物理可行的人类运动预测对安全无缝的人机协作至关重要。虽然最近的人体运动捕捉技术能实现实时姿态估计，但许多现有方法的实际价值因缺乏未来预测和未考虑物理约束而受限。传统运动预测方案严重依赖过去姿态，而这些姿态在现实场景中并不总是可用。

### 目的

解决现有人类运动预测方法的局限性，开发一种仅使用5个IMU进行人类运动预测的物理信息学习框架。

### 方法

提出了一种考虑人类运动空间特性的网络。训练过程中，将正向运动学和微分运动学函数作为额外损失组件纳入，以规范学习到的关节预测。推理阶段，通过细化前一次迭代的预测来更新关节状态缓冲区，该缓冲区用作网络的额外输入。

### 主要发现

实验结果表明，该方法实现了高精度、运动间的平滑过渡，并且对未见过的受试者具有良好的泛化能力。

### 结论

所提出的物理信息学习框架能够有效整合领域知识，仅使用5个IMU实现准确、物理可行的人类运动预测，为安全人机协作提供了实用解决方案。

### 翻译

准确且物理可行的人类运动预测对安全无缝的人机协作至关重要。虽然最近的人体运动捕捉技术的进步能够实现实时姿态估计，但许多现有方法的实际价值因缺乏未来预测和未考虑物理约束而受到限制。传统的运动预测方案严重依赖过去姿态，而这些姿态在现实场景中并不总是可用。为解决这些局限性，我们提出了一种物理信息学习框架，将领域知识整合到训练和推理中，仅使用5个IMU的惯性测量来预测人类运动。我们提出了一种考虑人类运动空间特性的网络。在训练过程中，我们将正向运动学和微分运动学函数作为额外的损失组件纳入，以规范学习到的关节预测。在推理阶段，我们通过细化前一次迭代的预测来更新关节状态缓冲区，该缓冲区用作网络的额外输入。实验结果表明，我们的方法实现了高精度、运动间的平滑过渡，并且对未见过的受试者具有良好的泛化能力。


### 论文摘要

Accurate and physically feasible human motion prediction is crucial for safe and seamless human-robot collaboration. While recent advancements in human motion capture enable real-time pose estimation, the practical value of many existing approaches is limited by the lack of future predictions and consideration of physical constraints. Conventional motion prediction schemes rely heavily on past poses, which are not always available in real-world scenarios. To address these limitations, we present a physics-informed learning framework that integrates domain knowledge into both training and inference to predict human motion using inertial measurements from only 5 IMUs. We propose a network that accounts for the spatial characteristics of human movements. During training, we incorporate forward and differential kinematics functions as additional loss components to regularize the learned joint predictions. At the inference stage, we refine the prediction from the previous iteration to update a joint state buffer, which is used as extra inputs to the network. Experimental results demonstrate that our approach achieves high accuracy, smooth transitions between motions, and generalizes well to unseen subjects

---

## 53. AccidentBench: Benchmarking Multimodal Understanding and Reasoning in Vehicle Accidents and Beyond

**论文链接:** [http://arxiv.org/abs/2509.26636v1](http://arxiv.org/abs/2509.26636v1)

**作者:** Shangding Gu, Xiaohan Wang, Donghao Ying, Haoyu Zhao, Runing Yang, Ming Jin, Boyi Li, Marco Pavone, Serena Yeung-Levy, Jun Wang, Dawn Song, Costas Spanos

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文介绍了AccidentBench，一个大规模基准测试，用于评估多模态模型在安全关键、动态现实世界场景中的理解和推理能力。该基准测试结合了车辆事故场景和航空、水域等安全关键场景，包含约2000个视频和超过19000个人类标注的问题-答案对。

### 背景

多模态模型正在快速发展，需要严格的基准测试来评估它们在安全关键、动态现实世界场景中的理解和推理能力。

### 目的

创建一个全面的、基于物理的测试平台，用于评估模型在现实世界变化条件下的表现，并揭示当前最先进模型在现实世界时间、空间和意图推理方面的差距。

### 方法

构建了AccidentBench基准测试，包含约2000个视频和超过19000个人类标注的问题-答案对，涵盖多种视频长度（短/中/长）和难度级别（简单/中等/困难）。任务系统地探索核心能力：时间、空间和意图理解和推理。

### 主要发现

对最先进模型（如Gemini-2.5 Pro和GPT-5）的评估显示，即使在最困难的任务和最长的视频上，最强的模型也只能达到约18%的准确率，揭示了在现实世界时间、空间和意图推理方面存在显著差距。

### 结论

AccidentBench旨在揭示这些关键差距，并推动开发更安全、更强大、更好地与现实世界安全关键挑战保持一致的多模态模型。

### 翻译

多模态模型的快速发展需要严格评估安全关键、动态现实世界场景中理解和推理能力的基准测试。我们提出了AccidentBench，这是一个大规模基准测试，将车辆事故场景与航空和水域等安全关键领域相结合，这些领域强调空间和时间推理（如导航、方向、多车辆运动）。该基准测试包含约2000个视频和超过19000个人类标注的问题-答案对，涵盖多种视频长度（短/中/长）和难度级别（简单/中等/困难）。任务系统地探索核心能力：时间、空间和意图理解和推理。通过统一以事故为中心的交通场景与航空和水域中更广泛的安全关键场景，AccidentBench提供了一个全面的、基于物理的测试平台，用于评估模型在现实世界变化条件下的表现。对最先进模型（如Gemini-2.5 Pro和GPT-5）的评估显示，即使在最困难的任务和最长的视频上，最强的模型也只能达到约18%的准确率，揭示了在现实世界时间、空间和意图推理方面存在显著差距。AccidentBench旨在揭示这些关键差距，并推动开发更安全、更强大、更好地与现实世界安全关键挑战保持一致的多模态模型。代码和数据集可在以下网址获取：https://github.com/SafeRL-Lab/AccidentBench


### 论文摘要

Rapid advances in multimodal models demand benchmarks that rigorously evaluate understanding and reasoning in safety-critical, dynamic real-world settings. We present AccidentBench, a large-scale benchmark that combines vehicle accident scenarios with Beyond domains, safety-critical settings in air and water that emphasize spatial and temporal reasoning (e.g., navigation, orientation, multi-vehicle motion). The benchmark contains approximately 2000 videos and over 19000 human-annotated question--answer pairs spanning multiple video lengths (short/medium/long) and difficulty levels (easy/medium/hard). Tasks systematically probe core capabilities: temporal, spatial, and intent understanding and reasoning. By unifying accident-centric traffic scenes with broader safety-critical scenarios in air and water, AccidentBench offers a comprehensive, physically grounded testbed for evaluating models under real-world variability. Evaluations of state-of-the-art models (e.g., Gemini-2.5 Pro and GPT-5) show that even the strongest models achieve only about 18% accuracy on the hardest tasks and longest videos, revealing substantial gaps in real-world temporal, spatial, and intent reasoning. AccidentBench is designed to expose these critical gaps and drive the development of multimodal models that are safer, more robust, and better aligned with real-world safety-critical challenges. The code and dataset are available at: https://github.com/SafeRL-Lab/AccidentBench

---

## 54. Optimizing Indoor Environmental Quality in Smart Buildings Using Deep Learning

**论文链接:** [http://arxiv.org/abs/2509.26187v1](http://arxiv.org/abs/2509.26187v1)

**作者:** Youssef Sabiri, Walid Houmaidi, Aaya Bougrine, Salmane El Mansour Billah

**发布时间:** 2025-09-30

**备注:** 10 pages, 4 figures, 1 table. Accepted and presented at the 5th  International Conference on Digital Technologies and Applications (ICDTA  2025), April 17-18, 2025, Al Akhawayn University, Ifrane, Morocco

### GPT解析

### 总结

本研究提出了一种深度学习方法来主动管理室内环境质量参数，同时平衡建筑能源效率，通过比较不同深度学习架构的预测性能，为智能建筑管理系统提供可行的见解。

### 背景

确保最佳室内环境质量对居住者的健康和生产率至关重要，但传统空调系统往往需要高能源成本。

### 目的

提出一种深度学习方法来主动管理室内环境质量参数(特别是CO2浓度、温度和湿度)，同时平衡建筑能源效率。

### 方法

利用从净零能耗学术建筑收集的ROBOD数据集，对比了三种架构(LSTM、GRU和混合CNN-LSTM)在不同时间范围内预测室内环境质量变量的能力。

### 主要发现

GRU在短期预测准确性方面表现最佳，计算开销较低；CNN-LSTM在提取主导特征方面表现出色，适用于扩展预测窗口；LSTM提供强大的长程时间建模能力；预测可靠性取决于数据分辨率、传感器放置和变化的占用条件。

### 结论

这些发现为智能建筑管理系统实施预测性HVAC控制提供了可行的见解，从而减少能源消耗并提高实际建筑运营中居住者的舒适度。

### 翻译

确保最佳的室内环境质量对居住者的健康和生产率至关重要，但传统的供暖、通风和空调系统往往需要高昂的能源成本。本文提出了一种深度学习方法来主动管理室内环境质量参数，特别是二氧化碳浓度、温度和湿度，同时平衡建筑能源效率。利用从净零能耗学术建筑收集的ROBOD数据集，我们比较了三种架构——长短期记忆网络、门控循环单元和混合卷积神经网络长短期记忆网络——在不同时间范围内预测室内环境质量变量的能力。我们的结果表明，门控循环单元在短期预测准确性方面取得了最佳效果，计算开销较低，而混合卷积神经网络长短期记忆网络在提取主导特征方面表现出色，适用于扩展预测窗口。同时，长短期记忆网络提供了强大的长程时间建模能力。比较分析表明，预测可靠性取决于数据分辨率、传感器放置和变化的占用条件。这些发现为智能建筑管理系统实施预测性暖通空调控制提供了可行的见解，从而减少能源消耗并提高实际建筑运营中居住者的舒适度。


### 论文摘要

Ensuring optimal Indoor Environmental Quality (IEQ) is vital for occupant health and productivity, yet it often comes at a high energy cost in conventional Heating, Ventilation, and Air Conditioning (HVAC) systems. This paper proposes a deep learning driven approach to proactively manage IEQ parameters specifically CO2 concentration, temperature, and humidity while balancing building energy efficiency. Leveraging the ROBOD dataset collected from a net-zero energy academic building, we benchmark three architectures--Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and a hybrid Convolutional Neural Network LSTM (CNN-LSTM)--to forecast IEQ variables across various time horizons. Our results show that GRU achieves the best short-term prediction accuracy with lower computational overhead, whereas CNN-LSTM excels in extracting dominant features for extended forecasting windows. Meanwhile, LSTM offers robust long-range temporal modeling. The comparative analysis highlights that prediction reliability depends on data resolution, sensor placement, and fluctuating occupancy conditions. These findings provide actionable insights for intelligent Building Management Systems (BMS) to implement predictive HVAC control, thereby reducing energy consumption and enhancing occupant comfort in real-world building operations.

---

## 55. NuRisk: A Visual Question Answering Dataset for Agent-Level Risk Assessment in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2509.25944v1](http://arxiv.org/abs/2509.25944v1)

**作者:** Yuan Gao, Mattia Piccinini, Roberto Brusnicki, Yuchen Zhang, Johannes Betz

**发布时间:** 2025-09-30

**备注:** 8 pages

### GPT解析

### 总结

该研究提出了NuRisk数据集用于自动驾驶中的时空风险评估，发现现有视觉语言模型表现不佳，而微调后的7B VLM代理模型显著提高了准确率并降低了延迟。

### 背景

理解自动驾驶风险需要感知、预测以及关于代理行为和上下文的高级推理。当前基于视觉语言模型的方法主要基于静态图像，缺乏捕捉风险随时间演化的时空推理能力。

### 目的

解决现有方法在时空推理方面的不足，提供一个全面的视觉问答数据集，支持自动驾驶中的时空风险评估。

### 方法

构建了NuRisk数据集，包含2,900个场景和110万个代理级别样本，基于nuScenes和Waymo的真实世界数据并补充CommonRoad模拟器的安全关键场景；提供基于鸟瞰视图的序列图像和定量风险标注；对知名VLMs进行基准测试；微调7B VLM代理模型。

### 主要发现

现有VLMs无法执行明确的时空推理，峰值准确率仅为33%；微调后的7B VLM代理模型将准确率提高到41%，延迟减少75%；展示了专有模型所缺乏的时空推理能力。

### 结论

尽管取得了显著进展，但适度的准确率凸显了任务的深刻挑战，NuRisk成为推进自动驾驶中时空推理的关键基准。

### 翻译

理解自动驾驶风险不仅需要感知和预测，还需要对代理行为和上下文进行高级推理。当前基于视觉语言模型的方法主要将代理定位在静态图像中并提供定性判断，缺乏捕捉风险如何随时间演化的时空推理能力。为解决这一差距，我们提出了NuRisk，一个全面的视觉问答数据集，包含2,900个场景和110万个代理级别样本，基于nuScenes和Waymo的真实世界数据构建，并补充了CommonRoad模拟器的安全关键场景。该数据集提供了基于鸟瞰视图的序列图像，带有定量、代理级别的风险标注，使时空推理成为可能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶中缺乏智能体级定量风险评估能力的问题，特别是时空推理来捕捉风险如何随时间演变。这个问题重要是因为自动驾驶需要高级风险推理能力来处理罕见安全关键情况，而现有方法仅提供定性判断，无法准确计算像时间到碰撞(TTC)和距离到碰撞(DTC)这样的安全指标，导致运动规划器只能做出保守决策而非精确判断。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有VLMs在自动驾驶风险评估中的局限性，包括时空推理不足、缺乏定量标签和安全关键场景覆盖不足。他们借鉴了nuScenes和Waymo真实世界数据集，使用CommonRoad模拟器生成安全关键场景，参考了LLaVA对话格式构建VQA数据集，并采用LoRA方法进行参数高效微调。整体设计思路是构建综合VQA数据集，评估现有模型表现，然后通过微调改进时空推理能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建专门用于自动驾驶智能体级定量风险评估的VQA数据集，评估现有模型表现，并通过微调实现显式时空推理。流程包括：1)从三个来源收集数据并多阶段处理(数据提取、BEV图像生成、风险标注)；2)创建VQA数据集(预处理、对话格式转换、质量保证)；3)模型评估与微调(基准测试、物理增强分析、LoRA微调VLM代理)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)NuRisk数据集-首个专注于智能体级定量时空风险评估的VQA数据集；2)系统化评估框架-揭示现有模型在时空推理方面的不足；3)微调VLM代理-实现显式时空推理。相比之前工作，NuRisk填补了现有VQA数据集缺乏定量风险评估和时空推理的空白，超越了基于静态图像的定性判断方法，实现了专门针对自动驾驶风险任务的优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了NuRisk，首个专注于自动驾驶智能体级定量时空风险评估的VQA数据集，并通过微调VLM实现了显式时空推理能力，显著提升了风险评估的准确性和效率。'}


### 论文摘要

Understanding risk in autonomous driving requires not only perception and prediction, but also high-level reasoning about agent behavior and context. Current Vision Language Models (VLMs)-based methods primarily ground agents in static images and provide qualitative judgments, lacking the spatio-temporal reasoning needed to capture how risks evolve over time. To address this gap, we propose NuRisk, a comprehensive Visual Question Answering (VQA) dataset comprising 2,900 scenarios and 1.1 million agent-level samples, built on real-world data from nuScenes and Waymo, supplemented with safety-critical scenarios from the CommonRoad simulator. The dataset provides Bird-Eye-View (BEV) based sequential images with quantitative, agent-level risk annotations, enabling spatio-temporal reasoning. We benchmark well-known VLMs across different prompting techniques and find that they fail to perform explicit spatio-temporal reasoning, resulting in a peak accuracy of 33% at high latency. To address these shortcomings, our fine-tuned 7B VLM agent improves accuracy to 41% and reduces latency by 75%, demonstrating explicit spatio-temporal reasoning capabilities that proprietary models lacked. While this represents a significant step forward, the modest accuracy underscores the profound challenge of the task, establishing NuRisk as a critical benchmark for advancing spatio-temporal reasoning in autonomous driving.

---

## 56. V-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs

**论文链接:** [http://arxiv.org/abs/2509.25773v1](http://arxiv.org/abs/2509.25773v1)

**作者:** Zhengpeng Shi, Hengli Li, Yanpeng Zhao, Jianqun Zhou, Yuxuan Wang, Qinrong Cui, Wei Bi, Songchun Zhu, Bo Zhao, Zilong Zheng

**发布时间:** 2025-09-30

### GPT解析

### 总结

这篇论文介绍了v-HUB，一个用于评估多模态大语言模型理解幽默能力的视频幽默理解基准。研究团队收集了最少语言的短视频，配有丰富注释，并评估了多种模型，发现仅通过视觉线索理解幽默存在困难，但加入音频有助于提高理解能力。

### 背景

AI模型理解幽默能力具有现实意义，可以增强人机交互的参与度。然而，多模态大语言模型在仅通过视觉线索理解幽默方面的能力尚未得到充分评估。

### 目的

创建并评估v-HUB基准，用于评估和诊断多模态大语言模型在理解视觉幽默方面的能力。

### 方法

研究团队构建了v-HUB基准，包含从经典无声电影和在线资源中收集的最少语言短视频，每个视频片段配有字幕、描述和解释等丰富注释。他们还构建了一个开放式的视频问答任务，使其能够轻松集成到现有的视频理解基准中。研究团队评估了从专业Video-LLMs到能够处理音频的多功能OmniLLMs等多种模型。

### 主要发现

实验结果表明，多模态大语言模型在仅通过视觉线索理解幽默方面面临困难。例如，所有模型在从基于文本评估迁移到基于视频评估（无音频）时，字幕匹配任务表现明显下降。研究还表明，加入音频有助于视频幽默理解，突显了声音信息的重要性，以及为复杂视频理解任务整合更丰富模态的潜力。

### 结论

多模态大语言模型在仅通过视觉线索理解幽默方面存在局限性，但整合音频等多模态信息可以显著提高理解能力，为未来复杂视频理解任务提供了方向。

### 翻译

能够理解幽默的AI模型具有现实世界的应用前景——例如，增强人机交互的参与度。为了评估和诊断多模态大语言模型理解幽默的能力，我们引入了v-HUB，一种新颖的视觉中心视频幽默理解基准。v-HUB包含一个精心策划的最少语言短视频集合，源自经典无声电影和在线资源，反映了仅通过视觉线索就能欣赏幽默的真实世界场景。每个视频片段都配有丰富的注释，包括字幕、描述和解释，支持字幕匹配和幽默解释等评估任务。为了扩大其适用性，我们进一步构建了一个开放式的视频问答任务，使其能够轻松集成到现有的视频理解基准中。我们评估了多样化的MLLMs集合，从专业的Video-LLMs到能够处理音频的多功能OmniLLMs，涵盖了开源和专有领域。实验结果暴露了MLLMs在仅通过视觉线索理解幽默方面面临的困难。例如，所有模型在从基于文本评估迁移到基于视频评估（无音频）时，字幕匹配任务表现明显下降。我们的研究结果还表明，加入音频有助于视频幽默理解，突显了声音信息的重要性以及为复杂视频理解任务整合更丰富模态的潜力。


### 论文摘要

AI models capable of comprehending humor hold real-world promise -- for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel visual-centric video humor understanding benchmark. v-HUB comprises a curated collection of minimally verbal short videos, sourced from classic silent films and online resources, and reflecting real-world scenarios where humor can be appreciated purely through visual cues. Each video clip is paired with rich annotations, including captions, descriptions, and explanations, supporting evaluation tasks like caption matching and humor explanation. To broaden its applicability, we further construct an open-ended video QA task, making it readily integrable into existing video understanding benchmarks. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. For example, all models exhibit a marked performance drop on caption matching when moving from text-based to video-based evaluation (without audio). Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the informativeness of sound and the promise of integrating richer modalities for complex video understanding tasks.

---

## 57. FinCap: Topic-Aligned Captions for Short-Form Financial YouTube Videos

**论文链接:** [http://arxiv.org/abs/2509.25745v1](http://arxiv.org/abs/2509.25745v1)

**作者:** Siddhant Sukhani, Yash Bhardwaj, Riya Bhadani, Veer Kejriwal, Michael Galarnyk, Sudheer Chava

**发布时间:** 2025-09-30

**备注:** ICCV Short Video Understanding Workshop Paper

### GPT解析

### 总结

该研究评估了多模态大语言模型在金融短视频主题对齐字幕生成中的表现，通过测试转录文本、音频和视频的联合推理能力。

### 背景

金融短视频包含多种主题内容，如主要推荐、情感分析、视频目的、视觉分析和金融实体识别。

### 目的

评估多模态大语言模型在金融短视频主题对齐字幕生成中的性能，确定最佳模态组合。

### 方法

使用624个标注的YouTube短视频，评估七种模态组合（T, A, V, TA, TV, AV, TAV）在五个主题上的表现。

### 主要发现

单独使用视频在五个主题中的四个上表现强劲，强调了视频在捕捉视觉上下文和有效线索方面的价值；选择性组合如TV或AV通常优于TAV，表明过多的模态可能会引入噪声。

### 结论

这些结果建立了金融短视频字幕生成的第一个基线，并说明了在该领域中锚定复杂视觉线索的潜力和挑战。

### 翻译

我们通过测试转录文本(T)、音频(A)和视频(V)的联合推理，评估了多模态大语言模型(MLLMs)在金融短视频(SVs)主题对齐字幕生成中的性能。使用624个标注的YouTube SVs，我们评估了所有七种模态组合(T, A, V, TA, TV, AV, TAV)在五个主题上的表现：主要推荐、情感分析、视频目的、视觉分析和金融实体识别。单独使用视频在五个主题中的四个上表现强劲，强调了其在捕捉视觉上下文和有效线索(如情感、手势和肢体语言)方面的价值。选择性组合如TV或AV通常优于TAV，表明过多的模态可能会引入噪声。这些结果建立了金融短视频字幕生成的第一个基线，并说明了在该领域中锚定复杂视觉线索的潜力和挑战。所有代码和数据可以在我们的Github上找到，使用CC-BY-NC-SA 4.0许可证。


### 论文摘要

We evaluate multimodal large language models (MLLMs) for topic-aligned captioning in financial short-form videos (SVs) by testing joint reasoning over transcripts (T), audio (A), and video (V). Using 624 annotated YouTube SVs, we assess all seven modality combinations (T, A, V, TA, TV, AV, TAV) across five topics: main recommendation, sentiment analysis, video purpose, visual analysis, and financial entity recognition. Video alone performs strongly on four of five topics, underscoring its value for capturing visual context and effective cues such as emotions, gestures, and body language. Selective pairs such as TV or AV often surpass TAV, implying that too many modalities may introduce noise. These results establish the first baselines for financial short-form video captioning and illustrate the potential and challenges of grounding complex visual cues in this domain. All code and data can be found on our Github under the CC-BY-NC-SA 4.0 license.

---

## 58. Building the EHR Foundation Model via Next Event Prediction

**论文链接:** [http://arxiv.org/abs/2509.25591v1](http://arxiv.org/abs/2509.25591v1)

**作者:** Zekai Chen, Arda Pekis, Kevin Brown

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为NextEvent Prediction (NEP)的框架，通过自回归微调增强大型语言模型在电子健康记录时间推理方面的能力，在多个临床任务中表现出色。

### 背景

电子健康记录包含丰富的动态时间信息，但传统编码方法无法充分捕捉；大型语言模型虽在EHR建模方面有潜力，但难以推理顺序临床事件和时序依赖关系。

### 目的

开发一种能够增强大型语言模型时间推理能力的框架，以更好地建模电子健康记录中的动态信息。

### 方法

将EHR重新表述为带时间戳的事件链，通过自回归微调大型语言模型来预测未来医疗事件，从而明确建模疾病进展模式和因果关系。

### 主要发现

在肿瘤学生存预测和临床诊断任务上，NEP比专门EHR模型高出4.6%的AUROC，比通用LLMs高出7.2%的C-index；结合了最先进的预测精度与符合已知疾病途径的临床可解释注意力模式。

### 结论

NEP框架能有效捕捉EHR中的时间动态，提高预测性能，并提供临床上有意义的解释。

### 翻译

电子健康记录(EHRs)包含丰富的动态时间信息，而传统编码方法无法充分捕捉。虽然大型语言模型(LLMs)在EHR建模方面显示出潜力，但它们难以推理顺序临床事件和时序依赖关系。我们提出了NextEvent Prediction (NEP)，一个通过在临床事件序列上进行自回归微调来增强LLMs时间推理能力的框架。通过将EHR重新表述为带时间戳的事件链并预测未来的医疗事件，NEP明确地建模了疾病进展模式和因果关系。在肿瘤学生存预测和临床诊断任务上的广泛评估证明了NEP的优越性，在时间推理任务中，比专门的EHR模型高出4.6%的AUROC，比通用LLMs高出7.2%的C-index。我们的分析揭示了双重好处：结合了最先进的预测精度与符合已知疾病途径的临床可解释注意力模式。


### 论文摘要

Electronic Health Records (EHRs) contain rich temporal dynamics that conventional encoding approaches fail to adequately capture. While Large Language Models (LLMs) show promise for EHR modeling, they struggle to reason about sequential clinical events and temporal dependencies. We propose Next Event Prediction (NEP), a framework that enhances LLMs' temporal reasoning through autoregressive fine-tuning on clinical event sequences. By reformulating EHRs as timestamped event chains and predicting future medical events, NEP explicitly models disease progression patterns and causal relationships. Extensive evaluations across oncology survival prediction and clinical diagnosis tasks demonstrate NEP's superiority, outperforming specialized EHR models by 4.6% AUROC and general-purpose LLMs by 7.2% C-index in temporal reasoning tasks. Our analyses reveal dual benefits: state-of-the-art prediction accuracy combined with clinically interpretable attention patterns that align with known disease pathways.

---

## 59. FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting

**论文链接:** [http://arxiv.org/abs/2509.24304v2](http://arxiv.org/abs/2509.24304v2)

**作者:** Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, Yu Cheng

**发布时间:** 2025-09-29

### GPT解析

### 总结

FrameThinker是一种新型框架，使大型视觉语言模型能够通过迭代查询视频内容来进行长视频推理，采用两阶段训练策略解决了模型适应新视频动作和设计奖励函数的挑战。

### 背景

大型视觉语言模型在视频理解方面取得了显著进展，但在长视频推理应用中面临均匀帧采样和静态文本推理的效率问题，难以处理视觉密集型视频任务。

### 目的

克服现有LVLMs在长视频推理中的局限性，提出一种使模型能够迭代查询视频内容的框架，提高处理长视频的效率和效果。

### 方法

提出FrameThinker框架，采用两阶段训练策略：首先使用监督微调培养基础动作能力，然后使用强化学习优化战略决策策略；在RL阶段对每个动作和格式的奖励设计进行了深入探索。

### 主要发现

在多个推理和长视频理解基准测试上，FrameThinker比基线模型平均提高10.4%，同时大幅减少处理帧数；7B模型在LongVideo-Reason上达到76.1%准确率，仅使用平均20.6帧，比竞争模型LongVILA-R1表现更好且帧数减少20倍以上。

### 结论

FrameThinker通过迭代查询视频内容的方式，在长视频推理任务中展现出卓越的效率和效果，为处理视觉密集型视频任务提供了新思路。

### 翻译

本文介绍了大型视觉语言模型在视频理解方面的进展及其在长视频推理中面临的挑战，提出了FrameThinker框架，通过两阶段训练策略和创新的奖励设计，实现了在减少处理帧数的同时提高推理性能的效果。


### 论文摘要

While Large Vision-Language Models (LVLMs) have achieved substantial progress in video understanding, their application to long video reasoning is hindered by uniform frame sampling and static textual reasoning, which are inefficient and struggle to handle visually intensive video tasks. To overcome these challenges, in this paper, we introduce the concept of thinking with long videos and propose a novel framework FrameThinker. Within this framework, LVLMs are able to iteratively interrogate video content. Developing such video reasoning capabilities in LVLMs presents notable challenges, particularly in adapting the model to new video actions (e.g. select frame), and designing reward functions to guide LVLMs to adopt the newly introduced action. To solve these challenges, we propose a two-phase training strategy, first employing Supervised Fine-Tuning (SFT) to instill fundamental action capabilities, followed by Reinforcement Learning (RL) to optimize a strategic decision-making policy. Notably, in this RL phase, we conduct an in-depth and comprehensive exploration of the reward design for each action and format reward. Extensive experiments on reasoning benchmarks like Video-Holmes, LongVideo-Reason, and long-video understanding benchmarks such as LongVideoBench, MLVU, VideoMME, and LVBench, demonstrate that FrameThinker achieves a significant average improvement of +10.4% over baselines while drastically reducing the number of processed frames. Most notably, our 7B model, FrameThinker establishes a new state-of-the-art on LongVideo-Reason, achieving 76.1% accuracy using an average of only 20.6 frames. This not only outperforms the competitive LongVILA-R1 (72.0%) but does so with over 20x fewer frames (vs. 512), demonstrating unparalleled efficiency and effectiveness.

---

## 60. UniVid: The Open-Source Unified Video Model

**论文链接:** [http://arxiv.org/abs/2509.24200v2](http://arxiv.org/abs/2509.24200v2)

**作者:** Jiabin Luo, Junhui Lin, Zeyu Zhang, Biao Wu, Meng Fang, Ling Chen, Hao Tang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了UniVid，一种统一架构，通过轻量级适配器将多模态大模型(MLLM)与扩散解码器耦合，实现了视频理解和生成的统一建模。

### 背景

统一视频建模结合生成和理解能力日益重要，但面临两大挑战：基于流生成过程中因文本-视觉标记不平衡导致的语义忠实度问题，以及流轨迹上统一跨模态注意力的局限性；此外，还需高效将以图像为中心的MLLM扩展到视频，避免昂贵的重新训练。

### 目的

开发一个能够同时处理视频理解和生成任务的统一架构，解决现有方法中的关键挑战。

### 方法

提出UniVid架构，引入温度模态对齐(Temperature Modality Alignment)提高提示遵循性，以及金字塔反射(Pyramid Reflection)通过动态关键帧选择实现高效的时间推理。

### 主要发现

在标准基准测试上取得了最先进性能：在VBench-Long总分上比EasyAnimateV5.1提高2.2%，在MSVD-QA和ActivityNet-QA上分别比最佳之前的7B基线提高1.0%和3.3%的准确率。

### 结论

UniVid成功解决了统一视频建模中的关键挑战，实现了高效的视频理解和生成能力，无需昂贵的重新训练。

### 翻译

统一结合生成和理解能力的视频建模变得越来越重要，但面临两个关键挑战：由于文本-视觉标记不平衡，在基于流的生成过程中保持语义忠实度，以及在流轨迹上统一跨模态注意力的局限性，以及高效地将以图像为中心的多模态大模型扩展到视频而无需昂贵的重新训练。我们提出了UniVid，一种通过轻量级适配器将MLLM与扩散解码器耦合的统一架构，实现了视频理解和生成。我们引入温度模态对齐以提高提示遵循性，并通过动态关键帧选择提出金字塔反射以实现高效的时间推理。在标准基准上的大量实验展示了最先进的性能，与EasyAnimateV5.1相比，在VBench-Long总分上提高了2.2%，与最佳之前的7B基线相比，在MSVD-QA和ActivityNet-QA上分别提高了1.0%和3.3%的准确率。代码：https://github.com/AIGeeksGroup/UniVid。网站：https://aigeeksgroup.github.io/UniVid。


### 论文摘要

Unified video modeling that combines generation and understanding capabilities is increasingly important but faces two key challenges: maintaining semantic faithfulness during flow-based generation due to text-visual token imbalance and the limitations of uniform cross-modal attention across the flow trajectory, and efficiently extending image-centric MLLMs to video without costly retraining. We present UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. We introduce Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection. Extensive experiments on standard benchmarks demonstrate state-of-the-art performance, achieving a 2.2% improvement on VBench-Long total score compared to EasyAnimateV5.1, and 1.0% and 3.3% accuracy gains on MSVD-QA and ActivityNet-QA, respectively, compared with the best prior 7B baselines. Code: https://github.com/AIGeeksGroup/UniVid. Website: https://aigeeksgroup.github.io/UniVid.

---

## 61. FameMind: Frame-Interleaved Video Reasoning via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2509.24008v2](http://arxiv.org/abs/2509.24008v2)

**作者:** Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, Yujun Cai

**发布时间:** 2025-09-28

**备注:** Underreview

### GPT解析

### 总结

本文介绍了FrameMind，一个通过强化学习训练的端到端框架，使视频理解模型能够在推理过程中动态请求视觉信息，显著提升了在需要广泛时间覆盖或精细空间细节的任务上的性能。

### 背景

当前视频理解模型依赖固定帧采样策略，处理预定的视觉输入而忽略不同问题的具体推理需求，限制了模型自适应收集视觉证据的能力。

### 目的

开发一个能够动态请求视觉信息的视频理解框架，以克服静态采样方法的局限性，提升模型在复杂视频理解任务中的表现。

### 方法

提出FrameMind框架，使用帧交错思维链(FiCOT)在文本推理和主动视觉感知间交替操作；引入动态分辨率帧采样(DRFS)使模型接触多样化时间-空间权衡；开发DRFS-GRPO组相对策略优化算法，基于结果奖励学习而无需帧级注释。

### 主要发现

在MLVU和VideoMME等基准测试上的实验表明，该方法显著优于现有模型，推进了灵活高效视频理解的最先进水平。

### 结论

FrameMind通过动态采样策略和强化学习方法，有效解决了传统视频理解模型的局限性，为灵活高效的视频理解提供了新范式。

### 翻译

当前视频理解模型依赖于固定的帧采样策略，无论每个问题的具体推理需求如何，都处理预定的视觉输入。这种静态方法限制了模型自适应收集视觉证据的能力，导致在需要广泛时间覆盖或精细空间细节的任务上表现不佳。在本文中，我们介绍了FrameMind，一个通过强化训练的端到端框架，使模型能够通过帧交错思维链(FiCOT)在推理过程中动态请求视觉信息。与传统方法不同，FrameMind在多轮操作中运行，模型在文本推理和主动视觉感知之间交替，使用工具根据识别的知识差距提取目标帧或视频剪辑。为了训练有效的动态采样策略，我们提出了动态分辨率帧采样(DRFS)，在学习过程中使模型接触多样化的时间-空间权衡，以及DRFS-GRPO，一种基于结果奖励学习的组相对策略优化算法，不需要帧级注释。在MLVU和VideoMME等具有挑战性的基准测试上的大量实验表明，我们的方法显著优于现有模型，推进了灵活高效视频理解的最先进水平。


### 论文摘要

Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding.

---

## 62. Nephrobase Cell+: Multimodal Single-Cell Foundation Model for Decoding Kidney Biology

**论文链接:** [http://arxiv.org/abs/2509.26223v1](http://arxiv.org/abs/2509.26223v1)

**作者:** Chenyu Li, Elias Ziyadeh, Yash Sharma, Bernhard Dumoulin, Jonathan Levinsohn, Eunji Ha, Siyu Pan, Vishwanatha Rao, Madhav Subramaniyam, Mario Szegedy, Nancy Zhang, Katalin Susztak

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究开发了Nephrobase Cell+，首个肾脏特异性大型基础模型，在约3950万个单细胞和单核图谱上预训练，显著优于现有模型，为肾脏研究提供强大资源。

### 背景

大型基础模型已革新单细胞分析，但缺乏肾脏特异性模型；肾脏复杂细胞结构使大规模组学数据整合困难，现有框架难以校正批次效应、捕获跨模态变异和跨物种泛化。

### 目的

开发首个肾脏特异性大型基础模型，评估其在肾脏单细胞分析中的性能，并与现有模型进行比较。

### 方法

开发了Nephrobase Cell+，使用基于transformer的编码器-解码器架构，具有基因标记交叉注意力和专家混合模块，在约4,319个样本的约3950万个单细胞和单核图谱的约1000亿个token上预训练。

### 主要发现

Nephrobase Cell+在人类和小鼠肾脏中产生紧密聚集、生物连贯的嵌入，远超Geneformer、scGPT、UCE等基础模型及PCA、自编码器等传统方法；实现最高聚类一致性和批次混合分数，有效去除批次效应同时保留细胞类型结构；跨物种评估显示同源细胞类型优越对齐，主要肾脏谱系零样本注释准确率超90%；即使是较小参数变体也优于所有现有模型。

### 结论

Nephrobase Cell+提供肾脏生物学统一的高保真表示，具有鲁棒性和跨物种可转移性，为肾脏基因组学和疾病研究提供强大资源，性能远超当前单细胞基础模型。

### 翻译

背景：大型基础模型已经彻底改变了单细胞分析，但目前尚无肾脏特异性模型，也不清楚器官特异性模型是否优于通用模型。肾脏复杂的细胞结构进一步整合大规模组学数据的复杂性，当前在有限数据集上训练的框架难以校正批次效应、捕获跨模态变异并跨物种泛化。方法：我们开发了Nephrobase Cell+，这是第一个肾脏特异性大型基础模型，在约4,319个样本的约3950万个单细胞和单核图谱的约1000亿个token上进行了预训练。Nephrobase Cell+使用基于transformer的编码器-解码器架构，具有基因标记交叉注意力和专家混合模块，用于可扩展的表示学习。结果：Nephrobase Cell+为肾脏单细胞分析树立了新基准。在人类和小鼠肾脏中产生紧密聚集、生物连贯的嵌入，远远超过了之前的基础模型，如Geneformer、scGPT和UCE，也优于传统方法，如PCA和自编码器。它实现了最高的聚类一致性和批次混合分数，有效去除供体/分析批次效应，同时保留细胞类型结构。跨物种评估显示同源细胞类型的优越对齐，在人类和小鼠中主要肾脏谱系超过90%的零样本注释准确性。即使其10亿参数和5亿参数的变体也持续优于所有现有模型。结论：Nephrobase Cell+提供肾脏生物学统一的高保真表示，具有鲁棒性、跨物种可转移性，当前单细胞基础模型无法与之匹敌，为肾脏基因组学和疾病研究提供了强大的资源。


### 论文摘要

Background: Large foundation models have revolutionized single-cell analysis, yet no kidney-specific model currently exists, and it remains unclear whether organ-focused models can outperform generalized models. The kidney's complex cellular architecture further complicate integration of large-scale omics data, where current frameworks trained on limited datasets struggle to correct batch effects, capture cross-modality variation, and generalize across species. Methods: We developed Nephrobase Cell+, the first kidney-focused large foundation model, pretrained on ~100 billion tokens from ~39.5 million single-cell and single-nucleus profiles across 4,319 samples. Nephrobase Cell+ uses a transformer-based encoder-decoder architecture with gene-token cross-attention and a mixture-of-experts module for scalable representation learning. Results: Nephrobase Cell+ sets a new benchmark for kidney single-cell analysis. It produces tightly clustered, biologically coherent embeddings in human and mouse kidneys, far surpassing previous foundation models such as Geneformer, scGPT, and UCE, as well as traditional methods such as PCA and autoencoders. It achieves the highest cluster concordance and batch-mixing scores, effectively removing donor/assay batch effects while preserving cell-type structure. Cross-species evaluation shows superior alignment of homologous cell types and >90% zero-shot annotation accuracy for major kidney lineages in both human and mouse. Even its 1B-parameter and 500M variants consistently outperform all existing models. Conclusions: Nephrobase Cell+ delivers a unified, high-fidelity representation of kidney biology that is robust, cross-species transferable, and unmatched by current single-cell foundation models, offering a powerful resource for kidney genomics and disease research.

---

## 63. EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2509.26157v1](http://arxiv.org/abs/2509.26157v1)

**作者:** Sachith Abeywickrama, Emadeldeen Eldele, Min Wu, Xiaoli Li, Chau Yuen

**发布时间:** 2025-09-30

**备注:** Preprint. Under Review

### GPT解析

### 总结

本文提出了EntroPE（熵引导动态patch编码器），一种新颖的时间感知框架，通过动态检测过渡点并智能放置patch边界，解决了现有Transformer时间序列预测模型中时间连续性被破坏的问题，同时保留了patch的计算优势。

### 背景

基于Transformer的模型显著推进了时间序列预测的发展，基于patch的输入策略提供了效率并改善了长期建模能力。然而，现有方法依赖于时间上无感知的patch构建，其中任意的起始位置和固定长度通过分割边界处的自然过渡来破坏时间连续性，削弱了短期依赖关系和表示学习效果。

### 目的

提出一个时间感知的框架，动态检测过渡点并动态放置patch边界，以保留时间结构的同时保持patch的计算优势，从而提高时间序列预测的准确性和效率。

### 方法

EntroPE包含两个关键模块：1) 基于熵的动态分割器(EDP)，应用信息论标准来定位自然时间变化并确定patch边界；2) 自适应patch编码器(APE)，使用池化和交叉注意力来捕获patch内依赖关系并生成固定大小的潜在表示。这些嵌入随后由全局transformer处理，以建模patch间动态。

### 主要发现

在长期预测基准测试中，EntroPE提高了准确性和效率，证明了熵引导的动态patching是时间序列建模的一种有前途的新范式。

### 结论

EntroPE框架通过智能的时间感知patch策略，有效解决了现有时间序列预测模型中的连续性问题，同时保持了计算效率。代码已在GitHub上公开。

### 翻译

基于Transformer的模型显著推进了时间序列预测的发展，基于patch的输入策略提供了效率并改善了长期建模能力。然而，现有方法依赖于时间上无感知的patch构建，其中任意的起始位置和固定长度通过分割边界处的自然过渡来破坏时间连续性。这种简单的分割经常破坏短期依赖关系并削弱表示学习。为此，我们提出了EntroPE（熵引导动态patch编码器），一种新颖的时间感知框架，它通过条件熵动态检测过渡点并动态放置patch边界。这保留了时间结构，同时保持了patch的计算优势。EntroPE包含两个关键模块，即应用信息论标准来定位自然时间变化并确定patch边界的基于熵的动态分割器(EDP)，以及使用池化和交叉注意力来捕获patch内依赖关系并生成固定大小潜在表示的自适应patch编码器(APE)。这些嵌入随后由全局transformer处理，以建模patch间动态。跨长期预测基准的实验表明，EntroPE提高了准确性和效率，确立了熵引导的动态patching作为时间序列建模的一种有前途的新范式。代码可在https://github.com/Sachithx/EntroPE获取。


### 论文摘要

Transformer-based models have significantly advanced time series forecasting, with patch-based input strategies offering efficiency and improved long-horizon modeling. Yet, existing approaches rely on temporally-agnostic patch construction, where arbitrary starting positions and fixed lengths fracture temporal coherence by splitting natural transitions across boundaries. This naive segmentation often disrupts short-term dependencies and weakens representation learning. In response, we propose EntroPE (Entropy-Guided Dynamic Patch Encoder), a novel, temporally informed framework that dynamically detects transition points via conditional entropy and dynamically places patch boundaries. This preserves temporal structure while retaining the computational benefits of patching. EntroPE consists of two key modules, namely an Entropy-based Dynamic Patcher (EDP) that applies information-theoretic criteria to locate natural temporal shifts and determine patch boundaries, and an Adaptive Patch Encoder (APE) that employs pooling and cross-attention to capture intra-patch dependencies and produce fixed-size latent representations. These embeddings are then processed by a global transformer to model inter-patch dynamics. Experiments across long-term forecasting benchmarks demonstrate that EntroPE improves both accuracy and efficiency, establishing entropy-guided dynamic patching as a promising new paradigm for time series modeling. Code is available at: https://github.com/Sachithx/EntroPE.

---

## 64. Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies

**论文链接:** [http://arxiv.org/abs/2509.25822v1](http://arxiv.org/abs/2509.25822v1)

**作者:** Jing Wang, Weiting Peng, Jing Tang, Zeyu Gong, Xihua Wang, Bo Tao, Li Cheng

**发布时间:** 2025-09-30

**备注:** 42 pages, 17 figures, 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

### GPT解析

### 总结

本文提出了一种行动引导的扩散策略(DP-AG)，通过统一的表征学习方法建立了感知与行动之间的动态相互作用，显著提升了模仿学习性能。

### 背景

现有模仿学习方法将感知和行动解耦，忽视了人类自然利用的感官表征和行动执行之间的因果互惠关系，而这种关系对于适应性行为至关重要。

### 目的

弥合感知与行动解耦的差距，开发一种统一的表征学习方法，明确通过概率潜在动力学建立感知和行动之间的动态相互作用。

### 方法

引入行动引导的扩散策略(DP-AG)，通过变分推断将潜在观测编码为高斯后验，使用行动引导的随机微分方程演化它们，扩散策略噪声预测的向量-雅可比积作为结构化随机力驱动潜在更新，并引入循环一致的对比损失促进感知和行动之间的双向学习。

### 主要发现

DP-AG在模拟基准测试和真实世界UR5操作任务中显著优于最先进的方法，理论上证明了对比目标增强了潜在和行动轨迹的连续性。

### 结论

DP-AG为弥合生物适应性和人工策略学习之间的差距提供了有希望的步骤。

### 翻译

现有的模仿学习方法将感知和行动解耦，这忽视了人类自然利用的感官表征和行动执行之间的因果互惠关系，而这种关系对于适应性行为至关重要。为了弥合这一差距，我们引入了行动引导的扩散策略(DP-AG)，这是一种统一的表征学习方法，通过概率潜在动力学明确建立了感知和行动之间的动态相互作用。DP-AG通过变分推断将潜在观测编码为高斯后验，并使用行动引导的SDE演化它们，其中扩散策略噪声预测的向量-雅可比积作为结构化随机力驱动潜在更新。为了促进感知和行动之间的双向学习，我们引入了循环一致的对比损失，将噪声预测器的梯度流组织成连贯的感知-行动循环，强制潜在更新和行动精细化中的相互一致转换。理论上，我们推导了行动引导SDE的变分下界，并证明了对比目标增强了潜在和行动轨迹的连续性。经验上，DP-AG在模拟基准测试和真实世界UR5操作任务中显著优于最先进的方法。因此，我们的DP-AG为弥合生物适应性和人工策略学习提供了有希望的进展。


### 论文摘要

Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action--Guided Diffusion Policy (DP--AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP--AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle--consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception--action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP--AG significantly outperforms state--of--the--art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP--AG offers a promising step toward bridging biological adaptability and artificial policy learning.

---

## 65. Learning to Interact in World Latent for Team Coordination

**论文链接:** [http://arxiv.org/abs/2509.25550v1](http://arxiv.org/abs/2509.25550v1)

**作者:** Dongsu Lee, Daehee Lee, Yaru Niu, Honguk Woo, Amy Zhang, Ding Zhao

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一个名为interactive world latent (IWoL)的新型表征学习框架，用于促进多智能体强化学习中的团队协调。

### 背景

在多智能体强化学习中，构建有效的团队协调表征具有挑战性，主要源于多智能体交互产生的复杂动态性和局部观察导致的不完整信息。

### 目的

设计一种能够捕获智能体间关系和任务特定世界信息的表征学习框架，以解决团队协调问题。

### 方法

构建一个可学习的表征空间，通过直接建模通信协议来联合捕获智能体间关系和任务特定的世界信息。该表征既可以用作每个智能体的隐式潜在变量，也可以用作通信的显式消息。

### 主要发现

IWoL框架能够实现完全去中心化执行和隐式协调，同时避免了显式消息传递的固有缺点，如决策速度慢、易受恶意攻击者攻击以及对带宽限制敏感。在四个具有挑战性的MARL基准测试中，IWoL为团队协调提供了简单而强大的解决方案。

### 结论

IWoL是一种简单而强大的团队协调解决方案，可以与现有的MARL算法结合以进一步提高性能。

### 翻译

这项工作提出了一种新的表征学习框架——交互世界潜在表征(IWoL)，以促进多智能体强化学习中的团队协调。由于多智能体交互产生的复杂动态性和局部观察导致的不完整信息，构建有效的团队协调表征是一个具有挑战性的问题。我们的关键见解是构建一个可学习的表征空间，通过直接建模通信协议来联合捕获智能体间关系和任务特定的世界信息。我们维持这种表征完全去中心化执行和隐式协调，同时避免了显式消息传递的固有缺点，例如决策速度慢、易受恶意攻击者攻击以及对带宽限制敏感。在实践中，我们的表征不仅可以作为每个智能体的隐式潜在变量，还可以作为通信的显式消息。在四个具有挑战性的MARL基准测试中，我们评估了两种变体，并表明IWoL为团队协调提供了简单而强大的关键。此外，我们证明我们的表征可以与现有的MARL算法结合以进一步提高其性能。


### 论文摘要

This work presents a novel representation learning framework, interactive world latent (IWoL), to facilitate team coordination in multi-agent reinforcement learning (MARL). Building effective representation for team coordination is a challenging problem, due to the intricate dynamics emerging from multi-agent interaction and incomplete information induced by local observations. Our key insight is to construct a learnable representation space that jointly captures inter-agent relations and task-specific world information by directly modeling communication protocols. This representation, we maintain fully decentralized execution with implicit coordination, all while avoiding the inherent drawbacks of explicit message passing, e.g., slower decision-making, vulnerability to malicious attackers, and sensitivity to bandwidth constraints. In practice, our representation can be used not only as an implicit latent for each agent, but also as an explicit message for communication. Across four challenging MARL benchmarks, we evaluate both variants and show that IWoL provides a simple yet powerful key for team coordination. Moreover, we demonstrate that our representation can be combined with existing MARL algorithms to further enhance their performance.

---

## 66. Joint Embeddings Go Temporal

**论文链接:** [http://arxiv.org/abs/2509.25449v1](http://arxiv.org/abs/2509.25449v1)

**作者:** Sofiane Ennadir, Siavash Golkar, Leopoldo Sarra

**发布时间:** 2025-09-29

**备注:** Accepted at the Workshop on Time Series in the Age of Large Models -  NeurIPS 2024

### GPT解析

### 总结

本文提出了一种名为时间序列联合嵌入预测架构(TS-JEPA)的自监督学习方法，专门针对时间序列表征学习进行了优化，并在分类和预测任务上验证了其有效性。

### 背景

自监督学习在无监督表征学习中取得显著成功，推动了自然语言和图像处理的突破，但现有方法依赖自回归和掩码建模，容易受噪声或混淆变量影响。

### 目的

引入联合嵌入预测架构(JEPA)以在潜在空间中进行自监督学习，并将其进步应用于时间序列领域，开发专门为时间序列表征学习设计的架构。

### 方法

提出时间序列JEPA(TS-JEPA)，这是一种基于JEPA框架但专门为时间序列表征学习而调整的架构。

### 主要发现

TS-JEPA在不同的标准数据集上能够匹配或超越当前最先进的基线方法，在分类和预测任务上均表现出色，且在多样化任务上展现出强大的性能平衡性。

### 结论

这项工作为基于联合嵌入开发未来时间序列基础模型奠定了基础，TS-JEPA有望成为学习通用表征的强大基础。

### 翻译

自监督学习最近在无监督表征学习中取得了巨大成功，推动了自然语言和图像处理的突破。然而，这些方法通常依赖于自回归和掩码建模，旨在重现输入中被遮掩的信息，这容易受到噪声或混淆变量的影响。为了解决这个问题，联合嵌入预测架构(JEPA)被引入，旨在潜在空间中进行自监督学习。为了将这些进步应用于时间序列领域，我们引入了时间序列JEPA(TS-JEPA)，这是一种专门为时间序列表征学习而调整的架构。我们在分类和预测任务上验证了TS-JEPA，表明它可以在不同的标准数据集上匹配或超越当前最先进的基线方法。值得注意的是，我们的方法在多样化任务上表现出强大的性能平衡性，表明它作为学习通用表征的强大基础的潜力。因此，这项工作为基于联合嵌入开发未来时间序列基础模型奠定了基础。


### 论文摘要

Self-supervised learning has seen great success recently in unsupervised representation learning, enabling breakthroughs in natural language and image processing. However, these methods often rely on autoregressive and masked modeling, which aim to reproduce masked information in the input, which can be vulnerable to the presence of noise or confounding variables. To address this problem, Joint-Embedding Predictive Architectures (JEPA) has been introduced with the aim to perform self-supervised learning in the latent space. To leverage these advancements in the domain of time series, we introduce Time Series JEPA (TS-JEPA), an architecture specifically adapted for time series representation learning. We validate TS-JEPA on both classification and forecasting, showing that it can match or surpass current state-of-the-art baselines on different standard datasets. Notably, our approach demonstrates a strong performance balance across diverse tasks, indicating its potential as a robust foundation for learning general representations. Thus, this work lays the groundwork for developing future time series foundation models based on Joint Embedding.

---

## 67. Uncertainty-Aware Generative Oversampling Using an Entropy-Guided Conditional Variational Autoencoder

**论文链接:** [http://arxiv.org/abs/2509.25334v1](http://arxiv.org/abs/2509.25334v1)

**作者:** Amirhossein Zare, Amirhessam Zare, Parmida Sadat Pezeshki, Herlock, Rahimi, Ali Ebrahimi, Ignacio Vázquez-García, Leo Anthony Celi

**发布时间:** 2025-09-29

**备注:** 16 pages, 2 figures

### GPT解析

### 总结

本文提出了一种名为LEO-CVAE的局部熵引导过采样方法，结合条件变分自编码器和局部熵概念，特别关注不确定性和边界区域样本，以改善高维生物医学数据中的类别不平衡问题。

### 背景

类别不平衡是机器学习中的主要挑战，特别是在高维生物医学数据中，这些数据通常由非线性流形结构主导。传统过采样方法如SMOTE依赖局部线性插值，常产生不合理合成样本；深度生成模型如CVAE虽能更好捕获非线性分布，但标准变体将所有少数类样本同等对待，忽略了边界区域样本的重要性。

### 目的

开发一个能将局部不确定性明确纳入表示学习和数据生成的生成式过采样框架，提高在不平衡数据分类任务中的性能，特别是在具有复杂非线性结构的数据领域。

### 方法

提出LEO-CVAE方法，通过计算样本邻域类分布的香农熵来量化不确定性，高熵表示更大类重叠，作为不确定性代理。该方法通过两种机制利用这一信号：局部熵加权损失强调在不确定区域的鲁棒学习；熵引导采样策略在信息丰富、类重叠区域集中生成样本。

### 主要发现

在临床基因组学数据集ADNI和TCGA肺癌上应用LEO-CVAE，持续提高了分类器性能，优于传统过采样和生成基线，证明了不确定性感知生成式过采样在不平衡学习中的价值。

### 结论

LEO-CVAE通过明确考虑局部不确定性和边界区域样本，解决了传统过采样方法和标准生成模型的局限性，为处理高维生物医学数据中的类别不平衡问题提供了有效解决方案。

### 翻译

类别不平衡仍然是机器学习中的一个主要挑战，特别是在高维生物医学数据中，其中非线性流形结构占主导地位。传统的过采样方法如SMOTE依赖于局部线性插值，常常产生不合理的合成样本。深度生成模型如条件变分自编码器(CVAE)能更好地捕获非线性分布，但标准变体将所有少数类样本同等对待，忽略了边界区域样本的重要性，而启发式方法如Borderline-SMOTE和ADASYN则强调了这些样本的重要性。我们提出了一种结合条件变分自编码器的局部熵引导过采样方法(LEO-CVAE)，这是一个生成式过采样框架，明确将局部不确定性纳入表示学习和数据生成。为了量化不确定性，我们计算样本邻域中类分布的香农熵：高熵表示更大的类重叠，作为不确定性的代理。LEO-CVAE通过两种机制利用这一信号：(i)局部熵加权损失，强调在不确定区域的鲁棒学习；(ii)熵引导采样策略，在这些信息丰富、类重叠区域集中生成样本。应用于临床基因组学数据集(ADNI和TCGA肺癌)，LEO-CVAE持续提高了分类器性能，优于传统过采样和生成基线。这些结果强调了不确定性感知生成式过采样在不平衡学习中的价值，特别是在由复杂非线性结构主导的领域，如组学数据。


### 论文摘要

Class imbalance remains a major challenge in machine learning, especially for high-dimensional biomedical data where nonlinear manifold structures dominate. Traditional oversampling methods such as SMOTE rely on local linear interpolation, often producing implausible synthetic samples. Deep generative models like Conditional Variational Autoencoders (CVAEs) better capture nonlinear distributions, but standard variants treat all minority samples equally, neglecting the importance of uncertain, boundary-region examples emphasized by heuristic methods like Borderline-SMOTE and ADASYN.   We propose Local Entropy-Guided Oversampling with a CVAE (LEO-CVAE), a generative oversampling framework that explicitly incorporates local uncertainty into both representation learning and data generation. To quantify uncertainty, we compute Shannon entropy over the class distribution in a sample's neighborhood: high entropy indicates greater class overlap, serving as a proxy for uncertainty. LEO-CVAE leverages this signal through two mechanisms: (i) a Local Entropy-Weighted Loss (LEWL) that emphasizes robust learning in uncertain regions, and (ii) an entropy-guided sampling strategy that concentrates generation in these informative, class-overlapping areas.   Applied to clinical genomics datasets (ADNI and TCGA lung cancer), LEO-CVAE consistently improves classifier performance, outperforming both traditional oversampling and generative baselines. These results highlight the value of uncertainty-aware generative oversampling for imbalanced learning in domains governed by complex nonlinear structures, such as omics data.

---

## 68. Interpretable Kernel Representation Learning at Scale: A Unified Framework Utilizing Nyström Approximation

**论文链接:** [http://arxiv.org/abs/2509.24467v2](http://arxiv.org/abs/2509.24467v2)

**作者:** Maedeh Zarvandi, Michael Timothy, Theresa Wasserer, Debarghya Ghoshdastidar

**发布时间:** 2025-09-29

**备注:** 19 Pages, 3 figures

### GPT解析

### 总结

核方法虽有理论基础但可扩展性受限，作者提出KREPES框架解决基于核的表示学习可扩展性问题，并证明了其效率和可解释性优势。

### 背景

核方法为非线性非参数学习提供理论框架，但受时间和内存成本限制；核回归已有扩展进展，但基于核的表示学习缺乏可扩展框架，限制了在基础模型中的应用。

### 目的

开发一个统一、可扩展的基于核的表示学习框架，使其能够在基础模型时代从未标记的大量数据中学习表示。

### 方法

引入KREPES框架，通过Nyström近似实现基于核的表示学习，适用于广泛的非监督和自监督损失函数。

### 主要发现

KREPES在大型图像和表格数据集上表现出效率，且使学习到的表示具有可解释性，优于深度模型。

### 结论

KREPES为基于核的表示学习提供了可扩展的解决方案，同时保持了核方法的理论优势，并提供了可解释性的额外好处。

### 翻译

核方法为非线性与非参数学习提供了理论依据扎实的框架，具有强大的解析基础和统计保证。然而，它们的可扩展性一直受到时间和内存成本的限制。虽然在扩展核回归方面已有进展，但尚不存在可扩展的基于核的表示学习框架，这限制了它们在基础模型时代的使用，因为基础模型需要从未标记的大量数据中学习表示。我们引入了KREPES——一种通过Nyström近似实现的统一、可扩展的基于核的表示学习框架。KREPES适用于广泛的非监督和自监督损失函数，在大型图像和表格数据集上的实验证明了其效率。关键的是，KREPES使学习到的表示具有可解释性，这是深度模型的一个直接优势，我们通过专门的分析证实了这一点。


### 论文摘要

Kernel methods provide a theoretically grounded framework for non-linear and non-parametric learning, with strong analytic foundations and statistical guarantees. Yet, their scalability has long been limited by prohibitive time and memory costs. While progress has been made in scaling kernel regression, no framework exists for scalable kernel-based representation learning, restricting their use in the era of foundation models where representations are learned from massive unlabeled data. We introduce KREPES -- a unified, scalable framework for kernel-based representation learning via Nystr\"om approximation. KREPES accommodates a wide range of unsupervised and self-supervised losses, and experiments on large image and tabular datasets demonstrate its efficiency. Crucially, KREPES enables principled interpretability of the learned representations, an immediate benefit over deep models, which we substantiate through dedicated analysis.

---

## 69. InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions

**论文链接:** [http://arxiv.org/abs/2509.25270v1](http://arxiv.org/abs/2509.25270v1)

**作者:** Liangjian Wen, Qun Dai, Jianzhuang Liu, Jiangtao Zheng, Yong Dai, Dongkai Wang, Zhao Kang, Jun Wang, Zenglin Xu, Jiang Duan

**发布时间:** 2025-09-28

### GPT解析

### 总结

论文提出了一种名为InfMasking的新方法，通过无限掩码策略增强多模态表征学习中的协同信息，该方法在七个基准测试上取得了最先进的性能。

### 背景

多模态表征学习中，模态间的协同交互不仅能提供互补信息，还能通过特定交互模式产生单一模态无法实现的独特结果。现有方法难以有效捕捉全面的协同信息，导致在依赖此类交互的任务中表现不佳，而协同信息构成了多模态表征的基本价值。

### 目的

解决现有方法难以有效捕捉全面协同信息的问题，提出一种增强模态间协同信息的方法。

### 方法

引入InfMasking，一种对比协同信息提取方法，使用无限掩码策略增强协同信息。在融合过程中随机遮挡每个模态的大部分特征，只保留部分信息创建不同协同模式的表征，通过互信息最大化对齐未掩码与掩码融合表征以编码全面协同信息。推导出InfMasking损失函数来近似计算无限掩码下的互信息估计。

### 主要发现

InfMasking有效地增强了模态间的协同信息，在七个大规模真实世界数据集的基准测试中取得了最先进的性能。

### 结论

InfMasking是一种有效的多模态表征学习方法，能够增强模态间的协同信息并提高性能，相关代码已在GitHub上发布。

### 翻译

多模态表征学习中，模态间的协同交互不仅能提供互补信息，还能通过特定的交互模式产生单一模态无法实现的独特结果。现有方法可能难以有效捕捉全面的协同信息，导致在依赖此类交互的任务中表现不佳。这尤其成问题，因为协同信息构成了多模态表征的基本价值主张。为了应对这一挑战，我们引入了InfMasking，一种对比协同信息提取方法，旨在通过'无限掩码'策略增强协同信息。InfMasking在融合过程中随机遮挡每个模态的大部分特征，只保留部分信息以创建具有不同协同模式的表征。然后通过互信息最大化将未掩码的融合表征与掩码表征对齐，以编码全面的协同信息。这种无限掩码策略通过在训练过程中暴露模型到多样化的部分模态组合，使模型能够捕捉更丰富的交互。由于计算无限掩码下的互信息估计在计算上代价过高，我们推导出InfMasking损失函数来近似这一计算。通过受控实验，我们证明了InfMasking有效地增强了模态间的协同信息。在大型真实世界数据集的评估中，InfMasking在七个基准测试中取得了最先进的性能。代码已在https://github.com/brightest66/InfMasking发布。


### 论文摘要

In multimodal representation learning, synergistic interactions between modalities not only provide complementary information but also create unique outcomes through specific interaction patterns that no single modality could achieve alone. Existing methods may struggle to effectively capture the full spectrum of synergistic information, leading to suboptimal performance in tasks where such interactions are critical. This is particularly problematic because synergistic information constitutes the fundamental value proposition of multimodal representation. To address this challenge, we introduce InfMasking, a contrastive synergistic information extraction method designed to enhance synergistic information through an \textbf{Inf}inite \textbf{Masking} strategy. InfMasking stochastically occludes most features from each modality during fusion, preserving only partial information to create representations with varied synergistic patterns. Unmasked fused representations are then aligned with masked ones through mutual information maximization to encode comprehensive synergistic information. This infinite masking strategy enables capturing richer interactions by exposing the model to diverse partial modality combinations during training. As computing mutual information estimates with infinite masking is computationally prohibitive, we derive an InfMasking loss to approximate this calculation. Through controlled experiments, we demonstrate that InfMasking effectively enhances synergistic information between modalities. In evaluations on large-scale real-world datasets, InfMasking achieves state-of-the-art performance across seven benchmarks. Code is released at https://github.com/brightest66/InfMasking.

---

