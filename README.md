# 今日论文推荐 - 2025-10-28

共 66 篇论文

---

## 1. Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment

**论文链接:** [http://arxiv.org/abs/2510.23224v1](http://arxiv.org/abs/2510.23224v1)

**作者:** Hongyi Wang, Zhengjie Zhu, Jiabo Ma, Fang Wang, Yue Shi, Bo Luo, Jili Wang, Qiuyu Cai, Xiuming Zhang, Yen-Wei Chen, Lanfen Lin, Hao Chen

**发布时间:** 2025-10-27

### GPT解析

### 总结

PathSearch是一个结合细粒度注意力马赛克表示与全局幻灯片嵌入的检索框架，通过视觉语言对比学习对齐，在数字病理学中实现了准确和灵活的幻灯片检索，提高了诊断准确性和观察者一致性。

### 背景

病理切片的快速数字化为临床和研究工作流程中的计算工具开辟了新可能。基于内容的幻灯片检索使病理学家能够识别形态学和语义上相似的病例，支持精确诊断、增强观察者间一致性并协助基于案例的教育。然而，全幻灯片图像的有效检索仍具挑战性，因其具有千兆像素规模且在大量无关内容中捕捉细微语义差异困难。

### 目的

开发一个检索框架来克服全幻灯片图像检索的挑战，实现准确、高效的幻灯片检索功能。

### 方法

提出PathSearch检索框架，统一细粒度注意力马赛克表示与全局幻灯片嵌入，通过视觉语言对比学习对齐。在6,926个幻灯片-报告对语料库上训练，支持两种关键功能：(1)基于马赛克的图像到图像检索；(2)多模态检索，文本查询可直接检索相关幻灯片。

### 主要发现

在四个公共病理数据集和三个内部队列上进行了严格评估，涵盖解剖部位检索、肿瘤亚型分类、肿瘤与非肿瘤鉴别及跨多个器官的分级任务。外部结果显示PathSearch优于传统图像到图像检索框架。多中心读者研究证明在真实临床场景中提高了诊断准确性，增强了信心，并提高了观察者间一致性。

### 结论

PathSearch被确立为数字病理学中可扩展和通用的检索解决方案。

### 翻译

病理切片的快速数字化为临床和研究工作流程中的计算工具开辟了新的可能性。在这些工具中，基于内容的幻灯片检索脱颖而出，使病理学家能够识别形态学和语义上相似的病例，从而支持精确诊断，增强观察者间的一致性，并协助基于案例的教育。然而，由于全幻灯片图像的千兆像素规模以及在大量无关内容中捕捉细微语义差异的困难，全幻灯片图像的有效检索仍然具有挑战性。为了克服这些挑战，我们提出了PathSearch，这是一个检索框架，统一了细粒度注意力马赛克表示与全局幻灯片嵌入，通过视觉语言对比学习对齐。在包含6,926个幻灯片-报告对的语料库上训练，PathSearch捕捉了细粒度形态学线索和高级语义模式，以实现准确和灵活的检索。该框架支持两个关键功能：(1)基于马赛克的图像到图像检索，确保准确和高效的幻灯片研究；(2)多模态检索，文本查询可以直接检索相关幻灯片。PathSearch在四个公共病理数据集和三个内部队列上进行了严格评估，涵盖了解剖部位检索、肿瘤亚型分类、肿瘤与非肿瘤鉴别以及跨乳腺、肺、肾、肝和胃等多种器官的分级任务。外部结果显示，PathSearch优于传统的图像到图像检索框架。多中心读者研究进一步证明，在真实临床场景中，PathSearch提高了病理学家的诊断准确性，增强了信心，并提高了观察者间的一致性。这些结果确立了PathSearch作为数字病理学中可扩展和通用的检索解决方案。


### 论文摘要

The rapid digitization of histopathology slides has opened up new possibilities for computational tools in clinical and research workflows. Among these, content-based slide retrieval stands out, enabling pathologists to identify morphologically and semantically similar cases, thereby supporting precise diagnoses, enhancing consistency across observers, and assisting example-based education. However, effective retrieval of whole slide images (WSIs) remains challenging due to their gigapixel scale and the difficulty of capturing subtle semantic differences amid abundant irrelevant content. To overcome these challenges, we present PathSearch, a retrieval framework that unifies fine-grained attentive mosaic representations with global-wise slide embeddings aligned through vision-language contrastive learning. Trained on a corpus of 6,926 slide-report pairs, PathSearch captures both fine-grained morphological cues and high-level semantic patterns to enable accurate and flexible retrieval. The framework supports two key functionalities: (1) mosaic-based image-to-image retrieval, ensuring accurate and efficient slide research; and (2) multi-modal retrieval, where text queries can directly retrieve relevant slides. PathSearch was rigorously evaluated on four public pathology datasets and three in-house cohorts, covering tasks including anatomical site retrieval, tumor subtyping, tumor vs. non-tumor discrimination, and grading across diverse organs such as breast, lung, kidney, liver, and stomach. External results show that PathSearch outperforms traditional image-to-image retrieval frameworks. A multi-center reader study further demonstrates that PathSearch improves diagnostic accuracy, boosts confidence, and enhances inter-observer agreement among pathologists in real clinical scenarios. These results establish PathSearch as a scalable and generalizable retrieval solution for digital pathology.

---

## 2. MATCH: Task-Driven Code Evaluation through Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.23169v1](http://arxiv.org/abs/2510.23169v1)

**作者:** Marah Ghoummaid, Vladimir Tchuiev, Ofek Glick, Michal Moschkovitz, Dotan Di Castro

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文提出了一种名为MATCH的新型无参考代码评估指标，使用对比学习为代码和自然语言任务描述生成有意义的嵌入，实现相似性评分。

### 背景

AI代码生成日益普及，GitHub Copilot估计生成GitHub上46%的代码。准确评估生成代码与开发者意图的匹配度是重大挑战，传统方法如单元测试难以扩展且成本高，语法相似性指标无法捕捉代码功能，而需要参考代码的指标如CodeBERTScore并不总是可用。

### 目的

解决无参考代码评估的空白，提供一种不依赖参考代码的代码生成质量评估方法。

### 方法

引入MATCH指标，使用对比学习技术为代码和自然语言任务描述生成有意义的嵌入，实现反映生成代码实现任务程度的相似性评分。

### 主要发现

MATCH在多种编程语言上实现了比现有指标与功能正确性和人类偏好更强的相关性。

### 结论

MATCH是一种有效的无参考代码评估指标，能够更好地评估生成代码与任务意图的匹配度。

### 翻译

基于AI的代码生成日益普及，GitHub Copilot估计生成了GitHub上46%的代码。准确评估生成代码与开发者意图的匹配度仍然是一个重大挑战。传统的评估方法，如单元测试，通常难以扩展且成本高昂。语法相似性指标（如BLEU、ROUGE）无法捕捉代码功能，而像CodeBERTScore这样的指标需要参考代码，但参考代码并不总是可用的。为了解决无参考评估的空白，除了ICE-Score等少数替代方案外，本文引入了MATCH，一种新型的无参考指标。MATCH使用对比学习为代码和自然语言任务描述生成有意义的嵌入，实现反映生成代码实现任务程度的相似性评分。我们证明，在多种编程语言上，MATCH比现有指标实现了与功能正确性和人类偏好更强的相关性。


### 论文摘要

AI-based code generation is increasingly prevalent, with GitHub Copilot estimated to generate 46% of the code on GitHub. Accurately evaluating how well generated code aligns with developer intent remains a critical challenge. Traditional evaluation methods, such as unit tests, are often unscalable and costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code functionality, and metrics like CodeBERTScore require reference code, which is not always available. To address the gap in reference-free evaluation, with few alternatives such as ICE-Score, this paper introduces MATCH, a novel reference-free metric. MATCH uses Contrastive Learning to generate meaningful embeddings for code and natural language task descriptions, enabling similarity scoring that reflects how well generated code implements the task. We show that MATCH achieves stronger correlations with functional correctness and human preference than existing metrics across multiple programming languages.

---

## 3. VALA: Learning Latent Anchors for Training-Free and Temporally Consistent

**论文链接:** [http://arxiv.org/abs/2510.22970v1](http://arxiv.org/abs/2510.22970v1)

**作者:** Zhangkai Wu, Xuhui Fan, Zhongyuan Xie, Kaize Shi, Longbing Cao

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文提出了VALA（Variational Alignment for Latent Anchors）变分对齐模块，用于解决现有无需训练的视频编辑方法中帧选择和时间一致性的问题。

### 背景

无需训练的视频编辑技术最近取得了进展，利用预训练的文本到图像扩散模型实现了轻量级和精确的跨帧生成。然而，现有方法通常依赖启发式帧选择来维持DDIM反演过程中的时间一致性，这引入了人工偏差并降低了端到端推理的可扩展性。

### 目的

开发一种自适应选择关键帧并将它们的潜在特征压缩为语义锚点的方法，以实现一致的视频编辑。

### 方法

VALA采用具有对比学习目标的变分框架，将跨帧潜在表示转换为保留内容和时间一致性的压缩潜在锚点。该方法可以完全集成到无需训练的基于文本到图像的视频编辑模型中。

### 主要发现

在真实世界视频编辑基准上的大量实验表明，VALA在反演保真度、编辑质量和时间一致性方面达到了最先进的性能，同时相比之前的方法提供了更高的效率。

### 结论

VALA是一种有效的变分对齐模块，能够解决现有视频编辑方法中帧选择和时间一致性的挑战，提高了视频编辑的质量和效率。

### 翻译

最近无需训练的视频编辑技术的进步使得利用预训练的文本到图像扩散模型实现了轻量级和精确的跨帧生成。然而，现有方法通常依赖启发式帧选择来维持DDIM反演过程中的时间一致性，这引入了人工偏差并降低了端到端推理的可扩展性。在本文中，我们提出了VALA（变分锚点对齐），这是一种变分对齐模块，可以自适应选择关键帧并将它们的潜在特征压缩为语义锚点，以实现一致的视频编辑。为了学习有意义的分配，VALA提出了一个具有对比学习目标的变分框架。因此，它可以将跨帧潜在表示转换为保留内容和时间一致性的压缩潜在锚点。我们的方法可以完全集成到无需训练的基于文本到图像的视频编辑模型中。在真实世界视频编辑基准上的大量实验表明，VALA在反演保真度、编辑质量和时间一致性方面达到了最先进的性能，同时相比之前的方法提供了更高的效率。


### 论文摘要

Recent advances in training-free video editing have enabled lightweight and precise cross-frame generation by leveraging pre-trained text-to-image diffusion models. However, existing methods often rely on heuristic frame selection to maintain temporal consistency during DDIM inversion, which introduces manual bias and reduces the scalability of end-to-end inference. In this paper, we propose~\textbf{VALA} (\textbf{V}ariational \textbf{A}lignment for \textbf{L}atent \textbf{A}nchors), a variational alignment module that adaptively selects key frames and compresses their latent features into semantic anchors for consistent video editing. To learn meaningful assignments, VALA propose a variational framework with a contrastive learning objective. Therefore, it can transform cross-frame latent representations into compressed latent anchors that preserve both content and temporal coherence. Our method can be fully integrated into training-free text-to-image based video editing models. Extensive experiments on real-world video editing benchmarks show that VALA achieves state-of-the-art performance in inversion fidelity, editing quality, and temporal consistency, while offering improved efficiency over prior methods.

---

## 4. Cross-Lingual Sponsored Search via Dual-Encoder and Graph Neural Networks for Context-Aware Query Translation in Advertising Platforms

**论文链接:** [http://arxiv.org/abs/2510.22957v1](http://arxiv.org/abs/2510.22957v1)

**作者:** Ziyang Gao, Yuanliang Qu, Yi Han

**发布时间:** 2025-10-27

### GPT解析

### 总结

AdGraphTrans是一种结合图神经网络的双编码器框架，用于广告中的上下文感知查询翻译，显著提高了跨语言搜索广告的效果。

### 背景

跨语言搜索广告对全球广告平台至关重要，但传统机器翻译方法无法捕捉查询特定的上下文线索，导致语义歧义，影响点击率和转化率。

### 目的

解决传统翻译方法在广告搜索中的局限性，提高跨语言搜索广告的效果。

### 方法

提出AdGraphTrans框架，使用多语言Transformer编码器独立编码用户查询和广告内容，将上下文关系建模为异构图，应用图注意力网络改进嵌入，并通过对比学习对齐嵌入以减少翻译歧义。

### 主要发现

AdGraphTrans在EN-ZH、EN-ES、EN-FR语言对上实现BLEU得分38.9和语义相似度0.83，优于mBERT和M2M-100基线方法；在下游广告检索任务中提高4.67%点击率和1.72%转化率。

### 结论

将基于图的上下文信号与双编码器翻译相结合，为增强广告平台中的跨语言搜索广告提供了强大的解决方案。

### 翻译

跨语言搜索广告对全球广告平台至关重要，用户来自不同语言背景并与多语言广告互动。传统机器翻译方法往往无法捕捉查询特定的上下文线索，导致语义歧义，对点击率和转化率产生负面影响。为应对这一挑战，我们提出了AdGraphTrans，一种结合图神经网络的双编码器新框架，用于广告中的上下文感知查询翻译。具体而言，使用多语言Transformer编码器独立编码用户查询和广告内容，并将上下文关系（如共同点击的广告、用户搜索会话和查询-广告共现）建模为异构图。然后应用图注意力网络利用语义和行为上下文改进嵌入。这些嵌入通过对比学习对齐，以减少翻译歧义。在从Google Ads和Amazon Ads收集的跨语言搜索广告数据集（EN-ZH、EN-ES、EN-FR对）上进行的实验表明，AdGraphTrans显著提高了查询翻译质量，实现BLEU得分38.9和语义相似度0.83，优于mBERT和M2M-100等强基线方法。此外，在下游广告检索任务中，AdGraphTrans比基线方法提高了4.67%的点击率和1.72%的转化率。这些结果证实，将基于图的上下文信号与双编码器翻译相结合，为增强广告平台中的跨语言搜索广告提供了强大的解决方案。


### 论文摘要

Cross-lingual sponsored search is crucial for global advertising platforms, where users from different language backgrounds interact with multilingual ads. Traditional machine translation methods often fail to capture query-specific contextual cues, leading to semantic ambiguities that negatively impact click-through rates (CTR) and conversion rates (CVR). To address this challenge, we propose AdGraphTrans, a novel dual-encoder framework enhanced with graph neural networks (GNNs) for context-aware query translation in advertising. Specifically, user queries and ad contents are independently encoded using multilingual Transformer-based encoders (mBERT/XLM-R), and contextual relations-such as co-clicked ads, user search sessions, and query-ad co-occurrence-are modeled as a heterogeneous graph. A graph attention network (GAT) is then applied to refine embeddings by leveraging semantic and behavioral context. These embeddings are aligned via contrastive learning to reduce translation ambiguity. Experiments conducted on a cross-lingual sponsored search dataset collected from Google Ads and Amazon Ads (EN-ZH, EN-ES, EN-FR pairs) demonstrate that AdGraphTrans significantly improves query translation quality, achieving a BLEU score of 38.9 and semantic similarity (cosine score) of 0.83, outperforming strong baselines such as mBERT and M2M-100. Moreover, in downstream ad retrieval tasks, AdGraphTrans yields +4.67% CTR and +1.72% CVR improvements over baseline methods. These results confirm that incorporating graph-based contextual signals with dual-encoder translation provides a robust solution for enhancing cross-lingual sponsored search in advertising platforms.

---

## 5. Bi-Encoder Contrastive Learning for Fingerprint and Iris Biometrics

**论文链接:** [http://arxiv.org/abs/2510.22937v1](http://arxiv.org/abs/2510.22937v1)

**作者:** Matthew So, Judah Goldfeder, Mark Lis, Hod Lipson

**发布时间:** 2025-10-27

### GPT解析

### 总结

研究挑战了生物特征统计学上不相关的传统假设，证明同一个体的生物特征（特别是虹膜）实际上存在相关性，并使用双编码器网络和深度学习模型验证了这一发现。

### 背景

历史上一直假设个体的生物特征在统计学上是不相关的，这一假设需要被检验。

### 目的

测试个体生物特征统计学上不相关的假设，通过训练双编码器网络进行三种生物特征验证任务。

### 方法

在274名受试者上训练双编码器网络（约10万张指纹图像和7千张虹膜图像），进行三种匹配任务：指纹到指纹匹配、虹膜到虹膜匹配、跨模态指纹到虹膜匹配。使用ResNet-50和Vision Transformer骨干网络构建双编码器架构，最小化来自同一个体的图像之间的对比损失。

### 主要发现

虹膜ResNet架构在虹膜到虹膜匹配中达到91的ROC AUC分数，证明个体的左右虹膜是相关的；指纹模型重现了先前工作提出的正样本内相关性；这是首次尝试使用Vision Transformer进行此类匹配；跨模态匹配仅略高于随机水平。

### 结论

这些发现继续挑战生物特征的独立性假设，研究计划将这项工作扩展到其他生物特征。

### 翻译

历史上一直假设个体的生物特征在统计学上是不相关的。我们通过在三种验证任务上训练双编码器网络来测试这一假设，包括指纹到指纹匹配、虹膜到虹膜匹配，以及使用274名受试者（约10万张指纹和7千张虹膜图像）进行的跨模态指纹到虹膜匹配。我们在双编码器架构中训练了ResNet-50和Vision Transformer骨干网络，以最小化来自同一个体的图像之间的对比损失。虹膜ResNet架构在虹膜到虹膜匹配中达到91的ROC AUC分数，提供了个体左右虹膜相关的明确证据。指纹模型重现了该领域先前工作所提出的正样本内相关性。这是首次尝试使用Vision Transformer进行此类匹配。跨模态匹配仅略高于随机水平，这表明需要更多数据和更复杂的管道才能获得令人信服的结果。这些发现继续挑战生物特征的独立性假设，我们计划将这项工作扩展到其他生物特征。代码可用：https://github.com/MatthewSo/bio_fingerprints_iris。


### 论文摘要

There has been a historic assumption that the biometrics of an individual are statistically uncorrelated. We test this assumption by training Bi-Encoder networks on three verification tasks, including fingerprint-to-fingerprint matching, iris-to-iris matching, and cross-modal fingerprint-to-iris matching using 274 subjects with $\sim$100k fingerprints and 7k iris images. We trained ResNet-50 and Vision Transformer backbones in Bi-Encoder architectures such that the contrastive loss between images sampled from the same individual is minimized. The iris ResNet architecture reaches 91 ROC AUC score for iris-to-iris matching, providing clear evidence that the left and right irises of an individual are correlated. Fingerprint models reproduce the positive intra-subject suggested by prior work in this space. This is the first work attempting to use Vision Transformers for this matching. Cross-modal matching rises only slightly above chance, which suggests that more data and a more sophisticated pipeline is needed to obtain compelling results. These findings continue challenge independence assumptions of biometrics and we plan to extend this work to other biometrics in the future. Code available: https://github.com/MatthewSo/bio_fingerprints_iris.

---

## 6. Semantic-Preserving Cross-Style Visual Reasoning for Robust Multi-Modal Understanding in Large Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.22838v1](http://arxiv.org/abs/2510.22838v1)

**作者:** Aya Nakayama, Brian Wong, Yuji Nishimura, Kaito Tanaka

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了一种名为SP-CSVR的新型框架，用于解决大型视觉-语言模型中的'风格陷阱'问题，实现稳定语义理解和跨风格视觉推理。

### 背景

'风格陷阱'阻碍了大型视觉-语言模型在不同视觉风格下的稳健语义理解，特别是在上下文学习中。现有方法难以有效解耦风格与内容，限制了模型的泛化能力。

### 目的

开发一种能够实现稳定语义理解和自适应跨风格视觉推理的框架，以克服风格陷阱问题。

### 方法

提出语义保持跨风格视觉推理器(SP-CSVR)，包含三个核心组件：跨风格特征编码器(CSFE)用于风格-内容解耦；语义对齐的上下文解码器(SAICD)用于高效的少样本风格适应；自适应语义一致性模块(ASCM)采用多任务对比学习强制跨风格语义不变性。

### 主要发现

在具有挑战性的多风格数据集上，SP-CSVR在视觉描述、视觉问答和上下文风格适应方面达到了最先进的性能。消融研究和泛化分析证实了该方法在增强稳健性、泛化能力和效率方面的有效性。

### 结论

SP-CSVR成功解决了大型视觉-语言模型中的风格陷阱问题，实现了跨风格的稳定语义理解和高效推理。

### 翻译

本文提出了一种名为'语义保持跨风格视觉推理器'(SP-CSVR)的新型框架，旨在解决大型视觉-语言模型(LVLMs)中的'风格陷阱'问题，从而实现稳定的语义理解和自适应的跨风格视觉推理。SP-CSVR集成了跨风格特征编码器(CSFE)用于风格-内容解耦，语义对齐的上下文解码器(SAICD)用于高效的少样本风格适应，以及采用多任务对比学习的自适应语义一致性模块(ASCM)来强制跨风格语义不变性。在具有挑战性的多风格数据集上的广泛实验表明，SP-CSVR在视觉描述、视觉问答和上下文风格适应方面达到了最先进的性能。包括消融研究和泛化分析在内的全面评估证实了SP-CSVR在增强稳健性、泛化能力和效率方面的有效性。


### 论文摘要

The "style trap" poses a significant challenge for Large Vision-Language Models (LVLMs), hindering robust semantic understanding across diverse visual styles, especially in in-context learning (ICL). Existing methods often fail to effectively decouple style from content, hindering generalization. To address this, we propose the Semantic-Preserving Cross-Style Visual Reasoner (SP-CSVR), a novel framework for stable semantic understanding and adaptive cross-style visual reasoning. SP-CSVR integrates a Cross-Style Feature Encoder (CSFE) for style-content disentanglement, a Semantic-Aligned In-Context Decoder (SAICD) for efficient few-shot style adaptation, and an Adaptive Semantic Consistency Module (ASCM) employing multi-task contrastive learning to enforce cross-style semantic invariance. Extensive experiments on a challenging multi-style dataset demonstrate SP-CSVR's state-of-the-art performance across visual captioning, visual question answering, and in-context style adaptation. Comprehensive evaluations, including ablation studies and generalization analysis, confirm SP-CSVR's efficacy in enhancing robustness, generalization, and efficiency across diverse visual styles.

---

## 7. IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.22706v1](http://arxiv.org/abs/2510.22706v1)

**作者:** Hao Li, Zhengyu Zou, Fangfu Liu, Xuanyang Zhang, Fangzhou Hong, Yukang Cao, Yushi Lan, Manyuan Zhang, Gang Yu, Dingwen Zhang, Ziwei Liu

**发布时间:** 2025-10-26

**备注:** https://github.com/lifuguan/IGGT_official

### GPT解析

### 总结

本文提出了InstanceGrounded Geometry Transformer (IGGT)，一个端到端的大型统一transformer，用于统一空间重建和实例级上下文理解的知识，并通过3D一致的对比学习策略仅使用2D视觉输入编码具有几何结构和基于实例聚类的统一表示。

### 背景

人类自然地将3D世界的几何结构和语义内容视为相互交织的维度，但大多数先前方法优先训练大型几何模型进行低级3D重建，将高级空间理解孤立处理，忽视了这两个基本方面间的关键相互作用，导致下游3D理解任务表现不佳。最近的尝试通过简单对齐3D模型与特定语言模型来缓解问题，但限制了感知能力和下游任务适应性。

### 目的

开发一个能够统一几何结构和语义理解的模型，改善3D场景的理解和重建能力，提高在下游任务中的泛化性能。

### 方法

设计了IGGT模型和3D一致的对比学习策略，指导模型仅通过2D视觉输入编码具有几何结构和基于实例聚类的统一表示。同时构建了InsScene-15K数据集，包含高质量的RGB图像、姿态、深度图和3D一致的实例级掩码注释。

### 主要发现

通过统一几何结构和语义理解，可以有效地将2D视觉输入一致提升到具有明显不同对象实例的连贯3D场景，改善3D场景的理解和重建。

### 结论

IGGT模型和InsScene-15K数据集能够有效解决3D场景分析中几何结构和语义理解分离的问题，提高下游3D理解任务的性能。

### 翻译

人类自然地将3D世界的几何结构和语义内容视为交织的维度，使能够连贯准确地理解复杂场景。然而，大多数先前方法优先训练大型几何模型进行低级3D重建，并将高级空间理解孤立处理，忽视了这两个3D场景分析基本方面之间的关键相互作用，从而限制了泛化能力，导致在下游3D理解任务中表现不佳。最近的尝试通过简单地将3D模型与特定语言模型对齐来缓解这一问题，但这限制了感知能力，并限制了下游任务的适应性。在本文中，我们提出了InstanceGrounded Geometry Transformer (IGGT)，一个端到端的大型统一transformer，用于统一空间重建和实例级上下文理解的知识。具体来说，我们设计了一种3D一致的对比学习策略，指导IGGT仅通过2D视觉输入编码具有几何结构和基于实例聚类的统一表示。这种表示支持将2D视觉输入一致提升到具有明显不同对象实例的连贯3D场景。为了促进这一任务，我们进一步构建了InsScene-15K，一个包含高质量RGB图像、姿态、深度图和3D一致的实例级掩码注释的大规模数据集，采用了新颖的数据整理流程。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D几何重建和高级语义理解被分离处理的问题。人类自然将几何结构和语义内容作为交织维度理解3D世界，但当前方法要么优先处理低级几何重建而忽视高级语义理解，要么简单将几何与特定语言模型对齐。这种分离限制了模型泛化能力，导致在下游3D理解任务中表现不佳。解决这个问题对实现接近人类的空间智能理解至关重要，对机器人操作、AR/VR和空间规划等应用具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到人类自然地将几何结构和语义内容作为交织维度理解3D世界，然后指出当前方法的局限性：将几何重建和语义理解分离，或者简单地将几何与语言模型对齐。他们设计了一种新思路：通过联合训练将几何和实例级语义特征耦合，让模型自主学习3D实例级语义与其几何结构之间的关系。该方法借鉴了VGGT的Transformer架构、DINOv2的图像特征提取、DPT的密集预测架构以及SAM2的视频对象分割技术，并在实例空间跟踪中受到SAMPart3D的启发。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过联合训练将几何结构和实例级语义特征耦合，使模型能够自主学习3D实例级语义与其几何结构之间的关系，并使用实例掩码作为桥梁连接统一表示与各种视觉语言模型。整体流程为：1)输入多张RGB图像；2)使用大型统一Transformer编码为统一标记表示；3)通过几何头部和实例头部解码生成几何点图和实例聚类场；4)使用跨模态融合块增强实例特征的空间感知；5)应用3D一致的对比学习策略确保跨视图一致性；6)使用无监督聚类生成实例掩码；7)这些掩码引导视觉语言模型执行下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出IGGT框架，统一几何重建和语义理解；2)设计3D一致的对比学习策略；3)构建InsScene-15K数据集；4)提出实例级场景理解范式。相比之前工作，不同之处在于：不再将几何和语义分离处理，而是通过联合训练实现相互增强；不再绑定特定语言模型，而是通过实例掩码灵活集成各种视觉语言模型；能区分同一类别内的不同对象实例；实现更好的多视图一致性，特别是在大视角变化和复杂场景中。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IGGT通过统一几何重建和语义理解，并引入实例级场景理解范式，实现了高质量、一致的语义3D重建，并能灵活支持多种下游应用。'}


### 论文摘要

Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.

---

## 8. CLEANet: Robust and Efficient Anomaly Detection in Contaminated Multivariate Time Series

**论文链接:** [http://arxiv.org/abs/2510.22619v1](http://arxiv.org/abs/2510.22619v1)

**作者:** Songhan Zhang, Yuanhao Lai, Pengfei Zheng, Boxi Yu, Xiaoying Tang, Qiuai Fu, Pinjia He

**发布时间:** 2025-10-26

### GPT解析

### 总结

本研究提出了一种名为CLEANet的稳健高效的多变量时间序列异常检测框架，用于解决训练数据污染和模型推理效率低下的问题。CLEANet通过抗污染训练框架和轻量级共轭MLP设计，显著提高了检测性能并降低了计算成本。

### 背景

多变量时间序列异常检测对于维护工业系统可靠性至关重要，但现实部署面临两大挑战：训练数据污染（噪声和隐藏异常）和低效的模型推理。现有无监督方法假设训练数据干净，但污染会扭曲学习模式并降低检测准确性；同时，复杂深度模型容易过度拟合到污染数据上且延迟高，限制了实际应用。

### 目的

开发一种稳健且高效的多变量时间序列异常检测框架，能够有效处理训练数据污染问题，避免模型过度拟合，并提高计算效率，从而提升异常检测的准确性和实用性。

### 方法

作者提出了CLEANet框架，包含两个核心组件：1) 抗污染训练框架(CRTF)，通过自适应重建权重策略结合聚类引导的对比学习减轻污染样本影响；2) 轻量级共轭MLP，用于分离时间跨特征依赖关系，避免过度拟合并提高计算效率。

### 主要发现

在五个公共数据集上，CLEANet比十个最先进的基线方法实现了高达73.04%的F1提升和81.28%的运行时间减少。此外，将CRTF集成到三个先进模型中平均获得5.35%的F1提升，证明了其良好的泛化能力。

### 结论

CLEANet框架有效解决了多变量时间序列异常检测中的训练数据污染和模型推理效率问题，通过抗污染训练策略和轻量级模型设计显著提升了检测性能和计算效率，具有良好的实用价值和泛化能力。

### 翻译

多变量时间序列异常检测对于维护工业系统可靠性至关重要，但现实部署受到两个关键挑战的阻碍：训练数据污染（噪声和隐藏异常）和低效的模型推理。现有无监督方法假设训练数据干净，但污染会扭曲学习模式并降低检测准确性。同时，复杂深度模型往往过度拟合到污染数据上并遭受高延迟，限制了实际应用。为解决这些挑战，我们提出了CLEANet，一个在受污染多变量时间序列中稳健且高效的异常检测框架。CLEANet引入了抗污染训练框架(CRTF)，通过自适应重建权重策略结合聚类引导的对比学习减轻污染样本的影响，从而增强稳健性。为进一步避免在污染数据上过度拟合并提高计算效率，我们设计了一个轻量级共轭MLP，用于分离时间跨特征依赖关系。在五个公共数据集上，CLEANet比十个最先进的基线方法实现了高达73.04%的F1提升和81.28%的运行时间减少。此外，将CRTF集成到三个先进模型中平均获得5.35%的F1提升，证实了其强大的泛化能力。


### 论文摘要

Multivariate time series (MTS) anomaly detection is essential for maintaining the reliability of industrial systems, yet real-world deployment is hindered by two critical challenges: training data contamination (noises and hidden anomalies) and inefficient model inference. Existing unsupervised methods assume clean training data, but contamination distorts learned patterns and degrades detection accuracy. Meanwhile, complex deep models often overfit to contamination and suffer from high latency, limiting practical use. To address these challenges, we propose CLEANet, a robust and efficient anomaly detection framework in contaminated multivariate time series. CLEANet introduces a Contamination-Resilient Training Framework (CRTF) that mitigates the impact of corrupted samples through an adaptive reconstruction weighting strategy combined with clustering-guided contrastive learning, thereby enhancing robustness. To further avoid overfitting on contaminated data and improve computational efficiency, we design a lightweight conjugate MLP that disentangles temporal and cross-feature dependencies. Across five public datasets, CLEANet achieves up to 73.04% higher F1 and 81.28% lower runtime compared with ten state-of-the-art baselines. Furthermore, integrating CRTF into three advanced models yields an average 5.35% F1 gain, confirming its strong generalizability.

---

## 9. Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers

**论文链接:** [http://arxiv.org/abs/2510.22555v1](http://arxiv.org/abs/2510.22555v1)

**作者:** Dongyi Liu, Jiangtong Li, Dawei Cheng, Changjun Jiang

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了CP-GBA(Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers)，一种新的可转移图后门攻击方法，通过图提示学习训练通用子图触发器，实现了跨学习范式的有效攻击。

### 背景

图神经网络易受后门攻击，现有触发器生成器结构简单，过度依赖特定特征，局限于单一图学习范式（如图监督学习、图对比学习或图提示学习），导致跨范式转移性差，无法充分利用图数据的复杂结构信息和节点多样性。

### 目的

解决现有触发器生成器的局限性，提高攻击成功率，开发一种能在多种学习范式中有效工作的可转移图后门攻击方法。

### 方法

提出CP-GBA方法，首先从目标图中提炼出紧凑且具有表达力的触发器集合，通过联合强制类感知性、特征丰富度和结构保真度来实现；其次探索了GPL在基于提示的目标下训练这些触发器的理论可转移性，使其能泛化到多样化和未见过的测试时范式。

### 主要发现

CP-GBA在多个真实数据集和防御场景中实现了最先进的攻击成功率，证明了其在不同学习范式间的有效泛化能力。

### 结论

CP-GBA通过利用图提示学习训练通用子图触发器，解决了现有触发器生成器的局限性，提高了攻击成功率，为图神经网络的后门攻击研究提供了新思路。

### 翻译

图神经网络(GNNs)容易受到后门攻击，攻击者植入恶意触发器来操纵模型预测。现有的触发器生成器通常结构简单，过度依赖特定特征，将其限制在单一图学习范式中。这种专门化设计导致触发器在应用于其他学习范式时转移性差。此外，这些简单生成器通常无法利用图数据中的复杂结构信息或节点多样性，限制了攻击成功率。因此，我们提出了CP-GBA，采用图提示学习训练一组通用子图触发器，通过提炼紧凑且具有表达力的触发器集合和探索理论可转移性，实现了在多种学习范式中的有效攻击，并在多个数据集和防御场景中取得了最先进的攻击成功率。


### 论文摘要

Graph Neural Networks(GNNs) are vulnerable to backdoor attacks, where adversaries implant malicious triggers to manipulate model predictions.   Existing trigger generators are often simplistic in structure and overly reliant on specific features, confining them to a single graph learning paradigm, such as graph supervised learning, graph contrastive learning, or graph prompt learning.   This specialized design, which aligns the trigger with one learning objective, results in poor transferability when applied to other learning paradigms.   For instance, triggers generated for the graph supervised learning paradigm perform poorly when tested within graph contrastive learning or graph prompt learning environments.   Furthermore, these simple generators often fail to utilize complex structural information or node diversity within the graph data.   These constraints limit the attack success rates of such methods in general testing scenarios.   Therefore, to address these limitations, we propose Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers(CP-GBA), a new transferable graph backdoor attack that employs graph prompt learning(GPL) to train a set of universal subgraph triggers.   First, we distill a compact yet expressive trigger set from target graphs, which is structured as a queryable repository, by jointly enforcing class-awareness, feature richness, and structural fidelity.   Second, we conduct the first exploration of the theoretical transferability of GPL to train these triggers under prompt-based objectives, enabling effective generalization to diverse and unseen test-time paradigms.   Extensive experiments across multiple real-world datasets and defense scenarios show that CP-GBA achieves state-of-the-art attack success rates.

---

## 10. PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding

**论文链接:** [http://arxiv.org/abs/2510.22264v1](http://arxiv.org/abs/2510.22264v1)

**作者:** Iliass Ayaou, Denis Cavallucci

**发布时间:** 2025-10-25

### GPT解析

### 总结

论文提出了PatenTEB，一个全面的专利文本嵌入基准测试，包含15个任务，跨越检索、分类、释义和聚类领域，共206万个示例。同时开发了patembed模型家族，通过多任务训练，参数量从67M到344M不等，上下文长度最高可达4096个token。外部验证显示patembed-base在MTEB BigPatentClustering.v2上达到最先进水平（0.494 V-measure vs. 之前的0.445最佳），而patembed-large在DAPFAM上达到0.377 NDCG@100。

### 背景

专利文本嵌入能够实现现有技术搜索、技术景观分析和专利分析，但现有的基准测试无法充分捕捉专利特有的挑战。

### 目的

开发一个能够更好地捕捉专利特定挑战的基准测试和模型，以提高专利文本嵌入的性能和适用性。

### 方法

1. 创建PatenTEB基准测试，包含15个任务，跨越检索、分类、释义和聚类领域；2. 使用领域分层分割、领域特定硬负挖掘和系统覆盖不对称片段到文档匹配场景；3. 开发patembed模型家族，通过多任务训练，参数量从67M到344M；4. 使用领域预训练初始化。

### 主要发现

1. 多任务训练提高了外部泛化能力，尽管对基准测试有轻微影响；2. 领域预训练初始化在任务家族中提供了持续的优势；3. patembed-base在MTEB BigPatentClustering.v2上达到最先进水平（0.494 V-measure）；4. patembed-large在DAPFAM上达到0.377 NDCG@100。

### 结论

PatenTEB基准测试和patembed模型家族能够有效解决专利文本嵌入中的特定挑战，并通过多任务训练和领域预训练初始化实现了更好的性能和泛化能力。

### 翻译

专利检索、句子嵌入、多任务学习、不对称检索、基准测试评估、对比学习


### 论文摘要

Patent text embeddings enable prior art search, technology landscaping, and patent analysis, yet existing benchmarks inadequately capture patent-specific challenges. We introduce PatenTEB, a comprehensive benchmark comprising 15 tasks across retrieval, classification, paraphrase, and clustering, with 2.06 million examples. PatenTEB employs domain-stratified splits, domain specific hard negative mining, and systematic coverage of asymmetric fragment-to-document matching scenarios absent from general embedding benchmarks. We develop the patembed model family through multi-task training, spanning 67M to 344M parameters with context lengths up to 4096 tokens. External validation shows strong generalization: patembed-base achieves state-of-the-art on MTEB BigPatentClustering.v2 (0.494 V-measure vs. 0.445 previous best), while patembed-large achieves 0.377 NDCG@100 on DAPFAM. Systematic ablations reveal that multi-task training improves external generalization despite minor benchmark costs, and that domain-pretrained initialization provides consistent advantages across task families. All resources will be made available at https://github.com/iliass-y/patenteb. Keywords: patent retrieval, sentence embeddings, multi-task learning, asymmetric retrieval, benchmark evaluation, contrastive learning.

---

## 11. Attention Residual Fusion Network with Contrast for Source-free Domain Adaptation

**论文链接:** [http://arxiv.org/abs/2510.22142v1](http://arxiv.org/abs/2510.22142v1)

**作者:** Renrong Shao, Wei Zhang, Jun Wang

**发布时间:** 2025-10-25

**DOI:** 10.1109/TCSVT.2025.3626247

**备注:** 13 pages, 8 figures

### GPT解析

### 总结

论文提出了一种基于对比学习的注意力残差融合网络(ARFNet)框架，用于解决源域无适应(SFDA)中的负迁移和域偏移问题。

### 背景

源域无适应(SFDA)是在源域训练模型后应用于相关目标域，但适应过程中无法访问源数据和标签的任务。场景信息复杂和缺乏源域数据使SFDA成为一项困难任务。现有研究虽取得一定成果，但许多方法只关注域偏移而忽略负迁移的影响。

### 目的

解决SFDA中的负迁移和域偏移问题，提高模型在适应过程中的性能。

### 方法

提出ARFNet框架，利用三种技术：1)注意力机制捕获目标对象的判别区域；2)将注意力特征分解为空间和通道注意力，实现跨层注意力残差融合和自蒸馏；3)对比全局和局部表示，提高类别感知能力；4)动态质心评估策略评估可信质心和标签，用于自监督自蒸馏，减轻域偏移。

### 主要发现

在五个不同规模的基准测试上进行的综合实验表明，该方法优于其他技术，在SFDA基准测试中取得了优越的性能。

### 结论

ARFNet框架有效解决了SFDA中的负迁移和域偏移问题，在多个基准测试上证明了其有效性。

### 翻译

源域无适应(SFDA)涉及在源域训练模型，然后将其应用于相关目标域，但在适应过程中无法访问源数据和标签。场景信息复杂和缺乏源域数据使SFDA成为一项困难任务。最近研究显示出有希望的结果，但许多域适应方法集中在域偏移上，而忽略了负迁移的影响，这可能阻碍模型在适应过程中的性能提升。在本文中，针对这一问题，我们提出了一个基于对比学习的注意力残差融合网络(ARFNet)框架，用于SFDA，以减轻适应过程中的负迁移和域偏移，其中利用了注意力残差融合、全局-局部注意力对比和动态质心评估。具体来说，首先利用注意力机制捕获目标对象的判别区域。然后，在每个块中，注意力特征被分解为空间注意力和通道注意力，以逐步实现跨层注意力残差融合和自蒸馏。在适应过程中，我们对比全局和局部表示，以提高不同类别的感知能力，使模型能够区分类内和类间变化。最后，利用动态质心评估策略评估可信质心和标签，用于自监督自蒸馏，旨在准确近似源域中心和伪标签，以减轻域偏移。为了验证有效性，我们在五个不同规模的基准上进行了综合实验。实验结果表明，我们的方法优于其他技术，在SFDA基准测试中取得了优越的性能。


### 论文摘要

Source-free domain adaptation (SFDA) involves training a model on source domain and then applying it to a related target domain without access to the source data and labels during adaptation. The complexity of scene information and lack of the source domain make SFDA a difficult task. Recent studies have shown promising results, but many approaches to domain adaptation concentrate on domain shift and neglect the effects of negative transfer, which may impede enhancements of model performance during adaptation. n this paper, addressing this issue, we propose a novel framework of Attention Residual Fusion Network (ARFNet) based on contrast learning for SFDA to alleviate negative transfer and domain shift during the progress of adaptation, in which attention residual fusion, global-local attention contrast, and dynamic centroid evaluation are exploited. Concretely, the attention mechanism is first exploited to capture the discriminative region of the target object. Then, in each block, attention features are decomposed into spatial-wise and channel-wise attentions to achieve the cross-layer attention residual fusion progressively and self-distillation. During adaptation progress, we contrast global and local representations to improve the perceptual capabilities of different categories, which enables the model to discriminate variations between inner-class and intra-class. Finally, a dynamic centroid evaluation strategy is exploited to evaluate the trustworthy centroids and labels for self-supervised self-distillation, which aims to accurately approximate the center of the source domain and pseudo-labels to mitigate domain shift. To validate the efficacy, we execute comprehensive experiments on five benchmarks of varying scales. Experimental outcomes indicate that our method surpasses other techniques, attaining superior performance across SFDA benchmarks.

---

## 12. LOC: A General Language-Guided Framework for Open-Set 3D Occupancy Prediction

**论文链接:** [http://arxiv.org/abs/2510.22141v1](http://arxiv.org/abs/2510.22141v1)

**作者:** Yuhang Gao, Xiang Xiang, Sheng Zhong, Guoyou Wang

**发布时间:** 2025-10-25

### GPT解析

### 总结

这篇论文提出了LOC框架，一种用于3D场景理解的视觉语言模型方法，通过密集对比学习增强开放集识别能力。

### 背景

视觉语言模型在开放集挑战中取得了显著进展，但3D数据集的有限可用性限制了它们在3D场景理解中的有效应用。

### 目的

开发一个通用的语言引导框架，适应各种占据网络，支持监督和自监督学习范式，以改善VLMs在3D场景理解中的应用。

### 方法

提出LOC框架，融合多帧LiDAR点，使用泊松重建填补空洞，通过KNN分配体素语义，引入DCL缓解特征过度同质化，预测嵌入CLIP特征空间的密集体素特征。

### 主要发现

在nuScenes数据集上的实验表明，该方法对已知类别实现了高精度预测，能够区分未知类别而无需额外训练数据。

### 结论

LOC框架有效解决了VLMs在3D场景理解中的应用限制，通过密集对比学习增强了开放集识别能力，同时支持监督和自监督学习。

### 翻译

视觉语言模型在开放集挑战中已显示出显著进展。然而，3D数据集的有限可用性阻碍了它们在3D场景理解中的有效应用。我们提出了LOC，一个通用的语言引导框架，可适应各种占据网络，支持监督和自监督学习范式。对于自监督任务，我们采用了一种融合多帧LiDAR点以处理动态/静态场景的策略，使用泊松重建填补空洞，并通过K近邻为体素分配语义，以获得全面的体素表示。为了缓解直接高维特征蒸馏导致的特征过度同质化问题，我们引入了密集对比学习。DCL利用密集体素语义信息和预定义的文本提示，有效增强了开放集识别能力，无需密集像素级监督，我们的框架还可以利用现有真实数据进一步改善性能。我们的模型预测嵌入在CLIP特征空间中的密集体素特征，整合文本和图像像素信息，并基于文本和语义相似性进行分类。在nuScenes数据集上的实验证明了该方法的优越性能，对已知类别实现了高精度预测，并能区分未知类别而无需额外训练数据。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决开放集3D占用预测问题，即让模型能够识别训练数据中未包含的新物体类别。这个问题在自动驾驶等领域非常重要，因为现实世界中物体种类繁多，训练数据无法覆盖所有可能的物体类别，系统需要能够识别未知物体以确保安全。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D占用预测方法的局限性，即它们只能识别已知类别。然后借鉴了视觉语言模型(如CLIP)的知识，利用它们在大量图像-文本对上训练的优势。同时采用了多帧LiDAR点云融合、Poisson重建和KNN等技术来处理3D数据的稀疏性问题。最终设计了LOC框架，结合了监督学习和自监督学习，并通过密集对比学习解决了特征过度同质化的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练的视觉语言模型的丰富语义知识，通过密集对比学习将2D空间的知识转移到3D空间，增强模型对未知类别的识别能力。整体流程包括：1)将2D图像特征投影到3D体素空间；2)使用占用头预测体素状态；3)通过语言头将体素特征映射到文本嵌入空间；4)应用鲁棒密集化策略生成密集3D表示；5)使用密集对比学习对齐体素特征与文本提示；6)结合两个头的输出实现开放集预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)LOC框架，首个用于开放集3D占用预测的语言引导框架；2)密集对比学习(DCL)，有效增强开放集识别能力并避免特征过度同质化；3)鲁棒密集化策略，生成高质量密集3D占用表示。相比之前的工作，LOC能够同时处理已知和未知类别，而传统方法只能识别已知类别；LOC避免了直接高维特征蒸馏的问题；LOC支持监督和自监督学习，能更好地利用有限标注数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LOC是一个通用的语言引导框架，通过密集对比学习将2D视觉语言模型的知识有效转移到3D空间，实现了对已知类别的高精度预测和对未知类别的有效区分，无需额外训练数据。'}


### 论文摘要

Vision-Language Models (VLMs) have shown significant progress in open-set challenges. However, the limited availability of 3D datasets hinders their effective application in 3D scene understanding. We propose LOC, a general language-guided framework adaptable to various occupancy networks, supporting both supervised and self-supervised learning paradigms. For self-supervised tasks, we employ a strategy that fuses multi-frame LiDAR points for dynamic/static scenes, using Poisson reconstruction to fill voids, and assigning semantics to voxels via K-Nearest Neighbor (KNN) to obtain comprehensive voxel representations. To mitigate feature over-homogenization caused by direct high-dimensional feature distillation, we introduce Densely Contrastive Learning (DCL). DCL leverages dense voxel semantic information and predefined textual prompts. This efficiently enhances open-set recognition without dense pixel-level supervision, and our framework can also leverage existing ground truth to further improve performance. Our model predicts dense voxel features embedded in the CLIP feature space, integrating textual and image pixel information, and classifies based on text and semantic similarity. Experiments on the nuScenes dataset demonstrate the method's superior performance, achieving high-precision predictions for known classes and distinguishing unknown classes without additional training data.

---

## 13. Towards Low-Latency and Adaptive Ransomware Detection Using Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.21957v1](http://arxiv.org/abs/2510.21957v1)

**作者:** Zhixin Pan, Ziyu Shu, Amberbir Alemayoh

**发布时间:** 2025-10-24

**备注:** This paper was accepted in the 2025 IEEE International Conference on  Computer Design (ICCD)

### GPT解析

### 总结

本文提出了一种结合自监督对比学习和神经架构搜索的框架，用于解决勒索软件检测中的三大局限性，实现了更高的检测准确性和更快的响应时间。

### 背景

勒索软件已成为网络安全的严重威胁，因其快速演变、需要早期检测且多样性增加，对传统检测方法构成重大挑战。

### 目的

解决现有AI勒索软件检测方法的三大局限性：特征依赖性、响应延迟和对未知变种的适应性有限。

### 方法

提出一个结合自监督对比学习和神经架构搜索的框架，具体包括：(1)设计结合硬件性能计数器的对比学习框架分析勒索软件运行时行为；(2)引入自定义损失函数实现早期检测减少延迟；(3)部署神经架构搜索框架自动构建自适应模型架构。

### 主要发现

实验结果表明，与现有方法相比，提出的方法在检测准确性上提升高达16.1%，响应时间改善高达6倍，并在规避攻击下保持鲁棒性。

### 结论

所提出的方法在检测准确性、响应时间和鲁棒性方面均优于现有方法。

### 翻译

勒索软件由于其快速演变、早期检测的必要性和日益增长的多样性，已成为网络安全的关键威胁，对传统检测方法构成了重大挑战。虽然先前的研究提出了基于人工智能的方法来辅助勒索软件检测，但现有方法存在三个主要局限性：特定的特征依赖性、响应延迟以及对未见变种的适应性有限。在本文中，我们提出了一种结合自监督对比学习和神经架构搜索(NAS)的框架来解决这些挑战。具体来说，本文提供了三个重要贡献：(1)我们设计了一个结合硬件性能计数器(HPC)的对比学习框架，用于分析目标勒索软件的运行时行为。(2)我们引入了一个自定义的损失函数，鼓励对恶意活动的早期检测，并显著减少了检测延迟。(3)我们部署了一个神经架构搜索(NAS)框架，自动构建自适应的模型架构，使检测器能够灵活地与未见的勒索软件变种保持一致。实验结果表明，与现有方法相比，我们提出的方法在检测准确性(高达16.1%)和响应时间(高达6倍)方面都有显著提高，同时在规避攻击下保持鲁棒性。


### 论文摘要

Ransomware has become a critical threat to cybersecurity due to its rapid evolution, the necessity for early detection, and growing diversity, posing significant challenges to traditional detection methods. While AI-based approaches had been proposed by prior works to assist ransomware detection, existing methods suffer from three major limitations, ad-hoc feature dependencies, delayed response, and limited adaptability to unseen variants. In this paper, we propose a framework that integrates self-supervised contrastive learning with neural architecture search (NAS) to address these challenges. Specifically, this paper offers three important contributions. (1) We design a contrastive learning framework that incorporates hardware performance counters (HPC) to analyze the runtime behavior of target ransomware. (2) We introduce a customized loss function that encourages early-stage detection of malicious activity, and significantly reduces the detection latency. (3) We deploy a neural architecture search (NAS) framework to automatically construct adaptive model architectures, allowing the detector to flexibly align with unseen ransomware variants. Experimental results show that our proposed method achieves significant improvements in both detection accuracy (up to 16.1%) and response time (up to 6x) compared to existing approaches while maintaining robustness under evasive attacks.

---

## 14. Learning Neural Observer-Predictor Models for Limb-level Sampling-based Locomotion Planning

**论文链接:** [http://arxiv.org/abs/2510.22789v1](http://arxiv.org/abs/2510.22789v1)

**作者:** Abhijeet M. Kulkarni, Ioannis Poulakakis, Guoquan Huang

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了一种基于学习的观察器-预测器框架，用于准确预测足式机器人的全身运动，解决了简化运动学模型无法捕捉复杂闭环动力学的问题。该系统通过神经观察器提供可靠状态估计，并使用高效预测器评估潜在轨迹，在四足机器人上成功实现了肢体感知的运动规划。

### 背景

准确的全身运动预测对足式机器人的安全自主导航至关重要，特别是在复杂环境中进行肢体级碰撞检查。简化的运动学模型往往无法捕捉机器人和其底层控制器的复杂闭环动力学，导致预测仅限于简单的平面运动。

### 目的

开发一种能够准确预测足式机器人复杂全身运动的框架，克服传统简化模型的局限性。

### 方法

提出一个基于学习的观察器-预测器框架，包含：1)具有可证明UUB保证的神经观察器，从本体感觉测量历史中提供可靠的潜在状态估计；2)计算高效的预测器，能够快速并行评估数千条潜在轨迹；3)将系统集成到基于MPPI的规划器中，并在Vision 60四足机器人上进行了硬件实验验证。

### 主要发现

硬件实验成功展示了系统在具有挑战性的狭窄通道和小物体上的有效肢体感知运动规划能力，证明该系统为动态机器人平台上的高性能、碰撞感知规划提供了稳健基础。

### 结论

所提出的基于学习的观察器-预测器框架能够准确预测足式机器人的全身运动，为复杂环境中的安全自主导航和碰撞感知规划提供了有效解决方案。

### 翻译

准确的全身运动预测对足式机器人的安全自主导航至关重要，能够实现如杂乱环境中肢体级碰撞检查等关键功能。简化的运动学模型往往无法捕捉机器人和其底层控制器的复杂闭环动力学，限制了它们仅能预测简单的平面运动。为此，我们提出了一种基于学习的观察器-预测器框架，能够准确预测这种运动。我们的方法特点是一个具有可证明UUB保证的神经观察器，它从本体感觉测量历史中提供可靠的潜在状态估计。这个稳定的估计初始化了一个计算高效的预测器，专为现代采样规划器所需的大量潜在轨迹的快速并行评估而设计。我们通过将神经预测器集成到Vision 60四足机器人的基于MPPI的规划器中验证了该系统。硬件实验成功展示了在具有挑战性的狭窄通道和小物体上的有效肢体感知运动规划，突显了我们系统为动态机器人平台上的高性能、碰撞感知规划提供稳健基础的能力。


### 论文摘要

Accurate full-body motion prediction is essential for the safe, autonomous navigation of legged robots, enabling critical capabilities like limb-level collision checking in cluttered environments. Simplified kinematic models often fail to capture the complex, closed-loop dynamics of the robot and its low-level controller, limiting their predictions to simple planar motion. To address this, we present a learning-based observer-predictor framework that accurately predicts this motion. Our method features a neural observer with provable UUB guarantees that provides a reliable latent state estimate from a history of proprioceptive measurements. This stable estimate initializes a computationally efficient predictor, designed for the rapid, parallel evaluation of thousands of potential trajectories required by modern sampling-based planners. We validated the system by integrating our neural predictor into an MPPI-based planner on a Vision 60 quadruped. Hardware experiments successfully demonstrated effective, limb-aware motion planning in a challenging, narrow passage and over small objects, highlighting our system's ability to provide a robust foundation for high-performance, collision-aware planning on dynamic robotic platforms.

---

## 15. AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing

**论文链接:** [http://arxiv.org/abs/2510.19661v2](http://arxiv.org/abs/2510.19661v2)

**作者:** Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang

**发布时间:** 2025-10-22

**备注:** 13 pages, 10 pages

### GPT解析

### 总结

AgentSense是一种混合的、无需训练的框架，将大型语言模型集成到参与式城市感知中，通过多智能体进化系统适应动态城市条件，提供自然语言解释以提高透明度。

### 背景

基于网络的参与式城市感知已成为现代城市管理的重要方法，利用移动个体作为分布式传感器。然而，现有系统在跨不同城市场景的泛化能力有限，且在决策过程中可解释性差。

### 目的

解决现有城市感知系统的局限性，提高系统的泛化能力和可解释性。

### 方法

AgentSense框架将大型语言模型集成到参与式城市感知中，通过多智能体进化系统实现。它首先使用经典规划器生成基线解决方案，然后迭代改进这些解决方案以适应动态城市条件和异构工人偏好，同时生成自然语言解释以提高透明度和信任度。

### 主要发现

在两个大规模移动数据集和七种动态干扰的广泛实验中，AgentSense在适应性和可解释性方面明显优于传统方法。与单一智能体LLM基线相比，该方法在性能和鲁棒性方面表现更好，并提供更合理和透明的解释。

### 结论

AgentSense是部署自适应和可解释的网络城市感知系统的重要进展。

### 翻译

基于网络的参与式城市感知已通过利用移动个体作为分布式传感器，成为现代城市管理的重要方法。然而，现有的城市感知系统在跨不同城市场景的泛化能力有限，且在决策过程中的可解释性较差。在这项工作中，我们介绍了AgentSense，这是一种混合的、无需训练的框架，通过多智能体进化系统将大型语言模型（LLMs）集成到参与式城市感知中。AgentSense首先使用经典规划器生成基线解决方案，然后迭代改进这些解决方案，以使感知任务分配适应动态城市条件和异构工人偏好，同时生成自然语言解释以提高透明度和信任度。在两个大规模移动数据集和七种动态干扰的广泛实验中证明，与传统方法相比，AgentSense在适应性和可解释性方面具有明显优势。此外，与单一智能体LLM基线相比，我们的方法在性能和鲁棒性方面都更优，并提供更合理和透明的解释。这些结果表明，AgentSense是向网络部署自适应和可解释的城市感知系统迈进的重要进展。


### 论文摘要

Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.

---

## 16. CURVETE: Curriculum Learning and Progressive Self-supervised Training for Medical Image Classification

**论文链接:** [http://arxiv.org/abs/2510.23442v1](http://arxiv.org/abs/2510.23442v1)

**作者:** Asmaa Abbas, Mohamed Gaber, Mohammed M. Abdelsamea

**发布时间:** 2025-10-27

**备注:** Accepted for publication in the proceedings of ICONIP 2025

### GPT解析

### 总结

CURVETE是一种创新的深度卷积神经网络，通过课程学习和类别分解方法解决了医学图像分析中的样本有限和类别分布不规则的挑战，在各种医学图像数据集上表现出优越的分类性能。

### 背景

在医学图像分析中，识别高质量且易于获取的标注样本是一个显著挑战。迁移学习技术利用预训练数据为这一问题提供了灵活的解决方案。然而，当数据集在类别间呈现不规则分布时，微调的效果会减弱。

### 目的

提出一种名为课程学习和渐进式自监督训练(CURVETE)的新型深度卷积神经网络，解决与样本有限相关的挑战，增强模型泛化能力，并提高整体分类性能。

### 方法

CURVETE采用基于样本分解粒度的课程学习策略，在训练通用未标记样本时使用；在下游任务中整合类别分解方法，解决类别分布不规则的挑战；在脑肿瘤、数字膝盖X光和Mini-DDSM三个医学图像数据集上进行评估，研究了使用通用自监督样本分解方法的分类性能，包括和不包括课程学习组件。

### 主要发现

CURVETE模型在测试集上实现了优越的性能，使用基线ResNet-50在脑肿瘤数据集上达到96.60%的准确率，在数字膝盖X光数据集上达到75.60%，在Mini-DDSM数据集上达到93.35%；使用基线DenseNet-121，在三个数据集上分别达到95.77%、80.36%和93.22%的准确率，优于其他训练策略。

### 结论

CURVETE模型能够有效解决医学图像分析中的样本有限和类别分布不规则的挑战，通过课程学习和渐进式自监督训练，显著提高了分类性能。

### 翻译

在医学图像分析中，识别高质量且易于获取的标注样本是一个显著挑战。迁移学习技术利用预训练数据为这一问题提供了灵活的解决方案。然而，当数据集在类别间呈现不规则分布时，微调的效果会减弱。本文提出了一种名为课程学习和渐进式自监督训练(CURVETE)的新型深度卷积神经网络。CURVETE通过在训练通用未标记样本时采用基于样本分解粒度的课程学习策略，解决了与样本有限相关的挑战，增强了模型泛化能力，并提高了整体分类性能。此外，CURVETE通过在下游任务中整合类别分解方法，解决了类别分布不规则的挑战。该方法在三个不同的医学图像数据集上进行了评估：脑肿瘤、数字膝盖X光和Mini-DDSM数据集。我们研究了使用通用自监督样本分解方法进行分类性能，包括和不包括在训练预任务中使用课程学习组件。实验结果表明，CURVETE模型在测试集上实现了优越的性能，使用基线ResNet-50在脑肿瘤数据集上达到96.60%的准确率，在数字膝盖X光数据集上达到75.60%，在Mini-DDSM数据集上达到93.35%。此外，使用基线DenseNet-121，在脑肿瘤、数字膝盖X光和Mini-DDSM数据集上分别达到95.77%、80.36%和93.22%的准确率，优于其他训练策略。


### 论文摘要

Identifying high-quality and easily accessible annotated samples poses a notable challenge in medical image analysis. Transfer learning techniques, leveraging pre-training data, offer a flexible solution to this issue. However, the impact of fine-tuning diminishes when the dataset exhibits an irregular distribution between classes. This paper introduces a novel deep convolutional neural network, named Curriculum Learning and Progressive Self-supervised Training (CURVETE). CURVETE addresses challenges related to limited samples, enhances model generalisability, and improves overall classification performance. It achieves this by employing a curriculum learning strategy based on the granularity of sample decomposition during the training of generic unlabelled samples. Moreover, CURVETE address the challenge of irregular class distribution by incorporating a class decomposition approach in the downstream task. The proposed method undergoes evaluation on three distinct medical image datasets: brain tumour, digital knee x-ray, and Mini-DDSM datasets. We investigate the classification performance using a generic self-supervised sample decomposition approach with and without the curriculum learning component in training the pretext task. Experimental results demonstrate that the CURVETE model achieves superior performance on test sets with an accuracy of 96.60% on the brain tumour dataset, 75.60% on the digital knee x-ray dataset, and 93.35% on the Mini-DDSM dataset using the baseline ResNet-50. Furthermore, with the baseline DenseNet-121, it achieved accuracies of 95.77%, 80.36%, and 93.22% on the brain tumour, digital knee x-ray, and Mini-DDSM datasets, respectively, outperforming other training strategies.

---

## 17. DREaM: Drug-Drug Relation Extraction via Transfer Learning Method

**论文链接:** [http://arxiv.org/abs/2510.23189v1](http://arxiv.org/abs/2510.23189v1)

**作者:** Ali Fata, Hossein Rahmani, Parinaz Soltanzadeh, Amirhossein Derakhshan, Behrouz Minaei Bidgoli

**发布时间:** 2025-10-27

### GPT解析

### 总结

本研究提出了一种名为DREAM的方法，用于药物关系抽取，通过结合关系抽取模型和大型语言模型构建药物关系本体并验证结果。

### 背景

药物关系抽取对识别药物相互作用和预测副作用至关重要。机器学习方法和大型医学文本数据库的发展降低了关系抽取成本，但目前缺乏专门针对药物关系抽取的数据集。

### 目的

由于缺乏专业数据集，需要采用迁移学习方法来应用机器学习技术进行药物关系抽取，并构建药物关系本体。

### 方法

DREAM方法首先使用训练好的关系抽取模型发现实体间关系，然后将模型应用于医学文本语料库构建药物关系本体，最后使用大型语言模型验证抽取的关系。

### 主要发现

定量结果显示，大型语言模型同意从PubMed摘要子集中提取的71个关系。定性分析表明该方法能揭示医学领域的模糊性，突显了关系抽取的挑战。

### 结论

通过迁移学习和大型语言模型验证，DREAM方法能有效提取药物关系并构建药物关系本体，同时揭示了医学领域中关系抽取的固有挑战。

### 翻译

药物之间的关系抽取在识别药物-药物相互作用和预测副作用方面起着至关重要的作用。机器学习方法在关系抽取方面的进步，以及大型医学文本数据库的发展，使得与其他通常需要专业知识的方法相比，这种关系的提取成本更低。然而，据我们所知，目前专门用于药物关系抽取的数据集非常有限。因此，采用迁移学习成为在该领域应用机器学习方法的必要手段。在本研究中，我们提出了DREAM方法，该方法首先使用训练好的关系抽取模型发现实体间的关系，然后将该模型应用于医学文本语料库以构建药物关系本体。随后使用大型语言模型验证抽取的关系。定量结果表明，大型语言模型同意从PubMed摘要子集中提取的71个关系。此外，我们的定性分析表明，这种方法可以揭示医学领域中的模糊性，突显了该领域关系抽取的固有挑战。


### 论文摘要

Relation extraction between drugs plays a crucial role in identifying drug drug interactions and predicting side effects. The advancement of machine learning methods in relation extraction, along with the development of large medical text databases, has enabled the low cost extraction of such relations compared to other approaches that typically require expert knowledge. However, to the best of our knowledge, there are limited datasets specifically designed for drug drug relation extraction currently available. Therefore, employing transfer learning becomes necessary to apply machine learning methods in this domain. In this study, we propose DREAM, a method that first employs a trained relation extraction model to discover relations between entities and then applies this model to a corpus of medical texts to construct an ontology of drug relationships. The extracted relations are subsequently validated using a large language model. Quantitative results indicate that the LLM agreed with 71 of the relations extracted from a subset of PubMed abstracts. Furthermore, our qualitative analysis indicates that this approach can uncover ambiguities in the medical domain, highlighting the challenges inherent in relation extraction in this field.

---

## 18. LightPFP: A Lightweight Route to Ab Initio Accuracy at Scale

**论文链接:** [http://arxiv.org/abs/2510.23064v1](http://arxiv.org/abs/2510.23064v1)

**作者:** Wenwen Li, Nontawat Charoenphakdee, Yong-Bin Zhuang, Ryuhei Okuno, Yuta Tsuboi, So Takamoto, Junichi Ishida, Ju Li

**发布时间:** 2025-10-27

**备注:** 15 pages, 10 figures

### GPT解析

### 总结

LightPFP是一种数据高效的知识蒸馏框架，利用通用机器学习原子势(u-MLIP)生成针对特定材料的高质量训练数据，结合预训练轻量级MLIP提高效率，实现了比传统DFT方法快三个数量级的模型开发速度，同时保持与第一性原理相当的准确性，且生成的特定任务MLIP(ts-MLIP)在保持高精度的同时实现了1-2个数量级的推理速度提升。

### 背景

原子模拟方法已从量子力学发展到密度泛函理论(DFT)，再到机器学习原子势(MLIPs)。通用MLIPs(u-MLIPs)具有良好的可转移性但计算开销大，限制了大规模应用；特定任务MLIPs(ts-MLIPs)效率更高但为每个材料系统生成DFT数据的成本极高。

### 目的

提出一种数据高效的知识蒸馏框架LightPFP，解决传统方法中DFT计算成本高的问题，实现快速开发高精度、高效的特定任务MLIPs。

### 方法

LightPFP框架利用u-MLIP生成针对特定材料的高质量训练数据，并使用预训练的轻量级MLIP进一步提高数据效率，通过知识蒸馏技术生成ts-MLIP，同时支持高效的精度迁移学习。

### 主要发现

LightPFP比传统基于DFT的方法快三个数量级的模型开发速度，同时保持与第一性原理预测相当的准确性；蒸馏出的ts-MLIP比u-MLIP快1-2个数量级的推理速度；仅需10个高精度DFT数据点即可校正u-MLIP的系统误差。

### 结论

这种u-MLIP驱动的蒸馏方法能够为材料科学应用快速开发高保真度、高效的MLIPs。

### 翻译

原子模拟方法已经通过连续的计算层级逐步发展，每个层级都建立在更基础的方法之上：从量子力学到密度泛函理论(DFT)，随后发展到机器学习原子势(MLIPs)。虽然通用MLIPs(u-MLIPs)具有广泛的可转移性，但其计算开销限制了大规模应用。特定任务MLIPs(ts-MLIPs)实现了更高的效率，但为每个材料系统生成DFT数据的成本高得令人望而却步。在本文中，我们提出了LightPFP，一种数据高效的知识蒸馏框架。LightPFP不使用昂贵的DFT计算，而是利用u-MLIP生成针对特定材料定制的高质量训练数据，并使用预训练的轻量级MLIP进一步提高数据效率，从而生成蒸馏的ts-MLIP。在包括固态电解质、高熵合金和反应离子系统在内的广泛材料范围内，LightPFP比传统基于DFT的方法快三个数量级的模型开发速度，同时保持与第一性原理预测相当的准确性。此外，蒸馏出的ts-MLs进一步维持了大规模分子动力学计算所必需的计算效率，比u-MLIPs快1-2个数量级的推理速度。该框架还支持高效的精度迁移学习，其中可以使用少至10个高精度DFT数据点来校正u-MLIP的系统误差，如在MgO熔点预测中所演示的。这种u-MLIP驱动的蒸馏方法能够为材料科学应用快速开发高保真度、高效的MLIPs。


### 论文摘要

Atomistic simulation methods have evolved through successive computational levels, each building upon more fundamental approaches: from quantum mechanics to density functional theory (DFT), and subsequently, to machine learning interatomic potentials (MLIPs). While universal MLIPs (u-MLIPs) offer broad transferability, their computational overhead limits large-scale applications. Task-specific MLIPs (ts-MLIPs) achieve superior efficiency but require prohibitively expensive DFT data generation for each material system. In this paper, we propose LightPFP, a data-efficient knowledge distillation framework. Instead of using costly DFT calculations, LightPFP generates a distilled ts-MLIP by leveraging u-MLIP to generate high-quality training data tailored for specific materials and utilizing a pre-trained light-weight MLIP to further enhance data efficiency. Across a broad spectrum of materials, including solid-state electrolytes, high-entropy alloys, and reactive ionic systems, LightPFP delivers three orders of magnitude faster model development than conventional DFT-based methods, while maintaining accuracy on par with first-principles predictions. Moreover, the distilled ts-MLIPs further sustain the computational efficiency essential for large-scale molecular dynamics, achieving 1-2 orders of magnitude faster inference than u-MLIPs. The framework further enables efficient precision transfer learning, where systematic errors from the u-MLIP can be corrected using as few as 10 high-accuracy DFT data points, as demonstrated for MgO melting point prediction. This u-MLIP-driven distillation approach enables rapid development of high-fidelity, efficient MLIPs for materials science applications.

---

## 19. TLCD: A Deep Transfer Learning Framework for Cross-Disciplinary Cognitive Diagnosis

**论文链接:** [http://arxiv.org/abs/2510.23062v1](http://arxiv.org/abs/2510.23062v1)

**作者:** Zhifeng Wang, Meixin Su, Yang Yang, Chunyan Zeng, Lizhi Ye

**发布时间:** 2025-10-27

**备注:** 10 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种创新的跨学科认知诊断方法(TLCD)，结合深度学习和迁移学习策略，解决了跨学科领域中认知诊断面临的挑战，提高了对学生学习情况评估的准确性。

### 背景

在线教育模式已成为教育产业的重要组成部分。认知诊断技术可利用学生学习数据评估其能力水平，但跨学科领域存在特征提取复杂性和学科数据稀缺性问题，传统认知诊断方法面临挑战。

### 目的

针对不同学科间知识系统、认知结构和数据特征的差异，研究神经网络认知诊断和知识关联神经网络认知诊断，提出创新的跨学科认知诊断方法。

### 方法

提出跨学科认知诊断方法(TLCD)，结合深度学习技术和迁移学习策略，通过利用主学科的共同特征来提高模型在目标学科中的性能。

### 主要发现

基于深度学习的跨学科认知诊断模型在跨学科认知诊断任务中表现优于基础模型，能够更准确地评估学生的学习情况。

### 结论

跨学科认知诊断方法(TLCD)有效解决了跨学科认知诊断中的挑战，提高了诊断的准确性和性能，对智能教育领域具有重要意义。

### 翻译

受智能教育和人工智能技术的双重驱动，在线教育模式已迅速成为教育产业的重要组成部分。认知诊断技术可以利用教育评估中学生学习的数据和反馈信息，准确评估他们在知识层面的能力水平。然而，大量信息虽然提供了丰富的数据资源，但也带来了特征提取的复杂性和学科数据的稀缺性。在跨学科领域，传统的认知诊断方法仍面临许多挑战。鉴于不同学科之间知识系统、认知结构和数据特征的差异，本文对神经网络认知诊断和知识关联神经网络认知诊断进行了深入研究，并提出了一种创新的跨学科认知诊断方法(TLCD)。该方法结合了深度学习技术和迁移学习策略，通过利用主学科的共同特征来提高模型在目标学科中的性能。实验结果表明，基于深度学习的跨学科认知诊断模型在跨学科认知诊断任务中表现优于基础模型，能够更准确地评估学生的学习情况。


### 论文摘要

Driven by the dual principles of smart education and artificial intelligence technology, the online education model has rapidly emerged as an important component of the education industry. Cognitive diagnostic technology can utilize students' learning data and feedback information in educational evaluation to accurately assess their ability level at the knowledge level. However, while massive amounts of information provide abundant data resources, they also bring about complexity in feature extraction and scarcity of disciplinary data. In cross-disciplinary fields, traditional cognitive diagnostic methods still face many challenges. Given the differences in knowledge systems, cognitive structures, and data characteristics between different disciplines, this paper conducts in-depth research on neural network cognitive diagnosis and knowledge association neural network cognitive diagnosis, and proposes an innovative cross-disciplinary cognitive diagnosis method (TLCD). This method combines deep learning techniques and transfer learning strategies to enhance the performance of the model in the target discipline by utilizing the common features of the main discipline. The experimental results show that the cross-disciplinary cognitive diagnosis model based on deep learning performs better than the basic model in cross-disciplinary cognitive diagnosis tasks, and can more accurately evaluate students' learning situation.

---

## 20. Survey of Multimodal Geospatial Foundation Models: Techniques, Applications, and Challenges

**论文链接:** [http://arxiv.org/abs/2510.22964v1](http://arxiv.org/abs/2510.22964v1)

**作者:** Liling Yang, Ning Chen, Jun Yue, Yidan Liu, Jiayi Ma, Pedram Ghamisi, Antonio Plaza, Leyuan Fang

**发布时间:** 2025-10-27

### GPT解析

### 总结

这篇综述从模态驱动视角对多模态地理空间基础模型(GFMs)进行全面回顾，涵盖五种核心视觉和视觉-语言模态，分析其在遥感图像分析中的应用、挑战和未来发展方向。

### 背景

基础模型已改变自然语言处理和计算机视觉领域，其影响正在重塑遥感图像分析。基础模型强大的泛化和迁移学习能力与遥感数据的多模态、多分辨率和多时态特性自然契合。

### 目的

解决遥感领域的独特挑战，通过多模态地理空间基础模型(GFMs)这一专门研究前沿，提供从模态驱动视角的全面回顾，并分析关键技术、评估模型性能和应用场景。

### 方法

涵盖五种核心视觉和视觉-语言模态，检查成像物理和数据表示差异如何塑造交互设计，分析对齐、集成和知识转移的关键技术，系统评估训练范式、架构和适应策略进展，在十个下游任务上评估代表性模型，并通过真实案例研究展示应用潜力。

### 主要发现

多模态GFMs在土地覆盖制图、农业监测、灾害响应、气候研究和地理空间情报等领域展现实际应用潜力，不同模型在架构、性能和应用场景上存在差异，需要针对模态异构性、分布偏移和语义差距进行优化。

### 结论

领域泛化、可解释性、效率和隐私是GFMs发展面临的紧迫挑战，未来研究需要在这些方面探索有前途的方向，进一步提升模型性能和应用范围。

### 翻译

基础模型已经改变了自然语言处理和计算机视觉，它们的影响现在正在重塑遥感图像分析。凭借强大的泛化和迁移学习能力，它们与遥感数据的多模态、多分辨率和多时态特性自然契合。为解决该领域的独特挑战，多模态地理空间基础模型(GFMs)已成为专门的研究前沿。这篇综述从模态驱动视角对多模态GFMs进行全面回顾，涵盖五种核心视觉和视觉-语言模态。我们检查成像物理和数据表示差异如何塑造交互设计，并分析对齐、集成和知识转移的关键技术，以处理模态异构性、分布偏移和语义差距。系统评估了训练范式、架构和任务特定适应策略的进展，以及大量新兴基准。在十个下游任务上评估了代表性的多模态视觉和视觉-语言GFMs，深入了解它们的架构、性能和应用场景。真实案例研究，涵盖土地覆盖制图、农业监测、灾害响应、气候研究和地理空间情报，展示了GFMs的实际潜力。最后，我们概述了领域泛化、可解释性、效率和隐私方面的紧迫挑战，并为未来研究规划了有前途的方向。


### 论文摘要

Foundation models have transformed natural language processing and computer vision, and their impact is now reshaping remote sensing image analysis. With powerful generalization and transfer learning capabilities, they align naturally with the multimodal, multi-resolution, and multi-temporal characteristics of remote sensing data. To address unique challenges in the field, multimodal geospatial foundation models (GFMs) have emerged as a dedicated research frontier. This survey delivers a comprehensive review of multimodal GFMs from a modality-driven perspective, covering five core visual and vision-language modalities. We examine how differences in imaging physics and data representation shape interaction design, and we analyze key techniques for alignment, integration, and knowledge transfer to tackle modality heterogeneity, distribution shifts, and semantic gaps. Advances in training paradigms, architectures, and task-specific adaptation strategies are systematically assessed alongside a wealth of emerging benchmarks. Representative multimodal visual and vision-language GFMs are evaluated across ten downstream tasks, with insights into their architectures, performance, and application scenarios. Real-world case studies, spanning land cover mapping, agricultural monitoring, disaster response, climate studies, and geospatial intelligence, demonstrate the practical potential of GFMs. Finally, we outline pressing challenges in domain generalization, interpretability, efficiency, and privacy, and chart promising avenues for future research.

---

## 21. Inductive Transfer Learning for Graph-Based Recommenders

**论文链接:** [http://arxiv.org/abs/2510.22799v1](http://arxiv.org/abs/2510.22799v1)

**作者:** Florian Grötschla, Elia Trachsel, Luca A. Lanzendörfer, Roger Wattenhofer

**发布时间:** 2025-10-26

**备注:** Accepted at the New Perspectives in Graph Machine Learning Workshop  at NeurIPS 2025

### GPT解析

### 总结

本文提出了NBF-Rec，一种支持跨不同数据集进行归纳迁移学习的图推荐模型，能够在不重新训练的情况下处理新用户、新项目或新数据集。

### 背景

图推荐系统通常在归纳设置下训练，这限制了它们对新用户、新项目或新数据集的应用。

### 目的

提出一种支持在不同数据集上进行归纳迁移学习的图推荐模型，解决传统方法需要为每个领域重新训练的问题。

### 方法

提出NBF-Rec模型，一种基于图的推荐模型，可以在用户和项目集合不相交的数据集之间进行归纳迁移学习。与传统基于嵌入的方法不同，NBF-Rec在推理时动态计算节点嵌入，无需为每个领域重新训练。

### 主要发现

NBF-Rec在七个真实世界数据集（涵盖电影、音乐、电子商务和地点签到等领域）上进行了评估，在零样本设置下（不使用目标域数据进行训练）取得了具有竞争力的性能，并通过轻量级微调进一步提高了性能。

### 结论

归纳迁移在图推荐中是可行的，交互级别的消息传递支持跨数据集的泛化，而无需对齐用户或项目。

### 翻译

基于图的推荐系统通常在归纳设置下进行训练，这限制了它们对新用户、新项目或新数据集的适用性。我们提出了NBF-Rec，一种基于图的推荐模型，支持在不同用户和项目集合不相交的数据集上进行归纳迁移学习。与需要为每个领域重新训练的传统基于嵌入的方法不同，NBF-Rec在推理时动态计算节点嵌入。我们在七个涵盖电影、音乐、电子商务和地点签到的真实世界数据集上评估了该方法。NBF-Rec在零样本设置下（不使用目标域数据进行训练）取得了具有竞争力的性能，并通过轻量级微调展示了进一步的改进。这些结果表明，归纳迁移在图推荐中是可行的，并且交互级别的消息传递支持跨数据集的泛化，而无需对齐用户或项目。


### 论文摘要

Graph-based recommender systems are commonly trained in transductive settings, which limits their applicability to new users, items, or datasets. We propose NBF-Rec, a graph-based recommendation model that supports inductive transfer learning across datasets with disjoint user and item sets. Unlike conventional embedding-based methods that require retraining for each domain, NBF-Rec computes node embeddings dynamically at inference time. We evaluate the method on seven real-world datasets spanning movies, music, e-commerce, and location check-ins. NBF-Rec achieves competitive performance in zero-shot settings, where no target domain data is used for training, and demonstrates further improvements through lightweight fine-tuning. These results show that inductive transfer is feasible in graph-based recommendation and that interaction-level message passing supports generalization across datasets without requiring aligned users or items.

---

## 22. Qlustering: Harnessing Network-Based Quantum Transport for Data Clustering

**论文链接:** [http://arxiv.org/abs/2510.22727v1](http://arxiv.org/abs/2510.22727v1)

**作者:** Shmuel Lorber, Yonatan Dubi

**发布时间:** 2025-10-26

**备注:** 13 pages

### GPT解析

### 总结

本文介绍了Qlustering，一种受量子启发的无监督学习算法，利用基于网络的量子传输进行数据聚类，在多种数据集上表现出与经典方法相当或更优的性能，特别是在处理非凸或高维数据时。

### 背景

传统聚类方法主要基于距离度量，而量子计算提供了新的计算范式，可以解决传统方法难以处理的问题。

### 目的

开发一种新的量子启发式聚类算法，能够有效处理非凸或高维数据，并具有计算效率和物理可实现性。

### 方法

Qlustering将数据编码为紧束缚哈密顿量框架中的输入状态，通过量子粒子在网络中的传播动力学进行计算，聚类分配从终端节点的稳态输出电流中产生，算法通过迭代优化网络哈密顿量和随机更新实现收敛。

### 主要发现

在合成数据集、定位问题、QM9分子数据库和Iris数据集上，Qlustering与k-means等经典方法相比具有竞争力或更优的性能，特别是在处理非凸或高维数据时表现出色。

### 结论

Qlustering具有内在的鲁棒性、低计算复杂性和与光子实现的兼容性，为构建物理可实现的、量子原生的聚类架构提供了有前途的途径。

### 翻译

我们引入Qlustering，一种用于无监督学习的受量子启发的算法，它利用基于网络的量子传输来执行数据聚类。与传统的基于距离的方法不同，Qlustering将量子粒子通过网络传播的稳态动力学视为计算资源。数据被编码为由Lindblad主方程控制的紧束缚哈密顿量框架中的输入状态，聚类分配从终端节点的稳态输出电流中产生。该算法迭代地优化网络的哈密顿量以最小化物理动机的成本函数，通过随机更新实现收敛。我们在合成数据集、定位问题以及真实的化学和生物数据（即QM9分子数据库和Iris数据集的子集）上对Qlustering进行了基准测试。在这些多样化的任务中，Qlustering展示了与k-means等经典方法相比具有竞争力或更优的性能，特别是对于非凸或高维数据。其内在的鲁棒性、低计算复杂性和与光子实现的兼容性表明，其有望实现物理可实现的、量子原生的聚类架构。


### 论文摘要

We introduce Qlustering, a quantum-inspired algorithm for unsupervised learning that leverages network-based quantum transport to perform data clustering. In contrast to traditional distance-based methods, Qlustering treats the steady-state dynamics of quantum particles propagating through a network as a computational resource. Data are encoded as input states in a tight-binding Hamiltonian framework governed by the Lindblad master equation, and cluster assignments emerge from steady-state output currents at terminal nodes. The algorithm iteratively optimizes the network's Hamiltonian to minimize a physically motivated cost function, achieving convergence through stochastic updates. We benchmark Qlustering on synthetic datasets, a localization problem, and real-world chemical and biological data, namely subsets of the QM9 molecular database and the Iris dataset. Across these diverse tasks, Qlustering demonstrates competitive or superior performance compared with classical methods such as k-means, particularly for non-convex or high-dimensional data. Its intrinsic robustness, low computational complexity, and compatibility with photonic implementations suggest a promising route toward physically realizable, quantum-native clustering architectures.

---

## 23. Cross-Species Transfer Learning in Agricultural AI: Evaluating ZebraPose Adaptation for Dairy Cattle Pose Estimation

**论文链接:** [http://arxiv.org/abs/2510.22618v1](http://arxiv.org/abs/2510.22618v1)

**作者:** Mackenzie Tapp, Sibi Chakravarthy Parivendan, Kashfia Sailunaz, Suresh Neethirajan

**发布时间:** 2025-10-26

**备注:** 20 pages, 11 figures, 6 Tables

### GPT解析

### 总结

本研究评估了使用在斑马图像上训练的ZebraPose模型进行奶牛姿态估计的跨物种迁移学习潜力，发现在不同环境间存在显著泛化挑战。

### 背景

姿态估计是计算机视觉的核心技术，用于理解动物姿态、行为和福利，但农业应用受限于缺乏大型标注的牲畜数据集，特别是奶牛数据集。

### 目的

评估跨物种迁移学习的潜力和局限性，通过将ZebraPose模型适应于谷仓条件下奶牛的27个关键点检测。

### 方法

使用三种配置评估模型：自定义农场数据集（375张图像）、APT-36K基准数据集的子集以及它们的组合，系统评估了模型在不同环境中的准确性和泛化能力。

### 主要发现

组合模型在分布内数据上表现良好，但在未见过的谷仓和奶牛群体上出现显著泛化失败，表明合成到真实域差距是农业AI部署的主要障碍，物种形态相似性不足以实现跨域迁移。

### 结论

研究强调了数据集多样性、环境变化性和计算约束对现实世界部署的影响，呼吁以农业为先的AI设计，优先考虑农场级真实性、跨环境鲁棒性和开放基准数据集。

### 翻译

姿态估计作为计算机视觉的基石，用于理解动物姿态、行为和福利。然而，农业应用仍然受限于大型标注牲畜数据集的稀缺，特别是奶牛。本研究通过将ZebraPose（一种基于视觉变换器的模型，在合成斑马图像上训练）适应于谷仓条件下奶牛的27个关键点检测，评估了跨物种迁移学习的潜力和局限性。使用三种配置——自定义农场数据集（375张图像，加拿大新不伦瑞克州苏塞克斯）、APT-36K基准数据集的子集以及它们的组合，我们系统评估了模型在不同环境中的准确性和泛化能力。虽然组合模型在分布内数据上取得了有希望的性能，但当应用于未见过的谷仓和奶牛群体时，出现了显著的泛化失败。这些发现揭示了合成到真实域差距是农业AI部署的主要障碍，并强调物种间的形态相似性不足以进行跨域迁移。研究提供了关于数据集多样性、环境变化性和计算约束影响现实世界部署的实践见解。我们呼吁以农业为先的AI设计，优先考虑农场级真实性、跨环境鲁棒性和开放基准数据集，以推进可信和可扩展的以动物为中心的技术。


### 论文摘要

Pose estimation serves as a cornerstone of computer vision for understanding animal posture, behavior, and welfare. Yet, agricultural applications remain constrained by the scarcity of large, annotated datasets for livestock, especially dairy cattle. This study evaluates the potential and limitations of cross-species transfer learning by adapting ZebraPose - a vision transformer-based model trained on synthetic zebra imagery - for 27-keypoint detection in dairy cows under real barn conditions. Using three configurations - a custom on-farm dataset (375 images, Sussex, New Brunswick, Canada), a subset of the APT-36K benchmark dataset, and their combination, we systematically assessed model accuracy and generalization across environments. While the combined model achieved promising performance (AP = 0.86, AR = 0.87, PCK 0.5 = 0.869) on in-distribution data, substantial generalization failures occurred when applied to unseen barns and cow populations. These findings expose the synthetic-to-real domain gap as a major obstacle to agricultural AI deployment and emphasize that morphological similarity between species is insufficient for cross-domain transfer. The study provides practical insights into dataset diversity, environmental variability, and computational constraints that influence real-world deployment of livestock monitoring systems. We conclude with a call for agriculture-first AI design, prioritizing farm-level realism, cross-environment robustness, and open benchmark datasets to advance trustworthy and scalable animal-centric technologies.

---

## 24. A roadmap for curvature-based geometric data analysis and learning

**论文链接:** [http://arxiv.org/abs/2510.22599v1](http://arxiv.org/abs/2510.22599v1)

**作者:** Yasharth Yadav, Kelin Xia

**发布时间:** 2025-10-26

### GPT解析

### 总结

这篇论文提供了对离散曲率模型的首次全面综述，涵盖了数学基础、计算公式以及在数据分析和学习中的实际应用。文章从黎曼几何和度量几何角度讨论离散曲率，并提出了一种曲率驱动的数据分析系统流程。

### 背景

几何数据分析和学习已成为一个独特且快速发展的研究领域，因其跨领域的有效性而日益受到认可。曲率是该领域的核心概念，它能够捕捉内在几何结构并支持从社区检测到几何深度学习的众多任务。针对图、单纯复形、立方体复形和流形采样点云等多种数据表示，已经提出了广泛的离散曲率模型。

### 目的

这篇论文旨在对现有的离散曲率模型进行首次全面综述，涵盖其数学基础、计算公式以及在数据分析和学习中的实际应用。

### 方法

作者从黎曼几何和度量几何两个角度讨论离散曲率，并提出了一种曲率驱动的数据分析系统流程。他们还检查了不同数据表示下的相应计算算法，提供了详细的比较和见解。

### 主要发现

离散曲率模型不仅为数据几何提供了有效的表征，而且构成了几何学习框架的基本组成部分。这些模型在各种数据表示上都有应用，并在监督和无监督学习中取得了最先进的应用效果。

### 结论

这篇综述为研究人员提供了一个概念性和实践性的路线图，帮助他们更好地理解离散曲率作为几何理解和学习的基本工具。

### 翻译

几何数据分析和学习已成为一个独特且快速发展的研究领域，其有效性在多样化的应用中日益得到认可。该领域的核心是曲率，一个强大且可解释的概念，它捕捉内在的几何结构并支撑着从社区检测到几何深度学习的众多任务。针对图、单纯复形、立方体复形和从流形采样的点云等多种数据表示，已经提出了广泛的离散曲率模型。这些模型不仅为数据几何提供了有效的表征，而且构成了几何学习框架的基本组成部分。在本文中，我们首次对现有的离散曲率模型进行了全面综述，涵盖了它们的数学基础、计算公式以及数据分析和学习中的实际应用。特别是，我们从黎曼几何和度量几何的角度讨论了离散曲率，并提出了一个曲率驱动的数据分析系统流程。我们进一步检查了不同数据表示下的相应计算算法，提供了详细的比较和见解。最后，我们回顾了曲率在监督和无监督学习中的最先进应用。本综述为研究人员提供了一个概念性和实践性的路线图，使他们能够更好地理解离散曲率作为几何理解和学习的基本工具。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是离散曲率模型的系统综述和整合。目前存在多种离散曲率模型（如Forman-Ricci、Ollivier-Ricci等），它们基于不同数学原理，应用于不同数据表示，但缺乏统一框架和比较。这个问题的重要性在于几何数据分析和学习已成为快速发展的研究领域，曲率是理解数据内在几何结构的关键概念，从社区检测到几何深度学习都有重要应用，而离散曲率模型为数据几何提供了高效表征，并构成几何学习框架的基本组成部分。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过系统性地回顾现有文献来构建他们的方法。首先介绍几何数据分析和学习的背景，强调曲率的重要性；然后回顾黎曼几何和度量几何中曲率的数学基础；接着介绍多种离散曲率模型的定义和计算方法；最后提出一个三步流程用于基于曲率的数据分析。作者借鉴了大量现有工作，包括Forman基于Bochner-Weitzenböck公式的组合曲率方法、Ollivier基于最优输运的粗糙Ricci曲率、Bakry-Émery的Ricci曲率下界、Joharinad和Jost的sectional曲率，以及Menger曲率、Haantjes曲率和电阻曲率等网络分析中的定义。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是曲率作为理解数据内在几何结构的基本工具，可以用于分析和学习各种非欧几里得数据。整体实现流程是三步流程：1）数据表示：根据应用领域提取合适的拓扑表示（如图、单纯复形、立方体复形或超图）；2）离散曲率计算：在提取的拓扑表示上应用适当的曲率定义（如Forman-Ricci、Ollivier-Ricci等）；3）特征提取：从计算出的曲率中提取有意义的几何特征，如边基曲率特征化边或连接，顶点基曲率特征化顶点或数据点。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次全面综述离散曲率模型；系统化分类曲率模型；跨领域整合不同领域的曲率概念；提出实用三步流程；提供各种曲率模型在不同数据表示上的具体计算方法。相比之前的工作，这篇论文的不同之处在于：之前的文献通常专注于单一曲率模型或特定应用，而本文提供了全面视角，比较了不同模型的优缺点；不仅关注理论，还关注实际计算和应用；强调了曲率在不同数据表示上的通用性，展示了其在几何深度学习中的广泛应用潜力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次提供了离散曲率模型的全面综述，建立了从理论到实践的系统性框架，使研究者能够理解和应用曲率作为几何数据分析和学习的基本工具。'}


### 论文摘要

Geometric data analysis and learning has emerged as a distinct and rapidly developing research area, increasingly recognized for its effectiveness across diverse applications. At the heart of this field lies curvature, a powerful and interpretable concept that captures intrinsic geometric structure and underpins numerous tasks, from community detection to geometric deep learning. A wide range of discrete curvature models have been proposed for various data representations, including graphs, simplicial complexes, cubical complexes, and point clouds sampled from manifolds. These models not only provide efficient characterizations of data geometry but also constitute essential components in geometric learning frameworks. In this paper, we present the first comprehensive review of existing discrete curvature models, covering their mathematical foundations, computational formulations, and practical applications in data analysis and learning. In particular, we discuss discrete curvature from both Riemannian and metric geometry perspectives and propose a systematic pipeline for curvature-driven data analysis. We further examine the corresponding computational algorithms across different data representations, offering detailed comparisons and insights. Finally, we review state-of-the-art applications of curvature in both supervised and unsupervised learning. This survey provides a conceptual and practical roadmap for researchers to gain a better understanding of discrete curvature as a fundamental tool for geometric understanding and learning.

---

## 25. AnyECG-Lab: An Exploration Study of Fine-tuning an ECG Foundation Model to Estimate Laboratory Values from Single-Lead ECG Signals

**论文链接:** [http://arxiv.org/abs/2510.22301v1](http://arxiv.org/abs/2510.22301v1)

**作者:** Yujie Xiao, Gongzhen Tang, Wenhui Liu, Jun Li, Guangkun Nie, Zhuoran Kan, Deyun Zhang, Qinghao Zhao, Shenda Hong

**发布时间:** 2025-10-25

### GPT解析

### 总结

本研究探索了利用心电图(ECG)通过深度学习技术估计实验室值的可行性，提供了一种非侵入性、快速的临床决策支持方法。

### 背景

当前实验室检测依赖于侵入性静脉采样，存在延迟问题。心电图作为无创且广泛可用的信号，为快速估计实验室值提供了有前景的途径，但现有模型受限于低信噪比、个体间变异性大、数据多样性有限以及泛化能力不足。

### 目的

探索使用迁移学习微调大规模预训练的ECG基础模型(ECGFounder)，实现从ECG信号中估计实验室值，并建立实时、无创估计实验室值的可行性范围。

### 方法

利用斯坦福大学的多模式临床监测急诊数据集(MC-MED)进行探索性研究，使用迁移学习技术对ECGFounder大型预训练模型进行微调，并生成了超过2000万个标准化的10秒ECG片段以增强对细微生化相关性的敏感性。

### 主要发现

在内部验证中，模型对33项实验室指标表现出强的预测性能(曲线下面积高于0.65)，对59项指标表现出中等性能(曲线下面积在0.55到0.65之间)，对16项指标表现有限(曲线下面积低于0.55)。

### 结论

该研究提供了一种高效的人工智能驱动解决方案，并建立了实时、无创估计实验室值的可行性范围。

### 翻译

及时获取实验室值对临床决策至关重要，但当前方法依赖于侵入性静脉采样且本质上存在延迟。心电图作为一种无创且广泛可用的信号，为快速估计实验室值提供了有前景的方式。深度学习的最新进展使得从心电图中提取潜在的血液学特征成为可能。然而，现有模型受限于低信噪比、显著的个体间变异性、有限的数据多样性和次优的泛化能力，特别是在适配到导联数较少的可穿戴设备时。在本工作中，我们进行了一项探索性研究，利用迁移学习技术在斯坦福大学的多模式临床监测急诊数据集(MC-MED)上微调ECGFounder——一个大规模预训练的心电基础模型。我们生成了超过2000万个标准化的十秒心电片段，以增强对细微生化相关性的敏感性。在内部验证中，该模型对三十三项实验室指标表现出强的预测性能(曲线下面积高于0.65)，对五十九项指标表现出中等性能(在0.55和0.65之间)，对十六项指标表现有限(低于0.55)。该研究提供了一种高效的人工智能驱动解决方案，并建立了实时、无创估计实验室值的可行性范围。


### 论文摘要

Timely access to laboratory values is critical for clinical decision-making, yet current approaches rely on invasive venous sampling and are intrinsically delayed. Electrocardiography (ECG), as a non-invasive and widely available signal, offers a promising modality for rapid laboratory estimation. Recent progress in deep learning has enabled the extraction of latent hematological signatures from ECGs. However, existing models are constrained by low signal-to-noise ratios, substantial inter-individual variability, limited data diversity, and suboptimal generalization, especially when adapted to low-lead wearable devices. In this work, we conduct an exploratory study leveraging transfer learning to fine-tune ECGFounder, a large-scale pre-trained ECG foundation model, on the Multimodal Clinical Monitoring in the Emergency Department (MC-MED) dataset from Stanford. We generated a corpus of more than 20 million standardized ten-second ECG segments to enhance sensitivity to subtle biochemical correlates. On internal validation, the model demonstrated strong predictive performance (area under the curve above 0.65) for thirty-three laboratory indicators, moderate performance (between 0.55 and 0.65) for fifty-nine indicators, and limited performance (below 0.55) for sixteen indicators. This study provides an efficient artificial-intelligence driven solution and establishes the feasibility scope for real-time, non-invasive estimation of laboratory values.

---

## 26. Synthetic-to-Real Transfer Learning for Chromatin-Sensitive PWS Microscopy

**论文链接:** [http://arxiv.org/abs/2510.22239v1](http://arxiv.org/abs/2510.22239v1)

**作者:** Jahidul Arafat, Sanjaya Poudel

**发布时间:** 2025-10-25

**备注:** 24 pages, 5 figures and 4 tables

### GPT解析

### 总结

研究提出CFU Net，一种分层分割架构，使用三阶段课程在合成多模态数据上训练，实现了近乎完美的细胞核分割性能，应用于超过一万个细胞核的自动分析，提取了区分正常与癌前组织的染色质生物标志物，为专业显微镜中的合成到真实迁移学习提供了通用框架。

### 背景

染色质敏感部分波谱(csPWS)显微镜技术能够无标记检测发生在可见细胞转化之前的纳米级染色质包装变化，但手动细胞核分割限制了群体规模分析，且缺乏注释的csPWS成像数据阻碍了标准深度学习方法的使用。

### 目的

解决手动细胞核分割限制群体规模分析的问题，克服缺乏注释csPWS成像数据的挑战，开发能够自动分析染色质包装变化的方法，用于早期癌症检测中的生物标志物发现。

### 方法

提出CFU Net分层分割架构，使用三阶段课程在合成多模态数据上训练；采用基于物理的渲染，结合染色质包装统计、Mie散射模型和模态特定噪声；整合五种架构元素：ConvNeXt骨干网络、特征金字塔网络、UNet++密集连接、双注意力和深度监督；实现INT8量化以提高效率。

### 主要发现

在合成测试数据上实现近乎完美性能（Dice 0.9879，IoU 0.9895）；与基础UNet相比Dice提高8.3%；通过量化实现240倍吞吐量增益；提取的染色质生物标志物区分正常与癌前组织效应量显著（Cohen's d在1.31到2.98之间）；分类准确率达94%。

### 结论

该工作为专业显微镜中的合成到真实迁移学习提供了通用框架，并提供了社区在临床标本上进行验证的开放资源，有效应用于早期癌症检测中的生物标志物发现。

### 翻译

染色质敏感部分波谱(csPWS)显微镜技术能够无标记检测发生在可见细胞转化之前的纳米级染色质包装变化。然而，手动细胞核分割限制了早期癌症检测中生物标志物发现所需的群体规模分析。缺乏注释的csPWS成像数据阻碍了标准深度学习方法的直接使用。我们提出了CFU Net，一种使用三阶段课程在合成多模态数据上训练的分层分割架构。CFU Net在代表多样化光谱成像条件的保留合成测试数据上实现了近乎完美的性能，无需手动注释（Dice 0.9879，IoU 0.9895）。我们的方法使用基于物理的渲染，结合经验支持的染色质包装统计、Mie散射模型和模态特定噪声，并结合一个从对抗性RGB预训练进展到光谱微调和组织学验证的课程。CFU Net整合了五种架构元素（ConvNeXt骨干网络、特征金字塔网络、UNet++密集连接、双注意力和深度监督），这些元素共同使Dice比基础UNet提高了8.3%。我们展示了可部署的INT8量化，压缩率为74.9%，推理时间为0.15秒，比手动分析提高了240倍的吞吐量。应用于来自合成测试数据的超过一万个自动分割的细胞核，该流程提取了区分正常与癌前组织的染色质生物标志物，具有大的效应量（Cohen's d在1.31到2.98之间），达到94%的分类准确率。这项工作为专业显微镜中的合成到真实迁移学习提供了通用框架，并为社区在临床标本上进行验证提供了开放资源。


### 论文摘要

Chromatin sensitive partial wave spectroscopic (csPWS) microscopy enables label free detection of nanoscale chromatin packing alterations that occur before visible cellular transformation. However, manual nuclear segmentation limits population scale analysis needed for biomarker discovery in early cancer detection. The lack of annotated csPWS imaging data prevents direct use of standard deep learning methods. We present CFU Net, a hierarchical segmentation architecture trained with a three stage curriculum on synthetic multimodal data. CFU Net achieves near perfect performance on held out synthetic test data that represent diverse spectroscopic imaging conditions without manual annotations (Dice 0.9879, IoU 0.9895). Our approach uses physics based rendering that incorporates empirically supported chromatin packing statistics, Mie scattering models, and modality specific noise, combined with a curriculum that progresses from adversarial RGB pretraining to spectroscopic fine tuning and histology validation. CFU Net integrates five architectural elements (ConvNeXt backbone, Feature Pyramid Network, UNet plus plus dense connections, dual attention, and deep supervision) that together improve Dice over a baseline UNet by 8.3 percent. We demonstrate deployment ready INT8 quantization with 74.9 percent compression and 0.15 second inference, giving a 240 times throughput gain over manual analysis. Applied to more than ten thousand automatically segmented nuclei from synthetic test data, the pipeline extracts chromatin biomarkers that distinguish normal from pre cancerous tissue with large effect sizes (Cohens d between 1.31 and 2.98), reaching 94 percent classification accuracy. This work provides a general framework for synthetic to real transfer learning in specialized microscopy and open resources for community validation on clinical specimens.

---

## 27. Automatic Assessment of Students' Classroom Engagement with Bias Mitigated Multi-task Model

**论文链接:** [http://arxiv.org/abs/2510.22057v1](http://arxiv.org/abs/2510.22057v1)

**作者:** James Thiering, Tarun Sethupat Radha Krishna, Dylan Zelkin, Ashis Kumer Biswas

**发布时间:** 2025-10-24

**备注:** 13 pages, 12 figures, and 1 table

### GPT解析

### 总结

本研究开发了一种自动化系统来检测在线学习期间的学生参与度，同时确保模型不依赖性别等敏感特征进行预测，提高了模型的公平性和可解释性。

### 背景

随着在线和虚拟学习的兴起，监控和提升学生参与度已成为有效教育的重要方面，但传统评估方法可能不直接适用于虚拟环境。

### 目的

开发一个自动化系统来检测在线学习期间学生的参与度水平，解决传统方法在虚拟环境中不适用的问题。

### 方法

提出了一种新的训练方法，应用属性正则正交化技术到分割模型分类器中，并使用多种迁移学习策略，以阻止模型利用敏感特征如性别进行预测。

### 主要发现

所提出的方法不仅有助于执行道德标准，还能增强模型预测的可解释性；通过该方法，预测敏感群体的分布差异从未缓解模型的皮尔逊相关系数0.897降低到缓解模型的0.999。

### 结论

成功开发了一个能够检测在线学习中学生参与度的自动化系统，同时确保了模型的公平性和可解释性，源代码已在GitHub上公开。

### 翻译

随着在线和虚拟学习的兴起，监控和提升学生参与度已成为有效教育的重要方面。评估学生参与度的传统方法可能不直接适用于虚拟环境。在本研究中，我们关注这一问题，致力于开发一个自动化系统来检测在线学习期间学生的参与度水平。我们提出了一种新的训练方法，可以阻止模型利用性别等敏感特征进行预测。所提出的方法不仅在执行道德标准方面有益，还能增强模型预测的可解释性。我们将属性正则正交化技术应用于分割模型分类器，该分类器使用多种迁移学习策略，在减少预测敏感群体的分布差异方面取得了有效成果，从未缓解模型的皮尔逊相关系数0.897降低到缓解模型的0.999。该项目的源代码可在https://github.com/ashiskb/elearning-engagement-study获取。


### 论文摘要

With the rise of online and virtual learning, monitoring and enhancing student engagement have become an important aspect of effective education. Traditional methods of assessing a student's involvement might not be applicable directly to virtual environments. In this study, we focused on this problem and addressed the need to develop an automated system to detect student engagement levels during online learning. We proposed a novel training method which can discourage a model from leveraging sensitive features like gender for its predictions. The proposed method offers benefits not only in the enforcement of ethical standards, but also to enhance interpretability of the model predictions. We applied an attribute-orthogonal regularization technique to a split-model classifier, which uses multiple transfer learning strategies to achieve effective results in reducing disparity in the distribution of prediction for sensitivity groups from a Pearson correlation coefficient of 0.897 for the unmitigated model, to 0.999 for the mitigated model. The source code for this project is available on https://github.com/ashiskb/elearning-engagement-study .

---

## 28. LiteDiff

**论文链接:** [http://arxiv.org/abs/2510.22004v1](http://arxiv.org/abs/2510.22004v1)

**作者:** Ruchir Namjoshi, Nagasai Thadishetty, Vignesh Kumar, Hemanth Venkateshwara

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文提出了Lite-Diff，一种轻量级扩散模型适应方法，通过将轻量适应层集成到冻结的扩散U-Net中，结合潜在形态自编码器和像素级判别器，显著降低了计算成本并减少了过拟合，即使在数据有限的情况下也能高效工作。

### 背景

扩散模型在高保真图像合成方面取得了显著成功，但在特定领域（如医学成像）微调这些模型仍然具有挑战性，原因是领域特定数据有限和完整模型适应的高计算成本。

### 目的

开发一种高效微调方法，使扩散模型能够在特定领域（如医学成像）有效适应，同时降低计算成本并减少过拟合风险。

### 方法

Lite-Diff将轻量级适应层集成到冻结的扩散U-Net中，同时使用潜在形态自编码器（用于领域特定潜在一致性）和像素级判别器（用于对抗对齐）来增强训练。通过冻结基础模型权重并仅优化小型残差适配器模块实现轻量化。

### 主要发现

选择性在不同U-Net块中集成适应层可以找到效率与性能的最佳平衡。在三个胸部X光数据集（Kaggle Chest X-Ray Pneumonia、NIH Chest X-ray14和VinBigData Chest X_ray）上的实验表明，Lite-Diff相比传统完整微调实现了更好的适应效率。

### 结论

Lite-Diff框架为扩散模型的迁移学习提供了有希望的方向，促进了它们在多样化低数据领域中的部署。

### 翻译

近年来，扩散模型在高保真图像合成方面表现出色。然而，由于领域特定数据有限和完整模型适应的高计算成本，将这些模型微调到专业领域（如医学成像）仍然具有挑战性。在本文中，我们引入了Lite-Diff（轻量级扩散模型适应），一种新的微调方法，它将轻量级适应层集成到冻结的扩散U-Net中，同时使用潜在形态自编码器（用于领域特定潜在一致性）和像素级判别器（用于对抗对齐）来增强训练。通过冻结基础模型的权重并仅优化小型残差适配器模块，Lite-Diff显著降低了计算开销并减轻了过拟合，即使在数据有限的情况下也是如此。此外，我们进行了消融研究，分析了在不同U-Net块中选择性集成适应层的效果，揭示了效率与性能之间的最佳平衡。在三个胸部X光数据集 - (1) Kaggle胸部X光肺炎、(2) NIH胸部X光14和(3) VinBigData胸部X光上的实验表明，Lite-Diff相比传统完整微调实现了更好的适应效率。我们的框架为扩散模型的迁移学习提供了有希望的方向，促进了它们在多样化低数据领域中的部署。


### 论文摘要

In recent years, diffusion models have demonstrated remarkable success in high-fidelity image synthesis. However, fine-tuning these models for specialized domains, such as medical imaging, remains challenging due to limited domain-specific data and the high computational cost of full model adaptation. In this paper, we introduce Lite-Diff (Lightweight Diffusion Model Adaptation), a novel finetuning approach that integrates lightweight adaptation layers into a frozen diffusion U-Net while enhancing training with a latent morphological autoencoder (for domain-specific latent consistency) and a pixel level discriminator(for adversarial alignment). By freezing weights of the base model and optimizing only small residual adapter modules, LiteDiff significantly reduces the computational overhead and mitigates overfitting, even in minimal-data settings. Additionally, we conduct ablation studies to analyze the effects of selectively integrating adaptation layers in different U-Net blocks, revealing an optimal balance between efficiency and performance. Experiments on three chest X-ray datasets - (1) Kaggle Chest X-Ray Pneumonia, (2) NIH Chest X-ray14 and (3) VinBigData Chest X_ray demonstrate that LiteDiff achieves superior adaptation efficiency compared to naive full fine-tuning. Our framework provides a promising direction for transfer learning in diffusion models, facilitating their deployment in diverse low data domains.

---

## 29. Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG Classification

**论文链接:** [http://arxiv.org/abs/2510.21969v1](http://arxiv.org/abs/2510.21969v1)

**作者:** Weiyu Chen, Arnaud Delorme

**发布时间:** 2025-10-24

**备注:** 8 pages, 5 figures. Submitted to IEEE BIBM 2025 Workshop on Machine  Learning for EEG Signal Processing (MLESP)

### GPT解析

### 总结

研究提出了一种自适应分割最大均值差异训练(AS-MMD)方法，用于解决从脑电图(EEG)中检测单次试验P300时数据量有限的问题，特别是在跨数据集迁移学习中的分布偏移挑战。

### 背景

当只有少量标记试验可用时，从脑电图(EEG)中检测单次试验P300是困难的。当尝试通过迁移学习用大型源数据集增强小型目标集时，会出现跨数据集偏移问题。

### 目的

研究两个公共视觉oddball ERP数据集之间的迁移学习，解决在小样本设置下(目标:每个受试者10次试验；源:每个受试者80次试验)的跨数据集分布不一致问题。

### 方法

提出自适应分割最大均值差异训练(AS-MMD)，结合了三种技术：(1)与源/目标大小比值相关的目标加权损失和预热；(2)具有共享参数和每域统计的分割批量归一化；(3)使用中带带宽启发式的无参数对数级RBF核最大均值差异项。该方法在EEG Conformer上实现，与主干网络无关且保持推理模型不变。

### 主要发现

在两种迁移方向上，AS-MMD均优于仅目标训练和联合训练(Active Visual Oddball: 准确率/AUC为0.66/0.74；ERP CORE P3: 0.61/0.65)，与联合训练相比的增益在统计上显著。消融研究表明所有三个组件都对性能提升有贡献。

### 结论

AS-MMD方法有效解决了小样本条件下EEG信号P300检测中的跨数据集迁移学习挑战，通过结合三种创新技术显著提高了检测性能。

### 翻译

当只有少量标记试验可用时，从脑电图(EEG)中检测单次试验P300是困难的。当尝试通过迁移学习用大型源数据集增强小型目标集时，会出现跨数据集偏移。为应对这一挑战，我们在严格的小样本设置下(目标:每个受试者10次试验；源:每个受试者80次试验)，研究了使用五个共享电极(Fz, Pz, P3, P4, Oz)在两个公共视觉oddball ERP数据集之间的迁移学习。我们引入了自适应分割最大均值差异训练(AS-MMD)，它结合了(i)与源/目标大小比值的平方根相关的目标加权损失和预热，(ii)具有共享仿射参数和每域运行统计的分割批量归一化(Split-BN)，以及(iii)使用中带带宽启发式的无参数对数级径向基函数核最大均值差异(RBF-MMD)项。在EEG Conformer上实现后，AS-MMD与主干网络无关且保持推理时模型不变。在两种迁移方向上，它都优于仅目标训练和联合训练(Active Visual Oddball: 准确率/AUC为0.66/0.74；ERP CORE P3: 0.61/0.65)，与联合训练相比的增益在校正后的配对t检验下显著。消融研究将改进归因于所有三个组件。


### 论文摘要

Detecting single-trial P300 from EEG is difficult when only a few labeled trials are available. When attempting to boost a small target set with a large source dataset through transfer learning, cross-dataset shift arises. To address this challenge, we study transfer between two public visual-oddball ERP datasets using five shared electrodes (Fz, Pz, P3, P4, Oz) under a strict small-sample regime (target: 10 trials/subject; source: 80 trials/subject). We introduce Adaptive Split Maximum Mean Discrepancy Training (AS-MMD), which combines (i) a target-weighted loss with warm-up tied to the square root of the source/target size ratio, (ii) Split Batch Normalization (Split-BN) with shared affine parameters and per-domain running statistics, and (iii) a parameter-free logit-level Radial Basis Function kernel Maximum Mean Discrepancy (RBF-MMD) term using the median-bandwidth heuristic. Implemented on an EEG Conformer, AS-MMD is backbone-agnostic and leaves the inference-time model unchanged. Across both transfer directions, it outperforms target-only and pooled training (Active Visual Oddball: accuracy/AUC 0.66/0.74; ERP CORE P3: 0.61/0.65), with gains over pooling significant under corrected paired t-tests. Ablations attribute improvements to all three components.

---

## 30. An unsupervised tour through the hidden pathways of deep neural networks

**论文链接:** [http://arxiv.org/abs/2510.21582v1](http://arxiv.org/abs/2510.21582v1)

**作者:** Diego Doimo

**发布时间:** 2025-10-24

**备注:** PhD thesis

### GPT解析

### 总结

这项研究旨在深入理解深度人工神经网络创建有意义表示并实现泛化的内部机制。研究重点关注使用无监督学习工具描述隐藏表示的语义内容，并利用数据的低维结构。论文介绍了Gride方法用于估计数据内在维度，研究了深度神经网络中隐藏层概率密度的演变，以及探讨了深度神经网络中的泛化问题。

### 背景

深度神经网络虽然取得了显著成功，但其内部工作机制和泛化能力仍不完全清楚。理解神经网络如何创建有意义的表示以及它们如何能够泛化到未见数据是深度学习领域的关键挑战。

### 目的

提高对深度人工神经网络创建有意义表示和泛化能力的内部机制的理解。重点在于使用无监督学习工具描述隐藏表示的语义内容，并利用数据的低维结构。

### 方法

1. 开发了Gride方法，用于估计数据内在维度作为尺度的显式函数，无需降采样数据集。2. 研究了最先进深度神经网络中隐藏层概率密度的演变。3. 研究了深度神经网络中的泛化问题，特别是添加参数如何提高泛化性能。

### 主要发现

1. Gride方法基于严格的分布结果，能够量化估计的不确定性，且计算效率高。2. 深度神经网络的初始层生成单模态概率密度，消除与分类无关的结构；后续层中密度峰以分层方式出现，反映概念的语义层次。3. 宽神经网络学习冗余表示而非对虚假相关性过拟合，冗余神经元仅在网络被正则化且训练误差为零时出现。

### 结论

深度神经网络通过分层结构创建有意义的表示，初始层消除无关结构，后续层建立语义层次。网络的泛化能力与冗余表示学习相关，而非传统的偏差-方差权衡。Gride方法为分析数据内在结构提供了有效工具。

### 翻译

这篇论文的目的是提高我们对深度人工神经网络创建有意义表示并能够泛化的内部机制的理解。我们专注于使用无监督学习工具描述隐藏表示的语义内容的挑战，这些工具部分由我们开发并在本论文中描述，它们允许利用数据的低维结构。第2章介绍了Gride，一种允许将数据的内在维度估计为尺度的显式函数的方法，而无需对数据集进行任何降采样。我们的方法基于严格的分布结果，能够量化估计的不确定性。此外，我们的方法简单且计算高效，因为它仅依赖于最近数据点之间的距离。在第3章中，我们研究了一些最先进的深度神经网络中隐藏层概率密度的演变。我们发现初始层生成单模态概率密度，消除任何与分类无关的结构。在后续层中，密度峰以分层方式出现，反映概念的语义层次结构。这个过程在输出层的概率密度中留下了足迹，其中峰的地形可以重建类别的语义关系。在第4章中，我们研究了深度神经网络中的泛化问题：向插值其训练数据的网络添加参数通常会提高其泛化性能，这与经典的偏差-方差权衡相悖。我们证明宽神经网络学习冗余表示，而不是对虚假相关性过拟合，并且只有当网络被正则化且训练误差为零时，冗余神经元才会出现。


### 论文摘要

The goal of this thesis is to improve our understanding of the internal mechanisms by which deep artificial neural networks create meaningful representations and are able to generalize. We focus on the challenge of characterizing the semantic content of the hidden representations with unsupervised learning tools, partially developed by us and described in this thesis, which allow harnessing the low-dimensional structure of the data. Chapter 2. introduces Gride, a method that allows estimating the intrinsic dimension of the data as an explicit function of the scale without performing any decimation of the data set. Our approach is based on rigorous distributional results that enable the quantification of uncertainty of the estimates. Moreover, our method is simple and computationally efficient since it relies only on the distances among nearest data points. In Chapter 3, we study the evolution of the probability density across the hidden layers in some state-of-the-art deep neural networks. We find that the initial layers generate a unimodal probability density getting rid of any structure irrelevant to classification. In subsequent layers, density peaks arise in a hierarchical fashion that mirrors the semantic hierarchy of the concepts. This process leaves a footprint in the probability density of the output layer, where the topography of the peaks allows reconstructing the semantic relationships of the categories. In Chapter 4, we study the problem of generalization in deep neural networks: adding parameters to a network that interpolates its training data will typically improve its generalization performance, at odds with the classical bias-variance trade-off. We show that wide neural networks learn redundant representations instead of overfitting to spurious correlation and that redundant neurons appear only if the network is regularized and the training error is zero.

---

## 31. Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning

**论文链接:** [http://arxiv.org/abs/2510.21379v1](http://arxiv.org/abs/2510.21379v1)

**作者:** Dong Bok Lee, Aoxuan Silvia Zhang, Byungjoo Kim, Junhyeon Park, Steven Adriaensen, Juho Lee, Sung Ju Hwang, Hae Beom Lee

**发布时间:** 2025-10-24

**备注:** Published at NeurIPS 2025

### GPT解析

### 总结

本文提出了一种基于冻结-解冻贝叶斯优化的成本敏感超参数优化方法，通过引入效用函数、新的获取函数和停止准则，实现了在成本和性能之间的动态权衡，并通过迁移学习提高了样本效率。

### 背景

研究基于冻结-解冻贝叶斯优化的成本敏感超参数优化问题，关注用户在预期性能改进相对于额外计算成本不够满意时提前停止HPO过程的场景。

### 目的

引入描述成本与性能之间权衡的效用函数，结合新的获取函数和停止准则，动态选择最优配置并自动停止HPO过程，同时通过迁移学习提高样本效率。

### 方法

提出成本敏感HPO方法，引入效用函数，设计新的获取函数和停止准则，使用迁移学习开发专门的代理模型，提高冻结-解冻方法的样本效率。

### 主要发现

在多保真度HPO基准测试上验证了算法性能，优于所有考虑的冻结-解冻BO和迁移-BO基线方法，实现了成本和性能之间显著更好的权衡。

### 结论

所提方法在成本敏感HPO问题上表现出色，代码已在GitHub公开。

### 翻译

在本文中，我们解决了基于冻结-解冻贝叶斯优化（BO）的成本敏感超参数优化（HPO）问题。具体而言，我们假设一种场景，即当预期性能改进相对于额外计算成本不够令人满意时，用户希望提前停止HPO过程。受此场景启发，我们在冻结-解冻框架中引入了'效用'，这是一个描述成本与性能之间权衡的函数，可以从用户偏好数据中估计。这个效用函数结合我们新的获取函数和停止准则，使我们能够动态继续训练我们预期未来效用最大化的配置，并在效用最大值附近自动停止HPO过程。此外，我们通过迁移学习改进了现有冻结-解冻方法的样本效率，为成本敏感HPO问题开发了专门的代理模型。我们在既定的多保真度HPO基准上验证了我们的算法，并表明它优于我们考虑的所有先前冻结-解冻BO和迁移-BO基线方法，同时实现了成本和性能之间显著更好的权衡。我们的代码已在https://github.com/db-Lee/CFBO公开可用。


### 论文摘要

In this paper, we address the problem of \emph{cost-sensitive} hyperparameter optimization (HPO) built upon freeze-thaw Bayesian optimization (BO). Specifically, we assume a scenario where users want to early-stop the HPO process when the expected performance improvement is not satisfactory with respect to the additional computational cost. Motivated by this scenario, we introduce \emph{utility} in the freeze-thaw framework, a function describing the trade-off between the cost and performance that can be estimated from the user's preference data. This utility function, combined with our novel acquisition function and stopping criterion, allows us to dynamically continue training the configuration that we expect to maximally improve the utility in the future, and also automatically stop the HPO process around the maximum utility. Further, we improve the sample efficiency of existing freeze-thaw methods with transfer learning to develop a specialized surrogate model for the cost-sensitive HPO problem. We validate our algorithm on established multi-fidelity HPO benchmarks and show that it outperforms all the previous freeze-thaw BO and transfer-BO baselines we consider, while achieving a significantly better trade-off between the cost and performance. Our code is publicly available at https://github.com/db-Lee/CFBO.

---

## 32. $α$-LoRA: Effective Fine-Tuning via Base Model Rescaling

**论文链接:** [http://arxiv.org/abs/2510.21345v1](http://arxiv.org/abs/2510.21345v1)

**作者:** Aymane El Firdoussi, El Mahdi Chayti, Mohamed El Amine Seddik, Martin Jaggi

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文介绍了一类新的用于迁移学习的重参数化方法，旨在提高微调模型的泛化能力，并通过理论分析和实验验证了其有效性。

### 背景

微调被证明是使预训练模型在少量数据样本上在新任务上表现更好的有效方法，其中重参数化方法是最广泛使用的方法之一。

### 目的

设计一类新的重参数化方法，以增强微调模型的泛化能力。

### 方法

提出一类新的重参数化方法，在高维二分类设置中使用随机矩阵理论工具建立其有效性，并通过微调大型语言模型等实验进行验证。

### 主要发现

所提出的重参数化方法在高维二分类任务中表现有效，且通过微调LLMs的实验进一步验证了理论发现。

### 结论

新提出的重参数化方法能够有效提高微调模型的泛化能力。

### 翻译

微调已被证明是使预训练模型在少量数据样本上在新任务上表现更好的有效方法。其中最广泛使用的方法是重参数化方法，它们通过添加一个额外的可训练权重矩阵来更新目标模块的冻结权重矩阵。最突出的例子是低秩适应(LoRA)，近年来受到了广泛关注。在本文中，我们介绍了一类用于迁移学习的新型重参数化方法，旨在提高微调模型的泛化能力。我们使用随机矩阵理论工具在高维二分类设置中建立了该方法的有效性，并通过更真实的实验（如微调大型语言模型）进一步验证了我们的理论发现。


### 论文摘要

Fine-tuning has proven to be highly effective in adapting pre-trained models to perform better on new desired tasks with minimal data samples. Among the most widely used approaches are reparameterization methods, which update a target module by augmenting its frozen weight matrix with an additional trainable weight matrix. The most prominent example is Low Rank Adaption (LoRA), which gained significant attention in recent years. In this paper, we introduce a new class of reparameterization methods for transfer learning, designed to enhance the generalization ability of fine-tuned models. We establish the effectiveness of our approach in a high-dimensional binary classification setting using tools from Random Matrix Theory, and further validate our theoretical findings through more realistic experiments, such as fine-tuning LLMs.

---

## 33. Adaptive Graph Mixture of Residual Experts: Unsupervised Learning on Diverse Graphs with Heterogeneous Specialization

**论文链接:** [http://arxiv.org/abs/2510.21207v1](http://arxiv.org/abs/2510.21207v1)

**作者:** Yunlong Chu, Minglai Shao, Zengyi Wo, Bing Hao, Yuhang Liu, Ruijie Wang, Jianxin Li

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文提出ADaMoRE框架，解决了图神经网络在适应多样化图结构方面的挑战，通过无监督训练实现了异构专家的有效组合，在各种任务中表现出色。

### 背景

图神经网络面临基本适应性挑战：固定的消息传递架构难以应对现实世界图的巨大多样性，最优计算策略因局部结构和任务而异。现有图专家混合方法依赖监督信号且训练异构专家时存在不稳定性。

### 目的

引入ADaMoRE框架，实现在图上进行异构专家混合的稳健、完全无监督训练。

### 方法

ADaMoRE采用骨干-残差专家架构，基础编码器提供稳定性，残差专家捕获不同计算模式；结构感知门控网络执行细粒度节点路由；通过统一无监督目标进行端到端训练，结合重建任务和信息论多样性正则化器强制专家功能专业化。

### 主要发现

在16个基准测试上验证了ADaMoRE在无监督节点分类和少样本学习方面的最先进性能，以及优越的泛化能力、训练效率和更快收敛速度。

### 结论

ADaMoRE框架通过无监督训练有效解决了图神经网络适应性问题，在多样化图和任务上展现出卓越性能。

### 翻译

图神经网络(GNNs)面临一个基本的适应性挑战：它们固定的消息传递架构难以应对现实世界图的巨大多样性，而最优的计算策略因局部结构和任务的不同而异。尽管专家混合(MoE)为适应性提供了一条有前景的路径，但现有的图MoE方法仍然依赖于监督信号，并且在训练异构专家时存在不稳定性。我们引入ADaMoRE(Adaptive Mixture of Residual Experts)，这是一个原则性框架，能够在图上实现异构MoE的稳健、完全无监督训练。ADaMoRE采用骨干-残差专家架构，其中基础编码器提供稳定性，而专门的残差专家捕获不同的计算模式。一个结构感知的门控网络执行细粒度的节点路由。整个架构通过统一的无监督目标进行端到端训练，该目标结合了主要的重建任务和信息论多样性正则化器，以明确强制专家之间的功能专业化。理论分析证实了他们的设计提高了数据效率和训练稳定性。在16个基准测试上的广泛评估验证了ADaMoRE在无监督节点分类和少样本学习方面的最先进性能，以及在多样化图和任务上的优越泛化能力、训练效率和更快收敛速度。


### 论文摘要

Graph Neural Networks (GNNs) face a fundamental adaptability challenge: their fixed message-passing architectures struggle with the immense diversity of real-world graphs, where optimal computational strategies vary by local structure and task. While Mixture-of-Experts (MoE) offers a promising pathway to adaptability, existing graph MoE methods remain constrained by their reliance on supervised signals and instability when training heterogeneous experts. We introduce ADaMoRE (Adaptive Mixture of Residual Experts), a principled framework that enables robust, fully unsupervised training of heterogeneous MoE on graphs. ADaMoRE employs a backbone-residual expert architecture where foundational encoders provide stability while specialized residual experts capture diverse computational patterns. A structurally-aware gating network performs fine-grained node routing. The entire architecture is trained end-to-end using a unified unsupervised objective, which integrates a primary reconstruction task with an information-theoretic diversity regularizer to explicitly enforce functional specialization among the experts. Theoretical analysis confirms our design improves data efficiency and training stability. Extensive evaluation across 16 benchmarks validates ADaMoRE's state-of-the-art performance in unsupervised node classification and few-shot learning, alongside superior generalization, training efficiency, and faster convergence on diverse graphs and tasks.

---

## 34. CIPHER: Scalable Time Series Analysis for Physical Sciences with Application to Solar Wind Phenomena

**论文链接:** [http://arxiv.org/abs/2510.21022v1](http://arxiv.org/abs/2510.21022v1)

**作者:** Jasmine R. Kobayashi, Daniela Martin, Valmir P Moraes Filho, Connor O'Brien, Jinsu Hong, Sudeshna Boro Saikia, Hala Lamdouar, Nathan D. Miles, Marcella Scoczynski, Mavis Stone, Sairam Sundaresan, Anna Jungbluth, Andrés Muñoz-Jaramillo, Evangelia Samara, Joseph Gallego

**发布时间:** 2025-10-23

**备注:** 5 pages, 2 figures, Machine Learning and the Physical Sciences  Workshop @ NeurIPS 2025

### GPT解析

### 总结

这篇论文介绍了一个名为CIPHER的框架，用于加速物理学中复杂时间序列的大规模标注。该框架结合了可索引符号聚合近似、基于密度的聚类和人类专家验证，解决了物理科学中时间序列标注稀缺、成本高且不一致的问题。

### 背景

在物理科学中，时间序列的标注或分类是一个持续的挑战。专家标注稀缺、成本高且往往不一致，但稳健的标注对于启用机器学习模型进行理解、预测和预测至关重要。

### 目的

设计一个框架来加速物理学中复杂时间序列的大规模标注，解决专家标注稀缺的问题。

### 方法

CIPHER框架集成了以下组件：1. 可索引符号聚合近似用于可解释的压缩和索引；2. 基于密度的聚类来分组重复出现的现象；3. 人类在环中的步骤用于高效的专家验证。领域科学家对代表性样本进行标注，然后将这些标注传播到整个集群中，产生系统化的、可扩展的分类。

### 主要发现

作者在OMNI数据中分类太阳风现象的任务上评估了CIPHER，这是空间天气研究中的一个核心挑战。结果表明，该框架能够识别有意义的现象，如日冕物质抛射和流相互作用区域。

### 结论

CIPHER展示了一种结合符号表示、无监督学习和专业知识的通用策略，以解决物理科学中时间序列标注稀缺的问题。研究所用的代码和配置文件是公开的，以支持可重复性。

### 翻译

时间序列的标注或分类在物理科学中是一个持续的挑战，其中专家标注稀缺、成本高昂且往往不一致。然而，稳健的标注对于启用机器学习模型进行理解、预测和预测至关重要。我们提出了'聚类与索引管道及人类评估用于识别'，这是一个旨在加速物理学中复杂时间序列大规模标注的框架。CIPHER集成了可索引符号聚合近似用于可解释的压缩和索引，基于密度的聚类来分组重复出现的现象，以及一个人机交互的步骤用于高效的专家验证。代表性样本由领域科学家标注，这些标注被传播到整个集群中，产生系统化、可扩展的分类。我们在OMNI数据中分类太阳风现象的任务上评估了CIPHER，这是空间天气研究中的一个核心挑战，结果表明该框架能够识别有意义的现象，如日冕物质抛射和流相互作用区域。除了这个案例研究，CIPHER强调了一种结合符号表示、无监督学习和专业知识的通用策略，以解决物理科学中时间序列的标注稀缺问题。本研究使用的代码和配置文件是公开的，以支持可重复性。


### 论文摘要

Labeling or classifying time series is a persistent challenge in the physical sciences, where expert annotations are scarce, costly, and often inconsistent. Yet robust labeling is essential to enable machine learning models for understanding, prediction, and forecasting. We present the \textit{Clustering and Indexation Pipeline with Human Evaluation for Recognition} (CIPHER), a framework designed to accelerate large-scale labeling of complex time series in physics. CIPHER integrates \textit{indexable Symbolic Aggregate approXimation} (iSAX) for interpretable compression and indexing, density-based clustering (HDBSCAN) to group recurring phenomena, and a human-in-the-loop step for efficient expert validation. Representative samples are labeled by domain scientists, and these annotations are propagated across clusters to yield systematic, scalable classifications. We evaluate CIPHER on the task of classifying solar wind phenomena in OMNI data, a central challenge in space weather research, showing that the framework recovers meaningful phenomena such as coronal mass ejections and stream interaction regions. Beyond this case study, CIPHER highlights a general strategy for combining symbolic representations, unsupervised learning, and expert knowledge to address label scarcity in time series across the physical sciences. The code and configuration files used in this study are publicly available to support reproducibility.

---

## 35. Memory Constrained Dynamic Subnetwork Update for Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.20979v1](http://arxiv.org/abs/2510.20979v1)

**作者:** Aël Quélennec, Pavlo Mozharovskyi, Van-Tam Nguyen, Enzo Tartaglione

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了一种名为MeDyate的框架，用于解决在设备上神经网络训练面临的内存限制问题，通过动态子网络适应方法实现了在严格内存预算下的有效微调。

### 背景

在设备上的神经网络训练面临严重的内存限制，这些限制阻碍了预训练模型对下游任务的适应。

### 目的

提出一个有理论依据的框架，用于内存受限的动态子网络适应，实现在严格内存预算下的有效微调。

### 方法

MeDyate框架包含两个关键创新：LaRa（Layer Ranking）作为改进的层重要性度量实现有原则的层预选择，以及动态通道采样策略利用微调过程中通道重要性分布的时间稳定性；根据重要性加权概率在周期之间动态重新采样通道，确保在尊重内存预算的同时全面探索参数空间。

### 主要发现

在广泛的任务和架构上进行的大量评估表明，MeDyate在极端内存限制下实现了最先进的性能，一致优于现有的静态和动态方法，同时保持高计算效率。

### 结论

该方法代表了推动设备上高效学习的重要一步，证明了在内存预算低至几百KB RAM的情况下进行有效微调的可能性。

### 翻译

设备上的神经网络训练面临关键的内存限制，这些限制阻碍了预训练模型对下游任务的适应。我们提出了MeDyate，一个有理论依据的框架，用于内存受限的动态子网络适应。我们的方法引入了两个关键创新：LaRa（Layer Ranking），一种改进的层重要性度量，能够实现有原则的层预选择，以及动态通道采样策略，利用微调过程中通道重要性分布的时间稳定性。MeDyate根据重要性加权概率在周期之间动态重新采样通道，确保在尊重严格内存预算的同时全面探索参数空间。在广泛的任务和架构上进行的大量评估表明，MeDyate在极端内存限制下实现了最先进的性能，一致优于现有的静态和动态方法，同时保持高计算效率。我们的方法代表了推动设备上高效学习的重要一步，证明了在内存预算低至几百KB RAM的情况下进行有效微调的可能性。


### 论文摘要

On-device neural network training faces critical memory constraints that limit the adaptation of pre-trained models to downstream tasks. We present MeDyate, a theoretically-grounded framework for memory-constrained dynamic subnetwork adaptation. Our approach introduces two key innovations: LaRa (Layer Ranking), an improved layer importance metric that enables principled layer pre-selection, and a dynamic channel sampling strategy that exploits the temporal stability of channel importance distributions during fine-tuning. MeDyate dynamically resamples channels between epochs according to importance-weighted probabilities, ensuring comprehensive parameter space exploration while respecting strict memory budgets. Extensive evaluation across a large panel of tasks and architectures demonstrates that MeDyate achieves state-of-the-art performance under extreme memory constraints, consistently outperforming existing static and dynamic approaches while maintaining high computational efficiency. Our method represents a significant step towards enabling efficient on-device learning by demonstrating effective fine-tuning with memory budgets as low as a few hundred kB of RAM.

---

## 36. AG-Fusion: adaptive gated multimodal fusion for 3d object detection in complex scenes

**论文链接:** [http://arxiv.org/abs/2510.23151v1](http://arxiv.org/abs/2510.23151v1)

**作者:** Sixian Liu, Chen Xu, Qiang Wang, Donghai Shi, Yiwen Li

**发布时间:** 2025-10-27

### GPT解析

### 总结

该研究提出了一种自适应门控融合方法，通过选择性整合跨模态知识，在复杂场景中实现了更鲁棒的3D目标检测。

### 背景

多模态相机-激光雷达融合技术在3D目标检测中应用广泛，但在传感器退化或环境干扰等具有挑战性的场景中，现有方法性能显著下降。

### 目的

提出一种新的自适应门控融合方法，通过识别可靠的模式来选择性地整合跨模态知识，以实现复杂场景中的鲁棒检测。

### 方法

将每个模态的特征投影到统一的BEV空间并使用基于窗口的注意力机制增强特征，然后设计基于跨模态注意力的自适应门控融合模块来整合这些特征，同时构建了一个名为Excavator3D（E3D）的新数据集，专注于具有挑战性的挖掘机操作场景。

### 主要发现

在标准的KITTI数据集上达到93.92%的准确率，在具有挑战性的E3D数据集上比基线方法高出24.88%，证明了在复杂工业场景中对不可靠模态信息具有优越的鲁棒性。

### 结论

提出的AG-Fusion方法在复杂场景中表现优异，特别是在处理不可靠模态信息时具有更强的鲁棒性。

### 翻译

多模态相机-激光雷达融合技术在3D目标检测中已得到广泛应用，并显示出令人鼓舞的性能表现。然而，在传感器退化或环境干扰等具有挑战性的场景中，现有方法表现出显著的性能下降。我们提出了一种新颖的自适应门控融合方法，通过识别可靠的模式来选择性地整合跨模态知识，从而在复杂场景中实现鲁棒检测。具体而言，我们首先将每个模态的特征投影到统一的BEV空间，并使用基于窗口的注意力机制增强这些特征。随后，我们设计了一个基于跨模态注意力的自适应门控融合模块，将这些特征整合为可靠的BEV表示，以应对具有挑战性的环境。此外，我们构建了一个名为Excavator3D（E3D）的新数据集，专注于具有挑战性的挖掘机操作场景，以在复杂条件下评估性能。我们的方法不仅在标准的KITTI数据集上实现了93.92%的准确率，具有竞争力的性能，而且在具有挑战性的E3D数据集上比基线方法高出24.88%，证明了在复杂工业场景中对不可靠模态信息具有优越的鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态相机-激光雷达融合技术在复杂场景（如挖掘机操作环境）中性能显著下降的问题。这些问题在现实中很重要，因为灰尘、光照变化导致图像退化，机械部件遮挡和金属表面反射干扰点云数据，这些挑战限制了自动驾驶和工业自动化技术在真实世界中的应用，现有方法在标准数据集上表现良好但在复杂场景中性能大幅下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有BEV融合技术的局限性，指出它们主要依赖卷积操作，无法自适应建模跨模态交互。针对工业场景中的挑战，作者在BEVFusion基础上进行改进，借鉴了Swin Transformer的窗口自注意力机制设计SA-E模块增强特征，并引入双向交叉注意力和自适应门控机制实现更智能的融合。整体设计思路是根据场景特点动态调整不同模态的贡献权重。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自适应选择和融合来自相机和激光雷达的可靠信息，提高在复杂场景中的3D目标检测性能。方法首先使用窗口自注意力增强每个模态的特征；然后通过双向交叉注意力和自适应门控机制融合这些特征，根据场景特点动态调整不同模态的贡献权重；最后将所有特征流集成到统一的BEV表示中进行3D检测。整体流程包括特征提取、增强特征提取、跨模态门控融合和多级特征聚合四个主要步骤。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 自适应门控多模态融合(AG-Fusion)框架，结合双向交叉注意力和空间自适应门控机制；2) 基于窗口的自注意力增强(SA-E)模块，有效降低计算复杂度；3) 构建了专门的挖掘机3D检测数据集(E3D)。相比之前的工作，该方法不再依赖静态或局部约束的特征聚合，能够处理遮挡和传感器噪声，在复杂工业场景中表现显著优于现有方法，在E3D数据集上比基线方法提升24.88%的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种自适应门控多模态融合方法，通过动态整合相机和激光雷达的可靠信息，显著提升了在复杂工业场景中的3D目标检测性能，并构建了专门的挖掘机3D检测数据集验证方法的有效性。'}


### 论文摘要

Multimodal camera-LiDAR fusion technology has found extensive application in 3D object detection, demonstrating encouraging performance. However, existing methods exhibit significant performance degradation in challenging scenarios characterized by sensor degradation or environmental disturbances. We propose a novel Adaptive Gated Fusion (AG-Fusion) approach that selectively integrates cross-modal knowledge by identifying reliable patterns for robust detection in complex scenes. Specifically, we first project features from each modality into a unified BEV space and enhance them using a window-based attention mechanism. Subsequently, an adaptive gated fusion module based on cross-modal attention is designed to integrate these features into reliable BEV representations robust to challenging environments. Furthermore, we construct a new dataset named Excavator3D (E3D) focusing on challenging excavator operation scenarios to benchmark performance in complex conditions. Our method not only achieves competitive performance on the standard KITTI dataset with 93.92% accuracy, but also significantly outperforms the baseline by 24.88% on the challenging E3D dataset, demonstrating superior robustness to unreliable modal information in complex industrial scenes.

---

## 37. DQ3D: Depth-guided Query for Transformer-Based 3D Object Detection in Traffic Scenarios

**论文链接:** [http://arxiv.org/abs/2510.23144v1](http://arxiv.org/abs/2510.23144v1)

**作者:** Ziyu Wang, Wenhao Li, Ji Wu

**发布时间:** 2025-10-27

### GPT解析

### 总结

论文提出了一个名为DQ3D的深度引导查询生成器，用于解决3D目标检测中的参考点采样问题，并通过混合注意力机制处理部分遮挡目标，在nuScenes数据集上取得了显著性能提升。

### 背景

多视角图像中的3D目标检测在交通场景中近年来受到广泛关注。现有方法依赖于从3D参考点生成的目标查询来定位物体。

### 目的

解决现有方法中参考点可能远离目标物体导致误检的问题，并处理当前帧中部分遮挡的目标物体。

### 方法

提出了深度引导查询生成器(DQ3D)，利用深度信息和2D检测确保参考点从物体表面或内部采样；引入混合注意力机制，将历史检测结果与深度引导查询融合，形成混合查询。

### 主要发现

在nuScenes数据集上的评估表明，该方法在平均精度(mAP)上比基线提高了6.3%，在NuScenes检测分数(NDS)上提高了4.3%。

### 结论

深度引导查询生成器和混合注意力机制能有效提高3D目标检测的性能，特别是在处理参考点采样和遮挡物体方面。

### 翻译

近年来，交通场景中基于多视角图像的3D目标检测受到了广泛关注。许多现有方法依赖于从3D参考点生成的目标查询来定位物体。然而，这些方法的一个局限性是，一些参考点通常远离目标物体，这可能导致误检。在本文中，我们提出了一个用于3D目标检测的深度引导查询生成器(DQ3D)，它利用深度信息和2D检测确保参考点从物体表面或内部采样。此外，为了解决当前帧中部分遮挡的物体，我们引入了一种混合注意力机制，将历史检测结果与深度引导查询融合，从而形成混合查询。在nuScenes数据集上的评估表明，我们的方法在平均精度(mAP)上比基线提高了6.3%，在NuScenes检测分数(NDS)上提高了4.3%。


### 论文摘要

3D object detection from multi-view images in traffic scenarios has garnered significant attention in recent years. Many existing approaches rely on object queries that are generated from 3D reference points to localize objects. However, a limitation of these methods is that some reference points are often far from the target object, which can lead to false positive detections. In this paper, we propose a depth-guided query generator for 3D object detection (DQ3D) that leverages depth information and 2D detections to ensure that reference points are sampled from the surface or interior of the object. Furthermore, to address partially occluded objects in current frame, we introduce a hybrid attention mechanism that fuses historical detection results with depth-guided queries, thereby forming hybrid queries. Evaluation on the nuScenes dataset demonstrates that our method outperforms the baseline by 6.3\% in terms of mean Average Precision (mAP) and 4.3\% in the NuScenes Detection Score (NDS).

---

## 38. RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation

**论文链接:** [http://arxiv.org/abs/2509.16500v2](http://arxiv.org/abs/2509.16500v2)

**作者:** Tianyi Yan, Wencheng Han, Xia Zhou, Xueyang Zhang, Kun Zhan, Cheng-zhong Xu, Jianbing Shen

**发布时间:** 2025-09-20

**备注:** NeurIPS 2025

### GPT解析

### 总结

该研究解决了自动驾驶系统中合成视频数据的几何失真问题，提出了强化学习与几何反馈(RLGF)方法，显著提高了合成数据的几何准确性和3D目标检测性能。

### 背景

合成数据对推进自动驾驶系统至关重要，但当前最先进的视频生成模型尽管视觉上逼真，却存在微妙的几何失真，限制了其在下游感知任务中的应用。研究显示，使用合成数据与真实数据进行3D目标检测时存在显著性能差距。

### 目的

开发一种方法来减少合成视频数据中的几何失真，提高其在自动驾驶感知任务中的效用，缩小合成数据与真实数据之间的性能差距。

### 方法

研究引入了'带有几何反馈的强化学习'(RLGF)，该方法通过整合来自专用潜在空间自动驾驶感知模型的奖励来优化视频扩散模型。其核心组件包括：1) 潜在空间窗口优化技术，用于在扩散过程中提供针对性反馈；2) 分层几何奖励(HGR)系统，为点线面对齐和场景占用一致性提供多级奖励。研究还提出了GeoScores来量化几何失真。

### 主要发现

应用RLGF到DiVE模型上，在nuScenes数据集上显著减少了几何误差（例如：消失点误差降低21%，深度误差降低57%），并大幅提高了3D目标检测mAP达12.7%，缩小了与真实数据性能的差距。

### 结论

RLGF为自动驾驶开发提供了一种即插即用的解决方案，能够生成几何准确可靠的合成视频，有助于推进自动驾驶系统的训练和测试。

### 翻译

合成数据对推进自动驾驶系统至关重要，但当前最先进的视频生成模型尽管视觉上逼真，却存在微妙的几何失真，限制了其在下游感知任务中的应用。研究确定了并量化了这一关键问题，展示了使用合成数据与真实数据进行3D目标检测时的显著性能差距。为此，研究引入了'带有几何反馈的强化学习'(RLGF)。RLGF通过整合来自专用潜在空间自动驾驶感知模型的奖励，独特地优化了视频扩散模型。其核心组件包括：1) 用于在扩散过程中提供针对性反馈的高效潜在空间窗口优化技术；2) 提供点线面对齐和场景占用一致性多级奖励的分层几何奖励(HGR)系统。为了量化这些失真，研究提出了GeoScores。将RLGF应用于nuScenes上的DiVE等模型，显著减少了几何误差（例如：消失点误差降低21%，深度误差降低57%），并大幅提高了3D目标检测mAP达12.7%，缩小了与真实数据性能的差距。RLGF为生成几何准确可靠的合成视频用于自动驾驶开发提供了一种即插即用的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶视频生成中存在的几何失真问题。虽然当前视频生成模型在视觉上看起来很真实，但它们生成的视频中存在微妙的几何扭曲，这些扭曲限制了它们在下游感知任务中的实用性。这个问题很重要，因为自动驾驶系统需要大量高质量的合成数据进行训练和验证，而这些几何扭曲会导致基于合成数据训练的3D目标检测等下游任务性能显著下降，影响自动驾驶系统的可靠性和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别并量化了当前视频生成模型中的几何失真问题，提出了GeoScores指标来评估这些失真。通过实验发现当前模型虽然保留了2D外观但无法捕捉准确的3D场景结构，主要原因是潜在的几何不一致性。作者借鉴了强化学习从人类反馈的成功经验（如LLMs中的PPO或DPO）、视频扩散模型的研究以及自动驾驶感知模型的设计。但作者指出现有方法主要依赖像素级对齐，无法显式强制遵守复杂的底层几何原理，因此设计了RLGF框架，利用专门的预训练自动驾驶感知模型作为奖励提供者，确保几何保真度。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入基于感知模型的几何反馈来增强视频扩散模型的几何完整性，使生成的自动驾驶场景视频不仅在视觉上逼真，而且在几何结构上准确可靠。整体流程包括：1)预训练两个专门的感知模型（潜在几何感知模型Pgeo和潜在占用预测模型Pocc）；2)设计分层几何奖励（HGR）系统，提供点线面几何反馈和场景级占用反馈；3)实现潜在空间窗口化优化，在扩散过程的中间步骤提供反馈；4)使用强化学习微调预训练的视频扩散模型，将感知模型作为奖励提供者，通过LoRA高效更新模型参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统量化了自动驾驶视频生成中的几何失真问题，提出GeoScores评估指标；2)引入强化学习与几何反馈（RLGF）框架，将感知模型驱动的几何空间约束直接注入视频生成过程；3)提出潜在空间窗口化优化技术，在扩散过程的中间步骤而非仅最终输出提供反馈；4)设计分层几何奖励（HGR）系统，结合点线面几何反馈和场景级占用反馈。相比之前的工作，RLGF专注于几何完整性而非仅像素级视觉保真度，使用具体的、可解释的几何约束而非人类偏好或高层次奖励信号，并且是即插即用的解决方案，可与现有视频扩散模型集成。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了RLGF框架，通过引入基于感知模型的几何反馈强化学习，有效解决了自动驾驶视频生成中的几何失真问题，显著提升了合成数据的3D感知任务性能。'}


### 论文摘要

Synthetic data is crucial for advancing autonomous driving (AD) systems, yet current state-of-the-art video generation models, despite their visual realism, suffer from subtle geometric distortions that limit their utility for downstream perception tasks. We identify and quantify this critical issue, demonstrating a significant performance gap in 3D object detection when using synthetic versus real data. To address this, we introduce Reinforcement Learning with Geometric Feedback (RLGF), RLGF uniquely refines video diffusion models by incorporating rewards from specialized latent-space AD perception models. Its core components include an efficient Latent-Space Windowing Optimization technique for targeted feedback during diffusion, and a Hierarchical Geometric Reward (HGR) system providing multi-level rewards for point-line-plane alignment, and scene occupancy coherence. To quantify these distortions, we propose GeoScores. Applied to models like DiVE on nuScenes, RLGF substantially reduces geometric errors (e.g., VP error by 21\%, Depth error by 57\%) and dramatically improves 3D object detection mAP by 12.7\%, narrowing the gap to real-data performance. RLGF offers a plug-and-play solution for generating geometrically sound and reliable synthetic videos for AD development.

---

## 39. When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning

**论文链接:** [http://arxiv.org/abs/2510.23532v1](http://arxiv.org/abs/2510.23532v1)

**作者:** Anirban Das, Irtaza Khalid, Rafael Peñaloza, Steven Schockaert

**发布时间:** 2025-10-27

**备注:** accepted at NeurIPS 2025 D&B track

### GPT解析

### 总结

本文提出了NoRA新基准测试，用于评估系统关系推理模型的泛化能力，突破了现有基于路径组合的简化假设。

### 背景

设计能够系统化学习的模型是重要挑战，近年已提出多种解决方案，包括神经符号方法、Transformer变体和图神经网络，但现有基准测试过于简化。

### 目的

支持神经网络在系统关系推理领域的进一步发展，引入更全面的评估基准。

### 方法

开发NoRA基准测试，增加多个难度级别，要求模型超越简单的路径组合推理。

### 主要发现

现有基准测试基于推理可简化为关系路径组合的假设，导致模型在这些基准上表现良好但难以泛化到其他场景。

### 结论

需要NoRA这样的新基准来更全面地评估模型的真实系统推理能力，推动领域发展。

### 翻译

设计能够以系统化方式学习的模型是一个重要且长期存在的挑战。近年来，针对系统关系推理的特定案例提出了多种解决方案，包括神经符号方法、Transformer架构的变体和专门的图神经网络。然而，现有的系统关系推理基准测试基于过于简化的设置，基于推理可以简化为组合关系路径的假设。事实上，这个假设被嵌入到几个最新模型的架构中，导致这些方法在现有基准上表现良好但难以推广到其他设置。为了支持神经网络在系统关系推理领域的进一步发展，我们引入NoRA，一个新的基准测试，它增加了多个难度级别，要求模型超越基于路径的推理。


### 论文摘要

Designing models that can learn to reason in a systematic way is an important and long-standing challenge. In recent years, a wide range of solutions have been proposed for the specific case of systematic relational reasoning, including Neuro-Symbolic approaches, variants of the Transformer architecture, and specialised Graph Neural Networks. However, existing benchmarks for systematic relational reasoning focus on an overly simplified setting, based on the assumption that reasoning can be reduced to composing relational paths. In fact, this assumption is hard-baked into the architecture of several recent models, leading to approaches that can perform well on existing benchmarks but are difficult to generalise to other settings. To support further progress in the field of systematic relational reasoning with neural networks, we introduce NoRA, a new benchmark which adds several levels of difficulty and requires models to go beyond path-based reasoning.

---

## 40. iPac: Incorporating Intra-image Patch Context into Graph Neural Networks for Medical Image Classification

**论文链接:** [http://arxiv.org/abs/2510.23504v1](http://arxiv.org/abs/2510.23504v1)

**作者:** Usama Zidan, Mohamed Gaber, Mohammed M. Abdelsamea

**发布时间:** 2025-10-27

**备注:** Accepted for publication in the proceedings of ICONIP 2025

### GPT解析

### 总结

iPac是一种创新的图神经网络方法，通过引入图像的新型图表示来改进图像分类性能，特别在医学图像分类中表现出色，比基线方法平均提高5%的准确率。

### 背景

图神经网络已成为图像处理的一种有前景的范式，但在图像分类任务中的表现受到限制，因为它们对视觉实体之间的底层结构和关系考虑不足。

### 目的

提出iPac方法，引入图像的新型图表示，增强图神经网络在医学图像分类中的性能，通过认识到底层结构和关系在医学图像分类中的重要性。

### 方法

iPac集成了多个阶段，包括patch分区、特征提取、聚类、图构建和基于图的学习，将这些阶段整合到一个统一的网络中，通过捕获相关特征并将它们组织成簇，构建有意义的图表示，有效封装图像的语义。

### 主要发现

在多种医学图像数据集上的实验评估证明了iPac的有效性，与基线方法相比，平均准确率提高了高达5%。

### 结论

该方法为图像分类提供了一种通用且灵活的解决方案，特别是在医学图像领域，通过利用图表示并考虑视觉实体之间的固有结构和关系。

### 翻译

图神经网络已成为图像处理的一种有前景的范式，但它们在图像分类任务中的表现受到对视觉实体之间底层结构和关系考虑不足的限制。本文提出了iPac，一种通过引入图像的新型图表示来增强图神经网络图像分类的新方法，通过认识到底层结构和关系在医学图像分类中的重要性。iPac将多个阶段（包括patch分区、特征提取、聚类、图构建和基于图的学习）整合到一个统一的网络中，以推进图神经网络图像分类。通过捕获相关特征并将它们组织成簇，我们构建了一个有意义的图表示，有效地封装了图像的语义。在多种医学图像数据集上的实验评估证明了iPac的有效性，与基线方法相比，平均准确率提高了高达5%。我们的方法通过利用图表示并考虑视觉实体之间的固有结构和关系，为图像分类提供了一种通用且灵活的解决方案，特别是在医学图像领域。


### 论文摘要

Graph neural networks have emerged as a promising paradigm for image processing, yet their performance in image classification tasks is hindered by a limited consideration of the underlying structure and relationships among visual entities. This work presents iPac, a novel approach to introduce a new graph representation of images to enhance graph neural network image classification by recognizing the importance of underlying structure and relationships in medical image classification. iPac integrates various stages, including patch partitioning, feature extraction, clustering, graph construction, and graph-based learning, into a unified network to advance graph neural network image classification. By capturing relevant features and organising them into clusters, we construct a meaningful graph representation that effectively encapsulates the semantics of the image. Experimental evaluation on diverse medical image datasets demonstrates the efficacy of iPac, exhibiting an average accuracy improvement of up to 5% over baseline methods. Our approach offers a versatile and generic solution for image classification, particularly in the realm of medical images, by leveraging the graph representation and accounting for the inherent structure and relationships among visual entities.

---

## 41. Adaptive Dual Prompting: Hierarchical Debiasing for Fairness-aware Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.23469v1](http://arxiv.org/abs/2510.23469v1)

**作者:** Yuhan Yang, Xingbo Fu, Jundong Li

**发布时间:** 2025-10-27

### GPT解析

### 背景

近年来，通过在无标签图数据上进行自监督学习来预训练图神经网络已成为图学习中的广泛采用范式。虽然这种范式对预训练强大的GNN模型有效，但预训练和下游任务之间通常存在目标差距。图提示通过额外的可学习提示来调整预训练的GNN模型以适应特定下游任务，同时保持预训练的GNN模型冻结。

### 目的

解决现有图提示方法在设计提示时往往忽视公平性的问题。预训练的GNN模型会在不同人口统计子群中产生有区别性的节点表示，因为下游图数据在节点属性和图结构中固有地包含偏见。

### 方法

提出了一种自适应双提示(ADPrompt)框架，用于增强预训练GNN模型适应下游任务的公平性。设计了自适应特征校正模块，学习自定义的属性提示，在输入层抑制敏感信息，从源头减少偏见。提出了自适应消息校准模块，在每一层生成结构提示，调整来自邻居节点的消息，实现信息流的动态和软校准。联合优化两个提示模块，以适应预训练的GNN同时增强公平性。

### 主要发现

在四个数据集上使用四种预训练策略进行了广泛的实验来评估ADPrompt的性能。结果表明，ADPrompt在节点分类任务上优于七种基线方法。

### 结论

ADPrompt框架能够有效提升预训练GNN模型在下游任务中的公平性和性能。

### 翻译

近年来，通过在无标签图数据上进行自监督学习来预训练图神经网络已成为图学习中的广泛采用范式。虽然这种范式对预训练强大的GNN模型有效，但预训练和下游任务之间通常存在目标差距。为了弥合这一差距，图提示通过额外的可学习提示来调整预训练的GNN模型以适应特定的下游任务，同时保持预训练的GNN模型冻结。由于最近的图提示方法主要关注增强模型在下游任务上的效用，它们在设计提示进行适应时往往忽视了公平性问题。实际上，预训练的GNN模型会在不同人口统计子群中产生有区别性的节点表示，因为下游图数据在节点属性和图结构中固有地包含偏见。为了解决这个问题，我们提出了一个自适应双提示(ADPrompt)框架，用于增强预训练GNN模型适应下游任务的公平性。为了减轻属性偏见，我们设计了一个自适应特征校正模块，学习自定义的属性提示，在输入层抑制敏感信息，从源头减少偏见。之后，我们提出了一个自适应消息校准模块，在每一层生成结构提示，调整来自邻居节点的消息，实现信息流的动态和软校准。最后，ADPrompt联合优化两个提示模块，以适应预训练的GNN同时增强公平性。我们在四个数据集上使用四种预训练策略进行了广泛的实验来评估ADPrompt的性能。结果表明，我们提出的ADPrompt在节点分类任务上优于七种基线方法。


### 论文摘要

In recent years, pre-training Graph Neural Networks (GNNs) through self-supervised learning on unlabeled graph data has emerged as a widely adopted paradigm in graph learning. Although the paradigm is effective for pre-training powerful GNN models, the objective gap often exists between pre-training and downstream tasks. To bridge this gap, graph prompting adapts pre-trained GNN models to specific downstream tasks with extra learnable prompts while keeping the pre-trained GNN models frozen. As recent graph prompting methods largely focus on enhancing model utility on downstream tasks, they often overlook fairness concerns when designing prompts for adaptation. In fact, pre-trained GNN models will produce discriminative node representations across demographic subgroups, as downstream graph data inherently contains biases in both node attributes and graph structures. To address this issue, we propose an Adaptive Dual Prompting (ADPrompt) framework that enhances fairness for adapting pre-trained GNN models to downstream tasks. To mitigate attribute bias, we design an Adaptive Feature Rectification module that learns customized attribute prompts to suppress sensitive information at the input layer, reducing bias at the source. Afterward, we propose an Adaptive Message Calibration module that generates structure prompts at each layer, which adjust the message from neighboring nodes to enable dynamic and soft calibration of the information flow. Finally, ADPrompt jointly optimizes the two prompting modules to adapt the pre-trained GNN while enhancing fairness. We conduct extensive experiments on four datasets with four pre-training strategies to evaluate the performance of ADPrompt. The results demonstrate that our proposed ADPrompt outperforms seven baseline methods on node classification tasks.

---

## 42. Improving Predictions of Molecular Properties with Graph Featurisation and Heterogeneous Ensemble Models

**论文链接:** [http://arxiv.org/abs/2510.23428v1](http://arxiv.org/abs/2510.23428v1)

**作者:** Michael L. Parker, Samar Mahmoud, Bailey Montefiore, Mario Öeren, Himani Tandon, Charlotte Wharrick, Matthew D. Segall

**发布时间:** 2025-10-27

**DOI:** 10.1021/acs.jcim.5c01844

### GPT解析

### 总结

本研究提出了一种结合图神经网络学习描述符与传统分子描述符的混合方法，通过MetaModel框架整合多种机器学习模型，以提高分子性质预测的准确性。

### 背景

分子性质预测是化学信息学和药物发现中的关键任务，现有的方法通常专注于单一类型的特征或模型架构，可能无法充分利用不同方法的优势。

### 目的

开发一种能够结合多种描述符和机器学习模型的'最佳结合'方法，以提高分子性质预测的准确性，特别是在广泛的回归和分类任务中。

### 方法

1. 引入MetaModel框架聚合来自多样化领先ML模型的预测；2. 设计特征化方案结合任务特定的GNN衍生特征与传统分子描述符；3. 使用图神经网络(GNN)学习分子描述符；4. 结合通用描述符和混合机器学习模型集成。

### 主要发现

在所有测试的回归数据集上优于最先进的ChemProp模型；在9个分类数据集中的6个上优于ChemProp模型；包含从ChemProp衍生的GNN特征可以提升集成模型在多个数据集上的性能；该方法在多种类型的分子性质预测任务中表现优异。

### 结论

为了在广泛问题上实现最佳性能，结合通用描述符与任务特定的学习特征，并使用多样化的机器学习模型进行预测至关重要。

### 翻译

我们通过结合图神经网络(GNN)学习到的分子描述符与通用描述符以及混合的机器学习模型集成，探索了一种用于建模分子性质的'最佳结合'方法。我们引入了一个MetaModel框架来聚合来自多样化领先ML模型的预测。我们提出了一种特征化方案，用于结合任务特定的GNN衍生特征与传统分子描述符。我们证明，在所有测试的回归数据集上以及在9个分类数据集中的6个上，我们的框架优于最先进的ChemProp模型。我们进一步表明，包含从ChemProp衍生的GNN特征可以提升集成模型在多个数据集上的性能，否则这些数据集上的性能会较差。我们得出结论，为了在广泛问题上实现最佳性能，结合通用描述符与任务特定的学习特征，并使用多样化的ML模型进行预测至关重要。


### 论文摘要

We explore a "best-of-both" approach to modelling molecular properties by combining learned molecular descriptors from a graph neural network (GNN) with general-purpose descriptors and a mixed ensemble of machine learning (ML) models. We introduce a MetaModel framework to aggregate predictions from a diverse set of leading ML models. We present a featurisation scheme for combining task-specific GNN-derived features with conventional molecular descriptors.   We demonstrate that our framework outperforms the cutting-edge ChemProp model on all regression datasets tested and 6 of 9 classification datasets. We further show that including the GNN features derived from ChemProp boosts the ensemble model's performance on several datasets where it otherwise would have underperformed. We conclude that to achieve optimal performance across a wide set of problems, it is vital to combine general-purpose descriptors with task-specific learned features and use a diverse set of ML models to make the predictions.

---

## 43. A Novel Framework for Multi-Modal Protein Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.23273v1](http://arxiv.org/abs/2510.23273v1)

**作者:** Runjie Zheng, Zhen Wang, Anjie Qiao, Jiancong Xie, Jiahua Rao, Yuedong Yang

**发布时间:** 2025-10-27

**备注:** 35 pages, 5 figures, 4 tables

### GPT解析

### 总结

DAMPE是一个统一框架，通过最优传输(OT)表示对齐和条件图生成(CGG)信息融合解决了蛋白质功能预测中的跨模态异质性和嘈杂数据问题，在标准基准上取得了优于或匹配最先进方法的性能。

### 背景

准确的蛋白质功能预测需要整合异构的内在信号(如序列和结构)与嘈杂的外部上下文(如蛋白质相互作用和GO术语注释)。然而，两个关键挑战阻碍了有效融合：(i)预训练的内在编码器产生的嵌入之间的跨模态分布不匹配；(ii)外部数据的嘈杂关系图降低了基于GNN的信息聚合效果。

### 目的

开发一个统一框架来解决蛋白质功能预测中的跨模态分布不匹配和嘈杂数据关系图问题，提高蛋白质功能预测的准确性。

### 方法

提出了DAMPE(Diffused and Aligned Multi-modal Protein Embedding)框架，包含两个核心机制：(1)基于最优传输(OT)的表示对齐，建立不同模态内在嵌入空间之间的对应关系，缓解跨模态异质性；(2)基于条件图生成(CGG)的信息融合方法，条件编码器融合对齐的内在嵌入为图重建提供信息提示。理论分析表明CGG目标驱动条件编码器将图感知知识吸收到蛋白质表示中。

### 主要发现

DAMPE在标准GO基准测试上优于或匹配了DPFunc等最先进方法，实现了0.002-0.013 pp的AUPR增益和0.004-0.007 pp的Fmax增益。消融研究表明基于OT的对齐贡献了0.043-0.064 pp的AUPR，基于CGG的融合增加了0.005-0.111 pp的Fmax。

### 结论

DAMPE为稳健的多模态蛋白质表示学习提供了一种可扩展且理论上有依据的方法，显著提高了蛋白质功能预测的准确性。

### 翻译

准确的蛋白质功能预测需要整合异构的内在信号(如序列和结构)与嘈杂的外部上下文(如蛋白质相互作用和GO术语注释)。然而，两个关键挑战阻碍了有效融合：(i)由预训练的内在编码器产生的嵌入之间的跨模态分布不匹配，以及(ii)外部数据的嘈杂关系图降低了基于GNN的信息聚合效果。我们提出了DAMPE(扩散和对齐的多模态蛋白质嵌入)，一个通过两个核心机制解决这些问题的统一框架。首先，我们提出了基于最优传输(OT)的表示对齐，建立了不同模态内在嵌入空间之间的对应关系，有效缓解了跨模态异质性。其次，我们开发了基于条件图生成(CGG)的信息融合方法，其中条件编码器融合对齐的内在嵌入，为图重建提供信息提示。同时，我们的理论分析表明CGG目标驱动条件编码器将其产生的蛋白质表示吸收图感知知识。经验上，DAMPE在标准GO基准测试上优于或匹配了DPFunc等最先进方法，实现了0.002-0.013 pp的AUPR增益和0.004-0.007 pp的Fmax增益。消融研究进一步表明，基于OT的对齐贡献了0.043-0.064 pp的AUPR，而基于CGG的融合增加了0.005-0.111 pp的Fmax。总体而言，DAMPE为稳健的多模态蛋白质表示学习提供了一种可扩展且理论上有依据的方法，显著提高了蛋白质功能预测。


### 论文摘要

Accurate protein function prediction requires integrating heterogeneous intrinsic signals (e.g., sequence and structure) with noisy extrinsic contexts (e.g., protein-protein interactions and GO term annotations). However, two key challenges hinder effective fusion: (i) cross-modal distributional mismatch among embeddings produced by pre-trained intrinsic encoders, and (ii) noisy relational graphs of extrinsic data that degrade GNN-based information aggregation. We propose Diffused and Aligned Multi-modal Protein Embedding (DAMPE), a unified framework that addresses these through two core mechanisms. First, we propose Optimal Transport (OT)-based representation alignment that establishes correspondence between intrinsic embedding spaces of different modalities, effectively mitigating cross-modal heterogeneity. Second, we develop a Conditional Graph Generation (CGG)-based information fusion method, where a condition encoder fuses the aligned intrinsic embeddings to provide informative cues for graph reconstruction. Meanwhile, our theoretical analysis implies that the CGG objective drives this condition encoder to absorb graph-aware knowledge into its produced protein representations. Empirically, DAMPE outperforms or matches state-of-the-art methods such as DPFunc on standard GO benchmarks, achieving AUPR gains of 0.002-0.013 pp and Fmax gains 0.004-0.007 pp. Ablation studies further show that OT-based alignment contributes 0.043-0.064 pp AUPR, while CGG-based fusion adds 0.005-0.111 pp Fmax. Overall, DAMPE offers a scalable and theoretically grounded approach for robust multi-modal protein representation learning, substantially enhancing protein function prediction.

---

## 44. GTR-Mamba: Geometry-to-Tangent Routing for Hyperbolic POI Recommendation

**论文链接:** [http://arxiv.org/abs/2510.22942v1](http://arxiv.org/abs/2510.22942v1)

**作者:** Zhuoxuan Li, Jieyuan Pei, Tangwei Ye, Zhongyuan Lai, Zihan Liu, Fengyuan Xu, Qi Zhang, Liang Hu

**发布时间:** 2025-10-27

**备注:** 14 pages, 8 figures, 4 tables, submitted to ICDE 2026

### GPT解析

### 总结

GTR-Mamba是一种创新的下一个兴趣点推荐框架，通过结合双曲几何和欧几里得切线空间的优势，解决了现有模型难以同时捕捉空间层次结构和时间动态性的问题。

### 背景

下一个兴趣点推荐是现代位置社交网络中的关键任务，现有模型主要基于图神经网络和序列模型，但存在基本局限性。

### 目的

克服现有模型的限制，能够同时捕捉空间选择的内在层次结构和用户特定时间上下文的动态变化，提供更精准的个性化推荐。

### 方法

提出GTR-Mamba框架，利用双曲几何建模静态树状偏好层次结构，在欧几里得切线空间中通过Mamba层处理动态序列更新，并通过跨流形通道融合时空信息引导状态空间模型。

### 主要发现

在三个真实数据集上的实验表明，GTR-Mamba在下一个POI推荐任务上持续优于最先进的基线模型。

### 结论

GTR-Mamba框架有效解决了现有模型在捕捉空间层次结构和时间动态性方面的局限性，提升了推荐的准确性。

### 翻译

下一个兴趣点推荐是现代位置社交网络中的关键任务，旨在对人类移动的复杂决策过程进行建模，为用户的下一个签到位置提供个性化推荐。现有的POI推荐模型主要基于图神经网络和序列模型，已得到广泛研究。然而，这些模型面临一个基本限制：它们难以同时捕捉空间选择的内在层次结构和用户特定时间上下文的动态及不规则变化。为克服这一限制，我们提出了GTR-Mamba，一个用于跨流形条件和路由的新框架。GTR-Mamba利用不同数学空间的不同优势处理不同任务：它在双曲几何中建模静态的树状偏好层次结构，同时将动态序列更新路由到计算稳定且高效的欧几里得切线空间中的新型Mamba层。这一过程由跨流形通道协调，该通道融合时空信息以明确引导状态空间模型，实现对上下文变化的灵活适应。在三个真实数据集上的大量实验表明，GTR-Mamba在下一个POI推荐方面持续优于最先进的基线模型。


### 论文摘要

Next Point-of-Interest (POI) recommendation is a critical task in modern Location-Based Social Networks (LBSNs), aiming to model the complex decision-making process of human mobility to provide personalized recommendations for a user's next check-in location. Existing POI recommendation models, predominantly based on Graph Neural Networks and sequential models, have been extensively studied. However, these models face a fundamental limitation: they struggle to simultaneously capture the inherent hierarchical structure of spatial choices and the dynamics and irregular shifts of user-specific temporal contexts. To overcome this limitation, we propose GTR-Mamba, a novel framework for cross-manifold conditioning and routing. GTR-Mamba leverages the distinct advantages of different mathematical spaces for different tasks: it models the static, tree-like preference hierarchies in hyperbolic geometry, while routing the dynamic sequence updates to a novel Mamba layer in the computationally stable and efficient Euclidean tangent space. This process is coordinated by a cross-manifold channel that fuses spatio-temporal information to explicitly steer the State Space Model (SSM), enabling flexible adaptation to contextual changes. Extensive experiments on three real-world datasets demonstrate that GTR-Mamba consistently outperforms state-of-the-art baseline models in next POI recommendation.

---

## 45. Diffuse to Detect: A Generalizable Framework for Anomaly Detection with Diffusion Models Applications to UAVs and Beyond

**论文链接:** [http://arxiv.org/abs/2510.22928v1](http://arxiv.org/abs/2510.22928v1)

**作者:** Mingze Gong, Juan Du, Jianbang You

**发布时间:** 2025-10-27

### GPT解析

### 总结

DTD是一种创新的异常检测框架，通过适应扩散模型并采用单步过程实现快速精确的异常识别，结合图神经网络捕捉时空异常，并通过双分支架构平衡可扩展性和可解释性。

### 背景

复杂、高维数据（如无人机传感器读数）中的异常检测对于操作安全至关重要，但现有方法在敏感性、可扩展性和捕捉复杂依赖关系方面存在局限性。

### 目的

开发一种创新的异常检测方法，解决现有方法的局限性，实现快速、精确的异常识别。

### 方法

提出Diffuse to Detect (DTD)框架，采用单步扩散过程预测噪声模式；集成图神经网络建模传感器关系为动态图；使用双分支架构（参数化神经网络能量评分和非参数统计方法）平衡可扩展性和可解释性。

### 主要发现

DTD能够在不产生重构错误的情况下快速精确识别异常；在无人机传感器数据、多元时间序列和图像上的评估表明DTD优于现有方法；DTD具有跨不同数据模态的通用性。

### 结论

DTD因其多功能性和适应性，成为安全关键应用（包括工业监测等）的变革性解决方案。

### 翻译

复杂、高维数据（如无人机传感器读数）中的异常检测对于操作安全至关重要，但由于现有方法的敏感性、可扩展性有限且无法捕捉复杂依赖关系，这仍然是一个挑战。我们提出了Diffuse to Detect (DTD)框架，这是一种创新的方法，将扩散模型适应于异常检测，不同于其在具有高推理时间的生成任务中的常规使用。相比之下，DTD采用单步扩散过程来预测噪声模式，能够快速精确地识别异常而不会产生重构错误。这种方法基于稳健的理论基础，将噪声预测与数据分布的得分函数联系起来，确保可靠的偏差检测。通过集成图神经网络将传感器关系建模为动态图，DTD有效捕捉了空间（传感器间）和时间异常。其双分支架构采用基于参数化神经网络的能量评分实现可扩展性，非参数统计方法提供可解释性，在计算效率和透明度之间提供了灵活的权衡。在无人机传感器数据、多元时间序列和图像上的广泛评估证明了DTD优于现有方法的性能，强调了其在不同数据模态上的通用性。这种多功能性及其适应性使DTD成为安全关键应用的变革性解决方案，包括工业监测等。


### 论文摘要

Anomaly detection in complex, high-dimensional data, such as UAV sensor readings, is essential for operational safety but challenging for existing methods due to their limited sensitivity, scalability, and inability to capture intricate dependencies. We propose the Diffuse to Detect (DTD) framework, a novel approach that innovatively adapts diffusion models for anomaly detection, diverging from their conventional use in generative tasks with high inference time. By comparison, DTD employs a single-step diffusion process to predict noise patterns, enabling rapid and precise identification of anomalies without reconstruction errors. This approach is grounded in robust theoretical foundations that link noise prediction to the data distribution's score function, ensuring reliable deviation detection. By integrating Graph Neural Networks to model sensor relationships as dynamic graphs, DTD effectively captures spatial (inter-sensor) and temporal anomalies. Its two-branch architecture, with parametric neural network-based energy scoring for scalability and nonparametric statistical methods for interpretability, provides flexible trade-offs between computational efficiency and transparency. Extensive evaluations on UAV sensor data, multivariate time series, and images demonstrate DTD's superior performance over existing methods, underscoring its generality across diverse data modalities. This versatility, combined with its adaptability, positions DTD as a transformative solution for safety-critical applications, including industrial monitoring and beyond.

---

## 46. FastJAM: a Fast Joint Alignment Model for Images

**论文链接:** [http://arxiv.org/abs/2510.22842v1](http://arxiv.org/abs/2510.22842v1)

**作者:** Omri Hirsch, Ron Shapira Weber, Shira Ifergane, Oren Freifeld

**发布时间:** 2025-10-26

**备注:** Accepted to NeurIPS 2025. Pages 1-10 are the Main Paper. Pages 23-31  are Supplemental Material. FastJAM website -  https://bgu-cs-vil.github.io/FastJAM/

### GPT解析

### 总结

FastJAM是一种快速的基于图的图像联合对齐方法，能够显著降低计算复杂度，实现从小时或分钟到秒级的速度提升，同时保持或提高对齐质量。

### 背景

图像联合对齐(JA)旨在将一组图像对齐到统一坐标系，使语义相似特征出现在对应空间位置。现有方法通常需要长时间训练、大容量模型和大量超参数调整。

### 目的

开发一种快速、高效的图像联合对齐方法，减少计算时间和资源需求，同时保持或提高对齐质量。

### 方法

FastJAM利用现成的图像匹配器计算的对和快速非参数聚类构建图，表示图像内和图像间关键点关系。通过图神经网络传播和聚合这些对应关系，使用图像级池化预测单应性参数。采用逆组合损失消除正则化项需求，避免相关超参数调整。

### 主要发现

实验结果表明，FastJAM在对齐质量方面优于现有现代JA方法，同时将计算时间从小时或分钟减少到几秒钟。

### 结论

FastJAM是一种高效、快速的图像联合对齐方法，能够在保持高质量对齐的同时显著减少计算时间。

### 翻译

图像联合对齐(JA)旨在将一组图像对齐到统一的坐标系中，使语义相似的特征出现在对应的空间位置。大多数现有方法通常需要长时间的训练、大容量模型和大量的超参数调整。我们引入了FastJAM，一种快速的基于图的方法，显著降低了联合对齐任务的计算复杂度。FastJAM利用现成的图像匹配器计算的对，以及快速的非参数聚类，来构建表示图像内和图像间关键点关系的图。图神经网络传播和聚合这些对应关系，通过图像级池化有效地预测每个图像的单应性参数。利用逆组合损失消除了对预测变换的正则化项的需求（因此也避免了与这些项相关的超参数调整），FastJAM能够快速有效地执行图像联合对齐。在几个基准测试上的实验结果表明，FastJAM在对齐质量方面优于现有的现代JA方法，同时将计算时间从小时或分钟减少到几秒钟。我们的代码可在项目网页获取，https://bgu-cs-vil.github.io/FastJAM/


### 论文摘要

Joint Alignment (JA) of images aims to align a collection of images into a unified coordinate frame, such that semantically-similar features appear at corresponding spatial locations. Most existing approaches often require long training times, large-capacity models, and extensive hyperparameter tuning. We introduce FastJAM, a rapid, graph-based method that drastically reduces the computational complexity of joint alignment tasks. FastJAM leverages pairwise matches computed by an off-the-shelf image matcher, together with a rapid nonparametric clustering, to construct a graph representing intra- and inter-image keypoint relations. A graph neural network propagates and aggregates these correspondences, efficiently predicting per-image homography parameters via image-level pooling. Utilizing an inverse-compositional loss, that eliminates the need for a regularization term over the predicted transformations (and thus also obviates the hyperparameter tuning associated with such terms), FastJAM performs image JA quickly and effectively. Experimental results on several benchmarks demonstrate that FastJAM achieves results better than existing modern JA methods in terms of alignment quality, while reducing computation time from hours or minutes to mere seconds. Our code is available at our project webpage, https://bgu-cs-vil.github.io/FastJAM/

---

## 47. Graph Neural Network Assisted Genetic Algorithm for Structural Dynamic Response and Parameter Optimization

**论文链接:** [http://arxiv.org/abs/2510.22839v1](http://arxiv.org/abs/2510.22839v1)

**作者:** Sagnik Mukherjee

**发布时间:** 2025-10-26

**备注:** 13 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种结合图神经网络(GNN)代理模型和遗传算法(GA)优化器的混合数据驱动框架，用于优化结构参数（质量、刚度和阻尼系数）。该方法克服了传统数值方法在迭代优化任务中计算成本高的问题，实现了强收敛性、良好泛化能力和显著降低的计算成本，为自动化和智能结构设计提供了有效途径。

### 背景

结构参数（质量m、刚度k和阻尼系数c）的优化对设计高效、有韧性和稳定的结构至关重要。传统的数值方法如有限元法(FEM)和计算流体动力学(CFD)模拟虽能提供高精度结果，但在迭代优化任务中计算成本高昂，因为每次评估都需要为每个参数组合求解控制方程。

### 目的

开发一种混合数据驱动框架，结合图神经网络(GNN)代理模型和遗传算法(GA)优化器，以克服传统数值方法在结构参数优化中的计算成本高问题，实现更高效、准确的参数优化。

### 方法

研究采用了一种混合数据驱动框架，包括：1) 使用图神经网络(GNN)作为代理模型，学习结构参数与动态位移响应之间的非线性映射；2) 使用Newmark Beta方法生成具有不同质量、刚度和阻尼配置的单自由度(SDOF)系统响应数据集；3) 应用遗传算法(GA)通过最小化预测位移和提高动态稳定性来搜索全局最优参数集。

### 主要发现

GNN和GA框架实现了强收敛性、良好的泛化能力，并显著降低了计算成本，相比传统模拟方法具有明显优势。该方法能够准确学习结构参数与动态响应之间的复杂关系，实现快速预测和优化。

### 结论

结合机器学习代理与进化优化的方法在自动化和智能结构设计中具有显著有效性。该框架为结构参数优化提供了一种高效、准确的解决方案，克服了传统数值方法的计算瓶颈。

### 翻译

结构参数（如质量m、刚度k和阻尼系数c）的优化对设计高效、有韧性和稳定的结构至关重要。传统的数值方法，包括有限元法(FEM)和计算流体动力学(CFD)模拟，能提供高精度结果，但在迭代优化任务中计算成本高，因为每次评估都需要为每个参数组合求解控制方程。本研究提出了一种混合数据驱动框架，结合图神经网络(GNN)代理模型和遗传算法(GA)优化器来克服这些挑战。GNN被训练来准确学习结构参数和动态位移响应之间的非线性映射，从而能够快速预测而无需重复求解系统方程。使用Newmark Beta方法生成了具有不同质量、刚度和阻尼配置的单自由度(SDOF)系统响应数据集。然后，GA通过最小化预测位移和提高动态稳定性来搜索全局最优参数集。结果表明，与传统模拟相比，GNN和GA框架实现了强收敛性、良好的泛化能力和显著降低的计算成本。这种方法强调了将机器学习代理与进化优化相结合在自动化和智能结构设计中的有效性。


### 论文摘要

The optimization of structural parameters, such as mass(m), stiffness(k), and damping coefficient(c), is critical for designing efficient, resilient, and stable structures. Conventional numerical approaches, including Finite Element Method (FEM) and Computational Fluid Dynamics (CFD) simulations, provide high-fidelity results but are computationally expensive for iterative optimization tasks, as each evaluation requires solving the governing equations for every parameter combination. This study proposes a hybrid data-driven framework that integrates a Graph Neural Network (GNN) surrogate model with a Genetic Algorithm (GA) optimizer to overcome these challenges. The GNN is trained to accurately learn the nonlinear mapping between structural parameters and dynamic displacement responses, enabling rapid predictions without repeatedly solving the system equations. A dataset of single-degree-of-freedom (SDOF) system responses is generated using the Newmark Beta method across diverse mass, stiffness, and damping configurations. The GA then searches for globally optimal parameter sets by minimizing predicted displacements and enhancing dynamic stability. Results demonstrate that the GNN and GA framework achieves strong convergence, robust generalization, and significantly reduced computational cost compared to conventional simulations. This approach highlights the effectiveness of combining machine learning surrogates with evolutionary optimization for automated and intelligent structural design.

---

## 48. Policies over Poses: Reinforcement Learning based Distributed Pose-Graph Optimization for Multi-Robot SLAM

**论文链接:** [http://arxiv.org/abs/2510.22740v1](http://arxiv.org/abs/2510.22740v1)

**作者:** Sai Krishna Ghanta, Ramviyas Parasuraman

**发布时间:** 2025-10-26

**备注:** IEEE International Symposium on Multi-Robot & Multi-Agent Systems  (MRS) 2025

### GPT解析

### 总结

该论文提出了一种基于多智能体强化学习的分布式姿态图优化框架，解决了传统方法收敛到局部最小值的问题，显著提高了轨迹估计的准确性和计算效率。

### 背景

分布式姿态图优化（PGO）是多机器人同时定位与地图构建（SLAM）中精确轨迹估计的基础。传统迭代方法将高度非凸优化目标线性化，需要重复求解正规方程，通常收敛到局部最小值，从而产生次优估计。

### 目的

开发一个可扩展、抗异常值的分布式平面PGO框架，利用多智能体强化学习（MARL）来提高轨迹估计的准确性和效率。

### 方法

将分布式PGO建模为部分可观察马尔可夫博弈，每个机器人运行带有自适应边缘门控的循环边缘条件图神经网络（GNN）编码器来去噪，通过混合策略利用先验动作记忆和图嵌入优化姿态，最后使用一致性方案解决机器人间的不一致。

### 主要发现

在合成和真实世界数据集上的评估表明，该方法比最先进的分布式PGO框架平均减少37.5%的全局目标，同时将推理效率提高至少6倍；单个学习策略无需重新训练即可扩展到更大的机器人团队。

### 结论

基于MARL的分布式PGO框架在提高轨迹估计精度的同时显著增强了计算效率，具有良好的可扩展性和实用性。

### 翻译

我们考虑分布式姿态图优化（PGO）问题，这是多机器人同时定位与地图构建（SLAM）中精确轨迹估计的基础。传统迭代方法将高度非凸优化目标线性化，需要重复求解正规方程，通常收敛到局部最小值，从而产生次优估计。我们提出了一种使用多智能体强化学习（MARL）的可扩展、抗异常值的分布式平面PGO框架。我们将分布式PGO构建为定义在局部姿态图上的部分可观察马尔可夫博弈，其中每个动作优化单个边的姿态估计。图分区器分解全局姿态图，每个机器人运行带有自适应边缘门控的循环边缘条件图神经网络（GNN）编码器来去噪噪声边缘。机器人通过利用先验动作记忆和图嵌入的混合策略顺序优化姿态。在局部图校正后，使用一致性方案解决机器人间的不一致，产生全局一致的估计。我们在综合的合成和真实世界数据集上的广泛评估表明，我们学习的基于MARL的智能体比最先进的分布式PGO框架平均减少37.5%的全局目标，同时将推理效率提高至少6倍。我们还证明了智能体复制允许单个学习策略无需重新训练即可轻松扩展到更大的机器人团队。代码可在https://github.com/herolab-uga/policies-over-poses公开获取。


### 论文摘要

We consider the distributed pose-graph optimization (PGO) problem, which is fundamental in accurate trajectory estimation in multi-robot simultaneous localization and mapping (SLAM). Conventional iterative approaches linearize a highly non-convex optimization objective, requiring repeated solving of normal equations, which often converge to local minima and thus produce suboptimal estimates. We propose a scalable, outlier-robust distributed planar PGO framework using Multi-Agent Reinforcement Learning (MARL). We cast distributed PGO as a partially observable Markov game defined on local pose-graphs, where each action refines a single edge's pose estimate. A graph partitioner decomposes the global pose graph, and each robot runs a recurrent edge-conditioned Graph Neural Network (GNN) encoder with adaptive edge-gating to denoise noisy edges. Robots sequentially refine poses through a hybrid policy that utilizes prior action memory and graph embeddings. After local graph correction, a consensus scheme reconciles inter-robot disagreements to produce a globally consistent estimate. Our extensive evaluations on a comprehensive suite of synthetic and real-world datasets demonstrate that our learned MARL-based actors reduce the global objective by an average of 37.5% more than the state-of-the-art distributed PGO framework, while enhancing inference efficiency by at least 6X. We also demonstrate that actor replication allows a single learned policy to scale effortlessly to substantially larger robot teams without any retraining. Code is publicly available at https://github.com/herolab-uga/policies-over-poses.

---

## 49. SpoofTrackBench: Interpretable AI for Spoof-Aware UAV Tracking and Benchmarking

**论文链接:** [http://arxiv.org/abs/2510.22726v1](http://arxiv.org/abs/2510.22726v1)

**作者:** Van Le, Tan Le

**发布时间:** 2025-10-26

### GPT解析

### 总结

SpoofTrackBench是一个可重现、模块化的基准测试，用于评估雷达欺骗下的实时定位和跟踪系统的对抗鲁棒性。

### 背景

雷达欺骗攻击对实时定位和跟踪系统构成威胁，需要有效的评估方法。

### 目的

开发一个基准测试框架，评估不同跟踪架构在雷达欺骗攻击下的性能和鲁棒性。

### 方法

使用Hampton University Skyler Radar Sensor数据集，模拟漂移、幽灵和镜像类型的欺骗攻击，并使用联合概率数据关联(JPDA)和全局最近邻(GNN)架构评估跟踪器性能。框架分离干净和欺骗的检测流，可视化轨迹偏离，并量化分配错误。

### 主要发现

通过聚类叠加、注入感知时间线和场景自适应可视化，实现了不同欺骗类型和配置下的结果可解释性。评估图表和日志自动导出，确保可重现性。

### 结论

SpoofTrackBench为开放、合乎道德的欺骗感知跟踪管道基准测试设定了新标准，实现了严格的跨架构分析和社区验证。

### 翻译

SpoofTrackBench是一个可重现、模块化的基准测试，用于评估雷达欺骗下的实时定位和跟踪(RTLS)系统的对抗鲁棒性。利用Hampton University Skyler雷达传感器数据集，我们模拟了漂移、幽灵和镜像类型的欺骗攻击，并使用联合概率数据关联(JPDA)和全局最近邻(GNN)架构评估跟踪器性能。我们的框架分离干净和欺骗的检测流，可视化欺骗引起的轨迹偏离，并通过直接偏离真相的指标量化分配错误。聚类叠加、注入感知时间线和场景自适应可视化使不同欺骗类型和配置下的结果可解释。评估图表和日志自动导出，用于可重现的比较。SpoofTrackBench为开放、合乎道德的欺骗感知跟踪管道基准测试设定了新标准，实现了严格的跨架构分析和社区验证。


### 论文摘要

SpoofTrackBench is a reproducible, modular benchmark for evaluating adversarial robustness in real-time localization and tracking (RTLS) systems under radar spoofing. Leveraging the Hampton University Skyler Radar Sensor dataset, we simulate drift, ghost, and mirror-type spoofing attacks and evaluate tracker performance using both Joint Probabilistic Data Association (JPDA) and Global Nearest Neighbor (GNN) architectures. Our framework separates clean and spoofed detection streams, visualizes spoof-induced trajectory divergence, and quantifies assignment errors via direct drift-from-truth metrics. Clustering overlays, injection-aware timelines, and scenario-adaptive visualizations enable interpretability across spoof types and configurations. Evaluation figures and logs are auto-exported for reproducible comparison. SpoofTrackBench sets a new standard for open, ethical benchmarking of spoof-aware tracking pipelines, enabling rigorous cross-architecture analysis and community validation.

---

## 50. If You Want to Be Robust, Be Wary of Initialization

**论文链接:** [http://arxiv.org/abs/2510.22652v1](http://arxiv.org/abs/2510.22652v1)

**作者:** Sofiane Ennadir, Johannes F. Lutzeyer, Michalis Vazirgiannis, El Houcine Bergou

**发布时间:** 2025-10-26

**备注:** Accepted at NeurIPS 2024

### GPT解析

### 总结

本研究探讨了图神经网络(GNNs)的权重初始化和超参数对对抗性鲁棒性的影响，发现适当的初始化方法可显著提升模型防御能力。

### 背景

图神经网络在各种图相关任务中表现出色，但容易受到对抗性扰动的攻击。现有防御策略主要关注预处理技术和自适应消息传递方案，而忽略了权重初始化的影响。

### 目的

探索权重初始化和相关超参数(如训练周期)对模型鲁棒性的影响，建立初始化策略与网络抗扰动能力之间的理论联系。

### 方法

引入连接初始化策略和网络鲁棒性的理论框架，分析初始权重、训练周期与模型脆弱性的关系，并将框架扩展到深度神经网络。通过多种模型和真实数据集的对抗性攻击实验验证理论发现。

### 主要发现

适当的权重初始化不仅能保证模型在干净数据集上的性能，还能显著提升对抗性防御能力，与其他初始化方法相比性能差距可达50%。

### 结论

权重初始化是提升图神经网络对抗鲁棒性的重要因素，为防御对抗性攻击提供了超越传统机制的新视角。

### 翻译

图神经网络(GNNs)在各种图相关任务中表现出色，然而人们对它们容易受到对抗性扰动的担忧依然存在。虽然主流防御策略主要关注预处理技术和自适应消息传递方案，但本研究探讨了一个尚未充分研究的维度：权重初始化及相关超参数(如训练周期)对模型鲁棒性的影响。我们引入了一个理论框架，连接初始化策略与网络对抗扰动的恢复能力。我们的分析揭示了初始权重、训练周期数量与模型脆弱性之间的直接关系，为对抗性鲁棒性提供了超越传统防御机制的新见解。虽然我们的主要关注点是图神经网络，但我们扩展了理论框架，提供了一个适用于深度神经网络的通用上界。跨越多种模型和真实世界数据集的广泛实验( subjected to various adversarial attacks)验证了我们的发现。我们说明，选择适当的初始化不仅能确保在干净数据集上的性能，还能增强模型对抗扰动的鲁棒性，与其他初始化方法相比观察到高达50%的性能差距。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated remarkable performance across a spectrum of graph-related tasks, however concerns persist regarding their vulnerability to adversarial perturbations. While prevailing defense strategies focus primarily on pre-processing techniques and adaptive message-passing schemes, this study delves into an under-explored dimension: the impact of weight initialization and associated hyper-parameters, such as training epochs, on a model's robustness. We introduce a theoretical framework bridging the connection between initialization strategies and a network's resilience to adversarial perturbations. Our analysis reveals a direct relationship between initial weights, number of training epochs and the model's vulnerability, offering new insights into adversarial robustness beyond conventional defense mechanisms. While our primary focus is on GNNs, we extend our theoretical framework, providing a general upper-bound applicable to Deep Neural Networks. Extensive experiments, spanning diverse models and real-world datasets subjected to various adversarial attacks, validate our findings. We illustrate that selecting appropriate initialization not only ensures performance on clean datasets but also enhances model robustness against adversarial perturbations, with observed gaps of up to 50\% compared to alternative initialization approaches.

---

## 51. Enhancing Graph Classification Robustness with Singular Pooling

**论文链接:** [http://arxiv.org/abs/2510.22643v1](http://arxiv.org/abs/2510.22643v1)

**作者:** Sofiane Ennadir, Oleg Smirnov, Yassine Abbahaddou, Lele Cao, Johannes F. Lutzeyer

**发布时间:** 2025-10-26

**备注:** Accepted at Neurips 2025

### GPT解析

### 总结

本研究探讨了图神经网络在图分类任务中的对抗鲁棒性问题，特别关注了池化操作在塑造鲁棒性方面的作用，并提出了一种名为鲁棒奇异池化(RS-Pool)的新策略。

### 背景

图神经网络在图表示学习任务中表现出色，但其在图分类任务中的对抗鲁棒性研究相对较少，尤其是在与节点分类相比时。大多数现有防御方法集中在消息传递组件上，而忽略了池化操作的作用。

### 目的

研究池化操作在图神经网络对抗鲁棒性中的角色，并开发一种能够提高图分类任务鲁棒性的新池化策略。

### 方法

对标准扁平池化方法(求和、平均和最大值)进行理论分析，推导对抗风险上限，确定不同攻击场景和图结构下的脆弱性。基于这些见解，提出利用节点嵌入矩阵主奇异向量构建鲁棒图级表示的RS-Pool策略。

### 主要发现

RS-Pool在遭受最先进对抗攻击时，比其他池化方法提供更好的鲁棒性，同时保持有竞争力的干净准确率。该策略与模型无关，可通过幂迭代有效实现，理论分析表明其具有良好的鲁棒性特性。

### 结论

池化操作在图神经网络的对抗鲁棒性中扮演着重要角色，所提出的RS-Pool方法能够有效提高图分类任务的鲁棒性，代码已在GitHub公开。

### 翻译

图神经网络(GNNs)在一系列图表示学习任务中已取得强大性能，然而与节点分类相比，其在图分类中的对抗鲁棒性研究仍然不足。虽然大多数现有防御方法集中在消息传递组件上，但本研究调查了池化操作在塑造鲁棒性中被忽视的作用。我们对标准扁平池化方法(求和、平均和最大值)进行了理论分析，推导了它们对抗风险的上限，并确定了它们在不同攻击场景和图结构下的脆弱性。受这些见解启发，我们提出了鲁棒奇异池化(RS-Pool)，这是一种新颖的池化策略，利用节点嵌入矩阵的主奇异向量构建鲁棒的图级表示。我们从理论上研究了RS-Pool的鲁棒性，并解释了所得界限，从而加深对我们提出的池化算子的理解。虽然我们的分析集中在图卷积网络(GCNs)上，但RS-Pool是与模型无关的，可以通过幂迭代有效实现。真实世界基准测试的实证结果表明，当遭受最先进的对抗攻击时，RS-Pool比考虑的池化方法提供更好的鲁棒性，同时保持有竞争力的干净准确率。我们的代码已在GitHub公开。


### 论文摘要

Graph Neural Networks (GNNs) have achieved strong performance across a range of graph representation learning tasks, yet their adversarial robustness in graph classification remains underexplored compared to node classification. While most existing defenses focus on the message-passing component, this work investigates the overlooked role of pooling operations in shaping robustness. We present a theoretical analysis of standard flat pooling methods (sum, average and max), deriving upper bounds on their adversarial risk and identifying their vulnerabilities under different attack scenarios and graph structures. Motivated by these insights, we propose \textit{Robust Singular Pooling (RS-Pool)}, a novel pooling strategy that leverages the dominant singular vector of the node embedding matrix to construct a robust graph-level representation. We theoretically investigate the robustness of RS-Pool and interpret the resulting bound leading to improved understanding of our proposed pooling operator. While our analysis centers on Graph Convolutional Networks (GCNs), RS-Pool is model-agnostic and can be implemented efficiently via power iteration. Empirical results on real-world benchmarks show that RS-Pool provides better robustness than the considered pooling methods when subject to state-of-the-art adversarial attacks while maintaining competitive clean accuracy. Our code is publicly available at:\href{https://github.com/king/rs-pool}{https://github.com/king/rs-pool}.

---

## 52. Iteratively Refined Early Interaction Alignment for Subgraph Matching based Graph Retrieval

**论文链接:** [http://arxiv.org/abs/2510.22538v1](http://arxiv.org/abs/2510.22538v1)

**作者:** Ashwin Ramachandran, Vaibhav Raj, Indrayumna Roy, Soumen Chakrabarti, Abir De

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了IsoNet++，一种基于子图同构的改进图检索方法，通过技术创新显著提升了检索性能。

### 背景

基于子图同构的图检索在场景图检索、分子指纹检测和电路设计等领域有广泛应用。Roy等人提出的IsoNet是一种后期交互模型，用于子图匹配，它先独立计算每对图的节点和边嵌入，再计算可训练的对齐映射。

### 目的

开发IsoNet++，一种早期交互图神经网络，通过技术创新改进现有的子图匹配方法，提高图检索的准确性。

### 方法

IsoNet++包含三个技术创新：1)通过在两个输入图内部和之间传递消息计算所有节点嵌入，由节点间的单射对齐引导；2)以惰性方式在多轮中更新对齐，每轮基于当前对齐状态从头运行逐层GNN；3)引入节点对伙伴交互概念，将节点对而非单个节点视为潜在伙伴。

### 主要发现

实验表明，随着轮次增加，对齐 progressively 得到细化，检索性能显著优于现有方法，且所有三个创新都对提升准确性有贡献。

### 结论

IsoNet++通过三个技术创新显著提高了图检索性能，代码和数据集已公开在https://github.com/structlearning/isonetpp。

### 翻译

基于子图同构的图检索在场景图检索、分子指纹检测和电路设计等现实世界应用中有多种应用。Roy等人提出了IsoNet，一种用于子图匹配的后期交互模型，它首先独立计算每对图中节点和边的嵌入，然后计算可训练的对齐映射。本文提出了IsoNet++，一种早期交互图神经网络(GNN)，基于几项技术创新。首先，我们通过在两个输入图内部和之间传递消息来计算所有节点的嵌入，这些消息由节点之间的单射对齐引导。其次，我们在多轮中以惰性方式更新对齐。每轮中，我们基于当前对齐状态从头开始运行逐层GNN。一轮GNN完成后，我们使用最后一层嵌入更新对齐，然后进入下一轮。第三，IsoNet++引入了节点对伙伴交互的新概念。传统早期交互计算节点与另一图中潜在伙伴之间的注意力，注意力控制跨图传递的消息。相比之下，我们将节点对（而非单个节点）视为潜在伙伴。一个图中节点间存在边而另一图中不存在，提供了细化对齐的重要信号。我们在多个数据集上的实验表明，对齐随着轮次的推进逐步细化，检索性能显著优于现有方法。我们证明了所有三个创新都对提高准确性做出了贡献。我们的代码和数据集已在https://github.com/structlearning/isonetpp公开。


### 论文摘要

Graph retrieval based on subgraph isomorphism has several real-world applications such as scene graph retrieval, molecular fingerprint detection and circuit design. Roy et al. [35] proposed IsoNet, a late interaction model for subgraph matching, which first computes the node and edge embeddings of each graph independently of paired graph and then computes a trainable alignment map. Here, we present IsoNet++, an early interaction graph neural network (GNN), based on several technical innovations. First, we compute embeddings of all nodes by passing messages within and across the two input graphs, guided by an injective alignment between their nodes. Second, we update this alignment in a lazy fashion over multiple rounds. Within each round, we run a layerwise GNN from scratch, based on the current state of the alignment. After the completion of one round of GNN, we use the last-layer embeddings to update the alignments, and proceed to the next round. Third, IsoNet++ incorporates a novel notion of node-pair partner interaction. Traditional early interaction computes attention between a node and its potential partners in the other graph, the attention then controlling messages passed across graphs. In contrast, we consider node pairs (not single nodes) as potential partners. Existence of an edge between the nodes in one graph and non-existence in the other provide vital signals for refining the alignment. Our experiments on several datasets show that the alignments get progressively refined with successive rounds, resulting in significantly better retrieval performance than existing methods. We demonstrate that all three innovations contribute to the enhanced accuracy. Our code and datasets are publicly available at https://github.com/structlearning/isonetpp.

---

## 53. Toward Robust Signed Graph Learning through Joint Input-Target Denoising

**论文链接:** [http://arxiv.org/abs/2510.22513v1](http://arxiv.org/abs/2510.22513v1)

**作者:** Junran Wu, Beng Chin Ooi, Ke Xu

**发布时间:** 2025-10-26

**备注:** ACM MM 2025

### GPT解析

### 总结

本文提出了RIDGE框架，一种通过联合去噪图输入和监督目标来实现的鲁棒符号图学习方法，有效提高了SGNN在噪声环境下的鲁棒性。

### 背景

符号图神经网络（SGNNs）被广泛用于分析包含正负链接的符号图中的复杂模式。鉴于现实世界连接的噪声特性，SGNN的鲁棒性也已成为一个关键研究领域。在经验属性监督下，图结构学习已在符号图表示学习中显示出其鲁棒性，然而，缺乏理论指导的鲁棒SGNN研究仍然较少。

### 目的

受图信息瓶颈（GIB）在信息提取中的成功启发，提出一种有理论指导的鲁棒SGNN框架，通过联合去噪图输入和监督目标来提高鲁棒性。

### 方法

作者提出了RIDGE框架，扩展了GIB理论以支持目标空间去噪，因为输入和目标空间都存在噪声。通过重参数化机制和变分近似产生的可处理目标函数，RIDGE有效清理输入数据和监督目标。

### 主要发现

在四个常用的符号图数据集上的广泛验证表明，RIDGE在各种噪声水平下显著提高了流行SGNN模型的鲁棒性。

### 结论

RIDGE框架能够有效提高SGNN在噪声环境下的鲁棒性，为符号图分析提供了新的理论指导和实践方法。

### 翻译

符号图神经网络（SGNNs）被广泛用于分析包含正负链接的符号图中的复杂模式。鉴于现实世界连接的噪声特性，SGNN的鲁棒性也已成为一个关键研究领域。在经验属性监督下，图结构学习已在符号图表示学习中显示出其鲁棒性，然而，缺乏理论指导的鲁棒SGNN研究仍然较少。受图信息瓶颈（GIB）在信息提取中的成功启发，我们提出了RIDGE，一种通过联合去噪图输入和监督目标来实现鲁棒符号图学习的新框架。与基本GIB不同，我们扩展了GIB理论，使其能够对目标空间进行去噪，因为输入和目标空间都存在噪声。在实例化中，RIDGE通过重参数化机制和变分近似产生的可处理目标函数有效清理输入数据和监督目标。我们在四个常用的符号图数据集上广泛验证了我们的方法，结果表明，在各种噪声水平下，RIDGE显著提高了流行SGNN模型的鲁棒性。


### 论文摘要

Signed Graph Neural Networks (SGNNs) are widely adopted to analyze complex patterns in signed graphs with both positive and negative links. Given the noisy nature of real-world connections, the robustness of SGNN has also emerged as a pivotal research area. Under the supervision of empirical properties, graph structure learning has shown its robustness on signed graph representation learning, however, there remains a paucity of research investigating a robust SGNN with theoretical guidance. Inspired by the success of graph information bottleneck (GIB) in information extraction, we propose RIDGE, a novel framework for Robust sI gned graph learning through joint Denoising of Graph inputs and supervision targEts. Different from the basic GIB, we extend the GIB theory with the capability of target space denoising as the co-existence of noise in both input and target spaces. In instantiation, RIDGE effectively cleanses input data and supervision targets via a tractable objective function produced by reparameterization mechanism and variational approximation. We extensively validate our method on four prevalent signed graph datasets, and the results show that RIDGE clearly improves the robustness of popular SGNN models under various levels of noise.

---

## 54. GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.22451v1](http://arxiv.org/abs/2510.22451v1)

**作者:** Xingbo Fu, Zhenyu Lei, Zihan Chen, Binchi Zhang, Chuxu Zhang, Jundong Li

**发布时间:** 2025-10-25

**备注:** Accepted by the 39 Annual Conference on Neural Information Processing  Systems (NeurIPS 2025)

### GPT解析

### 总结

该研究提出了GraphTOP框架，开创性地探索了面向图拓扑的提示方法，通过修改图拓扑而非仅修改节点特征来适配预训练的图神经网络模型。

### 背景

图神经网络(GNNs)通过大规模图数据学习强大的图表示，'预训练、适配'方案是训练强大GNN的常见模式。在适配阶段，图提示是一种有效策略，可修改输入图数据同时保持预训练GNN模型冻结。

### 目的

进行图提示在图拓扑方面的开创性研究，提出第一个图拓扑导向提示(GraphTOP)框架，有效适配预训练GNN模型用于下游任务。

### 方法

将拓扑导向提示表述为多跳局部子图中的边重连问题，通过重参数化将其松弛到连续概率空间，同时确保紧密松弛并保持图的稀疏性。

### 主要发现

在四种预训练策略下的五个图数据集上进行大量实验，GraphTOP在多个节点分类数据集上优于六个基线方法。

### 结论

GraphTOP框架在图提示领域，特别是面向拓扑的提示方面表现出色。

### 翻译

图神经网络(GNNs)通过大规模图数据学习强大的图表示，革新了图学习领域。作为训练强大GNN的常见模式，'预训练、适配'方案首先在无标签图数据上预训练GNN，然后将其适配到特定下游任务。在适配阶段，图提示是一种有效策略，即可学习提示修改输入图数据，同时保持预训练GNN模型冻结。通常，现有图提示研究主要关注面向特征的方法，将图提示应用于节点特征或隐藏表示。然而，这些研究表现次优，因为它们持续忽视了面向拓扑提示的潜力，后者通过修改图拓扑来适配预训练GNN。在本研究中，我们从图拓扑角度对图提示进行了开创性研究。我们提出了第一个图拓扑导向提示(GraphTOP)框架，有效适配预训练GNN模型用于下游任务。更具体地说，我们将拓扑导向提示表述为多跳局部子图中的边重连问题，并通过重参数化将其松弛到连续概率空间，同时确保紧密松弛并保持图的稀疏性。在四种预训练策略下的五个图数据集上进行的大量实验表明，我们提出的GraphTOP在多个节点分类数据集上优于六个基线方法。我们的代码可在https://github.com/xbfu/GraphTOP获取。


### 论文摘要

Graph Neural Networks (GNNs) have revolutionized the field of graph learning by learning expressive graph representations from massive graph data. As a common pattern to train powerful GNNs, the "pre-training, adaptation" scheme first pre-trains GNNs over unlabeled graph data and subsequently adapts them to specific downstream tasks. In the adaptation phase, graph prompting is an effective strategy that modifies input graph data with learnable prompts while keeping pre-trained GNN models frozen. Typically, existing graph prompting studies mainly focus on *feature-oriented* methods that apply graph prompts to node features or hidden representations. However, these studies often achieve suboptimal performance, as they consistently overlook the potential of *topology-oriented* prompting, which adapts pre-trained GNNs by modifying the graph topology. In this study, we conduct a pioneering investigation of graph prompting in terms of graph topology. We propose the first **Graph** **T**opology-**O**riented **P**rompting (GraphTOP) framework to effectively adapt pre-trained GNN models for downstream tasks. More specifically, we reformulate topology-oriented prompting as an edge rewiring problem within multi-hop local subgraphs and relax it into the continuous probability space through reparameterization while ensuring tight relaxation and preserving graph sparsity. Extensive experiments on five graph datasets under four pre-training strategies demonstrate that our proposed GraphTOP outshines six baselines on multiple node classification datasets. Our code is available at https://github.com/xbfu/GraphTOP.

---

## 55. Beyond Augmentation: Leveraging Inter-Instance Relation in Self-Supervised Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.22322v1](http://arxiv.org/abs/2510.22322v1)

**作者:** Ali Javidani, Babak Nadjar Araabi, Mohammad Amin Sadeghi

**发布时间:** 2025-10-25

**DOI:** 10.1109/LSP.2025.3610549

**备注:** Accepted in IEEE Signal Processing Letters, 2025

### GPT解析

### 总结

本文提出了一种将图论与自监督表示学习相结合的新方法，通过构建k近邻图并利用图神经网络进行表示学习，显著提升了模型性能。

### 背景

传统自监督表示学习方法主要关注通过数据增强技术生成的实例内变化，但往往忽略了实例间的重要关系信息。

### 目的

在保留实例内属性的同时，有效捕获实例间关系，并通过图神经网络实现更广泛的上下文集成，提升表示学习效果。

### 方法

在预训练阶段为教师和学生流构建k近邻(KNN)图，其中节点表示样本及其潜在表示，边编码实例间的相似性；在表示细化阶段，使用图神经网络在多个跃点间传播消息，实现更广泛的上下文整合。

### 主要发现

在CIFAR-10、ImageNet-100和ImageNet-1K三个数据集上，分别实现了7.3%、3.2%和1.0%的准确率提升，显著优于现有最先进方法。

### 结论

基于图的机制在自监督表示学习中具有显著优势，能够有效提升模型性能，代码已公开可获取。

### 翻译

这篇论文介绍了一种将图论整合到自监督表示学习中的新方法。传统方法专注于应用增强技术生成的实例内变化。然而，它们常常忽略了重要的实例间关系。虽然我们的方法保留了实例内属性，但通过在预训练期间为教师和学生流构建k近邻(KNN)图，进一步捕获了实例间关系。在这些图中，节点表示样本及其潜在表示，边编码实例之间的相似性。预训练后，执行表示细化阶段。在此阶段，图神经网络不仅可以在直接邻居之间传播消息，还可以跨越多个跃点，从而实现更广泛的上下文集成。在CIFAR-10、ImageNet-100和ImageNet-1K上的实验结果分别比最先进的方法提高了7.3%、3.2%和1.0%的准确率。这些结果突显了所提出的基于图机制的有效性。代码可在https://github.com/alijavidani/SSL-GraphNNCLR公开获取。


### 论文摘要

This paper introduces a novel approach that integrates graph theory into self-supervised representation learning. Traditional methods focus on intra-instance variations generated by applying augmentations. However, they often overlook important inter-instance relationships. While our method retains the intra-instance property, it further captures inter-instance relationships by constructing k-nearest neighbor (KNN) graphs for both teacher and student streams during pretraining. In these graphs, nodes represent samples along with their latent representations. Edges encode the similarity between instances. Following pretraining, a representation refinement phase is performed. In this phase, Graph Neural Networks (GNNs) propagate messages not only among immediate neighbors but also across multiple hops, thereby enabling broader contextual integration. Experimental results on CIFAR-10, ImageNet-100, and ImageNet-1K demonstrate accuracy improvements of 7.3%, 3.2%, and 1.0%, respectively, over state-of-the-art methods. These results highlight the effectiveness of the proposed graph based mechanism. The code is publicly available at https://github.com/alijavidani/SSL-GraphNNCLR.

---

## 56. Does Homophily Help in Robust Test-time Node Classification?

**论文链接:** [http://arxiv.org/abs/2510.22289v1](http://arxiv.org/abs/2510.22289v1)

**作者:** Yan Jiang, Ruihong Qiu, Zi Huang

**发布时间:** 2025-10-25

### GPT解析

### 总结

这项研究提出了一种名为GrapHoST的测试时图结构转换方法，通过调整测试图中的同质性来提高预训练图神经网络在节点分类任务上的鲁棒性和性能，无需重新训练模型。

### 背景

同质性是现实世界图的基本属性，但现有方法主要关注训练图的学习。然而，测试图常面临数据质量问题和分布偏移，如社交网络中不同地区用户的领域偏移和引文网络中的时间演化偏移，这些因素会降低预训练模型的鲁棒性。

### 目的

提高预训练GNN模型在面对测试时数据质量问题和分布偏移情况下的鲁棒性和性能。

### 方法

提出GrapHoST方法，开发同质性预测器来区分测试边，通过预测同质性得分的置信度实现自适应的测试时图结构转换。

### 主要发现

通过增加同质性图中的同质性或减少异质性图中的同质性来转换测试图结构，可以显著提高预训练GNN在节点分类任务上的鲁棒性和性能，无需模型训练或更新。

### 结论

在九个基准数据集上的实验表明，GrapHoST在各种测试时数据质量问题下始终实现了最先进的性能，最高提升达10.92%，代码已公开。

### 翻译

同质性，即同一类别节点倾向于连接的特性，是现实世界图的基本属性，支撑着引文网络和社会网络等领域中的结构和语义模式。现有方法通过设计同质性感知的GNN架构或图结构学习策略来利用同质性，但它们主要关注训练图的GNN学习。然而，在现实场景中，测试图常常面临数据质量问题和分布偏移，如社交网络中不同地区用户之间的领域偏移，以及在不同时间段收集的引文网络图中的时间演化偏移。这些因素显著降低了预训练模型的鲁棒性，导致测试时性能下降。通过实证观察和理论分析，我们揭示出通过转换测试图结构——在同质性图中增加同质性或在异质性图中减少同质性——可以显著提高预训练GNN在节点分类任务上的鲁棒性和性能，无需模型训练或更新。基于这些见解，我们提出了一种基于同质性的新颖测试时图结构转换方法，名为GrapHoST。具体来说，开发了一个同质性预测器来区分测试边，通过预测同质性得分的置信度实现自适应的测试时图结构转换。在九个基准数据集上针对多种测试时数据质量问题的广泛实验表明，GrapHoST始终实现了最先进的性能，最高提升达10.92%。我们的代码已在https://github.com/YanJiangJerry/GrapHoST发布。


### 论文摘要

Homophily, the tendency of nodes from the same class to connect, is a fundamental property of real-world graphs, underpinning structural and semantic patterns in domains such as citation networks and social networks. Existing methods exploit homophily through designing homophily-aware GNN architectures or graph structure learning strategies, yet they primarily focus on GNN learning with training graphs. However, in real-world scenarios, test graphs often suffer from data quality issues and distribution shifts, such as domain shifts across users from different regions in social networks and temporal evolution shifts in citation network graphs collected over varying time periods. These factors significantly compromise the pre-trained model's robustness, resulting in degraded test-time performance. With empirical observations and theoretical analysis, we reveal that transforming the test graph structure by increasing homophily in homophilic graphs or decreasing it in heterophilic graphs can significantly improve the robustness and performance of pre-trained GNNs on node classifications, without requiring model training or update. Motivated by these insights, a novel test-time graph structural transformation method grounded in homophily, named GrapHoST, is proposed. Specifically, a homophily predictor is developed to discriminate test edges, facilitating adaptive test-time graph structural transformation by the confidence of predicted homophily scores. Extensive experiments on nine benchmark datasets under a range of test-time data quality issues demonstrate that GrapHoST consistently achieves state-of-the-art performance, with improvements of up to 10.92%. Our code has been released at https://github.com/YanJiangJerry/GrapHoST.

---

## 57. Dynamic Graph Neural Network for Data-Driven Physiologically Based Pharmacokinetic Modeling

**论文链接:** [http://arxiv.org/abs/2510.22096v1](http://arxiv.org/abs/2510.22096v1)

**作者:** Su Liu, Xin Hu, Shurong Wen, Jiaqi Liu, Jiexi Xu, Lanruo Wang

**发布时间:** 2025-10-25

### GPT解析

### 总结

本研究探索了使用深度学习替代传统生理药代动力学建模方法，提出了一种动态图神经网络(Dynamic GNN)来模拟器官间的相互作用，实现了更高的预测性能。

### 背景

生理药代动力学(PBPK)建模在药物开发中通过预测器官间药物浓度动态发挥关键作用。传统方法依赖常微分方程和强简化假设，限制了其对非线性生理相互作用的适应性。

### 目的

探索使用深度学习进行PBPK预测的数据驱动替代方法，以提高预测准确性和适应性。

### 方法

实现两种基线架构：多层感知器(MLP)和长短期记忆(LSTM)网络，分别用于捕捉分子和时间依赖性。提出动态图神经网络(Dynamic GNN)将生理连接建模为器官间的递归消息传递过程。

### 主要发现

动态GNN在所有模型中表现最佳，预测性能最高，R平方值为0.9342，均方根误差为0.0159，平均绝对误差为0.0116。相比之下，MLP基线获得R平方值0.8705，LSTM获得0.8059。明确建模器官相互作用的时空依赖性可实现更准确和可推广的药物浓度预测。

### 结论

动态GNN为传统PBPK公式提供了可扩展的、无方程的替代方案，在临床前和临床研究中，数据驱动的药代动力学建模展现出巨大潜力。

### 翻译

生理药代动力学(PBPK)建模通过预测器官间药物浓度动态在药物开发中发挥关键作用。传统方法依赖常微分方程和强简化假设，限制了其对非线性生理相互作用的适应性。本研究探索了使用深度学习进行PBPK预测的数据驱动替代方法。实现了两种基线架构：多层感知器(MLP)和长短期记忆(LSTM)网络，分别用于捕捉分子和时间依赖性。为了整合器官间相互作用，我们提出了一种动态图神经网络(Dynamic GNN)，将生理连接建模为器官间的递归消息传递过程。实验结果表明，所提出的动态GNN在所有模型中实现了最高的预测性能，R平方值为0.9342，均方根误差为0.0159，平均绝对误差为0.0116。相比之下，MLP基线获得R平方值0.8705，LSTM获得0.8059。这些结果表明，明确建模器官相互作用的时空依赖性可实现更准确和可推广的药物浓度预测。动态GNN为传统PBPK公式提供了可扩展的、无方程的替代方案，并在临床前和临床研究中展现出数据驱动药代动力学建模的巨大潜力。


### 论文摘要

Physiologically Based Pharmacokinetic (PBPK) modeling plays a critical role in drug development by predicting drug concentration dynamics across organs. Traditional approaches rely on ordinary differential equations with strong simplifying assumptions, which limit their adaptability to nonlinear physiological interactions. In this study, we explore data-driven alternatives for PBPK prediction using deep learning. Two baseline architectures - a multilayer perceptron (MLP) and a long short-term memory (LSTM) network - are implemented to capture molecular and temporal dependencies, respectively. To incorporate inter-organ interactions, we propose a Dynamic Graph Neural Network (Dynamic GNN) that models physiological connections as recurrent message-passing processes between organs. Experimental results demonstrate that the proposed Dynamic GNN achieves the highest predictive performance among all models, with an R^2 of 0.9342, an RMSE of 0.0159, and an MAE of 0.0116. In comparison, the MLP baseline obtains an R^2 of 0.8705 and the LSTM achieves 0.8059. These results highlight that explicitly modeling the spatial and temporal dependencies of organ interactions enables more accurate and generalizable drug concentration prediction. The Dynamic GNN provides a scalable, equation-free alternative to traditional PBPK formulations and demonstrates strong potential for data-driven pharmacokinetic modeling in preclinical and clinical research.

---

## 58. Hierarchical Graph Networks for Accurate Weather Forecasting via Lightweight Training

**论文链接:** [http://arxiv.org/abs/2510.22094v1](http://arxiv.org/abs/2510.22094v1)

**作者:** Thomas Bailie, S. Karthik Mukkavilli, Varvara Vetrova, Yun Sing Koh

**发布时间:** 2025-10-25

### GPT解析

### 总结

研究团队开发了HiFlowCast和HiAntFlow两种层级图神经网络模型，通过创新机制提高气候事件预测准确性，同时降低训练成本和环境影响。

### 背景

气候事件由复杂的全球尺度驱动因素导致的多元动态过程产生，对食物、能源和基础设施有深远影响。然而，由于物理过程跨越不同的时空尺度，固定分辨率方法无法捕捉这些过程，导致准确天气预测仍然困难。

### 目的

开发一种能够捕捉多尺度物理过程的气候预测方法，提高预测准确性，特别是对于极端事件。

### 方法

提出HiFlowCast和其集合变体HiAntFlow，这是一种层级图神经网络，将物理学嵌入多尺度预测框架。创新点包括：1)潜在记忆保留机制，在向下遍历过程中保持全局趋势；2)从潜在到物理的分支，整合不同尺度上的偏微分方程解场。

### 主要发现

在13天提前期的预测中，模型将误差减少了5%以上；在第一和第九十九百分位极端情况下，误差减少了5-8%，提高了罕见事件的可靠性。利用预训练模型权重，模型在一个周期内收敛，显著降低了训练成本和碳足迹。

### 结论

提高预测效率对于应对机器学习规模增长带来的可持续性挑战和研究可及性限制至关重要，代码和模型权重见补充材料。

### 翻译

气候事件源于由全球尺度驱动的复杂多变量动态过程，深刻影响食物、能源和基础设施。然而，由于物理过程跨越多样的时空尺度展开，固定分辨率方法无法捕捉，准确的天气预测仍然难以实现。层级图神经网络提供多尺度表示，但非线性向下映射通常会抹去全局趋势，削弱物理学与预测的整合。我们引入HiFlowCast及其集合变体HiAntFlow，这些图神经网络将物理学嵌入多尺度预测框架。两个创新支撑了它们的设计：潜在记忆保留机制在向下遍历过程中保持全局趋势，以及从潜在到物理的分支整合不同尺度上的偏微分方程解场。我们的模型在13天提前期将误差减少5%以上，在第一和第九十九百分位极端情况下减少5-8%，提高了罕见事件的可靠性。利用预训练模型权重，它们在一个周期内收敛，降低了训练成本和碳足迹。这种效率至关重要，因为机器学习规模的不断增长挑战可持续性并限制研究可及性。代码和模型权重见补充材料。


### 论文摘要

Climate events arise from intricate, multivariate dynamics governed by global-scale drivers, profoundly impacting food, energy, and infrastructure. Yet, accurate weather prediction remains elusive due to physical processes unfolding across diverse spatio-temporal scales, which fixed-resolution methods cannot capture. Hierarchical Graph Neural Networks (HGNNs) offer a multiscale representation, but nonlinear downward mappings often erase global trends, weakening the integration of physics into forecasts. We introduce HiFlowCast and its ensemble variant HiAntFlow, HGNNs that embed physics within a multiscale prediction framework. Two innovations underpin their design: a Latent-Memory-Retention mechanism that preserves global trends during downward traversal, and a Latent-to-Physics branch that integrates PDE solution fields across diverse scales. Our Flow models cut errors by over 5% at 13-day lead times and by 5-8% under 1st and 99th quantile extremes, improving reliability for rare events. Leveraging pretrained model weights, they converge within a single epoch, reducing training cost and their carbon footprint. Such efficiency is vital as the growing scale of machine learning challenges sustainability and limits research accessibility. Code and model weights are in the supplementary materials.

---

## 59. Pruning and Quantization Impact on Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.22058v1](http://arxiv.org/abs/2510.22058v1)

**作者:** Khatoon Khedri, Reza Rawassizadeh, Qifu Wen, Mehdi Hosseinzadeh

**发布时间:** 2025-10-24

### GPT解析

### 总结

本研究探讨了神经网络压缩方法(剪枝和量化)对图神经网络(GNNs)的影响，发现非结构化细粒度和全局剪枝可显著减小模型大小(50%)同时保持或提高精度，而不同量化方法在不同数据集上有不同影响。

### 背景

图神经网络(GNNs)在图结构数据学习上具有高准确性，但面临高计算和资源成本问题。

### 目的

研究神经网络压缩方法(剪枝和量化)如何减小GNN模型大小同时保持合理准确性。

### 方法

在三个图数据集(Cora, Proteins, BBBBP)上评估三种剪枝方法和三种量化方法对三种GNN任务(图分类、节点分类和链接预测)的影响。

### 主要发现

非结构化细粒度和全局剪枝可显著减小模型大小(50%)，同时在微调后保持甚至提高精度；不同量化方法对GNN的准确性、推理时间和模型大小在不同数据集上有不同影响。

### 结论

神经网络压缩技术(特别是剪枝)可以有效减少GNN模型大小而不牺牲性能。

### 翻译

图神经网络(GNNs)在从图结构数据学习方面以高精度著称，但它们面临高计算和资源成本的问题。神经网络压缩方法用于减小模型大小同时保持合理准确性。两种常见的神经网络压缩技术包括剪枝和量化。在这项研究中，我们实证检验了三种剪枝方法和三种量化方法对不同GNN模型的影响，包括图分类任务、节点分类任务和链接预测。我们在三个图数据集上进行了所有实验，包括Cora、Proteins和BBBP。我们的研究结果表明，非结构化细粒度和全局剪枝可以显著减小模型大小(50%)，同时在微调剪枝后的模型后保持甚至提高精度。对GNN上不同量化方法的评估显示，在不同数据集上，这些方法对准确性、推理时间和模型大小有不同影响。


### 论文摘要

Graph neural networks (GNNs) are known to operate with high accuracy on learning from graph-structured data, but they suffer from high computational and resource costs. Neural network compression methods are used to reduce the model size while maintaining reasonable accuracy. Two of the common neural network compression techniques include pruning and quantization. In this research, we empirically examine the effects of three pruning methods and three quantization methods on different GNN models, including graph classification tasks, node classification tasks, and link prediction. We conducted all experiments on three graph datasets, including Cora, Proteins, and BBBP. Our findings demonstrate that unstructured fine-grained and global pruning can significantly reduce the model's size(50\%) while maintaining or even improving precision after fine-tuning the pruned model. The evaluation of different quantization methods on GNN shows diverse impacts on accuracy, inference time, and model size across different datasets.

---

## 60. PF$Δ$: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations

**论文链接:** [http://arxiv.org/abs/2510.22048v1](http://arxiv.org/abs/2510.22048v1)

**作者:** Ana K. Rivera, Anvita Bhagavathula, Alvaro Carbonero, Priya Donti

**发布时间:** 2025-10-24

**备注:** 31 pages, 14 figures. Accepted at NeurIPS 2025

### GPT解析

### 总结

本文介绍了PFΔ，一个用于潮流计算的基准数据集，旨在解决电力系统操作中的计算瓶颈和不确定性挑战。

### 背景

潮流计算是实时电网操作的基础，但存在计算瓶颈问题。可再生能源整合和气候引起的极端天气增加了电力系统操作的不确定性，需要能够准确高效模拟各种场景的工具。

### 目的

引入一个能够捕捉负荷、发电和拓扑多样变化的潮流计算基准数据集PFΔ，以评估现有方法并确定未来研究方向。

### 方法

构建包含859,800个已解决潮流计算实例的数据集，涵盖六种不同总线系统规模，包含三种应急场景（N、N-1和N-2），以及接近稳态电压稳定性极限的案例。

### 主要发现

评估了传统求解器和基于GNN的方法，突出了现有方法面临的挑战领域，并确定了未来研究的开放性问题。

### 结论

PFΔ数据集和相关代码已公开发布，为电力系统潮流计算研究提供了新资源。

### 翻译

潮流计算是实时电网操作的基础，贯穿于工作流程中，如contingency analysis（重复PF评估评估停电情况下的电网安全）和拓扑优化（涉及基于PF的组合式大动作空间搜索）。在操作时间尺度上运行这些计算或在大评估空间中运行仍然是主要的计算瓶颈。此外，可再生能源整合和气候引起的极端天气导致的电力系统操作中不断增加的不确定性，也需要能够准确高效地模拟各种场景和操作条件的工具。机器学习方法相比传统求解器提供了潜在的速度提升，但它们在捕捉现实世界变异性的基准上尚未得到系统性评估。本文引入了PFΔ，一个潮流计算的基准数据集，捕捉了负荷、发电和拓扑的多样变化。PFΔ包含859,800个已解决的潮流计算实例，涵盖六种不同总线系统规模，捕获三种类型的应急场景（N、N-1和N-2），并包括接近稳态电压稳定性极限的接近不可行案例。我们评估了传统求解器和基于GNN的方法，突出了现有方法遇到困难的领域，并确定了未来研究的开放性问题。我们的数据集可在https://huggingface.co/datasets/pfdelta/pfdelta/tree/main获取，我们的代码包含数据生成脚本和模型实现，位于https://github.com/MOSSLab-MIT/pfdelta。


### 论文摘要

Power flow (PF) calculations are the backbone of real-time grid operations, across workflows such as contingency analysis (where repeated PF evaluations assess grid security under outages) and topology optimization (which involves PF-based searches over combinatorially large action spaces). Running these calculations at operational timescales or across large evaluation spaces remains a major computational bottleneck. Additionally, growing uncertainty in power system operations from the integration of renewables and climate-induced extreme weather also calls for tools that can accurately and efficiently simulate a wide range of scenarios and operating conditions. Machine learning methods offer a potential speedup over traditional solvers, but their performance has not been systematically assessed on benchmarks that capture real-world variability. This paper introduces PF$\Delta$, a benchmark dataset for power flow that captures diverse variations in load, generation, and topology. PF$\Delta$ contains 859,800 solved power flow instances spanning six different bus system sizes, capturing three types of contingency scenarios (N , N -1, and N -2), and including close-to-infeasible cases near steady-state voltage stability limits. We evaluate traditional solvers and GNN-based methods, highlighting key areas where existing approaches struggle, and identifying open problems for future research. Our dataset is available at https://huggingface.co/datasets/pfdelta/pfdelta/tree/main and our code with data generation scripts and model implementations is at https://github.com/MOSSLab-MIT/pfdelta.

---

## 61. A Hybrid GNN-LSE Method for Fast, Robust, and Physically-Consistent AC Power Flow

**论文链接:** [http://arxiv.org/abs/2510.22020v1](http://arxiv.org/abs/2510.22020v1)

**作者:** Mohamed Shamseldein

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文提出了一种结合物理信息图神经网络(GNN)和线性状态估计(LSE)的两阶段混合方法，用于解决传统交流潮流求解器在大规模电力系统中的计算和收敛挑战。

### 背景

传统的交流潮流(ACPF)求解器如牛顿-拉夫森法(NR)在现代大规模电力系统中面临显著的计算和收敛挑战。

### 目的

开发一种快速且物理一致的电力系统求解方法，适用于实时操作和分析。

### 方法

结合物理信息图神经网络与迭代线性状态估计：使用物理信息损失函数训练GNN预测高质量初始系统状态，然后通过LSE细化步骤解决线性方程以强制执行物理定律，绕过传统求解器的非线性和收敛问题。

### 主要发现

该方法在IEEE 33节点、69节点和118节点系统上得到验证；GNN变体比牛顿-拉夫森法快高达8400倍；LSE细化能快速获得物理一致解；重载压力测试和N-1 contingencies证明了方法的可靠性和泛化能力。

### 结论

该框架成功连接了快速数据驱动模型与电力系统物理约束，为实时电力系统操作和分析提供了实用工具。

### 翻译

传统的交流潮流(ACPF)求解器如牛顿-拉夫森法(NR)在现代大规模电力系统中面临显著的计算和收敛挑战。本文提出了一种新颖的两阶段混合方法，结合物理信息图神经网络(GNN)和稳健的迭代线性状态估计(LSE)细化步骤，以产生快速且物理一致的解。使用具有高效动态加权方案的物理信息损失函数训练的GNN可快速预测高质量的初始系统状态。然后使用受状态估计技术启发的迭代直接线性求解器进行细化。LSE细化步骤解决一系列线性方程以强制执行物理定律，有效绕过传统求解器的非线性和收敛问题。所提出的GNN-LSE框架在从小的辐射状配电网(IEEE 33节点、69节点)到大型网状输电系统(IEEE 118节点)的各种系统上得到了全面验证。结果表明，我们的GNN变体比NR快高达8400倍。LSE细化提供了一条快速获得物理一致解的途径，而重载压力测试(标称值的120%-150%)和N-1 contingencies展示了该方法的可靠性和泛化能力。这项工作提出了一个强大而灵活的框架，用于连接快速的数据驱动模型与电力系统物理的严格约束，为实时操作和分析提供了实用工具。


### 论文摘要

Conventional AC Power Flow (ACPF) solvers like Newton-Raphson (NR) face significant computational and convergence challenges in modern, large-scale power systems. This paper proposes a novel, two-stage hybrid method that integrates a Physics-Informed Graph Neural Network (GNN) with a robust, iterative Linear State Estimation (LSE) refinement step to produce fast and physically-consistent solutions. The GNN, trained with a physics-informed loss function featuring an efficient dynamic weighting scheme, rapidly predicts a high-quality initial system state. This prediction is then refined using an iterative, direct linear solver inspired by state estimation techniques. This LSE refinement step solves a series of linear equations to enforce physical laws, effectively bypassing the non-linearities and convergence issues of traditional solvers. The proposed GNN-LSE framework is comprehensively validated on systems ranging from small radial distribution networks (IEEE 33-bus, 69-bus) to a large, meshed transmission system (IEEE 118-bus). Results show that our GNN variants are up to $8.4 \times 10^3$ times faster than NR. The LSE refinement provides a fast route to a physically-consistent solution, while heavy-loading stress tests (120%-150% of nominal) and N-1 contingencies demonstrate the method's reliability and generalization. This work presents a powerful and flexible framework for bridging fast, data-driven models with the rigorous constraints of power system physics, offering a practical tool for real-time operations and analysis.

---

## 62. A Multimodal Human Protein Embeddings Database: DeepDrug Protein Embeddings Bank (DPEB)

**论文链接:** [http://arxiv.org/abs/2510.22008v1](http://arxiv.org/abs/2510.22008v1)

**作者:** Md Saiful Islam Sajol, Magesh Rajasekaran, Hayden Gemeinhardt, Adam Bess, Chris Alvin, Supratik Mukhopadhyay

**发布时间:** 2025-10-24

### GPT解析

### 总结

DPEB是一个整合了四种蛋白质嵌入类型的数据集，用于提高蛋白质-蛋白质相互作用(PPI)预测的准确性，并在多种蛋白质分类任务中表现出色。

### 背景

计算预测蛋白质-蛋白质相互作用(PPI)具有挑战性，主要原因是缺乏整合的多模态蛋白质表示方法。

### 目的

创建一个整合多种蛋白质嵌入类型的数据集，填补AlphaFold2内部神经网络嵌入不可用的空白，为计算建模提供支持。

### 方法

DPEB是一个包含22,043个人类蛋白质的精选集合，整合了四种嵌入类型：结构嵌入(AlphaFold2)、基于transformer的序列嵌入(BioEmbeddings)、上下文氨基酸模式(ESM-2)和基于序列的n-gram统计(ProtVec)。

### 主要发现

GraphSAGE与BioEmbedding结合实现了最高的PPI预测性能(87.37% AUROC, 79.16%准确率)；该框架在酶分类任务上达到77.42%的准确率；在蛋白质家族分类任务上达到86.04%的准确率。

### 结论

DPEB支持多种图神经网络方法进行PPI预测，可应用于系统生物学、药物靶点识别、通路分析和疾病机制研究。

### 翻译

计算预测蛋白质-蛋白质相互作用(PPI)具有挑战性，由于缺乏整合的多模态蛋白质表示。DPEB是一个包含22,043个人类蛋白质的精选集合，整合了四种嵌入类型：结构(AlphaFold2)、基于transformer的序列(BioEmbeddings)、上下文氨基酸模式(ESM-2: Evolutionary Scale Modeling)和基于序列的n-gram统计(ProtVec)。AlphaFold2蛋白质结构可通过公共数据库(如AlphaFold2蛋白质结构数据库)获取，但内部神经网络嵌入不可用。DPEB通过提供AlphaFold2衍生的嵌入用于计算建模来填补这一空白。我们的基准评估显示，GraphSAGE与BioEmbedding结合实现了最高的PPI预测性能(87.37% AUROC, 79.16%准确率)。该框架在酶分类上实现了77.42%的准确率，在蛋白质家族分类上实现了86.04%的准确率。DPEB支持多种图神经网络方法进行PPI预测，能够在系统生物学、药物靶点识别、通路分析和疾病机制研究中应用。


### 论文摘要

Computationally predicting protein-protein interactions (PPIs) is challenging due to the lack of integrated, multimodal protein representations. DPEB is a curated collection of 22,043 human proteins that integrates four embedding types: structural (AlphaFold2), transformer-based sequence (BioEmbeddings), contextual amino acid patterns (ESM-2: Evolutionary Scale Modeling), and sequence-based n-gram statistics (ProtVec]). AlphaFold2 protein structures are available through public databases (e.g., AlphaFold2 Protein Structure Database), but the internal neural network embeddings are not. DPEB addresses this gap by providing AlphaFold2-derived embeddings for computational modeling. Our benchmark evaluations show GraphSAGE with BioEmbedding achieved the highest PPI prediction performance (87.37% AUROC, 79.16% accuracy). The framework also achieved 77.42% accuracy for enzyme classification and 86.04% accuracy for protein family classification. DPEB supports multiple graph neural network methods for PPI prediction, enabling applications in systems biology, drug target identification, pathway analysis, and disease mechanism studies.

---

## 63. Deep Learning on Real-World Graphs

**论文链接:** [http://arxiv.org/abs/2510.21994v1](http://arxiv.org/abs/2510.21994v1)

**作者:** Emanuele Rossi

**发布时间:** 2025-10-24

**DOI:** 10.25560/112863

**备注:** The thesis was submitted for the degree of Doctor of Philosophy in  Computing at Imperial College London (February 2024), under the supervision  of Prof. Michael M. Bronstein. It includes work published at ICML, ICLR,  NeurIPS, and the Learning on Graphs Conference

### GPT解析

### 总结

该论文提出了一系列图神经网络模型，解决了GNNs在实际应用中的关键挑战，包括可扩展性、时间性、方向性、数据不完整性和结构不确定性等问题。

### 背景

图神经网络已成为学习图结构数据的核心工具，但在实际系统中的应用受到可扩展性、时间性、方向性、数据不完整性和结构不确定性等关键挑战的限制。

### 目的

解决GNNs在实际应用中的限制，使其能够应用于工业规模的图数据。

### 方法

作者提出了五个模型：SIGN用于可扩展图学习，TGN用于时间图，Dir-GNN用于有向和异质网络，Feature Propagation用于处理缺失节点特征，NuGget用于博弈论结构推断。

### 主要发现

这些模型共同弥合了学术基准和工业规模图之间的差距，使GNNs能够在社交系统和推荐系统等领域使用。

### 结论

通过这些创新模型，GNNs的实际应用限制得到了解决，使其能够在真实世界系统中有效应用。

### 翻译

图神经网络已成为学习图结构数据的核心工具，但它们在实际系统中的应用受到可扩展性、时间性、方向性、数据不完整性和结构不确定性等关键挑战的限制。本论文引入了一系列解决这些限制的模型：SIGN用于可扩展图学习，TGN用于时间图，Dir-GNN用于有向和异质网络，Feature Propagation (FP)用于学习缺失节点特征，NuGget用于博弈论结构推断。这些贡献共同弥合了学术基准和工业规模图之间的差距，使GNNs能够在社交和推荐系统等领域使用。


### 论文摘要

Graph Neural Networks (GNNs) have become a central tool for learning on graph-structured data, yet their applicability to real-world systems remains limited by key challenges such as scalability, temporality, directionality, data incompleteness, and structural uncertainty. This thesis introduces a series of models addressing these limitations: SIGN for scalable graph learning, TGN for temporal graphs, Dir-GNN for directed and heterophilic networks, Feature Propagation (FP) for learning with missing node features, and NuGget for game-theoretic structural inference. Together, these contributions bridge the gap between academic benchmarks and industrial-scale graphs, enabling the use of GNNs in domains such as social and recommender systems.

---

## 64. Leveraging Classical Algorithms for Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.21574v1](http://arxiv.org/abs/2510.21574v1)

**作者:** Jason Wu, Petar Veličković

**发布时间:** 2025-10-24

### GPT解析

### 总结

该研究探索了通过在经典算法上预训练图神经网络(GNNs)来提升其在分子属性预测任务上的性能。研究证明将经典算法的先验知识嵌入到GNNs中能够提供有用的归纳偏置，从而提升在复杂真实世界图数据上的表现。

### 背景

神经网络在处理非结构化数据方面表现出色，但通常无法分布外泛化；而经典算法虽然保证正确性但缺乏灵活性。

### 目的

探索通过在经典算法上预训练图神经网络(GNNs)来改善其在Open Graph Benchmark上的分子属性预测任务中的性能，包括ogbg-molhiv(HIV抑制)和ogbg-molclintox(临床毒性)任务。

### 方法

使用从CLRS算法推理基准中的24个经典算法训练的GNNs，来初始化并冻结第二GNN的选定层，用于分子预测任务。

### 主要发现

与随机初始化的基线相比，预训练模型取得了一致的胜利或平局。其中，基于Segments Intersect算法的预训练在ogbg-molhiv上取得了6%的绝对增益，基于Dijkstra的预训练在ogbg-molclintox上取得了3%的增益。

### 结论

将经典算法的先验知识嵌入到GNNs中可以提供有用的归纳偏置，提高在复杂、真实世界图数据上的性能。

### 翻译

神经网络在处理非结构化数据方面表现出色，但通常无法分布外泛化，而经典算法虽然保证正确性但缺乏灵活性。我们探索了通过在经典算法上预训练图神经网络(GNNs)来改善其在Open Graph Benchmark上的分子属性预测任务中的性能，包括ogbg-molhiv(HIV抑制)和ogbg-molclintox(临床毒性)任务。使用从CLRS算法推理基准中的24个经典算法训练的GNNs，来初始化并冻结第二GNN的选定层用于分子预测。与随机初始化的基线相比，预训练模型取得了一致的胜利或平局，其中基于Segments Intersect算法的预训练在ogbg-molhiv上取得了6%的绝对增益，基于Dijkstra的预训练在ogbg-molclintox上取得了3%的增益。这些结果表明将经典算法的先验知识嵌入到GNNs中可以提供有用的归纳偏置，提高在复杂、真实世界图数据上的性能。


### 论文摘要

Neural networks excel at processing unstructured data but often fail to generalise out-of-distribution, whereas classical algorithms guarantee correctness but lack flexibility. We explore whether pretraining Graph Neural Networks (GNNs) on classical algorithms can improve their performance on molecular property prediction tasks from the Open Graph Benchmark: ogbg-molhiv (HIV inhibition) and ogbg-molclintox (clinical toxicity). GNNs trained on 24 classical algorithms from the CLRS Algorithmic Reasoning Benchmark are used to initialise and freeze selected layers of a second GNN for molecular prediction. Compared to a randomly initialised baseline, the pretrained models achieve consistent wins or ties, with the Segments Intersect algorithm pretraining yielding a 6% absolute gain on ogbg-molhiv and Dijkstra pretraining achieving a 3% gain on ogbg-molclintox. These results demonstrate embedding classical algorithmic priors into GNNs provides useful inductive biases, boosting performance on complex, real-world graph data.

---

## 65. HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing

**论文链接:** [http://arxiv.org/abs/2510.21542v1](http://arxiv.org/abs/2510.21542v1)

**作者:** Johann Flemming Gloy, Simon Olsson

**发布时间:** 2025-10-24

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本文介绍了一种名为HollowFlow的流模型，利用非回溯图神经网络解决大规模系统中的计算瓶颈问题，实现了高达O(n²)的加速，使基于流的生成模型能够应用于更大规模的科学问题。

### 背景

流和扩散模型已成为科学应用的强大工具，特别适用于采样非归一化概率分布，如玻尔兹曼生成器(BGs)。然而，这些模型在实际部署时面临关键挑战：它们依赖于样本似然计算，而这种计算的系统规模n呈指数级增长，使得大规模问题难以处理。

### 目的

为了解决流模型在大规模系统中的计算效率问题，作者引入了HollowFlow，旨在显著提高似然评估速度，使BGs能够扩展到更大的系统。

### 方法

作者提出了HollowFlow，一种利用新型非回溯图神经网络(NoBGNN)的基于流的生成模型。通过强制块对角雅可比结构，HollowFlow的似然评估可以在n中用常数次反向传播完成。该框架具有普适性，任何等变GNN或基于注意力的架构都可以被适配为NoBGNN。

### 主要发现

作者通过在两个不同规模的系统上训练BGs验证了HollowFlow。对于这两个系统，采样和似然评估时间都显著减少，遵循了理论上的缩放规律。对于较大的系统，作者获得了100倍的加速，展示了基于HollowFlow的方法在高维科学问题上的潜力。

### 结论

HollowFlow为基于流的生成模型在大规模科学问题中的应用提供了有效解决方案，通过创新的图神经网络架构显著提高了计算效率，使得以前因计算限制而无法处理的高维问题现在变得可行。

### 翻译

流和扩散模型已成为科学应用的强大工具，特别适用于采样非归一化概率分布，如玻尔兹曼生成器(BGs)。部署这些模型的一个关键挑战是它们依赖于样本似然计算，而这种计算的系统规模n呈指数级增长，通常使得大规模问题变得不可行。为了解决这个问题，我们引入了HollowFlow，这是一种基于流的生成模型，利用了一种新型的非回溯图神经网络(NoBGNN)。通过强制块对角雅可比结构，HollowFlow的似然评估可以在n中用常数次反向传播完成，实现高达O(n²)的加速：这是将BGs扩展到更大系统的重要一步。重要的是，我们的框架具有普适性：任何等变GNN或基于注意力的架构都可以被适配为NoBGNN。我们通过在两个不同规模的系统上训练BGs来验证HollowFlow。对于这两个系统，采样和似然评估时间都显著减少，遵循了理论上的缩放规律。对于较大的系统，我们获得了100倍的加速，清楚地展示了基于HollowFlow的方法在高维科学问题上的潜力，这些问题以前因计算瓶颈而受到阻碍。


### 论文摘要

Flow and diffusion-based models have emerged as powerful tools for scientific applications, particularly for sampling non-normalized probability distributions, as exemplified by Boltzmann Generators (BGs). A critical challenge in deploying these models is their reliance on sample likelihood computations, which scale prohibitively with system size $n$, often rendering them infeasible for large-scale problems. To address this, we introduce $\textit{HollowFlow}$, a flow-based generative model leveraging a novel non-backtracking graph neural network (NoBGNN). By enforcing a block-diagonal Jacobian structure, HollowFlow likelihoods are evaluated with a constant number of backward passes in $n$, yielding speed-ups of up to $\mathcal{O}(n^2)$: a significant step towards scaling BGs to larger systems. Crucially, our framework generalizes: $\textbf{any equivariant GNN or attention-based architecture}$ can be adapted into a NoBGNN. We validate HollowFlow by training BGs on two different systems of increasing size. For both systems, the sampling and likelihood evaluation time decreases dramatically, following our theoretical scaling laws. For the larger system we obtain a $10^2\times$ speed-up, clearly illustrating the potential of HollowFlow-based approaches for high-dimensional scientific problems previously hindered by computational bottlenecks.

---

## 66. Estimating Treatment Effects in Networks using Domain Adversarial Training

**论文链接:** [http://arxiv.org/abs/2510.21457v1](http://arxiv.org/abs/2510.21457v1)

**作者:** Daan Caljon, Jente Van Belle, Wouter Verbeke

**发布时间:** 2025-10-24

### GPT解析

### 总结

这篇论文提出了HINet方法，通过结合图神经网络和领域对抗训练，解决了网络环境中估计异质治疗效应时面临的干扰、未知暴露映射和网络层面协变量偏移等问题。

### 背景

在网络环境中估计异质治疗效应受到干扰困扰，一个实例的结果可能受到他人治疗状态的影响。现有方法通常假设已知的暴露映射，这往往不现实。同质性与治疗分配机制的相互作用可能导致网络层面的协变量偏移，进而导致治疗效应估计不准确，这种现象尚未被明确研究。

### 目的

提出一种能够在未知暴露映射下估计治疗效应的方法，同时减轻网络层面协变量偏移的影响。

### 方法

提出了HINet，一种新颖的方法，结合了图神经网络和领域对抗训练。这种组合允许在未知暴露映射下估计治疗效应，同时减轻网络层面协变量偏移的影响。

### 主要发现

在合成和半合成网络数据集上的广泛实证评估证明了该方法的有效性。

### 结论

HINet方法成功解决了网络环境中估计异质治疗效应的挑战。

### 翻译

在网络环境中估计异质治疗效应因干扰而复杂化，这意味着一个实例的结果可能受到他人治疗状态的影响。现有的因果机器学习方法通常假设已知的暴露映射，该映射总结了给定实例的结果如何受他人治疗的影响，这是一种简化的假设，通常不切实际。此外，同质性——相似实例倾向于连接——与治疗分配机制之间的相互作用可能引发网络层面的协变量偏移，可能导致不准确的治疗效应估计，这种现象尚未被明确研究。为了应对这些挑战，我们提出了HINet，一种将图神经网络与领域对抗训练相结合的新颖方法。这种组合允许在未知暴露映射的情况下估计治疗效应，同时减轻（网络层面）协变量偏移的影响。在合成和半合成网络数据集上的广泛实证评估证明了我们方法的有效性。


### 论文摘要

Estimating heterogeneous treatment effects in network settings is complicated by interference, meaning that the outcome of an instance can be influenced by the treatment status of others. Existing causal machine learning approaches usually assume a known exposure mapping that summarizes how the outcome of a given instance is influenced by others' treatment, a simplification that is often unrealistic. Furthermore, the interaction between homophily -- the tendency of similar instances to connect -- and the treatment assignment mechanism can induce a network-level covariate shift that may lead to inaccurate treatment effect estimates, a phenomenon that has not yet been explicitly studied. To address these challenges, we propose HINet, a novel method that integrates graph neural networks with domain adversarial training. This combination allows estimating treatment effects under unknown exposure mappings while mitigating the impact of (network-level) covariate shift. An extensive empirical evaluation on synthetic and semi-synthetic network datasets demonstrates the effectiveness of our approach.

---

