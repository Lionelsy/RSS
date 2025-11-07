# 今日论文推荐 - 2025-11-07

共 40 篇论文

---

## 1. RCMCL: A Unified Contrastive Learning Framework for Robust Multi-Modal (RGB-D, Skeleton, Point Cloud) Action Understanding

**论文链接:** [http://arxiv.org/abs/2511.04351v1](http://arxiv.org/abs/2511.04351v1)

**作者:** Hasan Akgul, Mari Eplik, Javier Rojas, Akira Yamamoto, Rajesh Kumar, Maya Singh

**发布时间:** 2025-11-06

**备注:** 11 pages, 6 figures,

### GPT解析

### 总结

该研究提出了一种名为鲁棒跨模态对比学习(RCMCL)的自监督框架，用于解决多模态人类动作识别中传感器故障或噪声导致性能下降的问题。

### 背景

人类动作识别(HAR)使用多模态输入(RGB-D、骨骼、点云)可实现高精度，但通常依赖大型标记数据集，并在传感器故障或噪声情况下性能急剧下降。

### 目的

开发一种能学习模态不变表示且在模态丢失和损坏情况下保持可靠性的自监督框架。

### 方法

RCMCL联合优化三个目标：(1)跨模态对比目标对齐异构流，(2)模态内自蒸馏目标提高视图不变性并减少冗余，(3)退化模拟目标训练模型从掩码或损坏输入中恢复；推理时使用自适应模态门控(AMG)网络为各模态分配数据驱动的可靠性权重。

### 主要发现

在NTU RGB+D 120和UWA3D-II数据集上，RCMCL在标准设置下达到最先进准确率，在严重双模态丢失情况下仅显示11.5%性能下降，显著优于监督融合基线。

### 结论

自监督跨模态对齐，结合明确的退化建模和自适应融合，是实现可部署多模态HAR的关键。

### 翻译

人类动作识别(HAR)使用多模态输入(RGB-D、骨骼、点云)可以实现高精度，但通常依赖大型标记数据集，并且在传感器故障或噪声情况下性能会急剧下降。我们提出了鲁棒跨模态对比学习(RCMCL)，一种自监督框架，学习模态不变表示，并在模态丢失和损坏情况下保持可靠性。RCMCL联合优化了：(i)对齐异构流的跨模态对比目标，(ii)提高视图不变性并减少冗余的模态内自蒸馏目标，(iii)明确训练模型从掩码或损坏输入中恢复的退化模拟目标。在推理时，自适应模态门控(AMG)网络为每个模态分配数据驱动的可靠性权重以实现鲁棒融合。在NTU RGB+D 120(CS/CV)和UWA3D-II上，RCMCL在标准设置下达到最先进的准确率，并且表现出明显更好的鲁棒性：在严重的双模态丢失情况下，仅显示11.5%的性能下降，显著优于强大的监督融合基线。这些结果表明，自监督跨模态对齐，结合明确的退化建模和自适应融合，是可部署的多模态HAR的关键。


### 论文摘要

Human action recognition (HAR) with multi-modal inputs (RGB-D, skeleton, point cloud) can achieve high accuracy but typically relies on large labeled datasets and degrades sharply when sensors fail or are noisy. We present Robust Cross-Modal Contrastive Learning (RCMCL), a self-supervised framework that learns modality-invariant representations and remains reliable under modality dropout and corruption. RCMCL jointly optimizes (i) a cross-modal contrastive objective that aligns heterogeneous streams, (ii) an intra-modal self-distillation objective that improves view-invariance and reduces redundancy, and (iii) a degradation simulation objective that explicitly trains models to recover from masked or corrupted inputs. At inference, an Adaptive Modality Gating (AMG) network assigns data-driven reliability weights to each modality for robust fusion. On NTU RGB+D 120 (CS/CV) and UWA3D-II, RCMCL attains state-of-the-art accuracy in standard settings and exhibits markedly better robustness: under severe dual-modality dropout it shows only an 11.5% degradation, significantly outperforming strong supervised fusion baselines. These results indicate that self-supervised cross-modal alignment, coupled with explicit degradation modeling and adaptive fusion, is key to deployable multi-modal HAR.

---

## 2. Active Domain Adaptation for mmWave-based HAR via Renyi Entropy-based Uncertainty Estimation

**论文链接:** [http://arxiv.org/abs/2511.04219v1](http://arxiv.org/abs/2511.04219v1)

**作者:** Mingzhi Lin, Teng Huang, Han Ding, Cui Zhao, Fei Wang, Ge Wang, Wei Xi

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种名为mmADA的主动域适应框架，用于解决毫米波雷达人类活动识别中的域偏移问题，通过Renyi熵不确定性估计、对比学习和伪标记技术，以最少的标记数据实现高准确率，在各种跨域场景中表现出色。

### 背景

人类活动识别(HAR)使用毫米波雷达是传统基于传感器方法的非侵入式替代方案，但存在域偏移问题，即模型在新用户、新位置或新环境中性能下降。

### 目的

提出一个名为mmADA的主动域适应(ADA)框架，以最少的标记数据高效适应基于毫米波的HAR模型。

### 方法

mmADA通过引入基于Renyi熵的不确定性估计来识别和标记最具信息量的目标样本，从而增强适应性。此外，它利用对比学习和伪标记来使用未标记数据改进特征对齐。

### 主要发现

使用TI IWR1443BOOST雷达在多个用户、位置和环境中的评估显示，mmADA在各种跨域设置中实现了超过90%的准确率。与五个基线的比较确认了其优越的适应性能，而在未见过的用户、环境以及另外两个开源数据集上的进一步测试验证了其鲁棒性和泛化能力。

### 结论

mmADA框架能够有效解决毫米波雷达HAR中的域偏移问题，以最少的标记数据实现高准确率，并在各种跨域场景中表现出色。

### 翻译

人类活动识别(HAR)使用毫米波雷达为传统基于传感器的方法提供了非侵入式替代方案，但在新用户、新位置或新环境中存在域偏移问题，导致模型性能下降。为解决这一问题，我们提出了mmADA，这是一种主动域适应(ADA)框架，能够以最少的标记数据高效适应基于毫米波的HAR模型。mmADA通过引入基于Renyi熵的不确定性估计来识别和标记最具信息量的目标样本，从而增强适应性。此外，它还利用对比学习和伪标记来使用未标记数据改进特征对齐。在多个用户、位置和环境中使用TI IWR1443BOOST雷达进行的评估表明，mmADA在各种跨域设置中实现了超过90%的准确率。与五个基线的比较确认了其优越的适应性能，而在未见过的用户、环境以及另外两个开源数据集上的进一步测试验证了其鲁棒性和泛化能力。


### 论文摘要

Human Activity Recognition (HAR) using mmWave radar provides a non-invasive alternative to traditional sensor-based methods but suffers from domain shift, where model performance declines in new users, positions, or environments. To address this, we propose mmADA, an Active Domain Adaptation (ADA) framework that efficiently adapts mmWave-based HAR models with minimal labeled data. mmADA enhances adaptation by introducing Renyi Entropy-based uncertainty estimation to identify and label the most informative target samples. Additionally, it leverages contrastive learning and pseudo-labeling to refine feature alignment using unlabeled data. Evaluations with a TI IWR1443BOOST radar across multiple users, positions, and environments show that mmADA achieves over 90% accuracy in various cross-domain settings. Comparisons with five baselines confirm its superior adaptation performance, while further tests on unseen users, environments, and two additional open-source datasets validate its robustness and generalization.

---

## 3. DeNoise: Learning Robust Graph Representations for Unsupervised Graph-Level Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.04086v1](http://arxiv.org/abs/2511.04086v1)

**作者:** Qingfeng Chen, Haojin Zeng, Jingyi Jie, Shichao Zhang, Debo Cheng

**发布时间:** 2025-11-06

### GPT解析

### 总结

论文提出了DeNoise框架，解决了无监督图级别异常检测(UGAD)中训练数据被异常图污染的问题，通过对抗性训练、编码器锚点对齐和对比学习技术学习抗噪声的图表示，在多个数据集上显著优于现有方法。

### 背景

随着关键领域中图结构数据的快速增长，无监督图级别异常检测(UGAD)已成为重要任务。然而，大多数图神经网络方法假设训练集只包含正常图，这在实践中很少成立，即使少量异常图污染也会扭曲学习表示并降低性能。

### 目的

设计一个健壮的UGAD框架，专门处理被污染的训练数据，学习对噪声不敏感的图表示。

### 方法

DeNoise通过对抗性目标联合优化图级别编码器、属性解码器和结构解码器；引入编码器锚点对齐去噪机制，将正常图中高信息节点嵌入融合到所有图嵌入中；使用对比学习组件在潜在空间中压缩正常图嵌入并排斥异常嵌入。

### 主要发现

在八个真实数据集上的实验表明，DeNoise能够在不同噪声强度下学习可靠的图级别表示，显著优于最先进的UGAD基线方法。

### 结论

DeNoise是一个有效的框架，能够处理被污染的训练数据，通过创新的去噪机制提高了UGAD的性能，在实际应用中具有重要价值。

### 翻译

随着关键领域中图结构数据的快速增长，无监督图级别异常检测(UGAD)已成为一项关键任务。UGAD旨在识别偏离正常行为模式的整个图。然而，大多数图神经网络方法隐含假设训练集是干净的，只包含正常图，这在实践中很少成立。即使有少量异常图的污染也会扭曲学习到的表示并显著降低性能。为应对这一挑战，我们提出了DeNoise，一个专为被污染训练数据设计的健壮UGAD框架。它通过对抗性目标联合优化图级别编码器、属性解码器和结构解码器，学习抗噪声的嵌入。此外，DeNoise引入了编码器锚点对齐去噪机制，将正常图中高信息节点嵌入融合到所有图嵌入中，提高表示质量同时抑制异常干扰。然后，对比学习组件在潜在空间中压缩正常图嵌入并排斥异常嵌入。在八个真实数据集上的广泛实验表明，DeNoise能够在不同噪声强度下一致学习可靠的图级别表示，并显著优于最先进的UGAD基线方法。


### 论文摘要

With the rapid growth of graph-structured data in critical domains, unsupervised graph-level anomaly detection (UGAD) has become a pivotal task. UGAD seeks to identify entire graphs that deviate from normal behavioral patterns. However, most Graph Neural Network (GNN) approaches implicitly assume that the training set is clean, containing only normal graphs, which is rarely true in practice. Even modest contamination by anomalous graphs can distort learned representations and sharply degrade performance. To address this challenge, we propose DeNoise, a robust UGAD framework explicitly designed for contaminated training data. It jointly optimizes a graph-level encoder, an attribute decoder, and a structure decoder via an adversarial objective to learn noise-resistant embeddings. Further, DeNoise introduces an encoder anchor-alignment denoising mechanism that fuses high-information node embeddings from normal graphs into all graph embeddings, improving representation quality while suppressing anomaly interference. A contrastive learning component then compacts normal graph embeddings and repels anomalous ones in the latent space. Extensive experiments on eight real-world datasets demonstrate that DeNoise consistently learns reliable graph-level representations under varying noise intensities and significantly outperforms state-of-the-art UGAD baselines.

---

## 4. KAN-Enhanced Contrastive Learning Accelerating Crystal Structure Identification from XRD Patterns

**论文链接:** [http://arxiv.org/abs/2511.04055v1](http://arxiv.org/abs/2511.04055v1)

**作者:** Chenlei Xu, Tianhao Su, Jie Xiong, Yue Wu, Shuya Dong, Tian Jiang, Mengwei He, Shuai Chen, Tong-Yi Zhang

**发布时间:** 2025-11-06

### GPT解析

### 总结

XCCP是一种物理引导的对比学习框架，用于粉末X射线衍射分析，通过将衍射图与晶体结构在共享嵌入空间中对齐，实现高效结构检索和对称性识别，在结构检索和空间群识别任务中表现出色，并支持零样本迁移。

### 背景

准确确定晶体结构对材料科学至关重要，粉末X射线衍射是一种关键技术，但当前分析流程仍严重依赖专家知识和缓慢的迭代拟合，限制了在高通量和自主环境中的可扩展性。

### 目的

引入一种称为XCCP的物理引导对比学习框架，使粉末衍射图与候选晶体结构在共享嵌入空间中对齐，实现高效的结构检索和对称性识别。

### 方法

XRD编码器采用双专家设计，带有Kolmogorov-Arnold Network投影头，一个分支强调低角度反射反映长程有序性，另一个分支捕获由对称性形成的密集高角度峰，结合晶体图编码器进行对比预训练。

### 主要发现

XCCP在结构检索任务中达到0.89准确率，空间群识别达到0.93准确率，可推广到成分相似的多主元合金，并展示了对实验模式的零样本迁移能力。

### 结论

XCCP是一种稳健、可解释和可扩展的方法，为X射线衍射分析提供了新范式，促进高通量筛选、快速结构验证并集成到自主实验室中。

### 翻译

准确确定晶体结构对材料科学至关重要，支撑着成分-结构-性能关系的理解和新材料的发现。粉末X射线衍射是这一追求中的关键技术，因其多功能性和可靠性。然而，当前分析流程仍然严重依赖专家知识和缓慢的迭代拟合，限制了它们在高通量和自主环境中的可扩展性。在此，我们介绍了一种称为XCCP的物理引导对比学习框架。它使粉末衍射图与候选晶体结构在共享嵌入空间中对齐，以实现高效的结构检索和对称性识别。XRD编码器采用双专家设计，带有Kolmogorov-Arnold Network投影头，一个分支强调反映长程有序性的低角度反射，而另一个分支捕获由对称性塑造的密集高角度峰。与晶体图编码器相结合，对比预训练产生物理基础的表示。XCCP在各项任务中表现出色，结构检索达到0.89，空间群识别达到0.93的准确率。该框架进一步推广到成分相似的多主元合金，并展示了对实验模式的零样本迁移能力。这些结果确立了XCCP是一种稳健、可解释和可扩展的方法，为X射线衍射分析提供了新范式。XCCP促进高通量筛选、快速结构验证并集成到自主实验室中。


### 论文摘要

Accurate determination of crystal structures is central to materials science, underpinning the understanding of composition-structure-property relationships and the discovery of new materials. Powder X-ray diffraction is a key technique in this pursuit due to its versatility and reliability. However, current analysis pipelines still rely heavily on expert knowledge and slow iterative fitting, limiting their scalability in high-throughput and autonomous settings. Here, we introduce a physics-guided contrastive learning framework termed as XCCP. It aligns powder diffraction patterns with candidate crystal structures in a shared embedding space to enable efficient structure retrieval and symmetry recognition. The XRD encoder employs a dual-expert design with a Kolmogorov-Arnold Network projection head, one branch emphasizes low angle reflections reflecting long-range order, while the other captures dense high angle peaks shaped by symmetry. Coupled with a crystal graph encoder, contrastive pretraining yields physically grounded representations. XCCP demonstrates strong performance across tasks, with structure retrieval reaching 0.89 and space group identification attains 0.93 accuracy. The framework further generalizes to compositionally similar multi principal element alloys and demonstrates zero-shot transfer to experimental patterns. These results establish XCCP as a robust, interpretable, and scalable approach that offers a new paradigm for X-ray diffraction analysis. XCCP facilitates high-throughput screening, rapid structural validation and integration into autonomous laboratories.

---

## 5. Climbing the label tree: Hierarchy-preserving contrastive learning for medical imaging

**论文链接:** [http://arxiv.org/abs/2511.03771v1](http://arxiv.org/abs/2511.03771v1)

**作者:** Alif Elham Khan

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了一种层次保持对比框架，使医学图像标签的层次结构成为训练信号和评估目标，通过HWC和LAM两个插件目标改进自监督学习在医学图像表示学习中的性能。

### 背景

医学图像标签通常按分类法组织（如器官-组织-亚型），但标准自监督学习忽略了这种层次结构。

### 目的

开发一种能够尊重标签树结构的医学图像表示学习方法，提高性能和可解释性。

### 方法

提出层次加权对比（HWC）和层级感知边界（LAM）两个插件目标，HWC通过共享祖先缩放正/负对强度，LAM在不同层级上分离祖先组，适用于欧几里得和双曲嵌入。

### 主要发现

在多个基准测试中，包括乳腺组织病理学，所提出的方法比强SSL基线提高了表示质量，同时更好地尊重了分类法；HWC和LAM即使在没有曲率的情况下也有效，结合两者产生最符合分类法的表示。

### 结论

这种方法为学习尊重标签树的医学图像表示提供了一种简单、通用的方法，在层次丰富的领域中同时提高了性能和可解释性。

### 翻译

医学图像标签通常按分类法（例如器官-组织-亚型）组织，但标准自监督学习忽略了这种结构。我们提出了一种层次保持对比框架，使标签树成为一级训练信号和评估目标。我们的方法引入了两个插件目标：层次加权对比（HWC），通过共享祖先缩放正/负对强度以促进父级内部一致性，以及层级感知边界（LAM），一种在不同层级上分离祖先组的原型边界。该公式与几何无关，适用于欧几里得和双曲嵌入，无需架构更改。在包括乳腺组织病理学在内的多个基准测试中，所提出的目标始终比强大的SSL基线提高表示质量，同时更好地尊重分类法。我们使用适合层次忠实度的指标进行评估：HF1（层次F1）、H-Acc（树距离加权准确率）和父距离违规率。我们还报告了top-1准确率以保持完整性。消融实验表明，即使没有曲率，HWC和LAM也是有效的，而结合两者会产生最符合分类法的表示。总之，这些结果为学习尊重标签树的医学图像表示提供了一种简单、通用的方法，并在层次丰富的领域中同时提高了性能和可解释性。


### 论文摘要

Medical image labels are often organized by taxonomies (e.g., organ - tissue - subtype), yet standard self-supervised learning (SSL) ignores this structure. We present a hierarchy-preserving contrastive framework that makes the label tree a first-class training signal and an evaluation target. Our approach introduces two plug-in objectives: Hierarchy-Weighted Contrastive (HWC), which scales positive/negative pair strengths by shared ancestors to promote within-parent coherence, and Level-Aware Margin (LAM), a prototype margin that separates ancestor groups across levels. The formulation is geometry-agnostic and applies to Euclidean and hyperbolic embeddings without architectural changes. Across several benchmarks, including breast histopathology, the proposed objectives consistently improve representation quality over strong SSL baselines while better respecting the taxonomy. We evaluate with metrics tailored to hierarchy faithfulness: HF1 (hierarchical F1), H-Acc (tree-distance-weighted accuracy), and parent-distance violation rate. We also report top-1 accuracy for completeness. Ablations show that HWC and LAM are effective even without curvature, and combining them yields the most taxonomy-aligned representations. Taken together, these results provide a simple, general recipe for learning medical image representations that respect the label tree and advance both performance and interpretability in hierarchy-rich domains.

---

## 6. MacroNav: Multi-Task Context Representation Learning Enables Efficient Navigation in Unknown Environments

**论文链接:** [http://arxiv.org/abs/2511.04320v1](http://arxiv.org/abs/2511.04320v1)

**作者:** Kuankuan Sima, Longbin Tang, Haozhe Ma, Lin Zhao

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种名为MacroNav的基于学习的导航框架，解决了未知环境中自主导航的挑战，能够在保持计算效率的同时提高导航性能。

### 背景

未知环境中的自主导航需要在部分可观测性下进行紧凑而富有表现力的空间理解，以支持高层决策制定。现有方法难以在丰富的上下文表示和导航效率和效率之间取得平衡。

### 目的

开发一种能够平衡丰富上下文表示与导航效率的导航框架，提高未知环境中的自主导航性能。

### 方法

MacroNav框架包含两个关键组件：(1)通过多任务自监督学习训练的轻量级上下文编码器，用于捕获多尺度、以导航为中心的空间表示；(2)一种强化学习策略，将这些表示与基于图的推理无缝集成，用于高效的动作选择。

### 主要发现

上下文编码器能够高效且稳健地理解环境；在实际部署中，MacroNav在成功率（SR）和路径长度加权成功率（SPL）方面显著优于最先进的导航方法，同时保持较低的计算成本。

### 结论

MacroNav是一种有效的导航框架，能够在保持计算效率的同时提高导航性能，解决了现有方法在上下文表示与导航效率之间的平衡问题。

### 翻译

未知环境中的自主导航需要在部分可观测性下进行紧凑而富有表现力的空间理解，以支持高层决策制定。现有方法难以在丰富的上下文表示与导航效率之间取得平衡。我们提出了MacroNav，一种基于学习的导航框架，具有两个关键组件：(1)通过多任务自监督学习训练的轻量级上下文编码器，用于捕获多尺度、以导航为中心的空间表示；(2)一种强化学习策略，将这些表示与基于图的推理无缝集成，用于高效的动作选择。大量实验证明了上下文编码器的高效和稳健的环境理解能力。实际部署进一步验证了MacroNav的有效性，在成功率（SR）和路径长度加权成功率（SPL）方面比最先进的导航方法有显著提升，同时保持较低的计算成本。代码将在接受后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在未知环境中自主导航时如何平衡丰富的环境表示与导航效率之间的矛盾。这个问题非常重要，因为自主导航是机器人的基本挑战，需要在部分可观察条件下实时决策并找到到达目标的路径，而现有方法要么计算成本高，要么在复杂环境中表现不佳，限制了机器人在救援、自动驾驶和家庭服务等实际应用中的效能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括传统方法在动态场景中的不足和基于学习的方法的泛化问题。他们借鉴了视觉变换器架构用于空间表示学习、自监督学习方法（但针对导航任务专门设计）、强化学习中的软演员-批评家算法以及图推理方法。作者的创新在于设计了三个互补的自监督任务（随机路径掩码、视野预测和掩码自编码）来同时捕捉全局结构、局部几何和遮挡鲁棒性，并通过分层交叉注意力将这些表示与导航策略融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多任务自监督学习学习针对导航优化的上下文表示，然后与强化学习策略结合实现高效导航。整体流程包括：1) 将占用地图分割成块序列，通过三个自监督任务训练上下文编码器；2) 在机器人当前位置周围采样候选航点构建局部拓扑图；3) 使用分层交叉注意力融合上下文表示和节点特征；4) 通过指针注意力模块生成航点选择概率；5) 使用软演员-批评家算法训练导航策略，平衡任务完成、轨迹效率和目标导向行为。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 多任务自监督学习框架，包含三个互补任务（SPM、FOV和MAE）分别针对全局结构、局部几何和遮挡鲁棒性；2) 上下文感知的RL导航策略，通过分层交叉注意力融合多尺度上下文表示与拓扑图推理；3) 轻量级但高效的环境理解能力。相比之前工作，不同之处在于：不依赖手工规则（与传统方法相比）、显式建模环境上下文（与端到端RL方法相比）、保留细粒度几何信息（与图结构方法相比）、以及专门针对导航任务设计（与通用视觉模型相比）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MacroNav通过多任务自监督学习学习针对导航优化的多尺度上下文表示，并结合强化学习实现了在未知环境中高效且鲁棒的自主导航，显著提升了成功率和路径效率。'}


### 论文摘要

Autonomous navigation in unknown environments requires compact yet expressive spatial understanding under partial observability to support high-level decision making. Existing approaches struggle to balance rich contextual representation with navigation efficiency. We present MacroNav, a learning-based navigation framework featuring two key components: (1) a lightweight context encoder trained via multi-task self-supervised learning to capture multi-scale, navigation-centric spatial representations; and (2) a reinforcement learning policy that seamlessly integrates these representations with graph-based reasoning for efficient action selection. Extensive experiments demonstrate the context encoder's efficient and robust environmental understanding. Real-world deployments further validate MacroNav's effectiveness, yielding significant gains over state-of-the-art navigation methods in both Success Rate (SR) and Success weighted by Path Length (SPL), while maintaining low computational cost. Code will be released upon acceptance.

---

## 7. CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation

**论文链接:** [http://arxiv.org/abs/2511.03992v1](http://arxiv.org/abs/2511.03992v1)

**作者:** Yuwen Tao, Kanglei Zhou, Xin Tan, Yuan Xie

**发布时间:** 2025-11-06

### GPT解析

### 总结

这篇论文提出了CaRF框架，解决了3D高斯分割中的跨视图一致性问题，通过直接在3D高斯空间中操作并引入相机感知的高斯场编码和训练中配对视图监督，实现了显著的性能提升。

### 背景

现有的3D高斯分割方法虽然实现了语言和3D几何之间的跨模态对齐，但由于依赖2D渲染的伪监督和特定于视图的特征学习，仍然面临跨视图一致性的挑战。

### 目的

开发一个完全可分的框架，直接在3D高斯空间中操作，实现多视图一致性，提高3D场景理解的可靠性。

### 方法

提出CaRF框架，包括高斯场相机编码（GFCE）将相机几何纳入高斯文本交互，以及训练中配对视图监督（ITPVS）对齐校准视图的每个高斯logits。

### 主要发现

在三个基准测试上，CaRF在mIoU上分别比最先进方法提高了16.8%（Ref LERF）、4.3%（LERF OVS）和2.0%（3D OVS）。

### 结论

CaRF框架促进了更可靠和视图一致的3D场景理解，对具身AI、AR/VR交互和自主感知有潜在益处。

### 翻译

Referring 3D Gaussian Splatting Segmentation (R3DGS)指的是3D高斯喷洒分割，Camera Aware Referring Field (CaRF)是相机感知的参考场，Gaussian Field Camera Encoding (GFCE)是高斯场相机编码，In Training Paired View Supervision (ITPVS)是训练中配对视图监督。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决Referring 3D Gaussian Splatting Segmentation (R3DGS)中的多视角一致性问题。现有方法在不同视角下会产生不一致的分割结果，这影响了3D场景理解的准确性和可靠性。这个问题在现实中很重要，因为一致的3D场景理解对于具身AI、AR/VR交互和自主感知等应用至关重要，而不一致的分割会导致空间推理错误和用户体验下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有R3DGS方法的局限性，特别是它们依赖单视角伪监督和视图特定特征学习的问题。他们借鉴了3D高斯溅射(3DGS)的基本框架和R3DGS的基本概念，但认识到需要在3D高斯空间中直接操作。作者的创新思路是引入相机几何信息和多视角监督机制，设计了Gaussian Field Camera Encoding (GFCE)将相机参数整合到高斯-文本交互中，以及In-Training Paired-View Supervision (ITPVS)在训练过程中对齐校准视角下的高斯特征，从而实现多视角一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'CaRF的核心思想是通过直接在3D高斯空间中操作并引入相机几何信息来实现多视角一致性。具体包括：1)相机感知的referring场，将相机参数整合到高斯特征空间中以捕获视角依赖线索；2)训练过程中的配对视角监督，将选定高斯投影到两个校准视角中。整体流程包括：预处理阶段训练3DGS模型并生成伪真实掩码；训练阶段采样视角对，执行跨模态交互，生成相机特征，计算视角感知特征，渲染referring掩码，计算双视角损失和对比损失，更新参数；推理阶段使用训练好的模型渲染任意视角的referring掩码。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Gaussian Field Camera Encoding (GFCE)，将相机参数嵌入高斯特征空间明确建模视角依赖变化；2)In-Training Paired-View Supervision (ITPVS)，在训练过程中对齐校准视角下的高斯logit；3)完全可微框架直接在3D高斯空间操作。相比之前的工作，CaRF与ReferSplat不同在于引入多视角监督和显式几何条件；与基于2D特征/掩码的方法不同在于直接利用3D几何信息；与传统多视图一致性方法不同在于避免了非可微分组件并更好地利用了3D几何。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CaRF通过引入相机感知的高斯场编码和训练过程中的配对视角监督，显著提升了3D高斯溅射分割中的多视角一致性，实现了更可靠和几何一致的3D场景理解。'}


### 论文摘要

Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.

---

## 8. Simple 3D Pose Features Support Human and Machine Social Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.03988v1](http://arxiv.org/abs/2511.03988v1)

**作者:** Wenshuo Qin, Leyla Isik

**发布时间:** 2025-11-06

**备注:** 28 pages, 6 figures

### GPT解析

### 总结

研究通过比较人类和AI系统在社交互动识别上的能力，发现人类依赖3D视觉空间姿态信息，而大多数AI模型缺乏这种明确的表示。研究推导出简化的3D社交姿态特征，不仅能匹配人类判断，还能提升现有AI模型性能。

### 背景

人类能够快速轻松地从视觉输入中提取关于他人社交互动的各种信息，从基本的视觉空间线索到更高级的信息。然而，支持这些能力的计算机制仍不清楚，社交互动识别甚至对最先进的AI视觉系统来说仍然是一个挑战。

### 目的

验证人类是否依赖3D视觉空间姿态信息来进行社交互动判断，而这些信息在大多数AI视觉模型中是缺失的。

### 方法

结合最先进的姿态和深度估计算法，从描绘日常人类行为的短视频中提取人的3D关节位置，并将其预测人类社交互动判断的能力与当前AI视觉模型进行比较。此外，推导出一组紧凑的3D社交姿态特征，仅描述视频中面部的3D位置和方向。

### 主要发现

1) 3D关节位置的表现优于大多数当前AI视觉模型；2) 关键社交信息在明确的身体位置中可用，而不是在大多数视觉模型的特征中；3) 最小化的3D社交姿态特征与完整3D关节集的预测强度相匹配；4) 当与现有AI视觉模型的嵌入结合时，这些特征显著提高了模型性能；5) 每个现成AI视觉模型中3D社交姿态特征的表示程度预测了模型匹配人类社交判断的能力。

### 结论

研究结果提供了强有力的证据，表明人类社交场景理解依赖于明确的3D姿态表示，并且可以通过简单的、结构化的视觉空间原语来支持。

### 翻译

人类能够快速轻松地从视觉输入中提取关于他人社交互动的各种信息，从基本的视觉空间线索（如两个人是否面对面）到更高级的信息。然而，支持这些能力的计算机制仍不清楚，社交互动识别甚至对最先进的AI视觉系统来说仍然是一个挑战。在这里，我们假设人类依赖3D视觉空间姿态信息来进行社交互动判断，而这些信息在大多数AI视觉模型中是缺失的。为了验证这一点，我们结合了最先进的姿态和深度估计算法，从描绘日常人类行为的短视频中提取人的3D关节位置，并将其预测人类社交互动判断的能力与当前AI视觉模型进行比较。令人惊讶的是，3D关节位置的表现优于大多数当前AI视觉模型，这表明关键社交信息在明确的身体位置中可用，而不是在大多数视觉模型的特征中，包括甚至用于提取关节位置的姿态模型的逐层嵌入。为了揭示人类用于社交判断的关键姿态特征，我们推导出一组紧凑的3D社交姿态特征，仅描述视频中面部的3D位置和方向。我们发现这些最小描述符与完整3D关节集的预测强度相匹配，并且当与现成AI视觉模型的嵌入结合时，显著提高了这些模型的性能。此外，每个现成AI视觉模型中3D社交姿态特征的表示程度预测了模型匹配人类社交判断的能力。总之，我们的研究结果提供了强有力的证据，表明人类社交场景理解依赖于明确的3D姿态表示，并且可以通过简单的、结构化的视觉空间原语来支持。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决人类社交场景理解与AI视觉模型之间的差距问题。人类能快速从视觉输入中提取社交互动信息，而最先进的AI系统在这方面表现不佳。这个问题重要是因为社交理解是人类核心能力，也是AI系统在社交机器人、人机交互等领域应用的关键瓶颈，理解人类如何进行社交判断有助于改进AI系统。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于认知理论和现有AI模型的局限性思考，假设人类依赖3D视觉空间姿态信息进行社交判断，而大多数AI模型缺乏这种显式表示。他们借鉴了认知理论中关于简单视觉空间线索（如距离和朝向）是社交互动基础的观点，以及现有计算机视觉中的姿态估计和深度估计算法。设计上结合最先进技术提取3D关节位置，并创造性地简化为仅包含头部位置和朝向的紧凑特征集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是人类社交理解依赖于显式的3D视觉空间姿态信息，特别是简单的几何基元（如头部位置和朝向），而当前AI模型缺乏这种表示。整体流程包括：1)使用250个短视频数据集和人类评分；2)提取AI视觉模型特征和3D关节位置；3)从3D关节推导出紧凑的3D社交姿态特征；4)通过岭回归将特征映射到五个社交维度；5)比较不同特征集预测人类评分的能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)发现显式3D关节位置预测社交判断优于大多数AI模型；2)提出仅含头部位置和朝向的紧凑3D社交姿态特征；3)证明结合3D姿态特征可显著提升AI模型性能；4)发现能更好编码3D社交姿态的模型更符合人类判断。相比之前工作，本研究使用显式、可解释的3D表示而非从数据中学习的特征，采用简单几何基元而非复杂网络，首次证明3D信息对社交理解的关键作用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文证明了人类社交场景理解依赖于简单的3D视觉空间姿态信息，并展示了显式的几何表示可以显著提升AI系统对社交场景的理解能力。'}


### 论文摘要

Humans can quickly and effortlessly extract a variety of information about others' social interactions from visual input, ranging from visuospatial cues like whether two people are facing each other to higher-level information. Yet, the computations supporting these abilities remain poorly understood, and social interaction recognition continues to challenge even the most advanced AI vision systems. Here, we hypothesized that humans rely on 3D visuospatial pose information to make social interaction judgments, which is absent in most AI vision models. To test this, we combined state-of-the-art pose and depth estimation algorithms to extract 3D joint positions of people in short video clips depicting everyday human actions and compared their ability to predict human social interaction judgments with current AI vision models. Strikingly, 3D joint positions outperformed most current AI vision models, revealing that key social information is available in explicit body position but not in the learned features of most vision models, including even the layer-wise embeddings of the pose models used to extract joint positions. To uncover the critical pose features humans use to make social judgments, we derived a compact set of 3D social pose features describing only the 3D position and direction of faces in the videos. We found that these minimal descriptors matched the predictive strength of the full set of 3D joints and significantly improved the performance of off-the-shelf AI vision models when combined with their embeddings. Moreover, the degree to which 3D social pose features were represented in each off-the-shelf AI vision model predicted the model's ability to match human social judgments. Together, our findings provide strong evidence that human social scene understanding relies on explicit representations of 3D pose and can be supported by simple, structured visuospatial primitives.

---

## 9. SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.03325v2](http://arxiv.org/abs/2511.03325v2)

**作者:** Mauro Orazio Drago, Luca Carlini, Pelinsu Celebi Balyemez, Dennis Pierantozzi, Chiara Lena, Cesare Hassan, Danail Stoyanov, Elena De Momi, Sophia Bano, Mobarak I. Hoque

**发布时间:** 2025-11-05

### GPT解析

### 总结

本研究提出了SurgViVQA，一个专门用于手术领域的视频问答模型，能够处理时间连贯的事件而非孤立图像，从而增强手术过程中的理解能力。研究团队还创建了REAL-Colon-VQA数据集来评估模型性能。

### 背景

当前手术视频问答方法局限于静态图像特征，且可用数据集通常缺乏时间标注，忽略了准确解读手术程序所需的关键动态信息。

### 目的

开发能够从静态图像视觉推理扩展到动态手术场景的视频问答模型，捕捉时间线索如运动和工具-组织交互，从而更有效地解释动态手术程序背景。

### 方法

1. 提出SurgViVQA模型，使用掩码视频-文本编码器融合视频和问题特征；2. 捕捉运动和工具-组织交互等时间线索；3. 使用微调的大语言模型解码这些特征为连贯答案；4. 创建REAL-Colon-VQA结肠镜视频数据集，包含运动相关问题、诊断属性和模板外问题；5. 在REAL-Colon-VQA和公开的EndoVis18-VQA数据集上验证性能。

### 主要发现

1. SurgViVQA在关键词准确性上优于现有基于图像的VQA基准模型；2. 在REAL-Colon-VQA上比PitVQA提高11%；3. 在EndoVis18-VQA上比PitVQA提高9%；4. 问题扰动研究证实了模型对问题表述变化的泛化能力和鲁棒性。

### 结论

SurgViVQA和REAL-Colon-VQA数据集为手术视频问答中的时间感知理解提供了框架，使AI模型能够更有效地解释动态程序背景。

### 翻译

手术领域的视频问答旨在通过使AI模型能够对时间连贯的事件而非孤立帧进行推理，从而增强手术中的理解能力。当前方法仅限于静态图像特征，且可用数据集通常缺乏时间标注，忽略了准确解读手术程序所需的关键动态信息。我们提出了SurgViVQA，一种手术视频问答模型，将视觉推理从静态图像扩展到动态手术场景。它使用掩码视频-文本编码器融合视频和问题特征，捕捉运动和工具-组织交互等时间线索，然后由微调的大语言模型解码为连贯答案。为评估其性能，我们整理了REAL-Colon-VQA，一个包含运动相关问题、诊断属性以及重新表述或语义改变的问题表述的结肠镜视频数据集，以评估模型的鲁棒性。在REAL-Colon-VQA和公开的EndoVis18-VQA数据集上的实验验证表明，SurgViVQA在关键词准确性上优于现有的基于图像的VQA基准模型，在REAL-Colon-VQA上比PitVQA提高11%，在EndoVis18-VQA上提高9%。对问题的扰动研究进一步证实了模型对问题表述变化的泛化能力和鲁棒性。SurgViVQA和REAL-Colon-VQA数据集为手术视频问答中的时间感知理解提供了框架，使AI模型能够更有效地解释动态程序背景。代码和数据集可在https://github.com/madratak/SurgViVQA获取。


### 论文摘要

Video Question Answering (VideoQA) in the surgical domain aims to enhance intraoperative understanding by enabling AI models to reason over temporally coherent events rather than isolated frames. Current approaches are limited to static image features, and available datasets often lack temporal annotations, ignoring the dynamics critical for accurate procedural interpretation. We propose SurgViVQA, a surgical VideoQA model that extends visual reasoning from static images to dynamic surgical scenes. It uses a Masked Video--Text Encoder to fuse video and question features, capturing temporal cues such as motion and tool--tissue interactions, which a fine-tuned large language model (LLM) then decodes into coherent answers. To evaluate its performance, we curated REAL-Colon-VQA, a colonoscopic video dataset that includes motion-related questions and diagnostic attributes, as well as out-of-template questions with rephrased or semantically altered formulations to assess model robustness. Experimental validation on REAL-Colon-VQA and the public EndoVis18-VQA dataset shows that SurgViVQA outperforms existing image-based VQA benchmark models, particularly in keyword accuracy, improving over PitVQA by +11\% on REAL-Colon-VQA and +9\% on EndoVis18-VQA. A perturbation study on the questions further confirms improved generalizability and robustness to variations in question phrasing. SurgViVQA and the REAL-Colon-VQA dataset provide a framework for temporally-aware understanding in surgical VideoQA, enabling AI models to interpret dynamic procedural contexts more effectively. Code and dataset available at https://github.com/madratak/SurgViVQA.

---

## 10. UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.04595v1](http://arxiv.org/abs/2511.04595v1)

**作者:** Chen Shi, Shaoshuai Shi, Xiaoyang Lyu, Chunyang Liu, Kehua Sheng, Bo Zhang, Li Jiang

**发布时间:** 2025-11-06

### GPT解析

### 总结

论文提出了UniSplat，一种通用的前馈3D重建框架，通过统一的潜在时空融合学习鲁棒的动态场景重建，解决自动驾驶中稀疏、非重叠摄像头视图和复杂场景动态性的挑战。

### 背景

自动驾驶中的前馈3D重建技术发展迅速，但现有方法难以同时处理稀疏、非重叠的摄像头视图和复杂场景动态性这两个联合挑战。

### 目的

提出一个通用的前馈框架UniSplat，通过统一的潜在时空融合学习鲁棒的动态场景重建，解决现有方法面临的挑战。

### 方法

1. 构建3D潜在支架，利用预训练基础模型捕获几何和语义场景上下文；2. 引入高效融合机制，直接在3D支架内操作实现时空对齐；3. 设计双分支解码器，结合点锚定细化与基于体素的生成；4. 保持静态高斯函数的持久记忆实现流式场景完成。

### 主要发现

在真实世界数据集上，UniSplat在新视角合成方面达到最先进性能，即使对于原始摄像头覆盖范围之外的视点，也能提供鲁棒且高质量的渲染。

### 结论

UniSpat是一个有效的通用前馈框架，能够通过统一的潜在时空融合学习鲁棒的动态场景重建，在稀疏、非重叠的摄像头视图和复杂场景动态性条件下提供高质量的3D重建结果。

### 翻译

自动驾驶的前馈3D重建已经迅速发展，然而现有方法难以处理稀疏、非重叠摄像头视图和复杂场景动态性的联合挑战。我们提出了UniSplat，一个通用的前馈框架，通过统一的潜在时空融合学习鲁棒的动态场景重建。UniSplat构建一个3D潜在支架，这是一种结构化表示，通过利用预训练的基础模型捕获几何和语义场景上下文。为了有效跨空间视图和时间帧整合信息，我们引入了一个高效的融合机制，直接在3D支架内操作，实现一致的时空对齐。为确保完整和详细的重建，我们设计了一个双分支解码器，通过结合点锚定细化和基于体素的生成，从融合支架生成动态感知的高斯函数，并保持静态高斯函数的持久记忆，以实现当前摄像头覆盖范围之外的流式场景完成。在真实世界数据集上的大量实验表明，UniSplat在新视角合成方面取得了最先进的性能，同时即使对于原始摄像头覆盖范围之外的视点，也能提供鲁棒且高质量的渲染。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶场景中3D重建面临的挑战，特别是处理稀疏、非重叠的摄像头视图和复杂动态场景的问题。这个问题在现实中非常重要，因为高质量的3D场景重建是自动驾驶系统的核心能力，支持仿真、场景理解和长期规划，对于自动驾驶系统的感知、决策和控制至关重要。现有方法通常假设输入图像之间存在大量视点重叠，且需要针对每个场景进行优化，这限制了它们在实时驾驶场景中的适用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性，包括传统方法在2D空间中融合空间信息时受限于有限视图重叠，以及现有方法通常处理原始历史图像而非潜在表示。作者借鉴了几何基础模型来推断多视图图像的连贯3D结构，视觉基础模型提取语义信息，以及3D高斯溅射作为场景表示基础。作者的创新设计包括构建统一的3D潜在支架来融合多视图空间信息和多帧时间信息，在3D支架空间中直接执行空间融合，在支架表示内直接整合时间线索，以及设计双分支解码器生成动态感知的高斯原语。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个统一的3D潜在支架，融合多视图空间信息和多帧时间信息，在统一的以自我为中心的空间中编码显式3D几何，支持直接和高效的多视图和多帧时空融合，并使用双分支解码器生成动态感知的高斯原语。整体流程包括：1)3D支架构建：使用几何基础模型预测3D点图，通过尺度对齐解决模糊问题，组织成稀疏体素网格融合几何和语义特征；2)统一的时空支架融合：使用稀疏3D U-Net进行空间融合，通过变形前一帧支架到当前坐标系进行时间融合；3)动态感知的高斯生成：通过点解码器和体素解码器双分支生成高斯原语，并维护静态高斯存储器进行长期场景补全。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的3D潜在支架表示，融合多视图空间和多帧时间信息；2)基于支架的融合机制，支持统一的时空对齐和渐进式场景记忆集成；3)双分支高斯生成机制，结合点锚定细化和基于体素的生成；4)动态感知处理，通过动态分数识别静态内容并维护跨帧记忆。相比之前工作，不同之处在于UniSplat在统一的3D潜在支架中同时处理时空信息，而非分别处理；直接在3D空间进行融合而非2D图像空间；在支架表示内直接整合时间线索而非处理原始图像；能够更好地分离动态与静态场景内容。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniSplat通过统一的3D潜在支架表示，实现了自动驾驶场景中稀疏、非重叠摄像头视图下高效、高质量的动态场景重建与新颖视图合成，同时支持长期场景记忆和补全。'}


### 论文摘要

Feed-forward 3D reconstruction for autonomous driving has advanced rapidly, yet existing methods struggle with the joint challenges of sparse, non-overlapping camera views and complex scene dynamics. We present UniSplat, a general feed-forward framework that learns robust dynamic scene reconstruction through unified latent spatio-temporal fusion. UniSplat constructs a 3D latent scaffold, a structured representation that captures geometric and semantic scene context by leveraging pretrained foundation models. To effectively integrate information across spatial views and temporal frames, we introduce an efficient fusion mechanism that operates directly within the 3D scaffold, enabling consistent spatio-temporal alignment. To ensure complete and detailed reconstructions, we design a dual-branch decoder that generates dynamic-aware Gaussians from the fused scaffold by combining point-anchored refinement with voxel-based generation, and maintain a persistent memory of static Gaussians to enable streaming scene completion beyond current camera coverage. Extensive experiments on real-world datasets demonstrate that UniSplat achieves state-of-the-art performance in novel view synthesis, while providing robust and high-quality renderings even for viewpoints outside the original camera coverage.

---

## 11. Landslide Hazard Mapping with Geospatial Foundation Models: Geographical Generalizability, Data Scarcity, and Band Adaptability

**论文链接:** [http://arxiv.org/abs/2511.04474v1](http://arxiv.org/abs/2511.04474v1)

**作者:** Wenwen Li, Sizhe Wang, Hyunho Lee, Chenyan Lu, Sujit Roy, Rahul Ramachandran, Chia-Yu Hsu

**发布时间:** 2025-11-06

### GPT解析

### 总结

本研究提出了一种三轴分析框架，用于适应地理空间基础模型(GeoFMs)进行滑坡测绘，展示了其在不同条件下的优越性能和鲁棒性。

### 背景

滑坡对生命、基础设施和环境造成严重损害，准确及时的测绘对灾害准备和响应至关重要。然而，传统深度学习模型在不同传感器、区域或训练数据有限的情况下表现不佳。

### 目的

提出一个三轴分析框架（传感器、标签和领域）来适应地理空间基础模型(GeoFMs)，专注于Prithvi-EO-2.0模型用于滑坡测绘。

### 方法

基于全球预训练、自监督和可适应微调构建模型，并通过一系列实验比较不同模型性能。

### 主要发现

该模型在性能上超越了特定任务的CNNs（U-Net，U-Net++）、视觉变换器（Segformer，SwinV2-B）和其他GeoFMs（TerraMind，SatMAE）；对光谱变化具有鲁棒性；在标签稀缺的情况下保持准确性；在不同数据集和地理环境中更可靠地泛化。

### 结论

GeoFMs为减少滑坡风险和环境监测提供了更强大和可扩展的方法。研究也指出了剩余挑战，如计算成本和滑坡研究中可重用的AI就绪训练数据的有限可用性。

### 翻译

滑坡对生命、基础设施和环境造成严重损害，使得准确及时的测绘对灾害准备和响应至关重要。然而，传统的深度学习模型通常在不同传感器、区域或训练数据有限的情况下表现不佳。为应对这些挑战，我们提出了一个三轴分析框架（传感器、标签和领域）来适应地理空间基础模型(GeoFMs)，专注于Prithvi-EO-2.0用于滑坡测绘。通过一系列实验，我们表明它始终超越特定任务的CNNs（U-Net，U-Net++）、视觉变换器（Segformer，SwinV2-B）和其他GeoFMs（TerraMind，SatMAE）。该模型基于全球预训练、自监督和可适应微调，证明了对光谱变化的鲁棒性，在标签稀缺的情况下保持准确性，并在不同数据集和地理环境中更可靠地泛化。除了这些优势，我们还强调了剩余的挑战，如计算成本和滑坡研究中可重用的AI就绪训练数据的有限可用性。总体而言，我们的研究将GeoFMs定位为减少滑坡风险和环境监测的更强大和可扩展的方法。


### 论文摘要

Landslides cause severe damage to lives, infrastructure, and the environment, making accurate and timely mapping essential for disaster preparedness and response. However, conventional deep learning models often struggle when applied across different sensors, regions, or under conditions of limited training data. To address these challenges, we present a three-axis analytical framework of sensor, label, and domain for adapting geospatial foundation models (GeoFMs), focusing on Prithvi-EO-2.0 for landslide mapping. Through a series of experiments, we show that it consistently outperforms task-specific CNNs (U-Net, U-Net++), vision transformers (Segformer, SwinV2-B), and other GeoFMs (TerraMind, SatMAE). The model, built on global pretraining, self-supervision, and adaptable fine-tuning, proved resilient to spectral variation, maintained accuracy under label scarcity, and generalized more reliably across diverse datasets and geographic settings. Alongside these strengths, we also highlight remaining challenges such as computational cost and the limited availability of reusable AI-ready training data for landslide research. Overall, our study positions GeoFMs as a step toward more robust and scalable approaches for landslide risk reduction and environmental monitoring.

---

## 12. Vision Foundation Models in Agriculture: Toward Domain-Specific Adaptation for Weed Herbicide Trials Assessment

**论文链接:** [http://arxiv.org/abs/2511.04288v1](http://arxiv.org/abs/2511.04288v1)

**作者:** Leire Benito-Del-Valle, Artzai Picón, Daniel Mugica, Manuel Ramos, Eva Portillo, Javier Romero, Carlos Javier Jimenez, Ramón Navarra-Mestre

**发布时间:** 2025-11-06

### GPT解析

### 总结

本研究开发了一种针对除草剂试验的领域特定视觉模型，通过自监督学习在农业数据集上训练，显著提高了物种识别和损伤分类的准确性，特别是在未见条件下表现更佳，同时大幅减少了标注需求。

### 背景

除草剂田间试验需要准确识别植物物种并评估除草剂引起的损伤。通用视觉基础模型在复杂视觉领域表现良好，但在农业领域表现有限，因为农业中需要精细区分物种和损伤类型。

### 目的

将通用视觉基础模型适应于除草剂试验特征描述，提高物种识别和损伤分类的准确性。

### 方法

使用自监督学习方法在大型精选农业数据集上训练模型，学习针对除草剂试验图像优化的丰富且可迁移的表示。

### 主要发现

领域特定模型在物种识别(F1分数从0.91提高到0.94)和损伤分类(从0.26提高到0.33)方面显著优于通用模型；在未见条件下获得更大提升；在无人机图像等领域转换场景中保持强性能；领域特定预训练提高了分割准确性；在未见条件下，领域特定模型比通用模型实现5.4%更高的F1分数，同时使用80%更少的标记样本。

### 结论

领域特定基础模型具有强大的泛化能力，可以显著减少手动标注工作量，为除草剂试验分析提供可扩展和自动化的解决方案。

### 翻译

除草剂田间试验需要准确识别植物物种并评估除草剂引起的损伤，跨越不同环境。虽然通用视觉基础模型在复杂视觉领域显示出有希望的结果，但它们在农业领域的表现可能有限，因为物种和损伤类型之间的精细区分至关重要。在这项工作中，我们将通用视觉基础模型适应于除草剂试验特征描述。使用自监督学习方法在大型精选农业数据集上训练，该模型学习了针对除草剂试验图像优化的丰富且可迁移的表示。我们的领域特定模型在物种识别(F1分数从0.91提高到0.94)和损伤分类(从0.26提高到0.33)方面显著优于最佳通用基础模型。在未见条件下(新地点和其他时间)，它获得更大提升(物种识别从0.56提高到0.66；损伤分类从0.17提高到0.27)。在领域转换场景中，如无人机图像，它保持强性能(物种分类从0.49提高到0.60)。此外，我们表明领域特定预训练提高了分割准确性，特别是在低标注情况下。标注效率分析显示，在未见条件下，领域特定模型比通用模型实现5.4%更高的F1分数，同时使用80%更少的标记样本。这些结果证明了领域特定基础模型的泛化能力及其显著减少手动标注工作的潜力，为除草剂试验分析提供了可扩展和自动化的解决方案。


### 论文摘要

Herbicide field trials require accurate identification of plant species and assessment of herbicide-induced damage across diverse environments. While general-purpose vision foundation models have shown promising results in complex visual domains, their performance can be limited in agriculture, where fine-grained distinctions between species and damage types are critical.   In this work, we adapt a general-purpose vision foundation model to herbicide trial characterization. Trained using a self-supervised learning approach on a large, curated agricultural dataset, the model learns rich and transferable representations optimized for herbicide trials images.   Our domain-specific model significantly outperforms the best general-purpose foundation model in both species identification (F1 score improvement from 0.91 to 0.94) and damage classification (from 0.26 to 0.33). Under unseen conditions (new locations and other time), it achieves even greater gains (species identification from 0.56 to 0.66; damage classification from 0.17 to 0.27). In domain-shift scenarios, such as drone imagery, it maintains strong performance (species classification from 0.49 to 0.60).   Additionally, we show that domain-specific pretraining enhances segmentation accuracy, particularly in low-annotation regimes. An annotation-efficiency analysis reveals that, under unseen conditions, the domain-specific model achieves 5.4% higher F1 score than the general-purpose model, while using 80% fewer labeled samples.   These results demonstrate the generalization capabilities of domain-specific foundation models and their potential to significantly reduce manual annotation efforts, offering a scalable and automated solution for herbicide trial analysis.

---

## 13. A Parallel Region-Adaptive Differential Privacy Framework for Image Pixelization

**论文链接:** [http://arxiv.org/abs/2511.04261v1](http://arxiv.org/abs/2511.04261v1)

**作者:** Ming Liu

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种新的并行、区域自适应像素化框架，结合差分隐私的理论严谨性和实际效率，解决了高分辨率视觉系统中视频应用的隐私风险问题。

### 背景

高分辨率视觉传感系统的广泛部署和基础模型的兴起增加了基于视频的应用中的隐私风险。差分私有像素化虽然提供数学保证的保护，但在保持任务保真度、实现可扩展性和高效实时部署方面仍面临挑战。

### 目的

开发一种并行、区域自适应像素化框架，结合差分隐私的理论严谨性和实际效率，解决现有方法的局限性。

### 方法

提出自适应调整基于区域复杂性的网格大小和噪声规模的方法，利用GPU并行性实现运行时加速，引入轻量级存储方案减少空间开销，并在拉普拉斯机制和并行组合定理下提供正式隐私分析。

### 主要发现

在PETS、Venice-2和PPM-100数据集上的实验展示了有利的隐私-效用权衡和显著的运行时/存储减少。CelebA上的面部重新识别攻击实验证实了该方法在防止身份推断方面的有效性。

### 结论

该方法适合实时关键隐私应用，如老年人护理、智能家居监控、驾驶员行为分析和人群行为监控。

### 翻译

高分辨率视觉传感系统的广泛部署，结合基础模型的兴起，增加了基于视频的应用中的隐私风险。差分私有像素化通过基于网格的噪声添加为视觉数据提供数学保证的保护，但在保持任务相关的保真度、实现可扩展性和实现高效实时部署方面仍存在挑战。为此，我们提出了一种新颖的并行、区域自适应像素化框架，结合差分隐私的理论严谨性和实际效率。我们的方法基于区域复杂性自适应调整网格大小和噪声规模，利用GPU并行性实现比经典基线显著的运行时加速。通过仅保留必要的噪声统计信息，引入了轻量级存储方案，显著减少了空间开销。在拉普拉斯机制和并行组合定理下提供了正式的隐私分析。在PETS、Venice-2和PPM-100数据集上的大量实验展示了有利的隐私-效用权衡和显著的运行时/存储减少。在CelebA上的面部重新识别攻击实验进一步证实了该方法在防止身份推断方面的有效性。这验证了其适合于实时关键隐私应用，如老年人护理、智能家居监控、驾驶员行为分析和人群行为监控。


### 论文摘要

The widespread deployment of high-resolution visual sensing systems, coupled with the rise of foundation models, has amplified privacy risks in video-based applications. Differentially private pixelization offers mathematically guaranteed protection for visual data through grid-based noise addition, but challenges remain in preserving task-relevant fidelity, achieving scalability, and enabling efficient real-time deployment. To address this, we propose a novel parallel, region-adaptive pixelization framework that combines the theoretical rigor of differential privacy with practical efficiency. Our method adaptively adjusts grid sizes and noise scales based on regional complexity, leveraging GPU parallelism to achieve significant runtime acceleration compared to the classical baseline. A lightweight storage scheme is introduced by retaining only essential noisy statistics, significantly reducing space overhead. Formal privacy analysis is provided under the Laplace mechanism and parallel composition theorem. Extensive experiments on the PETS, Venice-2, and PPM-100 datasets demonstrate favorable privacy-utility trade-offs and significant runtime/storage reductions. A face re-identification attack experiment on CelebA further confirms the method's effectiveness in preventing identity inference. This validates its suitability for real-time privacy-critical applications such as elderly care, smart home monitoring, driver behavior analysis, and crowd behavior monitoring.

---

## 14. MedSapiens: Taking a Pose to Rethink Medical Imaging Landmark Detection

**论文链接:** [http://arxiv.org/abs/2511.04255v1](http://arxiv.org/abs/2511.04255v1)

**作者:** Marawan Elbatel, Anbang Wang, Keyuan Liu, Kaouther Mouheb, Enrique Almar-Munoz, Lizhuo Lin, Yanqi Yang, Karim Lekadir, Xiaomeng Li

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文重新审视了以人为中心的基础模型在医学影像解剖标志检测中的应用，提出了MedSapiens模型，在多个数据集上实现了最先进的性能。

### 背景

解剖标志检测传统上依赖特定领域模型，而大规模预训练视觉模型的出现提供了新机会。以人为中心的基础模型具有空间姿态定位的优化潜力，但这一潜力尚未被充分利用。

### 目的

研究将Sapiens（一种为姿态估计设计的以人为中心的基础模型）通过多数据集预训练适应到医学影像中，建立新的最先进性能。

### 方法

通过多数据集预训练将Sapiens模型适应到医学影像领域，提出MedSapiens模型，并在多个数据集和有限数据设置下评估其性能。

### 主要发现

MedSapiens在多个数据集上建立了新的最先进性能；相比通用模型，平均成功检测率提高5.26%；相比专业模型提高21.81%；在有限数据设置下，相比少样本最先进方法提高2.69%。

### 结论

以人为中心的基础模型因其内在的空间姿态定位优化，为解剖标志检测提供了强大的先验知识，这一潜力可以通过适当的方法得到充分利用。

### 翻译

本文并未引入新颖的架构；相反，它重新审视了一个基本但被忽视的基线：将以人为中心的基础模型适应用于医学影像中的解剖标志检测。虽然标志检测传统上依赖于特定领域的模型，但大规模预训练视觉模型的出现提供了新的机会。在本研究中，我们通过多数据集预训练研究将Sapiens（一种为姿态估计设计的以人为中心的基础模型）适应到医学影像中，在多个数据集上建立了新的最先进性能。我们提出的MedSapiens模型证明，以人为中心的基础模型，因其内在的空间姿态定位优化，为解剖标志检测提供了强大的先验知识，但这一潜力在很大程度上尚未被利用。我们将MedSapiens与现有最先进模型进行基准测试，在平均成功检测率上相比通用模型提高5.26%，相比专业模型提高21.81%。为了进一步评估MedSapiens在标注数据有限的情况下对新下游任务的适应能力，我们在有限数据设置下评估了其性能，在SDR上相比少样本最先进方法提高2.69%。代码和模型权重可在https://github.com/xmed-lab/MedSapiens获取。


### 论文摘要

This paper does not introduce a novel architecture; instead, it revisits a fundamental yet overlooked baseline: adapting human-centric foundation models for anatomical landmark detection in medical imaging. While landmark detection has traditionally relied on domain-specific models, the emergence of large-scale pre-trained vision models presents new opportunities. In this study, we investigate the adaptation of Sapiens, a human-centric foundation model designed for pose estimation, to medical imaging through multi-dataset pretraining, establishing a new state of the art across multiple datasets. Our proposed model, MedSapiens, demonstrates that human-centric foundation models, inherently optimized for spatial pose localization, provide strong priors for anatomical landmark detection, yet this potential has remained largely untapped. We benchmark MedSapiens against existing state-of-the-art models, achieving up to 5.26% improvement over generalist models and up to 21.81% improvement over specialist models in the average success detection rate (SDR). To further assess MedSapiens adaptability to novel downstream tasks with few annotations, we evaluate its performance in limited-data settings, achieving 2.69% improvement over the few-shot state of the art in SDR. Code and model weights are available at https://github.com/xmed-lab/MedSapiens .

---

## 15. BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.04131v1](http://arxiv.org/abs/2511.04131v1)

**作者:** Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了BFM-Zero框架，一种用于类人机器人的行为基础模型，通过学习共享潜在表示将运动、目标和奖励嵌入公共空间，实现单一策略支持多种下游任务，无需重新训练。

### 背景

现有方法要么只在模拟环境中部署，要么专门针对特定任务如跟踪，缺乏统一的多任务控制能力。

### 目的

开发一种能够在真实世界中实现多样化且稳健整体技能的行为基础模型框架，支持零样本运动跟踪、目标到达和奖励优化等多种推理方法。

### 方法

基于无监督强化学习和前向-后向(FB)模型构建，结合奖励塑造、领域随机化和历史依赖的非对称学习来弥合模拟到现实的差距。

### 主要发现

BFM-Zero在Unitree G1类人机器人上实现了多样化且稳健的整体技能，通过结构良好的潜在空间支持多种推理方法，且关键设计选择在模拟中得到了定量验证。

### 结论

BFM-Zero作为首创模型，为可扩展、可提示的行为基础模型用于整体类人控制奠定了基础。

### 翻译

为类人机器人构建行为基础模型(BFMs)有潜力将多样化的控制任务统一在单个可提示的通用策略下。然而，现有方法要么仅在模拟的类人角色上部署，要么专门针对特定任务如跟踪。我们提出了BFM-Zero框架，它学习了一个有效的共享潜在表示，将运动、目标和奖励嵌入公共空间，使单一策略能够被提示执行多个下游任务而无需重新训练。BFM-Zero中的这种结构良好的潜在空间使得在真实世界的Unitree G1类人机器人上的多样化且稳健的整体技能成为可能，通过多样化的推理方法，包括零样本运动跟踪、目标到达和奖励优化，以及少样本基于优化的自适应。与先前的在线强化学习(RL)框架不同，BFM-Zero建立在无监督RL和前向-后向(FB)模型的最新进展之上，这些模型为中心目标、可解释和流畅的整体运动潜在表示提供了支持。我们进一步通过关键的奖励塑造、领域随机化和历史依赖的非对称学习扩展了BFM-Zero，以弥合模拟到现实的差距。这些关键设计选择在模拟中进行了定量消融实验。作为首创模型，BFM-Zero为可扩展、可提示的行为基础模型用于整体类人控制迈出了第一步。


### 论文摘要

Building Behavioral Foundation Models (BFMs) for humanoid robots has the potential to unify diverse control tasks under a single, promptable generalist policy. However, existing approaches are either exclusively deployed on simulated humanoid characters, or specialized to specific tasks such as tracking. We propose BFM-Zero, a framework that learns an effective shared latent representation that embeds motions, goals, and rewards into a common space, enabling a single policy to be prompted for multiple downstream tasks without retraining. This well-structured latent space in BFM-Zero enables versatile and robust whole-body skills on a Unitree G1 humanoid in the real world, via diverse inference methods, including zero-shot motion tracking, goal reaching, and reward optimization, and few-shot optimization-based adaptation. Unlike prior on-policy reinforcement learning (RL) frameworks, BFM-Zero builds upon recent advancements in unsupervised RL and Forward-Backward (FB) models, which offer an objective-centric, explainable, and smooth latent representation of whole-body motions. We further extend BFM-Zero with critical reward shaping, domain randomization, and history-dependent asymmetric learning to bridge the sim-to-real gap. Those key design choices are quantitatively ablated in simulation. A first-of-its-kind model, BFM-Zero establishes a step toward scalable, promptable behavioral foundation models for whole-body humanoid control.

---

## 16. How Natural Language Proficiency Shapes GenAI Code for Software Engineering Tasks

**论文链接:** [http://arxiv.org/abs/2511.04115v1](http://arxiv.org/abs/2511.04115v1)

**作者:** Ruksit Rojpaisarnkit, Youmei Fan, Kenichi Matsumoto, Raula Gaikovina Kula

**发布时间:** 2025-11-06

**DOI:** 10.1109/MS.2025.3622690

**备注:** 7 pages, 4 tables, 1 figure

### GPT解析

### 总结

本研究探讨了英语语言水平对大型语言模型生成代码质量和正确性的影响，发现自然语言能力是控制代码生成的关键因素。

### 背景

随着基础模型驱动的工具在软件工程中的广泛采用，自然语言提示已成为开发人员与大型语言模型之间的重要接口，但自然语言能力对代码生成质量的影响研究不足。

### 目的

调查英语语言能力本身（独立于提示技术）是否会影响大型语言模型生成代码的能力和正确性。

### 方法

使用HumanEval数据集，对164个编程任务的提示英语水平从基础到高级进行系统变化，并测量生成的代码水平和正确性。

### 主要发现

大型语言模型默认使用中级（B2）自然语言水平；对生成代码水平的影响因模型而异；在所有模型中，更高水平的提示始终产生更正确的代码。

### 结论

自然语言能力是控制代码生成的关键杠杆，可以帮助开发人员定制AI输出并提高解决方案的可靠性。

### 翻译

随着基础模型驱动的工具在软件工程中的广泛采用，自然语言提示已成为开发人员与大型语言模型之间的重要接口。虽然许多研究关注提示结构，但自然语言能力这一影响生成代码质量的因素却很少被探索。本文研究了英语语言能力本身（独立于提示技术）是否会影响大型语言模型生成代码的能力和正确性。使用HumanEval数据集，我们对164个编程任务的提示英语水平从基础到高级进行了系统变化，并测量了生成的代码水平和正确性。我们的发现表明，大型语言模型默认使用中级（B2）自然语言水平。虽然对生成代码水平的影响因模型而异，但我们发现所有模型中更高水平的提示始终产生更正确的代码。这些结果表明，自然语言能力是控制代码生成的关键杠杆，可以帮助开发人员定制AI输出并提高解决方案的可靠性。


### 论文摘要

With the widespread adoption of Foundation Model (FM)-powered tools in software engineering, the natural language prompt has become a critical interface between developers and Large Language Models (LLMs). While much research has focused on prompt structure, the natural language proficiency is an underexplored factor that can influence the quality of generated code. This paper investigates whether the English language proficiency itself independent of the prompting technique affects the proficiency and correctness of code generated by LLMs. Using the HumanEval dataset, we systematically varied the English proficiency of prompts from basic to advanced for 164 programming tasks and measured the resulting code proficiency and correctness. Our findings show that LLMs default to an intermediate (B2) natural language level. While the effect on the resulting code proficiency was model-dependent, we found that higher-proficiency prompts consistently yielded more correct code across all models. These results demonstrate that natural language proficiency is a key lever for controlling code generation, helping developers tailor AI output and improve the reliability of solutions.

---

## 17. Tiny-WiFo: A Lightweight Wireless Foundation Model for Channel Prediction via Multi-Component Adaptive Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2511.04015v1](http://arxiv.org/abs/2511.04015v1)

**作者:** Haotian Zhang, Shijian Gao, Xiang Cheng

**发布时间:** 2025-11-06

**备注:** 5 pages, 1 figures, 3 tables

### GPT解析

### 总结

本文提出了一种多组件自适应知识蒸馏框架，通过选择性知识提取和学习策略平衡，实现了无线基础模型在边缘设备上的高效部署。

### 背景

无线基础模型规模巨大，阻碍了它们在边缘设备上的实时部署。

### 目的

超越标准知识蒸馏方法，提出新的知识蒸馏框架以实现模型的高效压缩与边缘部署。

### 方法

引入基于交叉注意力的知识选择模块从教师模型中选择性识别关键特征，以及自主学习-被动学习策略平衡知识转移与独立学习，在可管理计算成本下实现高训练效率。

### 主要发现

应用于WiFo FM时，蒸馏出的Tiny-WiFo模型仅有550万个参数，在边缘硬件上实现1.6毫秒推理时间，同时保留了WiFo超过98%的性能及其关键的零样本泛化能力。

### 结论

所提出的方法使得无线基础模型在边缘设备上的实时部署成为可能。

### 翻译

无线基础模型的巨大规模阻碍了它们在边缘设备上的实时部署。本文通过引入一种新的多组件自适应知识蒸馏框架，超越了标准知识蒸馏。关键创新包括一个基于交叉注意力的知识选择模块，它从教师模型中选择性地识别关键特征，以及一个自主学习-被动学习策略，该策略平衡知识转移与独立学习，以在可管理的计算成本下实现高训练效率。当应用于WiFo FM时，蒸馏出的Tiny-WiFo模型仅有550万个参数，在边缘硬件上实现1.6毫秒的推理时间，同时保留了WiFo超过98%的性能及其关键的零样本泛化能力，使得实时FM部署成为可能。


### 论文摘要

The massive scale of Wireless Foundation Models (FMs) hinders their real-time deployment on edge devices. This letter moves beyond standard knowledge distillation by introducing a novel Multi-Component Adaptive Knowledge Distillation (MCAKD) framework. Key innovations include a Cross-Attention-Based Knowledge Selection (CA-KS) module that selectively identifies critical features from the teacher model, and an Autonomous Learning-Passive Learning (AL-PL) strategy that balances knowledge transfer with independent learning to achieve high training efficiency at a manageable computational cost. When applied to the WiFo FM, the distilled Tiny-WiFo model, with only 5.5M parameters, achieves a 1.6 ms inference time on edge hardware while retaining over 98% of WiFo's performance and its crucial zero-shot generalization capability, making real-time FM deployment viable.

---

## 18. SynQuE: Estimating Synthetic Dataset Quality Without Annotations

**论文链接:** [http://arxiv.org/abs/2511.03928v1](http://arxiv.org/abs/2511.03928v1)

**作者:** Arthur Chen, Victor Zhong

**发布时间:** 2025-11-06

**备注:** Under review

### GPT解析

### 总结

作者引入并形式化了合成数据集质量评估(SynQuE)问题，建立了首个全面基准，提出了代理指标LENS，并通过实验证明SynQuE代理能有效选择合成数据提高任务性能。

### 背景

由于收集成本或隐私限制导致数据稀缺，这是一个关键且开放的挑战。

### 目的

建立SynQuE问题的首个全面基准，通过引入和评估代理指标来选择合成数据以最大化在真实数据上的任务性能。

### 方法

引入首个SynQuE代理指标，通过嵌入模型调整基于分布和多样性的距离度量；提出LENS，一个利用大语言模型推理的新代理，以解决这些指标在复杂规划任务上的不足。

### 主要发现

SynQuE代理与多种任务（情感分析、Text2SQL、网页导航和图像分类）的真实任务性能相关；LENS在复杂任务上始终优于其他代理；在文本到SQL解析任务中，通过SynQuE代理选择的前3个合成数据集训练，平均准确率提高了8.1%。

### 结论

这项工作建立了SynQuE作为真实数据稀缺下合成数据选择的实用框架，并激励了基于基础模型的数据表征和细粒度数据选择的未来研究。

### 翻译

我们引入并形式化了合成数据集质量评估(SynQuE)问题：仅使用有限的无标注真实数据，按预期真实世界任务性能对合成数据集进行排序。这解决了一个关键且开放的挑战，由于收集成本或隐私限制导致数据稀缺。我们通过引入和评估选择合成数据用于训练以最大化在真实数据上任务性能的代理指标，为这个问题建立了首个全面基准。我们通过嵌入模型调整基于分布和多样性的距离度量，引入了首个SynQuE代理指标。为了解决这些指标在复杂规划任务上的不足，我们提出了LENS，一个利用大语言模型推理的新颖代理。我们的结果显示，SynQuE代理与多种任务（包括情感分析、Text2SQL、网页导航和图像分类）的真实任务性能相关，LENS通过捕捉细微特征在复杂任务上始终优于其他代理。例如，在文本到SQL解析任务中，通过SynQuE代理选择的前3个合成数据集训练，平均准确率从不加选择地选择数据的30.4%提高到38.4%（+8.1%）。这项工作建立了SynQuE作为真实数据稀缺下合成数据选择的实用框架，并激励了基于基础模型的数据表征和细粒度数据选择的未来研究。


### 论文摘要

We introduce and formalize the Synthetic Dataset Quality Estimation (SynQuE) problem: ranking synthetic datasets by their expected real-world task performance using only limited unannotated real data. This addresses a critical and open challenge where data is scarce due to collection costs or privacy constraints. We establish the first comprehensive benchmarks for this problem by introducing and evaluating proxy metrics that choose synthetic data for training to maximize task performance on real data. We introduce the first proxy metrics for SynQuE by adapting distribution and diversity-based distance measures to our context via embedding models. To address the shortcomings of these metrics on complex planning tasks, we propose LENS, a novel proxy that leverages large language model reasoning. Our results show that SynQuE proxies correlate with real task performance across diverse tasks, including sentiment analysis, Text2SQL, web navigation, and image classification, with LENS consistently outperforming others on complex tasks by capturing nuanced characteristics. For instance, on text-to-SQL parsing, training on the top-3 synthetic datasets selected via SynQuE proxies can raise accuracy from 30.4% to 38.4 (+8.1)% on average compared to selecting data indiscriminately. This work establishes SynQuE as a practical framework for synthetic data selection under real-data scarcity and motivates future research on foundation model-based data characterization and fine-grained data selection.

---

## 19. PLLuM: A Family of Polish Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.03823v1](http://arxiv.org/abs/2511.03823v1)

**作者:** Jan Kocoń, Maciej Piasecki, Arkadiusz Janz, Teddy Ferdinan, Łukasz Radliński, Bartłomiej Koptyra, Marcin Oleksy, Stanisław Woźniak, Paweł Walkowiak, Konrad Wojtasik, Julia Moska, Tomasz Naskręt, Bartosz Walkowiak, Mateusz Gniewkowski, Kamil Szyc, Dawid Motyka, Dawid Banach, Jonatan Dalasiński, Ewa Rudnicka, Bartłomiej Alberski, Tomasz Walkowiak, Aleksander Szczęsny, Maciej Markiewicz, Tomasz Bernaś, Hubert Mazur, Kamil Żyta, Mateusz Tykierko, Grzegorz Chodak, Tomasz Kajdanowicz, Przemysław Kazienko, Agnieszka Karlińska, Karolina Seweryn, Anna Kołos, Maciej Chrabąszcz, Katarzyna Lorenc, Aleksandra Krasnodębska, Artur Wilczek, Katarzyna Dziewulska, Paula Betscher, Zofia Cieślińska, Katarzyna Kowol, Daria Mikoś, Maciej Trzciński, Dawid Krutul, Marek Kozłowski, Sławomir Dadas, Rafał Poświata, Michał Perełkiewicz, Małgorzata Grębowiec, Maciej Kazuła, Marcin Białas, Roman Roszko, Danuta Roszko, Jurgita Vaičenonienė, Andrius Utka, Paweł Levchuk, Paweł Kowalski, Irena Prawdzic-Jankowska, Maciej Ogrodniczuk, Monika Borys, Anna Bulińska, Wiktoria Gumienna, Witold Kieraś, Dorota Komosińska, Katarzyna Krasnowska-Kieraś, Łukasz Kobyliński, Martyna Lewandowska, Marek Łaziński, Mikołaj Łątkowski, Dawid Mastalerz, Beata Milewicz, Agnieszka Anna Mykowiecka, Angelika Peljak-Łapińska, Sandra Penno, Zuzanna Przybysz, Michał Rudolf, Piotr Rybak, Karolina Saputa, Aleksandra Tomaszewska, Aleksander Wawer, Marcin Woliński, Joanna Wołoszyn, Alina Wróblewska, Bartosz Żuk, Filip Żarnecki, Konrad Kaczyński, Anna Cichosz, Zuzanna Deckert, Monika Garnys, Izabela Grabarczyk, Wojciech Janowski, Sylwia Karasińska, Aleksandra Kujawiak, Piotr Misztela, Maria Szymańska, Karolina Walkusz, Igor Siek, Jakub Kwiatkowski, Piotr Pęzik

**发布时间:** 2025-11-05

**备注:** 83 pages, 19 figures

### GPT解析

### 总结

PLLuM是最大的开源波兰语基础模型家族，由波兰主要研究机构联盟开发，旨在提供高质量、透明且文化相关的波兰语语言模型，以应对英语主导的商业语言模型格局。

### 背景

大型语言模型在人工智能中扮演核心角色，但其发展主要集中于英语，导致对其他语言的支持有限。

### 目的

开发专门针对波兰语言的高质量、透明和文化相关的基础模型，满足非英语语言需求，并促进开放研究和加强波兰的主权AI技术。

### 方法

由波兰主要研究机构联盟开发；构建了1400亿个波兰语文本语料库进行预训练；创建了77k的自定义指令数据集和100k的偏好优化数据集；采用负责任的AI框架，包括严格的数据治理和用于输出校正与安全过滤的混合模块；详细说明了模型的架构、训练过程和对齐技术。

### 主要发现

PLLuM模型在公共管理领域的下游任务中展示了其实用性，证明了专门针对特定语言开发的大型语言模型的有效性。

### 结论

通过公开发布PLLuM模型，旨在促进开放研究并加强波兰的主权AI技术发展。

### 翻译

大型语言模型在现代人工智能中扮演核心角色，但其开发主要集中于英语，导致对其他语言的支持有限。我们提出了PLLuM（波兰大型语言模型），这是专门为波兰语言定制的最大开源基础模型家族。由波兰主要研究机构联盟开发，PLLuM满足了在英语主导的商业格局之外，对高质量、透明且与文化相关的语言模型的需求。我们描述了开发过程，包括为预训练构建新的1400亿个波兰语文本语料库、77k的自定义指令数据集和100k的偏好优化数据集。关键组成部分是一个负责任的AI框架，它采用严格的数据治理和用于输出校正与安全过滤的混合模块。我们详细说明了基础模型和指令调整变体的架构、训练过程和对齐技术，并在公共管理领域的下游任务中展示了它们的实用性。通过公开发布这些模型，PLLuM旨在促进开放研究并加强波兰的主权AI技术。


### 论文摘要

Large Language Models (LLMs) play a central role in modern artificial intelligence, yet their development has been primarily focused on English, resulting in limited support for other languages. We present PLLuM (Polish Large Language Model), the largest open-source family of foundation models tailored specifically for the Polish language. Developed by a consortium of major Polish research institutions, PLLuM addresses the need for high-quality, transparent, and culturally relevant language models beyond the English-centric commercial landscape. We describe the development process, including the construction of a new 140-billion-token Polish text corpus for pre-training, a 77k custom instructions dataset, and a 100k preference optimization dataset. A key component is a Responsible AI framework that incorporates strict data governance and a hybrid module for output correction and safety filtering. We detail the models' architecture, training procedures, and alignment techniques for both base and instruction-tuned variants, and demonstrate their utility in a downstream task within public administration. By releasing these models publicly, PLLuM aims to foster open research and strengthen sovereign AI technologies in Poland.

---

## 20. FusionDP: Foundation Model-Assisted Differentially Private Learning for Partially Sensitive Features

**论文链接:** [http://arxiv.org/abs/2511.03806v1](http://arxiv.org/abs/2511.03806v1)

**作者:** Linghui Zeng, Ruixuan Liu, Atiquer Rahman Sarkar, Xiaoqian Jiang, Joyce C. Ho, Li Xiong

**发布时间:** 2025-11-05

### GPT解析

### 总结

FusionDP是一个两步框架，用于在特征级别差分隐私下增强模型效用，通过利用大型基础模型估算敏感特征并修改DP-SGD算法，在保持隐私的同时提高模型性能。

### 背景

在隐私保护机器学习中，确保敏感训练数据的隐私至关重要，但实际场景中可能只需要对部分特征进行隐私保护，例如ICU数据中的人口统计属性比原始实验室结果更敏感。

### 目的

开发一种方法，能够在特征级别差分隐私下增强模型效用，避免传统DP-SGD方法对所有特征强制执行隐私保护导致的过度噪声注入和效用下降问题。

### 方法

FusionDP采用两步框架：首先利用大型基础模型根据非敏感特征估算敏感特征，作为外部先验提供高质量估计；其次引入修改的DP-SGD算法，在原始和估算特征上同时训练模型，同时保留原始敏感特征的隐私。

### 主要发现

在PhysioNet的败血症预测任务和MIMIC-III的临床笔记分类任务上评估显示，FusionDP与隐私保护基线相比，在保持严格特征级别隐私的同时显著提高了模型性能。

### 结论

基础模型驱动的估算能够增强各种模态的隐私-效用权衡，为隐私保护机器学习提供了新的有效方法。

### 翻译

确保敏感训练数据的隐私在隐私保护机器学习中至关重要。然而，在实际场景中，可能只需要对部分特征进行隐私保护。例如，在ICU数据中，年龄和性别等人口统计属性由于重新识别的潜在风险而具有更高的隐私风险，而原始实验室结果通常敏感性较低。传统的DP-SGD对样本中的所有特征强制执行隐私保护，导致过度噪声注入和显著的效用下降。我们提出了FusionDP，一个两步框架，用于在特征级别差分隐私下增强模型效用。首先，FusionDP利用大型基础模型根据非敏感特征估算敏感特征，将它们作为外部先验，在模型训练期间不访问真实值的情况下提供高质量敏感属性估计。其次，我们引入了修改的DP-SGD算法，在原始和估算特征上同时训练模型，同时正式保留原始敏感特征的隐私。我们在两种模态上评估了FusionDP：来自PhysioNet的表格数据上的败血症预测任务和来自MIMIC-III的临床笔记分类任务。通过与隐私保护基线进行比较，我们的结果显示FusionDP在保持严格特征级别隐私的同时显著提高了模型性能，展示了基础模型驱动的估算在增强各种模态的隐私-效用权衡方面的潜力。


### 论文摘要

Ensuring the privacy of sensitive training data is crucial in privacy-preserving machine learning. However, in practical scenarios, privacy protection may be required for only a subset of features. For instance, in ICU data, demographic attributes like age and gender pose higher privacy risks due to their re-identification potential, whereas raw lab results are generally less sensitive. Traditional DP-SGD enforces privacy protection on all features in one sample, leading to excessive noise injection and significant utility degradation. We propose FusionDP, a two-step framework that enhances model utility under feature-level differential privacy. First, FusionDP leverages large foundation models to impute sensitive features given non-sensitive features, treating them as external priors that provide high-quality estimates of sensitive attributes without accessing the true values during model training. Second, we introduce a modified DP-SGD algorithm that trains models on both original and imputed features while formally preserving the privacy of the original sensitive features. We evaluate FusionDP on two modalities: a sepsis prediction task on tabular data from PhysioNet and a clinical note classification task from MIMIC-III. By comparing against privacy-preserving baselines, our results show that FusionDP significantly improves model performance while maintaining rigorous feature-level privacy, demonstrating the potential of foundation model-driven imputation to enhance the privacy-utility trade-off for various modalities.

---

## 21. 论文ID: 2511.04155v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.04155v1.json'

---

## 22. Transfer Learning for Transformer-Based Modeling of Nonlinear Pulse Evolution in Er-Doped Fiber Amplifiers

**论文链接:** [http://arxiv.org/abs/2511.04057v1](http://arxiv.org/abs/2511.04057v1)

**作者:** Anastasia Bednyakova, Artem Gemuzov, Mikhail Mishevsky, Karina Saraeva, Alexey Redyuk, Aram Mkrtchyan, Albert Nasibulin, Yuriy Gladush

**发布时间:** 2025-11-06

### GPT解析

### 总结

本研究开发了一种基于Transformer架构的神经网络模型，用于预测掺铒光纤放大器中光脉冲的非线性演化。通过两阶段训练策略（先在合成数据上预训练，再用少量实验数据微调），成功解决了数据稀缺问题，能够准确重现实验中观察到的光脉冲精细光谱结构。

### 背景

在实验数据有限的情况下预测掺铒光纤放大器中光脉冲的非线性演化是一个挑战。传统的数值模拟方法可能难以准确捕捉实验中观察到的精细光谱结构，而纯粹的机器学习方法又面临数据稀缺的问题。

### 目的

开发一种能够准确预测掺铒光纤放大器中光脉冲非线性演化的神经网络模型，特别是在实验数据有限的情况下，能够准确重现实验中观察到的光脉冲精细光谱结构。

### 方法

采用基于Transformer架构的神经网络模型，并实施两阶段训练策略：第一阶段在通过数值模拟生成的合成数据集上预训练模型；第二阶段使用少量实验测量数据对模型进行微调。

### 主要发现

该方法能够准确重现实验中观察到的光脉冲的精细光谱结构，在各种非线性演化情况下表现良好，包括调制不稳定性的发展和高阶孤子的传播。

### 结论

结合物理模型（通过数值模拟生成合成数据）和机器学习（神经网络模型）的两阶段训练策略，可以有效解决数据稀缺问题，实现对光脉冲非线性演化的准确预测。

### 翻译

基于Transformer架构的神经网络模型已经开发出来，用于在实验数据有限的情况下预测掺铒光纤放大器中光脉冲的非线性演化。为了解决数据稀缺问题，采用了两阶段训练策略。在第一阶段，模型在通过放大器非线性动力学的数值模拟生成的合成数据集上进行预训练。在第二阶段，使用少量实验测量数据对模型进行微调。这种方法能够准确重现实验中观察到的各种非线性演化情况下的光脉冲精细光谱结构，包括调制不稳定性的发展和高阶孤子的传播。


### 论文摘要

A neural network model based on the Transformer architecture has been developed to predict the nonlinear evolution of optical pulses in Er-doped fiber amplifier under conditions of limited experimental data. To address data scarcity, a two-stage training strategy is employed. In the first stage, the model is pretrained on a synthetic dataset generated through numerical simulations of the amplifier's nonlinear dynamics. In the second stage, the model is fine-tuned using a small set of experimental measurements. This approach enables accurate reproduction of the fine spectral structure of optical pulses observed in experiments across various nonlinear evolution regimes, including the development of modulational instability and the propagation of high-order solitons.

---

## 23. Shared Spatial Memory Through Predictive Coding

**论文链接:** [http://arxiv.org/abs/2511.04235v1](http://arxiv.org/abs/2511.04235v1)

**作者:** Zhengru Fang, Yu Guo, Jingjing Wang, Yuang Zhang, Haonan An, Yinhai Wang, Yuguang Fang

**发布时间:** 2025-11-06

**备注:** We have prepared the open-source code and video demonstration pages:  1. Code: github.com/fangzr/SSM-PC 2. Demo: fangzr.github.io/SSM-PC/index.html

### GPT解析

### 总结

本文介绍了一种多智能体预测编码框架，将协调视为智能体间相互不确定性的最小化。该框架通过信息瓶颈目标促使智能体学习何时以及与谁通信什么内容，并采用类似网格细胞的度量作为内部空间编码。基于此，智能体发展出带宽高效的通信机制和专门编码伙伴位置的神经群体，通过分层强化学习策略主动探索以减少联合不确定性。

### 背景

在多智能体系统中，共享和重建一致的空间记忆是一个关键挑战。部分可观测性和有限带宽常常导致协调中的灾难性故障。

### 目的

开发一种多智能体预测编码框架，将协调表述为智能体间相互不确定性的最小化，并解决带宽限制下的协调问题。

### 方法

1. 引入多智能体预测编码框架，将协调表述为最小化智能体间的不确定性；2. 使用信息瓶颈目标促使智能体学习何时以及与谁通信什么内容；3. 采用类似网格细胞的度量作为内部空间编码用于自我定位；4. 通过自我监督的运动预测形成内部空间编码；5. 基于内部空间编码发展带宽高效的通信机制；6. 开发专门编码伙伴位置的神经群体；7. 使用分层强化学习策略主动探索以减少联合不确定性。

### 主要发现

在Memory-Maze基准测试中，该方法对带宽限制表现出极强的鲁棒性：当带宽从128位/步缩减到4位/步时，成功率从73.5%逐渐下降到64.4%；而全广播基线方法则从67.6%急剧下降到28.6%。

### 结论

研究结果为复杂社交表征如何从统一的预测驱动中涌现提供了理论上合理且生物学 plausible 的基础，从而形成社交集体智能。

### 翻译

共享和重建一致的空间记忆是多智能体系统中的一个关键挑战，其中部分可观测性和有限带宽常常导致协调中的灾难性故障。我们引入了一种多智能体预测编码框架，将协调表述为智能体间相互不确定性的最小化。作为信息瓶颈目标的实例化，它促使智能体学习不仅与谁通信什么内容，还有何时通信。该框架的基础是一种类似网格细胞的度量，作为自我定位的内部空间编码，通过自我监督的运动预测自发形成。基于这种内部空间编码，智能体逐渐发展出带宽高效的通信机制和专门编码伙伴位置的神经群体：海马体社交位置细胞的一种人工模拟。这些社交表示通过分层强化学习策略进一步实现，该策略主动探索以减少联合不确定性。在Memory-Maze基准测试中，我们的方法对带宽限制表现出极强的鲁棒性：当带宽从128位/步缩减到4位/步时，成功率从73.5%逐渐下降到64.4%，而全广播基线方法则从67.6%急剧下降到28.6%。我们的研究结果为复杂社交表征如何从统一的预测驱动中涌现提供了理论上合理且生物学 plausible 的基础，从而形成社交集体智能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决多智能体系统中的共享空间记忆问题，特别是在通信带宽受限的情况下如何实现有效的协调。这个问题在现实中非常重要，因为多智能体系统在探索、搜索和救援等领域具有广泛应用，但通信带宽往往是有限的资源。在生物学上，理解多智能体如何共享空间记忆有助于理解生物群体的协调机制。在技术层面，解决'通信瓶颈'问题对于构建高效的多智能体系统至关重要，避免因通信限制导致的协调失败。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过整合生物学启发、预测编码理论和信息论原理设计该方法。首先借鉴了生物学中的海马体-内嗅系统和社会位置细胞的发现，然后结合预测编码理论（大脑通过预测感官输入并最小化误差处理信息）和信息瓶颈理论（在压缩率和预测效用间权衡）。作者设计了三层框架：个体感知（网格细胞网络）、社会通信（信息瓶颈目标）和战略探索（分层强化学习）。该方法借鉴了多智能体强化学习的MAPPO算法，并受到大型语言模型'下一个token预测'目标的启发，将预测学习作为构建共享世界模型的通用机制。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过最小化智能体之间的相互预测不确定性，使多智能体系统能自发形成高效的空间记忆共享机制。整体实现流程分为三步：1) 个体层面，每个智能体使用网格细胞网络（路径积分）和视觉预测编码（BEV地图生成）构建内部空间模型；2) 社会层面，通过信息瓶颈目标学习传输压缩的离散符号，在关键时刻减少伙伴的不确定性；3) 战略层面，使用分层强化学习框架（HRL-ICM），基于预测不确定性做出决策，协调探索以找到隐藏目标。这种方法使智能体在极低带宽条件下（4-128位/步）保持高效协调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一预测框架，通过三个层次实现共享空间记忆；2) 自发形成的网格细胞样表征，用于自我定位；3) 基于信息瓶颈的高效社会通信机制，形成有意义的符号词汇表；4) 涌现的社会位置细胞，编码伙伴位置；5) 分层强化学习框架，基于预测不确定性做决策。相比之前工作，本文方法在带宽受限条件下表现出色（带宽从128降到4位/步时，成功率仅从73.5%降到64.4%，而全广播基线从67.6%降到28.6%），且表征是自发涌现而非预设的，为集体智能提供了生物学合理的解释。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过基于预测编码的多智能体框架，展示了智能体如何在有限带宽条件下自发形成高效的空间记忆共享机制，包括网格细胞样的空间度量、带宽优化的社会通信和社会位置细胞表征，为理解生物集体智能和设计高效的人工多智能体系统提供了理论基础。'}


### 论文摘要

Sharing and reconstructing a consistent spatial memory is a critical challenge in multi-agent systems, where partial observability and limited bandwidth often lead to catastrophic failures in coordination. We introduce a multi-agent predictive coding framework that formulate coordination as the minimization of mutual uncertainty among agents. Instantiated as an information bottleneck objective, it prompts agents to learn not only who and what to communicate but also when. At the foundation of this framework lies a grid-cell-like metric as internal spatial coding for self-localization, emerging spontaneously from self-supervised motion prediction. Building upon this internal spatial code, agents gradually develop a bandwidth-efficient communication mechanism and specialized neural populations that encode partners' locations: an artificial analogue of hippocampal social place cells (SPCs). These social representations are further enacted by a hierarchical reinforcement learning policy that actively explores to reduce joint uncertainty. On the Memory-Maze benchmark, our approach shows exceptional resilience to bandwidth constraints: success degrades gracefully from 73.5% to 64.4% as bandwidth shrinks from 128 to 4 bits/step, whereas a full-broadcast baseline collapses from 67.6% to 28.6%. Our findings establish a theoretically principled and biologically plausible basis for how complex social representations emerge from a unified predictive drive, leading to social collective intelligence.

---

## 24. SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.04668v1](http://arxiv.org/abs/2511.04668v1)

**作者:** Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, Saining Xie

**发布时间:** 2025-11-06

**备注:** Project page: https://ellisbrown.github.io/sims-v

### GPT解析

### 总结

该研究解决了多模态语言模型在空间推理方面的挑战，通过创新的模拟数据生成框架SIMS-V，有效提高了模型在真实世界空间任务上的性能。

### 背景

多模态语言模型在高级视频理解方面表现出色，但在跨时间和空间的空间推理方面存在困难。当前的空间训练方法依赖于真实世界的视频数据，但获取具有精确空间标注的多样化素材仍然是一个瓶颈。

### 目的

为了解决这一瓶颈，作者提出了SIMS-V框架，这是一个系统性的数据生成框架，利用3D模拟器的特权信息来创建空间丰富的视频训练数据，用于多模态语言模型。

### 方法

作者使用这个框架，通过系统性地消融问题类型、混合和规模，研究了哪些模拟数据的特性能够有效促进真实世界的迁移学习。

### 主要发现

识别出三个最有效的问题类别（度量测量、视角依赖推理和时间跟踪），它们对于开发可迁移的空间智能最为有效；尽管使用的问类型较少，但这三个类别比全面覆盖更有效；仅使用25K个模拟示例进行微调的70亿参数视频LLM，性能超过了更大的720亿基线模型；在严格的真实世界空间推理基准测试中，与专有模型实现了竞争性性能。

### 结论

该方法展示了强大的泛化能力，在保持一般视频理解性能的同时，在具身和真实世界空间任务上显示出显著改进。

### 翻译

尽管在高级视频理解方面表现出色，多模态语言模型在跨时间和空间的空间推理方面仍然存在困难。虽然当前的空间训练方法依赖于真实世界的视频数据，但获取具有精确空间标注的多样化素材仍然是一个瓶颈。为了缓解这一瓶颈，我们提出了SIMS-V——一个系统性的数据生成框架，它利用3D模拟器的特权信息来创建空间丰富的视频训练数据，用于多模态语言模型。使用这个框架，我们通过系统性地消融问题类型、混合和规模，研究了哪些模拟数据的特性能够有效促进真实世界的迁移学习。我们确定了一个最小的三个问题类别集合（度量测量、视角依赖推理和时间跟踪），它们被证明对于开发可迁移的空间智能最为有效，尽管使用的问类型较少，但比全面覆盖更有效。这些见解实现了高效的训练：仅使用25K个模拟示例进行微调的70亿参数视频LLM，性能超过了更大的720亿基线模型，并在严格的真实世界空间推理基准测试中与专有模型实现了竞争性性能。我们的方法展示了强大的泛化能力，在保持一般视频理解性能的同时，在具身和真实世界空间任务上显示出显著改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态语言模型在空间推理方面的局限性，特别是在跨越时间和空间维度的空间理解能力。这个问题很重要，因为空间推理是人类智能的核心能力，对于理解和交互物理世界至关重要，但获取真实世界视频数据并添加精确的空间标注非常困难和昂贵，成为数据瓶颈，限制了模型在这方面的能力发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到多模态语言模型在空间推理方面的弱点，然后考虑使用3D模拟器作为解决方案，因为模拟器可以提供完美的空间信息和可扩展的数据生成。他们设计了SIMS-V框架，系统化地研究哪些模拟数据属性能有效转移到真实世界。作者借鉴了现有的3D模拟器工作（如AI2-THOR、ProcTHOR）、多模态语言模型架构（如LLaVA）以及空间推理基准测试（如VSI-Bench），并在这些基础上进行了创新和扩展。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D模拟器的特权信息（完美的空间标注和可控制的环境）生成高质量的、空间丰富的视频训练数据，用于训练多模态语言模型的空间推理能力。整体流程包括：1) 使用AI2-THOR等模拟器生成多样化室内环境和捕获代理导航轨迹；2) 提取两种互补的元数据（观察级数据和全局空间数据）；3) 基于这些元数据程序化生成精确的空间推理问题及答案；4) 使用生成的数据训练模型并评估其在真实世界空间推理任务上的表现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 系统化的模拟数据生成框架SIMS-V；2) 通过系统消融实验识别出三种最有效的问题类别（度量测量、视角相关推理和时间跟踪）；3) 展示了高效的数据利用，仅用数千个示例就能获得强大性能；4) 实现了强大的模拟到真实世界的迁移能力。相比之前工作，SIMS-V更专注于视频中的时空推理而非静态图像，系统研究了哪些模拟数据属性能驱动真实世界迁移，训练效率更高，且在保持一般视频理解能力的同时展示了在具身推理和真实世界场景中的强泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SIMS-V通过系统化的3D模拟数据生成框架和识别的最小有效问题集，实现了高效的空间推理能力训练，使小型模型在真实世界空间推理任务上达到与大型专有模型相当的性能，同时保持强大的泛化能力。'}


### 论文摘要

Despite impressive high-level video comprehension, multimodal language models struggle with spatial reasoning across time and space. While current spatial training approaches rely on real-world video data, obtaining diverse footage with precise spatial annotations remains a bottleneck. To alleviate this bottleneck, we present SIMS-V -- a systematic data-generation framework that leverages the privileged information of 3D simulators to create spatially-rich video training data for multimodal language models. Using this framework, we investigate which properties of simulated data drive effective real-world transfer through systematic ablations of question types, mixes, and scales. We identify a minimal set of three question categories (metric measurement, perspective-dependent reasoning, and temporal tracking) that prove most effective for developing transferable spatial intelligence, outperforming comprehensive coverage despite using fewer question types. These insights enable highly efficient training: our 7B-parameter video LLM fine-tuned on just 25K simulated examples outperforms the larger 72B baseline and achieves competitive performance with proprietary models on rigorous real-world spatial reasoning benchmarks. Our approach demonstrates robust generalization, maintaining performance on general video understanding while showing substantial improvements on embodied and real-world spatial tasks.

---

## 25. NovisVQ: A Streaming Convolutional Neural Network for No-Reference Opinion-Unaware Frame Quality Assessment

**论文链接:** [http://arxiv.org/abs/2511.04628v1](http://arxiv.org/abs/2511.04628v1)

**作者:** Kylie Cancilla, Alexander Moore, Amar Saini, Carmen Carrano

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种可扩展的、基于流的无参考且无意见感知的视频质量评估模型，通过时间感知的卷积架构直接从退化视频中预测全参考指标。

### 背景

视频质量评估对计算机视觉任务至关重要，但现有方法存在局限性：全参考指标需要干净的参考视频，大多数无参考模型依赖昂贵的人类意见标签，且大多数无意见感知方法基于图像忽略了时间上下文。

### 目的

开发一种可扩展的、基于流的无参考且无意见感知的视频质量评估模型，解决现有方法的局限性。

### 方法

利用DAVIS数据集的合成退化，训练时间感知的卷积架构，直接从退化视频中预测全参考指标（LPIPS、PSNR、SSIM），推理时不需要参考视频。

### 主要发现

流式方法优于基于图像的基线模型，能够推广到不同退化类型；时间建模对现实世界视觉系统中可扩展VQA有价值；与广泛使用的BRISQUE基线相比，模型与全参考指标的相关性更高。

### 结论

时间感知的无意见感知方法在视频质量评估中有效，为解决现有方法的局限性提供了新思路。

### 翻译

视频质量评估（VQA）对计算机视觉任务至关重要，但现有方法面临主要局限性：全参考（FR）指标需要干净的参考视频，大多数无参考（NR）模型依赖于昂贵的人类意见标签进行训练。此外，大多数无意见感知的NR方法是基于图像的，忽略了视频对象检测中关键的时间上下文。在这项工作中，我们提出了一种可扩展的、基于流的无参考且无意见感知的VQA模型。我们的模型利用DAVIS数据集的合成退化，训练一个时间感知的卷积架构，直接从退化视频中预测FR指标（LPIPS、PSNR、SSIM），推理时不需要参考。我们表明，我们的流式方法通过推广到各种退化类型，优于我们自己的基于图像的基线，强调了时间建模对现实世界视觉系统中可扩展VQA的价值。此外，我们证明与广泛使用的有意见感知的图像质量评估基线BRISQUE相比，我们的模型与全参考指标的相关性更高，验证了我们时间感知、无意见感知方法的有效性。


### 论文摘要

Video quality assessment (VQA) is vital for computer vision tasks, but existing approaches face major limitations: full-reference (FR) metrics require clean reference videos, and most no-reference (NR) models depend on training on costly human opinion labels. Moreover, most opinion-unaware NR methods are image-based, ignoring temporal context critical for video object detection. In this work, we present a scalable, streaming-based VQA model that is both no-reference and opinion-unaware. Our model leverages synthetic degradations of the DAVIS dataset, training a temporal-aware convolutional architecture to predict FR metrics (LPIPS , PSNR, SSIM) directly from degraded video, without references at inference. We show that our streaming approach outperforms our own image-based baseline by generalizing across diverse degradations, underscoring the value of temporal modeling for scalable VQA in real-world vision systems. Additionally, we demonstrate that our model achieves higher correlation with full-reference metrics compared to BRISQUE, a widely-used opinion-aware image quality assessment baseline, validating the effectiveness of our temporal, opinion-unaware approach.

---

## 26. Probabilistic Textual Time Series Depression Detection

**论文链接:** [http://arxiv.org/abs/2511.04476v1](http://arxiv.org/abs/2511.04476v1)

**作者:** Fabian Schmidt, Seyedehmoniba Ravan, Vladimir Vlassov

**发布时间:** 2025-11-06

**备注:** 14 pages, 8 figures, 4 tables

### GPT解析

### 总结

本文提出了一种名为PTTSD的概率性文本时间序列抑郁检测框架，能够从语句级别的临床访谈中预测抑郁严重程度并提供不确定性估计，在多个数据集上实现了最先进的性能。

### 背景

准确且可解释的抑郁严重程度预测对临床决策支持至关重要，然而现有模型通常缺乏不确定性估计和时间建模能力。

### 目的

开发一个能够从语句级别的临床访谈中预测PHQ-8分数的框架，同时对时间维度上的不确定性进行建模。

### 方法

提出PTTSD框架，包含序列到序列和序列到一两种变体，结合双向LSTM、自注意力和残差连接，使用高斯或Student-t输出头，通过负对数似然进行训练。

### 主要发现

在E-DAIC和DAIC-WOZ数据集上评估，PTTSD在纯文本系统中达到最先进性能（E-DAIC上MAE为3.85，DAIC上为3.55），产生校准良好的预测区间；消融研究证实了注意力和概率建模的价值；与MentalBERT的比较建立了通用性；三部分校准分析和案例研究强调了不确定性感知预测的可解释性和临床相关性。

### 结论

PTTSD框架能有效预测抑郁严重程度并提供不确定性估计，这种不确定性感知的预测具有临床相关性和可解释性。

### 翻译

准确且可解释的抑郁严重程度预测对临床决策支持至关重要，但现有模型通常缺乏不确定性估计和时间建模能力。我们提出了PTTSD，一种概率性文本时间序列抑郁检测框架，能够从语句级别的临床访谈中预测PHQ-8分数，同时对时间维度上的不确定性进行建模。PTTSD包含序列到序列和序列到一两种变体，两者都结合了双向LSTM、自注意力和残差连接，并通过负对数似然训练的高斯或Student-t输出头。在E-DAIC和DAIC-WOZ上的评估显示，PTTSD在纯文本系统中达到最先进的性能（例如，E-DAIC上MAE为3.85，DAIC上为3.55），并产生校准良好的预测区间。消融研究证实了注意力和概率建模的价值，与MentalBERT的比较建立了通用性。三部分校准分析和定性案例研究进一步强调了不确定性感知预测的可解释性和临床相关性。


### 论文摘要

Accurate and interpretable predictions of depression severity are essential for clinical decision support, yet existing models often lack uncertainty estimates and temporal modeling. We propose PTTSD, a Probabilistic Textual Time Series Depression Detection framework that predicts PHQ-8 scores from utterance-level clinical interviews while modeling uncertainty over time. PTTSD includes sequence-to-sequence and sequence-to-one variants, both combining bidirectional LSTMs, self-attention, and residual connections with Gaussian or Student-t output heads trained via negative log-likelihood. Evaluated on E-DAIC and DAIC-WOZ, PTTSD achieves state-of-the-art performance among text-only systems (e.g., MAE = 3.85 on E-DAIC, 3.55 on DAIC) and produces well-calibrated prediction intervals. Ablations confirm the value of attention and probabilistic modeling, while comparisons with MentalBERT establish generality. A three-part calibration analysis and qualitative case studies further highlight the interpretability and clinical relevance of uncertainty-aware forecasting.

---

## 27. If I Could Turn Back Time: Temporal Reframing as a Historical Reasoning Task for LLMs

**论文链接:** [http://arxiv.org/abs/2511.04432v1](http://arxiv.org/abs/2511.04432v1)

**作者:** Lars Bungum, Charles Yijia Huang, Abeer Kashar

**发布时间:** 2025-11-06

**备注:** 8 pages, 1 figure, 3 tables, submitted to aconference

### GPT解析

### 总结

本研究探讨了大型语言模型进行时间推理的能力，通过使用1940年的挪威书籍中的问题测试了多种LLMs在不同语言和模型规模下的表现。

### 背景

大型语言模型在处理时间相关任务的能力尚未充分研究，特别是对于历史语境的理解。

### 目的

评估LLMs在不同语言和模型规模下进行时间推理的能力，特别关注历史语境的理解。

### 方法

使用1940年的挪威书籍中的琐事问题，让LLMs以1940年的身份回答，同时使用英语和挪威语提示。答案以句子形式呈现，评分采用LLM作为评判者，并由母语人士抽样检查。

### 主要发现

1) 英语提示比挪威语提示效果更好，这一结果出乎意料；2) 使用更大的LLMs可以提高回答质量；3) 测试了多种模型家族，包括DeepSeek-R1、Gemma3、Qwen3和Llama3.1，以及专门为挪威语设计的最大LLM。

### 结论

LLMs在时间推理任务中表现出色，但语言选择和模型规模对结果有显著影响，英语提示比挪威语提示效果更好，而更大的模型通常能提供更准确的回答。

### 翻译

在本研究中，我们尝试了大型语言模型进行时间推理的能力。我们使用了一本1940年的挪威书籍，其中包含琐事问题，我们提示LLMs以1940年的身份回答这些问题。我们还用英语和挪威语提出了问题。正确答案通常以句子形式呈现，评分方式是使用LLM作为评判者，并由母语人士进行抽样检查。使用英语提示比挪威语提示效果更好，这是一个意外的结果。相比之下，使用更大的LLMs提高了结果。我们测试了DeepSeek-R1、Gemma3、Qwen3和Llama3.1模型系列，以及专门为挪威语设计的最大可用LLM。


### 论文摘要

In this study, we experiment with the ability of LLMs to do temporal reasoning. Using a Norwegian book from 1940 containing trivia questions, we prompt the LLMs to answer the questions as if it were 1940. We also pose the questions in both English and Norwegian. Correct answers are often presented as sentences, and grading is done by means of LLM-as-judge, with sampled checks by a native speaker. Prompting in English consistently gave better results than in Norwegian, an unexpected result. In contrast, using larger LLMs improved results. We tested the DeepSeek-R1, Gemma3, Qwen3, and Llama3.1 model families, and also the largest available LLM especially crafted for Norwegian.

---

## 28. LUME-DBN: Full Bayesian Learning of DBNs from Incomplete data in Intensive Care

**论文链接:** [http://arxiv.org/abs/2511.04333v1](http://arxiv.org/abs/2511.04333v1)

**作者:** Federico Pirola, Fabio Stella, Marco Grzegorczyk

**发布时间:** 2025-11-06

**备注:** 27 pages, 8 figures, 3 tables, presented at HC@AIxIA + HYDRA 2025  Workshop located at ECAI 2025 Conference

### GPT解析

### 总结

本研究提出了一种基于Gibbs采样的新方法，用于从不完整数据中学习动态贝叶斯网络，解决了医疗数据中处理缺失数据的问题，特别是在重症监护等需要理解时间动态性的场景中。

### 背景

动态贝叶斯网络在医疗保健中日益被使用，因为它们能够对病人数据中的复杂时间关系进行建模，同时保持可解释性。然而，处理纵向临床数据中缺失数据的方法大多来自静态贝叶斯网络文献，未能充分考虑数据的时间性质，这限制了随时间量化不确定性的能力，特别是在重症监护等环境中。

### 目的

开发一个完整的贝叶斯框架，整合缺失数据处理，以便在动态贝叶斯网络中进行更可靠的推断和更准确的缺失值插补。

### 方法

提出了一种基于Gibbs采样的新方法，用于从不完整数据中学习动态贝叶斯网络。该方法将每个缺失值视为遵循高斯分布的未知参数。在每次迭代中，从未观测值的完全条件分布中采样，允许进行合理的插补和不确定性估计。

### 主要发现

在模拟数据集和来自重症监护病人的真实世界数据上评估了该方法。与标准的模型无关技术(如MICE)相比，这种贝叶斯方法展示了更好的重建准确性和收敛特性。这些结果强调了在时间模型中整合完整贝叶斯推断的临床相关性，提供了更可靠的插补和对模型行为的更深入见解。

### 结论

该方法支持更安全和更明智的临床决策，特别是在缺失数据频繁且可能产生重大影响的场景中。

### 翻译

动态贝叶斯网络(DBNs)在医疗保健中的应用日益增多，因为它们能够对病人数据中的复杂时间关系进行建模，同时保持可解释性，这是临床决策的必要特征。然而，处理纵向临床数据中缺失数据的现有方法大多来自静态贝叶斯网络文献，未能充分考虑数据的时间性质。这一局限限制了随时间量化不确定性的能力，在重症监护等环境中尤为重要，因为理解时间动态性对模型的可信度和在不同患者群体中的适用性至关重要。尽管动态贝叶斯网络具有潜力，但整合缺失数据处理的完整贝叶斯框架仍然发展不完善。在这项工作中，我们提出了一种基于Gibbs采样的新方法，用于从不完整数据中学习动态贝叶斯网络。我们的方法将每个缺失值视为遵循高斯分布的未知参数。在每次迭代中，从未观测值的完全条件分布中采样，允许进行合理的插补和不确定性估计。我们在模拟数据集和来自重症监护病人的真实世界数据上评估了我们的方法。与标准的模型无关技术(如MICE)相比，我们的贝叶斯方法展示了更好的重建准确性和收敛特性。这些结果强调了在时间模型中整合完整贝叶斯推断的临床相关性，提供了更可靠的插补和对模型行为的更深入见解。我们的方法支持更安全和更明智的临床决策，特别是在缺失数据频繁且可能产生重大影响的场景中。


### 论文摘要

Dynamic Bayesian networks (DBNs) are increasingly used in healthcare due to their ability to model complex temporal relationships in patient data while maintaining interpretability, an essential feature for clinical decision-making. However, existing approaches to handling missing data in longitudinal clinical datasets are largely derived from static Bayesian networks literature, failing to properly account for the temporal nature of the data. This gap limits the ability to quantify uncertainty over time, which is particularly critical in settings such as intensive care, where understanding the temporal dynamics is fundamental for model trustworthiness and applicability across diverse patient groups. Despite the potential of DBNs, a full Bayesian framework that integrates missing data handling remains underdeveloped. In this work, we propose a novel Gibbs sampling-based method for learning DBNs from incomplete data. Our method treats each missing value as an unknown parameter following a Gaussian distribution. At each iteration, the unobserved values are sampled from their full conditional distributions, allowing for principled imputation and uncertainty estimation. We evaluate our method on both simulated datasets and real-world intensive care data from critically ill patients. Compared to standard model-agnostic techniques such as MICE, our Bayesian approach demonstrates superior reconstruction accuracy and convergence properties. These results highlight the clinical relevance of incorporating full Bayesian inference in temporal models, providing more reliable imputations and offering deeper insight into model behavior. Our approach supports safer and more informed clinical decision-making, particularly in settings where missing data are frequent and potentially impactful.

---

## 29. Plan of Knowledge: Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering

**论文链接:** [http://arxiv.org/abs/2511.04072v1](http://arxiv.org/abs/2511.04072v1)

**作者:** Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Xiaojie Yuan

**发布时间:** 2025-11-06

**备注:** Submitted to the IEEE for possible publication

### GPT解析

### 总结

本文提出了一种名为PoK的Plan of Knowledge框架，结合结构化规划与时序知识检索，解决了大语言模型在时间推理方面的局限性，显著提高了时序知识图谱问答的检索精度和推理准确性。

### 背景

先前研究使用预训练的时序知识图谱嵌入或图神经网络注入时间知识，但未能充分理解时间约束的复杂语义信息。大语言模型虽有强大的语义理解和推理泛化能力，但时间推理能力有限，且常出现幻觉和知识缺乏问题。

### 目的

解决大语言模型在时间推理方面的局限性，提高时序知识图谱问答的检索精度和推理准确性，增强时间推理的可解释性和事实一致性。

### 方法

提出Plan of Knowledge框架(PoK)，包含对比时间检索器。Plan of Knowledge模块将复杂时间问题分解为从预定义工具中提取的子目标序列；构建带有对比检索框架的时序知识库(TKS)，使模型能选择性地从时序知识图谱中检索语义和时间对齐的事实。

### 主要发现

PoK有效提高了时间推理的可解释性和事实一致性。在四个基准时序知识图谱问答数据集上的实验表明，PoK显著提高了大语言模型的检索精度和推理准确性，最多比最先进的时序知识图谱问答方法高出56.0%的性能。

### 结论

通过结合结构化规划与时序知识检索，PoK框架成功解决了大语言模型在时间推理方面的局限性，显著提升了时序知识图谱问答的性能，为未来研究提供了新思路。

### 翻译

时序知识图谱问答(TKGQA)旨在通过利用时序知识图谱(TKGs)中的事实信息来回答时间敏感问题。虽然先前的研究采用预训练的时序知识图谱嵌入或图神经网络来注入时间知识，但它们未能完全理解时间约束的复杂语义信息。最近，大语言模型(LLMs)已显示出显著进展，受益于其强大的语义理解和推理泛化能力。然而，它们的时间推理能力仍然有限。LLMs经常出现幻觉和知识缺乏问题。为解决这些局限性，我们提出了带有对比时间检索器的Plan of Knowledge框架，命名为PoK。具体而言，所提出的Plan of Knowledge模块将复杂的时间问题分解为从预定义工具中提取的一系列子目标，作为推理探索的中间指导。同时，我们构建了一个带有对比检索框架的时序知识库(TKS)，使模型能够从时序知识图谱中选择性地检索语义和时间对齐的事实。通过结合结构化规划与时序知识检索，PoK有效提高了时间推理的可解释性和事实一致性。在四个基准时序知识图谱问答数据集上的大量实验表明，PoK显著提高了大语言模型的检索精度和推理准确性，最多比最先进的时序知识图谱问答方法高出56.0%。


### 论文摘要

Temporal Knowledge Graph Question Answering (TKGQA) aims to answer time-sensitive questions by leveraging factual information from Temporal Knowledge Graphs (TKGs). While previous studies have employed pre-trained TKG embeddings or graph neural networks to inject temporal knowledge, they fail to fully understand the complex semantic information of time constraints. Recently, Large Language Models (LLMs) have shown remarkable progress, benefiting from their strong semantic understanding and reasoning generalization capabilities. However, their temporal reasoning ability remains limited. LLMs frequently suffer from hallucination and a lack of knowledge. To address these limitations, we propose the Plan of Knowledge framework with a contrastive temporal retriever, which is named PoK. Specifically, the proposed Plan of Knowledge module decomposes a complex temporal question into a sequence of sub-objectives from the pre-defined tools, serving as intermediate guidance for reasoning exploration. In parallel, we construct a Temporal Knowledge Store (TKS) with a contrastive retrieval framework, enabling the model to selectively retrieve semantically and temporally aligned facts from TKGs. By combining structured planning with temporal knowledge retrieval, PoK effectively enhances the interpretability and factual consistency of temporal reasoning. Extensive experiments on four benchmark TKGQA datasets demonstrate that PoK significantly improves the retrieval precision and reasoning accuracy of LLMs, surpassing the performance of the state-of-the-art TKGQA methods by 56.0% at most.

---

## 30. Machine learning-driven elasticity prediction in advanced inorganic materials via convolutional neural networks

**论文链接:** [http://arxiv.org/abs/2511.04468v1](http://arxiv.org/abs/2511.04468v1)

**作者:** Yujie Liu, Zhenyu Wang, Hang Lei, Guoyu Zhang, Jiawei Xian, Zhibin Gao, Jun Sun, Haifeng Song, Xiangdong Ding

**发布时间:** 2025-11-06

**DOI:** 10.7498/aps.74.20250127

**备注:** 21 pages, 7 figures,All the data presented in this paper are openly  available at https://doi.org/10.57760/sciencedb.j00213.00104.Published in  Acta Physica Sinica

### GPT解析

### 总结

研究使用晶体图卷积神经网络(CGCNNs)成功预测了大量无机晶体材料的弹性性质，为材料设计提供了重要数据支持。

### 背景

无机晶体材料因优异的物理化学性质具有广阔应用前景，其弹性性质(剪切模量、体积模量)对预测材料的电导率、热导率和机械性能至关重要。传统实验测量成本高且效率低，而基于图神经网络的机器学习方法已成为有效替代方案。

### 目的

预测无机晶体材料的弹性性质(剪切模量和体积模量)，建立大规模材料弹性性质数据库。

### 方法

使用Matbench v0.1数据集中10987种材料的剪切模量和体积模量数据训练两个CGCNN模型；筛选材料保留带隙在0.1-3.0 eV之间的材料，排除含放射性元素的化合物；预测数据集包括Materials Project数据库中的54359个晶体结构和Merchant等人发现的26305个晶体结构。

### 主要发现

训练的CGCNN模型具有高精度(平均绝对误差小于13，决定系数R平方接近1)和良好的泛化能力；完成了80664种无机晶体剪切模量和体积模量的预测。

### 结论

这项工作丰富了现有的材料弹性数据资源，为材料设计提供了强有力的支持，所有数据已在指定平台公开可用。

### 翻译

无机晶体材料因优异的物理化学性质具有广阔的应用潜力，其弹性性质(剪切模量、体积模量)对预测材料的电导率、热导率和机械性能至关重要。传统实验测量成本高且效率低，而理论模拟和基于图神经网络的机器学习方法——特别是晶体图卷积神经网络(CGCNNs)——已成为有效替代方法，在预测材料弹性性质方面取得了显著成果。本研究使用Matbench v0.1数据集中10987种材料的剪切模量和体积模量数据训练了两个CGCNN模型，这些模型表现出高精度(平均绝对误差<13，决定系数R²接近1)和良好的泛化能力。对材料进行筛选，保留带隙在0.1-3.0 eV之间的材料，排除含放射性元素的化合物。最终的预测数据集包括两部分：Materials Project数据库中的54359个晶体结构和Merchant等人(2023 Nature 624 80)发现的26305个晶体结构。最终，本研究完成了80664种无机晶体剪切模量和体积模量的预测。这项工作丰富了现有的材料弹性数据资源，为材料设计提供了强有力的支持，所有数据已在https://doi.org/10.57760/sciencedb.j00213.00104公开可用。


### 论文摘要

Inorganic crystal materials have broad application potential due to excellent physical and chemical properties, with elastic properties (shear modulus, bulk modulus) crucial for predicting materials' electrical conductivity, thermal conductivity and mechanical properties. Traditional experimental measurement suffers from high cost and low efficiency, while theoretical simulation and graph neural network-based machine learning methods--especially crystal graph convolutional neural networks (CGCNNs)--have become effective alternatives, achieving remarkable results in predicting material elastic properties. This study trained two CGCNN models using shear modulus and bulk modulus data of 10987 materials from the Matbench v0.1 dataset, which exhibit high accuracy (mean absolute error <13, coefficient of determination R-squared close to 1) and good generalization ability. Materials were screened to retain those with band gaps between 0.1-3.0 eV and exclude radioactive element-containing compounds. The final predicted dataset comprises two parts: 54359 crystal structures from the Materials Project database and 26305 crystal structures discovered by Merchant et al. (2023 Nature 624 80). Ultimately, this study completed the prediction of shear modulus and bulk modulus for 80664 inorganic crystals. This work enriches existing material elastic data resources and provides robust support for material design, with all data openly available at https://doi.org/10.57760/sciencedb.j00213.00104.

---

## 31. Denoised Recommendation Model with Collaborative Signal Decoupling

**论文链接:** [http://arxiv.org/abs/2511.04237v1](http://arxiv.org/abs/2511.04237v1)

**作者:** Zefeng Li, Ning Yang

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种名为DRCSD的新型GNN-based CF模型，用于去噪用户-物品交互矩阵中的不稳定交互，通过协同信号解耦和阶段式去噪模块提高了推荐系统的鲁棒性和准确性。

### 背景

协同过滤(CF)算法在推荐系统中取得了显著性能，但用户-物品交互矩阵中的噪声导致推荐性能次优。现有的去噪研究大多在单个图上进行，可能导致协同信号衰减。

### 目的

解决现有去噪方法在单一图上去噪导致的协同信号衰减问题，提高推荐系统对不稳定交互的鲁棒性。

### 方法

提出DRCSD模型，包含两个核心模块：1)协同信号解耦模块，根据结构特征将信号分解为不同阶；2)阶段式去噪模块，对每个阶进行有针对性的去噪。同时修改传统GNN-based CF模型的信息聚合机制，避免跨阶信号干扰。

### 主要发现

在三个公共真实世界数据集上的实验表明，DRCSD对不稳定交互具有更强的鲁棒性，与最先进的基线模型相比，在推荐准确性指标上实现了统计学上显著的性能改进。

### 结论

DRCSD模型通过协同信号解耦和阶段式去噪有效解决了协同过滤中的噪声问题，显著提高了推荐系统的性能和鲁棒性。

### 翻译

尽管协同过滤(CF)算法在推荐系统中取得了显著性能，但由于用户-物品交互矩阵中的噪声，其推荐性能次优。许多去噪研究改进了推荐模型，但大多数现有方法在单个图上进行去噪。这可能导致协同信号衰减：移除两个节点之间的边会中断其他节点之间的路径，弱化路径依赖的协同信息。为解决这些局限性，本研究提出了一种名为DRCSD的新型基于GNN的CF模型，用于去噪不稳定交互。DRCSD包含两个核心模块：协同信号解耦模块（根据结构特征将信号分解为不同阶）和阶段式去噪模块（对每个阶进行有针对性的去噪）。此外，修改了传统基于GNN的CF模型的信息聚合机制，避免跨阶信号干扰，直到最终池化操作。在三个公共真实世界数据集上的大量实验表明，DRCSD对不稳定交互具有更强的鲁棒性，与最先进的基线模型相比，在推荐准确性指标上实现了统计学上显著的性能改进。


### 论文摘要

Although the collaborative filtering (CF) algorithm has achieved remarkable performance in recommendation systems, it suffers from suboptimal recommendation performance due to noise in the user-item interaction matrix. Numerous noise-removal studies have improved recommendation models, but most existing approaches conduct denoising on a single graph. This may cause attenuation of collaborative signals: removing edges between two nodes can interrupt paths between other nodes, weakening path-dependent collaborative information. To address these limitations, this study proposes a novel GNN-based CF model called DRCSD for denoising unstable interactions. DRCSD includes two core modules: a collaborative signal decoupling module (decomposes signals into distinct orders by structural characteristics) and an order-wise denoising module (performs targeted denoising on each order). Additionally, the information aggregation mechanism of traditional GNN-based CF models is modified to avoid cross-order signal interference until the final pooling operation. Extensive experiments on three public real-world datasets show that DRCSD has superior robustness against unstable interactions and achieves statistically significant performance improvements in recommendation accuracy metrics compared to state-of-the-art baseline models.

---

## 32. Graph Neural Networks for User Satisfaction Classification in Human-Computer Interaction

**论文链接:** [http://arxiv.org/abs/2511.04166v1](http://arxiv.org/abs/2511.04166v1)

**作者:** Rui Liu, Runsheng Zhang, Shixiao Wang

**发布时间:** 2025-11-06

### GPT解析

### 总结

本研究提出了一种基于图神经网络的用户满意度分类框架，能够有效处理复杂交互关系和多维特征，通过图结构建模、图卷积和注意力机制提高了分类性能。

### 背景

传统方法在处理用户满意度分类问题时，难以有效处理复杂的交互关系和多维特征，存在局限性。

### 目的

开发一个基于图神经网络的框架，解决传统方法在处理复杂交互关系和多维特征方面的局限性。

### 方法

将用户行为、界面元素及其潜在连接抽象为图结构，使用节点和边的联合建模捕获交互过程中的语义和依赖关系；引入图卷积和注意力机制融合局部特征和全局上下文；应用全局池化和分类层实现自动满意度分类；从结构化数据中提取深度模式，提高多源异构和动态环境下的适应性和鲁棒性。

### 主要发现

在Kaggle公开用户满意度调查数据集上的实验表明，该方法在准确性、F1分数、AUC和精确度等指标上优于多个基线模型，验证了基于图建模在满意度预测任务中的优势。

### 结论

该研究不仅丰富了用户建模的理论框架，还在优化人机交互体验方面展示了实际应用价值。

### 翻译

本研究关注用户满意度分类问题，并提出了一种基于图神经网络的框架，以解决传统方法在处理复杂交互关系和多维特征方面的局限性。用户行为、界面元素及其潜在连接被抽象为图结构，并通过节点和边的联合建模来捕获交互过程中的语义和依赖关系。引入图卷积和注意力机制来融合局部特征和全局上下文，并应用全局池化和分类层实现自动满意度分类。该方法从结构化数据中提取深度模式，提高了多源异构和动态环境下的适应性和鲁棒性。为了验证有效性，使用了Kaggle上的公开用户满意度调查数据集，并与多个基线模型在多个性能指标上进行了比较。实验表明，该方法在准确性、F1分数、AUC和精确度方面优于现有方法，展示了基于图建模在满意度预测任务中的优势。该研究不仅丰富了用户建模的理论框架，还突显了其在优化人机交互体验方面的实际价值。


### 论文摘要

This study focuses on the problem of user satisfaction classification and proposes a framework based on graph neural networks to address the limitations of traditional methods in handling complex interaction relationships and multidimensional features. User behaviors, interface elements, and their potential connections are abstracted into a graph structure, and joint modeling of nodes and edges is used to capture semantics and dependencies in the interaction process. Graph convolution and attention mechanisms are introduced to fuse local features and global context, and global pooling with a classification layer is applied to achieve automated satisfaction classification. The method extracts deep patterns from structured data and improves adaptability and robustness in multi-source heterogeneous and dynamic environments. To verify effectiveness, a public user satisfaction survey dataset from Kaggle is used, and results are compared with multiple baseline models across several performance metrics. Experiments show that the method outperforms existing approaches in accuracy, F1-Score, AUC, and Precision, demonstrating the advantage of graph-based modeling in satisfaction prediction tasks. The study not only enriches the theoretical framework of user modeling but also highlights its practical value in optimizing human-computer interaction experience.

---

## 33. ScaleDL: Towards Scalable and Efficient Runtime Prediction for Distributed Deep Learning Workloads

**论文链接:** [http://arxiv.org/abs/2511.04162v1](http://arxiv.org/abs/2511.04162v1)

**作者:** Xiaokai Wang, Shaoyuan Huang, Yuting Li, Xiaofei Wang

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了ScaleDL，一种结合非线性逐层建模和基于图神经网络的跨层交互机制的新型运行时预测框架，实现了深度神经网络运行时预测的高准确性和跨架构泛化能力，同时采用D-optimal方法降低数据收集成本。

### 背景

深度神经网络是现代AI服务的基础，支持自动驾驶、聊天机器人和推荐系统等应用。随着模型规模和复杂度增加，DNN工作负载对计算资源需求剧增，准确的运行时预测对优化开发和资源分配至关重要。传统加性计算单元模型准确性和泛化能力有限，而图增强建模虽提高性能但显著增加数据收集成本。

### 目的

开发一种在准确性、泛化能力和数据收集成本之间取得平衡的深度神经网络运行时预测方法。

### 方法

提出ScaleDL框架，结合非线性逐层建模和基于图神经网络的跨层交互机制实现准确预测和跨架构泛化，同时采用D-optimal方法减少数据收集成本。

### 主要发现

在五种流行DNN模型工作负载上的实验证明，ScaleDL相比基线模型实现了6倍更低的平均相对误差和5倍更低的均方根误差，显著提高了运行时预测的准确性和泛化能力。

### 结论

ScaleDL框架通过创新的建模方法解决了深度神经网络运行时预测中的准确性、泛化能力和数据收集成本之间的平衡问题，为DNN开发和资源优化提供了有效工具。

### 翻译

深度神经网络(DNN)构成了现代AI服务的基础，支持包括自动驾驶、聊天机器人和推荐系统在内的广泛应用。随着模型规模和复杂度的增加，DNN工作负载如训练和推理任务对分布式计算资源提出了前所未有的需求，这使得准确的运行时预测对于优化开发和资源分配至关重要。传统方法依赖于加性计算单元模型，限制了其准确性和泛化能力。相比之下，基于图增强的建模提高了性能，但显著增加了数据收集成本。因此，亟需一种能够在准确性、泛化能力和数据收集成本之间取得平衡的方法。为解决这些挑战，我们提出了ScaleDL，一种新颖的运行时预测框架，它结合了非线性逐层建模和基于图神经网络(GNN)的跨层交互机制，实现了准确的DNN运行时预测和跨不同网络架构的层次泛化能力。此外，我们采用D-optimal方法来减少数据收集成本。在五种流行DNN模型工作负载上的实验证明，ScaleDL提高了运行时预测的准确性和泛化能力，相比基线模型实现了6倍更低的平均相对误差和5倍更低的均方根误差。


### 论文摘要

Deep neural networks (DNNs) form the cornerstone of modern AI services, supporting a wide range of applications, including autonomous driving, chatbots, and recommendation systems. As models increase in size and complexity, DNN workloads like training and inference tasks impose unprecedented demands on distributed computing resources, making the accurate prediction of runtime essential for optimizing development and resource allocation. Traditional methods rely on additive computational unit models, limiting their accuracy and generalizability. In contrast, graph-enhanced modeling improves performance but significantly increases data collection costs. Therefore, there is a critical need for a method that strikes a balance between accuracy, generalizability, and the costs of data collection. To address these challenges, we propose ScaleDL, a novel runtime prediction framework that combines nonlinear layer-wise modeling with graph neural network (GNN)-based cross-layer interaction mechanism, enabling accurate DNN runtime prediction and hierarchical generalizability across different network architectures. Additionally, we employ the D-optimal method to reduce data collection costs. Experiments on the workloads of five popular DNN models prove that ScaleDL enhances runtime prediction accuracy and generalizability, achieving 6$\times$ lower MRE and 5$\times$ lower RMSE compared to baseline models.

---

## 34. KGFR: A Foundation Retriever for Generalized Knowledge Graph Question Answering

**论文链接:** [http://arxiv.org/abs/2511.04093v1](http://arxiv.org/abs/2511.04093v1)

**作者:** Yuanning Cui, Zequn Sun, Wei Hu, Zhangjie Fu

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出LLM-KGFR协作框架，结合大型语言模型与知识图谱基础检索器，解决知识密集型问题处理中的限制，提高在大型知识图谱上的可扩展性和泛化能力。

### 背景

大型语言模型在推理方面表现出色，但由于上下文和参数化知识的限制，在处理知识密集型问题时存在困难。现有方法依赖于微调的LLMs或GNN检索器，但受限于数据集特定的调优以及在大型或未见过的图上的可扩展性。

### 目的

提出一个协作框架，解决知识密集型问题处理中的限制，提高在大型知识图谱上的可扩展性和泛化能力。

### 方法

提出LLM-KGFR协作框架，其中LLM与结构化检索器KGFR协同工作。KGFR使用LLM生成的描述对关系进行编码，并根据实体在问题中的作用初始化实体，实现零样本泛化。采用非对称渐进传播处理大型图，通过节点级、边级和路径级接口形成可控的推理循环。

### 主要发现

实验证明LLM-KGFR在保持可扩展性和泛化能力的同时实现了强大的性能。

### 结论

LLM-KGFR为知识图谱增强推理提供了实用的解决方案。

### 翻译

大型语言模型在推理方面表现出色，但由于上下文和参数化知识的限制，在处理知识密集型问题时存在困难。然而，现有的依赖于微调LLMs或GNN检索器的方法受限于数据集特定的调优以及在大型或未见过的图上的可扩展性。我们提出了LLM-KGFR协作框架，其中LLM与结构化检索器知识图谱基础检索器协同工作。KGFR使用LLM生成的描述对关系进行编码，并根据实体在问题中的作用初始化实体，实现对未见过的知识图谱的零样本泛化。为有效处理大型图，它采用非对称渐进传播——一种逐步扩展方法，在选择性地限制高阶节点的同时保留信息路径。通过节点级、边级和路径级接口，LLM迭代地请求候选答案、支持事实和推理路径，形成可控的推理循环。实验证明LLM-KGFR在保持可扩展性和泛化能力的同时实现了强大的性能，为知识图谱增强推理提供了实用的解决方案。


### 论文摘要

Large language models (LLMs) excel at reasoning but struggle with knowledge-intensive questions due to limited context and parametric knowledge. However, existing methods that rely on finetuned LLMs or GNN retrievers are limited by dataset-specific tuning and scalability on large or unseen graphs. We propose the LLM-KGFR collaborative framework, where an LLM works with a structured retriever, the Knowledge Graph Foundation Retriever (KGFR). KGFR encodes relations using LLM-generated descriptions and initializes entities based on their roles in the question, enabling zero-shot generalization to unseen KGs. To handle large graphs efficiently, it employs Asymmetric Progressive Propagation (APP)- a stepwise expansion that selectively limits high-degree nodes while retaining informative paths. Through node-, edge-, and path-level interfaces, the LLM iteratively requests candidate answers, supporting facts, and reasoning paths, forming a controllable reasoning loop. Experiments demonstrate that LLM-KGFR achieves strong performance while maintaining scalability and generalization, providing a practical solution for KG-augmented reasoning.

---

## 35. GNN-MoE: Context-Aware Patch Routing using GNNs for Parameter-Efficient Domain Generalization

**论文链接:** [http://arxiv.org/abs/2511.04008v1](http://arxiv.org/abs/2511.04008v1)

**作者:** Mahmoud Soliman, Omar Abdelaziz, Ahmed Radwan, Anand, Mohamed Shehata

**发布时间:** 2025-11-06

**备注:** 6 pages, 3 figures

### GPT解析

### 总结

本文提出了一种名为GNN-MoE的新方法，通过结合图神经网络和专家混合框架，实现了在领域泛化任务中高效且鲁棒的Vision Transformer微调。

### 背景

领域泛化(DG)旨在使Vision Transformer在未见过的领域上保持鲁棒性能，但高效地预训练ViT用于DG具有挑战性，标准的微调方法成本高昂且可能损害泛化能力。

### 目的

增强参数高效微调(PEFT)在领域泛化任务上的应用性能，实现更高效、更鲁棒的模型适应。

### 方法

提出GNN-MoE方法，使用专家混合(MoE)框架结合高效的Kronecker适配器，采用基于图神经网络的路由器(GCN, GAT, SAGE)在补丁间图上操作，动态分配补丁给专门专家，利用补丁间关系更好地适应域偏移。

### 主要发现

GNN-MoE在领域泛化基准测试中实现了最先进或具有竞争力的性能，同时保持高参数效率。

### 结论

基于图的上下文路由对于实现鲁棒、轻量级的领域泛化具有实用价值。

### 翻译

领域泛化(DG)寻求在未见过的领域上实现鲁棒的Vision Transformer(ViT)性能。高效地预训练ViT用于DG具有挑战性；标准的微调成本高昂且可能损害泛化能力。我们提出了GNN-MoE，使用高效的Kronecker适配器，通过专家混合(MoE)框架增强了参数高效微调(PEFT)在DG上的应用。与基于token的路由不同，一种新颖的图神经网络(GNN)路由器(GCN, GAT, SAGE)在补丁间图上操作，动态地将补丁分配给专门的专家。这种上下文感知的GNN路由利用补丁间关系，更好地适应域偏移。GNN-MoE以高参数效率实现了最先进或具有竞争力的DG基准性能，突显了基于图的上下文路由对于鲁棒、轻量级DG的实用性。


### 论文摘要

Domain generalization (DG) seeks robust Vision Transformer (ViT) performance on unseen domains. Efficiently adapting pretrained ViTs for DG is challenging; standard fine-tuning is costly and can impair generalization. We propose GNN-MoE, enhancing Parameter-Efficient Fine-Tuning (PEFT) for DG with a Mixture-of-Experts (MoE) framework using efficient Kronecker adapters. Instead of token-based routing, a novel Graph Neural Network (GNN) router (GCN, GAT, SAGE) operates on inter-patch graphs to dynamically assign patches to specialized experts. This context-aware GNN routing leverages inter-patch relationships for better adaptation to domain shifts. GNN-MoE achieves state-of-the-art or competitive DG benchmark performance with high parameter efficiency, highlighting the utility of graph-based contextual routing for robust, lightweight DG.

---

## 36. Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.03824v1](http://arxiv.org/abs/2511.03824v1)

**作者:** Ryien Hosseini, Filippo Simini, Venkatram Vishwanath, Rebecca Willett, Henry Hoffmann

**发布时间:** 2025-11-05

**备注:** To appear at NeurIPS 2025

### GPT解析

### 总结

本文提出了一种通过注入随机化的全局节点特征嵌入（称为'草图随机特征'）来增强图神经网络性能的方法，有效解决了GNN面临的三个关键挑战。

### 背景

图神经网络通过迭代聚合局部邻域信息在图结构数据上进行学习。这种局部消息传递范式虽然提供了强大的归纳偏置并利用了图稀疏性，但也带来了三个关键挑战。

### 目的

解决图神经网络面临的三个关键挑战：(i) 长程信息的过度压缩，(ii) 节点表示的过度平滑，(iii) 有限的表达能力。

### 方法

将随机化的节点特征全局嵌入（称为'草图随机特征'）注入到标准GNN中，使它们能够有效地捕获长程依赖关系。这些嵌入是唯一的、距离敏感的且与拓扑无关的。

### 主要发现

通过分析和实验证明，当将这些嵌入注入到GNN中时，可以减轻上述提到的局限性。在真实世界的图学习任务上的实验结果表明，这种策略比基线GNN能持续提高性能。

### 结论

该策略既可以作为独立解决方案，也可以作为现有技术（如图位置编码）的补充增强。源代码已在GitHub上公开。

### 翻译

图神经网络通过迭代聚合图结构数据上的局部邻域信息进行学习。虽然这种局部消息传递范式提供了强大的归纳偏置并利用了图稀疏性，但也带来了三个关键挑战：(i) 长程信息的过度压缩，(ii) 节点表示的过度平滑，(iii) 有限的表达能力。在这项工作中，我们将节点特征的随机化全局嵌入（我们称之为'草图随机特征'）注入到标准GNN中，使它们能够有效地捕获长程依赖关系。这些嵌入是唯一的、距离敏感的且与拓扑无关的——通过分析和实验，我们证明了当这些嵌入注入到GNN中时，可以减轻上述提到的局限性。在真实世界图学习任务上的实验结果证实，这种策略比基线GNN能持续提高性能，既可作为独立解决方案，也可作为现有技术（如图位置编码）的补充增强。我们的源代码可在https://github.com/ryienh/sketched-random-features获取。


### 论文摘要

Graph Neural Networks learn on graph-structured data by iteratively aggregating local neighborhood information. While this local message passing paradigm imparts a powerful inductive bias and exploits graph sparsity, it also yields three key challenges: (i) oversquashing of long-range information, (ii) oversmoothing of node representations, and (iii) limited expressive power. In this work we inject randomized global embeddings of node features, which we term \textit{Sketched Random Features}, into standard GNNs, enabling them to efficiently capture long-range dependencies. The embeddings are unique, distance-sensitive, and topology-agnostic -- properties which we analytically and empirically show alleviate the aforementioned limitations when injected into GNNs. Experimental results on real-world graph learning tasks confirm that this strategy consistently improves performance over baseline GNNs, offering both a standalone solution and a complementary enhancement to existing techniques such as graph positional encodings. Our source code is available at \href{https://github.com/ryienh/sketched-random-features}{https://github.com/ryienh/sketched-random-features}.

---

## 37. Causal Graph Neural Networks for Healthcare

**论文链接:** [http://arxiv.org/abs/2511.02531v2](http://arxiv.org/abs/2511.02531v2)

**作者:** Munib Mesinovic, Max Buhlan, Tingting Zhu

**发布时间:** 2025-11-04

### GPT解析

### 总结

医疗人工智能系统在跨机构部署时经常失败，表现为性能下降和历史数据中歧视性模式的持续存在。这种脆弱性部分源于系统学习的是统计关联而非因果机制。因果图神经网络通过结合生物医学数据的图表示和因果推理原则，学习不变机制而非虚假相关性，以解决分布偏移、歧视性和不可解释性三重危机。

### 背景

医疗人工智能系统在跨机构部署时经常失败，有记录显示性能下降且延续了历史数据中的歧视性模式。这种脆弱性部分源于系统学习的是统计关联而非因果机制。

### 目的

通过因果图神经网络解决医疗AI面临的分布偏移、歧视性和不可解释性三重危机，学习不变机制而非虚假相关性。

### 方法

结合生物医学数据的图表示和因果推理原则，采用结构因果模型、解纠缠因果表征学习，以及图上的干预预测和反事实推理技术。

### 主要发现

因果图神经网络在多个医疗领域展示了临床价值：精神疾病诊断通过脑网络分析、癌症亚型分析通过多组学因果整合、连续生理监测与机制解释、纠正处方偏差的药物推荐。

### 结论

这些进展为患者特异性因果数字孪生奠定了基础，可实现计算机内临床实验，并整合大语言模型进行假设生成和因果图神经网络进行机制验证。仍存在重大障碍，包括计算需求限制实时部署、验证挑战需要多模态证据三角测量，以及因果清洗风险。建议提出区分因果启发架构和因果验证发现的分层框架，并确定关键研究重点，提出因果而非纯关联性主张。

### 翻译

医疗人工智能系统在跨机构部署时经常失败，有记录显示性能下降且延续了历史数据中的歧视性模式。这种脆弱性部分源于学习的是统计关联而非因果机制。因果图神经网络通过结合生物医学数据的图表示和因果推理原则，学习不变机制而非虚假相关性，以解决分布偏移、歧视性和不可解释性三重危机。本综述审视了结构因果模型、解纠缠因果表征学习，以及图上的干预预测和反事实推理技术等方法论基础。我们分析了在精神疾病诊断通过脑网络分析、癌症亚型分析通过多组学因果整合、连续生理监测与机制解释、纠正处方偏差的药物推荐等领域展示临床价值的应用。这些进展为患者特异性因果数字孪生奠定了基础，可实现计算机内临床实验，并整合大语言模型进行假设生成和因果图神经网络进行机制验证。仍存在重大障碍，包括计算需求限制实时部署、验证挑战需要多模态证据三角测量，以及因果清洗风险（方法使用因果术语但缺乏严格证据支持）。我们提出区分因果启发架构和因果验证发现的分层框架，并确定提出因果而非纯关联性主张的关键研究重点。


### 论文摘要

Healthcare artificial intelligence systems routinely fail when deployed across institutions, with documented performance drops and perpetuation of discriminatory patterns embedded in historical data. This brittleness stems, in part, from learning statistical associations rather than causal mechanisms. Causal graph neural networks address this triple crisis of distribution shift, discrimination, and inscrutability by combining graph-based representations of biomedical data with causal inference principles to learn invariant mechanisms rather than spurious correlations. This Review examines methodological foundations spanning structural causal models, disentangled causal representation learning, and techniques for interventional prediction and counterfactual reasoning on graphs. We analyse applications demonstrating clinical value across psychiatric diagnosis through brain network analysis, cancer subtyping via multi-omics causal integration, continuous physiological monitoring with mechanistic interpretation, and drug recommendation correcting prescription bias. These advances establish foundations for patient-specific Causal Digital Twins, enabling in silico clinical experimentation, with integration of large language models for hypothesis generation and causal graph neural networks for mechanistic validation. Substantial barriers remain, including computational requirements precluding real-time deployment, validation challenges demanding multi-modal evidence triangulation beyond cross-validation, and risks of causal-washing where methods employ causal terminology without rigorous evidentiary support. We propose tiered frameworks distinguishing causally-inspired architectures from causally-validated discoveries and identify critical research priorities making causal rather than purely associational claims.

---

## 38. Evaluating the Impact of Weather-Induced Sensor Occlusion on BEVFusion for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.04347v1](http://arxiv.org/abs/2511.04347v1)

**作者:** Sanjay Kumar, Tim Brophy, Eoin Martino Grua, Ganesh Sistu, Valentina Donzella, Ciaran Eising

**发布时间:** 2025-11-06

### GPT解析

### 总结

该研究探讨了传感器遮挡对基于鸟瞰图(BEV)的3D物体检测系统性能的影响，发现摄像头和激光雷达在不同遮挡条件下的表现差异显著，且融合系统更依赖激光雷达数据。

### 背景

准确的3D物体检测对自动驾驶车辆在复杂环境中安全导航至关重要。鸟瞰图(BEV)表示方法通过将多传感器数据投影到俯视空间格式，已成为强大的鲁棒感知方法。然而，由环境条件引起的传感器遮挡对3D检测精度的影响尚未得到充分探索。

### 目的

研究遮挡对摄像头和激光雷达(LiDAR)输出在3D检测任务中的影响，评估不同传感器在不利环境条件下的性能表现。

### 方法

使用BEVFusion架构进行研究，在nuScenes数据集上评估，并采用平均精度均值(mAP)和nuScenes检测分数(NDS)作为性能测量指标。

### 主要发现

中等程度的摄像头遮挡导致仅基于摄像头的检测mAP下降41.3%；激光雷达仅在严重遮挡下性能急剧下降，mAP下降47.3%，且对远距离检测影响严重；在融合系统中，遮挡摄像头导致轻微4.1%的性能下降，而遮挡激光雷达则导致26.8%的较大下降，表明模型更依赖激光雷达进行3D物体检测。

### 结论

研究结果强调了未来研究需要关注感知遮挡的评估方法，并开发能在部分传感器失效或降级时保持检测精度的改进传感器融合技术。

### 翻译

准确的3D物体检测对自动驾驶车辆在复杂真实世界环境中安全导航至关重要。鸟瞰图(BEV)表示方法通过将多传感器数据投影到俯视空间格式，已成为一种强大的鲁棒感知方法。尽管基于BEV的融合架构通过多模态集成展示了强大的性能，但由雾、霾或物理障碍等环境条件引起的传感器遮挡对3D检测精度的影响仍未得到充分探索。在这项工作中，我们使用BEVFusion架构研究了遮挡对摄像头和激光雷达(LiDAR)输出的影响，并在nuScenes数据集上进行了评估。检测性能使用平均精度均值(mAP)和nuScenes检测分数(NDS)进行测量。我们的结果表明，当仅基于摄像头检测时，中等程度的摄像头遮挡导致mAP下降41.3%(从35.6%降至20.9%)。另一方面，激光雷达仅在严重遮挡下性能急剧下降，mAP下降47.3%(从64.7%降至34.1%)，对远距离检测有严重影响。在融合设置中，效果取决于哪个传感器被遮挡：遮挡摄像头导致轻微4.1%的下降(从68.5%降至65.7%)，而遮挡激光雷达导致较大26.8%的下降(降至50.1%)，揭示了模型在3D物体检测任务中更依赖激光雷达。我们的研究结果强调了未来研究需要关注感知遮挡的评估方法，以及改进传感器融合技术，以便在部分传感器失效或因不利环境条件而降级时保持检测精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文研究天气引起的传感器遮挡（如雾、霾或物理障碍物）对BEVFusion在3D物体检测中的影响。这个问题很重要，因为自动驾驶车辆需要在复杂环境中安全导航，准确的3D物体检测至关重要。虽然基于鸟瞰图(BEV)的传感器融合方法表现强大，但大多数研究都在理想条件下评估，而实际环境中传感器经常因天气因素退化，可能导致检测性能大幅下降，带来安全风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到先前研究只部分探索了遮挡问题：Xie等人研究了相机遮挡但未考虑激光雷达退化，Brophy等人分析了雨对相机检测的影响但未检查多模态融合。作者决定系统性地研究BEVFusion架构中两种传感器的遮挡影响。他们借鉴了Woodscape数据集的污染掩码来模拟相机遮挡，并采用随机点云丢弃来模拟激光雷达退化，这种方法参考了Chan等人的工作。作者选择BEVFusion是因为它处理不同传感器模态独立，适合隔离遮挡效应。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是评估传感器遮挡对BEVFusion架构在3D物体检测中的影响，模拟真实世界中的传感器退化情况。实现流程包括：1)使用nuScenes数据集；2)模拟相机遮挡（应用Woodscape污染掩码和高斯滤波）；3)模拟激光雷达遮挡（随机丢弃点云）；4)在未修改的BEVFusion上评估性能；5)使用mAP和NDS指标测量检测性能；6)分析不同遮挡程度对单独传感器和融合设置的影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统研究BEV融合模型中传感器特定遮挡的影响；2)结合相机和激光雷达的遮挡模拟方法；3)定量分析不同遮挡程度对检测性能的影响；4)揭示模型对激光雷达的较强依赖性。相比之前工作，本文不仅研究相机或激光雷达单独的遮挡影响，还研究融合情况下的表现，提供了更全面的遮挡影响分析，并展示了融合模型如何在不同传感器退化条件下保持性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统评估传感器遮挡对BEVFusion 3D物体检测的影响，揭示了模型对激光雷达的较强依赖性，并为开发能在恶劣环境条件下保持检测准确性的鲁棒传感器融合技术提供了重要指导。'}


### 论文摘要

Accurate 3D object detection is essential for automated vehicles to navigate safely in complex real-world environments. Bird's Eye View (BEV) representations, which project multi-sensor data into a top-down spatial format, have emerged as a powerful approach for robust perception. Although BEV-based fusion architectures have demonstrated strong performance through multimodal integration, the effects of sensor occlusions, caused by environmental conditions such as fog, haze, or physical obstructions, on 3D detection accuracy remain underexplored. In this work, we investigate the impact of occlusions on both camera and Light Detection and Ranging (LiDAR) outputs using the BEVFusion architecture, evaluated on the nuScenes dataset. Detection performance is measured using mean Average Precision (mAP) and the nuScenes Detection Score (NDS). Our results show that moderate camera occlusions lead to a 41.3% drop in mAP (from 35.6% to 20.9%) when detection is based only on the camera. On the other hand, LiDAR sharply drops in performance only under heavy occlusion, with mAP falling by 47.3% (from 64.7% to 34.1%), with a severe impact on long-range detection. In fused settings, the effect depends on which sensor is occluded: occluding the camera leads to a minor 4.1% drop (from 68.5% to 65.7%), while occluding LiDAR results in a larger 26.8% drop (to 50.1%), revealing the model's stronger reliance on LiDAR for the task of 3D object detection. Our results highlight the need for future research into occlusion-aware evaluation methods and improved sensor fusion techniques that can maintain detection accuracy in the presence of partial sensor failure or degradation due to adverse environmental conditions.

---

## 39. DORAEMON: A Unified Library for Visual Object Modeling and Representation Learning at Scale

**论文链接:** [http://arxiv.org/abs/2511.04394v1](http://arxiv.org/abs/2511.04394v1)

**作者:** Ke Du, Yimin Peng, Chao Gao, Fan Zhou, Siqiao Xue

**发布时间:** 2025-11-06

**备注:** code: https://github.com/wuji3/DORAEMON

### GPT解析

### 总结

DORAEMON是一个开源的PyTorch库，统一了不同尺度下的视觉对象建模和表示学习。

### 背景

视觉识别和表示学习领域需要整合多种技术、模型和数据集，以加速研究进展并应用于实际场景。

### 目的

提供一个可扩展的基础平台，用于快速实验视觉识别和表示学习，促进研究成果向实际应用的高效转化。

### 方法

创建单一的YAML驱动工作流程涵盖分类、检索和度量学习；提供1000多个预训练骨干网络通过timm兼容接口访问；包含模块化损失函数、数据增强和分布式训练工具；支持一键导出到ONNX或HuggingFace格式。

### 主要发现

可复现的配方在ImageNet-1K、MS-Celeb-1M和Stanford在线产品上匹配或超过了参考结果；通过整合数据集、模型和训练技术，提供了可扩展的基础平台。

### 结论

DORAEMON通过将数据集、模型和训练技术整合到一个平台，为视觉识别和表示学习的快速实验提供了可扩展的基础，实现了研究成果向实际应用的高效转化。

### 翻译

DORAEMON是一个开源的PyTorch库，统一了不同尺度下的视觉对象建模和表示学习。单一的YAML驱动工作流程涵盖分类、检索和度量学习；通过timm兼容接口提供了1000多个预训练骨干网络，以及模块化损失函数、数据增强和分布式训练工具。可复现的配方在ImageNet-1K、MS-Celeb-1M和Stanford在线产品上匹配或超过了参考结果，而一键导出到ONNX或HuggingFace则连接了研究和部署。通过将数据集、模型和训练技术整合到一个平台，DORAEMON为视觉识别和表示学习的快速实验提供了可扩展的基础，实现了研究成果向实际应用的高效转化。代码仓库可在https://github.com/wuji3/DORAEMON获取。


### 论文摘要

DORAEMON is an open-source PyTorch library that unifies visual object modeling and representation learning across diverse scales. A single YAML-driven workflow covers classification, retrieval and metric learning; more than 1000 pretrained backbones are exposed through a timm-compatible interface, together with modular losses, augmentations and distributed-training utilities. Reproducible recipes match or exceed reference results on ImageNet-1K, MS-Celeb-1M and Stanford online products, while one-command export to ONNX or HuggingFace bridges research and deployment. By consolidating datasets, models, and training techniques into one platform, DORAEMON offers a scalable foundation for rapid experimentation in visual recognition and representation learning, enabling efficient transfer of research advances to real-world applications. The repository is available at https://github.com/wuji3/DORAEMON.

---

## 40. DINOv2 Driven Gait Representation Learning for Video-Based Visible-Infrared Person Re-identification

**论文链接:** [http://arxiv.org/abs/2511.04281v1](http://arxiv.org/abs/2511.04281v1)

**作者:** Yujie Yang, Shuang Li, Jun Ye, Neng Dong, Fan Li, Huafeng Li

**发布时间:** 2025-11-06

### GPT解析

### 总结

该研究提出了一种基于DINOv2的步态表示学习框架(DinoGRL)，用于解决可见光-红外视频行人重识别问题，通过结合步态特征和外观特征，实现了跨模态视频匹配的显著改进。

### 背景

现有视频可见光-红外行人重识别方法主要利用模态不变视觉特征，但忽略了富含时间动态信息的步态特征，这限制了它们对跨模态视频匹配中时空一致性的建模能力。

### 目的

开发一种能够有效利用步态特征的跨模态视频行人重识别方法，提高可见光和红外模态下同一行人的检索准确率。

### 方法

提出DINOv2-Driven Gait Representation Learning (DinoGRL)框架，包含语义感知的轮廓和步态学习(SASGL)模型和渐进式双向多粒度增强(PBMGE)模块，利用DINOv2的视觉先验学习步态特征，并通过多粒度双向交互优化特征表示。

### 主要发现

通过在HITSZ-VCM和BUPT数据集上的实验证明，该方法显著优于现有的最先进方法，有效结合了步态特征和外观特征的互补优势。

### 结论

通过整合步态特征与外观特征，并利用DINOv2的丰富视觉先验和多粒度增强策略，该方法成功提高了跨模态视频行人重识别的性能。

### 翻译

基于视频的可见光-红外行人重识别(VVI-ReID)旨在从视频序列中跨可见光和红外模态检索同一行人。现有方法倾向于利用模态不变的视觉特征，但 largely 忽略了步态特征，步态特征不仅是模态不变的，还富含时间动态信息，从而限制了它们对跨模态视频匹配中必不可少的时空一致性建模能力。为解决这些挑战，我们提出了DINOv2驱动的步态表示学习(DinoGRL)框架，利用DINOv2的丰富视觉先验学习与外观线索互补的步态特征，促进跨模态检索的鲁棒序列级表示。具体而言，我们引入了语义感知的轮廓和步态学习(SASGL)模型，利用DINOv2的通用语义先验生成和增强轮廓表示，并与ReID目标联合优化，实现语义丰富且任务自适应的步态特征学习。此外，我们开发了渐进式双向多粒度增强(PBMGE)模块，通过在步态和外观流之间实现多空间粒度的双向交互，逐步细化特征表示，充分利用它们的互补性，用丰富的局部细节增强全局表示，产生高度判别性的特征。在HITSZ-VCM和BUPT数据集上的广泛实验证明了我们方法的优越性，显著优于现有的最先进方法。


### 论文摘要

Video-based Visible-Infrared person re-identification (VVI-ReID) aims to retrieve the same pedestrian across visible and infrared modalities from video sequences. Existing methods tend to exploit modality-invariant visual features but largely overlook gait features, which are not only modality-invariant but also rich in temporal dynamics, thus limiting their ability to model the spatiotemporal consistency essential for cross-modal video matching. To address these challenges, we propose a DINOv2-Driven Gait Representation Learning (DinoGRL) framework that leverages the rich visual priors of DINOv2 to learn gait features complementary to appearance cues, facilitating robust sequence-level representations for cross-modal retrieval. Specifically, we introduce a Semantic-Aware Silhouette and Gait Learning (SASGL) model, which generates and enhances silhouette representations with general-purpose semantic priors from DINOv2 and jointly optimizes them with the ReID objective to achieve semantically enriched and task-adaptive gait feature learning. Furthermore, we develop a Progressive Bidirectional Multi-Granularity Enhancement (PBMGE) module, which progressively refines feature representations by enabling bidirectional interactions between gait and appearance streams across multiple spatial granularities, fully leveraging their complementarity to enhance global representations with rich local details and produce highly discriminative features. Extensive experiments on HITSZ-VCM and BUPT datasets demonstrate the superiority of our approach, significantly outperforming existing state-of-the-art methods.

---

