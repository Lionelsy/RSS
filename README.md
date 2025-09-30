# 今日论文推荐 - 2025-09-30

共 145 篇论文

---

## 1. Towards a foundation model for astrophysical source detection: An End-to-End Gamma-Ray Data Analysis Pipeline Using Deep Learning

**论文链接:** [http://arxiv.org/abs/2509.25128v1](http://arxiv.org/abs/2509.25128v1)

**作者:** Judit Pérez-Romero, Saptashwa Bhattacharyya, Sascha Caron, Dmitry Malyshev, Rodney Nicolas, Giacomo Principe, Zoja Rokavec, Roberto Ruiz de Austri, Danijel Skočaj, Fiorenzo Stoppa, Domen Tabernik, Gabrijela Zaharijas

**发布时间:** 2025-09-29

**备注:** 6 pages, 3 figures, presented at EuCAIFCon 2025

### GPT解析

### 总结

研究人员提出了一种基于深度学习的伽马射线源检测、定位和表征流程，扩展了AutoSourceID方法，并成功应用于CTAO模拟数据。

### 背景

随着伽马射线数据量的不断增加，需要能够处理大规模数据集并提供稳健源检测的新分析方法。

### 目的

开发一个基于深度学习的伽马射线源检测、定位和表征流程。

### 方法

扩展了AutoSourceID (ASID)方法，该方法最初使用Fermi-LAT模拟数据和光学数据(MeerLICHT)进行测试，现在扩展到切伦科夫望远镜阵列天文台(CTAO)模拟数据。

### 主要发现

该端到端流程展示了未来应用于其他调查的通用框架，并可能作为天体物理源检测基础模型的构建模块。

### 结论

基于深度学习的检测方法在伽马射线源分析中具有广阔的应用前景，可作为未来天体物理源检测的基础模型。

### 翻译

日益增长的伽马射线数据量需要能够处理大规模数据集并提供稳健源检测的新分析方法。我们提出了一种基于深度学习的伽马射线源检测、定位和表征流程。我们将AutoSourceID (ASID)方法扩展到切伦科夫望远镜阵列天文台(CTAO)模拟数据，该方法最初使用Fermi-LAT模拟数据和光学数据(MeerLICHT)进行测试。这个端到端流程展示了未来应用于其他调查的通用框架，并可能作为天体物理源检测基础模型的构建模块。


### 论文摘要

The increasing volume of gamma-ray data demands new analysis approaches that can handle large-scale datasets while providing robustness for source detection. We present a Deep Learning (DL) based pipeline for detection, localization, and characterization of gamma-ray sources. We extend our AutoSourceID (ASID) method, initially tested with \textit{Fermi}-LAT simulated data and optical data (MeerLICHT), to Cherenkov Telescope Array Observatory (CTAO) simulated data. This end-to-end pipeline demonstrates a versatile framework for future application to other surveys and potentially serves as a building block for a foundational model for astrophysical source detection.

---

## 2. Towards Reliable Generation of Executable Workflows by Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.25117v1](http://arxiv.org/abs/2509.25117v1)

**作者:** Sogol Masoumzadeh, Keheliya Gallaba, Dayi Lin, Ahmed E. Hassan

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一种利用静态分析反馈使基础模型(FMs)能够检测和修复其生成的领域特定语言(DSL)工作流中缺陷的框架，为可靠和自动化地生成可执行工作流提供了重要一步。

### 背景

基础模型在理解和执行复杂自然语言任务方面取得显著进展，但手动将任务分解为工作流需要大量专业知识和努力。虽然FMs可帮助生成DSL工作流，但准确性和可靠性仍是挑战。

### 目的

开发一个框架，使FMs能够检测和修复其生成的DSL工作流中的缺陷，提高工作流的可靠性和自动化生成水平。

### 方法

引入静态分析反馈机制，开发名为Timon的静态分析工具，并通过整合Timon的反馈引导基于FM的工具Pumbaa修复检测到的缺陷。

### 主要发现

1) 首次提出FM生成DSL工作流缺陷的分类法，分为18种类型；2) 87.27%的FM生成DSL工作流实例至少包含一个缺陷；3) 其中9种缺陷类型可通过静态分析有效识别；4) 使用Timon反馈可指导修复检测到的缺陷。

### 结论

通过系统检测和修复缺陷，该研究为从自然语言需求可靠和自动化地生成可执行工作流提供了关键步骤。

### 翻译

最近基础模型(FMs)的进展在理解复杂自然语言以执行复杂任务方面展示了显著进步。成功执行这些任务通常需要编排对FMs的调用以及其他软件组件。然而，手动将任务分解为连贯的、逻辑上聚合的步骤序列（通常称为工作流）需要大量努力和专业知识。虽然FMs可以帮助生成用领域特定语言(DSLs)指定的工作流，但在这个过程中实现准确性和可靠性仍然是一个挑战。这项工作引入了一个框架，利用静态分析反馈使FMs能够检测和修复它们生成的基于DSL的工作流中的缺陷。我们首先提出了FM生成的DSL工作流缺陷实例的首个分类法，将其分为18种不同类型。此外，我们观察到FM生成的DSL工作流中缺陷普遍存在，所研究实例的87.27%至少包含一个缺陷。这反过来强调了实践中问题的严重性，并突显了实施缓解策略的必要性。在此基础上，我们展示了这些缺陷中的9种类型可以通过工作流的静态分析有效识别。为此，我们开发了Timon，这是第一个专门为FM生成的DSL工作流设计的静态分析工具。最后，我们展示通过整合Timon的反馈，可以引导基于FM的工具Pumbaa修复检测到的缺陷实例。通过系统检测和修复缺陷，我们的工作为从自然语言需求可靠和自动化地生成可执行工作流提供了关键一步。


### 论文摘要

Recent advancements in Foundation Models (FMs) have demonstrated significant progress in comprehending complex natural language to perform intricate tasks. Successfully executing these tasks often requires orchestrating calls to FMs alongside other software components. However, manually decomposing a task into a coherent sequence of smaller, logically aggregated steps, commonly referred to as workflows, demands considerable effort and specialized domain knowledge. While FMs can assist in generating such workflows specified in domain-specific languages (DSLs), achieving accuracy and reliability in this process remains a challenge.   This work introduces a framework that leverages static analysis feedback to enable FMs to detect and repair defects in the DSL-based workflows they generate. We begin by presenting the first-ever taxonomy of incidences of defects in FM-generated DSL workflows, categorizing them into 18 distinct types. Furthermore, we observe a high prevalence of defects across FM-generated DSL workflows, with 87.27% of the studied instances containing at least one defect. This, in turn, emphasizes the magnitude of the problem in practice and underscores the necessity for implementing mitigation strategies. Following this, we demonstrate that nine types of these defects can be effectively identified through static analysis of the workflows. For this purpose, we develop Timon, the first-of-its-kind static analyzer specifically designed for FM-generated DSL workflows. Finally, we show that by incorporating feedback from Timon, we can guide Pumbaa, an FM-based tool, to repair the detected defect incidences. By systematically detecting and repairing defects, our work provides a crucial step towards the reliable and automated generation of executable workflows from natural language requirements.

---

## 3. Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks

**论文链接:** [http://arxiv.org/abs/2509.25095v1](http://arxiv.org/abs/2509.25095v1)

**作者:** M A Al-Masud, Juan Miguel Lopez Alcaraz, Nils Strodthoff

**发布时间:** 2025-09-29

**备注:** 26 pages, 3 figures source code under  https://github.com/AI4HealthUOL/ecg-fm-benchmarking

### GPT解析

### 总结

本研究对八种心电图基础模型在26个临床相关任务上的性能进行了基准测试，比较了它们与监督基线的表现，并分析了不同数据集规模下的扩展行为。研究发现基础模型在成人心电图解释方面表现优异，但在心脏结构、结果预测和患者特征方面仍有差距。值得注意的是，ECG-CPC尽管模型小且计算资源消耗少，但在多个任务中表现优异。

### 背景

十二导联心电图是一种长期存在的诊断工具，但心电图解释的机器学习研究仍然支离破碎，通常局限于狭窄的任务或数据集。基础模型有望实现更广泛的适应性，但它们在不同心电图任务上的泛化能力尚未得到充分理解。

### 目的

评估心电图基础模型在不同临床任务上的性能，了解它们与监督基线的比较，以及分析不同数据集规模下的扩展行为。

### 方法

研究使用包含1650个回归和分类目标的12个公共数据集，对八种心电图基础模型在26个临床相关任务上进行基准测试。模型在微调和冻结设置下进行了评估，并针对不同数据集规模进行了扩展分析。

### 主要发现

1. 不同领域的性能存在异质性；2. 在成人心电图解释领域，三种基础模型始终优于强大的监督基线；3. ECG-CPC在大多数基础模型未能超越监督学习的其他类别中占据主导地位；4. 基础模型显示出与数据集规模相关的不同扩展行为，这对小规模临床应用至关重要；5. 尽管ECG-CPC小几个数量级且消耗的计算资源极少，但其性能强大。

### 结论

尽管基础模型在成人心电图分析方面显示出前景，但在心脏结构、结果预测和患者特征方面仍存在显著差距。ECG-CPC的突显性能表明推进心电图基础模型存在未开发的机会。

### 翻译

十二导联心电图是一种长期存在的诊断工具。然而，心电图解释的机器学习仍然支离破碎，通常局限于狭窄的任务或数据集。基础模型有望实现更广泛的适应性，但它们在不同心电图任务上的泛化能力尚未得到充分理解。我们在使用包含1650个回归和分类目标的12个公共数据集的26个临床相关任务上对八种心电图基础模型进行了基准测试。模型在微调和冻结设置下进行了评估，并针对不同数据集规模进行了扩展分析。结果显示，不同领域的性能存在异质性：在研究最广泛的领域——成人心电图解释中，三种基础模型始终优于强大的监督基线。相比之下，在HEEDB上预训练的紧凑型结构状态空间模型ECG-CPC在大多数基础模型未能超越监督学习的其他类别中占据主导地位。基础模型还显示出与数据集规模相关的不同扩展行为，这对小规模临床应用至关重要。总体而言，尽管基础模型在成人心电图分析方面显示出前景，但在心脏结构、结果预测和患者特征方面仍存在显著差距。值得注意的是，尽管ECG-CPC小几个数量级且消耗的计算资源极少，但其强大的性能突显了推进心电图基础模型的未开发机会。


### 论文摘要

The 12-lead electrocardiogram (ECG) is a long-standing diagnostic tool. Yet machine learning for ECG interpretation remains fragmented, often limited to narrow tasks or datasets. Foundation models promise broader adaptability, but their generalization across diverse ECG tasks is not well understood. We benchmarked eight ECG foundation models on 26 clinically relevant tasks using 12 public datasets comprising 1,650 regression and classification targets. Models were evaluated under fine-tuning and frozen settings, with scaling analyses across dataset sizes. Results show heterogeneous performance across domains: in the most widely studied domain, adult ECG interpretation, three foundation models consistently outperformed strong supervised baselines. In contrast, ECG-CPC, a compact structured state-space model pretrained on HEEDB, dominated other categories where most foundation models failed to surpass supervised learning. Foundation models also displayed distinct scaling behaviors with dataset size, which are critical for small-scale clinical applications. Overall, while foundation models show promise for adult ECG analysis, substantial gaps remain in cardiac structure, outcome prediction, and patient characterization. Notably, ECG-CPC's strong performance despite being orders of magnitude smaller and consuming minimal computational resources highlights untapped opportunities for advancing ECG foundation models.

---

## 4. DAM: Dual Active Learning with Multimodal Foundation Model for Source-Free Domain Adaptation

**论文链接:** [http://arxiv.org/abs/2509.24896v1](http://arxiv.org/abs/2509.24896v1)

**作者:** Xi Chen, Hongxun Yao, Zhaopan Xu, Kui Jiang

**发布时间:** 2025-09-29

**备注:** 5 pages

### GPT解析

### 总结

提出DAM（双主动学习与多模态）基础模型框架，整合视觉语言模型的多模态监督与稀疏人工标注形成双重监督信号，通过双向蒸馏机制促进目标模型与双重监督间的知识交换，在SFADA任务中取得最佳性能。

### 背景

源域无主动领域适应（SFADA）利用主动学习选择的有限人工标注增强从源模型到无标注目标域的知识转移。现有领域适应研究虽引入视觉语言（ViL）模型提高伪标签质量或特征对齐，但常将ViL-based和数据监督视为独立来源，缺乏有效融合。

### 目的

克服现有方法中ViL-based和数据监督分离的局限性，设计能够整合多模态监督与稀疏人工标注的新型框架，形成有效的双重监督信号。

### 方法

提出DAM框架，整合ViL模型的多模态监督补充稀疏人工标注；初始化稳定的ViL引导目标；采用双向蒸馏机制促进目标模型与双重监督间的知识交换；在迭代适应过程中实现相互知识交流。

### 主要发现

大量实验表明，DAM在多个SFADA基准和主动学习策略上持续优于现有方法，并建立了新的最先进水平。

### 结论

DAM框架通过有效融合ViL模型的多模态监督与人工标注，解决了现有方法中监督信号分离的问题，在源域无主动领域适应任务中取得了显著的性能提升。

### 翻译

源域无主动领域适应（SFADA）利用主动学习选择的有限人工标注来增强从源模型到无标注目标域的知识转移。虽然最近的领域适应研究引入了视觉语言（ViL）模型来提高伪标签质量或特征对齐，但它们通常将基于ViL的监督和数据监督视为独立来源，缺乏有效融合。为了克服这一限制，我们提出了双主动学习与多模态（DAM）基础模型，这是一种新型框架，整合了来自ViL模型的多模态监督，以补充稀疏的人工标注，从而形成双重监督信号。DAM初始化稳定的ViL引导目标，并采用双向蒸馏机制，在迭代适应过程中促进目标模型与双重监督之间的知识交换。大量实验证明，DAM在多个SFADA基准和主动学习策略上持续优于现有方法，并建立了新的最先进水平。


### 论文摘要

Source-free active domain adaptation (SFADA) enhances knowledge transfer from a source model to an unlabeled target domain using limited manual labels selected via active learning. While recent domain adaptation studies have introduced Vision-and-Language (ViL) models to improve pseudo-label quality or feature alignment, they often treat ViL-based and data supervision as separate sources, lacking effective fusion. To overcome this limitation, we propose Dual Active learning with Multimodal (DAM) foundation model, a novel framework that integrates multimodal supervision from a ViL model to complement sparse human annotations, thereby forming a dual supervisory signal. DAM initializes stable ViL-guided targets and employs a bidirectional distillation mechanism to foster mutual knowledge exchange between the target model and the dual supervisions during iterative adaptation. Extensive experiments demonstrate that DAM consistently outperforms existing methods and sets a new state-of-the-art across multiple SFADA benchmarks and active learning strategies.

---

## 5. Environment-Aware Satellite Image Generation with Diffusion Models

**论文链接:** [http://arxiv.org/abs/2509.24875v1](http://arxiv.org/abs/2509.24875v1)

**作者:** Nikos Kostagiolas, Pantelis Georgiades, Yannis Panagakis, Mihalis A. Nicolaou

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种基于环境上下文条件的新型扩散模型，能够通过文本、元数据和视觉数据三种控制信号的任意组合生成卫星图像。该模型是首个将动态环境条件作为控制信号的卫星图像生成方法，并采用元数据融合策略处理部分损坏或缺失数据。实验表明，该方法在单图像和时间生成任务中均优于先前方法，具有更高的鲁棒性和响应性，生成的图像质量也更好。

### 背景

扩散基础模型因其能生成高质量高保真图像而在生成建模领域受到关注。最近这些模型被应用于遥感领域，标志着利用包含多模态信息的大型公开数据集的成功尝试。然而，现有方法存在明显限制：依赖有限的环境上下文，难以处理缺失或损坏的数据，往往无法可靠反映用户意图。

### 目的

提出一种基于环境上下文条件的新型扩散模型，能够通过三种不同控制信号的任意组合（文本、元数据和视觉数据）来生成卫星图像。

### 方法

所提出的方法是首个将动态环境条件作为控制信号一部分来条件化卫星图像生成的方法，并采用元数据融合策略，该策略建模属性嵌入交互以处理部分损坏和/或缺失的观测值。

### 主要发现

在单图像和时间生成试验中，该方法在定性和定量上都优于先前方法。定性优势包括对缺失元数据的鲁棒性更高，对控制输入的响应性更强；定量优势体现在生成结果具有更高的保真度、准确性和质量（使用6种不同指标测量）。此外，收集的三模态数据集是首个结合这三种不同媒介数据的公开可用数据集。

### 结论

研究结果支持了以下假设：条件化环境上下文可以提高基础模型在卫星图像方面的性能，使该模型成为下游任务使用的有前途的候选者。

### 翻译

基于扩散的基础模型最近在生成建模领域引起了广泛关注，因为它们能够生成高质量和高保真度的图像。虽然并非直接可行，但它们最近在遥感领域的应用标志着利用包含多模态信息的大型公开数据集的成功尝试。尽管取得了成功，现有方法仍面临相当大的限制：它们依赖于有限的环境上下文，难以处理缺失或损坏的数据，并且往往无法在生成的输出中可靠地反映用户意图。在这项工作中，我们提出了一种基于环境上下文条件的新型扩散模型，能够通过三种不同控制信号的任意组合生成卫星图像：a)文本，b)元数据，c)视觉数据。与先前的工作相比，所提出的方法是i)据我们所知，首个将动态环境条件作为控制信号一部分来条件化卫星图像生成的方法，以及ii)结合了元数据融合策略，该策略建模属性嵌入交互以处理部分损坏和/或缺失的观测值。在单图像和时间生成的试验中，我们的方法在定性（对缺失元数据的鲁棒性更高，对控制输入的响应性更强）和定量（使用6种不同指标测量的生成结果具有更高的保真度、准确性和质量）上都优于先前方法。报告的结果支持了我们的假设，即条件化环境上下文可以提高基础模型在卫星图像方面的性能，并使我们的模型成为下游任务使用的有前途的候选者。据我们所知，收集的三模态数据集是首个结合这三种不同媒介数据的公开可用数据集。


### 论文摘要

Diffusion-based foundation models have recently garnered much attention in the field of generative modeling due to their ability to generate images of high quality and fidelity. Although not straightforward, their recent application to the field of remote sensing signaled the first successful trials towards harnessing the large volume of publicly available datasets containing multimodal information. Despite their success, existing methods face considerable limitations: they rely on limited environmental context, struggle with missing or corrupted data, and often fail to reliably reflect user intentions in generated outputs. In this work, we propose a novel diffusion model conditioned on environmental context, that is able to generate satellite images by conditioning from any combination of three different control signals: a) text, b) metadata, and c) visual data. In contrast to previous works, the proposed method is i) to our knowledge, the first of its kind to condition satellite image generation on dynamic environmental conditions as part of its control signals, and ii) incorporating a metadata fusion strategy that models attribute embedding interactions to account for partially corrupt and/or missing observations. Our method outperforms previous methods both qualitatively (robustness to missing metadata, higher responsiveness to control inputs) and quantitatively (higher fidelity, accuracy, and quality of generations measured using 6 different metrics) in the trials of single-image and temporal generation. The reported results support our hypothesis that conditioning on environmental context can improve the performance of foundation models for satellite imagery, and render our model a promising candidate for usage in downstream tasks. The collected 3-modal dataset is to our knowledge, the first publicly-available dataset to combine data from these three different mediums.

---

## 6. DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning

**论文链接:** [http://arxiv.org/abs/2509.24868v1](http://arxiv.org/abs/2509.24868v1)

**作者:** Jiayi Li, Flora D. Salim

**发布时间:** 2025-09-29

### GPT解析

### 总结

DRIFT-Net是一种新型的双分支神经网络架构，用于学习偏微分方程(PDE)动力学，通过结合光谱分支和图像分支，有效解决了全局耦合弱化导致的误差累积和漂移问题，在保持高吞吐量的同时显著提高了精度并减少了参数量。

### 背景

学习PDE动力学的神经网络求解器相比经典数值求解器可显著提高效率和准确性。近年来PDE基础模型多采用多尺度窗口自注意力机制，如Poseidon中的scOT骨干网络。然而，由于这些模型的局部性，真正全局一致的光谱耦合只能通过深度堆叠和窗口移位逐渐传播，导致全局耦合弱化并在闭环运行过程中产生误差累积和漂移。

### 目的

解决现有PDE基础模型中全局耦合弱化导致的误差累积和漂移问题，提出一种能够同时捕获全局结构和局部细节的新型网络架构。

### 方法

提出DRIFT-Net，采用双分支设计：光谱分支负责捕获全局、大尺度低频信息，图像分支专注于局部细节和非平稳结构。具体实现包括：在低频范围内进行受控的轻量级混合；通过带权融合将光谱和图像路径在每一层融合，避免简单连接导致的宽度膨胀和训练不稳定；将融合结果转换回空间域并添加到图像分支，保留各尺度上的全局结构和高频细节。

### 主要发现

与基于注意力的强基线相比，DRIFT-Net在相同训练设置和预算下使用更少参数实现了更低误差和更高吞吐量。在Navier-Stokes基准测试中：相对L1误差降低了7%-54%，参数计数减少了约15%，吞吐量仍高于scOT。消融研究和理论分析进一步证明了这种设计的稳定性和有效性。

### 结论

DRIFT-Net通过双分支设计和带权融合策略，有效解决了PDE求解中全局耦合弱化的问题，在保持高吞吐量的同时显著降低了误差并减少了参数数量，为PDE动力学的高效求解提供了新思路。

### 翻译

使用神经网络求解器学习偏微分方程动力学可以显著提高时钟效率和准确性，相比经典数值求解器。近年来，PDE的基础模型大多采用了多尺度窗口自注意力机制，其中Poseidon中的scOT骨干网络是一个典型代表。然而，由于这些模型的局部性，真正全局一致的光谱耦合只能通过深度堆叠和窗口移位逐渐传播。这削弱了全局耦合，并在闭环运行过程中导致误差累积和漂移。为此，我们提出了DRIFT-Net。它采用双分支设计，包含光谱分支和图像分支。光谱分支负责捕获全局、大尺度低频信息，而图像分支则专注于局部细节和非平稳结构。具体来说，我们首先在低频范围内进行受控的轻量级混合。然后通过带权融合将光谱和图像路径在每一层融合，避免了简单连接导致的宽度膨胀和训练不稳定。融合结果被转换回空间域并添加到图像分支，从而在各尺度上保留全局结构和高频细节。与基于注意力的强基线相比，DRIFT-Net在相同训练设置和预算下，使用更少参数实现了更低误差和更高吞吐量。在Navier-Stokes基准测试中，相对L1误差降低了7%-54%，参数计数减少了约15%，吞吐量仍高于scOT。消融研究和理论分析进一步证明了这种设计的稳定性和有效性。代码可在https://github.com/cruiseresearchgroup/DRIFT-Net获取。


### 论文摘要

Learning PDE dynamics with neural solvers can significantly improve wall-clock efficiency and accuracy compared with classical numerical solvers. In recent years, foundation models for PDEs have largely adopted multi-scale windowed self-attention, with the scOT backbone in \textsc{Poseidon} serving as a representative example.   However, because of their locality, truly globally consistent spectral coupling can only be propagated gradually through deep stacking and window shifting. This weakens global coupling and leads to error accumulation and drift during closed-loop rollouts. To address this, we propose \textbf{DRIFT-Net}. It employs a dual-branch design comprising a spectral branch and an image branch. The spectral branch is responsible for capturing global, large-scale low-frequency information, whereas the image branch focuses on local details and nonstationary structures. Specifically, we first perform controlled, lightweight mixing within the low-frequency range. Then we fuse the spectral and image paths at each layer via bandwise weighting, which avoids the width inflation and training instability caused by naive concatenation. The fused result is transformed back into the spatial domain and added to the image branch, thereby preserving both global structure and high-frequency details across scales. Compared with strong attention-based baselines, DRIFT-Net achieves lower error and higher throughput with fewer parameters under identical training settings and budget. On Navier--Stokes benchmarks, the relative $L_{1}$ error is reduced by 7\%--54\%, the parameter count decreases by about 15\%, and the throughput remains higher than scOT. Ablation studies and theoretical analyses further demonstrate the stability and effectiveness of this design. The code is available at https://github.com/cruiseresearchgroup/DRIFT-Net.

---

## 7. Cell2Text: Multimodal LLM for Generating Single-Cell Descriptions from RNA-Seq Data

**论文链接:** [http://arxiv.org/abs/2509.24840v1](http://arxiv.org/abs/2509.24840v1)

**作者:** Oussama Kharouiche, Aris Markogiannakis, Xiao Fei, Michail Chatzianastasis, Michalis Vazirgiannis

**发布时间:** 2025-09-29

### GPT解析

### 总结

Cell2Text是一个多模态生成框架，能够将单细胞RNA测序(scRNA-seq)数据转换为结构化的自然语言描述，提供比传统方法更丰富的细胞信息解释。

### 背景

单细胞RNA测序通过在细胞分辨率测量基因表达，改变了生物学研究，提供了细胞类型、状态和疾病背景的信息。然而，现有的单细胞基础模型仅限于离散预测头，将细胞复杂性简化为预定义标签，无法提供生物学家需要的更丰富解释。

### 目的

开发Cell2Text框架，将scRNA-seq谱转换为结构化的自然语言描述，捕获细胞的丰富特征并提供更可解释的输出。

### 方法

Cell2Text整合来自单细胞基础模型的基因级嵌入与预训练的大型语言模型，生成连贯的摘要，能够捕获细胞身份、组织来源、疾病关联和通路活性，并推广到未见过的细胞。

### 主要发现

Cell2Text在分类准确性上优于基线方法，使用基于PageRank的相似性指标显示出强的本体一致性，在文本生成中实现高语义保真度。

### 结论

将表达数据与自然语言结合可以提供更强的预测性能和内在可解释的输出，为未见细胞的标记高效表征提供可扩展的路径。

### 翻译

单细胞RNA测序通过在细胞分辨率测量基因表达改变了生物学，提供了细胞类型、状态和疾病背景的信息。最近，单细胞基础模型作为从表达谱中学习可迁移表征的强大工具出现，提高了分类和聚类任务性能。然而，这些模型仅限于离散预测头，将细胞复杂性简化为预定义标签，无法捕获生物学家需要的更丰富、上下文相关的解释。我们引入Cell2Text，一个多模态生成框架，将scRNA-seq谱转换为结构化的自然语言描述。通过整合来自单细胞基础模型的基因级嵌入与预训练的大型语言模型，Cell2Text生成连贯的摘要，捕获细胞身份、组织来源、疾病关联和通路活性，能推广到未见细胞。实验表明，Cell2Text在分类准确性上优于基线，使用基于PageRank的相似性指标显示出强的本体一致性，在文本生成中实现高语义保真度。这些结果表明将表达数据与自然语言结合可提供更强的预测性能和内在可解释的输出，为未见细胞的标记高效表征指出了可扩展路径。


### 论文摘要

Single-cell RNA sequencing has transformed biology by enabling the measurement of gene expression at cellular resolution, providing information for cell types, states, and disease contexts. Recently, single-cell foundation models have emerged as powerful tools for learning transferable representations directly from expression profiles, improving performance on classification and clustering tasks. However, these models are limited to discrete prediction heads, which collapse cellular complexity into predefined labels that fail to capture the richer, contextual explanations biologists need. We introduce Cell2Text, a multimodal generative framework that translates scRNA-seq profiles into structured natural language descriptions. By integrating gene-level embeddings from single-cell foundation models with pretrained large language models, Cell2Text generates coherent summaries that capture cellular identity, tissue origin, disease associations, and pathway activity, generalizing to unseen cells. Empirically, Cell2Text outperforms baselines on classification accuracy, demonstrates strong ontological consistency using PageRank-based similarity metrics, and achieves high semantic fidelity in text generation. These results demonstrate that coupling expression data with natural language offers both stronger predictive performance and inherently interpretable outputs, pointing to a scalable path for label-efficient characterization of unseen cells.

---

## 8. Sparse Autoencoders Make Audio Foundation Models more Explainable

**论文链接:** [http://arxiv.org/abs/2509.24793v1](http://arxiv.org/abs/2509.24793v1)

**作者:** Théo Mariotte, Martin Lebourdais, Antonio Almudévar, Marie Tahon, Alfonso Ortega, Nicolas Dugué

**发布时间:** 2025-09-29

**备注:** 5 pages, 5 figures, 1 table, submitted to ICASSP 2026

### GPT解析

### 总结

本研究探索了使用稀疏自编码器(SAEs)分析音频预训练模型的隐藏表示，以揭示其内部结构并增强声乐属性的解缠结。

### 背景

音频预训练模型被广泛应用于语音处理、声音事件检测和音乐信息检索等任务，但这些模型学习到的表示不明确，分析主要局限于隐藏表示的线性探测。

### 目的

探索使用稀疏自编码器(SAEs)来分析预训练模型的隐藏表示，重点关注歌唱技巧分类的案例研究。

### 方法

采用稀疏自编码器(SAEs)分析预训练音频模型的隐藏表示，研究其在歌唱技巧分类任务中的应用。

### 主要发现

SAEs既保留了原始表示的信息，又保留了类别标签的信息，能够为自监督学习系统提供见解；同时，SAEs增强了声乐属性的解缠结，成为识别表示中编码的潜在因素的有效工具。

### 结论

稀疏自编码器是分析预训练音频模型内部表示的有效工具，有助于理解自监督学习系统并揭示表示中编码的潜在因素。

### 翻译

音频预训练模型被广泛应用于解决语音处理、声音事件检测或音乐信息检索中的各种任务。然而，这些模型学习到的表示不明确，对其分析主要局限于对隐藏表示的线性探测。在本工作中，我们探索使用稀疏自编码器(SAEs)来分析预训练模型的隐藏表示，重点关注歌唱技巧分类的案例研究。我们首先证明SAEs既保留了原始表示的信息，又保留了类别标签的信息，使其内部结构能够为自监督学习系统提供见解。此外，我们展示SAEs增强了声乐属性的解缠结，使其成为识别表示中编码的潜在因素的有效工具。


### 论文摘要

Audio pretrained models are widely employed to solve various tasks in speech processing, sound event detection, or music information retrieval. However, the representations learned by these models are unclear, and their analysis mainly restricts to linear probing of the hidden representations. In this work, we explore the use of Sparse Autoencoders (SAEs) to analyze the hidden representations of pretrained models, focusing on a case study in singing technique classification. We first demonstrate that SAEs retain both information about the original representations and class labels, enabling their internal structure to provide insights into self-supervised learning systems. Furthermore, we show that SAEs enhance the disentanglement of vocal attributes, establishing them as an effective tool for identifying the underlying factors encoded in the representations.

---

## 9. Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation

**论文链接:** [http://arxiv.org/abs/2509.24739v1](http://arxiv.org/abs/2509.24739v1)

**作者:** Huu Tien Nguyen, Dac Thai Nguyen, The Minh Duc Nguyen, Trung Thanh Nguyen, Thao Nguyen Truong, Huy Hieu Pham, Johan Barthelemy, Minh Quan Tran, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Quynh Anh Chau, Hong Son Mai, Thanh Trung Nguyen, Phi Le Nguyen

**发布时间:** 2025-09-29

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS  2025)

### GPT解析

### 总结

本文介绍了一种新的越南语多模态医学数据集，包含156万对CT-PET图像和2757份完整临床报告，旨在填补医学AI发展中PET/CT成像数据和低资源语言的空白，并通过实验证明该数据集能显著提高现有VLMs的性能。

### 背景

视觉-语言基础模型(VLMs)在多模态数据训练下推动了AI发展，但应用于医学成像面临挑战，原因是多样化成像模态和多语言临床数据有限。现有医学VLMs训练模态单一，主要关注高资源语言，限制了泛化能力和临床实用性。

### 目的

1. 解决现有VLMs训练语料库中缺乏PET/CT成像数据的问题；2. 解决医学视觉语言研究中低资源语言(特别是越南语)代表性不足的问题。

### 方法

1. 创建包含1,567,062对CT-PET图像和2,757份完整临床报告的越南语多模态医学数据集；2. 引入训练框架增强VLMs学习，包括数据增强和专家验证测试集；3. 对最先进VLMs在医疗报告生成和视觉问答等下游任务进行基准测试。

### 主要发现

实验结果表明整合该数据集能显著提高现有VLMs性能。据作者所知，这是第一个提供全面PET/CT-报告对越南语的数据集。

### 结论

该数据集和基准测试将作为推动更强大医学成像VLMs发展的重要步骤，特别是在低资源语言方面，并提高它们在越南医疗保健中的临床相关性。

### 翻译

视觉-语言基础模型(VLMs)是在大规模多模态数据集上训练的，通过实现丰富的跨模态推理，推动了人工智能的重大进展。尽管这些模型在通用领域取得了成功，但由于多样化的成像模态和多语言临床数据的可用性有限，将这些模型应用于医学成像仍然具有挑战性。大多数现有的医学VLMs只在部分成像模态上进行训练，并主要关注高资源语言，从而限制了它们的泛化能力和临床实用性。为了解决这些局限性，我们引入了一个新颖的越南语多模态医学数据集，包含1,567,062对CT-PET图像和相应的2,757份完整临床报告。


### 论文摘要

Vision-Language Foundation Models (VLMs), trained on large-scale multimodal datasets, have driven significant advances in Artificial Intelligence by enabling rich cross-modal reasoning. Despite their success in general domains, applying these models to medical imaging remains challenging due to the limited availability of diverse imaging modalities and multilingual clinical data. Most existing medical VLMs are trained on a subset of imaging modalities and focus primarily on high-resource languages, thus limiting their generalizability and clinical utility. To address these limitations, we introduce a novel Vietnamese-language multimodal medical dataset comprising 1,567,062 paired CT-PET images and corresponding 2,757 full-length clinical reports. This dataset is designed to fill two pressing gaps in medical AI development: (1) the lack of PET/CT imaging data in existing VLMs training corpora, which hinders the development of models capable of handling functional imaging tasks; and (2) the underrepresentation of low-resource languages, particularly the Vietnamese language, in medical vision-language research. To the best of our knowledge, this is the first dataset to provide comprehensive PET/CT-report pairs in Vietnamese. We further introduce a training framework to enhance VLMs' learning, including data augmentation and expert-validated test sets. We conduct comprehensive experiments benchmarking state-of-the-art VLMs on downstream tasks, including medical report generation and visual question answering. The experimental results show that incorporating our dataset significantly improves the performance of existing VLMs. We believe this dataset and benchmark will serve as a pivotal step in advancing the development of more robust VLMs for medical imaging, particularly in low-resource languages, and improving their clinical relevance in Vietnamese healthcare.

---

## 10. Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens

**论文链接:** [http://arxiv.org/abs/2509.24693v1](http://arxiv.org/abs/2509.24693v1)

**作者:** Zijian Dong, Ruilin Li, Joanna Su Xian Chong, Niousha Dehestani, Yinghui Teng, Yi Lin, Zhizhou Li, Yichi Zhang, Yapei Xie, Leon Qi Rong Ooi, B. T. Thomas Yeo, Juan Helen Zhou

**发布时间:** 2025-09-29

**备注:** NeurIPS 2025. The first two authors contributed equally

### GPT解析

### 总结

BrainHarmonix是一种多模态脑基础模型，将结构形态和功能动态统一为一维标记表示，在大规模神经影像数据集上预训练，能够处理不同重复时间的fMRI时间序列，在各种神经科学任务中表现优异。

### 背景

神经影像数据规模庞大且复杂，包括结构MRI和功能MRI数据。现有模型在处理不同重复时间的fMRI时间序列方面存在局限性，且缺乏将高维神经影像信号深度压缩为统一表示的方法。

### 目的

开发一种能够统一结构形态和功能动态的多模态脑基础模型，将高维神经影像信号压缩为紧凑的一维标记表示，提高在下游神经科学任务中的性能。

### 方法

BrainHarmonix采用模块化预训练过程，包括单模态训练和几何预对齐，然后通过共享的脑枢纽标记进行模态融合。动态编码器专门设计用于处理具有不同重复时间的fMRI时间序列。模型在包含64,594个T1加权结构MRI和70,933个功能MRI时间序列的大型数据集上预训练。

### 主要发现

BrainHarmonix成功将高维神经影像信号深度压缩为统一、连续的一维标记，形成人脑的紧凑潜在空间。该模型在各种下游任务中表现出强大的泛化能力，包括神经发育和神经退行性疾病分类以及认知预测，性能优于先前的方法。

### 结论

BrainHarmonix代表了AI驱动神经科学的新时代，通过大规模多模态神经影像预训练模型，为神经科学研究提供了强大的工具。

### 翻译

我们提出脑和谐（BrainHarmonix），这是第一个多模态脑基础模型，将结构形态和功能动态统一为紧凑的一维标记表示。该模型迄今为止在两个最大的神经影像数据集上进行预训练，包括64,594个T1加权结构MRI 3D体积（约1400万张图像）和70,933个功能MRI（fMRI）时间序列。BrainHarmonix基于两个基础神经科学原理：结构补充功能 - 结构和功能模态提供关于大脑组织的不同但协同的见解；功能遵循结构 - 脑功能动态由皮层形态塑造。模块化预训练过程涉及单模态训练和几何预对齐，然后通过共享的脑枢纽标记进行模态融合。值得注意的是，我们的动态编码器能够处理具有不同重复时间（TRs）的fMRI时间序列，解决了现有模型的一个主要限制。BrainHarmonix也是首个将高维神经影像信号深度压缩为统一、连续的一维标记的模型，形成人脑的紧凑潜在空间。BrainHarmonix在各种下游任务中实现了强大的泛化能力，包括神经发育和神经退行性疾病分类以及认知预测 - 性能持续优于先前的方法。我们的模型 - 在8个H100 GPU上预训练 - 旨在催化一个由大规模多模态神经影像驱动的新AI时代。


### 论文摘要

We present Brain Harmony (BrainHarmonix), the first multimodal brain foundation model that unifies structural morphology and functional dynamics into compact 1D token representations. The model was pretrained on two of the largest neuroimaging datasets to date, encompassing 64,594 T1-weighted structural MRI 3D volumes (~ 14 million images) and 70,933 functional MRI (fMRI) time series. BrainHarmonix is grounded in two foundational neuroscience principles: structure complements function - structural and functional modalities offer distinct yet synergistic insights into brain organization; function follows structure - brain functional dynamics are shaped by cortical morphology. The modular pretraining process involves single-modality training with geometric pre-alignment followed by modality fusion through shared brain hub tokens. Notably, our dynamics encoder uniquely handles fMRI time series with heterogeneous repetition times (TRs), addressing a major limitation in existing models. BrainHarmonix is also the first to deeply compress high-dimensional neuroimaging signals into unified, continuous 1D tokens, forming a compact latent space of the human brain. BrainHarmonix achieves strong generalization across diverse downstream tasks, including neurodevelopmental and neurodegenerative disorder classification and cognition prediction - consistently outperforming previous approaches. Our models - pretrained on 8 H100 GPUs - aim to catalyze a new era of AI-driven neuroscience powered by large-scale multimodal neuroimaging.

---

## 11. Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.24510v1](http://arxiv.org/abs/2509.24510v1)

**作者:** Jonas Hübotter, Patrik Wolf, Alexander Shevchenko, Dennis Jüni, Andreas Krause, Gil Kur

**发布时间:** 2025-09-29

### GPT解析

### 总结

本研究探讨了测试时训练(TTT)的有效性及其工作机制，提出基础模型仍是全局欠参数化的，TTT提供了一种在泛化后专业化的机制，使模型能够专注于与测试任务相关的概念。

### 背景

最近的实证研究表明，测试时训练(TTT)能够显著提高模型性能，但对其有效性和适用条件的理解仍然有限。早期解释主要关注TTT在分布外适应或使用特权数据时的作用，但随着基础模型规模的扩大，大多数测试数据为分布内数据，这些解释受到质疑。

### 目的

理解测试时训练(TTT)为何有效以及何时有效，探索其工作机制，特别是针对基础模型在测试数据为分布内情况下的表现。

### 方法

在线性表示假设下提出一个理论模型，证明TTT比全局训练能获得更小的分布内测试误差；通过在ImageNet上训练稀疏自编码器验证模型关键假设；在图像和语言任务上进行扩展研究，确认模型的实践意义。

### 主要发现

1) 基础模型仍然是全局欠参数化的；2) TTT提供了一种在泛化后专业化的机制，使模型能够专注于与测试任务相关的概念；3) 在线性表示假设下，TTT能比全局训练获得更小的分布内测试误差；4) 语义相关的数据点仅由少数共享概念解释；5) 确认了专业化的有效条件范围。

### 结论

测试时训练(TTT)的有效性源于它允许基础模型在泛化后进行专业化，专注于与测试任务相关的概念。这种机制在分布内数据上也有效，而不仅限于分布外适应或特权数据场景。研究确定了专业化的最有效条件，为理解和应用TTT提供了理论框架。

### 翻译

最近的实证研究探索了在给定任务期间继续训练模型的思路，称为测试时训练(TTT)，并发现它能够带来显著的性能提升。然而，对于TTT为何有效以及何时有效的理解仍然有限。早期的解释主要关注TTT在应用于分布外适应或与特权数据一起使用时可能有效。但随着基础模型规模的扩大，大多数测试数据为分布内数据，这些解释受到了质疑。我们反而认为基础模型仍然是全局欠参数化的，TTT为泛化后的专业化提供了一种机制，将能力集中在与测试任务相关的概念上。具体来说，在线性表示假设下，我们提出了一个模型，其中TTT比全局训练获得显著更小的分布内测试误差。我们通过在ImageNet上训练稀疏自编码器，验证了模型的关键假设，显示语义相关的数据点仅由少数共享概念解释。最后，我们在图像和语言任务上进行了扩展研究，确认了我们模型的实践意义，确定了专业化最有效的条件范围。


### 论文摘要

Recent empirical studies have explored the idea of continuing to train a model at test-time for a given task, known as test-time training (TTT), and have found it to yield significant performance improvements. However, there is limited understanding of why and when TTT is effective. Earlier explanations mostly focused on the observation that TTT may help when applied to out-of-distribution adaptation or used with privileged data. However, the growing scale of foundation models with most test data being in-distribution questions these explanations. We instead posit that foundation models remain globally underparameterized, with TTT providing a mechanism for specialization after generalization, focusing capacity on concepts relevant to the test task. Specifically, under the linear representation hypothesis, we propose a model in which TTT achieves a substantially smaller in-distribution test error than global training. We empirically validate our model's key assumptions by training a sparse autoencoder on ImageNet, showing that semantically related data points are explained by only a few shared concepts. Finally, we perform scaling studies across image and language tasks that confirm the practical implications of our model, identifying the regimes where specialization is most effective.

---

## 12. Interpretable Kernel Representation Learning at Scale: A Unified Framework Utilizing Nyström Approximation

**论文链接:** [http://arxiv.org/abs/2509.24467v1](http://arxiv.org/abs/2509.24467v1)

**作者:** Maedeh Zarvandi, Michael Timothy, Theresa Wasserer, Debarghya Ghoshdastidar

**发布时间:** 2025-09-29

**备注:** 19 Pages, 3 figures

### GPT解析

### 总结

KREPES是一个通过Nyström近似实现的统一、可扩展的基于核的表示学习框架，适用于各种无监督和自监督损失函数，并在大型数据集上展示了效率。

### 背景

核方法为非线性学习提供了理论框架，但其可扩展性受限于时间和内存成本。尽管核回归已有扩展方案，但缺乏可扩展的基于核的表示学习框架，限制了其在基础模型时代的使用。

### 目的

开发一个可扩展的基于核的表示学习框架，以处理大规模无标记数据的学习表示问题。

### 方法

提出KREPES框架，通过Nyström近似实现基于核的表示学习，适用于多种无监督和自监督损失函数。

### 主要发现

KREPES在大规模图像和表格数据集上表现出高效性，并且能够对学习到的表示进行可解释性分析，这是深度模型的优势。

### 结论

KREPES为基于核的表示学习提供了可扩展的解决方案，同时保持了核方法的优点，并提供了深度模型所具备的可解释性。

### 翻译

核方法为非线性与非参数学习提供了理论基础，具有强大的分析基础和统计保证。然而，其可扩展性长期以来受到时间和内存成本的限制。虽然在扩展核回归方面已有进展，但尚不存在可扩展的基于核的表示学习框架，这限制了它们在基础模型时代的使用，因为基础模型需要从未标记的海量数据中学习表示。我们引入了KREPES——一个通过Nyström近似实现的统一、可扩展的基于核的表示学习框架。KREPES适用于各种无监督和自监督损失函数，在大型图像和表格数据集上的实验证明了其效率。重要的是，KREPES能够对学习到的表示进行可解释性分析，这是深度模型的直接优势，我们通过专门的分析证实了这一点。


### 论文摘要

Kernel methods provide a theoretically grounded framework for non-linear and non-parametric learning, with strong analytic foundations and statistical guarantees. Yet, their scalability has long been limited by prohibitive time and memory costs. While progress has been made in scaling kernel regression, no framework exists for scalable kernel-based representation learning, restricting their use in the era of foundation models where representations are learned from massive unlabeled data. We introduce KREPES -- a unified, scalable framework for kernel-based representation learning via Nystr\"om approximation. KREPES accommodates a wide range of unsupervised and self-supervised losses, and experiments on large image and tabular datasets demonstrate its efficiency. Crucially, KREPES enables principled interpretability of the learned representations, an immediate benefit over deep models, which we substantiate through dedicated analysis.

---

## 13. Evolutionary hypergame dynamics: Introspection reasoning and social learning

**论文链接:** [http://arxiv.org/abs/2509.24398v1](http://arxiv.org/abs/2509.24398v1)

**作者:** Feipeng Zhang, Te Wu, Guofeng Zhang, Long Wang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本研究探索了超博弈框架下的进化动态，特别是在社会困境中的表现，发现更高的理性能显著促进合作行为。

### 背景

传统进化博弈论假设所有玩家都拥有完整知识和无限制访问策略空间的能力，但现实社会中个体间存在知识、经验和背景的差异。

### 目的

探索超博弈的进化后果，特别是在社会困境中不同策略动态下的行为模式。

### 方法

使用包含三种策略（合作、背叛和独行者）的模型，这些策略表现出循环支配关系。研究涵盖混合群体和空间晶格群体，分析策略集在进化超博弈动态中的学习和进化过程。

### 主要发现

与传统进化博弈动态相比，超博弈框架下出现了更复杂的阶段，包括独行者主导、多种策略集共存、合作与独行者主导的组合等。更高的理性显著促进了合作行为。

### 结论

超博弈框架能够更好地捕捉现实社会中策略选择的异质性，产生比传统模型更丰富的动态行为。

### 翻译

在进化博弈论领域，标准框架通常假设每个玩家都拥有全面的知识和无限制访问整个策略空间的权限。然而，现实人类社会本质上存在个体间知识、经验和背景的多样性。超博弈通过允许个体在完整策略集的访问上存在差异来体现这种异质性，反映了认知或信息限制，并产生了非对称的战略互动。然而，它们的进化后果尚未得到充分探索。我们的研究采用了包含三种可用策略的原型模型，专注于涉及合作、背叛和独行者策略的社会困境。这些策略表现出循环支配关系，类似于众所周知的石头-剪刀-布动态，这是博弈论中的基础模型。我们的研究涵盖了混合群体和空间晶格群体，深入探讨了在进化超博弈动态中策略集的学习和进化。与传统进化博弈动态形成鲜明对比的是，我们的研究结果揭示了细致而复杂的阶段，包括独行者主导的场景、多种策略集的共存、合作与独行者主导的组合等。值得注意的是，我们察觉到更高的理性显著促进了合作行为。


### 论文摘要

In the realm of evolutionary game theory, standard frameworks typically presuppose that every player possesses comprehensive knowledge and unrestricted access to the entire strategy space. However, real-world human society inherently harbors diverse levels of knowledge, experience, and background among individuals. Hypergames incorporate this heterogeneity by permitting individuals to differ in their access to the full strategy set, reflecting cognitive or informational constraints and giving rise to asymmetric strategic interactions. Yet, their evolutionary consequences remain underexplored. Our inquiry employs prototype models featuring three available strategies, focusing on social dilemmas involving cooperation, defection, and loner. These strategies manifest cyclic dominance, akin to the well-studied rock-paper-scissors dynamics, a foundational model in game theory. Our study spans both well-mixed and spatial lattice populations, delving into the intricacies of learning and evolution of the strategy set within the evolutionary hypergame dynamics. In stark contrast to traditional evolutionary game dynamics, our findings unveil nuanced and intricate phases, encompassing scenarios of loner dominance, coexistence of multiple strategy sets, combinations of cooperation and loner dominance, and more. Remarkably, we discern that heightened rationality significantly promotes cooperative behaviors.

---

## 14. UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities

**论文链接:** [http://arxiv.org/abs/2509.24391v1](http://arxiv.org/abs/2509.24391v1)

**作者:** Xuenan Xu, Jiahao Mei, Zihao Zheng, Ye Tao, Zeyu Xie, Yaoyun Zhang, Haohe Liu, Yuning Wu, Ming Yan, Wen Wu, Chao Zhang, Mengyue Wu

**发布时间:** 2025-09-29

**备注:** Project page: https://wsntxxn.github.io/uniflow_audio

### GPT解析

### 总结

论文提出了UniFlow-Audio，一个基于flow matching的通用音频生成框架，支持多种音频生成任务，包括语音、音乐和音效。该框架使用双融合机制和任务平衡的数据采样方法，在7个任务上取得了良好结果，使用少于8K小时的公共训练数据和少于10亿的可训练参数。

### 背景

音频生成（包括语音、音乐和音效）近年来发展迅速。音频生成任务可分为两类：时间对齐（TA）任务和非时间对齐（NTA）任务。这两类任务通常采用不同的建模范式，导致研究轨迹分离。音频本质上不应被划分为这些类别，统一模型是通用音频生成的自然且必要的目标。之前的统一音频生成工作采用了自回归架构，而非自回归的统一方法尚未充分探索。

### 目的

开发一个统一的音频生成框架，能够同时处理时间对齐和非时间对齐的任务；探索非自回归的统一音频生成方法。

### 方法

提出了UniFlow-Audio，一个基于flow matching的通用音频生成框架；使用双融合机制：时间上将音频潜在变量与TA特征对齐，并通过交叉注意力整合NTA特征；采用任务平衡的数据采样方法，以在TA和NTA任务间保持强大性能；支持多模态输入，包括文本、音频和视频；利用多任务学习和flow matching的生成建模能力。

### 主要发现

UniFlow-Audio在7个任务上取得了强大结果；使用少于8K小时的公共训练数据和少于10亿的可训练参数；即使只有约2亿可训练参数的小型变体也显示出具有竞争力的性能。

### 结论

UniFlow-Audio是音频生成领域一个潜在的非自回归基础模型；代码和模型将在https://wsntxxn.github.io/uniflow_audio上提供。

### 翻译

音频生成，包括语音、音乐和音效，近年来发展迅速。这些任务可分为两类：时间对齐（TA）任务，其中每个输入单元对应输出音频的特定片段（例如，语音合成中音素与帧对齐）；和非时间对齐（NTA）任务，其中没有这种对齐。由于两种类型的建模范式通常不同，不同音频生成任务的研究传统上遵循不同的轨迹。然而，音频本质上并不分为这样的类别，因此统一模型是通用音频生成的自然且必要的目标。之前的统一音频生成工作采用了自回归架构，而非自回归的统一方法仍 largely 未被探索。在这项工作中，我们提出了UniFlow-Audio，一个基于flow matching的通用音频生成框架。我们提出了一个双融合机制，在时间上将音频潜在变量与TA特征对齐，并在每个模型块中通过交叉注意力整合NTA特征。采用任务平衡的数据采样来保持TA和NTA任务之间的强大性能。UniFlow-Audio支持全模态，包括文本、音频和视频。通过利用多任务学习的优势和flow matching的生成建模能力，UniFlow-Audio使用不到8K小时的公共训练数据和不到10亿的可训练参数在7个任务上取得了强大结果。即使是只有约2亿可训练参数的小型变体也显示出具有竞争力的性能，突显了UniFlow-Audio作为音频生成领域潜在的非自回归基础模型。代码和模型将在https://wsntxxn.github.io/uniflow_audio上提供。


### 论文摘要

Audio generation, including speech, music and sound effects, has advanced rapidly in recent years. These tasks can be divided into two categories: time-aligned (TA) tasks, where each input unit corresponds to a specific segment of the output audio (e.g., phonemes aligned with frames in speech synthesis); and non-time-aligned (NTA) tasks, where such alignment is not available. Since modeling paradigms for the two types are typically different, research on different audio generation tasks has traditionally followed separate trajectories. However, audio is not inherently divided into such categories, making a unified model a natural and necessary goal for general audio generation. Previous unified audio generation works have adopted autoregressive architectures, while unified non-autoregressive approaches remain largely unexplored. In this work, we propose UniFlow-Audio, a universal audio generation framework based on flow matching. We propose a dual-fusion mechanism that temporally aligns audio latents with TA features and integrates NTA features via cross-attention in each model block. Task-balanced data sampling is employed to maintain strong performance across both TA and NTA tasks. UniFlow-Audio supports omni-modalities, including text, audio, and video. By leveraging the advantage of multi-task learning and the generative modeling capabilities of flow matching, UniFlow-Audio achieves strong results across 7 tasks using fewer than 8K hours of public training data and under 1B trainable parameters. Even the small variant with only ~200M trainable parameters shows competitive performance, highlighting UniFlow-Audio as a potential non-auto-regressive foundation model for audio generation. Code and models will be available at https://wsntxxn.github.io/uniflow_audio.

---

## 15. DINOReg: Strong Point Cloud Registration with Vision Foundation Model

**论文链接:** [http://arxiv.org/abs/2509.24370v1](http://arxiv.org/abs/2509.24370v1)

**作者:** Congjia Chen, Yufu Qu

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出DINOReg，一种充分利用视觉和几何信息解决点云配准问题的网络，通过DINOv2提取视觉特征并在补丁级别融合视觉和几何特征，同时提出混合位置编码增强空间关系感知能力。

### 背景

点云配准是3D计算机视觉的基本任务，现有方法主要依赖几何信息，最近一些研究开始利用RGB-D数据的颜色信息，但这些方法未充分利用图像中的纹理和语义信息，且特征融合以图像有损方式进行，限制了性能。

### 目的

提出DINOReg网络，充分利用视觉和几何信息解决点云配准问题。

### 方法

采用DINOv2从图像中提取视觉特征，在补丁级别融合视觉和几何特征，结合DINOv2的纹理和语义信息与几何主干的几何结构信息；提出混合位置编码编码图像空间和点云空间的位置信息，增强补丁间空间关系感知能力。

### 主要发现

在RGBD-3DMatch和RGBD-3DLoMatch数据集上的实验表明，该方法在补丁内点比率提高14.2%，配准召回率提高15.7%，显著优于现有仅几何和多模态配准方法。

### 结论

DINOReg通过有效融合视觉和几何信息，解决了点云配准问题，并在多个指标上超越现有方法。

### 翻译

点云配准是3D计算机视觉中的基本任务。大多数现有方法仅依赖几何信息进行特征提取和匹配。最近，一些研究将RGB-D数据中的颜色信息纳入特征提取。尽管这些方法取得了显著改进，但它们尚未充分利用图像中丰富的纹理和语义信息，且特征融合是以图像有损方式进行的，这限制了它们的性能。在本文中，我们提出DINOReg，一个充分利用视觉和几何信息来解决点云配准问题的配准网络。受视觉基础模型进展的启发，我们采用DINOv2从图像中提取有信息的视觉特征，并在补丁级别融合视觉和几何特征。这种设计有效地结合了DINOv2提取的丰富纹理和全局语义信息与几何主干捕获的详细几何结构信息。此外，我们提出混合位置编码来编码图像空间和点云空间的位置信息，增强模型对补丁间空间关系的感知能力。在RGBD-3DMatch和RGBD-3DLoMatch数据集上的大量实验表明，我们的方法在仅几何和多模态配准方法上取得了显著改进，补丁内点比率提高了14.2%，配准召回率提高了15.7%。代码可在https://github.com/ccjccjccj/DINOReg公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云配准中仅依赖几何信息进行特征提取和匹配的局限性。现有方法未能充分利用图像中丰富的纹理和语义信息，特征融合过程存在信息丢失。点云配准是3D计算机视觉的基础任务，对3D重建、姿态估计等应用至关重要。当点云重叠区域小或几何结构模糊时，纯几何方法性能显著下降，解决此问题能提高配准的鲁棒性和准确性，扩大应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到视觉基础模型在图像特征提取方面的强大能力，借鉴了DINOv2作为视觉主干网络提取图像特征。在几何特征提取方面，采用了KPConv-FPN结构作为几何主干网络。同时，作者参考了transformer架构中的自注意力和交叉注意力机制进行全局上下文聚合。创新点在于将视觉基础模型与点云几何信息有效结合，通过patch级别的特征融合和改进的位置嵌入方法，充分利用图像的纹理语义信息和点云的几何结构信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型(DINOv2)提取图像中的丰富纹理和语义信息，同时提取点云的几何结构信息，在patch级别有效融合视觉和几何特征，并通过改进的位置编码增强模型对空间关系的感知能力。整体流程包括：1)多模态特征提取(使用DINOv2提取图像特征，KPConv-FPN提取点云特征)；2)空间映射和特征融合(确定patch间空间映射，使用窗口聚合策略，通过前馈网络融合特征)；3)视觉-几何Transformer(使用自注意力和交叉注意力机制，采用混合位置编码)；4)训练和变换估计(采用粗到精匹配策略，使用局部到全局配准方法估计变换)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)探索视觉基础模型在点云配准中的应用，提出DINOReg网络充分利用图像和点云数据，而之前工作大多仅使用几何信息或简单颜色信息；2)设计空间映射和窗口聚合策略在patch级别有效融合多模态特征，而之前方法通常在点级别融合，容易丢失结构信息；3)提出混合位置嵌入同时考虑2D图像空间和3D点云空间的位置信息，而之前的位置编码主要考虑3D几何结构；4)构建RGBD-3DMatch & RGBD-3DLoMatch数据集，解决了之前缺乏适合基准数据集的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DINOReg通过有效融合视觉基础模型提取的丰富纹理语义信息和点云的几何结构信息，显著提高了点云配准的性能，特别是在低重叠度场景下，并提出了新的数据集和位置编码方法以增强配准的准确性和鲁棒性。'}


### 论文摘要

Point cloud registration is a fundamental task in 3D computer vision. Most existing methods rely solely on geometric information for feature extraction and matching. Recently, several studies have incorporated color information from RGB-D data into feature extraction. Although these methods achieve remarkable improvements, they have not fully exploited the abundant texture and semantic information in images, and the feature fusion is performed in an image-lossy manner, which limit their performance. In this paper, we propose DINOReg, a registration network that sufficiently utilizes both visual and geometric information to solve the point cloud registration problem. Inspired by advances in vision foundation models, we employ DINOv2 to extract informative visual features from images, and fuse visual and geometric features at the patch level. This design effectively combines the rich texture and global semantic information extracted by DINOv2 with the detailed geometric structure information captured by the geometric backbone. Additionally, a mixed positional embedding is proposed to encode positional information from both image space and point cloud space, which enhances the model's ability to perceive spatial relationships between patches. Extensive experiments on the RGBD-3DMatch and RGBD-3DLoMatch datasets demonstrate that our method achieves significant improvements over state-of-the-art geometry-only and multi-modal registration methods, with a 14.2% increase in patch inlier ratio and a 15.7% increase in registration recall. The code is publicly available at https://github.com/ccjccjccj/DINOReg.

---

## 16. Towards Foundation Models for Cryo-ET Subtomogram Analysis

**论文链接:** [http://arxiv.org/abs/2509.24311v1](http://arxiv.org/abs/2509.24311v1)

**作者:** Runmin Jiang, Wanyue Feng, Yuntian Yang, Shriya Pingulkar, Hong Wang, Xi Xiao, Xiaoyu Cao, Genpei Zhang, Xiao Wang, Xiaolong Wu, Tianyang Wang, Yang Liu, Xingjian Li, Min Xu

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究为冷冻电子断层扫描（cryo-ET）子断层图分析开发了基础模型，解决了标注稀缺、严重噪声和泛化能力差的问题，在多个数据集上取得了最先进的性能。

### 背景

冷冻电子断层扫描（cryo-ET）可以在原位可视化大分子结构，子断层图的分析任务（如分类、对齐和平均）对结构确定至关重要，但有效分析受到标注稀缺、严重噪声和泛化能力差的影响。

### 目的

开发基础模型来解决cryo-ET子断层图分析面临的标注稀缺、严重噪声和泛化能力差的挑战。

### 方法

1. 引入CryoEngine，一个大规模合成数据生成器，可生成超过904k个子断层图用于预训练；2. 设计APT-ViT，一种自适应相位标记增强的视觉Transformer，提高了对几何和语义变化的鲁棒性；3. 引入NRCL策略，在严重噪声条件下稳定表示学习。

### 主要发现

在24个合成和真实数据集上的评估显示，在三个主要子断层图任务上都达到了最先进的性能，并且对未见过的数据集有很强的泛化能力。

### 结论

该研究推进了cryo-ET中可扩展和鲁棒的子断层图分析。

### 翻译

冷冻电子断层扫描（cryo-ET）能够实现大分子结构的原位可视化，其中子断层图分析任务如分类、对齐和平均对结构确定至关重要。然而，有效分析受到标注稀缺、严重噪声和泛化能力差的影响。为应对这些挑战，我们迈出了为cryo-ET子断层图开发基础模型的第一步。首先，我们引入了CryoEngine，一个大规模合成数据生成器，可从452个粒子类生成超过904k个子断层图用于预训练。其次，我们设计了自适应相位标记增强的视觉Transformer（APT-ViT），它将自适应相位标记作为等变性增强模块，提高了对几何和语义变化的鲁棒性。第三，我们引入了一种噪声鲁棒对比学习（NRCL）策略，在严重噪声条件下稳定表示学习。在24个合成和真实数据集上的评估表明，在所有三个主要子断层图任务上都达到了最先进的性能，并且对未见过的数据集有很强的泛化能力，推进了cryo-ET中可扩展和鲁棒的子断层图分析。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决冷冻电子断层扫描（cryo-ET）中亚tomogram分析面临的三大挑战：标注数据稀缺、严重噪声和泛化能力差。这个问题非常重要，因为冷冻电子断层扫描是观察细胞内大分子结构的关键技术，而亚tomogram分析（分类、对齐和平均化）是确定这些结构的核心步骤。这些挑战限制了高分辨率分子结构的解析，阻碍了我们对生物过程如细菌效应物分泌和哺乳动物神经功能的理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了冷冻电子断层扫描中亚tomogram分析的四个主要困难：标注数据稀缺、极低信噪比、随机方向位移和结构异质性。针对每个分析任务的具体挑战进行了详细研究。作者借鉴了基础模型在生物医学成像和结构生物学中的成功应用，认识到基础模型可以学习可泛化的表示，减少对稀缺标注数据的依赖。具体技术方案借鉴了多个领域：CryoEngine借鉴了生物物理成像原理和数据合成技术；APT-ViT借鉴了多相分解、SE(3)等变性和Vision Transformer技术；NRCL借鉴了对比学习框架和噪声鲁棒表示学习方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个专门针对冷冻电子断层扫描亚tomogram分析的基础模型，通过大规模合成数据预训练、等变增强架构设计和噪声鲁棒训练策略，解决数据稀缺、噪声干扰和泛化能力差的问题。整体流程包括：1）使用CryoEngine合成大规模训练数据，从蛋白质数据库获取原子结构，转换为密度体积并添加噪声；2）采用APT-ViT作为骨干网络，通过多相分解和自适应相位选择增强平移等变性，使用球形可卷积提供旋转等变性；3）应用NRCL训练策略，结合实例区分和噪声感知采样；4）预训练模型用于分类、对齐和平均化三个下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）CryoEngine：首个专门为冷冻电子断层扫描设计的生物物理信息合成数据引擎，生成904k个子tomogram；2）APT-ViT：首个结合自适应相位标记化的SE(3)等变性Vision Transformer，增强几何变换鲁棒性；3）NRCL：针对严重噪声条件的对比学习策略。相比之前的工作，这篇论文首次提出了冷冻电子断层扫描亚tomogram分析的基础模型框架，解决了数据稀缺问题，提出了统一的模型架构处理多个任务，在24个数据集上实现了最先进性能，且不依赖预定义参考模板。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了首个针对冷冻电子断层扫描亚tomogram分析的基础模型，通过大规模合成数据生成、等变增强架构设计和噪声鲁棒训练策略，在分类、对齐和平均化任务上实现了最先进的性能。'}


### 论文摘要

Cryo-electron tomography (cryo-ET) enables in situ visualization of macromolecular structures, where subtomogram analysis tasks such as classification, alignment, and averaging are critical for structural determination. However, effective analysis is hindered by scarce annotations, severe noise, and poor generalization. To address these challenges, we take the first step towards foundation models for cryo-ET subtomograms. First, we introduce CryoEngine, a large-scale synthetic data generator that produces over 904k subtomograms from 452 particle classes for pretraining. Second, we design an Adaptive Phase Tokenization-enhanced Vision Transformer (APT-ViT), which incorporates adaptive phase tokenization as an equivariance-enhancing module that improves robustness to both geometric and semantic variations. Third, we introduce a Noise-Resilient Contrastive Learning (NRCL) strategy to stabilize representation learning under severe noise conditions. Evaluations across 24 synthetic and real datasets demonstrate state-of-the-art (SOTA) performance on all three major subtomogram tasks and strong generalization to unseen datasets, advancing scalable and robust subtomogram analysis in cryo-ET.

---

## 17. Experience Paper: Adopting Activity Recognition in On-demand Food Delivery Business

**论文链接:** [http://arxiv.org/abs/2509.24303v1](http://arxiv.org/abs/2509.24303v1)

**作者:** Huatao Xu, Yan Zhang, Wei Gao, Guobin Shen, Mo Li

**发布时间:** 2025-09-29

**DOI:** 10.1145/3680207.3765261

**备注:** 13 pages

### GPT解析

### 总结

本研究介绍了人类活动识别(HAR)技术在按需食品配送行业中的全国首次部署，成功将LIMU-BERT基础模型适配到配送平台，实现了从城市试点到全国范围的扩展，展示了显著的经济和运营效益。

### 背景

HAR技术在按需食品配送行业中的应用尚属首次，需要从可行性研究扩展到全国范围的大规模应用。

### 目的

将最先进的LIMU-BERT基础模型适配到配送平台，实现HAR技术在食品配送行业的大规模应用，并评估其经济和运营效益。

### 方法

将LIMU-BERT基础模型适配到配送平台，分三个阶段历时两年进行部署：从扬州市的可行性研究扩展到全国范围，涉及中国367个城市中的50万名快递员。

### 主要发现

HAR技术的全国部署实现了一系列下游应用，大规模测试展示了其显著的经济和运营效益，证明了HAR技术在现实世界应用中的变革潜力。

### 结论

HAR技术在按需食品配送行业的大规模部署取得了成功，具有显著的经济和运营效益，并展示了在现实世界应用中的变革潜力。研究团队分享了部署经验并开源了预训练模型。

### 翻译

本文介绍了人类活动识别(HAR)技术在按需食品配送行业中的首次全国性部署。我们成功地将最先进的LIMU-BERT基础模型适配到配送平台。历时两年的部署分为三个阶段，从扬州市的可行性研究扩展到全国范围，涉及中国367个城市中的50万名快递员。这种应用实现了一系列下游应用，大规模测试展示了其显著的经济和运营效益，展示了HAR技术在现实世界应用中的变革潜力。此外，我们分享了从这次部署中获得的经验教训，并开源了使用数百万小时传感器数据预训练的LIMU-BERT模型。


### 论文摘要

This paper presents the first nationwide deployment of human activity recognition (HAR) technology in the on-demand food delivery industry. We successfully adapted the state-of-the-art LIMU-BERT foundation model to the delivery platform. Spanning three phases over two years, the deployment progresses from a feasibility study in Yangzhou City to nationwide adoption involving 500,000 couriers across 367 cities in China. The adoption enables a series of downstream applications, and large-scale tests demonstrate its significant operational and economic benefits, showcasing the transformative potential of HAR technology in real-world applications. Additionally, we share lessons learned from this deployment and open-source our LIMU-BERT pretrained with millions of hours of sensor data.

---

## 18. ELASTIQ: EEG-Language Alignment with Semantic Task Instruction and Querying

**论文链接:** [http://arxiv.org/abs/2509.24302v1](http://arxiv.org/abs/2509.24302v1)

**作者:** Muyun Jiang, Shuailei Zhang, Zhenjie Yang, Mengjun Wu, Weibang Jiang, Zhiwei Guo, Wei Zhang, Rui Liu, Shangen Zhang, Yong Li, Yi Ding, Cuntai Guan

**发布时间:** 2025-09-29

### GPT解析

### 总结

ELASTIQ是一种创新的EEG-语言对齐基础模型，通过整合任务感知的语义指导，生成结构化和语言对齐的EEG嵌入，从而提高解码的鲁棒性和可迁移性。

### 背景

脑电图(EEG)基础模型的发展加速了脑机接口(BCI)的发展，但现有方法难以将语言指令作为先验约束整合到EEG表示学习中，限制了利用语言中固有的语义知识来统一不同标签和任务的能力。

### 目的

解决现有方法难以整合语言指令作为EEG表示学习先验约束的问题，开发一个能够将EEG与语言对齐的基础模型。

### 方法

提出ELASTIQ模型，在预训练阶段引入联合频谱-时间重建(STR)模块，结合频率掩码、随机掩码和因果掩码；在指令调整阶段提出基于指令的Q-Former(IQF)，一种基于查询的交叉注意力transformer，将指令嵌入注入到EEG标记中并与文本标签嵌入对齐。

### 主要发现

ELASTIQ在20个数据集(涵盖运动想象、情感识别等五类任务)上评估，在14个数据集上取得最先进性能，在所有五个任务类别中获得最佳平均结果；分析首次揭示显式任务指令作为语义先验，指导EEG嵌入进入连贯且语言有基础的空间。

### 结论

ELASTIQ成功整合语言指令作为EEG表示学习的先验约束，通过语义任务指令和查询实现EEG与语言对齐，提高解码鲁棒性和可迁移性，代码和预训练权重将公开释放。

### 翻译

最近脑电图(EEG)基础模型的进展，这些模型可捕获可迁移的EEG表示，大大加速了脑机接口(BCI)的发展。然而，现有方法仍然难以将语言指令作为先验约束整合到EEG表示学习中，限制了它们利用语言中固有的语义知识来统一不同标签和任务的能力。为解决这一挑战，我们提出了ELASTIQ，一个用于EEG-语言对齐的基础模型，具有语义任务指令和查询功能。ELASTIQ整合了任务感知的语义指导，生成结构化和语言对齐的EEG嵌入，从而增强了解码的鲁棒性和可迁移性。在预训练阶段，我们引入了联合频谱-时间重建(STR)模块，该模块将频率掩码作为全局频谱扰动，与两种互补的时间目标相结合：随机掩码捕获上下文依赖性，因果掩码建模序列动态。在指令调整阶段，我们提出了基于指令的Q-Former(IQF)，一种基于查询的交叉注意力transformer，将指令嵌入注入到EEG标记中，并通过可学习的查询将它们与文本标签嵌入对齐。我们在20个数据集上评估了ELASTIQ，这些数据集涵盖运动想象、情感识别、稳态视觉诱发电位、隐性言语和医疗保健任务。ELASTIQ在20个数据集中的14个上取得了最先进的性能，并在所有五个任务类别中获得了最佳的平均结果。重要的是，我们的分析首次揭示显式任务指令作为语义先验，指导EEG嵌入进入连贯且语言有基础的空间。代码和预训练权重将被释放。


### 论文摘要

Recent advances in electroencephalography (EEG) foundation models, which capture transferable EEG representations, have greatly accelerated the development of brain-computer interfaces (BCI). However, existing approaches still struggle to incorporate language instructions as prior constraints for EEG representation learning, limiting their ability to leverage the semantic knowledge inherent in language to unify different labels and tasks. To address this challenge, we present ELASTIQ, a foundation model for EEG-Language Alignment with Semantic Task Instruction and Querying. ELASTIQ integrates task-aware semantic guidance to produce structured and linguistically aligned EEG embeddings, thereby enhancing decoding robustness and transferability. In the pretraining stage, we introduce a joint Spectral-Temporal Reconstruction (STR) module, which combines frequency masking as a global spectral perturbation with two complementary temporal objectives: random masking to capture contextual dependencies and causal masking to model sequential dynamics. In the instruction tuning stage, we propose the Instruction-conditioned Q-Former (IQF), a query-based cross-attention transformer that injects instruction embeddings into EEG tokens and aligns them with textual label embeddings through learnable queries. We evaluate ELASTIQ on 20 datasets spanning motor imagery, emotion recognition, steady-state visual evoked potentials, covert speech, and healthcare tasks. ELASTIQ achieves state-of-the-art performance on 14 of the 20 datasets and obtains the best average results across all five task categories. Importantly, our analyses reveal for the first time that explicit task instructions serve as semantic priors guiding EEG embeddings into coherent and linguistically grounded spaces. The code and pre-trained weights will be released.

---

## 19. G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge

**论文链接:** [http://arxiv.org/abs/2509.24276v1](http://arxiv.org/abs/2509.24276v1)

**作者:** Linhao Luo, Zicheng Zhao, Junnan Liu, Zhangchi Qiu, Junnan Dong, Serge Panev, Chen Gong, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung, Alan Wee-Chung Liew, Shirui Pan

**发布时间:** 2025-09-29

**备注:** 22 pages, 6 figures

### GPT解析

### 总结

研究提出G-reasoner框架，整合图和语言基础模型，用于在多样化图结构知识上进行推理，通过QuadGraph抽象和图基础模型(GFM)解决了现有检索增强生成(RAG)方法的局限性。

### 背景

大语言模型擅长复杂推理但受限于静态和不完整的参数化知识；现有检索增强生成(RAG)在知识密集型任务中表现不佳，原因是信息碎片化和知识结构建模薄弱；图提供建模知识关系的自然方式，但LLMs无法有效在图结构数据上推理；现有图增强RAG方法依赖临时图设计、启发式搜索或昂贵代理流水线，阻碍了可扩展性和泛化能力。

### 目的

解决现有GraphRAG方法的局限性，提出一个统一框架整合图和语言基础模型，用于推理多样化的图结构知识，提高LLMs在知识密集型任务中的表现。

### 方法

提出G-reasoner统一框架；核心是QuadGraph，标准化四层抽象，将异构知识源统一为通用图表示；引入3400万参数的图基础模型(GFM)，同时捕获图拓扑和文本语义；将GFM与LLMs集成增强下游推理；实现混合精度训练和分布式消息传递确保可扩展性和效率。

### 主要发现

在六个基准测试上，G-reasoner持续优于最先进基线；显著增强LLM推理能力；实现强大的计算效率和跨图泛化能力。

### 结论

G-reasoner是一个有效的统一框架，能够整合图和语言基础模型，提高LLMs在图结构知识上的推理能力，在性能、效率和泛化能力方面表现出色。

### 翻译

大型语言模型擅长复杂推理，但仍然受限于静态和不完整的参数化知识。检索增强生成通过整合外部知识缓解了这一问题，但现有RAG在知识密集型任务中表现不佳，原因是信息碎片化和知识结构建模薄弱。图提供了一种建模知识中关系的自然方式，但LLMs本质上是非结构化的，无法在图结构数据上进行有效推理。最近的图增强RAG尝试构建定制图并使LLMs在其上进行推理，但这些方法通常依赖于临时图设计、启发式搜索或昂贵的代理流水线，阻碍了可扩展性和泛化能力。为解决这些挑战，我们提出G-reasoner，一个统一框架，整合图和语言基础模型用于推理多样化的图结构知识。我们方法的核心是QuadGraph，一个标准化的四层抽象，将异构知识源统一为通用图表示。基于此，我们引入一个具有3400万参数的图基础模型(GFM)，同时捕获图拓扑和文本语义，并与LLMs集成以增强下游应用的推理能力。为确保可扩展性和效率，我们实现了混合精度训练和分布式消息传递，使GFM能够使用更多GPU进行扩展。在六个基准测试上的大量实验表明，G-reasoner持续优于最先进的基线，显著增强LLM推理能力，并实现了强大的效率和跨图泛化能力。


### 论文摘要

Large language models (LLMs) excel at complex reasoning but remain limited by static and incomplete parametric knowledge. Retrieval-augmented generation (RAG) mitigates this by incorporating external knowledge, yet existing RAGs struggle with knowledge-intensive tasks due to fragmented information and weak modeling of knowledge structure. Graphs offer a natural way to model relationships within knowledge, but LLMs are inherently unstructured and cannot effectively reason over graph-structured data. Recent graph-enhanced RAG (GraphRAG) attempts to bridge this gap by constructing tailored graphs and enabling LLMs to reason on them. However, these methods often depend on ad-hoc graph designs, heuristic search, or costly agent pipelines, which hinder scalability and generalization. To address these challenges, we present G-reasoner, a unified framework that integrates graph and language foundation models for reasoning over diverse graph-structured knowledge. Central to our approach is QuadGraph, a standardized four-layer abstraction that unifies heterogeneous knowledge sources into a common graph representation. Building on this, we introduce a 34M-parameter graph foundation model (GFM) that jointly captures graph topology and textual semantics, and is integrated with LLMs to enhance reasoning in downstream applications. To ensure scalability and efficiency, mixed-precision training and distributed message-passing are implemented to scale GFM with more GPUs. Extensive experiments on six benchmarks show that G-reasoner consistently outperforms state-of-the-art baselines, significantly enhances LLM reasoning, and achieves strong efficiency and cross-graph generalization.

---

## 20. Graph Foundation Models: Bridging Language Model Paradigms and Graph Optimization

**论文链接:** [http://arxiv.org/abs/2509.24256v1](http://arxiv.org/abs/2509.24256v1)

**作者:** Yunhao Liang, Pujun Zhang, Yuan Qu, Shaochong Lin, Zuo-jun Max Shen

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了图基础模型(GFM)，这是第一个能够解决图结构上所有基于距离的优化问题的框架。通过在图中随机游走生成的路径上引入类似大型语言模型的自监督预训练范式，GFM能够内化图的复杂拓扑和组合规则。实验表明，GFM在各种优化任务上与专用求解器具有竞争力的性能，同时保持更快的推理时间。

### 背景

预训练-迁移范式是大型语言模型成功的基础，能够从海量数据中学习可泛化的表示。然而，将这种范式扩展到图结构上的运筹学问题仍然具有挑战性，因为语言的统计灵活性与图的严格组合约束之间存在根本冲突。

### 目的

为了弥合语言模型与图优化之间的差距，作者引入了图基础模型(GFM)，旨在创建一个能够解决图结构上所有基于距离的优化问题的通用框架。

### 方法

通过在图中随机游走生成的路径上引入类似大型语言模型的自监督预训练范式，使GFM被迫内化图的复杂拓扑和组合规则，其中结构本身的连接性作为监督信号。与学习复杂和特定任务求解策略的现有神经方法不同，该方法利用预训练的GFM作为图内在结构的基础模型，进而使简单的生成启发式方法能够有效处理各种优化挑战。

### 主要发现

在从20到893个节点的网络上的综合实验表明，GFM在各种不同类别的优化任务上与专用求解器具有竞争力的性能，同时保持显著更快的推理时间。

### 结论

这项工作建立了将预训练-迁移框架适应图优化的新范式，为将基础模型创新应用于运筹学领域打开了大门。

### 翻译

预训练-迁移范式作为大型语言模型成功的基础，已展示了从海量数据中学习可泛化表示的基础模型的巨大力量。然而，由于语言的统计灵活性与图的严格组合约束之间的根本冲突，将这种范式扩展到图结构上的运筹学问题仍然具有挑战性。为了弥合这一差距，我们引入了图基础模型(GFM)，这是第一个能够解决图结构上所有基于距离的优化问题的框架。通过在图中随机游走生成的路径上引入类似大型语言模型的自监督预训练范式，GFM被迫内化图的复杂拓扑和组合规则，其中结构本身的连接性可以作为监督信号。与学习复杂和特定任务求解策略的现有神经方法不同，我们的方法利用预训练的GFM作为图内在结构的基础模型，进而使简单的生成启发式方法能够有效处理各种优化挑战。在从20到893个节点的网络上的综合实验表明，GFM在各种不同类别的优化任务上与专用求解器具有竞争力的性能，同时保持显著更快的推理时间。我们的工作建立了将预训练-迁移框架适应图优化的新范式，为将基础模型创新应用于运筹学领域打开了大门。


### 论文摘要

The pretrain-transfer paradigm, which underpins the success of large language models (LLMs), has demonstrated the immense power of creating foundation models that learn generalizable representations from vast datasets. However, extending this paradigm to Operations Research (OR) problems on graph structures remains challenging due to the fundamental conflict between the statistical flexibility of language and the strict combinatorial constraints of graphs. To bridge this gap, we introduce the Graph Foundation Model (GFM), the first framework capable of solving all distance-based optimization problems on graph structures. By introducing the LLM-like self-supervised pre-training paradigm on the paths generated from random walks in the graph, GFM is compelled to internalize the graph's complex topological and combinatorial rules, where the connectivity of the structure itself can be treated as the supervisory signal. Unlike existing neural methods that learn complex and task-specific solving policies, our approach leverages the pre-trained GFM as a foundational model of the graph's intrinsic structure, which in turn enables a simple generative heuristic to tackle a diverse range of optimization challenges effectively. Comprehensive experiments on networks ranging from 20 to 893 nodes demonstrate that GFM achieves competitive performance against specialized solvers across a variety of distinct optimization task classes, while maintaining significantly faster inference times. Our work establishes a new paradigm of adapting the pretrain-transfer framework to graph optimization, opening the door for applying foundation model innovations to OR.

---

## 21. FreeAction: Training-Free Techniques for Enhanced Fidelity of Trajectory-to-Video Generation

**论文链接:** [http://arxiv.org/abs/2509.24241v1](http://arxiv.org/abs/2509.24241v1)

**作者:** Seungwook Kim, Seunghyeon Lee, Minsu Cho

**发布时间:** 2025-09-29

**备注:** 8 pages, 4 figures, accepted to CoRL 2025 LSRW workshop

### GPT解析

### 总结

研究提出了两种训练免费、推理时使用的技术，通过主动利用显式动作参数改进基于扩散的机器人视频生成，提高了动作连贯性和视觉质量。

### 背景

生成逼真的机器人视频是构建有效的世界模型和机器人基础模型的关键步骤。

### 目的

引入两种技术，充分利用基于扩散的机器人视频生成中的显式动作参数，以增强生成视频的质量和可控性。

### 方法

将动作向量作为主动引导而非被动条件信号，包括：1) 动作缩放分类器自由引导，根据动作幅度动态调整引导强度；2) 动作缩放噪声截断，调整初始采样噪声分布以匹配期望的运动动力学。

### 主要发现

在真实机器人操作数据集上的实验表明，这些技术显著提高了各种机器人环境中动作的连贯性和视觉质量。

### 结论

通过主动融入显式动作参数到生成过程中，可以有效改进基于扩散的机器人视频生成，增强动作控制和视觉表现。

### 翻译

从明确的动作轨迹生成逼真的机器人视频是构建有效的世界模型和机器人基础模型的关键步骤。我们引入了两种训练免费、推理时使用的技术，这些技术充分利用基于扩散的机器人视频生成中的显式动作参数。我们不将动作向量视为被动的条件信号，而是主动将它们融入以引导分类器自由引导过程和高斯潜力的初始化。首先，动作缩放分类器自由引导根据动作幅度动态调整引导强度，增强对运动强度的可控性。其次，动作缩放噪声截断调整初始采样噪声的分布，使其更好地与期望的运动动力学保持一致。在真实机器人操作数据集上的实验表明，这些技术显著提高了各种机器人环境中动作的连贯性和视觉质量。


### 论文摘要

Generating realistic robot videos from explicit action trajectories is a critical step toward building effective world models and robotics foundation models. We introduce two training-free, inference-time techniques that fully exploit explicit action parameters in diffusion-based robot video generation. Instead of treating action vectors as passive conditioning signals, our methods actively incorporate them to guide both the classifier-free guidance process and the initialization of Gaussian latents. First, action-scaled classifier-free guidance dynamically modulates guidance strength in proportion to action magnitude, enhancing controllability over motion intensity. Second, action-scaled noise truncation adjusts the distribution of initially sampled noise to better align with the desired motion dynamics. Experiments on real robot manipulation datasets demonstrate that these techniques significantly improve action coherence and visual quality across diverse robot environments.

---

## 22. EVLF-FM: Explainable Vision Language Foundation Model for Medicine

**论文链接:** [http://arxiv.org/abs/2509.24231v1](http://arxiv.org/abs/2509.24231v1)

**作者:** Yang Bai, Haoran Cheng, Yang Zhou, Jun Zhou, Arun Thirunavukarasu, Yuhe Ke, Jie Yao, Kanae Fukutsu, Chrystie Wan Ning Quek, Ashley Hong, Laura Gutierrez, Zhen Ling Teo, Darren Shu Jeng Ting, Brian T. Soetikno, Christopher S. Nielsen, Tobias Elze, Zengxiang Li, Linh Le Dinh, Hiok Hong Chan, Victor Koh, Marcus Tan, Kelvin Z. Li, Leonard Yip, Ching Yu Cheng, Yih Chung Tham, Gavin Siew Wei Tan, Leopold Schmetterer, Marcus Ang, Rahat Hussain, Jod Mehta, Tin Aung, Lionel Tim-Ee Cheng, Tran Nguyen Tuan Anh, Chee Leong Cheng, Tien Yin Wong, Nan Liu, Iain Beehuat Tan, Soon Thye Lim, Eyal Klang, Tony Kiat Hon Lim, Rick Siow Mong Goh, Yong Liu, Daniel Shu Wei Ting

**发布时间:** 2025-09-29

### GPT解析

### 总结

EVLF-FM是一种多模态视觉语言基础模型，通过结合广泛的诊断能力和细粒度可解释性，解决了当前医学AI基础模型模态特定和缺乏透明推理的问题，在多个临床领域表现出色。

### 背景

当前医学AI基础模型存在局限性，它们模态特定且缺乏透明推理过程，阻碍了临床应用。

### 目的

提出EVLF-FM，一个多模态视觉语言基础模型(VLM)，旨在将广泛的诊断能力与细粒度可解释性统一起来。

### 方法

开发和测试覆盖了来自全球23个数据集的130多万个样本，涉及六个临床专业；外部验证使用了8,884个独立测试样本；采用混合训练策略结合监督和视觉强化微调；具备多种疾病诊断和视觉问答能力，具有像素级视觉定位和推理能力。

### 主要发现

在内部疾病诊断验证中实现了最高的平均准确率(0.858)和F1分数(0.797)；在医学视觉定位方面表现优异，平均mIOU为0.743，Acc@0.5为0.837；外部验证证实了强大的零样本和少样本性能；能够实现逐步推理，使输出与视觉证据保持一致。

### 结论

EVLF-FM是首批具有可解释性和推理能力的多疾病VLM模型，有可能促进基础模型在真实世界临床部署中的采用和信任。

### 翻译

尽管基础模型在医学AI中前景广阔，但当前系统仍然存在局限性——它们模态特定且缺乏透明推理过程，阻碍了临床应用。为解决这一差距，我们提出了EVLF-FM，这是一种多模态视觉语言基础模型(VLM)，旨在将广泛的诊断能力与细粒度可解释性统一起来。EVLF-FM的开发和测试涵盖了来自全球23个数据集的130多万个样本，涉及六个临床专业：皮肤科、肝病学、眼科学、病理学、肺病学和放射学的11种成像模态。外部验证使用了来自10个额外数据集的8,884个独立测试样本，涵盖五种成像模态。技术上，EVLF-FM被开发用于协助多种疾病诊断和视觉问答，具有像素级视觉定位和推理能力。在内部疾病诊断验证中，EVLF-FM实现了最高的平均准确率(0.858)和F1分数(0.797)，优于领先的全能和专科模型。在医学视觉定位方面，EVLF-FM在九种模态上也表现出色，平均mIOU为0.743，Acc@0.5为0.837。外部验证进一步证实了强大的零样本和少样本性能，尽管模型尺寸较小，但具有竞争力的F1分数。通过结合监督和视觉强化微调的混合训练策略，EVLF-FM不仅实现了最先进的准确率，还展示了逐步推理能力，使输出与视觉证据保持一致。EVLF-FM是首批具有可解释性和推理能力的多疾病VLM模型，可以促进基础模型在真实世界临床部署中的采用和信任。


### 论文摘要

Despite the promise of foundation models in medical AI, current systems remain limited - they are modality-specific and lack transparent reasoning processes, hindering clinical adoption. To address this gap, we present EVLF-FM, a multimodal vision-language foundation model (VLM) designed to unify broad diagnostic capability with fine-grain explainability. The development and testing of EVLF-FM encompassed over 1.3 million total samples from 23 global datasets across eleven imaging modalities related to six clinical specialties: dermatology, hepatology, ophthalmology, pathology, pulmonology, and radiology. External validation employed 8,884 independent test samples from 10 additional datasets across five imaging modalities. Technically, EVLF-FM is developed to assist with multiple disease diagnosis and visual question answering with pixel-level visual grounding and reasoning capabilities. In internal validation for disease diagnostics, EVLF-FM achieved the highest average accuracy (0.858) and F1-score (0.797), outperforming leading generalist and specialist models. In medical visual grounding, EVLF-FM also achieved stellar performance across nine modalities with average mIOU of 0.743 and Acc@0.5 of 0.837. External validations further confirmed strong zero-shot and few-shot performance, with competitive F1-scores despite a smaller model size. Through a hybrid training strategy combining supervised and visual reinforcement fine-tuning, EVLF-FM not only achieves state-of-the-art accuracy but also exhibits step-by-step reasoning, aligning outputs with visual evidence. EVLF-FM is an early multi-disease VLM model with explainability and reasoning capabilities that could advance adoption of and trust in foundation models for real-world clinical deployment.

---

## 23. Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.24222v1](http://arxiv.org/abs/2509.24222v1)

**作者:** Zhisheng Chen, Yingwei Zhang, Qizhen Lan, Tianyu Liu, Huacan Wang, Yi Ding, Ziyu Jia, Ronghao Chen, Kun Wang, Xinliang Zhou

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种统一神经拓扑基础模型(Uni-NTFM)，解决了现有脑基础模型在处理EEG信号时的三个主要局限性，通过创新的架构设计实现了对脑电图信号更有效的表示学习。

### 背景

基础模型在各种无标签数据上的预训练在自然语言和视觉领域取得了显著成功，但将其应用于脑电图(EEG)仍面临挑战。现有的脑基础模型继承了为文本或图像设计的架构，导致三个主要局限性：1)将时域波形模式与频域节律特征在单一处理流中混合；2)忽略不同标准电极的关键空间拓扑；3)依赖不灵活的密集网络处理功能不同的EEG模式。

### 目的

设计一个基于神经科学原理的基础模型，解决现有方法在EEG处理中的局限性，产生通用且可解释的脑电信号表示。

### 方法

引入Uni-NTFM模型，包含三个核心创新：1)解耦架构并行编码时间、频率和原始信号表示，然后进行跨域特征集成；2)拓扑嵌入机制统一不同国际标准的电极，为脑区生成结构化输入序列；3)专家混合神经Transformer通过将信号模式路由到专门子网络来高效扩展模型容量。最大模型Uni-NTFM_large有19亿参数，在超过28,000小时的多样化EEG数据上进行了预训练。

### 主要发现

Uni-NTFM在九个不同的下游任务上显著优于现有的任务特定方法和基础模型，在线性探测和微调设置下都表现出优越性能，展示了学习脑活动通用表示的卓越能力。

### 结论

Uni-NTFM通过创新的架构设计成功解决了现有脑基础模型在EEG应用中的局限性，实现了对脑电图信号更有效的表示学习，为脑电信号分析提供了新的基础模型框架。

### 翻译

在各种无标签数据上预训练的基础模型在自然语言和视觉领域已显示出显著成功，但它们在脑电图(EEG)中的应用仍因信号的独特特性而面临挑战。现有的脑基础模型继承了为文本或图像设计的架构，导致预训练中存在三个局限性：1)在单一处理流中混合时域波形模式和频域节律特征；2)忽略不同标准电极的关键空间拓扑；3)依赖不灵活的密集网络来处理功能上不同的EEG模式。为应对这些挑战，我们引入了统一神经拓扑基础模型(Uni-NTFM)，该模型基于神经科学原理设计，能够产生通用和可解释的表示。Uni-NTFM整合了三个核心创新：1)解耦架构并行编码时间、频率和原始信号表示，然后进行跨域特征集成；2)拓扑嵌入机制统一不同国际标准的电极，并为脑区生成结构化输入序列；3)专家混合神经Transformer通过将信号模式路由到专门子网络来高效扩展模型容量。最大的模型Uni-NTFM_large有创纪录的19亿参数，并通过双域掩码重建目标在超过28,000小时的多样化EEG数据上进行了预训练。在九个不同的下游任务上，Uni-NTFM在线性探测和微调设置下均显著优于现有的任务特定方法和基础模型，展示了学习脑活动通用表示的卓越能力。


### 论文摘要

Foundation models pretrained on various and unlabeled data have demonstrated significant success in natural language and vision, but their application to electroencephalography (EEG) remains challenged due to the signal's unique properties. Existing brain foundation models that inherit architectures designed for text or images lead to three limitations in pre-training: 1) conflating time-domain waveform patterns with frequency-domain rhythmic features in a single processing stream, 2) ignoring the critical spatial topology of electrodes with different standards, and 3) reliance on the inflexible, dense network to process functionally distinct EEG patterns. To address these challenges, we introduce the Unified Neural Topological Foundation Model (Uni-NTFM), which is designed based on neuroscience principles to produce universal and interpretable representations. Uni-NTFM integrates three core innovations: 1) a decoupled architecture parallelly encodes time, frequency, and raw signal representations before performing cross-domain feature integration; 2) a topological embedding mechanism to unify electrodes from different international standards and generate structured input sequences for brain regions; and 3) a Mixture-of-Experts neural Transformer that efficiently scales model capacity by routing signal patterns to specialized subnetworks. The largest model, Uni-NTFM$_{large}$, has a record-breaking 1.9B parameters and was pretrained on over 28,000 hours of diverse EEG data via a dual-domain masked reconstruction objective. Uni-NTFM significantly outperforms existing task-specific methods and foundation models across nine distinct downstream tasks under both linear probing and fine-tuning settings, demonstrating a superior ability to learn universal representations of brain activity.

---

## 24. BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2509.24204v1](http://arxiv.org/abs/2509.24204v1)

**作者:** Zelin Liu, Sicheng Dong, Bocheng Li, Yixuan Yang, Jiacheng Ruan, Chenxu Zhou, Suncheng Xiang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了BALR-SAM框架，一种边界感知的低秩适应方法，用于增强Segment Anything Model在医学图像分割领域的应用，实现了高性能与低资源消耗的平衡。

### 背景

视觉基础模型（如SAM）在大规模自然图像数据集上预训练后，在医学图像分割任务中表现不佳，因为缺乏领域特定的适应。在临床实践中，高效微调这些模型以满足医学下游任务的需求同时保持高性能具有挑战性。

### 目的

解决视觉基础模型在医学图像分割中的适应性问题，实现高效微调，减少资源需求，同时保持强大的性能。

### 方法

提出BALR-SAM框架，包含三个定制组件：(1)互补细节增强网络（CDEN）使用深度可分离卷积和多尺度融合捕获边界敏感特征；(2)低秩适配器集成到SAM的Vision Transformer块中，优化医学场景的特征表示和注意力，同时减少参数空间；(3)低秩张量注意力机制在掩码解码器中，减少75%内存使用并提高推理速度。

### 主要发现

在标准医学分割数据集上的实验表明，BALR-SAM不需要提示就优于包括完全微调的MedSAM在内的几种最先进方法，同时仅更新1.8%（11.7M）的参数。

### 结论

BALR-SAM是一种有效的框架，通过低秩适应和边界感知方法，成功增强了SAM在医学成像中的应用，实现了高性能和低资源消耗的平衡。

### 翻译

视觉基础模型如Segment Anything Model在大规模自然图像数据集上预训练后，常因缺乏领域特定适应而在医学图像分割中表现不佳。在临床实践中，高效微调此类模型以满足医学下游任务需求同时保持强大性能具有挑战性。为解决这些问题，我们提出了BALR-SAM，一种边界感知的低秩适应框架，用于增强SAM在医学成像中的应用。它结合了三个定制组件：(1)使用深度可分离卷积和多尺度融合的互补细节增强网络，捕获对准确分割至关重要的边界敏感特征；(2)集成到SAM Vision Transformer块中的低秩适配器，优化医学场景的特征表示和注意力，同时显著减少参数空间；(3)掩码解码器中的低秩张量注意力机制，将内存使用减少75%并提高推理速度。在标准医学分割数据集上的实验表明，BALR-SAM无需提示即可优于包括完全微调的MedSAM在内的几种最先进方法，同时仅更新其1.8%（11.7M）的参数。


### 论文摘要

Vision foundation models like the Segment Anything Model (SAM), pretrained on large-scale natural image datasets, often struggle in medical image segmentation due to a lack of domain-specific adaptation. In clinical practice, fine-tuning such models efficiently for medical downstream tasks with minimal resource demands, while maintaining strong performance, is challenging. To address these issues, we propose BALR-SAM, a boundary-aware low-rank adaptation framework that enhances SAM for medical imaging. It combines three tailored components: (1) a Complementary Detail Enhancement Network (CDEN) using depthwise separable convolutions and multi-scale fusion to capture boundary-sensitive features essential for accurate segmentation; (2) low-rank adapters integrated into SAM's Vision Transformer blocks to optimize feature representation and attention for medical contexts, while simultaneously significantly reducing the parameter space; and (3) a low-rank tensor attention mechanism in the mask decoder, cutting memory usage by 75% and boosting inference speed. Experiments on standard medical segmentation datasets show that BALR-SAM, without requiring prompts, outperforms several state-of-the-art (SOTA) methods, including fully fine-tuned MedSAM, while updating just 1.8% (11.7M) of its parameters.

---

## 25. FM-FoG: A Real-Time Foundation Model-based Wearable System for Freezing-of-Gait Mitigation

**论文链接:** [http://arxiv.org/abs/2509.24176v1](http://arxiv.org/abs/2509.24176v1)

**作者:** Chuntian Chi, John Clapham, Leslie Cloud, Ingrid Pretzer-Aboff, GinaMari Blackwell, Huajie Shao, Gang Zhou

**发布时间:** 2025-09-29

**备注:** This is a preprint version, 12 pages, 7 figures, 8 tables

### GPT解析

### 总结

FM-FoG是一种基于基础模型的实时可穿戴系统，能够在无需患者特定训练的情况下检测帕金森病患者的步态冻结(FoG)，在未见患者上达到98.5%的F1分数，同时延长电池寿命72%并保持低延迟。

### 背景

步态冻结(FoG)影响超过50%的中晚期帕金森病患者，显著损害行动独立性并降低生活质量。FoG表现为突然发作的步行无法开始或中断，仅发生在站立或行走时，不会在坐或躺时发生。

### 目的

开发一种能够在无需患者特定训练的情况下，实现对未见患者FoG检测的实时系统，解决当前系统需要大量患者特定训练数据和缺乏泛化能力的问题。

### 方法

结合在多样化惯性测量单元(IMU)数据集上进行自监督预训练，并整合传感器上下文信息。使用轻量级CNN-LSTM活动分类器仅在行走或站立时激活基础模型，避免不必要的计算。

### 主要发现

在包含23名PD患者的VCU FoG-IMU数据集上测试，FM-FoG在未见患者上达到98.5%的F1分数，显著优于竞争性基线方法。在Google Pixel 8a智能手机上部署时，系统将电池寿命延长了高达72%，同时保持低于20毫秒的干预延迟。

### 结论

FM-FoG能够实现实用、节能的医疗保健应用，无需个体训练要求即可跨患者泛化。

### 翻译

步态冻结(FoG)影响超过50%的中晚期帕金森病患者，显著损害患者的行动独立性并降低生活质量。FoG的特点是突然发作，步行无法开始或中断，仅发生在站立或行走时，而不会在坐或躺时发生。当前的FoG检测系统需要大量的患者特定训练数据，且缺乏泛化能力，限制了临床应用。为解决这些问题，我们引入了FM-FoG，一个基于基础模型的实时可穿戴系统，能够在无需患者特定训练的情况下，实现对未见患者的FoG检测。我们的方法结合了在多样化的惯性测量单元(IMU)数据集上进行自监督预训练，并整合了传感器上下文信息。由于FoG仅发生在活动状态下，使用轻量级的CNN-LSTM活动分类器仅在行走或站立时激活基础模型，避免不必要的计算。在包含23名PD患者的VCU FoG-IMU数据集上测试，当测试在之前未见过的患者上时，FM-FoG达到了98.5%的F1分数，显著优于竞争性基线方法。在Google Pixel 8a智能手机上部署时，系统将电池寿命延长了高达72%，同时保持低于20毫秒的干预延迟。结果表明，我们的FM-FoG能够实现实用、节能的医疗保健应用，无需个体训练要求即可跨患者泛化。


### 论文摘要

Freezing-of-Gait (FoG) affects over 50% of mid-to-late stage Parkinson's disease (PD) patients, significantly impairing patients' mobility independence and reducing quality of life. FoG is characterized by sudden episodes where walking cannot start or is interrupted, occurring exclusively during standing or walking, and never while sitting or lying down. Current FoG detection systems require extensive patient-specific training data and lack generalization, limiting clinical deployment. To address these issues, we introduce FM-FoG, a real-time foundation model-based wearable system achieving FoG detection in unseen patients without patient-specific training. Our approach combines self-supervised pretraining on diverse Inertial Measurement Unit (IMU) datasets with sensor context integration. Since FoG occurs only during ambulatory activities, a lightweight CNN-LSTM activity classifier selectively activates the foundation model only during walking or standing, avoiding unnecessary computation. Evaluated on the VCU FoG-IMU dataset with 23 PD patients, FM-FoG achieves a 98.5% F1-score when tested on previously unseen patients, substantially outperforming competitive baseline methods. Deployed on a Google Pixel 8a smartphone, the system extends battery life by up to 72% while maintaining sub-20ms intervention latency. The results indicate that our FM-FoG can enable practical, energy-efficient healthcare applications that generalize across patients without individual training requirements.

---

## 26. GPS-MTM: Capturing Pattern of Normalcy in GPS-Trajectories with self-supervised learning

**论文链接:** [http://arxiv.org/abs/2509.24031v1](http://arxiv.org/abs/2509.24031v1)

**作者:** Umang Garg, Bowen Zhang, Anantanjit Subrahmanya, Chandrakanth Gudavalli, BS Manjunath

**发布时间:** 2025-09-28

**备注:** 4 pages, 2 figures

### GPT解析

### 总结

GPS-MTM是一种用于大规模移动数据的基础模型，通过分解移动性为状态和动作两种互补模式，利用双向Transformer和自监督掩码建模目标，无需人工标签即可学习丰富的语义相关性，在多个基准数据集的下游任务上表现优异。

### 背景

Foundation models在文本、视觉和视频理解方面已取得显著进展，现在有望在轨迹建模方面实现类似突破。

### 目的

介绍GPSMasked Trajectory Transformer (GPS-MTM)，一个捕捉人类运动正常模式的基础模型。

### 方法

GPS-MTM将移动性分解为状态（兴趣点类别）和动作（代理转换）两种互补模式，利用双向Transformer和自监督掩码建模目标重建跨模态的缺失片段。

### 主要发现

在Numosim-LA、Urban Anomalies和Geolife等基准数据集上，GPS-MTM在轨迹填补和下一站预测等下游任务上始终优于先前方法，在需要上下文推理的动态任务中优势最为明显。

### 结论

GPS-MTM确立了作为轨迹分析的强大基础模型，将移动数据定位为大规模表示学习的一流模态。

### 翻译

Foundation models已在文本、视觉和视频理解方面推动了显著进展，现在有望在轨迹建模方面解锁类似突破。我们引入了GPSMasked Trajectory Transformer (GPS-MTM)，这是一个用于大规模移动数据的基础模型，能够捕捉人类运动的正常模式。与先前将轨迹展平为坐标流的方法不同，GPS-MTM将移动性分解为两种互补模式：状态（兴趣点类别）和动作（代理转换）。利用具有自监督掩码建模目标的双向Transformer，模型能够重建跨模态的缺失片段，使其无需人工标签即可学习丰富的语义相关性。在包括Numosim-LA、Urban Anomalies和Geolife在内的基准数据集上，GPS-MTM在轨迹填补和下一站预测等下游任务上始终表现更优。其在需要上下文推理的动态任务（逆向和正向动力学）中优势最为明显。这些结果确立了GPS-MTM作为轨迹分析的强大基础模型，将移动数据定位为大规模表示学习的一流模态。代码已发布供进一步参考。


### 论文摘要

Foundation models have driven remarkable progress in text, vision, and video understanding, and are now poised to unlock similar breakthroughs in trajectory modeling. We introduce the GPSMasked Trajectory Transformer (GPS-MTM), a foundation model for large-scale mobility data that captures patterns of normalcy in human movement. Unlike prior approaches that flatten trajectories into coordinate streams, GPS-MTM decomposes mobility into two complementary modalities: states (point-of-interest categories) and actions (agent transitions). Leveraging a bi-directional Transformer with a self-supervised masked modeling objective, the model reconstructs missing segments across modalities, enabling it to learn rich semantic correlations without manual labels. Across benchmark datasets, including Numosim-LA, Urban Anomalies, and Geolife, GPS-MTM consistently outperforms on downstream tasks such as trajectory infilling and next-stop prediction. Its advantages are most pronounced in dynamic tasks (inverse and forward dynamics), where contextual reasoning is critical. These results establish GPS-MTM as a robust foundation model for trajectory analytics, positioning mobility data as a first-class modality for large-scale representation learning. Code is released for further reference.

---

## 27. Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2509.23993v1](http://arxiv.org/abs/2509.23993v1)

**作者:** Muleilan Pei, Shaoshuai Shi, Shaojie Shen

**发布时间:** 2025-09-28

### GPT解析

### 总结

SMART-R1是一种针对多智能体交通行为模拟的新型R1风格强化微调范式，通过面向指标的策略优化和交替训练策略解决了数据驱动模拟器中的分布偏移问题，在Waymo开放智能体挑战赛中排名第一。

### 背景

多智能体交通行为的可扩展且真实的模拟对推进自动驾驶技术至关重要，现有数据驱动模拟器虽取得进展，但主要依赖监督学习，面临训练和测试间的分布偏移问题，损害模型泛化能力。

### 目的

提出SMART-R1，一种新型的R1风格强化微调范式，使智能体行为更好地与人类偏好和评估指标保持一致，解决分布偏移问题，提高模型在未见环境中的泛化能力。

### 方法

引入面向指标的策略优化算法改善分布对齐；采用迭代式'SFT-RFT-SFT'训练策略，交替进行监督微调(SFT)和强化微调(RFT)；在大型Waymo开放运动数据集(WOMD)上进行广泛实验验证。

### 主要发现

SMART-R1在Waymo开放智能体挑战赛(WOSAC)上取得最先进性能，整体真实感元得分为0.7858，在提交时排名第一。

### 结论

R1风格的训练框架简单而强大，能有效增强基础模型；SMART-R1方法在提高多智能体交通行为模拟方面表现出色。

### 翻译

多智能体交通行为的可扩展且真实的模拟对于推进自动驾驶技术至关重要。尽管现有的数据驱动模拟器在该领域已取得显著进展，但它们主要依赖监督学习来使模拟分布与真实驾驶场景保持一致。然而，一个持续的挑战存在于训练和测试之间出现的分布偏移，这常常损害模型在未见环境中的泛化能力。为解决这一局限，我们提出了SMART-R1，一种专门为下一预测模型设计的R1风格强化微调范式，以更好地使智能体行为与人类偏好和评估指标保持一致。我们的方法引入了面向指标的策略优化算法来改善分布对齐，以及一种迭代式的'SFT-RFT-SFT'训练策略，交替进行监督微调(SFT)和强化微调(RFT)，以最大化性能提升。在大型Waymo开放运动数据集(WOMD)上的广泛实验验证了这种简单而强大的R1风格训练框架在增强基础模型方面的有效性。在Waymo开放智能体挑战赛(WOSAC)上的结果显示，SMART-R1取得了最先进的性能，整体真实感元得分为0.7858，在提交时排名第一。


### 论文摘要

Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission.

---

## 28. RPG360: Robust 360 Depth Estimation with Perspective Foundation Models and Graph Optimization

**论文链接:** [http://arxiv.org/abs/2509.23991v1](http://arxiv.org/abs/2509.23991v1)

**作者:** Dongki Jung, Jaehoon Choi, Yonghan Lee, Dinesh Manocha

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为RPG360的无需训练的鲁棒360度单目深度估计方法，该方法结合透视基础模型和图优化技术，能够有效处理全景图像的深度估计问题。

### 背景

随着360度图像在各领域的应用日益增加，对全景图像的鲁棒深度估计技术需求迫切。然而，获取大规模标记的360度深度估计数据集仍然是一个重大挑战。

### 目的

开发一种无需训练的鲁棒360度单目深度估计方法，解决全景图像深度估计中的尺度不一致问题，并保持3D结构完整性。

### 方法

RPG360方法将360度图像转换为六面体立方图表示，使用透视基础模型估计深度和表面法线，并通过基于图的优化技术引入深度尺度对齐，解决不同面之间的深度尺度不一致问题。优化过程参数化预测的深度和法线图，包含每个面的额外尺度参数，确保深度尺度一致性和3D结构完整性。

### 主要发现

基础模型在零样本设置中表现出固有鲁棒性，该方法在Matterport3D、Stanford2D3D和360Loc等多个数据集上实现了优越性能。在下游任务中，特征匹配提升了3.2%~5.4%，运动结构重建在AUC@5指标上提升了0.2%~9.7%。

### 结论

RPG360结合透视基础模型和图优化技术有效解决了全景图像深度估计中的挑战，在多个数据集和下游任务中表现出色，是一种无需训练的360度单目深度估计的有效方法。

### 翻译

随着360度图像在各领域的应用日益增加，对全景图像的鲁棒深度估计技术的需求日益凸显。然而，获取大规模标记的360度深度估计数据集仍然是一个重大挑战。在本文中，我们提出了RPG360，一种无需训练的鲁棒360度单目深度估计方法，它利用透视基础模型和图优化。我们的方法将360度图像转换为六面体立方图表示，其中使用透视基础模型来估计深度和表面法线。为了解决立方图不同面之间的深度尺度不一致问题，我们引入了一种使用基于图优化的新颖深度尺度对齐技术，该技术参数化预测的深度和法线图，同时包含每个面的额外尺度参数。这种优化确保了六面体立方图之间的深度尺度一致性，同时保持3D结构完整性。此外，由于基础模型在零样本设置中表现出固有的鲁棒性，我们的方法在包括Matterport3D、Stanford2D3D和360Loc在内的多个数据集上实现了卓越的性能。我们还通过验证其在特征匹配提升3.2%~5.4%和运动结构重建在AUC@5上提升0.2%~9.7%等下游任务中的优势，证明了我们深度估计方法的多功能性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决360度图像（全景图像）的深度估计问题。这个问题在现实中非常重要，因为360度图像在机器人导航、虚拟现实、自动驾驶和沉浸式媒体等领域有广泛应用，而深度信息对于这些应用中的场景理解、障碍物检测和3D重建等任务至关重要。然而，现有的深度估计方法要么依赖于大量标注数据（成本高），要么在3D结构一致性方面表现不佳，限制了实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：监督学习方法需要大量标注数据，而优化方法在3D结构一致性方面表现不佳。作者注意到透视基础模型在零样本场景下表现出色，思考如何将这些模型应用于360度图像。作者将360度图像转换为六面体立方图表示，使用透视基础模型估计每个面的深度和法线，然后引入基于图的优化方法解决不同视角间的深度尺度不一致问题。该方法借鉴了透视基础模型、立方图投影技术、图优化技术和单目深度估计中的尺度模糊性解决方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用透视基础模型在零样本场景下的强大能力，结合基于图的优化方法解决360度图像深度估计中的深度尺度不一致问题。整体流程包括：1) 将输入的等距柱状投影图像转换为六面体立方图表示；2) 使用透视基础模型估计每个面的深度图和表面法线图；3) 将估计的深度图和法线图合并回ERP空间；4) 通过图优化参数化预测的深度图和法线图，添加每个面的尺度参数，确保深度尺度一致性和3D结构完整性；5) 输出尺度一致的ERP深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出无需训练的360度单目深度估计方法，解决了缺乏大规模标注数据的问题；2) 引入基于图的优化方法，结合深度和法线信息解决深度尺度不一致；3) 提出每个面的尺度参数用于全局尺度对齐；4) 在多种数据集上表现出色且在零样本场景下稳定。相比之前的工作，该方法不需要大量标注数据（区别于监督学习方法），显式考虑3D结构一致性（区别于之前的优化方法如360MonoDepth），不依赖于伪标签生成（区别于Depth Anywhere），在3D评估指标上表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RPG360通过结合透视基础模型和基于图的优化方法，实现了无需训练的、具有3D结构意识的360度图像深度估计，在多种数据集上表现出色并提升了下游任务的性能。'}


### 论文摘要

The increasing use of 360 images across various domains has emphasized the need for robust depth estimation techniques tailored for omnidirectional images. However, obtaining large-scale labeled datasets for 360 depth estimation remains a significant challenge. In this paper, we propose RPG360, a training-free robust 360 monocular depth estimation method that leverages perspective foundation models and graph optimization. Our approach converts 360 images into six-face cubemap representations, where a perspective foundation model is employed to estimate depth and surface normals. To address depth scale inconsistencies across different faces of the cubemap, we introduce a novel depth scale alignment technique using graph-based optimization, which parameterizes the predicted depth and normal maps while incorporating an additional per-face scale parameter. This optimization ensures depth scale consistency across the six-face cubemap while preserving 3D structural integrity. Furthermore, as foundation models exhibit inherent robustness in zero-shot settings, our method achieves superior performance across diverse datasets, including Matterport3D, Stanford2D3D, and 360Loc. We also demonstrate the versatility of our depth estimation approach by validating its benefits in downstream tasks such as feature matching 3.2 ~ 5.4% and Structure from Motion 0.2 ~ 9.7% in AUC@5.

---

## 29. HunyuanImage 3.0 Technical Report

**论文链接:** [http://arxiv.org/abs/2509.23951v1](http://arxiv.org/abs/2509.23951v1)

**作者:** Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lucaz Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Miles Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, Zhao Zhong

**发布时间:** 2025-09-28

### GPT解析

### 总结

HunyuanImage 3.0是一个原生多模态模型，在自回归框架内统一了多模态理解和生成功能，其图像生成模块已公开可用。该模型包含超过800亿参数的专家混合模型，每推理激活130亿参数，是目前最大且功能最强大的开源图像生成模型。

### 背景

多模态AI模型的发展是当前AI领域的重要研究方向，特别是图像生成领域需要更强大、更高效的基础模型。

### 目的

开发一个统一多模态理解和生成的原生模型，并通过开源方式促进多模态AI生态系统的发展。

### 方法

通过精心策划的数据、先进的架构设计、原生思维链模式、渐进式模型预训练、积极的模型后训练以及高效的基础设施实现，成功训练了包含超过800亿参数的专家混合模型。

### 主要发现

模型在文本-图像对齐和视觉质量方面的评估结果与最先进的模型相当；每推理激活130亿参数，成为目前最大且功能最强大的开源图像生成模型。

### 结论

通过发布HunyuanImage 3.0的代码和权重，使社区能够基于这一最先进的基础模型探索新想法，促进动态且充满活力的多模态生态系统的形成。

### 翻译

我们提出了HunyuanImage 3.0，这是一个原生多模态模型，在自回归框架内统一了多模态理解和生成功能，其图像生成模块已公开可用。HunyuanImage 3.0的成就依赖于几个关键组件，包括精心策划的数据、先进的架构设计、原生思维链模式、渐进式模型预训练、积极的模型后训练，以及能够支持大规模训练和推理的高效基础设施。通过这些进步，我们成功训练了一个包含超过800亿参数的专家混合模型，推理时每个token激活130亿参数，使其成为迄今为止最大且功能最强大的开源图像生成模型。我们进行了广泛的实验，文本-图像对齐和视觉质量的自动和人工评估结果表明，HunyuanImage 3.0可与之前的最先进模型相媲美。通过发布HunyuanImage 3.0的代码和权重，我们旨在使社区能够利用这一最先进的基础模型探索新想法，促进动态且充满活力的多模态生态系统。所有开源资源均可通过https://github.com/Tencent-Hunyuan/HunyuanImage-3.0公开获取。


### 论文摘要

We present HunyuanImage 3.0, a native multimodal model that unifies multimodal understanding and generation within an autoregressive framework, with its image generation module publicly available. The achievement of HunyuanImage 3.0 relies on several key components, including meticulous data curation, advanced architecture design, a native Chain-of-Thoughts schema, progressive model pre-training, aggressive model post-training, and an efficient infrastructure that enables large-scale training and inference. With these advancements, we successfully trained a Mixture-of-Experts (MoE) model comprising over 80 billion parameters in total, with 13 billion parameters activated per token during inference, making it the largest and most powerful open-source image generative model to date. We conducted extensive experiments and the results of automatic and human evaluation of text-image alignment and visual quality demonstrate that HunyuanImage 3.0 rivals previous state-of-the-art models. By releasing the code and weights of HunyuanImage 3.0, we aim to enable the community to explore new ideas with a state-of-the-art foundation model, fostering a dynamic and vibrant multimodal ecosystem. All open source assets are publicly available at https://github.com/Tencent-Hunyuan/HunyuanImage-3.0

---

## 30. SAR-KnowLIP: Towards Multimodal Foundation Models for Remote Sensing

**论文链接:** [http://arxiv.org/abs/2509.23927v1](http://arxiv.org/abs/2509.23927v1)

**作者:** Yi Yang, Xiaokun Zhang, Qingchen Fang, Ziqi Ye, Rui Li, Li Liu, Haipeng Wang

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出SAR-KnowLIP，首个通用SAR多模态基础模型，以及相关数据集和评估基线，填补了SAR影像建模的研究空白，在多个下游任务中表现优异。

### 背景

跨模态人工智能在自然图像研究中取得显著进展，但现有方法大多针对RGB影像，在合成孔径雷达(SAR)影像建模方面存在明显差距。SAR具有全天候、全天时的成像能力，在遥感场景理解中发挥着不可替代的作用。

### 目的

填补SAR影像建模的研究空白，提出首个通用的SAR多模态基础模型SAR-KnowLIP，以及可重用的数据和评估基线。

### 方法

1) 构建SAR-GEOVL-1M数据集，首个具有完整地理投影属性的大规模SAR数据集；2) 通过分层认知思维链生成对齐的结构化文本，提供多维语义标注；3) 设计自洽迭代优化机制，通过自监督闭环增强跨模态对齐；4) 建立统一评估基准，与14个领先基础模型进行比较。

### 主要发现

SAR-KnowLIP在11个代表性下游任务上表现出领先性能，特别是在目标计数和土地覆盖分类方面。

### 结论

SAR-KnowLIP的大规模多模态数据、可迁移模型架构和全面实验基准将显著推动SAR多模态基线模型的发展。

### 翻译

跨模态人工智能近年来受到广泛关注，在自然图像研究中取得了显著进展。然而，现有方法大多是为RGB图像设计的，在合成孔径雷达(SAR)图像建模方面存在明显差距。SAR凭借其全天、全天候的成像能力，在遥感场景理解中发挥着不可替代的作用。为填补这一空白，本文提出了SAR-KnowLIP，首个通用的SAR多模态基础模型，以及可重用的数据和评估基线。具体而言：(1) 本研究引入了遥感研究中关键但长期被忽视的地理信息属性，构建了SAR-GEOVL-1M（首个具有完整地理投影属性的大规模SAR数据集），涵盖多个卫星平台、12万张图像和135个城市。(2) 通过分层认知思维链(HCoT)生成对齐的结构化文本，提供超过100万个关于地形、区域功能、目标属性和空间关系的多维语义标注。(3) 我们设计了一种自洽迭代优化机制，通过在可迁移多模态编码器上进行对比学习、匹配学习和重建学习的自监督闭环，持续增强跨模态对齐。(4) 在11个代表性的下游视觉和视觉语言任务上建立了统一评估基准，与14个领先基础模型进行比较，SAR-KnowLIP表现出领先性能，特别是在目标计数和土地覆盖分类方面。我们期望SAR-KnowLIP的大规模多模态数据、可迁移模型架构和全面实验基准将显著推动SAR多模态基线模型的发展。


### 论文摘要

Cross-modal artificial intelligence has garnered widespread attention in recent years, achieving significant progress in the study of natural images. However, existing methods are mostly designed for RGB imagery, leaving a significant gap in modeling synthetic aperture radar (SAR) imagery. SAR, with its all-day, all-weather imaging capabilities, plays an irreplaceable role in remote sensing scene understanding. To address this gap, this paper proposes SAR-KnowLIP, the first universal SAR multimodal foundational model, along with reusable data and evaluation baselines. Specifically: (1) This work introduces the critical yet long-overlooked attribute of geographic information into remote sensing research, constructing SAR-GEOVL-1M (the first large-scale SAR dataset with complete geographic projection properties), covering multiple satellite platforms, 120,000 images, and 135 cities. (2) Aligned structured text is generated through a hierarchical cognitive chain-of-thought (HCoT), providing more than one million multi-dimensional semantic annotations of landforms, regional functions, target attributes, and spatial relationships. (3) We design a Self-Consistent Iterative Optimization mechanism that continuously enhances cross-modal alignment through a self-supervised closed loop of contrastive, matching, and reconstruction learning on a transferable multimodal encoder. (4) A unified evaluation benchmark is established across 11 representative downstream vision and vision-language tasks, with comparisons against 14 leading foundation models, where SAR-KnowLIP demonstrates leading performance, particularly in object counting and land-cover classification. We expect that SAR-KnowLIP's large-scale multimodal data, transferable model architecture, and comprehensive experimental benchmark will significantly advance the development of SAR multimodal baseline models.

---

## 31. Accelerating Dynamic Image Graph Construction on FPGA for Vision GNNs

**论文链接:** [http://arxiv.org/abs/2509.25121v1](http://arxiv.org/abs/2509.25121v1)

**作者:** Anvitha Ramachandran, Dhruv Parikh, Viktor Prasanna

**发布时间:** 2025-09-29

**备注:** IEEE HPEC 2025

### GPT解析

### 总结

这篇论文提出了一种用于动态图像图构建(DIGC)的FPGA加速器，解决了Vision Graph Neural Networks (ViGs)中的主要计算瓶颈问题。

### 背景

Vision Graph Neural Networks (ViGs)将图像表示为非结构化图，在计算机视觉任务中表现出色。DIGC通过基于特征相似性连接补丁来构建图像图，但在ViG推理中构成了50%以上的延迟，在高分辨率下高达95%，成为主要计算瓶颈。先前的工作主要从算法上优化图构建，但往往牺牲了DIGC的灵活性、准确性或通用性。

### 目的

开发一种高效、灵活且通用的硬件加速器来解决DIGC的计算瓶颈问题，而不损害其灵活性、准确性或通用性。

### 方法

提出一种流式、深度流水线FPGA加速器，具有片上缓冲器处理小而均匀的输入特征块。通过局部化计算最小化外部内存流量，使用堆插入在流式输入块上执行局部合并排序和全局k路合并。该模块化架构设计可无缝扩展到不同图像分辨率、ViG层类型、模型大小及变体。

### 主要发现

所设计的FPGA加速器在布局布线后实现了高时钟频率，相比优化的CPU和GPU DIGC基线，分别实现了高达16.6倍和6.8倍的加速。该设计支持各种基于ViG的视觉主干网络中的DIGC。

### 结论

通过硬件加速而非仅依赖算法优化，可以有效解决DIGC的计算瓶颈问题，同时保持其灵活性、准确性和通用性。

### 翻译

Vision Graph Neural Networks (Vision GNNs，或ViGs)将图像表示为非结构化图，在图像分类、目标检测和实例分割等计算机视觉任务中实现了最先进的性能。动态图像图构建(DIGC)通过基于特征相似性连接补丁(节点)来构建图像图，并在每个ViG层中基于GNN的补丁(节点)特征更新后动态重复进行。然而，DIGC构成了端到端ViG推理延迟的50%以上，在高图像分辨率下高达95%，使其成为主要的计算瓶颈。虽然硬件加速很有前景，但先前的工作主要从算法上优化图构建，常常以DIGC的灵活性、准确性或通用性为代价。为了解决这些限制，我们提出了一种用于DIGC的流式、深度流水线FPGA加速器，具有片上缓冲器，可以处理小而均匀的输入特征块。我们的设计通过局部化计算最小化外部内存流量，并通过堆插入直接在流式输入块上执行局部合并排序和全局k路合并，实现高效的并行排序。这种模块化架构可以无缝扩展到不同的图像分辨率、ViG层类型和模型大小及变体，并支持各种基于ViG的视觉主干网络中的DIGC。由于静态配置的并行性最小化了关键路径延迟，该设计在布局布线后实现了高时钟频率，并且相比优化的CPU和GPU DIGC基线，分别实现了高达16.6倍和6.8倍的加速。


### 论文摘要

Vision Graph Neural Networks (Vision GNNs, or ViGs) represent images as unstructured graphs, achieving state of the art performance in computer vision tasks such as image classification, object detection, and instance segmentation. Dynamic Image Graph Construction (DIGC) builds image graphs by connecting patches (nodes) based on feature similarity, and is dynamically repeated in each ViG layer following GNN based patch (node) feature updates. However, DIGC constitutes over 50% of end to end ViG inference latency, rising to 95% at high image resolutions, making it the dominant computational bottleneck. While hardware acceleration holds promise, prior works primarily optimize graph construction algorithmically, often compromising DIGC flexibility, accuracy, or generality. To address these limitations, we propose a streaming, deeply pipelined FPGA accelerator for DIGC, featuring on chip buffers that process input features in small, uniform blocks. Our design minimizes external memory traffic via localized computation and performs efficient parallel sorting with local merge sort and global k way merging directly on streaming input blocks via heap insertion. This modular architecture scales seamlessly across image resolutions, ViG layer types, and model sizes and variants, and supports DIGC across diverse ViG based vision backbones. The design achieves high clock frequencies post place and route due to the statically configured parallelism minimizing critical path delay and delivers up to 16.6x and 6.8x speedups over optimized CPU and GPU DIGC baselines.

---

## 32. Adaptive Canonicalization with Application to Invariant Anisotropic Geometric Networks

**论文链接:** [http://arxiv.org/abs/2509.24886v1](http://arxiv.org/abs/2509.24886v1)

**作者:** Ya-Wei Eileen Lin, Ron Levie

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种自适应标准化方法，解决了传统标准化方法在等变机器学习中引入不连续性的问题。该方法基于先验最大化，通过选择能最大化网络预测置信度的输入标准形式，实现了连续且对称尊重的模型，并在多个任务上表现优异。

### 背景

标准化是等变机器学习中广泛使用的策略，通过将每个输入映射到标准形式来强制神经网络对称性。然而，这种策略常常引入不连续性，影响训练稳定性，限制泛化能力，并使通用近似定理复杂化。

### 目的

解决传统标准化方法引入不连续性的问题，开发一种能够保持模型连续性和对称性的新方法。

### 方法

提出自适应标准化框架，其中标准化取决于输入和网络。具体实现是基于先验最大化的自适应标准化，选择能最大化网络预测置信度的输入标准形式。证明了该方法产生的模型具有连续性、对称尊重性和通用近似特性。

### 主要发现

自适应标准化方法在解决谱图神经网络中的特征基歧义和处理点云中的旋转对称性方面有效。在分子和蛋白质分类以及点云分类任务上的实证验证表明，该方法优于数据增强、标准标准化和等变架构等其他等变机器学习解决方案。

### 结论

自适应标准化是一种有效的策略，能够克服传统标准化方法的局限性，提供更稳定、更通用的等变机器学习解决方案。

### 翻译

标准化是等变机器学习中广泛使用的策略，通过将每个输入映射到标准形式来强制神经网络对称性。然而，它常常引入不连续性，这会影响训练期间的稳定性，限制泛化能力，并使通用近似定理复杂化。在本文中，我们通过引入自适应标准化来解决这一问题，这是一种通用框架，其中标准化取决于输入和网络。具体而言，我们提出了基于先验最大化的自适应标准化，其中输入的标准形式被选择为最大化网络的预测置信度。我们证明这种构建产生了连续且对称尊重的模型，并且具有通用近似特性。我们提出了我们设置的两种应用：(i) 解决谱图神经网络中的特征基歧义，以及 (ii) 处理点云中的旋转对称性。我们在分子和蛋白质分类以及点云分类任务上经验性地验证了我们的方法。我们的自适应标准化优于等变机器学习的其他三种常见解决方案：数据增强、标准标准化和等变架构。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决传统标准化方法在等变机器学习中引入的不连续性问题。这个问题很重要，因为不连续性会影响训练稳定性，限制模型泛化能力，并使通用近似定理复杂化。在几何表示学习中，构建尊重数据固有对称性的模型对处理图、图像和3D对象等数据至关重要，可以提高模型在分布变化情况下的鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到传统标准化方法的问题在于它只依赖于输入而忽略了网络本身。他们提出如果标准化同时依赖于输入和网络，可以解决不连续性问题。他们借鉴了现有的等变机器学习方法，包括等变架构、数据增强和标准化方法，但创新性地提出了让标准化形式与网络参数相关联的自适应框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是让输入的标准形式不仅取决于输入数据，还取决于处理该数据的网络。具体实现是：对于每个输入，尝试一组可能的变换（如旋转），选择使网络预测置信度最高的变换作为该输入的标准形式。在训练过程中，网络和标准化过程同时优化，网络不需要自己学习对称性，只需在最佳标准形式上表现良好。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出自适应标准化通用框架；2)基于先验最大化的具体实现；3)证明该方法产生连续且尊重对称性的模型；4)提出两种应用场景。相比之前工作，不同之处在于：传统标准化只依赖输入导致不连续，而自适应标准化同时依赖输入和网络确保连续；与等变架构不同，它使用非等变网络通过标准化实现等变性；与数据增强不同，它不需要呈现数据的多种姿态；与框架平均不同，它不需要在多个变换上平均输出。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种自适应标准化方法，通过让输入的标准形式同时依赖于输入数据和神经网络本身，解决了传统标准化方法中的不连续性问题，实现了连续、对称保持且具有通用近似特性的等变机器学习。'}


### 论文摘要

Canonicalization is a widely used strategy in equivariant machine learning, enforcing symmetry in neural networks by mapping each input to a standard form. Yet, it often introduces discontinuities that can affect stability during training, limit generalization, and complicate universal approximation theorems. In this paper, we address this by introducing \emph{adaptive canonicalization}, a general framework in which the canonicalization depends both on the input and the network. Specifically, we present the adaptive canonicalization based on prior maximization, where the standard form of the input is chosen to maximize the predictive confidence of the network. We prove that this construction yields continuous and symmetry-respecting models that admit universal approximation properties.   We propose two applications of our setting: (i) resolving eigenbasis ambiguities in spectral graph neural networks, and (ii) handling rotational symmetries in point clouds. We empirically validate our methods on molecular and protein classification, as well as point cloud classification tasks. Our adaptive canonicalization outperforms the three other common solutions to equivariant machine learning: data augmentation, standard canonicalization, and equivariant architectures.

---

## 33. ELPG-DTFS: Prior-Guided Adaptive Time-Frequency Graph Neural Network for EEG Depression Diagnosis

**论文链接:** [http://arxiv.org/abs/2509.24860v1](http://arxiv.org/abs/2509.24860v1)

**作者:** Jingru Qiu, Jiale Liang, Xuanhan Fan, Mingda Zhang, Zhenli He

**发布时间:** 2025-09-29

**备注:** 8 page,3 figures

### GPT解析

### 总结

提出了一种名为ELPG-DTFS的先验引导自适应时频图神经网络，用于提高脑电图在抑郁症诊断中的准确性和可解释性。

### 背景

抑郁症的及时客观筛查很重要，但诊断仍然依赖于主观量表。脑电图可以提供低成本生物标志物，但现有深度模型将频谱视为静态图像，固定通道间图，并忽略先验知识，限制了准确性和可解释性。

### 目的

开发一种新型深度学习模型，提高基于脑电图的抑郁症诊断准确性和可解释性。

### 方法

ELPG-DTFS是一种先验引导的自适应时频图神经网络，包含三个创新点：(1)带有跨带互信息的通道带注意力机制；(2)用于动态功能连接的可学习邻接矩阵；(3)注入神经科学先验的残差知识图谱路径。

### 主要发现

在128通道的MODMA数据集(53名受试者)上，ELPG-DTFS达到了97.63%的准确率和97.33%的F1值，超越了2025年最先进的ACM-GNN模型。消融实验显示，移除任何模块都会使F1值降低高达4.35，证实了这些模块的互补价值。

### 结论

ELPG-DTFS为下一代基于脑电图的抑郁症诊断提供了稳健且可解释的框架。

### 翻译

及时客观地筛查重度抑郁症(MDD)至关重要，但诊断仍依赖于主观量表。脑电图(EEG)提供了一种低成本生物标志物，但现有深度模型将频谱视为静态图像，固定通道间图，并忽略先验知识，限制了准确性和可解释性。我们提出了ELPG-DTFS，一种先验引导的自适应时频图神经网络，引入：(1)带有跨带互信息的通道带注意力，(2)用于动态功能连接的可学习邻接矩阵，以及(3)注入神经科学先验的残差知识图谱路径。在128通道的MODMA数据集(53名受试者)上，ELPG-DTFS实现了97.63%的准确率和97.33%的F1值，超越了2025年最先进的ACM-GNN。消融实验显示，移除任何模块都会使F1值降低高达4.35，证实了它们的互补价值。因此，ELPG-DTFS为下一代基于EEG的MDD诊断提供了稳健且可解释的框架。


### 论文摘要

Timely and objective screening of major depressive disorder (MDD) is vital, yet diagnosis still relies on subjective scales. Electroencephalography (EEG) provides a low-cost biomarker, but existing deep models treat spectra as static images, fix inter-channel graphs, and ignore prior knowledge, limiting accuracy and interpretability. We propose ELPG-DTFS, a prior-guided adaptive time-frequency graph neural network that introduces: (1) channel-band attention with cross-band mutual information, (2) a learnable adjacency matrix for dynamic functional links, and (3) a residual knowledge-graph pathway injecting neuroscience priors. On the 128-channel MODMA dataset (53 subjects), ELPG-DTFS achieves 97.63% accuracy and 97.33% F1, surpassing the 2025 state-of-the-art ACM-GNN. Ablation shows that removing any module lowers F1 by up to 4.35, confirming their complementary value. ELPG-DTFS thus offers a robust and interpretable framework for next-generation EEG-based MDD diagnostics.

---

## 34. Neural Message-Passing on Attention Graphs for Hallucination Detection

**论文链接:** [http://arxiv.org/abs/2509.24770v1](http://arxiv.org/abs/2509.24770v1)

**作者:** Fabrizio Frasca, Guy Bar-Shalom, Yftah Ziser, Haggai Maron

**发布时间:** 2025-09-29

**备注:** Preprint. 25 pages, 2 figures

### GPT解析

### 总结

该论文提出了CHARM方法，通过将大语言模型的计算痕迹表示为属性图，并应用图神经网络进行幻觉检测，在各种基准测试中表现优于现有方法。

### 背景

大语言模型经常生成不正确或无支持的内容，这种现象被称为'幻觉'。现有的检测方法依赖于启发式方法或简单模型，基于孤立的计算痕迹如激活值或注意力图。

### 目的

统一不同的计算信号，将它们表示为属性图，并将幻觉检测作为一个图学习任务来解决。

### 方法

提出CHARM方法，将标记表示为节点，注意力流表示为边，同时结合注意力分数和激活值作为特征，在构建的属性图上应用图神经网络进行幻觉检测。

### 主要发现

CHARM在理论上包含了先前的基于注意力的启发式方法；在实验中，CHARM在各种基准测试中持续优于其他方法；结果揭示了图结构的相关作用以及结合计算痕迹的好处；CHARM在跨数据集转移任务中显示出有希望的零样本性能。

### 结论

CHARM通过结合多种计算痕迹并利用图学习技术，能够有效检测大语言模型中的幻觉，性能优于现有方法。

### 翻译

大型语言模型（LLMs）经常生成不正确或无支持的内容，这种现象被称为'幻觉'。现有的检测方法依赖于启发式方法或简单模型，这些方法基于孤立的计算痕迹，如激活值或注意力图。我们将这些信号统一表示为属性图，其中标记是节点，边缘遵循注意力流，并且两者都携带来自注意力分数和激活值的特征。我们的方法CHARM将幻觉检测作为一个图学习任务，并通过在上述属性图上应用图神经网络来解决它。我们证明CHARM在理论上包含了先前的基于注意力的启发式方法，并且在实验中，它一致地在各种基准测试中优于其他领先方法。我们的结果揭示了图结构的相关作用以及结合计算痕迹的好处，同时显示CHARM在跨数据集转移任务中展现出有希望的零样本性能。


### 论文摘要

Large Language Models (LLMs) often generate incorrect or unsupported content, known as hallucinations. Existing detection methods rely on heuristics or simple models over isolated computational traces such as activations, or attention maps. We unify these signals by representing them as attributed graphs, where tokens are nodes, edges follow attentional flows, and both carry features from attention scores and activations. Our approach, CHARM, casts hallucination detection as a graph learning task and tackles it by applying GNNs over the above attributed graphs. We show that CHARM provably subsumes prior attention-based heuristics and, experimentally, it consistently outperforms other leading approaches across diverse benchmarks. Our results shed light on the relevant role played by the graph structure and on the benefits of combining computational traces, whilst showing CHARM exhibits promising zero-shot performance on cross-dataset transfer.

---

## 35. Beyond Softmax: A Natural Parameterization for Categorical Random Variables

**论文链接:** [http://arxiv.org/abs/2509.24728v1](http://arxiv.org/abs/2509.24728v1)

**作者:** Alessandro Manenti, Cesare Alippi

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一种名为catnat的函数，用于替代深度学习中的softmax函数，解决了潜在类别变量离散性对梯度下降学习的挑战。

### 背景

潜在类别变量在深度学习中被广泛使用，可建模离散强化学习环境中的动作、潜在变量模型中的类别或图神经网络中的关系。然而，其离散性质对梯度下降算法构成挑战。

### 目的

提供一种互补方法来改善离散变量的梯度下降学习，而非仅改进梯度估计技术。

### 方法

重新审视softmax函数的信息几何局限性，提出由分层二元分割组成的catnat函数，证明其对梯度下降的优势源于产生的对角费舍尔信息矩阵。

### 主要发现

在图结构学习、变分自编码器和强化学习等实验中，catnat函数提高了学习效率，产生了测试性能更高的模型。

### 结论

catnat函数实现简单，可无缝集成到现有代码库，与标准训练稳定技术兼容，是softmax函数的更好替代方案。

### 翻译

潜在类别变量频繁出现在深度学习架构中。它们可以建模离散强化学习环境中的动作，表示潜在变量模型中的类别，或在图神经网络中表达关系。尽管被广泛使用，它们的离散性质对梯度下降学习算法提出了重大挑战。虽然大量工作提供了改进的梯度估计技术，我们采取了一种互补的方法。具体来说，我们：1)重新审视普遍使用的softmax函数，并从信息几何角度展示其局限性；2)用catnat函数替代softmax，该函数由一系列分层二元分割组成；我们证明由于产生的对角费舍尔信息矩阵，这种选择对梯度下降有显著优势。丰富的实验集——包括图结构学习、变分自编码器和强化学习——经验性地表明，所提出的函数提高了学习效率，并产生了具有一致更高测试性能的模型。catnat易于实现，可以无缝集成到现有代码库中。此外，它仍然与标准训练稳定技术兼容，因此提供了比softmax函数更好的替代方案。


### 论文摘要

Latent categorical variables are frequently found in deep learning architectures. They can model actions in discrete reinforcement-learning environments, represent categories in latent-variable models, or express relations in graph neural networks. Despite their widespread use, their discrete nature poses significant challenges to gradient-descent learning algorithms. While a substantial body of work has offered improved gradient estimation techniques, we take a complementary approach. Specifically, we: 1) revisit the ubiquitous $\textit{softmax}$ function and demonstrate its limitations from an information-geometric perspective; 2) replace the $\textit{softmax}$ with the $\textit{catnat}$ function, a function composed of a sequence of hierarchical binary splits; we prove that this choice offers significant advantages to gradient descent due to the resulting diagonal Fisher Information Matrix. A rich set of experiments - including graph structure learning, variational autoencoders, and reinforcement learning - empirically show that the proposed function improves the learning efficiency and yields models characterized by consistently higher test performance. $\textit{Catnat}$ is simple to implement and seamlessly integrates into existing codebases. Moreover, it remains compatible with standard training stabilization techniques and, as such, offers a better alternative to the $\textit{softmax}$ function.

---

## 36. Community detection robustness of graph neural networks

**论文链接:** [http://arxiv.org/abs/2509.24662v1](http://arxiv.org/abs/2509.24662v1)

**作者:** Jaidev Goel, Pablo Moriano, Ramakrishnan Kannan, Yulia R. Gel

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究系统评估了六种图神经网络架构在社区检测任务中的鲁棒性，发现监督式GNN基线准确性更高，而无监督方法DMoN在面对有针对性攻击时更具弹性，社区强度显著影响模型鲁棒性。

### 背景

图神经网络(GNNs)越来越多地用于属性网络中的社区检测，它们通过消息传递和池化结合结构拓扑和节点属性，但其对不同扰动和有针对性攻击的鲁棒性尚不清楚。

### 目的

揭示GNN在社区检测任务中敏感性的潜在机制，对六种广泛采用的GNN架构进行系统性计算评估。

### 方法

评估六种GNN架构(GCN、GAT、Graph-SAGE、DiffPool、MinCUT和DMoN)，分析三种扰动类别(节点属性操作、边拓扑扭曲和对抗攻击)，使用元素中心相似性作为评估指标，在合成基准和真实世界引用网络上进行评估。

### 主要发现

监督式GNN通常实现更高的基线准确性；无监督方法特别是DMoN在有针对性扰动下保持更强弹性；社区强度显著影响鲁棒性，定义良好的社区减少性能损失；节点属性扰动与有针对性的边删除和属性分布偏移往往导致社区恢复的最大退化。

### 结论

基于GNN的社区检测中存在准确性和鲁棒性之间的重要权衡，为选择对噪声和对抗性攻击具有弹性的架构提供了新见解。

### 翻译

图神经网络(GNNs)越来越多地用于属性网络中的社区检测。它们通过消息传递和池化结合结构拓扑和节点属性。然而，它们在不同扰动和有针对性的攻击下与社区检测任务相关的鲁棒性或缺乏鲁棒性尚未得到充分理解。为了揭示GNN在社区检测任务中敏感性的潜在机制，我们对六种广泛采用的GNN架构进行了系统性计算评估：GCN、GAT、Graph-SAGE、DiffPool、MinCUT和DMoN。分析涵盖三种扰动类别：节点属性操作、边拓扑扭曲和对抗攻击。我们在合成基准和真实世界引用网络上使用元素中心相似性作为评估指标。我们的研究结果表明，监督式GNN往往实现更高的基线准确性，而无监督方法，特别是DMoN，在有针对性的和对抗性扰动下保持更强的弹性。此外，鲁棒性似乎受到社区强度的强烈影响，定义良好的社区减少了性能损失。在所有模型中，与有针对性的边删除和属性分布偏移相关的节点属性扰动往往导致社区恢复的最大退化。这些发现强调了基于GNN的社区检测中准确性和鲁棒性之间的重要权衡，并为选择对噪声和对抗性攻击具有弹性的架构提供了新的见解。


### 论文摘要

Graph neural networks (GNNs) are increasingly widely used for community detection in attributed networks. They combine structural topology with node attributes through message passing and pooling. However, their robustness or lack of thereof with respect to different perturbations and targeted attacks in conjunction with community detection tasks is not well understood. To shed light into latent mechanisms behind GNN sensitivity on community detection tasks, we conduct a systematic computational evaluation of six widely adopted GNN architectures: GCN, GAT, Graph-SAGE, DiffPool, MinCUT, and DMoN. The analysis covers three perturbation categories: node attribute manipulations, edge topology distortions, and adversarial attacks. We use element-centric similarity as the evaluation metric on synthetic benchmarks and real-world citation networks. Our findings indicate that supervised GNNs tend to achieve higher baseline accuracy, while unsupervised methods, particularly DMoN, maintain stronger resilience under targeted and adversarial perturbations. Furthermore, robustness appears to be strongly influenced by community strength, with well-defined communities reducing performance loss. Across all models, node attribute perturbations associated with targeted edge deletions and shift in attribute distributions tend to cause the largest degradation in community recovery. These findings highlight important trade-offs between accuracy and robustness in GNN-based community detection and offer new insights into selecting architectures resilient to noise and adversarial attacks.

---

## 37. Prompting Robot Teams with Natural Language

**论文链接:** [http://arxiv.org/abs/2509.24575v1](http://arxiv.org/abs/2509.24575v1)

**作者:** Nicolas Pfitzer, Eduardo Sebastián, Ajay Shankar, Amanda Prorok

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种通过自然语言向多机器人团队提示高级任务的框架，利用语言模型的推理能力进行任务理解和分解，应用于多机器人协作决策

### 背景

在集体中，个体行为难以指定和解释，且必须不断适应他人行动，需要一种既具备任务表示能力又支持去中心化和实时交互操作的框架

### 目的

利用语言模型在理解和分解人类意图表达方面的推理能力，重新应用于多机器人协作和决策系统

### 方法

将任务表示为确定性有限自动机(DFA)，用循环神经网络(RNN)编码多个自动机，将语言模型获得的子任务逻辑提炼到RNN中，训练基于RNN隐藏状态和语言嵌入条件的图神经网络(GNN)控制策略

### 主要发现

通过将任务表示为DFA并使用RNN编码，可将语言模型的推理能力转化为多机器人系统的可执行行动

### 结论

该轻量级可解释模型能够在各种模拟和真实世界多机器人任务中实现顺序和协作行为

### 翻译

这篇论文提出了一个框架，用于通过自然语言表达式向多机器人团队提示高级任务。我们的目标是利用最近语言模型在理解和分解人类意图表达方面展示的推理能力，并重新利用这些能力进行多机器人协作和决策。关键挑战在于，集体中个体的行为可能难以指定和解释，并且必须不断适应他人的行动。这需要一个既具备任务逻辑和语义所需表示能力，又支持去中心化和交互式实时操作的框架。我们通过认识到任务可以表示为确定性有限自动机(DFA)，并且循环神经网络(RNN)可以编码多个自动机来解决这一困境。这使得我们将从语言模型获得的子任务逻辑和顺序分解提炼到RNN中，并将其内部状态与给定任务的语义对齐。通过训练一个基于RNN隐藏状态和语言嵌入条件的图神经网络(GNN)控制策略，我们的方法使机器人能够以去中心化方式执行与任务相关的行动。我们在各种需要团队顺序和协作行为的模拟和真实世界多机器人任务上评估了这个单一轻量级可解释模型


### 论文摘要

This paper presents a framework towards prompting multi-robot teams with high-level tasks using natural language expressions. Our objective is to use the reasoning capabilities demonstrated by recent language models in understanding and decomposing human expressions of intent, and repurpose these for multi-robot collaboration and decision-making. The key challenge is that an individual's behavior in a collective can be hard to specify and interpret, and must continuously adapt to actions from others. This necessitates a framework that possesses the representational capacity required by the logic and semantics of a task, and yet supports decentralized and interactive real-time operation. We solve this dilemma by recognizing that a task can be represented as a deterministic finite automaton (DFA), and that recurrent neural networks (RNNs) can encode numerous automata. This allows us to distill the logic and sequential decompositions of sub-tasks obtained from a language model into an RNN, and align its internal states with the semantics of a given task. By training a graph neural network (GNN) control policy that is conditioned on the hidden states of the RNN and the language embeddings, our method enables robots to execute task-relevant actions in a decentralized manner. We present evaluations of this single light-weight interpretable model on various simulated and real-world multi-robot tasks that require sequential and collaborative behavior by the team -- sites.google.com/view/prompting-teams.

---

## 38. Graph-Based Learning of Free Surface Dynamics in Generalized Newtonian Fluids using Smoothed Particle Hydrodynamics

**论文链接:** [http://arxiv.org/abs/2509.24264v1](http://arxiv.org/abs/2509.24264v1)

**作者:** Hyo-Jin Kim, Jaekwang Kim, Hyung-Jun Park

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一种图神经网络模型，用于高效预测具有自由表面动力学的非牛顿流体的流动行为。

### 背景

非牛顿流体的数值分析存在重大挑战，传统算法难以处理其动态变化的流变特性。幂律流体的粘度随剪切率增加而指数下降，使模拟更加困难。在自由表面流动中，计算挑战进一步加剧。

### 目的

开发一种基于图神经网络的数值模型，提高非牛顿幂律流体流动模拟的计算效率，同时保持准确性。

### 方法

基于光滑粒子流体动力学(SPH)方法的优势，作者开发了一种新颖的GNN模型。该模型在SPH模拟数据上进行训练，学习基于流体幂律参数的SPH相互作用中粒子加速的影响。

### 主要发现

GNN模型显著加速了计算过程，同时在基准测试（包括水坝溃决和液滴冲击模拟）中保持了可靠的准确性。

### 结论

基于GNN的模拟框架在高效模拟非牛顿流体行为方面具有巨大潜力，为数据驱动流体模拟的未来发展开辟了道路。

### 翻译

该研究提出了一种图神经网络模型，用于高效预测具有自由表面动力学的非牛顿流体的流动行为。非牛顿流体的数值分析存在重大挑战，因为传统算法难以处理其动态变化的流变特性。幂律流体的粘度随剪切率增加而指数下降，使数值模拟特别困难。在自由表面流动场景中，计算挑战进一步加剧。在这种情况下，基于粒子的方法如光滑粒子流体动力学比传统的基于网格的技术具有优势。基于此方法，作者引入了一种新颖的基于GNN的数值模型，以提高非牛顿幂律流体流动模拟的计算效率。该模型在SPH模拟数据上进行训练，学习基于流体幂律参数的SPH相互作用中粒子加速的影响。GNN显著加速了计算，同时在基准测试中保持了可靠的准确性。结果强调了基于GNN的模拟框架在高效模拟非牛顿流体行为方面的潜力。


### 论文摘要

In this study, we propose a graph neural network (GNN) model for efficiently predicting the flow behavior of non-Newtonian fluids with free surface dynamics. The numerical analysis of non-Newtonian fluids presents significant challenges, as traditional algorithms designed for Newtonian fluids with constant viscosity often struggle to converge when applied to non-Newtonian cases, where rheological properties vary dynamically with flow conditions. Among these, power-law fluids exhibit viscosity that decreases exponentially as the shear rate increases, making numerical simulations particularly difficult. The complexity further escalates in free surface flow scenarios, where computational challenges intensify. In such cases, particle-based methods like smoothed particle hydrodynamics (SPH) provide advantages over traditional grid-based techniques, such as the finite element method (FEM). Building on this approach, we introduce a novel GNN-based numerical model to enhance the computational efficiency of non-Newtonian power-law fluid flow simulations. Our model is trained on SPH simulation data, learning the effects of particle accelerations in the presence of SPH interactions based on the fluid's power-law parameters. The GNN significantly accelerates computations while maintaining reliable accuracy in benchmark tests, including dam-break and droplet impact simulations. The results underscore the potential of GNN-based simulation frameworks for efficiently modeling non-Newtonian fluid behavior, paving the way for future advancements in data-driven fluid simulations.

---

## 39. Difference-in-Differences Under Network Interference

**论文链接:** [http://arxiv.org/abs/2509.24259v1](http://arxiv.org/abs/2509.24259v1)

**作者:** Kuan Sun, Zhiguo Xiao

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文开发了基于网络的差异中差异设计中直接处理效应和溢出处理效应的双重稳健估计量

### 背景

传统DiD方法未明确处理处理溢出和高维网络混杂问题

### 目的

提出适应处理溢出和高维网络混杂的估计方法

### 方法

基于条件平行趋势假设，使用图神经网络估计 nuisance 函数

### 主要发现

估计量在网络规模增加时具有一致性和渐近正态性

### 结论

解决了传统DiD方法忽视网络干扰的局限性

### 翻译

本文针对基于网络的差异中差异设计，开发了对处理对象的直接平均处理效应和溢出平均处理效应的双重稳健估计量。与标准DiD方法不同，所提出的方法明确适应了处理溢出和由单位间复杂依赖引起的高维网络混杂。识别依赖于调整高维网络混杂后成立的条件平行趋势假设。估计量在网络规模增加时具有一致性和渐近正态性，我们使用图神经网络来估计 nuisance 函数。模拟研究和对美国县级口罩强制令及其对COVID-19传播影响的实证应用展示了良好的有限样本性能，解决了忽视网络干扰的传统DiD方法的局限性。


### 论文摘要

This paper develops doubly robust estimators for direct (DATT) and spillover (SATT) average treatment effects on the treated in network-based difference-in-differences (DiD) designs. Unlike standard DiD methods, the proposed approach explicitly accommodates treatment spillovers and high-dimensional network confounding arising from complex inter-unit dependencies. Identification relies on a conditional parallel-trends assumption that holds after adjusting for high-dimensional network confounders. The estimators are consistent and asymptotically normal as the network size increases, and we use graph neural networks (GNNs) to estimate nuisance functions. Simulation studies and an empirical application to U.S. county-level mask mandates and their impact on COVID-19 transmission demonstrate favorable finite-sample performance, addressing limitations of conventional DiD methods that ignore network interference.

---

## 40. ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs

**论文链接:** [http://arxiv.org/abs/2509.24115v1](http://arxiv.org/abs/2509.24115v1)

**作者:** Evan Dramko, Yihuang Xiong, Yizhi Zhu, Geoffroy Hautier, Thomas Reps, Christopher Jermaine, Anastasios Kyrillidis

**发布时间:** 2025-09-28

**备注:** 14 total pages of main content, 4 of references, 3 in Appendix

### GPT解析

### 总结

该研究提出了一种名为ADAPT的新型机器学习力场，用于加速材料中点缺陷的结构弛豫计算，解决了现有方法中的过平滑和长程相互作用表示问题。

### 背景

点缺陷在驱动材料特性方面起着核心作用。第一性原理方法虽被广泛用于计算缺陷能量和结构，但计算成本高昂。现有的基于图神经网络的机器学习力场存在过平滑和长程相互作用表示不足的问题，这些问题在点缺陷建模中尤为突出。

### 目的

开发一种新的机器学习力场方法，解决现有基于图神经网络的机器学习力场在点缺陷建模中存在的问题，提高预测精度同时降低计算成本。

### 方法

研究者提出了ADAPT（Accelerated Deep Atomic Potential Transformer），用直接的空间坐标表示替代图表示，明确考虑所有原子间的成对相互作用，将原子视为标记，使用Transformer编码器建模它们的相互作用。

### 主要发现

应用于硅点缺陷数据集时，ADAPT相比最先进的基于图神经网络的模型，力和能量预测误差减少了约33%，同时只需要一小部分计算成本。

### 结论

ADAPT是一种有效的机器学习力场方法，能够更准确地预测点缺陷的性质，同时显著降低计算成本，为材料科学中的点缺陷研究提供了新的工具。

### 翻译

点缺陷在驱动材料特性方面起着核心作用。第一性原理方法被广泛用于计算缺陷的能量和结构，包括高通量缺陷数据库。然而，这些方法计算成本高昂，使得机器学习力场成为加速结构弛豫的替代方案。大多数现有的机器学习力场基于图神经网络，可能存在过平滑和长程相互作用表示不良的问题，这些问题在建模点缺陷时尤为突出。为了解决这些挑战，我们引入了加速深度原子势Transformer（ADAPT），这是一种机器学习力场，它用直接的空间坐标表示替代了图表示，并明确考虑了所有原子间的成对相互作用。原子被视为标记，使用Transformer编码器建模它们的相互作用。应用于硅点缺陷数据集时，ADAPT相比最先进的基于图神经网络的模型，力和能量预测误差减少了约33%，同时只需要一小部分计算成本。


### 论文摘要

Point defects play a central role in driving the properties of materials. First-principles methods are widely used to compute defect energetics and structures, including at scale for high-throughput defect databases. However, these methods are computationally expensive, making machine-learning force fields (MLFFs) an attractive alternative for accelerating structural relaxations. Most existing MLFFs are based on graph neural networks (GNNs), which can suffer from oversmoothing and poor representation of long-range interactions. Both of these issues are especially of concern when modeling point defects. To address these challenges, we introduce the Accelerated Deep Atomic Potential Transformer (ADAPT), an MLFF that replaces graph representations with a direct coordinates-in-space formulation and explicitly considers all pairwise atomic interactions. Atoms are treated as tokens, with a Transformer encoder modeling their interactions. Applied to a dataset of silicon point defects, ADAPT achieves a roughly 33 percent reduction in both force and energy prediction errors relative to a state-of-the-art GNN-based model, while requiring only a fraction of the computational cost.

---

## 41. From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.23912v1](http://arxiv.org/abs/2509.23912v1)

**作者:** Ouns El Harzli, Bernardo Cuenca Grau, Artur d'Avila Garcez, Ian Horrocks, Tarek R. Besold

**发布时间:** 2025-09-28

### GPT解析

### 总结

该研究建立了模态逻辑 fibring 与神经网络 fibring 之间的正式对应关系，推导了多种神经网络架构的非均匀逻辑表达能力结果，为使用计算逻辑工具解释神经网络学习的逻辑理论奠定了基础。

### 背景

模态逻辑的 fibring 是一种成熟的组合形式化方法，用于将可数个模态逻辑组合成具有共同语义的单一 fibring 语言，由 fibring 模型表征。受此启发，神经网络 fibring 被引入作为一种神经符号框架，用于在神经网络中结合学习和推理。

### 目的

通过形式化与神经网络 fibring 兼容的 fibring 模型来填补模态逻辑 fibring 与神经网络 fibring 之间的理论空白，长期目标是开辟使用 fibring 作为形式化工具解释神经网络学习逻辑理论的途径。

### 方法

形式化与神经网络 fibring 兼容的 fibring 模型概念，并利用这种对应关系推导图神经网络(GNNs)、图注意力网络(GATs)和 Transformer 编码器的非均匀逻辑表达能力结果。

### 主要发现

建立了神经网络 fibring 与模态逻辑 fibring 之间的正式对应关系，获得了关于 GNNs、GATs 和 Transformer 编码器的非均匀逻辑表达能力结果。

### 结论

该研究填补了神经网络 fibring 与模态逻辑 fibring 之间的理论空白，为使用计算逻辑工具解释神经网络学习的逻辑理论奠定了基础。

### 翻译

模态逻辑的 fibring 是一种成熟的形式化方法，用于将可数个模态逻辑组合成具有共同语义的单一 fibring 语言，由 fibring 模型表征。受此启发，神经网络 fibring 被引入作为一种神经符号框架，用于在神经网络中结合学习和推理。神经网络 fibring 使用训练网络的(预)激活来评估计算另一个网络权重的 fibring 函数，该网络的输出被注入回原始网络。然而，神经网络 fibring 与模态逻辑 fibring 之间的确切对应关系从未被正式建立。在本文中，我们通过形式化与神经网络 fibring 兼容的 fibring 模型思想来填补这一空白。利用这种对应关系，我们推导出图神经网络(GNNs)、图注意力网络(GATs)和 Transformer 编码器的非均匀逻辑表达能力结果。从长远来看，本文的目标是开辟使用 fibring 作为形式化工具的道路，用于用计算逻辑的工具解释神经网络学习的逻辑理论。


### 论文摘要

Fibring of modal logics is a well-established formalism for combining countable families of modal logics into a single fibred language with common semantics, characterized by fibred models. Inspired by this formalism, fibring of neural networks was introduced as a neurosymbolic framework for combining learning and reasoning in neural networks. Fibring of neural networks uses the (pre-)activations of a trained network to evaluate a fibring function computing the weights of another network whose outputs are injected back into the original network. However, the exact correspondence between fibring of neural networks and fibring of modal logics was never formally established. In this paper, we close this gap by formalizing the idea of fibred models \emph{compatible} with fibred neural networks. Using this correspondence, we then derive non-uniform logical expressiveness results for Graph Neural Networks (GNNs), Graph Attention Networks (GATs) and Transformer encoders. Longer-term, the goal of this paper is to open the way for the use of fibring as a formalism for interpreting the logical theories learnt by neural networks with the tools of computational logic.

---

## 42. Test-time GNN Model Evaluation on Dynamic Graphs

**论文链接:** [http://arxiv.org/abs/2509.23816v1](http://arxiv.org/abs/2509.23816v1)

**作者:** Bo Li, Xin Zheng, Ming Jin, Can Wang, Shirui Pan

**发布时间:** 2025-09-28

**备注:** Accepted by ICDM 2025

### GPT解析

### 总结

本文提出了一种名为DyGEval的动态图神经网络评估器，用于解决训练好的动态图神经网络在未见测试图上的性能不确定性问题。

### 背景

动态图神经网络是学习动态图数据的领先范式，常用于建模现实世界系统和应用。然而，动态图数据分布随时间演变，导致训练好的DGNN在实际部署中面对未见过的测试图时性能存在显著不确定性。

### 目的

引入DGNN模型评估这一新研究问题，旨在评估特定DGNN模型在观察到的动态图上的训练性能，并通过估计其在测试时间对未见动态图的性能。

### 方法

提出DyGEval评估器，采用两阶段框架：1) 测试时间动态图模拟，捕获训练-测试分布差异作为监督信号并训练评估器；2) DyGEval开发和训练，准确估计训练好的DGNN模型在测试时间动态图上的性能。

### 主要发现

广泛的实验表明，DyGEval作为评估器是有效的，能够评估不同动态图下各种DGNN骨干模型的性能，并能处理分布变化的情况。

### 结论

DyGEval能够有效解决DGNN模型评估这一新问题，为动态图神经网络的部署提供了性能评估工具。

### 翻译

动态图神经网络已成为学习动态图的领先范式，动态图常用于建模现实世界系统和应用。然而，由于动态图数据分布随时间演变的特点，训练良好的DGNN在实际部署中面对未见和未标记的测试图时，常常面临显著的性能不确定性。在这种情况下，评估已部署DGNN在测试时间的性能至关重要，以确定训练良好的DGNN是否适合对未见动态测试图进行推理。在这项工作中，我们引入了一个新的研究问题：DGNN模型评估，旨在评估在观察到的动态图上训练的特定DGNN模型的性能，通过估计其在测试时间对未见动态图的性能。具体来说，我们提出了一个动态图神经网络评估器，称为DyGEval，以解决这个新问题。所提出的DyGEval涉及一个两阶段框架：(1) 测试时间动态图模拟，捕获训练-测试分布差异作为监督信号并训练评估器；(2) DyGEval开发和训练，准确估计训练良好的DGNN模型在测试时间动态图上的性能。广泛的实验表明，所提出的DyGEval作为评估器是有效的，能够评估不同动态图下各种DGNN骨干模型的性能，特别是在分布变化的情况下。


### 论文摘要

Dynamic graph neural networks (DGNNs) have emerged as a leading paradigm for learning from dynamic graphs, which are commonly used to model real-world systems and applications. However, due to the evolving nature of dynamic graph data distributions over time, well-trained DGNNs often face significant performance uncertainty when inferring on unseen and unlabeled test graphs in practical deployment. In this case, evaluating the performance of deployed DGNNs at test time is crucial to determine whether a well-trained DGNN is suited for inference on an unseen dynamic test graph. In this work, we introduce a new research problem: DGNN model evaluation, which aims to assess the performance of a specific DGNN model trained on observed dynamic graphs by estimating its performance on unseen dynamic graphs during test time. Specifically, we propose a Dynamic Graph neural network Evaluator, dubbed DyGEval, to address this new problem. The proposed DyGEval involves a two-stage framework: (1) test-time dynamic graph simulation, which captures the training-test distributional differences as supervision signals and trains an evaluator; and (2) DyGEval development and training, which accurately estimates the performance of the well-trained DGNN model on the test-time dynamic graphs. Extensive experiments demonstrate that the proposed DyGEval serves as an effective evaluator for assessing various DGNN backbones across different dynamic graphs under distribution shifts.

---

## 43. Knowledge Homophily in Large Language Models

**论文链接:** [http://arxiv.org/abs/2509.23773v1](http://arxiv.org/abs/2509.23773v1)

**作者:** Utkarsh Sahu, Zhisheng Qi, Mahantesh Halappanavar, Nedim Lipka, Ryan A. Rossi, Franck Dernoncourt, Yu Zhang, Yao Ma, Yu Wang

**发布时间:** 2025-09-28

### GPT解析

### 总结

本研究探索了大型语言模型中知识的结构组织，发现模型对图中位置相近的实体拥有相似水平的知识，并基于此提出了一种图神经网络回归模型来估计实体级知识能力，提高了知识注入和检索的效率。

### 背景

大型语言模型被越来越多地研究作为神经知识库用于知识密集型应用，但其知识的结构组织尚未被探索。

### 目的

调查大型语言模型中的知识同质性模式，探索其知识的组织结构。

### 方法

通过在三元组和实体级别进行知识检查将LLM知识映射为图表示；分析实体与邻居间的知识能力关系；基于同质性原则提出图神经网络回归模型估计三元组的实体级知识能力分数。

### 主要发现

大型语言模型倾向于对图中位置更接近的实体拥有相似水平的知识。

### 结论

预测的知识能力可优先检查不太知名的三元组，在相同标注预算下最大化知识覆盖率，提高主动标注效率和推理密集型问答中的多跳路径检索能力。

### 翻译

大型语言模型（LLMs）被越来越多地研究作为神经知识库，用于支持知识密集型应用，如问答和事实核查。然而，其知识的结构组织仍未被探索。受认知神经科学发现的启发，如语义聚类和启动效应（了解一个事实会增加回忆相关事实的可能性），我们调查了LLMs中类似的知识同质性模式。为此，我们通过在三元组和实体级别进行知识检查，将LLM知识映射为图表示。随后，我们分析实体与其邻居之间的知识能力关系，发现LLMs倾向于对图中位置更接近的实体拥有相似水平的知识。受此同质性原则启发，我们提出了一种图神经网络（GNN）回归模型，通过利用邻域分数来估计三元组的实体级知识能力分数。预测的知识能力使我们能够优先检查不太知名的三元组，从而在相同的标注预算下最大化知识覆盖率。这不仅提高了主动标注以将知识注入LLMs的效率，还增强了推理密集型问答中的多跳路径检索。


### 论文摘要

Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, the structural organization of their knowledge remains unexplored. Inspired by cognitive neuroscience findings, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by this homophily principle, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging their neighborhood scores. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for fine-tuning to inject knowledge into LLMs but also enhances multi-hop path retrieval in reasoning-intensive question answering.

---

## 44. A Modality-Tailored Graph Modeling Framework for Urban Region Representation via Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2509.23772v1](http://arxiv.org/abs/2509.23772v1)

**作者:** Yaya Zhao, Kaiqi Zhao, Zixuan Tang, Zhiyuan Liu, Xiaoling Lu, Yalei Du

**发布时间:** 2025-09-28

### GPT解析

### 总结

MTGRR是一种模态定制图建模框架，用于城市区域表示，解决了现有方法的两个主要限制：对所有模态采用相同图神经网络架构和忽视空间异质性。

### 背景

基于图的模型已成为建模多模态城市数据和为各种下游任务学习区域表示的强大范式，但现有方法存在两个主要限制：无法捕捉模态特定结构和特征，以及忽视空间异质性导致次优表示。

### 目的

解决现有方法的两个主要限制，提出MTGRR框架，用于更有效地学习城市区域表示。

### 方法

MTGRR将模态分为聚合级和点级两类，对聚合级模态采用专家混合图架构，每个模态由专用专家GNN处理；对点级模态构建双级GNN；设计空间感知多模态融合机制动态推断区域特定融合权重；采用联合对比学习策略优化区域表示。

### 主要发现

在两个真实世界数据集上，跨越六个模态和三个任务的实验表明，MTGRR持续优于最先进的基线方法，验证了其有效性。

### 结论

MTGRR通过模态定制和空间感知融合机制，有效解决了现有方法的局限性，为城市区域表示学习提供了更强大的框架。

### 翻译

基于图的模型已成为建模多模态城市数据和为各种下游任务学习区域表示的强大范式。然而，现有方法面临两个主要限制：(1)它们通常对所有模态采用相同的图神经网络架构，无法捕捉模态特定的结构和特征。(2)在融合阶段，它们经常忽视空间异质性，假设不同模态的聚合权重在所有区域保持不变，导致次优表示。为解决这些问题，我们提出了MTGRR，一种用于城市区域表示的模态定制图建模框架，基于包含兴趣点(POI)、出租车流动性、土地利用、道路元素、遥感影像和街景图像的多模态数据集构建。(1)MTGRR根据空间密度和数据特性将模态分为两类：聚合级模态和点级模态。对于聚合级模态，MTGRR采用专家混合(MoE)图架构，每个模态由专用专家GNN处理以捕捉不同的模态特定特征。对于点级模态，构建双级GNN提取细粒度视觉语义特征。(2)为了在空间异质性下获得有效的区域表示，设计了空间感知多模态融合机制，动态推断区域特定模态融合权重。基于此图建模框架，MTGRR进一步采用联合对比学习策略，整合区域聚合级、点级和融合级目标以优化区域表示。在两个真实世界数据集上跨越六个模态和三个任务的实验表明，MTGRR持续优于最先进的基线方法，验证了其有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决城市区域表示学习中的两个关键问题：1) 现有方法对所有模态使用相同的图神经网络架构，无法捕捉不同模态特有的结构和特征；2) 在模态融合阶段，现有方法假设不同模态的聚合权重在所有区域保持不变，忽略了空间异质性。这些问题在研究中很重要，因为城市是多模态复杂系统，准确的城市区域表示对碳排放估算、GDP预测、人口预测等下游任务至关重要，而现有方法无法充分利用不同模态的独特特性和区域间的空间差异，限制了模型性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析现有方法的两个局限性，然后根据数据特征和空间密度将模态分为两类：聚合级模态（POI、出租车移动、土地利用、道路元素、遥感影像）和点级模态（街景图像）。针对聚合级模态，作者设计了混合专家图架构（MoE），包括全局异构图、专家GNN和模态感知门控机制；针对点级模态，设计了双层图神经网络。作者还借鉴了图神经网络、对比学习、混合专家模型和注意力机制等现有技术，但针对城市数据的特定特性进行了创新设计，引入了空间感知的多模态融合机制和联合对比学习策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是针对不同类型的数据模态使用专门的图神经网络架构，捕捉模态特定特性；考虑空间异质性，动态调整不同区域中各模态的融合权重；通过联合对比学习优化区域表示。整体流程：1) 收集六种城市模态数据并分类；2) 用混合专家图架构处理聚合级模态；3) 用双层图神经网络处理街景图像；4) 通过空间感知机制融合多模态信息；5) 使用联合对比学习优化表示；6) 将学习的表示应用于下游预测任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 模态特定的图建模框架，为不同模态设计专门架构；2) 空间感知的多模态融合机制，动态学习区域特定权重；3) 联合对比学习策略，整合三种对比目标；4) 全面整合六种城市数据模态。与之前工作不同：1) 区别于传统统一GNN架构，MTGRR为不同模态设计专门架构；2) 区别于固定权重融合，MTGRR动态调整区域特定权重；3) 区别于单一对比学习，MTGRR联合优化多种对比目标；4) 区别于单模态方法，MTGRR有效融合多模态信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MTGRR通过模态特定的图神经网络架构和空间感知的多模态融合机制，有效解决了城市区域表示学习中模态特性捕获不足和空间异质性被忽视的关键问题，显著提升了下游任务性能。'}


### 论文摘要

Graph-based models have emerged as a powerful paradigm for modeling multimodal urban data and learning region representations for various downstream tasks. However, existing approaches face two major limitations. (1) They typically employ identical graph neural network architectures across all modalities, failing to capture modality-specific structures and characteristics. (2) During the fusion stage, they often neglect spatial heterogeneity by assuming that the aggregation weights of different modalities remain invariant across regions, resulting in suboptimal representations. To address these issues, we propose MTGRR, a modality-tailored graph modeling framework for urban region representation, built upon a multimodal dataset comprising point of interest (POI), taxi mobility, land use, road element, remote sensing, and street view images. (1) MTGRR categorizes modalities into two groups based on spatial density and data characteristics: aggregated-level and point-level modalities. For aggregated-level modalities, MTGRR employs a mixture-of-experts (MoE) graph architecture, where each modality is processed by a dedicated expert GNN to capture distinct modality-specific characteristics. For the point-level modality, a dual-level GNN is constructed to extract fine-grained visual semantic features. (2) To obtain effective region representations under spatial heterogeneity, a spatially-aware multimodal fusion mechanism is designed to dynamically infer region-specific modality fusion weights. Building on this graph modeling framework, MTGRR further employs a joint contrastive learning strategy that integrates region aggregated-level, point-level, and fusion-level objectives to optimize region representations. Experiments on two real-world datasets across six modalities and three tasks demonstrate that MTGRR consistently outperforms state-of-the-art baselines, validating its effectiveness.

---

## 45. Graph Neural Networks with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion for Multivariate Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2509.23671v1](http://arxiv.org/abs/2509.23671v1)

**作者:** Jingqi Xu, Guibin Chen, Jingxi Lu, Yuzhang Lin

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为DIMIGNN的新型图神经网络方法，用于多元时间序列预测，通过多样性感知邻居选择和动态多尺度融合解决了现有方法的局限性。

### 背景

近年来，许多深度模型被提出以提高多元时间序列预测性能，其中基于图神经网络的方法因其能显式建模变量间依赖关系而显示出巨大潜力。

### 目的

解决现有GNN方法在多元时间序列预测中忽视邻居信息多样性以及仅依赖单一时间尺度表示的问题。

### 方法

提出了具有多样性感知邻居选择机制(DNSM)和动态多尺度融合模块(DMFM)的图神经网络DIMIGNN。DNSM确保变量与邻居共享高信息相似性同时保持邻居间多样性；DMFM动态调整不同时间尺度预测结果对最终预测的贡献。

### 主要发现

在真实世界数据集上的大量实验表明，DIMIGNN方法在性能上始终优于先前的方法。

### 结论

DIMIGNN通过多样性感知邻居选择和动态多尺度融合有效解决了现有多元时间序列预测方法的局限性，是一种有效的预测方法。

### 翻译

最近，许多深度模型被提出来增强多元时间序列(MTS)预测的性能。其中，基于图神经网络(GNNs)的方法因其能够显式建模变量间依赖关系而显示出巨大潜力。然而，这些方法通常忽视了邻居之间的信息多样性，可能导致冗余信息聚合。此外，它们的最终预测通常仅依赖于单一时间尺度的表示。为解决这些问题，我们提出了一种具有多样性感知邻居选择和动态多尺度融合的图神经网络(DIMIGNN)。DIMIGNN引入了多样性感知邻居选择机制(DNSM)，确保每个变量与其邻居共享高信息相似性，同时保持邻居之间的多样性。此外，还引入了动态多尺度融合模块(DMFM)，动态调整来自不同时间尺度的预测结果对最终预测结果的贡献。在真实世界数据集上的大量实验表明，DIMIGNN始终优于先前的方法。


### 论文摘要

Recently, numerous deep models have been proposed to enhance the performance of multivariate time series (MTS) forecasting. Among them, Graph Neural Networks (GNNs)-based methods have shown great potential due to their capability to explicitly model inter-variable dependencies. However, these methods often overlook the diversity of information among neighbors, which may lead to redundant information aggregation. In addition, their final prediction typically relies solely on the representation from a single temporal scale. To tackle these issues, we propose a Graph Neural Networks (GNNs) with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion (DIMIGNN). DIMIGNN introduces a Diversity-aware Neighbor Selection Mechanism (DNSM) to ensure that each variable shares high informational similarity with its neighbors while maintaining diversity among neighbors themselves. Furthermore, a Dynamic Multi-Scale Fusion Module (DMFM) is introduced to dynamically adjust the contributions of prediction results from different temporal scales to the final forecasting result. Extensive experiments on real-world datasets demonstrate that DIMIGNN consistently outperforms prior methods.

---

## 46. Pure Node Selection for Imbalanced Graph Node Classification

**论文链接:** [http://arxiv.org/abs/2509.23662v1](http://arxiv.org/abs/2509.23662v1)

**作者:** Fanlong Zeng, Wensheng Gan, Jiayang Wu, Philip S. Yu

**发布时间:** 2025-09-28

**备注:** Preprint, 8 tables, 9 figures

### GPT解析

### 总结

本文研究图结构数据中的类别不平衡问题，发现图神经网络(GNNs)通常假设类别平衡而忽视了这一问题。作者提出随机异常连接问题(RACP)概念，并设计了PNS(Pure Node Sampling)方法作为即插即用模块，在节点合成阶段直接操作以解决RACP。实验证明PNS能有效消除随机种子影响，提高模型稳定性，并在各种基准数据集上优于基线方法。

### 背景

类别不平衡问题指的是数据集中各类别数量分布不均，某些类别代表性严重不足。这种现象在图结构数据中普遍存在，但图神经网络(GNNs)通常基于类别平衡假设，忽视了这一问题。

### 目的

识别并解决图神经网络中由随机种子引起的随机异常连接问题(RACP)，消除算法中随机因素的影响，提高模型在类别不平衡情况下的性能和稳定性。

### 方法

提出PNS(Pure Node Sampling)方法，这是一个新颖的即插即用模块，在节点合成阶段直接操作以解决RACP问题。与现有的专门处理数量不平衡或拓扑不平衡的方法不同，PNS可以直接集成到现有GNN框架中。

### 主要发现

1) 某些现成图神经网络模型受随机种子影响导致性能显著下降，命名为随机异常连接问题(RACP)；2) PNS方法不仅消除不利随机种子影响，还能减轻节点邻居异常分布导致的性能下降；3) PNS在各种具有不同GNN骨干网络的基准数据集上表现优异。

### 结论

PNS是有效的即插即用模块，能解决图神经网络中的随机异常连接问题，提高模型在类别不平衡情况下的性能和稳定性，不受随机种子影响，且在多种测试中表现优异。

### 翻译

类别不平衡问题指的是数据集中各类别数量分布不均的情况，其中某些类别相对于其他类别代表性严重不足。类别不平衡在图结构数据中也普遍存在。图神经网络(GNNs)通常基于类别平衡的假设，往往忽视了类别不平衡的问题。在我们的研究中，我们识别了一个问题，我们称之为随机异常连接问题(RACP)，其中某些现成模型受到随机种子的影响，导致性能显著下降。为了消除算法中随机因素的影响，我们提出了PNS(Pure Node Sampling)来解决节点合成阶段的RACP问题。与现有的专门处理数量不平衡或拓扑不平衡的方法不同，PNS是一个新颖的即插即用模块，它在节点合成阶段直接操作以减轻RACP。此外，PNS还能减轻节点邻居异常分布导致的性能下降。我们进行了一系列实验以确定哪些因素受到随机种子的影响。实验结果证明了我们方法的有效性和稳定性，它不仅消除了不利随机种子的影响，还在各种具有不同GNN骨干网络的基准数据集上优于基线方法。数据和代码可在https://github.com/flzeng1/PNS获取。


### 论文摘要

The problem of class imbalance refers to an uneven distribution of quantity among classes in a dataset, where some classes are significantly underrepresented compared to others. Class imbalance is also prevalent in graph-structured data. Graph neural networks (GNNs) are typically based on the assumption of class balance, often overlooking the issue of class imbalance. In our investigation, we identified a problem, which we term the Randomness Anomalous Connectivity Problem (RACP), where certain off-the-shelf models are affected by random seeds, leading to a significant performance degradation. To eliminate the influence of random factors in algorithms, we proposed PNS (Pure Node Sampling) to address the RACP in the node synthesis stage. Unlike existing approaches that design specialized algorithms to handle either quantity imbalance or topological imbalance, PNS is a novel plug-and-play module that operates directly during node synthesis to mitigate RACP. Moreover, PNS also alleviates performance degradation caused by abnormal distribution of node neighbors. We conduct a series of experiments to identify what factors are influenced by random seeds. Experimental results demonstrate the effectiveness and stability of our method, which not only eliminates the effect of unfavorable random seeds but also outperforms the baseline across various benchmark datasets with different GNN backbones. Data and code are available at https://github.com/flzeng1/PNS.

---

## 47. Virtual Nodes based Heterogeneous Graph Convolutional Neural Network for Efficient Long-Range Information Aggregation

**论文链接:** [http://arxiv.org/abs/2509.23660v1](http://arxiv.org/abs/2509.23660v1)

**作者:** Ranhui Yan, Jia cai

**发布时间:** 2025-09-28

**DOI:** 10.1007/978-3-031-72344-5_15

### GPT解析

### 总结

本文提出了一种基于虚拟节点的异构图卷积网络(VN-HGCN)，通过引入虚拟节点增强图内信息流动，仅需4层即可有效聚合信息，并在三个真实异构图数据集上证明了其优越性。

### 背景

异构图神经网络(HGNNs)在异构图中表现强大，但现有模型难以捕捉长距离信息或需要堆叠多层学习依赖关系，导致计算复杂度高和过平滑问题。

### 目的

提出一种基于虚拟节点的异构图卷积网络，以增强图内信息流动并解决现有模型的局限性。

### 方法

通过引入虚拟节点，这些节点与图中特定类型的所有节点相连，促进不同类型节点和边缘间的长距离信息高效聚合，并将虚拟节点整合到图结构中。

### 主要发现

VN-HGCN可作为通用框架无缝应用于其他HGNN模型，展示了其通用性；实证验证了其有效性；在三个真实异构图数据集上的实验证明优于多种最先进基线方法。

### 结论

基于虚拟节点的异构图卷积网络能有效解决现有异构图模型的长距离信息捕捉问题，降低计算复杂度，避免过平滑问题，并在多种数据集上表现出优越性能。

### 翻译

异构图神经网络(HGNNs)通过聚合不同类型节点和边的信息，在异构图学习中展现出强大的性能。然而，现有的异构图模型往往难以捕捉长距离信息，或者需要堆叠大量层来学习此类依赖关系，导致计算复杂度高并遇到过平滑问题。在本文中，我们提出了一种基于虚拟节点的异构图卷积网络(VN-HGCN)，利用虚拟节点促进图内的增强信息流动。虚拟节点是辅助节点，与图中特定类型的所有节点相互连接，促进不同类型节点和边之间的长距离信息高效聚合。通过将虚拟节点整合到图结构中，VN-HGCN仅需4层即可实现有效的信息聚合。此外，我们证明了VN-HGCN可以作为通用框架，无缝应用于其他HGNN模型，展示了其通用性。实证评估验证了VN-HGCN的有效性，在三个真实异构图数据集上进行的大量实验证明了我们的模型优于几种最先进的基线方法。


### 论文摘要

Heterogeneous Graph Neural Networks (HGNNs) have exhibited powerful performance in heterogeneous graph learning by aggregating information from various types of nodes and edges. However, existing heterogeneous graph models often struggle to capture long-range information or necessitate stacking numerous layers to learn such dependencies, resulting in high computational complexity and encountering over-smoothing issues. In this paper, we propose a Virtual Nodes based Heterogeneous Graph Convolutional Network (VN-HGCN), which leverages virtual nodes to facilitate enhanced information flow within the graph. Virtual nodes are auxiliary nodes interconnected with all nodes of a specific type in the graph, facilitating efficient aggregation of long-range information across different types of nodes and edges. By incorporating virtual nodes into the graph structure, VN-HGCN achieves effective information aggregation with only $4$ layers. Additionally, we demonstrate that VN-HGCN can serve as a versatile framework that can be seamlessly applied to other HGNN models, showcasing its generalizability. Empirical evaluations validate the effectiveness of VN-HGCN, and extensive experiments conducted on three real-world heterogeneous graph datasets demonstrate the superiority of our model over several state-of-the-art baselines.

---

## 48. GraphIFE: Rethinking Graph Imbalance Node Classification via Invariant Learning

**论文链接:** [http://arxiv.org/abs/2509.23616v1](http://arxiv.org/abs/2509.23616v1)

**作者:** Fanlong Zeng, Wensheng Gan, Philip S. Yu

**发布时间:** 2025-09-28

**备注:** PrePrint, 16 pages, 7 tables, 6 figures

### GPT解析

### 总结

本文提出了一种名为GraphIFE的新框架，用于解决图结构数据中的类别不平衡问题，特别是在合成节点中的质量不一致问题，从而提高模型对少数类的识别能力。

### 背景

类别不平衡问题是指数据集中不同类别的样本分布不均衡，其中少数类样本显著不足。大多数图神经网络隐含假设类别分布均衡，无法处理类别不平衡带来的挑战，导致有偏差的学习和少数类性能下降。

### 目的

解决图不平衡条件下合成节点的质量不一致问题，提高模型识别不变特征的能力，从而增强对少数类的学习效果。

### 方法

提出GraphIFE（Graph Invariant Feature Extraction）框架，结合图不变学习的两个关键概念，引入加强嵌入空间表示的策略，增强模型识别不变特征的能力。

### 主要发现

在合成节点中存在质量不一致问题，这是导致图不平衡条件下性能次优的原因。GraphIFE通过加强嵌入空间表示和识别不变特征，有效缓解了这一问题。

### 结论

GraphIFE框架在多个数据集上始终优于各种基线方法，展示了其效率和强大的泛化能力。代码已在GitHub上公开（https://github.com/flzeng1/GraphIFE）。

### 翻译

类别不平衡问题指的是数据集中不同类别样本分布不均衡的现象，其中少数类样本显著不足。这个问题在图结构数据中也普遍存在。大多数图神经网络隐含地假设类别分布均衡，因此无法处理类别不平衡带来的挑战，这可能导致有偏差的学习和少数类性能下降。我们确定了合成节点中的质量不一致问题，这是导致图不平衡条件下性能次优的原因。为缓解此问题，我们提出了GraphIFE（Graph Invariant Feature Extraction），这是一种旨在缓解合成节点中质量不一致问题的新颖框架。我们的方法结合了图不变学习的两个关键概念，并引入了加强嵌入空间表示的策略，从而增强模型识别不变特征的能力。大量实验证明了该框架的效率和强大的泛化能力，因为GraphIFE在多个数据集上始终优于各种基线方法。代码已在https://github.com/flzeng1/GraphIFE上公开。


### 论文摘要

The class imbalance problem refers to the disproportionate distribution of samples across different classes within a dataset, where the minority classes are significantly underrepresented. This issue is also prevalent in graph-structured data. Most graph neural networks (GNNs) implicitly assume a balanced class distribution and therefore often fail to account for the challenges introduced by class imbalance, which can lead to biased learning and degraded performance on minority classes. We identify a quality inconsistency problem in synthesized nodes, which leads to suboptimal performance under graph imbalance conditions. To mitigate this issue, we propose GraphIFE (Graph Invariant Feature Extraction), a novel framework designed to mitigate quality inconsistency in synthesized nodes. Our approach incorporates two key concepts from graph invariant learning and introduces strategies to strengthen the embedding space representation, thereby enhancing the model's ability to identify invariant features. Extensive experiments demonstrate the framework's efficiency and robust generalization, as GraphIFE consistently outperforms various baselines across multiple datasets. The code is publicly available at https://github.com/flzeng1/GraphIFE.

---

## 49. Node Classification via Simplicial Interaction with Augmented Maximal Clique Selection

**论文链接:** [http://arxiv.org/abs/2509.23568v1](http://arxiv.org/abs/2509.23568v1)

**作者:** Eunho Koo, Tongseok Lim

**发布时间:** 2025-09-28

**备注:** To appear in Neurocomputing

### GPT解析

### 总结

该研究提出了一种增强的最大团策略，用于高效处理网络中的高阶交互，解决了传统方法中计算效率低和训练数据不平衡的问题，并在实验中证明了其优越性。

### 背景

考虑高阶交互可以让我们超越简单的成对连接，更全面地理解网络结构。虽然利用网络中的所有团来处理高阶交互是直观的，但由于高阶团和低阶团之间的重叠信息，这常常导致计算效率低下。

### 目的

解决利用所有团处理高阶交互时产生的计算效率问题，以及仅使用最大团时某些节点出现在多个最大团中导致训练数据不平衡的问题。

### 方法

提出了一种增强的最大团策略，该方法选择性地包含一些非最大团，以减轻特定节点的过度表示，促进网络中更平衡的学习。

### 主要发现

在合成网络和真实世界引用数据集上的比较分析表明，该方法优于基于成对交互、所有团或仅最大团的方法；通过将此策略集成到基于GNN的半监督学习中，建立了基于最大团的方法与GNN之间的联系，表明融入高阶结构可以提高预测准确性。

### 结论

增强的最大团策略为高阶网络学习提供了一种计算高效且有效的解决方案。

### 翻译

考虑高阶交互可以让我们超越简单的成对连接，更全面地理解网络结构。虽然利用网络中的所有团来处理高阶交互是直观的，但由于高阶团和低阶团之间的重叠信息，这常常导致计算效率低下。为了解决这个问题，我们提出了一种增强的最大团策略。尽管仅使用最大团可以减少不必要的重叠并提供网络的简洁表示，但某些节点可能仍然出现在多个最大团中，导致训练数据不平衡。因此，我们的增强最大团方法选择性地包含一些非最大团，以减轻特定节点的过度表示，并促进网络中更平衡的学习。在合成网络和真实世界引用数据集上的比较分析表明，我们的方法优于基于成对交互、所有团或仅最大团的方法。最后，通过将此策略集成到基于GNN的半监督学习中，我们建立了基于最大团的方法与GNN之间的联系，表明融入高阶结构可以提高预测准确性。因此，增强的最大团策略为高阶网络学习提供了一种计算高效且有效的解决方案。


### 论文摘要

Considering higher-order interactions allows for a more comprehensive understanding of network structures beyond simple pairwise connections. While leveraging all cliques in a network to handle higher-order interactions is intuitive, it often leads to computational inefficiencies due to overlapping information between higher-order and lower-order cliques. To address this issue, we propose an augmented maximal clique strategy. Although using only maximal cliques can reduce unnecessary overlap and provide a concise representation of the network, certain nodes may still appear in multiple maximal cliques, resulting in imbalanced training data. Therefore, our augmented maximal clique approach selectively includes some non-maximal cliques to mitigate the overrepresentation of specific nodes and promote more balanced learning across the network. Comparative analyses on synthetic networks and real-world citation datasets demonstrate that our method outperforms approaches based on pairwise interactions, all cliques, or only maximal cliques. Finally, by integrating this strategy into GNN-based semi-supervised learning, we establish a link between maximal clique-based methods and GNNs, showing that incorporating higher-order structures improves predictive accuracy. As a result, the augmented maximal clique strategy offers a computationally efficient and effective solution for higher-order network learning.

---

## 50. Hybrid Graph Embeddings and Louvain Algorithm for Unsupervised Community Detection

**论文链接:** [http://arxiv.org/abs/2509.23411v1](http://arxiv.org/abs/2509.23411v1)

**作者:** Dalila Khettaf, Djamel Djenouri, Zeinab Rezaeifar, Youcef Djenouri

**发布时间:** 2025-09-27

**备注:** to be published in ICMLT 2025 conference proceedings

### GPT解析

### 总结

这篇论文提出了一种结合Louvain算法和图神经网络的创新社区检测方法，无需预先知道社区数量即可发现社区结构。

### 背景

现有的社区检测方法大多需要预先知道社区数量，限制了它们的适用性。

### 目的

开发一种不需要先验知识的社区检测方法，能够自动发现社区结构并提高检测准确性。

### 方法

将Louvain算法与图神经网络结合，利用GNN生成的节点嵌入增强Louvain算法，并引入合并算法优化结果。

### 主要发现

该方法能够动态调整检测到的社区数量，比基准解决方案提高检测准确性，且不需要预先知道社区数量。

### 结论

将GNN与Louvain算法结合可以有效改进社区检测性能，为社区检测领域提供了新的研究方向。

### 翻译

本论文提出了一种新颖的社区检测方法，将Louvain算法与图神经网络(GNNs)相结合，使无需先验知识即可发现社区。与大多数现有解决方案相比，所提出的方法不需要预先知道社区数量。它利用GNN生成的节点嵌入来增强Louvain算法，以捕获更丰富的结构和特征信息。此外，它引入了一种合并算法来优化增强后的Louvain算法的结果，减少检测到的社区数量。据我们所知，这项工作是第一个使用GNN改进Louvain算法进行社区检测的。通过对真实数据集的评估，经验性地确认了所提出方法的改进。结果表明，与基准解决方案相比，该方法能够动态调整检测到的社区数量并提高检测准确性。


### 论文摘要

This paper proposes a novel community detection method that integrates the Louvain algorithm with Graph Neural Networks (GNNs), enabling the discovery of communities without prior knowledge. Compared to most existing solutions, the proposed method does not require prior knowledge of the number of communities. It enhances the Louvain algorithm using node embeddings generated by a GNN to capture richer structural and feature information. Furthermore, it introduces a merging algorithm to refine the results of the enhanced Louvain algorithm, reducing the number of detected communities. To the best of our knowledge, this work is the first one that improves the Louvain algorithm using GNNs for community detection. The improvement of the proposed method was empirically confirmed through an evaluation on real-world datasets. The results demonstrate its ability to dynamically adjust the number of detected communities and increase the detection accuracy in comparison with the benchmark solutions.

---

## 51. Mind the Links: Cross-Layer Attention for Link Prediction in Multiplex Networks

**论文链接:** [http://arxiv.org/abs/2509.23409v1](http://arxiv.org/abs/2509.23409v1)

**作者:** Devesh Sharma, Aditya Kishore, Ayush Garg, Debajyoti Mazumder, Debasis Mohapatra, Jasabanta Patro

**发布时间:** 2025-09-27

### GPT解析

### 总结

本文提出了一种基于多视图边分类的多图链接预测框架，通过跨层自注意力融合不同层的证据，解决了现有方法忽略层间依赖和可扩展性差的问题。

### 背景

多图能够捕捉共享节点间的多样化关系，但现有预测方法要么将各层折叠，要么独立处理各层，导致失去关键的层间依赖关系且可扩展性差。

### 目的

克服现有方法的局限性，更好地捕捉层间依赖关系并提高可扩展性。

### 方法

将多图链接预测构架为多视图边分类，为每对节点构建按层排列的边视图序列，应用跨层自注意力融合目标层证据；提出两种模型实例：Trans-SLE（基于静态嵌入的轻量级transformer）和Trans-GAT（结合GAT编码器和transformer融合）；引入Union-Set候选池和两种无泄漏协议（跨层和归纳子图泛化）。

### 主要发现

在六个公开多图数据集上，与强基线方法（MELL、HOPLP-MUL、RMNE）相比，在宏观F1分数上取得了一致的提升。

### 结论

所提出的方法简单、可扩展，能够有效捕捉多图中的层间依赖关系，且与预计算嵌入和GNN编码器兼容。

### 翻译

多图捕获共享节点之间的多样化关系。大多数预测方法要么折叠各层，要么独立处理各层。这失去了关键的层间依赖关系，并且在可扩展性方面存在困难。为了克服这一点，我们将多图链接预测构架为多视图边分类。对于每对节点，我们构建一个按层排列的边视图序列，并应用跨层自注意力来融合目标层的证据。我们提出了两种模型作为此框架的实例：Trans-SLE（一种基于静态嵌入的轻量级transformer）和Trans-GAT（结合了特定层的GAT编码器和transformer融合）。为确保可扩展性和公平性，我们引入了Union-Set候选池和两种无泄漏协议：跨层和归纳子图泛化。在六个公开多图数据集上的实验表明，与强基线方法（MELL、HOPLP-MUL、RMNE）相比，我们的方法在宏观F1分数上取得了一致的提升。我们的方法简单、可扩展，并且与预计算嵌入和GNN编码器兼容。


### 论文摘要

Multiplex graphs capture diverse relations among shared nodes. Most predictors either collapse layers or treat them independently. This loses crucial inter-layer dependencies and struggles with scalability. To overcome this, we frame multiplex link prediction as multi-view edge classification. For each node pair, we construct a sequence of per-layer edge views and apply cross-layer self-attention to fuse evidence for the target layer. We present two models as instances of this framework: Trans-SLE, a lightweight transformer over static embeddings, and Trans-GAT, which combines layer-specific GAT encoders with transformer fusion. To ensure scalability and fairness, we introduce a Union--Set candidate pool and two leakage-free protocols: cross-layer and inductive subgraph generalization. Experiments on six public multiplex datasets show consistent macro-F_1 gains over strong baselines (MELL, HOPLP-MUL, RMNE). Our approach is simple, scalable, and compatible with both precomputed embeddings and GNN encoders.

---

## 52. 论文ID: 2509.23347v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2509.23347v1.json'

---

## 53. Towards Quantum-Ready Blockchain Fraud Detection via Ensemble Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.23101v1](http://arxiv.org/abs/2509.23101v1)

**作者:** M. Z. Haider, Tayyaba Noreen, M. Salman

**发布时间:** 2025-09-27

### GPT解析

### 总结

研究提出了一种集成图神经网络框架用于区块链欺诈检测，实现了高召回率同时保持低假阳性率，并设计了量子就绪架构以适应未来技术发展。

### 背景

区块链业务应用和加密货币实现了安全、去中心化的价值转移，但其伪匿名性为非法活动创造了机会，对反洗钱执法构成挑战。

### 目的

开发能够捕捉区块链网络中结构和时间依赖关系、同时对噪声、不平衡和对抗性行为保持稳健性的模型，以增强区块链欺诈检测能力。

### 方法

提出整合图卷积网络(GCN)、图注意力网络(GAT)和图同构网络(GIN)的集成框架，使用真实Elliptic数据集，采用调优的软投票集成方法。

### 主要发现

该方法在非法交易上实现高召回率，同时将假阳性率保持在1%以下，优于单个GNN模型和基线方法；模块化架构包含量子就绪设计钩子，允许未来量子技术集成。

### 结论

集成GNN是实时加密货币监控的实用且前瞻性解决方案，提供即时的反洗钱效用，并为量子增强的金融安全分析提供发展路径。

### 翻译

区块链业务应用和加密货币可实现安全、去中心化的价值转移，但其伪匿名性为非法活动创造了机会，对监管者和交易所的反洗钱执法构成挑战。在区块链网络中检测欺诈交易需要能够捕捉结构和时间依赖关系同时保持对噪声、不平衡和对抗行为稳健性的模型。在本工作中，我们提出了一种整合图卷积网络(GCN)、图注意力网络(GAT)和图同构网络(GIN)的集成框架，以增强区块链欺诈检测。使用真实的Elliptic数据集，我们调优的软投票集成实现了非法交易的高召回率，同时将假阳性率保持在1%以下，优于单个GNN模型和基线方法。模块化架构包含量子就绪设计钩子，允许未来量子特征映射和混合量子经典图神经网络的无缝集成。这确保了可扩展性、稳健性和长期适应性，随着量子计算技术的成熟。我们的发现突显了集成GNN作为实时加密货币监控的实用且前瞻性解决方案，提供了即时的反洗钱效用和通往量子增强金融安全分析的途径。


### 论文摘要

Blockchain Business applications and cryptocurrencies such as enable secure, decentralized value transfer, yet their pseudonymous nature creates opportunities for illicit activity, challenging regulators and exchanges in anti money laundering (AML) enforcement. Detecting fraudulent transactions in blockchain networks requires models that can capture both structural and temporal dependencies while remaining resilient to noise, imbalance, and adversarial behavior. In this work, we propose an ensemble framework that integrates Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Graph Isomorphism Networks (GIN) to enhance blockchain fraud detection. Using the real-world Elliptic dataset, our tuned soft voting ensemble achieves high recall of illicit transactions while maintaining a false positive rate below 1%, beating individual GNN models and baseline methods. The modular architecture incorporates quantum-ready design hooks, allowing seamless future integration of quantum feature mappings and hybrid quantum classical graph neural networks. This ensures scalability, robustness, and long-term adaptability as quantum computing technologies mature. Our findings highlight ensemble GNNs as a practical and forward-looking solution for real-time cryptocurrency monitoring, providing both immediate AML utility and a pathway toward quantum-enhanced financial security analytics.

---

## 54. GuardNet: Graph-Attention Filtering for Jailbreak Defense in Large Language Models

**论文链接:** [http://arxiv.org/abs/2509.23037v1](http://arxiv.org/abs/2509.23037v1)

**作者:** Javad Forough, Mohammad Maheri, Hamed Haddadi

**发布时间:** 2025-09-27

### GPT解析

### 总结

本文提出了一种名为GuardNet的分层过滤框架，用于检测和过滤大型语言模型(LLMs)中的越狱攻击(jailbreak attacks)，显著提高了模型的安全性、可靠性和可信度。

### 背景

大型语言模型(LLMs)越来越容易受到越狱攻击的影响，这些攻击是绕过对齐约束并诱导未授权或有害行为的对抗性提示，削弱了LLM输出的安全性、可靠性和可信度，在医疗保健、金融和法律合规等领域构成关键风险。

### 目的

提出GuardNet，一个分层过滤框架，用于在推理前检测和过滤越狱提示，保护LLM免受对抗性攻击。

### 方法

GuardNet构建结构化图，结合顺序链接、句法依赖和注意力派生的标记关系，捕获越狱行为指示的语言结构和上下文模式。应用图神经网络在两个级别：(i)提示级过滤器检测全局对抗提示；(ii)标记级过滤器精确定位细粒度的对抗跨度。

### 主要发现

在三个数据集和多种攻击设置上的实验表明，GuardNet显著优于先前的防御方法。在LLM-Fuzzer上，提示级F1分数从66.4%提高到99.8%；在PLeak数据集上，从67-79%提高到94%以上。在标记级别，F1从48-75%提高到74-91%，IoU增益高达+28%。

### 结论

尽管GuardNet结构复杂，但它保持了可接受的延迟，并且在跨域评估中泛化良好，使其成为针对现实世界LLM部署中越狱威胁的实用且强大的防御。

### 翻译

大型语言模型(LLMs)越来越容易受到越狱攻击的影响，这些攻击是绕过对齐约束并诱导未授权或有害行为的对抗性提示。这些漏洞削弱了LLM输出的安全性、可靠性和可信度，在医疗保健、金融和法律合规等领域构成关键风险。在本文中，我们提出GuardNet，一个分层过滤框架，用于在推理前检测和过滤越狱提示。GuardNet构建结构化图，结合顺序链接、句法依赖和注意力派生的标记关系，以捕获越狱行为指示的语言结构和上下文模式。然后应用图神经网络在两个级别：(i)提示级过滤器检测全局对抗提示；(ii)标记级过滤器精确定位细粒度的对抗跨度。在三个数据集和多种攻击设置上的广泛实验表明，GuardNet显著优于先前的防御方法。它在LLM-Fuzzer上将提示级F1分数从66.4%提高到99.8%，在PLeak数据集上从67-79%提高到94%以上。在标记级别，GuardNet将F1从48-75%提高到74-91%，IoU增益高达+28%。尽管结构复杂，GuardNet保持了可接受的延迟，并且在跨域评估中泛化良好，使其成为针对现实世界LLM部署中越狱威胁的实用且强大的防御。


### 论文摘要

Large Language Models (LLMs) are increasingly susceptible to jailbreak attacks, which are adversarial prompts that bypass alignment constraints and induce unauthorized or harmful behaviors. These vulnerabilities undermine the safety, reliability, and trustworthiness of LLM outputs, posing critical risks in domains such as healthcare, finance, and legal compliance. In this paper, we propose GuardNet, a hierarchical filtering framework that detects and filters jailbreak prompts prior to inference. GuardNet constructs structured graphs that combine sequential links, syntactic dependencies, and attention-derived token relations to capture both linguistic structure and contextual patterns indicative of jailbreak behavior. It then applies graph neural networks at two levels: (i) a prompt-level filter that detects global adversarial prompts, and (ii) a token-level filter that pinpoints fine-grained adversarial spans. Extensive experiments across three datasets and multiple attack settings show that GuardNet substantially outperforms prior defenses. It raises prompt-level F$_1$ scores from 66.4\% to 99.8\% on LLM-Fuzzer, and from 67-79\% to over 94\% on PLeak datasets. At the token level, GuardNet improves F$_1$ from 48-75\% to 74-91\%, with IoU gains up to +28\%. Despite its structural complexity, GuardNet maintains acceptable latency and generalizes well in cross-domain evaluations, making it a practical and robust defense against jailbreak threats in real-world LLM deployments.

---

## 55. OptimES: Optimizing Federated Learning Using Remote Embeddings for Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.22922v1](http://arxiv.org/abs/2509.22922v1)

**作者:** Pranjal Naman, Yogesh Simmhan

**发布时间:** 2025-09-26

**备注:** Extended full-length version of paper that appeared at Euro-Par 2024:  "Optimizing Federated Learning Using Remote Embeddings for Graph Neural  Networks", Pranjal Naman and Yogesh Simmhan, in International European  Conference on Parallel and Distributed Computing (Euro-Par), 2024. DOI:  https://doi.org/10.1007/978-3-031-69766-1_32

### GPT解析

### 总结

该论文提出了一种名为OptimES的优化联邦图神经网络训练框架，通过远程邻域剪枝、嵌入推送与本地训练重叠以及动态嵌入拉取等技术，有效降低了通信成本和训练时间。

### 背景

图神经网络在近年快速发展，但在现实应用中如图金融交易网络和医疗网络，数据分散在不同所有者处，因隐私问题无法聚合。联邦学习虽能解决隐私问题并利用并行性，但现有方法如EmbC因高通信成本导致性能下降。

### 目的

解决联邦GNN训练中因高通信成本导致的性能限制，优化训练框架以减少网络成本和训练时间。

### 方法

提出OptimES框架，采用远程邻域剪枝技术，将嵌入推送到服务器与本地训练重叠进行，并动态拉取嵌入以减少网络开销。

### 主要发现

对于Reddit和Products等大型密集图，OptimES比EmbC收敛速度快约3.5倍，比默认联邦GNN学习准确率高约16%；对于Arxiv和Papers等稀疏图，虽然准确率提升有限，但达到目标准确速度快约11倍。

### 结论

OptimES框架通过优化通信策略显著提高了联邦GNN训练效率，特别是在大型密集图中效果更为明显，为隐私保护的图数据训练提供了有效解决方案。

### 翻译

图神经网络(GNNs)近年来因其能够从图数据结构中学习有意义表示而经历了快速发展。然而，在大多数现实场景中，如金融交易网络和医疗网络，这些数据分散在不同数据所有者处，由于隐私问题无法聚合。联邦学习(FL)已成为一种可行的机器学习方法，通过迭代聚合在分散数据上训练的本地模型来训练共享模型，这解决了隐私问题同时利用了并行性。最先进的方法通过服务器共享边界顶点的远程嵌入(EmbC)来提高尊重隐私的联邦GNN训练收敛准确率。然而，它们因高通信成本导致性能受限。在本文中，我们提出了OptimES，一种优化的联邦GNN训练框架，采用远程邻域剪枝，将嵌入推送到服务器与本地训练重叠，并动态拉取嵌入以减少网络成本和训练时间。我们对四种常见图数据集(最多包含111M个顶点和1.8B条边)进行了严格评估。我们发现，由于嵌入的预推送导致的每轮准确率适度下降，对于Reddit和Products这样大型密集图的训练时间减少更为显著，比EmbC收敛速度快约3.5倍，比默认联邦GNN学习准确率高约16%。对于Arxiv和Papers这样更稀疏的图，虽然比默认联邦GNN的准确率提升有限，但比EmbC达到目标准确速度快约11倍。


### 论文摘要

Graph Neural Networks (GNNs) have experienced rapid advancements in recent years due to their ability to learn meaningful representations from graph data structures. However, in most real-world settings, such as financial transaction networks and healthcare networks, this data is localized to different data owners and cannot be aggregated due to privacy concerns. Federated Learning (FL) has emerged as a viable machine learning approach for training a shared model that iteratively aggregates local models trained on decentralized data. This addresses privacy concerns while leveraging parallelism. State-of-the-art methods enhance the privacy-respecting convergence accuracy of federated GNN training by sharing remote embeddings of boundary vertices through a server (EmbC). However, they are limited by diminished performance due to large communication costs. In this article, we propose OptimES, an optimized federated GNN training framework that employs remote neighbourhood pruning, overlapping the push of embeddings to the server with local training, and dynamic pulling of embeddings to reduce network costs and training time. We perform a rigorous evaluation of these strategies for four common graph datasets with up to $111M$ vertices and $1.8B$ edges. We see that a modest drop in per-round accuracy due to the preemptive push of embeddings is out-stripped by the reduction in per-round training time for large and dense graphs like Reddit and Products, converging up to $\approx 3.5\times$ faster than EmbC and giving up to $\approx16\%$ better accuracy than the default federated GNN learning. While accuracy improvements over default federated GNNs are modest for sparser graphs like Arxiv and Papers, they achieve the target accuracy about $\approx11\times$ faster than EmbC.

---

## 56. Lexicon-Enriched Graph Modeling for Arabic Document Readability Prediction

**论文链接:** [http://arxiv.org/abs/2509.22870v1](http://arxiv.org/abs/2509.22870v1)

**作者:** Passant Elchafei, Mayar Osama, Mohamed Rageh, Mervat Abuelkheir

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究提出了一种结合词汇资源的基于图的方法来预测阿拉伯语文档级别的可读性，并通过实验验证了其有效性。

### 背景

这是BAREC 2025共享任务受限轨道的一部分，旨在提高阿拉伯语文档的可读性预测水平。

### 目的

开发一种能够准确预测阿拉伯语文档可读性的系统，特别是在文档级别和句子级别。

### 方法

将文档建模为句子级图，节点表示句子和词元，边捕获语言关系；使用SAMER词汇表特征和阿拉伯语transformer模型上下文嵌入增强句子节点；将图神经网络和transformer句子编码器作为独立分支训练，通过后期融合组合预测；使用最大池化聚合句子级输出进行文档级预测。

### 主要发现

混合方法在多个可读性指标上优于单独的GNN或transformer分支；融合在文档级别具有优势；仅使用GNN的方法在句子级可读性精确预测方面仍然更强。

### 结论

结合图神经网络和transformer的混合方法在文档级可读性预测中表现更优，而GNN-only方法在句子级预测中更为精确。

### 翻译

我们提出了一种结合词汇资源的基于图的方法来预测阿拉伯语文档级别的可读性，这是BAREC 2025共享任务受限轨道的一部分。我们的系统将每个文档建模为句子级图，其中节点表示句子和词元，边捕获语言关系，如词汇共现和类别成员关系。句子节点使用SAMER词汇表的特征以及阿拉伯语transformer模型的上下文嵌入进行增强。图神经网络(GNN)和transformer句子编码器作为两个独立分支进行训练，并在推理时通过后期融合组合它们的预测。对于文档级预测，使用最大池化聚合句子级输出以反映最困难的句子。实验结果表明，这种混合方法在多个可读性指标上优于单独的GNN或transformer分支。总体而言，研究结果表明融合在文档级别具有优势，但仅使用GNN的方法在句子级可读性的精确预测方面仍然更强。


### 论文摘要

We present a graph-based approach enriched with lexicons to predict document-level readability in Arabic, developed as part of the Constrained Track of the BAREC Shared Task 2025. Our system models each document as a sentence-level graph, where nodes represent sentences and lemmas, and edges capture linguistic relationships such as lexical co-occurrence and class membership. Sentence nodes are enriched with features from the SAMER lexicon as well as contextual embeddings from the Arabic transformer model. The graph neural network (GNN) and transformer sentence encoder are trained as two independent branches, and their predictions are combined via late fusion at inference. For document-level prediction, sentence-level outputs are aggregated using max pooling to reflect the most difficult sentence. Experimental results show that this hybrid method outperforms standalone GNN or transformer branches across multiple readability metrics. Overall, the findings highlight that fusion offers advantages at the document level, but the GNN-only approach remains stronger for precise prediction of sentence-level readability.

---

## 57. Neighborhood Sampling Does Not Learn the Same Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2509.22868v1](http://arxiv.org/abs/2509.22868v1)

**作者:** Zehao Niu, Mihai Anitescu, Jie Chen

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究探讨了邻域采样在图神经网络训练中的系统性行为，通过神经切线核理论分析表明不同采样方法在有限样本下表现不同且不存在绝对最优方法。

### 背景

邻域采样是大规模图神经网络训练的重要组成部分，可抑制邻域大小指数增长并保持合理资源消耗，但其系统性行为尚未被充分理解。

### 目的

通过理论分析研究邻域采样的系统性行为，特别是几种已建立采样方法及其对应的后验高斯过程特性。

### 方法

使用神经切线核作为分析工具，基于神经网络的无限宽对应物——高斯过程来表征神经网络训练动态。

### 主要发现

有限样本下不同采样方法对应不同的后验高斯过程，但随着样本量增加会收敛到同一后验；后验协方差不可比较，表明没有采样方法能够主导其他方法。

### 结论

邻域采样方法的选择取决于具体应用场景和样本量限制，不存在适用于所有情况的绝对最优采样方法。

### 翻译

邻域采样是大规模图神经网络训练中的一个重要组成部分。它抑制了网络各层邻域大小的指数级增长，并保持了可行的内存消耗和时间成本。虽然它在实践中已成为标准实现，但其系统性行为尚未得到充分理解。我们使用神经切线核这一工具进行理论分析，它基于神经网络的无限宽对应物——高斯过程(GPs)来表征(类似的)神经网络训练动态。我们研究了几种已建立的邻域采样方法及其对应的后验高斯过程。在有限样本下，这些后验各不相同，但随着样本量的增加，它们会收敛到同一个后验。此外，作为预测均方误差下界的后验协方差是不可比较的，这与观察结果一致，即没有一种采样方法能够主导其他方法。


### 论文摘要

Neighborhood sampling is an important ingredient in the training of large-scale graph neural networks. It suppresses the exponential growth of the neighborhood size across network layers and maintains feasible memory consumption and time costs. While it becomes a standard implementation in practice, its systemic behaviors are less understood. We conduct a theoretical analysis by using the tool of neural tangent kernels, which characterize the (analogous) training dynamics of neural networks based on their infinitely wide counterparts -- Gaussian processes (GPs). We study several established neighborhood sampling approaches and the corresponding posterior GP. With limited samples, the posteriors are all different, although they converge to the same one as the sample size increases. Moreover, the posterior covariance, which lower-bounds the mean squared prediction error, is uncomparable, aligning with observations that no sampling approach dominates.

---

## 58. Visual serial processing deficits explain divergences in human and VLM reasoning

**论文链接:** [http://arxiv.org/abs/2509.25142v1](http://arxiv.org/abs/2509.25142v1)

**作者:** Nicholas Budny, Kia Ghods, Declan Campbell, Raja Marjieh, Amogh Joshi, Sreejan Kumar, Jonathan D. Cohen, Taylor W. Webb, Thomas L. Griffiths

**发布时间:** 2025-09-29

### GPT解析

### 总结

研究探讨了视觉语言模型(VLMs)在简单视觉推理任务中表现不如人类的原因，发现序列处理能力不足是关键因素。

### 背景

VLMs在标准基准上取得成功，但在看似简单的视觉推理任务中无法匹敌人类表现。

### 目的

探究VLMs与人类在视觉推理任务中表现差异的原因，特别是关注序列处理能力的影响。

### 方法

比较人类和VLM在三个领域(几何推理、感知计数、心理旋转)的任务表现，这些任务被设计为改变序列处理需求，通过操纵几何概念复杂度、感知个体化负荷和转换难度等因素来改变序列处理负荷。

### 主要发现

所有领域都显示了一致模式：VLM准确率下降与人类反应时间增加(作为序列处理负荷的代理)有强相关性；当任务需要更复杂的序列处理时，VLM与人类的表现差距会可靠地扩大。

### 结论

序列、视觉基础推理的局限性是将当前VLM与人类区分开来的基本瓶颈。

### 翻译

为什么视觉语言模型(VLMs)尽管在标准基准上取得成功，但在看似简单的视觉推理任务中往往无法匹敌人类表现？尽管潜在的计算原理仍有争议，但我们假设一个关键因素是视觉基础序列处理的缺陷。为了验证这一假设，我们在三个不同领域(几何推理、感知计数、心理旋转)设计了任务，比较了人类和VLM的表现，这些任务通过操纵几何概念复杂度、感知个体化负荷和转换难度等因素来改变序列处理需求。在所有领域，我们的结果揭示了一致的模式：VLM准确率的降低与人类反应时间的增加(用作序列处理负荷的代理)有很强的相关性。随着任务需要更复杂的序列处理——无论是组合概念、计数项目还是执行心理转换——VLM与人类的性能差距会可靠地扩大。这些发现支持了我们的假设，表明序列、视觉基础推理的局限性是将当前VLM与人类区分开来的基本瓶颈。


### 论文摘要

Why do Vision Language Models (VLMs), despite success on standard benchmarks, often fail to match human performance on surprisingly simple visual reasoning tasks? While the underlying computational principles are still debated, we hypothesize that a crucial factor is a deficit in visually-grounded serial processing. To test this hypothesis, we compared human and VLM performance across tasks designed to vary serial processing demands in three distinct domains: geometric reasoning, perceptual enumeration, and mental rotation. Tasks within each domain varied serial processing load by manipulating factors such as geometric concept complexity, perceptual individuation load, and transformation difficulty. Across all domains, our results revealed a consistent pattern: decreased VLM accuracy was strongly correlated with increased human reaction time (used as a proxy for serial processing load). As tasks require more demanding serial processing -- whether composing concepts, enumerating items, or performing mental transformations -- the VLM-human performance gap widens reliably. These findings support our hypothesis, indicating that limitations in serial, visually grounded reasoning represent a fundamental bottleneck that distinguishes current VLMs from humans.

---

## 59. Vision-and-Language Navigation with Analogical Textual Descriptions in LLMs

**论文链接:** [http://arxiv.org/abs/2509.25139v1](http://arxiv.org/abs/2509.25139v1)

**作者:** Yue Zhang, Tianyi Ma, Zun Wang, Yanyuan Qiao, Parisa Kordjamshidi

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一种改进的视觉语言导航代理方法，通过整合多视角文本描述和基于文本的类比推理，增强代理的全局场景理解和空间推理能力，从而提高导航性能。

### 背景

将大型语言模型集成到具身AI模型中日益普遍。然而，现有的基于零样本LLM的视觉语言导航代理存在局限性：要么将图像编码为文本场景描述(可能过度简化视觉细节)，要么处理原始图像输入(可能无法捕捉高级推理所需的抽象语义)。

### 目的

改进导航代理的上下文理解能力，通过整合多视角文本描述促进图像间的类比推理，从而增强代理的全局场景理解和空间推理能力，实现更准确的动作决策。

### 方法

通过整合来自多个视角的文本描述，促进图像间的类比推理。利用基于文本的类比推理，增强代理的全局场景理解和空间推理能力，从而做出更准确的动作决策。

### 主要发现

在R2R数据集上的实验表明，该方法显著提高了导航性能，证明了多视角文本描述和类比推理在视觉语言导航任务中的有效性。

### 结论

通过整合多视角文本描述和利用基于文本的类比推理，可以显著提升视觉语言导航代理的性能，为未来具身AI与大型语言模型的集成提供了新的思路。

### 翻译

将大型语言模型集成到具身AI模型中日益普遍。然而，现有的基于零样本LLM的视觉语言导航代理要么将图像编码为文本场景描述，可能过度简化视觉细节；要么处理原始图像输入，可能无法捕捉高级推理所需的抽象语义。在本文中，我们通过整合来自多个视角的文本描述，促进图像间的类比推理，从而改进导航代理的上下文理解。通过利用基于文本的类比推理，代理增强了全局场景理解和空间推理能力，从而做出更准确的动作决策。我们在R2R数据集上评估了我们的方法，实验结果表明导航性能有显著提升。


### 论文摘要

Integrating large language models (LLMs) into embodied AI models is becoming increasingly prevalent. However, existing zero-shot LLM-based Vision-and-Language Navigation (VLN) agents either encode images as textual scene descriptions, potentially oversimplifying visual details, or process raw image inputs, which can fail to capture abstract semantics required for high-level reasoning. In this paper, we improve the navigation agent's contextual understanding by incorporating textual descriptions from multiple perspectives that facilitate analogical reasoning across images. By leveraging text-based analogical reasoning, the agent enhances its global scene understanding and spatial reasoning, leading to more accurate action decisions. We evaluate our approach on the R2R dataset, where our experiments demonstrate significant improvements in navigation performance.

---

## 60. Social 3D Scene Graphs: Modeling Human Actions and Relations for Interactive Service Robots

**论文链接:** [http://arxiv.org/abs/2509.24966v1](http://arxiv.org/abs/2509.24966v1)

**作者:** Ermanno Bartoli, Dennis Rotondi, Buwei He, Patric Jensfelt, Kai O. Arras, Iolanda Leite

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究引入了社交3D场景图表示方法，捕捉环境中人的属性、活动和关系，包括本地和远程关系，使用开放词汇框架，并创建新基准数据集，实验表明该方法能改进人类活动预测和环境关系推理，为社交智能机器人铺平道路。

### 背景

理解人们与环境和彼此的互动对机器人社交行为至关重要。现有3D场景图方法忽略了场景中的人，主要因缺乏人-环境关系标注。同时，现有方法仅捕捉单帧图像的开放词汇关系，限制了长距离交互建模能力。

### 目的

开发社交3D场景图表示方法，捕捉环境中人的属性、活动和关系；创建包含全面人-场景关系注释的合成环境和多样化查询类型的新基准，用于评估3D社交场景理解。

### 方法

引入社交3D场景图表示，一种增强的3D场景图，使用开放词汇框架捕捉环境中人的属性、活动和本地及远程关系；创建新基准数据集，包含合成环境和全面的人-场景关系注释及多样化查询类型。

### 主要发现

实验表明，社交3D场景图表示方法改进了人类活动预测能力，增强了对人类-环境关系的推理能力。

### 结论

社交3D场景图为开发具有社会智能的机器人铺平了道路。

### 翻译

理解人们如何与周围环境和彼此互动对于使机器人能够以符合社会规范且具有情境感知能力的方式行动至关重要。虽然3D场景图已成为场景理解的一种强大语义表示，但现有方法在很大程度上忽略了场景中的人，这部分是由于缺乏标注的人-环境关系。此外，现有方法通常只捕捉来自单帧图像的开放词汇关系，这限制了它们对观察内容之外的长距离交互进行建模的能力。我们引入了社交3D场景图，这是一种增强的3D场景图表示，使用开放词汇框架捕捉环境中的人、他们的属性、活动和关系，包括本地和远程关系。此外，我们引入了一个新的基准，包含具有全面人-场景关系注释的合成环境和多样化的查询类型，用于评估3D社交场景理解。实验证明，我们的表示方法改进了人类活动预测和对人类-环境关系的推理，为具有社会智能的机器人铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有3D场景图方法忽略人类及其活动关系的问题，以及无法建模超出单帧观察范围的远程交互问题。这个问题很重要，因为服务机器人需要理解人类如何与环境互动及其含义，才能实现符合社会规范、有预见性和情境感知的行为，例如识别休闲区域避免阻挡视线，或识别用餐需求提供相应服务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了3D场景图、视觉-语言模型(VLM)、大型语言模型(LLM)和头部姿态估计等现有技术。设计思路是扩展传统3D场景图以包含人类节点和活动关系边，通过理解人的视野、身体姿势和社会属性来估计活动，并利用人的注视方向推断远程交互。作者设计了ReaSoN模块，包含活动描述符、交互上下文估计器、活动求解器和活动合并四个子模块来实现这一目标。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是扩展传统3D场景图，包含代表人类及其活动的节点和边，通过人的注视方向推断其与环境中物体的交互关系。整体流程分为四步：1)活动描述符检测人类并生成行为描述，区分本地和远程活动；2)交互上下文估计器估计头部姿势和可见实体；3)活动求解器使用LLM验证远程活动；4)活动合并使用语义框架聚类活动并基于频率修剪，生成最终的社交3D场景图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出社交3D场景图表示，扩展传统3D场景图包含人类活动；2)设计ReaSoN模块捕获本地和远程活动；3)创建SocialGraph3D基准数据集；4)展示在人类活动预测和关系推理上的改进。相比之前工作，不同之处在于：明确包含人类及其活动，不仅关注空间关系，能捕获远程交互而非仅单帧关系，专注于人类-环境交互而非仅物体间功能关系，并提供公开可用基准数据集。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了社交3D场景图表示方法和ReaSoN构建模块，通过建模人类活动及其与环境的关系，显著提升了机器人在人类环境中的社交理解和交互能力。'}


### 论文摘要

Understanding how people interact with their surroundings and each other is essential for enabling robots to act in socially compliant and context-aware ways. While 3D Scene Graphs have emerged as a powerful semantic representation for scene understanding, existing approaches largely ignore humans in the scene, also due to the lack of annotated human-environment relationships. Moreover, existing methods typically capture only open-vocabulary relations from single image frames, which limits their ability to model long-range interactions beyond the observed content. We introduce Social 3D Scene Graphs, an augmented 3D Scene Graph representation that captures humans, their attributes, activities and relationships in the environment, both local and remote, using an open-vocabulary framework. Furthermore, we introduce a new benchmark consisting of synthetic environments with comprehensive human-scene relationship annotations and diverse types of queries for evaluating social scene understanding in 3D. The experiments demonstrate that our representation improves human activity prediction and reasoning about human-environment relations, paving the way toward socially intelligent robots.

---

## 61. CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D

**论文链接:** [http://arxiv.org/abs/2509.24528v1](http://arxiv.org/abs/2509.24528v1)

**作者:** Mohamad Amin Mirzaei, Pantea Amoie, Ali Ekhterachian, Matin Mirzababaei

**发布时间:** 2025-09-29

**备注:** 9 pages without the refrences, 4 figures, sybmitted for ICLR 2026  conference

### GPT解析

### 总结

该研究提出了一种改进的3D场景理解方法，通过结合SemanticSAM和上下文感知的CLIP编码策略，解决了现有方法中掩码碎片化和语义不准确的问题，显著提升了3D语义分割和对象检索任务的性能。

### 背景

3D场景理解对于具身AI和机器人技术至关重要，支持交互和导航的可靠感知。现有方法通过视觉语言模型生成2D掩码并投影到3D空间，实现零样本、开放词汇的3D语义映射，但存在掩码碎片化和语义不准确的问题。

### 目的

解决现有3D语义映射方法中掩码碎片化和语义分配不准确的问题，提高在复杂环境中的有效性。

### 方法

1) 利用SemanticSAM与渐进式粒度细化生成更准确和更多的对象级掩码，减轻过度分割问题；2) 采用上下文感知的CLIP编码策略，使用经验确定的权重整合每个掩码的多个上下文视图，提供更丰富的视觉上下文。

### 主要发现

实验结果表明，所提出的方法在多个3D场景理解任务上，包括3D语义分割和从语言查询中检索对象，相比现有方法有显著改进。

### 结论

通过改进掩码生成质量和增强语义上下文，该方法有效提升了3D场景理解能力，为具身AI和机器人技术提供了更可靠的感知支持。

### 翻译

3D场景理解对于具身AI和机器人技术至关重要，支持交互和导航的可靠感知。最近的方法通过视觉语言模型生成2D类别无关掩码，并将其分配嵌入向量，然后投影到3D空间，实现了零样本、开放词汇的3D语义映射。然而，这些方法通常产生碎片化的掩码和不准确的语义分配，这是由于直接使用原始掩码导致的，限制了它们在复杂环境中的有效性。为解决这一问题，我们利用SemanticSAM与渐进式粒度细化来生成更准确和更多的对象级掩码，减轻了普通SAM等掩码生成模型中常见的过度分割问题，并改进了下游3D语义分割。为进一步增强语义上下文，我们采用上下文感知的CLIP编码策略，使用经验确定的权重整合每个掩码的多个上下文视图，提供更丰富的视觉上下文。我们在多个3D场景理解任务上评估了该方法，包括3D语义分割和从语言查询中检索对象，使用了几个基准数据集。实验结果表明与现有方法相比有显著改进，突显了该方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D场景理解中的开放词汇检索问题，即在无需特定任务训练的情况下实现对3D场景中物体的准确识别和检索。这个问题很重要，因为3D场景理解是人工智能体和机器人的基础，支持可靠的感知以实现交互和导航；机器人操作和自主导航需要可靠的3D场景表示；AR/VR系统需要精确的物体级地图；而真实世界环境中存在遮挡、不完整观察以及获取大规模标注3D数据成本过高等挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法的不足：2D分割模型产生碎片化掩码、直接使用CLIP提供有限语义上下文、跨多帧聚合预测引入不一致性。然后借鉴了SemanticSAM用于生成更准确的物体级掩码，利用CLIP等视觉语言模型进行开放词汇分类，以及多视图几何技术。作者设计了渐进式粒度调整策略、上下文感知的CLIP编码策略和在3D中强制执行多视图一致性的方法，以解决现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过渐进式粒度调整生成更准确的物体级掩码，利用上下文感知的CLIP编码提供更丰富的视觉上下文，并在3D空间中强制执行多视图一致性。整体流程包括：1)掩码生成：使用SemanticSAM进行渐进式多粒度掩码生成并过滤；2)上下文感知的CLIP嵌入：为每个掩码提取五种互补视觉裁剪并加权组合；3)3D掩码合并和细化：将2D掩码提升到3D空间，使用体积重叠准则合并掩码；4)物体检索：将自然语言查询转换为结构化形式，挖掘候选物体，验证并选择最终预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)渐进式多粒度掩码生成，减少碎片化问题；2)上下文感知的CLIP编码策略，整合多个上下文视图；3)多视图3D一致性强制执行，生成连贯语义地图；4)自然语言物体检索框架，处理关系和方向约束。相比之前工作，该方法解决了掩码碎片化和语义上下文有限的问题，提高了3D语义分割准确性，生成更连贯的语义地图，并扩展到复杂自然语言查询的物体检索。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CORE-3D通过结合渐进式掩码生成、上下文感知嵌入和多视图3D一致性强制执行，实现了无需训练的开放词汇3D场景理解，显著提高了3D语义分割和自然语言物体检索的准确性。'}


### 论文摘要

3D scene understanding is fundamental for embodied AI and robotics, supporting reliable perception for interaction and navigation. Recent approaches achieve zero-shot, open-vocabulary 3D semantic mapping by assigning embedding vectors to 2D class-agnostic masks generated via vision-language models (VLMs) and projecting these into 3D. However, these methods often produce fragmented masks and inaccurate semantic assignments due to the direct use of raw masks, limiting their effectiveness in complex environments. To address this, we leverage SemanticSAM with progressive granularity refinement to generate more accurate and numerous object-level masks, mitigating the over-segmentation commonly observed in mask generation models such as vanilla SAM, and improving downstream 3D semantic segmentation. To further enhance semantic context, we employ a context-aware CLIP encoding strategy that integrates multiple contextual views of each mask using empirically determined weighting, providing much richer visual context. We evaluate our approach on multiple 3D scene understanding tasks, including 3D semantic segmentation and object retrieval from language queries, across several benchmark datasets. Experimental results demonstrate significant improvements over existing methods, highlighting the effectiveness of our approach.

---

## 62. PhysiAgent: An Embodied Agent Framework in Physical World

**论文链接:** [http://arxiv.org/abs/2509.24524v1](http://arxiv.org/abs/2509.24524v1)

**作者:** Zhihao Wang, Jianxiong Li, Jinliang Zheng, Wencong Zhang, Dongxiu Liu, Yinan Zheng, Haoyi Niu, Junzhi Yu, Xianyuan Zhan

**发布时间:** 2025-09-29

### GPT解析

### 总结

这篇论文提出了PhysiAgent框架，解决了VLA模型泛化能力有限的问题，通过创新的监控、记忆和自我反思机制，实现了VLMs和VLA之间的有效协作，在现实世界机器人任务中取得了显著的性能提升。

### 背景

Vision-Language-Action (VLA)模型已取得显著成功但泛化能力有限；将泛化的Vision-Language Models (VLMs)作为助手整合到VLA中是流行解决方案，但当前方法采用刚性顺序结构，导致协作效果不佳和基础挑战问题。

### 目的

提出一个能够在物理环境中有效操作的具身智能体框架PhysiAgent，通过整合监控、记忆、自我反思机制和轻量级现成工具箱，构建自主脚手架框架，促使VLMs根据VLA的实时熟练度反馈组织不同组件，以最大化发挥VLA能力。

### 方法

开发名为PhysiAgent的具身智能体框架，整合监控、记忆、自我反思机制和轻量级现成工具箱，构建自主脚手架框架，促使VLMs根据VLA的实时熟练度反馈组织不同组件，最大化利用VLA能力。

### 主要发现

在复杂现实世界机器人任务中任务解决性能显著提高；展示了VLMs的有效自我调节；展示了工具之间的连贯协作；展示了框架在执行过程中的适应性进化。

### 结论

PhysiAgent在整合VLMs和VLA方面做出了实际和开创性的努力，有效地将具身智能体框架扎根于现实世界环境中。

### 翻译

视觉-语言-行动模型已取得显著成功，但通常泛化能力有限。为解决这一问题，将泛化的视觉-语言模型作为助手整合到VLA中已成为一种流行解决方案。然而，当前方法通常以刚性的顺序结构结合这些模型：主要使用VLMs进行高级场景理解和任务规划，而VLA仅作为低级动作的执行者，导致协作效果不佳和基础挑战问题。在本文中，我们提出了一个PhysiAgent具身智能体框架，专为在物理环境中有效操作而设计。通过整合监控、记忆、自我反思机制和轻量级现成工具箱，PhysiAgent提供了一个自主的脚手架框架，促使VLMs根据VLA的实时熟练度反馈组织不同组件，以最大限度地发挥VLA的能力。实验结果表明，在复杂的现实世界机器人任务中，任务解决性能有显著提高，展示了VLMs的有效自我调节、工具之间的连贯协作以及框架执行过程中的适应性进化。PhysiAgent在整合VLMs和VLA方面做出了实际和开创性的努力，有效地将具身智能体框架扎根于现实世界环境中。


### 论文摘要

Vision-Language-Action (VLA) models have achieved notable success but often struggle with limited generalizations. To address this, integrating generalized Vision-Language Models (VLMs) as assistants to VLAs has emerged as a popular solution. However, current approaches often combine these models in rigid, sequential structures: using VLMs primarily for high-level scene understanding and task planning, and VLAs merely as executors of lower-level actions, leading to ineffective collaboration and poor grounding challenges. In this paper, we propose an embodied agent framework, PhysiAgent, tailored to operate effectively in physical environments. By incorporating monitor, memory, self-reflection mechanisms, and lightweight off-the-shelf toolboxes, PhysiAgent offers an autonomous scaffolding framework to prompt VLMs to organize different components based on real-time proficiency feedback from VLAs to maximally exploit VLAs' capabilities. Experimental results demonstrate significant improvements in task-solving performance on complex real-world robotic tasks, showcasing effective self-regulation of VLMs, coherent tool collaboration, and adaptive evolution of the framework during execution. PhysiAgent makes practical and pioneering efforts to integrate VLMs and VLAs, effectively grounding embodied agent frameworks in real-world settings.

---

## 63. UI2V-Bench: An Understanding-based Image-to-video Generation Benchmark

**论文链接:** [http://arxiv.org/abs/2509.24427v1](http://arxiv.org/abs/2509.24427v1)

**作者:** Ailing Zhang, Lina Lei, Dehong Kong, Zhixin Wang, Jiaqi Xu, Fenglong Song, Chun-Le Guo, Chang Liu, Fan Li, Jie Chen

**发布时间:** 2025-09-29

### GPT解析

### 总结

UI2V-Bench是一个新型基准测试，用于评估图像到视频生成模型的语义理解和推理能力，填补了现有评估基准的空白。

### 背景

生成扩散模型发展迅速，应用广泛，图像到视频生成已成为视频合成领域的主要焦点。然而，现有评估基准主要关注视频质量和时间一致性，忽视了模型对输入图像语义的理解能力以及生成视频与物理规律和人类常识的一致性。

### 目的

提出UI2V-Bench，一个专注于语义理解和推理的I2V模型评估基准，以填补现有评估框架的空白。

### 方法

引入四个主要评估维度：空间理解、属性绑定、类别理解和推理。设计两种基于多模态大语言模型的评估方法：实例级流水线用于细粒度语义理解，基于反馈的推理流水线用于逐步因果评估。UI2V-Bench包含约500个精心构建的文本-图像对，评估多种开源和闭源I2V模型，并纳入人工评估。

### 主要发现

基于MLLM的评估指标与人工评估结果有很强的一致性，表明UI2V-Bench能够有效评估I2V模型的语义理解和推理能力。

### 结论

UI2V-Bench通过强调语义理解和推理能力，填补了I2V评估的关键空白，为该领域的未来研究和模型开发提供了强有力的框架和数据集。

### 翻译

生成扩散模型正迅速发展并因其广泛的应用而日益受到关注。图像到视频生成已成为视频合成领域的主要焦点。然而，现有的评估基准主要关注视频质量和时间一致性等方面，而 largely 忽视了模型对输入图像中特定主题语义的理解能力，或无法确保生成的视频符合物理规律和人类常识。为了解决这一差距，我们提出了UI2V-Bench，一个专注于语义理解和推理的新型I2V模型评估基准。它引入了四个主要评估维度：空间理解、属性绑定、类别理解和推理。为了评估这些维度，我们设计了两种基于多模态大语言模型的评估方法：一种用于细粒度语义理解的实例级流水线，以及一种基于反馈的推理流水线，可实现逐步因果评估，从而进行更准确的评估。UI2V-Bench包含约500个精心构建的文本-图像对，并对各种开源和闭源I2V模型在所有定义维度上进行了评估。我们进一步纳入了人工评估，结果显示与提出的基于MLLM的指标有很强的一致性。总体而言，UI2V-Bench通过强调语义理解和推理能力，填补了I2V评估中的关键空白，为该领域的未来研究和模型开发提供了强有力的框架和数据集。


### 论文摘要

Generative diffusion models are developing rapidly and attracting increasing attention due to their wide range of applications. Image-to-Video (I2V) generation has become a major focus in the field of video synthesis. However, existing evaluation benchmarks primarily focus on aspects such as video quality and temporal consistency, while largely overlooking the model's ability to understand the semantics of specific subjects in the input image or to ensure that the generated video aligns with physical laws and human commonsense. To address this gap, we propose UI2V-Bench, a novel benchmark for evaluating I2V models with a focus on semantic understanding and reasoning. It introduces four primary evaluation dimensions: spatial understanding, attribute binding, category understanding, and reasoning. To assess these dimensions, we design two evaluation methods based on Multimodal Large Language Models (MLLMs): an instance-level pipeline for fine-grained semantic understanding, and a feedback-based reasoning pipeline that enables step-by-step causal assessment for more accurate evaluation. UI2V-Bench includes approximately 500 carefully constructed text-image pairs and evaluates a range of both open source and closed-source I2V models across all defined dimensions. We further incorporate human evaluations, which show strong alignment with the proposed MLLM-based metrics. Overall, UI2V-Bench fills a critical gap in I2V evaluation by emphasizing semantic comprehension and reasoning ability, offering a robust framework and dataset to support future research and model development in the field.

---

## 64. Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction-Reasoning Synergy

**论文链接:** [http://arxiv.org/abs/2509.24385v1](http://arxiv.org/abs/2509.24385v1)

**作者:** Haijier Chen, Bo Xu, Shoujian Zhang, Haoze Liu, Jiaxuan Lin, Jingrong Wang

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了Vid-LLM，一种基于视频的三维多模态大语言模型，可以直接处理视频输入而不需要外部三维数据，解决了现有3D-MLLMs依赖三维数据输入导致的可扩展性和泛化能力受限的问题。

### 背景

多模态大语言模型在二维视觉-语言推理方面取得了显著进展，但扩展到三维场景理解仍面临重大挑战。

### 目的

开发一种基于视频的3D-MLLM，直接处理视频输入而不需要外部三维数据，使其在实际部署中更加实用。

### 方法

1) 直接使用几何先验提高场景感知性能；2) 设计跨任务适配器(CTA)模块，将三维几何先验与视觉-语言表示对齐；3) 引入度量深度模型，从重建输出中恢复真实尺度几何；4) 采用两阶段蒸馏优化策略进行微调，实现快速收敛和稳定训练。

### 主要发现

该方法在三维问答、三维密集描述和三维视觉定位任务上展现出卓越的多任务能力。

### 结论

Vid-LLM通过直接处理视频输入而不依赖外部三维数据，解决了现有3D-MLLMs的可扩展性和泛化问题，在多种3D任务上表现出色。

### 翻译

多模态大语言模型的最新发展显著提高了二维视觉-语言推理能力。然而，将这些能力扩展到三维场景理解仍然是一个重大挑战。现有的三维多模态大语言模型通常依赖三维数据输入，这限制了其可扩展性和泛化能力。为解决这一限制，我们提出了Vid-LLM，一种基于视频的3D-MLLM，可以直接处理视频输入而不需要外部三维数据，使其在实际部署中更加实用。在我们的方法中，几何先验直接用于提高场景感知性能。为了将几何线索紧凑地集成到MLLM中，我们设计了一个跨任务适配器(CTA)模块，将三维几何先验与视觉-语言表示对齐。为确保几何一致性和完整性，我们引入了一个度量深度模型，从重建输出中恢复真实尺度几何。最后，模型通过两阶段蒸馏优化策略进行微调，实现了快速收敛和稳定训练。在多个基准测试上的广泛实验验证了我们的方法在三维问答、三维密集描述和三维视觉定位任务上的有效性，展示了其卓越的多任务能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将多模态大语言模型的能力扩展到3D场景理解中的挑战。现有的3D多模态大语言模型通常依赖复杂的3D数据输入，这限制了它们的可扩展性和泛化能力，导致数据收集、预处理和计算成本高昂。解决这个问题很重要，因为3D场景理解是机器人导航、增强现实、自动驾驶等应用的基础，而简化输入要求可以降低应用门槛，使技术更容易部署到实际场景中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D-MLLMs的局限性，特别是它们对3D数据的依赖。他们借鉴了多模态大语言模型(如LLaVA)和3D重建技术(如VGGT模型)，并认识到几何结构对语义理解的重要性，以及语义推理反过来为几何建模提供上下文先验。他们设计了紧凑的Vid-LLM模型，核心是Cross-Task Adapter(CTA)，紧密耦合重建与推理，实现几何-语义的交互。模型采用两阶段训练策略，第一阶段从重建模型和多模态LLM分别转移几何和语义知识，第二阶段联合优化所有下游模块。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过视频输入直接进行3D场景重建和视觉-语言推理，而不需要外部3D数据，利用几何先验增强场景感知，并通过跨任务适配器将3D几何先验与视觉语言表示对齐。整体流程：1)使用DINOv2提取视频特征；2)通过Cross-Task Adapter将特征分为几何和语义流；3)几何流通过全局帧注意力估计相机姿态和深度，度量深度模型恢复真实尺度；4)语义流与重建几何融合构建3D补丁；5)3D补丁输入LLM进行空间推理；6)采用两阶段训练策略确保模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出仅使用视频输入的紧凑3D多模态大语言模型Vid-LLM；2)设计Cross-Task Adapter实现几何和语义特征的双向交互；3)引入度量深度模型恢复真实尺度几何；4)采用两阶段训练策略提高训练效率。相比之前工作，Vid-LLM不依赖外部3D数据，通过CTA实现几何和语义的深度融合，将重建和推理紧密耦合，大幅提高了可扩展性和实用性，同时降低了部署门槛。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Vid-LLM通过创新的跨任务适配器设计，首次实现了仅从视频输入中同时进行3D场景重建和视觉-语言推理，大幅降低了3D多模态大语言模型的部署门槛，为实际应用提供了高效、可扩展的解决方案。'}


### 论文摘要

Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities.

---

## 65. Robust Partial 3D Point Cloud Registration via Confidence Estimation under Global Context

**论文链接:** [http://arxiv.org/abs/2509.24275v1](http://arxiv.org/abs/2509.24275v1)

**作者:** Yongqiang Wang, Weigang Li, Wenping Liu, Zhe Xu, Zhiqiang Tian

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为CEGC（全局上下文下的置信度估计）的统一框架，用于解决部分点云配准中的挑战性问题，包括结构歧义、部分可见性和噪声。

### 背景

部分点云配准对自主感知和3D场景理解至关重要，但由于结构歧义、部分可见性和噪声等问题，这一任务仍然具有挑战性。

### 目的

开发一个鲁棒的部分3D点云配准框架，能够在复杂场景中实现精确对齐，并处理各种挑战性条件。

### 方法

CEGC框架包含混合重叠置信度估计模块（集成语义描述符和几何相似性检测重叠区域）和上下文感知匹配策略（使用全局注意力分配软置信度分数），这些分数指导可微分的加权奇异值分解求解器计算精确变换，整个管道自适应地降低不确定区域权重并强调可靠的匹配。

### 主要发现

在ModelNet40、ScanObjectNN和7Scenes 3D视觉数据集上的实验表明，CEGC在准确性、鲁棒性和泛化能力方面优于最先进的方法。

### 结论

CEGC为具有挑战性条件下的部分点云配准提供了一种可解释且可扩展的解决方案。

### 翻译

部分点云配准对于自主感知和3D场景理解至关重要，但由于结构歧义、部分可见性和噪声等问题，它仍然具有挑战性。我们通过提出全局上下文下的置信度估计（CEGC）来解决这些问题，这是一个统一的、基于置信度的鲁棒部分3D配准框架。CEGC通过在共享的全局上下文中联合建模重叠置信度和对应关系可靠性，实现了复杂场景中的精确对齐。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决部分3D点云配准的问题，即当两个点云只有部分重叠时如何准确地对齐它们。这个问题在现实中非常重要，因为在自动驾驶、3D重建、机器人导航等领域，由于遮挡、传感器限制或视角受限，我们获取的点云往往是不完整的。部分重叠情况下的配准比完全重叠更具挑战性，需要准确识别重叠区域、拒绝非重叠区域的异常值，并在结构模糊情况下估计变换，这对实现可靠的3D场景理解至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：经典几何方法如ICP和FGR假设高重叠度和低噪声，在部分重叠环境中效果有限；而大多数基于学习的方法难以在部分到部分条件下泛化。作者借鉴了图神经网络和注意力机制等现有技术，但创新性地提出了一个统一的、由置信度驱动的框架CEGC。该方法通过联合建模重叠置信度和对应可靠性，并利用全局上下文来提高配准的鲁棒性，解决了现有方法未能有效处理的不确定性和结构模糊性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过置信度估计和全局上下文建模来提高部分点云配准的鲁棒性。整体流程分为四个阶段：1)特征提取和增强：使用自适应图神经网络提取局部几何特征，并通过自注意力和交叉注意力增强特征表示；2)重叠区域估计：通过混合重叠置信度估计模块，融合语义描述符和几何相似性来识别可靠的重叠区域；3)上下文感知的对应搜索：使用基于注意力的全局推理为点对应分配软置信度分数，处理模糊性和噪声；4)置信度引导的变换估计：使用可加权的奇异值分解计算精确的变换，其中权重由置信度分数决定。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出混合重叠置信度估计模块，整合语义描述符和几何相似性来检测重叠区域并抑制异常值；2)设计上下文感知匹配策略，利用全局注意力为对应分配软置信度分数；3)提出紧密耦合的流水线，自适应降低不确定区域的权重。与之前工作不同，CEGC是首个联合建模重叠置信度和对应可靠性并在共享全局上下文内工作的统一框架，而之前的方法通常将重叠检测、对应匹配和姿态估计作为分离的模块处理，缺乏一致的不确定性传播。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CEGC提出了一种基于全局上下文置信度估计的鲁棒部分3D点云配准框架，通过联合建模重叠置信度和对应可靠性，实现了在复杂场景中的精确对齐，并在多个数据集上展示了卓越的准确性、鲁棒性和泛化能力。'}


### 论文摘要

Partial point cloud registration is essential for autonomous perception and 3D scene understanding, yet it remains challenging owing to structural ambiguity, partial visibility, and noise. We address these issues by proposing Confidence Estimation under Global Context (CEGC), a unified, confidence-driven framework for robust partial 3D registration. CEGC enables accurate alignment in complex scenes by jointly modeling overlap confidence and correspondence reliability within a shared global context. Specifically, the hybrid overlap confidence estimation module integrates semantic descriptors and geometric similarity to detect overlapping regions and suppress outliers early. The context-aware matching strategy smitigates ambiguity by employing global attention to assign soft confidence scores to correspondences, improving robustness. These scores guide a differentiable weighted singular value decomposition solver to compute precise transformations. This tightly coupled pipeline adaptively down-weights uncertain regions and emphasizes contextually reliable matches. Experiments on ModelNet40, ScanObjectNN, and 7Scenes 3D vision datasets demonstrate that CEGC outperforms state-of-the-art methods in accuracy, robustness, and generalization. Overall, CEGC offers an interpretable and scalable solution to partial point cloud registration under challenging conditions.

---

## 66. Uni4D-LLM: A Unified SpatioTemporal-Aware VLM for 4D Understanding and Generation

**论文链接:** [http://arxiv.org/abs/2509.23828v1](http://arxiv.org/abs/2509.23828v1)

**作者:** Hanyu Zhou, Gim Hee Lee

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了Uni4D-LLM，这是第一个具有时空感知能力的统一视觉语言模型框架，用于4D场景理解和生成，解决了现有方法中的范式差距。

### 背景

视觉语言模型在2D场景理解和生成方面表现出色，但扩展到物理世界仍面临挑战。现有3D和4D方法通常将场景几何嵌入自回归模型进行语义理解，使用扩散模型进行内容生成，导致单个模型难以同时处理这两个任务，特别是在动态4D环境中。

### 目的

开发一个统一的视觉语言模型框架，具有时空感知能力，能够同时处理4D场景理解和生成任务。

### 方法

Uni4D-LLM基于两个关键见解：1)通过提取语义特征和噪声注入外观特征，融入4D几何线索，并通过自适应交叉注意力融合为时空感知的视觉表示；2)使用共享的Transformer架构，将自回归和扩散模型集成到单一LLM中，并通过视觉和语言表示对齐实现统一框架。

### 主要发现

在多个基准测试上的实验表明，Uni4D-LLM实现了与最先进模型相当或更好的结果，并首次实现了4D场景理解和生成的真正统一。

### 结论

Uni4D-LLM成功解决了现有方法中的范式差距，提供了一个统一的框架，能够同时处理4D场景理解和生成任务，为物理世界的视觉语言模型应用提供了新的可能性。

### 翻译

视觉语言模型在2D场景理解和生成方面已展现出强大的性能，但将这种统一性扩展到物理世界仍然是一个开放的挑战。现有的3D和4D方法通常将场景几何嵌入到自回归模型中进行语义理解，并使用扩散模型进行内容生成。这种范式差距使得单个模型难以同时处理这两个任务，特别是在动态4D环境中，时空建模至关重要。我们提出了Uni4D-LLM，这是第一个具有时空感知能力的统一VLM框架，用于4D场景理解和生成。我们的设计由两个关键见解指导：统一需要共享表示和共享架构。通过视觉和语言表示的对齐，Uni4D-LLM在一个基于Transformer的框架内产生理解和生成的预测。在多个基准测试上的大量实验表明，Uni4D-LLM实现了与最先进模型相当或更好的结果，并首次实现了4D场景理解和生成的真正统一。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是现有视觉语言模型(VLMs)在2D场景理解和生成方面表现出色，但无法有效扩展到物理世界(特别是4D场景)的问题。现有方法通常将场景理解和生成分离处理，使用不同的模型架构(自回归模型用于理解，扩散模型用于生成)，导致单一模型无法同时处理这两个任务，尤其是在需要时空建模的动态4D场景中。这个问题很重要，因为物理世界是三维且动态的，统一的4D模型对自动驾驶、机器人交互、虚拟现实等许多应用至关重要，同时也能提高系统效率和性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的设计基于两个关键洞察：1)统一需要共享的表示，提取语义特征和带噪声的外观特征，融入4D几何线索；2)统一需要共享的架构，因为自回归和扩散都基于Transformer骨干。作者借鉴了2D VLMs将图像离散化为文本标记的方法，3D方法将几何嵌入视觉表示的思路，以及4D方法融入时空线索的方式。但作者创新性地解决了这些方法中存在的表示分离和架构分离问题，通过自适应交叉注意力和混合LLM架构实现了真正的统一。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的时空感知视觉表示和混合LLM架构，使单一模型能同时处理4D场景理解和生成任务。整体实现流程分为三阶段：1)统一视觉表示：视频序列编码为视觉和几何潜在表示，提取语义特征和带噪声外观特征，结合时间生成4D几何特征，通过自适应交叉注意力融合；2)混合LLM架构：共享Transformer骨干，自回归头用于理解，扩散头用于生成，使用注意力掩码控制信息流；3)多模态对齐：将视觉表示投影到语言空间，联合优化理解和生成任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出Uni4D-LLM，第一个统一的4D视觉语言大模型；2)设计时空感知的统一视觉表示，通过自适应交叉注意力融合多特征；3)提出混合LLM架构，支持自回归和扩散两种范式；4)整合多样化4D数据集并应用指令微调。相比之前工作，不同之处在于：2D VLMs缺乏几何表示，3D方法将理解和生成分离，现有4D方法也采用分离解决方案，而Uni4D-LLM首次实现了4D场景理解和生成的真正统一，在单一框架内同时处理两种任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Uni4D-LLM通过统一的时空感知视觉表示和混合LLM架构，首次实现了4D场景理解和生成的真正统一，在多种基准测试上取得了竞争性或优于最先进模型的性能。'}


### 论文摘要

Vision-language models (VLMs) have demonstrated strong performance in 2D scene understanding and generation, but extending this unification to the physical world remains an open challenge. Existing 3D and 4D approaches typically embed scene geometry into autoregressive model for semantic understanding and diffusion model for content generation. This paradigm gap prevents a single model from jointly handling both tasks, especially in dynamic 4D settings where spatiotemporal modeling is critical. We propose Uni4D-LLM, the first unified VLM framework with spatiotemporal awareness for 4D scene understanding and generation. Our design is guided by two key insights: 1) Unification requires a shared representation. We extract semantic features for understanding and noisy-injected appearance features for generation, incorporate 4D geometric cues, and fuse them into a spatiotemporal-aware visual representation through adaptive cross-attention. 2) Unification requires a shared architecture. Both autoregression and diffusion are built on Transformer backbones, and this enables integration into a single LLM with task-specific heads. By aligning visual and linguistic representations, our Uni4D-LLM produces predictions for both understanding and generation within one Transformer-based framework. We further apply instruction fine-tuning on diverse 4D vision-language datasets to improve generalization across tasks. Extensive experiments on multiple benchmarks demonstrate that Uni4D-LLM achieves competitive or superior results compared to state-of-the-art models and offers the first true unification of 4D scene understanding and generation.

---

## 67. From Static to Dynamic: a Survey of Topology-Aware Perception in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2509.23641v1](http://arxiv.org/abs/2509.23641v1)

**作者:** Yixiao Chen, Ruining Yang, Xin Chen, Jia He, Dongliang Xu, Yue Yao

**发布时间:** 2025-09-28

**备注:** 13 pages, 3 figures

### GPT解析

### 总结

该综述系统回顾了拓扑感知主题下的四个核心研究方向：向量化地图构建、拓扑结构建模、先验知识融合和基于语言模型的感知。研究发现自动驾驶领域正从静态预构建地图向动态传感器驱动感知转变，各研究方向通过不同方式促进这一转变，共同推动更自适应、可扩展和可解释的自动驾驶系统发展。

### 背景

自动驾驶的关键在于拓扑感知，即对驾驶环境的结构化理解，重点关注车道拓扑和道路语义。

### 目的

系统性地回顾拓扑感知主题下的四个核心研究方向，分析其发展趋势和贡献。

### 方法

对四个核心研究方向进行综述分析：向量化地图构建、拓扑结构建模、先验知识融合和基于语言模型的感知。

### 主要发现

自动驾驶领域正经历从静态、预构建的地图向动态、传感器驱动的感知转变。传统静态地图虽提供语义上下文，但存在构建成本高、难以实时更新、区域泛化能力差等问题，限制了可扩展性。动态表示则利用车载传感器数据进行实时地图构建和拓扑推理。

### 结论

四个研究方向通过紧凑的空间建模、语义关系推理、鲁棒的领域知识集成以及由预训练语言模型支持的多模态场景理解，共同为更自适应、可扩展和可解释的自动驾驶系统铺平了道路。

### 翻译

实现自动驾驶的关键在于拓扑感知，即对驾驶环境进行结构化理解，重点关注车道拓扑和道路语义。本综述系统地回顾了这一主题下的四个核心研究方向：向量化地图构建、拓扑结构建模、先验知识融合和基于语言模型的感知。在这些方向中，我们观察到一种统一趋势：从静态、预构建的地图向动态、传感器驱动的感知转变。具体而言，传统静态地图为自主系统提供了语义上下文。然而，它们构建成本高，难以实时更新，且缺乏区域泛化能力，限制了其可扩展性。相比之下，动态表示利用车载传感器数据进行实时地图构建和拓扑推理。四个研究方向中的每一个都通过紧凑的空间建模、语义关系推理、鲁棒的领域知识集成以及由预训练语言模型支持的多模态场景理解促进了这一转变。它们共同为更自适应、可扩展和可解释的自动驾驶系统铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶中的拓扑感知问题，特别是如何从静态预构建地图转向动态传感器驱动的感知，实现对道路环境的结构化理解。这个问题在现实中很重要，因为自动驾驶系统需要准确理解道路结构、车道连接关系和交通规则以确保安全导航；在研究中也很重要，因为它涉及感知、理解和推理的多个方面，是自动驾驶系统实现高级别智能的关键。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作为一篇综述文章，作者没有设计新方法，而是系统梳理了现有研究，并总结了四个核心研究方向：矢量地图构建、拓扑结构建模、先验知识融合和基于语言模型的感知。作者借鉴了大量现有工作，包括HDMapNet、BEVLaneDet、VectorMapNet、MapTR、TopoMLP、PETR、DriveGPT-4等，这些工作分别属于不同研究方向，共同推动了从静态到动态的拓扑感知发展。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用车载传感器数据实时构建地图和进行拓扑推理，而非依赖预构建的静态地图。整体实现流程包括：利用多模态传感器数据进行感知，构建矢量地图，提取车道拓扑，融合先验知识，以及利用语言模型进行高级场景理解和推理。这种方法通过紧凑空间建模、语义关系推理、鲁棒领域知识集成和预训练语言模型驱动的多模态场景理解，实现更自适应、可扩展和可解释的自动驾驶系统。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '论文的关键创新点在于提供了一个统一的视角和跨模态思维的框架，系统性地总结了四个核心研究方向，揭示了从静态地图向动态、传感器驱动的感知转变的范式。相比之前的工作，论文提供了更全面的视角，不仅关注技术方法，还讨论了数据集、挑战和未来方向，为领域研究者提供了有价值的指导。论文还强调了多模态融合和语言模型在提升系统可解释性和交互性方面的新思路。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文系统综述了自动驾驶中拓扑感知的研究进展，揭示了从静态高清地图向动态、传感器驱动的感知转变的范式，为未来自适应、可扩展和可解释的自动驾驶系统铺平了道路。'}


### 论文摘要

The key to achieving autonomous driving lies in topology-aware perception, the structured understanding of the driving environment with an emphasis on lane topology and road semantics. This survey systematically reviews four core research directions under this theme: vectorized map construction, topological structure modeling, prior knowledge fusion, and language model-based perception. Across these directions, we observe a unifying trend: a paradigm shift from static, pre-built maps to dynamic, sensor-driven perception. Specifically, traditional static maps have provided semantic context for autonomous systems. However, they are costly to construct, difficult to update in real time, and lack generalization across regions, limiting their scalability. In contrast, dynamic representations leverage on-board sensor data for real-time map construction and topology reasoning. Each of the four research directions contributes to this shift through compact spatial modeling, semantic relational reasoning, robust domain knowledge integration, and multimodal scene understanding powered by pre-trained language models. Together, they pave the way for more adaptive, scalable, and explainable autonomous driving systems.

---

## 68. From Fields to Splats: A Cross-Domain Survey of Real-Time Neural Scene Representations

**论文链接:** [http://arxiv.org/abs/2509.23555v1](http://arxiv.org/abs/2509.23555v1)

**作者:** Javed Ahmad, Penggang Gao, Donatien Delehelle, Mennuti Canio, Nikhil Deshpande, Jesús Ortiz, Darwin G. Caldwell, Yonas Teodros Tefera

**发布时间:** 2025-09-28

**备注:** 18 pages

### GPT解析

### 总结

这篇综述探讨了3D高斯溅射(3DGS)作为神经辐射场(NeRF)替代方案的技术优势和应用领域，分析了其在不同领域的适应性、局限性，并提供了神经渲染在真实和虚拟环境中应用的路线图。

### 背景

神经场景表示如NeRF和3DGS已经改变了3D环境的建模、渲染和解释方式。NeRF通过体积渲染引入了视图一致的真实感；3DGS作为一种显式、高效的替代方案迅速崛起，支持高质量渲染、更快的优化，并能集成到混合管道中以增强真实感和任务驱动的场景理解。

### 目的

这篇综述检查了3DGS如何在SLAM、远程存在和远程操作、机器人操作以及3D内容生成等领域被采用，并解释为什么3DGS正在越来越多地取代基于NeRF的方法。

### 方法

围绕统一的研究问题组织综述，分析推动3DGS采用的技术优势、它如何适应不同的输入模态和领域特定约束、以及仍然存在的局限性。通过系统比较特定领域的管道，评估3DGS在真实感、几何保真度和计算效率方面的平衡。

### 主要发现

3DGS在真实感渲染、几何保真度和计算效率之间取得了平衡，使其成为NeRF的强大替代方案，能够满足不同领域对高质量3D表示的共同需求。

### 结论

该综述为利用神经渲染提供了一条路线图，不仅用于图像合成，还用于在真实和虚拟环境中的感知、交互和内容创建。

### 翻译

神经场景表示如神经辐射场(NeRF)和3D高斯溅射(3DGS)已经改变了3D环境的建模、渲染和解释方式。NeRF通过体积渲染引入了视图一致的真实感；3DGS已迅速崛起作为一种显式、高效的替代方案，支持高质量渲染、更快的优化，并能集成到混合管道中以增强真实感和任务驱动的场景理解。这篇综述检查了3DGS如何在SLAM、远程存在和远程操作、机器人操作以及3D内容生成等领域被采用。尽管这些领域有所不同，但它们有共同的目标：真实感渲染、有意义的3D结构和准确的下游任务。我们围绕统一的研究问题组织综述，解释为什么3DGS正在越来越多地取代基于NeRF的方法：推动其采用的技术优势是什么？它如何适应不同的输入模态和领域特定约束？仍然存在哪些局限性？通过系统比较特定领域的管道，我们展示了3DGS如何在真实感、几何保真度和计算效率之间取得平衡。该综述为利用神经渲染提供了一条路线图，不仅用于图像合成，还用于在真实和虚拟环境中的感知、交互和内容创建。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何评估和比较不同神经场景表示方法（特别是NeRF和3DGS）在实时应用中的性能及其在不同领域的适用性问题。这个问题很重要，因为随着NeRF和3DGS等神经场景表示方法的发展，研究者需要了解这些方法的优缺点和适用场景；实时3D场景表示对SLAM、远程呈现、机器人操作和3D内容生成等应用至关重要；3DGS正迅速取代传统方法，需要理解这一趋势的原因和影响。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '这是一篇综述文章，作者通过系统性地回顾和分析现有研究文献来组织内容。他们定义了一系列研究问题（RQs）来组织综述，进行了系统的文献搜索（140篇论文），使用关键词共现分析了解研究趋势，并将现有方法按领域分类比较。他们借鉴了现有调查工作，但采取了不同方法：将3DGS作为核心表示方法，按趋势导向分类，重点放在跨领域应用上，提供比较表格和统一图表。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '由于是综述文章，没有单一方法。论文探讨了两种主要神经场景表示方法：NeRF的核心思想是使用神经网络学习连续体积函数实现照片级真实渲染，通过体积光线投射和alpha合成实现；3DGS的核心思想是将场景显式建模为各向异性高斯集合，通过可微分光栅化实现实时渲染，使用有序alpha合成和参数优化。两种方法都通过损失函数优化参数，但3DGS使用显式高斯原语而非隐式体积表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：跨领域视角分析3DGS应用；定义结构化研究问题组织内容；提供综合比较框架和性能表格；采用趋势导向分类而非详细架构描述；强调应用级见解和跨领域定位。相比之前工作，这篇论文不只关注技术细节，而是关注3DGS作为下一代视觉和图形系统的基础；强调跨领域应用；提供更全面的分析；特别关注实时性能对实际应用的重要性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性地综述和比较3D Gaussian Splatting与Neural Radiance Fields在实时神经场景表示中的应用，揭示了3DGS作为跨领域统一基础表示的潜力，为下一代视觉和图形系统提供了研究路线图。'}


### 论文摘要

Neural scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have transformed how 3D environments are modeled, rendered, and interpreted. NeRF introduced view-consistent photorealism via volumetric rendering; 3DGS has rapidly emerged as an explicit, efficient alternative that supports high-quality rendering, faster optimization, and integration into hybrid pipelines for enhanced photorealism and task-driven scene understanding. This survey examines how 3DGS is being adopted across SLAM, telepresence and teleoperation, robotic manipulation, and 3D content generation. Despite their differences, these domains share common goals: photorealistic rendering, meaningful 3D structure, and accurate downstream tasks. We organize the review around unified research questions that explain why 3DGS is increasingly displacing NeRF-based approaches: What technical advantages drive its adoption? How does it adapt to different input modalities and domain-specific constraints? What limitations remain? By systematically comparing domain-specific pipelines, we show that 3DGS balances photorealism, geometric fidelity, and computational efficiency. The survey offers a roadmap for leveraging neural rendering not only for image synthesis but also for perception, interaction, and content creation across real and virtual environments.

---

## 69. FoR-SALE: Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing

**论文链接:** [http://arxiv.org/abs/2509.23452v1](http://arxiv.org/abs/2509.23452v1)

**作者:** Tanawan Premsri, Parisa Kordjamshidi

**发布时间:** 2025-09-27

**备注:** 9 pages, 3 Tables, 4 Figures, Under Reviewed

### GPT解析

### 总结

FoR-SALE是一种创新方法，通过整合参考系概念到多模态模型中，特别是文本到图像生成任务，解决了模型在非相机视角空间描述上的性能局限。

### 背景

参考系（FoR）是空间推理的基本概念，人类用它来理解和描述空间。多模态语言模型快速发展，是时候将这一长期被忽视的维度整合到这些模型中。在文本到图像（T2I）生成中，即使最先进的模型在从相机以外视角提供空间描述时也会表现出显著的性能差距。

### 目的

解决T2I模型在处理非相机视角空间描述时的性能限制。

### 方法

提出了基于参考系引导的空间调整方法（FoR-SALE），这是基于LLM的扩散编辑框架的扩展。FoR-SALE评估给定文本与初始生成图像之间的对齐情况，并根据空间表达中指定的参考系来优化图像。它使用视觉模块提取图像的空间配置，同时将空间表达映射到相应的相机视角，使语言和视觉之间的对齐可以直接评估。当检测到不对齐时，生成并应用所需的编辑操作，使用新颖的潜在空间操作来调整生成图像的朝向和深度。

### 主要发现

在两个专门设计用于评估参考系空间理解的基准测试上评估了FoR-SALE。该框架仅通过一轮校正，就将最先进的T2I模型的性能提高了最多5.3%。

### 结论

FoR-SALE方法能够有效提升T2I模型在处理不同视角空间描述时的性能。

### 翻译

参考系（FoR）是空间推理中的一个基本概念，人类利用它来理解和描述空间。随着多模态语言模型的快速发展，是时候将这一长期被忽视的维度整合到这些模型中了。特别是在文本到图像（T2I）生成中，即使是最先进的模型，在从相机以外的视角提供空间描述时也会表现出显著的性能差距。为了解决这一局限，我们提出了基于参考系引导的空间调整方法（FoR-SALE），这是基于LLM的扩散编辑（SLD）框架在T2I任务上的扩展。FoR-SALE评估给定文本与初始生成图像之间的对齐情况，并根据空间表达中指定的参考系来优化图像。它使用视觉模块提取图像的空间配置，同时将空间表达映射到相应的相机视角。这种统一的视角可以直接评估语言和视觉之间的对齐情况。当检测到不对齐时，生成并应用所需的编辑操作。FoR-SALE应用新颖的潜在空间操作来调整生成图像的朝向和深度。我们在两个专门设计用于评估参考系空间理解的基准测试上评估了FoR-SALE。我们的框架仅通过一轮校正，就将最先进的T2I模型的性能提高了最多5.3%。


### 论文摘要

Frame of Reference (FoR) is a fundamental concept in spatial reasoning that humans utilize to comprehend and describe space. With the rapid progress in Multimodal Language models, the moment has come to integrate this long-overlooked dimension into these models. In particular, in text-to-image (T2I) generation, even state-of-the-art models exhibit a significant performance gap when spatial descriptions are provided from perspectives other than the camera. To address this limitation, we propose Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing (FoR-SALE), an extension of the Self-correcting LLM-controlled Diffusion (SLD) framework for T2I. For-Sale evaluates the alignment between a given text and an initially generated image, and refines the image based on the Frame of Reference specified in the spatial expressions. It employs vision modules to extract the spatial configuration of the image, while simultaneously mapping the spatial expression to a corresponding camera perspective. This unified perspective enables direct evaluation of alignment between language and vision. When misalignment is detected, the required editing operations are generated and applied. FoR-SALE applies novel latent-space operations to adjust the facing direction and depth of the generated images. We evaluate FoR-SALE on two benchmarks specifically designed to assess spatial understanding with FoR. Our framework improves the performance of state-of-the-art T2I models by up to 5.3% using only a single round of correction.

---

## 70. 论文ID: 2509.23203v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2509.23203v1.json'

---

## 71. Good Weights: Proactive, Adaptive Dead Reckoning Fusion for Continuous and Robust Visual SLAM

**论文链接:** [http://arxiv.org/abs/2509.22910v1](http://arxiv.org/abs/2509.22910v1)

**作者:** Yanwei Du, Jing-Chen Peng, Patricio A. Vela

**发布时间:** 2025-09-26

**备注:** 8 pages, 9 figures, 1 table. Submitted to IEEE Conference

### GPT解析

### 总结

Good Weights算法是一种自适应整合航位推算与视觉SLAM的方法，用于在视觉质量下降的环境中实现连续准确的姿态估计。

### 背景

视觉SLAM在纹理缺失或视觉质量下降的环境（如纯白墙或低光照）中表现不佳，导致姿态估计不准确和跟踪丢失；而机器人配备的航位推算系统在短期表现良好但长期不可靠。

### 目的

开发一种能够自适应整合航位推算与被动视觉SLAM的框架，实现连续准确的帧级姿态估计，提高SLAM系统在视觉质量下降环境中的性能和鲁棒性。

### 方法

Good Weights算法通过自适应权重机制，在视觉跟踪不可靠时增加航位推算的影响，在视觉特征信息强时减少航位推算的影响，同时修改综合SLAM系统中的所有模块以整合航位推算。

### 主要发现

Good Weights算法能够在视觉质量下降的环境中维持姿态跟踪，不过度依赖航位推算，为移动导航提供了实用的解决方案，提高了视觉SLAM的性能和鲁棒性。

### 结论

Good Weights算法通过自适应整合航位推算和视觉SLAM，解决了视觉SLAM在纹理缺失或视觉质量下降环境中的挑战，实验证明其在实际应用中具有显著优势。

### 翻译

鉴于视觉SLAM依赖于外观线索进行定位和场景理解，纹理缺失或视觉退化的环境（例如纯白墙或低光照）会导致姿态估计不佳和跟踪丢失。然而，机器人通常配备传感器，提供某种形式的航位推算里程计，具有合理的短期性能但不可靠的长期性能。这里描述的Good Weights (GW)算法提供了一个框架，能够自适应地将航位推算(DR)与被动视觉SLAM整合，实现连续和准确的帧级姿态估计。重要的是，它描述了如何修改综合SLAM系统中的所有模块，将DR整合到其设计中。自适应权重在视觉跟踪不可靠时增加DR的影响，在视觉特征信息强时减少DR的影响，保持姿态跟踪而不过度依赖DR。Good Weights为移动导航提供了实用的解决方案，提高了视觉SLAM的性能和鲁棒性。在收集的数据集和实际部署中的实验证明了Good Weights的益处。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉SLAM在纹理稀少或视觉质量下降环境中（如纯白墙或低光照条件）导致的姿态估计不准确和跟踪丢失问题。这个问题在现实中非常重要，因为许多实际应用场景（如室内导航、走廊、隧道）都可能遇到这类环境，导致机器人导航系统失效，影响自主定位和地图构建能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到视觉SLAM在纹理稀少环境中的局限性，以及航位推算(DR)传感器在短期内的可靠性。他们设计了一种主动、自适应的融合策略，能够在视觉跟踪不可靠时增加DR影响，而在视觉信息强时减少DR影响。作者借鉴了多传感器融合框架、自适应融合SLAM研究、优化领域中的Levenberg-Marquardt算法和信任区域方法，以及鲁棒估计中的Huber或Cauchy核等技术，但创新性地将其应用于视觉SLAM的整个层次结构中，实现了主动而非被动的融合策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是一种自适应权重方案，根据视觉跟踪质量动态调节航位推算(DR)作为运动先验的影响。实现流程包括：1)引入Qt评分系统(0-1分)量化视觉跟踪质量；2)基于检测到的特征数量和跟踪到的特征数量计算Qt值；3)根据Qt值动态调整DR权重；4)将自适应DR集成到SLAM的各个模块：特征跟踪中提供更准确的运动模型，姿态估计时作为约束保持连续性，局部和全局束调整中强化图连接性。整个系统在视觉强时以视觉为主，视觉弱时DR辅助，视觉失效时DR维持连续性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)主动、自适应的权重方案，根据视觉跟踪质量实时调整DR影响；2)将DR先验全面集成到SLAM的跟踪、局部映射和回环检测模块中；3)基于特征统计的轻量级Qt评分系统；4)实验验证了在低纹理环境中的鲁棒性。相比之前工作，Good Weights具有主动性(而非反应性)，专门针对视觉SLAM设计(而非简单移植LiDAR方法)，使用动态调整权重(而非固定权重)，并在整个SLAM层次结构中实现模块级集成(而非简单传感器融合)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Good Weights提出了一种主动、自适应的航位推算融合框架，通过根据视觉跟踪质量动态调整权重，将DR先验集成到视觉SLAM的各个模块中，显著提高了系统在纹理稀少环境中的鲁棒性和连续性，同时保持了正常条件下的准确性。'}


### 论文摘要

Given that Visual SLAM relies on appearance cues for localization and scene understanding, texture-less or visually degraded environments (e.g., plain walls or low lighting) lead to poor pose estimation and track loss. However, robots are typically equipped with sensors that provide some form of dead reckoning odometry with reasonable short-time performance but unreliable long-time performance. The Good Weights (GW) algorithm described here provides a framework to adaptively integrate dead reckoning (DR) with passive visual SLAM for continuous and accurate frame-level pose estimation. Importantly, it describes how all modules in a comprehensive SLAM system must be modified to incorporate DR into its design. Adaptive weighting increases DR influence when visual tracking is unreliable and reduces when visual feature information is strong, maintaining pose track without overreliance on DR. Good Weights yields a practical solution for mobile navigation that improves visual SLAM performance and robustness. Experiments on collected datasets and in real-world deployment demonstrate the benefits of Good Weights.

---

## 72. TACO-Net: Topological Signatures Triumph in 3D Object Classification

**论文链接:** [http://arxiv.org/abs/2509.24802v1](http://arxiv.org/abs/2509.24802v1)

**作者:** Anirban Ghosh, Ayan Dutta

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为TACO-Net的新型3D物体分类技术，结合拓扑数据分析和图像过滤技术，在点云表示下实现高精度分类。

### 背景

3D物体分类在计算机视觉、机器人和自动驾驶等领域具有重要实际价值。尽管深度学习在点云分类上取得成功，但由于点云的无序性、不规则性和噪声问题，实现高分类准确率仍具挑战性。

### 目的

开发一种新颖的3D物体分类技术，结合拓扑数据分析和图像过滤技术，解决点云表示下的物体分类问题。

### 方法

将每个点云转换为体素化二值三维图像提取区分性拓扑特征，并使用提取的特征集训练轻量级一维卷积神经网络（1D CNN），构建TACO-Net框架。

### 主要发现

TACO-Net在ModelNet40和ModelNet10基准测试上分别达到99.05%和99.52%的准确率，创造了新的最先进水平；在OmniObject3D真实世界数据集上展示了鲁棒性；对十种不同类型的损坏输入表现出强大的恢复能力。

### 结论

TACO-Net通过结合拓扑数据分析和图像过滤技术，实现了3D物体分类的高准确率和鲁棒性，为点云分类提供了新的解决方案。

### 翻译

3D物体分类是一个关键问题，在计算机视觉、机器人和自动驾驶等多个领域具有重要的实际相关性。尽管近年来应用于在物体CAD模型上采样或由LiDAR或RGBD相机捕获的点云的深度学习方法取得了显著成功，但由于无序点云及其不规则性和噪声，实现高分类准确率仍然是一个具有挑战性的问题。为此，我们提出了一种新颖的3D物体分类技术，结合拓扑数据分析和各种图像过滤技术，对使用点云表示的物体进行分类。我们将每个点云转换为体素化二值三维图像，以提取区分性拓扑特征。接下来，我们使用从训练数据集中提取的特征集训练轻量级一维卷积神经网络（1D CNN）。我们的TACO-Net框架在广泛使用的合成基准测试集ModelNet40和ModelNet10上分别达到99.05%和99.52%的准确率，创造了新的最先进水平，并在大规模真实世界OmniObject3D数据集上进一步展示了其鲁棒性。当使用十种不同类型的损坏ModelNet40输入进行测试时，所提出的TACO-Net整体表现出强大的恢复能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云物体的分类问题。点云数据是由LiDAR或RGB-D相机捕获的3D物体的无序点集合，具有无序性、不规则性和噪声等特性，使得准确分类变得困难。这个问题在现实世界中非常重要，广泛应用于自动驾驶系统识别周围环境、机器人在复杂环境中操作、增强现实应用和3D场景理解等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有3D物体分类方法的局限性，大多数方法使用深度学习框架处理图片集合、原始点集或体积形状，但在处理点云的无序性和不规则性时仍有不足。作者创新性地提出将点云转换为体素化的3D二进制图像，然后使用拓扑数据分析(TDA)通过立方体持久性提取特征。该方法借鉴了拓扑数据分析在医学成像、生物医学等领域的应用，使用了立方体同调和持久性概念，并结合了多种图像过滤技术和轻量级一维卷积神经网络进行特征学习和分类。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用拓扑数据分析提取3D点云物体的拓扑特征，捕捉物体的拓扑结构(如连通组件、孔洞、空腔等)来区分不同物体，而不依赖于点云的具体几何排列。整体流程包括：1)将点云转换为体素化的3D二进制图像；2)应用57种不同的过滤技术生成3D灰度图像；3)对每个灰度图像应用立方体持久性分析提取36维拓扑特征；4)将所有特征向量连接形成最终特征向量；5)使用轻量级一维卷积神经网络处理特征向量并进行分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将拓扑数据分析通过立方体持久性应用于3D点云分类；创新的特征提取方法使用57种不同过滤技术；轻量级1D CNN架构；在多个数据集上实现最先进性能(ModelNet40上99.05%，ModelNet10上99.52%)；对损坏输入表现出强大鲁棒性。相比之前工作，TACO-Net不使用传统的几何特征或深度学习特征，而是使用拓扑特征表示物体；采用轻量级一维CNN而非复杂3D CNN或transformer架构；在多个数据集上实现了比现有方法更高的准确率，首次突破99%准确率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TACO-Net创新性地将拓扑数据分析与轻量级一维卷积神经网络相结合，通过提取点云的拓扑特征，在多个3D物体分类基准数据集上实现了最先进的性能，同时展现出强大的鲁棒性和泛化能力。'}


### 论文摘要

3D object classification is a crucial problem due to its significant practical relevance in many fields, including computer vision, robotics, and autonomous driving. Although deep learning methods applied to point clouds sampled on CAD models of the objects and/or captured by LiDAR or RGBD cameras have achieved remarkable success in recent years, achieving high classification accuracy remains a challenging problem due to the unordered point clouds and their irregularity and noise. To this end, we propose a novel state-of-the-art (SOTA) 3D object classification technique that combines topological data analysis with various image filtration techniques to classify objects when they are represented using point clouds. We transform every point cloud into a voxelized binary 3D image to extract distinguishing topological features. Next, we train a lightweight one-dimensional Convolutional Neural Network (1D CNN) using the extracted feature set from the training dataset. Our framework, TACO-Net, sets a new state-of-the-art by achieving $99.05\%$ and $99.52\%$ accuracy on the widely used synthetic benchmarks ModelNet40 and ModelNet10, and further demonstrates its robustness on the large-scale real-world OmniObject3D dataset. When tested with ten different kinds of corrupted ModelNet40 inputs, the proposed TACO-Net demonstrates strong resiliency overall.

---

## 73. CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations

**论文链接:** [http://arxiv.org/abs/2509.24661v1](http://arxiv.org/abs/2509.24661v1)

**作者:** Zhiyuan Wu, Rolandos Alexandros Potamias, Xuyang Zhang, Zhongqun Zhang, Jiankang Deng, Shan Luo

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出CEDex方法，用于跨形态灵巧抓取合成，通过连接人类抓取运动学与机器人运动学，解决了现有方法缺乏类人运动学理解或局限于类人结构的问题。

### 背景

跨形态灵巧抓取合成对实现多样化环境中的通用机器人操作至关重要，但需要大量可靠多样的抓取数据进行有效模型训练和鲁棒泛化。

### 目的

开发一种能够为不同形态的机器人手自适应生成和优化抓取的方法，克服现有方法在运动学理解和数据收集方面的局限性。

### 方法

CEDex方法包括三个主要步骤：1)使用在人类接触数据上预训练的条件变分自编码器生成类人接触表示；2)通过拓扑合并进行运动学人类接触对齐，将多个人类手部件合并为统一机器人组件；3)应用基于带物理感知约束的符号距离场的抓取优化。

### 主要发现

使用CEDex构建了迄今为止最大的跨形态抓取数据集，包含四种夹持器类型的50万个物体的2000万次抓取。实验表明CEDex优于现有最先进的方法，且数据集通过高质量的多样化抓取有利于跨形态抓取学习。

### 结论

CEDex成功桥接了人类抓取运动学与机器人运动学，为不同形态的机器人手提供了有效的灵巧抓取合成解决方案，显著提升了跨形态抓取学习的性能。

### 翻译

跨形态灵巧抓取合成指的是为具有不同形态的各种机器人手自适应生成和优化抓取。这种能力对于在多样化环境中实现通用机器人操作至关重要，需要大量可靠多样的抓取数据以进行有效的模型训练和鲁棒的泛化。然而，现有方法要么依赖于缺乏类人运动学理解的基于物理的优化，要么需要大量手动数据收集过程，且仅限于类人结构。在本文中，我们提出了CEDex，一种大规模跨形态灵巧抓取合成方法，通过将机器人运动学模型与生成的类人接触表示对齐来连接人类抓取运动学和机器人运动学。给定物体的点云和任意机器人手模型，CEDex首先使用在人类接触数据上预训练的条件变分自编码器生成类人接触表示。然后通过拓扑合并进行运动学人类接触对齐，将多个人类手部件合并为统一的机器人组件，接着进行基于带物理感知约束的符号距离场的抓取优化。使用CEDex，我们构建了迄今为止最大的跨形态抓取数据集，包含四种夹持器类型的50万个物体的2000万次抓取。大量实验表明CEDex优于最先进的方法，且我们的数据集通过高质量的多样化抓取有利于跨形态抓取学习。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决跨形态灵巧抓取生成问题，即如何为不同形态的机器人手自适应地生成和优化抓取动作。这个问题在现实中非常重要，因为现有的机器人抓取方法大多针对特定末端执行器设计，当面对新机器人手时需要昂贵的数据收集和重新训练，限制了机器人在多样化环境中的实际应用和可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人类具有卓越的抓取能力，能够适应不同约束并推广到不同手指配置。受此启发，作者注意到现有机器人抓取方法泛化能力有限，需要一种统一模型来处理各种机器人形态。作者借鉴了基于模型的抓取优化、接触表示建模、条件变分自编码器(CVAE)和符号距离函数(SDF)等现有技术，设计了两阶段方法：首先生成类人接触表示，然后通过运动学对齐将人类手部分合并为机器人组件，最后进行物理感知的抓取优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将机器人运动学模型与生成的类人接触表示对齐，桥接人类抓取运动学和机器人运动学，实现兼具类人自然性和物理稳定性的跨形态抓取。整体流程：1)输入物体点云和机器人手模型；2)使用预训练的CVAE生成类人接触表示(接触图和部分图)；3)通过拓扑合并将多个人类手部分整合为统一机器人组件；4)使用基于SDF的抓取优化，结合物理感知约束；5)输出针对给定机器人的物理稳定抓取配置。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)CEDex方法，结合类人运动学和物理约束；2)构建了最大跨形态抓取数据集(50万物体，2000万抓取)；3)运动学人类接触对齐技术，适应不同机器人形态；4)物理感知约束确保抓取稳定性。相比之前工作：不同于纯物理优化方法，它考虑了抓取动态过程；不同于基于人类演示的方法，它不需要手动数据收集和关节映射；相比现有数据集，它规模更大且同时考虑类人运动学和物理约束；在性能上，它实现了更高的成功率(88.7%)和更好的多样性(0.512弧度)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CEDex通过结合类人运动学理解和物理感知约束，实现了跨形态机器人手的高效灵巧抓取生成，并构建了迄今为止规模最大的跨形态抓取数据集，显著提升了机器人抓取的泛化能力和实际应用价值。'}


### 论文摘要

Cross-embodiment dexterous grasp synthesis refers to adaptively generating and optimizing grasps for various robotic hands with different morphologies. This capability is crucial for achieving versatile robotic manipulation in diverse environments and requires substantial amounts of reliable and diverse grasp data for effective model training and robust generalization. However, existing approaches either rely on physics-based optimization that lacks human-like kinematic understanding or require extensive manual data collection processes that are limited to anthropomorphic structures. In this paper, we propose CEDex, a novel cross-embodiment dexterous grasp synthesis method at scale that bridges human grasping kinematics and robot kinematics by aligning robot kinematic models with generated human-like contact representations. Given an object's point cloud and an arbitrary robotic hand model, CEDex first generates human-like contact representations using a Conditional Variational Auto-encoder pretrained on human contact data. It then performs kinematic human contact alignment through topological merging to consolidate multiple human hand parts into unified robot components, followed by a signed distance field-based grasp optimization with physics-aware constraints. Using CEDex, we construct the largest cross-embodiment grasp dataset to date, comprising 500K objects across four gripper types with 20M total grasps. Extensive experiments show that CEDex outperforms state-of-the-art approaches and our dataset benefits cross-embodiment grasp learning with high-quality diverse grasps.

---

## 74. Skeleton-based Robust Registration Framework for Corrupted 3D Point Clouds

**论文链接:** [http://arxiv.org/abs/2509.24273v1](http://arxiv.org/abs/2509.24273v1)

**作者:** Yongqiang Wang, Weigang Li, Wenping Liu, Zhiqiang Tian, Jinling Li

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一种基于骨架的鲁棒点云配准框架(SRRF)，通过引入抗损坏的骨架表示来提高配准的鲁棒性和准确性，解决了现实世界点云受传感器限制、环境噪声和预处理误差影响导致的配准挑战。

### 背景

点云配准是3D视觉应用中的基础技术，包括自动驾驶、机器人和医学成像等领域，精确的多点云对齐对于准确的环境重建至关重要。然而，现实世界点云常受传感器限制、环境噪声和预处理误差影响，导致密度失真、噪声污染和几何变形，使配准变得困难。

### 目的

解决现有点云配准方法依赖直接点匹配或表面特征提取而容易受到数据损坏影响的问题，提高在受污染环境下的配准鲁棒性和准确性。

### 方法

提出了一种基于骨架的鲁棒配准框架(SRRF)，将骨架结构整合到配准过程中，结合损坏点云配准和骨架配准获得的变换，并设计了分布距离损失函数来强制源骨架和目标骨架之间的一致性，同时考虑原始局部几何特征和骨架结构的全局稳定性。

### 主要发现

在各种损坏数据集上的实验评估表明，SRRF在密度失真、噪声污染和几何变形等各种损坏场景下，始终优于现有的最先进配准方法，证实了其在处理损坏点云方面的鲁棒性。

### 结论

SRRF是处理现实世界损坏点云的有效方法，为3D感知任务提供了有潜力的解决方案，能够实现鲁棒且准确的点云配准结果。

### 翻译

点云配准是3D视觉应用中的基础技术，包括自动驾驶、机器人和医学成像等领域，精确的多点云对齐对于准确的环境重建至关重要。然而，现实世界点云常受传感器限制、环境噪声和预处理误差影响，导致密度失真、噪声污染和几何变形，使配准变得困难。现有的配准方法依赖直接点匹配或表面特征提取，容易受到这些损坏的影响，导致配准精度降低。为解决这些挑战，该研究提出了一种基于骨架的鲁棒配准框架，引入了抗损坏的骨架表示来提高配准的鲁棒性和准确性。该框架将骨架结构整合到配准过程中，结合损坏点云配准和骨架配准获得的变换，以实现最佳配准。此外，还设计了分布距离损失函数来强制源骨架和目标骨架之间的一致性，显著提高了配准性能。该框架确保配准同时考虑原始局部几何特征和骨架结构的全局稳定性，从而实现鲁棒且准确的配准结果。在各种损坏数据集上的实验评估表明，SRRF在密度失真、噪声污染和几何变形等各种损坏场景下，始终优于现有的最先进配准方法。结果证实了SRRF在处理损坏点云方面的鲁棒性，使其成为现实世界3D感知任务的有潜力的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决 corrupted（损坏/污染）3D点云数据上的鲁棒点云配准问题。现实世界中，点云数据常受传感器限制、环境噪声和预处理错误影响，导致密度扭曲、噪声污染和几何变形，使配准变得困难。这一问题在自动驾驶、机器人和医学成像等领域至关重要，因为这些应用需要精确对齐多个点云以准确重建环境，而现有方法对这类损坏非常敏感，导致对齐精度降低。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现实点云数据常不完美，而现有配准方法对此不够鲁棒。他们观察到骨架作为紧凑稳定的几何表示，能有效捕捉点云核心结构特征，对噪声和扭曲不敏感。作者借鉴了现有工作：基于学习的点云配准方法（如ICP、DCP）、点云骨架表示（如Point2Skeleton）和深度学习模型架构（如PointNet++和Transformer）。通过整合这些技术，他们设计了专门针对损坏点云的骨架配准框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将骨架结构整合到配准过程中，结合从损坏点云对齐及其骨架对齐获得的变换，实现最优配准。整体流程包括：1) 骨架提取模块从点云中提取抗损坏的骨架结构；2) 结构特征提取和交互模块从点云和骨架中提取特征并增强交互；3) 几何匹配和变换估计模块建立点对应关系并估计变换；4) 变换集成模块结合点云级和骨架级的变换结果；5) 使用分布距离损失函数确保骨架一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 针对损坏点云设计的基于骨架的配准框架，利用骨架结构稳定性提高鲁棒性；2) 分布距离损失函数增强源骨架和目标骨架间的一致性；3) 构建了损坏点云的综合基准测试。相比之前工作，SRRF不直接依赖原始点分布，而是利用骨架表示增强抗损坏能力；解决了骨架不一致导致的配准误差问题；结合了点云级和骨架级配准，充分利用两种表示的优势。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于骨架的鲁棒配准框架，通过整合骨架结构和深度特征学习，显著提高了点云配准在密度变化、噪声污染和几何变形等损坏场景下的鲁棒性和准确性。'}


### 论文摘要

Point cloud registration is fundamental in 3D vision applications, including autonomous driving, robotics, and medical imaging, where precise alignment of multiple point clouds is essential for accurate environment reconstruction. However, real-world point clouds are often affected by sensor limitations, environmental noise, and preprocessing errors, making registration challenging due to density distortions, noise contamination, and geometric deformations. Existing registration methods rely on direct point matching or surface feature extraction, which are highly susceptible to these corruptions and lead to reduced alignment accuracy. To address these challenges, a skeleton-based robust registration framework is presented, which introduces a corruption-resilient skeletal representation to improve registration robustness and accuracy. The framework integrates skeletal structures into the registration process and combines the transformations obtained from both the corrupted point cloud alignment and its skeleton alignment to achieve optimal registration. In addition, a distribution distance loss function is designed to enforce the consistency between the source and target skeletons, which significantly improves the registration performance. This framework ensures that the alignment considers both the original local geometric features and the global stability of the skeleton structure, resulting in robust and accurate registration results. Experimental evaluations on diverse corrupted datasets demonstrate that SRRF consistently outperforms state-of-the-art registration methods across various corruption scenarios, including density distortions, noise contamination, and geometric deformations. The results confirm the robustness of SRRF in handling corrupted point clouds, making it a potential approach for 3D perception tasks in real-world scenarios.

---

## 75. Neural Visibility of Point Sets

**论文链接:** [http://arxiv.org/abs/2509.24150v1](http://arxiv.org/abs/2509.24150v1)

**作者:** Jun-Hao Wang, Yi-Yang Tian, Baoquan Chen, Peng-Shuai Wang

**发布时间:** 2025-09-29

**DOI:** 10.1145/3757377.3763869

**备注:** Accepted to SIGGRAPH Asia 2025

### GPT解析

### 总结

本文提出了一种基于深度学习的点云可见性确定方法，将可见性问题转化为二元分类任务，通过3D U-Net和多层感知器(MLP)组成的网络实现，显著提高了计算效率和准确性，并展示了在多种应用中的多功能性。

### 背景

点云是3D数据的常用表示，但从给定视点确定点的可见性具有挑战性，因为点云的稀疏性和缺乏显式连接性。传统的隐藏点移除(HPR)方法在计算效率、对噪声的鲁棒性以及处理凹区域或低密度点云方面存在局限性。

### 目的

开发一种新的点云可见性确定方法，解决传统方法的局限性，提高计算效率和准确性，并增强对噪声和不同密度点云的鲁棒性。

### 方法

将可见性确定问题表述为二元分类任务，网络核心包含一个3D U-Net用于提取视点无关的点特征，以及一个共享的多层感知器(MLP)，使用提取的特征和视点方向作为输入来预测点的可见性。网络通过从渲染的3D模型生成的真实可见性标签进行端到端训练。

### 主要发现

该方法在准确性和计算效率上都显著优于HPR，在大规模点云上实现了高达126倍的加速。网络对噪声和变化的点云密度具有鲁棒性，能够很好地泛化到未见过的形状。在ShapeNet、ABC数据集和真实世界数据集上的实验验证了该方法的有效性。

### 结论

该方法为点云可见性问题提供了一个高效、准确的解决方案，不仅解决了传统方法的局限性，还在点云可视化、表面重建、法线估计、阴影渲染和视点优化等多种应用中展示了其多功能性。

### 翻译

点云是3D数据的广泛应用表示，但从给定视点确定点的可见性仍然是一个具有挑战性的问题，因为它们的稀疏特性和缺乏显式连接性。传统方法，如隐藏点移除(HPR)，在计算效率、对噪声的鲁棒性以及处理凹区域或低密度点云方面面临局限性。在本文中，我们通过将点云可见性确定表述为二元分类任务，提出了一种新方法。我们网络的核心是一个3D U-Net，用于提取视点无关的点特征，以及一个共享的多层感知器(MLP)，使用提取的特征和视点方向作为输入来预测点的可见性。网络使用从渲染的3D模型生成的真实可见性标签进行端到端训练。我们的方法在准确性和计算效率上都显著优于HPR，在大规模点云上实现了高达126倍的加速。此外，我们的网络对噪声和变化的点云密度具有鲁棒性，并且能够很好地泛化到未见过的形状。我们通过在ShapeNet、ABC数据集和真实世界数据集上的大量实验验证了我们方法的有效性，在可见性准确性方面显示出显著改进。我们还展示了我们的方法在各种应用中的多功能性，包括点云可视化、表面重建、法线估计、阴影渲染和视点优化。我们的代码和模型可在https://github.com/octree-nn/neural-visibility获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云中点的可见性判断问题，即给定一个点云和一个视点，如何判断点云中的每个点从这个视点看是否可见。这个问题在现实中很重要，因为点云是3D数据在计算机图形学、计算机视觉和机器人领域最广泛使用的表示方法之一，而可见性判断是点云渲染、可视化和许多3D处理任务的基础。传统方法存在计算效率低、对噪声敏感、难以处理凹区域或低密度点云等局限性，限制了它们在实际应用中的适用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了点云可见性问题的挑战性，评估了现有方法（如表面重建后可见性测试、隐藏点去除HPR、投影到2D图像空间使用神经网络）的局限性。然后创新性地将可见性判断问题转化为二分类任务，使用神经网络直接预测点的可见性。方法设计上借鉴了U-Net架构进行特征提取，基于O-CNN的3D卷积框架，使用了视图方向编码方法和交叉熵损失函数，但将这些技术整合到一个全新的解决点云可见性问题的框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云可见性判断问题转化为二分类任务，使用深度神经网络直接学习从给定视点判断点云中每个点是否可见的能力，利用深度学习从数据中学习必要的先验信息。整体流程分为三步：1)特征提取：将点云转换为八叉树结构，使用基于八叉树的U-Net提取每个点的视图无关特征；2)可见性预测：将视图方向编码并与提取的特征结合，使用轻量级MLP预测每个点的可见性；3)训练与推理：使用合成数据训练网络，训练好的网络可高效预测新点云在给定视点下的可见性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的学习框架，将点云可见性判断表述为二分类任务；2)高效架构设计，使用基于八叉树的U-Net和轻量级MLP；3)视图方向编码方法；4)端到端训练能力。相比之前的工作，本文方法不需要先进行表面重建，计算效率显著提高（对大型点云速度提升可达126倍），对噪声和低密度点云更加鲁棒，能更好地处理凹区域，不需要参数调整，直接在3D空间操作不受图像分辨率限制，具有更好的泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于神经网络的点云可见性判断方法，通过将问题转化为二分类任务并使用深度学习直接预测点的可见性，显著提高了计算效率和准确性，同时增强了噪声鲁棒性和对不同点云密度的适应性。'}


### 论文摘要

Point clouds are widely used representations of 3D data, but determining the visibility of points from a given viewpoint remains a challenging problem due to their sparse nature and lack of explicit connectivity. Traditional methods, such as Hidden Point Removal (HPR), face limitations in computational efficiency, robustness to noise, and handling concave regions or low-density point clouds. In this paper, we propose a novel approach to visibility determination in point clouds by formulating it as a binary classification task. The core of our network consists of a 3D U-Net that extracts view-independent point-wise features and a shared multi-layer perceptron (MLP) that predicts point visibility using the extracted features and view direction as inputs. The network is trained end-to-end with ground-truth visibility labels generated from rendered 3D models. Our method significantly outperforms HPR in both accuracy and computational efficiency, achieving up to 126 times speedup on large point clouds. Additionally, our network demonstrates robustness to noise and varying point cloud densities and generalizes well to unseen shapes. We validate the effectiveness of our approach through extensive experiments on the ShapeNet, ABC Dataset and real-world datasets, showing substantial improvements in visibility accuracy. We also demonstrate the versatility of our method in various applications, including point cloud visualization, surface reconstruction, normal estimation, shadow rendering, and viewpoint optimization. Our code and models are available at https://github.com/octree-nn/neural-visibility.

---

## 76. Clebsch-Gordan Transformer: Fast and Global Equivariant Attention

**论文链接:** [http://arxiv.org/abs/2509.24093v1](http://arxiv.org/abs/2509.24093v1)

**作者:** Owen Lewis Howell, Linfeng Zhao, Xupeng Zhu, Yaoyao Qian, Haojie Huang, Lingfeng Sun, Wil Thomason, Robert Platt, Robin Walters

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为Clebsch-Gordan Transformer的新型架构，通过在SO(3)不可约表示上的Clebsch-Gordan卷积实现高效全局注意力，支持所有阶数的等变特征，同时保持O(N log N)的计算复杂度。

### 背景

全局注意力机制是Transformer架构成功的关键，但计算成本与token数量呈二次关系；等变模型虽能利用问题实例的底层几何结构实现更高精度，但需要额外计算资源；现有等变Transformer仅支持低阶特征和局部上下文窗口，限制了表达能力。

### 目的

开发一种支持所有阶数等变特征的高效全局注意力机制，突破现有等变Transformer在特征阶数和上下文窗口上的限制。

### 方法

提出Clebsch-Gordan Transformer，通过SO(3)不可约表示上的Clebsch-Gordan卷积实现高效全局注意力；利用Clebsch-Gordan矩阵的稀疏性处理高阶特征；通过权重共享或数据增强实现可选的token排列等变性。

### 主要发现

所提方法实现了所有阶数特征的等变建模，同时达到O(N log N)的输入token复杂度；在n体模拟、QM9、ModelNet点云分类和机器人抓取数据集等多个基准测试上表现优异。

### 结论

Clebsch-Gordan Transformer在GPU内存占用、计算速度和准确性方面均优于现有等变Transformer，为处理高阶等变特征提供了高效解决方案。

### 翻译

全局注意力机制是Transformer架构成功的关键之一，但它对token数量的计算成本是二次方的。另一方面，等变模型利用问题实例的底层几何结构，在物理、生化、计算机视觉和机器人任务中通常能实现更高的准确性，但需要额外的计算资源。因此，现有的等变Transformer仅支持低阶等变特征和局部上下文窗口，限制了它们的表达能力和性能。本文提出Clebsch-Gordan Transformer，通过在SO(3)不可约表示上的新颖Clebsch-Gordan卷积实现高效的全局注意力。我们的方法实现了所有阶数特征的等变建模，同时达到O(N log N)的输入token复杂度。此外，通过利用Clebsch-Gordan矩阵的稀疏性，所提出的方法能够很好地扩展到高阶不可约特征。最后，我们还通过权重共享或数据增强纳入了可选的token排列等变性。我们在n体模拟、QM9、ModelNet点云分类和机器人抓取数据集等多个基准测试上对我们的方法进行了基准测试，显示出在GPU内存大小、速度和准确性方面明显优于现有的等变Transformer。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决Transformer架构中全局注意力机制计算复杂度高（与token数量成二次方关系O(N²)）的问题，以及现有等变模型只能处理低阶等变特征和局部上下文窗口的限制。这个问题很重要，因为在物理、生物化学、计算机视觉和机器人等领域，等变模型能更好地处理几何结构问题，而全局注意力是Transformer成功的关键。现有方法限制了模型在大规模几何感知数据上的应用，影响了这些领域的学习效率和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到向量长卷积（vector long convolution）的启发，将其扩展为支持任意阶不可约表示的Clebsch-Gordan卷积。他们在图谱域应用注意力机制实现排列等变性，并利用Clebsch-Gordan矩阵的稀疏性提高计算效率。该方法借鉴了SE(3)-Hyena的核心思想，但扩展了其适用范围；也借鉴了向量长卷积的快速计算方法，以及Fourier变换技术和稀疏矩阵计算技术。作者通过结合这些现有工作的优点并进行创新，设计出了新的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用Clebsch-Gordan卷积在SO(3)不可约表示上实现高效的全局注意力，通过在图谱域应用注意力机制实现排列等变性，并利用Clebsch-Gordan矩阵的稀疏性降低计算复杂度。整体实现流程包括：1)将输入编码为查询、键和值；2)对查询和键进行快速傅里叶变换；3)应用Clebsch-Gordan卷积计算张量积；4)通过逆傅里叶变换将结果转换回空间域；5)将结果与值进行张量积计算；6)输出最终的注意力结果。同时结合了全局和局部注意力，以更好地处理不同尺度的信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次实现支持任意阶不可约表示的全局等变注意力，复杂度为O(N log N)；2)利用Clebsch-Gordan矩阵的稀疏性，实现O(L³)的谐波复杂度；3)在图谱域应用注意力机制，实现排列等变性；4)结合全局和局部注意力，更好地处理不同尺度的信息。相比之前的工作，该方法突破了现有等变Transformer只能处理低阶特征和局部上下文的限制，支持任意阶不可约表示的全局注意力，计算效率更高，内存使用更少，并在多个任务上表现更好。', '如果要用一句话总结这篇论文的贡献，我会怎么说？': 'Clebsch-Gordan Transformer通过创新的Clebsch-Gordan卷积技术，首次实现了支持任意阶不可约表示的高效全局等变注意力，在保持O(N log N)计算复杂度的同时，显著提升了在物理、化学、计算机视觉和机器人等几何感知任务中的性能。'}


### 论文摘要

The global attention mechanism is one of the keys to the success of transformer architecture, but it incurs quadratic computational costs in relation to the number of tokens. On the other hand, equivariant models, which leverage the underlying geometric structures of problem instance, often achieve superior accuracy in physical, biochemical, computer vision, and robotic tasks, at the cost of additional compute requirements. As a result, existing equivariant transformers only support low-order equivariant features and local context windows, limiting their expressiveness and performance. This work proposes Clebsch-Gordan Transformer, achieving efficient global attention by a novel Clebsch-Gordon Convolution on $\SO(3)$ irreducible representations. Our method enables equivariant modeling of features at all orders while achieving ${O}(N \log N)$ input token complexity. Additionally, the proposed method scales well with high-order irreducible features, by exploiting the sparsity of the Clebsch-Gordon matrix. Lastly, we also incorporate optional token permutation equivariance through either weight sharing or data augmentation. We benchmark our method on a diverse set of benchmarks including n-body simulation, QM9, ModelNet point cloud classification and a robotic grasping dataset, showing clear gains over existing equivariant transformers in GPU memory size, speed, and accuracy.

---

## 77. GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State

**论文链接:** [http://arxiv.org/abs/2509.23737v1](http://arxiv.org/abs/2509.23737v1)

**作者:** Guole Shen, Tianchen Deng, Yanbo Wang, Yongtao Chen, Yilin Shen, Jiuming Liu, Jingchuan Wang

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为GRS-SLAM3R的端到端SLAM框架，用于从RGB图像进行密集场景重建和姿态估计，无需场景或相机参数的先验知识。

### 背景

DUSt3R-based端到端场景重建在密集视觉SLAM中已显示出有希望的结果，但大多数现有方法仅使用图像对估计点图，忽略了空间记忆和全局一致性。

### 目的

解决现有DUSt3R框架只处理图像对并在局部坐标系中预测每对点图的问题，实现支持顺序输入并在全局坐标中增量估计度量尺度点云的SLAM框架。

### 方法

使用潜在状态进行空间记忆，设计基于transformer的门控更新模块来重置和更新空间记忆，将场景划分为子图，在每个子图内应用局部对齐，并使用相对约束将所有子图注册到共同的世界坐标系中。

### 主要发现

在各种数据集上的实验表明，该框架实现了优越的重建精度，同时保持了实时性能。

### 结论

GRS-SL3R框架通过引入空间记忆和全局一致性机制，改进了现有的DUSt3R-based方法，能够在保持实时性能的同时提供更精确的重建结果。

### 翻译

基于DUSt3R的端到端场景重建最近在密集视觉SLAM中显示出有希望的结果。然而，大多数现有方法仅使用图像对来估计点图，忽略了空间记忆和全局一致性。为此，我们引入了GRS-SLAM3R，一个端到端的SLAM框架，用于从RGB图像进行密集场景重建和姿态估计，无需对场景或相机参数有任何先验知识。与在所有图像对上操作并在局部坐标系中预测每对点图的现有DUSt3R框架不同，我们的方法支持顺序输入，并在全局坐标中增量估计度量尺度的点云。为了提高一致的空间相关性，我们使用潜在状态进行空间记忆，并设计了一个基于transformer的门控更新模块来重置和更新空间记忆，该模块持续聚合和跟踪跨帧的相关3D信息。此外，我们将场景划分为子图，在每个子图内应用局部对齐，并使用相对约束将所有子图注册到共同的世界坐标系中，产生全局一致的地图。在各种数据集上的实验表明，我们的框架在保持实时性能的同时实现了优越的重建精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于DUSt3R的密集SLAM方法只使用图像对估计点图而忽略空间记忆和全局一致性的问题。这个问题在现实中很重要，因为高质量的密集SLAM对于机器人导航、自动驾驶、增强现实等应用至关重要，能够实现实时、准确的3D场景重建和相机位姿估计，解决空间记忆和全局一致性问题可以大大提高系统在复杂环境中的鲁棒性和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有基于DUSt3R方法（如SLAM3R和MASt3R-SLAM）的局限性，即忽略了多帧空间相关性和空间记忆的重要性。为此，作者设计了增量式SLAM框架，引入潜在状态支持度量级点云估计，并创新性地设计了门控循环机制（包括更新门和重置门）来选择性整合信息。该方法借鉴了DUSt3R框架、CUT3R的循环状态机制、传统GRU结构和SLAM中的子图表示技术，但进行了创新性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用持久的潜在状态作为空间记忆，通过门控机制选择性控制信息流，并结合多子图表示与分层对齐策略。整体流程包括：1)输入RGB图像序列；2)使用视觉transformer编码器编码当前帧；3)通过门控循环模型（重置门、更新门和transformer解码器）更新潜在状态；4)生成度量级3D点云和相机位姿；5)根据关键帧变化构建子图；6)进行子图内局部对齐和子图间全局对齐；7)输出相机位姿和密集点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)GRS-SLAM3R框架，结合门控循环模型和分层子图对齐；2)具有transformer门控单元的潜在状态设计，用于选择性更新和重置空间记忆；3)多子图场景表示与分层对齐机制。相比之前工作，该方法支持序列化输入而非图像对处理，在全局坐标中估计点云而非局部坐标，通过门控机制选择性整合信息而非简单更新，并通过多子图策略有效减少长序列累积漂移。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了GRS-SLAM3R，一种基于门控循环状态和分层子图对齐的实时密集SLAM框架，通过选择性更新空间记忆和分层优化，实现了高质量的全局一致3D场景重建和相机位姿估计。'}


### 论文摘要

DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM. However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global consistency.To this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters. Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate. In order to improve consistent spatial correlation, we use a latent state for spatial memory and design a transformer-based gated update module to reset and update the spatial memory that continuously aggregates and tracks relevant 3D information across frames. Furthermore, we partition the scene into submaps, apply local alignment within each submap, and register all submaps into a common world frame using relative constraints, producing a globally consistent map. Experiments on various datasets show that our framework achieves superior reconstruction accuracy while maintaining real-time performance.

---

## 78. DiffPCN: Latent Diffusion Model Based on Multi-view Depth Images for Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2509.23723v1](http://arxiv.org/abs/2509.23723v1)

**作者:** Zijun Li, Hongyu Yan, Shijie Li, Kunming Luo, Li Lu, Xulei Yang, Weisi Lin

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出DiffPCN，一种基于扩散的从粗到细的点云补全框架，通过深度图像投影、点云去噪和关联感知上采样三个步骤，实现了高质量点云补全。

### 背景

潜在扩散模型(LDMs)在各种低级视觉任务中表现出强大的生成能力，但由于点云的无结构和不规则特性，其在点云补全方面的潜力尚未得到充分探索。

### 目的

提出DiffPCN框架，解决点云补全问题，提高点云生成的质量和完整性。

### 方法

DiffPCN包含两个阶段：1)初始阶段：将无序和不规则的局部点云投影为结构化深度图像，使用DepthLDM合成完成的多视图深度图像形成粗略点云；2)精细阶段：设计点云去噪网络去除粗略点云中的伪影，并通过关联感知点云上采样器利用局部关联特征指导上采样过程。

### 主要发现

实验结果表明，DiffPCN在几何准确性和形状完整性方面达到了最先进的性能，显著提高了点云补全的鲁棒性和一致性。

### 结论

DiffPCN通过结合扩散模型和点云处理技术，能够有效解决点云补全问题，生成高质量、高完整性的点云。

### 翻译

潜在扩散模型(LDMs)在各种低级视觉任务中表现出强大的生成能力。然而，由于点云的无结构和不规则特性，其在点云补全方面的潜力尚未得到充分探索。在这项工作中，我们提出了DiffPCN，一种新颖的基于扩散的从粗到细的点云补全框架。我们的方法包含两个阶段：一个用于生成粗略点云的初始阶段，以及一个通过点云去噪和上采样提高其质量的精细阶段。具体来说，我们首先将无序和不规则的局部点云投影为结构化的深度图像，这些图像作为精心设计的DepthLDM的条件，用于合成完成的多视图深度图像，这些图像用于形成粗略点云。通过这种方式，我们的DiffPCN能够利用LDM的强大生成和理解能力，产生高质量和高完整性的粗略点云。然后，由于LDM不可避免地在生成的深度图中引入异常值，我们设计了一个点云去噪网络，通过预测逐点距离分数从粗略点云中去除伪影。最后，我们设计了一个关联感知点云上采样器，它利用输入点云与相应粗略点之间的局部关联特征来指导上采样过程，进一步产生密集且高保真度的输出。实验结果表明，我们的DiffPCN在几何准确性和形状完整性方面达到了最先进的性能，显著提高了点云补全的鲁棒性和一致性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全问题。点云是由LiDAR和深度相机等传感器捕获的3D空间信息，但在现实世界中常因遮挡、传感器噪声和有限分辨率而变得稀疏和不完整。这种不完整性严重影响下游3D任务，如物体识别、场景理解和自动驾驶等。因此，点云补全技术对于提高3D感知和推理能力具有重要意义，可广泛应用于自动驾驶、机器人导航和增强现实等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点云补全方法的局限性：传统基于点的方法难以捕捉复杂几何形状，而基于扩散模型的方法直接在3D空间处理面临效率问题。作者借鉴了多个现有工作：受MVDD启发使用DDPM生成深度图像；参考Wonder3D的跨视角注意力机制；利用Point-Bert作为点编码器；采用类似PCN的coarse-to-fine重建流程。创新在于将3D点云转换为结构化2D深度图像，利用LDM生成能力，并结合专门设计的去噪和上精炼网络提高质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D无序点云转换为结构化多视角深度图像，利用潜在扩散模型生成完整深度图像，再投影回3D空间形成初始点云，然后通过去噪和上精炼得到高质量点云。整体流程分两阶段：1)初始阶段：将点云投影到多视角深度图像，使用VAE编码，通过DepthLDM结合跨视角和点对齐注意力生成一致深度图像，再投影回3D形成粗糙点云；2)精炼阶段：使用PDNet去除异常点，通过APU利用部分点和粗糙点间的关联关系学习局部变换特征，传播到整个点云生成最终密集点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)DepthLDM：将3D点云转为多视角深度图像，利用LDM生成能力；2)跨视角和点对齐注意力机制：确保多视角一致性和3D结构准确性；3)点去噪网络(PDNet)：去除投影回3D时引入的异常点；4)关联感知点上采样器(APU)：利用点间关联关系生成高质量点云。相比之前工作，主要不同在于不直接在3D空间处理点云，而是转换为结构化2D表示；结合多视角一致性和3D结构感知；专门设计去噪和上精炼网络；在几何准确性和形状完整性方面实现最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了DiffPCN，一种基于多视角深度图像的潜在扩散模型框架，通过将3D点云转换为结构化2D表示并利用扩散模型生成能力，结合专门设计的去噪和上精炼网络，实现了点云补全任务的最先进性能。'}


### 论文摘要

Latent diffusion models (LDMs) have demonstrated remarkable generative capabilities across various low-level vision tasks. However, their potential for point cloud completion remains underexplored due to the unstructured and irregular nature of point clouds. In this work, we propose DiffPCN, a novel diffusion-based coarse-to-fine framework for point cloud completion. Our approach comprises two stages: an initial stage for generating coarse point clouds, and a refinement stage that improves their quality through point denoising and upsampling. Specifically, we first project the unordered and irregular partial point cloud into structured depth images, which serve as conditions for a well-designed DepthLDM to synthesize completed multi-view depth images that are used to form coarse point clouds. In this way, our DiffPCN can yield high-quality and high-completeness coarse point clouds by leveraging LDM' s powerful generation and comprehension capabilities. Then, since LDMs inevitably introduce outliers into the generated depth maps, we design a Point Denoising Network to remove artifacts from the coarse point cloud by predicting a per-point distance score. Finally, we devise an Association-Aware Point Upsampler, which guides the upsampling process by leveraging local association features between the input point cloud and the corresponding coarse points, further yielding a dense and high-fidelity output. Experimental results demonstrate that our DiffPCN achieves state-of-the-art performance in geometric accuracy and shape completeness, significantly improving the robustness and consistency of point cloud completion.

---

## 79. StrucADT: Generating Structure-controlled 3D Point Clouds with Adjacency Diffusion Transformer

**论文链接:** [http://arxiv.org/abs/2509.23709v1](http://arxiv.org/abs/2509.23709v1)

**作者:** Zhenyu Shu, Jiajun Shen, Zhongui Chen, Xiaoguang Han, Shiqing Xin

**发布时间:** 2025-09-28

**DOI:** 10.1109/TVCG.2025.3600392

### GPT解析

### 总结

本文提出了一种名为StrucADT的新型结构可控点云生成模型，通过形状结构控制点云生成，解决了现有方法难以生成满足用户特定需求的问题。

### 背景

在3D点云生成领域，虽然许多模型能生成多样逼真的3D形状，但大多数方法难以生成满足用户特定需求的可控3D点云形状，阻碍了3D点云生成的大规模应用。

### 目的

解决3D点云生成中缺乏控制的问题，首次提出通过形状结构（包括部件存在性和部件相邻关系）来控制点云的生成。

### 方法

手动标注点云形状分割部件之间的相邻关系，构建StructureGraph表示；基于此表示引入StrucADT模型，包含StructureGraphNet模块提取结构感知潜在特征，cCNF Prior模块学习部件相邻控制的潜在特征分布，以及Diffusion Transformer模块生成结构一致的点云形状。

### 主要发现

结构可控的3D点云生成方法能够产生高质量且多样的点云形状，能够基于用户指定的形状结构生成可控的点云。

### 结论

在ShapeNet数据集上，该方法在可控点云生成方面取得了最先进的性能。

### 翻译

在3D点云生成领域，许多3D生成模型已经展示了生成多样且逼真的3D形状的能力。然而，大多数方法难以生成满足用户特定需求的可控3D点云形状，阻碍了3D点云生成的大规模应用。为解决3D点云生成中缺乏控制的问题，我们首次提出通过包含部件存在性和部件相邻关系的形状结构来控制点云的生成。我们手动标注了点云形状分割部件之间的相邻关系，从而构建了StructureGraph表示。基于这种StructureGraph表示，我们引入了StrucADT，一种新型的结构可控点云生成模型，它包含StructureGraphNet模块来提取结构感知的潜在特征，cCNF Prior模块来学习由部件相邻控制的潜在特征分布，以及基于潜在特征和部件相邻条件的Diffusion Transformer模块来生成结构一致的点云形状。实验结果表明，我们的结构可控3D点云生成方法能够产生高质量且多样的点云形状，能够基于用户指定的形状结构生成可控点云，并在ShapeNet数据集的可控点云生成方面取得了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云生成缺乏控制能力的问题。现有的3D点云生成模型虽然能生成多样且真实的3D形状，但难以生成符合用户特定需求的可控形状，这阻碍了3D点云生成技术在建模、动画和游戏等领域的广泛应用。在实际应用中，用户通常需要生成符合自己需求的特定形状，而不仅仅是随机生成一些形状，缺乏控制能力限制了3D生成技术的实用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到同一类别的3D形状虽然几何轮廓可能差异很大，但结构保持相似。核心洞察是利用3D点云形状的结构（部件存在性和部件相邻关系）来控制生成过程。作者手动标注了ShapeNet数据集上分割部件之间的相邻关系，构建了StructureGraph表示。方法借鉴了扩散模型用于点云生成、图注意力网络处理部件关系、连续归一化流学习潜在特征分布以及Transformer架构作为去噪网络等现有技术，但将其创新性地组合应用于结构控制点云生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过形状结构（部件存在性和相邻关系）控制3D点云生成，将3D形状表示为StructureGraph，并利用三个模块协同工作：1) StructureGraphNet模块提取结构感知的潜在特征；2) cCNF Prior模块学习由部件相邻关系控制的潜在特征分布；3) Diffusion Transformer模块基于潜在特征和部件相邻关系生成结构一致的点云。整体流程为：输入点云和结构信息→StructureGraphNet提取特征→cCNF Prior学习特征分布→Diffusion Transformer生成结构一致点云→输出最终结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次提出通过形状结构控制3D点云生成；2) 构建更精确的StructureGraph表示，手动标注部件相邻关系；3) 设计三个专门模块实现结构控制生成；4) 提出结构一致性准确度(SCA)评估指标。相比之前工作，不同之处在于：StrucADT直接通过形状结构控制生成，而非文本或图像；构建的StructureGraph比现有数据集更精确地表示部件连接；生成过程明确使用StructureGraph作为输入，并通过专门设计的模块融合结构信息，生成结构一致的点云形状。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'StrucADT首次通过形状结构控制3D点云生成，构建了StructureGraph表示，并设计了三个关键模块实现了高质量、结构可控的3D点云生成，在ShapeNet数据集上取得了最先进的性能。'}


### 论文摘要

In the field of 3D point cloud generation, numerous 3D generative models have demonstrated the ability to generate diverse and realistic 3D shapes. However, the majority of these approaches struggle to generate controllable 3D point cloud shapes that meet user-specific requirements, hindering the large-scale application of 3D point cloud generation. To address the challenge of lacking control in 3D point cloud generation, we are the first to propose controlling the generation of point clouds by shape structures that comprise part existences and part adjacency relationships. We manually annotate the adjacency relationships between the segmented parts of point cloud shapes, thereby constructing a StructureGraph representation. Based on this StructureGraph representation, we introduce StrucADT, a novel structure-controllable point cloud generation model, which consists of StructureGraphNet module to extract structure-aware latent features, cCNF Prior module to learn the distribution of the latent features controlled by the part adjacency, and Diffusion Transformer module conditioned on the latent features and part adjacency to generate structure-consistent point cloud shapes. Experimental results demonstrate that our structure-controllable 3D point cloud generation method produces high-quality and diverse point cloud shapes, enabling the generation of controllable point clouds based on user-specified shape structures and achieving state-of-the-art performance in controllable point cloud generation on the ShapeNet dataset.

---

## 80. DFG-PCN: Point Cloud Completion with Degree-Flexible Point Graph

**论文链接:** [http://arxiv.org/abs/2509.23703v1](http://arxiv.org/abs/2509.23703v1)

**作者:** Zhenyu Shu, Jian Yao, Shiqing Xin

**发布时间:** 2025-09-28

**DOI:** 10.1109/TVCG.2025.3612379

### GPT解析

### 总结

本文提出了一种名为DFG-PCN的点云补全框架，通过自适应分配节点度数和几何感知图集成模块，有效解决了传统方法在处理几何复杂度不均匀分布时的局限性，实验证明该方法优于最先进的方法。

### 背景

点云补全是一项重要任务，专注于重建完整的点云并解决由遮挡和有限传感器分辨率引起的不完整问题。传统方法依赖于固定的局部区域划分，如k近邻算法，无法考虑形状不同区域几何复杂度的极不均匀分布。

### 目的

解决传统点云补全方法在处理几何复杂度不均匀分布时的局限性，特别是在具有精细细节或结构不连续的区域，提高点云补全的效率和重建质量。

### 方法

提出DFG-PCN框架，包括：1)使用结合特征变化和曲率的细节感知指标自适应分配节点度数；2)引入几何感知图集成模块，使用曼哈顿距离进行边聚合；3)融合局部和全局特征以增强表示能力；4)专注于结构重要区域的处理。

### 主要发现

在多个基准数据集上的大量实验表明，DFG-PCN方法持续优于最先进的方法，特别是在处理具有精细细节或结构不连续的区域时表现更佳。

### 结论

DFG-PCN框架通过自适应地处理不同区域的几何复杂度，能够更有效地表示和重建点云，提高了点云补全的整体性能，特别是在具有精细细节或结构不连续的区域。

### 翻译

点云补全是一项重要任务，专注于重建完整的点云并解决由遮挡和有限传感器分辨率引起的不完整问题。传统方法依赖于固定的局部区域划分，如k近邻算法，这些方法无法考虑形状不同区域几何复杂度的极不均匀分布。这种局限性导致表示效率低下和重建不理想，特别是在具有精细细节或结构不连续的区域。本文提出了一种名为Degree-Flexible Point Graph Completion Network (DFG-PCN)的点云补全框架。它使用结合特征变化和曲率的细节感知指标自适应地分配节点度数，专注于结构重要区域。我们进一步引入了一个几何感知图集成模块，该模块使用曼哈顿距离进行边聚合，并融合局部和全局特征以增强表示能力。在多个基准数据集上的大量实验证明，我们的方法持续优于最先进的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全问题，特别是传统方法无法处理不同区域几何复杂度不均匀分布的问题。这个问题在现实中非常重要，因为点云是3D物体的常见表示形式，但实际获取的点云常因传感器限制、遮挡等因素而稀疏不完整。恢复完整点云对于3D场景理解、自动驾驶、机器人导航等应用至关重要，能帮助保留观察细节、推断缺失部分并加密集疏表面。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者分析了传统k近邻方法的局限性，指出点云补全任务存在明显的不平衡重建需求：只有少数高频点需要复杂重建，而大多数点位于平坦区域。作者借鉴了PointNet和PointNet++在点云深度学习中的成功，以及编码器-解码器结构的点云补全方法。在此基础上，作者提出'灵活度点图'模型，通过结合特征变化和曲率设计细节感知的度分配策略，并开发几何感知的图集成模块，同时利用局部和全局信息来提高重建质量。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是点云补全中不同区域的重建需求不平衡，需要自适应分配计算资源；通过'灵活度点图'模型为不同区域的点分配不同数量的连接，使结构重要区域获得更多关注。整体流程分为三部分：1)特征提取器：使用层次聚合和PointNet提取点云特征；2)种子生成器：生成低分辨率完整点云作为种子；3)点生成模块：通过三个DFG步骤逐步提高分辨率，每步包括PointNet特征提取、图构建、图聚合、图融合、MLP和反卷积，最终生成完整点云。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)灵活度点图模型(DFG)允许自适应调整节点度；2)细节感知的度分配策略结合特征变化和曲率为结构重要区域分配更多连接；3)几何感知的图集成模块使用曼哈顿距离进行边聚合和细节引导的特征融合。相比之前工作，DFG-PCN能处理不平衡重建需求，动态调整连接数量而非使用固定k近邻；结合曼哈顿距离更好地捕获方向结构和几何不连续性；同时考虑局部和全局信息而非单一尺度特征，显著提高了细节保留和结构一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于灵活度点图(DFG)的点云补全网络(DFG-PCN)，通过自适应调整节点度分配和几何感知的特征融合，显著提高了点云补全任务中细节保留和结构一致性，特别是在复杂几何区域的表现。'}


### 论文摘要

Point cloud completion is a vital task focused on reconstructing complete point clouds and addressing the incompleteness caused by occlusion and limited sensor resolution. Traditional methods relying on fixed local region partitioning, such as k-nearest neighbors, which fail to account for the highly uneven distribution of geometric complexity across different regions of a shape. This limitation leads to inefficient representation and suboptimal reconstruction, especially in areas with fine-grained details or structural discontinuities. This paper proposes a point cloud completion framework called Degree-Flexible Point Graph Completion Network (DFG-PCN). It adaptively assigns node degrees using a detail-aware metric that combines feature variation and curvature, focusing on structurally important regions. We further introduce a geometry-aware graph integration module that uses Manhattan distance for edge aggregation and detail-guided fusion of local and global features to enhance representation. Extensive experiments on multiple benchmark datasets demonstrate that our method consistently outperforms state-of-the-art approaches.

---

## 81. ZeroScene: A Zero-Shot Framework for 3D Scene Generation from a Single Image and Controllable Texture Editing

**论文链接:** [http://arxiv.org/abs/2509.23607v1](http://arxiv.org/abs/2509.23607v1)

**作者:** Xiang Tang, Ruotong Li, Xiaopeng Fan

**发布时间:** 2025-09-28

**备注:** 16 pages, 15 figures, Project page:  https://xdlbw.github.io/ZeroScene/

### GPT解析

### 总结

ZeroScene是一个创新的系统，利用大型视觉模型的先验知识，实现了单图像到3D场景的零样本重建和纹理编辑，解决了现有方法在资产质量和场景连贯性方面的局限性。

### 背景

在3D内容生成领域，单图像场景重建方法难以同时确保单个资产的质量和复杂环境中整个场景的连贯性，而纹理编辑技术通常无法同时保持局部连续性和多视角一致性。

### 目的

提出一个名为ZeroScene的新系统，利用大型视觉模型的先验知识，以零样本方式完成单图像到3D场景重建和纹理编辑。

### 方法

ZeroScene从输入图像中提取对象级别的2D分割和深度信息推断场景空间关系，联合优化点云的3D和2D投影损失更新对象姿态实现精确场景对齐，构建包含前景和背景的完整3D场景；通过约束扩散模型和引入掩码引导的渐进式图像生成策略保持多视角纹理一致性，并使用基于物理的渲染材料估计提高真实感。

### 主要发现

实验结果表明，该框架确保了生成资产的几何和外观准确性，忠实地重建了场景布局，生成了与文本提示高度一致的详细纹理。

### 结论

ZeroScene有效解决了单图像场景重建和纹理编辑中的挑战，实现了高质量的3D内容生成。

### 翻译

在3D内容生成领域，单图像场景重建方法仍然难以在复杂环境中同时确保单个资产的质量和整个场景的连贯性，而纹理编辑技术通常无法同时保持局部连续性和多视角一致性。在本文中，我们提出了一个新颖的系统ZeroScene，它利用大型视觉模型的先验知识，以零样本方式完成单图像到3D场景重建和纹理编辑。ZeroScene从输入图像中提取对象级别的2D分割和深度信息，以推断场景内的空间关系。然后它联合优化点云的3D和2D投影损失，更新对象姿态以实现精确的场景对齐，最终构建包含前景和背景的连贯完整的3D场景。此外，ZeroScene支持场景中对象的纹理编辑。通过对扩散模型施加约束并引入掩码引导的渐进式图像生成策略，我们有效保持了多视角间的纹理一致性，并通过基于物理的渲染(PBR)材料估计进一步提高了渲染结果的真实感。实验结果表明，我们的框架不仅确保了生成资产的几何和外观准确性，还忠实地重建了场景布局，并产生了与文本提示高度一致的详细纹理。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单张图像生成3D场景时难以同时保证单个对象质量和整体场景连贯性的问题，以及纹理编辑中难以保持局部连续性和多视角一致性的问题。这个问题在数字孪生、虚拟现实、游戏开发和机器人仿真等领域非常重要，因为它能大大降低3D内容创建的成本，提高生成质量，使3D场景重建和编辑更加实用和高效。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法在处理复杂场景时存在质量下降和空间关系不准确的问题，因此设计将前景和背景分开处理的策略。他们借鉴了图像分割、图像修复、3D重建等成熟视觉模型，使用了DUSt3R模型估计深度信息，参考了扩散模型在纹理生成中的应用，并借鉴了ControlNet在几何条件引导图像生成方面的技术。通过整合这些现有技术并加以创新，设计出了ZeroScene框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将前景和背景分开处理，利用大型视觉模型的先验知识实现零样本学习，通过联合优化3D和2D投影损失实现精确场景对齐，并采用特殊策略确保纹理编辑的多视角一致性。整体流程包括：1)前景对象生成和组合，包括实例分割、图像修复、3D模型生成和布局优化；2)背景处理，包括前景移除、点云估计、几何模型生成和布局优化；3)纹理编辑，包括几何条件渲染、扩散模型生成、掩码引导的多视角图像生成、反投影纹理合成和PBR材质估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)前景和背景分离处理策略；2)联合优化3D和2D投影损失的布局优化方法；3)几何感知的纹理编辑技术；4)掩码引导的渐进式图像生成策略确保多视角一致性；5)PBR材质估计增强渲染真实感。相比之前工作，不同之处在于：大多数现有方法只关注前景对象而忽略背景；传统布局优化通常只使用单一空间约束；纹理编辑方法难以保持多视角一致性；ZeroScene是零样本框架且生成显式三角形网格，便于下游应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ZeroScene提出了一种零样本框架，能够从单张图像生成高质量3D场景并支持可控纹理编辑，通过分离处理前景和背景、联合优化空间布局以及采用特殊的纹理生成策略，显著提高了场景重建的质量和一致性。'}


### 论文摘要

In the field of 3D content generation, single image scene reconstruction methods still struggle to simultaneously ensure the quality of individual assets and the coherence of the overall scene in complex environments, while texture editing techniques often fail to maintain both local continuity and multi-view consistency. In this paper, we propose a novel system ZeroScene, which leverages the prior knowledge of large vision models to accomplish both single image-to-3D scene reconstruction and texture editing in a zero-shot manner. ZeroScene extracts object-level 2D segmentation and depth information from input images to infer spatial relationships within the scene. It then jointly optimizes 3D and 2D projection losses of the point cloud to update object poses for precise scene alignment, ultimately constructing a coherent and complete 3D scene that encompasses both foreground and background. Moreover, ZeroScene supports texture editing of objects in the scene. By imposing constraints on the diffusion model and introducing a mask-guided progressive image generation strategy, we effectively maintain texture consistency across multiple viewpoints and further enhance the realism of rendered results through Physically Based Rendering (PBR) material estimation. Experimental results demonstrate that our framework not only ensures the geometric and appearance accuracy of generated assets, but also faithfully reconstructs scene layouts and produces highly detailed textures that closely align with text prompts.

---

## 82. UniPose: Unified Cross-modality Pose Prior Propagation towards RGB-D data for Weakly Supervised 3D Human Pose Estimation

**论文链接:** [http://arxiv.org/abs/2509.23376v1](http://arxiv.org/abs/2509.23376v1)

**作者:** Jinghong Zheng, Changlong Jiang, Jiaqi Li, Haohong Kuang, Hang Xu, Tingbing Yan

**发布时间:** 2025-09-27

**备注:** Accept at PRCV 2025

### GPT解析

### 总结

本文提出了一种名为UniPose的统一多模态姿态先验传播方法，用于利用未标注的单视角RGB-D序列进行弱监督3D人体姿态估计，无需劳动密集型的3D关键点标注。

### 背景

传统3D人体姿态估计方法需要大量3D关键点标注或依赖多视角相机校准及合成到真实数据的转换，存在效率低和泛化性差的问题。

### 目的

开发一种能够将2D HPE标注从大规模RGB数据集转换到3D领域的方法，消除对3D关键点标注的需求，同时避免多视角校准和合成数据转换的问题。

### 方法

UniPose通过在RGB-D序列上进行自监督学习实现2D到3D的转换；利用现成2D姿态估计作为点云网络的弱监督；融入时空约束；使用2D到3D反向投影损失和跨模态交互；将点云网络结果作为伪真实标签，通过锚点到关节预测方法进行3D提升。

### 主要发现

UniPose在CMU Panoptic和ITOP数据集上实现了与全监督方法相当的性能；结合大规模未标注数据可提高在挑战条件下的性能；提出的3D提升方法达到了最先进的结果。

### 结论

UniPose成功弥合了2D和3D领域之间的差距，展示了实际应用的潜力，且不受多视角相机校准或合成到真实数据转换相关问题的困扰。

### 翻译

本文提出UniPose，一种统一的多模态姿态先验传播方法，用于利用未标注的单视角RGB-D序列进行弱监督3D人体姿态估计。UniPose通过在容易获取的RGB-D序列上进行自监督学习，将大规模RGB数据集(如MSCOCO)中的2D HPE标注转换到3D领域，消除了对劳动密集型3D关键点标注的需求。这种方法弥合了2D和3D领域之间的差距，而不受多视角相机校准或合成到真实数据转换相关问题的困扰。在训练过程中，UniPose利用现成的2D姿态估计作为点云网络的弱监督，融入身体对称性和关节运动等时空约束。2D到3D的反向投影损失和跨模态交互进一步增强了这一过程。通过将点云网络的3D HPE结果作为伪真实标签，我们的锚点到关节预测方法对RGB和深度网络进行3D提升，使其比最先进方法更能抵抗2D HPE结果的不准确性。在CMU Panoptic和ITOP数据集上的实验表明，UniPose实现了与全监督方法相当的性能。结合大规模未标注数据(如NTU RGB+D 60)可在具有挑战性的条件下提高其性能，展示了实际应用的潜力。我们提出的3D提升方法也达到了最先进的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的是如何在没有大量3D人体姿态标注数据的情况下进行弱监督的3D人体姿态估计问题。这个问题很重要，因为3D人体姿态估计在动作识别、人机交互、虚拟现实等领域有广泛应用，而现有的3D标注数据收集和标注成本非常高，耗时耗力。如果能减少对大量3D标注数据的依赖，就能大大降低应用门槛，使这项技术更容易推广到实际应用中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考如何利用容易获取的无标注RGB-D数据序列（RGB图像、深度图和点云数据），同时避免多视角校准问题和合成到真实的域偏移问题。他们借鉴了现成的2D姿态估计模型作为弱监督信号，参考了点云处理网络（如PointNet）处理3D数据，借鉴了anchor-to-joint机制用于3D姿态提升，并利用了自监督学习中的时空约束。作者的创新在于将这些技术有机结合，设计了一个两阶段的统一框架，同时处理RGB、深度和点云三种模态的数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D姿态先验作为弱监督，结合人体物理约束作为自监督，通过多模态特征增强实现高质量3D姿态估计。整体流程分为两个阶段：第一阶段对点云进行弱监督3D姿态估计，通过多模态特征增强、2D到3D反向投影的弱监督信号和人体物理约束的自监督信号进行训练；第二阶段利用点云网络的3D姿态估计结果作为伪标签，对RGB图像和深度图进行基于anchor-to-joint机制的3D姿态提升，通过自适应采样3D锚点并建立锚点与特征的交互来估计关节的3D位置。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一跨模态姿态先验传播，首次针对RGB、深度和点云三种模态提出统一方法；2)新的弱监督信号构建，采用2D到3D反向投影而非传统的3D到2D投影；3)多模态特征增强，通过自适应投影器和动态对齐策略有效融合多模态特征；4)基于anchor-to-joint的3D姿态提升方法，提高对2D姿态估计误差的鲁棒性。相比之前工作，UniPose不依赖多视角或合成模型先验，统一处理多种模态，提供更一致的监督信号，且对2D姿态估计误差更具鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniPose提出了一种利用无标注RGB-D数据的统一跨模态姿态先验传播方法，实现了高质量的弱监督3D人体姿态估计，避免了传统方法的局限性，并在多个数据集上达到了与全监督方法相当的性能。'}


### 论文摘要

In this paper, we present UniPose, a unified cross-modality pose prior propagation method for weakly supervised 3D human pose estimation (HPE) using unannotated single-view RGB-D sequences (RGB, depth, and point cloud data). UniPose transfers 2D HPE annotations from large-scale RGB datasets (e.g., MS COCO) to the 3D domain via self-supervised learning on easily acquired RGB-D sequences, eliminating the need for labor-intensive 3D keypoint annotations. This approach bridges the gap between 2D and 3D domains without suffering from issues related to multi-view camera calibration or synthetic-to-real data shifts. During training, UniPose leverages off-the-shelf 2D pose estimations as weak supervision for point cloud networks, incorporating spatial-temporal constraints like body symmetry and joint motion. The 2D-to-3D back-projection loss and cross-modality interaction further enhance this process. By treating the point cloud network's 3D HPE results as pseudo ground truth, our anchor-to-joint prediction method performs 3D lifting on RGB and depth networks, making it more robust against inaccuracies in 2D HPE results compared to state-of-the-art methods. Experiments on CMU Panoptic and ITOP datasets show that UniPose achieves comparable performance to fully supervised methods. Incorporating large-scale unlabeled data (e.g., NTU RGB+D 60) enhances its performance under challenging conditions, demonstrating its potential for practical applications. Our proposed 3D lifting method also achieves state-of-the-art results.

---

## 83. CasPoinTr: Point Cloud Completion with Cascaded Networks and Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2509.23375v1](http://arxiv.org/abs/2509.23375v1)

**作者:** Yifan Yang, Yuxiang Yan, Boda Liu, Jian Pu

**发布时间:** 2025-09-27

**备注:** Accepted to IROS2025

### GPT解析

### 总结

本文介绍了CasPoinTr，一种使用级联网络和知识蒸馏的新型点云补全框架，通过分解任务为形状重建和融合补全两个阶段，有效解决了从高度不完整点云中预测整体形状和重建缺失区域的困难。

### 背景

现实环境收集的点云常因传感器分辨率有限、单一视点、遮挡和噪声等因素而不完整，这些挑战使点云补全成为各种应用中的关键技术任务。

### 目的

解决点云补全任务中的核心挑战，即从高度不完整的点云中预测整体形状和重建缺失区域，开发能够有效捕捉全局形状上下文同时细化局部细节的补全方法。

### 方法

CasPoinTr采用级联网络和知识蒸馏技术，将补全任务分解为形状重建阶段（生成辅助信息）和融合补全阶段（利用辅助信息和知识蒸馏生成最终输出），通过教师模型向学生模型传递不完整-完整关联知识。

### 主要发现

在ShapeNet-55数据集上的实验表明，CasPoinTr在不同难度设置下均优于现有方法，特别是在形状恢复和细节保留方面表现突出。

### 结论

级联网络和知识蒸馏的结合增强了模型捕捉全局形状上下文同时细化局部细节的能力，有效弥合了不完整输入和完整目标之间的差距。

### 翻译

从现实世界环境收集的点云通常由于传感器分辨率有限、单一视点、遮挡和噪声等因素而不完整。这些挑战使得点云补全对于各种应用至关重要。该任务的一个关键困难是从高度不完整的点云中预测整体形状和重建缺失区域。为了解决这个问题，我们引入了CasPoinTr，一种使用级联网络和知识蒸馏的新型点云补全框架。CasPoinTr将补全任务分解为两个协同阶段：形状重建，生成辅助信息；以及融合补全，利用这些信息和知识蒸馏生成最终输出。通过知识蒸馏，在更密集点云上训练的教师模型将不完整-完整关联知识传递给学生模型，增强其估计整体形状和预测缺失区域的能力。级联网络和知识蒸馏共同增强了模型捕捉全局形状上下文同时细化局部细节的能力，有效地弥合了不完整输入和完整目标之间的差距。在ShapeNet-55上针对不同难度设置的实验表明，CasPoinTr在形状恢复和细节保留方面优于现有方法，突显了我们级联结构和蒸馏策略的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全问题，即从不完整的三维点云数据预测完整的形状和重建缺失区域。这个问题在现实中很重要，因为实际收集的点云经常因传感器限制、遮挡和噪声而不完整，而完整的点云数据对自动驾驶、机器人导航、3D重建等许多应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人类处理点云补全时先推断整体形状再预测细节，受此启发，他们借鉴了计算机视觉中的级联网络和知识蒸馏技术。作者发现现有方法（如PoinTr和AdaPoinTr）虽然预测了缺失区域的中心点，但缺乏全局视角；同时注意到级联网络存在误差累积问题，知识蒸馏存在师生范式差距问题。基于这些思考，作者设计了CasPoinTr，结合辅助补全和精心设计的教师模型特权输入来解决这些问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云补全任务分解为两个协同阶段：形状重建生成辅助信息，融合补全利用这些信息和知识蒸馏生成最终输出。整体流程是：1)形状重建阶段直接进行4倍上采样生成密集点云；2)融合补全阶段以原始不完整点云为输入，利用形状重建的输出作为辅助信息；3)使用教师模型（具有2N×3分辨率的特权输入）通过知识蒸馏指导学生模型学习不完整-完整关联知识；4)结合原始损失函数和知识蒸馏损失（KL散度）进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于辅助补全的级联网络结构，提供补全目标信息同时减轻误差累积；2)基于知识蒸馏的训练策略，使用适当分辨率的特权输入解决师生范式差距和捷径问题；3)整体框架结合级联网络和知识蒸馏。相比之前的工作（如AdaPoinTr），CasPoinTr不再仅依赖不完整点云预测缺失区域，而是引入辅助信息指导补全过程，并通过知识蒸馏从教师模型学习更全面的不完整-完整关联知识，显著提高了形状恢复和细节保留能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CasPoinTr通过创新的级联网络结构和知识蒸馏策略，有效解决了点云补全中的误差累积和知识传递问题，显著提高了从稀疏点云恢复完整三维形状的能力。'}


### 论文摘要

Point clouds collected from real-world environments are often incomplete due to factors such as limited sensor resolution, single viewpoints, occlusions, and noise. These challenges make point cloud completion essential for various applications. A key difficulty in this task is predicting the overall shape and reconstructing missing regions from highly incomplete point clouds. To address this, we introduce CasPoinTr, a novel point cloud completion framework using cascaded networks and knowledge distillation. CasPoinTr decomposes the completion task into two synergistic stages: Shape Reconstruction, which generates auxiliary information, and Fused Completion, which leverages this information alongside knowledge distillation to generate the final output. Through knowledge distillation, a teacher model trained on denser point clouds transfers incomplete-complete associative knowledge to the student model, enhancing its ability to estimate the overall shape and predict missing regions. Together, the cascaded networks and knowledge distillation enhance the model's ability to capture global shape context while refining local details, effectively bridging the gap between incomplete inputs and complete targets. Experiments on ShapeNet-55 under different difficulty settings demonstrate that CasPoinTr outperforms existing methods in shape recovery and detail preservation, highlighting the effectiveness of our cascaded structure and distillation strategy.

---

## 84. On the Impact of LiDAR Point Cloud Compression on Remote Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2509.23341v1](http://arxiv.org/abs/2509.23341v1)

**作者:** Tiago de S. Fernandes, Ricardo L. de Queiroz

**发布时间:** 2025-09-27

**备注:** 5 pages, 8 figures

### GPT解析

### 总结

这篇简短论文研究了点云压缩对自动驾驶车辆语义分割性能的影响，并估算了所需的带宽要求

### 背景

自动驾驶车辆依赖激光雷达传感器生成三维点云用于精确分割和物体检测。在智能城市框架下，需要了解传输(压缩)对远程(云端)分割而非本地处理的影响

### 目的

了解点云压缩对语义分割性能的影响，并估算所需的带宽要求

### 方法

开发了一种新的失真指标评估影响；测试了MPEG的两种压缩算法(G-PCC和L3C2)和两种语义分割算法(2DPASS和PVKD)；使用Semantic KITTI数据集进行测试

### 主要发现

高质量分割需要约0.6 MB/s通信吞吐量(G-PCC)和2.8 MB/s(L3C2)

### 结论

这些结果对于规划自动驾驶导航的基础设施资源很重要

### 翻译

自动驾驶车辆依赖激光雷达传感器生成三维点云以实现精确的分割和物体检测。在智能城市框架下，我们希望了解传输(压缩)对远程(云端)分割而非本地处理的影响。在这篇简短论文中，我们试图了解点云压缩对语义分割性能的影响，并估算所需的带宽要求。我们开发了一种新的(合适的)失真指标来评估这种影响。在Semantic KITTI数据集上测试了MPEG的两种压缩算法(G-PCC和L3C2)以及两种领先的语义分割算法(2DPASS和PVKD)。结果表明，高质量的分割性能需要大约0.6 MB/s的通信吞吐量(G-PCC)和2.8 MB/s(L3C2)。这些结果对于规划自动驾驶导航的基础设施资源很重要。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文研究LiDAR点云压缩对远程语义分割性能的影响及其所需的带宽要求。这个问题在自动驾驶领域非常重要，因为自动驾驶车辆依赖LiDAR传感器生成大量3D点云数据，这些数据需要压缩后传输到云端处理，而数据压缩可能影响语义分割的准确性，进而影响自动驾驶系统的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统点云失真度量不适合评估压缩对语义分割的影响，因为压缩会导致点消失或类别标签改变。他们借鉴了现有的语义分割算法(2DPASS和PVKD)和压缩算法(G-PCC和L3C2)，但提出了一个新的失真度量方法，特别考虑了点云压缩后点数量减少的情况，并引入权重因子来强调人类类别标签变化的重要性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是比较原始点云和压缩点云中对应点的标签差异，计算失真率。具体流程是：1)对于原始点云中的每个点，找到压缩点云中最近的点；2)建立标签匹配集合和错误集合；3)计算失真率δ=错误点对数/总匹配点对数；4)对人类类别标签引入权重因子α，增加其错误的重要性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出新的语义分割失真度量方法，不依赖于传统点对应关系；2)特别关注人类类别标签变化的严重性，引入权重因子；3)评估了最新压缩算法和语义分割算法的组合。相比之前工作，该方法更适合评估点云压缩对语义分割的影响，且考虑了实际应用中人类标签变化的重要性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种新的度量方法来评估点云压缩对语义分割的影响，并确定了保持高质量分割所需的带宽要求，为自动驾驶车辆在云端处理点云数据提供了重要参考。'}


### 论文摘要

Autonomous vehicles rely on LiDAR sensors to generate 3D point clouds for accurate segmentation and object detection. In a context of a smart city framework, we would like to understand the effect that transmission (compression) can have on remote (cloud) segmentation, instead of local processing. In this short paper, we try to understand the impact of point cloud compression on semantic segmentation performance and to estimate the necessary bandwidth requirements. We developed a new (suitable) distortion metric to evaluate such an impact. Two of MPEG's compression algorithms (GPCC and L3C2) and two leading semantic segmentation algorithms (2DPASS and PVKD) were tested over the Semantic KITTI dataset. Results indicate that high segmentation quality requires communication throughput of approximately 0.6 MB/s for G-PCC and 2.8 MB/s for L3C2. These results are important in order to plan infrastructure resources for autonomous navigation.

---

## 85. LiDAR-based Human Activity Recognition through Laplacian Spectral Analysis

**论文链接:** [http://arxiv.org/abs/2509.23255v1](http://arxiv.org/abs/2509.23255v1)

**作者:** Sasan Sharifipour, Constantino Álvarez Casado, Le Nguyen, Tharindu Ekanayake, Manuel Lage Cañellas, Nhi Nguyen, Miguel Bordallo López

**发布时间:** 2025-09-27

**备注:** 9 pages, 5 figures, 4 tables, 22 references, conference; Code  available at https://github.com/Arritmic/oulu-pointcloud-har

### GPT解析

### 总结

本文提出了一种基于图谱分析的人类活动识别方法，利用LiDAR点云数据实现高准确率的活动分类。

### 背景

人类活动识别在医疗保健、制造业和人机交互中有广泛应用。LiDAR点云相比摄像头具有保护隐私和对光照变化鲁棒的优势。

### 目的

开发一种基于图谱分析的人类活动识别方法，利用LiDAR点云数据实现高效准确的活动分类。

### 方法

将每个LiDAR帧映射为邻近图（epsilon-graph），计算拉普拉斯谱，利用特征值和特征向量的统计量构成姿态描述符，通过滑动窗口的时间统计生成固定向量，最后使用支持向量机和随机森林进行分类。

### 主要发现

在MM-Fi数据集（40名受试者，27种活动）上，采用严格受试者独立协议，该方法在13类康复活动集上达到94.4%的准确率，在所有27类活动上达到90.3%的准确率，超越了基于骨架的基线方法。

### 结论

本文贡献了一种直接从点云几何导出的紧凑且可解释的特征集，为端到端深度学习提供了准确高效的替代方案。

### 翻译

人类活动识别支持医疗保健、制造业和人机交互中的应用。LiDAR点云提供了一种保护隐私的替代方案，且对光照变化具有鲁棒性。我们提出了一种基于图谱分析的人类活动识别方法。每个LiDAR帧被映射到一个邻近图（epsilon-graph），并计算拉普拉斯谱。特征值和特征向量的统计量构成姿态描述符，通过滑动窗口的时间统计产生固定向量，用于支持向量机和随机森林的分类。在包含40名受试者和27种活动的MM-Fi数据集上，采用严格的受试者独立协议，该方法在13类康复活动集上达到94.4%的准确率，在所有27类活动上达到90.3%的准确率。它还超越了MM-Fi上报告的基于骨架的基线方法。贡献是一种直接从点云几何导出的紧凑且可解释的特征集，为端到端深度学习提供了准确高效的替代方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于LiDAR点云的人体活动识别问题。这个问题很重要，因为人体活动识别在医疗保健、制造业和人机交互等领域有广泛应用，而LiDAR提供了一种既保护隐私又不受光照影响的替代方案，相比摄像头和可穿戴设备具有独特优势。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者分析了现有方法的局限性：摄像头有隐私问题，可穿戴设备需要用户配合，射频方法空间分辨率低。他们借鉴了谱图理论中的'形状DNA'概念，将其应用于点云分析。同时结合了图神经网络中的图表示思想和经典机器学习分类方法，设计出直接从点云几何提取特征的方法，避免了传统方法中的骨架提取步骤。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将每个LiDAR帧转换为邻近图，计算拉普拉斯谱，使用特征值和特征向量的统计量作为姿态描述符，并通过时间窗口聚合捕捉动作动态。流程包括：1)使用PV-RCNN检测人体；2)分割人体点云；3)构建邻近图；4)计算拉普拉斯谱；5)提取特征值和特征向量统计量；6)在时间窗口上聚合特征；7)使用SVM或随机森林等分类器进行活动识别。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)直接在点云上应用谱图理论；2)使用拉普拉斯谱作为'结构指纹'；3)结合全局和局部特征(整个身体和四个象限)；4)时间窗口聚合捕捉动态特性；5)提供紧凑高效的特征表示。相比之前工作，不同之处在于：避免了骨架提取步骤，保留了更丰富的几何信息；特征更紧凑高效；可解释性更强；不受光照影响且保护隐私。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于拉普拉斯谱分析的LiDAR点云人体活动识别方法，通过直接从点云几何中提取全局和局部特征，实现了高精度、高效率和强隐私保护的活动识别，无需中间的骨架提取步骤。'}


### 论文摘要

Human Activity Recognition supports applications in healthcare, manufacturing, and human-machine interaction. LiDAR point clouds offer a privacy-preserving alternative to cameras and are robust to illumination. We propose a HAR method based on graph spectral analysis. Each LiDAR frame is mapped to a proximity graph (epsilon-graph) and the Laplacian spectrum is computed. Eigenvalues and statistics of eigenvectors form pose descriptors, and temporal statistics over sliding windows yield fixed vectors for classification with support vector machines and random forests. On the MM-Fi dataset with 40 subjects and 27 activities, under a strict subject-independent protocol, the method reaches 94.4% accuracy on a 13-class rehabilitation set and 90.3% on all 27 activities. It also surpasses the skeleton-based baselines reported for MM-Fi. The contribution is a compact and interpretable feature set derived directly from point cloud geometry that provides an accurate and efficient alternative to end-to-end deep learning.

---

## 86. Unsupervised Online 3D Instance Segmentation with Synthetic Sequences and Dynamic Loss

**论文链接:** [http://arxiv.org/abs/2509.23194v1](http://arxiv.org/abs/2509.23194v1)

**作者:** Yifan Zhang, Wei Zhang, Chuangxin He, Zhonghua Miao, Junhui Hou

**发布时间:** 2025-09-27

**备注:** 10 pages, 6 figures

### GPT解析

### 总结

这篇论文提出了一种新的无监督在线三维实例分割框架，通过合成点云序列生成和灵活采样策略，解决了现有方法在训练多样性、时间采样和伪标签依赖方面的局限性，在多个数据集上取得了优于UNIT等基线方法的性能。

### 背景

无监督在线三维实例分割是一项基础但具有挑战性的任务，它要求在没有标注训练数据的情况下，维持激光雷达扫描之间的一致物体身份。现有方法如UNIT在这一方向上取得了一定进展，但仍存在训练多样性有限、时间采样刚性以及严重依赖噪声伪标签等问题。

### 目的

提出一个新的框架，通过合成点云序列生成丰富训练分布，增加训练多样性而不依赖手动标签或模拟引擎；同时通过灵活的采样策略和动态加权损失，提高模型捕捉时间动态和学习鲁棒表示的能力，从而实现更准确的三维实例分割和更可靠的时间关联。

### 方法

提出的新框架包括三个主要创新：1) 通过合成点云序列生成丰富训练分布，增加训练多样性；2) 采用灵活的采样策略，利用相邻和非相邻帧，使模型能够学习长期依赖和短期变化；3) 设计动态加权损失函数，强调自信和信息丰富的样本，引导网络学习更鲁棒的表示。

### 主要发现

通过在SemanticKITTI、nuScenes和PandaSet数据集上的大量实验，该方法一致优于UNIT和其他无监督基线方法，实现了更高的分割精度和更鲁棒的时间关联。这表明通过合成数据生成、灵活采样策略和动态加权损失可以有效解决无监督在线三维实例分割中的关键挑战。

### 结论

该研究成功开发了一种新的无监督在线三维实例分割框架，通过合成点云序列生成、灵活采样策略和动态加权损失，克服了现有方法的局限性，在多个数据集上取得了优越的性能，为无监督三维实例分割领域提供了新的思路和方法。

### 翻译

无监督在线三维实例分割是一项基础但具有挑战性的任务，因为它需要在不依赖标注训练数据的情况下，维持激光雷达扫描之间的一致物体身份。现有方法如UNIT已在这一方向取得进展，但仍受限于训练多样性有限、时间采样刚性以及对噪声伪标签的严重依赖。我们提出了一种新框架，通过合成点云序列生成丰富训练分布，在不依赖手动标签或模拟引擎的情况下实现更大的多样性。为更好地捕捉时间动态，我们的方法结合了灵活的采样策略，利用相邻和非相邻帧，使模型能够学习长期依赖和短期变化。此外，动态加权损失强调自信和信息丰富的样本，引导网络学习更鲁棒的表示。在SemanticKITTI、nuScenes和PandaSet上的大量实验表明，我们的方法一致优于UNIT和其他无监督基线，实现了更高的分割精度和更鲁棒的时间关联。代码将在github.com/Eaphan/SFT3D公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决无监督在线3D实例分割问题，即在不依赖人工标注数据的情况下，对连续的LiDAR扫描点云进行物体分割，并保持跨时间的一致性。这个问题在自动驾驶和机器人等领域至关重要，因为这些应用需要实时处理3D感知数据，而获取大规模标注数据既耗时又昂贵。在线处理能力对实时应用必不可少，因为系统必须能处理连续到达的数据而不能依赖未来信息。3D实例分割是场景理解的关键任务，对物体识别、跟踪和场景理解至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有UNIT方法的局限性：训练数据多样性有限、时间采样策略僵化、依赖噪声伪标签、对所有样本一视同仁。基于这些不足，作者设计了三个创新点：点云序列合成、灵活时间采样和动态加权损失。作者借鉴了现有的无监督3D实例分割方法（如UNIT、TARL-Seg等）和合成数据生成技术，但进行了创新改进，特别是直接从无监督聚类结果构建合成LiDAR序列，而不需要人工标注或外部模拟器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三个创新提升无监督在线3D实例分割性能：生成多样化的合成点云序列、使用灵活的时间采样策略捕获短期和长期依赖、应用动态加权损失关注自信和动态样本。整体流程：首先通过时空聚类生成初始伪标签；然后使用点云序列合成生成新训练数据；训练自回归基于查询的网络，一次处理一个扫描并在时间步间传播查询嵌入；采用灵活时间采样（混合相邻和非相邻帧及反转顺序）；使用动态加权损失降低低置信度点影响，提高动态实例权重；推理时模型完全在线运行，不访问未来扫描或显式多扫描配准。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三个：1）点云序列合成：直接从无监督聚类结果构建合成序列，不依赖人工标注或外部模拟器；2）灵活时间采样：不仅选择相邻帧，还引入非相邻帧采样并反转时间顺序，捕获长期依赖；3）动态加权损失：基于置信度缩放损失和基于运动的动态对象加权。相比UNIT等前人工作，不同之处在于：合成数据生成无需人工标注；时间采样更灵活，引入双向学习；损失函数能区分样本重要性，优先关注自信和动态实例。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过合成点云序列和动态损失函数来增强无监督在线3D实例分割的新方法，显著提高了物体分割的准确性和时间一致性，同时减少了对人工标注数据的依赖。'}


### 论文摘要

Unsupervised online 3D instance segmentation is a fundamental yet challenging task, as it requires maintaining consistent object identities across LiDAR scans without relying on annotated training data. Existing methods, such as UNIT, have made progress in this direction but remain constrained by limited training diversity, rigid temporal sampling, and heavy dependence on noisy pseudo-labels. We propose a new framework that enriches the training distribution through synthetic point cloud sequence generation, enabling greater diversity without relying on manual labels or simulation engines. To better capture temporal dynamics, our method incorporates a flexible sampling strategy that leverages both adjacent and non-adjacent frames, allowing the model to learn from long-range dependencies as well as short-term variations. In addition, a dynamic-weighting loss emphasizes confident and informative samples, guiding the network toward more robust representations. Through extensive experiments on SemanticKITTI, nuScenes, and PandaSet, our method consistently outperforms UNIT and other unsupervised baselines, achieving higher segmentation accuracy and more robust temporal associations. The code will be publicly available at github.com/Eaphan/SFT3D.

---

## 87. Desensitizing for Improving Corruption Robustness in Point Cloud Classification through Adversarial Training

**论文链接:** [http://arxiv.org/abs/2509.23010v1](http://arxiv.org/abs/2509.23010v1)

**作者:** Zhiqiang Tian, Weigang Li, Chunhua Deng, Junwei Hu, Yongqiang Wang, Wenping Liu

**发布时间:** 2025-09-27

### GPT解析

### 总结

本研究针对点云损坏问题，提出了一种去敏感对抗训练方法(DesenAT)，通过减少深度神经网络对点云特征的过度依赖来提高模型鲁棒性。

### 背景

由于场景复杂性、传感器不精确性和处理不精确性，点云损坏是不可避免的。深度神经网络过度依赖输入特征是其脆弱性的根本原因，但尚不清楚这一问题是否存在于3D点云任务中。

### 目的

探究深度神经网络对点云特征的敏感性，并验证减少对特征的依赖是否能增强模型对损坏点云的鲁棒性。

### 方法

使用Shapley值量化DNN对点云特征的敏感性；提出DesenAT方法，通过特征去敏感化生成对抗样本，并在自蒸馏框架内进行训练；消除高贡献成分的数据点，使用空间变换模拟损坏场景；利用自蒸馏将干净样本知识转移到对抗样本中。

### 主要发现

传统方法训练的模型对某些特征表现出高敏感性；在相同修剪比例下，优先修剪高敏感性特征比随机修剪造成更严重的性能损害；DesenAT方法能有效提高模型鲁棒性而不降低干净数据集性能。

### 结论

DesenAT方法通过减少DNN对点云特征的过度依赖，成功提高了模型对损坏点云的鲁棒性，在ModelNet-C和PointCloud-C数据集上验证了其有效性。

### 翻译

由于场景复杂性、传感器不精确性和处理不精确性，点云损坏是不可避免的。深度神经网络过度依赖输入特征是其脆弱性的根本原因。目前尚不清楚这个问题是否存在于涉及点云的3D任务中，以及减少对这些特征的依赖是否能增强模型对损坏点云的鲁棒性。本研究试图回答这些问题。具体而言，我们使用Shapley值量化了DNN对点云特征的敏感性，发现使用传统方法训练的模型对某些特征表现出高敏感性值。此外，在相同的修剪比例下，优先修剪高敏感性特征比随机修剪对模型性能造成更严重的损害。我们提出了'去敏感对抗训练'(DesenAT)，使用特征去敏感化生成对抗样本并在自蒸馏框架内进行训练，旨在通过平滑敏感性来减轻DNN对点云特征的过度依赖。首先，消除具有高贡献成分的数据点，并使用空间变换来模拟损坏场景，生成对抗样本并对模型进行对抗训练。接下来，为了补偿对抗样本中的信息损失，我们使用自蒸馏方法将干净样本的知识转移到对抗样本中，并以蒸馏方式进行对抗训练。在ModelNet-C和PointCloud-C上的大量实验表明，所提出的方法可以在不降低干净数据集性能的情况下有效提高模型的鲁棒性。此代码可在https://github.com/JerkyT/DesenAT/tree/master公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云分类模型对数据损坏的鲁棒性问题。由于场景复杂性、传感器不精确性和处理误差，点云数据不可避免会遭受损坏。当前模型过度依赖某些关键特征，导致这些特征在损坏时性能大幅下降。这一问题在现实世界中尤为重要，因为自动驾驶、机器人导航等应用依赖点云数据，而数据损坏会影响系统的可靠性和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过Shapley值量化了模型对点云特征的敏感性，发现传统训练的模型对某些关键特征高度依赖。观察到对抗训练能平滑特征贡献分布后，作者受此启发设计了去敏感对抗训练方法。该方法借鉴了Shapley值理论用于特征解释、对抗训练提高鲁棒性、以及知识蒸馏补偿信息损失等技术，但创新性地将这些技术结合用于解决点云损坏问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是减少模型对特定点云特征的过度依赖，通过平滑模型对特征的敏感性来提高对损坏数据的鲁棒性。整体流程包括：1)使用标准训练训练基准模型；2)用Shapley值计算各点贡献；3)移除高贡献点减少敏感性；4)应用空间变换生成对抗样本；5)进行对抗训练；6)使用自蒸馏方法将干净样本知识转移到对抗样本，弥补信息损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次提出模型对特定点云特征过度依赖导致损坏脆弱性的假设；2)提出通用的去敏感对抗训练(DesenAT)方法；3)首次将Shapley值引入3D点云训练框架；4)设计独特的自蒸馏机制在同一批次内转移知识。相比之前工作，该方法不针对特定损坏类型，而是从通用角度提高鲁棒性，且通过平滑特征敏感性促进对对象几何的更全面理解。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过Shapley值分析和去敏感对抗训练方法，有效提高了3D点云分类模型对数据损坏的鲁棒性，同时保持了在干净数据上的性能表现。'}


### 论文摘要

Due to scene complexity, sensor inaccuracies, and processing imprecision, point cloud corruption is inevitable. Over-reliance on input features is the root cause of DNN vulnerabilities. It remains unclear whether this issue exists in 3D tasks involving point clouds and whether reducing dependence on these features can enhance the model's robustness to corrupted point clouds. This study attempts to answer these questions. Specifically, we quantified the sensitivity of the DNN to point cloud features using Shapley values and found that models trained using traditional methods exhibited high sensitivity values for certain features. Furthermore, under an equal pruning ratio, prioritizing the pruning of highly sensitive features causes more severe damage to model performance than random pruning. We propose `Desensitized Adversarial Training' (DesenAT), generating adversarial samples using feature desensitization and conducting training within a self-distillation framework, which aims to alleviate DNN's over-reliance on point clouds features by smoothing sensitivity. First, data points with high contribution components are eliminated, and spatial transformation is used to simulate corruption scenes, generate adversarial samples, and conduct adversarial training on the model. Next, to compensate for information loss in adversarial samples, we use the self-distillation method to transfer knowledge from clean samples to adversarial samples, and perform adversarial training in a distillation manner.Extensive experiments on ModelNet-C and PointCloud-C demonstrate show that the propose method can effectively improve the robustness of the model without reducing the performance of clean data sets. This code is publicly available at \href{https://github.com/JerkyT/DesenAT/tree/master}{https://github.com/JerkyT/DesenAT}.

---

## 88. SkyLink: Unifying Street-Satellite Geo-Localization via UAV-Mediated 3D Scene Alignment

**论文链接:** [http://arxiv.org/abs/2509.24783v1](http://arxiv.org/abs/2509.24783v1)

**作者:** Hongyang Zhang, Yinhao Liu, Zhenyu Kuang

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了SkyLink方法，解决跨视角地理定位中由极端视角差异引起的语义退化问题，通过数据增强、特征聚合和3D场景信息整合提高特征检索的鲁棒性

### 背景

跨视角地理定位旨在建立不同视角之间的位置对应关系，现有方法通常通过直接特征相似性匹配学习跨视角相关性，但忽略了极端视角差异导致的语义退化问题

### 目的

解决视角变化下的鲁棒特征检索问题，提高跨视角地理定位的准确性

### 方法

使用Google检索增强模块进行街景图像数据增强；采用感知感知特征聚合模块强调局部特征聚合；集成多尺度无人机图像构建的3D场景信息作为街景和卫星视角间的桥梁；通过自监督和跨视角对比学习进行特征对齐

### 主要发现

实验结果表明该方法在不同城市场景中具有鲁棒性和泛化能力，在UAVM2025挑战赛的University-1652数据集上达到25.75%的Recall@1准确率

### 结论

SkyLink方法有效解决了跨视角地理定位中的语义退化问题，提高了特征检索的鲁棒性和泛化能力

### 翻译

跨视角地理定位旨在建立不同视角之间的位置对应关系。现有方法通常通过直接特征相似性匹配来学习跨视角相关性，但往往忽略了由极端视角差异引起的语义退化问题。为解决这一独特问题，我们专注于视角变化下的鲁棒特征检索，提出了新颖的SkyLink方法。我们首先利用Google检索增强模块对街景图像进行数据增强，缓解了受限街景视角下关键目标的遮挡问题。进一步采用感知感知特征聚合模块强调多个局部特征聚合，确保跨视角的特征提取一致性。同时，我们将从多尺度无人机图像构建的3D场景信息整合为街景和卫星视角之间的桥梁，并通过自监督和跨视角对比学习进行特征对齐。实验结果表明该方法在不同城市场景中具有鲁棒性和泛化能力，在UAVM2025挑战赛的University-1652数据集上达到25.75%的Recall@1准确率。代码将在https://github.com/HRT00/CVGL-3D发布

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决跨视角地理定位问题，即如何准确匹配街道视角（地面拍摄）的图像与卫星视角的图像，建立它们之间的地理位置对应关系。这个问题在现实中非常重要，因为它能帮助自动驾驶系统、导航应用、城市规划工具和灾害监测系统等将地面照片与卫星图像关联起来，提供更全面的空间信息。研究中，由于街道和卫星视角差异巨大（一个是地面仰视，一个是俯视），直接匹配非常困难，现有方法往往因视角差异导致的语义退化问题而表现不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了跨视角地理定位的核心挑战：视角差异大导致的外观和语义不一致。他们注意到关键结构元素（如道路、建筑轮廓）在不同视角中保持可见，可作为对应点。作者借鉴了现有工作：1) 受近期工作启发，将街道全景转换为鸟瞰图(BEV)图像；2) 认识到3D点云能更好保留空间布局；3) 使用了DINOv2作为基础视觉模型和PointCLIP作为点云特征提取器。在此基础上，作者设计了四个关键组件：Google检索增强模块(GREM)解决街道视角多样性问题；基于补丁的特征聚合模块(PAFA)增强特征交互；多尺度3D场景桥接模块(MSBM)连接不同视角；以及跨视角特征对齐方法确保语义一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用无人机拍摄的多尺度3D场景作为桥梁，连接街道视角和卫星视角，通过3D重建解决两者间的巨大视角差异。整体流程包括：1) 使用GREM模块从Google图像检索相似街道图像，增强数据多样性；2) 采用DINOv2孪生网络和PAFA模块从增强的街道和卫星图像中提取特征；3) 将不同高度的无人机图像重建为多尺度3D点云，作为中间表示；4) 通过跨视角对比损失和自监督对比学习联合优化，确保街道、卫星和3D点云特征间的语义一致性；5) 测试时使用冻结特征提取器和测试时增强(TTA)提高鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Google检索增强模块(GREM)，通过检索相似街道图像解决视角单一问题；2) 基于补丁的特征聚合模块(PAFA)，有效整合跨视角特征关系；3) 多尺度3D场景桥接模块(MSBM)，利用低中高三个尺度的无人机图像重建3D点云；4) 组合跨视角对比损失和自监督对比学习的优化策略。相比之前工作，SkyLink不使用传统的极坐标或BEV变换（这些会引入几何失真），而是用3D点云作为桥梁；主动增强街道数据多样性而非直接使用原始数据；采用更先进的特征聚合方式；同时考虑跨视角和自监督学习，提高了特征判别能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SkyLink通过无人机介导的多尺度3D场景重建和创新的特征对齐方法，有效解决了街道-卫星跨视角地理定位中的视角差异问题，显著提高了定位精度和鲁棒性。'}


### 论文摘要

Cross-view geo-localization aims at establishing location correspondences between different viewpoints. Existing approaches typically learn cross-view correlations through direct feature similarity matching, often overlooking semantic degradation caused by extreme viewpoint disparities. To address this unique problem, we focus on robust feature retrieval under viewpoint variation and propose the novel SkyLink method. We firstly utilize the Google Retrieval Enhancement Module to perform data enhancement on street images, which mitigates the occlusion of the key target due to restricted street viewpoints. The Patch-Aware Feature Aggregation module is further adopted to emphasize multiple local feature aggregations to ensure the consistent feature extraction across viewpoints. Meanwhile, we integrate the 3D scene information constructed from multi-scale UAV images as a bridge between street and satellite viewpoints, and perform feature alignment through self-supervised and cross-view contrastive learning. Experimental results demonstrate robustness and generalization across diverse urban scenarios, which achieve 25.75$\%$ Recall@1 accuracy on University-1652 in the UAVM2025 Challenge. Code will be released at https://github.com/HRT00/CVGL-3D.

---

## 89. Spatial-Functional awareness Transformer-based graph archetype contrastive learning for Decoding Visual Neural Representations from EEG

**论文链接:** [http://arxiv.org/abs/2509.24761v1](http://arxiv.org/abs/2509.24761v1)

**作者:** Yueming Sun, Long Yang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本研究提出了一种空间-功能感知Transformer-based图原型对比学习(SFTG)框架，用于增强基于脑电图(EEG)的视觉解码，显著优于现有方法。

### 背景

从脑电图(EEG)信号解码视觉神经表示具有挑战性，因为这类信号具有高维、噪声大和非欧几里得性质。

### 目的

开发一种有效的方法来增强基于EEG的视觉解码能力，克服现有方法的局限性。

### 方法

提出SFTG框架，包含EEG图Transformer(EGT)架构，用于同时编码大脑空间连接和神经时间动态；并引入图原型对比学习(GAC)来减轻受试者内部变异性，学习受试者特定的EEG图原型以提高特征一致性和类别可分性。

### 主要发现

在Things-EEG数据集上的受试者依赖和受试者独立评估表明，该方法显著优于先前最先进的EEG解码方法。

### 结论

将基于图的学习与对比目标相结合具有变革潜力，可增强基于EEG的脑解码，为更通用和稳健的神经表示开辟新途径。

### 翻译

从脑电图(EEG)信号解码视觉神经表示由于其高维、噪声大和非欧几里得的特性，仍然是一个巨大的挑战。在这项工作中，我们提出了一个空间-功能感知Transformer-based图原型对比学习(SFTG)框架，以增强基于EEG的视觉解码。具体而言，我们引入了EEG图Transformer(EGT)，这是一种新颖的基于图的神经架构，可同时编码大脑的空间连接和神经的时间动态。为了减轻受试者内部的高变异性，我们提出了图原型对比学习(GAC)，它学习受试者特定的EEG图原型，以提高特征一致性和类别可分性。此外，我们在Things-EEG数据集上进行了全面的受试者依赖和受试者独立评估，证明我们的方法显著优于先前最先进的EEG解码方法。这些结果强调了将基于图的学习与对比目标相结合以增强基于EEG的脑解码的变革潜力，为更通用和稳健的神经表示铺平了道路。


### 论文摘要

Decoding visual neural representations from Electroencephalography (EEG) signals remains a formidable challenge due to their high-dimensional, noisy, and non-Euclidean nature. In this work, we propose a Spatial-Functional Awareness Transformer-based Graph Archetype Contrastive Learning (SFTG) framework to enhance EEG-based visual decoding. Specifically, we introduce the EEG Graph Transformer (EGT), a novel graph-based neural architecture that simultaneously encodes spatial brain connectivity and temporal neural dynamics. To mitigate high intra-subject variability, we propose Graph Archetype Contrastive Learning (GAC), which learns subject-specific EEG graph archetypes to improve feature consistency and class separability. Furthermore, we conduct comprehensive subject-dependent and subject-independent evaluations on the Things-EEG dataset, demonstrating that our approach significantly outperforms prior state-of-the-art EEG decoding methods.The results underscore the transformative potential of integrating graph-based learning with contrastive objectives to enhance EEG-based brain decoding, paving the way for more generalizable and robust neural representations.

---

## 90. LEAF: A Robust Expert-Based Framework for Few-Shot Continual Event Detection

**论文链接:** [http://arxiv.org/abs/2509.24547v1](http://arxiv.org/abs/2509.24547v1)

**作者:** Bao-Ngoc Dao, Quang Nguyen, Luyen Ngo Dinh, Minh Le, Linh Ngo Van

**发布时间:** 2025-09-29

### GPT解析

### 总结

这篇论文提出了LEAF，一种新型且基于专家的少样本持续事件检测(FCED)框架，通过专家混合架构、语义感知专家选择机制、对比学习目标和知识蒸馏策略，解决了FCED中的有限数据学习和灾难性遗忘问题，在多个基准测试上取得了最先进性能。

### 背景

少样本持续事件检测(FCED)面临从有限数据学习和减轻跨顺序任务灾难性遗忘的双重挑战。现有方法通常因对共享基础模型进行完全微调而导致严重的遗忘问题，且依赖可能引入不自然或语义失真输入的数据增强策略。

### 目的

解决FCED中的知识遗忘问题，提高模型在有限数据条件下的泛化能力，避免数据增强带来的语义失真，实现更有效的持续学习。

### 方法

LEAF框架将专家混合架构集成到基础模型中，每个专家使用低秩适应(LoRA)矩阵参数化；通过语义感知专家选择机制动态将实例路由到最相关专家；采用由标签描述引导的对比学习目标提高有限数据设置下的泛化能力；使用知识蒸馏策略防止在记忆缓冲区上过拟合。

### 主要发现

在多个FCED基准测试上，LEAF框架一致取得了最先进的性能，证明了其在解决少样本学习和灾难性遗忘问题上的有效性。

### 结论

LEAF框架通过专家混合架构和多种创新机制，有效解决了FCED中的关键挑战，为事件检测领域的持续学习提供了新的解决方案。

### 翻译

少样本持续事件检测(Few-shot Continual Event Detection, FCED)带来了从有限数据学习和减轻跨顺序任务灾难性遗忘的双重挑战。现有方法通常因对共享基础模型进行完全微调而遭受严重的遗忘问题，这导致任务间的知识干扰。此外，它们经常依赖可能引入不自然或语义失真输入的数据增强策略。为解决这些局限性，我们提出了LEAF，一种用于FCED的新型且稳健的基于专家的框架。LEAF将专家混合架构集成到基础模型中，其中每个专家使用低秩适应(LoRA)矩阵参数化。一个语义感知的专家选择机制动态将实例路由到最相关的专家，实现专家专业化和减少知识干扰。为了提高有限数据设置下的泛化能力，LEAF融合了一个由标签描述引导的对比学习目标，该目标捕获了关于事件类型的高级语义信息。此外，为防止在记忆缓冲区上过拟合，我们的框架采用了一种将先前模型知识转移到当前模型的知识蒸馏策略。在多个FCED基准上的广泛实验表明，LEAF一致取得了最先进的性能。


### 论文摘要

Few-shot Continual Event Detection (FCED) poses the dual challenges of learning from limited data and mitigating catastrophic forgetting across sequential tasks. Existing approaches often suffer from severe forgetting due to the full fine-tuning of a shared base model, which leads to knowledge interference between tasks. Moreover, they frequently rely on data augmentation strategies that can introduce unnatural or semantically distorted inputs. To address these limitations, we propose LEAF, a novel and robust expert-based framework for FCED. LEAF integrates a specialized mixture of experts architecture into the base model, where each expert is parameterized with low-rank adaptation (LoRA) matrices. A semantic-aware expert selection mechanism dynamically routes instances to the most relevant experts, enabling expert specialization and reducing knowledge interference. To improve generalization in limited-data settings, LEAF incorporates a contrastive learning objective guided by label descriptions, which capture high-level semantic information about event types. Furthermore, to prevent overfitting on the memory buffer, our framework employs a knowledge distillation strategy that transfers knowledge from previous models to the current one. Extensive experiments on multiple FCED benchmarks demonstrate that LEAF consistently achieves state-of-the-art performance.

---

## 91. Contrastive Learning for Correlating Network Incidents

**论文链接:** [http://arxiv.org/abs/2509.24446v1](http://arxiv.org/abs/2509.24446v1)

**作者:** Jeremias Dötterl

**发布时间:** 2025-09-29

**备注:** Accepted at The 26th International Conference on Intelligent Data  Engineering and Automated Learning (IDEAL 2025). This work was partially  funded by the German Federal Ministry of Research, Technology and Space  (BMFTR) in the FRONT-RUNNER project (Grant 16KISR005K)

### GPT解析

### 总结

本文提出了一种基于自监督学习的网络故障关联方法，通过对比学习训练深度神经网络，实现了高精度的网络情况相似性关联。

### 背景

互联网服务提供商需要监控网络以检测、分类和修复服务故障。由于网络规模庞大，手动关联故障不可行，因此需要自动关联方法。

### 目的

开发一种自动化的方法来关联网络中的相似故障，确定过去是否发生过类似故障或网络其他地方是否同时发生故障。

### 方法

提出一种基于相似性的网络情况关联的自监督学习方法，使用对比学习在一个大型无标签的网络情况数据集上训练深度神经网络。

### 主要发现

在真实网络监控数据上的实验中，该方法实现了高精度，表明对比学习在网络故障关联方面具有很好的效果。

### 结论

对比学习是网络故障关联的一种很有前途的方法，能够有效解决大规模网络中故障自动关联的问题。

### 翻译

互联网服务提供商监控其网络以检测、分类和修复服务故障。当检测到故障时，确定过去是否发生过类似故障或网络其他地方是否同时发生故障非常重要。由于观察到的网络规模庞大，手动关联此类故障是不可行的，这使得自动关联成为必要。本文提出了一种基于相似性的网络情况关联的自监督学习方法。使用这种方法，通过对比学习在一个大型无标签的网络情况数据集上训练深度神经网络。在真实网络监控数据上的实验中 achieved 高精度，这表明对比学习是网络故障关联的一种很有前途的方法。


### 论文摘要

Internet service providers monitor their networks to detect, triage, and remediate service impairments. When an incident is detected, it is important to determine whether similar incidents have occurred in the past or are happening concurrently elsewhere in the network. Manual correlation of such incidents is infeasible due to the scale of the networks under observation, making automated correlation a necessity. This paper presents a self-supervised learning method for similarity-based correlation of network situations. Using this method, a deep neural network is trained on a large unlabeled dataset of network situations using contrastive learning. High precision achieved in experiments on real-world network monitoring data suggests that contrastive learning is a promising approach to network incident correlation.

---

## 92. REALIGN: Regularized Procedure Alignment with Matching Video Embeddings via Partial Gromov-Wasserstein Optimal Transport

**论文链接:** [http://arxiv.org/abs/2509.24382v1](http://arxiv.org/abs/2509.24382v1)

**作者:** Soumyadeep Chandra, Kaushik Roy

**发布时间:** 2025-09-29

**备注:** 10 pages, 4 figures, 6 tables

### GPT解析

### 总结

本文提出了一种名为REALIGN的自监督框架，用于从程序视频中学习，通过结合视觉对应关系和时间关系，有效处理教学视频中的背景片段、重复动作和非单调步骤顺序等问题。

### 背景

从程序视频中学习是自监督表征学习中的核心挑战，因为现实世界中的教学数据通常包含背景片段、重复动作和顺序混乱的步骤，这种变化性违反了许多对齐方法所依赖的强单调性假设。

### 目的

开发一种能够捕捉任务高阶时间结构并稳健处理程序视频中常见变体的自监督学习方法。

### 方法

引入基于正则化融合部分Gromov-Wasserstein最优传输（R-FPGWOT）的REALIGN框架，在部分对齐方案下联合建模视觉对应关系和时间关系，并将FPGWOT距离与序列间对比学习相结合以稳定训练。

### 主要发现

在第一人称（EgoProceL）和第三人称（ProceL、CrossTask）基准测试中，REALIGN实现了高达18.9%的平均F1分数提升和超过30%的时间IoU收益，同时产生更具可解释性的传输图，保留关键步骤顺序并过滤噪声。

### 结论

REALIGN通过联合建模视觉对应关系和时间关系，有效解决了现有方法仅依赖特征相似性的局限，能够稳健处理程序视频中的各种变体，显著提升了学习效果。

### 翻译

从程序视频中学习一直是自监督表征学习中的一个核心挑战，因为现实世界中的教学数据通常包含背景片段、重复动作以及顺序混乱的步骤。这种变化性违反了许多对齐方法所依赖的强单调性假设。先前最先进的方法（如OPEL）利用Kantorovich最优传输（KOT）来构建帧到帧的对应关系，但仅依赖于特征相似性，无法捕捉任务的高阶时间结构。在本文中，我们引入了REALIGN，这是一种基于正则化融合部分Gromov-Wasserstein最优传输（R-FPGWOT）的自监督框架。与KOT不同，我们的公式在部分对齐方案下联合建模视觉对应关系和时间关系，能够稳健处理教学视频中常见的无关帧、重复动作和非单调步骤顺序。为了稳定训练，我们将FPGWOT距离与序列间对比学习相结合，避免了需要多个正则化项并防止退化解。在第一人称（EgoProceL）和第三人称（ProceL、CrossTask）基准测试中，REALIGN实现了高达18.9%的平均F1分数提升和超过30%的时间IoU收益，同时生成更具可解释性的传输图，保留关键步骤顺序并过滤噪声。


### 论文摘要

Learning from procedural videos remains a core challenge in self-supervised representation learning, as real-world instructional data often contains background segments, repeated actions, and steps presented out of order. Such variability violates the strong monotonicity assumptions underlying many alignment methods. Prior state-of-the-art approaches, such as OPEL, leverage Kantorovich Optimal Transport (KOT) to build frame-to-frame correspondences, but rely solely on feature similarity and fail to capture the higher-order temporal structure of a task. In this paper, we introduce REALIGN, a self-supervised framework for procedure learning based on Regularized Fused Partial Gromov-Wasserstein Optimal Transport (R-FPGWOT). In contrast to KOT, our formulation jointly models visual correspondences and temporal relations under a partial alignment scheme, enabling robust handling of irrelevant frames, repeated actions, and non-monotonic step orders common in instructional videos. To stabilize training, we integrate FPGWOT distances with inter-sequence contrastive learning, avoiding the need for multiple regularizers and preventing collapse to degenerate solutions. Across egocentric (EgoProceL) and third-person (ProceL, CrossTask) benchmarks, REALIGN achieves up to 18.9% average F1-score improvements and over 30% temporal IoU gains, while producing more interpretable transport maps that preserve key-step orderings and filter out noise.

---

## 93. Preserving Cross-Modal Stability for Visual Unlearning in Multimodal Scenarios

**论文链接:** [http://arxiv.org/abs/2509.23895v1](http://arxiv.org/abs/2509.23895v1)

**作者:** Jinghan Xu Yuyang Zhang Qixuan Cai Jiancheng Chen Keqiu Li

**发布时间:** 2025-09-28

**备注:** 9 pages,4 figures

### GPT解析

### 总结

论文提出了一种跨模态对比遗忘（CCU）框架，用于解决多模态应用中视觉模态的隐私泄露问题，同时保持跨模态知识和模型性能。

### 背景

在现实世界多模态应用（如自动驾驶中的视觉和雷达数据）中，视觉模态最容易受到隐私泄露的影响。

### 目的

解决现有机器遗忘方法无法保留跨模态知识和保持保留数据类内结构稳定性的问题，避免在视觉遗忘过程中导致整体性能和其他模态性能下降。

### 方法

CCU框架整合三个关键组件：(a)选择性视觉遗忘：使用反向对比学习将视觉表示与其原始语义分离；(b)跨模态知识保留：通过语义一致性保持其他模态的判别能力；(c)双集对比分离：通过分离遗忘集和保留集之间的结构扰动来保持模型性能。

### 主要发现

在三个数据集上的广泛实验证明了CCU的优越性，与最准确的基线相比，该方法实现了7.12%的准确率提升，而遗忘时间仅为7%。

### 结论

CCU框架能够在有效解决视觉模态隐私泄露问题的同时，保持跨模态知识和模型性能，显著优于现有方法。

### 翻译

视觉模态在自动驾驶等现实世界多模态应用中（结合视觉和雷达数据）最容易受到隐私泄露；机器遗忘通过从预训练模型中移除特定训练数据来解决隐私泄露问题，然而现有方法无法保留跨模态知识并保持保留数据的类内结构稳定性，导致在视觉遗忘过程中整体性能和其他模态性能下降；为应对这些挑战，我们提出了一个跨模态对比遗忘（CCU）框架，该框架集成了三个关键组件：(a)选择性视觉遗忘：采用反向对比学习将视觉表示与其原始语义分离；(b)跨模态知识保留：通过语义一致性保持其他模态的判别能力；(c)双集对比分离：通过分离遗忘集和保留集之间的结构扰动来保持模型性能；在三个数据集上的广泛实验证明了CCU的优越性，与最高准确率的基线相比，我们的方法实现了7.12%的准确率提升，而遗忘时间仅为7%。


### 论文摘要

Visual modality is the most vulnerable to privacy leakage in real-world multimodal applications like autonomous driving with visual and radar data; Machine unlearning removes specific training data from pre-trained models to address privacy leakage, however, existing methods fail to preserve cross-modal knowledge and maintain intra-class structural stability of retain data, leading to reduced overall and other modalities' performance during visual unlearning; to address these challenges, we propose a Cross-modal Contrastive Unlearning (CCU) framework, which integrates three key components: (a) selective visual unlearning: employing inverse contrastive learning to dissociate visual representations from their original semantics, (b) cross-modal knowledge retention: preserving other modalities' discriminability through semantic consistency, and (c) dual-set contrastive separation: preserving the model performance via isolation of structural perturbations between the unlearn set and retain set; extensive experiments on three datasets demonstrate the superiority of CCU, and our method achieves a 7.12% accuracy improvement with only 7% of the unlearning time compared to the top-accuracy baseline.

---

## 94. STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2509.23802v1](http://arxiv.org/abs/2509.23802v1)

**作者:** Yao Luan, Ni Mu, Yiqin Yang, Bo Xu, Qing-Shan Jia

**发布时间:** 2025-09-28

**备注:** NeurIPS 2025

### GPT解析

### 总结

这篇论文提出了一种名为STAIR（STage-AlIgned Reward learning）的方法，用于解决基于偏好强化学习（PbRL）在多阶段任务中的阶段不匹配问题。通过时间距离学习阶段近似，并优先进行同阶段比较，STAIR有效提升了多阶段任务中的学习效果。

### 背景

基于偏好的强化学习（PbRL）通过直接从人类偏好中学习奖励，绕过了复杂的奖励工程，能够更好地与人类意图保持一致。然而，在多阶段任务中（代理顺序执行子任务，如导航、抓取等），其效果受到阶段不匹配的限制：比较来自不匹配阶段的片段（如移动与操作）会导致信息量不足的反馈，从而阻碍策略学习。

### 目的

验证阶段不匹配问题，并提出一种解决方法STAGE-AlIgned Reward learning (STAIR)，以改善PbRL在多阶段任务中的表现。

### 方法

STAIR首先基于时间距离学习阶段近似，然后优先进行同阶段比较。时间距离通过对比学习进行学习，将时间上接近的状态分组为连贯的阶段，无需预定义任务知识，并能动态适应策略变化。

### 主要发现

通过理论分析和实证实验验证了阶段不匹配问题。大量实验表明STAIR在多阶段任务中具有优越性，在单阶段任务中也具有竞争性表现。此外，人类研究表明STAIR近似得到的阶段与人类认知一致，证实了其在减轻阶段不匹配方面的有效性。

### 结论

STAIR方法有效解决了多阶段任务中的阶段不匹配问题，通过时间距离学习阶段近似并优先进行同阶段比较，显著提升了基于偏好强化学习在复杂多阶段任务中的表现。

### 翻译

基于偏好的强化学习（PbRL）通过直接从人类偏好中学习奖励，绕过了复杂的奖励工程，从而能够更好地与人类意图保持一致。然而，在多阶段任务中（代理顺序执行子任务，如导航、抓取等），其效果受到阶段不匹配的限制：比较来自不匹配阶段的片段（如移动与操作）会导致信息量不足的反馈，从而阻碍策略学习。在本文中，我们通过理论分析和实证实验验证了阶段不匹配问题。为解决此问题，我们提出了STAGE-AlIgned Reward learning（STAIR），该方法首先基于时间距离学习阶段近似，然后优先进行同阶段比较。时间距离通过对比学习进行学习，将时间上接近的状态分组为连贯的阶段，无需预定义任务知识，并能动态适应策略变化。大量实验证明了STAIR在多阶段任务中的优越性以及在单阶段任务中的竞争性表现。此外，人类研究表明STAIR近似得到的阶段与人类认知一致，证实了其在减轻阶段不匹配方面的有效性。


### 论文摘要

Preference-based reinforcement learning (PbRL) bypasses complex reward engineering by learning rewards directly from human preferences, enabling better alignment with human intentions. However, its effectiveness in multi-stage tasks, where agents sequentially perform sub-tasks (e.g., navigation, grasping), is limited by stage misalignment: Comparing segments from mismatched stages, such as movement versus manipulation, results in uninformative feedback, thus hindering policy learning. In this paper, we validate the stage misalignment issue through theoretical analysis and empirical experiments. To address this issue, we propose STage-AlIgned Reward learning (STAIR), which first learns a stage approximation based on temporal distance, then prioritizes comparisons within the same stage. Temporal distance is learned via contrastive learning, which groups temporally close states into coherent stages, without predefined task knowledge, and adapts dynamically to policy changes. Extensive experiments demonstrate STAIR's superiority in multi-stage tasks and competitive performance in single-stage tasks. Furthermore, human studies show that stages approximated by STAIR are consistent with human cognition, confirming its effectiveness in mitigating stage misalignment.

---

## 95. GenView++: Unifying Adaptive View Generation and Quality-Driven Supervision for Contrastive Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.23770v1](http://arxiv.org/abs/2509.23770v1)

**作者:** Xiaojie Li, Bei Wang, Jianlong Wu, Yue Yu, Liqiang Nie, Min Zhang

**发布时间:** 2025-09-28

**备注:** The code is available at  \url{https://github.com/xiaojieli0903/GenViewPlusPlus}

### GPT解析

### 总结

GenView++是一个统一的框架，通过多源自适应视图生成机制和质量驱动的对比学习机制解决了对比学习中高质量正样本对构建和利用的问题，在视觉和视觉语言任务上取得了显著效果。

### 背景

对比学习的成功依赖于高质量正样本对的构建和利用，但当前方法在构建方面（手工设计和生成的增强方法缺乏多样性和存在语义损坏风险）和学习方面（缺乏质量评估机制，所有样本对被同等对待）存在关键局限性。

### 目的

解决对比学习中高质量正样本对构建和利用的两个关键问题：提高样本对的多样性和语义一致性，以及引入质量评估机制优化监督效果。

### 方法

GenView++框架包含两个协同创新：1) 多源自适应视图生成机制：通过动态调整生成参数，在图像条件、文本条件和图像文本条件策略下合成多样且语义一致的视图；2) 质量驱动的对比学习机制：评估每个样本对的语义对齐和多样性，动态重新加权训练贡献，优先高质量样本对，抑制冗余或错位样本对。

### 主要发现

1) 在视觉表示学习方面，GenView++将MoCo v2在ImageNet线性分类上的性能提高了2.5%；2) 在视觉语言学习方面，GenView++在十个数据集上比CLIP提高平均零样本分类准确率12.31%，比SLIP提高5.31%；3) GenView++将Flickr30k文本检索R@5提高了3.2%。

### 结论

GenView++通过多源自适应视图生成和质量驱动的对比学习机制有效解决了对比学习中高质量正样本对构建和利用的关键问题，在多种视觉和视觉语言任务上取得了显著性能提升。

### 翻译

对比学习的成功依赖于高质量正样本对的构建和利用。然而，当前方法在构建方面面临两个关键限制：手工设计和生成的增强方法往往缺乏多样性并有语义损坏的风险；在学习方面，缺乏质量评估机制导致监督效果不佳，所有样本对被同等对待。为应对这些挑战，我们提出了GenView++，一个统一框架，通过引入两个协同创新来解决这两个方面的问题。为改进样本对构建，GenView++引入了多源自适应视图生成机制，通过在图像条件、文本条件和图像文本条件策略间动态调整生成参数，合成多样且语义一致的视图。其次，质量驱动的对比学习机制评估每个样本对的语义对齐和多样性，动态重新加权其训练贡献，优先高质量样本对同时抑制冗余或错位样本对。大量实验证明了GenView++在视觉和视觉语言任务上的有效性。对于视觉表示学习，它在ImageNet线性分类上将MoCo v2提高了2.5%。对于视觉语言学习，它在十个数据集上将平均零样本分类准确率比CLIP提高了12.31%，比SLIP提高了5.31%，并将Flickr30k文本检索R@5进一步提高了3.2%。代码可在https://github.com/xiaojieli0903/GenViewPlusPlus获取。


### 论文摘要

The success of contrastive learning depends on the construction and utilization of high-quality positive pairs. However, current methods face critical limitations on two fronts: on the construction side, both handcrafted and generative augmentations often suffer from limited diversity and risk semantic corruption; on the learning side, the absence of a quality assessment mechanism leads to suboptimal supervision where all pairs are treated equally. To tackle these challenges, we propose GenView++, a unified framework that addresses both fronts by introducing two synergistic innovations. To improve pair construction, GenView++ introduces a multi-source adaptive view generation mechanism to synthesize diverse yet semantically coherent views by dynamically modulating generative parameters across image-conditioned, text-conditioned, and image-text-conditioned strategies. Second, a quality-driven contrastive learning mechanism assesses each pair's semantic alignment and diversity to dynamically reweight their training contribution, prioritizing high-quality pairs while suppressing redundant or misaligned pairs. Extensive experiments demonstrate the effectiveness of GenView++ across both vision and vision-language tasks. For vision representation learning, it improves MoCov2 by +2.5% on ImageNet linear classification. For vision-language learning, it raises the average zero-shot classification accuracy by +12.31% over CLIP and +5.31% over SLIP across ten datasets, and further improves Flickr30k text retrieval R@5 by +3.2%. The code is available at https://github.com/xiaojieli0903/GenViewPlusPlus.

---

## 96. A Hierarchical Structure-Enhanced Personalized Recommendation Model for Traditional Chinese Medicine Formulas Based on KG Diffusion Guidance

**论文链接:** [http://arxiv.org/abs/2509.23560v1](http://arxiv.org/abs/2509.23560v1)

**作者:** ChaoBo Zhang, Long Tan

**发布时间:** 2025-09-28

**DOI:** 10.1145/3746252.3761428

**备注:** 10 pages, 10 figures, Proceedings of the 34th ACM International  Conference on Information and Knowledge Management (CIKM)

### GPT解析

### 总结

该研究提出了一种基于知识图谱扩散引导的中药处方个性化推荐模型TCM-HEDPR，解决了现有中药处方推荐中的三个主要问题：患者个性化信息不足、草药数据长尾分布问题以及草药配伍关系考虑不足。

### 背景

人工智能技术在中药处方推荐中扮演着重要角色，以往研究主要关注处方中的症状-草药关系，但存在明显局限性。

### 目的

开发一种能够克服现有局限性的中药处方推荐模型，提高推荐的准确性和有效性。

### 方法

1. 使用患者个性化提示序列预训练症状表示，并通过面向提示的对比学习进行数据增强；2. 采用知识图谱引导的同质图扩散方法结合自注意力机制，全局捕捉非线性症状-草药关系；3. 设计异构图层次网络，整合草药配伍关系与潜在症状，在细粒度水平指导处方生成过程，缓解草药数据长尾分布问题。

### 主要发现

在两个公共数据集和一个临床数据集上的大量实验证明了TCM-HEDPR模型的有效性。结合现代医学和网络药理学见解可以全面评估推荐的处方。

### 结论

该研究为现代中药推荐提供了新的范式，通过整合患者个性化信息、处理长尾数据分布和考虑草药配伍关系，显著提高了中药处方推荐的准确性和临床适用性。

### 翻译

人工智能技术在推荐中药处方方面发挥着关键作用。以往研究通过关注处方中的症状-草药关系取得了显著进展。然而，几个局限性阻碍了模型性能：(i) 对患者个性化信息如年龄、BMI和病史的关注不足，这阻碍了证型的准确识别并降低了疗效。(ii) 草药数据的典型长尾分布引入了训练偏差并影响了泛化能力。(iii) 忽略了草药之间的'君臣佐使'配伍关系，增加了毒性或副作用的风险，与临床中医的'辨证论治'原则相悖。因此，我们提出了一种基于知识图谱扩散引导的中药方剂层次结构增强个性化推荐模型，即TCM-HEDPR。具体而言，我们使用患者个性化提示序列预训练症状表示，并应用面向提示的对比学习进行数据增强。此外，我们采用知识图谱引导的同质图扩散方法结合自注意力机制，全局捕捉非线性症状-草药关系。最后，我们设计了一个异构图层次网络，将草药配伍关系与潜在证型相结合，在细粒度水平指导处方生成过程，缓解草药数据长尾分布问题。在两个公共数据集和一个临床数据集上的大量实验证明了TCM-HEDPR的有效性。此外，我们结合现代医学和网络药理学见解来全面评估推荐的处方。它可以为现代中药推荐提供新的范式。


### 论文摘要

Artificial intelligence technology plays a crucial role in recommending prescriptions for traditional Chinese medicine (TCM). Previous studies have made significant progress by focusing on the symptom-herb relationship in prescriptions. However, several limitations hinder model performance: (i) Insufficient attention to patient-personalized information such as age, BMI, and medical history, which hampers accurate identification of syndrome and reduces efficacy. (ii) The typical long-tailed distribution of herb data introduces training biases and affects generalization ability. (iii) The oversight of the 'monarch, minister, assistant and envoy' compatibility among herbs increases the risk of toxicity or side effects, opposing the 'treatment based on syndrome differentiation' principle in clinical TCM. Therefore, we propose a novel hierarchical structure-enhanced personalized recommendation model for TCM formulas based on knowledge graph diffusion guidance, namely TCM-HEDPR. Specifically, we pre-train symptom representations using patient-personalized prompt sequences and apply prompt-oriented contrastive learning for data augmentation. Furthermore, we employ a KG-guided homogeneous graph diffusion method integrated with a self-attention mechanism to globally capture the non-linear symptom-herb relationship. Lastly, we design a heterogeneous graph hierarchical network to integrate herbal dispensing relationships with implicit syndromes, guiding the prescription generation process at a fine-grained level and mitigating the long-tailed herb data distribution problem. Extensive experiments on two public datasets and one clinical dataset demonstrate the effectiveness of TCM-HEDPR. In addition, we incorporate insights from modern medicine and network pharmacology to evaluate the recommended prescriptions comprehensively. It can provide a new paradigm for the recommendation of modern TCM.

---

## 97. Contrastive Learning Enhances Language Model Based Cell Embeddings for Low-Sample Single Cell Transcriptomics

**论文链接:** [http://arxiv.org/abs/2509.23543v1](http://arxiv.org/abs/2509.23543v1)

**作者:** Luxuan Zhang, Douglas Jiang, Qinglong Wang, Haoqi Sun, Feng Tian

**发布时间:** 2025-09-28

**备注:** 14 pages, 4 figures, 2 tables

### GPT解析

### 总结

研究提出了一种将单细胞RNA测序与大型语言模型结合的计算框架，通过将高表达基因映射到NCBI基因描述并嵌入到文本模型中，改善了稀有细胞亚型的分类，揭示了生物学特征和选择性神经元易感性途径。

### 背景

大型语言模型在自然语言处理、生成、计算机视觉和多模态学习等领域表现出强大能力，但在生物医学数据分析中的应用仍处于初级阶段。单细胞转录组分析对于解析发育和疾病中的细胞亚型多样性至关重要，但稀有亚型对缩放定律提出了挑战。

### 目的

开发一个计算框架，整合单细胞RNA测序与大型语言模型，以获取知识引导的基因嵌入，解决稀有细胞亚型分析中的挑战。

### 方法

将每个细胞的高表达基因映射到NCBI基因描述，并使用text-embedding-ada-002、BioBERT和SciBERT等模型进行嵌入。将该策略应用于视网膜神经节细胞(RGCs)，这些细胞在青光眼相关神经变性的易感性方面存在差异。

### 主要发现

该策略改善了亚型分类，突出了生物学特征，并揭示了选择性神经元易感性背后的途径。LLM衍生的嵌入可以在数据有限条件下增强生物分析。

### 结论

大型语言模型衍生的嵌入可以在数据有限条件下增强生物分析，并为未来单细胞生物学的基础模型奠定基础。

### 翻译

大型语言模型(LLMs)在自然语言处理和生成、计算机视觉以及多模态学习等多个领域展现出强大的生成丰富表示的能力。然而，它们在生物医学数据分析中的应用仍处于初级阶段。单细胞转录组分析对于解析发育和疾病中的细胞亚型多样性至关重要，但稀有亚型对缩放定律提出了挑战。我们提出了一种计算框架，将单细胞RNA测序(scRNA-seq)与大型语言模型相结合，以获取知识引导的基因嵌入。将每个细胞的高表达基因映射到NCBI基因描述，并使用text-embedding-ada-002、BioBERT和SciBERT等模型进行嵌入。应用于在青光眼相关神经变性易感性方面存在差异的视网膜神经节细胞(RGCs)时，该策略改善了亚型分类，突出了生物学特征，并揭示了选择性神经元易感性背后的途径。更广泛地说，它展示了LLM衍生的嵌入如何在数据有限条件下增强生物分析，并为未来单细胞生物学的基础模型奠定基础。


### 论文摘要

Large language models (LLMs) have shown strong ability in generating rich representations across domains such as natural language processing and generation, computer vision, and multimodal learning. However, their application in biomedical data analysis remains nascent. Single-cell transcriptomic profiling is essential for dissecting cell subtype diversity in development and disease, but rare subtypes pose challenges for scaling laws. We present a computational framework that integrates single-cell RNA sequencing (scRNA-seq) with LLMs to derive knowledge-informed gene embeddings. Highly expressed genes for each cell are mapped to NCBI Gene descriptions and embedded using models such as text-embedding-ada-002, BioBERT, and SciBERT. Applied to retinal ganglion cells (RGCs), which differ in vulnerability to glaucoma-related neurodegeneration, this strategy improves subtype classification, highlights biologically significant features, and reveals pathways underlying selective neuronal vulnerability. More broadly, it illustrates how LLM-derived embeddings can augment biological analysis under data-limited conditions and lay the groundwork for future foundation models in single-cell biology.

---

## 98. Network Traffic Classification Using Self-Supervised Learning and Confident Learning

**论文链接:** [http://arxiv.org/abs/2509.23522v1](http://arxiv.org/abs/2509.23522v1)

**作者:** Ehsan Eslami, Walaa Hamouda

**发布时间:** 2025-09-27

### GPT解析

### 总结

本文提出了一种新颖的网络流量分类框架，结合自监督学习和自信学习技术，解决了标记数据有限的问题，同时提高了分类准确率。

### 背景

网络流量分类对网络管理、安全和性能优化至关重要，特别是在5G/6G技术背景下。传统方法如深度包检测和基于端口的识别在加密流量和动态端口分配方面存在困难。监督学习方法需要大量标记数据但难以获取，而无监督方法准确率较低。

### 目的

解决标记数据有限的问题，同时提高网络流量分类的准确率，为网络管理提供有效工具。

### 方法

提出一个两阶段框架：首先利用自监督学习技术（如自编码器或表格对比学习）从大量未标记数据生成伪标签；然后应用流量自适应的自信学习优化这些伪标签，减少噪声影响提高分类精度。

### 主要发现

通过ISCX VPN-nonVPN、自生成数据集和UCDavis-QUIC三个数据集的广泛模拟和评估，证明该方法相比现有技术实现了更高的网络流量分类准确率。

### 结论

该框架是一种通用解决方案，减少了对大量标记数据的需求，同时实现了高准确率的网络流量分类。

### 翻译

网络流量分类(NTC)对于高效网络管理、安全和性能优化至关重要，特别是在5G/6G技术背景下。传统方法如深度包检测(DPI)和基于端口的识别难以应对加密流量增加和动态端口分配的挑战。监督学习方法提供了可行的替代方案，但依赖于大型标记数据集，而网络流量的多样性和大量性使得获取这些数据集很困难。同时，无监督学习方法虽然较少依赖标记数据，但通常表现出较低的准确率。为解决这些限制，我们提出了一种新框架，首先利用自监督学习(SSL)技术（如自编码器或表格对比学习TabCL）从大量未标记数据集中生成伪标签，解决标记数据有限的问题。然后我们应用流量自适应的自信学习(CL)来优化这些伪标签，通过减轻噪声影响提高分类精度。我们提出的框架提供了一种通用解决方案，减少了对大量标记数据的需求，同时提供高准确率。使用三个数据集（ISCX VPN-nonVPN、自生成数据集和UCDavis-QUIC）进行的广泛模拟和评估表明，与最先进技术相比，我们的方法在分类网络流量方面实现了更高的准确率。


### 论文摘要

Network traffic classification (NTC) is vital for efficient network management, security, and performance optimization, particularly with 5G/6G technologies. Traditional methods, such as deep packet inspection (DPI) and port-based identification, struggle with the rise of encrypted traffic and dynamic port allocations. Supervised learning methods provide viable alternatives but rely on large labeled datasets, which are difficult to acquire given the diversity and volume of network traffic. Meanwhile, unsupervised learning methods, while less reliant on labeled data, often exhibit lower accuracy. To address these limitations, we propose a novel framework that first leverages Self-Supervised Learning (SSL) with techniques such as autoencoders or Tabular Contrastive Learning (TabCL) to generate pseudo-labels from extensive unlabeled datasets, addressing the challenge of limited labeled data. We then apply traffic-adopted Confident Learning (CL) to refine these pseudo-labels, enhancing classification precision by mitigating the impact of noise. Our proposed framework offers a generalizable solution that minimizes the need for extensive labeled data while delivering high accuracy. Extensive simulations and evaluations, conducted using three datasets (ISCX VPN-nonVPN, self-generated dataset, and UCDavis--QUIC), and demonstrate that our method achieves superior accuracy compared to state-of-the-art techniques in classifying network traffic.

---

## 99. C3-OWD: A Curriculum Cross-modal Contrastive Learning Framework for Open-World Detection

**论文链接:** [http://arxiv.org/abs/2509.23316v1](http://arxiv.org/abs/2509.23316v1)

**作者:** Siheng Wang, Zhengdao Li, Yanshu Li, Canran Xiao, Haibo Zhan, Zhengtao Yao, Xuzhi Zhang, Jiale Kang, Linshan Li, Weiming Liu, Zhikang Dong, Jifeng Shen, Junhao Dong, Qiang Sun, Piotr Koniusz

**发布时间:** 2025-09-27

### GPT解析

### 总结

本文提出C3-OWD框架，通过课程跨模态对比学习结合RGBT数据预训练和视觉-语言对齐，解决了物体检测中鲁棒性和泛化能力难以兼顾的问题，实验证明该方法在多个评估指标上表现优异。

### 背景

物体检测在封闭集设置中进展显著，但实际应用受限于两个挑战：对未见类别的泛化能力差，以及在不利条件下的鲁棒性不足。先前研究分别探索了可见光-红外检测（提高鲁棒性但缺乏泛化）和开放世界检测（利用视觉-语言对齐实现类别多样性但在极端环境下表现不佳），但难以同时实现鲁棒性和多样性。

### 目的

开发一个统一框架，同时提高物体检测的鲁棒性和泛化能力，解决两者难以兼顾的问题。

### 方法

提出C3-OWD（课程跨模态对比学习框架），分为两个阶段：第一阶段通过RGBT数据预训练增强鲁棒性；第二阶段通过视觉-语言对齐提高泛化能力。引入指数移动平均机制防止两个阶段之间的灾难性遗忘，理论上保证前期性能保留，具有有界的参数滞后和功能一致性。

### 主要发现

在FLIR、OV-COCO和OV-LVIS数据集上的实验表明：C3-OWD在FLIR上达到80.1 AP50，在OV-COCO上达到48.6 AP50Novel，在OV-LVIS上达到35.7 mAPr，在鲁棒性和多样性评估中都建立了具有竞争力的性能。

### 结论

C3-OWD框架成功统一了提高鲁棒性和泛化能力的两种方法，通过课程跨模态对比学习和EMA机制，实现了在两种评估维度上的优异表现，为物体检测在开放世界和不利条件下的应用提供了有效解决方案。

### 翻译

物体检测在封闭集设置中已取得显著进展，但实际部署仍面临两个挑战：对未见类别的泛化能力差以及在不利条件下的鲁棒性不足。先前研究分别探索了这些问题：可见光-红外检测提高了鲁棒性但缺乏泛化能力，而开放世界检测利用视觉-语言对齐策略实现类别多样性但在极端环境下表现不佳。这种权衡使得鲁棒性和多样性难以同时实现。为缓解这些问题，我们提出了C3-OWD，一个统一两种优势的课程跨模态对比学习框架。第一阶段通过RGBT数据预训练增强鲁棒性，第二阶段通过视觉-语言对齐提高泛化能力。为防止两个阶段之间的灾难性遗忘，我们引入了指数移动平均机制，理论上保证了前期性能的保留，具有有界的参数滞后和功能一致性。在FLIR、OV-COCO和OV-LVIS上的实验证明了我们方法的有效性：C3-OWD在FLIR上达到80.1 AP50，在OV-COCO上达到48.6 AP50Novel，在OV-LVIS上达到35.7 mAPr，在鲁棒性和多样性评估中都建立了具有竞争力的性能。代码可在https://github.com/justin-herry/C3-OWD.git获取。


### 论文摘要

Object detection has advanced significantly in the closed-set setting, but real-world deployment remains limited by two challenges: poor generalization to unseen categories and insufficient robustness under adverse conditions. Prior research has explored these issues separately: visible-infrared detection improves robustness but lacks generalization, while open-world detection leverages vision-language alignment strategy for category diversity but struggles under extreme environments. This trade-off leaves robustness and diversity difficult to achieve simultaneously. To mitigate these issues, we propose \textbf{C3-OWD}, a curriculum cross-modal contrastive learning framework that unifies both strengths. Stage~1 enhances robustness by pretraining with RGBT data, while Stage~2 improves generalization via vision-language alignment. To prevent catastrophic forgetting between two stages, we introduce an Exponential Moving Average (EMA) mechanism that theoretically guarantees preservation of pre-stage performance with bounded parameter lag and function consistency. Experiments on FLIR, OV-COCO, and OV-LVIS demonstrate the effectiveness of our approach: C3-OWD achieves $80.1$ AP$^{50}$ on FLIR, $48.6$ AP$^{50}_{\text{Novel}}$ on OV-COCO, and $35.7$ mAP$_r$ on OV-LVIS, establishing competitive performance across both robustness and diversity evaluations. Code available at: https://github.com/justin-herry/C3-OWD.git.

---

## 100. Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives

**论文链接:** [http://arxiv.org/abs/2509.25094v1](http://arxiv.org/abs/2509.25094v1)

**作者:** AmirHossein Zamani, Bruno Roy, Arianna Rampini

**发布时间:** 2025-09-29

### GPT解析

### 总结

该论文提出了一种无监督的可微分框架，用于自动化3D网格的UV参数化过程，解决了现有方法中语义感知和可见性感知的不足。

### 背景

近期3D生成模型能生成高质量纹理，但通常需要手动UV映射作为输入，这是一个耗时且需要专业技能的任务，成为3D内容创作的主要瓶颈。现有自动方法忽略了语义感知（UV图表应对齐语义相似的3D部分）和可见性感知（切割接缝应位于不易被看到的区域）这两个重要标准。

### 目的

克服现有方法的缺点，自动化网格参数化过程，提出一种同时考虑语义和可见性感知的无监督可微分UV学习框架。

### 方法

对于语义感知：将网格分割为语义3D部分，应用无监督学习的按部分UV参数化主干，将各部分图表聚合为统一UV图谱。对于可见性感知：使用环境光遮挡(AO)作为曝光代理，反向传播软可微分AO加权接缝目标，将切割接缝引导到遮挡区域。

### 主要发现

通过与最先进方法的定性和定量评估，表明所提出方法生成的UV图谱能更好地支持纹理生成，并减少可感知的接缝伪影。

### 结论

该提出的UV参数化方法在纹理生成质量和接缝减少方面优于现有基线，实现代码已在GitHub上公开。

### 翻译

近期的3D生成模型为3D网格对象产生高质量纹理。然而，它们通常依赖于一个重要假设，即输入的3D网格伴随着手动网格参数化（UV映射），这是一个需要技术精度和艺术判断的手动任务。行业调查显示，这个过程通常占资产创建的重要部分，成为3D内容创作者的主要瓶颈。此外，现有的自动方法通常忽略两个感知上重要的标准：(1)语义感知（UV图表应跨形状对齐语义相似的3D部分）和(2)可见性感知（切割接缝应位于不太可能被看到的区域）。为了克服这些缺点并自动化网格参数化过程，我们提出了一种无监督的可微分框架，通过增加语义和可见性感知目标来增强标准几何保持的UV学习。对于语义感知，我们的流程(i)将网格分割为语义3D部分，(ii)应用无监督学习的按部分UV参数化主干，以及(iii)将各部分图表聚合为统一的UV图谱。对于可见性感知，我们使用环境光遮挡(AO)作为曝光代理，并反向传播软可微分AO加权接缝目标，将切割接缝引导到遮挡区域。通过与最先进方法进行定性和定量评估，我们表明所提出的方法产生的UV图谱能更好地支持纹理生成，并减少与最近基线相比的可感知接缝伪影。我们的实现代码已在GitHub上公开：https://github.com/AHHHZ975/Semantic-Visibility-UV-Param。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D网格参数化（UV映射）的自动化问题，同时考虑语义感知和可见性感知两个重要标准。这个问题在现实中很重要，因为行业调查显示手动UV映射占资产创建的重要部分，成为3D内容创作的主要瓶颈；自动化此流程可提高创作效率，而考虑语义和可见性能生成更高质量的UV贴图，支持更好的纹理生成并减少可见的接缝伪影。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将问题分解为基础几何保持学习和两个感知目标（语义感知和可见性感知）的学习。语义感知方面，先分割网格成语义部分，再独立参数化每部分最后聚合；可见性感知方面，使用环境光遮蔽(AO)作为曝光代理，引导接缝向遮挡区域。作者确实借鉴了现有工作：基础UV参数化学习采用双向循环映射架构，3D分割使用形状直径函数(ShDF)，可见性感知使用环境光遮蔽(AO)作为代理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一个无监督可微框架，增强标准几何保持UV学习，同时加入语义感知和可见性感知目标。语义感知确保UV图表与有意义的3D表面部分对齐，可见性感知将接缝放置在不易被观察区域。整体流程分两阶段：第一阶段使用基于MLP的网络进行几何保持学习；第二阶段分为语义感知流程（3D分割→每部分参数化→图集聚合）和可见性感知流程（计算AO值→应用主干UV参数化→提取边界点→最小化AO加权平均值）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 无监督可微框架同时优化几何、语义和可见性目标；2) 语义感知目标引入分割-参数化策略；3) 可见性感知目标使用AO引导接缝 placement；4) 基于四个子网络的双向循环映射架构。相比之前工作，不同之处在于：1) 首个同时考虑语义和可见性的方法；2) 完全无监督，不需人工标注；3) 端到端可训练；4) 生成UV贴图更好支持纹理生成并减少接缝伪影。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种无监督表示学习方法，通过引入语义感知和可见性感知目标，自动化了3D网格的UV参数化过程，生成更高质量的UV贴图以支持纹理生成并减少可见接缝伪影。'}


### 论文摘要

Recent 3D generative models produce high-quality textures for 3D mesh objects. However, they commonly rely on the heavy assumption that input 3D meshes are accompanied by manual mesh parameterization (UV mapping), a manual task that requires both technical precision and artistic judgment. Industry surveys show that this process often accounts for a significant share of asset creation, creating a major bottleneck for 3D content creators. Moreover, existing automatic methods often ignore two perceptually important criteria: (1) semantic awareness (UV charts should align semantically similar 3D parts across shapes) and (2) visibility awareness (cutting seams should lie in regions unlikely to be seen). To overcome these shortcomings and to automate the mesh parameterization process, we present an unsupervised differentiable framework that augments standard geometry-preserving UV learning with semantic- and visibility-aware objectives. For semantic-awareness, our pipeline (i) segments the mesh into semantic 3D parts, (ii) applies an unsupervised learned per-part UV-parameterization backbone, and (iii) aggregates per-part charts into a unified UV atlas. For visibility-awareness, we use ambient occlusion (AO) as an exposure proxy and back-propagate a soft differentiable AO-weighted seam objective to steer cutting seams toward occluded regions. By conducting qualitative and quantitative evaluations against state-of-the-art methods, we show that the proposed method produces UV atlases that better support texture generation and reduce perceptible seam artifacts compared to recent baselines. Our implementation code is publicly available at: https://github.com/AHHHZ975/Semantic-Visibility-UV-Param.

---

## 101. Unsupervised Learning in a General Semiparametric Clusterwise Index Distribution Model

**论文链接:** [http://arxiv.org/abs/2509.24987v1](http://arxiv.org/abs/2509.24987v1)

**作者:** Jen-Chieh Teng, Chin-Tsang Chiang

**发布时间:** 2025-09-29

**备注:** 66 pages, 17 figures

### GPT解析

### 总结

研究提出了一种半参数聚类指数分布模型，分析潜在聚类对协变量-响应关系的影响，开发了参数估计方法、收敛算法和信息标准选择聚类数量，并通过模拟和实证研究验证了方法的有效性。

### 背景

现有方法可能缺乏对潜在聚类如何影响协变量-响应关系的有效分析

### 目的

开发一种能够分析潜在聚类对协变量-响应关系影响的半参数聚类指数分布模型，并提供参数估计和聚类数量选择的方法

### 方法

使用充分降维考虑协变量对聚类变量的影响；基于模型逐对象表示，提出分离惩罚估计方法；开发收敛算法和启发式初始化；使用分区估计器拟合聚类成员模型并构建最优分类规则；开发两种一致的半参数信息标准选择聚类数量

### 主要发现

估计的聚类结构是一致且最优的；参数估计器具有oracle性质；所提出的方法在模拟研究和实证数据分析中表现出色

### 结论

提出的半参数聚类指数分布模型能够有效分析潜在聚类对协变量-响应关系的影响，并通过多种创新方法确保了估计的准确性和聚类选择的合理性

### 翻译

这项研究引入了一种通用的半参数聚类指数分布模型，用于分析潜在聚类如何影响协变量-响应关系。通过采用充分降维来考虑协变量对聚类变量的影响，我们开发了一种估计模型参数的独特方法。基于模型的逐对象表示，所提出的分离惩罚估计方法对个体进行分区并估计聚类指数系数。我们为该估计过程提出了一个收敛算法，并采用启发式初始化来加速优化。得到的分区估计器随后用于拟合聚类成员模型并构建最优分类规则，这两个过程迭代更新分区和参数估计器。我们方法的另一个关键贡献是开发了两种一致的半参数信息标准，用于选择聚类数量。根据监督学习中分类和估计的原则，估计的聚类结构是一致且最优的，参数估计器具有oracle性质。全面的模拟研究和实证数据分析说明了所提出方法的有效性。


### 论文摘要

This study introduces a general semiparametric clusterwise index distribution model to analyze how latent clusters affect the covariate-response relationships. By employing sufficient dimension reduction to account for the effects of covariates on the cluster variable, we develop a distinct method for estimating model parameters. Building on a subjectwise representation of the underlying model, the proposed separation penalty estimation method partitions individuals and estimates cluster index coefficients. We propose a convergent algorithm for this estimation procedure and incorporate a heuristic initialization to expedite optimization. The resulting partition estimator is subsequently used to fit the cluster membership model and to construct an optimal classification rule, with both procedures iteratively updating the partition and parameter estimators. Another key contribution of our method is the development of two consistent semiparametric information criteria for selecting the number of clusters. In line with principles of classification and estimation in supervised learning, the estimated cluster structure is consistent and optimal, and the parameter estimators possess the oracle property. Comprehensive simulation studies and empirical data analyses illustrate the effectiveness of the proposed methodology.

---

## 102. Learning Distinguishable Representations in Deep Q-Networks for Linear Transfer

**论文链接:** [http://arxiv.org/abs/2509.24947v1](http://arxiv.org/abs/2509.24947v1)

**作者:** Sooraj Sathish, Keshav Goyal, Raghuram Bharadwaj Diddigi

**发布时间:** 2025-09-29

### GPT解析

### 总结

本研究提出了一种新的深度Q学习方法，通过引入正则化项减少状态特征表示之间的正相关，从而改善迁移学习性能并降低计算开销。

### 背景

深度强化学习已成功解决复杂顺序决策问题，但训练面临超参数调整需求大和高计算成本等挑战。迁移学习作为有前景的策略，可重用先前任务知识避免完全重新训练模型。

### 目的

研究深度RL模型学习到的内部表示（特别是最后一隐藏层的激活）是否可用作训练更简单模型（如线性函数逼近器）在新任务上的输入，并解决表示高度相关的问题。

### 方法

提出一种新的深度Q学习方法，引入正则化项减少状态特征表示之间的正相关，使线性函数逼近器在迁移学习中更有效使用，并在标准RL基准测试和MinAtar游戏上进行实验验证。

### 主要发现

标准深度RL模型学习到的表示可能高度相关，限制了它们与线性函数逼近一起使用时的有效性；通过减少特征相关性可以改善迁移学习性能并降低计算开销。

### 结论

所提出的方法通过减少特征表示之间的相关性，改善了迁移学习性能，使深度RL模型在迁移学习场景中更加实用高效。

### 翻译

深度强化学习(DRL)已通过将神经网络与强化学习框架相结合，展示了在解决复杂顺序决策问题方面的成功。然而，训练深度RL模型带来了几个挑战，如需要广泛的超参数调整和高计算成本。迁移学习已成为一种有前景的策略，通过重用先前学习任务的知识来应对新的相关任务，从而避免了完全从头开始重新训练模型的必要性。在RL中，迁移学习的常用方法是利用神经网络在训练期间学习到的内部表示。具体而言，来自最后一个隐藏层的激活可以被看作是精炼的状态表示，封装了输入的基本特征。在这项工作中，我们研究这些表示是否可以用作训练更简单模型（如线性函数逼近器）在新任务上的输入。我们观察到，标准深度RL模型学习到的表示可能高度相关，这限制了它们与线性函数逼近一起使用时的有效性。为了缓解这个问题，我们提出了一种新的深度Q学习方法，引入了一个正则化项来减少状态特征表示之间的正相关。通过利用这些相关性降低的特征，我们使线性函数逼近器在迁移学习中能够更有效地使用。通过在标准RL基准测试和MinAtar游戏上进行实验和消融研究，我们证明了该方法在改善迁移学习性能并减少计算开销方面的有效性。


### 论文摘要

Deep Reinforcement Learning (RL) has demonstrated success in solving complex sequential decision-making problems by integrating neural networks with the RL framework. However, training deep RL models poses several challenges, such as the need for extensive hyperparameter tuning and high computational costs. Transfer learning has emerged as a promising strategy to address these challenges by enabling the reuse of knowledge from previously learned tasks for new, related tasks. This avoids the need for retraining models entirely from scratch. A commonly used approach for transfer learning in RL is to leverage the internal representations learned by the neural network during training. Specifically, the activations from the last hidden layer can be viewed as refined state representations that encapsulate the essential features of the input. In this work, we investigate whether these representations can be used as input for training simpler models, such as linear function approximators, on new tasks. We observe that the representations learned by standard deep RL models can be highly correlated, which limits their effectiveness when used with linear function approximation. To mitigate this problem, we propose a novel deep Q-learning approach that introduces a regularization term to reduce positive correlations between feature representation of states. By leveraging these reduced correlated features, we enable more effective use of linear function approximation in transfer learning. Through experiments and ablation studies on standard RL benchmarks and MinAtar games, we demonstrate the efficacy of our approach in improving transfer learning performance and thereby reducing computational overhead.

---

## 103. PredNext: Explicit Cross-View Temporal Prediction for Unsupervised Learning in Spiking Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.24844v1](http://arxiv.org/abs/2509.24844v1)

**作者:** Yiting Dong, Jianhao Ding, Zijie Xu, Tong Bu, Zhaofei Yu, Tiejun Huang

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了PredNext方法，通过跨视图未来步骤预测和片段预测来建模时间关系，解决了当前无监督脉冲神经网络(SNNs)在处理长程时间依赖性和保持时间特征一致性方面的局限性，为大规模时间视频数据上的无监督深度SNNs提供了有效基础。

### 背景

脉冲神经网络(SNNs)具有时间处理能力和生物学 plausible 动态，为无监督表征学习提供了自然平台。然而，当前无监督SNNs主要采用浅层架构或局部可塑性规则，限制了它们建模长程时间依赖性和保持时间特征一致性的能力，导致语义不稳定的表征，阻碍了深度无监督SNNs在大规模时间视频数据上的发展。

### 目的

解决当前无监督SNNs的局限性，建立SNN自监督学习的标准基准，并开发一种能有效处理大规模时间视频数据的深度无监督SNN方法。

### 方法

提出了PredNext方法，通过跨视图未来步骤预测和片段预测明确建模时间关系。这个即插即用模块可以无缝集成到各种自监督目标中。研究在UCF101、HMDB51和MiniKinetics上建立了SNN自监督学习的标准基准，这些数据集比传统的DVS数据集大得多。

### 主要发现

PredNext在不同任务和自监督方法上都提供了显著的性能提升；仅在UCF101上进行无监督训练，PredNext就实现了与ImageNet预训练监督权重相当的性能；与强制一致性约束不同，PredNext显著提高了时间特征一致性，同时增强了网络的泛化能力。

### 结论

这项工作为大规模时间视频数据上的无监督深度SNNs提供了有效的基础。

### 翻译

脉冲神经网络(SNNs)凭借其时间处理能力和生物学 plausible 动态，为无监督表征学习提供了自然平台。然而，当前的无监督SNNs主要采用浅层架构或局部可塑性规则，限制了它们建模长程时间依赖性和保持时间特征一致性的能力。这导致语义不稳定的表征，从而阻碍了深度无监督SNNs在大规模时间视频数据上的发展。我们提出了PredNext，它通过跨视图未来步骤预测和片段预测明确建模时间关系。这个即插即用模块可以无缝集成到各种自监督目标中。我们首先在UCF101、HMDB51和MiniKinetics上建立了SNN自监督学习的标准基准，这些数据集比传统的DVS数据集大得多。PredNext在不同任务和自监督方法上都提供了显著的性能提升。PredNext仅通过在UCF101上进行无监督训练，就实现了与ImageNet预训练监督权重相当的性能。额外的实验表明，PredNext不同于强制一致性约束，在显著提高时间特征一致性的同时，增强了网络的泛化能力。这项工作为大规模时间视频数据上的无监督深度SNNs提供了有效的基础。


### 论文摘要

Spiking Neural Networks (SNNs), with their temporal processing capabilities and biologically plausible dynamics, offer a natural platform for unsupervised representation learning. However, current unsupervised SNNs predominantly employ shallow architectures or localized plasticity rules, limiting their ability to model long-range temporal dependencies and maintain temporal feature consistency. This results in semantically unstable representations, thereby impeding the development of deep unsupervised SNNs for large-scale temporal video data. We propose PredNext, which explicitly models temporal relationships through cross-view future Step Prediction and Clip Prediction. This plug-and-play module seamlessly integrates with diverse self-supervised objectives. We firstly establish standard benchmarks for SNN self-supervised learning on UCF101, HMDB51, and MiniKinetics, which are substantially larger than conventional DVS datasets. PredNext delivers significant performance improvements across different tasks and self-supervised methods. PredNext achieves performance comparable to ImageNet-pretrained supervised weights through unsupervised training solely on UCF101. Additional experiments demonstrate that PredNext, distinct from forced consistency constraints, substantially improves temporal feature consistency while enhancing network generalization capabilities. This work provides a effective foundation for unsupervised deep SNNs on large-scale temporal video data.

---

## 104. Discovering "Words" in Music: Unsupervised Learning of Compositional Sparse Code for Symbolic Music

**论文链接:** [http://arxiv.org/abs/2509.24603v1](http://arxiv.org/abs/2509.24603v1)

**作者:** Tianle Wang, Sirui Zhang, Xinyi Tong, Peiyang Yu, Jishang Chen, Liangke Zhao, Xinpu Gao, Yves Zhu, Tiezheng Ge, Bo Zheng, Duo Xu, Yang Liu, Xin Jin, Feng Yu, Songchun Zhu

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种无监督机器学习算法，用于从符号音乐数据中识别重复出现的模式，称为'音乐词'。这些模式是音乐结构的基础，反映了作曲过程中涉及的认知过程。

### 背景

提取这些模式具有挑战性，因为音乐解释中存在固有的语义模糊性。

### 目的

将音乐词发现任务制定为统计优化问题，并提出一个两阶段的基于期望最大化(EM)的学习框架。

### 方法

1. 开发音乐词词典；2. 重建音乐数据；3. 使用最小化代码长度来处理语义模糊性。

### 主要发现

算法在人类专家注释评估中达到了0.61的交并比(IoU)分数。最小化代码长度可以有效解决语义模糊性问题，表明人类对编码系统的优化塑造了音乐语义。

### 结论

这种方法使计算机能够从音乐数据中提取'基本构建块'，促进结构分析和稀疏编码。该方法有两个主要应用：在AI音乐中支持下游任务，以及在音乐学中提供分析作曲模式的工具。

### 翻译

本文提出了一种无监督机器学习算法，用于从符号音乐数据中识别重复出现的模式，称为'音乐词'。这些模式是音乐结构的基础，反映了作曲过程中涉及的认知过程。然而，由于音乐解释中固有的语义模糊性，提取这些模式仍然具有挑战性。我们将音乐词发现任务制定为统计优化问题，并提出一个两阶段的基于期望最大化(EM)的学习框架：1. 开发音乐词词典；2. 重建音乐数据。在与人类专家注释的评估中，该算法达到了0.61的交并比(IoU)分数。我们的研究结果表明，最小化代码长度可以有效解决语义模糊性问题，表明人类对编码系统的优化塑造了音乐语义。这种方法使计算机能够从音乐数据中提取'基本构建块'，促进结构分析和稀疏编码。该方法有两个主要应用。首先，在AI音乐中，它支持音乐生成、分类、风格迁移和即兴演奏等下游任务。其次，在音乐学中，它为分析作曲模式提供了工具，并深入了解不同音乐风格和作曲家中的最小编码原则。


### 论文摘要

This paper presents an unsupervised machine learning algorithm that identifies recurring patterns -- referred to as ``music-words'' -- from symbolic music data. These patterns are fundamental to musical structure and reflect the cognitive processes involved in composition. However, extracting these patterns remains challenging because of the inherent semantic ambiguity in musical interpretation. We formulate the task of music-word discovery as a statistical optimization problem and propose a two-stage Expectation-Maximization (EM)-based learning framework: 1. Developing a music-word dictionary; 2. Reconstructing the music data. When evaluated against human expert annotations, the algorithm achieved an Intersection over Union (IoU) score of 0.61. Our findings indicate that minimizing code length effectively addresses semantic ambiguity, suggesting that human optimization of encoding systems shapes musical semantics. This approach enables computers to extract ``basic building blocks'' from music data, facilitating structural analysis and sparse encoding. The method has two primary applications. First, in AI music, it supports downstream tasks such as music generation, classification, style transfer, and improvisation. Second, in musicology, it provides a tool for analyzing compositional patterns and offers insights into the principle of minimal encoding across diverse musical styles and composers.

---

## 105. An Efficient Transfer Learning Method Based on Adapter with Local Attributes for Speech Emotion Recognition

**论文链接:** [http://arxiv.org/abs/2509.23795v1](http://arxiv.org/abs/2509.23795v1)

**作者:** Haoyu Song, Ian McLoughlin, Qing Gu, Nan Jiang, Yan Song

**发布时间:** 2025-09-28

### GPT解析

### 总结

该论文提出了一种创新的适配器方法，结合局部属性学习，用于语音情感识别的迁移学习，通过WAP-Transformer主干、教师-学生分支结构和SAP池化模块，实现了在资源受限条件下的高效情感识别。

### 背景

现有的语音情感识别方法通常缺乏高质量的大规模语料库，部分原因是情感具有复杂的心理特性，使得准确标注既困难又耗时。最近，基于迁移学习的方法利用在大型语音语料库上预训练的编码器在下游SER任务中显示出强大潜力，但针对不同场景的任务特定微调仍然必要，且需要昂贵的编码器重新训练。

### 目的

解决现有方法需要为每个SER任务进行昂贵编码器重新训练的问题，提出一种带有局部属性的适配器用于高效的迁移学习。

### 方法

提出加权平均池化-Transformer（WAP-Transformer）作为轻量级主干网络来丰富帧级表示；利用带有教师-学生分支的适配器进行任务无关的迁移学习，其中学生分支通过掩码预测和自蒸馏目标联合优化，教师分支通过指数移动平均从学生在线获取；通过无监督聚类从教师分支学习局部属性，作为提供丰富语义监督的通用模型；提出统计注意力池化（SAP）模块获取用于微调的话语表示。

### 主要发现

在IEMOCAP数据集上进行的广泛实验表明，所提出的带有局部属性的适配器方法与之前在类似设置中的最先进方法相比，实现了优越的性能。

### 结论

提出的带有局部属性的适配器方法在语音情感识别任务中表现优异，能够有效利用预训练模型，同时针对特定任务进行高效微调。

### 翻译

现有的语音情感识别方法通常缺乏高质量的大规模语料库，部分原因是情感具有复杂的心理特性，使得准确标注既困难又耗时。最近，基于迁移学习的方法利用在大型语音语料库上预训练的编码器（如Wav2Vec2.0和HuBERT）在下游SER任务中显示出强大潜力。然而，针对不同主题、说话者和语言的各种对话场景，任务特定的微调仍然是必要的，以获得令人满意的性能。这通常需要为每个SER任务进行昂贵的编码器重新训练。为了解决这个问题，我们提出训练带有局部属性的适配器用于高效的迁移学习。具体来说，提出加权平均池化-Transformer（WAP-Transformer）作为轻量级主干来丰富帧级表示。利用带有教师-学生分支的适配器进行任务无关的迁移学习，其中学生分支通过掩码预测和自蒸馏目标联合优化，教师分支通过指数移动平均从学生在线获取。同时，通过无监督聚类从教师分支学习局部属性，旨在作为提供额外丰富语义监督的通用模型。提出统计注意力池化（SAP）模块来获取用于微调的话语表示。为了评估所提出的带有局部属性的适配器的有效性，在IEMOCAP上进行了广泛的实验。与之前在类似设置中的最先进方法相比，报告了优越的性能。


### 论文摘要

Existing speech emotion recognition (SER) methods commonly suffer from the lack of high-quality large-scale corpus, partly due to the complex, psychological nature of emotion which makes accurate labeling difficult and time consuming. Recently, transfer learning based methods that exploit the encoders pretrained on large-scale speech corpus (e.g., Wav2Vec2.0 and HuBERT) have shown strong potential for downstream SER tasks. However, task-specific fine-tuning remains necessary for various conversational scenarios of different topics, speakers and languages to achieve satisfactory performance. It generally requires costly encoder retraining for individual SER tasks. To address this issue, we propose to train an adapter with local attributes for efficient transfer learning. Specifically, a weighted average pooling-Transformer (WAP-Transformer) is proposed as a lightweight backbone to enrich the frame-level representation. An adapter with teacher-student branches is exploited for task-agnostic transfer learning, where the student branch is jointly optimized via mask prediction and self-distillation objectives, and the teacher branch is obtained online from the student via exponential moving average (EMA). Meanwhile, local attributes are learned from the teacher branch via unsupervised clustering, which aims to act as a universal model that provides additional semantic-rich supervisions. A statistical attentive pooling (SAP) module is proposed to obtain utterance representation for fine-tuning. To evaluate the effectiveness of the proposed adapter with local attributes, extensive experiments have been conducted on IEMOCAP. Superior performance has been reported, compared to the previous state-of-the-art methods in similar settings.

---

## 106. Satellite: Detecting and Analyzing Smart Contract Vulnerabilities caused by Subcontract Misuse

**论文链接:** [http://arxiv.org/abs/2509.23679v1](http://arxiv.org/abs/2509.23679v1)

**作者:** Zeqin Liao, Yuhong Nan, Zixu Gao, Henglong Liang, Sicheng Hao, Jiajing Wu, Zibin Zheng

**发布时间:** 2025-09-28

**备注:** This is the author version of the article accepted for publication in  IEEE Transactions on Software Engineering. The final version is available at  10.1109/TSE.2025.3613470

### GPT解析

### 总结

本文提出了Satellite框架，一个用于检测智能合约中子合约误用漏洞(SMV)的字节码级别静态分析工具，通过迁移学习、方法级特征提取和SMV指标总结实现了高精度的漏洞检测。

### 背景

智能合约开发者普遍重用子合约以提高开发效率，但这种重用可能意外引入漏洞。智能合约通常被编译为字节码，其类级信息和语义在编译后完全被掩盖，使得自动检测问题面临独特挑战。

### 目的

开发一个字节码级别的静态分析框架，用于检测智能合约中的子合约误用漏洞(SMV)。

### 方法

Satellite框架采用三种创新方法：1)利用迁移学习方法恢复继承方法；2)提取细粒度方法级特征并进行方法级比较，识别子合约重用部分；3)根据漏洞类型总结SMV指标以有效识别漏洞。

### 主要发现

Satellite在识别SMV方面表现出色，精确率达84.68%，召回率达92.11。在10,011个真实世界智能合约中发现了14个新的/未知SMV，影响了价值201,358美元的数字资产。

### 结论

Satellite框架能够有效检测智能合约中的子合约误用漏洞，对智能合约安全具有重要的实际应用价值。

### 翻译

智能合约开发者普遍重用子合约以提高开发效率。与任何编程语言一样，这种子合约重用可能会意外地将漏洞包含或引入到终端智能合约中。不幸的是，自动检测此类问题面临几个独特的挑战。特别是在大多数情况下，智能合约被编译为字节码，其类级信息（例如继承、虚函数表）甚至语义（例如控制流和数据流）在编译后完全被掩盖，作为一个单一的智能合约。在本文中，我们提出了Satellite，这是一个新的字节码级别静态分析框架，用于检测智能合约中的子合约误用漏洞(SMV)。Satellite采用了一系列新颖设计来提高其整体有效性。特别是，Satellite利用迁移学习方法来恢复继承方法，这对于识别智能合约中的子合约重用至关重要。此外，Satellite提取了一组细粒度的方法级特征并进行方法级比较，以识别智能合约中子合约的重用部分。最后，Satellite根据漏洞类型总结了一组SMV指标，从而有效识别SMV。为了评估Satellite，我们构建了一个包含58个来自真实世界攻击的SMV数据集，并从SOTA研究中收集了额外的56个SMV模式。实验结果表明，Satellite在识别SMV方面表现出良好性能，精确率为84.68%，召回率为92.11%。此外，Satellite在10,011个真实世界智能合约中成功识别出14个新的/未知SMV，影响了总价值为201,358美元的数字资产。


### 论文摘要

Developers of smart contracts pervasively reuse subcontracts to improve development efficiency. Like any program language, such subcontract reuse may unexpectedly include, or introduce vulnerabilities to the end-point smart contract. Unfortunately, automatically detecting such issues poses several unique challenges. Particularly, in most cases, smart contracts are compiled as bytecode, whose class-level information (e.g., inheritance, virtual function table), and even semantics (e.g., control flow and data flow) are fully obscured as a single smart contract after compilation.   In this paper, we propose Satellite, a new bytecode-level static analysis framework for subcontract misuse vulnerability (SMV) detection in smart contracts. Satellite incorporates a series of novel designs to enhance its overall effectiveness.. Particularly, Satellite utilizes a transfer learning method to recover the inherited methods, which are critical for identifying subcontract reuse in smart contracts. Further, Satellite extracts a set of fine-grained method-level features and performs a method-level comparison, for identifying the reuse part of subcontract in smart contracts. Finally, Satellite summarizes a set of SMV indicators according to their types, and hence effectively identifies SMVs. To evaluate Satellite, we construct a dataset consisting of 58 SMVs derived from real-world attacks and collect additional 56 SMV patterns from SOTA studies. Experiment results indicate that Satellite exhibits good performance in identifying SMV, with a precision rate of 84.68% and a recall rate of 92.11%. In addition, Satellite successfully identifies 14 new/unknown SMV over 10,011 real-world smart contracts, affecting a total amount of digital assets worth 201,358 USD.

---

## 107. Transfer Learning and Machine Learning for Training Five Year Survival Prognostic Models in Early Breast Cancer

**论文链接:** [http://arxiv.org/abs/2509.23268v1](http://arxiv.org/abs/2509.23268v1)

**作者:** Lisa Pilgram, Kai Yang, Ana-Alicia Beltran-Bless, Gregory R. Pond, Lisa Vandermeer, John Hilton, Marie-France Savard, Andréanne Leblanc, Lois Sheperd, Bingshu E. Chen, John M. S. Bartlett, Karen J. Taylor, Jane Bayani, Sarah L. Barker, Melanie Spears, Cornelis J. H. van der Velde, Elma Meershoek-Klein Kranenbarg, Luc Dirix, Elizabeth Mallon, Annette Hasenburg, Christos Markopoulos, Lamin Juwara, Fida K. Dankar, Mark Clemons, Khaled El Emam

**发布时间:** 2025-09-27

### GPT解析

### 总结

该研究评估了机器学习、迁移学习和集成集成方法改善乳腺癌生存预后的潜力，发现在信息缺失或数据集变化的情况下，这些方法比传统预后工具PREDICT v3表现更好。

### 背景

预后信息对乳腺癌管理中的决策制定至关重要。最近研究主要集中在基因组预后工具上，尽管临床病理预后方法成本更低且更易于获取。机器学习、迁移学习和集成集成方法为构建稳健的预后框架提供了机会。

### 目的

评估机器学习、迁移学习和集成集成方法改善乳腺癌生存预后的潜力，通过比较从头训练的机器学习模型、从预训练预后工具迁移学习以及集成集成方法。

### 方法

使用MA.27试验数据进行模型训练，并在TEAM试验和SEER队列上进行外部验证。迁移学习通过微调预训练的预后工具PREDICT v3实现，从头训练的机器学习包括随机生存森林和极端梯度提升，集成集成通过模型预测的加权和实现。

### 主要发现

迁移学习、随机生存森林和集成集成方法提高了校准性(ICI从0.042降低到≤0.007)，同时保持相当的判别能力(AUC从0.738增加到0.744-0.799)；由于信息缺失，PREDICT v3在23.8-25.8%的案例中预测无效，而机器学习方法可处理缺失信息；患者年龄、淋巴结状态、病理分级和肿瘤大小是最重要的预后因素；SEER队列的外部验证证实了这些方法的益处。

### 结论

迁移学习、随机生存森林和集成集成方法可以在PREDICT v3相关信息缺失或可能发生数据集变化的情况下改善预后预测。

### 翻译

预后信息对乳腺癌管理中的决策制定至关重要。最近的研究主要集中在基因组预后工具上，尽管临床病理预后方法成本更低且更易于获取。机器学习、迁移学习和集成集成方法为构建稳健的预后框架提供了机会。我们通过比较从头训练的机器学习模型、从预训练预后工具迁移学习以及集成集成方法，评估了改善乳腺癌生存预后的潜力。使用MA.27试验数据进行模型训练，并在TEAM试验和SEER队列上进行外部验证。迁移学习通过微调预训练的预后工具PREDICT v3实现，从头训练的机器学习包括随机生存森林和极端梯度提升，集成集成通过模型预测的加权和实现。迁移学习、从头训练的随机生存森林和集成集成方法在MA.27中的校准性优于预训练模型，而判别能力保持相当。由于信息缺失，在相当比例的个体中观察到无效的PREDICT v3预测，而机器学习模型可以处理缺失信息。在所有模型中，患者年龄、淋巴结状态、病理分级和肿瘤大小是最重要的预后因素。外部验证确认了这些方法的益处。这项研究表明，在相关信息缺失或可能发生数据集变化的情况下，这些先进方法可以改善预后预测。


### 论文摘要

Prognostic information is essential for decision-making in breast cancer management. Recently trials have predominantly focused on genomic prognostication tools, even though clinicopathological prognostication is less costly and more widely accessible. Machine learning (ML), transfer learning and ensemble integration offer opportunities to build robust prognostication frameworks. We evaluate this potential to improve survival prognostication in breast cancer by comparing de-novo ML, transfer learning from a pre-trained prognostic tool and ensemble integration. Data from the MA.27 trial was used for model training, with external validation on the TEAM trial and a SEER cohort. Transfer learning was applied by fine-tuning the pre-trained prognostic tool PREDICT v3, de-novo ML included Random Survival Forests and Extreme Gradient Boosting, and ensemble integration was realized through a weighted sum of model predictions. Transfer learning, de-novo RSF, and ensemble integration improved calibration in MA.27 over the pre-trained model (ICI reduced from 0.042 in PREDICT v3 to <=0.007) while discrimination remained comparable (AUC increased from 0.738 in PREDICT v3 to 0.744-0.799). Invalid PREDICT v3 predictions were observed in 23.8-25.8% of MA.27 individuals due to missing information. In contrast, ML models and ensemble integration could predict survival regardless of missing information. Across all models, patient age, nodal status, pathological grading and tumor size had the highest SHAP values, indicating their importance for survival prognostication. External validation in SEER, but not in TEAM, confirmed the benefits of transfer learning, RSF and ensemble integration. This study demonstrates that transfer learning, de-novo RSF, and ensemble integration can improve prognostication in situations where relevant information for PREDICT v3 is lacking or where a dataset shift is likely.

---

## 108. Brain Tumor Classification from MRI Scans via Transfer Learning and Enhanced Feature Representation

**论文链接:** [http://arxiv.org/abs/2509.22956v1](http://arxiv.org/abs/2509.22956v1)

**作者:** Ahta-Shamul Hoque Emran, Hafija Akter, Abdullah Al Shiam, Abu Saleh Musa Miah, Anichur Rahman, Fahmid Al Farid, Hezerul Abdul Karim

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究提出了一种基于深度学习的自动脑肿瘤检测框架，并创建了新的脑肿瘤MRI数据集，为脑肿瘤的早期检测提供了有效工具。

### 背景

脑肿瘤是中枢神经系统中的异常细胞生长，及时检测对改善患者预后至关重要。然而，可靠的脑肿瘤MRI资源缺乏，且类别不平衡问题影响深度学习研究。

### 目的

开发一个自动高效的深度学习框架，用于从MRI扫描中检测脑肿瘤，并创建一个可靠的脑肿瘤MRI数据集以解决资源缺乏问题。

### 方法

采用预训练的ResNet50模型进行特征提取，通过全局平均池化和线性投影获得高级图像表示；引入新型的Dense-Dropout序列增强非线性特征学习；创建了包含209名受试者MRI扫描的MMCBT数据集，并对肿瘤类别进行数据增强以解决类别不平衡问题。

### 主要发现

1. 提出的Dense-Dropout序列能有效增强非线性特征学习，减少过拟合，提高模型鲁棒性；2. 创建的MMCBT数据集包含3671张肿瘤图像和13273张非肿瘤图像，经过平衡处理，适合深度学习研究。

### 结论

该研究提出的深度学习框架和新创建的数据集为脑肿瘤的自动检测提供了有效解决方案，有助于提高脑肿瘤的早期检测率和患者预后。

### 翻译

脑肿瘤是中枢神经系统中的异常细胞生长，其及时检测对改善患者预后至关重要。本文提出了一种从磁共振成像扫描中自动高效检测脑肿瘤的深度学习框架。该框架采用预训练的ResNet50模型进行特征提取，随后通过全局平均池化和线性投影获得紧凑的高级图像表示。这些特征随后通过一种新型的Dense-Dropout序列处理，这是本工作的核心贡献，通过多样化的特征转换增强了非线性特征学习，减少了过拟合，并提高了鲁棒性。另一主要贡献是创建了Mymensingh医学院脑肿瘤数据集，旨在解决可靠的脑肿瘤MRI资源缺乏的问题。该数据集包含209名受试者年龄9至65岁的MRI扫描，包括3671张肿瘤图像和13273张非肿瘤图像，所有图像都在专家监督下临床验证。为解决类别不平衡问题，对肿瘤类别进行了数据增强，形成了一个适合深度学习研究的平衡数据集。


### 论文摘要

Brain tumors are abnormal cell growths in the central nervous system (CNS), and their timely detection is critical for improving patient outcomes. This paper proposes an automatic and efficient deep-learning framework for brain tumor detection from magnetic resonance imaging (MRI) scans. The framework employs a pre-trained ResNet50 model for feature extraction, followed by Global Average Pooling (GAP) and linear projection to obtain compact, high-level image representations. These features are then processed by a novel Dense-Dropout sequence, a core contribution of this work, which enhances non-linear feature learning, reduces overfitting, and improves robustness through diverse feature transformations. Another major contribution is the creation of the Mymensingh Medical College Brain Tumor (MMCBT) dataset, designed to address the lack of reliable brain tumor MRI resources. The dataset comprises MRI scans from 209 subjects (ages 9 to 65), including 3671 tumor and 13273 non-tumor images, all clinically verified under expert supervision. To overcome class imbalance, the tumor class was augmented, resulting in a balanced dataset well-suited for deep learning research.

---

## 109. Convolutional Set Transformer

**论文链接:** [http://arxiv.org/abs/2509.22889v1](http://arxiv.org/abs/2509.22889v1)

**作者:** Federico Chinello, Giacomo Boracchi

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究提出了一种名为卷积集合变换器（CST）的新型神经网络架构，能够直接处理3D图像张量，同时执行特征提取和上下文建模，在集合分类和异常检测等任务中表现优异，并与CNN可解释性方法兼容。

### 背景

现有的集合输入网络（如深度集合和集合变换器）仅限于向量输入，无法直接处理3D图像张量，因此必须与CNN特征提取器级联，限制了模型的整体性能和可解释性。

### 目的

设计一种能够直接处理3D图像张量的神经网络架构，实现特征提取和上下文建模的协同工作，提高集合处理任务的性能并增强模型的可解释性。

### 方法

提出卷积集合变换器（CST），一种可直接在3D图像张量上操作的神经网络架构，能够同时执行特征提取和上下文建模，无需级联额外的特征提取器。

### 主要发现

1. CST在集合分类和集合异常检测等任务中表现优于现有方法
2. CST与CNN可解释性方法（如Grad-CAM）具有原生兼容性
3. CST可以在大型数据集上进行预训练，并通过迁移学习适应新领域和新任务

### 结论

卷积集合变换器（CST）为处理具有高级语义但视觉异构的图像集合提供了有效解决方案，其直接处理3D图像张量的能力使特征提取和上下文建模能够协同工作，从而提高了性能并增强了可解释性。

### 翻译

我们引入了卷积集合变换器（CST），一种新型神经网络架构，设计用于处理任意数量的图像集合，这些图像在视觉上异构但共享高级语义 - 如共同类别、场景或概念。现有的集合输入网络（如深度集合和集合变换器）仅限于向量输入，无法直接处理3D图像张量。因此，它们必须与特征提取器（通常是CNN）级联，将图像编码为嵌入，然后集合输入网络才能建模图像间关系。相比之下，CST直接在3D图像张量上操作，同时执行特征提取和上下文建模，使两个过程能够协同工作。这种设计在集合分类和集合异常检测等任务中表现更优，并且与竞争方法不同，CST与CNN可解释性方法（如Grad-CAM）具有原生兼容性。最后，我们证明CST可以在大型数据集上预训练，并通过标准的迁移学习方案适应新领域和新任务。为支持进一步研究，我们发布了CST-15，这是在ImageNet上预训练的CST骨干网络（https://github.com/chinefed/convolutional-set-transformer）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的是如何有效处理图像集合的深度学习问题。现实中图像常以集合形式出现，如网页图片、社交媒体内容、医学影像中的多视图图像等，这些图像在视觉上可能不同但共享高级语义。现有方法（如Deep Sets和Set Transformer）只能处理向量输入，无法直接处理3D图像张量，需要与CNN级联，限制了特征提取和上下文建模之间的协同作用。这个问题很重要，因为联合分析图像集合可以同时利用单个图像的互补信息和它们之间的语义关系，提高识别准确率，在医学诊断、酒店分类、灾害评估等领域有广泛应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考的核心是如何结合CNN处理空间信息的能力和集合网络建模元素间关系的能力。他们意识到现有方法（Deep Sets和Set Transformer）仅限于向量输入，而CNN无法处理集合关系。因此，作者借鉴了CNN的卷积操作、Deep Sets和Set Transformer的集合处理思想，以及多头自注意力机制，创造性地设计了SetConv2D块。这个块可以直接处理3D图像张量，同时执行特征提取和上下文建模，使两个过程能够协同工作，而不是像传统方法那样顺序执行。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是直接在3D图像张量上操作，同时执行特征提取和上下文建模，使两个过程能够协同工作，并保留空间信息。整体实现流程基于SetConv2D块，它包含五个阶段：1)共享2D卷积层处理每个输入体积；2)全局平均池化将每个体积缩减为潜在向量；3)多头自注意力让不同体积的潜在向量交互；4)上下文感知偏置将注意力输出作为偏置添加到卷积体积；5)非线性激活函数。CST架构由排列等变编码器E（堆叠多个SetConv2D块）和任务特定的下游网络H组成，编码器同时处理特征提取和上下文建模，下游网络根据任务类型处理输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)SetConv2D块，可处理3D体积集合同时进行特征提取和上下文建模；2)直接处理3D图像张量，无需级联CNN；3)动态偏置机制，根据集合上下文调整卷积操作；4)原生支持CNN可解释性工具如Grad-CAM；5)大规模预训练和迁移学习能力；6)上下文感知图像分类(CIC)预训练任务；7)组合训练(CT)策略。相比之前工作，CST的不同之处在于：直接处理图像而非向量；保留空间信息而非压缩为向量；支持端到端训练而非级联设计；提供可靠解释；支持大规模预训练；训练效率更高。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CST通过创新的SetConv2D块设计，实现了对图像集合的直接处理，同时保留空间信息并支持可解释性，在多种任务上超越现有方法，并支持大规模预训练和迁移学习。'}


### 论文摘要

We introduce the Convolutional Set Transformer (CST), a novel neural architecture designed to process image sets of arbitrary cardinality that are visually heterogeneous yet share high-level semantics - such as a common category, scene, or concept. Existing set-input networks, e.g., Deep Sets and Set Transformer, are limited to vector inputs and cannot directly handle 3D image tensors. As a result, they must be cascaded with a feature extractor, typically a CNN, which encodes images into embeddings before the set-input network can model inter-image relationships. In contrast, CST operates directly on 3D image tensors, performing feature extraction and contextual modeling simultaneously, thereby enabling synergies between the two processes. This design yields superior performance in tasks such as Set Classification and Set Anomaly Detection and further provides native compatibility with CNN explainability methods such as Grad-CAM, unlike competing approaches that remain opaque. Finally, we show that CSTs can be pre-trained on large-scale datasets and subsequently adapted to new domains and tasks through standard Transfer Learning schemes. To support further research, we release CST-15, a CST backbone pre-trained on ImageNet (https://github.com/chinefed/convolutional-set-transformer).

---

## 110. Multimodal Slice Interaction Network Enhanced by Transfer Learning for Precise Segmentation of Internal Gross Tumor Volume in Lung Cancer PET/CT Imaging

**论文链接:** [http://arxiv.org/abs/2509.22841v1](http://arxiv.org/abs/2509.22841v1)

**作者:** Yi Luo, Yike Guo, Hamed Hooshangnejad, Rui Zhang, Xue Feng, Quan Chen, Wil Ngwa, Kai Ding

**发布时间:** 2025-09-26

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

本研究提出了一种基于迁移学习的多模态交互感知网络方法，结合切片交互模块(SIM)，用于提高肺癌PET/CT成像中内部大体肿瘤体积(IGTV)的分割准确性，解决了肿瘤边界PET信号衰减和标注数据有限的问题。

### 背景

肺癌是全球癌症相关死亡的主要原因。在PET/CT成像中准确勾画内部大体肿瘤体积(IGTV)对肺癌等移动性肿瘤的最佳放射治疗至关重要，但受到标注IGTV数据集有限以及肿瘤边界处PET信号强度衰减的限制。

### 目的

开发一种方法提高IGTV分割的准确性和可靠性，特别是在肺癌放射治疗计划中，并解决PET信号强度在IGTV外围切片中较弱的问题。

### 方法

提出一种基于迁移学习的方法，使用多模态交互感知网络结合MAMBA架构，在大量GTV数据集上预训练后在私有IGTV队列上微调。引入切片交互模块(SIM)在2.5D分割框架内，结合通道和空间注意力分支与深度卷积，有效建模切片间关系。

### 主要发现

在私有IGTV数据集上达到0.609的Dice系数，显著超过传统基线分数0.385。所提出的方法能够更稳健地学习切片间的依赖关系，提高整体分割性能。

### 结论

迁移学习结合先进的多模态技术和SIM模块增强了IGTV分割的可靠性和临床相关性，对肺癌放射治疗计划具有重要意义。

### 翻译

肺癌仍然是全球癌症相关死亡的主要原因。在PET/CT成像中准确勾画内部大体肿瘤体积(IGTV)对于肺癌等移动性肿瘤的最佳放射治疗至关重要，以考虑肿瘤运动，但受到标注IGTV数据集有限以及肿瘤边界处PET信号强度衰减的限制。在本研究中，我们提出了一种基于迁移学习的方法，利用多模态交互感知网络结合MAMBA，在大量大体肿瘤体积(GTV)数据集上预训练，随后在私有IGTV队列上微调。该队列是Lung-cancer Unified Cross-modal Imaging Dataset (LUCID)的PET/CT子集。为解决IGTV外围切片中PET信号强度较弱的问题，我们在2.5D分割框架内引入了切片交互模块(SIM)，有效建模切片间关系。我们提出的模块将通道和空间注意力分支与深度卷积相结合，使切片间依赖关系的学习更加稳健，从而提高整体分割性能。全面的实验评估表明，我们的方法在私有IGTV数据集上达到0.609的Dice系数，显著超过传统基线分数0.385。这项工作突显了迁移学习结合先进多模态技术和SIM模块的潜力，可增强肺癌放射治疗计划中IGTV分割的可靠性和临床相关性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决肺癌PET/CT图像中内部大体肿瘤体积(IGTV)的精确分割问题。这个问题在现实中非常重要，因为肺癌是全球癌症相关死亡的主要原因，而准确的IGTV分割对于肺癌放射治疗计划至关重要，特别是对于移动的肿瘤需要考虑呼吸运动导致的肿瘤位置变化。IGTV分割面临的主要挑战包括标注数据稀缺和肿瘤边界处PET信号衰减导致的边界模糊问题，直接影响治疗效果和患者生存率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先注意到IGTV标注数据有限，但GTV(大体肿瘤体积)有大量标注数据，因此采用迁移学习策略解决数据稀缺问题。他们借鉴了现有的CIPA(Cross-modal Interactive Perception Network)架构来融合PET和CT多模态信息，并观察到传统2D分割方法在处理肿瘤边界PET信号较弱的区域存在局限性。基于这些思考，作者设计了2.5D分割策略和切片交互模块(SIM)，通过同时处理三个连续切片和建模切片间关系来增强对肿瘤边界的识别能力。SIM模块包含三个互补分支：通道注意力、空间注意力和切片关系分支，专门针对IGTV分割的挑战进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用迁移学习解决IGTV数据稀缺问题，通过2.5D分割策略和切片交互模块(SIM)增强模型对肿瘤边界区域的识别能力，并结合PET和CT多模态信息提高分割准确性。整体实现流程分为三个阶段：1)数据准备阶段，使用PCLT20k数据集(21,930个PET/CT图像对)进行预训练，LUCID-PET/CT数据集(1,067个图像对)进行微调；2)模型架构阶段，基于CIPA架构使用Mamba作为骨干网络，采用2.5D输入策略并集成SIM模块；3)训练策略阶段，先在GTV数据集上预训练，再在IGTV数据集上微调同时将2D模型扩展为2.5D架构，使用Dice损失和二元交叉熵损失作为训练目标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将迁移学习应用于GTV到IGTV的分割任务，利用大量GTV标注数据解决IGTV数据稀缺问题；2)提出2.5D分割策略，同时处理三个连续切片提供更丰富的空间上下文；3)设计切片交互模块(SIM)，通过通道注意力、空间注意力和切片关系建模来处理肿瘤边界处PET信号弱的问题；4)优化多模态融合方式，更好地处理周围器官高摄取信号对肿瘤分割的干扰。相比之前的工作，本文专门针对IGTV分割的特点进行了优化，解决了数据稀缺和边界模糊两大核心挑战，显著提高了分割精度(Dice从0.385提升到0.609)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合迁移学习和多模态切片交互网络的方法，有效解决了肺癌PET/CT图像中内部大体肿瘤体积(IGTV)分割的数据稀缺和边界模糊问题，显著提高了分割精度。'}


### 论文摘要

Lung cancer remains the leading cause of cancerrelated deaths globally. Accurate delineation of internal gross tumor volume (IGTV) in PET/CT imaging is pivotal for optimal radiation therapy in mobile tumors such as lung cancer to account for tumor motion, yet is hindered by the limited availability of annotated IGTV datasets and attenuated PET signal intensity at tumor boundaries. In this study, we present a transfer learningbased methodology utilizing a multimodal interactive perception network with MAMBA, pre-trained on extensive gross tumor volume (GTV) datasets and subsequently fine-tuned on a private IGTV cohort. This cohort constitutes the PET/CT subset of the Lung-cancer Unified Cross-modal Imaging Dataset (LUCID). To further address the challenge of weak PET intensities in IGTV peripheral slices, we introduce a slice interaction module (SIM) within a 2.5D segmentation framework to effectively model inter-slice relationships. Our proposed module integrates channel and spatial attention branches with depthwise convolutions, enabling more robust learning of slice-to-slice dependencies and thereby improving overall segmentation performance. A comprehensive experimental evaluation demonstrates that our approach achieves a Dice of 0.609 on the private IGTV dataset, substantially surpassing the conventional baseline score of 0.385. This work highlights the potential of transfer learning, coupled with advanced multimodal techniques and a SIM to enhance the reliability and clinical relevance of IGTV segmentation for lung cancer radiation therapy planning.

---

## 111. Forge4D: Feed-Forward 4D Human Reconstruction and Interpolation from Uncalibrated Sparse-view Videos

**论文链接:** [http://arxiv.org/abs/2509.24209v1](http://arxiv.org/abs/2509.24209v1)

**作者:** Yingdong Hu, Yisheng He, Jinnan Chen, Weihao Yuan, Kejie Qiu, Zehong Lin, Siyu Zhu, Zilong Dong, Jun Zhang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了Forge4D，一个前馈4D人体重建和插值模型，能够从未校准的稀疏视角视频中高效重建时间对齐的表示，实现新视图和新时间的合成。

### 背景

从未校准的稀疏视角视频即时重建动态3D人体对许多下游应用至关重要，但现有方法要么受限于重建速度慢，要么无法生成新时间表示。

### 目的

解决现有方法的局限性，开发一种能够高效重建且能生成新时间表示的4D人体重建和插值模型。

### 方法

将4D重建和插值问题简化为流式3D高斯重建和密集运动预测的联合任务；通过从未校准稀疏视角图像重建静态3D高斯，并引入可学习状态标记实现时间一致性；设计运动预测模块预测相邻帧间3D高斯的密集运动，结合遮挡感知的高斯融合过程进行时间插值；采用自监督重定位损失和遮挡感知光流损失优化模型。

### 主要发现

在领域内和领域外数据集上的大量实验证明了Forge4D模型的有效性，能够实现高效且准确的动态3D人体重建。

### 结论

Forge4D成功解决了现有方法的局限性，实现了从未校准稀疏视角视频中即时重建动态3D人体的目标，并支持新视图和新时间的合成。

### 翻译

从未校准的稀疏视角视频中即时重建动态3D人体对众多下游应用至关重要。然而，现有方法要么受限于重建速度慢，要么无法生成新时间表示。为应对这些挑战，我们提出了Forge4D，一个前馈4D人体重建和插值模型，能够从未校准的稀疏视角视频中高效重建时间对齐的表示，实现新视图和新时间的合成。我们的模型将4D重建和插值问题简化为流式3D高斯重建和密集运动预测的联合任务。对于流式3D高斯重建任务，我们首先从未校准的稀疏视角图像重建静态3D高斯，然后引入可学习状态标记，通过在不同时间戳间交互更新共享信息，以内存友好的方式强制时间一致性。对于新时间合成，我们设计了一个新颖的运动预测模块来预测相邻帧之间每个3D高斯的密集运动，结合遮挡感知的高斯融合过程来插值任意时间戳的3D高斯。为克服密集运动监督缺乏真实值的问题，我们将密集运动预测制定为密集点匹配任务，并引入自监督重定位损失来优化此模块。此外，还引入了遮挡感知的光流损失，确保运动与合理的人体运动一致，提供更强的正则化。大量实验证明了我们的模型在领域内和领域外数据集上的有效性。项目页面和代码位于：https://zhenliuzju.github.io/huyingdong/Forge4D。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从未校准的稀疏视角视频中即时重建动态3D人体并实现时间插值的问题。这个问题在现实中非常重要，因为它能支持实时直播、体育赛事转播、增强/虚拟现实、关节建模和沉浸式全息通信等多种应用场景。现有方法要么重建速度慢，要么无法生成新时间点的表示，限制了这些技术的实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从现有工作的局限性出发进行思考：迭代优化方法需要整个密集视角视频序列且训练时间长；前馈视觉几何模型虽能重建点云但无法实现逼真新视角合成；扩展到静态3D高斯预测的方法无法处理动态场景。作者借鉴了VGGT等大型3D重建模型的知识先验，但创新性地解决了两个主要挑战：通过度量规对齐解决尺度不一致问题，通过状态标记以流式方式合并时间信息以提高效率并解决内存问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将4D重建和插值问题分解为两个任务：流式3D高斯重建和密集运动预测。这种设计简化了前馈回归问题，且重建的流式3D高斯为密集运动预测提供了视觉监督。整体流程分为三个阶段：1) 前馈静态3D高斯重建阶段，利用预训练VGGT模型并引入度量规对齐；2) 流式动态重建阶段，使用状态标记强制时间一致性；3) 密集运动预测和动态高斯融合阶段，预测运动并使用遮挡感知的高斯融合实现新时间合成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个前馈4D人体重建模型；2) 将问题分解为流式3D高斯预测和密集人体运动估计；3) 度量规正则化解决尺度不一致；4) 状态标记机制实现高效时间信息整合；5) 将运动预测作为点匹配问题并引入创新损失函数；6) 遮挡感知的高斯融合方法。相比之前工作，Forge4D是首个从未校准稀疏视角视频中实现前馈4D人体重建的方法，能同时处理新视角合成和新时间插值，通过问题分解和创新损失函数实现了更高效准确的重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Forge4D首次实现了从未校准稀疏视角视频中高效重建动态4D人体模型，并支持任意视角和时间点的实时渲染与插值。'}


### 论文摘要

Instant reconstruction of dynamic 3D humans from uncalibrated sparse-view videos is critical for numerous downstream applications. Existing methods, however, are either limited by the slow reconstruction speeds or incapable of generating novel-time representations. To address these challenges, we propose Forge4D, a feed-forward 4D human reconstruction and interpolation model that efficiently reconstructs temporally aligned representations from uncalibrated sparse-view videos, enabling both novel view and novel time synthesis. Our model simplifies the 4D reconstruction and interpolation problem as a joint task of streaming 3D Gaussian reconstruction and dense motion prediction. For the task of streaming 3D Gaussian reconstruction, we first reconstruct static 3D Gaussians from uncalibrated sparse-view images and then introduce learnable state tokens to enforce temporal consistency in a memory-friendly manner by interactively updating shared information across different timestamps. For novel time synthesis, we design a novel motion prediction module to predict dense motions for each 3D Gaussian between two adjacent frames, coupled with an occlusion-aware Gaussian fusion process to interpolate 3D Gaussians at arbitrary timestamps. To overcome the lack of the ground truth for dense motion supervision, we formulate dense motion prediction as a dense point matching task and introduce a self-supervised retargeting loss to optimize this module. An additional occlusion-aware optical flow loss is introduced to ensure motion consistency with plausible human movement, providing stronger regularization. Extensive experiments demonstrate the effectiveness of our model on both in-domain and out-of-domain datasets. Project page and code at: https://zhenliuzju.github.io/huyingdong/Forge4D.

---

## 112. Learning Adaptive Pseudo-Label Selection for Semi-Supervised 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2509.23880v1](http://arxiv.org/abs/2509.23880v1)

**作者:** Taehun Kong, Tae-Kyun Kim

**发布时间:** 2025-09-28

### GPT解析

### 总结

该研究提出了一种新颖的半监督3D目标检测框架，通过可学习的伪标记模块自动和自适应地选择高质量伪标签，引入两个网络评估伪标签质量并确定上下文自适应阈值，同时采用软监督策略处理伪标签噪声，实验证明该方法在KITTI和Waymo数据集上显著优于现有方法。

### 背景

半监督3D目标检测旨在利用未标记数据减少昂贵的3D标注成本。现有基于伪标签的教师-学生框架面临从教师预测中选择高质量伪标签的挑战。大多数方法通过手动设置的置信度阈值选择伪标签，而最新工作虽通过动态阈值或改进伪标签质量来应对挑战，但仍忽略了上下文信息（如物体距离、类别和学习状态），且仅使用网络提供的部分信息不充分评估伪标签质量。

### 目的

提出一个新颖的SS3DOD框架，具有可学习的伪标记模块，能够自动和自适应地选择高质量的伪标签。

### 方法

在教师输出级别引入两个网络，通过分数融合可靠地评估伪标签质量，并确定由伪标签与真实边界框对齐监督的上下文自适应阈值。同时引入软监督策略，使学生在伪标签噪声下能够稳健学习，优先选择更干净的标签而非有噪声的标签。

### 主要发现

在KITTI和Waymo数据集上的大量实验证明了该方法的有效性。所提出的方法在选择高精度伪标签的同时，保持了更广泛的上下文覆盖率和更高的召回率，显著改进了相关的SS3DOD方法。

### 结论

通过可学习的伪标记模块和软监督策略，该方法有效解决了半监督3D目标检测中伪标签选择的问题，实验证明在多个数据集上表现优异。

### 翻译

半监督3D目标检测旨在利用未标记数据减少昂贵的3D标注成本。近期研究采用基于伪标签的教师-学生框架并展示了令人印象深刻的性能。这些框架的主要挑战是从教师的预测中选择高质量的伪标签。然而，大多数先前方法通过比较超过手动设置的阈值的置信度分数来选择伪标签。最新工作通过动态阈值或改进伪标签质量来解决这一挑战。此类方法仍然忽略了上下文信息，例如物体距离、类别和学习状态，并且仅使用网络提供的部分信息不充分地评估伪标签质量。在本工作中，我们提出了一种新颖的SS3DOD框架，具有可学习的伪标记模块，能够自动和自适应地选择高质量的伪标签。我们的方法在教师输出级别引入两个网络。这些网络通过分数融合可靠地评估伪标签质量，并确定由伪标签与真实边界框对齐监督的上下文自适应阈值。此外，我们引入了一种软监督策略，可以在伪标签噪声下稳健学习。这有助于学生在半监督学习中优先选择更干净的标签而非有噪声的标签。在KITTI和Waymo数据集上的大量实验证明了我们方法的有效性。所提出的方法在选择高精度伪标签的同时，保持了更广泛的上下文覆盖率和更高的召回率，显著改进了相关的SS3DOD方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决半监督3D目标检测中伪标签选择的问题。现有方法通常使用手动设置的阈值来选择高质量伪标签，但这种方法忽略了物体距离、类别和学习状态等上下文信息，无法充分利用网络中的可用信息来评估伪标签质量。这个问题很重要，因为3D目标检测对自动驾驶、机器人和AR/VR等领域至关重要，而高质量的3D标注非常耗时且昂贵，导致未标记数据远多于标记数据，半监督学习能有效利用这些未标记数据提高性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到分类置信度和物体性分数在不同类别、距离和学习状态下有不同的分布，表明选择伪标签的最佳阈值应该是上下文相关的。同时，现有方法只使用部分信息来评估伪标签质量。作者借鉴了教师-学生的半监督学习框架和伪标签选择的基本思想，但创新性地引入了可学习的伪标签选择模块（PSM），通过神经网络而非手动设置来选择伪标签，并融合多种分数来评估伪标签质量，同时考虑上下文信息来自适应确定阈值。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用神经网络来学习如何选择高质量的伪标签，通过融合多种分数来评估伪标签质量，并根据上下文信息自适应确定阈值，同时引入软监督策略来减轻伪标签噪声的影响。整体流程包括：1) 燃烧阶段：在标记数据上初始化教师和学生模型；2) 伪标签阶段：使用弱增强和教师模型生成伪标签；3) 半监督阶段：学生模型在有标记和无标记数据上进行学习，使用伪标签选择模块选择高质量伪标签；4) 教师更新阶段：通过EMA更新教师模型。伪标签选择模块包含伪标签质量估计器（融合多种分数）和上下文感知阈值估计器（根据类别和距离确定阈值）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 学习型伪标签选择模块（PSM），首次使用神经网络建模伪标签选择过程；2) 伪标签质量估计器（PQE），融合多种分数提供更可靠的伪标签质量评估；3) 上下文感知阈值估计器（CTE），根据上下文自适应确定阈值；4) 软监督策略，结合GT采样增强和损失重加权减轻伪标签噪声影响。相比之前的工作（如HSSDA、ATF-3D等），本文方法不再依赖手动设置或简单动态阈值，而是通过神经网络学习自适应阈值；不仅考虑部分信息，而是融合多种分数全面评估伪标签质量；不仅考虑有限的上下文因素，而是全面考虑多种上下文信息；监督策略更简单有效，在保持高精度的同时实现了更广泛的上下文覆盖和更高的召回率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于神经网络的自适应伪标签选择方法，通过融合多种分数和考虑上下文信息，显著提高了半监督3D目标检测的性能，特别是在标注数据有限的情况下。'}


### 论文摘要

Semi-supervised 3D object detection (SS3DOD) aims to reduce costly 3D annotations utilizing unlabeled data. Recent studies adopt pseudo-label-based teacher-student frameworks and demonstrate impressive performance. The main challenge of these frameworks is in selecting high-quality pseudo-labels from the teacher's predictions. Most previous methods, however, select pseudo-labels by comparing confidence scores over thresholds manually set. The latest works tackle the challenge either by dynamic thresholding or refining the quality of pseudo-labels. Such methods still overlook contextual information e.g. object distances, classes, and learning states, and inadequately assess the pseudo-label quality using partial information available from the networks. In this work, we propose a novel SS3DOD framework featuring a learnable pseudo-labeling module designed to automatically and adaptively select high-quality pseudo-labels. Our approach introduces two networks at the teacher output level. These networks reliably assess the quality of pseudo-labels by the score fusion and determine context-adaptive thresholds, which are supervised by the alignment of pseudo-labels over GT bounding boxes. Additionally, we introduce a soft supervision strategy that can learn robustly under pseudo-label noises. This helps the student network prioritize cleaner labels over noisy ones in semi-supervised learning. Extensive experiments on the KITTI and Waymo datasets demonstrate the effectiveness of our method. The proposed method selects high-precision pseudo-labels while maintaining a wider coverage of contexts and a higher recall rate, significantly improving relevant SS3DOD methods.

---

## 113. Event-based Facial Keypoint Alignment via Cross-Modal Fusion Attention and Self-Supervised Multi-Event Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.24968v1](http://arxiv.org/abs/2509.24968v1)

**作者:** Donghwa Kang, Junho Kim, Dongwoo Kang

**发布时间:** 2025-09-29

**备注:** 11 pages, 7 figures

### GPT解析

### 总结

该研究提出了一种基于跨模态融合注意力和自监督多事件表示学习的新框架，用于解决事件相机在挑战性条件下进行人脸关键点对齐的问题，实验证明该方法在多个评估指标上优于现有方法。

### 背景

事件相机在低光和快速运动等挑战性条件下对人脸关键点对齐具有独特优势，因其高时间分辨率和对光照变化的鲁棒性。然而，现有的RGB方法在事件数据上表现不佳，且仅用事件数据训练会因空间信息有限而导致次优性能，同时缺乏全面标注的事件数据集也阻碍了该领域发展。

### 目的

解决事件相机人脸关键点对齐中的三个主要问题：现有RGB方法在事件数据上表现不佳、事件数据空间信息有限、以及缺乏全面标注的事件数据集。

### 方法

提出了一种结合跨模态融合注意力(CMFA)和自监督多事件表示学习(SSMER)的新框架。CMFA用于整合RGB数据，引导模型从事件输入中提取鲁棒的面部特征；SSMER则使模型能够从未标记的事件数据中进行有效特征学习，克服空间信息限制。

### 主要发现

在真实事件数据集E-SIE和公共WFLW-V基准的合成事件版本上的大量实验表明，该方法在多个评估指标上一致优于现有最先进方法。

### 结论

所提出的框架通过跨模态融合和自监督学习，有效解决了事件相机人脸关键点对齐中的挑战，克服了事件数据空间信息有限的限制。

### 翻译

事件相机由于其高时间分辨率和对不同光照变化的鲁棒性，在低光和快速运动等挑战性条件下为人脸关键点对齐提供了独特优势。然而，现有的RGB人脸关键点对齐方法在事件数据上表现不佳，而仅在事件数据上训练通常会导致次优性能，因为其空间信息有限。此外，缺乏全面标注的事件数据集进一步阻碍了该领域的进展。为解决这些问题，我们提出了一种基于跨模态融合注意力(CMFA)和自监督多事件表示学习(SSMER)的新框架，用于基于事件的人脸关键点对齐。我们的框架采用CMFA整合相应的RGB数据，引导模型从事件输入图像中提取鲁棒的面部特征。同时，SSMER使模型能够从未标记的事件数据进行有效的特征学习，克服了空间限制。在我们真实事件数据集E-SIE和公共WFLW-V基准的合成事件版本上的大量实验表明，我们的方法在多个评估指标上一致超越了最先进的方法。


### 论文摘要

Event cameras offer unique advantages for facial keypoint alignment under challenging conditions, such as low light and rapid motion, due to their high temporal resolution and robustness to varying illumination. However, existing RGB facial keypoint alignment methods do not perform well on event data, and training solely on event data often leads to suboptimal performance because of its limited spatial information. Moreover, the lack of comprehensive labeled event datasets further hinders progress in this area. To address these issues, we propose a novel framework based on cross-modal fusion attention (CMFA) and self-supervised multi-event representation learning (SSMER) for event-based facial keypoint alignment. Our framework employs CMFA to integrate corresponding RGB data, guiding the model to extract robust facial features from event input images. In parallel, SSMER enables effective feature learning from unlabeled event data, overcoming spatial limitations. Extensive experiments on our real-event E-SIE dataset and a synthetic-event version of the public WFLW-V benchmark show that our approach consistently surpasses state-of-the-art methods across multiple evaluation metrics.

---

## 114. When Audio Generators Become Good Listeners: Generative Features for Understanding Tasks

**论文链接:** [http://arxiv.org/abs/2509.24635v1](http://arxiv.org/abs/2509.24635v1)

**作者:** Zeyu Xie, Chenxing Li, Xuenan Xu, Mengyue Wu, Wenfu Wang, Ruibo Fu, Meng Yu, Dong Yu, Yuexian Zou

**发布时间:** 2025-09-29

### GPT解析

### 总结

这项工作探索了利用生成特征增强音频理解能力，通过结合判别式特征和生成式模型的优势，提出了一种有效的融合策略，在多个音频处理任务中取得了性能提升。

### 背景

传统的判别式特征直接优化后验概率，强调语义抽象但丢失了细粒度细节。而音频生成模型能够同时编码时空感知（捕捉时间和频率上的局部声学纹理）和语义先验（知道生成什么内容）。

### 目的

探索生成特征与判别特征之间的互补关系，提出一种有效的融合策略，以结合两者的优势，实现更全面的音频理解。

### 方法

系统研究生成特征与判别特征之间的差异和互补关系，并提出了一种融合策略。

### 主要发现

在声音事件分类、标注等任务中，特别是在音频字幕这种细粒度任务中，所提出的方法展示了持续的性能提升。生成与判别的互补性能够为音频理解提供详细的感知能力和语义意识。

### 结论

这项工作为音频表征学习引入了新的视角，强调了生成与判别互补性可以同时提供详细感知和语义意识，对音频理解具有重要意义。

### 翻译

这项工作开创了利用生成特征增强音频理解的先河。与直接优化后验而因此强调语义抽象同时丢失细粒度细节的传统判别特征不同，音频生成模型天生编码时空感知（捕捉时间和频率上的局部声学纹理）和语义先验（知道生成什么）。这促使我们探索这些互补优势之间的桥梁。我们系统研究了它们的差异和互补关系，并最终提出了一种有效的融合策略。在声音事件分类、标注等多个任务上的实验，特别是音频字幕的细粒度任务，展示了持续的性能提升。除了经验上的改进外，这项工作更重要的是为音频表征学习引入了新视角，强调生成判别互补性可以为音频理解提供详细感知和语义意识。


### 论文摘要

This work pioneers the utilization of generative features in enhancing audio understanding. Unlike conventional discriminative features that directly optimize posterior and thus emphasize semantic abstraction while losing fine grained details, audio generation models inherently encode both spatiotemporal perception (capturing local acoustic texture across time and frequency) and semantic prior (knowing what to generate). It motivates us to explore the bridge of these complementary strengths. We provide a systematic investigation of their differences and complementary relationships, and ultimately propose an effective fusion strategy. Experiments across multiple tasks, including sound event classification, tagging, and particularly the fine grained task of audio captioning, demonstrate consistent performance gains. Beyond empirical improvements, this work more importantly introduces a new perspective on audio representation learning, highlighting that generative discriminative complementarity can provide both detailed perception and semantic awareness for audio understanding.

---

## 115. NeoWorld: Neural Simulation of Explorable Virtual Worlds via Progressive 3D Unfolding

**论文链接:** [http://arxiv.org/abs/2509.24441v1](http://arxiv.org/abs/2509.24441v1)

**作者:** Yanpeng Zhao, Shanyan Guan, Yunbo Wang, Yanhao Ge, Wei Li, Xiaokang Yang

**发布时间:** 2025-09-29

### GPT解析

### 总结

NeoWorld是一个从单张输入图像生成交互式3D虚拟世界的深度学习框架，采用混合场景结构，对关键前景对象进行3D建模，对背景和未交互区域进行2D合成，确保效率并提供动态沉浸式体验。

### 背景

受科幻小说《Simulacron-3》(1964)中的按需世界构建概念启发，不同于之前依赖全局世界生成或2D幻觉的方法。

### 目的

开发一个能够从单张输入图像生成交互式3D虚拟世界的深度学习框架，实现高效且视觉上连贯的用户体验。

### 方法

采用对象中心的3D表示，对关键前景对象进行完整3D建模，对背景和未交互区域进行2D合成，使用先进的表示学习和对象到3D技术实现混合场景结构，允许用户通过自然语言命令控制对象外观和动态。

### 主要发现

NeoWorld在WorldScore基准测试中显著优于现有的2D和深度分层2.5D方法。

### 结论

NeoWorld通过创新的混合场景结构和先进的表示学习技术，实现了从单张输入图像生成交互式3D虚拟世界的高效方法，提供了动态、沉浸式且视觉连贯的探索体验。

### 翻译

我们介绍NeoWorld，这是一个从单张输入图像生成交互式3D虚拟世界的深度学习框架。受科幻小说《Simulacron-3》(1964)中的按需世界构建概念启发，我们的系统构建了广阔的环境，其中只有用户主动探索的区域通过对象中心的3D表示以高视觉保真度进行渲染。与依赖全局世界生成或2D幻觉的先前方法不同，NeoWorld对关键前景对象进行完整3D建模，同时合成背景和未交互区域的2D表示以确保效率。这种混合场景结构通过最先进的表示学习和对象到3D技术实现，支持灵活的视角操作和物理上合理的场景动画，允许用户使用自然语言命令控制对象外观和动态。当用户与环境交互时，虚拟世界逐步展开增加的3D细节，提供动态、沉浸式且视觉连贯的探索体验。在WorldScore基准测试中，NeoWorld显著优于现有的2D和深度分层2.5D方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单张图像生成交互式、无限扩展的3D虚拟世界的问题，同时保持计算效率和视觉真实感。这个问题在现实中很重要，因为现有的3D世界生成方法要么计算成本过高（全局生成），要么缺乏3D一致性（2D幻觉），而用户希望能够与虚拟世界进行交互，不仅仅是视觉导航。此外，物理上合理的动态模拟对于创建逼真的虚拟环境至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受科幻小说《Simulacron-3》中的'按需构建世界'概念启发，注意到现有方法在交互式世界生成方面的局限性。他们设计了一种混合场景结构，结合高效的2D表示和高质量的3D表示，并设计了渐进式3D展开机制。作者借鉴了多项现有工作：基于学习的交互式世界生成技术（如WonderJourney和WonderWorld）、对象中心表示学习（如Gaussian Splatting）、图像到3D重建技术（如Amodal3R）以及大型语言模型（如Gemini-2.5pro）用于理解用户意图。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个混合场景结构，背景使用轻量级2D表示，前景对象使用完整3D表示，随着用户探索或交互，渐进式地将2D对象表示展开为3D。整体实现流程包括：1) 初始化阶段将输入图像分解为前景和背景层；2) 通过逆渲染管道重建输入图像；3) 用户通过自然语言命令或相机移动探索环境；4) 根据用户交互将相关2D对象重建为完整3D表示；5) 将3D对象与原始2D图像在对象级别对齐；6) 支持基于物理的对象操作和动画。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 对象中心神经场景表示，结合分层高斯分裂与紧凑的实例感知特征；2) 渐进式2.5D到3D场景展开机制，根据对象接近度或用户提示优先级逐步展开；3) 用户-场景交互模块，支持直观的对象级操作和基于物理的动画。相比之前的工作，NeoWorld不仅支持视觉导航，还支持物理交互；将关键前景对象建模为完整3D，同时保持背景区域的2D表示以实现效率；实现了混合场景结构，支持灵活的视角操作和物理上合理的场景动画；用户可以使用自然语言命令控制对象；虚拟世界随用户交互以递增的3D细节逐步展开。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NeoWorld提出了一种创新的深度学习框架，通过渐进式3D展开技术，从单张图像生成交互式虚拟世界，实现了计算效率与视觉真实感的平衡，并支持基于物理的对象操作和动态模拟。'}


### 论文摘要

We introduce NeoWorld, a deep learning framework for generating interactive 3D virtual worlds from a single input image. Inspired by the on-demand worldbuilding concept in the science fiction novel Simulacron-3 (1964), our system constructs expansive environments where only the regions actively explored by the user are rendered with high visual realism through object-centric 3D representations. Unlike previous approaches that rely on global world generation or 2D hallucination, NeoWorld models key foreground objects in full 3D, while synthesizing backgrounds and non-interacted regions in 2D to ensure efficiency. This hybrid scene structure, implemented with cutting-edge representation learning and object-to-3D techniques, enables flexible viewpoint manipulation and physically plausible scene animation, allowing users to control object appearance and dynamics using natural language commands. As users interact with the environment, the virtual world progressively unfolds with increasing 3D detail, delivering a dynamic, immersive, and visually coherent exploration experience. NeoWorld significantly outperforms existing 2D and depth-layered 2.5D methods on the WorldScore benchmark.

---

## 116. Semantic Compression via Multimodal Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.24431v1](http://arxiv.org/abs/2509.24431v1)

**作者:** Eleonora Grassucci, Giordano Cicchetti, Aurelio Uncini, Danilo Comminiello

**发布时间:** 2025-09-29

### GPT解析

### 总结

该论文探讨了多模态表示学习中的语义压缩问题，提出了一种基于模态对齐的压缩方法，能够在减少内存占用的同时保持性能。

### 背景

多模态表示学习生成高维嵌入，将不同模态对齐到共享潜在空间，虽实现强泛化能力，但也带来存储和下游处理的可扩展性挑战。

### 目的

实现语义压缩，减少多模态嵌入的内存占用，同时保留其跨模态表示共享语义内容的能力。

### 方法

证明减少模态差距与训练后语义压缩可行性之间的强关联；当模态差距足够小时，表达相同语义的不同模态嵌入共享空间，可用单个质心替换多个嵌入；提出直接在预训练编码器上操作的语义压缩新方法。

### 主要发现

模态对齐是实现语义压缩的关键使能因素；所提方法在不牺牲性能的情况下可实现显著压缩。

### 结论

通过减少模态差距，可实现有效的语义压缩，同时保持多模态嵌入的语义表示能力。

### 翻译

多模态表示学习生成高维嵌入，将不同模态对齐到共享的潜在空间中。虽然这实现了强大的泛化能力，但也引入了存储和下游处理的可扩展性挑战。一个关键开放问题是如何实现语义压缩，减少多模态嵌入的内存占用，同时保留它们跨模态表示共享语义内容的能力。在本文中，我们证明了减少模态差距（即不同模态嵌入之间的残留分离）与训练后语义压缩的可行性之间的强关联。当差距足够减少时，表达相同语义的不同模态嵌入共享空间的一部分。因此，它们的质心是这种语义概念的忠实表示。这使得可以用单个质心替换多个嵌入，实现显著的内存节省。我们提出了一种基于上述直觉的语义压缩新方法，直接在预训练编码器上操作。我们在各种大规模多模态下游任务上证明了其有效性。我们的结果突显模态对齐是实现语义压缩的关键使能因素，表明所提出的方法在不牺牲性能的情况下实现了显著压缩。


### 论文摘要

Multimodal representation learning produces high-dimensional embeddings that align diverse modalities in a shared latent space. While this enables strong generalization, it also introduces scalability challenges, both in terms of storage and downstream processing. A key open problem is how to achieve semantic compression, reducing the memory footprint of multimodal embeddings while preserving their ability to represent shared semantic content across modalities. In this paper, we prove a strong connection between reducing the modality gap, which is the residual separation of embeddings from different modalities, and the feasibility of post-training semantic compression. When the gap is sufficiently reduced, embeddings from different modalities but expressing the same semantics share a common portion of the space. Therefore, their centroid is a faithful representation of such a semantic concept. This enables replacing multiple embeddings with a single centroid, yielding significant memory savings. We propose a novel approach for semantic compression grounded on the latter intuition, operating directly on pretrained encoders. We demonstrate its effectiveness across diverse large-scale multimodal downstream tasks. Our results highlight that modality alignment is a key enabler for semantic compression, showing that the proposed approach achieves significant compression without sacrificing performance.

---

## 117. ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2509.24414v1](http://arxiv.org/abs/2509.24414v1)

**作者:** Tao Yin, Xiaohong Zhang, Shaochen Fu, Zhibin Zhang, Li Huang, Yiyuan Yang, Kaixiang Yang, Meng Yan

**发布时间:** 2025-09-29

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS  2025)

### GPT解析

### 总结

论文提出了一种名为ScatterAD的新方法，用于解决工业物联网时间序列异常检测中复杂的时空耦合问题。该方法利用高维空间中样本的散射现象作为信号，通过拓扑编码器捕获图结构散射，时间编码器约束过度散射，并采用对比融合机制确保表示的互补性。

### 背景

工业物联网时间序列异常检测面临多元数据中复杂时空耦合的挑战。传统方法独立建模空间或时间依赖关系，导致表示学习不理想，对高维空间中的异常分散性敏感度有限。

### 目的

提高多元时间序列异常检测的性能，通过建模表示散射现象来增强时空异常检测能力。

### 方法

提出ScatterAD方法，包含拓扑编码器捕获图结构散射，时间编码器通过最小化相邻时间步均方误差约束过度散射，引入对比融合机制确保时间与拓扑表示的互补性，并从理论上证明最大化条件互信息可提高跨视图一致性和判别性表示。

### 主要发现

正常和异常样本在高维空间中都有分散趋势，特别是异常样本更为分散；将散射现象形式化为样本表示间的平均成对距离并作为信号可有效提升异常检测性能。

### 结论

ScatterAD在多个公共基准上实现了最先进的多元时间序列异常检测性能，证明了散射现象作为归纳信号的有效性。

### 翻译

工业物联网时间序列异常检测的一个主要挑战在于多元数据中复杂的时空耦合关系。然而，传统异常检测方法专注于独立建模空间或时间依赖关系，导致表示学习不理想，且对高维空间中的异常分散性敏感度有限。在这项工作中，我们进行了实证分析，表明正常和异常样本都倾向于在高维空间中分散，特别是异常样本分散更为明显。我们将这种分散现象形式化为散射，通过样本表示之间的平均成对距离来量化，并利用它作为归纳信号来增强时空异常检测。技术上，我们提出ScatterAD来建模跨时间和拓扑维度的表示散射。ScatterAD包含一个拓扑编码器用于捕获图结构散射，以及一个时间编码器通过最小化相邻时间步之间的均方误差来约束过度散射。我们引入了对比融合机制来确保学习到的时间和拓扑表示的互补性。此外，我们从理论上证明，最大化时间和拓扑视图之间的条件互信息可以提高跨视图一致性，并增强更具判别力的表示。在多个公共基准上的广泛实验表明，ScatterAD在多元时间序列异常检测方面取得了最先进的性能。代码可在以下仓库获取：https://github.com/jk-sounds/ScatterAD。


### 论文摘要

One main challenge in time series anomaly detection for industrial IoT lies in the complex spatio-temporal couplings within multivariate data. However, traditional anomaly detection methods focus on modeling spatial or temporal dependencies independently, resulting in suboptimal representation learning and limited sensitivity to anomalous dispersion in high-dimensional spaces. In this work, we conduct an empirical analysis showing that both normal and anomalous samples tend to scatter in high-dimensional space, especially anomalous samples are markedly more dispersed. We formalize this dispersion phenomenon as scattering, quantified by the mean pairwise distance among sample representations, and leverage it as an inductive signal to enhance spatio-temporal anomaly detection. Technically, we propose ScatterAD to model representation scattering across temporal and topological dimensions. ScatterAD incorporates a topological encoder for capturing graph-structured scattering and a temporal encoder for constraining over-scattering through mean squared error minimization between neighboring time steps. We introduce a contrastive fusion mechanism to ensure the complementarity of the learned temporal and topological representations. Additionally, we theoretically show that maximizing the conditional mutual information between temporal and topological views improves cross-view consistency and enhances more discriminative representations. Extensive experiments on multiple public benchmarks show that ScatterAD achieves state-of-the-art performance on multivariate time series anomaly detection. Code is available at this repository: https://github.com/jk-sounds/ScatterAD.

---

## 118. Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers

**论文链接:** [http://arxiv.org/abs/2509.24317v1](http://arxiv.org/abs/2509.24317v1)

**作者:** Xianhang Li, Chen Huang, Chun-Liang Li, Eran Malach, Josh Susskind, Vimal Thilak, Etai Littwin

**发布时间:** 2025-09-29

**备注:** Technical Report

### GPT解析

### 总结

本文提出了一种名为SALT(Static-teacher Asymmetric Latent Training)的新方法，用于视频表示学习。该方法通过两阶段训练：首先训练一个教师编码器进行像素重建，然后冻结教师并训练学生模型预测掩码区域的潜在表示。这种方法解耦了优化过程，提高了透明度、效率和可扩展性，同时保持了表示的泛化能力。

### 背景

Video Joint Embedding Predictive Architectures (V-JEPA) 通过指数移动平均(EMA)更新的教师模型预测潜在空间中的掩码区域来学习视频表示。虽然EMA防止了表示崩溃，但使可扩展的模型选择复杂化，并将教师和学生架构耦合在一起。

### 目的

重新审视掩码-潜在预测方法，展示冻结的教师模型就足够，并提出一种更简单、可扩展且计算效率高的替代方案。

### 方法

SALT是一种两阶段、无正则化的方案：第一阶段使用简单的像素重建目标训练目标编码器；第二阶段冻结目标编码器，训练学生模型预测教师模型在掩码区域的潜在表示。这种方法将优化解耦为像素重建(教师)和掩码潜在预测(学生)。

### 主要发现

SALT的学生模型在冻结骨干评估中优于V-JEPA 2编码器；在匹配的预训练计算量下，SALT实现了更高的探测精度；SALT的精度-计算效率Pareto前沿优于V-JEPA；学生质量对教师质量表现出显著的鲁棒性，即使使用小型次优教师也能产生高性能学生。

### 结论

SALT是基于EMA的自蒸馏在视频表示学习中的一个简单、可扩展且计算效率高的替代方案，计算预算分配应优先考虑学生模型。

### 翻译

视频联合嵌入预测架构(V-JEPA)通过指数移动平均(EMA)更新的教师模型预测潜在空间中的掩码区域来学习通用现成视频表示。虽然EMA防止了表示崩溃，但它使可扩展的模型选择复杂化，并将教师和学生架构耦合在一起。我们重新审视了掩码-潜在预测，并证明冻结的教师模型就足够了。具体来说，我们(i)在V-JEPA掩码下使用简单的像素重建目标训练目标编码器，然后(ii)冻结它并训练学生预测教师模型在掩码区域的潜在表示。这导致了一个我们称为SALT(静态教师非对称潜在训练)的两阶段、无正则化方案。SALT将优化解耦为像素重建(教师)和掩码潜在预测(学生)，提高了透明度、效率和可扩展性，同时保持了表示在冻结评估下的泛化能力。经验上，我们的学生模型在跨不同基准的冻结骨干评估中优于最近提出的V-JEPA 2编码器。它们也更具计算优化性：在匹配的预训练FLOPs下，我们的方法实现了更高的探测精度，其扩展曲线在精度-FLOPs帕累托前沿上优于V-JEPA。最后，我们发现学生质量对教师质量表现出显著的鲁棒性：即使是小型、次优的教师也能产生高性能的学生。这表明计算预算分配应该 overwhelmingly favor the student。这些结果将SALT定位为基于EMA的自蒸馏在视频表示学习中的一个简单、可扩展且计算效率高的替代方案。


### 论文摘要

Video Joint Embedding Predictive Architectures (V-JEPA) learn generalizable off-the-shelf video representation by predicting masked regions in latent space with an exponential moving average (EMA)-updated teacher. While EMA prevents representation collapse, it complicates scalable model selection and couples teacher and student architectures. We revisit masked-latent prediction and show that a frozen teacher suffices. Concretely, we (i) train a target encoder with a simple pixel-reconstruction objective under V-JEPA masking, then (ii) freeze it and train a student to predict the teacher's latents on masked regions. This leads to a two-stage, unregularized scheme that we refer to as SALT (Static-teacher Asymmetric Latent Training). SALT decouples optimization into pixel reconstruction (teacher) and masked latent prediction (student), increasing transparency, efficiency, and scalability while preserving the ability of representation to generalize under frozen evaluation. Empirically, our student models outperform recently proposed V-JEPA 2 encoders under frozen backbone evaluation across diverse benchmarks. They are also more compute-optimal: at matched pretraining FLOPs, our method achieves higher probing accuracy, and its scaling curves dominate V-JEPA's accuracy-FLOPs Pareto frontier. Finally, we find that student quality is remarkably robust to teacher quality: high-performing students emerge even with small, sub-optimal teachers. This points to a compute budget allocation that should overwhelmingly favor the student. These results position SALT as a simple, scalable, and compute-efficient alternative to EMA-based self-distillation for video representation learning.

---

## 119. PEARL: Performance-Enhanced Aggregated Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.24312v1](http://arxiv.org/abs/2509.24312v1)

**作者:** Wenhui Li, Shijin Gong, Xinyu Zhang

**发布时间:** 2025-09-29

**备注:** 23 pages, 1 figure, 5 tables

### GPT解析

### 总结

本文提出了一种性能增强的聚合表征学习方法，通过结合多种表征学习技术提高下游任务性能。该方法具有通用性和灵活性，适用于多种损失函数，并能确保计算效率。理论上证明了该方法在下游任务中渐近达到最优性能，实验结果表明其持续优于基线方法。

### 背景

表征学习是现代机器学习的关键技术，能够使模型在复杂数据中识别有意义的模式。然而，不同方法倾向于提取数据的不同方面，仅依赖单一方法可能会忽略与下游任务相关的重要见解。

### 目的

提出一种性能增强的聚合表征学习方法，结合多种表征学习方法提高下游任务性能，设计通用灵活的框架适应各种常见损失函数，并确保计算效率。

### 方法

提出聚合表征学习方法，框架设计为通用灵活，可适应各种常见机器学习损失函数。使用代理损失函数确保计算效率，促进实际权重估计。理论上证明该方法在下游任务中渐近达到最优性能，并会为正确指定的模型赋予非零权重。

### 主要发现

1) 该方法在下游任务中渐近达到最优性能，预测器风险渐近等价于理论最小值；2) 该方法渐近为正确指定的模型分配非零权重；3) 实验结果表明该方法持续优于基线方法。

### 结论

所提出的性能增强的聚合表征学习方法在现实机器学习场景中具有有效性和广泛适用性，通过结合多种表征学习方法提高下游任务性能。

### 翻译

表征学习是现代机器学习中的关键技术，它使模型能够在复杂数据中识别有意义的模式。然而，不同的方法往往提取数据的不同方面，仅依赖单一方法可能会忽略与下游任务相关的重要见解。本文提出了一种性能增强的聚合表征学习方法，它结合了多种表征学习方法以提高下游任务的性能。该框架被设计为通用且灵活的，适用于机器学习模型中常用的各种损失函数。为确保计算效率，我们使用代理损失函数来促进实际的权重估计。从理论上，我们证明了我们的方法在下游任务中渐近地达到最优性能，意味着我们的预测器的风险渐近等价于理论最小值。此外，我们推导出我们的方法会为正确指定的模型分配非零权重。我们通过与先进的机器学习模型进行比较，在各种任务上评估了我们的方法。实验结果表明，我们的方法持续优于基线方法，展示了它在现实机器学习场景中的有效性和广泛适用性。


### 论文摘要

Representation learning is a key technique in modern machine learning that enables models to identify meaningful patterns in complex data. However, different methods tend to extract distinct aspects of the data, and relying on a single approach may overlook important insights relevant to downstream tasks. This paper proposes a performance-enhanced aggregated representation learning method, which combines multiple representation learning approaches to improve the performance of downstream tasks. The framework is designed to be general and flexible, accommodating a wide range of loss functions commonly used in machine learning models. To ensure computational efficiency, we use surrogate loss functions to facilitate practical weight estimation. Theoretically, we prove that our method asymptotically achieves optimal performance in downstream tasks, meaning that the risk of our predictor is asymptotically equivalent to the theoretical minimum. Additionally, we derive that our method asymptotically assigns nonzero weights to correctly specified models. We evaluate our method on diverse tasks by comparing it with advanced machine learning models. The experimental results demonstrate that our method consistently outperforms baseline methods, showing its effectiveness and broad applicability in real-world machine learning scenarios.

---

## 120. Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement

**论文链接:** [http://arxiv.org/abs/2509.24291v1](http://arxiv.org/abs/2509.24291v1)

**作者:** Yu-Che Tsai, Kuan-Yu Chen, Yuan-Chi Li, Yuan-Hao Chen, Ching-Yu Tsai, Shou-De Lin

**发布时间:** 2025-09-29

### GPT解析

### 总结

论文提出了GIRCSE框架，利用自回归生成迭代精炼语义表示，优于传统仅编码器方法的LLM嵌入技术。

### 背景

现有大型语言模型(LLM)嵌入通常采用仅编码器范式，将LLM视为静态特征提取器，忽略了其核心的生成优势。

### 目的

引入GIRCSE(生成式迭代精炼对比句子嵌入)框架，利用自回归生成来迭代精炼语义表示。

### 方法

GIRCSE通过产生在对比目标下优化的软标记序列来工作，提出了迭代对比精炼(ICR)目标函数来指导这个过程，鼓励每个精炼步骤产生更好的表示。

### 主要发现

GIRCSE能够捕获仅编码器方法经常忽略的潜在概念和隐含语义；在MTEB基准和指令跟随任务上优于强大的基于LLM的嵌入基线；表现出测试时扩展特性：在推理时生成更多标记可以持续提高嵌入质量。

### 结论

研究结果确立了生成式迭代精炼作为表征学习的新范式。

### 翻译

现有的大型语言模型(LLM)嵌入通常采用仅编码器范式，将LLM视为静态特征提取器，忽略了其核心的生成优势。我们引入了GIRCSE(生成式迭代精炼对比句子嵌入)，这是一个利用自回归生成来迭代精炼语义表示的新框架。通过产生在对比目标下优化的软标记序列，GIRCSE捕获了仅编码器方法经常忽略的潜在概念和隐含语义。为了指导这一过程，我们提出了一个迭代对比精炼(ICR)目标函数，鼓励每个精炼步骤产生更好的表示。大量实验表明，GIRCSE在MTEB基准和指令跟随任务上优于强大的基于LLM的嵌入基线。此外，GIRCSE表现出测试时扩展特性：在推理时生成更多标记可以持续提高嵌入质量。我们的研究结果确立了生成式迭代精炼作为表征学习的新范式。


### 论文摘要

Existing large language model (LLM)-based embeddings typically adopt an encoder-only paradigm, treating LLMs as static feature extractors and overlooking their core generative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive Sentence Embeddings), a novel framework that leverages autoregressive generation to iteratively refine semantic representations. By producing sequences of soft tokens optimized under contrastive objective, GIRCSE captures latent concepts and implicit semantics that encoder-only methods often miss. To guide this process, we propose an Iterative Contrastive Refinement (ICR) objective that encourages each refinement step to yield better representations. Extensive experiments show that GIRCSE outperforms strong LLM-based embedding baselines on the MTEB benchmark and instruction-following tasks. Moreover, GIRCSE exhibits an emergent test-time scaling property: generating more tokens at inference steadily improves embedding quality. Our results establish generative iterative refinement as a new paradigm for representation learning.

---

## 121. HyMaTE: A Hybrid Mamba and Transformer Model for EHR Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.24118v1](http://arxiv.org/abs/2509.24118v1)

**作者:** Md Mozaharul Mottalib, Thao-Ly T. Phan, Rahmatollah Beheshti

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为HyMaTE的新型混合模型，结合了Mamba和Transformer的优势，用于电子健康记录(EHR)的表示学习。该模型能够有效处理EHR数据的复杂性，包括长序列、多变量、稀疏性和缺失值问题。

### 背景

电子健康记录(EHRs)已成为现代医疗的基石，但其复杂性对传统深度学习模型构成挑战。Transformer模型虽成功但计算复杂度高且上下文长度有限，而状态空间模型如Mamba虽效率高但主要关注序列级信息而非通道级数据。

### 目的

开发一种能够克服现有模型限制的新型混合模型，有效表示纵向EHR数据，捕获更丰富且更细微的表示。

### 方法

提出HyMaTE（混合Mamba和Transformer模型），结合状态空间模型和高级注意力机制的优势，在多个临床数据集的预测任务上进行测试。

### 主要发现

HyMaTE能够捕获EHR数据的有效、更丰富且更细微的统一表示，自注意力实现的解释性证明了模型的有效性。

### 结论

HyMaTE是处理EHR数据的有前途方法，可作为真实医疗保健应用的可扩展和通用解决方案。

### 翻译

电子健康记录(EHRs)已成为现代医疗的基石。它们是分析患者健康状况进展的关键部分；然而，其复杂性表现为长序列、多变量、稀疏性和缺失值，给传统深度学习建模带来了重大挑战。虽然基于Transformer的模型在建模EHR数据和预测临床结果方面已显示出成功，但其二次计算复杂性和有限的上下文长度限制了它们的效率和实际应用。另一方面，状态空间模型(SSMs)如Mamba提供了一种有前途的替代方案，提供线性时间序列建模和更好的处理长序列的效率，但主要关注混合序列级信息而非通道级数据。为了克服这些挑战，我们提出了HyMaTE（一种用于EHR表示学习的混合Mamba和Transformer模型），这是一种专为表示纵向数据设计的新型混合模型，结合了SSMs和高级注意力机制的优势。通过在多个临床数据集的预测任务上测试该模型，我们证明了HyMaTE能够捕获EHR数据的有效、更丰富且更细微的统一表示。此外，通过自注意力实现的结果解释性说明了我们的模型作为真实医疗保健应用的可扩展和通用解决方案的有效性。代码可在以下网址获取：https://github.com/healthylaife/HyMaTE。


### 论文摘要

Electronic health Records (EHRs) have become a cornerstone in modern-day healthcare. They are a crucial part for analyzing the progression of patient health; however, their complexity, characterized by long, multivariate sequences, sparsity, and missing values poses significant challenges in traditional deep learning modeling. While Transformer-based models have demonstrated success in modeling EHR data and predicting clinical outcomes, their quadratic computational complexity and limited context length hinder their efficiency and practical applications. On the other hand, State Space Models (SSMs) like Mamba present a promising alternative offering linear-time sequence modeling and improved efficiency for handling long sequences, but focus mostly on mixing sequence-level information rather than channel-level data. To overcome these challenges, we propose HyMaTE (A Hybrid Mamba and Transformer Model for EHR Representation Learning), a novel hybrid model tailored for representing longitudinal data, combining the strengths of SSMs with advanced attention mechanisms. By testing the model on predictive tasks on multiple clinical datasets, we demonstrate HyMaTE's ability to capture an effective, richer, and more nuanced unified representation of EHR data. Additionally, the interpretability of the outcomes achieved by self-attention illustrates the effectiveness of our model as a scalable and generalizable solution for real-world healthcare applications. Codes are available at: https://github.com/healthylaife/HyMaTE.

---

## 122. Joint Superpixel and Self-Representation Learning for Scalable Hyperspectral Image Clustering

**论文链接:** [http://arxiv.org/abs/2509.24027v1](http://arxiv.org/abs/2509.24027v1)

**作者:** Xianlu Li, Nicolas Nadisic, Shaoguang Huang, Aleksandra Pizurica

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种统一的端到端框架，联合优化超像素分割和子空间聚类，通过反馈机制实现聚类感知的分区，在高光谱图像分析中取得了优异的性能。

### 背景

子空间聚类是一种强大的无监督高光谱图像分析方法，但其高计算和内存成本限制了可扩展性。超像素分割可通过减少处理数据点数量提高效率，但现有方法通常独立于聚类任务执行分割，导致分区与后续聚类目标不一致。

### 目的

解决超像素分割与子空间聚类任务之间的不协调问题，提出一个联合优化两者的端到端框架，实现聚类感知的分区。

### 方法

框架核心是反馈机制：基于展开的交替方向乘子法的自表示网络提供模型驱动信号指导可微分超像素模块。联合优化产生聚类感知分区，同时保持光谱和空间结构。超像素网络为每个超像素学习独特紧凑度参数，实现更灵活自适应的分割。

### 主要发现

联合优化超像素分割和子空间聚类可提高聚类性能；为每个超像素学习独特紧凑度参数可实现更灵活自适应的分割；所提方法能保持光谱和空间结构。

### 结论

在基准高光谱数据集上的大量实验表明，该方法始终优于最先进的聚类方法。

### 翻译

子空间聚类是一种强大的无监督高光谱图像分析方法，但其高昂的计算和内存成本限制了可扩展性。超像素分割可以通过减少需要处理的数据点数量来提高效率。然而，现有的基于超像素的方法通常独立于聚类任务执行分割，经常产生与后续聚类目标不一致的分区。为此，我们提出一个统一的端到端框架，联合优化超像素分割和子空间聚类。其核心是一个反馈机制：基于展开的交替方向乘子法的自表示网络提供模型驱动信号来指导可微分超像素模块。这种联合优化产生聚类感知的分区，同时保持光谱和空间结构。此外，我们的超像素网络为每个超像素学习独特的紧凑度参数，实现更灵活和自适应的分割。在基准高光谱数据集上的大量实验表明，我们的方法始终优于最先进的聚类方法。


### 论文摘要

Subspace clustering is a powerful unsupervised approach for hyperspectral image (HSI) analysis, but its high computational and memory costs limit scalability. Superpixel segmentation can improve efficiency by reducing the number of data points to process. However, existing superpixel-based methods usually perform segmentation independently of the clustering task, often producing partitions that do not align with the subsequent clustering objective. To address this, we propose a unified end-to-end framework that jointly optimizes superpixel segmentation and subspace clustering. Its core is a feedback mechanism: a self-representation network based on unfolded Alternating Direction Method of Multipliers (ADMM) provides a model-driven signal to guide a differentiable superpixel module. This joint optimization yields clustering-aware partitions that preserve both spectral and spatial structure. Furthermore, our superpixel network learns a unique compactness parameter for each superpixel, enabling more flexible and adaptive segmentation. Extensive experiments on benchmark HSI datasets demonstrate that our method consistently achieves superior accuracy compared with state-of-the-art clustering approaches.

---

## 123. ResAD++: Towards Class Agnostic Anomaly Detection via Residual Feature Learning

**论文链接:** [http://arxiv.org/abs/2509.23741v1](http://arxiv.org/abs/2509.23741v1)

**作者:** Xincheng Yao, Chao Shi, Muming Zhao, Guangtao Zhai, Chongyang Zhang

**发布时间:** 2025-09-28

**备注:** This paper is an extended version of our NeurIPS 2024 paper, ResAD.  arXiv admin note: substantial text overlap with arXiv:2410.20047

### GPT解析

### 总结

本文提出了一种名为ResAD的类无关异常检测方法，通过残差特征和特征超球约束提高模型在新类别上的泛化能力，并进一步改进为ResAD++，在多个数据集上取得了优于现有方法的结果。

### 背景

当前的单类和多类异常检测方法在应用于新类别时表现仍然不令人满意。一个根本原因是现有方法中的表示学习仍然与类别相关，即特征相关性。

### 目的

训练一个类无关的异常检测模型，能够泛化到检测不同领域中各种新类别的异常，而无需在目标数据上进行重新训练或微调。

### 方法

1. 提出残差特征，通过匹配和减去正常参考特征形成，实现特征去相关；2. 提出特征超球约束方法，将初始正常残差特征约束到空间超球中，使不同类别的特征尺度尽可能一致；3. 提出对数屏障双向收缩OCC损失和基于向量量化的特征分布匹配模块，增强ResAD，改进为ResAD++。

### 主要发现

残差特征分布比初始特征分布更稳定，即使在新的类别中，正常残差特征的分布也不会明显偏离已学习的分布。

### 结论

ResAD++在八个真实世界的异常检测数据集上取得了显著的异常检测结果，直接用于新类别时优于最先进的竞争方法，也超过了ResAD。

### 翻译

这篇论文探讨了类无关异常检测(AD)问题，其目标是训练一个类无关的AD模型，能够泛化到检测来自不同领域的各种新类别的异常，而无需在目标数据上进行重新训练或微调。当应用于新类别时，当前的单类和多类AD方法的性能仍然不令人满意。一个根本原因是现有方法中的表示学习仍然与类别相关，即特征相关性。为了解决这个问题，我们提出了残差特征，并构建了一个简单但有效的框架，称为ResAD。我们的核心见解是学习残差特征分布而非初始特征分布。残差特征是通过匹配然后减去正常参考特征形成的。通过这种方式，我们可以有效地实现特征去相关。即使在新的类别中，正常残差特征的分布也不会明显偏离已学习的分布。此外，我们认为残差特征仍然存在一个问题：尺度相关性。为此，我们提出了一种特征超球约束方法，学习将初始正常残差特征约束到空间超球中，使不同类别的特征尺度尽可能一致。此外，我们提出了一种新颖的对数屏障双向收缩OCC损失和基于向量量化的特征分布匹配模块来增强ResAD，改进版的ResAD(ResAD++)。在八个真实世界的AD数据集上的综合实验表明，我们的ResAD++在直接用于新类别时能够取得显著的AD结果，优于最先进的竞争方法，也超过了ResAD。代码可在https://github.com/xcyao00/ResAD获取。


### 论文摘要

This paper explores the problem of class-agnostic anomaly detection (AD), where the objective is to train one class-agnostic AD model that can generalize to detect anomalies in diverse new classes from different domains without any retraining or fine-tuning on the target data. When applied for new classes, the performance of current single- and multi-class AD methods is still unsatisfactory. One fundamental reason is that representation learning in existing methods is still class-related, namely, feature correlation. To address this issue, we propose residual features and construct a simple but effective framework, termed ResAD. Our core insight is to learn the residual feature distribution rather than the initial feature distribution. Residual features are formed by matching and then subtracting normal reference features. In this way, we can effectively realize feature decorrelation. Even in new classes, the distribution of normal residual features would not remarkably shift from the learned distribution. In addition, we think that residual features still have one issue: scale correlation. To this end, we propose a feature hypersphere constraining approach, which learns to constrain initial normal residual features into a spatial hypersphere for enabling the feature scales of different classes as consistent as possible. Furthermore, we propose a novel logbarrier bidirectional contraction OCC loss and vector quantization based feature distribution matching module to enhance ResAD, leading to the improved version of ResAD (ResAD++). Comprehensive experiments on eight real-world AD datasets demonstrate that our ResAD++ can achieve remarkable AD results when directly used in new classes, outperforming state-of-the-art competing methods and also surpassing ResAD. The code is available at https://github.com/xcyao00/ResAD.

---

## 124. Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action models

**论文链接:** [http://arxiv.org/abs/2509.23655v1](http://arxiv.org/abs/2509.23655v1)

**作者:** Rokas Bendikas, Daniel Dijkman, Markus Peschl, Sanjay Haresh, Pietro Mazzaglia

**发布时间:** 2025-09-28

**备注:** Presented at 9th Conference on Robot Learning (CoRL 2025), Seoul,  Korea

### GPT解析

### 总结

本研究提出了一种名为Oat-VLA的高效Vision-Language-Action模型训练方法，通过减少视觉标记数量显著降低了计算成本，同时保持了或提高了性能。

### 背景

Vision-Language-Action (VLA)模型通过重新利用大型预训练Vision-Language-Models (VLM)来输出机器人动作，是学习机器人操作的关键方法。然而，将VLMs适应机器人领域带来了不必要的高计算成本，这主要归因于视觉输入的标记化方案。

### 目的

提出Oat-VLA（一种面向VLA的以对象-智能体为中心的标记化方法），旨在实现高效的VLA训练。

### 方法

基于以对象为中心的表征学习见解，Oat-VLA方法引入了一种偏向场景对象和智能体自身视觉信息的归纳偏置，从而优化视觉标记过程。

### 主要发现

Oat-VLA可以将视觉标记数量大幅减少到仅几个标记而不牺牲性能；在LIBERO套件上的收敛速度至少比OpenVLA快两倍；在多样化的现实世界抓取和放置任务中表现优于OpenVLA。

### 结论

Oat-VLA通过以对象-智能体为中心的标记化方案，显著提高了VLA模型的训练效率，同时保持了或提高了模型在机器人任务中的表现。

### 翻译

视觉-语言-动作（VLA）模型通过重新利用大型预训练视觉-语言模型（VLM）来输出机器人动作，为大规模学习机器人操作提供了关键方法。然而，将VLMs适应机器人领域会带来不必要的高计算成本，我们认为这是由于视觉输入的标记化方案导致的。在本研究中，我们通过提出Oat-VLA（一种面向VLA的以对象-智能体为中心的标记化方法），旨在实现高效的VLA训练。基于以对象为中心的表征学习的见解，我们的方法引入了一种偏向场景对象和智能体自身视觉信息的归纳偏置。因此，我们发现Oat-VLA可以将视觉标记数量大幅减少到仅几个标记而不会牺牲性能。我们揭示Oat-VLA在LIBERO套件上的收敛速度至少比OpenVLA快两倍，并且在多样化的现实世界抓取和放置任务中表现优于OpenVLA。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决Vision-Language-Action (VLA) 模型训练中的计算效率问题。当前VLA模型处理视觉输入时，通常将图像分割成大量小块并生成大量视觉标记，导致计算成本高昂、训练时间长。这个问题很重要，因为它限制了VLA模型在资源有限的研究实验室中的应用，阻碍了机器人技术的发展，而现实世界需要高效、可扩展的机器人解决方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出当前VLA模型的主要瓶颈在于视觉标记化过程，然后认识到执行任务只需关注场景中的特定部分(如感兴趣对象)，背景信息可被忽略。基于物体中心化表示学习的见解，设计了自适应标记化重要部分的方法。作者借鉴了物体中心化表示学习(使用FT-Dinosaur模型)、视觉语言模型的标记化策略，并考虑了与OpenVLA架构的兼容性以重用预训练知识。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过关注场景中的物体和机器人自身信息，而非处理整个图像，大幅减少视觉标记数量同时保留关键信息。流程分为三步：1)物体中心化标记提取 - 使用物体分割模型获取物体掩码，对每个物体的视觉标记进行池化；2)智能体中心化标记提取 - 检测机器人夹持器位置，提取周围区域标记；3)标记整合 - 将两种标记结合并通过MLP投影后输入LLM进行动作预测。对于224×224图像，Oat-VLA仅使用16个标记(原256个)，减少93.75%。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)结合物体和智能体中心化标记化；2)大幅减少视觉标记数量(93.75%)；3)设计高效训练架构可重用现有VLA检查点；4)提出不依赖机器人标定的通用夹持器检测方法。不同之处：与通用标记减少方法不同，Oat-VLA利用任务特定先验；与仅关注物体的方法不同，Oat-VLA特别关注智能体操作信息；与使用SAM的方法不同，Oat-VLA采用无监督物体分割；与OpenVLA相比，保留了架构兼容性但从根本上改变了视觉标记化方式；与小型VLA不同，Oat-VLA通过智能标记化而非减小模型规模提高效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Oat-VLA通过创新的物体-智能体中心化标记化方法，将视觉标记数量减少93.75%，实现了VLA模型训练速度提升2倍以上，同时保持或提高了任务性能，使更多研究实验室能够高效训练机器人视觉-语言-动作模型。'}


### 论文摘要

Vision-Language-Action (VLA) models offer a pivotal approach to learning robotic manipulation at scale by repurposing large pre-trained Vision-Language-Models (VLM) to output robotic actions. However, adapting VLMs for robotic domains comes with an unnecessarily high computational cost, which we attribute to the tokenization scheme of visual inputs. In this work, we aim to enable efficient VLA training by proposing Oat-VLA, an Object-Agent-centric Tokenization for VLAs. Building on the insights of object-centric representation learning, our method introduces an inductive bias towards scene objects and the agent's own visual information. As a result, we find that Oat-VLA can drastically reduce the number of visual tokens to just a few tokens without sacrificing performance. We reveal that Oat-VLA converges at least twice as fast as OpenVLA on the LIBERO suite, as well as outperform OpenVLA in diverse real-world pick and place tasks.

---

## 125. Channel, Trend and Periodic-Wise Representation Learning for Multivariate Long-term Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2509.23583v1](http://arxiv.org/abs/2509.23583v1)

**作者:** Zhangyao Song, Nanqing Jiang, Miaohong He, Xiaoyu Zhao, Tao Guo

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了CTPNet框架，解决了下采样方法在时间序列预测中忽略子序列间和通道间相互作用的问题，通过整合三个层次的依赖关系提高了预测准确性。

### 背景

下采样方法在时间序列预测中受到关注，因为它们在捕捉序列趋势方面具有优势。

### 目的

提出一个名为CTPNet的新框架，解决下采样方法主要捕获子序列内依赖关系而忽略子序列间和通道间相互作用的问题，以提高预测准确性。

### 方法

CTPNet从三个角度学习表示：1) 通过基于时间查询的多头注意力机制捕获通道间依赖关系；2) 使用Transformer建模子序列内依赖关系以表征趋势变化；3) 通过重用编码器与残差连接提取子序列间依赖关系和全局周期模式。

### 主要发现

通过联合整合这三个层次的依赖关系，所提出的方法提供了时间动态的更整体表示。

### 结论

大量实验证明了CTPNet方法的优越性。

### 翻译

基于下采样的时间序列预测方法因其捕捉序列趋势的优势而受到越来越多的关注。然而，这些方法主要捕获子序列内的依赖关系，而忽略了子序列间和通道间的相互作用，这限制了预测准确性。为了解决这些局限性，我们提出了CTPNet，一个新颖的框架，从三个角度明确学习表示：i) 通过基于时间查询的多头注意力机制捕获的通道间依赖关系；ii) 通过Transformer建模的子序列内依赖关系，用于表征趋势变化；iii) 通过重用编码器与残差连接提取的子序列间依赖关系，用于捕获全局周期模式。通过联合整合这些层次，所提出的方法提供了时间动态的更整体表示。大量实验证明了所提出方法的优越性。


### 论文摘要

Downsampling-based methods for time series forecasting have attracted increasing attention due to their superiority in capturing sequence trends. However, this approaches mainly capture dependencies within subsequences but neglect inter-subsequence and inter-channel interactions, which limits forecasting accuracy. To address these limitations, we propose CTPNet, a novel framework that explicitly learns representations from three perspectives: i) inter-channel dependencies, captured by a temporal query-based multi-head attention mechanism; ii) intra-subsequence dependencies, modeled via a Transformer to characterize trend variations; and iii) inter-subsequence dependencies, extracted by reusing the encoder with residual connections to capture global periodic patterns. By jointly integrating these levels, proposed method provides a more holistic representation of temporal dynamics. Extensive experiments demonstrate the superiority of the proposed method.

---

## 126. Dyadic Neural Dynamics: Extending Representation Learning to Social Neuroscience

**论文链接:** [http://arxiv.org/abs/2509.23479v1](http://arxiv.org/abs/2509.23479v1)

**作者:** Maria Glushanina, Jeffrey Huang, Michelle McCleod, Brendan Ames, Evie Malaia

**发布时间:** 2025-09-27

### GPT解析

### 总结

该研究首次将CEBRA方法应用于二人脑电图超扫描数据，用于学习社交互动中的联合神经活动表示，捕捉个体角色和行为指标，为社交神经科学和临床应用开辟新方向。

### 背景

社交交流涉及至少两个相互作用的脑，构成了独特的建模问题。传统方法主要关注个体对刺激的神经反应，而非人际互动中的神经动力学。

### 目的

扩展建模范式以人际神经动力学，开发能够捕捉社交互动中个体角色和行为指标的方法，并解决基础模型开发中的可扩展性和跨主题泛化问题。

### 方法

应用CEBRA（Contrastive Embedding for Behavioral and Neural Analysis）方法分析二人脑电图超扫描数据，通过结构化社交互动学习联合神经活动的表示。

### 主要发现

CEBRA能够学习有意义的联合神经活动表示，有效捕捉个体角色（如说话者-倾听者）和其他行为指标，展示了良好的可扩展性和跨主题泛化能力。

### 结论

关注人际互动而非个体神经反应的方法，解决了基础模型开发的关键原则，为社交神经科学和临床应用中的表示学习开辟了新方向。

### 翻译

社交交流基本上涉及至少两个相互作用的脑，构成了独特的建模问题。我们首次将用于行为和神经分析的对比嵌入（CEBRA）应用于二人脑电图超扫描数据，将建模范式扩展到人际神经动力学。利用参与者之间的结构化社交互动，我们证明CEBRA可以学习有意义的联合神经活动表示，捕捉个体角色（说话者-倾听者）和其他行为指标。我们描述互动的方法，与对刺激的个体神经反应相反，解决了基础模型开发的关键原则：可扩展性和跨主题泛化，为社交神经科学和临床应用中的表示学习开辟了新方向。


### 论文摘要

Social communication fundamentally involves at least two interacting brains, creating a unique modeling problem. We present the first application of Contrastive Embedding for Behavioral and Neural Analysis (CEBRA) to dyadic EEG hyperscanning data, extending modeling paradigms to interpersonal neural dynamics. Using structured social interactions between participants, we demonstrate that CEBRA can learn meaningful representations of joint neural activity that captures individual roles (speaker-listener) and other behavioral metrics. Our approach to characterizing interactions, as opposed to individual neural responses to stimuli, addresses the key principles of foundational model development: scalability and cross-subject generalization, opening new directions for representation learning in social neuroscience and clinical applications.

---

## 127. LLM Interpretability with Identifiable Temporal-Instantaneous Representation

**论文链接:** [http://arxiv.org/abs/2509.23323v1](http://arxiv.org/abs/2509.23323v1)

**作者:** Xiangchen Song, Jiaqi Sun, Zijian Li, Yujia Zheng, Kun Zhang

**发布时间:** 2025-09-27

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一种专门针对大型语言模型的高维概念空间设计的可识别时间因果表征学习框架，成功捕获了时间延迟和瞬时因果关系，提高了LLM的可解释性。

### 背景

大型语言模型具有显著能力但内部表征理解困难；现有机制可解释性工具如稀疏自编码器(SAEs)缺乏时间依赖建模、瞬时关系表示和理论保证；因果表征学习方法(CRL)理论基础扎实但无法扩展到LLM的丰富概念空间。

### 目的

弥合现有方法与LLM需求之间的差距，开发一个能够捕获时间延迟和瞬时因果关系、具有理论保证的可识别时间因果表征学习框架。

### 方法

设计专门针对LLM高维概念空间的可识别时间因果表征学习框架；将SAE技术与时间因果框架相结合；在扩展到匹配真实世界复杂性的合成数据集上验证方法有效性。

### 主要发现

通过结合SAE技术与时间因果框架，成功发现了LLM激活中的有意义概念关系；建模时间和瞬时概念关系能够提高LLM的可解释性。

### 结论

建模时间和瞬时概念关系有助于推进LLM的可解释性；所提框架提供了理论保证并在复杂数据集上证明了有效性。

### 翻译

尽管大型语言模型具有显著能力，但理解其内部表征仍然具有挑战性。诸如稀疏自编码器(SAEs)之类的机制可解释性工具被开发用于从LLM中提取可解释的特征，但缺乏时间依赖建模、瞬时关系表示，更重要的是缺乏理论保证，这削弱了后续分析所需的理论基础和实践信心。虽然因果表征学习(CRL)为揭示潜在概念提供了理论基础扎实的方法，但现有方法由于计算效率低下而无法扩展到LLM的丰富概念空间。为了弥合这一差距，我们引入了一个专门为LLM高维概念空间设计的可识别时间因果表征学习框架，同时捕获时间延迟和瞬时因果关系。我们的方法提供了理论保证，并在扩展到匹配真实世界复杂性的合成数据集上证明了有效性。通过将SAE技术与我们的时间因果框架相结合，我们成功发现了LLM激活中的有意义概念关系。我们的研究表明，建模时间和瞬时概念关系能够提高LLM的可解释性。


### 论文摘要

Despite Large Language Models' remarkable capabilities, understanding their internal representations remains challenging. Mechanistic interpretability tools such as sparse autoencoders (SAEs) were developed to extract interpretable features from LLMs but lack temporal dependency modeling, instantaneous relation representation, and more importantly theoretical guarantees, undermining both the theoretical foundations and the practical confidence necessary for subsequent analyses. While causal representation learning (CRL) offers theoretically grounded approaches for uncovering latent concepts, existing methods cannot scale to LLMs' rich conceptual space due to inefficient computation. To bridge the gap, we introduce an identifiable temporal causal representation learning framework specifically designed for LLMs' high-dimensional concept space, capturing both time-delayed and instantaneous causal relations. Our approach provides theoretical guarantees and demonstrates efficacy on synthetic datasets scaled to match real-world complexity. By extending SAE techniques with our temporal causal framework, we successfully discover meaningful concept relationships in LLM activations. Our findings show that modeling both temporal and instantaneous conceptual relationships advances the interpretability of LLMs.

---

## 128. WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms

**论文链接:** [http://arxiv.org/abs/2509.23238v1](http://arxiv.org/abs/2509.23238v1)

**作者:** Goksenin Yuksel, Pierre Guetschel, Michael Tangermann, Marcel van Gerven, Kiki van der Heijden

**发布时间:** 2025-09-27

**备注:** Still under review

### GPT解析

### 总结

该研究提出了WavJEPA，一种基于原始波形的联合嵌入预测架构，用于通用音频表示学习，并展示了其在多种下游任务上的优越性能和计算效率。

### 背景

从原始波形学习音频表示可克服基于频谱图方法的局限性，如计算延迟和相位信息丢失。尽管基于原始波形的自监督语音表示学习已成功，但通用音频表示学习尚未取得类似进展。

### 目的

提出一种基于原始波形的通用音频表示学习方法，克服语音单元或标记级别的表示学习局限性，开发低延迟、鲁棒的时间域音频基础模型。

### 方法

提出WavJEPA，一种基于波形的联合嵌入预测架构版本，利用高级语义表示学习；进一步提出WavJEPA-Nat，这是WavJEPA的多通道扩展，在模拟自然场景上训练以提高在嘈杂和混响环境中的鲁棒性。

### 主要发现

WavJEPA在多种下游基准任务上显著优于最先进的时间域音频基础模型，同时需要更少的计算资源；WavJEPA-Nat对混响和噪声具有很高的鲁棒性。

### 结论

从原始波形进行通用音频表示学习是可行且计算高效的，低延迟、鲁棒的时间域音频基础模型在现实世界应用中具有潜力。

### 翻译

从原始波形学习音频表示克服了基于频谱图的音频表示学习的关键局限性，如频谱图计算的长时间延迟和相位信息的丢失。然而，尽管从原始波形进行自监督语音表示学习已取得显著成功，但这些方法尚未在从波形进行通用音频表示学习方面取得类似的成就。在此，我们提出了WavJEPA，一种基于波形的联合嵌入预测架构版本。WavJEPA利用高级语义表示学习来解决语音单元或标记级别的表示学习的缺点。我们证明这种方法在多种下游基准任务上显著优于最先进的时间域音频基础模型，同时需要更少的计算资源。此外，为了克服时间域模型在嘈杂和混响的真实世界声学环境中通常表现出的性能下降，我们提出了WavJEPA-Nat。WavJEPA-Nat是WavJEPA架构的多通道扩展，在模拟自然场景上训练。我们发现WavJEPA-Nat对混响和噪声具有很高的鲁棒性。这些结果突显了从原始波形进行通用音频表示学习的可行性和计算效率，展示了低延迟、鲁棒的时间域音频基础模型在现实世界应用中的潜力。


### 论文摘要

Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.

---

## 129. GLUE: Global-Local Unified Encoding for Imitation Learning via Key-Patch Tracking

**论文链接:** [http://arxiv.org/abs/2509.23220v1](http://arxiv.org/abs/2509.23220v1)

**作者:** Ye Chen, Zichen Zhou, Jianyu Dou, Te Cui, Yi Yang, Yufeng Yue

**发布时间:** 2025-09-27

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

GLUE是一种基于关键块跟踪的全局-局部统一编码框架，通过文本引导机制选择和跟踪关键块作为局部表征，并采用新颖的融合框架结合全局和局部特征，解决了复杂分布外环境下的视觉表征学习问题，显著提高了机器人模仿学习的性能和鲁棒性。

### 背景

视觉表征学习在机器人模仿学习中受到广泛关注，但在复杂分布外环境下，如杂乱和遮挡场景，全局视觉表征的关注度会被稀释或干扰，导致策略性能下降。

### 目的

提出一种解决方案，利用局部表征的不变性来应对相关任务对象，通过有效利用这些局部表征，将训练和测试数据映射到更相似的特征空间，从而缓解协变量偏移问题。

### 方法

提出了GLUE框架，采用文本引导机制选择和跟踪关键块作为关键局部表征；设计了一种新颖的融合框架，其中全局块特征查询局部块以提取关键信息，生成相对于全局上下文具有低异质性的细粒度局部特征。

### 主要发现

融合表征使机器人的视觉关注转向任务相关对象，保留精确的全局上下文，将训练和测试分布对齐到相似且具有任务信息的特征空间，最终增强了模仿学习策略的鲁棒性。

### 结论

实验证明GLUE在模拟和现实世界环境中的多样化任务上取得了强大性能，在模拟环境中比最强基线高出17.6%，在现实环境中比最强基线高出36.3%，在现实世界泛化设置上比最强基线高出58.3%。

### 翻译

近年来，视觉表征学习在机器人模仿学习中获得了广泛关注。然而，在由杂乱和遮挡表征的复杂分布外设置中，全局视觉表征的关注可能会被稀释或干扰，导致策略性能下降。任务相关对象的局部表征的不变性提供了一种解决方案。通过有效利用这些局部表征，训练和测试数据可以映射到更相似的特征空间，从而缓解协变量偏移问题。因此，我们提出了GLUE，一种基于关键块跟踪的模仿学习全局-局部统一编码框架。GLUE采用文本引导机制选择和跟踪关键块作为关键局部表征。它具有一种新颖的融合框架，其中全局块特征查询局部块以提取关键信息，产生相对于全局上下文具有低异质性的细粒度局部特征。这种融合表征使机器人的视觉关注转向任务相关对象，并保留精确的全局上下文，将训练和测试分布对齐到相似且具有任务信息的特征空间，最终增强了模仿学习策略的鲁棒性。实验证明，GLUE在模拟和现实世界环境中的多样化任务上取得了强大性能，在模拟环境中比最强基线高出17.6%，在现实环境中比最强基线高出36.3%，在现实世界泛化设置上比最强基线高出58.3%。GLUE的项目网站可在https://GLUE666.github.io/获取。


### 论文摘要

In recent years, visual representation learning has gained widespread attention in robotic imitation learning. However, in complex Out-of-Distribution(OOD) settings characterized by clutter and occlusion, the attention of global visual representations can be diluted or interfered, leading to degraded policy performance. The invariance of local representations for task-relevant objects offers a solution. By efficiently utilizing these local representations, training and testing data can be mapped to a more similar feature space, thereby mitigating the covariate shift problem. Accordingly, we propose GLUE, a global-local unified encoding framework for imitation learning based on key-patch tracking. GLUE selects and tracks key-patches as critical local representations by employing a text-guided mechanism. It features a novel fusion framework where global patch features query local patches to distill essential information, yielding fine-grained local features with low heterogeneity relative to the global context. This fused representation steers the robot's visual attention toward task-relevant objects and preserves precise global context, which together align the training and testing distributions into a similar and task-informative feature space, ultimately enhancing the robustness of the imitation learning policy. Experiments demonstrate that GLUE achieves strong performance across diverse tasks in both simulation and real-world settings, outperforming the strongest baseline by 17.6% in simulation, 36.3% in real-world environments, and 58.3% on real-world generalization settings. The project website of GLUE is available at https://GLUE666.github.io/.

---

## 130. Mask What Matters: Controllable Text-Guided Masking for Self-Supervised Medical Image Analysis

**论文链接:** [http://arxiv.org/abs/2509.23054v1](http://arxiv.org/abs/2509.23054v1)

**作者:** Ruilang Wang, Shuotong Xu, Bowen Liu, Runlin Huang, Donglong Chen, Weifeng Su

**发布时间:** 2025-09-27

### GPT解析

### 总结

本研究提出'Mask What Matters'框架，一种可控的文本引导掩码方法，用于医学图像分析的自监督学习。该方法通过视觉语言模型进行基于提示的区域定位，对诊断相关区域应用差异化掩码，同时减少背景区域的冗余，实现了更好的语义对齐、改进的表示学习和跨任务泛化能力，并在多种医学成像模态上优于现有方法。

### 背景

在医学成像等专业领域中，标注数据的稀缺对训练鲁棒视觉模型构成重大挑战。虽然自监督掩码图像建模（MIM）提供了有前景的解决方案，但现有方法主要依赖随机高比例掩码，导致效率低下和语义对齐不良。此外，区域感知变体通常依赖于重建启发式方法或监督信号，限制了它们在任务和模态间的适应性。

### 目的

开发一种可控的文本引导掩码框架，用于自监督医学图像分析，以解决现有MIM方法在效率、语义对齐和跨任务泛化方面的问题。

### 方法

提出'Mask What Matters'，一种可控的文本引导掩码框架。该方法利用视觉语言模型进行基于提示的区域定位，灵活地对强调的诊断相关区域应用差异化掩码，同时减少背景区域的冗余。这种可控设计实现了更好的语义对齐、改进的表示学习和更强的跨任务泛化能力。

### 主要发现

在脑部MRI、胸部CT和肺部X光等多种医学成像模态上的全面评估表明，'Mask What Matters'持续优于现有的MIM方法（如SparK），在分类准确率上提高最多3.1个百分点，在检测的框平均精度上提高1.3，在掩码平均精度上提高1.1。值得注意的是，它在总体掩码比例低得多的情况下（例如40%对比70%）实现了这些改进。

### 结论

这项研究证明了可控的文本驱动掩码可以实现语义对齐的自监督学习，推动医学图像分析鲁棒视觉模型的发展。

### 翻译

医学成像等专业领域标注数据的稀缺对训练鲁棒视觉模型构成了重大挑战。虽然自监督掩码图像建模（MIM）提供了有前景的解决方案，但现有方法主要依赖随机高比例掩码，导致效率低下和语义对齐不良。此外，区域感知变体通常依赖于重建启发式方法或监督信号，限制了它们在任务和模态间的适应性。我们提出了'Mask What Matters'，一种用于自监督医学图像分析的可控文本引导掩码框架。通过利用视觉语言模型进行基于提示的区域定位，我们的方法灵活地对强调的诊断相关区域应用差异化掩码，同时减少背景区域的冗余。这种可控设计实现了更好的语义对齐、改进的表示学习和更强的跨任务泛化能力。在脑部MRI、胸部CT和肺部X光等多种医学成像模态上的全面评估表明，'Mask What Matters'持续优于现有的MIM方法（如SparK），在分类准确率上提高最多3.1个百分点，在检测的框平均精度上提高1.3，在掩码平均精度上提高1.1。值得注意的是，它在总体掩码比例低得多的情况下（例如40%对比70%）实现了这些改进。这项工作证明了可控的文本驱动掩码可以实现语义对齐的自监督学习，推动医学图像分析鲁棒视觉模型的发展。


### 论文摘要

The scarcity of annotated data in specialized domains such as medical imaging presents significant challenges to training robust vision models. While self-supervised masked image modeling (MIM) offers a promising solution, existing approaches largely rely on random high-ratio masking, leading to inefficiency and poor semantic alignment. Moreover, region-aware variants typically depend on reconstruction heuristics or supervised signals, limiting their adaptability across tasks and modalities. We propose Mask What Matters, a controllable text-guided masking framework for self-supervised medical image analysis. By leveraging vision-language models for prompt-based region localization, our method flexibly applies differentiated masking to emphasize diagnostically relevant regions while reducing redundancy in background areas. This controllable design enables better semantic alignment, improved representation learning, and stronger cross-task generalizability. Comprehensive evaluation across multiple medical imaging modalities, including brain MRI, chest CT, and lung X-ray, shows that Mask What Matters consistently outperforms existing MIM methods (e.g., SparK), achieving gains of up to +3.1 percentage points in classification accuracy, +1.3 in box average precision (BoxAP), and +1.1 in mask average precision (MaskAP) for detection. Notably, it achieves these improvements with substantially lower overall masking ratios (e.g., 40\% vs. 70\%). This work demonstrates that controllable, text-driven masking can enable semantically aligned self-supervised learning, advancing the development of robust vision models for medical image analysis.

---

## 131. Understanding Catastrophic Interference On the Identifibility of Latent Representations

**论文链接:** [http://arxiv.org/abs/2509.23027v1](http://arxiv.org/abs/2509.23027v1)

**作者:** Yuke Li, Yujia Zheng, Tianyi Xiong, Zhenyi Wang, Heng Huang

**发布时间:** 2025-09-27

### GPT解析

### 总结

论文从潜在表示学习角度建模灾难性干扰问题，提出新理论框架将其表述为识别问题，并开发名为\ourmeos的方法采用两阶段训练策略来学习共享潜在变量，有效减轻机器学习系统中的灾难性干扰。

### 背景

灾难性干扰（也称为灾难性遗忘）是机器学习中的基本挑战，指训练好的模型在适应新任务时逐渐失去在先前学习任务上的性能表现。

### 目的

从潜在表示学习的角度更好地理解和建模灾难性干扰问题，并提出一种新的理论框架来解决这一问题。

### 方法

提出名为\ourmeos的方法，采用两阶段训练策略：首先使用最大似然估计从部分任务感知（PTA）和全任务感知（ATA）配置中学习潜在表示；随后优化KL散度以识别和学习共享的潜在变量。

### 主要发现

遗忘现象可通过PTA和ATA设置之间的距离来量化；基于可识别性理论，通过识别这些设置之间的共享潜在变量可最小化这个距离；识别和学习这些共享表示可有效减轻灾难性干扰。

### 结论

所提出的方法在理论和实践上都提供了保证，在合成和基准数据集上都取得了性能改进。

### 翻译

灾难性干扰，也称为灾难性遗忘，是机器学习中的一个基本挑战，当一个训练好的学习模型在适应新任务时，会逐渐失去在先前学习任务上的性能表现。在本文中，我们旨在从潜在表示学习的角度更好地理解和建模灾难性干扰问题，并提出一种新的理论框架，将灾难性干扰表述为一个识别问题。我们的分析表明，遗忘现象可以通过部分任务感知（PTA）和全任务感知（ATA）设置之间的距离来量化。基于可识别性理论的最新进展，我们证明可以通过识别这些设置之间的共享潜在变量来最小化这个距离。在学习过程中，我们提出了一种名为\ourmeos的方法，采用两阶段训练策略：首先，我们使用最大似然估计从PTA和ATA配置中学习潜在表示。随后，我们优化KL散度以识别和学习共享的潜在变量。通过理论保证和实证验证，我们确定识别和学习这些共享表示可以有效减轻机器学习系统中的灾难性干扰。我们的方法在合成和基准数据集上都提供了理论保证和实际的性能改进。


### 论文摘要

Catastrophic interference, also known as catastrophic forgetting, is a fundamental challenge in machine learning, where a trained learning model progressively loses performance on previously learned tasks when adapting to new ones. In this paper, we aim to better understand and model the catastrophic interference problem from a latent representation learning point of view, and propose a novel theoretical framework that formulates catastrophic interference as an identification problem. Our analysis demonstrates that the forgetting phenomenon can be quantified by the distance between partial-task aware (PTA) and all-task aware (ATA) setups. Building upon recent advances in identifiability theory, we prove that this distance can be minimized through identification of shared latent variables between these setups. When learning, we propose our method \ourmeos with two-stage training strategy: First, we employ maximum likelihood estimation to learn the latent representations from both PTA and ATA configurations. Subsequently, we optimize the KL divergence to identify and learn the shared latent variables. Through theoretical guarantee and empirical validations, we establish that identifying and learning these shared representations can effectively mitigate catastrophic interference in machine learning systems. Our approach provides both theoretical guarantees and practical performance improvements across both synthetic and benchmark datasets.

---

## 132. Perceive, Reflect and Understand Long Video: Progressive Multi-Granular Clue Exploration with Interactive Agents

**论文链接:** [http://arxiv.org/abs/2509.24943v1](http://arxiv.org/abs/2509.24943v1)

**作者:** Jiahua Li, Kun Wei, Zhe Xu, Zibo Su, Xu Yang, Cheng Deng

**发布时间:** 2025-09-29

### GPT解析

### 总结

CogniGPT是一种受人类渐进式视觉认知启发的框架，通过多粒度感知代理和验证增强反思代理之间的交互循环，实现了高效可靠的长视频理解。

### 背景

长视频具有时间复杂性和任务相关信息稀疏的特点，对AI系统构成重大推理挑战。现有的基于大型语言模型的方法在长视频理解方面难以同时实现完整性和效率。

### 目的

提出一种能够同时实现完整性和效率的长视频理解框架，有效捕捉任务关键信息。

### 方法

提出CogniGPT框架，利用多粒度感知代理(MGPA)模拟人类视觉发散和聚焦注意力捕捉相关信息，验证增强反思代理(VERA)验证感知的关键线索以减少幻觉并优化后续感知策略，通过交互循环探索最少且可靠的任务相关线索。

### 主要发现

在EgoSchema、Video-MME、NExT-QA和MovieChat数据集上的实验表明，CogniGPT在准确性和效率方面均表现出色，仅使用11.2帧就超越了现有的无需训练的方法，性能与Gemini 1.5-Pro相当。

### 结论

CogniGPT是一种高效可靠的长视频理解框架，通过模拟人类的渐进式视觉认知过程，有效解决了长视频理解中的挑战。

### 翻译

长视频以其时间复杂性和任务相关信息稀疏为特征，对AI系统构成了重大推理挑战。尽管各种基于大型语言模型的方法已经推进了长视频理解，但在捕捉任务关键信息方面仍然难以同时实现完整性和效率。受人类渐进式视觉认知的启发，我们提出了CogniGPT，这是一个利用多粒度感知代理(MGPA)和验证增强反思代理(VERA)之间交互循环的框架，用于高效可靠的长视频理解。具体来说，MGPA模拟人类的视觉发散和聚焦注意力来捕捉任务相关信息，而VERA验证感知的关键线索以减少幻觉并优化后续感知策略。通过这种交互过程，CogniGPT探索了最少的信息量且可靠的任务相关线索集合。在EgoSchema、Video-MME、NExT-QA和MovieChat数据集上的广泛实验证明了CogniGPT在准确性和效率方面的优越性。值得注意的是，在EgoSchema上，它仅使用11.2帧就超越了现有的无需训练的方法，并实现了与Gemini 1.5-Pro相当的性能。


### 论文摘要

Long videos, characterized by temporal complexity and sparse task-relevant information, pose significant reasoning challenges for AI systems. Although various Large Language Model (LLM)-based approaches have advanced long video understanding, they still struggle to achieve both completeness and efficiency in capturing task-critical information. Inspired by human progressive visual cognition, we propose CogniGPT, a framework that leverages an interactive loop between Multi-Granular Perception Agent (MGPA) and Verification-Enhanced Reflection Agent (VERA) for efficient and reliable long video understanding. Specifically, MGPA mimics human visual divergent and focused attention to capture task-related information, while VERA verifies perceived key clues to mitigate hallucination and optimize subsequent perception strategies. Through this interactive process, CogniGPT explores a minimal set of informative and reliable task-related clues. Extensive experiments on EgoSchema, Video-MME, NExT-QA, and MovieChat datasets demonstrate CogniGPT's superiority in both accuracy and efficiency. Notably, on EgoSchema, it surpasses existing training-free methods using only 11.2 frames and achieves performance comparable to Gemini 1.5-Pro.

---

## 133. StreamForest: Efficient Online Video Understanding with Persistent Event Memory

**论文链接:** [http://arxiv.org/abs/2509.24871v1](http://arxiv.org/abs/2509.24871v1)

**作者:** Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, Yi Wang, Limin Wang

**发布时间:** 2025-09-29

**备注:** Accepted as a Spotlight at NeurIPS 2025

### GPT解析

### 总结

StreamForest是一种针对实时流视频理解的新型架构，通过持久事件记忆森林和细粒度时空窗口解决了传统MLLM在流媒体场景中的局限性。

### 背景

多模态大语言模型(MLLMs)在视频理解方面取得了显著进展，但在实时流媒体场景中效果有限，主要受限于历史视觉特征的存储限制和实时空时推理不足。

### 目的

解决MLLMs在实时流媒体场景中的局限性，提出一种专门为流视频理解设计的新型架构。

### 方法

提出StreamForest架构，其核心是持久事件记忆森林，一种自适应地将视频帧组织成多个事件级别树结构的记忆机制，由基于时间距离、内容相似性和合并频率的惩罚函数指导；引入细粒度时空窗口增强实时感知；提出OnlineIT指令微调数据集；引入ODV-Bench基准测试。

### 主要发现

StreamForest在多个基准测试上取得了最先进的性能：StreamingBench上77.3%，OVBench上60.5%，OVO-Bench上55.6%；即使在极端视觉令牌压缩(限制为1024个令牌)的情况下，模型在八个基准测试中保留了96.8%的平均准确率。

### 结论

结果证明了StreamForest在流视频理解方面的鲁棒性、效率和通用性。

### 翻译

多模态大语言模型最近在视频理解方面取得了显著进展。然而，由于历史视觉特征的存储限制和实时空时推理不足，它们在实时流媒体场景中的有效性仍然有限。为了解决这些挑战，我们提出了StreamForest，一种专门为流视频理解设计的新型架构。StreamForest的核心是持久事件记忆森林，一种自适应地将视频帧组织成多个事件级别树结构的记忆机制。这一过程由基于时间距离、内容相似性和合并频率的惩罚函数指导，使模型在有限的计算资源下能够高效地保留长期记忆。为了增强实时感知，我们引入了细粒度时空窗口，它捕获详细的短期视觉线索以改善当前场景感知。此外，我们提出了OnlineIT，一个为流视频任务定制的指令微调数据集。OnlineIT显著提高了MLLM在实时感知和未来预测方面的性能。为了评估实际应用中的泛化能力，我们引入了ODV-Bench，一个新的基准测试，专注于自动驾驶场景中的实时流视频理解。实验结果表明，StreamForest取得了最先进的性能，在StreamingBench上的准确率为77.3%，在OVBench上为60.5%，在OVO-Bench上为55.6%。特别是，即使在极端视觉令牌压缩的情况下，模型在八个基准测试中相对于默认设置保留了96.8%的平均准确率。这些结果强调了StreamForest在流视频理解方面的鲁棒性、效率和通用性。


### 论文摘要

Multimodal Large Language Models (MLLMs) have recently achieved remarkable progress in video understanding. However, their effectiveness in real-time streaming scenarios remains limited due to storage constraints of historical visual features and insufficient real-time spatiotemporal reasoning. To address these challenges, we propose StreamForest, a novel architecture specifically designed for streaming video understanding. Central to StreamForest is the Persistent Event Memory Forest, a memory mechanism that adaptively organizes video frames into multiple event-level tree structures. This process is guided by penalty functions based on temporal distance, content similarity, and merge frequency, enabling efficient long-term memory retention under limited computational resources. To enhance real-time perception, we introduce a Fine-grained Spatiotemporal Window, which captures detailed short-term visual cues to improve current scene perception. Additionally, we present OnlineIT, an instruction-tuning dataset tailored for streaming video tasks. OnlineIT significantly boosts MLLM performance in both real-time perception and future prediction. To evaluate generalization in practical applications, we introduce ODV-Bench, a new benchmark focused on real-time streaming video understanding in autonomous driving scenarios. Experimental results demonstrate that StreamForest achieves the state-of-the-art performance, with accuracies of 77.3% on StreamingBench, 60.5% on OVBench, and 55.6% on OVO-Bench. In particular, even under extreme visual token compression (limited to 1024 tokens), the model retains 96.8% of its average accuracy in eight benchmarks relative to the default setting. These results underscore the robustness, efficiency, and generalizability of StreamForest for streaming video understanding.

---

## 134. LOVE-R1: Advancing Long Video Understanding with an Adaptive Zoom-in Mechanism via Multi-Step Reasoning

**论文链接:** [http://arxiv.org/abs/2509.24786v1](http://arxiv.org/abs/2509.24786v1)

**作者:** Shenghao Fu, Qize Yang, Yuan-Ming Li, Xihan Wei, Xiaohua Xie, Wei-Shi Zheng

**发布时间:** 2025-09-29

### GPT解析

### 总结

LOVE-R1模型通过自适应帧采样机制解决了长视频理解中时序理解与空间感知之间的冲突，在长视频理解任务上取得显著性能提升。

### 背景

长视频理解对大型视频语言模型(LVLMs)仍然具有挑战性，因为长时序理解与详细空间感知之间存在冲突。当前LVLMs使用均匀帧采样机制，不可避免地牺牲时间线索或空间细节，导致次优解决方案。

### 目的

缓解长视频理解中时序理解与空间感知之间的冲突，提出一种能够自适应放大视频片段的模型。

### 方法

提出LOVE-R1模型，首先提供密集采样但分辨率小的帧，根据需要基于推理放大感兴趣片段的帧分辨率，直到获得关键视觉信息。整个过程实现为多步推理过程。在38k高质量CoT数据上微调模型，并通过解耦的强化微调增强。将多步推理解耦为多个单步推理，明确优化内部放大能力。

### 主要发现

在长视频理解基准上的实验表明，LOVE-R1模型通过慢速-快速自适应帧采样机制在采样密度和帧分辨率之间取得了很好的平衡，在4个常见长视频理解基准上平均比基线Qwen2.5-VL高出3.1个百分点。

### 结论

LOVE-R1模型通过自适应帧采样机制有效解决了长视频理解中时序理解与空间感知之间的冲突，能够更好地处理长视频内容。

### 翻译

长视频理解对于最近的大型视频语言模型(LVLMs)仍然具有挑战性，这是由于长时序理解与详细空间感知之间的冲突。具有均匀帧采样机制的LVLMs（以相同帧大小和固定采样率采样帧）不可避免地牺牲时间线索或空间细节，导致次优解决方案。为了缓解这一困境，我们提出了LOVE-R1，一个能够自适应放大视频片段的模型。该模型首先被提供密集采样的帧，但分辨率较小。如果需要某些空间细节，模型可以基于其推理放大感兴趣片段的帧分辨率，直到获得关键视觉信息。整个过程被实现为多步推理过程。为了训练推理能力，我们首先在收集的38k高质量CoT数据上微调模型，并通过解耦的强化微调增强它。由于结果奖励无法提供细粒度的过程监督，我们将多步推理解耦为多个单步推理，并明确优化内部放大能力。在长视频理解基准上的实验表明，我们的模型通过慢速-快速自适应帧采样机制在采样密度和帧分辨率之间取得了很好的平衡，并且LOVE-R1在4个常见长视频理解基准上平均比我们的基线Qwen2.5-VL高出3.1个百分点。


### 论文摘要

Long video understanding is still challenging for recent Large Video-Language Models (LVLMs) due to the conflict between long-form temporal understanding and detailed spatial perception. LVLMs with a uniform frame sampling mechanism, which samples frames with an equal frame size and fixed sampling rate, inevitably sacrifice either temporal clues or spatial details, resulting in suboptimal solutions. To mitigate this dilemma, we propose LOVE-R1, a model that can adaptively zoom in on a video clip. The model is first provided with densely sampled frames but in a small resolution. If some spatial details are needed, the model can zoom in on a clip of interest with a large frame resolution based on its reasoning until key visual information is obtained. The whole process is implemented as a multi-step reasoning process. To train the reasoning ability, we first finetune the model on our collected 38k high-quality CoT data and enhance it with decoupled reinforcement finetuning. As outcome rewards can not provide fine-grained process supervision, we decouple multi-step reasoning into multiple single-step reasoning and optimize the internal zoom-in ability explicitly. Experiments on long video understanding benchmarks show that our model with the slow-fast adaptive frame sampling mechanism achieves a great trade-off between sampling density and frame resolutions, and LOVE-R1 outperforms our baseline Qwen2.5-VL by an average of 3.1% points across 4 common long video understanding benchmarks.

---

## 135. NeMo: Needle in a Montage for Video-Language Understanding

**论文链接:** [http://arxiv.org/abs/2509.24563v1](http://arxiv.org/abs/2509.24563v1)

**作者:** Zi-Yuan Hu, Shuo Liang, Duo Zheng, Yanyang Li, Yeyao Tao, Shijia Huang, Wei Feng, Jia Qin, Jianguang Yu, Jing Huang, Meng Fang, Yin Li, Liwei Wang

**发布时间:** 2025-09-29

### GPT解析

### 总结

该研究提出了一个名为Needle in a Montage (NeMo)的新任务，用于评估视频大语言模型(VideoLLMs)的时间推理能力，并构建了NeMoBench基准测试，包含31,378个自动生成的问答对，来自13,486个不同时长的视频。

### 背景

视频大语言模型(VideoLLMs)的最新进展需要新的评估协议和基准，特别是针对视频理解中复杂时间推理能力的评估。

### 目的

引入NeMo任务以评估VideoLLMs的关键推理能力，包括长上下文回忆和时间定位能力。

### 方法

开发可扩展的自动化数据生成管道生成高质量视频问答数据，基于此构建NeMoBench基准，包含31,378个问答对和13,486个视频，并评估了20个最先进模型。

### 主要发现

实验证明该管道能可靠自动生成高质量评估数据，NeMoBench可持续更新最新视频，提供了各模型能力和局限性的关键见解。

### 结论

NeMo任务和NeMoBench基准为评估VideoLLMs时间推理能力提供了有效工具，项目页面可通过提供的链接访问。

### 翻译

视频大语言模型(VideoLLMs)的最新进展需要对视频理解中复杂时间推理能力采用新的评估协议和基准。受LLMs广泛使用的'大海捞针'测试启发，我们引入了'蒙太奇中的针'(NeMo)这一新任务，旨在评估VideoLLMs的关键推理能力，包括长上下文回忆和时间定位。为生成我们任务的视频问答数据，我们开发了一个可扩展的自动化数据生成管道，促进高质量数据合成。基于提出的管道，我们展示了NeMoBench，一个以我们任务为中心的视频语言基准。具体而言，我们完整的NeMoBench集成了来自13,486个视频的31,378个自动生成的问答对，这些视频时长从几秒到几小时不等。实验证明我们的管道可以可靠且自动地生成高质量的评估数据，使NeMoBench能够持续更新最新的视频。我们在我们的基准上评估了20个最先进的模型，提供了它们能力和局限性的广泛结果和关键见解。我们的项目页面可在以下网址访问：https://lavi-lab.github.io/NeMoBench。


### 论文摘要

Recent advances in video large language models (VideoLLMs) call for new evaluation protocols and benchmarks for complex temporal reasoning in video-language understanding. Inspired by the needle in a haystack test widely used by LLMs, we introduce a novel task of Needle in a Montage (NeMo), designed to assess VideoLLMs' critical reasoning capabilities, including long-context recall and temporal grounding. To generate video question answering data for our task, we develop a scalable automated data generation pipeline that facilitates high-quality data synthesis. Built upon the proposed pipeline, we present NeMoBench, a video-language benchmark centered on our task. Specifically, our full set of NeMoBench features 31,378 automatically generated question-answer (QA) pairs from 13,486 videos with various durations ranging from seconds to hours. Experiments demonstrate that our pipeline can reliably and automatically generate high-quality evaluation data, enabling NeMoBench to be continuously updated with the latest videos. We evaluate 20 state-of-the-art models on our benchmark, providing extensive results and key insights into their capabilities and limitations. Our project page is available at: https://lavi-lab.github.io/NeMoBench.

---

## 136. Trading Carbon for Physics: On the Resource Efficiency of Machine Learning for Spatio-Temporal Forecasting

**论文链接:** [http://arxiv.org/abs/2509.24517v1](http://arxiv.org/abs/2509.24517v1)

**作者:** Sophia N. Wilson, Jens Hesselbjerg Christensen, Raghavendra Selvan

**发布时间:** 2025-09-29

**备注:** Source code available at  https://github.com/sophiawilson18/FlowMatching

### GPT解析

### 总结

这篇论文探讨了如何通过物理归纳偏置在模型效能和效率之间取得平衡，减少机器学习模型的碳足迹，同时保持或提高模型性能。

### 背景

现代深度学习方法的发展主要关注提高模型效能（准确性指标），这导致了大规模模型的发展，需要大量资源，并在模型生命周期中产生相当大的碳足迹。

### 目的

探索物理归纳偏置如何在模型效能和模型效率（计算、能源和碳）之间提供有用的权衡，提高模型的可持续性。

### 方法

研究多种用于时空预测的模型，这些任务受物理定律支配，适合探索不同程度的物理归纳偏置。结合标准的物理信息时空模型和流匹配等较新模型进行实验。

### 主要发现

将物理归纳偏置嵌入模型设计可以在保持或提高任务效能的同时获得显著的效率提升；结合物理归纳偏置提供了一种改进模型效率和减少机器学习模型碳足迹的原则性方法。

### 结论

模型效率应与模型效能一起成为推动机器学习模型开发和部署的核心考虑因素。

### 翻译

现代深度学习方法的发展主要是由提高模型效能（准确性指标）的推动所驱动的。这种对效能的单一关注引导了大规模模型的发展，这些模型需要大量资源，并在模型生命周期中产生相当大的碳足迹。在这项工作中，我们探讨了物理归纳偏置如何在模型效能和模型效率（计算、能源和碳）之间提供有用的权衡。我们研究了多种用于时空预测的模型，这些任务受物理定律支配，非常适合探索不同程度的物理归纳偏置。我们表明，将物理归纳偏置嵌入模型设计可以在保持甚至提高所考虑任务的效能的同时获得显著的效率提升。除了使用标准的物理信息时空模型外，我们还展示了流匹配等较新模型作为时空预测通用方法的有用性。我们的实验表明，结合物理归纳偏置提供了一种改进模型效率和减少机器学习模型碳足迹的原则性方法。我们认为，模型效率应与模型效能一起成为推动机器学习模型开发和部署的核心考虑因素。


### 论文摘要

Development of modern deep learning methods has been driven primarily by the push for improving model efficacy (accuracy metrics). This sole focus on efficacy has steered development of large-scale models that require massive resources, and results in considerable carbon footprint across the model life-cycle. In this work, we explore how physics inductive biases can offer useful trade-offs between model efficacy and model efficiency (compute, energy, and carbon). We study a variety of models for spatio-temporal forecasting, a task governed by physical laws and well-suited for exploring different levels of physics inductive bias. We show that embedding physics inductive biases into the model design can yield substantial efficiency gains while retaining or even improving efficacy for the tasks under consideration. In addition to using standard physics-informed spatio-temporal models, we demonstrate the usefulness of more recent models like flow matching as a general purpose method for spatio-temporal forecasting. Our experiments show that incorporating physics inductive biases offer a principled way to improve the efficiency and reduce the carbon footprint of machine learning models. We argue that model efficiency, along with model efficacy, should become a core consideration driving machine learning model development and deployment.

---

## 137. BiHDTrans: binary hyperdimensional transformer for efficient multivariate time series classification

**论文链接:** [http://arxiv.org/abs/2509.24425v1](http://arxiv.org/abs/2509.24425v1)

**作者:** Jingtao Zhang, Yi Liu, Qi Shen, Changhong Wang

**发布时间:** 2025-09-29

### GPT解析

### 总结

BiHDTrans是一种创新的神经符号二进制超维Transformer，结合了超维计算的高效性和Transformer的时序建模能力，在多元时间序列分类任务中实现了高准确率和低延迟，特别适合资源受限的边缘环境。

### 背景

物联网设备激增导致大量多元时间序列数据产生，在资源受限的边缘环境中需要高效准确的处理。超维计算在分类任务中效率高但难以捕获复杂时序模式，Transformer擅长序列建模但计算和内存开销大。

### 目的

开发一种结合超维计算效率和Transformer时序建模能力的方法，解决HD计算难以捕获复杂时序模式以及Transformer计算开销高的问题。

### 方法

提出BiHDTrans，将自注意力机制集成到超维计算范式中，统一HD计算的表示效率与Transformer的时序建模能力，并通过二量化进一步优化性能。

### 主要发现

BiHDTrans比最先进HD计算模型性能提升至少14.47%，比最先进二进制Transformer平均准确率高6.67%；在FPGA加速下，推理延迟降低39.4倍；即使维度减少64%，仍保持竞争力，模型尺寸小4.4倍，延迟进一步降低49.8%。

### 结论

BiHDTrans成功弥合了Transformer表达能力和HD计算效率之间的差距，实现了准确、可扩展和低延迟的多元时间序列分类，适用于资源受限的边缘环境。

### 翻译

物联网(IoT)设备的激增导致了前所未有的多元时间序列(MTS)数据量，需要在资源受限的边缘环境中进行高效准确的处理以实现及时决策。超维(HD)计算凭借其固有的效率和并行性，在分类任务中显示出潜力，但难以捕获复杂的时序模式，而Transformer在序列建模方面表现出色，但会带来高昂的计算和内存开销。我们引入了BiHDTrans，这是一种高效的神经符号二进制超维Transformer，将自注意力机制集成到HD计算范式中，统一了HD计算的表示效率与Transformer的时序建模能力。实验证明，BiHDTrans比最先进的HD计算模型性能提升至少14.47%，平均比最先进的二进制Transformer准确率高6.67%。在FPGA硬件加速下，我们的流水线实现利用了高维表示的独立同分布特性，比最先进的二进制Transformer推理延迟低39.4倍。理论分析表明，在全息高维空间中进行二量化比直接对神经网络进行二量化导致的信息失真显著减少，解释了BiHDTrans的优越准确率。此外，维度实验证实，即使超空间维度减少64%，BiHDTrans仍保持竞争力，以小4.4倍的模型尺寸比最先进的二进制Transformer准确率高1-2%，并且比全维基线进一步降低了49.8%的延迟。这些贡献共同弥合了Transformer表达能力和HD计算效率之间的差距，实现了准确、可扩展和低延迟的MTS分类。


### 论文摘要

The proliferation of Internet-of-Things (IoT) devices has led to an unprecedented volume of multivariate time series (MTS) data, requiring efficient and accurate processing for timely decision-making in resource-constrained edge environments. Hyperdimensional (HD) computing, with its inherent efficiency and parallelizability, has shown promise in classification tasks but struggles to capture complex temporal patterns, while Transformers excel at sequence modeling but incur high computational and memory overhead. We introduce BiHDTrans, an efficient neurosymbolic binary hyperdimensional Transformer that integrates self-attention into the HD computing paradigm, unifying the representational efficiency of HD computing with the temporal modeling power of Transformers. Empirically, BiHDTrans outperforms state-of-the-art (SOTA) HD computing models by at least 14.47% and achieves 6.67% higher accuracy on average than SOTA binary Transformers. With hardware acceleration on FPGA, our pipelined implementation leverages the independent and identically distributed properties of high-dimensional representations, delivering 39.4 times lower inference latency than SOTA binary Transformers. Theoretical analysis shows that binarizing in holographic high-dimensional space incurs significantly less information distortion than directly binarizing neural networks, explaining BiHDTrans's superior accuracy. Furthermore, dimensionality experiments confirm that BiHDTrans remains competitive even with a 64% reduction in hyperspace dimensionality, surpassing SOTA binary Transformers by 1-2% in accuracy with 4.4 times less model size, as well as further reducing the latency by 49.8% compare to the full-dimensional baseline. Together, these contributions bridge the gap between the expressiveness of Transformers and the efficiency of HD computing, enabling accurate, scalable, and low-latency MTS classification.

---

## 138. FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting

**论文链接:** [http://arxiv.org/abs/2509.24304v1](http://arxiv.org/abs/2509.24304v1)

**作者:** Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, Yu Cheng

**发布时间:** 2025-09-29

**备注:** Submitted to ICLR 2026

### GPT解析

### 总结

FrameThinker是一个新型框架，通过迭代查询视频内容和两阶段训练策略，解决了大型视觉-语言模型在长视频推理中的效率问题，实现了显著的性能提升。

### 背景

大型视觉-语言模型(LVLMs)在视频理解方面取得了显著进展，但在长视频推理应用中受到均匀帧采样和静态文本推理的限制，这些方法效率低下且难以处理视觉密集型视频任务。

### 目的

为了克服这些挑战，作者引入了'与长视频一起思考'的概念，并提出了一个名为FrameThinker的新框架，使LVLMs能够迭代地查询视频内容。

### 方法

作者提出了一种两阶段训练策略：首先使用监督微调(SFT)来培养基础行动能力，然后使用强化学习(RL)来优化战略决策政策。在RL阶段，对每个动作和格式奖励的设计进行了深入和全面的探索。

### 主要发现

在多个推理和理解基准上的实验表明，FrameThinker比基线模型实现了平均+10.4%的显著改进，同时大幅减少了处理的帧数。7B模型FrameThinker在LongVideo-Reason上实现了76.1%的准确率，仅使用平均20.6帧，比竞争模型LongVILA-R1(72.0%)性能更好，且帧数减少了20多倍。

### 结论

FrameThinker通过迭代查询视频内容和优化的训练策略，有效地解决了长视频推理中的效率问题，实现了显著的性能提升，展示了无与伦比的效率和效果。

### 翻译

虽然大型视觉-语言模型(LVLMs)在视频理解方面取得了实质性进展，但它们在长视频推理中的应用受到均匀帧采样和静态文本推理的限制，这些方法效率低下且难以处理视觉密集型视频任务。为了克服这些挑战，在本文中，我们引入了与长视频一起思考的概念，并提出了一个名为FrameThinker的新框架。在这个框架中，LVLMs能够迭代地查询视频内容。在LVLMs中开发这种视频推理能力带来了显著挑战，特别是在使模型适应新的视频动作(例如选择帧)方面，以及设计奖励函数来指导LVLMs采用新引入的动作。为了解决这些挑战，我们提出了一种两阶段训练策略，首先采用监督微调(SFT)来培养基本动作能力，然后使用强化学习(RL)来优化战略决策政策。值得注意的是，在这个RL阶段，我们对每个动作和格式奖励的设计进行了深入和全面的探索。在Video-Holmes、LongVideo-Reason等推理基准以及LongVideoBench、MLVU、VideoMME和LVBench等长视频理解基准上的大量实验表明，FrameThinker比基线模型实现了平均+10.4%的显著改进，同时大幅减少了处理的帧数。最重要的是，我们的7B模型FrameThinker在LongVideo-Reason上建立了新的最先进水平，仅使用平均20.6帧就实现了76.1%的准确率。这不仅超过了竞争性的LongVILA-R1(72.0%)，而且使用的帧数减少了20多倍(对比512帧)，展示了无与伦比的效率和效果。


### 论文摘要

While Large Vision-Language Models (LVLMs) have achieved substantial progress in video understanding, their application to long video reasoning is hindered by uniform frame sampling and static textual reasoning, which are inefficient and struggle to handle visually intensive video tasks. To overcome these challenges, in this paper, we introduce the concept of thinking with long videos and propose a novel framework FrameThinker. Within this framework, LVLMs are able to iteratively interrogate video content. Developing such video reasoning capabilities in LVLMs presents notable challenges, particularly in adapting the model to new video actions (e.g. select frame), and designing reward functions to guide LVLMs to adopt the newly introduced action. To solve these challenges, we propose a two-phase training strategy, first employing Supervised Fine-Tuning (SFT) to instill fundamental action capabilities, followed by Reinforcement Learning (RL) to optimize a strategic decision-making policy.Notably, in this RL phase, we conduct an in-depth and comprehensive exploration of the reward design for each action and format reward. Extensive experiments on reasoning benchmarks like Video-Holmes, LongVideo-Reason, and long-video understanding benchmarks such as LongVideoBench, MLVU, VideoMME, and LVBench, demonstrate that FrameThinker achieves a significant average improvement of +10.4% over baselines while drastically reducing the number of processed frames. Most notably, our 7B model, FrameThinker establishes a new state-of-the-art on LongVideo-Reason, achieving 76.1% accuracy using an average of only 20.6 frames. This not only outperforms the competitive LongVILA-R1 (72.0%) but does so with over 20x fewer frames (vs. 512), demonstrating unparalleled efficiency and effectiveness.

---

## 139. Understanding Cognitive States from Head & Hand Motion Data

**论文链接:** [http://arxiv.org/abs/2509.24255v1](http://arxiv.org/abs/2509.24255v1)

**作者:** Kaiang Wen, Mark Roman Miller

**发布时间:** 2025-09-29

### GPT解析

### 总结

这项研究探讨了消费级VR系统捕获的头部和手部运动数据如何编码用户的认知状态。研究者创建了一个新的数据集，包含在结构化决策任务中收集的头部和手部运动数据，并对这些运动数据进行了认知状态的帧级标注。研究发现，深度时间模型仅从运动数据就能推断出这些微妙的认知状态，性能与人类观察者相当。

### 背景

随着虚拟现实(VR)和增强现实(AR)的普及，消费级VR系统捕获的头部和手部运动数据变得越来越普遍。先前的工作表明，这类遥测数据具有很高的识别性，并能反映广泛用户特征，通常与身体语言的直观'民间理论'一致。然而，目前尚不清楚运动运动学能在多大程度上编码更微妙的认知状态，如困惑、犹豫和准备状态，这些状态与运动缺乏明确的关联。

### 目的

研究的主要目的是调查运动数据是否能够编码更微妙的认知状态（如困惑、犹豫和准备状态），以及这些状态如何从VR系统捕获的头部和手部运动数据中推断出来。

### 方法

研究者引入了一个新的数据集，包含在结构化决策任务中收集的头部和手部运动数据，并对这些数据进行了认知状态的帧级标注。他们使用深度时间模型来仅从运动数据中推断这些认知状态，并将模型性能与人类观察者进行比较。

### 主要发现

研究发现，深度时间模型仅从运动数据就能推断出微妙的认知状态，性能与人类观察者相当。这表明标准VR遥测数据包含与用户内部认知过程相关的强模式。

### 结论

这项研究证明了标准VR遥测数据包含与用户内部认知过程相关的强模式，为新一代自适应虚拟环境开辟了道路。为了增强可重复性并支持未来的工作，研究者将公开提供他们的数据集和建模框架。

### 翻译

随着虚拟现实(VR)和增强现实(AR)的持续普及，消费级VR系统捕获的头部和手部运动数据已变得无处不在。先前的工作表明，这类遥测数据具有很高的识别性，并能反映广泛的用户特征，通常与身体语言的直观'民间理论'一致。然而，目前尚不清楚运动运动学能在多大程度上编码更微妙的认知状态，如困惑、犹豫和准备状态，这些状态与运动缺乏明确的关联。为了调查这一点，我们引入了一个新的数据集，包含在结构化决策任务中收集的头部和手部运动数据，并对这些状态进行了帧级标注。我们的研究结果表明，深度时间模型仅从运动数据就能推断出微妙的认知状态，性能与人类观察者相当。这项工作证明了标准VR遥测数据包含与用户内部认知过程相关的强模式，为新一代自适应虚拟环境开辟了道路。为了增强可重复性并支持未来的工作，我们将公开提供我们的数据集和建模框架。


### 论文摘要

As virtual reality (VR) and augmented reality (AR) continue to gain popularity, head and hand motion data captured by consumer VR systems have become ubiquitous. Prior work shows that such telemetry can be highly identifying and reflect broad user traits, often aligning with intuitive "folk theories" of body language. However, it remains unclear to what extent motion kinematics encode more nuanced cognitive states, such as confusion, hesitation, and readiness, which lack clear correlates with motion. To investigate this, we introduce a novel dataset of head and hand motion with frame-level annotations of these states collected during structured decision-making tasks. Our findings suggest that deep temporal models can infer subtle cognitive states from motion alone, achieving comparable performance with human observers. This work demonstrates that standard VR telemetry contains strong patterns related to users' internal cognitive processes, which opens the door for a new generation of adaptive virtual environments. To enhance reproducibility and support future work, we will make our dataset and modeling framework publicly available.

---

## 140. UniVid: The Open-Source Unified Video Model

**论文链接:** [http://arxiv.org/abs/2509.24200v1](http://arxiv.org/abs/2509.24200v1)

**作者:** Jiabin Luo, Junhui Lin, Zeyu Zhang, Biao Wu, Meng Fang, Ling Chen, Hao Tang

**发布时间:** 2025-09-29

### GPT解析

### 总结

本文提出了一种名为UniVid的统一视频建模架构，通过轻量级适配器将多模态大语言模型(MLLM)与扩散解码器耦合，实现了视频理解和生成能力的统一。

### 背景

统一视频建模（结合生成和理解能力）日益重要，但面临两个关键挑战：在基于流的生成过程中保持语义保真度时存在文本-视觉标记不平衡问题，以及统一跨模态注意力在流轨迹中的局限性，此外还需高效地将以图像为中心的MLLM扩展到视频而无需昂贵的重新训练。

### 目的

开发一个统一架构，能够同时处理视频理解和视频生成任务，解决现有方法中的标记不平衡和注意力限制问题，并提供一种高效的模型扩展方案。

### 方法

提出UniVid架构，通过轻量级适配器连接MLLM和扩散解码器；引入温度模态对齐(Temperature Modality Alignment)提高提示遵循度；实现金字塔反射(Pyramid Reflection)通过动态关键帧选择进行高效时间推理。

### 主要发现

在标准基准测试中，UniVid实现了最先进的性能，与EasyAnimateV5.1相比在VBench-Long总分上提升2.2%，与最佳之前的7B基线相比在MSVD-QA和ActivityNet-QA上分别提升1.0%和3.3%的准确率。

### 结论

UniVid成功统一了视频理解和生成能力，解决了文本-视觉标记不平衡和跨模态注意力限制问题，为高效扩展以图像为中心的MLLM到视频领域提供了有效方案。

### 翻译

统一视频建模结合生成和理解能力变得越来越重要，但面临两个关键挑战：由于文本-视觉标记不平衡，在基于流的生成过程中保持语义保真度，以及在流轨迹中统一跨模态注意力的局限性，以及高效地将以图像为中心的MLLM扩展到视频而无需昂贵的重新训练。我们提出了UniVid，一种统一架构，通过轻量级适配器将MLLM与扩散解码器耦合，使视频理解和生成成为可能。我们引入温度模态对齐来提高提示遵循度，并通过动态关键帧选择实现高效时间推理的金字塔反射。在标准基准上的广泛实验展示了最先进的性能，与EasyAnimateV5.1相比在VBench-Long总分上实现了2.2%的改进，与最佳之前的7B基线相比在MSVD-QA和ActivityNet-QA上分别实现了1.0%和3.3%的准确率提升。


### 论文摘要

Unified video modeling that combines generation and understanding capabilities is increasingly important but faces two key challenges: maintaining semantic faithfulness during flow-based generation due to text-visual token imbalance and the limitations of uniform cross-modal attention across the flow trajectory, and efficiently extending image-centric MLLMs to video without costly retraining. We present UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. We introduce Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection. Extensive experiments on standard benchmarks demonstrate state-of-the-art performance, achieving a 2.2% improvement on VBench-Long total score compared to EasyAnimateV5.1, and 1.0% and 3.3% accuracy gains on MSVD-QA and ActivityNet-QA, respectively, compared with the best prior 7B baselines.

---

## 141. SVAC: Scaling Is All You Need For Referring Video Object Segmentation

**论文链接:** [http://arxiv.org/abs/2509.24109v1](http://arxiv.org/abs/2509.24109v1)

**作者:** Li Zhang, Haoxiang Gao, Zhihao Zhang, Luoxiao Huang, Tao Zhang

**发布时间:** 2025-09-28

**备注:** This paper is accepted to BMVC 2025

### GPT解析

### 总结

这篇论文提出了一种名为SVAC的统一模型，用于改进基于自然语言描述的视频目标分割(RVOS)。该模型通过扩大输入帧和分割标记来增强视频-语言交互和分割精度，并引入了专门的压缩和分配策略来解决计算挑战和动态对象行为处理问题。

### 背景

Referring Video Object Segmentation (RVOS)旨在根据自然语言描述对视频序列中的目标对象进行分割。虽然最近多模态大语言模型(MLLMs)的进展通过增强文本-视频理解提高了RVOS性能，但仍面临几个挑战。

### 目的

解决当前RVOS方法中的三个主要挑战：(1)MLLMs先验知识利用不足；(2)长视频计算和内存成本过高；(3)复杂动态时序处理不足。

### 方法

作者提出了SVAC模型，包括两个关键组件：(1)基于锚点的时空压缩(ASTC)模块：压缩视觉标记同时保留基本时空结构；(2)特定剪辑分配(CSA)策略：更好地处理视频剪辑间的动态对象行为。

### 主要发现

实验结果表明，SVAC在多个RVOS基准测试上实现了最先进的性能，同时保持了有竞争力的效率。

### 结论

SVAC通过扩大输入规模和引入专门的压缩和分配策略，有效解决了RVOS中的关键挑战，实现了性能和效率的平衡。

### 翻译

基于引用的视频目标分割(RVOS)旨在根据自然语言描述对视频序列中的目标对象进行分割。虽然最近多模态大语言模型(MLLMs)的进展通过增强文本-视频理解提高了RVOS性能，但仍存在几个挑战，包括对MLLMs先验知识利用不足、长视频计算和内存成本过高以及对复杂动态时序处理不足。在这项工作中，我们提出了SVAC，一个统一模型，通过扩大输入帧和分割标记来增强视频-语言交互和分割精度，从而改进RVOS。为解决由此产生的计算挑战，SVAC集成了基于锚点的时空压缩(ASTC)模块，在保留基本时空结构的同时压缩视觉标记。此外，还引入了特定剪辑分配(CSA)策略，以更好地处理视频剪辑间的动态对象行为。实验结果表明，SVAC在多个RVOS基准测试上实现了最先进的性能，同时保持了有竞争力的效率。我们的代码可在https://github.com/lizhang1998/SVAC获取。


### 论文摘要

Referring Video Object Segmentation (RVOS) aims to segment target objects in video sequences based on natural language descriptions. While recent advances in Multi-modal Large Language Models (MLLMs) have improved RVOS performance through enhanced text-video understanding, several challenges remain, including insufficient exploitation of MLLMs' prior knowledge, prohibitive computational and memory costs for long-duration videos, and inadequate handling of complex temporal dynamics. In this work, we propose SVAC, a unified model that improves RVOS by scaling up input frames and segmentation tokens to enhance video-language interaction and segmentation precision. To address the resulting computational challenges, SVAC incorporates the Anchor-Based Spatio-Temporal Compression (ASTC) module to compress visual tokens while preserving essential spatio-temporal structure. Moreover, the Clip-Specific Allocation (CSA) strategy is introduced to better handle dynamic object behaviors across video clips. Experimental results demonstrate that SVAC achieves state-of-the-art performance on multiple RVOS benchmarks with competitive efficiency. Our code is available at https://github.com/lizhang1998/SVAC.

---

## 142. FrameMind: Frame-Interleaved Chain-of-Thought for Video Reasoning via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2509.24008v1](http://arxiv.org/abs/2509.24008v1)

**作者:** Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, Yujun Cai

**发布时间:** 2025-09-28

**备注:** Underreview

### GPT解析

### 总结

本文提出了一种名为FrameMind的新型视频理解框架，通过强化学习训练模型在推理过程中动态请求视觉信息，显著提升了视频理解任务的性能。

### 背景

当前视频理解模型依赖固定帧采样策略，处理预定的视觉输入而忽略不同问题的特定推理需求，这种静态方法限制了模型自适应收集视觉证据的能力。

### 目的

开发一个能够动态获取视觉信息的视频理解框架，解决需要广泛时间覆盖或精细空间细节的任务中的性能瓶颈问题。

### 方法

提出FrameMind端到端框架，结合帧交错思维链(FiCOT)使模型在文本推理和主动视觉感知间交替；设计动态分辨率帧采样(DRFS)策略和DRFS-GRPO组相对策略优化算法进行训练。

### 主要发现

在MLVU和VideoMME等挑战性基准测试上，FrameMind显著优于现有模型，证明了动态视觉信息收集策略的有效性。

### 结论

FrameMind框架代表了视频理解领域的重要进展，实现了更灵活和高效的视觉信息处理能力，为未来视频理解研究提供了新方向。

### 翻译

当前的视频理解模型依赖于固定的帧采样策略，处理预定的视觉输入，而不管每个问题的特定推理需求。这种静态方法限制了它们自适应收集视觉证据的能力，导致在需要广泛时间覆盖或精细空间细节的任务上表现不佳。在本文中，我们介绍了FrameMind，这是一个通过强化训练的端到端框架，通过帧交错思维链(FiCOT)使模型能够在推理过程中动态请求视觉信息。与传统方法不同，FrameMind在多个轮次中运行，模型在文本推理和主动视觉感知之间交替，使用工具提取基于识别出的知识差距的目标帧或视频片段。为了训练有效的动态采样策略，我们提出了动态分辨率帧采样(DRFS)，在学习过程中使模型接触多样化的时空权衡，以及DRFS-GRPO，一种基于结果奖励学习的组相对策略优化算法，无需帧级注释。在MLVU和VideoMME等具有挑战性的基准测试上进行的广泛实验表明，我们的方法显著优于现有模型，推进了灵活高效视频理解的最先进水平。


### 论文摘要

Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding.

---

## 143. Video Panels for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2509.23724v1](http://arxiv.org/abs/2509.23724v1)

**作者:** Lars Doorenbos, Federico Spurio, Juergen Gall

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种无需训练、参数和模型特定的视觉提示策略，通过将多帧组合为图像面板来提高视频语言模型在长视频理解任务上的性能，在多个基准测试上取得了显著提升，特别是在长视频问答任务上准确率提高了高达19.4%。

### 背景

现有的视频语言模型(VLMs)在长视频理解任务上表现有限，相比图像或短视频任务仍有差距，导致研究人员通过引入新模块和增加复杂性来改进VLMs的长上下文建模能力。

### 目的

采用不同于传统微调的方法，最大化现有VLMs模型的性能，而非使用有限数据进行微调。

### 方法

提出一种专门为长视频理解设计的视觉提示策略，通过将多帧组合为一个图像面板，在空间细节和时间分辨率之间进行权衡，实现无需训练、无需参数、与模型无关的解决方案，可无缝集成到现有VLMs中。

### 主要发现

在五个涵盖不同模型架构、大小和上下文窗口的基准测试上验证了方法的一致性，在TimeScope (Long)数据集上，视频问答准确率提高了高达19.4%。

### 结论

该方法为长视频理解模型设立了新的性能标准，作者将在论文接受后公开代码。

### 翻译

近期的视频语言模型(VLMs)在长视频理解方面取得了有希望的结果，但它们在涉及图像或短视频的任务上的表现仍然优于长视频任务。这导致人们通过引入新模块和增加额外复杂性来提高VLMs的长上下文建模能力。在本文中，我们采取了不同的方法：不是用有限的数据微调VLMs，而是尝试最大化现有模型的性能。为此，我们提出了一种专门为长视频理解设计的新型视觉提示策略。通过将多帧组合为一个图像的面板，我们有效地在空间细节和时间分辨率之间进行权衡。我们的方法无需训练、无需参数、与模型无关，可以无缝集成到现有的VLMs中。在五个涵盖广泛模型架构、大小和上下文窗口的既定基准测试上进行的广泛实验证实了我们方法的一致性。对于包含最长视频的TimeScope (Long)数据集，视频问答的准确率提高了高达19.4%。总体而言，我们的方法为长视频理解模型设立了新的标准。我们将在论文接受后公开代码。


### 论文摘要

Recent Video-Language Models (VLMs) achieve promising results on long-video understanding, but their performance still lags behind that achieved on tasks involving images or short videos. This has led to great interest in improving the long context modeling of VLMs by introducing novel modules and additional complexity. % additional training time. In this paper, we take a different approach: rather than fine-tuning VLMs with the limited data available, we attempt to maximize the performance of existing models. To this end, we propose a novel visual prompting strategy specifically designed for long-video understanding. By combining multiple frames as panels into one image, we effectively trade off spatial details for temporal resolution. Our approach is training-free, parameter-free, and model-agnostic, and can be seamlessly integrated into existing VLMs. Extensive experiments on five established benchmarks across a wide range of model architectures, sizes, and context windows confirm the consistency of our approach. For the TimeScope (Long) dataset, which has the longest videos, the accuracy for video question answering is improved by up to 19.4\%. Overall, our method raises the bar for long video understanding models. We will make our code available upon acceptance.

---

## 144. Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding

**论文链接:** [http://arxiv.org/abs/2509.23672v1](http://arxiv.org/abs/2509.23672v1)

**作者:** Xixi Jiang, Chen Yang, Dong Zhang, Pingcheng Dong, Xin Yang, Kwang-Ting Cheng

**发布时间:** 2025-09-28

### GPT解析

### 总结

本文提出了一种名为STIM-TM的时空信息挖掘标记合并方法，专门针对手术视频理解任务，通过解耦策略独立处理时空维度，显著降低了计算成本同时保持准确性。

### 背景

Vision Transformer模型在手术视频理解中表现出色，但现有方法因处理大量时空标记导致计算成本过高。先前的标记合并方法未充分考虑视频数据的固有时空结构和信息分布异质性，导致性能不佳。

### 目的

开发一种专门针对手术视频理解的标记合并方法，解决现有方法的计算效率问题。

### 方法

提出STIM-TM方法，采用解耦策略：时间组件使用显著性权重合并连续帧中的空间对应标记；空间组件通过时间稳定性分析优先合并静态标记，保护动态区域。该方法以无训练方式操作。

### 主要发现

STIM-TM实现了超过65%的GFLOPs减少，同时保持竞争性的准确性，支持长序列手术视频的高效训练，解决了手术应用中的计算瓶颈。

### 结论

STIM-TM是首个专门针对手术视频理解的标记合并方法，通过解耦策略有效解决了时空标记处理的高计算成本问题。

### 翻译

Vision Transformer模型通过长程依赖建模在手术视频理解任务中表现出色。然而，当前方法因处理跨视频帧的大量时空标记而导致计算成本过高。虽然先前的标记合并工作提高了模型效率，但它们未能充分考虑视频数据的固有时空结构和信息分布的异质性，导致性能不佳。本文提出了一种时空信息挖掘标记合并(STIM-TM)方法，这是首个专门针对手术视频理解的方法。STIM-TM引入了一种解耦策略，独立减少时间和空间维度上的标记冗余。具体而言，时间组件使用显著性权重合并连续帧中的空间对应标记，保留关键顺序信息和连续性。同时，空间组件通过时间稳定性分析优先合并静态标记，保护包含重要手术信息的动态区域。以无训练方式操作，STIM-TM实现了显著的效率提升，减少超过65%的GFLOPs，同时在各种手术视频任务中保持竞争性的准确性。我们的方法还支持长序列手术视频的高效训练，解决了手术应用中的计算瓶颈。


### 论文摘要

Vision Transformer models have shown impressive effectiveness in the surgical video understanding tasks through long-range dependency modeling. However, current methods suffer from prohibitive computational costs due to processing massive spatiotemporal tokens across video frames. While prior work on token merging has advanced model efficiency, they fail to adequately consider the inherent spatiotemporal structure of video data and overlook the heterogeneous nature of information distribution, leading to suboptimal performance. In this paper, we propose a spatiotemporal information mining token merging (STIM-TM) method, representing the first dedicated approach for surgical video understanding. STIM-TM introduces a decoupled strategy that reduces token redundancy along temporal and spatial dimensions independently. Specifically, the temporal component merges spatially corresponding tokens from consecutive frames using saliency weighting, preserving critical sequential information and maintaining continuity. Meanwhile, the spatial component prioritizes merging static tokens through temporal stability analysis, protecting dynamic regions containing essential surgical information. Operating in a training-free manner, STIM-TM achieves significant efficiency gains with over $65\%$ GFLOPs reduction while preserving competitive accuracy across comprehensive surgical video tasks. Our method also supports efficient training of long-sequence surgical videos, addressing computational bottlenecks in surgical applications.

---

## 145. EditGRPO: Reinforcement Learning with Post -Rollout Edits for Clinically Accurate Chest X-Ray Report Generation

**论文链接:** [http://arxiv.org/abs/2509.22812v1](http://arxiv.org/abs/2509.22812v1)

**作者:** Kai Zhang, Christopher Malon, Lichao Sun, Martin Renqiang Min

**发布时间:** 2025-09-26

### GPT解析

### 总结

EditGRPO是一种混合策略强化学习算法，通过临床动机的奖励优化放射科报告生成，在多个指标上表现优于基线模型，并具有更好的泛化能力。

### 背景

放射科报告生成需要先进的医学图像分析、有效的时间推理和准确的文本生成。虽然多模态大语言模型(MLLMs)已显示出改进性能，但其监督微调目标与临床效果没有明确对齐。

### 目的

引入一种专门设计用来通过临床动机的奖励优化生成的混合策略强化学习算法。

### 方法

EditGRPO算法整合了在线策略探索与离线策略指导，通过在训练过程中注入句子级别的详细修正来解决RL中的探索困境和采样效率问题。该算法应用于使用监督微调初始化的Qwen2.5-VL-3B MLLM模型。

### 主要发现

EditGRPO优于SFT和普通GRPO基线，在四个胸部X光报告生成数据集上的CheXbert、GREEN、Radgraph和RATEScore指标上平均提高了3.4%。此外，EditGRPO在未见过的数据集上显示出平均5.9%的性能增益，表明其优越的域外泛化能力。

### 结论

EditGRPO是一种有效的混合策略强化学习算法，能够优化放射科报告生成，在临床相关指标上表现出色，并且具有更好的泛化能力。

### 翻译

放射科报告生成需要先进的医学图像分析、有效的时间推理和准确的文本生成。尽管最近的创新，特别是多模态大语言模型(MLLMs)，已经显示出改进的性能，但它们的监督微调(SFT)目标与临床效果没有明确对齐。在这项工作中，我们引入了EditGRPO，这是一种专门设计用来通过临床动机的奖励优化生成的混合策略强化学习(RL)算法。EditGRPO通过在训练过程中注入句子级别的详细修正，将在线策略探索与离线策略指导相结合。这种混合策略方法解决了RL中通常遇到的探索困境和采样效率问题。应用于一个使用监督微调(SFT)初始化的Qwen2.5-VL-3B MLLM，EditGRPO优于SFT和普通GRPO基线，在四个主要胸部X光报告生成数据集上的CheXbert、GREEN、Radgraph和RATEScore指标上平均提高了3.4%。值得注意的是，EditGRPO还展示了优越的域外泛化能力，在未见过的数据集上平均性能提高了5.9%。


### 论文摘要

Radiology report generation requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. Although recent innovations, particularly multimodal large language models (MLLMs), have shown improved performance, their supervised fine-tuning (SFT) objective is not explicitly aligned with clinical efficacy. In this work, we introduce EditGRPO, a mixed-policy reinforcement learning (RL) algorithm designed specifically to optimize the generation through clinically motivated rewards. EditGRPO integrates on-policy exploration with off-policy guidance by injecting sentence-level detailed corrections during training rollouts. This mixed-policy approach addresses the exploration dilemma and sampling efficiency issues typically encountered in RL. Applied to a Qwen2.5-VL-3B MLLM initialized with supervised fine-tuning (SFT), EditGRPO outperforms both SFT and vanilla GRPO baselines, achieving an average improvement of 3.4% in CheXbert, GREEN, Radgraph, and RATEScore metrics across four major chest X-ray report generation datasets. Notably, EditGRPO also demonstrates superior out-of-domain generalization, with an average performance gain of 5.9% on unseen datasets.

---

