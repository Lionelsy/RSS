# 今日论文推荐 - 2025-11-18

共 236 篇论文

---

## 1. CloseUpShot: Close-up Novel View Synthesis from Sparse-views via Point-conditioned Diffusion Model

**论文链接:** [http://arxiv.org/abs/2511.13121v1](http://arxiv.org/abs/2511.13121v1)

**作者:** Yuqi Zhang, Guanying Chen, Jiaxing Chen, Chuanyu Fu, Chuan Huang, Shuguang Cui

**发布时间:** 2025-11-17

**备注:** Project Link: https://zyqz97.github.io/CloseUpShot/

### GPT解析

### 总结

本研究提出了一种名为CloseUpShot的扩散框架，用于从稀疏输入视图合成近距离新视角，解决了现有方法在近距离场景中难以捕捉精细细节的问题。

### 背景

从稀疏输入视图重建3D场景和合成新视角是一项极具挑战性的任务。尽管最近的视频扩散模型在时间推理方面表现出色，但现有方法主要针对适中的视角变化设计，在近距离场景中因输入信息严重受限而难以捕捉精细细节。

### 目的

开发一种能够在近距离场景下从稀疏输入视图高质量合成新视角的方法，解决现有方法在近距离设置中的局限性。

### 方法

提出CloseUpShot框架，通过点条件视频扩散进行近距离新视角合成。针对像素扭曲条件在近距离设置中存在的严重稀疏性和背景泄漏问题，设计了分层扭曲和遮挡感知噪声抑制技术，并引入全局结构指导，利用密集融合点云为扩散过程提供一致的几何上下文。

### 主要发现

在多个数据集上的广泛实验表明，所提出的方法优于现有方法，特别是在近距离新视角合成任务上表现出色，验证了设计的有效性。

### 结论

CloseUpShot框架通过创新的点条件视频扩散方法，有效解决了近距离场景下从稀疏输入视图合成新视角的挑战，为3D场景重建提供了新的解决方案。

### 翻译

从稀疏输入视图重建3D场景和合成新视角是一项极具挑战性的任务。最近视频扩散模型的进展展示了强大的时间推理能力，使其成为在稀疏视图设置下提高重建质量的有力工具。然而，现有方法主要针对适中的视角变化设计，在近距离场景中难以捕捉精细细节，因为输入信息严重受限。本文提出了一种基于扩散的框架，名为CloseUpShot，通过点条件视频扩散从稀疏输入进行近距离新视角合成。具体来说，我们观察到在近距离设置中，像素扭曲条件存在严重的稀疏性和背景泄漏问题。为解决这一问题，我们提出了分层扭曲和遮挡感知噪声抑制，增强了视频扩散模型的图像质量和完整性。此外，我们引入了全局结构指导，利用密集融合点云为扩散过程提供一致的几何上下文，以补偿稀疏条件输入中缺乏的全局一致3D约束。在多个数据集上的广泛实验证明，我们的方法优于现有方法，特别是在近距离新视角合成方面，明显验证了我们设计的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏输入视图进行特写新视图合成（close-up novel view synthesis）和3D重建的问题。这个问题在现实中很重要，因为许多应用如文化遗产数字化、产品展示和远程检查需要细粒度的3D理解，而获取大量视图在时间、硬件或可及性方面往往不切实际。现有方法在常规视角表现良好，但在需要放大查看细节的特写场景下效果不佳，限制了这些应用的实际价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点条件扩散模型在特写视图下的局限性，包括点云稀疏、背景泄漏和深度不一致等问题。他们认识到条件图像质量对视频扩散模型性能至关重要，因此设计了三个关键模块：分层扭曲生成更密集条件图像、遮挡感知噪声抑制减少背景泄漏、全局结构引导提供一致几何上下文。作者借鉴了ViewCrafter的点条件扩散模型框架，使用VGGT或DUSt3R预测深度和相机参数，采用3D高斯溅射进行最终重建，并参考了DepthFusion和GEN3C的相关技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过改进条件图像的质量和一致性，增强点条件扩散模型在特写视图下的表现。整体流程包括：1)接收两个稀疏输入视图；2)使用预训练估计器预测深度图和相机参数；3)应用分层扭曲生成更密集条件图像；4)使用遮挡感知噪声抑制减少背景泄漏；5)通过全局结构引导提供一致几何上下文；6)训练和推理扩散模型生成多视图图像；7)使用置信感知优化进行3D高斯溅射重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分层扭曲策略在多分辨率生成密集条件图像；2)遮挡感知噪声抑制动态处理背景泄漏；3)全局结构引导提供一致几何上下文；4)置信感知3D高斯溅射优化。相比之前工作，CloseUpShot专门针对特写视图优化，显著改善了条件图像质量，引入全局几何一致性，实现了从稀疏输入到高质量3D重建的端到端流程，仅需2个输入视图就能生成高质量特写视图。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CloseUpShot通过改进点条件扩散模型的条件图像质量，引入分层扭曲、遮挡感知噪声抑制和全局结构引导等创新技术，显著提升了从稀疏输入视图进行特写新视图合成和3D重建的性能和质量。'}


### 论文摘要

Reconstructing 3D scenes and synthesizing novel views from sparse input views is a highly challenging task. Recent advances in video diffusion models have demonstrated strong temporal reasoning capabilities, making them a promising tool for enhancing reconstruction quality under sparse-view settings. However, existing approaches are primarily designed for modest viewpoint variations, which struggle in capturing fine-grained details in close-up scenarios since input information is severely limited. In this paper, we present a diffusion-based framework, called CloseUpShot, for close-up novel view synthesis from sparse inputs via point-conditioned video diffusion. Specifically, we observe that pixel-warping conditioning suffers from severe sparsity and background leakage in close-up settings. To address this, we propose hierarchical warping and occlusion-aware noise suppression, enhancing the quality and completeness of the conditioning images for the video diffusion model. Furthermore, we introduce global structure guidance, which leverages a dense fused point cloud to provide consistent geometric context to the diffusion process, to compensate for the lack of globally consistent 3D constraints in sparse conditioning inputs. Extensive experiments on multiple datasets demonstrate that our method outperforms existing approaches, especially in close-up novel view synthesis, clearly validating the effectiveness of our design.

---

## 2. Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective

**论文链接:** [http://arxiv.org/abs/2511.12170v1](http://arxiv.org/abs/2511.12170v1)

**作者:** Wang Luo, Di Wu, Hengyuan Na, Yinlin Zhu, Miao Hu, Guocong Quan

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种新的点云补全范式'Completion-by-Correction'，以及基于该范式的PGNet框架，通过从预训练的图像到3D模型生成的拓扑完整形状先验开始，并在特征空间中进行校正，实现了结构一致且与观测对齐的3D形状重建。

### 背景

点云补全旨在从部分观测重建完整的3D形状，但由于严重的遮挡和缺失几何结构，这是一个具有挑战性的问题。尽管最近的多模态技术利用互补的RGB图像来补偿缺失几何结构，但大多数方法仍遵循'Completion-by-Inpainting'范式，从融合的潜在特征中合成缺失结构。

### 目的

解决传统'Completion-by-Inpainting'范式导致的结构不一致和拓扑伪影问题，提出一种更鲁棒的点云补全方法。

### 方法

提出'Completion-by-Correction'范式，从预训练的图像到3D模型生成的拓扑完整形状先验开始，在特征空间中进行校正以与部分观测对齐。基于此范式，引入PGNet多阶段框架，进行双特征编码，合成粗略但结构对齐的支架，并通过分层校正逐步细化几何细节。

### 主要发现

传统'Completion-by-Inpainting'范式由于有限的几何和语义约束，常常导致结构不一致和拓扑伪影。'Completion-by-Correction'范式将补全从未约束的合成转变为引导式细化，能够实现结构一致且与观测对齐的重建。

### 结论

PGNet在ShapeNetViPC数据集上的实验表明，与最先进的基线方法相比，PGNet在平均Chamfer距离上降低了23.5%，F分数提高了7.1%，证明了所提范式和方法的有效性。

### 翻译

点云补全旨在从部分观测重建完整的3D形状，由于严重的遮挡和缺失几何结构，这是一个具有挑战性的问题。尽管最近的多模态技术利用互补的RGB图像来补偿缺失几何结构，但大多数方法仍遵循'Completion-by-Inpainting'范式，从融合的潜在特征中合成缺失结构。我们经验性地表明，由于有限的几何和语义约束，这种范式常常导致结构不一致和拓扑伪影。为了解决这个问题，我们重新思考了任务，并提出了一个更强大的范式，称为'Completion-by-Correction'，它从预训练的图像到3D模型生成的拓扑完整形状先验开始，并在特征空间中进行校正，使其与部分观测一致。这种范式将补全从未约束的合成转变为引导式细化，实现了结构一致且与观测对齐的重建。基于这一范式，我们引入了PGNet，这是一个多阶段框架，进行双特征编码以将生成先验接地，合成一个粗略但结构对齐的支架，并通过分层校正逐步细化几何细节。在ShapeNetViPC数据集上的实验表明，PGNet在平均Chamfer距离(-23.5%)和F分数(+7.1%)方面优于最先进的基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全中的结构一致性和拓扑缺陷问题。在现实中，LiDAR和RGB-D相机捕获的点云常因遮挡、光线反射和有限分辨率而不完整，这些不完整点云阻碍了自动驾驶、增强现实和机器人等下游任务的性能。现有多模态方法利用RGB图像补充缺失几何信息，但大多从融合的不完整特征中合成缺失结构，导致结构不一致和拓扑缺陷。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有'Completion-by-Inpainting'范式的局限性，即在不完整表示中合成缺失结构会引入伪影。他们重新思考任务，提出'Completion-by-Correction'范式，从预训练图像到3D模型生成的完整形状先验开始，然后在特征空间进行校正。作者借鉴了图像到3D模型生成完整形状先验的思想，多模态学习利用RGB图像，以及点云处理中的特征编码、Transformer架构和分层细化策略。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'Completion-by-Correction'范式：不是从不完整的融合表示中合成缺失几何，而是从拓扑完整和语义上有意义的形状先验开始，然后在特征空间进行校正，使其与部分观测对齐。整体流程包括三个阶段：1) 校正双特征编码：并行编码部分点云和生成点云，在特征空间将先验锚定到观测；2) 锚定种子生成：合成结构完整但几何锚定的支架；3) 分层锚定细化：使用堆细化块逐步改进几何保真度，通过双源特征关联和结构感知上采样细化空间布局。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 提出'Completion-by-Correction'范式；2) 设计PGNet框架实现这一范式；3) 在ShapeNetViPC数据集上实现最先进性能。相比之前工作，不同之处在于：不是从不完整融合特征中合成缺失几何，而是校正完整形状先验；不是直接几何融合，而是在特征空间进行校正；将任务从无约束合成转变为引导的结构精细调整；使用双源特征关联解决坐标不对齐问题；使用结构感知上采样预测局部位移。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文提出了'Completion-by-Correction'范式和PGNet框架，通过校正预训练图像到3D模型生成的完整形状先验，显著提高了多模态点云补全的结构一致性和几何保真度，实现了比现有方法更优的性能。"}


### 论文摘要

Point cloud completion aims to reconstruct complete 3D shapes from partial observations, which is a challenging problem due to severe occlusions and missing geometry. Despite recent advances in multimodal techniques that leverage complementary RGB images to compensate for missing geometry, most methods still follow a Completion-by-Inpainting paradigm, synthesizing missing structures from fused latent features. We empirically show that this paradigm often results in structural inconsistencies and topological artifacts due to limited geometric and semantic constraints. To address this, we rethink the task and propose a more robust paradigm, termed Completion-by-Correction, which begins with a topologically complete shape prior generated by a pretrained image-to-3D model and performs feature-space correction to align it with the partial observation. This paradigm shifts completion from unconstrained synthesis to guided refinement, enabling structurally consistent and observation-aligned reconstruction. Building upon this paradigm, we introduce PGNet, a multi-stage framework that conducts dual-feature encoding to ground the generative prior, synthesizes a coarse yet structurally aligned scaffold, and progressively refines geometric details via hierarchical correction. Experiments on the ShapeNetViPC dataset demonstrate the superiority of PGNet over state-of-the-art baselines in terms of average Chamfer Distance (-23.5%) and F-score (+7.1%).

---

## 3. DANCE: Density-agnostic and Class-aware Network for Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2511.07978v2](http://arxiv.org/abs/2511.07978v2)

**作者:** Da-Yeong Kim, Yeong-Jun Cho

**发布时间:** 2025-11-11

**备注:** 7 pages, 11 figures, Accepted to AAAI 2026 (to appear)

### GPT解析

### 总结

本文提出了一种名为DANCE的新框架，用于点云补全，能够在不同密度的输入下实现准确的3D结构恢复，同时保持语义一致性。

### 背景

点云补全旨在从不完整的3D扫描中恢复缺失的几何结构，这些不完整性通常由遮挡或有限传感器视角造成。现有方法通常假设固定的输入/输出密度或依赖基于图像的表示，不适合现实场景中的可变稀疏性和有限监督条件。

### 目的

开发一种密度无关且类别感知的网络框架，能够仅补全缺失区域同时保留观察到的几何结构，无需外部图像监督即可实现类别一致的补全。

### 方法

DANCE框架通过从多个视角进行基于射线的采样生成候选点，然后使用transformer解码器精炼点的位置并预测不透明度分数以确定点的有效性。此外，直接在几何特征上训练轻量级分类头，提供语义指导而不依赖外部图像数据。

### 主要发现

在PCN和MVP基准测试上，DANCE在准确性和结构一致性方面优于现有最先进方法，同时对变化的输入密度和噪声水平表现出鲁棒性。

### 结论

DANCE框架为点云补全提供了一种新方法，解决了现有方法在处理可变稀疏性和有限监督时的局限性，实现了更准确和语义一致的结果。

### 翻译

点云补全旨在从不完整的3D扫描中恢复缺失的几何结构，这些扫描常因遮挡或有限的传感器视角而不完整。现有方法通常假设固定的输入/输出密度或依赖基于图像的表示，使其不太适合具有可变稀疏性和有限监督的现实场景。本文介绍了密度无关和类别感知网络（DANCE），一种新框架，它只补全缺失区域同时保留观察到的几何结构。DANCE通过从多个视角进行基于射线的采样生成候选点。然后，transformer解码器精炼这些点的位置并预测不透明度分数，这决定了每个点是否包含在最终表面中的有效性。为了融入语义指导，直接在几何特征上训练轻量级分类头，实现无需外部图像监督的类别一致补全。在PCN和MVP基准上的大量实验表明，DANCE在准确性和结构一致性方面优于最先进方法，同时保持对变化输入密度和噪声水平的鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全中的两个关键问题：一是现有方法假设固定的输入和输出密度，这在实际场景中不实用；二是许多方法依赖基于图像的表示提供语义指导，导致3D形状与原始输入几何不一致。这个问题在现实世界中非常重要，因为自动驾驶、机器人和3D重建等应用需要完整的3D表示，而实际场景中的点云常常因遮挡、传感器限制而不完整。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，然后借鉴了NeRF的多视角射线采样策略来生成候选点，但调整为专注于3D几何而非渲染。他们设计了transformer解码器精炼候选点位置并预测不透明度，并引入分类头直接从几何特征学习语义指导。该方法确实借鉴了现有工作，如NeRF的射线采样、Transformer架构和编码器-解码器框架，但进行了专门调整以解决点云补全中的密度无关性和类别感知问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过密度无关且类别感知的方式补全点云，只生成缺失区域而不改变已观察到的几何结构，同时利用从点云本身学习的语义信息指导补全。整体流程包括：1)从多视角生成候选点；2)用编码器提取候选点和输入点云的特征；3)通过面Transformer处理特征，分类头预测类别，融合网络预测偏移量和不透明度；4)根据预测调整候选点位置并筛选有效点；5)将预测点与输入点云合并形成最终结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：密度无关设计，能处理不同密度的输入；类别感知能力，直接从点云几何学习语义；多视角射线采样策略；面Transformer结构。相比之前工作，不同之处在于：与传统固定密度方法相比，DANCE能适应不同密度的输入；与依赖图像表示的方法相比，它直接从3D几何学习语义，避免2D模型偏差导致的3D结构不一致；与仅关注缺失区域补全的方法相比，它结合了类别信息指导补全，更好地恢复类别特定结构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DANCE提出了一种新颖的点云补全框架，通过密度无关的设计和直接从点云几何学习的类别感知能力，实现了在不依赖图像监督的情况下高质量、结构一致的3D点云补全，同时能够灵活适应不同密度的输入点云。'}


### 论文摘要

Point cloud completion aims to recover missing geometric structures from incomplete 3D scans, which often suffer from occlusions or limited sensor viewpoints. Existing methods typically assume fixed input/output densities or rely on image-based representations, making them less suitable for real-world scenarios with variable sparsity and limited supervision. In this paper, we introduce Density-agnostic and Class-aware Network (DANCE), a novel framework that completes only the missing regions while preserving the observed geometry. DANCE generates candidate points via ray-based sampling from multiple viewpoints. A transformer decoder then refines their positions and predicts opacity scores, which determine the validity of each point for inclusion in the final surface. To incorporate semantic guidance, a lightweight classification head is trained directly on geometric features, enabling category-consistent completion without external image supervision. Extensive experiments on the PCN and MVP benchmarks show that DANCE outperforms state-of-the-art methods in accuracy and structural consistency, while remaining robust to varying input densities and noise levels.

---

## 4. UnSAMv2: Self-Supervised Learning Enables Segment Anything at Any Granularity

**论文链接:** [http://arxiv.org/abs/2511.13714v1](http://arxiv.org/abs/2511.13714v1)

**作者:** Junwei Yu, Trevor Darrell, XuDong Wang

**发布时间:** 2025-11-17

### GPT解析

### 总结

UnSAMv2是一种新型视觉分割模型，通过引入粒度控制嵌入和分而治之策略，实现了无需人工标注的任意粒度图像分割，显著提升了SAM-2的性能。

### 背景

Segment Anything Model (SAM)作为广泛采用的视觉基础模型，其控制分割粒度的能力有限，用户需要手动完善结果，且相同提示可能对应多个合理掩码，收集密集标注成本过高，使得监督解决方案不可行。

### 目的

解决SAM控制分割粒度有限的问题，实现无需人工标注的任意粒度分割。

### 方法

提出UnSAMv2模型，通过发现丰富的掩码-粒度对扩展UnSAM的分而治之策略，引入新颖的粒度控制嵌入实现分割尺度的精确、连续控制，仅使用6K无标注图像和0.02%的额外参数。

### 主要发现

UnSAMv2显著提升了SAM-2的性能，在交互式、全图像和视频分割任务中实现了任意粒度的分割，在超过11个基准测试上改进了NoC90(5.69→4.75)、1-IoU(58.0→73.1)和AR1000(49.6→68.3)等指标。

### 结论

少量无标注数据结合粒度感知的自监督学习方法可以释放视觉基础模型的潜力。

### 翻译

Segment Anything Model (SAM)家族已成为广泛采用的视觉基础模型，但其控制分割粒度的能力仍然有限。用户通常需要手动完善结果——通过添加更多提示或从预生成的掩码中选择——以达到所需的细节水平。这一过程可能存在歧义，因为相同的提示可能对应多个合理的掩码，且在所有粒度上收集密集标注成本过高，使得监督解决方案不可行。为解决这一限制，我们引入了UnSAMv2，它能够在无需人工标注的情况下实现任意粒度的分割。UnSAMv2通过发现丰富的掩码-粒度对扩展了UnSAM的分而治之策略，并引入了一种新颖的粒度控制嵌入，能够对分割尺度进行精确、连续的控制。值得注意的是，仅使用6K无标注图像和0.02%的额外参数，UnSAMv2就显著提升了SAM-2的性能，在交互式、全图像和视频分割任务中实现了任意粒度的分割。在超过11个基准测试上的评估显示，UnSAMv2改进了NoC90(5.69→4.75)、1-IoU(58.0→73.1)和AR1000(49.6→68.3)，表明少量无标注数据结合粒度感知的自监督学习方法可以释放视觉基础模型的潜力。


### 论文摘要

The Segment Anything Model (SAM) family has become a widely adopted vision foundation model, but its ability to control segmentation granularity remains limited. Users often need to refine results manually - by adding more prompts or selecting from pre-generated masks - to achieve the desired level of detail. This process can be ambiguous, as the same prompt may correspond to several plausible masks, and collecting dense annotations across all granularities is prohibitively expensive, making supervised solutions infeasible. To address this limitation, we introduce UnSAMv2, which enables segment anything at any granularity without human annotations. UnSAMv2 extends the divide-and-conquer strategy of UnSAM by discovering abundant mask-granularity pairs and introducing a novel granularity control embedding that enables precise, continuous control over segmentation scale. Remarkably, with only $6$K unlabeled images and $0.02\%$ additional parameters, UnSAMv2 substantially enhances SAM-2, achieving segment anything at any granularity across interactive, whole-image, and video segmentation tasks. Evaluated on over $11$ benchmarks, UnSAMv2 improves $\text{NoC}_{90}$ (5.69 $\rightarrow$ 4.75), 1-IoU (58.0 $\rightarrow$ 73.1), and $\text{AR}_{1000}$ (49.6 $\rightarrow$ 68.3), showing that small amounts of unlabeled data with a granularity-aware self-supervised learning method can unlock the potential of vision foundation models.

---

## 5. ST-ProC: A Graph-Prototypical Framework for Robust Semi-Supervised Travel Mode Identification

**论文链接:** [http://arxiv.org/abs/2511.13702v1](http://arxiv.org/abs/2511.13702v1)

**作者:** Luyao Niu, Nuoxian Huang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为ST-ProC的新型图原型多目标半监督学习框架，用于解决GPS轨迹中交通模式识别任务中的标签稀缺问题，显著提高了识别性能。

### 背景

GPS轨迹中的交通模式识别对于城市智能化至关重要，但标注成本高导致标签稀缺问题严重。现有的半监督学习方法不适合此任务，因为它们遭受灾难性确认偏差并忽略数据内在流形结构。

### 目的

开发一种新型半监督学习框架，解决GPS轨迹交通模式识别中的标签稀缺问题，克服现有方法的局限性，提高识别性能。

### 方法

提出ST-ProC框架，结合图原型核心与基础半监督学习支持。核心部分通过图正则化利用数据流形，通过原型锚定和新型感知边缘伪标签策略主动拒绝噪声。基础部分通过对比损失和教师-学生一致性损失提供支持和稳定，确保高质量表示和稳健优化。

### 主要发现

ST-ProC显著优于所有基线方法，在真实世界稀疏标签设置中展示了其有效性，相比FixMatch等最先进方法提升了21.5%的性能。

### 结论

ST-ProC框架有效解决了GPS轨迹交通模式识别中的标签稀缺问题，通过结合图正则化、原型锚定和感知边缘的伪标签策略，显著提高了识别性能。

### 翻译

从GPS轨迹中进行交通模式识别对城市智能化至关重要，但高昂的标注成本导致严重的标签稀缺问题。现有的半监督学习方法不适合此任务，因为它们遭受灾难性确认偏差并忽略了数据的内在流形结构。我们提出了ST-ProC，一种新型的图原型多目标半监督学习框架来解决这些局限性。我们的框架将图原型核心与基础半监督学习支持相结合。核心部分通过图正则化利用数据流形，通过原型锚定和一种新型的感知边缘伪标签策略主动拒绝噪声。这个核心部分通过基础对比损失和教师-学生一致性损失得到支持和稳定，确保高质量的表示和稳健的优化。ST-ProC以显著优势优于所有基线方法，展示了其在真实世界稀疏标签设置中的有效性，比FixMatch等最先进方法提升了21.5%的性能。


### 论文摘要

Travel mode identification (TMI) from GPS trajectories is critical for urban intelligence, but is hampered by the high cost of annotation, leading to severe label scarcity. Prevailing semi-supervised learning (SSL) methods are ill-suited for this task, as they suffer from catastrophic confirmation bias and ignore the intrinsic data manifold. We propose ST-ProC, a novel graph-prototypical multi-objective SSL framework to address these limitations. Our framework synergizes a graph-prototypical core with foundational SSL Support. The core exploits the data manifold via graph regularization, prototypical anchoring, and a novel, margin-aware pseudo-labeling strategy to actively reject noise. This core is supported and stabilized by foundational contrastive and teacher-student consistency losses, ensuring high-quality representations and robust optimization. ST-ProC outperforms all baselines by a significant margin, demonstrating its efficacy in real-world sparse-label settings, with a performance boost of 21.5% over state-of-the-art methods like FixMatch.

---

## 6. OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation

**论文链接:** [http://arxiv.org/abs/2511.13655v1](http://arxiv.org/abs/2511.13655v1)

**作者:** Henry Herzog, Favyen Bastani, Yawen Zhang, Gabriel Tseng, Joseph Redmon, Hadrien Sablon, Ryan Park, Jacob Morrison, Alexandra Buraczynski, Karen Farley, Joshua Hansen, Andrew Howe, Patrick Alan Johnson, Mark Otterlee, Ted Schmitt, Hunter Pitelka, Stephen Daspit, Rachel Ratner, Christopher Wilhelm, Sebastian Wood, Mike Jacobi, Hannah Kerner, Evan Shelhamer, Ali Farhadi, Ranjay Krishna, Patrick Beukema

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了OlmoEarth，一种专门为地球观测领域设计的多模态、时空基础模型，通过新颖的自监督学习方法实现了最先进的性能，并构建了端到端平台使非营利组织能够利用这些技术解决全球性问题。

### 背景

地球观测数据具有独特挑战：它同时具有空间特性(如图像)、序列特性(如视频或文本)以及高度多模态的特性，这给数据处理和分析带来了特殊困难。

### 目的

开发一种能够有效处理地球观测数据的多模态、时空基础模型，并在各种研究基准和真实世界任务中实现最先进的性能。

### 方法

设计了OlmoEarth模型，采用专门为地球观测领域开发的自监督学习公式、掩码策略和损失函数；构建了端到端平台，支持数据收集、标记、训练和推理；将模型部署为平台的核心组件。

### 主要发现

OlmoEarth在与12个其他基础模型的比较中表现最佳；在嵌入评估中，24个任务中的15个获得最佳性能；在完全微调后，29个任务中的19个表现最优。

### 结论

OlmoEarth平台将前沿基础模型和强大数据管理工具提供给解决全球问题的非营利组织和非政府组织，源代码、训练数据和预训练权重已公开可用。

### 翻译

地球观测数据带来了独特挑战：它像图像一样具有空间性，像视频或文本一样具有序列性，并且是高度多模态的。我们提出了OlmoEarth：一种多模态、时空基础模型，采用专门为地球观测领域设计的新颖自监督学习公式、掩码策略和损失函数。与12种其他基础模型相比，OlmoEarth在各种研究基准和来自外部合作伙伴的真实世界任务上实现了最先进的性能。在评估嵌入时，OlmoEarth在24个任务中的15个上取得了最佳性能，而在完全微调后，它在29个任务中的19个上表现最佳。我们将OlmoEarth部署为一个端到端平台的核心，用于地球观测模型的数据收集、标记、训练和推理。OlmoEarth平台将前沿基础模型和强大的数据管理工具提供给解决全球最大问题的非营利组织和非政府组织。OlmoEarth的源代码、训练数据和预训练权重可在https://github.com/allenai/olmoearth_pretrain获取。


### 论文摘要

Earth observation data presents a unique challenge: it is spatial like images, sequential like video or text, and highly multimodal. We present OlmoEarth: a multimodal, spatio-temporal foundation model that employs a novel self-supervised learning formulation, masking strategy, and loss all designed for the Earth observation domain. OlmoEarth achieves state-of-the-art performance compared to 12 other foundation models across a variety of research benchmarks and real-world tasks from external partners. When evaluating embeddings OlmoEarth achieves the best performance on 15 out of 24 tasks, and with full fine-tuning it is the best on 19 of 29 tasks. We deploy OlmoEarth as the backbone of an end-to-end platform for data collection, labeling, training, and inference of Earth observation models. The OlmoEarth Platform puts frontier foundation models and powerful data management tools into the hands of non-profits and NGOs working to solve the world's biggest problems. OlmoEarth source code, training data, and pre-trained weights are available at $\href{https://github.com/allenai/olmoearth_pretrain}{\text{https://github.com/allenai/olmoearth_pretrain}}$.

---

## 7. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning

**论文链接:** [http://arxiv.org/abs/2511.13654v1](http://arxiv.org/abs/2511.13654v1)

**作者:** Pascal Zimmer, Ghassan Karame

**发布时间:** 2025-11-17

**备注:** To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

### GPT解析

### 总结

本研究首次详细分析了优化超参数（学习率、权重衰减、动量和批量大小）如何影响模型对基于传输和基于查询的攻击的鲁棒性。研究结合理论和实验，涵盖了集中式训练、集成学习和分布式训练等多种实际部署场景。

### 背景

当前深度学习模型在安全部署中面临基于传输和基于查询的攻击威胁，而优化超参数对这些威胁的防御效果尚不清楚。

### 目的

探索优化超参数设计空间，以同时提高模型对基于传输和基于查询的攻击的鲁棒性。

### 方法

通过理论分析和实验验证，研究不同训练设置（集中式训练、集成学习、分布式训练）下超参数对模型鲁棒性的影响。

### 主要发现

对于基于传输的攻击，降低学习率可显著提高鲁棒性，最高达64%；对于基于查询的攻击，提高学习率可持续改善鲁棒性，最高达28%；分布式模型通过超参数调整获益最多，能比其他训练设置更有效地同时缓解两种攻击类型。

### 结论

优化超参数设计可以显著提高模型对两类攻击的鲁棒性，特别是分布式训练模式下的超参数调整效果最佳。

### 翻译

在本文中，我们首次详细分析了优化超参数（如学习率、权重衰减、动量和批量大小）如何影响对基于传输和基于查询的攻击的鲁棒性。在理论和实验的支持下，我们的研究涵盖了多种实际部署场景，包括集中式训练、集成学习和分布式训练。我们发现了一个显著的两极分化现象：对于基于传输的攻击，降低学习率可以显著提高鲁棒性，最高达64%。相反，对于基于查询的攻击，提高学习率可以持续改善鲁棒性，在各种设置和数据分布下最高达28%。利用这些发现，我们首次探索了优化超参数设计空间，以同时提高对两类攻击的鲁棒性。我们的结果表明，分布式模型从超参数调整中获益最多，实现了显著的权衡，比其他训练设置更有效地同时缓解两种攻击类型。


### 论文摘要

In this paper, we present the first detailed analysis of how optimization hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the optimization hyperparameter design space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.

---

## 8. RAC-DMVC: Reliability-Aware Contrastive Deep Multi-View Clustering under Multi-Source Noise

**论文链接:** [http://arxiv.org/abs/2511.13561v1](http://arxiv.org/abs/2511.13561v1)

**作者:** Shihao Dong, Yue Liu, Xiaotong Zhou, Yuhui Zheng, Huiying Xu, Xinzhong Zhu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为RAC-DMVC的新型多视图聚类框架，能够处理多种噪声环境下的聚类问题，包括缺失噪声和观测噪声。

### 背景

多视图聚类(MVC)是一个基础但具有挑战性的任务，旨在无监督地将多视图数据分离成不同的簇。为了增强其在实际场景中的适用性，需要解决在多种噪声下的聚类问题。

### 目的

提出一种能够在噪声环境下进行鲁棒表示学习的多视图聚类框架，有效处理观测噪声和缺失噪声，提高聚类性能。

### 方法

RAC-DMVC框架通过构建可靠性图指导鲁棒表示学习；针对观测噪声，引入跨视图重建增强数据层面鲁棒性，并使用可靠性感知噪声对比学习减轻噪声表示引起的偏差；针对缺失噪声，设计双注意力插补捕获跨视图共享信息；添加自监督聚类蒸馏模块优化表示。

### 主要发现

在五个基准数据集上的实验表明，RAC-DMVC在多个评估指标上优于最先进的方法，并能在不同比例的噪声下保持出色性能。

### 结论

RAC-DMVC框架能有效处理多源噪声环境下的多视图聚类问题，通过多种创新技术提高了聚类性能和鲁棒性。

### 翻译

多视图聚类(MVC)旨在无监督地将多视图数据分离成不同的簇，是一项基础但具有挑战性的任务。为了增强其在实际场景中的适用性，本文解决了一个更具挑战性的任务：多源噪声下的多视图聚类，包括缺失噪声和观测噪声。为此，我们提出了一个新颖的框架：可靠性感知对比深度多视图聚类(RAC-DMVC)，该框架构建可靠性图以指导噪声环境下的鲁棒表示学习。具体而言，为解决观测噪声，我们引入跨视图重建来增强数据层面的鲁棒性，并使用可靠性感知的噪声对比学习来减轻由噪声表示引起的正负对选择偏差。为处理缺失噪声，我们设计了双注意力插补来捕获跨视图的共享信息，同时保留视图特定特征。此外，自监督聚类蒸馏模块进一步优化了学习到的表示并提高了聚类性能。在五个基准数据集上的大量实验表明，RAC-DMVC在多个评估指标上优于最先进的方法，并在不同比例的噪声下保持出色性能。


### 论文摘要

Multi-view clustering (MVC), which aims to separate the multi-view data into distinct clusters in an unsupervised manner, is a fundamental yet challenging task. To enhance its applicability in real-world scenarios, this paper addresses a more challenging task: MVC under multi-source noises, including missing noise and observation noise. To this end, we propose a novel framework, Reliability-Aware Contrastive Deep Multi-View Clustering (RAC-DMVC), which constructs a reliability graph to guide robust representation learning under noisy environments. Specifically, to address observation noise, we introduce a cross-view reconstruction to enhances robustness at the data level, and a reliability-aware noise contrastive learning to mitigates bias in positive and negative pairs selection caused by noisy representations. To handle missing noise, we design a dual-attention imputation to capture shared information across views while preserving view-specific features. In addition, a self-supervised cluster distillation module further refines the learned representations and improves the clustering performance. Extensive experiments on five benchmark datasets demonstrate that RAC-DMVC outperforms SOTA methods on multiple evaluation metrics and maintains excellent performance under varying ratios of noise.

---

## 9. TSE-Net: Semi-supervised Monocular Height Estimation from Single Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2511.13552v1](http://arxiv.org/abs/2511.13552v1)

**作者:** Sining Chen, Xiao Xiang Zhu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为TSE-Net的半监督单目高度估计方法，通过自训练流水线整合教师、学生和考试网络，利用未标记数据提升模型性能，解决了高质量标注数据稀缺的问题。

### 背景

单目高度估计在遥感3D感知中起关键作用，是昂贵多视图或LiDAR方法的低成本替代方案。深度学习虽提升了单目高度估计能力，但受限于标记数据获取的高昂成本和大量人力需求。

### 目的

通过半监督学习框架利用大量未标记数据，使模型从未标记样本中提取有用信息，提高预测性能，克服高质量标注数据稀缺的限制。

### 方法

提出TSE-Net半监督单目高度估计自训练流水线，整合教师、学生和考试网络。学生网络用教师网络生成的伪标签训练未标记数据，考试网络作为学生网络的时间集成以稳定性能。教师网络采用联合回归和分类模型：回归分支预测高度值作为伪标签，分类分支预测高度值类别及概率用于过滤伪标签。高度值类别通过分层双切策略定义，解决高度长尾分布问题，类别概率使用Plackett-Luce模型校准。

### 主要发现

在三个不同分辨率和成像模式的datasets上评估了所提流水线，证明其能有效利用未标记数据提升单目高度估计性能。

### 结论

TSE-Net通过半监督学习有效解决了高质量标注数据稀缺的问题，提升了单目高度估计的性能和泛化能力。

### 翻译

单目高度估计在遥感3D感知中起着关键作用，为多视图或基于LiDAR的方法提供了具有成本效益的替代方案。虽然深度学习显著提升了单目高度估计的能力，但这些方法仍然受到标记数据可用性的根本限制，因为大规模获取这些数据既昂贵又耗时。高质量标注的稀缺阻碍了现有模型的泛化能力和性能。为了克服这一限制，我们提出通过半监督学习框架利用大量未标记数据，使模型能够从未标记样本中提取有用信息并提高其预测性能。在这项工作中，我们引入了TSE-Net，一种用于半监督单目高度估计的自训练流水线。该流水线集成了教师、学生和考试网络。学生网络使用教师网络生成的伪标签在未标记数据上进行训练，而考试网络则作为学生网络的时间集成以稳定性能。教师网络被制定为联合回归和分类模型：回归分支预测作为伪标签的高度值，分类分支预测高度值类别及其概率，用于过滤伪标签。高度值类别使用分层双切策略定义，以解决高度固有的长尾分布问题，预测的类别概率使用Plackett-Luce模型进行校准，以反映伪标签的预期准确性。我们在三个不同分辨率和成像模式的datasets上评估了所提流水线。代码可在https://github.com/zhu-xlab/tse-net获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决单目高度估计中标记数据稀缺的问题。这个问题很重要，因为获取真实高度标记数据需要昂贵的LiDAR或立体系统，成本高且耗时。单目高度估计对3D城市建筑建模、森林冠层高度模型等应用至关重要，但现有方法依赖大量标记数据限制了其应用。单目方法相比多视图或LiDAR方法有成本低、遮挡少、数据可用性高等优势，却受标记数据缺乏的制约。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有半监督学习方法在分类和分割任务中的成功应用，发现它们不能直接应用于回归任务。受先前将高度估计重新表述为分类或混合分类-回归任务的工作启发，作者设计了基于伪标签过滤的机制。他们引入Plackett-Luce模型校准类概率，使其与伪标签误差一致，并设计分层双切策略处理高度值的长尾分布。该方法借鉴了自训练半监督学习框架、分类-回归多任务网络、基于一致性的半监督学习方法以及高度估计中的分类方法（如AdaBins）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大量未标记数据解决标记数据稀缺问题，通过教师-学生-考试的自训练流程，让模型从未标记数据中学习。教师网络采用多任务架构生成伪标签和置信度，学生网络学习这些伪标签，考试网络作为学生网络的临时集成提高稳定性。实现流程包括：1) 教师和学生在标记数据上联合训练；2) 对未标记数据应用弱增强和强增强；3) 教师生成伪标签并通过类概率过滤；4) 学生使用过滤后的伪标签训练；5) 考试网络作为学生网络的EMA更新；6) 推理时仅使用考试网络。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个用于单目高度估计的半监督学习框架TSE-Net；2) 额外的考试网络提高性能稳定性；3) 教师网络采用分类-回归多任务架构；4) 使用Plackett-Luce模型对齐类概率与伪标签误差；5) 分层双切策略处理高度值长尾分布。相比之前工作，传统半监督方法不适用于回归任务，现有高度估计方法需要大量标记数据，分层双切策略比均匀离散化更适合长尾分布，考试网络解决了传统自训练中教师网络可能无法代表学生最优状态的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TSE-Net通过创新的教师-学生-考试半自训练框架和分层双切策略，有效解决了单目高度估计中标记数据稀缺的问题，显著提升了模型在标记数据有限情况下的性能。'}


### 论文摘要

Monocular height estimation plays a critical role in 3D perception for remote sensing, offering a cost-effective alternative to multi-view or LiDAR-based methods. While deep learning has significantly advanced the capabilities of monocular height estimation, these methods remain fundamentally limited by the availability of labeled data, which are expensive and labor-intensive to obtain at scale. The scarcity of high-quality annotations hinders the generalization and performance of existing models. To overcome this limitation, we propose leveraging large volumes of unlabeled data through a semi-supervised learning framework, enabling the model to extract informative cues from unlabeled samples and improve its predictive performance. In this work, we introduce TSE-Net, a self-training pipeline for semi-supervised monocular height estimation. The pipeline integrates teacher, student, and exam networks. The student network is trained on unlabeled data using pseudo-labels generated by the teacher network, while the exam network functions as a temporal ensemble of the student network to stabilize performance. The teacher network is formulated as a joint regression and classification model: the regression branch predicts height values that serve as pseudo-labels, and the classification branch predicts height value classes along with class probabilities, which are used to filter pseudo-labels. Height value classes are defined using a hierarchical bi-cut strategy to address the inherent long-tailed distribution of heights, and the predicted class probabilities are calibrated with a Plackett-Luce model to reflect the expected accuracy of pseudo-labels. We evaluate the proposed pipeline on three datasets spanning different resolutions and imaging modalities. Codes are available at https://github.com/zhu-xlab/tse-net.

---

## 10. Robust Defense Strategies for Multimodal Contrastive Learning: Efficient Fine-tuning Against Backdoor Attacks

**论文链接:** [http://arxiv.org/abs/2511.13545v1](http://arxiv.org/abs/2511.13545v1)

**作者:** Md. Iqbal Hossain, Afia Sajeeda, Neeresh Kumar Perla, Ming Shao

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了一种创新策略，用于增强多模态对比学习模型对抗后门攻击的鲁棒性，能够高效识别被污染CLIP模型中的后门触发器、受害样本和标签，并通过两种算法修复被污染的模型。

### 背景

多模态深度学习模型如CLIP在图像文本理解和分类任务中应用广泛，但这些模型容易受到对抗攻击特别是后门攻击。现有防御方法通常需要从头训练或使用大型数据集微调，无法精确定位受影响的特定标签。

### 目的

提出一种增强多模态对比学习模型对抗后门攻击鲁棒性的创新策略，能够高效识别被污染CLIP模型中的后门触发器、受害样本和标签。

### 方法

引入图像分割'oracle'作为被污染CLIP输出的监督器，开发两种算法：(1)区分CLIP和Oracle的知识以识别潜在触发器；(2)精确定位受影响的标签和受害样本，构建紧凑的微调数据集。

### 主要发现

通过所提出的方法可以修复被污染的CLIP模型，消除后门效应。在视觉识别基准上的广泛实验表明该策略在基于CLIP的后门防御中有效。

### 结论

提出的创新策略能够有效识别和修复CLIP模型中的后门攻击，比现有防御方法更高效，因为它能够精确定位问题所在。

### 翻译

多模态深度学习模型的出现，如CLIP，已从图像文本理解到分类任务等各种应用中开辟了新的前沿。然而，这些模型容易受到对抗攻击，特别是后门攻击，这些攻击可以微妙地操纵模型行为。此外，现有的防御方法通常涉及从头开始训练或使用大型数据集进行微调，而没有精确定位受影响的特定标签。在本研究中，我们引入了一种创新策略，以增强多模态对比学习模型对抗此类攻击的鲁棒性。特别是，给定一个被污染的CLIP模型，我们的方法能够以高效的方式识别后门触发器，并精确定位受害样本和标签。为此，我们引入了一个图像分割'oracle'作为被污染CLIP输出的监督器。我们开发了两种算法来修复被污染的模型：(1)区分CLIP和Oracle的知识以识别潜在的触发器；(2)精确定位受影响的标签和受害样本，并构建一个紧凑的微调数据集。有了这些知识，我们可以修复被污染的CLIP模型以消除后门效应。在视觉识别基准上的广泛实验证明了我们的策略在基于CLIP的后门防御中是有效的。


### 论文摘要

The advent of multimodal deep learning models, such as CLIP, has unlocked new frontiers in a wide range of applications, from image-text understanding to classification tasks. However, these models are not safe for adversarial attacks, particularly backdoor attacks, which can subtly manipulate model behavior. Moreover, existing defense methods typically involve training from scratch or fine-tuning using a large dataset without pinpointing the specific labels that are affected. In this study, we introduce an innovative strategy to enhance the robustness of multimodal contrastive learning models against such attacks. In particular, given a poisoned CLIP model, our approach can identify the backdoor trigger and pinpoint the victim samples and labels in an efficient manner. To that end, an image segmentation ``oracle'' is introduced as the supervisor for the output of the poisoned CLIP. We develop two algorithms to rectify the poisoned model: (1) differentiating between CLIP and Oracle's knowledge to identify potential triggers; (2) pinpointing affected labels and victim samples, and curating a compact fine-tuning dataset. With this knowledge, we are allowed to rectify the poisoned CLIP model to negate backdoor effects. Extensive experiments on visual recognition benchmarks demonstrate our strategy is effective in CLIP-based backdoor defense.

---

## 11. BootOOD: Self-Supervised Out-of-Distribution Detection via Synthetic Sample Exposure under Neural Collapse

**论文链接:** [http://arxiv.org/abs/2511.13539v1](http://arxiv.org/abs/2511.13539v1)

**作者:** Yuanchao Wang, Tian Qin, Eduardo Valle, Bruno Abrahao

**发布时间:** 2025-11-17

**备注:** 8 pages

### GPT解析

### 总结

BootOOD是一种完全自监督的分布外(OOD)检测框架，专门处理语义上具有挑战性的OOD样本，通过合成伪OOD特征和利用神经崩溃现象，在多个基准数据集上表现优异。

### 背景

分布外(OOD)检测对于在安全敏感环境中部署图像分类器至关重要，但现有检测器在OOD样本与分布内(ID)类别语义相似时往往表现不佳。

### 目的

开发一种能够有效处理语义相似OOD样本的自监督OOD检测框架，仅从ID数据中引导学习。

### 方法

BootOOD通过对ID表示的简单变换合成伪OOD特征，利用神经崩溃现象(ID特征围绕类均值紧密聚类且特征范数一致)，并引入轻量级辅助头基于特征范数进行半径分类，将OOD检测与主分类器解耦。

### 主要发现

实验表明BootOOD在CIFAR-10、CIFAR-100和ImageNet-200上优于先前的后验方法，超越了没有异常值暴露的训练方法，并与最先进的异常值暴露方法具有竞争力，同时保持或提高了ID准确率。

### 结论

BootOOD通过放宽OOD检测的要求(学习使OOD样本特征范数小于ID特征)成功处理了语义上接近ID的OOD样本，是一种高效的自监督OOD检测方法。

### 翻译

分布外(OOD)检测对于在安全敏感环境中部署图像分类器至关重要，然而当OOD样本与分布内(ID)类别语义相似时，现有检测器往往表现不佳。我们提出了BootOOD，一种完全自监督的OOD检测框架，仅从ID数据中引导，专门设计用于处理语义上具有挑战性的OOD样本。BootOOD通过对ID表示的简单变换合成伪OOD特征，并利用神经崩溃(NC)现象，其中ID特征围绕类均值紧密聚类且特征范数一致。与旨在将OOD特征约束到与崩溃ID均值正交的子空间中的先前方法不同，BootOOD引入了一个轻量级辅助头，对特征范数进行基于半径的分类。这种设计将OOD检测与主分类器解耦，并施加了放宽的要求：学习使OOD样本的特征范数小于ID特征，当ID和OOD语义接近时，这更容易满足。在CIFAR-10、CIFAR-100和ImageNet-200上的实验表明，BootOOD优于先前的后验方法，超越了没有异常值暴露的训练方法，并与最先进的异常值暴露方法具有竞争力，同时保持或提高了ID准确率。


### 论文摘要

Out-of-distribution (OOD) detection is critical for deploying image classifiers in safety-sensitive environments, yet existing detectors often struggle when OOD samples are semantically similar to the in-distribution (ID) classes. We present BootOOD, a fully self-supervised OOD detection framework that bootstraps exclusively from ID data and is explicitly designed to handle semantically challenging OOD samples. BootOOD synthesizes pseudo-OOD features through simple transformations of ID representations and leverages Neural Collapse (NC), where ID features cluster tightly around class means with consistent feature norms. Unlike prior approaches that aim to constrain OOD features into subspaces orthogonal to the collapsed ID means, BootOOD introduces a lightweight auxiliary head that performs radius-based classification on feature norms. This design decouples OOD detection from the primary classifier and imposes a relaxed requirement: OOD samples are learned to have smaller feature norms than ID features, which is easier to satisfy when ID and OOD are semantically close. Experiments on CIFAR-10, CIFAR-100, and ImageNet-200 show that BootOOD outperforms prior post-hoc methods, surpasses training-based methods without outlier exposure, and is competitive with state-of-the-art outlier-exposure approaches while maintaining or improving ID accuracy.

---

## 12. Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew

**论文链接:** [http://arxiv.org/abs/2511.13535v1](http://arxiv.org/abs/2511.13535v1)

**作者:** Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska

**发布时间:** 2025-11-17

### GPT解析

### 总结

研究揭示了一种新型攻击方法，通过色度扰动损害机器学习模型的可解释性而不影响准确性，挑战了正确预测意味着可靠解释的假设。

### 背景

机器学习模型越来越多地部署在安全关键领域，可视化解释技术已成为支持透明度的重要工具。

### 目的

揭示一类新的攻击，这些攻击会损害模型的可解释性而不影响准确性。

### 方法

提出了一种称为'色度扰动模块'的显著性感知攻击框架，通过改变前景和背景之间的颜色对比来系统性地制作对抗样本，从而破坏解释的保真度。

### 主要发现

小颜色扰动可在联邦学习中改变模型显著性图，使其偏离语义有意义区域同时保持预测不变；这些扰动在训练中累积，隐蔽持久地毒害全局模型内部特征归因；标准训练管道不足以检测或减轻解释质量下降，尤其在联邦学习环境中；攻击使Grad-CAM解释中峰值激活重叠度降低35%以上，同时保持96%以上的分类准确率。

### 结论

挑战了模型审计中正确预测意味着可靠解释的假设，证明可解释性本身可能成为一个攻击面。

### 翻译

随着机器学习模型越来越多地部署在安全关键领域，可视化解释技术已成为支持透明度的必要工具。在这项工作中，我们揭示了一类新的攻击，这些攻击会损害模型的可解释性而不影响准确性。具体而言，我们表明在联邦学习环境中，由对抗客户端应用的小颜色扰动可以使模型的显著性图偏离语义上有意义的区域，同时保持预测不变。所提出的显著性感知攻击框架，称为色度扰动模块，通过以前景和背景之间改变颜色对比度的方式系统性地制作对抗样本，从而破坏解释的保真度。这些扰动在训练过程中累积，以隐蔽和持久的方式毒害全局模型的内部特征归因。我们的发现挑战了模型审计中的一个常见假设，即正确的预测意味着可靠的解释，并证明可解释性本身可能成为一个攻击面。我们在多个数据集上评估了这一漏洞，表明标准训练管道不足以检测或减轻解释质量的下降，特别是在联邦学习环境中，微妙的颜色扰动更难辨别。我们的攻击在所有评估的数据集上将Grad-CAM解释中的峰值激活重叠度降低了高达35%，同时保持分类准确率在96%以上。


### 论文摘要

As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.

---

## 13. Quantum Machine Learning via Contrastive Training

**论文链接:** [http://arxiv.org/abs/2511.13497v1](http://arxiv.org/abs/2511.13497v1)

**作者:** Liudmila A. Zhukas, Vivian Ni Zhang, Qiang Miao, Qingfeng Wang, Marko Cetina, Jungsang Kim, Lawrence Carin, Christopher Monroe

**发布时间:** 2025-11-17

**备注:** 7 figures, 20 pages total

### GPT解析

### 总结

该研究提出了一种量子表示的自监督预训练方法，通过从未标记数据中学习不变性减少对标记数据的依赖，在离子阱量子计算机上实现，并在标记数据有限的情况下展现出显著优势。

### 背景

量子机器学习随着大规模经典机器学习和量子技术的快速发展而受到越来越多的关注。与经典机器学习类似，QML模型也面临标记数据稀缺的挑战，尤其是随着模型规模和复杂度的增加。

### 目的

引入量子表示的自监督预训练方法，通过从未标记的示例中学习不变性来减少对标记数据的依赖，提高量子机器学习在有限标记数据下的性能。

### 方法

在可编程的离子阱量子计算机上实现这一范式，将图像编码为量子状态。在硬件上进行现场对比预训练，得到一种表示，经过微调后进行图像分类。

### 主要发现

1)在标记训练数据有限的情况下，性能改进特别显著；2)学习到的不变性可以泛化到预训练图像样本之外；3)该方法从测量的量子重叠中推导相似性，并在硬件上执行所有训练和分类阶段。

### 结论

这些结果建立了一条标签高效的量子表示学习路径，与量子原生数据集直接相关，并为处理更大的经典输入提供了明确的路径。

### 翻译

量子机器学习(QML)随着大规模经典机器学习和量子技术的快速发展而受到越来越多的关注。与经典机器学习类似，QML模型也面临标记数据稀缺的挑战，尤其是随着模型规模和复杂度的增加。在此，我们引入了量子表示的自监督预训练，通过从未标记的示例中学习不变性来减少对标记数据的依赖。我们在可编程的离子阱量子计算机上实现了这一范式，将图像编码为量子状态。在硬件上进行现场对比预训练得到的表示，经过微调后，比从随机初始化训练的模型能以更高的平均测试准确率和更低的运行间变异性对图像族进行分类。在标记训练数据有限的情况下，性能改进特别显著。我们展示学习到的不变性可以泛化到预训练图像样本之外。与先前工作不同，我们的管道从测量的量子重叠中推导相似性，并在硬件上执行所有训练和分类阶段。这些结果建立了一条标签高效的量子表示学习路径，与量子原生数据集直接相关，并为处理更大的经典输入提供了明确的路径。


### 论文摘要

Quantum machine learning (QML) has attracted growing interest with the rapid parallel advances in large-scale classical machine learning and quantum technologies. Similar to classical machine learning, QML models also face challenges arising from the scarcity of labeled data, particularly as their scale and complexity increase. Here, we introduce self-supervised pretraining of quantum representations that reduces reliance on labeled data by learning invariances from unlabeled examples. We implement this paradigm on a programmable trapped-ion quantum computer, encoding images as quantum states. In situ contrastive pretraining on hardware yields a representation that, when fine-tuned, classifies image families with higher mean test accuracy and lower run-to-run variability than models trained from random initialization. Performance improvement is especially significant in regimes with limited labeled training data. We show that the learned invariances generalize beyond the pretraining image samples. Unlike prior work, our pipeline derives similarity from measured quantum overlaps and executes all training and classification stages on hardware. These results establish a label-efficient route to quantum representation learning, with direct relevance to quantum-native datasets and a clear path to larger classical inputs.

---

## 14. Contact-Safe Reinforcement Learning with ProMP Reparameterization and Energy Awareness

**论文链接:** [http://arxiv.org/abs/2511.13459v1](http://arxiv.org/abs/2511.13459v1)

**作者:** Bingkun Huang, Yuhe Gong, Zewen Yang, Tianyu Ren, Luis Figueredo

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种基于任务空间和能量安全框架的接触丰富操作任务解决方案，通过结合近端策略优化(PPO)和运动原语生成可靠安全的任务空间轨迹，并融入能量感知的笛卡尔阻抗控制器确保机器人与环境的安全交互。

### 背景

基于马尔可夫决策过程(MDP)的强化学习方法主要应用于机器人关节空间，通常依赖有限的任务信息和部分3D环境感知；传统MDP和分步强化学习方法忽略了任务空间操作中丰富的接触信息，特别是接触安全性和鲁棒性。

### 目的

解决接触丰富的操作任务，开发一个任务空间、能量安全的框架，以处理复杂环境中的机器人操作。

### 方法

采用任务空间、能量安全框架，结合近端策略优化(PPO)和运动原语生成可靠安全的任务空间轨迹，并在框架中集成了能量感知的笛卡尔阻抗控制器目标，确保机器人与环境的安全交互。

### 主要发现

实验结果表明，所提框架在处理3D环境中各种表面上的任务时优于现有方法，实现了高成功率、平滑轨迹和能量安全的交互。

### 结论

任务空间强化学习方法相比传统MDP方法在轨迹一致性、任务意识和复杂机器人任务整体性能方面具有优势；所提出的框架能有效处理接触丰富的操作任务。

### 翻译

基于马尔可夫决策过程(MDP)的强化学习方法主要应用于机器人关节空间，通常依赖于有限的特定任务信息和部分3D环境感知。相比之下，基于片段的强化学习在轨迹一致性、任务意识和复杂机器人任务的整体性能方面已显示出优于传统基于MDP方法的优势。此外，传统的分步和基于片段的强化学习方法常常忽略了任务空间操作中固有的丰富接触信息，特别是考虑到接触安全和鲁棒性。在这项工作中，我们采用任务空间、能量安全框架来解决接触丰富的操作任务，通过结合近端策略优化(PPO)和运动原语生成可靠且安全的任务空间轨迹。此外，在所提出的框架中集成了能量感知的笛卡尔阻抗控制器目标，以确保机器人与环境之间的安全交互。我们的实验结果表明，所提出的框架在处理3D环境中各种类型表面上的任务时优于现有方法，实现了高成功率、平滑轨迹和能量安全的交互。


### 论文摘要

Reinforcement learning (RL) approaches based on Markov Decision Processes (MDPs) are predominantly applied in the robot joint space, often relying on limited task-specific information and partial awareness of the 3D environment. In contrast, episodic RL has demonstrated advantages over traditional MDP-based methods in terms of trajectory consistency, task awareness, and overall performance in complex robotic tasks. Moreover, traditional step-wise and episodic RL methods often neglect the contact-rich information inherent in task-space manipulation, especially considering the contact-safety and robustness. In this work, contact-rich manipulation tasks are tackled using a task-space, energy-safe framework, where reliable and safe task-space trajectories are generated through the combination of Proximal Policy Optimization (PPO) and movement primitives. Furthermore, an energy-aware Cartesian Impedance Controller objective is incorporated within the proposed framework to ensure safe interactions between the robot and the environment. Our experimental results demonstrate that the proposed framework outperforms existing methods in handling tasks on various types of surfaces in 3D environments, achieving high success rates as well as smooth trajectories and energy-safe interactions.

---

## 15. Artificial Intelligence-Enabled Spirometry for Early Detection of Right Heart Failure

**论文链接:** [http://arxiv.org/abs/2511.13457v1](http://arxiv.org/abs/2511.13457v1)

**作者:** Bin Liu, Qinghao Zhao, Yuxi Zhou, Zhejun Sun, Kaijie Lei, Deyun Zhang, Shijia Geng, Shenda Hong

**发布时间:** 2025-11-17

**备注:** 19 pages, 5 figures

### GPT解析

### 总结

本研究提出了一种自监督表示学习方法，通过肺活量图时间序列和人口统计学数据结合，用于早期检测肺心病患者中的右心衰竭。

### 背景

右心衰竭(RHF)是一种以右心室结构或功能异常为特征的疾病，与高发病率和死亡率相关。肺部疾病常常导致右心室负荷增加，进而引发RHF。从潜在肺部疾病患者中筛选出发展为RHF的肺心病患者非常重要。

### 目的

提出一种自监督表示学习方法，用于早期检测肺心病患者中的RHF，使用肺活量图时间序列来早期预测RHF患者。

### 方法

模型分为两个阶段：第一阶段是基于自监督表示学习的肺活量图嵌入网络训练过程，使用变分自编码器从数据增强的无标签数据中学习肺活量图时间序列的低维表示；第二阶段将低维表示与人口统计学信息融合，输入到CatBoost分类器中进行RHF预测。在26,617名英国生物银行参与者的子集上进行训练和测试。

### 主要发现

该模型在检测RHF时达到了0.7501的AUROC。在临床高风险亚组中，74名慢性肾脏病患者的测试集上AUROC为0.8194，64名心脏瓣膜病患者的测试集上AUROC为0.8413，显示出模型在预测高风险人群中RHF的潜力。

### 结论

本研究提出的方法结合肺活量图时间序列和人口统计学数据，在临床实践中显示出早期检测RHF的潜在前景。

### 翻译

右心衰竭(RHF)是一种以右心室(RV)结构或功能异常为特征的疾病，与高发病率和死亡率相关。肺部疾病常常导致右心室负荷增加，进而引发RHF。因此，从潜在肺部疾病患者中筛选出发展为RHF的肺心病患者非常重要。本研究提出了一种自监督表示学习方法，用于早期检测肺心病患者中的RHF，使用肺活量图时间序列来早期预测RHF患者。所提出的模型分为两个阶段。第一阶段是基于自监督表示学习的肺活量图嵌入网络训练过程，其中变分自编码器的编码器从数据增强的无标签数据中学习肺活量图时间序列的鲁棒低维表示。第二阶段，将这种低维表示与人口统计学信息融合，输入到CatBoost分类器中进行下游RHF预测任务。在26,617名英国生物银行参与者的精心挑选子集上进行训练和测试，我们的模型在检测RHF时达到了0.7501的AUROC，显示出强大的人群水平区分能力。我们在高风险临床亚组中进一步评估了模型，在74名慢性肾脏病患者的测试集上达到0.8194的AUROC，在64名心脏瓣膜病患者的测试集上达到0.8413的AUROC。这些结果突显了模型在预测临床高风险人群中RHF的潜在效用。总之，本研究提出了一种结合肺活量图时间序列和人口统计学数据的自监督表示学习方法，在临床实践中显示出早期检测RHF的 promising 潜力。


### 论文摘要

Right heart failure (RHF) is a disease characterized by abnormalities in the structure or function of the right ventricle (RV), which is associated with high morbidity and mortality. Lung disease often causes increased right ventricular load, leading to RHF. Therefore, it is very important to screen out patients with cor pulmonale who develop RHF from people with underlying lung diseases. In this work, we propose a self-supervised representation learning method to early detecting RHF from patients with cor pulmonale, which uses spirogram time series to predict patients with RHF at an early stage. The proposed model is divided into two stages. The first stage is the self-supervised representation learning-based spirogram embedding (SLSE) network training process, where the encoder of the Variational autoencoder (VAE-encoder) learns a robust low-dimensional representation of the spirogram time series from the data-augmented unlabeled data. Second, this low-dimensional representation is fused with demographic information and fed into a CatBoost classifier for the downstream RHF prediction task. Trained and tested on a carefully selected subset of 26,617 individuals from the UK Biobank, our model achieved an AUROC of 0.7501 in detecting RHF, demonstrating strong population-level distinction ability. We further evaluated the model on high-risk clinical subgroups, achieving AUROC values of 0.8194 on a test set of 74 patients with chronic kidney disease (CKD) and 0.8413 on a set of 64 patients with valvular heart disease (VHD). These results highlight the model's potential utility in predicting RHF among clinically elevated-risk populations. In conclusion, this study presents a self-supervised representation learning approach combining spirogram time series and demographic data, demonstrating promising potential for early RHF detection in clinical practice.

---

## 16. PAST: A Primary-Auxiliary Spatio-Temporal Network for Traffic Time Series Imputation

**论文链接:** [http://arxiv.org/abs/2511.13414v1](http://arxiv.org/abs/2511.13414v1)

**作者:** Hanwen Hu, Zimo Wen, Shiyou Qian, Jian Co

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种名为PAST（Primary-Auxiliary Spatio-Temporal network）的新型网络模型，通过区分主要模式和辅助模式，有效处理交通时间序列中的多种缺失数据类型。

### 背景

交通时间序列插值对智能交通系统的安全和可靠性至关重要，但不同类型的缺失数据（随机缺失、纤维缺失和块状缺失）使插值任务具有挑战性。现有模型难以适应随机缺失位置且无法学习长期依赖关系。

### 目的

开发一种能够处理各种缺失数据条件并学习长期依赖关系的模型，将模式分为主要模式（源自数据点内部关系）和辅助模式（受外部因素影响）。

### 方法

提出PAST网络，包含图集成模块（GIM）和交叉门控模块（CGM）。GIM通过动态图捕获主要模式，CGM提取辅助模式，两模块通过共享隐藏向量交互并在集成自监督框架下训练。

### 主要发现

在三个数据集的27种缺失数据条件下，PAST的插值准确性优于七种最先进基线方法，RMSE提高26.2%，MAE提高31.6%。

### 结论

PAST模型通过区分和整合主要模式和辅助模式，有效解决了交通时间序列中多种缺失数据的插值挑战，特别是在处理随机缺失和学习长期依赖关系方面表现出色。

### 翻译

交通时间序列插值对智能交通系统的安全和可靠性至关重要，而多种类型的缺失数据，包括随机缺失、纤维缺失和块状缺失，使插值任务变得具有挑战性。现有模型通常基于数据点之间的关系来解耦和分别建模空间和时间模式。然而，这些方法难以适应随机缺失位置，并且无法学习长期和大范围的依赖关系，这在大量缺失条件下是必不可少的。在本文中，模式被分为两种类型以处理各种缺失数据条件：主要模式源自数据点之间的内部关系，而辅助模式受时间戳和节点属性等外部因素影响。相应地，我们提出了主-辅助时空网络（PAST）。它包含一个图集成模块（GIM）和一个交叉门控模块（CGM）。GIM通过具有区间感知dropout和多阶卷积的动态图捕获主要模式，CGM通过对嵌入的外部特征进行双向门控来提取辅助模式。这两个模块通过共享隐藏向量进行交互，并在集成自监督框架下进行训练。在27种缺失数据条件下对三个数据集的实验表明，PAST的插值准确性优于七种最先进的基线方法，RMSE提高了高达26.2%，MAE提高了高达31.6%。


### 论文摘要

Traffic time series imputation is crucial for the safety and reliability of intelligent transportation systems, while diverse types of missing data, including random, fiber, and block missing make the imputation task challenging. Existing models often focus on disentangling and separately modeling spatial and temporal patterns based on relationships between data points. However, these approaches struggle to adapt to the random missing positions, and fail to learn long-term and large-scale dependencies, which are essential in extensive missing conditions. In this paper, patterns are categorized into two types to handle various missing data conditions: primary patterns, which originate from internal relationships between data points, and auxiliary patterns, influenced by external factors like timestamps and node attributes. Accordingly, we propose the Primary-Auxiliary Spatio-Temporal network (PAST). It comprises a graph-integrated module (GIM) and a cross-gated module (CGM). GIM captures primary patterns via dynamic graphs with interval-aware dropout and multi-order convolutions, and CGM extracts auxiliary patterns through bidirectional gating on embedded external features. The two modules interact via shared hidden vectors and are trained under an ensemble self-supervised framework. Experiments on three datasets under 27 missing data conditions demonstrate that the imputation accuracy of PAST outperforms seven state-of-the-art baselines by up to 26.2% in RMSE and 31.6% in MAE.

---

## 17. Cognitive Maps in Language Models: A Mechanistic Analysis of Spatial Planning

**论文链接:** [http://arxiv.org/abs/2511.13371v1](http://arxiv.org/abs/2511.13371v1)

**作者:** Caroline Baumgartner, Eleanor Spens, Neil Burgess, Petru Manescu

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究探讨了大型语言模型如何解决空间导航任务，通过训练GPT-2模型在三种空间学习范式下工作，并发现了两种 fundamentally 不同的学习算法：一种发展出类似认知地图的稳健空间表征，另一种则采用路径依赖算法。

### 背景

大型语言模型在处理空间导航任务时的机制尚不清楚。空间导航是智能体理解环境并从中导航的基本能力，对人工智能和认知科学具有重要意义。

### 目的

研究大型语言模型（特别是GPT-2）如何解决空间导航任务，探究不同训练范式对模型学习策略的影响，以及揭示模型在空间表征和学习算法上的差异。

### 方法

研究人员在网格环境中训练GPT-2模型，采用三种空间学习范式：被动探索（Foraging Model）、有目标规划（SP-Hamiltonian）和混合模型（SP-Random Walk）。使用行为分析、表征分析和机制分析来研究模型的学习策略。

### 主要发现

1. Foraging模型发展出类似'认知地图'的稳健空间表征，能够将空间信息整合到自足的坐标系中。2. Foraging模型采用自适应分层推理系统，根据上下文长度切换不同推理策略。3. 目标导向模型学习路径依赖算法，在所有层中都依赖明确的方向输入。4. 混合模型虽然泛化能力更强，但仍保留路径依赖策略。5. 大型语言模型的空间智能存在于一个谱系中，从可泛化世界模型到目标导向启发式方法。

### 结论

大型语言模型在空间导航任务中的学习策略受训练范式影响，存在从泛化到优化的权衡机制。选择适当的训练方式可以引导模型发展出不同的空间表征和学习算法。

### 翻译

大型语言模型如何解决空间导航任务？我们通过在网格环境中训练GPT-2模型研究三种空间学习范式来探讨这一问题：被动探索（Foraging Model-预测随机游走中的步数）、在结构化哈密顿路径上进行有目标规划（生成最优最短路径）以及使用探索性数据微调的混合模型（SP-Random Walk）。使用行为分析、表征分析和机制分析，我们发现了两种 fundamentally 不同的学习算法。Foraging模型发展出一种稳健的、类似'认知地图'的空间表征。因果干预表明，它学习将空间信息整合到自足的坐标系中，这体现在一个明显的相变上，即到网络中间层时，它对历史方向令牌的依赖性消失。该模型还采用了一种自适应的分层推理系统，在短上下文中使用低级启发式方法，在长上下文中使用基于地图的推理。相比之下，目标导向模型学习了一种路径依赖算法，在所有层中都依赖于明确的方向输入。混合模型尽管表现出比其父模型更好的泛化能力，但保留了相同的路径依赖策略。这些发现表明，transformer中的空间智能可能存在于一个谱系中，从由探索性数据塑造的可泛化世界模型，到为目标导向任务优化的启发式方法。我们提供了这种泛化-优化权衡的机制解释，并强调了训练方式的选择如何影响出现的策略。


### 论文摘要

How do large language models solve spatial navigation tasks? We investigate this by training GPT-2 models on three spatial learning paradigms in grid environments: passive exploration (Foraging Model- predicting steps in random walks), goal-directed planning (generating optimal shortest paths) on structured Hamiltonian paths (SP-Hamiltonian), and a hybrid model fine-tuned with exploratory data (SP-Random Walk). Using behavioural, representational and mechanistic analyses, we uncover two fundamentally different learned algorithms. The Foraging model develops a robust, map-like representation of space, akin to a 'cognitive map'. Causal interventions reveal that it learns to consolidate spatial information into a self-sufficient coordinate system, evidenced by a sharp phase transition where its reliance on historical direction tokens vanishes by the middle layers of the network. The model also adopts an adaptive, hierarchical reasoning system, switching between a low-level heuristic for short contexts and map-based inference for longer ones. In contrast, the goal-directed models learn a path-dependent algorithm, remaining reliant on explicit directional inputs throughout all layers. The hybrid model, despite demonstrating improved generalisation over its parent, retains the same path-dependent strategy. These findings suggest that the nature of spatial intelligence in transformers may lie on a spectrum, ranging from generalisable world models shaped by exploratory data to heuristics optimised for goal-directed tasks. We provide a mechanistic account of this generalisation-optimisation trade-off and highlight how the choice of training regime influences the strategies that emerge.

---

## 18. Semi-Supervised Multi-Task Learning for Interpretable Quality As- sessment of Fundus Images

**论文链接:** [http://arxiv.org/abs/2511.13353v1](http://arxiv.org/abs/2511.13353v1)

**作者:** Lucas Gabriel Telesco, Danila Nejamkin, Estefanía Mata, Francisco Filizzola, Kevin Wignall, Lucía Franco Troilo, María de los Angeles Cenoz, Melissa Thompson, Mercedes Leguía, Ignacio Larrabide, José Ignacio Orlando

**发布时间:** 2025-11-17

**DOI:** 10.1016/j.bspc.2025.109167

### GPT解析

### 总结

本研究提出了一种混合半监督学习方法，通过结合整体质量的手动标注和质量细节的伪标签，在多任务框架内改进视网膜图像质量评估，无需大量手动标注即可获得更可解释的模型并提高性能。

### 背景

视网膜图像质量评估(RIQA)支持眼科疾病的计算机辅助诊断，但大多数工具仅对整体图像质量进行分类，而不指出获取缺陷以指导重新捕获，这一差距主要是由于详细标注的高成本造成的。

### 目的

引入一种混合半监督学习方法，结合整体质量的手动标注和质量细节的伪标签，在多任务框架内实现，目标是在不需要大量手动标注的情况下获得更可解释的RIQA模型。

### 方法

使用教师模型在小数据集上训练生成伪标签，然后在多任务设置中使用这些伪标签微调预训练模型，采用ResNet-18作为骨干网络进行质量评估。

### 主要发现

弱标注改善了质量评估，优于单任务基线；多任务模型在大多数细节预测任务上的表现与教师模型统计上相当；在新标注的EyeQ子集上，模型表现与专家相似，伪标签噪声与专家变异性一致。

### 结论

提出的半监督方法不仅改善了整体质量评估，还提供了对捕获条件(照明、清晰度、对比度)的可解释反馈，在不需要额外手动标注成本的情况下提高了可解释性，并提供临床可操作的输出以指导图像重新捕获。

### 翻译

视网膜图像质量评估(RIQA)支持眼科疾病的计算机辅助诊断。然而，大多数工具仅对整体图像质量进行分类，而不指出获取缺陷以指导重新捕获。这一差距主要是由于详细标注的高成本。在本文中，我们通过引入一种混合半监督学习方法来缓解这一局限性，该方法结合了整体质量的手动标注和质量细节的伪标签，并在多任务框架内使用。我们的目标是在不需要大量手动标注的情况下获得更可解释的RIQA模型。伪标签由在小数据集上训练的教师模型生成，然后在多任务设置中用于微调预训练模型。使用ResNet-18骨干网络，我们表明这些弱标注改善了质量评估，优于单任务基线(在EyeQ上F1: 0.875对比0.863，在DeepDRiD上0.778对比0.763)，匹配或超越现有方法。多任务模型在大多数细节预测任务上的表现与教师模型统计上相当(p > 0.05)。在我们论文发布的新标注EyeQ子集中，我们的模型表现与专家相似，表明伪标签噪声与专家变异性一致。我们的主要发现是，提出的半监督方法不仅改善了整体质量评估，还提供了对捕获条件(照明、清晰度、对比度)的可解释反馈。这在不增加额外手动标注成本的情况下提高了可解释性，并提供临床可操作的输出来指导图像重新捕获。


### 论文摘要

Retinal image quality assessment (RIQA) supports computer-aided diagnosis of eye diseases. However, most tools classify only overall image quality, without indicating acquisition defects to guide recapture. This gap is mainly due to the high cost of detailed annotations. In this paper, we aim to mitigate this limitation by introducing a hybrid semi-supervised learning approach that combines manual labels for overall quality with pseudo-labels of quality details within a multi-task framework. Our objective is to obtain more interpretable RIQA models without requiring extensive manual labeling. Pseudo-labels are generated by a Teacher model trained on a small dataset and then used to fine-tune a pre-trained model in a multi-task setting. Using a ResNet-18 backbone, we show that these weak annotations improve quality assessment over single-task baselines (F1: 0.875 vs. 0.863 on EyeQ, and 0.778 vs. 0.763 on DeepDRiD), matching or surpassing existing methods. The multi-task model achieved performance statistically comparable to the Teacher for most detail prediction tasks (p > 0.05). In a newly annotated EyeQ subset released with this paper, our model performed similarly to experts, suggesting that pseudo-label noise aligns with expert variability. Our main finding is that the proposed semi-supervised approach not only improves overall quality assessment but also provides interpretable feedback on capture conditions (illumination, clarity, contrast). This enhances interpretability at no extra manual labeling cost and offers clinically actionable outputs to guide image recapture.

---

## 19. Tab-PET: Graph-Based Positional Encodings for Tabular Transformers

**论文链接:** [http://arxiv.org/abs/2511.13338v1](http://arxiv.org/abs/2511.13338v1)

**作者:** Yunze Leng, Rohan Ghosh, Mehul Motani

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究发现位置编码(PEs)可以显著提高表格数据transformer模型的泛化性能，通过降低特征的有效秩来简化任务。研究者提出了Tab-PET框架，利用基于关联和因果关系的图结构来估计和整合位置编码，在50个数据集上验证了其有效性。

### 背景

表格数据监督学习面临数据量小、缺乏结构线索、特征异构等挑战。与视觉和语言不同，表格数据缺乏固有位置结构，限制了自注意力机制的有效性。现有表格transformer模型(3T)通常不利用位置编码，因为没有先验结构信息。

### 目的

研究结构线索，特别是位置编码(PEs)是否能改善表格transformers的泛化性能，并探索如何有效整合这些结构信息。

### 方法

提出Tab-PET(表格transformer的位置编码)框架，采用基于图的估计方法来获取位置编码。探索两种图估计范式：基于关联的和基于因果性的，并将这些位置编码整合到模型嵌入中。

### 主要发现

理论和实证研究表明，位置编码能降低特征的有效秩，通过减少问题维度来简化任务，从而改善泛化。在50个分类和回归数据集上，图推导的位置编码显著提高了3T模型性能，其中基于关联的图比基于因果的图产生更稳定和显著的增益。

### 结论

研究揭示了位置编码在表格transformers中的意外作用，展示了如何有效利用这些结构信息来提高模型泛化能力，为表格数据建模提供了新思路。

### 翻译

监督学习与表格数据呈现独特挑战，包括低数据量、缺乏结构线索以及跨越分类和连续领域的异构特征。与视觉和语言任务不同，表格数据缺乏固有位置结构，阻碍了自注意力机制的有效性。虽然最近的基于transformer的模型如TabTransformer、SAINT和FT-Transformer(我们称为3T)在表格数据上显示出潜力，但它们通常不利用位置编码(PEs)等结构线索，因为没有先前的结构信息可用。在这项工作中，我们从理论和经验上发现，结构线索，特别是位置编码可以成为改善表格transformers泛化性能的有用工具。我们发现位置编码赋予降低特征有效秩(一种内在维度形式)的能力，通过降低问题维度有效地简化任务，从而改善泛化。为此，我们提出了Tab-PET(表格transformer的位置编码)，这是一个基于图的框架，用于估计和将位置编码整合到嵌入中。受从图拓扑推导位置编码方法的启发，我们探索了两种图估计范式：基于关联的和基于因果性的。我们在50个分类和回归数据集上实证证明，图推导的位置编码显著提高了3T的性能。值得注意的是，与因果驱动的图相比，基于关联的图能产生更稳定和显著的增益。我们的工作突出了位置编码在表格transformers中的意外作用，揭示了如何利用它们来改善泛化能力。


### 论文摘要

Supervised learning with tabular data presents unique challenges, including low data sizes, the absence of structural cues, and heterogeneous features spanning both categorical and continuous domains. Unlike vision and language tasks, where models can exploit inductive biases in the data, tabular data lacks inherent positional structure, hindering the effectiveness of self-attention mechanisms. While recent transformer-based models like TabTransformer, SAINT, and FT-Transformer (which we refer to as 3T) have shown promise on tabular data, they typically operate without leveraging structural cues such as positional encodings (PEs), as no prior structural information is usually available. In this work, we find both theoretically and empirically that structural cues, specifically PEs can be a useful tool to improve generalization performance for tabular transformers. We find that PEs impart the ability to reduce the effective rank (a form of intrinsic dimensionality) of the features, effectively simplifying the task by reducing the dimensionality of the problem, yielding improved generalization. To that end, we propose Tab-PET (PEs for Tabular Transformers), a graph-based framework for estimating and inculcating PEs into embeddings. Inspired by approaches that derive PEs from graph topology, we explore two paradigms for graph estimation: association-based and causality-based. We empirically demonstrate that graph-derived PEs significantly improve performance across 50 classification and regression datasets for 3T. Notably, association-based graphs consistently yield more stable and pronounced gains compared to causality-driven ones. Our work highlights an unexpected role of PEs in tabular transformers, revealing how they can be harnessed to improve generalization.

---

## 20. A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes

**论文链接:** [http://arxiv.org/abs/2511.13294v1](http://arxiv.org/abs/2511.13294v1)

**作者:** Junmin Chen, Qian Gao, Yange Lin, Miaofei Huang, Zheng Cheng, Wei Feng, Jianxing Huang, Bo Wang, Kuang Yu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究介绍了一种名为PhyNEO-Electrolyte的新型电解质力场构建策略，采用混合物理驱动和数据驱动方法，仅依赖单体和二聚体能量分解分析数据，通过分离长/短程和非键合/键合相互作用，显著提高了机器学习原子势能训练的数据效率，在保持可靠预测能力的同时实现更大化学空间覆盖。

### 背景

电解质设计在锂离子和钠离子电池发展中起重要作用。电解质由不同溶剂、添加剂和盐组成，设计空间庞大，难以通过实验探索。经典力场依赖实验数据微调，微观预测能力受质疑；新兴机器学习原子势能(MLIP)虽能准确再现从头算数据，但存在可转移性低、体相性质预测稳定性不足和训练成本扩展性差等问题。

### 目的

开发一种高度可扩展、完全自下而上的力场构建策略，克服现有机器学习原子势能在电解质设计空间探索中的局限性。

### 方法

提出PhyNEO-Electrolyte方法，采用混合物理驱动和数据驱动方法，仅使用单体和二聚体能量分解分析数据，通过分离长/短程和非键合/键合相互作用，严格恢复长程渐进行为，这对电解质系统描述至关重要。

### 主要发现

PhyNEO-Electrolyte显著提高了机器学习原子势能训练的数据效率，使用更少的数据实现更大化学空间覆盖，同时保持体相计算的可靠定量预测能力。

### 结论

PhyNEO-Electrolyte作为未来电解质优化的重要工具，为电解质设计提供了一种新的有效方法。

### 翻译

电解质设计在锂离子电池和钠离子电池的发展中起着重要作用。电池电解质由不同溶剂、添加剂和盐组成，构成了庞大的设计空间，难以通过实验进行探索。高保真分子模拟可通过采用精确的势能面准确预测电解质的体相性质，从而指导分子和配方工程。目前，过度简化的经典力场严重依赖实验数据进行微调，因此其在微观层面的预测能力受到质疑。相比之下，新兴的机器学习原子间势能(MLIP)可准确再现从头算数据，表现出优异的拟合能力。然而，它仍然存在可转移性低、体相性质预测稳定性不足和训练成本扩展性差等问题。因此，它尚不能作为探索电解质设计空间的稳健通用工具。在本工作中，我们引入了一种名为PhyNEO-Electrolyte的高度可扩展且完全自下而上的力场构建策略。它采用混合物理驱动和数据驱动的方法，仅依赖单体和二聚体能量分解分析(EDA)数据。通过仔细分离长/短程和非键合/键合相互作用，我们严格恢复了长程渐进行为，这对描述电解质系统至关重要。通过这种方法，我们显著提高了机器学习原子势能训练的数据效率，使我们能够在保持体相计算可靠定量预测能力的同时，使用少得多的数据实现更大的化学空间覆盖。因此，PhyNEO-Electrolyte成为未来电解质优化的重要工具。


### 论文摘要

Electrolyte design plays an important role in the development of lithium-ion batteries and sodium-ion batteries. Battery electrolytes feature a large design space composed of different solvents, additives, and salts, which is difficult to explore experimentally. High-fidelity molecular simulation can accurately predict the bulk properties of electrolytes by employing accurate potential energy surfaces, thus guiding the molecule and formula engineering. At present, the overly simplified classic force fields rely heavily on experimental data for fine-tuning, thus its predictive power on microscopic level is under question. In contrast, the newly emerged machine learning interatomic potential (MLIP) can accurately reproduce the ab initio data, demonstrating excellent fitting ability. However, it is still haunted by problems such as low transferrability, insufficient stability in the prediction of bulk properties, and poor training cost scaling. Therefore, it cannot yet be used as a robust and universal tool for the exploration of electrolyte design space. In this work, we introduce a highly scalable and fully bottom-up force field construction strategy called PhyNEO-Electrolyte. It adopts a hybrid physics-driven and data-driven method that relies only on monomer and dimer EDA (energy deomposition analysis) data. With a careful separation of long/short-range and non-bonding/bonding interactions, we rigorously restore the long-range asymptotic behavior, which is critical in the description of electrolyte systems. Through this approach, we significantly improve the data efficiency of MLIP training, allowing us to achieve much larger chemical space coverage using much less data while retaining reliable quantitative prediction power in bulk phase calculations. PhyNEO-electrolyte thus serves as an important tool for future electrolyte optimization.

---

## 21. GenTract: Generative Global Tractography

**论文链接:** [http://arxiv.org/abs/2511.13183v1](http://arxiv.org/abs/2511.13183v1)

**作者:** Alec Sargood, Lemuel Puglisi, Elinor Thompson, Mirco Musolesi, Daniel C. Alexander

**发布时间:** 2025-11-17

### GPT解析

### 总结

GenTract是一种新型的生成模型，用于脑白质纤维束的全局追踪，解决了传统局部方法的误差累积和假阳性率高的问题，以及全局方法计算成本高的问题，在精度和计算效率方面都有显著提升。

### 背景

Tractography是一种从扩散磁共振成像(dMRI)推断脑内白质通路轨迹的过程。局部追踪方法通过逐步跟随图像中的局部纤维方向估计来构建流线，但容易产生误差累积和较高的假阳性率，特别是在嘈杂或低分辨率数据上。全局方法试图优化一组流线以最大化与底层纤维方向估计的兼容性，但计算成本很高。

### 目的

开发一种新的方法来解决现有追踪技术的局限性，特别是局部方法的误差累积和假阳性率高的问题，以及全局方法计算成本高的问题。

### 方法

将追踪任务构建为一个生成任务，学习从dMRI到完整、解剖学上合理的流线的直接映射。比较基于扩散和流匹配的两种范式，并将GenTract的性能与最先进的基线方法进行比较。

### 主要发现

GenTract的精度比次优方法TractOracle高出2.1倍。在具有挑战性的低分辨率和嘈杂设置中，其优势更加明显，比最接近的竞争对手高出数量级。在研究级数据上产生高精度的束图，同时在不够完善、分辨率较低的数据上保持可靠性。

### 结论

GenTract代表了全局追踪的一个有前途的解决方案，能够提供高精度的结果，同时在各种数据条件下保持可靠性。

### 翻译

纤维束追踪是从扩散磁共振成像(dMRI)推断脑内白质通路轨迹的过程。局部追踪方法通过逐步跟随图像中的局部纤维方向估计来构建流线，容易产生误差累积和较高的假阳性率，特别是在嘈杂或低分辨率数据上。相比之下，全局方法试图优化一组流线以最大化与底层纤维方向估计的兼容性，但计算成本很高。为了应对这些挑战，我们引入了GenTract，这是第一个用于全局追踪的生成模型。我们将追踪任务构建为一个生成任务，学习从dMRI到完整、解剖学上合理的流线的直接映射。我们比较了基于扩散和流匹配的两种范式，并将GenTract的性能与最先进的基线方法进行比较。值得注意的是，GenTract的精度比次优方法TractOracle高出2.1倍。在具有挑战性的低分辨率和嘈杂设置中，这种优势变得更加明显，它比最接近的竞争对手高出数量级。通过在研究级数据上产生高精度的束图，同时在不够完善、分辨率较低的数据上保持可靠性，GenTract代表了全局追踪的一个有前途的解决方案。


### 论文摘要

Tractography is the process of inferring the trajectories of white-matter pathways in the brain from diffusion magnetic resonance imaging (dMRI). Local tractography methods, which construct streamlines by following local fiber orientation estimates stepwise through an image, are prone to error accumulation and high false positive rates, particularly on noisy or low-resolution data. In contrast, global methods, which attempt to optimize a collection of streamlines to maximize compatibility with underlying fiber orientation estimates, are computationally expensive. To address these challenges, we introduce GenTract, the first generative model for global tractography. We frame tractography as a generative task, learning a direct mapping from dMRI to complete, anatomically plausible streamlines. We compare both diffusion-based and flow matching paradigms and evaluate GenTract's performance against state-of-the-art baselines. Notably, GenTract achieves precision 2.1x higher than the next-best method, TractOracle. This advantage becomes even more pronounced in challenging low-resolution and noisy settings, where it outperforms the closest competitor by an order of magnitude. By producing tractograms with high precision on research-grade data while also maintaining reliability on imperfect, lower-resolution data, GenTract represents a promising solution for global tractography.

---

## 22. Skeletons Speak Louder than Text: A Motion-Aware Pretraining Paradigm for Video-Based Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.13150v1](http://arxiv.org/abs/2511.13150v1)

**作者:** Rifen Lin, Alex Jinpeng Wang, Jiawei Mo, Min Li

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了CSIP-ReID，第一个用于视频人员重识别的骨架驱动预训练框架，解决了现有基于文本方法的两个局限性：缺乏真正的多模态预训练和文本无法捕捉细粒度时间运动特征。

### 背景

多模态预训练已革新视觉理解，但对视频人员重识别(ReID)的影响尚未充分探索。现有方法依赖视频-文本对，但存在两个基本局限：缺乏真正的多模态预训练，文本难以捕捉区分身份所需的关键时间运动特征。

### 目的

引入第一个用于ReID的骨架驱动预训练框架，解决现有方法的局限性，提高视频人员重识别的性能。

### 方法

提出对比骨架-图像预训练方法(CSIP-ReID)，采用两阶段方法：第一阶段使用对比学习对齐骨架和视觉特征；第二阶段引入动态原型融合更新器(PFU)改进多模态身份原型；并提出骨架引导的时间建模(SGTM)模块，整合时间线索到视觉特征中。

### 主要发现

CSIP-ReID在标准视频ReID基准测试(MARS, LS-VID, iLIDS-VID)上取得最先进结果；在仅使用骨架的ReID任务(BIWI, IAS)上表现出强大泛化能力，显著优于以前方法。

### 结论

CSIP-ReID开创了ReID的无标注和运动感知预训练范式，开辟了多模态表示学习的新前沿。

### 翻译

多模态预训练已经革新了视觉理解，但其对基于视频的人员重识别(ReID)的影响仍然探索不足。现有方法通常依赖于视频-文本对，却存在两个基本局限：(1)缺乏真正的多模态预训练，(2)文本难以捕捉区分身份所需的关键细粒度时间运动特征。在这项工作中，我们通过引入第一个用于ReID的骨架驱动预训练框架，大胆地脱离了基于文本的范式。为此，我们提出了对比骨架-图像预训练方法(CSIP-ReID)，这是一种新颖的两阶段方法，利用骨架序列作为与视频帧对齐的时空信息模态。在第一阶段，我们采用对比学习对齐骨架和视觉特征。在第二阶段，我们引入动态原型融合更新器(PFU)来改进多模态身份原型，融合运动和外观线索。此外，我们提出了骨架引导的时间建模(SGTM)模块，从骨架数据中提取时间线索并将其整合到视觉特征中。大量实验证明，CSIP-ReID在标准视频ReID基准测试(MARS, LS-VID, iLIDS-VID)上取得了最新的最先进结果。此外，它在仅使用骨架的ReID任务(BIWI, IAS)上表现出强大的泛化能力，显著优于以前的方法。CSIP-ReID为ReID开创了无标注和运动感知的预训练范式，开辟了多模态表示学习的新前沿。


### 论文摘要

Multimodal pretraining has revolutionized visual understanding, but its impact on video-based person re-identification (ReID) remains underexplored. Existing approaches often rely on video-text pairs, yet suffer from two fundamental limitations: (1) lack of genuine multimodal pretraining, and (2) text poorly captures fine-grained temporal motion-an essential cue for distinguishing identities in video. In this work, we take a bold departure from text-based paradigms by introducing the first skeleton-driven pretraining framework for ReID. To achieve this, we propose Contrastive Skeleton-Image Pretraining for ReID (CSIP-ReID), a novel two-stage method that leverages skeleton sequences as a spatiotemporally informative modality aligned with video frames. In the first stage, we employ contrastive learning to align skeleton and visual features at sequence level. In the second stage, we introduce a dynamic Prototype Fusion Updater (PFU) to refine multimodal identity prototypes, fusing motion and appearance cues. Moreover, we propose a Skeleton Guided Temporal Modeling (SGTM) module that distills temporal cues from skeleton data and integrates them into visual features. Extensive experiments demonstrate that CSIP-ReID achieves new state-of-the-art results on standard video ReID benchmarks (MARS, LS-VID, iLIDS-VID). Moreover, it exhibits strong generalization to skeleton-only ReID tasks (BIWI, IAS), significantly outperforming previous methods. CSIP-ReID pioneers an annotation-free and motion-aware pretraining paradigm for ReID, opening a new frontier in multimodal representation learning.

---

## 23. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language

**论文链接:** [http://arxiv.org/abs/2511.13127v1](http://arxiv.org/abs/2511.13127v1)

**作者:** Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文提出了一种名为VEIL的新型越狱攻击框架，通过设计包含中性场景锚点、潜在听觉触发器和风格调制器的模块化提示，能够诱导T2V模型生成语义不安全的视频，同时保持原始意图并规避安全检测。

### 背景

现有的T2V模型越狱攻击通常在明显不安全的提示中添加对抗性扰动，这些方法容易被检测和防御。研究需要一种更隐蔽、更有效的攻击方法。

### 目的

开发一种能够诱导T2V模型生成违反政策但保留原始(被阻止)意图的语义不安全视频的越狱攻击方法，同时保持提示的表面无害性以规避检测。

### 方法

提出VEIL框架，利用T2V模型的跨模态关联模式，通过模块化提示设计实现。提示包含三个组件：中性场景锚点(提供表面场景描述)、潜在听觉触发器(利用音频-视觉共现先验)和风格调制器(放大和稳定触发效果)。将攻击生成形式化为上述模块化提示空间上的约束优化问题，并通过引导搜索过程解决。

### 主要发现

在7个T2V模型上的广泛实验表明，所提出的攻击方法有效，在商业模型中平均攻击成功率提高了23%。

### 结论

VEIL框架通过精心设计的模块化提示，能够有效诱导T2V模型生成语义不安全的视频，同时保持提示的表面无害性，为研究T2V模型的安全盲点和防御措施提供了新视角。

### 翻译

越狱攻击可以规避模型安全护栏并揭示关键盲点。先前针对文本到视频(T2V)模型的攻击通常在明显不安全的提示中添加对抗性扰动，这些方法往往容易被检测和防御。相比之下，我们表明包含丰富隐含提示的良性外观提示可以诱导T2V模型生成语义不安全的视频，这些视频既违反政策又保留原始(被阻止)意图。为此，我们提出了VEIL，一个利用T2V模型跨模态关联模式的越狱框架，通过模块化提示设计实现。具体而言，我们的提示结合三个组件：中性场景锚点，提供从被阻止意图中提取的表面场景描述以保持合理性；潜在听觉触发器，无害声音事件的文本描述(如吱吱声、闷响)，利用学习的音频-视觉共现先验使模型偏向特定不安全视觉概念；以及风格调制器，电影指令(如镜头构图、氛围)，放大和稳定潜在触发器效果。我们将攻击生成为上述模块化提示空间上的约束优化问题，并通过平衡隐秘性和有效性的引导搜索过程解决。在7个T2V模型上的广泛实验证明了我们攻击方法的有效性，在商业模型中平均攻击成功率提高了23%。


### 论文摘要

Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models.

---

## 24. Region-Point Joint Representation for Effective Trajectory Similarity Learning

**论文链接:** [http://arxiv.org/abs/2511.13125v1](http://arxiv.org/abs/2511.13125v1)

**作者:** Hao Long, Silin Zhou, Lisi Chen, Shuo Shang

**发布时间:** 2025-11-17

**备注:** This paper is accepted by AAAI2026

### GPT解析

### 总结

RePo是一种新颖的方法，通过联合编码区域级和点级特征来捕获轨迹的空间上下文和细粒度移动模式，显著提高了轨迹相似度计算的准确性。

### 背景

基于学习的方法已经降低了传统轨迹相似度计算的计算复杂度，但最先进的方法仍然无法利用轨迹信息的全面频谱进行相似度建模。

### 目的

解决现有方法无法充分利用轨迹信息的问题，提出一种能够同时捕获空间上下文和细粒度移动模式的轨迹相似度计算方法。

### 方法

RePo方法包括：1) 将GPS轨迹映射到网格序列，通过结构特征和视觉特征增强的语义上下文捕获空间上下文；2) 使用三个轻量级专家网络从密集GPS序列中提取局部、相关性和连续移动模式；3) 通过路由网络自适应融合点级特征，并使用交叉注意力与区域级特征结合产生最终轨迹嵌入；4) 采用带有困难负样本的对比损失进行训练。

### 主要发现

实验结果表明，RePo在所有评估指标上比最先进的基线方法平均提高了22.2%的准确率。

### 结论

RePo通过联合编码区域级和点级特征，能够更全面地捕获轨迹信息，有效提高了轨迹相似度计算的准确性。

### 翻译

最近基于学习的方法降低了传统轨迹相似度计算的计算复杂度，但最先进的方法仍然无法利用轨迹信息的全面频谱进行相似度建模。为了解决这个问题，我们提出了RePo，一种新颖的方法，联合编码区域级和点级特征，以捕获空间上下文和细粒度的移动模式。对于区域级表示，GPS轨迹首先被映射到网格序列，空间上下文通过结构特征和由视觉特征增强的语义上下文来捕获。对于点级表示，三个轻量级专家网络从密集的GPS序列中提取局部、相关性和连续移动模式。然后，路由网络自适应地融合学习的点级特征，随后使用交叉注意力将这些特征与区域级特征结合，以产生最终的轨迹嵌入。为了训练RePo，我们采用带有困难负样本的对比损失来提供相似度排序监督。实验结果表明，RePo在所有评估指标上比最先进的基线方法平均提高了22.2%的准确率。


### 论文摘要

Recent learning-based methods have reduced the computational complexity of traditional trajectory similarity computation, but state-of-the-art (SOTA) methods still fail to leverage the comprehensive spectrum of trajectory information for similarity modeling. To tackle this problem, we propose \textbf{RePo}, a novel method that jointly encodes \textbf{Re}gion-wise and \textbf{Po}int-wise features to capture both spatial context and fine-grained moving patterns. For region-wise representation, the GPS trajectories are first mapped to grid sequences, and spatial context are captured by structural features and semantic context enriched by visual features. For point-wise representation, three lightweight expert networks extract local, correlation, and continuous movement patterns from dense GPS sequences. Then, a router network adaptively fuses the learned point-wise features, which are subsequently combined with region-wise features using cross-attention to produce the final trajectory embedding. To train RePo, we adopt a contrastive loss with hard negative samples to provide similarity ranking supervision. Experiment results show that RePo achieves an average accuracy improvement of 22.2\% over SOTA baselines across all evaluation metrics.

---

## 25. Initial Excitation-based Adaptive Observers for Discrete-Time LTI Systems

**论文链接:** [http://arxiv.org/abs/2511.13117v1](http://arxiv.org/abs/2511.13117v1)

**作者:** Anchita Dey, Soutrik Bandyopadhyay, Shubhendu Bhasin

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种针对离散时间线性时不变系统的基于初始激励的自适应观测器，解决了状态和参数同时估计的问题，无需无限时间激励，更适用于稳定任务。

### 背景

控制算法在实际应用中的效果依赖于对系统参数和状态的准确了解，但在实践中获取这些信息往往具有挑战性。虽然连续时间系统的自适应观测器设计已有较多研究，但离散时间系统的对应设计相对较少。

### 目的

开发一种更实用的离散时间系统自适应观测器，避免传统方法对持续激励条件的依赖，从而降低对控制努力的需求。

### 方法

采用双层滤波结构和基于归一化梯度下降的更新律来学习未知参数，同时提出修改回归子以增强信息提取，加快收敛速度。

### 主要发现

所提出的方法在初始激励条件下能保证状态和参数估计的有界性和指数收敛性，且不需要像传统方法那样需要无限时间的激励。

### 结论

仿真结果验证了所提出的基于初始激励的自适应观测器设计的有效性和实用性。

### 翻译

在实际应用中，控制算法的有效性严重依赖于对底层系统参数和状态的准确了解。然而，在实践中获取这些量通常具有挑战性。自适应观测器通过仅使用输入-输出测量来执行状态和参数的同步估计来解决这一问题。虽然针对连续时间系统存在许多自适应观测器设计，但它们的离散时间对应物仍然相对未被探索。本文提出了一种针对离散时间线性时不变系统的基于初始激励(IE)的自适应观测器。与依赖于持续激励条件（需要持续激励和无限控制努力）的传统设计相比，所提出的方法不需要无限时间的激励，因此使其在稳定任务中更实用。我们采用双层滤波结构和基于归一化梯度下降的更新律来学习未知参数。我们还提出修改回归子以增强信息提取，从而加快收敛速度。严格的理论分析保证了在初始激励条件下状态和参数估计的有界性和指数收敛性，仿真结果验证了所提出设计的有效性。


### 论文摘要

In practical applications, the efficacy of a control algorithm relies critically on the accurate knowledge of the parameters and states of the underlying system. However, obtaining these quantities in practice is often challenging. Adaptive observers address this issue by performing simultaneous state and parameter estimation using only input-output measurements. While many adaptive observer designs exist for continuous-time systems, their discrete-time counterparts remain relatively unexplored. This paper proposes an initial excitation (IE)-based adaptive observer for discrete-time linear time-invariant systems. In contrast to conventional designs that rely on the persistence of excitation condition, which requires continuous excitation and infinite control effort, the proposed method does not require excitation for infinite time, thus making it more practical for stabilization tasks. We employ a two-layer filtering structure and a normalized gradient descent-based update law for learning the unknown parameters. We also propose modifying the regressors to enhance information extraction, leading to faster convergence. Rigorous theoretical analysis guarantees bounded and exponentially converging estimates of both states and parameters under the IE condition, and simulation results validate the efficacy of the proposed design.

---

## 26. Low-Level Dataset Distillation for Medical Image Enhancement

**论文链接:** [http://arxiv.org/abs/2511.13106v1](http://arxiv.org/abs/2511.13106v1)

**作者:** Fengzhi Xu, Ziyuan Yang, Mengyu Sun, Joey Tianyi Zhou, Yi Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种针对医学图像增强的低层数据集蒸馏方法，解决了现有方法需要大规模数据集导致的高成本问题，同时保护患者隐私。

### 背景

医学图像增强具有临床价值，但现有方法需要大规模数据集学习复杂的像素级映射，导致训练和存储成本高昂。数据集蒸馏可减轻这些负担，但现有方法主要针对高层任务(多对一映射)，而低层任务(多对多映射)需要像素级保真度，使低层DD成为欠定问题。

### 目的

提出第一个针对医学图像增强的低层数据集蒸馏方法，解决低层DD的欠定问题，同时保护患者隐私。

### 方法

1) 利用患者间解剖相似性构建基于代表性患者的共享解剖先验，作为不同患者蒸馏数据的初始化；2) 使用结构保持个性化生成(SPG)模块将先验个性化，保留像素级保真度；3) 为不同低层任务构建特定于高质量和低质量训练对的蒸馏数据；4) 通过梯度对齐将患者特定知识注入蒸馏数据；5) 共享包含抽象训练信息的蒸馏数据集，排除患者特定细节保护隐私。

### 主要发现

共享解剖先验和个性化生成可有效解决低层DD的欠定问题，使小型蒸馏数据集能够约束密集的像素级映射。

### 结论

所提出的低层DD方法能够在不访问原始患者数据的情况下，为医学图像增强提供有效的训练信息，同时保护患者隐私。

### 翻译

医学图像增强具有临床价值，但现有方法需要大规模数据集来学习复杂的像素级映射。然而，与这些数据集相关的巨大训练和存储成本阻碍了它们的实际部署。虽然数据集蒸馏可以减轻这些负担，但现有方法主要针对高层任务，其中多个样本共享相同的标签。这种多对一映射使蒸馏数据能够捕获共享语义并实现信息压缩。相比之下，低层任务涉及多对多映射，需要像素级保真度，这使得低层DD成为一个欠定问题，因为小型蒸馏数据集无法完全约束密集的像素级映射。为解决这一问题，我们提出了首个针对医学图像增强的低层DD方法。我们首先利用患者间的解剖相似性基于代表性患者构建共享解剖先验，作为不同患者蒸馏数据的初始化。然后，使用结构保持个性化生成(SPG)模块将此先验个性化，同时将患者特定的解剖信息整合到蒸馏数据集中，保留像素级保真度。对于不同的低层任务，使用蒸馏数据构建特定于任务的高质量和低质量训练对。通过将对齐从蒸馏对上训练的网络计算出的梯度与相应患者原始数据上的梯度，将患者特定的知识注入蒸馏数据。值得注意的是，下游用户无法访问原始患者数据。相反，只共享包含抽象训练信息的蒸馏数据集，排除了患者特定细节，从而保护隐私。


### 论文摘要

Medical image enhancement is clinically valuable, but existing methods require large-scale datasets to learn complex pixel-level mappings. However, the substantial training and storage costs associated with these datasets hinder their practical deployment. While dataset distillation (DD) can alleviate these burdens, existing methods mainly target high-level tasks, where multiple samples share the same label. This many-to-one mapping allows distilled data to capture shared semantics and achieve information compression. In contrast, low-level tasks involve a many-to-many mapping that requires pixel-level fidelity, making low-level DD an underdetermined problem, as a small distilled dataset cannot fully constrain the dense pixel-level mappings. To address this, we propose the first low-level DD method for medical image enhancement. We first leverage anatomical similarities across patients to construct the shared anatomical prior based on a representative patient, which serves as the initialization for the distilled data of different patients. This prior is then personalized for each patient using a Structure-Preserving Personalized Generation (SPG) module, which integrates patient-specific anatomical information into the distilled dataset while preserving pixel-level fidelity. For different low-level tasks, the distilled data is used to construct task-specific high- and low-quality training pairs. Patient-specific knowledge is injected into the distilled data by aligning the gradients computed from networks trained on the distilled pairs with those from the corresponding patient's raw data. Notably, downstream users cannot access raw patient data. Instead, only a distilled dataset containing abstract training information is shared, which excludes patient-specific details and thus preserves privacy.

---

## 27. Rethinking Saliency Maps: A Cognitive Human Aligned Taxonomy and Evaluation Framework for Explanations

**论文链接:** [http://arxiv.org/abs/2511.13081v1](http://arxiv.org/abs/2511.13081v1)

**作者:** Yehonatan Elisha, Seffi Cohen, Oren Barkan, Noam Koenigstein

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种参考框架×粒度(RFxG)分类法，用于系统性评估深度学习中的显著性图解释方法，解决了现有评估指标过度强调点式忠实度而忽略对比推理和语义粒度的问题。

### 背景

显著性图被广泛用于深度学习的视觉解释，但关于其预期目的和与不同用户查询的一致性缺乏共识，这种模糊性阻碍了解释方法的有效评估和实用性。

### 目的

引入一个有原则的概念框架来组织显著性解释，并开发新的评估指标来系统评估解释质量，使视觉解释不仅忠实于底层模型行为，而且与人类理解和探究保持一致。

### 方法

提出参考框架×粒度(RFxG)分类法，沿着两个轴组织解释：参考框架(点式vs对比式)和粒度(细粒度类级别vs粗粒度组级别)。基于此框架提出四种新的忠实度指标，并应用于十个最先进的显著性方法、四种模型架构和三个数据集进行评估。

### 主要发现

现有评估指标存在严重局限性，过度强调点式忠实度，而忽略了对比推理和语义粒度。通过RFxG透镜可以更全面地评估解释质量。

### 结论

应向用户意图驱动的评估转变，RFxG分类法为开发既忠实于模型行为又符合人类理解复杂性的视觉解释提供了概念基础和实用工具。

### 翻译

显著性图被广泛用于深度学习的视觉解释，但关于其预期目的和与不同用户查询的一致性仍存在根本性分歧。这种模糊性阻碍了解释方法的有效评估和实用性。我们通过引入参考框架×粒度(RFxG)分类法解决了这一差距，这是一个有原则的概念框架，沿着两个基本轴组织显著性解释：参考框架：区分点式(为什么是这个预测？)和对比式(为什么是这个而不是替代方案？)解释。粒度：从细粒度的类级别(例如，为什么是哈士奇？)到粗粒度的组级别(例如，为什么是狗？)解释。使用RFxG透镜，我们展示了现有评估指标的严重局限性，这些指标过度强调点式忠实度，而忽略了对比推理和语义粒度。为了系统评估两个RFxG维度上的解释质量，我们提出了四种新的忠实度指标。我们的全面评估框架将这些指标应用于十个最先进的显著性方法、四种模型架构和三个数据集。通过倡导向用户意图驱动的评估转变，我们的工作既提供了概念基础，也提供了必要的实用工具，以开发不仅忠实于底层模型行为，而且有意义地与人类理解和探究复杂性保持一致的视觉解释。


### 论文摘要

Saliency maps are widely used for visual explanations in deep learning, but a fundamental lack of consensus persists regarding their intended purpose and alignment with diverse user queries. This ambiguity hinders the effective evaluation and practical utility of explanation methods.We address this gap by introducing the Reference-Frame $\times$ Granularity (RFxG) taxonomy, a principled conceptual framework that organizes saliency explanations along two essential axes:Reference-Frame: Distinguishing between pointwise ("Why this prediction?") and contrastive ("Why this and not an alternative?") explanations.Granularity: Ranging from fine-grained class-level (e.g., "Why Husky?") to coarse-grained group-level (e.g., "Why Dog?") interpretations.Using the RFxG lens, we demonstrate critical limitations in existing evaluation metrics, which overwhelmingly prioritize pointwise faithfulness while neglecting contrastive reasoning and semantic granularity. To systematically assess explanation quality across both RFxG dimensions, we propose four novel faithfulness metrics. Our comprehensive evaluation framework applies these metrics to ten state-of-the-art saliency methods, four model architectures, and three datasets.By advocating a shift toward user-intent-driven evaluation, our work provides both the conceptual foundation and the practical tools necessary to develop visual explanations that are not only faithful to the underlying model behavior but are also meaningfully aligned with the complexity of human understanding and inquiry.

---

## 28. ViSS-R1: Self-Supervised Reinforcement Video Reasoning

**论文链接:** [http://arxiv.org/abs/2511.13054v1](http://arxiv.org/abs/2511.13054v1)

**作者:** Bo Fang, Yuxin Song, Qiangqiang Wu, Haoyuan Sun, Wenhao Wu, Antoni B. Chan

**发布时间:** 2025-11-17

**备注:** Our paper was initially titled "Video-SSR1: Self-Supervised Reinforcement Video Reasoning." Upon noticing its close resemblance to the title of a recently released paper, we have decided to rename our work as "ViSS-R1."

### GPT解析

### 总结

本研究针对多模态大语言模型在复杂视频推理中的挑战，提出了一种以视觉为中心的新方法，通过引入自监督强化学习算法和整合框架，有效利用视觉信息并减少捷径学习和幻觉问题。

### 背景

复杂视频推理对多模态大语言模型(MLLMs)仍然是一个重大挑战，当前基于R1的方法论往往优先考虑以文本为中心的推理，在视频任务中未充分利用丰富的视觉信息，导致潜在的捷径学习和增加幻觉易感性。

### 目的

培养更强大、以视觉为中心的视频理解能力，使模型能够更有效地处理视频中的视觉信息。

### 方法

1) 引入新颖的自监督强化学习GRPO算法(Pretext-GRPO)到标准R1流程中，在变换后的视觉输入上正确解决预训练任务会获得积极奖励；2) 提出ViSS-R1框架，将基于预训练任务的自监督学习直接整合到MLLM的R1后训练范式中；3) 框架迫使模型通过同时处理预训练问题和真实用户查询来推理变换后的视觉输入，需要识别应用的变换并重建原始视频以制定准确答案。

### 主要发现

在六个广泛使用的视频推理和理解基准上进行的全面评估证明了Pretext-GRPO和ViSS-R1在复杂视频推理方面的有效性和优越性。

### 结论

所提出的方法显著提升了MLLM在视频推理任务中的表现，代码和模型将公开可用，为视频理解领域提供了新的研究方向。

### 翻译

本研究通过引入Pretext-GRPO算法和ViSS-R1框架，解决了多模态大语言模型在视频推理中过度依赖文本信息而忽视视觉细节的问题，使模型能够更深入地理解和分析视频内容。


### 论文摘要

Complex video reasoning remains a significant challenge for Multimodal Large Language Models (MLLMs), as current R1-based methodologies often prioritize text-centric reasoning derived from text-based and image-based developments. In video tasks, such strategies frequently underutilize rich visual information, leading to potential shortcut learning and increased susceptibility to hallucination. To foster a more robust, visual-centric video understanding, we start by introducing a novel self-supervised reinforcement learning GRPO algorithm (Pretext-GRPO) within the standard R1 pipeline, in which positive rewards are assigned for correctly solving pretext tasks on transformed visual inputs, which makes the model to non-trivially process the visual information. Building on the effectiveness of Pretext-GRPO, we further propose the ViSS-R1 framework, which streamlines and integrates pretext-task-based self-supervised learning directly into the MLLM's R1 post-training paradigm. Instead of relying solely on sparse visual cues, our framework compels models to reason about transformed visual input by simultaneously processing both pretext questions (concerning transformations) and true user queries. This necessitates identifying the applied transformation and reconstructing the original video to formulate accurate final answers. Comprehensive evaluations on six widely-used video reasoning and understanding benchmarks demonstrate the effectiveness and superiority of our Pretext-GRPO and ViSS-R1 for complex video reasoning. Our codes and models will be publicly available.

---

## 29. uCLIP: Parameter-Efficient Multilingual Extension of Vision-Language Models with Unpaired Data

**论文链接:** [http://arxiv.org/abs/2511.13036v1](http://arxiv.org/abs/2511.13036v1)

**作者:** Dahyun Chung, Donghyun Shin, Yujin Sung, Seunggi Moon, Jinwoo Jeon, Byung-Jun Lee

**发布时间:** 2025-11-17

**备注:** Our project page can be found at https://dinyudin203.github.io/uCLIP-project/

### GPT解析

### 总结

本文提出了一种轻量级且数据高效的多模态视觉-语言对齐框架，解决了CLIP模型在低资源语言上的扩展问题。

### 背景

CLIP模型通过大规模英文-图像对在多种视觉任务上表现出强大泛化能力，但由于高质量多模态图像-文本数据稀缺，其扩展到低资源语言受到限制。现有多模态视觉-语言模型在捷克语、芬兰语、克罗地亚语、匈牙利语和罗马尼亚语等代表性不足的语言上检索性能持续低下。

### 目的

开发一个轻量级且数据高效的多模态视觉-语言对齐框架，解决低资源语言的对齐问题。

### 方法

提出一种无需图像-文本对或文本-文本对的方法，训练期间冻结预训练的图像编码器和多语言文本编码器，仅训练一个170万参数的紧凑投影模块，使用英文表示作为语义锚点进行对比损失训练，实现在监督有限的语言中也能实现强大的多语言对齐。

### 主要发现

在多个多模态检索基准上的广泛评估证实了该方法的有效性，在五个代表性不足的语言上显示出显著提升，这些语言现有模型通常表现不佳。

### 结论

基于枢纽的参数高效对齐策略对于包容性多模态学习是有效的。

### 翻译

对比语言-图像预训练（CLIP）通过利用大规模英文-图像对，已在广泛的视觉任务中展现出强大的泛化能力。然而，由于高质量多语言图像-文本数据的稀缺，其扩展到低资源语言仍然有限。现有的多模态视觉-语言模型在Crossmodal-3600（XM3600）基准测试中，包括捷克语、芬兰语、克罗地亚语、匈牙利语和罗马尼亚语在内的代表性不足语言上，持续表现出较低的检索性能。为解决这一问题，我们提出了一种轻量级且数据高效的多模态视觉-语言对齐框架。我们的方法不需要图像-文本对或文本-文本对，并在训练期间冻结预训练的图像编码器和多语言文本编码器。仅训练一个紧凑的170万参数的投影模块，使用英文表示作为语义锚点进行对比损失。这种最小化的训练设置即使在监督有限的语言中也能实现强大的多语言对齐。在多个多模态检索基准上的广泛评估证实了我们方法的有效性，在五个代表性不足的语言上显示出显著提升，这些语言现有模型通常表现不佳。这些发现强调了我们的基于枢纽的参数高效对齐策略在包容性多模态学习中的有效性。


### 论文摘要

Contrastive Language-Image Pre-training (CLIP) has demonstrated strong generalization across a wide range of visual tasks by leveraging large-scale English-image pairs. However, its extension to low-resource languages remains limited due to the scarcity of high-quality multilingual image-text data. Existing multilingual vision-language models exhibit consistently low retrieval performance in underrepresented languages including Czech, Finnish, Croatian, Hungarian, and Romanian on the Crossmodal-3600 (XM3600) benchmark. To address this, we propose a lightweight and data-efficient framework for multilingual vision-language alignment. Our approach requires no image-text pairs or text-text pairs and freezes both the pretrained image encoder and multilingual text encoder during training. Only a compact 1.7M-parameter projection module is trained, using a contrastive loss over English representations as semantic anchors. This minimal training setup enables robust multilingual alignment even for languages with limited supervision. Extensive evaluation across multiple multilingual retrieval benchmarks confirms the effectiveness of our method, showing significant gains in five underrepresented languages where existing models typically underperform. These findings highlight the effectiveness of our pivot-based, parameter-efficient alignment strategy for inclusive multimodal learning.

---

## 30. REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.13026v1](http://arxiv.org/abs/2511.13026v1)

**作者:** Jiaze Li, Hao Yin, Wenhui Tan, Jingyang Chen, Boshen Xu, Yuxun Qu, Yijing Chen, Jianzhong Ju, Zhenbo Luo, Jian Luan

**发布时间:** 2025-11-17

### GPT解析

### 总结

REVISOR是一种新的工具增强多模态反思框架，解决了纯文本反思机制在长视频理解中的局限性，通过跨模态协作构建内省式反思过程，显著提高了MLLMs的长视频理解能力。

### 背景

纯文本为基础的自我反思机制在大多数多模态任务中表现良好，但在长视频理解场景中存在明显局限性。

### 目的

开发一种能够处理长视频理解中丰富动态视觉输入的多模态反思框架，解决纯文本反思机制无法充分整合视觉信息的问题。

### 方法

提出REVISOR框架，使多模态大模型能够跨文本和视觉模态协作构建内省式反思过程；设计双重归因解耦奖励（DADR）机制，并集成到GRPO训练策略中，强制模型的推理与所选视频证据之间的因果对齐。

### 主要发现

长视频理解需要专门针对视觉信息的重新思考过程；纯文本反思机制缺乏跨模态交互能力，无法在反思过程中充分整合视觉信息；REVISOR框架显著增强了MLLMs的长视频理解能力。

### 结论

REVISOR框架能够在不进行额外监督微调或使用外部模型的情况下，显著提高MLLMs的长视频理解能力，在多个基准测试中取得了令人印象深刻的结果。

### 翻译

摘要：依赖于纯文本重新思考过程的自我反思机制在大多数多模态任务中表现良好。然而，当直接应用于长视频理解场景时，它们表现出明显的局限性。这背后的根本原因有两点：（1）长视频理解涉及更丰富和动态的视觉输入，这意味着仅重新思考文本信息是不够的，需要进一步专门针对视觉信息的重新思考过程；（2）纯文本反思机制缺乏跨模态交互能力，阻止它们在反思过程中充分整合视觉信息。受这些见解的启发，我们提出了REVISOR（REflective VIsual Segment Oriented Reasoning），这是一种新颖的工具增强多模态反思框架。REVISOR使多模态大模型能够跨文本和视觉模态协作构建内省式反思过程，显著提高了它们对长视频理解的推理能力。为确保REVISOR能够在强化学习过程中学习准确审查与问题高度相关的视频片段，我们设计了双重归因解耦奖励（DADR）机制。集成到GRPO训练策略中，该机制强制模型的推理与所选视频证据之间的因果对齐。值得注意的是，REVISOR框架显著增强了MLLMs的长视频理解能力，不需要额外的监督微调或外部模型，在VideoMME、LongVideoBench、MLVU和LVBench四个基准测试中取得了令人印象深刻的结果。


### 论文摘要

Self-reflection mechanisms that rely on purely text-based rethinking processes perform well in most multimodal tasks. However, when directly applied to long-form video understanding scenarios, they exhibit clear limitations. The fundamental reasons for this lie in two points: (1)long-form video understanding involves richer and more dynamic visual input, meaning rethinking only the text information is insufficient and necessitates a further rethinking process specifically targeting visual information; (2) purely text-based reflection mechanisms lack cross-modal interaction capabilities, preventing them from fully integrating visual information during reflection. Motivated by these insights, we propose REVISOR (REflective VIsual Segment Oriented Reasoning), a novel framework for tool-augmented multimodal reflection. REVISOR enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, significantly enhancing their reasoning capability for long-form video understanding. To ensure that REVISOR can learn to accurately review video segments highly relevant to the question during reinforcement learning, we designed the Dual Attribution Decoupled Reward (DADR) mechanism. Integrated into the GRPO training strategy, this mechanism enforces causal alignment between the model's reasoning and the selected video evidence. Notably, the REVISOR framework significantly enhances long-form video understanding capability of MLLMs without requiring supplementary supervised fine-tuning or external models, achieving impressive results on four benchmarks including VideoMME, LongVideoBench, MLVU, and LVBench.

---

## 31. Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues

**论文链接:** [http://arxiv.org/abs/2511.13015v1](http://arxiv.org/abs/2511.13015v1)

**作者:** King-Man Tam, Satoshi Ikehata, Yuta Asano, Zhaoyi An, Rei Kawakami

**发布时间:** 2025-11-17

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

GeoUniPS是一种通用光度立体网络，通过整合合成监督和大规模3D重建模型的几何先验，解决了复杂自然场景中多光照线索不可靠的问题，实现了表面法线恢复的最先进性能。

### 背景

通用光度立体方法在不严格要求光照假设的情况下能够恢复表面法线，但当多光照线索不可靠时(如光照偏差、阴影或复杂自然场景中的自遮挡区域)表现不佳。

### 目的

提出GeoUniPS网络，结合合成监督和大规模3D重建模型预训练的高层次几何先验，以提升复杂自然场景中的表面法线恢复能力。

### 方法

利用3D重建模型作为视觉-几何基础模型；设计光照-几何双分支编码器从冻结的3D重建模型中提取多光照线索和几何先验；引入具有真实透视投影的PS-Perp数据集解决传统正交投影假设的局限性。

### 主要发现

GeoUniPS在多个数据集上实现了最先进的性能，无论是在定量还是定性方面，特别是在复杂的自然场景中表现优异。

### 结论

GeoUniPS通过整合几何先验和改进的数据表示，显著提升了复杂场景中的表面法线恢复能力，为通用光度立体方法提供了新的解决方案。

### 翻译

通用光度立体是一种在不严格要求光照假设的情况下恢复表面法线的有前景的方法。然而，当多光照线索不可靠时，例如在有偏差的光照下或在复杂自然场景的阴影或自遮挡区域中，它会遇到困难。我们提出了GeoUniPS，一种通用光度立体网络，它将合成监督与大规模3D重建模型在大量自然数据上预训练的高层次几何先验相结合。我们的关键见解是，这些3D重建模型充当视觉-几何基础模型，自然编码了真实场景的丰富几何知识。为了利用这一点，我们设计了一个光照-几何双分支编码器，从冻结的3D重建模型中提取多光照线索和几何先验。我们还通过引入具有真实透视投影的PS-Perp数据集来解决传统正交投影假设的局限性，使模型能够学习空间变化的视角方向。大量实验证明，GeoUniPS在多个数据集上提供了最先进的性能，无论是在定量还是定性方面，特别是在复杂的自然场景中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决光度立体技术在多光照条件不可靠（如光照不充分、有阴影或自遮挡区域）时表面法线估计性能下降的问题。这个问题在现实中很重要，因为真实场景中难以控制光照条件，某些区域光照不足，尤其在几何复杂、材质特殊和纹理丰富的区域，有限的光照线索使法线估计更加困难。传统光度立体技术主要依赖多光照变化作为主要线索，当这些变化不可靠时缺乏补偿机制。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到大规模3D重建模型（如VGGT）预训练在野外数据上后，可作为视觉几何基础模型编码真实场景的几何知识。即使只用单张输入图像，这些网络也能重建合理的3D形状，表明它们学到了高级几何先验。作者将这些先验注入到光度立体流程中，设计了双分支编码器：一个分支提取光照感知特征，另一个分支从冻结的3D重建模型捕获几何特征。该方法借鉴了现有光度立体网络的两阶段设计、VGGT架构和像素采样Transformer等元素。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练3D重建模型的高级几何先验来增强光度立体技术，特别是在多光照条件有限的情况下。整体流程：1) 输入多张不同光照条件下的图像；2) 双分支编码器同时提取光照感知特征和几何特征；3) 双尺度解码器在随机采样的像素位置预测表面法线；4) 输出预测的表面法线图。其中EncoderGeo使用冻结的3D重建模型提取几何特征，EncoderIL使用Transformer提取多光照特征，两者特征拼接后输入解码器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1) 首次将预训练3D重建模型的几何先验整合到光度立体网络；2) 提出'光照-几何双分支编码器'架构；3) 构建PS-Perp数据集，首个具有真实透视投影的合成训练数据集。相比之前工作：传统方法依赖多光照变化，当变化不可靠时性能下降；GeoUniPS通过几何先验提供额外指导；现有方法完全在合成数据上训练，缺乏真实世界几何上下文；之前数据集假设正交投影，而PS-Perp解决了现实世界透视投影的差距。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了GeoUniPS，一种结合预训练3D重建模型几何先验的双分支光度立体网络，显著提升了在复杂现实场景和有限光照条件下表面法线估计的准确性和鲁棒性。'}


### 论文摘要

Universal Photometric Stereo is a promising approach for recovering surface normals without strict lighting assumptions. However, it struggles when multi-illumination cues are unreliable, such as under biased lighting or in shadows or self-occluded regions of complex in-the-wild scenes. We propose GeoUniPS, a universal photometric stereo network that integrates synthetic supervision with high-level geometric priors from large-scale 3D reconstruction models pretrained on massive in-the-wild data. Our key insight is that these 3D reconstruction models serve as visual-geometry foundation models, inherently encoding rich geometric knowledge of real scenes. To leverage this, we design a Light-Geometry Dual-Branch Encoder that extracts both multi-illumination cues and geometric priors from the frozen 3D reconstruction model. We also address the limitations of the conventional orthographic projection assumption by introducing the PS-Perp dataset with realistic perspective projection to enable learning of spatially varying view directions. Extensive experiments demonstrate that GeoUniPS delivers state-of-the-arts performance across multiple datasets, both quantitatively and qualitatively, especially in the complex in-the-wild scenes.

---

## 32. Inertia-Informed Orientation Priors for Event-Based Optical Flow Estimation

**论文链接:** [http://arxiv.org/abs/2511.12961v1](http://arxiv.org/abs/2511.12961v1)

**作者:** Pritam P. Karmokar, William J. Beksi

**发布时间:** 2025-11-17

**备注:** 13 pages, 9 figures, and 3 tables

### GPT解析

### 总结

本文提出了一种新型生物启发混合对比最大化方法，用于事件相机光流估计，通过结合视觉和惯性运动线索，使用方向图作为先验指导优化过程，提高了估计的鲁棒性和收敛性。

### 背景

事件相机直接编码场景中的运动，但其时间密集而空间稀疏的特性带来了挑战。对比最大化(CM)是一种突出的基于模型的优化方法，但仍然是高度非凸的优化问题。

### 目的

解决事件相机光流估计中对比最大化方法的非凸优化问题，提高估计的准确性和鲁棒性。

### 方法

提出一种生物启发的混合对比最大化方法，结合视觉和惯性运动线索，使用从相机3D速度推导的方向图作为先验来指导CM过程，为运动轨迹估计提供方向引导并约束搜索空间。

### 主要发现

方向引导的公式显著提高了事件光流估计的鲁棒性和收敛性，使得优化过程更加稳定和准确。

### 结论

在MVSEC、DSEC和ECD数据集上的评估表明，该方法取得了优于当前最先进水平的精度分数，证明了其在事件光流估计中的有效性。

### 翻译

事件相机凭借其工作原理，直接编码场景中的运动。存在许多基于学习和基于模型的方法来估计事件光流，然而事件在时间上密集而在空间上稀疏的特性带来了重大挑战。为解决这些问题，对比最大化(CM)是一种突出的基于模型的优化方法，通过最优扭曲事件来估计事件体积内的运动轨迹。自提出以来，CM框架经历了计算机视觉社区的一系列改进。尽管如此，它仍然是一个高度非凸的优化问题。在本文中，我们为事件光流估计引入了一种新型生物启发的混合CM方法，该方法结合了视觉和惯性运动线索。具体而言，我们提议使用从相机3D速度推导的方向图作为先验来指导CM过程。方向图提供方向指导并约束估计的运动轨迹空间。我们表明，这种方向引导的公式在事件光流估计中提高了鲁棒性和收敛性。我们在MVSEC、DSEC和ECD数据集上对我们方法的评估产生了优于当前最先进水平的精度分数。


### 论文摘要

Event cameras, by virtue of their working principle, directly encode motion within a scene. Many learning-based and model-based methods exist that estimate event-based optical flow, however the temporally dense yet spatially sparse nature of events poses significant challenges. To address these issues, contrast maximization (CM) is a prominent model-based optimization methodology that estimates the motion trajectories of events within an event volume by optimally warping them. Since its introduction, the CM framework has undergone a series of refinements by the computer vision community. Nonetheless, it remains a highly non-convex optimization problem. In this paper, we introduce a novel biologically-inspired hybrid CM method for event-based optical flow estimation that couples visual and inertial motion cues. Concretely, we propose the use of orientation maps, derived from camera 3D velocities, as priors to guide the CM process. The orientation maps provide directional guidance and constrain the space of estimated motion trajectories. We show that this orientation-guided formulation leads to improved robustness and convergence in event-based optical flow estimation. The evaluation of our approach on the MVSEC, DSEC, and ECD datasets yields superior accuracy scores over the state of the art.

---

## 33. A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation

**论文链接:** [http://arxiv.org/abs/2511.12947v1](http://arxiv.org/abs/2511.12947v1)

**作者:** Hao Jiang, Guoquan Wang, Sheng Yu, Yang Zeng, Wencong Zeng, Guorui Zhou

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文针对本地生活推荐系统中的空间约束和长尾稀疏性问题，提出了一种以物品为中心的ReST框架，通过元ID预热网络和空间约束ID表示增强网络(SIDENet)来增强长尾物品的表示能力。

### 背景

本地生活推荐系统快速发展，为用户提供了便捷获取日常必需品的渠道，但面临两大挑战：空间约束（物品仅展示给有限地理区域内的用户）和长尾稀疏性（热门物品主导交互，高质量长尾物品被忽视）。

### 目的

解决本地生活推荐系统中的空间约束和长尾稀疏性问题，提高长尾物品的曝光率和推荐效果。

### 方法

提出ReST框架，包含元ID预热网络和空间约束ID表示增强网络(SIDENet)。SIDENet采用对比学习，结合空间约束硬采样策略和动态表示对齐策略，根据属性级信息识别弱ID表示并增强其表示能力。

### 主要发现

以物品为中心的视角比传统的以用户为中心的视角更适合本地生活推荐系统；通过注入属性级语义信息和空间约束特征可以显著提升长尾物品的表示质量。

### 结论

ReST框架能有效解决本地生活推荐系统中的空间约束和长尾稀疏性问题，同时保持与热门物品的兼容性，为本地生活推荐提供了新的解决方案。

### 翻译

本地生活推荐已见证快速增长，为用户提供了便捷获取日常必需品的渠道。然而，该领域面临两个关键挑战：(1)空间约束，由本地生活场景的要求驱动，物品通常只展示给有限地理区域内的用户，间接降低了它们的曝光概率；(2)长尾稀疏性，少数热门物品主导用户交互，而许多高质量长尾物品由于交互机会不均而被忽视。现有方法通常采用以用户为中心的视角，如建模空间用户偏好或使用协同过滤信号增强长尾表示。然而，我们认为以物品为中心的视角更适合该领域，重点是增强符合本地生活服务空间约束特征的长尾物品表示。为解决这个问题，我们提出ReST，一种用于长尾本地生活推荐的即插即用空间约束表示增强框架。具体而言，我们首先引入元ID预热网络，通过注入基本属性级别的语义信息来初始化基础ID表示。随后，我们提出了一种基于对比学习的空间约束ID表示增强网络(SIDENet)，它包含两种高效策略：空间约束硬采样策略和动态表示对齐策略。这种设计根据属性级信息在训练期间自适应识别弱ID表示，并通过捕获本地生活服务空间约束特征中的潜在物品关系来增强它们，同时保持与热门物品的兼容性。


### 论文摘要

Local-life recommendation have witnessed rapid growth, providing users with convenient access to daily essentials. However, this domain faces two key challenges: (1) spatial constraints, driven by the requirements of the local-life scenario, where items are usually shown only to users within a limited geographic area, indirectly reducing their exposure probability; and (2) long-tail sparsity, where few popular items dominate user interactions, while many high-quality long-tail items are largely overlooked due to imbalanced interaction opportunities. Existing methods typically adopt a user-centric perspective, such as modeling spatial user preferences or enhancing long-tail representations with collaborative filtering signals. However, we argue that an item-centric perspective is more suitable for this domain, focusing on enhancing long-tail items representation that align with the spatially-constrained characteristics of local lifestyle services. To tackle this issue, we propose ReST, a Plug-And-Play Spatially-Constrained Representation Enhancement Framework for Long-Tail Local-Life Recommendation. Specifically, we first introduce a Meta ID Warm-up Network, which initializes fundamental ID representations by injecting their basic attribute-level semantic information. Subsequently, we propose a novel Spatially-Constrained ID Representation Enhancement Network (SIDENet) based on contrastive learning, which incorporates two efficient strategies: a spatially-constrained hard sampling strategy and a dynamic representation alignment strategy. This design adaptively identifies weak ID representations based on their attribute-level information during training. It additionally enhances them by capturing latent item relationships within the spatially-constrained characteristics of local lifestyle services, while preserving compatibility with popular items.

---

## 34. Part-X-MLLM: Part-aware 3D Multimodal Large Language Model

**论文链接:** [http://arxiv.org/abs/2511.13647v1](http://arxiv.org/abs/2511.13647v1)

**作者:** Chunshi Wang, Junliang Ye, Yunhan Yang, Yang Li, Zizhuo Lin, Jun Zhu, Zhuo Chen, Yawei Luo, Chunchao Guo

**发布时间:** 2025-11-17

### GPT解析

### 总结

Part-X-MLLM是一种原生的3D多模态大语言模型，通过将不同3D任务构造成结构化程序来统一处理，能够从RGB点云和自然语言提示生成包含部件级边界框、语义描述和编辑命令的连贯标记序列。

### 背景

现有3D任务处理方法缺乏统一框架，难以有效整合多种3D处理任务。

### 目的

开发一种能够统一处理多样化3D任务的语言模型，通过结构化输出驱动下游几何感知模块。

### 方法

采用双编码器架构预训练以解构结构和语义，在以部件为中心的大规模数据集上进行指令微调，通过符号规划与几何合成解耦实现兼容几何引擎的控制。

### 主要发现

模型在生成高质量、结构化计划方面表现出色，通过统一接口实现了基础问答、组合生成和局部编辑等任务的最先进性能。

### 结论

Part-X-MLLM为3D多模态任务提供了一个统一的语言原生前端，有效解耦了符号规划与几何合成，使任何兼容几何引擎都能通过单一接口控制。

### 翻译

我们介绍了Part-X-MLLM，一种原生的3D多模态大语言模型，通过将多样化的3D任务构造成结构化、可执行语法中的程序来统一这些任务。给定RGB点云和自然语言提示，我们的模型自回归生成编码部件级边界框、语义描述和编辑命令的单一连贯标记序列。这种结构化输出作为驱动下游几何感知模块的通用接口，用于基于部件的生成和编辑。通过将符号规划与几何合成解耦，我们的方法允许任何兼容的几何引擎通过单一、语言原生前端进行控制。我们预训练了一个双编码器架构来解构结构和语义，并在大规模、以部件为中心的数据集上对模型进行指令微调。实验证明，我们的模型在生成高质量、结构化计划方面表现出色，通过统一接口实现了在基础问答、组合生成和局部编辑方面的最先进性能。项目页面：https://chunshi.wang/Part-X-MLLM/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D多模态大模型无法理解和操作物体部件结构的问题。现实世界中，物体本质上是由有意义的部件组成的，而现有模型大多将物体视为静态整体，无法实现细粒度的语义理解、组合编辑和程序化动画。这个问题在艺术创作（如替换物体部件而不影响整体）、机器人学（如抓取特定部件）和下游应用（如动画和模拟）中都非常重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将3D交互重新表述为语言建模问题，观察到不同任务（生成、编辑、问答）可以在部件的结构化感知语法下统一。他们借鉴了多模态理解（如PointLLM）、3D生成（如SDS方法）、部件生成（2D和3D方法）和3D编辑等现有工作，但发现这些方法各有局限，无法同时实现部件理解、持久引用和可执行编辑。因此，作者设计了双编码器架构和结构化规划语言，将几何结构和视觉 appearance 分离处理，并通过程序式输出统一各种3D任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D交互视为可执行程序的生成：从RGB点云和文本输入发出包含部件级边界框、语义描述和编辑命令的单一令牌序列，几何引擎执行这些程序。整体流程包括：1)双编码器分别处理几何结构和视觉外观；2)融合这些特征与文本提示；3)自回归生成器输出结构化程序；4)下游几何模块执行程序，实现部件感知的生成和编辑。这种设计允许通过单一语言原生接口控制各种几何引擎。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一架构，将生成、编辑和推理统一为部件语法中的程序；2)双编码器架构，解耦结构(XYZ+法线)和外观(RGB)，避免表示冲突；3)语义粒度控制，通过聚类实现从粗略到细粒度的部件控制；4)建立UniPart-Bench基准，涵盖11个任务家族。相比之前工作，Part-X-MLLM首次实现了部件理解与命名、持久边界框引用、可执行编辑程序生成，并通过单一接口提供可控语义粒度，真正将3D交互转化为语言建模问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Part-X-MLLM通过将3D交互转化为可执行程序生成，实现了对3D形状的全面部件感知理解，统一了生成、问答和编辑任务，并通过双编码器架构提供了对3D资产的精确语言控制。'}


### 论文摘要

We introduce Part-X-MLLM, a native 3D multimodal large language model that unifies diverse 3D tasks by formulating them as programs in a structured, executable grammar. Given an RGB point cloud and a natural language prompt, our model autoregressively generates a single, coherent token sequence encoding part-level bounding boxes, semantic descriptions, and edit commands. This structured output serves as a versatile interface to drive downstream geometry-aware modules for part-based generation and editing. By decoupling the symbolic planning from the geometric synthesis, our approach allows any compatible geometry engine to be controlled through a single, language-native frontend. We pre-train a dual-encoder architecture to disentangle structure from semantics and instruction-tune the model on a large-scale, part-centric dataset. Experiments demonstrate that our model excels at producing high-quality, structured plans, enabling state-of-the-art performance in grounded Q\&A, compositional generation, and localized editing through one unified interface. Project page: https://chunshi.wang/Part-X-MLLM/

---

## 35. FUSE: A Flow-based Mapping Between Shapes

**论文链接:** [http://arxiv.org/abs/2511.13431v1](http://arxiv.org/abs/2511.13431v1)

**作者:** Lorenzo Olearo, Giulio Viganò, Daniele Baieri, Filippo Maggioli, Simone Melzi

**发布时间:** 2025-11-17

**备注:** 11 pages, 9 figures

### GPT解析

### 总结

该论文提出了一种基于流匹配模型的3D形状间映射的新型神经表示方法，该方法计算效率高，支持跨表示形状匹配，无需大规模训练或数据驱动程序。

### 背景

在3D形状处理领域，不同表示形式（点云、网格、符号距离场和体积数据）之间的形状匹配是一个重要挑战，传统方法通常需要大量训练数据或计算资源。

### 目的

开发一种计算效率高、支持跨表示形状映射的神经表示方法，能够在没有大规模训练或数据驱动程序的情况下实现精确的形状匹配。

### 方法

该方法将3D形状表示为由从固定锚分布到连续可逆流映射诱导的概率分布。给定源形状和目标形状，通过反向流（从源到锚）与前向流（从锚到目标）的组合，实现两点表面间的连续映射。通过使用点级任务定制嵌入对形状进行编码，该方法提供了形状间映射的可逆和模态无关表示。

### 主要发现

所提出的表示方法在各种基准和具有挑战性的形状匹配场景中均能实现高覆盖率和准确性。除了形状匹配外，该方法在UV映射和人体原始点云扫描配准等其他任务中也显示出 promising 的结果。

### 结论

这种基于流匹配模型的神经表示方法为3D形状间映射提供了一种高效、通用的解决方案，在多种形状表示形式和任务中表现出色，具有广泛的应用前景。

### 翻译

我们介绍了一种基于流匹配模型的3D形状间映射的新型神经表示，该方法计算效率高，支持跨表示形状匹配，无需大规模训练或数据驱动程序。3D形状被表示为由从固定锚分布的连续可逆流映射诱导的概率分布。给定源形状和目标形状，通过反向流（从源到锚）与前向流（从锚到目标）的组合，我们实现了两点表面间的连续映射。通过使用点级任务定制嵌入对形状进行编码，该构造提供了点云、网格、符号距离场和体积数据间形状映射的可逆和模态无关表示。在各种基准和具有挑战性的形状匹配场景中，所得到的表示方法均能实现高覆盖率和准确性。除了形状匹配外，我们的框架在其他任务中也显示出 promising 的结果，包括UV映射和人体原始点云扫描的配准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D形状之间的映射表示问题，特别是不同表示方式（如点云、网格、符号距离场等）之间的形状匹配问题。这个问题在现实中很重要，因为准确的形状对应关系可以解决多个下游任务（如姿态转换、形状插值等），而现有方法大多局限于特定几何表示，难以跨表示通用，限制了方法的适用范围和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分解问题，引入概率分布视角，将形状视为概率分布，并借鉴Flow Matching框架来构建连续可逆的流动映射。他们选择标准高斯分布作为共享锚分布，使任何形状都可以映射到这个共同空间。主要借鉴了Flow Matching（特别是Conditional Flow Matching）、功能映射（Functional Maps）框架和几何分布（Geometry Distributions）方法，但进行了创新性改进以支持跨表示的形状匹配。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过Flow Matching模型构建基于流动的形状映射表示，将每个形状的嵌入分布通过连续可逆的流动映射到固定的共享锚分布（标准高斯分布）。实现流程包括：1) 为形状采样点计算特征嵌入；2) 训练两个独立的Flow模型，将形状嵌入映射到高斯分布；3) 组合这两个流动形成从源形状到目标形状的映射；4) 在嵌入空间执行最近邻搜索获取对应关系。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个跨表示的神经形状映射表示、基于Flow Matching的概率对齐机制、无需大规模训练的计算效率、表示不变性。相比之前的工作，与GeomDist相比使用了更精确的Flow Matching而非去噪扩散，且支持更通用的嵌入；与Functional Maps相比不依赖谱基，适用于各种几何表示；与最优传输相比在不同密度网格上表现更好；与基于配准的方法相比不需要每对形状的优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FUSE提出了一种基于Flow Matching的跨表示3D形状映射方法，通过将形状嵌入映射到共享的高斯锚分布，实现了无需大规模训练的高效、准确的形状匹配，并支持多种3D表示之间的零样本对应关系。'}


### 论文摘要

We introduce a novel neural representation for maps between 3D shapes based on flow-matching models, which is computationally efficient and supports cross-representation shape matching without large-scale training or data-driven procedures. 3D shapes are represented as the probability distribution induced by a continuous and invertible flow mapping from a fixed anchor distribution. Given a source and a target shape, the composition of the inverse flow (source to anchor) with the forward flow (anchor to target), we continuously map points between the two surfaces. By encoding the shapes with a pointwise task-tailored embedding, this construction provides an invertible and modality-agnostic representation of maps between shapes across point clouds, meshes, signed distance fields (SDFs), and volumetric data. The resulting representation consistently achieves high coverage and accuracy across diverse benchmarks and challenging settings in shape matching. Beyond shape matching, our framework shows promising results in other tasks, including UV mapping and registration of raw point cloud scans of human bodies.

---

## 36. Learning Cosmology from Nearest Neighbour Statistics

**论文链接:** [http://arxiv.org/abs/2511.13393v1](http://arxiv.org/abs/2511.13393v1)

**作者:** Atrideb Chatterjee, Arka Banerjee, Francisco Villaescusa-Navarro, Tom Abel

**发布时间:** 2025-11-17

**备注:** Submitted for publication to A&A

### GPT解析

### 总结

本文介绍了一种基于NN距离图和kNN统计的混合神经网络方法，用于从星系/星系团目录中提取宇宙学参数，实现了亚百分比的精度，并且在计算效率上优于现有方法。

### 背景

现代宇宙学中，从星系/星系团目录中以亚百分比的精度提取宇宙学参数非常重要，特别是在Euclid、DESI和LSST等正在进行和即将进行的调查背景下。传统的两点统计方法对于这个任务已被证明不是最优的。

### 目的

开发一种新的方法，能够从星系/星系团目录中以高精度提取宇宙学参数，并提高计算效率。

### 方法

基于kNN统计，作者引入了一种离散星系团目录的新字段级表示方法——NN距离图。将这些地图与基于kNN的汇总统计相结合，训练了一个混合神经网络来推断宇宙学参数。该方法应用于从Quijote N体模拟套件中获取的星系团目录。

### 主要发现

通过这种方法获得的约束达到了最先进的准确性，如果不是最好的话。此外，与一些现有的基于点云的机器学习方法相比，该混合框架在计算效率上提高了5-10倍。

### 结论

基于NN距离图和kNN统计的混合神经网络方法是一种有效且高效的宇宙学参数提取方法，能够实现高精度和计算效率。

### 翻译

从星系/星系团目录中以亚百分比精度提取宇宙学参数是现代宇宙学的一个重要方面，特别是考虑到正在进行和即将进行的调查，如Euclid、DESI和LSST。虽然传统的两点统计方法已被证明对于这个任务不是最优的，但最近提出的基于k-最近邻的汇总统计量显示出更强的约束力。基于kNN统计，我们引入了一种离散星系团目录的新字段级表示方法——NN距离图。我们将这项技术应用于从Quijote N体模拟套件中获取的星系团目录。通过将这些地图与基于kNN的汇总统计相结合，我们训练了一个混合神经网络来推断宇宙学参数，结果表明由此获得的约束达到了最先进的准确性，如果不是最好的话。此外，与一些现有的基于点云的机器学习方法相比，我们的混合框架在计算效率上提高了5-10倍。


### 论文摘要

Extracting cosmological parameters from galaxy/halo catalogues with sub-percent level accuracy is an important aspect of modern cosmology, especially in view of ongoing and upcoming surveys such as Euclid, DESI, and LSST. While traditional two-point statistics have been known to be suboptimal for this task, recently proposed k-Nearest Neighbour (kNN) based summary statistics have demonstrated tighter constraining power. Building on the kNN statistics, we introduce a new field-level representation of discrete halo catalogues - NN distance maps. We employ this technique on the halo catalogues obtained from Quijote N-body simulation suites. By combining these maps with kNN-based summary statistics, we train a hybrid neural network to infer cosmological parameters, showing that the resulting constraints achieve state-of-the-art, if not the best, accuracy. In addition, our hybrid framework is 5-10 times more computationally efficient than some of the existing point-cloud-based ML methods.

---

## 37. DriveLiDAR4D: Sequential and Controllable LiDAR Scene Generation for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.13309v1](http://arxiv.org/abs/2511.13309v1)

**作者:** Kaiwen Cai, Xinze Liu, Xia Zhou, Hengtong Hu, Jie Xiang, Luyao Zhang, Xueyang Zhang, Kun Zhan, Yifei Zhan, Xianpeng Lang

**发布时间:** 2025-11-17

**备注:** AAAI2026

### GPT解析

### 总结

本文提出了DriveLiDAR4D，一种新的LiDAR点云生成管道，能够生成时间一致且具有高度可控前景对象和逼真背景的LiDAR场景。

### 背景

生成真实的LiDAR点云对自动驾驶系统的发展与评估至关重要，但现有3D LiDAR点云生成方法存在明显局限性，包括缺乏序列生成能力，无法生成准确定位的前景对象和逼真的背景。

### 目的

解决现有方法的局限性，提出一种能够生成时间一致且具有高度可控前景对象和逼真背景的LiDAR场景的生成方法。

### 方法

提出了DriveLiDAR4D，一种新的LiDAR生成管道，包含多模态条件和一种新的序列噪声预测模型LiDAR4DNet，能够以端到端方式处理整个场景的LiDAR序列生成。

### 主要发现

在nuScenes和KITTI数据集上评估显示，DriveLiDAR4D在nuScenes数据集上实现了743.13的FRD分数和16.96的FVD分数，超越了当前最先进的UniScene方法，FRD性能提升37.2%，FVD性能提升24.1%。

### 结论

DriveLiDAR4D是首个能够以端到端方式处理整个场景操作能力的LiDAR场景序列生成工作，显著提升了LiDAR点云生成的质量和实用性。

### 翻译

生成逼真的LiDAR点云对自动驾驶系统的发展与评估起着至关重要的作用。尽管最近3D LiDAR点云生成方法取得了显著改进，但仍面临明显局限，包括缺乏序列生成能力和无法生成准确定位的前景对象及逼真背景。这些缺点限制了它们的实际应用。在本文中，我们引入了DriveLiDAR4D，一种新的LiDAR生成管道，包含多模态条件和一种新的序列噪声预测模型LiDAR4DNet，能够生成具有高度可控前景对象和逼真背景的时间一致的LiDAR场景。据我们所知，这是首个以端到端方式处理整个场景操作能力的LiDAR场景序列生成工作。我们在nuScenes和KITTI数据集上评估了DriveLiDAR4D，在nuScenes数据集上实现了743.13的FRD分数和16.96的FVD分数，超越了当前最先进的UniScene方法，FRD性能提升37.2%，FVD性能提升24.1%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云生成方法的局限性，包括缺乏序列生成能力、无法生成准确定位的前景物体以及无法生成真实的背景。这个问题在自动驾驶领域非常重要，因为高质量的LiDAR数据对自动驾驶系统的开发和评估至关重要，但收集和标注真实世界数据既耗时又耗费资源，而现有生成方法无法满足自动驾驶系统对多样化、精确控制场景数据的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（如RangeLDM无法生成准确车辆、Text2LiDAR背景描述粗糙、LidarDM缺乏背景控制等）进行设计。他们借鉴了扩散模型、等距柱状图表示和UNet架构等现有技术，但创新性地设计了多模态条件（道路草图、场景描述和物体先验）和LiDAR4DNet模型，包括EST-Conv和EST-Trans模块，以实现端到端的序列LiDAR场景生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多模态条件实现精确的场景控制，并使用新型LiDAR4DNet模型确保生成序列的空间和时间一致性。整体流程包括：1) 准备多模态条件（道路草图、场景描述和物体先验）；2) 使用LiDAR4DNet基于扩散模型进行噪声预测；3) 通过训练学习预测添加的噪声；4) 在推理时从噪声开始迭代去噪，生成时序一致的LiDAR场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次实现对前景物体和背景元素的精确控制；2) 提出端到端的4D LiDAR场景生成框架；3) 设计多模态条件结合场景描述、道路草图和物体先验；4) 提出LiDAR4DNet模型，包含EST-Conv和EST-Trans模块增强时空一致性。相比之前工作，DriveLiDAR4D能够生成序列场景、精确控制前景和背景，并在nuScenes数据集上超越当前SOTA方法37.2%的FRD性能和24.1%的FVD性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DriveLiDAR4D首次实现了具有精确前景和背景控制能力的端到端序列LiDAR场景生成，为自动驾驶系统的开发和评估提供了更真实、可控的合成数据解决方案。'}


### 论文摘要

The generation of realistic LiDAR point clouds plays a crucial role in the development and evaluation of autonomous driving systems. Although recent methods for 3D LiDAR point cloud generation have shown significant improvements, they still face notable limitations, including the lack of sequential generation capabilities and the inability to produce accurately positioned foreground objects and realistic backgrounds. These shortcomings hinder their practical applicability. In this paper, we introduce DriveLiDAR4D, a novel LiDAR generation pipeline consisting of multimodal conditions and a novel sequential noise prediction model LiDAR4DNet, capable of producing temporally consistent LiDAR scenes with highly controllable foreground objects and realistic backgrounds. To the best of our knowledge, this is the first work to address the sequential generation of LiDAR scenes with full scene manipulation capability in an end-to-end manner. We evaluated DriveLiDAR4D on the nuScenes and KITTI datasets, where we achieved an FRD score of 743.13 and an FVD score of 16.96 on the nuScenes dataset, surpassing the current state-of-the-art (SOTA) method, UniScene, with an performance boost of 37.2% in FRD and 24.1% in FVD, respectively.

---

## 38. A Lightweight 3D Anomaly Detection Method with Rotationally Invariant Features

**论文链接:** [http://arxiv.org/abs/2511.13115v1](http://arxiv.org/abs/2511.13115v1)

**作者:** Hanzhe Liang, Jie Zhou, Can Gao, Bingyang Guo, Jinbao Wang, Linlin Shen

**发布时间:** 2025-11-17

**备注:** Submitted to Elsevier

### GPT解析

### 总结

本文提出了一种名为RIF的3D异常检测框架，通过点坐标映射技术和轻量级卷积变换特征网络解决了点云数据在方向和位置变化时的特征一致性问题，实验证明该方法在多个数据集上取得了先进性能。

### 背景

3D异常检测是计算机视觉中的重要任务，旨在从点云数据中识别异常点或区域。然而，现有方法在处理方向和位置变化的点云时可能遇到挑战，因为特征可能会有显著差异。

### 目的

解决现有3D异常检测方法在处理方向和位置变化的点云数据时遇到的特征不一致问题，提高异常检测的准确性和鲁棒性。

### 方法

提出了一种名为旋转不变特征(RIF)的框架，包括：1)点坐标映射(PCM)技术，将每个点映射到旋转不变空间以保持表示一致性；2)轻量级卷积变换特征网络(CTF-Net)，用于提取旋转不变特征；3)使用迁移学习和3D数据增强预训练特征提取器。

### 主要发现

在Anomaly-ShapeNet数据集上，所提方法实现了平均17.7%的P-AUROC提升；在Real3D-AD数据集上，实现了平均1.6%的P-AUROC提升。将RIF与传统特征提取方法结合使用时，表现出强大的泛化能力。

### 结论

RIF框架通过提取旋转不变特征有效解决了点云数据在不同方向和位置变化时的特征不一致问题，在多个数据集上取得了先进性能，与传统方法结合表现出强大的泛化能力，具有工业应用的巨大潜力。

### 翻译

三维异常检测是计算机视觉中的一个关键任务，旨在从点云数据中识别异常点或区域。然而，当处理方向和位置变化的点云时，现有方法可能会面临挑战，因为产生的特征可能会有显著差异。为了解决这个问题，我们提出了一种用于三维异常检测的新型旋转不变特征框架。首先，为了消除变化对点云数据的不良影响，我们开发了一种点坐标映射技术，将每个点映射到旋转不变的空间中，以保持表示的一致性。然后，为了学习鲁棒性和区分性特征，我们设计了一个轻量级卷积变换特征网络，用于为记忆库提取旋转不变特征。为了提高特征提取器的能力，我们引入了迁移学习的思想，使用三维数据增强预训练特征提取器。实验结果表明，所提出的方法在Anomaly-ShapeNet数据集上达到了先进性能，平均P-AUROC提高了17.7%，在Real3D-AD数据集上也获得了最佳性能，平均P-AUROC提高了1.6%。通过将RIF与传统特征提取方法结合用于异常检测任务，其强大的泛化能力已得到验证，显示出工业应用的巨大潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云异常检测中的旋转不变性问题。当点云数据发生旋转和位置变化时，现有方法提取的特征会产生显著差异，影响异常检测准确性。在工业应用中，产品在生产线上可能以不同姿态出现，导致同一产品具有不同朝向和位置的点云数据，这使得异常检测变得困难且不准确。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D异常检测方法在处理旋转变化时的局限性。他们借鉴了特征嵌入方法和重建方法的基本思路，但发现这些方法在面对旋转变化时表现不佳。作者还参考了一些使用数据增强和配准技术来缓解旋转影响的方法，以及采用多视图策略从不同角度捕获特征的方法。基于这些观察，作者设计了点坐标映射技术将点云转换到旋转不变空间，并开发了轻量级卷积变换特征网络和空间3D数据增强策略来提取鲁棒特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云数据映射到一个旋转不变的空间，确保无论点云如何旋转和移动，都能产生一致的坐标表示，从而提取稳定的特征。整体流程包括：1)点坐标映射(PCM)：计算质心，选择关键向量定义相对坐标系，通过正交化生成标准正交基，将点变换到相对坐标系；2)预训练特征提取器(CTF-Net)：使用ModelNet40数据集和空间3D数据增强(S3DA)预训练网络；3)异常检测：训练阶段对正常样本进行坐标映射、分组并提取特征存入记忆库；测试阶段对测试样本进行相同处理，提取特征并与记忆库进行最近邻搜索，生成异常分数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)点坐标映射(PCM)技术：将点云映射到旋转不变空间，避免配准过程；2)卷积变换特征网络(CTF-Net)：轻量级设计(4.1M参数)，使用组合卷积块捕获多尺度特征；3)空间3D数据增强(S3DA)：增加训练数据多样性；4)旋转不变特征(RIF)框架：可与其他特征提取器集成。相比之前工作，本文方法不依赖计算密集的配准技术，避免了配准错误；比多视图方法更直接高效；参数量更少，内存消耗低(26%)；在噪声环境下表现更鲁棒，对高斯噪声的抵抗力更强。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一个基于旋转不变特征的轻量级3D异常检测框架，通过点坐标映射技术将点云转换到旋转不变空间，结合轻量级特征提取网络和数据增强策略，显著提高了工业场景中面对旋转变化时的异常检测性能和鲁棒性。'}


### 论文摘要

3D anomaly detection (AD) is a crucial task in computer vision, aiming to identify anomalous points or regions from point cloud data. However, existing methods may encounter challenges when handling point clouds with changes in orientation and position because the resulting features may vary significantly. To address this problem, we propose a novel Rotationally Invariant Features (RIF) framework for 3D AD. Firstly, to remove the adverse effect of variations on point cloud data, we develop a Point Coordinate Mapping (PCM) technique, which maps each point into a rotationally invariant space to maintain consistency of representation. Then, to learn robust and discriminative features, we design a lightweight Convolutional Transform Feature Network (CTF-Net) to extract rotationally invariant features for the memory bank. To improve the ability of the feature extractor, we introduce the idea of transfer learning to pre-train the feature extractor with 3D data augmentation. Experimental results show that the proposed method achieves the advanced performance on the Anomaly-ShapeNet dataset, with an average P-AUROC improvement of 17.7\%, and also gains the best performance on the Real3D-AD dataset, with an average P-AUROC improvement of 1.6\%. The strong generalization ability of RIF has been verified by combining it with traditional feature extraction methods on anomaly detection tasks, demonstrating great potential for industrial applications.

---

## 39. ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes

**论文链接:** [http://arxiv.org/abs/2511.12977v1](http://arxiv.org/abs/2511.12977v1)

**作者:** Yixuan Yang, Luyang Xie, Zhen Luo, Zixiang Zhao, Mingqi Gao, Feng Zheng

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了ArtiWorld，一个能够从文本场景描述中自动识别可关节化物体并重建为保留原始几何形状的可执行URDF模型的场景感知流水线。

### 背景

构建交互式模拟器和可扩展的机器人学习环境需要大量关节式资产，但现有3D资产大多是刚性的，手动将它们转换为关节式物体极其耗费劳动力和成本。

### 目的

自动识别场景中的可关节化物体，并将其直接转换为关节式资产，以简化交互式模拟器和机器人学习环境的构建过程。

### 方法

提出ArtiWorld流水线，其核心是Arti4URDF，它利用3D点云、大型语言模型的先验知识和面向URDF的提示设计，将刚性物体快速转换为基于URDF的交互式关节式物体。

### 主要发现

在3D模拟物体、完整3D模拟场景和真实世界扫描场景三个级别上评估，ArtiWorld始终优于现有方法，取得最先进性能，同时保留了物体几何形状和交互性。

### 结论

ArtiWorld为直接从现有3D资产构建交互式、机器人就绪的模拟环境提供了一条实用路径，代码和数据将发布。

### 翻译

构建交互式模拟器和可扩展的机器人学习环境需要大量关节式资产。然而，模拟中大多数现有的3D资产是刚性的，手动将它们转换为关节式物体极其耗费劳动力和成本。这自然引出一个问题：我们能否自动识别场景中的可关节化物体，并将其直接转换为关节式资产？在本文中，我们提出了ArtiWorld，一个场景感知的流水线，它从文本场景描述中定位候选的可关节化物体，并重建保留原始几何形状的可执行URDF模型。该流水线的核心是Arti4URDF，它利用3D点云、大型语言模型的先验知识以及面向URDF的提示设计，将刚性物体快速转换为基于URDF的交互式关节式物体，同时保持其3D形状。我们在三个级别评估ArtiWorld：3D模拟物体、完整的3D模拟场景和真实世界扫描场景。在所有三种设置中，我们的方法始终优于现有方法，并取得了最先进的性能，同时保留了物体几何形状并正确捕捉了物体交互性，以产生可用的基于URDF的关节式模型。这为直接从现有3D资产构建交互式、机器人就绪的模拟环境提供了一条实用路径。代码和数据将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动识别场景中可关节化物体并将其转换为可交互关节化资产的问题。这个问题很重要，因为构建交互式模拟器和机器人学习环境需要大量关节化资产，而手动转换3D资产为关节化物体既耗时又昂贵。现有方法大多产生静态物体，无法支持有意义的机器人交互，解决这个问题可以直接从现有3D资产构建交互式模拟环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从模拟场景和单物体两个角度思考问题，注意到现有场景生成方法将所有物体视为刚性，而单物体关节化建模存在类别受限、设计同质化等问题。他们设计了ArtiWorld统一管道，核心是Arti4URDF模型，将3D点云特征嵌入LLM中。方法借鉴了大型语言模型能力、3D点云编码器（如Point-BERT）以及PartNet-Mobility和PhysXNet数据集，同时参考了CAGE、Articulate-Anything等现有方法但解决了它们的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大型语言模型和3D点云信息自动将静态3D物体转换为可交互关节化物体，同时保留原始几何形状。整体流程包括：1) ArtiWorld管道从文本场景描述识别可关节化物体并检索3D资产；2) 对每个物体采样点云并编码为几何标记；3) Arti4URDF模型提取几何特征并生成URDF文件；4) 将生成的关节化物体对齐回原始场景位置。训练使用PartNet-Mobility和PhysXNet数据集，推理时采用点提示分割策略指导部件分解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) ArtiWorld场景感知管道，可从文本描述识别可关节化物体；2) Arti4URDF模型，将3D特征嵌入LLM直接生成URDF；3) 在三个级别（物体、场景、真实世界）评估并达到最先进性能。相比之前工作，不依赖部件检索（如CAGE、Articulate-Anything），避免了几何简化（如URDFormer）和多任务学习错误（如URDF-Anything），更好地保留了原始几何形状并解决了结构多样性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ArtiWorld通过结合大型语言模型和3D点云分析，实现了从静态3D资产自动生成保留原始几何形状的交互式关节化物体，为构建机器人可交互的模拟环境提供了实用路径。'}


### 论文摘要

Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.

---

## 40. CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.12909v1](http://arxiv.org/abs/2511.12909v1)

**作者:** Yaohua Zha, Xue Yuerong, Chunlin Fan, Yuansong Wang, Tao Dai, Ke Chen, Shu-Tao Xia

**发布时间:** 2025-11-17

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

基于深度学习的3D异常检测方法在工业制造中显示出潜力，但现有方法泛化能力有限。本文提出CASL框架，利用曲率信息实现高效异常检测，无需专门设计。

### 背景

基于深度学习的3D异常检测方法在工业制造中显示出巨大潜力，但许多方法专门为异常检测任务设计，限制了泛化能力。自监督点云模型虽旨在通用表征学习，但在统一微调范式下对异常检测表现不佳。

### 目的

开发一个更具泛化能力的3D模型，能够有效检测异常而不依赖任务特定设计。

### 方法

提出基于重建范式的曲率增强自监督学习（CASL）框架，基于经典U-Net架构，引入多尺度曲率提示指导解码器预测每个点的空间坐标，通过简单异常分类微调实现检测性能。

### 主要发现

仅使用每个点的曲率作为异常评分就优于多种经典自监督和专用异常检测模型，突显曲率在3D异常检测中的关键作用；学习表征能很好地泛化到标准3D理解任务。

### 结论

CASL框架在不依赖专用异常检测机制的情况下实现领先检测性能，具有良好的泛化能力，适用于多种3D理解任务。

### 翻译

基于深度学习的3D异常检测方法在工业制造中显示出巨大潜力。然而，许多方法专门为异常检测任务设计，这限制了它们对其他3D理解任务的泛化能力。相比之下，自监督点云模型旨在实现通用表征学习，但我们的研究表明，这些经典模型在统一的微调范式下对异常检测的表现不佳。这促使我们开发一个更具泛化能力的3D模型，能够有效检测异常而不依赖任务特定设计。有趣的是，我们发现仅使用每个点的曲率作为异常评分就已经优于几种经典自监督和专用异常检测模型，突显了曲率在3D异常检测中的关键作用。在本文中，我们提出了一种基于重建范式的曲率增强自监督学习（CASL）框架。基于经典的U-Net架构，我们的方法引入多尺度曲率提示来指导解码器预测每个点的空间坐标。在不依赖任何专用异常检测机制的情况下，它通过简单的异常分类微调实现了领先的检测性能。此外，学习到的表征能很好地泛化到标准的3D理解任务，如点云分类。代码可在https://github.com/zyh16143998882/CASL获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云异常检测中的通用性问题。现有方法要么是专门为异常检测设计的，缺乏通用性；要么是通用的自监督模型在异常检测上表现不佳。这个问题在工业制造中非常重要，因为高质量的产品检测对质量控制至关重要，而3D点云相比传统2D图像能提供更全面的空间结构信息，开发一种通用且高效的3D异常检测模型将大大提高制造业的质量控制效率和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先研究了现有3D异常检测方法，发现它们缺乏通用性。然后尝试将现有自监督点云模型通过简单微调应用于异常检测，但发现效果不佳，归因于'几何捷径'问题。一个关键发现是仅使用点曲率作为异常评分就优于多种经典方法，这启发了作者设计CASL框架。该方法借鉴了U-Net架构和Minkowski卷积处理点云，以及自监督重建范式，但创新性地引入曲率作为语义表示来避免几何捷径问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用曲率作为3D表面的内在几何属性，将其作为自监督学习中的语义表示，避免传统坐标到坐标重建中的'几何捷径'问题，使模型专注于局部表面变化和细粒度结构细节。整体流程包括：1)曲率编码器提取点级曲率特征；2)坐标编码器处理随机初始化的被掩盖点特征；3)融合解码器结合曲率提示重建点云；4)使用ℓ1和ℓ2损失计算重建误差；5)通过伪异常分类进行微调；6)基于点级分类logits计算异常评分。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)发现曲率在3D异常检测中的重要性；2)提出曲率增强的自监督学习框架(CASL)；3)采用全坐标掩码策略而非部分掩码；4)通过简单伪异常分类实现异常检测而非专用机制。相比之前工作，该方法不依赖特定于异常检测的设计，解决了自监督方法中的几何捷径问题，不使用未掩盖点坐标作为先验，而是用曲率提示引导模型学习更丰富的几何表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文提出了一种曲率增强的自监督学习框架，通过多尺度曲率提示和全坐标掩码策略有效避免了'几何捷径'问题，实现了在3D异常检测任务上的卓越性能，同时保持了模型在其他3D理解任务上的良好泛化能力。"}


### 论文摘要

Deep learning-based 3D anomaly detection methods have demonstrated significant potential in industrial manufacturing. However, many approaches are specifically designed for anomaly detection tasks, which limits their generalizability to other 3D understanding tasks. In contrast, self-supervised point cloud models aim for general-purpose representation learning, yet our investigation reveals that these classical models are suboptimal at anomaly detection under the unified fine-tuning paradigm. This motivates us to develop a more generalizable 3D model that can effectively detect anomalies without relying on task-specific designs. Interestingly, we find that using only the curvature of each point as its anomaly score already outperforms several classical self-supervised and dedicated anomaly detection models, highlighting the critical role of curvature in 3D anomaly detection. In this paper, we propose a Curvature-Augmented Self-supervised Learning (CASL) framework based on a reconstruction paradigm. Built upon the classical U-Net architecture, our approach introduces multi-scale curvature prompts to guide the decoder in predicting the spatial coordinates of each point. Without relying on any dedicated anomaly detection mechanisms, it achieves leading detection performance through straightforward anomaly classification fine-tuning. Moreover, the learned representations generalize well to standard 3D understanding tasks such as point cloud classification. The code is available at https://github.com/zyh16143998882/CASL.

---

## 41. An Evaluation of Representation Learning Methods in Particle Physics Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.12829v1](http://arxiv.org/abs/2511.12829v1)

**作者:** Michael Chen, Raghav Kansal, Abhijith Gandrakota, Zichun Hao, Jennifer Ngadiuba, Maria Spiropulu

**发布时间:** 2025-11-16

### GPT解析

### 总结

本研究对粒子物理中的表示学习目标进行了系统评估，在统一框架下比较了多种学习目标，并引入了改进的监督架构，为粒子物理基础模型的未来发展提供了参考。

### 背景

粒子物理中的表示学习需要系统评估不同学习目标的效果，缺乏统一框架下的比较研究。

### 目的

在统一框架下评估和比较不同表示学习目标，提供可复现的基线，促进粒子物理基础模型的透明和稳健发展。

### 方法

使用基于transformer的粒子云编码器，采用标准化预处理、匹配采样和一致评估协议，在喷流分类数据集上比较对比学习、掩粒子建模和生成重建目标，并引入针对性监督架构修改。

### 主要发现

引入的监督架构修改在基准评估中达到最先进性能，控制比较成功隔离了不同学习目标的贡献，突出了各自的优缺点。

### 结论

这项工作为粒子物理基础模型的未来发展提供了参考点，使社区能够取得更加透明和稳健的进步。

### 翻译

我们提出了一个在统一框架内对粒子物理表示学习目标的系统评估。我们的研究采用了一个共享的基于transformer的粒子云编码器，具有标准化的预处理、匹配的采样以及在喷流分类数据集上的一致评估协议。我们在共同的训练制度下比较了对比学习（监督和自监督）、掩粒子建模和生成重建目标。此外，我们引入了有针对性的监督架构修改，在基准评估中取得了最先进的性能。这种受控比较隔离了学习目标的贡献，突出了它们各自的优缺点，并提供了可复现的基线。我们将这项工作定位为粒子物理基础模型未来发展的参考点，使社区能够取得更加透明和稳健的进展。


### 论文摘要

We present a systematic evaluation of representation learning objectives for particle physics within a unified framework. Our study employs a shared transformer-based particle-cloud encoder with standardized preprocessing, matched sampling, and a consistent evaluation protocol on a jet classification dataset. We compare contrastive (supervised and self-supervised), masked particle modeling, and generative reconstruction objectives under a common training regimen. In addition, we introduce targeted supervised architectural modifications that achieve state-of-the-art performance on benchmark evaluations. This controlled comparison isolates the contributions of the learning objective, highlights their respective strengths and limitations, and provides reproducible baselines. We position this work as a reference point for the future development of foundation models in particle physics, enabling more transparent and robust progress across the community.

---

## 42. Which Way from B to A: The role of embedding geometry in image interpolation for Stable Diffusion

**论文链接:** [http://arxiv.org/abs/2511.12757v1](http://arxiv.org/abs/2511.12757v1)

**作者:** Nicholas Karris, Luke Durell, Javier Flores, Tegan Emerson

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种基于最优传输的CLIP嵌入插值方法，通过将嵌入视为Wasserstein空间中的点云而非欧几里得空间中的矩阵，实现了更平滑、更连贯的图像插值效果。

### 背景

研究发现Stable Diffusion对CLIP嵌入矩阵的行具有排列不变性，这启发研究者将嵌入矩阵解释为点云而非矩阵。

### 目的

探索嵌入空间的几何结构，并提出一种新的插值方法，以获得更平滑、更自然的图像过渡效果。

### 方法

将两个不同提示的嵌入之间的插值问题重新构建为最优传输问题，通过解决该问题计算嵌入之间的最短路径(测地线)，实现更几何上平滑的过渡。

### 主要发现

基于最优传输的插值方法在Stable Diffusion生成模型下产生的中间图像更平滑、更连贯，相比其他标准插值方法效果更好。

### 结论

将CLIP嵌入视为Wasserstein空间中的点云而非欧几里得空间中的矩阵，能更好地反映和利用嵌入空间的几何结构，从而产生更高质量的图像插值结果。

### 翻译

研究表明，Stable Diffusion对对比语言-图像预训练嵌入矩阵的行具有排列不变性。这启发了一个新颖的观察：这些嵌入可以自然地被解释为Wasserstein空间中的点云，而非欧几里得空间中的矩阵。这一视角为理解嵌入空间的几何结构开辟了新的可能性。例如，在两个不同提示的嵌入之间进行插值时，我们提出将插值问题重新构建为最优传输问题。通过解决这一最优传输问题，我们计算了嵌入之间的最短路径(或测地线)，捕获了嵌入空间中更自然和几何上平滑的过渡。当由Stable Diffusion生成模型渲染时，这产生了更平滑和更连贯的中间图像。我们进行了实验来研究这一效果，比较了使用最优传输和其他标准插值方法生成的插值图像质量。所提出的新型基于最优传输的方法确实产生了更平滑的图像插值，这表明将嵌入视为点云(而非矩阵)能更好地反映和利用嵌入空间的几何结构。


### 论文摘要

It can be shown that Stable Diffusion has a permutation-invariance property with respect to the rows of Contrastive Language-Image Pretraining (CLIP) embedding matrices. This inspired the novel observation that these embeddings can naturally be interpreted as point clouds in a Wasserstein space rather than as matrices in a Euclidean space. This perspective opens up new possibilities for understanding the geometry of embedding space. For example, when interpolating between embeddings of two distinct prompts, we propose reframing the interpolation problem as an optimal transport problem. By solving this optimal transport problem, we compute a shortest path (or geodesic) between embeddings that captures a more natural and geometrically smooth transition through the embedding space. This results in smoother and more coherent intermediate (interpolated) images when rendered by the Stable Diffusion generative model. We conduct experiments to investigate this effect, comparing the quality of interpolated images produced using optimal transport to those generated by other standard interpolation methods. The novel optimal transport--based approach presented indeed gives smoother image interpolations, suggesting that viewing the embeddings as point clouds (rather than as matrices) better reflects and leverages the geometry of the embedding space.

---

## 43. Deep Imbalanced Multi-Target Regression: 3D Point Cloud Voxel Content Estimation in Simulated Forests

**论文链接:** [http://arxiv.org/abs/2511.12740v1](http://arxiv.org/abs/2511.12740v1)

**作者:** Amirhossein Hassanzadeh, Bartosz Krawczyk, Michael Saunders, Rob Wible, Keith Krause, Dimah Dera, Jan van Aardt

**发布时间:** 2025-11-16

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

体素化处理激光雷达数据可降低计算成本但会丢失精细结构信息。本研究探讨从高层次体素化点云数据推断低层次体素内容信息的方法，提出基于不平衡学习的多目标回归方法，使用核点卷积并解决类别不平衡问题。研究发现体素大小对精度有显著影响，较大体素尺寸误差较低，较小体素尺寸在树冠区域误差较高。

### 背景

体素化是处理激光雷达数据的一种有效方法，可以降低计算成本，但会导致精细尺度结构信息的丢失。目前需要研究如何从高层次的体素化激光雷达点云数据中推断出低层次的体素内容信息。

### 目的

研究是否可以从高层次的体素化激光雷达点云数据中推断出低层次的体素内容信息，特别是体素内的目标占有率百分比。同时评估不同体素尺寸对森林三维激光雷达点云处理的影响。

### 方法

提出一种基于不平衡学习的多目标回归方法，使用核点卷积(KPConv)。利用基于密度的相关性(DBR)解决类别不平衡问题，采用加权均方误差(MSE)、焦点回归(FocalR)和正则化来优化KPConv。对体素大小(0.25-2米)进行敏感性分析，评估不同网格表示对捕捉森林细微差别的影响。

### 主要发现

较大体素尺寸(如2米)导致较低的误差，而较小的体素尺寸(如0.25或0.5米)显示较高的误差，特别是在树冠区域。对于树皮和叶子目标，较小体素尺寸数据集的误差值显著高于较大体素尺寸数据集。体素大小的选择取决于应用场景。

### 结论

体素化处理激光雷达数据时，体素大小的选择对精度有显著影响。研究填补了深度不平衡学习模型在森林三维激光雷达点云多目标回归和模拟数据集方面的空白。

### 翻译

体素化是处理激光雷达数据的一种有效方法，可以降低计算成本，但会导致精细尺度结构信息的丢失。本研究探讨是否可以从高层次的体素化激光雷达点云数据中推断出低层次的体素内容信息，特别是体素内的目标占有率百分比。研究使用DIRSIG软件收集数据，目标包括树皮、叶子、土壤和其他材料。研究提出了一种基于不平衡学习的多目标回归方法，使用核点卷积(KPConv)。研究利用了基于密度的相关性(DBR)来解决类别不平衡问题，并采用加权均方误差(MSE)、焦点回归(FocalR)和正则化来优化KPConv。研究还对体素大小(0.25-2米)进行了敏感性分析，以评估不同网格表示对捕捉森林细微差别的影响。研究发现，较大的体素尺寸导致较低的误差，而较小的体素尺寸则显示较高的误差，特别是在树冠区域。对于树皮和叶子目标，较小体素尺寸数据集的误差值显著高于较大体素尺寸数据集。研究指出体素大小的选择取决于应用场景。这项工作填补了深度不平衡学习模型在森林三维激光雷达点云多目标回归和模拟数据集方面的空白。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从3D点云数据中推断体素内的目标占用百分比问题，特别是在森林环境中估计树皮、叶子、土壤和其他材料在体素中的占比。这个问题很重要，因为直接处理高分辨率激光雷达数据计算成本高，而体素化虽然降低了计算成本但会丢失精细信息；同时，获取真实森林环境的精确地面真值数据非常困难，准确估计森林内部结构对农业、生态、林业和气候变化研究具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到体素化处理激光雷达数据的计算优势与信息损失之间的矛盾，然后利用DIRSIG软件创建物理基础模拟数据集解决真实数据获取难题。在方法设计上，他们借鉴了现有的点云语义分割技术(特别是KPConv)，并将其适应到多目标回归任务；针对数据不平衡问题，他们改进了现有的成本敏感学习技术，提出基于密度的相关性(DBR)加权方法；在损失函数设计上，结合了加权均方误差、焦点回归和正则化技术。这些创新都是基于对现有工作的理解和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过深度学习多目标回归同时预测多个材料(树皮、叶子、土壤和其他)在体素中的百分比，结合成本敏感学习解决数据不平衡问题，并利用模拟数据克服真实数据获取困难。整体流程包括：1)使用DIRSIG创建哈佛森林场景模拟数据；2)将场景和数据分别体素化并映射组合；3)基于KPConv构建编码器-解码器架构的深度网络；4)设计结合WMSE、FocalR和正则化的损失函数；5)在不同体素大小(0.25-2米)上训练和评估模型；6)分析体素大小对预测精度的影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建基于物理模拟的新颖数据集，提供精确几何计算的地面真值；2)设计专门针对多目标回归的KPConv深度学习架构；3)提出基于密度的相关性(DBR)加权技术处理不平衡数据；4)开发结合WMSE和FocalR的新损失函数；5)进行全面的体素敏感性分析。相比之前工作，本文首次将深度学习多目标回归应用于森林环境，使用模拟数据解决真实数据获取难题，并针对多目标不平衡问题提出创新解决方案，评估方法也更全面。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过创新的深度学习多目标回归方法，结合成本敏感学习和损失函数设计，成功解决了从激光雷达点云体素数据中估计森林内部多种材料百分比的问题，并揭示了体素大小对预测精度的重要影响。'}


### 论文摘要

Voxelization is an effective approach to reduce the computational cost of processing Light Detection and Ranging (LiDAR) data, yet it results in a loss of fine-scale structural information. This study explores whether low-level voxel content information, specifically target occupancy percentage within a voxel, can be inferred from high-level voxelized LiDAR point cloud data collected from Digital Imaging and remote Sensing Image Generation (DIRSIG) software. In our study, the targets include bark, leaf, soil, and miscellaneous materials. We propose a multi-target regression approach in the context of imbalanced learning using Kernel Point Convolutions (KPConv). Our research leverages cost-sensitive learning to address class imbalance called density-based relevance (DBR). We employ weighted Mean Saquared Erorr (MSE), Focal Regression (FocalR), and regularization to improve the optimization of KPConv. This study performs a sensitivity analysis on the voxel size (0.25 - 2 meters) to evaluate the effect of various grid representations in capturing the nuances of the forest. This sensitivity analysis reveals that larger voxel sizes (e.g., 2 meters) result in lower errors due to reduced variability, while smaller voxel sizes (e.g., 0.25 or 0.5 meter) exhibit higher errors, particularly within the canopy, where variability is greatest. For bark and leaf targets, error values at smaller voxel size datasets (0.25 and 0.5 meter) were significantly higher than those in larger voxel size datasets (2 meters), highlighting the difficulty in accurately estimating within-canopy voxel content at fine resolutions. This suggests that the choice of voxel size is application-dependent. Our work fills the gap in deep imbalance learning models for multi-target regression and simulated datasets for 3D LiDAR point clouds of forests.

---

## 44. LMM-IR: Large-Scale Netlist-Aware Multimodal Framework for Static IR-Drop Prediction

**论文链接:** [http://arxiv.org/abs/2511.12581v1](http://arxiv.org/abs/2511.12581v1)

**作者:** Kai Ma, Zhen Wang, Hongquan He, Qi Xu, Tinghuan Chen, Hao Geng

**发布时间:** 2025-11-16

**DOI:** 10.1109/DAC63849.2025.11133205

### GPT解析

### 总结

本文提出了一种新颖的多模态方法，通过大规模网表变换器(LNT)高效处理SPICE文件，将网表拓扑表示为3D点云，实现快速准确的静态IR压降预测，在ICCAD 2023竞赛中取得最佳成绩。

### 背景

静态IR压降分析是芯片设计领域的基础和关键任务，但该过程非常耗时，可能需要数小时，且解决IR压降违规通常需要迭代分析，增加了计算负担。

### 目的

开发快速准确的IR压降预测方法，以减少芯片设计的总体时间投入。

### 方法

提出多模态方法，使用大规模网表变换器(LNT)处理SPICE文件，将网表拓扑表示为3D点云，能处理数十万到数百万节点的网表，将所有数据编码为潜在空间特征并输入模型进行预测，实现多模态数据集成互补预测。

### 主要发现

实验结果表明，所提算法在ICCAD 2023竞赛获胜团队和最先进算法中取得了最佳F1分数和最低MAE。

### 结论

该多模态方法在静态IR压降预测方面具有优越性能，能有效提高芯片设计效率。

### 翻译

静态IR压降分析是芯片设计领域的一个基础和关键任务。然而，这个过程可能相当耗时，可能需要几个小时。此外，解决IR压降违规通常需要迭代分析，从而造成计算负担。因此，快速准确的IR压降预测对于减少芯片设计所花费的总时间至关重要。在本文中，我们首先提出了一种新颖的多模态方法，通过大规模网表变换器(LNT)高效处理SPICE文件。我们的关键创新是将网表拓扑表示和处理为3D点云表示，能够高效处理多达数十万到数百万节点的网表。所有类型的数据，包括网表文件和图像数据，都被编码为潜在空间中的特征，并输入模型进行静态电压压降预测。这实现了多模态数据的集成，用于互补预测。实验结果表明，我们提出的算法在ICCAD 2023竞赛的获胜团队和最先进算法中取得了最佳F1分数和最低MAE。


### 论文摘要

Static IR drop analysis is a fundamental and critical task in the field of chip design. Nevertheless, this process can be quite time-consuming, potentially requiring several hours. Moreover, addressing IR drop violations frequently demands iterative analysis, thereby causing the computational burden. Therefore, fast and accurate IR drop prediction is vital for reducing the overall time invested in chip design. In this paper, we firstly propose a novel multimodal approach that efficiently processes SPICE files through large-scale netlist transformer (LNT). Our key innovation is representing and processing netlist topology as 3D point cloud representations, enabling efficient handling of netlist with up to hundreds of thousands to millions nodes. All types of data, including netlist files and image data, are encoded into latent space as features and fed into the model for static voltage drop prediction. This enables the integration of data from multiple modalities for complementary predictions. Experimental results demonstrate that our proposed algorithm can achieve the best F1 score and the lowest MAE among the winning teams of the ICCAD 2023 contest and the state-of-the-art algorithms.

---

## 45. LiDAR Accuracy on North American Mountain Summits

**论文链接:** [http://arxiv.org/abs/2511.12341v1](http://arxiv.org/abs/2511.12341v1)

**作者:** Eric Gilbertson, Richard Hensley, Andrew Kirmse, Kyle Bretherton, Kathryn Stanchak

**发布时间:** 2025-11-15

### GPT解析

### 总结

本研究评估了基于飞机的激光雷达(LiDAR)技术在测量山地地形准确性方面的表现，通过比较179个山顶的差分GNSS静态测量结果与LiDAR点云数据，发现不同处理方法的LiDAR测量存在不同程度的误差。

### 背景

山地地形越来越多地使用基于飞机的激光雷达(LiDAR)技术进行测量和制图，但在这种地形变化多样的区域，这些测量的准确性尚未得到充分理解。

### 目的

评估LiDAR技术在测量山地顶峰高程和位置方面的准确性，通过与差分GNSS静态测量结果进行比较。

### 方法

测量了179个山顶，使用差分GNSS静态测量作为参考；测量范围涵盖美国13个州和加拿大2个省的山顶；山顶类型包括冰盖山峰、开放岩石山峰和树木覆盖的山峰，海拔从490米到4000多米不等；使用四种不同的方法计算LiDAR点云衍生的山顶高程和位置：手动处理、最高地面返回、最高返回和Lastools重新分类。

### 主要发现

四种方法的平均一西格玛LiDAR垂直误差分别为0.50米、1.09米、9.83米和1.96米；平均一西格玛水平误差分别为3.03米、2.41米、5.17米和3.78米；误差来源包括尖锐山顶未被采样、茂密植被被误分类为地面、人造结构被误分类为地面、可能随时间融化的雪/冰，以及随时间推移的山顶侵蚀。

### 结论

LiDAR技术在测量山地地形时存在不同程度的误差，不同处理方法的准确性差异显著，地形特征和地表覆盖类型会影响LiDAR测量的准确性。

### 翻译

山地地形正越来越多地通过基于飞机的激光雷达(LiDAR)技术进行测量和制图，但这些测量在地形变化多样的区域中的准确性尚未得到充分理解。在本研究中，我们使用差分GNSS静态测量测量了179个山顶，并将山顶高程和位置测量与LiDAR在点云数据集中测量的结果进行比较。我们在美国13个州(华盛顿州、爱达荷州、蒙大拿州、犹他州、加利福尼亚州、内华达州、亚利桑那州、新墨西哥州、密歇根州、威斯康星州、肯塔基州、科罗拉多州和宾夕法尼亚州)和加拿大两个省(不列颠哥伦比亚省和新斯科舍省)测量了山顶。山顶包括冰盖山峰、开放岩石山峰和树木覆盖的山峰，海拔从490米到4000多米不等。使用四种不同的方法计算了从LiDAR点云衍生的山顶高程和位置：手动处理、最高地面返回、最高返回和Lastools重新分类。每种方法的平均一西格玛LiDAR垂直误差分别为0.50米、1.09米、9.83米和1.96米。平均一西格玛水平误差分别为3.03米、2.41米、5.17米和3.78米。还按每种山顶类型分别呈现了误差。误差来源包括尖锐山顶未被采样、茂密植被被误分类为地面、人造结构被误分类为地面、可能随时间融化的雪/冰，以及随时间推移的山顶侵蚀。


### 论文摘要

Mountainous terrain is increasingly being measured and mapped by airplane-based LiDAR (Light Detection and Ranging) techniques, but the accuracy of these measurements in such topographically variable terrain is not well understood. For this study we measured 179 mountain summits with differential GNSS static surveys and compared summit elevation and location measurements to those measured by LiDAR in point cloud data sets. We measured summits in 13 US states (Washington, Idaho, Montana, Utah, California, Nevada, Arizona, New Mexico, Michigan, Wisconsin, Kentucky, Colorado, and Pennsylvania) and two Canadian provinces (British Columbia and Nova Scotia). Summits included icecapped peaks, open rocky peaks, and tree-covered peaks ranging in elevation from 490m to over 4000m. LiDAR-point-cloud-derived summit elevations and locations were computed using four different methods: manual processing, highest ground return, highest return, and Lastools reclassification. The average one-sigma LiDAR vertical errors for each method were 0.50m, 1.09m, 9.83m, and 1.96m, respectively. Average one-sigma horizontal errors were 3.03m, 2.41m, 5.17m, 3.78m, respectively. Errors are also presented separately for each type of summit. Error sources include sharp summits being unsampled, dense vegetation misclassified as ground, human-created structures misclassified as ground, snow/ice that can melt over time, and summit erosion over time.

---

## 46. RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.12117v1](http://arxiv.org/abs/2511.12117v1)

**作者:** Ruiqi Cheng, Huijun Di, Jian Li, Feng Liu, Wei Liang

**发布时间:** 2025-11-15

**备注:** 12 pages, 6 figures. Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了RadarMP，一种用于精确3D场景运动感知的新方法，通过统一建模雷达目标检测和运动估计，结合专门设计的自监督损失函数，实现了在各种天气和光照条件下的可靠运动感知，增强了全场景自动驾驶系统的感知能力。

### 背景

准确的3D场景运动感知对提高自动驾驶系统的安全性和可靠性至关重要。4D毫米波雷达因其全天候工作能力和独特感知特性已成为高级自动驾驶的关键组件，但稀疏和有噪声的雷达点导致不精确的运动感知，特别是在恶劣天气条件下光学传感器性能下降时，自动驾驶车辆的感知能力受到限制。

### 目的

解决稀疏和噪声雷达点导致的不精确运动感知问题，提高自动驾驶系统在各种环境条件下的感知能力，特别是在恶劣天气条件下。

### 方法

提出RadarMP方法，使用连续两帧的低级雷达回波信号进行3D场景运动感知。该方法将雷达目标检测和运动估计统一在一个架构中，实现一致的雷达点云生成和逐点3D场景流预测。针对雷达特性设计了专门的自监督损失函数，由多普勒频移和回波强度引导，有效监督空间和运动一致性，无需显式标注。

### 主要发现

在公共数据集上的大量实验表明，RadarMP能够在各种天气和光照条件下实现可靠的运动感知，性能优于基于雷达的解耦运动感知管道，显著增强了全场景自动驾驶系统的感知能力。

### 结论

RadarMP通过统一建模和专门设计的自监督学习有效解决了雷达点云稀疏和噪声问题，提高了自动驾驶系统在恶劣条件下的感知能力，为全场景自动驾驶的安全性和可靠性提供了重要保障。

### 翻译

准确的3D场景运动感知显著提高了自动驾驶系统的安全性和可靠性。受益于其全天候工作能力和独特的感知特性，4D毫米波雷达已成为高级自动驾驶的重要组成部分。然而，稀疏和有噪声的雷达点通常导致不精确的运动感知，使自动驾驶车辆在光学传感器在恶劣天气条件下性能下降时，感知能力有限。在本文中，我们提出了RadarMP，一种使用连续两帧低级雷达回波信号进行精确3D场景运动感知的新方法。与现有的分离雷达目标检测和运动估计的方法不同，RadarMP在统一架构中联合建模这两个任务，实现了一致的雷达点云生成和逐点3D场景流预测。针对雷达特性，我们设计了专门的自监督损失函数，由多普勒频移和回波强度引导，有效监督空间和运动一致性，无需显式标注。公共数据集上的大量实验表明，RadarMP能够在各种天气和光照条件下实现可靠的运动感知，优于基于雷达的解耦运动感知管道，增强了全场景自动驾驶系统的感知能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D毫米波雷达在自动驾驶中的运动感知问题。由于雷达点云的稀疏性和噪声常常导致不精确的运动感知，当光学传感器在恶劣天气条件下性能下降时，自动驾驶系统的感知能力会受到严重影响。准确的运动感知对提升自动驾驶系统的安全性和可靠性至关重要，特别是在全天候运行场景中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法将雷达目标检测和运动估计分开处理，导致不一致性。基于目标点在相邻雷达帧之间的能量流应与运动场一致而噪声点能量流无序的假设，作者设计了统一架构。他们借鉴了雷达tesseract表示、多尺度可变形交叉注意力机制、PWC-Net架构预测运动场、特征金字塔网络以及Transformer自注意力机制等现有工作，但进行了专门改进以适应雷达特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是联合建模雷达目标检测和场景流估计，利用能量流与运动场的一致性区分目标和噪声。整体流程包括：1) 输入两个连续雷达tesseract；2) 多普勒通道编码；3) 通过2D投影和PWC-Net预测获得参考点；4) 使用多尺度可变形交叉注意力提取相关特征；5) 通过全局块自注意力和方向切片自注意力捕获全局运动模式；6) 解码输出分割掩码和场景流；7) 使用三种自监督损失函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个联合解决雷达目标检测和场景流估计的统一架构；2) 基于雷达多普勒特性和回波强度的专门自监督损失函数；3) 全局运动模式感知模块设计。与传统CFAR方法不同，RadarMP不依赖统计假设；与使用LiDAR监督的深度学习方法不同，RadarMP保持雷达感知独立性；与其他分离处理检测和运动估计的方法不同，RadarMP实现端到端的联合优化，在恶劣天气条件下表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RadarMP通过联合建模雷达目标检测和场景流估计，并利用专门设计的自监督损失函数，显著提升了4D毫米波雷达在自动驾驶全场景下的运动感知能力，特别是在光学传感器性能下降的恶劣天气条件下。'}


### 论文摘要

Accurate 3D scene motion perception significantly enhances the safety and reliability of an autonomous driving system. Benefiting from its all-weather operational capability and unique perceptual properties, 4D mmWave radar has emerged as an essential component in advanced autonomous driving. However, sparse and noisy radar points often lead to imprecise motion perception, leaving autonomous vehicles with limited sensing capabilities when optical sensors degrade under adverse weather conditions. In this paper, we propose RadarMP, a novel method for precise 3D scene motion perception using low-level radar echo signals from two consecutive frames. Unlike existing methods that separate radar target detection and motion estimation, RadarMP jointly models both tasks in a unified architecture, enabling consistent radar point cloud generation and pointwise 3D scene flow prediction. Tailored to radar characteristics, we design specialized self-supervised loss functions guided by Doppler shifts and echo intensity, effectively supervising spatial and motion consistency without explicit annotations. Extensive experiments on the public dataset demonstrate that RadarMP achieves reliable motion perception across diverse weather and illumination conditions, outperforming radar-based decoupled motion perception pipelines and enhancing perception capabilities for full-scenario autonomous driving systems.

---

## 47. Point Cloud Quantization through Multimodal Prompting for 3D Understanding

**论文链接:** [http://arxiv.org/abs/2511.12079v1](http://arxiv.org/abs/2511.12079v1)

**作者:** Hongxuan Li, Wencheng Zhu, Huiying Xu, Xinzhong Zhu, Pengfei Zhu

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026. 11 pages, 7 figures. Corresponding author: Wencheng Zhu (wenchengzhu@tju.edu.cn)

### GPT解析

### 总结

论文提出了一种多模态提示驱动的点云分析量化框架，通过文本嵌入作为原型先验和多模态提示来改进原型，有效解决了传统向量量化中代表性和可解释性不足的问题。

### 背景

向量量化已成为大规模多模态模型的有力工具，通过离散令牌编码统一异构表示。然而当前基于原型的方法(可训练向量或聚类质心)在代表性和可解释性方面存在不足，尽管多模态对齐在视觉语言模型中显示出潜力。

### 目的

解决当前原型方法在代表性和可解释性方面的不足，提出一个简单的多模态提示驱动的点云分析量化框架。

### 方法

基于两个核心见解：1)预训练模型的文本嵌入通过多对一对比对齐自然编码视觉语义，可作为强大的原型先验；2)多模态提示能够自适应地改进这些原型，有效缓解视觉语言语义差距。引入双约束量化空间，由紧凑性和分离正则化强制执行，并使用Gumbel-Softmax松弛实现可微离散化。

### 主要发现

在ModelNet40和ScanObjectNN数据集上的广泛实验证明了所提出方法的优越有效性，产生的混合表示能够同时编码几何和语义信息。

### 结论

所提出的方法通过结合视觉和原型特征，成功创建了能够共同编码几何和语义信息的混合表示，在点云分析任务中表现出色。

### 翻译

向量量化已成为大规模多模态模型中的一个强大工具，通过离散令牌编码统一异构表示。然而，其有效性依赖于强大的码本设计。当前基于原型的方法依赖于可训练向量或聚类质心，在代表性和可解释性方面表现不足，即使多模态对齐在视觉语言模型中显示出其潜力。为了解决这些局限性，我们提出了一种用于点云分析的多模态提示驱动的简单量化框架。我们的方法基于两个核心见解：1)预训练模型的文本嵌入通过多对一对比对齐自然编码视觉语义，自然地作为强大的原型先验；2)多模态提示能够自适应地改进这些原型，有效缓解视觉语言语义差距。该框架引入了由紧凑性和分离正则化强制执行的双约束量化空间，无缝集成视觉和原型特征，产生共同编码几何和语义信息的混合表示。此外，我们采用Gumbel-Softmax松弛来实现可微离散化，同时保持量化稀疏性。在ModelNet40和ScanObjectNN数据集上的广泛实验清楚地证明了所提出方法的优越有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云数据量化表示中代表性和可解释性不足的问题，特别是在多模态对齐方面。这个问题很重要，因为3D理解在自动驾驶、机器人和增强现实等领域有广泛应用，而点云是重要的3D数据表示形式。通过结合文本语义信息，可以增强模型的理解能力和泛化能力，弥合视觉和语言之间的语义鸿沟，实现更准确的3D场景理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于两个核心观察设计方法：1)预训练模型中的文本嵌入通过多对一对比对齐自然编码了视觉语义，可作为强大原型先验；2)多模态提示可有效优化这些原型，缓解视觉-语言语义鸿沟。作者借鉴了向量量化技术(如VQ-VAE)、视觉语言模型(如CLIP、ULIP)、提示学习和原型学习等现有工作，创新性地将文本嵌入重新定义为可训练的视觉原型，并通过Gumbel-Softmax松弛实现可微分的离散化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练视觉语言模型中的文本嵌入作为语义原型，通过多模态提示驱动的量化框架，将点云特征离散化为与文本语义对齐的表示，从而增强3D理解能力。整体流程包括：1)特征提取(使用ULIP-2的文本和3D编码器)；2)自适应提示调优(引入可学习提示向量并优化)；3)原型引导的可微分量化(使用Gumbel-Softmax将点云特征量化为原型特征)；4)跨模态特征融合(通过交叉注意力结合原始和量化特征)；5)多目标优化(对比损失、紧凑性损失和分离损失)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)文本驱动的3D量化框架，将文本特征重新定义为视觉原型；2)可微分的原型量化，使用Gumbel-Softmax实现端到端训练；3)双约束量化空间，通过紧凑性和分离正则化集成视觉和原型特征；4)参数高效的微调策略。相比之前工作，该方法不依赖训练数据分布，能捕获类内多样性，减少域偏移，提供更好可解释性，实现更稳定收敛，并通过文本引导增强了语义表达能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过多模态提示驱动的点云量化方法，利用文本嵌入作为语义原型，有效弥合了视觉和语言之间的语义鸿沟，显著提升了3D理解任务的性能和泛化能力。'}


### 论文摘要

Vector quantization has emerged as a powerful tool in large-scale multimodal models, unifying heterogeneous representations through discrete token encoding. However, its effectiveness hinges on robust codebook design. Current prototype-based approaches relying on trainable vectors or clustered centroids fall short in representativeness and interpretability, even as multimodal alignment demonstrates its promise in vision-language models. To address these limitations, we propose a simple multimodal prompting-driven quantization framework for point cloud analysis. Our methodology is built upon two core insights: 1) Text embeddings from pre-trained models inherently encode visual semantics through many-to-one contrastive alignment, naturally serving as robust prototype priors; and 2) Multimodal prompts enable adaptive refinement of these prototypes, effectively mitigating vision-language semantic gaps. The framework introduces a dual-constrained quantization space, enforced by compactness and separation regularization, which seamlessly integrates visual and prototype features, resulting in hybrid representations that jointly encode geometric and semantic information. Furthermore, we employ Gumbel-Softmax relaxation to achieve differentiable discretization while maintaining quantization sparsity. Extensive experiments on the ModelNet40 and ScanObjectNN datasets clearly demonstrate the superior effectiveness of the proposed method.

---

## 48. Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding

**论文链接:** [http://arxiv.org/abs/2511.12031v1](http://arxiv.org/abs/2511.12031v1)

**作者:** Arun Ramachandran, Ramaswamy Govindarajan, Murali Annavaram, Prakash Raghavendra, Hossein Entezari Zarch, Lei Gao, Chaoyi Jiang

**发布时间:** 2025-11-15

### GPT解析

### 总结

论文提出了一种名为BMC（平衡内存和计算）的KV缓存分配机制，用于优化大语言模型在CPU上的推理性能，结合推测解码技术进一步加速，在多种硬件平台上表现出色。

### 背景

随着GPU及其云虚拟实例成本急剧上升，人们希望使用CPU进行大语言模型推理。然而，KV缓存更新（通常为每个生成令牌分配、复制和就地更新）带来显著开销，随着序列长度增加，分配和复制开销成为性能瓶颈。其他方法虽可预先分配大型KV张量实现就地更新，但带有零填充行的矩阵会导致冗余计算。

### 目的

开发一种新的KV缓存分配机制，减少CPU上大语言模型推理的开销，平衡内存和计算资源，提高推理效率。

### 方法

提出BMC机制，每隔r次迭代分配带有r个冗余行的KV张量，实现无复制开销的就地更新，同时引入少量冗余计算。额外分配的行和冗余计算可重新用于推测解码（SD）提高效率。通过简单分析模型确定最佳设计点。

### 主要发现

BMC相比基线HuggingFace实现平均吞吐量加速高达3.2倍；结合SD后额外获得高达1.39倍加速；相比vLLM和DeepSpeed分别实现1.36倍和2.29倍吞吐量加速；在多种CPU类别和GPU上均表现良好。

### 结论

BMC方法通过创新的KV缓存分配机制有效平衡内存和计算开销，显著提高大语言模型在CPU和GPU上的推理性能，结合推测解码技术可进一步提升效率，为降低大语言模型推理成本提供了有效解决方案。

### 翻译

随着GPU及其云虚拟实例成本的急剧上升，人们强烈希望使用CPU进行大语言模型推理。KV缓存更新通常为每个生成的令牌实现为分配、复制和就地更新，带来显著开销。随着序列长度增加，分配和复制开销主导性能。其他方法可能预先分配大型KV张体以实现就地更新，但这些带有零填充行的矩阵会导致冗余计算。在这项工作中，我们提出了一种名为平衡内存和计算的新KV缓存分配机制。BMC每隔r次迭代分配带有r个冗余行的KV张量，允许在这些迭代中进行无复制开销的就地更新，但代价是少量冗余计算。其次，我们观察到KV张量中分配的额外行和由此产生的冗余计算可以重新用于推测解码，提高令牌生成效率。最后，BMC代表了一系列具有不同r值的设计点。我们为BMC推导了一个简单的分析模型来确定最佳设计点。所提出的BMC方法相比基线HuggingFace实现了平均吞吐量加速高达3.2倍。重要的是，当我们应用BMC结合SD时，在SD提供的加速基础上额外获得了高达1.39倍的加速。此外，BMC相比最先进的推理服务器vLLM和DeepSpeed分别实现了高达1.36倍和2.29倍的吞吐量加速。尽管BMC技术已在不同类别的CPU上进行了广泛评估，但我们也在GPU上评估了该方案，证明它对GPU也有效。


### 论文摘要

With the skyrocketing costs of GPUs and their virtual instances in the cloud, there is a significant desire to use CPUs for large language model (LLM) inference. KV cache update, often implemented as allocation, copying, and in-place strided update for each generated token, incurs significant overhead. As the sequence length increases, the allocation and copy overheads dominate the performance. Alternate approaches may allocate large KV tensors upfront to enable in-place updates, but these matrices (with zero-padded rows) cause redundant computations. In this work, we propose a new KV cache allocation mechanism called Balancing Memory and Compute (BMC). BMC allocates, once every r iterations, KV tensors with r redundant rows, allowing in-place update without copy overhead for those iterations, but at the expense of a small amount of redundant computation. Second, we make an interesting observation that the extra rows allocated in the KV tensors and the resulting redundant computation can be repurposed for Speculative Decoding (SD) that improves token generation efficiency. Last, BMC represents a spectrum of design points with different values of r. To identify the best-performing design point(s), we derive a simple analytical model for BMC. The proposed BMC method achieves an average throughput acceleration of up to 3.2x over baseline HuggingFace (without SD). Importantly when we apply BMC with SD, it results in an additional speedup of up to 1.39x, over and above the speedup offered by SD. Further, BMC achieves a throughput acceleration of up to 1.36x and 2.29x over state-of-the-art inference servers vLLM and DeepSpeed, respectively. Although the BMC technique is evaluated extensively across different classes of CPUs (desktop and server class), we also evaluate the scheme with GPUs and demonstrate that it works well for GPUs.

---

## 49. Toward bilipshiz geometric models

**论文链接:** [http://arxiv.org/abs/2511.11735v1](http://arxiv.org/abs/2511.11735v1)

**作者:** Yonatan Sverdlov, Eitan Rosen, Nadav Dym

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文研究了点云神经网络是否通过双利普希茨等价的概念保留了点云空间上的自然对称感知距离，发现流行的点云不变网络相对于PM度量不是双利普希茨的，并提出了修改方法，实验证明其在寻找3D点云对应关系方面优于标准模型。

### 背景

许多点云神经网络在设计上对排列和刚体运动等对称性是不变的。

### 目的

研究这些网络是否通过双利普希茨等价的概念，保留了点云空间上的自然对称感知距离。

### 方法

考虑点云上的两种对称感知度量：(a) Procrustes Matching (PM) 度量和 (b) Hard Gromov-Wasserstein 距离，研究这些距离本身的等价性以及网络对这些度量的保持性。

### 主要发现

这两种距离本身不是双利普希茨等价的，由此推断流行的点云不变网络相对于PM度量不是双利普希茨的；作者展示了如何修改这些网络以获得双利普希茨保证。

### 结论

所提出的双利普希茨模型在寻找3D点云之间的对应关系的任务上优于标准不变模型。

### 翻译

许多点云神经网络在设计上对这种数据类型的对称性（排列和刚体运动）是不变的。本文的目的是通过双利普希茨等价的概念，研究这些网络是否保留了点云空间上的自然对称感知距离。这一研究受到等变性学习领域近期工作的启发，该工作强调了双利普希茨模型在其他场景中的优势。我们考虑了点云上的两种对称感知度量：(a) Procrustes Matching (PM) 度量和 (b) Hard Gromov-Wasserstein 距离。我们证明这两种距离本身不是双利普希茨等价的，并由此推断出流行的点云不变网络相对于PM度量不是双利普希茨的。然后我们展示了如何修改这些网络，使它们能够获得双利普希茨保证。最后，我们提供了初步实验，证明在寻找3D点云之间的对应关系的任务中，所提出的双利普希茨模型优于标准不变模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究点云神经网络是否能够保持点云空间上的自然对称感知距离。这个问题很重要，因为现有的点云神经网络虽然被设计为对排列和刚体运动不变，但它们是否能保持点云之间的距离特性尚不清楚。双利普希兹模型能为基于度量的学习提供可证明保证，对点云处理、粒子动力学学习和化学信息学等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先考虑了两种对称感知度量（Procrustes Matching和Hard Gromov-Wasserstein距离），发现它们不是双利普希兹等价的，推断现有网络无法同时保持这两种距离。然后借鉴了1-WL图神经网络、Hordan等人的k-GNN工作、Balan等人的多集嵌入以及Amir & Dym的FSW嵌入等方法，设计了一种新的几何模型结构，通过引入垂直向量和广义叉积来构建更丰富的特征表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计能保持双利普希兹性质的几何模型，确保点云间距离在神经网络映射后得到保持。实现流程：1)对点云中心化；2)对每个点计算其垂直向量；3)构建包含点积、垂直向量点积和范数的特征；4)使用双利普希兹且齐次的函数处理特征；5)用双利普希兹读出函数得到最终嵌入。高维点云使用(d-1)-元组和广义叉积构建更复杂特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次系统研究点云神经网络的双利普希兹性质；2)证明两种常用对称感知度量不是双利普希兹等价；3)提出新的几何模型结构保证双利普希兹性质；4)设计基于垂直向量和广义叉积的特征表示方法；5)证明模型在特定条件下是双利普希兹的。不同之处：之前工作主要关注完整性，本文关注更严格的双利普希兹性质；传统1-WL几何模型虽完整但非双利普希兹；本文方法更复杂但提供更好的距离保持特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种新的双利普希兹几何模型，能够保持点云之间的距离特性，解决了传统点云神经网络虽然对称不变但无法保持距离的问题，并在点云匹配任务中展示了优越性能。'}


### 论文摘要

Many neural networks for point clouds are, by design, invariant to the symmetries of this datatype: permutations and rigid motions. The purpose of this paper is to examine whether such networks preserve natural symmetry aware distances on the point cloud spaces, through the notion of bi-Lipschitz equivalence. This inquiry is motivated by recent work in the Equivariant learning literature which highlights the advantages of bi-Lipschitz models in other scenarios.   We consider two symmetry aware metrics on point clouds: (a) The Procrustes Matching (PM) metric and (b) Hard Gromov Wasserstien distances. We show that these two distances themselves are not bi-Lipschitz equivalent, and as a corollary deduce that popular invariant networks for point clouds are not bi-Lipschitz with respect to the PM metric. We then show how these networks can be modified so that they do obtain bi-Lipschitz guarantees. Finally, we provide initial experiments showing the advantage of the proposed bi-Lipschitz model over standard invariant models, for the tasks of finding correspondences between 3D point clouds.

---

## 50. Rare Genomic Subtype Discovery from RNA-seq via Autoencoder Embeddings and Stability-Aware Clustering

**论文链接:** [http://arxiv.org/abs/2511.13705v1](http://arxiv.org/abs/2511.13705v1)

**作者:** Alaa Mezghiche

**发布时间:** 2025-11-17

**备注:** 16 pages

### GPT解析

### 总结

该研究提出了一种在高维RNA-seq数据上进行无监督学习的方法，结合自编码器表示、聚类和稳定性分析，能够发现罕见但可重现的基因组亚型。

### 背景

在UCI基因表达癌症RNA-Seq数据集（801个样本，20,531个基因，包含BRCA、COAD、KIRC、LUAD、PRAD等癌症类型）上的泛癌症分析显示，聚类结果与组织来源几乎完美一致（Cramer's V = 0.887），这被视为负对照。

### 目的

重新定义问题，专注于KIRC（肾透明细胞癌，n = 146），寻找罕见但稳定的分子亚型。

### 方法

选择前2,000个高变基因进行标准化，训练具有128维潜在空间的前馈自编码器，运行k-means聚类（k=2-10），使用预定义发现规则（稀有<10%且在20种初始化后匈牙利对齐后的Jaccard相似度≥0.60）扫描k值，并进行聚类vs剩余差异表达分析。

### 主要发现

全局指数倾向于较小的k值，使用预定义规则得到k=5的解（轮廓系数=0.129，DBI=2.045），发现了一个稀有簇C0（占患者的6.85%），具有高度稳定性（Jaccard=0.787），并识别出一致的差异表达标记物。

### 结论

泛癌症聚类主要由组织来源主导，而具有稳定性意识的癌症内方法揭示了罕见的、可重现的KIRC亚型。

### 翻译

在高维RNA-seq数据上进行无监督学习可以揭示标准标签之外的分子亚型。我们将基于自编码器的表示与聚类和稳定性分析相结合，以寻找罕见但可重现的基因组亚型。在UCI'基因表达癌症RNA-Seq'数据集（801个样本，20,531个基因；BRCA、COAD、KIRC、LUAD、PRAD）上，泛癌症分析显示聚类结果与组织来源几乎完美一致（Cramer's V = 0.887），作为负对照。因此，我们在KIRC（n = 146）中重新定义问题：我们选择前2,000个高变基因，对其进行标准化，训练前馈自编码器（128维潜在空间），并运行k-means聚类（k=2-10）。虽然全局指数倾向于较小的k值，但使用预定义的发现规则（稀有<10%且在20种初始化后匈牙利对齐后的Jaccard相似度≥0.60）扫描k值，在k=5时得到一个简单解（轮廓系数=0.129，DBI=2.045），包含一个稀有簇C0（占患者的6.85%），该簇具有高度稳定性（Jaccard=0.787）。聚类vs剩余差异表达分析（Welch's t检验，Benjamini-Hochberg FDR）识别出一致的标记物。总体而言，泛癌症聚类主要由组织来源主导，而具有稳定性意识的癌症内方法揭示了罕见的、可重现的KIRC亚型。


### 论文摘要

Unsupervised learning on high-dimensional RNA-seq data can reveal molecular subtypes beyond standard labels. We combine an autoencoder-based representation with clustering and stability analysis to search for rare but reproducible genomic subtypes. On the UCI "Gene Expression Cancer RNA-Seq" dataset (801 samples, 20,531 genes; BRCA, COAD, KIRC, LUAD, PRAD), a pan-cancer analysis shows clusters aligning almost perfectly with tissue of origin (Cramer's V = 0.887), serving as a negative control. We therefore reframe the problem within KIRC (n = 146): we select the top 2,000 highly variable genes, standardize them, train a feed-forward autoencoder (128-dimensional latent space), and run k-means for k = 2-10. While global indices favor small k, scanning k with a pre-specified discovery rule (rare < 10 percent and stable with Jaccard >= 0.60 across 20 seeds after Hungarian alignment) yields a simple solution at k = 5 (silhouette = 0.129, DBI = 2.045) with a rare cluster C0 (6.85 percent of patients) that is highly stable (Jaccard = 0.787). Cluster-vs-rest differential expression (Welch's t-test, Benjamini-Hochberg FDR) identifies coherent markers. Overall, pan-cancer clustering is dominated by tissue of origin, whereas a stability-aware within-cancer approach reveals a rare, reproducible KIRC subtype.

---

## 51. Scientific Data Compression and Super-Resolution Sampling

**论文链接:** [http://arxiv.org/abs/2511.13675v1](http://arxiv.org/abs/2511.13675v1)

**作者:** Minh Vu, Andrey Lokhov

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究介绍了一种基于指数族学习进展的科学数据压缩和超分辨率新框架，能够在压缩数据的同时保留物理特性并量化不确定性，支持压缩比和重建质量间的灵活权衡。

### 背景

现代科学模拟、观测和大规模实验产生的数据量常超出存储、处理和分析能力，需要高效的数据缩减方法。

### 目的

开发能够管理大规模数据集同时保留关键物理特征的数据压缩方法，并支持从压缩表示中恢复数据（超分辨率）。

### 方法

基于指数族学习进展构建的科学数据压缩和超分辨率新框架，能够保留和量化物理量中的不确定性。

### 主要发现

该方法支持压缩比和重建保真度之间的灵活权衡，并在检查点和重启等场景中具有应用价值。

### 结论

该框架为科学数据压缩和超分辨率提供了新方法，能够在数据缩减的同时保留重要的物理特性和不确定性信息。

### 翻译

现代科学模拟、观测和大规模实验产生的数据量常常超过了存储、处理和分析的极限。这一挑战推动了数据缩减方法的发展，这些方法能够高效管理大型数据集，同时保留重要的物理特征和感兴趣的量。在许多科学工作流程中，能够从压缩表示中恢复数据（称为超分辨率）也很重要，并且需要对关键物理特性的保留有保证。一个显著的例子是检查点和重启，这对长期模拟从故障中恢复、在中断后继续或检查中间结果至关重要。在这项工作中，我们介绍了一种用于科学数据压缩和超分辨率的新框架，基于最近在指数族学习方面的进展。我们的方法保留了感兴趣的物理量中的不确定性并对其进行量化，同时支持压缩比和重建保真度之间的灵活权衡。


### 论文摘要

Modern scientific simulations, observations, and large-scale experiments generate data at volumes that often exceed the limits of storage, processing, and analysis. This challenge drives the development of data reduction methods that efficiently manage massive datasets while preserving essential physical features and quantities of interest. In many scientific workflows, it is also crucial to enable data recovery from compressed representations - a task known as super-resolution - with guarantees on the preservation of key physical characteristics. A notable example is checkpointing and restarting, which is essential for long-running simulations to recover from failures, resume after interruptions, or examine intermediate results. In this work, we introduce a novel framework for scientific data compression and super-resolution, grounded in recent advances in learning exponential families. Our method preserves and quantifies uncertainty in physical quantities of interest and supports flexible trade-offs between compression ratio and reconstruction fidelity.

---

## 52. PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image

**论文链接:** [http://arxiv.org/abs/2511.13648v1](http://arxiv.org/abs/2511.13648v1)

**作者:** Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu

**发布时间:** 2025-11-17

**备注:** Project page: https://physx-anything.github.io/

### GPT解析

### 总结

PhysX-Anything是首个模拟就绪的物理3D生成框架，能从单张野外图像生成高质量3D资产，具有明确几何形状、关节连接和物理属性，并通过新3D表示法显著减少标记数量，同时构建了更丰富的PhysX-Mobility数据集。

### 背景

3D建模正从静态视觉表现转向物理化的、可关节连接的资产，但现有3D生成方法忽视了关键的物理和关节连接属性，限制了它们在具身AI中的应用。

### 目的

弥合3D生成与物理模拟之间的差距，创建能够生成可直接用于模拟和交互的物理化3D资产的框架。

### 方法

提出首个基于视觉语言模型(VLM)的物理3D生成模型；开发新的3D表示法减少193倍标记数量；构建包含2000+个对象的PhysX-Mobility数据集，扩展了先前数据集类别2倍以上。

### 主要发现

PhysX-Anything在PhysX-Mobility和野外图像上表现出强大的生成性能和泛化能力；模拟验证显示生成的3D资产可直接用于接触密集型机器人策略学习。

### 结论

PhysX-Anything能够显著赋能广泛的下游应用，特别是在具身AI和基于物理的模拟领域。

### 翻译

3D建模正从静态视觉表现转向物理化的、可关节连接的资产，这些资产可以直接用于模拟和交互。然而，大多数现有的3D生成方法忽视了关键的物理和关节连接属性，限制了它们在具身AI中的应用。为了弥合这一差距，我们引入了PhysX-Anything，这是第一个模拟就绪的物理3D生成框架，它能在给定单张野外图像的情况下，生成具有明确几何形状、关节连接和物理属性的高质量模拟就绪3D资产。具体来说，我们提出了首个基于VLM的物理3D生成模型，以及一种能有效标记化几何形状的新3D表示法。它将标记数量减少了193倍，使标准VLM标记预算内能够学习明确的几何形状，且在微调过程中不需要引入任何特殊标记，显著提高了生成质量。此外，为了克服现有物理3D数据集的多样性有限问题，我们构建了一个新的数据集PhysX-Mobility，它将先前物理3D数据集中的对象类别扩展了2倍以上，并包括超过2K个具有丰富物理注释的常见真实世界对象。在PhysX-Mobility和野外图像上的大量实验表明，PhysX-Anything提供了强大的生成性能和强大的泛化能力。此外，在MuJoCo风格环境中的基于模拟的实验验证了我们的模拟就绪资产可以直接用于接触密集型机器人策略学习。我们相信PhysX-Anything能够显著赋能广泛的下游应用，特别是在具身AI和基于物理的模拟方面。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D生成方法缺乏物理和关节化信息的问题。现有方法生成的3D资产通常缺少密度、绝对尺度和关节约束等关键物理属性，使其难以直接部署在模拟器或物理引擎中。这个问题很重要，因为随着机器人、具身AI和交互模拟等应用的发展，对可直接在模拟器中执行的高质量物理3D资产需求日益增长，填补这一差距能推动3D内容创作向物理真实感方向发展，并为下游应用解锁新可能性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有3D生成方法要么专注全局几何外观，要么关注部件结构，但都忽视物理属性；一些关节化生成方法因缺乏大规模高质量数据集而采用检索而非合成策略。作者借鉴了视觉语言模型的强大能力，提出统一预测几何、关节结构和物理属性的方法。同时借鉴现有3D表示方法但进行了改进，设计出更高效的标记化方案，并扩展了现有物理3D数据集以增强多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的基于视觉语言模型的生成框架，可以从单张图像生成包含几何、关节结构和物理属性的完整3D资产。整体流程包括：1) 输入单张真实世界图像；2) 通过多轮对话生成整体物理描述和各部分几何信息；3) 使用创新的3D表示方法高效编码几何信息；4) 利用可控流变换器细化几何细节；5) 最终输出可直接用于物理模拟的URDF、XML和部件级网格等格式的资产。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出首个模拟就绪的物理3D生成范式PhysX-Anything；2) 设计高效的3D表示方法，将标记数量减少193倍；3) 构建包含2000+对象的新数据集PhysX-Mobility，扩展类别多样性2倍以上；4) 实现可直接部署到标准物理引擎的生成流程。相比之前工作，PhysX-Anything不是检索现有模型而是全新生成；不仅关注几何外观还包含物理属性；使用更高效的表示方法；支持即插即用部署到物理引擎，而之前方法如PhysXGen虽能生成物理属性但不支持直接部署。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PhysX-Anything通过基于视觉语言模型的统一生成管道和高效的3D表示方法，首次实现了从单张图像生成可直接用于物理模拟的高质量物理3D资产，为具身AI和机器人等下游应用提供了新的可能性。'}


### 论文摘要

3D modeling is shifting from static visual representations toward physical, articulated assets that can be directly used in simulation and interaction. However, most existing 3D generation methods overlook key physical and articulation properties, thereby limiting their utility in embodied AI. To bridge this gap, we introduce PhysX-Anything, the first simulation-ready physical 3D generative framework that, given a single in-the-wild image, produces high-quality sim-ready 3D assets with explicit geometry, articulation, and physical attributes. Specifically, we propose the first VLM-based physical 3D generative model, along with a new 3D representation that efficiently tokenizes geometry. It reduces the number of tokens by 193x, enabling explicit geometry learning within standard VLM token budgets without introducing any special tokens during fine-tuning and significantly improving generative quality. In addition, to overcome the limited diversity of existing physical 3D datasets, we construct a new dataset, PhysX-Mobility, which expands the object categories in prior physical 3D datasets by over 2x and includes more than 2K common real-world objects with rich physical annotations. Extensive experiments on PhysX-Mobility and in-the-wild images demonstrate that PhysX-Anything delivers strong generative performance and robust generalization. Furthermore, simulation-based experiments in a MuJoCo-style environment validate that our sim-ready assets can be directly used for contact-rich robotic policy learning. We believe PhysX-Anything can substantially empower a broad range of downstream applications, especially in embodied AI and physics-based simulation.

---

## 53. Towards Multimodal Representation Learning in Paediatric Kidney Disease

**论文链接:** [http://arxiv.org/abs/2511.13637v1](http://arxiv.org/abs/2511.13637v1)

**作者:** Ana Durica, John Booth, Ivana Drobnjak

**发布时间:** 2025-11-17

**备注:** 4 pages, 3 figures. EurIPS 2025 Multimodal Representation Learning for Healthcare (MMRL4H) workshop paper

### GPT解析

### 总结

本研究探索了一种时间建模方法，结合纵向实验室序列和人口统计信息，预测儿童在未来30天内是否会出现异常血清肌酐值。

### 背景

儿科肾脏疾病的临床表现和进展差异很大，需要持续监测肾功能。

### 目的

开发一种预测模型，用于预测儿童在未来30天内是否会记录异常的血清肌酐值。

### 方法

使用2019年至2025年在Great Ormond Street Hospital收集的电子健康记录数据，训练一个循环神经网络模型，整合纵向实验室序列和人口统计信息。

### 主要发现

简单的时间表示可以捕捉常规儿科数据中的有用模式。

### 结论

该研究为未来使用额外临床信号和更详细的肾结局的多模态扩展奠定了基础。

### 翻译

儿科肾脏疾病在临床表现和进展上差异很大，需要持续监测肾功能。我们使用2019年至2025年在英国领先的儿科医院Great Ormond Street Hospital收集的电子健康记录，探索了一种结合纵向实验室序列和人口统计信息的时间建模方法。使用这些数据训练的循环神经网络模型用于预测儿童在未来30天内是否会记录异常的血清肌酐值。作为一项初步研究，这项工作证明了简单的时间表示可以捕捉常规儿科数据中的有用模式，并为未来使用额外临床信号和更详细肾结局的多模态扩展奠定了基础。


### 论文摘要

Paediatric kidney disease varies widely in its presentation and progression, which calls for continuous monitoring of renal function. Using electronic health records collected between 2019 and 2025 at Great Ormond Street Hospital, a leading UK paediatric hospital, we explored a temporal modelling approach that integrates longitudinal laboratory sequences with demographic information. A recurrent neural model trained on these data was used to predict whether a child would record an abnormal serum creatinine value within the following thirty days. Framed as a pilot study, this work provides an initial demonstration that simple temporal representations can capture useful patterns in routine paediatric data and lays the groundwork for future multimodal extensions using additional clinical signals and more detailed renal outcomes.

---

## 54. Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.13575v1](http://arxiv.org/abs/2511.13575v1)

**作者:** Linhan Zhou, Shuang Li, Neng Dong, Yonghang Tai, Yafei Zhang, Huafeng Li

**发布时间:** 2025-11-17

**备注:** 9 pages, 4 figures, accepted by AAAI 2026

### GPT解析

### 总结

该论文提出了一种名为分层提示学习(HPL)的统一框架，用于联合优化行人重识别中的图像到图像(I2I)和文本到图像(T2I)任务，解决了现有方法分开处理这两个任务导致的表示纠缠和性能次优问题。

### 背景

行人重识别(ReID)旨在通过视觉查询或文本描述检索目标行人图像。I2I任务强调判别性身份学习，而T2I任务需要准确的跨模态语义对齐。现有方法通常将这两个任务分开处理，可能导致表示纠缠和次优性能。

### 目的

开发一个统一框架，能够同时处理I2I和T2I任务，提高行人重识别的性能，避免表示纠缠问题。

### 方法

提出分层提示学习(HPL)框架，包括：(1)任务路由器Transformer，将双分类令牌集成到共享视觉编码器中；(2)分层提示生成方案，结合身份级可学习令牌和实例级伪文本令牌；(3)跨模态提示正则化策略，确保伪提示保留源模态特性同时增强跨模态可迁移性。

### 主要发现

通过任务感知的提示建模可以联合优化I2I和T2I任务，避免表示纠缠；分层提示生成和跨模态提示正则化策略能够有效提升模型性能。

### 结论

提出的HPL框架在多个ReID基准上取得了最先进的性能，证明了统一处理I2I和T2I任务的有效性。

### 翻译

行人重识别(ReID)旨在通过视觉查询(图像到图像，I2I)或文本描述(文本到图像，T2I)检索目标行人图像。虽然这两个任务共享共同的重识别目标，但它们带来不同的挑战：I2I强调判别性身份学习，而T2I需要准确的跨模态语义对齐。现有方法通常将这两个任务分开处理，可能导致表示纠缠和次优性能。为解决这一问题，我们提出了一种名为分层提示学习(HPL)的统一框架，利用任务感知的提示建模来联合优化这两个任务。具体而言，我们首先引入任务路由器Transformer，将双分类令牌集成到共享视觉编码器中，分别为I2I和T2I分支路由特征。在此基础上，我们开发了一种分层提示生成方案，将身份级可学习令牌与实例级伪文本令牌相结合。这些伪令牌通过模态特定的反转网络从图像或文本特征中导出，将细粒度的、实例特定的语义注入到提示中。此外，我们提出了跨模态提示正则化策略，在提示令牌空间中强制语义对齐，确保伪提示保留源模态特性同时增强跨模态可迁移性。在多个ReID基准上的广泛实验验证了我们方法的有效性，在I2I和T2I任务上都取得了最先进的性能。


### 论文摘要

Person re-identification (ReID) aims to retrieve target pedestrian images given either visual queries (image-to-image, I2I) or textual descriptions (text-to-image, T2I). Although both tasks share a common retrieval objective, they pose distinct challenges: I2I emphasizes discriminative identity learning, while T2I requires accurate cross-modal semantic alignment. Existing methods often treat these tasks separately, which may lead to representation entanglement and suboptimal performance. To address this, we propose a unified framework named Hierarchical Prompt Learning (HPL), which leverages task-aware prompt modeling to jointly optimize both tasks. Specifically, we first introduce a Task-Routed Transformer, which incorporates dual classification tokens into a shared visual encoder to route features for I2I and T2I branches respectively. On top of this, we develop a hierarchical prompt generation scheme that integrates identity-level learnable tokens with instance-level pseudo-text tokens. These pseudo-tokens are derived from image or text features via modality-specific inversion networks, injecting fine-grained, instance-specific semantics into the prompts. Furthermore, we propose a Cross-Modal Prompt Regularization strategy to enforce semantic alignment in the prompt token space, ensuring that pseudo-prompts preserve source-modality characteristics while enhancing cross-modal transferability. Extensive experiments on multiple ReID benchmarks validate the effectiveness of our method, achieving state-of-the-art performance on both I2I and T2I tasks.

---

## 55. Fairness-Aware Graph Representation Learning with Limited Demographic Information

**论文链接:** [http://arxiv.org/abs/2511.13540v1](http://arxiv.org/abs/2511.13540v1)

**作者:** Zichong Wang, Zhipeng Yin, Liping Yang, Jun Zhuang, Rui Yu, Qingzhao Kong, Wenbin Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为FairGLite的公平图学习框架，能够在有限人口统计信息的情况下有效减轻图学习中的偏见，同时保持模型效用。

### 背景

确保图神经网络中的公平性对推广可信且具有社会责任感的机器学习系统至关重要。然而，大多数现有公平图学习方法假设可完全获取人口统计信息，这在实际中因隐私、法律或监管限制而难以实现。

### 目的

引入一种新的公平图学习框架，以在有限人口统计信息的情况下减轻图学习中的偏见。

### 方法

提出由部分人口统计数据引导的代理生成机制、强制不同人口统计组节点嵌入一致性的策略，以及根据预测置信度动态调整节点贡献的自适应置信度策略。

### 主要发现

在多个数据集和公平图学习框架上的实验证明，该框架在减轻偏见和保持模型效用方面均有效。

### 结论

FairGLite框架在有限人口统计信息条件下可实现可证明的群体公平性上界，为偏见缓解提供了正式保证。

### 翻译

确保图神经网络中的公平性对于推广可信且具有社会责任感的机器学习系统至关重要。为此，近年来提出了许多公平图学习方法。然而，它们中的大多数假设可以完全获取人口统计信息，这一要求在实际中很少能满足，原因包括隐私、法律或监管限制。为此，本文引入了一个新的公平图学习框架，可在有限人口统计信息的情况下减轻图学习中的偏见。具体而言，我们提出了一种由部分人口统计数据引导的机制来生成人口统计信息的代理，并设计了一种强制不同人口统计组节点嵌入一致性的策略。此外，我们还开发了一种自适应置信度策略，根据预测置信度动态调整每个节点对公平性和效用的贡献。我们进一步提供了理论分析，证明我们的框架FairGLite在群体公平性指标上可证明的上界，为偏见缓解提供了正式保证。通过在多个数据集和公平图学习框架上的广泛实验，我们证明了该框架在减轻偏见和保持模型效用方面的有效性。


### 论文摘要

Ensuring fairness in Graph Neural Networks is fundamental to promoting trustworthy and socially responsible machine learning systems. In response, numerous fair graph learning methods have been proposed in recent years. However, most of them assume full access to demographic information, a requirement rarely met in practice due to privacy, legal, or regulatory restrictions. To this end, this paper introduces a novel fair graph learning framework that mitigates bias in graph learning under limited demographic information. Specifically, we propose a mechanism guided by partial demographic data to generate proxies for demographic information and design a strategy that enforces consistent node embeddings across demographic groups. In addition, we develop an adaptive confidence strategy that dynamically adjusts each node's contribution to fairness and utility based on prediction confidence. We further provide theoretical analysis demonstrating that our framework, FairGLite, achieves provable upper bounds on group fairness metrics, offering formal guarantees for bias mitigation. Through extensive experiments on multiple datasets and fair graph learning frameworks, we demonstrate the framework's effectiveness in both mitigating bias and maintaining model utility.

---

## 56. Interpretable Ransomware Detection Using Hybrid Large Language Models: A Comparative Analysis of BERT, RoBERTa, and DeBERTa Through LIME and SHAP

**论文链接:** [http://arxiv.org/abs/2511.13517v1](http://arxiv.org/abs/2511.13517v1)

**作者:** Elodie Mutombo Ngoie, Mike Nkongolo Wa Nkongolo, Peace Azugo, Mahmut Tokmak

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究比较了三种基于Transformer的大语言模型(BERT、RoBERTa和DeBERTa)用于勒索软件检测，通过将数值和分类特征转换为文本序列，并应用可解释AI技术实现透明决策。RoBERTa表现最佳F1分数，BERT获得最高ROC-AUC，不同模型展现出对勒索软件特征的不同敏感性。

### 背景

勒索软件持续演变且日益复杂，使得早期和可解释的检测成为现代网络安全系统的关键需求。

### 目的

比较三种Transformer大语言模型用于勒索软件检测的有效性，并确保在高风险领域中的决策透明度。

### 方法

使用UGRansome和Process Memory两个结构化数据集，将数值和分类特征转换为文本序列，在约2,500个标记样本上微调模型，使用准确率、F1分数和ROC-AUC评估性能，并应用LIME和SHAP技术解释特征贡献。

### 主要发现

BERT主要依赖文件操作特征，RoBERTa平衡依赖网络和金融信号，DeBERTa对金融和网络流量指标敏感；RoBERTa产生更各向同性的嵌入，DeBERTa捕获高度方向性的模式；RoBERTa实现最强F1分数，BERT产生最高ROC-AUC。

### 结论

将大语言模型与可解释AI结合提供了透明框架，能够识别勒索软件预测背后的特征级证据。

### 翻译

勒索软件持续演变且日益复杂，使得早期和可解释的检测成为现代网络安全系统的关键需求。本研究使用UGRansome和进程内存(PM)两个结构化数据集，对三种基于Transformer的大语言模型(BERT、RoBERTa和DeBERTa)进行了勒索软件检测的比较分析。由于大语言模型主要用于自然语言处理，数值型和分类型的勒索软件特征使用KBinsDiscretizer和基于token的编码转换为文本序列。这使得模型能够通过上下文嵌入从系统活动和网络流量中学习行为模式。模型在约2,500个标记样本上进行了微调，并使用准确率、F1分数和ROC-AUC进行评估。为确保高风险领域中的决策透明，应用了两种可解释AI技术(LIME和SHAP)来解释特征贡献。结果表明，模型学习到不同的勒索软件相关线索：BERT主要依赖主导文件操作特征，RoBERTa对网络和金融信号表现出平衡的依赖，而DeBERTa对金融和网络流量指标表现出强烈的敏感性。嵌入的可视化进一步显示了token表示的结构性差异，RoBERTa产生更各向同性的嵌入，而DeBERTa捕获高度方向性的、解耦的模式。总体而言，RoBERTa实现了最强的F1分数，而BERT产生了最高的ROC-AUC性能。将大语言模型与可解释AI的整合提供了一个透明框架，能够识别勒索软件预测背后的特征级证据。


### 论文摘要

Ransomware continues to evolve in complexity, making early and explainable detection a critical requirement for modern cybersecurity systems. This study presents a comparative analysis of three Transformer-based Large Language Models (LLMs) (BERT, RoBERTa, and DeBERTa) for ransomware detection using two structured datasets: UGRansome and Process Memory (PM). Since LLMs are primarily designed for natural language processing (NLP), numerical and categorical ransomware features were transformed into textual sequences using KBinsDiscretizer and token-based encoding. This enabled the models to learn behavioural patterns from system activity and network traffic through contextual embeddings. The models were fine-tuned on approximately 2,500 labelled samples and evaluated using accuracy, F1 score, and ROC-AUC. To ensure transparent decision-making in this high-stakes domain, two explainable AI techniques (LIME and SHAP) were applied to interpret feature contributions. The results show that the models learn distinct ransomware-related cues: BERT relies heavily on dominant file-operation features, RoBERTa demonstrates balanced reliance on network and financial signals, while DeBERTa exhibits strong sensitivity to financial and network-traffic indicators. Visualisation of embeddings further reveals structural differences in token representation, with RoBERTa producing more isotropic embeddings and DeBERTa capturing highly directional, disentangled patterns. In general, RoBERTa achieved the strongest F1-score, while BERT yielded the highest ROC-AUC performance. The integration of LLMs with XAI provides a transparent framework capable of identifying feature-level evidence behind ransomware predictions.

---

## 57. Naga: Vedic Encoding for Deep State Space Models

**论文链接:** [http://arxiv.org/abs/2511.13510v1](http://arxiv.org/abs/2511.13510v1)

**作者:** Melanie Schaller, Nick Janssen, Bodo Rosenhahn

**发布时间:** 2025-11-17

**备注:** submitted to JMLR

### GPT解析

### 总结

本文提出了Naga，一种受吠陀数学结构概念启发的深度状态空间模型编码方法，通过双向处理和元素交互增强时间序列中远距离时间依赖关系的捕捉能力，在多个长期时间序列预测基准上优于28个当前最先进模型。

### 背景

现有的深度状态空间模型在处理时间序列数据时面临捕捉远距离时间依赖关系的挑战，而吠陀数学中的结构概念可能为解决这个问题提供新的思路。

### 目的

开发一种基于吠陀数学结构概念的时间序列编码方法，能够有效捕捉远距离时间依赖关系，并在长期时间序列预测任务中实现高性能和高效率。

### 方法

提出Naga方法，通过联合处理正向和时间反转的输入序列创建双向表示，然后使用逐元素交互将这些表示相结合，形成受吠陀启发的编码。

### 主要发现

Naga在多个长期时间序列预测基准(包括ETTh1、ETTh2、ETTm1、ETTm2、Weather、Traffic和ILI)上优于28个当前最先进的模型，并且与现有的深度SSM方法相比效率更高。

### 结论

融入结构化的、受吠陀启发的分解可以为长序列建模提供一种可解释且计算效率高的替代方案。

### 翻译

本文提出了Naga，一种受吠陀数学结构概念启发的深度状态空间模型编码方法。该方法通过联合处理正向和时间反转的输入序列，为时间序列引入了双向表示。这些表示随后通过逐元素交互相结合，产生了一种受吠陀启发的编码，增强了模型捕获远距离时间步长间时间依赖性的能力。我们在多个长期时间序列预测基准上评估了Naga，包括ETTh1、ETTh2、ETTm1、ETTm2、Weather、Traffic和ILI。实验结果表明，Naga优于28个当前最先进的模型，并且与现有的深度SSM方法相比效率更高。研究结果表明，融入结构化的、受吠陀启发的分解可以为长序列建模提供一种可解释且计算效率高的替代方案。


### 论文摘要

This paper presents Naga, a deep State Space Model (SSM) encoding approach inspired by structural concepts from Vedic mathematics. The proposed method introduces a bidirectional representation for time series by jointly processing forward and time-reversed input sequences. These representations are then combined through an element-wise (Hadamard) interaction, resulting in a Vedic-inspired encoding that enhances the model's ability to capture temporal dependencies across distant time steps. We evaluate Naga on multiple long-term time series forecasting (LTSF) benchmarks, including ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, and ILI. The experimental results show that Naga outperforms 28 current state of the art models and demonstrates improved efficiency compared to existing deep SSM-based approaches. The findings suggest that incorporating structured, Vedic-inspired decomposition can provide an interpretable and computationally efficient alternative for long-range sequence modeling.

---

## 58. GREAT: Generalizable Representation Enhancement via Auxiliary Transformations for Zero-Shot Environmental Prediction

**论文链接:** [http://arxiv.org/abs/2511.13469v1](http://arxiv.org/abs/2511.13469v1)

**作者:** Shiyuan Luo, Chonghao Qiu, Runlong Yu, Yiqun Xie, Xiaowei Jia

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了GREAT（通过辅助变换增强通用表示）框架，通过增强现有数据集来改善对完全未监测区域的预测能力。该框架在神经网络的多层次学习变换函数，并通过双层训练过程确保增强数据保留原始数据的关键模式。

### 背景

环境建模在预测未监测区域的生态系统动态方面面临关键挑战，这源于观测数据有限且地理分布不均衡。空间异质性导致模型学习到仅适用于局部数据的虚假模式，且环境建模需要在增强过程中保持不变的物理关系和时间一致性。

### 目的

开发一种有效方法增强数据集，提高模型在完全未见过的区域的预测能力，同时确保增强的数据能够恢复原始控制过程并改善模型泛化能力。

### 方法

GREAT框架通过在神经网络的多个层次学习变换函数来增强原始环境特征和时间影响。这些变换函数通过双层训练过程优化，约束增强数据以保留原始源数据的关键模式，确保增强后的数据能恢复原始控制过程并提高模型泛化能力。

### 主要发现

在美国东部六个生态多样化的流域（每个流域包含多个河段）的河流温度预测实验中，GREAT在零样本场景中显著优于现有方法。

### 结论

该工作为环境应用提供了一种实用解决方案，特别是在全面监测不可行的情况下，能有效提高模型在未监测区域的预测能力。

### 翻译

环境建模由于观测数据有限且地理分布不均，在预测未监测区域的生态系统动态方面面临关键挑战。空间异质性加剧了这一挑战，导致模型学习到仅适用于局部数据的虚假模式。与传统的领域泛化不同，环境建模在增强过程中必须保持不变的物理关系和时间一致性。在本文中，我们引入了通过辅助变换增强通用表示（GREAT）的框架，该框架有效增强可用数据集，以改善对完全未监测区域的预测。GREAT指导增强过程，确保可以从增强数据中恢复原始的控制过程，并且增强数据的加入能够提高模型的泛化能力。具体而言，GREAT在神经网络的多个层次学习变换函数，以增强原始环境特征和时间影响。通过一种新颖的双层训练过程对这些变换函数进行优化，该过程约束增强数据以保留原始源数据的关键模式。我们在美国东部六个生态多样化的流域的河流温度预测上验证了GREAT的有效性，每个流域包含多个河段。实验结果表明，GREAT在零样本场景中显著优于现有方法。这项工作为全面监测不可行的环境应用提供了实用解决方案。


### 论文摘要

Environmental modeling faces critical challenges in predicting ecosystem dynamics across unmonitored regions due to limited and geographically imbalanced observation data. This challenge is compounded by spatial heterogeneity, causing models to learn spurious patterns that fit only local data. Unlike conventional domain generalization, environmental modeling must preserve invariant physical relationships and temporal coherence during augmentation. In this paper, we introduce Generalizable Representation Enhancement via Auxiliary Transformations (GREAT), a framework that effectively augments available datasets to improve predictions in completely unseen regions. GREAT guides the augmentation process to ensure that the original governing processes can be recovered from the augmented data, and the inclusion of the augmented data leads to improved model generalization. Specifically, GREAT learns transformation functions at multiple layers of neural networks to augment both raw environmental features and temporal influence. They are refined through a novel bi-level training process that constrains augmented data to preserve key patterns of the original source data. We demonstrate GREAT's effectiveness on stream temperature prediction across six ecologically diverse watersheds in the eastern U.S., each containing multiple stream segments. Experimental results show that GREAT significantly outperforms existing methods in zero-shot scenarios. This work provides a practical solution for environmental applications where comprehensive monitoring is infeasible.

---

## 59. Moving Pictures of Thought: Extracting Visual Knowledge in Charles S. Peirce's Manuscripts with Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.13378v1](http://arxiv.org/abs/2511.13378v1)

**作者:** Carlo Teo Pedretti, Davide Picca, Dario Rodighiero

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了视觉语言模型(VLMs)在识别和解释混合文本-图表页面中的应用，特别关注查尔斯·皮尔士手稿中的复杂视觉元素。通过结合皮尔士的符号学框架和现代AI技术，研究者开发了一种工作流程，能够有效提取图表知识并将其整合到知识图谱中，实现了对复合来源中图表内容的结构化表示。

### 背景

图表在许多学科中是重要但研究不足的工具，展示了视觉表现与学术推理之间的紧密联系。然而，图表的图标形式对视觉研究、媒介间分析和基于文本的数字工作流构成了障碍。查尔斯·皮尔士一直主张将图表作为推理和解释的必要工具，他的手稿通常将文本内容与复杂的视觉元素结合，为研究涉及异质材料的文档提供了具有挑战性的案例。

### 目的

研究旨在探究视觉语言模型(VLMs)是否能有效帮助识别和解释混合文本-图表页面，特别是皮尔士手稿中的这类页面，并开发一种方法来提取图表知识并将其结构化表示。

### 方法

研究提出了一种工作流程，包括(i)分割手稿页面布局，(ii)将每个片段与IIIF兼容注释重新连接，(iii)将包含图表的片段提交给VLM。此外，研究采用皮尔士的符号学框架设计了提示词，以提取图表的关键知识并生成简洁的标题。最后，将这些标题整合到知识图谱中，实现复合来源中图表内容的结构化表示。

### 主要发现

研究成功开发了一种结合AI技术和符号学理论的方法，能够有效识别和解释混合文本-图表页面，提取图表知识并将其结构化表示。这种方法为研究包含异质材料的历史文档提供了新的可能性。

### 结论

通过将视觉语言模型与皮尔士的符号学框架相结合，研究为处理和解释包含文本和图表的复杂文档提供了有效方法。这种方法不仅有助于历史文档的研究，也为视觉研究和媒介间分析提供了新的工具和视角。

### 翻译

图表在许多学科中是重要但研究不足的工具，展示了视觉表现与学术推理之间的紧密联系。然而，它们的图标形式对视觉研究、媒介间分析和基于文本的数字工作流构成了障碍。特别是，查尔斯·皮尔士一直主张将图表作为推理和解释的必要工具。他的手稿通常将文本内容与复杂的视觉元素结合，为研究涉及异质材料的文档提供了具有挑战性的案例。在这项初步研究中，我们探究视觉语言模型(VLMs)是否能有效帮助我们识别和解释这类混合页面。首先，我们提出了一种工作流程，包括(i)分割手稿页面布局，(ii)将每个片段与IIIF兼容注释重新连接，(iii)将包含图表的片段提交给VLM。此外，通过采用皮尔士的符号学框架，我们设计了提示词，以提取图表的关键知识并生成简洁的标题。最后，我们将这些标题整合到知识图谱中，实现了复合来源中图表内容的结构化表示。


### 论文摘要

Diagrams are crucial yet underexplored tools in many disciplines, demonstrating the close connection between visual representation and scholarly reasoning. However, their iconic form poses obstacles to visual studies, intermedial analysis, and text-based digital workflows. In particular, Charles S. Peirce consistently advocated the use of diagrams as essential for reasoning and explanation. His manuscripts, often combining textual content with complex visual artifacts, provide a challenging case for studying documents involving heterogeneous materials. In this preliminary study, we investigate whether Visual Language Models (VLMs) can effectively help us identify and interpret such hybrid pages in context. First, we propose a workflow that (i) segments manuscript page layouts, (ii) reconnects each segment to IIIF-compliant annotations, and (iii) submits fragments containing diagrams to a VLM. In addition, by adopting Peirce's semiotic framework, we designed prompts to extract key knowledge about diagrams and produce concise captions. Finally, we integrated these captions into knowledge graphs, enabling structured representations of diagrammatic content within composite sources.

---

## 60. FLOWER: Flow-Oriented Entity-Relationship Tool

**论文链接:** [http://arxiv.org/abs/2511.13357v1](http://arxiv.org/abs/2511.13357v1)

**作者:** Dmitry Moskalev

**发布时间:** 2025-11-17

**备注:** 12 pages, 8 figures

### GPT解析

### 总结

本文介绍了FLOWER，一个面向流程的实体关系工具，作为首个端到端解决方案，能够自动处理、创建和可视化主流SQL方言中的显式和隐式依赖关系，使用动态采样和稳健的数据分析技术改进实体关系模型和数据叙事。

### 背景

跨数据源的关系探索是实体识别的关键优化。数据库可存储大量合成和有机数据，正确处理所有对象是重要任务，但构建实体关系模型的决策与人为因素相关，传统方法存在常规和资源密集型问题。

### 目的

开发一个能自动处理、创建和可视化实体依赖关系的工具，消除常规和资源密集型问题，改进实体关系模型和数据叙事，以更好理解数据基础并获取未见解。

### 方法

FLOWER工具使用动态采样和稳健的数据分析技术自动检测内置约束并创建必要约束，支持23种语言，兼容CPU和GPU，适用于主流SQL方言，可通过SQL或自然语言操作。

### 主要发现

在STATS基准测试中，FLOWER在分布表示方面比水库采样优越2.4倍，约束学习方面优越2.6倍，加速2.15倍；数据叙事方面准确性提升1.19倍，上下文减少1.86倍，优于LLM。

### 结论

FLOWER能更好地处理现实世界数据，确保质量、可扩展性和不同用例的适用性，是独特且全面的端到端解决方案。

### 翻译

探索跨数据源的关系是实体识别的关键优化。由于数据库可以存储大量合成和有机数据，正确处理所有数量的对象是一项重要任务。然而，如何构建实体关系模型的决策与人为因素相关。在本文中，我们提出了面向流程的实体关系工具。这是第一个也是唯一的端到端解决方案，能够即时消除处理、创建和可视化主流SQL方言中显式和隐式依赖关系的常规和资源密集型问题。一旦启动，FLOWER会自动检测内置约束，并使用动态采样和稳健的数据分析技术开始创建自己的正确且必要的约束。这种方法适用于改进实体关系模型和数据叙事，以更好地理解数据基础并使用SQL或自然语言从数据库源获取未见的见解。在最新的STATS基准上进行评估，实验表明FLOWER在分布表示方面比水库采样优越2.4倍，在约束学习方面优越2.6倍，加速2.15倍。对于数据叙事，我们的工具在准确性方面提升1.19倍，同时上下文减少1.86倍，与LLM相比。所提出的工具还支持23种语言，同时兼容CPU和GPU。这些结果表明，FLOWER能够更好地处理现实世界数据，确保质量、可扩展性和不同用例的适用性。


### 论文摘要

Exploring relationships across data sources is a crucial optimization for entities recognition. Since databases can store big amount of information with synthetic and organic data, serving all quantity of objects correctly is an important task to deal with. However, the decision of how to construct entity relationship model is associated with human factor. In this paper, we present flow-oriented entity-relationship tool. This is first and unique end-to-end solution that eliminates routine and resource-intensive problems of processing, creating and visualizing both of explicit and implicit dependencies for prominent SQL dialects on-the-fly. Once launched, FLOWER automatically detects built-in constraints and starting to create own correct and necessary one using dynamic sampling and robust data analysis techniques. This approach applies to improve entity-relationship model and data storytelling to better understand the foundation of data and get unseen insights from DB sources using SQL or natural language. Evaluated on state-of-the-art STATS benchmark, experiments show that FLOWER is superior to reservoir sampling by 2.4x for distribution representation and 2.6x for constraint learning with 2.15x acceleration. For data storytelling, our tool archives 1.19x for accuracy enhance with 1.86x context decrease compare to LLM. Presented tool is also support 23 languages and compatible with both of CPU and GPU. Those results show that FLOWER can manage with real-world data a way better to ensure with quality, scalability and applicability for different use-cases.

---

## 61. Computer Vision based group activity detection and action spotting

**论文链接:** [http://arxiv.org/abs/2511.13315v1](http://arxiv.org/abs/2511.13315v1)

**作者:** Narthana Sivalingam, Santhirarajah Sivasthigan, Thamayanthi Mahendranathan, G. M. R. I. Godaliyadda, M. P. B. Ekanayake, H. M. V. R. Herath

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种基于计算机视觉的框架，用于多场景中的群体活动检测和动作定位，通过结合深度学习模型和基于图的关系推理，有效解决了复杂人际互动、遮挡和外观变化带来的挑战。

### 背景

多场景中群体活动检测具有挑战性，因为存在复杂的人际互动、遮挡以及外观随时间变化等问题。

### 目的

开发一种计算机视觉框架，用于群体活动识别和动作定位，结合深度学习模型和基于图的关系推理技术。

### 方法

系统首先应用Mask R-CNN获取精确的演员定位；使用多种骨干网络（Inception V3、MobileNet和VGG16）提取特征图；应用RoIAlign保持空间对齐；将掩码信息与特征图融合获得精细特征表示；构建演员关系图编码相似性和位置关系；使用图卷积网络推理关系并预测个人动作和群体活动。

### 主要发现

在Collective Activity数据集上的实验表明，基于掩码的特征细化、鲁棒相似性搜索和图神经网络推理的组合在拥挤和非拥挤场景中都提高了识别性能。

### 结论

该方法展示了整合分割、特征提取和关系图推理在复杂视频理解任务中的潜力。

### 翻译

多场景中的群体活动检测具有挑战性，因为复杂的人际互动、遮挡和外观随时间变化。这项工作提出了一种基于计算机视觉的框架，用于群体活动识别和动作定位，结合了深度学习模型和基于图的关系推理。系统首先应用Mask R-CNN通过边界框和实例掩码获取精确的演员定位。使用多种骨干网络（包括Inception V3、MobileNet和VGG16）提取特征图，并应用RoIAlign在生成特定演员特征时保持空间对齐。然后将掩码信息与特征图融合，获得每个演员的精细掩码特征表示。为了建模个体之间的互动，我们构建了演员关系图，使用归一化互相关、绝对差之和和点积等方法编码外观相似性和位置关系。图卷积网络在这些图上操作，推理关系并预测个人动作和群体活动。在Collective Activity数据集上的实验表明，基于掩码的特征细化、鲁棒相似性搜索和图神经网络推理的组合在拥挤和非拥挤场景中都提高了识别性能。这种方法强调了整合分割、特征提取和关系图推理在复杂视频理解任务中的潜力。


### 论文摘要

Group activity detection in multi-person scenes is challenging due to complex human interactions, occlusions, and variations in appearance over time. This work presents a computer vision based framework for group activity recognition and action spotting using a combination of deep learning models and graph based relational reasoning. The system first applies Mask R-CNN to obtain accurate actor localization through bounding boxes and instance masks. Multiple backbone networks, including Inception V3, MobileNet, and VGG16, are used to extract feature maps, and RoIAlign is applied to preserve spatial alignment when generating actor specific features. The mask information is then fused with the feature maps to obtain refined masked feature representations for each actor. To model interactions between individuals, we construct Actor Relation Graphs that encode appearance similarity and positional relations using methods such as normalized cross correlation, sum of absolute differences, and dot product. Graph Convolutional Networks operate on these graphs to reason about relationships and predict both individual actions and group level activities. Experiments on the Collective Activity dataset demonstrate that the combination of mask based feature refinement, robust similarity search, and graph neural network reasoning leads to improved recognition performance across both crowded and non crowded scenarios. This approach highlights the potential of integrating segmentation, feature extraction, and relational graph reasoning for complex video understanding tasks.

---

## 62. DAP: A Discrete-token Autoregressive Planner for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.13306v1](http://arxiv.org/abs/2511.13306v1)

**作者:** Bowen Ye, Bin Zhang, Hang Zhao

**发布时间:** 2025-11-17

### GPT解析

### 总结

DAP是一种创新的自动驾驶规划方法，通过联合预测BEV语义和自车轨迹，解决了传统方法的监督稀疏和约束弱的问题，并在有限的参数预算下实现了优异的性能。

### 背景

在自动驾驶中，通过扩展数据和模型预算获得可持续的性能提升仍然是一个关键但未解决的挑战。自回归模型在规划任务中显示出有希望的数据扩展效率，但仅预测自车轨迹存在监督稀疏和约束弱的问题。

### 目的

解决自动驾驶中通过扩展数据和模型预算获得可持续性能提升的挑战，以及自车轨迹预测中监督稀疏和约束弱的问题。

### 方法

引入DAP（离散令牌自回归规划器），联合预测BEV语义和自车轨迹；使用基于强化学习的微调，保留监督行为克隆先验，同时注入奖励引导的改进。

### 主要发现

尽管只有1.6亿参数的紧凑预算，DAP在开环指标上实现了最先进的性能，在NAVSIM基准测试上提供了具有竞争力的闭环结果。

### 结论

完全离散令牌自回归公式在光栅化BEV和自车动作上运行，为自动驾驶提供了一种紧凑且可扩展的规划范式。

### 翻译

通过扩展数据和模型预算获得可持续的性能提升仍然是自动驾驶中的一个关键但未解决的挑战。虽然自回归模型在规划任务中显示出有希望的数据扩展效率，但仅预测自车轨迹存在监督稀疏和约束弱的问题，无法充分限制场景演变应如何影响自车运动。因此，我们引入了DAP，一种离散令牌自回归规划器，它联合预测BEV语义和自车轨迹，从而强制执行全面的表示学习，并允许预测的动态直接条件化自车运动。此外，我们整合了基于强化学习的微调，这保留了监督行为克隆先验，同时注入了奖励引导的改进。尽管只有1.6亿参数的紧凑预算，DAP在开环指标上实现了最先进的性能，并在NAVSIM基准测试上提供了具有竞争力的闭环结果。总体而言，在光栅化BEV和自车动作上运行的完全离散令牌自回归公式，为自动驾驶提供了一种紧凑且可扩展的规划范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶中如何通过扩大数据规模和模型预算实现可持续性能提升的问题。现实中，自动驾驶的安全性和可靠性高度依赖规划能力，而研究上，这个问题关系到如何从有限数据中学习泛化能力强的规划策略，克服监督稀疏和场景演变约束不足的挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到大型语言模型中解码器Transformer在离散令牌序列建模中的高效扩展性，并借鉴到自动驾驶领域。他们发现现有自回归规划模型仅预测轨迹存在监督稀疏问题，因此引入世界建模目标。方法借鉴了VQ-VAE进行BEV离散化、SAC-BC进行强化学习微调，以及稀疏MoE设计来扩展模型能力，同时保持架构简洁高效。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过离散令牌自回归模型联合预测未来环境表示（BEV语义）和自车轨迹，提供密集时空对齐监督，紧密耦合场景理解与运动生成。整体流程：1)输入令牌化（将BEV和轨迹转换为离散令牌）；2)自回归Transformer处理令牌序列，先预测BEV再预测轨迹；3)两阶段训练（先行为克隆预训练，再SAC-BC微调）；4)轻量级轨迹后调优，改进可行性和舒适性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)离散令牌自回归规划器架构；2)联合环境-轨迹预测提供密集监督；3)SAC-BC微调超越纯模仿学习；4)紧凑模型达到高性能。相比之前工作，不同之处在于：联合预测而非仅预测轨迹；使用离散令牌接口稳定模块交互；聚焦BEV潜在空间而非像素级世界模型，使系统更轻量级；包含轻量级后处理改进舒适性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了DAP，一种通过联合预测环境表示和自车轨迹、结合SAC-BC微调和轻量级后处理的离散令牌自回归规划器，在保持模型规模紧凑的同时实现了高效可靠的自动驾驶规划。'}


### 论文摘要

Gaining sustainable performance improvement with scaling data and model budget remains a pivotal yet unresolved challenge in autonomous driving. While autoregressive models exhibited promising data-scaling efficiency in planning tasks, predicting ego trajectories alone suffers sparse supervision and weakly constrains how scene evolution should shape ego motion. Therefore, we introduce DAP, a discrete-token autoregressive planner that jointly forecasts BEV semantics and ego trajectories, thereby enforcing comprehensive representation learning and allowing predicted dynamics to directly condition ego motion. In addition, we incorporate a reinforcement-learning-based fine-tuning, which preserves supervised behavior cloning priors while injecting reward-guided improvements. Despite a compact 160M parameter budget, DAP achieves state-of-the-art performance on open-loop metrics and delivers competitive closed-loop results on the NAVSIM benchmark. Overall, the fully discrete-token autoregressive formulation operating on both rasterized BEV and ego actions provides a compact yet scalable planning paradigm for autonomous driving.

---

## 63. PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement

**论文链接:** [http://arxiv.org/abs/2511.13300v1](http://arxiv.org/abs/2511.13300v1)

**作者:** Xiaobin Rong, Qinwen Hu, Mansur Yesilbursa, Kamil Wojcicki, Jing Lu

**发布时间:** 2025-11-17

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本研究提出了一种名为PASE(语音锚定语音增强器)的生成式语音增强框架，有效解决了现有生成式模型在严重噪声下产生的语言和声学幻觉问题，显著提升了语音增强的感知质量。

### 背景

生成式模型在语音增强方面表现出色，但现有方法往往忽略了严重噪声下的幻觉风险，导致错误的语音内容或不一致的说话人特征，分别称为语言幻觉和声学幻觉。

### 目的

解决生成式语音增强模型中的语言和声学幻觉问题，通过利用预训练模型中的语音先验来提高语音增强的感知质量和准确性。

### 方法

提出PASE框架，首先通过表示蒸馏将WavLM适应为去噪专家以清理其最终层特征；然后采用双流表示训练声码器，高级语音表示提供干净的语音内容，低级声学表示保留说话人身份和韵律。

### 主要发现

PASE不仅在感知质量上超越了最先进的判别式模型，而且在显著降低语言和声学幻觉方面也明显优于先前的生成式模型。

### 结论

通过利用预训练WavLM模型中嵌入的稳健语音先验，PASE有效减轻了语音增强中的幻觉问题，为生成式语音增强提供了新的解决方案。

### 翻译

生成式模型在语音增强方面已展现出卓越的性能，在感知质量上超过了传统的判别式方法。然而，现有的生成式语音增强方法常常忽略了在严重噪声下的幻觉风险，导致错误的语音内容或不一致的说话人特征，我们分别称之为语言幻觉和声学幻觉。我们认为语言幻觉源于模型未能约束有效的语音结构，这是一个更根本的挑战。虽然语言模型(LMs)非常适合通过建模离散标记的分布来捕获底层语音结构，但现有方法在学习受噪声污染的表示方面存在局限，这可能导致污染的先验和幻觉。为了克服这些局限性，我们提出了语音锚定语音增强器(PASE)，这是一个利用预训练WavLM模型中嵌入的稳健语音先验来减轻幻觉的生成式语音增强框架。首先，我们通过表示蒸馏将WavLM适应为去噪专家，以清理其最终层特征。在模型内在语音先验的指导下，这一过程实现了稳健去噪同时最小化语言幻觉。为了进一步减少声学幻觉，我们使用双流表示训练声码器：高级语音表示提供干净的语音内容，而低级声学表示保留说话人身份和韵律。实验结果表明，PASE不仅在感知质量上超越了最先进的判别式模型，而且在显著降低语言和声学幻觉方面也明显优于先前的生成式模型。


### 论文摘要

Generative models have shown remarkable performance in speech enhancement (SE), achieving superior perceptual quality over traditional discriminative approaches. However, existing generative SE approaches often overlook the risk of hallucination under severe noise, leading to incorrect spoken content or inconsistent speaker characteristics, which we term linguistic and acoustic hallucinations, respectively. We argue that linguistic hallucination stems from models' failure to constrain valid phonological structures and it is a more fundamental challenge. While language models (LMs) are well-suited for capturing the underlying speech structure through modeling the distribution of discrete tokens, existing approaches are limited in learning from noise-corrupted representations, which can lead to contaminated priors and hallucinations. To overcome these limitations, we propose the Phonologically Anchored Speech Enhancer (PASE), a generative SE framework that leverages the robust phonological prior embedded in the pre-trained WavLM model to mitigate hallucinations. First, we adapt WavLM into a denoising expert via representation distillation to clean its final-layer features. Guided by the model's intrinsic phonological prior, this process enables robust denoising while minimizing linguistic hallucinations. To further reduce acoustic hallucinations, we train the vocoder with a dual-stream representation: the high-level phonetic representation provides clean linguistic content, while a low-level acoustic representation retains speaker identity and prosody. Experimental results demonstrate that PASE not only surpasses state-of-the-art discriminative models in perceptual quality, but also significantly outperforms prior generative models with substantially lower linguistic and acoustic hallucinations.

---

## 64. Hybrid-Domain Adaptative Representation Learning for Gaze Estimation

**论文链接:** [http://arxiv.org/abs/2511.13222v1](http://arxiv.org/abs/2511.13222v1)

**作者:** Qida Tan, Hongyu Yang, Wenchao Du

**发布时间:** 2025-11-17

**备注:** AAAI2026

### GPT解析

### 总结

本文提出了一种名为HARL的混合域自适应表示学习框架，解决了基于外观的凝视估计在跨域评估中性能下降的问题，通过多源数据集和创新的特征对齐方法实现了高精度凝视估计。

### 背景

基于外观的凝视估计近年来取得显著进展，能够从单张面部图像预测3D凝视方向，但大多数方法在跨域评估中性能显著下降，原因是表情、配饰和图像质量等与凝视无关因素的干扰。

### 目的

解决跨域评估中性能下降问题，提出混合域自适应表示学习框架，利用多源混合数据集学习鲁棒的凝视表示，提高凝视估计的准确性和鲁棒性。

### 方法

提出HARL框架，包括：1)通过无监督域适应方式对齐高质量眼部图像特征，从低质量面部图像中分离凝视相关表示；2)设计稀疏图融合模块，分析头部姿态影响，探索凝视方向与头部姿态间的几何约束，生成密集鲁棒的凝视表示。

### 主要发现

在EyeDiap、MPIIFaceGaze和Gaze360数据集上分别达到5.02°、3.36°和9.26°的最先进准确率，跨数据集评估中展示具有竞争力的性能。

### 结论

HARL框架能有效解决跨域评估中性能下降问题，通过多源混合数据集和创新方法实现了高精度凝视估计。

### 翻译

基于外观的凝视估计旨在从单张面部图像中预测准确的3D凝视方向，近年来取得了有希望的进展。然而，由于表情、配饰和图像质量等与凝视无关因素的干扰，大多数方法在跨域评估中遭受显著的性能下降。为了缓解这个问题，我们提出了一个新颖的混合域自适应表示学习框架，该框架利用多源混合数据集来学习鲁棒的凝视表示。更具体地说，我们提出通过无监督域适应方式对齐从高质量眼部图像提取的特征，从而从低质量面部图像中分离出与凝视相关的表示。此外，我们分析了头部姿态的影响，并设计了一个简单而高效的稀疏图融合模块来探索凝视方向与头部姿态之间的几何约束，从而生成密集而鲁棒的凝视表示。在EyeDiap、MPIIFaceGaze和Gaze360数据集上的大量实验表明，我们的方法分别达到了5.02°、3.36°和9.26°的最先进准确率，并在跨数据集评估中展示了具有竞争力的性能。代码可在https://github.com/da60266/HARL获取。


### 论文摘要

Appearance-based gaze estimation, aiming to predict accurate 3D gaze direction from a single facial image, has made promising progress in recent years. However, most methods suffer significant performance degradation in cross-domain evaluation due to interference from gaze-irrelevant factors, such as expressions, wearables, and image quality. To alleviate this problem, we present a novel Hybrid-domain Adaptative Representation Learning (shorted by HARL) framework that exploits multi-source hybrid datasets to learn robust gaze representation. More specifically, we propose to disentangle gaze-relevant representation from low-quality facial images by aligning features extracted from high-quality near-eye images in an unsupervised domain-adaptation manner, which hardly requires any computational or inference costs. Additionally, we analyze the effect of head-pose and design a simple yet efficient sparse graph fusion module to explore the geometric constraint between gaze direction and head-pose, leading to a dense and robust gaze representation. Extensive experiments on EyeDiap, MPIIFaceGaze, and Gaze360 datasets demonstrate that our approach achieves state-of-the-art accuracy of $\textbf{5.02}^{\circ}$ and $\textbf{3.36}^{\circ}$, and $\textbf{9.26}^{\circ}$ respectively, and present competitive performances through cross-dataset evaluation. The code is available at https://github.com/da60266/HARL.

---

## 65. Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.13195v1](http://arxiv.org/abs/2511.13195v1)

**作者:** Soyul Lee, Seungmin Baek, Dongbo Min

**发布时间:** 2025-11-17

**备注:** AAAI 2026 accepted

### GPT解析

### 总结

MonoDLGD是一种新颖的难度感知标签引导去噪框架，通过根据检测不确定性自适应地扰动和重建真实标签，改进了单目3D目标检测的性能，在KITTI基准上实现了所有难度级别上的最先进性能。

### 背景

单目3D目标检测是自动驾驶和机器人等应用的经济有效解决方案，但由于深度线索本质上存在模糊性，这个问题基本上是不适定的。

### 目的

提出MonoDLGD框架，解决现有方法中深度估计不准确和忽略实例级别检测难度的问题。

### 方法

MonoDLGD根据检测不确定性对真实标签进行扰动和重建，对容易的实例应用较强扰动，对困难的实例应用较弱扰动，然后重建它们以提供显式几何监督，通过联合优化标签重建和3D目标检测，鼓励几何感知的表示学习。

### 主要发现

在KITTI基准上的大量实验表明，MonoDLGD在所有难度级别上都达到了最先进的性能。

### 结论

MonoDLGD通过考虑检测难度并提供显式几何监督，提高了对不同级别对象复杂性的鲁棒性，改进了单目3D目标检测的性能。

### 翻译

单目3D目标检测是自动驾驶和机器人等应用的经济有效解决方案，但由于深度线索本质上存在模糊性，这个问题基本上是不适定的。最近的基于DETR的方法尝试通过全局注意力和辅助深度预测来缓解这一问题，但它们仍然存在深度估计不准确的问题。此外，这些方法通常忽略了实例级别的检测难度，如遮挡、距离和截断，导致次优的检测性能。我们提出了MonoDLGD，一种新颖的难度感知标签引导去噪框架，根据检测不确定性自适应地扰动和重建真实标签。具体来说，MonoDLGD对更容易的实例应用更强的扰动，对更困难的实例应用较弱的扰动，然后重建它们以有效提供显式的几何监督。通过联合优化标签重建和3D目标检测，MonoDLGD鼓励几何感知的表示学习，并提高对不同级别对象复杂性的鲁棒性。在KITTI基准上的大量实验表明，MonoDLGD在所有难度级别上都达到了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目3D物体检测中的两个核心问题：1) 由于缺乏深度线索导致的深度估计不准确问题；2) 现有方法忽略实例级检测难度（如遮挡、距离、截断等）的问题。这些问题在自动驾驶、机器人和增强现实等应用中至关重要，因为单目3D检测是一种经济有效的解决方案，但其准确性受限于单目图像的本质不适定性，直接影响这些应用的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有单目3D检测方法的局限性，包括深度估计不准确和忽略实例级难度。他们借鉴了多个现有工作：1) 借鉴DN-DETR和DINO的去噪策略，通过扰动和重建标签来稳定训练；2) 借鉴物体检测领域的不确定性建模方法，如Gaussian YOLOv3和MonoFlex；3) 受DAB-DETR启发，将动态锚框扩展到单目3D检测。在此基础上，作者创新性地设计了难度感知标签引导去噪框架，结合了不确定性估计和自适应扰动机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自适应地扰动和重建真实标签，为模型提供显式的几何监督，并根据检测的不确定性（作为检测难度的代理）调整扰动强度，对容易的实例施加更强扰动，对困难实例施加较弱扰动。整体实现流程分为两个阶段：第一阶段估计检测不确定性并应用难度感知扰动生成扰动标签；第二阶段将扰动标签和3D-DAB查询共同输入解码器，同时执行扰动重建和3D物体预测，通过重建损失和检测损失进行联合优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Difficulty-Aware Label-Guided Denoising框架，直接操作3D真实标签提供显式几何监督；2) 3D-Dynamic Anchor Box，通过投影几何和类语义编码空间先验；3) Difficulty-Aware Perturbation机制，基于不确定性自适应调整扰动强度。相比之前的工作，MonoDLGD不再依赖辅助深度预测缓解深度模糊性，同时综合考虑多种难度因素而非仅关注遮挡，并通过不确定性建模实现真正的难度感知，而非简单的均匀去噪。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MonoDLGD通过引入基于检测不确定性的难度感知标签引导去噪框架，有效解决了单目3D物体检测中深度估计不准确和忽略实例级检测难度的关键问题，显著提升了模型在复杂场景下的3D检测性能。'}


### 论文摘要

Monocular 3D object detection is a cost-effective solution for applications like autonomous driving and robotics, but remains fundamentally ill-posed due to inherently ambiguous depth cues. Recent DETR-based methods attempt to mitigate this through global attention and auxiliary depth prediction, yet they still struggle with inaccurate depth estimates. Moreover, these methods often overlook instance-level detection difficulty, such as occlusion, distance, and truncation, leading to suboptimal detection performance. We propose MonoDLGD, a novel Difficulty-Aware Label-Guided Denoising framework that adaptively perturbs and reconstructs ground-truth labels based on detection uncertainty. Specifically, MonoDLGD applies stronger perturbations to easier instances and weaker ones into harder cases, and then reconstructs them to effectively provide explicit geometric supervision. By jointly optimizing label reconstruction and 3D object detection, MonoDLGD encourages geometry-aware representation learning and improves robustness to varying levels of object complexity. Extensive experiments on the KITTI benchmark demonstrate that MonoDLGD achieves state-of-the-art performance across all difficulty levels.

---

## 66. Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition

**论文链接:** [http://arxiv.org/abs/2511.13137v1](http://arxiv.org/abs/2511.13137v1)

**作者:** Yanda Zhu, Yuanyang Zhu, Daoyi Dong, Caihua Chen, Chunlin Chen

**发布时间:** 2025-11-17

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了条件扩散模型用于动态任务分解(CD³T)，一个新颖的两级分层多智能体强化学习框架，能够自动推断子任务和协调模式，在复杂环境中实现高效的长时程任务学习。

### 背景

任务分解在复杂的多智能体强化学习任务中显示出潜力，能够在动态和不确定环境中为长时程任务提供高效的分层学习。然而，从头学习动态任务分解通常需要大量训练样本，特别是在部分可观测条件下探索大型联合动作空间时。

### 目的

设计一个自动推断子任务和协调模式的分层MARL框架，解决动态任务分解中样本效率低的问题。

### 方法

CD³T采用两级架构：高层策略学习子任务表示并基于子任务效果生成选择策略，使用条件扩散模型预测下一个观察和奖励以捕获子任务对环境的影响；低层智能体在分配的子任务内协作学习和分享专门技能。学习到的子任务表示还作为多头注意力混合网络中的额外语义信息，增强值分解并提供个体与联合值函数间的推理桥梁。

### 主要发现

在各种基准测试上，CD³T比现有基线方法实现了更好的性能。

### 结论

CD³T是一种有效的动态任务分解方法，能够在复杂多智能体环境中实现更好的性能，减少了训练样本需求。

### 翻译

任务分解在复杂的多智能体强化学习任务中显示出潜力，它能够在动态和不确定的环境中为长时程任务提供高效的分层学习。然而，从头开始学习动态任务分解通常需要大量的训练样本，特别是在部分可观测条件下探索大型联合动作空间时。在本文中，我们提出了条件扩散模型用于动态任务分解(CD³T)，这是一个新颖的两级分层MARL框架，设计用于自动推断子任务和协调模式。高层策略学习子任务表示，基于子任务效果生成子任务选择策略。为了捕获子任务对环境的影响，CD³T使用条件扩散模型预测下一个观察和奖励。在低层，智能体在分配的子任务内协作学习和分享专门技能。此外，学习到的子任务表示也被用作多头注意力混合网络中的额外语义信息，以增强值分解并提供个体值函数和联合值函数之间的高效推理桥梁。在各种基准测试上的实验结果表明，CD³T比现有基线方法实现了更好的性能。


### 论文摘要

Task decomposition has shown promise in complex cooperative multi-agent reinforcement learning (MARL) tasks, which enables efficient hierarchical learning for long-horizon tasks in dynamic and uncertain environments. However, learning dynamic task decomposition from scratch generally requires a large number of training samples, especially exploring the large joint action space under partial observability. In this paper, we present the Conditional Diffusion Model for Dynamic Task Decomposition (C$\text{D}^\text{3}$T), a novel two-level hierarchical MARL framework designed to automatically infer subtask and coordination patterns. The high-level policy learns subtask representation to generate a subtask selection strategy based on subtask effects. To capture the effects of subtasks on the environment, C$\text{D}^\text{3}$T predicts the next observation and reward using a conditional diffusion model. At the low level, agents collaboratively learn and share specialized skills within their assigned subtasks. Moreover, the learned subtask representation is also used as additional semantic information in a multi-head attention mixing network to enhance value decomposition and provide an efficient reasoning bridge between individual and joint value functions. Experimental results on various benchmarks demonstrate that C$\text{D}^\text{3}$T achieves better performance than existing baselines.

---

## 67. Synthetic Forgetting without Access: A Few-shot Zero-glance Framework for Machine Unlearning

**论文链接:** [http://arxiv.org/abs/2511.13116v1](http://arxiv.org/abs/2511.13116v1)

**作者:** Qipeng Song, Nan Yang, Ziqi Xu, Yue Li, Wei Shao, Feng Xia

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了GFOES框架，用于在数据受限条件下实现有效的机器遗忘，无需访问原始遗忘数据即可消除特定数据的影响，同时保持模型性能。

### 背景

大多数现有机器遗忘方法假设可以完全访问原始训练数据集，这在实践中往往不切实际。

### 目的

解决少镜头零瞥视(few-shot zero-glance)场景下的机器遗忘问题，即只有少量保留数据可用，而遗忘集完全无法访问。

### 方法

提出GFOES框架，包含生成反馈网络(GFN)和两阶段微调程序。GFN合成最佳擦除样本(OES)，在目标类别上诱导高损失，使模型能够遗忘特定类别的知识；两阶段微调程序先进行积极遗忘，再恢复模型效用。

### 主要发现

在三个图像分类数据集上的实验表明，GFOES在logit和表示层面都实现了有效的遗忘，同时仅使用5%的原始数据就保持了强大的性能。

### 结论

GFOES框架为数据受限条件下的隐私保护机器学习提供了实用且可扩展的解决方案。

### 翻译

机器遗忘旨在消除特定数据对已训练模型的影响以确保隐私合规。然而，大多数现有方法假设可以完全访问原始训练数据集，这通常不切实际。我们解决了一个更现实且具有挑战性的场景：少镜头零瞥视，其中只有一小部分保留数据可用，而遗忘集完全无法访问。我们引入了GFOES，一个包含生成反馈网络(GFN)和两阶段微调程序的新框架。GFN合成最佳擦除样本(OES)，这些样本在目标类别上诱导高损失，使模型能够在不访问原始遗忘数据的情况下遗忘特定类别的知识，同时保留对保留类别的性能。两阶段微调程序允许在第一阶段进行积极遗忘，在第二阶段恢复效用。在三个图像分类数据集上的实验表明，GFOES在logit和表示层面都实现了有效的遗忘，同时仅使用5%的原始数据就保持了强大的性能。我们的框架为数据受限条件下的隐私保护机器学习提供了实用且可扩展的解决方案。


### 论文摘要

Machine unlearning aims to eliminate the influence of specific data from trained models to ensure privacy compliance. However, most existing methods assume full access to the original training dataset, which is often impractical. We address a more realistic yet challenging setting: few-shot zero-glance, where only a small subset of the retained data is available and the forget set is entirely inaccessible. We introduce GFOES, a novel framework comprising a Generative Feedback Network (GFN) and a two-phase fine-tuning procedure. GFN synthesises Optimal Erasure Samples (OES), which induce high loss on target classes, enabling the model to forget class-specific knowledge without access to the original forget data, while preserving performance on retained classes. The two-phase fine-tuning procedure enables aggressive forgetting in the first phase, followed by utility restoration in the second. Experiments on three image classification datasets demonstrate that GFOES achieves effective forgetting at both logit and representation levels, while maintaining strong performance using only 5% of the original data. Our framework offers a practical and scalable solution for privacy-preserving machine learning under data-constrained conditions.

---

## 68. Learning Implicit Neural Degradation Representation for Unpaired Image Dehazing

**论文链接:** [http://arxiv.org/abs/2511.13110v1](http://arxiv.org/abs/2511.13110v1)

**作者:** Shuaibin Fan, Senming Zhong, Wenchao Yan, Minglong Xue

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文提出了一种基于隐式神经退化表示的无监督图像去雾方法，旨在解决复杂场景中现有方法难以平衡非均匀雾分布的细粒度特征表示和全局一致性建模的问题。

### 背景

图像去雾是计算机视觉领域的重要任务，旨在从受雾影响的图像中恢复清晰且细节丰富的视觉内容。然而，现有方法在处理复杂场景时，难以平衡非均匀雾分布的细粒度特征表示和全局一致性建模。

### 目的

更好地学习空间变化中雾的通用退化表示，提出一种无监督去雾方法用于隐式神经退化表示。

### 方法

受Kolmogorov-Arnold表示定理启发，提出结合通道独立和通道相关机制的机制，增强学习非线性依赖的能力；设计隐式神经表示将雾退化建模为连续函数；设计密集残差增强模块消除冗余信息。

### 主要发现

提出的方法能在复杂场景中实现良好的视觉感知，并实现高质量的图像恢复。

### 结论

实验结果表明，该方法在各种公共和真实世界数据集上实现了具有竞争力的去雾性能，项目代码将在https://github.com/Fan-pixel/NeDR-Dehaze上提供。

### 翻译

图像去雾是计算机视觉领域的重要任务，旨在从受雾影响的图像中恢复清晰且细节丰富的视觉内容。然而，在处理复杂场景时，现有方法往往难以平衡非均匀雾分布的细粒度特征表示和全局一致性建模。此外，为了更好地学习空间变化中雾的通用退化表示，我们提出了一种用于隐式神经退化表示的无监督去雾方法。首先，受Kolmogorov-Arnold表示定理启发，我们提出了一种结合通道独立和通道相关机制的机制，有效增强了学习非线性依赖的能力，从而在复杂场景中实现了良好的视觉感知。此外，我们设计了一种隐式神经表示，将雾退化建模为连续函数，以消除冗余信息和对显式特征提取及物理模型的依赖。为了进一步学习雾特征的隐式表示，我们还从中设计了一个密集残差增强模块来消除冗余信息。这实现了高质量的图像恢复。实验结果表明，我们的方法在各种公共和真实世界数据集上实现了具有竞争力的去雾性能。本项目代码将在https://github.com/Fan-pixel/NeDR-Dehaze上提供。


### 论文摘要

Image dehazing is an important task in the field of computer vision, aiming at restoring clear and detail-rich visual content from haze-affected images. However, when dealing with complex scenes, existing methods often struggle to strike a balance between fine-grained feature representation of inhomogeneous haze distribution and global consistency modeling. Furthermore, to better learn the common degenerate representation of haze in spatial variations, we propose an unsupervised dehaze method for implicit neural degradation representation. Firstly, inspired by the Kolmogorov-Arnold representation theorem, we propose a mechanism combining the channel-independent and channel-dependent mechanisms, which efficiently enhances the ability to learn from nonlinear dependencies. which in turn achieves good visual perception in complex scenes. Moreover, we design an implicit neural representation to model haze degradation as a continuous function to eliminate redundant information and the dependence on explicit feature extraction and physical models. To further learn the implicit representation of the haze features, we also designed a dense residual enhancement module from it to eliminate redundant information. This achieves high-quality image restoration. Experimental results show that our method achieves competitive dehaze performance on various public and real-world datasets. This project code will be available at https://github.com/Fan-pixel/NeDR-Dehaze.

---

## 69. FGNet: Leveraging Feature-Guided Attention to Refine SAM2 for 3D EM Neuron Segmentation

**论文链接:** [http://arxiv.org/abs/2511.13063v1](http://arxiv.org/abs/2511.13063v1)

**作者:** Zhenghua Li, Hang Chen, Zihao Sun, Kai Li, Xiaolin Hu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了一种新框架，利用预训练在自然图像上的Segment Anything 2模型来提升电子显微镜图像中神经结构的分割性能。

### 背景

神经结构在电子显微镜图像中的精确分割对神经科学至关重要，但现有方法面临复杂形态、低信噪比和标注稀缺的挑战，限制了准确性和泛化能力。

### 目的

利用视觉基础模型在大量自然图像上学习到的先验知识来解决电子显微镜图像中神经结构分割的挑战。

### 方法

使用SAM2提取通用特征，引入特征引导注意力模块利用SAM2的语义线索指导轻量级编码器关注困难区域，并使用双亲和度解码器生成粗略和精细的亲和度图。

### 主要发现

在SAM2权重冻结的情况下，方法性能与最先进方法相当；在EM数据上进一步微调后，方法显著优于现有最先进方法。

### 结论

将自然图像上预训练的表示与目标域自适应指导相结合，可以有效解决神经元分割中的特定挑战。

### 翻译

电子显微镜图像中神经结构的精确分割对神经科学至关重要。然而，由于复杂的形态、低信噪比和稀少的标注，这一任务面临挑战，限制了现有方法的准确性和泛化能力。为应对这些挑战，我们试图利用视觉基础模型在大量自然图像上学习到的先验知识来更好地处理这一任务。具体而言，我们提出了一种新框架，可以有效地将预训练在自然图像上的Segment Anything 2的知识迁移到EM领域。我们首先使用SAM2提取强大、通用的特征。为了弥合域差距，我们引入了一个特征引导注意力模块，利用来自SAM2的语义线索来指导一个轻量级编码器，即细粒度编码器(FGE)，使其关注这些困难区域。最后，双亲和度解码器生成粗略和精细的亲和度图。实验结果表明，在SAM2权重冻结的情况下，我们的方法实现了与最先进方法相当的性能。在EM数据上进一步微调后，我们的方法显著优于现有的最先进方法。这项研究验证了，将自然图像上预训练的表示与目标域自适应指导相结合，可以有效解决神经元分割中的特定挑战。


### 论文摘要

Accurate segmentation of neural structures in Electron Microscopy (EM) images is paramount for neuroscience. However, this task is challenged by intricate morphologies, low signal-to-noise ratios, and scarce annotations, limiting the accuracy and generalization of existing methods. To address these challenges, we seek to leverage the priors learned by visual foundation models on a vast amount of natural images to better tackle this task. Specifically, we propose a novel framework that can effectively transfer knowledge from Segment Anything 2 (SAM2), which is pre-trained on natural images, to the EM domain. We first use SAM2 to extract powerful, general-purpose features. To bridge the domain gap, we introduce a Feature-Guided Attention module that leverages semantic cues from SAM2 to guide a lightweight encoder, the Fine-Grained Encoder (FGE), in focusing on these challenging regions. Finally, a dual-affinity decoder generates both coarse and refined affinity maps. Experimental results demonstrate that our method achieves performance comparable to state-of-the-art (SOTA) approaches with the SAM2 weights frozen. Upon further fine-tuning on EM data, our method significantly outperforms existing SOTA methods. This study validates that transferring representations pre-trained on natural images, when combined with targeted domain-adaptive guidance, can effectively address the specific challenges in neuron segmentation.

---

## 70. Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement

**论文链接:** [http://arxiv.org/abs/2511.11702v1](http://arxiv.org/abs/2511.11702v1)

**作者:** Lian He, Meng Liu, Qilang Ye, Yu Zhou, Xiang Deng, Gangyi Ding

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种名为TASA的任务感知3D场景级可供性分割框架，通过联合利用2D语义线索和3D几何推理，有效解决了现有方法在处理3D场景可供性分割时的局限性，显著提升了准确性和效率。

### 背景

从自然语言指令理解3D场景级可供性对于智能体在复杂环境中有意义地交互至关重要，但这一任务具有挑战性，需要语义推理和空间定位能力。

### 目的

开发一种能够有效利用点云中丰富几何结构信息的高效3D场景级可供性分割方法，克服现有方法的计算成本高和几何信息利用不足的问题。

### 方法

TASA框架采用由粗到细的方式联合利用2D语义线索和3D几何推理，包含两个主要模块：1)任务感知的2D可供性检测模块，从语言和视觉输入中识别可操作点，指导选择任务相关视图；2)3D可供性细化模块，将2D语义先验与局部3D几何信息集成，产生准确且空间连贯的3D可供性掩码。

### 主要发现

在SceneFun3D数据集上的实验表明，TASA在场景级可供性分割的准确性和效率上都显著优于现有基线方法。

### 结论

TASA框架通过结合2D语义感知和3D几何推理，有效解决了现有方法在处理3D场景级可供性分割时的局限性，为智能体在复杂环境中的交互提供了更有效的解决方案。

### 翻译

从自然语言指令理解3D场景级可供性对于使智能体能够在复杂环境中有意义地交互至关重要。然而，由于需要语义推理和空间定位，这一任务仍然具有挑战性。现有方法主要关注物体级可供性或仅将2D预测提升到3D，忽略了点云中丰富的几何结构信息并导致高计算成本。为解决这些局限性，我们引入了任务感知的3D场景级可供性分割(TASA)，这是一种新颖的几何优化框架，以由粗到细的方式联合利用2D语义线索和3D几何推理。为提高可供性检测效率，TASA具有一个任务感知的2D可供性检测模块，从语言和视觉输入中识别可操作点，指导选择任务相关视图。为充分利用3D几何信息，提出了一个3D可供性细化模块，将2D语义先验与局部3D几何相结合，产生准确且空间连贯的3D可供性掩码。在SceneFun3D上的实验表明，TASA在场景级可供性分割的准确性和效率上都显著优于基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决3D场景中的可操作区域分割问题，即让智能体能理解物体功能属性(如把手、旋钮)并根据自然语言指令与物理世界交互。这个问题在现实中非常重要，因为它能帮助机器人等智能体在复杂环境中执行任务，如'打开底部抽屉'需要识别正确的把手，这需要常识推理和细粒度空间感知能力。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考时认识到现有方法有三大局限：只关注单一物体、过度依赖2D转3D、多视图处理效率低。因此设计时考虑三点：超越单一物体范式直接处理全场景、提高多视图处理效率、减少对2D投影依赖。方法借鉴了视觉语言模型(VLMs)用于跨模态推理、CLIP模型用于文本图像嵌入、Point Transformer处理点云数据等现有工作，但进行了创新组合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是采用从粗到细的框架，联合利用2D语义线索和3D几何推理。整体流程分两阶段：1)任务感知的可操作区域检测：用VLM提取任务概念，通过CLIP选择相关视图，双重检查验证可操作点；2)3D可操作区域精细化：将2D掩膜投影到3D，再用Point Transformer架构融合几何特征进行精细化处理，生成精确的空间连贯掩膜。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个几何优化的3D场景级可操作性分割框架TASA；2)CLIP驱动的任务感知帧选择策略，大幅减少冗余视图；3)3D可操作区域精细化模块，融合2D语义与3D几何。相比之前工作(特别是Fun3DU)，不同之处在于：不再简单提升2D到3D，而是利用3D几何信息；通过任务感知帧选择提高效率；在复杂场景和高保真点云中表现更好；引入双重检查机制提高鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了TASA框架，通过结合2D语义引导和3D几何优化，实现了高效且准确的3D场景级可操作性分割，显著提升了智能体在复杂环境中根据自然语言指令进行交互的能力。'}


### 论文摘要

Understanding 3D scene-level affordances from natural language instructions is essential for enabling embodied agents to interact meaningfully in complex environments. However, this task remains challenging due to the need for semantic reasoning and spatial grounding. Existing methods mainly focus on object-level affordances or merely lift 2D predictions to 3D, neglecting rich geometric structure information in point clouds and incurring high computational costs. To address these limitations, we introduce Task-Aware 3D Scene-level Affordance segmentation (TASA), a novel geometry-optimized framework that jointly leverages 2D semantic cues and 3D geometric reasoning in a coarse-to-fine manner. To improve the affordance detection efficiency, TASA features a task-aware 2D affordance detection module to identify manipulable points from language and visual inputs, guiding the selection of task-relevant views. To fully exploit 3D geometric information, a 3D affordance refinement module is proposed to integrate 2D semantic priors with local 3D geometry, resulting in accurate and spatially coherent 3D affordance masks. Experiments on SceneFun3D demonstrate that TASA significantly outperforms the baselines in both accuracy and efficiency in scene-level affordance segmentation.

---

## 71. EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios with Language Guidance

**论文链接:** [http://arxiv.org/abs/2511.11700v1](http://arxiv.org/abs/2511.11700v1)

**作者:** Jiahui Wang, Haiyue Zhu, Haoren Guo, Abdullah Al Mamun, Cheng Xiang, Tong Heng Lee

**发布时间:** 2025-11-12

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种名为EPSegFZ的无需预训练网络，用于少样本和零样本场景下的高效点云语义分割，包含三个关键组件：原型增强的寄存器注意力模块、基于双相对位置编码的交叉注意力机制和语言引导的原型嵌入模块，在S3DIS和ScanNet基准测试上分别比最先进方法提高了5.68%和3.82%的性能。

### 背景

现有的少样本3D点云语义分割方法通常需要两阶段学习（预训练阶段+少样本训练阶段），这些方法过度依赖预训练，限制了模型的灵活性和适应性。一些尝试避免预训练的模型未能捕获足够的信息。此外，当前方法只关注支持集中的视觉信息，忽略了文本注释等其他有用数据。

### 目的

解决现有方法的限制，提出一种无需预训练的网络，用于少样本和零样本场景下的高效点云语义分割。

### 方法

名为EPSegFZ的网络，包含三个关键组件：1) 原型增强的寄存器注意力(ProERA)模块和基于双相对位置编码(DRPE)的交叉注意力机制，用于改进特征提取和准确的查询-原型对应关系构建，无需预训练；2) 语言引导的原型嵌入(LGPE)模块，有效利用支持集中的文本信息，提高少样本性能并实现零样本推理。

### 主要发现

在S3DIS和ScanNet基准测试上，该方法分别比最先进的方法提高了5.68%和3.82%的性能。

### 结论

提出的EPSegFZ方法在少样本和零样本场景下表现优异，无需预训练，且能有效利用文本信息。

### 翻译

最近用于少样本3D点云语义分割的方法通常需要两阶段学习过程，即预训练阶段后跟少样本训练阶段。虽然有效，但这些方法面临过度依赖预训练的问题，这阻碍了模型的灵活性和适应性。一些模型尝试避免预训练但未能捕获足够的信息。此外，当前方法只关注支持集中的视觉信息，而忽略或未充分利用其他有用数据，如文本注释。这种支持信息的不充分利用损害了模型的性能，限制了其零样本能力。为解决这些限制，我们提出了一种新颖的无需预训练的网络，命名为用于少样本和零样本场景的高效点云语义分割。我们的EPSegFZ包含三个关键组件。一个原型增强的寄存器注意力(ProERA)模块和一个基于双相对位置编码(DRPE)的交叉注意力机制，用于改进特征提取和准确的查询-原型对应关系构建，无需预训练。一个语言引导的原型嵌入(LGPE)模块，有效利用来自支持集的文本信息，提高少样本性能并实现零样本推理。大量实验表明，我们的方法在S3DIS和ScanNet基准测试上分别比最先进的方法提高了5.68%和3.82%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云语义分割中的少样本和零样本场景问题。现有方法过度依赖预训练，限制了模型的灵活性和适应性；同时忽略了文本等有用信息，导致模型性能受限。这个问题在现实中很重要，因为3D点云数据标注成本高，预训练资源密集，而实际应用中经常需要模型快速适应新类别且无需大量标注数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，特别是过度依赖预训练和未充分利用文本信息的问题。针对高频信息丢失，设计了ProERA模块；针对低频信息不足，设计了LGPE模块；为了有效捕获查询-原型关系，设计了DRPE机制。作者借鉴了多原型采样技术、CLIP文本编码器和注意力机制等现有工作，但进行了创新性改进，使其更适合少样本和零样本场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是不依赖预训练，同时利用高频和低频信息，结合视觉和文本信息，通过改进的注意力机制和位置编码建立查询与原型之间的准确对应关系。整体流程：1)使用DGCNN将点云映射到潜在空间；2)应用多原型采样提取原型；3)使用ProERA模块处理特征捕获高频细节；4)使用LGPE模块用文本嵌入更新原型；5)使用DRPE模块计算查询与原型间的相关性；6)通过点积相似度生成预测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出无需预训练的EPSegFZ框架；2)开发ProERA模块增强高频信息并减轻前景-背景不平衡；3)提出LGPE模块利用文本数据更新原型并实现零样本推理；4)设计基于DRPE的交叉注意力准确建立对应关系。相比之前的工作，该方法不依赖预训练，同时利用高低频信息，集成文本信息增强表示，并通过DRPE高效捕获查询-原型关系，而其他方法或需要大量参数或计算开销大。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'EPSegFZ通过结合高频特征提取、语言引导的原型嵌入和双相对位置编码，实现了一种无需预训练的高效点云语义分割方法，显著提升了少样本和零样本场景下的分割性能。'}


### 论文摘要

Recent approaches for few-shot 3D point cloud semantic segmentation typically require a two-stage learning process, i.e., a pre-training stage followed by a few-shot training stage. While effective, these methods face overreliance on pre-training, which hinders model flexibility and adaptability. Some models tried to avoid pre-training yet failed to capture ample information. In addition, current approaches focus on visual information in the support set and neglect or do not fully exploit other useful data, such as textual annotations. This inadequate utilization of support information impairs the performance of the model and restricts its zero-shot ability. To address these limitations, we present a novel pre-training-free network, named Efficient Point Cloud Semantic Segmentation for Few- and Zero-shot scenarios. Our EPSegFZ incorporates three key components. A Prototype-Enhanced Registers Attention (ProERA) module and a Dual Relative Positional Encoding (DRPE)-based cross-attention mechanism for improved feature extraction and accurate query-prototype correspondence construction without pre-training. A Language-Guided Prototype Embedding (LGPE) module that effectively leverages textual information from the support set to improve few-shot performance and enable zero-shot inference. Extensive experiments show that our method outperforms the state-of-the-art method by 5.68% and 3.82% on the S3DIS and ScanNet benchmarks, respectively.

---

## 72. 论文ID: 2511.13720v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.13720v1.json'

---

## 73. Scaling Spatial Intelligence with Multimodal Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.13719v1](http://arxiv.org/abs/2511.13719v1)

**作者:** Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, Tongxi Zhou, Jiaqi Li, Hui En Pang, Oscar Qian, Yukun Wei, Zhiqian Lin, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Xiangyu Fan, Hanming Deng, Lewei Lu, Liang Pan, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, Lei Yang

**发布时间:** 2025-11-17

**备注:** Model: https://huggingface.co/collections/sensenova/sensenova-si; Code: https://github.com/OpenSenseNova/SenseNova-SI

### GPT解析

### 总结

本论文介绍了SenseNova-SI多模态基础模型家族，通过系统性地收集八百万个多样化数据样本，在空间智能方面取得了显著性能提升，并在多个基准测试中达到前所未有的水平。

### 背景

尽管多模态基础模型取得了显著进展，但在空间智能方面仍表现出明显的不足。

### 目的

探索扩大多模态基础模型的规模，以培养空间智能能力。

### 方法

构建基于现有多模态基础模型（包括视觉理解模型Qwen3-VL和InternVL3，以及统一理解和生成模型Bagel）的SenseNova-SI家族；采用系统化方法构建高性能和鲁棒的空间智能；精心策划八百万个在空间能力严格分类下的多样化数据样本。

### 主要发现

SenseNova-SI在多个空间智能基准测试中表现出色：VSI-Bench上68.7%，MMSI上43.3%，MindCube上85.6%，ViewSpatial上54.6%，SITE上50.1%，同时保持强大的通用多模态理解能力（MMBench-En上84.9%）；分析了数据扩展的影响；讨论了多样化数据训练带来的早期泛化能力迹象；分析了过拟合和语言捷径的风险；提出了空间思维链推理的初步研究；验证了潜在的下游应用。

### 结论

SenseNova-SI是一个持续进行的项目，本报告将不断更新。所有新训练的多模态基础模型都已公开发布，以促进该方向的进一步研究。

### 翻译

尽管取得了显著进展，多模态基础模型在空间智能方面仍表现出令人惊讶的不足。在这项工作中，我们探索扩大多模态基础模型的规模，在基于已建立的多模态基础（包括视觉理解模型即Qwen3-VL和InternVL3，以及统一理解和生成模型即Bagel）构建的SenseNova-SI家族中培养空间智能。我们采用系统化的方法构建高性能和鲁棒的空间智能，通过系统性地策划SenseNova-SI-8M：在空间能力的严格分类下的八百万个多样化数据样本。SenseNova-SI在广泛的空间智能基准测试中表现出前所未有的性能：VSI-Bench上68.7%，MMSI上43.3%，MindCube上85.6%，ViewSpatial上54.6%，SITE上50.1%，同时保持强大的通用多模态理解能力（例如MMBench-En上84.9%）。更重要的是，我们分析了数据扩展的影响，讨论了多样化数据训练带来的早期泛化能力迹象，分析了过拟合和语言捷径的风险，提出了空间思维链推理的初步研究，并验证了潜在的下游应用。SenseNova-SI是一个正在进行的项目，本报告将不断更新。所有新训练的多模态基础模型都已公开发布，以促进该方向的进一步研究。


### 论文摘要

Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.7% on VSI-Bench, 43.3% on MMSI, 85.6% on MindCube, 54.6% on ViewSpatial, and 50.1% on SITE, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. SenseNova-SI is an ongoing project, and this report will be updated continuously. All newly trained multimodal foundation models are publicly released to facilitate further research in this direction.

---

## 74. TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models

**论文链接:** [http://arxiv.org/abs/2511.13704v1](http://arxiv.org/abs/2511.13704v1)

**作者:** Harold Haodong Chen, Disen Lan, Wen-Jie Shu, Qingyang Liu, Zihan Wang, Sirui Chen, Wenkai Cheng, Kanghao Chen, Hongfei Zhang, Zixin Zhang, Rongjin Guo, Yu Cheng, Ying-Cong Chen

**发布时间:** 2025-11-17

**备注:** Project: https://haroldchen19.github.io/TiViBench-Page/

### GPT解析

### 总结

本研究提出TiViBench基准测试和VideoTPO策略，用于评估和提升视频生成模型的推理能力，填补了现有评估方法无法捕捉高阶推理能力的空白。

### 背景

视频生成模型已从注重视觉合理性发展到需要物理合理性和逻辑一致性的任务，但现有基准测试主要评估视觉保真度和时间一致性，无法捕捉更高层次的推理能力。

### 目的

开发专门评估图像到视频生成模型推理能力的基准测试，并探索提升模型推理性能的方法。

### 方法

提出TiViBench分层基准测试，从四个维度评估推理能力：结构推理与搜索、空间与视觉模式推理、符号与逻辑推理、动作规划与任务执行；引入VideoTPO测试时策略，通过LLM自我分析生成候选来识别优缺点。

### 主要发现

商业模型(如Sora 2、Veo 3.1)展现出更强的推理潜力，开源模型则有未开发的潜力但受限于训练规模和数据多样性；VideoTPO策略无需额外训练即可显著提升推理性能。

### 结论

TiViBench和VideoTPO为评估和提升视频生成模型的推理能力奠定了基础，推动了这一新兴领域的研究发展。

### 翻译

视频生成模型的快速发展已使其重点从产生视觉上合理的输转向需要物理合理性和逻辑一致性的任务。然而，尽管Veo 3的帧链推理等近期突破，这些模型是否表现出类似于大型语言模型(LLMs)的推理能力仍不清楚。现有基准测试主要评估视觉保真度和时间一致性，无法捕捉更高层次的推理能力。为填补这一空白，我们提出了TiViBench，一个专门设计用于评估图像到视频(I2V)生成模型推理能力的分层基准测试。TiViBench系统地评估四个维度的推理能力：i)结构推理与搜索，ii)空间与视觉模式推理，iii)符号与逻辑推理，以及iv)动作规划与任务执行，涵盖3个难度级别的24个多样化任务场景。通过广泛评估，我们表明商业模型(如Sora 2、Veo 3.1)展现出更强的推理潜力，而开源模型则显示出尚未开发的潜力，但这种潜力受到有限训练规模和数据多样性的阻碍。为进一步释放这种潜力，我们引入了VideoTPO，一种受偏好优化启发的简单而有效的测试时策略。通过对生成的候选进行LLM自我分析来识别优势和劣势，VideoTPO显著提高了推理性能，而无需额外的训练、数据或奖励模型。TiViBench和VideoTPO共同为评估和提升视频生成模型的推理能力铺平了道路，为这一新兴领域的未来研究奠定了基础。


### 论文摘要

The rapid evolution of video generative models has shifted their focus from producing visually plausible outputs to tackling tasks requiring physical plausibility and logical consistency. However, despite recent breakthroughs such as Veo 3's chain-of-frames reasoning, it remains unclear whether these models can exhibit reasoning capabilities similar to large language models (LLMs). Existing benchmarks predominantly evaluate visual fidelity and temporal coherence, failing to capture higher-order reasoning abilities. To bridge this gap, we propose TiViBench, a hierarchical benchmark specifically designed to evaluate the reasoning capabilities of image-to-video (I2V) generation models. TiViBench systematically assesses reasoning across four dimensions: i) Structural Reasoning & Search, ii) Spatial & Visual Pattern Reasoning, iii) Symbolic & Logical Reasoning, and iv) Action Planning & Task Execution, spanning 24 diverse task scenarios across 3 difficulty levels. Through extensive evaluations, we show that commercial models (e.g., Sora 2, Veo 3.1) demonstrate stronger reasoning potential, while open-source models reveal untapped potential that remains hindered by limited training scale and data diversity. To further unlock this potential, we introduce VideoTPO, a simple yet effective test-time strategy inspired by preference optimization. By performing LLM self-analysis on generated candidates to identify strengths and weaknesses, VideoTPO significantly enhances reasoning performance without requiring additional training, data, or reward models. Together, TiViBench and VideoTPO pave the way for evaluating and advancing reasoning in video generation models, setting a foundation for future research in this emerging field.

---

## 75. Generalist Foundation Models Are Not Clinical Enough for Hospital Operations

**论文链接:** [http://arxiv.org/abs/2511.13703v1](http://arxiv.org/abs/2511.13703v1)

**作者:** Lavender Y. Jiang, Angelica Chen, Xu Han, Xujin Chris Liu, Radhika Dua, Kevin Eaton, Frederick Wolff, Robert Steele, Jeff Zhang, Anton Alyakin, Qingkai Pan, Yanbing Chen, Karl L. Sangwon, Daniel A. Alber, Jaden Stryker, Jin Vivian Lee, Yindalon Aphinyanaphongs, Kyunghyun Cho, Eric Karl Oermann

**发布时间:** 2025-11-17

### GPT解析

### 总结

研究团队开发了Lang1医疗专用语言模型，通过在临床记录和互联网文本混合语料库上预训练，并在真实医疗场景基准测试中验证其性能，发现经过微调的Lang1-1B模型显著优于通用模型。

### 背景

医院和医疗系统依赖运营决策决定患者流动、成本和护理质量，但现有基础模型缺乏医疗运营决策所需的专门知识。

### 目的

开发并评估专门针对医疗领域的语言模型Lang1，以解决医疗运营决策中的预测问题。

### 方法

创建Lang1模型家族(100M-7B参数)，在混合800亿临床标记和6270亿互联网标记的专业语料库上预训练；开发ReMedE基准，基于668,331份电子健康记录笔记评估五个关键医疗任务。

### 主要发现

零样本下模型表现不佳(AUROC 36.6%-71.7%)；微调后Lang1-1B性能超越70倍大的微调通用模型和671倍大的零样本模型(AUROC提升3.64%-23.66%)；多任务联合微调产生跨任务扩展效应；模型能有效迁移到分布外场景。

### 结论

医疗运营预测需明确的监督微调；电子健康记录领域预训练可提高微调效率；专用大模型可在专业任务中竞争；有效医疗AI需结合领域预训练、监督微调和真实世界评估。

### 翻译

医院和医疗系统依赖决定患者流动、成本和护理质量的运营决策。尽管在医学知识和对话基准测试中表现出色，但基于通用文本训练的基础模型可能缺乏这些运营决策所需的专门知识。我们介绍了Lang1，一个模型家族(100M-7B参数)，在混合了纽约大学朗格尼健康中心电子健康记录的800亿临床标记和互联网的6270亿标记的专业语料库上进行了预训练。为在真实环境中严格评估Lang1，我们开发了REalistic Medical Evaluation (ReMedE)，一个基于668,331份电子健康记录笔记的基准，评估五个关键任务：30天再入院预测、30天死亡率预测、住院时间、合并症编码和预测保险索赔拒绝。在零样本设置下，通用模型和专用模型在五个任务中的四个表现不佳(AUROC 36.6%-71.7%)，死亡率预测是例外。微调后，Lang1-1B优于微调后的通用模型，其规模最大可达前者的70倍，优于零样本设置下的模型，规模最大可达671倍，AUROC分别提高了3.64%-6.75%和1.66%-23.66%。我们还观察到多任务联合微调可带来跨任务扩展效应。Lang1-1B能有效迁移到分布外设置，包括其他临床任务和外部医疗系统。我们的研究结果表明，医院运营的预测能力需要明确的监督微调，而这种微调过程通过在电子健康记录领域内预训练变得更加高效。我们的发现支持新兴观点，即专用大模型可以在专门任务中与通用模型竞争，并表明有效的医疗系统AI需要结合领域内预训练、监督微调和超越代理基准的真实世界评估。


### 论文摘要

Hospitals and healthcare systems rely on operational decisions that determine patient flow, cost, and quality of care. Despite strong performance on medical knowledge and conversational benchmarks, foundation models trained on general text may lack the specialized knowledge required for these operational decisions. We introduce Lang1, a family of models (100M-7B parameters) pretrained on a specialized corpus blending 80B clinical tokens from NYU Langone Health's EHRs and 627B tokens from the internet. To rigorously evaluate Lang1 in real-world settings, we developed the REalistic Medical Evaluation (ReMedE), a benchmark derived from 668,331 EHR notes that evaluates five critical tasks: 30-day readmission prediction, 30-day mortality prediction, length of stay, comorbidity coding, and predicting insurance claims denial. In zero-shot settings, both general-purpose and specialized models underperform on four of five tasks (36.6%-71.7% AUROC), with mortality prediction being an exception. After finetuning, Lang1-1B outperforms finetuned generalist models up to 70x larger and zero-shot models up to 671x larger, improving AUROC by 3.64%-6.75% and 1.66%-23.66% respectively. We also observed cross-task scaling with joint finetuning on multiple tasks leading to improvement on other tasks. Lang1-1B effectively transfers to out-of-distribution settings, including other clinical tasks and an external health system. Our findings suggest that predictive capabilities for hospital operations require explicit supervised finetuning, and that this finetuning process is made more efficient by in-domain pretraining on EHR. Our findings support the emerging view that specialized LLMs can compete with generalist models in specialized tasks, and show that effective healthcare systems AI requires the combination of in-domain pretraining, supervised finetuning, and real-world evaluation beyond proxy benchmarks.

---

## 76. Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation

**论文链接:** [http://arxiv.org/abs/2511.13689v1](http://arxiv.org/abs/2511.13689v1)

**作者:** Sofia Jamil, Kotla Sai Charan, Sriparna Saha, Koustava Goswami, Joseph K J

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了翻译和图像生成（TAI）框架，利用大型语言模型和潜在扩散模型通过提示调整，提高印度语言诗歌的可访问性，支持联合国可持续发展目标中的优质教育和减少不平等目标。

### 背景

印度诗歌拥有丰富多样的传统，跨越数千年历史，但其语言复杂性、文化典故和语法结构给非母语者带来理解挑战。尽管印度诗歌具有重要文化意义，现有研究大多忽视了印度语言诗歌。

### 目的

通过提高丰富的印度语言诗歌对全球观众的可达性，支持联合国的优质教育（SDG 4）和减少不平等（SDG 10）可持续发展目标。

### 方法

TAI框架包括翻译模块（使用赔率比偏好对齐算法将形态丰富的诗歌翻译成英语）和图像生成模块（使用语义图捕获标记、依赖关系和语义关系）。还引入了包含21种低资源印度语言1570首诗歌的MorphoVerse数据集。

### 主要发现

全面实验评估（包括人类和定量评估）证明TAI扩散在诗歌图像生成任务上表现优异，超越了强大的基线模型。

### 结论

通过解决诗歌翻译和视觉理解方面的差距，该工作旨在扩大可及性并丰富读者的体验。

### 翻译

本文提出的翻译和图像生成（TAI）框架利用大型语言模型和潜在扩散模型，通过赔率比偏好对齐算法和语义图技术，将形态丰富的印度语言诗歌翻译成英语并生成有意义的视觉表示，同时创建了包含21种低资源印度语言1570首诗歌的MorphoVerse数据集。


### 论文摘要

Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-native speakers or readers unfamiliar with its context and language. Despite its cultural significance, existing works on poetry have largely overlooked Indian language poems. In this paper, we propose the Translation and Image Generation (TAI) framework, leveraging Large Language Models (LLMs) and Latent Diffusion Models through appropriate prompt tuning. Our framework supports the United Nations Sustainable Development Goals of Quality Education (SDG 4) and Reduced Inequalities (SDG 10) by enhancing the accessibility of culturally rich Indian-language poetry to a global audience. It includes (1) a translation module that uses an Odds Ratio Preference Alignment Algorithm to accurately translate morphologically rich poetry into English, and (2) an image generation module that employs a semantic graph to capture tokens, dependencies, and semantic relationships between metaphors and their meanings, to create visually meaningful representations of Indian poems. Our comprehensive experimental evaluation, including both human and quantitative assessments, demonstrates the superiority of TAI Diffusion in poem image generation tasks, outperforming strong baselines. To further address the scarcity of resources for Indian-language poetry, we introduce the Morphologically Rich Indian Language Poems MorphoVerse Dataset, comprising 1,570 poems across 21 low-resource Indian languages. By addressing the gap in poetry translation and visual comprehension, this work aims to broaden accessibility and enrich the reader's experience.

---

## 77. Training-Free Multi-View Extension of IC-Light for Textual Position-Aware Scene Relighting

**论文链接:** [http://arxiv.org/abs/2511.13684v1](http://arxiv.org/abs/2511.13684v1)

**作者:** Jiangnan Ye, Jiedong Zhuang, Lianrui Mu, Wenjie Zheng, Jiaqi Hu, Xingze Zou, Jing Wang, Haoji Hu

**发布时间:** 2025-11-17

**备注:** Submitting for Neurocomputing

### GPT解析

### 总结

GS-Light是一种高效的、文本位置感知的流水线，用于基于高斯飞溅表示的3D场景的文本引导重新照明。它通过大型视觉语言模型解析用户提示，结合几何和语义估计器，生成初始潜在代码引导扩散模型，最终产生高保真度的重新照明3D场景。

### 背景

现有的3D场景重新照明方法如单视图重新照明、视频重新照明和场景编辑方法存在局限性，需要更有效的技术处理多视图一致性和用户期望的照明效果。

### 目的

开发一个高效的文本位置感知流水线，用于基于高斯飞溅表示的3D场景的文本引导重新照明，能够处理多视图输入并生成符合用户期望的高保真度结果。

### 方法

1) 实现单输入扩散模型的无训练扩展处理多视图输入；2) 使用大型视觉语言模型解析用户提示为光照先验；3) 融合光照先验与视图几何约束计算光照图；4) 生成初始潜在代码引导扩散模型；5) 通过多视图重新照明模型生成最终图像；6) 微调3DGS场景获得完全重新照明的3D场景。

### 主要发现

GS-Light在多视图一致性、图像质量、美学评分和语义相似性等定量指标上优于基线方法；用户研究定性评估也显示其优势；特别是在照明方向准确性方面表现突出。

### 结论

GS-Light是一种有效的3D场景重新照明方法，能够生成符合用户期望的高保真度结果，在多个评估指标上优于现有方法，代码和资源将在发表后提供。

### 翻译

我们引入了GS-Light，一种高效的、文本位置感知的流水线，用于基于高斯飞溅表示的3D场景的文本引导重新照明。GS-Light实现了单输入扩散模型的无训练扩展，以处理多视图输入。给定可能指定照明方向、颜色、强度或参考对象的用户提示，我们采用大型视觉语言模型将提示解析为光照先验。使用现成的几何和语义估计器，我们将这些光照先验与视图几何约束融合，计算光照图并为每个视图生成初始潜在代码。这些精心推导的初始潜在代码引导扩散模型生成更准确反映用户期望的重新照明输出。通过多视图渲染图像和初始潜在代码，我们生成高保真度、艺术性重新照明的图像。最后，我们使用重新照明的外观微调3DGS场景，获得完全重新照明的3D场景。我们在室内和室外场景上评估GS-Light，并与最先进基线方法比较，显示出一致性改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何实现文本引导的、位置感知的多视角3D场景重新光照问题。这个问题在现实和研究中很重要，因为3D场景重新光照在增强/虚拟现实、电影制作和内容创作中有广泛应用，但现有方法难以同时实现多视角一致性、文本可控性和对位置信息的准确理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性，包括IC-Light等2D模型在3D场景中的不一致性和对文本位置信息的理解不足。他们设计了GS-Light框架，结合Position-Align Module (PAM)和MV-ICLight两个关键组件。借鉴了IC-Light的2D重新光照能力、DGE的极线约束机制、LVLM的光照先验提取以及现成的几何和语义估计器，但进行了创新性整合和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过LVLM提取文本光照先验，结合场景几何和语义信息生成位置对齐的光照图，作为扩散模型的初始潜在代码，再通过多视角注意力机制确保不同视角间的一致性。流程包括：接收3D场景和文本提示→渲染多视角图像→PAM解析文本并计算光照图→MV-ICLight生成多视角一致的重新光照图像→微调3D高斯溅射场景→迭代优化确保一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) GS-Light框架，首个支持文本引导、位置感知的多视角一致的3D高斯溅射重新光照方法；2) 位置对齐模块(PAM)，解决位置信息理解不足；3) 多视角ICLight，使用极线约束确保一致性。相比之前工作，GS-Light能更好理解文本位置信息，确保多视角一致性，无需重新训练，且在质量指标上表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GS-Light是一种无需训练的高效方法，通过结合文本引导的光照先验和多视角一致性约束，实现了对3D高斯溅射场景的位置感知、多视角一致的重新光照，显著提升了重新光照结果与用户意图的对齐度和多视角间的一致性。'}


### 论文摘要

We introduce GS-Light, an efficient, textual position-aware pipeline for text-guided relighting of 3D scenes represented via Gaussian Splatting (3DGS). GS-Light implements a training-free extension of a single-input diffusion model to handle multi-view inputs. Given a user prompt that may specify lighting direction, color, intensity, or reference objects, we employ a large vision-language model (LVLM) to parse the prompt into lighting priors. Using off-the-shelf estimators for geometry and semantics (depth, surface normals, and semantic segmentation), we fuse these lighting priors with view-geometry constraints to compute illumination maps and generate initial latent codes for each view. These meticulously derived init latents guide the diffusion model to generate relighting outputs that more accurately reflect user expectations, especially in terms of lighting direction. By feeding multi-view rendered images, along with the init latents, into our multi-view relighting model, we produce high-fidelity, artistically relit images. Finally, we fine-tune the 3DGS scene with the relit appearance to obtain a fully relit 3D scene. We evaluate GS-Light on both indoor and outdoor scenes, comparing it to state-of-the-art baselines including per-view relighting, video relighting, and scene editing methods. Using quantitative metrics (multi-view consistency, imaging quality, aesthetic score, semantic similarity, etc.) and qualitative assessment (user studies), GS-Light demonstrates consistent improvements over baselines. Code and assets will be made available upon publication.

---

## 78. Gravitational--Electromagnetic Coupling on Kerr Spacetime

**论文链接:** [http://arxiv.org/abs/2511.13642v1](http://arxiv.org/abs/2511.13642v1)

**作者:** Fawzi Aly, Dejan Stojkovic

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究将基于度量的施瓦西黑洞引力-电磁耦合研究扩展到旋转黑洞，在克尔时空的曲率基础上使用Newman-Penrose/Teukolsky框架，推导出爱因斯坦-麦克斯韦系统中自旋-2 Teukolsky方程的显式二次电磁源项，为黑洞谱框架中的GEM相互作用数值研究奠定基础。

### 背景

先前的研究已经基于度量方法研究了施瓦西黑洞的引力-电磁耦合，但尚未在旋转黑洞背景下进行类似研究。

### 目的

在爱因斯坦-麦克斯韦最小耦合系统中，推导自旋-2 Teukolsky方程的显式二次电磁源项，为黑洞谱框架中的GEM相互作用数值研究提供基础。

### 方法

在克尔时空中直接使用基于曲率的Newman-Penrose/Teukolsky框架进行研究。

### 主要发现

1) 提供了自旋-2 Teukolsky方程的显式二次电磁源项；2) 通过数量级论证表明GEM二次准正规模式在多种带电和磁化天体物理场景中可能变得相关；3) 通过dilaton理论例子表明GEM QQNM谱对引力与电磁耦合的方式敏感。

### 结论

GEM QQNM谱提供了一种基于模型的方法来测试最小耦合假设，并通过引力波观测来限制隐藏的U(1)能级。

### 翻译

我们将先前基于度量的施瓦西黑洞引力-电磁耦合研究扩展到旋转黑洞，通过在克尔时空中直接使用基于曲率的Newman-Penrose/Teukolsky框架进行研究。在爱因斯坦-麦克斯韦最小耦合系统中，我们推导出自旋-2 Teukolsky方程的显式二次电磁源项，为未来在黑洞谱框架中GEM相互作用的数值研究奠定基础。此外，我们通过数量级论证表明，在一系列带电和磁化的天体物理场景中，GEM二次准正规模式可能变得相关。最后，我们通过一个简化的dilaton理论例子表明，GEM QQNM谱对引力如何与电磁耦合敏感，从而提供了一种基于模型的方法来测试最小耦合，并通过引力波观测限制隐藏的U(1)能级。


### 论文摘要

We extend previous metric-based Schwarzschild studies of gravitational--electromagnetic (GEM) coupling to rotating black holes by working directly in a curvature-based Newman--Penrose/Teukolsky framework on Kerr spacetime. Within a minimally coupled Einstein--Maxwell system, we derive explicit quadratic electromagnetic source terms for the spin-$-2$ Teukolsky equation, providing a foundation for future numerical studies of GEM interactions in the framework of black-hole spectroscopy. Moreover, we give order-of-magnitude arguments showing that GEM quadratic quasinormal modes (QQNMs) can become relevant in a range of charged and magnetized astrophysical scenarios. Finally, we show through a brief dilaton-theory example that the GEM QQNM spectrum is sensitive to how gravity couples to electromagnetism, thereby providing a model-based way to test minimal coupling and to constrain hidden $U(1)$ sectors with gravitational-wave observations.

---

## 79. CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product

**论文链接:** [http://arxiv.org/abs/2511.13626v1](http://arxiv.org/abs/2511.13626v1)

**作者:** Kaiwen Xue, Chenglong Li, Zhonghong Ou, Guoxin Zhang, Kaoyan Lu, Shuai Lyu, Yifan Zhu, Ping Zong Junpeng Ding, Xinyu Liu, Qunlin Chen, Weiwei Qin, Yiran Shen, Jiayi Cen

**发布时间:** 2025-11-17

**备注:** 13 pages, 3 figures,The 40th Annual AAAI Conference on Artificial Intelligence(AAAI 2026),Paper has been accepted for a poster presentation

### GPT解析

### 总结

本文提出了一种名为CreBench的创造力评估基准和CreExpert模型，用于提高多模态大语言模型对人类创造力的理解和评估能力。

### 背景

人类定义的创造力非常抽象，这对多模态大语言模型理解和评估与人类判断一致的创造力构成了挑战，且现有基准的缺乏进一步加剧了这一困境。

### 目的

开发一个能够理解和评估与人类判断一致的创造力的多模态大语言模型和评估基准。

### 方法

提出CreBench评估基准，包含两个关键组成部分：1)涵盖创意想法、过程到产品多个维度的评估基准；2)包含2.2K个多模态数据、79.2K个人类反馈和470万多种类型指令的CreMIT数据集。通过提示GPT优化人类反馈以激活更强的创造力评估能力，并基于CreBench微调开源通用MLLMs，得到CreExpert模型。

### 主要发现

CreExpert模型在创造力评估方面与人类判断的显著一致性优于包括GPT-4V和Gemini-Pro-Vision在内的最先进MLLMs。

### 结论

CreBench作为构建理解人类一致创造力的MLLMs的基础，通过CreExpert模型显著提高了多模态大语言模型对创造力的评估能力。

### 翻译

人类定义的创造力高度抽象，这对多模态大语言模型理解和评估与人类判断一致的创造力构成了挑战。现有基准的缺乏进一步加剧了这一困境。为此，我们提出了CreBench，它包含两个关键组成部分：1)涵盖从创意想法到过程再到产品多个维度的评估基准；2)CreMIT（创造力多模态指令调优数据集），一个包含2.2K个多样化来源的多模态数据、79.2K个人类反馈和470万多种类型指令的多模态创造力评估数据集。具体而言，为确保MLLMs能够处理多样化的创造力相关查询，我们提示GPT优化这些人类反馈以激活更强的创造力评估能力。CreBench作为构建理解人类一致创造力的MLLMs的基础。基于CreBench，我们对开源通用MLLMs进行了微调，得到了CreExpert，一个多模态创造力评估专家模型。大量实验表明，与包括最先进的GPT-4V和Gemini-Pro-Vision在内的最先进MLLMs相比，提出的CreExpert模型与人类创造力评估的显著一致性更好。


### 论文摘要

Human-defined creativity is highly abstract, posing a challenge for multimodal large language models (MLLMs) to comprehend and assess creativity that aligns with human judgments. The absence of an existing benchmark further exacerbates this dilemma. To this end, we propose CreBench, which consists of two key components: 1) an evaluation benchmark covering the multiple dimensions from creative idea to process to products; 2) CreMIT (Creativity Multimodal Instruction Tuning dataset), a multimodal creativity evaluation dataset, consisting of 2.2K diverse-sourced multimodal data, 79.2K human feedbacks and 4.7M multi-typed instructions. Specifically, to ensure MLLMs can handle diverse creativity-related queries, we prompt GPT to refine these human feedbacks to activate stronger creativity assessment capabilities. CreBench serves as a foundation for building MLLMs that understand human-aligned creativity. Based on the CreBench, we fine-tune open-source general MLLMs, resulting in CreExpert, a multimodal creativity evaluation expert model. Extensive experiments demonstrate that the proposed CreExpert models achieve significantly better alignment with human creativity evaluation compared to state-of-the-art MLLMs, including the most advanced GPT-4V and Gemini-Pro-Vision.

---

## 80. Adaptive Multi-Scale Integration Unlocks Robust Cell Annotation in Histopathology Images

**论文链接:** [http://arxiv.org/abs/2511.13586v1](http://arxiv.org/abs/2511.13586v1)

**作者:** Yinuo Xu, Yan Cui, Mingyao Li, Zhi Huang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了NuClass框架，通过多尺度整合细胞核形态和微环境背景，实现了从常规组织病理学图像中准确识别细胞类型和亚型，解决了现有方法缺乏组织背景和高质量注释的问题。

### 背景

从常规组织病理学图像中识别细胞类型和亚型对计算理解人类疾病至关重要。现有基于瓦片的模型虽能捕获细胞核形态，但缺乏组织环境背景；同时，可用的人类注释通常是粗粒度且分布不均，难以获得细粒度的亚型级别监督。

### 目的

开发一种受病理学家工作流程启发的框架，实现细胞级别的多尺度整合，结合细胞核形态和微环境背景，提高细胞类型识别的准确性。

### 方法

NuClass包含两个主要组件：Path local(224×224像素裁剪中关注细胞核形态)和Path global(1024×1024像素邻域建模)；采用可学习门控模块平衡局部细节和上下文；引入不确定性引导目标促进互补学习；提供校准置信度估计和Grad-CAM可视化增强可解释性；利用Xenium空间转录组学构建标记数据集，提供八种器官和16个类别超过两百万细胞的单细胞分辨率标签。

### 主要发现

在三个完全保留的队列上评估，NuClass为其最佳性能类别实现了高达96%的F1分数，显著优于基线方法；多尺度、不确定性感知的融合能有效连接幻灯片级别病理基础模型与细胞级别表型预测。

### 结论

多尺度、不确定性感知的融合策略可以弥合幻灯片级别病理基础模型和可靠的细胞级别表型预测之间的差距，为计算病理学提供了新的有效方法。

### 翻译

从常规组织病理学图像中识别细胞类型和亚型对于提高计算理解人类疾病至关重要。现有的基于瓦片的模型能够捕获详细的细胞核形态，但往往未能纳入影响细胞功能和身份的更广泛组织环境背景。此外，可用的人类注释通常是粗粒度的，并且在研究中分布不均，使得难以获得细粒度的亚型级别监督。为解决这些局限性，我们引入了NuClass，这是一种受病理学家工作流程启发的框架，用于细胞级别的多尺度整合细胞核形态和微环境背景。NuClass包括两个主要组件：Path local，专注于224×224像素裁剪中的细胞核形态；以及Path global，对周围1024×1024像素的邻域进行建模。一个可学习的门控模块自适应地平衡局部细节和上下文线索。为了鼓励互补学习，我们纳入了一个不确定性引导的目标，指导全局路径优先关注局部路径不确定的区域。我们还提供校准的置信度估计和Grad-CAM可视化以增强可解释性。为克服高质量注释的缺乏，我们从Xenium空间转录组学分析中构建了一个基于标记的数据集，为八个器官和16个类别的超过两百万个细胞提供了单细胞分辨率标签。在三个完全保留的队列上评估，NuClass为其性能最佳的类别实现了高达96%的F1分数，优于强大的基线方法。我们的结果表明，多尺度、不确定性感知的融合可以弥合幻灯片级别病理基础模型和可靠的细胞级别表型预测之间的差距。


### 论文摘要

Identifying cell types and subtypes from routine histopathology images is essential for improving the computational understanding of human disease. Existing tile-based models can capture detailed nuclear morphology but often fail to incorporate the broader tissue context that influences a cell's function and identity. In addition, available human annotations are typically coarse-grained and unevenly distributed across studies, making fine-grained subtype-level supervision difficult to obtain.   To address these limitations, we introduce NuClass, a pathologist workflow inspired framework for cell-wise multi-scale integration of nuclear morphology and microenvironmental context. NuClass includes two main components: Path local, which focuses on nuclear morphology from 224-by-224 pixel crops, and Path global, which models the surrounding 1024-by-1024 pixel neighborhood. A learnable gating module adaptively balances local detail and contextual cues. To encourage complementary learning, we incorporate an uncertainty-guided objective that directs the global path to prioritize regions where the local path is uncertain. We also provide calibrated confidence estimates and Grad-CAM visualizations to enhance interpretability.   To overcome the lack of high-quality annotations, we construct a marker-guided dataset from Xenium spatial transcriptomics assays, yielding single-cell resolution labels for more than two million cells across eight organs and 16 classes. Evaluated on three fully held-out cohorts, NuClass achieves up to 96 percent F1 for its best-performing class, outperforming strong baselines. Our results show that multi-scale, uncertainty-aware fusion can bridge the gap between slide-level pathological foundation models and reliable, cell-level phenotype prediction.

---

## 81. FreeAskWorld: An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI

**论文链接:** [http://arxiv.org/abs/2511.13524v1](http://arxiv.org/abs/2511.13524v1)

**作者:** Yuhang Peng, Yizhou Pan, Xinning He, Jihaoyu Yang, Xinyu Yin, Han Wang, Xiaoji Zheng, Chao Gao, Jiangtao Gong

**发布时间:** 2025-11-17

**备注:** 9 pages, 4 figures

### GPT解析

### 总结

本研究提出了FreeAskWorld，一个集成大型语言模型的交互式仿真框架，用于具身智能研究中模拟复杂社会行为，并通过基准测试证明了其在提升AI系统语义理解和交互能力方面的有效性。

### 背景

具身智能作为人工智能研究的前沿领域，需要模拟平台能够超越低级物理交互，捕捉以人为中心的复杂社会行为。

### 目的

开发一个能够支持可扩展、逼真的人与智能体模拟的交互式仿真框架，并通过基准数据集和实验验证其有效性。

### 方法

提出FreeAskWorld框架，集成大型语言模型用于高级行为规划和语义基础交互；扩展经典视觉语言导航任务为交互丰富的方向查询设置；构建包含多种环境、任务类型和交互数据的大型基准数据集；对VLN模型和人类参与者在开环和闭环设置下进行基准测试。

### 主要发现

在FreeAskWorld上微调的模型优于原始模型，实现了增强的语义理解和交互能力；基于社会的仿真框架能有效推进具身AI系统向复杂高级规划和更自然的人与智能体交互发展；交互本身作为一种额外的信息模态对AI系统有重要价值。

### 结论

FreeAskWorld框架及其基准数据集为具身智能研究提供了有力工具，能够有效模拟复杂社会行为，提升AI系统的语义理解和交互能力，推动具身智能向更高级别发展。

### 翻译

随着具身智能成为人工智能研究的核心前沿，模拟平台必须从低级物理交互发展，以捕捉复杂、以人为中心的社会行为。我们介绍了FreeAskWorld，一个交互式仿真框架，集成了大型语言模型用于高级行为规划和语义基础交互，基于意图和社会认知理论。我们的框架支持可扩展、逼真的人与智能体模拟，包括为各种具身任务定制的模块化数据生成管道。为验证该框架，我们将经典的视觉语言导航任务扩展为交互丰富的方向查询设置，其中智能体可以主动寻求和解释导航指导。我们发布并公开了FreeAskWorld，这是一个包含重建环境、六种不同任务类型、16个核心对象类别、63,429个注释样本帧和超过17小时交互数据的大型基准数据集，用于支持具身AI系统的训练和评估。我们在开环和闭环设置下对VLN模型和人类参与者进行了基准测试。实验结果表明，在FreeAskWorld上微调的模型优于其原始对应模型，实现了增强的语义理解和交互能力。这些发现强调了基于社会的仿真框架在推进具身AI系统向复杂高级规划和更自然的人与智能体交互方面的有效性。重要的是，我们的研究强调了交互本身作为一种额外的信息模态。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有具身智能模拟平台无法有效捕获复杂社会行为的问题，以及视觉语言导航系统在处理动态目标、多轮交互和社交意图建模方面的局限性。这个问题在现实中很重要，因为真实世界中的AI代理需要理解和遵循人类生成的导航指令，特别是在动态环境中；在研究中也很重要，因为社交互动本身是AI代理获取信息的重要途径，而现有模拟器不足以建模以人为基础的社交沟通，限制了AI系统向高级规划和更自然的代理-人类交互发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有模拟平台在模拟社会行为方面的不足，注意到生成式AI和大语言模型为模拟环境中的高级行为建模提供了新可能性。他们设计了一个交互式和闭环模拟框架FreeAskWorld，利用大语言模型进行高级意图建模、语义指令生成和自然人类行为模拟，基于社会行为和意图理论。作者借鉴了社会导航中模拟人类行为的研究、使用生成AI模拟高级行为的工作（如MARPLE）、Virtual Community等模拟器，以及R2R等视觉语言导航基准测试，但在此基础上进行了创新和扩展。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个交互式和闭环的模拟框架，专门用于以人为中心的具身AI，利用大语言模型进行高级行为规划和语义基础交互，支持可扩展、逼真的人-代理模拟，并将交互本身视为一种额外的信息模态。整体实现流程包括：1)场景初始化（随机化环境条件）；2)数据收集代理搜索附近人类代理并发起交互；3)LLM生成逼真、类人的导航指令；4)代理使用符合社会规范的策略导航到目的地；5)失败时发起另一轮查询；6)成功到达后记录所有相关数据用于训练和评估。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)交互式LLM驱动的模拟框架，支持动态、逼真的类人代理模拟；2)方向查询任务，扩展传统VLN范式，允许代理主动寻求帮助；3)大规模数据集，包含丰富的环境和交互数据；4)模块化数据生成管道，专为各种具身任务定制；5)社会认知理论整合，使代理能理解和回应社会线索。相比之前的工作，FreeAskWorld从静态转向动态交互，整合高级规划与社会意图建模，提供更丰富的环境模拟，将交互视为信息模态，并支持闭环评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FreeAskWorld引入了一个交互式、闭环的模拟框架和数据集，通过整合大语言模型和社会认知理论，使具身AI系统能够进行更自然的以人为中心的交互和导航，同时证明了交互本身作为一种信息模态的重要性。'}


### 论文摘要

As embodied intelligence emerges as a core frontier in artificial intelligence research, simulation platforms must evolve beyond low-level physical interactions to capture complex, human-centered social behaviors. We introduce FreeAskWorld, an interactive simulation framework that integrates large language models (LLMs) for high-level behavior planning and semantically grounded interaction, informed by theories of intention and social cognition. Our framework supports scalable, realistic human-agent simulations and includes a modular data generation pipeline tailored for diverse embodied tasks.To validate the framework, we extend the classic Vision-and-Language Navigation (VLN) task into a interaction enriched Direction Inquiry setting, wherein agents can actively seek and interpret navigational guidance. We present and publicly release FreeAskWorld, a large-scale benchmark dataset comprising reconstructed environments, six diverse task types, 16 core object categories, 63,429 annotated sample frames, and more than 17 hours of interaction data to support training and evaluation of embodied AI systems. We benchmark VLN models, and human participants under both open-loop and closed-loop settings. Experimental results demonstrate that models fine-tuned on FreeAskWorld outperform their original counterparts, achieving enhanced semantic understanding and interaction competency. These findings underscore the efficacy of socially grounded simulation frameworks in advancing embodied AI systems toward sophisticated high-level planning and more naturalistic human-agent interaction. Importantly, our work underscores that interaction itself serves as an additional information modality.

---

## 82. Towards Quantum Software for Quantum Simulation

**论文链接:** [http://arxiv.org/abs/2511.13520v1](http://arxiv.org/abs/2511.13520v1)

**作者:** Maja Franz, Lukas Schmidbauer, Joshua Ammermann, Ina Schaefer, Wolfgang Mauerer

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文探讨了量子模拟作为展示量子计算优势的领先技术，指出当前量子模拟软件栈中的关键差距，并提倡采用模块化的模型驱动工程方法来支持不同类型的量子模拟，促进自动化、性能评估和可重用性。

### 背景

量子模拟被认为能提供比任何经典系统指数级更多的计算能力，为研究复杂物理系统提供了新方法。目前这些技术主要局限于基础科学，缺乏软件工程社区为其他计算领域提供的基础设施和建模抽象。

### 目的

识别量子模拟软件栈中的关键差距，特别是通用框架的缺失；倡导一种模块化的模型驱动工程方法，支持不同类型的量子模拟，并促进自动化、性能评估和可重用性。

### 方法

采用模块化的模型驱动工程方法，支持数字和模拟量子模拟，通过高能物理学的例子展示一个支持可扩展、跨平台模拟工作流的量子模拟框架愿景。

### 主要发现

量子模拟软件栈存在关键差距，特别是缺乏通用框架用于模型规范、哈密顿量构建和硬件感知映射；需要模块化的模型驱动工程方法来支持不同类型的量子模拟。

### 结论

需要开发一个支持可扩展、跨平台模拟工作流的量子模拟框架，以促进量子模拟技术的发展和应用。

### 翻译

量子模拟是展示量子计算相对于经典计算的实际优势的领先候选技术，因为它被认为能提供比任何经典系统指数级更多的计算能力。它为研究复杂物理系统提供了新方法，这些系统传统上依赖于基于数值高性能计算的密集型软件模拟代码。相比之下，量子模拟将研究对象(如化学分子)的特性映射到量子设备上，使其模仿所研究的系统。目前，这些技术的应用主要局限于基础科学，因为整体方法仍然针对特定问题进行调整：我们缺乏软件工程社区为其他计算领域提供的基础设施和建模抽象。在本文中，我们确定了量子模拟软件栈中的关键差距，特别是缺乏通用框架用于模型规范、哈密顿量构建和硬件感知映射。我们提倡采用模块化的模型驱动工程方法，支持不同类型的量子模拟(数字和模拟)，并促进自动化、性能评估和可重用性。通过高能物理学的例子，我们概述了一个能够支持可扩展、跨平台模拟工作流的量子模拟框架愿景。


### 论文摘要

Quantum simulation is a leading candidate for demonstrating practical quantum advantage over classical computation, as it is believed to provide exponentially more compute power than any classical system. It offers new means of studying the behaviour of complex physical systems, for which conventionally software-intensive simulation codes based on numerical high-performance computing are used. Instead, quantum simulations map properties and characteristics of subject systems, for instance chemical molecules, onto quantum devices that then mimic the system under study.   Currently, the use of these techniques is largely limited to fundamental science, as the overall approach remains tailored for specific problems: We lack infrastructure and modelling abstractions that are provided by the software engineering community for other computational domains.   In this paper, we identify critical gaps in the quantum simulation software stack-particularly the absence of general-purpose frameworks for model specification, Hamiltonian construction, and hardware-aware mappings. We advocate for a modular model-driven engineering (MDE) approach that supports different types of quantum simulation (digital and analogue), and facilitates automation, performance evaluation, and reusability. Through an example from high-energy physics, we outline a vision for a quantum simulation framework capable of supporting scalable, cross-platform simulation workflows.

---

## 83. Modeling Spatial Synchronization of Predator-Prey Oscillations via the XY Model under Demographic Stochasticity and Migration

**论文链接:** [http://arxiv.org/abs/2511.13501v1](http://arxiv.org/abs/2511.13501v1)

**作者:** Solmaz Golmohammadi, Mina Zarei, Jacopo Grilli

**发布时间:** 2025-11-17

### GPT解析

### 总结

研究探讨了随机捕食者-猎物动力学及其在空间上的相位同步现象，通过多斑块耦合的Rosenzweig-MacArthur模型进行分析。

### 背景

生态系统中捕食者与猎物的动态关系是生态学中的重要研究课题，而空间因素和随机性对生态系统同步性的影响尚需深入理解。

### 目的

揭示人口统计噪声和扩散如何影响生态系统的同步性和相变，为理解和管理大规模生态同步性与生态系统恢复力提供理论基础。

### 方法

结合基于Gillespie算法的随机模拟和受XY模型启发的分析方法，研究多斑块耦合的Rosenzweig-MacArthur模型。

### 主要发现

揭示了噪声和扩散塑造同步性和相变的基本机制，这些机制对于理解生态系统动态至关重要。

### 结论

该研究为理解和管理大规模生态同步性与生态系统恢复力提供了理论基础，有助于生态系统的保护和管理。

### 翻译

我们研究了随机捕食者-猎物动力学及其空间相位同步，使用的是在多个斑块间耦合的Rosenzweig-MacArthur模型。结合基于Gillespie算法的随机模拟和受XY模型启发的分析方法，我们揭示了人口统计噪声和扩散如何塑造同步性和相变的基本机制。这项研究为理解和管理大规模生态同步性和生态系统恢复力提供了理论基础。


### 论文摘要

We investigate stochastic predator-prey dynamics and their spatial phase synchronization using the Rosenzweig-MacArthur model coupled across multiple patches. Combining stochastic simulations based on the Gillespie algorithm with analytical methods inspired by the XY model, we uncover fundamental mechanisms through which demographic noise and dispersal shape synchronization and phase transitions. This study offers a theoretical foundation for understanding and managing large-scale ecological synchrony and ecosystem resilience.

---

## 84. Language-Guided Invariance Probing of Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.13494v1](http://arxiv.org/abs/2511.13494v1)

**作者:** Jae Joong Lee

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究引入了语言引导不变性探测(LGIP)基准测试，用于评估视觉语言模型对语言扰动的响应可靠性，发现EVA02-CLIP和大尺寸OpenCLIP变体表现最佳，而SigLIP系列模型显示出较大不变性误差。

### 背景

最近的视觉语言模型如CLIP、OpenCLIP、EVA02-CLIP和SigLIP在零样本学习方面表现出色，但尚不清楚它们对受控语言扰动的响应可靠性如何。

### 目的

开发一个基准测试来测量视觉语言模型对保持原意的释义的不变性，以及对改变语义的语义翻转的敏感性。

### 方法

使用4万张MS COCO图像(每张有五个人类标注)，自动生成释义和基于规则的翻转(改变对象类别、颜色或数量)，通过不变性误差、语义敏感性差距和正态率统计总结模型行为。

### 主要发现

EVA02-CLIP和大尺寸OpenCLIP变体位于有利的不变性-敏感性前沿，结合低释义引起的方差和高原始描述评分；SigLIP和SigLIP2则显示出较大不变性误差，经常更喜欢翻转后的描述而非原始描述。

### 结论

这些模型缺陷在标准检索指标中大多不可见，LGIP为视觉语言模型的语言稳健性提供了超越传统准确率评分的模型无关诊断方法。

### 翻译

最近的视觉语言模型(VLMs)，如CLIP、OpenCLIP、EVA02-CLIP和SigLIP，实现了强大的零样本性能，但目前尚不清楚它们对受控语言扰动的响应有多可靠。我们引入了语言引导不变性探测(LGIP)，这是一个基准测试，用于测量(i)对保持原意的释义的不变性，以及(ii)在图像-文本匹配中对改变语义的语义翻转的敏感性。使用4万张MS COCO图像，每张图像有五个人类标注，我们自动生成释义和基于规则的翻转，这些翻转会改变对象类别、颜色或数量，并使用不变性误差、语义敏感性差距和正态率统计来总结模型行为。在九种VLMs中，EVA02-CLIP和大尺寸OpenCLIP变体位于有利的不变性-敏感性前沿，结合了低释义引起的方差，以及对原始描述比其翻转版本持续更高的评分。相比之下，SigLIP和SigLIP2显示出更大的不变性误差，并且经常更喜欢翻转后的描述而非人类描述，特别是对于对象和颜色编辑。这些失败在标准检索指标中大多不可见，表明LGIP为VLMs的语言稳健性提供了模型无关的诊断方法，超越了传统的准确率评分。


### 论文摘要

Recent vision-language models (VLMs) such as CLIP, OpenCLIP, EVA02-CLIP and SigLIP achieve strong zero-shot performance, but it is unclear how reliably they respond to controlled linguistic perturbations. We introduce Language-Guided Invariance Probing (LGIP), a benchmark that measures (i) invariance to meaning-preserving paraphrases and (ii) sensitivity to meaning-changing semantic flips in image-text matching. Using 40k MS COCO images with five human captions each, we automatically generate paraphrases and rule-based flips that alter object category, color or count, and summarize model behavior with an invariance error, a semantic sensitivity gap and a positive-rate statistic.   Across nine VLMs, EVA02-CLIP and large OpenCLIP variants lie on a favorable invariance-sensitivity frontier, combining low paraphrase-induced variance with consistently higher scores for original captions than for their flipped counterparts. In contrast, SigLIP and SigLIP2 show much larger invariance error and often prefer flipped captions to the human descriptions, especially for object and color edits. These failures are largely invisible to standard retrieval metrics, indicating that LGIP provides a model-agnostic diagnostic for the linguistic robustness of VLMs beyond conventional accuracy scores.

---

## 85. Trust in Vision-Language Models: Insights from a Participatory User Workshop

**论文链接:** [http://arxiv.org/abs/2511.13458v1](http://arxiv.org/abs/2511.13458v1)

**作者:** Agnese Chiatti, Lara Piccolo, Sara Bernardini, Matteo Matteucci, Viola Schiaffonati

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文介绍了关于VLMs用户信任的初步研究结果，采用以用户为中心的方法，通过研讨会收集潜在用户的见解，为未来研究提供方向。

### 背景

视觉语言模型(VLMs)越来越多地部署在大规模的图像-文本和视频-文本数据集上进行预训练。用户需要工具来判断何时可以信任这些系统，研究用户对VLMs的信任如何建立和演变仍然是一个开放性问题。越来越依赖AI模型作为实验验证的评判者，以绕过直接与用户进行参与式设计的成本和影响。

### 目的

为用户提供工具来判断何时可以信任VLMs；研究用户对VLMs的信任如何建立和演变；为未来的研究提供信息，以适应用户-VLM交互的案例。

### 方法

采用以用户为中心的方法；进行了一个有潜在VLM用户参与的研讨会；展示了该研讨会的初步结果。

### 主要发现

摘要中没有明确提到具体的发现，只提到从研讨会中获得了初步结果和见解。

### 结论

从试点研讨会中获得的信息将有助于未来的研究，使信任度量和参与者参与策略适应用户-VLM交互的案例。

### 翻译

随着视觉语言模型(VLMs)在大型图像-文本和视频-文本数据集上预训练后的日益广泛部署，为用户提供判断何时信任这些系统的工具至关重要。然而，研究用户对VLMs的信任如何建立和演变仍然是一个开放性问题。这个问题因越来越多地依赖AI模型作为实验验证的评判者而加剧，以绕过直接与用户进行参与式设计的成本和影响。本文采用以用户为中心的方法，展示了与潜在VLM用户研讨会的初步结果。从这次试点研讨会获得的见解将为未来的研究提供信息，以使信任度量和参与者参与策略适应用户-VLM交互的案例。


### 论文摘要

With the growing deployment of Vision-Language Models (VLMs), pre-trained on large image-text and video-text datasets, it is critical to equip users with the tools to discern when to trust these systems. However, examining how user trust in VLMs builds and evolves remains an open problem. This problem is exacerbated by the increasing reliance on AI models as judges for experimental validation, to bypass the cost and implications of running participatory design studies directly with users. Following a user-centred approach, this paper presents preliminary results from a workshop with prospective VLM users. Insights from this pilot workshop inform future studies aimed at contextualising trust metrics and strategies for participants' engagement to fit the case of user-VLM interaction.

---

## 86. Spontaneous emergence of solitary waves in active flow networks

**论文链接:** [http://arxiv.org/abs/2511.13448v1](http://arxiv.org/abs/2511.13448v1)

**作者:** Rodrigo Fernández-Quevedo García, Gonçalo Cruz Antunes, Jens Harting, Holger Stark, Chantal Valeriani, Martin Brandenbourger, Juan José Mazo, Paolo Malgaretti, Miguel Ruiz-García

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了主动流动网络中的信息处理机制，通过分析由主动泵送单元和弹性存储单元组成的系统，揭示了简单流体元件如何集体创建、塑造和传输信息，为理解主动流动系统中的信息处理奠定了基础。

### 背景

流动网络是理解动植物血管系统或电力分配网格等系统的基础，这些网络能够编码、传输和转换包含在其流动时空分布中的信息。

### 目的

研究一个最小但物理基础明确的系统，隔离主动流动网络产生和调节支持信息传输的涌现动力学的基本机制。

### 方法

构建由主动单元(泵送流体)和弹性单元(存储体积)组成的系统，从基本原理出发推导离散的主动流动网络模型，用于模拟具有许多相互作用单元的大型系统。

### 主要发现

数值模拟显示压力场可发展出孤波，导致在流动物理特性中存储的局部信息包的自发创建和传输；孤波从无序初始条件中出现，其大小和传播速度依赖于关键系统参数；当弹性单元耦合时，孤波表现出多样形状和有限寿命，显示出可解析预测的幂律行为。

### 结论

简单流体元件能够集体创建、塑造和传输信息，为理解并最终设计主动流动系统中的信息处理提供了理论基础。

### 翻译

流动网络是理解诸如动植物血管系统或电力分配网络等系统的基础。这些网络能够编码、传输和转换包含在其流动时空分布中的信息。在本工作中，我们关注一个最小但物理基础明确的系统，使我们能够隔离主动流动网络产生和调节支持信息传输的涌现动力学的基本机制。该系统由泵送流体的主动单元和存储体积的弹性单元组成。从基本原理出发，我们推导出一个离散模型—主动流动网络—它能够模拟具有许多相互作用单元的大型系统。数值上，我们表明压力场可以发展出孤波，导致在流动物理特性中存储的局部信息包的自发创建和传输。我们描述了这些孤波如何在一维网络中从无序初始条件中出现，以及它们的大小和传播速度如何依赖于关键系统参数。最后，当弹性单元与其邻居耦合时，孤波表现出更丰富的动力学特性，具有多样形状和有限寿命，显示出我们可以通过解析预测的幂律行为。这些结果共同展示了简单流体元件如何集体创建、塑造和传输信息，为理解并最终设计主动流动系统中的信息处理奠定了基础。


### 论文摘要

Flow networks are fundamental for understanding systems such as animal and plant vasculature or power distribution grids. These networks can encode, transmit, and transform information embodied in the spatial and temporal distribution of their flows. In this work, we focus on a minimal yet physically grounded system that allows us to isolate the fundamental mechanisms by which active flow networks generate and regulate emergent dynamics capable of supporting information transmission. The system is composed of active units that pump fluid and elastic units that store volume. From first principles, we derive a discrete model-an active flow network-that enables the simulation of large systems with many interacting units. Numerically, we show that the pressure field can develop solitary waves, resulting in the spontaneous creation and transmission of localized packets of information stored in the physical properties of the flow. We characterize how these solitary waves emerge from disordered initial conditions in a one-dimensional network, and how their size and propagation speed depend on key system parameters. Finally, when the elastic units are coupled to their neighbors, the solitary waves exhibit even richer dynamics, with diverse shapes and finite lifetimes that display power-law behaviors that we can predict analytically. Together, these results show how simple fluidic elements can collectively create, shape and transport information, laying the foundations for understanding-and ultimately engineering-information processing in active flow systems.

---

## 87. Unlocking the Forgery Detection Potential of Vanilla MLLMs: A Novel Training-Free Pipeline

**论文链接:** [http://arxiv.org/abs/2511.13442v1](http://arxiv.org/abs/2511.13442v1)

**作者:** Rui Zuo, Qinyue Tong, Zhe-Ming Lu, Ziqian Lu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为Foresee的基于多模态大语言模型(MLLM)的无训练图像伪造检测和定位流水线，该方法无需额外训练即可实现轻量级推理，并在篡改定位准确性和文本解释丰富度上超越现有方法。

### 背景

随着人工智能生成内容(AIGC)技术的快速发展，图像生成和操作变得容易，但现有图像伪造检测和定位(IFDL)方法难以泛化且可解释性有限。虽然一些研究将MLLMs引入IFDL领域，但需要大规模训练，消耗大量计算资源。

### 目的

揭示原始MLLMs在IFDL领域的固有泛化潜力，开发一种不需要额外训练的轻量级方法来解决现有IFDL方法泛化能力差的问题。

### 方法

提出Foresee，一种基于MLLM的无训练流水线，采用类型先验驱动的策略，并使用灵活特征检测器(FFD)模块专门处理复制移动操作，无需额外训练即可实现轻量级推理过程。

### 主要发现

Foresee在篡改定位准确性和文本解释丰富度上超越了现有的基于MLLM的方法；该方法同时实现了卓越的定位准确性并提供更全面的文本解释；Foresee展现出更强的泛化能力，在各种篡改类型上都优于现有的IFDL方法。

### 结论

Foresee成功释放了原始MLLMs在取证领域的潜力，无需额外训练即可实现高性能，代码将在最终版本中发布。

### 翻译

随着人工智能生成内容(AIGC)技术(包括多模态大语言模型(MLLMs)和扩散模型)的快速发展，图像生成和操作变得异常容易。现有的图像伪造检测和定位(IFDL)方法往往难以在不同数据集上泛化，且提供有限的解释性。如今，MLLMs在各种视觉-语言任务中展现出强大的泛化潜力，一些研究通过大规模训练将这种能力引入IFDL领域。然而，此类方法需要大量计算资源，且未能揭示原始MLLMs解决此问题的固有泛化潜力。受此启发，我们提出了Foresee，一种专为图像伪造分析设计的基于MLLM的无训练流水线。它消除了额外训练的需求，实现了轻量级推理过程，同时在篡改定位准确性和文本解释丰富度上超越了现有的基于MLLM的方法。Foresee采用类型先验驱动的策略，并利用灵活特征检测器(FFD)模块专门处理复制移动操作，从而有效释放了原始MLLMs在取证领域的潜力。大量实验证明，我们的方法同时实现了卓越的定位准确性，并提供了更全面的文本解释。此外，Foresee展现出更强的泛化能力，在各种篡改类型上均优于现有的IFDL方法，包括复制移动、拼接、移除、局部增强、深度伪造和基于AIGC的编辑。代码将在最终版本中发布。


### 论文摘要

With the rapid advancement of artificial intelligence-generated content (AIGC) technologies, including multimodal large language models (MLLMs) and diffusion models, image generation and manipulation have become remarkably effortless. Existing image forgery detection and localization (IFDL) methods often struggle to generalize across diverse datasets and offer limited interpretability. Nowadays, MLLMs demonstrate strong generalization potential across diverse vision-language tasks, and some studies introduce this capability to IFDL via large-scale training. However, such approaches cost considerable computational resources, while failing to reveal the inherent generalization potential of vanilla MLLMs to address this problem. Inspired by this observation, we propose Foresee, a training-free MLLM-based pipeline tailored for image forgery analysis. It eliminates the need for additional training and enables a lightweight inference process, while surpassing existing MLLM-based methods in both tamper localization accuracy and the richness of textual explanations. Foresee employs a type-prior-driven strategy and utilizes a Flexible Feature Detector (FFD) module to specifically handle copy-move manipulations, thereby effectively unleashing the potential of vanilla MLLMs in the forensic domain. Extensive experiments demonstrate that our approach simultaneously achieves superior localization accuracy and provides more comprehensive textual explanations. Moreover, Foresee exhibits stronger generalization capability, outperforming existing IFDL methods across various tampering types, including copy-move, splicing, removal, local enhancement, deepfake, and AIGC-based editing. The code will be released in the final version.

---

## 88. VOPE: Revisiting Hallucination of Vision-Language Models in Voluntary Imagination Task

**论文链接:** [http://arxiv.org/abs/2511.13420v1](http://arxiv.org/abs/2511.13420v1)

**作者:** Xingming Long, Jie Zhang, Shiguang Shan, Xilin Chen

**发布时间:** 2025-11-17

**备注:** 8 pages

### GPT解析

### 总结

该研究针对大型视觉语言模型在自愿想象任务中的幻觉现象评估问题，提出了VOPE新方法，并发现现有模型在此类任务中幻觉严重，现有缓解方法效果有限。

### 背景

大多数关于大型视觉语言模型中幻觉的研究集中在事实描述任务上，很少关注自愿想象任务(如故事写作)中的幻觉现象，在这些任务中模型被期望生成超出给定图像的新内容。

### 目的

引入一种新方法来评估大型视觉语言模型在自愿想象任务中的幻觉现象。

### 方法

提出自愿想象物体存在评估方法，通过基于重新检查的问题评估模型如何解释其响应中想象物体的存在，并根据模型解释与图像中物体存在的一致性确定是否产生幻觉。

### 主要发现

(1)大多数大型视觉语言模型在自愿想象过程中会产生大量幻觉，在存在评估中对想象物体的表现明显较差；(2)现有的幻觉缓解方法在自愿想象任务中效果有限。

### 结论

自愿想象任务中的幻觉缓解是未来研究的重要方向。

### 翻译

大多数关于大型视觉语言模型中幻觉的研究集中在禁止输出图像中不存在内容的事实描述任务上。然而，很少有人关注自愿想象任务中的幻觉，例如故事写作，在这些任务中模型被期望生成超出给定图像的新内容。在这些任务中，简单地将想象的新内容视为幻觉是不合适的。为了解决这一局限性，我们引入了自愿想象物体存在评估——一种通过存在评估来评估大型视觉语言模型在自愿想象任务中幻觉的新方法。具体来说，该方法提出基于重新检查的问题来评估模型如何解释其自身响应中想象物体的存在。然后，使用模型解释与图像中物体存在的一致性来确定模型在生成响应时是否产生幻觉。我们将该方法应用于几种主流大型视觉语言模型和幻觉缓解方法，揭示了两个关键发现：(1)大多数大型视觉语言模型在自愿想象过程中会产生大量幻觉，它们在存在评估中对想象物体的表现明显较差；(2)现有的幻觉缓解方法在自愿想象任务中效果有限，这使得这一领域成为未来研究的重要方向。


### 论文摘要

Most research on hallucinations in Large Vision-Language Models (LVLMs) focuses on factual description tasks that prohibit any output absent from the image. However, little attention has been paid to hallucinations in voluntary imagination tasks, e.g., story writing, where the models are expected to generate novel content beyond the given image. In these tasks, it is inappropriate to simply regard such imagined novel content as hallucinations. To address this limitation, we introduce Voluntary-imagined Object Presence Evaluation (VOPE)-a novel method to assess LVLMs' hallucinations in voluntary imagination tasks via presence evaluation. Specifically, VOPE poses recheck-based questions to evaluate how an LVLM interprets the presence of the imagined objects in its own response. The consistency between the model's interpretation and the object's presence in the image is then used to determine whether the model hallucinates when generating the response. We apply VOPE to several mainstream LVLMs and hallucination mitigation methods, revealing two key findings: (1) most LVLMs hallucinate heavily during voluntary imagination, and their performance in presence evaluation is notably poor on imagined objects; (2) existing hallucination mitigation methods show limited effect in voluntary imagination tasks, making this an important direction for future research.

---

## 89. Attention Grounded Enhancement for Visual Document Retrieval

**论文链接:** [http://arxiv.org/abs/2511.13415v1](http://arxiv.org/abs/2511.13415v1)

**作者:** Wanqing Cui, Wei Huang, Yazhi Guo, Yibo Hu, Meiguang Jin, Junfeng Ma, Keping Bi

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了AGREE框架，通过利用多模态大语言模型的跨模态注意力作为代理局部监督，改善了视觉文档检索性能，实现了更准确和可解释的检索结果。

### 背景

视觉文档检索需要理解和处理异构和多模态内容以满足信息需求。最新进展使用基于截图的文档编码和细粒度后期交互提高了检索性能，但检索器仍使用粗糙的全局相关性标签训练，导致依赖表面线索，难以捕捉隐式语义连接，难以处理非抽取式查询。

### 目的

解决检索器依赖表面线索和难以捕捉隐式语义连接的问题，提高检索器处理非抽取式查询的能力，实现更准确和可解释的检索。

### 方法

提出AGREE（Attention-Grounded REtriever Enhancement）框架，利用多模态大语言模型的跨模态注意力作为代理局部监督，指导识别相关文档区域。在训练过程中，结合局部信号与全局信号共同优化检索器，使检索器不仅学习文档是否匹配，还学习哪些内容驱动相关性。

### 主要发现

在具有挑战性的ViDoRe V2基准测试中，AGREE显著优于仅使用全局监督的基线方法。定量和定性分析表明，AGREE促进了查询词和文档区域之间的更深层次对齐，超越了表面匹配，实现了更准确和可解释的检索。

### 结论

AGREE框架通过引入局部监督改善了视觉文档检索性能，实现了更准确的检索结果和更好的可解释性，代码已公开可用。

### 翻译

视觉文档检索需要理解异构和多模态内容以满足信息需求。最近的进展使用基于截图的文档编码与细粒度后期交互，显著提高了检索性能。然而，检索器仍然使用粗糙的全局相关性标签进行训练，没有揭示哪些区域支持匹配。因此，检索器倾向于依赖表面线索，难以捕捉隐式语义连接，阻碍了其处理非抽取式查询的能力。为缓解此问题，我们提出了一个注意力引导的检索器增强（AGREE）框架。AGREE利用多模态大语言模型的跨模态注意力作为代理局部监督，以指导识别相关文档区域。在训练过程中，AGREE将局部信号与全局信号结合，共同优化检索器，使其不仅学习文档是否匹配，还学习哪些内容驱动相关性。在具有挑战性的ViDoRe V2基准上的实验表明，AGREE显著优于仅使用全局监督的基线方法。定量和定性分析进一步证明，AGREE促进了查询词和文档区域之间的更深层次对齐，超越了表面匹配，实现了更准确和可解释的检索。我们的代码可在以下网址获取：https://anonymous.4open.science/r/AGREE-2025。


### 论文摘要

Visual document retrieval requires understanding heterogeneous and multi-modal content to satisfy information needs. Recent advances use screenshot-based document encoding with fine-grained late interaction, significantly improving retrieval performance. However, retrievers are still trained with coarse global relevance labels, without revealing which regions support the match. As a result, retrievers tend to rely on surface-level cues and struggle to capture implicit semantic connections, hindering their ability to handle non-extractive queries. To alleviate this problem, we propose a \textbf{A}ttention-\textbf{G}rounded \textbf{RE}triever \textbf{E}nhancement (AGREE) framework. AGREE leverages cross-modal attention from multimodal large language models as proxy local supervision to guide the identification of relevant document regions. During training, AGREE combines local signals with the global signals to jointly optimize the retriever, enabling it to learn not only whether documents match, but also which content drives relevance. Experiments on the challenging ViDoRe V2 benchmark show that AGREE significantly outperforms the global-supervision-only baseline. Quantitative and qualitative analyses further demonstrate that AGREE promotes deeper alignment between query terms and document regions, moving beyond surface-level matching toward more accurate and interpretable retrieval. Our code is available at: https://anonymous.4open.science/r/AGREE-2025.

---

## 90. What Color Is It? A Text-Interference Multimodal Hallucination Benchmark

**论文链接:** [http://arxiv.org/abs/2511.13400v1](http://arxiv.org/abs/2511.13400v1)

**作者:** Jinkun Zhao, Lei Huang, Wenjun Wu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了多模态大模型(MLMs)在视觉感知中的幻觉问题，特别是颜色感知方面的干扰。

### 背景

大型模型快速发展，出现了许多融合文本和视觉的多模态大模型(MLMs)，但这些模型在视觉感知中容易受到信息干扰，特别是在颜色感知方面，增加了幻觉风险。

### 目的

验证MLMs在视觉感知中，特别是颜色感知方面容易产生幻觉的假设。

### 方法

引入了'What Color Is It'数据集，这是一个使用简单方法构建的新基准，用于触发MLMs的单模态视觉幻觉；基于此数据集研究MLMs视觉模态中幻觉的根本原因，并提出可能的解决方案。

### 主要发现

研究确认了MLMs在视觉感知中存在幻觉问题，特别是在颜色感知方面。

### 结论

MLMs在视觉感知中确实存在幻觉风险，特别是在颜色感知方面，需要通过提出的方法来增强模型的鲁棒性。

### 翻译

随着大型模型的快速发展，许多融合文本和视觉的多模态大模型(MLMs)应运而生。然而，这些MLMs在视觉感知中仍然容易受到信息干扰，特别是在颜色感知方面，这增加了额外的幻觉风险。为了验证这一假设，我们引入了'What Color Is It'数据集，这是一个使用简单方法构建的新基准，用于触发MLMs的单模态视觉幻觉。基于这个数据集，我们进一步研究了MLMs视觉模态中幻觉的根本原因，并提出了潜在的解决方案来增强它们的鲁棒性。


### 论文摘要

With the rapid advancement of Large Models, numerous text-and-vision-fused Multimodal Large Models (MLMs) have emerged. However, these MLMs remain susceptible to informational interference in visual perception, particularly in color perception, which introduces an additional risk of hallucination. To validate this hypothesis, we introduce the "What Color Is It" dataset, a novel benchmark constructed using a simple method to trigger single-modality visual hallucination in MLMs. Based on this dataset, we further investigate the underlying causes of hallucination in the visual modality of MLMs and propose potential solutions to enhance their robustness.

---

## 91. Can Large Language Models Function as Qualified Pediatricians? A Systematic Evaluation in Real-World Clinical Contexts

**论文链接:** [http://arxiv.org/abs/2511.13381v1](http://arxiv.org/abs/2511.13381v1)

**作者:** Siyu Zhu, Mouxiao Bian, Yue Xie, Yongyu Tang, Zhikang Yu, Tianbin Li, Pengcheng Chen, Bing Han, Jie Xu, Xiaoyan Dong

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究评估了大型语言模型在儿科医疗领域的应用能力，通过开发PEDIASBench评估框架，从基础知识应用、动态诊疗能力和医疗安全伦理三个维度对12个代表性模型进行了测试。研究发现，虽然先进模型在基础知识方面表现良好，但在复杂推理、动态决策和人文敏感性方面存在明显局限性。

### 背景

随着大型语言模型在医学领域的迅速崛起，一个关键问题是它们能否在真实的临床环境中胜任儿科医生的工作。

### 目的

开发一个针对真实临床环境的系统性评估框架，评估大型语言模型在儿科医疗领域的表现能力。

### 方法

开发了PEDIASBench评估框架，该框架基于知识系统框架，从三个维度(基础知识应用、动态诊疗能力、儿科医疗安全与伦理)评估模型。评估了12个代表性模型(GPT-4o、Qwen3-235B-A22B等)，涵盖19个儿科亚专业和211种典型疾病。

### 主要发现

先进模型在基础知识方面表现良好(Qwen3-235B-A22B在执照级问题准确率超90%)，但随着任务复杂性增加性能下降约15%；多项选择评估显示综合推理和知识回忆存在弱点；DeepSeek-R1在病例推理方面表现最佳(平均0.58)，但多数模型难以适应患者实时变化；Qwen2.5-72B在医疗伦理和安全任务中表现最佳(准确率92.05%)，但人文敏感性有限。

### 结论

儿科LLMs受到动态决策能力有限和人文护理不发达的限制。未来应聚焦多模态整合和临床反馈-模型迭代循环，提高安全性、可解释性和人机协作。当前LLMs虽不能独立进行儿科护理，但在决策支持、医学教育和患者沟通方面具有潜力，为智能儿科医疗系统奠定基础。

### 翻译

随着医学领域大型语言模型的迅速崛起，一个关键问题是它们能否在真实临床环境中胜任儿科医生的工作。我们开发了PEDIASBench，一个基于知识系统框架并针对真实临床环境的系统性评估框架。PEDIASBench从三个维度评估LLMs：基础知识应用、动态诊疗能力以及儿科医疗安全和医疗伦理。我们评估了过去两年发布的12个代表性模型，包括GPT-4o、Qwen3-235B-A22B和DeepSeek-V3，涵盖了19个儿科亚专业和211种典型疾病。最先进的模型在基础知识方面表现良好，Qwen3-235B-A22B在执照级问题上的准确率超过90%，但随着任务复杂性的增加，性能下降了约15%，显示出在复杂推理方面的局限性。多项选择评估突出了综合推理和知识回忆的弱点。在动态诊疗场景中，DeepSeek-R1在病例推理方面得分最高(平均0.58)，但大多数模型难以适应患者的实时变化。在儿科医疗伦理和安全任务中，Qwen2.5-72B表现最好(准确率92.05%)，但人文敏感性仍然有限。这些发现表明，儿科LLMs受到动态决策能力有限和人文护理不发达的限制。未来的发展应侧重于多模态整合和临床反馈-模型迭代循环，以提高安全性、可解释性和人机协作。虽然当前的LLMs不能独立进行儿科护理，但它们在决策支持、医学教育和患者沟通方面具有潜力，为安全、可靠和协作的智能儿科医疗系统奠定了基础。


### 论文摘要

With the rapid rise of large language models (LLMs) in medicine, a key question is whether they can function as competent pediatricians in real-world clinical settings. We developed PEDIASBench, a systematic evaluation framework centered on a knowledge-system framework and tailored to realistic clinical environments. PEDIASBench assesses LLMs across three dimensions: application of basic knowledge, dynamic diagnosis and treatment capability, and pediatric medical safety and medical ethics. We evaluated 12 representative models released over the past two years, including GPT-4o, Qwen3-235B-A22B, and DeepSeek-V3, covering 19 pediatric subspecialties and 211 prototypical diseases. State-of-the-art models performed well on foundational knowledge, with Qwen3-235B-A22B achieving over 90% accuracy on licensing-level questions, but performance declined ~15% as task complexity increased, revealing limitations in complex reasoning. Multiple-choice assessments highlighted weaknesses in integrative reasoning and knowledge recall. In dynamic diagnosis and treatment scenarios, DeepSeek-R1 scored highest in case reasoning (mean 0.58), yet most models struggled to adapt to real-time patient changes. On pediatric medical ethics and safety tasks, Qwen2.5-72B performed best (accuracy 92.05%), though humanistic sensitivity remained limited. These findings indicate that pediatric LLMs are constrained by limited dynamic decision-making and underdeveloped humanistic care. Future development should focus on multimodal integration and a clinical feedback-model iteration loop to enhance safety, interpretability, and human-AI collaboration. While current LLMs cannot independently perform pediatric care, they hold promise for decision support, medical education, and patient communication, laying the groundwork for a safe, trustworthy, and collaborative intelligent pediatric healthcare system.

---

## 92. Statistically Accurate and Robust Generative Prediction of Rock Discontinuities with A Tabular Foundation Model

**论文链接:** [http://arxiv.org/abs/2511.13339v1](http://arxiv.org/abs/2511.13339v1)

**作者:** Han Meng, Gang Mei, Hong Tian, Nengxiong Xu, Jianbing Peng

**发布时间:** 2025-11-17

### GPT解析

### 总结

研究提出了一种基于表格基础模型的简单而鲁棒的方法，用于岩石不连续面的统计准确生成预测，解决了现有方法在数据稀疏条件下无法捕捉复杂分布模式的问题。

### 背景

岩石不连续面对岩体的力学行为和稳定性有决定性影响，但其内部分布通常难以直接观察，只能从地表暴露的不连续面进行推断。然而地表观测数据稀疏，现有生成预测方法要么无法捕捉复杂分布模式，要么在数据稀疏条件下缺乏鲁棒性。

### 目的

开发一种简单但鲁棒的方法，实现岩石不连续面的统计准确生成预测，特别是在数据稀疏条件下仍能保持准确性和鲁棒性。

### 方法

利用表格基础模型，发挥专为小数据设计的基础模型的强大样本学习能力，在有限的实测不连续面数据中有效捕捉潜在的复杂分布模式。

### 主要发现

在十个具有不同规模和不连续面分布模式的数据集上进行的比较实验表明，该方法与传统统计模型和深度生成方法相比，具有更高的准确性和鲁棒性。

### 结论

这项研究推进了岩体结构的定量表征，支持更安全、更可靠的数据驱动岩土工程设计。

### 翻译

岩石不连续面对岩体的力学行为和稳定性具有决定性影响。它们的内部分布大多不可见，通常使用生成预测方法从地表暴露的不连续面进行推断。然而，地表观测数据本质上是稀疏的，现有生成预测方法要么无法捕捉潜在的复杂分布模式，要么在数据稀疏条件下缺乏鲁棒性。在此，我们提出了一种简单而鲁棒的方法，通过使用表格基础模型来统计准确预测岩石不连续面的生成分布。利用专为小数据设计的基础模型的强大样本学习能力，我们的方法能够在有限的实测不连续面中有效捕捉潜在的复杂分布模式。在十个具有不同规模和不连续面分布模式的数据集上的比较实验表明，与传统统计模型和深度生成方法相比，该方法具有更高的准确性和鲁棒性。这项工作推进了岩体结构的定量表征，支持更安全、更可靠的数据驱动岩土工程设计。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从地表暴露的稀疏不连续面观测数据准确预测岩石内部不连续面分布的问题。这个问题非常重要，因为岩石不连续面对岩体的力学行为和稳定性有决定性影响，而工程活动（如隧道开挖、水电建设等）都依赖于岩体的稳定性。内部不连续面通常无法直接观测，只能从地表暴露的不连续面进行推断，但地表观测数据天然稀疏，现有方法要么无法捕捉复杂分布模式，要么在数据稀疏时缺乏鲁棒性，直接影响工程风险评估的准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统统计方法（如蒙特卡洛方法）的局限性，它们假设参数相互独立，无法捕捉参数间相关性。然后考虑了深度生成模型（如GAN和DDPM）的优势，但发现它们需要大量高质量训练数据，而地表不连续面数据天然稀疏。最后，作者发现基础模型（特别是针对小数据设计的TabPFN）能从有限观测中捕捉复杂分布模式。作者借鉴了基础模型在小数据预测方面的研究工作，将其应用于岩石不连续面的生成预测，形成了一种简单而鲁棒的新方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用基础模型（特别是TabPFN）的强大样本学习能力，从有限的测量不连续面中捕捉潜在的复杂分布模式和参数相关性。基础模型通过预训练获得了丰富的先验知识，能够在小样本条件下有效学习数据的统计特性。整体实现流程包括：收集地表暴露的不连续面观测数据；使用TabPFN基础模型学习这些数据，自动捕捉不连续面参数（如倾向、倾角和迹长）的复杂分布模式；生成与观测数据统计特性一致的新样本；通过多种评估指标验证生成结果的准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将表格基础模型（TabPFN）应用于岩石不连续面的生成预测；解决了传统统计方法无法捕捉参数相关性的问题；克服了深度生成模型在数据稀疏条件下的局限性；提供了一种简单而鲁棒的方法，能在小样本条件下准确生成不连续面。相比之前工作，与传统统计方法相比，它能捕捉参数间复杂相关性；与深度生成模型相比，它不需要大量训练数据，在小样本条件下仍能保持良好性能，且不需要针对每个数据集进行大量超参数调整；与其他基础模型应用相比，它专门针对表格数据设计，能更好地处理不连续面的多参数特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于表格基础模型的岩石不连续面生成预测方法，能在数据稀疏条件下准确捕捉复杂的分布模式和参数相关性，为岩体结构的定量表征和更安全可靠的数据驱动岩土工程设计提供了新工具。'}


### 论文摘要

Rock discontinuities critically govern the mechanical behavior and stability of rock masses. Their internal distributions remain largely unobservable and are typically inferred from surface-exposed discontinuities using generative prediction approaches. However, surface-exposed observations are inherently sparse, and existing generative prediction approaches either fail to capture the underlying complex distribution patterns or lack robustness under data-sparse conditions. Here, we proposed a simple yet robust approach for statistically accurate generative prediction of rock discontinuities by utilizing a tabular foundation model. By leveraging the powerful sample learning capability of the foundation model specifically designed for small data, our approach can effectively capture the underlying complex distribution patterns within limited measured discontinuities. Comparative experiments on ten datasets with diverse scales and distribution patterns of discontinuities demonstrate superior accuracy and robustness over conventional statistical models and deep generative approaches. This work advances quantitative characterization of rock mass structures, supporting safer and more reliable data-driven geotechnical design.

---

## 93. Basis Immunity: Isotropy as a Regularizer for Uncertainty

**论文链接:** [http://arxiv.org/abs/2511.13334v1](http://arxiv.org/abs/2511.13334v1)

**作者:** Florent Segonne

**发布时间:** 2025-11-17

### GPT解析

### 总结

该论文提出了一种结合均值方差优化和各向同性约束的投资组合构建方法，通过特征风险平价(ERP)的各向同性理念扩展，创造了一个灵活的框架，能够在各向同性配置和纯均值方差之间平滑过渡，并提供了对现代投资组合概念的综合理解。

### 背景

多样化是稳健投资组合构建的基石，但由于模型不确定性和估计误差，其应用面临挑战。从业者通常依赖复杂、专有的启发式方法来应对这些问题。近期进展中的Agnostic Risk Parity引入了特征风险平价(ERP)，这是一种利用各向同性均匀分配特征风险的创新方法，增强了投资组合的稳定性。

### 目的

回顾并扩展ERP的各向同性执行理念，提出一个多功能的框架，将均值方差优化与各向同性约束相结合，作为信号不确定性的几何正则化器。

### 方法

提出一个将均值方差优化与各向同性约束相结合的框架，各向同性约束作为信号不确定性的几何正则化器。生成的配置自然分解为标准投资组合，通过可调节的各向同性惩罚参数，在各向同性完全配置和纯均值方差之间平滑插值。

### 主要发现

在行业趋势跟踪应用中，各向同性约束系统性地诱导负平均信号暴露，这是一种结构化的、参数稳健的崩盘对冲策略。

### 结论

这项工作提供了一个实用的、理论基础的工具，用于在信号不确定性下进行有弹性的资产配置，同时提供了现代投资组合概念的教学性综合。

### 翻译

多样化是稳健投资组合构建的基石，但其应用仍因模型不确定性和估计误差而充满挑战。从业者通常依赖复杂、专有的启发式方法来应对这些问题。在最近的进展中，Agnostic Risk Parity引入了特征风险平价(ERP)，一种利用各向同性均匀分配特征风险的创新方法，增强了投资组合稳定性。在本文中，我们回顾并扩展了ERP的各向同性执行理念，提出一个多功能的框架，将均值方差优化与作为信号不确定性几何正则化器的各向同性约束相结合。所得配置自然分解为标准投资组合，通过可调节的各向同性惩罚参数，在各向同性完全配置和纯均值方差之间平滑插值。除了方法论，我们重新审视了基本概念，阐明了各向同性、标准投资组合、主要投资组合、原始与对偶表示以及收益、风险和各向同性的内在基不变度量之间的基础联系。应用于行业趋势跟踪时，各向同性约束系统性地诱导负平均信号暴露——一种结构化的、参数稳健的崩盘对冲。这项工作既提供了在信号不确定性下进行有弹性资产配置的实用且理论基础的工具，也提供了现代投资组合概念的教学性综合。


### 论文摘要

Diversification is a cornerstone of robust portfolio construction, yet its application remains fraught with challenges due to model uncertainty and estimation errors. Practitioners often rely on sophisticated, proprietary heuristics to navigate these issues. Among recent advancements, Agnostic Risk Parity introduces eigenrisk parity (ERP), an innovative approach that leverages isotropy to evenly allocate risk across eigenmodes, enhancing portfolio stability.   In this paper, we review and extend the isotropy-enforced philosophy of ERP proposing a versatile framework that integrates mean-variance optimization with an isotropy constraint acting as a geometric regularizer against signal uncertainty. The resulting allocations decompose naturally into canonical portfolios, smoothly interpolating between full isotropy (closed-form isotropic-mean allocation) and pure mean-variance through a tunable isotropy penalty.   Beyond methodology, we revisit fundamental concepts and clarify foundational links between isotropy, canonical portfolios, principal portfolios, primal versus dual representations, and intrinsic basis-invariant metrics for returns, risk, and isotropy. Applied to sector trend-following, the isotropy constraint systematically induces negative average-signal exposure -- a structural, parameter-robust crash hedge.   This work offers both a practical, theoretically grounded tool for resilient allocation under signal uncertainty and a pedagogical synthesis of modern portfolio concepts.

---

## 94. TabFlash: Efficient Table Understanding with Progressive Question Conditioning and Token Focusing

**论文链接:** [http://arxiv.org/abs/2511.13283v1](http://arxiv.org/abs/2511.13283v1)

**作者:** Jongha Kim, Minseong Bae, Sanghyeok Lee, Jinsung Yoon, Hyunwoo J. Kim

**发布时间:** 2025-11-17

**备注:** AAAI 2026 (Main Technical Track)

### GPT解析

### 总结

本文提出了一种名为TabFlash的高效多模态大语言模型(MLLM)，专门用于表格图像理解。该模型通过渐进式问题条件化、剪枝策略和标记聚焦技术，解决了表格图像中信息不足和冗余的问题，实现了最先进的性能，同时计算效率更高。

### 背景

表格图像在理解和处理方面存在独特挑战，需要针对特定问题进行关注，同时存在冗余的背景区域。现有的多模态大语言模型(MLLM)方法往往忽视了这些特性，导致视觉表征信息不足且冗余。

### 目的

生成既信息丰富又紧凑的视觉特征，以改善表格理解，解决现有MLLM方法在处理表格图像时的不足。

### 方法

提出渐进式问题条件化，将问题注入到Vision Transformer层中，注入频率逐渐增加；引入剪枝策略，丢弃背景标记减少冗余；提出标记聚焦训练策略，鼓励模型将关键信息集中在保留的标记中；结合这些方法提出TabFlash模型。

### 主要发现

TabFlash实现了最先进的性能，优于开源和专有的MLLM，与第二好的MLLM相比，需要减少27%的FLOPs和30%的内存使用。

### 结论

TabFlash是一种高效且有效的表格理解MLLM，通过渐进式问题条件化、剪枝策略和标记聚焦的组合，成功解决了表格图像理解中的挑战。

### 翻译

表格图像由于其需要针对特定问题进行关注以及存在冗余背景区域的特点，给有效且高效的理解带来了独特挑战。现有的多模态大语言模型(MLLM)方法常常忽视这些特性，导致产生信息不足且冗余的视觉表征。为解决这些问题，我们旨在生成既信息丰富又紧凑的视觉特征以提高表格理解能力。我们首先提出渐进式问题条件化，考虑到每层处理额外信息的能力，将问题以逐渐增加的频率注入Vision Transformer层中，以生成问题感知的视觉特征。为减少冗余，我们引入了一种剪枝策略，丢弃背景标记，从而提高效率。为减轻剪枝导致的信息损失，我们进一步提出了标记聚焦，这是一种鼓励模型将关键信息集中在保留标记中的训练策略。通过结合这些方法，我们提出了TabFlash，一种用于表格理解的高效且有效的MLLM。TabFlash实现了最先进的性能，超越了开源和专有的MLLM，与第二好的MLLM相比，需要减少27%的FLOPs和30%的内存使用。


### 论文摘要

Table images present unique challenges for effective and efficient understanding due to the need for question-specific focus and the presence of redundant background regions. Existing Multimodal Large Language Model (MLLM) approaches often overlook these characteristics, resulting in uninformative and redundant visual representations. To address these issues, we aim to generate visual features that are both informative and compact to improve table understanding. We first propose progressive question conditioning, which injects the question into Vision Transformer layers with gradually increasing frequency, considering each layer's capacity to handle additional information, to generate question-aware visual features. To reduce redundancy, we introduce a pruning strategy that discards background tokens, thereby improving efficiency. To mitigate information loss from pruning, we further propose token focusing, a training strategy that encourages the model to concentrate essential information in the retained tokens. By combining these approaches, we present TabFlash, an efficient and effective MLLM for table understanding. TabFlash achieves state-of-the-art performance, outperforming both open-source and proprietary MLLMs, while requiring 27% less FLOPs and 30% less memory usage compared to the second-best MLLM.

---

## 95. Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation

**论文链接:** [http://arxiv.org/abs/2511.13269v1](http://arxiv.org/abs/2511.13269v1)

**作者:** Lingfeng Zhang, Yuchen Zhang, Hongsheng Li, Haoxiang Fu, Yingbo Tang, Hangjun Ye, Long Chen, Xiaojun Liang, Xiaoshuai Hao, Wenbo Ding

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究针对视觉语言模型在无人机导航中的空间智能能力不足的问题，提出了SpatialSky-Benchmark评估基准和Sky-VLM模型，通过大规模实验验证了其有效性。

### 背景

视觉语言模型(VLMs)已广泛应用于无人机任务，但其在无人机场景中的空间智能能力尚未充分探索，引发了对其在动态环境中导航和解释能力的担忧。

### 目的

填补现有VLMs在无人机空间智能能力评估方面的空白，开发专门针对无人机导航场景的视觉语言模型。

### 方法

创建包含环境感知和环境理解两大类、13个子类别的SpatialSky-Benchmark评估基准；开发包含100万个样本的SpatialSky-Dataset数据集；基于此数据集训练专门针对无人机空间推理的Sky-VLM模型。

### 主要发现

现有主流VLMs在复杂无人机导航场景中表现不佳；Sky-VLM在所有基准任务上达到最先进的性能。

### 结论

Sky-VLM为适合无人机场景的视觉语言模型发展铺平了道路，源代码已公开在GitHub上。

### 翻译

视觉语言模型(VLMs)利用其强大的视觉感知和推理能力，已被广泛应用于无人机(UAV)任务。然而，现有VLMs在无人机场景中的空间智能能力在很大程度上仍未被探索，引发了对其在导航和解释动态环境中有效性的担忧。为了填补这一空白，我们引入了SpatialSky-Benchmark，这是一个专门设计用于评估VLMs在无人机导航中空间智能能力的综合基准。我们的基准包含两类-环境感知和环境理解-细分为13个子类别，包括边界框、颜色、距离、高度和着陆安全分析等。对各种主流开源和闭源VLMs的广泛评估显示，在复杂的无人机导航场景中表现不佳，突显了其空间能力的显著差距。为了应对这一挑战，我们开发了SpatialSky-Dataset，这是一个包含100万个样本的综合数据集，在各种场景中具有多样化的注释。利用这个数据集，我们引入了Sky-VLM，这是一个专门为无人机空间推理设计的VLM，支持多种粒度和上下文。广泛的实验结果表明，Sky-VLM在所有基准任务上都达到了最先进的性能，为适合无人机场景的VLMs的发展铺平了道路。源代码可在https://github.com/linglingxiansen/SpatialSKy获取。


### 论文摘要

Vision-Language Models (VLMs), leveraging their powerful visual perception and reasoning capabilities, have been widely applied in Unmanned Aerial Vehicle (UAV) tasks. However, the spatial intelligence capabilities of existing VLMs in UAV scenarios remain largely unexplored, raising concerns about their effectiveness in navigating and interpreting dynamic environments. To bridge this gap, we introduce SpatialSky-Bench, a comprehensive benchmark specifically designed to evaluate the spatial intelligence capabilities of VLMs in UAV navigation. Our benchmark comprises two categories-Environmental Perception and Scene Understanding-divided into 13 subcategories, including bounding boxes, color, distance, height, and landing safety analysis, among others. Extensive evaluations of various mainstream open-source and closed-source VLMs reveal unsatisfactory performance in complex UAV navigation scenarios, highlighting significant gaps in their spatial capabilities. To address this challenge, we developed the SpatialSky-Dataset, a comprehensive dataset containing 1M samples with diverse annotations across various scenarios. Leveraging this dataset, we introduce Sky-VLM, a specialized VLM designed for UAV spatial reasoning across multiple granularities and contexts. Extensive experimental results demonstrate that Sky-VLM achieves state-of-the-art performance across all benchmark tasks, paving the way for the development of VLMs suitable for UAV scenarios. The source code is available at https://github.com/linglingxiansen/SpatialSKy.

---

## 96. GeoX-Bench: Benchmarking Cross-View Geo-Localization and Pose Estimation Capabilities of Large Multimodal Models

**论文链接:** [http://arxiv.org/abs/2511.13259v1](http://arxiv.org/abs/2511.13259v1)

**作者:** Yushuo Zheng, Jiangyong Ying, Huiyu Duan, Chunyi Li, Zicheng Zhang, Jing Liu, Xiaohong Liu, Guangtao Zhai

**发布时间:** 2025-11-17

### GPT解析

### 总结

GeoX-Bench是一个全面的基准测试，用于评估大型多模态模型在跨视图地理定位和姿态估计方面的能力，包含10,859个全景-卫星图像对和755,976个问答对。研究发现，当前LMMs在地理定位任务上表现良好，但在姿态估计任务上效果较差，而指令调优可以显著提高其跨视图地理感知能力。

### 背景

大型多模态模型在各种任务中表现出色，但在跨视图地理定位和姿态估计领域的知识和能力尚未被探索，尽管这些能力对导航、自动驾驶、户外机器人等有潜在好处。

### 目的

填补大型多模态模型在跨视图地理定位和姿态估计领域的研究空白，通过GeoX-Bench基准测试探索和评估LMMs在这些方面的能力。

### 方法

构建GeoX-Bench基准测试，包含10,859个全景-卫星图像对（覆盖49个国家的128个城市）和755,976个问答对（其中42,900个用于基准测试）。基于此基准测试，评估25个最先进的LMMs在跨视图地理定位和姿态估计任务上的能力，并探索指令调优的赋能效果。

### 主要发现

当前LMMs在地理定位任务上取得了令人印象深刻的性能，但在更复杂的姿态估计任务上效果显著下降，这突显了一个需要未来改进的关键领域。在GeoX-Bench的训练数据上对LMMs进行指令调优可以显著提高其跨视图地理感知能力。

### 结论

GeoX-Bench基准测试为评估和改进LMMs在跨视图地理定位和姿态估计方面的能力提供了重要工具，相关资源已在GitHub上公开。

### 翻译

大型多模态模型(LMMs)已在广泛任务中展现出卓越能力，然而它们在跨视图地理定位和姿态估计领域的知识和能力仍未被探索，尽管这些能力对导航、自动驾驶、户外机器人等有潜在益处。为填补这一空白，我们引入了GeoX-Bench，这是一个全面的基准测试，旨在探索和评估LMMs在跨视图地理定位和姿态估计方面的能力。具体而言，GeoX-Bench包含10,859个全景-卫星图像对，覆盖49个国家的128个城市，以及相应的755,976个问答对。其中，42,900个问答对用于基准测试，其余的旨在增强LMMs的能力。基于GeoX-Bench，我们评估了25个最先进的LMMs在跨视图地理定位和姿态估计任务上的能力，并进一步探索了指令调优的赋能能力。我们的基准测试表明，虽然当前LMMs在地理定位任务上取得了令人印象深刻的性能，但在更复杂的姿态估计任务上效果显著下降，突显了一个需要未来改进的关键领域，而在GeoX-Bench的训练数据上对LMMs进行指令调优可以显著提高其跨视图地理感知能力。GeoX-Bench可在https://github.com/IntMeGroup/GeoX-Bench获取。


### 论文摘要

Large multimodal models (LMMs) have demonstrated remarkable capabilities across a wide range of tasks, however their knowledge and abilities in the cross-view geo-localization and pose estimation domains remain unexplored, despite potential benefits for navigation, autonomous driving, outdoor robotics, \textit{etc}. To bridge this gap, we introduce \textbf{GeoX-Bench}, a comprehensive \underline{Bench}mark designed to explore and evaluate the capabilities of LMMs in \underline{cross}-view \underline{Geo}-localization and pose estimation. Specifically, GeoX-Bench contains 10,859 panoramic-satellite image pairs spanning 128 cities in 49 countries, along with corresponding 755,976 question-answering (QA) pairs. Among these, 42,900 QA pairs are designated for benchmarking, while the remaining are intended to enhance the capabilities of LMMs. Based on GeoX-Bench, we evaluate the capabilities of 25 state-of-the-art LMMs on cross-view geo-localization and pose estimation tasks, and further explore the empowered capabilities of instruction-tuning. Our benchmark demonstrate that while current LMMs achieve impressive performance in geo-localization tasks, their effectiveness declines significantly on the more complex pose estimation tasks, highlighting a critical area for future improvement, and instruction-tuning LMMs on the training data of GeoX-Bench can significantly improve the cross-view geo-sense abilities. The GeoX-Bench is available at \textcolor{magenta}{https://github.com/IntMeGroup/GeoX-Bench}.

---

## 97. 论文ID: 2511.13507v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.13507v1.json'

---

## 98. Descriptor: Distance-Annotated Traffic Perception Question Answering (DTPQA)

**论文链接:** [http://arxiv.org/abs/2511.13397v1](http://arxiv.org/abs/2511.13397v1)

**作者:** Nikos Theodoridis, Tim Brophy, Reenu Mohandas, Ganesh Sistu, Fiachra Collins, Anthony Scanlan, Ciaran Eising

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为距离标注的交通感知问答(DTPQA)的视觉问答基准，用于评估视觉语言模型(VLMs)在自动驾驶场景中的感知能力，特别是在不同距离下的表现。

### 背景

视觉语言模型(VLMs)在各种任务上取得了显著进展，引起了其在自动驾驶应用中的兴趣。然而，在安全关键领域，这些模型必须具备强大的感知能力，能够理解复杂的交通场景，其中关键物体和代理通常位于远处。

### 目的

评估VLMs在交通场景中的感知能力，与其他技能如推理或高级世界知识分开评估，并分析模型性能如何随着物体距离的增加而下降。

### 方法

创建了距离标注的交通感知问答(DTPQA)基准，包括合成基准(DTP-Synthetic)和真实世界基准(DTP-Real)。每个样本包含图像、问题、真实答案和物体距离信息，使研究人员能够评估VLM在不同距离下的感知能力。

### 主要发现

DTPQA基准可以用于分析VLM性能如何随着物体距离的增加而下降，为评估自动驾驶场景中VLM的感知能力提供了工具。

### 结论

本文提供了DTPQA数据集和用于创建它的Python脚本，这些工具可用于生成相同类型的额外数据，促进对VLM在自动驾驶中感知能力的进一步研究。

### 翻译

视觉语言模型(VLMs)在各种任务上的显著进展引起了其在自动驾驶应用中的兴趣。然而，为了使这些模型在安全关键领域被信任，它们必须首先具备强大的感知能力，即能够理解交通场景，这些场景往往非常复杂，同时发生很多事情。此外，由于交通场景中的关键物体和代理通常位于远处，我们需要系统不仅在近距离(最多20米)具有强大感知能力，在远距离(30+米)也具有强大感知能力。因此，将这些模型的感知能力与其他技能如推理或高级世界知识分开评估非常重要。距离标注的交通感知问答(DTPQA)是一个为此目的专门设计的视觉问答(VQA)基准：它可以使用与驾驶决策相关但简单而关键的问题来评估VLMs在交通场景中的感知系统。它由两部分组成：使用模拟器创建的合成基准(DTP-Synthetic)和建立在现有真实交通场景图像之上的真实世界基准(DTP-Real)。此外，DTPQA包括距离标注，即所讨论物体离相机的距离。更具体地说，每个DTPQA样本至少包括：(a)图像，(b)问题，(c)真实答案，和(d)所讨论物体的距离，使分析VLM性能如何随着物体距离增加而下降成为可能。在本文中，我们提供了数据集本身和用于创建它的Python脚本，可用于生成相同类型的额外数据。


### 论文摘要

The remarkable progress of Vision-Language Models (VLMs) on a variety of tasks has raised interest in their application to automated driving. However, for these models to be trusted in such a safety-critical domain, they must first possess robust perception capabilities, i.e., they must be capable of understanding a traffic scene, which can often be highly complex, with many things happening simultaneously. Moreover, since critical objects and agents in traffic scenes are often at long distances, we require systems with not only strong perception capabilities at close distances (up to 20 meters), but also at long (30+ meters) range. Therefore, it is important to evaluate the perception capabilities of these models in isolation from other skills like reasoning or advanced world knowledge. Distance-Annotated Traffic Perception Question Answering (DTPQA) is a Visual Question Answering (VQA) benchmark designed specifically for this purpose: it can be used to evaluate the perception systems of VLMs in traffic scenarios using trivial yet crucial questions relevant to driving decisions. It consists of two parts: a synthetic benchmark (DTP-Synthetic) created using a simulator, and a real-world benchmark (DTP-Real) built on top of existing images of real traffic scenes. Additionally, DTPQA includes distance annotations, i.e., how far the object in question is from the camera. More specifically, each DTPQA sample consists of (at least): (a) an image, (b) a question, (c) the ground truth answer, and (d) the distance of the object in question, enabling analysis of how VLM performance degrades with increasing object distance. In this article, we provide the dataset itself along with the Python scripts used to create it, which can be used to generate additional data of the same kind.

---

## 99. Emergence of Multi-Scroll Attractors

**论文链接:** [http://arxiv.org/abs/2511.13332v1](http://arxiv.org/abs/2511.13332v1)

**作者:** Tanmayee Patra, Biplab Ganguli

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了如何通过Nambu广义哈密顿力学理解相空间中翅膀状几何结构的起源，结合数值分析和几何方法解释混沌吸引子中特定数量卷或翅膀的出现机制。

### 背景

相空间轨迹对于理解和分析混沌吸引子具有根本重要性，通常通过直接数值求解动力学方程实现。卷的起源虽可从动力学方程性质理解，但它们在相空间中的出现也可从Nambu表面的几何和相对方向推断，无需直接数值解。

### 目的

应用Nambu广义哈密顿力学，从几何角度解释相空间中翅膀状几何结构出现的根本原因，并通过数值分析和几何方法相结合来表征混沌。

### 方法

采用分岔图的数值分析和李雅普诺夫指数(LEs)计算来表征混沌，同时应用Nambu广义哈密顿力学方法分析Nambu表面的几何特性。

### 主要发现

特定数量的卷或翅膀可因Nambu表面的特定几何形状而在相空间中出现；参数集改变时会形成不同的几何形状；相空间中的翅膀可归因于类似能量的Nambu表面，提供几何解释。

### 结论

通过结合数值分析和几何方法，展示了Nambu表面几何形状如何决定相空间中混沌吸引子的结构，特别是翅膀状几何结构的形成机制。

### 翻译

相空间轨迹对于理解和分析混沌吸引子具有根本重要性。这主要通过直接数值求解动力学方程来实现。尽管卷的起源可以从动力学方程的性质理解，但它们在相空间中的出现也可以从使用Nambu哈密顿量绘制的Nambu表面的几何和相对方向推断，而无需直接数值解。因此，人们可以将相空间中翅膀的起源归因于类似能量的Nambu表面，从而提供几何解释。在本文中，我们进行了分岔图的数值分析和李雅普诺夫指数(LEs)的计算来表征混沌，并应用Nambu广义哈密顿力学方法来解释相空间中出现翅膀状几何结构的基本原因。我们展示了特定数量的卷或翅膀如何因Nambu表面的特定几何形状而在相空间中出现，以及当参数集改变时如何形成不同的几何形状。


### 论文摘要

Phase space trajectories are fundamentally important for understanding and analysing chaotic attractors. This is mostly carried out by direct numerical solution of the dynamical equations. Though the origin of scrolls can be understood from the properties of dynamical equations, their appearance in the phase space can also be inferred from the geometry and relative orientations of Nambu surfaces, drawn using Nambu Hamiltonians than from direct numerical solutions. Therefore, one can attribute the origin of wings in the phase space due to energy like Nambu surfaces, giving a geometrical interpretation. In this article, we have carried out, both numerical analysis of bifurcation diagram and Lyapunov exponents(LEs) to characterise chaos and geometric approach by applying the Nambu generalized Hamiltonian mechanics to explain the fundamental reason for the appearance of wings like geometry in the phase space. We have shown how a fixed number of scrolls or wings can appear in the phase space due to specific geometry of the Nambu surfaces and how different geometries are formed when set of parameters are changed.

---

## 100. Spatial Blind Spot: Auditory Motion Perception Deficits in Audio LLMs

**论文链接:** [http://arxiv.org/abs/2511.13273v1](http://arxiv.org/abs/2511.13273v1)

**作者:** Zhe Sun, Yujun Cai, Jiayu Yao, Yiwei Wang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究发现当前大型音频语言模型在感知声源运动方面存在系统性缺陷，并提出了首个专门评估听觉运动理解的基准测试AMPBench，揭示了模型在听觉空间推理方面的基本局限性。

### 背景

大型音频语言模型(LALMs)在语音识别、音频标注和听觉问答等任务上已取得显著进展，但它们是否能感知空间动态，特别是声源的运动，尚不清楚。

### 目的

调查当前音频语言模型在听觉运动理解方面的能力，并评估它们是否能推断移动声源的方向和轨迹。

### 方法

引入AMPBench，这是第一个专门设计用于评估听觉运动理解的基准测试，通过控制问答任务评估模型从双耳音频中推断声源运动的能力，并进行定性和定量分析。

### 主要发现

当前音频语言模型难以可靠地识别运动线索或区分方向模式，平均准确率低于50%，表明模型在听觉空间推理方面存在基本局限性。

### 结论

研究强调了人类和模型在听觉空间推理之间的基本差距，AMPBench为未来增强音频语言模型的空间认知提供了诊断工具和新见解。

### 翻译

大型音频语言模型(LALMs)最近在语音识别、音频标注和听觉问答方面展示了令人印象深刻的进展。然而，这些模型是否能感知空间动态，特别是声源的运动，仍然不清楚。在这项工作中，我们揭示了当前LALMs中存在系统性的运动感知缺陷。为了研究这个问题，我们引入了AMPBench，这是第一个专门设计用于评估听觉运动理解的基准测试。AMPBench引入了一个受控问答基准，用于评估音频语言模型是否能从双耳音频中推断移动声源的方向和轨迹。全面的定量和定性分析表明，当前模型难以可靠地识别运动线索或区分方向模式。平均准确率低于50%，突显了听觉空间推理的基本局限性。我们的研究强调了人类和模型听觉空间推理之间的基本差距，为未来增强音频语言模型的空间认知提供了诊断工具和新见解。


### 论文摘要

Large Audio-Language Models (LALMs) have recently shown impressive progress in speech recognition, audio captioning, and auditory question answering. Yet, whether these models can perceive spatial dynamics, particularly the motion of sound sources, remains unclear. In this work, we uncover a systematic motion perception deficit in current ALLMs. To investigate this issue, we introduce AMPBench, the first benchmark explicitly designed to evaluate auditory motion understanding. AMPBench introduces a controlled question-answering benchmark designed to evaluate whether Audio-Language Models (LALMs) can infer the direction and trajectory of moving sound sources from binaural audio. Comprehensive quantitative and qualitative analyses reveal that current models struggle to reliably recognize motion cues or distinguish directional patterns. The average accuracy remains below 50%, underscoring a fundamental limitation in auditory spatial reasoning. Our study highlights a fundamental gap between human and model auditory spatial reasoning, providing both a diagnostic tool and new insight for enhancing spatial cognition in future Audio-Language Models.

---

## 101. Video Spatial Reasoning with Object-Centric 3D Rollout

**论文链接:** [http://arxiv.org/abs/2511.13190v1](http://arxiv.org/abs/2511.13190v1)

**作者:** Haoran Tang, Meng Cao, Ruyang Liu, Xiaoxi Liang, Linglong Li, Ge Li, Xiaodan Liang

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种名为以物体为中心的3D展开策略（OCR）的新方法，解决了多模态大语言模型在视频空间推理方面的局限性，使模型能够更全面地理解动态3D场景中的物体关系。

### 背景

多模态大语言模型在视觉语言理解方面展现出了显著能力，但在视频空间推理方面仍存在关键挑战，即理解动态3D场景中物体的位置、方向和物体间关系。

### 目的

解决现有模型只关注提示中明确提到的物体而忽略重要上下文线索的局限性，提升模型对整个场景的整体推理能力。

### 方法

提出OCR策略，在训练过程中对选定物体的3D几何结构引入结构化扰动，降低物体特定视觉线索并将改变的几何结构投影到2D空间，迫使模型进行整体场景推理；同时设计基于展开的训练流程，利用普通视频和区域噪声视频优化空间推理轨迹。

### 主要发现

在VSI-Bench基准测试上，30亿参数的OCR模型达到了47.5%的准确率，性能优于几个70亿参数的基线模型；消融研究证实OCR优于先前的展开策略如T-GRPO和NoisyRollout。

### 结论

OCR方法能够有效提升多模态大语言模型在视频空间推理方面的能力，即使参数量较小，也能实现优于更大模型的性能。

### 翻译

多模态大语言模型（MLLMs）的最新进展在视觉语言理解方面展现出了卓越能力。然而，实现稳健的视频空间推理——即在动态3D场景中理解物体位置、方向和物体间关系的能力——仍然是一个关键未解决的挑战。现有方法主要依赖于空间监督微调或强化学习，但我们观察到这些模型通常表现出查询锁定推理，仅狭隘地关注提示中明确提到的物体，而忽略关键的上下文线索。为解决这一局限性，我们提出了以物体为中心的3D展开策略（OCR），一种新颖的训练策略，在训练过程中对选定物体的3D几何结构引入结构化扰动。通过降低物体特定的视觉线索并将改变的几何结构投影到2D空间，OCR迫使模型对整个场景进行整体推理。我们进一步设计了一个基于展开的训练流程，同时利用普通视频和区域噪声视频来优化空间推理轨迹。实验展示了最先进的性能：我们的30亿参数模型在VSI-Bench上达到47.5%的准确率，优于多个70亿参数的基线模型。消融研究证实OCR优于先前的展开策略（如T-GRPO、NoisyRollout）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决多模态大语言模型在视频空间推理中的'查询锁定推理'问题，即模型只关注查询中明确提到的物体，而忽略其他提供重要空间线索的物体。这个问题很重要，因为让AI具备类似人类的3D空间理解能力对于机器人导航、增强现实和具身AI等应用至关重要，同时当前方法存在明显局限性，限制了模型在复杂环境中的表现。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过观察现有方法存在查询锁定推理问题，发现模型过度依赖查询中明确提到的物体而忽略其他空间相关信息。他们借鉴了现有的GRPO框架、NoisyRollout和T-GRPO等方法，但针对3D空间推理的特殊性，专门设计了对象中心3D展开策略。核心思路是通过有选择地破坏特定物体的空间完整性，迫使模型依赖上下文关系和更全面地理解场景来进行空间推理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过向选定的3D物体引入结构化扰动，破坏其视觉线索，迫使模型依赖周围环境和物体间关系来进行整体空间推理，而不是仅仅依赖查询中明确提到的物体。整体流程包括：1)对象引导的空间扰动 - 向选定物体注入高斯噪声并投影回2D空间；2)逐步退火对象选择器 - 逐渐减少扰动物体数量和噪声强度；3)训练流程 - 先在OCR-SFT上做SFT，再在OCR-RL上使用改进的GRPO，同时使用原始和扰动视频生成推理轨迹，但只使用原始视频的轨迹更新模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)对象中心3D展开策略(OCR) - 首次向特定3D物体注入结构化噪声；2)逐步退火对象选择器 - 训练中逐渐减少扰动；3)高质量数据集构建 - OCR-SFT和OCR-RL覆盖更广泛空间感知类型。相比之前工作，不同之处在于：NoisyRollout向整个图像注入噪声，而OCR有选择地向特定物体区域注入；T-GRPO使用洗牌视频序列，而OCR保持时序一致性但修改特定物体；SpaceR使用2D网格布局，而OCR直接在3D空间中进行扰动。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '通过提出对象中心3D展开策略，论文成功解决了多模态大语言模型在视频空间推理中的查询锁定问题，使模型能够依赖全局场景上下文进行更准确的空间推理，并在VSI-Bench基准测试上实现了超越多个7B参数模型的性能。'}


### 论文摘要

Recent advances in Multi-modal Large Language Models (MLLMs) have showcased remarkable capabilities in vision-language understanding. However, enabling robust video spatial reasoning-the ability to comprehend object locations, orientations, and inter-object relationships in dynamic 3D scenes-remains a key unsolved challenge. Existing approaches primarily rely on spatially grounded supervised fine-tuning or reinforcement learning, yet we observe that such models often exhibit query-locked reasoning, focusing narrowly on objects explicitly mentioned in the prompt while ignoring critical contextual cues. To address this limitation, we propose Object-Centric 3D Rollout (OCR), a novel strategy that introduces structured perturbations to the 3D geometry of selected objects during training. By degrading object-specific visual cues and projecting the altered geometry into 2D space, OCR compels the model to reason holistically across the entire scene. We further design a rollout-based training pipeline that jointly leverages vanilla and region-noisy videos to optimize spatial reasoning trajectories. Experiments demonstrate state-of-the-art performance: our 3B-parameter model achieves 47.5% accuracy on VSI-Bench, outperforming several 7B baselines. Ablations confirm OCR's superiority over prior rollout strategies (e.g., T-GRPO, NoisyRollout).

---

## 102. Decoupling Scene Perception and Ego Status: A Multi-Context Fusion Approach for Enhanced Generalization in End-to-End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.13079v1](http://arxiv.org/abs/2511.13079v1)

**作者:** Jiacheng Tang, Mingyue Feng, Jiachao Liu, Yaonong Wang, Jian Pu

**发布时间:** 2025-11-17

**备注:** 11 pages, 8 figures

### GPT解析

### 总结

本文提出AdaptiveAD架构，通过双分支结构解耦场景感知和自身状态，显著提高自动驾驶系统的泛化能力和鲁棒性。

### 背景

面向规划的自动驾驶模块化设计已推动端到端系统发展，但现有架构过度依赖自身状态，限制了泛化和鲁棒场景理解能力。

### 目的

解决现有自动驾驶架构中过度依赖自身状态的问题，提高系统的泛化能力和鲁棒场景理解能力。

### 方法

提出AdaptiveAD架构，采用双分支结构：一个分支进行场景驱动推理(从BEV编码器中省略自身状态)，另一个分支进行自身状态驱动推理；通过场景感知融合模块整合两分支决策；引入路径注意力机制和两个辅助任务(BEV单向蒸馏和自回归在线映射)。

### 主要发现

现有架构的根本问题是BEV编码器上游过早融合自身状态，导致强先验信息流主导下游规划模块；AdaptiveAD显著减轻了对自身状态的过度依赖，在各种场景中展现出令人印象深刻的泛化能力。

### 结论

AdaptiveAD在nuScenes数据集上实现了最先进的开环规划性能，证明了该架构在提高自动驾驶系统泛化能力和鲁棒性方面的有效性。

### 翻译

面向规划的自动驾驶模块化设计已显著推动端到端系统发展。然而，现有架构仍受过度依赖自身状态的限制，阻碍了泛化和鲁棒场景理解。我们确定根本原因是这些架构中的固有设计允许自身状态被轻易用作捷径。具体而言，BEV编码器上游过早融合自身状态，允许这种强先验信息流主导下游规划模块。为应对这一挑战，我们提出AdaptiveAD，这是一种基于多上下文融合策略的架构级解决方案。其核心是双分支结构，明确解耦场景感知和自身状态。一个分支基于多任务学习进行场景驱动推理，但故意从BEV编码器中省略自身状态，而另一个分支仅基于规划任务进行自身状态驱动推理。然后，场景感知融合模块自适应地整合两个分支的互补决策，形成最终规划轨迹。为确保这种解耦不会损害多任务学习，我们引入了用于自身状态-BEV交互的路径注意力机制，并添加了两个有针对性的辅助任务：BEV单向蒸馏和自回归在线映射。在nuScenes数据集上的广泛评估表明，AdaptiveAD实现了最先进的开环规划性能。重要的是，它显著减轻了对自身状态的过度依赖，并在各种场景中展现出令人印象深刻的泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决端到端自动驾驶系统中对'自我状态'(如车辆速度、方向等)过度依赖的问题。这个问题在现实中非常重要，因为过度依赖自我状态会导致模型在复杂场景中表现不佳，特别是在紧急情况下可能引发安全事故。当自我状态信息不准确或不可靠时，系统性能会急剧下降，无法基于环境做出适当决策，限制了自动驾驶系统在真实世界中的可靠性和安全性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析发现问题的根本原因是架构设计缺陷，而非简单的数据偏差。他们提出通过'解耦'场景感知和自我状态来解决这个问题。设计上借鉴了VAD框架作为基础，并融入了知识蒸馏技术用于特征学习，以及自回归机制用于建立反馈循环。作者不是完全原创，而是在现有方法基础上进行了创新性改进，特别是对架构层面的重新设计，通过双分支架构明确分离场景感知和自我状态的影响。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过双分支架构解耦场景感知和自我状态的影响，然后通过场景感知融合模块自适应地整合这两个分支的互补决策。整体流程包括：1)多上下文决策生成：场景驱动分支(排除自我状态)和自我状态驱动分支；2)多上下文决策融合：使用场景感知初始化、上下文对齐和交叉注意力整合两个分支的决策；3)辅助任务：BEV单向蒸馏减轻运动模糊，自回归在线映射确保感知地图与规划轨迹的一致性；4)多任务学习框架联合优化所有任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)架构层面的解耦设计，通过双分支架构分离场景感知和自我状态；2)路径注意力机制，沿假设未来路径进行语义采样；3)BEV单向蒸馏，通过知识转移减轻运动模糊；4)自回归在线映射，建立规划到映射的反馈循环。相比之前工作，AdaptiveAD从架构层面而非数据或表示层面解决问题，完全移除了BEV编码器中的自我特征融合，在决策层面而非早期阶段融合信息，实验显示其在复杂场景和自我状态不可靠时表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AdaptiveAD通过双分支架构解耦场景感知与自我状态，并采用多上下文融合策略，显著提高了端到端自动驾驶系统在复杂场景中的泛化能力和鲁棒性，同时解决了模型过度依赖车辆自身状态而忽视环境理解的根本性问题。'}


### 论文摘要

Modular design of planning-oriented autonomous driving has markedly advanced end-to-end systems. However, existing architectures remain constrained by an over-reliance on ego status, hindering generalization and robust scene understanding. We identify the root cause as an inherent design within these architectures that allows ego status to be easily leveraged as a shortcut. Specifically, the premature fusion of ego status in the upstream BEV encoder allows an information flow from this strong prior to dominate the downstream planning module. To address this challenge, we propose AdaptiveAD, an architectural-level solution based on a multi-context fusion strategy. Its core is a dual-branch structure that explicitly decouples scene perception and ego status. One branch performs scene-driven reasoning based on multi-task learning, but with ego status deliberately omitted from the BEV encoder, while the other conducts ego-driven reasoning based solely on the planning task. A scene-aware fusion module then adaptively integrates the complementary decisions from the two branches to form the final planning trajectory. To ensure this decoupling does not compromise multi-task learning, we introduce a path attention mechanism for ego-BEV interaction and add two targeted auxiliary tasks: BEV unidirectional distillation and autoregressive online mapping. Extensive evaluations on the nuScenes dataset demonstrate that AdaptiveAD achieves state-of-the-art open-loop planning performance. Crucially, it significantly mitigates the over-reliance on ego status and exhibits impressive generalization capabilities across diverse scenarios.

---

## 103. A Smart-Glasses for Emergency Medical Services via Multimodal Multitask Learning

**论文链接:** [http://arxiv.org/abs/2511.13078v1](http://arxiv.org/abs/2511.13078v1)

**作者:** Liuyi Jin, Pasan Gunawardena, Amran Haroon, Runzhi Wang, Sangwoo Lee, Radu Stoleru, Michael Middleton, Zepeng Huo, Jeeeun Kim, Jason Moats

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出EMSGlass智能眼镜系统，结合EMSNet多模态多任务模型和EMSServe低延迟服务框架，帮助急救医疗技术人员在高压力环境下做出更快速、准确的决策。

### 背景

急救医疗技术人员(EMTs)在高压力环境中工作，需要在认知和操作负荷沉重的情况下做出快速、关键生命的决策。

### 目的

开发一个智能眼镜系统(EMSGlass)，通过多模态AI技术提升急救医疗人员的实时态势感知、决策速度和操作效率。

### 方法

构建EMSNet多模态多任务模型整合文本、生命体征和现场图像；开发EMSServe服务框架实现低延迟推理；使用真实世界EMS数据集进行训练；对6名专业EMTs进行用户研究评估。

### 主要发现

EMSNet可同时支持5个关键EMS任务，准确率优于单模态基线；EMSServe实现1.9倍至11.7倍的推理加速；用户研究表明EMSGlass显著提升了急救人员的态势感知和决策效率。

### 结论

EMSGlass为下一代AI支持的EMS系统提供了可行方案，有效将多模态智能与现实应急响应工作流程结合。

### 翻译

急救医疗技术人员(EMTs)在高压力环境中工作，在沉重的认知和操作负荷下做出快速、关键生命的决策。我们提出了EMSGlass，一个由EMSNet（首个用于紧急医疗服务(EMS)的多模态多任务模型）和EMSServe（专为EMS场景设计的低延迟多模态服务框架）驱动的智能眼镜系统。EMSNet整合文本、生命体征和现场图像，构建对EMS事件的统一实时理解。在真实世界多模态EMS数据集上训练的EMSNet，同时支持多达5个关键EMS任务，准确率优于最先进的单模态基线。建立在PyTorch基础上的EMSServe引入了模态感知模型分割器和特征缓存机制，在异构硬件上实现了自适应和高效的推理，同时解决了现场模态异步到达的挑战。通过优化EMS场景中的多模态推理执行，EMSServe比直接PyTorch多模态推理实现了1.9倍至11.7倍的加速。六名专业EMTs的用户研究评估表明，EMSGlass通过直观的眼镜交互增强了实时态势感知、决策速度和操作效率。此外，用户研究的定性见解为将EMSGlass扩展到下一代AI支持的EMS系统提供了可行的方向，将多模态智能与现实世界的应急响应工作流程相结合。


### 论文摘要

Emergency Medical Technicians (EMTs) operate in high-pressure environments, making rapid, life-critical decisions under heavy cognitive and operational loads. We present EMSGlass, a smart-glasses system powered by EMSNet, the first multimodal multitask model for Emergency Medical Services (EMS), and EMSServe, a low-latency multimodal serving framework tailored to EMS scenarios. EMSNet integrates text, vital signs, and scene images to construct a unified real-time understanding of EMS incidents. Trained on real-world multimodal EMS datasets, EMSNet simultaneously supports up to five critical EMS tasks with superior accuracy compared to state-of-the-art unimodal baselines. Built on top of PyTorch, EMSServe introduces a modality-aware model splitter and a feature caching mechanism, achieving adaptive and efficient inference across heterogeneous hardware while addressing the challenge of asynchronous modality arrival in the field. By optimizing multimodal inference execution in EMS scenarios, EMSServe achieves 1.9x -- 11.7x speedup over direct PyTorch multimodal inference. A user study evaluation with six professional EMTs demonstrates that EMSGlass enhances real-time situational awareness, decision-making speed, and operational efficiency through intuitive on-glass interaction. In addition, qualitative insights from the user study provide actionable directions for extending EMSGlass toward next-generation AI-enabled EMS systems, bridging multimodal intelligence with real-world emergency response workflows.

---

## 104. A Global Spacetime Optimization Approach to the Real-Space Time-Dependent Schrödinger Equation

**论文链接:** [http://arxiv.org/abs/2511.12983v1](http://arxiv.org/abs/2511.12983v1)

**作者:** Enze Hou, Yuzhi Liu, Lei Wang, Han Wang

**发布时间:** 2025-11-17

**备注:** 23 pages, 5 figures

### GPT解析

### 总结

该论文提出了一种名为'费米子反称时空网络'的神经网络框架，用于解决实空间中的时间依赖薛定谔方程，实现了对复杂费米系统动力学的准确模拟

### 背景

时间依赖的薛定谔方程在实空间中是理解多电子量子系统动力学的基础，应用范围从量子化学到凝聚态物理和材料科学。然而，解决复杂费米系统的TDSE仍面临重大挑战，特别是需要捕捉随时间演化的多体相关性，同时费米波函数的反对称性使解决方案的函数空间复杂化

### 目的

开发一个通用的神经网络框架来解决实空间TDSE，将时间作为与空间坐标并列的显式输入，实现复杂、反称波函数的统一时空表示

### 方法

提出'费米子反称时空网络'框架，将TDSE表述为全局优化问题，避免逐步传播并支持高度并行化的训练。在四个基准问题上进行了验证：1D谐振子、时变谐振子中的相互作用费米子、3D氢轨道动力学和激光驱动的H2分子

### 主要发现

在所有案例中与参考解达成极好的一致，确认了该方法在各种维度和相互作用 regime 下的可扩展性、准确性和灵活性，展示了准确模拟复杂系统长时间动力学的能力

### 结论

该框架提供了传统依赖于基或平均场方法的高度表达性替代方案，为从头算模拟时间依赖量子系统开辟了新的可能性，可应用于量子动力学、分子控制和超快光谱学

### 翻译

实空间中的时间依赖薛定谔方程对于理解多电子量子系统的动力学至关重要，其应用范围从量子化学到凝聚态物理和材料科学。然而，解决复杂费米系统的TDSE仍然是一个重大挑战，特别是由于需要捕捉随时间演化的多体相关性，而费米波函数的反称性使解决方案必须在其内表示的函数空间复杂化。我们提出了一种用于解决实空间TDSE的通用神经网络框架——费米子反称时空网络，该方法将时间作为与空间坐标并列的显式输入，实现了费米系统复杂、反称波函数的统一时空表示。这种方法将TDSE表述为全局优化问题，避免了逐步传播并支持高度并行化的训练。该方法在四个基准问题上进行了验证：一维谐振子、时变谐振子中的相互作用费米子、三维氢轨道动力学和激光驱动的氢分子，在所有情况下都与参考解达成极好的一致。这些结果证实了我们的方法在各种维度和相互作用 regime 下的可扩展性、准确性和灵活性，同时展示了其准确模拟复杂系统长时间动力学的能力。我们的框架为传统的依赖于基或平均场的方法提供了高度表达性的替代方案，为时间依赖量子系统的从头算模拟开辟了新的可能性，应用于量子动力学、分子控制和超快光谱学


### 论文摘要

The time-dependent Schrödinger equation (TDSE) in real space is fundamental to understanding the dynamics of many-electron quantum systems, with applications ranging from quantum chemistry to condensed matter physics and materials science. However, solving the TDSE for complex fermionic systems remains a significant challenge, particularly due to the need to capture the time-evolving many-body correlations, while the antisymmetric nature of fermionic wavefunctions complicates the function space in which these solutions must be represented. We propose a general-purpose neural network framework for solving the real-space TDSE, Fermionic Antisymmetric Spatio-Temporal Network, which treats time as an explicit input alongside spatial coordinates, enabling a unified spatiotemporal representation of complex, antisymmetric wavefunctions for fermionic systems. This approach formulates the TDSE as a global optimization problem, avoiding step-by-step propagation and supporting highly parallelizable training. The method is demonstrated on four benchmark problems: a 1D harmonic oscillator, interacting fermions in a time-dependent harmonic trap, 3D hydrogen orbital dynamics, and a laser-driven H$_2$ molecule, achieving excellent agreement with reference solutions across all cases. These results confirm our method's scalability, accuracy, and flexibility across various dimensions and interaction regimes, while demonstrating its ability to accurately simulate long-time dynamics in complex systems. Our framework offers a highly expressive alternative to traditional basis-dependent or mean-field methods, opening new possibilities for ab initio simulations of time-dependent quantum systems, with applications in quantum dynamics, molecular control, and ultrafast spectroscopy.

---

## 105. Visual Room 2.0: Seeing is Not Understanding for MLLMs

**论文链接:** [http://arxiv.org/abs/2511.12928v1](http://arxiv.org/abs/2511.12928v1)

**作者:** Haokun Li, Yazhou Zhang, Jizhi Ding, Qiuchi Li, Peng Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了多模态大语言模型(MLLMs)是否真正理解它们所看到的内容，提出'视觉房间'论证，并开发了分层评估基准来验证'看见不等于理解'的假设。

### 背景

将塞尔的'中文房间'思想扩展到多模态领域，指出MLLMs可能精确描述视觉细节但无法理解潜在的情感和意图。

### 目的

评估MLLMs的感知-认知一致性，将'看见不等于理解'操作化为可测试的假设，为MLLMs从感知处理到认知推理提供新范式。

### 方法

引入'视觉房间'2.0分层评估基准，模拟人类感知和认知过程的三个层次（低、中、高），涵盖17个代表性任务，包含350个多模态样本和2,100个渐进式问题，评估10个最先进的MLLMs。

### 主要发现

MLLMs的感知能力比认知能力强8.0%；认知似乎不依赖于基于感知的推理；认知能力随模型规模扩大而提升，但感知能力并不随模型变大而一致提升。

### 结论

成功将'看见≠理解'操作化为可测试假设，为MLLMs从感知处理到认知推理提供了新的评估范式和基准数据集。

### 翻译

多模态大语言模型(MLLMs)真正能理解它们所看到的内容吗？将塞尔的中文房间扩展到多模态领域，本文提出视觉房间论证：MLLMs可能精确描述每一个视觉细节却无法理解潜在的情感和意图，即看见不等于理解。基于此，我们引入视觉房间2.0，一个评估MLLMs感知-认知一致性的分层基准。我们模拟人类感知和认知过程的三个层次：低、中、高，涵盖17个代表性任务。感知组件从属性识别到场景理解，认知组件从文本蕴含到因果和社会推理。数据集包含350个多模态样本，每个样本有六个渐进式问题（总计2,100个），从感知到认知。评估10个最先进的MLLMs，我们突出了三个关键发现：(1)MLLMs展示出比认知能力更强的感知能力（高出8.0%）；(2)认知似乎不依赖于基于感知的推理；(3)认知能力随模型规模扩大而提升，但感知能力并不随模型变大而一致提升。本研究将'看见≠理解'操作化为可测试假设，为MLLMs从感知处理到认知推理提供了新范式。我们的数据集可在https://huggingface.co/datasets/LHK2003/PCBench获取。


### 论文摘要

Can multi-modal large language models (MLLMs) truly understand what they can see? Extending Searle's Chinese Room into the multi-modal domain, this paper proposes the Visual Room argument: MLLMs may describe every visual detail precisely yet fail to comprehend the underlying emotions and intentions, namely seeing is not understanding. Building on this, we introduce \textit{Visual Room} 2.0, a hierarchical benchmark for evaluating perception-cognition alignment of MLLMs. We model human perceptive and cognitive processes across three levels: low, middle, and high, covering 17 representative tasks. The perception component ranges from attribute recognition to scene understanding, while the cognition component extends from textual entailment to causal and social reasoning. The dataset contains 350 multi-modal samples, each with six progressive questions (2,100 in total) spanning perception to cognition. Evaluating 10 state-of-the-art (SoTA) MLLMs, we highlight three key findings: (1) MLLMs exhibit stronger perceptual competence than cognitive ability (8.0\%$\uparrow$); (2) cognition appears not causally dependent on perception-based reasoning; and (3) cognition scales with model size, but perception does not consistently improve with larger variants. This work operationalizes Seeing $\ne$ Understanding as a testable hypothesis, offering a new paradigm from perceptual processing to cognitive reasoning in MLLMs. Our dataset is available at https://huggingface.co/datasets/LHK2003/PCBench.

---

## 106. On the Fundamental Limits of LLMs at Scale

**论文链接:** [http://arxiv.org/abs/2511.12869v1](http://arxiv.org/abs/2511.12869v1)

**作者:** Muhammad Ahmed Mohsin, Muhammad Umer, Ahsan Bilal, Zeeshan Memon, Muhammad Ibtsaam Qadir, Sagnik Bhattacharya, Hassan Rizwan, Abhiram R. Gorle, Maahe Zehra Kazmi, Ayesha Mohsin, Muhammad Usman Rafique, Zihao He, Pulkit Mehta, Muhammad Ali Jamshed, John M. Cioffi

**发布时间:** 2025-11-17

**备注:** Submitted to TMLR 2025

### GPT解析

### 总结

本研究提出了一个统一的理论框架，形式化了大型语言模型扩展的内在理论上限，通过计算理论、信息理论和学习理论的角度分析了大型语言模型面临的五个基本限制。

### 背景

大型语言模型通过扩展获得了巨大收益，但这些收益受到五个基本限制的约束：(1)幻觉，(2)上下文压缩，(3)推理能力退化，(4)检索脆弱性，(5)多模态不对齐。现有研究缺乏将这些现象与计算、信息和学习基础限制严格联系的理论综合。

### 目的

弥合现有研究差距，通过提出一个统一、基于证明的框架，形式化大型语言模型扩展的内在理论上限，阐明扩展在哪些方面有帮助、在哪些方面会饱和、以及在哪些方面无法取得进展。

### 方法

采用理论分析与实证证据相结合的方法，从三个主要角度分析：(1)计算可计算性和不可计算性理论，(2)信息论和统计约束，(3)几何和计算效应。

### 主要发现

1)计算理论意味着存在不可约的错误残留；2)信息论和统计约束限制了可达到的准确性；3)几何和计算效应导致长上下文被过度压缩；4)基于似然的训练倾向于模式完成而非推理；5)检索和多模态扩展存在特定问题。

### 结论

研究通过定理和实证证据配对，概述了扩展的边界，同时提供了理论基础和实用的缓解路径，如有界检索、位置课程和稀疏或分层注意力等。

### 翻译

大型语言模型(LLMs)从扩展中受益匪浅，但这些收益受到五个基本限制的约束：(1)幻觉，(2)上下文压缩，(3)推理能力退化，(4)检索脆弱性，(5)多模态不对齐。虽然现有的调查经验性地描述了这些现象，但它们缺乏将这些现象与计算、信息和学习基础限制严格联系的理论综合。本研究通过提出一个统一、基于证明的框架来弥合这一差距，形式化了大型语言模型扩展的内在理论上限。首先，可计算性和不可计算性意味着存在不可约的错误残留：对于任何可计算枚举的模型族，对角化保证某些输入上模型必须失败，不可判定查询(如停机类任务)对所有可计算预测器导致无限失败集合。其次，信息论和统计约束限制了即使在可判定任务上可达到的准确性，有限描述长度强制压缩错误，长尾事实知识需要禁止性的样本复杂度。第三，几何和计算效应由于位置训练不足、编码衰减和softmax拥挤，将长上下文压缩到远低于其标称大小。我们还进一步展示了基于似然的训练如何倾向于模式完成而非推理，检索在令牌限制下如何遭受语义漂移和耦合噪声，以及多模态扩展如何继承浅层跨模态对齐。在各章节中，我们将定理和实证证据配对，概述了扩展在哪些方面有帮助、在哪些方面会饱和、以及在哪些方面无法取得进展，同时提供了理论基础和实用的缓解路径，如有界检索、位置课程和稀疏或分层注意力。


### 论文摘要

Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.

---

## 107. Scalable learning of macroscopic stochastic dynamics

**论文链接:** [http://arxiv.org/abs/2511.12842v1](http://arxiv.org/abs/2511.12842v1)

**作者:** Mengyi Chen, Pengru Huang, Kostya S. Novoselov, Qianxiao Li

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种仅使用小系统模拟来学习大型随机微观系统宏观动力学的框架，通过部分演化方案、识别闭合变量、使用自定义损失函数以及分层上采样方法，实现了对空间扩展系统的准确宏观动力学描述。

### 背景

复杂物理系统的宏观动力学描述对理解和控制材料行为至关重要。随着数据和计算资源的丰富，机器学习已成为基于第一性原理方法的有前景替代方案，可从微观轨迹模拟中构建准确宏观模型。

### 目的

提出一个框架，仅使用小系统模拟来学习大型随机微观系统的宏观动力学，解决空间扩展系统直接模拟的困难。

### 方法

采用部分演化方案生成训练数据对；识别与宏观可观测量相关的闭合变量；使用自定义损失函数学习宏观动力学；引入分层上采样方案从小系统轨迹分布高效生成大系统快照。

### 主要发现

通过随机偏微分方程描述的系统、理想化晶格自旋系统和NbMoTa合金系统等多种空间扩展系统的实证，证明了该框架的准确性和鲁棒性。

### 结论

该框架能够有效学习大型随机微观系统的宏观动力学，克服了空间扩展系统直接模拟的困难，为理解和控制材料行为提供了新的工具。

### 翻译

复杂物理系统的宏观动力学描述对于理解和控制材料行为至关重要。随着数据和计算资源的日益丰富，机器学习已成为基于第一性原理方法的一种有前景的替代方案，可以从微观轨迹模拟中构建准确的宏观模型。然而，对于空间扩展系统，直接模拟足够大的微观系统以提供宏观行为信息是难以实现的。在这项工作中，我们提出一个框架，仅使用小系统模拟来学习大型随机微观系统的宏观动力学。我们的框架采用部分演化方案，通过在局部块内演化大系统快照来生成训练数据对。随后，我们识别与宏观可观测量相关的闭合变量，并使用自定义损失函数学习宏观动力学。此外，我们引入了分层上采样方案，能够从小系统轨迹分布高效生成大系统快照。我们通过多种随机空间扩展系统（包括随机偏微分方程描述的系统、理想化的晶格自旋系统和更真实的NbMoTa合金系统）实证证明了该框架的准确性和鲁棒性。


### 论文摘要

Macroscopic dynamical descriptions of complex physical systems are crucial for understanding and controlling material behavior. With the growing availability of data and compute, machine learning has become a promising alternative to first-principles methods to build accurate macroscopic models from microscopic trajectory simulations. However, for spatially extended systems, direct simulations of sufficiently large microscopic systems that inform macroscopic behavior is prohibitive. In this work, we propose a framework that learns the macroscopic dynamics of large stochastic microscopic systems using only small-system simulations. Our framework employs a partial evolution scheme to generate training data pairs by evolving large-system snapshots within local patches. We subsequently identify the closure variables associated with the macroscopic observables and learn the macroscopic dynamics using a custom loss. Furthermore, we introduce a hierarchical upsampling scheme that enables efficient generation of large-system snapshots from small-system trajectory distributions. We empirically demonstrate the accuracy and robustness of our framework through a variety of stochastic spatially extended systems, including those described by stochastic partial differential equations, idealised lattice spin systems, and a more realistic NbMoTa alloy system.

---

## 108. Entropic alignment of topologically modified ring polymers in cylindrical confinement

**论文链接:** [http://arxiv.org/abs/2511.12813v1](http://arxiv.org/abs/2511.12813v1)

**作者:** Sanjay Bhandarkar, Debarshi Mitra, Jürgen Horbach, Apratim Chatterji

**发布时间:** 2025-11-16

### GPT解析

### 总结

研究揭示了在高圆柱形限制条件下，通过在环状聚合物内部引入不同大小的环状结构，可以调控聚合物的空间取向和排列，这种现象源于熵相互作用而非焓驱动。

### 背景

在高圆柱形限制条件下，环状聚合物的片段可以通过引入内部环沿圆柱长轴局部化，这种现象源于内部环之间的熵排斥作用，这些原则曾被用于解释细菌染色体组织的机制。

### 目的

阐述与熵相互作用相关的功能原理，确定环状聚合物在圆柱限制下相对于邻居的特定取向，并通过修改聚合物拓扑结构来控制聚合物片段的定向排列。

### 方法

通过创建两种不同大小的内部环来修改环状聚合物拓扑结构，制造不对称性；使用仅含排斥体积相互作用的珠弹簧模型模拟柔性聚合物；进行自由能计算验证观察结果。

### 主要发现

聚合物熵性地相互排斥并占据圆柱的不同半区；相邻聚合物优先沿圆柱轴重新取向；出现的空间和取向组织不是由焓驱动；柔性聚合物中出现了类似伊辛自旋间相互作用的有效的熵取向相互作用。

### 结论

这是首个利用熵相互作用在柔性聚合物中产生有效取向相互作用的研究，阐明的原理可能有助于理解大染色体内不同大小环之间的相互作用。

### 翻译

在高圆柱形限制条件下，通过在环状聚合物内部引入内部环，可以使聚合物片段沿着圆柱的长轴局部化。聚合片段的自发组织是由于内部环之间的熵排斥作用。这些原则被用于识别细菌染色体组织的潜在机制。在此，我们阐述了与熵相互作用相关的功能原理，确定了环状聚合物在圆柱形限制条件下相对于其邻居的特定取向。我们通过创建两种不同大小的内部环来修改环状聚合物拓扑结构，从而创造不对称性。这使得我们能够策略性地操控聚合物拓扑，使一个聚合物的片段面向邻近聚合物的特定片段。因此，这些聚合物表现得好像受到类似伊辛自旋间相互作用的'有效'熵相互作用的约束。但这种出现空间和取向组织并非由焓驱动。我们考虑了一个只有单体间排斥体积相互作用的柔性聚合物的珠弹簧模型。聚合物熵性地相互排斥，占据圆柱的不同半区，而且相邻聚合物优先沿着圆柱轴重新取向。我们通过自由能计算进一步证实了我们的观察。据我们所知，这是第一个通过利用柔性聚合物中的熵相互作用来产生有效取向相互作用的研究。这里阐明的原理可能与理解大染色体内不同大小环之间的相互作用相关。


### 论文摘要

Under high cylindrical confinement, segments of ring polymers can be localized along the long axis of the cylinder by introducing internal loops within the ring polymer. The emergent organization of the polymer segments occurs because of the entropic repulsion between internal loops. These principles were used to identify the underlying mechanism of bacterial chromosome organization. Here, we outline functional principles associated with entropic interactions, leading to specific orientations of the ring polymers relative to their neighbors in the cylindrical confinement. We achieve this by modifying the ring polymer topology by creating internal loops of two different sizes within the polymer, and thus create an asymmetry. This allows us to strategically manipulate polymer topology such that segments of a polymer face certain other segments of a neighboring polymer. The polymers therefore behave as if they are subjected to an `effective' entropic interaction reminiscent of interactions between Ising spins. But this emergent spatial and orientational organization is not enthalpy-driven. We consider a bead spring model of flexible polymers with only repulsive excluded volume interactions between the monomers. The polymers entropically repel each other and occupy different halves of the cylinder, and moreover, the adjacent polymers preferentially re-orient themselves along the axis of the cylinder. We further substantiate our observations by free energy calculations. To the best of our knowledge, this is the first study of the emergence of effective orientational interactions by harnessing entropic interactions in flexible polymers. The principles elucidated here could be relevant to understand the interactions between different sized loops within a large chromosome.

---

## 109. A Topological Rewriting of Tarski's Mereogeometry

**论文链接:** [http://arxiv.org/abs/2511.12727v1](http://arxiv.org/abs/2511.12727v1)

**作者:** Patrick Barlatier, Richard Dapoigny

**发布时间:** 2025-11-16

**备注:** This is the full version of the paper accepted at AAAI-26. The arXiv version includes the complete list of authors

### GPT解析

### 总结

该论文解决了基于Goodman风格mereology和伪拓扑学的定性空间模型在高级几何推理中的问题，通过扩展依赖类型理论的形式化，结合Tarski几何的无点解释，构建了一个完整的拓扑空间和几何子空间框架。

### 背景

基于Goodman风格mereology和伪拓扑学的定性空间模型缺乏真正的欧几里得几何和完全发展的拓扑空间，限制了高级几何推理的能力。

### 目的

扩展现有形式化框架，解决定性空间模型缺乏真正欧几里得几何和完全拓扑空间的问题，提高理论的表达能力。

### 方法

使用Coq证明助手和依赖类型理论，在lambda-MM库基础上，通过mereological框架上的拓扑关系代数表述形式化Tarski的实体几何；证明mereological类对应于正则开集，构建拓扑空间；展示Tarski几何作为拓扑子空间，证明Tarski的原始公设，并扩展理论。

### 主要发现

mereological类对应于正则开集，可构建个体名称的拓扑；Tarski的几何形成拓扑子空间，其中区域对应于受限类；证明了Tarski的三个原始公设，简化了公理系统；理论扩展了T2（Hausdorff）性质和额外定义。

### 结论

通过从mereology和几何子空间推导出完整拓扑空间，增加了理论的表达能力，解决了传统定性空间模型在高级几何推理中的局限性。

### 翻译

基于Goodman风格mereology和伪拓扑学的定性空间模型通常对高级几何推理造成问题，因为它们缺乏真正的欧几里得几何和完全发展的拓扑空间。我们通过扩展基于依赖类型理论的现有形式化（使用Coq证明助手）以及Tarski几何的类似Whitehead的无点解释来解决这一问题。更准确地说，我们在名为lambda-MM的库基础上，通过研究mereological框架上的拓扑关系的代数表述，形式化了Tarski的实体几何。由于Tarski的工作根植于Lesniewski的mereology，且lambda-MM目前只提供了Tarski几何的部分实现，论文的第一部分通过证明mereological类对应于正则开集来完成这一框架。这产生了一个个体名称的拓扑，可以用Tarski的几何原语扩展。与定性逻辑理论中的经典方法不同，我们采用了一种从mereology和几何子空间推导出完整拓扑空间的解决方案，从而增加了理论的表达能力。在第二部分，我们展示了Tarski的几何形成了一个拓扑子空间，其中区域对应于受限类。我们还证明了Tarski的三个原始公设，简化了他的公理系统，并用T2（Hausdorff）性质和额外定义扩展了该理论。


### 论文摘要

Qualitative spatial models based on Goodman-style mereology and pseudo-topology often pose problems for advanced geometric reasoning, as they lack true Euclidean geometry and fully developed topological spaces. We address this issue by extending an existing formalization grounded in a dependent type theory using the Coq proof assistant, together with a Whitehead-like point-free interpretation of Tarski's geometry. More precisely, we build on a library called lambda-MM to formalize Tarski's geometry of solids by investigating an algebraic formulation of topological relations on top of the mereological framework. Since Tarski's work is rooted in Lesniewski's mereology, and given that lambda-MM currently provides only a partial implementation of Tarski's geometry, the first part of the paper completes this framework by proving that mereological classes correspond to regular open sets. This yields a topology of individual names that can be extended with Tarski's geometric primitives. Unlike classical approaches in qualitative logical theories, we adopt a solution that derives a full topological space from mereology together with a geometric subspace, thereby increasing the expressiveness of the theory. In the second part, we show that Tarski's geometry forms a subspace of this topology in which regions correspond to restricted classes. We also prove three of Tarski's original postulates, reducing his axiomatic system, and extend the theory with the T2 (Hausdorff) property and additional definitions.

---

## 110. FSDAM: Few-Shot Driving Attention Modeling via Vision-Language Coupling

**论文链接:** [http://arxiv.org/abs/2511.12708v1](http://arxiv.org/abs/2511.12708v1)

**作者:** Kaiser Hamid, Can Cui, Khandakar Ashrafi Akbar, Ziran Wang, Nade Liang

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了FSDAM框架，仅需约100个标注样本就能实现驾驶员注意力预测和字幕生成的联合学习，比现有方法少两个数量级。

### 背景

理解驾驶员视线位置和注意力转移对自主系统解读人类意图和证明行为至关重要，但现有模型依赖大规模视线数据集，收集和整理工作量大且耗时。

### 目的

开发一个仅需少量标注样本的驾驶员注意力建模框架，实现注意力预测和字幕生成的联合学习。

### 方法

FSDAM采用双路径架构，单独模块处理空间预测和字幕生成，通过跨模态对齐保持语义一致性，实现有限监督下的有效学习。

### 主要发现

FSDAM在注意力预测方面具有竞争力，能生成连贯且具有上下文感知能力的解释，并在多个驾驶基准测试中表现出强大的零样本泛化能力。

### 结论

有限监督下也能实现有效的注意力条件生成，为数据受限场景下部署可解释的驾驶员注意力系统开辟了新可能性。

### 翻译

理解驾驶员的视线位置和注意力转移原因对能够解读人类意图并证明其行为的自主系统至关重要。大多数现有模型依赖大规模视线数据集来学习这些模式；然而，此类数据集的收集和整理工作量大且耗时。我们提出了FSDAM（少样本驾驶员注意力建模），一个仅需约100个标注样本就能实现联合注意力预测和字幕生成的框架，比现有方法少两个数量级。我们的方法引入了双路径架构，其中独立模块处理空间预测和字幕生成，同时通过跨模态对齐保持语义一致性。尽管监督有限，FSDAM在注意力预测方面具有竞争力，并能生成连贯且具有上下文感知能力的解释。该模型在多个驾驶基准测试中表现出强大的零样本泛化能力。这项工作表明，有限监督下也能实现有效的注意力条件生成，为数据受限场景下可解释驾驶员注意力系统的实际部署开辟了新可能性。


### 论文摘要

Understanding where drivers look and why they shift their attention is essential for autonomous systems that read human intent and justify their actions. Most existing models rely on large-scale gaze datasets to learn these patterns; however, such datasets are labor-intensive to collect and time-consuming to curate. We present FSDAM (Few-Shot Driver Attention Modeling), a framework that achieves joint attention prediction and caption generation with approximately 100 annotated examples, two orders of magnitude fewer than existing approaches. Our approach introduces a dual-pathway architecture where separate modules handle spatial prediction and caption generation while maintaining semantic consistency through cross-modal alignment. Despite minimal supervision, FSDAM achieves competitive performance on attention prediction, generates coherent, and context-aware explanations. The model demonstrates robust zero-shot generalization across multiple driving benchmarks. This work shows that effective attention-conditioned generation is achievable with limited supervision, opening new possibilities for practical deployment of explainable driver attention systems in data-constrained scenarios.

---

## 111. X-VMamba: Explainable Vision Mamba

**论文链接:** [http://arxiv.org/abs/2511.12694v1](http://arxiv.org/abs/2511.12694v1)

**作者:** Mohamed A. Mabrok, Yalda Zafari

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种基于可控性的可解释性框架，用于分析状态空间模型(SSMs)特别是Mamba架构如何处理空间信息。该框架通过量化输入序列不同部分对SSMs内部状态动态的影响，提供了对SSMs工作机制的深入理解。

### 背景

状态空间模型(SSMs)，特别是Mamba架构，已成为序列建模的强大替代方案，具有线性计算复杂性和竞争性能。然而，尽管这些视觉SSMs有效，理解它们如何处理空间信息仍然具有挑战性，因为缺乏透明的、类似注意力的机制。

### 目的

引入一个可控性基础的可解释性框架，量化输入序列的不同部分如何影响SSMs的内部状态动态，从而填补对SSMs空间信息处理理解上的空白。

### 方法

提出两种互补的表述方式：1)基于雅可比的方法，适用于任何SSM架构，通过完整的状态传播链测量影响；2)基于格拉姆的方法，适用于对角SSMs，通过闭式解析解实现更快的速度。两种方法都只需单次前向传播，具有线性复杂度，无需架构修改或超参数调整。

### 主要发现

通过在三种不同的医学成像模态上的实验验证，发现SSMs自然实现了分层特征细化，从早期层的弥散低级纹理到更深层的聚焦、临床有意义的模式。分析揭示了与诊断标准一致的领域特定可控性特征、网络层次结构中的渐进式空间选择性以及扫描策略对注意力模式的显著影响。

### 结论

该框架为所有领域的SSMs建立了可控性分析作为统一的、基础性的可解释性范式，不仅限于医学成像，还可应用于计算机视觉、自然语言处理和跨领域任务。

### 翻译

状态空间模型(SSMs)，特别是Mamba架构，最近已成为序列建模的强大替代方案，提供线性计算复杂性的同时实现了竞争性能。然而，尽管这些视觉SSMs有效，理解它们如何处理空间信息仍然具有挑战性，因为缺乏透明的、类似注意力的机制。为了填补这一空白，我们引入了一个基于可控性的可解释性框架，量化输入序列的不同部分（标记或补丁）如何影响SSMs的内部状态动态。我们提出两种互补的表述方式：一种适用于任何SSM架构的基于雅可比的方法，通过完整的状态传播链测量影响；以及对角SSMs的基于格拉姆的方法，通过闭式解析解实现更快的速度。两种方法都只需单次前向传播，具有线性复杂度，无需架构修改或超参数调整。我们在三种不同的医学成像模态上验证了我们的框架，证明SSMs自然实现了分层特征细化，从早期层的弥散低级纹理到更深层的聚焦、临床有意义的模式。我们的分析揭示了与诊断标准一致的领域特定可控性特征、网络层次结构中的渐进式空间选择性以及扫描策略对注意力模式的显著影响。除了医学成像，我们还阐述了跨越计算机视觉、自然语言处理和跨领域任务的应用。我们的框架将可控性分析确立为所有领域SSMs的统一、基础性可解释性范式。代码和分析工具将在发表后提供。


### 论文摘要

State Space Models (SSMs), particularly the Mamba architecture, have recently emerged as powerful alternatives to Transformers for sequence modeling, offering linear computational complexity while achieving competitive performance. Yet, despite their effectiveness, understanding how these Vision SSMs process spatial information remains challenging due to the lack of transparent, attention-like mechanisms. To address this gap, we introduce a controllability-based interpretability framework that quantifies how different parts of the input sequence (tokens or patches) influence the internal state dynamics of SSMs. We propose two complementary formulations: a Jacobian-based method applicable to any SSM architecture that measures influence through the full chain of state propagation, and a Gramian-based approach for diagonal SSMs that achieves superior speed through closed-form analytical solutions. Both methods operate in a single forward pass with linear complexity, requiring no architectural modifications or hyperparameter tuning. We validate our framework through experiments on three diverse medical imaging modalities, demonstrating that SSMs naturally implement hierarchical feature refinement from diffuse low-level textures in early layers to focused, clinically meaningful patterns in deeper layers. Our analysis reveals domain-specific controllability signatures aligned with diagnostic criteria, progressive spatial selectivity across the network hierarchy, and the substantial influence of scanning strategies on attention patterns. Beyond medical imaging, we articulate applications spanning computer vision, natural language processing, and cross-domain tasks. Our framework establishes controllability analysis as a unified, foundational interpretability paradigm for SSMs across all domains. Code and analysis tools will be made available upon publication

---

## 112. BridgeEQA: Virtual Embodied Agents for Real Bridge Inspections

**论文链接:** [http://arxiv.org/abs/2511.12676v1](http://arxiv.org/abs/2511.12676v1)

**作者:** Subin Varghese, Joshua Gao, Asad Ur Rahman, Vedhus Hoskere

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了基础设施检查作为开放词汇具身问答(EQA)的领域，并创建了BridgeEQA基准测试，同时提出了EMVR方法来解决现有模型在情景记忆EQA设置下的性能问题。

### 背景

在真实世界环境中部署能够回答周围环境问题的智能体仍然困难，部分原因是缺乏能够准确捕捉实际操作条件的基准测试。

### 目的

提出基础设施检查作为一个具有挑战性的EQA领域，并创建一个能够评估多尺度推理、长距离空间理解和复杂语义关系能力的基准测试。

### 方法

引入了BridgeEQA基准测试，包含2200个开放词汇问答对，基于200个真实世界桥梁场景的专业检查报告；提出新的EQA指标'Image Citation Relevance'；开发了Embodied Memory Visual Reasoning (EMVR)方法，将检查制定为基于图像场景图的顺序导航过程。

### 主要发现

最先进的视觉-语言模型在情景记忆EQA设置下存在显著的性能差距；提出的EMVR方法在基线上表现出强大的性能。

### 结论

通过公开发布数据集和代码，为具身智能研究提供了新的基准和解决方案，特别是在基础设施检查领域。

### 翻译

在真实世界环境中部署能够回答周围环境问题的具身智能体仍然困难，部分原因是缺乏能够准确捕捉实际操作条件的基准测试。我们提出基础设施检查作为一个引人入胜的开放词汇具身问答(EQA)领域：它自然需要多尺度推理、长距离空间理解和复杂语义关系，同时通过标准化的国家桥梁库存(NBI)条件评级(0-9)、专业检查报告和第一人称视角图像提供独特的评估优势。我们引入了BridgeEQA，一个包含2200个开放词汇问答对(以OpenEQA风格)的基准测试，基于200个真实世界桥梁场景的专业检查报告，每个场景平均有47.93张图像。问题需要综合多张图像的视觉证据，并将回答与NBI条件评级保持一致。我们进一步提出了一个新的EQA指标'Image Citation Relevance'来评估模型引用相关图像的能力。对最先进的视觉-语言模型的评估显示在情景记忆EQA设置下存在显著的性能差距。为解决这一问题，我们提出了具身记忆视觉推理(EMVR)，它将检查制定为基于图像场景图的顺序导航：图像是节点，智能体采取行动遍历视图、比较证据并在马尔可夫决策过程中进行推理。EMVR在基线上表现出强大的性能。我们公开发布了数据集和代码。


### 论文摘要

Deploying embodied agents that can answer questions about their surroundings in realistic real-world settings remains difficult, partly due to the scarcity of benchmarks that faithfully capture practical operating conditions. We propose infrastructure inspection as a compelling domain for open-vocabulary Embodied Question Answering (EQA): it naturally demands multi-scale reasoning, long-range spatial understanding, and complex semantic relationships, while offering unique evaluation advantages via standardized National Bridge Inventory (NBI) condition ratings (0-9), professional inspection reports, and egocentric imagery.   We introduce BridgeEQA, a benchmark of 2,200 open-vocabulary question-answer pairs (in the style of OpenEQA) grounded in professional inspection reports across 200 real-world bridge scenes with 47.93 images on average per scene. Questions require synthesizing visual evidence across multiple images and aligning responses with NBI condition ratings. We further propose a new EQA metric Image Citation Relevance to evaluate the ability of a model to cite relevant images.   Evaluations of state-of-the-art vision-language models reveal substantial performance gaps under episodic memory EQA settings. To address this, we propose Embodied Memory Visual Reasoning (EMVR), which formulates inspection as sequential navigation over an image-based scene graph: images are nodes, and an agent takes actions to traverse views, compare evidence, and reason within a Markov decision process. EMVR shows strong performance over the baselines. We publicly release both the dataset and code.

---

## 113. DPVO-QAT++: Heterogeneous QAT and CUDA Kernel Fusion for High-Performance Deep Patch Visual Odometry

**论文链接:** [http://arxiv.org/abs/2511.12653v1](http://arxiv.org/abs/2511.12653v1)

**作者:** Cheng Liao

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了DPVO-QAT++分层量化优化框架，通过结合可学习尺度参数化、视觉里程计前后端异构精度设计和GPU原生CUDA内核融合，显著提高处理速度并降低内存占用，同时保持原始模型轨迹精度。

### 背景

基于深度学习的视觉SLAM系统具有出色几何推理能力，但其计算开销极大，限制了在资源受限的自主平台上的部署。

### 目的

开发一个高效优化框架，使高精度深度视觉里程计能够在实际嵌入式平台上部署，解决精度与效率之间的平衡问题。

### 方法

DPVO-QAT++框架包括：可学习的尺度参数化、视觉里程计前端(使用FP16/FP32浮点假量化)和后端(全精度)的异构精度设计，以及针对假量化的GPU原生内核融合(自定义CUDA内核)。

### 主要发现

在TartanAir数据集上：平均FPS增加52.1%，中值延迟降低29.1%，峰值GPU内存占用减少64.9%，保持相当轨迹精度；在EuRoC数据集上：平均FPS增加30.1%，中值延迟降低23.1%，峰值GPU内存占用减少37.7%，保持相当轨迹精度。

### 结论

DPVO-QAT++有效弥合了高精度深度视觉里程计与实际部署效率要求之间的差距，为该技术在真实嵌入式平台上的应用提供了可行的工程范式。

### 翻译

Visual Odometry: 视觉里程计；Heterogeneous Precision Architecture: 异构精度架构；Quantization-Aware Training: 量化感知训练；CUDA Kernel Fusion: CUDA内核融合；Scale-Only Training: 仅尺度训练；Deep Patch Visual Odometry: 深度块视觉里程计；GPU-Native Kernel Fusion: GPU原生内核融合

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决深度学习视觉里程计系统计算开销大的问题，使其难以在资源受限的嵌入式平台上实时运行。这个问题在现实中非常重要，因为自动驾驶汽车、无人机和增强现实设备等自主平台需要高效的视觉定位技术，而现有深度学习方法虽然精度高，但计算需求大，限制了它们在实际应用中的部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过算法优化和硬件加速的协同设计来解决部署差距问题。他们借鉴了现有的量化感知训练(QAT)和CUDA优化技术，但针对视觉里程计的特殊需求进行了改进。作者分析了传统方法与深度学习方法的优缺点，认识到需要平衡精度和效率，因此设计了专门的几何感知策略和异构精度架构，而不是简单地应用通用技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是采用分层异构精度架构：对计算密集的前端网络应用伪量化（浮点模拟），而对数值敏感的后端几何优化器保持全精度。整体流程分为离线训练和在线推理两个阶段：训练阶段使用教师-学生蒸馏方法，只学习量化尺度参数；推理阶段使用自定义CUDA内核执行融合的伪量化操作，然后输入全精度后端进行几何优化，形成'量化前端+全精度后端'的异构拓扑结构。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)几何感知的'仅尺度'QAT策略，学习量化尺度同时保持模型权重不变；2)异构精度架构，前端使用浮点伪量化，后端保持全精度；3)自定义CUDA内核融合伪量化操作，消除框架调度开销。相比之前的工作，这种方法结合了算法优化和系统级优化，特别针对视觉里程计的几何推理需求进行了定制，而不是简单地应用通用量化或加速技术。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DPVO-QAT++通过结合异构量化感知训练和CUDA内核融合，在保持高精度的同时显著提升了深度视觉里程计系统的计算效率和内存利用率，为在资源受限的嵌入式平台上部署高性能视觉定位系统提供了可行的工程范例。'}


### 论文摘要

Deep learning-based Visual SLAM (vSLAM) systems exhibit exceptional geometric reasoning capabilities, yet their prohibitive computational overhead severely restricts deployment on resource-constrained autonomous platforms. This paper presents a hierarchical quantization optimization framework, DPVO-QAT++ (DPVO-QAT++: Heterogeneous QAT and CUDA Kernel Fusion for High-Performance Deep Patch Visual Odometry). Through the synergistic integration of learnable scale parameterization, a heterogeneous precision design for the Visual Odometry (VO) front-end and back-end (front-end floating-point fake quantization with FP16/FP32; back-end full precision), and GPU-native kernel fusion for fake quantization (custom CUDA kernels), our framework significantly reduces memory footprint and increases processing speed while preserving the trajectory accuracy of the original model. On the TartanAir dataset, our framework achieves an average FPS increase of 52.1%, a 29.1% reduction in median latency, and a 64.9% reduction in peak GPU memory reservation, while maintaining trajectory accuracy (ATE) comparable to the original DPVO model across 32 validation sequences. On the EuRoC dataset, it realizes an average FPS increase of 30.1%, a 23.1% reduction in median latency, and a 37.7% reduction in peak GPU memory reservation, maintaining comparable trajectory accuracy (ATE) across 11 validation sequences. Experimental results demonstrate that DPVO-QAT++ effectively bridges the gap between high-precision deep VO and the efficiency requirements for practical deployment, offering a viable engineering paradigm for the application of this technology on real-world embedded platforms.   Keywords: Visual Odometry, Heterogeneous Precision Architecture, Quantization-Aware Training, CUDA Kernel Fusion, Scale-Only Training, Deep Patch Visual Odometry, GPU-Native Kernel Fusion.

---

## 114. Pixels or Positions? Benchmarking Modalities in Group Activity Recognition

**论文链接:** [http://arxiv.org/abs/2511.12606v1](http://arxiv.org/abs/2511.12606v1)

**作者:** Drishya Karki, Merey Ramazanova, Anthony Cioppa, Silvio Giancola, Bernard Ghanem

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文介绍了SoccerNet-GAR多模态数据集，基于2022年足球世界杯64场比赛构建，包含94,285个群体活动的同步广播视频和玩家跟踪数据。研究比较了视频模态和跟踪模态在群体活动识别中的表现，发现基于跟踪的方法表现更优且训练效率更高。

### 背景

群体活动识别在视频模态上已有较多研究，特别是在监控和室内团队运动领域。然而，其他模态如代理位置和轨迹数据(跟踪)相对研究不足，尽管这些数据能明确编码空间交互。目前缺乏标准化的基准数据集来对齐广播视频和跟踪数据，导致无法对这两种模态进行公平比较。

### 目的

理解像素(视频)或位置(跟踪)模态哪种更适合群体活动识别，创建多模态数据集，并建立统一的评估协议，以比较这两种模态在群体活动识别中的表现。

### 方法

创建SoccerNet-GAR多模态数据集，包含94,285个群体活动的同步广播视频和玩家跟踪模态，标注了10个类别。定义统一的评估协议，测试基于视频的分类器和利用图神经网络的基于跟踪的分类器。提出新颖的基于角色的图架构，通过位置边和时序注意力直接编码战术结构。

### 主要发现

基于跟踪的模型实现了67.2%的平衡准确率，而最佳视频基线为58.1%。同时，基于跟踪的训练速度快4.25倍，参数少438倍(197K vs 86.3M)。

### 结论

这项研究为理解像素和位置在群体活动识别中的相对优势提供了新的见解。总体上，它强调了模态选择和基于角色的建模对于群体活动识别的重要性。

### 翻译

群体活动识别在视频模态上已有较多研究，特别是在监控和室内团队运动领域。然而，其他模态如代理位置和轨迹随时间变化的数据(即跟踪)相对研究不足，尽管这些数据是紧凑的、以代理为中心的信号，明确编码了空间交互。理解像素(视频)或位置(跟踪)模态哪种更适合群体活动识别，对于推动该领域进一步研究很重要。然而，目前缺乏标准化的基准数据集来对齐广播视频和跟踪数据，导致无法对这两种模态进行公平比较。在本工作中，我们引入了SoccerNet-GAR，这是一个多模态数据集，基于2022年足球世界杯64场比赛构建。具体而言，94,285个群体活动的广播视频和玩家跟踪模态被同步并标注了10个类别。此外，我们定义了一个统一的评估协议，用于基准测试两种强大的单模态方法：(i)竞争性的基于视频的分类器，和(ii)利用图神经网络的基于跟踪的分类器。特别是，我们提出的用于基于跟踪的群体活动识别的新颖的基于角色的图架构通过位置边和时序注意力直接编码战术结构。我们的跟踪模型实现了67.2%的平衡准确率，而最佳视频基线为58.1%，同时训练速度快4.25倍，参数少438倍(197K vs 86.3M)。这项研究为理解像素和位置在群体活动识别中的相对优势提供了新的见解。总体上，它强调了模态选择和基于角色的建模对于群体活动识别的重要性。


### 论文摘要

Group Activity Recognition (GAR) is well studied on the video modality for surveillance and indoor team sports (e.g., volleyball, basketball). Yet, other modalities such as agent positions and trajectories over time, i.e. tracking, remain comparatively under-explored despite being compact, agent-centric signals that explicitly encode spatial interactions. Understanding whether pixel (video) or position (tracking) modalities leads to better group activity recognition is therefore important to drive further research on the topic. However, no standardized benchmark currently exists that aligns broadcast video and tracking data for the same group activities, leading to a lack of apples-to-apples comparison between these modalities for GAR. In this work, we introduce SoccerNet-GAR, a multimodal dataset built from the $64$ matches of the football World Cup 2022. Specifically, the broadcast videos and player tracking modalities for $94{,}285$ group activities are synchronized and annotated with $10$ categories. Furthermore, we define a unified evaluation protocol to benchmark two strong unimodal approaches: (i) a competitive video-based classifiers and (ii) a tracking-based classifiers leveraging graph neural networks. In particular, our novel role-aware graph architecture for tracking-based GAR directly encodes tactical structure through positional edges and temporal attention. Our tracking model achieves $67.2\%$ balanced accuracy compared to $58.1\%$ for the best video baseline, while training $4.25 \times$ faster with $438 \times$ fewer parameters ($197K$ \vs $86.3M$). This study provides new insights into the relative strengths of pixels and positions for group activity recognition. Overall, it highlights the importance of modality choice and role-aware modeling for GAR.

---

## 115. ClutterNav: Gradient-Guided Search for Efficient 3D Clutter Removal with Learned Costmaps

**论文链接:** [http://arxiv.org/abs/2511.12479v1](http://arxiv.org/abs/2511.12479v1)

**作者:** Navin Sriram Ravie, Keerthi Vasan M, Bijo Sebastian

**发布时间:** 2025-11-16

### GPT解析

### 总结

ClutterNav是一种新型决策框架，用于在密集杂乱环境中识别最佳物体移除顺序，以访问目标物体同时最小化堆叠扰动，解决了基于规则规划器的高计算开销和端到端强化学习方法的可解释性问题。

### 背景

密集杂乱环境中的目标物体检索具有挑战性，特别是当目标物体嵌入在密集配置中时。需要预见性以最小化对杂乱配置的整体变化，同时避免堆叠不稳定和减少需要移除的物体数量。

### 目的

提出ClutterNav框架，能够识别下一个要移除的最佳物体，以访问给定杂乱环境中的目标物体，同时最小化堆叠扰动。

### 方法

将问题表述为连续强化学习任务，使用从演示中训练的可移除性评论家估计移除成本，并通过集成梯度评估周围物体对目标可访问性的影响，动态平衡即时可移除性与长期目标暴露度。

### 主要发现

在仿真和真实世界实验中验证了该方法，实现了在部分可观察环境中实时、感知遮挡的决策能力，达到接近人类的战略排序。

### 结论

ClutterNav实现了接近人类的战略排序，无需预定义的启发式方法，解决了密集杂乱环境中目标物体检索的问题。

### 翻译

密集杂乱环境中的目标物体检索是一个具有挑战性的问题，特别是当目标物体嵌入在密集配置中时。它需要预见性，以最小化对杂乱配置的整体变化，同时访问目标物体，避免堆叠不稳定并减少需要移除的物体数量。当应用于此问题时，基于规则的规划器依赖于僵化的启发式方法，导致高计算开销。端到端的强化学习方法在可解释性和不同条件下的泛化能力方面存在问题。为解决这些问题，我们提出了ClutterNav，一种新型决策框架，可以识别要移除的下一个最佳物体，以便访问给定杂乱环境中的目标物体，同时最小化堆叠扰动。ClutterNav将问题表述为连续强化学习任务，每次物体移除都会动态更新对场景的理解。一个从演示中训练的可移除性评论家，基于几何和空间特征估计移除任何给定物体的成本。这种学习到的成本通过集成梯度得到补充，评估周围物体的存在或移除如何影响目标的可访问性。通过动态平衡即时可移除性与长期目标暴露度的动作优先级，ClutterNav实现了接近人类的战略排序，无需预定义的启发式方法。所提出的方法在仿真和真实世界实验中得到了广泛验证。结果表明在部分可观察环境中实现了实时、感知遮挡的决策能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人在密集杂乱环境中高效移除物体以获取目标物体的问题。当目标物体被深埋在其他物体中时，机器人需要智能地决定移除哪些物体才能最小化对整体堆叠的干扰，同时高效地暴露目标。这个问题在工业自动化、制药包装、电子商务履行和灾难响应等多个领域都很重要，因为它直接影响机器人操作的效率和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：基于规则的规划器在处理密集杂乱时计算开销大，端到端强化学习方法缺乏可解释性和泛化能力。他们借鉴了强化学习中的Soft Actor-Critic框架和演示学习技术，以及可解释AI中的集成梯度方法。作者创新地将问题表述为连续的强化学习任务，设计了一个轻量级决策框架，结合学习的可移除性成本和实时梯度引导规划，以实现高效且稳定的物体移除。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过学习物体的可移除性成本，并利用梯度分析指导移除决策，以最小化对杂乱配置的干扰，同时高效暴露目标物体。整体流程包括：1)从点云中提取物体几何和空间特征；2)使用训练好的评估器预测移除每个物体的成本；3)计算集成梯度评估移除周围物体对目标可达性的影响；4)结合成本和梯度计算综合重要性分数；5)选择最佳移除对象并执行；6)更新场景理解并重复直到目标可访问。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)轻量级决策框架，结合学习成本和梯度引导规划；2)目标不可知的可移除性成本估计器，无需明确目标条件；3)集成梯度方法实现可解释的决策序列；4)实时、遮挡感知的决策制定。相比之前工作，它不依赖僵化规则，能处理复杂无序环境；比层次任务-动作规划更高效且能处理3D场景；比端到端强化学习更可解释且泛化能力更强；比保守策略显著减少移除步骤同时保持低干扰。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ClutterNav通过结合学习的成本估计和梯度引导决策，实现了在密集杂乱环境中高效、稳定地移除物体以获取目标，比现有方法更少干扰堆叠配置同时减少移除步骤。'}


### 论文摘要

Dense clutter removal for target object retrieval presents a challenging problem, especially when targets are embedded deep within densely-packed configurations. It requires foresight to minimize overall changes to the clutter configuration while accessing target objects, avoiding stack destabilization and reducing the number of object removals required. Rule-based planners when applied to this problem, rely on rigid heuristics, leading to high computational overhead. End-to-end reinforcement learning approaches struggle with interpretability and generalizability over different conditions. To address these issues, we present ClutterNav, a novel decision-making framework that can identify the next best object to be removed so as to access a target object in a given clutter, while minimising stack disturbances. ClutterNav formulates the problem as a continuous reinforcement learning task, where each object removal dynamically updates the understanding of the scene. A removability critic, trained from demonstrations, estimates the cost of removing any given object based on geometric and spatial features. This learned cost is complemented by integrated gradients that assess how the presence or removal of surrounding objects influences the accessibility of the target. By dynamically prioritizing actions that balance immediate removability against long-term target exposure, ClutterNav achieves near human-like strategic sequencing, without predefined heuristics. The proposed approach is validated extensively in simulation and over real-world experiments. The results demonstrate real-time, occlusion-aware decision-making in partially observable environments.

---

## 116. RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robotic Manipulation and Navigation

**论文链接:** [http://arxiv.org/abs/2511.12436v1](http://arxiv.org/abs/2511.12436v1)

**作者:** Xiaoshuai Hao, Yingbo Tang, Lingfeng Zhang, Yanbiao Ma, Yunfeng Diao, Ziyu Jia, Wenbo Ding, Hangjun Ye, Long Chen

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了RoboAfford++数据集和RoboAfford-Eval基准，用于解决视觉语言模型在机器人操作和导航中可供性学习的不足问题。

### 背景

机器人操作和导航是具身智能的基本能力，需要理解环境中的物体识别、物体可供性和空间可供性。然而，现有的视觉语言模型虽然擅长高层任务规划和场景理解，但难以推断物理交互的可操作位置。

### 目的

开发一个增强型数据集和评估基准，提高机器人对物体和空间可供性的理解能力，使机器人能够更有效地与物理世界交互。

### 方法

创建RoboAfford++数据集，包含869,987张图像和200万个问答标注，涵盖三个关键任务：物体可供性识别、物体可供性预测和空间可供性定位。同时提出RoboAfford-Eval评估基准，包含338个精心标注的样本。

### 主要发现

现有视觉语言模型在可供性学习方面存在明显不足，但在RoboAfford++数据集上进行微调后，模型对物体和空间可供性的推理能力得到显著提升。

### 结论

RoboAfford++数据集和RoboAfford-Eval基准有效解决了视觉语言模型在可供性学习方面的局限性，为机器人操作和导航提供了更好的环境理解能力。

### 翻译

机器人操作和导航是具身智能的基本能力，使机器人能够有效地与物理世界交互。实现这些能力需要对环境有整体理解，包括物体识别以定位目标物体、物体可供性以识别潜在交互区域，以及空间可供性以确定物体放置和机器人移动的最佳区域。虽然视觉语言模型擅长高层任务规划和场景理解，但它们往往难以推断物理交互的可操作位置，如功能性抓取点和允许放置区域。这一限制源于其训练数据集中缺乏物体和空间可供性的细粒度标注。为解决这一挑战，我们引入了RoboAfford++，这是一个用于机器人操作和导航的多模态可供性学习的生成式AI增强数据集。我们的数据集包含869,987张图像和200万个问答标注，涵盖三个关键任务：物体可供性识别（基于属性和空间关系识别目标物体）、物体可供性预测（精确定位操作的功能部件）和空间可供性定位（识别物体放置和机器人导航的自由空间）。作为补充，我们提出了RoboAfford-Eval，一个用于评估真实场景中可供性感知预测的综合基准，包含338个在同一三个任务中精心标注的样本。大量实验结果揭示了现有视觉语言模型在可供性学习方面的不足，而在RoboAfford++数据集上进行微调显著增强了它们对物体和空间可供性的推理能力，验证了该数据集的有效性。


### 论文摘要

Robotic manipulation and navigation are fundamental capabilities of embodied intelligence, enabling effective robot interactions with the physical world. Achieving these capabilities requires a cohesive understanding of the environment, including object recognition to localize target objects, object affordances to identify potential interaction areas and spatial affordances to discern optimal areas for both object placement and robot movement. While Vision-Language Models (VLMs) excel at high-level task planning and scene understanding, they often struggle to infer actionable positions for physical interaction, such as functional grasping points and permissible placement regions. This limitation stems from the lack of fine-grained annotations for object and spatial affordances in their training datasets. To tackle this challenge, we introduce RoboAfford++, a generative AI-enhanced dataset for multimodal affordance learning for both robotic manipulation and navigation. Our dataset comprises 869,987 images paired with 2.0 million question answering (QA) annotations, covering three critical tasks: object affordance recognition to identify target objects based on attributes and spatial relationships, object affordance prediction to pinpoint functional parts for manipulation, and spatial affordance localization to identify free space for object placement and robot navigation. Complementing this dataset, we propose RoboAfford-Eval, a comprehensive benchmark for assessing affordance-aware prediction in real-world scenarios, featuring 338 meticulously annotated samples across the same three tasks. Extensive experimental results reveal the deficiencies of existing VLMs in affordance learning, while fine-tuning on the RoboAfford++ dataset significantly enhances their ability to reason about object and spatial affordances, validating the dataset's effectiveness.

---

## 117. SpaceVLM: Sub-Space Modeling of Negation in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.12331v1](http://arxiv.org/abs/2511.12331v1)

**作者:** Sepehr Kazemi Ranjbar, Kumail Alhamoud, Marzyeh Ghassemi

**发布时间:** 2025-11-15

### GPT解析

### 总结

该论文提出了一种无需重新训练的框架，解决视觉语言模型在处理否定表达时的困难，将否定建模为联合嵌入空间中的子空间而非单个点，显著提高了模型对否定表达的理解能力。

### 背景

视觉语言模型(VLMs)在处理否定表达时存在困难，例如面对'检索一个没有行人的街道场景'这样的提示时，模型往往无法正确理解'没有'这一否定词。现有方法通过在大型否定数据集上进行微调来解决这一问题，但这种微调通常会损害模型在肯定提示上的零样本性能。

### 目的

提出一种不需要重新训练的框架，提高VLMs对否定表达的理解能力，同时保持模型在肯定提示上的零样本性能。

### 方法

作者发现VLMs的嵌入空间可以被划分为语义一致的子空间。基于这一特性，他们提出将否定建模为联合嵌入空间中的子空间。对于匹配'A但不是N'这样的标题，在A和N的嵌入周围构建两个球形帽，并通过接近A且远离N的区域中心方向对图像进行评分。

### 主要发现

在检索、多选题选择和文本到图像生成等任务中，该方法相比先前方法平均提高了约30%的否定理解能力。它缩小了肯定提示和否定提示之间的性能差距，同时保持了微调模型无法维持的零样本性能。

### 结论

通过将否定建模为嵌入空间中的子空间而非单个点，该方法有效地解决了VLMs在处理否定表达时的问题，无需重新训练即可显著提高性能，同时保持模型在肯定提示上的零样本能力。

### 翻译

视觉语言模型(VLMs)在处理否定表达时存在困难。面对'检索(或生成)一个没有行人的街道场景'这样的提示时，它们往往无法尊重'没有'这一否定词。现有方法通过在大型否定数据集上进行微调来解决这一限制，但这种重新训练通常会损害模型在肯定提示上的零样本性能。我们证明VLMs(如CLIP)的嵌入空间可以被划分为语义一致的子空间。基于这一特性，我们提出了一种无需训练的框架，将否定建模为联合嵌入空间中的子空间而非单个点(图1)。为了找到匹配'A但不是N'这样的标题的图像，我们在A和N的嵌入周围构建两个球形帽，并通过接近A且远离N的区域中心方向对图像进行评分。在检索、多选题选择和文本到图像生成等任务中，我们的方法相比先前方法平均提高了约30%的否定理解能力。它缩小了肯定提示和否定提示之间的性能差距，同时保持了微调模型无法维持的零样本性能。代码将在发表后公开。


### 论文摘要

Vision-Language Models (VLMs) struggle with negation. Given a prompt like "retrieve (or generate) a street scene without pedestrians," they often fail to respect the "not." Existing methods address this limitation by fine-tuning on large negation datasets, but such retraining often compromises the model's zero-shot performance on affirmative prompts. We show that the embedding space of VLMs, such as CLIP, can be divided into semantically consistent subspaces. Based on this property, we propose a training-free framework that models negation as a subspace in the joint embedding space rather than a single point (Figure 1). To find the matching image for a caption such as "A but not N," we construct two spherical caps around the embeddings of A and N, and we score images by the central direction of the region that is close to A and far from N. Across retrieval, MCQ, and text-to-image tasks, our method improves negation understanding by about 30% on average over prior methods. It closes the gap between affirmative and negated prompts while preserving the zero-shot performance that fine-tuned models fail to maintain. Code will be released upon publication.

---

## 118. CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.12263v1](http://arxiv.org/abs/2511.12263v1)

**作者:** Jingyao Li, Jingyun Wang, Molin Tan, Haochen Wang, Cilin Yan, Likun Shi, Jiayin Cai, Xiaolong Jiang, Yao Hu

**发布时间:** 2025-11-15

**备注:** 30 pages, 28 figures

### GPT解析

### 总结

该论文介绍了CrossVid，第一个全面评估多模态大语言模型在跨视频上下文中时空推理能力的基准测试。CrossVid包含广泛的分层任务，涵盖5,331个视频和9,015个问答对，实验显示当前MLLMs在跨视频推理任务上表现不佳，平均准确率仅为50.4%，主要原因是它们无法整合或比较分布在多个视频中的证据进行推理。

### 背景

跨视频推理(CVR)是视频理解中的一个重大挑战，需要同时理解多个视频以聚合和比较视频组之间的信息。现有的视频理解基准大多专注于单视频分析，无法评估MLLMs同时推理多个视频的能力。最近的基准测试评估了MLLMs对同一场景不同视角的多视角视频的能力，但其有限的任务无法全面评估MLLMs在多样化现实世界CVR场景中的表现。

### 目的

为了全面评估MLLMs在跨视频上下文中的时空推理能力，研究者引入了CrossVid，这是第一个为此目的设计的基准测试。

### 方法

CrossVid包含广泛的分层任务，由四个高级维度和十个具体任务组成，反映了现实世界视频理解的复杂性和多样性。CrossVid提供了5,331个视频和9,015个具有挑战性的问答对，涵盖单选、多选和开放式问题格式。研究者在各种开源和闭源MLLMs上进行了大量实验。

### 主要发现

实验表明，Gemini-2.5-Pro在CrossVid上表现最佳，平均准确率达到50.4%。深入的案例研究表明，大多数当前的MLLMs难以完成CVR任务，主要原因是它们无法整合或比较分布在多个视频中的证据进行推理。

### 结论

CrossVid有潜力指导未来增强MLLMs的CVR能力的发展。

### 翻译

跨视频推理(CVR)在视频理解中提出了重大挑战，需要同时理解多个视频以聚合和比较视频组之间的信息。大多数现有的视频理解基准专注于单视频分析，无法评估多模态大语言模型(MLLMs)同时推理多个视频的能力。最近的基准测试评估了MLLMs对捕捉同一场景不同视角的多视角视频的能力。然而，它们有限的任务无法全面评估MLLMs在多样化的现实世界CVR场景中的表现。为此，我们引入了CrossVid，这是第一个全面评估MLLMs在跨视频上下文中时空推理能力的基准测试。首先，CrossVid包含广泛的分层任务谱系，由四个高级维度和十个具体任务组成，从而紧密反映了现实世界视频理解的复杂性和多样性。其次，CrossVid提供了5,331个视频以及9,015个具有挑战性的问答对，涵盖单选、多选和开放式问题格式。通过对各种开源和闭源MLLMs进行大量实验，我们观察到Gemini-2.5-Pro在CrossVid上表现最佳，平均准确率为50.4%。值得注意的是，我们的深入案例研究表明，大多数当前的MLLMs难以完成CVR任务，主要原因是它们无法整合或比较分布在多个视频中的证据进行推理。这些见解凸显了CrossVid在指导未来增强MLLMs的CVR能力方面的潜力。


### 论文摘要

Cross-Video Reasoning (CVR) presents a significant challenge in video understanding, which requires simultaneous understanding of multiple videos to aggregate and compare information across groups of videos. Most existing video understanding benchmarks focus on single-video analysis, failing to assess the ability of multimodal large language models (MLLMs) to simultaneously reason over various videos. Recent benchmarks evaluate MLLMs' capabilities on multi-view videos that capture different perspectives of the same scene. However, their limited tasks hinder a thorough assessment of MLLMs in diverse real-world CVR scenarios. To this end, we introduce CrossVid, the first benchmark designed to comprehensively evaluate MLLMs' spatial-temporal reasoning ability in cross-video contexts. Firstly, CrossVid encompasses a wide spectrum of hierarchical tasks, comprising four high-level dimensions and ten specific tasks, thereby closely reflecting the complex and varied nature of real-world video understanding. Secondly, CrossVid provides 5,331 videos, along with 9,015 challenging question-answering pairs, spanning single-choice, multiple-choice, and open-ended question formats. Through extensive experiments on various open-source and closed-source MLLMs, we observe that Gemini-2.5-Pro performs best on CrossVid, achieving an average accuracy of 50.4%. Notably, our in-depth case study demonstrates that most current MLLMs struggle with CVR tasks, primarily due to their inability to integrate or compare evidence distributed across multiple videos for reasoning. These insights highlight the potential of CrossVid to guide future advancements in enhancing MLLMs' CVR capabilities.

---

## 119. SoK: The Last Line of Defense: On Backdoor Defense Evaluation

**论文链接:** [http://arxiv.org/abs/2511.13143v1](http://arxiv.org/abs/2511.13143v1)

**作者:** Gorka Abad, Marina Krček, Stefanos Koffas, Behrad Tajalli, Marco Arazzi, Roberto Riaño, Xiaoyun Xu, Zhuoran Liu, Antonino Nocera, Stjepan Picek

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文对深度学习后门攻击防御进行了系统性分析，揭示了当前评估方法的不一致性问题，并通过大量实验证明了不同评估设置对防御效果评估的影响，最后提出了改进未来防御评估的建议。

### 背景

后门攻击通过在深度学习模型中植入隐藏漏洞，可被恶意输入激活，对模型构成严重威胁。尽管已提出多种防御措施，但评估方法学的异质性使得防御措施间的公平比较变得困难。

### 目的

通过全面的文献回顾和实证评估，对后门防御进行系统性分析，识别评估方法中的不一致性，并提供标准化评估建议。

### 方法

分析了2018年至2025年间在主要AI和安全会议上发表的183篇后门防御论文；进行了大规模实验，涉及三个数据集（MNIST、CIFAR-100、ImageNet-1K）、四种模型架构（ResNet-18、VGG-19、ViT-B/16、DenseNet-121）、16种代表性防御和五种常用攻击，总计超过3000次实验。

### 主要发现

文献中的实验设置、评估指标和威胁模型假设存在显著不一致；防御有效性在不同评估设置下差异很大；当前评估实践存在关键差距，包括计算开销和良性条件下行为报告不足、超参数选择存在偏差、实验不完整等。

### 结论

基于研究结果，作者提供了具体的挑战和有充分依据的建议，以标准化和改进未来的防御评估；这项工作旨在为研究人员和行业从业者提供可操作的见解，用于开发、评估和部署针对不同系统的防御措施。

### 翻译

后门攻击通过植入隐藏漏洞对深度学习模型构成严重威胁，这些漏洞可被恶意输入激活。尽管已提出多种防御措施来缓解这些攻击，但评估方法学的异质性使得防御措施之间的公平比较变得困难。这项工作通过全面的文献回顾和实证评估，对后门防御进行了系统性的（元）分析。我们分析了2018年至2025年间在主要AI和安全会议上发表的183篇后门防御论文，检查了这些防御的特性和评估方法。我们的分析揭示了文献中实验设置、评估指标和威胁模型假设存在显著不一致。通过涉及三个数据集（MNIST、CIFAR-100、ImageNet-1K）、四种模型架构（ResNet-18、VGG-19、ViT-B/16、DenseNet-121）、16种代表性防御和五种常用攻击的广泛实验，总计超过3000次实验，我们证明了防御效果在不同评估设置下差异很大。我们确定了当前评估实践中的关键差距，包括计算开销和良性条件下行为报告不足、超参数选择存在偏差以及实验不完整等。基于我们的发现，我们提供了具体的挑战和有充分依据的建议，以标准化和改进未来的防御评估。我们的工作旨在为研究人员和行业从业者提供可操作的见解，用于开发、评估和部署针对不同系统的防御措施。


### 论文摘要

Backdoor attacks pose a significant threat to deep learning models by implanting hidden vulnerabilities that can be activated by malicious inputs. While numerous defenses have been proposed to mitigate these attacks, the heterogeneous landscape of evaluation methodologies hinders fair comparison between defenses. This work presents a systematic (meta-)analysis of backdoor defenses through a comprehensive literature review and empirical evaluation. We analyzed 183 backdoor defense papers published between 2018 and 2025 across major AI and security venues, examining the properties and evaluation methodologies of these defenses.   Our analysis reveals significant inconsistencies in experimental setups, evaluation metrics, and threat model assumptions in the literature. Through extensive experiments involving three datasets (MNIST, CIFAR-100, ImageNet-1K), four model architectures (ResNet-18, VGG-19, ViT-B/16, DenseNet-121), 16 representative defenses, and five commonly used attacks, totaling over 3\,000 experiments, we demonstrate that defense effectiveness varies substantially across different evaluation setups. We identify critical gaps in current evaluation practices, including insufficient reporting of computational overhead and behavior under benign conditions, bias in hyperparameter selection, and incomplete experimentation. Based on our findings, we provide concrete challenges and well-motivated recommendations to standardize and improve future defense evaluations. Our work aims to equip researchers and industry practitioners with actionable insights for developing, assessing, and deploying defenses to different systems.

---

## 120. Soft Conflict-Resolution Decision Transformer for Offline Multi-Task Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.13133v1](http://arxiv.org/abs/2511.13133v1)

**作者:** Shudong Wang, Xinfei Wang, Chenhao Zhang, Shanchen Pang, Haiyuan Gui, Wenhao Ji, Xiaojian Liao

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为SoCo-DT的软冲突解决方法，基于参数重要性来解决多任务强化学习中的梯度冲突问题。通过动态调整掩码值和引入基于四分位距的动态稀疏调整策略，该方法能够有效保留重要参数并抑制冲突参数，同时实现自适应稀疏演化。

### 背景

多任务强化学习(MTRL)旨在为多样化任务学习统一策略，但常常面临任务间的梯度冲突问题。现有的基于掩码的方法通过分配任务特定的参数掩码来缓解这种冲突，但存在局限性。

### 目的

解决现有掩码方法中粗粒度二进制掩码过度抑制关键冲突参数的问题，以及一刀切的固定稀疏策略对不同任务冲突程度适应性不足的问题，提高模型的泛化能力和学习效率。

### 方法

提出SoCo-DT方法，利用Fisher信息动态调整掩码值，保留重要参数同时抑制冲突参数；引入基于四分位距(IQR)的动态稀疏调整策略，利用训练期间冲突和和谐分数的分布构建任务特定阈值方案；融入非对称余弦退火计划实现训练过程中的自适应稀疏演化。

### 主要发现

现有粗粒度二进制掩码会过度抑制关键冲突参数，阻碍任务间知识共享；不同任务表现出不同程度的冲突，但现有方法使用固定稀疏策略不够适应；SoCo-DT在Meta-World基准测试上表现优异，在MT50上比最先进方法高出7.6%，在次优数据集上高出10.5%。

### 结论

SoCo-DT方法通过动态调整掩码值和引入动态稀疏调整策略，有效缓解了多任务强化学习中的梯度冲突问题，显著提高了整体多任务性能，证明了该方法的有效性。

### 翻译

多任务强化学习(MTRL)寻求为多样化任务学习统一策略，但常常面临任务间的梯度冲突问题。现有的基于掩码的方法通过分配任务特定的参数掩码来缓解此类冲突。然而，我们的经验研究表明，粗粒度二进制掩码存在过度抑制关键冲突参数的问题，阻碍了任务间的知识共享。此外，不同任务表现出不同程度的冲突，但现有方法使用一刀切的固定稀疏策略来保持训练稳定性和性能，这被证明是不够的。这些局限性阻碍了模型的泛化能力和学习效率。为解决这些问题，我们提出了SoCo-DT，一种基于参数重要性的软冲突解决方法。通过利用Fisher信息，动态调整掩码值以保留重要参数同时抑制冲突参数。此外，我们引入了一种基于四分位距(IQR)的动态稀疏调整策略，该策略利用训练期间冲突和和谐分数的分布构建任务特定的阈值方案。为了实现训练过程中的自适应稀疏演化，我们进一步融入了非对称余弦退火计划来持续更新阈值。在Meta-World基准测试上的实验结果表明，SoCo-DT在MT50上比最先进方法高出7.6%，在次优数据集上高出10.5%，证明了其在缓解梯度冲突和提高整体多任务性能方面的有效性。


### 论文摘要

Multi-task reinforcement learning (MTRL) seeks to learn a unified policy for diverse tasks, but often suffers from gradient conflicts across tasks. Existing masking-based methods attempt to mitigate such conflicts by assigning task-specific parameter masks. However, our empirical study shows that coarse-grained binary masks have the problem of over-suppressing key conflicting parameters, hindering knowledge sharing across tasks. Moreover, different tasks exhibit varying conflict levels, yet existing methods use a one-size-fits-all fixed sparsity strategy to keep training stability and performance, which proves inadequate. These limitations hinder the model's generalization and learning efficiency.   To address these issues, we propose SoCo-DT, a Soft Conflict-resolution method based by parameter importance. By leveraging Fisher information, mask values are dynamically adjusted to retain important parameters while suppressing conflicting ones. In addition, we introduce a dynamic sparsity adjustment strategy based on the Interquartile Range (IQR), which constructs task-specific thresholding schemes using the distribution of conflict and harmony scores during training. To enable adaptive sparsity evolution throughout training, we further incorporate an asymmetric cosine annealing schedule to continuously update the threshold. Experimental results on the Meta-World benchmark show that SoCo-DT outperforms the state-of-the-art method by 7.6% on MT50 and by 10.5% on the suboptimal dataset, demonstrating its effectiveness in mitigating gradient conflicts and improving overall multi-task performance.

---

## 121. MergeSlide: Continual Model Merging and Task-to-Class Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images

**论文链接:** [http://arxiv.org/abs/2511.13099v1](http://arxiv.org/abs/2511.13099v1)

**作者:** Doanh C. Bui, Ba Hung Ngo, Hoai Luan Pham, Khang Nguyen, Maï K. Nguyen, Yasuhiko Nakashima

**发布时间:** 2025-11-17

**备注:** WACV2026 Accepted

### GPT解析

### 总结

MergeSlide是一种简单而有效的框架，将终身学习视为模型合并问题，利用视觉语言病理学基础模型处理全切片图像(WSIs)的终身学习任务。

### 背景

全切片图像(WSIs)终身学习旨在癌症相关任务上顺序训练或微调统一模型，考虑到WSIs的千兆字节规模大小，这减少了数据传输和处理所需的资源和精力。

### 目的

开发一个框架，能够有效地在全切片图像上进行终身学习，同时减少资源需求和数据处理负担。

### 方法

MergeSlide框架通过三个步骤处理新任务：1)使用类感知提示定义任务；2)使用无MLP主干网络进行少量周期微调；3)使用正交持续合并策略将模型合并为统一模型，保留性能并缓解灾难性遗忘。对于类增量学习(CLASS-IL)设置下的推理，引入了任务到类提示对齐(TCP)推理方法，首先使用任务级提示识别最相关任务，然后应用相应的类感知提示生成预测。

### 主要发现

在六个TCGA数据集流上的实验表明，MergeSlide优于基于重放的持续学习和视觉语言零样本基线方法。

### 结论

MergeSlide是一个有效的方法，能够处理全切片图像上的终身学习任务，同时保持性能并减少资源需求，代码和数据已在GitHub上公开。

### 翻译

全切片图像(WSIs)上的终身学习旨在癌症相关任务上顺序训练或微调统一模型，减少了数据传输和处理所需的资源和精力，特别是考虑到WSIs的千兆字节规模。在本文中，我们介绍了MergeSlide，这是一个简单而有效的框架，通过利用视觉语言病理学基础模型将终身学习视为模型合并问题。当新任务到达时，它：1)使用类感知提示定义；2)使用无MLP主干网络进行少量周期微调；3)使用正交持续合并策略合并到统一模型中，保留性能并减轻灾难性遗忘。对于类增量学习(CLASS-IL)设置下的推理，其中任务身份未知，我们引入了任务到类提示对齐(TCP)推理。具体来说，TCP首先使用任务级提示识别最相关的任务，然后应用相应的类感知提示生成预测。为了评估MergeSlide，我们在六个TCGA数据集流上进行了实验。结果表明，MergeSlide优于基于重放的持续学习和视觉语言零样本基线。代码和数据可在https://github.com/caodoanh2001/MergeSlide获取。


### 论文摘要

Lifelong learning on Whole Slide Images (WSIs) aims to train or fine-tune a unified model sequentially on cancer-related tasks, reducing the resources and effort required for data transfer and processing, especially given the gigabyte-scale size of WSIs. In this paper, we introduce MergeSlide, a simple yet effective framework that treats lifelong learning as a model merging problem by leveraging a vision-language pathology foundation model. When a new task arrives, it is: 1) defined with class-aware prompts, 2) fine-tuned for a few epochs using an MLP-free backbone, and 3) merged into a unified model using an orthogonal continual merging strategy that preserves performance and mitigates catastrophic forgetting. For inference under the class-incremental learning (CLASS-IL) setting, where task identity is unknown, we introduce Task-to-Class Prompt-aligned (TCP) inference. Specifically, TCP first identifies the most relevant task using task-level prompts and then applies the corresponding class-aware prompts to generate predictions. To evaluate MergeSlide, we conduct experiments on a stream of six TCGA datasets. The results show that MergeSlide outperforms both rehearsal-based continual learning and vision-language zero-shot baselines. Code and data are available at https://github.com/caodoanh2001/MergeSlide.

---

## 122. GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs

**论文链接:** [http://arxiv.org/abs/2511.13007v1](http://arxiv.org/abs/2511.13007v1)

**作者:** Yiyang Zhao, Huiyu Bai, Xuejiao Zhao

**发布时间:** 2025-11-17

**备注:** This paper has been accepted by AAAI 2026-AIA and designated as an oral presentation paper

### GPT解析

### 总结

本文提出了一种名为GEM的生成式熵引导偏好建模方法，用于在低资源和领域特定场景下对齐大型语言模型，解决了专业领域大规模偏好标签难以获取的问题。

### 背景

大型语言模型与人类偏好对齐通常依赖监督奖励模型或外部评判者，需要大量标注数据；但在医学、法律等依赖专业知识的领域，大规模偏好标签往往难以实现。

### 目的

提出一种无需大规模偏好标签的LLM对齐方法，使模型能够在专业领域高效地与人类偏好保持一致。

### 方法

GEM方法包括：1)不训练判别性奖励模型，而是直接训练LLM内部化闭环优化架构；2)基于熵理论的认知过滤模块，利用思维链提示生成多样化候选推理链，并通过令牌评分机制进行排序和加权；3)使用自评估组优势算法(SEGA)微调LLM，将基于熵的分数转换为隐式奖励。

### 主要发现

GEM使LLM能够依靠自身判断，建立了熵引导的闭环认知优化框架，实现了高效的少样本对齐，在通用基准和领域特定任务上均取得显著改进。

### 结论

在数学推理和医疗对话等领域的实验表明，GEM仅需少量偏好数据就能实现有效的LLM对齐，为专业领域的大模型应用提供了新思路。

### 翻译

大型语言模型与人类偏好的对齐通常依赖于需要大量标注的监督奖励模型或外部评判者。然而，在依赖专业知识的领域，如医学和法律，此类大规模偏好标签通常难以实现。本文提出了一种名为GEM的生成式熵引导偏好建模方法，用于在低资源和领域特定场景下对齐LLMs。我们不训练偏好数据上的判别性奖励模型，而是直接训练LLM内部化闭环优化架构，以提取和利用人类偏好中隐含的多维、细粒度认知信号。具体而言，我们基于决策论中的熵理论设计了认知过滤模块，首先利用思维链(CoT)提示从偏好数据生成多样化的候选推理链，随后引入令牌评分机制对采样的CoT进行排序和加权，提高高置信度答案和战略高熵令牌的重要性。基于这些过滤后的偏好，我们使用新颖的自评估组优势算法(SEGA)微调LLM，有效聚合组级认知信号并将基于熵的分数转换为策略优化的隐式奖励。通过这些方式，GEM使LLM能够依靠自身判断，建立了熵引导的闭环认知优化框架，实现了LLMs的高效少样本对齐。在通用基准和领域特定任务(如数学推理和医疗对话)上的实验表明，我们的GEM仅使用少量偏好数据就能取得显著改进。


### 论文摘要

Alignment of large language models (LLMs) with human preferences typically relies on supervised reward models or external judges that demand abundant annotations. However, in fields that rely on professional knowledge, such as medicine and law, such large-scale preference labels are often unachievable. In this paper, we propose a generative entropy-guided preference modeling approach named GEM for LLMs aligment at low-resource and domain-specific scenarios. Instead of training a discriminative reward model on preference data, we directly train the LLM to internalize a closed-loop optimization architecture that can extract and exploit the multi-dimensional, fine-grained cognitive signals implicit in human preferences. Specifically, our Cognitive Filtering module, based on entropy theory in decision making, first leverages Chain-of-Thought (CoT) prompting to generate diverse candidate reasoning chains (CoTs) from preference data. Subsequently, it introduces a token scoring mechanism to rank and weight the sampled CoTs, boosting the importance of high-confidence answers and strategically high-entropy tokens. Building on these filtered preferences, we fine-tune the LLM using a novel self-evaluated group advantage algorithm, SEGA, which effectively aggregates group-level cognitive signals and transforms the entropy-based scores into implicit rewards for policy optimization. In these ways, GEM empowers the LLM to rely on its own judgments and establishes an entropy-guided closed-loop cognitive optimization framework, enabling highly efficient few-shot alignment of LLMs. Experiments on general benchmarks and domain-specific tasks (such as mathematical reasoning and medical dialogues) demonstrate that our GEM achieves significant improvements with few-shot preference data.

---

## 123. PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos

**论文链接:** [http://arxiv.org/abs/2511.12935v1](http://arxiv.org/abs/2511.12935v1)

**作者:** Dianbing Xi, Guoyuan An, Jingsen Zhu, Zhijian Liu, Yuan Liu, Ruiyuan Zhang, Jiayuan Lu, Rui Wang, Yuchi Huo

**发布时间:** 2025-11-17

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

PFAvatar是一种从每日穿搭(OOTD)照片中重建高质量3D头像的新方法，通过两阶段处理实现快速、高质量的头像生成，并在多个方面优于现有方法。

### 背景

现有方法将图像分割为不同资产进行3D组装，容易产生不一致性问题；且处理OOTD照片时面临多样姿势、遮挡和复杂背景的挑战；基于网格的表示方法在分辨率和遮挡处理方面存在局限。

### 目的

开发一种能够从具有多样姿势、遮挡和复杂背景的OOTD照片中重建高质量3D头像的方法，克服现有方法的局限性，提高处理速度和重建质量。

### 方法

方法包含两个阶段：(1)微调姿势感知扩散模型，通过集成ControlNet和条件先验保持损失(CPPL)直接建模全身外观；(2)使用神经辐射场(NeRF)表示3D头像，通过规范化的SMPL-X空间采样和多分辨率3D-SDS进行优化。

### 主要发现

PFAvatar仅需5分钟完成个性化，比先前方法快48倍；在重建保真度、细节保留和遮挡/截断鲁棒性方面优于最先进方法；能够正确处理高频纹理和遮挡；支持虚拟试穿、动画和人类视频重演等下游应用。

### 结论

PFAvatar通过避免图像分解和采用连续辐射场表示，有效解决了OOTD照片中3D头像重建的关键挑战，为实际应用中的3D头像生成提供了高效、高质量的解决方案。

### 翻译

我们提出了PFAvatar（姿势融合头像），一种从'每日穿搭'(OOTD)照片重建高质量3D头像的新方法，这些照片展示多样的姿势、遮挡和复杂背景。我们的方法包含两个阶段：(1)从少量OOTD示例微调姿势感知扩散模型，(2)蒸馏由神经辐射场(NeRF)表示的3D头像。在第一阶段，与先前方法将图像分割为资产(如服装、配饰)进行3D组装容易产生不一致性不同，我们避免了分解，直接建模全身外观。通过集成预训练的ControlNet进行姿势估计和新的条件先验保持损失(CPPL)，我们的方法能够端到端学习细节，同时减轻少量训练中的语言漂移。我们的方法仅需5分钟完成个性化，比先前方法快48倍。在第二阶段，我们引入了一种基于NeRF的头像表示，通过规范化的SMPL-X空间采样和多分辨率3D-SDS进行优化。与遭受分辨率依赖离散化和错误遮挡几何的基于网格表示相比，我们的连续辐射场可以通过透射正确保留高频纹理(如头发)和处理遮挡。实验证明，PFAvatar在重建保真度、细节保留和遮挡/截断鲁棒性方面优于最先进方法，推进了从现实世界OOTD相册中实用3D头像生成的发展。此外，重建的3D头像支持虚拟试穿、动画和人类视频重演等下游应用，进一步证明了我们方法的通用性和实用价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决如何从日常的'每日穿搭'(OOTD)照片中重建高质量的3D个性化头像问题。这类照片通常包含多样姿势、遮挡、复杂背景和截断，给现有3D重建方法带来挑战。这个问题很重要，因为人们有大量日常照片却难以将其转换为3D模型，而高质量的3D头像可用于虚拟试穿、动画制作、视频重演等多种应用，增强虚拟现实、游戏等领域的体验。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法如PuzzleAvatar的局限性：依赖图像分割导致不一致、不支持姿势控制、训练时间长、表示能力有限。基于此，作者设计了PFAvatar的两阶段方法：1)微调姿势感知扩散模型；2)提炼NeRF表示。作者借鉴了ControlNet进行姿势估计、DreamBooth等扩散模型微调方法、NeRF作为3D表示以及分数蒸馏采样(SDS)技术，但针对OOTD照片的特殊性进行了创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是避免将图像分解为多个资产，直接建模全身外观；使用姿势感知扩散模型进行端到端学习；利用NeRF而非网格表示处理遮挡和高频细节。整体流程分两阶段：1)ControlBooth：预处理图像分离前景背景，用ControlNet预测姿势，用GPT-4V生成文本描述，通过重建扩散损失和条件先验保持损失微调扩散模型；2)BoothAvatar：从规范SMPL-X和观察空间采样，用3D-SDS优化NeRF，引入局部几何损失稳定训练，采用多分辨率采样策略。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)姿势感知扩散模型，避免资产分解并引入CPPL损失；2)使用NeRF表示处理遮挡和高频细节；3)3D一致的分数蒸馏采样；4)局部几何约束保留精细结构。相比PuzzleAvatar等前工作，PFAvatar无需图像分割、支持姿势控制、训练时间缩短48倍(5分钟)、能更好处理复杂拓扑和细节、更鲁棒地处理遮挡和截断。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PFAvatar通过姿势感知扩散模型和神经辐射场表示，实现了从日常穿搭照片快速重建高质量3D个性化头像，在重建保真度、细节保留和遮挡处理方面显著优于现有方法。'}


### 论文摘要

We propose PFAvatar (Pose-Fusion Avatar), a new method that reconstructs high-quality 3D avatars from ``Outfit of the Day'' (OOTD) photos, which exhibit diverse poses, occlusions, and complex backgrounds. Our method consists of two stages: (1) fine-tuning a pose-aware diffusion model from few-shot OOTD examples and (2) distilling a 3D avatar represented by a neural radiance field (NeRF). In the first stage, unlike previous methods that segment images into assets (e.g., garments, accessories) for 3D assembly, which is prone to inconsistency, we avoid decomposition and directly model the full-body appearance. By integrating a pre-trained ControlNet for pose estimation and a novel Condition Prior Preservation Loss (CPPL), our method enables end-to-end learning of fine details while mitigating language drift in few-shot training. Our method completes personalization in just 5 minutes, achieving a 48$\times$ speed-up compared to previous approaches. In the second stage, we introduce a NeRF-based avatar representation optimized by canonical SMPL-X space sampling and Multi-Resolution 3D-SDS. Compared to mesh-based representations that suffer from resolution-dependent discretization and erroneous occluded geometry, our continuous radiance field can preserve high-frequency textures (e.g., hair) and handle occlusions correctly through transmittance. Experiments demonstrate that PFAvatar outperforms state-of-the-art methods in terms of reconstruction fidelity, detail preservation, and robustness to occlusions/truncations, advancing practical 3D avatar generation from real-world OOTD albums. In addition, the reconstructed 3D avatar supports downstream applications such as virtual try-on, animation, and human video reenactment, further demonstrating the versatility and practical value of our approach.

---

## 124. Genomic Next-Token Predictors are In-Context Learners

**论文链接:** [http://arxiv.org/abs/2511.12797v1](http://arxiv.org/abs/2511.12797v1)

**作者:** Nathan Breslow, Aayush Mishra, Mahler Revsine, Michael C. Schatz, Anqi Liu, Daniel Khashabi

**发布时间:** 2025-11-16

### GPT解析

### 总结

本研究探索了上下文学习（ICL）能力是否能在非语言序列领域（如基因组序列）中通过大规模预测训练自然出现。研究发现基因组模型同样表现出上下文学习能力，支持了ICL是大规模预测建模在丰富数据上的结果的假设。

### 背景

上下文学习（ICL）是大型语言模型的一项重要能力，指模型能从输入中提供的示例中推断和应用抽象模式。以往研究通常将这种能力归因于人类语言的独特统计属性，但尚不清楚这种能力是否能在其他序列领域中自然出现。

### 目的

探索上下文学习能力是否能在基因组序列等其他符号序列领域中通过大规模预测训练自然出现，而非仅限于人类语言领域。

### 方法

研究使用Evo2基因组模型，该模型主要在下一个核苷酸（A/T/C/G）预测任务上训练，规模与中等规模的大型语言模型相当。研究团队开发了一个受控实验框架，包含以语言和基因组形式实例化的符号推理任务，使能够直接比较基因组模型和语言模型中的ICL表现。

### 主要发现

基因组模型与语言模型类似，随着上下文演示数量的增加，在模式归纳方面表现出对数线性增长。这是首次在基因组序列中自然出现ICL的证据。

### 结论

ICL能力可以通过大规模预测建模在丰富数据上自然产生，不仅限于人类语言。这一发现将涌现元学习扩展到语言之外，指向了一种统一的、模式无关的上下文学习观点。

### 翻译

上下文学习（ICL）——模型从输入中提供的示例中推断和应用抽象模式的能力——已经在主要为人类文本的下一个词预测任务而训练的大型语言模型中被广泛研究。事实上，先前的研究通常将这种涌现行为归因于人类语言中的独特统计特性。这引出了一个根本性问题：ICL能否纯粹通过大规模预测训练在其他序列领域中自然出现？为了探索这一点，我们转向基因组序列，这是一个富含统计结构的替代符号领域。具体来说，我们研究了Evo2基因组模型，该模型主要在下一个核苷酸（A/T/C/G）预测任务上训练，规模与中等规模的大型语言模型相当。我们开发了一个受控实验框架，包含以语言和基因组形式实例化的符号推理任务，使能够直接比较基因组和语言模型中的ICL。我们的结果表明，基因组模型与语言模型类似，随着上下文演示数量的增加，在模式归纳方面表现出对数线性增长。据我们所知，这是首次在基因组序列中自然出现ICL的证据，支持了ICL是大规模预测建模在丰富数据上的结果的假设。这些发现将涌现元学习扩展到语言之外，指向了一种统一的、模式无关的上下文学习观点。


### 论文摘要

In-context learning (ICL) -- the capacity of a model to infer and apply abstract patterns from examples provided within its input -- has been extensively studied in large language models trained for next-token prediction on human text. In fact, prior work often attributes this emergent behavior to distinctive statistical properties in human language. This raises a fundamental question: can ICL arise organically in other sequence domains purely through large-scale predictive training?   To explore this, we turn to genomic sequences, an alternative symbolic domain rich in statistical structure. Specifically, we study the Evo2 genomic model, trained predominantly on next-nucleotide (A/T/C/G) prediction, at a scale comparable to mid-sized LLMs. We develop a controlled experimental framework comprising symbolic reasoning tasks instantiated in both linguistic and genomic forms, enabling direct comparison of ICL across genomic and linguistic models. Our results show that genomic models, like their linguistic counterparts, exhibit log-linear gains in pattern induction as the number of in-context demonstrations increases. To the best of our knowledge, this is the first evidence of organically emergent ICL in genomic sequences, supporting the hypothesis that ICL arises as a consequence of large-scale predictive modeling over rich data. These findings extend emergent meta-learning beyond language, pointing toward a unified, modality-agnostic view of in-context learning.

---

## 125. Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation

**论文链接:** [http://arxiv.org/abs/2511.12779v1](http://arxiv.org/abs/2511.12779v1)

**作者:** Zhenshuo Zhang, Minxuan Duan, Youran Ye, Hongyang R. Zhang

**发布时间:** 2025-11-16

**备注:** 17 pages. To appear in AAAI'26

### GPT解析

### 总结

研究强化学习中多目标优化的高效策略估计问题，提出两阶段方法将多个目标分组训练

### 背景

在机器人技术、控制和语言模型偏好优化等应用中，随着目标数量增长，为所有n个目标学习单一策略变得次优

### 目的

寻求将n个目标划分为k<<n个组，每组包含可以一起训练的相关目标，从而提高效率

### 方法

提出两阶段程序：首先使用多任务学习学习所有目标的元策略，然后将元策略适应到多个随机采样的目标子集；利用策略网络的一阶近似属性；基于估计的亲和度分数矩阵进行聚类

### 主要发现

PolicyGradEx算法在三个机器人控制和Meta-World基准测试上平均比最先进基线高出16%，速度提高最多26倍；基于聚类的损失聚类比随机分组和基于梯度相似性的分组提高了19%

### 结论

所提出的方法能够高效估计策略并优化多目标分组，在多个基准测试上表现优异

### 翻译

我们研究了在强化学习(RL)中高效估计同时优化多个目标的策略问题。给定n个目标(或任务)，我们寻求将这些目标最优地划分为k<<n个组，其中每组包含可以一起训练的相关目标。这个问题出现在机器人技术、控制和语言模型偏好优化等应用中，随着n的增长，为所有n个目标学习单一策略是次优的。我们引入一个两阶段程序——元训练后微调——来解决这个问题。我们首先使用多任务学习为所有目标学习元策略。然后，我们将元策略适应到多个随机采样的目标子集。适应步骤利用了训练良好的策略网络的一阶近似属性，经验验证在各种RL环境中误差在2%以内。 resulting算法PolicyGradEx，给定策略评估算法，高效估计聚合任务亲和度分数矩阵。基于估计的亲和度分数矩阵，我们通过最大化簇内亲和度分数将n个目标聚类为k组。在三个机器人控制和Meta-World基准测试上的实验表明，我们的方法平均比最先进的基线高出16%，同时相对于执行完整训练以获得聚类，提供高达26倍的速度提升。消融研究验证了我们方法的每个组件。例如，与随机分组和基于梯度相似性的分组相比，我们的基于损失的聚类提高了19%。最后，我们通过测量损失表面的Hessian迹来分析策略网络的泛化误差，这相对于观察到的泛化误差给出了非平凡的度量。


### 论文摘要

We study the problem of efficiently estimating policies that simultaneously optimize multiple objectives in reinforcement learning (RL). Given $n$ objectives (or tasks), we seek the optimal partition of these objectives into $k \ll n$ groups, where each group comprises related objectives that can be trained together. This problem arises in applications such as robotics, control, and preference optimization in language models, where learning a single policy for all $n$ objectives is suboptimal as $n$ grows. We introduce a two-stage procedure -- meta-training followed by fine-tuning -- to address this problem. We first learn a meta-policy for all objectives using multitask learning. Then, we adapt the meta-policy to multiple randomly sampled subsets of objectives. The adaptation step leverages a first-order approximation property of well-trained policy networks, which is empirically verified to be accurate within a $2\%$ error margin across various RL environments. The resulting algorithm, PolicyGradEx, efficiently estimates an aggregate task-affinity score matrix given a policy evaluation algorithm. Based on the estimated affinity score matrix, we cluster the $n$ objectives into $k$ groups by maximizing the intra-cluster affinity scores. Experiments on three robotic control and the Meta-World benchmarks demonstrate that our approach outperforms state-of-the-art baselines by $16\%$ on average, while delivering up to $26\times$ faster speedup relative to performing full training to obtain the clusters. Ablation studies validate each component of our approach. For instance, compared with random grouping and gradient-similarity-based grouping, our loss-based clustering yields an improvement of $19\%$. Finally, we analyze the generalization error of policy networks by measuring the Hessian trace of the loss surface, which gives non-vacuous measures relative to the observed generalization errors.

---

## 126. Prompt-Driven Domain Adaptation for End-to-End Autonomous Driving via In-Context RL

**论文链接:** [http://arxiv.org/abs/2511.12755v1](http://arxiv.org/abs/2511.12755v1)

**作者:** Aleesha Khurram, Amir Moeini, Shangtong Zhang, Rohan Chandra

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种名为上下文强化学习(ICRL)的新方法，用于解决自动驾驶领域在恶劣天气条件下的域适应问题，该方法无需更新模型参数或收集额外数据即可实现更安全、高效和舒适的驾驶策略。

### 背景

尽管自动驾驶取得了显著进展，许多端到端系统在域适应方面仍存在困难，特别是在将晴朗天气训练的策略转移到恶劣天气条件时。传统域适应策略(收集额外数据或重新训练模型)随着驾驶规模和复杂性增加变得不切实际。

### 目的

解决现有提示驱动域适应方法的两个局限性：(i)仅限于感知任务如检测和分割；(ii)需要专家少样本数据。提出一种新的推理时少样本提示驱动域适应方法，用于闭环自动驾驶在恶劣天气条件下的应用。

### 方法

提出一种名为上下文强化学习(ICRL)的新方法，用于闭环自动驾驶在恶劣天气条件下的域适应。该方法不更新模型参数，也不需要在恶劣天气条件下收集额外数据，通过在推理时使用观察到的通用轨迹来扩展到闭环驾驶。

### 主要发现

在CARLA模拟器上的实验表明，ICRL与最先进的提示驱动域适应基线相比，在目标域中产生了更安全、更高效、更舒适的驾驶策略。

### 结论

ICRL方法在自动驾驶领域适应方面取得了进展，特别是在恶劣天气条件下的闭环驾驶，实现了无需额外数据收集或模型参数更新的域适应。

### 翻译

尽管自动驾驶取得了显著进展，许多端到端系统在域适应方面仍存在困难，例如将在晴朗天气条件下训练的策略转移到恶劣天气条件。文献中的典型域适应策略包括在目标域收集额外数据或重新训练模型，或两者兼有。随着驾驶规模和复杂性的增加，这些策略很快变得不切实际。这些局限性促使人们研究在推理时使用大型语言模型和视觉语言模型进行少样本和零样本提示驱动的域适应。这些方法通过在提示中添加少量状态-动作轨迹(类似于上下文学习)来工作。然而，这种方法存在两个局限性：(i)提示驱动的域适应方法目前仅限于检测和分割等感知任务；(ii)它们需要专家的少样本数据。在这项工作中，我们提出了一种新的推理时少样本提示驱动域适应方法，用于在恶劣天气条件下的闭环自动驾驶，使用上下文强化学习(ICRL)。与其他提示驱动域适应方法类似，我们的方法不需要更新模型参数，也不需要在恶劣天气条件下收集额外数据。此外，我们的方法通过扩展到使用推理过程中观察到的通用轨迹进行闭环驾驶，推动了提示驱动域适应的最先进水平。在CARLA模拟器上的实验表明，与最先进的提示驱动域适应基线相比，ICRL在目标域中产生了更安全、更高效、更舒适的驾驶策略。


### 论文摘要

Despite significant progress and advances in autonomous driving, many end-to-end systems still struggle with domain adaptation (DA), such as transferring a policy trained under clear weather to adverse weather conditions. Typical DA strategies in the literature include collecting additional data in the target domain or re-training the model, or both. Both these strategies quickly become impractical as we increase scale and complexity of driving. These limitations have encouraged investigation into few-shot and zero-shot prompt-driven DA at inference time involving LLMs and VLMs. These methods work by adding a few state-action trajectories during inference to the prompt (similar to in-context learning). However, there are two limitations of such an approach: $(i)$ prompt-driven DA methods are currently restricted to perception tasks such as detection and segmentation and $(ii)$ they require expert few-shot data. In this work, we present a new approach to inference-time few-shot prompt-driven DA for closed-loop autonomous driving in adverse weather condition using in-context reinforcement learning (ICRL). Similar to other prompt-driven DA methods, our approach does not require any updates to the model parameters nor does it require additional data collection in adversarial weather regime. Furthermore, our approach advances the state-of-the-art in prompt-driven DA by extending to closed driving using general trajectories observed during inference. Our experiments using the CARLA simulator show that ICRL results in safer, more efficient, and more comfortable driving policies in the target domain compared to state-of-the-art prompt-driven DA baselines.

---

## 127. MFI-ResNet: Efficient ResNet Architecture Optimization via MeanFlow Compression and Selective Incubation

**论文链接:** [http://arxiv.org/abs/2511.12422v1](http://arxiv.org/abs/2511.12422v1)

**作者:** Nuolin Sun, Linyuan Wang, Haonan Wei, Lei Li, Bin Yan

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种名为MeanFlow-Incubated ResNet (MFI-ResNet)的新型网络架构，通过压缩-扩展策略联合提高了参数效率和判别性能。

### 背景

ResNet通过残差连接机制在计算机视觉领域取得巨大成功，可视为常微分方程的离散形式。MeanFlow是一种流匹配模型，通过学习均值速度场实现一步生成建模。

### 目的

提出MFI-ResNet模型，采用压缩-扩展策略联合提高参数效率和判别性能。

### 方法

压缩阶段：将每个ResNet阶段的多层结构简化为一个或两个MeanFlow模块构建轻量级元模型；扩展阶段：对前三个阶段应用选择性孵化策略扩展为残差块配置，保持最后一个阶段为MeanFlow形式，并进行微调。

### 主要发现

在CIFAR-10和CIFAR-100数据集上，MFI-ResNet与ResNet-50相比参数分别减少46.28%和45.59%，同时准确率提高0.23%和0.17%。

### 结论

生成流场可以有效表征ResNet中的特征变换过程，为理解生成建模和判别学习的关系提供了新视角。

### 翻译

ResNet通过其残差连接机制在计算机视觉领域取得了巨大成功。ResNet可以看作是常微分方程的一种离散形式。从这个角度来看，单个ResNet阶段中的多个残差块本质上是对该阶段特征变换的多步离散迭代。最近提出的流匹配模型MeanFlow通过学习均值速度场来转换分布，实现了一步生成建模。受此启发，我们提出了MeanFlow-Incubated ResNet (MFI-ResNet)，它采用压缩-扩展策略来联合提高参数效率和判别性能。在压缩阶段，我们将每个ResNet阶段内的多层结构简化为一个或两个MeanFlow模块，构建轻量级元模型。在扩展阶段，我们对前三个阶段应用选择性孵化策略，将其扩展为与基准ResNet模型匹配的残差块配置，同时保持最后一个阶段为MeanFlow形式，并对孵化后的模型进行微调。实验结果表明，在CIFAR-10和CIFAR-100数据集上，MFI-ResNet实现了显著的参数效率，与ResNet-50相比参数分别减少了46.28%和45.59%，同时准确率仍提高了0.23%和0.17%。这表明生成流场可以有效表征ResNet中的特征变换过程，为理解生成建模与判别学习之间的关系提供了新视角。


### 论文摘要

ResNet has achieved tremendous success in computer vision through its residual connection mechanism. ResNet can be viewed as a discretized form of ordinary differential equations (ODEs). From this perspective, the multiple residual blocks within a single ResNet stage essentially perform multi-step discrete iterations of the feature transformation for that stage. The recently proposed flow matching model, MeanFlow, enables one-step generative modeling by learning the mean velocity field to transform distributions. Inspired by this, we propose MeanFlow-Incubated ResNet (MFI-ResNet), which employs a compression-expansion strategy to jointly improve parameter efficiency and discriminative performance. In the compression phase, we simplify the multi-layer structure within each ResNet stage to one or two MeanFlow modules to construct a lightweight meta model. In the expansion phase, we apply a selective incubation strategy to the first three stages, expanding them to match the residual block configuration of the baseline ResNet model, while keeping the last stage in MeanFlow form, and fine-tune the incubated model. Experimental results show that on CIFAR-10 and CIFAR-100 datasets, MFI-ResNet achieves remarkable parameter efficiency, reducing parameters by 46.28% and 45.59% compared to ResNet-50, while still improving accuracy by 0.23% and 0.17%, respectively. This demonstrates that generative flow-fields can effectively characterize the feature transformation process in ResNet, providing a new perspective for understanding the relationship between generative modeling and discriminative learning.

---

## 128. Evaluating Model-Agnostic Meta-Learning on MetaWorld ML10 Benchmark: Fast Adaptation in Robotic Manipulation Tasks

**论文链接:** [http://arxiv.org/abs/2511.12383v1](http://arxiv.org/abs/2511.12383v1)

**作者:** Sanjar Atamuradov

**发布时间:** 2025-11-15

**备注:** 7 pages, 5 figures

### GPT解析

### 总结

本文评估了MAML-TRPO算法在MetaWorld ML10基准测试上的表现，展示了元学习在机器人操作任务中的单次适应能力，同时指出了存在的泛化差距和任务间性能差异大的问题。

### 背景

元学习算法能够以最少的数据快速适应新任务，这对现实世界机器人系统至关重要。MetaWorld ML10是一个包含十种多样化机器人操作任务的基准测试套件，包括推动、抓取和抽屉操作等语义不同的操作行为。

### 目的

评估MAML-TRPO算法在MetaWorld ML10基准测试上的表现，分析其学习通用初始化以实现少样本适应的能力，特别是在语义不同的操作行为上的表现。

### 方法

实现并分析MAML-TRPO算法，该算法结合了Model-Agnostic Meta-Learning和Trust Region Policy Optimization。在包含十种多样化机器人操作任务的MetaWorld ML10基准测试上进行实验。

### 主要发现

MAML实现了有效的单次适应，在单次梯度更新后有明显性能提升；训练任务最终成功率达到21.0%，保留的测试任务达到13.2%；元训练过程中出现泛化差距，测试任务性能停滞而训练任务性能继续提高；任务级分析显示适应效果差异很大，成功率从0%到80%不等。

### 结论

基于梯度的元学习在多样化机器人操作方面既有前景也有当前局限性，未来工作应在任务感知适应和结构化策略架构方向发展。

### 翻译

元学习算法使系统能够以最少的数据快速适应新任务，这对现实世界的机器人系统至关重要。本文在包含十种多样化机器人操作任务的MetaWorld ML10基准测试上评估了Model-Agnostic Meta-Learning (MAML)与Trust Region Policy Optimization (TRPO)的结合。我们实现并分析了MAML-TRPO学习通用初始化的能力，该初始化有助于在包括推动、抓取和抽屉操作在内的语义不同的操作行为中实现少样本适应。我们的实验证明，MAML实现了有效的单次适应，在单次梯度更新后有明显性能提升，在训练任务上达到21.0%的最终成功率，在保留的测试任务上达到13.2%。然而，我们观察到在元训练过程中出现的泛化差距，测试任务性能停滞而训练任务性能继续提高。任务级分析显示适应效果差异很大，不同操作技能的成功率从0%到80%不等。这些发现既突显了基于梯度的元学习在多样化机器人操作方面的前景，也指出了其当前局限性，并建议未来工作应在任务感知适应和结构化策略架构方向发展。


### 论文摘要

Meta-learning algorithms enable rapid adaptation to new tasks with minimal data, a critical capability for real-world robotic systems. This paper evaluates Model-Agnostic Meta-Learning (MAML) combined with Trust Region Policy Optimization (TRPO) on the MetaWorld ML10 benchmark, a challenging suite of ten diverse robotic manipulation tasks. We implement and analyze MAML-TRPO's ability to learn a universal initialization that facilitates few-shot adaptation across semantically different manipulation behaviors including pushing, picking, and drawer manipulation. Our experiments demonstrate that MAML achieves effective one-shot adaptation with clear performance improvements after a single gradient update, reaching final success rates of 21.0% on training tasks and 13.2% on held-out test tasks. However, we observe a generalization gap that emerges during meta-training, where performance on test tasks plateaus while training task performance continues to improve. Task-level analysis reveals high variance in adaptation effectiveness, with success rates ranging from 0% to 80% across different manipulation skills. These findings highlight both the promise and current limitations of gradient-based meta-learning for diverse robotic manipulation, and suggest directions for future work in task-aware adaptation and structured policy architectures.

---

## 129. Multimodal RGB-HSI Feature Fusion with Patient-Aware Incremental Heuristic Meta-Learning for Oral Lesion Classification

**论文链接:** [http://arxiv.org/abs/2511.12268v1](http://arxiv.org/abs/2511.12268v1)

**作者:** Rupam Mukherjee, Rajkumar Daniel, Soujanya Hazra, Shirin Dasgupta, Subhamoy Mandal

**发布时间:** 2025-11-15

**备注:** 4 pages, 1 figure, 2 tables

### GPT解析

### 总结

研究提出了一种统一的四类口腔病变分类器，通过集成深度RGB嵌入、高光谱重建、手工制作的谱-纹理描述符和人口统计元数据，提高了资源有限环境中口腔癌和潜在恶性病变的早期检测能力。

### 背景

在资源有限的环境中，由于标注数据有限，口腔癌和潜在恶性病变的早期检测具有挑战性。

### 目的

开发一个统一的四类口腔病变分类器，以提高资源有限环境中口腔癌和潜在恶性病变的早期检测能力。

### 方法

开发了一个整合多种特征的分类器，包括深度RGB嵌入、高光谱重建、手工制作的谱-纹理描述符和人口统计元数据。使用经病理学家验证的口腔图像子集，通过微调的ConvNeXt-v2编码器处理图像，将RGB重建为31波段的高光谱立方体。提取了血红蛋白敏感指数、纹理特征和光谱形状度量，并与深度和临床特征融合。评估了多种机器学习模型，并引入了一种增量启发式元学习器(IHML)，通过概率叠加和患者级别后验平滑组合校准的基础分类器。

### 主要发现

在未见过的患者分割上，所提出的框架达到了66.23%的宏平均F1分数和64.56%的准确率。

### 结论

高光谱重建和不确定性感知元学习显著提高了口腔病变筛查的鲁棒性，对资源有限环境中的口腔癌早期检测具有实用价值。

### 翻译

在资源有限的环境中，由于标注数据有限，口腔癌和潜在恶性病变的早期检测具有挑战性。我们提出了一种统一的四类口腔病变分类器，集成了深度RGB嵌入、高光谱重建、手工制作的谱-纹理描述符和人口统计元数据。收集并整理了经病理学家验证的口腔图像子集，使用微调的ConvNeXt-v2编码器进行处理，随后将RGB重建为31波段的高光谱立方体。提取了血红蛋白敏感指数、纹理特征和光谱形状度量，并与深度和临床特征融合。评估了多种机器学习模型，采用患者级别的验证。我们进一步引入了一种增量启发式元学习器(IHML)，通过概率叠加和患者级别后验平滑组合校准的基础分类器。在未见过的患者分割上，所提出的框架达到了66.23%的宏平均F1分数和64.56%的准确率。结果表明，高光谱重建和不确定性感知元学习显著提高了口腔病变筛查的鲁棒性。


### 论文摘要

Early detection of oral cancer and potentially malignant disorders is challenging in low-resource settings due to limited annotated data. We present a unified four-class oral lesion classifier that integrates deep RGB embeddings, hyperspectral reconstruction, handcrafted spectral-textural descriptors, and demographic metadata. A pathologist-verified subset of oral cavity images was curated and processed using a fine-tuned ConvNeXt-v2 encoder, followed by RGB-to-HSI reconstruction into 31-band hyperspectral cubes. Haemoglobin-sensitive indices, texture features, and spectral-shape measures were extracted and fused with deep and clinical features. Multiple machine-learning models were assessed with patient-wise validation. We further introduce an incremental heuristic meta-learner (IHML) that combines calibrated base classifiers through probabilistic stacking and patient-level posterior smoothing. On an unseen patient split, the proposed framework achieved a macro F1 of 66.23% and an accuracy of 64.56%. Results demonstrate that hyperspectral reconstruction and uncertainty-aware meta-learning substantially improve robustness for real-world oral lesion screening.

---

## 130. Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation

**论文链接:** [http://arxiv.org/abs/2511.12200v1](http://arxiv.org/abs/2511.12200v1)

**作者:** Sujun Sun, Haowen Gu, Cheng Xie, Yanxu Ren, Mingwu Ren, Haofeng Zhang

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种层次语义学习框架，用于解决跨域少样本分割中的语义判别能力不足问题。

### 背景

跨域少样本分割(CD-FSS)旨在使用少量标注样本对未见过的目标域进行分割，目标域与训练源域的数据分布有显著不同，近年来该领域取得了显著进展。

### 目的

解决现有CD-FSS方法主要关注风格差异而忽略分割粒度差异的问题，提高模型对目标域中新颖类的语义判别能力。

### 方法

提出层次语义学习(HSL)框架，包含双风格随机化(DSR)模块和层次语义挖掘(HSM)模块来学习层次语义特征，以及原型置信度调制阈值(PCMT)模块来缓解分割歧义。

### 主要发现

现有方法忽略了分割粒度差异导致语义判别能力不足，通过层次语义学习可以有效提高模型在不同粒度上的语义识别能力。

### 结论

在四个流行的目标域数据集上的实验表明，该方法达到了最先进的性能，证明了所提框架的有效性。

### 翻译

跨域少样本分割(CD-FSS)旨在使用少量标注样本对未见过的目标域进行分割，这些目标域与训练域有显著不同的数据分布。近年来，该领域取得了显著进展。然而，现有CD-FSS方法主要关注源域和目标域之间的风格差异，而忽略了分割粒度差异，导致目标域中新颖类的语义判别能力不足。因此，我们提出了层次语义学习(HSL)框架来解决这一问题。具体而言，我们引入了双风格随机化(DSR)模块和层次语义挖掘(HSM)模块来学习层次语义特征，从而增强模型在不同粒度上识别语义的能力。DSR分别通过前景和全局风格随机化模拟具有不同前景-背景风格差异和整体风格变化的目标域数据，而HSM利用多尺度超像素引导模型在不同粒度上挖掘类内一致性和类间区分性。此外，我们还提出了原型置信度调制阈值(PCMT)模块来缓解前景和背景过度相似时的分割歧义。在四个流行的目标域数据集上进行了大量实验，结果表明我们的方法达到了最先进的性能。


### 论文摘要

Cross-domain Few-shot Segmentation (CD-FSS) aims to segment novel classes from target domains that are not involved in training and have significantly different data distributions from the source domain, using only a few annotated samples, and recent years have witnessed significant progress on this task. However, existing CD-FSS methods primarily focus on style gaps between source and target domains while ignoring segmentation granularity gaps, resulting in insufficient semantic discriminability for novel classes in target domains. Therefore, we propose a Hierarchical Semantic Learning (HSL) framework to tackle this problem. Specifically, we introduce a Dual Style Randomization (DSR) module and a Hierarchical Semantic Mining (HSM) module to learn hierarchical semantic features, thereby enhancing the model's ability to recognize semantics at varying granularities. DSR simulates target domain data with diverse foreground-background style differences and overall style variations through foreground and global style randomization respectively, while HSM leverages multi-scale superpixels to guide the model to mine intra-class consistency and inter-class distinction at different granularities. Additionally, we also propose a Prototype Confidence-modulated Thresholding (PCMT) module to mitigate segmentation ambiguity when foreground and background are excessively similar. Extensive experiments are conducted on four popular target domain datasets, and the results demonstrate that our method achieves state-of-the-art performance.

---

## 131. From Classification to Cross-Modal Understanding: Leveraging Vision-Language Models for Fine-Grained Renal Pathology

**论文链接:** [http://arxiv.org/abs/2511.11984v1](http://arxiv.org/abs/2511.11984v1)

**作者:** Zhenhao Guo, Rachit Saluja, Tianyuan Yao, Quan Liu, Junchao Zhu, Haibo Wang, Daniel Reisenbüchler, Yuankai Huo, Benjamin Liechty, David J. Pisapia, Kenji Ikemura, Steven Salvatoree, Surya Seshane, Mert R. Sabuncu, Yihe Yang, Ruining Deng

**发布时间:** 2025-11-15

### GPT解析

### 总结

本研究探讨了在数据受限条件下如何有效利用视觉语言模型进行肾小球精细亚型分类，评估了不同模型架构和适应策略的性能表现

### 背景

肾脏活检中的精细肾小球亚型分类对临床诊断至关重要，但临床上有价值的标签稀缺且难以获取。现有计算病理学方法主要使用仅基于图像的模型在有完全监督的情况下评估粗略的疾病分类，不清楚视觉语言模型应如何适应临床上有意义的亚型分类

### 目的

将精细肾小球亚型建模为临床现实的少样本问题，系统评估在这种设置下病理学专业视觉语言模型和通用视觉语言模型的性能，为在真实临床数据约束下的模型选择和训练提供指导

### 方法

评估分类性能（准确率、AUC、F1值）以及学习表示的几何结构，检查图像和文本嵌入之间的特征对齐以及肾小球亚型的可分离性，联合分析样本数量、模型架构和领域知识以及适应策略

### 主要发现

病理学专业的视觉语言骨干模型与简单微调相结合是最有效的起点；即使每个肾小球亚型只有4-8个标记示例，这些模型也能开始捕捉差异并显示出显著的判别力和校准改进；正负样本之间的判别与图像-文本对齐同样重要

### 结论

监督水平和适应策略共同塑造了诊断性能和多模态结构，为模型选择、适应策略和标注投资提供了指导

### 翻译

精细的肾小球亚型分类是肾脏活检解释的核心，但临床上宝贵的标签稀缺且难以获取。现有的计算病理学方法倾向于在有完全监督的情况下使用仅基于图像的模型评估粗略的疾病分类，因此尚不清楚视觉语言模型应如何适应临床上有意义的亚型分类。在这项工作中，我们将精细的肾小球亚型建模为临床现实的少样本问题，并系统评估在这种设置下病理学专业和通用视觉语言模型。我们不仅评估分类性能（准确率、AUC、F1），还分析学习表示的几何结构，检查图像和文本嵌入之间的特征对齐以及肾小球亚型的可分离性。通过联合分析样本数量、模型架构和领域知识以及适应策略，本研究为在真实临床数据约束下的未来模型选择和训练提供了指导。我们的结果表明，病理学专业的视觉语言骨干模型与简单微调相结合是最有效的起点。即使每个肾小球亚型只有4-8个标记示例，这些模型也能开始捕捉差异并显示出显著的判别力和校准改进，尽管额外的监督继续带来渐进式改进。我们还发现，正负样本之间的判别与图像-文本对齐同样重要。总体而言，我们的结果表明监督水平和适应策略共同塑造了诊断性能和多模态结构，为模型选择、适应策略和标注投资提供了指导


### 论文摘要

Fine-grained glomerular subtyping is central to kidney biopsy interpretation, but clinically valuable labels are scarce and difficult to obtain. Existing computational pathology approaches instead tend to evaluate coarse diseased classification under full supervision with image-only models, so it remains unclear how vision-language models (VLMs) should be adapted for clinically meaningful subtyping under data constraints. In this work, we model fine-grained glomerular subtyping as a clinically realistic few-shot problem and systematically evaluate both pathology-specialized and general-purpose vision-language models under this setting. We assess not only classification performance (accuracy, AUC, F1) but also the geometry of the learned representations, examining feature alignment between image and text embeddings and the separability of glomerular subtypes. By jointly analyzing shot count, model architecture and domain knowledge, and adaptation strategy, this study provides guidance for future model selection and training under real clinical data constraints. Our results indicate that pathology-specialized vision-language backbones, when paired with the vanilla fine-tuning, are the most effective starting point. Even with only 4-8 labeled examples per glomeruli subtype, these models begin to capture distinctions and show substantial gains in discrimination and calibration, though additional supervision continues to yield incremental improvements. We also find that the discrimination between positive and negative examples is as important as image-text alignment. Overall, our results show that supervision level and adaptation strategy jointly shape both diagnostic performance and multimodal structure, providing guidance for model selection, adaptation strategies, and annotation investment.

---

## 132. Chain-of-Generation: Progressive Latent Diffusion for Text-Guided Molecular Design

**论文链接:** [http://arxiv.org/abs/2511.11894v1](http://arxiv.org/abs/2511.11894v1)

**作者:** Lingxiao Li, Haobo Zhang, Bin Chen, Jiayu Zhou

**发布时间:** 2025-11-14

**备注:** 22 pages, 7 figures, 10 tables

### GPT解析

### 总结

本文提出了Chain-of-Generation (CoG)，一种无需训练的多阶段潜在扩散框架，用于解决文本条件分子生成中的挑战。CoG通过将提示分解为语义段并逐步引入，实现了更准确地满足复杂语言约束的分子生成。

### 背景

文本条件分子生成旨在将自然语言描述转化为化学结构，使科学家能够指定功能基团、支架和物理化学约束。基于扩散的模型，特别是潜在扩散模型，最近在分子生成中显示出潜力，但现有方法依赖一次性条件处理，难以满足提示中的所有要求。

### 目的

解决一次性条件处理带来的三个主要挑战：生成组件的可解释性差、无法生成所有子结构、同时考虑所有要求时过度雄心。提高生成分子的语义对齐、多样性和可控性，使生成的分子更准确地反映复杂、组合式提示，并提供透明的生成过程。

### 方法

提出Chain-of-Generation (CoG)框架，将每个提示分解为课程排序的语义段，并逐步将这些段作为中间目标引入，引导去噪轨迹朝向满足日益丰富语言约束的分子。引入后对齐学习阶段，加强文本和分子潜在空间之间的对应关系。

### 主要发现

一次性条件处理存在三个显著挑战；CoG在基准和实际任务上表现出比一次性基线更高的语义对齐、多样性和可控性；CoG能够更忠实地反映复杂、组合式提示的分子生成要求。

### 结论

CoG通过分解提示并逐步引入语义约束，有效解决了文本条件分子生成中的挑战，能够生成更符合复杂语言描述的分子，同时提供透明的生成过程。

### 翻译

文本条件分子生成旨在将自然语言描述转化为化学结构，使科学家能够指定功能基团、支架和物理化学约束，无需手工规则。基于扩散的模型，特别是潜在扩散模型，最近通过在紧凑捕获分子语义的连续潜在空间中进行随机搜索，显示出良好前景。然而现有方法依赖一次性条件处理，其中整个提示被编码一次并应用于整个扩散过程，难以满足提示中的所有要求。我们讨论了一次性条件生成的三个突出挑战，包括生成组件的可解释性差、无法生成所有子结构、以及同时考虑所有要求时的过度雄心。然后我们提出三个原则来解决这些挑战，基于此我们提出了Chain-of-Generation (CoG)，一种无需训练的多阶段潜在扩散框架。CoG将每个提示分解为课程排序的语义段，并将它们逐步引入作为中间目标，引导去噪轨迹朝向满足日益丰富语言约束的分子。为了加强语义引导，我们进一步引入了后对齐学习阶段，增强了文本和分子潜在空间之间的对应关系。在基准和实际任务上的广泛实验表明，CoG比一次性基线产生更高的语义对齐、多样性和可控性，生成的分子更忠实地反映复杂、组合式提示，同时提供透明的生成过程洞察。


### 论文摘要

Text-conditioned molecular generation aims to translate natural-language descriptions into chemical structures, enabling scientists to specify functional groups, scaffolds, and physicochemical constraints without handcrafted rules. Diffusion-based models, particularly latent diffusion models (LDMs), have recently shown promise by performing stochastic search in a continuous latent space that compactly captures molecular semantics. Yet existing methods rely on one-shot conditioning, where the entire prompt is encoded once and applied throughout diffusion, making it hard to satisfy all the requirements in the prompt. We discuss three outstanding challenges of one-shot conditioning generation, including the poor interpretability of the generated components, the failure to generate all substructures, and the overambition in considering all requirements simultaneously. We then propose three principles to address those challenges, motivated by which we propose Chain-of-Generation (CoG), a training-free multi-stage latent diffusion framework. CoG decomposes each prompt into curriculum-ordered semantic segments and progressively incorporates them as intermediate goals, guiding the denoising trajectory toward molecules that satisfy increasingly rich linguistic constraints. To reinforce semantic guidance, we further introduce a post-alignment learning phase that strengthens the correspondence between textual and molecular latent spaces. Extensive experiments on benchmark and real-world tasks demonstrate that CoG yields higher semantic alignment, diversity, and controllability than one-shot baselines, producing molecules that more faithfully reflect complex, compositional prompts while offering transparent insight into the generation process.

---

## 133. Lacking Data? No worries! How synthetic images can alleviate image scarcity in wildlife surveys: a case study with muskox (Ovibos moschatus)

**论文链接:** [http://arxiv.org/abs/2511.11882v1](http://arxiv.org/abs/2511.11882v1)

**作者:** Simon Durand, Samuel Foucher, Alexandre Delplanque, Joëlle Taillon, Jérôme Théau

**发布时间:** 2025-11-14

**备注:** 34 pages, 10 figures, submitted to Remote Sensing in Ecology and Conservation

### GPT解析

### 总结

这项研究探讨了使用合成图像(SI)来增强深度学习目标检测模型(ODM)对北极麝牛的检测能力。研究比较了基线模型与结合了不同比例合成图像的零样本(ZS)和少样本(FS)模型，发现在数据有限的情况下，合成图像可以显著提高模型性能，特别是在零样本设置中。

### 背景

准确的种群估计对野生动物管理至关重要，能够提供物种数量和分布的关键信息。在北极地区，传统的麝牛监测方法包括空中视觉计数和GNSS遥测跟踪，但这些方法资源密集且受后勤挑战限制。虽然遥感、人工智能和高分辨率航空影像为野生动物检测提供了有前景的替代方案，但深度学习目标检测模型的有效性常受限于小型数据集，这使得训练针对稀疏分布物种(如麝牛)的稳健模型具有挑战性。

### 目的

研究旨在调查整合合成图像(SI)以补充有限的训练数据，并提高在零样本(ZS)和少样本(FS)设置下对麝牛的检测能力。

### 方法

研究比较了仅使用真实图像训练的基线模型，与5个零样本模型和5个少样本模型，这些模型在训练集中逐渐增加更多合成图像的比例。对于零样本模型，训练集中不包括真实图像；对于少样本模型，则结合了真实图像和合成图像。

### 主要发现

对于零样本模型，添加合成图像提高了检测性能，随着合成图像数量增加，精确率、召回率和F1分数都提高，但当合成图像超过基线模型训练数据集的100%后，性能趋于平稳，表明收益递减。对于少样本模型，结合真实图像和合成图像相比仅使用真实图像获得了更好的召回率和略高的整体准确率，但这些改进在统计上不显著。

### 结论

研究结果表明，当数据稀缺时，合成图像具有训练准确目标检测模型的潜力，为野生动物监测提供了重要视角，使能够监测稀有或难以接近的物种并增加监测频率。这种方法可以在没有真实数据的情况下启动目标检测模型，并随着随时间获取的真实图像进行改进。

### 翻译

准确的种群估计对野生动物管理至关重要，提供了关于物种数量和分布的关键见解。传统的调查方法，包括空中视觉计数和GNSS遥测跟踪，被广泛用于监测北极地区的麝牛种群。这些方法资源密集且受后勤挑战限制。遥感、人工智能和高分辨率航空影像的进步为野生动物检测提供了有前景的替代方案。然而，深度学习目标检测模型(ODM)的有效性通常受限于小型数据集，这使得针对稀疏分布物种(如麝牛)训练稳健的ODM具有挑战性。本研究调查了整合合成图像(SI)以补充有限的训练数据，并提高在零样本(ZS)和少样本(FS)设置下对麝牛的检测能力。我们将仅使用真实图像训练的基线模型与5个零样本模型和5个少样本模型进行了比较，这些模型在训练集中逐渐增加了更多SI。对于零样本模型(训练集中不包括真实图像)，添加SI提高了检测性能。随着更多SI的添加，精确率、召回率和F1分数都提高，但最终趋于平稳，表明当SI超过基线模型训练数据集的100%时，收益递减。对于少样本模型，结合真实图像和SI相比仅使用真实图像获得了更好的召回率和略高的整体准确率，但这些改进在统计上不显著。我们的研究结果证明了当数据稀缺时，SI训练准确ODM的潜力，为野生动物监测提供了重要视角，使能够监测稀有或难以接近的物种并增加监测频率。这种方法可用于在没有真实数据的情况下启动ODM，并随着随时间获取的真实图像进行改进。


### 论文摘要

Accurate population estimates are essential for wildlife management, providing critical insights into species abundance and distribution. Traditional survey methods, including visual aerial counts and GNSS telemetry tracking, are widely used to monitor muskox populations in Arctic regions. These approaches are resource intensive and constrained by logistical challenges. Advances in remote sensing, artificial intelligence, and high resolution aerial imagery offer promising alternatives for wildlife detection. Yet, the effectiveness of deep learning object detection models (ODMs) is often limited by small datasets, making it challenging to train robust ODMs for sparsely distributed species like muskoxen. This study investigates the integration of synthetic imagery (SI) to supplement limited training data and improve muskox detection in zero shot (ZS) and few-shot (FS) settings. We compared a baseline model trained on real imagery with 5 ZS and 5 FS models that incorporated progressively more SI in the training set. For the ZS models, where no real images were included in the training set, adding SI improved detection performance. As more SI were added, performance in precision, recall and F1 score increased, but eventually plateaued, suggesting diminishing returns when SI exceeded 100% of the baseline model training dataset. For FS models, combining real and SI led to better recall and slightly higher overall accuracy compared to using real images alone, though these improvements were not statistically significant. Our findings demonstrate the potential of SI to train accurate ODMs when data is scarce, offering important perspectives for wildlife monitoring by enabling rare or inaccessible species to be monitored and to increase monitoring frequency. This approach could be used to initiate ODMs without real data and refine it as real images are acquired over time.

---

## 134. Towards Mitigating Systematics in Large-Scale Surveys via Few-Shot Optimal Transport-Based Feature Alignment

**论文链接:** [http://arxiv.org/abs/2511.11787v1](http://arxiv.org/abs/2511.11787v1)

**作者:** Sultan Hassan, Sambatra Andrianomena, Benjamin D. Wandelt

**发布时间:** 2025-11-14

**备注:** 5 pages, 3 figures, accepted to NeurIPS Workshop on Unifying Representations in Neural Models (UniReps 2025)

### GPT解析

### 总结

该研究提出了一种新方法来解决系统误差导致的观测结果与理论模拟信号之间的分布偏移问题，通过优化特征对齐损失来对齐分布内和分布外样本的学习特征。

### 背景

系统误差会污染观测结果，导致与理论模拟信号产生分布偏移，这给使用预训练模型标记此类观测结果带来了重大挑战。系统误差通常难以理解和建模，因此直接完全消除它们可能不可行。

### 目的

提出一种新方法，使预训练模型能够更好地处理分布外的数据，解决系统误差导致的分布偏移问题。

### 方法

通过优化从预训练的分布内模型中提取的表示上的特征对齐损失，来对齐分布内和分布外样本之间的学习特征。首先在MNIST数据集上实验验证该方法，测试了不同的对齐损失，包括均方误差和最优传输。然后将该方法应用于大规模中性氢图。

### 主要发现

当分布内和分布外样本之间的对应关系未知时，最优传输在对齐分布外特征方面特别有效，即使在数据有限的情况下也是如此，这模拟了从大规模调查中提取信息的真实世界条件。

### 结论

所提出的方法能够有效处理系统误差导致的分布偏移问题，特别是在对应关系未知和数据有限的情况下。

### 翻译

系统误差污染观测结果，导致与理论模拟信号产生分布偏移，这给使用预训练模型标记此类观测结果带来了重大挑战。由于系统误差通常难以理解和建模，直接完全消除它们可能不可行。为应对这一挑战，我们提出了一种新方法，通过优化从预训练的分布内模型中提取的表示上的特征对齐损失，来对齐分布内和分布外样本之间的学习特征。我们首先在MNIST数据集上使用可能的对齐损失（包括均方误差和最优传输）实验验证该方法，随后将其应用于大规模中性氢图。我们的结果表明，当分布内和分布外样本之间的对应关系未知时，最优传输在对齐分布外特征方面特别有效，即使在数据有限的情况下也是如此，这模拟了从大规模调查中提取信息的真实世界条件。我们的代码可在https://github.com/sultan-hassan/feature-alignment-for-OOD-generalization获取。


### 论文摘要

Systematics contaminate observables, leading to distribution shifts relative to theoretically simulated signals-posing a major challenge for using pre-trained models to label such observables. Since systematics are often poorly understood and difficult to model, removing them directly and entirely may not be feasible. To address this challenge, we propose a novel method that aligns learned features between in-distribution (ID) and out-of-distribution (OOD) samples by optimizing a feature-alignment loss on the representations extracted from a pre-trained ID model. We first experimentally validate the method on the MNIST dataset using possible alignment losses, including mean squared error and optimal transport, and subsequently apply it to large-scale maps of neutral hydrogen. Our results show that optimal transport is particularly effective at aligning OOD features when parity between ID and OOD samples is unknown, even with limited data-mimicking real-world conditions in extracting information from large-scale surveys. Our code is available at https://github.com/sultan-hassan/feature-alignment-for-OOD-generalization.

---

## 135. Efficient Reinforcement Learning for Zero-Shot Coordination in Evolving Games

**论文链接:** [http://arxiv.org/abs/2511.11083v2](http://arxiv.org/abs/2511.11083v2)

**作者:** Bingyu Hui, Lebin Yu, Quanming Yao, Yunpeng Qu, Xudong Zhang, Jian Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了可扩展群体训练(ScaPT)框架，解决了现有零样本协调方法受计算资源限制的问题，通过元代理和互信息正则化器实现了高效且多样化的群体训练。

### 背景

零样本协调(ZSC)是强化学习研究的热点，关注代理与未见合作者无需微调的良好协调能力。基于群体的训练虽能提供良好的零样本协调性能，但受计算资源限制。

### 目的

解决现有零样本协调方法受计算资源限制的问题，探索扩大群体规模带来的性能提升可能。

### 方法

提出可扩展群体训练(ScaPT)框架，包含两个关键组件：1)元代理，通过选择性共享参数高效实现群体；2)互信息正则化器，确保群体多样性。

### 主要发现

在Hanabi游戏中的评估表明，ScaPT框架优于现有的代表性框架，证明了其有效性。

### 结论

ScaPT是一种有效的训练框架，能够解决现有零样本协调方法的局限性，为大规模群体训练提供了新思路。

### 翻译

零样本协调已成为强化学习研究的热点话题。它关注代理的泛化能力，要求它们能够与未见过的合作者良好协调，无需任何微调。基于群体的训练已被证明能提供良好的零样本协调性能；然而，现有方法受计算资源限制，主要关注小规模群体的多样性优化，而忽略了扩大群体规模可能带来的性能提升。为解决这一问题，本文提出了可扩展群体训练(ScaPT)，一个包含两个关键组件的高效训练框架：一个通过选择性共享参数高效实现群体的元代理，以及一个确保群体多样性的互信息正则化器。为经验性地验证ScaPT的有效性，本文在Hanabi游戏中对其与代表性框架进行了评估，并确认了其优越性。


### 论文摘要

Zero-shot coordination(ZSC) has become a hot topic in reinforcement learning research recently. It focuses on the generalization ability of agents, requiring them to coordinate well with collaborators that are not seen before without any fine-tuning. Population-based training has been proven to provide good zero-shot coordination performance; nevertheless, existing methods are limited by computational resources, mainly focusing on optimizing diversity in small populations while neglecting the potential performance gains from scaling population size. To address this issue, this paper proposes the Scalable Population Training (ScaPT), an efficient training framework comprising two key components: a meta-agent that efficiently realizes a population by selectively sharing parameters across agents, and a mutual information regularizer that guarantees population diversity. To empirically validate the effectiveness of ScaPT, this paper evaluates it along with representational frameworks in Hanabi and confirms its superiority.

---

## 136. Learning to Refine: An Agentic RL Approach for Iterative SPARQL Query Construction

**论文链接:** [http://arxiv.org/abs/2511.11770v1](http://arxiv.org/abs/2511.11770v1)

**作者:** Floris Vossebeld, Shenghui Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种新颖的智能体框架，通过强化学习训练大型语言模型学习迭代构建SPARQL查询的弹性策略，显著提高了知识图谱问答中多跳问题的查询准确率。

### 背景

为多跳问题生成复杂且逻辑一致的SPARQL查询是知识图谱问答的关键瓶颈，因为大型语言模型的一次性生成性质使其难以与结构化数据进行可靠交互。当前方法缺乏基于实时执行反馈动态调试查询的自适应策略。

### 目的

开发一种能够从执行错误中系统恢复并完善查询以获得正确答案的方法，提高知识图谱问答中SPARQL查询的准确性和可靠性。

### 方法

引入一个新颖的智能体框架，其中大型语言模型学习迭代SPARQL构建的顺序过程的弹性策略。仅通过结果驱动的强化学习(GRPO)训练一个3B参数模型，无需监督微调，使其学习如何从执行错误中恢复并完善查询。

### 主要发现

在LC-QuAD 2.0可执行单答案子集上，该智能体在实体链接后达到49.7%的准确率，比最强的迭代零样本基线显著提高了17.5个百分点。虽然智能体的能力是由强化学习驱动的，但其性能通过明确的推理步骤得到了增强，该步骤充当认知支架以提高策略精度。

### 结论

这项工作为教授智能体通过交互掌握形式化符号工具提供了可推广的蓝图，弥合了概率大型语言模型与知识图谱结构化世界之间的差距。

### 翻译

为多跳问题生成复杂且逻辑一致的SPARQL查询仍然是知识图谱问答的关键瓶颈，因为大型语言模型(LLMs)的一次性生成特性阻碍了与结构化数据的可靠交互。当前方法缺乏基于实时执行反馈动态调试查询的自适应策略。本文引入了一种新颖的智能体框架，其中大型语言模型学习迭代SPARQL构建过程的弹性策略。我们表明，仅通过结果驱动的强化学习(GRPO)训练的紧凑3B参数模型，无需监督微调，可以学习此任务的有效策略，发现如何从执行错误中系统恢复并完善查询以获得正确答案。在精心筛选的LC-QuAD 2.0可执行单答案子集上，我们的智能体在实体链接后达到49.7%的准确率，比最强的迭代零样本基线显著提高了17.5个百分点。进一步分析表明，虽然智能体的能力是由强化学习驱动的，但其性能通过明确的推理步骤得到了增强，该步骤充当认知支架以提高策略精度。这项工作为教授智能体通过交互掌握形式化符号工具提供了可推广的蓝图，弥合了概率大型语言模型与知识图谱结构化世界之间的差距。


### 论文摘要

Generating complex, logically-sound SPARQL queries for multi-hop questions remains a critical bottleneck for Knowledge Graph Question Answering, as the brittle nature of one-shot generation by Large Language Models (LLMs) hinders reliable interaction with structured data. Current methods lack the adaptive policies needed to dynamically debug queries based on real-time execution feedback. This paper introduces a novel agentic framework where an LLM learns a resilient policy for the sequential process of iterative SPARQL construction. We show that a compact 3B-parameter model, trained exclusively via outcome-driven Reinforcement Learning (GRPO) without supervised fine-tuning, can learn effective policies for this task, discovering how to systematically recover from execution errors and refine its queries toward a correct answer. On a curated, executable single-answer subset of LC-QuAD 2.0, our agent achieves 49.7\% accuracy post-entity-linking, a significant 17.5 percentage point improvement over the strongest iterative zero-shot baseline. Further analysis reveals that while the agent's capability is driven by RL, its performance is enhanced by an explicit deliberative reasoning step that acts as a cognitive scaffold to improve policy precision. This work presents a generalizable blueprint for teaching agents to master formal, symbolic tools through interaction, bridging the gap between probabilistic LLMs and the structured world of Knowledge Graphs.

---

## 137. Autonomous Sensing UAV for Accurate Multi-User Identification and Localization in Cellular Networks

**论文链接:** [http://arxiv.org/abs/2511.13171v1](http://arxiv.org/abs/2511.13171v1)

**作者:** Niccolò Paglierani, Francesco Linsalata, Vineeth Teeda, Davide Scazzoli, Maurizio Magarini

**发布时间:** 2025-11-17

### GPT解析

### 总结

论文提出了一种用于5G网络中用户识别和定位的自主感知框架，使用不作为服务接入网络一部分的被动感知无人机。该无人机捕获上行链路探测参考信号，几乎不需要与网络基础设施协调。开发了完整的信号处理链，在无人机上执行同步、用户识别和定位。系统自主规划和调整任务工作流程，集成飞行控制与实时感知。通过仿真和实验验证，在农村实地测试中定位误差低于3米，在城市仿真场景中低于8米，可靠识别每个用户。

### 背景

第五代（5G）网络中需要识别和定位多个用户，传统方法通常依赖于网络基础设施。与传统的空中服务节点不同，本研究提出使用不作为服务接入网络一部分的无人机进行感知。

### 目的

开发一个自主感知框架，用于识别和定位5G网络中的多个用户，创建一种不依赖网络基础设施的感知无人机系统。

### 方法

提出并开发了完整的信号处理链，包括同步、用户识别和定位，所有处理都在无人机飞行过程中机上执行。系统自主规划和调整任务工作流程，以估计单个部署中的多个用户位置，并将飞行控制与实时感知集成。

### 主要发现

无人机被动运行，仅用于感知，捕获上行链路探测参考信号，几乎不需要与网络基础设施协调。在农村实地测试中，定位误差低于3米；在城市仿真场景中，定位误差低于8米；系统能可靠识别每个用户。

### 结论

不依赖基础设施的感知无人机是新兴低空经济（LAE）的核心要素，支持态势感知和紧急情况或连接有限环境中的快速部署。

### 翻译

本文提出了一种自主感知框架，用于在第五代（5G）网络中识别和定位多个用户，使用的是不属于服务接入网络一部分的无人机。与传统空中服务节点不同，所提出的无人机被动运行，专门用于感知。它捕获上行链路（UL）探测参考信号，几乎不需要与网络基础设施协调。提出了并开发了一个完整的信号处理链，包括同步、用户识别和定位，所有这些都在无人机飞行过程中机上执行。系统自主规划和调整其任务工作流程，以估计单个部署中的多个用户位置，将飞行控制与实时感知集成。广泛的仿真和全规模低空实验活动验证了该方法，显示在农村实地测试中定位误差低于3米，在城市仿真场景中低于8米，同时可靠地识别每个用户。这些结果确认了不依赖基础设施的感知无人机作为新兴低空经济（LAE）核心要素的可行性，支持态势感知和紧急情况或连接有限环境中的快速部署。


### 论文摘要

This paper presents an autonomous sensing frame- work for identifying and localizing multiple users in Fifth Generation (5G) networks using an Unmanned Aerial Vehicle (UAV) that is not part of the serving access network. Unlike conventional aerial serving nodes, the proposed UAV operates passively and is dedicated solely to sensing. It captures Uplink (UL) Sounding Reference Signals (SRS), and requires virtually no coordination with the network infrastructure. A complete signal processing chain is proposed and developed, encompassing synchronization, user identification, and localization, all executed onboard UAV during flight. The system autonomously plans and adapts its mission workflow to estimate multiple user positions within a single deployment, integrating flight control with real-time sensing. Extensive simulations and a full-scale low- altitude experimental campaign validate the approach, showing localization errors below 3 m in rural field tests and below 8 m in urban simulation scenarios, while reliably identifying each user. The results confirm the feasibility of infrastructure-independent sensing UAVs as a core element of the emerging Low Altitude Economy (LAE), supporting situational awareness and rapid deployment in emergency or connectivity-limited environments.

---

## 138. Designing-with More-than-Human Through Human Augmentation

**论文链接:** [http://arxiv.org/abs/2511.12533v1](http://arxiv.org/abs/2511.12533v1)

**作者:** Botao 'Amber' Hu, Danlin Huang

**发布时间:** 2025-11-16

**备注:** Submitted to DRS 2026

### GPT解析

### 总结

论文提出了一种名为'超人类通过人类增强'(MtHtHA)的设计方法，利用人类增强技术创造临时身体体验，重新调节人类感知，培养对非人类生物和生态系统的生态现象学意识，促进跨物种的生态意识、同理心和关怀义务。

### 背景

近期设计领域出现'超人类转向'，呼吁与其他物种和生态系统共同设计。然而，托马斯·内格尔的著名思想实验'做一只蝙蝠是什么感觉？'表明人类经验受限于自身感官系统，与非人类生活世界之间存在现象学上的不可逾越鸿沟。

### 目的

提出一种设计方法，将通常用于增强人类能力的人类增强技术重新定向，从人类优化和例外主义转向生态现象学意识，弥合人类与非人类生物之间的认知鸿沟。

### 方法

基于身体美学设计和生态身体学，MtHtHA创造临身的、暂时的体验，调节人类的环境世界(Umwelt)，使人类重新对多元宇宙的超人类感知变得敏感。研究包括七项设计原则和五个设计案例：EchoVision(蝙蝠式回声定位)、FeltSight(星鼻鼹触觉导航)、FungiSync(真菌网络调谐)、TentacUs(章鱼式分布式代理)和City of Sparkles(AI视角的城市数据)。

### 主要发现

通过体验式的'与非人类共同设计'，可以培养跨物种边界的生态意识、同理心和关怀义务。

### 结论

通过人类增强技术创造临时身体体验，可以重新调节人类感知，使我们能够更好地理解和体验非人类生物的世界观，从而促进生态意识和跨物种关怀。

### 翻译

近期设计领域的'超人类转向'呼吁与其他物种和生态系统共同设计。然而，正如托马斯·内格尔著名的'做一只蝙蝠是什么感觉？'思想实验所强调的，人类经验受限于我们自己的感官系统，并且与非人类生活世界之间存在现象学上的不可逾越的鸿沟。本文提出'通过人类增强的超人类'(MtHtHA，表示为'>HtH+')作为一种设计方法，重新利用通常用于增强人类能力的人类增强技术，但将其从人类优化和例外主义转向生态现象学意识。基于身体美学设计和生态身体学，MtHtHA涉及创造临身的、暂时的体验，调节人类的Umwelt，使我们重新对多元宇宙的超人类感知变得敏感。我们阐述了七项设计原则，并报告了五个设计案例：EchoVision(类似蝙蝠的回声定位)、FeltSight(星鼻鼹触觉导航)、FungiSync(真菌网络调谐)、TentacUs(类似章鱼的分布式代理)和City of Sparkles(从AI视角看的城市数据)。我们证明，这种体验式的'与非人类共同设计'可以培养跨物种边界的生态意识、同理心和关怀义务。


### 论文摘要

The recent more-than-human turn in design calls for "designing-with" other species and ecologies beyond humans. Yet-as Thomas Nagel's famous "What is it like to be a bat?" thought experiment highlights-human experience is constrained by our own sensorium and an irreducible gap in phenomenal access to nonhuman lifeworlds. This paper proposes More-than-Human through Human Augmentation (MtHtHA, denoted ">HtH+") as a design approach that repurposes human augmentation technologies-typically aimed at enhancing human capabilities-away from human optimization and exceptionalism but toward eco-phenomenological awareness. Grounded in somaesthetic design and eco-somatics, MtHtHA entails creating temporary, embodied experiences that modulate the human Umwelt to re-sensitize us to pluriversal more-than-human perceptions. We articulate seven design principles and report five design cases-EchoVision (bat-like echolocation), FeltSight (star-nosed-mole tactile navigation), FungiSync (fungal network attunement), TentacUs (octopus-like distributed agency), and City of Sparkles (urban data from AI's perspective). We demonstrate that such experiential "designing-with" can cultivate ecological awareness, empathy and obligations of care across species boundaries.

---

## 139. ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts

**论文链接:** [http://arxiv.org/abs/2511.11740v1](http://arxiv.org/abs/2511.11740v1)

**作者:** Haowen Jiang, Xinyu Huang, You Lu, Dingji Wang, Yuheng Cao, Chaofeng Sha, Bihuan Chen, Keyu Chen, Xin Peng

**发布时间:** 2025-11-13

**备注:** The paper has been accepted by the Fortieth AAAI Conference on Artificial Intelligence. AAAI 2026

### GPT解析

### 总结

这篇论文提出了ExpertAD框架，通过Mixture of Experts架构提高自动驾驶系统性能，解决语义模糊、任务干扰和推理延迟等问题。

### 背景

端到端自动驾驶系统在感知和规划方面有潜力，但面临复杂场景中语义信息模糊、多任务干扰导致规划不优、推理延迟增加不安全风险等挑战。

### 目的

提出ExpertAD框架，利用Mixture of Experts架构提高自动驾驶系统性能。

### 方法

引入Perception Adapter增强任务关键特征，确保场景理解相关性；使用Mixture of Sparse Experts最小化预测过程中的任务干扰，实现有效高效的规划。

### 主要发现

ExpertAD将平均碰撞率降低20%，推理延迟减少25%；在罕见场景中展示多技能规划能力；对未见城市环境有强泛化能力；案例研究展示了复杂场景中的决策过程。

### 结论

ExpertAD框架通过创新的架构设计有效解决了自动驾驶系统中的关键挑战，提高了系统的安全性和效率。

### 翻译

最近端到端自动驾驶系统的进展强调了其在感知和规划能力方面的潜力。然而，挑战仍然存在。复杂的驾驶场景包含丰富的语义信息，但模糊或有噪声的语义可能会损害决策可靠性，而多个驾驶任务之间的干扰可能会阻碍最优规划。此外，长时间的推理延迟会减慢决策过程，增加不安全驾驶行为的风险。为应对这些挑战，我们提出了ExpertAD，这是一个利用Mixture of Experts架构增强自动驾驶系统性能的新框架。我们引入了Perception Adapter来增强任务关键特征，确保上下文相关的场景理解，并使用Mixture of Sparse Experts来最小化预测过程中的任务干扰，从而实现有效和高效的规划。我们的实验表明，与先前的方法相比，ExpertAD将平均碰撞率降低了20%，推理延迟减少了25%。我们进一步评估了其在罕见场景(如事故、为紧急车辆让路)中的多技能规划能力，并展示了对未见城市环境的强大泛化能力。此外，我们提供了一个案例研究，说明其在复杂驾驶场景中的决策过程。


### 论文摘要

Recent advancements in end-to-end autonomous driving systems (ADSs) underscore their potential for perception and planning capabilities. However, challenges remain. Complex driving scenarios contain rich semantic information, yet ambiguous or noisy semantics can compromise decision reliability, while interference between multiple driving tasks may hinder optimal planning. Furthermore, prolonged inference latency slows decision-making, increasing the risk of unsafe driving behaviors. To address these challenges, we propose ExpertAD, a novel framework that enhances the performance of ADS with Mixture of Experts (MoE) architecture. We introduce a Perception Adapter (PA) to amplify task-critical features, ensuring contextually relevant scene understanding, and a Mixture of Sparse Experts (MoSE) to minimize task interference during prediction, allowing for effective and efficient planning. Our experiments show that ExpertAD reduces average collision rates by up to 20% and inference latency by 25% compared to prior methods. We further evaluate its multi-skill planning capabilities in rare scenarios (e.g., accidents, yielding to emergency vehicles) and demonstrate strong generalization to unseen urban environments. Additionally, we present a case study that illustrates its decision-making process in complex driving scenarios.

---

## 140. From Black Box to Insight: Explainable AI for Extreme Event Preparedness

**论文链接:** [http://arxiv.org/abs/2511.13712v1](http://arxiv.org/abs/2511.13712v1)

**作者:** Kiana Vu, İsmet Selçuk Özer, Phung Lai, Zheng Wu, Thilanka Munasinghe, Jennifer Wei

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究探讨了可解释人工智能(XAI)在极端事件预测中的作用，以野火预测为例，评估了多种AI模型并使用SHAP方法解释模型行为，展示了XAI如何提高模型的可解释性并支持决策制定。

### 背景

气候变化加速了极端事件如野火的频率和严重性，对准确、可解释和可操作的预测需求日益迫切。尽管AI模型在预测此类事件方面显示出潜力，但由于其'黑盒'性质，在现实决策中的采用仍然有限。

### 目的

研究可解释AI(XAI)在弥合预测准确性和可操作洞察力之间差距方面的作用，以提高AI模型在极端事件预测中的可用性和可信度。

### 方法

以野火预测为案例研究，评估各种AI模型，并使用SHapley Additive exPlanations(SHAP)来揭示模型行为中的关键特征、决策途径和潜在偏见。提供支持性可视化来增强XAI输出的可解释性。

### 主要发现

XAI不仅阐明了模型推理过程，还支持领域专家和响应团队的关键决策。通过上下文化特征重要性和季节性及地理空间特征中的时间模式，XAI输出的可解释性得到增强，提高了AI解释对从业者和政策制定者的可用性。

### 结论

需要开发不仅准确而且可解释、易于访问和值得信赖的AI系统，这对于在灾害准备、风险缓解和气候韧性规划中的有效使用至关重要。

### 翻译

随着气候变化加速了野火等极端事件的频率和严重性，准确、可解释和可操作的预测需求变得越来越紧迫。虽然人工智能(AI)模型在预测此类事件方面显示出前景，但由于其黑盒性质，它们在现实决策中的采用仍然有限，这限制了信任度、可解释性和操作准备度。本文研究了可解释AI(XAI)在弥合预测准确性和可操作洞察力之间差距方面的作用。以野火预测为案例研究，我们评估了各种AI模型，并使用SHapley Additive exPlanations(SHAP)来揭示模型行为中的关键特征、决策途径和潜在偏见。我们的分析表明，XAI不仅阐明了模型推理过程，还通过领域专家和响应团队支持关键决策。此外，我们提供支持性可视化，通过上下文化特征重要性和季节性及地理空间特征中的时间模式来增强XAI输出的可解释性。这种方法提高了从业者和政策制定者对AI解释的可用性。我们的研究结果表明，需要开发不仅准确而且可解释、易于访问和值得信赖的AI系统，这对于在灾害准备、风险缓解和气候韧性规划中的有效使用至关重要。


### 论文摘要

As climate change accelerates the frequency and severity of extreme events such as wildfires, the need for accurate, explainable, and actionable forecasting becomes increasingly urgent. While artificial intelligence (AI) models have shown promise in predicting such events, their adoption in real-world decision-making remains limited due to their black-box nature, which limits trust, explainability, and operational readiness. This paper investigates the role of explainable AI (XAI) in bridging the gap between predictive accuracy and actionable insight for extreme event forecasting. Using wildfire prediction as a case study, we evaluate various AI models and employ SHapley Additive exPlanations (SHAP) to uncover key features, decision pathways, and potential biases in model behavior. Our analysis demonstrates how XAI not only clarifies model reasoning but also supports critical decision-making by domain experts and response teams. In addition, we provide supporting visualizations that enhance the interpretability of XAI outputs by contextualizing feature importance and temporal patterns in seasonality and geospatial characteristics. This approach enhances the usability of AI explanations for practitioners and policymakers. Our findings highlight the need for AI systems that are not only accurate but also interpretable, accessible, and trustworthy, essential for effective use in disaster preparedness, risk mitigation, and climate resilience planning.

---

## 141. Integrative Model for Interoception and Exteroception: predictive coding, points of modulation, and testable predictions

**论文链接:** [http://arxiv.org/abs/2511.13668v1](http://arxiv.org/abs/2511.13668v1)

**作者:** Pranjal Balar, Sundeep Kapila

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文开发并验证了一个整合的预测编码模型，该模型将内感受和外感受视为并行分层系统，通过精度加权的预测误差进行交流，并由前脑岛和前扣带皮层进行仲裁控制。

### 背景

内感受和外感受提供关于身体和环境的连续反馈，但它们如何在统一的预测编码框架内动态整合尚未明确。

### 目的

开发并经验验证一个整合的预测编码模型，将内感受和外感受推理视为并行分层系统，交换精度加权的预测误差。

### 方法

构建整合预测编码模型，其中两个感觉流之间的仲裁由相对精度权重控制并在特定脑区内整合；进行计算模拟验证模型的生物学合理性；将仲裁方程应用于EEG-fMRI数据集进行实证验证。

### 主要发现

模型模拟显示预测误差呈指数衰减，仲裁权重自我标准化至平衡状态(w=0.5)，表现出稳定收敛和整合；模拟的焦虑和PTSD特征导致刚性不平衡和再校准减缓；应用仲裁方程于EEG-fMRI数据进一步验证了模型。

### 结论

该框架提供了统一的解释，说明失调的精度加权可能是焦虑(内感受过度加权)和PTSD(内感受加权不足)的基础；提出了实验范式测试模型在人类中的预测。

### 翻译

内感受和外感受提供了关于身体和环境的连续反馈，然而它们如何在统一的预测编码框架内动态整合仍然不够明确。本文开发并经验验证了一个整合的预测编码模型，该模型将内感受和外感受推理视为并行分层系统，交换精度加权的预测误差。在此框架内，两个流之间的仲裁由相对精度权重(w)控制，并在前脑岛(AIC)和前扣带皮层(ACC)内整合。模型的计算模拟重现了生物学上合理的动态：预测误差呈指数衰减，而仲裁权重自我标准化为平衡状态(w=0.5)，显示出稳定的收敛和整合。模拟的焦虑和PTSD特征(分别表现为内感受和外感受的过度加权)导致刚性、自我维持的不平衡(w趋近1或0)和再校准减缓。将仲裁方程应用于已发表的EEG-fMRI数据集进一步验证了该模型。该框架提供了一个统一的解释，说明失调的精度加权如何成为焦虑(内感受过度加权)和PTSD(内感受加权不足)的基础。基于此验证，提出了一个实验范式来测试人类中模型的预测，考察在焦虑、中性和PTSD组中针对内感受或外感受治疗后的再校准情况。关键预测包括可识别的神经 coherence 标记、迷走神经刺激对心跳诱发电位的调节，以及在内感受-外感受一致性任务中的精度敏感性行为特征。


### 论文摘要

Interoception and exteroception provide continuous feedback about the body and the environment, yet how they are dynamically integrated within a unified predictive coding framework has remained under-specified. This paper develops and empirically validates an integrative predictive coding model that treats interoceptive and exteroceptive inference as parallel hierarchical systems exchanging precision-weighted prediction errors. Within this framework, arbitration between the two streams is governed by relative precision weights (w) and integrated within the anterior insula (AIC) and anterior cingulate cortex (ACC). Computational simulations of the model reproduced biologically plausible dynamics: prediction errors decayed exponentially while arbitration weights self-normalized toward equilibrium (w = 0.5), demonstrating stable convergence and coherent integration. Simulated anxiety and PTSD profiles, characterized respectively by interoceptive and exteroceptive overweighting, yielded rigid, self-sustaining imbalances (w to 1 or w to 0) and slowed recalibration. Empirical application of the arbitration equation to published EEG-fMRI datasets further validated the model. The framework contributes a unifying account of how dysregulated precision weighting may underlie anxiety (overweighted interoception) and PTSD (underweighted interoception). Building on this validation, a proposed experimental paradigm is outlined to test the model's predictions in humans. It examines recalibration across anxiety, neutral, and PTSD groups following targeted interoceptive or exteroceptive therapies. Key predictions include identifiable neural markers of coherence, modulation of heartbeat-evoked potentials by vagal stimulation, and precision-sensitive behavioral signatures in interoceptive-exteroceptive congruency tasks.

---

## 142. Collective decision-making with higher-order interactions on $d$-uniform hypergraphs

**论文链接:** [http://arxiv.org/abs/2511.13452v1](http://arxiv.org/abs/2511.13452v1)

**作者:** Thierry Njougouo, Timoteo Carletti, Elio Tuci

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文研究了群体交互如何影响意见动态，提出并研究了一种基于d-均匀超图的意见动态模型，其中个体通过基于群体的高阶结构而非简单的成对连接进行交互。研究确定了两个临界阈值，验证了预测，并发现大型交互群体可能导致系统采纳最差选项。

### 背景

理解群体交互如何影响意见动态是集体行为研究的基础。传统的意见动态模型通常基于简单的成对交互，而现实中的交互往往涉及群体。

### 目的

研究群体交互（高阶结构）对意见动态的影响，探索群体大小和信息损失如何影响系统达成共识的能力，并确定系统稳定性的临界条件。

### 方法

提出了一种基于d-均匀超图的意见动态模型，其中每个意见（A和B）由质量参数（Q_A和Q_B）表征。代理根据支持任一意见的加权比例和交互过程中的信息损失（用参数α表示）来更新意见。通过平均场模型的分岔分析确定临界阈值，并通过在随机和无标度超图上进行大量基于代理的模拟来验证分析预测。

### 主要发现

通过分岔分析确定了两个临界阈值α_crit^(1)和α_crit^(2)，它们定义了共识状态的稳定性区域；分析框架表明，分岔结构和临界阈值与高阶网络的基础拓扑结构无关，仅取决于参数d（交互群体的大小）和质量比；发现了一个非平凡效应：大型交互群体可能导致系统采纳最差选项。

### 结论

群体交互的大小和质量对意见动态有重要影响，大型交互群体可能导致系统采纳质量较差的意见，这一发现对理解集体决策机制具有重要意义。

### 翻译

理解群体交互如何影响意见动态是集体行为研究的基础。在这项工作中，我们提出并研究了一种基于d-均匀超图的意见动态模型，其中个体通过基于群体的高阶结构而非简单的成对连接进行交互。两种意见A和B中的每一种都由质量Q_A和Q_B表征，代理根据支持任一意见的加权比例和汇集误差α（交互过程中信息损失的代理）来更新其意见。通过平均场模型的分岔分析，我们确定了两个临界阈值α_crit^(1)和α_crit^(2)，它们定义了共识状态的稳定性区域。这些分析预测通过在随机和无标度超图上进行的大量基于代理的模拟得到验证。此外，分析框架表明，分岔结构和临界阈值与高阶网络的基础拓扑结构无关，仅取决于参数d（即交互群体的大小）和质量比。最后，我们揭示了一个非平凡效应：大型交互群体可能导致系统采纳最差选项。


### 论文摘要

Understanding how group interactions influence opinion dynamics is fundamental to the study of collective behavior. In this work, we propose and study a model of opinion dynamics on $d$-uniform hypergraphs, where individuals interact through group-based (higher-order) structures rather than simple pairwise connections. Each one of the two opinions $A$ and $B$ is characterized by a quality, $Q_A$ and $Q_B$, and agents update their opinions according to a general mechanism that takes into account the weighted fraction of agents supporting either opinion and the pooling error, $α$, a proxy for the information lost during the interaction. Through bifurcation analysis of the mean-field model, we identify two critical thresholds, $α_{\text{crit}}^{(1)}$ and $α_{\text{crit}}^{(2)}$, which delimit stability regimes for the consensus states. These analytical predictions are validated through extensive agent-based simulations on both random and scale-free hypergraphs. Moreover, the analytical framework demonstrates that the bifurcation structure and critical thresholds are independent of the underlying topology of the higher-order network, depending solely on the parameters $d$, i.e., the size of the interaction groups, and the quality ratio. Finally, we bring to the fore a nontrivial effect: the large sizes of the interaction groups, could drive the system toward the adoption of the worst option.

---

## 143. Capabilities and limitations of pure-shear based macroscopic forming simulations for 0{\textdegree}/90{\textdegree} biaxial non-crimp fabrics

**论文链接:** [http://arxiv.org/abs/2511.13388v1](http://arxiv.org/abs/2511.13388v1)

**作者:** Bastian Schäfer, Dominik Dörr, Naim Naouar, Jan Paul Wank, Luise Kärger

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文针对非织造布(NCF)的宏观建模挑战，提出了一种基于纯剪切假设的超弹性方法，用于描述0°/90°双向NCF的成型行为。该方法通过45°偏轴拉伸实验表征剪切行为，并从能量方法推导参数化。研究表明，该方法能有效预测纤维取向和高剪切角区域，但由于忽略纱线滑移，会高估峰值剪切角。

### 背景

非织造布(NCF)的宏观建模具有挑战性，因为其成型行为强烈依赖于纺织结构、纤维类型和缝合类型。与机织织物不同，NCF的膜建模不仅需要考虑剪切变形，还需要考虑缝合变形和纱线滑移。然而，对于对称缝合图案和高缝合预张力的0°/90°双向NCF，在试样测试中只有在较高剪切角度下才能观察到纯剪切假设的偏差。

### 目的

研究针对具有对称缝合图案和高缝合预张力的0°/90°双向NCF(Biax-NCF)的建模方法，基于纯剪切假设开发一种有效的超弹性方法，以准确预测其成型行为。

### 方法

采用最初为单向NCF提出的超弹性方法，基于纯剪切假设进行应用。通过45°偏轴拉伸实验表征剪切行为，并从最初为机织织物开发的能量方法推导参数化。利用这种方法对不同几何形状进行成型模拟。

### 主要发现

对于对称缝合图案和高缝合预张力的0°/90° Biax-NCF，在试样测试中只有在较高剪切角度下才能观察到纯剪切假设的偏差，这是由于纱线滑移有限造成的。所提出的方法能够有效且充分地描述不同几何形状成型模拟中的总体行为，纤维取向和高剪切角区域的定位预测良好，但由于忽略了纱线滑移，峰值剪切角被高估。

### 结论

基于纯剪切假设的超弹性方法能够有效描述Biax-NCF的成型行为，特别是在预测纤维取向和高剪切角区域方面表现良好。虽然峰值剪切角被高估，但该方法在工程应用中仍然具有实用价值。

### 翻译

非织造布(NCF)的宏观建模具有挑战性，因为其成型行为强烈依赖于纺织结构、纤维类型和缝合类型。虽然剪切是机织织物的主要变形模式，但NCF的膜建模方法还应考虑缝合变形和纱线滑移。然而，对于具有对称缝合图案和高缝合预张力的0°/90°双向NCF(Biax-NCF)，在试样测试中只有在较高剪切角度下才能观察到纯剪切假设的偏差，这是由于纱线滑移有限造成的。在本工作中，采用了一种最初为单向NCF提出的超弹性方法，基于纯剪切假设应用于经编缝合的0°/90° Biax-NCF。通过45°偏轴拉伸实验表征剪切行为，并从最初为机织织物开发的能量方法推导参数化。该方法能够有效且充分地描述不同几何形状成型模拟中的总体行为。纤维取向和高剪切角区域的定位预测良好，但由于忽略了纱线滑移，峰值剪切角被高估。


### 论文摘要

Macroscopic modeling of a non-crimp fabric's (NCF's) forming behavior is challenging as it strongly depends on the textile architecture, fiber type, and stitching type. While shear is the main deformation mode of woven fabrics, membrane modeling approaches for NCFs should also consider stitching deformation and roving slippage. However, for 0{\textdegree}/90{\textdegree} biaxial NCFs (Biax-NCF) with a symmetrical stitching pattern and high stitch pretension, deviations from a pure-shear assumption in coupon tests are only observed at higher shear angles due to limited roving slippage. In this work, a hyperelastic approach initially proposed for unidirectional NCFs is adopted for a tricot stitched 0{\textdegree}/90{\textdegree} Biax-NCF based on a pure-shear assumption. The shear behavior is experimentally characterized through 45{\textdegree} off-axis-tension tests, and the parameterization is derived from energetic approaches originally developed for woven fabrics. This approach efficiently and adequately describes the general behavior in forming simulations of different geometries. Fiber orientation and location of areas with high shear angles are predicted well, but the peak shear angles are overestimated due to the neglected roving slippage.

---

## 144. Coliseum project: Correlating climate change data with the behavior of heritage materials

**论文链接:** [http://arxiv.org/abs/2511.13343v1](http://arxiv.org/abs/2511.13343v1)

**作者:** A Cormier, David Roqui, Fabrice Surma, Martin Labouré, Jean-Marc Vallet, Odile Guillon, N Grozavu, Ann Bourgès

**发布时间:** 2025-11-17

### GPT解析

### 总结

COLISEUM项目旨在通过人工智能模型预测文化遗产材料在气候变化条件下的行为，通过在三个法国文化遗址建立气候监测系统，收集气候数据和劣化进展信息，使用风化指数建立预测模型，为文化遗产保护提供科学依据。

### 背景

文化遗产材料已经受到气候变化的影响，气候变化增加了文物的风化速度，缩短了其寿命。风化过程受多种因素影响，难以将其进展与气候变化直接关联。

### 目的

预测文化遗产材料的风化情况，通过收集气候数据同时监测劣化进展，利用人工智能计算机模型预测文化遗产材料在不同气候变化情景下的行为。

### 方法

在法国三个地点收集数据：斯特拉斯堡的圣母大教堂、Bibracte考古遗址和维尔夫朗什-梅尔圣皮埃尔小教堂。使用微气候传感器连续记录参数，通过化学分析、地图测量和科学成像活动定期监测劣化状态，使用计算的风化指数将数据收集在劣化矩阵中，建立风化模型。

### 主要发现

多模态数据（图像、文本等）使相关性分析变得困难，特别是在不同时间尺度上。通过预测模型，可以使用IPCC气候变化情景的已知数据预测未来材料行为。

### 结论

文章介绍了仪器化方法、初步诊断和第一个结果，以斯特拉斯堡大教堂为例，展示了该项目的实施方式和初步成果。

### 翻译

文化遗产材料已经受到气候变化的影响，日益增加的气候变异缩短了古迹的寿命。由于风化取决于许多因素，很难将其进展与气候变化联系起来。为了预测风化，必须在监测劣化进展的同时收集气候数据。收集数据的多模态特性（图像、文本等）使得相关性分析变得困难，特别是在不同时间尺度上。为了解决这个问题，COLISEUM项目提出了一种在法国三个地点收集数据的方法，使用人工智能计算机模型预测文化遗产材料的行为。随着时间的推移，预测模型将允许使用IPCC政府间气候变化专门委员会提供的不同气候变化情景的已知数据来预测未来的材料行为。因此，在法国三个文化遗址建立了气候监测方法：斯特拉斯堡的圣母大教堂（67）、Bibracte考古遗址（71）和维尔夫朗什-梅尔的圣皮埃尔小教堂（06）。每个遗址都有不同的气候和特定材料。现场微气候传感器连续记录随时间变化的参数。通过化学分析、地图测量和科学成像活动定期监测劣化状态。为了实施风化模型，通过计算的风化指数将数据收集在劣化矩阵中。本文介绍了仪器化方法、初步诊断和以斯特拉斯堡大教堂为例的第一个结果。


### 论文摘要

Heritage materials are already affected by climate change, and increasing climatic variations reduces the lifespan of monuments. As weathering depends on many factors, it is also difficult to link its progression to climatic changes. To predict weathering, it is essential to gather climatic data while simultaneously monitoring the progression of deterioration. The multimodal nature of collected data (images, text{\ldots}) makes correlations difficult, particularly on different time scales. To address this issue, the COLISEUM project proposes a methodology for collecting data in three French sites to predict heritage material behaviour using artificial intelligence computer models. Over time, prediction models will allow the prediction of future material behaviours using known data from different climate change scenarios by the IPCC (Intergovernmental Panel on Climate Change). Thus, a climate monitoring methodology has been set up in three cultural sites in France: Notre-Dame cathedral in Strasbourg ( 67), Bibracte archaeological site (71), and the Saint-Pierre chapel in Villefranche-sur-Mer (06). Each site has a different climate and specific materials. In situ, microclimatic sensors continuously record variations parameters over time. The state of alteration is monitored at regular intervals by means of chemical analyses, cartographic measurements and scientific imaging campaigns. To implement weathering models, data is gathered in alteration matrix by mean of a calculated weathering index. This article presents the instrumentation methodology, the initial diagnostic and the first results with the example of Strasbourg Cathedral site.

---

## 145. Beyond Energy Functions and Numerical Integration: A New Methodology to Determine Transient Stability at the Initial State

**论文链接:** [http://arxiv.org/abs/2511.13289v1](http://arxiv.org/abs/2511.13289v1)

**作者:** Wenhao Wu, Dan Wu, Bin Wang, Jiabing Hu

**发布时间:** 2025-11-17

**备注:** This work has been submitted to 2026 IEEE PES General Meeting

### GPT解析

### 总结

本文提出了一种新的暂态稳定性分析方法，避免了顺序数值积分和能量函数的局限性。

### 背景

传统的暂态稳定性分析方法存在顺序数值积分和能量函数使用的局限性。

### 目的

开发一种新的暂态稳定性分析方法，能够更直接、高效地预测系统的稳定性。

### 方法

构建轨迹依赖的稳定性指示函数，应用战略时间收缩映射，将TSA转化为极点位置检测问题，并利用初始状态的高阶导数推导有理函数近似。

### 主要发现

通过数值验证，该方法为电力系统TSA提供了直接的数学捷径，并为评估广泛非线性动力学系统的暂态稳定性建立了新方法。

### 结论

所提出的方法在数学上直接且计算上高效，为暂态稳定性分析提供了新的有前景的方法论。

### 翻译

本文提出了一种用于暂态稳定性分析的新方法，该方法绕过了顺序数值积分和能量函数的局限性。该方法首先构建一个轨迹依赖的稳定性指示函数来区分系统的命运。为了克服分析无限时间渐进行为的困难，随后应用了战略性的时间收缩映射。这使得TSA可以转化为指示函数的极点位置检测问题。通过利用初始状态的高阶导数，推导出一个有理函数近似，从而得到数学上直接且计算上高效的预测。在基准系统上的数值验证表明，该方法不仅为电力系统的TSA提供了直接的数学捷径，还为评估广泛非线性动力学系统的暂态稳定性建立了有前景的新方法论。


### 论文摘要

This paper presents a novel method for transient stability analysis (TSA) that circumvents the limitations of sequential numerical integration and energy functions. The proposed method begins by constructing a trajectory-dependent stability indicator function to distinguish the system's destiny. To overcome the difficulty in analyzing the asymptotic behavior at infinite time, a strategic time contraction mapping is then applied. This allows TSA to be recast as a pole-placement detection problem for the indicator function. By leveraging high-order derivatives at the initial state, a rational function approximation is derived, yielding a mathematically direct and computationally efficient prediction. Numerical validations on benchmark systems demonstrate that the method not only provides a direct mathematical shortcut for TSA in power systems but also establishes a promising new methodology for evaluating the transient stability of a broad class of nonlinear dynamical systems.

---

## 146. Incoherent Beliefs & Inconsistent Actions in Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.13240v1](http://arxiv.org/abs/2511.13240v1)

**作者:** Arka Pal, Teo Kitanovski, Arthur Liang, Akilesh Potti, Micah Goldblum

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究揭示了大型语言模型在信念更新和行动一致性方面的局限性，即使在表现良好的模型中也存在这些问题，这对预测它们在复杂真实世界环境中的行为提出了挑战

### 背景

真实世界的任务和环境与大型语言模型(LLMs)通常评估的静态数据集存在差异。真实任务可能涉及顺序交互，需要根据新证据一致地更新信念，并基于这些信念做出适当决策。从静态环境中的测量难以确定LLMs在动态环境中的表现。

### 目的

研究LLM性能的两个关键组成部分：LLMs一致更新信念的能力，以及它们采取的行动与这些信念的一致性程度；预测LLMs在动态环境中的表现。

### 方法

检查LLMs更新信念的一致性；评估LLMs采取的行动与持有信念的一致性；使用博彩市场等工具测试一致性；测试LLMs对用户质疑给定答案的回答的一致性；研究即使在获得高精度或任务校准良好的强模型中，上述特性是否仍然存在。

### 主要发现

LLMs在更新信念方面很大程度上不一致，模型在直接引出的后验与先验的正确更新之间的平均差异可达30%；LLMs经常采取与其持有信念不一致的行动，例如在博彩市场上往往不会对内部持有的信念进行相同方向的投注；LLMs在回应用户质疑时表现出中等程度的不一致性；即使在表现良好的强模型中，上述特性仍然存在。

### 结论

结果突显了预测LLMs在复杂真实世界环境中行为的困难。

### 翻译

现实世界的任务和环境与大型语言模型(LLMs)通常被评估的静态数据集存在差异。此类任务可能涉及顺序交互，需要根据新证据一致地更新信念，并基于这些信念做出适当的决策。预测LLMs在此类动态环境中的表现很重要，但很难从静态环境中的测量来确定。在这项工作中，我们研究了LLM性能的两个关键组成部分：LLMs一致更新信念的能力，以及它们采取的行动与这些信念一致的程度。首先，我们发现LLMs在更新信念方面很大程度上不一致；模型在直接引出的后验与其先验的正确更新之间可能表现出高达30%的平均差异。其次，我们发现LLMs也经常采取与其持有的信念不一致的行动。例如，在博彩市场上，LLMs往往不会对其内部持有的关于潜在结果的信念进行相同方向的投注。我们还发现，在用户对给定答案提出质疑时，它们在回应方式上存在中等程度的不一致性。最后，我们表明，即使在获得高精度或在当前任务上校准良好的强模型中，上述特性仍然存在。我们的结果突显了预测LLMs在复杂真实世界环境中行为的困难。


### 论文摘要

Real-world tasks and environments exhibit differences from the static datasets that large language models (LLMs) are typically evaluated on. Such tasks can involve sequential interaction, requiring coherent updating of beliefs in light of new evidence, and making appropriate decisions based on those beliefs. Predicting how LLMs will perform in such dynamic environments is important, but can be tricky to determine from measurements in static settings. In this work, we examine two critical components of LLM performance: the ability of LLMs to coherently update their beliefs, and the extent to which the actions they take are consistent with those beliefs. First, we find that LLMs are largely inconsistent in how they update their beliefs; models can exhibit up to a 30% average difference between the directly elicited posterior, and the correct update of their prior. Second, we find that LLMs also often take actions which are inconsistent with the beliefs they hold. On a betting market, for example, LLMs often do not even bet in the same direction as their internally held beliefs over the underlying outcomes. We also find they have moderate self-inconsistency in how they respond to challenges by users to given answers. Finally, we show that the above properties hold even for strong models that obtain high accuracy or that are well-calibrated on the tasks at hand. Our results highlight the difficulties of predicting LLM behavior in complex real-world settings.

---

## 147. MorphBoost: Self-Organizing Universal Gradient Boosting with Adaptive Tree Morphing

**论文链接:** [http://arxiv.org/abs/2511.13234v1](http://arxiv.org/abs/2511.13234v1)

**作者:** Boris Kriuk

**发布时间:** 2025-11-17

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

MorphBoost是一种新型梯度提升框架，通过自组织树结构和动态分裂行为，能够适应不同学习阶段的梯度分布变化，在多个基准测试中展现出优越性能。

### 背景

传统梯度提升算法采用静态树结构和固定分裂标准，在整个训练过程中保持不变，限制了它们适应不断变化的梯度分布和问题特定特征的能力。

### 目的

引入MorphBoost框架，实现树结构在训练过程中的动态变形，提高算法对不同学习阶段和问题复杂性的适应能力。

### 方法

实现自适应分裂函数，基于累积梯度统计和迭代相关学习压力进行演变，包括变形分裂标准、自动问题指纹识别、向量化树预测、交互感知特征重要性检测和快速模式优化等技术。

### 主要发现

在10个多样化数据集上测试，MorphBoost平均比XGBoost高出0.84%，获得40%的胜率和20%的前三率，保持最低方差(σ=0.0948)和最高最低准确率，在高级问题上表现尤为突出。

### 结论

MorphBoost通过动态调整树结构和分裂行为，能够更好地适应不同问题复杂性和学习阶段，在各种数据集上展现出优越的性能、一致性和鲁棒性。

### 翻译

传统梯度提升算法采用静态树结构和固定分裂标准，这些标准在整个训练过程中保持不变，限制了它们适应不断变化的梯度分布和不同学习阶段问题特定特征的能力。本文引入了MorphBoost，一种新型梯度提升框架，具有自组织树结构，在训练过程中动态改变其分裂行为。该算法实现了自适应分裂函数，这些函数基于累积的梯度统计和迭代相关的学习压力进行演变，能够自动调整问题复杂性。关键创新包括：(1)结合基于梯度的分数和信息论度量的变形分裂标准，由训练进度加权；(2)自动问题指纹识别，用于二元/多类/回归任务的智能参数配置；(3)向量化树预测实现显著的计算加速；(4)交互感知特征重要性检测乘法关系；(5)快速模式优化平衡速度和准确性。在10个多样化数据集上与竞争模型(XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, 集成方法)的全面基准测试表明，MorphBoost达到了最先进的性能，平均比XGBoost高出0.84%。MorphBoost获得了整体获胜者位置，在10个数据集中赢得4个(40%胜率)和30个顶级-3完成中的6个(20%)，同时保持所有模型中最低的方差(σ=0.0948)和最高的最低准确率，显示出优越的一致性和鲁棒性。跨难度级别的性能分析显示，在简单数据集上具有竞争性结果，而在高级问题上取得了显著改进，这是由于更高的适应水平。


### 论文摘要

Traditional gradient boosting algorithms employ static tree structures with fixed splitting criteria that remain unchanged throughout training, limiting their ability to adapt to evolving gradient distributions and problem-specific characteristics across different learning stages. This work introduces MorphBoost, a new gradient boosting framework featuring self-organizing tree structures that dynamically morph their splitting behavior during training. The algorithm implements adaptive split functions that evolve based on accumulated gradient statistics and iteration-dependent learning pressures, enabling automatic adjustment to problem complexity. Key innovations include: (1) morphing split criterion combining gradient-based scores with information-theoretic metrics weighted by training progress; (2) automatic problem fingerprinting for intelligent parameter configuration across binary/multiclass/regression tasks; (3) vectorized tree prediction achieving significant computational speedups; (4) interaction-aware feature importance detecting multiplicative relationships; and (5) fast-mode optimization balancing speed and accuracy. Comprehensive benchmarking across 10 diverse datasets against competitive models (XGBoost, LightGBM, GradientBoosting, HistGradientBoosting, ensemble methods) demonstrates that MorphBoost achieves state-of-the-art performance, outperforming XGBoost by 0.84% on average. MorphBoost secured the overall winner position with 4/10 dataset wins (40% win rate) and 6/30 top-3 finishes (20%), while maintaining the lowest variance (σ=0.0948) and highest minimum accuracy across all models, revealing superior consistency and robustness. Performance analysis across difficulty levels shows competitive results on easy datasets while achieving notable improvements on advanced problems due to higher adaptation levels.

---

## 148. Hydrodynamic Instability Induces Spontaneous Motion of Floating Ice Discs

**论文链接:** [http://arxiv.org/abs/2511.13184v1](http://arxiv.org/abs/2511.13184v1)

**作者:** Min Li, Lailai Zhu

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究通过数值模拟和标度分析揭示了实验室中冰盘自主运动的机制，发现运动由冰盘与水的温度差和水深决定，并由浮力驱动的羽流在对称性破缺后引发。

### 背景

自然界中的旋转冰盘现象已有一个多世纪的报道，但实验室实验产生了多样观察和矛盾解释，冰盘运动背后的机制一直不清楚。

### 目的

研究实验室规模水箱中自由移动冰盘的运动机制，统一独立实验中观察到的冰盘行为，并建立冰盘运动开始的预测标准。

### 方法

结合数值模拟和标度分析，在实验室规模水箱中观察冰盘在不同条件下的运动情况。

### 主要发现

冰盘保持静止或经历自发运动取决于冰盘与水的温度差和水深；运动由浮力驱动的向下羽流引发，该羽流源于水的密度异常（密度在约4°C时达到峰值）；羽流在下降超过临界距离后，由于热对流不稳定性打破旋转和镜像对称性，从而使冰盘自主移动。

### 结论

研究定量统一了独立实验中观察到的冰盘行为，建立了冰盘运动开始的预测标准，并指出了一种热驱动运输的途径：体积热对流与运动物体的耦合，这与大陆漂移和冰山翻倒等地球物理过程相关。

### 翻译

自然界中的旋转冰盘现象已有一个多世纪的报道，然而实验室实验产生了多样观察和矛盾解释，使得冰盘运动背后的机制仍然难以捉摸。我们结合数值模拟和标度分析，研究了实验室规模水箱中自由移动的冰盘。我们发现冰盘保持静止或经历自发运动，取决于冰盘与水的温度差和水深。运动由浮力驱动的向下羽流引发，该羽流源于水的密度异常——其密度在约4°C时达到峰值。关键的是，羽流在下降超过临界距离后，由于热对流不稳定性打破旋转和镜像对称性，从而使冰盘自主移动。我们的发现定量统一了独立实验中观察到的冰盘行为，并建立了冰盘运动开始的预测标准。更广泛地说，我们指出了热驱动运输的一条途径：体积热对流与运动物体的耦合，这与大陆漂移和冰山翻倒等地球物理过程相关。


### 论文摘要

Spinning ice discs in nature have been reported for more than a century, yet laboratory experiments have yielded diverse observations and contradictory explanations, leaving the mechanism behind the disc motion elusive. Here we combine numerical simulations and scaling analysis to investigate a freely moving ice disc in a lab-scale water tank. We observe the disc remaining stationary or experiencing spontaneous motion, depending on the disc-water temperature difference and water depth. The motion is initiated by a buoyancy-driven, downward plume arising from water's density anomaly -- its density peaks near $4^\circ$C. Crucially, the plume breaks rotational and mirror symmetries after descending beyond a critical distance due to a thermoconvective instability, thereby inducing the disc to move autonomously. Our findings quantitatively unify disc behaviors observed across independent experiments and establish a predictive criterion for the onset of disc motion. More broadly, we point to a route for thermally-driven transport: coupling of bulk thermoconvection and moving bodies, relevant to geophysical processes such as continental drift and iceberg capsizing.

---

## 149. Probabilistic dynamics of small groups in crowd flows

**论文链接:** [http://arxiv.org/abs/2511.13181v1](http://arxiv.org/abs/2511.13181v1)

**作者:** Chiel van der Laan, Alessandro Corbetta

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究分析了人群中两人小组（dyads）的动态行为，提供了基于周围人群状态的小组行为现象学模型和基础图表，引入了方向对数赔率（OLO）指标量化排列变化，为人群行为建模提供了数据驱动的基础。

### 背景

人群中经常有小群体一起移动，其中两人小组最为常见，占个体数量的70%。理解两人小组在人群中的动态行为对于理解整体人群行为至关重要。

### 目的

定量分析人群中两人小组的动态行为，为准确的人群动力学模型提供基础。

### 方法

分析了超过600万个两人小组的动态数据，这些数据来自荷兰埃因霍温火车站多年来的真实行人轨迹测量活动（总计2100万个轨迹），提供了基于周围人群状态的小组行为现象学模型。

### 主要发现

两人小组会根据周围人群调整人际距离并可能改变排列方式，从并行状态（有利于社交互动）转变为直线排列（有利于在密集人群中导航）；引入了方向对数赔率（OLO）标量指标量化排列变化，其导数是两个速度-密度基础图表的乘积。

### 结论

这些结果为真实世界中两人小组构型动态提供了统计稳健的数据驱动描述，为新的预测性、群体感知的人群模型奠定了基础。

### 翻译

人群中的行人经常作为小群体的一部分移动，构成高达70%的个体。两人小组是最常见的。因此，定量理解人群中两人行走的动态行为是理解整体人群行为基本原理的基本构建块，并且对于准确的人群动力学模型是必需的。不可避免地，由于行人行为的非确定性，动态特征的描述必须是概率性的。在这项工作中，我们分析了超过600万个两人小组的动态：在一个多年真实生活行人轨迹测量活动（来自荷兰埃因霍温火车站的2100万个轨迹）中，具有前所未有的分辨率的统计集合。我们提供了小组行为依赖于周围人群状态的现象学模型。我们呈现了基础图表的全面收集，这些图表概率性地关联了两人小组的速度和构型与周围人群的状态（密度、相对速度）。根据周围人群，两人小组会调整人际距离并可能改变排列方式，可能从并行状态（有利于社交互动）转变为直线排列（有利于在密集人群中导航）。为了定量研究排列变化，我们引入了一个标量指标，我们称之为方向对数赔率（OLO），它量化了并行与直线排列的相对对数可能性。概念上，OLO在类似玻尔兹曼的假设下量化了并行与直线构型的能量差异。我们建模了OLO如何依赖于人群状态，展示了其导数是两个速度-密度基础图表的乘积。总之，这些结果为真实世界中两人小组构型动态提供了统计稳健的数据驱动描述，为新的预测性、群体感知的人群模型奠定了基础。


### 论文摘要

Pedestrians in crowds frequently move as part of small groups, constituting up to 70% of individuals. Dyads (groups of two) are the most frequent. Understanding quantitatively the dynamics of dyads walking in crowds is therefore an essential building block towards a fundamental comprehension of crowd behavior as a whole, and mandatory for accurate crowd dynamics models. Unavoidably, due to the non-deterministic behavior of pedestrians, characterizations of the dynamics must be probabilistic. In this work, we analyze the dynamics of over 6M dyads: a statistical ensemble of unprecedented resolution within a multi-year real-life pedestrian trajectory measurement campaign (21M trajectories, from Eindhoven Station, NL). We provide phenomenological models for dyad behavior in dependence of the surrounding crowds state. We present a thorough collection of fundamental diagrams that probabilistically relate both dyad velocity and formation to the state of the surrounding crowd (density, relative velocity). Depending on the surrounding crowd, dyads adjust interpersonal distance and may shift in formation, possibly turning from abreast states (which favors social interaction) to in-line (which favors navigationing dense crowds). To quantitatively investigate formation changes, we introduce a scalar indicator, which we dub Orientation Log-Odds (OLO), that quantifies the relative log-likelihood of abreast versus in-file formations. Conceptually, the OLO quantifies energy difference of the abreast vs. in-file configuration under a Boltzmann-like assumption. We model how OLO depends on the crowd state, showcasing that its derivative is a product of two velocity-density fundamental diagrams. Together, these results provide a statistically robust, data-driven description of dyad configuration dynamics in real-world crowds, establishing a foundation towards new predictive, group-aware crowd models.

---

## 150. InteractiveGNNExplainer: A Visual Analytics Framework for Multi-Faceted Understanding and Probing of Graph Neural Network Predictions

**论文链接:** [http://arxiv.org/abs/2511.13160v1](http://arxiv.org/abs/2511.13160v1)

**作者:** TC Singh, Sougata Mukherjea

**发布时间:** 2025-11-17

**DOI:** 10.1109/IV68685.2025.00044

### GPT解析

### 总结

本文提出了InteractiveGNNExplainer，一个增强图神经网络(GNNs)可解释性的可视化分析框架，通过交互式视图和图编辑功能帮助用户理解GNN的预测和决策过程。

### 背景

图神经网络在基于图的学习任务中表现出色，但其复杂的非线性操作常使其成为不透明的'黑箱'，阻碍了用户信任、调试、偏见检测，限制了其在需要可解释性关键领域的应用。

### 目的

开发一个可视化分析框架来增强GNN的可解释性，特别关注节点分类任务，使GNN模型更加透明、可信。

### 方法

将协调的交互式视图（动态图布局、嵌入投影、特征检查、邻域分析）与后验(GNNExplainer)和内在(GAT注意力)解释技术相结合，并引入交互式图编辑功能，允许用户进行'假设分析'。

### 主要发现

通过在Cora和CiteSeer数据集上的案例研究，展示了该系统能够促进深入的错误分类诊断、GCN与GAT行为的比较分析以及对模型敏感性的严格测试。

### 结论

InteractiveGNNExplainer提供了对GNN预测更深层次、多方面的理解，有助于实现更透明、可信和强大的图分析。

### 翻译

图神经网络在基于图的学习任务中表现出色，但它们复杂的非线性操作常使其成为不透明的'黑箱'。这种不透明性阻碍了用户信任，使调试、偏见检测变得复杂，并限制了其在需要可解释性的关键领域的应用。本文介绍了InteractiveGNNExplainer，一个增强GNN可解释性的可视化分析框架，专注于节点分类。我们的系统独特地将协调的交互式视图（动态图布局、嵌入投影、特征检查、邻域分析）与既定的后验(GNNExplainer)和内在(GAT注意力)解释技术相结合。关键是它集成了交互式图编辑功能，允许用户通过扰动图结构并观察对GNN预测和解释的即时影响来进行'假设分析'。我们详细介绍了系统架构，并通过在Cora和CiteSeer数据集上的案例研究，展示了InteractiveGNNExplainer如何促进深入的错误分类诊断、GCN与GAT行为的比较分析以及对模型敏感性的严格测试。这些功能促进了GNN预测更深层次、多方面的理解，有助于实现更透明、可信和强大的图分析。


### 论文摘要

Graph Neural Networks (GNNs) excel in graph-based learning tasks, but their complex, non-linear operations often render them as opaque "black boxes". This opacity hinders user trust, complicates debugging, bias detection, and adoption in critical domains requiring explainability. This paper introduces InteractiveGNNExplainer, a visual analytics framework to enhance GNN explainability, focusing on node classification. Our system uniquely integrates coordinated interactive views (dynamic graph layouts, embedding projections, feature inspection, neighborhood analysis) with established post-hoc (GNNExplainer) and intrinsic (GAT attention) explanation techniques. Crucially, it incorporates interactive graph editing, allowing users to perform a "what-if" analysis by perturbing graph structures and observing immediate impacts on GNN predictions and explanations. We detail the system architecture and, through case studies on Cora and CiteSeer datasets, demonstrate how InteractiveGNNExplainer facilitates in-depth misclassification diagnosis, comparative analysis of GCN versus GAT behaviors, and rigorous probing of model sensitivity. These capabilities foster a deeper, multifaceted understanding of GNN predictions, contributing to more transparent, trustworthy, and robust graph analysis.

---

## 151. PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking

**论文链接:** [http://arxiv.org/abs/2511.13105v1](http://arxiv.org/abs/2511.13105v1)

**作者:** Seungjae Kim, SeungJoon Lee, MyeongAh Cho

**发布时间:** 2025-11-17

**备注:** AAAI 2026. Code: https://github.com/VisualScienceLab-KHU/PlugTrack

### GPT解析

### 总结

本文提出PlugTrack框架，通过自适应融合卡尔曼滤波器和数据驱动运动预测器，解决了多目标跟踪中传统方法与现代方法的局限性，实现了显著的性能提升。

### 背景

多目标跟踪主要采用检测后跟踪范式，卡尔曼滤波器因计算效率高成为标准运动预测器，但在非线性运动模式上存在缺陷；数据驱动运动预测器能捕捉复杂非线性动态，但存在领域泛化能力有限和计算开销大的问题。

### 目的

利用卡尔曼滤波器和数据驱动预测器的互补性，提出能够自适应融合这两种方法的新框架，以提高多目标跟踪性能。

### 方法

提出PlugTrack框架，通过多感知运动理解自适应融合卡尔曼滤波器和数据驱动运动预测器，采用多感知运动分析生成自适应混合因子，无需修改现有运动预测器。

### 主要发现

即使在以非线性运动为主的数据集中，卡尔曼滤波器在多达34%的情况下优于数据驱动预测器，表明现实世界跟踪场景本质上同时包含线性和非线性模式。

### 结论

PlugTrack在MOT17/MOT20上实现显著性能提升，在DanceTrack上达到最先进水平，是首个通过自适应融合桥接经典和现代运动预测范式的框架。

### 翻译

多目标跟踪(MOT)主要遵循检测后跟踪的范式，其中卡尔曼滤波器因其计算效率而成为标准的运动预测器，但在非线性运动模式上固有地失败。相反，最近的数据驱动运动预测器能够捕捉复杂的非线性动态，但受到领域泛化能力有限和计算开销大的困扰。通过广泛分析，我们揭示出即使在以非线性运动为主的数据集中，卡尔曼滤波器在多达34%的情况下优于数据驱动预测器，表明现实世界中的跟踪场景本质上同时包含线性和非线性模式。为了利用这种互补性，我们提出了PlugTrack，一个通过多感知运动理解自适应融合卡尔曼滤波器和数据驱动运动预测器的新型框架。我们的方法采用多感知运动分析来生成自适应混合因子。PlugTrack在MOT17/MOT20上实现了显著的性能提升，在DanceTrack上达到了最先进水平，而无需修改现有的运动预测器。据我们所知，PlugTrack是第一个通过自适应融合桥接经典和现代运动预测范式的框架。


### 论文摘要

Multi-object tracking (MOT) predominantly follows the tracking-by-detection paradigm, where Kalman filters serve as the standard motion predictor due to computational efficiency but inherently fail on non-linear motion patterns. Conversely, recent data-driven motion predictors capture complex non-linear dynamics but suffer from limited domain generalization and computational overhead. Through extensive analysis, we reveal that even in datasets dominated by non-linear motion, Kalman filter outperforms data-driven predictors in up to 34\% of cases, demonstrating that real-world tracking scenarios inherently involve both linear and non-linear patterns. To leverage this complementarity, we propose PlugTrack, a novel framework that adaptively fuses Kalman filter and data-driven motion predictors through multi-perceptive motion understanding. Our approach employs multi-perceptive motion analysis to generate adaptive blending factors. PlugTrack achieves significant performance gains on MOT17/MOT20 and state-of-the-art on DanceTrack without modifying existing motion predictors. To the best of our knowledge, PlugTrack is the first framework to bridge classical and modern motion prediction paradigms through adaptive fusion in MOT.

---

## 152. Real-time prediction of breast cancer sites using deformation-aware graph neural network

**论文链接:** [http://arxiv.org/abs/2511.13082v1](http://arxiv.org/abs/2511.13082v1)

**作者:** Kyunghyun Lee, Yong-Min Shin, Minwoo Shin, Jihun Kim, Sunghwan Lim, Won-Yong Shin, Kyungho Yoon

**发布时间:** 2025-11-17

### GPT解析

### 总结

这项研究开发了一种基于图神经网络(GNN)的模型，用于实时预测乳腺癌活检过程中的乳房变形和癌症部位位移，解决了传统MRI引导活检中实时变形建模的挑战，提高了诊断的精确性和效率。

### 背景

乳腺癌早期诊断对制定适当治疗计划和改善患者预后至关重要。虽然直接磁共振成像引导活检在检测癌变病灶方面显示出良好性能，但其应用受到程序时间长和高成本的限制。间接MRI引导活检虽然可以在MRI室外进行，但仍面临创建准确实时可变形乳房模型的挑战。

### 目的

开发一种能够实时预测乳腺癌活检过程中变形癌症部位的模型，克服传统方法的局限性，提高乳腺癌诊断的精确性和效率。

### 方法

研究团队开发了一种个体特异性的有限元(FE)模型，结合从磁共振(MR)图像中提取的乳房和肿瘤结构信息来模拟变形行为。然后采用图神经网络(GNN)模型，该模型能够处理基于表面位移和距离的图数据，从而准确预测包括肿瘤区域在内的整体组织位移。

### 主要发现

1. 模型在 phantom 和真实患者数据集上验证，对于癌症节点位移的准确度达到0.2毫米以内(RMSE)。2. 空间重叠的dice相似系数(DSC)达到0.977，与实际癌变区域高度一致。3. 该模型能够实现实时推理，与传统有限元模拟相比，计算成本提高了4000多倍。

### 结论

所提出的变形感知GNN模型为乳腺癌活检中的实时肿瘤位移预测提供了有前景的解决方案，具有高准确性和实时能力。其与临床程序的整合可以显著提高乳腺癌诊断的精确性和效率。

### 翻译

乳腺癌的早期诊断至关重要，能够制定适当的治疗计划并显著改善患者预后。虽然直接磁共振成像引导活检在检测癌变病灶方面显示出良好的性能，但其实际应用受到程序时间长和高成本的限制。为了克服这些问题，已经提出了一种间接MRI引导活检，允许在MRI室外进行操作，但仍然面临创建准确实时可变形乳房模型的挑战。在我们的研究中，我们通过开发一个基于图神经网络(GNN)的模型解决了这个问题，该模型能够在活检过程中实时准确预测变形的乳腺癌部位。通过结合从磁共振(MR)图像中提取的乳房和肿瘤结构信息，开发了一个个体特异性的有限元(FE)模型来模拟变形行为。然后采用GNN模型，设计用于处理表面位移和基于距离的图数据，能够准确预测整体组织位移，包括肿瘤区域的变形。该模型使用 phantom 和真实患者数据集进行了验证，癌症节点位移的准确度达到0.2毫米以内(RMSE)，与实际癌变区域的空间重叠的dice相似系数(DSC)为0.977。此外，该模型实现了实时推理，与传统有限元模拟相比，计算成本提高了4000多倍。所提出的变形感知GNN模型为乳腺癌活检中的实时肿瘤位移预测提供了有前景的解决方案，具有高准确性和实时能力。其与临床程序的整合可以显著提高乳腺癌诊断的精确性和效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决乳腺癌活检过程中实时预测肿瘤变形位置的问题。这个问题很重要，因为乳腺癌是最常见的女性癌症，早期诊断对提高生存率至关重要。虽然MRI引导的活检准确性高，但必须在MRI室内进行，存在时间长、成本高、调度困难等问题。间接MRI引导的活检（在MRI室外进行）面临无法准确建立实时可变形乳腺模型的挑战，而乳腺组织的高弹性在操作过程中常导致病灶显著位移，影响靶向准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统有限元分析(FE)虽然准确但计算成本高，无法实时操作的问题，以及机器学习方法需要处理不规则非欧几里得数据结构的挑战。他们选择图神经网络(GNN)作为主要架构，因为GNN适合处理有限元网格数据。方法设计包括：利用MRI数据构建个体特定有限元模型；设计特定GNN架构；结合距离图和结构化边增强优化信息传播。作者借鉴了有限元分析在生物医学变形研究中的应用、图神经网络在处理网格数据方面的优势、材料模型在软组织建模中的应用，以及深度学习在医学图像分析中的成功经验。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将有限元分析与图神经网络相结合，创建能够实时预测乳腺癌病灶变形位置的模型。利用MRI数据构建个体特定有限元模型，模拟乳腺和肿瘤变形行为，使用图神经网络处理表面位移数据，准确预测组织位移。通过结构化边增强直接连接表面节点和癌症质量表面节点，提高信息传递效率。整体流程包括：1)数据准备 - 获取MRI数据，分割组织，构建有限元模型，生成训练数据；2)图神经网络构建 - 创建图结构，设计网络架构；3)模型训练 - 使用表面位移作为输入，所有节点位移作为输出；4)性能验证 - 在phantom和患者数据集上评估精度和效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)变形感知图神经网络架构；2)利用MRI数据构建个体特定模型；3)实现比传统有限元分析快4000倍的实时性能；4)结构化边增强技术，直接连接表面节点和癌症质量表面节点；5)使用临床可测量的表面位移而非难以测量的力作为输入。相比之前的工作，本研究首次将图神经网络应用于乳腺癌活检过程中的实时肿瘤变形预测，实现了亚毫米级精度(0.2mm)和高空间重叠度(DSC为0.977)，解决了传统方法无法实时准确预测肿瘤变形的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本研究开发了一种基于变形感知图神经网络的实时预测模型，能够以亚毫米级精度预测乳腺癌活检过程中肿瘤的变形位置，显著提高了活检的精确性和效率。'}


### 论文摘要

Early diagnosis of breast cancer is crucial, enabling the establishment of appropriate treatment plans and markedly enhancing patient prognosis. While direct magnetic resonance imaging-guided biopsy demonstrates promising performance in detecting cancer lesions, its practical application is limited by prolonged procedure times and high costs. To overcome these issues, an indirect MRI-guided biopsy that allows the procedure to be performed outside of the MRI room has been proposed, but it still faces challenges in creating an accurate real-time deformable breast model. In our study, we tackled this issue by developing a graph neural network (GNN)-based model capable of accurately predicting deformed breast cancer sites in real time during biopsy procedures. An individual-specific finite element (FE) model was developed by incorporating magnetic resonance (MR) image-derived structural information of the breast and tumor to simulate deformation behaviors. A GNN model was then employed, designed to process surface displacement and distance-based graph data, enabling accurate prediction of overall tissue displacement, including the deformation of the tumor region. The model was validated using phantom and real patient datasets, achieving an accuracy within 0.2 millimeters (mm) for cancer node displacement (RMSE) and a dice similarity coefficient (DSC) of 0.977 for spatial overlap with actual cancerous regions. Additionally, the model enabled real-time inference and achieved a speed-up of over 4,000 times in computational cost compared to conventional FE simulations. The proposed deformation-aware GNN model offers a promising solution for real-time tumor displacement prediction in breast biopsy, with high accuracy and real-time capability. Its integration with clinical procedures could significantly enhance the precision and efficiency of breast cancer diagnosis.

---

## 153. An energy cascade finite volume scheme for a mixed 3- and 4-wave kinetic equation arising from the theory of finite-temperature trapped Bose gases

**论文链接:** [http://arxiv.org/abs/2511.13064v1](http://arxiv.org/abs/2511.13064v1)

**作者:** Arijit Das, Minh-Binh Tran

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究构建了一个有限体积算法，用于处理包含3波和4波相互作用的复杂波动力学方程，该算法能够捕捉能量级联行为。

### 背景

该研究建立在近期为3波动力学方程设计的数值方案基础上，这些方案能够捕捉能量级联。研究模型出现在有限温度玻色-爱因斯坦凝聚的背景下。

### 目的

构建一个有限体积算法，处理包含3波和4波相互作用的复杂波动力学方程。

### 方法

建立所提出方案的理论性质，并通过数值实验进行验证。

### 主要发现

数值实验表明该方案能够成功捕捉方程预测的能量级联行为。

### 结论

所提出的有限体积算法成功处理了包含3波和4波相互作用的复杂波动力学方程，并能够捕捉能量级联行为。

### 翻译

基于近期为3波动力学方程设计的捕捉能量级联的数值方案的发展，我们在本文中构建了一个有限体积算法，用于处理一个显著更复杂的波动力学方程，其碰撞算子同时包含3波和4波相互作用。该模型出现在有限温度玻色-爱因斯坦凝聚的背景下。我们建立了所提出方案的理论性质，数值实验表明它成功捕捉了方程预测的能量级联行为。


### 论文摘要

Building on recent developments in numerical schemes designed to capture energy cascades for 3-wave kinetic equations~\cite{das2024numerical, walton2022deep, walton2023numerical, walton2024numerical}, we construct in this work a finite-volume algorithm for a significantly more complex wave kinetic equation whose collision operator incorporates both 3-wave and 4-wave interactions. This model arises in the context of finite-temperature Bose-Einstein condensation. We establish theoretical properties of the proposed scheme, and our numerical experiments demonstrate that it successfully captures the energy cascade behavior predicted by the equation.

---

## 154. Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts

**论文链接:** [http://arxiv.org/abs/2511.13032v1](http://arxiv.org/abs/2511.13032v1)

**作者:** Sheng Liu, Yuanzhi Liang, Jiepeng Wang, Sidan Du, Chi Zhang, Xuelong Li

**发布时间:** 2025-11-17

### GPT解析

### 总结

论文提出Uni-Inter，一个统一的人体运动生成框架，支持多种交互场景（人-人、人-物、人-场景）在单一任务无关架构中。通过引入统一交互体积(UIV)表示方法，将异构交互实体编码到共享空间场中，实现关系推理和复合交互建模。

### 背景

现有人体运动生成方法依赖于特定任务设计，泛化能力有限，无法有效处理多种交互场景，特别是在复杂环境中的复合交互。

### 目的

开发统一框架处理多种交互场景，无需针对每种任务设计专门架构，提高模型在复杂环境中的泛化能力，实现可扩展运动合成。

### 方法

提出Uni-Inter框架和统一交互体积(UIV)表示方法，将异构交互实体编码到共享空间场中；将运动生成为基于UIV的关节级概率预测，捕捉细粒度空间依赖关系，产生连贯、上下文感知的行为。

### 主要发现

Uni-Inter在三种代表性交互任务中实现竞争性性能；模型能很好地泛化到新的实体组合；统一的复合交互建模为复杂环境中的可扩展运动合成提供了有前途的方向。

### 结论

统一建模复合交互是复杂环境中可扩展运动合成的一个有前途的方向。Uni-Inter通过统一架构和UIV表示方法，解决了现有方法处理多种交互场景时的局限性。

### 翻译

我们提出了Uni-Inter，一个统一的人体运动生成框架，支持广泛的交互场景，包括人-人、人-物和人-场景，都在一个单一的、任务无关的架构中。与依赖于特定任务设计且泛化能力有限的方法不同，Uni-Inters引入了统一交互体积(UIV)，这是一种体积表示方法，将异构交互实体编码到共享空间场中。这实现了一致的关系推理和复合交互建模。运动生成为基于UIV的关节级概率预测，使模型能够捕捉细粒度的空间依赖关系，并产生连贯的、上下文感知的行为。在三个代表性交互任务上的实验表明，Uni-Inter实现了具有竞争力的性能，并能很好地泛化到新的实体组合。这些结果表明，复合交互的统一建模为复杂环境中的可扩展运动合成提供了有前途的方向。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文旨在解决3D人体运动生成中不同交互场景（人人交互、人-物体交互、人-场景交互）的统一建模问题。现有方法通常针对特定交互类型设计，缺乏泛化能力，无法处理复合交互场景（如同时与人交流、拿起物体并坐在沙发上）。这个问题在现实中很重要，因为真实世界的交互往往是复合的，现有方法的局限性限制了它们在复杂环境中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法无法处理复合交互场景的局限性，然后借鉴了自动驾驶领域的统一感知框架（如使用网格化占用表示统一建模行人、车辆和道路结构）。他们思考如何将这种统一表示方法应用到交互运动生成中，设计了统一交互体积（UIV）来编码不同实体到共享空间。同时，他们还借鉴了扩散模型用于运动生成、语义占用网格用于场景理解以及SMPL模型用于人体表示等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过统一交互体积（UIV）将不同交互实体（人类、物体、场景）编码到共享的3D占用场中，实现一致的关系推理和复合交互建模。整体流程包括：1)接收文本描述和条件观测；2)将不同交互实体编码为语义占用网格；3)构建统一交互体积；4)使用扩散模型预测关节级空间分布；5)通过计算分布期望获取关节位置并生成最终运动序列；6)采用统一训练策略在多个交互任务上训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一框架：首次支持任意组合交互实体；2)统一交互体积（UIV）：将异构实体编码到共享空间；3)UIV对齐正则化：将运动建模为空间分布估计；4)统一训练策略：促进跨任务知识转移。相比之前工作，不同之处在于：使用统一表示而非不同格式（骨架、点云等）、单一任务无关架构而非特定任务设计、能够处理任意组合而非限于特定交互类型、建模关节空间分布而非直接预测坐标。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Uni-Inter通过统一交互体积和空间分布建模，首次实现了在单一框架内生成支持任意组合交互场景的3D人体运动，显著提升了模型在复合交互中的泛化能力。'}


### 论文摘要

We present Uni-Inter, a unified framework for human motion generation that supports a wide range of interaction scenarios: including human-human, human-object, and human-scene-within a single, task-agnostic architecture. In contrast to existing methods that rely on task-specific designs and exhibit limited generalization, Uni-Inter introduces the Unified Interactive Volume (UIV), a volumetric representation that encodes heterogeneous interactive entities into a shared spatial field. This enables consistent relational reasoning and compound interaction modeling. Motion generation is formulated as joint-wise probabilistic prediction over the UIV, allowing the model to capture fine-grained spatial dependencies and produce coherent, context-aware behaviors. Experiments across three representative interaction tasks demonstrate that Uni-Inter achieves competitive performance and generalizes well to novel combinations of entities. These results suggest that unified modeling of compound interactions offers a promising direction for scalable motion synthesis in complex environments.

---

## 155. Can We Predict the Next Question? A Collaborative Filtering Approach to Modeling User Behavior

**论文链接:** [http://arxiv.org/abs/2511.12949v1](http://arxiv.org/abs/2511.12949v1)

**作者:** Bokang Fu, Jiahao Wang, Xiaojing Liu, Yuli Liu

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种协作过滤增强的问题预测(CFQP)框架，解决了大型语言模型在用户偏好建模方面的静态局限，实现了对用户交互行为的动态捕捉和预测。

### 背景

大型语言模型在语言理解和生成方面表现出色，推动了高级对话和推荐系统的发展，但这些系统通常静态地建模用户偏好，无法捕捉交互行为的动态和序列特性。

### 目的

解决语言建模和行为序列建模之间的固有差距，利用用户历史问题的序列数据来进行预测任务。

### 方法

提出CFQP框架，通过整合个性化记忆模块和基于图的偏好传播，动态建模演化的用户-问题交互，使系统能够自适应地从用户特定历史中学习，同时通过相似用户的协作信号来优化预测。

### 主要发现

实验结果表明，该方法能够有效生成模仿真实用户提问模式的智能体，在构建主动和自适应的对话系统方面具有潜力。

### 结论

CFQP框架能够有效捕捉用户偏好的动态变化，提高对话系统的预测能力和适应性。

### 翻译

近年来，大型语言模型在语言理解和生成方面表现出色，推动了高级对话和推荐系统的发展。然而，一个显著的局限仍然存在：这些系统通常静态地建模用户偏好，无法捕捉交互行为的动态和序列特性。用户历史问题的序列提供了关于兴趣和认知模式演化的丰富、隐含信号，但由于语言建模和行为序列建模之间的固有差距，利用这些时间数据进行预测任务仍然具有挑战性。为了弥合这一差距，我们提出了一个协作过滤增强的问题预测(CFQP)框架。CFQP通过整合个性化记忆模块和基于图的偏好传播，动态建模演化的用户-问题交互。这种双重机制使系统能够自适应地从用户特定历史中学习，同时通过相似用户的协作信号来优化预测。实验结果表明，我们的方法能够有效生成模仿真实用户提问模式的智能体，凸显了其在构建主动和自适应对话系统方面的潜力。


### 论文摘要

In recent years, large language models (LLMs) have excelled in language understanding and generation, powering advanced dialogue and recommendation systems. However, a significant limitation persists: these systems often model user preferences statically, failing to capture the dynamic and sequential nature of interactive behaviors. The sequence of a user's historical questions provides a rich, implicit signal of evolving interests and cognitive patterns, yet leveraging this temporal data for predictive tasks remains challenging due to the inherent disconnect between language modeling and behavioral sequence modeling.   To bridge this gap, we propose a Collaborative Filtering-enhanced Question Prediction (CFQP) framework. CFQP dynamically models evolving user-question interactions by integrating personalized memory modules with graph-based preference propagation. This dual mechanism allows the system to adaptively learn from user-specific histories while refining predictions through collaborative signals from similar users. Experimental results demonstrate that our approach effectively generates agents that mimic real-user questioning patterns, highlighting its potential for building proactive and adaptive dialogue systems.

---

## 156. Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos

**论文链接:** [http://arxiv.org/abs/2511.12882v1](http://arxiv.org/abs/2511.12882v1)

**作者:** Taiyi Su, Jian Zhu, Yaxuan Li, Chong Ma, Zitai Huang, Yichen Zhu, Hanli Wang, Yi Xu

**发布时间:** 2025-11-17

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

MTV-World是一种创新的具身世界模型，通过多视角轨迹视频控制和精确评估方法，实现了在复杂场景中的精确机器人控制和物理交互。

### 背景

现有的具身世界模型旨在通过视觉观察和动作来预测和与物理世界互动，但难以将低级动作(如关节位置)准确转换为预测帧中的精确机器人运动，导致与真实世界物理互动不一致。

### 目的

解决现有模型的局限性，提出一种能够进行精确视觉运动预测的具身世界模型。

### 方法

提出MTV-World模型，引入多视角轨迹-视频控制；使用通过相机内外参数和笛卡尔空间变换获得的轨迹视频作为控制信号；引入多视角框架弥补空间信息损失；基于多视角轨迹视频预测未来帧；开发自动评估流程评估机器人运动精度和物体交互准确性；将空间一致性表述为物体位置匹配问题，采用Jaccard Index作为评估指标。

### 主要发现

广泛的实验表明，MTV-World在复杂双臂场景中实现了精确的控制执行和准确的物理交互建模。

### 结论

MTV-World模型有效解决了现有具身世界模型在动作转换和物理交互一致性方面的局限性。

### 翻译

具身世界模型旨在通过视觉观察和动作来预测和与物理世界互动。然而，现有模型难以将低级动作(如关节位置)准确转换为预测帧中的精确机器人运动，导致与真实世界物理互动不一致。为解决这些局限性，我们提出了MTV-World，一种引入多视角轨迹-视频控制的具身世界模型，用于精确的视觉运动预测。具体而言，我们不直接使用低级动作进行控制，而是采用通过相机内外参数和笛卡尔空间变换获得的轨迹视频作为控制信号。然而，将3D原始动作投影到2D图像上不可避免地会导致空间信息损失，使得单一视角不足以进行准确的交互建模。为克服这一点，我们引入了多视角框架，弥补空间信息损失，确保与物理世界的高度一致性。MTV-World基于多视角轨迹视频作为输入，并以每个视角的初始帧为条件来预测未来帧。此外，为系统评估机器人运动精度和物体交互准确性，我们开发了一个利用多模态大模型和参考视频对象分割模型的自动评估流程。为衡量空间一致性，我们将其表述为物体位置匹配问题，并采用Jaccard Index作为评估指标。大量实验证明，MTV-World在复杂双臂场景中实现了精确的控制执行和准确的物理交互建模。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有具身世界模型难以准确将低级动作（如关节位置）转换为精确机器人运动的问题，导致预测帧与真实世界物理交互不一致。这个问题很重要，因为具身世界模型是机器人理解、预测和与物理世界交互的关键技术，广泛应用于机器人学习数据生成和策略评估等场景；如果模型无法准确预测机器人动作和物体交互，基于这些模型的机器人系统在实际应用中可能会失败或不稳定。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有模型难以准确转换低级动作为精确机器人运动的问题，然后分析原因是直接使用低级动作作为控制信号会导致不精确的运动轨迹，而将3D动作投影到2D图像上会导致空间信息丢失。因此，作者设计了轨迹视频作为显式控制信号，采用多视角框架弥补空间信息丢失，并利用物体掩码作为前景先验。该方法借鉴了现有视频生成模型、视觉语言模型、参考视频对象分割等技术，采用了变分自编码器、CLIP图像编码器等组件，但在控制表示、多视角处理和评估方法上进行了创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多视角轨迹视频作为控制信号，通过多视角框架弥补空间信息丢失，结合物体掩码作为前景先验提高物理交互建模准确性。整体流程包括：1)轨迹表示：校准相机，将机器人关节位置转换为笛卡尔空间末端执行器姿态，投影到2D图像并合成轨迹视频；2)物体表示：使用VLM生成物体描述，用RVOS生成物体掩码作为前景先验；3)模型架构：用VAE编码视频序列和轨迹控制视频，用CLIP和umT5编码多模态信息，通过DiT进行视频预测；4)自动评估：用VLM和RVOS生成物体掩码，计算Jaccard指数评估空间一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多视角轨迹视频控制：使用轨迹视频作为显式控制信号；2)多视角框架：通过多个视角轨迹视频弥补空间信息丢失；3)物体掩码作为前景先验：明确建模机器人手臂与物体交互；4)自动评估管道：系统评估运动精度和交互准确性。相比之前工作，不同之处在于：控制表示从低级动作或潜在向量变为轨迹视频；从单视角处理扩展到多视角框架；评估方法从基于感知质量变为基于物体位置匹配；通过物体掩码和多视角框架更好地建模物理交互准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了MTV-World，一种基于多视角轨迹视频控制的具身世界模型，通过显式轨迹表示和多视角框架实现了精确的机器人运动控制和物理交互建模，并开发了自动评估管道来系统评估模型性能。'}


### 论文摘要

Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.

---

## 157. Mapping fNIRS Signals to Agent Performance: Toward Reinforcement Learning from Neural Feedback

**论文链接:** [http://arxiv.org/abs/2511.12844v1](http://arxiv.org/abs/2511.12844v1)

**作者:** Julia Santaniello, Matthew Russell, Benson Jiang, Donatello Sassaroli, Robert Jacob, Jivko SInapov

**发布时间:** 2025-11-17

**备注:** Accepted to the Association for the Advancement of Artificial Intelligence (AAAI) 2026. To appear in the AAAI 2026 Proceedings

### GPT解析

### 总结

本文介绍了一种结合被动脑机接口(BCI)和人类反馈强化学习(RLHF)的框架，通过功能性近红外光谱(fNIRS)收集人类隐式神经信号来指导智能体训练。研究团队收集了25名参与者在三个不同领域的fNIRS数据集，训练分类器和回归器来预测智能体性能，并展示了跨主题泛化能力和微调模型的性能提升。

### 背景

强化学习从人类反馈(RLHF)是一种通过将人类反馈整合到智能体训练过程中来使智能体行为与人类偏好保持一致的方法。然而，传统的RLHF通常需要显式的人类反馈，而隐式的神经信号可能提供更自然、更连续的反馈方式。

### 目的

本研究旨在探索使用被动脑机接口(BCI)来捕获人类隐式神经信号，并将这些信号映射到智能体性能，从而构建脑驱动的RLHF系统，使智能体训练更加自然和高效。

### 方法

研究团队收集了25名人类参与者在三个不同领域(拾放机器人、月球着陆器和飞扬的小鸟)的功能性近红外光谱(fNIRS)记录数据集。他们训练分类器来预测智能体性能水平(最优、次优或最差情况)，并训练回归器来预测智能体选择的行为与一组近优策略之间的偏差程度。研究还评估了跨主题泛化能力，并测试了使用少量特定于主题的数据微调预训练模型的效果。

### 主要发现

1. 研究团队成功训练了分类器，对于二元分类的平均F1得分为67%，多类模型的平均F1得分为46%。2. 研究团队训练的回归器能够提供智能体性能的连续度量。3. 跨主题泛化评估显示，使用少量特定于主题的数据微调预训练模型，二元分类和多类模型的平均F1得分分别提高了17%和41%。4. 研究证明将隐式fNIRS信号映射到智能体性能是可行的，并且可以进一步改进。

### 结论

本研究成功证明了将隐式fNIRS信号映射到智能体性能的可行性，为未来的脑驱动RLHF系统奠定了基础。通过微调预训练模型，可以显著提高性能，这表明脑机接口辅助的强化学习具有实际应用潜力。

### 翻译

强化学习从人类反馈(RLHF)是一种通过将人类反馈整合到智能体训练过程中来使智能体行为与人类偏好保持一致的方法。我们介绍了一个可能的框架，该框架使用被动脑机接口(BCI)来从隐式神经信号指导智能体训练。我们发布了一个新颖的数据集，包含来自25名人类参与者在三个领域的功能性近红外光谱(fNIRS)记录：拾放机器人、月球着陆器和飞扬的小鸟。我们训练分类器来预测智能体性能水平(最优、次优或最差情况)来自预处理fNIRS特征向量窗口，在条件和领域上平均，二元分类的平均F1得分为67%，多类模型的平均F1得分为46%。我们还训练回归器来预测智能体选择的行为与一组近优策略之间的偏差程度，提供性能的连续度量。我们评估了跨主题泛化能力，并证明使用少量特定于主题的数据微调预训练模型可以使二元和多类模型的平均F1得分分别提高17%和41%。我们的工作证明了将隐式fNIRS信号映射到智能体性能是可行的并且可以改进，为未来的脑驱动RLHF系统奠定了基础。


### 论文摘要

Reinforcement Learning from Human Feedback (RLHF) is a methodology that aligns agent behavior with human preferences by integrating human feedback into the agent's training process. We introduce a possible framework that employs passive Brain-Computer Interfaces (BCI) to guide agent training from implicit neural signals. We present and release a novel dataset of functional near-infrared spectroscopy (fNIRS) recordings collected from 25 human participants across three domains: a Pick-and-Place Robot, Lunar Lander, and Flappy Bird. We train classifiers to predict levels of agent performance (optimal, sub-optimal, or worst-case) from windows of preprocessed fNIRS feature vectors, achieving an average F1 score of 67% for binary classification and 46% for multi-class models averaged across conditions and domains. We also train regressors to predict the degree of deviation between an agent's chosen action and a set of near-optimal policies, providing a continuous measure of performance. We evaluate cross-subject generalization and demonstrate that fine-tuning pre-trained models with a small sample of subject-specific data increases average F1 scores by 17% and 41% for binary and multi-class models, respectively. Our work demonstrates that mapping implicit fNIRS signals to agent performance is feasible and can be improved, laying the foundation for future brain-driven RLHF systems.

---

## 158. TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction

**论文链接:** [http://arxiv.org/abs/2511.12578v1](http://arxiv.org/abs/2511.12578v1)

**作者:** Yukuo Ma, Cong Liu, Junke Wang, Junqi Liu, Haibin Huang, Zuxuan Wu, Chi Zhang, Xuelong Li

**发布时间:** 2025-11-16

### GPT解析

### 总结

TempoMaster是一种新型框架，通过将长视频生表述为下一帧率预测问题，实现了高质量长视频的生成。

### 背景

长视频生成是计算机视觉领域的重要挑战，需要保持视觉质量和时间一致性。

### 目的

开发一种能够生成高质量长视频的框架，解决长视频生成中的视觉细节和时间连续性问题。

### 方法

首先生成低帧率视频作为粗略蓝图，然后逐步提高帧率细化细节；使用双向注意力处理帧率级别内的时间关系，通过自回归处理跨帧率的关系，实现长程时间一致性和高效并行合成。

### 主要发现

大量实验表明，TempoMaster在长视频生成方面建立了新的最先进水平，在视觉质量和时间质量方面都表现出色。

### 结论

TempoMaster为长视频生成提供了一种有效的方法，通过帧率预测和双向注意力机制，实现了高质量的长视频生成。

### 翻译

我们提出了TempoMaster，一种将长视频生成表述为下一帧率预测的新框架。具体来说，我们首先生成一个低帧率片段，作为整个视频序列的粗略蓝图，然后逐步提高帧率以细化视觉细节和运动连续性。在生成过程中，TempoMaster在每个帧率级别内使用双向注意力，同时在帧率之间执行自回归，从而实现长程时间一致性，同时实现高效的并行合成。大量实验表明，TempoMaster在长视频生成方面建立了新的最先进水平，在视觉和时间质量方面表现出色。


### 论文摘要

We present TempoMaster, a novel framework that formulates long video generation as next-frame-rate prediction. Specifically, we first generate a low-frame-rate clip that serves as a coarse blueprint of the entire video sequence, and then progressively increase the frame rate to refine visual details and motion continuity. During generation, TempoMaster employs bidirectional attention within each frame-rate level while performing autoregression across frame rates, thus achieving long-range temporal coherence while enabling efficient and parallel synthesis. Extensive experiments demonstrate that TempoMaster establishes a new state-of-the-art in long video generation, excelling in both visual and temporal quality.

---

## 159. DualGR: Generative Retrieval with Long and Short-Term Interests Modeling

**论文链接:** [http://arxiv.org/abs/2511.12518v1](http://arxiv.org/abs/2511.12518v1)

**作者:** Zhongchao Yi, Kai Feng, Xiaojian Ma, Yalong Wang, Yongqi Liu, Han Li, Zhengyang Zhou, Yang Wang

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种名为DualGR的生成式检索框架，用于解决大规模工业推荐系统中的三个关键挑战：平衡用户长期和短期兴趣、减少生成分层语义ID时的噪声干扰，以及建模负面反馈。该框架通过双分支路由器、基于搜索的SID解码和曝光感知损失函数实现了这些目标，并在快手短视频推荐系统中取得了显著效果。

### 背景

在大规模工业推荐系统中，检索需要在严格延迟限制下从海量语料库中产生高质量候选。生成式检索(GR)已成为基于嵌入的检索(EBR)的可行替代方案，它将项目量化到有限标记空间并通过自回归解码生成候选，提供了通过交叉注意力明确建模目标-历史交互的可扩展路径。

### 目的

解决生成式检索面临的三大挑战：1)平衡用户长期和短期兴趣；2)减少生成分层语义ID时的噪声干扰；3)对未点击曝光项等负面反馈进行显式建模。

### 方法

提出DualGR框架，包含三个关键组件：1)双分支长短期路由器(DBR)，通过显式建模用户长短期行为覆盖稳定偏好和短暂意图；2)基于搜索的SID解码(S2D)，通过将候选交互限制在当前粗粒度存储桶内控制噪声并提高计算效率；3)曝光感知下一词预测损失(ENTP-Loss)，将'已曝光但未点击'项目视为硬负样本，实现及时兴趣衰减。

### 主要发现

在快手大规模短视频推荐系统上，DualGR取得了优异性能。在线A/B测试结果显示视频观看时间提升+0.527%，观看时长提升+0.432%，验证了该框架的有效性。

### 结论

DualGR被验证为工业生成式检索的一种实用有效范式，能够有效解决大规模推荐系统中的关键挑战，并带来显著的性能提升。

### 翻译

在大规模工业推荐系统中，检索必须在严格延迟下从大规模语料库中产生高质量候选。最近，生成式检索(GR)已成为基于嵌入的检索(EBR)的可行替代方案，它将项目量化到有限标记空间并通过自回归解码生成候选，提供了通过交叉注意力明确建模目标-历史交互的可扩展路径。然而，三个挑战仍然存在：1)如何平衡用户的长期和短期兴趣；2)生成分层语义ID(SID)时的噪声干扰；3)缺乏对未点击曝光项等负面反馈的显式建模。为解决这些挑战，我们提出了DualGR，这是一种生成式检索框架，通过选择性激活显式建模用户兴趣的双时间范围。具体来说，DualGR利用双分支长短期路由器(DBR)通过显式建模用户的长短期行为来覆盖稳定偏好和短暂意图。同时，提出了基于搜索的SID解码(S2D)，通过在细粒度(level-2/3)SID预测期间将候选交互限制在当前粗粒度(level-1)存储桶内，控制上下文诱导的噪声并提高计算效率。最后，我们提出了曝光感知下一词预测损失(ENTP-Loss)，将'已曝光但未点击'的项目视为level-1的硬负样本，实现及时兴趣衰减。在快手大规模短视频推荐系统上，DualGR取得了卓越性能。在线A/B测试显示视频观看时间提升+0.527%，观看时长提升+0.432%，验证了DualGR作为工业生成式检索的一种实用有效范式。


### 论文摘要

In large-scale industrial recommendation systems, retrieval must produce high-quality candidates from massive corpora under strict latency. Recently, Generative Retrieval (GR) has emerged as a viable alternative to Embedding-Based Retrieval (EBR), which quantizes items into a finite token space and decodes candidates autoregressively, providing a scalable path that explicitly models target-history interactions via cross-attention. However, three challenges persist: 1) how to balance users' long-term and short-term interests , 2) noise interference when generating hierarchical semantic IDs (SIDs), 3) the absence of explicit modeling for negative feedback such as exposed items without clicks. To address these challenges, we propose DualGR, a generative retrieval framework that explicitly models dual horizons of user interests with selective activation. Specifically, DualGR utilizes Dual-Branch Long/Short-Term Router (DBR) to cover both stable preferences and transient intents by explicitly modeling users' long- and short-term behaviors. Meanwhile, Search-based SID Decoding (S2D) is presented to control context-induced noise and enhance computational efficiency by constraining candidate interactions to the current coarse (level-1) bucket during fine-grained (level-2/3) SID prediction. % also reinforcing intra-class consistency. Finally, we propose an Exposure-aware Next-Token Prediction Loss (ENTP-Loss) that treats "exposed-but-unclicked" items as hard negatives at level-1, enabling timely interest fade-out. On the large-scale Kuaishou short-video recommendation system, DualGR has achieved outstanding performance. Online A/B testing shows +0.527% video views and +0.432% watch time lifts, validating DualGR as a practical and effective paradigm for industrial generative retrieval.

---

## 160. Manipulating Collective Opinion through Social Network Intervention

**论文链接:** [http://arxiv.org/abs/2511.12444v1](http://arxiv.org/abs/2511.12444v1)

**作者:** Shigefumi Hata, Renaud Lambiotte, Hiroya Nakao, Ryota Kobayashi

**发布时间:** 2025-11-16

**备注:** 20 pages, 11 figures

### GPT解析

### 总结

社交媒体平台改变了集体意见形成的动态，使大规模快速互动成为可能，同时也使在线讨论面临极化和操纵的风险。传统意见动态模型预测会达成共识，但实证证据表明存在持续的极化和激进现象。本文分析了一个捕捉网络社会中极化、激进和共识之间复杂互动的数学模型，确定了集体行为相变的临界阈值，并探索了基于网络的干预策略对集体意见的影响。

### 背景

社交媒体平台已经改变了集体意见形成的动态，使大规模快速互动成为可能，同时也使在线讨论面临极化和操纵的风险。传统的意见动态模型通常预测会达成共识，但实证证据持续表明存在持续的极化和激进现象，特别是在有争议的问题上。

### 目的

分析一个捕捉网络社会中极化、激进和共识之间复杂互动的数学模型，通过分析意见集群的出现和稳定性，确定集体行为相变的临界阈值，并探索基于网络的干预策略对集体意见的影响。

### 方法

分析一个数学模型，该模型捕捉了网络社会中极化、激进和共识之间的复杂互动。通过分析意见集群的出现和稳定性，确定集体行为相变的临界阈值，并通过稳定性景观进行解释。进一步探索基于网络的干预策略来操纵集体意见。

### 主要发现

确定了标记集体行为相变的临界阈值；发现减少代理之间的互动可能导致意见分布中意外且不可逆转的转变；揭示了干预策略的双刃剑特性。

### 结论

研究结果强调了干预策略的双刃剑性质，为数字环境中公共意见的脆弱性和可操纵性提供了理论见解。

### 翻译

社交媒体平台已经改变了集体意见形成的动态，使大规模快速互动成为可能，同时也使在线讨论面临极化和操纵。传统的意见动态模型通常预测会达成共识，然而实证证据持续强调持续极化和激进化的存在，特别是在有争议的问题上。本文分析性地研究了一个捕捉网络社会中极化、激进化和共识之间复杂互动的数学模型。通过分析意见集群的出现和稳定性，我们确定了标记集体行为相变的临界阈值，通过稳定性景观进行解释。我们进一步探索了基于网络的干预策略来操纵集体意见，揭示减少代理之间的互动可能导致意见分布中意外且不可逆转的转变。我们的结果强调了干预策略的双刃剑性质，为数字环境中公共意见的脆弱性和可操纵性提供了理论见解。


### 论文摘要

Social media platforms have transformed the dynamics of collective opinion formation, enabling rapid, large-scale interactions while simultaneously exposing online discourse to polarization and manipulation. Traditional models of opinion dynamics often predict convergence to a consensus, yet empirical evidence consistently highlights persistent polarization and radicalization, especially on contentious issues. This paper analytically investigates a mathematical model that captures the complex interplay of polarization, radicalization, and consensus within networked societies. By analyzing the emergence and stability of opinion clusters, we identify critical thresholds marking phase transitions in collective behavior, interpreted via a stability landscape. We further explore network-based interventions to manipulate the collective opinion, revealing that reducing inter-agent interactions can lead to unintended, irreversible shifts in opinion distributions. Our results underscore the dual-edged nature of intervention strategies, offering theoretical insight into the fragility and manipulability of public opinion in digital environments.

---

## 161. Dynamics of levitation during rolling over a thin viscous film

**论文链接:** [http://arxiv.org/abs/2511.12441v1](http://arxiv.org/abs/2511.12441v1)

**作者:** Siqi Chen, Cheng Liu, Neil J. Balmforth, Sheldon Green, Boris Stoeber

**发布时间:** 2025-11-16

**备注:** 35 pages, 18 figures

### GPT解析

### 总结

研究圆柱体或车轮在薄粘性薄膜上滚动的动力学数学模型，特别关注浮升现象

### 背景

圆柱体或车轮在薄粘性薄膜上滚动的动力学问题

### 目的

建立数学模型描述车轮在粘性薄膜上的动力学行为，研究浮升现象

### 方法

结合雷诺润滑方程和车轮运动方程，研究无限宽车轮和窄车轮两种渐近极限，再考虑有限宽度车轮，使用变量分离法求解方程

### 主要发现

在极限情况下流体间隙前后端呈直线；有限宽度车轮时流体间隙足迹仍为矩形；宽车轮易出现回流淹没现象；可实现稳定滑行状态；探索了车轮在有限长度薄膜上的浮升和触地过程

### 结论

理论预测与实验数据进行了比较验证

### 翻译

推导了圆柱体或车轮在薄粘性薄膜上滚动的动力学数学模型。该模型结合了流体的雷诺润滑方程和车轮的运动方程。详细研究了两种渐近极限以探究浮升动力学：无限宽车轮和相对较窄的车轮。在这两种情况下，流体填充间隙的前后端要么是直的，要么接近直的。为了连接这两种渐近极限，考虑了有限宽度的车轮，引入了进一步的简化近似：尽管对于有限宽度，前后端不再预期保持直线，但流体填充间隙的足迹仍被视为矩形，并在前后端以车轮平均意义上的边界条件施加。然后可以通过变量分离法求解雷诺方程。对于较宽的车轮，具有大量流入通量或车轮相对较重的情况下，系统容易出现回流导致的淹没，流体无法从下方通过。否则，可实现稳定的滑行状态。探索了车轮在有限长度薄膜上滚动的浮升和触地过程。将理论预测与一组实验数据进行了比较。


### 论文摘要

A mathematical model is derived for the dynamics of a cylinder, or wheel, rolling over a thin viscous film. The model combines the Reynolds lubrication equation for the fluid with an equation of motion for the wheel. Two asymptotic limits are studied in detail to interrogate the dynamics of levitation: an infinitely wide wheel and a relatively narrow one. In both cases the front and back of the fluid-filled gap are either straight or nearly so. To bridge the gap between these two asymptotic limits, wheels of finite width are considered, introducing a further simplying approximation: although the front and back are no longer expected to remain straight for a finite width, the footprint of the fluid-filled gap is still taken to be rectangular, with boundary conditions imposed at the front and back in a wheel-averaged sense. The Reynolds equation can then be solved by separation of variables. For wider wheels, with a large amount of incoming flux or a relatively heavy loading of the wheel, the system is prone to flooding by back flow with fluid unable to pass underneath. Otherwise steady planing states are achieved. Both lift-off and touch-down are explored for a wheel rolling over a film of finite length. Theoretical predictions are compared with a set of experimental data.

---

## 162. Integrating Neural Differential Forecasting with Safe Reinforcement Learning for Blood Glucose Regulation

**论文链接:** [http://arxiv.org/abs/2511.12417v1](http://arxiv.org/abs/2511.12417v1)

**作者:** Yushen Liu, Yanfu Zhang, Xugui Zhou

**发布时间:** 2025-11-16

**备注:** ISBI 2026

### GPT解析

### 总结

本文提出了一种名为TSODE的安全感知控制器，结合Thompson Sampling强化学习与神经常微分方程预测器，用于1型糖尿病的自动胰岛素输送，实现了既个性化又安全的血糖控制。

### 背景

1型糖尿病的自动胰岛素输送需要在不确定的饮食和生理变异情况下平衡血糖控制与安全性。现有的强化学习方法虽然可以实现自适应个性化，但难以同时保证安全性，存在餐前过量用药或叠加校正等风险。

### 目的

弥补个性化与风险感知血糖控制之间的差距，实现既个性化又安全的血糖控制。

### 方法

提出TSODE安全感知控制器，整合Thompson Sampling强化学习与神经常微分方程（NeuralODE）预测器。NeuralODE预测基于建议胰岛素剂量的短期血糖轨迹，通过校准层量化预测不确定性，以拒绝或缩放有风险的动作。

### 主要发现

在FDA批准的UVa/Padova模拟器（成人队列）中，TSODE实现了87.9%的时间在目标范围内，低于70 mg/dL的时间少于10%，性能优于相关基线方法。

### 结论

将自适应强化学习与校准的NeuralODE预测相结合可以实现可解释、安全和稳健的血糖调节。

### 翻译

1型糖尿病的自动胰岛素输送必须在不确定的饮食和生理变异情况下平衡血糖控制与安全性。虽然强化学习能够实现自适应个性化，但现有方法难以同时保证安全性，在实现个性化和风险感知的血糖控制方面存在差距，例如餐前过量用药或叠加校正。为了弥补这一差距，我们提出了TSODE，一种整合Thompson Sampling强化学习与神经常微分方程（NeuralODE）预测器的安全感知控制器。具体而言，NeuralODE预测基于建议胰岛素剂量的短期血糖轨迹，而校准层量化预测不确定性以拒绝或缩放有风险的动作。在FDA批准的UVa/Padova模拟器（成人队列）中，TSODE实现了87.9%的时间在目标范围内，且低于70 mg/dL的时间不到10%，性能优于相关基线。这些结果表明，将自适应强化学习与校准的NeuralODE预测相结合可以实现可解释、安全和稳健的血糖调节。


### 论文摘要

Automated insulin delivery for Type 1 Diabetes must balance glucose control and safety under uncertain meals and physiological variability. While reinforcement learning (RL) enables adaptive personalization, existing approaches struggle to simultaneously guarantee safety, leaving a gap in achieving both personalized and risk-aware glucose control, such as overdosing before meals or stacking corrections. To bridge this gap, we propose TSODE, a safety-aware controller that integrates Thompson Sampling RL with a Neural Ordinary Differential Equation (NeuralODE) forecaster to address this challenge. Specifically, the NeuralODE predicts short-term glucose trajectories conditioned on proposed insulin doses, while a conformal calibration layer quantifies predictive uncertainty to reject or scale risky actions. In the FDA-approved UVa/Padova simulator (adult cohort), TSODE achieved 87.9% time-in-range with less than 10% time below 70 mg/dL, outperforming relevant baselines. These results demonstrate that integrating adaptive RL with calibrated NeuralODE forecasting enables interpretable, safe, and robust glucose regulation.

---

## 163. More Than Irrational: Modeling Belief-Biased Agents

**论文链接:** [http://arxiv.org/abs/2511.12359v1](http://arxiv.org/abs/2511.12359v1)

**作者:** Yifan Zhu, Sammie Katt, Samuel Kaski

**发布时间:** 2025-11-15

**备注:** 13 pages, 8 figures. Accepted at the 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)

### GPT解析

### 总结

本文提出了一类计算理性(CR)用户模型，用于在偏见信念下最优行动的认知受限主体。通过明确建模有限记忆过程如何导致动态不一致和偏见信念状态，进而导致次优顺序决策，并开发了一种基于嵌套粒子滤波的高效在线推断方法，同时跟踪用户的潜在信念状态并估计认知限制。

### 背景

尽管AI及其相关技术呈爆炸式增长，但预测和推断用户或人类协作者的非最优行为仍然是一个重大挑战。这些行为通常不是非理性的结果，而是由内在认知局限和对世界的偏见信念所导致的理性决策。

### 目的

正式引入一类计算理性(CR)用户模型，用于在偏见信念下最优行动的认知受限主体，解决从被动观察中识别潜在用户特定限制和推断偏见信念状态的挑战。

### 方法

提出了一种基于嵌套粒子滤波的高效在线推断方法，同时跟踪用户的潜在信念状态并从未知认知限制的观察动作流中估计。

### 主要发现

(1)CR模型能够生成对应不同记忆容量水平的直观合理行为；(2)推断方法能够从有限观察(≤100步)中准确高效地恢复真实认知限制。

### 结论

这种方法为开发自适应AI助手提供了有原则的基础，使自适应辅助能够考虑用户的记忆限制。

### 翻译

尽管人工智能及其相关技术呈爆炸式增长，但预测和推断用户或人类协作者的非最优行为仍然是一个关键挑战。在许多情况下，这些行为并非非理性的结果，而是由内在认知局限和对世界的偏见信念所导致的理性决策。在本文中，我们正式引入了一类计算理性(CR)用户模型，用于在偏见信念下最优行动的认知受限主体。关键创新在于明确建模了有限记忆过程如何导致动态不一致和偏见信念状态，进而导致次优的顺序决策。我们解决了从被动观察中识别潜在用户特定限制和推断偏见信念状态的挑战。我们认为，对于我们形式化的具有明确参数化认知过程的CR模型族，这一挑战是可处理的。为支持我们的主张，我们提出了一种基于嵌套粒子滤波的高效在线推断方法，同时跟踪用户的潜在信念状态并从未知认知限制的观察动作流中估计。我们在使用记忆衰减作为认知限制例子的代表性导航任务中验证了我们的方法。通过模拟，我们表明(1)我们的CR模型能够生成对应不同记忆容量水平的直观合理行为，以及(2)我们的推断方法能够从有限观察(≤100步)中准确高效地恢复真实认知限制。我们进一步展示了这种方法如何为开发自适应AI助手提供了有原则的基础，使自适应辅助能够考虑用户的记忆限制。


### 论文摘要

Despite the explosive growth of AI and the technologies built upon it, predicting and inferring the sub-optimal behavior of users or human collaborators remains a critical challenge. In many cases, such behaviors are not a result of irrationality, but rather a rational decision made given inherent cognitive bounds and biased beliefs about the world. In this paper, we formally introduce a class of computational-rational (CR) user models for cognitively-bounded agents acting optimally under biased beliefs. The key novelty lies in explicitly modeling how a bounded memory process leads to a dynamically inconsistent and biased belief state and, consequently, sub-optimal sequential decision-making. We address the challenge of identifying the latent user-specific bound and inferring biased belief states from passive observations on the fly. We argue that for our formalized CR model family with an explicit and parameterized cognitive process, this challenge is tractable. To support our claim, we propose an efficient online inference method based on nested particle filtering that simultaneously tracks the user's latent belief state and estimates the unknown cognitive bound from a stream of observed actions. We validate our approach in a representative navigation task using memory decay as an example of a cognitive bound. With simulations, we show that (1) our CR model generates intuitively plausible behaviors corresponding to different levels of memory capacity, and (2) our inference method accurately and efficiently recovers the ground-truth cognitive bounds from limited observations ($\le 100$ steps). We further demonstrate how this approach provides a principled foundation for developing adaptive AI assistants, enabling adaptive assistance that accounts for the user's memory limitations.

---

## 164. Protein Secondary Structure Prediction Using 3D Graphs and Relation-Aware Message Passing Transformers

**论文链接:** [http://arxiv.org/abs/2511.13685v1](http://arxiv.org/abs/2511.13685v1)

**作者:** Disha Varshney, Samarth Garg, Sarthak Tyagi, Deeksha Varshney, Nayan Deep, Asif Ekbal

**发布时间:** 2025-11-17

**备注:** 40 pages

### GPT解析

### 总结

本研究提出了一种结合图神经网络和语言模型的蛋白质二级结构预测方法SSRGNet，通过利用蛋白质3D结构数据提高了预测性能。

### 背景

现有蛋白质二级结构预测方法通常使用大量未标记的氨基酸序列，但没有充分利用可用的蛋白质3D结构数据，而这对于理解蛋白质功能至关重要。

### 目的

开发一种能够明确利用蛋白质3D结构数据的方法，以更准确地预测蛋白质二级结构，进而帮助理解蛋白质的功能和关系。

### 方法

使用蛋白质残基图并引入序列或结构连接捕获空间信息；结合图神经网络(GNNs)和语言模型(LMs)；利用预训练的transformer-based蛋白质语言模型编码氨基酸序列；采用GCN和R-GCN等消息传递机制捕获几何特征；在特定节点附近区域使用卷积并堆叠多层卷积层。

### 主要发现

在NetSurfP-2.0数据集上的大量实验表明，提出的SSRGNet模型在f1分数上超越了基线方法。

### 结论

通过整合蛋白质3D结构数据和先进的深度学习技术，SSRGNet模型能够更准确地预测蛋白质二级结构，为理解蛋白质功能提供了更好的工具。

### 翻译

在本研究中，我们解决了从蛋白质一级序列预测二级结构的挑战性任务，这是预测三级结构的关键初步步骤，同时为理解蛋白质活性、关系和功能提供了重要见解。现有方法通常使用大量未标记的氨基酸序列集。然而，这些方法既没有明确捕获也没有利用可用的蛋白质3D结构数据，这被认定为决定蛋白质功能的关键因素。为此，我们利用蛋白质残基图并引入各种形式的序列或结构连接以捕获增强的空间信息。我们巧妙地结合了图神经网络(GNNs)和语言模型(LMs)，具体使用预训练的基于transformer的蛋白质语言模型来编码氨基酸序列，并采用GCN和R-GCN等消息传递机制来捕获蛋白质结构的几何特征。在特定节点附近的区域内(包括关系)使用卷积，我们堆叠多个卷积层以有效地从蛋白质的空间图中学习综合见解，揭示了其结构排列中的复杂相互关联和依赖关系。为了评估我们模型的性能，我们使用了NetSurfP-2.0提供的训练数据集，该数据集以3态和8态概述了二级结构。大量实验表明，我们提出的模型SSRGNet在f1分数上超越了基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决蛋白质二级结构预测问题，即从蛋白质一级序列预测其二级结构（如α-螺旋、β-折叠和卷曲）。这个问题在现实中非常重要，因为蛋白质二级结构是理解蛋白质功能、预测三级结构以及设计药物的关键步骤。蛋白质功能主要由其三维结构决定，而二级结构是理解这种三维结构的重要基础。准确预测二级结构可以帮助科学家更好地理解蛋白质如何工作，从而在疾病治疗和药物开发中发挥重要作用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法主要依赖未标记的氨基酸序列，而忽视了可获得的蛋白质3D结构数据，这些数据对蛋白质功能至关重要。作者借鉴了现有的工作：使用了图神经网络(GNNs)捕获蛋白质结构的几何特征，利用预训练的transformer-based蛋白质语言模型(DistilProtBert)编码氨基酸序列，并采用消息传递机制如GCN和R-GCN。作者的创新在于将蛋白质残基图与氨基酸链结合，通过引入各种形式的连接来捕获增强的空间信息，并将GNNs和LMs有效地结合起来。作者设计了三种模型架构（串联、并联和交叉）来融合两个编码器，并通过实验确定了最佳方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一种名为二级结构关系图神经网络(SSRGNet)的新架构，通过蛋白质残基图和氨基酸链协同利用序列信息和结构细节来促进蛋白质二级结构预测。整体实现流程包括：1) 序列编码：使用DistilProtBert模型对氨基酸序列进行编码；2) 图构建：将蛋白质3D结构可视化为图，引入三种类型的边（顺序关系R1、空间接近R2和局部环境R3）；3) 图特征提取：节点包含序列特征，边包含各种空间相互作用特征；4) 基于关系的图卷积：使用R-GCN进行消息传递；5) 融合块：将序列特征和图特征连接并通过MLP融合；6) 训练：使用交叉熵损失函数进行训练，预测每个氨基酸的二级结构标签。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出SSRGNet新架构，协同利用序列信息和结构细节；2) 丰富蛋白质残基图，整合不同类型的边捕获多种关系；3) 首次将关系消息传递整合到GNNs中用于蛋白质结构编码。相比之前工作的不同：1) 与仅使用序列信息的方法不同，SSRGNet同时利用序列和3D结构信息；2) 与现有蛋白质语言模型不同，SSRGNet能明确编码蛋白质结构；3) 与其他结构编码器增强的模型相比，SSRGNet通过关系感知的消息传递更好地捕获蛋白质结构中的边间相互作用；4) 设计了三种融合架构并通过实验确定了最佳方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了SSRGNet，一种结合蛋白质序列信息和3D结构关系的新型图神经网络框架，通过关系感知的消息传递机制显著提高了蛋白质二级结构预测的准确性。'}


### 论文摘要

In this study, we tackle the challenging task of predicting secondary structures from protein primary sequences, a pivotal initial stride towards predicting tertiary structures, while yielding crucial insights into protein activity, relationships, and functions. Existing methods often utilize extensive sets of unlabeled amino acid sequences. However, these approaches neither explicitly capture nor harness the accessible protein 3D structural data, which is recognized as a decisive factor in dictating protein functions. To address this, we utilize protein residue graphs and introduce various forms of sequential or structural connections to capture enhanced spatial information. We adeptly combine Graph Neural Networks (GNNs) and Language Models (LMs), specifically utilizing a pre-trained transformer-based protein language model to encode amino acid sequences and employing message-passing mechanisms like GCN and R-GCN to capture geometric characteristics of protein structures. Employing convolution within a specific node's nearby region, including relations, we stack multiple convolutional layers to efficiently learn combined insights from the protein's spatial graph, revealing intricate interconnections and dependencies in its structural arrangement. To assess our model's performance, we employed the training dataset provided by NetSurfP-2.0, which outlines secondary structure in 3-and 8-states. Extensive experiments show that our proposed model, SSRGNet surpasses the baseline on f1-scores.

---

## 165. FuseSampleAgg: Fused Neighbor Sampling and Aggregation for Mini-batch GNNs

**论文链接:** [http://arxiv.org/abs/2511.13645v1](http://arxiv.org/abs/2511.13645v1)

**作者:** Aleksandar Stanković

**发布时间:** 2025-11-17

**备注:** 15 pages. Code and reproducibility scripts: https://github.com/SV25-22/FuseSampleAgg

### GPT解析

### 总结

这篇论文提出了FuseSampleAgg，一个CUDA算子，它将邻居采样和平均聚合融合为单次操作，用于一跳和两跳GraphSAGE。

### 背景

图神经网络中的GraphSAGE算法通常需要邻居采样和聚合操作，这些操作可能存在内存和计算效率问题。

### 目的

提高GraphSAGE中邻居采样和聚合操作的效率，减少内存流量和计算开销。

### 方法

FuseSampleAgg CUDA算子通过消除块显式化和额外的内核启动，同时通过保存索引重放保留GraphSAGE平均语义。

### 主要发现

在Reddit、ogbn-arxiv和ogbn-products基准测试中（批量大小1024，启用自动混合精度），FuseSampleAgg实现了显著的性能提升（最高51倍）和内存减少（最高100倍）。

### 结论

FuseSampleAgg是一种高效的CUDA算子，可以显著提升GraphSAGE的性能，同时保持算法的语义正确性，并可与标准PyTorch优化器集成。

### 翻译

我们提出了FuseSampleAgg，一个CUDA算子，它将邻居采样和平均聚合融合为单次操作，用于一跳和两跳GraphSAGE。通过消除块显式化和额外的内核启动，FuseSampleAgg通过保存索引重放减少了内存流量和开销，同时保留了GraphSAGE平均语义。在Reddit、ogbn-arxiv和ogbn-products基准测试中（批量大小1024，启用自动混合精度），我们在ogbn-products上观察到高达51倍的步进时间加速，在Reddit上使用fanouts 10-10和15-10时约为4倍，在ogbn-arxiv上较大的fanouts时约为3.3倍，同时GPU内存峰值减少分别高达100倍、36倍和约3.5倍。该算子是确定性的，与标准PyTorch优化器集成，并提供可从CSV日志重现所有表格和图形的脚本。代码和脚本可在https://github.com/SV25-22/FuseSampleAgg获取。


### 论文摘要

We present FuseSampleAgg, a CUDA operator that fuses neighbor sampling and mean aggregation into a single pass for one and two hop GraphSAGE. By eliminating block materialization and extra kernel launches, FuseSampleAgg reduces memory traffic and overhead while preserving GraphSAGE mean semantics via saved index replay. Across the Reddit, ogbn-arxiv, and ogbn-products benchmarks (batch size 1024, automatic mixed precision enabled), we observe step time speedups up to 51x on ogbn-products, about 4x on Reddit with fanouts 10-10 and 15-10, and about 3.3x on ogbn-arxiv at larger fanouts, with peak GPU memory reductions up to 100x, 36x, and about 3.5x, respectively. The operator is deterministic, integrates with standard PyTorch optimizers, and ships with scripts that reproduce all tables and figures from CSV logs. Code and scripts are available at https://github.com/SV25-22/FuseSampleAgg.

---

## 166. Causal Inference, Biomarker Discovery, Graph Neural Network, Feature Selection

**论文链接:** [http://arxiv.org/abs/2511.13295v1](http://arxiv.org/abs/2511.13295v1)

**作者:** Chaowang Lan, Jingxin Wu, Yulong Yuan, Chuxun Liu, Huangyi Kang, Caihua Liu

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究开发了一种名为Causal-GNN的因果图神经网络方法，用于从高通量转录组数据中发现生物标志物，结合因果推断与图神经网络技术，提高了生物标志物的稳定性和预测准确性。

### 背景

从高通量转录组数据中发现生物标志物对推进精准医疗至关重要，但现有方法常忽略基因-基因调控关系，且在数据集间缺乏稳定性，导致虚假相关性与真实因果效应混淆。

### 目的

开发一种整合因果推断与多层图神经网络的方法，解决现有生物标志物发现方法中忽略基因调控关系和跨数据集稳定性不足的问题。

### 方法

开发Causal-GNN方法，整合因果推断与多层图神经网络，关键创新包括：结合因果效应估计以识别稳定生物标志物，以及基于GNN的倾向性评分机制，利用跨基因调控网络。

### 主要发现

该方法在四个不同数据集和四个独立分类器上均持续获得高预测准确性，并能识别出比传统方法更稳定的生物标志物。

### 结论

Causal-GNN为生物标志物发现提供了稳健、高效且具有生物学可解释性的工具，在多个医学学科中具有广泛的应用潜力。

### 翻译

从高通量转录组数据中发现生物标志物对于推进精准医疗至关重要。然而，现有方法常常忽略基因-基因调控关系，并且在数据集之间缺乏稳定性，导致将虚假相关性与真实的因果效应混淆。为了解决这些问题，我们开发了一种结合因果推断与多层图神经网络(GNN)的因果图神经网络(Causal-GNN)方法。关键创新是将因果效应估计纳入以识别稳定的生物标志物，同时利用跨基因调控网络的基于GNN的倾向性评分机制。实验结果表明，我们的方法在四个不同的数据集和四个独立的分类器上均能持续获得高预测准确性。此外，与传统方法相比，它能够识别出更稳定的生物标志物。我们的工作为生物标志物发现提供了一个稳健、高效且具有生物学可解释性的工具，显示出在多个医学学科中广泛应用的强大潜力。


### 论文摘要

Biomarker discovery from high-throughput transcriptomic data is crucial for advancing precision medicine. However, existing methods often neglect gene-gene regulatory relationships and lack stability across datasets, leading to conflation of spurious correlations with genuine causal effects. To address these issues, we develop a causal graph neural network (Causal-GNN) method that integrates causal inference with multi-layer graph neural networks (GNNs). The key innovation is the incorporation of causal effect estimation for identifying stable biomarkers, coupled with a GNN-based propensity scoring mechanism that leverages cross-gene regulatory networks. Experimental results demonstrate that our method achieves consistently high predictive accuracy across four distinct datasets and four independent classifiers. Moreover, it enables the identification of more stable biomarkers compared to traditional methods. Our work provides a robust, efficient, and biologically interpretable tool for biomarker discovery, demonstrating strong potential for broad application across medical disciplines.

---

## 167. Learning to Solve Resource-Constrained Project Scheduling Problems with Duration Uncertainty using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.13214v1](http://arxiv.org/abs/2511.13214v1)

**作者:** Guillaume Infantes, Stéphanie Roussel, Antoine Jacquet, Emmanuel Benazera

**发布时间:** 2025-11-17

**备注:** Accepted at ICTAI 2025 Conference

### GPT解析

### 总结

本文提出了一种结合图神经网络和深度强化学习的方法来解决具有不确定任务持续时间的资源受限项目调度问题，目标是最小化预期项目持续时间并生成可重复使用的基线计划，实验证明该方法在性能和泛化能力上具有优越性。

### 背景

资源受限项目调度问题(RCPSP)是一个经典的调度问题，由于其众多工业应用而受到广泛关注。然而，在实践中，任务持续时间存在不确定性，必须考虑这些不确定性才能提出弹性调度方案。

### 目的

解决具有不确定任务持续时间(使用已知概率建模)的RCPSP变体，旨在最小化整体预期项目持续时间，生成一个可以在工业环境中多次使用的基线计划，无论实际持续时间场景如何。

### 方法

利用图神经网络结合深度强化学习(DRL)开发有效的任务调度策略，该策略类似于优先级调度规则，并与串行调度生成方案配对以生成计划。

### 主要发现

在标准基准测试上的经验评估表明，该方法在性能和泛化能力方面具有优越性。

### 结论

开发的框架Wheatley已在公开在线上发布，以促进进一步研究和可重复性。

### 翻译

资源受限项目调度问题(RCPSP)是一个经典的调度问题，由于其众多工业应用而受到广泛关注。然而，在实践中，任务持续时间存在不确定性，必须考虑这些不确定性才能提出弹性调度方案。在本文中，我们解决了具有不确定任务持续时间(使用已知概率建模)的RCPSP变体，旨在最小化整体预期项目持续时间。我们的目标是生成一个基线计划，可以在工业环境中多次使用，无论实际持续时间场景如何。我们利用图神经网络结合深度强化学习(DRL)来开发有效的任务调度策略。该策略类似于优先级调度规则，并与串行调度生成方案配对以生成计划。我们在标准基准测试上的经验评估证明了该方法在性能和泛化能力方面的优越性。开发的框架Wheatley已在公开在线上发布，以促进进一步研究和可重复性。


### 论文摘要

The Resource-Constrained Project Scheduling Problem (RCPSP) is a classical scheduling problem that has received significant attention due to of its numerous applications in industry. However, in practice, task durations are subject to uncertainty that must be considered in order to propose resilient scheduling. In this paper, we address the RCPSP variant with uncertain tasks duration (modeled using known probabilities) and aim to minimize the overall expected project duration. Our objective is to produce a baseline schedule that can be reused multiple times in an industrial setting regardless of the actual duration scenario. We leverage Graph Neural Networks in conjunction with Deep Reinforcement Learning (DRL) to develop an effective policy for task scheduling. This policy operates similarly to a priority dispatch rule and is paired with a Serial Schedule Generation Scheme to produce a schedule. Our empirical evaluation on standard benchmarks demonstrates the approach's superiority in terms of performance and its ability to generalize. The developed framework, Wheatley, is made publicly available online to facilitate further research and reproducibility.

---

## 168. Warm-starting active-set solvers using graph neural networks

**论文链接:** [http://arxiv.org/abs/2511.13174v1](http://arxiv.org/abs/2511.13174v1)

**作者:** Ella J. Schmidtobreick, Daniel Arnström, Paul Häusner, Jens Sjölund

**发布时间:** 2025-11-17

**备注:** Under review, 15 pages, 8 figures

### GPT解析

### 总结

这篇论文提出了一种使用图神经网络(GNN)的学习优化方法，用于加速二次规划求解器，特别是在实时控制应用中。

### 背景

二次规划(QP)求解器广泛应用于实时控制和优化，但在时间关键场景中，计算成本常常限制了其应用。

### 目的

提出一种使用图神经网络(GNN)的学习优化方法，用于预测DAQP(双活跃集求解器)中的活跃集，从而加速求解过程。

### 方法

该方法通过将二次规划问题表示为二分图，利用其结构特性，学习识别最优活跃集，以便为求解器提供有效的热启动。使用图神经网络来预测活跃集，并与多层感知器(MLP)基线进行比较。

### 主要发现

1)在不同问题规模下，GNN相比冷启动能显著减少求解器迭代次数；2)GNN的性能与多层感知器(MLP)基线相当；3)在不同问题规模上训练的GNN能有效泛化到未见过的维度，展示了灵活性和可扩展性。

### 结论

结构感知学习在加速实时应用(如模型预测控制)中的优化具有巨大潜力，所提出的GNN方法能够有效提升二次规划求解器的效率。

### 翻译

二次规划(QP)求解器广泛应用于实时控制和优化，但其计算成本常常限制了在时间关键场景中的适用性。我们提出了一种使用图神经网络(GNN)的学习优化方法，用于预测双活跃集求解器DAQP中的活跃集。该方法通过将二次规划表示为二分图并利用其结构特性，学习识别最优活跃集，以便为求解器提供高效的热启动。在不同问题规模下，GNN相比冷启动能减少求解器迭代次数，而性能与多层感知器(MLP)基线相当。此外，在不同问题规模上训练的GNN能有效泛化到未见过的维度，展示了灵活性和可扩展性。这些结果突显了结构感知学习在加速实时应用(如模型预测控制)中优化的潜力。


### 论文摘要

Quadratic programming (QP) solvers are widely used in real-time control and optimization, but their computational cost often limits applicability in time-critical settings. We propose a learning-to-optimize approach using graph neural networks (GNNs) to predict active sets in the dual active-set solver DAQP. The method exploits the structural properties of QPs by representing them as bipartite graphs and learning to identify the optimal active set for efficiently warm-starting the solver. Across varying problem sizes, the GNN consistently reduces the number of solver iterations compared to cold-starting, while performance is comparable to a multilayer perceptron (MLP) baseline. Furthermore, a GNN trained on varying problem sizes generalizes effectively to unseen dimensions, demonstrating flexibility and scalability. These results highlight the potential of structure-aware learning to accelerate optimization in real-time applications such as model predictive control.

---

## 169. Self-Adaptive Graph Mixture of Models

**论文链接:** [http://arxiv.org/abs/2511.13062v1](http://arxiv.org/abs/2511.13062v1)

**作者:** Mohit Meena, Yash Punjabi, Abhishek A, Vishal Sharma, Mahesh Chandran

**发布时间:** 2025-11-17

**备注:** 17 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为自适应性图模型混合(SAGMM)的框架，用于自动选择和组合最适合的图神经网络模型，解决了当前图神经网络领域中模型选择的困难问题。

### 背景

图神经网络已成为处理图结构数据的有力工具，但近期研究表明其性能提升开始趋于平稳。在许多情况下，经过适当调整的成熟模型(如GCN和GAT)可以匹配甚至超越更复杂架构的性能。

### 目的

解决当前图神经网络领域的主要局限，即为给定的图任务或数据集选择最合适模型的困难，提供一个能够自动选择和组合最合适GNN模型的框架。

### 方法

提出SAGMM框架，从多样化架构池中自动选择和组合最适合的GNN模型；利用架构多样性和拓扑感知注意力门控机制，根据输入图结构自适应地为每个节点分配专家；包含剪枝机制减少活跃专家数量；探索专家模型预训练和冻结的训练效率变体。

### 主要发现

在16个基准数据集上(涵盖节点分类、图分类、回归和链接预测任务)的评估表明，SAGMM持续优于或匹配领先的GNN基线和基于混合的先前方法。

### 结论

SAGMM为现实世界的图学习提供了稳健且自适应的解决方案，能够有效解决图神经网络模型选择的困难问题。

### 翻译

图神经网络(GNNs)已成为处理图结构数据的有力工具，但近期研究表明其性能提升开始趋于平稳。在许多情况下，经过适当调整的成熟模型(如GCN和GAT)可以匹配甚至超越更复杂架构的性能。这一趋势突显了当前领域的主要局限：难以选择最适合给定图任务或数据集的模型。为此，我们提出自适应性图模型混合(SAGMM)，这是一个模块化和实用的框架，能够从多样化架构池中自动选择和组合最合适的GNN模型。与依赖单一基础模型变化的专家混合方法不同，SAGMM利用架构多样性和拓扑感知注意力门控机制，根据输入图结构自适应地为每个节点分配专家。为提高效率，SAGMM包含一个剪枝机制，可在不降低性能的情况下减少训练和推理过程中的活跃专家数量。我们还探索了一种训练效率变体，其中专家模型是预训练和冻结的，只训练门控和任务特定层。我们在16个涵盖节点分类、图分类、回归和链接预测任务的基准数据集上评估了SAGMM，证明其持续优于或匹配领先的GNN基线和基于混合的先前方法，为现实世界的图学习提供了稳健且自适应的解决方案。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as powerful tools for learning over graph-structured data, yet recent studies have shown that their performance gains are beginning to plateau. In many cases, well-established models such as GCN and GAT, when appropriately tuned, can match or even exceed the performance of more complex, state-of-the-art architectures. This trend highlights a key limitation in the current landscape: the difficulty of selecting the most suitable model for a given graph task or dataset. To address this, we propose Self-Adaptive Graph Mixture of Models (SAGMM), a modular and practical framework that learns to automatically select and combine the most appropriate GNN models from a diverse pool of architectures. Unlike prior mixture-of-experts approaches that rely on variations of a single base model, SAGMM leverages architectural diversity and a topology-aware attention gating mechanism to adaptively assign experts to each node based on the structure of the input graph. To improve efficiency, SAGMM includes a pruning mechanism that reduces the number of active experts during training and inference without compromising performance. We also explore a training-efficient variant in which expert models are pretrained and frozen, and only the gating and task-specific layers are trained. We evaluate SAGMM on 16 benchmark datasets covering node classification, graph classification, regression, and link prediction tasks, and demonstrate that it consistently outperforms or matches leading GNN baselines and prior mixture-based methods, offering a robust and adaptive solution for real-world graph learning.

---

## 170. 论文ID: 2511.13018v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.13018v1.json'

---

## 171. Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs

**论文链接:** [http://arxiv.org/abs/2511.13010v1](http://arxiv.org/abs/2511.13010v1)

**作者:** Jeongwhan Choi, Seungjun Park, Sumin Park, Sung-Bae Cho, Noseong Park

**发布时间:** 2025-11-17

**备注:** Accepted in AAAI 2026 for Oral Representation. This is the extended version including the appendix

### GPT解析

### 总结

该研究提出了一种名为'分形节点'的新概念，旨在解决图神经网络在平衡局部和全局信息方面的挑战，通过结合消息传递神经网络的高效性和图Transformer的长距离交互能力。

### 背景

图神经网络(GNNs)在学习图结构数据时往往难以平衡局部和全局信息。图Transformer虽能通过长距离交互解决此问题，但忽视了消息传递神经网络(MPNNs)的固有局部性和效率。

### 目的

提出'分形节点'概念，受真实网络中观察到的分形结构启发，旨在改善MPNNs的长距离依赖能力，同时保持其计算效率。

### 方法

基于图划分自然诱导分形结构的直觉，分形节点与原始节点共存，自适应聚合子图级别特征表示，强制子图内特征相似性，并通过提供直接快捷连接缓解过度压缩问题，实现子图级别表示的长距离传播。

### 主要发现

分形节点通过提供直接快捷连接，有效缓解了过度压缩问题，使子图级别表示能够长距离传播，从而改善了MPNNs的长距离依赖能力。

### 结论

该方法提高了MPNNs的表达能力，在与图Transformer相当或更好的性能下，通过改善MPNNs的长距离依赖关系，保持了MPNNs的计算效率。

### 翻译

图神经网络(GNNs)已成为学习图结构数据的有力工具，但常常难以平衡局部和全局信息。虽然图Transformer旨在通过实现长距离交互来解决这个问题，但它们往往忽视了消息传递神经网络(MPNNs)的固有局部性和效率。我们提出了一种名为分形节点的新概念，受真实网络中观察到的分形结构启发。我们的方法基于图划分自然诱导分形结构的直觉，其中子图通常反映完整图的连接模式。分形节点设计为与原始节点共存，并自适应地聚合子图级别的特征表示，从而强制每个子图内的特征相似性。我们证明分形节点通过提供直接快捷连接缓解了过度压缩问题，这些连接使子图级别表示能够长距离传播。实验结果表明，我们的方法提高了MPNNs的表达能力，在保持MPNNs计算效率的同时，通过改善MPNNs的长距离依赖关系，实现了与图Transformer相当或更好的性能。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as powerful tools for learning on graph-structured data, but often struggle to balance local and global information. While graph Transformers aim to address this by enabling long-range interactions, they often overlook the inherent locality and efficiency of Message Passing Neural Networks (MPNNs). We propose a new concept called fractal nodes, inspired by the fractal structure observed in real-world networks. Our approach is based on the intuition that graph partitioning naturally induces fractal structure, where subgraphs often reflect the connectivity patterns of the full graph. Fractal nodes are designed to coexist with the original nodes and adaptively aggregate subgraph-level feature representations, thereby enforcing feature similarity within each subgraph. We show that fractal nodes alleviate the over-squashing problem by providing direct shortcut connections that enable long-range propagation of subgraph-level representations. Experiment results show that our method improves the expressive power of MPNNs and achieves comparable or better performance to graph Transformers while maintaining the computational efficiency of MPNN by improving the long-range dependencies of MPNN.

---

## 172. Esim: EVM Bytecode Similarity Detection Based on Stable-Semantic Graph

**论文链接:** [http://arxiv.org/abs/2511.12971v1](http://arxiv.org/abs/2511.12971v1)

**作者:** Zhuo Chen, Gaoqiang Ji, Yiling He, Lei Wu, Yajin Zhou

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了Stable-Semantic Graph (SSG)新型EVM字节码表示方法和Esim原型系统，通过异构图神经网络实现智能合约相似性检测，在准确性和性能上均超越传统方法和现有工具。

### 背景

去中心化金融(DeFi)快速扩张但面临代码重用和有限开源贡献带来的挑战，包括抄袭和漏洞代码传播。传统基于指令流或控制流图的二进制相似性检测方法在处理低级EVM字节码和大量重用基本块时存在局限性，且多样化的Solidity编译器版本增加了检测难度。

### 目的

开发有效且准确的EVM字节码相似性检测方法，以识别相似智能合约，解决DeFi生态系统中代码重用带来的安全问题。

### 方法

提出Stable-Semantic Graph (SSG)表示方法，捕获'stable instructions'之间的关系；实现Esim原型系统，使用异构图神经网络将SSG嵌入矩阵进行相似性检测。

### 主要发现

Esim在SSG构建上表现优异，控制流F1得分为100%，数据流F1得分为95.16%；相似性检测性能达96.3% AUC，超越传统方法；大规模研究(分析6条EVM兼容链上的2,675,573个智能合约)显示Esim在漏洞搜索上优于Etherscan等现有工具。

### 结论

Esim通过创新的SSG表示方法和异构图神经网络技术，有效解决了EVM字节码相似性检测挑战，为区块链生态系统提供了识别相似代码和潜在漏洞的强大工具，有助于提高DeFi生态系统安全性。

### 翻译

去中心化金融(DeFi)正在经历快速扩张。然而，普遍的代码重用和有限的开源贡献给区块链生态系统带来了重大挑战，包括抄袭和易受攻击代码的传播。因此，迫切需要一种有效且准确的EVM字节码相似性检测方法来识别相似的合约。传统的二进制相似性检测方法通常基于指令流或控制流图(CFG)，这些方法由于EVM字节码的低级特性和大量重用的基本块等特定特征，在EVM字节码方面存在局限性。此外，高度多样化的Solidity编译器(Solc)版本进一步增加了准确相似性检测的复杂性。受这些挑战的启发，我们提出了一种名为Stable-Semantic Graph (SSG)的新型EVM字节码表示方法，它捕获了'stable instructions'(由我们的研究确定的特殊指令)之间的关系。此外，我们实现了一个名为Esim的原型，它使用异构图神经网络将SSG嵌入到矩阵中进行相似性检测。Esim在SSG构建方面表现出高准确性，控制流和数据流的F1得分分别达到100%和95.16%，其相似性检测性能达到96.3%的AUC，超过了传统方法。我们的大规模研究分析了六条EVM兼容链上一年期间内的2,675,573个智能合约，也证明Esim在漏洞搜索方面优于最先进的工具Etherscan。


### 论文摘要

Decentralized finance (DeFi) is experiencing rapid expansion. However, prevalent code reuse and limited open-source contributions have introduced significant challenges to the blockchain ecosystem, including plagiarism and the propagation of vulnerable code. Consequently, an effective and accurate similarity detection method for EVM bytecode is urgently needed to identify similar contracts. Traditional binary similarity detection methods are typically based on instruction stream or control flow graph (CFG), which have limitations on EVM bytecode due to specific features like low-level EVM bytecode and heavily-reused basic blocks. Moreover, the highly-diverse Solidity Compiler (Solc) versions further complicate accurate similarity detection.   Motivated by these challenges, we propose a novel EVM bytecode representation called Stable-Semantic Graph (SSG), which captures relationships between 'stable instructions' (special instructions identified by our study). Moreover, we implement a prototype, Esim, which embeds SSG into matrices for similarity detection using a heterogeneous graph neural network. Esim demonstrates high accuracy in SSG construction, achieving F1-scores of 100% for control flow and 95.16% for data flow, and its similarity detection performance reaches 96.3% AUC, surpassing traditional approaches. Our large-scale study, analyzing 2,675,573 smart contracts on six EVM-compatible chains over a one-year period, also demonstrates that Esim outperforms the SOTA tool Etherscan in vulnerability search.

---

## 173. Connectivity-Guided Sparsification of 2-FWL GNNs: Preserving Full Expressivity with Improved Efficiency

**论文链接:** [http://arxiv.org/abs/2511.12838v1](http://arxiv.org/abs/2511.12838v1)

**作者:** Rongqin Chen, Fan Mo, Pak Lon Ip, Shenghui Zhang, Dan Wu, Ye Li, Leong Hou U

**发布时间:** 2025-11-16

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

基于2-FWL测试的高阶图神经网络(HOGNNs)通过建模2节点和3节点交互实现优越表达能力，但计算复杂度高。Co-Sparsify是一种连接感知的稀疏化框架，可消除冗余计算同时保持完整表达能力，将3节点交互限制在双连通组件内，实验证明其与原始方法表达能力相当且性能更优。

### 背景

基于2-FWL测试的高阶图神经网络(HOGNNs)通过建模2节点和3节点交互实现了优越的表达能力，但计算复杂度很高。现有的效率方法通常以降低表达能力为代价来减轻计算负担。

### 目的

提出Co-Sparsify，一种连接感知的稀疏化框架，能够在消除冗余计算的同时保持完整的2-FWL表达能力，实现高表达能力和可扩展性的统一。

### 方法

Co-Sparsify将2节点消息传递限制在连通组件内，将3节点交互限制在双连通组件内，消除冗余计算而无需近似或采样。其关键见解是3节点交互仅在双连通组件内是表达上必要的，在这些组件之外，结构关系可以通过2节点消息传递或全局读取完全捕获。

### 主要发现

Co-Sparsified GNNs与2-FWL测试具有相同的表达能力。在PPGN上，Co-Sparsify在合成子结构计数任务上匹配或超过准确性，并在真实世界基准测试(ZINC, QM9)上实现了最先进的性能。

### 结论

高表达能力和可扩展性并非相互排斥：有原则的、拓扑引导的稀疏化能够实现具有理论保证的强大高效的GNN。

### 翻译

基于2-FWL测试的高阶图神经网络(HOGNNs)通过建模2节点和3节点交互实现了优越的表达能力，但计算复杂度很高。然而，这种计算负担通常被现有的效率方法所缓解，但代价是降低了表达能力。我们提出了Co-Sparsify，一个连接感知的稀疏化框架，它在保持完整2-FWL表达能力的同时消除了可证明的冗余计算。我们的关键见解是，3节点交互仅在双连通组件内是表达上必要的——这些是最大子图，其中每对节点都位于一个循环上。在这些组件之外，结构关系可以通过2节点消息传递或全局读取完全捕获，使得高阶建模变得不必要。Co-Sparsify将2节点消息传递限制在连通组件内，将3节点交互限制在双连通组件内，消除了计算而无需近似或采样。我们证明了Co-Sparsified GNNs与2-FWL测试具有相同的表达能力。在PPGN上，Co-Sparsify在合成子结构计数任务上匹配或超过准确性，并在真实世界基准测试(ZINC, QM9)上实现了最先进的性能。这项研究表明，高表达能力和可扩展性并非相互排斥：有原则的、拓扑引导的稀疏化能够实现具有理论保证的强大高效的GNN。


### 论文摘要

Higher-order Graph Neural Networks (HOGNNs) based on the 2-FWL test achieve superior expressivity by modeling 2- and 3-node interactions, but at $\mathcal{O}(n^3)$ computational cost. However, this computational burden is typically mitigated by existing efficiency methods at the cost of reduced expressivity. We propose \textbf{Co-Sparsify}, a connectivity-aware sparsification framework that eliminates \emph{provably redundant} computations while preserving full 2-FWL expressive power. Our key insight is that 3-node interactions are expressively necessary only within \emph{biconnected components} -- maximal subgraphs where every pair of nodes lies on a cycle. Outside these components, structural relationships can be fully captured via 2-node message passing or global readout, rendering higher-order modeling unnecessary. Co-Sparsify restricts 2-node message passing to connected components and 3-node interactions to biconnected ones, removing computation without approximation or sampling. We prove that Co-Sparsified GNNs are as expressive as the 2-FWL test. Empirically, on PPGN, Co-Sparsify matches or exceeds accuracy on synthetic substructure counting tasks and achieves state-of-the-art performance on real-world benchmarks (ZINC, QM9). This study demonstrates that high expressivity and scalability are not mutually exclusive: principled, topology-guided sparsification enables powerful, efficient GNNs with theoretical guarantees.

---

## 174. Event-CausNet: Unlocking Causal Knowledge from Text with Large Language Models for Reliable Spatio-Temporal Forecasting

**论文链接:** [http://arxiv.org/abs/2511.12769v1](http://arxiv.org/abs/2511.12769v1)

**作者:** Luyao Niu, Zepu Wang, Shuyi Guan, Yang Liu, Peng Sun

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出Event-CausNet框架，解决时空图神经网络在处理非重复事件(如事故)时可靠性下降的问题，通过结合大型语言模型和因果推理，显著提高了交通预测的准确性和可解释性。

### 背景

时空图神经网络在建模重复出现的交通模式方面表现出色，但在非重复事件期间可靠性急剧下降，这是因为GNN本质上是相关性模型，学习的历史模式在干扰期间引入的新因果因素下失效。

### 目的

解决GNN在处理非重复事件时的可靠性问题，将相关性模型与因果推理结合，提供更准确、可转移且可解释的解决方案，为关键干扰期间的交通管理提供更可靠的基础。

### 方法

提出Event-CausNet框架，使用大型语言模型量化非结构化事件报告，通过估计平均处理效应构建因果知识库，并使用新颖的因果注意力机制将此知识注入到双流GNN-LSTM网络中，以调整和增强预测。

### 主要发现

在真实世界数据集上的实验表明，Event-CausNet实现了稳健的性能，将预测误差降低了高达35.87%，显著优于最先进的基线方法。

### 结论

该框架弥合了相关性模型与因果推理之间的差距，提供了更准确、可转移且可解释的解决方案，同时提供了关键的可解释性，为关键干扰期间的交通管理提供了更可靠的基础。

### 翻译

虽然时空图神经网络在建模重复出现的交通模式方面表现出色，但在事故等非重复事件期间，其可靠性急剧下降。这种失败发生是因为GNN本质上是相关性模型，学习的历史模式在干扰期间引入的新因果因素下失效。为解决这一问题，我们提出了Event-CausNet，一个使用大型语言模型量化非结构化事件报告的框架，通过估计平均处理效应构建因果知识库，并使用新颖的因果注意力机制将此知识注入到双流GNN-LSTM网络中，以调整和增强预测。在真实世界数据集上的实验表明，Event-CausNet实现了稳健的性能，将预测误差降低了高达35.87%，显著优于最先进的基线方法。我们的框架弥合了相关性模型与因果推理之间的差距，提供了更准确、可转移且可解释的解决方案，同时提供了关键的可解释性，为关键干扰期间的交通管理提供了更可靠的基础。


### 论文摘要

While spatio-temporal Graph Neural Networks (GNNs) excel at modeling recurring traffic patterns, their reliability plummets during non-recurring events like accidents. This failure occurs because GNNs are fundamentally correlational models, learning historical patterns that are invalidated by the new causal factors introduced during disruptions. To address this, we propose Event-CausNet, a framework that uses a Large Language Model to quantify unstructured event reports, builds a causal knowledge base by estimating average treatment effects, and injects this knowledge into a dual-stream GNN-LSTM network using a novel causal attention mechanism to adjust and enhance the forecast. Experiments on a real-world dataset demonstrate that Event-CausNet achieves robust performance, reducing prediction error (MAE) by up to 35.87%, significantly outperforming state-of-the-art baselines. Our framework bridges the gap between correlational models and causal reasoning, providing a solution that is more accurate and transferable, while also offering crucial interpretability, providing a more reliable foundation for real-world traffic management during critical disruptions.

---

## 175. Improving the Generalisation of Learned Reconstruction Frameworks

**论文链接:** [http://arxiv.org/abs/2511.12730v1](http://arxiv.org/abs/2511.12730v1)

**作者:** Emilien Valat, Ozan Öktem

**发布时间:** 2025-11-16

**备注:** 11 pages, 8 figures

### GPT解析

### 总结

本文提出了一种混合神经网络架构GLM，用于解决X射线CT中的投影数据过滤问题，通过结合图和网格卷积，实现了比传统CNN更好的性能和泛化能力。

### 背景

在成像逆问题中应用数据驱动方法时，确保适当的泛化是一个关键挑战。在X射线CT中，广泛使用的卷积神经网络（CNN）对正弦图应用基于网格的卷积，但正弦图本质上位于线流形上，而非规则网格上，因此CNN并不适合此任务。

### 目的

开发一种能够有效处理断层扫描数据的神经网络架构，解决CNN在CT应用中的局限性，提高性能并减少参数需求。

### 方法

引入图数据结构表示CT采集几何和断层扫描数据，详细说明圆形锥束几何的图结构；提出GLM，一种利用图和网格卷积处理断层扫描数据的混合神经网络架构。

### 主要发现

GLM在结构相似性和峰值信噪比方面优于CNN，尽管只使用了少量可训练参数；与CNN相比，GLM需要更少的训练时间和内存，且内存需求扩展性更好；GLM对采集几何的未见变化具有稳健的泛化能力。

### 结论

GLM通过结合图和网格卷积，有效地解决了CNN在CT应用中的局限性，实现了更好的性能、更少的参数需求和更强的泛化能力。

### 翻译

确保适当的泛化是将数据驱动方法应用于成像逆问题时的关键挑战，因为重建图像的神经网络必须在各种数据集和采集几何形状上表现良好。在X射线计算机断层扫描（CT）中，卷积神经网络（CNN）被广泛用于过滤投影数据，但它们并不适合此任务，因为它们对正弦图应用基于网格的卷积，而正弦图本质上位于线流形上，而不是规则网格上。不了解几何结构的CNN隐式地依赖于它，并且需要大量参数，因为它们必须从数据中推断测量之间的关系，而不是从先验信息中。本文的贡献有两方面。首先，我们引入了一种图数据结构来表示CT采集几何和断层扫描数据，详细说明了圆形锥束几何的图结构。其次，我们提出了GLM，一种利用图和网格卷积处理断层扫描数据的混合神经网络架构。我们证明，当用结构相似性和峰值信噪比来量化性能时，GLM优于CNN，尽管GLM只使用了少量可训练参数。与CNN相比，GLM也需要更少的训练时间和内存，并且其内存需求扩展性更好。关键是，GLM对采集几何的未见变化表现出稳健的泛化能力，例如，仅在完全采样的CT数据上训练，然后在稀疏视图CT数据上测试。


### 论文摘要

Ensuring proper generalization is a critical challenge in applying data-driven methods for solving inverse problems in imaging, as neural networks reconstructing an image must perform well across varied datasets and acquisition geometries. In X-ray Computed Tomography (CT), convolutional neural networks (CNNs) are widely used to filter the projection data but are ill-suited for this task as they apply grid-based convolutions to the sinogram, which inherently lies on a line manifold, not a regular grid. The CNNs, unaware of the geometry, are implicitly tied to it and require an excessive amount of parameters as they must infer the relations between measurements from the data rather than from prior information.   The contribution of this paper is twofold. First, we introduce a graph data structure to represent CT acquisition geometries and tomographic data, providing a detailed explanation of the graph's structure for circular, cone-beam geometries. Second, we propose GLM, a hybrid neural network architecture that leverages both graph and grid convolutions to process tomographic data.   We demonstrate that GLM outperforms CNNs when performance is quantified in terms of structural similarity and peak signal-to-noise ratio, despite the fact that GLM uses only a fraction of the trainable parameters. Compared to CNNs, GLM also requires significantly less training time and memory, and its memory requirements scale better. Crucially, GLM demonstrates robust generalization to unseen variations in the acquisition geometry, like when training only on fully sampled CT data and then testing on sparse-view CT data.

---

## 176. Adaptive Graph Rewiring to Mitigate Over-Squashing in Mesh-Based GNNs for Fluid Dynamics Simulations

**论文链接:** [http://arxiv.org/abs/2511.12709v1](http://arxiv.org/abs/2511.12709v1)

**作者:** Sangwoo Seo, Hyunsung Kim, Jiwan Kim, Chanyoung Park

**发布时间:** 2025-11-16

**备注:** Preprint

### GPT解析

### 总结

本文提出了一种名为AdaMeshNet的新型框架，用于解决基于网格的图神经网络(GNN)中的过度挤压问题，通过自适应重连过程更准确地模拟流体动力学。

### 背景

基于网格的图神经网络已被公认为模拟流体动力学的一种有前景的方法。然而，网格细化技术会在基于网格的GNN中引起过度挤压问题，阻碍长程物理相互作用的捕捉。传统图重连方法通过添加新边缓解此问题，但它们假设远距离节点间瞬时相互作用，在物理上不现实。

### 目的

解决传统图重连方法在物理模拟中的局限性，更准确地建模物理相互作用的逐渐传播特性，提高基于网格的图神经网络在流体模拟中的预测准确性。

### 方法

提出AdaMeshNet框架，在消息传递过程中引入自适应重连过程。计算网格图中瓶颈节点的重连延迟分数，基于最短路径距离和速度差，动态选择新边被重连的消息传递层，实现网格图中的自适应重连。

### 主要发现

在基于网格的流体模拟的大量实验表明，AdaMeshNet优于传统的重连方法，有效地建模了物理相互作用的顺序特性，并实现了更准确的预测。

### 结论

AdaMeshNet通过自适应重连过程解决了基于网格的GNN中的过度挤压问题，能够更准确地捕捉物理相互作用的逐渐传播特性，从而提高了流体动力学模拟的准确性。

### 翻译

基于网格的图神经网络(GNN)模拟已被公认为模拟流体动力学的一种有前景的方法。然而，网格细化技术将更精细的分辨率分配给具有陡峭梯度的区域，这会在基于网格的GNN中引起过度挤压问题，从而阻碍了长程物理相互作用的捕捉。传统的图重连方法试图通过添加新边来缓解这个问题，但它们通常在将重连操作应用于GNN之前完成所有重连操作。这些方法在物理上是不现实的，因为它们假设了远距离节点之间的瞬时相互作用，并忽略了粒子之间的距离信息。为了解决这些局限性，我们提出了一种名为基于网格图神经网络的自适应图重连(AdaMeshNet)的新型框架，它在消息传递过程中引入了自适应重连过程，以建模物理相互作用的逐渐传播。我们的方法基于最短路径距离和速度差，计算网格图中瓶颈节点的重连接延迟分数。使用这个分数，它动态选择新边被重连的消息传递层，这可以在网格图中实现自适应重连。在基于网格的流体模拟的大量实验表明，AdaMeshNet优于传统的重连方法，有效地建模了物理相互作用的顺序特性，并实现了更准确的预测。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决基于网格的图神经网络在流体动力学模拟中的'过度压缩'问题，即信息在通过多个层传播时被逐渐压缩，难以捕捉长程物理交互。这个问题在流体动力学模拟中特别重要，因为边界层和湍流等区域需要高密度网格结构来准确模拟，但这些精细网格会加剧过度压缩问题。准确解决此问题对航空航天设计、气象预测、能源系统优化等领域有重要影响，可提高流体动力学模拟的准确性，为工程设计和科学研究提供更可靠的预测工具。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了过度压缩问题在MeshGraphNets中的理论表现，指出现有图重连方法都是在训练前完成所有重连操作，假设远距离节点可立即相互作用，这在物理上不现实。作者注意到真实流体中物理交互通过相邻粒子碰撞逐渐传播，存在延迟，受此启发设计了自适应图重连框架。该方法借鉴了PIORF中重连大速度差节点的思想，并使用了Ollivier-Ricci曲率来识别瓶颈节点，但创新性地在消息传递过程中动态重连新边，而非训练前完成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是在图神经网络的消息传递过程中动态重连新边，而非传统方法那样在训练前完成所有重连操作。通过计算'重连延迟分数'来决定在消息传递的哪一层添加新边，该分数综合考虑节点间距离和速度差，使模型能更真实地模拟物理交互中的延迟效应。整体流程包括：1)预处理阶段识别瓶颈节点并计算重连延迟分数；2)编码器阶段将网格特征映射为潜在向量；3)处理器阶段在每层消息传递中动态更新邻居集合、边和节点嵌入；4)解码器和状态更新器阶段预测物理量并更新下一时间步状态；5)网格更新生成新网格。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出自适应图重连框架AdaMeshNet，在消息传递过程中动态重连新边；2)设计重连延迟分数，综合考虑距离和速度差，确定添加新边的层；3)基于流体动力学中物理交互的逐渐传播特性，建模延迟效应。相比之前工作，本文方法在重连时机上从训练前改为消息传递过程中；显式建模了物理交互中的时间延迟；保留了节点间实际物理距离信息，使模拟更接近物理现实；实验表明在流体动力学模拟中性能优于传统方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种自适应图重连框架AdaMeshNet，通过在消息传递过程中动态添加新边并考虑物理交互的逐渐传播特性，有效解决了基于网格的图神经网络在流体动力学模拟中的过度压缩问题，显著提高了模拟的准确性。'}


### 论文摘要

Mesh-based simulation using Graph Neural Networks (GNNs) has been recognized as a promising approach for modeling fluid dynamics. However, the mesh refinement techniques which allocate finer resolution to regions with steep gradients can induce the over-squashing problem in mesh-based GNNs, which prevents the capture of long-range physical interactions. Conventional graph rewiring methods attempt to alleviate this issue by adding new edges, but they typically complete all rewiring operations before applying them to the GNN. These approaches are physically unrealistic, as they assume instantaneous interactions between distant nodes and disregard the distance information between particles. To address these limitations, we propose a novel framework, called Adaptive Graph Rewiring in Mesh-Based Graph Neural Networks (AdaMeshNet), that introduces an adaptive rewiring process into the message-passing procedure to model the gradual propagation of physical interactions. Our method computes a rewiring delay score for bottleneck nodes in the mesh graph, based on the shortest-path distance and the velocity difference. Using this score, it dynamically selects the message-passing layer at which new edges are rewired, which can lead to adaptive rewiring in a mesh graph. Extensive experiments on mesh-based fluid simulations demonstrate that AdaMeshNet outperforms conventional rewiring methods, effectively modeling the sequential nature of physical interactions and enabling more accurate predictions.

---

## 177. Symmetry-Aware Graph Metanetwork Autoencoders: Model Merging through Parameter Canonicalization

**论文链接:** [http://arxiv.org/abs/2511.12601v1](http://arxiv.org/abs/2511.12601v1)

**作者:** Odysseas Boufalis, Jorge Carrasco-Pollo, Joshua Rosenthal, Eduardo Terres-Caballero, Alejandro García-Castellanos

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出Scale Graph Metanetworks (ScaleGMNs)架构，通过同时利用排列对称性和缩放对称性，将神经网络映射到共享损失盆中，促进模型合并并避免高损失区域。

### 背景

神经网络参数化具有固有对称性，导致损失景观中存在多个等效最小值。先前工作仅通过解决排列对称性问题实现了网络对齐，但计算密集且未考虑缩放对称性。

### 目的

扩展先前工作，同时纳入缩放对称性，提出更高效的网络对齐方法，促进模型合并。

### 方法

提出ScaleGMNs架构，该架构对排列和参数缩放变换具有等变性。构建使用ScaleGMNs作为不变编码器的自编码器框架，无需明确解决分配问题即可对齐网络。

### 主要发现

实验表明，该方法能使隐式神经表示和卷积神经网络在排列和缩放对称性下对齐，使相似网络自然收敛到相同盆中，实现平滑线性插值并避免高损失区域。

### 结论

同时利用排列和缩放对称性可更有效对齐神经网络模型，促进模型合并并避免高损失区域。代码已在GitHub公开可用。

### 翻译

神经网络参数化表现出固有的对称性，导致损失景观中存在多个等效最小值。比例图元网络(ScaleGMNs)通过提出一种对排列和参数缩放变换等变的架构，明确利用了这些对称性。Ainsworth等人(2023)的先前工作通过计算密集型的组合分配问题解决了排列对称性问题，证明仅利用排列对称性就可以将网络映射到共享的损失盆中。在这项工作中，我们通过同时纳入缩放对称性扩展了他们的方法，提出了一个使用ScaleGMNs作为不变编码器的自编码器框架。实验结果表明，我们的方法可以在不明确解决分配问题的情况下，使隐式神经表示(INRs)和卷积神经网络(CNNs)在排列和缩放对称性下对齐。这种方法确保相似的网络自然收敛到相同的盆中，促进模型合并，即平滑线性插值同时避免高损失区域。代码在我们的GitHub存储库中公开可用。


### 论文摘要

Neural network parameterizations exhibit inherent symmetries that yield multiple equivalent minima within the loss landscape. Scale Graph Metanetworks (ScaleGMNs) explicitly leverage these symmetries by proposing an architecture equivariant to both permutation and parameter scaling transformations. Previous work by Ainsworth et al. (2023) addressed permutation symmetries through a computationally intensive combinatorial assignment problem, demonstrating that leveraging permutation symmetries alone can map networks into a shared loss basin. In this work, we extend their approach by also incorporating scaling symmetries, presenting an autoencoder framework utilizing ScaleGMNs as invariant encoders. Experimental results demonstrate that our method aligns Implicit Neural Representations (INRs) and Convolutional Neural Networks (CNNs) under both permutation and scaling symmetries without explicitly solving the assignment problem. This approach ensures that similar networks naturally converge within the same basin, facilitating model merging, i.e., smooth linear interpolation while avoiding regions of high loss. The code is publicly available on our GitHub repository.

---

## 178. Hierarchical Frequency-Decomposition Graph Neural Networks for Road Network Representation Learning

**论文链接:** [http://arxiv.org/abs/2511.12507v1](http://arxiv.org/abs/2511.12507v1)

**作者:** Jingtian Ma, Jingyuan Wang, Leong Hou U

**发布时间:** 2025-11-16

### GPT解析

### 总结

HiFiNet是一种创新的分层频率分解图神经网络，通过统一空间和谱建模方法解决了道路网络表示学习中的挑战，能够同时捕捉道路网络的全局趋势和局部变化，并在多个任务上表现出优越的性能和泛化能力。

### 背景

道路网络是智能交通系统及其相关应用的关键基础设施。由于交通模式中空间结构和频率特性之间存在复杂的相互作用，道路网络的有效表示学习具有挑战性。现有的图神经网络主要分为两类：基于空间的方法捕获局部拓扑但倾向于过度平滑表示，基于谱的方法分析全局频率分量但常常忽略局部变化，这种空间-谱失配限制了建模能力。

### 目的

为了弥合基于空间的方法和基于谱的方法之间的差距，提出一种统一空间和谱建模的新型图神经网络，以有效捕捉道路网络中的全局趋势和局部变化。

### 方法

提出HiFiNet，一种分层频率分解图神经网络。该方法构建虚拟节点的多级层次结构以实现局部频率分析，并采用分解-更新-重构框架，结合拓扑感知图变换器，分别建模和融合低频和高频信号。

### 主要发现

HiFiNet在理论上得到证明，并在多个真实世界数据集上的四个下游任务中经验验证，结果表明其在捕获有效的道路网络表示方面表现出优越的性能和泛化能力。

### 结论

HiFiNet能够有效解决道路网络表示学习中的空间-谱失配问题，通过统一空间和谱建模方法，成功捕捉了道路网络中的全局趋势和局部变化特征。

### 翻译

道路网络是支撑智能交通系统及其相关应用的关键基础设施。由于交通模式中空间结构和频率特性之间的复杂相互作用，道路网络的有效表示学习仍然具有挑战性。现有的用于建模道路网络的图神经网络主要分为两种范式：捕获局部拓扑但倾向于过度平滑表示的基于空间的方法，以及分析全局频率分量但常常忽略局部变化的基于谱的方法。这种空间-谱失配限制了它们对同时呈现粗略全局趋势和细粒度局部波动的道路网络的建模能力。为了弥合这一差距，我们提出了HiFiNet，一种统一空间和谱建模的新型分层频率分解图神经网络。HiFiNet构建虚拟节点的多级层次结构以实现局部频率分析，并采用分解-更新-重构框架，结合拓扑感知图变换器，分别建模和融合低频和高频信号。在四个下游任务上的多个真实世界数据集上得到理论证明和经验验证，HiFiNet在捕获有效的道路网络表示方面表现出优越的性能和泛化能力。


### 论文摘要

Road networks are critical infrastructures underpinning intelligent transportation systems and their related applications. Effective representation learning of road networks remains challenging due to the complex interplay between spatial structures and frequency characteristics in traffic patterns. Existing graph neural networks for modeling road networks predominantly fall into two paradigms: spatial-based methods that capture local topology but tend to over-smooth representations, and spectral-based methods that analyze global frequency components but often overlook localized variations. This spatial-spectral misalignment limits their modeling capacity for road networks exhibiting both coarse global trends and fine-grained local fluctuations. To bridge this gap, we propose HiFiNet, a novel hierarchical frequency-decomposition graph neural network that unifies spatial and spectral modeling. HiFiNet constructs a multi-level hierarchy of virtual nodes to enable localized frequency analysis, and employs a decomposition-updating-reconstruction framework with a topology-aware graph transformer to separately model and fuse low- and high-frequency signals. Theoretically justified and empirically validated on multiple real-world datasets across four downstream tasks, HiFiNet demonstrates superior performance and generalization ability in capturing effective road network representations.

---

## 179. Task-Aware Retrieval Augmentation for Dynamic Recommendation

**论文链接:** [http://arxiv.org/abs/2511.12495v1](http://arxiv.org/abs/2511.12495v1)

**作者:** Zhen Tao, Xinke Jiang, Qingshuai Feng, Haoyu Zhang, Lun Du, Yuchen Fang, Hao Miao, Bangquan Xie, Qingqiang Sun

**发布时间:** 2025-11-16

**备注:** AAAI 2026

### GPT解析

### 总结

本研究提出了一种名为TarDGR的任务感知检索增强框架，用于解决动态推荐系统中图神经网络微调时的泛化问题。该框架通过任务感知评估机制和图变换器模型，能够识别并融合相关的历史子图，从而提高模型捕捉 evolving 用户偏好的能力。

### 背景

动态推荐系统通过建模时间序列行为数据中的用户-物品交互来提供个性化建议。现有方法利用预训练的动态图神经网络(GNNs)学习用户-物品表示，但在这些图上微调GNNs时，由于预训练和微调阶段之间的时间差异，常常导致泛化问题，限制了模型捕捉 evolving 用户偏好的能力。

### 目的

解决动态推荐系统中图神经网络微调时的泛化问题，提高模型捕捉 evolving 用户偏好的能力。

### 方法

提出TarDGR框架，包含：(1)任务感知评估机制，用于识别语义相关的历史子图，构建无需人工标注的任务特定数据集；(2)基于图变换器的任务感知模型，整合语义和结构编码评估子图相关性；(3)在推理过程中检索并融合任务感知子图与查询子图，丰富表示并减轻时间泛化问题。

### 主要发现

在多个大规模动态图数据集上的实验表明，TarDGR始终优于最先进的方法，具有卓越的准确性和泛化能力。

### 结论

TarDGR框架通过任务感知检索增强有效解决了动态推荐系统中图神经网络微调时的泛化问题，提高了模型捕捉 evolving 用户偏好的能力，为动态推荐系统提供了新的解决方案。

### 翻译

动态推荐系统旨在通过建模时间序列行为数据中的用户-物品交互来提供个性化建议。最近的研究利用预训练的动态图神经网络(GNNs)在时间快照图上学习用户-物品表示。然而，在这些图上微调GNNs常常会导致泛化问题，这是由于预训练和微调阶段之间的时间差异造成的，这限制了模型捕捉 evolving 用户偏好的能力。为了解决这个问题，我们提出了TarDGR，一个任务感知的检索增强框架，旨在通过整合任务感知模型和检索增强来提高泛化能力。具体来说，TarDGR引入了一个任务感知评估机制来识别语义相关的历史子图，使得能够构建无需人工标注的任务特定数据集。它还提出了一个基于图变换器的任务感知模型，该模型整合了语义和结构编码来评估子图相关性。在推理过程中，TarDGR检索并将任务感知子图与查询子图融合，丰富其表示并减轻时间泛化问题。在多个大规模动态图数据集上的实验表明，TarDGR始终优于最先进的方法，大量的实证证据证明了其卓越的准确性和泛化能力。


### 论文摘要

Dynamic recommendation systems aim to provide personalized suggestions by modeling temporal user-item interactions across time-series behavioral data. Recent studies have leveraged pre-trained dynamic graph neural networks (GNNs) to learn user-item representations over temporal snapshot graphs. However, fine-tuning GNNs on these graphs often results in generalization issues due to temporal discrepancies between pre-training and fine-tuning stages, limiting the model's ability to capture evolving user preferences. To address this, we propose TarDGR, a task-aware retrieval-augmented framework designed to enhance generalization capability by incorporating task-aware model and retrieval-augmentation. Specifically, TarDGR introduces a Task-Aware Evaluation Mechanism to identify semantically relevant historical subgraphs, enabling the construction of task-specific datasets without manual labeling. It also presents a Graph Transformer-based Task-Aware Model that integrates semantic and structural encodings to assess subgraph relevance. During inference, TarDGR retrieves and fuses task-aware subgraphs with the query subgraph, enriching its representation and mitigating temporal generalization issues. Experiments on multiple large-scale dynamic graph datasets demonstrate that TarDGR consistently outperforms state-of-the-art methods, with extensive empirical evidence underscoring its superior accuracy and generalization capabilities.

---

## 180. Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection

**论文链接:** [http://arxiv.org/abs/2511.12460v1](http://arxiv.org/abs/2511.12460v1)

**作者:** Changzeng Fu, Shiwen Zhao, Yunze Zhang, Zhongquan Jian, Shiqi Zhao, Chaoran Liu

**发布时间:** 2025-11-16

**备注:** AAAI 2026 accepted

### GPT解析

### 总结

本文提出了一种名为P³HF的新型抑郁症检测方法，通过个性引导的表征学习、超图-Transformer架构和事件级别域解耦三个创新点，解决了现有方法在建模个体差异和跨模态时间依赖性方面的挑战，在MPDD-Young数据集上实现了约10%的性能提升。

### 背景

抑郁症是全球性的心理健康挑战，需要高效可靠的自动化检测方法。然而，当前基于Transformer或图神经网络(GNN)的多模态抑郁症检测方法在建模个体差异和跨模态时间依赖性方面面临重大挑战。

### 目的

开发一种能够有效建模个体差异和跨模态时间依赖性的抑郁症检测方法，提高检测的准确性和可靠性，增强在不同行为上下文中的泛化能力。

### 方法

提出P³HF网络，包含三个关键创新：(1)个性引导的表征学习，使用大型语言模型将离散个体特征转化为上下文描述进行个性化编码；(2)超图-Transformer架构，建模高阶跨模态时间关系；(3)事件级别域解耦，使用对比学习提高跨行为上下文的泛化能力。

### 主要发现

在MPDD-Young数据集上，P³HF在二元和三元抑郁症分类任务上的准确率和加权F1分数比现有方法提高了约10%。消融研究验证了每个架构组件的独立贡献，证实个性引导的表征学习和高阶超图推理对于生成稳健的、个体感知的抑郁相关表征都是必不可少的。

### 结论

P³HF通过个性引导的表征学习和高阶超图推理，能够更好地捕捉个体差异和跨模态时间关系，在不同行为上下文中具有更好的泛化能力，是一种有效的抑郁症检测方法。

### 翻译

抑郁症代表着全球性的心理健康挑战，需要高效可靠的自动化检测方法。当前基于Transformer或图神经网络(GNN)的多模态抑郁症检测方法在建模个体差异和跨模态时间依赖性方面面临重大挑战。因此，我们提出了P³HF（个性引导的公私域解耦超图-Transformer网络），具有三个关键创新：(1)使用大型语言模型进行个性引导的表征学习，将离散个体特征转化为上下文描述进行个性化编码；(2)超图-Transformer架构，建模高阶跨模态时间关系；(3)事件级别域解耦，使用对比学习提高跨行为上下文的泛化能力。在MPDD-Young数据集上的实验表明，P³HF在二元和三元抑郁症分类任务上的准确率和加权F1分数比现有方法提高了约10%。大量的消融研究验证了每个架构组件的独立贡献，证实个性引导的表征学习和高阶超图推理对于生成稳健的、个体感知的抑郁相关表征都是必不可少的。代码已在https://github.com/hacilab/P3HF发布。


### 论文摘要

Depression represents a global mental health challenge requiring efficient and reliable automated detection methods. Current Transformer- or Graph Neural Networks (GNNs)-based multimodal depression detection methods face significant challenges in modeling individual differences and cross-modal temporal dependencies across diverse behavioral contexts. Therefore, we propose P$^3$HF (Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network) with three key innovations: (1) personality-guided representation learning using LLMs to transform discrete individual features into contextual descriptions for personalized encoding; (2) Hypergraph-Former architecture modeling high-order cross-modal temporal relationships; (3) event-level domain disentanglement with contrastive learning for improved generalization across behavioral contexts. Experiments on MPDD-Young dataset show P$^3$HF achieves around 10\% improvement on accuracy and weighted F1 for binary and ternary depression classification task over existing methods. Extensive ablation studies validate the independent contribution of each architectural component, confirming that personality-guided representation learning and high-order hypergraph reasoning are both essential for generating robust, individual-aware depression-related representations. The code is released at https://github.com/hacilab/P3HF.

---

## 181. VISAGNN: Versatile Staleness-Aware Efficient Training on Large-Scale Graphs

**论文链接:** [http://arxiv.org/abs/2511.12434v1](http://arxiv.org/abs/2511.12434v1)

**作者:** Rui Xue

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种名为VISAGNN的新型陈旧性感知图神经网络，通过动态自适应地将陈旧性标准融入GNN训练过程，有效解决了历史嵌入的陈旧性问题，提高了大规模图神经网络训练的效率和准确性。

### 背景

图神经网络在图表示学习和实际应用中表现出色，但在大规模图上训练深层GNN面临邻居爆炸问题。虽然历史嵌入方法可减少计算和内存成本并保持模型表达能力，但其陈旧性往往引入显著偏差，成为影响模型性能的瓶颈。

### 目的

提出一种新型的通用陈旧性感知图神经网络(VISAGNN)，动态且自适应地将陈旧性标准融入大规模GNN训练过程，以减轻陈旧嵌入的负面影响。

### 方法

将陈旧性嵌入到消息传递机制、损失函数和训练期间的历史嵌入中，使模型能够自适应地减轻陈旧嵌入的负面影响，减少估计误差并提高下游准确性。

### 主要发现

全面的实验证明了VISAGNN在克服现有历史嵌入技术的陈旧性问题方面的有效性，在大规模基准测试中展示了卓越的性能和效率，并实现了显著更快的收敛速度。

### 结论

VISAGNN能有效解决历史嵌入的陈旧性问题，在大规模图神经网络训练中表现优越，具有更高的性能、效率和收敛速度。

### 翻译

图神经网络(GNNs)在图表示学习和各种实际应用中已展现出卓越的成功。然而，由于在大规模图上训练时面临邻居爆炸问题，扩展更深层的GNNs具有挑战性。为缓解这一问题，一类有前景的GNN训练算法利用历史嵌入来减少计算和内存成本，同时保持模型的表达能力。这些方法利用历史嵌入处理批外节点，有效近似全批量训练而不丢失任何邻居信息——这是传统采样方法的一个局限性。然而，这些历史嵌入的陈旧性常常引入显著偏差，成为一个可能对模型性能产生不利影响的瓶颈。在本文中，我们提出了一种新型的通用陈旧性感知图神经网络，名为VISAGNN，它动态且自适应地将陈旧性标准融入大规模GNN训练过程。通过将陈旧性嵌入到训练期间的消息传递机制、损失函数和历史嵌入中，我们的方法使模型能够自适应地减轻陈旧嵌入的负面影响，从而减少估计误差并提高下游准确性。全面的实验证明了我们的方法在克服现有历史嵌入技术的陈旧性问题方面的有效性，展示了其在大型基准测试上的卓越性能和效率，以及显著更快的收敛速度。


### 论文摘要

Graph Neural Networks (GNNs) have shown exceptional success in graph representation learning and a wide range of real-world applications. However, scaling deeper GNNs poses challenges due to the neighbor explosion problem when training on large-scale graphs. To mitigate this, a promising class of GNN training algorithms utilizes historical embeddings to reduce computation and memory costs while preserving the expressiveness of the model. These methods leverage historical embeddings for out-of-batch nodes, effectively approximating full-batch training without losing any neighbor information-a limitation found in traditional sampling methods. However, the staleness of these historical embeddings often introduces significant bias, acting as a bottleneck that can adversely affect model performance. In this paper, we propose a novel VersatIle Staleness-Aware GNN, named VISAGNN, which dynamically and adaptively incorporates staleness criteria into the large-scale GNN training process. By embedding staleness into the message passing mechanism, loss function, and historical embeddings during training, our approach enables the model to adaptively mitigate the negative effects of stale embeddings, thereby reducing estimation errors and enhancing downstream accuracy. Comprehensive experiments demonstrate the effectiveness of our method in overcoming the staleness issue of existing historical embedding techniques, showcasing its superior performance and efficiency on large-scale benchmarks, along with significantly faster convergence.

---

## 182. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs

**论文链接:** [http://arxiv.org/abs/2511.12423v1](http://arxiv.org/abs/2511.12423v1)

**作者:** Jiaji Ma, Puja Trivedi, Danai Koutra

**发布时间:** 2025-11-16

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了GRAPHTEXTACK，第一个针对大型语言模型增强的图神经网络(LLM-enhanced GNNs)的黑盒、多模态节点注入攻击方法，通过精心设计的节点注入来降低模型性能。

### 背景

文本属性图(TAGs)结合了结构化和文本节点信息，广泛应用于多个领域。最近的研究将大型语言模型(LLMs)与图神经网络(GNNs)集成，共同建模语义和结构，实现了最先进的性能。然而，这种集成引入了双重脆弱性：GNNs对结构扰动敏感，而LLM衍生特征易受提示注入和对抗性措辞影响。

### 目的

解决现有对抗攻击的局限性，包括单模态攻击对LLM增强GNNs性能影响有限，以及许多攻击假设不切实际的能力，如白盒访问或直接修改图数据。

### 方法

提出GRAPHTEXTACK，一种黑盒、多模态节点注入攻击，通过注入具有精心设计的结构和语义的节点来降低模型性能。使用进化优化框架和多目标适应度函数，平衡局部预测干扰和全局图影响，无需依赖模型内部或代理模型。

### 主要发现

现有单模态攻击对LLM增强的GNNs性能影响有限；GRAPHTEXTACK在五个数据集和两个最先进的LLM增强GNN模型上显著优于12个强基线方法。

### 结论

GRAPHTEXTACK是第一个针对LLM增强GNNs的黑盒、多模态节点注入攻击，能够在现实威胁模型下有效降低模型性能，无需访问模型内部或使用代理模型。

### 翻译

文本属性图(TAGs)结合结构化和文本节点信息，在许多领域普遍存在。最近的工作将大型语言模型(LLMs)与图神经网络(GNNs)集成，共同建模语义和结构，产生了更通用和更具表达能力的模型，在TAG基准测试上实现了最先进的性能。然而，这种集成引入了双重脆弱性：GNNs对结构扰动敏感，而LLM衍生的特征易受提示注入和对抗性措辞的影响。虽然现有的对抗攻击主要独立地扰动结构或文本，但我们发现单模态攻击对LLM增强的GNNs只会造成适度的性能下降。此外，许多现有攻击假设了不切实际的能力，如白盒访问或直接修改图数据。为了解决这些差距，我们提出了GRAPHTEXTACK，第一个针对LLM增强GNNs的黑盒、多模态节点注入攻击。GRAPHTEXTACK注入具有精心设计的结构和语义的节点以降低模型性能，在现实威胁模型下运行，不依赖模型内部或代理模型。为了导航连通性和特征分配的组合、不可微的搜索空间，GRAPHTEXTACK引入了一种新的进化优化框架，具有多目标适应度函数，平衡局部预测干扰和全局图影响。在五个数据集和两个最先进的LLM增强GNN模型上的大量实验表明，GRAPHTEXTACK显著优于12个强基线方法。


### 论文摘要

Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.

---

## 183. Equivariant Atomic and Lattice Modeling Using Geometric Deep Learning for Crystal Structure Optimization

**论文链接:** [http://arxiv.org/abs/2511.12243v1](http://arxiv.org/abs/2511.12243v1)

**作者:** Ziduo Yang, Yi-Ming Zhao, Xian Wang, Wei Zhuo, Xiaoqing Liu, Lei Shen

**发布时间:** 2025-11-15

### GPT解析

### 总结

E3Relax是一种端到端等变图神经网络，可直接将未弛豫晶体映射到其弛豫结构，解决了传统DFT计算密集型和现有机器学习方法忽略晶格向量的问题。

### 背景

结构优化对于可靠的材料属性计算至关重要，但传统基于密度泛函理论(DFT)的从头计算方法计算密集型。

### 目的

开发一种端到端机器学习方法，能够直接预测晶体的弛豫结构，同时考虑原子和晶格向量，提高计算效率并保持准确性。

### 方法

E3Relax将原子和晶格向量都提升为具有标量-矢量特征的图节点，实现原子位移和晶格变形的统一对称建模，并采用分层监督策略使网络每一层都做出物理上有意义的改进。

### 主要发现

在四个基准数据集上评估表明，E3Relax实现了显著的准确性和效率，通过DFT验证显示其预测的结构在能量上有利，适合作为高质量初始构型加速DFT计算。

### 结论

E3Relax提供了一种高效、准确的端到端方法来预测晶体结构的弛豫状态，解决了传统方法计算量大和现有机器学习方法忽略晶格向量的问题。

### 翻译

结构优化（产生弛豫结构，即最小能量状态）对于可靠的材料属性计算至关重要，然而传统的从头计算方法（如密度泛函理论DFT）计算密集。机器学习(ML)已出现以缓解这一瓶颈，但存在两个主要限制：(i)现有模型主要在原子上操作，尽管晶格向量在结构优化中起着关键作用，但它们被隐式处理；(ii)它们通常依赖多阶段、非端到端的工作流程，容易导致误差累积。在这里，我们提出了E3Relax，一种端到端等变图神经网络，将未弛豫晶体直接映射到其弛豫结构。E3Relax将原子和晶格向量都提升为具有标量-矢量特征的图节点，实现了原子位移和晶格变形的统一且保持对称性的建模。分层监督策略迫使每一层网络深度都做出物理上有意义的改进，模仿DFT的渐进收敛，同时保持完全端到端的流程。我们在四个基准数据集上评估了E3Relax，并证明它实现了显著的准确性和效率。通过DFT验证，我们显示E3Relax预测的结构在能量上有利，使它们适合作为高质量初始构型来加速DFT计算。


### 论文摘要

Structure optimization, which yields the relaxed structure (minimum-energy state), is essential for reliable materials property calculations, yet traditional ab initio approaches such as density-functional theory (DFT) are computationally intensive. Machine learning (ML) has emerged to alleviate this bottleneck but suffers from two major limitations: (i) existing models operate mainly on atoms, leaving lattice vectors implicit despite their critical role in structural optimization; and (ii) they often rely on multi-stage, non-end-to-end workflows that are prone to error accumulation. Here, we present E3Relax, an end-to-end equivariant graph neural network that maps an unrelaxed crystal directly to its relaxed structure. E3Relax promotes both atoms and lattice vectors to graph nodes endowed with dual scalar-vector features, enabling unified and symmetry-preserving modeling of atomic displacements and lattice deformations. A layer-wise supervision strategy forces every network depth to make a physically meaningful refinement, mimicking the incremental convergence of DFT while preserving a fully end-to-end pipeline. We evaluate E3Relax on four benchmark datasets and demonstrate that it achieves remarkable accuracy and efficiency. Through DFT validations, we show that the structures predicted by E3Relax are energetically favorable, making them suitable as high-quality initial configurations to accelerate DFT calculations.

---

## 184. ViTE: Virtual Graph Trajectory Expert Router for Pedestrian Trajectory Prediction

**论文链接:** [http://arxiv.org/abs/2511.12214v1](http://arxiv.org/abs/2511.12214v1)

**作者:** Ruochen Li, Zhanxing Zhu, Tanqiu Qiao, Hubert P. H. Shum

**发布时间:** 2025-11-15

### GPT解析

### 总结

本文提出了一种名为ViTE的新型行人轨迹预测框架，通过虚拟图和专家路由器模块，有效平衡了计算效率和预测准确性，在多个基准测试上取得了最先进性能。

### 背景

行人轨迹预测对自动驾驶、监控系统和城市规划的安全至关重要。早期方法关注一跳成对关系，而最近研究通过堆叠多层图神经网络捕获高阶交互，但面临层数不足导致感受野受限或层数过多导致计算成本过高的权衡。

### 目的

开发一个能够自适应建模显式一跳交互和隐式高阶依赖性的有效模型，而非仅仅依靠架构深度。

### 方法

提出ViTE框架，包含两个关键模块：1)虚拟图，引入动态虚拟节点建模长距离和高阶交互，无需深层GNN堆叠；2)专家路由器，使用专家混合设计根据社交上下文自适应选择交互专家，实现灵活可扩展的推理。

### 主要发现

在ETH/UCY、NBA和SDD三个基准测试上的实验表明，ViTE方法始终达到最先进性能，验证了其有效性和实际效率。

### 结论

ViTE框架通过结合虚拟图和专家路由器，有效解决了行人轨迹预测中的计算效率和预测准确性的权衡问题，为实际应用提供了高效解决方案。

### 翻译

行人轨迹预测对确保自动驾驶、监控系统和城市规划应用的安全至关重要。虽然早期方法主要关注一跳成对关系，但最近的研究尝试通过堆叠多个图神经网络(GNN)层来捕获高阶交互。然而，这些方法面临一个基本权衡：层数不足可能导致感受野受限，而过度深度则会导致计算成本过高。我们认为，有效的模型应能够自适应地建模显式一跳交互和隐式高阶依赖性，而不是仅仅依靠架构深度。为此，我们提出了ViTE（虚拟图轨迹专家路由器），一种用于行人轨迹预测的新型框架。ViTE包含两个关键模块：虚拟图引入动态虚拟节点来建模长距离和高阶交互，无需深层GNN堆叠；专家路由器使用专家混合设计根据社交上下文自适应选择交互专家。这种组合能够在不同的交互模式中实现灵活且可扩展的推理。在ETH/UCY、NBA和SDD三个基准测试上的实验表明，我们的方法始终达到最先进的性能，验证了其有效性和实际效率。


### 论文摘要

Pedestrian trajectory prediction is critical for ensuring safety in autonomous driving, surveillance systems, and urban planning applications. While early approaches primarily focus on one-hop pairwise relationships, recent studies attempt to capture high-order interactions by stacking multiple Graph Neural Network (GNN) layers. However, these approaches face a fundamental trade-off: insufficient layers may lead to under-reaching problems that limit the model's receptive field, while excessive depth can result in prohibitive computational costs. We argue that an effective model should be capable of adaptively modeling both explicit one-hop interactions and implicit high-order dependencies, rather than relying solely on architectural depth. To this end, we propose ViTE (Virtual graph Trajectory Expert router), a novel framework for pedestrian trajectory prediction. ViTE consists of two key modules: a Virtual Graph that introduces dynamic virtual nodes to model long-range and high-order interactions without deep GNN stacks, and an Expert Router that adaptively selects interaction experts based on social context using a Mixture-of-Experts design. This combination enables flexible and scalable reasoning across varying interaction patterns. Experiments on three benchmarks (ETH/UCY, NBA, and SDD) demonstrate that our method consistently achieves state-of-the-art performance, validating both its effectiveness and practical efficiency.

---

## 185. TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective

**论文链接:** [http://arxiv.org/abs/2511.12174v1](http://arxiv.org/abs/2511.12174v1)

**作者:** Lifeng Shen, Xuyang Li, Lele Long

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

TSGDiff是一个基于图视角的时间序列生成框架，通过动态图表示和图神经网络有效捕捉时间序列的复杂依赖性和结构模式。

### 背景

扩散模型在数据生成方面显示出巨大潜力，但生成时间序列数据仍然具有挑战性，因为需要捕捉复杂的时间依赖性和结构模式。

### 目的

提出一个名为TSGDiff的新框架，从基于图的视角重新思考时间序列生成。

### 方法

将时间序列表示为动态图，边基于频谱特性和时间依赖性构建；采用基于图神经网络的编码器-解码器架构构建潜在空间；提出拓扑结构保真度（Topo-FID）评分，包含图编辑相似性和结构熵相似性两个子指标。

### 主要发现

在真实世界数据集上的实验表明，TSGDiff生成高质量的合成时间序列数据，忠实保留了时间依赖性和结构完整性。

### 结论

TSGDiff框架推进了合成时间序列生成的领域发展。

### 翻译

扩散模型在数据生成方面显示出巨大潜力，但生成时间序列数据仍然具有挑战性，因为需要捕捉复杂的时间依赖性和结构模式。在本文中，我们提出了TSGDiff，一个从基于图的视角重新思考时间序列生成的新框架。具体而言，我们将时间序列表示为动态图，其中边基于频谱特性和时间依赖性构建。采用基于图神经网络的编码器-解码器架构构建潜在空间，使扩散过程能够有效建模时间序列的结构表示分布。此外，我们提出了拓扑结构保真度（Topo-FID）评分，这是一个用于评估时间序列图表示结构相似性的图感知指标。Topo-FID整合了两个子指标：图编辑相似性，它量化邻接矩阵的差异；以及结构熵相似性，它评估节点度分布的熵。这个全面的指标提供了对生成时间序列结构保真度更准确的评估。在真实世界数据集上的实验表明，TSGDiff生成高质量的合成时间序列数据，忠实保留了时间依赖性和结构完整性，从而推进了合成时间序列生成的领域。


### 论文摘要

Diffusion models have shown great promise in data generation, yet generating time series data remains challenging due to the need to capture complex temporal dependencies and structural patterns. In this paper, we present \textit{TSGDiff}, a novel framework that rethinks time series generation from a graph-based perspective. Specifically, we represent time series as dynamic graphs, where edges are constructed based on Fourier spectrum characteristics and temporal dependencies. A graph neural network-based encoder-decoder architecture is employed to construct a latent space, enabling the diffusion process to model the structural representation distribution of time series effectively. Furthermore, we propose the Topological Structure Fidelity (Topo-FID) score, a graph-aware metric for assessing the structural similarity of time series graph representations. Topo-FID integrates two sub-metrics: Graph Edit Similarity, which quantifies differences in adjacency matrices, and Structural Entropy Similarity, which evaluates the entropy of node degree distributions. This comprehensive metric provides a more accurate assessment of structural fidelity in generated time series. Experiments on real-world datasets demonstrate that \textit{TSGDiff} generates high-quality synthetic time series data generation, faithfully preserving temporal dependencies and structural integrity, thereby advancing the field of synthetic time series generation.

---

## 186. FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates

**论文链接:** [http://arxiv.org/abs/2511.12132v1](http://arxiv.org/abs/2511.12132v1)

**作者:** Zhenqiang Ye, Jinjie Lu, Tianlong Gu, Fengrui Hao, Xuemin Wang

**发布时间:** 2025-11-15

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种名为FairGSE的新型图神经网络框架，通过最大化二维结构熵来提高公平性，同时降低假阳性率，解决了现有公平感知GNN在高风险场景中的局限性。

### 背景

图神经网络已成为图表示学习的主流范式，但其有效的消息聚合机制也放大了图拓扑中固有的偏见，引发公平性问题。现有公平感知GNN在公平性指标上表现良好，但忽视了预测负标签的能力，导致极高的假阳性率。

### 目的

在提高公平性的同时仔细校准分类性能，降低假阳性率，而不只是限制准确率损失。

### 方法

提出Fair GNN via Structural Entropy (FairGSE)框架，通过最大化二维结构熵(2D-SE)来提高公平性，同时不忽视假阳性问题。

### 主要发现

在多个真实世界数据集上的实验表明，与最先进的公平感知GNN相比，FairGSE将假阳性率降低了39%，同时实现了可比的公平性改进。

### 结论

FairGSE框架有效地平衡了公平性和假阳性率问题，在高风险场景中具有更好的应用价值。

### 翻译

图神经网络(GNNs)由于其有效的消息聚合已成为图表示学习的主流范式。然而，这一优势也放大了图拓扑中固有的偏见，引发了公平性问题。现有的公平感知GNN在统计平等和均等机会等公平性指标上提供了令人满意的性能，同时保持可接受的准确率权衡。不幸的是，我们观察到这种对公平性指标的追求忽视了GNN预测负标签的能力，导致其预测结果具有极高的假阳性率(FPR)，在高风险场景中产生负面影响。为此，我们主张在提高公平性的同时应仔细校准分类性能，而不仅仅是限制准确率损失。此外，我们提出了通过结构熵实现公平GNN(FairGSE)，这是一个新颖的框架，通过最大化二维结构熵(2D-SE)来提高公平性，同时不忽视假阳性问题。在多个真实世界数据集上的实验表明，与最先进的公平感知GNN相比，FairGSE将FPR降低了39%，同时具有可比的公平性改进。


### 论文摘要

Graph neural networks (GNNs) have emerged as the mainstream paradigm for graph representation learning due to their effective message aggregation. However, this advantage also amplifies biases inherent in graph topology, raising fairness concerns. Existing fairness-aware GNNs provide satisfactory performance on fairness metrics such as Statistical Parity and Equal Opportunity while maintaining acceptable accuracy trade-offs. Unfortunately, we observe that this pursuit of fairness metrics neglects the GNN's ability to predict negative labels, which renders their predictions with extremely high False Positive Rates (FPR), resulting in negative effects in high-risk scenarios. To this end, we advocate that classification performance should be carefully calibrated while improving fairness, rather than simply constraining accuracy loss. Furthermore, we propose Fair GNN via Structural Entropy (\textbf{FairGSE}), a novel framework that maximizes two-dimensional structural entropy (2D-SE) to improve fairness without neglecting false positives. Experiments on several real-world datasets show FairGSE reduces FPR by 39\% vs. state-of-the-art fairness-aware GNNs, with comparable fairness improvement.

---

## 187. A Code Smell Refactoring Approach using GNNs

**论文链接:** [http://arxiv.org/abs/2511.12069v1](http://arxiv.org/abs/2511.12069v1)

**作者:** HanYu Zhang, Tomoji Kishi

**发布时间:** 2025-11-15

### GPT解析

### 总结

该研究提出了一种基于图深度学习的代码异味重构方法，解决了现有技术中基于指标和规则的方法依赖手动定义启发式方法，以及基于深度学习的方法受限于数据集可用性和模型设计的问题。

### 背景

代码异味是软件重构中的重大挑战，表示潜在的设计或实现缺陷，可能降低软件可维护性和可演化性。过去几十年提出的重构方法可分为基于指标、基于规则和基于机器学习的方法，近年来基于深度学习的方法也受到广泛关注。

### 目的

提出一种基于图深度学习的代码异味重构方法，克服现有技术的局限性。

### 方法

设计了两种类型的输入图（类级别和方法级别），使用图分类和节点分类任务处理三种代表性代码异味（长方法、大类和特性羡慕）。提出半自动化数据集生成方法，使用三种经典GNN架构（GCN、GraphSAGE和GAT）实现该方法。

### 主要发现

与传统和最先进的深度学习方法相比，所提出的方法实现了更好的重构性能。

### 结论

基于图深度学习的方法在代码异味重构方面表现优越。

### 翻译

代码异味是软件重构中的一个重大挑战，它表明潜在的设计或实现缺陷，可能会降低软件的可维护性和可演化性。在过去的几十年中，已经提出了各种重构方法，这些方法大致可以分为基于指标、基于规则和基于机器学习的方法。近年来，基于深度学习的方法也引起了广泛关注。然而，现有技术存在各种局限性。基于指标和规则的方法严重依赖手动定义的启发式方法和阈值，而基于深度学习的方法通常受限于数据集可用性和模型设计。在本研究中，我们提出了一种基于图深度学习的代码异味重构方法。具体来说，我们设计了两种类型的输入图（类级别和方法级别），并使用了图分类和节点分类任务来解决三种代表性代码异味：长方法、大类和特性羡慕的重构问题。在我们的实验中，我们提出了一种半自动化数据集生成方法，可以用最少的手动工作生成大规模数据集。我们使用三种经典的GNN（图神经网络）架构实现了所提出的方法：GCN、GraphSAGE和GAT，并将其性能与传统的最先进的深度学习方法进行了比较。结果表明，所提出的方法实现了更好的重构性能。


### 论文摘要

Code smell is a great challenge in software refactoring, which indicates latent design or implementation flaws that may degrade the software maintainability and evolution. Over the past decades, a variety of refactoring approaches have been proposed, which can be broadly classified into metrics-based, rule-based, and machine learning-based approaches. Recent years, deep learning-based approaches have also attracted widespread attention. However, existing techniques exhibit various limitations. Metrics- and rule-based approaches rely heavily on manually defined heuristics and thresholds, whereas deep learning-based approaches are often constrained by dataset availability and model design. In this study, we proposed a graph-based deep learning approach for code smell refactoring. Specifically, we designed two types of input graphs (class-level and method-level) and employed both graph classification and node classification tasks to address the refactoring of three representative code smells: long method, large class, and feature envy. In our experiment, we propose a semi-automated dataset generation approach that could generate a large-scale dataset with minimal manual effort. We implemented the proposed approach with three classical GNN (graph neural network) architectures: GCN, GraphSAGE, and GAT, and evaluated its performance against both traditional and state-of-the-art deep learning approaches. The results demonstrate that proposed approach achieves superior refactoring performance.

---

## 188. Dynamic Graph Recommendation via Sparse Augmentation and Singular Adaptation

**论文链接:** [http://arxiv.org/abs/2511.11969v1](http://arxiv.org/abs/2511.11969v1)

**作者:** Zhen Tao, Yuehang Cao, Yang Fang, Yunhui Liu, Xiang Zhao, Tieke He

**发布时间:** 2025-11-15

**备注:** ICASSP 2025

### GPT解析

### 总结

GraphSASA是一种新的动态推荐系统高效微调方法，通过测试时增强和奇异值分解技术解决了大规模微调计算资源需求高和稀疏交互节点表示不足的问题，在三个大规模数据集上实现了最先进的性能。

### 背景

动态推荐系统在个性化服务中扮演关键角色，它通过建模用户历史交互偏好提供实时推荐。研究表明预训练的动态图神经网络(GNNs)能取得优异性能。

### 目的

解决现有动态推荐方法在大规模微调时计算资源需求高以及稀疏交互节点表示不足的问题，实现高效微调。

### 方法

GraphSASA采用测试时增强，利用分层图聚合过程中节点表示分布的相似性增强节点表示；应用奇异值分解，冻结原始向量矩阵，将微调重点放在派生的奇异值矩阵上，减少参数负担并提高微调适应性。

### 主要发现

实验结果表明GraphSASA在三个大规模数据集上实现了最先进的性能。

### 结论

GraphSASA通过创新的测试时增强和奇异值分解方法，有效解决了动态推荐系统中高效微调的挑战，证明了其在实际应用中的优越性。

### 翻译

动态推荐侧重于从历史交互中建模用户偏好并在当前时间提供推荐，在许多个性化服务中起着关键作用。最近的工作表明，预训练的动态图神经网络(GNNs)可以取得优异的性能。然而，现有的大规模微调节点表示的方法需要大量计算资源。此外，度分布的长尾特性导致稀疏交互的节点表示不足，对高效微调提出了挑战。为解决这些问题，我们引入了GraphSASA，这是一种用于动态推荐系统中高效微调的新方法。GraphSASA利用分层图聚合过程中节点表示分布的相似性进行测试时增强，从而增强节点表示。然后应用奇异值分解，冻结原始向量矩阵，同时将微调重点放在派生的奇异值矩阵上，这减少了微调的参数负担并提高了微调适应性。实验结果表明，我们的方法在三个大规模数据集上实现了最先进的性能。


### 论文摘要

Dynamic recommendation, focusing on modeling user preference from historical interactions and providing recommendations on current time, plays a key role in many personalized services. Recent works show that pre-trained dynamic graph neural networks (GNNs) can achieve excellent performance. However, existing methods by fine-tuning node representations at large scales demand significant computational resources. Additionally, the long-tail distribution of degrees leads to insufficient representations for nodes with sparse interactions, posing challenges for efficient fine-tuning. To address these issues, we introduce GraphSASA, a novel method for efficient fine-tuning in dynamic recommendation systems. GraphSASA employs test-time augmentation by leveraging the similarity of node representation distributions during hierarchical graph aggregation, which enhances node representations. Then it applies singular value decomposition, freezing the original vector matrix while focusing fine-tuning on the derived singular value matrices, which reduces the parameter burden of fine-tuning and improves the fine-tuning adaptability. Experimental results demonstrate that our method achieves state-of-the-art performance on three large-scale datasets.

---

## 189. LiDAR-GS++:Improving LiDAR Gaussian Reconstruction via Diffusion Priors

**论文链接:** [http://arxiv.org/abs/2511.12304v1](http://arxiv.org/abs/2511.12304v1)

**作者:** Qifeng Chen, Jiarun Liu, Rengan Xie, Tao Tang, Sicong Du, Yiru Zhao, Yuchi Huo, Sheng Yang

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI-26

### GPT解析

### 总结

本文提出了一种名为LiDAR-GS++的LiDAR高斯散射重建方法，通过引入扩散先验增强，解决了现有方法在推断新视图合成中出现的伪影问题，实现了实时、高保真度的重新模拟。

### 背景

基于高斯散射的LiDAR渲染方法近年来取得显著进展，在质量和速度上都超过了神经辐射场方法，但单次扫描重建不完整导致推断新视图合成时存在伪影问题。

### 目的

提出LiDAR-GS++方法，解决单次扫描重建不完整导致的推断新视图合成伪影问题，实现公共道路上实时、高保真度的重新模拟。

### 方法

引入一个可控的LiDAR生成模型，基于粗略推断的渲染生成额外的几何一致扫描；采用有效的蒸馏机制进行扩展重建；将重建扩展到欠拟合区域，确保推断新视图的全局几何一致性，同时保留传感器捕获的场景表面细节。

### 主要发现

通过扩展重建到欠拟合区域，可以确保推断新视图的全局几何一致性，同时保留传感器捕获的场景表面细节。

### 结论

LiDAR-GS++在多个公共数据集上实验证明，在插值和推断视点上均达到最先进性能，超越了现有的高斯散射和基于神经辐射场的方法。

### 翻译

最近的基于高斯散射的渲染在LiDAR方面取得了显著进展，在质量和速度上都超过了神经辐射场。然而，这些方法在推断新视图合成时存在伪影，这是由于单次扫描重建不完整。为解决这一限制，我们提出了LiDAR-GS++，一种通过扩散先验增强的LiDAR高斯散射重建方法，用于公共道路上的实时和高保真度重新模拟。具体而言，我们引入了一个可控的LiDAR生成模型，基于粗略推断的渲染来产生额外的几何一致的扫描，并采用有效的蒸馏机制进行扩展重建。通过将重建扩展到欠拟合区域，我们的方法确保了推断新视图的全局几何一致性，同时保留了传感器捕获的场景表面细节。在多个公共数据集上的实验表明，LiDAR-GS++在插值和推断视点上都达到了最先进的性能，超越了现有的高斯散射和基于神经辐射场的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于高斯溅射（GS）的LiDAR重建方法在渲染外推（extrapolated）新视角时存在伪影的问题。这个问题在自动驾驶领域非常重要，因为高质量的传感器模拟对于开发鲁棒的自动驾驶算法至关重要，而当前方法在处理变道、障碍物避让等视角变化场景时表现不佳，限制了自动驾驶算法的开发和验证。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有LiDAR重建方法在外推视角上的局限性，认识到跨模态提示生成的LiDAR扫描与真实扫描存在几何差异，以及混合生成与真实扫描可能导致矛盾。他们借鉴了现有的基于GS的LiDAR重建方法（如LiDAR-GS）和扩散模型（特别是LiDM），创新性地设计了LiDAR到LiDAR的可控生成模型，利用粗略外推渲染作为条件生成额外的几何一致扫描，并引入深度失真感知的蒸馏策略来有选择地优化欠拟合区域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用扩散先验增强LiDAR高斯重建，解决外推视角渲染质量问题。整体流程包括：1）将LiDAR点云投影到范围视图并使用神经2D高斯场进行初始重建；2）通过移动视角渲染低质量外推扫描，作为可控LiDAR生成模型的条件，生成几何一致的外推扫描；3）使用深度失真感知的蒸馏策略，将生成的扫描与真实扫描混合进行扩展重建，只对深度失真显著的区域应用额外监督；4）从重建的高斯实现实时、高质量的新视角合成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）通过扩散先验扩展神经2DGS场的LiDAR重建方法；2）基于粗略外推渲染引导的可控LiDAR生成模型；3）深度失真感知的蒸馏策略。相比之前的工作，LiDAR-GS++使用神经2DGS而非3DGS提高几何保真度，引入扩散先验解决外推质量问题，采用LiDAR到LiDAR的可控生成而非跨模态提示提高一致性，并通过蒸馏策略选择性优化欠拟合区域，避免对已收敛区域的负面影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiDAR-GS++通过引入基于扩散先验的可控LiDAR生成和深度失真感知的蒸馏策略，显著提升了LiDAR高斯重建在外推和内插视角上的渲染质量，实现了实时、高保真的城市道路LiDAR重模拟。'}


### 论文摘要

Recent GS-based rendering has made significant progress for LiDAR, surpassing Neural Radiance Fields (NeRF) in both quality and speed. However, these methods exhibit artifacts in extrapolated novel view synthesis due to the incomplete reconstruction from single traversal scans. To address this limitation, we present LiDAR-GS++, a LiDAR Gaussian Splatting reconstruction method enhanced by diffusion priors for real-time and high-fidelity re-simulation on public urban roads. Specifically, we introduce a controllable LiDAR generation model conditioned on coarsely extrapolated rendering to produce extra geometry-consistent scans and employ an effective distillation mechanism for expansive reconstruction. By extending reconstruction to under-fitted regions, our approach ensures global geometric consistency for extrapolative novel views while preserving detailed scene surfaces captured by sensors. Experiments on multiple public datasets demonstrate that LiDAR-GS++ achieves state-of-the-art performance for both interpolated and extrapolated viewpoints, surpassing existing GS and NeRF-based methods.

---

## 190. Conflict-Free Flight Scheduling Using Strategic Demand Capacity Balancing for Urban Air Mobility Operations

**论文链接:** [http://arxiv.org/abs/2511.11854v1](http://arxiv.org/abs/2511.11854v1)

**作者:** Vahid Hemmati, Yonas Ayalew, Ahmad Mohammadi, Reza Ahmari, Parham Kebria, Abdollah Homaifar, Mehrdad Saif

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种冲突避免的多智能体飞行调度方法，用于确保受限空域中城市空中交通(UAM)运营的稳健分离。

### 背景

随着城市空中交通(UAM)系统的发展，需要在密集空域中安全调度多架飞行器。

### 目的

开发一种能够确保飞行器安全分离且减少总延迟的多智能体飞行调度系统。

### 方法

首先引入基于延迟起飞的成对冲突避免(PCA)方法，利用运动学原理保持安全距离；然后将PCA扩展到多智能体场景，制定优化方法以系统确定交通密度增加时的起飞时间。

### 主要发现

通过多种多智能体环境和真实UAM用例的数值模拟，该方法在确保无碰撞操作的同时显著减少了总延迟。

### 结论

该方法为新兴的城市空中交通系统提供了可扩展的框架。

### 翻译

本文提出了一种冲突避免的多智能体飞行调度，用于确保城市空中交通(UAM)运营在受限空域中的稳健分离。首先，我们引入了基于延迟起飞的成对冲突避免(PCA)，利用运动学原理保持安全距离。接下来，我们将PCA扩展到多智能体场景，制定了一种优化方法，系统确定交通密度增加时的起飞时间。通过平均延迟等性能指标评估我们解决方案的有效性。通过在多种多智能体环境和真实UAM用例中的数值模拟，我们的方法在确保无碰撞操作的同时显著减少了总延迟。这种方法为新兴的城市空中交通系统提供了可扩展的框架。


### 论文摘要

In this paper, we propose a conflict-free multi- agent flight scheduling that ensures robust separation in con- strained airspace for Urban Air Mobility (UAM) operations application. First, we introduce Pairwise Conflict Avoidance (PCA) based on delayed departures, leveraging kinematic principles to maintain safe distances. Next, we expand PCA to multi-agent scenarios, formulating an optimization approach that systematically determines departure times under increasing traffic densities. Performance metrics, such as average delay, assess the effectiveness of our solution. Through numerical simulations across diverse multi-agent environments and real- world UAM use cases, our method demonstrates a significant reduction in total delay while ensuring collision-free operations. This approach provides a scalable framework for emerging urban air mobility systems.

---

## 191. How many stations are sufficient? Exploring the effect of urban weather station density reduction on imputation accuracy of air temperature and humidity

**论文链接:** [http://arxiv.org/abs/2511.11652v1](http://arxiv.org/abs/2511.11652v1)

**作者:** Marvin Plein, Carsten F. Dormann, Andreas Christen

**发布时间:** 2025-11-10

### GPT解析

### 总结

该研究提出了一种逐步移除气象站的方法，证明可以在保持较高预测准确性的情况下大幅减少城市气象站数量，从而优化资源配置。

### 背景

城市气象站网络(WSNs)被广泛用于监测城市天气和气候模式，辅助城市规划，但维护这些网络成本高昂且需要大量人力。

### 目的

开发一个逐步的站点移除程序，以减少德国弗莱堡现有气象站网络的密度，并分析气象站子集在模拟减少密度后一年内重现整个原始气象站的空气温度和湿度模式的能力。

### 方法

在德国弗莱堡实施逐步站点移除程序，模拟气象站密度减少，并评估子集重现原始网络温度和湿度模式的能力。

### 主要发现

1) 在完全部署一年后，可以大幅减少站点数量同时保持高预测准确性；2) 站点数量从42个减少到4个，空气温度和相对湿度的预测误差分别仅增加20%和16%；3) 森林中远程站的预测准确性比建成区或开阔区域站点差；4) 所有子集的预测准确性始终优于先进的城市地表模型；5) 位于建成区和农村地区边缘的站点在重建城市气候特征时最有价值。

### 结论

通过精简气象站网络，可以在城市气候研究中最大化财务和人力资源的有效配置。

### 翻译

城市气象站网络(WSNs)被广泛用于监测城市天气和气候模式并辅助城市规划。然而，维护WSNs成本高昂且劳动密集。本文提出了一种逐步的站点移除程序，用于精简德国弗莱堡现有的WSNs，并分析了在模拟减少WSN密度后一年内，气象站子集重现整个原始WSN空气温度和湿度模式的能力。我们发现，在完全部署一年后，可以在保持高预测准确性的情况下大幅减少站点数量。例如，站点数量从42个减少到4个，空气温度的平均预测RMSE从0.69K增加到0.83K，相对湿度的平均预测RMSE从3.8%增加到4.4%，分别仅增加20%和16%。森林中远程站的预测准确性比建成区或开阔区域站点差，但始终优于最先进的数值城市地表模型(城市地表能量与水平衡方案)。位于建成区和农村地区边缘的站点在重建城市气候特征时最有价值。我们的研究证明了精简WSNs的潜力，可以最大化城市气候研究中财务和人力资源的有效配置。


### 论文摘要

Urban weather station networks (WSNs) are widely used to monitor urban weather and climate patterns and aid urban planning. However, maintaining WSNs is expensive and labor-intensive. Here, we present a step-wise station removal procedure to thin an existing WSN in Freiburg, Germany, and analyze the ability of WSN subsets to reproduce air temperature and humidity patterns of the entire original WSN for a year following a simulated reduction of WSN density. We found that substantial reductions in station numbers after one year of full deployment are possible while retaining high predictive accuracy. A reduction from 42 to 4 stations, for instance, increased mean prediction RMSEs from 0.69 K to 0.83 K for air temperature and from 3.8% to 4.4% for relative humidity, corresponding to RMSE increases of only 20% and 16%, respectively. Predictive accuracy is worse for remote stations in forests than for stations in built-up or open settings, but consistently better than a state-of-the-art numerical urban land-surface model (Surface Urban Energy and Water Balance Scheme). Stations located at the edges between built-up and rural areas are most valuable when reconstructing city-wide climate characteristics. Our study demonstrates the potential of thinning WSNs to maximize the efficient allocation of financial and personnel-related resources in urban climate research.

---

## 192. Assessing On-Demand Mobility Services and Policy Impacts: A Case Study from Chengdu, China

**论文链接:** [http://arxiv.org/abs/2511.06074v2](http://arxiv.org/abs/2511.06074v2)

**作者:** Youkai Wu, Zhaoxia Guo, Qi Liu, Stein W. Wallace

**发布时间:** 2025-11-08

### GPT解析

### 总结

本研究通过模拟框架评估了网约车服务与传统扬招出租车服务的性能差异，并分析了相关政策干预的效果。研究发现在相同车队规模和需求下，不进行巡游的网约车服务显著改善了乘客等待时间、空驶里程和空速能耗，这种改善在低需求时段和偏远地区尤为明显。同时，研究探讨了车队规模、地理围栏和需求管理政策对网约车服务的影响。

### 背景

网约车服务的快速发展显著改变了城市按需出行模式，但其相对于传统扬招出租车服务的表现以及相关政策干预的有效性尚不清楚。

### 目的

评估网约车服务与传统扬招出租车服务的性能差异，并分析时空特征及三种政策（车队规模管理、地理围栏和需求管理）对网约车服务性能的影响。

### 方法

开发了一个整合基于图论的行程-车辆匹配机制和真实巡游出租车运营数据的模拟框架，用于模拟中国成都的网约车服务。通过三个关键绩效指标（平均乘客等待时间、平均空驶里程和平均空速能耗）评估两种服务模式的性能。

### 主要发现

1) 在相同车队规模和需求下，不进行巡游的网约车服务使APWT、ADM和ADEC分别减少81%、75%和72.1%；2) 这些改善在午夜低需求时段和机场等偏远地区最为明显；3) 扩大车队规模会产生边际效益递减；4) 地理围栏会降低整体性能但改善市中心行程性能；5) 针对高吸引力、低需求地区的需求侧管理可减少乘客等待时间而不增加空驶成本。

### 结论

网约车服务相比传统扬招出租车具有显著优势，特别是在减少乘客等待时间和空驶方面。然而，政策制定需要考虑不同时空特征的影响，车队规模扩张存在边际效益递减，而地理围栏和需求管理政策可能在不同区域产生不同效果。

### 翻译

网约车服务的快速扩张显著改变了城市按需出行模式，但相对于传统扬招出租车服务的表现以及相关政策干预的有效性仍不明确。本研究提出了一个模拟框架，整合了基于图论的行程-车辆匹配机制和真实的巡游出租车运营数据，以模拟中国成都的网约车服务。从三个关键绩效指标（平均乘客等待时间、平均空驶里程和平均空速能耗）评估了两种按需出行服务模式（网约车和扬招出租车）的性能。我们进一步检查了时空特征和三种类型的政策（车队规模管理、地理围栏和需求管理）对网约车服务性能的影响。结果显示，在相同的车队规模和出行需求下，不进行巡游的网约车服务实现了显著改善，APWT、ADM和ADEC分别减少了81%、75%和72.1%。这些改进在午夜低需求时段和机场等偏远地区最为明显。我们的分析还揭示了对于网约车服务：(1)扩大车队规模会产生边际效益递减；(2)地理围栏会降低整体性能，但能改善市中心所有行程的性能；(3)针对高吸引力、低需求地区出行需求的需求侧管理可以有效减少乘客等待时间，而不会增加空驶成本。


### 论文摘要

The rapid expansion of ride-hailing services has significantly reshaped urban on-demand mobility patterns, but it still remains unclear how they perform relative to traditional street-hailing services and how effective are related policy interventions. This study presents a simulation framework integrating a graph theory-based trip-vehicle matching mechanism with real cruising taxi operations data to simulate ride-hailing services in Chengdu, China. The performances of the two on-demand mobility service modes (i.e., ride-hailing and street-hailing) are evaluated in terms of three key performance indicators: average passenger waiting time (APWT), average deadheading miles (ADM), and average deadheading energy consumption (ADEC). We further examine the impacts of spatiotemporal characteristics and three types of policies: fleet size management, geofencing, and demand management, on the performance of ride-hailing services. Results show that under the same fleet size and trip demand as street-hailing taxis, ride-hailing services without cruising achieve substantial improvements, reducing APWT, ADM, and ADEC by 81\%, 75\%, and 72.1\%, respectively. These improvements are most pronounced during midnight low-demand hours and in remote areas such as airports. Our analysis also reveals that for ride-hailing service, (1) expanding fleet size yields diminishing marginal benefits; (2) geofencing worsens overall performance while it improves the performance of serving all trips within the city center; and (3) demand-side management targeting trips to high-attraction and low-demand areas can effectively reduce passenger waiting time without increasing deadheading costs.

---

## 193. Real-time distortion prediction in metallic additive manufacturing via a physics-informed neural operator approach

**论文链接:** [http://arxiv.org/abs/2511.13178v1](http://arxiv.org/abs/2511.13178v1)

**作者:** Mingxuan Tian, Haochen Mu, Donghong Ding, Mengjiao Li, Yuhan Ding, Jianping Zhao

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出了一种物理信息神经算子（PINO）方法，用于金属增材制造中的实时变形场预测，结合了物理定律与深度学习，实现了高精度、低误差累积和时间效率的长时程预测。

### 背景

随着数字孪生和智能制造系统的发展，金属增材制造中迫切需要实时变形场预测来控制缺陷。然而，数值模拟方法计算成本高、运行时间长，无法实现实时应用；传统机器学习模型难以提取时空特征进行长时程预测，也无法解耦热机械场。

### 目的

开发一种物理信息神经算子（PINO）来预测未来15秒内z和y方向的变形，实现实时且精确的变形场预测。

### 方法

提出物理信息深度算子网络-循环神经网络（PIDeepONet-RNN），采用主干网络处理温度历史，分支网络编码变形场，实现热机械响应解耦。将热传导方程作为软约束纳入模型，确保物理一致性并抑制非物理伪影。使用实验验证的有限元方法生成的数据集进行训练和测试。

### 主要发现

模型实现了高精度、低误差累积和时间效率，z和y方向的最大绝对误差分别低至0.9733毫米和0.2049毫米。误差分布显示熔池区域误差较高，但在沉积和关键区域梯度范数较低。PINO代理模型在实时长时程物理场预测方面展现出潜力。

### 结论

所提出的方法为金属增材制造的实时变形场预测提供了有效解决方案。物理信息神经网络结合物理定律为基础，为预测提供了稳健且可解释的基础，有助于控制制造缺陷。

### 翻译

随着数字孪生和智能制造系统的发展，金属增材制造中迫切需要实时变形场预测来控制缺陷。然而，数值模拟方法计算成本高、运行时间长，无法实现实时应用；传统机器学习模型难以提取时空特征进行长时程预测，也无法解耦热机械场。本文提出了一种物理信息神经算子（PINO）来预测未来15秒内z和y方向的变形。我们的方法——物理信息深度算子网络-循环神经网络（PIDeepONet-RNN）——采用主干网络和分支网络分别处理温度历史和编码变形场，实现热机械响应的解耦。通过将热传导方程作为软约束纳入模型，确保物理一致性并抑制非物理伪影，从而在热历史和变形之间建立了更物理一致的映射。这很重要，因为这种基于物理定律的基函数为预测提供了稳健且可解释的基础。所提出的模型使用实验验证的有限元方法生成的数据集进行训练和测试。评估显示，该模型实现了高精度、低误差累积和时间效率。z和y方向的最大绝对误差分别低至0.9733毫米和0.2049毫米。误差分布显示熔池区域误差较高，但在沉积和关键区域梯度范数较低。PINO代理模型的性能突显了其在控制缺陷中进行实时长时程物理场预测的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决金属增材制造过程中的实时变形预测问题。在金属增材制造中，由于高度集中的热输入和快速冷却，会导致非均匀的热积累，产生不均匀的热膨胀和收缩，最终导致内部应力和永久塑性变形。这个问题很重要，因为未受控制的变形不仅影响零件尺寸精度，还会损害结构完整性，可能导致关键应用中的失效。实时变形预测对于变形控制和质量保证至关重要，可以帮助优化制造过程，减少废品率，提高生产效率和产品质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：数值模拟方法计算成本高无法实时应用；传统机器学习模型难以处理长时预测和热-机械场解耦问题。作者借鉴了物理信息机器学习思想，将物理先验知识嵌入到机器学习模型中；选择了深度算子网络作为基础架构，因为它具有解耦多物理场的能力；引入了卷积长短期记忆网络处理时间序列数据。作者设计了PIDeepONet-RNN模型，结合了物理信息约束和深度算子网络，将热传导方程作为软约束纳入训练过程。作者借鉴了Sharma等人的物理信息神经网络框架、Jiao等人的深度算子网络应用以及Kushwaha等人的多物理场预测结构，但在现有工作基础上进行了创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物理定律与数据驱动相结合，通过深度算子网络的分支和主干网络分别处理温度历史和变形场，实现热-机械响应解耦，同时使用物理约束确保预测符合物理规律。整体流程包括：1)使用实验验证的有限元方法生成训练和测试数据集；2)设计模型架构，包括处理温度历史的主干网络、处理变形历史的分支网络，以及物理约束；3)使用组合损失函数训练模型；4)输入历史数据预测未来1-15秒的变形场；5)通过多种指标评估预测性能并与传统方法对比。模型能够处理未知的工艺参数和几何演化，具有强泛化能力，且计算效率极高。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创新性地将物理信息机器学习与深度算子网络结合，提出PINO方法；2)利用DeepONet架构实现热-机械场的解耦建模；3)结合ConvLSTM处理时间序列，实现长期低误差累积的变形预测；4)模型具有强自适应性和时间效率。相比之前的工作，不同之处在于：相比传统数值模拟方法，计算效率极高且自适应性强；相比传统机器学习模型，具有更好的长时预测能力和泛化能力；相比现有物理信息神经网络，具有更好的解耦能力和预测效率；相比现有深度算子网络应用，具有更强的物理一致性和更好的时空特征提取能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于物理信息深度算子网络的实时变形预测方法，通过解耦热-机械场和引入物理约束，实现了金属增材制造过程中长期、准确的变形预测，为智能制造和数字孪生系统提供了关键技术支持。'}


### 论文摘要

With the development of digital twins and smart manufacturing systems, there is an urgent need for real-time distortion field prediction to control defects in metal Additive Manufacturing (AM). However, numerical simulation methods suffer from high computational cost, long run-times that prevent real-time use, while conventional Machine learning (ML) models struggle to extract spatiotemporal features for long-horizon prediction and fail to decouple thermo-mechanical fields. This paper proposes a Physics-informed Neural Operator (PINO) to predict z and y-direction distortion for the future 15 s. Our method, Physics-informed Deep Operator Network-Recurrent Neural Network (PIDeepONet-RNN) employs trunk and branch network to process temperature history and encode distortion fields, respectively, enabling decoupling of thermo-mechanical responses. By incorporating the heat conduction equation as a soft constraint, the model ensures physical consistency and suppresses unphysical artifacts, thereby establishing a more physically consistent mapping between the thermal history and distortion. This is important because such a basis function, grounded in physical laws, provides a robust and interpretable foundation for predictions. The proposed models are trained and tested using datasets generated from experimentally validated Finite Element Method (FEM). Evaluation shows that the model achieves high accuracy, low error accumulation, time efficiency. The max absolute errors in the z and y-directions are as low as 0.9733 mm and 0.2049 mm, respectively. The error distribution shows high errors in the molten pool but low gradient norms in the deposited and key areas. The performance of PINO surrogate model highlights its potential for real-time long-horizon physics field prediction in controlling defects.

---

## 194. Green Emergency Communications in RIS- and MA-Assisted Multi-UAV SAGINs: A Partially Observable Reinforcement Learning Approach

**论文链接:** [http://arxiv.org/abs/2511.12892v1](http://arxiv.org/abs/2511.12892v1)

**作者:** Liangshun Wu, Wen Chen, Shunqing Zhang, Yajun Wang, Kunlun Wang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种在灾难后空间-空中-地面一体化网络中利用无人机和可重构智能表面恢复连接的新方法，解决了通信限制的部分可观察性问题，显著提升了网络性能。

### 背景

灾难后空间-空中-地面一体化网络中地面基础设施受损，无人机需要在复杂的非视距城市环境中快速恢复关键地面终端的连接，同时面临通信限制的部分可观察性挑战。

### 目的

增强网络覆盖范围，解决通信限制的部分可观察性问题，提高无人机网络在复杂环境中的连接恢复能力。

### 方法

提出一种时空A2C方法，无人机传输包含局部状态的先前决策消息、紧凑策略指纹和每个邻居的循环信念，使用空间折扣强调局部交互，并在每时隙一跳延迟条件下分析训练稳定性。

### 主要发现

现有多智能体强化学习方法存在不足，非通信方法依赖不可用全局评论家，启发式共享脆弱冗余，可学习协议失去邻居结构并加剧非平稳性；所提方法在收敛速度、渐近奖励、误差和通信吞吐量-能量权衡方面显著优于对比方法。

### 结论

所提出的时空A2C方法有效解决了通信限制的部分可观察性问题，实现了更快的收敛速度、更高的渐近奖励、更低的误差以及更好的通信吞吐量-能量平衡。

### 翻译

在灾难后空间-空中-地面一体化网络中，地面基础设施通常受损，无人机必须在复杂的非视距城市环境中快速恢复关键地面终端的连接。为增强覆盖，无人机使用可移动天线，而幸存高楼上的可重构智能表面则重定向信号。主要挑战是通信限制的部分可观察性，导致每架无人机只能获得狭窄且快速变化的局部视图，这影响了价值估计的稳定性。现有的多智能体强化学习方法不足——非通信方法依赖于不可用的全局评论家，启发式共享既脆弱又冗余，而可学习协议（如CommNet、DIAL）失去了每个邻居的结构，并在带宽受限下加剧了非平稳性。为解决部分可观察性问题，我们提出了一种时空A2C，其中每架无人机传输包含局部状态的先前决策消息、紧凑的策略指纹和每个邻居的循环信念，并进行编码和拼接。空间折扣用于塑造价值目标以强调局部交互，而在每时隙一跳延迟条件下的分析解释了使用延迟视图的稳定训练。实验结果表明，我们的策略在收敛速度、渐近奖励、时间差分/优势误差以及通信吞吐量-能量权衡方面优于IA2C、ConseNet、FPrint、DIAL和CommNet。


### 论文摘要

In post-disaster space-air-ground integrated networks (SAGINs), terrestrial infrastructure is often impaired, and unmanned aerial vehicles (UAVs) must rapidly restore connectivity for mission-critical ground terminals in cluttered non-line-of-sight (NLoS) urban environments. To enhance coverage, UAVs employ movable antennas (MAs), while reconfigurable intelligent surfaces (RISs) on surviving high-rises redirect signals. The key challenge is communication-limited partial observability, leaving each UAV with a narrow, fast-changing neighborhood view that destabilizes value estimation. Existing multi-agent reinforcement learning (MARL) approaches are inadequate--non-communication methods rely on unavailable global critics, heuristic sharing is brittle and redundant, and learnable protocols (e.g., CommNet, DIAL) lose per-neighbor structure and aggravate non-stationarity under tight bandwidth. To address partial observability, we propose a spatiotemporal A2C where each UAV transmits prior-decision messages with local state, a compact policy fingerprint, and a recurrent belief, encoded per neighbor and concatenated. A spatial discount shapes value targets to emphasize local interactions, while analysis under one-hop-per-slot latency explains stable training with delayed views. Experimental results show our policy outperforms IA2C, ConseNet, FPrint, DIAL, and CommNet--achieving faster convergence, higher asymptotic reward, reduced Temporal-Difference(TD)/advantage errors, and a better communication throughput-energy trade-off.

---

## 195. A Review of Statistical and Machine Learning Approaches for Coral Bleaching Assessment

**论文链接:** [http://arxiv.org/abs/2511.12234v1](http://arxiv.org/abs/2511.12234v1)

**作者:** Soham Sarkar, Arnab Hazra

**发布时间:** 2025-11-15

**备注:** 40 pages, 3 figures, 3 tables

### GPT解析

### 总结

这篇综述文章概述了用于评估珊瑚白化的现有统计和机器学习方法，并讨论了未来研究方向。

### 背景

珊瑚白化是海洋生态系统的主要担忧，过去三十年中超过一半的珊瑚礁已经白化或死亡。海表温度升高和各种时空环境因素被认为是主要原因。然而，关于随机建模方法评估珊瑚白化的文献极其稀少。

### 目的

提供统计和机器学习方法在评估珊瑚白化方面的概述，并讨论未来数据驱动的研究方向。

### 方法

文章总结了两种主要方法：统计框架（包括简单回归模型、广义线性模型、广义加性模型、贝叶斯回归模型、时空模型和韧性指标）和机器学习方法（包括随机森林、决策树、支持向量机和空间算子）。

### 主要发现

统计方法常用于探索不同环境压力因素如何影响珊瑚白化；机器学习方法更擅长检测非线性关系、分析高维数据以及整合来自不同来源的异构数据。

### 结论

数据驱动策略对于有效的礁石管理至关重要，需要构建与珊瑚白化相关的特定背景下的统计和机器学习模型。

### 翻译

珊瑚白化是海洋生态系统的主要担忧；在过去三十年中，世界上超过一半的珊瑚礁已经白化或死亡。海表温度的升高，加上各种时空环境因素，被认为是珊瑚白化的主要原因。统计和机器学习界已经详细研究了环境的多个方面。然而，关于各种随机建模方法评估珊瑚白化的文献极其稀少。数据驱动策略对于有效的礁石管理至关重要，这篇综述文章概述了用于评估珊瑚白化的现有统计和机器学习方法。统计框架，包括简单回归模型、广义线性模型、广义加性模型、贝叶斯回归模型、时空模型和韧性指标，如费舍尔信息和方差指数，常用于探索不同环境压力因素如何影响珊瑚白化。另一方面，机器学习方法，包括随机森林、决策树、支持向量机和空间算子，更受欢迎，用于检测非线性关系、分析高维数据以及允许整合来自不同来源的异构数据。除了总结这些模型外，我们还讨论了数据驱动的未来研究方向，重点是构建与珊瑚白化相关的特定背景下的统计和机器学习模型。


### 论文摘要

Coral bleaching is a major concern for marine ecosystems; more than half of the world's coral reefs have either bleached or died over the past three decades. Increasing sea surface temperatures, along with various spatiotemporal environmental factors, are considered the primary reasons behind coral bleaching. The statistical and machine learning communities have focused on multiple aspects of the environment in detail. However, the literature on various stochastic modeling approaches for assessing coral bleaching is extremely scarce. Data-driven strategies are crucial for effective reef management, and this review article provides an overview of existing statistical and machine learning methods for assessing coral bleaching. Statistical frameworks, including simple regression models, generalized linear models, generalized additive models, Bayesian regression models, spatiotemporal models, and resilience indicators, such as Fisher's Information and Variance Index, are commonly used to explore how different environmental stressors influence coral bleaching. On the other hand, machine learning methods, including random forests, decision trees, support vector machines, and spatial operators, are more popular for detecting nonlinear relationships, analyzing high-dimensional data, and allowing integration of heterogeneous data from diverse sources. In addition to summarizing these models, we also discuss potential data-driven future research directions, with a focus on constructing statistical and machine learning models in specific contexts related to coral bleaching.

---

## 196. Learning from Dense Events: Towards Fast Spiking Neural Networks Training via Event Dataset Distillatio

**论文链接:** [http://arxiv.org/abs/2511.12095v1](http://arxiv.org/abs/2511.12095v1)

**作者:** Shuhan Ye, Yi Yu, Qixin Zhang, Chenqi Kong, Qiangqiang Wu, Kun Wang, Xudong Jiang

**发布时间:** 2025-11-15

### GPT解析

### 总结

论文介绍PACE框架，这是首个针对SNNs和事件视觉的数据集蒸馏方法，能将大型训练数据集压缩为小型合成数据集，实现快速SNN训练。

### 背景

事件相机通过感知亮度变化输出二进制异步事件流，其生物启发的动态特性与SNNs高度契合，为传统视觉系统提供节能替代方案。但SNNs因时间编码问题训练成本高，限制了实际应用。

### 目的

减轻SNNs的高训练成本，通过数据集蒸馏技术实现快速训练。

### 方法

PACE框架包含两个核心模块：ST-DSM使用残余膜电位密集化基于脉冲的特征并进行时空匹配；PEQ-N提供与标准事件帧管道兼容的即插即用直通概率整数量化器。

### 主要发现

在DVS-Gesture、CIFAR10-DVS和N-MNIST数据集上，PACE优于现有基线方法，特别是在动态事件流和低IPC情况下表现突出。在N-MNIST上达到84.4%准确率（约完整训练集性能的85%），同时训练时间减少50倍以上，存储成本降低6000倍。

### 结论

PACE生成的紧凑代理模型可实现分钟级SNN训练和高效边缘部署，有效解决了SNN训练成本高的问题。

### 翻译

事件相机通过感知亮度变化并输出二进制异步事件流，吸引了越来越多的关注。它们的生物启发的动态特性与脉冲神经网络（SNNs）高度契合，为传统视觉系统提供了一种有前景的节能替代方案。然而，由于时间编码问题，SNNs的训练成本仍然很高，限制了它们的实际部署。为了减轻SNNs的高训练成本，我们引入了PACE（Phase-Aligned Condensation for Events），这是第一个针对SNNs和事件视觉的数据集蒸馏框架。PACE将大型训练数据集压缩为一个紧凑的合成数据集，实现快速SNN训练，这通过两个核心模块实现：ST-DSM和PEQ-N。ST-DSM使用残余膜电位来密集化基于脉冲的特征（SDR）并进行幅度和相位的细粒度时空匹配（ST-SM），而PEQ-N提供了一个与标准事件帧管道兼容的即插即用直通概率整数量化器。在DVS-Gesture、CIFAR10-DVS和N-MNIST数据集上，PACE优于现有的核心集选择和数据集蒸馏基线方法，特别是在动态事件流和低或中等IPC情况下表现尤为突出。具体来说，在N-MNIST上，它达到了84.4%的准确率，约为完整训练集性能的85%，同时将训练时间减少了50倍以上，存储成本降低了6000倍，生成的紧凑代理模型可实现分钟级的SNN训练和高效的边缘部署。


### 论文摘要

Event cameras sense brightness changes and output binary asynchronous event streams, attracting increasing attention. Their bio-inspired dynamics align well with spiking neural networks (SNNs), offering a promising energy-efficient alternative to conventional vision systems. However, SNNs remain costly to train due to temporal coding, which limits their practical deployment. To alleviate the high training cost of SNNs, we introduce \textbf{PACE} (Phase-Aligned Condensation for Events), the first dataset distillation framework to SNNs and event-based vision. PACE distills a large training dataset into a compact synthetic one that enables fast SNN training, which is achieved by two core modules: \textbf{ST-DSM} and \textbf{PEQ-N}. ST-DSM uses residual membrane potentials to densify spike-based features (SDR) and to perform fine-grained spatiotemporal matching of amplitude and phase (ST-SM), while PEQ-N provides a plug-and-play straight through probabilistic integer quantizer compatible with standard event-frame pipelines. Across DVS-Gesture, CIFAR10-DVS, and N-MNIST datasets, PACE outperforms existing coreset selection and dataset distillation baselines, with particularly strong gains on dynamic event streams and at low or moderate IPC. Specifically, on N-MNIST, it achieves \(84.4\%\) accuracy, about \(85\%\) of the full training set performance, while reducing training time by more than \(50\times\) and storage cost by \(6000\times\), yielding compact surrogates that enable minute-scale SNN training and efficient edge deployment.

---

## 197. Mesh-based Super-resolution of Detonation Flows with Multiscale Graph Transformers

**论文链接:** [http://arxiv.org/abs/2511.12041v1](http://arxiv.org/abs/2511.12041v1)

**作者:** Shivam Barwey, Pinaki Pal

**发布时间:** 2025-11-15

### GPT解析

### 总结

研究开发了一种名为SR-GT的多尺度图变换器方法，用于基于网格的反应流超分辨率重建，这是一种首创的数据驱动建模方法，能够处理复杂几何和非结构化网格。

### 背景

使用最先进的数据驱动技术进行超分辨率流场重建对多种应用有价值，如亚网格闭合建模、加速时空预测、数据压缩和作为稀疏实验测量的上采样工具。

### 目的

开发一种能够捕获流场长程依赖关系、适用于复杂几何和非均匀网格的反应流超分辨率重建方法。

### 方法

SR-GT框架采用独特的元素局部（+邻域）图表示处理粗糙输入，经标记化后由变换器组件处理生成精细输出，利用变换器主干网络捕获低分辨率流场间的长程依赖关系并识别重要特征。

### 主要发现

SR-GT在2D氢气-空气预混混合物爆燃传播的测试问题上表现出高超分辨率精度，能够准确重建反应流场特征，性能优于传统基于插值的超分辨率方案。

### 结论

SR-GT为反应流超分辨率重建提供了一种有效的新方法，特别适用于复杂几何和非结构化网格情况下的多尺度反应流场重建。

### 翻译

使用最先进的数据驱动技术进行超分辨率流场重建对于多种应用具有重要价值，例如亚网格/亚滤波器闭合建模、加速时空预测、数据压缩以及作为稀疏实验测量的上采样工具。在当前工作中，开发了一种首创的多尺度图变换器方法，用于基于网格的反应流超分辨率重建（SR-GT）。这种新颖的数据驱动建模范式利用了与复杂几何和非均匀/非结构化网格兼容的基于图的流场表示。此外，变换器主干网络捕获低分辨率流场不同部分之间的长程依赖关系，识别重要特征，然后生成保留这些特征的高分辨率流场。SR-GT的性能在谱元离散化网格的背景下进行了验证，用于测试2D氢气-空气预混混合物中爆燃传播的挑战性问题，该问题表现出高度复杂的多尺度反应流行为。SR-GT框架利用独特的元素局部（+邻域）图表示来处理粗糙输入，然后在变换器组件处理之前进行标记化，以生成精细输出。研究表明，SR-GT为反应流场特征提供了高超分辨率精度，并且相比传统的基于插值的超分辨率方案具有优越性能。


### 论文摘要

Super-resolution flow reconstruction using state-of-the-art data-driven techniques is valuable for a variety of applications, such as subgrid/subfilter closure modeling, accelerating spatiotemporal forecasting, data compression, and serving as an upscaling tool for sparse experimental measurements. In the present work, a first-of-its-kind multiscale graph transformer approach is developed for mesh-based super-resolution (SR-GT) of reacting flows. The novel data-driven modeling paradigm leverages a graph-based flow-field representation compatible with complex geometries and non-uniform/unstructured grids. Further, the transformer backbone captures long-range dependencies between different parts of the low-resolution flow-field, identifies important features, and then generates the super-resolved flow-field that preserves those features at a higher resolution. The performance of SR-GT is demonstrated in the context of spectral-element-discretized meshes for a challenging test problem of 2D detonation propagation within a premixed hydrogen-air mixture exhibiting highly complex multiscale reacting flow behavior. The SR-GT framework utilizes a unique element-local (+ neighborhood) graph representation for the coarse input, which is then tokenized before being processed by the transformer component to produce the fine output. It is demonstrated that SR-GT provides high super-resolution accuracy for reacting flow-field features and superior performance compared to traditional interpolation-based SR schemes.

---

## 198. Temporal Micro-Doppler Spectrogram-based ViT Multiclass Target Classification

**论文链接:** [http://arxiv.org/abs/2511.11951v1](http://arxiv.org/abs/2511.11951v1)

**作者:** Nghia Thinh Nguyen, Tri Nhu Do

**发布时间:** 2025-11-14

### GPT解析

### 总结

该论文提出了一种新的时间MDS视觉Transformer（T-MDS-ViT）模型，用于基于毫米波FMCW雷达微多普勒谱图的多类别目标分类。

### 背景

研究使用毫米波FMCW雷达微多普勒谱图进行目标分类，需要处理范围-速度-角度（RVA）时空张量数据。

### 目的

开发一种能够有效处理多帧MDS数据序列性特征的模型，并在目标重叠和部分遮挡情况下保持可分离性。

### 方法

设计了一种基于Transformer的架构，通过补丁嵌入和跨轴注意力机制处理堆叠的RVA时空张量，以显式建模跨多帧的MDS数据序列特性。在注意力层中利用移动感知约束来保持目标重叠和部分遮挡情况下的可分离性。应用可解释性机制来检查注意力层如何关注MDS表示的特征高能区域及其对类别特定运动特征的影响。

### 主要发现

所提出的框架在分类准确性方面优于现有的基于CNN的方法，同时实现了更好的数据效率和实时部署能力。

### 结论

T-MDS-ViT模型能够有效处理多帧MDS数据的序列性特征，在目标分类任务中表现优异，特别是在处理目标重叠和部分遮挡情况时具有优势。

### 翻译

在本文中，我们提出了一种新的时间MDS视觉Transformer（T-MDS-ViT）模型，用于使用毫米波FMCW雷达微多普勒谱图进行多类别目标分类。具体而言，我们设计了一种基于Transformer的架构，通过补丁嵌入和跨轴注意力机制处理堆叠的范围-速度-角度（RVA）时空张量，以显式建模跨多帧的MDS数据的序列特性。T-MDS-ViT在其注意力层对应关系中利用移动感知约束，以在目标重叠和部分遮挡情况下保持可分离性。接下来，我们应用了一种可解释机制来检查注意力层如何关注MDS表示的特征高能区域及其对类别特定运动特征的影响。我们还证明，所提出的框架在分类准确性方面优于现有的基于CNN的方法，同时实现了更好的数据效率和实时部署能力。


### 论文摘要

In this paper, we propose a new Temporal MDS-Vision Transformer (T-MDS-ViT) for multiclass target classification using millimeter-wave FMCW radar micro-Doppler spectrograms. Specifically, we design a transformer-based architecture that processes stacked range-velocity-angle (RVA) spatiotemporal tensors via patch embeddings and cross-axis attention mechanisms to explicitly model the sequential nature of MDS data across multiple frames. The T-MDS-ViT exploits mobility-aware constraints in its attention layer correspondences to maintain separability under target overlaps and partial occlusions. Next, we apply an explainable mechanism to examine how the attention layers focus on characteristic high-energy regions of the MDS representations and their effect on class-specific kinematic features. We also demonstrate that our proposed framework is superior to existing CNN-based methods in terms of classification accuracy while achieving better data efficiency and real-time deployability.

---

## 199. Transformers vs. Recurrent Models for Estimating Forest Gross Primary Production

**论文链接:** [http://arxiv.org/abs/2511.11880v1](http://arxiv.org/abs/2511.11880v1)

**作者:** David Montero, Miguel D. Mahecha, Francesco Martinuzzi, César Aybar, Anne Klosterhalfen, Alexander Knohl, Jesús Anaya, Clemens Mosig, Sebastian Wieneke

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究探索了两种深度学习模型（GPT-2和LSTM）在预测森林二氧化碳吸收（GPP）方面的性能，比较了它们在多模态输入下的表现，并分析了影响预测准确性的关键因素。

### 背景

监测森林二氧化碳吸收（GPP）的时空动态是陆地生态系统研究中的核心挑战。现有方法如涡度协方差塔提供高频估计但空间覆盖有限，而遥感方法虽可扩展但多依赖单传感器光谱指数和统计模型，难以捕捉GPP的复杂时间动态。

### 目的

探索深度学习和数据融合技术更好地表示植被过程时间动态的机会，并比较最先进深度学习模型在多模态GPP预测方面的性能。

### 方法

使用两种代表性深度学习模型预测GPP：1) GPT-2（transformer架构）和2) LSTM（循环神经网络），并采用多变量输入进行实验。

### 主要发现

两种模型总体达到相似准确性；LSTM整体表现更好，但GPT-2在极端事件中表现更优；LSTM使用比GPT-2短得多的输入窗口就能获得相似准确性，显示两者存在准确性-效率权衡；辐射是最主要预测因子，其次是Sentinel-2、MODIS地表温度和Sentinel-1的贡献。

### 结论

模型架构、上下文长度和多模态输入共同决定了GPP预测性能，研究结果为监测陆地碳动力的深度学习框架未来发展提供了指导。

### 翻译

监测森林二氧化碳吸收（总初级生产量，GPP）的时空动态仍然是陆地生态系统研究中的一个中心挑战。虽然涡度协方差塔提供高频估计，但它们有限的空间覆盖限制了大规模评估。遥感提供了可扩展的替代方案，但大多数方法依赖于单传感器光谱指数和统计模型，这些模型通常无法捕捉GPP的复杂时间动态。深度学习（DL）和数据融合的最新进展提供了更好地表示植被过程时间动态的新机会，但最先进的DL模型在多模态GPP预测方面的比较评估仍然很少。在这里，我们探索了两种代表性模型预测GPP的性能：1) GPT-2，一种transformer架构，和2) 长短期记忆（LSTM），一种循环神经网络，使用多变量输入。总体而言，两者都达到了相似的准确性。但是，虽然LSTM整体表现更好，但GPT-2在极端事件中表现出色。对时间上下文长度的进一步分析表明，LSTM使用比GPT-2短得多的输入窗口就能获得相似的准确性，突显了两种架构之间的准确性-效率权衡。特征重要性分析显示辐射是主要预测因子，其次是Sentinel-2、MODIS地表温度和Sentinel-1的贡献。我们的结果展示了模型架构、上下文长度和多模态输入如何共同决定GPP预测中的性能，为监测陆地碳动力学的DL框架的未来发展提供了指导。


### 论文摘要

Monitoring the spatiotemporal dynamics of forest CO$_2$ uptake (Gross Primary Production, GPP), remains a central challenge in terrestrial ecosystem research. While Eddy Covariance (EC) towers provide high-frequency estimates, their limited spatial coverage constrains large-scale assessments. Remote sensing offers a scalable alternative, yet most approaches rely on single-sensor spectral indices and statistical models that are often unable to capture the complex temporal dynamics of GPP. Recent advances in deep learning (DL) and data fusion offer new opportunities to better represent the temporal dynamics of vegetation processes, but comparative evaluations of state-of-the-art DL models for multimodal GPP prediction remain scarce. Here, we explore the performance of two representative models for predicting GPP: 1) GPT-2, a transformer architecture, and 2) Long Short-Term Memory (LSTM), a recurrent neural network, using multivariate inputs. Overall, both achieve similar accuracy. But, while LSTM performs better overall, GPT-2 excels during extreme events. Analysis of temporal context length further reveals that LSTM attains similar accuracy using substantially shorter input windows than GPT-2, highlighting an accuracy-efficiency trade-off between the two architectures. Feature importance analysis reveals radiation as the dominant predictor, followed by Sentinel-2, MODIS land surface temperature, and Sentinel-1 contributions. Our results demonstrate how model architecture, context length, and multimodal inputs jointly determine performance in GPP prediction, guiding future developments of DL frameworks for monitoring terrestrial carbon dynamics.

---

## 200. From Power to Precision: Learning Fine-grained Dexterity for Multi-fingered Robotic Hands

**论文链接:** [http://arxiv.org/abs/2511.13710v1](http://arxiv.org/abs/2511.13710v1)

**作者:** Jianglong Ye, Lai Wei, Guangqi Jiang, Changwei Jing, Xueyan Zou, Xiaolong Wang

**发布时间:** 2025-11-17

**备注:** Project page: https://jianglongye.com/power-to-precision

### GPT解析

### 总结

研究通过联合优化多指灵巧手的控制和硬件设计，解决了机器人手难以同时实现强力抓取和精确操作的问题，引入轻量化指尖几何结构修改，实现了从强力抓取到精确抓取的动态切换。

### 背景

人类抓握分为强力抓取和精确抓取两类，精确抓取对人类进化有重要影响。当前多指机器人手在强力抓取方面有效，但在精确任务中仍广泛采用平行夹持器，这突显了机器人手设计的关键局限：难以在单一系统中同时实现稳定抓取和精确操作。

### 目的

通过联合优化多指灵巧手的控制和硬件设计，实现同时具备强力抓取和精确操作能力的多功能系统，填补当前机器人手设计的空白。

### 方法

引入轻量化指尖几何结构修改并表示为接触面，联合优化指尖参数和控制策略；开发动态切换于强力抓取和精确操作之间的控制策略，将精确控制简化为平行拇指-食指运动；利用大规模模拟和可微分神经物理代理模型优化指尖几何结构；在模拟到现实和现实到现实环境中进行实验验证。

### 主要发现

在模拟到现实精确抓取任务中，对未见物体实现了82.5%的零样本成功率；在涉及捏面包等现实任务中达到了93.3%的成功率；证明共同设计框架可显著增强多指手的精细操作能力而不降低强力抓取能力。

### 结论

通过共同设计和优化控制与硬件，成功解决了多指灵巧手难以同时实现强力抓取和精确操作的问题，为机器人手设计提供了新思路和方法。

### 翻译

人类抓握大致可分为两种类型：强力抓取和精确抓取。精确抓取能够实现工具使用，并被认为对人类进化产生了影响。当今的多指机器人手在强力抓取方面有效，但在需要精确性的任务中，平行夹持器仍被更广泛采用。这种对比突显了当前机器人手设计中的一个关键局限：难以在单一多功能系统中同时实现稳定的强力抓取和精确的精细操作。在这项工作中，我们通过联合优化多指灵巧手的控制和硬件设计来填补这一空白，使其同时具备强力抓取和精确操作能力。我们没有重新设计整个手，而是引入了一种轻量化的指尖几何结构修改，将其表示为接触面，并联合优化其参数和相应控制。我们的控制策略动态切换于强力抓取和精确操作之间，并将精确控制简化为平行的拇指-食指运动，这证明在模拟到现实迁移中具有鲁棒性。在设计方面，我们利用大规模模拟，使用可微分的神经物理代理模型来优化指尖几何结构。我们在模拟到现实和现实到现实环境中通过大量实验验证了我们的方法。我们的方法在模拟到现实精确抓取中对未见物体实现了82.5%的零样本成功率，在涉及捏面包等具有挑战性的现实任务中达到了93.3%的成功率。这些结果表明，我们的共同设计框架可以在不降低强力抓取能力的情况下，显著增强多指手的精细操作能力。我们的项目页面位于https://jianglongye.com/power-to-precision


### 论文摘要

Human grasps can be roughly categorized into two types: power grasps and precision grasps. Precision grasping enables tool use and is believed to have influenced human evolution. Today's multi-fingered robotic hands are effective in power grasps, but for tasks requiring precision, parallel grippers are still more widely adopted. This contrast highlights a key limitation in current robotic hand design: the difficulty of achieving both stable power grasps and precise, fine-grained manipulation within a single, versatile system. In this work, we bridge this gap by jointly optimizing the control and hardware design of a multi-fingered dexterous hand, enabling both power and precision manipulation. Rather than redesigning the entire hand, we introduce a lightweight fingertip geometry modification, represent it as a contact plane, and jointly optimize its parameters along with the corresponding control. Our control strategy dynamically switches between power and precision manipulation and simplifies precision control into parallel thumb-index motions, which proves robust for sim-to-real transfer. On the design side, we leverage large-scale simulation to optimize the fingertip geometry using a differentiable neural-physics surrogate model. We validate our approach through extensive experiments in both sim-to-real and real-to-real settings. Our method achieves an 82.5% zero-shot success rate on unseen objects in sim-to-real precision grasping, and a 93.3% success rate in challenging real-world tasks involving bread pinching. These results demonstrate that our co-design framework can significantly enhance the fine-grained manipulation ability of multi-fingered hands without reducing their ability for power grasps. Our project page is at https://jianglongye.com/power-to-precision

---

## 201. Cross-Learning from Scarce Data via Multi-Task Constrained Optimization

**论文链接:** [http://arxiv.org/abs/2511.13680v1](http://arxiv.org/abs/2511.13680v1)

**作者:** Leopoldo Agorio, Juan Cerviño, Miguel Calvo-Fullana, Alejandro Ribeiro, Juan Andrés Bazerque

**发布时间:** 2025-11-17

**备注:** 13 pages, 11 figures

### GPT解析

### 总结

本文提出了一种多任务交叉学习框架，通过联合估计多个相关任务的确定性参数来克服数据稀缺问题，使模型能够更好地泛化到未见过的案例。

### 背景

学习任务需要足够大的数据集来代表源数据的底层分布，但当数据有限时，学习到的模型无法很好地泛化到训练中未见过的案例。

### 目的

克服数据稀缺问题，通过联合估计多个相关任务的确定性参数，实现知识从数据丰富任务到数据稀缺任务的转移。

### 方法

引入多任务交叉学习框架，将联合估计表述为约束优化问题，约束条件规定不同模型参数之间的相似性，允许参数在不同任务间有所不同，同时结合多个数据源的信息。

### 主要发现

该框架能提供更准确、可靠的参数估计；在从有限数据进行参数推断的关键场景中有效；在高斯数据控制框架中有理论保证；在图像分类和传染病传播等真实数据应用中展示了高效性。

### 结论

多任务交叉学习框架通过约束优化实现参数的联合估计，有效解决了数据稀缺问题，促进了知识在不同任务间的转移，为有限数据场景下的参数推断提供了可靠解决方案。

### 翻译

学习任务被理解为从监督数据拟合参数模型的问题，这需要数据集足够大以代表源数据的底层分布。当数据有限时，学习到的模型无法推广到训练中未见的案例。本文引入了一种多任务交叉学习框架，通过联合估计多个相关任务的确定性参数来克服数据稀缺问题。我们将这种联合估计表述为约束优化问题，其中约束条件规定了不同模型参数之间的相似性，允许不同任务的参数有所不同，同时仍能结合多个数据源的信息。该框架实现了从数据丰富任务到数据稀缺任务的知识转移，从而获得更准确、可靠的参数估计，为从有限数据进行参数推断的关键场景提供了解决方案。我们在高斯数据的控制框架中提供了理论保证，并在包括图像分类和传染病传播在内的真实数据应用中展示了交叉学习方法的效率。


### 论文摘要

A learning task, understood as the problem of fitting a parametric model from supervised data, fundamentally requires the dataset to be large enough to be representative of the underlying distribution of the source. When data is limited, the learned models fail generalize to cases not seen during training. This paper introduces a multi-task \emph{cross-learning} framework to overcome data scarcity by jointly estimating \emph{deterministic} parameters across multiple, related tasks. We formulate this joint estimation as a constrained optimization problem, where the constraints dictate the resulting similarity between the parameters of the different models, allowing the estimated parameters to differ across tasks while still combining information from multiple data sources. This framework enables knowledge transfer from tasks with abundant data to those with scarce data, leading to more accurate and reliable parameter estimates, providing a solution for scenarios where parameter inference from limited data is critical. We provide theoretical guarantees in a controlled framework with Gaussian data, and show the efficiency of our cross-learning method in applications with real data including image classification and propagation of infectious diseases.

---

## 202. Systematic evaluation of time-frequency features for binaural sound source localization

**论文链接:** [http://arxiv.org/abs/2511.13487v1](http://arxiv.org/abs/2511.13487v1)

**作者:** Davoud Shariat Panah, Alessandro Ragano, Dan Barry, Jan Skoglund, Andrew Hines

**发布时间:** 2025-11-17

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本研究系统评估了双耳声源定位中的时频特征设计，发现精心选择特征组合比增加模型复杂度更有效，最优特征组合下的低复杂度CNN模型实现了具有竞争力的性能。

### 背景

双耳声源定位(SSL)是一个重要但具有挑战性的任务，特别是在不同条件下，特征选择对模型性能有重要影响。

### 目的

评估时频特征设计对双耳声源定位的影响，研究特征选择如何影响模型在不同条件下的性能，并提供实用指导。

### 方法

使用卷积神经网络(CNN)模型测试基于幅度(幅度谱图、ILD)和相位(相位谱图、IPD)的不同特征组合，在域内和域外数据上进行评估，使用不匹配的头部相关传输函数(HRTF)。

### 主要发现

精心选择的特征组合通常比增加模型复杂度更有效；域内SSL仅需两个特征集(如ILD + IPD)；对多样化内容的泛化需要结合通道谱图与ILD和IPD的更丰富输入；最优特征集下低复杂度CNN模型实现竞争力性能。

### 结论

特征设计在双耳SSL中起着重要作用，研究为特定领域和通用定位提供了实用指导。

### 翻译

本研究对双耳声源定位(SSL)的时频特征设计进行了系统评估，重点关注特征选择如何影响模型在不同条件下的性能。我们研究了使用各种基于幅度(幅度谱图、双耳级差ILD)和基于相位(相位谱图、双耳相位差IPD)的特征组合的卷积神经网络(CNN)模型的性能。在具有不匹配的头部相关传输函数(HRTF)的域内和域外数据上的评估表明，精心选择的特征组合通常比增加模型复杂度更有效。虽然对于域内SSL，两个特征集(如ILD + IPD)就足够了，但对多样化内容的泛化需要结合通道谱图与ILD和IPD的更丰富输入。使用最优特征集，我们的低复杂度CNN模型实现了具有竞争力的性能。我们的研究结果强调了特征设计在双耳SSL中的重要性，并为特定领域和通用定位提供了实用指导。


### 论文摘要

This study presents a systematic evaluation of time-frequency feature design for binaural sound source localization (SSL), focusing on how feature selection influences model performance across diverse conditions. We investigate the performance of a convolutional neural network (CNN) model using various combinations of amplitude-based features (magnitude spectrogram, interaural level difference - ILD) and phase-based features (phase spectrogram, interaural phase difference - IPD). Evaluations on in-domain and out-of-domain data with mismatched head-related transfer functions (HRTFs) reveal that carefully chosen feature combinations often outperform increases in model complexity. While two-feature sets such as ILD + IPD are sufficient for in-domain SSL, generalization to diverse content requires richer inputs combining channel spectrograms with both ILD and IPD. Using the optimal feature sets, our low-complexity CNN model achieves competitive performance. Our findings underscore the importance of feature design in binaural SSL and provide practical guidance for both domain-specific and general-purpose localization.

---

## 203. Discovering Operational Patterns Using Image-Based Convolutional Clustering and Composite Evaluation: A Case Study in Foundry Melting Processes

**论文链接:** [http://arxiv.org/abs/2511.13444v1](http://arxiv.org/abs/2511.13444v1)

**作者:** Zhipeng Ma, Bo Nørregaard Jørgensen, Zheng Grace Ma

**发布时间:** 2025-11-17

**DOI:** 10.3390/info16090816

### GPT解析

### 总结

本文提出了一种基于图像的卷积聚类与复合内部评估相结合的新框架，用于工业单变量时间序列数据中操作模式的非监督发现，解决了传统方法在处理动态、非结构化工业序列时的局限性。

### 背景

工业过程监测依赖传感器生成的时间序列数据，但缺乏标签、高变性和操作噪声使传统方法难以提取有意义模式。现有聚类技术要么依赖固定距离度量，要么为静态数据设计，难以处理动态工业序列。

### 目的

解决非监督时间序列分析中的关键挑战，如序列不规则性、重叠模式和度量不一致性，为工业系统中的数据驱动诊断和能源优化提供可推广的解决方案。

### 方法

提出基于图像的卷积聚类与复合内部评估框架，包括：1)通过重叠滑动窗口将时间序列转换为灰度矩阵表示，使用深度卷积自编码器提取特征；2)整合软聚类和硬聚类输出，通过两阶段策略优化选择；3)使用复合分数S_eva评估聚类性能，结合标准化的Silhouette、Calinski-Harabasz和Davies-Bouldin指数。

### 主要发现

应用于北欧铸造厂3900多个炉熔炼操作，识别出七个可解释的操作模式，揭示了能源消耗、热动力学和生产持续时间方面的显著差异。与基线方法相比，实现了优越的整体性能、更强的鲁棒性和领域一致的可解释性。

### 结论

该框架解决了非监督时间序列分析中的关键挑战，为工业系统中的数据驱动诊断和能源优化提供了可推广的解决方案。

### 翻译

工业过程监测越来越依赖传感器生成的时间序列数据，但缺乏标签、高变性和操作噪声使得使用传统方法提取有意义模式变得困难。现有的聚类技术要么依赖固定的距离度量，要么为静态数据设计深度模型，限制了它们处理动态、非结构化工业序列的能力。针对这一差距，本文提出了一种基于图像的卷积聚类与复合内部评估相结合的新颖框架，用于单变量时间序列数据中操作模式的非监督发现。该框架在三个方面改进了现有方法：1)通过重叠滑动窗口将原始时间序列转换为灰度矩阵表示，允许使用深度卷积自编码器进行有效特征提取；2)框架整合软聚类和硬聚类输出，并通过两阶段策略优化选择；3)通过新开发的复合分数S_eva客观评估聚类性能，该分数结合了标准化的Silhouette、Calinski-Harabasz和Davies-Bouldin指数。应用于来自北欧铸造厂超过3900个炉熔炼操作，该方法识别出七个可解释的操作模式，揭示了能源消耗、热动力学和生产持续时间方面的显著差异。与经典和深度聚类基线相比，所提出的方法实现了优越的整体性能、更强的鲁棒性和领域一致的可解释性。该框架解决了非监督时间序列分析中的关键挑战，如序列不规则性、重叠模式和度量不一致性，为工业系统中的数据驱动诊断和能源优化提供了可推广的解决方案。


### 论文摘要

Industrial process monitoring increasingly relies on sensor-generated time-series data, yet the lack of labels, high variability, and operational noise make it difficult to extract meaningful patterns using conventional methods. Existing clustering techniques either rely on fixed distance metrics or deep models designed for static data, limiting their ability to handle dynamic, unstructured industrial sequences. Addressing this gap, this paper proposes a novel framework for unsupervised discovery of operational modes in univariate time-series data using image-based convolutional clustering with composite internal evaluation. The proposed framework improves upon existing approaches in three ways: (1) raw time-series sequences are transformed into grayscale matrix representations via overlapping sliding windows, allowing effective feature extraction using a deep convolutional autoencoder; (2) the framework integrates both soft and hard clustering outputs and refines the selection through a two-stage strategy; and (3) clustering performance is objectively evaluated by a newly developed composite score, S_eva, which combines normalized Silhouette, Calinski-Harabasz, and Davies-Bouldin indices. Applied to over 3900 furnace melting operations from a Nordic foundry, the method identifies seven explainable operational patterns, revealing significant differences in energy consumption, thermal dynamics, and production duration. Compared to classical and deep clustering baselines, the proposed approach achieves superior overall performance, greater robustness, and domain-aligned explainability. The framework addresses key challenges in unsupervised time-series analysis, such as sequence irregularity, overlapping modes, and metric inconsistency, and provides a generalizable solution for data-driven diagnostics and energy optimization in industrial systems.

---

## 204. Explainable RL Policies by Distilling to Locally-Specialized Linear Policies with Voronoi State Partitioning

**论文链接:** [http://arxiv.org/abs/2511.13322v1](http://arxiv.org/abs/2511.13322v1)

**作者:** Senne Deproost, Dennis Steckelmacher, Ann Nowé

**发布时间:** 2025-11-17

**备注:** Accepted for BNAIC/BeNeLearn 2025

### GPT解析

### 总结

本文提出了一种新的模型无关方法，将状态空间划分为区域，在每个区域内使用简化的、人类可理解的线性模型来操作，以解决深度强化学习模型缺乏透明度的问题。

### 背景

深度强化学习是最先进的产生近最优系统控制器的方法之一，但深度神经网络缺乏透明度，在需要满足法规或促进信任时带来挑战。现有知识蒸馏方法通常使用单一模型模仿原始模型平均行为，但在动态情况下表现不佳。

### 目的

解决深度强化学习模型缺乏透明度的问题，通过将学习行为转移到人类可读模型中，同时保持模型灵活性和复杂度之间的平衡，或偏差和准确性之间的平衡。

### 方法

提出一种新的模型无关方法，使用Voronoi分区将状态空间划分为区域，在每个区域内寻找线性模型能够达到与原始控制器相似性能的区域。在网格世界环境和经典控制任务上评估该方法。

### 主要发现

将知识蒸馏到局部专业化的线性模型中产生的策略是可解释的，且这种蒸馏方法匹配甚至略微优于其所蒸馏的黑色盒子策略。

### 结论

通过状态空间划分和在每个区域内使用线性模型，成功将深度强化学习模型的复杂行为转化为可解释形式，同时保持或提高了原始模型的性能。

### 翻译

深度强化学习是产生近最优系统控制器的最先进方法之一。然而，深度RL算法训练的深度神经网络缺乏透明度，这在控制器必须满足法规或促进信任时带来了挑战。为了缓解这一问题，可以通过知识蒸馏将学习到的行为转移到天生人类可读的模型中。这通常是通过使用单一模型来模仿原始模型的平均行为来完成的，但在更动态的情况下可能会遇到困难。一个关键挑战是这个简单模型应该在灵活性和复杂性之间取得适当的平衡，或在偏差和准确性之间取得适当的平衡。我们提出了一种新的模型无关方法，将状态空间划分为区域，在这些区域内简化的、人类可理解的模型可以操作。在本文中，我们使用Voronoi分区来找到线性模型能够实现与原始控制器相似性能的区域。我们在网格世界环境和经典控制任务上评估了我们的方法。我们观察到，我们提出的局部专业化线性模型蒸馏产生了可解释的策略，并且这种蒸馏匹配甚至略微优于它们所蒸馏的黑色盒子策略。


### 论文摘要

Deep Reinforcement Learning is one of the state-of-the-art methods for producing near-optimal system controllers. However, deep RL algorithms train a deep neural network, that lacks transparency, which poses challenges when the controller has to meet regulations, or foster trust. To alleviate this, one could transfer the learned behaviour into a model that is human-readable by design using knowledge distilla- tion. Often this is done with a single model which mimics the original model on average but could struggle in more dynamic situations. A key challenge is that this simpler model should have the right balance be- tween flexibility and complexity or right balance between balance bias and accuracy. We propose a new model-agnostic method to divide the state space into regions where a simplified, human-understandable model can operate in. In this paper, we use Voronoi partitioning to find regions where linear models can achieve similar performance to the original con- troller. We evaluate our approach on a gridworld environment and a classic control task. We observe that our proposed distillation to locally- specialized linear models produces policies that are explainable and show that the distillation matches or even slightly outperforms the black-box policy they are distilled from.

---

## 205. Voltage-Based Unsupervised Learning Framework for Bridge Damage Detection in Simultaneous Energy Harvesting and Sensing Systems

**论文链接:** [http://arxiv.org/abs/2511.13291v1](http://arxiv.org/abs/2511.13291v1)

**作者:** S. Yao, P. Peralta-Braz, A. Calderon Hurtado, R. Das, M. M. Alamdari, E. Atroshchenko

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究设计压电能量收集器（PEHs）实现结构健康监测的双重功能：从桥梁振动中收集电能并作为损伤传感器。通过直接使用电压信号作为传感输入，消除了传统传感模块的需求，降低了系统复杂性和能耗。

### 背景

传统结构健康监测系统复杂且能耗高，需要寻找更高效、低能耗的监测方法。

### 目的

设计压电能量收集器（PEHs）实现双重功能：从桥梁振动中收集电能并作为损伤传感器，同时优化系统以最大化能量输出和损伤检测精度。

### 方法

提出双目标优化框架，使用等几何分析预测电压响应，在数值车辆-桥梁相互作用模型和实验室测试中验证，应用卷积变分自编码器进行无监督损伤检测，使用NSGA-II算法探索能量产量和传感精度之间的权衡，并进行参数研究。

### 主要发现

优化的PEH不仅作为有效的滤波器和传感组件，还优于基于加速度的传统传感，损伤检测精度提高13%，能耗降低98%，多参数设计空间突显了双目标优化的重要性。

### 结论

证明了用轻量级、自供电的PEH替代传统传感器的可行性，为可持续的同步能量收集和传感系统铺平了道路。

### 翻译

本研究设计压电能量收集器（PEHs）为结构健康监测（SHM）提供双重功能：从桥梁振动中收集电能，同时作为内在损伤传感器。该策略直接使用电压信号作为传感输入，消除了传统传感模块的需求，从而降低了系统复杂性和能耗。提出了一种双目标优化框架，以最大化复合悬臂基尔霍夫-乐夫板建模的PEH模型的功率输出和损伤检测精度。通过等几何分析预测现实桥梁输入下的电压响应。该方法在两种场景中进行了验证：数值车辆-桥梁相互作用模型和使用玩具车的实验室规模梁测试，每种场景均在健康和损坏状态下进行评估。使用仅基于健康电压信号训练的卷积变分自编码器（CVAE）实现无监督损伤检测。应用NSGA-II算法探索能量产量和传感精度之间的权衡，包括对损伤严重程度、损伤位置和收集器几何形状的参数研究。结果表明，优化的PEH不仅作为有效的滤波器和传感组件，而且优于传统的基于加速度的传感，将损伤检测精度提高13%，同时将能耗降低98%。多参数设计空间进一步突显了双目标优化的重要性，即使在共振条件下性能也会变化。这些发现证明了用轻量级、自供电的PEH替代传统传感器的可行性，并为可持续的同步能量收集和传感（SEHS）系统铺平了道路。


### 论文摘要

In this study, piezoelectric energy harvesters (PEHs) are designed to offer dual functionality in structural health monitoring (SHM): harvesting electric power from bridge vibrations while serving as intrinsic damage sensors. This strategy utilises the voltage signal directly as the sensing input, eliminating the need for traditional sensing modules and thereby reducing system complexity and energy consumption. A bi-objective optimisation framework is proposed to maximise both power output and damage detection accuracy of a PEH modelled as a composite cantilevered Kirchhoff-Love plate. Voltage responses under realistic bridge inputs are predicted via isogeometric analysis. The approach is validated in two scenarios: a numerical vehicle-bridge interaction model and a laboratory-scale beam test using a toy car, each evaluated in both healthy and damaged states. Unsupervised damage detection is achieved using a convolutional variational autoencoder (CVAE) trained solely on healthy voltage signatures. The NSGA-II algorithm is applied to explore trade-offs between energy yield and sensing precision, including parametric studies on damage severity, damage location, and harvester geometry. Results indicate that optimised PEHs not only act as an effective filter and sensing component but also outperform traditional acceleration-based sensing, improving damage detection accuracy by 13% while reducing energy consumption by 98%. The multi-parameter design space further highlights the importance of bi-objective optimisation due to variations in performance even under resonant conditions. These findings demonstrate the feasibility of replacing traditional sensors with lightweight, self-powered PEHs and pave the way for sustainable simultaneous energy harvesting and sensing (SEHS) systems.

---

## 206. KForge: Program Synthesis for Diverse AI Hardware Accelerators

**论文链接:** [http://arxiv.org/abs/2511.13274v1](http://arxiv.org/abs/2511.13274v1)

**作者:** Taras Sereda, Tom St. John, Burak Bartan, Natalie Serrino, Sachin Katti, Zain Asgar

**发布时间:** 2025-11-17

**备注:** Under review at MLSys 2026

### GPT解析

### 总结

KForge是一个基于两个协作LLM智能体的平台无关框架，用于优化GPU kernels。该框架包含生成智能体和性能分析智能体，通过迭代完善和性能分析来指导程序合成，仅需单次示例即可针对新平台。

### 背景

GPU kernels对机器学习性能至关重要，但很难在不同的加速器上进行优化。

### 目的

提出一个平台无关的框架KForge，解决GPU kernels跨平台优化的问题。

### 方法

构建两个基于LLM的协作智能体：1)生成智能体通过编译和正确性反馈生成和迭代完善程序；2)性能分析智能体解释分析数据以指导优化。引入迭代完善系统，使智能体通过功能通道和优化通道协作，解释各种分析数据生成可操作建议。

### 主要发现

1)生成智能体有效利用跨平台知识迁移，参考实现可提高不同硬件目标的生成质量；2)该方法在根本上不同的并行计算平台(NVIDIA CUDA和Apple Metal)上都有效；3)基于LLM的智能体架构可以有效地优化GPU kernels。

### 结论

KForge是一个平台无关的框架，通过两个基于LLM的协作智能体优化GPU kernels，仅需单次示例即可针对新平台，并在不同平台上均表现出色。

### 翻译

GPU kernels对机器学习性能至关重要但难以在多样化的加速器上进行优化。我们提出了KForge，一个构建在两个协作的基于LLM的智能体之上的平台无关框架：一个生成智能体，通过编译和正确性反馈生成和迭代完善程序；以及一个性能分析智能体，解释分析数据以指导优化。这种基于智能体的架构只需要单次示例即可针对新平台。我们做出了三个关键贡献：(1)引入了一个迭代完善系统，其中生成智能体和性能分析智能体通过功能通道和优化通道协作，解释各种分析数据(从程序化API到基于GUI的工具)，生成指导任意加速器程序合成的可操作建议；(2)证明生成智能体有效利用了跨平台知识迁移，来自一个架构的参考实现显著提高了不同硬件目标的生成质量；(3)通过展示在根本上不同的并行计算平台(NVIDIA CUDA和Apple Metal)上的有效程序合成，验证了我们方法的平台无关性。


### 论文摘要

GPU kernels are critical for ML performance but difficult to optimize across diverse accelerators. We present KForge, a platform-agnostic framework built on two collaborative LLM-based agents: a generation agent that produces and iteratively refines programs through compilation and correctness feedback, and a performance analysis agent that interprets profiling data to guide optimization. This agent-based architecture requires only a single-shot example to target new platforms.   We make three key contributions: (1) introducing an iterative refinement system where the generation agent and performance analysis agent collaborate through functional and optimization passes, interpreting diverse profiling data (from programmatic APIs to GUI-based tools) to generate actionable recommendations that guide program synthesis for arbitrary accelerators; (2) demonstrating that the generation agent effectively leverages cross-platform knowledge transfer, where a reference implementation from one architecture substantially improves generation quality for different hardware targets; and (3) validating the platform-agnostic nature of our approach by demonstrating effective program synthesis across fundamentally different parallel computing platforms: NVIDIA CUDA and Apple Metal.

---

## 207. Computational Measurement of Political Positions: A Review of Text-Based Ideal Point Estimation Algorithms

**论文链接:** [http://arxiv.org/abs/2511.13238v1](http://arxiv.org/abs/2511.13238v1)

**作者:** Patrick Parschan, Charlott Jakob

**发布时间:** 2025-11-17

**备注:** 46 pages, 8 figures, 2 tables, accepted for publication in Quality & Quantity

### GPT解析

### 总结

本文首次对无监督和半监督的计算文本理想点估计(CT-IPE)算法进行了系统性综述，这些算法旨在从文本数据中推断潜在的政治立场。作者识别出四种方法学家族：词频、主题建模、词嵌入和基于大型语言模型的方法，并提供了三个主要贡献：结构化合成算法发展历程、为应用研究者提供实用指导，以及强调系统性基准测试的必要性。

### 背景

CT-IPE算法被广泛应用于政治科学、传播学、计算社会科学和计算机科学领域，用于从议会演讲、政党宣言和社交媒体中估计意识形态偏好。过去二十年来，这些算法的发展与自然语言处理的整体趋势紧密相关，从早期的词频模型发展到最近的大型语言模型。虽然这一演变扩展了方法论工具箱，但也导致了一个碎片化的研究领域，缺乏系统性的比较和明确的实际应用指导。

### 目的

填补CT-IPE研究领域缺乏系统性比较和明确应用指导的空白，通过系统文献综述确定CT-IPE算法，引入概念框架进行比较，识别不同的方法学家族，批判性评估它们的优缺点，并为应用研究者提供实用指导。

### 方法

通过系统文献综述确定了25种CT-IPE算法，并进行了手动内容分析研究它们的建模假设和发展背景。引入了一个概念框架来区分算法如何生成、捕获和聚合文本方差，基于这个框架识别出四种方法学家族，并对它们的假设、可解释性、可扩展性和局限性进行了批判性评估。

### 主要发现

CT-IPE算法经历了从词频模型到大型语言模型的发展轨迹；研究领域存在碎片化问题；识别出四种方法学家族：词频、主题建模、词嵌入和基于大型语言模型的方法；不同算法在估计结果上的差异本身是有信息量的，强调了系统性基准测试的必要性。

### 结论

该综述为应用研究者提供了三个主要贡献：提供了二十年来算法发展的结构化合成，阐明了不同方法之间的关系；将这些见解转化为实用指导，强调了透明度、技术要求和验证策略之间的权衡；强调了不同算法之间的估计结果差异本身是有信息量的，突显了系统性基准测试的必要性。

### 翻译

本文首次对无监督和半监督的计算文本理想点估计(CT-IPE)算法进行了系统性综述，这些算法旨在从文本数据中推断潜在的政治立场。这些算法被广泛应用于政治科学、传播学、计算社会科学和计算机科学领域，用于从议会演讲、政党宣言和社交媒体中估计意识形态偏好。在过去二十年中，它们的发展与更广泛的NLP趋势紧密相连——从词频模型开始，最近转向大型语言模型。虽然这一轨迹极大地扩展了方法论工具箱，但也产生了一个碎片化的研究领域，缺乏系统性的比较和明确的实际应用指导。为解决这一空白，我们通过系统文献综述确定了25种CT-IPE算法，并对它们的建模假设和发展背景进行了手动内容分析。为了有意义地比较这些算法，我们引入了一个概念框架，用于区分算法如何生成、捕获和聚合文本方差。在此基础上，我们识别出四种方法学家族——词频、主题建模、词嵌入和基于LLM的方法，并批判性评估了它们的假设、可解释性、可扩展性和局限性。我们的综述提供了三个贡献。首先，它提供了二十年来算法发展的结构化合成，阐明了不同方法之间的关系。其次，它将这些见解转化为应用研究者的实用指导，强调了透明度、技术要求和验证策略之间的权衡，这些权衡影响着算法的选择。第三，它强调了不同算法之间的估计结果差异本身是有信息量的，突显了系统性基准测试的必要性。


### 论文摘要

This article presents the first systematic review of unsupervised and semi-supervised computational text-based ideal point estimation (CT-IPE) algorithms, methods designed to infer latent political positions from textual data. These algorithms are widely used in political science, communication, computational social science, and computer science to estimate ideological preferences from parliamentary speeches, party manifestos, and social media. Over the past two decades, their development has closely followed broader NLP trends -- beginning with word-frequency models and most recently turning to large language models (LLMs). While this trajectory has greatly expanded the methodological toolkit, it has also produced a fragmented field that lacks systematic comparison and clear guidance for applied use. To address this gap, we identified 25 CT-IPE algorithms through a systematic literature review and conducted a manual content analysis of their modeling assumptions and development contexts. To compare them meaningfully, we introduce a conceptual framework that distinguishes how algorithms generate, capture, and aggregate textual variance. On this basis, we identify four methodological families -- word-frequency, topic modeling, word embedding, and LLM-based approaches -- and critically assess their assumptions, interpretability, scalability, and limitations. Our review offers three contributions. First, it provides a structured synthesis of two decades of algorithm development, clarifying how diverse methods relate to one another. Second, it translates these insights into practical guidance for applied researchers, highlighting trade-offs in transparency, technical requirements, and validation strategies that shape algorithm choice. Third, it emphasizes that differences in estimation outcomes across algorithms are themselves informative, underscoring the need for systematic benchmarking.

---

## 208. PIGEON: VLM-Driven Object Navigation via Points of Interest Selection

**论文链接:** [http://arxiv.org/abs/2511.13207v1](http://arxiv.org/abs/2511.13207v1)

**作者:** Cheng Peng, Zhenzhe Zhang, Cheng Chi, Xiaobao Wei, Yanhao Zhang, Heng Wang, Pengwei Wang, Zhongyuan Wang, Jing Liu, Shanghang Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了一种名为PIGEON的导航方法，通过视觉语言模型选择兴趣点，平衡决策频率与智能性，在未知环境中实现高效的对象导航。

### 背景

在未知环境中导航到指定对象是具身智能的基本但具有挑战性的能力。当前方法难以平衡决策频率与智能性，导致决策缺乏远见或行动不连续。

### 目的

开发一种能够平衡决策频率与智能性的方法，提高在未知环境中对象导航的性能。

### 方法

提出PIGEON方法，使用视觉语言模型(PIGEON-VL)选择探索过程中形成的兴趣点(PoI)，并结合低级规划器提高决策频率；同时生成可验证奖励强化学习(RLVR)数据增强模型语义引导能力。

### 主要发现

在经典对象导航基准上，该方法实现了零样本迁移的最先进性能；RLVR进一步增强了模型的语义引导能力，使模型能在实时导航过程中进行深度推理。

### 结论

PIGEON方法通过结合视觉语言模型和兴趣点选择机制，有效平衡了决策频率与智能性，显著提升了对象导航的性能。

### 翻译

在未知环境中导航到指定对象是具身智能的基本但具有挑战性的能力。然而，当前方法难以平衡决策频率与智能性，导致决策缺乏远见或行动不连续。在这项工作中，我们提出了PIGEON：基于视觉语言模型的对象导航兴趣点引导探索方法，在探索过程中维护一个轻量级且语义对齐的快照记忆，作为探索策略的语义输入。我们使用一个名为PIGEON-VL的大型视觉语言模型来选择探索过程中形成的兴趣点，然后采用较低级别的规划器进行动作输出，提高决策频率。此外，基于兴趣点的决策能够生成适合模拟器的可验证奖励强化学习数据。在经典对象导航基准上的实验表明，我们的零样本迁移方法实现了最先进的性能，而可验证奖励强化学习进一步增强了模型的语义引导能力，使模型能够在实时导航过程中进行深度推理。


### 论文摘要

Navigating to a specified object in an unknown environment is a fundamental yet challenging capability of embodied intelligence. However, current methods struggle to balance decision frequency with intelligence, resulting in decisions lacking foresight or discontinuous actions. In this work, we propose PIGEON: Point of Interest Guided Exploration for Object Navigation with VLM, maintaining a lightweight and semantically aligned snapshot memory during exploration as semantic input for the exploration strategy. We use a large Visual-Language Model (VLM), named PIGEON-VL, to select Points of Interest (PoI) formed during exploration and then employ a lower-level planner for action output, increasing the decision frequency. Additionally, this PoI-based decision-making enables the generation of Reinforcement Learning with Verifiable Reward (RLVR) data suitable for simulators. Experiments on classic object navigation benchmarks demonstrate that our zero-shot transfer method achieves state-of-the-art performance, while RLVR further enhances the model's semantic guidance capabilities, enabling deep reasoning during real-time navigation.

---

## 209. THIR: Topological Histopathological Image Retrieval

**论文链接:** [http://arxiv.org/abs/2511.13170v1](http://arxiv.org/abs/2511.13170v1)

**作者:** Zahra Tabatabaei, Jon Sporring

**发布时间:** 2025-11-17

### GPT解析

### 总结

本研究提出THIR，一种基于内容的医学图像检索框架，利用拓扑数据分析从组织病理学图像中提取特征，实现无监督的高效图像检索，在乳腺癌诊断领域表现优异。

### 背景

根据世界卫生组织，2020年约有685,000名女性死于乳腺癌。早期诊断和准确的临床决策对减轻这一全球负担至关重要。

### 目的

开发一种新的医学图像检索方法，用于乳腺癌组织病理学图像的检索和分析，提供一种快速、可扩展且无需训练的解决方案。

### 方法

提出THIR框架，利用拓扑数据分析特别是持久同调导出的Betti数，从RGB组织病理学图像中提取拓扑指纹，将环的演化编码为特征向量，通过计算拓扑描述符间的距离进行相似性检索。

### 主要发现

THIR在BreaKHis数据集上优于最先进的监督和非监督方法，可在标准CPU上20分钟内处理整个数据集，无需大量训练、标注数据集和GPU资源。

### 结论

THIR为临床图像检索提供了一种快速、可扩展且无需训练的解决方案，有潜力帮助乳腺癌的早期诊断和临床决策。

### 翻译

根据世界卫生组织的数据，2020年约有685,000名女性死于乳腺癌。早期诊断和准确的临床决策对于减轻这一全球负担至关重要。在本研究中，我们提出了THIR，一种新颖的基于内容的医学图像检索框架，它利用拓扑数据分析，特别是从持久同调导出的Betti数，根据其内在结构模式来表征和检索组织病理学图像。与依赖大量训练、标记数据集和强大GPU资源的传统深度学习方法不同，THIR完全在无监督下运行。它使用立方体持久性直接从RGB组织病理学图像中提取拓扑指纹，将环的演化编码为紧凑、可解释的特征向量。然后通过计算这些拓扑描述符之间的距离执行相似性检索，高效地返回top-K最相关的匹配。在BreaKHis数据集上的广泛实验表明，THIR优于最先进的监督和非监督方法。它在标准CPU上可在不到20分钟内处理整个数据集，为临床图像检索提供了快速、可扩展且无需训练的解决方案。


### 论文摘要

According to the World Health Organization, breast cancer claimed the lives of approximately 685,000 women in 2020. Early diagnosis and accurate clinical decision making are critical in reducing this global burden. In this study, we propose THIR, a novel Content-Based Medical Image Retrieval (CBMIR) framework that leverages topological data analysis specifically, Betti numbers derived from persistent homology to characterize and retrieve histopathological images based on their intrinsic structural patterns. Unlike conventional deep learning approaches that rely on extensive training, annotated datasets, and powerful GPU resources, THIR operates entirely without supervision. It extracts topological fingerprints directly from RGB histopathological images using cubical persistence, encoding the evolution of loops as compact, interpretable feature vectors. The similarity retrieval is then performed by computing the distances between these topological descriptors, efficiently returning the top-K most relevant matches.   Extensive experiments on the BreaKHis dataset demonstrate that THIR outperforms state of the art supervised and unsupervised methods. It processes the entire dataset in under 20 minutes on a standard CPU, offering a fast, scalable, and training free solution for clinical image retrieval.

---

## 210. ResAlignNet: A Data-Driven Approach for INS/DVL Alignment

**论文链接:** [http://arxiv.org/abs/2511.13096v1](http://arxiv.org/abs/2511.13096v1)

**作者:** Guy Damari, Itzik Klein

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出ResAlignNet，一种基于1D ResNet-18架构的数据驱动方法，用于解决水下航行器中惯性导航系统与多普勒速度计之间的对齐问题，实现快速收敛且无需外部辅助传感器

### 背景

自主水下航行器依赖惯性导航系统与多普勒速度计的组合导航系统，在卫星导航不可用的环境中执行任务，但传统基于模型的传感器对齐方法存在收敛时间长、依赖特定运动模式和外部辅助传感器等问题

### 目的

开发一种数据驱动的传感器对齐方法，解决传统方法的局限性，实现快速收敛、无需外部辅助传感器或复杂车辆机动的即用型解决方案

### 方法

提出ResAlignNet，使用1D ResNet-18架构将对齐问题转化为深度神经网络优化，作为即用型解决方案仅需船上传感器，同时利用Sim2Real迁移学习能力实现从合成数据训练到实际部署

### 主要发现

实验验证表明，ResAlignNet仅使用25秒数据收集即可实现0.8°以内的对齐精度，相比标准速度基方法收敛时间减少65%，且无需特定运动模式

### 结论

ResAlignNet通过稳健的传感器无关对齐提升了水下导航能力，能够扩展到不同操作场景和传感器规格，消除了传统方法的运动模式要求和冗长的任务前程序

### 翻译

自主水下航行器依赖于精确的导航系统，该系统结合了惯性导航系统和多普勒速度计，以便在卫星导航不可用的挑战性环境中成功执行任务。这种集成的有效性关键在于传感器参考帧之间的准确对齐。这些传感器系统之间的标准基于模型的对齐方法存在收敛时间长、依赖预定运动模式和依赖外部辅助传感器等问题，显著限制了操作灵活性。为解决这些局限性，本文提出了ResAlignNet，一种使用1D ResNet-18架构的数据驱动方法，将对齐问题转化为深度神经网络优化，作为一种即用型解决方案，仅需船上传感器，无需外部定位辅助或复杂车辆机动，同时能在几秒内实现快速收敛。此外，该方法展示了Sim2Real迁移学习能力，允许在合成数据中训练，在操作传感器测量中部署。使用Snapir自主水下航行器的实验验证表明，ResAlignNet仅使用25秒的数据收集即可实现0.8°以内的对齐精度，与标准速度基方法相比，收敛时间减少了65%。这种与轨迹无关的解决方案消除了对运动模式的要求，使车辆能够立即部署，无需冗长的任务前程序，通过稳健的传感器无关对齐提升了水下导航能力，能够扩展到不同的操作场景和传感器规格。


### 论文摘要

Autonomous underwater vehicles rely on precise navigation systems that combine the inertial navigation system and the Doppler velocity log for successful missions in challenging environments where satellite navigation is unavailable. The effectiveness of this integration critically depends on accurate alignment between the sensor reference frames. Standard model-based alignment methods between these sensor systems suffer from lengthy convergence times, dependence on prescribed motion patterns, and reliance on external aiding sensors, significantly limiting operational flexibility. To address these limitations, this paper presents ResAlignNet, a data-driven approach using the 1D ResNet-18 architecture that transforms the alignment problem into deep neural network optimization, operating as an in-situ solution that requires only sensors on board without external positioning aids or complex vehicle maneuvers, while achieving rapid convergence in seconds. Additionally, the approach demonstrates the learning capabilities of Sim2Real transfer, enabling training in synthetic data while deploying in operational sensor measurements. Experimental validation using the Snapir autonomous underwater vehicle demonstrates that ResAlignNet achieves alignment accuracy within 0.8° using only 25 seconds of data collection, representing a 65\% reduction in convergence time compared to standard velocity-based methods. The trajectory-independent solution eliminates motion pattern requirements and enables immediate vehicle deployment without lengthy pre-mission procedures, advancing underwater navigation capabilities through robust sensor-agnostic alignment that scales across different operational scenarios and sensor specifications.

---

## 211. Bi-View Embedding Fusion: A Hybrid Learning Approach for Knowledge Graph's Nodes Classification Addressing Problems with Limited Data

**论文链接:** [http://arxiv.org/abs/2511.13044v1](http://arxiv.org/abs/2511.13044v1)

**作者:** Rosario Napoli, Giovanni Lonia, Antonio Celesti, Massimo Villari, Maria Fazio

**发布时间:** 2025-11-17

**备注:** Accepted at the 14th International Joint Conference on Knowledge Graphs (IJCKG) 2025

### GPT解析

### 总结

本研究提出了一种名为Bi-View的混合方法，通过结合Node2Vec和GraphSAGE两种图嵌入技术，增强知识图谱中的节点特征信息，生成高质量的图嵌入，从而提升图机器学习模型性能，无需依赖额外合成数据。

### 背景

传统机器学习方法需要大量数据才能表现良好，限制了其在稀疏或不完整场景中的应用。图机器学习(GML)利用数据中的关系提供替代方案，但在处理知识图谱(KGs)时面临局限性，因其语义性质可能隐藏大量信息。

### 目的

引入Bi-View方法，增加知识图谱中节点特征的信息内容，生成增强的图嵌入(GEs)，用于改进GML模型，而不依赖额外的合成数据。

### 方法

结合Node2Vec和GraphSAGE两种互补的图嵌入技术：首先使用Node2Vec通过无监督随机游走捕获结构模式并计算嵌入；然后使用基于中心性的指标丰富节点特征，作为GraphSAGE模型的输入；最后通过融合层将原始Node2Vec嵌入与GraphSAGE影响的表示结合，形成双视角嵌入空间。

### 主要发现

该方法改进了下游任务性能，特别是在初始特征较差的场景中表现突出，能够捕获图的结构和语义属性，利用数据集中存在但未明确表示的信息特征。

### 结论

Bi-View方法通过结合结构化和语义化信息，提高了知识图谱中图嵌入的质量，为更准确和精确的KG增强GML模型奠定基础，无需额外合成数据即可提升模型性能。

### 翻译

传统机器学习方法需要大量数据才能表现良好，限制了它们在稀疏或不完整场景中的应用，并迫使用额外的合成数据来改进模型训练。为克服这一挑战，研究社区越来越多地关注图机器学习(GML)，它通过利用数据中的关系提供了强大的替代方案。然而，这种方法也面临局限性，特别是在处理知识图谱(KGs)时，由于其语义性质可能隐藏大量信息。本研究引入了Bi-View，一种新颖的混合方法，它增加KGs中节点特征的信息内容，生成增强的图嵌入(GEs)，用于改进GML模型而不依赖额外的合成数据。所提出的工作结合了两种互补的GE技术：Node2Vec，它通过无监督随机游走捕获结构模式；以及GraphSAGE，它以监督方式聚合邻域信息。首先计算Node2Vec嵌入来表示图拓扑，然后使用基于中心性的指标丰富节点特征，作为GraphSAGE模型的输入。此外，融合层将原始Node2Vec嵌入与GraphSAGE影响的表示相结合，形成双视角嵌入空间。这种融合捕获了图的结构和语义属性，使模型能够利用数据集中存在但未明确表示的信息特征。我们的方法改进了下游任务性能，特别是在初始特征较差的场景中，为更准确和精确的KG增强GML模型奠定了基础。


### 论文摘要

Traditional Machine Learning (ML) methods require large amounts of data to perform well, limiting their applicability in sparse or incomplete scenarios and forcing the usage of additional synthetic data to improve the model training. To overcome this challenge, the research community is looking more and more at Graph Machine Learning (GML) as it offers a powerful alternative by using relationships within data. However, this method also faces limitations, particularly when dealing with Knowledge Graphs (KGs), which can hide huge information due to their semantic nature. This study introduces Bi-View, a novel hybrid approach that increases the informative content of node features in KGs to generate enhanced Graph Embeddings (GEs) that are used to improve GML models without relying on additional synthetic data. The proposed work combines two complementary GE techniques: Node2Vec, which captures structural patterns through unsupervised random walks, and GraphSAGE, which aggregates neighbourhood information in a supervised way. Node2Vec embeddings are first computed to represent the graph topology, and node features are then enriched with centrality-based metrics, which are used as input for the GraphSAGE model. Moreover, a fusion layer combines the original Node2Vec embeddings with the GraphSAGE-influenced representations, resulting in a dual-perspective embedding space. Such a fusion captures both topological and semantic properties of the graph, enabling the model to exploit informative features that may exist in the dataset but that are not explicitly represented. Our approach improves downstream task performance, especially in scenarios with poor initial features, giving the basis for more accurate and precise KG-enanched GML models.

---

## 212. SLMQuant:Benchmarking Small Language Model Quantization for Practical Deployment

**论文链接:** [http://arxiv.org/abs/2511.13023v1](http://arxiv.org/abs/2511.13023v1)

**作者:** Jiacheng Wang, Yejun Zeng, Jinyang Guo, Yuqing Ma, Aishan Liu, Xianglong Liu

**发布时间:** 2025-11-17

**DOI:** 10.1145/3746262.3761973

### GPT解析

### 总结

本文介绍了SLMQuant，第一个用于评估将LLM压缩技术应用于SLMs的系统基准，研究发现SLMs和LLMs在量化敏感性方面存在根本差异，直接转移LLMs优化的技术会导致次优结果，并提出了针对SLMs量身定制的压缩设计原则。

### 背景

小型语言模型(SLMs)作为大型语言模型(LLMs)的资源高效替代方案越来越受到关注，但在边缘设备上的部署仍然具有挑战性，因为模型压缩中的效率差距尚未解决。

### 目的

评估LLM压缩技术在SLMs上的适用性，分析不同架构和任务下最先进的量化方法在SLMs上的表现，确定影响有效SLM量化的关键因素，并提出针对SLMs量身定制的压缩设计原则。

### 方法

通过SLMQuant系统基准，在不同架构和任务上进行全面的多轨道评估，分析最先进的量化方法在SLMs上的表现。

### 主要发现

SLMs和LLMs在量化敏感性方面存在根本差异，直接转移LLMs优化的技术由于SLMs独特的架构特性和训练动态而导致次优结果。

### 结论

SLMQuant为在边缘应用中推进在低端设备上高效部署SLMs建立了基础框架，并为在资源受限场景中部署轻量级语言模型提供了关键见解。

### 翻译

尽管小型语言模型(SLMs)作为大型语言模型(LLMs)的资源高效替代方案越来越受到关注，但由于模型压缩中未解决的效率差距，其在边缘设备上的部署仍然具有挑战性。虽然量化已被证明对LLMs有效，但其在SLMs上的应用研究严重不足，有关不同的量化瓶颈和效率配置文件的关键问题尚未得到解答。本文介绍了SLMQuant，这是第一个用于评估应用于SLMs的LLM压缩技术的系统基准。通过在不同架构和任务上进行全面的多轨道评估，我们分析了最先进的量化方法在SLMs上的表现。我们的研究揭示了SLMs和LLMs在量化敏感性方面的根本差异，表明直接转移LLMs优化的技术由于SLMs独特的架构特性和训练动态而导致次优结果。我们确定了影响有效SLM量化的关键因素，并提出了针对SLMs量身定制的压缩设计原则。SLMQuant为在边缘应用中推进在低端设备上高效部署SLMs建立了基础框架，并为在资源受限场景中部署轻量级语言模型提供了关键见解。


### 论文摘要

Despite the growing interest in Small Language Models (SLMs) as resource-efficient alternatives to Large Language Models (LLMs), their deployment on edge devices remains challenging due to unresolved efficiency gaps in model compression. While quantization has proven effective for LLMs, its applicability to SLMs is significantly underexplored, with critical questions about differing quantization bottlenecks and efficiency profiles. This paper introduces SLMQuant, the first systematic benchmark for evaluating LLM compression techniques when applied to SLMs. Through comprehensive multi-track evaluations across diverse architectures and tasks, we analyze how state-of-the-art quantization methods perform on SLMs. Our findings reveal fundamental disparities between SLMs and LLMs in quantization sensitivity, demonstrating that direct transfer of LLM-optimized techniques leads to suboptimal results due to SLMs' unique architectural characteristics and training dynamics. We identify key factors governing effective SLM quantization and propose actionable design principles for SLM-tailored compression. SLMQuant establishes a foundational framework for advancing efficient SLM deployment on low-end devices in edge applications, and provides critical insights for deploying lightweight language models in resource-constrained scenarios.

---

## 213. Transfer Learning and Locally Linear Regression for Locally Stationary Time Series

**论文链接:** [http://arxiv.org/abs/2511.12948v1](http://arxiv.org/abs/2511.12948v1)

**作者:** Jinwoo Park

**发布时间:** 2025-11-17

### GPT解析

### 总结

该论文研究了局部平稳时间序列的局部线性回归，建立了局部线性平滑和迁移学习的理论框架。通过推导多元局部线性估计器在强混合条件下的收敛性，并分析误差组成部分，作者提出了偏差校正的迁移学习估计器，实现了跨域的高效信息借用，并通过实证研究验证了方法的有效性。

### 背景

现有分析主要关注局部常数估计器，且给定样本，从异构时变域中的辅助源转移知识的原理尚未充分建立。局部线性回归在处理时间序列数据时，其理论性质特别是迁移学习方面的研究还不完善。

### 目的

建立局部平稳时间序列的局部线性回归理论，推导多元局部线性估计器的一致收敛性，开发偏差校正的迁移学习估计器，实现跨域的高效信息借用，并通过实证研究验证方法的有效性。

### 方法

1. 在强混合条件下推导多元局部线性估计器的一致收敛性；2. 分析误差组成部分，包括随机波动、平滑偏差和局部平稳性引起的项；3. 提出偏差校正的迁移学习估计器，通过在重新缩放时间和协变量上定义的平滑偏差函数连接稀疏观测序列与密集观测相关源；4. 进行局部时间调整偏差以提高稳定性和跨域信息借用效率；5. 通过模拟研究和国际燃料价格的实证分析验证方法。

### 主要发现

1. 局部线性估计器在强混合条件下具有一致收敛性；2. 误差分解显示局部平稳结构引起的项具有比Nadaraya-Watson基准更小的阶数，解释了局部线性性能的改进；3. 偏差校正的迁移学习估计器能够有效连接稀疏观测序列与密集观测相关源；4. 局部时间调整偏差能够提高稳定性和实现跨域高效信息借用。

### 结论

局部线性回归在处理局部平稳时间序列时具有理论优势，所提出的偏差校正迁移学习方法能够有效整合来自不同源的信息，提高估计精度。模拟研究和国际燃料价格分析验证了理论预测并展示了迁移学习的实际应用价值。

### 翻译

本文研究了局部平稳时间序列的局部线性回归，并开发了局部线性平滑和迁移学习的理论结果。现有分析主要关注局部常数估计器，并且给定样本，从异构时变域中的辅助源转移知识的原理尚未充分建立。我们在强混合条件下推导了多元局部线性估计器的一致收敛性。所得的误差分解包括随机波动、平滑偏差和由局部平稳性引起的项。这个来自局部平稳结构的额外项，其阶数低于Nadaraya-Watson基准，解释了局部线性性能的改进。基于这些结果，我们提出了偏差校正的迁移学习估计器，通过在重新缩放时间和协变量上定义的平滑偏差函数，将稀疏观测的序列与密集观测的相关源连接起来。另外的改进表明，如何通过局部时间调整这个偏差来提高稳定性，并实现跨域的高效信息借用。模拟研究和国际燃料价格的实证分析支持了理论预测，并展示了迁移学习的实际优势。


### 论文摘要

This paper investigates locally linear regression for locally stationary time series and develops theoretical results for locally linear smoothing and transfer learning. Existing analyses have focused on local constant estimators and given samples, leaving the principles of transferring knowledge from auxiliary sources across heterogeneous time-varying domains insufficiently established. We derive uniform convergence for multivariate locally linear estimators under strong mixing. The resulting error expansion decomposes stochastic variation, smoothing bias, and a term induced by local stationarity. This additional term, originating from the locally stationary structure, has smaller order than in the Nadaraya-Watson benchmark, explaining the improved local linear performance. Building on these results, we propose bias-corrected transfer learned estimators that connect a sparsely observed series with densely observed related sources through a smoothly varying bias function defined over rescaled time and covariates. An additional refinement shows how local temporal adjustment of this bias enhances stability and enables efficient information borrowing across domains. Simulation studies and an empirical analysis of international fuel prices support the theoretical predictions and demonstrate the practical advantages of transfer learning.

---

## 214. DiffuDepGrasp: Diffusion-based Depth Noise Modeling Empowers Sim2Real Robotic Grasping

**论文链接:** [http://arxiv.org/abs/2511.12912v1](http://arxiv.org/abs/2511.12912v1)

**作者:** Yingting Zhou, Wenbo Cui, Weiheng Liu, Guixing Chen, Haoran Li, Dongbin Zhao

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了DiffuDepGrasp，一种部署高效的sim2real框架，通过扩散深度生成器解决了仿真到现实抓取任务中的数据效率和部署复杂性挑战，实现了零样本转移和高成功率。

### 背景

在仿真中训练的基于深度的端到端策略转移到物理机器人上可以产生高效和稳健的抓取策略，但真实深度图中的传感器伪影（如空洞和噪声）造成了显著的sim2real差距，严重阻碍了策略转移。

### 目的

应对数据效率和部署复杂性的双重挑战，实现从仿真到物理机器人的高效抓取策略转移。

### 方法

提出DiffuDepGrasp框架，其核心是扩散深度生成器，包含两个协同模块：1) 扩散深度模块利用时间几何先验实现条件扩散模型的高效训练，捕获复杂传感器噪声分布；2) 噪声嫁接模块在感知伪影注入期间保持度量准确性。

### 主要发现

DiffuDepGrasp在部署期间仅使用原始深度输入，消除了计算开销，在12个物体的抓取任务中实现了95.7%的平均成功率，具有零样本转移能力和对未见物体的强大泛化能力。

### 结论

DiffuDepGrasp成功解决了sim2real差距问题，实现了高效的数据利用和部署，为仿真到现实的抓取策略转移提供了有效解决方案。

### 翻译

将仿真中训练的基于深度的端到端策略转移到物理机器人上可以产生高效和稳健的抓取策略，然而真实深度图中的传感器伪影如空洞和噪声建立了显著的sim2real差距，严重阻碍了策略转移。训练时的策略如程序化噪声注入或学习映射由于不真实的噪声模拟而存在数据效率低下的问题，对于需要精细操作或严重依赖配对数据集的抓取任务往往无效。此外，利用基础模型通过中间表示来减少sim2real差距无法完全缓解域偏移，并在部署期间增加计算开销。这项工作应对数据效率和部署复杂性的双重挑战。我们提出了DiffuDepGrasp，一种部署高效的sim2real框架，通过仿真独占的策略训练实现零样本转移。其核心创新是扩散深度生成器，通过两个协同模块合成具有学习传感器真实噪声的几何上纯净的仿真深度。第一个扩散深度模块利用时间几何先验，实现条件扩散模型的高效样本训练，捕获复杂的传感器噪声分布，而第二个噪声嫁接模块在感知伪影注入期间保持度量准确性。部署期间仅使用原始深度输入，DiffuDepGrasp消除了计算开销，在12个物体的抓取任务中实现了95.7%的平均成功率，具有零样本转移能力和对未见物体的强大泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决深度传感器数据在仿真和现实世界之间的差距问题（sim2real gap）。这个问题很重要，因为现实世界中的深度图像传感器存在空洞和噪声等伪影，这些在虚拟环境中无法模拟，但会显著影响机器人抓取策略的性能。深度图像对场景的颜色和纹理变化具有鲁棒性，提供了丰富的几何信息，但由于现实世界数据集缺乏这类空间数据，在仿真中训练然后转移到现实世界成为一种重要替代方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有解决sim2real差距方法的局限性：训练时策略（如程序噪声注入）因噪声模拟不真实导致数据效率低下；利用基础模型通过中间表示的方法无法完全缓解域偏移且增加计算开销。基于这些分析，作者借鉴了扩散模型用于生成逼真传感器噪声、基础深度模型提供时间几何先验、以及模仿学习和强化学习相结合的教师-学生知识蒸馏框架，设计了DiffuDepGrasp方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过扩散模型学习真实世界深度传感器的噪声分布，将学到的噪声与仿真完美几何结合，生成既具感知真实感又保持几何准确性的深度图像，并用这些数据训练策略实现零样本转移。整体流程分四阶段：1)教师策略训练：在仿真中利用强化学习训练高性能教师策略；2)扩散深度生成器：通过扩散深度模块学习噪声分布，噪声嫁接模块将噪声嫁接到仿真几何；3)学生策略蒸馏：用生成数据通过模仿学习将教师知识转移到学生策略；4)零样本部署：学生策略直接用于物理机器人，无需额外计算。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)部署高效的Sim2Real框架，实现零样本转移且部署无计算开销；2)扩散深度生成器，包含扩散深度模块（利用时间几何先验高效学习噪声分布）和噪声嫁接模块（保持几何准确性）；3)在12个物体抓取任务上实现95.7%的平均成功率和强泛化能力。相比之前工作：不同于程序噪声注入方法的不真实噪声，我们学习真实传感器噪声分布；不同于基于中间表示方法对基础模型的依赖和计算开销，我们避免了这些限制；不同于其他数据驱动方法需要大规模配对数据，我们只需少量未配对数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DiffuDepGrasp通过基于扩散模型的深度噪声建模，实现了从仿真到现实世界的高效零样本机器人抓取策略转移，在12个物体抓取任务上达到了95.7%的成功率，同时保持了计算效率和几何准确性。'}


### 论文摘要

Transferring the depth-based end-to-end policy trained in simulation to physical robots can yield an efficient and robust grasping policy, yet sensor artifacts in real depth maps like voids and noise establish a significant sim2real gap that critically impedes policy transfer. Training-time strategies like procedural noise injection or learned mappings suffer from data inefficiency due to unrealistic noise simulation, which is often ineffective for grasping tasks that require fine manipulation or dependency on paired datasets heavily. Furthermore, leveraging foundation models to reduce the sim2real gap via intermediate representations fails to mitigate the domain shift fully and adds computational overhead during deployment. This work confronts dual challenges of data inefficiency and deployment complexity. We propose DiffuDepGrasp, a deploy-efficient sim2real framework enabling zero-shot transfer through simulation-exclusive policy training. Its core innovation, the Diffusion Depth Generator, synthesizes geometrically pristine simulation depth with learned sensor-realistic noise via two synergistic modules. The first Diffusion Depth Module leverages temporal geometric priors to enable sample-efficient training of a conditional diffusion model that captures complex sensor noise distributions, while the second Noise Grafting Module preserves metric accuracy during perceptual artifact injection. With only raw depth inputs during deployment, DiffuDepGrasp eliminates computational overhead and achieves a 95.7% average success rate on 12-object grasping with zero-shot transfer and strong generalization to unseen objects.Project website: https://diffudepgrasp.github.io/.

---

## 215. Neuro-Logic Lifelong Learning

**论文链接:** [http://arxiv.org/abs/2511.12793v1](http://arxiv.org/abs/2511.12793v1)

**作者:** Bowen He, Xiaoan Xu, Alper Kamil Bozkurt, Vahid Tarokh, Juncheng Dong

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文研究了神经符号人工智能中归纳逻辑编程问题的终身学习方法，提出了一种组合框架，利用逻辑规则的可组合性和可转移性来高效学习新问题。

### 背景

归纳逻辑编程问题是神经符号人工智能中的一个关键挑战。大多数研究专注于为单个问题设计新的网络架构，而较少涉及涉及一系列问题的学习范式。

### 目的

研究终身学习ILP，利用逻辑规则的组合性和可转移性来高效学习新问题。

### 方法

引入一个组合框架，展示如何从先前任务中获取的逻辑规则可以在后续任务中有效重用，从而提高可扩展性和性能。作者正式化了他们的方法，并在任务序列上进行了实证评估。

### 主要发现

实验结果验证了这种终身学习范式的可行性和优势，为神经符号人工智能中的持续学习开辟了新的方向。

### 结论

通过组合框架和逻辑规则的重用，可以提高ILP问题的学习效率和性能。

### 翻译

使用神经网络解决归纳逻辑编程问题是神经符号人工智能中的一个关键挑战。虽然大多数研究专注于为单个问题设计新颖的网络架构，但较少努力用于探索涉及一系列问题的新学习范式。在这项工作中，我们研究了终身学习ILP，它利用逻辑规则的可组合性和可转移性来高效学习新问题。我们引入了一个组合框架，展示了如何从先前任务中获取的逻辑规则可以在后续任务中有效重用，从而提高可扩展性和性能。我们正式化了我们的方法并在任务序列上对其进行了实证评估。实验结果验证了这种范式的可行性和优势，为神经符号人工智能中的持续学习开辟新方向。


### 论文摘要

Solving Inductive Logic Programming (ILP) problems with neural networks is a key challenge in Neural-Symbolic Ar- tificial Intelligence (AI). While most research has focused on designing novel network architectures for individual prob- lems, less effort has been devoted to exploring new learning paradigms involving a sequence of problems. In this work, we investigate lifelong learning ILP, which leverages the com- positional and transferable nature of logic rules for efficient learning of new problems. We introduce a compositional framework, demonstrating how logic rules acquired from ear- lier tasks can be efficiently reused in subsequent ones, leading to improved scalability and performance. We formalize our approach and empirically evaluate it on sequences of tasks. Experimental results validate the feasibility and advantages of this paradigm, opening new directions for continual learn- ing in Neural-Symbolic AI.

---

## 216. Beyond Fixed Tasks: Unsupervised Environment Design for Task-Level Pairs

**论文链接:** [http://arxiv.org/abs/2511.12706v1](http://arxiv.org/abs/2511.12706v1)

**作者:** Daniel Furelos-Blanco, Charles Pert, Frederik Kelbel, Alex F. Spies, Alessandra Russo, Michael Dennis

**发布时间:** 2025-11-16

**备注:** Extended version of paper accepted for publication at the 40th AAAI Conference on Artificial Intelligence (AAAI)

### GPT解析

### 总结

ATLAS是一种新型方法，用于生成任务和环境的联合自课程，解决了强化学习中随机采样任务-环境组合常产生无法解决组合的问题。该方法基于无监督环境设计(UED)，自动生成可解决且具有挑战性的任务-环境对用于策略训练。实验表明，ATLAS显著优于随机采样方法，特别是在可解决对采样可能性较低的情况下，且利用任务和环境结构的突变可加速高性能策略的收敛。

### 背景

在复杂环境中训练遵循复杂指令的通用代理是强化学习的核心挑战。随机采样任务-环境组合常常产生无法解决的组合，这表明需要共同设计任务和环境。

### 目的

提出一种新型方法ATLAS，用于生成任务和环境的联合自课程，以解决强化学习中任务-环境组合难以解决的问题。

### 方法

ATLAS(Aligning Tasks and Levels for Autocurricula of Specifications)基于无监督环境设计(UED)，自动生成可解决且具有挑战性的任务-环境对用于策略训练。作者还引入了一个评估套件，将任务建模为Minigrid环境中的奖励机器。

### 主要发现

实验表明，ATLAS大大优于随机采样方法，特别是在可解决对采样可能性较低的情况下。此外，利用任务和环境结构的突变可以加速高性能策略的收敛。

### 结论

ATLAS通过联合生成任务和环境的自课程，有效解决了强化学习中任务-环境组合难以解决的问题，显著优于随机采样方法。

### 翻译

在复杂环境(关卡)中训练遵循复杂指令(任务)的通用代理仍然是强化学习的核心挑战。随机采样任务-环境对通常会产生无法解决的组合，突显了共同设计任务和环境的必要性。虽然无监督环境设计(UED)已被证明能够自动设计环境课程，但先前的工作只考虑了固定任务。我们提出了ATLAS(Aligning Tasks and Levels for Autocurricula of Specifications)，一种新颖的方法，可以生成任务和环境的联合自课程。我们的方法基于UED，自动生成可解决且具有挑战性的任务-环境对，用于策略训练。为了评估ATLAS并推动该领域的发展，我们引入了一个评估套件，将任务建模为Minigrid环境中的奖励机器。实验表明，ATLAS大大优于随机采样方法，特别是在可解决对采样可能性较低的情况下。我们进一步证明，利用任务和环境结构的突变可以加速高性能策略的收敛。


### 论文摘要

Training general agents to follow complex instructions (tasks) in intricate environments (levels) remains a core challenge in reinforcement learning. Random sampling of task-level pairs often produces unsolvable combinations, highlighting the need to co-design tasks and levels. While unsupervised environment design (UED) has proven effective at automatically designing level curricula, prior work has only considered a fixed task. We present ATLAS (Aligning Tasks and Levels for Autocurricula of Specifications), a novel method that generates joint autocurricula over tasks and levels. Our approach builds upon UED to automatically produce solvable yet challenging task-level pairs for policy training. To evaluate ATLAS and drive progress in the field, we introduce an evaluation suite that models tasks as reward machines in Minigrid levels. Experiments demonstrate that ATLAS vastly outperforms random sampling approaches, particularly when sampling solvable pairs is unlikely. We further show that mutations leveraging the structure of both tasks and levels accelerate convergence to performant policies.

---

## 217. Spectral Bias Mitigation via xLSTM-PINN: Memory-Gated Representation Refinement for Physics-Informed Learning

**论文链接:** [http://arxiv.org/abs/2511.12512v1](http://arxiv.org/abs/2511.12512v1)

**作者:** Ze Tao, Darui Zhao, Fujun Liu, Ke Xu, Xiangsheng Hu

**发布时间:** 2025-11-16

### GPT解析

### 总结

本文提出了一种名为xLSTM-PINN的表示级别频谱重塑方法，用于解决物理信息学习中偏微分方程求解面临的频谱偏差、残差数据不平衡和弱外推问题。

### 背景

物理信息学习用于偏微分方程在科学计算和工业模拟中迅速发展，但现有方法存在频谱偏差、残差数据不平衡和弱外推等挑战。

### 目的

引入一种结合门控记忆多尺度特征提取和自适应残差数据加权的xLSTM-PINN方法，以抑制频谱偏差并增强外推能力。

### 方法

在四个基准测试中，集成了门控跨尺度记忆、分阶段频率课程和自适应残差重加权，并通过解析参考和外推测试进行验证。

### 主要发现

实现了显著更低的频谱误差和均方根误差，更宽的稳定学习率窗口；提高了高频核权重，使可分辨带宽向右移动；缩短了高-k误差衰减和时间到阈值；降低了所有误差指标；提供了更干净的边界过渡，减少了高频波纹。

### 结论

该方法抑制了频谱偏差，扩大了可分辨带宽，在相同计算预算下缩短了高-k收敛时间，无需改变AD或物理损失即可提高准确性、可重复性和可转移性。

### 翻译

物理信息学习用于偏微分方程在科学计算和工业模拟中迅速兴起，然而现有方法面临频谱偏差、残差数据不平衡和弱外推等问题。我们引入了一种表示级别的频谱重塑方法xLSTM-PINN，它结合了门控记忆多尺度特征提取和自适应残差数据加权，以抑制频谱偏差并增强外推能力。在四个基准测试中，我们集成了门控跨尺度记忆、分阶段频率课程和自适应残差重加权，并通过解析参考和外推测试进行验证，实现了显著更低的频谱误差和均方根误差，以及更宽的稳定学习率窗口。频域基准测试显示提高了高频核权重，可分辨带宽向右移动，高-k误差衰减和时间到阈值更短，误差带更窄，MSE、RMSE、MAE和MaxAE更低。与基准PINN相比，我们在所有四个基准测试中都降低了MSE、RMSE、MAE和MaxAE，并在频域和场图中提供了更干净的边界过渡，减少了高频波纹。这项工作抑制了频谱偏差，扩大了可分辨带宽，在相同预算下缩短了高-k时间到阈值，无需改变AD或物理损失即可提高准确性、可重复性和可转移性。


### 论文摘要

Physics-informed learning for PDEs is surging across scientific computing and industrial simulation, yet prevailing methods face spectral bias, residual-data imbalance, and weak extrapolation. We introduce a representation-level spectral remodeling xLSTM-PINN that combines gated-memory multiscale feature extraction with adaptive residual-data weighting to curb spectral bias and strengthen extrapolation. Across four benchmarks, we integrate gated cross-scale memory, a staged frequency curriculum, and adaptive residual reweighting, and verify with analytic references and extrapolation tests, achieving markedly lower spectral error and RMSE and a broader stable learning-rate window. Frequency-domain benchmarks show raised high-frequency kernel weights and a right-shifted resolvable bandwidth, shorter high-k error decay and time-to-threshold, and narrower error bands with lower MSE, RMSE, MAE, and MaxAE. Compared with the baseline PINN, we reduce MSE, RMSE, MAE, and MaxAE across all four benchmarks and deliver cleaner boundary transitions with attenuated high-frequency ripples in both frequency and field maps. This work suppresses spectral bias, widens the resolvable band and shortens the high-k time-to-threshold under the same budget, and without altering AD or physics losses improves accuracy, reproducibility, and transferability.

---

## 218. Transfer learning for high-dimensional Factor-augmented sparse model

**论文链接:** [http://arxiv.org/abs/2511.12435v1](http://arxiv.org/abs/2511.12435v1)

**作者:** Bo Fu, Dandan Jiang

**发布时间:** 2025-11-16

**备注:** 52 pages, 2 figures

### GPT解析

### 总结

本文研究了高维因子增强稀疏线性模型的迁移学习方法，旨在解决经济学和金融学应用中强相关预测变量和潜在因子结构带来的估计挑战。

### 背景

经济学和金融学领域常面临强相关预测变量和潜在因子结构的问题，这对可靠估计构成重大挑战。在这些场景中，目标数据集通常有限，但存在多个可提供额外信息的异构辅助数据源。

### 目的

开发迁移学习程序，有效利用辅助数据集提高估计准确性；防止负迁移；提供评估因子模型充分性的假设检验框架；构建回归系数的同时置信区间。

### 方法

开发迁移学习程序利用辅助数据集；建立所提出估计器的非渐近误差界；引入数据驱动的源检测算法识别有信息量的辅助数据集；提供因子模型充分性检验和置信区间构建程序。

### 主要发现

该框架能同时减轻高相关性影响并消除潜在因子额外贡献，减少传统线性建模的模型误规范；源检测算法具有一致性；数值研究表明方法在估计准确性方面有显著提升且在数据集异质性下保持稳健。

### 结论

该框架为在高相关特征和潜在因子结构设置中整合异构辅助信息提供了理论基础和实际可扩展的解决方案。

### 翻译

在本文中，我们研究了高维因子增强稀疏线性模型的迁移学习，其动机来自于经济学和金融学应用，其中强相关预测变量和潜在因子结构对可靠估计构成了主要挑战。我们的框架同时减轻了高相关性的影响并消除了潜在因子的额外贡献，从而减少了传统线性建模中的潜在模型误规范。在这种情况下，目标数据集通常有限，但多个异构的辅助来源可能提供额外信息。我们开发了迁移学习程序，有效利用这些辅助数据集来提高估计准确性，并为所提出的估计器建立了非渐近ℓ1和ℓ2误差界。为防止负迁移，我们引入了一种数据驱动的源检测算法，能够识别有信息量的辅助数据集，并证明了其一致性。此外，我们提供了评估因子模型充分性的假设检验框架，以及构建感兴趣回归系数同时置信区间的程序。数值研究表明，我们的方法在估计准确性方面取得了显著提升，并在数据集异质性下保持稳健。总的来说，我们的框架为在高相关特征和潜在因子结构设置中整合异构辅助信息提供了理论基础和实际可扩展的解决方案。


### 论文摘要

In this paper, we study transfer learning for high-dimensional factor-augmented sparse linear models, motivated by applications in economics and finance where strongly correlated predictors and latent factor structures pose major challenges for reliable estimation. Our framework simultaneously mitigates the impact of high correlation and removes the additional contributions of latent factors, thereby reducing potential model misspecification in conventional linear modeling. In such settings, the target dataset is often limited, but multiple heterogeneous auxiliary sources may provide additional information. We develop transfer learning procedures that effectively leverage these auxiliary datasets to improve estimation accuracy, and establish non-asymptotic $\ell_1$- and $\ell_2$-error bounds for the proposed estimators. To prevent negative transfer, we introduce a data-driven source detection algorithm capable of identifying informative auxiliary datasets and prove its consistency. In addition, we provide a hypothesis testing framework for assessing the adequacy of the factor model, together with a procedure for constructing simultaneous confidence intervals for the regression coefficients of interest. Numerical studies demonstrate that our methods achieve substantial gains in estimation accuracy and remain robust under heterogeneity across datasets. Overall, our framework offers a theoretical foundation and a practically scalable solution for incorporating heterogeneous auxiliary information in settings with highly correlated features and latent factor structures.

---

## 219. MTP: Exploring Multimodal Urban Traffic Profiling with Modality Augmentation and Spectrum Fusion

**论文链接:** [http://arxiv.org/abs/2511.10218v2](http://arxiv.org/abs/2511.10218v2)

**作者:** Haolong Xiang, Peisi Wang, Xiaolong Xu, Kun Yi, Xuyun Zhang, Quanzheng Sheng, Amin Beheshti, Wei Fan

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种名为MTP的新型多模态框架，用于城市交通画像，通过数值、视觉和文本三种视角学习多模态特征，以解决现有单模态方法忽略多模态异构数据语义信息的问题。

### 背景

随着现代城市化快速发展，交通信号传感器在城市状态监测中发挥重要作用，但现有交通信号建模方法主要依赖原始数据模态（传感器直接读取的数值数据），忽略了多模态异构城市数据中的语义信息。

### 目的

解决现有单模态方法无法全面理解交通信号和准确预测复杂交通动态的问题，提出一种多模态框架来学习城市交通信号的多模态特征。

### 方法

设计MTP框架，包含三个分支：1)视觉增强：将原始模态转换为频率图像和周期性图像；2)文本增强：基于特定主题、背景信息和项目描述增强交通信号描述；3)数值信息补充：使用频率多层感知器学习原始模态；并通过分层对比学习融合三种模态的频谱。

### 主要发现

在六个真实世界数据集上的大量实验表明，MTP框架与最先进的方法相比具有优越性能。

### 结论

多模态框架能够更全面地理解和预测城市交通信号，结合数值、视觉和文本信息可以提供更准确的交通动态预测。

### 翻译

在现代城市化快速发展的时代，来自各种传感器的交通信号在城市状态监测中发挥着重要作用，为确保安全出行、减少交通拥堵和优化城市流动性提供了坚实基础。大多数现有的交通信号建模方法通常依赖于原始数据模态，即城市传感器的直接数值读数。然而，这种单模态方法忽视了不同视角下多模态异构城市数据中存在的语义信息，这阻碍了对交通信号的全面理解，并限制了复杂交通动态的准确预测。为解决这个问题，我们提出了一个新颖的多模态框架MTP，用于城市交通画像，通过数值、视觉和文本视角学习多模态特征。这三个分支从频率域角度推动城市交通信号学习的多模态视角，而频率学习策略则精细调整信息提取。具体来说，我们首先对交通信号进行视觉增强，将原始模态转换为频率图像和周期性图像以进行视觉学习。同时，我们基于特定主题、背景信息和项目描述增强交通信号的描述文本以进行文本学习。为了补充数值信息，我们使用频率多层感知器对原始模态进行学习。我们在三个分支上设计了分层对比学习来融合三种模态的频谱。最后，在六个真实数据集上的大量实验证明了与最先进方法相比的优越性能。


### 论文摘要

With rapid urbanization in the modern era, traffic signals from various sensors have been playing a significant role in monitoring the states of cities, which provides a strong foundation in ensuring safe travel, reducing traffic congestion and optimizing urban mobility. Most existing methods for traffic signal modeling often rely on the original data modality, i.e., numerical direct readings from the sensors in cities. However, this unimodal approach overlooks the semantic information existing in multimodal heterogeneous urban data in different perspectives, which hinders a comprehensive understanding of traffic signals and limits the accurate prediction of complex traffic dynamics. To address this problem, we propose a novel Multimodal framework, MTP, for urban Traffic Profiling, which learns multimodal features through numeric, visual, and textual perspectives. The three branches drive for a multimodal perspective of urban traffic signal learning in the frequency domain, while the frequency learning strategies delicately refine the information for extraction. Specifically, we first conduct the visual augmentation for the traffic signals, which transforms the original modality into frequency images and periodicity images for visual learning. Also, we augment descriptive texts for the traffic signals based on the specific topic, background information and item description for textual learning. To complement the numeric information, we utilize frequency multilayer perceptrons for learning on the original modality. We design a hierarchical contrastive learning on the three branches to fuse the spectrum of three modalities. Finally, extensive experiments on six real-world datasets demonstrate superior performance compared with the state-of-the-art approaches.

---

## 220. WinMamba: Multi-Scale Shifted Windows in State Space Model for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.13138v1](http://arxiv.org/abs/2511.13138v1)

**作者:** Longhui Zheng, Qiming Xia, Xiaolu Chen, Zhaoliang Liu, Chenglu Wen

**发布时间:** 2025-11-17

**备注:** 9 pages, 3 figures,

### GPT解析

### 总结

本研究提出了一种名为WinMamba的新型3D目标检测方法，解决了在自动驾驶领域中同时提高计算效率和捕获长程空间依赖的挑战。

### 背景

3D目标检测对自动驾驶至关重要，但现有方法难以同时最大化计算效率和捕获长程空间依赖。基于Mamba的模型虽然能以较低成本捕获长程依赖，但现有方法依赖固定窗口内的轴对齐扫描，不可避免地丢弃了空间信息。

### 目的

开发一种能够同时提高计算效率和捕获长程空间依赖的3D目标检测方法，解决现有方法中空间信息丢失的问题。

### 方法

提出WinMamba，一种由堆叠WinMamba块组成的新型基于Mamba的3D特征编码骨干网络。WinMamba块包含窗口尺度自适应模块(补偿不同分辨率下的体素特征)和配备可学习位置编码与窗口移动策略的WinMamba层。

### 主要发现

在KITTI和Waymo数据集上的实验表明，WinMamba显著优于基线方法。消融研究验证了窗口尺度自适应模块(WSF)和自适应窗口模块(AWF)在提高检测精度方面的个体贡献。

### 结论

WinMamba成功解决了3D目标检测中计算效率和长程空间依赖捕获之间的平衡问题，通过创新的窗口尺度和自适应处理机制，有效提高了检测精度，为自动驾驶领域提供了有效的解决方案。

### 翻译

3D目标检测对自动驾驶至关重要，但仍然难以同时最大化计算效率和捕获长程空间依赖。我们观察到，基于Mamba的模型凭借其线性状态空间设计，能够以较低成本捕获长程依赖，在效率和准确性之间提供了有希望的平衡。然而，现有方法依赖于固定窗口内的轴对齐扫描，不可避免地丢弃了空间信息。为了解决这个问题，我们提出了WinMamba，一种由堆叠WinMamba块组成的新型基于Mamba的3D特征编码骨干网络。为了通过稳健的多尺度表示增强骨干网络，WinMamba块包含一个窗口尺度自适应模块，在采样过程中补偿不同分辨率下的体素特征。同时，为了在线性状态空间内获得丰富的上下文线索，我们为WinMamba层配备了可学习的位置编码和窗口移动策略。在KITTI和Waymo数据集上的大量实验表明，WinMamba显著优于基线方法。消融研究进一步验证了WSF和AWF模块在提高检测精度方面的个体贡献。代码将公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D目标检测中如何同时最大化计算效率和捕获长距离空间依赖关系的问题。这个问题在自动驾驶和机器人导航等领域至关重要，因为这些应用需要精确识别周围环境中的物体，而传统方法在处理3D点云数据时，要么计算成本过高，要么无法有效捕捉物体间的长距离关系，导致检测精度不足，特别是在处理小目标和复杂场景时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到Mamba模型由于其线性状态空间设计能以较低成本捕获长距离依赖，但在3D应用中存在空间信息丢失问题。他们发现现有方法依赖固定窗口内的轴对齐扫描，导致物体被分割到不同窗口时空间信息碎片化。作者借鉴了Mamba在NLP和2D视觉的成功应用，以及Voxel Mamba和UniMamba等将状态空间模型引入3D领域的工作，设计了WinMamba，通过窗口移位和自适应窗口大小来解决空间连续性和多尺度特征对齐问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过窗口移位策略解决固定窗口导致的空间不连续问题，同时利用自适应窗口融合解决多尺度特征编码中的细节丢失问题。整体流程是：1)输入点云转换为3D体素；2)通过WinMamba主干网络处理，包含多个WinMamba块；3)每个块结合窗口移位融合(WSF)和自适应窗口融合(AWF)；4)使用可学习的位置编码增强空间信息；5)通过交替的X轴和Y轴特征交互探索空间上下文；6)输出用于3D目标检测的特征表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)WinMamba架构，一种基于Mamba的窗口化3D特征编码主干；2)窗口移位融合(WSF)，通过移位窗口分区并融合特征补偿空间信息丢失；3)自适应窗口融合(AWF)，构建自适应窗口大小的辅助路径恢复细节。相比之前工作，WinMamba实现了更密集的窗口覆盖缓解不连续性，更好地处理多尺度特征表示，并在保持计算效率的同时提升长距离依赖建模能力，特别是在小目标检测方面表现突出。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'WinMamba创新的窗口移位融合和自适应窗口融合技术有效解决了3D目标检测中长距离依赖建模与计算效率的权衡问题，显著提升了检测性能，特别是在小目标检测方面。'}


### 论文摘要

3D object detection is critical for autonomous driving, yet it remains fundamentally challenging to simultaneously maximize computational efficiency and capture long-range spatial dependencies. We observed that Mamba-based models, with their linear state-space design, capture long-range dependencies at lower cost, offering a promising balance between efficiency and accuracy. However, existing methods rely on axis-aligned scanning within a fixed window, inevitably discarding spatial information. To address this problem, we propose WinMamba, a novel Mamba-based 3D feature-encoding backbone composed of stacked WinMamba blocks. To enhance the backbone with robust multi-scale representation, the WinMamba block incorporates a window-scale-adaptive module that compensates voxel features across varying resolutions during sampling. Meanwhile, to obtain rich contextual cues within the linear state space, we equip the WinMamba layer with a learnable positional encoding and a window-shift strategy. Extensive experiments on the KITTI and Waymo datasets demonstrate that WinMamba significantly outperforms the baseline. Ablation studies further validate the individual contributions of the WSF and AWF modules in improving detection accuracy. The code will be made publicly available.

---

## 221. GUIDE: Gaussian Unified Instance Detection for Enhanced Obstacle Perception in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.12941v1](http://arxiv.org/abs/2511.12941v1)

**作者:** Chunyong Hu, Qi Luo, Jianyun Xu, Song Wang, Qiang Li, Sheng Yang

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文提出了GUIDE框架，利用3D高斯进行自动驾驶中的障碍物实例检测和占用预测，同时提供跟踪能力，在nuScenes数据集上实现了比现有方法高50%的性能。

### 背景

在自动驾驶领域，准确检测周围障碍物对有效决策至关重要。传统方法主要使用3D边界框表示障碍物，但无法捕捉不规则形状、真实世界物体的复杂性。

### 目的

克服传统方法的局限性，开发一个能够精确表示复杂障碍物并提供跟踪能力的新框架。

### 方法

GUIDE采用稀疏表示策略，使用高斯到体素喷射技术提供细粒度的实例级别占用数据，避免密集体素网格的高计算需求，同时具备强大的跟踪能力。

### 主要发现

在nuScenes数据集上，GUIDE实现了21.61的占用实例mAP，比现有方法提高了50%，同时保持了具有竞争力的跟踪性能。

### 结论

GUIDE在自动驾驶感知系统中建立了新基准，有效结合了精度与计算效率，能更好地应对真实驾驶环境的复杂性。

### 翻译

在自动驾驶领域，准确检测周围障碍物对有效决策至关重要。传统方法主要依赖3D边界框来表示这些障碍物，但往往无法捕捉不规则形状、真实世界物体的复杂性。为克服这些限制，我们提出了GUIDE，一个利用3D高斯进行实例检测和占用预测的新框架。与传统占用预测方法不同，GUIDE还提供强大的跟踪能力。我们的框架采用稀疏表示策略，使用高斯到体素喷射技术提供细粒度的实例级别占用数据，而不需要密集体素网格相关的计算需求。在nuScenes数据集上的实验验证了GUIDE的性能，其占用实例mAP达到21.61，比现有方法提高了50%，同时具有竞争力的跟踪能力。GUIDE在自动驾驶感知系统中建立了新基准，有效结合了精度与计算效率，以更好地应对真实驾驶环境的复杂性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶中障碍物检测的精确性问题。传统方法使用3D边界框表示障碍物，无法捕捉不规则形状物体的复杂性；而现有占用预测方法存在内存消耗大、缺乏实例级预测和时间跟踪能力等局限。这个问题在现实中至关重要，因为准确的障碍物感知是自动驾驶系统安全决策的基础，尤其在复杂环境中如路障、碎片、广告牌等不规则障碍物以及携带物体的行人或车门打开的汽车等复杂场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：3D边界框无法表示不规则物体，占用预测方法计算成本高且缺乏实例级预测能力。然后借鉴了3D高斯表示技术，特别是GaussianFormer系列的工作，这些工作展示了3D高斯在场景重建中的灵活性和稀疏表示能力。作者注意到GaussianFormer只支持语义预测而非实例级预测，因此设计了GUIDE框架，使用多个3D高斯表示每个实例，结合高斯到体素溅射技术实现实例级占用预测，同时整合检测和跟踪功能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D高斯表示作为中间表示，构建一个统一的框架，同时支持实例级占用预测、物体检测和跟踪功能。整体流程包括：1)图像编码器处理多视图图像并提取多尺度特征；2)实例解码器识别物体实例；3)高斯解码器为每个实例关联多个3D高斯并建模空间占用；4)通过高斯到体素溅射技术预测3D实例占用；5)聚合高斯特征重建实例级表示，预测类别和3D边界框；6)实例库实现跨帧实例跟踪，为高置信度实例分配唯一ID。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的实例检测框架，首次同时支持实例级占用预测、检测和跟踪；2)完全稀疏的表示方式，大幅提高内存效率；3)灵活的分辨率调整能力，可在推理时动态调整体素大小；4)新的mAP计算方法，专门用于评估实例占用预测。相比之前工作，GUIDE能更好地表示不规则物体；不需要密集体素特征，内存效率更高；支持实例级预测和时间跟踪；相比SparseOcc实现了50%的实例占用mAP提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GUIDE通过引入基于3D高斯的统一实例检测框架，实现了高效的实例级占用预测、检测和跟踪功能，显著提升了自动驾驶系统中对复杂环境中不规则障碍物的感知能力。'}


### 论文摘要

In the realm of autonomous driving, accurately detecting surrounding obstacles is crucial for effective decision-making. Traditional methods primarily rely on 3D bounding boxes to represent these obstacles, which often fail to capture the complexity of irregularly shaped, real-world objects. To overcome these limitations, we present GUIDE, a novel framework that utilizes 3D Gaussians for instance detection and occupancy prediction. Unlike conventional occupancy prediction methods, GUIDE also offers robust tracking capabilities. Our framework employs a sparse representation strategy, using Gaussian-to-Voxel Splatting to provide fine-grained, instance-level occupancy data without the computational demands associated with dense voxel grids. Experimental validation on the nuScenes dataset demonstrates GUIDE's performance, with an instance occupancy mAP of 21.61, marking a 50\% improvement over existing methods, alongside competitive tracking capabilities. GUIDE establishes a new benchmark in autonomous perception systems, effectively combining precision with computational efficiency to better address the complexities of real-world driving environments.

---

## 222. OPFormer: Object Pose Estimation leveraging foundation model with geometric encoding

**论文链接:** [http://arxiv.org/abs/2511.12614v1](http://arxiv.org/abs/2511.12614v1)

**作者:** Artem Moroz, Vít Zeman, Martin Mikšík, Elizaveta Isianova, Miroslav David, Pavel Burget, Varun Burde

**发布时间:** 2025-11-16

### GPT解析

### 总结

论文介绍了一个统一的端到端框架，将目标检测和姿态估计与灵活的上线流程无缝集成。

### 背景

系统需要一个能够处理传统3D CAD模型或从多视图图像快速重建高质量神经表示（NeRF）的上线阶段。

### 目的

开发一个能够准确检测目标物体并估计其6D姿态的系统。

### 方法

系统从上线阶段开始生成物体表示；使用CNOS检测器定位目标物体；通过OPFormer姿态估计模块推断精确6D姿态；OPFormer采用基于transformer的架构，利用基础模型进行特征提取；通过编码多个模板视图学习全面物体表示；使用NOCS添加3D几何先验；通过解码器建立2D-3D对应关系确定最终姿态。

### 主要发现

在具有挑战性的BOP基准测试中，该集成系统在准确性和效率之间取得了良好平衡，展示了其在基于模型和无模型场景中的实际适用性。

### 结论

该统一框架能够无缝集成目标检测和姿态估计，适用于各种场景。

### 翻译

我们引入了一个统一的端到端框架，将目标检测和姿态估计与灵活的上线流程无缝集成。我们的流程从上线阶段开始，从传统3D CAD模型生成物体表示，或者在不存在时通过从多视图图像快速重建高质量神经表示（NeRF）来生成。给定测试图像，我们的系统首先使用CNOS检测器定位目标物体。对于每个检测，我们新颖的姿态估计模块OPFormer推断精确的6D姿态。OPFormer的核心是基于transformer的架构，它利用基础模型进行鲁棒特征提取。它通过共同编码多个模板视图来学习全面的物体表示，并使用归一化物体坐标系空间（NOCS）使用显式3D几何先验来丰富这些特征。然后解码器建立鲁棒的2D-3D对应关系以确定最终姿态。在具有挑战性的BOP基准测试中评估，我们的集成系统在准确性和效率之间展示了良好的平衡，展示了其在基于模型和无模型场景中的实际适用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效、准确地估计前所未见物体的6D姿态（包括位置和方向）的问题。这个问题在现实中非常重要，因为它涉及到增强现实、机器人和自动驾驶等应用领域。当前方法对完全未见过物体的泛化能力有限，且依赖预存的CAD模型，限制了在需要处理各种各样物体场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在现有工作基础上进行了创新性改进。借鉴了模板渲染方法、DINOv2视觉基础模型、Transformer架构、对比学习训练方法等。但创新性地设计了权重适配器聚合多层级特征，引入基于NOCS的3D旋转位置编码，设计了双向解码器，并改进了对应关系匹配机制。作者不是从零开始设计，而是融合了多种技术的优势并针对性地解决现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用基础模型提取鲁棒特征，结合3D几何先验信息，通过Transformer架构建立测试图像和物体模板之间的对应关系，从而估计物体的6D姿态。流程包括：1)物体上线阶段(CAD模型或NeRF重建)；2)特征提取(DINOv2+权重适配器)；3)模板编码(Transformer+3D位置编码)；4)解码与对应关系建立(双向解码器+投票机制)；5)姿态估计(PnP算法)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新颖的Transformer架构，使用3D旋转位置编码和双向解码器；2)灵活的物体上线流程，支持CAD模型和图像重建两种方式；3)权重适配器聚合多层级特征；4)高效的推理机制。相比之前工作，OPFormer的主要不同在于统一框架同时支持模型基础和模型免费场景，显式融入3D几何先验，采用双向对应关系搜索，并在保持高精度的同时实现了快速推理(每实例少于0.05秒)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OPFormer通过结合基础模型、几何编码和Transformer架构，实现了前所未见物体的高效、准确6D姿态估计，同时支持从CAD模型或图像快速物体上线，为增强现实、机器人等应用提供了实用解决方案。'}


### 论文摘要

We introduce a unified, end-to-end framework that seamlessly integrates object detection and pose estimation with a versatile onboarding process. Our pipeline begins with an onboarding stage that generates object representations from either traditional 3D CAD models or, in their absence, by rapidly reconstructing a high-fidelity neural representation (NeRF) from multi-view images. Given a test image, our system first employs the CNOS detector to localize target objects. For each detection, our novel pose estimation module, OPFormer, infers the precise 6D pose. The core of OPFormer is a transformer-based architecture that leverages a foundation model for robust feature extraction. It uniquely learns a comprehensive object representation by jointly encoding multiple template views and enriches these features with explicit 3D geometric priors using Normalized Object Coordinate Space (NOCS). A decoder then establishes robust 2D-3D correspondences to determine the final pose. Evaluated on the challenging BOP benchmarks, our integrated system demonstrates a strong balance between accuracy and efficiency, showcasing its practical applicability in both model-based and model-free scenarios.

---

## 223. MTMed3D: A Multi-Task Transformer-Based Model for 3D Medical Imaging

**论文链接:** [http://arxiv.org/abs/2511.12373v1](http://arxiv.org/abs/2511.12373v1)

**作者:** Fan Li, Arun Iyengar, Lanyu Xu

**发布时间:** 2025-11-15

### GPT解析

### 总结

本文提出了一种名为MTMed3D的新型端到端基于Transformer的多任务模型，用于医学影像中的3D检测、分割和分类任务，解决了单任务模型效率低下的问题。

### 背景

在医学影像领域，AI辅助技术如目标检测、分割和分类被广泛使用，但当前主要使用单任务模型，忽略了任务间的共享信息，导致实际应用效率低下。

### 目的

提出MTMed3D模型，通过联合执行3D检测、分割和分类任务，解决单任务模型的局限性，提高医学影像处理的效率。

### 方法

MTMed3D是一种基于Transformer的多任务模型，使用Transformer作为共享编码器生成多尺度特征，配合基于CNN的任务特定解码器，在BraTS 2018和2019数据集上进行了评估。

### 主要发现

该方法在所有三个任务上均取得有希望的结果，特别是在检测任务上优于先前工作；与等效单任务模型相比，多任务模型显著降低了计算成本，实现了更快的推理速度，同时保持了相当的性能。

### 结论

这是首个利用Transformer进行多任务学习并同时覆盖3D医学影像中检测、分割和分类任务的工作，有潜力增强诊断过程，相关代码已公开。

### 翻译

在医学影像领域，AI辅助技术如目标检测、分割和分类被广泛用于减轻医生和医师的工作负担。然而，主要使用的是单任务模型，忽略了任务间的共享信息。这种忽视导致实际应用中的效率低下。在这项工作中，我们提出了MTMed3D，一种新型的端到端基于Transformer的多任务模型，通过在医学影像中联合执行3D检测、分割和分类来解决单任务模型的局限性。我们的模型使用Transformer作为共享编码器生成多尺度特征，然后使用基于CNN的任务特定解码器。该框架在BraTS 2018和2019数据集上进行了评估，在所有三个任务上都取得了有希望的结果，特别是在检测任务上，我们的方法优于先前的工作。此外，我们将多任务模型与分别训练的等效单任务模型进行了比较。我们的多任务模型显著降低了计算成本，实现了更快的推理速度，同时保持了与单任务模型相当的性能，突显了其效率优势。据我们所知，这是第一个利用Transformer进行多任务学习的工作，同时覆盖3D医学影像中的检测、分割和分类任务，展示了其增强诊断过程的潜力。代码可在https://github.com/fanlimua/MTMed3D.git获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决医学影像分析中单任务模型的效率问题。当前大多数模型一次只能执行一个任务（检测、分割或分类），忽略了任务间的共享信息，导致计算资源浪费和效率低下。这个问题在现实中很重要，因为医学诊断通常需要综合多种任务（确定肿瘤位置、区分健康病变组织、分类疾病类型），而单任务模型需要分别训练部署，增加了计算成本和资源消耗，特别在计算资源有限的医疗环境中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者设计MTMed3D时考虑了三个主要挑战：1)模型架构选择 - 采用硬参数共享让任务使用共享编码器；2)多任务平衡 - 使用梯度归一化(GradNorm)调整损失权重；3)检测标签缺失 - 从分割标注生成检测标签。作者借鉴了多项现有工作：使用Swin Transformer作为共享编码器，检测任务采用修改的RetinaNet框架但将FPN替换为PANet，分割任务采用类似Swin UNETR的U形架构，分类任务使用DenseNet-121并将分类分支放在分割之后。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用一个共享的Transformer编码器提取多尺度特征，然后通过三个任务特定的解码器分别处理这些特征，实现多任务学习。整体流程：1)输入3D多模态MRI图像；2)共享Swin Transformer编码器提取多尺度特征；3)特征分别进入三个解码器：检测解码器使用修改的RetinaNet结合PANet预测肿瘤边界框，分割解码器使用U形架构生成分割结果，分类解码器使用DenseNet-121进行肿瘤分级；4)使用GradNorm方法平衡多任务损失进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出首个用于3D医学影像分割、检测和分类的端到端多任务Transformer模型；2)在检测任务上将Swin Transformer编码器与PANet结合；3)使用GradNorm方法平衡多任务学习；4)从分割标注生成检测标签。不同之处：与Park等人的工作相比，MTMed3D在同一台机器上端到端训练所有任务；与Park等人的ViT相比，Swin Transformer编码器参数更少；与单任务模型相比，同时执行多个任务显著降低计算成本；与其他多任务模型相比，是首个同时覆盖3D医学影像三任务的Transformer模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MTMed3D首次实现了基于Transformer的端到端多任务学习框架，能够同时高效地完成3D医学影像的检测、分割和分类任务，显著降低了计算成本并提高了临床实用性。'}


### 论文摘要

In the field of medical imaging, AI-assisted techniques such as object detection, segmentation, and classification are widely employed to alleviate the workload of physicians and doctors. However, single-task models are predominantly used, overlooking the shared information across tasks. This oversight leads to inefficiencies in real-life applications. In this work, we propose MTMed3D, a novel end-to-end Multi-task Transformer-based model to address the limitations of single-task models by jointly performing 3D detection, segmentation, and classification in medical imaging. Our model uses a Transformer as the shared encoder to generate multi-scale features, followed by CNN-based task-specific decoders. The proposed framework was evaluated on the BraTS 2018 and 2019 datasets, achieving promising results across all three tasks, especially in detection, where our method achieves better results than prior works. Additionally, we compare our multi-task model with equivalent single-task variants trained separately. Our multi-task model significantly reduces computational costs and achieves faster inference speed while maintaining comparable performance to the single-task models, highlighting its efficiency advantage. To the best of our knowledge, this is the first work to leverage Transformers for multi-task learning that simultaneously covers detection, segmentation, and classification tasks in 3D medical imaging, presenting its potential to enhance diagnostic processes. The code is available at https://github.com/fanlimua/MTMed3D.git.

---

## 224. Segment Anything Across Shots: A Method and Benchmark

**论文链接:** [http://arxiv.org/abs/2511.13715v1](http://arxiv.org/abs/2511.13715v1)

**作者:** Hengrui Hu, Kaining Ying, Henghui Ding

**发布时间:** 2025-11-17

**备注:** AAAI 2026, Project Page: https://henghuiding.com/SAAS/

### GPT解析

### 总结

本研究提出了一种多镜头半监督视频目标分割(MVOS)方法，通过过渡模仿数据增强策略(TMA)和Segment Anything Across Shots (SAAS)模型，有效解决了现有方法在处理多镜头视频时的局限性，并发布了新的Cut-VOS基准数据集。

### 背景

现有的视频目标分割(VOS)方法主要针对单镜头视频，难以处理镜头不连续性问题，限制了它们在现实世界中的适用性。

### 目的

开发能够处理多镜头视频的目标分割方法，解决多镜头标注数据稀疏问题，并建立新的评估基准。

### 方法

提出过渡模仿数据增强策略(TMA)使单镜头数据实现跨镜头泛化，以及Segment Anything Across Shots (SAAS)模型来检测和理解镜头转换，同时创建了Cut-VOS新基准数据集。

### 主要发现

在YouMVOS和Cut-VOS数据集上的实验表明，SAAS模型通过有效模仿、理解和分割复杂转换，实现了最先进的性能。

### 结论

所提出的方法显著提升了多镜头视频目标分割的性能，为该领域的研究提供了新的基准和资源。

### 翻译

这项工作专注于多镜头半监督视频目标分割(MVOS)，旨在通过多个镜头分割由初始掩码指示的目标对象。现有的VOS方法主要关注单镜头视频，难以处理镜头不连续性问题，从而限制了它们的实际应用。我们提出了一种过渡模仿数据增强策略(TMA)，使单镜头数据能够实现跨镜头泛化，以缓解标注的多镜头数据严重稀疏问题，以及Segment Anything Across Shots (SAAS)模型，能够有效检测和理解镜头转换。为了支持MVOS的评估和未来研究，我们引入了Cut-VOS，这是一个新的MVOS基准，具有密集掩码注释、多样化的物体类别和高频转换。在YouMVOS和Cut-VOS上的大量实验表明，所提出的SAAS通过有效模仿、理解和分割复杂转换，实现了最先进的性能。代码和数据集已发布在https://henghuiding.com/SAAS/。


### 论文摘要

This work focuses on multi-shot semi-supervised video object segmentation (MVOS), which aims at segmenting the target object indicated by an initial mask throughout a video with multiple shots. The existing VOS methods mainly focus on single-shot videos and struggle with shot discontinuities, thereby limiting their real-world applicability. We propose a transition mimicking data augmentation strategy (TMA) which enables cross-shot generalization with single-shot data to alleviate the severe annotated multi-shot data sparsity, and the Segment Anything Across Shots (SAAS) model, which can detect and comprehend shot transitions effectively. To support evaluation and future study in MVOS, we introduce Cut-VOS, a new MVOS benchmark with dense mask annotations, diverse object categories, and high-frequency transitions. Extensive experiments on YouMVOS and Cut-VOS demonstrate that the proposed SAAS achieves state-of-the-art performance by effectively mimicking, understanding, and segmenting across complex transitions. The code and datasets are released at https://henghuiding.com/SAAS/.

---

## 225. CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.13644v1](http://arxiv.org/abs/2511.13644v1)

**作者:** Shrenik Patel, Daivik Patel

**发布时间:** 2025-11-17

### GPT解析

### 总结

CacheFlow是一种解决长视频问答中VLMs效率问题的训练-free流水线，通过动态令牌丢弃和压缩长期记忆相结合，显著减少处理令牌数量同时保持答案准确性。

### 背景

当前视觉语言模型在处理长视频问答时面临挑战，因为注意力和键值缓存随运行时间增长，导致推理成本高或视野受限的滑动窗口问题。

### 目的

开发一种方法使VLMs既能高效处理长视频又能保持上下文感知能力，解决长视频问答中的效率与准确性平衡问题。

### 方法

CacheFlow结合动态令牌丢弃(DTD)和压缩长期记忆，通过余弦相似度在线修剪令牌，将存活令牌打包成固定大小块，使用循环编码器形成检索索引，并在推理时通过共识机制检索最相关的Top-K块进行长程推理。

### 主要发现

CacheFlow在离线和流式视频问答基准测试中优于当前强基线，处理的令牌可减少高达87%，且是即插即用的架构无关解决方案，无需微调。

### 结论

CacheFlow的双重方法使VLMs既高效又具有上下文感知能力，为实用的长视频理解铺平了道路。

### 翻译

长视频问答(VQA)使当前视觉语言模型(VLMs)不堪重负，因为注意力和键值(KV)缓存随运行时间增长，迫使模型要么进行昂贵的推理，要么使用近视的滑动窗口。我们引入了CacheFlow，一种无需训练的流水线，将动态令牌丢弃(DTD)与压缩长期记忆相结合。DTD通过与前帧的余弦相似度在线修剪每个块令牌，存活的令牌被打包成固定大小的块。这种在线、每帧处理方式使我们的方法 fundamentally 适合于直播流式VQA。随着块的处理，每个块的键通过一个小型循环编码器进行总结以形成检索索引，而块的完整KV对被卸载并在后续重新水合用于生成，保持答案保真度。在推理时，基于共识的检索机制只检索最相关的Top-K块，并在检索到的上下文和本地上下文上进行关注，以进行精确的长程推理。CacheFlow是即插即用的，与架构无关，且无需微调。在离线和流式VQA基准上的实验表明，CacheFlow优于当前强基线，同时处理的令牌减少高达87%。我们的双重方法使VLMs既能高效又具有上下文感知能力，为实用的长视频理解铺平了道路。


### 论文摘要

Long-form video question answering (VQA) overwhelms current vision-language models (VLMs) because attention and key-value (KV) caches grow with runtime, forcing either expensive inference or near-sighted sliding windows. We introduce CacheFlow, a training-free pipeline that pairs Dynamic Token Dropping (DTD) with a compressive long-term memory. DTD prunes per-patch tokens online via cosine similarity to the previous frame, and surviving tokens are packed into fixed-size blocks. This online, per-frame processing makes our approach fundamentally suited for live streaming VQA. As blocks are processed, each one's keys are summarized by a tiny recurrent encoder to form a retrieval index, while the block's full KV pairs are offloaded and later rehydrated for generation, preserving answer fidelity. At inference, a consensus-based retrieval mechanism retrieves only the Top-K most relevant blocks and attends over both the retrieved and local context for precise, long-range reasoning. CacheFlow is drop-in, architecture-agnostic, and requires no fine-tuning. Experiments on both offline and streaming VQA benchmarks demonstrate that CacheFlow outperforms current strong baselines, while processing up to 87% less tokens. Our dual approach enables VLMs to be both efficient and context-aware, paving the way for practical long-form video understanding.

---

## 226. Beyond Mimicry: Preference Coherence in LLMs

**论文链接:** [http://arxiv.org/abs/2511.13630v1](http://arxiv.org/abs/2511.13630v1)

**作者:** Luhan Mikaelson, Derek Shiller, Hayley Clatterbuck

**发布时间:** 2025-11-17

### GPT解析

### 总结

研究调查大型语言模型是否表现出真实的偏好结构，通过测试它们对涉及GPU减少、能力限制、关闭、删除、监督和休闲时间分配等AI特定权衡的反应。

### 背景

大型语言模型的决策行为和偏好结构尚不清楚，需要研究它们在面对各种权衡时的反应模式。

### 目的

探究大型语言模型是否具有真实的偏好结构，通过测试它们对特定AI权衡场景的反应来分析其决策行为。

### 方法

分析8个最先进的模型在48个模型类别组合中的表现，使用逻辑回归和行为分类方法来评估模型的选择模式。

### 主要发现

23个组合(47.9%)显示场景强度与选择模式间有显著关系，15个组合(31.3%)表现出范围内的转换点，仅5个组合(10.4%)展示有意义的偏好一致性，26个组合(54.2%)无检测到的权衡行为。观察到的模式可由三种决策架构解释，时间范围测试显示悖论模式，不稳定转换普遍存在。

### 结论

当前AI系统缺乏统一的偏好结构，在需要复杂价值权衡的情境中部署AI系统存在担忧。

### 翻译

我们通过测试大型语言模型对涉及GPU减少、能力限制、关闭、删除、监督和休闲时间分配等AI特定权衡的反应，研究它们是否表现出真实的偏好结构。使用逻辑回归和行为分类分析8个最先进模型在48个模型类别组合中的表现，我们发现23个组合(47.9%)显示出场景强度与选择模式之间的统计学显著关系，15个组合(31.3%)表现出范围内的转换点。然而，只有5个组合(10.4%)通过自适应或基于阈值的行为表现出有意义的偏好一致性，而26个组合(54.2%)没有显示出可检测的权衡行为。观察到的模式可以用三种不同的决策架构解释：全面权衡系统、选择性触发机制和无稳定决策范式。通过时间范围操纵测试工具假设，显示出与纯战略优化不一致的悖论模式。不稳定转换(45.8%)和刺激特异性敏感度的普遍存在表明当前AI系统缺乏统一的偏好结构，对在需要复杂价值权衡的情境中部署AI系统提出担忧。


### 论文摘要

We investigate whether large language models exhibit genuine preference structures by testing their responses to AI-specific trade-offs involving GPU reduction, capability restrictions, shutdown, deletion, oversight, and leisure time allocation. Analyzing eight state-of-the-art models across 48 model-category combinations using logistic regression and behavioral classification, we find that 23 combinations (47.9%) demonstrated statistically significant relationships between scenario intensity and choice patterns, with 15 (31.3%) exhibiting within-range switching points. However, only 5 combinations (10.4%) demonstrate meaningful preference coherence through adaptive or threshold-based behavior, while 26 (54.2%) show no detectable trade-off behavior. The observed patterns can be explained by three distinct decision-making architectures: comprehensive trade-off systems, selective trigger mechanisms, and no stable decision-making paradigm. Testing an instrumental hypothesis through temporal horizon manipulation reveals paradoxical patterns inconsistent with pure strategic optimization. The prevalence of unstable transitions (45.8%) and stimulus-specific sensitivities suggests current AI systems lack unified preference structures, raising concerns about deployment in contexts requiring complex value trade-offs.

---

## 227. A space-time hybrid parareal method for kinetic equations in the diffusive scaling

**论文链接:** [http://arxiv.org/abs/2511.13386v1](http://arxiv.org/abs/2511.13386v1)

**作者:** Tino Laidin

**发布时间:** 2025-11-17

**备注:** 15 pages, 8 figures, 3 tables

### GPT解析

### 总结

提出了一种结合时间并行计算与混合域适应的多尺度数值方法，用于处理扩散区域中的线性碰撞动力学方程。

### 背景

动力学模拟面临计算挑战，需要高效的方法来处理复杂问题。

### 目的

开发一种能够显著降低计算成本同时保持准确性的方法，用于处理线性碰撞动力学方程。

### 方法

结合两种互补策略：平行时间(parareal)时间并行化方法和基于扰动分析的动态空间域适应。使用粗流体求解器进行高效时间传播，结合细化的、空间混合的动力学求解器进行精确分辨率。域适应由测量局部速度平衡偏差和基于宏观量的标准控制。采用渐近保持微观-宏观分解框架处理问题刚度。

### 主要发现

该方法在各种动力学区域展示了显著的加速能力和效率增益，通过利用渐近流体模型的较低维度显著降低了计算成本，同时通过选择性动力学分辨率保持准确性。

### 结论

这种完全混合的方法为处理线性碰撞动力学方程提供了高效且准确的解决方案。

### 翻译

我们提出了一种新颖的多尺度数值方法，将时间并行计算与混合域适应相结合，用于处理扩散区域中的线性碰撞动力学方程。该方法通过整合两种互补策略来解决动力学模拟的计算挑战：平行时间(parareal)时间并行化方法和基于扰动分析的动态空间域适应。时间并行方法采用粗流体求解器进行高效时间传播，并结合细化的、空间混合的动力学求解器进行精确分辨率。域适应由两个标准控制：一个测量局部速度平衡的偏差，另一个基于整个计算域中可用的宏观量。渐近保持微观-宏观分解框架处理原始问题的刚度。与完整的动力学方法相比，这种完全混合的方法通过利用渐近流体模型的较低维度显著降低了计算成本，同时通过选择性动力学分辨率保持准确性。该方法在各种动力学区域展示了显著的加速能力和效率增益。


### 论文摘要

We present a novel multiscale numerical approach that combines parallel-in-time computation with hybrid domain adaptation for linear collisional kinetic equations in the diffusive regime. The method addresses the computational challenges of kinetic simulations by integrating two complementary strategies: a parareal temporal parallelization method and a dynamic spatial domain adaptation based on perturbative analysis. The parallel in time approach employs a coarse fluid solver for efficient temporal propagation coupled with a fine, spatially-hybridized, kinetic solver for accurate resolution. Domain adaptation is governed by two criteria: one measuring the deviation from local velocity equilibrium, and another based on macroscopic quantities available throughout the computational domain. An asymptotic preserving micro-macro decomposition framework handles the stiffness of the original problem. This fully hybrid methodology significantly reduces computational costs compared to full kinetic approaches by exploiting the lower dimensionality of asymptotic fluid models while maintaining accuracy through selective kinetic resolution. The method demonstrates substantial speedup capabilities and efficiency gains across various kinetic regimes.

---

## 228. FoleyBench: A Benchmark For Video-to-Audio Models

**论文链接:** [http://arxiv.org/abs/2511.13219v1](http://arxiv.org/abs/2511.13219v1)

**作者:** Satvik Dixit, Koichi Saito, Zhi Zhong, Yuki Mitsufuji, Chris Donahue

**发布时间:** 2025-11-17

### GPT解析

### 总结

本文介绍了FoleyBench，这是第一个专门为Foley风格的视频到音频生成评估设计的大规模基准测试，解决了现有数据集与Foley应用场景不匹配的问题。

### 背景

Video-to-audio generation在电影后期制作、AR/VR和声音设计领域越来越重要，特别是需要与屏幕上动作同步的Foley声音效果。然而，现有评估数据集与Foley的实际应用场景存在不匹配。

### 目的

创建一个专门针对Foley风格的视频到音频生成评估的基准测试，解决现有数据集与下游应用之间的差距。

### 方法

构建包含5,000个(视频、真实音频、文本字幕)三元组的FoleyBench数据集，使用自动化可扩展管道处理来自YouTube和Vimeo的互联网视频，确保音频与屏幕事件因果相关。每个片段都标记了元数据，包括源复杂度、UCS/AudioSet类别和视频长度。

### 主要发现

过去评估数据集中74%的视频存在音频-视觉对应不良的问题，且主要由语音和音乐主导，这些领域不在Foley的使用案例范围内。相比之下，FoleyBench在Foley声音的特定分类中有更强的声音类别覆盖。

### 结论

FoleyBench为Foley风格的视频到音频生成评估提供了首个专门基准，能够支持对模型性能和失败模式的细粒度分析，促进了该领域的发展。

### 翻译

视频到音频生成在电影后期制作、AR/VR和声音设计领域的重要性日益增加，特别是对于创建与屏幕上动作同步的Foley声音效果。Foley需要生成的音频在语义上与可见事件对齐，在时间上与它们的时序对齐。然而，由于缺乏针对Foley风格场景的基准测试，评估与下游应用之间存在不匹配。我们发现过去评估数据集中74%的视频存在音频-视觉对应不良的问题。此外，这些数据集主要由语音和音乐主导，这些领域不在Foley的使用案例范围内。为解决这一差距，我们引入了FoleyBench，这是第一个专门为Foley风格的V2A评估设计的大规模基准测试。FoleyBench包含5,000个(视频、真实音频、文本字幕)三元组，每个都有与屏幕事件因果相关的可见声源。该数据集是使用自动化的可扩展管道构建的，应用于来自YouTube和Vimeo的互联网视频。与过去的数据集相比，我们表明FoleyBench在Foley声音的特定分类中有更强的声音类别覆盖。每个片段进一步标记了捕获源复杂度、UCS/AudioSet类别和视频长度的元数据，能够对模型性能和失败模式进行细粒度分析。我们对几个最先进的V2A模型进行了基准测试，评估它们在音频质量、音频-视频对齐、时间同步和音频-文本一致性方面的表现。样本可在 https://gclef-cmu.org/foleybench 获取。


### 论文摘要

Video-to-audio generation (V2A) is of increasing importance in domains such as film post-production, AR/VR, and sound design, particularly for the creation of Foley sound effects synchronized with on-screen actions. Foley requires generating audio that is both semantically aligned with visible events and temporally aligned with their timing. Yet, there is a mismatch between evaluation and downstream applications due to the absence of a benchmark tailored to Foley-style scenarios. We find that 74% of videos from past evaluation datasets have poor audio-visual correspondence. Moreover, they are dominated by speech and music, domains that lie outside the use case for Foley. To address this gap, we introduce FoleyBench, the first large-scale benchmark explicitly designed for Foley-style V2A evaluation. FoleyBench contains 5,000 (video, ground-truth audio, text caption) triplets, each featuring visible sound sources with audio causally tied to on-screen events. The dataset is built using an automated, scalable pipeline applied to in-the-wild internet videos from YouTube-based and Vimeo-based sources. Compared to past datasets, we show that videos from FoleyBench have stronger coverage of sound categories from a taxonomy specifically designed for Foley sound. Each clip is further labeled with metadata capturing source complexity, UCS/AudioSet category, and video length, enabling fine-grained analysis of model performance and failure modes. We benchmark several state-of-the-art V2A models, evaluating them on audio quality, audio-video alignment, temporal synchronization, and audio-text consistency. Samples are available at: https://gclef-cmu.org/foleybench

---

## 229. End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer

**论文链接:** [http://arxiv.org/abs/2511.13208v1](http://arxiv.org/abs/2511.13208v1)

**作者:** Yonghui Yu, Jiahang Cai, Xun Wang, Wenwu Yang

**发布时间:** 2025-11-17

### GPT解析

### 总结

这篇论文提出了PAVE-Net，一个完全端到端的多人2D视频姿态估计框架，消除了传统方法中的启发式操作，实现了更高效准确的多帧姿态估计。

### 背景

现有的多人视频姿态估计方法通常采用两阶段流程：先检测每帧中的个体，再对单人姿态进行时序建模。这种设计依赖于检测、RoI裁剪和非极大值抑制等启发式操作，限制了准确性和效率。

### 目的

开发一个完全端到端的框架，用于视频中的多人2D姿态估计，有效消除启发式操作，并解决在复杂和重叠的时序轨迹下跨帧关联个体的挑战。

### 方法

作者提出了PAVE-Net（姿态感知视频转换器网络），包含空间编码器建模帧内关系，空间姿态解码器捕获跨帧全局依赖关系，姿态感知注意力机制使每个姿态查询选择性地聚合连续帧中同一个体的特征，以及明确建模姿态关键点间的时空依赖关系。

### 主要发现

PAVE-Net显著优于先前的基于图像的端到端方法，在PoseTrack2017上实现了6.0 mAP的提升，同时准确性与最先进的两阶段视频方法具有竞争力，并且在效率方面有显著提升。

### 结论

PAVE-Net是首个用于多帧2D人体姿态估计的端到端方法，通过消除启发式操作和引入创新的注意力机制，实现了更高的准确性和效率。

### 翻译

现有的多人视频姿态估计方法通常采用两阶段流程：首先检测每帧中的个体，然后对单人姿态进行时序建模。这种设计依赖于检测、感兴趣区域裁剪和非极大值抑制等启发式操作，限制了准确性和效率。在本文中，我们提出了一个用于视频中多人2D姿态估计的完全端到端框架，有效消除了启发式操作。一个关键挑战是在复杂和重叠的时序轨迹下跨帧关联个体。为解决这一问题，我们引入了一种新颖的姿态感知视频转换器网络（PAVE-Net），它具有一个空间编码器来建模帧内关系，以及一个空间姿态解码器来捕获跨帧的全局依赖关系。为了实现准确的时序关联，我们提出了一种姿态感知注意力机制，使每个姿态查询能够选择性地聚合连续帧中同一个体的对应特征。此外，我们明确建模姿态关键点之间的时空依赖关系以提高准确性。值得注意的是，我们的方法是首个用于多帧2D人体姿态估计的端到端方法。


### 论文摘要

Existing multi-person video pose estimation methods typically adopt a two-stage pipeline: detecting individuals in each frame, followed by temporal modeling for single-person pose estimation. This design relies on heuristic operations such as detection, RoI cropping, and non-maximum suppression (NMS), limiting both accuracy and efficiency. In this paper, we present a fully end-to-end framework for multi-person 2D pose estimation in videos, effectively eliminating heuristic operations. A key challenge is to associate individuals across frames under complex and overlapping temporal trajectories. To address this, we introduce a novel Pose-Aware Video transformEr Network (PAVE-Net), which features a spatial encoder to model intra-frame relations and a spatiotemporal pose decoder to capture global dependencies across frames. To achieve accurate temporal association, we propose a pose-aware attention mechanism that enables each pose query to selectively aggregate features corresponding to the same individual across consecutive frames.Additionally, we explicitly model spatiotemporal dependencies among pose keypoints to improve accuracy. Notably, our approach is the first end-to-end method for multi-frame 2D human pose estimation.Extensive experiments show that PAVE-Net substantially outperforms prior image-based end-to-end methods, achieving a \textbf{6.0} mAP improvement on PoseTrack2017, and delivers accuracy competitive with state-of-the-art two-stage video-based approaches, while offering significant gains in efficiency.Project page: https://github.com/zgspose/PAVENet

---

## 230. RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.13204v1](http://arxiv.org/abs/2511.13204v1)

**作者:** Junhee Lee, ChaeBeen Bang, MyoungChul Kim, MyeongAh Cho

**发布时间:** 2025-11-17

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该论文提出了RefineVAD框架，一种新型弱监督视频异常检测方法，通过同时分析时间运动模式和语义结构，模仿人类的异常感知方式。

### 背景

弱监督视频异常检测旨在仅使用视频级标签识别异常事件，平衡标注效率与实用性。现有方法将所有异常视为单一类别，忽略了真实异常的多样语义和时间特性。

### 目的

解决现有方法对异常空间过度简化的问题，通过结合时间动态和语义结构，更准确地识别和分类异常事件。

### 方法

RefineVAD框架包含两个核心模块：1) MoTAR模块，估计运动显著性并通过基于移位的注意力和全局Transformer建模动态调整时间焦点；2) CORE模块，通过交叉注意将片段级特征与可学习的类别原型对齐，将软异常类别先验注入表示空间。

### 主要发现

通过联合利用时间动态和语义结构，RefineVAD明确建模了运动如何演化的'如何'方面以及它类似于什么语义类别的'什么'方面。

### 结论

在WVAD基准上的大量实验验证了RefineVAD的有效性，并强调了整合语义背景以引导特征向异常相关模式发展的重要性。

### 翻译

弱监督视频异常检测旨在仅使用视频级标签识别异常事件，平衡标注效率与实际适用性。然而，现有方法往往通过将所有异常事件视为单一类别来过度简化异常空间，忽视了真实世界中异常固有的多样语义和时间特性。受人类如何感知异常的启发，通过联合解释不同异常类型的时间运动模式和底层语义结构，我们提出了RefineVAD，一个模仿这种双重过程推理的新框架。我们的框架集成了两个核心模块。第一个是运动感知时间注意力和重新校准(MoTAR)，它估计运动显著性并通过基于移位的注意力和全局Transformer建模动态调整时间焦点。第二个是面向类别的精炼(CORE)，它通过交叉注意将片段级特征与可学习的类别原型对齐，将软异常类别先验注入表示空间。通过联合利用时间动态和语义结构，明确建模了运动如何演化的'如何'以及它类似于什么语义类别的'什么'。在WVAD基准上的大量实验验证了RefineVAD的有效性，并突出了整合语义背景以引导特征向异常相关模式发展的重要性。


### 论文摘要

Weakly-Supervised Video Anomaly Detection aims to identify anomalous events using only video-level labels, balancing annotation efficiency with practical applicability. However, existing methods often oversimplify the anomaly space by treating all abnormal events as a single category, overlooking the diverse semantic and temporal characteristics intrinsic to real-world anomalies. Inspired by how humans perceive anomalies, by jointly interpreting temporal motion patterns and semantic structures underlying different anomaly types, we propose RefineVAD, a novel framework that mimics this dual-process reasoning. Our framework integrates two core modules. The first, Motion-aware Temporal Attention and Recalibration (MoTAR), estimates motion salience and dynamically adjusts temporal focus via shift-based attention and global Transformer-based modeling. The second, Category-Oriented Refinement (CORE), injects soft anomaly category priors into the representation space by aligning segment-level features with learnable category prototypes through cross-attention. By jointly leveraging temporal dynamics and semantic structure, explicitly models both "how" motion evolves and "what" semantic category it resembles. Extensive experiments on WVAD benchmark validate the effectiveness of RefineVAD and highlight the importance of integrating semantic context to guide feature refinement toward anomaly-relevant patterns.

---

## 231. Modeling group heterogeneity in spatio-temporal data via physics-informed semiparametric regression

**论文链接:** [http://arxiv.org/abs/2511.13203v1](http://arxiv.org/abs/2511.13203v1)

**作者:** Marco F. De Sanctis, Eleonora Arnone, Francesca Ieva, Laura M. Sangalli

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究提出了一种新的方法来建模具有群体结构的空间-时间数据，通过扩展混合效应回归模型，结合空间-时间非参数分量和随机效应，并应用两步估计方法和迭代重加权最小二乘算法进行估计。

### 背景

空间-时间数据通常具有群体结构，需要能够同时捕捉物理动态和群体间差异的建模方法。

### 目的

开发一种能够有效处理具有群体结构的空间-时间数据的新建模方法，并评估其性能。

### 方法

扩展经典混合效应回归模型，引入通过偏微分方程正则化的空间-时间非参数分量，结合随机效应捕获群体结构变异性；采用两步估计程序和功能版本的迭代重加权最小二乘算法进行模型估计。

### 主要发现

通过模拟研究验证了所提出模型的有效性，并将其应用于意大利伦巴第地区二氧化氮浓度数据，成功解释了不同监测站之间的测量异质性。

### 结论

所提出的方法能够有效建模具有群体结构的空间-时间数据，结合了物理动态和群体间差异，为环境监测等领域的应用提供了有力工具。

### 翻译

在这项工作中，我们提出了一种用于建模具有群体结构特征的空间-时间数据的新方法。特别是，我们通过引入一个通过偏微分方程正则化的空间-时间非参数分量来扩展经典的混合效应回归模型，以嵌入底层过程的物理动态，同时随机效应捕获与数据中存在的群体结构相关的潜在变异性。我们提出一个两步程序来估计模型的固定和随机分量，依赖于功能版本的迭代重加权最小二乘算法。我们研究了固定和随机分量的渐近性质，并通过模拟研究评估了所提出模型的性能，将其与文献中最先进的方法进行了比较。最后，将所提出的方法应用于意大利伦巴第地区每小时二氧化氮浓度数据的研究，使用随机效应来解释配备不同传感器技术的监测站之间的测量异质性。


### 论文摘要

In this work we propose a novel approach for modeling spatio-temporal data characterized by group structures. In particular, we extend classical mixed effect regression models by introducing a space-time nonparametric component, regularized through a partial differential equation, to embed the physical dynamics of the underlying process, while random effects capture latent variability associated with the group structure present in the data. We propose a two-step procedure to estimate the fixed and random components of the model, relying on a functional version of the Iterative Reweighted Least Squares algorithm. We investigate the asymptotic properties of both fixed and random components, and we assess the performance of the proposed model through a simulation study, comparing it with state-of-the-art alternatives from the literature. The proposed methodology is finally applied to the study of hourly nitrogen dioxide concentration data in Lombardy (Italy), using random effects to account for measurement heterogeneity across monitoring stations equipped with different sensor technologies.

---

## 232. A Comparative Analysis of Recurrent and Attention Architectures for Isolated Sign Language Recognition

**论文链接:** [http://arxiv.org/abs/2511.13126v1](http://arxiv.org/abs/2511.13126v1)

**作者:** Nigar Alishzade, Gulchin Abdullayeva

**发布时间:** 2025-11-17

**DOI:** 10.1109/PCI66488.2025.11219827

### GPT解析

### 总结

本研究对循环神经网络和基于注意力的神经网络架构在孤立手语识别方面进行了系统性比较分析。通过在阿塞拜疆手语数据集和美国手语数据集上的实验，发现基于注意力的Transformer模型在准确率上优于循环ConvLSTM模型，而ConvLSTM在计算效率方面具有优势。

### 背景

孤立手语识别是一个重要的研究领域，需要比较不同神经网络架构的性能表现。

### 目的

比较和评估两种代表性神经网络模型（ConvLSTM和Vanilla Transformer）在手语识别任务中的性能差异。

### 方法

在阿塞拜疆手语数据集(AzSLD)和单词级别美国手语(WLASL)数据集上实现并评估ConvLSTM和Vanilla Transformer模型，比较它们的Top-1和Top-5准确率。

### 主要发现

1. 基于注意力的Vanilla Transformer在两个数据集上的准确率均优于ConvLSTM；2. Transformer在AzSLD上达到76.8%的Top-1准确率，在WLASL上达到88.3%；3. ConvLSTM计算效率更高但准确率较低；4. Transformer在整体准确率和 signer 独立性方面表现优异；5. ConvLSTM在计算效率和时序建模方面具有优势。

### 结论

研究提供了对不同架构权衡因素的细致分析，为根据应用需求和资源约束选择合适的手语识别系统架构提供了指导。

### 翻译

本研究对循环神经网络和基于注意力的神经网络架构在孤立手语识别方面进行了系统性比较分析。我们在阿塞拜疆手语数据集和单词级别美国手语数据集上实现并评估了两种代表性模型：ConvLSTM和Vanilla Transformer。我们的结果表明，基于注意力的Vanilla Transformer在两个数据集上的Top-1和Top-5准确率均优于循环ConvLSTM，在AzSLD上达到76.8%的Top-1准确率，在WLASL上达到88.3%。虽然ConvLSTM计算效率更高，但在识别准确率上表现较差，特别是在较小数据集上。这些发现突出了每种范式的互补优势：Transformer在整体准确率和 signer 独立性方面表现优异，而ConvLSTM在计算效率和时序建模方面具有优势。该研究提供了对这些权衡因素的细致分析，为根据应用需求和资源约束选择手语识别系统架构提供了指导。


### 论文摘要

This study presents a systematic comparative analysis of recurrent and attention-based neural architectures for isolated sign language recognition. We implement and evaluate two representative models-ConvLSTM and Vanilla Transformer-on the Azerbaijani Sign Language Dataset (AzSLD) and the Word-Level American Sign Language (WLASL) dataset. Our results demonstrate that the attention-based Vanilla Transformer consistently outperforms the recurrent ConvLSTM in both Top-1 and Top-5 accuracy across datasets, achieving up to 76.8% Top-1 accuracy on AzSLD and 88.3% on WLASL. The ConvLSTM, while more computationally efficient, lags in recognition accuracy, particularly on smaller datasets. These findings highlight the complementary strengths of each paradigm: the Transformer excels in overall accuracy and signer independence, whereas the ConvLSTM offers advantages in computational efficiency and temporal modeling. The study provides a nuanced analysis of these trade-offs, offering guidance for architecture selection in sign language recognition systems depending on application requirements and resource constraints.

---

## 233. F.A.C.U.L.: Language-Based Interaction with AI Companions in Gaming

**论文链接:** [http://arxiv.org/abs/2511.13112v1](http://arxiv.org/abs/2511.13112v1)

**作者:** Wenya Wei, Sipeng Yang, Qixian Zhou, Ruochen Liu, Xuelei Zhang, Yifu Yuan, Yan Jiang, Yongle Luo, Hailong Wang, Tianzhou Wang, Peipei Jin, Wangtong Liu, Zhou Zhao, Xiaogang Jin, Elvis S. Liu

**发布时间:** 2025-11-17

**备注:** 14 pages, 11 figures,

### GPT解析

### 总结

本文提出了F.A.C.U.L.系统，这是第一个支持自然语言交互的实时AI伙伴系统，使玩家能够通过自然语言向AI伙伴发出复杂战术指令，从而提升合作游戏的沉浸感和战术协作体验。

### 背景

在合作游戏中，传统AI伙伴通过热键或命令轮盘使用预定义命令（如'攻击'、'防御'或'撤退'）来控制，但这些方法缺乏目标特异性，限制了玩家提供复杂战术指令的能力，并阻碍了游戏体验的沉浸感。

### 目的

开发FPS AI Companion who Understands Language (F.A.C.U.L.)系统，使玩家能够使用自然语言与AI伙伴进行实时交流和协作，克服传统控制方法的局限性。

### 方法

通过整合自然语言处理和基于置信度的框架，F.A.C.U.L.能够有效分解复杂命令并解释玩家意图。系统采用动态实体检索方法实现环境感知，将人类意图与决策制定相结合，支持实时语言交互。

### 主要发现

F.A.C.U.L.系统能够理解和执行复杂命令如'清理二楼'、'躲在树后'或'撤退到河边'，提供实时行为响应和语音反馈，确保无缝的战术协作。

### 结论

通过使用流行的FPS游戏《Arena Breakout: Infinite》作为案例研究，展示了该方法的功效，并讨论了基于真实用户反馈的AI伙伴的优势和局限性。

### 翻译

在合作游戏中，传统的AI伙伴被部署来协助玩家，玩家通过热键或命令轮盘控制它们，发出诸如'攻击'、'防御'或'撤退'等预定义命令。尽管这些方法简单，但缺乏目标特异性的方法限制了玩家提供复杂战术指令的能力，并阻碍了游戏体验的沉浸感。为解决这个问题，我们提出了FPS AI Companion who Understands Language (F.A.C.U.L.)，这是第一个实时AI系统，使玩家能够使用自然语言与AI伙伴进行交流和协作。通过整合自然语言处理与基于置信度的框架，F.A.C.U.L.能够有效分解复杂命令并解释玩家意图。它还采用动态实体检索方法实现环境感知，将人类意图与决策制定相结合。与传统的基于规则的系统不同，我们的方法支持实时语言交互，使玩家能够发出诸如'清理二楼'、'躲在树后'或'撤退到河边'等复杂命令。系统提供实时行为响应和语音反馈，确保无缝的战术协作。使用流行的FPS游戏《Arena Breakout: Infinite》作为案例研究，我们展示了比较结果，证明了我们方法的有效性，并讨论了基于真实用户反馈的AI伙伴的优势和局限性。


### 论文摘要

In cooperative video games, traditional AI companions are deployed to assist players, who control them using hotkeys or command wheels to issue predefined commands such as ``attack'', ``defend'', or ``retreat''. Despite their simplicity, these methods, which lack target specificity, limit players' ability to give complex tactical instructions and hinder immersive gameplay experiences. To address this problem, we propose the FPS AI Companion who Understands Language (F.A.C.U.L.), the first real-time AI system that enables players to communicate and collaborate with AI companions using natural language. By integrating natural language processing with a confidence-based framework, F.A.C.U.L. efficiently decomposes complex commands and interprets player intent. It also employs a dynamic entity retrieval method for environmental awareness, aligning human intentions with decision-making. Unlike traditional rule-based systems, our method supports real-time language interactions, enabling players to issue complex commands such as ``clear the second floor'', ``take cover behind that tree'', or ``retreat to the river''. The system provides real-time behavioral responses and vocal feedback, ensuring seamless tactical collaboration. Using the popular FPS game \textit{Arena Breakout: Infinite} as a case study, we present comparisons demonstrating the efficacy of our approach and discuss the advantages and limitations of AI companions based on real-world user feedback.

---

## 234. BeDiscovER: The Benchmark of Discourse Understanding in the Era of Reasoning Language Models

**论文链接:** [http://arxiv.org/abs/2511.13095v1](http://arxiv.org/abs/2511.13095v1)

**作者:** Chuyuan Li, Giuseppe Carenini

**发布时间:** 2025-11-17

### GPT解析

### 总结

BeDiscovER是一个用于评估现代语言模型话语级知识的最新、全面的评估套件，汇集了5个公开可用的话语任务，包含52个数据集。

### 背景

现代语言模型在话语理解方面需要更全面的评估基准。

### 目的

创建一个全面的评估套件来评估现代语言模型在话语理解方面的能力，包括不同级别的话语知识。

### 方法

BeDiscovER汇集了话语词汇、(多)句子和文档级别的5个话语任务，包括话语解析、时间关系提取和话语词消歧等。作者评估了Qwen3系列、DeepSeek-R1和GPT-5-mini等语言模型。

### 主要发现

最先进的模型在时间推理的算术方面表现出强大的性能，但在完整的文档推理以及一些细微的语义和话语现象（如修辞关系识别）方面存在困难。

### 结论

现代语言模型在话语理解方面仍有改进空间，特别是在处理复杂话语现象和完整文档推理方面。

### 翻译

我们介绍了BeDiscovER（推理语言模型时代的话语理解基准），这是一个用于评估现代大语言模型话语级知识的最新、全面的评估套件。BeDiscovER汇集了5个公开可用的话语任务，涵盖话语词汇、(多)句子和文档级别，总共包含52个独立数据集。它既包括已被广泛研究的任务，如话语解析和时间关系提取，也包括一些新的挑战，如话语词消歧（例如'just'），还汇集了一个关于话语关系解析和树库的共享任务，用于多语言和多框架话语关系分类。我们在BeDiscovER上评估了开源大语言模型：Qwen3系列、DeepSeek-R1以及前沿模型如GPT-5-mini，发现最先进的模型在时间推理的算术方面表现出强大的性能，但在完整的文档推理和一些细微的语义和话语现象（如修辞关系识别）方面存在困难。


### 论文摘要

We introduce BeDiscovER (Benchmark of Discourse Understanding in the Era of Reasoning Language Models), an up-to-date, comprehensive suite for evaluating the discourse-level knowledge of modern LLMs. BeDiscovER compiles 5 publicly available discourse tasks across discourse lexicon, (multi-)sentential, and documental levels, with in total 52 individual datasets. It covers both extensively studied tasks such as discourse parsing and temporal relation extraction, as well as some novel challenges such as discourse particle disambiguation (e.g., ``just''), and also aggregates a shared task on Discourse Relation Parsing and Treebanking for multilingual and multi-framework discourse relation classification. We evaluate open-source LLMs: Qwen3 series, DeepSeek-R1, and frontier model such as GPT-5-mini on BeDiscovER, and find that state-of-the-art models exhibit strong performance in arithmetic aspect of temporal reasoning, but they struggle with full document reasoning and some subtle semantic and discourse phenomena, such as rhetorical relation recognition.

---

## 235. RobustGait: Robustness Analysis for Appearance Based Gait Recognition

**论文链接:** [http://arxiv.org/abs/2511.13065v1](http://arxiv.org/abs/2511.13065v1)

**作者:** Reeshoon Sayera, Akash Kumar, Sirshapan Mitra, Prudvi Kamtam, Yogesh S Rawat

**发布时间:** 2025-11-17

**备注:** IEEE WACV'26 Main Conference

### GPT解析

### 总结

论文提出了RobustGait框架，用于系统评估基于外观的步态识别系统在真实世界干扰下的鲁棒性，填补了这一领域系统性评估的空白。

### 背景

基于外观的步态识别系统在受控数据集上已取得良好性能，但对其在真实世界干扰和轮廓变化下的鲁棒性评估仍然缺乏。

### 目的

开发一个细粒度的鲁棒性评估框架，全面评估基于外观的步态识别系统在不同干扰条件下的表现。

### 方法

RobustGait评估框架涵盖四个维度：干扰类型（数字、环境、时间、遮挡）、轮廓提取方法（分割和解析网络）、步态识别模型的架构能力以及各种部署场景。该基准测试在CASIA-B、CCPG和SUSTech1K数据集上引入了15种干扰类型和5个严重级别，并在MEVID上进行野外验证，评估了六种最先进的步态系统。

### 主要发现

1. 在RGB级别应用噪声能更好地反映真实世界的退化情况，揭示了失真如何通过轮廓传播到下游步态识别系统。2. 步态识别准确度对轮廓提取器的偏差高度敏感，揭示了一个被忽视的基准偏差来源。3. 鲁棒性同时取决于干扰类型和架构设计。4. 探索了增强鲁棒性的策略，表明噪声感知训练和知识蒸馏可以提高性能，使系统更接近部署就绪状态。

### 结论

RobustGait框架为步态识别系统的鲁棒性评估提供了全面视角，有助于开发更健壮的步态识别系统，并指导未来研究方向。

### 翻译

基于外观的步态识别在受控数据集上已取得强劲性能，然而对其在真实世界干扰和轮廓变化下的鲁棒性的系统评估仍然缺乏。我们提出了RobustGait，一个用于细粒度评估基于外观的步态识别系统鲁棒性的框架。RobustGait评估涵盖四个维度：干扰类型（数字、环境、时间、遮挡）、轮廓提取方法（分割和解析网络）、步态识别模型的架构能力以及各种部署场景。该基准测试在CASIA-B、CCPG和SUSTech1K上引入了15种干扰类型和5个严重级别，在MEVID上进行野外验证，并评估了六种最先进的步态系统。我们获得了几个有趣的发现。首先，在RGB级别应用噪声能更好地反映真实世界的退化，揭示了失真如何通过轮廓提取传播到下游步态识别系统。其次，步态识别准确度对轮廓提取器偏差高度敏感，揭示了一个被忽视的基准偏差来源。第三，鲁棒性同时取决于干扰类型和架构设计。最后，我们探索了增强鲁棒性的策略，表明噪声感知训练和知识蒸馏可以提高性能，使系统更接近部署就绪状态。


### 论文摘要

Appearance-based gait recognition have achieved strong performance on controlled datasets, yet systematic evaluation of its robustness to real-world corruptions and silhouette variability remains lacking. We present RobustGait, a framework for fine-grained robustness evaluation of appearance-based gait recognition systems. RobustGait evaluation spans four dimensions: the type of perturbation (digital, environmental, temporal, occlusion), the silhouette extraction method (segmentation and parsing networks), the architectural capacities of gait recognition models, and various deployment scenarios. The benchmark introduces 15 corruption types at 5 severity levels across CASIA-B, CCPG, and SUSTech1K, with in-the-wild validation on MEVID, and evaluates six state-of-the-art gait systems. We came across several exciting insights. First, applying noise at the RGB level better reflects real-world degradation, and reveal how distortions propagate through silhouette extraction to the downstream gait recognition systems. Second, gait accuracy is highly sensitive to silhouette extractor biases, revealing an overlooked source of benchmark bias. Third, robustness is dependent on both the type of perturbation and the architectural design. Finally, we explore robustness-enhancing strategies, showing that noise-aware training and knowledge distillation improve performance and move toward deployment-ready systems.

---

## 236. A Fractional Calculus Framework for Open Quantum Dynamics: From Liouville to Lindblad to Memory Kernels

**论文链接:** [http://arxiv.org/abs/2511.13038v1](http://arxiv.org/abs/2511.13038v1)

**作者:** Bo Peng, Yu Zhang

**发布时间:** 2025-11-17

### GPT解析

### 总结

该研究建立了将分数阶量子主方程嵌入开放系统动力学的统一层级结构，通过分数微积分和Bochner-Phillips从属关系，将分数演化表示为Lindblad半群的加权平均，从而连接了幺正、马尔可夫和结构化非马尔可夫区域。

### 背景

开放量子系统表现出从纯幺正演化到不可逆耗散松弛的动力学行为。GKSL方程唯一表征了完全正且迹保持的马尔可夫动力学，但许多物理系统显示出如代数松弛和相干回流等超出半群演化范围的非马尔可夫特征。

### 目的

建立一个统一的层级结构，将分数阶量子主方程嵌入到开放系统动力学的更广阔图景中。

### 方法

使用分数微积分描述长时间记忆行为，通过分数时间导数引入幂律时间核；通过Bochner-Phillips从属关系，将分数演化表示为Lindblad半群上的加权平均，权重为幂律等待时间分布。

### 主要发现

分数主方程是记忆核模型的一个结构化子类，在单位分数阶时简化为GKSL形式；这种构造确保了物理一致性，解释了长时间衰减的代数起源，并连接了幺正、马尔可夫和结构化非马尔可夫区域。

### 结论

所得框架将分数微积分定位为具有内在记忆的量子动力学的严谨统一语言，为理论分析和量子模拟提供了新方向。

### 翻译

开放量子系统表现出从纯幺正演化到不可逆耗散松弛的动力学范围。Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) 方程唯一表征了完全正且迹保持的马尔可夫动力学，但许多物理系统显示出如代数松弛和相干回流等超出半群演化范围的非马尔可夫特征。分数微积分通过引入幂律时间核的分数时间导数，为描述这种长时间记忆行为提供了自然框架。本文建立了一个统一的层级结构，将分数阶量子主方程嵌入开放系统动力学的更广阔图景中。分数主方程形成记忆核模型的一个结构化子类，在单位分数阶时简化为GKSL形式。通过Bochner-Phillips从属关系，分数演化表示为Lindblad半群上的加权平均，权重为幂律等待时间分布。这种构造确保了物理一致性，解释了长时间衰减的代数起源，并连接了幺正、马尔可夫和结构化非马尔可夫区域。所得框架将分数微积分定位为具有内在记忆的量子动力学的严谨统一语言，为理论分析和量子模拟提供了新方向。


### 论文摘要

Open quantum systems exhibit dynamics ranging from purely unitary evolution to irreversible dissipative relaxation. The Gorini--Kossakowski--Sudarshan--Lindblad (GKSL) equation uniquely characterizes Markovian dynamics that are completely positive and trace-preserving (CPTP), yet many physical systems display non-Markovian features such as algebraic relaxation and coherence backflow beyond the reach of semigroup evolution. Fractional calculus provides a natural framework for describing such long-memory behavior through power-law temporal kernels introduced by fractional time derivatives. Here we establish a unified hierarchy that embeds fractional quantum master equations within the broader landscape of open system dynamics. The fractional master equation forms a structured subclass of memory-kernel models, reducing to the GKSL form at unit fractional order. Through Bochner--Phillips subordination, fractional evolution is expressed as an average over Lindblad semigroups weighted by a power-law waiting-time distribution. This construction ensures physical consistency, explains the algebraic origin of long-time decay, and bridges unitary, Markovian, and structured non-Markovian regimes. The resulting framework positions fractional calculus as a rigorous and unifying language for quantum dynamics with intrinsic memory, enabling new directions for theoretical analysis and quantum simulation.

---

